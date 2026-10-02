#include "runtime/continuation.h"

#include "core/host_kv_arena.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ninfer::targets::gemma4::detail {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'N', 'I', 'G', '4', 'C', 'N', 'T', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kMaximumAnchors = 64;
constexpr std::uint32_t kMaximumIdentityBytes = 4096;

std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes) noexcept {
    std::uint64_t value = 14695981039346656037ULL;
    for (const std::uint8_t byte : bytes) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}

class Writer {
public:
    explicit Writer(std::vector<std::uint8_t>& out) : out_(out) {}

    void bytes(const void* data, std::size_t count) {
        if (count == 0) { return; }
        const std::size_t offset = out_.size();
        if (count > std::numeric_limits<std::size_t>::max() - offset) {
            throw std::overflow_error("Gemma continuation byte size overflow");
        }
        out_.resize(offset + count);
        std::memcpy(out_.data() + offset, data, count);
    }

    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(value));
    }

    void string(std::string_view value) {
        if (value.size() > kMaximumIdentityBytes) {
            throw std::invalid_argument("Gemma continuation identity is too long");
        }
        pod<std::uint32_t>(static_cast<std::uint32_t>(value.size()));
        bytes(value.data(), value.size());
    }

private:
    std::vector<std::uint8_t>& out_;
};

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* destination, std::size_t count) {
        if (count > remaining()) { throw std::invalid_argument("Gemma continuation is truncated"); }
        std::memcpy(destination, data_.data() + cursor_, count);
        cursor_ += count;
    }

    template <class T>
    [[nodiscard]] T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        bytes(&value, sizeof(value));
        return value;
    }

    [[nodiscard]] std::string string() {
        const std::uint32_t count = pod<std::uint32_t>();
        if (count > kMaximumIdentityBytes) {
            throw std::invalid_argument("Gemma continuation identity is too long");
        }
        std::string value(count, '\0');
        bytes(value.data(), value.size());
        return value;
    }

    [[nodiscard]] std::span<const std::uint8_t> span(std::size_t count) {
        if (count > remaining()) { throw std::invalid_argument("Gemma continuation is truncated"); }
        const auto value = data_.subspan(cursor_, count);
        cursor_ += count;
        return value;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

const HeterogeneousKVHostGroupImage& require_group(const HeterogeneousKVHostImage& image,
                                                   std::uint32_t id) {
    const auto found = std::find_if(image.groups.begin(), image.groups.end(),
                                    [id](const auto& group) { return group.group_id == id; });
    if (found == image.groups.end()) {
        throw std::invalid_argument("Gemma continuation is missing a KV group");
    }
    return *found;
}

HeterogeneousKVHostGroupImage& require_group(HeterogeneousKVHostImage& image,
                                             std::uint32_t id) {
    const auto found = std::find_if(image.groups.begin(), image.groups.end(),
                                    [id](const auto& group) { return group.group_id == id; });
    if (found == image.groups.end()) {
        throw std::invalid_argument("Gemma continuation is missing a KV group");
    }
    return *found;
}

const KvGroupSpec& require_spec(std::span<const KvGroupSpec> groups, std::uint32_t id) {
    const auto found = std::find_if(groups.begin(), groups.end(),
                                    [id](const auto& group) { return group.group_id == id; });
    if (found == groups.end()) {
        throw std::invalid_argument("Gemma continuation descriptor is missing a KV group");
    }
    return *found;
}

std::uint32_t pages_for(std::uint32_t frontier, std::uint32_t page_tokens) {
    return frontier == 0 ? 0U : 1U + (frontier - 1U) / page_tokens;
}

void validate_group_image(const HeterogeneousKVHostGroupImage& image, const KvGroupSpec& spec,
                          bool sliding_only) {
    if (image.group_id != spec.group_id || image.frontier == 0 ||
        image.frontier > spec.maximum_context) {
        throw std::invalid_argument("Gemma continuation KV group identity/frontier is invalid");
    }
    const std::uint32_t page_count = pages_for(image.frontier, spec.geometry.page_tokens);
    const std::uint32_t visible_begin =
        spec.retention == KvGroupRetention::SlidingWindow && image.frontier > spec.window_tokens
            ? image.frontier - spec.window_tokens
            : 0U;
    const std::uint32_t first = visible_begin / spec.geometry.page_tokens;
    const std::uint32_t retained = page_count - first;
    if (sliding_only && spec.retention != KvGroupRetention::SlidingWindow) {
        throw std::invalid_argument("Gemma continuation anchor does not name sliding KV");
    }
    if (image.logical_blocks.size() != retained) {
        throw std::invalid_argument("Gemma continuation KV group page count is invalid");
    }
    for (std::uint32_t page = 0; page < retained; ++page) {
        if (image.logical_blocks[page] != first + page) {
            throw std::invalid_argument("Gemma continuation KV blocks are not contiguous");
        }
    }
    const HostKVPageLayout layout = plan_host_kv_page_layout(spec.geometry);
    if (retained > std::numeric_limits<std::size_t>::max() / layout.page_stride ||
        image.payload.size() != static_cast<std::size_t>(retained) * layout.page_stride) {
        throw std::invalid_argument("Gemma continuation KV payload size is invalid");
    }
}

void write_group(Writer& writer, const HeterogeneousKVHostGroupImage& group) {
    writer.pod(group.group_id);
    writer.pod(group.frontier);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(group.logical_blocks.size()));
    writer.pod<std::uint64_t>(group.payload.size());
    writer.bytes(group.logical_blocks.data(),
                 group.logical_blocks.size() * sizeof(std::uint32_t));
    writer.bytes(group.payload.data(), group.payload.size());
}

HeterogeneousKVHostGroupImage read_group(Reader& reader, const KvGroupSpec& spec,
                                         bool sliding_only) {
    HeterogeneousKVHostGroupImage group;
    group.group_id = reader.pod<std::uint32_t>();
    group.frontier = reader.pod<std::uint32_t>();
    const std::uint32_t pages = reader.pod<std::uint32_t>();
    const std::uint64_t payload_bytes = reader.pod<std::uint64_t>();
    if (group.group_id != spec.group_id || group.frontier == 0 ||
        group.frontier > spec.maximum_context ||
        pages > pages_for(spec.maximum_context, spec.geometry.page_tokens) ||
        payload_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Gemma continuation KV extent is out of range");
    }
    const std::uint32_t visible_begin =
        spec.retention == KvGroupRetention::SlidingWindow &&
                group.frontier > spec.window_tokens
            ? group.frontier - spec.window_tokens
            : 0U;
    const std::uint32_t retained =
        pages_for(group.frontier, spec.geometry.page_tokens) -
        visible_begin / spec.geometry.page_tokens;
    const HostKVPageLayout layout = plan_host_kv_page_layout(spec.geometry);
    if (pages != retained || retained > std::numeric_limits<std::size_t>::max() /
                                           layout.page_stride ||
        payload_bytes != static_cast<std::uint64_t>(retained) * layout.page_stride) {
        throw std::invalid_argument("Gemma continuation KV extent is inconsistent");
    }
    group.logical_blocks.resize(pages);
    reader.bytes(group.logical_blocks.data(),
                 group.logical_blocks.size() * sizeof(std::uint32_t));
    group.payload.resize(static_cast<std::size_t>(payload_bytes));
    reader.bytes(group.payload.data(), group.payload.size());
    validate_group_image(group, spec, sliding_only);
    return group;
}

void validate_state(const ContinuationState& state, std::span<const KvGroupSpec> specs,
                    std::int32_t token_domain, std::uint32_t sliding_group_id,
                    std::uint32_t global_group_id) {
    if (token_domain <= 0 || state.execution_frontier == 0 ||
        state.execution_frontier > state.ledger.size() ||
        state.ledger.size() - state.execution_frontier > 1) {
        throw std::invalid_argument("Gemma continuation ledger/frontier is invalid");
    }
    for (const std::int32_t token : state.ledger) {
        if (token < 0 || token >= token_domain) {
            throw std::invalid_argument("Gemma continuation token is out of domain");
        }
    }
    if (state.endpoint.groups.size() != specs.size()) {
        throw std::invalid_argument("Gemma continuation endpoint group count changed");
    }
    for (const KvGroupSpec& spec : specs) {
        const auto& group = require_group(state.endpoint, spec.group_id);
        validate_group_image(group, spec, false);
        if (group.frontier != state.execution_frontier) {
            throw std::invalid_argument("Gemma continuation endpoint frontiers diverged");
        }
    }
    const KvGroupSpec& sliding = require_spec(specs, sliding_group_id);
    (void)require_spec(specs, global_group_id);
    std::uint32_t previous = 0;
    for (const ContinuationAnchorImage& anchor : state.anchors) {
        if (anchor.frontier <= previous || anchor.frontier >= state.execution_frontier ||
            anchor.sliding.frontier != anchor.frontier) {
            throw std::invalid_argument("Gemma continuation anchor directory is invalid");
        }
        validate_group_image(anchor.sliding, sliding, true);
        previous = anchor.frontier;
    }
}

} // namespace

ContinuationState make_continuation_state(
    std::span<const std::int32_t> ledger, HeterogeneousKVHostImage endpoint,
    std::vector<HeterogeneousKVHostImage> anchors, std::uint32_t sliding_group_id,
    std::uint32_t global_group_id) {
    const auto& endpoint_sliding = require_group(endpoint, sliding_group_id);
    const auto& endpoint_global = require_group(endpoint, global_group_id);
    if (endpoint_sliding.frontier == 0 ||
        endpoint_sliding.frontier != endpoint_global.frontier ||
        endpoint_sliding.frontier > ledger.size() ||
        ledger.size() - endpoint_sliding.frontier > 1 || anchors.size() > kMaximumAnchors) {
        throw std::invalid_argument("Gemma continuation capture frontiers are invalid");
    }
    ContinuationState state{
        .ledger = std::vector<std::int32_t>(ledger.begin(), ledger.end()),
        .execution_frontier = endpoint_sliding.frontier,
        .endpoint = std::move(endpoint),
    };
    state.anchors.reserve(anchors.size());
    for (HeterogeneousKVHostImage& anchor : anchors) {
        auto& sliding = require_group(anchor, sliding_group_id);
        const auto& global = require_group(anchor, global_group_id);
        if (sliding.frontier != global.frontier) {
            throw std::invalid_argument("Gemma continuation anchor frontiers diverged");
        }
        const std::uint32_t frontier = sliding.frontier;
        state.anchors.push_back({.frontier = frontier, .sliding = std::move(sliding)});
    }
    std::sort(state.anchors.begin(), state.anchors.end(),
              [](const auto& left, const auto& right) { return left.frontier < right.frontier; });
    return state;
}

std::vector<std::uint8_t> encode_continuation(
    const ContinuationState& state, std::string_view model_binding,
    std::string_view artifact_fingerprint, std::span<const KvGroupSpec> groups,
    std::uint32_t sliding_group_id, std::uint32_t global_group_id) {
    if (model_binding.empty() || artifact_fingerprint.empty()) {
        throw std::invalid_argument("Gemma continuation identity is empty");
    }
    validate_state(state, groups, std::numeric_limits<std::int32_t>::max(), sliding_group_id,
                   global_group_id);
    const std::vector<std::byte> descriptors = encode_kv_group_descriptors(groups);
    std::vector<std::uint8_t> body;
    Writer writer(body);
    writer.string(model_binding);
    writer.string(artifact_fingerprint);
    writer.pod<std::uint64_t>(descriptors.size());
    writer.bytes(descriptors.data(), descriptors.size());
    writer.pod<std::uint64_t>(state.ledger.size());
    writer.bytes(state.ledger.data(), state.ledger.size() * sizeof(std::int32_t));
    writer.pod(state.execution_frontier);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(state.endpoint.groups.size()));
    for (const auto& group : state.endpoint.groups) write_group(writer, group);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(state.anchors.size()));
    for (const ContinuationAnchorImage& anchor : state.anchors) {
        writer.pod(anchor.frontier);
        write_group(writer, anchor.sliding);
    }

    std::vector<std::uint8_t> encoded;
    Writer envelope(encoded);
    envelope.bytes(kMagic.data(), kMagic.size());
    envelope.pod(kVersion);
    envelope.pod<std::uint64_t>(body.size());
    envelope.pod(fnv1a64(body));
    envelope.bytes(body.data(), body.size());
    return encoded;
}

ContinuationState decode_continuation(
    std::span<const std::uint8_t> encoded, std::string_view expected_model_binding,
    std::string_view expected_artifact_fingerprint, std::span<const KvGroupSpec> expected_groups,
    std::int32_t token_domain, std::uint32_t sliding_group_id,
    std::uint32_t global_group_id) {
    if (expected_groups.empty()) {
        throw std::invalid_argument("Gemma continuation expected KV layout is empty");
    }
    Reader envelope(encoded);
    std::array<std::uint8_t, kMagic.size()> magic{};
    envelope.bytes(magic.data(), magic.size());
    if (magic != kMagic) { throw std::invalid_argument("file is not a Gemma continuation"); }
    if (envelope.pod<std::uint32_t>() != kVersion) {
        throw std::invalid_argument("Gemma continuation version is unsupported");
    }
    const std::uint64_t body_bytes = envelope.pod<std::uint64_t>();
    const std::uint64_t checksum = envelope.pod<std::uint64_t>();
    if (body_bytes != envelope.remaining()) {
        throw std::invalid_argument("Gemma continuation byte count is invalid");
    }
    const auto body_span = envelope.span(static_cast<std::size_t>(body_bytes));
    if (fnv1a64(body_span) != checksum) {
        throw std::invalid_argument("Gemma continuation checksum is invalid");
    }
    Reader reader(body_span);
    if (reader.string() != expected_model_binding) {
        throw std::invalid_argument("Gemma continuation was saved for a different model");
    }
    if (reader.string() != expected_artifact_fingerprint) {
        throw std::invalid_argument("Gemma continuation artifact fingerprint changed");
    }
    const std::uint64_t descriptor_bytes = reader.pod<std::uint64_t>();
    const std::vector<std::byte> expected_descriptors =
        encode_kv_group_descriptors(expected_groups);
    if (descriptor_bytes != expected_descriptors.size()) {
        throw std::invalid_argument("Gemma continuation KV layout changed");
    }
    const auto descriptor_span = reader.span(static_cast<std::size_t>(descriptor_bytes));
    if (std::memcmp(descriptor_span.data(), expected_descriptors.data(),
                    descriptor_span.size()) != 0) {
        throw std::invalid_argument("Gemma continuation KV layout changed");
    }

    const std::uint64_t token_count = reader.pod<std::uint64_t>();
    if (token_count == 0 || token_count > expected_groups.front().maximum_context ||
        token_count > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
        throw std::invalid_argument("Gemma continuation ledger extent is invalid");
    }
    ContinuationState state;
    state.ledger.resize(static_cast<std::size_t>(token_count));
    reader.bytes(state.ledger.data(), state.ledger.size() * sizeof(std::int32_t));
    state.execution_frontier = reader.pod<std::uint32_t>();
    const std::uint32_t group_count = reader.pod<std::uint32_t>();
    if (group_count != expected_groups.size()) {
        throw std::invalid_argument("Gemma continuation endpoint group count changed");
    }
    state.endpoint.groups.reserve(group_count);
    for (std::uint32_t index = 0; index < group_count; ++index) {
        state.endpoint.groups.push_back(read_group(reader, expected_groups[index], false));
    }
    const std::uint32_t anchor_count = reader.pod<std::uint32_t>();
    if (anchor_count > kMaximumAnchors) {
        throw std::invalid_argument("Gemma continuation anchor count is out of range");
    }
    const KvGroupSpec& sliding = require_spec(expected_groups, sliding_group_id);
    state.anchors.reserve(anchor_count);
    for (std::uint32_t index = 0; index < anchor_count; ++index) {
        ContinuationAnchorImage anchor;
        anchor.frontier = reader.pod<std::uint32_t>();
        anchor.sliding = read_group(reader, sliding, true);
        state.anchors.push_back(std::move(anchor));
    }
    if (reader.remaining() != 0) {
        throw std::invalid_argument("Gemma continuation has trailing bytes");
    }
    validate_state(state, expected_groups, token_domain, sliding_group_id, global_group_id);
    return state;
}

ContinuationRestoreSelection select_continuation_restore(
    const ContinuationState& state, std::span<const std::int32_t> incoming,
    std::span<const KvGroupSpec> groups, std::uint32_t sliding_group_id,
    std::uint32_t global_group_id) {
    const auto prefix_matches = [&](std::uint32_t frontier) {
        return frontier <= incoming.size() && frontier <= state.ledger.size() &&
               std::equal(state.ledger.begin(), state.ledger.begin() + frontier,
                          incoming.begin());
    };
    const ContinuationAnchorImage* selected_anchor = nullptr;
    std::uint32_t selected_frontier = 0;
    for (const ContinuationAnchorImage& anchor : state.anchors) {
        if (prefix_matches(anchor.frontier) && anchor.frontier > selected_frontier) {
            selected_anchor = &anchor;
            selected_frontier = anchor.frontier;
        }
    }
    const bool endpoint = prefix_matches(state.execution_frontier) &&
                          state.execution_frontier > selected_frontier;
    if (endpoint) selected_frontier = state.execution_frontier;
    if (selected_frontier == 0) { return {}; }

    const auto& endpoint_sliding = require_group(state.endpoint, sliding_group_id);
    const auto& endpoint_global = require_group(state.endpoint, global_group_id);
    const KvGroupSpec& global_spec = require_spec(groups, global_group_id);
    HeterogeneousKVHostGroupImage sliding =
        endpoint ? endpoint_sliding : selected_anchor->sliding;
    HeterogeneousKVHostGroupImage global{
        .group_id = endpoint_global.group_id,
        .frontier = selected_frontier,
    };
    const std::uint32_t page_count =
        pages_for(selected_frontier, global_spec.geometry.page_tokens);
    if (endpoint_global.logical_blocks.empty() || page_count > endpoint_global.logical_blocks.size()) {
        throw std::logic_error("Gemma continuation global prefix geometry is inconsistent");
    }
    const std::size_t page_stride =
        endpoint_global.payload.size() / endpoint_global.logical_blocks.size();
    global.logical_blocks.assign(endpoint_global.logical_blocks.begin(),
                                 endpoint_global.logical_blocks.begin() + page_count);
    global.payload.assign(endpoint_global.payload.begin(),
                          endpoint_global.payload.begin() +
                              static_cast<std::ptrdiff_t>(page_count * page_stride));

    ContinuationRestoreSelection selection{
        .frontier = selected_frontier,
        .endpoint = endpoint,
    };
    selection.kv.groups.reserve(state.endpoint.groups.size());
    for (const auto& group : state.endpoint.groups) {
        if (group.group_id == sliding_group_id) {
            selection.kv.groups.push_back(std::move(sliding));
        } else if (group.group_id == global_group_id) {
            selection.kv.groups.push_back(std::move(global));
        } else {
            throw std::logic_error("Gemma continuation endpoint contains an unknown KV group");
        }
    }
    return selection;
}

} // namespace ninfer::targets::gemma4::detail
