#include "core/host_kv_arena.h"
#include "targets/gemma4/impl/runtime/continuation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

using ninfer::HeterogeneousKVHostGroupImage;
using ninfer::HeterogeneousKVHostImage;
using ninfer::KVPageGeometry;
using ninfer::KvGroupRetention;
using ninfer::KvGroupSpec;
using ninfer::targets::gemma4::detail::ContinuationState;

[[noreturn]] void fail(std::string_view message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, std::string_view message) {
    if (!condition) fail(message);
}

template <class Function>
void require_invalid(Function&& function, std::string_view message) {
    try {
        function();
    } catch (const std::invalid_argument&) { return; }
    fail(message);
}

KvGroupSpec group(std::uint32_t id, KvGroupRetention retention) {
    return {
        .group_id = id,
        .retention = retention,
        .layer_count = id == 0 ? 50U : 10U,
        .maximum_context = 1024,
        .window_tokens = retention == KvGroupRetention::SlidingWindow ? 64U : 0U,
        .maximum_transaction_tokens = 8,
        .table_rows = 1,
        .physical_page_groups = 32,
        .geometry = KVPageGeometry{
            .page_tokens = 64,
            .planes = {{ninfer::DType::U8, 4, 2, 256}},
        },
    };
}

HeterogeneousKVHostGroupImage image_group(const KvGroupSpec& spec, std::uint32_t frontier,
                                          std::uint8_t seed) {
    const std::uint32_t pages = 1U + (frontier - 1U) / spec.geometry.page_tokens;
    const std::uint32_t visible =
        spec.retention == KvGroupRetention::SlidingWindow && frontier > spec.window_tokens
            ? frontier - spec.window_tokens
            : 0U;
    const std::uint32_t first = visible / spec.geometry.page_tokens;
    HeterogeneousKVHostGroupImage out{.group_id = spec.group_id, .frontier = frontier};
    for (std::uint32_t page = first; page < pages; ++page) {
        out.logical_blocks.push_back(page);
    }
    const auto layout = ninfer::plan_host_kv_page_layout(spec.geometry);
    out.payload.resize(out.logical_blocks.size() * layout.page_stride);
    for (std::size_t index = 0; index < out.payload.size(); ++index) {
        out.payload[index] = static_cast<std::byte>(seed + static_cast<std::uint8_t>(index));
    }
    return out;
}

HeterogeneousKVHostImage image_at(std::span<const KvGroupSpec> specs, std::uint32_t frontier,
                                  std::uint8_t seed) {
    HeterogeneousKVHostImage image;
    image.groups.push_back(image_group(specs[0], frontier, seed));
    image.groups.push_back(image_group(specs[1], frontier, seed + 31U));
    return image;
}

} // namespace

int main() {
    using namespace ninfer::targets::gemma4::detail;
    constexpr std::string_view binding = "gemma4_31b_it\ngemma4-31b-it\ngroupwise-int";
    constexpr std::string_view fingerprint = "sha256:registered-target";
    const std::array specs{group(0, KvGroupRetention::SlidingWindow),
                           group(1, KvGroupRetention::FullHistory)};
    std::vector<std::int32_t> ledger(131);
    for (std::size_t index = 0; index < ledger.size(); ++index) {
        ledger[index] = static_cast<std::int32_t>(1000 + index);
    }

    std::vector<HeterogeneousKVHostImage> anchors;
    anchors.push_back(image_at(specs, 65, 17));
    ContinuationState state =
        make_continuation_state(ledger, image_at(specs, 130, 71), std::move(anchors), 0, 1);
    require(state.anchors.size() == 1 && state.anchors[0].sliding.group_id == 0,
            "capture retained global KV in an anchor");

    const std::vector<std::uint8_t> encoded =
        encode_continuation(state, binding, fingerprint, specs, 0, 1);
    const ContinuationState decoded =
        decode_continuation(encoded, binding, fingerprint, specs, 4096, 0, 1);
    require(decoded.ledger == ledger && decoded.execution_frontier == 130 &&
                decoded.endpoint == state.endpoint && decoded.anchors.size() == 1 &&
                decoded.anchors[0].sliding == state.anchors[0].sliding,
            "continuation round trip changed state");

    const auto endpoint = select_continuation_restore(decoded, ledger, specs, 0, 1);
    require(endpoint.endpoint && endpoint.frontier == 130 &&
                endpoint.kv.groups[0] == state.endpoint.groups[0] &&
                endpoint.kv.groups[1] == state.endpoint.groups[1],
            "exact ledger did not select the endpoint");

    std::vector<std::int32_t> edited = ledger;
    edited[100] = 2000;
    const auto anchor = select_continuation_restore(decoded, edited, specs, 0, 1);
    require(!anchor.endpoint && anchor.frontier == 65 &&
                anchor.kv.groups[0] == state.anchors[0].sliding &&
                anchor.kv.groups[1].frontier == 65 &&
                anchor.kv.groups[1].logical_blocks == std::vector<std::uint32_t>({0, 1}),
            "edited ledger did not restore the deepest compatible anchor");
    edited[0] = 2001;
    require(select_continuation_restore(decoded, edited, specs, 0, 1).frontier == 0,
            "incompatible ledger selected a stale anchor");

    const std::size_t naive_anchor_bytes = state.endpoint.payload_bytes() +
                                           image_at(specs, 65, 17).payload_bytes();
    const std::size_t deduplicated_payload =
        state.endpoint.payload_bytes() + state.anchors[0].sliding.payload.size();
    require(deduplicated_payload < naive_anchor_bytes && encoded.size() > deduplicated_payload,
            "continuation did not deduplicate anchor global history");

    require_invalid(
        [&] { (void)decode_continuation(encoded, "wrong", fingerprint, specs, 4096, 0, 1); },
        "continuation accepted a different model binding");
    require_invalid(
        [&] { (void)decode_continuation(encoded, binding, "wrong", specs, 4096, 0, 1); },
        "continuation accepted a different artifact fingerprint");
    auto changed_specs = specs;
    changed_specs[1].geometry.planes[0].head_extent = 3;
    require_invalid(
        [&] {
            (void)decode_continuation(encoded, binding, fingerprint, changed_specs, 4096, 0, 1);
        },
        "continuation accepted a different KV layout");
    std::vector<std::uint8_t> corrupt = encoded;
    corrupt.back() ^= 0x80U;
    require_invalid(
        [&] { (void)decode_continuation(corrupt, binding, fingerprint, specs, 4096, 0, 1); },
        "continuation accepted payload corruption");
    std::vector<std::uint8_t> wrong_version = encoded;
    wrong_version[8] = 2;
    require_invalid(
        [&] {
            (void)decode_continuation(wrong_version, binding, fingerprint, specs, 4096, 0, 1);
        },
        "continuation accepted an unsupported version");
    require_invalid(
        [&] {
            (void)decode_continuation(
                std::span<const std::uint8_t>(encoded.data(), encoded.size() - 1), binding,
                fingerprint, specs, 4096, 0, 1);
        },
        "continuation accepted truncation");

    std::cout << "PASS\n";
    return 0;
}
