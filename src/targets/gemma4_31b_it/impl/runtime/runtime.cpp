#include "targets/gemma4_31b_it/impl/runtime/variant.h"

#include <ninfer/ops/softmax_attention.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/heterogeneous_kv_cache.h"
#include "runtime/engine/kv_capacity.h"
#include "targets/gemma4/impl/runtime/continuation.h"
#include "targets/gemma4_31b_it/impl/runtime/kv_groups.h"
#include "targets/gemma4_31b_it/impl/runtime/runtime_chunk.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

using GemmaVariant = ::ninfer::targets::gemma4_31b_it::detail::Variant;

struct GemmaWorkspacePlan {
    std::size_t activation = 0;
    std::size_t attention  = 0;
    std::size_t mtp_state  = 0;
    std::size_t capacity   = 0;
};

template <>
struct SequencePlanImpl<GemmaVariant> {
    GemmaVariant::WeightsProfile weights_profile{};
    std::uint32_t capacity        = 0;
    std::uint32_t kv_capacity     = 0;
    std::uint32_t main_page_groups = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t prefill_chunk   = 64;
    int device                    = 0;
    ContextCacheOptions context_cache;
    SpeculativeOptions speculative;
    bool use_cuda_graph = false;
    std::array<KvGroupSpec, 2> groups;
    HeterogeneousKVCacheLayout cache_layout;
    GemmaWorkspacePlan workspace;
    std::size_t sequence_capacity_bytes = 0;
    std::size_t device_reservation_bytes = 0;
};

template <>
struct SequencePlannerImpl<GemmaVariant> {
    GemmaVariant::WeightsProfile weights_profile{};
    EngineOptions options;
    runtime::SequenceCapacityCurve curve;
};

template <>
struct RequestBasePlanImpl<GemmaVariant> {
    runtime::RequestPlanSummary summary;
    qwen3_6::PreparedContextCache context_cache;
    ResolvedSamplingParameters sampling;
    std::vector<std::array<std::uint64_t, 2>> prefix_digests;
    std::uint32_t prefix_identity_tag = 0x47454d34U;
    bool allow_prefix_reuse = false;
};

template <>
struct AdmissionCandidateImpl<GemmaVariant> {
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment identity_assessment;
    ResolvedSamplingParameters sampling;
    runtime::LaneId destination{};
    std::uint32_t source_index = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t source_generation = 0;
    std::uint32_t reuse_frontier = 0;
    runtime::PlanningOwnerId victim_owner{};
    std::uint32_t victim_index = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t victim_generation = 0;
    bool has_source = false;
    bool has_victim = false;
    bool needs_transfer = false;
};

template <>
struct CapturePressureCandidateImpl<GemmaVariant> {};

template <>
struct PressurePlanningSessionImpl<GemmaVariant>;

template <>
class ProgramImpl<GemmaVariant>;

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::gemma4_31b_it::detail {
namespace {

using Variant                  = gemma4_31b_it::detail::Variant;
using SequencePlanImpl         = qwen3_6::detail::SequencePlanImpl<Variant>;
using SequencePlannerImpl      = qwen3_6::detail::SequencePlannerImpl<Variant>;
using RequestBasePlanImpl      = qwen3_6::detail::RequestBasePlanImpl<Variant>;
using AdmissionCandidateImpl   = qwen3_6::detail::AdmissionCandidateImpl<Variant>;
using ContractAccess           = qwen3_6::detail::RuntimeContractAccess<Variant>;
using PreparedPromptData       = qwen3_6::PreparedPromptData;
using ContinuationHandle       = qwen3_6::ContinuationHandle<Variant>;
using SequenceHandle           = qwen3_6::SequenceHandle<Variant>;
using PendingBatch             = qwen3_6::PendingBatch<Variant>;
using ContinuationSummary      = qwen3_6::ContinuationSummary;
using CheckpointSummary        = qwen3_6::CheckpointSummary;
using PrefixShortlistKey       = qwen3_6::PrefixShortlistKey;
using PhysicalUsageSnapshot    = qwen3_6::PhysicalUsageSnapshot;
using RetainedSessionSnapshot  = qwen3_6::RetainedSessionSnapshot;
using SessionSnapshotTraffic   = qwen3_6::SessionSnapshotTraffic;

constexpr std::uint32_t kPageTokens = static_cast<std::uint32_t>(kPagedKVPageSize);
constexpr std::uint32_t kMaximumPrefillChunk = 2048;
constexpr ops::AttentionHeadGeometry kFullGeometry{
    TextConfig::full_head_dim, TextConfig::query_heads, TextConfig::full_kv_heads};

std::uint32_t pages_for(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / kPageTokens;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

void append_prefix_digest(std::array<std::uint64_t, 2>& digest, TokenId token) noexcept {
    const std::uint32_t value = static_cast<std::uint32_t>(token);
    for (unsigned shift = 0; shift < 32; shift += 8) {
        digest[0] ^= static_cast<std::uint8_t>(value >> shift);
        digest[0] *= 1099511628211ULL;
    }
    digest[1] ^= static_cast<std::uint64_t>(value) + 0x9e3779b97f4a7c15ULL +
                 (digest[1] << 6U) + (digest[1] >> 2U);
}

std::array<std::uint64_t, 2> prefix_digest(std::span<const TokenId> tokens) noexcept {
    std::array<std::uint64_t, 2> digest{1469598103934665603ULL, 0x9e3779b97f4a7c15ULL};
    for (TokenId token : tokens) append_prefix_digest(digest, token);
    return digest;
}

std::string digest_hex(std::span<const TokenId> tokens) {
    const std::uint64_t value = prefix_digest(tokens)[0];
    constexpr char digits[] = "0123456789abcdef";
    std::string out(16, '0');
    for (std::size_t index = 0; index < out.size(); ++index) {
        out[out.size() - 1U - index] = digits[(value >> (index * 4U)) & 0xfU];
    }
    return out;
}

std::array<KvGroupSpec, 2> group_specs(std::uint32_t maximum_context,
                                       std::uint32_t maximum_transaction_tokens,
                                       std::uint32_t global_page_groups,
                                       std::uint32_t concurrency) {
    const std::uint64_t tokens = static_cast<std::uint64_t>(global_page_groups) * kPageTokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Gemma aggregate KV capacity exceeds uint32");
    }
    return make_text_kv_group_specs({
        .maximum_context = maximum_context,
        .maximum_transaction_tokens = maximum_transaction_tokens,
        .global_resident_token_capacity = static_cast<std::uint32_t>(tokens),
        .local_checkpoint_capacity = 0,
        .table_rows = static_cast<std::int32_t>(concurrency),
    });
}

struct PlannedLayout {
    std::array<KvGroupSpec, 2> groups;
    HeterogeneousKVCacheLayout cache;
    std::size_t cache_bytes = 0;
    std::size_t attention_bytes = 0;
    std::size_t total_bytes = 0;
};

std::size_t activation_workspace_bytes(std::uint32_t prefill_chunk,
                                       std::uint32_t mtp_draft_tokens) {
    std::size_t bytes = runtime_activation_workspace_capacity_bytes(
        static_cast<std::int32_t>(prefill_chunk), false);
    if (mtp_draft_tokens != 0) {
        bytes = std::max(bytes, runtime_activation_workspace_capacity_bytes(
                                    static_cast<std::int32_t>(mtp_draft_tokens + 1U), true));
    }
    return bytes;
}

PlannedLayout plan_layout(std::uint32_t maximum_context, std::uint32_t prefill_chunk,
                          std::uint32_t page_groups, std::uint32_t concurrency,
                          bool use_mtp) {
    PlannedLayout out;
    out.groups = group_specs(maximum_context, prefill_chunk, page_groups, concurrency);
    LayoutBuilder builder;
    out.cache = plan_heterogeneous_kv_cache(builder, out.groups);
    out.cache_bytes = builder.finish(256, "Gemma heterogeneous KV arena");
    out.attention_bytes = ops::causal_full_softmax_attention_workspace_capacity_bytes(
        kFullGeometry, KvCacheStorage::RK4V4E8, {1, maximum_context}, 1,
        static_cast<std::int32_t>(std::min(prefill_chunk, kPageTokens)));
    const std::size_t activation_bytes = activation_workspace_bytes(
        prefill_chunk, use_mtp ? AssistantConfig::maximum_draft_size : 0U);
    out.total_bytes = checked_add(out.cache_bytes, activation_bytes,
                                  "Gemma runtime reservation overflow");
    out.total_bytes = checked_add(out.total_bytes, std::max<std::size_t>(out.attention_bytes, 256),
                                  "Gemma runtime reservation overflow");
    if (use_mtp) {
        constexpr std::size_t state_per_lane = 2ULL * TextConfig::hidden * 2ULL;
        out.total_bytes = checked_add(out.total_bytes, state_per_lane * concurrency,
                                      "Gemma MTP state reservation overflow");
    }
    return out;
}

std::unique_ptr<SequencePlannerImpl>
make_planner(DeviceContext&, const EngineOptions& options, Variant::WeightsProfile profile) {
    if (options.max_context < TextConfig::sliding_window ||
        options.max_context > TextConfig::maximum_position) {
        throw std::invalid_argument("Gemma max_context must be in [1024,262144]");
    }
    if (options.purpose == EnginePurpose::CausalScoring && options.max_concurrency != 1) {
        throw std::invalid_argument("Gemma causal scoring requires max_concurrency=1");
    }
    if (options.speculative.backend == SpeculativeBackend::Mtp &&
        (options.speculative.draft_tokens == 0 ||
         options.speculative.draft_tokens > AssistantConfig::maximum_draft_size)) {
        throw std::invalid_argument("Gemma MTP draft_tokens must be in [1,6]");
    }
    const std::uint32_t chunk = std::min(options.prefill_chunk, kMaximumPrefillChunk);
    if (chunk == 0) { throw std::invalid_argument("Gemma prefill chunk must be nonzero"); }
    const std::uint32_t minimum_pages = pages_for(options.max_context);
    const std::uint64_t maximum_tokens =
        static_cast<std::uint64_t>(options.max_context) * options.max_concurrency;
    const std::uint32_t maximum_pages = static_cast<std::uint32_t>(
        (maximum_tokens + kPageTokens - 1U) / kPageTokens);
    const PlannedLayout minimum =
        plan_layout(options.max_context, chunk, minimum_pages, options.max_concurrency,
                    options.speculative.backend == SpeculativeBackend::Mtp);
    const PlannedLayout one_more = minimum_pages < maximum_pages
                                       ? plan_layout(options.max_context, chunk,
                                                     minimum_pages + 1U,
                                                     options.max_concurrency,
                                                     options.speculative.backend ==
                                                         SpeculativeBackend::Mtp)
                                       : minimum;
    auto planner = std::make_unique<SequencePlannerImpl>();
    planner->weights_profile = profile;
    planner->options = options;
    planner->curve = {
        .main_page_tokens = kPageTokens,
        .minimum_main_page_groups = minimum_pages,
        .maximum_main_page_groups = maximum_pages,
        .minimum_device_reservation_bytes = minimum.total_bytes,
        .bytes_per_additional_main_page_group =
            minimum_pages < maximum_pages ? one_more.total_bytes - minimum.total_bytes : 0,
    };
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_planner(std::unique_ptr<SequencePlannerImpl> planner, std::uint32_t page_groups) {
    if (!planner) { throw std::logic_error("Gemma sequence planner is empty"); }
    const auto& options = planner->options;
    const std::uint32_t chunk = std::min(options.prefill_chunk, kMaximumPrefillChunk);
    PlannedLayout layout =
        plan_layout(options.max_context, chunk, page_groups, options.max_concurrency,
                    options.speculative.backend == SpeculativeBackend::Mtp);
    auto plan = std::make_unique<SequencePlanImpl>();
    plan->weights_profile = planner->weights_profile;
    plan->capacity = options.max_context;
    plan->kv_capacity = planner->curve.resolved_tokens(page_groups);
    plan->main_page_groups = page_groups;
    plan->max_concurrency = options.max_concurrency;
    plan->prefill_chunk = chunk;
    plan->device = options.device;
    plan->context_cache = options.context_cache;
    plan->speculative = options.speculative;
    plan->use_cuda_graph = options.use_cuda_graph;
    plan->groups = std::move(layout.groups);
    plan->cache_layout = std::move(layout.cache);
    const std::size_t mtp_state = options.speculative.backend == SpeculativeBackend::Mtp
                                      ? 2ULL * TextConfig::hidden * dtype_size(DType::BF16) *
                                            options.max_concurrency
                                      : 0;
    const std::size_t activation_bytes = activation_workspace_bytes(
        chunk, options.speculative.backend == SpeculativeBackend::Mtp
                   ? options.speculative.draft_tokens
                   : 0U);
    plan->workspace = {.activation = activation_bytes,
                       .attention = std::max<std::size_t>(layout.attention_bytes, 256),
                       .mtp_state = mtp_state,
                       .capacity = checked_add(activation_bytes,
                                               std::max<std::size_t>(layout.attention_bytes, 256),
                                               "Gemma workspace capacity overflow")};
    plan->workspace.capacity = checked_add(plan->workspace.capacity, mtp_state,
                                           "Gemma MTP workspace capacity overflow");
    plan->sequence_capacity_bytes = layout.cache_bytes;
    plan->device_reservation_bytes = layout.total_bytes;
    if (plan->device_reservation_bytes != planner->curve.reservation_bytes(page_groups)) {
        throw std::logic_error("Gemma KV capacity curve is not affine");
    }
    return plan;
}

std::uint64_t random_word(std::uint64_t seed, std::uint32_t position) noexcept {
    std::uint64_t value = seed ^ (static_cast<std::uint64_t>(position) << 32U) ^
                          0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

TokenId sample_logits(std::span<const float> logits, const ResolvedSamplingParameters& config,
                      std::span<const std::uint32_t> counts, std::uint32_t position,
                      std::span<const TokenId> tentative = {}) {
    const auto adjusted = [&](TokenId token) {
        float value = logits[static_cast<std::size_t>(token)];
        const std::uint32_t count = counts[static_cast<std::size_t>(token)] +
                                    static_cast<std::uint32_t>(
                                        std::count(tentative.begin(), tentative.end(), token));
        if (count != 0) { value -= config.presence_penalty; }
        value -= config.frequency_penalty * static_cast<float>(count);
        return value;
    };
    if (!(config.temperature > 0.0F)) {
        TokenId best = 0;
        float best_value = adjusted(0);
        for (TokenId token = 1; token < static_cast<TokenId>(logits.size()); ++token) {
            const float value = adjusted(token);
            if (value > best_value) { best_value = value; best = token; }
        }
        return best;
    }
    const std::size_t top_k = std::min<std::size_t>(
        static_cast<std::size_t>(std::max(config.top_k, 1)), logits.size());
    std::vector<std::pair<float, TokenId>> ranked;
    ranked.reserve(logits.size());
    for (TokenId token = 0; token < static_cast<TokenId>(logits.size()); ++token) {
        ranked.emplace_back(adjusted(token), token);
    }
    const auto better = [](const auto& left, const auto& right) {
        return left.first > right.first ||
               (left.first == right.first && left.second < right.second);
    };
    if (top_k < ranked.size()) {
        std::nth_element(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(top_k),
                         ranked.end(), better);
    }
    ranked.resize(top_k);
    std::sort(ranked.begin(), ranked.end(), better);
    const float maximum = ranked.front().first / config.temperature;
    std::vector<double> weights(ranked.size());
    for (std::size_t index = 0; index < ranked.size(); ++index) {
        weights[index] = std::exp(static_cast<double>(ranked[index].first / config.temperature -
                                                       maximum));
    }
    if (config.min_p > 0.0F) {
        const double threshold = static_cast<double>(config.min_p) * weights.front();
        while (weights.size() > 1 && weights.back() < threshold) {
            weights.pop_back(); ranked.pop_back();
        }
    }
    if (config.top_p < 1.0F) {
        const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        double cumulative = 0.0;
        std::size_t keep = 0;
        do { cumulative += weights[keep++]; }
        while (keep < weights.size() && cumulative < total * config.top_p);
        weights.resize(keep); ranked.resize(keep);
    }
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    const double unit = static_cast<double>(random_word(config.seed, position) >> 11U) *
                        (1.0 / 9007199254740992.0);
    double cumulative = 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index) {
        cumulative += weights[index] / total;
        if (unit < cumulative) { return ranked[index].second; }
    }
    return ranked.back().second;
}

CheckpointSummary checkpoint_summary(std::span<const TokenId> ledger,
                                     std::uint32_t frontier, std::uint32_t prefill_chunk) {
    CheckpointSummary out;
    out.ref = {.kind = runtime::CheckpointKind::SessionEndpoint, .frontier = frontier};
    out.scope = runtime::CheckpointScope::Private;
    out.shortlist_key = {.digests = prefix_digest(ledger.first(frontier)),
                         .frontier = frontier,
                         .identity_tag = 0x47454d34U};
    out.state_residency = runtime::ReplicaResidency::HostOnly;
    out.required_kv = {.main_frontier = frontier,
                       .main_pages = pages_for(frontier)};
    out.rebuild_work = runtime::make_prefill_work(0, frontier, 0, 0, prefill_chunk);
    return out;
}

} // namespace
} // namespace ninfer::targets::gemma4_31b_it::detail

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<GemmaVariant> {
public:
    using Variant = GemmaVariant;
    using ContractAccess = RuntimeContractAccess<Variant>;
    using SequenceHandle = qwen3_6::SequenceHandle<Variant>;
    using ContinuationHandle = qwen3_6::ContinuationHandle<Variant>;
    using PendingBatch = qwen3_6::PendingBatch<Variant>;
    using ContextProgress = qwen3_6::ContextTransactionProgress<Variant>;

    ProgramImpl(const Variant::ModelView& weights, const SequencePlanImpl<Variant>& plan,
                DeviceContext& device, const StartupObserver&)
        : weights_(weights), plan_(plan), device_(device), cache_backing_(plan.sequence_capacity_bytes),
          cache_({cache_backing_.p, cache_backing_.bytes}, plan.cache_layout),
          activation_backing_(plan.workspace.activation),
          attention_backing_(plan.workspace.attention),
          activations_({activation_backing_.p, activation_backing_.bytes}),
          attention_({attention_backing_.p, attention_backing_.bytes}),
          lanes_(plan.max_concurrency), continuations_(
              plan.context_cache.max_private_continuations.value_or(plan.max_concurrency)) {
        if (!plan.context_cache.max_private_continuations) {
            throw std::logic_error("Gemma context cache options were not normalized");
        }
        if (plan.speculative.backend == SpeculativeBackend::Mtp) {
            if (!weights_.assistant) {
                throw std::invalid_argument("Gemma MTP requested without resident assistant weights");
            }
            mtp_hidden_backing_.emplace(plan.workspace.mtp_state);
            constexpr std::size_t state_bytes = gemma4_31b_it::TextConfig::hidden * 2ULL;
            auto* base = static_cast<std::byte*>(mtp_hidden_backing_->p);
            for (std::size_t lane = 0; lane < lanes_.size(); ++lane) {
                lanes_[lane].target_hidden =
                    Tensor(base + (2U * lane) * state_bytes, DType::BF16,
                           {static_cast<std::int32_t>(gemma4_31b_it::TextConfig::hidden)});
                lanes_[lane].assistant_feedback =
                    Tensor(base + (2U * lane + 1U) * state_bytes, DType::BF16,
                           {static_cast<std::int32_t>(gemma4_31b_it::TextConfig::hidden)});
            }
        }
        if (plan.use_cuda_graph) {
            ordinary_graphs_.reserve(lanes_.size());
            mtp_graphs_.reserve(lanes_.size());
            for (std::size_t lane = 0; lane < lanes_.size(); ++lane) {
                ordinary_graphs_.push_back(
                    std::make_unique<gemma4_31b_it::detail::RuntimeDecodeGraph>());
                mtp_graphs_.push_back(
                    std::make_unique<gemma4_31b_it::detail::RuntimeMtpVerifyGraph>());
            }
        }
    }

    struct ContinuationRecord {
        gemma4::detail::ContinuationState state;
        std::uint64_t generation = 0;
        std::size_t bytes = 0;
    };

    struct Lane {
        std::uint64_t epoch = 0;
        bool active = false;
        bool prefilling = false;
        bool publish_continuation = false;
        std::uint32_t cursor = 0;
        std::uint32_t reserved_tokens = 0;
        runtime::BeginSummary begin;
        ResolvedSamplingParameters sampling;
        qwen3_6::PreparedPromptData prompt;
        std::vector<TokenId> ledger;
        std::vector<std::uint32_t> token_counts;
        std::optional<TokenId> current_token;
        std::optional<TokenId> ready_token;
        Tensor target_hidden;
        Tensor assistant_feedback;
        bool mtp_enabled = false;
        SpeculativeStats speculative;
        GenerationTimings timings;
    };

    struct Materialization {
        AdmissionCandidateImpl<Variant> candidate;
        qwen3_6::PreparedPromptData prompt;
    };

    [[nodiscard]] Lane& require_sequence(SequenceHandle handle) {
        if (ContractAccess::owner(handle) != this ||
            ContractAccess::lane(handle).value >= lanes_.size()) {
            throw std::invalid_argument("Gemma sequence handle is invalid");
        }
        Lane& lane = lanes_[ContractAccess::lane(handle).value];
        if (!lane.active || lane.epoch != ContractAccess::epoch(handle)) {
            throw std::invalid_argument("Gemma sequence handle is stale");
        }
        return lane;
    }

    [[nodiscard]] const ContinuationRecord* continuation(const ContinuationHandle& handle) const {
        const std::uint32_t index = ContractAccess::index(handle);
        if (ContractAccess::owner(handle) != this || index >= continuations_.size() ||
            !continuations_[index] ||
            continuations_[index]->generation != ContractAccess::epoch(handle)) {
            return nullptr;
        }
        return &*continuations_[index];
    }

    [[nodiscard]] ContinuationRecord* continuation(const ContinuationHandle& handle) {
        return const_cast<ContinuationRecord*>(std::as_const(*this).continuation(handle));
    }

    [[nodiscard]] std::uint32_t active_reserved() const noexcept {
        std::uint64_t total = 0;
        for (const Lane& lane : lanes_) if (lane.active) total += lane.reserved_tokens;
        return total > std::numeric_limits<std::uint32_t>::max()
                   ? std::numeric_limits<std::uint32_t>::max()
                   : static_cast<std::uint32_t>(total);
    }

    void reset_lane(std::uint32_t row) noexcept {
        Tensor target_hidden = lanes_[row].target_hidden;
        Tensor assistant_feedback = lanes_[row].assistant_feedback;
        lanes_[row] = {};
        lanes_[row].target_hidden = target_hidden;
        lanes_[row].assistant_feedback = assistant_feedback;
    }

    [[nodiscard]] TokenId run_and_sample(Lane& lane, std::uint32_t row,
                                         std::span<const TokenId> tokens,
                                         bool produce_logits = true) {
        const std::uint32_t first = cache_.frontier(static_cast<std::int32_t>(row));
        std::vector<float> logits;
        if (produce_logits && plan_.use_cuda_graph && tokens.size() == 1) {
            auto& graph = *ordinary_graphs_[row];
            if (!graph.ready()) {
                graph.capture(weights_.target, cache_, static_cast<std::int32_t>(row),
                              plan_.capacity, activations_, attention_, device_.stream,
                              lane.mtp_enabled ? lane.target_hidden : Tensor{});
            }
            logits = graph.replay(cache_, static_cast<std::int32_t>(row), first,
                                  tokens.front(), activations_, device_.stream);
        } else {
            logits = gemma4_31b_it::detail::execute_runtime_chunk(
                weights_.target, cache_, static_cast<std::int32_t>(row), first, tokens,
                plan_.capacity, activations_, attention_, device_.stream, produce_logits,
                produce_logits && lane.mtp_enabled ? lane.target_hidden : Tensor{});
        }
        if (!produce_logits) { return -1; }
        return gemma4_31b_it::detail::sample_logits(logits, lane.sampling, lane.token_counts,
                                                    first + static_cast<std::uint32_t>(tokens.size()));
    }

    [[nodiscard]] qwen3_6::ContinuationSummary summary(const ContinuationRecord& record) const {
        qwen3_6::ContinuationSummary out;
        const std::uint32_t frontier = record.state.endpoint.groups.empty()
                                           ? 0
                                           : record.state.endpoint.groups.front().frontier;
        if (frontier != 0 && frontier <= record.state.ledger.size()) {
            out.endpoint = gemma4_31b_it::detail::checkpoint_summary(
                record.state.ledger, frontier, plan_.prefill_chunk);
        }
        return out;
    }

    const Variant::ModelView weights_;
    const SequencePlanImpl<Variant> plan_;
    DeviceContext& device_;
    DeviceBuffer cache_backing_;
    HeterogeneousKVCache cache_;
    DeviceBuffer activation_backing_;
    DeviceBuffer attention_backing_;
    std::optional<DeviceBuffer> mtp_hidden_backing_;
    std::vector<std::unique_ptr<gemma4_31b_it::detail::RuntimeDecodeGraph>> ordinary_graphs_;
    std::vector<std::unique_ptr<gemma4_31b_it::detail::RuntimeMtpVerifyGraph>> mtp_graphs_;
    WorkspaceArena activations_;
    WorkspaceArena attention_;
    std::vector<Lane> lanes_;
    std::vector<std::optional<ContinuationRecord>> continuations_;
    std::optional<Materialization> materialization_;
    bool materialization_terminal_ = false;
    std::vector<TokenId> pending_tokens_;
    std::array<std::int32_t, kMaximumConcurrency> pending_counts_{};
    std::array<SequenceHandle, kMaximumConcurrency> pending_rows_{};
    std::size_t pending_row_count_ = 0;
    std::array<std::optional<HeterogeneousKVTransaction>, kMaximumConcurrency>
        pending_mtp_transactions_;
    std::array<std::uint32_t, kMaximumConcurrency> pending_mtp_drafted_{};
    std::array<std::uint32_t, kMaximumConcurrency> pending_mtp_accepted_{};
    std::uint64_t pending_id_ = 0;
    std::uint64_t next_pending_id_ = 1;
    std::uint64_t next_generation_ = 1;
    runtime::ProgramResourceRevision revision_{1};
    qwen3_6::SessionSnapshotTraffic snapshot_traffic_;
};

template <>
struct PressurePlanningSessionImpl<GemmaVariant> {
    using Variant = GemmaVariant;
    using Program = ProgramImpl<Variant>;
    using Admission = qwen3_6::AdmissionCandidate<Variant>;
    using AdmissionImpl = AdmissionCandidateImpl<Variant>;
    using Continuation = qwen3_6::ContinuationHandle<Variant>;
    using Assessed = qwen3_6::AssessedPressureTarget<Variant>;
    using PreparedExpansion = qwen3_6::PreparedPressureExpansion<Variant>;
    using Access = RuntimeContractAccess<Variant>;

    struct Candidate {
        const AdmissionImpl* admission = nullptr;
        runtime::PlanningCandidateId id;
        std::uint32_t identity_target = 0;
        std::vector<std::uint32_t> eviction_targets;
    };

    struct Owner {
        const Continuation* handle = nullptr;
        runtime::PlanningOwnerId id;
        std::uint32_t index = 0;
        std::uint64_t generation = 0;
        qwen3_6::ContinuationSummary summary;
    };

    struct Target {
        std::uint32_t candidate = 0;
        std::optional<std::uint32_t> victim;
        std::uint32_t ordinal = 0;
        bool root_maximal = false;
        std::vector<runtime::PressureOwnerOutcome> outcomes;
        std::vector<std::vector<runtime::CheckpointRecoveryAlternativeWork>>
            checkpoint_recovery;
        std::vector<runtime::PressureCheckpointRecoveryImpact> checkpoint_impacts;
    };

    PressurePlanningSessionImpl(
        Program& owner, std::span<const Admission* const> admissions,
        std::span<const runtime::PlanningCandidateId> candidate_ids,
        std::span<const Continuation* const> private_owners,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const qwen3_6::SharedPrefixHandle<Variant>* const> shared_owners,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids)
        : program(&owner), resource_revision(owner.revision_) {
        if (admissions.size() != candidate_ids.size() ||
            private_owners.size() != private_owner_ids.size() ||
            shared_owners.size() != shared_owner_ids.size() || !shared_owners.empty()) {
            throw std::invalid_argument("Gemma pressure planning inputs are not aligned");
        }
        owners.reserve(private_owners.size());
        for (std::size_t row = 0; row < private_owners.size(); ++row) {
            if (private_owners[row] == nullptr) {
                throw std::invalid_argument("Gemma pressure owner is empty");
            }
            const auto* record = owner.continuation(*private_owners[row]);
            if (record == nullptr) {
                throw std::invalid_argument("Gemma pressure owner is stale");
            }
            owners.push_back(Owner{
                .handle = private_owners[row],
                .id = private_owner_ids[row],
                .index = Access::index(*private_owners[row]),
                .generation = Access::epoch(*private_owners[row]),
                .summary = owner.summary(*record),
            });
        }

        candidates.reserve(admissions.size());
        targets.reserve(admissions.size() * (owners.size() + 1U));
        for (std::size_t row = 0; row < admissions.size(); ++row) {
            if (admissions[row] == nullptr || admissions[row]->impl_ == nullptr) {
                throw std::invalid_argument("Gemma pressure candidate is empty");
            }
            Candidate candidate{
                .admission = admissions[row]->impl_.get(),
                .id = candidate_ids[row],
                .identity_target = static_cast<std::uint32_t>(targets.size()),
            };
            targets.push_back(Target{
                .candidate = static_cast<std::uint32_t>(row),
                .ordinal = static_cast<std::uint32_t>(targets.size()),
            });
            for (std::size_t victim = 0; victim < owners.size(); ++victim) {
                const Owner& selected = owners[victim];
                if (candidate.admission->has_source &&
                    selected.index == candidate.admission->source_index &&
                    selected.generation == candidate.admission->source_generation) {
                    continue;
                }
                Target target{
                    .candidate = static_cast<std::uint32_t>(row),
                    .victim = static_cast<std::uint32_t>(victim),
                    .ordinal = static_cast<std::uint32_t>(targets.size()),
                };
                const std::uint32_t checkpoints = checkpoint_count(selected.summary);
                target.checkpoint_recovery.reserve(checkpoints);
                target.checkpoint_impacts.reserve(checkpoints);
                target.outcomes.push_back(runtime::PressureOwnerOutcome{
                    .owner = selected.id,
                    .disposition = runtime::VictimDisposition::Evicted,
                    .degradation_units = 1,
                    .dropped_checkpoints = checkpoints,
                });
                append_checkpoint_impacts(target, selected);
                candidate.eviction_targets.push_back(target.ordinal);
                targets.push_back(std::move(target));
            }
            if (!candidate.eviction_targets.empty()) {
                targets[candidate.eviction_targets.front()].root_maximal = true;
            }
            candidates.push_back(std::move(candidate));
        }
    }

    [[nodiscard]] qwen3_6::PressureTargetHandle
    identity_target(runtime::PlanningCandidateId id) const {
        return handle(candidates.at(candidate_index(id)).identity_target);
    }

    [[nodiscard]] qwen3_6::PressureTargetHandle
    root_maximal_target(runtime::PlanningCandidateId id) const {
        const Candidate& candidate = candidates.at(candidate_index(id));
        if (candidate.eviction_targets.empty()) {
            throw std::logic_error("Gemma pressure planning has no eligible victim");
        }
        return handle(candidate.eviction_targets.front());
    }

    [[nodiscard]] qwen3_6::PressureTargetHandle
    maximal_target(runtime::PlanningCandidateId id) const {
        return root_maximal_target(id);
    }

    [[nodiscard]] qwen3_6::PressureConstructionCursor
    begin_construction(qwen3_6::PressureTargetHandle target, bool) {
        (void)require_target(target);
        return qwen3_6::PressureConstructionCursor(this, 0, ++construction_generation,
                                                   &release_construction);
    }

    [[nodiscard]] runtime::PressureConstructionStep
    next_construction_option(qwen3_6::PressureConstructionCursor&) {
        return runtime::PressureConstructionStep{.exhausted = true};
    }

    void choose_construction(qwen3_6::PressureConstructionCursor&,
                             runtime::PressureConstructionOptionId) {
        throw std::logic_error("Gemma pressure construction has no deferred options");
    }

    [[nodiscard]] std::optional<qwen3_6::PressureTargetHandle>
    construction_target(const qwen3_6::PressureConstructionCursor&) const {
        return std::nullopt;
    }

    [[nodiscard]] runtime::PressureTargetGuidance
    guidance(qwen3_6::PressureTargetHandle pressure_target) const {
        const Target& target = require_target(pressure_target);
        const Candidate& candidate = candidates.at(target.candidate);
        return runtime::PressureTargetGuidance{
            .physical = runtime::PressurePhysicalGuidance{
                .estimated_remaining_steps = target.victim ? 0U : 1U,
            },
            .estimated_machine_work = candidate.admission->identity_assessment.machine_work,
            .owner_outcomes = target.outcomes,
            .candidate = candidate.id,
            .stable_target_ordinal = target.ordinal,
            .degradation_units = target.victim ? 1U : 0U,
            .dropped_checkpoints = target.victim
                                       ? target.outcomes.front().dropped_checkpoints
                                       : 0U,
            .source_mode = candidate.admission->identity_assessment.source_mode,
            .recovery_estimate_complete = true,
        };
    }

    [[nodiscard]] Assessed assess(qwen3_6::PressureTargetHandle pressure_target) const {
        const Target& target = require_target(pressure_target);
        const Candidate& candidate = candidates.at(target.candidate);
        auto executable_impl = std::make_unique<AdmissionImpl>(*candidate.admission);
        if (target.victim) {
            const Owner& victim = owners.at(*target.victim);
            executable_impl->has_victim = true;
            executable_impl->victim_owner = victim.id;
            executable_impl->victim_index = victim.index;
            executable_impl->victim_generation = victim.generation;
        }
        std::optional<Admission> executable;
        executable.emplace(std::move(executable_impl));
        runtime::PressureTargetAssessment assessment{
            .physical_status = runtime::MaterializationPhysicalStatus::Feasible,
            .source_mode = candidate.admission->identity_assessment.source_mode,
            .machine_work = candidate.admission->identity_assessment.machine_work,
            .owner_outcomes = target.outcomes,
            .checkpoint_impacts = target.checkpoint_impacts,
            .candidate = candidate.id,
            .stable_target_ordinal = target.ordinal,
            .degradation_units = target.victim ? 1U : 0U,
            .dropped_checkpoints = target.victim
                                       ? target.outcomes.front().dropped_checkpoints
                                       : 0U,
            .projection_work = 1,
            .assessment_digest = resource_revision.value ^
                                 (static_cast<std::uint64_t>(target.ordinal) << 32U),
            .expandable = !target.victim && !candidate.eviction_targets.empty(),
            .root_maximal = target.root_maximal,
        };
        return Assessed(this, generation, target.ordinal, assessment, 0, 0, nullptr,
                        std::move(executable), std::nullopt);
    }

    [[nodiscard]] PreparedExpansion
    prepare_expansion(qwen3_6::PressureTargetHandle parent, std::uint32_t maximum_owners) {
        const Target& target = require_target(parent);
        prepared_children.clear();
        if (!target.victim) {
            const auto& children = candidates.at(target.candidate).eviction_targets;
            const std::size_t count = std::min<std::size_t>(children.size(), maximum_owners);
            prepared_children.insert(prepared_children.end(), children.begin(),
                                     children.begin() + static_cast<std::ptrdiff_t>(count));
        }
        prepared_parent = target.ordinal;
        ++scratch_generation;
        return PreparedExpansion(this, generation, scratch_generation, target.ordinal,
                                 static_cast<std::uint32_t>(prepared_children.size()));
    }

    [[nodiscard]] qwen3_6::PressureExpansionView
    commit_expansion(PreparedExpansion&& prepared) {
        if (prepared.session_ != this || prepared.session_generation_ != generation ||
            prepared.scratch_generation_ != scratch_generation ||
            prepared.parent_index_ != prepared_parent) {
            throw std::logic_error("Gemma pressure expansion is stale");
        }
        committed_children.clear();
        committed_children.reserve(prepared_children.size());
        for (std::uint32_t child : prepared_children) {
            committed_children.push_back(handle(child));
        }
        prepared.session_ = nullptr;
        return qwen3_6::PressureExpansionView{
            .children = committed_children,
            .new_canonical_count = static_cast<std::uint32_t>(committed_children.size()),
            .complete = true,
        };
    }

    void discard_expansion(PreparedExpansion&& prepared) noexcept {
        prepared.session_ = nullptr;
        prepared_children.clear();
    }

    [[nodiscard]] runtime::PrefillWork shared_capture_split_prefill_work(
        const Assessed&, const qwen3_6::PreparedPromptData&,
        std::span<const std::uint32_t>) const {
        return {};
    }

    [[nodiscard]] std::optional<Admission>
    seal(Assessed&& assessed, const qwen3_6::PreparedPromptData&,
         runtime::FinalScheduleIntent) {
        if (assessed.session_ != this || assessed.session_generation_ != generation ||
            resource_revision != program->revision_ || !assessed.executable_) {
            return std::nullopt;
        }
        std::optional<Admission> sealed;
        sealed.emplace(std::move(*assessed.executable_));
        assessed.executable_.reset();
        return sealed;
    }

    [[nodiscard]] static std::uint32_t
    checkpoint_count(const qwen3_6::ContinuationSummary& summary) noexcept {
        return static_cast<std::uint32_t>(summary.endpoint.has_value() +
                                          summary.rewrite.has_value() +
                                          summary.long_anchors.size());
    }

    static void append_checkpoint_impacts(Target& target, const Owner& owner) {
        const auto append_checkpoint = [&](const qwen3_6::CheckpointSummary& checkpoint) {
            target.checkpoint_recovery.push_back(
                {runtime::CheckpointRecoveryAlternativeWork{
                    .prefill = checkpoint.rebuild_work,
                }});
            target.checkpoint_impacts.push_back(runtime::PressureCheckpointRecoveryImpact{
                .owner = owner.id,
                .checkpoint = checkpoint.ref,
                .target_recovery_work = target.checkpoint_recovery.back(),
                .survives = false,
            });
        };
        if (owner.summary.endpoint) { append_checkpoint(*owner.summary.endpoint); }
        if (owner.summary.rewrite) { append_checkpoint(*owner.summary.rewrite); }
        for (const auto& checkpoint : owner.summary.long_anchors) {
            append_checkpoint(checkpoint);
        }
    }

    [[nodiscard]] std::uint32_t candidate_index(runtime::PlanningCandidateId id) const {
        const auto found = std::find_if(candidates.begin(), candidates.end(),
                                        [&](const Candidate& candidate) {
                                            return candidate.id == id;
                                        });
        if (found == candidates.end()) {
            throw std::invalid_argument("Gemma pressure candidate ID is unknown");
        }
        return static_cast<std::uint32_t>(found - candidates.begin());
    }

    [[nodiscard]] qwen3_6::PressureTargetHandle handle(std::uint32_t index) const {
        qwen3_6::PressureTargetHandle out;
        out.session_ = this;
        out.generation_ = generation;
        out.index_ = index;
        return out;
    }

    [[nodiscard]] const Target& require_target(qwen3_6::PressureTargetHandle target) const {
        if (target.session_ != this || target.generation_ != generation ||
            target.index_ >= targets.size()) {
            throw std::invalid_argument("Gemma pressure target is stale");
        }
        return targets[target.index_];
    }

    static void release_construction(const void*, std::uint32_t, std::uint32_t) noexcept {}

    Program* program = nullptr;
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t generation = 1;
    std::uint32_t scratch_generation = 1;
    std::uint32_t construction_generation = 0;
    std::uint32_t prepared_parent = 0;
    std::vector<Candidate> candidates;
    std::vector<Owner> owners;
    std::vector<Target> targets;
    std::vector<std::uint32_t> prepared_children;
    std::vector<qwen3_6::PressureTargetHandle> committed_children;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6 {

using GemmaVariant = ::ninfer::targets::gemma4_31b_it::detail::Variant;
using GemmaProgramImpl = detail::ProgramImpl<GemmaVariant>;
using GemmaSequencePlanImpl = detail::SequencePlanImpl<GemmaVariant>;
using GemmaSequencePlannerImpl = detail::SequencePlannerImpl<GemmaVariant>;
using GemmaRequestBasePlanImpl = detail::RequestBasePlanImpl<GemmaVariant>;
using GemmaAdmissionCandidateImpl = detail::AdmissionCandidateImpl<GemmaVariant>;
using GemmaAccess = detail::RuntimeContractAccess<GemmaVariant>;
using GemmaTextConfig = ::ninfer::targets::gemma4_31b_it::TextConfig;
inline constexpr std::uint32_t kGemmaSlidingKvGroup =
    ::ninfer::targets::gemma4_31b_it::kSlidingKvGroup;
inline constexpr std::uint32_t kGemmaGlobalKvGroup =
    ::ninfer::targets::gemma4_31b_it::kGlobalKvGroup;

template <>
SequencePlan<GemmaVariant>::SequencePlan(std::unique_ptr<GemmaSequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
template <> SequencePlan<GemmaVariant>::SequencePlan(SequencePlan&&) noexcept = default;
template <> SequencePlan<GemmaVariant>& SequencePlan<GemmaVariant>::operator=(SequencePlan&&) noexcept = default;
template <> SequencePlan<GemmaVariant>::~SequencePlan() = default;
template <> std::uint32_t SequencePlan<GemmaVariant>::capacity() const noexcept { return impl_ ? impl_->capacity : 0; }
template <> std::uint32_t SequencePlan<GemmaVariant>::kv_capacity() const noexcept { return impl_ ? impl_->kv_capacity : 0; }
template <> std::uint32_t SequencePlan<GemmaVariant>::max_concurrency() const noexcept { return impl_ ? impl_->max_concurrency : 0; }
template <> std::size_t SequencePlan<GemmaVariant>::device_reservation_bytes() const noexcept { return impl_ ? impl_->device_reservation_bytes : 0; }
template <> std::size_t SequencePlan<GemmaVariant>::workspace_capacity_bytes() const noexcept { return impl_ ? impl_->workspace.capacity : 0; }

template <>
SequencePlanner<GemmaVariant>::SequencePlanner(std::unique_ptr<GemmaSequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}
template <> SequencePlanner<GemmaVariant>::SequencePlanner(SequencePlanner&&) noexcept = default;
template <> SequencePlanner<GemmaVariant>& SequencePlanner<GemmaVariant>::operator=(SequencePlanner&&) noexcept = default;
template <> SequencePlanner<GemmaVariant>::~SequencePlanner() = default;
template <> const runtime::SequenceCapacityCurve& SequencePlanner<GemmaVariant>::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ ? impl_->curve : empty;
}
template <> SequencePlan<GemmaVariant> SequencePlanner<GemmaVariant>::finalize(std::uint32_t pages) && {
    return SequencePlan<GemmaVariant>(gemma4_31b_it::detail::finalize_planner(std::move(impl_), pages));
}

template <>
RequestBasePlan<GemmaVariant>::RequestBasePlan(std::unique_ptr<GemmaRequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
template <> RequestBasePlan<GemmaVariant>::RequestBasePlan(RequestBasePlan&&) noexcept = default;
template <> RequestBasePlan<GemmaVariant>& RequestBasePlan<GemmaVariant>::operator=(RequestBasePlan&&) noexcept = default;
template <> RequestBasePlan<GemmaVariant>::~RequestBasePlan() = default;
template <> const runtime::RequestPlanSummary& RequestBasePlan<GemmaVariant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ ? impl_->summary : empty;
}
template <> const PreparedContextCache& RequestBasePlan<GemmaVariant>::context_cache() const noexcept {
    static const PreparedContextCache empty;
    return impl_ ? impl_->context_cache : empty;
}
template <> std::optional<PrefixShortlistKey>
RequestBasePlan<GemmaVariant>::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (!impl_ || frontier == 0 || frontier >= impl_->prefix_digests.size()) return std::nullopt;
    return PrefixShortlistKey{.digests = impl_->prefix_digests[frontier],
                              .frontier = frontier,
                              .identity_tag = impl_->prefix_identity_tag};
}
template <> std::optional<runtime::PrefillWork>
RequestBasePlan<GemmaVariant>::shared_candidate_rebuild_work(std::uint32_t) const noexcept {
    return std::nullopt;
}

template <>
AdmissionCandidate<GemmaVariant>::AdmissionCandidate(std::unique_ptr<GemmaAdmissionCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}
template <> AdmissionCandidate<GemmaVariant>::AdmissionCandidate(AdmissionCandidate&&) noexcept = default;
template <> AdmissionCandidate<GemmaVariant>& AdmissionCandidate<GemmaVariant>::operator=(AdmissionCandidate&&) noexcept = default;
template <> AdmissionCandidate<GemmaVariant>::~AdmissionCandidate() = default;
template <> const runtime::RequestPlanSummary& AdmissionCandidate<GemmaVariant>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ ? impl_->summary : empty;
}
template <> const runtime::IdentityMaterializationAssessment&
AdmissionCandidate<GemmaVariant>::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty;
    return impl_ ? impl_->identity_assessment : empty;
}

template <>
CapturePressureCandidate<GemmaVariant>::CapturePressureCandidate(
    std::unique_ptr<detail::CapturePressureCandidateImpl<GemmaVariant>> impl) noexcept
    : impl_(std::move(impl)) {}
template <> CapturePressureCandidate<GemmaVariant>::CapturePressureCandidate(CapturePressureCandidate&&) noexcept = default;
template <> CapturePressureCandidate<GemmaVariant>& CapturePressureCandidate<GemmaVariant>::operator=(CapturePressureCandidate&&) noexcept = default;
template <> CapturePressureCandidate<GemmaVariant>::~CapturePressureCandidate() = default;

template <>
Program<GemmaVariant>::Program(std::unique_ptr<GemmaProgramImpl> impl) noexcept : impl_(std::move(impl)) {}
template <> Program<GemmaVariant>::~Program() noexcept = default;

template <> RequestBasePlan<GemmaVariant>
Program<GemmaVariant>::plan_request(const PreparedPrompt& prompt,
                                    const runtime::ResolvedExecutionOptions& options) {
    const auto& data = PreparedPromptAccess::view(prompt);
    if (data.token_ids.empty() || data.token_ids.size() > impl_->plan_.capacity) {
        throw std::invalid_argument("Gemma prompt is empty or exceeds configured context");
    }
    auto base = std::make_unique<GemmaRequestBasePlanImpl>();
    base->context_cache = data.context_cache;
    base->sampling = options.sampling;
    base->allow_prefix_reuse = options.allow_prefix_reuse;
    base->summary.prompt_tokens = static_cast<std::uint32_t>(data.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        impl_->plan_.capacity - base->summary.prompt_tokens + 1U;
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason =
        options.requested_output_tokens <= capacity_output ? FinishReason::OutputLimit
                                                           : FinishReason::ContextCapacity;
    base->summary.publish_continuation = options.allow_prefix_reuse && data.identity.reusable &&
                                         impl_->plan_.context_cache.enabled;
    const std::uint32_t decode_units = base->summary.effective_output_tokens == 0
                                           ? 0
                                           : base->summary.effective_output_tokens - 1U;
    base->summary.service_work_quanta =
        (base->summary.prompt_tokens + impl_->plan_.prefill_chunk - 1U) /
            impl_->plan_.prefill_chunk + decode_units;
    base->prefix_digests.resize(data.token_ids.size() + 1U);
    std::array<std::uint64_t, 2> digest{1469598103934665603ULL, 0x9e3779b97f4a7c15ULL};
    for (std::size_t frontier = 1; frontier <= data.token_ids.size(); ++frontier) {
        gemma4_31b_it::detail::append_prefix_digest(digest, data.token_ids[frontier - 1U]);
        base->prefix_digests[frontier] = digest;
    }
    return RequestBasePlan<GemmaVariant>(std::move(base));
}

template <> bool Program<GemmaVariant>::isolated_request_feasible(
    const RequestBasePlan<GemmaVariant>& base) const noexcept {
    if (!base.impl_) return false;
    const std::uint64_t reservation = base.impl_->summary.prompt_tokens +
        (base.impl_->summary.effective_output_tokens == 0
             ? 0U : base.impl_->summary.effective_output_tokens - 1U);
    return reservation <= impl_->plan_.kv_capacity;
}

template <> std::optional<AdmissionCandidate<GemmaVariant>> Program<GemmaVariant>::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan<GemmaVariant>& base,
    runtime::LaneId destination, const ContinuationHandle<GemmaVariant>* source,
    const SharedPrefixHandle<GemmaVariant>*, std::optional<runtime::CheckpointRef>,
    bool must_retain_private_source) {
    if (!base.impl_ || destination.value >= impl_->lanes_.size() ||
        impl_->lanes_[destination.value].active) return std::nullopt;
    if (source && !base.impl_->allow_prefix_reuse) { return std::nullopt; }
    const std::uint32_t reservation = base.impl_->summary.prompt_tokens +
        (base.impl_->summary.effective_output_tokens == 0
             ? 0U : base.impl_->summary.effective_output_tokens - 1U);
    if (static_cast<std::uint64_t>(impl_->active_reserved()) + reservation >
        impl_->plan_.kv_capacity) return std::nullopt;
    auto candidate = std::make_unique<GemmaAdmissionCandidateImpl>();
    candidate->summary = base.impl_->summary;
    candidate->sampling = base.impl_->sampling;
    candidate->destination = destination;
    candidate->identity_assessment.physical_status =
        runtime::MaterializationPhysicalStatus::Feasible;
    candidate->identity_assessment.assessment_digest = impl_->revision_.value;
    candidate->identity_assessment.machine_work.remaining_prefill_work =
        runtime::make_prefill_work(0, candidate->summary.prompt_tokens, 0, 0,
                                   impl_->plan_.prefill_chunk);
    if (source) {
        const auto* record = impl_->continuation(*source);
        if (!record) { return std::nullopt; }
        const auto& incoming = PreparedPromptAccess::view(prompt).token_ids;
        auto selected = gemma4::detail::select_continuation_restore(
            record->state, incoming, impl_->plan_.groups, kGemmaSlidingKvGroup,
            kGemmaGlobalKvGroup);
        // An endpoint at the complete incoming prompt has no retained logits. It is not a reuse
        // candidate: the root path must recompute at least the final prompt token independently.
        if (selected.frontier == 0 || selected.frontier >= incoming.size()) {
            return std::nullopt;
        }
        candidate->has_source = true;
        candidate->source_index = GemmaAccess::index(*source);
        candidate->source_generation = GemmaAccess::epoch(*source);
        candidate->reuse_frontier = selected.frontier;
        candidate->summary.reusable_prompt_tokens = selected.frontier;
        candidate->summary.prefix_reuse_path = selected.endpoint
            ? PrefixReusePath::PrivateEndpoint : PrefixReusePath::PrivateLongAnchor;
        candidate->identity_assessment.source_mode =
            must_retain_private_source ? runtime::PrivateSourceMode::Retain
                                       : runtime::PrivateSourceMode::ConsumeToActive;
        candidate->identity_assessment.machine_work.reused_prompt_tokens = selected.frontier;
        candidate->identity_assessment.machine_work.remaining_prefill_work =
            runtime::make_prefill_work(selected.frontier, incoming.size() - selected.frontier,
                                       0, 0, impl_->plan_.prefill_chunk);
    }
    return AdmissionCandidate<GemmaVariant>(std::move(candidate));
}

template <> std::optional<ResourcePlan<GemmaVariant>> Program<GemmaVariant>::seal_identity(
    const AdmissionCandidate<GemmaVariant>& candidate, const PreparedPrompt&,
    runtime::FinalScheduleIntent) {
    if (!candidate.impl_ || candidate.impl_->identity_assessment.assessment_digest !=
                                impl_->revision_.value) return std::nullopt;
    auto copy = std::make_unique<GemmaAdmissionCandidateImpl>(*candidate.impl_);
    AdmissionCandidate<GemmaVariant> admission(std::move(copy));
    return ResourcePlan<GemmaVariant>(std::move(admission), impl_->revision_, false);
}

template <> runtime::PrefillWork Program<GemmaVariant>::shared_capture_split_prefill_work(
    const AdmissionCandidate<GemmaVariant>&, const PreparedPrompt&,
    std::span<const std::uint32_t>) { return {}; }

template <> runtime::ContextTransactionReserveStatus
Program<GemmaVariant>::start_resource_transaction(ResourcePlan<GemmaVariant>&& plan,
                                                   PreparedPrompt&& prompt,
                                                   runtime::CancellationFlagView cancellation) {
    if (cancellation.requested() || impl_->materialization_ || impl_->materialization_terminal_ ||
        plan.revision_ != impl_->revision_ || !plan.admission_.impl_) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    impl_->materialization_.emplace(GemmaProgramImpl::Materialization{
        .candidate = std::move(*plan.admission_.impl_),
        .prompt = PreparedPromptAccess::take(std::move(prompt)),
    });
    return runtime::ContextTransactionReserveStatus::Reserved;
}

template <> ContextTransactionProgress<GemmaVariant>
Program<GemmaVariant>::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    if (!impl_->materialization_ || impl_->materialization_terminal_) {
        throw std::logic_error("Gemma has no context transaction");
    }
    MaterializationResult<GemmaVariant> result;
    auto& transaction = *impl_->materialization_;
    if (transaction.candidate.has_victim) { result.victims.reserve(1); }
    const auto append_uncommitted_victim = [&] {
        if (transaction.candidate.has_victim) {
            result.victims.push_back(MaterializationVictimResult{
                .owner = transaction.candidate.victim_owner,
                .disposition = runtime::VictimDisposition::Retained,
                .pressure_committed = false,
            });
        }
    };
    if (cancellation.requested()) {
        result.status = runtime::ContextTransactionStatus::Aborted;
        if (transaction.candidate.has_source) {
            result.source = MaterializationSourceResult{.mode = runtime::PrivateSourceMode::Retain};
        }
        append_uncommitted_victim();
        impl_->materialization_terminal_ = true;
        return result;
    }
    const std::uint32_t row = transaction.candidate.destination.value;
    try {
        impl_->cache_.activate(static_cast<std::int32_t>(row), impl_->device_.stream);
        std::uint32_t reuse = 0;
        if (transaction.candidate.has_source) {
            ContinuationHandle<GemmaVariant> source = GemmaAccess::make_continuation(
                impl_.get(), transaction.candidate.source_index,
                transaction.candidate.source_generation);
            auto* record = impl_->continuation(source);
            if (!record) { throw std::logic_error("Gemma continuation changed after sealing"); }
            auto selected = gemma4::detail::select_continuation_restore(
                record->state, transaction.prompt.token_ids, impl_->plan_.groups,
                kGemmaSlidingKvGroup, kGemmaGlobalKvGroup);
            if (selected.frontier != transaction.candidate.reuse_frontier) {
                throw std::logic_error("Gemma continuation reuse frontier changed");
            }
            impl_->cache_.import_host(static_cast<std::int32_t>(row), selected.kv,
                                      impl_->device_.stream);
            CUDA_CHECK(cudaStreamSynchronize(impl_->device_.stream));
            reuse = selected.frontier;
            const runtime::PrivateSourceMode mode =
                transaction.candidate.identity_assessment.source_mode;
            if (mode == runtime::PrivateSourceMode::ConsumeToActive) {
                impl_->continuations_[transaction.candidate.source_index].reset();
                result.source = MaterializationSourceResult{.mode = mode};
            } else {
                result.source = MaterializationSourceResult{
                    .mode = mode,
                    .final_summary = impl_->summary(*record),
                };
            }
        }
        auto& lane = impl_->lanes_[row];
        impl_->reset_lane(row);
        auto& published_lane = impl_->lanes_[row];
        published_lane.active = true;
        published_lane.prefilling = true;
        published_lane.publish_continuation = transaction.candidate.summary.publish_continuation;
        published_lane.cursor = reuse;
        published_lane.reserved_tokens = transaction.candidate.summary.prompt_tokens +
            (transaction.candidate.summary.effective_output_tokens == 0
                 ? 0U : transaction.candidate.summary.effective_output_tokens - 1U);
        published_lane.begin = {.prompt_tokens = transaction.candidate.summary.prompt_tokens,
                      .reused_prompt_tokens = reuse,
                      .prefix_reuse_path = transaction.candidate.summary.prefix_reuse_path};
        published_lane.sampling = transaction.candidate.sampling;
        published_lane.prompt = std::move(transaction.prompt);
        published_lane.ledger.assign(published_lane.prompt.token_ids.begin(),
                                     published_lane.prompt.token_ids.begin() + reuse);
        published_lane.token_counts.assign(GemmaTextConfig::vocabulary, 0);
        published_lane.mtp_enabled =
            impl_->plan_.speculative.backend == SpeculativeBackend::Mtp;
        published_lane.speculative.backend = impl_->plan_.speculative.backend;
        published_lane.speculative.enabled = published_lane.mtp_enabled;
        published_lane.speculative.draft_window = impl_->plan_.speculative.draft_tokens;
        published_lane.speculative.accepted_per_position.assign(
            impl_->plan_.speculative.draft_tokens, 0);
        published_lane.epoch = ++impl_->next_generation_;
        if (transaction.candidate.has_victim) {
            ContinuationHandle<GemmaVariant> victim = GemmaAccess::make_continuation(
                impl_.get(), transaction.candidate.victim_index,
                transaction.candidate.victim_generation);
            if (!impl_->continuation(victim)) {
                throw std::logic_error("Gemma pressure victim changed after sealing");
            }
            result.victims.push_back(MaterializationVictimResult{
                .owner = transaction.candidate.victim_owner,
                .disposition = runtime::VictimDisposition::Evicted,
                .pressure_committed = true,
            });
            impl_->continuations_[transaction.candidate.victim_index].reset();
        }
        ++impl_->revision_.value;
        result.status = runtime::ContextTransactionStatus::Published;
        result.published = StartResult<GemmaVariant>{
            .sequence = GemmaAccess::make_sequence(impl_.get(), runtime::LaneId{row},
                                                   published_lane.epoch)};
        impl_->materialization_terminal_ = true;
        return result;
    } catch (...) {
        impl_->cache_.deactivate(static_cast<std::int32_t>(row));
        result.status = runtime::ContextTransactionStatus::Aborted;
        if (transaction.candidate.has_source && !result.source) {
            result.source = MaterializationSourceResult{.mode = runtime::PrivateSourceMode::Retain};
        }
        if (transaction.candidate.has_victim && result.victims.empty()) {
            append_uncommitted_victim();
        }
        impl_->materialization_terminal_ = true;
        return result;
    }
}

template <> void Program<GemmaVariant>::finalize_context_transaction() noexcept {
    impl_->materialization_.reset();
    impl_->materialization_terminal_ = false;
}
template <> bool Program<GemmaVariant>::has_context_transaction() const noexcept {
    return impl_->materialization_.has_value();
}

template <> PrefillProgress<GemmaVariant> Program<GemmaVariant>::advance_prefill(
    SequenceHandle<GemmaVariant> sequence, runtime::ExecutionTiming* failed) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    auto& lane = impl_->require_sequence(sequence);
    if (!lane.prefilling) { throw std::logic_error("Gemma sequence is not prefilling"); }
    const std::uint32_t row = GemmaAccess::lane(sequence).value;
    const std::uint32_t next = std::min(lane.cursor + impl_->plan_.prefill_chunk,
                                        static_cast<std::uint32_t>(lane.prompt.token_ids.size()));
    const bool complete = next == lane.prompt.token_ids.size();
    const auto chunk = std::span<const TokenId>(lane.prompt.token_ids).subspan(
        lane.cursor, next - lane.cursor);
    const std::uint32_t processed = next - lane.cursor;
    TokenId sampled = -1;
    if (!chunk.empty()) {
        const auto started = std::chrono::steady_clock::now();
        sampled = impl_->run_and_sample(lane, row, chunk, complete);
        lane.timings.prefill_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        lane.ledger.insert(lane.ledger.end(), chunk.begin(), chunk.end());
    }
    lane.cursor = next;
    PrefillProgress<GemmaVariant> out;
    out.summary = lane.begin;
    out.processed_prompt_tokens = processed;
    out.complete = complete;
    if (complete) {
        lane.prefilling = false;
        impl_->pending_tokens_.assign(1, sampled);
        impl_->pending_counts_[0] = 1;
        impl_->pending_rows_[0] = sequence;
        impl_->pending_row_count_ = 1;
        impl_->pending_id_ = impl_->next_pending_id_++;
        out.pending.emplace(GemmaAccess::make_pending(
            impl_.get(), impl_->pending_id_, std::span<const SequenceHandle<GemmaVariant>>(
                impl_->pending_rows_.data(), 1), impl_->pending_tokens_,
            std::span<const std::int32_t>(impl_->pending_counts_.data(), 1), 1, {}));
    }
    out.timing = timing.finish();
    return out;
}

template <> PendingBatch<GemmaVariant> Program<GemmaVariant>::decode(
    std::span<const SequenceHandle<GemmaVariant>> sequences,
    std::span<const runtime::RoundBudget> budgets, runtime::ExecutionTiming* failed) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    if (sequences.empty() || sequences.size() != budgets.size() ||
        sequences.size() > kMaximumConcurrency || impl_->pending_id_ != 0) {
        throw std::invalid_argument("Gemma decode batch is invalid");
    }
    const std::uint32_t row_stride = impl_->plan_.speculative.backend == SpeculativeBackend::Mtp
                                         ? impl_->plan_.speculative.draft_tokens + 1U
                                         : 1U;
    impl_->pending_tokens_.assign(sequences.size() * row_stride, 0);
    for (std::size_t row_index = 0; row_index < sequences.size(); ++row_index) {
        impl_->pending_mtp_transactions_[row_index].reset();
        impl_->pending_mtp_drafted_[row_index] = 0;
        impl_->pending_mtp_accepted_[row_index] = 0;
        auto& lane = impl_->require_sequence(sequences[row_index]);
        const auto decode_started = std::chrono::steady_clock::now();
        if (budgets[row_index].generated_tokens_remaining == 0 || !lane.current_token) {
            throw std::logic_error("Gemma decode row has no committed input token");
        }
        if (lane.ready_token) {
            impl_->pending_tokens_[row_index * row_stride] = *lane.ready_token;
            lane.ready_token.reset();
            impl_->pending_counts_[row_index] = 1;
        } else if (lane.mtp_enabled) {
            const std::int32_t physical_row =
                static_cast<std::int32_t>(GemmaAccess::lane(sequences[row_index]).value);
            const std::uint32_t first = impl_->cache_.frontier(physical_row);
            std::vector<TokenId> tentative;
            tentative.reserve(impl_->plan_.speculative.draft_tokens + 1U);
            const gemma4_31b_it::detail::RuntimeMtpSampler sample_target =
                [&](std::span<const float> logits, std::uint32_t output_offset) {
                    const TokenId token = gemma4_31b_it::detail::sample_logits(
                        logits, lane.sampling, lane.token_counts, first + output_offset + 1U,
                        tentative);
                    tentative.push_back(token);
                    return token;
                };
            auto round = gemma4_31b_it::detail::execute_runtime_mtp_round(
                impl_->weights_.target, *impl_->weights_.assistant, impl_->cache_,
                physical_row, first,
                *lane.current_token, impl_->plan_.speculative.draft_tokens,
                impl_->plan_.capacity, budgets[row_index].generated_tokens_remaining,
                lane.target_hidden, lane.assistant_feedback, impl_->activations_,
                impl_->attention_, impl_->device_.stream, sample_target,
                impl_->plan_.use_cuda_graph &&
                        impl_->plan_.speculative.draft_tokens ==
                            gemma4_31b_it::AssistantConfig::production_draft_size &&
                        budgets[row_index].generated_tokens_remaining >=
                            impl_->plan_.speculative.draft_tokens
                    ? impl_->mtp_graphs_[static_cast<std::size_t>(physical_row)].get()
                    : nullptr);
            std::copy(round.output_tokens.begin(), round.output_tokens.end(),
                      impl_->pending_tokens_.begin() +
                          static_cast<std::ptrdiff_t>(row_index * row_stride));
            impl_->pending_counts_[row_index] =
                static_cast<std::int32_t>(round.output_tokens.size());
            impl_->pending_mtp_drafted_[row_index] = round.drafted_tokens;
            impl_->pending_mtp_accepted_[row_index] = round.accepted_drafts;
            impl_->pending_mtp_transactions_[row_index].emplace(
                std::move(round.transaction));
        } else {
            const TokenId input = *lane.current_token;
            impl_->pending_tokens_[row_index * row_stride] =
                impl_->run_and_sample(lane, GemmaAccess::lane(sequences[row_index]).value,
                                      std::span<const TokenId>(&input, 1));
            lane.ledger.push_back(input);
            impl_->pending_counts_[row_index] = 1;
        }
        lane.timings.decode_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_started).count();
        impl_->pending_rows_[row_index] = sequences[row_index];
    }
    impl_->pending_row_count_ = sequences.size();
    impl_->pending_id_ = impl_->next_pending_id_++;
    const runtime::ExecutionTiming measured = timing.finish();
    return GemmaAccess::make_pending(
        impl_.get(), impl_->pending_id_,
        std::span<const SequenceHandle<GemmaVariant>>(impl_->pending_rows_.data(), sequences.size()),
        impl_->pending_tokens_,
        std::span<const std::int32_t>(impl_->pending_counts_.data(), sequences.size()), row_stride,
        measured);
}

template <> CommitResult<GemmaVariant> Program<GemmaVariant>::commit(
    PendingBatch<GemmaVariant>&& pending, std::span<const runtime::CommitDecision> decisions,
    runtime::CommitObservation, runtime::ExecutionTiming*) {
    CommitResult<GemmaVariant> out;
    if (GemmaAccess::owner(pending) != impl_.get() ||
        GemmaAccess::transaction(pending) != impl_->pending_id_ ||
        decisions.size() != impl_->pending_row_count_) return out;
    const auto rows = GemmaAccess::rows(pending);
    const auto tokens = pending.tokens();
    const std::uint32_t row_stride = pending.row_stride();
    out.row_count = rows.size();
    for (std::size_t row = 0; row < rows.size(); ++row) {
        auto& lane = impl_->require_sequence(rows[row]);
        const auto& decision = decisions[row];
        const std::uint32_t lane_index = GemmaAccess::lane(rows[row]).value;
        if (decision.cancelled) {
            impl_->pending_mtp_transactions_[row].reset();
            out.rows[row].timings = lane.timings;
            out.rows[row].speculative = lane.speculative;
            impl_->cache_.deactivate(static_cast<std::int32_t>(lane_index));
            impl_->reset_lane(lane_index);
            out.rows[row].disposition = runtime::CommitDisposition::CancelledReleased;
        } else {
            const std::uint32_t produced = static_cast<std::uint32_t>(impl_->pending_counts_[row]);
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > produced ||
                (!decision.terminal && decision.accepted_tokens != produced)) {
                throw std::logic_error("Gemma round has an invalid committed output prefix");
            }
            const std::size_t token_base = row * row_stride;
            if (impl_->pending_mtp_transactions_[row]) {
                impl_->pending_mtp_transactions_[row]->commit_prefix(decision.accepted_tokens,
                                                                     impl_->device_.stream);
                lane.ledger.push_back(*lane.current_token);
                if (decision.accepted_tokens > 1) {
                    lane.ledger.insert(lane.ledger.end(), tokens.begin() + token_base,
                                       tokens.begin() + token_base + decision.accepted_tokens - 1U);
                }
                ++lane.speculative.rounds;
                lane.speculative.drafted_tokens += impl_->pending_mtp_drafted_[row];
                const std::uint32_t accepted_drafts = std::min(
                    {impl_->pending_mtp_accepted_[row], decision.accepted_tokens,
                     static_cast<std::uint32_t>(lane.speculative.accepted_per_position.size())});
                lane.speculative.accepted_tokens += accepted_drafts;
                for (std::uint32_t position = 0; position < accepted_drafts; ++position) {
                    ++lane.speculative.accepted_per_position[position];
                }
                if (decision.accepted_tokens > accepted_drafts) {
                    ++lane.speculative.fallback_steps;
                }
                impl_->pending_mtp_transactions_[row].reset();
            } else if (decision.accepted_tokens != 1) {
                throw std::logic_error("ordinary Gemma round must commit one output token");
            }
            const TokenId token = tokens[token_base + decision.accepted_tokens - 1U];
            lane.current_token = token;
            for (std::uint32_t accepted = 0; accepted < decision.accepted_tokens; ++accepted) {
                ++lane.token_counts[static_cast<std::size_t>(tokens[token_base + accepted])];
            }
            out.rows[row].speculative = lane.speculative;
            out.rows[row].timings = lane.timings;
            out.rows[row].disposition = decision.terminal
                ? runtime::CommitDisposition::Finishable
                : runtime::CommitDisposition::Active;
        }
    }
    GemmaAccess::consume(pending);
    impl_->pending_id_ = 0;
    impl_->pending_row_count_ = 0;
    return out;
}

template <> DiscardResult<GemmaVariant>
Program<GemmaVariant>::abort_pending(PendingBatch<GemmaVariant>&& pending) noexcept {
    DiscardResult<GemmaVariant> out;
    if (GemmaAccess::owner(pending) == impl_.get() &&
        GemmaAccess::transaction(pending) == impl_->pending_id_) {
        out.status = runtime::ConsumeStatus::Consumed;
        out.row_count = impl_->pending_row_count_;
        for (std::size_t row = 0; row < impl_->pending_row_count_; ++row) {
            impl_->pending_mtp_transactions_[row].reset();
        }
        GemmaAccess::consume(pending);
        impl_->pending_id_ = 0;
        impl_->pending_row_count_ = 0;
    }
    return out;
}

template <> runtime::ExecutionTiming Program<GemmaVariant>::append_forced_tokens(
    std::span<const SequenceHandle<GemmaVariant>> sequences,
    std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
    std::span<const std::optional<std::uint32_t>>, runtime::ExecutionTiming* failed) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed);
    if (row_stride == 0 || row_major_tokens.size() != sequences.size() * row_stride) {
        throw std::invalid_argument("Gemma forced-token batch is invalid");
    }
    for (std::size_t row = 0; row < sequences.size(); ++row) {
        auto& lane = impl_->require_sequence(sequences[row]);
        const auto decode_started = std::chrono::steady_clock::now();
        if (!lane.current_token) { throw std::logic_error("Gemma control row has no input token"); }
        std::vector<TokenId> inputs;
        inputs.reserve(row_stride + 1U);
        inputs.push_back(*lane.current_token);
        const auto controls = row_major_tokens.subspan(row * row_stride, row_stride);
        inputs.insert(inputs.end(), controls.begin(), controls.end());
        lane.ready_token = impl_->run_and_sample(lane, GemmaAccess::lane(sequences[row]).value,
                                                 inputs);
        lane.timings.decode_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_started).count();
        lane.ledger.insert(lane.ledger.end(), inputs.begin(), inputs.end());
        for (TokenId token : controls) ++lane.token_counts[static_cast<std::size_t>(token)];
        lane.current_token = controls.back();
    }
    return timing.finish();
}

template <> FinishResult<GemmaVariant> Program<GemmaVariant>::finish(
    SequenceHandle<GemmaVariant> sequence) noexcept {
    FinishResult<GemmaVariant> out;
    try {
        auto& lane = impl_->require_sequence(sequence);
        const std::uint32_t row = GemmaAccess::lane(sequence).value;
        out.status = runtime::ConsumeStatus::Consumed;
        out.timings = lane.timings;
        out.speculative = lane.speculative;
        if (!lane.publish_continuation) {
            impl_->cache_.deactivate(static_cast<std::int32_t>(row));
            impl_->reset_lane(row);
            out.disposition = runtime::FinishDisposition::Released;
            ++impl_->revision_.value;
            return out;
        }
        HeterogeneousKVHostImage image =
            impl_->cache_.export_host(static_cast<std::int32_t>(row), impl_->device_.stream);
        CUDA_CHECK(cudaStreamSynchronize(impl_->device_.stream));
        auto state = gemma4::detail::make_continuation_state(
            lane.ledger, std::move(image), {}, kGemmaSlidingKvGroup, kGemmaGlobalKvGroup);
        std::size_t bytes = state.endpoint.payload_bytes();
        std::size_t occupied = 0;
        for (const auto& record : impl_->continuations_) if (record) occupied += record->bytes;
        std::size_t slot = impl_->continuations_.size();
        for (std::size_t index = 0; index < impl_->continuations_.size(); ++index) {
            if (!impl_->continuations_[index]) { slot = index; break; }
        }
        if (slot == impl_->continuations_.size() ||
            bytes > impl_->plan_.context_cache.host_kv_capacity_bytes -
                        std::min(occupied, impl_->plan_.context_cache.host_kv_capacity_bytes)) {
            impl_->cache_.deactivate(static_cast<std::int32_t>(row));
            impl_->reset_lane(row);
            out.disposition = runtime::FinishDisposition::Released;
            ++impl_->revision_.value;
            return out;
        }
        GemmaProgramImpl::ContinuationRecord record{
            .state = std::move(state), .generation = ++impl_->next_generation_, .bytes = bytes};
        out.summary = impl_->summary(record);
        impl_->continuations_[slot].emplace(std::move(record));
        out.continuation.emplace(GemmaAccess::make_continuation(
            impl_.get(), static_cast<std::uint32_t>(slot),
            impl_->continuations_[slot]->generation));
        out.disposition = runtime::FinishDisposition::Catalogued;
        impl_->cache_.deactivate(static_cast<std::int32_t>(row));
        impl_->reset_lane(row);
        ++impl_->revision_.value;
    } catch (...) {}
    return out;
}

template <> AbortResult<GemmaVariant> Program<GemmaVariant>::abort(
    SequenceHandle<GemmaVariant> sequence) noexcept {
    AbortResult<GemmaVariant> out;
    try {
        auto& lane = impl_->require_sequence(sequence);
        const std::uint32_t row = GemmaAccess::lane(sequence).value;
        out.timings = lane.timings;
        out.speculative = lane.speculative;
        impl_->cache_.deactivate(static_cast<std::int32_t>(row));
        impl_->reset_lane(row);
        ++impl_->revision_.value;
        out.status = runtime::ConsumeStatus::Consumed;
    } catch (...) {}
    return out;
}

template <> ReleaseResult<GemmaVariant> Program<GemmaVariant>::release_continuation(
    ContinuationHandle<GemmaVariant>&& handle) noexcept {
    ReleaseResult<GemmaVariant> out;
    if (impl_->continuation(handle)) {
        impl_->continuations_[GemmaAccess::index(handle)].reset();
        GemmaAccess::consume(handle);
        ++impl_->revision_.value;
        out.status = runtime::ConsumeStatus::Consumed;
    }
    return out;
}
template <> ReleaseResult<GemmaVariant> Program<GemmaVariant>::release_shared_prefix(
    SharedPrefixHandle<GemmaVariant>&&) noexcept { return {}; }

template <> void Program<GemmaVariant>::fail_all_cleanup() noexcept {
    impl_->materialization_.reset();
    impl_->materialization_terminal_ = false;
    impl_->pending_id_ = 0;
    for (auto& transaction : impl_->pending_mtp_transactions_) { transaction.reset(); }
    for (std::size_t row = 0; row < impl_->lanes_.size(); ++row) {
        if (impl_->lanes_[row].active) impl_->cache_.deactivate(static_cast<std::int32_t>(row));
        impl_->reset_lane(static_cast<std::uint32_t>(row));
    }
}

template <> std::uint32_t Program<GemmaVariant>::continuation_depth(
    const ContinuationHandle<GemmaVariant>& handle) const noexcept {
    const auto* record = impl_->continuation(handle);
    return record ? record->state.execution_frontier : 0;
}
template <> std::string Program<GemmaVariant>::continuation_digest(
    const ContinuationHandle<GemmaVariant>& handle) const {
    const auto* record = impl_->continuation(handle);
    return record ? gemma4_31b_it::detail::digest_hex(record->state.ledger) : std::string{};
}
template <> std::vector<SlotCheckpoint> Program<GemmaVariant>::continuation_checkpoints(
    const ContinuationHandle<GemmaVariant>& handle) const {
    const auto* record = impl_->continuation(handle);
    if (!record) return {};
    return {{.frontier = record->state.execution_frontier,
             .session_digest = gemma4_31b_it::detail::digest_hex(
                 std::span<const TokenId>(record->state.ledger)
                     .first(record->state.execution_frontier))}};
}
template <> RetainedSessionSnapshot Program<GemmaVariant>::save_continuation(
    const ContinuationHandle<GemmaVariant>& handle, std::string_view model_binding) {
    const auto* record = impl_->continuation(handle);
    if (!record) throw std::invalid_argument("Gemma continuation handle is invalid");
    RetainedSessionSnapshot out;
    out.bytes = gemma4::detail::encode_continuation(
        record->state, model_binding, gemma4_31b_it::Package::artifact_compatibility_fingerprint,
        impl_->plan_.groups, kGemmaSlidingKvGroup, kGemmaGlobalKvGroup);
    out.tokens = record->state.execution_frontier;
    out.session_digest = gemma4_31b_it::detail::digest_hex(record->state.ledger);
    return out;
}
template <> ContinuationHandle<GemmaVariant> Program<GemmaVariant>::restore_continuation(
    std::span<const std::uint8_t> snapshot, std::string_view model_binding) {
    std::size_t slot = impl_->continuations_.size();
    for (std::size_t index = 0; index < impl_->continuations_.size(); ++index)
        if (!impl_->continuations_[index]) { slot = index; break; }
    if (slot == impl_->continuations_.size()) {
        throw std::runtime_error("Gemma continuation catalog has no free physical slot");
    }
    auto state = gemma4::detail::decode_continuation(
        snapshot, model_binding, gemma4_31b_it::Package::artifact_compatibility_fingerprint,
        impl_->plan_.groups, GemmaTextConfig::vocabulary, kGemmaSlidingKvGroup,
        kGemmaGlobalKvGroup);
    const std::size_t bytes = state.endpoint.payload_bytes();
    impl_->continuations_[slot].emplace(GemmaProgramImpl::ContinuationRecord{
        .state = std::move(state), .generation = ++impl_->next_generation_, .bytes = bytes});
    ++impl_->revision_.value;
    return GemmaAccess::make_continuation(impl_.get(), static_cast<std::uint32_t>(slot),
                                          impl_->continuations_[slot]->generation);
}
template <> ContinuationSummary Program<GemmaVariant>::continuation_summary(
    const ContinuationHandle<GemmaVariant>& handle) const {
    const auto* record = impl_->continuation(handle);
    if (!record) throw std::invalid_argument("Gemma continuation handle is invalid");
    return impl_->summary(*record);
}
template <> SessionSnapshotTraffic Program<GemmaVariant>::session_snapshot_traffic() const noexcept {
    return impl_->snapshot_traffic_;
}

template <> runtime::ProgramResourceRevision Program<GemmaVariant>::resource_revision() const noexcept {
    return impl_->revision_;
}
template <> PhysicalUsageSnapshot Program<GemmaVariant>::physical_usage() const noexcept {
    PhysicalUsageSnapshot out;
    out.resource_revision = impl_->revision_;
    for (const auto& lane : impl_->lanes_) if (lane.active) ++out.device_state_slots;
    for (const auto& record : impl_->continuations_) if (record) {
        ++out.host_state_slots; out.host_kv_bytes += record->bytes;
    }
    return out;
}
template <> MemorySummary Program<GemmaVariant>::memory_summary() const noexcept {
    MemorySummary out;
    out.device = impl_->plan_.device;
    out.max_context = impl_->plan_.capacity;
    out.kv_capacity = impl_->plan_.kv_capacity;
    out.kv_cache = KvCacheStorage::RK4V4E8;
    if (impl_->weights_.weights_arena != nullptr) {
        out.weights = {.capacity_bytes = impl_->weights_.weights_arena->capacity(),
                       .used_bytes = impl_->weights_.weights_arena->used(),
                       .peak_used_bytes = impl_->weights_.weights_arena->peak_used()};
    }
    out.sequence = {.capacity_bytes = impl_->cache_backing_.bytes,
                    .used_bytes = impl_->cache_backing_.bytes,
                    .peak_used_bytes = impl_->cache_backing_.bytes};
    out.workspace = {.capacity_bytes = impl_->plan_.workspace.capacity,
                     .used_bytes = impl_->plan_.workspace.capacity,
                     .peak_used_bytes = impl_->plan_.workspace.capacity};
    out.workspace_logical_peak_bytes = impl_->plan_.workspace.capacity;
    for (const auto& graph : impl_->ordinary_graphs_) {
        out.cuda_graph_allowance_bytes += graph->device_bytes();
    }
    for (const auto& graph : impl_->mtp_graphs_) {
        out.cuda_graph_allowance_bytes += graph->device_bytes();
    }
    out.kv_payload_bytes = impl_->plan_.cache_layout.payload_bytes();
    out.text_kv_bytes = out.kv_payload_bytes;
    out.host_state_capacity_slots = static_cast<std::uint32_t>(impl_->continuations_.size());
    out.host_kv_capacity_bytes = impl_->plan_.context_cache.host_kv_capacity_bytes;
    for (const auto& record : impl_->continuations_) if (record) {
        ++out.host_state_occupied_slots; out.host_kv_occupied_bytes += record->bytes;
    }
    return out;
}
template <> void Program<GemmaVariant>::reset_memory_peaks() noexcept {
    if (impl_->weights_.weights_arena != nullptr) { impl_->weights_.weights_arena->reset_peak(); }
}

template <> std::vector<float> Program<GemmaVariant>::causal_score(
    PreparedPrompt&& prompt, std::uint32_t first_target) {
    auto data = PreparedPromptAccess::take(std::move(prompt));
    if (first_target == 0 || first_target >= data.token_ids.size()) {
        throw std::invalid_argument("Gemma causal score target range is invalid");
    }
    if (impl_->lanes_[0].active) throw std::logic_error("Gemma scoring lane is busy");
    impl_->cache_.activate(0, impl_->device_.stream);
    std::vector<float> scores;
    scores.reserve(data.token_ids.size() - first_target);
    try {
        for (std::uint32_t input = 0; input + 1U < data.token_ids.size(); ++input) {
            const TokenId token = data.token_ids[input];
            const auto logits = gemma4_31b_it::detail::execute_runtime_chunk(
                impl_->weights_.target, impl_->cache_, 0, input,
                std::span<const TokenId>(&token, 1),
                impl_->plan_.capacity, impl_->activations_, impl_->attention_,
                impl_->device_.stream, true);
            if (input + 1U < first_target) continue;
            const float maximum = *std::max_element(logits.begin(), logits.end());
            double denominator = 0.0;
            for (float value : logits) denominator += std::exp(static_cast<double>(value - maximum));
            const TokenId target = data.token_ids[input + 1U];
            scores.push_back(logits[static_cast<std::size_t>(target)] - maximum -
                             static_cast<float>(std::log(denominator)));
        }
        impl_->cache_.deactivate(0);
    } catch (...) { impl_->cache_.deactivate(0); throw; }
    return scores;
}

// Gemma retains only private host endpoints in Phase 15. It exposes bounded private eviction to
// the common materialization planner, but no shared-prefix capture offers are minted.
template <> CaptureAssessment Program<GemmaVariant>::inspect_capture(
    const CaptureOffer<GemmaVariant>&, const SharedPrefixHandle<GemmaVariant>*,
    const SharedPrefixHandle<GemmaVariant>*, std::optional<runtime::CheckpointRef>, bool) const {
    throw std::logic_error("Gemma does not publish active capture offers");
}
template <> std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program<GemmaVariant>::checkpoint_recovery_work(const ContinuationHandle<GemmaVariant>&,
                                                runtime::CheckpointRef) const { return {}; }
template <> std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program<GemmaVariant>::checkpoint_recovery_work(const SharedPrefixHandle<GemmaVariant>&,
                                                runtime::CheckpointRef) const { return {}; }
template <> bool Program<GemmaVariant>::shared_capture_matches(
    const CaptureOffer<GemmaVariant>&, const SharedPrefixHandle<GemmaVariant>&) const { return false; }
template <> void Program<GemmaVariant>::skip_capture(CaptureOffer<GemmaVariant>&& offer) {
    GemmaAccess::consume(offer);
}
template <> runtime::ContextTransactionReserveStatus Program<GemmaVariant>::reserve_active_capture(
    CaptureOffer<GemmaVariant>&&, const SharedPrefixHandle<GemmaVariant>*,
    const SharedPrefixHandle<GemmaVariant>*, std::optional<runtime::CheckpointRef>, bool,
    runtime::CancellationFlagView) { return runtime::ContextTransactionReserveStatus::Aborted; }
template <> runtime::ContextTransactionReserveStatus
Program<GemmaVariant>::reserve_active_capture_with_pressure(
    CaptureOffer<GemmaVariant>&&, const SharedPrefixHandle<GemmaVariant>*,
    const SharedPrefixHandle<GemmaVariant>*, std::optional<runtime::CheckpointRef>, bool,
    CapturePressurePlan<GemmaVariant>&&, runtime::CancellationFlagView) {
    return runtime::ContextTransactionReserveStatus::Aborted;
}

template <> std::optional<PersistentBackfillProof<GemmaVariant>>
Program<GemmaVariant>::prove_persistent_backfill(
    const RequestBasePlan<GemmaVariant>&, const ResourcePlan<GemmaVariant>&,
    std::span<const SequenceHandle<GemmaVariant>>) const { return std::nullopt; }

template <> CapturePressurePlanningSession<GemmaVariant>
Program<GemmaVariant>::begin_capture_pressure_planning(
    const CaptureAssessment&, std::span<const ContinuationHandle<GemmaVariant>* const>,
    std::span<const runtime::PlanningOwnerId>,
    std::span<const SharedPrefixHandle<GemmaVariant>* const>,
    std::span<const runtime::PlanningOwnerId>) {
    throw std::logic_error("Gemma does not publish active capture offers");
}

template <> PressurePlanningSession<GemmaVariant>::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl<GemmaVariant>> impl) noexcept
    : impl_(std::move(impl)) {}
template <> PressurePlanningSession<GemmaVariant>::PressurePlanningSession(PressurePlanningSession&&) noexcept = default;
template <> PressurePlanningSession<GemmaVariant>& PressurePlanningSession<GemmaVariant>::operator=(PressurePlanningSession&&) noexcept = default;
template <> PressurePlanningSession<GemmaVariant>::~PressurePlanningSession() = default;

template <> PressurePlanningSession<GemmaVariant> Program<GemmaVariant>::begin_pressure_planning(
    std::span<const AdmissionCandidate<GemmaVariant>* const> candidates,
    std::span<const runtime::PlanningCandidateId> candidate_ids,
    std::span<const ContinuationHandle<GemmaVariant>* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle<GemmaVariant>* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids) {
    return PressurePlanningSession<GemmaVariant>(
        std::make_unique<detail::PressurePlanningSessionImpl<GemmaVariant>>(
            *impl_, candidates, candidate_ids, private_owners, private_owner_ids,
            shared_owners, shared_owner_ids));
}

template <> PressureTargetHandle PressurePlanningSession<GemmaVariant>::identity_target(
    runtime::PlanningCandidateId candidate) const {
    return impl_->identity_target(candidate);
}
template <> PressureTargetHandle PressurePlanningSession<GemmaVariant>::root_maximal_target(
    runtime::PlanningCandidateId candidate) {
    return impl_->root_maximal_target(candidate);
}
template <> PressureTargetHandle PressurePlanningSession<GemmaVariant>::maximal_target(
    runtime::PlanningCandidateId candidate) {
    return impl_->maximal_target(candidate);
}
template <> PressureConstructionCursor PressurePlanningSession<GemmaVariant>::begin_construction(
    PressureTargetHandle target, bool restore) {
    return impl_->begin_construction(target, restore);
}
template <> runtime::PressureConstructionStep
PressurePlanningSession<GemmaVariant>::next_construction_option(
    PressureConstructionCursor& cursor) {
    return impl_->next_construction_option(cursor);
}
template <> void PressurePlanningSession<GemmaVariant>::choose_construction(
    PressureConstructionCursor& cursor, runtime::PressureConstructionOptionId option) {
    impl_->choose_construction(cursor, option);
}
template <> std::optional<PressureTargetHandle>
PressurePlanningSession<GemmaVariant>::construction_target(const PressureConstructionCursor& cursor) {
    return impl_->construction_target(cursor);
}
template <> runtime::PressureTargetGuidance PressurePlanningSession<GemmaVariant>::guidance(
    PressureTargetHandle target) {
    return impl_->guidance(target);
}
template <> AssessedPressureTarget<GemmaVariant> PressurePlanningSession<GemmaVariant>::assess(
    PressureTargetHandle target) {
    return impl_->assess(target);
}
template <> PreparedPressureExpansion<GemmaVariant>
PressurePlanningSession<GemmaVariant>::prepare_expansion(PressureTargetHandle parent,
                                                          std::uint32_t maximum_owners) {
    return impl_->prepare_expansion(parent, maximum_owners);
}
template <> PressureExpansionView PressurePlanningSession<GemmaVariant>::commit_expansion(
    PreparedPressureExpansion<GemmaVariant>&& prepared) {
    return impl_->commit_expansion(std::move(prepared));
}
template <> void PressurePlanningSession<GemmaVariant>::discard_expansion(
    PreparedPressureExpansion<GemmaVariant>&& prepared) noexcept {
    impl_->discard_expansion(std::move(prepared));
}
template <> runtime::PrefillWork
PressurePlanningSession<GemmaVariant>::shared_capture_split_prefill_work(
    const AssessedPressureTarget<GemmaVariant>& assessed, const PreparedPrompt& prompt,
    std::span<const std::uint32_t> frontiers) const {
    return impl_->shared_capture_split_prefill_work(
        assessed, PreparedPromptAccess::view(prompt), frontiers);
}
template <> std::optional<ResourcePlan<GemmaVariant>>
PressurePlanningSession<GemmaVariant>::seal(AssessedPressureTarget<GemmaVariant>&& assessed,
                                             const PreparedPrompt& prompt,
                                             runtime::FinalScheduleIntent intent) {
    auto sealed = impl_->seal(std::move(assessed), PreparedPromptAccess::view(prompt), intent);
    if (!sealed) { return std::nullopt; }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan<GemmaVariant>(std::move(*sealed), impl_->resource_revision,
                                      needs_transfer);
}
template <> std::optional<CapturePressurePlan<GemmaVariant>>
PressurePlanningSession<GemmaVariant>::seal_capture(
    AssessedPressureTarget<GemmaVariant>&&) {
    return std::nullopt;
}

template <> CapturePressurePlanningSession<GemmaVariant>::CapturePressurePlanningSession(
    CapturePressurePlanningSession&&) noexcept = default;
template <> CapturePressurePlanningSession<GemmaVariant>&
CapturePressurePlanningSession<GemmaVariant>::operator=(CapturePressurePlanningSession&&) noexcept = default;
template <> CapturePressurePlanningSession<GemmaVariant>::~CapturePressurePlanningSession() = default;

#define GEMMA_CAPTURE_PRESSURE_THROW(name, ret, args) \
    template <> ret CapturePressurePlanningSession<GemmaVariant>::name args { \
        throw std::logic_error("Gemma capture pressure planning is unreachable"); \
    }
GEMMA_CAPTURE_PRESSURE_THROW(identity_target, PressureTargetHandle, () const)
GEMMA_CAPTURE_PRESSURE_THROW(guidance, runtime::PressureTargetGuidance,
                             (PressureTargetHandle))
GEMMA_CAPTURE_PRESSURE_THROW(assess, AssessedPressureTarget<GemmaVariant>,
                             (PressureTargetHandle))
GEMMA_CAPTURE_PRESSURE_THROW(prepare_expansion, PreparedPressureExpansion<GemmaVariant>,
                             (PressureTargetHandle))
GEMMA_CAPTURE_PRESSURE_THROW(commit_expansion, PressureExpansionView,
                             (PreparedPressureExpansion<GemmaVariant>&&))
GEMMA_CAPTURE_PRESSURE_THROW(seal, std::optional<CapturePressurePlan<GemmaVariant>>,
                             (AssessedPressureTarget<GemmaVariant>&&))
#undef GEMMA_CAPTURE_PRESSURE_THROW
template <> void CapturePressurePlanningSession<GemmaVariant>::discard_expansion(
    PreparedPressureExpansion<GemmaVariant>&&) noexcept {}

template <> SequencePlanner<GemmaVariant> make_sequence_planner<GemmaVariant>(
    DeviceContext& device, const EngineOptions& options, GemmaVariant::WeightsProfile profile) {
    return SequencePlanner<GemmaVariant>(gemma4_31b_it::detail::make_planner(device, options, profile));
}

template <> std::unique_ptr<Program<GemmaVariant>> create_program<GemmaVariant>(
    const GemmaVariant::ModelView& model, GemmaVariant::WeightsProfile profile,
    SequencePlan<GemmaVariant>&& plan, DeviceContext& device,
    const StartupObserver& observer) {
    if (!plan.impl_ || plan.impl_->weights_profile != profile) {
        throw std::invalid_argument("Gemma sequence plan does not match loaded weights");
    }
    auto impl = std::make_unique<GemmaProgramImpl>(model, *plan.impl_, device, observer);
    plan.impl_.reset();
    return std::unique_ptr<Program<GemmaVariant>>(new Program<GemmaVariant>(std::move(impl)));
}

} // namespace ninfer::targets::qwen3_6
