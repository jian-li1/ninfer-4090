#include "ninfer/engine.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

class RecordingSink final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart value) override {
        ++starts;
        prompt_tokens = value.prompt.prompt_tokens;
    }
    void progress(ninfer::PromptProgress value) override {
        ++progress_updates;
        processed_prompt_tokens = value.processed_prompt_tokens;
    }
    void timing(ninfer::GenerationTimingObservation) override { ++timing_updates; }
    void publish(ninfer::OutputDelta delta) override {
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            reasoning += delta.text;
        } else {
            content += delta.text;
        }
    }

    std::uint32_t starts = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t progress_updates = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint32_t timing_updates = 0;
    std::string reasoning;
    std::string content;
};

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 1024;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk = 64;
    options.kv_cache = ninfer::KvCacheStorage::RK4V4E8;
    options.speculative.backend = ninfer::SpeculativeBackend::None;
    options.use_cuda_graph = true;
    options.max_concurrency = 2;
    options.context_cache.host_kv_capacity_bytes = 256ULL * 1024ULL * 1024ULL;
    options.context_cache.max_private_continuations = 2;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature = 0.0F;
    options.execution.allow_prefix_reuse = reuse;
    options.stop.include_model_defaults = false;
    return options;
}

ninfer::PromptInput simple_chat() {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.messages.push_back(ninfer::ChatMessage{
        .role = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                     .text = "Reply with one short greeting."}},
    });
    return input;
}

int run_ordinary(const char* artifact, std::vector<ninfer::TokenId>& reference_tokens,
                 std::vector<ninfer::TokenId>& sampled_reference) {
    ninfer::Engine engine(engine_options(artifact));
    const auto load = engine.load_summary();
    const auto memory = engine.memory_summary();
    const auto defaults = engine.sampling_defaults();
    const auto capabilities = engine.prompt_capabilities();
    if (load.target != "gemma4_31b_it" || load.model_id != "gemma4-31b-it" ||
        load.weights_id != "groupwise-int" || load.host_to_device_bytes == 0 ||
        memory.weights.used_bytes == 0 || memory.sequence.capacity_bytes == 0 ||
        memory.workspace.capacity_bytes == 0 || memory.max_context != 1024 ||
        memory.kv_capacity < 1024 || memory.kv_cache != ninfer::KvCacheStorage::RK4V4E8 ||
        defaults.thinking.temperature != 1.0F || defaults.thinking.top_p != 0.95F ||
        defaults.thinking.top_k != 64 || defaults.maximum_top_k != 64 ||
        !capabilities.enable_thinking || !engine.healthy()) {
        std::cerr << "Gemma Engine startup contract is incomplete\n";
        return 1;
    }

    auto prepared = engine.prepare(simple_chat());
    const std::vector<ninfer::TokenId> prompt(prepared.token_ids().begin(),
                                               prepared.token_ids().end());
    RecordingSink sink;
    auto handle = engine.submit(std::move(prepared), greedy(5),
                                ninfer::OutputConsumerMode::Streaming,
                                {.phase_timings = true,
                                 .live_timings = true,
                                 .prompt_progress = true});
    const auto first = handle.wait(&sink);
    if (first.generated_token_ids.size() != 5 ||
        first.finish_reason != ninfer::FinishReason::OutputLimit || first.slot < 0 ||
        first.session_digest.empty() || sink.starts != 1 ||
        sink.prompt_tokens != first.prompt.prompt_tokens || sink.progress_updates == 0 ||
        sink.processed_prompt_tokens != first.prompt.prompt_tokens || sink.timing_updates == 0 ||
        sink.content != first.content || sink.reasoning != first.reasoning) {
        std::cerr << "Gemma generation did not complete through the streaming Engine path: "
                  << "tokens=" << first.generated_token_ids.size()
                  << " finish=" << static_cast<int>(first.finish_reason)
                  << " slot=" << first.slot << " digest=" << first.session_digest
                  << " starts=" << sink.starts << " prompt=" << sink.prompt_tokens << '/'
                  << first.prompt.prompt_tokens << " progress=" << sink.progress_updates << ':'
                  << sink.processed_prompt_tokens << " timing=" << sink.timing_updates
                  << " content_equal=" << (sink.content == first.content)
                  << " reasoning_equal=" << (sink.reasoning == first.reasoning) << '\n';
        return 1;
    }
    reference_tokens = first.generated_token_ids;
    if (engine.memory_summary().cuda_graph_allowance_bytes == 0) {
        std::cerr << "Gemma public ordinary route did not capture its decode graph\n";
        return 1;
    }

    const std::uint32_t slot = static_cast<std::uint32_t>(first.slot);
    const std::filesystem::path snapshot =
        std::filesystem::temp_directory_path() / "ninfer_gemma4_engine_real.slot";
    std::error_code ignored;
    std::filesystem::remove(snapshot, ignored);
    const auto saved = engine.save_slot(slot, snapshot.string(), first.session_digest);
    if (saved.tokens == 0 || saved.bytes == 0 || saved.session_digest != first.session_digest ||
        engine.erase_slot(slot, first.session_digest) != saved.tokens) {
        std::cerr << "Gemma public slot save/erase contract failed\n";
        std::filesystem::remove(snapshot, ignored);
        return 1;
    }
    const auto restored = engine.restore_slot(slot, snapshot.string());
    std::filesystem::remove(snapshot, ignored);
    if (restored.tokens != saved.tokens || restored.session_digest != saved.session_digest) {
        std::cerr << "Gemma public slot restore contract failed\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(107);
    const auto reused = engine.generate(engine.prepare_tokens(std::move(continuation)), greedy(1));
    if (reused.generated_token_ids.size() != 1 || reused.reused_prompt_tokens == 0) {
        std::cerr << "Gemma restored continuation was not reused through Engine\n";
        return 1;
    }

    const auto before_pressure = engine.runtime_stats();
    const auto filler_one = engine.generate(engine.prepare(simple_chat()), greedy(1));
    const auto filler_two = engine.generate(engine.prepare(simple_chat()), greedy(1));
    const auto after_pressure = engine.runtime_stats();
    if (filler_one.generated_token_ids.size() != 1 ||
        filler_two.generated_token_ids.size() != 1 ||
        after_pressure.pressure_private_owners_evicted <=
            before_pressure.pressure_private_owners_evicted ||
        after_pressure.pressure_checkpoints_dropped <=
            before_pressure.pressure_checkpoints_dropped ||
        !engine.healthy()) {
        std::cerr << "Gemma bounded host continuation replacement failed: evictions="
                  << before_pressure.pressure_private_owners_evicted << '/'
                  << after_pressure.pressure_private_owners_evicted << " drops="
                  << before_pressure.pressure_checkpoints_dropped << '/'
                  << after_pressure.pressure_checkpoints_dropped << '\n';
        return 1;
    }

    ninfer::PromptInput media = simple_chat();
    media.messages.front().parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Media,
        .media = {.kind = ninfer::MediaKind::Image, .media_type = "image/png"},
    });
    bool rejected_media = false;
    try {
        (void)engine.prepare(std::move(media));
    } catch (const ninfer::RequestError& error) {
        rejected_media = error.kind() == ninfer::RequestErrorKind::InvalidMedia &&
                         std::string_view(error.what()).find("text-only") != std::string_view::npos;
    }
    if (!rejected_media || !engine.healthy()) {
        std::cerr << "Gemma text-only capability rejection is not product-safe\n";
        return 1;
    }
    ninfer::RequestOptions sampled;
    sampled.execution.requested_output_tokens = 5;
    sampled.execution.allow_prefix_reuse = false;
    sampled.execution.sampling.seed = 0x47454d34ULL;
    sampled.stop.include_model_defaults = false;
    sampled_reference =
        engine.generate(engine.prepare(simple_chat()), std::move(sampled)).generated_token_ids;
    auto concurrent_one = engine.submit(engine.prepare(simple_chat()), greedy(2, false));
    auto concurrent_two = engine.submit(engine.prepare(simple_chat()), greedy(2, false));
    const auto one = concurrent_one.wait();
    const auto two = concurrent_two.wait();
    if (one.generated_token_ids.size() != 2 || two.generated_token_ids.size() != 2 ||
        one.generated_token_ids != two.generated_token_ids || !engine.healthy()) {
        std::cerr << "Gemma compact two-request Engine round is not lane-safe\n";
        return 1;
    }
    const auto cancelled = engine.generate(
        engine.prepare(simple_chat()), greedy(32, false), nullptr,
        ninfer::CancellationView([] { return true; }));
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled || !engine.healthy()) {
        std::cerr << "Gemma cancellation did not release its Engine resources\n";
        return 1;
    }
    return 0;
}

int run_mtp(const char* artifact, const std::vector<ninfer::TokenId>& reference_tokens,
            const std::vector<ninfer::TokenId>& sampled_reference) {
    ninfer::EngineOptions options = engine_options(artifact);
    options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 1;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.context_cache.enabled = false;
    ninfer::Engine engine(std::move(options));
    const auto result = engine.generate(engine.prepare(simple_chat()), greedy(5, false));
    if (result.generated_token_ids != reference_tokens ||
        result.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
        !result.speculative.enabled || result.speculative.draft_window != 1 ||
        result.speculative.rounds == 0 || result.speculative.drafted_tokens == 0 ||
        result.speculative.accepted_per_position.size() != 1) {
        std::cerr << "Gemma public MTP route changed greedy output or omitted speculation stats: "
                  << "outputs=" << result.generated_token_ids.size()
                  << " rounds=" << result.speculative.rounds
                  << " drafted=" << result.speculative.drafted_tokens
                  << " accepted=" << result.speculative.accepted_tokens << '\n';
        return 1;
    }
    ninfer::RequestOptions sampled;
    sampled.execution.requested_output_tokens = 5;
    sampled.execution.allow_prefix_reuse = false;
    sampled.execution.sampling.seed = 0x47454d34ULL;
    sampled.stop.include_model_defaults = false;
    const auto sampled_result =
        engine.generate(engine.prepare(simple_chat()), std::move(sampled));
    if (sampled_result.generated_token_ids != sampled_reference ||
        !sampled_result.speculative.enabled || sampled_result.speculative.rounds == 0) {
        std::cerr << "Gemma public MTP route changed deterministic stochastic sampling\n";
        return 1;
    }
    if (reference_tokens.size() < 2 || reference_tokens[0] == reference_tokens[1]) {
        std::cerr << "Gemma MTP stop fixture lacks distinct leading tokens\n";
        return 1;
    }
    auto stopped_options = greedy(5, false);
    stopped_options.stop.token_ids.push_back(reference_tokens[1]);
    const auto stopped = engine.generate(engine.prepare(simple_chat()), stopped_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.size() != 2 ||
        !std::equal(stopped.generated_token_ids.begin(), stopped.generated_token_ids.end(),
                    reference_tokens.begin())) {
        std::cerr << "Gemma public MTP transaction did not stop at its second token\n";
        return 1;
    }
    return 0;
}

int run_scoring(const char* artifact) {
    ninfer::EngineOptions options = engine_options(artifact);
    options.purpose = ninfer::EnginePurpose::CausalScoring;
    options.max_concurrency = 1;
    options.context_cache.enabled = false;
    options.speculative.backend = ninfer::SpeculativeBackend::None;
    ninfer::Engine engine(std::move(options));
    std::vector<ninfer::TokenId> tokens = engine.tokenize_text("one two three four five");
    if (tokens.size() < 4) {
        std::cerr << "Gemma scoring fixture tokenized too narrowly\n";
        return 1;
    }
    if (tokens.size() > 8) tokens.resize(8);
    const auto all = engine.score_tokens(tokens, 1);
    const auto suffix = engine.score_tokens(tokens, 3);
    const auto repeated = engine.score_tokens(tokens, 3);
    if (all.size() != tokens.size() - 1U || suffix.size() != tokens.size() - 3U ||
        suffix != repeated) {
        std::cerr << "Gemma public causal-scoring shape or reset semantics failed\n";
        return 1;
    }
    for (std::size_t index = 0; index < suffix.size(); ++index) {
        if (!std::isfinite(suffix[index]) ||
            std::abs(suffix[index] - all[index + 2U]) > 0.25F) {
            std::cerr << "Gemma public causal-scoring overlap is unstable\n";
            return 1;
        }
    }
    return 0;
}

int run_wide_prefill_equivalence(const char* artifact) {
    std::vector<ninfer::TokenId> prompt;
    std::vector<ninfer::TokenId> reference;
    {
        auto options = engine_options(artifact);
        options.max_concurrency = 1;
        options.context_cache = {.enabled = false};
        options.prefill_chunk = 64;
        ninfer::Engine engine(std::move(options));
        std::string text;
        for (int repeat = 0; repeat < 160; ++repeat) {
            text += "The quick brown fox records a deterministic cache boundary. ";
        }
        prompt = engine.tokenize_text(text);
        if (prompt.size() < 256) {
            std::cerr << "Gemma wide-prefill fixture tokenized too narrowly\n";
            return 1;
        }
        if (prompt.size() > 768) prompt.resize(768);
        reference = engine.generate(engine.prepare_tokens(prompt), greedy(3, false))
                        .generated_token_ids;
    }
    {
        auto options = engine_options(artifact);
        options.max_concurrency = 1;
        options.context_cache = {.enabled = false};
        options.prefill_chunk = 1024;
        ninfer::Engine engine(std::move(options));
        const auto wide = engine.generate(engine.prepare_tokens(prompt), greedy(3, false));
        if (wide.generated_token_ids != reference || wide.generated_token_ids.size() != 3 ||
            !engine.healthy()) {
            std::cerr << "Gemma T=1024 model chunk changed T=64 greedy output\n";
            return 1;
        }
    }
    return 0;
}

int run_full_context_mtp_admission(const char* artifact) {
    auto options = engine_options(artifact);
    options.max_context = 262144;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(262144);
    options.prefill_chunk = 1024;
    options.max_concurrency = 1;
    options.context_cache = {.enabled = false};
    options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 1;
    ninfer::Engine engine(std::move(options));
    const auto memory = engine.memory_summary();
    if (!engine.healthy() || memory.max_context != 262144 || memory.kv_capacity != 262144 ||
        memory.workspace.capacity_bytes >= 512ULL * 1024ULL * 1024ULL ||
        memory.available_after_startup_bytes < 512ULL * 1024ULL * 1024ULL) {
        std::cerr << "Gemma full-context MTP profile lost its bounded workspace/headroom"
                  << " workspace=" << memory.workspace.capacity_bytes
                  << " free=" << memory.available_after_startup_bytes << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_GEMMA4_ARTIFACT is not set\n";
        return 77;
    }
    try {
        bool rejected_wrong_codec = false;
        try {
            auto invalid = engine_options(artifact);
            invalid.kv_cache = ninfer::KvCacheStorage::BFloat16;
            ninfer::Engine engine(std::move(invalid));
        } catch (const std::invalid_argument& error) {
            rejected_wrong_codec =
                std::string_view(error.what()).find("rk4v4-e8") != std::string_view::npos;
        }
        if (!rejected_wrong_codec) {
            std::cerr << "Gemma startup accepted a non-heterogeneous KV codec\n";
            return 1;
        }
        std::vector<ninfer::TokenId> reference_tokens;
        std::vector<ninfer::TokenId> sampled_reference;
        if (const int result = run_ordinary(artifact, reference_tokens, sampled_reference);
            result != 0) {
            return result;
        }
        if (const int result = run_mtp(artifact, reference_tokens, sampled_reference);
            result != 0) {
            return result;
        }
        if (const int result = run_scoring(artifact); result != 0) return result;
        if (const int result = run_wide_prefill_equivalence(artifact); result != 0) return result;
        if (const int result = run_full_context_mtp_admission(artifact); result != 0) return result;
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
