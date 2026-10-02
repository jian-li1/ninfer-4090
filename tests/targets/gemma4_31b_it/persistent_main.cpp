#include "targets/gemma4_31b_it/impl/runtime/persistent_model.h"
#include "targets/gemma4_31b_it/impl/runtime/reference_model.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA4_ARTIFACT");
    if (artifact == nullptr) return 77;
    try {
        using namespace ninfer::targets::gemma4_31b_it::detail;
        const std::vector<std::int32_t> exact_ids(5, 2);
        const auto reference = run_reference_prefix(artifact, exact_ids);
        PersistentRunOptions options;
        options.maximum_context = 1024;
        options.input_tokens = exact_ids;
        options.chunk_tokens = static_cast<std::uint32_t>(exact_ids.size());
        options.input_token = 2;
        options.run_deep_decode = false;
        const auto short_persistent = run_persistent_target(artifact, options);
        if (short_persistent.greedy_token != reference.greedy_token ||
            short_persistent.final_frontier != exact_ids.size() ||
            short_persistent.kv_payload_bytes == 0 || short_persistent.workspace_bytes == 0) {
            std::cerr << "short persistent Gemma target changed its reference result\n";
            return 1;
        }

        const std::vector<std::int32_t> varied_ids{
            9259, 236764, 147224, 236743, 236812, 236888};
        options.input_tokens = varied_ids;
        options.chunk_tokens = static_cast<std::uint32_t>(varied_ids.size());
        options.generation_tokens = 3;
        const auto whole_chunk = run_persistent_target(artifact, options);
        options.chunk_tokens = 3;
        options.use_cuda_graph = true;
        options.qualify_graph_transactions = true;
        const auto half_chunk = run_persistent_target(artifact, options);
        if (whole_chunk.generated_tokens != half_chunk.generated_tokens ||
            whole_chunk.generated_tokens.size() != options.generation_tokens ||
            whole_chunk.final_frontier != varied_ids.size() + options.generation_tokens - 1 ||
            half_chunk.final_frontier != whole_chunk.final_frontier ||
            half_chunk.graph_capture_count != 1 || half_chunk.graph_replay_count != 3 ||
            !half_chunk.graph_transaction_checks_passed) {
            std::cerr << "graph Gemma decode changed normal execution or transaction state\n";
            return 1;
        }

        options.generation_tokens = 2;
        options.use_cuda_graph = false;
        options.qualify_graph_transactions = false;
        options.input_tokens.resize(65);
        for (std::size_t index = 0; index < options.input_tokens.size(); ++index) {
            options.input_tokens[index] = varied_ids[index % varied_ids.size()];
        }
        options.chunk_tokens = 64;
        const auto page_chunked = run_persistent_target(artifact, options);
        options.chunk_tokens = 32;
        options.use_cuda_graph = true;
        const auto half_page_chunked = run_persistent_target(artifact, options);
        if (page_chunked.generated_tokens != half_page_chunked.generated_tokens ||
            page_chunked.final_frontier != 66 || half_page_chunked.final_frontier != 66 ||
            half_page_chunked.graph_capture_count != 1 ||
            half_page_chunked.graph_replay_count != 1) {
            std::cerr << "persistent Gemma target changed across chunk schedules\n";
            return 1;
        }

        options.maximum_context = 2048;
        options.generation_tokens = 3;
        options.input_tokens.resize(1088);
        for (std::size_t index = 0; index < options.input_tokens.size(); ++index) {
            options.input_tokens[index] = varied_ids[index % varied_ids.size()];
        }
        options.chunk_tokens = 64;
        options.use_cuda_graph = false;
        const auto eager_wrap = run_persistent_target(artifact, options);
        options.use_cuda_graph = true;
        const auto graph_wrap = run_persistent_target(artifact, options);
        if (eager_wrap.generated_tokens != graph_wrap.generated_tokens ||
            eager_wrap.final_frontier != 1090 || graph_wrap.final_frontier != 1090 ||
            graph_wrap.graph_capture_count != 1 || graph_wrap.graph_replay_count != 2) {
            std::cerr << "graph Gemma decode changed across the local ring wrap\n";
            return 1;
        }

        options.maximum_context = 2048;
        options.input_tokens = varied_ids;
        options.chunk_tokens = static_cast<std::uint32_t>(varied_ids.size());
        options.generation_tokens = 64;
        options.use_cuda_graph = false;
        options.mtp_draft_tokens = 0;
        const auto ordinary_mtp_reference = run_persistent_target(artifact, options);
        std::vector<std::int32_t> expected_first_round;
        PersistentRunResult mtp1;
        for (std::uint32_t width = 1; width <= 6; ++width) {
            options.mtp_draft_tokens = width;
            const auto mtp = run_persistent_target(artifact, options);
            if (mtp.generated_tokens != ordinary_mtp_reference.generated_tokens ||
                mtp.final_frontier != ordinary_mtp_reference.final_frontier ||
                mtp.assistant_weights_bytes == 0 || mtp.mtp_workspace_bytes == 0 ||
                mtp.kv_payload_bytes != ordinary_mtp_reference.kv_payload_bytes ||
                mtp.kv_metadata_bytes != ordinary_mtp_reference.kv_metadata_bytes ||
                mtp.mtp_draft_width != width || mtp.mtp_rounds == 0 ||
                mtp.mtp_proposed_tokens < mtp.mtp_rounds ||
                mtp.mtp_proposed_tokens <= mtp.mtp_accepted_tokens ||
                mtp.mtp_first_round_drafts.size() != width ||
                mtp.mtp_accepted_per_position.size() != width ||
                !(mtp.proposal_head_milliseconds > 0.0)) {
                std::cerr << "Gemma MTP" << width
                          << " changed greedy output or omitted assistant execution"
                          << " draft=" << mtp.mtp_first_draft_token
                          << " rounds=" << mtp.mtp_rounds
                          << " proposed=" << mtp.mtp_proposed_tokens
                          << " accepted=" << mtp.mtp_accepted_tokens << '\n';
                return 1;
            }
            if (!expected_first_round.empty() &&
                !std::equal(expected_first_round.begin(), expected_first_round.end(),
                            mtp.mtp_first_round_drafts.begin())) {
                std::cerr << "Gemma assistant feedback changed an earlier draft at width "
                          << width << '\n';
                return 1;
            }
            expected_first_round = mtp.mtp_first_round_drafts;
            if (width == 1) { mtp1 = mtp; }
        }
        options.mtp_draft_tokens = 1;
        options.use_cuda_graph = true;
        const auto mtp1_graph = run_persistent_target(artifact, options);
        // Pinned BF16 assistant oracle for the six-token checkpoint prompt at position 6.
        constexpr std::int32_t expected_first_draft = 236743;
        if (mtp1_graph.generated_tokens != ordinary_mtp_reference.generated_tokens ||
            mtp1_graph.final_frontier != ordinary_mtp_reference.final_frontier ||
            mtp1.mtp_accepted_tokens == 0 ||
            mtp1.mtp_first_draft_token != expected_first_draft ||
            mtp1_graph.graph_capture_count != 1 ||
            mtp1_graph.graph_replay_count != mtp1_graph.mtp_rounds) {
            std::cerr << "Gemma MTP1 changed greedy output or omitted assistant execution"
                      << " draft=" << mtp1.mtp_first_draft_token
                      << " proposed=" << mtp1.mtp_proposed_tokens
                      << " accepted=" << mtp1.mtp_accepted_tokens << "\nordinary:";
            for (std::int32_t token : ordinary_mtp_reference.generated_tokens) {
                std::cerr << ' ' << token;
            }
            std::cerr << "\nmtp1:";
            for (std::int32_t token : mtp1.generated_tokens) {
                std::cerr << ' ' << token;
            }
            std::cerr << '\n';
            return 1;
        }

        options.maximum_context = 2048;
        options.input_tokens.resize(1153);
        for (std::size_t index = 0; index < options.input_tokens.size(); ++index) {
            options.input_tokens[index] = varied_ids[index % varied_ids.size()];
        }
        options.chunk_tokens = 64;
        options.generation_tokens = 4;
        options.use_cuda_graph = true;
        options.qualify_graph_transactions = false;
        options.mtp_draft_tokens = 0;
        options.save_continuation = true;
        options.continuation_anchors = {1089};
        auto saved = run_persistent_target(artifact, options);
        if (saved.continuation_snapshot.empty() ||
            saved.continuation_snapshot.size() <= saved.continuation_payload_bytes ||
            saved.continuation_anchor_count != 1 || saved.restored_tokens != 0 ||
            saved.computed_prefill_tokens != options.input_tokens.size()) {
            std::cerr << "Gemma continuation capture omitted heterogeneous state\n";
            return 1;
        }

        options.save_continuation = false;
        options.continuation_anchors.clear();
        options.restore_continuation = std::move(saved.continuation_snapshot);
        const auto exact_restore = run_persistent_target(artifact, options);
        if (exact_restore.generated_tokens != saved.generated_tokens ||
            exact_restore.final_frontier != saved.final_frontier ||
            exact_restore.restored_tokens != 1152 ||
            exact_restore.computed_prefill_tokens != 1) {
            std::cerr << "Gemma endpoint restore changed output or re-prefilled its prefix\n";
            return 1;
        }

        auto snapshot = std::move(options.restore_continuation);
        options.input_tokens[1100] = 2;
        options.restore_continuation.clear();
        const auto edited_cold = run_persistent_target(artifact, options);
        options.restore_continuation = std::move(snapshot);
        const auto edited_restore = run_persistent_target(artifact, options);
        if (edited_restore.generated_tokens != edited_cold.generated_tokens ||
            edited_restore.final_frontier != edited_cold.final_frontier ||
            edited_restore.restored_tokens != 1089 ||
            edited_restore.computed_prefill_tokens != 64) {
            std::cerr << "Gemma anchor restore reused stale local KV or re-prefilled its prefix\n";
            return 1;
        }
        std::cout << "persistent Gemma target smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "persistent Gemma target smoke failed: " << error.what() << '\n';
        return 1;
    }
}
