#include "targets/gemma4_31b_it/impl/runtime/persistent_model.h"
#include "targets/gemma4_31b_it/impl/runtime/reference_model.h"

#include <cstdlib>
#include <exception>
#include <iostream>
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
        std::cout << "persistent Gemma target smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "persistent Gemma target smoke failed: " << error.what() << '\n';
        return 1;
    }
}
