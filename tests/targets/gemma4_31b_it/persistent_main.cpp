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
        const std::vector<std::int32_t> short_ids(5, 2);
        const auto reference = run_reference_prefix(artifact, short_ids);
        PersistentRunOptions options;
        options.maximum_context = 1024;
        options.prefill_tokens = 5;
        options.chunk_tokens = 5;
        options.input_token = 2;
        options.run_deep_decode = false;
        const auto short_persistent = run_persistent_target(artifact, options);
        if (short_persistent.greedy_token != reference.greedy_token ||
            short_persistent.final_frontier != 5 ||
            short_persistent.kv_payload_bytes == 0 || short_persistent.workspace_bytes == 0) {
            std::cerr << "short persistent Gemma target changed its reference result\n";
            return 1;
        }

        options.prefill_tokens = 65;
        options.chunk_tokens = 64;
        const auto page_chunked = run_persistent_target(artifact, options);
        options.chunk_tokens = 32;
        const auto half_page_chunked = run_persistent_target(artifact, options);
        if (page_chunked.greedy_token != half_page_chunked.greedy_token ||
            page_chunked.final_frontier != 65 || half_page_chunked.final_frontier != 65) {
            std::cerr << "persistent Gemma target changed across chunk schedules\n";
            return 1;
        }
        std::cout << "persistent Gemma target smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "persistent Gemma target smoke failed: " << error.what() << '\n';
        return 1;
    }
}
