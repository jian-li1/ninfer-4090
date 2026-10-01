#include "targets/gemma4_31b_it/impl/runtime/reference_model.h"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        std::filesystem::path artifact;
        std::filesystem::path dump;
        std::vector<std::int32_t> ids;
        const bool checkpoint_fixture = argc == 1;
        if (argc == 1) {
            const char* requested = std::getenv("NINFER_GEMMA4_ARTIFACT");
            if (requested == nullptr) { return 77; }
            artifact = requested;
            const char* requested_dump = std::getenv("NINFER_GEMMA4_DUMP");
            dump = requested_dump == nullptr ? "/tmp/ninfer_gemma4_phase4_dump" : requested_dump;
            ids = {9259, 236764, 147224, 236743, 236812, 236888};
        } else {
        if (argc < 4) {
            std::cerr << "usage: " << argv[0]
                      << " ARTIFACT DUMP_DIRECTORY TOKEN_ID [TOKEN_ID ...]\n";
            return 2;
        }
        artifact = argv[1];
        dump = argv[2];
        ids.reserve(static_cast<std::size_t>(argc - 3));
        for (int index = 3; index < argc; ++index) {
            ids.push_back(static_cast<std::int32_t>(std::stol(argv[index])));
        }
        if (ids.empty()) { throw std::invalid_argument("at least one token id is required"); }
        }
        const auto result = ninfer::targets::gemma4_31b_it::detail::run_reference_prefix(
            artifact, ids, dump);
        if (checkpoint_fixture && result.greedy_token != 100) {
            throw std::runtime_error("Gemma checkpoint greedy token does not match reference 100");
        }
        std::cout << "{\"greedy_token\":" << result.greedy_token << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
