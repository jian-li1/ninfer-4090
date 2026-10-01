#include "targets/gemma4_31b_it/impl/runtime/persistent_model.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\nusage: ninfer_gemma4_31b_it_long_context ARTIFACT "
                 "[--max-context N] [--prefill N] [--chunk N] [--token ID] "
                 "[--tokens-file PATH] [--generate N] [--dump PATH] [--no-decode]\n",
                 message);
    std::exit(2);
}

std::vector<std::int32_t> read_tokens(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) usage("cannot open --tokens-file");
    std::vector<std::int32_t> tokens;
    std::int64_t token = 0;
    while (input >> token) {
        if (token < 0 || token > 262143) usage("invalid token in --tokens-file");
        tokens.push_back(static_cast<std::int32_t>(token));
    }
    if (!input.eof() || tokens.empty()) usage("invalid or empty --tokens-file");
    return tokens;
}

std::uint32_t parse_u32(const char* value, std::uint32_t minimum, std::uint32_t maximum,
                        const char* flag) {
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::uint32_t>(parsed);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) usage("missing artifact");
    std::filesystem::path artifact = argv[1];
    ninfer::targets::gemma4_31b_it::detail::PersistentRunOptions options;
    bool prefill_set = false;
    std::filesystem::path tokens_file;
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) {
            if (++index == argc) usage(flag);
            return argv[index];
        };
        if (argument == "--max-context") {
            options.maximum_context = parse_u32(next("--max-context"), 1, 262144, "--max-context");
        } else if (argument == "--prefill") {
            options.prefill_tokens = parse_u32(next("--prefill"), 1, 262144, "--prefill");
            prefill_set = true;
        } else if (argument == "--chunk") {
            options.chunk_tokens = parse_u32(next("--chunk"), 1, 64, "--chunk");
        } else if (argument == "--token") {
            options.input_token = static_cast<std::int32_t>(
                parse_u32(next("--token"), 0, 262143, "--token"));
        } else if (argument == "--tokens-file") {
            tokens_file = next("--tokens-file");
        } else if (argument == "--generate") {
            options.generation_tokens = parse_u32(next("--generate"), 1, 1024, "--generate");
            options.run_deep_decode = false;
        } else if (argument == "--dump") {
            options.dump_directory = next("--dump");
        } else if (argument == "--no-decode") {
            options.run_deep_decode = false;
        } else {
            usage("unknown argument");
        }
    }
    if (!tokens_file.empty()) {
        options.input_tokens = read_tokens(tokens_file);
        if (prefill_set && options.prefill_tokens != options.input_tokens.size()) {
            usage("--prefill does not match --tokens-file");
        }
        options.prefill_tokens = static_cast<std::uint32_t>(options.input_tokens.size());
    } else if (!prefill_set) {
        options.prefill_tokens = options.maximum_context - 1;
    }
    try {
        const auto result =
            ninfer::targets::gemma4_31b_it::detail::run_persistent_target(artifact, options);
        std::printf(
            "GEMMA4_LONG_CONTEXT max_context=%u frontier=%u weights=%zu kv_payload=%zu "
            "kv_metadata=%zu workspace=%zu free_weights=%zu free_cache=%zu free_workspace=%zu "
            "prefill_ms=%.3f decode_ms=%.3f greedy=%d\n",
            options.maximum_context, result.final_frontier, result.weights_bytes,
            result.kv_payload_bytes, result.kv_metadata_bytes, result.workspace_bytes,
            result.free_after_weights, result.free_after_cache, result.free_after_workspace,
            result.prefill_milliseconds, result.decode_milliseconds, result.greedy_token);
        if (!result.generated_tokens.empty()) {
            std::printf("GEMMA4_GENERATED_IDS=");
            for (std::size_t index = 0; index < result.generated_tokens.size(); ++index) {
                std::printf("%s%d", index == 0 ? "" : ",", result.generated_tokens[index]);
            }
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gemma4_31b_it_long_context: %s\n", error.what());
        return 1;
    }
}
