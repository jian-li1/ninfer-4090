#include "ninfer/engine.h"

#include <cerrno>
#include <chrono>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <span>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\nusage: ninfer_gemma4_31b_it_retrieval ARTIFACT "
                 "(--tokens-file PATH|--prompt-tokens N) [--max-context N] "
                 "[--prefill-chunk N] [--generate N] [--mtp N] [--tokens-out PATH]\n",
                 message);
    std::exit(2);
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

std::vector<ninfer::TokenId> read_tokens(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) usage("cannot open --tokens-file");
    std::vector<ninfer::TokenId> tokens;
    std::int64_t token = 0;
    while (input >> token) {
        if (token < 0 || token >= 262144) usage("invalid token in --tokens-file");
        tokens.push_back(static_cast<ninfer::TokenId>(token));
    }
    if (!input.eof() || tokens.empty()) usage("invalid or empty --tokens-file");
    return tokens;
}

constexpr std::array<std::pair<std::string_view, std::string_view>, 5> kNeedles{{
    {"KEY_917263", "orange-telescope"},
    {"KEY_204811", "violet-compass"},
    {"KEY_731509", "silver-lantern"},
    {"KEY_448027", "cobalt-orchid"},
    {"KEY_665193", "amber-harbor"},
}};
constexpr std::array<double, 5> kNeedleRatios{{0.05, 0.25, 0.50, 0.75, 0.95}};

void append_text(ninfer::Engine& engine, std::vector<ninfer::TokenId>& tokens,
                 std::string_view text) {
    auto encoded = engine.tokenize_text(text);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
}

void extend_repeated(std::vector<ninfer::TokenId>& tokens,
                     std::span<const ninfer::TokenId> filler, std::size_t stop) {
    if (filler.empty() || tokens.size() > stop) {
        throw std::runtime_error("retrieval prompt material exceeded its fixed position");
    }
    while (tokens.size() + filler.size() <= stop) {
        tokens.insert(tokens.end(), filler.begin(), filler.end());
    }
    tokens.insert(tokens.end(), filler.begin(),
                  filler.begin() + static_cast<std::ptrdiff_t>(stop - tokens.size()));
}

std::vector<ninfer::TokenId> build_five_needle_prompt(ninfer::Engine& engine,
                                                       std::uint32_t prompt_tokens) {
    std::vector<ninfer::TokenId> tokens;
    append_text(engine, tokens,
                "<bos><|turn>user\n"
                "You are reading an immutable synthetic archive. Remember every KEY value "
                "exactly. Ignore unrelated archive records.\n");
    const auto filler = engine.tokenize_text(
        "Archive filler record: routine calibration completed normally; this record contains "
        "no key assignment and no requested answer.\n");
    const auto suffix = engine.tokenize_text(
        "\nEnd of archive. Return the values for KEY_917263, KEY_204811, KEY_731509, "
        "KEY_448027, and KEY_665193 in that order. Reply with only the five values separated "
        "by single spaces.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    for (std::size_t index = 0; index < kNeedles.size(); ++index) {
        const auto desired = static_cast<std::size_t>(
            std::llround(static_cast<double>(prompt_tokens) * kNeedleRatios[index]));
        extend_repeated(tokens, filler, desired);
        append_text(engine, tokens,
                    "\nImmutable archive fact: " + std::string(kNeedles[index].first) + " = " +
                        std::string(kNeedles[index].second) + ".\n");
    }
    if (suffix.size() > prompt_tokens) {
        throw std::runtime_error("retrieval suffix exceeds requested prompt size");
    }
    extend_repeated(tokens, filler, prompt_tokens - suffix.size());
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    if (tokens.size() != prompt_tokens) {
        throw std::runtime_error("retrieval prompt construction produced the wrong token count");
    }
    return tokens;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char byte) {
        return static_cast<char>(std::tolower(byte));
    });
    return value;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) usage("missing artifact");
    const std::filesystem::path artifact = argv[1];
    std::filesystem::path tokens_file;
    std::filesystem::path tokens_out;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t maximum_context = 262144;
    std::uint32_t prefill_chunk = 1024;
    std::uint32_t generate = 32;
    std::uint32_t mtp = 0;
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) {
            if (++index == argc) usage(flag);
            return argv[index];
        };
        if (argument == "--tokens-file") {
            tokens_file = next("--tokens-file");
        } else if (argument == "--prompt-tokens") {
            prompt_tokens = parse_u32(next("--prompt-tokens"), 4096, 261120,
                                      "--prompt-tokens");
        } else if (argument == "--max-context") {
            maximum_context = parse_u32(next("--max-context"), 1024, 262144, "--max-context");
        } else if (argument == "--prefill-chunk") {
            prefill_chunk = parse_u32(next("--prefill-chunk"), 1, 2048, "--prefill-chunk");
        } else if (argument == "--generate") {
            generate = parse_u32(next("--generate"), 1, 1024, "--generate");
        } else if (argument == "--mtp") {
            mtp = parse_u32(next("--mtp"), 1, 6, "--mtp");
        } else if (argument == "--tokens-out") {
            tokens_out = next("--tokens-out");
        } else {
            usage("unknown argument");
        }
    }
    if (tokens_file.empty() == (prompt_tokens == 0)) {
        usage("select exactly one of --tokens-file and --prompt-tokens");
    }

    try {
        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = artifact;
        engine_options.max_context = maximum_context;
        engine_options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(maximum_context);
        engine_options.prefill_chunk = prefill_chunk;
        engine_options.kv_cache = ninfer::KvCacheStorage::RK4V4E8;
        engine_options.context_cache.enabled = false;
        engine_options.use_cuda_graph = true;
        if (mtp != 0) {
            engine_options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            engine_options.speculative.draft_tokens = mtp;
        }

        ninfer::Engine engine(std::move(engine_options));
        const ninfer::MemorySummary memory = engine.memory_summary();
        auto tokens = tokens_file.empty() ? build_five_needle_prompt(engine, prompt_tokens)
                                          : read_tokens(tokens_file);
        if (!tokens_out.empty()) {
            std::ofstream output(tokens_out, std::ios::trunc);
            for (ninfer::TokenId token : tokens) output << token << '\n';
            if (!output) throw std::runtime_error("failed to write --tokens-out");
        }
        if (tokens.size() + generate > maximum_context) {
            usage("prompt plus generation exceeds --max-context");
        }
        auto prompt = engine.prepare_tokens(std::move(tokens), false);
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = generate;
        request.execution.allow_prefix_reuse = false;
        request.execution.sampling.temperature = 0.0F;
        request.stop.include_model_defaults = false;
        request.output.raw = true;
        request.output.preserve_special_tokens = true;

        const auto started = std::chrono::steady_clock::now();
        ninfer::GenerationResult result = engine.generate(std::move(prompt), request);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        if (result.generated_token_ids.size() != generate ||
            result.finish_reason != ninfer::FinishReason::OutputLimit) {
            throw std::runtime_error("retrieval request did not reach the requested output limit");
        }
        const std::string generated_text = result.reasoning + result.content;
        const std::string normalized = lowercase(generated_text);
        std::size_t missing = 0;
        std::size_t search_from = 0;
        for (const auto& [key, value] : kNeedles) {
            (void)key;
            const std::size_t found = normalized.find(lowercase(std::string(value)), search_from);
            if (found == std::string::npos) {
                ++missing;
            } else {
                search_from = found + value.size();
            }
        }

        std::printf(
            "GEMMA4_ENGINE_RETRIEVAL prompt_tokens=%u generated_tokens=%zu "
            "prefill_ms=%.3f decode_ms=%.3f total_ms=%.3f wall_ms=%.3f "
            "weights=%zu sequence=%zu workspace=%zu kv_payload=%zu free_startup=%zu "
            "mtp_width=%u mtp_rounds=%llu mtp_drafted=%llu mtp_accepted=%llu "
            "missing_values=%zu success=%u\n",
            result.prompt.prompt_tokens, result.generated_token_ids.size(),
            1000.0 * result.timings.prefill_seconds, 1000.0 * result.timings.decode_seconds,
            1000.0 * result.timings.total_seconds, 1000.0 * elapsed,
            memory.weights.capacity_bytes, memory.sequence.capacity_bytes,
            memory.workspace.capacity_bytes, memory.kv_payload_bytes,
            memory.available_after_startup_bytes, mtp,
            static_cast<unsigned long long>(result.speculative.rounds),
            static_cast<unsigned long long>(result.speculative.drafted_tokens),
            static_cast<unsigned long long>(result.speculative.accepted_tokens), missing,
            missing == 0 ? 1U : 0U);
        std::printf("GEMMA4_GENERATED_IDS=");
        for (std::size_t index = 0; index < result.generated_token_ids.size(); ++index) {
            std::printf("%s%d", index == 0 ? "" : ",", result.generated_token_ids[index]);
        }
        std::printf("\n");
        std::printf("GEMMA4_GENERATED_TEXT_BEGIN\n%s\nGEMMA4_GENERATED_TEXT_END\n",
                    generated_text.c_str());
        return missing == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gemma4_31b_it_retrieval: %s\n", error.what());
        return 1;
    }
}
