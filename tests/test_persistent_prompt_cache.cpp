#include "serve/persistent_prompt_cache.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ninfer::TokenId;
using ninfer::serve::PersistentPromptCache;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct TempDirectory {
    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("ninfer-persistent-cache-test-" +
                                            std::to_string(stamp));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
    fs::path path;
};

ninfer::SlotSaveResult fake_save(const fs::path& path, std::string_view digest,
                                 std::span<const TokenId> tokens, std::size_t payload_bytes = 32) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    std::vector<char> payload(payload_bytes, 's');
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    out.close();
    return ninfer::SlotSaveResult{.tokens         = static_cast<std::uint32_t>(tokens.size()),
                                  .bytes          = payload.size(),
                                  .seconds        = 0.0,
                                  .session_digest = std::string(digest)};
}

} // namespace

int main() {
    int failures = 0;
    TempDirectory temp;
    const std::vector<TokenId> short_tokens{1, 2, 3};
    const std::vector<TokenId> long_tokens{1, 2, 3, 4, 5};
    const std::vector<TokenId> prompt{1, 2, 3, 4, 5, 6};

    PersistentPromptCache cache(temp.path, 0);
    cache.store(short_tokens, "short", [&](const fs::path& path, std::string_view digest) {
        return fake_save(path, digest, short_tokens);
    });
    cache.store(long_tokens, "long", [&](const fs::path& path, std::string_view digest) {
        return fake_save(path, digest, long_tokens);
    });

    const auto match = cache.longest_prefix(prompt);
    failures += check(match && match->digest == "long" && match->tokens == long_tokens.size(),
                      "longest persistent prefix was not selected");
    failures += check(cache.prefix_for_digest("short", prompt) == short_tokens.size(),
                      "digest prefix lookup lost a matching ledger");
    failures += check(cache.prefix_for_digest("long", std::vector<TokenId>{9, 2, 3}) == 0,
                      "digest prefix lookup accepted a different prompt");
    failures += check(!cache.prefix_for_digest("missing", prompt),
                      "digest prefix lookup fabricated an absent entry");

    // Entries survive catalog reconstruction; incomplete and malformed crash remnants do not.
    const fs::path entries = temp.path / "entries";
    {
        std::ofstream(entries / "orphan.bin") << "orphan";
        std::ofstream(entries / "broken.tok") << "broken";
        std::ofstream(entries / "broken.bin") << "snapshot";
        std::ofstream(entries / "stale.tmp.1") << "partial";
    }
    PersistentPromptCache rescanned(temp.path, 0);
    const auto rescanned_match = rescanned.longest_prefix(prompt);
    failures += check(rescanned_match && rescanned_match->digest == "long",
                      "startup scan lost a valid persistent entry");
    failures += check(!fs::exists(entries / "orphan.bin") &&
                          !fs::exists(entries / "broken.tok") &&
                          !fs::exists(entries / "broken.bin") &&
                          !fs::exists(entries / "stale.tmp.1"),
                      "startup scan retained an incomplete persistent entry");

    if (rescanned_match) { rescanned.invalidate(*rescanned_match); }
    failures += check(!fs::exists(entries / "long.bin") && !fs::exists(entries / "long.tok"),
                      "cache invalidation did not remove both entry files");

    // Each entry is 32 snapshot bytes plus a 20-byte header and its token payload. A 70-byte
    // limit can retain the newer 3-token entry (64 bytes) but not both entries.
    TempDirectory lru_temp;
    PersistentPromptCache lru(lru_temp.path, 70);
    lru.store(short_tokens, "old", [&](const fs::path& path, std::string_view digest) {
        return fake_save(path, digest, short_tokens);
    });
    lru.store(short_tokens, "new", [&](const fs::path& path, std::string_view digest) {
        return fake_save(path, digest, short_tokens);
    });
    const auto lru_match = lru.longest_prefix(prompt);
    failures += check(lru_match && lru_match->digest == "new",
                      "persistent cache did not evict the least-recent entry");
    failures += check(!fs::exists(lru_temp.path / "entries" / "old.bin") &&
                          !fs::exists(lru_temp.path / "entries" / "old.tok"),
                      "persistent cache LRU left old entry files behind");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
