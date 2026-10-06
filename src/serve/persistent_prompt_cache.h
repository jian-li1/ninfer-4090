#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// Disk catalog for complete native retained-session snapshots. The snapshot payload remains
// owned by Engine; this class stores only the rendered text-token ledger needed for cheap prefix
// discovery and manages publication, startup cleanup, and the disk LRU.
class PersistentPromptCache {
public:
    struct Match {
        std::filesystem::path snapshot_path;
        std::string digest;
        std::uint32_t tokens = 0;
    };

    using SaveSnapshot =
        std::function<SlotSaveResult(const std::filesystem::path&, std::string_view)>;

    enum class StoreDisposition : std::uint8_t {
        Stored,
        AlreadyPresent,
    };

    // max_bytes == 0 means unlimited. Entries are private implementation files below
    // directory/entries so a cache root can never collide with manual /slots filenames.
    PersistentPromptCache(std::filesystem::path directory, std::uint64_t max_bytes);

    // Returns the entry whose complete token ledger is the longest exact prefix of prompt.
    [[nodiscard]] std::optional<Match> longest_prefix(std::span<const TokenId> prompt);

    // nullopt means the digest is absent; zero means it exists but is not a prefix.
    [[nodiscard]] std::optional<std::size_t>
    prefix_for_digest(std::string_view digest, std::span<const TokenId> prompt) const;

    // save_snapshot must atomically publish a native Engine snapshot at the supplied path and
    // enforce the supplied digest as its retained-session precondition. An exact reusable-prefix
    // duplicate is touched for LRU purposes and returned without invoking save_snapshot.
    [[nodiscard]] StoreDisposition store(std::span<const TokenId> ledger,
                                         std::string_view expected_digest,
                                         const SaveSnapshot& save_snapshot);

    void invalidate(const Match& match) noexcept;

private:
    struct Entry {
        std::filesystem::path snapshot_path;
        std::filesystem::path tokens_path;
        std::string digest;
        std::vector<TokenId> tokens;
        std::uint64_t bytes = 0;
        std::filesystem::file_time_type last_used{};
    };

    std::filesystem::path directory_;
    std::filesystem::path entries_directory_;
    std::uint64_t max_bytes_ = 0;

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;

    void scan();
    void prune_locked();

    [[nodiscard]] static bool is_prefix(std::span<const TokenId> prefix,
                                        std::span<const TokenId> prompt) noexcept;
    [[nodiscard]] static bool is_exact(std::span<const TokenId> lhs,
                                       std::span<const TokenId> rhs) noexcept;
    static void touch_entry(Entry& entry) noexcept;
    [[nodiscard]] static std::string checked_digest_key(std::string_view digest);
    [[nodiscard]] static std::vector<TokenId> read_tokens(const std::filesystem::path& path);
    static void write_tokens_atomic(const std::filesystem::path& path,
                                    std::span<const TokenId> tokens);
    [[nodiscard]] static std::uint64_t
    file_size_noexcept(const std::filesystem::path& path) noexcept;
};

} // namespace ninfer::serve
