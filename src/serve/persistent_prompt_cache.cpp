#include "serve/persistent_prompt_cache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace ninfer::serve {
namespace {

namespace fs = std::filesystem;

constexpr std::array<char, 8> kTokenMagic = {'N', 'I', 'N', 'K', 'V', 'T', '1', '\0'};
constexpr std::uint32_t kTokenVersion     = 1;
constexpr std::uint64_t kTokenHeaderBytes = kTokenMagic.size() + sizeof(std::uint32_t) +
                                             sizeof(std::uint64_t);

static_assert(sizeof(TokenId) == sizeof(std::int32_t));

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!out.good()) {
        throw std::runtime_error("failed writing persistent-cache token metadata");
    }
}

template <typename T>
T read_scalar(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!in.good()) { throw std::runtime_error("truncated persistent-cache token metadata"); }
    return value;
}

std::string temporary_suffix() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return ".tmp." + std::to_string(static_cast<unsigned long long>(now)) + "." +
           std::to_string(static_cast<unsigned long long>(tid));
}

std::uint64_t saturating_add(std::uint64_t lhs, std::uint64_t rhs) noexcept {
    return rhs > std::numeric_limits<std::uint64_t>::max() - lhs
               ? std::numeric_limits<std::uint64_t>::max()
               : lhs + rhs;
}

} // namespace

PersistentPromptCache::PersistentPromptCache(fs::path directory, std::uint64_t max_bytes)
    : directory_(std::move(directory)), entries_directory_(directory_ / "entries"),
      max_bytes_(max_bytes) {
    if (directory_.empty()) {
        throw std::invalid_argument("persistent cache directory must not be empty");
    }
    std::error_code error;
    fs::create_directories(entries_directory_, error);
    if (error) {
        throw std::invalid_argument("failed to create persistent cache directory: " +
                                    error.message());
    }
    scan();
}

bool PersistentPromptCache::is_prefix(std::span<const TokenId> prefix,
                                      std::span<const TokenId> prompt) noexcept {
    return prefix.size() <= prompt.size() && std::equal(prefix.begin(), prefix.end(), prompt.begin());
}

std::string PersistentPromptCache::checked_digest_key(std::string_view digest) {
    if (digest.empty()) {
        throw std::invalid_argument("persistent cache session digest is empty");
    }
    std::string result;
    result.reserve(digest.size());
    for (const char ch : digest) {
        const auto byte = static_cast<unsigned char>(ch);
        if (!(std::isalnum(byte) || ch == '-' || ch == '_')) {
            throw std::invalid_argument(
                "persistent cache session digest contains an unsafe filename character");
        }
        result.push_back(ch);
    }
    return result;
}

std::uint64_t PersistentPromptCache::file_size_noexcept(const fs::path& path) noexcept {
    std::error_code error;
    const std::uintmax_t size = fs::file_size(path, error);
    return error ? 0 : static_cast<std::uint64_t>(size);
}

std::vector<TokenId> PersistentPromptCache::read_tokens(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open persistent-cache token file");
    }
    std::array<char, kTokenMagic.size()> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in.good() || magic != kTokenMagic) {
        throw std::runtime_error("invalid persistent-cache token magic");
    }
    const std::uint32_t version = read_scalar<std::uint32_t>(in);
    if (version != kTokenVersion) {
        throw std::runtime_error("unsupported persistent-cache token version");
    }
    const std::uint64_t count = read_scalar<std::uint64_t>(in);
    if (count > std::numeric_limits<std::uint32_t>::max() ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                           sizeof(TokenId))) {
        throw std::runtime_error("persistent-cache token count is too large");
    }
    const std::uint64_t expected_bytes = count * sizeof(TokenId);
    const std::uint64_t actual_bytes   = file_size_noexcept(path);
    if (actual_bytes != kTokenHeaderBytes + expected_bytes) {
        throw std::runtime_error("persistent-cache token file has an invalid size");
    }
    if (expected_bytes >
        static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("persistent-cache token file is too large");
    }
    std::vector<TokenId> tokens(static_cast<std::size_t>(count));
    if (!tokens.empty()) {
        in.read(reinterpret_cast<char*>(tokens.data()),
                static_cast<std::streamsize>(expected_bytes));
        if (!in.good()) { throw std::runtime_error("truncated persistent-cache token payload"); }
    }
    return tokens;
}

void PersistentPromptCache::write_tokens_atomic(const fs::path& path,
                                                std::span<const TokenId> tokens) {
    const fs::path staging(path.string() + temporary_suffix());
    try {
        std::ofstream out(staging, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error("failed to create persistent-cache token file");
        }
        out.write(kTokenMagic.data(), static_cast<std::streamsize>(kTokenMagic.size()));
        if (!out.good()) {
            throw std::runtime_error("failed writing persistent-cache token magic");
        }
        write_scalar(out, kTokenVersion);
        const auto count = static_cast<std::uint64_t>(tokens.size());
        write_scalar(out, count);
        const std::uint64_t bytes = count * sizeof(TokenId);
        if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("persistent-cache token file is too large");
        }
        if (!tokens.empty()) {
            out.write(reinterpret_cast<const char*>(tokens.data()),
                      static_cast<std::streamsize>(bytes));
            if (!out.good()) {
                throw std::runtime_error("failed writing persistent-cache tokens");
            }
        }
        out.flush();
        if (!out.good()) {
            throw std::runtime_error("failed flushing persistent-cache token file");
        }
        out.close();
        std::error_code error;
        fs::rename(staging, path, error);
        if (error) {
            fs::remove(staging);
            throw std::runtime_error("failed publishing persistent-cache token file: " +
                                     error.message());
        }
    } catch (...) {
        std::error_code ignored;
        fs::remove(staging, ignored);
        throw;
    }
}

void PersistentPromptCache::scan() {
    std::lock_guard lock(mutex_);
    entries_.clear();
    std::error_code iteration_error;
    for (const fs::directory_entry& file :
         fs::directory_iterator(entries_directory_, iteration_error)) {
        if (iteration_error) { break; }
        if (!file.is_regular_file()) { continue; }
        const std::string filename = file.path().filename().string();
        if (filename.find(".tmp.") != std::string::npos) {
            std::error_code ignored;
            fs::remove(file.path(), ignored);
            continue;
        }
        if (file.path().extension() != ".tok") { continue; }
        const std::string digest = file.path().stem().string();
        const fs::path snapshot  = entries_directory_ / (digest + ".bin");
        if (!fs::exists(snapshot)) {
            std::error_code ignored;
            fs::remove(file.path(), ignored);
            continue;
        }
        try {
            Entry entry;
            entry.tokens_path   = file.path();
            entry.snapshot_path = snapshot;
            entry.digest        = checked_digest_key(digest);
            entry.tokens        = read_tokens(entry.tokens_path);
            entry.bytes = saturating_add(file_size_noexcept(entry.tokens_path),
                                         file_size_noexcept(entry.snapshot_path));
            std::error_code time_error;
            entry.last_used = fs::last_write_time(entry.snapshot_path, time_error);
            if (time_error) { entry.last_used = fs::file_time_type::min(); }
            entries_.push_back(std::move(entry));
        } catch (...) {
            std::error_code ignored;
            fs::remove(file.path(), ignored);
            fs::remove(snapshot, ignored);
        }
    }
    if (iteration_error) {
        throw std::invalid_argument("failed scanning persistent cache directory: " +
                                    iteration_error.message());
    }

    // A crash between native snapshot publication and sidecar publication leaves an orphaned
    // .bin. It cannot be matched safely, so reclaim it during startup.
    iteration_error.clear();
    for (const fs::directory_entry& file :
         fs::directory_iterator(entries_directory_, iteration_error)) {
        if (iteration_error) { break; }
        if (!file.is_regular_file() || file.path().extension() != ".bin") { continue; }
        const fs::path tokens = entries_directory_ / (file.path().stem().string() + ".tok");
        if (!fs::exists(tokens)) {
            std::error_code ignored;
            fs::remove(file.path(), ignored);
        }
    }
    if (iteration_error) {
        throw std::invalid_argument("failed cleaning persistent cache directory: " +
                                    iteration_error.message());
    }
    prune_locked();
}

std::optional<PersistentPromptCache::Match>
PersistentPromptCache::longest_prefix(std::span<const TokenId> prompt) {
    std::lock_guard lock(mutex_);
    Entry* best = nullptr;
    for (Entry& entry : entries_) {
        if (is_prefix(entry.tokens, prompt) &&
            (best == nullptr || entry.tokens.size() > best->tokens.size())) {
            best = &entry;
        }
    }
    if (best == nullptr) { return std::nullopt; }
    const auto now = fs::file_time_type::clock::now();
    best->last_used = now;
    std::error_code ignored;
    fs::last_write_time(best->snapshot_path, now, ignored);
    ignored.clear();
    fs::last_write_time(best->tokens_path, now, ignored);
    return Match{.snapshot_path = best->snapshot_path,
                 .digest        = best->digest,
                 .tokens        = static_cast<std::uint32_t>(best->tokens.size())};
}

std::optional<std::size_t>
PersistentPromptCache::prefix_for_digest(std::string_view digest,
                                         std::span<const TokenId> prompt) const {
    std::lock_guard lock(mutex_);
    const auto found = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& entry) {
        return entry.digest == digest;
    });
    if (found == entries_.end()) { return std::nullopt; }
    return is_prefix(found->tokens, prompt) ? found->tokens.size() : std::size_t{0};
}

void PersistentPromptCache::store(std::span<const TokenId> ledger,
                                  std::string_view expected_digest,
                                  const SaveSnapshot& save_snapshot) {
    if (ledger.empty() || expected_digest.empty()) { return; }
    if (!save_snapshot) {
        throw std::invalid_argument("persistent cache snapshot writer is empty");
    }
    const std::string digest  = checked_digest_key(expected_digest);
    const fs::path snapshot   = entries_directory_ / (digest + ".bin");
    const fs::path token_file = entries_directory_ / (digest + ".tok");
    const SlotSaveResult saved = save_snapshot(snapshot, digest);
    if (saved.tokens != ledger.size() || saved.session_digest != digest) {
        std::error_code ignored;
        fs::remove(snapshot, ignored);
        fs::remove(token_file, ignored);
        throw std::runtime_error("persistent cache snapshot identity mismatch");
    }
    try {
        write_tokens_atomic(token_file, ledger);
    } catch (...) {
        std::error_code ignored;
        fs::remove(snapshot, ignored);
        fs::remove(token_file, ignored);
        throw;
    }

    Entry entry;
    entry.snapshot_path = snapshot;
    entry.tokens_path   = token_file;
    entry.digest        = digest;
    entry.tokens.assign(ledger.begin(), ledger.end());
    entry.bytes = saturating_add(file_size_noexcept(snapshot), file_size_noexcept(token_file));
    std::error_code time_error;
    entry.last_used = fs::last_write_time(snapshot, time_error);
    if (time_error) { entry.last_used = fs::file_time_type::clock::now(); }

    std::lock_guard lock(mutex_);
    std::erase_if(entries_, [&](const Entry& existing) { return existing.digest == digest; });
    entries_.push_back(std::move(entry));
    prune_locked();
}

void PersistentPromptCache::invalidate(const Match& match) noexcept {
    std::lock_guard lock(mutex_);
    const fs::path token_file = entries_directory_ / (match.digest + ".tok");
    std::error_code ignored;
    fs::remove(match.snapshot_path, ignored);
    ignored.clear();
    fs::remove(token_file, ignored);
    std::erase_if(entries_, [&](const Entry& entry) { return entry.digest == match.digest; });
}

void PersistentPromptCache::prune_locked() {
    if (max_bytes_ == 0) { return; }
    std::uint64_t total = 0;
    for (const Entry& entry : entries_) { total = saturating_add(total, entry.bytes); }
    while (total > max_bytes_ && !entries_.empty()) {
        const auto oldest = std::min_element(
            entries_.begin(), entries_.end(),
            [](const Entry& lhs, const Entry& rhs) { return lhs.last_used < rhs.last_used; });
        if (oldest == entries_.end()) { break; }
        const std::uint64_t bytes = oldest->bytes;
        std::error_code ignored;
        fs::remove(oldest->snapshot_path, ignored);
        ignored.clear();
        fs::remove(oldest->tokens_path, ignored);
        entries_.erase(oldest);
        total = bytes > total ? 0 : total - bytes;
    }
}

} // namespace ninfer::serve
