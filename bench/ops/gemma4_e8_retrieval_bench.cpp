#include "core/paged_kv_storage.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kHeadDim = 512;
constexpr int kGroup = 64;
constexpr int kTop = 32;

class Random {
public:
    explicit Random(std::uint32_t state) : state_(state) {}

    float symmetric() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 17;
        state_ ^= state_ << 5;
        return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }

private:
    std::uint32_t state_;
};

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp = (bits >> 23) & 0xffu;
    std::uint32_t mantissa = bits & 0x007fffffu;
    if (exp == 0xffu) return static_cast<std::uint16_t>(sign | 0x7c00u);
    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x00800000u;
        const int shift = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) ++half_mantissa;
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail = mantissa & 0x1fffu;
    std::uint32_t rounded_exp = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exponent = (bits >> 10) & 0x1f;
    const int mantissa = bits & 0x03ff;
    const float magnitude = exponent == 0
                                ? std::ldexp(static_cast<float>(mantissa), -24)
                                : std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                                             exponent - 15);
    return negative ? -magnitude : magnitude;
}

int round_even(float value) {
    const float lower_f = std::floor(value);
    const int lower = static_cast<int>(lower_f);
    const float tail = value - lower_f;
    if (tail < 0.5f) return lower;
    if (tail > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

void hadamard64(float* values) {
    for (int stride = 1; stride < kGroup; stride <<= 1) {
        for (int base = 0; base < kGroup; base += 2 * stride) {
            for (int offset = 0; offset < stride; ++offset) {
                const int lo = base + offset;
                const int hi = lo + stride;
                const float a = values[lo];
                const float b = values[hi];
                values[lo] = a + b;
                values[hi] = a - b;
            }
        }
    }
    for (int d = 0; d < kGroup; ++d) values[d] *= 0.125f;
}

void project_e8(float* values) {
    float d8[8]{};
    float coset[8]{};
    int d8_sum = 0;
    int coset_sum = 0;
    float d8_error = -1.0f;
    float coset_error = -1.0f;
    int d8_worst = 0;
    int coset_worst = 0;
    for (int d = 0; d < 8; ++d) {
        d8[d] = static_cast<float>(round_even(values[d]));
        d8_sum += static_cast<int>(d8[d]);
        const float error = std::abs(values[d] - d8[d]);
        if (error > d8_error) {
            d8_error = error;
            d8_worst = d;
        }
        const float shifted = values[d] - 0.5f;
        const float nearest = static_cast<float>(round_even(shifted));
        coset[d] = nearest + 0.5f;
        coset_sum += static_cast<int>(nearest);
        const float shifted_error = std::abs(shifted - nearest);
        if (shifted_error > coset_error) {
            coset_error = shifted_error;
            coset_worst = d;
        }
    }
    if ((d8_sum & 1) != 0) d8[d8_worst] += values[d8_worst] >= d8[d8_worst] ? 1.0f : -1.0f;
    if ((coset_sum & 1) != 0) {
        const float shifted = values[coset_worst] - 0.5f;
        const float nearest = coset[coset_worst] - 0.5f;
        coset[coset_worst] += shifted >= nearest ? 1.0f : -1.0f;
    }
    float d8_distance = 0.0f;
    float coset_distance = 0.0f;
    for (int d = 0; d < 8; ++d) {
        const float delta0 = values[d] - d8[d];
        const float delta1 = values[d] - coset[d];
        d8_distance += delta0 * delta0;
        coset_distance += delta1 * delta1;
    }
    for (int d = 0; d < 8; ++d) values[d] = d8_distance <= coset_distance ? d8[d] : coset[d];
}

void normalize(std::array<float, kHeadDim>& values) {
    double sum = 0.0;
    for (float value : values) sum += static_cast<double>(value) * value;
    const float inverse = 1.0f / static_cast<float>(std::sqrt(sum));
    for (float& value : values) value *= inverse;
}

float dot(const std::array<float, kHeadDim>& left,
          const std::array<float, kHeadDim>& right) {
    float result = 0.0f;
    for (int d = 0; d < kHeadDim; ++d) result += left[d] * right[d];
    return result;
}

float quantized_dot(const std::array<float, kHeadDim>& query,
                    const std::array<float, kHeadDim>& key) {
    float result = 0.0f;
    for (int group = 0; group < kHeadDim / kGroup; ++group) {
        float values[kGroup]{};
        for (int d = 0; d < kGroup; ++d) values[d] = key[group * kGroup + d];
        hadamard64(values);
        float absmax = 0.0f;
        for (float value : values) absmax = std::max(absmax, std::abs(value));
        const float scale = f16_bits_to_f32(f32_to_f16_bits(absmax / 7.0f));
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int base = 0; base < kGroup; base += 8) {
            for (int d = 0; d < 8; ++d) values[base + d] *= inverse;
            project_e8(values + base);
            for (int d = 0; d < 8; ++d) {
                values[base + d] = static_cast<float>(
                    std::clamp(round_even(values[base + d]), -8, 7)) * scale;
            }
        }
        hadamard64(values);
        for (int d = 0; d < kGroup; ++d) result += query[group * kGroup + d] * values[d];
    }
    return result;
}

std::vector<std::size_t> top_indices(const std::vector<float>& scores) {
    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + kTop, order.end(),
                      [&](std::size_t left, std::size_t right) {
                          return scores[left] > scores[right];
                      });
    order.resize(kTop);
    return order;
}

int run_retrieval(std::size_t context, std::uint32_t seed) {
    Random random(seed);
    std::array<float, kHeadDim> query{};
    for (float& value : query) value = random.symmetric();
    normalize(query);
    const std::size_t needle = context * 7 / 13;
    std::vector<float> exact(context);
    std::vector<float> quantized(context);
    for (std::size_t token = 0; token < context; ++token) {
        std::array<float, kHeadDim> key{};
        for (float& value : key) value = random.symmetric();
        normalize(key);
        if (token == needle) {
            const float projection = dot(query, key);
            for (int d = 0; d < kHeadDim; ++d) key[d] -= projection * query[d];
            normalize(key);
            constexpr float kNeedleCosine = 0.25f;
            constexpr float kNoise = 0.9682458366f;
            for (int d = 0; d < kHeadDim; ++d) {
                key[d] = kNeedleCosine * query[d] + kNoise * key[d];
            }
        }
        exact[token] = dot(query, key);
        quantized[token] = quantized_dot(query, key);
    }
    const auto exact_top = top_indices(exact);
    const auto quantized_top = top_indices(quantized);
    int overlap = 0;
    for (std::size_t index : exact_top) {
        overlap += std::find(quantized_top.begin(), quantized_top.end(), index) !=
                   quantized_top.end();
    }
    const auto quantized_needle =
        std::find(quantized_top.begin(), quantized_top.end(), needle);
    const int needle_rank = quantized_needle == quantized_top.end()
                                ? kTop + 1
                                : static_cast<int>(quantized_needle - quantized_top.begin()) + 1;
    double exact_squared = 0.0;
    double quantized_squared = 0.0;
    double cross = 0.0;
    for (std::size_t token = 0; token < context; ++token) {
        exact_squared += static_cast<double>(exact[token]) * exact[token];
        quantized_squared += static_cast<double>(quantized[token]) * quantized[token];
        cross += static_cast<double>(exact[token]) * quantized[token];
    }
    const double score_cosine = cross / std::sqrt(exact_squared * quantized_squared);
    const double recall = static_cast<double>(overlap) / kTop;
    std::cout << "E8_RETRIEVAL context=" << context << " needle_exact_rank="
              << (std::find(exact_top.begin(), exact_top.end(), needle) - exact_top.begin() + 1)
              << " needle_quantized_rank=" << needle_rank << " recall_at_" << kTop << '='
              << recall << " score_cosine=" << score_cosine << '\n';
    return needle_rank > 3 || recall < 0.65 || score_cosine < 0.975 ? 1 : 0;
}

void report_bytes(std::size_t context) {
    const auto local = paged_kv_storage_layout(KvCacheStorage::RK4V4E8, 256);
    const auto global = paged_kv_storage_layout(KvCacheStorage::RK4V4E8, 512);
    const std::size_t local_min_bytes =
        50ULL * 16 * 1024 * local.physical_bytes_per_token_head();
    const std::size_t local_max_bytes =
        50ULL * 16 * 1088 * local.physical_bytes_per_token_head();
    const std::size_t global_bytes =
        10ULL * 4 * context * global.physical_bytes_per_token_head();
    std::cout << "E8_BYTES context=" << context << " local_window_min_bytes=" << local_min_bytes
              << " local_window_max_bytes=" << local_max_bytes
              << " global_bytes=" << global_bytes << " combined_min_bytes="
              << local_min_bytes + global_bytes << " combined_max_bytes="
              << local_max_bytes + global_bytes << '\n';
}

} // namespace

int main() {
    int failures = 0;
    failures += run_retrieval(32768, 0x409031u);
    failures += run_retrieval(65536, 0x409064u);
    failures += run_retrieval(240000, 0x409240u);
    failures += run_retrieval(260000, 0x409260u);
    report_bytes(32768);
    report_bytes(65536);
    report_bytes(240000);
    report_bytes(260000);
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_e8_retrieval\n";
    return failures ? 1 : 0;
}
