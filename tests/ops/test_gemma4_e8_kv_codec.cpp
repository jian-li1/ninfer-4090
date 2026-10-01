#include "ninfer/ops/kv_cache_append.h"

#include "core/paged_kv_storage.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kPage          = 64;
constexpr int kLogicalPages  = 4;
constexpr int kPhysicalPages = 4;
constexpr int kGroup         = 64;

std::size_t input_index(int head_dim, int kv_heads, int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(head_dim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(int leading, int kv_heads, int d, int head, int position,
                        int physical_page) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(leading) *
               (static_cast<std::size_t>(position % kPage) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kv_heads) *
                         static_cast<std::size_t>(physical_page)));
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }
    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) ++half_mantissa;
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exponent  = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exponent == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude =
            std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exponent - 15);
    }
    return negative ? -magnitude : magnitude;
}

int round_even(float value) {
    const float lower_f = std::floor(value);
    const int lower     = static_cast<int>(lower_f);
    const float tail    = value - lower_f;
    if (tail < 0.5f) return lower;
    if (tail > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

void hadamard64(std::vector<float>& values, int begin) {
    for (int stride = 1; stride < kGroup; stride <<= 1) {
        for (int base = 0; base < kGroup; base += 2 * stride) {
            for (int offset = 0; offset < stride; ++offset) {
                const int lo = begin + base + offset;
                const int hi = lo + stride;
                const float a = values[static_cast<std::size_t>(lo)];
                const float b = values[static_cast<std::size_t>(hi)];
                values[static_cast<std::size_t>(lo)] = a + b;
                values[static_cast<std::size_t>(hi)] = a - b;
            }
        }
    }
    for (int d = 0; d < kGroup; ++d) values[static_cast<std::size_t>(begin + d)] *= 0.125f;
}

void project_e8(float* values) {
    float d8[8]{};
    int d8_sum     = 0;
    float d8_error = -1.0f;
    int d8_worst   = 0;
    for (int d = 0; d < 8; ++d) {
        d8[d] = static_cast<float>(round_even(values[d]));
        d8_sum += static_cast<int>(d8[d]);
        const float error = std::abs(values[d] - d8[d]);
        if (error > d8_error) {
            d8_error = error;
            d8_worst = d;
        }
    }
    if ((d8_sum & 1) != 0) {
        d8[d8_worst] += values[d8_worst] >= d8[d8_worst] ? 1.0f : -1.0f;
    }

    float coset[8]{};
    int coset_sum     = 0;
    float coset_error = -1.0f;
    int coset_worst   = 0;
    for (int d = 0; d < 8; ++d) {
        const float shifted = values[d] - 0.5f;
        const float nearest = static_cast<float>(round_even(shifted));
        coset[d]             = nearest + 0.5f;
        coset_sum += static_cast<int>(nearest);
        const float error = std::abs(shifted - nearest);
        if (error > coset_error) {
            coset_error = error;
            coset_worst = d;
        }
    }
    if ((coset_sum & 1) != 0) {
        const float shifted = values[coset_worst] - 0.5f;
        const float nearest = coset[coset_worst] - 0.5f;
        coset[coset_worst] += shifted >= nearest ? 1.0f : -1.0f;
    }

    float d8_distance = 0.0f;
    float coset_distance = 0.0f;
    for (int d = 0; d < 8; ++d) {
        const float d8_delta = values[d] - d8[d];
        const float coset_delta = values[d] - coset[d];
        d8_distance += d8_delta * d8_delta;
        coset_distance += coset_delta * coset_delta;
    }
    for (int d = 0; d < 8; ++d) {
        values[d] = d8_distance <= coset_distance ? d8[d] : coset[d];
    }
}

std::uint8_t pack_i4(int lo, int hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

struct Quality {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double dot = 0.0;
    double squared_decoded = 0.0;
};

void add_quality(Quality& quality, const std::vector<float>& reference,
                 std::vector<float> decoded) {
    hadamard64(decoded, 0);
    for (int d = 0; d < kGroup; ++d) {
        const double ref = reference[static_cast<std::size_t>(d)];
        const double got = decoded[static_cast<std::size_t>(d)];
        const double delta = got - ref;
        quality.squared_error += delta * delta;
        quality.squared_reference += ref * ref;
        quality.dot += ref * got;
        quality.squared_decoded += got * got;
    }
}

void encode_group(const std::vector<float>& source, int source_begin, bool key,
                  std::vector<std::uint8_t>& codes, std::vector<std::uint16_t>& scales,
                  int head_dim, int kv_heads, int group, int head, int position,
                  int physical_page, Quality& quality) {
    std::vector<float> original(kGroup);
    std::vector<float> rotated(kGroup);
    for (int d = 0; d < kGroup; ++d) {
        original[static_cast<std::size_t>(d)] = source[static_cast<std::size_t>(source_begin + d)];
        rotated[static_cast<std::size_t>(d)] = original[static_cast<std::size_t>(d)];
    }
    hadamard64(rotated, 0);
    float absmax = 0.0f;
    for (float value : rotated) absmax = std::max(absmax, std::abs(value));
    const std::uint16_t scale_bits = f32_to_f16_bits(absmax > 0.0f ? absmax / 7.0f : 0.0f);
    const float scale = f16_bits_to_f32(scale_bits);
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    scales[cache_index(head_dim / kGroup, kv_heads, group, head, position, physical_page)] =
        scale_bits;

    std::vector<float> quantized(kGroup);
    if (key) {
        for (int base = 0; base < kGroup; base += 8) {
            float projected[8]{};
            for (int d = 0; d < 8; ++d) {
                projected[d] = rotated[static_cast<std::size_t>(base + d)] * inverse;
            }
            project_e8(projected);
            for (int d = 0; d < 8; ++d) {
                quantized[static_cast<std::size_t>(base + d)] = static_cast<float>(
                    std::clamp(round_even(projected[d]), -8, 7));
            }
        }
    } else {
        for (int d = 0; d < kGroup; ++d) {
            quantized[static_cast<std::size_t>(d)] = static_cast<float>(
                std::clamp(round_even(rotated[static_cast<std::size_t>(d)] * inverse), -7, 7));
        }
    }
    std::vector<float> decoded(kGroup);
    for (int d = 0; d < kGroup; d += 2) {
        const int q0 = static_cast<int>(quantized[static_cast<std::size_t>(d)]);
        const int q1 = static_cast<int>(quantized[static_cast<std::size_t>(d + 1)]);
        const int packed_d = group * (kGroup / 2) + d / 2;
        codes[cache_index(head_dim / 2, kv_heads, packed_d, head, position, physical_page)] =
            pack_i4(q0, q1);
        decoded[static_cast<std::size_t>(d)] = static_cast<float>(q0) * scale;
        decoded[static_cast<std::size_t>(d + 1)] = static_cast<float>(q1) * scale;
    }
    add_quality(quality, original, decoded);
}

int run_case(int head_dim, int kv_heads, int tokens, bool zeros) {
    const int first_position = 60;
    const int groups = head_dim / kGroup;
    const std::vector<std::int32_t> mapping{2, 0, 3, 1};
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) positions[static_cast<std::size_t>(token)] =
        first_position + token;

    const std::size_t input_count =
        static_cast<std::size_t>(head_dim) * kv_heads * tokens;
    std::vector<float> host_k(input_count);
    std::vector<float> host_v(input_count);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < kv_heads; ++head) {
            for (int d = 0; d < head_dim; ++d) {
                const std::size_t index = input_index(head_dim, kv_heads, d, head, token);
                host_k[index] = zeros ? 0.0f :
                    std::sin(0.071f * static_cast<float>(d + 1) + 0.31f * head + 0.17f * token) *
                        (0.25f + 0.01f * static_cast<float>((d + token) % 19));
                host_v[index] = zeros ? 0.0f :
                    std::cos(0.053f * static_cast<float>(d + 3) - 0.19f * head + 0.11f * token) *
                        (0.2f + 0.015f * static_cast<float>((d + head) % 17));
            }
        }
    }
    round_to_bf16(host_k);
    round_to_bf16(host_v);
    DeviceBuffer d_k = to_device_bf16(host_k);
    DeviceBuffer d_v = to_device_bf16(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_mapping = to_device(mapping);

    const std::size_t code_count =
        static_cast<std::size_t>(head_dim / 2) * kPage * kv_heads * kPhysicalPages;
    const std::size_t scale_count =
        static_cast<std::size_t>(groups) * kPage * kv_heads * kPhysicalPages;
    GuardedDeviceBuffer cache_k(code_count);
    GuardedDeviceBuffer cache_v(code_count);
    GuardedDeviceBuffer scale_k(scale_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer scale_v(scale_count * sizeof(std::uint16_t));
    std::vector<std::uint8_t> expected_k(code_count, 0x5au);
    std::vector<std::uint8_t> expected_v(code_count, 0xa5u);
    std::vector<std::uint16_t> expected_scale_k(scale_count, 0x3555u);
    std::vector<std::uint16_t> expected_scale_v(scale_count, 0x3aaau);
    cache_k.copy_from_host(expected_k.data(), expected_k.size());
    cache_v.copy_from_host(expected_v.data(), expected_v.size());
    scale_k.copy_from_host(expected_scale_k.data(), expected_scale_k.size() * sizeof(std::uint16_t));
    scale_v.copy_from_host(expected_scale_v.data(), expected_scale_v.size() * sizeof(std::uint16_t));

    PagedKVLayerView cache{
        .k_pages = Tensor(cache_k.data(), DType::U8,
                          {head_dim / 2, kPage, kv_heads, kPhysicalPages}),
        .v_pages = Tensor(cache_v.data(), DType::U8,
                          {head_dim / 2, kPage, kv_heads, kPhysicalPages}),
        .k_scale_pages = Tensor(scale_k.data(), DType::FP16,
                                {groups, kPage, kv_heads, kPhysicalPages}),
        .v_scale_pages = Tensor(scale_v.data(), DType::FP16,
                                {groups, kPage, kv_heads, kPhysicalPages}),
        .block_table = Tensor(d_mapping.p, DType::I32, {kLogicalPages}),
        .head_dim = head_dim,
        .num_kv_heads = kv_heads,
        .storage = KvCacheStorage::RK4V4E8,
    };

    Quality key_quality{};
    Quality value_quality{};
    for (int token = 0; token < tokens; ++token) {
        const int position = positions[static_cast<std::size_t>(token)];
        const int page = mapping[static_cast<std::size_t>(position / kPage)];
        for (int head = 0; head < kv_heads; ++head) {
            const int source = static_cast<int>(input_index(head_dim, kv_heads, 0, head, token));
            for (int group = 0; group < groups; ++group) {
                encode_group(host_k, source + group * kGroup, true, expected_k, expected_scale_k,
                             head_dim, kv_heads, group, head, position, page, key_quality);
                encode_group(host_v, source + group * kGroup, false, expected_v, expected_scale_v,
                             head_dim, kv_heads, group, head, position, page, value_quality);
            }
        }
    }

    Tensor k(d_k.p, DType::BF16, {head_dim, kv_heads, tokens});
    Tensor v(d_v.p, DType::BF16, {head_dim, kv_heads, tokens});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens});
    ops::kv_cache_append(k, v, position_tensor, cache, nullptr);
    cuda_synchronize();

    const std::string label = "gemma4 rk4v4-e8 D=" + std::to_string(head_dim) +
                              " Hkv=" + std::to_string(kv_heads) +
                              " T=" + std::to_string(tokens) + (zeros ? " zero" : "");
    int failures = 0;
    failures += verify_exact((label + " k codes").c_str(),
                             from_device<std::uint8_t>(cache_k.data(), code_count), expected_k);
    failures += verify_exact((label + " v codes").c_str(),
                             from_device<std::uint8_t>(cache_v.data(), code_count), expected_v);
    failures += verify_exact((label + " k scales").c_str(),
                             from_device<std::uint16_t>(scale_k.data(), scale_count),
                             expected_scale_k);
    failures += verify_exact((label + " v scales").c_str(),
                             from_device<std::uint16_t>(scale_v.data(), scale_count),
                             expected_scale_v);
    failures += cache_k.verify_guards((label + " k guards").c_str());
    failures += cache_v.verify_guards((label + " v guards").c_str());
    failures += scale_k.verify_guards((label + " k scale guards").c_str());
    failures += scale_v.verify_guards((label + " v scale guards").c_str());

    if (!zeros) {
        const auto check_quality = [&](const char* plane, const Quality& quality) {
            const double relative_l2 =
                std::sqrt(quality.squared_error / quality.squared_reference);
            const double cosine = quality.dot /
                std::sqrt(quality.squared_reference * quality.squared_decoded);
            std::cout << "E8_QUALITY geometry=D" << head_dim << "/H" << kv_heads
                      << " plane=" << plane << " rel_l2=" << relative_l2
                      << " cosine=" << cosine << '\n';
            return relative_l2 > 0.22 || cosine < 0.975 ? 1 : 0;
        };
        failures += check_quality("k", key_quality);
        failures += check_quality("v", value_quality);
    }
    return failures;
}

int validate_layouts() {
    int failures = 0;
    const auto local = paged_kv_storage_layout(KvCacheStorage::RK4V4E8, 256);
    const auto global = paged_kv_storage_layout(KvCacheStorage::RK4V4E8, 512);
    if (local.key.data_leading_extent != 128 || local.key.scale_leading_extent != 4 ||
        local.physical_bytes_per_token_head() != 272) {
        std::cerr << "FAIL gemma4 local E8 physical layout\n";
        ++failures;
    }
    if (global.key.data_leading_extent != 256 || global.key.scale_leading_extent != 8 ||
        global.physical_bytes_per_token_head() != 544) {
        std::cerr << "FAIL gemma4 global E8 physical layout\n";
        ++failures;
    }
    try {
        (void)paged_kv_storage_layout(KvCacheStorage::Int8Group64, 512);
        std::cerr << "FAIL unregistered D512 int8 layout accepted\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = validate_layouts();
    failures += run_case(256, 16, 7, false);
    failures += run_case(256, 16, 33, false);
    failures += run_case(512, 4, 7, false);
    failures += run_case(512, 4, 33, false);
    failures += run_case(512, 4, 1, true);
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_e8_kv_codec\n";
    return failures ? 1 : 0;
}
