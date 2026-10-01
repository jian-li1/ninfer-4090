#include "ninfer/ops/softmax_attention.h"

#include "core/paged_kv_storage.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD = 256;
constexpr int kQHeads = 32;
constexpr int kKVHeads = 16;
constexpr int kWindow = 1024;
constexpr int kPage = 64;
constexpr int kRingPages = 17;
constexpr int kGroups = 4;
constexpr float kScale = 1.0F;
constexpr ops::AttentionHeadGeometry kGeometry{kD, kQHeads, kKVHeads};
constexpr ReductionCriterion kCriterion{3.5e-3, 8.0e-4, 8.0e-3};

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

void hadamard64(float* values) {
    for (int stride = 1; stride < 64; stride <<= 1) {
        for (int base = 0; base < 64; base += 2 * stride) {
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
    for (int d = 0; d < 64; ++d) values[d] *= 0.125F;
}

std::size_t cache_index(int leading, int physical_page, int head, int position, int d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(leading) *
               (static_cast<std::size_t>(position & 63) +
                64ULL * (static_cast<std::size_t>(head) +
                         static_cast<std::size_t>(kKVHeads) * physical_page));
}

std::size_t input_index(int heads, int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(heads) * static_cast<std::size_t>(token));
}

std::vector<float> decode_row(const std::vector<std::uint8_t>& codes,
                              const std::vector<std::uint16_t>& scales,
                              const std::vector<std::int32_t>& table, int head, int position) {
    const int physical_page = table[static_cast<std::size_t>(position / kPage) % kRingPages];
    std::vector<float> row(kD);
    for (int group = 0; group < kGroups; ++group) {
        const float scale = f16_bits_to_f32(
            scales[cache_index(kGroups, physical_page, head, position, group)]);
        for (int d = 0; d < 64; ++d) {
            const int absolute_d = group * 64 + d;
            const std::uint8_t packed =
                codes[cache_index(kD / 2, physical_page, head, position, absolute_d / 2)];
            const unsigned nibble = (absolute_d & 1) != 0 ? packed >> 4 : packed & 0x0fu;
            const int code = static_cast<int>(nibble ^ 8u) - 8;
            row[static_cast<std::size_t>(absolute_d)] = static_cast<float>(code) * scale;
        }
        hadamard64(row.data() + group * 64);
    }
    return row;
}

std::vector<double> oracle_token(const std::vector<float>& q, int query_token, int position,
                                 const std::vector<std::uint8_t>& key_codes,
                                 const std::vector<std::uint8_t>& value_codes,
                                 const std::vector<std::uint16_t>& key_scales,
                                 const std::vector<std::uint16_t>& value_scales,
                                 const std::vector<std::int32_t>& table) {
    std::vector<double> expected(static_cast<std::size_t>(kD) * kQHeads);
    const int first_key = std::max(0, position - kWindow + 1);
    for (int q_head = 0; q_head < kQHeads; ++q_head) {
        const int kv_head = q_head / 2;
        std::vector<double> scores(static_cast<std::size_t>(position - first_key + 1));
        double maximum = -std::numeric_limits<double>::infinity();
        for (int key = first_key; key <= position; ++key) {
            const auto decoded = decode_row(key_codes, key_scales, table, kv_head, key);
            double score = 0.0;
            for (int d = 0; d < kD; ++d) {
                score += static_cast<double>(q[input_index(kQHeads, d, q_head, query_token)]) *
                         decoded[static_cast<std::size_t>(d)];
            }
            score *= kScale;
            scores[static_cast<std::size_t>(key - first_key)] = score;
            maximum = std::max(maximum, score);
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int key = first_key; key <= position; ++key) {
            const double probability =
                scores[static_cast<std::size_t>(key - first_key)] / denominator;
            const auto decoded = decode_row(value_codes, value_scales, table, kv_head, key);
            for (int d = 0; d < kD; ++d) {
                expected[static_cast<std::size_t>(q_head) * kD + d] +=
                    probability * decoded[static_cast<std::size_t>(d)];
            }
        }
    }
    return expected;
}

struct Fixture {
    std::vector<std::int32_t> table;
    DeviceBuffer d_table;
    GuardedDeviceBuffer key_codes;
    GuardedDeviceBuffer value_codes;
    GuardedDeviceBuffer key_scales;
    GuardedDeviceBuffer value_scales;
    PagedKVLayerView cache;

    Fixture()
        : table(kRingPages), d_table(kRingPages * sizeof(std::int32_t)),
          key_codes(static_cast<std::size_t>(kD / 2) * kPage * kKVHeads * kRingPages),
          value_codes(key_codes.bytes()),
          key_scales(static_cast<std::size_t>(kGroups) * kPage * kKVHeads * kRingPages * 2),
          value_scales(key_scales.bytes()) {
        for (int slot = 0; slot < kRingPages; ++slot) table[slot] = (slot * 7) % kRingPages;
        d_table.copy_from_host(table.data(), d_table.bytes);
        key_codes.fill(0);
        value_codes.fill(0);
        key_scales.fill(0);
        value_scales.fill(0);
        cache = {
            .k_pages = Tensor(key_codes.data(), DType::U8,
                              {kD / 2, kPage, kKVHeads, kRingPages}),
            .v_pages = Tensor(value_codes.data(), DType::U8,
                              {kD / 2, kPage, kKVHeads, kRingPages}),
            .k_scale_pages = Tensor(key_scales.data(), DType::FP16,
                                    {kGroups, kPage, kKVHeads, kRingPages}),
            .v_scale_pages = Tensor(value_scales.data(), DType::FP16,
                                    {kGroups, kPage, kKVHeads, kRingPages}),
            .block_table = Tensor(d_table.p, DType::I32, {kRingPages}),
            .head_dim = kD,
            .num_kv_heads = kKVHeads,
            .storage = KvCacheStorage::RK4V4E8,
        };
    }
};

std::vector<float> make_values(int heads, int first, int tokens, float phase) {
    std::vector<float> values(static_cast<std::size_t>(kD) * heads * tokens);
    for (int token = 0; token < tokens; ++token) {
        const int position = first + token;
        for (int head = 0; head < heads; ++head) {
            for (int d = 0; d < kD; ++d) {
                const float value =
                    std::sin(phase + 0.017F * d + 0.071F * head + 0.013F * position) *
                    (0.15F + 0.01F * static_cast<float>((d + head + position) % 23));
                values[input_index(heads, d, head, token)] = value;
            }
        }
    }
    round_to_bf16(values);
    return values;
}

int run_segment(Fixture& fixture, int first, int tokens, bool check) {
    std::vector<float> q = make_values(kQHeads, first, tokens, 0.1F);
    std::vector<float> k = make_values(kKVHeads, first, tokens, 0.7F);
    std::vector<float> v = make_values(kKVHeads, first, tokens, 1.3F);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = first + token;
    }
    DeviceBuffer d_q = to_device_bf16(q);
    DeviceBuffer d_k = to_device_bf16(k);
    DeviceBuffer d_v = to_device_bf16(v);
    DeviceBuffer d_positions = to_device(positions);
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(kD) * kQHeads * tokens * 2);
    Tensor tq(d_q.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tk(d_k.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tv(d_v.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tp(d_positions.p, DType::I32, {tokens});
    Tensor out(d_out.data(), DType::BF16, {kD, kQHeads, tokens});
    ops::causal_sliding_softmax_attention(tq, tk, tv, tp, kGeometry, kWindow, kScale,
                                          fixture.cache, out, nullptr);
    cuda_synchronize();

    int failures = d_out.verify_guards("gemma4 sliding output guards");
    if (!check) return failures;
    const auto host_key_codes =
        from_device<std::uint8_t>(fixture.key_codes.data(), fixture.key_codes.bytes());
    const auto host_value_codes =
        from_device<std::uint8_t>(fixture.value_codes.data(), fixture.value_codes.bytes());
    const auto host_key_scales = from_device<std::uint16_t>(
        fixture.key_scales.data(), fixture.key_scales.bytes() / sizeof(std::uint16_t));
    const auto host_value_scales = from_device<std::uint16_t>(
        fixture.value_scales.data(), fixture.value_scales.bytes() / sizeof(std::uint16_t));
    const auto got = from_device_bf16(
        d_out.data(), static_cast<std::size_t>(kD) * kQHeads * tokens);
    const auto check_token = [&](int token) {
        const auto expected = oracle_token(q, token, first + token, host_key_codes,
                                           host_value_codes, host_key_scales,
                                           host_value_scales, fixture.table);
        std::vector<double> actual(static_cast<std::size_t>(kD) * kQHeads);
        const std::size_t begin = static_cast<std::size_t>(token) * kD * kQHeads;
        std::copy_n(got.begin() + static_cast<std::ptrdiff_t>(begin), actual.size(),
                    actual.begin());
        const std::string label = "gemma4 sliding P=" + std::to_string(first + token);
        return verify_reduction(label, actual, expected, kCriterion);
    };
    failures += check_token(0);
    if (tokens > 1) failures += check_token(tokens - 1);
    return failures;
}

int run_sequence(int tokens) {
    Fixture fixture;
    int failures = 0;
    for (int begin = 0; begin < tokens; begin += kPage) {
        const int count = std::min(kPage, tokens - begin);
        failures += run_segment(fixture, begin, count, begin + count == tokens);
    }
    std::vector<float> q = make_values(kQHeads, tokens, 1, 0.1F);
    const std::vector<std::int32_t> last_key{tokens - 1};
    DeviceBuffer d_q = to_device_bf16(q);
    DeviceBuffer d_last_key = to_device(last_key);
    GuardedDeviceBuffer d_shared_out(static_cast<std::size_t>(kD) * kQHeads * 2);
    Tensor tq(d_q.p, DType::BF16, {kD, kQHeads, 1});
    Tensor tp(d_last_key.p, DType::I32, {1});
    Tensor out(d_shared_out.data(), DType::BF16, {kD, kQHeads, 1});
    ops::shared_kv_sliding_softmax_attention(tq, tp, kGeometry, kWindow, kScale,
                                             fixture.cache, out, nullptr);
    cuda_synchronize();
    failures += d_shared_out.verify_guards("gemma4 shared sliding output guards");
    const auto host_key_codes =
        from_device<std::uint8_t>(fixture.key_codes.data(), fixture.key_codes.bytes());
    const auto host_value_codes =
        from_device<std::uint8_t>(fixture.value_codes.data(), fixture.value_codes.bytes());
    const auto host_key_scales = from_device<std::uint16_t>(
        fixture.key_scales.data(), fixture.key_scales.bytes() / sizeof(std::uint16_t));
    const auto host_value_scales = from_device<std::uint16_t>(
        fixture.value_scales.data(), fixture.value_scales.bytes() / sizeof(std::uint16_t));
    const auto expected = oracle_token(q, 0, tokens - 1, host_key_codes, host_value_codes,
                                       host_key_scales, host_value_scales, fixture.table);
    const auto got = from_device_bf16(d_shared_out.data(), static_cast<std::size_t>(kD) * kQHeads);
    failures += verify_reduction("gemma4 shared sliding T=" + std::to_string(tokens), got,
                                 expected, kCriterion);
    const std::string label = "gemma4 sliding T=" + std::to_string(tokens);
    failures += fixture.key_codes.verify_guards(label + " K guards");
    failures += fixture.value_codes.verify_guards(label + " V guards");
    failures += fixture.key_scales.verify_guards(label + " K scale guards");
    failures += fixture.value_scales.verify_guards(label + " V scale guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (int tokens : {1, 2, 127, 128, 1023, 1024, 1025, 2048}) {
        failures += run_sequence(tokens);
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_sliding_attention\n";
    return failures ? 1 : 0;
}
