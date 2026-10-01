#include "ninfer/ops/softmax_attention.h"

#include "core/paged_kv_storage.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
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

constexpr int kD = 512;
constexpr int kQHeads = 32;
constexpr int kKVHeads = 4;
constexpr int kGroups = 8;
constexpr int kPage = 64;
constexpr int kCapacity = 131072;
constexpr float kScale = 1.0F;
constexpr std::uint16_t kKeyScaleBits = 0x2400;   // 2^-6
constexpr std::uint16_t kValueScaleBits = 0x2000; // 2^-7
constexpr ops::AttentionHeadGeometry kGeometry{kD, kQHeads, kKVHeads};
constexpr ReductionCriterion kCriterion{2.5e-3, 3.0e-4, 3.0e-3};

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

std::uint8_t pack_i4(int lo, int hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

int unpack_i4(std::uint8_t packed, bool high) {
    const unsigned nibble = high ? packed >> 4 : packed & 0x0fu;
    return static_cast<int>(nibble ^ 8u) - 8;
}

void hadamard64(double* values) {
    for (int stride = 1; stride < 64; stride <<= 1) {
        for (int base = 0; base < 64; base += 2 * stride) {
            for (int offset = 0; offset < stride; ++offset) {
                const int lo = base + offset;
                const int hi = lo + stride;
                const double a = values[lo];
                const double b = values[hi];
                values[lo] = a + b;
                values[hi] = a - b;
            }
        }
    }
    for (int d = 0; d < 64; ++d) values[d] *= 0.125;
}

std::size_t cache_index(int leading, int physical_page, int head, int position, int d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(leading) *
               (static_cast<std::size_t>(position & (kPage - 1)) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kKVHeads) * physical_page));
}

int key_code(int position, int head, int d) {
    constexpr int patterns[4][8] = {
        {1, -1, 2, -2, 1, -1, 0, 0},
        {2, 0, -1, -1, 1, 1, -2, 0},
        {-2, 1, 1, 0, -1, 2, -1, 0},
        {0, 1, -2, 1, 2, -1, -1, 0},
    };
    const int block = d / 8;
    const int pattern = (position + 3 * head + block) & 3;
    const int sign = ((position / 7 + head + block / 3) & 1) == 0 ? 1 : -1;
    return sign * patterns[pattern][d & 7];
}

int value_code(int position, int head, int d) {
    return ((position * 3 + head * 5 + d * 7) % 7) - 1;
}

struct Fixture {
    int pages = kCapacity / kPage;
    std::vector<std::int32_t> table;
    std::vector<std::uint8_t> key_codes;
    std::vector<std::uint8_t> value_codes;
    std::vector<std::uint16_t> key_scales;
    std::vector<std::uint16_t> value_scales;
    DeviceBuffer d_table;
    GuardedDeviceBuffer d_key_codes;
    GuardedDeviceBuffer d_value_codes;
    GuardedDeviceBuffer d_key_scales;
    GuardedDeviceBuffer d_value_scales;
    PagedKVLayerView cache;

    Fixture()
        : table(static_cast<std::size_t>(pages)),
          key_codes(static_cast<std::size_t>(kD / 2) * kPage * kKVHeads * pages),
          value_codes(key_codes.size()),
          key_scales(static_cast<std::size_t>(kGroups) * kPage * kKVHeads * pages,
                     kKeyScaleBits),
          value_scales(key_scales.size(), kValueScaleBits),
          d_table(table.size() * sizeof(std::int32_t)), d_key_codes(key_codes.size()),
          d_value_codes(value_codes.size()),
          d_key_scales(key_scales.size() * sizeof(std::uint16_t)),
          d_value_scales(value_scales.size() * sizeof(std::uint16_t)) {
        for (int logical = 0; logical < pages; ++logical) {
            table[static_cast<std::size_t>(logical)] = (logical * 5) % pages;
        }
        for (int position = 0; position < kCapacity; ++position) {
            const int physical_page = table[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kKVHeads; ++head) {
                for (int packed_d = 0; packed_d < kD / 2; ++packed_d) {
                    const int d0 = 2 * packed_d;
                    const std::size_t index =
                        cache_index(kD / 2, physical_page, head, position, packed_d);
                    key_codes[index] =
                        pack_i4(key_code(position, head, d0), key_code(position, head, d0 + 1));
                    value_codes[index] = pack_i4(value_code(position, head, d0),
                                                 value_code(position, head, d0 + 1));
                }
            }
        }
        d_table.copy_from_host(table.data(), d_table.bytes);
        d_key_codes.copy_from_host(key_codes.data(), key_codes.size());
        d_value_codes.copy_from_host(value_codes.data(), value_codes.size());
        d_key_scales.copy_from_host(key_scales.data(),
                                    key_scales.size() * sizeof(std::uint16_t));
        d_value_scales.copy_from_host(value_scales.data(),
                                      value_scales.size() * sizeof(std::uint16_t));
        cache = {
            .k_pages = Tensor(d_key_codes.data(), DType::U8,
                              {kD / 2, kPage, kKVHeads, pages}),
            .v_pages = Tensor(d_value_codes.data(), DType::U8,
                              {kD / 2, kPage, kKVHeads, pages}),
            .k_scale_pages = Tensor(d_key_scales.data(), DType::FP16,
                                    {kGroups, kPage, kKVHeads, pages}),
            .v_scale_pages = Tensor(d_value_scales.data(), DType::FP16,
                                    {kGroups, kPage, kKVHeads, pages}),
            .block_table = Tensor(d_table.p, DType::I32, {pages}),
            .head_dim = kD,
            .num_kv_heads = kKVHeads,
            .storage = KvCacheStorage::RK4V4E8,
        };
    }

    void mark_zero(int first, int tokens) {
        for (int position = first; position < first + tokens; ++position) {
            const int physical_page = table[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kKVHeads; ++head) {
                for (int packed_d = 0; packed_d < kD / 2; ++packed_d) {
                    const std::size_t index =
                        cache_index(kD / 2, physical_page, head, position, packed_d);
                    key_codes[index] = 0;
                    value_codes[index] = 0;
                }
                for (int group = 0; group < kGroups; ++group) {
                    const std::size_t index =
                        cache_index(kGroups, physical_page, head, position, group);
                    key_scales[index] = 0;
                    value_scales[index] = 0;
                }
            }
        }
    }

    double decoded(const std::vector<std::uint8_t>& codes,
                   const std::vector<std::uint16_t>& scales, int head, int position, int d) const {
        const int physical_page = table[static_cast<std::size_t>(position / kPage)];
        const std::uint8_t packed =
            codes[cache_index(kD / 2, physical_page, head, position, d / 2)];
        const float scale = f16_bits_to_f32(
            scales[cache_index(kGroups, physical_page, head, position, d / 64)]);
        return static_cast<double>(unpack_i4(packed, (d & 1) != 0)) * scale;
    }
};

std::vector<float> make_query(int first, int tokens) {
    std::vector<float> query(static_cast<std::size_t>(kD) * kQHeads * tokens);
    for (int token = 0; token < tokens; ++token) {
        const int position = first + token;
        for (int head = 0; head < kQHeads; ++head) {
            for (int d = 0; d < kD; ++d) {
                const std::size_t index = static_cast<std::size_t>(d) +
                    static_cast<std::size_t>(kD) *
                        (static_cast<std::size_t>(head) +
                         static_cast<std::size_t>(kQHeads) * token);
                query[index] =
                    std::sin(0.01F * d + 0.07F * head + 0.003F * position) * 0.035F;
            }
        }
    }
    round_to_bf16(query);
    return query;
}

std::vector<double> oracle_head(const Fixture& fixture, const std::vector<float>& query,
                                int query_token, int position, int q_head) {
    const int kv_head = q_head / 8;
    std::array<double, kD> rotated_q{};
    const std::size_t query_base = static_cast<std::size_t>(kD) *
        (static_cast<std::size_t>(q_head) + static_cast<std::size_t>(kQHeads) * query_token);
    for (int d = 0; d < kD; ++d) rotated_q[static_cast<std::size_t>(d)] = query[query_base + d];
    for (int group = 0; group < kGroups; ++group) {
        hadamard64(rotated_q.data() + group * 64);
    }

    std::vector<double> scores(static_cast<std::size_t>(position + 1));
    double maximum = -std::numeric_limits<double>::infinity();
    for (int key = 0; key <= position; ++key) {
        double score = 0.0;
        for (int d = 0; d < kD; ++d) {
            score += rotated_q[static_cast<std::size_t>(d)] *
                     fixture.decoded(fixture.key_codes, fixture.key_scales, kv_head, key, d);
        }
        scores[static_cast<std::size_t>(key)] = score;
        maximum = std::max(maximum, score);
    }
    double denominator = 0.0;
    for (double& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
    }
    std::vector<double> result(kD);
    for (int key = 0; key <= position; ++key) {
        const double probability = scores[static_cast<std::size_t>(key)] / denominator;
        for (int d = 0; d < kD; ++d) {
            result[static_cast<std::size_t>(d)] +=
                probability * fixture.decoded(fixture.value_codes, fixture.value_scales,
                                              kv_head, key, d);
        }
    }
    for (int group = 0; group < kGroups; ++group) {
        hadamard64(result.data() + group * 64);
    }
    return result;
}

int run_case(Fixture& fixture, int first, int tokens, const std::vector<int>& query_tokens,
             const std::vector<int>& heads) {
    std::vector<float> query = make_query(first, tokens);
    std::vector<float> zeros(static_cast<std::size_t>(kD) * kKVHeads * tokens, 0.0F);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) positions[token] = first + token;
    DeviceBuffer d_query = to_device_bf16(query);
    DeviceBuffer d_key = to_device_bf16(zeros);
    DeviceBuffer d_value = to_device_bf16(zeros);
    DeviceBuffer d_positions = to_device(positions);
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(kD) * kQHeads * tokens * 2);
    Tensor tq(d_query.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tk(d_key.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tv(d_value.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tp(d_positions.p, DType::I32, {tokens});
    Tensor out(d_out.data(), DType::BF16, {kD, kQHeads, tokens});
    const ops::CausalAttentionExecutionEnvelope envelope{
        static_cast<std::uint32_t>(first + 1), static_cast<std::uint32_t>(first + tokens)};
    const std::size_t workspace_bytes = ops::causal_full_softmax_attention_workspace_capacity_bytes(
        kGeometry, KvCacheStorage::RK4V4E8, envelope, tokens, tokens);
    DeviceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
    ops::causal_full_softmax_attention(tq, tk, tv, tp, kGeometry, kScale, fixture.cache,
                                       envelope, workspace, out, nullptr);
    cuda_synchronize();
    fixture.mark_zero(first, tokens);

    int failures = d_out.verify_guards("gemma4 global output guards");
    const auto got =
        from_device_bf16(d_out.data(), static_cast<std::size_t>(kD) * kQHeads * tokens);
    for (int query_token : query_tokens) {
        std::vector<double> actual;
        std::vector<double> expected;
        actual.reserve(static_cast<std::size_t>(heads.size()) * kD);
        expected.reserve(actual.capacity());
        for (int head : heads) {
            const auto reference =
                oracle_head(fixture, query, query_token, first + query_token, head);
            const std::size_t base = static_cast<std::size_t>(kD) *
                (static_cast<std::size_t>(head) +
                 static_cast<std::size_t>(kQHeads) * query_token);
            for (int d = 0; d < kD; ++d) {
                actual.push_back(got[base + d]);
                expected.push_back(reference[static_cast<std::size_t>(d)]);
            }
        }
        const std::string label = "gemma4 global P=" +
                                  std::to_string(first + query_token) +
                                  " T=" + std::to_string(tokens);
        failures += verify_reduction(label, actual, expected, kCriterion);
    }
    return failures;
}

std::vector<int> all_heads() {
    std::vector<int> heads(kQHeads);
    for (int head = 0; head < kQHeads; ++head) heads[static_cast<std::size_t>(head)] = head;
    return heads;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    Fixture fixture;
    int failures = 0;
    failures += run_case(fixture, 64, 64, {0, 63}, all_heads());
    failures += run_case(fixture, 960, 64, {0, 63}, all_heads());
    failures += run_case(fixture, 1023, 1, {0}, all_heads());
    failures += run_case(fixture, 4092, 4, {0, 3}, all_heads());
    failures += run_case(fixture, 4093, 3, {0, 2}, all_heads());
    failures += run_case(fixture, 4094, 2, {0, 1}, all_heads());
    failures += run_case(fixture, 4095, 1, {0}, all_heads());
    failures += run_case(fixture, 16383, 1, {0}, {0, 1, 7, 8, 15, 16, 23, 24, 31});
    failures += run_case(fixture, 131071, 1, {0}, {0, 8, 16, 24});
    failures += fixture.d_key_codes.verify_guards("gemma4 global K guards");
    failures += fixture.d_value_codes.verify_guards("gemma4 global V guards");
    failures += fixture.d_key_scales.verify_guards("gemma4 global K scale guards");
    failures += fixture.d_value_scales.verify_guards("gemma4 global V scale guards");
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_global_attention\n";
    return failures ? 1 : 0;
}
