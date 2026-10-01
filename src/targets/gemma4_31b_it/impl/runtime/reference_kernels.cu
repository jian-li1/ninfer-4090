#include "targets/gemma4_31b_it/impl/runtime/reference_kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::gemma4_31b_it::detail {
namespace {

__global__ void scale_bf16_kernel(__nv_bfloat16* values, std::int64_t count, float scale) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        values[i] = __float2bfloat16(__bfloat162float(values[i]) * scale);
    }
}

__device__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_sums[warp] = value; }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;
}

__global__ void prepare_qkv_kernel(
    const __nv_bfloat16* packed, const __nv_bfloat16* query_gain,
    const __nv_bfloat16* key_gain, int rows, int tokens, int head_dim, int kv_heads,
    float theta, int active_pairs, int first_position, __nv_bfloat16* query, __nv_bfloat16* key,
    __nv_bfloat16* value) {
    const int logical_row = static_cast<int>(blockIdx.x);
    const int token = logical_row / (32 + 2 * kv_heads);
    const int within = logical_row - token * (32 + 2 * kv_heads);
    const bool is_q = within < 32;
    const bool is_k = !is_q && within < 32 + kv_heads;
    const int head = is_q ? within : (is_k ? within - 32 : within - 32 - kv_heads);
    const int q_rows = 32 * head_dim;
    const int k_rows = kv_heads * head_dim;
    const bool shared_kv = rows == q_rows + k_rows;
    const int source_base = token * rows +
        (is_q ? 0 : is_k || shared_kv ? q_rows : q_rows + k_rows);
    const int output_base = (token * (is_q ? 32 : kv_heads) + head) * head_dim;

    float square = 0.0F;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float x = __bfloat162float(packed[source_base + head * head_dim + d]);
        square += x * x;
    }
    square = block_sum(square);
    __shared__ float inverse;
    if (threadIdx.x == 0) { inverse = rsqrtf(square / head_dim + 1.0e-6F); }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float x = __bfloat162float(packed[source_base + head * head_dim + d]);
        const float gain = is_q ? __bfloat162float(query_gain[d])
                                : is_k ? __bfloat162float(key_gain[d]) : 1.0F;
        const __nv_bfloat16 normalized_bf16 = __float2bfloat16(x * inverse * gain);
        if (!is_q && !is_k) {
            value[output_base + d] = normalized_bf16;
            continue;
        }
        const int half = head_dim / 2;
        const int pair = d < half ? d : d - half;
        float angle = 0.0F;
        if (pair < active_pairs) {
            const float inverse_frequency = powf(theta, -2.0F * pair / head_dim);
            angle = (first_position + token) * inverse_frequency;
        }
        const __nv_bfloat16 cosine = __float2bfloat16(cosf(angle));
        const __nv_bfloat16 sine = __float2bfloat16(sinf(angle));
        const int other_d = d < half ? d + half : d - half;
        const float normalized = __bfloat162float(normalized_bf16);
        const float other_x = __bfloat162float(
            packed[source_base + head * head_dim + other_d]);
        const float other_gain = is_q ? __bfloat162float(query_gain[other_d])
                                      : __bfloat162float(key_gain[other_d]);
        const float other = __bfloat162float(__float2bfloat16(other_x * inverse * other_gain));
        const float rotated = d < half ? -other : other;
        const __nv_bfloat16 result = __float2bfloat16(
            normalized * __bfloat162float(cosine) + rotated * __bfloat162float(sine));
        (is_q ? query : key)[output_base + d] = result;
    }
}

__global__ void attention_kernel(const __nv_bfloat16* query, const __nv_bfloat16* key,
                                 const __nv_bfloat16* value, int tokens, int head_dim,
                                 int kv_heads, int window, __nv_bfloat16* output) {
    extern __shared__ float probability[];
    const int query_token = static_cast<int>(blockIdx.x) / 32;
    const int query_head = static_cast<int>(blockIdx.x) - query_token * 32;
    const int kv_head = query_head / (32 / kv_heads);
    const int first_key = window > 0 ? max(0, query_token - window + 1) : 0;
    const auto* q = query + (query_token * 32 + query_head) * head_dim;
    for (int key_token = first_key + threadIdx.x; key_token <= query_token;
         key_token += blockDim.x) {
        const auto* k = key + (key_token * kv_heads + kv_head) * head_dim;
        float score = 0.0F;
        for (int d = 0; d < head_dim; ++d) {
            score += __bfloat162float(q[d]) * __bfloat162float(k[d]);
        }
        probability[key_token] = score;
    }
    __syncthreads();

    __shared__ float maximum;
    // Max reduction uses shared scores because token count, not CTA width, is the semantic axis.
    if (threadIdx.x == 0) {
        float result = -INFINITY;
        for (int key_token = first_key; key_token <= query_token; ++key_token) {
            result = fmaxf(result, probability[key_token]);
        }
        maximum = result;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float denominator = 0.0F;
        for (int key_token = first_key; key_token <= query_token; ++key_token) {
            const float e = expf(probability[key_token] - maximum);
            probability[key_token] = e;
            denominator += e;
        }
        for (int key_token = first_key; key_token <= query_token; ++key_token) {
            probability[key_token] = __bfloat162float(
                __float2bfloat16(probability[key_token] / denominator));
        }
    }
    __syncthreads();
    auto* out = output + (query_token * 32 + query_head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0F;
        for (int key_token = first_key; key_token <= query_token; ++key_token) {
            const auto* v = value + (key_token * kv_heads + kv_head) * head_dim;
            sum += probability[key_token] * __bfloat162float(v[d]);
        }
        out[d] = __float2bfloat16(sum);
    }
}

__device__ float gelu_tanh(float x) {
    constexpr float coefficient = 0.7978845608028654F;
    return 0.5F * x * (1.0F + tanhf(coefficient * (x + 0.044715F * x * x * x)));
}

__global__ void gelu_mul_kernel(const __nv_bfloat16* gate_up, int intermediate, int tokens,
                                __nv_bfloat16* product) {
    const std::int64_t count = static_cast<std::int64_t>(intermediate) * tokens;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int token = static_cast<int>(i / intermediate);
        const int row = static_cast<int>(i - static_cast<std::int64_t>(token) * intermediate);
        const std::int64_t base = static_cast<std::int64_t>(token) * 2 * intermediate;
        const float gate = __bfloat162float(gate_up[base + row]);
        const float activated = __bfloat162float(__float2bfloat16(gelu_tanh(gate)));
        const float up = __bfloat162float(gate_up[base + intermediate + row]);
        product[i] = __float2bfloat16(activated * up);
    }
}

__global__ void add_scaled_kernel(const __nv_bfloat16* update, const __nv_bfloat16* scalar,
                                  __nv_bfloat16* residual, std::int64_t count) {
    const float scale = __bfloat162float(scalar[0]);
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const __nv_bfloat16 sum = __float2bfloat16(
            __bfloat162float(residual[i]) + __bfloat162float(update[i]));
        residual[i] = __float2bfloat16(__bfloat162float(sum) * scale);
    }
}

__global__ void fp8_logits_kernel(const __nv_bfloat16* hidden, const std::uint8_t* codes,
                                  const __nv_bfloat16* scales, int vocabulary, int width,
                                  float softcap, float* logits) {
    const int row = static_cast<int>(blockIdx.x);
    float partial = 0.0F;
    const auto* code_row = codes + static_cast<std::int64_t>(row) * width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        const __nv_fp8_e4m3 code = *reinterpret_cast<const __nv_fp8_e4m3*>(code_row + d);
        partial += __bfloat162float(hidden[d]) * static_cast<float>(code);
    }
    partial = block_sum(partial);
    if (threadIdx.x == 0) {
        const float dot = __bfloat162float(__float2bfloat16(
            partial * __bfloat162float(scales[row])));
        const float divided = __bfloat162float(__float2bfloat16(dot / softcap));
        const float softened = __bfloat162float(__float2bfloat16(tanhf(divided)));
        logits[row] = __bfloat162float(__float2bfloat16(softcap * softened));
    }
}

__global__ void bf16_to_fp32_kernel(const __nv_bfloat16* input, float* output,
                                    std::int64_t count) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        output[i] = __bfloat162float(input[i]);
    }
}

int grid_for(std::int64_t count) {
    return static_cast<int>(std::min<std::int64_t>(16384, (count + 255) / 256));
}

} // namespace

void scale_embedding(Tensor& hidden, float scale, cudaStream_t stream) {
    scale_bf16_kernel<<<grid_for(hidden.numel()), 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hidden.data), hidden.numel(), scale);
    CUDA_CHECK(cudaGetLastError());
}

void prepare_qkv(const Tensor& packed, const Tensor& query_gain, const Tensor& key_gain,
                 std::int32_t head_dim, std::int32_t kv_heads, float theta,
                 std::int32_t active_pairs, std::int32_t first_position, Tensor& query, Tensor& key, Tensor& value,
                 cudaStream_t stream) {
    const int tokens = packed.ne[1];
    prepare_qkv_kernel<<<tokens * (32 + 2 * kv_heads), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed.data),
        static_cast<const __nv_bfloat16*>(query_gain.data),
        static_cast<const __nv_bfloat16*>(key_gain.data), packed.ne[0], tokens, head_dim,
        kv_heads, theta, active_pairs, first_position, static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

void reference_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                         std::int32_t window, Tensor& output, cudaStream_t stream) {
    const int tokens = query.ne[2];
    const int head_dim = query.ne[0];
    const int kv_heads = key.ne[1];
    attention_kernel<<<tokens * 32, 256, static_cast<std::size_t>(tokens) * sizeof(float), stream>>>(
        static_cast<const __nv_bfloat16*>(query.data),
        static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(value.data), tokens, head_dim, kv_heads, window,
        static_cast<__nv_bfloat16*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}

void gelu_tanh_mul_packed(const Tensor& gate_up, Tensor& product, cudaStream_t stream) {
    gelu_mul_kernel<<<grid_for(product.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up.data), product.ne[0], product.ne[1],
        static_cast<__nv_bfloat16*>(product.data));
    CUDA_CHECK(cudaGetLastError());
}

void add_scaled(const Tensor& update, const Tensor& scalar, Tensor& residual,
                cudaStream_t stream) {
    add_scaled_kernel<<<grid_for(residual.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(update.data),
        static_cast<const __nv_bfloat16*>(scalar.data),
        static_cast<__nv_bfloat16*>(residual.data), residual.numel());
    CUDA_CHECK(cudaGetLastError());
}

void fp8_tied_logits(const Tensor& last_hidden, const Weight& embedding, float softcap,
                     Tensor& logits, cudaStream_t stream) {
    fp8_logits_kernel<<<embedding.n, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(last_hidden.data),
        static_cast<const std::uint8_t*>(embedding.qdata),
        static_cast<const __nv_bfloat16*>(embedding.scales), embedding.n, embedding.k,
        softcap, static_cast<float*>(logits.data));
    CUDA_CHECK(cudaGetLastError());
}

void bf16_to_fp32(const Tensor& input, Tensor& output, cudaStream_t stream) {
    bf16_to_fp32_kernel<<<grid_for(input.numel()), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<float*>(output.data),
        input.numel());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::gemma4_31b_it::detail
