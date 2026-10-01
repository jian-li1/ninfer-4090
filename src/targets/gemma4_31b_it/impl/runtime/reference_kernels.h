#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::gemma4_31b_it::detail {

void scale_embedding(Tensor& hidden, float scale, cudaStream_t stream);

void prepare_qkv(const Tensor& packed, const Tensor& query_gain, const Tensor& key_gain,
                 std::int32_t head_dim, std::int32_t kv_heads, float theta,
                 std::int32_t active_pairs, const Tensor& positions, Tensor& query,
                 Tensor& key, Tensor& value, cudaStream_t stream);

void prepare_query(const Tensor& projected, const Tensor& query_gain,
                   std::int32_t head_dim, float theta, std::int32_t active_pairs,
                   const Tensor& positions, Tensor& query, cudaStream_t stream);

void reference_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                         std::int32_t window, Tensor& output, cudaStream_t stream);

void gelu_tanh_mul_packed(const Tensor& gate_up, Tensor& product, cudaStream_t stream);

void add_scaled(const Tensor& update, const Tensor& scalar, Tensor& residual,
                cudaStream_t stream);

void fp8_tied_logits(const Tensor& last_hidden, const Weight& embedding, float softcap,
                     Tensor& logits, cudaStream_t stream);

void fp8_tied_logits_raw(const Tensor& hidden, const Weight& embedding, Tensor& logits,
                         cudaStream_t stream);

void bf16_to_fp32(const Tensor& input, Tensor& output, cudaStream_t stream);

} // namespace ninfer::targets::gemma4_31b_it::detail
