#pragma once

#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::gemma4 {

struct MlpWeights {
    Weight gate;
    Weight up;
    Weight down;
};

struct AttentionWeights {
    Weight query;
    Weight key;
    // Full-attention layers omit this view: their projected K source also supplies V.
    std::optional<Weight> value;
    Weight output;
    Tensor query_norm;
    Tensor key_norm;
};

struct DecoderLayerView {
    Tensor input_norm;
    AttentionWeights attention;
    Tensor post_attention_norm;
    Tensor pre_feedforward_norm;
    MlpWeights mlp;
    Tensor post_feedforward_norm;
};

struct AssistantAttentionWeights {
    Weight query;
    Weight output;
    Tensor query_norm;
};

struct AssistantLayerView {
    Tensor input_norm;
    AssistantAttentionWeights attention;
    Tensor post_attention_norm;
    Tensor pre_feedforward_norm;
    MlpWeights mlp;
    Tensor post_feedforward_norm;
    std::uint32_t target_kv_layer = 0;
};

template <std::size_t LayerCount>
struct AssistantModelView {
    Weight embedding;
    Weight input_projection;
    std::array<AssistantLayerView, LayerCount> layers;
    Tensor final_norm;
    Weight output_projection;
};

template <std::size_t LayerCount, std::size_t AssistantLayerCount>
struct ModelView {
    Weight embedding;
    std::array<DecoderLayerView, LayerCount> layers;
    Tensor final_norm;
    // The output head is tied to embedding; it is not a second physical allocation.
    Weight output_head;
    std::optional<AssistantModelView<AssistantLayerCount>> assistant;
};

} // namespace ninfer::targets::gemma4
