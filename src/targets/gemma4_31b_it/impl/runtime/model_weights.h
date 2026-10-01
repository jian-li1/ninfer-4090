#pragma once

#include <ninfer/targets/gemma4_31b_it/config.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "targets/gemma4_31b_it/impl/load/bindings.h"

#include <array>
#include <cstddef>
#include <optional>

namespace ninfer::targets::gemma4_31b_it::detail {

struct AttentionWeights {
    Weight input;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
};

struct LayerWeights {
    Tensor input_norm;
    AttentionWeights attention;
    Tensor post_attention_norm;
    Tensor pre_feedforward_norm;
    Weight gate_up;
    Weight down;
    Tensor post_feedforward_norm;
    Tensor layer_scalar;
};

struct ModelWeights {
    Weight embedding;
    std::array<LayerWeights, TextConfig::layers> layers;
    Tensor final_norm;
};

struct AssistantAttentionWeights {
    Weight query;
    Tensor query_norm;
    Weight output;
};

struct AssistantLayerWeights {
    Tensor input_norm;
    AssistantAttentionWeights attention;
    Tensor post_attention_norm;
    Tensor pre_feedforward_norm;
    Weight gate_up;
    Weight down;
    Tensor post_feedforward_norm;
    Tensor layer_scalar;
};

struct AssistantWeights {
    Weight output_head;
    Weight input_projection;
    std::array<AssistantLayerWeights, AssistantConfig::layers> layers;
    Tensor final_norm;
    Weight output_projection;
};

inline ModelWeights load_weights(const artifact::MaterializedArtifact& artifact,
                                 const BindingPlan& plan) {
    using artifact::NumericFormat;
    ModelWeights out;
    out.embedding = artifact::materialized_weight(
        artifact, plan.token_embedding, NumericFormat::FP8_E4M3FN_ROW_BF16S,
        TextConfig::vocabulary, TextConfig::hidden);
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const bool full = TextConfig::is_full_attention(layer);
        const auto& source = plan.layers[layer];
        auto& target = out.layers[layer];
        const int head_dim = full ? TextConfig::full_head_dim : TextConfig::sliding_head_dim;
        const int input_rows = full ? 18432 : 16384;
        const int output_columns = full ? 16384 : 8192;
        target.input_norm = artifact::materialized_tensor(
            artifact, source.input_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.attention.input = artifact::materialized_weight(
            artifact, source.attention.input_projection, NumericFormat::Q4G64_F16S,
            input_rows, TextConfig::hidden);
        target.attention.query_norm = artifact::materialized_tensor(
            artifact, source.attention.query_norm, NumericFormat::BF16, {head_dim});
        target.attention.key_norm = artifact::materialized_tensor(
            artifact, source.attention.key_norm, NumericFormat::BF16, {head_dim});
        target.attention.output = artifact::materialized_weight(
            artifact, source.attention.output, NumericFormat::Q4G64_F16S,
            TextConfig::hidden, output_columns);
        target.post_attention_norm = artifact::materialized_tensor(
            artifact, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.pre_feedforward_norm = artifact::materialized_tensor(
            artifact, source.pre_feedforward_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.gate_up = artifact::materialized_weight(
            artifact, source.mlp.gate_up, NumericFormat::Q4G64_F16S,
            2 * TextConfig::intermediate, TextConfig::hidden);
        target.down = artifact::materialized_weight(
            artifact, source.mlp.down, NumericFormat::Q4G64_F16S,
            TextConfig::hidden, TextConfig::intermediate);
        target.post_feedforward_norm = artifact::materialized_tensor(
            artifact, source.post_feedforward_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.layer_scalar = artifact::materialized_tensor(
            artifact, source.layer_scalar, NumericFormat::BF16, {1});
    }
    out.final_norm = artifact::materialized_tensor(
        artifact, plan.final_norm, NumericFormat::BF16, {TextConfig::hidden});
    return out;
}

inline AssistantWeights load_assistant_weights(
    const artifact::MaterializedArtifact& artifact, const AssistantPlan& plan) {
    using artifact::NumericFormat;
    AssistantWeights out;
    out.output_head = artifact::materialized_weight(
        artifact, plan.output_head, NumericFormat::FP8_E4M3FN_ROW_BF16S,
        AssistantConfig::vocabulary, AssistantConfig::hidden);
    out.input_projection = artifact::materialized_weight(
        artifact, plan.input_projection, NumericFormat::W8G32_F16S,
        AssistantConfig::hidden, AssistantConfig::input_rows);
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const bool full = AssistantConfig::layer_types[layer] == gemma4::AttentionType::Full;
        const std::int32_t head_dim =
            full ? AssistantConfig::full_head_dim : AssistantConfig::sliding_head_dim;
        const std::int32_t query_rows = AssistantConfig::query_heads * head_dim;
        const auto& source = plan.layers[layer];
        auto& target = out.layers[layer];
        target.input_norm = artifact::materialized_tensor(
            artifact, source.input_norm, NumericFormat::BF16, {AssistantConfig::hidden});
        target.attention.query = artifact::materialized_weight(
            artifact, source.attention.query, NumericFormat::W8G32_F16S,
            query_rows, AssistantConfig::hidden);
        target.attention.query_norm = artifact::materialized_tensor(
            artifact, source.attention.query_norm, NumericFormat::BF16, {head_dim});
        target.attention.output = artifact::materialized_weight(
            artifact, source.attention.output, NumericFormat::W8G32_F16S,
            AssistantConfig::hidden, query_rows);
        target.post_attention_norm = artifact::materialized_tensor(
            artifact, source.post_attention_norm, NumericFormat::BF16,
            {AssistantConfig::hidden});
        target.pre_feedforward_norm = artifact::materialized_tensor(
            artifact, source.pre_feedforward_norm, NumericFormat::BF16,
            {AssistantConfig::hidden});
        target.gate_up = artifact::materialized_weight(
            artifact, source.mlp.gate_up, NumericFormat::W8G32_F16S,
            2 * AssistantConfig::intermediate, AssistantConfig::hidden);
        target.down = artifact::materialized_weight(
            artifact, source.mlp.down, NumericFormat::W8G32_F16S,
            AssistantConfig::hidden, AssistantConfig::intermediate);
        target.post_feedforward_norm = artifact::materialized_tensor(
            artifact, source.post_feedforward_norm, NumericFormat::BF16,
            {AssistantConfig::hidden});
        target.layer_scalar = artifact::materialized_tensor(
            artifact, source.layer_scalar, NumericFormat::BF16, {1});
    }
    out.final_norm = artifact::materialized_tensor(
        artifact, plan.final_norm, NumericFormat::BF16, {AssistantConfig::hidden});
    out.output_projection = artifact::materialized_weight(
        artifact, plan.output_projection, NumericFormat::W8G32_F16S,
        AssistantConfig::output_rows, AssistantConfig::hidden);
    return out;
}

} // namespace ninfer::targets::gemma4_31b_it::detail
