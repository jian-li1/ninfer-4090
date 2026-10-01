#include "targets/gemma4_31b_it/impl/load/bindings.h"

#include <ninfer/targets/gemma4_31b_it/config.h>

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ninfer::targets::gemma4_31b_it::detail {
namespace {

using artifact::NumericFormat;
using artifact::TensorPlacement;

artifact::ObjectHandle bind(artifact::Binder& binder, std::string_view name, NumericFormat format,
                            std::initializer_list<std::uint64_t> shape,
                            TensorPlacement placement = TensorPlacement::Device) {
    return artifact::bind_tensor(binder, name, format, shape, placement);
}

AssistantPlan bind_assistant(artifact::Binder& binder) {
    constexpr TensorPlacement placement = TensorPlacement::ValidateOnly;
    AssistantPlan out;
    out.token_embedding = bind(binder, "assistant/token_embedding",
                               NumericFormat::FP8_E4M3FN_ROW_BF16S, {262144, 1024}, placement);
    out.output_head      = out.token_embedding;
    out.input_projection = bind(binder, "assistant/input_projection",
                                NumericFormat::W8G32_F16S, {1024, 10752}, placement);
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        auto& target             = out.layers[layer];
        const std::string prefix = "assistant/layers/" + std::to_string(layer) + "/";
        const bool full          = layer == 3;
        const std::uint64_t rows = full ? 16384 : 8192;
        const std::uint64_t norm = full ? 512 : 256;
        target.input_norm = bind(binder, prefix + "input_norm", NumericFormat::BF16, {1024}, placement);
        target.attention.query = bind(binder, prefix + "attention/query", NumericFormat::W8G32_F16S,
                                      {rows, 1024}, placement);
        target.attention.query_norm = bind(binder, prefix + "attention/query_norm",
                                           NumericFormat::BF16, {norm}, placement);
        target.attention.output = bind(binder, prefix + "attention/output",
                                       NumericFormat::W8G32_F16S, {1024, rows}, placement);
        target.post_attention_norm = bind(binder, prefix + "post_attention_norm",
                                          NumericFormat::BF16, {1024}, placement);
        target.pre_feedforward_norm = bind(binder, prefix + "pre_feedforward_norm",
                                           NumericFormat::BF16, {1024}, placement);
        target.mlp.gate_up = bind(binder, prefix + "mlp/gate_up", NumericFormat::W8G32_F16S,
                                  {16384, 1024}, placement);
        target.mlp.down = bind(binder, prefix + "mlp/down", NumericFormat::W8G32_F16S,
                               {1024, 8192}, placement);
        target.post_feedforward_norm = bind(binder, prefix + "post_feedforward_norm",
                                            NumericFormat::BF16, {1024}, placement);
        target.layer_scalar = bind(binder, prefix + "layer_scalar",
                                   NumericFormat::BF16, {1}, placement);
    }
    out.final_norm = bind(binder, "assistant/final_norm", NumericFormat::BF16, {1024}, placement);
    out.output_projection = bind(binder, "assistant/output_projection", NumericFormat::W8G32_F16S,
                                 {5376, 1024}, placement);
    return out;
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder) {
    ArtifactLoadPlan load_plan;
    auto& out = load_plan.bindings;
    out.frontend = {
        .tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json = artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja"),
        .generation_config_json = artifact::bind_raw_resource(binder, "frontend/generation_config.json"),
    };
    out.token_embedding = artifact::bind_device_tensor(
        binder, "text/token_embedding", NumericFormat::FP8_E4M3FN_ROW_BF16S, {262144, 5376});
    out.output_head = out.token_embedding;

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        auto& target             = out.layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        const bool full          = TextConfig::is_full_attention(layer);
        const std::uint64_t input_rows = full ? 18432 : 16384;
        const std::uint64_t output_columns = full ? 16384 : 8192;
        const std::uint64_t norm_width = full ? 512 : 256;
        target.input_norm = bind(binder, prefix + "input_norm", NumericFormat::BF16, {5376});
        target.attention.input_projection = bind(binder, prefix + "attention/input_projection",
                                                 NumericFormat::Q4G64_F16S, {input_rows, 5376});
        target.attention.query_norm = bind(binder, prefix + "attention/query_norm",
                                           NumericFormat::BF16, {norm_width});
        target.attention.key_norm = bind(binder, prefix + "attention/key_norm",
                                         NumericFormat::BF16, {norm_width});
        target.attention.output = bind(binder, prefix + "attention/output",
                                       NumericFormat::Q4G64_F16S, {5376, output_columns});
        target.post_attention_norm = bind(binder, prefix + "post_attention_norm",
                                          NumericFormat::BF16, {5376});
        target.pre_feedforward_norm = bind(binder, prefix + "pre_feedforward_norm",
                                           NumericFormat::BF16, {5376});
        target.mlp.gate_up = bind(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S,
                                  {43008, 5376});
        target.mlp.down = bind(binder, prefix + "mlp/down", NumericFormat::Q4G64_F16S,
                               {5376, 21504});
        target.post_feedforward_norm = bind(binder, prefix + "post_feedforward_norm",
                                            NumericFormat::BF16, {5376});
        target.layer_scalar = bind(binder, prefix + "layer_scalar", NumericFormat::BF16, {1});
    }
    out.final_norm = bind(binder, "text/final_norm", NumericFormat::BF16, {5376});
    if (binder.has_object("assistant/token_embedding")) {
        out.assistant = bind_assistant(binder);
    }
    load_plan.materialization = binder.finish();
    return load_plan;
}

} // namespace ninfer::targets::gemma4_31b_it::detail
