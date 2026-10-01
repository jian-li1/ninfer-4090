#pragma once

#include "artifact/binder.h"

#include <array>
#include <optional>

namespace ninfer::targets::gemma4_31b_it::detail {

struct FrontendPlan {
    artifact::ObjectHandle tokenizer_json;
    artifact::ObjectHandle tokenizer_config_json;
    artifact::ObjectHandle chat_template_jinja;
    artifact::ObjectHandle generation_config_json;
};

struct AttentionPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
};

struct MlpPlan {
    artifact::ObjectHandle gate_up;
    artifact::ObjectHandle down;
};

struct LayerPlan {
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle pre_feedforward_norm;
    MlpPlan mlp;
    artifact::ObjectHandle post_feedforward_norm;
    artifact::ObjectHandle layer_scalar;
};

struct AssistantAttentionPlan {
    artifact::ObjectHandle query;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle output;
};

struct AssistantLayerPlan {
    artifact::ObjectHandle input_norm;
    AssistantAttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle pre_feedforward_norm;
    MlpPlan mlp;
    artifact::ObjectHandle post_feedforward_norm;
    artifact::ObjectHandle layer_scalar;
};

struct AssistantPlan {
    artifact::ObjectHandle token_embedding;
    artifact::ObjectHandle output_head;
    artifact::ObjectHandle input_projection;
    std::array<AssistantLayerPlan, 4> layers;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle output_projection;
};

struct BindingPlan {
    FrontendPlan frontend;
    artifact::ObjectHandle token_embedding;
    artifact::ObjectHandle output_head;
    std::array<LayerPlan, 60> layers;
    artifact::ObjectHandle final_norm;
    std::optional<AssistantPlan> assistant;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

[[nodiscard]] ArtifactLoadPlan bind_artifact(artifact::Binder& binder);

} // namespace ninfer::targets::gemma4_31b_it::detail
