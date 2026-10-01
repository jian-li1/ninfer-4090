#pragma once

#include <ninfer/targets/gemma4/layer_schedule.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::gemma4_31b_it {

struct TextConfig {
    static constexpr std::uint32_t vocabulary              = 262144;
    static constexpr std::uint32_t hidden                  = 5376;
    static constexpr std::uint32_t intermediate            = 21504;
    static constexpr std::uint32_t layers                  = 60;
    static constexpr std::uint32_t maximum_position        = 262144;
    static constexpr std::uint32_t query_heads             = 32;
    static constexpr std::uint32_t sliding_kv_heads        = 16;
    static constexpr std::uint32_t sliding_head_dim        = 256;
    static constexpr std::uint32_t sliding_window          = 1024;
    static constexpr float sliding_rope_theta              = 10000.0F;
    static constexpr std::uint32_t sliding_query_rows      = 8192;
    static constexpr std::uint32_t sliding_kv_rows         = 4096;
    static constexpr std::uint32_t full_kv_heads           = 4;
    static constexpr std::uint32_t full_head_dim           = 512;
    static constexpr std::uint32_t full_rotary_active_dim  = 128;
    static constexpr std::uint32_t full_query_rows         = 16384;
    static constexpr std::uint32_t full_kv_rows            = 2048;
    static constexpr float full_rope_theta                 = 1000000.0F;
    static constexpr std::uint32_t full_attention_interval = 6;
    static constexpr float rms_epsilon                     = 1.0e-6F;
    static constexpr float attention_scale                 = 1.0F;
    static constexpr float final_logit_softcap             = 30.0F;
    static constexpr float embedding_scale_bf16            = 73.5F;
    static constexpr std::uint32_t quantization_bits       = 4;
    static constexpr std::uint32_t quantization_group      = 32;
    static constexpr bool tied_embeddings                  = true;
    static constexpr bool attention_bias                   = false;
    static constexpr bool full_attention_k_equals_v        = true;
    static constexpr bool softmax_fp32                      = true;
    static constexpr bool gelu_tanh                         = true;
    static constexpr bool value_norm_has_scale             = false;

    static constexpr auto layer_types =
        gemma4::make_layer_schedule<layers, full_attention_interval>();
    static constexpr std::size_t sliding_layers =
        gemma4::count_layers(layer_types, gemma4::AttentionType::Sliding);
    static constexpr std::size_t full_layers =
        gemma4::count_layers(layer_types, gemma4::AttentionType::Full);

    [[nodiscard]] static constexpr bool is_full_attention(std::size_t layer) {
        return layer < layer_types.size() && layer_types[layer] == gemma4::AttentionType::Full;
    }
};

struct AssistantConfig {
    static constexpr std::uint32_t layers                      = 4;
    static constexpr std::uint32_t hidden                      = 1024;
    static constexpr std::uint32_t intermediate                = 8192;
    static constexpr std::uint32_t vocabulary                  = TextConfig::vocabulary;
    static constexpr std::uint32_t maximum_position            = TextConfig::maximum_position;
    static constexpr std::uint32_t query_heads                 = TextConfig::query_heads;
    static constexpr std::uint32_t sliding_kv_heads            = TextConfig::sliding_kv_heads;
    static constexpr std::uint32_t sliding_head_dim            = TextConfig::sliding_head_dim;
    static constexpr std::uint32_t sliding_window              = TextConfig::sliding_window;
    static constexpr float sliding_rope_theta                  = TextConfig::sliding_rope_theta;
    static constexpr std::uint32_t full_kv_heads               = TextConfig::full_kv_heads;
    static constexpr std::uint32_t full_head_dim               = TextConfig::full_head_dim;
    static constexpr std::uint32_t full_rotary_active_dim      =
        TextConfig::full_rotary_active_dim;
    static constexpr float full_rope_theta                     = TextConfig::full_rope_theta;
    static constexpr std::uint32_t input_rows                  = 2 * TextConfig::hidden;
    static constexpr std::uint32_t output_rows                 = TextConfig::hidden;
    static constexpr std::uint32_t shared_kv_layers            = 4;
    static constexpr std::uint32_t centroids                   = 2048;
    static constexpr std::uint32_t centroid_intermediate_top_k = 32;
    static constexpr std::uint32_t production_draft_size       = 1;
    static constexpr std::uint32_t maximum_draft_size          = 6;
    static constexpr bool tied_embeddings                      = true;
    static constexpr bool owns_key_value_projections           = false;

    static constexpr std::array<gemma4::AttentionType, layers> layer_types{
        gemma4::AttentionType::Sliding,
        gemma4::AttentionType::Sliding,
        gemma4::AttentionType::Sliding,
        gemma4::AttentionType::Full,
    };
    static constexpr std::array<std::uint32_t, layers> shared_target_kv_layers{58, 58, 58, 59};
};

[[nodiscard]] consteval bool valid_config() {
    if (TextConfig::layers != 60 || TextConfig::sliding_layers != 50 ||
        TextConfig::full_layers != 10) {
        return false;
    }
    for (std::size_t layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::is_full_attention(layer) != ((layer + 1) % 6 == 0)) { return false; }
    }
    return TextConfig::query_heads * TextConfig::sliding_head_dim == 8192 &&
           TextConfig::query_heads * TextConfig::full_head_dim == 16384 &&
           TextConfig::sliding_kv_heads * TextConfig::sliding_head_dim ==
               TextConfig::sliding_kv_rows &&
           TextConfig::full_kv_heads * TextConfig::full_head_dim == TextConfig::full_kv_rows &&
           TextConfig::full_rotary_active_dim * 4 == TextConfig::full_head_dim &&
           AssistantConfig::shared_kv_layers == AssistantConfig::layers &&
           AssistantConfig::input_rows == 10752 && AssistantConfig::output_rows == 5376 &&
           AssistantConfig::shared_target_kv_layers.back() == 59;
}

static_assert(valid_config());
static_assert(TextConfig::is_full_attention(5));
static_assert(TextConfig::is_full_attention(59));
static_assert(!TextConfig::is_full_attention(0));

} // namespace ninfer::targets::gemma4_31b_it
