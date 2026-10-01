#pragma once

#include <ninfer/targets/gemma4/capabilities.h>
#include <ninfer/targets/gemma4/model_view.h>
#include <ninfer/targets/gemma4_31b_it/config.h>

#include <string_view>

namespace ninfer::targets::gemma4_31b_it {

struct Package {
    static constexpr std::string_view model_id   = "gemma4-31b-it";
    static constexpr std::string_view weights_id = "groupwise-int";
    static constexpr std::string_view target_key = "gemma4_31b_it";

    static constexpr gemma4::Capabilities capabilities{
        .text   = gemma4::Capability::Required,
        .mtp    = gemma4::Capability::ArtifactOptional,
        .vision = gemma4::Capability::Unavailable,
        .audio  = gemma4::Capability::Unavailable,
    };

    using Config           = TextConfig;
    using Assistant        = AssistantConfig;
    using RuntimeModelView = gemma4::ModelView<TextConfig::layers, AssistantConfig::layers>;
};

} // namespace ninfer::targets::gemma4_31b_it
