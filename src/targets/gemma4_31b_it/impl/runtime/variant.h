#pragma once

#include <ninfer/targets/gemma4_31b_it/package.h>

#include "targets/gemma4_31b_it/impl/runtime/model_weights.h"

namespace ninfer::targets::gemma4_31b_it::detail {

// Closed target identity used only to specialize the common opaque Engine/Program carriers.
// The execution schedule itself is implemented in this package and does not instantiate Qwen's
// Program algorithm.
struct Variant {
    using WeightsProfile = detail::WeightsProfile;
    struct ModelView {
        detail::ModelWeights target;
        std::optional<detail::AssistantWeights> assistant;
        DeviceArena* weights_arena = nullptr;
        SpeculativeOptions speculative;
        bool use_cuda_graph = false;
    };
};

} // namespace ninfer::targets::gemma4_31b_it::detail
