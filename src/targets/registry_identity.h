#pragma once

#include <string_view>

namespace ninfer::artifact {
struct ArtifactIdentity;
}

namespace ninfer::targets {

struct RegisteredTargetDescriptor {
    std::string_view model_id;
    std::string_view weights_id;
    std::string_view target_key;
};

[[nodiscard]] const RegisteredTargetDescriptor*
identify_registered_target(const artifact::ArtifactIdentity& identity) noexcept;

} // namespace ninfer::targets
