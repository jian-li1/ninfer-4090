#pragma once

#include <cstdint>

namespace ninfer::targets::gemma4 {

enum class Capability : std::uint8_t {
    Unavailable,
    Required,
    ArtifactOptional,
};

struct Capabilities {
    Capability text;
    Capability mtp;
    Capability vision;
    Capability audio;

    bool operator==(const Capabilities&) const = default;
};

} // namespace ninfer::targets::gemma4
