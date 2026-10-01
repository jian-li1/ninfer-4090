#pragma once

#include <array>
#include <cstddef>

namespace ninfer::targets::gemma4 {

enum class AttentionType {
    Sliding,
    Full,
};

template <std::size_t LayerCount, std::size_t FullAttentionInterval>
[[nodiscard]] consteval std::array<AttentionType, LayerCount> make_layer_schedule() {
    static_assert(FullAttentionInterval > 0);
    std::array<AttentionType, LayerCount> schedule{};
    for (std::size_t layer = 0; layer < LayerCount; ++layer) {
        schedule[layer] = (layer + 1) % FullAttentionInterval == 0
                              ? AttentionType::Full
                              : AttentionType::Sliding;
    }
    return schedule;
}

template <std::size_t LayerCount>
[[nodiscard]] consteval std::size_t
count_layers(const std::array<AttentionType, LayerCount>& schedule, AttentionType type) {
    std::size_t count = 0;
    for (AttentionType layer_type : schedule) {
        if (layer_type == type) { ++count; }
    }
    return count;
}

} // namespace ninfer::targets::gemma4
