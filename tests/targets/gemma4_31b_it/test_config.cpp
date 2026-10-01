#include "artifact/reader.h"
#include "targets/registry_identity.h"

#include <ninfer/targets/gemma4/capabilities.h>
#include <ninfer/targets/gemma4/layer_schedule.h>
#include <ninfer/targets/gemma4_31b_it/package.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using ninfer::targets::gemma4::AttentionType;
using ninfer::targets::gemma4::Capability;
using ninfer::targets::gemma4_31b_it::AssistantConfig;
using ninfer::targets::gemma4_31b_it::Package;
using ninfer::targets::gemma4_31b_it::TextConfig;

static_assert(ninfer::targets::gemma4_31b_it::valid_config());
static_assert(TextConfig::layers == 60);
static_assert(TextConfig::sliding_layers == 50);
static_assert(TextConfig::full_layers == 10);
static_assert(Package::capabilities.text == Capability::Required);
static_assert(Package::capabilities.mtp == Capability::ArtifactOptional);
static_assert(Package::capabilities.vision == Capability::Unavailable);
static_assert(Package::capabilities.audio == Capability::Unavailable);
static_assert(AssistantConfig::layer_types ==
              std::array{AttentionType::Sliding, AttentionType::Sliding,
                         AttentionType::Sliding, AttentionType::Full});
static_assert(AssistantConfig::shared_target_kv_layers ==
              std::array<std::uint32_t, 4>{58, 58, 58, 59});
static_assert(!AssistantConfig::owns_key_value_projections);

[[noreturn]] void fail(std::string_view message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

} // namespace

int main() {
    constexpr std::array<std::size_t, 10> expected_full_layers{5, 11, 17, 23, 29,
                                                               35, 41, 47, 53, 59};
    std::size_t full_index = 0;
    for (std::size_t layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::layer_types[layer] == AttentionType::Full) {
            if (full_index >= expected_full_layers.size() ||
                expected_full_layers[full_index] != layer) {
                fail("full-attention schedule differs from the pinned target config");
            }
            ++full_index;
        }
    }
    if (full_index != expected_full_layers.size()) { fail("full-attention count is incomplete"); }

    const ninfer::artifact::ArtifactIdentity identity{
        .model_id   = std::string(Package::model_id),
        .weights_id = std::string(Package::weights_id),
    };
    const auto* descriptor = ninfer::targets::identify_registered_target(identity);
    if (descriptor == nullptr || descriptor->model_id != Package::model_id ||
        descriptor->weights_id != Package::weights_id ||
        descriptor->target_key != Package::target_key) {
        fail("registry did not identify the synthetic Gemma artifact descriptor");
    }

    const ninfer::artifact::ArtifactIdentity wrong_weights{
        .model_id = std::string(Package::model_id), .weights_id = "nvfp4"};
    if (ninfer::targets::identify_registered_target(wrong_weights) != nullptr) {
        fail("registry accepted an unregistered Gemma weights identity");
    }

    const ninfer::artifact::ArtifactIdentity existing_qwen{
        .model_id = "qwen3.8-27b", .weights_id = "groupwise-int"};
    const auto* qwen_descriptor = ninfer::targets::identify_registered_target(existing_qwen);
    if (qwen_descriptor == nullptr || qwen_descriptor->target_key != "qwen3_8_27b") {
        fail("existing Qwen registry identity changed");
    }

    std::cout << "PASS\n";
    return 0;
}
