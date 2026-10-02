#pragma once

#include "ninfer/types.h"
#include "runtime/engine/context_cost.h"
#include "targets/registry_identity.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#if NINFER_BUILD_GEMMA4_31B_IT
#include <ninfer/targets/gemma4_31b_it/package.h>
#endif

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;
#if NINFER_BUILD_GEMMA4_31B_IT
using Gemma4_31B_IT = gemma4_31b_it::Package;
#endif

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_6_27B::SequencePlan sequence_plan, DeviceContext& device,
                        const StartupObserver& startup_observer);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

struct LoadedQwen3_6_35BA3B {
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> model;
    Qwen3_6_35BA3B::Frontend frontend;

    LoadedQwen3_6_35BA3B(std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model,
                         const EngineOptions& options);
    ~LoadedQwen3_6_35BA3B();

    LoadedQwen3_6_35BA3B(const LoadedQwen3_6_35BA3B&)            = delete;
    LoadedQwen3_6_35BA3B& operator=(const LoadedQwen3_6_35BA3B&) = delete;
};

struct Qwen3_6_35BA3BInstance {
    using Package = Qwen3_6_35BA3B;

    std::unique_ptr<LoadedQwen3_6_35BA3B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_35BA3B::Program> program;

    Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           Qwen3_6_35BA3B::SequencePlan sequence_plan, DeviceContext& device,
                           const StartupObserver& startup_observer);
    ~Qwen3_6_35BA3BInstance();

    Qwen3_6_35BA3BInstance(const Qwen3_6_35BA3BInstance&)            = delete;
    Qwen3_6_35BA3BInstance& operator=(const Qwen3_6_35BA3BInstance&) = delete;
};

#if NINFER_BUILD_GEMMA4_31B_IT
struct LoadedGemma4_31B_IT {
    std::unique_ptr<Gemma4_31B_IT::LoadedModel> model;
    Gemma4_31B_IT::Frontend frontend;

    LoadedGemma4_31B_IT(std::unique_ptr<Gemma4_31B_IT::LoadedModel> stable_model,
                        const EngineOptions& options);
    ~LoadedGemma4_31B_IT();

    LoadedGemma4_31B_IT(const LoadedGemma4_31B_IT&)            = delete;
    LoadedGemma4_31B_IT& operator=(const LoadedGemma4_31B_IT&) = delete;
};

struct Gemma4_31B_ITInstance {
    using Package = Gemma4_31B_IT;

    std::unique_ptr<LoadedGemma4_31B_IT> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Gemma4_31B_IT::Program> program;

    Gemma4_31B_ITInstance(std::unique_ptr<LoadedGemma4_31B_IT> stable_loaded,
                          runtime::KvCapacityResolution resolution,
                          Gemma4_31B_IT::SequencePlan sequence_plan, DeviceContext& device,
                          const StartupObserver& startup_observer);
    ~Gemma4_31B_ITInstance();

    Gemma4_31B_ITInstance(const Gemma4_31B_ITInstance&)            = delete;
    Gemma4_31B_ITInstance& operator=(const Gemma4_31B_ITInstance&) = delete;
};
#endif

using ActiveTarget =
    std::variant<std::unique_ptr<Qwen3_6_27BInstance>, std::unique_ptr<Qwen3_6_35BA3BInstance>
#if NINFER_BUILD_GEMMA4_31B_IT
                 , std::unique_ptr<Gemma4_31B_ITInstance>
#endif
                 >;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    runtime::ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer
