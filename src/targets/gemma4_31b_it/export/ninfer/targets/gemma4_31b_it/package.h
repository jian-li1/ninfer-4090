#pragma once

#include <ninfer/targets/gemma4/capabilities.h>
#include <ninfer/targets/gemma4/frontend.h>
#include <ninfer/targets/gemma4/model_view.h>
#include <ninfer/targets/gemma4_31b_it/config.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {
struct DeviceContext;
namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
}
}

namespace ninfer::targets::gemma4_31b_it {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
};

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct gemma4_31b_it::Package;
};

class LoadedModel {
public:
    ~LoadedModel();
    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
    friend struct gemma4_31b_it::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "gemma4-31b-it";
    static constexpr std::string_view weights_id = "groupwise-int";
    static constexpr std::string_view target_key = "gemma4_31b_it";
    // The registered converter accepts exactly this immutable target checkpoint. KV layout is
    // fingerprinted separately by the continuation's encoded group descriptors.
    static constexpr std::string_view artifact_compatibility_fingerprint =
        "sha256:1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20";

    static constexpr gemma4::Capabilities capabilities{
        .text   = gemma4::Capability::Required,
        .mtp    = gemma4::Capability::ArtifactOptional,
        .vision = gemma4::Capability::Unavailable,
        .audio  = gemma4::Capability::Unavailable,
    };

    using Config           = TextConfig;
    using Assistant        = AssistantConfig;
    using RuntimeModelView = gemma4::ModelView<TextConfig::layers, AssistantConfig::layers>;

    using WeightsProfile             = detail::WeightsProfile;
    using LoadPlan                   = detail::LoadPlan;
    using LoadedModel                = detail::LoadedModel;
    using Frontend                   = gemma4::Frontend;
    using PreparedPrompt             = gemma4::PreparedPrompt;
    using OutputSession              = gemma4::OutputSession;
    using PublishedOutput            = gemma4::PublishedOutput;
    using SequencePlanner            = qwen3_6::SequencePlanner<detail::Variant>;
    using SequencePlan               = qwen3_6::SequencePlan<detail::Variant>;
    using RequestBasePlan            = qwen3_6::RequestBasePlan<detail::Variant>;
    using AdmissionCandidate         = qwen3_6::AdmissionCandidate<detail::Variant>;
    using ResourcePlan               = qwen3_6::ResourcePlan<detail::Variant>;
    using PersistentBackfillProof    = qwen3_6::PersistentBackfillProof<detail::Variant>;
    using SequenceHandle             = qwen3_6::SequenceHandle<detail::Variant>;
    using ContinuationHandle         = qwen3_6::ContinuationHandle<detail::Variant>;
    using SharedPrefixHandle         = qwen3_6::SharedPrefixHandle<detail::Variant>;
    using CaptureOffer               = qwen3_6::CaptureOffer<detail::Variant>;
    using CacheSessionKey            = qwen3_6::PreparedSessionKey;
    using ContinuationSummary        = qwen3_6::ContinuationSummary;
    using SharedPrefixSummary        = qwen3_6::SharedPrefixSummary;
    using PressurePlanningSession    = qwen3_6::PressurePlanningSession<detail::Variant>;
    using PressureTargetHandle       = qwen3_6::PressureTargetHandle;
    using AssessedPressureTarget     = qwen3_6::AssessedPressureTarget<detail::Variant>;
    using CapturePressurePlan        = qwen3_6::CapturePressurePlan<detail::Variant>;
    using MaterializationResult      = qwen3_6::MaterializationResult<detail::Variant>;
    using ContextTransactionProgress = qwen3_6::ContextTransactionProgress<detail::Variant>;
    using CaptureAssessment          = qwen3_6::CaptureAssessment;
    using ActiveCaptureResult        = qwen3_6::ActiveCaptureResult<detail::Variant>;
    using PendingBatch               = qwen3_6::PendingBatch<detail::Variant>;
    using StartResult                = qwen3_6::StartResult<detail::Variant>;
    using PrefillProgress            = qwen3_6::PrefillProgress<detail::Variant>;
    using CommitResult               = qwen3_6::CommitResult<detail::Variant>;
    using DiscardResult              = qwen3_6::DiscardResult<detail::Variant>;
    using FinishResult               = qwen3_6::FinishResult<detail::Variant>;
    using AbortResult                = qwen3_6::AbortResult<detail::Variant>;
    using ReleaseResult              = qwen3_6::ReleaseResult<detail::Variant>;
    using Program                    = qwen3_6::Program<detail::Variant>;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                   const StartupObserver& startup_observer);
};

} // namespace ninfer::targets::gemma4_31b_it
