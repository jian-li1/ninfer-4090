#include <ninfer/targets/gemma4_31b_it/package.h>

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/gemma4_31b_it/impl/load/bindings.h"
#include "targets/gemma4_31b_it/impl/runtime/model_weights.h"
#include "targets/gemma4_31b_it/impl/runtime/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::gemma4_31b_it::detail {
namespace {

std::string take_string(artifact::MaterializedArtifact& materialized,
                        artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

class LoadPlan::Impl {
public:
    Impl(WeightsProfile profile_in, ArtifactLoadPlan plan_in,
         SpeculativeOptions speculative_in, bool use_cuda_graph_in)
        : profile(profile_in), plan(std::move(plan_in)), speculative(speculative_in),
          use_cuda_graph(use_cuda_graph_in) {}

    WeightsProfile profile;
    ArtifactLoadPlan plan;
    SpeculativeOptions speculative;
    bool use_cuda_graph = false;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan() = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (!impl_) { throw std::logic_error("Gemma target load plan is empty"); }
    return impl_->plan.materialization;
}

class LoadedModel::Impl {
public:
    Impl(WeightsProfile profile_in, ArtifactLoadPlan plan_in,
         artifact::MaterializedArtifact materialized_in, SpeculativeOptions speculative_in,
         bool use_cuda_graph_in)
        : profile(profile_in), plan(std::move(plan_in)),
          frontend{
              .tokenizer_json = take_string(materialized_in, plan.bindings.frontend.tokenizer_json),
              .tokenizer_config_json =
                  take_string(materialized_in, plan.bindings.frontend.tokenizer_config_json),
              .chat_template_jinja =
                  take_string(materialized_in, plan.bindings.frontend.chat_template_jinja),
              .generation_config_json =
                  take_string(materialized_in, plan.bindings.frontend.generation_config_json),
          },
          materialized(std::move(materialized_in)),
          weights(load_weights(materialized, plan.bindings)), speculative(speculative_in),
          use_cuda_graph(use_cuda_graph_in) {
        if (speculative.backend == SpeculativeBackend::Mtp && plan.bindings.assistant) {
            assistant = load_assistant_weights(materialized, *plan.bindings.assistant);
        }
    }

    WeightsProfile profile;
    ArtifactLoadPlan plan;
    gemma4::FrontendResources frontend;
    artifact::MaterializedArtifact materialized;
    ModelWeights weights;
    std::optional<AssistantWeights> assistant;
    SpeculativeOptions speculative;
    bool use_cuda_graph = false;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::gemma4_31b_it::detail

namespace ninfer::targets::gemma4_31b_it {
namespace {

constexpr SamplingPreset kGemmaDefaults{
    .temperature       = 1.0F,
    .top_k             = 64,
    .top_p             = 0.95F,
    .min_p             = 0.0F,
    .presence_penalty  = 0.0F,
    .frequency_penalty = 0.0F,
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model != model_id) {
        throw std::runtime_error("model '" + std::string(model) +
                                 "' has no sampling defaults in target package '" +
                                 std::string(target_key) + "'");
    }
    return {.thinking = kGemmaDefaults, .non_thinking = kGemmaDefaults, .maximum_top_k = 64};
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == weights_id) {
        return WeightsProfile::GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" +
                             identity.weights_id + "' is not supported by target '" +
                             std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile profile) {
    if (options.enable_vision) {
        throw std::invalid_argument(
            "Gemma 4 31B artifact is text-only; the optional vision extension is not enabled");
    }
    if (options.kv_cache != KvCacheStorage::RK4V4E8) {
        throw std::invalid_argument(
            "Gemma 4 31B requires --kv-dtype rk4v4-e8 for its heterogeneous KV layout");
    }
    if (options.speculative.backend != SpeculativeBackend::None &&
        options.speculative.backend != SpeculativeBackend::Mtp) {
        throw std::invalid_argument("Gemma 4 31B supports only ordinary or MTP generation");
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        profile,
        detail::bind_artifact(binder, options.speculative.backend == SpeculativeBackend::Mtp),
        options.speculative, options.use_cuda_graph));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan,
                                artifact::MaterializedArtifact&& materialized) {
    if (!plan.impl_) { throw std::invalid_argument("Gemma target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->profile, std::move(plan.impl_->plan), std::move(materialized),
        plan.impl_->speculative, plan.impl_->use_cuda_graph);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model,
                                         const EngineOptions& options) {
    if (!model.impl_) { throw std::invalid_argument("loaded Gemma model is empty"); }
    return gemma4::make_frontend(model.impl_->frontend,
                                 gemma4::FrontendOptions{.max_context = options.max_context});
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                        const StartupObserver& startup_observer) {
    if (!model.impl_) { throw std::invalid_argument("loaded Gemma model is empty"); }
    detail::Variant::ModelView runtime_model{
        .target = model.impl_->weights,
        .assistant = model.impl_->assistant,
        .weights_arena = &model.impl_->materialized.device_arena(),
        .speculative = model.impl_->speculative,
        .use_cuda_graph = model.impl_->use_cuda_graph,
    };
    return qwen3_6::create_program<detail::Variant>(runtime_model, model.impl_->profile,
                                                   std::move(plan), device, startup_observer);
}

} // namespace ninfer::targets::gemma4_31b_it
