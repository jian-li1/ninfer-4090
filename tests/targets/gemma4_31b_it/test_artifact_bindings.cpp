#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "targets/gemma4_31b_it/impl/load/bindings.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Json = nlohmann::json;
using ninfer::artifact::NumericFormat;
using ninfer::artifact::StorageLayout;

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct DirectoryBuilder {
    Json objects = Json::array();
    std::uint64_t cursor = 0;

    void resource(std::string_view name) {
        objects.push_back({{"name", name}, {"kind", "resource"}, {"encoding", "raw-bytes-v1"},
                           {"offset", cursor}, {"bytes", 1}});
        ++cursor;
    }

    void tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                std::initializer_list<std::uint64_t> shape) {
        const std::array<std::uint64_t, 2> empty{};
        const std::span<const std::uint64_t> dimensions(shape.begin(), shape.size());
        cursor = align_up(cursor, ninfer::artifact::tensor_alignment(layout));
        const std::uint64_t bytes = ninfer::artifact::tensor_encoded_size(layout, format, dimensions);
        objects.push_back({{"name", name}, {"kind", "tensor"},
                           {"shape", Json(dimensions)}, {"format", ninfer::artifact::format_name(format)},
                           {"layout", ninfer::artifact::layout_name(layout)},
                           {"offset", cursor}, {"bytes", bytes}});
        cursor += bytes;
    }
};

Json complete_directory() {
    DirectoryBuilder out;
    out.resource("frontend/tokenizer.json");
    out.resource("frontend/tokenizer_config.json");
    out.resource("frontend/chat_template.jinja");
    out.resource("frontend/generation_config.json");
    out.tensor("text/token_embedding", NumericFormat::FP8_E4M3FN_ROW_BF16S,
               StorageLayout::RowScaleV1, {262144, 5376});
    for (std::size_t layer = 0; layer < 60; ++layer) {
        const bool full = (layer + 1) % 6 == 0;
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        out.tensor(prefix + "input_norm", NumericFormat::BF16, StorageLayout::ContiguousLeV1, {5376});
        out.tensor(prefix + "attention/input_projection", NumericFormat::Q4G64_F16S,
                   StorageLayout::RowSplitK128V1, {full ? 18432ULL : 16384ULL, 5376});
        out.tensor(prefix + "attention/query_norm", NumericFormat::BF16,
                   StorageLayout::ContiguousLeV1, {full ? 512ULL : 256ULL});
        out.tensor(prefix + "attention/key_norm", NumericFormat::BF16,
                   StorageLayout::ContiguousLeV1, {full ? 512ULL : 256ULL});
        out.tensor(prefix + "attention/output", NumericFormat::Q4G64_F16S,
                   StorageLayout::RowSplitK128V1, {5376, full ? 16384ULL : 8192ULL});
        out.tensor(prefix + "post_attention_norm", NumericFormat::BF16,
                   StorageLayout::ContiguousLeV1, {5376});
        out.tensor(prefix + "pre_feedforward_norm", NumericFormat::BF16,
                   StorageLayout::ContiguousLeV1, {5376});
        out.tensor(prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S,
                   StorageLayout::RowSplitK128V1, {43008, 5376});
        out.tensor(prefix + "mlp/down", NumericFormat::Q4G64_F16S,
                   StorageLayout::RowSplitK128V1, {5376, 21504});
        out.tensor(prefix + "post_feedforward_norm", NumericFormat::BF16,
                   StorageLayout::ContiguousLeV1, {5376});
        out.tensor(prefix + "layer_scalar", NumericFormat::BF16,
                   StorageLayout::ContiguousLeV1, {1});
    }
    out.tensor("text/final_norm", NumericFormat::BF16, StorageLayout::ContiguousLeV1, {5376});
    return {{"identity", {{"model_id", "gemma4-31b-it"}, {"weights_id", "groupwise-int"}}},
            {"objects", std::move(out.objects)}};
}

ninfer::test::artifact_fixture::TemporaryArtifact write_sparse(const Json& directory,
                                                                std::string_view suffix) {
    const std::string encoded = directory.dump();
    const std::uint64_t payload_offset = align_up(16 + encoded.size(), 4096);
    std::uint64_t payload_bytes = 0;
    for (const auto& object : directory.at("objects")) {
        payload_bytes = std::max(payload_bytes, object.at("offset").get<std::uint64_t>() +
                                                object.at("bytes").get<std::uint64_t>());
    }
    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_gemma4_binding_" + std::string(suffix) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(ninfer::test::artifact_fixture::kMagic.data()), 8);
    std::array<std::byte, 8> size{};
    ninfer::test::artifact_fixture::write_u64_le(size.data(), encoded.size());
    output.write(reinterpret_cast<const char*>(size.data()), size.size());
    output.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
    output.seekp(static_cast<std::streamoff>(payload_offset + payload_bytes - 1));
    output.put('\0');
    if (!output) { throw std::runtime_error("failed to write sparse Gemma fixture"); }
    return {std::move(path)};
}

template <class Function>
void expect_artifact_error(Function&& function, std::string_view message) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(message));
}

void test_complete_target() {
    auto fixture = write_sparse(complete_directory(), "complete");
    ninfer::artifact::Reader reader(fixture.path);
    ninfer::artifact::Binder binder(reader);
    const auto plan = ninfer::targets::gemma4_31b_it::detail::bind_artifact(binder);
    if (plan.materialization.object_count != 666 ||
        plan.materialization.device_objects.size() != 662 ||
        plan.materialization.host_objects.size() != 4 ||
        plan.materialization.device_capacity_bytes != 16'971'062'784ULL ||
        plan.bindings.output_head.index != plan.bindings.token_embedding.index ||
        plan.bindings.assistant.has_value()) {
        throw std::runtime_error("Gemma target-only binding plan is incomplete");
    }
    ninfer::artifact::Binder mtp_binder(reader);
    expect_artifact_error(
        [&] {
            (void)ninfer::targets::gemma4_31b_it::detail::bind_artifact(mtp_binder, true);
        },
        "MTP residency accepted an artifact without an assistant companion");
}

void test_missing_and_malformed_objects() {
    {
        auto directory = complete_directory();
        directory["objects"].erase(directory["objects"].end() - 1);
        auto fixture = write_sparse(directory, "missing");
        expect_artifact_error([&] {
            ninfer::artifact::Reader reader(fixture.path);
            ninfer::artifact::Binder binder(reader);
            (void)ninfer::targets::gemma4_31b_it::detail::bind_artifact(binder);
        }, "missing final norm was accepted");
    }
    {
        auto directory = complete_directory();
        auto& final_norm = directory["objects"].back();
        final_norm["shape"] = {1};
        final_norm["bytes"] = 2;
        auto fixture = write_sparse(directory, "malformed");
        expect_artifact_error([&] {
            ninfer::artifact::Reader reader(fixture.path);
            ninfer::artifact::Binder binder(reader);
            (void)ninfer::targets::gemma4_31b_it::detail::bind_artifact(binder);
        }, "malformed final norm was accepted");
    }
    {
        auto directory = complete_directory();
        const auto offset = directory["objects"].back()["offset"].get<std::uint64_t>() +
                            directory["objects"].back()["bytes"].get<std::uint64_t>();
        directory["objects"].push_back({{"name", "vision/forbidden"}, {"kind", "resource"},
                                         {"encoding", "raw-bytes-v1"}, {"offset", offset}, {"bytes", 1}});
        auto fixture = write_sparse(directory, "vision_extra");
        expect_artifact_error([&] {
            ninfer::artifact::Reader reader(fixture.path);
            ninfer::artifact::Binder binder(reader);
            (void)ninfer::targets::gemma4_31b_it::detail::bind_artifact(binder);
        }, "multimodal extra object was accepted");
    }
}

void test_real_artifact_when_requested() {
    const char* path = std::getenv("NINFER_GEMMA4_ARTIFACT");
    if (path == nullptr) { return; }
    ninfer::artifact::Reader reader(path);
    if (reader.identity().model_id != "gemma4-31b-it" ||
        reader.identity().weights_id != "groupwise-int") {
        throw std::runtime_error("real Gemma artifact identity is wrong");
    }
    ninfer::artifact::Binder binder(reader);
    const auto plan = ninfer::targets::gemma4_31b_it::detail::bind_artifact(binder);
    if (plan.materialization.object_count != 710 ||
        plan.materialization.device_objects.size() != 662 ||
        plan.materialization.host_objects.size() != 4 ||
        plan.materialization.device_capacity_bytes != 16'971'062'784ULL ||
        !plan.bindings.assistant.has_value()) {
        throw std::runtime_error("real Gemma target-plus-assistant binding plan is incomplete");
    }
    ninfer::artifact::Binder mtp_binder(reader);
    const auto mtp_plan =
        ninfer::targets::gemma4_31b_it::detail::bind_artifact(mtp_binder, true);
    if (mtp_plan.materialization.object_count != 710 ||
        mtp_plan.materialization.device_objects.size() != 706 ||
        mtp_plan.materialization.host_objects.size() != 4 ||
        mtp_plan.materialization.device_capacity_bytes != 17'453'691'904ULL ||
        !mtp_plan.bindings.assistant.has_value() ||
        mtp_plan.bindings.assistant->output_head.index !=
            mtp_plan.bindings.assistant->token_embedding.index) {
        throw std::runtime_error("real Gemma assistant residency plan is incomplete");
    }
}

} // namespace

int main() {
    try {
        test_complete_target();
        test_missing_and_malformed_objects();
        test_real_artifact_when_requested();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
