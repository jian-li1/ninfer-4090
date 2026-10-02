#include <ninfer/targets/gemma4/frontend.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::ChatMessage;
using ninfer::ChatRole;
using ninfer::MessagePart;
using ninfer::MessagePartKind;
using ninfer::PromptInput;
using ninfer::RequestError;
using ninfer::RequestErrorKind;
using ninfer::TokenId;
using ninfer::ToolCall;
using ninfer::targets::gemma4::Frontend;
using ninfer::targets::gemma4::FrontendOptions;
using ninfer::targets::gemma4::FrontendResources;

constexpr std::array<char, 8> kMagic{'N', 'I', 'N', 'F', 'E', 'R', '\0', '\2'};

std::uint64_t read_u64_le(const std::array<unsigned char, 8>& bytes) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
    }
    return value;
}

FrontendResources read_resources(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("failed to open Gemma artifact"); }
    std::array<char, 8> magic{};
    std::array<unsigned char, 8> json_size_bytes{};
    stream.read(magic.data(), magic.size());
    stream.read(reinterpret_cast<char*>(json_size_bytes.data()), json_size_bytes.size());
    if (!stream || magic != kMagic) { throw std::runtime_error("invalid Gemma artifact prefix"); }
    const std::uint64_t json_size = read_u64_le(json_size_bytes);
    std::string directory(static_cast<std::size_t>(json_size), '\0');
    stream.read(directory.data(), static_cast<std::streamsize>(directory.size()));
    const auto root = nlohmann::json::parse(directory);
    const std::uint64_t payload_offset = (16ULL + json_size + 4095ULL) & ~4095ULL;
    const auto read = [&](std::string_view name) {
        const auto found = std::find_if(root.at("objects").begin(), root.at("objects").end(),
                                        [&](const nlohmann::json& object) {
                                            return object.at("name").get<std::string_view>() == name;
                                        });
        if (found == root.at("objects").end() || found->at("kind") != "resource") {
            throw std::runtime_error("Gemma artifact omits frontend resource " +
                                     std::string(name));
        }
        const std::uint64_t bytes = found->at("bytes").get<std::uint64_t>();
        std::string value(static_cast<std::size_t>(bytes), '\0');
        stream.seekg(static_cast<std::streamoff>(payload_offset +
                                                 found->at("offset").get<std::uint64_t>()));
        stream.read(value.data(), static_cast<std::streamsize>(value.size()));
        if (!stream) { throw std::runtime_error("failed to read Gemma frontend resource"); }
        return value;
    };
    return {
        .tokenizer_json         = read("frontend/tokenizer.json"),
        .tokenizer_config_json  = read("frontend/tokenizer_config.json"),
        .chat_template_jinja    = read("frontend/chat_template.jinja"),
        .generation_config_json = read("frontend/generation_config.json"),
    };
}

ChatMessage message(ChatRole role, std::string text) {
    return {.role = role,
            .parts = {MessagePart{.kind = MessagePartKind::Text, .text = std::move(text)}}};
}

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

void require_render(const Frontend& frontend, PromptInput input, std::string_view expected,
                    std::string_view label) {
    const auto prepared = frontend.prepare(std::move(input));
    const std::vector<TokenId> expected_tokens = frontend.tokenize_text(expected);
    require(std::ranges::equal(prepared.token_ids(), expected_tokens), label);
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_GEMMA4_ARTIFACT");
    if (artifact == nullptr) { return 77; }
    try {
        FrontendResources resources = read_resources(artifact);
        Frontend frontend = ninfer::targets::gemma4::make_frontend(
            resources, FrontendOptions{.max_context = 4096});

        PromptInput thinking;
        thinking.messages.push_back(message(ChatRole::User, "Hello"));
        require_render(frontend, thinking,
                       "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello"
                       "<turn|>\n<|turn>model\n",
                       "thinking prompt differs from the pinned Gemma template");

        PromptInput direct = thinking;
        direct.options.enable_thinking = false;
        require_render(frontend, direct,
                       "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"
                       "<|channel>thought\n<channel|>",
                       "non-thinking prompt differs from the pinned Gemma template");

        PromptInput tools;
        tools.options.enable_thinking = true;
        tools.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"weather","description":"Weather","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City"}},"required":["city"]}}})");
        tools.messages.push_back(message(ChatRole::User, "Plan"));
        ChatMessage assistant = message(ChatRole::Assistant, "");
        assistant.tool_calls.push_back(
            ToolCall{.id = "call_1", .name = "weather", .arguments_json = R"({"city":"Paris"})"});
        tools.messages.push_back(std::move(assistant));
        ChatMessage response = message(ChatRole::Tool, "sunny");
        response.tool_call_id = "call_1";
        tools.messages.push_back(std::move(response));
        require_render(
            frontend, tools,
            "<bos><|turn>system\n<|think|>\n"
            "<|tool>declaration:weather{description:<|\"|>Weather<|\"|>,parameters:{"
            "properties:{city:{description:<|\"|>City<|\"|>,type:<|\"|>STRING<|\"|>}},"
            "required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n"
            "<|turn>user\nPlan<turn|>\n<|turn>model\n"
            "<|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
            "<|tool_response>response:weather{value:<|\"|>sunny<|\"|>}<tool_response|>"
            "<|channel>thought\n",
            "tool loop differs from the pinned Gemma template");

        PromptInput media = direct;
        media.messages.front().parts.push_back(MessagePart{
            .kind = MessagePartKind::Media,
            .media = {.kind = ninfer::MediaKind::Image, .media_type = "image/png"}});
        bool rejected_media = false;
        try {
            (void)frontend.prepare(std::move(media));
        } catch (const RequestError& error) {
            rejected_media = error.kind() == RequestErrorKind::InvalidMedia &&
                             std::string_view(error.what()).find("text-only") != std::string_view::npos;
        }
        require(rejected_media, "Gemma frontend did not clearly reject media");

        auto prepared = frontend.prepare(thinking);
        auto output = frontend.make_output_session(prepared, {});
        const std::vector<TokenId> emitted = frontend.tokenize_text(
            "work<channel|><|channel>final\n"
            "<|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
            "<|tool_response>");
        const auto decision = output.preview_model(emitted, 256, ninfer::FinishReason::OutputLimit);
        require(decision.finish_reason == ninfer::FinishReason::StopToken,
                "Gemma official tool-response stop token was not terminal");
        const auto deltas = output.commit_preview();
        require(std::none_of(deltas.begin(), deltas.end(), [](const ninfer::OutputDelta& delta) {
                    return delta.text.find("<|tool_call>") != std::string::npos;
                }),
                "structured Gemma tool syntax leaked into published text");
        auto calls = output.take_tool_calls();
        const auto diagnostics = output.tool_call_parse_diagnostics();
        std::string published;
        for (const auto& delta : deltas) {
            published += delta.channel == ninfer::OutputChannel::Reasoning ? "[reasoning]" : "[content]";
            published += delta.text;
        }
        require(calls.size() == 1,
                "Gemma tool call was not published (markers=" +
                    std::to_string(diagnostics.marker_seen) + ", fallback=" +
                    std::to_string(static_cast<int>(diagnostics.fallback_reason)) +
                    ", deltas=" + published + ")");
        require(calls.front().name == "weather",
                "Gemma tool call published the wrong name: " + calls.front().name);
        require(nlohmann::json::parse(calls.front().arguments_json).at("city") == "Paris",
                "Gemma tool arguments were not published as valid JSON: " +
                    calls.front().arguments_json);

        FrontendResources wrong_template = resources;
        wrong_template.chat_template_jinja.push_back(' ');
        bool rejected_template = false;
        try {
            (void)ninfer::targets::gemma4::make_frontend(
                wrong_template, FrontendOptions{.max_context = 4096});
        } catch (const std::invalid_argument& error) {
            rejected_template = std::string_view(error.what()).find("unsupported Gemma") !=
                                std::string_view::npos;
        }
        require(rejected_template, "Gemma frontend accepted an unpinned chat template");
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
