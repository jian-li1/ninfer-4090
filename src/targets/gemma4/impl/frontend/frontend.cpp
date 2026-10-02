#include <ninfer/targets/gemma4/frontend.h>

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "targets/qwen3_6/impl/frontend/digest.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace ninfer::targets::gemma4 {
namespace {

using Json      = nlohmann::json;
using Tokenizer = qwen3_6::frontend_internal::Tokenizer;
using Clock     = std::chrono::steady_clock;

constexpr std::string_view kBos             = "<bos>";
constexpr std::string_view kTurnOpen        = "<|turn>";
constexpr std::string_view kTurnClose       = "<turn|>\n";
constexpr std::string_view kThinkEnable     = "<|think|>\n";
constexpr std::string_view kThoughtOpen     = "<|channel>thought\n";
constexpr std::string_view kChannelClose    = "<channel|>";
constexpr std::string_view kFinalOpen       = "<|channel>final\n";
constexpr std::string_view kToolCallOpen    = "<|tool_call>call:";
constexpr std::string_view kToolCallClose   = "<tool_call|>";
constexpr std::string_view kToolResponseOpen  = "<|tool_response>response:";
constexpr std::string_view kToolResponseClose = "<tool_response|>";
constexpr std::string_view kThinkingControl = "<channel|><|channel>final\n";
constexpr std::string_view kPinnedTemplateSha256 =
    "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4";

[[noreturn]] void context_too_long(std::uint32_t maximum) {
    throw RequestError(RequestErrorKind::ContextLengthExceeded,
                       "prepared prompt exceeds Engine max_context " +
                           std::to_string(maximum));
}

void check_control(const PreparationControl& control, std::string_view phase = {}) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "Gemma prompt preparation was cancelled");
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "Gemma prompt preparation deadline expired" +
                               (phase.empty() ? std::string{} : " during " + std::string(phase)));
    }
}

std::string trim(std::string_view text) {
    std::size_t first = 0;
    while (first < text.size() &&
           std::isspace(static_cast<unsigned char>(text[first])) != 0) {
        ++first;
    }
    std::size_t last = text.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(text[last - 1])) != 0) {
        --last;
    }
    return std::string(text.substr(first, last - first));
}

void reject_media(const PromptInput& input) {
    for (const ChatMessage& message : input.messages) {
        for (const MessagePart& part : message.parts) {
            if (part.kind == MessagePartKind::Media) {
                const char* kind = part.media.kind == MediaKind::Image ? "image" : "video";
                throw RequestError(
                    RequestErrorKind::InvalidMedia,
                    std::string("Gemma 4 31B is a text-only target; ") + kind +
                        " content requires the optional vision extension, which is not enabled");
            }
        }
    }
}

std::string message_text(const ChatMessage& message, bool separate_parts = false) {
    std::string out;
    for (const MessagePart& part : message.parts) {
        if (part.kind != MessagePartKind::Text) { continue; }
        if (separate_parts && !out.empty()) { out.push_back(' '); }
        out += trim(part.text);
    }
    return out;
}

std::string strip_thinking(std::string_view text) {
    std::string out;
    std::size_t first = 0;
    while (first <= text.size()) {
        const std::size_t close = text.find(kChannelClose, first);
        const std::string_view part = text.substr(
            first, close == std::string_view::npos ? text.size() - first : close - first);
        const std::size_t open = part.find("<|channel>");
        out.append(part.substr(0, open));
        if (close == std::string_view::npos) { break; }
        first = close + kChannelClose.size();
    }
    return trim(out);
}

std::string quote_template_string(std::string_view value) {
    // Gemma's pinned template uses its own quoted-string delimiters rather than JSON quotes.
    return "<|\"|>" + std::string(value) + "<|\"|>";
}

std::string format_argument(const Json& value, bool quote_keys = true) {
    if (value.is_null()) { return "null"; }
    if (value.is_boolean()) { return value.get<bool>() ? "true" : "false"; }
    if (value.is_string()) { return quote_template_string(value.get_ref<const std::string&>()); }
    if (value.is_number()) { return value.dump(); }
    if (value.is_array()) {
        std::string out = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) { out.push_back(','); }
            out += format_argument(value[index], quote_keys);
        }
        out.push_back(']');
        return out;
    }
    if (value.is_object()) {
        std::vector<std::string> keys;
        keys.reserve(value.size());
        for (const auto& [key, _] : value.items()) { keys.push_back(key); }
        std::sort(keys.begin(), keys.end());
        std::string out = "{";
        for (std::size_t index = 0; index < keys.size(); ++index) {
            if (index != 0) { out.push_back(','); }
            out += quote_keys ? quote_template_string(keys[index]) : keys[index];
            out.push_back(':');
            out += format_argument(value.at(keys[index]), quote_keys);
        }
        out.push_back('}');
        return out;
    }
    throw std::invalid_argument("Gemma tool schema contains an unsupported JSON value");
}

void append_comma(std::string& out, bool& found) {
    if (found) { out.push_back(','); }
    found = true;
}

std::string format_required(const Json& required) {
    std::string out = "[";
    if (required.is_array()) {
        for (std::size_t index = 0; index < required.size(); ++index) {
            if (index != 0) { out.push_back(','); }
            out += quote_template_string(required[index].get<std::string>());
        }
    }
    out.push_back(']');
    return out;
}

std::string uppercase_type(const Json& type) {
    if (type.is_string()) {
        std::string value = type.get<std::string>();
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        return quote_template_string(value);
    }
    if (type.is_array()) {
        Json values = Json::array();
        for (const Json& item : type) {
            std::string value = item.get<std::string>();
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            values.push_back(value);
        }
        return format_argument(values);
    }
    return format_argument(type);
}

std::string format_parameters(const Json& properties, bool filter_keys = false) {
    static const std::array<std::string_view, 5> standard{
        "description", "type", "properties", "required", "nullable"};
    if (!properties.is_object()) { return {}; }
    std::vector<std::string> keys;
    for (const auto& [key, _] : properties.items()) {
        if (filter_keys && std::find(standard.begin(), standard.end(), key) != standard.end()) {
            continue;
        }
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    std::string out;
    bool first_property = true;
    for (const std::string& key : keys) {
        const Json& value = properties.at(key);
        if (!value.is_object()) { continue; }
        if (!first_property) { out.push_back(','); }
        first_property = false;
        out += key + ":{";
        bool field = false;
        if (value.contains("description") && value["description"].is_string() &&
            !value["description"].get_ref<const std::string&>().empty()) {
            append_comma(out, field);
            out += "description:" +
                   quote_template_string(value["description"].get<std::string>());
        }
        std::string type;
        if (value.contains("type") && value["type"].is_string()) {
            type = value["type"].get<std::string>();
            std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
        }
        if (type == "STRING" && value.contains("enum") && value["enum"].is_array() &&
            !value["enum"].empty()) {
            append_comma(out, field);
            out += "enum:" + format_argument(value["enum"]);
        } else if (type == "ARRAY" && value.contains("items") && value["items"].is_object() &&
                   !value["items"].empty()) {
            append_comma(out, field);
            out += "items:{";
            bool item_field = false;
            std::vector<std::string> item_keys;
            for (const auto& [item_key, item_value] : value["items"].items()) {
                if (!item_value.is_null()) { item_keys.push_back(item_key); }
            }
            std::sort(item_keys.begin(), item_keys.end());
            for (const std::string& item_key : item_keys) {
                append_comma(out, item_field);
                const Json& item = value["items"].at(item_key);
                if (item_key == "properties") {
                    out += "properties:{" + format_parameters(item) + "}";
                } else if (item_key == "required") {
                    out += "required:" + format_required(item);
                } else if (item_key == "type") {
                    out += "type:" + uppercase_type(item);
                } else {
                    out += item_key + ":" + format_argument(item);
                }
            }
            out.push_back('}');
        }
        if (value.value("nullable", false)) {
            append_comma(out, field);
            out += "nullable:true";
        }
        if (type == "OBJECT") {
            append_comma(out, field);
            if (value.contains("properties") && value["properties"].is_object()) {
                out += "properties:{" + format_parameters(value["properties"]) + "}";
            } else {
                out += "properties:{" + format_parameters(value, true) + "}";
            }
            if (value.contains("required") && value["required"].is_array() &&
                !value["required"].empty()) {
                append_comma(out, field);
                out += "required:" + format_required(value["required"]);
            }
        }
        append_comma(out, field);
        out += "type:" + quote_template_string(type);
        out.push_back('}');
    }
    return out;
}

std::string format_tool(const std::string& encoded) {
    Json tool;
    try {
        tool = Json::parse(encoded);
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("malformed tool definition: ") + error.what());
    }
    if (!tool.is_object() || !tool.contains("function") || !tool["function"].is_object() ||
        !tool["function"].contains("name") || !tool["function"]["name"].is_string()) {
        throw std::invalid_argument("Gemma tool definition must contain function.name");
    }
    const Json& function = tool["function"];
    std::string out = "<|tool>declaration:" + function["name"].get<std::string>() + "{";
    out += "description:" + quote_template_string(function.value("description", std::string{}));
    if (function.contains("parameters") && function["parameters"].is_object() &&
        !function["parameters"].empty()) {
        const Json& parameters = function["parameters"];
        out += ",parameters:{";
        if (parameters.contains("properties") && parameters["properties"].is_object() &&
            !parameters["properties"].empty()) {
            out += "properties:{" + format_parameters(parameters["properties"]) + "},";
        }
        if (parameters.contains("required") && parameters["required"].is_array() &&
            !parameters["required"].empty()) {
            out += "required:" + format_required(parameters["required"]) + ",";
        }
        if (parameters.contains("type")) {
            out += "type:" + uppercase_type(parameters["type"]);
        }
        out.push_back('}');
    }
    if (function.contains("response") && function["response"].is_object()) {
        const Json& response = function["response"];
        out += ",response:{";
        if (response.contains("description") && response["description"].is_string() &&
            !response["description"].get_ref<const std::string&>().empty()) {
            out += "description:" + quote_template_string(response["description"].get<std::string>()) + ",";
        }
        if (response.contains("type")) { out += "type:" + uppercase_type(response["type"]); }
        out.push_back('}');
    }
    out += "}<tool|>";
    return out;
}

std::string tool_name_for(const std::unordered_map<std::string, std::string>& calls,
                          const ChatMessage& message) {
    const auto found = calls.find(message.tool_call_id);
    return found == calls.end() ? "unknown" : found->second;
}

struct RenderedPrompt {
    std::string text;
    std::vector<std::size_t> message_byte_boundaries;
    bool starts_in_reasoning = false;
};

RenderedPrompt render_prompt(const PromptInput& input) {
    RenderedPrompt rendered;
    rendered.text += kBos;
    std::size_t first_message = 0;
    const bool leading_instruction =
        !input.messages.empty() &&
        (input.messages.front().role == ChatRole::System ||
         input.messages.front().role == ChatRole::Developer);
    if (input.options.enable_thinking || !input.options.tool_jsons.empty() ||
        leading_instruction) {
        rendered.text += "<|turn>system\n";
        if (input.options.enable_thinking) { rendered.text += kThinkEnable; }
        if (leading_instruction) {
            rendered.text += trim(message_text(input.messages.front()));
            first_message = 1;
        }
        for (const std::string& tool : input.options.tool_jsons) {
            rendered.text += format_tool(tool);
        }
        rendered.text += kTurnClose;
    }

    std::size_t last_user = std::numeric_limits<std::size_t>::max();
    for (std::size_t index = 0; index < input.messages.size(); ++index) {
        if (input.messages[index].role == ChatRole::User) { last_user = index; }
    }
    std::unordered_map<std::string, std::string> call_names;
    ChatRole previous_non_tool = ChatRole::Tool;
    enum class MessageType { None, ToolCall, ToolResponse };
    MessageType previous_type = MessageType::None;
    for (std::size_t index = first_message; index < input.messages.size();) {
        const ChatMessage& message = input.messages[index];
        if (message.role == ChatRole::Tool) {
            // Tool responses are serialized in the preceding model turn by the canonical
            // template. A standalone response remains explicit and resolvable by call id.
            rendered.text += kToolResponseOpen;
            rendered.text += tool_name_for(call_names, message);
            rendered.text += "{value:" + quote_template_string(message_text(message)) + "}";
            rendered.text += kToolResponseClose;
            rendered.message_byte_boundaries.push_back(rendered.text.size());
            previous_type = MessageType::ToolResponse;
            ++index;
            continue;
        }

        const bool assistant = message.role == ChatRole::Assistant;
        const bool continue_model = assistant && previous_non_tool == ChatRole::Assistant;
        if (!continue_model) {
            rendered.text += kTurnOpen;
            switch (message.role) {
            case ChatRole::System: rendered.text += "system\n"; break;
            case ChatRole::Developer: rendered.text += "developer\n"; break;
            case ChatRole::User: rendered.text += "user\n"; break;
            case ChatRole::Assistant: rendered.text += "model\n"; break;
            case ChatRole::Tool: break;
            }
        }
        const bool thinking_gate =
            (last_user == std::numeric_limits<std::size_t>::max() || index > last_user) ||
            (input.options.preserve_thinking && !message.tool_calls.empty());
        if (assistant && !message.reasoning_content.empty() && thinking_gate) {
            rendered.text += kThoughtOpen;
            rendered.text += message.reasoning_content;
            rendered.text += "\n";
            rendered.text += kChannelClose;
        }
        for (const ToolCall& call : message.tool_calls) {
            Json arguments;
            try {
                arguments = Json::parse(call.arguments_json.empty() ? "{}" : call.arguments_json);
            } catch (const Json::exception& error) {
                throw std::invalid_argument(std::string("malformed tool call arguments: ") +
                                            error.what());
            }
            if (!arguments.is_object()) {
                throw std::invalid_argument("Gemma tool call arguments must be a JSON object");
            }
            rendered.text += kToolCallOpen;
            rendered.text += call.name;
            rendered.text += format_argument(arguments, false);
            rendered.text += kToolCallClose;
            call_names[call.id] = call.name;
        }
        const std::string content = assistant ? strip_thinking(message_text(message))
                                              : message_text(message);

        bool followed_by_tool = false;
        std::size_t scan = index + 1;
        if (assistant && index + 1 < input.messages.size() &&
            input.messages[index + 1].role == ChatRole::Tool) {
            followed_by_tool = true;
            while (scan < input.messages.size() && input.messages[scan].role == ChatRole::Tool) {
                const ChatMessage& response = input.messages[scan];
                rendered.text += kToolResponseOpen;
                rendered.text += tool_name_for(call_names, response);
                rendered.text += "{value:" + quote_template_string(message_text(response)) + "}";
                rendered.text += kToolResponseClose;
                ++scan;
            }
        }
        rendered.text += content;

        std::optional<ChatRole> next_non_tool;
        for (std::size_t next = scan; next < input.messages.size(); ++next) {
            if (input.messages[next].role != ChatRole::Tool) {
                next_non_tool = input.messages[next].role;
                break;
            }
        }
        const bool continues_into_next =
            assistant && next_non_tool == ChatRole::Assistant &&
            (message.tool_calls.empty() || followed_by_tool);
        if (assistant && !message.tool_calls.empty() && !followed_by_tool) {
            rendered.text += kToolResponseOpen;
            previous_type = MessageType::ToolCall;
        } else if (continues_into_next) {
            previous_type = followed_by_tool ? MessageType::ToolResponse : MessageType::None;
        } else if (!(followed_by_tool && trim(content).empty() && !next_non_tool)) {
            rendered.text += kTurnClose;
            previous_type = followed_by_tool ? MessageType::ToolResponse : MessageType::None;
        } else {
            previous_type = MessageType::ToolResponse;
        }
        previous_non_tool = message.role;
        rendered.message_byte_boundaries.push_back(rendered.text.size());
        index = followed_by_tool ? scan : index + 1;
    }

    if (input.options.continuation == PromptContinuationMode::NewAssistantTurn) {
        if (previous_type != MessageType::ToolResponse && previous_type != MessageType::ToolCall) {
            rendered.text += "<|turn>model\n";
            if (!input.options.enable_thinking) {
                rendered.text += kThoughtOpen;
                rendered.text += kChannelClose;
            }
        } else if (previous_type == MessageType::ToolResponse && input.options.enable_thinking) {
            rendered.text += kThoughtOpen;
        }
        rendered.starts_in_reasoning = input.options.enable_thinking &&
                                       previous_type != MessageType::ToolCall;
    }
    return rendered;
}

qwen3_6::PreparedContextCache prepare_cache(const ContextCacheHints& hints) {
    qwen3_6::PreparedContextCache out;
    if (hints.session_key) {
        if (hints.session_key->empty() ||
            hints.session_key->size() > qwen3_6::kPreparedSessionKeyCapacity) {
            throw std::invalid_argument("context cache session_key must contain 1 to 256 bytes");
        }
        qwen3_6::PreparedSessionKey key;
        key.size = static_cast<std::uint16_t>(hints.session_key->size());
        std::copy(hints.session_key->begin(), hints.session_key->end(), key.bytes.begin());
        out.session_key = key;
    }
    switch (hints.retention) {
    case CacheRetentionHint::Default:
        out.retention = out.session_key ? runtime::RetentionClass::LiveSession
                                        : runtime::RetentionClass::RecentPrivate;
        break;
    case CacheRetentionHint::LiveSession:
        if (!out.session_key) {
            throw std::invalid_argument("LiveSession retention requires a session_key");
        }
        out.retention = runtime::RetentionClass::LiveSession;
        break;
    case CacheRetentionHint::Disposable:
        out.retention = runtime::RetentionClass::Disposable;
        break;
    }
    out.update_session_index = hints.update_session_index;
    return out;
}

StopPolicy merge_stop_policy(const Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto add = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                        std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (int token : tokenizer.default_stop_token_ids()) { add(token); }
    }
    for (TokenId token : caller.token_ids) { add(token); }
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        result.strings.push_back(stop);
    }
    return result;
}

struct RenderedOutput {
    std::string reasoning;
    std::string content;
    std::string structured_content;
    bool reasoning_closed = false;
};

std::string publishable_content(std::string_view content) {
    std::string out;
    std::size_t begin = 0;
    while (begin < content.size()) {
        const std::size_t open = content.find(kToolCallOpen, begin);
        if (open == std::string_view::npos) {
            out.append(content.substr(begin));
            break;
        }
        out.append(content.substr(begin, open - begin));
        const std::size_t close = content.find(kToolCallClose, open + kToolCallOpen.size());
        if (close == std::string_view::npos) { break; }
        begin = close + kToolCallClose.size();
    }
    return out;
}

bool is_output_structure(std::string_view token) noexcept {
    return token == "<|tool>" || token == "<tool|>" || token == "<|tool_call>" ||
           token == "<tool_call|>" || token == "<|\"|>" || token == "<|channel>" ||
           token == "<channel|>";
}

RenderedOutput render_output(std::string_view raw, bool starts_in_reasoning, bool preserve_special) {
    if (preserve_special || !starts_in_reasoning) {
        return {.content = std::string(raw),
                .structured_content = std::string(raw),
                .reasoning_closed = !starts_in_reasoning};
    }
    RenderedOutput out;
    const std::size_t close = raw.find(kChannelClose);
    if (close == std::string_view::npos) {
        const std::size_t hold = [&] {
            const std::size_t maximum = std::min(raw.size(), kChannelClose.size());
            for (std::size_t size = maximum; size != 0; --size) {
                if (raw.substr(raw.size() - size) == kChannelClose.substr(0, size)) { return size; }
            }
            return std::size_t{0};
        }();
        out.reasoning.assign(raw.substr(0, raw.size() - hold));
        return out;
    }
    out.reasoning.assign(raw.substr(0, close));
    std::string_view content = raw.substr(close + kChannelClose.size());
    if (content.starts_with(kFinalOpen)) { content.remove_prefix(kFinalOpen.size()); }
    while (!content.empty() && std::isspace(static_cast<unsigned char>(content.front())) != 0) {
        content.remove_prefix(1);
    }
    out.structured_content.assign(content);
    out.content = publishable_content(content);
    out.reasoning_closed = true;
    return out;
}

bool valid_tool_name(std::string_view name, std::uint32_t maximum) {
    return !name.empty() && name.size() <= maximum &&
           std::all_of(name.begin(), name.end(), [](char c) {
               return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '-';
           });
}

class ArgumentParser {
public:
    explicit ArgumentParser(std::string_view source) : source_(source) {}

    Json parse() {
        Json value = parse_value();
        skip_space();
        if (cursor_ != source_.size()) { fail(); }
        return value;
    }

private:
    [[noreturn]] static void fail() {
        throw std::invalid_argument("malformed Gemma tool-call arguments");
    }

    void skip_space() {
        while (cursor_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[cursor_])) != 0) {
            ++cursor_;
        }
    }

    bool take(char value) {
        skip_space();
        if (cursor_ == source_.size() || source_[cursor_] != value) { return false; }
        ++cursor_;
        return true;
    }

    bool take(std::string_view value) {
        skip_space();
        if (!source_.substr(cursor_).starts_with(value)) { return false; }
        cursor_ += value.size();
        return true;
    }

    std::string parse_string() {
        constexpr std::string_view delimiter = "<|\"|>";
        if (!take(delimiter)) { fail(); }
        const std::size_t end = source_.find(delimiter, cursor_);
        if (end == std::string_view::npos) { fail(); }
        std::string value(source_.substr(cursor_, end - cursor_));
        cursor_ = end + delimiter.size();
        return value;
    }

    std::string parse_key() {
        skip_space();
        if (source_.substr(cursor_).starts_with("<|\"|>")) { return parse_string(); }
        const std::size_t first = cursor_;
        while (cursor_ < source_.size() && source_[cursor_] != ':' &&
               source_[cursor_] != ',' && source_[cursor_] != '}') {
            ++cursor_;
        }
        std::string key = trim(source_.substr(first, cursor_ - first));
        if (key.empty()) { fail(); }
        return key;
    }

    Json parse_object() {
        if (!take('{')) { fail(); }
        Json object = Json::object();
        if (take('}')) { return object; }
        while (true) {
            std::string key = parse_key();
            if (!take(':')) { fail(); }
            object[std::move(key)] = parse_value();
            if (take('}')) { return object; }
            if (!take(',')) { fail(); }
        }
    }

    Json parse_array() {
        if (!take('[')) { fail(); }
        Json array = Json::array();
        if (take(']')) { return array; }
        while (true) {
            array.push_back(parse_value());
            if (take(']')) { return array; }
            if (!take(',')) { fail(); }
        }
    }

    Json parse_value() {
        skip_space();
        if (cursor_ == source_.size()) { fail(); }
        if (source_[cursor_] == '{') { return parse_object(); }
        if (source_[cursor_] == '[') { return parse_array(); }
        if (source_.substr(cursor_).starts_with("<|\"|>")) { return parse_string(); }
        if (take("true")) { return true; }
        if (take("false")) { return false; }
        if (take("null")) { return nullptr; }
        const std::size_t first = cursor_;
        while (cursor_ < source_.size() && source_[cursor_] != ',' &&
               source_[cursor_] != '}' && source_[cursor_] != ']' &&
               std::isspace(static_cast<unsigned char>(source_[cursor_])) == 0) {
            ++cursor_;
        }
        if (cursor_ == first) { fail(); }
        Json number = Json::parse(source_.substr(first, cursor_ - first), nullptr, false);
        if (number.is_discarded() || !number.is_number()) { fail(); }
        return number;
    }

    std::string_view source_;
    std::size_t cursor_ = 0;
};

std::vector<GeneratedToolCall> parse_tool_calls(std::string& content, std::uint32_t maximum,
                                                ToolCallParseDiagnostics& diagnostics) {
    std::vector<GeneratedToolCall> calls;
    std::size_t cursor = content.find(kToolCallOpen);
    if (cursor == std::string::npos) { return calls; }
    diagnostics.marker_seen = true;
    std::string ordinary = content.substr(0, cursor);
    while (cursor != std::string::npos) {
        const std::size_t body = cursor + kToolCallOpen.size();
        const std::size_t close = content.find(kToolCallClose, body);
        if (close == std::string::npos) {
            diagnostics.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            return {};
        }
        const std::string_view region(content.data() + body, close - body);
        const std::size_t brace = region.find('{');
        if (brace == std::string_view::npos || !region.ends_with('}')) {
            diagnostics.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            return {};
        }
        std::string_view name = region.substr(0, brace);
        if (name.starts_with("call:")) { name.remove_prefix(5); }
        if (!valid_tool_name(name, maximum)) {
            diagnostics.fallback_reason = ToolCallParseFallbackReason::InvalidToolName;
            return {};
        }
        Json arguments;
        try {
            arguments = ArgumentParser(region.substr(brace)).parse();
        } catch (const std::invalid_argument&) {
            diagnostics.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            return {};
        }
        if (!arguments.is_object()) {
            diagnostics.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            return {};
        }
        calls.push_back({.name = std::string(name), .arguments_json = arguments.dump()});
        ++diagnostics.structured_call_count;
        cursor = content.find(kToolCallOpen, close + kToolCallClose.size());
        if (cursor == std::string::npos) {
            ordinary += content.substr(close + kToolCallClose.size());
        } else {
            ordinary += content.substr(close + kToolCallClose.size(),
                                       cursor - close - kToolCallClose.size());
        }
    }
    content = std::move(ordinary);
    return calls;
}

std::shared_ptr<const Tokenizer> make_tokenizer(const FrontendResources& resources) {
    const std::string template_digest = qwen3_6::frontend_internal::sha256_hex(
        qwen3_6::frontend_internal::sha256(resources.chat_template_jinja));
    if (template_digest != kPinnedTemplateSha256) {
        throw std::invalid_argument("unsupported Gemma frontend/chat_template.jinja (sha256 " +
                                    template_digest + ")");
    }
    return std::make_shared<const Tokenizer>(qwen3_6::frontend_internal::TokenizerResources{
        .tokenizer_json         = resources.tokenizer_json,
        .tokenizer_config_json  = resources.tokenizer_config_json,
        .generation_config_json = resources.generation_config_json});
}

} // namespace

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, FrontendOptions options)
        : tokenizer(make_tokenizer(resources)),
          max_context(options.max_context) {
        if (max_context == 0) { throw std::invalid_argument("frontend max_context must be nonzero"); }
        if (!tokenizer->has_exact_token_domain(262144)) {
            throw std::invalid_argument("Gemma tokenizer does not expose the registered 262144-token domain");
        }
        const std::vector<int> bos = tokenizer->encode(kBos);
        if (bos.size() != 1 || bos.front() != 2) {
            throw std::invalid_argument("Gemma tokenizer does not map <bos> to token 2");
        }
        for (int expected : {1, 106, 50}) {
            if (std::find(tokenizer->default_stop_token_ids().begin(),
                          tokenizer->default_stop_token_ids().end(), expected) ==
                tokenizer->default_stop_token_ids().end()) {
                throw std::invalid_argument("Gemma generation config omits an official stop token");
            }
        }
        defaults.token_ids.assign(tokenizer->default_stop_token_ids().begin(),
                                  tokenizer->default_stop_token_ids().end());
        thinking_control = tokenizer->encode(kThinkingControl);
        if (thinking_control.empty()) {
            throw std::invalid_argument("Gemma tokenizer cannot encode its thinking close control");
        }
    }

    std::shared_ptr<const Tokenizer> tokenizer;
    std::uint32_t max_context = 0;
    StopPolicy defaults;
    std::vector<TokenId> thinking_control;
};

struct OutputState {
    std::string raw;
    RenderedOutput rendered;
    std::uint32_t reasoning_tokens = 0;
    bool terminal = false;
};

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const Tokenizer> tokenizer_in, StopPolicy policy_in, OutputOptions output,
         bool starts, ThinkingControlOptions thinking_in, std::vector<TokenId> control)
        : tokenizer(std::move(tokenizer_in)), policy(std::move(policy_in)),
          thinking_control(std::move(control)), starts_in_reasoning(starts),
          preserve_special(output.raw || output.preserve_special_tokens),
          raw_output(output.raw), tool_name_max_length(output.tool_name_max_length),
          thinking(thinking_in) {
        if (thinking.budget && *thinking.budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        state.rendered = render_output({}, starts_in_reasoning, preserve_special);
    }

    std::shared_ptr<const Tokenizer> tokenizer;
    StopPolicy policy;
    std::vector<TokenId> thinking_control;
    OutputState state;
    OutputState preview;
    bool starts_in_reasoning = false;
    bool preserve_special = false;
    bool raw_output = false;
    std::uint32_t tool_name_max_length = 128;
    ThinkingControlOptions thinking;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens = 0;
    bool control_pending = false;
    bool preview_control_pending = false;
    bool control_applied = false;
    bool preview_ready = false;
    std::optional<std::string> matched_stop;
    std::optional<std::string> preview_matched_stop;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_diagnostics;
};

OutputSession::OutputSession() noexcept = default;
OutputSession::~OutputSession() = default;
OutputSession::OutputSession(OutputSession&&) noexcept = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;
OutputSession::OutputSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t remaining,
                                                     FinishReason limit_reason) {
    if (!impl_ || impl_->state.terminal || impl_->preview_ready || tokens.empty() ||
        tokens.size() > remaining) {
        throw std::logic_error("invalid Gemma output preview state");
    }
    impl_->preview = impl_->state;
    impl_->preview_control_pending = impl_->control_pending;
    impl_->preview_matched_stop.reset();
    std::uint32_t accepted = 0;
    FinishReason reason = FinishReason::None;
    for (TokenId token : tokens) {
        ++accepted;
        const bool stop_token = std::find(impl_->policy.token_ids.begin(),
                                          impl_->policy.token_ids.end(), token) !=
                                impl_->policy.token_ids.end();
        if (!stop_token || impl_->policy.publish_stop_token) {
            const auto decoded = impl_->tokenizer->decoded_token(token);
            if (impl_->preserve_special || !decoded.special || is_output_structure(decoded.bytes)) {
                impl_->preview.raw += decoded.bytes;
            }
        }
        impl_->preview.rendered = render_output(impl_->preview.raw, impl_->starts_in_reasoning,
                                                impl_->preserve_special);
        if (impl_->starts_in_reasoning && !impl_->preview.rendered.reasoning_closed) {
            ++impl_->preview.reasoning_tokens;
        }
        for (const StopString& stop : impl_->policy.strings) {
            std::string* channel = stop.channel == OutputChannel::Reasoning
                                       ? &impl_->preview.rendered.reasoning
                                       : &impl_->preview.rendered.content;
            const std::size_t found = channel->find(stop.text);
            if (found == std::string::npos) { continue; }
            channel->erase(found + (stop.include_in_output ? stop.text.size() : 0));
            impl_->preview_matched_stop = stop.text;
            reason = FinishReason::StopString;
            break;
        }
        if (reason != FinishReason::None) { break; }
        if (stop_token) { reason = FinishReason::StopToken; break; }
    }
    if (reason == FinishReason::None && accepted == remaining) { reason = limit_reason; }
    if (reason != FinishReason::None) { impl_->preview.terminal = true; }
    if (reason == FinishReason::None && impl_->thinking.budget &&
        !impl_->preview.rendered.reasoning_closed &&
        impl_->preview.reasoning_tokens >= *impl_->thinking.budget) {
        impl_->preview_control_pending = true;
    }
    impl_->preview_ready = true;
    return {.accepted_tokens = accepted,
            .finish_reason = reason,
            .continuation = impl_->preview_control_pending
                                ? runtime::ContinuationAction::ApplyTargetControl
                                : runtime::ContinuationAction::Decode};
}

std::uint32_t OutputSession::model_token_budget_remaining(std::uint32_t remaining) const noexcept {
    if (!impl_ || !impl_->thinking.budget || impl_->control_applied ||
        impl_->state.rendered.reasoning_closed) {
        return remaining;
    }
    if (impl_->control_pending || impl_->state.reasoning_tokens >= *impl_->thinking.budget) {
        return 0;
    }
    return std::min(remaining, *impl_->thinking.budget - impl_->state.reasoning_tokens);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    return impl_ && impl_->control_pending ? std::span<const TokenId>(impl_->thinking_control)
                                           : std::span<const TokenId>{};
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t remaining) {
    if (!impl_ || !impl_->control_pending || tokens.size() > remaining ||
        !std::equal(tokens.begin(), tokens.end(), impl_->thinking_control.begin(),
                    impl_->thinking_control.end())) {
        throw std::invalid_argument("Gemma thinking control requires the exact pending span");
    }
    impl_->preview = impl_->state;
    impl_->preview_control_pending = false;
    impl_->preview.raw += kThinkingControl;
    impl_->preview.rendered = render_output(impl_->preview.raw, impl_->starts_in_reasoning,
                                            impl_->preserve_special);
    impl_->preview_ready = true;
    return {.accepted_tokens = static_cast<std::uint32_t>(tokens.size())};
}

void OutputSession::validate_generation_capacity(std::uint32_t capacity) const {
    if (impl_ && impl_->thinking.budget && capacity > *impl_->thinking.budget &&
        capacity - *impl_->thinking.budget <= impl_->thinking_control.size()) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the Gemma close control and one final token");
    }
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (!impl_ || reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid Gemma terminal preview");
    }
    impl_->preview = impl_->state;
    impl_->preview_control_pending = impl_->control_pending;
    impl_->preview.terminal = true;
    impl_->preview_ready = true;
    return {.finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (!impl_ || !impl_->preview_ready) { std::terminate(); }
    std::vector<GeneratedToolCall> parsed_tool_calls;
    ToolCallParseDiagnostics parsed_diagnostics;
    if (impl_->preview.terminal && !impl_->raw_output) {
        parsed_tool_calls = parse_tool_calls(impl_->preview.rendered.structured_content,
                                             impl_->tool_name_max_length,
                                             parsed_diagnostics);
    }
    PublishedOutput out;
    const auto append_delta = [&](OutputChannel channel, const std::string& before,
                                  const std::string& after) {
        if (after.size() >= before.size() &&
            std::equal(before.begin(), before.end(), after.begin()) &&
            after.size() != before.size()) {
            out.push_back({.channel = channel, .text = after.substr(before.size())});
        }
    };
    append_delta(OutputChannel::Reasoning, impl_->state.rendered.reasoning,
                 impl_->preview.rendered.reasoning);
    append_delta(OutputChannel::Content, impl_->state.rendered.content,
                 impl_->preview.rendered.content);
    const bool applied_control = impl_->control_pending &&
                                 impl_->preview.raw.size() > impl_->state.raw.size() &&
                                 impl_->preview.raw.ends_with(kThinkingControl);
    impl_->state = std::move(impl_->preview);
    impl_->matched_stop = std::move(impl_->preview_matched_stop);
    impl_->model_thinking_tokens = impl_->state.reasoning_tokens;
    if (applied_control) {
        impl_->control_pending = false;
        impl_->control_applied = true;
        impl_->injected_tokens += static_cast<std::uint32_t>(impl_->thinking_control.size());
    } else {
        impl_->control_pending = impl_->preview_control_pending;
    }
    impl_->preview_ready = false;
    if (impl_->state.terminal && !impl_->raw_output) {
        impl_->tool_calls = std::move(parsed_tool_calls);
        impl_->tool_diagnostics = parsed_diagnostics;
    }
    return out;
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ ? std::move(impl_->tool_calls) : std::vector<GeneratedToolCall>{};
}
ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ ? impl_->tool_diagnostics : ToolCallParseDiagnostics{};
}
std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ ? impl_->state.reasoning_tokens : 0;
}
ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (!impl_) { return {}; }
    return {.configured_budget = impl_->thinking.budget,
            .model_thinking_tokens = impl_->model_thinking_tokens,
            .injected_tokens = impl_->injected_tokens,
            .applied = impl_->control_applied};
}
std::optional<std::string> OutputSession::matched_stop_string() const {
    return impl_ ? impl_->matched_stop : std::nullopt;
}

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}
Frontend::Frontend(const Frontend&) = default;
Frontend& Frontend::operator=(const Frontend&) = default;
Frontend::Frontend(Frontend&&) noexcept = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend() = default;

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, options));
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    if (!impl_) { throw std::logic_error("frontend is empty"); }
    check_control(control);
    reject_media(input);
    const auto started = Clock::now();
    const RenderedPrompt rendered = render_prompt(input);
    check_control(control, "rendering");
    const auto tokenize_started = Clock::now();
    std::vector<TokenId> tokens = impl_->tokenizer->encode(rendered.text);
    const double tokenize_seconds =
        std::chrono::duration<double>(Clock::now() - tokenize_started).count();
    check_control(control, "tokenization");
    if (tokens.size() > impl_->max_context) { context_too_long(impl_->max_context); }
    qwen3_6::PreparedPromptData data;
    data.token_ids = std::move(tokens);
    data.positions.resize(data.token_ids.size() * 3U);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t token = 0; token < data.token_ids.size(); ++token) {
            data.positions[axis * data.token_ids.size() + token] =
                static_cast<std::int32_t>(token);
        }
    }
    data.token_types.assign(data.token_ids.size(), 0);
    data.context_cache = prepare_cache(input.context_cache);
    data.identity.reusable = true;
    data.starts_in_reasoning = rendered.starts_in_reasoning;
    data.prepare.tokenize_seconds = tokenize_seconds;
    data.prepare.seconds = std::chrono::duration<double>(Clock::now() - started).count();
    return qwen3_6::PreparedPromptAccess::make(std::move(data));
}

std::uint32_t Frontend::count_tokens(PromptInput input,
                                     const PreparationControl& control) const {
    return prepare(std::move(input), control).summary().prompt_tokens;
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> tokens,
                                        bool allow_prefix_identity) const {
    if (!impl_) { throw std::logic_error("frontend is empty"); }
    if (tokens.size() > impl_->max_context) { context_too_long(impl_->max_context); }
    for (TokenId token : tokens) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the Gemma vocabulary: " +
                                    std::to_string(token));
        }
    }
    qwen3_6::PreparedPromptData data;
    data.token_ids = std::move(tokens);
    data.token_types.assign(data.token_ids.size(), 0);
    data.positions.resize(data.token_ids.size() * 3U);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t token = 0; token < data.token_ids.size(); ++token) {
            data.positions[axis * data.token_ids.size() + token] =
                static_cast<std::int32_t>(token);
        }
    }
    data.identity.reusable = allow_prefix_identity;
    data.context_cache.update_session_index = false;
    return qwen3_6::PreparedPromptAccess::make(std::move(data));
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    if (!impl_) { throw std::logic_error("frontend is empty"); }
    return impl_->tokenizer->encode(text);
}

PromptCapabilities Frontend::prompt_capabilities() const noexcept {
    return {.enable_thinking = true};
}
MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output,
                                            const ThinkingControlOptions& thinking) const {
    const auto& data = qwen3_6::PreparedPromptAccess::view(prompt);
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    return OutputSession(std::make_unique<OutputSession::Impl>(
        impl_->tokenizer, std::move(policy), output, data.starts_in_reasoning, thinking,
        impl_->thinking_control));
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

} // namespace ninfer::targets::gemma4
