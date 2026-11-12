//! OpenAI-compatible request and response values. They are plain structs across the HTTP and JSON
//! boundary, because a boundary carries values. `std.json` parses `ChatRequest` from a request
//! body, and `std.json.Stringify` serializes the rest back.
//!
//! Parse. Every optional field has a default, so a minimal request parses. The parse drops
//! unknown fields (`ParseOptions.ignore_unknown_fields`), so a full OpenAI client with tools still
//! round-trips. The map layer logs each of the first `UnknownFields.table_cap` unknown field names
//! once per process. A field that takes more than one JSON form (`content`, `stop`) parses as a
//! `std.json.Value`, and the map layer checks the form.
//!
//! Sampling. Every sampling field is a named field here, so no sampling value drops in silence.
//!
//! Serialize. The serializer omits a null optional field, so every body keeps the OpenAI shape.
//! A streamed `Delta` carries `{role}` first, then `{reasoning_content}` chunks for a think block
//! and `{content}` chunks for the reply. The final chunk carries `{}` plus a `finish_reason`.

const std = @import("std");

// -- types ------------------------------------------------------------------

/// One message as the client sends it. The map layer narrows the free string `role` to `chat.Role`.
pub const ChatMessage = struct {
    role: []const u8,
    /// A string, an array of content parts, or null. An absent field is the empty string.
    content: std.json.Value = .{ .string = "" },
};

pub const ChatRequest = struct {
    model: []const u8 = "",
    messages: []const ChatMessage = &.{},
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    /// Signed, so a negative value gets the range refusal and not a JSON error.
    top_k: ?i64 = null,
    min_p: ?f32 = null,
    max_tokens: ?u32 = null,
    max_completion_tokens: ?u32 = null,
    stream: bool = false,
    seed: ?u64 = null,
    presence_penalty: ?f32 = null,
    repetition_penalty: ?f32 = null,
    frequency_penalty: ?f32 = null,
    /// The vLLM field: -1 takes the server value, 0 turns thinking off.
    thinking_token_budget: ?i64 = null,
    n: ?u32 = null,
    /// A string or an array of strings. Null and an absent field set no stop string.
    stop: std.json.Value = .null,
    chat_template_kwargs: ?ChatTemplateKwargs = null,
};

/// The template switches of the vLLM and llama.cpp servers. Only `enable_thinking` has an effect.
pub const ChatTemplateKwargs = struct {
    enable_thinking: ?bool = null,
};

pub const Usage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    /// Null when the turn has no think tokens, so the serializer omits it.
    completion_tokens_details: ?CompletionDetails = null,
};

/// The OpenAI breakdown of the completion tokens.
pub const CompletionDetails = struct {
    reasoning_tokens: u32,
};

pub const RespMessage = struct {
    role: []const u8 = "assistant",
    content: []const u8,
    reasoning_content: ?[]const u8 = null,
};

pub const Choice = struct {
    index: u32 = 0,
    message: RespMessage,
    finish_reason: []const u8,
};

pub const ChatResponse = struct {
    id: []const u8,
    object: []const u8 = "chat.completion",
    created: i64 = 0,
    model: []const u8,
    choices: []const Choice,
    usage: Usage,
};

/// A streamed delta. With `emit_null_optional_fields = false`, the null fields vanish.
pub const Delta = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
    reasoning_content: ?[]const u8 = null,
};

pub const ChunkChoice = struct {
    index: u32 = 0,
    delta: Delta,
    finish_reason: ?[]const u8 = null,
};

pub const ChatChunk = struct {
    id: []const u8,
    object: []const u8 = "chat.completion.chunk",
    created: i64 = 0,
    model: []const u8,
    choices: []const ChunkChoice,
};

pub const ModelEntry = struct {
    id: []const u8,
    object: []const u8 = "model",
    created: i64 = 0,
    owned_by: []const u8 = "zigferencer",
};

pub const ModelsList = struct {
    object: []const u8 = "list",
    data: []const ModelEntry,
};

/// The optional fields follow the llama.cpp server. Only a context overflow sets them.
pub const ErrorBody = struct {
    code: ?u16 = null,
    message: []const u8,
    type: []const u8,
    n_prompt_tokens: ?u32 = null,
    n_ctx: ?u32 = null,
};

/// The OpenAI error envelope: `{"error": {...}}`. The quoted identifier spells a keyword field.
pub const ErrorObject = struct {
    @"error": ErrorBody,
};
