//! OpenAI-compatible request and response values. They are plain structs
//! across the HTTP and JSON boundary, because a boundary carries values.
//! `std.json` parses `ChatRequest` from a request body, and
//! `std.json.Stringify` serializes the rest back.
//!
//! Every optional field has a default, so a minimal request parses. The parse
//! also drops unknown fields (`ParseOptions.ignore_unknown_fields`), so a full
//! OpenAI client with tools and stop strings still round-trips. The map layer
//! logs each unknown field name once per process. Every sampling field is a
//! named field here, so no sampling value drops in silence.

// -- types ------------------------------------------------------------------

/// One message as the client sends it. `role` is a free string here. The map
/// layer narrows it to a `chat.Role`.
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8 = "",
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
    chat_template_kwargs: ?ChatTemplateKwargs = null,
};

/// The template switches of the vLLM and llama.cpp servers. Only
/// `enable_thinking` has an effect here. Null keeps the server's mode.
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

/// The serializer omits a null `reasoning_content`, so a reply with no think
/// block keeps the OpenAI shape.
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

/// A streamed delta. With `emit_null_optional_fields = false`, the null fields
/// vanish. The first chunk carries `{role}`. A think block streams as
/// `{reasoning_content}` chunks and the reply as `{content}` chunks. The final
/// chunk carries `{}` plus a `finish_reason`.
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

/// The optional fields follow the llama.cpp server. Only a context overflow
/// sets them. The serializer omits a null field, so every other error body
/// keeps the OpenAI shape.
pub const ErrorBody = struct {
    code: ?u16 = null,
    message: []const u8,
    type: []const u8,
    n_prompt_tokens: ?u32 = null,
    n_ctx: ?u32 = null,
};

/// The OpenAI error envelope: `{"error": {...}}`. `@"error"` is the field name
/// on the wire. The quoted identifier is only how Zig spells a keyword field.
pub const ErrorObject = struct {
    @"error": ErrorBody,
};
