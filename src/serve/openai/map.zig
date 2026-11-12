//! The datapath of the OpenAI frontend. It parses and validates an untrusted request body
//! into a value, and it maps a rejection to an HTTP error. The HTTP shell (`http_server.zig`) calls
//! them and owns the transport.
//!
//! Comptime. The refusal messages and the type probe walk the fields of `ChatRequest`, so a
//! new field needs no table edit.
//!
//! Trust boundary. Every guard turns malformed input into a typed `ReqError` that the shell
//! renders as a 4xx, never into an assert.
//!
//! Parse. `parseChatRequest` rejects the input the server cannot answer correctly, such as
//! `n > 1`, a sampling field out of its range, or a frequency penalty. It ignores the other
//! fields it does not support (tools, logprobs), so a generic OpenAI client works. It logs each of
//! the first `UnknownFields.table_cap` ignored names once per process. The `model` string passes
//! through for the shell to check. A known field with a value of the wrong type is
//! `error.BadField`, and `diag` names it. Each sampling guard reads as the range it accepts, so a
//! NaN fails it. The ranges are the binary wire guards, so the scheduler accepts every value that
//! passes.
//!
//! Thinking. A budget of 0 turns thinking off, the vLLM rule, so it wins over `enable_thinking`.
//! A model with no think tags accepts only the closed modes, the rule of the binary wire. A
//! budget above 0 also needs the ids of a forced close, or the scheduler refuses it.
//!
//! Reasoning split. `ReasoningSplit` routes the token events of one turn to `reasoning_content`
//! and `content`. Each think tag is one token, so a two-state machine over the token ids splits
//! the stream, and a tag never straddles two events. It drops the tags and the newlines that lead
//! the reply after a close. A close outside the block is plain reply text, so a turn that opens
//! closed and never sees `<think>` keeps every byte as content.
//!
//! Content. A message `content` is a string, an array of text parts, or null on an assistant
//! message. The parse joins the texts of the parts in order with no separator, so the joined text
//! renders as one plain range, as a string does. A part of another type gets a 400 that names it.
//!
//! Stop. `StopMatcher` cuts the reply text before the first stop string. It holds back the longest
//! tail that can still start a stop string, at most `limits.openai_max_stop_bytes - 1` bytes, so no
//! emitted byte belongs to a later match.
//!
//! UTF-8. A streamed token can be a partial multi-byte sequence (byte-fallback tokens, emoji). A
//! JSON string must be valid UTF-8, so the shell emits the prefix that `completeUtf8Len` returns
//! and carries the remainder of at most three bytes into the next chunk.

const std = @import("std");
const Allocator = std.mem.Allocator;
const chat = @import("../../core/chat.zig");
const limits = @import("../../core/limits.zig");
const types = @import("types.zig");
const protocol = @import("../protocol.zig");
const prop = @import("../../core/utils/prop.zig");

const log = std.log.scoped(.serve_openai);

// -- constants --------------------------------------------------------------

/// The refusal message of each known field with a value of the wrong type.
const bad_field_messages = blk: {
    const fields = @typeInfo(types.ChatRequest).@"struct".fields;
    var messages: [fields.len][]const u8 = undefined;
    for (fields, &messages) |f, *m| {
        m.* = "'" ++ f.name ++ "' has a value of the wrong JSON type";
    }

    break :blk messages;
};

const thinking_unsupported_message = "enable_thinking true does not apply to this model, " ++
    "because its chat template has no thinking mode";
const budget_unsupported_message = "thinking_token_budget above 0 does not apply to this " ++
    "model, because its chat template has no thinking mode";
const budget_no_forced_close_message =
    "thinking_token_budget above 0 does not apply to this model, " ++
    "because the server has no forced close for its think block";

// -- types ------------------------------------------------------------------

pub const ReqError = error{
    BadJson,
    /// A known field holds a JSON value of the wrong type. `Diag.field` names it.
    BadField,
    NoMessages,
    TooManyMessages,
    NTooLarge,
    BadTemperature,
    BadTopP,
    BadTopK,
    BadMinP,
    BadPresencePenalty,
    BadRepetitionPenalty,
    FrequencyUnsupported,
    BadThinkingBudget,
    ThinkingUnsupported,
    BudgetUnsupported,
    BudgetNoForcedClose,
    /// `stop` is not a string or an array of short strings.
    BadStop,
    /// A message holds more content parts than `limits.openai_max_content_parts`.
    TooManyParts,
    /// A content part is not text. `Diag.detail` names its type.
    ContentPartUnsupported,
    OutOfMemory,
};

/// A known top-level field of a chat request.
pub const ChatField = std.meta.FieldEnum(types.ChatRequest);

/// The detail of a refusal that the error tag cannot carry.
pub const Diag = struct {
    /// Set with `error.BadField`, null otherwise.
    field: ?ChatField = null,
    /// The arena-owned message of `error.ContentPartUnsupported`, null otherwise.
    detail: ?[]const u8 = null,
};

/// The set of unknown top-level field names the server has reported. A name
/// longer than `name_cap` bytes compares by its first `name_cap` bytes.
pub const UnknownFields = struct {
    pub const table_cap = 32;
    pub const name_cap = 64;

    pub const Verdict = enum { seen, new, full };

    lock: std.atomic.Mutex = .unlocked,
    count: u32 = 0,
    full_reported: bool = false,
    lens: [table_cap]u8 = undefined,
    names: [table_cap][name_cap]u8 = undefined,

    /// Record `name`. `new` tells the caller to log it once.
    /// `full` is the first name past the table, so the caller logs that the
    /// table is full. `seen` needs no log.
    pub fn note(self: *UnknownFields, name: []const u8) Verdict {
        const key = capped(name);
        while (!self.lock.tryLock()) {
            std.atomic.spinLoopHint();
        }

        defer self.lock.unlock();

        for (self.names[0..self.count], self.lens[0..self.count]) |*have, len| {
            if (std.mem.eql(u8, have[0..len], key)) return .seen;
        }

        if (self.count == table_cap) {
            if (self.full_reported) return .seen;
            self.full_reported = true;
            return .full;
        }

        @memcpy(self.names[self.count][0..key.len], key);
        self.lens[self.count] = @intCast(key.len);
        self.count += 1;
        std.debug.assert(self.count <= table_cap);
        return .new;
    }

    /// The part of `name` that the table compares and the log prints.
    pub fn capped(name: []const u8) []const u8 {
        return name[0..@min(name.len, name_cap)];
    }
};

/// The source of an omitted sampling field. `model` takes the card row of
/// the resolved thinking mode, the vLLM priority. `greedy` keeps the greedy
/// identity for every omitted field.
pub const SamplingDefaults = enum {
    model,
    greedy,

    pub fn parse(raw: []const u8) ?SamplingDefaults {
        return std.meta.stringToEnum(SamplingDefaults, raw);
    }
};

/// The default sampling row of each thinking mode. A model with no card has
/// the greedy row in both modes.
pub const DefaultRows = struct {
    think: protocol.Sampling,
    plain: protocol.Sampling,

    pub const greedy: DefaultRows = .{
        .think = protocol.Sampling.greedy,
        .plain = protocol.Sampling.greedy,
    };

    pub fn init(
        mode: SamplingDefaults,
        card_think: protocol.Sampling,
        card_plain: protocol.Sampling,
    ) DefaultRows {
        // The engine has no frequency penalty, so a card row never carries one.
        std.debug.assert(card_think.frequency_penalty == 0);
        std.debug.assert(card_plain.frequency_penalty == 0);

        return switch (mode) {
            .model => .{ .think = card_think, .plain = card_plain },
            .greedy => greedy,
        };
    }

    pub fn row(self: DefaultRows, think_open: bool) protocol.Sampling {
        return if (think_open) self.think else self.plain;
    }
};

/// The sampling fields as the request sent them. Null is an omitted field,
/// which takes the default row of the resolved thinking mode.
pub const SamplingFields = struct {
    temperature: ?f32 = null,
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    min_p: ?f32 = null,
    presence_penalty: ?f32 = null,
    repetition_penalty: ?f32 = null,
    seed: ?u64 = null,
};

/// `thinking_token_budget` in the vLLM shape. `server` is an absent field or
/// -1. `off` is 0, which closes the block before the reply. `tokens` forces
/// the close after that count of think tokens.
pub const ThinkBudget = union(enum) {
    server,
    off,
    tokens: u32,
};

/// The thinking of one turn, resolved against the model.
pub const TurnMode = struct {
    kind: chat.Kind,
    /// Null takes the server budget.
    think_budget: ?u32,
};

/// A validated chat request: arena-owned `chat.Message` values, the sampling fields, the budgets.
pub const Request = struct {
    /// The `model` string as sent, empty when omitted. The shell answers 404 on a mismatch.
    model: []const u8,
    messages: []const chat.Message,
    sampling: SamplingFields,
    max_tokens: u32,
    stream: bool,
    /// `enable_thinking`: true is `on`, false is `off`, absent or null is `auto`.
    thinking: chat.ThinkingFlag,
    think_budget: ThinkBudget,
    /// At most `limits.openai_max_stops` strings, each 1 to `limits.openai_max_stop_bytes` long.
    stop: []const []const u8,
};

pub const HttpError = struct {
    status: u16,
    type: []const u8,
    message: []const u8,
    overflow: ?ContextOverflow = null,
};

/// The numbers of a prompt that does not fit the session window.
pub const ContextOverflow = struct { n_prompt_tokens: u32, n_ctx: u32 };

/// The reply field that a piece of token text goes to.
pub const Field = enum { reasoning, content };

pub const Routed = struct { field: Field, text: []const u8 };

/// Routes the token events of one turn to `reasoning_content` and `content`.
pub const ReasoningSplit = struct {
    tags: ?chat.ThinkIds,
    field: Field,
    /// A close sets it. The first piece with a byte other than a newline clears it.
    trim: bool = false,

    /// `open` is true when the turn opens inside the think block. Only a
    /// model with think tags can open one.
    pub fn init(tags: ?chat.ThinkIds, open: bool) ReasoningSplit {
        if (open) std.debug.assert(tags != null);
        if (tags) |t| std.debug.assert(t.open != t.close);
        return .{ .tags = tags, .field = if (open) .reasoning else .content };
    }

    /// The field and the text of one token event. Null drops the event: a
    /// tag, or a piece of newlines alone right after a close.
    pub fn route(self: *ReasoningSplit, id: u32, text: []const u8) ?Routed {
        if (self.trim) std.debug.assert(self.field == .content);

        const tags = self.tags orelse return .{ .field = .content, .text = text };
        if (id == tags.open) {
            self.field = .reasoning;
            self.trim = false;
            return null;
        }

        if (id == tags.close and self.field == .reasoning) {
            self.field = .content;
            self.trim = true;
            return null;
        }

        if (!self.trim) return .{ .field = self.field, .text = text };

        const reply = std.mem.trimStart(u8, text, "\n");
        if (reply.len == 0) return null;

        self.trim = false;
        return .{ .field = .content, .text = reply };
    }
};

/// Cuts the content text of one turn before the first stop string. Feed each content piece in
/// order, and emit what `feed` returns. At the natural end of the turn, emit what `finish`
/// returns. The returned text borrows the matcher until the next call.
pub const StopMatcher = struct {
    const hold_max = limits.openai_max_stop_bytes - 1;

    stops: []const []const u8,
    /// `buf[start..len]` holds the bytes held back, and `buf[0..start]` the text of the last call.
    buf: [hold_max + limits.max_token_text_bytes]u8 = undefined,
    start: usize = 0,
    len: usize = 0,
    done: bool = false,

    pub const Out = struct {
        text: []const u8,
        /// The text ends just before a stop string, so the turn must stop.
        stopped: bool,
    };

    pub fn init(stops: []const []const u8) StopMatcher {
        std.debug.assert(stops.len <= limits.openai_max_stops);
        for (stops) |stop| {
            std.debug.assert(stop.len >= 1 and stop.len <= limits.openai_max_stop_bytes);
        }

        return .{ .stops = stops };
    }

    pub fn feed(self: *StopMatcher, piece: []const u8) Out {
        std.debug.assert(!self.done);
        std.debug.assert(piece.len <= limits.max_token_text_bytes);

        // Move the held bytes to the front, and append the piece.
        const held = self.len - self.start;
        std.debug.assert(held <= hold_max);
        std.mem.copyForwards(u8, self.buf[0..held], self.buf[self.start..self.len]);
        @memcpy(self.buf[held..][0..piece.len], piece);
        self.len = held + piece.len;
        const text = self.buf[0..self.len];

        // The earliest start of any stop string wins.
        var cut: ?usize = null;
        for (self.stops) |stop| {
            const at = std.mem.find(u8, text, stop) orelse continue;
            if (cut == null or at < cut.?) {
                cut = at;
            }
        }

        if (cut) |at| {
            self.done = true;
            self.start = 0;
            self.len = 0;
            return .{ .text = text[0..at], .stopped = true };
        }

        // Hold back the longest tail that is a proper prefix of a stop string.
        var keep: usize = 0;
        for (self.stops) |stop| {
            var k = @min(stop.len - 1, text.len);
            while (k > keep) : (k -= 1) {
                if (std.mem.endsWith(u8, text, stop[0..k])) {
                    keep = k;
                    break;
                }
            }
        }

        std.debug.assert(keep <= hold_max);
        self.start = text.len - keep;
        return .{ .text = text[0..self.start], .stopped = false };
    }

    /// The held bytes at the natural end of the turn. No stop string can complete them.
    pub fn finish(self: *StopMatcher) []const u8 {
        std.debug.assert(!self.done);
        self.done = true;
        return self.buf[self.start..self.len];
    }
};

// -- state ------------------------------------------------------------------

var unknown_fields: UnknownFields = .{};

// -- api --------------------------------------------------------------------

/// The budget of a request that names none. A flag of 0 is the whole window.
pub fn resolveDefaultMaxTokens(flag: u32, max_seq: u32) u32 {
    std.debug.assert(max_seq >= 1);

    const resolved = if (flag == 0) max_seq else @min(flag, max_seq);
    std.debug.assert(resolved >= 1);
    std.debug.assert(resolved <= max_seq);
    return resolved;
}

/// The llama.cpp answer to a prompt longer than the window. A client can read
/// both numbers and shorten the conversation.
pub fn exceedContextError(
    arena: Allocator,
    n_prompt_tokens: u32,
    n_ctx: u32,
) error{OutOfMemory}!HttpError {
    const message = try std.fmt.allocPrint(
        arena,
        "the request of {d} tokens exceeds the context of {d} tokens",
        .{ n_prompt_tokens, n_ctx },
    );
    return .{
        .status = 400,
        .type = "exceed_context_size_error",
        .message = message,
        .overflow = .{ .n_prompt_tokens = n_prompt_tokens, .n_ctx = n_ctx },
    };
}

/// The wire value of an HTTP error.
pub fn errorObject(e: HttpError) types.ErrorObject {
    const overflow = e.overflow orelse return .{
        .@"error" = .{ .message = e.message, .type = e.type },
    };
    return .{ .@"error" = .{
        .code = e.status,
        .message = e.message,
        .type = e.type,
        .n_prompt_tokens = overflow.n_prompt_tokens,
        .n_ctx = overflow.n_ctx,
    } };
}

/// Map a rejection to its OpenAI HTTP error. A malformed request is a client error (4xx), and an
/// allocation failure is a 500. The shell raises the `internal` 5xx on an engine fault.
/// `diag` is the detail that `parseChatRequest` gives with the error.
pub fn reqErrorToHttp(e: ReqError, diag: Diag) HttpError {
    // Check that the diagnostic matches the error.
    if (e == error.BadField) std.debug.assert(diag.field != null);
    if (e != error.BadField) std.debug.assert(diag.field == null);
    if (e == error.ContentPartUnsupported) std.debug.assert(diag.detail != null);
    if (e != error.ContentPartUnsupported) std.debug.assert(diag.detail == null);

    // Pick the message of the error.
    const message: []const u8 = switch (e) {
        error.BadJson => "malformed JSON body",
        error.BadField => bad_field_messages[@intFromEnum(diag.field.?)],
        error.NoMessages => "'messages' is required and must be non-empty",
        error.TooManyMessages => "too many messages",
        error.NTooLarge => "only n=1 is supported",
        error.BadTemperature => "temperature must be a finite number of 0 or more",
        error.BadTopP => "top_p must be more than 0 and at most 1",
        error.BadTopK => std.fmt.comptimePrint(
            "top_k must be -1 or in the range 0 to {d}",
            .{limits.max_top_k},
        ),
        error.BadMinP => "min_p must be in the range 0 to 1",
        error.BadPresencePenalty => std.fmt.comptimePrint(
            "presence_penalty must be in the range -{d} to {d}",
            .{ limits.max_presence_penalty, limits.max_presence_penalty },
        ),
        error.BadRepetitionPenalty => std.fmt.comptimePrint(
            "repetition_penalty must be more than 0 and at most {d}",
            .{limits.max_repetition_penalty},
        ),
        error.FrequencyUnsupported => protocol.frequency_unsupported_detail ++
            ". The only accepted value is 0",
        error.BadThinkingBudget => std.fmt.comptimePrint(
            "thinking_token_budget must be -1, 0, or a count from 1 to {d}",
            .{limits.max_model_ctx},
        ),
        error.ThinkingUnsupported => thinking_unsupported_message,
        error.BudgetUnsupported => budget_unsupported_message,
        error.BudgetNoForcedClose => budget_no_forced_close_message,
        error.BadStop => std.fmt.comptimePrint(
            "stop must be a string or an array of at most {d} strings, each of 1 to {d} bytes",
            .{ limits.openai_max_stops, limits.openai_max_stop_bytes },
        ),
        error.TooManyParts => std.fmt.comptimePrint(
            "a message content holds more than {d} parts",
            .{limits.openai_max_content_parts},
        ),
        error.ContentPartUnsupported => diag.detail.?,
        error.OutOfMemory => return .{
            .status = 500,
            .type = "server_error",
            .message = "out of memory",
        },
    };

    // Return the client error.
    return .{ .status = 400, .type = "invalid_request_error", .message = message };
}

/// Resolve the thinking of a turn against the model.
pub fn resolveTurn(
    server_kind: chat.Kind,
    flag: chat.ThinkingFlag,
    budget: ThinkBudget,
    forced_close: bool,
) error{ ThinkingUnsupported, BudgetUnsupported, BudgetNoForcedClose }!TurnMode {
    const tagged = chat.thinkText(server_kind) != null;
    const effective: chat.ThinkingFlag = if (budget == .off) .off else flag;
    const kind = try chat.resolveThinking(server_kind, effective);
    const think_budget: ?u32 = switch (budget) {
        .server, .off => null,
        .tokens => |n| blk: {
            if (!tagged) return error.BudgetUnsupported;
            if (!forced_close) return error.BudgetNoForcedClose;
            break :blk n;
        },
    };

    if (budget == .off) std.debug.assert(!chat.thinkOpen(kind));
    if (think_budget) |n| std.debug.assert(n >= 1 and n <= limits.max_model_ctx);

    return .{ .kind = kind, .think_budget = think_budget };
}

/// The sampling of one request: each field the request sent, else the field
/// of the default row. `seed` is the request seed or a fresh draw.
pub fn resolveSampling(
    fields: SamplingFields,
    row: protocol.Sampling,
    seed: u64,
) protocol.Sampling {
    std.debug.assert(row.frequency_penalty == 0);

    if (fields.seed) |s| std.debug.assert(s == seed);

    const out: protocol.Sampling = .{
        .temperature = fields.temperature orelse row.temperature,
        .top_k = fields.top_k orelse row.top_k,
        .top_p = fields.top_p orelse row.top_p,
        .seed = seed,
        .presence_penalty = fields.presence_penalty orelse row.presence_penalty,
        .repetition_penalty = fields.repetition_penalty orelse row.repetition_penalty,
        .min_p = fields.min_p orelse row.min_p,
    };
    // The binary wire guards hold, so no request value reaches a sampler assert.
    std.debug.assert(out.temperature >= 0);
    std.debug.assert(out.top_p > 0 and out.top_p <= 1);
    std.debug.assert(out.top_k <= limits.max_top_k);
    std.debug.assert(out.min_p >= 0 and out.min_p <= 1);
    std.debug.assert(
        out.repetition_penalty > 0 and out.repetition_penalty <= limits.max_repetition_penalty,
    );
    std.debug.assert(out.frequency_penalty == 0);
    return out;
}

/// Parse and validate `body` into a `Request`. `default_max_tokens` covers a request that omits
/// a budget. `max_seq` is the session's hard ceiling.
pub fn parseChatRequest(
    arena: Allocator,
    body: []const u8,
    default_max_tokens: u32,
    max_seq: u32,
    diag: *Diag,
) ReqError!Request {
    return parseChatRequestWith(&unknown_fields, arena, body, default_max_tokens, max_seq, diag);
}

/// Length of the longest prefix of `bytes` that ends on a UTF-8 codepoint boundary.
pub fn completeUtf8Len(bytes: []const u8) usize {
    if (bytes.len == 0) return 0;

    const max_back = @min(bytes.len, 4);
    var back: usize = 1;
    while (back <= max_back) : (back += 1) {
        const b = bytes[bytes.len - back];
        if (b & 0xC0 == 0x80) continue; // a continuation byte, so keep walking left

        const seq_len = std.unicode.utf8ByteSequenceLength(
            b,
        ) catch return bytes.len; // invalid lead -> emit as-is
        return if (back == seq_len) bytes.len else bytes.len - back;
    }

    return bytes.len; // no lead byte in the last 4 (malformed) -> emit as-is
}

// -- helpers ----------------------------------------------------------------

fn reportUnknown(seen: *UnknownFields, name: []const u8) void {
    switch (seen.note(name)) {
        .seen => {},
        .new => log.info(
            "request field \"{f}\" is not supported, so the server ignores it",
            .{unknownName(name)},
        ),
        .full => log.info(
            "the table of ignored request field names is full, " ++
                "so later unknown names are not logged",
            .{},
        ),
    }
}

/// The client chooses the name, so the log prints its capped part with every
/// control byte and quote escaped.
fn unknownName(name: []const u8) std.fmt.Alt([]const u8, std.zig.stringEscape) {
    return std.zig.fmtString(UnknownFields.capped(name));
}

fn parseChatRequestWith(
    seen: *UnknownFields,
    arena: Allocator,
    body: []const u8,
    default_max_tokens: u32,
    max_seq: u32,
    diag: *Diag,
) ReqError!Request {
    diag.* = .{};

    // Parse the JSON into a tree.
    const tree = std.json.parseFromSliceLeaky(
        std.json.Value,
        arena,
        body,
        .{},
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.BadJson,
    };
    if (tree != .object) return error.BadJson;

    // Map the tree onto the request struct. A wrong type names its field.
    const req = std.json.parseFromValueLeaky(
        types.ChatRequest,
        arena,
        tree,
        json_options,
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            diag.field = try wrongTypedField(arena, tree.object) orelse return error.BadJson;
            return error.BadField;
        },
    };

    // Report each unknown field name once.
    for (tree.object.keys()) |key| {
        const known = inline for (@typeInfo(types.ChatRequest).@"struct".fields) |f| {
            if (std.mem.eql(u8, key, f.name)) break true;
        } else false;

        if (!known) {
            reportUnknown(seen, key);
        }
    }

    // Check the messages and `n`.
    if (req.messages.len == 0) return error.NoMessages;
    if (req.messages.len > limits.openai_max_messages) return error.TooManyMessages;

    if (req.n) |n| {
        if (n > 1) return error.NTooLarge;
    }

    const msgs = try arena.alloc(chat.Message, req.messages.len);
    for (req.messages, msgs) |src, *dst| {
        const role = parseRole(src.role);
        dst.* = .{ .role = role, .content = try messageText(arena, role, src.content, diag) };
    }

    const stop = try parseStop(arena, req.stop);

    // Resolve the sampling, the budgets, and the thinking switch.
    const sampling = try parseSampling(req);
    const think_budget = try parseThinkBudget(req.thinking_token_budget);

    const want = req.max_completion_tokens orelse req.max_tokens orelse default_max_tokens;
    const max_tokens = if (want == 0) default_max_tokens else @min(want, max_seq);
    std.debug.assert(max_tokens >= 1);

    const kwargs = req.chat_template_kwargs orelse types.ChatTemplateKwargs{};
    const thinking: chat.ThinkingFlag = if (kwargs.enable_thinking) |on|
        (if (on) .on else .off)
    else
        .auto;

    return .{
        .model = req.model,
        .messages = msgs,
        .sampling = sampling,
        .max_tokens = max_tokens,
        .stream = req.stream,
        .thinking = thinking,
        .think_budget = think_budget,
        .stop = stop,
    };
}

const json_options: std.json.ParseOptions = .{ .ignore_unknown_fields = true };

/// The first known field whose value does not parse as its own type, or null
/// when each one parses. The whole-request parse fails without a field name,
/// so this probe parses each known field alone to find the one to name.
fn wrongTypedField(arena: Allocator, object: std.json.ObjectMap) error{OutOfMemory}!?ChatField {
    inline for (@typeInfo(types.ChatRequest).@"struct".fields) |f| {
        if (object.get(f.name)) |value| {
            _ = std.json.parseFromValueLeaky(
                f.type,
                arena,
                value,
                json_options,
            ) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return @field(ChatField, f.name),
            };
        }
    }

    return null;
}

/// Check each sampling guard.
fn parseSampling(req: types.ChatRequest) ReqError!SamplingFields {
    var out: SamplingFields = .{ .seed = req.seed };
    if (req.temperature) |t| {
        if (!(std.math.isFinite(t) and t >= 0)) return error.BadTemperature;
        out.temperature = t;
    }

    if (req.top_p) |p| {
        if (!(p > 0 and p <= 1)) return error.BadTopP;
        out.top_p = p;
    }

    if (req.top_k) |k| {
        // -1 is the vLLM value for no top_k cut, the same as 0 here.
        if (!(k >= -1 and k <= limits.max_top_k)) return error.BadTopK;
        out.top_k = if (k == -1) 0 else @intCast(k);
    }

    if (req.min_p) |m| {
        if (!(m >= 0 and m <= 1)) return error.BadMinP;
        out.min_p = m;
    }

    if (req.presence_penalty) |pp| {
        if (!(@abs(pp) <= limits.max_presence_penalty)) return error.BadPresencePenalty;
        out.presence_penalty = pp;
    }

    if (req.repetition_penalty) |r| {
        if (!(r > 0 and r <= limits.max_repetition_penalty)) return error.BadRepetitionPenalty;
        out.repetition_penalty = r;
    }

    // The engine has no frequency penalty, so the refusal says the server did not apply it.
    if (req.frequency_penalty) |f| {
        if (f != 0) return error.FrequencyUnsupported;
    }

    return out;
}

fn parseThinkBudget(raw: ?i64) ReqError!ThinkBudget {
    const n = raw orelse return .server;
    if (n == -1) return .server;
    if (n == 0) return .off;
    if (!(n >= 1 and n <= limits.max_model_ctx)) return error.BadThinkingBudget;

    return .{ .tokens = @intCast(n) };
}

/// The text of one message `content`: a string, the joined texts of an array of text parts, or
/// the empty text of a null on an assistant message.
fn messageText(
    arena: Allocator,
    role: chat.Role,
    content: std.json.Value,
    diag: *Diag,
) ReqError![]const u8 {
    switch (content) {
        .string => |text| return text,
        .null => if (role == .assistant) return "",
        .array => |parts| {
            if (parts.items.len > limits.openai_max_content_parts) return error.TooManyParts;

            var text: std.ArrayList(u8) = .empty;
            for (parts.items) |part| {
                const fields = switch (part) {
                    .object => |o| o,
                    else => return badMessages(diag),
                };
                const kind = switch (fields.get("type") orelse return badMessages(diag)) {
                    .string => |k| k,
                    else => return badMessages(diag),
                };
                if (!std.mem.eql(u8, kind, "text")) {
                    diag.detail = try std.fmt.allocPrint(
                        arena,
                        "content part type \"{f}\" is not supported, because this model " ++
                            "takes text only",
                        .{unknownName(kind)},
                    );
                    return error.ContentPartUnsupported;
                }

                switch (fields.get("text") orelse return badMessages(diag)) {
                    .string => |t| try text.appendSlice(arena, t),
                    else => return badMessages(diag),
                }
            }

            return text.items;
        },
        else => {},
    }

    return badMessages(diag);
}

fn badMessages(diag: *Diag) ReqError {
    diag.field = .messages;
    return error.BadField;
}

/// The stop strings of `value`: none for null, one for a string, or each string of an array.
fn parseStop(arena: Allocator, value: std.json.Value) ReqError![]const []const u8 {
    const items: []const std.json.Value = switch (value) {
        .null => return &.{},
        .string => (try arena.dupe(std.json.Value, &.{value})),
        .array => |a| a.items,
        else => return error.BadStop,
    };
    if (items.len > limits.openai_max_stops) return error.BadStop;

    const stops = try arena.alloc([]const u8, items.len);
    for (items, stops) |item, *dst| {
        const text = switch (item) {
            .string => |t| t,
            else => return error.BadStop,
        };
        if (text.len == 0 or text.len > limits.openai_max_stop_bytes) return error.BadStop;
        dst.* = text;
    }

    return stops;
}

fn parseRole(s: []const u8) chat.Role {
    if (std.mem.eql(u8, s, "system")) return .system;
    if (std.mem.eql(u8, s, "assistant")) return .assistant;
    return .user; // "user" and any other or unknown role (for example "tool")
}

// -- tests ------------------------------------------------------------------

const ParseWant = struct {
    err: ?ReqError = null,
    /// The field that a `BadField` refusal names.
    field: ?ChatField = null,
    model: ?[]const u8 = null,
    messages: ?usize = null,
    first_role: ?chat.Role = null,
    first_content: ?[]const u8 = null,
    stream: ?bool = null,
    max_tokens: ?u32 = null,
    sampling: ?SamplingFields = null,
    think_budget: ?ThinkBudget = null,
    thinking: ?chat.ThinkingFlag = null,
    stops: ?usize = null,
};

const ParseCase = struct {
    body: []const u8,
    default_max_tokens: u32 = 64,
    window: u32 = 512,
    want: ParseWant = .{},
};

/// A body with one user message and the `extra` fields.
fn withExtra(comptime extra: []const u8) []const u8 {
    return "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]" ++ extra ++ "}";
}

/// A case whose body holds `extra`, and whose parse must fail with `err`.
fn refuse(comptime extra: []const u8, err: ReqError) ParseCase {
    return .{ .body = withExtra(extra), .want = .{ .err = err } };
}

/// A case whose body holds `extra`, and whose parse must refuse the type of `field`.
fn refuseType(comptime extra: []const u8, field: ChatField) ParseCase {
    return .{ .body = withExtra(extra), .want = .{ .err = error.BadField, .field = field } };
}

/// A case whose body holds `extra`, and whose think budget must read `budget`.
fn budgetCase(comptime extra: []const u8, budget: ThinkBudget) ParseCase {
    return .{ .body = withExtra(extra), .want = .{ .think_budget = budget } };
}

/// A case whose body holds `extra`, and whose thinking switch must read `flag`.
fn thinkingCase(comptime extra: []const u8, flag: chat.ThinkingFlag) ParseCase {
    return .{ .body = withExtra(extra), .want = .{ .thinking = flag } };
}

/// A request field maps to its value, or its refusal. A field that the
/// case does not name stays unchecked.
fn checkParse(c: ParseCase) !void {
    // Parse the body.
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var diag: Diag = .{};
    const got = parseChatRequest(
        arena_state.allocator(),
        c.body,
        c.default_max_tokens,
        c.window,
        &diag,
    );

    // Check the error, or each field that the case names.
    try std.testing.expectEqual(c.want.field, diag.field);
    if (c.want.err) |err| return std.testing.expectError(err, got);

    const r = try got;
    const w = c.want;
    if (w.model) |v| {
        try std.testing.expectEqualStrings(v, r.model);
    }

    if (w.messages) |v| {
        try std.testing.expectEqual(v, r.messages.len);
    }

    if (w.first_role) |v| {
        try std.testing.expectEqual(v, r.messages[0].role);
    }

    if (w.first_content) |v| {
        try std.testing.expectEqualStrings(v, r.messages[0].content);
    }

    if (w.stream) |v| {
        try std.testing.expectEqual(v, r.stream);
    }

    if (w.max_tokens) |v| {
        try std.testing.expectEqual(v, r.max_tokens);
    }

    if (w.sampling) |v| {
        try std.testing.expectEqual(v, r.sampling);
    }

    if (w.think_budget) |v| {
        try std.testing.expectEqual(v, r.think_budget);
    }

    if (w.thinking) |v| {
        try std.testing.expectEqual(v, r.thinking);
    }

    if (w.stops) |v| {
        try std.testing.expectEqual(v, r.stop.len);
    }
}

fn budgetField(comptime n: u32) []const u8 {
    return std.fmt.comptimePrint(",\"thinking_token_budget\":{d}", .{n});
}

test "parseChatRequest: unknown fields parse and each name is recorded once" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var seen: UnknownFields = .{};
    const body = withExtra(",\"reasoning_effort\":\"low\",\"tools\":[]");
    for (0..2) |_| {
        var diag: Diag = .{};
        const req = try parseChatRequestWith(&seen, arena_state.allocator(), body, 64, 512, &diag);
        try std.testing.expectEqual(@as(usize, 1), req.messages.len);
    }

    try std.testing.expectEqual(@as(u32, 2), seen.count);
    try std.testing.expectEqual(UnknownFields.Verdict.seen, seen.note("reasoning_effort"));
    try std.testing.expectEqual(UnknownFields.Verdict.seen, seen.note("tools"));
    try std.testing.expectEqual(@as(u32, 2), seen.count);
}

test "UnknownFields: a full table reports once, then stays quiet" {
    var seen: UnknownFields = .{};
    var buf: [8]u8 = undefined;
    for (0..UnknownFields.table_cap) |i| {
        const name = std.fmt.bufPrint(&buf, "f{d}", .{i}) catch unreachable;
        try std.testing.expectEqual(UnknownFields.Verdict.new, seen.note(name));
    }

    try std.testing.expectEqual(UnknownFields.Verdict.full, seen.note("extra"));
    try std.testing.expectEqual(UnknownFields.Verdict.seen, seen.note("extra2"));
    try std.testing.expectEqual(UnknownFields.Verdict.seen, seen.note("f0"));
}

test "unknownName: the log prints the capped name with no raw control byte" {
    var long: [3 * UnknownFields.name_cap]u8 = undefined;
    for (&long, 0..) |*b, i| {
        b.* = @intCast(i % 256);
    }

    const names = [_][]const u8{ "tools", "a\"b\\c", "line\nbreak\x1b[31m", "\xff\x00", &long };
    var buf: [4 * UnknownFields.name_cap]u8 = undefined;
    for (names) |name| {
        const shown = try std.fmt.bufPrint(&buf, "{f}", .{unknownName(name)});
        for (shown) |b| {
            try std.testing.expect(b >= 0x20 and b < 0x7f);
        }

        try std.testing.expect(shown.len <= 4 * UnknownFields.name_cap);
    }

    try std.testing.expectEqualStrings(
        "tools",
        try std.fmt.bufPrint(&buf, "{f}", .{unknownName("tools")}),
    );
    try std.testing.expectEqualStrings(
        "line\\nbreak\\x1b[31m",
        try std.fmt.bufPrint(&buf, "{f}", .{unknownName("line\nbreak\x1b[31m")}),
    );
}

test "parseChatRequest: each field maps to its value, and a value past its edge is refused" {
    const kwargs_think = ",\"chat_template_kwargs\":{\"enable_thinking\":";
    const body_full =
        \\{"model":"m","messages":[{"role":"user","content":"hi"}],
        \\"frequency_penalty":0,"tools":[],"stop":["x"]}
    ;
    const body_two_messages =
        \\{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"}],
        \\"stream":true,"temperature":0.7,"max_tokens":99999}
    ;
    const body_no_role =
        \\{"messages":[{"content":"x"}]}
    ;
    const body_text_messages =
        \\{"messages":"hi"}
    ;
    const cases = [_]ParseCase{
        // Shape of the body.
        .{
            .body = body_full,
            .default_max_tokens = 128,
            .want = .{
                .model = "m",
                .messages = 1,
                .first_role = .user,
                .first_content = "hi",
                .sampling = .{},
                .think_budget = .server,
                .max_tokens = 128,
                .stream = false,
            },
        },
        .{
            .body = body_two_messages,
            .default_max_tokens = 128,
            .want = .{
                .messages = 2,
                .stream = true,
                .sampling = .{ .temperature = 0.7 },
                .max_tokens = 512,
            },
        },
        .{ .body = "{not json", .want = .{ .err = error.BadJson } },
        .{ .body = "{\"messages\":[]}", .want = .{ .err = error.NoMessages } },
        refuse(",\"n\":2", error.NTooLarge),
        // Sampling fields: the edges, and the first value past each.
        .{ .body = withExtra(",\"temperature\":0") },
        refuse(",\"temperature\":-1", error.BadTemperature),
        .{ .body = withExtra(",\"top_p\":1") },
        refuse(",\"top_p\":0", error.BadTopP),
        refuse(",\"top_p\":1.01", error.BadTopP),
        .{ .body = withExtra(",\"top_k\":0"), .want = .{ .sampling = .{ .top_k = 0 } } },
        .{ .body = withExtra(",\"top_k\":256") },
        refuse(",\"top_k\":257", error.BadTopK),
        .{ .body = withExtra(",\"top_k\":-1"), .want = .{ .sampling = .{ .top_k = 0 } } },
        refuse(",\"top_k\":-2", error.BadTopK),
        .{ .body = withExtra(",\"min_p\":0") },
        .{ .body = withExtra(",\"min_p\":1") },
        refuse(",\"min_p\":-0.01", error.BadMinP),
        refuse(",\"min_p\":1.5", error.BadMinP),
        .{ .body = withExtra(",\"presence_penalty\":-2") },
        .{ .body = withExtra(",\"presence_penalty\":2") },
        refuse(",\"presence_penalty\":2.5", error.BadPresencePenalty),
        refuse(",\"presence_penalty\":-3", error.BadPresencePenalty),
        .{ .body = withExtra(",\"repetition_penalty\":2") },
        .{ .body = withExtra(",\"repetition_penalty\":0.01") },
        refuse(",\"repetition_penalty\":0", error.BadRepetitionPenalty),
        refuse(",\"repetition_penalty\":2.5", error.BadRepetitionPenalty),
        refuse(",\"repetition_penalty\":-1", error.BadRepetitionPenalty),
        .{ .body = withExtra(",\"frequency_penalty\":0") },
        refuse(",\"frequency_penalty\":0.5", error.FrequencyUnsupported),
        refuse(",\"frequency_penalty\":-2", error.FrequencyUnsupported),
        refuse(",\"frequency_penalty\":7", error.FrequencyUnsupported),
        .{
            .body = withExtra(",\"temperature\":0.5,\"top_k\":40,\"top_p\":0.9,\"min_p\":0.05," ++
                "\"presence_penalty\":1.5,\"repetition_penalty\":1.1,\"seed\":7," ++
                "\"frequency_penalty\":0"),
            .want = .{ .sampling = .{
                .temperature = 0.5,
                .top_k = 40,
                .top_p = 0.9,
                .min_p = 0.05,
                .presence_penalty = 1.5,
                .repetition_penalty = 1.1,
                .seed = 7,
            } },
        },
        // The think budget: -1 and absent are the server value, 0 is off, N is N.
        budgetCase("", .server),
        budgetCase(",\"thinking_token_budget\":null", .server),
        budgetCase(",\"thinking_token_budget\":-1", .server),
        budgetCase(",\"thinking_token_budget\":0", .off),
        budgetCase(",\"thinking_token_budget\":64", .{ .tokens = 64 }),
        refuse(",\"thinking_token_budget\":-2", error.BadThinkingBudget),
        budgetCase(budgetField(limits.max_model_ctx), .{ .tokens = limits.max_model_ctx }),
        refuse(budgetField(limits.max_model_ctx + 1), error.BadThinkingBudget),
        // The budget lives at the top level only. The template switches carry no budget.
        budgetCase(",\"chat_template_kwargs\":{\"thinking_token_budget\":0}", .server),
        // The thinking switch.
        thinkingCase("", .auto),
        thinkingCase(",\"chat_template_kwargs\":null", .auto),
        thinkingCase(",\"chat_template_kwargs\":{}", .auto),
        thinkingCase(kwargs_think ++ "null}", .auto),
        thinkingCase(kwargs_think ++ "true}", .on),
        thinkingCase(kwargs_think ++ "false,\"other\":1}", .off),
        // A known field with a value of the wrong type is refused by name.
        refuseType(kwargs_think ++ "\"yes\"}", .chat_template_kwargs),
        refuseType(",\"temperature\":\"hot\"", .temperature),
        refuseType(",\"max_tokens\":4294967296", .max_tokens),
        refuseType(",\"stream\":null", .stream),
        .{ .body = body_no_role, .want = .{ .err = error.BadField, .field = .messages } },
        .{ .body = body_text_messages, .want = .{ .err = error.BadField, .field = .messages } },
        .{ .body = "[1]", .want = .{ .err = error.BadJson } },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("parse case {d}: {s}\n", .{ i, c.body });

        try checkParse(c);
    }
}

test "reqErrorToHttp: every field refusal is a 400 that names the field and its range" {
    const Case = struct { e: ReqError, want: []const u8 };
    const want_no_forced_close = "thinking_token_budget above 0 does not apply to this model, " ++
        "because the server has no forced close for its think block";
    const want_no_thinking_mode = "thinking_token_budget above 0 does not apply to this model, " ++
        "because its chat template has no thinking mode";
    const want_repetition = "repetition_penalty must be more than 0 and at most 2";
    const want_frequency = "frequency_penalty is not supported. The only accepted value is 0";
    const want_budget_range = "thinking_token_budget must be -1, 0, or a count from 1 to 262144";
    const cases = [_]Case{
        .{ .e = error.BadTemperature, .want = "temperature must be a finite number of 0 or more" },
        .{ .e = error.BadTopP, .want = "top_p must be more than 0 and at most 1" },
        .{ .e = error.BadTopK, .want = "top_k must be -1 or in the range 0 to 256" },
        .{ .e = error.BudgetNoForcedClose, .want = want_no_forced_close },
        .{ .e = error.BadMinP, .want = "min_p must be in the range 0 to 1" },
        .{ .e = error.BadPresencePenalty, .want = "presence_penalty must be in the range -2 to 2" },
        .{ .e = error.BadRepetitionPenalty, .want = want_repetition },
        .{ .e = error.FrequencyUnsupported, .want = want_frequency },
        .{ .e = error.BadThinkingBudget, .want = want_budget_range },
        .{ .e = error.BudgetUnsupported, .want = want_no_thinking_mode },
    };
    const bad_top_k = reqErrorToHttp(error.BadField, .{ .field = .top_k });
    try std.testing.expectEqual(@as(u16, 400), bad_top_k.status);
    try std.testing.expectEqualStrings(
        "'top_k' has a value of the wrong JSON type",
        bad_top_k.message,
    );
    for (cases) |c| {
        const h = reqErrorToHttp(c.e, .{});
        try std.testing.expectEqual(@as(u16, 400), h.status);
        try std.testing.expectEqualStrings("invalid_request_error", h.type);
        try std.testing.expectEqualStrings(c.want, h.message);
    }
}

const test_card_think: protocol.Sampling = .{
    .temperature = 0.6,
    .top_k = 20,
    .top_p = 0.95,
    .seed = 0,
    .presence_penalty = 1.5,
};
const test_card_plain: protocol.Sampling = .{
    .temperature = 0.7,
    .top_k = 20,
    .top_p = 0.8,
    .seed = 0,
    .min_p = 0.05,
    .repetition_penalty = 1.1,
};

test "resolveSampling: an omitted field takes the row of the resolved mode, a sent field wins" {
    const rows: DefaultRows = .init(.model, test_card_think, test_card_plain);
    const think = resolveSampling(.{}, rows.row(true), 9);
    try std.testing.expectEqual(
        protocol.Sampling{
            .temperature = 0.6,
            .top_k = 20,
            .top_p = 0.95,
            .seed = 9,
            .presence_penalty = 1.5,
        },
        think,
    );
    const plain = resolveSampling(.{}, rows.row(false), 9);
    try std.testing.expectEqual(
        protocol.Sampling{
            .temperature = 0.7,
            .top_k = 20,
            .top_p = 0.8,
            .seed = 9,
            .min_p = 0.05,
            .repetition_penalty = 1.1,
        },
        plain,
    );

    // Each sent field beats its card field, and only that field moves.
    const sent: SamplingFields = .{
        .temperature = 0,
        .top_k = 0,
        .top_p = 1,
        .min_p = 0,
        .presence_penalty = 0,
        .repetition_penalty = 1,
        .seed = 5,
    };
    try std.testing.expectEqual(
        protocol.Sampling{ .temperature = 0, .top_k = 0, .top_p = 1, .seed = 5 },
        resolveSampling(sent, rows.row(true), 5),
    );
    const one = resolveSampling(.{ .min_p = 0.2 }, rows.row(true), 3);
    try std.testing.expectEqual(@as(f32, 0.2), one.min_p);
    try std.testing.expectEqual(@as(f32, 0.6), one.temperature);
    try std.testing.expectEqual(@as(f32, 1.5), one.presence_penalty);
}

test "resolveSampling: greedy defaults keep every omitted field at the greedy identity" {
    const rows: DefaultRows = .init(.greedy, test_card_think, test_card_plain);
    for ([_]bool{ true, false }) |open| {
        const s = resolveSampling(.{}, rows.row(open), 11);
        try std.testing.expectEqual(
            protocol.Sampling{ .temperature = 0, .top_k = 0, .top_p = 1, .seed = 11 },
            s,
        );
    }

    // A model with no card has the greedy rows under `model` too.
    const no_card: DefaultRows = .init(.model, protocol.Sampling.greedy, protocol.Sampling.greedy);
    try std.testing.expectEqual(DefaultRows.greedy, no_card);
    // A sent field still wins under the greedy defaults.
    try std.testing.expectEqual(
        @as(f32, 0.9),
        resolveSampling(.{ .temperature = 0.9 }, rows.row(false), 1).temperature,
    );
    try std.testing.expectEqual(SamplingDefaults.greedy, SamplingDefaults.parse("greedy").?);
    try std.testing.expectEqual(SamplingDefaults.model, SamplingDefaults.parse("model").?);
    try std.testing.expect(SamplingDefaults.parse("card") == null);
}

test "resolveTurn: the budget picks the mode, and a tagless model accepts only the closed modes" {
    const qwen_on: chat.Kind = .{ .chatml_qwen = .on };
    const qwen_off: chat.Kind = .{ .chatml_qwen = .off };

    // Budget 0 closes the block even against enable_thinking true.
    for ([_]chat.ThinkingFlag{ .auto, .on, .off }) |flag| {
        const t = try resolveTurn(qwen_on, flag, .off, true);
        try std.testing.expect(!chat.thinkOpen(t.kind));
        try std.testing.expectEqual(@as(?u32, null), t.think_budget);
    }

    const server = try resolveTurn(qwen_on, .auto, .server, true);
    try std.testing.expect(chat.thinkOpen(server.kind));
    try std.testing.expectEqual(@as(?u32, null), server.think_budget);
    const capped = try resolveTurn(qwen_off, .on, .{ .tokens = 64 }, true);
    try std.testing.expect(chat.thinkOpen(capped.kind));
    try std.testing.expectEqual(@as(?u32, 64), capped.think_budget);
    // A budget does not open the block. It caps a block that the mode opens.
    const closed = try resolveTurn(qwen_off, .auto, .{ .tokens = 64 }, true);
    try std.testing.expect(!chat.thinkOpen(closed.kind));

    const smol: chat.Kind = .chatml_smollm2;
    try std.testing.expectError(
        error.BudgetUnsupported,
        resolveTurn(smol, .auto, .{ .tokens = 1 }, true),
    );
    try std.testing.expectError(error.ThinkingUnsupported, resolveTurn(smol, .on, .server, true));
    for ([_]ThinkBudget{ .server, .off }) |b| {
        const t = try resolveTurn(smol, .auto, b, true);
        try std.testing.expectEqual(smol, t.kind);
        try std.testing.expectEqual(@as(?u32, null), t.think_budget);
    }

    // Budget 0 turns enable_thinking true into off, so it passes on a tagless model.
    _ = try resolveTurn(smol, .on, .off, true);
    // A tagged model with no forced close refuses a budget above 0 and keeps the rest.
    try std.testing.expectError(
        error.BudgetNoForcedClose,
        resolveTurn(qwen_on, .auto, .{ .tokens = 8 }, false),
    );
    for ([_]ThinkBudget{ .server, .off }) |b| {
        _ = try resolveTurn(qwen_on, .auto, b, false);
    }
}

test "resolveDefaultMaxTokens: zero is the window, a flag past the window is the window" {
    const window: u32 = 128;
    var flag: u32 = 0;
    while (flag <= 2 * window) : (flag += 1) {
        const resolved = resolveDefaultMaxTokens(flag, window);
        try std.testing.expect(resolved >= 1);
        try std.testing.expect(resolved <= window);
        const want = if (flag == 0 or flag > window) window else flag;
        try std.testing.expectEqual(want, resolved);
    }

    try std.testing.expectEqual(window, resolveDefaultMaxTokens(std.math.maxInt(u32), window));
    try std.testing.expectEqual(@as(u32, 1), resolveDefaultMaxTokens(0, 1));
}

test "parseChatRequest: an omitted budget under the default flag is the whole window" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const window: u32 = 512;
    const default_max_tokens = resolveDefaultMaxTokens(limits.openai_default_max_tokens, window);
    try std.testing.expectEqual(window, default_max_tokens);
    var diag: Diag = .{};
    const r = try parseChatRequest(
        arena,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
        default_max_tokens,
        window,
        &diag,
    );
    try std.testing.expectEqual(window, r.max_tokens);
}

test "exceedContextError: the overflow carries both numbers, other errors carry none" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const e = try exceedContextError(arena, 130, 128);
    try std.testing.expectEqual(@as(u16, 400), e.status);
    try std.testing.expectEqualStrings("exceed_context_size_error", e.type);
    try std.testing.expectEqualStrings(
        "the request of 130 tokens exceeds the context of 128 tokens",
        e.message,
    );
    const wire = errorObject(e).@"error";
    try std.testing.expectEqual(@as(?u16, 400), wire.code);
    try std.testing.expectEqual(@as(?u32, 130), wire.n_prompt_tokens);
    try std.testing.expectEqual(@as(?u32, 128), wire.n_ctx);

    const plain = errorObject(reqErrorToHttp(error.BadJson, .{})).@"error";
    try std.testing.expectEqual(@as(?u16, null), plain.code);
    try std.testing.expectEqual(@as(?u32, null), plain.n_prompt_tokens);
    try std.testing.expectEqual(@as(?u32, null), plain.n_ctx);
    try std.testing.expectEqualStrings("malformed JSON body", plain.message);
}

test "completeUtf8Len: holds a split codepoint, passes a whole one" {
    // "a" + the first two bytes of a 3-byte '€' (E2 82 AC): emit "a", hold E2 82.
    const split = [_]u8{ 'a', 0xE2, 0x82 };
    try std.testing.expectEqual(@as(usize, 1), completeUtf8Len(&split));
    // Whole codepoint -> emit all.
    const whole = [_]u8{ 'a', 0xE2, 0x82, 0xAC };
    try std.testing.expectEqual(@as(usize, 4), completeUtf8Len(&whole));
    // Plain ASCII -> emit all.
    try std.testing.expectEqual(@as(usize, 3), completeUtf8Len("abc"));
    try std.testing.expectEqual(@as(usize, 0), completeUtf8Len(""));
}

test "reqErrorToHttp: thinking on a model without a thinking mode is a 400 in one sentence" {
    try std.testing.expectError(
        error.ThinkingUnsupported,
        chat.resolveThinking(.chatml_smollm2, .on),
    );
    const e = reqErrorToHttp(error.ThinkingUnsupported, .{});
    try std.testing.expectEqual(@as(u16, 400), e.status);
    try std.testing.expectEqualStrings("invalid_request_error", e.type);
    try std.testing.expectEqualStrings(
        "enable_thinking true does not apply to this model, " ++
            "because its chat template has no thinking mode",
        e.message,
    );
}

const test_tags: chat.ThinkIds = .{ .open = 1, .close = 2 };

const TestEvent = struct { id: u32, text: []const u8 };

const TestSplit = struct { reasoning: []const u8, content: []const u8 };

fn splitAll(
    arena: Allocator,
    tags: ?chat.ThinkIds,
    open: bool,
    events: []const TestEvent,
) !TestSplit {
    var split: ReasoningSplit = .init(tags, open);
    var reasoning: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    for (events) |ev| {
        const routed = split.route(ev.id, ev.text) orelse continue;
        switch (routed.field) {
            .reasoning => try reasoning.appendSlice(arena, routed.text),
            .content => try content.appendSlice(arena, routed.text),
        }
    }

    return .{ .reasoning = reasoning.items, .content = content.items };
}

test "ReasoningSplit: a turn that opens in the block splits at the natural close" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const events = [_]TestEvent{
        .{ .id = 1, .text = "<think>" },
        .{ .id = 10, .text = "Let me" },
        .{ .id = 11, .text = " think.\n" },
        .{ .id = 2, .text = "</think>" },
        .{ .id = 12, .text = "\n\n" },
        .{ .id = 13, .text = "Answer." },
        .{ .id = 14, .text = "\n" },
    };
    const got = try splitAll(arena_state.allocator(), test_tags, true, &events);
    try std.testing.expectEqualStrings("Let me think.\n", got.reasoning);
    try std.testing.expectEqualStrings("Answer.\n", got.content);
}

test "ReasoningSplit: a turn that opens closed keeps every piece as content" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    // A close outside the block is reply text, and no newline drops after it.
    const events = [_]TestEvent{
        .{ .id = 10, .text = "Hi" },
        .{ .id = 2, .text = "</think>" },
        .{ .id = 12, .text = "\n\n" },
        .{ .id = 11, .text = " there" },
    };
    const got = try splitAll(arena_state.allocator(), test_tags, false, &events);
    try std.testing.expectEqualStrings("", got.reasoning);
    try std.testing.expectEqualStrings("Hi</think>\n\n there", got.content);
}

test "ReasoningSplit: a block with no close before the finish is all reasoning" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const events = [_]TestEvent{
        .{ .id = 1, .text = "<think>" },
        .{ .id = 10, .text = "a" },
        .{ .id = 12, .text = "\n\n" },
        .{ .id = 11, .text = "b" },
    };
    const got = try splitAll(arena_state.allocator(), test_tags, true, &events);
    try std.testing.expectEqualStrings("a\n\nb", got.reasoning);
    try std.testing.expectEqualStrings("", got.content);
}

test "ReasoningSplit: the newlines after the close drop across events" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const spread = [_]TestEvent{
        .{ .id = 1, .text = "<think>" },
        .{ .id = 10, .text = "r" },
        .{ .id = 2, .text = "</think>" },
        .{ .id = 12, .text = "\n" },
        .{ .id = 13, .text = "\n\n" },
        .{ .id = 14, .text = "\n\nA\n" },
        .{ .id = 15, .text = "\n" },
    };
    const got = try splitAll(arena, test_tags, true, &spread);
    try std.testing.expectEqualStrings("r", got.reasoning);
    try std.testing.expectEqualStrings("A\n\n", got.content);

    // A close right after the echo leaves the reasoning empty.
    const immediate = [_]TestEvent{
        .{ .id = 1, .text = "<think>" },
        .{ .id = 2, .text = "</think>" },
        .{ .id = 14, .text = "A" },
    };
    const empty = try splitAll(arena, test_tags, true, &immediate);
    try std.testing.expectEqualStrings("", empty.reasoning);
    try std.testing.expectEqualStrings("A", empty.content);
}

test "ReasoningSplit: a model with no think tags passes every piece through" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const events = [_]TestEvent{
        .{ .id = 1, .text = "<think>" },
        .{ .id = 10, .text = "a" },
        .{ .id = 2, .text = "</think>" },
        .{ .id = 12, .text = "\n\n" },
    };
    const got = try splitAll(arena_state.allocator(), null, false, &events);
    try std.testing.expectEqualStrings("", got.reasoning);
    try std.testing.expectEqualStrings("<think>a</think>\n\n", got.content);
}

fn checkReasoningSplit(_: void, seed: u64) anyerror!void {
    const pieces = [_][]const u8{ "a", "b c", " ", "\n", "\n\n", "\nd", "e\n", "", "\xc3\xa9" };
    const run_max = 12;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const arena = arena_state.allocator();

    // Build the shapes the scheduler emits: an open turn echoes the tag and can close.
    const open = r.boolean();
    const closes = open and r.boolean();
    var events: std.ArrayList(TestEvent) = .empty;
    var untagged: std.ArrayList(u8) = .empty;
    var want_reasoning: std.ArrayList(u8) = .empty;
    var tail: std.ArrayList(u8) = .empty;
    if (open) {
        try events.append(arena, .{ .id = test_tags.open, .text = "<think>" });
        for (0..r.uintLessThan(usize, run_max + 1)) |_| {
            const k = r.uintLessThan(usize, pieces.len);
            try events.append(arena, .{ .id = 10 + @as(u32, @intCast(k)), .text = pieces[k] });
            try want_reasoning.appendSlice(arena, pieces[k]);
            try untagged.appendSlice(arena, pieces[k]);
        }
    }

    if (closes) {
        try events.append(arena, .{ .id = test_tags.close, .text = "</think>" });
    }

    if (!open or closes) {
        for (0..r.uintLessThan(usize, run_max + 1)) |_| {
            const k = r.uintLessThan(usize, pieces.len);
            try events.append(arena, .{ .id = 10 + @as(u32, @intCast(k)), .text = pieces[k] });
            try tail.appendSlice(arena, pieces[k]);
            try untagged.appendSlice(arena, pieces[k]);
        }
    }

    // Check the split against the model.
    const want_content = if (closes) std.mem.trimStart(u8, tail.items, "\n") else tail.items;
    const dropped = tail.items.len - want_content.len;

    const got = try splitAll(arena, test_tags, open, events.items);
    try std.testing.expectEqualStrings(want_reasoning.items, got.reasoning);
    try std.testing.expectEqualStrings(want_content, got.content);
    const at = got.reasoning.len;
    for (untagged.items[at..][0..dropped]) |b| {
        try std.testing.expectEqual(@as(u8, '\n'), b);
    }

    const joined = try std.mem.concat(arena, u8, &.{ got.reasoning, got.content });
    const kept = try std.mem.concat(
        arena,
        u8,
        &.{ untagged.items[0..at], untagged.items[at + dropped ..] },
    );
    try std.testing.expectEqualStrings(kept, joined);
}

test "ReasoningSplit property: the two fields equal the stream minus tags and dropped newlines" {
    try prop.forAll(
        .{ .site = 0x5E1D_7A65, .cases = 2000 },
        {},
        u64,
        prop.genSeed(void),
        checkReasoningSplit,
    );
}

/// The name that the message of each field refusal carries.
fn refusedField(e: ReqError, diag: Diag) []const u8 {
    return switch (e) {
        error.BadJson => "JSON",
        error.BadField => @tagName(diag.field.?),
        error.NoMessages, error.TooManyMessages => "messages",
        error.NTooLarge => "n=1",
        error.BadTemperature => "temperature",
        error.BadTopP => "top_p",
        error.BadTopK => "top_k",
        error.BadMinP => "min_p",
        error.BadPresencePenalty => "presence_penalty",
        error.BadRepetitionPenalty => "repetition_penalty",
        error.FrequencyUnsupported => "frequency_penalty",
        error.BadThinkingBudget,
        error.BudgetUnsupported,
        error.BudgetNoForcedClose,
        => "thinking_token_budget",
        error.ThinkingUnsupported => "enable_thinking",
        error.BadStop => "stop",
        error.TooManyParts, error.ContentPartUnsupported => "content",
        error.OutOfMemory => unreachable,
    };
}

const valid_messages =
    \\[{"role":"system","content":"s"},{"role":"user","content":"u"}]
;

/// Each field of a chat body with a value that the parse accepts.
const body_fields = [_]struct { name: []const u8, valid: []const u8 }{
    .{ .name = "model", .valid = "\"m\"" },
    .{ .name = "messages", .valid = valid_messages },
    .{ .name = "temperature", .valid = "0.7" },
    .{ .name = "top_p", .valid = "0.9" },
    .{ .name = "top_k", .valid = "40" },
    .{ .name = "min_p", .valid = "0.05" },
    .{ .name = "max_tokens", .valid = "16" },
    .{ .name = "stream", .valid = "true" },
    .{ .name = "seed", .valid = "7" },
    .{ .name = "presence_penalty", .valid = "1.5" },
    .{ .name = "repetition_penalty", .valid = "1.1" },
    .{ .name = "frequency_penalty", .valid = "0" },
    .{ .name = "n", .valid = "1" },
    .{ .name = "stop", .valid = "[\"x\"]" },
    .{ .name = "max_completion_tokens", .valid = "8" },
    .{ .name = "chat_template_kwargs", .valid = "{\"enable_thinking\":true}" },
    .{ .name = "thinking_token_budget", .valid = "64" },
};

/// Values at and past the edges of every field, and values of the wrong type.
const body_values = [_][]const u8{
    "0",
    "1",
    "-1",
    "-2",
    "0.5",
    "1.01",
    "2",
    "2.5",
    "256",
    "257",
    "262144",
    "262145",
    "4294967296",
    "1e999",
    "-1e999",
    "1e-45",
    "-0",
    "null",
    "true",
    "\"x\"",
    "[]",
    "{}",
    "[{}]",
    "[{\"role\":\"tool\",\"content\":\"\"}]",
    "{\"enable_thinking\":\"yes\"}",
    "99999999999999999999",
    "-9223372036854775808",
};

/// A random mutation of a valid chat body: each field keeps, edges, or drops, and some bodies
/// take raw byte writes. The parse gives a 400 that names the field, or a request that the
/// sampling and thinking resolvers accept for every model kind.
fn checkChatMutation(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    // Build a body whose fields keep their value, take an edge value, or drop out.
    var body: std.ArrayList(u8) = .empty;
    try body.append(arena, '{');
    var first = true;
    var edged: std.StaticBitSet(body_fields.len) = .empty;
    for (body_fields, 0..) |field, i| {
        const value = switch (r.uintLessThan(u8, 6)) {
            0 => continue,
            1 => blk: {
                edged.set(i);
                break :blk body_values[r.uintLessThan(usize, body_values.len)];
            },
            else => field.valid,
        };

        if (!first) {
            try body.append(arena, ',');
        }

        first = false;
        try body.print(arena, "\"{s}\":{s}", .{ field.name, value });
    }

    try body.append(arena, '}');

    // Corrupt some bodies with raw byte writes.
    const raw = r.uintLessThan(u8, 4) == 0;
    if (raw) {
        for (0..1 + r.uintLessThan(usize, 4)) |_| {
            body.items[r.uintLessThan(usize, body.items.len)] = r.int(u8);
        }
    }

    // Parse. A refusal must name its field.
    const window = 1 + r.uintLessThan(u32, 4096);
    const default_max = 1 + r.uintLessThan(u32, window);
    var diag: Diag = .{};
    const req = parseChatRequest(arena, body.items, default_max, window, &diag) catch |err| {
        const http = reqErrorToHttp(err, diag);
        try std.testing.expectEqual(@as(u16, 400), http.status);
        try std.testing.expect(std.mem.find(u8, http.message, refusedField(err, diag)) != null);
        // Whole JSON syntax is never malformed, and a wrong type names an edged field.
        if (!raw) {
            try std.testing.expect(err != error.BadJson);
        }

        if (!raw and err == error.BadField) {
            const name = @tagName(diag.field.?);
            const at = for (body_fields, 0..) |field, i| {
                if (std.mem.eql(u8, field.name, name)) break i;
            } else return error.TestUnexpectedResult;

            try std.testing.expect(edged.isSet(at));
        }

        return;
    };

    // Resolve the accepted request for every model kind.
    try std.testing.expect(req.messages.len >= 1);
    try std.testing.expect(req.max_tokens >= 1 and req.max_tokens <= window);
    const kinds = [_]chat.Kind{
        .none,
        .chatml_smollm2,
        .llama3,
        .{ .chatml_qwen = .off },
        .{ .chatml_qwen = .on },
    };
    for (kinds) |kind| {
        for ([_]bool{ false, true }) |forced_close| {
            const turn = resolveTurn(
                kind,
                req.thinking,
                req.think_budget,
                forced_close,
            ) catch |err| {
                const http = reqErrorToHttp(err, .{});
                try std.testing.expectEqual(@as(u16, 400), http.status);
                try std.testing.expect(
                    std.mem.find(u8, http.message, refusedField(err, .{})) != null,
                );
                continue;
            };
            const rows = DefaultRows.init(.model, test_card_think, test_card_plain);
            // The resolver asserts every range that the wire guards hold.
            _ = resolveSampling(
                req.sampling,
                rows.row(chat.thinkOpen(turn.kind)),
                req.sampling.seed orelse 0,
            );
        }
    }
}

test "property: a mutated body is a 400 that names its field, or a request all resolvers accept" {
    try prop.forAll(
        .{ .site = 0x0A1_B0D1, .cases = 1500 },
        {},
        u64,
        prop.genSeed(void),
        checkChatMutation,
    );
}

/// Feed `pieces` and return the emitted text, whether a stop cut it, and the held bytes that a
/// natural end flushes.
const MatchRun = struct { text: []const u8, stopped: bool };

fn runMatcher(arena: Allocator, stops: []const []const u8, pieces: []const []const u8) !MatchRun {
    var m: StopMatcher = .init(stops);
    var out: std.ArrayList(u8) = .empty;
    for (pieces) |piece| {
        const o = m.feed(piece);
        try out.appendSlice(arena, o.text);
        if (o.stopped) return .{ .text = out.items, .stopped = true };
    }

    try out.appendSlice(arena, m.finish());
    return .{ .text = out.items, .stopped = false };
}

test "StopMatcher: the reply ends just before the first stop string, in every split" {
    const Case = struct {
        stops: []const []const u8,
        pieces: []const []const u8,
        text: []const u8,
        stopped: bool,
    };
    const cases = [_]Case{
        // Inside one piece, across two, and across three.
        .{ .stops = &.{"END"}, .pieces = &.{"abcENDxyz"}, .text = "abc", .stopped = true },
        .{ .stops = &.{"END"}, .pieces = &.{ "abcE", "NDxyz" }, .text = "abc", .stopped = true },
        .{ .stops = &.{"END"}, .pieces = &.{ "abE", "N", "Dx" }, .text = "ab", .stopped = true },
        // Overlapping candidates: the first "aab" fails, the second one matches.
        .{ .stops = &.{"aab"}, .pieces = &.{ "xa", "a", "aab" }, .text = "xaa", .stopped = true },
        // Two stop strings: the earlier position wins, whatever the list order.
        .{ .stops = &.{ "yz", "bc" }, .pieces = &.{"abcxyz"}, .text = "a", .stopped = true },
        // A stop string that is a prefix of another one.
        .{ .stops = &.{ "abc", "ab" }, .pieces = &.{ "xa", "bc" }, .text = "x", .stopped = true },
        // A two-byte code point split at the boundary, inside a stop string.
        .{
            .stops = &.{"\u{e9}t"},
            .pieces = &.{ "caf\xc3", "\xa9t!" },
            .text = "caf",
            .stopped = true,
        },
        // A held prefix that never completes flushes at the end.
        .{ .stops = &.{"END"}, .pieces = &.{ "abcE", "N" }, .text = "abcEN", .stopped = false },
        .{ .stops = &.{"END"}, .pieces = &.{ "abcE", "Nx" }, .text = "abcENx", .stopped = false },
        // No stop string passes every byte at once.
        .{ .stops = &.{}, .pieces = &.{ "ab", "c" }, .text = "abc", .stopped = false },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("StopMatcher case {d}\n", .{i});

        var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena_state.deinit();

        const got = try runMatcher(arena_state.allocator(), c.stops, c.pieces);
        try std.testing.expectEqualStrings(c.text, got.text);
        try std.testing.expectEqual(c.stopped, got.stopped);
    }
}

test "StopMatcher: a held tail stays below the longest stop string" {
    var m: StopMatcher = .init(&.{"abcd"});
    // Every byte can start the stop string, so each call holds the longest matching tail.
    try std.testing.expectEqualStrings("", m.feed("a").text);
    try std.testing.expectEqualStrings("", m.feed("b").text);
    try std.testing.expectEqualStrings("", m.feed("c").text);
    // "abca" holds no stop string, and only its last "a" can still start one.
    try std.testing.expectEqualStrings("abc", m.feed("a").text);
    try std.testing.expectEqualStrings("a", m.finish());
}

test "parseChatRequest: content forms and stop strings" {
    const cases = [_]ParseCase{
        // Text parts join in order with no separator.
        .{
            .body = "{\"messages\":[{\"role\":\"user\",\"content\":[" ++
                "{\"type\":\"text\",\"text\":\"ab\"},{\"type\":\"text\",\"text\":\"c\"}]}]}",
            .want = .{ .first_content = "abc" },
        },
        // Null on an assistant message is the empty text, and absent content is too.
        .{
            .body = "{\"messages\":[{\"role\":\"assistant\",\"content\":null}]}",
            .want = .{ .first_content = "" },
        },
        .{
            .body = "{\"messages\":[{\"role\":\"user\"}]}",
            .want = .{ .first_content = "" },
        },
        // Null on a user message, a number, and a malformed part keep the wrong-type 400.
        .{
            .body = "{\"messages\":[{\"role\":\"user\",\"content\":null}]}",
            .want = .{ .err = error.BadField, .field = .messages },
        },
        .{
            .body = "{\"messages\":[{\"role\":\"user\",\"content\":7}]}",
            .want = .{ .err = error.BadField, .field = .messages },
        },
        .{
            .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"text\":\"x\"}]}]}",
            .want = .{ .err = error.BadField, .field = .messages },
        },
        .{
            .body = "{\"messages\":[{\"role\":\"user\",\"content\":[" ++
                "{\"type\":\"input_audio\"}]}]}",
            .want = .{ .err = error.ContentPartUnsupported },
        },
        .{
            .body = "{\"messages\":[{\"role\":\"user\",\"content\":[" ++
                ("{\"type\":\"text\",\"text\":\"x\"}," ** 64) ++
                "{\"type\":\"text\",\"text\":\"x\"}]}]}",
            .want = .{ .err = error.TooManyParts },
        },
        // Stop: a string, an array, and each edge.
        .{ .body = withExtra(",\"stop\":\"x\""), .want = .{ .stops = 1 } },
        .{ .body = withExtra(",\"stop\":[\"a\",\"b\",\"c\",\"d\"]"), .want = .{ .stops = 4 } },
        .{ .body = withExtra(",\"stop\":null"), .want = .{ .stops = 0 } },
        refuse(",\"stop\":[\"a\",\"b\",\"c\",\"d\",\"e\"]", error.BadStop),
        refuse(",\"stop\":\"\"", error.BadStop),
        refuse(",\"stop\":[1]", error.BadStop),
        refuse(",\"stop\":7", error.BadStop),
        refuse(",\"stop\":\"" ++ "x" ** 65 ++ "\"", error.BadStop),
        .{ .body = withExtra(",\"stop\":\"" ++ "x" ** 64 ++ "\""), .want = .{ .stops = 1 } },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("content and stop case {d}\n", .{i});
        try checkParse(c);
    }
}

test "reqErrorToHttp: a content part of another type is a 400 that names the type" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var diag: Diag = .{};
    const body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\"}]}]}";
    const got = parseChatRequest(arena_state.allocator(), body, 8, 64, &diag);
    try std.testing.expectError(error.ContentPartUnsupported, got);
    const http = reqErrorToHttp(error.ContentPartUnsupported, diag);
    try std.testing.expectEqual(@as(u16, 400), http.status);
    try std.testing.expect(std.mem.find(u8, http.message, "\"image_url\"") != null);
    try std.testing.expect(std.mem.find(u8, http.message, "text only") != null);
}
