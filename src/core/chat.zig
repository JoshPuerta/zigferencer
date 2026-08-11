//! Per-model chat-template wrap: a raw user message in, tokenizer-ready text
//! out. The wrap interpolates the message into the right scaffolding for the
//! model's family (SmolLM2 ChatML, Llama-3 chat, Qwen ChatML, or pass-through).
//! The GGUF embeds a full Jinja2 template, but a Jinja interpreter costs
//! hundreds of LOC against zero deps for marginal value. Hand-coded templates
//! per family stay legible. They also pair-validate against the special tokens
//! the BPE encoder matches greedily: a template that uses
//! `<|start_header_id|>` works only because `specials_sorted` carries it. To
//! add a family, add a tag to the `Kind` union. A family with a mode carries
//! the mode as the payload, as `chatml_qwen` carries `Thinking`. Then add a
//! template literal, dispatch in each wrap, and set `Model.chat_template_kind`
//! in `registry.zig`.

const std = @import("std");
const prop = @import("prop.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.run);

// -- types ------------------------------------------------------------------

/// The thinking mode of a family that can reason before it answers. The text
/// after the assistant header selects it.
pub const Thinking = enum { off, on };

pub const Kind = union(enum) {
    /// No wrap. `--chat` passes the prompt through unchanged (callers can
    /// inline their own template). The right default for base models or for
    /// running an instruct model in plain-text-continuation mode.
    none,
    /// SmolLM2-Instruct's ChatML wrap, including the trained system message.
    /// Without this exact wrap, SmolLM2 generates raw text and loops quickly.
    chatml_smollm2,
    /// Llama-3 chat format: BOS + `<|start_header_id|>user<|end_header_id|>`
    /// + content + `<|eot_id|>` + assistant header. EOS is `<|eot_id|>`
    /// (id 128009 for Llama-3.1), which the existing eos-stop logic catches.
    llama3,
    /// Qwen ChatML: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`,
    /// then the thinking tail of the mode. No BOS (Qwen sets
    /// add_bos_token=false). EOS is `<|im_end|>`.
    chatml_qwen: Thinking,
};

/// The value of a `--thinking` flag. `auto` keeps the registry mode of the
/// model. `on` and `off` set the mode of a family that has one.
pub const ThinkingFlag = enum {
    auto,
    on,
    off,

    pub fn parse(raw: []const u8) ?ThinkingFlag {
        return std.meta.stringToEnum(ThinkingFlag, raw);
    }
};

/// The ids of the `<think>` and `</think>` tags in a model's vocab.
pub const ThinkIds = struct { open: u32, close: u32 };

/// A conversation role, in the OpenAI sense. `wrapMessagesAlloc` renders it
/// verbatim into the model's chat scaffolding.
pub const Role = enum {
    system,
    user,
    assistant,

    fn str(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
        };
    }
};

/// One message in a conversation, the value the OpenAI connector passes to
/// `wrapMessagesAlloc`.
pub const Message = struct { role: Role, content: []const u8 };

/// Where a kept (user, assistant) pair sits in a conversation. It names the
/// user wrap that a live turn at the same place used.
pub const PairPlace = enum { open_first, first_after_system, continuation };

/// The system prompt SmolLM2-Instruct was trained with. The `chatml_smollm2`
/// template of `wrapAlloc` embeds it. `wrapMessagesAlloc` injects it when a
/// conversation carries no system message, so a bare user turn reproduces the
/// `wrapAlloc` text exactly (pair-validated in tests).
const smollm2_default_system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

const chatml_smollm2 =
    \\<|im_start|>system
    \\You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
    \\<|im_start|>user
    \\{s}<|im_end|>
    \\<|im_start|>assistant
    \\
;

const chatml_system =
    \\<|im_start|>system
    \\{s}<|im_end|>
    \\
;

const chatml_first_turn =
    \\<|im_start|>user
    \\{s}<|im_end|>
    \\<|im_start|>assistant
    \\
;

// No BOS for the same reason as `llama3` below: the encoder owns it.
const llama3_system =
    \\<|start_header_id|>system<|end_header_id|>
    \\
    \\{s}<|eot_id|>
;

// No <|begin_of_text|> here: the encoder prepends BOS per the model's
// tokenizer metadata (vocab.add_bos), and a template that carries it too
// doubles it.
const llama3 =
    \\<|start_header_id|>user<|end_header_id|>
    \\
    \\{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    \\
    \\
;

// Follow-up turns carry no BOS. Lead with <|eot_id|> to close the previous
// assistant reply (its terminating <|eot_id|> was never decoded into the KV).
const chatml_smollm2_continuation =
    \\<|im_end|>
    \\<|im_start|>user
    \\{s}<|im_end|>
    \\<|im_start|>assistant
    \\
;

const llama3_continuation =
    \\<|eot_id|><|start_header_id|>user<|end_header_id|>
    \\
    \\{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    \\
    \\
;

const chatml_qwen =
    \\<|im_start|>user
    \\{s}<|im_end|>
    \\<|im_start|>assistant
    \\
;

const chatml_qwen_continuation =
    \\<|im_end|>
    \\<|im_start|>user
    \\{s}<|im_end|>
    \\<|im_start|>assistant
    \\
;

/// The think tags of a family as text. `separator` follows the close tag
/// before the answer starts.
pub const ThinkText = struct { open: []const u8, close: []const u8, separator: []const u8 };

/// The tags the Qwen templates write.
pub const qwen_think_text: ThinkText = .{ .open = "<think>", .close = "</think>", .separator = "\n\n" };

// The model file's own template writes these tails after the assistant header.
// The closed block tells the model to answer with no reasoning.
const qwen_think_open = qwen_think_text.open ++ "\n";
const qwen_think_closed = qwen_think_text.open ++ "\n\n" ++ qwen_think_text.close ++ qwen_think_text.separator;

// -- api --------------------------------------------------------------------

/// Apply a `--thinking` flag to a registry kind. `auto` keeps the kind. `on`
/// and `off` set the `chatml_qwen` mode. A family with no thinking mode never
/// thinks, so `off` keeps it and `on` fails.
pub fn resolveThinking(kind: Kind, flag: ThinkingFlag) error{ThinkingUnsupported}!Kind {
    const out: Kind = switch (kind) {
        .chatml_qwen => |mode| .{ .chatml_qwen = switch (flag) {
            .auto => mode,
            .on => .on,
            .off => .off,
        } },
        .none, .chatml_smollm2, .llama3 => if (flag == .on) return error.ThinkingUnsupported else kind,
    };
    std.debug.assert(std.meta.activeTag(out) == std.meta.activeTag(kind));
    if (flag == .on) std.debug.assert(thinkOpen(out));
    if (flag == .off) std.debug.assert(!thinkOpen(out));
    return out;
}

/// True when the wrap leaves a `<think>` block open, so the model starts its
/// reply inside the block.
pub fn thinkOpen(kind: Kind) bool {
    return switch (kind) {
        .chatml_qwen => |mode| mode == .on,
        .none, .chatml_smollm2, .llama3 => false,
    };
}

/// The think tags of `kind` as text, or null for a family with no thinking
/// mode. The serve launcher encodes them into the ids the scheduler forces.
pub fn thinkText(kind: Kind) ?ThinkText {
    return switch (kind) {
        .chatml_qwen => qwen_think_text,
        .none, .chatml_smollm2, .llama3 => null,
    };
}

/// The reply of an assistant message without its think block: the text after
/// the last `close` tag, with leading newlines dropped. Content with no close
/// tag comes back unchanged. The model file's template renders an earlier
/// reply the same way.
pub fn stripThink(content: []const u8, close: []const u8) []const u8 {
    std.debug.assert(close.len > 0);
    const at = std.mem.findLast(u8, content, close) orelse return content;
    const reply = std.mem.trimStart(u8, content[at + close.len ..], "\n");
    std.debug.assert(reply.ptr + reply.len == content.ptr + content.len);
    std.debug.assert(std.mem.find(u8, reply, close) == null);
    return reply;
}

pub fn wrapAlloc(arena: Allocator, kind: Kind, user_msg: []const u8) Allocator.Error![]u8 {
    log.debug("chat template: {t}, think open {}, single turn, {d} msg chars", .{ kind, thinkOpen(kind), user_msg.len });
    return switch (kind) {
        .none => arena.dupe(u8, user_msg),
        .chatml_smollm2 => std.fmt.allocPrint(arena, chatml_smollm2, .{user_msg}),
        .llama3 => std.fmt.allocPrint(arena, llama3, .{user_msg}),
        .chatml_qwen => |mode| std.fmt.allocPrint(arena, chatml_qwen ++ "{s}", .{ user_msg, qwenThinkTail(mode) }),
    };
}

/// Wrap a follow-up user turn for an ongoing conversation. Unlike `wrapAlloc`,
/// it emits no BOS or system scaffolding (already in the KV cache from turn 0).
/// It opens with the turn-end token that closes the previous assistant reply.
/// `bench-model --turns N` uses it so follow-up turns produce real responses
/// instead of degenerating to immediate EOS.
pub fn wrapContinuationAlloc(arena: Allocator, kind: Kind, user_msg: []const u8) Allocator.Error![]u8 {
    return switch (kind) {
        .none => arena.dupe(u8, user_msg),
        .chatml_smollm2 => std.fmt.allocPrint(arena, chatml_smollm2_continuation, .{user_msg}),
        .llama3 => std.fmt.allocPrint(arena, llama3_continuation, .{user_msg}),
        .chatml_qwen => |mode| std.fmt.allocPrint(arena, chatml_qwen_continuation ++ "{s}", .{ user_msg, qwenThinkTail(mode) }),
    };
}

/// Wrap a system prompt as the conversation's opening system block, with no
/// user turn. The serving layer prefills this once at session open. The first
/// user turn then arrives through `wrapFirstTurnAlloc`. For ChatML-family
/// templates, system + first-turn concatenated reproduce the full `wrapAlloc`
/// text exactly (pair-validated in tests). `.none` passes the system prompt
/// through as a raw preamble.
pub fn wrapSystemAlloc(arena: Allocator, kind: Kind, system_prompt: []const u8) Allocator.Error![]u8 {
    return switch (kind) {
        .none => arena.dupe(u8, system_prompt),
        .chatml_smollm2 => std.fmt.allocPrint(arena, chatml_system, .{system_prompt}),
        .llama3 => std.fmt.allocPrint(arena, llama3_system, .{system_prompt}),
        .chatml_qwen => std.fmt.allocPrint(arena, chatml_system, .{system_prompt}),
    };
}

/// Wrap the FIRST user turn after an opening system block. The result is the
/// full template's user+assistant scaffolding without its system block, and
/// without a leading turn-close (the system block closed itself). Later turns
/// use `wrapContinuationAlloc`.
pub fn wrapFirstTurnAlloc(arena: Allocator, kind: Kind, user_msg: []const u8) Allocator.Error![]u8 {
    return switch (kind) {
        .none => arena.dupe(u8, user_msg),
        .chatml_smollm2 => std.fmt.allocPrint(arena, chatml_first_turn, .{user_msg}),
        // Llama-3's base template carries no system block, so its first turn
        // after one is exactly the full wrap.
        .llama3 => std.fmt.allocPrint(arena, llama3, .{user_msg}),
        .chatml_qwen => |mode| std.fmt.allocPrint(arena, chatml_first_turn ++ "{s}", .{ user_msg, qwenThinkTail(mode) }),
    };
}

/// The user part of a kept (user, assistant) pair: the user wrap of a live
/// turn at `place`, with the closed think tail. The template of the model file
/// drops the reasoning of an earlier turn, so a pair never opens a block. The
/// caller encodes the assistant text apart and appends it. The wrap writes no
/// end of the assistant turn, because the next wrap opens with it.
pub fn wrapPairUserAlloc(arena: Allocator, kind: Kind, place: PairPlace, user_msg: []const u8) Allocator.Error![]u8 {
    const closed: Kind = switch (kind) {
        .chatml_qwen => .{ .chatml_qwen = .off },
        .none, .chatml_smollm2, .llama3 => kind,
    };
    std.debug.assert(!thinkOpen(closed));
    std.debug.assert(std.meta.activeTag(closed) == std.meta.activeTag(kind));
    return switch (place) {
        .open_first => wrapAlloc(arena, closed, user_msg),
        .first_after_system => wrapFirstTurnAlloc(arena, closed, user_msg),
        .continuation => wrapContinuationAlloc(arena, closed, user_msg),
    };
}

/// Render a whole conversation into one tokenizer-ready string for `kind`,
/// ending with the open assistant turn. The result is the OpenAI connector's
/// prefill: an ephemeral session has an empty KV, so the whole history renders
/// at once. Built from the same per-family fragments as the single-turn
/// wrappers, and pair-validated against them: `[user]` equals
/// `wrapAlloc(user)`, and `[system, user]` equals
/// `wrapSystemAlloc(system) ++ wrapFirstTurnAlloc(user)`. BOS is the encoder's
/// job (`sequence_start`), never the template's. A Qwen assistant message
/// renders without its think block (`stripThink`).
pub fn wrapMessagesAlloc(arena: Allocator, kind: Kind, messages: []const Message) Allocator.Error![]u8 {
    std.debug.assert(messages.len >= 1);
    var out: std.ArrayList(u8) = .empty;
    switch (kind) {
        .none => for (messages, 0..) |m, i| {
            if (i > 0) try out.append(arena, '\n');
            try out.appendSlice(arena, m.content);
        },
        .chatml_smollm2, .chatml_qwen => {
            // SmolLM2 expects its trained system block. Inject it when the
            // caller sent none, so a bare user turn equals `wrapAlloc`. Qwen's
            // base wrap carries no system block, so it injects nothing.
            if (kind == .chatml_smollm2 and !hasSystem(messages)) {
                try out.appendSlice(arena, "<|im_start|>system\n");
                try out.appendSlice(arena, smollm2_default_system);
                try out.appendSlice(arena, "<|im_end|>\n");
            }
            for (messages) |m| {
                const think = if (m.role == .assistant) thinkText(kind) else null;
                try out.appendSlice(arena, "<|im_start|>");
                try out.appendSlice(arena, m.role.str());
                try out.append(arena, '\n');
                try out.appendSlice(arena, if (think) |t| stripThink(m.content, t.close) else m.content);
                try out.appendSlice(arena, "<|im_end|>\n");
            }
            try out.appendSlice(arena, "<|im_start|>assistant\n");
            if (kind == .chatml_qwen) try out.appendSlice(arena, qwenThinkTail(kind.chatml_qwen));
        },
        .llama3 => {
            for (messages) |m| {
                try out.appendSlice(arena, "<|start_header_id|>");
                try out.appendSlice(arena, m.role.str());
                try out.appendSlice(arena, "<|end_header_id|>\n\n");
                try out.appendSlice(arena, m.content);
                try out.appendSlice(arena, "<|eot_id|>");
            }
            try out.appendSlice(arena, "<|start_header_id|>assistant<|end_header_id|>\n\n");
        },
    }
    std.debug.assert(out.items.len > 0);
    log.debug("chat template: {t}, think open {}, {d} messages, {d} chars", .{ kind, thinkOpen(kind), messages.len, out.items.len });
    return out.items;
}

// -- helpers ----------------------------------------------------------------

fn hasSystem(messages: []const Message) bool {
    for (messages) |m| if (m.role == .system) return true;
    return false;
}

fn qwenThinkTail(mode: Thinking) []const u8 {
    return switch (mode) {
        .on => qwen_think_open,
        .off => qwen_think_closed,
    };
}

// -- tests ------------------------------------------------------------------

const templated_kinds = [_]Kind{ .chatml_smollm2, .llama3, .{ .chatml_qwen = .off }, .{ .chatml_qwen = .on } };
const all_kinds = [_]Kind{.none} ++ templated_kinds;

const WrapCase = struct {
    kind: Kind,
    text: []const u8,
    exact: ?[]const u8 = null,
    starts: ?[]const u8 = null,
    ends: ?[]const u8 = null,
    has: []const []const u8 = &.{},
    lacks: []const []const u8 = &.{},
};

fn checkWrap(c: WrapCase) !void {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const out = try wrapAlloc(arena_state.allocator(), c.kind, c.text);
    if (c.exact) |v| try std.testing.expectEqualStrings(v, out);
    if (c.starts) |v| try std.testing.expect(std.mem.startsWith(u8, out, v));
    if (c.ends) |v| try std.testing.expect(std.mem.endsWith(u8, out, v));
    for (c.has) |v| try std.testing.expect(std.mem.indexOf(u8, out, v) != null);
    for (c.lacks) |v| try std.testing.expect(std.mem.indexOf(u8, out, v) == null);
}

test "wrapAlloc: each template wraps its scaffolding around the text" {
    const cases = [_]WrapCase{
        .{ .kind = .none, .text = "Hello", .exact = "Hello" },
        .{ .kind = .chatml_smollm2, .text = "Hi", .has = &.{"<|im_start|>user\nHi<|im_end|>"}, .ends = "<|im_start|>assistant\n" },
        // BOS comes from the encoder (vocab.add_bos), never from the template. A
        // template that carries it too doubles it.
        .{
            .kind = .llama3,
            .text = "Hi",
            .lacks = &.{"<|begin_of_text|>"},
            .starts = "<|start_header_id|>user<|end_header_id|>",
            .has = &.{ "Hi<|eot_id|>", "<|start_header_id|>assistant<|end_header_id|>" },
        },
    };
    for (cases) |c| try checkWrap(c);
}

test "wrapSystemAlloc + wrapFirstTurnAlloc reconstruct the full SmolLM2 wrap byte-for-byte" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The trained system prompt through the split path must equal the
    // monolithic template. That is the pair validation that the boundary
    // introduces nothing.
    const sys = try wrapSystemAlloc(arena, .chatml_smollm2, "You are a helpful AI assistant named SmolLM, trained by Hugging Face");
    const turn = try wrapFirstTurnAlloc(arena, .chatml_smollm2, "Hi");
    const joined = try std.mem.concat(arena, u8, &.{ sys, turn });
    const whole = try wrapAlloc(arena, .chatml_smollm2, "Hi");
    try std.testing.expectEqualStrings(whole, joined);
}

test "wrapSystemAlloc .llama3 emits a lone system block, no BOS" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const sys = try wrapSystemAlloc(arena, .llama3, "You are a helpful assistant.");
    try std.testing.expect(std.mem.indexOf(u8, sys, "<|begin_of_text|>") == null);
    try std.testing.expect(std.mem.startsWith(u8, sys, "<|start_header_id|>system<|end_header_id|>"));
    try std.testing.expect(std.mem.endsWith(u8, sys, "You are a helpful assistant.<|eot_id|>"));
    // First turn after it carries no system scaffolding of its own.
    const turn = try wrapFirstTurnAlloc(arena, .llama3, "Hi");
    try std.testing.expect(std.mem.indexOf(u8, turn, "system") == null);
    try std.testing.expect(std.mem.startsWith(u8, turn, "<|start_header_id|>user<|end_header_id|>"));
}

test "wrapMessagesAlloc reconstructs the single-turn wrappers byte-for-byte" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // A bare user turn equals the monolithic wrap (SmolLM2 injects its trained
    // system, Qwen injects nothing). The OpenAI path produces the same tokens
    // as the native serve path, so the cross-backend token hash holds.
    for (templated_kinds) |kind| {
        const msgs = [_]Message{.{ .role = .user, .content = "Hi" }};
        const whole = try wrapMessagesAlloc(arena, kind, &msgs);
        const ref = try wrapAlloc(arena, kind, "Hi");
        try std.testing.expectEqualStrings(ref, whole);
    }

    // A system + user pair equals the split system/first-turn path.
    for (templated_kinds) |kind| {
        const msgs = [_]Message{
            .{ .role = .system, .content = "Be brief." },
            .{ .role = .user, .content = "Hi" },
        };
        const whole = try wrapMessagesAlloc(arena, kind, &msgs);
        const sys = try wrapSystemAlloc(arena, kind, "Be brief.");
        const turn = try wrapFirstTurnAlloc(arena, kind, "Hi");
        const ref = try std.mem.concat(arena, u8, &.{ sys, turn });
        try std.testing.expectEqualStrings(ref, whole);
    }
}

test "wrapContinuationAlloc .llama3 omits BOS and closes the prior turn" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const out = try wrapContinuationAlloc(arena, .llama3, "More?");
    try std.testing.expect(std.mem.indexOf(u8, out, "<|begin_of_text|>") == null);
    try std.testing.expect(std.mem.startsWith(u8, out, "<|eot_id|><|start_header_id|>user<|end_header_id|>"));
    try std.testing.expect(std.mem.indexOf(u8, out, "More?<|eot_id|>") != null);
    try std.testing.expect(std.mem.endsWith(u8, out, "<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

test "the qwen wraps end with the model file tail of each mode" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const off: Kind = .{ .chatml_qwen = .off };
    const on: Kind = .{ .chatml_qwen = .on };
    try std.testing.expectEqualStrings(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        try wrapAlloc(arena, off, "Hi"),
    );
    try std.testing.expectEqualStrings(
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
        try wrapAlloc(arena, on, "Hi"),
    );
    try std.testing.expectEqualStrings(
        "<|im_end|>\n<|im_start|>user\nMore?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        try wrapContinuationAlloc(arena, on, "More?"),
    );
    try std.testing.expectEqualStrings(
        try wrapSystemAlloc(arena, off, "Be brief."),
        try wrapSystemAlloc(arena, on, "Be brief."),
    );
}

test "every wrap leaves one think block open exactly when thinkOpen holds" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const msgs = [_]Message{.{ .role = .user, .content = "Hi" }};
    for (all_kinds) |kind| {
        const wraps = [_][]const u8{
            try wrapAlloc(arena, kind, "Hi"),
            try wrapContinuationAlloc(arena, kind, "Hi"),
            try wrapFirstTurnAlloc(arena, kind, "Hi"),
            try wrapMessagesAlloc(arena, kind, &msgs),
        };
        for (wraps) |w| {
            const opens = std.mem.count(u8, w, "<think>");
            const closes = std.mem.count(u8, w, "</think>");
            try std.testing.expectEqual(@as(usize, @intFromBool(thinkOpen(kind))), opens - closes);
            try std.testing.expectEqual(thinkOpen(kind), std.mem.endsWith(u8, w, "<think>\n"));
        }
    }
}

test "resolveThinking maps every kind and flag" {
    const Case = struct { kind: Kind, flag: ThinkingFlag, want: ?Kind };
    const off: Kind = .{ .chatml_qwen = .off };
    const on: Kind = .{ .chatml_qwen = .on };
    const cases = [_]Case{
        .{ .kind = .none, .flag = .auto, .want = .none },
        .{ .kind = .none, .flag = .off, .want = .none },
        .{ .kind = .none, .flag = .on, .want = null },
        .{ .kind = .chatml_smollm2, .flag = .auto, .want = .chatml_smollm2 },
        .{ .kind = .chatml_smollm2, .flag = .off, .want = .chatml_smollm2 },
        .{ .kind = .chatml_smollm2, .flag = .on, .want = null },
        .{ .kind = .llama3, .flag = .auto, .want = .llama3 },
        .{ .kind = .llama3, .flag = .off, .want = .llama3 },
        .{ .kind = .llama3, .flag = .on, .want = null },
        .{ .kind = off, .flag = .auto, .want = off },
        .{ .kind = off, .flag = .off, .want = off },
        .{ .kind = off, .flag = .on, .want = on },
        .{ .kind = on, .flag = .auto, .want = on },
        .{ .kind = on, .flag = .off, .want = off },
        .{ .kind = on, .flag = .on, .want = on },
    };
    comptime std.debug.assert(cases.len == all_kinds.len * std.meta.fields(ThinkingFlag).len);
    for (cases) |c| {
        if (c.want) |want| {
            try std.testing.expectEqual(want, try resolveThinking(c.kind, c.flag));
        } else {
            try std.testing.expectError(error.ThinkingUnsupported, resolveThinking(c.kind, c.flag));
        }
    }
}

test "ThinkingFlag.parse accepts on, off, and auto only" {
    try std.testing.expectEqual(ThinkingFlag.on, ThinkingFlag.parse("on").?);
    try std.testing.expectEqual(ThinkingFlag.off, ThinkingFlag.parse("off").?);
    try std.testing.expectEqual(ThinkingFlag.auto, ThinkingFlag.parse("auto").?);
    try std.testing.expect(ThinkingFlag.parse("On") == null);
    try std.testing.expect(ThinkingFlag.parse("true") == null);
    try std.testing.expect(ThinkingFlag.parse("") == null);
}

test "stripThink keeps the reply after the last think block" {
    const plain = "Just an answer.";
    try std.testing.expectEqual(plain.ptr, stripThink(plain, qwen_think_text.close).ptr);
    try std.testing.expectEqualStrings(plain, stripThink(plain, qwen_think_text.close));
    try std.testing.expectEqualStrings("Answer.", stripThink("<think>\nreason\n</think>\n\nAnswer.", qwen_think_text.close));
    try std.testing.expectEqualStrings("Answer.", stripThink("Before<think>reason</think>\nAnswer.", qwen_think_text.close));
    try std.testing.expectEqualStrings("B", stripThink("<think>a</think>A<think>b</think>\nB", qwen_think_text.close));
    try std.testing.expectEqualStrings("", stripThink("<think>a</think>\n\n", qwen_think_text.close));
    // Only newlines drop, as in the model file's template.
    try std.testing.expectEqualStrings(" A", stripThink("</think>\n A", qwen_think_text.close));
}

fn checkStripThink(_: void, seed: u64) anyerror!void {
    const fragments = [_][]const u8{ "<think>", "</think>", "</", "think>", "<", "\n", "\n\n", "a", "b c" };
    const max_fragments = 16;
    var buf: [max_fragments * "</think>".len]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var len: usize = 0;
    for (0..r.uintLessThan(usize, max_fragments + 1)) |_| {
        const f = fragments[r.uintLessThan(usize, fragments.len)];
        @memcpy(buf[len..][0..f.len], f);
        len += f.len;
    }
    const content = buf[0..len];
    const reply = stripThink(content, qwen_think_text.close);
    try std.testing.expect(reply.ptr + reply.len == content.ptr + content.len);
    try std.testing.expect(std.mem.find(u8, reply, "</think>") == null);
    const dropped = content[0 .. content.len - reply.len];
    if (std.mem.find(u8, content, "</think>") == null) {
        try std.testing.expectEqual(@as(usize, 0), dropped.len);
    } else {
        try std.testing.expect(!std.mem.startsWith(u8, reply, "\n"));
        try std.testing.expect(std.mem.endsWith(u8, std.mem.trimEnd(u8, dropped, "\n"), "</think>"));
    }
}

test "stripThink property: the reply is the suffix after the last closing tag" {
    try prop.forAll(.{ .site = 0x7417_5EED, .cases = 2000 }, {}, u64, prop.genSeed(void), checkStripThink);
}

test "wrapMessagesAlloc drops the think block of an earlier qwen reply" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const reply = "<think>\nreason\n</think>\n\nA1";
    const msgs = [_]Message{
        .{ .role = .user, .content = "Q1" },
        .{ .role = .assistant, .content = reply },
        .{ .role = .user, .content = "Q2" },
    };
    try std.testing.expectEqualStrings(
        "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n" ++
            "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n",
        try wrapMessagesAlloc(arena, .{ .chatml_qwen = .on }, &msgs),
    );
    // A family with no thinking mode renders the reply unchanged.
    const smol = try wrapMessagesAlloc(arena, .chatml_smollm2, &msgs);
    try std.testing.expect(std.mem.find(u8, smol, reply) != null);
}

test "a kept pair renders the live user wrap of its place with the closed think tail" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // A pair equals the live turn of a server with thinking off, whatever the
    // mode of the server.
    const qwen_off: Kind = .{ .chatml_qwen = .off };
    for (all_kinds) |kind| {
        const live: Kind = if (kind == .chatml_qwen) qwen_off else kind;
        try std.testing.expectEqualStrings(try wrapAlloc(arena, live, "Q"), try wrapPairUserAlloc(arena, kind, .open_first, "Q"));
        try std.testing.expectEqualStrings(try wrapFirstTurnAlloc(arena, live, "Q"), try wrapPairUserAlloc(arena, kind, .first_after_system, "Q"));
        try std.testing.expectEqualStrings(try wrapContinuationAlloc(arena, live, "Q"), try wrapPairUserAlloc(arena, kind, .continuation, "Q"));
        for (std.enums.values(PairPlace)) |place| {
            const w = try wrapPairUserAlloc(arena, kind, place, "Q");
            try std.testing.expectEqual(std.mem.count(u8, w, "<think>"), std.mem.count(u8, w, "</think>"));
        }
    }
    try std.testing.expectEqualStrings(
        "<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        try wrapPairUserAlloc(arena, .{ .chatml_qwen = .on }, .continuation, "Q"),
    );
}
