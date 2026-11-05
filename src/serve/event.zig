//! The values that cross the serve layer's two internal boundaries. `Request`
//! flows reader-thread -> engine thread, and `OutEvent` flows engine thread ->
//! writer-thread. `Codec` carries the text<->token knowledge (chat
//! template + tokenizer + piece table) the scheduler needs without owning it.
//!
//! `OutEvent` is a fixed-size value. Token text, error detail, and model
//! name are inline bounded arrays, so the bounded output queues hold plain
//! copies and need no ownership rule. `Request` is the one value that owns
//! bytes across a thread hop. The reader dupes system-prompt, turn, and pair
//! text into the gpa, and the scheduler frees it after tokenizing.

const std = @import("std");
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const zmtp = @import("zmtp.zig");
const limits = @import("../core/limits.zig");
const chat = @import("../core/chat.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");

const log = std.log.scoped(.serve);

// -- types ------------------------------------------------------------------

/// One decoded client frame plus its origin, handed to the engine thread.
/// The request owns text payloads in the gpa. The scheduler frees them.
pub const Request = struct {
    conn: u32,
    request_id: u32,
    session_id: u32,
    body: Body,

    pub const Body = union(enum) {
        hello,
        session_open: struct { sampling: protocol.Sampling, system_prompt: []u8 },
        session_clone: struct { sampling: protocol.Sampling },
        generate: struct {
            max_tokens: u32,
            sampling: ?protocol.Sampling,
            text: []u8,
            /// Whether the turn opens inside the think block. Null takes the
            /// server-wide mode.
            think_open: ?bool = null,
            /// Think tokens before the forced close, 0 unbounded. Null takes
            /// the server value.
            think_budget: ?u32 = null,
            /// The seed of `sampling` restarts the sampling stream. False
            /// keeps the stream, and the seed then means nothing.
            reseed: bool = false,
        },
        /// The pair records of a `session_append`, as the decoder checked
        /// them, duped into the gpa.
        session_append: struct {
            n: u16,
            bytes: []u8,

            pub fn pairs(self: @This()) protocol.AppendPairs {
                return .{ .n = self.n, .bytes = self.bytes };
            }
        },
        session_keep: struct { ttl_ms: u32 },
        session_resume: struct { token: protocol.Token },
        cancel,
        session_close,
        stats,
        ping,
        shutdown,
        /// The server synthesizes this when a connection drops. The scheduler
        /// then sweeps the connection's sessions. This request never carries a
        /// reply.
        conn_closed,
        /// A writer task synthesizes this after draining a queue that had
        /// parked an event. The request wakes an idle engine loop, so the next
        /// tick retries the park. The scheduler itself treats it as a no-op.
        wakeup,
    };
};

/// One server frame for a connection's writer task. Its size is fixed by
/// construction. The writer converts it to a `protocol.ServerMsg`.
pub const OutEvent = struct {
    request_id: u32,
    session_id: u32,
    body: Body,

    pub const Body = union(enum) {
        hello_ack: struct {
            max_seq: u32,
            max_sessions: u16,
            max_batch: u16,
            vocab_size: u32,
            eos_id: u32,
            think_mode: protocol.ThinkMode,
            think_budget: u32,
            resume_ttl_max_ms: u32,
            recommended_think: protocol.Sampling,
            recommended_plain: protocol.Sampling,
            name_len: u8,
            name: [limits.max_model_name_bytes]u8,
        },
        session_opened: struct { kv_pos: u32 },
        token: struct {
            token_id: u32,
            channel: protocol.Channel,
            text_len: u16,
            text: [limits.max_token_text_bytes]u8,
        },
        done: struct {
            finish_reason: protocol.FinishReason,
            turn_prompt_tokens: u32,
            gen_tokens: u32,
            kv_pos: u32,
            prefill_ms: f32,
            decode_ms: f32,
            reasoning_tokens: u32,
        },
        err: struct { code: protocol.ErrorCode, detail_len: u8, detail: [64]u8 },
        stats_ack: struct {
            sessions_open: u16,
            sessions_decoding: u16,
            tokens_generated_total: u64,
            uptime_ms: u64,
        },
        pong,
        session_closed,
        shutdown_ack,
        appended: protocol.Appended,
        kept: struct { granted_ms: u32, token: protocol.Token },
        resumed: protocol.Resumed,
        /// The writer task is the single writer of the stream, so the reader
        /// hands it the PONG for a peer PING through the out-queue.
        zmtp_pong: struct { len: u8, context: [zmtp.max_ping_context_bytes]u8 },
        /// The writer task is the single writer of the stream, so the reader
        /// hands it the ERROR that refuses a handshake through the out-queue.
        zmtp_error: struct { len: u8, reason: [64]u8 },
    };

    pub fn token(
        request_id: u32,
        session_id: u32,
        id: u32,
        channel: protocol.Channel,
        piece: []const u8,
    ) OutEvent {
        comptime std.debug.assert(limits.max_token_text_bytes <= std.math.maxInt(u16));
        std.debug.assert(piece.len <= limits.max_token_text_bytes);
        var body: Body = .{
            .token = .{
                .token_id = id,
                .channel = channel,
                .text_len = @intCast(piece.len),
                .text = undefined,
            },
        };
        @memcpy(body.token.text[0..piece.len], piece);
        return .{ .request_id = request_id, .session_id = session_id, .body = body };
    }

    pub fn fail(
        request_id: u32,
        session_id: u32,
        code: protocol.ErrorCode,
        detail: []const u8,
    ) OutEvent {
        std.debug.assert(detail.len <= 64);

        var body: Body = .{
            .err = .{ .code = code, .detail_len = @intCast(detail.len), .detail = undefined },
        };
        @memcpy(body.err.detail[0..detail.len], detail);
        return .{ .request_id = request_id, .session_id = session_id, .body = body };
    }

    pub fn zmtpPong(context: []const u8) OutEvent {
        std.debug.assert(context.len <= zmtp.max_ping_context_bytes);

        var body: Body = .{ .zmtp_pong = .{ .len = @intCast(context.len), .context = undefined } };
        @memcpy(body.zmtp_pong.context[0..context.len], context);
        return .{ .request_id = 0, .session_id = 0, .body = body };
    }

    pub fn zmtpError(reason: []const u8) OutEvent {
        std.debug.assert(reason.len <= 64);

        var body: Body = .{ .zmtp_error = .{ .len = @intCast(reason.len), .reason = undefined } };
        @memcpy(body.zmtp_error.reason[0..reason.len], reason);
        return .{ .request_id = 0, .session_id = 0, .body = body };
    }
};

/// How a turn's text enters the context. `system` and `open_first` start the
/// sequence (BOS via the encoder). The other two continue one.
pub const TurnKind = enum { system, open_first, first_after_system, continuation };

/// The real text<->token codec over the loaded model's vocab and chat
/// template. The scheduler is generic over this shape so its tests can run a
/// trivial fake with no model on disk.
pub const Codec = struct {
    vocab: *const vocab_mod.Vocab,
    kind: chat.Kind,
    /// id -> displayable bytes, precomputed at server start (each <=
    /// `limits.max_token_text_bytes`, validated there).
    pieces: []const []const u8,
    eos_id: u32,

    /// Wrap `text` per the turn's position in the conversation and tokenize.
    /// `open` is the think switch that the scheduler resolved for the turn,
    /// so the tail of the wrap and the think state of the slot agree.
    /// Caller owns the returned ids (gpa).
    pub fn turnTokens(
        self: *const Codec,
        gpa: Allocator,
        turn: TurnKind,
        text: []const u8,
        open: bool,
    ) ![]u32 {
        const kind = renderKind(self.kind, open);
        const wrapped = switch (turn) {
            .system => try chat.wrapSystemAlloc(gpa, kind, text),
            .open_first => try chat.wrapAlloc(gpa, kind, text),
            .first_after_system => try chat.wrapFirstTurnAlloc(gpa, kind, text),
            .continuation => try chat.wrapContinuationAlloc(gpa, kind, text),
        };

        defer gpa.free(wrapped);

        const sequence_start = turn == .system or turn == .open_first;
        const ids = try bpe.encodeAlloc(self.vocab, gpa, wrapped, sequence_start);
        log.debug("encode: {t} turn, {d} B -> {d} tokens", .{ turn, text.len, ids.len });
        return ids;
    }

    /// Tokenize one kept (user, assistant) pair at `place`. The user wrap and
    /// the reply encode apart, as a live turn feeds the wrap and then samples
    /// the reply. A reply that re-encodes to its sampled ids therefore gives
    /// the ids of the live turn. Caller owns the ids (gpa).
    pub fn pairTokens(
        self: *const Codec,
        gpa: Allocator,
        place: chat.PairPlace,
        user: []const u8,
        assistant: []const u8,
    ) ![]u32 {
        const wrapped = try chat.wrapPairUserAlloc(gpa, self.kind, place, user);
        defer gpa.free(wrapped);

        const head = try bpe.encodeAlloc(self.vocab, gpa, wrapped, place == .open_first);
        defer gpa.free(head);

        const reply = try bpe.encodeAlloc(self.vocab, gpa, assistant, false);
        defer gpa.free(reply);

        const ids = try std.mem.concat(gpa, u32, &.{ head, reply });
        log.debug(
            "encode: {t} pair, {d}+{d} B -> {d}+{d} tokens",
            .{ place, user.len, assistant.len, head.len, reply.len },
        );
        return ids;
    }

    pub fn piece(self: *const Codec, id: u32) []const u8 {
        std.debug.assert(id < self.pieces.len);

        return self.pieces[id];
    }
};

// -- helpers ----------------------------------------------------------------

/// The kind that renders a turn with the think switch `open`. A family with
/// no thinking mode renders one way, and the scheduler refuses an open turn
/// on it before any render.
fn renderKind(kind: chat.Kind, open: bool) chat.Kind {
    const out = chat.resolveThinking(kind, if (open) .on else .off) catch kind;
    std.debug.assert(std.meta.activeTag(out) == std.meta.activeTag(kind));
    if (chat.thinkText(kind) != null) std.debug.assert(chat.thinkOpen(out) == open);

    return out;
}

// -- tests ------------------------------------------------------------------

test "a token event carries a piece at the full text bound" {
    const long = [_]u8{'x'} ** limits.max_token_text_bytes;
    const ev = OutEvent.token(1, 2, 3, .content, &long);
    try std.testing.expectEqual(@as(u16, limits.max_token_text_bytes), ev.body.token.text_len);
    try std.testing.expectEqualSlices(u8, &long, ev.body.token.text[0..ev.body.token.text_len]);
}

test "renderKind: the switch picks the think tail of the turn wrap" {
    const gpa = std.testing.allocator;
    const qwen_on: chat.Kind = .{ .chatml_qwen = .on };
    const qwen_off: chat.Kind = .{ .chatml_qwen = .off };
    for ([_]chat.Kind{ qwen_on, qwen_off }) |server| {
        for ([_]bool{ true, false }) |open| {
            const kind = renderKind(server, open);
            try std.testing.expectEqual(open, chat.thinkOpen(kind));
            const want = try chat.wrapFirstTurnAlloc(gpa, if (open) qwen_on else qwen_off, "hi");
            defer gpa.free(want);

            const got = try chat.wrapFirstTurnAlloc(gpa, kind, "hi");
            defer gpa.free(got);

            try std.testing.expectEqualStrings(want, got);
        }
    }

    // The open tail differs from the closed tail, so the switch is visible.
    const open_text = try chat.wrapContinuationAlloc(gpa, renderKind(qwen_off, true), "x");
    defer gpa.free(open_text);

    const closed_text = try chat.wrapContinuationAlloc(gpa, renderKind(qwen_on, false), "x");
    defer gpa.free(closed_text);

    try std.testing.expect(!std.mem.eql(u8, open_text, closed_text));
    // A family with no thinking mode renders one way.
    try std.testing.expectEqual(chat.Kind.chatml_smollm2, renderKind(.chatml_smollm2, false));
    try std.testing.expectEqual(chat.Kind.llama3, renderKind(.llama3, true));
}
