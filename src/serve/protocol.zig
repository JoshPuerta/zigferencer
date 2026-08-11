//! The wire format between the serve layer and a client, and the document a
//! client implementer reads. The frames are binary and little-endian
//! throughout, and they do not depend on a transport. A ZeroMQ message
//! carries one frame. The server speaks ZMTP 3.x with the NULL security
//! method, and a DEALER socket is the client.
//!
//! One frame = a 20-byte header + payload:
//!
//!     off  field        type  rule
//!     0    magic        u32   0x4647495A ("ZIGF" read as bytes)
//!     4    version      u16   3. Any other value is refused.
//!     6    msg_type     u16   known MsgType value
//!     8    request_id   u32   client-chosen, and echoed verbatim in every reply
//!     12   session_id   u32   0 = server scope, otherwise a live session id
//!     16   payload_len  u32   <= max_frame_payload_bytes
//!
//! This module is pure. It imports no I/O and it allocates nothing. Decoded
//! messages BORROW their text slices from the payload buffer, and the caller
//! copies what it keeps. Every decode checks every length, and a payload must
//! be consumed exactly. Trailing bytes are malformed. The socket is a trust
//! boundary, and these checks are its guards. A decode never panics on input.
//! It returns a typed error the connection layer maps to an `error` frame.
//!
//! One rule every client must implement: `token.text` is RAW BYTES. One token
//! can be a partial UTF-8 sequence. The concatenation of a turn's token texts
//! is well-formed. Buffer the bytes until they are displayable.
//!
//! `serve/README.md` names the client method and the terminal message of every
//! verb, and the ZMTP carrier under it.

const std = @import("std");
const limits = @import("../core/limits.zig");
const prop = @import("../core/prop.zig");

// -- types ------------------------------------------------------------------

pub const magic: u32 = 0x4647495A;
pub const protocol_version: u16 = 3;
pub const header_bytes: usize = 20;

/// Hard cap on a frame payload. A system prompt or turn beyond 64 KiB of UTF-8
/// is outside the serving use case. This is a boundary guard, not a growth
/// path. This one *is* a wire-format fact, so it stays with the framing. The
/// piece and model-name bounds are serve bounds and live in `core/limits.zig`.
pub const max_frame_payload_bytes: u32 = 64 * 1024;

/// The largest system prompt that one `session_open` frame carries. The
/// payload holds the sampling knobs, a u32 length, and the prompt.
pub const system_prompt_bytes_max: usize = max_frame_payload_bytes - Sampling.wire_bytes - 4;

comptime {
    const prompt = [_]u8{0} ** system_prompt_bytes_max;
    const open: ClientMsg = .{ .session_open = .{ .sampling = Sampling.greedy, .system_prompt = &prompt } };
    std.debug.assert(clientFrameLen(open) == header_bytes + max_frame_payload_bytes);
}

/// The detail texts of a `turn_too_long` refusal. They live with the wire
/// because a client reads the numbers back from the detail.
pub const turn_too_long_fmt = "turn of {d} tokens, {d} positions left";
pub const system_too_long_fmt = "system prompt of {d} tokens, window of {d} positions";

/// The largest frame the server encodes. A `token` payload is the largest:
/// 4 bytes of token id, 1 of channel, 2 of text length, and at most
/// `limits.max_token_text_bytes` of text. The frame is then 20 + 4 + 1 + 2 +
/// 256 = 283 bytes. A `hello_ack` payload holds at most 101 + 64 bytes, and an
/// `appended` payload at most 2 + 4 * `limits.max_append_pairs`. The server
/// caps an `err` detail at 64 bytes, so an `err` payload holds at most 4 + 64.
/// A `kept` frame is 40 bytes and a `resumed` frame is 58 bytes.
pub const max_server_frame_bytes: usize = header_bytes + 4 + 1 + 2 + limits.max_token_text_bytes;

/// The bytes of a lease token on the wire.
pub const token_bytes: usize = 16;

/// The bearer secret of a session lease. All zero bytes name no lease.
pub const Token = [token_bytes]u8;

/// The flag bits of a `generate` payload. Bit 2 is legal only with bit 1,
/// bit 4 only with bit 0, and bits 5 to 7 are refused.
const generate_flag_sampling: u8 = 1 << 0;
const generate_flag_think_switch: u8 = 1 << 1;
const generate_flag_think_open: u8 = 1 << 2;
const generate_flag_think_budget: u8 = 1 << 3;
const generate_flag_reseed: u8 = 1 << 4;
const generate_flags_known: u8 = 0x1F;

/// The bytes of a `generate` payload outside its text: `max_tokens`, the
/// flags, the sampling block, the budget, and the text length.
const generate_fixed_bytes_max: usize = 4 + 1 + Sampling.wire_bytes + 4 + 4;

/// The longest text that one `generate` frame carries with every optional
/// field present. A client refuses a longer text before it encodes.
pub const generate_text_bytes_max: usize = max_frame_payload_bytes - generate_fixed_bytes_max;

/// The pair record bytes of one `session_append` payload, after the u16 count.
pub const append_pairs_bytes_max: usize = max_frame_payload_bytes - 2;

/// The bytes of one pair record outside its two texts: the two u32 lengths.
pub const pair_overhead_bytes: usize = 8;

comptime {
    std.debug.assert(header_bytes + 2 + 4 * @as(usize, limits.max_append_pairs) <= max_server_frame_bytes);
    // A pair costs at least its two length fields, so one frame cannot name
    // more pairs than this.
    std.debug.assert(limits.max_append_pairs <= append_pairs_bytes_max / pair_overhead_bytes);
}

pub const MsgType = enum(u16) {
    hello = 0x0001,
    session_open = 0x0002,
    generate = 0x0003,
    cancel = 0x0004,
    session_close = 0x0005,
    stats = 0x0006,
    ping = 0x0007,
    shutdown = 0x0008,
    session_clone = 0x0009,
    session_append = 0x000A,
    session_keep = 0x000B,
    session_resume = 0x000C,

    hello_ack = 0x8001,
    session_opened = 0x8002,
    token = 0x8003,
    done = 0x8004,
    err = 0x8005,
    stats_ack = 0x8006,
    pong = 0x8007,
    session_closed = 0x8008,
    shutdown_ack = 0x8009,
    appended = 0x800A,
    kept = 0x800B,
    resumed = 0x800C,
};

pub const Header = struct {
    msg_type: MsgType,
    request_id: u32,
    session_id: u32,
    payload_len: u32,
};

/// Codes 1-4 are fatal to the connection (the server replies, flushes, and
/// closes). Codes 5-12 fail the request, and the connection lives.
pub const ErrorCode = enum(u16) {
    malformed_frame = 1,
    unsupported_version = 2,
    unknown_msg_type = 3,
    frame_too_large = 4,
    unknown_session = 5,
    sessions_exhausted = 6,
    session_busy = 7,
    turn_too_long = 8,
    connections_exhausted = 9,
    internal = 10,
    /// The request is valid but this server refuses it. The cause is a server
    /// policy, such as a `shutdown` that the operator did not allow, or a
    /// request field that the model or the engine does not support.
    forbidden = 11,
    /// A cancel stopped a `session_append` before its last pair. The rows
    /// already fed stay in the session, so the client closes it.
    cancelled = 12,
};

pub const FinishReason = enum(u8) {
    eos = 0,
    max_tokens = 1,
    context_full = 2,
    cancelled = 3,
};

/// Sampling knobs on the wire, 36 bytes:
///
///     off  field               type  rule
///     0    temperature         f32   finite, >= 0. 0 is greedy.
///     4    top_k               u32   <= limits.max_top_k. 0 is no cap.
///     8    top_p               f32   in (0, 1]
///     12   seed                u64   any
///     20   presence_penalty    f32   in [-2, 2]. 0 is the identity.
///     24   frequency_penalty   f32   in [-2, 2]. The server has no
///                                    frequency penalty yet. A request
///                                    with a nonzero value fails with
///                                    `forbidden`, and the connection
///                                    lives. The server never drops it.
///     28   repetition_penalty  f32   in (0, limits.max_repetition_penalty].
///                                    1 is the identity.
///     32   min_p               f32   in [0, 1]. 0 is the identity.
///
/// The fields mirror `runtime/sampler.Params` plus the explicit seed.
pub const Sampling = struct {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
    presence_penalty: f32 = 0,
    frequency_penalty: f32 = 0,
    repetition_penalty: f32 = 1,
    min_p: f32 = 0,

    pub const wire_bytes: usize = 36;
    pub const greedy: Sampling = .{ .temperature = 0, .top_k = 0, .top_p = 1, .seed = 0 };
};

/// The last finished request of a resumed session.
pub const LastDone = struct {
    request_id: u32,
    finish: FinishReason,
    gen_tokens: u32,
    reasoning_tokens: u32,
};

/// The reply to a `session_resume`, 38 bytes of payload:
///
///     off  field             type     rule
///     0    token             [16]u8   the new token, not all zero
///     16   kv_pos            u32      the final position of the session
///     20   turns_done        u32      the finished turns and kept pairs
///     24   has_last          u8       0 or 1
///     25   last_request_id   u32      0 when has_last is 0
///     29   finish            u8       a FinishReason, 0 when has_last is 0
///     30   gen_tokens        u32      0 when has_last is 0
///     34   reasoning_tokens  u32      0 when has_last is 0
pub const Resumed = struct {
    token: Token,
    kv_pos: u32,
    turns_done: u32,
    last: ?LastDone,

    pub const wire_bytes: usize = token_bytes + 4 + 4 + 1 + 4 + 1 + 4 + 4;
};

/// What one streamed token is. A tag keeps its text, so a client that ignores
/// the channel prints the same bytes.
pub const Channel = enum(u8) {
    content = 0,
    reasoning = 1,
    open_tag = 2,
    close_tag = 3,
};

/// The think mode of the server. `none` is a model with no think tags.
pub const ThinkMode = enum(u8) {
    none = 0,
    off = 1,
    on = 2,
};

/// One kept (user, assistant) pair of a `session_append`.
pub const Pair = struct { user: []const u8, assistant: []const u8 };

/// The pairs of a `session_append` frame. `bytes` holds `n` records of
/// `u32 ulen, user, u32 alen, assistant`. The decoder checks every length
/// before it makes this value, so `iterator` trusts them.
pub const AppendPairs = struct {
    n: u16,
    bytes: []const u8,

    pub fn iterator(self: AppendPairs) PairIterator {
        return .{ .bytes = self.bytes, .left = self.n };
    }
};

pub const PairIterator = struct {
    bytes: []const u8,
    i: usize = 0,
    left: u16,

    pub fn next(it: *PairIterator) ?Pair {
        if (it.left == 0) {
            std.debug.assert(it.i == it.bytes.len);
            return null;
        }
        it.left -= 1;
        const user = it.field();
        const assistant = it.field();
        return .{ .user = user, .assistant = assistant };
    }

    fn field(it: *PairIterator) []const u8 {
        std.debug.assert(it.i + 4 <= it.bytes.len);
        const len = std.mem.readInt(u32, it.bytes[it.i..][0..4], .little);
        it.i += 4;
        std.debug.assert(it.i + len <= it.bytes.len);
        defer it.i += len;
        return it.bytes[it.i..][0..len];
    }
};

/// The reply to a `session_append`: the position after each pair, in order.
pub const Appended = struct {
    n: u16,
    kv_pos: [limits.max_append_pairs]u32 = @splat(0),

    pub fn positions(self: *const Appended) []const u32 {
        std.debug.assert(self.n >= 1);
        std.debug.assert(self.n <= limits.max_append_pairs);
        return self.kv_pos[0..self.n];
    }
};

pub const ClientMsg = union(enum) {
    hello,
    session_open: struct { sampling: Sampling, system_prompt: []const u8 },
    /// `max_tokens`, the flags byte, the sampling block when bit 0 is set,
    /// a u32 think budget when bit 3 is set, then the text. Bit 1 says that
    /// the request carries a think switch, and bit 2 is its value. A budget
    /// of 0 is unbounded, the `hello_ack` encoding. A switch that opens the
    /// block or a budget on a model with no think tags fails the request.
    /// Bit 4 restarts the sampling stream of the session at the seed of the
    /// sampling block. Without bit 4 the server ignores that seed, and the
    /// stream continues.
    generate: struct {
        max_tokens: u32,
        sampling: ?Sampling,
        text: []const u8,
        think_open: ?bool = null,
        think_budget: ?u32 = null,
        reseed: bool = false,
    },
    cancel,
    session_close,
    stats,
    ping,
    shutdown,
    /// Fork the session in `header.session_id`. The new session shares its
    /// whole context (system prompt + conversation so far) but samples with
    /// its own sampling + seed. One prefilled prefix serves many divergent
    /// sessions.
    /// Reply is `session_opened` whose header carries the NEW session's id.
    /// Errors: `unknown_session`, `session_busy` (source mid-turn),
    /// `sessions_exhausted`, `internal` (backend cannot copy slot state).
    session_clone: struct { sampling: Sampling },
    /// Feed kept (user, assistant) pairs into an idle session. The server
    /// renders each pair with the wraps of a live turn. Reply is `appended`.
    /// Errors: `unknown_session`, `session_busy`, `turn_too_long` (no pair
    /// fed), `cancelled`, `internal`.
    session_append: AppendPairs,
    /// Keep the session in `header.session_id` past a drop of this
    /// connection for at most `ttl_ms`. A `ttl_ms` of 0 drops the lease.
    /// Reply is `kept`. Errors: `unknown_session`, `forbidden` (resumption
    /// off and `ttl_ms` > 0), `session_busy` (a resume in flight),
    /// `internal` (no secure random source).
    session_keep: struct { ttl_ms: u32 },
    /// Attach the session of `token` to this connection. The header session
    /// id must be 0, and any other id is malformed. Reply is `resumed`, sent when the session is idle. Errors:
    /// `unknown_session` (no such lease), `sessions_exhausted`, `internal`.
    session_resume: struct { token: Token },
};

pub const ServerMsg = union(enum) {
    hello_ack: struct {
        protocol_version: u16,
        max_seq: u32,
        max_sessions: u16,
        max_batch: u16,
        vocab_size: u32,
        eos_id: u32,
        think_mode: ThinkMode,
        /// Think tokens before a forced close. 0 is unbounded.
        think_budget: u32,
        /// The longest lease a `session_keep` gets. 0 is resumption off.
        resume_ttl_max_ms: u32,
        /// The sampling the model card recommends for a turn inside the
        /// think block and for a turn outside it, or greedy for a model with
        /// no entry.
        recommended_think: Sampling,
        recommended_plain: Sampling,
        model_name: []const u8,
    },
    session_opened: struct { kv_pos: u32 },
    token: struct { token_id: u32, channel: Channel, text: []const u8 },
    done: struct {
        finish_reason: FinishReason,
        turn_prompt_tokens: u32,
        gen_tokens: u32,
        kv_pos: u32,
        prefill_ms: f32,
        decode_ms: f32,
        /// The tokens inside the think block, both tags excluded.
        reasoning_tokens: u32,
    },
    err: struct { code: ErrorCode, detail: []const u8 },
    stats_ack: struct {
        sessions_open: u16,
        sessions_decoding: u16,
        tokens_generated_total: u64,
        uptime_ms: u64,
    },
    pong,
    session_closed,
    shutdown_ack,
    appended: Appended,
    /// The granted time to live, then the token. A grant of 0 drops the
    /// lease and carries 16 zero bytes. A positive grant carries a token
    /// that is not all zero. The frame is 40 bytes either way.
    kept: struct { granted_ms: u32, token: Token },
    resumed: Resumed,
};

pub const DecodeError = error{
    BadMagic,
    BadVersion,
    UnknownMsgType,
    FrameTooLarge,
    PayloadMalformed,
};

/// The detail of the refusal of a nonzero frequency penalty. The decoder
/// accepts the value in range, and the server refuses the request.
pub const frequency_unsupported_detail = "frequency_penalty is not supported";
/// The OpenAI range of the frequency penalty is the presence range, [-2, 2].
pub const frequency_penalty_max: f32 = limits.max_presence_penalty;

/// Bounds-checked little-endian reader over a payload. A `take` past the end
/// returns `PayloadMalformed`. `expectDone` rejects trailing bytes, so a frame
/// parses exactly or not at all.
const Cursor = struct {
    buf: []const u8,
    i: usize = 0,

    fn take(c: *Cursor, comptime T: type) DecodeError!T {
        const n = @sizeOf(T);
        if (c.i + n > c.buf.len) return error.PayloadMalformed;
        defer c.i += n;
        return switch (T) {
            f32 => @bitCast(std.mem.readInt(u32, c.buf[c.i..][0..4], .little)),
            else => std.mem.readInt(T, c.buf[c.i..][0..n], .little),
        };
    }

    fn takeSlice(c: *Cursor, len: usize) DecodeError![]const u8 {
        if (c.i + len > c.buf.len) return error.PayloadMalformed;
        defer c.i += len;
        return c.buf[c.i..][0..len];
    }

    fn expectDone(c: *const Cursor) DecodeError!void {
        if (c.i != c.buf.len) return error.PayloadMalformed;
    }
};

/// Asserting little-endian writer. The server owns every encode buffer and
/// sizes it with `clientFrameLen` or `serverFrameLen`, so an overflow is a
/// programmer error and not a runtime one.
const Writer = struct {
    buf: []u8,
    i: usize = 0,

    fn put(w: *Writer, comptime T: type, v: T) void {
        const n = @sizeOf(T);
        std.debug.assert(w.i + n <= w.buf.len);
        defer w.i += n;
        switch (T) {
            f32 => std.mem.writeInt(u32, w.buf[w.i..][0..4], @bitCast(v), .little),
            else => std.mem.writeInt(T, w.buf[w.i..][0..n], v, .little),
        }
    }

    fn putSlice(w: *Writer, bytes: []const u8) void {
        std.debug.assert(w.i + bytes.len <= w.buf.len);
        @memcpy(w.buf[w.i..][0..bytes.len], bytes);
        w.i += bytes.len;
    }
};

// -- api --------------------------------------------------------------------

pub fn decodeHeader(bytes: *const [header_bytes]u8) DecodeError!Header {
    if (std.mem.readInt(u32, bytes[0..4], .little) != magic) return error.BadMagic;
    if (std.mem.readInt(u16, bytes[4..6], .little) != protocol_version) return error.BadVersion;
    const raw_type = std.mem.readInt(u16, bytes[6..8], .little);
    const msg_type = std.enums.fromInt(MsgType, raw_type) orelse return error.UnknownMsgType;
    const payload_len = std.mem.readInt(u32, bytes[16..20], .little);
    if (payload_len > max_frame_payload_bytes) return error.FrameTooLarge;
    const session_id = std.mem.readInt(u32, bytes[12..16], .little);
    // A resume names its session by the token alone, so a header id is a
    // client fault and not a second key.
    if (msg_type == .session_resume and session_id != 0) return error.PayloadMalformed;
    return .{
        .msg_type = msg_type,
        .request_id = std.mem.readInt(u32, bytes[8..12], .little),
        .session_id = session_id,
        .payload_len = payload_len,
    };
}

fn encodeHeader(h: Header, out: *[header_bytes]u8) void {
    std.debug.assert(h.payload_len <= max_frame_payload_bytes);
    std.mem.writeInt(u32, out[0..4], magic, .little);
    std.mem.writeInt(u16, out[4..6], protocol_version, .little);
    std.mem.writeInt(u16, out[6..8], @intFromEnum(h.msg_type), .little);
    std.mem.writeInt(u32, out[8..12], h.request_id, .little);
    std.mem.writeInt(u32, out[12..16], h.session_id, .little);
    std.mem.writeInt(u32, out[16..20], h.payload_len, .little);
}

/// Decode a client-direction payload. Slices borrow from `payload`.
pub fn decodeClientPayload(t: MsgType, payload: []const u8) DecodeError!ClientMsg {
    var c: Cursor = .{ .buf = payload };
    const msg: ClientMsg = switch (t) {
        .hello => .hello,
        .cancel => .cancel,
        .session_close => .session_close,
        .stats => .stats,
        .ping => .ping,
        .shutdown => .shutdown,
        .session_open => .{ .session_open = .{
            .sampling = try takeSampling(&c),
            .system_prompt = try c.takeSlice(try c.take(u32)),
        } },
        .session_clone => .{ .session_clone = .{ .sampling = try takeSampling(&c) } },
        .session_append => .{ .session_append = try takeAppendPairs(&c) },
        .session_keep => .{ .session_keep = .{ .ttl_ms = try c.take(u32) } },
        .session_resume => blk: {
            const token = try takeToken(&c);
            if (std.mem.allEqual(u8, &token, 0)) return error.PayloadMalformed;
            break :blk .{ .session_resume = .{ .token = token } };
        },
        .generate => blk: {
            const max_tokens = try c.take(u32);
            if (max_tokens == 0) return error.PayloadMalformed;
            const flags = try c.take(u8);
            if (flags & ~generate_flags_known != 0) return error.PayloadMalformed;
            const has_switch = flags & generate_flag_think_switch != 0;
            const open = flags & generate_flag_think_open != 0;
            if (open and !has_switch) return error.PayloadMalformed;
            const reseed = flags & generate_flag_reseed != 0;
            if (reseed and flags & generate_flag_sampling == 0) return error.PayloadMalformed;
            const sampling: ?Sampling = if (flags & generate_flag_sampling != 0) try takeSampling(&c) else null;
            const budget: ?u32 = if (flags & generate_flag_think_budget != 0) try c.take(u32) else null;
            if (budget) |b| {
                if (b > limits.max_model_ctx) return error.PayloadMalformed;
            }
            const text = try c.takeSlice(try c.take(u32));
            break :blk .{ .generate = .{
                .max_tokens = max_tokens,
                .sampling = sampling,
                .text = text,
                .think_open = if (has_switch) open else null,
                .think_budget = budget,
                .reseed = reseed,
            } };
        },
        else => return error.UnknownMsgType, // a server-direction type on the client path
    };
    try c.expectDone();
    return msg;
}

/// Decode a server-direction payload (the reference client and the round-trip
/// tests). Slices borrow from `payload`.
pub fn decodeServerPayload(t: MsgType, payload: []const u8) DecodeError!ServerMsg {
    var c: Cursor = .{ .buf = payload };
    const msg: ServerMsg = switch (t) {
        .pong => .pong,
        .session_closed => .session_closed,
        .shutdown_ack => .shutdown_ack,
        .hello_ack => blk: {
            const ver = try c.take(u16);
            const max_seq = try c.take(u32);
            const max_sessions = try c.take(u16);
            const max_batch = try c.take(u16);
            const vocab_size = try c.take(u32);
            const eos_id = try c.take(u32);
            const think_mode = std.enums.fromInt(ThinkMode, try c.take(u8)) orelse return error.PayloadMalformed;
            const think_budget = try c.take(u32);
            const resume_ttl_max_ms = try c.take(u32);
            if (resume_ttl_max_ms > limits.max_resume_ttl_ms) return error.PayloadMalformed;
            const recommended_think = try takeSampling(&c);
            const recommended_plain = try takeSampling(&c);
            const name = try c.takeSlice(try c.take(u16));
            if (name.len > limits.max_model_name_bytes) return error.PayloadMalformed;
            break :blk .{ .hello_ack = .{
                .protocol_version = ver,
                .max_seq = max_seq,
                .max_sessions = max_sessions,
                .max_batch = max_batch,
                .vocab_size = vocab_size,
                .eos_id = eos_id,
                .think_mode = think_mode,
                .think_budget = think_budget,
                .resume_ttl_max_ms = resume_ttl_max_ms,
                .recommended_think = recommended_think,
                .recommended_plain = recommended_plain,
                .model_name = name,
            } };
        },
        .session_opened => .{ .session_opened = .{ .kv_pos = try c.take(u32) } },
        .token => blk: {
            const id = try c.take(u32);
            const channel = std.enums.fromInt(Channel, try c.take(u8)) orelse return error.PayloadMalformed;
            const text = try c.takeSlice(try c.take(u16));
            if (text.len > limits.max_token_text_bytes) return error.PayloadMalformed;
            break :blk .{ .token = .{ .token_id = id, .channel = channel, .text = text } };
        },
        .done => .{ .done = .{
            .finish_reason = std.enums.fromInt(FinishReason, try c.take(u8)) orelse return error.PayloadMalformed,
            .turn_prompt_tokens = try c.take(u32),
            .gen_tokens = try c.take(u32),
            .kv_pos = try c.take(u32),
            .prefill_ms = try c.take(f32),
            .decode_ms = try c.take(f32),
            .reasoning_tokens = try c.take(u32),
        } },
        .err => blk: {
            const code = std.enums.fromInt(ErrorCode, try c.take(u16)) orelse return error.PayloadMalformed;
            const detail = try c.takeSlice(try c.take(u16));
            break :blk .{ .err = .{ .code = code, .detail = detail } };
        },
        .stats_ack => .{ .stats_ack = .{
            .sessions_open = try c.take(u16),
            .sessions_decoding = try c.take(u16),
            .tokens_generated_total = try c.take(u64),
            .uptime_ms = try c.take(u64),
        } },
        .appended => .{ .appended = try takeAppended(&c) },
        .kept => blk: {
            const granted = try c.take(u32);
            const token = try takeToken(&c);
            if (granted > limits.max_resume_ttl_ms) return error.PayloadMalformed;
            if ((granted == 0) != std.mem.allEqual(u8, &token, 0)) return error.PayloadMalformed;
            break :blk .{ .kept = .{ .granted_ms = granted, .token = token } };
        },
        .resumed => .{ .resumed = try takeResumed(&c) },
        else => return error.UnknownMsgType, // a client-direction type on the server path
    };
    try c.expectDone();
    return msg;
}

/// Encode header + payload into `out`. Returns the frame's total byte length.
/// `out` must hold the whole frame, which an assert checks. Size it with
/// `clientFrameLen`.
pub fn encodeClientMsg(msg: ClientMsg, request_id: u32, session_id: u32, out: []u8) usize {
    std.debug.assert(out.len >= clientFrameLen(msg));
    var w: Writer = .{ .buf = out[header_bytes..] };
    switch (msg) {
        .hello, .cancel, .session_close, .stats, .ping, .shutdown => {},
        .session_open => |m| {
            putSampling(&w, m.sampling);
            w.put(u32, @intCast(m.system_prompt.len));
            w.putSlice(m.system_prompt);
        },
        .session_clone => |m| putSampling(&w, m.sampling),
        .session_append => |m| {
            std.debug.assert(m.n >= 1);
            std.debug.assert(m.n <= limits.max_append_pairs);
            w.put(u16, m.n);
            w.putSlice(m.bytes);
        },
        .generate => |m| {
            std.debug.assert(m.max_tokens > 0);
            if (m.think_budget) |b| std.debug.assert(b <= limits.max_model_ctx);
            if (m.reseed) std.debug.assert(m.sampling != null);
            std.debug.assert(m.text.len <= generate_text_bytes_max);
            w.put(u32, m.max_tokens);
            w.put(u8, generateFlags(m.sampling != null, m.think_open, m.think_budget != null, m.reseed));
            if (m.sampling) |s| putSampling(&w, s);
            if (m.think_budget) |b| w.put(u32, b);
            w.put(u32, @intCast(m.text.len));
            w.putSlice(m.text);
        },
        .session_keep => |m| w.put(u32, m.ttl_ms),
        .session_resume => |m| {
            std.debug.assert(!std.mem.allEqual(u8, &m.token, 0));
            w.putSlice(&m.token);
        },
    }
    encodeHeader(.{
        .msg_type = clientMsgType(msg),
        .request_id = request_id,
        .session_id = session_id,
        .payload_len = @intCast(w.i),
    }, out[0..header_bytes]);
    return header_bytes + w.i;
}

pub fn encodeServerMsg(msg: ServerMsg, request_id: u32, session_id: u32, out: []u8) usize {
    std.debug.assert(out.len >= serverFrameLen(msg));
    var w: Writer = .{ .buf = out[header_bytes..] };
    switch (msg) {
        .pong, .session_closed, .shutdown_ack => {},
        .hello_ack => |m| {
            std.debug.assert(m.model_name.len <= limits.max_model_name_bytes);
            w.put(u16, m.protocol_version);
            w.put(u32, m.max_seq);
            w.put(u16, m.max_sessions);
            w.put(u16, m.max_batch);
            w.put(u32, m.vocab_size);
            w.put(u32, m.eos_id);
            w.put(u8, @intFromEnum(m.think_mode));
            w.put(u32, m.think_budget);
            std.debug.assert(m.resume_ttl_max_ms <= limits.max_resume_ttl_ms);
            w.put(u32, m.resume_ttl_max_ms);
            putSampling(&w, m.recommended_think);
            putSampling(&w, m.recommended_plain);
            w.put(u16, @intCast(m.model_name.len));
            w.putSlice(m.model_name);
        },
        .session_opened => |m| w.put(u32, m.kv_pos),
        .token => |m| {
            std.debug.assert(m.text.len <= limits.max_token_text_bytes);
            w.put(u32, m.token_id);
            w.put(u8, @intFromEnum(m.channel));
            w.put(u16, @intCast(m.text.len));
            w.putSlice(m.text);
        },
        .done => |m| {
            w.put(u8, @intFromEnum(m.finish_reason));
            w.put(u32, m.turn_prompt_tokens);
            w.put(u32, m.gen_tokens);
            w.put(u32, m.kv_pos);
            w.put(f32, m.prefill_ms);
            w.put(f32, m.decode_ms);
            w.put(u32, m.reasoning_tokens);
        },
        .err => |m| {
            w.put(u16, @intFromEnum(m.code));
            w.put(u16, @intCast(m.detail.len));
            w.putSlice(m.detail);
        },
        .stats_ack => |m| {
            w.put(u16, m.sessions_open);
            w.put(u16, m.sessions_decoding);
            w.put(u64, m.tokens_generated_total);
            w.put(u64, m.uptime_ms);
        },
        .appended => |*m| {
            const kv = m.positions();
            w.put(u16, m.n);
            for (kv, 0..) |pos, i| {
                if (i > 0) std.debug.assert(kv[i - 1] <= pos);
                w.put(u32, pos);
            }
        },
        .kept => |m| {
            std.debug.assert(m.granted_ms <= limits.max_resume_ttl_ms);
            std.debug.assert((m.granted_ms == 0) == std.mem.allEqual(u8, &m.token, 0));
            w.put(u32, m.granted_ms);
            w.putSlice(&m.token);
        },
        .resumed => |m| putResumed(&w, m),
    }
    encodeHeader(.{
        .msg_type = serverMsgType(msg),
        .request_id = request_id,
        .session_id = session_id,
        .payload_len = @intCast(w.i),
    }, out[0..header_bytes]);
    return header_bytes + w.i;
}

fn clientFrameLen(msg: ClientMsg) usize {
    return header_bytes + switch (msg) {
        .hello, .cancel, .session_close, .stats, .ping, .shutdown => 0,
        .session_open => |m| Sampling.wire_bytes + 4 + m.system_prompt.len,
        .session_clone => Sampling.wire_bytes,
        .session_append => |m| 2 + m.bytes.len,
        .generate => |m| 4 + 1 + (if (m.sampling != null) Sampling.wire_bytes else 0) +
            (if (m.think_budget != null) @as(usize, 4) else 0) + 4 + m.text.len,
        .session_keep => 4,
        .session_resume => token_bytes,
    };
}

fn serverFrameLen(msg: ServerMsg) usize {
    return header_bytes + switch (msg) {
        .pong, .session_closed, .shutdown_ack => 0,
        .hello_ack => |m| hello_ack_fixed_bytes + m.model_name.len,
        .session_opened => 4,
        .token => |m| 4 + 1 + 2 + m.text.len,
        .done => 1 + 4 + 4 + 4 + 4 + 4 + 4,
        .err => |m| 2 + 2 + m.detail.len,
        .stats_ack => 2 + 2 + 8 + 8,
        .appended => |m| 2 + 4 * @as(usize, m.n),
        .kept => 4 + token_bytes,
        .resumed => Resumed.wire_bytes,
    };
}

/// The `hello_ack` payload bytes outside the model name.
const hello_ack_fixed_bytes: usize = 2 + 4 + 2 + 2 + 4 + 4 + 1 + 4 + 4 + 2 * Sampling.wire_bytes + 2;

comptime {
    // Every server frame fits the writer frame, at its largest form.
    const name = [_]u8{'m'} ** limits.max_model_name_bytes;
    const hello: ServerMsg = .{ .hello_ack = .{
        .protocol_version = protocol_version,
        .max_seq = 0,
        .max_sessions = 0,
        .max_batch = 0,
        .vocab_size = 0,
        .eos_id = 0,
        .think_mode = .none,
        .think_budget = 0,
        .resume_ttl_max_ms = 0,
        .recommended_think = Sampling.greedy,
        .recommended_plain = Sampling.greedy,
        .model_name = &name,
    } };
    std.debug.assert(serverFrameLen(hello) == header_bytes + 165);
    std.debug.assert(serverFrameLen(hello) <= max_server_frame_bytes);
    const kept: ServerMsg = .{ .kept = .{ .granted_ms = 0, .token = @splat(0) } };
    std.debug.assert(serverFrameLen(kept) == 40);
    const resumed: ServerMsg = .{ .resumed = .{ .token = @splat(1), .kv_pos = 0, .turns_done = 0, .last = null } };
    std.debug.assert(serverFrameLen(resumed) == 58);
    std.debug.assert(serverFrameLen(resumed) <= max_server_frame_bytes);
    // Every client frame outside the text fits the payload bound.
    const keep: ClientMsg = .{ .session_keep = .{ .ttl_ms = 0 } };
    std.debug.assert(clientFrameLen(keep) == header_bytes + 4);
    const resume_msg: ClientMsg = .{ .session_resume = .{ .token = @splat(1) } };
    std.debug.assert(clientFrameLen(resume_msg) == header_bytes + token_bytes);
    const gen: ClientMsg = .{ .generate = .{ .max_tokens = 1, .sampling = Sampling.greedy, .text = "", .think_open = true, .think_budget = 0, .reseed = true } };
    std.debug.assert(clientFrameLen(gen) == header_bytes + generate_fixed_bytes_max);
    // The longest text with every optional field fills the payload bound.
    std.debug.assert(clientFrameLen(gen) + generate_text_bytes_max == header_bytes + max_frame_payload_bytes);
}

/// The bytes that `pairs` take inside a `session_append` payload, after `n`.
pub fn appendPairsBytes(pairs: []const Pair) usize {
    var total: usize = 0;
    for (pairs) |p| total += pair_overhead_bytes + p.user.len + p.assistant.len;
    return total;
}

/// Write the pair records of `pairs` into `out` and return the frame value.
/// `out` must hold `appendPairsBytes(pairs)`.
pub fn encodeAppendPairs(pairs: []const Pair, out: []u8) AppendPairs {
    std.debug.assert(pairs.len >= 1);
    std.debug.assert(pairs.len <= limits.max_append_pairs);
    const len = appendPairsBytes(pairs);
    std.debug.assert(out.len >= len);
    var w: Writer = .{ .buf = out[0..len] };
    for (pairs) |p| {
        w.put(u32, @intCast(p.user.len));
        w.putSlice(p.user);
        w.put(u32, @intCast(p.assistant.len));
        w.putSlice(p.assistant);
    }
    std.debug.assert(w.i == len);
    return .{ .n = @intCast(pairs.len), .bytes = out[0..len] };
}

// -- helpers ----------------------------------------------------------------

fn clientMsgType(msg: ClientMsg) MsgType {
    return switch (msg) {
        .hello => .hello,
        .session_open => .session_open,
        .generate => .generate,
        .cancel => .cancel,
        .session_close => .session_close,
        .stats => .stats,
        .ping => .ping,
        .shutdown => .shutdown,
        .session_clone => .session_clone,
        .session_append => .session_append,
        .session_keep => .session_keep,
        .session_resume => .session_resume,
    };
}

fn serverMsgType(msg: ServerMsg) MsgType {
    return switch (msg) {
        .hello_ack => .hello_ack,
        .session_opened => .session_opened,
        .token => .token,
        .done => .done,
        .err => .err,
        .stats_ack => .stats_ack,
        .pong => .pong,
        .session_closed => .session_closed,
        .shutdown_ack => .shutdown_ack,
        .appended => .appended,
        .kept => .kept,
        .resumed => .resumed,
    };
}

fn takeSampling(c: *Cursor) DecodeError!Sampling {
    const s: Sampling = .{
        .temperature = try c.take(f32),
        .top_k = try c.take(u32),
        .top_p = try c.take(f32),
        .seed = try c.take(u64),
        .presence_penalty = try c.take(f32),
        .frequency_penalty = try c.take(f32),
        .repetition_penalty = try c.take(f32),
        .min_p = try c.take(f32),
    };
    // Value guards live here, at the trust boundary, so the sampler's asserts
    // can never fire on remote input. A NaN fails every comparison, so each
    // guard reads as the range that it accepts.
    if (!std.math.isFinite(s.temperature) or s.temperature < 0) return error.PayloadMalformed;
    if (!std.math.isFinite(s.top_p) or s.top_p <= 0 or s.top_p > 1) return error.PayloadMalformed;
    if (s.top_k > limits.max_top_k) return error.PayloadMalformed;
    if (!std.math.isFinite(s.presence_penalty) or @abs(s.presence_penalty) > limits.max_presence_penalty) return error.PayloadMalformed;
    if (!(s.repetition_penalty > 0 and s.repetition_penalty <= limits.max_repetition_penalty)) return error.PayloadMalformed;
    if (!(s.min_p >= 0 and s.min_p <= 1)) return error.PayloadMalformed;
    if (!(@abs(s.frequency_penalty) <= frequency_penalty_max)) return error.PayloadMalformed;
    return s;
}

fn takeToken(c: *Cursor) DecodeError!Token {
    return (try c.takeSlice(token_bytes))[0..token_bytes].*;
}

/// The token must not be all zero, and a reply with no last request holds
/// zeros in its four fields, so one reply has one encoding.
fn takeResumed(c: *Cursor) DecodeError!Resumed {
    const token = try takeToken(c);
    if (std.mem.allEqual(u8, &token, 0)) return error.PayloadMalformed;
    const kv_pos = try c.take(u32);
    const turns_done = try c.take(u32);
    const has_last = try c.take(u8);
    if (has_last > 1) return error.PayloadMalformed;
    const request_id = try c.take(u32);
    const finish_raw = try c.take(u8);
    const gen_tokens = try c.take(u32);
    const reasoning_tokens = try c.take(u32);
    if (has_last == 0) {
        if (request_id != 0 or finish_raw != 0 or gen_tokens != 0 or reasoning_tokens != 0) return error.PayloadMalformed;
        return .{ .token = token, .kv_pos = kv_pos, .turns_done = turns_done, .last = null };
    }
    const finish = std.enums.fromInt(FinishReason, finish_raw) orelse return error.PayloadMalformed;
    return .{ .token = token, .kv_pos = kv_pos, .turns_done = turns_done, .last = .{
        .request_id = request_id,
        .finish = finish,
        .gen_tokens = gen_tokens,
        .reasoning_tokens = reasoning_tokens,
    } };
}

fn putResumed(w: *Writer, r: Resumed) void {
    std.debug.assert(!std.mem.allEqual(u8, &r.token, 0));
    const start = w.i;
    w.putSlice(&r.token);
    w.put(u32, r.kv_pos);
    w.put(u32, r.turns_done);
    const last: LastDone = r.last orelse .{ .request_id = 0, .finish = .eos, .gen_tokens = 0, .reasoning_tokens = 0 };
    w.put(u8, @intFromBool(r.last != null));
    w.put(u32, last.request_id);
    w.put(u8, @intFromEnum(last.finish));
    w.put(u32, last.gen_tokens);
    w.put(u32, last.reasoning_tokens);
    std.debug.assert(w.i - start == Resumed.wire_bytes);
}

/// The flags byte of a `generate`. A switch value of null sets neither
/// think bit.
fn generateFlags(sampling: bool, think_open: ?bool, budget: bool, reseed: bool) u8 {
    var flags: u8 = 0;
    if (sampling) flags |= generate_flag_sampling;
    if (reseed) flags |= generate_flag_reseed;
    if (think_open) |open| {
        flags |= generate_flag_think_switch;
        if (open) flags |= generate_flag_think_open;
    }
    if (budget) flags |= generate_flag_think_budget;
    std.debug.assert(flags & ~generate_flags_known == 0);
    return flags;
}

/// The pair count, then each record. Every length must fit the payload, and
/// the payload must end at the last record.
fn takeAppendPairs(c: *Cursor) DecodeError!AppendPairs {
    const n = try c.take(u16);
    if (n == 0 or n > limits.max_append_pairs) return error.PayloadMalformed;
    const start = c.i;
    for (0..n) |_| {
        _ = try c.takeSlice(try c.take(u32));
        _ = try c.takeSlice(try c.take(u32));
    }
    return .{ .n = n, .bytes = c.buf[start..c.i] };
}

/// The positions must not decrease, because each pair only adds rows.
fn takeAppended(c: *Cursor) DecodeError!Appended {
    const n = try c.take(u16);
    if (n == 0 or n > limits.max_append_pairs) return error.PayloadMalformed;
    var a: Appended = .{ .n = n };
    for (a.kv_pos[0..n], 0..) |*pos, i| {
        pos.* = try c.take(u32);
        if (i > 0 and a.kv_pos[i - 1] > pos.*) return error.PayloadMalformed;
    }
    return a;
}

fn putSampling(w: *Writer, s: Sampling) void {
    w.put(f32, s.temperature);
    w.put(u32, s.top_k);
    w.put(f32, s.top_p);
    w.put(u64, s.seed);
    w.put(f32, s.presence_penalty);
    w.put(f32, s.frequency_penalty);
    w.put(f32, s.repetition_penalty);
    w.put(f32, s.min_p);
}

// -- tests ------------------------------------------------------------------

fn roundTripClient(msg: ClientMsg, request_id: u32, session_id: u32) !void {
    var buf: [max_frame_payload_bytes]u8 = undefined;
    const n = encodeClientMsg(msg, request_id, session_id, &buf);
    try std.testing.expectEqual(clientFrameLen(msg), n);
    const h = try decodeHeader(buf[0..header_bytes]);
    try std.testing.expectEqual(request_id, h.request_id);
    try std.testing.expectEqual(session_id, h.session_id);
    try std.testing.expectEqual(n - header_bytes, h.payload_len);
    const decoded = try decodeClientPayload(h.msg_type, buf[header_bytes..n]);
    try std.testing.expectEqualDeep(msg, decoded);
}

fn roundTripServer(msg: ServerMsg, request_id: u32, session_id: u32) !void {
    var buf: [max_frame_payload_bytes]u8 = undefined;
    const n = encodeServerMsg(msg, request_id, session_id, &buf);
    try std.testing.expectEqual(serverFrameLen(msg), n);
    const h = try decodeHeader(buf[0..header_bytes]);
    try std.testing.expectEqual(request_id, h.request_id);
    try std.testing.expectEqual(session_id, h.session_id);
    const decoded = try decodeServerPayload(h.msg_type, buf[header_bytes..n]);
    try std.testing.expectEqualDeep(msg, decoded);
}

test "round-trip: every client message" {
    try roundTripClient(.hello, 1, 0);
    try roundTripClient(.{ .session_open = .{
        .sampling = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = 0xDEAD_BEEF },
        .system_prompt = "You are a helpful assistant.",
    } }, 2, 0);
    try roundTripClient(.{ .generate = .{ .max_tokens = 64, .sampling = null, .text = "Hi" } }, 3, 7);
    try roundTripClient(.{ .generate = .{
        .max_tokens = 8,
        .sampling = Sampling.greedy,
        .text = "Tell me more",
    } }, 4, 7);
    try roundTripClient(.cancel, 5, 7);
    try roundTripClient(.session_close, 6, 7);
    try roundTripClient(.stats, 7, 0);
    try roundTripClient(.ping, 8, 0);
    try roundTripClient(.shutdown, 9, 0);
    try roundTripClient(.{ .session_clone = .{
        .sampling = .{ .temperature = 0.6, .top_k = 20, .top_p = 0.9, .seed = 42, .presence_penalty = -1.5, .repetition_penalty = 1.3, .min_p = 0.05 },
    } }, 10, 7);
    // Every combination of the generate flags round trips.
    for ([_]?bool{ null, false, true }) |think_open| {
        for ([_]?u32{ null, 0, 512, limits.max_model_ctx }) |think_budget| {
            for ([_]?Sampling{ null, .{ .temperature = 0.7, .top_k = 40, .top_p = 0.8, .seed = 9, .min_p = 1, .repetition_penalty = 2 } }) |sampling| {
                try roundTripClient(.{ .generate = .{
                    .max_tokens = 5,
                    .sampling = sampling,
                    .text = "go",
                    .think_open = think_open,
                    .think_budget = think_budget,
                } }, 12, 7);
            }
        }
    }
    try roundTripClient(.{ .session_keep = .{ .ttl_ms = 0 } }, 13, 7);
    try roundTripClient(.{ .session_keep = .{ .ttl_ms = 60_000 } }, 14, 7);
    var token: Token = @splat(0xA5);
    token[15] = 1;
    try roundTripClient(.{ .session_resume = .{ .token = token } }, 15, 0);

    var pair_buf: [64]u8 = undefined;
    const pairs = [_]Pair{
        .{ .user = "Hi", .assistant = "Hello." },
        .{ .user = "2+2?", .assistant = "" },
    };
    const append = encodeAppendPairs(&pairs, &pair_buf);
    try std.testing.expectEqual(appendPairsBytes(&pairs), append.bytes.len);
    try roundTripClient(.{ .session_append = append }, 11, 7);
    var it = append.iterator();
    for (pairs) |want| {
        const got = it.next().?;
        try std.testing.expectEqualStrings(want.user, got.user);
        try std.testing.expectEqualStrings(want.assistant, got.assistant);
    }
    try std.testing.expect(it.next() == null);
}

test "round-trip: every server message" {
    try roundTripServer(.{ .hello_ack = .{
        .protocol_version = 1,
        .max_seq = 512,
        .max_sessions = 8,
        .max_batch = 8,
        .vocab_size = 49152,
        .eos_id = 2,
        .think_mode = .on,
        .think_budget = 512,
        .resume_ttl_max_ms = 60_000,
        .recommended_think = .{ .temperature = 1, .top_k = 20, .top_p = 0.95, .seed = 0, .presence_penalty = 1.5 },
        .recommended_plain = .{ .temperature = 0.7, .top_k = 20, .top_p = 0.8, .seed = 0, .presence_penalty = 1.5, .min_p = 0.1, .repetition_penalty = 1.1 },
        .model_name = "SmolLM2-135M-Instruct-Q8_0",
    } }, 1, 0);
    try roundTripServer(.{ .kept = .{ .granted_ms = 0, .token = @splat(0) } }, 11, 3);
    try roundTripServer(.{ .kept = .{ .granted_ms = limits.max_resume_ttl_ms, .token = @splat(7) } }, 12, 3);
    try roundTripServer(.{ .resumed = .{ .token = @splat(9), .kv_pos = 40, .turns_done = 3, .last = null } }, 13, 3);
    for (std.enums.values(FinishReason)) |finish| {
        try roundTripServer(.{ .resumed = .{ .token = @splat(9), .kv_pos = 40, .turns_done = 3, .last = .{
            .request_id = 77,
            .finish = finish,
            .gen_tokens = 12,
            .reasoning_tokens = 5,
        } } }, 14, 3);
    }
    try roundTripServer(.{ .session_opened = .{ .kv_pos = 37 } }, 2, 3);
    for (std.enums.values(Channel)) |channel| {
        try roundTripServer(.{ .token = .{ .token_id = 1234, .channel = channel, .text = " hello" } }, 3, 3);
    }
    try roundTripServer(.{ .done = .{
        .finish_reason = .eos,
        .turn_prompt_tokens = 12,
        .gen_tokens = 40,
        .kv_pos = 52,
        .prefill_ms = 12.5,
        .decode_ms = 800.0,
        .reasoning_tokens = 31,
    } }, 4, 3);
    var appended: Appended = .{ .n = 3 };
    appended.kv_pos[0..3].* = .{ 40, 40, 97 };
    try roundTripServer(.{ .appended = appended }, 10, 3);
    try roundTripServer(.{ .err = .{ .code = .turn_too_long, .detail = "turn exceeds context" } }, 5, 3);
    for (std.enums.values(ErrorCode)) |code| {
        try roundTripServer(.{ .err = .{ .code = code, .detail = @tagName(code) } }, 5, 0);
    }
    try roundTripServer(.{ .stats_ack = .{
        .sessions_open = 2,
        .sessions_decoding = 1,
        .tokens_generated_total = 12345,
        .uptime_ms = 99999,
    } }, 6, 0);
    try roundTripServer(.pong, 7, 0);
    try roundTripServer(.session_closed, 8, 3);
    try roundTripServer(.shutdown_ack, 9, 0);
}

test "the longest token message fills max_server_frame_bytes" {
    const text = [_]u8{'t'} ** 256;
    try std.testing.expectEqual(limits.max_token_text_bytes, text.len);
    const token: ServerMsg = .{ .token = .{ .token_id = 1234, .channel = .reasoning, .text = &text } };
    try std.testing.expectEqual(max_server_frame_bytes, serverFrameLen(token));
    try std.testing.expectEqual(@as(usize, 283), max_server_frame_bytes);

    const name = [_]u8{'m'} ** limits.max_model_name_bytes;
    const hello_ack: ServerMsg = .{ .hello_ack = .{
        .protocol_version = protocol_version,
        .max_seq = 1,
        .max_sessions = 1,
        .max_batch = 1,
        .vocab_size = 1,
        .eos_id = 0,
        .think_mode = .none,
        .think_budget = 0,
        .resume_ttl_max_ms = limits.max_resume_ttl_ms,
        .recommended_think = Sampling.greedy,
        .recommended_plain = Sampling.greedy,
        .model_name = &name,
    } };
    try std.testing.expectEqual(@as(usize, header_bytes + 165), serverFrameLen(hello_ack));
    try std.testing.expect(serverFrameLen(hello_ack) < max_server_frame_bytes);
    var buf: [max_server_frame_bytes]u8 = undefined;
    try std.testing.expectEqual(serverFrameLen(hello_ack), encodeServerMsg(hello_ack, 1, 0, &buf));
    const full: ServerMsg = .{ .appended = .{ .n = limits.max_append_pairs } };
    try std.testing.expect(serverFrameLen(full) <= max_server_frame_bytes);
}

const GuardMsg = union(enum) { client: ClientMsg, server: ServerMsg };

/// A byte-level edit of an encoded frame. Offsets count from the frame start.
const Patch = union(enum) {
    set: struct { at: usize, bytes: []const u8 },
    zero_from: usize,
    /// The new payload length.
    truncate: usize,
    /// Cut this many bytes from the end.
    shorten: usize,
    /// Append this many zero bytes.
    extend: usize,
};

const GuardCase = struct {
    name: []const u8,
    msg: GuardMsg,
    session_id: u32 = 1,
    patches: []const Patch = &.{},
    header_only: bool = false,
    /// Decode as another message type than the one that was encoded.
    decode_as: ?MsgType = null,
    /// Null means the frame decodes.
    want: ?DecodeError = null,
};

fn le(comptime T: type, v: T) [@sizeOf(T)]u8 {
    return std.mem.toBytes(std.mem.nativeToLittle(T, v));
}

fn f32le(v: f32) [4]u8 {
    return le(u32, @bitCast(v));
}

/// Encode the message, apply the patches, and decode. A refused frame must
/// give the exact error. An accepted frame must encode back to its own bytes,
/// so the decoder drops no field and reads none twice.
fn checkGuard(c: GuardCase) !void {
    var buf: [512]u8 = undefined;
    var end = switch (c.msg) {
        .client => |m| encodeClientMsg(m, 1, c.session_id, &buf),
        .server => |m| encodeServerMsg(m, 1, c.session_id, &buf),
    };
    try std.testing.expectEqual(protocol_version, std.mem.readInt(u16, buf[4..6], .little));
    for (c.patches) |patch| switch (patch) {
        .set => |s| @memcpy(buf[s.at..][0..s.bytes.len], s.bytes),
        .zero_from => |at| @memset(buf[at..end], 0),
        .truncate => |len| end = header_bytes + len,
        .shorten => |n| end -= n,
        .extend => |n| {
            @memset(buf[end..][0..n], 0);
            end += n;
        },
    };
    if (c.header_only) {
        const got = decodeHeader(buf[0..header_bytes]);
        if (c.want) |err| return std.testing.expectError(err, got);
        try std.testing.expectEqual(c.session_id, (try got).session_id);
        return;
    }
    const payload = buf[header_bytes..end];
    var again: [512]u8 = undefined;
    switch (c.msg) {
        .client => |m| {
            const t = c.decode_as orelse clientMsgType(m);
            if (c.want) |err| return std.testing.expectError(err, decodeClientPayload(t, payload));
            const k = encodeClientMsg(try decodeClientPayload(t, payload), 1, c.session_id, &again);
            try std.testing.expectEqualSlices(u8, payload, again[header_bytes..k]);
        },
        .server => |m| {
            const t = c.decode_as orelse serverMsgType(m);
            if (c.want) |err| return std.testing.expectError(err, decodeServerPayload(t, payload));
            const k = encodeServerMsg(try decodeServerPayload(t, payload), 1, c.session_id, &again);
            try std.testing.expectEqualSlices(u8, payload, again[header_bytes..k]);
        },
    }
}

fn setAt(comptime at: usize, comptime bytes: []const u8) []const Patch {
    return &.{.{ .set = .{ .at = at, .bytes = bytes } }};
}

const gen_x: GuardMsg = .{ .client = .{ .generate = .{ .max_tokens = 4, .sampling = null, .text = "x" } } };
const open_greedy: GuardMsg = .{ .client = .{ .session_open = .{ .sampling = Sampling.greedy, .system_prompt = "" } } };
const ping_msg: GuardMsg = .{ .client = .ping };
const flags_at = header_bytes + 4;
const frequency_at = header_bytes + 24;
const repetition_at = header_bytes + 28;
const min_p_at = header_bytes + 32;

fn samplingGuard(comptime name: []const u8, comptime at: usize, comptime v: f32, want: ?DecodeError) GuardCase {
    return .{ .name = name, .msg = open_greedy, .patches = setAt(at, &f32le(v)), .want = want };
}

fn penaltyOk(comptime field: []const u8, comptime v: f32) GuardCase {
    var s = Sampling.greedy;
    @field(s, field) = v;
    return .{ .name = "penalty in range decodes", .msg = .{ .client = .{ .session_open = .{ .sampling = s, .system_prompt = "" } } } };
}

fn kvPosAppended() GuardMsg {
    var reply: Appended = .{ .n = 2 };
    reply.kv_pos[0..2].* = .{ 10, 12 };
    return .{ .server = .{ .appended = reply } };
}

const append_two: GuardMsg = .{ .client = .{ .session_append = .{
    .n = 2,
    .bytes = &.{ 2, 0, 0, 0, 'a', 'b', 1, 0, 0, 0, 'c', 1, 0, 0, 0, 'd', 0, 0, 0, 0 },
} } };

const resumed_full: GuardMsg = .{ .server = .{ .resumed = .{ .token = @splat(9), .kv_pos = 4, .turns_done = 1, .last = .{
    .request_id = 3,
    .finish = .eos,
    .gen_tokens = 2,
    .reasoning_tokens = 0,
} } } };
const resumed_none: GuardMsg = .{ .server = .{ .resumed = .{ .token = @splat(9), .kv_pos = 4, .turns_done = 1, .last = null } } };
const kept_msg: GuardMsg = .{ .server = .{ .kept = .{ .granted_ms = 10, .token = @splat(4) } } };
const rp = header_bytes;

test "guards: every refused frame returns its typed error, and every edge decodes" {
    const bad: DecodeError = error.PayloadMalformed;
    const cases = [_]GuardCase{
        // Header.
        .{ .name = "header intact", .msg = ping_msg, .header_only = true },
        .{ .name = "bad magic", .msg = ping_msg, .header_only = true, .patches = setAt(0, &.{0xFF}), .want = error.BadMagic },
        .{ .name = "version 99", .msg = ping_msg, .header_only = true, .patches = setAt(4, &le(u16, 99)), .want = error.BadVersion },
        .{ .name = "version 2", .msg = ping_msg, .header_only = true, .patches = setAt(4, &le(u16, 2)), .want = error.BadVersion },
        .{ .name = "version 1", .msg = ping_msg, .header_only = true, .patches = setAt(4, &le(u16, 1)), .want = error.BadVersion },
        .{ .name = "version 4", .msg = ping_msg, .header_only = true, .patches = setAt(4, &le(u16, 4)), .want = error.BadVersion },
        .{ .name = "unknown type", .msg = ping_msg, .header_only = true, .patches = setAt(6, &le(u16, 0x7777)), .want = error.UnknownMsgType },
        .{ .name = "frame too large", .msg = ping_msg, .header_only = true, .patches = setAt(16, &le(u32, max_frame_payload_bytes + 1)), .want = error.FrameTooLarge },
        .{ .name = "resume with no session id", .msg = .{ .client = .{ .session_resume = .{ .token = @splat(3) } } }, .session_id = 0, .header_only = true },
        .{ .name = "resume with a session id", .msg = .{ .client = .{ .session_resume = .{ .token = @splat(3) } } }, .session_id = 0, .header_only = true, .patches = setAt(12, &le(u32, 7)), .want = bad },
        .{ .name = "keep keeps its session id", .msg = .{ .client = .{ .session_keep = .{ .ttl_ms = 1 } } }, .session_id = 7, .header_only = true },
        // Direction confusion.
        .{ .name = "server type to the client decoder", .msg = .{ .client = .ping }, .decode_as = .token, .want = error.UnknownMsgType },
        .{ .name = "client type to the server decoder", .msg = .{ .server = .pong }, .decode_as = .generate, .want = error.UnknownMsgType },
        // Generate flags.
        .{ .name = "think switch alone", .msg = gen_x, .patches = setAt(flags_at, &.{generate_flag_think_switch}) },
        .{ .name = "think switch and open", .msg = gen_x, .patches = setAt(flags_at, &.{generate_flag_think_switch | generate_flag_think_open}) },
        .{ .name = "think open alone", .msg = gen_x, .patches = setAt(flags_at, &.{generate_flag_think_open}), .want = bad },
        .{ .name = "reseed without sampling", .msg = gen_x, .patches = setAt(flags_at, &.{generate_flag_reseed}), .want = bad },
        .{ .name = "flag bit 5", .msg = gen_x, .patches = setAt(flags_at, &.{1 << 5}), .want = bad },
        .{ .name = "flag bit 6", .msg = gen_x, .patches = setAt(flags_at, &.{1 << 6}), .want = bad },
        .{ .name = "flag bit 7", .msg = gen_x, .patches = setAt(flags_at, &.{1 << 7}), .want = bad },
        .{ .name = "sampling without reseed", .msg = .{ .client = .{ .generate = .{ .max_tokens = 4, .sampling = Sampling.greedy, .text = "x", .reseed = false } } } },
        .{ .name = "sampling with reseed", .msg = .{ .client = .{ .generate = .{ .max_tokens = 4, .sampling = Sampling.greedy, .text = "x", .reseed = true } } } },
        .{ .name = "think budget at the bound", .msg = .{ .client = .{ .generate = .{ .max_tokens = 4, .sampling = null, .text = "x", .think_budget = limits.max_model_ctx } } } },
        .{
            .name = "think budget past the bound",
            .msg = .{ .client = .{ .generate = .{ .max_tokens = 4, .sampling = null, .text = "x", .think_budget = limits.max_model_ctx } } },
            .patches = setAt(header_bytes + 5, &le(u32, limits.max_model_ctx + 1)),
            .want = bad,
        },
        .{ .name = "budget flag with no field", .msg = gen_x, .patches = setAt(flags_at, &.{generate_flag_think_budget}), .want = bad },
        .{ .name = "text length past the payload", .msg = .{ .client = .{ .generate = .{ .max_tokens = 4, .sampling = null, .text = "abc" } } }, .patches = setAt(header_bytes + 5, &le(u32, 1000)), .want = bad },
        .{ .name = "zero max_tokens", .msg = gen_x, .patches = setAt(header_bytes, &le(u32, 0)), .want = bad },
        // Payload shape.
        .{ .name = "cut session_open", .msg = open_greedy, .patches = &.{.{ .truncate = 3 }}, .want = bad },
        .{ .name = "trailing byte", .msg = .{ .client = .{ .session_open = .{ .sampling = Sampling.greedy, .system_prompt = "p" } } }, .patches = &.{.{ .extend = 1 }}, .want = bad },
        .{ .name = "channel past the last", .msg = .{ .server = .{ .token = .{ .token_id = 1, .channel = .close_tag, .text = "x" } } }, .patches = setAt(header_bytes + 4, &.{4}), .want = bad },
        .{ .name = "last error code decodes", .msg = .{ .server = .{ .err = .{ .code = .cancelled, .detail = "x" } } } },
        .{ .name = "error code 0", .msg = .{ .server = .{ .err = .{ .code = .cancelled, .detail = "x" } } }, .patches = setAt(header_bytes, &le(u16, 0)), .want = bad },
        .{ .name = "error code past the last", .msg = .{ .server = .{ .err = .{ .code = .cancelled, .detail = "x" } } }, .patches = setAt(header_bytes, &le(u16, @intFromEnum(ErrorCode.cancelled) + 1)), .want = bad },
        // Sampling values.
        samplingGuard("negative temperature", header_bytes, -1, bad),
        samplingGuard("presence 2.01", header_bytes + 20, 2.01, bad),
        samplingGuard("presence -2.01", header_bytes + 20, -2.01, bad),
        samplingGuard("presence NaN", header_bytes + 20, std.math.nan(f32), bad),
        samplingGuard("presence inf", header_bytes + 20, std.math.inf(f32), bad),
        penaltyOk("presence_penalty", 2),
        penaltyOk("presence_penalty", -2),
        penaltyOk("presence_penalty", 0),
        samplingGuard("frequency 2.01", frequency_at, 2.01, bad),
        samplingGuard("frequency -2.01", frequency_at, -2.01, bad),
        samplingGuard("frequency inf", frequency_at, std.math.inf(f32), bad),
        samplingGuard("frequency NaN", frequency_at, std.math.nan(f32), bad),
        penaltyOk("frequency_penalty", 0.5),
        penaltyOk("frequency_penalty", -2),
        penaltyOk("frequency_penalty", 2),
        samplingGuard("repetition 0", repetition_at, 0, bad),
        samplingGuard("repetition -1", repetition_at, -1, bad),
        samplingGuard("repetition past the ceiling", repetition_at, limits.max_repetition_penalty + 0.01, bad),
        samplingGuard("repetition NaN", repetition_at, std.math.nan(f32), bad),
        samplingGuard("min_p -0.01", min_p_at, -0.01, bad),
        samplingGuard("min_p 1.01", min_p_at, 1.01, bad),
        samplingGuard("min_p NaN", min_p_at, std.math.nan(f32), bad),
        samplingGuard("min_p inf", min_p_at, std.math.inf(f32), bad),
        .{ .name = "sampling upper edges decode", .session_id = 3, .msg = .{ .client = .{ .session_clone = .{ .sampling = edgeSampling(limits.max_repetition_penalty, 1) } } } },
        .{ .name = "sampling lower edges decode", .session_id = 3, .msg = .{ .client = .{ .session_clone = .{ .sampling = edgeSampling(std.math.floatMin(f32), 0) } } } },
        // Lease frames.
        .{ .name = "resume with a zero token", .session_id = 0, .msg = .{ .client = .{ .session_resume = .{ .token = @splat(3) } } }, .patches = &.{.{ .zero_from = rp }}, .want = bad },
        .{ .name = "cut resume", .session_id = 0, .msg = .{ .client = .{ .session_resume = .{ .token = @splat(3) } } }, .patches = &.{.{ .shorten = 1 }}, .want = bad },
        .{ .name = "cut keep", .session_id = 2, .msg = .{ .client = .{ .session_keep = .{ .ttl_ms = 5 } } }, .patches = &.{.{ .shorten = 1 }}, .want = bad },
        .{ .name = "kept intact", .session_id = 2, .msg = kept_msg },
        .{ .name = "kept with a zero grant", .session_id = 2, .msg = kept_msg, .patches = setAt(rp, &le(u32, 0)), .want = bad },
        .{ .name = "kept with a zero token", .session_id = 2, .msg = kept_msg, .patches = &.{.{ .zero_from = rp + 4 }}, .want = bad },
        .{ .name = "kept past the ceiling", .session_id = 2, .msg = kept_msg, .patches = setAt(rp, &le(u32, limits.max_resume_ttl_ms + 1)), .want = bad },
        .{ .name = "resumed intact", .session_id = 2, .msg = resumed_full },
        .{ .name = "resumed has_last 2", .session_id = 2, .msg = resumed_full, .patches = setAt(rp + 24, &.{2}), .want = bad },
        .{ .name = "resumed bad finish", .session_id = 2, .msg = resumed_full, .patches = setAt(rp + 29, &.{4}), .want = bad },
        .{ .name = "resumed zero token", .session_id = 2, .msg = resumed_full, .patches = setAt(rp, &([_]u8{0} ** token_bytes)), .want = bad },
        .{ .name = "resumed without last, intact", .session_id = 2, .msg = resumed_none },
        .{ .name = "resumed without last, byte 25", .session_id = 2, .msg = resumed_none, .patches = setAt(rp + 25, &.{1}), .want = bad },
        .{ .name = "resumed without last, byte 29", .session_id = 2, .msg = resumed_none, .patches = setAt(rp + 29, &.{1}), .want = bad },
        .{ .name = "resumed without last, byte 30", .session_id = 2, .msg = resumed_none, .patches = setAt(rp + 30, &.{1}), .want = bad },
        .{ .name = "resumed without last, byte 34", .session_id = 2, .msg = resumed_none, .patches = setAt(rp + 34, &.{1}), .want = bad },
        .{ .name = "cut resumed", .session_id = 2, .msg = resumed_none, .patches = &.{.{ .shorten = 1 }}, .want = bad },
        // Append.
        .{ .name = "append intact", .session_id = 3, .msg = append_two },
        .{ .name = "append count 0", .session_id = 3, .msg = append_two, .patches = setAt(rp, &le(u16, 0)), .want = bad },
        .{ .name = "append count past the bound", .session_id = 3, .msg = append_two, .patches = setAt(rp, &le(u16, limits.max_append_pairs + 1)), .want = bad },
        .{ .name = "append count the bytes do not hold", .session_id = 3, .msg = append_two, .patches = setAt(rp, &le(u16, 3)), .want = bad },
        .{ .name = "append user length past the payload", .session_id = 3, .msg = append_two, .patches = setAt(rp + 2, &le(u32, 1000)), .want = bad },
        .{ .name = "append trailing byte", .session_id = 3, .msg = append_two, .patches = &.{.{ .extend = 1 }}, .want = bad },
        .{ .name = "append cut", .session_id = 3, .msg = append_two, .patches = &.{.{ .shorten = 1 }}, .want = bad },
        .{ .name = "appended list intact", .session_id = 3, .msg = kvPosAppended() },
        .{ .name = "appended list decreases", .session_id = 3, .msg = kvPosAppended(), .patches = setAt(rp + 2, &le(u32, 13)), .want = bad },
        .{ .name = "appended list empty", .session_id = 3, .msg = kvPosAppended(), .patches = &.{ .{ .set = .{ .at = rp, .bytes = &le(u16, 0) } }, .{ .truncate = 2 } }, .want = bad },
    };
    for (cases) |c| {
        errdefer std.debug.print("guard case: {s}\n", .{c.name});
        try checkGuard(c);
    }
}

fn edgeSampling(repetition: f32, min_p: f32) Sampling {
    var s = Sampling.greedy;
    s.repetition_penalty = repetition;
    s.min_p = min_p;
    return s;
}

test "mutation: one flipped byte in each new frame decodes to a typed error or a valid message" {
    var token: Token = @splat(0x5A);
    token[0] = 1;
    const clients = [_]ClientMsg{
        .{ .session_keep = .{ .ttl_ms = 30_000 } },
        .{ .session_resume = .{ .token = token } },
        .{ .generate = .{ .max_tokens = 8, .sampling = .{ .temperature = 0.5, .top_k = 10, .top_p = 0.9, .seed = 3, .min_p = 0.1, .repetition_penalty = 1.2 }, .text = "hi", .think_open = true, .think_budget = 64, .reseed = true } },
    };
    var buf: [256]u8 = undefined;
    for (clients) |msg| {
        const n = encodeClientMsg(msg, 1, 1, &buf);
        for (header_bytes..n) |at| {
            for (0..8) |bit| {
                var m = buf;
                m[at] ^= @as(u8, 1) << @intCast(bit);
                const got = decodeClientPayload(clientMsgType(msg), m[header_bytes..n]) catch continue;
                // A decoded mutant re-encodes to its own bytes, so no field
                // is dropped or read twice.
                var again: [256]u8 = undefined;
                const k = encodeClientMsg(got, 1, 1, &again);
                try std.testing.expectEqualSlices(u8, m[header_bytes..n], again[header_bytes..k]);
            }
        }
    }
    const servers = [_]ServerMsg{
        .{ .kept = .{ .granted_ms = 500, .token = token } },
        .{ .resumed = .{ .token = token, .kv_pos = 9, .turns_done = 2, .last = .{ .request_id = 4, .finish = .max_tokens, .gen_tokens = 3, .reasoning_tokens = 1 } } },
    };
    for (servers) |msg| {
        const n = encodeServerMsg(msg, 1, 1, &buf);
        for (header_bytes..n) |at| {
            for (0..8) |bit| {
                var m = buf;
                m[at] ^= @as(u8, 1) << @intCast(bit);
                const got = decodeServerPayload(serverMsgType(msg), m[header_bytes..n]) catch continue;
                var again: [256]u8 = undefined;
                const k = encodeServerMsg(got, 1, 1, &again);
                try std.testing.expectEqualSlices(u8, m[header_bytes..n], again[header_bytes..k]);
            }
        }
    }
}

const MutationCase = struct { msg: u8, seed: u64 };

fn genMutation(_: void, random: std.Random, size: u32) MutationCase {
    _ = size;
    return .{ .msg = random.int(u8), .seed = random.int(u64) };
}

fn checkMutation(_: void, c: MutationCase) anyerror!void {
    var prng = std.Random.DefaultPrng.init(c.seed);
    const random = prng.random();
    var buf: [512]u8 = undefined;
    const seeds = [_]ClientMsg{
        .{ .session_open = .{ .sampling = Sampling.greedy, .system_prompt = "a system prompt" } },
        .{ .generate = .{ .max_tokens = 16, .sampling = .{
            .temperature = 0.7,
            .top_k = 50,
            .top_p = 0.9,
            .seed = 42,
        }, .text = "hello there" } },
        .ping,
        .{ .session_append = .{ .n = 1, .bytes = &.{ 2, 0, 0, 0, 'h', 'i', 1, 0, 0, 0, 'o' } } },
    };
    const n = encodeClientMsg(seeds[c.msg % seeds.len], 1, 1, &buf);
    var mutated: [512]u8 = buf;
    const flips = 1 + random.uintLessThan(usize, 6);
    for (0..flips) |_| {
        mutated[random.uintLessThan(usize, n)] ^= @as(u8, 1) << random.int(u3);
    }
    const h = decodeHeader(mutated[0..header_bytes]) catch return;
    const end = @min(header_bytes + h.payload_len, n);
    const got = decodeClientPayload(h.msg_type, mutated[header_bytes..end]) catch return;
    // A decoded mutant re-encodes to its own payload, so the decoder drops no byte.
    var again: [1024]u8 = undefined;
    const k = encodeClientMsg(got, 1, 1, &again);
    try std.testing.expectEqualSlices(u8, mutated[header_bytes..end], again[header_bytes..k]);
}

test "property: random byte mutations decode to a typed error or a valid message, never a crash" {
    try prop.forAll(.{ .site = 0x5EED, .cases = 1200 }, {}, MutationCase, genMutation, checkMutation);
}

/// A case of the codec properties: one seed, and a size that bounds every
/// generated length.
const CodecCase = struct { seed: u64, size: u32 };

fn genCodec(_: void, random: std.Random, size: u32) CodecCase {
    return .{ .seed = random.int(u64), .size = size };
}

/// A value in `[lo, hi]`, with each edge drawn one time in eight.
fn floatIn(r: std.Random, lo: f32, hi: f32) f32 {
    return switch (r.uintLessThan(u8, 8)) {
        0 => lo,
        1 => hi,
        else => lo + (hi - lo) * r.float(f32),
    };
}

/// A sampling block that the decoder guards accept.
fn genSampling(r: std.Random) Sampling {
    const pmax = limits.max_presence_penalty;
    return .{
        .temperature = floatIn(r, 0, 4),
        .top_k = r.uintAtMost(u32, limits.max_top_k),
        .top_p = floatIn(r, std.math.floatMin(f32), 1),
        .seed = r.int(u64),
        .presence_penalty = floatIn(r, -pmax, pmax),
        .frequency_penalty = floatIn(r, -frequency_penalty_max, frequency_penalty_max),
        .repetition_penalty = floatIn(r, std.math.floatMin(f32), limits.max_repetition_penalty),
        .min_p = floatIn(r, 0, 1),
    };
}

fn genBytes(r: std.Random, buf: []u8, len_max: usize) []u8 {
    const len = r.uintAtMost(usize, @min(len_max, buf.len));
    r.bytes(buf[0..len]);
    return buf[0..len];
}

/// A token that is not all zero.
fn genToken(r: std.Random) Token {
    var t: Token = undefined;
    r.bytes(&t);
    t[r.uintLessThan(usize, token_bytes)] |= 1;
    return t;
}

fn genClientMsg(r: std.Random, size: u32, text: *[256]u8, pair_buf: *[256]u8) ClientMsg {
    const tag = r.enumValue(std.meta.Tag(ClientMsg));
    return switch (tag) {
        inline .hello, .cancel, .session_close, .stats, .ping, .shutdown => |t| t,
        .session_open => .{ .session_open = .{ .sampling = genSampling(r), .system_prompt = genBytes(r, text, size * 4) } },
        .session_clone => .{ .session_clone = .{ .sampling = genSampling(r) } },
        .session_keep => .{ .session_keep = .{ .ttl_ms = r.int(u32) } },
        .session_resume => .{ .session_resume = .{ .token = genToken(r) } },
        .generate => blk: {
            const sampling: ?Sampling = if (r.boolean()) genSampling(r) else null;
            break :blk .{ .generate = .{
                .max_tokens = 1 + r.uintLessThan(u32, std.math.maxInt(u32)),
                .sampling = sampling,
                .text = genBytes(r, text, size * 2),
                .think_open = if (r.boolean()) r.boolean() else null,
                .think_budget = if (r.boolean()) r.uintAtMost(u32, limits.max_model_ctx) else null,
                .reseed = sampling != null and r.boolean(),
            } };
        },
        .session_append => blk: {
            var pairs: [8]Pair = undefined;
            const n = 1 + r.uintLessThan(usize, pairs.len);
            var used: usize = 0;
            for (pairs[0..n]) |*p| {
                p.user = genBytes(r, text[used..], 8);
                used += p.user.len;
                p.assistant = genBytes(r, text[used..], 8);
                used += p.assistant.len;
            }
            break :blk .{ .session_append = encodeAppendPairs(pairs[0..n], pair_buf) };
        },
    };
}

fn genServerMsg(r: std.Random, size: u32, text: *[256]u8) ServerMsg {
    const tag = r.enumValue(std.meta.Tag(ServerMsg));
    return switch (tag) {
        inline .pong, .session_closed, .shutdown_ack => |t| t,
        .hello_ack => .{ .hello_ack = .{
            .protocol_version = r.int(u16),
            .max_seq = r.int(u32),
            .max_sessions = r.int(u16),
            .max_batch = r.int(u16),
            .vocab_size = r.int(u32),
            .eos_id = r.int(u32),
            .think_mode = r.enumValue(ThinkMode),
            .think_budget = r.int(u32),
            .resume_ttl_max_ms = r.uintAtMost(u32, limits.max_resume_ttl_ms),
            .recommended_think = genSampling(r),
            .recommended_plain = genSampling(r),
            .model_name = genBytes(r, text, @min(size * 2, limits.max_model_name_bytes)),
        } },
        .session_opened => .{ .session_opened = .{ .kv_pos = r.int(u32) } },
        .token => .{ .token = .{
            .token_id = r.int(u32),
            .channel = r.enumValue(Channel),
            .text = genBytes(r, text, @min(size * 4, limits.max_token_text_bytes)),
        } },
        .done => .{ .done = .{
            .finish_reason = r.enumValue(FinishReason),
            .turn_prompt_tokens = r.int(u32),
            .gen_tokens = r.int(u32),
            .kv_pos = r.int(u32),
            .prefill_ms = floatIn(r, 0, 1e6),
            .decode_ms = floatIn(r, 0, 1e6),
            .reasoning_tokens = r.int(u32),
        } },
        .err => .{ .err = .{ .code = r.enumValue(ErrorCode), .detail = genBytes(r, text, size * 4) } },
        .stats_ack => .{ .stats_ack = .{
            .sessions_open = r.int(u16),
            .sessions_decoding = r.int(u16),
            .tokens_generated_total = r.int(u64),
            .uptime_ms = r.int(u64),
        } },
        .appended => blk: {
            var a: Appended = .{ .n = 1 + r.uintLessThan(u16, limits.max_append_pairs) };
            var pos: u32 = r.uintAtMost(u32, 1 << 20);
            for (a.kv_pos[0..a.n]) |*p| {
                pos += r.uintAtMost(u32, 64);
                p.* = pos;
            }
            break :blk .{ .appended = a };
        },
        .kept => blk: {
            const granted = r.uintAtMost(u32, limits.max_resume_ttl_ms);
            break :blk .{ .kept = .{ .granted_ms = granted, .token = if (granted == 0) @splat(0) else genToken(r) } };
        },
        .resumed => .{ .resumed = .{
            .token = genToken(r),
            .kv_pos = r.int(u32),
            .turns_done = r.int(u32),
            .last = if (r.boolean()) .{
                .request_id = r.int(u32),
                .finish = r.enumValue(FinishReason),
                .gen_tokens = r.int(u32),
                .reasoning_tokens = r.int(u32),
            } else null,
        } },
    };
}

/// Every truncation of the payload of `msg` is a cut frame, and the guard
/// table names `PayloadMalformed` for a cut frame.
fn expectCutsRefused(msg: GuardMsg, session_id: u32) !void {
    var buf: [512]u8 = undefined;
    const n = switch (msg) {
        .client => |m| encodeClientMsg(m, 1, session_id, &buf),
        .server => |m| encodeServerMsg(m, 1, session_id, &buf),
    };
    for (1..n - header_bytes + 1) |cut| {
        try checkGuard(.{
            .name = "cut",
            .msg = msg,
            .session_id = session_id,
            .patches = &.{.{ .shorten = cut }},
            .want = error.PayloadMalformed,
        });
    }
}

/// A generated message of each direction decodes to itself, and every
/// truncation of its payload returns `PayloadMalformed`.
fn checkCodec(_: void, c: CodecCase) anyerror!void {
    var prng = std.Random.DefaultPrng.init(c.seed);
    const r = prng.random();
    var text: [256]u8 = undefined;
    var pair_buf: [256]u8 = undefined;
    const client = genClientMsg(r, c.size, &text, &pair_buf);
    // A resume names its session by the token alone.
    const sid: u32 = if (client == .session_resume) 0 else r.int(u32);
    try roundTripClient(client, r.int(u32), sid);
    try expectCutsRefused(.{ .client = client }, sid);

    const server = genServerMsg(r, c.size, &text);
    try roundTripServer(server, r.int(u32), r.int(u32));
    try expectCutsRefused(.{ .server = server }, 1);
}

test "property: every generated message round trips, and every cut of its payload is malformed" {
    try prop.forAll(.{ .site = 0xC0DE_C0DE, .cases = 600 }, {}, CodecCase, genCodec, checkCodec);
}
