//! The reference client for the serve wire format. A port to another language copies this file,
//! and the integration tests and `bench-serve` drive the server through it. The client uses
//! ZMTP 3.x as a DEALER, and a ZMQ binding in any language can replace it.
//!
//! Methods. Each request method is send-only or terminal-wait. A send-only method writes one
//! frame and returns its request id. The caller then reads frames through `next` or
//! `collectTurn`. A terminal-wait method writes one frame, consumes frames until the terminal
//! message of that request, and returns the reply as a value. Terminal-wait methods assume one
//! request in flight. A caller that multiplexes sessions loops over `next` itself.
//!
//! Tasks. `next` and `send` touch disjoint fields, so one task can read while another task
//! writes. `deinit` must run after both tasks return.
//!
//! Borrowing. `next` returns a message whose text slices BORROW the client's payload buffer. Copy
//! a slice before the next call to keep it. Token text is raw bytes and can hold a partial UTF-8
//! sequence. Join a whole turn before you display it.
//!
//! Refusals. A refusal is `error.ServerError` with the reason in `last_error`. Read it
//! immediately after that error, because a later call overwrites it.
//!
//! Leases. A keep issues a new token, and the old one opens nothing. A resume also rotates the
//! token, so the caller stores the token of the reply. `unknown_session` answers an unknown,
//! rotated, or expired token.
//!
//! Cancel. A cancel ends the turn in flight, and that turn's `done` carries `cancelled` under the
//! request id of its `generate`, unless the turn ends first. A cancel on an idle session produces
//! no frame. An unknown session produces an `err` frame under the returned request id.
//!
//! Collected turns. `collectTurn` fills `ids_out` and `text_out` each as far as it fits, and
//! `n_tokens` and `text_total` stay the true totals. Once `hello` has run, the text of the
//! end-of-sequence token is not copied, and its id still lands in `ids_out`.

const std = @import("std");
const Io = std.Io;
const net = Io.net;
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const zmtp = @import("zmtp.zig");
const limits = @import("../core/limits.zig");

const log = std.log.scoped(.serve_client);

// -- constants --------------------------------------------------------------

/// The long ZMTP frame head: one flags byte and an 8-byte size.
const zmtp_head_bytes: usize = 9;

// -- types ------------------------------------------------------------------

/// The hook of `collectTurn`, which needs no work per token.
const NoTokenHook = struct {
    fn onToken(_: NoTokenHook, _: u32, _: protocol.Channel, _: []const u8) void {}
};

pub const Event = struct {
    header: protocol.Header,
    msg: protocol.ServerMsg,
};

/// A fixed-size copy of a borrowed `Event`, for a caller that hands frames across tasks.
pub const OwnedEvent = struct {
    header: protocol.Header,
    body: Body,

    pub const Body = union(enum) {
        hello_ack: Client.HelloInfo,
        session_opened: @FieldType(protocol.ServerMsg, "session_opened"),
        token: struct {
            token_id: u32,
            channel: protocol.Channel,
            len: u16,
            text: [limits.max_token_text_bytes]u8,
        },
        done: @FieldType(protocol.ServerMsg, "done"),
        appended: protocol.Appended,
        kept: @FieldType(protocol.ServerMsg, "kept"),
        resumed: protocol.Resumed,
        err: Client.LastError,
        stats_ack: Client.Stats,
        pong,
        session_closed,
        shutdown_ack,
    };
};

pub const Client = struct {
    gpa: Allocator,
    io: Io,
    stream: net.Stream,
    reader: net.Stream.Reader,
    writer: net.Stream.Writer,
    read_buf: []u8,
    write_buf: []u8,
    payload_buf: []u8,
    frame_buf: []u8,
    /// The pair records of one `session_append`, written before the frame.
    pair_buf: []u8,
    next_request_id: u32 = 1,
    /// The end-of-sequence token id from `hello_ack`. `collectTurn` skips its text.
    eos_id: ?u32 = null,
    /// The last `err` frame a convenience call consumed, for the server's words of a refusal.
    last_error: ?LastError = null,

    /// A recorded server error. The buffer matches the 64-byte detail bound of `OutEvent.fail`.
    /// A longer detail from a foreign server is cut at the bound, and `truncated` is set.
    pub const LastError = struct {
        code: protocol.ErrorCode,
        len: u8,
        truncated: bool,
        buf: [detail_cap]u8,

        pub const detail_cap = 64;

        pub fn detail(self: *const LastError) []const u8 {
            return self.buf[0..self.len];
        }
    };

    /// `HandshakeRefused` is a server that answers the greeting with a ZMTP ERROR.
    /// `PeerTypeRefused` is a server that is not a ROUTER.
    pub const HandshakeError = error{
        HandshakeRefused,
        PeerTypeRefused,
        ReadFailed,
        EndOfStream,
        WriteFailed,
    } || zmtp.Error;

    /// A send fails only on the socket. `writer.err` holds the cause.
    pub const SendError = Io.Writer.Error;

    /// `error.TextTooLong` is a text past the bound of one `generate` frame.
    pub const GenerateError = SendError || error{TextTooLong};

    /// `error.ServerError` is a refusal, with the reason in `last_error`.
    pub const CollectError = NextError || error{ ServerError, UnexpectedReply };

    /// The errors of a call that sends one request and waits for its reply.
    pub const ReplyError = SendError || CollectError;

    pub fn connectTcp(
        gpa: Allocator,
        io: Io,
        address: net.IpAddress,
    ) (net.IpAddress.ConnectError || Allocator.Error || HandshakeError)!Client {
        var addr = address;
        const stream = try addr.connect(io, .{ .mode = .stream });
        errdefer stream.close(io);

        return fromStream(gpa, io, stream);
    }

    pub fn connectUnix(
        gpa: Allocator,
        io: Io,
        path: []const u8,
    ) (net.UnixAddress.InitError ||
        net.UnixAddress.ConnectError ||
        Allocator.Error ||
        HandshakeError)!Client {
        const addr = try net.UnixAddress.init(path);
        const stream = try addr.connect(io);
        errdefer stream.close(io);

        return fromStream(gpa, io, stream);
    }

    fn fromStream(
        gpa: Allocator,
        io: Io,
        stream: net.Stream,
    ) (Allocator.Error || HandshakeError)!Client {
        const read_buf = try gpa.alloc(u8, 4096);
        errdefer gpa.free(read_buf);

        const write_buf = try gpa.alloc(u8, 4096);
        errdefer gpa.free(write_buf);

        const payload_buf = try gpa.alloc(u8, protocol.max_frame_payload_bytes);
        errdefer gpa.free(payload_buf);

        const frame_buf = try gpa.alloc(
            u8,
            zmtp_head_bytes + protocol.header_bytes + protocol.max_frame_payload_bytes,
        );
        errdefer gpa.free(frame_buf);

        const pair_buf = try gpa.alloc(u8, protocol.append_pairs_bytes_max);
        errdefer gpa.free(pair_buf);

        var client: Client = .{
            .gpa = gpa,
            .io = io,
            .stream = stream,
            .reader = stream.reader(io, read_buf),
            .writer = stream.writer(io, write_buf),
            .read_buf = read_buf,
            .write_buf = write_buf,
            .payload_buf = payload_buf,
            .frame_buf = frame_buf,
            .pair_buf = pair_buf,
        };
        try client.handshake();
        return client;
    }

    /// Sends the greeting and READY of the DEALER side, then reads the greeting and READY of the
    /// server. The reads hold at most `zmtp.greeting_bytes`, one frame head, and
    /// `zmtp.max_ready_bytes`.
    fn handshake(self: *Client) HandshakeError!void {
        const w = &self.writer.interface;
        var greeting: [zmtp.greeting_bytes]u8 = undefined;
        zmtp.encodeGreeting(&greeting);
        try w.writeAll(&greeting);
        const body = self.frame_buf[zmtp_head_bytes..];
        const n = zmtp.encodeReady(zmtp.socket_type_dealer, body);
        const head_len = zmtp.encodeFrameHead(n, true, self.frame_buf[0..zmtp_head_bytes]);
        try w.writeAll(self.frame_buf[0..head_len]);
        try w.writeAll(body[0..n]);
        try w.flush();
        log.debug("handshake: sent greeting and READY", .{});

        const r = &self.reader.interface;
        try zmtp.decodeGreeting(try r.takeArray(zmtp.greeting_bytes));
        log.debug("handshake: accepted server greeting", .{});
        const head = try readFrameHead(r);
        if (!head.command) return error.Malformed;
        if (head.size > zmtp.max_ready_bytes) return error.MessageTooLarge;

        const command_body = self.payload_buf[0..@intCast(head.size)];
        try r.readSliceAll(command_body);
        const ready = switch (try zmtp.decodeCommand(command_body)) {
            .ready => |ready| ready,
            .err => |reason| {
                log.debug("handshake: server refused, {s}", .{reason});
                return error.HandshakeRefused;
            },
            .ping, .pong, .other => return error.Malformed,
        };

        if (!std.ascii.eqlIgnoreCase(ready.socket_type, zmtp.socket_type_router)) {
            return error.PeerTypeRefused;
        }

        log.debug("handshake: accepted server type '{s}'", .{ready.socket_type});
    }

    /// End the connection from this side and wake a task blocked in `next`.
    /// `deinit` must run after every task that uses the client returns.
    pub fn hangUp(self: *Client) void {
        self.stream.shutdown(self.io, .both) catch {};
    }

    pub fn deinit(self: *Client) void {
        self.stream.close(self.io);
        self.gpa.free(self.read_buf);
        self.gpa.free(self.write_buf);
        self.gpa.free(self.payload_buf);
        self.gpa.free(self.frame_buf);
        self.gpa.free(self.pair_buf);
        self.* = undefined;
    }

    /// Send one client message. This returns the request id the message carried.
    pub fn send(self: *Client, msg: protocol.ClientMsg, session_id: u32) SendError!u32 {
        const request_id = self.next_request_id;
        self.next_request_id += 1;
        const body = self.frame_buf[zmtp_head_bytes..];
        const n = protocol.encodeClientMsg(msg, request_id, session_id, body);
        const head_len = zmtp.encodeFrameHead(n, false, self.frame_buf[0..zmtp_head_bytes]);
        try self.writer.interface.writeAll(self.frame_buf[0..head_len]);
        try self.writer.interface.writeAll(body[0..n]);
        try self.writer.interface.flush();
        return request_id;
    }

    pub const NextError = error{ ReadFailed, EndOfStream } || protocol.DecodeError || zmtp.Error;

    /// Block for the next server frame. Slices borrow `payload_buf`.
    pub fn next(self: *Client) NextError!Event {
        const r = &self.reader.interface;
        // Each pass reads one frame. This client sends no PING, so the first message ends the loop.
        while (true) {
            const head = try readFrameHead(r);
            if (head.command) {
                if (head.size > zmtp.max_ready_bytes) return error.MessageTooLarge;

                const command_body = self.payload_buf[0..@intCast(head.size)];
                try r.readSliceAll(command_body);
                _ = try zmtp.decodeCommand(command_body);
                continue;
            }

            if (head.more or head.size < protocol.header_bytes) return error.Malformed;
            if (head.size > zmtp.max_message_bytes) return error.MessageTooLarge;

            var header: [protocol.header_bytes]u8 = undefined;
            try r.readSliceAll(&header);
            const h = try protocol.decodeHeader(&header);
            const payload_len: usize = @intCast(head.size - protocol.header_bytes);
            if (h.payload_len != payload_len) return error.Malformed;

            const payload = self.payload_buf[0..payload_len];
            try r.readSliceAll(payload);
            return .{ .header = h, .msg = try protocol.decodeServerPayload(h.msg_type, payload) };
        }
    }

    // -- convenience: one request in flight on this connection --------------

    pub const HelloInfo = struct {
        max_seq: u32,
        max_sessions: u16,
        max_batch: u16,
        vocab_size: u32,
        eos_id: u32,
        think_mode: protocol.ThinkMode,
        think_budget: u32,
        /// The longest lease a keep gets. 0 is resumption off.
        resume_ttl_max_ms: u32,
        recommended_think: protocol.Sampling,
        recommended_plain: protocol.Sampling,
        /// A copy of the model name, because the frame text is borrowed.
        model_name_len: u8,
        model_name_buf: [limits.max_model_name_bytes]u8,

        comptime {
            std.debug.assert(limits.max_model_name_bytes <= std.math.maxInt(u8));
        }

        pub fn modelName(info: *const HelloInfo) []const u8 {
            return info.model_name_buf[0..info.model_name_len];
        }

        /// The card row of a turn that opens inside the think block, or of
        /// one that stays outside it.
        pub fn recommendedFor(info: *const HelloInfo, open: bool) protocol.Sampling {
            return if (open) info.recommended_think else info.recommended_plain;
        }
    };

    pub fn hello(self: *Client) ReplyError!HelloInfo {
        _ = try self.send(.hello, 0);
        const ev = try self.next();
        if (self.recordErr(ev.msg)) return error.ServerError;
        if (ev.msg != .hello_ack) return error.UnexpectedReply;

        self.eos_id = ev.msg.hello_ack.eos_id;
        return helloInfoFrom(ev.msg.hello_ack);
    }

    pub const Stats = struct {
        sessions_open: u16,
        sessions_decoding: u16,
        tokens_generated_total: u64,
        uptime_ms: u64,
    };

    pub fn stats(self: *Client) ReplyError!Stats {
        _ = try self.send(.stats, 0);
        const ev = try self.next();
        if (self.recordErr(ev.msg)) return error.ServerError;
        if (ev.msg != .stats_ack) return error.UnexpectedReply;

        return statsFrom(ev.msg.stats_ack);
    }

    pub fn ping(self: *Client) ReplyError!void {
        _ = try self.send(.ping, 0);
        const ev = try self.next();
        if (self.recordErr(ev.msg)) return error.ServerError;
        if (ev.msg != .pong) return error.UnexpectedReply;
    }

    /// The id of a new session and the positions it uses, from `session_opened`.
    pub const Opened = struct { id: u32, kv_pos: u32 };

    /// Opens a session with the system prompt prefilled on the server. Returns its id.
    pub fn openSession(
        self: *Client,
        system_prompt: []const u8,
        sampling: protocol.Sampling,
    ) ReplyError!u32 {
        return (try self.openSessionInfo(system_prompt, sampling)).id;
    }

    /// `openSession` with the `kv_pos` of `session_opened` too.
    pub fn openSessionInfo(
        self: *Client,
        system_prompt: []const u8,
        sampling: protocol.Sampling,
    ) ReplyError!Opened {
        _ = try self.send(
            .{ .session_open = .{ .sampling = sampling, .system_prompt = system_prompt } },
            0,
        );
        const ev = try self.next();
        if (self.recordErr(ev.msg)) return error.ServerError;
        if (ev.msg != .session_opened) return error.UnexpectedReply;

        return .{ .id = ev.header.session_id, .kv_pos = ev.msg.session_opened.kv_pos };
    }

    /// Fork an open session on the server. The fork shares the context but has its own sampling
    /// and seed. Returns the new session id.
    pub fn cloneSession(
        self: *Client,
        source_id: u32,
        sampling: protocol.Sampling,
    ) ReplyError!u32 {
        return (try self.cloneSessionInfo(source_id, sampling)).id;
    }

    /// `cloneSession` with the `kv_pos` of `session_opened` too.
    pub fn cloneSessionInfo(
        self: *Client,
        source_id: u32,
        sampling: protocol.Sampling,
    ) ReplyError!Opened {
        _ = try self.send(.{ .session_clone = .{ .sampling = sampling } }, source_id);
        const ev = try self.next();
        if (self.recordErr(ev.msg)) return error.ServerError;
        if (ev.msg != .session_opened) return error.UnexpectedReply;

        return .{ .id = ev.header.session_id, .kv_pos = ev.msg.session_opened.kv_pos };
    }

    /// Close a session and wait for `session_closed`. A close mid-turn sends no `done`.
    /// The frames queued for that turn arrive first, and this call skips them.
    pub fn closeSession(self: *Client, session_id: u32) ReplyError!void {
        _ = try self.send(.session_close, session_id);
        while (true) {
            const ev = try self.next();
            switch (ev.msg) {
                .token, .done => {},
                .session_closed => return,
                .err => {
                    _ = self.recordErr(ev.msg);
                    return error.ServerError;
                },
                else => return error.UnexpectedReply,
            }
        }
    }

    /// Feed kept pairs into an idle session and wait for `appended`. The pairs must fit one
    /// frame: `limits.max_append_pairs` pairs and `protocol.append_pairs_bytes_max` bytes.
    /// Returns the position after each pair. A `turn_too_long` refusal fed no pair.
    pub fn append(
        self: *Client,
        session_id: u32,
        pairs: []const protocol.Pair,
    ) ReplyError!protocol.Appended {
        const request_id = try self.sendAppend(session_id, pairs);
        const ev = try self.next();
        if (self.recordErr(ev.msg)) return error.ServerError;
        if (ev.msg != .appended or ev.header.request_id != request_id) return error.UnexpectedReply;
        if (ev.msg.appended.n != pairs.len) return error.UnexpectedReply;

        return ev.msg.appended;
    }

    /// Send kept pairs and return the request id. The caller reads the reply through `next`.
    /// The bounds of `append` hold.
    pub fn sendAppend(
        self: *Client,
        session_id: u32,
        pairs: []const protocol.Pair,
    ) SendError!u32 {
        std.debug.assert(pairs.len >= 1);
        std.debug.assert(pairs.len <= limits.max_append_pairs);
        std.debug.assert(protocol.appendPairsBytes(pairs) <= protocol.append_pairs_bytes_max);

        return self.send(
            .{ .session_append = protocol.encodeAppendPairs(pairs, self.pair_buf) },
            session_id,
        );
    }

    /// Send a turn. The caller then loops `next` for `token` and `done` events.
    pub fn generate(
        self: *Client,
        session_id: u32,
        text: []const u8,
        max_tokens: u32,
        sampling: ?protocol.Sampling,
    ) GenerateError!u32 {
        return self.generateWith(
            session_id,
            text,
            .{ .max_tokens = max_tokens, .sampling = sampling },
        );
    }

    /// The per-turn fields of a `generate`. A null field takes the session or server value.
    /// A sampling block replaces the session sampling, and its seed restarts the stream only with
    /// `reseed`.
    pub const TurnOptions = struct {
        max_tokens: u32,
        sampling: ?protocol.Sampling = null,
        /// Restart the sampling stream at the seed of `sampling`, so the turn replays.
        reseed: bool = false,
        /// The think switch of the turn.
        think_open: ?bool = null,
        /// Think tokens before the forced close, at most `limits.max_model_ctx`. 0 is unbounded.
        think_budget: ?u32 = null,
    };

    /// `generate` with every per-turn field of the wire. A text past
    /// `protocol.generate_text_bytes_max` is `error.TextTooLong`, and no frame goes out.
    pub fn generateWith(
        self: *Client,
        session_id: u32,
        text: []const u8,
        opts: TurnOptions,
    ) GenerateError!u32 {
        std.debug.assert(opts.max_tokens > 0);

        if (opts.think_budget) |b| std.debug.assert(b <= limits.max_model_ctx);
        if (opts.reseed) std.debug.assert(opts.sampling != null);
        if (text.len > protocol.generate_text_bytes_max) return error.TextTooLong;

        return self.send(.{ .generate = .{
            .max_tokens = opts.max_tokens,
            .sampling = opts.sampling,
            .text = text,
            .think_open = opts.think_open,
            .think_budget = opts.think_budget,
            .reseed = opts.reseed,
        } }, session_id);
    }

    /// The grant of a keep: the time to live and the token. A grant of 0 holds no lease.
    pub const Kept = @FieldType(protocol.ServerMsg, "kept");

    /// Keep `session_id` past a drop of this connection for at most `ttl_ms`, and wait for
    /// `kept`. A `ttl_ms` of 0 drops the lease. Store the token to resume on a new connection.
    pub fn keepSession(self: *Client, session_id: u32, ttl_ms: u32) ReplyError!Kept {
        const request_id = try self.send(.{ .session_keep = .{ .ttl_ms = ttl_ms } }, session_id);
        const ev = try self.replyTo(request_id);
        if (self.recordErr(ev.msg)) return error.ServerError;
        if (ev.msg != .kept or ev.header.session_id != session_id) return error.UnexpectedReply;

        return ev.msg.kept;
    }

    /// The session of a resume and its state at the moment it became idle.
    pub const ResumedSession = struct { id: u32, state: protocol.Resumed };

    /// Attach the session of `token` to this connection, and wait for `resumed`.
    /// The server answers when the session is idle. Store the new token of the reply.
    pub fn resumeSession(self: *Client, token: protocol.Token) ReplyError!ResumedSession {
        const request_id = try self.send(.{ .session_resume = .{ .token = token } }, 0);
        const ev = try self.replyTo(request_id);
        if (self.recordErr(ev.msg)) return error.ServerError;
        if (ev.msg != .resumed or ev.header.session_id == 0) return error.UnexpectedReply;

        return .{ .id = ev.header.session_id, .state = ev.msg.resumed };
    }

    /// The reply to `request_id`. A turn of the same session can still stream, and this call
    /// skips its `token` and `done` frames. Every other frame is an unexpected reply.
    fn replyTo(self: *Client, request_id: u32) !Event {
        while (true) {
            const ev = try self.next();
            if (ev.header.request_id == request_id) return ev;
            switch (ev.msg) {
                .token, .done => {},
                else => return error.UnexpectedReply,
            }
        }
    }

    /// Stop the turn in flight on `session_id`. A turn of that session must be in flight.
    /// See "Cancel" in the header.
    pub fn cancel(self: *Client, session_id: u32) SendError!u32 {
        return self.send(.cancel, session_id);
    }

    pub const Turn = struct {
        finish_reason: protocol.FinishReason,
        /// The token frames of the turn, the end-of-sequence frame included.
        n_tokens: usize,
        /// The text bytes in `text_out`. The copy stops at the first token that does not fit.
        text_len: usize,
        /// The text bytes of the whole reply. A cut reply has more than `text_len`.
        text_total: usize,
        kv_pos: u32,
        /// The server's count from `done`.
        gen_tokens: u32,
        /// The tokens inside the think block, from `done`.
        reasoning_tokens: u32,
        prefill_ms: f32,
        decode_ms: f32,
    };

    /// Collect one whole turn after `generate`, and return on `done`. See "Collected turns" in
    /// the header.
    pub fn collectTurn(self: *Client, ids_out: []u32, text_out: []u8) CollectError!Turn {
        return self.collectTurnWith(ids_out, text_out, NoTokenHook{});
    }

    /// `collectTurn` with a per-token hook. `on_token.onToken(token_id, channel, text)` runs once
    /// for each token frame, in order, the end-of-sequence frame included. The text borrows the
    /// payload buffer until the hook returns.
    pub fn collectTurnWith(
        self: *Client,
        ids_out: []u32,
        text_out: []u8,
        on_token: anytype,
    ) CollectError!Turn {
        var n_tokens: usize = 0;
        var text_len: usize = 0;
        var text_total: usize = 0;
        while (true) {
            const ev = try self.next();
            switch (ev.msg) {
                .token => |t| {
                    // Count the token and keep its id while the buffer has room.
                    on_token.onToken(t.token_id, t.channel, t.text);
                    if (n_tokens < ids_out.len) {
                        ids_out[n_tokens] = t.token_id;
                    }

                    n_tokens += 1;
                    if (self.eos_id != null and t.token_id == self.eos_id.?) continue;

                    // Copy the text only while no earlier token was cut.
                    if (text_len == text_total and text_len + t.text.len <= text_out.len) {
                        @memcpy(text_out[text_len..][0..t.text.len], t.text);
                        text_len += t.text.len;
                    }

                    text_total += t.text.len;
                },
                .done => |d| return .{
                    .finish_reason = d.finish_reason,
                    .n_tokens = n_tokens,
                    .text_len = text_len,
                    .text_total = text_total,
                    .kv_pos = d.kv_pos,
                    .gen_tokens = d.gen_tokens,
                    .reasoning_tokens = d.reasoning_tokens,
                    .prefill_ms = d.prefill_ms,
                    .decode_ms = d.decode_ms,
                },
                .err => {
                    _ = self.recordErr(ev.msg);
                    return error.ServerError;
                },
                else => return error.UnexpectedReply,
            }
        }
    }

    /// Ask the server to exit, and wait for the ack. A server that bars a wire shutdown answers
    /// `err{forbidden}` under this request id.
    pub fn shutdown(self: *Client) ReplyError!void {
        const request_id = try self.send(.shutdown, 0);
        while (true) {
            const ev = try self.next();
            if (ev.msg == .shutdown_ack) return;
            if (ev.msg == .err and ev.header.request_id == request_id) {
                _ = self.recordErr(ev.msg);
                return error.ServerError;
            }
            // In-flight turns can flush done{cancelled} first.
        }
    }

    /// Record `msg` into `last_error` when it is an `err` frame. Returns true when it records one.
    fn recordErr(self: *Client, msg: protocol.ServerMsg) bool {
        const e = switch (msg) {
            .err => |e| e,
            else => return false,
        };

        self.last_error = lastErrorFrom(e);
        return true;
    }

    fn lastErrorFrom(e: @FieldType(protocol.ServerMsg, "err")) LastError {
        var le: LastError = .{
            .code = e.code,
            .len = @intCast(@min(e.detail.len, LastError.detail_cap)),
            .truncated = e.detail.len > LastError.detail_cap,
            .buf = undefined,
        };
        @memcpy(le.buf[0..le.len], e.detail[0..le.len]);
        return le;
    }

    fn helloInfoFrom(m: @FieldType(protocol.ServerMsg, "hello_ack")) HelloInfo {
        // The decoder refuses a longer name.
        std.debug.assert(m.model_name.len <= limits.max_model_name_bytes);

        var info: HelloInfo = .{
            .max_seq = m.max_seq,
            .max_sessions = m.max_sessions,
            .max_batch = m.max_batch,
            .vocab_size = m.vocab_size,
            .eos_id = m.eos_id,
            .think_mode = m.think_mode,
            .think_budget = m.think_budget,
            .resume_ttl_max_ms = m.resume_ttl_max_ms,
            .recommended_think = m.recommended_think,
            .recommended_plain = m.recommended_plain,
            .model_name_len = @intCast(m.model_name.len),
            .model_name_buf = undefined,
        };
        @memcpy(info.model_name_buf[0..m.model_name.len], m.model_name);
        return info;
    }

    fn statsFrom(m: @FieldType(protocol.ServerMsg, "stats_ack")) Stats {
        return .{
            .sessions_open = m.sessions_open,
            .sessions_decoding = m.sessions_decoding,
            .tokens_generated_total = m.tokens_generated_total,
            .uptime_ms = m.uptime_ms,
        };
    }
};

// -- api --------------------------------------------------------------------

pub fn toOwned(ev: Event) OwnedEvent {
    const body: OwnedEvent.Body = switch (ev.msg) {
        .hello_ack => |m| .{ .hello_ack = Client.helloInfoFrom(m) },
        .session_opened => |m| .{ .session_opened = m },
        .token => |m| blk: {
            std.debug.assert(m.text.len <= limits.max_token_text_bytes);

            var b: OwnedEvent.Body = .{
                .token = .{
                    .token_id = m.token_id,
                    .channel = m.channel,
                    .len = @intCast(m.text.len),
                    .text = undefined,
                },
            };
            @memcpy(b.token.text[0..m.text.len], m.text);
            break :blk b;
        },
        .done => |m| .{ .done = m },
        .err => |m| .{ .err = Client.lastErrorFrom(m) },
        .stats_ack => |m| .{ .stats_ack = Client.statsFrom(m) },
        .pong => .pong,
        .session_closed => .session_closed,
        .shutdown_ack => .shutdown_ack,
        .appended => |m| .{ .appended = m },
        .kept => |m| .{ .kept = m },
        .resumed => |m| .{ .resumed = m },
    };

    return .{ .header = ev.header, .body = body };
}

// -- helpers ----------------------------------------------------------------

/// Reads the flags byte, then the rest of the head that the flags announce.
/// The head holds at most `zmtp_head_bytes`.
fn readFrameHead(r: *Io.Reader) (Io.Reader.Error || zmtp.Error)!zmtp.FrameHead {
    var head: [zmtp_head_bytes]u8 = undefined;
    head[0] = try r.takeByte();
    const n = zmtp.headLen(head[0]);
    std.debug.assert(n <= zmtp_head_bytes);
    try r.readSliceAll(head[1..n]);
    return zmtp.decodeFrameHead(head[0..n]);
}

// -- tests ------------------------------------------------------------------

/// Copies each hook call, because the text borrows the payload buffer.
const HookLog = struct {
    ids: [8]u32 = undefined,
    n: usize = 0,
    text: [64]u8 = undefined,
    text_len: usize = 0,
    channels: [8]protocol.Channel = undefined,

    pub fn onToken(
        self: *HookLog,
        token_id: u32,
        channel: protocol.Channel,
        text: []const u8,
    ) void {
        self.channels[self.n] = channel;
        self.ids[self.n] = token_id;
        self.n += 1;
        @memcpy(self.text[self.text_len..][0..text.len], text);
        self.text[self.text_len + text.len] = '|';
        self.text_len += text.len + 1;
    }
};

fn appendServerFrame(gpa: Allocator, out: *std.ArrayList(u8), msg: protocol.ServerMsg) !void {
    var body: [protocol.max_server_frame_bytes]u8 = undefined;
    const n = protocol.encodeServerMsg(msg, 2, 1, &body);
    var head: [zmtp_head_bytes]u8 = undefined;
    const head_len = zmtp.encodeFrameHead(n, false, &head);
    try out.appendSlice(gpa, head[0..head_len]);
    try out.appendSlice(gpa, body[0..n]);
}

test "collectTurnWith: the hook sees every token frame, the end-of-sequence frame included" {
    const gpa = std.testing.allocator;
    var frames: std.ArrayList(u8) = .empty;
    defer frames.deinit(gpa);

    const Sent = struct { id: u32, channel: protocol.Channel, text: []const u8 };
    const sent = [_]Sent{
        .{ .id = 5, .channel = .open_tag, .text = "<think>" },
        .{ .id = 6, .channel = .reasoning, .text = "ab" },
        .{ .id = 8, .channel = .close_tag, .text = "</think>" },
        .{ .id = 9, .channel = .content, .text = "ok" },
        .{ .id = 7, .channel = .content, .text = "<eos>" },
    };
    for (sent) |s| {
        try appendServerFrame(
            gpa,
            &frames,
            .{ .token = .{ .token_id = s.id, .channel = s.channel, .text = s.text } },
        );
    }

    try appendServerFrame(gpa, &frames, .{ .done = .{
        .finish_reason = .eos,
        .turn_prompt_tokens = 3,
        .gen_tokens = 5,
        .kv_pos = 7,
        .prefill_ms = 1,
        .decode_ms = 2,
        .reasoning_tokens = 1,
    } });

    // A turn reads only the reader, the payload buffer, and the end-of-sequence id.
    const payload = try gpa.alloc(u8, protocol.max_frame_payload_bytes);
    defer gpa.free(payload);

    var client: Client = undefined;
    client.reader.interface = .fixed(frames.items);
    client.payload_buf = payload;
    client.eos_id = 7;

    var hook: HookLog = .{};
    var ids: [8]u32 = undefined;
    var text: [64]u8 = undefined;
    const turn = try client.collectTurnWith(&ids, &text, &hook);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 8, 9, 7 }, hook.ids[0..hook.n]);
    try std.testing.expectEqualStrings(
        "<think>|ab|</think>|ok|<eos>|",
        hook.text[0..hook.text_len],
    );
    for (sent, hook.channels[0..hook.n]) |s, channel| {
        try std.testing.expectEqual(s.channel, channel);
    }

    // The hook changes nothing in the turn that `collectTurn` returns.
    try std.testing.expectEqual(@as(usize, 5), turn.n_tokens);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 8, 9, 7 }, ids[0..5]);
    try std.testing.expectEqualStrings("<think>ab</think>ok", text[0..turn.text_len]);
    try std.testing.expectEqual(turn.text_len, turn.text_total);
    try std.testing.expectEqual(@as(u32, 5), turn.gen_tokens);
    try std.testing.expectEqual(@as(u32, 1), turn.reasoning_tokens);
}

test "the public methods name their error sets" {
    const Row = struct { f: type, set: type };
    const rows = [_]Row{
        .{ .f = @TypeOf(Client.send), .set = Client.SendError },
        .{ .f = @TypeOf(Client.cancel), .set = Client.SendError },
        .{ .f = @TypeOf(Client.generate), .set = Client.GenerateError },
        .{ .f = @TypeOf(Client.hello), .set = Client.ReplyError },
        .{ .f = @TypeOf(Client.closeSession), .set = Client.ReplyError },
        .{ .f = @TypeOf(Client.shutdown), .set = Client.ReplyError },
        .{ .f = @TypeOf(Client.collectTurn), .set = Client.CollectError },
    };
    comptime for (rows) |row| {
        const returned = @typeInfo(row.f).@"fn".return_type.?;
        std.debug.assert(@typeInfo(returned).error_union.error_set == row.set);
    };
}

test "generateWith: a text past the frame bound is refused before any frame goes out" {
    const gpa = std.testing.allocator;
    const text = try gpa.alloc(u8, protocol.generate_text_bytes_max + 1);
    defer gpa.free(text);

    @memset(text, 'a');
    // The refusal reads no field of the client, so an unset client shows that no write ran.
    var client: Client = undefined;
    try std.testing.expectError(error.TextTooLong, client.generateWith(1, text, .{
        .max_tokens = 1,
        .sampling = protocol.Sampling.greedy,
        .reseed = true,
        .think_open = true,
        .think_budget = 1,
    }));
}
