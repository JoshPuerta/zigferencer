//! The binary wire protocol as a serve-engine adapter. `serve/engine.zig` owns the engine thread,
//! queues, sink, and connection lifecycle. This file supplies the binary transport: two tasks per
//! connection. A frame reader decodes `protocol.zig` into `Request` values, and a frame writer
//! encodes `OutEvent` values back. The connection uses ZMTP 3.x: the server answers a DEALER peer
//! as a ROUTER, and one ZMTP message carries one `protocol.zig` frame.
//!
//! Comptime. Generic over the engine and the codec, so the adapter builds against any engine type.
//!
//! Task roots. `io.concurrent` builds an argument tuple for the function it spawns, and it cannot
//! do that for an `anytype` parameter. So the spawned reader and writer live in a struct keyed on
//! the engine type (`BinaryTasks`). The helpers they call (`readerLoop`, `fatalFrame`) stay
//! `anytype`, because a caller invokes them and no spawn takes them.
//!
//! Handshake order. ZMTP lets each peer send its greeting and READY before it reads those of the
//! other peer. So the writer and the reader need no new synchronization.
//!
//! Loopback tests. They drive the whole adapter over 127.0.0.1: accept, frame decode, the queues,
//! the engine loop, the writer, and the shutdown handshake. The fake model is the scheduler tests'
//! one. Its chain is `next = (token + 1) % 8`, its codec maps `byte % 8`, a piece is '0' + id, and
//! eos is 7. A test skips when the sandbox forbids a loopback bind.

const std = @import("std");
const Io = std.Io;
const net = Io.net;
const Allocator = std.mem.Allocator;
const protocol = @import("protocol.zig");
const span = @import("../core/span.zig");
const zmtp = @import("zmtp.zig");
const event = @import("event.zig");
const chat = @import("../core/chat.zig");
const engine_mod = @import("engine.zig");
const Request = event.Request;
const OutEvent = event.OutEvent;

const log = std.log.scoped(.serve);

// -- constants --------------------------------------------------------------

/// One frame head plus the largest body the server sends. The PONG and ERROR command bodies are
/// smaller than that body.
const writer_frame_bytes: usize = zmtp.long_head_bytes + protocol.max_server_frame_bytes;

// -- types ------------------------------------------------------------------

/// Where to bind: TCP or a unix socket.
pub const Transport = engine_mod.Endpoint;

/// The reader maps these errors to a silent close or to a ZMTP ERROR.
const HandshakeError = Io.Reader.Error || zmtp.Error || error{ReadyExpected};

/// Two tasks per connection (reader + writer) over `protocol.zig` frames.
const BinaryAdapter = struct {
    pub const ConnExt = struct {
        /// One ZMTP message body: the 20-byte header and the payload.
        payload_buf: [zmtp.max_message_bytes]u8 = undefined,
        reader_future: Io.Future(void) = undefined,
        writer_future: Io.Future(void) = undefined,
    };

    /// Spawn the writer then the reader, and store their futures in `conn.ext`. A spawn failure
    /// leaves the stream closed and returns false, so the engine reclaims the slot.
    pub fn onAccept(engine: anytype, conn: anytype) bool {
        log.debug("accept: conn {d}", .{conn.id});
        const Tasks = BinaryTasks(@TypeOf(engine.*));
        conn.tasks_live.store(2, .release);
        conn.ext.writer_future = engine.io.concurrent(Tasks.writer, .{ engine, conn }) catch {
            conn.stream.close(engine.io);
            return false;
        };
        conn.ext.reader_future = engine.io.concurrent(Tasks.reader, .{ engine, conn }) catch {
            // The writer is up, so close its queue and let it shut the stream down.
            engine.closeOutput(conn);
            conn.ext.writer_future.await(engine.io);
            conn.stream.close(engine.io);
            return false;
        };
        return true;
    }

    /// Close the out-queue. The writer drains it and shuts the stream down, which wakes the
    /// reader, and both tasks exit.
    pub fn beginClose(engine: anytype, conn: anytype) void {
        log.debug("close: conn {d}", .{conn.id});
        engine.closeOutput(conn);
    }

    pub fn awaitConn(engine: anytype, conn: anytype) void {
        conn.ext.reader_future.await(engine.io);
        conn.ext.writer_future.await(engine.io);
    }

    /// Sends the greeting and READY of the ROUTER side, then the refusal as one message, and
    /// closes. It reads nothing, so a silent peer cannot stall the acceptor.
    pub fn refuse(io: Io, stream: *net.Stream) void {
        log.debug("refuse: connection slots full", .{});
        const ev = OutEvent.fail(0, 0, .connections_exhausted, "connection slots full");
        var frame: [writer_frame_bytes]u8 = undefined;
        var write_buf: [zmtp.greeting_bytes + 2 * writer_frame_bytes]u8 = undefined;
        var w = stream.writer(io, &write_buf);
        send: {
            writeOpening(&w.interface, &frame) catch break :send;
            writeEvent(&w.interface, &frame, &ev) catch break :send;
            w.interface.flush() catch break :send;
        }

        // A close over the unread peer handshake resets the connection. Shut the send side first,
        // so the peer reads the refusal and then the end of the stream.
        stream.shutdown(io, .send) catch {};
        stream.close(io);
    }
};

/// The serve engine with the binary transport.
pub fn Server(comptime Engine: type, comptime CodecT: type) type {
    return engine_mod.ServeEngine(Engine, CodecT, BinaryAdapter);
}

/// The per-connection task roots, concrete over the engine type.
fn BinaryTasks(comptime Eng: type) type {
    return struct {
        fn reader(engine: *Eng, conn: *Eng.Conn) void {
            span.nameThread("zf-read");
            readerLoop(engine, conn);
            // The engine must hear about the drop exactly once, even when the queue is briefly
            // full. A closed queue means teardown already runs.
            engine.submit(.{
                .conn = conn.id,
                .request_id = 0,
                .session_id = 0,
                .body = .conn_closed,
            }) catch {};
            engine.taskExit(conn);
        }

        fn writer(engine: *Eng, conn: *Eng.Conn) void {
            span.nameThread("zf-write");
            var write_buf: [4096]u8 = undefined;
            var w = conn.stream.writer(engine.io, &write_buf);
            var frame: [writer_frame_bytes]u8 = undefined;
            send: {
                // Send the opening first.
                writeOpening(&w.interface, &frame) catch break :send;
                w.interface.flush() catch break :send;
                log.debug("handshake: conn {d} sent greeting and READY", .{conn.id});
                while (true) {
                    const ev = conn.out_q.getOne(engine.io) catch break;
                    const sp = span.begin(.socket_write);
                    defer sp.end();

                    writeEvent(&w.interface, &frame, &ev) catch break;
                    w.interface.flush() catch break;
                    if (conn.parked_hint.swap(false, .acq_rel)) {
                        engine.wake(conn.id);
                    }
                }
            }

            conn.stream.shutdown(engine.io, .both) catch {};
            engine.taskExit(conn);
        }
    };
}

// -- helpers ----------------------------------------------------------------

fn readerLoop(engine: anytype, conn: anytype) void {
    // Read the peer handshake, or refuse it with a ZMTP ERROR.
    var read_buf: [4096]u8 = undefined;
    var r = conn.stream.reader(engine.io, &read_buf);
    const peer = readHandshake(&r.interface, &conn.ext.payload_buf) catch |e| {
        const reason: []const u8 = switch (e) {
            error.ReadFailed, error.EndOfStream => return,
            // A peer that is not ZMTP 3 cannot read an ERROR command.
            error.BadSignature, error.UnsupportedVersion => {
                log.debug("handshake: conn {d} closed on the greeting, {t}", .{ conn.id, e });
                return;
            },
            error.MechanismUnsupported => "NULL only",
            error.PeerTypeRefused => "DEALER only",
            error.ReadyExpected, error.Malformed, error.MessageTooLarge => "READY expected",
        };

        return fatalHandshake(engine, conn, reason);
    };
    log.debug("handshake: conn {d} accepted peer type '{s}'", .{ conn.id, peer });

    // Read one frame per pass. A command body holds at most `zmtp.max_ready_bytes`, and a message
    // at most `zmtp.max_message_bytes`.
    while (true) {
        const head = zmtp.readFrameHead(&r.interface) catch |e| switch (e) {
            error.ReadFailed, error.EndOfStream => return,
            else => return fatalFrame(engine, conn, .malformed_frame, "bad frame head"),
        };
        // The span starts after the frame head, so it holds no idle wait.
        const sp = span.begin(.socket_read);
        defer sp.end();

        // Answer a PING, and ignore the other commands.
        if (head.command) {
            if (head.size > zmtp.max_ready_bytes) {
                return fatalFrame(engine, conn, .frame_too_large, "command too large");
            }

            const body = conn.ext.payload_buf[0..@intCast(head.size)];
            r.interface.readSliceAll(body) catch return;
            const command = zmtp.decodeCommand(body) catch return fatalFrame(
                engine,
                conn,
                .malformed_frame,
                "bad command",
            );
            switch (command) {
                // A full queue holds the reader here until the writer drains.
                .ping => |ping| conn.out_q.putOne(
                    engine.io,
                    OutEvent.zmtpPong(ping.context),
                ) catch return,
                .ready, .pong, .err, .other => {},
            }

            continue;
        }

        // Read one whole message and check its header.
        if (head.more or head.size < protocol.header_bytes) {
            return fatalFrame(engine, conn, .malformed_frame, "not one frame");
        }
        if (head.size > zmtp.max_message_bytes) {
            return fatalFrame(engine, conn, .frame_too_large, "message too large");
        }

        const message = conn.ext.payload_buf[0..@intCast(head.size)];
        r.interface.readSliceAll(message) catch return;
        const h = protocol.decodeHeader(message[0..protocol.header_bytes]) catch |e| {
            return fatalFrame(engine, conn, decodeErrorCode(e), @errorName(e));
        };
        const payload = message[protocol.header_bytes..];
        if (h.payload_len != payload.len) {
            return fatalFrame(engine, conn, .malformed_frame, "payload_len mismatch");
        }

        // Decode the payload and hand the request to the engine.
        const msg = protocol.decodeClientPayload(h.msg_type, payload) catch |e| {
            return fatalFrame(engine, conn, decodeErrorCode(e), @errorName(e));
        };
        const req = toRequest(engine.gpa, conn.id, h, msg) catch |err| {
            log.debug("conn {d}: request failed: {t}", .{ conn.id, err });
            return;
        };
        engine.submit(req) catch {
            engine_mod.freeRequestText(engine.gpa, req);
            return;
        };
    }
}

/// Reads the peer greeting and READY, which hold at most `zmtp.greeting_bytes`, one frame head,
/// and `zmtp.max_ready_bytes`. Returns the peer socket type, which borrows `body_buf`.
fn readHandshake(r: *Io.Reader, body_buf: []u8) HandshakeError![]const u8 {
    std.debug.assert(body_buf.len >= zmtp.max_ready_bytes);

    try zmtp.decodeGreeting(try r.takeArray(zmtp.greeting_bytes));
    const head = try zmtp.readFrameHead(r);
    if (!head.command or head.size > zmtp.max_ready_bytes) return error.ReadyExpected;

    const body = body_buf[0..@intCast(head.size)];
    try r.readSliceAll(body);
    const ready = switch (try zmtp.decodeCommand(body)) {
        .ready => |ready| ready,
        .ping, .pong, .err, .other => return error.ReadyExpected,
    };

    if (!zmtp.peerAccepted(ready.socket_type)) return error.PeerTypeRefused;

    return ready.socket_type;
}

/// Codes 1-4: reply, then close THAT connection. The process never dies on input.
fn fatalFrame(engine: anytype, conn: anytype, code: protocol.ErrorCode, detail: []const u8) void {
    log.debug("frame: conn {d} closed, {t}: {s}", .{ conn.id, code, detail });
    const ev = OutEvent.fail(0, 0, code, detail);
    _ = conn.out_q.put(engine.io, &.{ev}, 0) catch {};
    engine.closeOutput(conn);
}

fn decodeErrorCode(e: protocol.DecodeError) protocol.ErrorCode {
    return switch (e) {
        error.BadMagic, error.PayloadMalformed => .malformed_frame,
        error.BadVersion => .unsupported_version,
        error.UnknownMsgType => .unknown_msg_type,
        error.FrameTooLarge => .frame_too_large,
    };
}

/// A refused handshake answers with one ZMTP ERROR, then closes THAT connection.
fn fatalHandshake(engine: anytype, conn: anytype, reason: []const u8) void {
    log.debug("handshake: conn {d} refused, {s}", .{ conn.id, reason });
    _ = conn.out_q.put(engine.io, &.{OutEvent.zmtpError(reason)}, 0) catch {};
    engine.closeOutput(conn);
}

/// Writes the greeting and the READY command of the ROUTER side. `frame` holds the READY frame.
fn writeOpening(w: *Io.Writer, frame: *[writer_frame_bytes]u8) Io.Writer.Error!void {
    var greeting: [zmtp.greeting_bytes]u8 = undefined;
    zmtp.encodeGreeting(&greeting);
    try w.writeAll(&greeting);
    const body = frame[zmtp.long_head_bytes..];
    const n = zmtp.encodeReady(zmtp.socket_type_router, body);
    const head_len = zmtp.encodeFrameHead(n, true, frame[0..zmtp.long_head_bytes]);
    try w.writeAll(frame[0..head_len]);
    try w.writeAll(body[0..n]);
}

/// Writes one event as one ZMTP frame. A ZMTP variant becomes a command frame, and every other
/// variant becomes a message that holds one `protocol.zig` frame.
fn writeEvent(
    w: *Io.Writer,
    frame: *[writer_frame_bytes]u8,
    ev: *const OutEvent,
) Io.Writer.Error!void {
    const body = frame[zmtp.long_head_bytes..];
    const command = ev.body == .zmtp_pong or ev.body == .zmtp_error;
    const n = switch (ev.body) {
        .zmtp_pong => |*m| zmtp.encodePong(m.context[0..m.len], body[0..32]),
        .zmtp_error => |*m| zmtp.encodeError(m.reason[0..m.len], body),
        else => encodeMessage(ev, body),
    };

    std.debug.assert(n <= protocol.max_server_frame_bytes);
    const head_len = zmtp.encodeFrameHead(n, command, frame[0..zmtp.long_head_bytes]);
    try w.writeAll(frame[0..head_len]);
    try w.writeAll(body[0..n]);
}

/// Lift a borrowed wire message into an owned Request. The text goes into the gpa, and the
/// scheduler frees it.
fn toRequest(
    gpa: Allocator,
    conn_id: u32,
    h: protocol.Header,
    msg: protocol.ClientMsg,
) Allocator.Error!Request {
    // Copy the message body into the request body.
    const body: Request.Body = switch (msg) {
        .hello => .hello,
        .cancel => .cancel,
        .session_close => .session_close,
        .stats => .stats,
        .ping => .ping,
        .shutdown => .shutdown,
        .session_open => |m| .{ .session_open = .{
            .sampling = m.sampling,
            .system_prompt = try gpa.dupe(u8, m.system_prompt),
        } },
        .session_clone => |m| .{ .session_clone = .{ .sampling = m.sampling } },
        .generate => |m| .{ .generate = .{
            .max_tokens = m.max_tokens,
            .sampling = m.sampling,
            .text = try gpa.dupe(u8, m.text),
            .think_open = m.think_open,
            .think_budget = m.think_budget,
            .reseed = m.reseed,
        } },
        .session_keep => |m| .{ .session_keep = .{ .ttl_ms = m.ttl_ms } },
        .session_resume => |m| .{ .session_resume = .{ .token = m.token } },
        .session_append => |m| .{ .session_append = .{
            .n = m.n,
            .bytes = try gpa.dupe(u8, m.bytes),
        } },
    };

    // Add the ids of the header.
    return .{
        .conn = conn_id,
        .request_id = h.request_id,
        .session_id = h.session_id,
        .body = body,
    };
}

/// OutEvent -> wire frame. The inline arrays become borrowed slices for the encode.
fn encodeMessage(ev: *const OutEvent, frame: []u8) usize {
    const msg: protocol.ServerMsg = switch (ev.body) {
        // Capture by pointer where a slice borrows from the event, because a by-value capture
        // dies at the end of the prong.
        .hello_ack => |*m| .{ .hello_ack = .{
            .protocol_version = protocol.protocol_version,
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
            .model_name = m.name[0..m.name_len],
        } },
        .session_opened => |m| .{ .session_opened = .{ .kv_pos = m.kv_pos } },
        .token => |*m| .{
            .token = .{
                .token_id = m.token_id,
                .channel = m.channel,
                .text = m.text[0..m.text_len],
            },
        },
        .done => |m| .{ .done = .{
            .finish_reason = m.finish_reason,
            .turn_prompt_tokens = m.turn_prompt_tokens,
            .gen_tokens = m.gen_tokens,
            .kv_pos = m.kv_pos,
            .prefill_ms = m.prefill_ms,
            .decode_ms = m.decode_ms,
            .reasoning_tokens = m.reasoning_tokens,
        } },
        .err => |*m| .{ .err = .{ .code = m.code, .detail = m.detail[0..m.detail_len] } },
        .stats_ack => |m| .{ .stats_ack = .{
            .sessions_open = m.sessions_open,
            .sessions_decoding = m.sessions_decoding,
            .tokens_generated_total = m.tokens_generated_total,
            .uptime_ms = m.uptime_ms,
        } },
        .pong => .pong,
        .session_closed => .session_closed,
        .shutdown_ack => .shutdown_ack,
        .appended => |m| .{ .appended = m },
        .kept => |m| .{ .kept = .{ .granted_ms = m.granted_ms, .token = m.token } },
        .resumed => |m| .{ .resumed = m },
        // `writeEvent` encodes these as command frames.
        .zmtp_pong, .zmtp_error => unreachable,
    };

    return protocol.encodeServerMsg(msg, ev.request_id, ev.session_id, frame);
}

// -- tests ------------------------------------------------------------------

const Sessions = @import("../runtime/session.zig").Sessions;
const StepRow = @import("../runtime/step_row.zig").StepRow;
const Client = @import("client.zig").Client;

const test_fakes = @import("../runtime/test_fakes.zig");
const test_vocab = test_fakes.vocab;

const TestRunner = struct {
    n_slots: u32,
    max_seq: u32,
    logits: []f32,

    pub fn maxStepRows(self: *const TestRunner) u32 {
        return self.n_slots;
    }

    pub fn stepBatch(self: *TestRunner, rows: []const StepRow) !void {
        test_fakes.writeCannedLogits(self.logits, rows);
    }
};

const FakeCodec = struct {
    piece_table: [test_vocab][1]u8,

    fn initFake() FakeCodec {
        var t: [test_vocab][1]u8 = undefined;
        for (&t, 0..) |*p, i| {
            p[0] = @intCast('0' + i);
        }

        return .{ .piece_table = t };
    }

    pub fn turnTokens(
        self: *const FakeCodec,
        gpa: Allocator,
        turn: event.TurnKind,
        text: []const u8,
        plain: []const chat.PlainRange,
        open: bool,
    ) ![]u32 {
        _ = self;
        _ = turn;
        _ = plain;
        _ = open;
        const ids = try gpa.alloc(u32, text.len);
        for (text, 0..) |b, i| {
            ids[i] = @intCast(b % test_vocab);
        }

        return ids;
    }

    pub fn pairTokens(
        self: *const FakeCodec,
        gpa: Allocator,
        order: chat.PairOrder,
        user: []const u8,
        assistant: []const u8,
    ) ![]u32 {
        _ = self;
        _ = order;
        const ids = try gpa.alloc(u32, user.len + assistant.len);
        for (user, 0..) |b, i| {
            ids[i] = @intCast(b % test_vocab);
        }

        for (assistant, 0..) |b, i| {
            ids[user.len + i] = @intCast(b % test_vocab);
        }

        return ids;
    }

    pub fn piece(self: *const FakeCodec, id: u32) []const u8 {
        return &self.piece_table[id];
    }
};

const TestServer = Server(Sessions(TestRunner), FakeCodec);

fn runServerTask(server: *TestServer) void {
    server.run() catch {};
}

test "loopback: hello, system-prompt session, one turn, clean shutdown" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    var runner: TestRunner = .{
        .n_slots = 2,
        .max_seq = 64,
        .logits = try arena.alloc(f32, 2 * test_vocab),
    };
    @memset(runner.logits, 0);
    var engine = try Sessions(TestRunner).init(arena, &runner);
    const codec = FakeCodec.initFake();

    var server = TestServer.init(arena, gpa, io, &engine, &codec, .{
        .max_sessions = 2,
        .max_batch = 2,
        .max_seq = 64,
        .max_step_rows = 2,
        .allow_shutdown = true,
        .model_name = "fake",
        .vocab_size = test_vocab,
        .eos_id = 7,
    }, .{ .tcp = .{} }) catch return error.SkipZigTest;

    var run_future = try io.concurrent(runServerTask, .{&server});

    var client = try Client.connectTcp(gpa, io, server.boundAddress());
    defer client.deinit();

    const info = try client.hello();
    try std.testing.expectEqual(@as(u32, 64), info.max_seq);
    try std.testing.expectEqual(@as(u32, 7), info.eos_id);
    try std.testing.expectEqual(@as(u16, 2), info.max_sessions);

    const sid = try client.openSession("abc", protocol.Sampling.greedy);

    // 'x' -> token 0. Greedy chain 1, 2, 3 capped by max_tokens = 3.
    _ = try client.generate(sid, "x", 3, null);
    var ids: [8]u32 = undefined;
    var text: [64]u8 = undefined;
    const turn = try client.collectTurn(&ids, &text);
    try std.testing.expectEqual(protocol.FinishReason.max_tokens, turn.finish_reason);
    try std.testing.expectEqual(@as(usize, 3), turn.n_tokens);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, ids[0..3]);
    try std.testing.expectEqualStrings("123", text[0..turn.text_len]);
    // Prompt "abc" (3) + turn "x" (1) + two of the three sampled tokens. The third commits on
    // the next step.
    try std.testing.expectEqual(@as(u32, 6), turn.kv_pos);

    // EOS path: feeding byte 6 makes the first sampled token 7 == eos.
    _ = try client.generate(sid, "\x06", 10, null);
    const turn2 = try client.collectTurn(&ids, &text);
    try std.testing.expectEqual(protocol.FinishReason.eos, turn2.finish_reason);

    // A refused request gives the typed error, plus the recorded code and detail.
    _ = try client.generate(999, "x", 1, null);
    try std.testing.expectError(error.ServerError, client.collectTurn(&ids, &text));
    const le = client.last_error.?;
    try std.testing.expectEqual(protocol.ErrorCode.unknown_session, le.code);
    try std.testing.expectEqualStrings("no such session", le.detail());
    try std.testing.expectEqual(false, le.truncated);

    // An append returns the position after each pair. The fake codec maps each byte to one id.
    const kept = try client.openSession("", protocol.Sampling.greedy);
    const appended = try client.append(
        kept,
        &.{ .{ .user = "ab", .assistant = "c" }, .{ .user = "d", .assistant = "" } },
    );
    try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, appended.positions());
    try client.closeSession(kept);

    try client.shutdown();
    run_future.await(io);
}

test "loopback: keep, hang up, resume, and the next turn equals an undropped twin" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    var runner: TestRunner = .{
        .n_slots = 3,
        .max_seq = 64,
        .logits = try arena.alloc(f32, 3 * test_vocab),
    };
    @memset(runner.logits, 0);
    var engine = try Sessions(TestRunner).init(arena, &runner);
    const codec = FakeCodec.initFake();

    var server = TestServer.init(arena, gpa, io, &engine, &codec, .{
        .max_sessions = 3,
        .max_batch = 3,
        .max_seq = 64,
        .max_step_rows = 3,
        .allow_shutdown = true,
        .model_name = "fake",
        .vocab_size = test_vocab,
        .eos_id = 7,
        .resume_ttl_max_ms = 60_000,
    }, .{ .tcp = .{} }) catch return error.SkipZigTest;
    var run_future = try io.concurrent(runServerTask, .{&server});

    var ids: [8]u32 = undefined;
    var text: [64]u8 = undefined;

    // The twin runs the same two turns on one connection with no drop.
    var twin = try Client.connectTcp(gpa, io, server.boundAddress());
    defer twin.deinit();

    const info = try twin.hello();
    try std.testing.expectEqual(@as(u32, 60_000), info.resume_ttl_max_ms);
    const tsid = try twin.openSession("abc", protocol.Sampling.greedy);
    _ = try twin.generate(tsid, "x", 3, null);
    const t1 = try twin.collectTurn(&ids, &text);

    // The first client runs turn 1, keeps the session, and hangs up.
    var token: protocol.Token = undefined;
    var request_id: u32 = undefined;
    {
        var first = try Client.connectTcp(gpa, io, server.boundAddress());
        defer first.deinit();

        _ = try first.hello();
        const sid = try first.openSession("abc", protocol.Sampling.greedy);
        request_id = try first.generate(sid, "x", 3, null);
        const turn = try first.collectTurn(&ids, &text);
        try std.testing.expectEqual(t1.kv_pos, turn.kv_pos);
        const kept = try first.keepSession(sid, 120_000);
        try std.testing.expectEqual(@as(u32, 60_000), kept.granted_ms);
        token = kept.token;
        first.hangUp();
    }

    // A new connection resumes at the position of the twin after turn 1.
    var second = try Client.connectTcp(gpa, io, server.boundAddress());
    defer second.deinit();

    _ = try second.hello();
    const resumed = try second.resumeSession(token);
    try std.testing.expectEqual(t1.kv_pos, resumed.state.kv_pos);
    try std.testing.expectEqual(@as(u32, 1), resumed.state.turns_done);
    try std.testing.expectEqual(
        protocol.LastDone{
            .request_id = request_id,
            .finish = .max_tokens,
            .gen_tokens = 3,
            .reasoning_tokens = 0,
        },
        resumed.state.last.?,
    );
    try std.testing.expect(!std.mem.eql(u8, &token, &resumed.state.token));

    // The old token is spent.
    try std.testing.expectError(error.ServerError, second.resumeSession(token));
    try std.testing.expectEqual(protocol.ErrorCode.unknown_session, second.last_error.?.code);
    try std.testing.expectEqualStrings("no such lease", second.last_error.?.detail());

    // Turn 2 on the resumed session equals turn 2 of the twin, in ids and position.
    _ = try second.generate(resumed.id, "x", 3, null);
    var got: [8]u32 = undefined;
    const r2 = try second.collectTurn(&got, &text);
    _ = try twin.generate(tsid, "x", 3, null);
    const t2 = try twin.collectTurn(&ids, &text);
    try std.testing.expectEqual(t2.n_tokens, r2.n_tokens);
    try std.testing.expectEqualSlices(u32, ids[0..t2.n_tokens], got[0..r2.n_tokens]);
    try std.testing.expectEqual(t2.kv_pos, r2.kv_pos);

    try second.closeSession(resumed.id);
    try twin.shutdown();
    run_future.await(io);
}
