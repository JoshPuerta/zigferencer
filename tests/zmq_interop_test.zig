//! The interop oracle for the serve wire. The reference client is a port of
//! the protocol, so a test that uses it alone proves only that the two ports
//! agree. A real libzmq DEALER proves that the server speaks the ZeroMQ wire
//! that other bindings speak. Each test drives the server through the system
//! libzmq and compares the stream with the stream of the reference client.
//!
//! The tests open `libzmq.so.5` at run time, so the build needs no ZeroMQ
//! headers or link step. A test skips where the library is absent, and it
//! logs the libzmq version it drove at the info level. Every test also
//! returns `error.SkipZigTest` when no fixture exists.

const std = @import("std");
const z = @import("zigferencer");
const protocol = z.serve_protocol;
const Client = z.serve_client.Client;
const Event = z.serve_client.Event;
const harness = @import("serve_harness.zig");
const Harness = harness.Harness;
const Stream = harness.Stream;
const max_seq = harness.max_seq;
const turn_tokens_max = harness.turn_tokens_max;
const prompts = harness.prompts;
const oneTurn = harness.oneTurn;
const expectSameStream = harness.expectSameStream;

// -- fixtures ---------------------------------------------------------------

/// The values of `zmq.h` in libzmq 4.x.
const zmq_dealer: c_int = 5;
const zmq_linger: c_int = 17;
const zmq_rcvtimeo: c_int = 27;
const zmq_sndtimeo: c_int = 28;

/// A wire bug fails the test after this wait instead of a hang. A greedy
/// turn of `turn_tokens_max` tokens on the fixture ends well inside it.
const io_timeout_ms: c_int = 30_000;

/// The largest message either side sends.
const max_message_bytes: usize = z.serve_zmtp.max_message_bytes;

/// The function pointers the tests call, looked up in the system libzmq.
/// Every field after `lib` is the libzmq symbol `zmq_<field name>`.
const Zmq = struct {
    lib: std.DynLib,
    version: *const fn (*c_int, *c_int, *c_int) callconv(.c) void,
    ctx_new: *const fn () callconv(.c) ?*anyopaque,
    ctx_term: *const fn (*anyopaque) callconv(.c) c_int,
    socket: *const fn (*anyopaque, c_int) callconv(.c) ?*anyopaque,
    close: *const fn (*anyopaque) callconv(.c) c_int,
    setsockopt: *const fn (*anyopaque, c_int, *const anyopaque, usize) callconv(.c) c_int,
    connect: *const fn (*anyopaque, [*:0]const u8) callconv(.c) c_int,
    send: *const fn (*anyopaque, *const anyopaque, usize, c_int) callconv(.c) c_int,
    recv: *const fn (*anyopaque, *anyopaque, usize, c_int) callconv(.c) c_int,
    errno: *const fn () callconv(.c) c_int,
    strerror: *const fn (c_int) callconv(.c) [*:0]const u8,

    /// Null when the library is absent. A library without one of the symbols
    /// is not the libzmq the tests target, and that fails the test.
    fn open() !?Zmq {
        var lib = std.DynLib.open("libzmq.so.5") catch return null;
        errdefer lib.close();
        var self: Zmq = undefined;
        self.lib = lib;
        inline for (@typeInfo(Zmq).@"struct".fields[1..]) |field| {
            @field(self, field.name) = lib.lookup(field.type, "zmq_" ++ field.name) orelse {
                std.debug.print("libzmq.so.5 has no symbol zmq_{s}\n", .{field.name});
                return error.ZmqSymbolMissing;
            };
        }
        return self;
    }

    fn deinit(self: *Zmq) void {
        self.lib.close();
        self.* = undefined;
    }

    fn fail(self: *const Zmq, what: []const u8) error{ZmqCallFailed} {
        self.report(what);
        return error.ZmqCallFailed;
    }

    /// A teardown path has no error to return, so it reports the failed call
    /// alone.
    fn report(self: *const Zmq, what: []const u8) void {
        const code = self.errno();
        std.debug.print("{s} failed: {s}\n", .{ what, std.mem.span(self.strerror(code)) });
    }

    fn logVersion(self: *const Zmq) void {
        var major: c_int = 0;
        var minor: c_int = 0;
        var patch: c_int = 0;
        self.version(&major, &minor, &patch);
        std.log.info("driving libzmq {d}.{d}.{d}", .{ major, minor, patch });
    }
};

/// One libzmq DEALER on its own context.
const Dealer = struct {
    zmq: *const Zmq,
    ctx: *anyopaque,
    sock: *anyopaque,
    /// A received message borrows this buffer until the next `recv`, so the
    /// caller copies what it keeps, as with `Client.next`.
    buf: [max_message_bytes]u8 = undefined,

    fn init(zmq: *const Zmq, address: [:0]const u8) !Dealer {
        const ctx = zmq.ctx_new() orelse return zmq.fail("zmq_ctx_new");
        errdefer _ = zmq.ctx_term(ctx);
        const sock = zmq.socket(ctx, zmq_dealer) orelse return zmq.fail("zmq_socket");
        errdefer _ = zmq.close(sock);
        // Linger 0 lets `zmq_ctx_term` return at once after a failed test
        // that leaves messages unsent.
        try setInt(zmq, sock, zmq_linger, 0);
        try setInt(zmq, sock, zmq_rcvtimeo, io_timeout_ms);
        try setInt(zmq, sock, zmq_sndtimeo, io_timeout_ms);
        if (zmq.connect(sock, address.ptr) != 0) return zmq.fail("zmq_connect");
        return .{ .zmq = zmq, .ctx = ctx, .sock = sock };
    }

    fn deinit(self: *Dealer) void {
        if (self.zmq.close(self.sock) != 0) self.zmq.report("zmq_close");
        if (self.zmq.ctx_term(self.ctx) != 0) self.zmq.report("zmq_ctx_term");
        self.* = undefined;
    }

    /// Send one request as one single-part message.
    fn send(self: *Dealer, msg: protocol.ClientMsg, request_id: u32, session_id: u32) !void {
        var out: [max_message_bytes]u8 = undefined;
        const n = protocol.encodeClientMsg(msg, request_id, session_id, &out);
        const sent = self.zmq.send(self.sock, &out, n, 0);
        if (sent < 0) return self.zmq.fail("zmq_send");
        try std.testing.expectEqual(n, @as(usize, @intCast(sent)));
    }

    /// Receive one reply. Slices in the message borrow `buf`.
    fn recv(self: *Dealer) !Event {
        const got = self.zmq.recv(self.sock, &self.buf, self.buf.len, 0);
        if (got < 0) return self.zmq.fail("zmq_recv");
        const n: usize = @intCast(got);
        // libzmq reports the full size of a message that it cut to the
        // buffer, and the server never sends a message above the bound.
        try std.testing.expect(n <= self.buf.len);
        if (n < protocol.header_bytes) return error.MessageTooShort;
        const h = try protocol.decodeHeader(self.buf[0..protocol.header_bytes]);
        try std.testing.expectEqual(n - protocol.header_bytes, h.payload_len);
        const msg = try protocol.decodeServerPayload(h.msg_type, self.buf[protocol.header_bytes..n]);
        return .{ .header = h, .msg = msg };
    }

    /// Read one turn up to its `done`. Each turn sends at most
    /// `turn_tokens_max` tokens and one `done`.
    fn readTurn(self: *Dealer, request_id: u32, session_id: u32) !Stream {
        var s: Stream = .{};
        var frames: usize = 0;
        while (frames < turn_tokens_max + 1) : (frames += 1) {
            const ev = try self.recv();
            try std.testing.expectEqual(request_id, ev.header.request_id);
            try std.testing.expectEqual(session_id, ev.header.session_id);
            switch (ev.msg) {
                .token => |t| {
                    try std.testing.expect(s.n < s.ids.len);
                    s.ids[s.n] = t.token_id;
                    s.n += 1;
                },
                .done => |d| {
                    s.reason = d.finish_reason;
                    return s;
                },
                else => return error.UnexpectedReply,
            }
        }
        return error.TurnNeverEnded;
    }

    fn hello(self: *Dealer, request_id: u32) !void {
        try self.send(.hello, request_id, 0);
        const ev = try self.recv();
        try std.testing.expect(ev.msg == .hello_ack);
        try std.testing.expectEqual(request_id, ev.header.request_id);
        try std.testing.expectEqual(max_seq, ev.msg.hello_ack.max_seq);
    }

    fn openGreedy(self: *Dealer, request_id: u32) !u32 {
        try self.send(.{ .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = "" } }, request_id, 0);
        const ev = try self.recv();
        try std.testing.expect(ev.msg == .session_opened);
        try std.testing.expectEqual(request_id, ev.header.request_id);
        try std.testing.expect(ev.header.session_id != 0);
        return ev.header.session_id;
    }

    fn generate(self: *Dealer, request_id: u32, session_id: u32, prompt: []const u8) !void {
        try self.send(.{ .generate = .{ .max_tokens = turn_tokens_max, .sampling = null, .text = prompt } }, request_id, session_id);
    }

    fn closeSession(self: *Dealer, request_id: u32, session_id: u32) !void {
        try self.send(.session_close, request_id, session_id);
        const ev = try self.recv();
        try std.testing.expect(ev.msg == .session_closed);
        try std.testing.expectEqual(request_id, ev.header.request_id);
        try std.testing.expectEqual(session_id, ev.header.session_id);
    }
};

fn setInt(zmq: *const Zmq, sock: *anyopaque, option: c_int, value: c_int) !void {
    if (zmq.setsockopt(sock, option, &value, @sizeOf(c_int)) != 0) return zmq.fail("zmq_setsockopt");
}

/// A prompt alone on a fresh session of the reference client, closed after.
fn soloReference(client: *Client, prompt: []const u8) !Stream {
    const sid = try client.openSession("", protocol.Sampling.greedy);
    const s = try oneTurn(client, sid, prompt);
    try std.testing.expect(s.n > 0);
    try client.closeSession(sid);
    return s;
}

/// One whole greedy turn through a DEALER: hello, open, generate, close.
fn dealerTurn(dealer: *Dealer, prompt: []const u8) !Stream {
    try dealer.hello(1);
    const sid = try dealer.openGreedy(2);
    try dealer.generate(3, sid, prompt);
    const s = try dealer.readTurn(3, sid);
    try dealer.closeSession(4, sid);
    return s;
}

fn tcpEndpoint(buf: []u8, h: *const Harness) ![:0]const u8 {
    return std.fmt.bufPrintZ(buf, "tcp://127.0.0.1:{d}", .{h.address.getPort()});
}

// -- tests ------------------------------------------------------------------

test "a libzmq DEALER drives the server over tcp" {
    var zmq = (try Zmq.open()) orelse return error.SkipZigTest;
    defer zmq.deinit();
    zmq.logVersion();

    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();
    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);
    const reference = try soloReference(&client, prompts[0]);

    var endpoint_buf: [64]u8 = undefined;
    var dealer = try Dealer.init(&zmq, try tcpEndpoint(&endpoint_buf, &h));
    defer dealer.deinit();
    try expectSameStream(reference, try dealerTurn(&dealer, prompts[0]));
}

test "a libzmq DEALER drives the server over ipc" {
    var zmq = (try Zmq.open()) orelse return error.SkipZigTest;
    defer zmq.deinit();
    zmq.logVersion();

    var h = try Harness.boot(std.testing.allocator, std.testing.io, .unix, .{});
    defer h.deinit();
    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);
    const reference = try soloReference(&client, prompts[0]);

    // The socket path is relative to the cwd, and libzmq resolves an `ipc://`
    // path against the same cwd.
    var endpoint_buf: [16 + std.Io.net.UnixAddress.max_len]u8 = undefined;
    const endpoint = try std.fmt.bufPrintZ(&endpoint_buf, "ipc://{s}", .{h.path_buf[0..h.path_len]});
    var dealer = try Dealer.init(&zmq, endpoint);
    defer dealer.deinit();
    try expectSameStream(reference, try dealerTurn(&dealer, prompts[0]));
}

test "two libzmq DEALERs hold sessions at once and each stream equals its reference" {
    var zmq = (try Zmq.open()) orelse return error.SkipZigTest;
    defer zmq.deinit();
    zmq.logVersion();

    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{ .max_sessions = 4, .max_batch = 4 });
    defer h.deinit();
    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);
    var reference: [prompts.len]Stream = undefined;
    for (&reference, prompts) |*ref, prompt| ref.* = try soloReference(&client, prompt);

    var endpoint_buf: [64]u8 = undefined;
    const endpoint = try tcpEndpoint(&endpoint_buf, &h);
    var dealers: [prompts.len]Dealer = undefined;
    var n_live: usize = 0;
    defer for (dealers[0..n_live]) |*d| d.deinit();
    while (n_live < prompts.len) : (n_live += 1) dealers[n_live] = try Dealer.init(&zmq, endpoint);

    var sids: [prompts.len]u32 = undefined;
    for (&dealers, &sids) |*d, *sid| {
        try d.hello(1);
        sid.* = try d.openGreedy(2);
    }
    // The test sends both turns before it reads any frame, so the two
    // sessions share the batch.
    for (&dealers, sids, prompts) |*d, sid, prompt| try d.generate(3, sid, prompt);
    for (&dealers, sids, reference) |*d, sid, ref| {
        try expectSameStream(ref, try d.readTurn(3, sid));
        try d.closeSession(4, sid);
    }
}

test "a libzmq DEALER appends kept pairs and reads the position after each" {
    var zmq = (try Zmq.open()) orelse return error.SkipZigTest;
    defer zmq.deinit();
    zmq.logVersion();

    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();
    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    // The reference client measures the cost of each pair on its own session.
    const pairs = [_]protocol.Pair{
        .{ .user = prompts[0], .assistant = " the hills." },
        .{ .user = prompts[1], .assistant = " a dragon." },
    };
    const ref_sid = try client.openSession("", protocol.Sampling.greedy);
    const ref = try client.append(ref_sid, &pairs);
    try client.closeSession(ref_sid);

    var endpoint_buf: [64]u8 = undefined;
    var dealer = try Dealer.init(&zmq, try tcpEndpoint(&endpoint_buf, &h));
    defer dealer.deinit();
    try dealer.hello(1);
    const sid = try dealer.openGreedy(2);
    var pair_buf: [256]u8 = undefined;
    try dealer.send(.{ .session_append = protocol.encodeAppendPairs(&pairs, &pair_buf) }, 3, sid);
    const ev = try dealer.recv();
    try std.testing.expectEqual(@as(u32, 3), ev.header.request_id);
    try std.testing.expectEqual(sid, ev.header.session_id);
    try std.testing.expect(ev.msg == .appended);
    try std.testing.expectEqualSlices(u32, ref.positions(), ev.msg.appended.positions());
    try std.testing.expect(ev.msg.appended.positions()[0] > 0);
    try dealer.closeSession(4, sid);
}

test "a libzmq DEALER that speaks version 2 gets unsupported_version" {
    var zmq = (try Zmq.open()) orelse return error.SkipZigTest;
    defer zmq.deinit();
    zmq.logVersion();

    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();
    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    var endpoint_buf: [64]u8 = undefined;
    var dealer = try Dealer.init(&zmq, try tcpEndpoint(&endpoint_buf, &h));
    defer dealer.deinit();
    var out: [protocol.header_bytes]u8 = undefined;
    const n = protocol.encodeClientMsg(.hello, 1, 0, &out);
    std.mem.writeInt(u16, out[4..6], 2, .little);
    const sent = zmq.send(dealer.sock, &out, n, 0);
    try std.testing.expectEqual(@as(c_int, @intCast(n)), sent);
    const ev = try dealer.recv();
    try std.testing.expect(ev.msg == .err);
    try std.testing.expectEqual(protocol.ErrorCode.unsupported_version, ev.msg.err.code);
}

test "a libzmq DEALER keeps a session, closes, and a new DEALER resumes it" {
    var zmq = (try Zmq.open()) orelse return error.SkipZigTest;
    defer zmq.deinit();
    zmq.logVersion();

    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{ .resume_ttl_max_ms = 60_000 });
    defer h.deinit();
    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    // The undropped twin runs both turns on the reference client.
    const twin = try client.openSession("", protocol.Sampling.greedy);
    _ = try oneTurn(&client, twin, prompts[0]);
    const want = try oneTurn(&client, twin, prompts[1]);
    try client.closeSession(twin);

    var endpoint_buf: [64]u8 = undefined;
    const endpoint = try tcpEndpoint(&endpoint_buf, &h);
    var token: protocol.Token = undefined;
    var sid: u32 = undefined;
    {
        var first = try Dealer.init(&zmq, endpoint);
        defer first.deinit();
        try first.hello(1);
        sid = try first.openGreedy(2);
        try first.generate(3, sid, prompts[0]);
        _ = try first.readTurn(3, sid);
        try first.send(.{ .session_keep = .{ .ttl_ms = 30_000 } }, 4, sid);
        const ev = try first.recv();
        try std.testing.expectEqual(@as(u32, 4), ev.header.request_id);
        try std.testing.expect(ev.msg == .kept);
        try std.testing.expectEqual(@as(u32, 30_000), ev.msg.kept.granted_ms);
        token = ev.msg.kept.token;
    }

    var second = try Dealer.init(&zmq, endpoint);
    defer second.deinit();
    try second.hello(1);
    try second.send(.{ .session_resume = .{ .token = token } }, 2, 0);
    const ev = try second.recv();
    try std.testing.expectEqual(@as(u32, 2), ev.header.request_id);
    try std.testing.expectEqual(sid, ev.header.session_id);
    try std.testing.expect(ev.msg == .resumed);
    try std.testing.expectEqual(@as(u32, 1), ev.msg.resumed.turns_done);
    try second.generate(3, sid, prompts[1]);
    try expectSameStream(want, try second.readTurn(3, sid));
    try second.closeSession(4, sid);
}
