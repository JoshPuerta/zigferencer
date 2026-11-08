//! End-to-end serve gate over a real model and real sockets. Each test boots its own in-process
//! server through `serve_harness.zig` and drives it through the reference client.
//!
//! Comptime. `keptPairsMatchLive` takes the harness type, so each runner family shares one body.
//!
//! Batch equality. The load-bearing assertion is exact. A turn that the server runs while other
//! sessions decode in the same batch produces the same token stream the turn produces alone. This
//! holds on one connection and across four at once. Batched serving is sequential serving, seen
//! from the wire.
//!
//! First test. It also checks that context_full surfaces as a `done` and not as an error. It checks
//! that a close in the middle of a generate leaves the server healthy and the slot reusable. It
//! checks that the shutdown handshake is clean.
//!
//! Client surface. The other tests drive every client method against a live server. They cover
//! ping, stats, open, clone, and close. They cover cancel in prefill, in decode, on an idle
//! session, and on an unknown one. They check the unix transport against TCP and the connection
//! cap. They check the per-connection session share, the sweep after a hang-up, and a wire shutdown
//! that the server refuses.
//!
//! Cancel in prefill. The spare-rows rule keeps prefill at one row per decode slot, so the prompt
//! takes many ticks. With no decoder, one tick feeds `max_batch - 0 = 3` prompt rows, capped by
//! `max_step_rows = 3`. The 41 tokens of the prompt need ceil(41 / 3) = 14 ticks before the first
//! token. The cancel is one frame behind the generate on the same connection, and the reader
//! submits each frame as soon as it decodes it. The engine takes up to 16 queued requests per pass,
//! so the cancel lands while the prompt still prefills. The default rule takes the prompt in one
//! tick.
//!
//! Refusals. Two tests read a `turn_too_long` refusal, which names the turn size and the positions
//! left. One runs on a fresh session, and one runs after a turn fills the window.
//!
//! Kept pairs. A conversation runs live on one session, then its turns go into a fresh session as
//! kept pairs. Each `appended` position must equal the live position, and the next turn must stream
//! the same ids on both sessions. The server runs the real chat template with thinking off, so the
//! kept pair renders the wraps of the live turn. A turn that stops at its budget leaves its last
//! token pending. The pair holds that token, so its position is one past the live one.
//!
//! Skips. Every test returns `error.SkipZigTest` when no fixture exists.

const std = @import("std");
const zigferencer = @import("zigferencer");
const protocol = zigferencer.serve_protocol;
const Client = zigferencer.serve_client.Client;
const event = zigferencer.serve_event;
const serve_engine = zigferencer.serve_engine;
const harness = @import("serve_harness.zig");
const Harness = harness.Harness;
const Stream = harness.Stream;
const max_seq = harness.max_seq;
const turn_tokens_max = harness.turn_tokens_max;
const prompts = harness.prompts;
const oneTurn = harness.oneTurn;
const expectSameStream = harness.expectSameStream;

// -- fixtures ---------------------------------------------------------------

/// A token budget that keeps a turn decoding while the next request lands.
/// The prompt plus the budget still fits in `max_seq`.
const long_turn_tokens: u32 = 40;

/// Four clients of two sessions each fill both the pool and the batch.
const n_clients = 4;
const clients_opts: harness.Options = .{
    .max_sessions = 8,
    .max_batch = 8,
    .max_connections = n_clients,
};

const Done = @FieldType(protocol.ServerMsg, "done");

const TurnEnd = struct {
    tokens: u32,
    done: Done,
    request_id: u32,
};

fn openGreedy(client: *Client) !u32 {
    return client.openSession("", protocol.Sampling.greedy);
}

/// A prompt alone on a fresh session, closed after. Every batched or
/// interleaved run must reproduce this stream.
fn soloReference(client: *Client, prompt: []const u8) !Stream {
    const sid = try openGreedy(client);
    const s = try oneTurn(client, sid, prompt);
    try std.testing.expect(s.n > 0);
    try client.closeSession(sid);
    return s;
}

/// Read one turn through `next` up to its `done`, so the test sees the
/// fields that `collectTurn` drops. Each token takes one KV position, so at
/// most `max_seq` tokens precede the `done`.
fn readTurn(client: *Client) !TurnEnd {
    var tokens: u32 = 0;
    while (tokens <= max_seq) {
        const ev = try client.next();
        switch (ev.msg) {
            .token => tokens += 1,
            .done => |d| return .{
                .tokens = tokens,
                .done = d,
                .request_id = ev.header.request_id,
            },
            else => return error.UnexpectedReply,
        }
    }

    return error.TurnNeverEnded;
}

/// Read the turns of `sids`, in flight together on one connection, up to one
/// `done` each. The frames interleave, so the reader sorts them by session
/// id. Each turn sends at most `turn_tokens_max` tokens and one `done`.
fn collectInterleaved(client: *Client, sids: [prompts.len]u32) ![prompts.len]Stream {
    var got: [prompts.len]Stream = @splat(.{});
    var done_seen: [prompts.len]bool = @splat(false);
    var dones: usize = 0;
    var frames: usize = 0;
    while (dones < prompts.len) : (frames += 1) {
        try std.testing.expect(frames < prompts.len * (turn_tokens_max + 1));
        const ev = try client.next();
        const slot: usize = for (sids, 0..) |sid, i| {
            if (ev.header.session_id == sid) break i;
        } else return error.UnknownSession;

        try std.testing.expect(!done_seen[slot]);
        switch (ev.msg) {
            .token => |t| {
                try std.testing.expect(got[slot].n < got[slot].ids.len);
                got[slot].ids[got[slot].n] = t.token_id;
                got[slot].n += 1;
            },
            .done => |d| {
                got[slot].reason = d.finish_reason;
                done_seen[slot] = true;
                dones += 1;
            },
            else => return error.UnexpectedReply,
        }
    }

    return got;
}

fn expectTokens(client: *Client, n: usize) !void {
    for (0..n) |_| {
        const ev = try client.next();
        try std.testing.expect(ev.msg == .token);
    }
}

/// The token count the server's own codec gives a turn, so a test can state
/// the numbers a refusal names.
fn turnTokenCount(h: *const Harness, turn: event.TurnKind, text: []const u8) !usize {
    const ids = try h.codec.turnTokens(std.testing.allocator, turn, text, false);
    defer std.testing.allocator.free(ids);

    return ids.len;
}

/// Read the refusal of the request `request_id`: a `turn_too_long` that names
/// the turn size and the positions left.
fn expectTurnTooLong(client: *Client, request_id: u32, tokens: usize, room: u32) !void {
    const ev = try client.next();
    try std.testing.expect(ev.msg == .err);
    try std.testing.expectEqual(request_id, ev.header.request_id);
    try std.testing.expectEqual(protocol.ErrorCode.turn_too_long, ev.msg.err.code);
    var want_buf: [Client.LastError.detail_cap]u8 = undefined;
    const want = try std.fmt.bufPrint(&want_buf, protocol.turn_too_long_fmt, .{ tokens, room });
    try std.testing.expectEqualStrings(want, ev.msg.err.detail);
}

/// An `Io` whose `accept` plays a list of failures, and whose wait returns at once, so a test
/// proves the retry without a clock. The list lives in file state, because the vtable functions
/// receive only the userdata of the real `Io`.
const AcceptScript = struct {
    const AcceptError = std.Io.net.Server.AcceptError;

    var failures: []const AcceptError = &.{};
    var accepts = std.atomic.Value(u32).init(0);
    var waits = std.atomic.Value(u32).init(0);
    var wait_result: std.Io.Cancelable!void = {};
    var vtable: std.Io.VTable = undefined;

    fn start(script: []const AcceptError) std.Io {
        // A back-off warns by design. The runner resets this level for each test.
        std.testing.log_level = .err;
        failures = script;
        accepts.store(0, .release);
        waits.store(0, .release);
        wait_result = {};
        vtable = std.testing.io.vtable.*;
        vtable.netAccept = netAccept;
        vtable.sleep = skipWait;
        return .{ .userdata = std.testing.io.userdata, .vtable = &vtable };
    }

    fn netAccept(
        userdata: ?*anyopaque,
        server: std.Io.net.Socket.Handle,
        options: std.Io.net.Server.AcceptOptions,
    ) AcceptError!std.Io.net.Socket {
        const at = accepts.fetchAdd(1, .acq_rel);
        if (at < failures.len) return failures[at];

        return std.testing.io.vtable.netAccept(userdata, server, options);
    }

    fn skipWait(userdata: ?*anyopaque, timeout: std.Io.Timeout) std.Io.Cancelable!void {
        _ = userdata;
        _ = timeout;
        _ = waits.fetchAdd(1, .acq_rel);
        return wait_result;
    }
};

/// A listener for `acceptNext` when the script ends in a failure, so no real socket is read.
fn unusedListener() std.Io.net.Server {
    var server: std.Io.net.Server = undefined;
    server.options = {};
    return server;
}

// -- tests ------------------------------------------------------------------

test "acceptAction: every error name has its action" {
    const Row = struct { err: AcceptScript.AcceptError, want: serve_engine.AcceptAction };
    const rows = [_]Row{
        .{ .err = error.Canceled, .want = .stop },
        .{ .err = error.SocketNotListening, .want = .stop },
        .{ .err = error.WouldBlock, .want = .stop },
        .{ .err = error.ConnectionAborted, .want = .retry_now },
        .{ .err = error.ProtocolFailure, .want = .retry_now },
        .{ .err = error.BlockedByFirewall, .want = .retry_now },
        .{ .err = error.ProcessFdQuotaExceeded, .want = .back_off },
        .{ .err = error.SystemFdQuotaExceeded, .want = .back_off },
        .{ .err = error.SystemResources, .want = .back_off },
        .{ .err = error.NetworkDown, .want = .back_off },
        .{ .err = error.Unexpected, .want = .back_off },
    };
    // A new error name in std fails this count, so the table stays complete.
    const names = @typeInfo(AcceptScript.AcceptError).error_set.?;
    try std.testing.expectEqual(names.len, rows.len);
    for (rows) |row| {
        try std.testing.expectEqual(row.want, serve_engine.acceptAction(row.err));
    }
}

test "acceptNext outlives each failure and waits only on a host failure" {
    const io = AcceptScript.start(&.{
        error.ConnectionAborted,
        error.ProcessFdQuotaExceeded,
        error.ProtocolFailure,
        error.SystemResources,
        error.Canceled,
    });
    var listener = unusedListener();
    try std.testing.expectEqual(null, serve_engine.acceptNext(&listener, io));
    try std.testing.expectEqual(5, AcceptScript.accepts.load(.acquire));
    try std.testing.expectEqual(2, AcceptScript.waits.load(.acquire));
}

test "acceptNext ends on a cancel of the wait" {
    const io = AcceptScript.start(&.{ error.SystemResources, error.ConnectionAborted });
    AcceptScript.wait_result = error.Canceled;
    var listener = unusedListener();
    try std.testing.expectEqual(null, serve_engine.acceptNext(&listener, io));
    try std.testing.expectEqual(1, AcceptScript.accepts.load(.acquire));
}

test "a client connects after accept failures, and a turn runs" {
    const gpa = std.testing.allocator;
    const io = AcceptScript.start(&.{ error.ConnectionAborted, error.ProcessFdQuotaExceeded });
    var server = try Harness.boot(gpa, io, .tcp, .{});
    defer server.deinit();

    var client = try server.connect();
    defer client.deinit();

    const stream = try oneTurn(&client, try openGreedy(&client), prompts[0]);
    try std.testing.expect(stream.n > 0);
    server.stop(&client);
    try std.testing.expectEqual(1, AcceptScript.waits.load(.acquire));
}

test "interleaved sessions match their solo reference streams over the wire" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    _ = try client.hello();

    // -- reference: each prompt alone, fresh session, closed after ----------
    var reference: [prompts.len]Stream = undefined;
    for (prompts, 0..) |prompt, i| {
        const sid = try client.openSession("", protocol.Sampling.greedy);
        reference[i] = try oneTurn(&client, sid, prompt);
        try std.testing.expect(reference[i].n > 0);
        _ = try client.send(.session_close, sid);
        const ev = try client.next();
        try std.testing.expect(ev.msg == .session_closed);
    }

    // -- interleaved: both turns in flight, events multiplexed by id --------
    var sids: [prompts.len]u32 = undefined;
    for (&sids) |*sid| {
        sid.* = try client.openSession("", protocol.Sampling.greedy);
    }

    for (prompts, 0..) |prompt, i| {
        _ = try client.generate(sids[i], prompt, turn_tokens_max, null);
    }

    const got = try collectInterleaved(&client, sids);
    for (reference, got) |ref, g| {
        try expectSameStream(ref, g);
    }

    // -- context_full is a done, and the session reports honestly -----------
    {
        const sid = try client.openSession("", protocol.Sampling.greedy);
        _ = try client.generate(sid, "Count:", 1000, null);
        var ids: [max_seq]u32 = undefined;
        var text: [4096]u8 = undefined;
        const t = try client.collectTurn(&ids, &text);
        try std.testing.expectEqual(protocol.FinishReason.context_full, t.finish_reason);
        try std.testing.expectEqual(max_seq, t.kv_pos);
        _ = try client.send(.session_close, sid);
        const ev = try client.next();
        try std.testing.expect(ev.msg == .session_closed);
    }

    // -- close mid-generate: no done, a session_closed, slot reusable -------
    {
        const sid = try client.openSession("", protocol.Sampling.greedy);
        _ = try client.generate(sid, "Once upon a time", 40, null);
        var seen_tokens: usize = 0;
        while (seen_tokens < 2) {
            const ev = try client.next();
            if (ev.msg == .token) {
                seen_tokens += 1;
            } else {
                return error.UnexpectedReply;
            }
        }

        _ = try client.send(.session_close, sid);
        while (true) {
            const ev = try client.next();
            if (ev.msg == .session_closed) break;
            // In-flight tokens can still drain, and the turn can even finish
            // (a done) before the close lands. Anything else is a failure.
            try std.testing.expect(ev.msg == .token or ev.msg == .done);
        }

        const sid2 = try client.openSession("", protocol.Sampling.greedy);
        const again = try oneTurn(&client, sid2, prompts[0]);
        try expectSameStream(reference[0], again);
    }
}

test "turn_too_long names its numbers, and the same session then answers a fitting turn" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    const reference = try soloReference(&client, prompts[0]);
    // A fresh session holds the whole window, and this prompt needs more.
    const long_prompt = "The dragon flew over the quiet hills at dawn. " ** 8;
    const n = try turnTokenCount(&h, .open_first, long_prompt);
    try std.testing.expect(n > max_seq);
    const sid = try openGreedy(&client);
    const request_id = try client.generate(sid, long_prompt, turn_tokens_max, null);
    try expectTurnTooLong(&client, request_id, n, max_seq);

    // The refused turn never entered the window, so the session answers as a
    // fresh one does.
    try expectSameStream(reference, try oneTurn(&client, sid, prompts[0]));
}

test "a turn capped at the window ends context_full, and the next turn gets the numbers" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    // A cap of the whole window can never stop a turn first. The prompt takes
    // at least one position, so the window fills before the cap. The greedy
    // stream of this prompt reaches the window with no end-of-sequence.
    const info = try client.hello();
    try std.testing.expectEqual(max_seq, info.max_seq);
    const sid = try openGreedy(&client);
    _ = try client.generate(sid, "Count:", info.max_seq, null);
    var ids: [max_seq]u32 = undefined;
    var text: [4096]u8 = undefined;
    const t = try client.collectTurn(&ids, &text);
    try std.testing.expectEqual(protocol.FinishReason.context_full, t.finish_reason);
    try std.testing.expectEqual(info.max_seq, t.kv_pos);

    // A full window carries no token, so the next turn is its own text alone,
    // and no position is left for it.
    const n = try turnTokenCount(&h, .continuation, prompts[1]);
    const request_id = try client.generate(sid, prompts[1], turn_tokens_max, null);
    try expectTurnTooLong(&client, request_id, n, 0);
    try client.ping();
}

test "ping answers pong on the server scope" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    const request_id = try client.send(.ping, 0);
    const ev = try client.next();
    try std.testing.expect(ev.msg == .pong);
    try std.testing.expectEqual(@as(u32, 0), ev.header.session_id);
    try std.testing.expectEqual(request_id, ev.header.request_id);

    const sid = try openGreedy(&client);
    const s = try oneTurn(&client, sid, prompts[0]);
    try std.testing.expect(s.n > 0);
    try client.ping();
}

test "stats counts open sessions and generated tokens" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    // Every sampled token, the EOS token included, sends one token frame and
    // adds one to the turn's `gen_tokens` and one to the server total. The
    // total is therefore the plain sum of the turns' `gen_tokens`.
    var gen_sum: u64 = 0;
    for (prompts) |prompt| {
        const sid = try openGreedy(&client);
        _ = try client.generate(sid, prompt, turn_tokens_max, null);
        const end = try readTurn(&client);
        try std.testing.expectEqual(end.tokens, end.done.gen_tokens);
        gen_sum += end.done.gen_tokens;
    }

    try std.testing.expect(gen_sum > 0);

    const first = try client.stats();
    try std.testing.expectEqual(@as(u16, 2), first.sessions_open);
    try std.testing.expectEqual(@as(u16, 0), first.sessions_decoding);
    try std.testing.expectEqual(gen_sum, first.tokens_generated_total);
    const second = try client.stats();
    try std.testing.expect(second.uptime_ms >= first.uptime_ms);
}

test "close on an idle session frees its id" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    const sid = try openGreedy(&client);
    try client.closeSession(sid);
    const request_id = try client.generate(sid, prompts[0], turn_tokens_max, null);
    const ev = try client.next();
    try std.testing.expect(ev.msg == .err);
    try std.testing.expectEqual(protocol.ErrorCode.unknown_session, ev.msg.err.code);
    try std.testing.expectEqual(request_id, ev.header.request_id);
}

test "close during a turn drains the stream before session_closed" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    const reference = try soloReference(&client, prompts[1]);
    // The greedy stream of `prompts[0]` runs the whole budget without an EOS,
    // so the close lands while the turn still decodes.
    const sid = try openGreedy(&client);
    _ = try client.generate(sid, prompts[0], long_turn_tokens, null);
    try expectTokens(&client, 2);
    try client.closeSession(sid);

    const fresh = try openGreedy(&client);
    try expectSameStream(reference, try oneTurn(&client, fresh, prompts[1]));
}

test "clone continues exactly like its source" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    const a = try openGreedy(&client);
    const opening = try oneTurn(&client, a, prompts[0]);
    try std.testing.expect(opening.n > 0);
    const b = try client.cloneSession(a, protocol.Sampling.greedy);
    try std.testing.expect(b != a);

    const from_a = try oneTurn(&client, a, prompts[1]);
    const from_b = try oneTurn(&client, b, prompts[1]);
    try std.testing.expect(from_a.n > 0);
    try expectSameStream(from_a, from_b);
    try std.testing.expectEqual(@as(u16, 2), (try client.stats()).sessions_open);
}

test "cancel during prefill finishes the turn with no tokens" {
    var h = try Harness.boot(
        std.testing.allocator,
        std.testing.io,
        .tcp,
        .{ .prefill = .spare_rows },
    );
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    // The spare-rows rule makes the prompt take 14 ticks, so the cancel lands during prefill.
    const long_prompt = "The dragon flew over the quiet hills at dawn. " ** 4;
    const sid = try openGreedy(&client);
    const gen_id = try client.generate(sid, long_prompt, turn_tokens_max, null);
    _ = try client.cancel(sid);

    const end = try readTurn(&client);
    try std.testing.expectEqual(protocol.FinishReason.cancelled, end.done.finish_reason);
    try std.testing.expectEqual(@as(u32, 0), end.done.gen_tokens);
    try std.testing.expectEqual(@as(u32, 0), end.tokens);
    try std.testing.expectEqual(gen_id, end.request_id);
    try std.testing.expect(end.done.turn_prompt_tokens >= 40);

    const after = try oneTurn(&client, sid, prompts[0]);
    try std.testing.expect(after.n > 0);
}

test "cancel during decode finishes the turn as cancelled" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    // The greedy stream of `prompts[0]` runs the whole budget without an EOS,
    // so only the cancel can end the turn early.
    const sid = try openGreedy(&client);
    const gen_id = try client.generate(sid, prompts[0], long_turn_tokens, null);
    try expectTokens(&client, 2);
    _ = try client.cancel(sid);

    const end = try readTurn(&client);
    try std.testing.expectEqual(protocol.FinishReason.cancelled, end.done.finish_reason);
    try std.testing.expect(end.done.gen_tokens >= 2);
    try std.testing.expectEqual(end.tokens + 2, end.done.gen_tokens);
    try std.testing.expectEqual(gen_id, end.request_id);

    const after = try oneTurn(&client, sid, prompts[1]);
    try std.testing.expect(after.n > 0);
    try std.testing.expectEqual(@as(u16, 0), (try client.stats()).sessions_decoding);
}

test "cancel on an idle session sends nothing" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    // The cancel comes after the turn's done, the late cancel a client sends
    // when the turn ends first.
    const sid = try openGreedy(&client);
    const s = try oneTurn(&client, sid, prompts[0]);
    try std.testing.expect(s.n > 0);
    _ = try client.cancel(sid);
    // A reply to the cancel arrives before the pong on this connection, and
    // `ping` fails on any frame but a pong.
    try client.ping();
}

test "cancel on an unknown session fails with unknown_session" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    const cancel_id = try client.cancel(999);
    const ev = try client.next();
    try std.testing.expect(ev.msg == .err);
    try std.testing.expectEqual(protocol.ErrorCode.unknown_session, ev.msg.err.code);
    try std.testing.expectEqual(cancel_id, ev.header.request_id);
}

test "the unix transport carries the same stream as tcp" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const over_tcp = blk: {
        var h = try Harness.boot(gpa, io, .tcp, .{});
        defer h.deinit();

        var client = try h.connect();
        defer client.deinit();
        defer h.stop(&client);

        break :blk try soloReference(&client, prompts[0]);
    };

    var h = try Harness.boot(gpa, io, .unix, .{});
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    const info = try client.hello();
    try std.testing.expectEqual(max_seq, info.max_seq);
    try expectSameStream(over_tcp, try soloReference(&client, prompts[0]));
}

test "four connections interleave sessions and each stream equals its solo reference" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, clients_opts);
    defer h.deinit();

    var clients: [n_clients]Client = undefined;
    var n_live: usize = 0;
    defer for (clients[0..n_live]) |*c| c.deinit();
    defer if (n_live > 0) h.stop(&clients[0]);

    while (n_live < n_clients) : (n_live += 1) {
        clients[n_live] = try h.connect();
    }

    var reference: [prompts.len]Stream = undefined;
    for (&reference, prompts) |*ref, prompt| {
        ref.* = try soloReference(&clients[0], prompt);
    }

    // The test sends every turn before it reads any frame, so the eight
    // sessions share the batch.
    var sids: [n_clients][prompts.len]u32 = undefined;
    for (&clients, &sids) |*c, *row| {
        for (row) |*sid| {
            sid.* = try openGreedy(c);
        }
    }

    for (&clients, sids) |*c, row| {
        for (row, prompts) |sid, prompt| {
            _ = try c.generate(sid, prompt, turn_tokens_max, null);
        }
    }

    for (&clients, sids) |*c, row| {
        const got = try collectInterleaved(c, row);
        for (reference, got) |ref, g| {
            try expectSameStream(ref, g);
        }

        // A third done or a stray token arrives before the pong, and `ping`
        // fails on it. So this client saw exactly its two dones.
        try c.ping();
    }
}

test "the connection past the cap is refused after the handshake" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, clients_opts);
    defer h.deinit();

    var clients: [n_clients]Client = undefined;
    var n_live: usize = 0;
    defer for (clients[0..n_live]) |*c| c.deinit();
    defer if (n_live > 0) h.stop(&clients[0]);

    while (n_live < n_clients) : (n_live += 1) {
        clients[n_live] = try h.connect();
    }

    // A pong proves that the acceptor gave the connection a slot, so all four
    // slots are taken before the fifth connection arrives.
    for (&clients) |*c| {
        try c.ping();
    }

    var fifth = try h.connect();
    defer fifth.deinit();

    const ev = try fifth.next();
    try std.testing.expect(ev.msg == .err);
    try std.testing.expectEqual(protocol.ErrorCode.connections_exhausted, ev.msg.err.code);
    try std.testing.expectError(error.EndOfStream, fifth.next());

    for (&clients) |*c| {
        try c.ping();
    }
}

test "a hang-up mid-turn sweeps only its own sessions" {
    var h = try Harness.boot(
        std.testing.allocator,
        std.testing.io,
        .tcp,
        .{ .max_sessions = 2, .max_batch = 2, .max_connections = 4 },
    );
    defer h.deinit();

    var b = try h.connect();
    defer b.deinit();
    defer h.stop(&b);

    const reference = try soloReference(&b, prompts[1]);
    const sid_b = try openGreedy(&b);
    {
        // A decodes `prompts[0]`, whose greedy stream runs the whole budget
        // without an EOS, so the hang-up lands mid-turn. B decodes the other
        // prompt, so a stream that crosses sessions fails the comparison.
        var a = try h.connect();
        defer a.deinit();

        const sid_a = try openGreedy(&a);
        _ = try a.generate(sid_a, prompts[0], long_turn_tokens, null);
        try expectTokens(&a, 2);
        // The pool of two is full while A holds its session.
        try std.testing.expectError(error.ServerError, openGreedy(&b));
        try std.testing.expectEqual(protocol.ErrorCode.sessions_exhausted, b.last_error.?.code);
        a.hangUp();
    }

    try expectSameStream(reference, try oneTurn(&b, sid_b, prompts[1]));

    // No order joins the sweep of A's session and this open. The loop waits on the session table,
    // and the bound turns a sweep that never lands into a failure.
    const open_attempts_max = 64;
    var sid_new: ?u32 = null;
    for (0..open_attempts_max) |_| {
        sid_new = openGreedy(&b) catch |err| {
            if (err != error.ServerError) return err;
            try std.testing.expectEqual(protocol.ErrorCode.sessions_exhausted, b.last_error.?.code);
            continue;
        };
        break;
    }

    if (sid_new == null) {
        std.debug.print(
            "the hang-up sweep freed no session within {d} open attempts\n",
            .{open_attempts_max},
        );
        return error.TestUnexpectedResult;
    }

    try std.testing.expectEqual(@as(u16, 2), (try b.stats()).sessions_open);
}

test "a connection past its session share gets sessions_exhausted" {
    var h = try Harness.boot(
        std.testing.allocator,
        std.testing.io,
        .tcp,
        .{ .max_sessions = 4, .sessions_per_connection = 2 },
    );
    defer h.deinit();

    var a = try h.connect();
    defer a.deinit();
    defer h.stop(&a);

    var b = try h.connect();
    defer b.deinit();

    const a1 = try openGreedy(&a);
    const a2 = try openGreedy(&a);
    try std.testing.expect(a1 != a2);
    try std.testing.expectError(error.ServerError, openGreedy(&a));
    try std.testing.expectEqual(protocol.ErrorCode.sessions_exhausted, a.last_error.?.code);
    // The pool of four still has room, so the refusal above is the share.
    _ = try openGreedy(&b);
    try std.testing.expectEqual(@as(u16, 3), (try b.stats()).sessions_open);
}

test "shutdown from the wire is refused unless allowed" {
    var h = try Harness.boot(
        std.testing.allocator,
        std.testing.io,
        .tcp,
        .{ .allow_shutdown = false },
    );
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();

    try std.testing.expectError(error.ServerError, client.shutdown());
    try std.testing.expectEqual(protocol.ErrorCode.forbidden, client.last_error.?.code);
    try client.ping();
    h.stopFromOwner();
}

// -- kept pairs -------------------------------------------------------------

const kept_system = "You are a helpful assistant.";
const kept_prompts = [_][]const u8{ "Name a color.", "Name an animal.", "Name a city." };
const kept_window: u32 = 256;
const kept_turn_tokens: u32 = 16;

const LiveTurn = struct {
    reply: [512]u8 = undefined,
    len: usize = 0,
    kv_pos: u32 = 0,
    carried: bool = false,
};

fn keptPairsMatchLive(comptime H: type, kind: harness.ChatKind) !void {
    var h = try H.boot(
        std.testing.allocator,
        std.testing.io,
        .tcp,
        .{ .window = kept_window, .kind = kind },
    );
    defer h.deinit();

    var client = try h.connect();
    defer client.deinit();
    defer h.stop(&client);

    _ = try client.hello();

    const live = try client.openSession(kept_system, protocol.Sampling.greedy);
    var turns: [2]LiveTurn = .{ .{}, .{} };
    var ids: [kept_window]u32 = undefined;
    for (&turns, kept_prompts[0..2]) |*t, prompt| {
        _ = try client.generate(live, prompt, kept_turn_tokens, null);
        const done = try client.collectTurn(&ids, &t.reply);
        try std.testing.expectEqual(done.text_len, done.text_total);
        t.len = done.text_len;
        t.kv_pos = done.kv_pos;
        t.carried = done.finish_reason == .max_tokens;
    }

    const kept = try client.openSession(kept_system, protocol.Sampling.greedy);
    const pairs = [_]protocol.Pair{
        .{ .user = kept_prompts[0], .assistant = turns[0].reply[0..turns[0].len] },
        .{ .user = kept_prompts[1], .assistant = turns[1].reply[0..turns[1].len] },
    };
    const appended = try client.append(kept, &pairs);
    // A carried token sits at the start of the next live turn, and inside its own pair reply.
    // So each position differs from the live one by the carry of its own turn.
    try std.testing.expectEqual(
        turns[0].kv_pos + @intFromBool(turns[0].carried),
        appended.positions()[0],
    );
    try std.testing.expectEqual(
        turns[1].kv_pos + @intFromBool(turns[1].carried),
        appended.positions()[1],
    );

    var live_ids: [kept_turn_tokens]u32 = undefined;
    var kept_ids: [kept_turn_tokens]u32 = undefined;
    var text: [512]u8 = undefined;
    _ = try client.generate(live, kept_prompts[2], kept_turn_tokens, null);
    const a = try client.collectTurn(&live_ids, &text);
    _ = try client.generate(kept, kept_prompts[2], kept_turn_tokens, null);
    const b = try client.collectTurn(&kept_ids, &text);
    try std.testing.expectEqual(a.n_tokens, b.n_tokens);
    try std.testing.expectEqualSlices(u32, live_ids[0..a.n_tokens], kept_ids[0..b.n_tokens]);
    try std.testing.expectEqual(a.kv_pos, b.kv_pos);
}

test "kept pairs give the ids of the live turns on SmolLM2" {
    try keptPairsMatchLive(Harness, .chatml_smollm2);
}

test "kept pairs give the ids of the live turns on the qwen CPU runner, thinking off" {
    try keptPairsMatchLive(harness.QwenHarness, .{ .chatml_qwen = .off });
}
