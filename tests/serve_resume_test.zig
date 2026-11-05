//! The session resumption gate over a real model and real sockets. Each test boots its own
//! in-process server through `serve_harness.zig` and drives it through the reference client.
//!
//! Twin. The load-bearing assertion is exact. A session that a client keeps, drops, and resumes on
//! a new connection continues with the ids and the position of an undropped twin that ran the same
//! turns. This holds for greedy and for seeded sampling on SmolLM2, and for greedy on the Qwen CPU
//! runner.
//!
//! Drop and lease. A drop in the middle of a turn cuts that turn as a cancel does, and the session
//! keeps the carry of the cut turn. An expired lease answers `unknown_session`. A live open on a
//! full table takes the detached slot with the earliest deadline.
//!
//! Rebuild. A session rebuilt from the client transcript with kept pairs continues as the twin
//! does.
//!
//! Resumption off. A keep is forbidden, and a drop frees the session.
//!
//! Skips. Every test returns `error.SkipZigTest` when no fixture exists.

const std = @import("std");
const zigferencer = @import("zigferencer");
const protocol = zigferencer.serve_protocol;
const Client = zigferencer.serve_client.Client;
const harness = @import("serve_harness.zig");
const Harness = harness.Harness;
const oneTurn = harness.oneTurn;
const expectSameStream = harness.expectSameStream;

// -- fixtures ---------------------------------------------------------------

const system = "You are a helpful assistant.";
const turn_prompts = [_][]const u8{
    "Name a color.",
    "Name an animal.",
    "Name a city.",
    "Name a fruit.",
};
const window: u32 = 256;
const turn_tokens: u32 = 16;
/// Short turns keep the Qwen case inside its time budget.
const qwen_turn_tokens: u32 = 6;
/// Thinking turns stay short for the same reason.
const qwen_think_tokens: u32 = 12;
const ttl_max_ms: u32 = 60_000;
const ttl_ms: u32 = 30_000;

/// The cut turn runs long enough that the drop always lands while it
/// decodes. The window holds two turns and the whole budget.
const cut_window: u32 = 512;
const cut_tokens: u32 = 200;
const cut_prompt = "Count from one to one hundred in words, one number per line.";

/// A drop and an open travel on two connections, and no order joins them.
/// Each attempt is one full round trip that reads the session table, so a
/// loop waits on an observable state and not on a clock.
const open_attempts_max = 64;

const seeded: protocol.Sampling = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = 1234 };
const reseeded: protocol.Sampling = .{ .temperature = 0.9, .top_k = 20, .top_p = 0.9, .seed = 99 };

const Turn = struct {
    ids: [turn_tokens]u32 = undefined,
    n: usize = 0,
    reply: [512]u8 = undefined,
    len: usize = 0,
    kv_pos: u32 = 0,
    finish: protocol.FinishReason = .eos,
    gen_tokens: u32 = 0,
    reasoning_tokens: u32 = 0,
    request_id: u32 = 0,
};

fn opts(cfg: harness.Options) harness.Options {
    var o = cfg;
    o.kind = .chatml_smollm2;
    return o;
}

/// Collect the turn of `request_id` in full. The ids and the reply text
/// must fit, so a comparison never reads a cut copy.
fn collect(client: *Client, request_id: u32) !Turn {
    var t: Turn = .{ .request_id = request_id };
    const done = try client.collectTurn(&t.ids, &t.reply);
    try std.testing.expect(done.n_tokens <= t.ids.len);
    try std.testing.expectEqual(done.text_len, done.text_total);
    t.n = done.n_tokens;
    t.len = done.text_len;
    t.kv_pos = done.kv_pos;
    t.finish = done.finish_reason;
    t.gen_tokens = done.gen_tokens;
    t.reasoning_tokens = done.reasoning_tokens;
    return t;
}

/// Send one turn on each of two sessions before either reads, so the
/// two share the batch, then collect both.
fn turnPair(
    a: *Client,
    a_sid: u32,
    b: *Client,
    b_sid: u32,
    prompt: []const u8,
    turn: Client.TurnOptions,
) ![2]Turn {
    const a_id = try a.generateWith(a_sid, prompt, turn);
    const b_id = try b.generateWith(b_sid, prompt, turn);
    return .{ try collect(a, a_id), try collect(b, b_id) };
}

fn expectSameTurn(want: Turn, got: Turn) !void {
    try std.testing.expectEqual(want.finish, got.finish);
    try std.testing.expectEqualSlices(u32, want.ids[0..want.n], got.ids[0..got.n]);
    try std.testing.expectEqual(want.kv_pos, got.kv_pos);
}

fn expectRefused(client: *Client, code: protocol.ErrorCode, detail: []const u8) !void {
    const le = client.last_error.?;
    try std.testing.expectEqual(code, le.code);
    try std.testing.expectEqualStrings(detail, le.detail());
}

/// Open on `client` until the session table has room. A refusal before
/// that is `sessions_exhausted`.
fn openWhenFree(client: *Client) !u32 {
    for (0..open_attempts_max) |_| {
        return client.openSession(system, protocol.Sampling.greedy) catch |err| {
            if (err != error.ServerError) return err;
            try expectRefused(client, .sessions_exhausted, "no free session");
            continue;
        };
    }

    std.debug.print("the drop freed no slot within {d} open attempts\n", .{open_attempts_max});
    return error.TestUnexpectedResult;
}

/// A point on the pinned scheduler clock.
fn atMs(ms: i64) std.Io.Clock.Timestamp {
    return .{ .raw = .fromNanoseconds(@as(i96, ms) * std.time.ns_per_ms), .clock = .awake };
}

const Counters = struct { expires: u64 = 0, yields: u64 = 0, lost: u64 = 0, cut: u64 = 0 };

/// The lease counters of the scheduler. The engine thread owns them, so
/// the read must come after the server stops.
fn counters(h: anytype) Counters {
    std.debug.assert(!h.running);

    const s = &h.server.sched;
    return .{
        .expires = s.lease_expires,
        .yields = s.lease_yields,
        .lost = s.lease_yields_lost,
        .cut = s.lease_prefills_cut,
    };
}

fn keep(client: *Client, sid: u32, ttl: u32) !protocol.Token {
    const kept = try client.keepSession(sid, ttl);
    try std.testing.expectEqual(@min(ttl, ttl_max_ms), kept.granted_ms);
    try std.testing.expect(!std.mem.allEqual(u8, &kept.token, 0));
    return kept.token;
}

/// Four turns on a twin and on a session that a client keeps, drops after
/// turn 3, and resumes on a new connection. Turn 4 must give the ids and
/// the position of the twin. With `sampling` set, both sides reseed turn 4.
fn resumeMatchesTwin(
    comptime H: type,
    kind: harness.ChatKind,
    budget: u32,
    sampling: protocol.Sampling,
    turn4: ?protocol.Sampling,
) !void {
    return resumeMatchesTwinWith(H, kind, sampling, .{ .max_tokens = budget }, .{
        .max_tokens = budget,
        .sampling = turn4,
        .reseed = turn4 != null,
    });
}

/// The twin run with the per-turn fields spelled out: `early` for turns 1 to
/// 3 and `last` for turn 4. The `resumed` reply must also carry the
/// reasoning count of turn 3.
fn resumeMatchesTwinWith(
    comptime H: type,
    kind: harness.ChatKind,
    sampling: protocol.Sampling,
    early: Client.TurnOptions,
    last: Client.TurnOptions,
) !void {
    std.debug.assert(early.max_tokens <= turn_tokens);
    std.debug.assert(last.max_tokens <= turn_tokens);

    var h = try H.boot(std.testing.allocator, std.testing.io, .tcp, .{
        .max_sessions = 2,
        .max_batch = 2,
        .window = window,
        .kind = kind,
        .resume_ttl_max_ms = ttl_max_ms,
    });
    defer h.deinit();

    var twin = try h.connect();
    defer twin.deinit();

    try std.testing.expectEqual(ttl_max_ms, (try twin.hello()).resume_ttl_max_ms);
    const tsid = try twin.openSession(system, sampling);

    var token: protocol.Token = undefined;
    var sid: u32 = undefined;
    var turn3: [2]Turn = undefined;
    {
        var first = try h.connect();
        defer first.deinit();

        _ = try first.hello();
        sid = try first.openSession(system, sampling);
        for (turn_prompts[0..3]) |prompt| {
            turn3 = try turnPair(&twin, tsid, &first, sid, prompt, early);
            try expectSameTurn(turn3[0], turn3[1]);
        }

        token = try keep(&first, sid, ttl_ms);
        first.hangUp();
    }

    var second = try h.connect();
    defer second.deinit();

    _ = try second.hello();
    const r = try second.resumeSession(token);
    try std.testing.expectEqual(sid, r.id);
    try std.testing.expect(!std.mem.eql(u8, &token, &r.state.token));
    // A thinking turn must count reasoning, or the equality below proves nothing.
    if (early.think_open == true) {
        try std.testing.expect(turn3[0].reasoning_tokens > 0);
    }

    try std.testing.expectEqual(turn3[0].kv_pos, r.state.kv_pos);
    try std.testing.expectEqual(@as(u32, 3), r.state.turns_done);
    try std.testing.expectEqual(protocol.LastDone{
        .request_id = turn3[1].request_id,
        .finish = turn3[1].finish,
        .gen_tokens = turn3[1].gen_tokens,
        .reasoning_tokens = turn3[0].reasoning_tokens,
    }, r.state.last.?);

    const t4 = try turnPair(&twin, tsid, &second, r.id, turn_prompts[3], last);
    try std.testing.expect(t4[0].n > 0);
    try expectSameTurn(t4[0], t4[1]);

    h.stop(&twin);
    try std.testing.expectEqual(Counters{}, counters(&h));
}

// -- tests ------------------------------------------------------------------

test "resumption: a dropped and resumed SmolLM2 session gives the turn 4 ids of its twin" {
    try resumeMatchesTwin(Harness, .chatml_smollm2, turn_tokens, protocol.Sampling.greedy, null);
}

test "resumption: a resumed seeded session reseeded at turn 4 gives the ids of its twin" {
    try resumeMatchesTwin(Harness, .chatml_smollm2, turn_tokens, seeded, reseeded);
}

test "resumption: a resumed Qwen session gives the turn 4 ids of its twin, thinking off" {
    try resumeMatchesTwin(
        harness.QwenHarness,
        .{ .chatml_qwen = .off },
        qwen_turn_tokens,
        protocol.Sampling.greedy,
        null,
    );
}

test "resumption: sampling state, presence set, and repetition set carry across a drop" {
    // Turns 2 to 4 send no sampling block and no reseed, so the stream, the
    // presence set, and the repetition context of the session all carry.
    const open: protocol.Sampling = .{
        .temperature = 0.9,
        .top_k = 40,
        .top_p = 0.95,
        .seed = 4321,
        .presence_penalty = 0.6,
        .repetition_penalty = 1.3,
        .min_p = 0.05,
    };
    try resumeMatchesTwinWith(
        Harness,
        .chatml_smollm2,
        open,
        .{ .max_tokens = turn_tokens },
        .{ .max_tokens = turn_tokens },
    );
}

test "resumption: a Qwen session with a think budget matches its twin on ids and reasoning count" {
    const turn: Client.TurnOptions = .{
        .max_tokens = qwen_think_tokens,
        .think_open = true,
        .think_budget = 16,
    };
    try resumeMatchesTwinWith(
        harness.QwenHarness,
        .{ .chatml_qwen = .on },
        protocol.Sampling.greedy,
        turn,
        turn,
    );
}

test "resumption: a drop mid-decode cuts the turn as cancelled, and the next turn uses the carry" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, opts(.{
        .max_sessions = 2,
        .max_batch = 2,
        .window = cut_window,
        .resume_ttl_max_ms = ttl_max_ms,
    }));
    defer h.deinit();

    var twin = try h.connect();
    defer twin.deinit();

    const tsid = try twin.openSession(system, protocol.Sampling.greedy);

    var token: protocol.Token = undefined;
    var sid: u32 = undefined;
    var cut_id: u32 = undefined;
    var cut_head: [2]u32 = undefined;
    var total_before: u64 = undefined;
    {
        var first = try h.connect();
        defer first.deinit();

        sid = try first.openSession(system, protocol.Sampling.greedy);
        for (turn_prompts[0..2]) |prompt| {
            const t = try turnPair(
                &twin,
                tsid,
                &first,
                sid,
                prompt,
                .{ .max_tokens = turn_tokens },
            );
            try expectSameTurn(t[0], t[1]);
        }

        token = try keep(&first, sid, ttl_ms);
        total_before = (try first.stats()).tokens_generated_total;
        cut_id = try first.generate(sid, cut_prompt, cut_tokens, null);
        for (&cut_head) |*id| {
            const ev = try first.next();
            try std.testing.expect(ev.msg == .token);
            id.* = ev.msg.token.token_id;
        }

        first.hangUp();
    }

    // The drop takes the cancel rule, so the summary names the cut turn.
    var second = try h.connect();
    defer second.deinit();

    const r = try second.resumeSession(token);
    try std.testing.expectEqual(sid, r.id);
    try std.testing.expectEqual(@as(u32, 3), r.state.turns_done);
    const last = r.state.last.?;
    try std.testing.expectEqual(cut_id, last.request_id);
    try std.testing.expectEqual(protocol.FinishReason.cancelled, last.finish);
    try std.testing.expect(last.gen_tokens >= cut_head.len);
    try std.testing.expect(last.gen_tokens < cut_tokens);
    // The server total counts every token of the cut turn, sent or not.
    try std.testing.expectEqual(
        total_before + last.gen_tokens,
        (try second.stats()).tokens_generated_total,
    );

    // A turn that stops at its budget keeps the same carry as a cancel at
    // the same count. The twin thus reaches the position of the resumed
    // session, and turn 4 matches.
    _ = try twin.generate(tsid, cut_prompt, last.gen_tokens, null);
    var twin_ids: [cut_tokens]u32 = undefined;
    var text: [4096]u8 = undefined;
    const twin_cut = try twin.collectTurn(&twin_ids, &text);
    try std.testing.expectEqual(protocol.FinishReason.max_tokens, twin_cut.finish_reason);
    try std.testing.expectEqual(@as(usize, last.gen_tokens), twin_cut.n_tokens);
    try std.testing.expectEqualSlices(u32, &cut_head, twin_ids[0..cut_head.len]);
    try std.testing.expectEqual(twin_cut.kv_pos, r.state.kv_pos);

    const t4 = try turnPair(
        &twin,
        tsid,
        &second,
        r.id,
        turn_prompts[3],
        .{ .max_tokens = turn_tokens },
    );
    try std.testing.expect(t4[0].n > 0);
    try expectSameTurn(t4[0], t4[1]);

    // A cut turn is no prefill, and the lease neither expired nor yielded.
    h.stop(&twin);
    try std.testing.expectEqual(Counters{}, counters(&h));
}

test "resumption: an expired lease is unknown, and kept pairs rebuild the session of the twin" {
    var h = try Harness.boot(std.testing.allocator, std.testing.io, .tcp, opts(.{
        .max_sessions = 3,
        .max_batch = 3,
        .window = window,
        .resume_ttl_max_ms = ttl_max_ms,
    }));
    defer h.deinit();

    // The pin starts before the first request, so every lease read of the
    // engine thread sees the pinned clock.
    h.server.sched.pinClock(atMs(0));
    var twin = try h.connect();
    defer twin.deinit();

    const tsid = try twin.openSession(system, protocol.Sampling.greedy);

    var transcript: [3]Turn = undefined;
    var token: protocol.Token = undefined;
    {
        var first = try h.connect();
        defer first.deinit();

        const sid = try first.openSession(system, protocol.Sampling.greedy);
        for (&transcript, turn_prompts[0..3]) |*t, prompt| {
            const pair = try turnPair(
                &twin,
                tsid,
                &first,
                sid,
                prompt,
                .{ .max_tokens = turn_tokens },
            );
            try expectSameTurn(pair[0], pair[1]);
            t.* = pair[1];
        }

        // The unkept session frees at the drop, and its free slot makes the
        // drop observable from another connection.
        _ = try first.openSession(system, protocol.Sampling.greedy);
        token = try keep(&first, sid, 1);
        first.hangUp();
    }

    const probe = try openWhenFree(&twin);
    try twin.closeSession(probe);
    // The deadline is 1 ms after the detach on the pinned clock, so every read after this write
    // is past it. The engine thread reads the pin at each tick, so the pin is atomic.
    h.server.sched.pinClock(atMs(2));

    var second = try h.connect();
    defer second.deinit();

    try std.testing.expectError(error.ServerError, second.resumeSession(token));
    try expectRefused(&second, .unknown_session, "no such lease");

    const rebuilt = try second.openSession(system, protocol.Sampling.greedy);
    var pairs: [transcript.len]protocol.Pair = undefined;
    for (&pairs, &transcript, turn_prompts[0..3]) |*p, *t, prompt| {
        p.* = .{ .user = prompt, .assistant = t.reply[0..t.len] };
    }

    const appended = try second.append(rebuilt, &pairs);
    // A turn that stops at its budget carries its last token into the next
    // turn, and the kept pair holds that token in its reply.
    const carried = transcript[2].finish == .max_tokens;
    try std.testing.expectEqual(
        transcript[2].kv_pos + @intFromBool(carried),
        appended.positions()[2],
    );

    const t4 = try turnPair(
        &twin,
        tsid,
        &second,
        rebuilt,
        turn_prompts[3],
        .{ .max_tokens = turn_tokens },
    );
    try std.testing.expect(t4[0].n > 0);
    try expectSameTurn(t4[0], t4[1]);

    h.stop(&twin);
    try std.testing.expectEqual(Counters{ .expires = 1 }, counters(&h));
}

test "resumption: a live open on a full table takes the detached slot with the earliest deadline" {
    var h = try Harness.boot(
        std.testing.allocator,
        std.testing.io,
        .tcp,
        .{ .resume_ttl_max_ms = ttl_max_ms },
    );
    defer h.deinit();

    var live = try h.connect();
    defer live.deinit();

    const a = try live.openSession(system, protocol.Sampling.greedy);
    const reference = try oneTurn(&live, a, harness.prompts[0]);
    try std.testing.expect(reference.n > 0);

    var late: protocol.Token = undefined;
    var early: protocol.Token = undefined;
    var late_sid: u32 = undefined;
    {
        var first = try h.connect();
        defer first.deinit();

        late_sid = try first.openSession(system, protocol.Sampling.greedy);
        const early_sid = try first.openSession(system, protocol.Sampling.greedy);
        late = try keep(&first, late_sid, ttl_ms);
        early = try keep(&first, early_sid, ttl_ms - 10_000);
        first.hangUp();
    }

    // Every slot is in use. The open succeeds only once a slot detaches,
    // and then it takes the slot with the earliest deadline.
    const b = try openWhenFree(&live);
    try std.testing.expectError(error.ServerError, live.resumeSession(early));
    try expectRefused(&live, .unknown_session, "no such lease");
    const r = try live.resumeSession(late);
    try std.testing.expectEqual(late_sid, r.id);
    // The yielded slot holds nothing of its old session.
    try expectSameStream(reference, try oneTurn(&live, b, harness.prompts[0]));

    h.stop(&live);
    try std.testing.expectEqual(Counters{ .yields = 1 }, counters(&h));
}

test "resumption: with resumption off a keep is forbidden, and a drop frees the session" {
    var h = try Harness.boot(
        std.testing.allocator,
        std.testing.io,
        .tcp,
        .{ .max_sessions = 1, .max_batch = 1 },
    );
    defer h.deinit();

    var other = try h.connect();
    defer other.deinit();

    try std.testing.expectEqual(@as(u32, 0), (try other.hello()).resume_ttl_max_ms);
    {
        var first = try h.connect();
        defer first.deinit();

        const sid = try first.openSession(system, protocol.Sampling.greedy);
        try std.testing.expectError(error.ServerError, first.keepSession(sid, 1_000));
        try expectRefused(&first, .forbidden, "resumption is off");
        // A keep of 0 drops the lease, so it succeeds with resumption off.
        const dropped = try first.keepSession(sid, 0);
        try std.testing.expectEqual(@as(u32, 0), dropped.granted_ms);
        try std.testing.expect(std.mem.allEqual(u8, &dropped.token, 0));
        try std.testing.expectError(error.ServerError, first.resumeSession(@splat(0xA5)));
        try expectRefused(&first, .unknown_session, "no such lease");
        first.hangUp();
    }

    _ = try openWhenFree(&other);
    try std.testing.expectEqual(@as(u16, 1), (try other.stats()).sessions_open);

    h.stop(&other);
    try std.testing.expectEqual(Counters{}, counters(&h));
}
