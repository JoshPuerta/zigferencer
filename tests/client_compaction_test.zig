//! The client compaction over a real model and a real socket. A conversation outgrows a window of
//! 64 positions on SmolLM2. The test drives it with the rules of the client: it records each turn,
//! and it compacts at once after a turn that ends `context_full`. The reserve of each compaction
//! follows the longest turn so far, as in the client. It answers a refused turn with `retryPlan`, a
//! compaction, and one resend.
//!
//! Cycle. Each compaction is one cycle: close the session, open one with the original system
//! prompt, and append the kept pairs. The open must hit the prefix cache, so it costs exactly the
//! system rows. The position after the appends must not pass the estimate of the plan.
//!
//! Refit. A second test plants a turn whose recorded cost is too low, so the append refuses and the
//! cycle refits with the learned cost.
//!
//! Script. Six short turns fill the window while the longest turn stays small, so the adaptive
//! reserve stays near its floor. The server then refuses the poem prompt, and that refusal
//! compaction keeps the newest short turn. The turn about flight ends `context_full`. The village
//! prompt arrives when few positions are left, so the server refuses it too.
//!
//! Running. Set `ZIGFERENCER_TEST_VERBOSE` to print one line for each compaction. The tests return
//! `error.SkipZigTest` when no fixture exists.
//! fixture exists.

const std = @import("std");
const zigferencer = @import("zigferencer");
const ct = zigferencer.client_transcript;
const protocol = zigferencer.serve_protocol;
const limits = zigferencer.limits;
const Client = zigferencer.serve_client.Client;
const harness = @import("serve_harness.zig");
const Harness = harness.Harness;
const max_seq = harness.max_seq;

// -- fixtures ---------------------------------------------------------------

const system_prompt = "You are a helpful assistant.";

/// The prefix cache holds the system prompt of every re-open.
const opts: harness.Options = .{ .prefix_bytes = 64 << 20 };

const Step = struct { prompt: []const u8, max_tokens: u32 };

const castle_prompt = "Describe the castle where the old dragon sleeps, " ++
    "with its towers, gates, walls, and the river around it.";
const village_prompt = "Describe the village below the mountain, " ++
    "with its farms, its market, its mill, and the people who live there.";

/// The script of turns. See "Script" in the header.
const steps = [_]Step{
    .{ .prompt = "Tell me about dragons.", .max_tokens = 5 },
    .{ .prompt = "What do they eat?", .max_tokens = 5 },
    .{ .prompt = "Where do they live?", .max_tokens = 5 },
    .{ .prompt = "Are they big?", .max_tokens = 5 },
    .{ .prompt = "Can they swim?", .max_tokens = 5 },
    .{ .prompt = "Do they sleep?", .max_tokens = 5 },
    .{ .prompt = "Write a short poem about the brave little dragon.", .max_tokens = 12 },
    .{ .prompt = "Tell me a story about the dragon.", .max_tokens = max_seq },
    .{ .prompt = "Where does it fly at night?", .max_tokens = max_seq },
    .{ .prompt = castle_prompt, .max_tokens = 12 },
    .{ .prompt = "Who are its friends?", .max_tokens = 12 },
    .{ .prompt = "How does the story end?", .max_tokens = 12 },
    .{ .prompt = village_prompt, .max_tokens = 12 },
};

const Path = enum { context_full, refusal, planted };

/// The state of one conversation, as the client keeps it.
const Chat = struct {
    client: *Client,
    t: *ct.Transcript,
    session: Client.Opened,
    system_rows: u32,
    verbose: bool,
    counts: std.EnumArray(Path, u32) = .initFill(0),
    /// The turns that the compactions of each path kept, summed.
    kept: std.EnumArray(Path, u32) = .initFill(0),
    /// The resent turns that the server admitted after a refusal compaction
    /// that kept at least one turn.
    resent_after_kept: u32 = 0,
    /// The cycles that restarted after a refused append.
    refits: u32 = 0,

    /// One compaction cycle. Each pass opens one session. A refused frame
    /// teaches its cost and keeps fewer pairs, so the loop runs at most
    /// `count + 1` passes.
    fn compact(c: *Chat, path: Path, r: u32) !ct.Plan {
        var p = ct.plan(c.t, .{ .window = max_seq, .reserve = r, .system_rows = c.system_rows });
        c.t.dropOldest(p.dropped);
        const passes_max = c.t.count() + 1;
        var passes: u32 = 0;
        cycle: while (true) : (passes += 1) {
            try std.testing.expect(passes < passes_max);
            try c.client.closeSession(c.session.id);
            c.session = try c.client.openSessionInfo(system_prompt, protocol.Sampling.greedy);
            // The original system prompt opens each cycle, so the open costs
            // exactly the system rows of the first session.
            try std.testing.expectEqual(c.system_rows, c.session.kv_pos);
            var next: u32 = 0;
            while (next < c.t.count()) {
                const end = ct.frameEnd(c.t, next, c.t.count());
                var pairs: [limits.max_append_pairs]protocol.Pair = undefined;
                for (next..end, 0..) |i, j| {
                    pairs[j] = c.t.pair(@intCast(i));
                }

                const a = c.client.append(c.session.id, pairs[0 .. end - next]) catch |err| {
                    if (err != error.ServerError) return err;

                    const e = c.client.last_error.?;
                    try std.testing.expectEqual(protocol.ErrorCode.turn_too_long, e.code);
                    const tl = ct.parseTooLong(e.detail()).?;
                    ct.learnRefusal(c.t, next, end, tl.tokens);
                    const before = p.kept;
                    p = ct.plan(
                        c.t,
                        .{ .window = max_seq, .reserve = r, .system_rows = c.system_rows },
                    );
                    try std.testing.expect(p.kept < before);
                    c.t.dropOldest(p.dropped);
                    c.refits += 1;
                    continue :cycle;
                };
                c.session.kv_pos = ct.learnAppended(c.t, next, c.session.kv_pos, a.positions());
                next = end;
            }

            break;
        }

        if (c.verbose) {
            std.debug.print(
                "compaction {t}: kept {d}, reserve {d}, estimate {d}, kv_pos {d}/{d}\n",
                .{
                    path, p.kept, r, p.estimate, c.session.kv_pos, max_seq,
                },
            );
        }

        try std.testing.expect(c.session.kv_pos <= p.estimate);
        try std.testing.expect(c.session.kv_pos + r <= max_seq);
        c.counts.getPtr(path).* += 1;
        c.kept.getPtr(path).* += p.kept;
        return p;
    }

    /// One prompt, with at most one resend after a refusal.
    fn turn(c: *Chat, step: Step, ids: []u32, text: []u8) !void {
        var retried = false;
        var refusal_kept: u32 = 0;
        // The loop runs at most two passes: the prompt and one resend.
        while (true) {
            c.t.begin(step.prompt);
            _ = try c.client.generate(c.session.id, step.prompt, step.max_tokens, null);
            const done = c.client.collectTurn(ids, text) catch |err| {
                _ = c.t.abandon();
                if (err != error.ServerError or retried) return err;

                const e = c.client.last_error.?;
                try std.testing.expectEqual(protocol.ErrorCode.turn_too_long, e.code);
                const tl = ct.parseTooLong(e.detail()).?;
                try std.testing.expectEqual(ct.TooLong.Kind.turn, tl.kind);
                switch (ct.retryPlan(
                    c.system_rows,
                    max_seq,
                    ct.reserveFor(max_seq, c.t.longest_cost),
                    tl.tokens,
                )) {
                    .refuse => return error.RetryRefused,
                    .retry => |r| refusal_kept = (try c.compact(.refusal, r)).kept,
                }

                retried = true;
                continue;
            };
            try std.testing.expect(done.text_len == done.text_total);
            if (done.text_len > 0) {
                c.t.appendReply(text[0..done.text_len]);
            }

            const rec_cost = ct.liveCost(
                done.kv_pos - c.session.kv_pos,
                done.finish_reason,
                done.gen_tokens,
            );
            const rec = c.t.finish(rec_cost);
            try std.testing.expect(!rec.barrier);
            c.session.kv_pos = done.kv_pos;
            if (c.verbose) {
                std.debug.print(
                    "turn: cost {d}, kv_pos {d}/{d}, {t}\n",
                    .{ rec_cost, done.kv_pos, max_seq, done.finish_reason },
                );
            }

            if (retried and refusal_kept > 0) {
                c.resent_after_kept += 1;
            }

            if (done.finish_reason == .context_full) {
                _ = try c.compact(.context_full, ct.reserveFor(max_seq, c.t.longest_cost));
            }

            return;
        }
    }
};

const Setup = struct {
    h: Harness,
    client: Client,
    t: *ct.Transcript,
    chat: Chat,

    fn boot(s: *Setup, gpa: std.mem.Allocator) !void {
        s.h = try Harness.boot(gpa, std.testing.io, .tcp, opts);
        errdefer s.h.deinit();

        s.client = try s.h.connect();
        errdefer s.client.deinit();

        const info = try s.client.hello();
        try std.testing.expectEqual(max_seq, info.max_seq);
        const session = try s.client.openSessionInfo(system_prompt, protocol.Sampling.greedy);
        s.t = try gpa.create(ct.Transcript);
        s.t.init();
        s.chat = .{
            .client = &s.client,
            .t = s.t,
            .session = session,
            .system_rows = session.kv_pos,
            .verbose = std.testing.environ.getPosix("ZIGFERENCER_TEST_VERBOSE") != null,
        };
    }

    fn deinit(s: *Setup, gpa: std.mem.Allocator) void {
        s.h.stop(&s.client);
        gpa.destroy(s.t);
        s.client.deinit();
        s.h.deinit();
    }
};

// -- tests ------------------------------------------------------------------

test "a conversation that outgrows its window continues after compaction" {
    const gpa = std.testing.allocator;
    var s: Setup = undefined;
    try s.boot(gpa);
    defer s.deinit(gpa);

    var ids: [max_seq]u32 = undefined;
    var text: [4096]u8 = undefined;

    const hits_before = s.h.server.sched.prefix.hits;
    for (steps) |step| {
        try s.chat.turn(step, &ids, &text);
    }

    const chat = &s.chat;
    const full = chat.counts.get(.context_full);
    const refusal = chat.counts.get(.refusal);
    if (chat.verbose) {
        std.debug.print(
            "compactions: context_full {d}, refusal {d}, refusal kept {d}\n",
            .{ full, refusal, chat.kept.get(.refusal) },
        );
    }

    try std.testing.expect(full >= 1);
    try std.testing.expect(refusal >= 1);
    try std.testing.expect(full + refusal >= 3);
    try std.testing.expect(chat.kept.get(.refusal) >= 1);
    try std.testing.expect(chat.resent_after_kept >= 1);
    // A live cost never undercounts, so no append frame was refused.
    try std.testing.expectEqual(@as(u32, 0), chat.refits);
    // Every re-open restored the system prompt from the prefix cache.
    // The engine thread wrote the counter before it sent the reply that the test read last.
    try std.testing.expect(s.h.server.sched.prefix.hits - hits_before >= full + refusal);
}

test "an estimate that undercounts a kept pair refits the cycle with the learned cost" {
    const gpa = std.testing.allocator;
    var s: Setup = undefined;
    try s.boot(gpa);
    defer s.deinit(gpa);

    var ids: [max_seq]u32 = undefined;
    var text: [4096]u8 = undefined;
    for (steps[0..2]) |step| {
        try s.chat.turn(step, &ids, &text);
    }

    // A planted turn with a cost of one position. Its prompt alone holds far
    // more tokens than the window, so the plan keeps it and the server
    // refuses the frame.
    const t = s.chat.t;
    t.begin("a dragon " ** 40);
    _ = t.finish(1);
    try std.testing.expectEqual(@as(u32, 3), t.count());
    const p = try s.chat.compact(.planted, ct.reserveFor(max_seq, t.longest_cost));
    // A refusal names the sum of its frame only, so the excess lands on the oldest pair.
    // Each refit drops one pair, until the planted turn learns its own cost and leaves too.
    try std.testing.expect(s.chat.refits >= 1);
    try std.testing.expect(s.chat.refits <= 3);
    try std.testing.expectEqual(@as(u32, 0), p.kept);
    try std.testing.expectEqual(@as(u32, 0), t.count());

    // The session answers a turn after the refit.
    try s.chat.turn(steps[2], &ids, &text);
}
