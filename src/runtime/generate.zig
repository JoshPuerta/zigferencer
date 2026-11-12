//! The decode core: greedy generation from a runner. `run` and `bench-model` share this one
//! control loop.
//!
//! Comptime. `@hasDecl` and `@hasField` find the optional sink and runner hooks, so an absent hook
//! costs nothing.
//!
//! Ownership. The loop owns the sequencing (sample, step, stop) and the timing. The caller owns
//! the per-token effect and passes it in as a `sink` value, because boundaries are values. A sink
//! streams the detokenized piece, or folds the id into a hash. The caller picks the prefill because
//! it varies: per-token CPU, batched GPU, or cache restore. The two prefill loops live here too.
//! `decode` picks up from the `last_logits` and `pos` that prefill left.
//!
//! Sink. A sink is any value with `onToken(id: u32) !void`. The loop hands every sampled id to the
//! sink, the terminal EOS included. A hash therefore counts EOS, and a streamer can choose not to
//! print it. A sink can also declare `onStep(pos: u32, ns: u64) void`. The loop then hands it the
//! position each step wrote and the wall time since the previous mark. The first mark is the first
//! token. A sink without `onStep` makes no clock read per step.
//!
//! Stop conditions. The three stops are asymmetric on purpose. EOS and a full cache stop before
//! the loop steps the last token. The token budget stops after the step and commits the KV of that
//! token, so a follow-up turn resumes with it in context.

const std = @import("std");
const Io = std.Io;
const sampler = @import("sampler.zig");
const step_stats = @import("../core/step_stats.zig");
const time = @import("../core/utils/time.zig");
const span = @import("../core/span.zig");
const log = std.log.scoped(.run);

// -- types ------------------------------------------------------------------

/// Timing and extent of one decode. The fields match the per-turn value of `bench`.
pub const Generated = struct {
    ttft_ms: f64,
    /// Decode wall time from the first token to the stop, divided by `steps`. Zero with no step.
    itl_ms: f64,
    tokens_generated: u32,
    /// Runner steps the loop ran. An EOS or a full-cache stop leaves the last token unstepped.
    steps: u32,
    pos_end: u32,
    /// Host time of the decode host steps. A fused runner fills the first four, and every runner
    /// gets the argmax step. A zero count means the step did not run.
    host: step_stats.DecodeHost = .{},
};

/// The end of a prefill: the next position and the logits of the last row.
pub const Prefilled = struct {
    pos: u32,
    last_logits: []f32,
};

/// A sink or a runner arrives by pointer or by value. `@hasDecl` and `@hasField` need the struct.
fn Pointee(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |p| p.child,
        else => T,
    };
}

// -- api --------------------------------------------------------------------

/// Decode up to `max_tokens` ids greedily from `last_logits` at `pos`, and hand each one to
/// `sink.onToken`. The caller captures `t_start` before prefill, so TTFT spans prefill plus the
/// first token.
pub fn decode(
    runner: anytype,
    last_logits_in: []f32,
    pos_in: u32,
    max_tokens: u32,
    max_seq: u32,
    eos_id: u32,
    ignore_eos: bool,
    io: Io,
    t_start: Io.Clock.Timestamp,
    sink: anytype,
) !Generated {
    std.debug.assert(max_seq > 0);
    std.debug.assert(last_logits_in.len > 0);

    // Detect the optional sink and runner hooks at comptime.
    const has_on_step = comptime @hasDecl(Pointee(@TypeOf(sink)), "onStep");
    const host_out = decodeHost(runner);
    if (host_out) |h| {
        h.* = .{};
    }

    var argmax: step_stats.Stat = .{};

    log.debug(
        "decode start: pos {d}, budget {d} tokens, max_seq {d}",
        .{ pos_in, max_tokens, max_seq },
    );

    // Return at once when the budget is zero.
    var pos = pos_in;
    if (max_tokens == 0) {
        return .{
            .ttft_ms = time.msBetween(t_start, Io.Clock.Timestamp.now(io, .awake)),
            .itl_ms = 0,
            .tokens_generated = 0,
            .steps = 0,
            .pos_end = pos,
        };
    }

    // Sample and emit the first token.
    var last_logits = last_logits_in;
    var id = try timedGreedy(last_logits, io, &argmax);
    const t_first = Io.Clock.Timestamp.now(io, .awake);
    var t_mark = t_first;
    try sink.onToken(id);
    var n: u32 = 1;
    var steps: u32 = 0;

    // Step and sample until a stop condition holds.
    var stop: []const u8 = "eos";
    while (true) {
        if (!ignore_eos and id == eos_id) break; // EOS: stop before stepping it.

        if (pos >= max_seq) {
            stop = "cache_full";
            break; // Cache full: nowhere to write the next token.
        }

        const sp = span.begin(.decode_step);
        last_logits = try runner.step(id, pos);
        sp.end();
        steps += 1;
        if (comptime has_on_step) {
            const t_now = Io.Clock.Timestamp.now(io, .awake);
            sink.onStep(pos, time.elapsedNs(t_mark, t_now));
            t_mark = t_now;
        }

        pos += 1;
        if (n >= max_tokens) {
            stop = "budget";
            break; // Budget reached. The step above already committed it.
        }

        id = try timedGreedy(last_logits, io, &argmax);
        try sink.onToken(id);
        n += 1;
    }

    log.debug("decode stop: {s}, {d} tokens produced, {d} steps", .{ stop, n, steps });

    // Check the loop counters and build the result.
    const t_end = Io.Clock.Timestamp.now(io, .awake);

    std.debug.assert(n <= max_tokens);
    std.debug.assert(pos <= max_seq);
    std.debug.assert(pos - pos_in == steps);
    std.debug.assert(steps == n or steps + 1 == n);
    var host: step_stats.DecodeHost = if (host_out) |h| h.* else .{};
    host.argmax = argmax;
    return .{
        .ttft_ms = time.msBetween(t_start, t_first),
        .itl_ms = if (steps > 0)
            time.msBetween(t_first, t_end) / @as(f64, @floatFromInt(steps))
        else
            0,
        .tokens_generated = n,
        .steps = steps,
        .pos_end = pos,
        .host = host,
    };
}

/// Prefill `prompt_ids` from position zero in chunks of at most `prefill_batch_max` rows. Each
/// chunk writes its KV rows before its rows attend, so the chunks compose. The callers refuse a
/// prompt past `max_seq` first, so the bound at `max_seq` cuts nothing.
pub fn prefillChunks(runner: anytype, prompt_ids: []const u32) !Prefilled {
    const window: u32 = runner.max_seq;
    var pos: u32 = 0;
    var last_logits: []f32 = undefined;

    // Each pass feeds at least one token or stops, so the loop ends in
    // `prompt_ids.len` passes.
    while (pos < prompt_ids.len and pos < window) {
        const room = @min(runner.prefill_batch_max, window - pos);
        const remaining = prompt_ids.len - pos;
        const chunk: u32 = @intCast(@min(remaining, @as(usize, room)));

        const sp = span.beginArg(.prefill_chunk, chunk);
        last_logits = try runner.prefillBatch(prompt_ids[pos..][0..chunk], pos);
        sp.end();

        pos += chunk;
    }

    return .{ .pos = pos, .last_logits = last_logits };
}

/// Prefill `prompt_ids` one `step` per token, for a runner without `chunked_prefill`. The callers
/// refuse a prompt past `max_seq` first, so the bound at `max_seq` cuts nothing.
pub fn prefillTokens(runner: anytype, prompt_ids: []const u32) !Prefilled {
    const window: u32 = runner.max_seq;
    var pos: u32 = 0;
    var last_logits: []f32 = undefined;

    for (prompt_ids) |id| {
        if (pos >= window) break;

        const sp = span.beginArg(.prefill_chunk, 1);
        last_logits = try runner.step(id, pos);
        sp.end();

        pos += 1;
    }

    return .{ .pos = pos, .last_logits = last_logits };
}

// -- helpers ----------------------------------------------------------------

/// Greedy argmax with its wall time folded into `stat`. A row with a NaN, a +inf, or no finite
/// logit ends the turn with an error, because argmax over it names no real token.
fn timedGreedy(logits: []const f32, io: Io, stat: *step_stats.Stat) sampler.Error!u32 {
    sampler.checkLogits(logits) catch |err| {
        log.err("the logits of a step hold a NaN, a +inf, or no finite value. " ++
            "A model weight can hold a non-finite value", .{});
        return err;
    };

    const t0 = Io.Clock.Timestamp.now(io, .awake);
    const id = sampler.greedy(logits);
    const t1 = Io.Clock.Timestamp.now(io, .awake);
    stat.add(time.elapsedNs(t0, t1));
    span.record(.sample, t0.raw, t1.raw);
    return id;
}

/// The decode host times of a runner that declares them, or null.
fn decodeHost(runner: anytype) ?*step_stats.DecodeHost {
    const R = Pointee(@TypeOf(runner));
    if (comptime @hasDecl(R, "decodeHost")) return runner.decodeHost();
    return null;
}

// -- tests ------------------------------------------------------------------

// A canned-logits runner plus a recording sink test the control loop without a model.
// Each fixed logits vector names the next token, and the recorder captures what the loop emitted.
const FakeRunner = struct {
    seq: []const []f32,
    i: usize = 0,

    fn step(self: *FakeRunner, id: u32, pos: u32) ![]f32 {
        _ = id;
        _ = pos;
        defer self.i += 1;

        return self.seq[self.i];
    }
};

const Recorder = struct {
    buf: []u32,
    n: usize = 0,

    fn onToken(self: *Recorder, id: u32) !void {
        self.buf[self.n] = id;
        self.n += 1;
    }
};

test "decode: greedy walk, EOS is emitted then ends the run (not stepped)" {
    const io = std.testing.io;
    var l0 = [_]f32{ 0, 9, 0, 0 }; // -> 1
    var l1 = [_]f32{ 0, 0, 9, 0 }; // -> 2
    var l2 = [_]f32{ 0, 0, 0, 9 }; // -> 3 (eos)
    var seq = [_][]f32{ &l1, &l2 };
    var fake: FakeRunner = .{ .seq = &seq };
    var rec_buf: [8]u32 = undefined;
    var rec: Recorder = .{ .buf = &rec_buf };
    const g = try decode(
        &fake,
        &l0,
        0,
        16,
        100,
        3,
        false,
        io,
        Io.Clock.Timestamp.now(io, .awake),
        &rec,
    );
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, rec.buf[0..rec.n]);
    try std.testing.expectEqual(@as(u32, 3), g.tokens_generated);
    try std.testing.expectEqual(@as(u32, 2), g.steps); // N - 1: the EOS is never stepped
    try std.testing.expectEqual(@as(u32, 2), g.pos_end); // stepped 1 and 2, not the EOS 3
}

test "decode: budget stop commits the last token's step" {
    const io = std.testing.io;
    var l0 = [_]f32{ 9, 0, 0, 0 }; // -> 0
    var l1 = [_]f32{ 0, 9, 0, 0 }; // -> 1
    var seq = [_][]f32{ &l1, &l1 }; // step(0)->1, then the trailing commit step
    var fake: FakeRunner = .{ .seq = &seq };
    var rec_buf: [8]u32 = undefined;
    var rec: Recorder = .{ .buf = &rec_buf };
    const g = try decode(
        &fake,
        &l0,
        0,
        2,
        100,
        99,
        false,
        io,
        Io.Clock.Timestamp.now(io, .awake),
        &rec,
    );
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, rec.buf[0..rec.n]);
    try std.testing.expectEqual(@as(u32, 2), g.tokens_generated);
    try std.testing.expectEqual(
        @as(u32, 2),
        g.steps,
    ); // N: the budget stop steps the last token too
    try std.testing.expectEqual(@as(u32, 2), g.pos_end); // both tokens stepped (KV committed)
}

test "decode: stops at the cache cap without stepping past it" {
    const io = std.testing.io;
    var l0 = [_]f32{ 0, 9, 0, 0 }; // -> 1
    var l1 = [_]f32{ 0, 0, 9, 0 }; // -> 2
    var seq = [_][]f32{&l1};
    var fake: FakeRunner = .{ .seq = &seq };
    var rec_buf: [8]u32 = undefined;
    var rec: Recorder = .{ .buf = &rec_buf };
    // pos 4, cap 5: emit 1, step (pos->5), emit 2, pos>=cap -> stop.
    const g = try decode(
        &fake,
        &l0,
        4,
        16,
        5,
        99,
        false,
        io,
        Io.Clock.Timestamp.now(io, .awake),
        &rec,
    );
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, rec.buf[0..rec.n]);
    try std.testing.expectEqual(@as(u32, 1), g.steps);
    try std.testing.expectEqual(@as(u32, 5), g.pos_end);
}

// A sink that declares `onStep` records every step the loop reports.
const StepRecorder = struct {
    tokens: u32 = 0,
    pos: [8]u32 = undefined,
    n_steps: u32 = 0,

    fn onToken(self: *StepRecorder, id: u32) !void {
        _ = id;
        self.tokens += 1;
    }

    fn onStep(self: *StepRecorder, pos: u32, ns: u64) void {
        _ = ns;
        self.pos[self.n_steps] = pos;
        self.n_steps += 1;
    }
};

test "decode: onStep fires once per step with the position the step wrote" {
    const io = std.testing.io;
    var l0 = [_]f32{ 9, 0, 0, 0 }; // -> 0
    var l1 = [_]f32{ 0, 9, 0, 0 }; // -> 1
    var l3 = [_]f32{ 0, 0, 0, 9 }; // -> 3 (eos)

    // A budget stop at N = 3 from pos 7: three steps at 7, 8, 9.
    var seq_budget = [_][]f32{ &l1, &l1, &l1 };
    var fake_budget: FakeRunner = .{ .seq = &seq_budget };
    var budget: StepRecorder = .{};
    const gb = try decode(
        &fake_budget,
        &l0,
        7,
        3,
        100,
        99,
        false,
        io,
        Io.Clock.Timestamp.now(io, .awake),
        &budget,
    );
    try std.testing.expectEqual(@as(u32, 3), gb.steps);
    try std.testing.expectEqual(gb.steps, budget.n_steps);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 9 }, budget.pos[0..budget.n_steps]);
    try std.testing.expectEqual(@as(u32, 10), gb.pos_end);

    // An EOS stop at N = 3 from pos 2: two steps at 2 and 3, the EOS unstepped.
    var seq_eos = [_][]f32{ &l1, &l3 };
    var fake_eos: FakeRunner = .{ .seq = &seq_eos };
    var eos: StepRecorder = .{};
    const ge = try decode(
        &fake_eos,
        &l0,
        2,
        16,
        100,
        3,
        false,
        io,
        Io.Clock.Timestamp.now(io, .awake),
        &eos,
    );
    try std.testing.expectEqual(@as(u32, 3), eos.tokens);
    try std.testing.expectEqual(@as(u32, 2), ge.steps);
    try std.testing.expectEqual(ge.steps, eos.n_steps);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3 }, eos.pos[0..eos.n_steps]);
}
