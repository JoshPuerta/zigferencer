//! `zigferencer-dev bench-decode-host <pub_name>` splits the cost of one decode step
//! into the host pieces the fused runners pay. A step records one command
//! buffer, submits it, waits for the queue, copies the logits, and runs an
//! argmax. The command times four questions apart:
//!
//! Comptime. Generic over the model family and the runner. The fused build option picks the GPU
//! path.
//!
//!   record  how long does the record pass take with the profiler off?
//!   resub   what is the step floor when the buffer records once and replays?
//!   sync    what does one submit and wait cost for an empty buffer, with a
//!           queue-idle wait and with a fence?
//!   gap     how long does the GPU sit idle between two decode steps?
//!
//! The resub mode replays one recording, so the KV row and the recurrent state
//! change on every replay. Its output has no meaning and only its time counts.
//! The command needs a fused-GPU build, because only those runners record one
//! command buffer per step.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const vk = if (build_options.use_vulkan_fused_ops) @import("vulkan") else void;
const build_options = @import("build_options");

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;
const style = zigferencer.cli.style;
const select = zigferencer.cli.select;
const boot_resources = zigferencer.cli.boot_resources;
const registry = zigferencer.registry;
const loader = zigferencer.loader;
const limits = zigferencer.limits;
const kv_budget = zigferencer.kv_budget;
const dtype = zigferencer.dtype;
const step_stats = zigferencer.step_stats;
const family = zigferencer.family;
const sampler = zigferencer.sampler;
const log = std.log.scoped(.bench_decode_host);

const vulkan_ctx_mod = if (build_options.use_vulkan_fused_ops)
    zigferencer.vulkan_ctx
else
    void;
const kernels = if (build_options.use_vulkan_fused_ops)
    zigferencer.ops_vulkan
else
    void;

// -- constants --------------------------------------------------------------

/// Bound on `--reps`. It also bounds the timestamp readback array.
const reps_max: u32 = 1024;

/// Untimed repetitions before each timed series.
const warm_reps: u32 = 3;

// -- types ------------------------------------------------------------------

const Mode = enum { record, resubmit, sync, gap, all };

/// What every mode of one bench reads.
const Bench = if (build_options.use_vulkan_fused_ops) struct {
    ctx: context.Context,
    ctxp: *vulkan_ctx_mod.Context,
    loaded: *const loader.Loaded,
    model: *const registry.Model,
    opt: Opt,
    ceiling: u32,
    window: u32,
    w: *Io.Writer,
} else void;

/// The flags of `bench-decode-host`.
const Opt = struct {
    pub_name: ?[]const u8 = null,
    pos: u32 = 2000,
    reps: u32 = 200,
    mode: Mode = .all,
    kv_dtype: ?dtype.KvDtype = null,

    pub const positional = "pub_name";
    pub const help: args.Help = .{
        .usage = "zigferencer-dev bench-decode-host <pub_name> [flags]   (fused-GPU build only)",
        .about = "Split one decode step into its host steps.\n",
        .footer =
        \\modes:
        \\  record    cost of recording one decode command buffer, no submit
        \\  resubmit  floor of a resubmit: record once, submit and wait many times
        \\  sync      floor of a wait: empty buffer, queue wait against fence wait
        \\  gap       GPU idle gap between decode steps, from device timestamps
        \\
        \\Modes resubmit and gap change the KV row and the recurrent state. Their output is
        \\not meaningful, so read the times only.
        \\On a driver whose timestamp counter pauses while the GPU idles, the gap
        \\reads low.
        \\
        ,
    };
    pub const docs = .{
        .pos = args.Doc{ .text = "KV depth to prefill before timing, at least 1" },
        .reps = args.Doc{
            .text = "timed repetitions per mode, 1..1024. The depth plus\n" ++
                "reps plus 4 positions must fit the model ceiling",
        },
        .mode = args.Doc{ .text = "the host step to time, or all of them (see modes below)" },
        .kv_dtype = args.Doc{ .text = "KV element type", .default = "the model's registered one" },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const opt = try args.parseInto(Opt, ctx) orelse return;
    if (opt.reps == 0 or opt.reps > reps_max) {
        return args.fail(ctx.io, "--reps must be in 1..{d}", .{reps_max});
    }
    if (opt.pos == 0) {
        return args.fail(
            ctx.io,
            "--pos must be >= 1 (the decode step attends to committed rows)",
            .{},
        );
    }

    if (comptime build_options.use_vulkan_fused_ops) return runFused(ctx, opt.pub_name, opt);

    log.err(
        "bench-decode-host times the fused-GPU decode step. " ++
            "Build -Druntime=vulkan -Dops-backend=vulkan_fused",
        .{},
    );
    return error.UnsupportedServeBackend;
}

/// Positions the modes write past `--pos`: the warm steps and the timed steps
/// of the gap mode, plus the row of the next token.
pub fn windowFor(pos: u32, reps: u32) u32 {
    std.debug.assert(reps >= 1);
    std.debug.assert(reps <= reps_max);

    return pos + warm_reps + reps + 1;
}

/// Idle time between the end of step `i` and the start of step `i + 1`, in
/// nanoseconds, from raw device ticks. A tick pair out of order means the
/// queue wrote the stamps wrong, so the caller must not read a number.
pub fn gapNs(end_prev: u64, start_next: u64, period_ns: f64) error{StampsOutOfOrder}!u64 {
    if (start_next < end_prev) return error.StampsOutOfOrder;
    return @trunc(@as(f64, @floatFromInt(start_next - end_prev)) * period_ns);
}

// -- helpers ----------------------------------------------------------------

fn runFused(ctx: context.Context, pub_name: ?[]const u8, opt: Opt) !void {
    // Load the model and check that the window fits.
    const model = try select.model(ctx, pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);

    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    const ceiling = limits.contextCeiling(loaded.card.context_length, model.context_limit);
    const window = windowFor(opt.pos, opt.reps);
    if (window > ceiling) {
        return args.fail(
            ctx.io,
            "--pos {d} plus {d} decode steps passes the {d}-token model ceiling. " ++
                "Lower --pos or --reps",
            .{ opt.pos, window - opt.pos, ceiling },
        );
    }

    // Open the device and pick the family.
    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, ctx.vk_options);
    defer ctxp.deinit();

    const bench: Bench = .{
        .ctx = ctx,
        .ctxp = ctxp,
        .loaded = &loaded,
        .model = model,
        .opt = opt,
        .ceiling = ceiling,
        .window = window,
        .w = &ctx.stdout.interface,
    };
    switch (loaded.card.arch) {
        inline else => |arch| return benchFamily(family.Of(arch), &bench),
    }
}

/// Plan one slot at the window the modes need, then build the fused runner of
/// `Family` and time it.
fn benchFamily(comptime Family: type, b: *const Bench) !void {
    const ctx = b.ctx;
    const card = b.loaded.card;
    const kv_dtype = boot_resources.kvDtypeFor(b.opt.kv_dtype, b.model);

    const vram_free = boot_resources.vramFigure(b.ctxp);
    const caps = try Family.kvCaps(ctx.gpa, b.loaded, b.ceiling, vram_free, 1, kv_dtype);
    const cost = Family.Fused.kvCostModel(card, kv_dtype);
    const plan = try kv_budget.kvPlan(caps, cost, .{ .window = b.window, .slots = 1, .batch = 1 });

    const runner = try Family.initFused(ctx.gpa, b.ctxp, b.loaded, plan, kv_dtype);
    defer runner.deinit();

    return benchOne(Family.Fused, runner, b);
}

fn benchOne(comptime RunnerT: type, runner: *RunnerT, b: *const Bench) !void {
    const ctx = b.ctx;
    const ctxp = b.ctxp;
    const opt = b.opt;
    const vocab = b.loaded.card.vocab_size;
    const w = b.w;
    std.debug.assert(runner.max_seq >= b.window);

    // Print the header.
    const header = "model:    {s}   pos: {d}   reps: {d}   window: {d} tok   " ++
        "timestamp period: {d:.1} ns\n";
    try w.print(header, .{
        b.model.pub_name,
        opt.pos,
        opt.reps,
        runner.max_seq,
        ctxp.physical_device_props.limits.timestamp_period,
    });
    try w.flush();

    // Prefill to depth `--pos`. The tokens are synthetic, because every timing is data independent.
    var last_logits: []f32 = undefined;
    var pos: u32 = 0;
    while (pos < opt.pos) {
        const chunk: u32 = @min(runner.prefill_batch_max, opt.pos - pos);
        const ids = try ctx.arena.alloc(u32, chunk);
        for (ids, 0..) |*t, i| {
            t.* = @intCast(((pos + i) * 131 + 7) % vocab);
        }

        last_logits = try runner.prefillBatch(ids, pos);
        pos += chunk;
    }

    std.debug.assert(pos == opt.pos);
    const first_token = sampler.greedy(last_logits);

    // Run the selected modes.
    if (opt.mode == .sync or opt.mode == .all) {
        try syncFloor(ctxp, ctx.io, opt.reps, w);
    }

    if (opt.mode == .record or opt.mode == .all) {
        try recordCost(RunnerT, runner, ctx.io, opt, w);
    }

    if (opt.mode == .gap or opt.mode == .all) {
        try gapProbe(RunnerT, runner, ctx.io, opt, first_token, w);
    }

    if (opt.mode == .resubmit or opt.mode == .all) {
        try resubmit(RunnerT, runner, ctx.io, opt, w);
    }

    try w.flush();
}

/// Record the decode buffer `reps` times with the profiler off and no submit.
fn recordCost(comptime RunnerT: type, runner: *RunnerT, io: Io, opt: Opt, w: *Io.Writer) !void {
    for (0..warm_reps) |_| {
        try runner.probeRecord(opt.pos);
    }

    var stat: step_stats.Stat = .{};
    for (0..opt.reps) |_| {
        const t0 = Io.Clock.Timestamp.now(io, .awake);
        try runner.probeRecord(opt.pos);
        stat.add(nsBetween(t0, Io.Clock.Timestamp.now(io, .awake)));
    }

    try w.print(
        "\nrecord cost (profiler off, no submit, n={d}, pos={d})\n",
        .{ opt.reps, opt.pos },
    );
    try printStat(w, "record", stat);
}

/// Record once without the one-time flag, then submit and wait `reps` times.
fn resubmit(comptime RunnerT: type, runner: *RunnerT, io: Io, opt: Opt, w: *Io.Writer) !void {
    try runner.probeRecordReusable(opt.pos);
    for (0..warm_reps) |_| {
        try runner.probeSubmitWait();
    }

    var stat: step_stats.Stat = .{};
    for (0..opt.reps) |_| {
        const t0 = Io.Clock.Timestamp.now(io, .awake);
        try runner.probeSubmitWait();
        stat.add(nsBetween(t0, Io.Clock.Timestamp.now(io, .awake)));
    }

    try w.print(
        "\nresubmit floor (one recording, submit + queue wait, n={d}, pos={d})\n",
        .{ opt.reps, opt.pos },
    );
    try w.writeAll(
        "  the output of these steps is not meaningful: " ++
            "every replay rewrites the KV row and the recurrent state\n",
    );
    try printStat(w, "submit+wait", stat);
}

/// An empty command buffer submitted `reps` times, once per wait style.
fn syncFloor(ctxp: *vulkan_ctx_mod.Context, io: Io, reps: u32, w: *Io.Writer) !void {
    // Record the empty buffer and make the fence.
    const vkd = ctxp.vkd;
    const cb = try kernels.allocateOnePrimaryCmdBuffer(ctxp);
    defer vkd.freeCommandBuffers(ctxp.device, ctxp.command_pool, &[_]vk.CommandBuffer{cb});

    try vkd.beginCommandBuffer(cb, &.{});
    try vkd.endCommandBuffer(cb);

    const fence = try vkd.createFence(ctxp.device, &.{}, null);
    defer vkd.destroyFence(ctxp.device, fence, null);

    const cmds = [_]vk.CommandBuffer{cb};
    const submits = [_]vk.SubmitInfo{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }};
    const fences = [_]vk.Fence{fence};

    // Time the queue-idle wait.
    var idle: step_stats.Stat = .{};
    for (0..warm_reps + reps) |i| {
        const t0 = Io.Clock.Timestamp.now(io, .awake);
        try vkd.queueSubmit(ctxp.queue, &submits, .null_handle);
        try vkd.queueWaitIdle(ctxp.queue);
        if (i >= warm_reps) {
            idle.add(nsBetween(t0, Io.Clock.Timestamp.now(io, .awake)));
        }
    }

    // Time the fence wait.
    var fenced: step_stats.Stat = .{};
    for (0..warm_reps + reps) |i| {
        const t0 = Io.Clock.Timestamp.now(io, .awake);
        try vkd.queueSubmit(ctxp.queue, &submits, fence);
        const r = try vkd.waitForFences(ctxp.device, &fences, .true, std.math.maxInt(u64));
        std.debug.assert(r == .success);
        try vkd.resetFences(ctxp.device, &fences);
        if (i >= warm_reps) {
            fenced.add(nsBetween(t0, Io.Clock.Timestamp.now(io, .awake)));
        }
    }

    try w.print("\nsync floor (empty command buffer, n={d})\n", .{reps});
    try printStat(w, "queue wait", idle);
    try printStat(w, "fence wait", fenced);
}

/// Run `reps` normal decode steps whose buffers carry two timestamps each. The
/// loop includes the argmax between steps, because a real decode pays it.
fn gapProbe(
    comptime RunnerT: type,
    runner: *RunnerT,
    io: Io,
    opt: Opt,
    first_token: u32,
    w: *Io.Writer,
) !void {
    // Warm the decode path.
    var id = first_token;
    var pos = opt.pos;
    for (0..warm_reps) |_| {
        id = sampler.greedy(try runner.step(id, pos));
        pos += 1;
    }

    // Raise the ring for the timed steps. It writes the start and end stamps at `submit` and above.
    const level = runner.decode.level();

    if (level == .off) {
        try runner.setDecodeLevel(.submit);
    }

    defer if (level == .off) runner.setDecodeLevel(.off) catch unreachable;

    // Run the timed steps and keep the stamps of each.
    runner.host = .{};
    var ticks: [2 * reps_max]u64 = undefined;
    var period: f64 = 0;
    const t_all = Io.Clock.Timestamp.now(io, .awake);
    for (0..opt.reps) |i| {
        const logits = try runner.step(id, pos);
        const r = runner.lastDecodeReadout().?;
        ticks[2 * i] = r.gpu_start;
        ticks[2 * i + 1] = r.gpu_end;
        period = r.period_ns;
        id = sampler.greedy(logits);
        pos += 1;
    }

    const wall_ms = @as(
        f64,
        @floatFromInt(nsBetween(t_all, Io.Clock.Timestamp.now(io, .awake))),
    ) / 1e6;
    std.debug.assert(pos <= runner.max_seq);

    // Turn the stamps into gaps.
    var gap: step_stats.Stat = .{};
    var busy: step_stats.Stat = .{};
    for (0..opt.reps) |i| {
        busy.add(try gapNs(ticks[2 * i], ticks[2 * i + 1], period));
        if (i + 1 < opt.reps) {
            gap.add(try gapNs(ticks[2 * i + 1], ticks[2 * i + 2], period));
        }
    }

    // Print the gap, the busy time, and the host steps.
    try w.print(
        "\ngap: GPU idle between steps (n={d} steps, {d} gaps, pos={d}..{d}, argmax included)\n",
        .{ opt.reps, gap.count, opt.pos + warm_reps, pos - 1 },
    );
    try printStat(w, "gpu idle gap", gap);
    try printStat(w, "gpu busy", busy);
    try w.print(
        "  step wall       avg={d:.3} ms (record, submit, wait, copy, argmax)\n",
        .{wall_ms / @as(f64, @floatFromInt(opt.reps))},
    );
    inline for (std.meta.fields(step_stats.DecodeHost)) |f| {
        if (f.type != step_stats.Stat) continue;
        if (comptime std.mem.eql(u8, f.name, "argmax")) continue;
        try printStat(w, "host " ++ f.name, @field(runner.host, f.name));
    }

    const r = runner.host.ring;
    try w.print(
        "  record ring     hits={d}  misses={d}  speculated={d}  wasted={d}\n",
        .{ r.hits, r.misses, r.speculated, r.wasted },
    );
}

fn printStat(w: *Io.Writer, label: []const u8, s: step_stats.Stat) !void {
    try w.print(
        "  {s:<15} avg={d:.3}  min={d:.3}  max={d:.3} ms\n",
        .{ label, s.avgMs(), s.minMs(), s.maxMs() },
    );
}

fn nsBetween(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) u64 {
    return @intCast(t0.durationTo(t1).raw.toNanoseconds());
}

// -- tests ------------------------------------------------------------------

test "windowFor: depth plus warm steps plus timed steps plus one" {
    try std.testing.expectEqual(@as(u32, 2000 + warm_reps + 200 + 1), windowFor(2000, 200));
    try std.testing.expectEqual(@as(u32, 1 + warm_reps + 1 + 1), windowFor(1, 1));
}

test "gapNs: ticks scale by the period, a reversed pair is refused" {
    try std.testing.expectEqual(@as(u64, 2500), try gapNs(100, 200, 25.0));
    try std.testing.expectEqual(@as(u64, 0), try gapNs(100, 100, 25.0));
    try std.testing.expectError(error.StampsOutOfOrder, gapNs(200, 100, 25.0));
}
