//! `zigferencer-dev bench-swap <pub_name>` times one KV slot's evict (D2H) and restore
//! (H2D) against a decode tick. The numbers weigh the synchronous swap stall `t + s`
//! against the per-token budget. The bench needs a fused-GPU build, because the swap
//! is a VRAM to RAM transfer that no CPU engine has.
//!
//! Comptime. Generic over the model family and the runner. Build options pick the fused path.
//!
//! Two windows answer whether `s` is transfer or fixed overhead: `--budget` off takes
//! the serve default of the model, and on takes the larger VK_EXT_memory_budget window.
//! The change in `s` across the slot sizes is the transfer slope.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const build_options = @import("build_options");

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;
const select = zigferencer.cli.select;
const boot_resources = zigferencer.cli.boot_resources;
const boot_lines = zigferencer.cli.boot_lines;
const family = zigferencer.family;
const dtype = zigferencer.dtype;
const kv_budget = zigferencer.kv_budget;
const limits = zigferencer.limits;
const registry = zigferencer.registry;
const loader = zigferencer.loader;
const Sessions = zigferencer.session.Sessions;

const log = std.log.scoped(.bench_swap);

const fused_build = build_options.use_vulkan_fused_ops;

// The device code resolves to `void` on a CPU build, which never analyzes it.
const vulkan_ctx_mod = if (fused_build) zigferencer.vulkan_ctx else void;
const kernels = if (fused_build) zigferencer.ops_vulkan else void;

// -- constants --------------------------------------------------------------

/// Bound on `--reps`. It sizes the sample array of the decode timing.
const reps_max: u32 = 256;

/// Untimed decode ticks before each timed series.
const warm_ticks: u32 = 3;

/// Untimed swaps before the timed series.
const swap_warm_reps = 5;

/// Bound on `--chunk-mib`. The sweep reaches 256 MiB, and the bound leaves one
/// step of headroom above it.
const chunk_mib_max: u32 = 1024;

/// Bytes of `/proc/self/status` the probe reads. The file is about 1.5 KiB.
const status_text_max: usize = 4096;

/// Bound on the fd walk of one probe. A count at the bound prints as a floor.
const fd_count_max: u32 = 1 << 16;

// -- types ------------------------------------------------------------------

/// The flags of `bench-swap`.
const Opt = struct {
    pub_name: ?[]const u8 = null,
    streams: u32 = 2,
    pos: u32 = 256,
    reps: u32 = 50,
    budget: bool = false,
    /// `--kv-dtype`: the element type every KV row frames at. It prices the blob
    /// this bench moves. Absent takes the model's default.
    kv_dtype: ?dtype.KvDtype = null,
    /// `--chunk-mib`: the chunk one staged copy moves, in MiB. The sweep that
    /// fixes `limits.swap_stage_chunk_bytes` sets it.
    chunk_mib: u32 = @intCast(limits.swap_stage_chunk_bytes >> 20),
    /// The resolved dtype, set once the model states its capability set. It is
    /// no flag.
    resolved_kv_dtype: dtype.KvDtype = .f32,

    pub const positional = "pub_name";
    pub const omit: []const []const u8 = &.{"resolved_kv_dtype"};
    pub const help: args.Help = .{
        .usage = "zigferencer-dev bench-swap <pub_name> [flags]   (fused-GPU build only)",
        .about =
        \\Time the evict (D2H) and the restore (H2D) of one slot against a decode
        \\tick, so the synchronous swap stall weighs against the per-token budget.
        \\
        ,
    };
    pub const docs = .{
        .streams = args.Doc{ .text = "physical KV slots to allocate, 2..8" },
        .pos = args.Doc{
            .text = "KV depth to fill before timing, at least 1. The depth\n" ++
                "plus 2 x (3 + reps) decode steps must fit the window",
        },
        .reps = args.Doc{ .text = "timed repetitions per measurement, 1..256" },
        .budget = args.Doc{
            .text = std.fmt.comptimePrint(
                "size the window from VK_EXT_memory_budget (else the\n" ++
                    "{d} MiB serve session budget, capped by the model ceiling)",
                .{limits.serve_session_kv_bytes >> 20},
            ),
        },
        .kv_dtype = args.Doc{
            .text = "KV element type. It prices the blob a swap moves",
            .default = "the model's registered one",
        },
        .chunk_mib = args.Doc{
            .text = std.fmt.comptimePrint(
                "the chunk one staged copy moves, 1..{d} MiB. The run also\n" ++
                    "checks that the chunked copies equal one whole-blob copy\nbyte for byte",
                .{chunk_mib_max},
            ),
        },
    };
};

/// What one bench reads. The arch switch of `runFused` picks the family.
const Bench = if (fused_build) struct {
    ctx: context.Context,
    ctxp: *vulkan_ctx_mod.Context,
    loaded: *const loader.Loaded,
    ceiling: u32,
    model_name: []const u8,
    opt: Opt,
    w: *Io.Writer,
} else void;

/// The timed decode ticks, in ms: the median at batch=streams and at batch=1.
const DecodeTimes = struct {
    batch: f64,
    one: f64,
};

/// One sample per timed swap, in ms. The part series are empty on a runner that
/// does not split its swaps.
const SwapTimes = struct {
    evict: []f64,
    restore: []f64,
    evict_dma: []f64,
    evict_copy: []f64,
    restore_dma: []f64,
    restore_copy: []f64,
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const opt = try args.parseInto(Opt, ctx) orelse return;
    if (opt.streams < 2 or opt.streams > limits.max_batch) {
        return args.fail(
            ctx.io,
            "--streams must be in 2..{d} (multi-stream KV is what swaps)",
            .{limits.max_batch},
        );
    }
    if (opt.reps == 0 or opt.reps > reps_max) {
        return args.fail(ctx.io, "--reps must be in 1..{d}", .{reps_max});
    }
    if (opt.chunk_mib == 0 or opt.chunk_mib > chunk_mib_max) {
        return args.fail(ctx.io, "--chunk-mib must be in 1..{d}", .{chunk_mib_max});
    }
    // An eviction moves the committed rows, so a zero depth moves nothing.
    if (opt.pos == 0) {
        return args.fail(ctx.io, "--pos must be >= 1 (the swap moves the committed KV rows)", .{});
    }

    if (comptime fused_build) return runFused(ctx, opt);

    log.err("bench-swap measures a VRAM<->RAM KV swap. " ++
        "Build -Druntime=vulkan -Dops-backend=vulkan_fused", .{});
    return error.UnsupportedServeBackend;
}

// -- helpers ----------------------------------------------------------------

fn runFused(ctx: context.Context, opt_in: Opt) !void {
    // Load the model and resolve the KV type.
    const model = try select.model(ctx, opt_in.pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    // The KV dtype resolves ahead of every price, plan, and device call, so a
    // refusal costs a GGUF map and nothing more.
    var opt = opt_in;
    opt.resolved_kv_dtype = boot_resources.kvDtypeFor(opt.kv_dtype, model);

    // Open the device and pick the family.
    const ctxp = try boot_resources.openDevice(ctx);
    defer ctxp.deinit();

    const bench: Bench = .{
        .ctx = ctx,
        .ctxp = ctxp,
        .loaded = &loaded,
        .ceiling = limits.contextCeiling(loaded.card.context_length, model.context_limit),
        .model_name = model.pub_name,
        .opt = opt,
        .w = &ctx.stdout.interface,
    };
    switch (loaded.card.arch) {
        inline else => |arch| return benchOne(family.Of(arch), &bench),
    }
}

/// The window is the point of the bench. `--budget` derives the budget from the
/// live free VRAM, and the slot count's share of it buys the window. Without the
/// flag the planner takes the serve default of the model.
fn planFor(comptime Family: type, b: *const Bench) !kv_budget.KvPlan {
    const opt = b.opt;
    const kv_dtype = opt.resolved_kv_dtype;

    const free: u64 = if (opt.budget) blk: {
        const budget = b.ctxp.deviceLocalBudget() orelse break :blk 0;
        break :blk budget.budget -| budget.usage;
    } else 0;

    const caps = Family.Fused.kvCaps(
        b.loaded,
        b.ceiling,
        free,
        b.ctxp.driver,
        opt.streams,
        kv_dtype,
    );
    const plan = try kv_budget.kvPlan(caps, Family.Fused.kvCostModel(b.loaded.card, kv_dtype), .{
        .x_bytes = if (opt.budget) kv_budget.derivedX(caps) else null,
        .slots = opt.streams,
        .batch = opt.streams,
    });
    boot_lines.logPlan(plan, kv_dtype);
    return plan;
}

/// Positions the decode timing writes into slot 0 past `--pos`. Slot 0 decodes
/// in both series, at batch=streams and at batch=1, and each tick advances it.
fn decodeTicks(reps: u32) u32 {
    std.debug.assert(reps >= 1);
    std.debug.assert(reps <= reps_max);

    return 2 * (warm_ticks + reps);
}

fn benchOne(comptime Family: type, b: *const Bench) !void {
    const Runner = Family.Fused;
    const ctx = b.ctx;
    const opt = b.opt;
    const w = b.w;
    const pos = opt.pos;

    // Plan the window first, so a refused depth costs no VRAM.
    const plan = try planFor(Family, b);
    const window = plan.window;
    const ticks = decodeTicks(opt.reps);
    if (@as(u64, pos) + ticks > window) {
        return args.fail(
            ctx.io,
            "--pos {d} plus {d} decode steps passes the {d}-token window. Lower --pos or --reps",
            .{ pos, ticks, window },
        );
    }

    // Print the header and build the runner.
    try w.print("model:    {s}   streams: {d}   pos: {d}   reps: {d}   window: {d} tok ({s})\n", .{
        b.model_name, opt.streams, pos, opt.reps, window, if (opt.budget) "budget" else "default",
    });

    const runner = try Family.Fused.init(ctx.gpa, b.ctxp, b.loaded, plan, opt.resolved_kv_dtype);
    defer runner.deinit();

    // Set the chunk before the first swap, because a stage only grows.
    runner.commands.swap_chunk_bytes = @as(u64, opt.chunk_mib) << 20;
    std.debug.assert(runner.max_seq == window);
    std.debug.assert(pos >= 1);
    std.debug.assert(pos + ticks <= window);

    // Size the slot by the full window, and the blob by the committed rows.
    const slot_bytes: usize = @intCast(runner.evictSlotBytesFor(window));
    const blob_bytes: usize = @intCast(runner.evictSlotBytesFor(pos));
    try printSizes(b, slot_bytes, blob_bytes);
    try printProbe(ctx.io, b.ctxp, "start", w);

    // Fill every slot to depth `pos` with a dummy token, then time a decode tick.
    var pool = try Sessions(Runner).init(ctx.arena, runner);
    const ids = try fillSlots(Runner, b, &pool);
    try printProbe(ctx.io, b.ctxp, "fill", w);

    const out = try ctx.arena.alloc(u32, opt.streams);
    const t_batch = try timeDecode(Runner, ctx.io, &pool, ids, out, opt.reps);
    const t_one = try timeDecode(Runner, ctx.io, &pool, ids[0..1], out[0..1], opt.reps);

    // Time the swaps and the host copy, and check the chunked copies.
    const blob = try ctx.arena.alloc(u8, blob_bytes);
    const blob2 = try ctx.arena.alloc(u8, blob_bytes);
    const swaps = try timeSwaps(Runner, b, runner, blob);
    const copies = try timeHostCopy(b, blob, blob2);

    try checkChunks(b, runner, blob, blob2);
    try printProbe(ctx.io, b.ctxp, "end", w);

    const decode: DecodeTimes = .{ .batch = t_batch, .one = t_one };
    return printReport(Runner, b, decode, swaps, copies, blob_bytes);
}

/// The sizes of one swap, and how many slots the live budget affords.
fn printSizes(b: *const Bench, slot_bytes: usize, blob_bytes: usize) !void {
    const w = b.w;

    try w.print("slot KV:  {d:.1} MiB   blob at pos {d}: {d:.1} MiB   chunk: {d} MiB\n", .{
        mib(slot_bytes), b.opt.pos, mib(blob_bytes), b.opt.chunk_mib,
    });
    if (b.ctxp.deviceLocalBudget()) |budget| {
        const free = budget.budget -| budget.usage;
        try w.print("vram:     budget {d:.1} GiB, used {d:.1} GiB, " ++
            "free {d:.1} GiB -> ~{d} slots fit\n", .{
            gib(budget.budget), gib(budget.usage), gib(free), free / @as(u64, slot_bytes),
        });
    } else {
        try w.print("vram:     VK_EXT_memory_budget absent (no live budget)\n", .{});
    }
}

/// Create one session per stream, and fill each to depth `--pos`.
fn fillSlots(comptime Runner: type, b: *const Bench, pool: *Sessions(Runner)) ![]const u32 {
    const arena = b.ctx.arena;

    const fill = try arena.alloc(u32, b.opt.pos);
    @memset(fill, 1);

    const ids = try arena.alloc(u32, b.opt.streams);
    for (ids) |*id| {
        id.* = try pool.create(0, .{});
    }

    for (ids) |id| {
        _ = try pool.feed(id, fill, true);
    }

    return ids;
}

/// Warm the path, then time the evict (D2H) and the restore (H2D) of slot 0. The
/// runner splits each swap into its DMA and its mapped-staging memcpy parts, which
/// says which side of a swap costs. No whole-call figure can say that.
fn timeSwaps(comptime Runner: type, b: *const Bench, runner: *Runner, blob: []u8) !SwapTimes {
    const io = b.ctx.io;
    const arena = b.ctx.arena;
    const pos = b.opt.pos;
    const reps = b.opt.reps;
    const has_split = comptime hasSwapSplit(Runner);

    // Warm the path.
    for (0..swap_warm_reps) |_| {
        try runner.evictSlot(0, pos, blob);
        try runner.restoreSlot(0, pos, blob);
    }

    // Time each swap, with its parts where the runner splits them.
    const part_reps: u32 = if (has_split) reps else 0;
    const times: SwapTimes = .{
        .evict = try arena.alloc(f64, reps),
        .restore = try arena.alloc(f64, reps),
        .evict_dma = try arena.alloc(f64, part_reps),
        .evict_copy = try arena.alloc(f64, part_reps),
        .restore_dma = try arena.alloc(f64, part_reps),
        .restore_copy = try arena.alloc(f64, part_reps),
    };
    for (0..reps) |i| {
        const e0 = Io.Clock.Timestamp.now(io, .awake);
        try runner.evictSlot(0, pos, blob);
        times.evict[i] = ms(e0, Io.Clock.Timestamp.now(io, .awake));
        if (comptime has_split) {
            times.evict_dma[i] = runner.last_swap_split.dma_ms;
            times.evict_copy[i] = runner.last_swap_split.memcpy_ms;
        }

        const r0 = Io.Clock.Timestamp.now(io, .awake);
        try runner.restoreSlot(0, pos, blob);
        times.restore[i] = ms(r0, Io.Clock.Timestamp.now(io, .awake));
        if (comptime has_split) {
            times.restore_dma[i] = runner.last_swap_split.dma_ms;
            times.restore_copy[i] = runner.last_swap_split.memcpy_ms;
        }
    }

    return times;
}

/// The host memcpy of one slot, in isolation. It is the fixed CPU cost inside both
/// evict and restore, which a pinned and persistently mapped blob removes.
fn timeHostCopy(b: *const Bench, blob: []const u8, copy: []u8) ![]f64 {
    const times = try b.ctx.arena.alloc(f64, b.opt.reps);
    for (times) |*t| {
        const c0 = Io.Clock.Timestamp.now(b.ctx.io, .awake);
        @memcpy(copy, blob);
        t.* = ms(c0, Io.Clock.Timestamp.now(b.ctx.io, .awake));
    }

    if (copy[copy.len - 1] != blob[blob.len - 1]) return error.BenchCopyMismatch;

    return times;
}

/// A chunked evict, and an evict after a chunked restore, must both equal an evict
/// in one chunk of the whole blob. That evict runs last, because a stage only grows.
fn checkChunks(b: *const Bench, runner: anytype, blob: []u8, whole: []u8) !void {
    const pos = b.opt.pos;

    try runner.evictSlot(0, pos, blob);
    try runner.restoreSlot(0, pos, blob);

    const round = try b.ctx.arena.alloc(u8, blob.len);
    try runner.evictSlot(0, pos, round);

    runner.commands.swap_chunk_bytes = std.math.maxInt(u64);
    try runner.evictSlot(0, pos, whole);

    if (!std.mem.eql(u8, blob, whole)) return error.BenchChunkMismatch;
    if (!std.mem.eql(u8, round, whole)) return error.BenchChunkMismatch;
}

fn printReport(
    comptime Runner: type,
    b: *const Bench,
    decode: DecodeTimes,
    swaps: SwapTimes,
    copies: []f64,
    blob_bytes: usize,
) !void {
    const w = b.w;
    const opt = b.opt;

    // Reduce the samples.
    const e_med = median(swaps.evict);
    const e_p99 = pct(swaps.evict, 99);
    const r_med = median(swaps.restore);
    const r_p99 = pct(swaps.restore, 99);
    const s = e_med + r_med;
    const t = decode.batch;

    // Print the timings.
    try w.print("\ndecode t: batch={d}: {d:.2} ms    batch=1: {d:.2} ms\n", .{
        opt.streams, t, decode.one,
    });
    try w.print("evict :   {d:.2} ms (p99 {d:.2})\n", .{ e_med, e_p99 });
    try w.print("restore:  {d:.2} ms (p99 {d:.2})\n", .{ r_med, r_p99 });
    try w.print("host cpy: {d:.2} ms (the memcpy inside each)\n", .{median(copies)});
    try w.print("chunked:  {d} chunks of {d} MiB, " ++
        "evict and restore equal one whole-blob copy byte for byte\n", .{
        kv_budget.ChunkWalk.count(blob_bytes, @as(u64, opt.chunk_mib) << 20), opt.chunk_mib,
    });

    // The rate divides by the whole blob. A hybrid must divide by its device half instead.
    if (comptime hasSwapSplit(Runner)) {
        try printParts("e", swaps.evict_dma, swaps.evict_copy, blob_bytes, w);
        try printParts("r", swaps.restore_dma, swaps.restore_copy, blob_bytes, w);
    }

    // Print the implied cost.
    try w.print("s = evict+restore = {d:.2} ms\n", .{s});
    try w.print("\nimplied:  swapped tick t+s = {d:.2} ms   ({d:.1}x a clean tick)\n", .{
        t + s, (t + s) / t,
    });
    try w.print("          throughput hit at f=1 swap/tick: {d:.0}%\n", .{100.0 * s / t});
    try w.flush();
}

fn printParts(
    label: []const u8,
    dma: []f64,
    copy: []f64,
    blob_bytes: usize,
    w: *Io.Writer,
) !void {
    const dma_ms = median(dma);
    const copy_ms = median(copy);
    try w.print("parts ({s}): dma {d:.2} ms ({d:.1} GB/s)   memcpy {d:.2} ms ({d:.1} GB/s)\n", .{
        label, dma_ms, gbs(blob_bytes, dma_ms), copy_ms, gbs(blob_bytes, copy_ms),
    });
}

fn hasSwapSplit(comptime Runner: type) bool {
    return @hasField(Runner, "last_swap_split");
}

fn timeDecode(
    comptime Runner: type,
    io: Io,
    pool: *Sessions(Runner),
    ids: []const u32,
    out: []u32,
    reps: u32,
) !f64 {
    std.debug.assert(reps >= 1);
    std.debug.assert(reps <= reps_max);

    for (0..warm_ticks) |_| {
        try pool.decodeBatch(ids, out);
    }

    var samples: [reps_max]f64 = undefined;
    for (samples[0..reps]) |*sample| {
        const t0 = Io.Clock.Timestamp.now(io, .awake);
        try pool.decodeBatch(ids, out);
        sample.* = ms(t0, Io.Clock.Timestamp.now(io, .awake));
    }

    return median(samples[0..reps]);
}

// -- helpers: process probe -------------------------------------------------

/// Print one probe line: the staging submit count, the usage and budget of
/// heaps 0 and 1, the resident set, and the open fd count. A leak below the
/// driver shows as a slope in one of these across the three probes.
fn printProbe(io: Io, ctxp: *const vulkan_ctx_mod.Context, label: []const u8, w: *Io.Writer) !void {
    try w.print("probe {s}: staging submits {d}", .{ label, kernels.stagingSubmitCount() });
    if (!ctxp.has_memory_budget) {
        try w.print("   heaps: VK_EXT_memory_budget absent", .{});
    } else {
        for ([_]u32{ 0, 1 }) |heap| {
            if (ctxp.heapBudget(heap)) |b| {
                try w.print("   heap {d} used {d:.1} of {d:.1} MiB", .{
                    heap, mib(b.usage), mib(b.budget),
                });
            } else {
                try w.print("   heap {d} absent", .{heap});
            }
        }
    }

    var status_buf: [status_text_max]u8 = undefined;
    const status = Io.Dir.cwd().readFile(io, "/proc/self/status", &status_buf) catch null;
    if (if (status) |text| statusKib(text, "VmRSS") else null) |kib| {
        try w.print("   VmRSS {d} KiB", .{kib});
    } else {
        try w.print("   VmRSS unknown", .{});
    }

    if (countFds(io)) |fds| {
        try w.print("   fds {s}{d}\n", .{ if (fds >= fd_count_max) ">=" else "", fds });
    } else {
        try w.print("   fds unknown\n", .{});
    }

    try w.flush();
}

/// The value of the `key:` line of a `/proc/<pid>/status` text, in KiB. The
/// kernel prints these lines as `VmRSS:\t   1234 kB`. `null` when no line has
/// the key or the value is not a KiB count.
fn statusKib(text: []const u8, key: []const u8) ?u64 {
    std.debug.assert(key.len >= 1);

    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len <= key.len) continue;
        if (!std.mem.startsWith(u8, line, key)) continue;
        if (line[key.len] != ':') continue;

        const rest = std.mem.trim(u8, line[key.len + 1 ..], " \t");
        const unit = std.mem.findScalar(u8, rest, ' ') orelse return null;
        if (!std.mem.eql(u8, std.mem.trimStart(u8, rest[unit..], " "), "kB")) return null;

        return std.fmt.parseInt(u64, rest[0..unit], 10) catch null;
    }

    return null;
}

/// Open file descriptors of this process, at most `fd_count_max`. The walk
/// holds one descriptor of its own, which the count leaves out.
fn countFds(io: Io) ?u32 {
    var dir = Io.Dir.cwd().openDir(io, "/proc/self/fd", .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var it = dir.iterate();
    var count: u32 = 0;
    while (count < fd_count_max + 1) {
        const entry = (it.next(io) catch return null) orelse break;
        _ = entry;
        count += 1;
    }

    std.debug.assert(count >= 1);
    return @min(count - 1, fd_count_max);
}

// -- helpers: timing + stats ------------------------------------------------

fn ms(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) f64 {
    const ns: i64 = @intCast(t0.durationTo(t1).raw.toNanoseconds());
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return pctSorted(xs, 50);
}

fn pct(xs: []f64, p: usize) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return pctSorted(xs, p);
}

fn pctSorted(sorted: []const f64, p: usize) f64 {
    if (sorted.len == 0) return 0;
    return sorted[@min(sorted.len - 1, (sorted.len * p) / 100)];
}

/// Bandwidth one part implied, in GB/s over the blob it moved. A part the runner
/// never ran reads as zero instead of dividing by it.
fn gbs(bytes: usize, part_ms: f64) f64 {
    if (part_ms <= 0) return 0;

    return @as(f64, @floatFromInt(bytes)) / (part_ms * 1e6);
}

fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

// -- tests ------------------------------------------------------------------

test "statusKib: reads the KiB count of the named line only" {
    const text = "Name:\tzigferencer\nVmPeak:\t  912345 kB\nVmRSSX:\t       7 kB\n" ++
        "VmRSS:\t  654321 kB\nThreads:\t17\n";
    try std.testing.expectEqual(@as(?u64, 654321), statusKib(text, "VmRSS"));
    try std.testing.expectEqual(@as(?u64, 912345), statusKib(text, "VmPeak"));
    // A line without the kB unit, a missing key, and a key prefix all read as null.
    try std.testing.expectEqual(@as(?u64, null), statusKib(text, "Threads"));
    try std.testing.expectEqual(@as(?u64, null), statusKib(text, "VmSwap"));
    try std.testing.expectEqual(@as(?u64, null), statusKib(text, "VmR"));
    try std.testing.expectEqual(@as(?u64, null), statusKib("VmRSS:\t12x kB\n", "VmRSS"));
    try std.testing.expectEqual(@as(?u64, null), statusKib("", "VmRSS"));
}

test "statusKib: reads the live status of this process" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;

    var buf: [status_text_max]u8 = undefined;
    const text = try Io.Dir.cwd().readFile(std.testing.io, "/proc/self/status", &buf);
    try std.testing.expect(text.len < buf.len);
    const kib = statusKib(text, "VmRSS") orelse return error.TestUnexpectedResult;
    try std.testing.expect(kib >= 1);
}

test "countFds: counts open descriptors and leaves out its own" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;

    const before = countFds(std.testing.io) orelse return error.TestUnexpectedResult;
    var dir = try Io.Dir.cwd().openDir(std.testing.io, "/proc", .{});
    const during = countFds(std.testing.io) orelse return error.TestUnexpectedResult;
    dir.close(std.testing.io);
    const after = countFds(std.testing.io) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(before + 1, during);
    try std.testing.expectEqual(before, after);
}
