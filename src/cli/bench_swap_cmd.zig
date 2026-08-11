//! `zigferencer bench-swap <pub_name>` is the tiered-KV swap-cost micro-bench,
//! the go or no-go for an async and double-buffered swap. It times one KV slot's
//! evict (D2H) and restore (H2D) against a decode tick. The numbers weigh the
//! synchronous swap stall `t + s` against the per-token latency budget, before
//! anyone commits to the FSM and the fences of an async swap.
//!
//! Two windows answer whether `s` is transfer or fixed overhead. Run with
//! `--budget` off, which takes the serve default of the model, and on, which
//! takes the larger VK_EXT_memory_budget window. The change in `s` across the
//! slot-size change is the transfer slope. The residue is the per-swap submit,
//! waitIdle and memcpy floor. The bench needs a fused-GPU build, because the
//! swap is a VRAM to RAM transfer that no CPU engine has.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const cli = @import("command.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const select = @import("select.zig");
const serve_common = @import("serve_common.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const Sessions = @import("../runtime/session.zig").Sessions;
const log = std.log.scoped(.bench_swap);

const llama_fused = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/llama_fused.zig")
else
    void;
const qwen_fused = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/qwen_fused.zig")
else
    void;
const vulkan_ctx_mod = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/vulkan_ctx.zig")
else
    void;
const kernels = if (build_options.use_vulkan_fused_ops)
    @import("../ops_vulkan/kernels.zig")
else
    void;

// -- types ------------------------------------------------------------------

/// Bound on `--reps`. It sizes the sample array of the decode timing.
const reps_max: u32 = 256;

/// Untimed decode ticks before each timed series.
const warm_ticks: u32 = 3;

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
        .pos = args.Doc{ .text = "KV depth to fill before timing, at least 1. The depth\nplus 2 x (3 + reps) decode steps must fit the window" },
        .reps = args.Doc{ .text = "timed repetitions per measurement, 1..256" },
        .budget = args.Doc{ .text = std.fmt.comptimePrint("size the window from VK_EXT_memory_budget (else the\n{d} MiB serve session budget, capped by the model ceiling)", .{limits.serve_session_kv_bytes >> 20}) },
        .kv_dtype = args.Doc{
            .text = "KV element type. It prices the blob a swap moves",
            .default = "the model's registered one",
        },
        .chunk_mib = args.Doc{ .text = std.fmt.comptimePrint("the chunk one staged copy moves, 1..{d} MiB. The run also\nchecks that the chunked copies equal one whole-blob copy\nbyte for byte", .{chunk_mib_max}) },
    };
};

/// Bound on `--chunk-mib`. The sweep reaches 256 MiB, and the bound leaves one
/// step of headroom above it.
const chunk_mib_max: u32 = 1024;

/// Bytes of `/proc/self/status` the probe reads. The file is about 1.5 KiB.
const status_text_max: usize = 4096;

/// Bound on the fd walk of one probe. A count at the bound prints as a floor.
const fd_count_max: u32 = 1 << 16;

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const opt = try args.parseInto(Opt, ctx) orelse return;
    if (opt.streams < 2 or opt.streams > limits.max_batch) {
        return args.fail(ctx.io, "--streams must be in 2..{d} (multi-stream KV is what swaps)", .{limits.max_batch});
    }
    if (opt.reps == 0 or opt.reps > reps_max) {
        return args.fail(ctx.io, "--reps must be in 1..{d}", .{reps_max});
    }
    if (opt.chunk_mib == 0 or opt.chunk_mib > chunk_mib_max) {
        return args.fail(ctx.io, "--chunk-mib must be in 1..{d}", .{chunk_mib_max});
    }
    // An eviction moves the committed rows, so a zero depth moves nothing.
    if (opt.pos == 0) return args.fail(ctx.io, "--pos must be >= 1 (the swap moves the committed KV rows)", .{});
    if (comptime build_options.use_vulkan_fused_ops) return runFused(ctx, opt.pub_name, opt);
    log.err("bench-swap measures a VRAM<->RAM KV swap. Build -Druntime=vulkan -Dops-backend=vulkan_fused", .{});
    return error.UnsupportedServeBackend;
}

// -- helpers ----------------------------------------------------------------

fn runFused(ctx: cli.Context, pub_name: ?[]const u8, opt_in: Opt) !void {
    const model = try select.model(ctx, pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();
    if (loaded.card.arch != .llama and loaded.card.arch != .qwen35) {
        log.err("bench-swap drives llama-family and qwen3.5 models (model is {t})", .{loaded.card.arch});
        return error.UnsupportedServeArch;
    }
    // The KV dtype resolves here, ahead of every price, plan, and device call, so
    // a refusal costs a GGUF map and nothing more.
    var opt = opt_in;
    opt.resolved_kv_dtype = serve_common.kvDtypeFor(opt.kv_dtype, model);
    // One ceiling per boot, from the file and the registry entry, the way serve
    // resolves it. The runner caps derive both default windows from it.
    const ceiling = limits.contextCeiling(loaded.card.context_length, model.context_limit);

    var stdout_buf: [4096]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stdout(), ctx.io, &stdout_buf);
    const w = &fw.interface;

    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    defer ctxp.deinit();

    if (loaded.card.arch == .qwen35) {
        return benchOne(qwen_fused.QwenFusedRunner, ctx, ctxp, &loaded, ceiling, model.pub_name, opt, w);
    }
    return benchOne(llama_fused.LlamaFusedRunner, ctx, ctxp, &loaded, ceiling, model.pub_name, opt, w);
}

fn planFor(comptime RunnerT: type, ctx: cli.Context, ctxp: *vulkan_ctx_mod.Context, loaded: *const loader.Loaded, ceiling: u32, opt: Opt) !kv_budget.KvPlan {
    // The window is the point of the bench. --budget derives X from the live free
    // VRAM, the way serve's --vram-slots does, and the pinned slot count's share
    // of it buys the window. Without the flag the planner takes the serve default
    // of the model, as serve does. Neither request names a window.
    const free: u64 = if (opt.budget) blk: {
        const b = ctxp.deviceLocalBudget() orelse break :blk 0;
        break :blk b.budget -| b.usage;
    } else 0;
    const plan = if (RunnerT == qwen_fused.QwenFusedRunner) blk: {
        const caps = qwen_fused.QwenFusedRunner.kvCaps(loaded, ceiling, free, opt.streams, opt.resolved_kv_dtype);
        break :blk try kv_budget.kvPlan(caps, qwen_fused.QwenFusedRunner.kvCostModel(loaded.card, opt.resolved_kv_dtype), .{
            .x_bytes = if (opt.budget) kv_budget.derivedX(caps) else null,
            .slots = opt.streams,
            .batch = opt.streams,
        });
    } else blk: {
        const caps = try llama_fused.LlamaFusedRunner.kvCaps(ctx.gpa, loaded, ceiling, free, opt.streams, opt.resolved_kv_dtype);
        break :blk try kv_budget.kvPlan(caps, llama_fused.LlamaFusedRunner.kvCostModel(loaded.card, opt.resolved_kv_dtype), .{
            .x_bytes = if (opt.budget) kv_budget.derivedX(caps) else null,
            .slots = opt.streams,
            .batch = opt.streams,
        });
    };
    serve_common.logPlan(plan, opt.resolved_kv_dtype);
    return plan;
}

fn initRunner(comptime RunnerT: type, ctx: cli.Context, ctxp: *vulkan_ctx_mod.Context, loaded: *const loader.Loaded, plan: kv_budget.KvPlan, kv_dtype: dtype.KvDtype) !*RunnerT {
    if (RunnerT == qwen_fused.QwenFusedRunner) {
        return qwen_fused.QwenFusedRunner.init(ctx.gpa, ctxp, loaded, plan, kv_dtype);
    }
    return llama_fused.LlamaFusedRunner.initFromLoaded(ctx.gpa, ctxp, loaded, plan, kv_dtype);
}

/// Positions the decode timing writes into slot 0 past `--pos`. Slot 0 decodes
/// in both series, at batch=streams and at batch=1, and each tick advances it.
fn decodeTicks(reps: u32) u32 {
    std.debug.assert(reps >= 1);
    std.debug.assert(reps <= reps_max);
    return 2 * (warm_ticks + reps);
}

fn benchOne(
    comptime RunnerT: type,
    ctx: cli.Context,
    ctxp: *vulkan_ctx_mod.Context,
    loaded: *const loader.Loaded,
    ceiling: u32,
    model_name: []const u8,
    opt: Opt,
    w: *Io.Writer,
) !void {
    // The plan fixes the window before the runner allocates, so the refusal of
    // a depth the window cannot hold costs no VRAM.
    const plan = try planFor(RunnerT, ctx, ctxp, loaded, ceiling, opt);
    const window = plan.window;
    const ticks = decodeTicks(opt.reps);
    if (@as(u64, opt.pos) + ticks > window) {
        return args.fail(ctx.io, "--pos {d} plus {d} decode steps passes the {d}-token window. Lower --pos or --reps", .{ opt.pos, ticks, window });
    }
    try w.print("model:    {s}   streams: {d}   pos: {d}   reps: {d}   window: {d} tok ({s})\n", .{
        model_name, opt.streams, opt.pos, opt.reps, window, if (opt.budget) "budget" else "default",
    });

    const runner = try initRunner(RunnerT, ctx, ctxp, loaded, plan, opt.resolved_kv_dtype);
    defer runner.deinit();
    // The sweep sets the chunk before the first swap, because a stage only grows.
    runner.swap_chunk_bytes = @as(u64, opt.chunk_mib) << 20;
    std.debug.assert(runner.max_seq == window);
    const pos = opt.pos;
    std.debug.assert(pos >= 1);
    std.debug.assert(pos + ticks <= window);
    // Two figures, because they answer different questions: the full window sizes
    // the VRAM slot, and the committed rows size the blob a swap moves.
    const slot_bytes: usize = @intCast(runner.evictSlotBytesFor(window));
    const blob_bytes: usize = @intCast(runner.evictSlotBytesFor(pos));

    // The live budget says how many slots the VRAM affords.
    try w.print("slot KV:  {d:.1} MiB   blob at pos {d}: {d:.1} MiB   chunk: {d} MiB\n", .{ mib(slot_bytes), pos, mib(blob_bytes), opt.chunk_mib });
    if (ctxp.deviceLocalBudget()) |b| {
        const free = b.budget -| b.usage;
        try w.print("vram:     budget {d:.1} GiB, used {d:.1} GiB, free {d:.1} GiB -> ~{d} slots fit\n", .{
            gib(b.budget), gib(b.usage), gib(free), free / @as(u64, slot_bytes),
        });
    } else {
        try w.print("vram:     VK_EXT_memory_budget absent (no live budget)\n", .{});
    }
    try printProbe(ctx.io, ctxp, "start", w);

    // Every slot fills to depth `pos` with a dummy token, and the bench then
    // measures a decode tick at that depth, at batch=streams and at batch=1.
    var pool = try Sessions(RunnerT).init(ctx.arena, runner);
    const fill = try ctx.arena.alloc(u32, pos);
    @memset(fill, 1);
    var ids = try ctx.arena.alloc(u32, opt.streams);
    for (ids) |*id| id.* = try pool.create(0, .{});
    if (pos > 0) for (ids) |id| {
        _ = try pool.feed(id, fill, true);
    };
    try printProbe(ctx.io, ctxp, "fill", w);

    var out = try ctx.arena.alloc(u32, opt.streams);
    const t_batch = try timeDecode(RunnerT, ctx.io, &pool, ids, out, opt.reps);
    const t_one = try timeDecode(RunnerT, ctx.io, &pool, ids[0..1], out[0..1], opt.reps);

    // s: warm the path, then time the evict (D2H) and the restore (H2D) of slot 0.
    const blob = try ctx.arena.alloc(u8, blob_bytes);
    const blob2 = try ctx.arena.alloc(u8, blob_bytes);
    for (0..5) |_| {
        try runner.evictSlot(0, pos, blob);
        try runner.restoreSlot(0, pos, blob);
    }
    var evict = try ctx.arena.alloc(f64, opt.reps);
    var restore = try ctx.arena.alloc(f64, opt.reps);
    // The runner splits each swap into its DMA leg and its mapped-staging memcpy
    // leg. Reading the legs apart says which side of a swap costs, which neither
    // the whole-call figure nor the host-to-host probe below can say.
    const has_split = @hasField(RunnerT, "last_swap_split");
    const leg_reps: u32 = if (has_split) opt.reps else 0;
    var e_dma = try ctx.arena.alloc(f64, leg_reps);
    var e_cpy = try ctx.arena.alloc(f64, leg_reps);
    var r_dma = try ctx.arena.alloc(f64, leg_reps);
    var r_cpy = try ctx.arena.alloc(f64, leg_reps);
    for (0..opt.reps) |i| {
        const e0 = Io.Clock.Timestamp.now(ctx.io, .awake);
        try runner.evictSlot(0, pos, blob);
        evict[i] = ms(e0, Io.Clock.Timestamp.now(ctx.io, .awake));
        if (comptime has_split) {
            e_dma[i] = runner.last_swap_split.dma_ms;
            e_cpy[i] = runner.last_swap_split.memcpy_ms;
        }
        const r0 = Io.Clock.Timestamp.now(ctx.io, .awake);
        try runner.restoreSlot(0, pos, blob);
        restore[i] = ms(r0, Io.Clock.Timestamp.now(ctx.io, .awake));
        if (comptime has_split) {
            r_dma[i] = runner.last_swap_split.dma_ms;
            r_cpy[i] = runner.last_swap_split.memcpy_ms;
        }
    }
    // The host memcpy of one slot, in isolation. It is the fixed CPU cost inside
    // both evict and restore, which a pinned and persistently mapped blob removes.
    var cpy = try ctx.arena.alloc(f64, opt.reps);
    for (0..opt.reps) |i| {
        const c0 = Io.Clock.Timestamp.now(ctx.io, .awake);
        @memcpy(blob2, blob);
        cpy[i] = ms(c0, Io.Clock.Timestamp.now(ctx.io, .awake));
    }
    if (blob2[blob_bytes - 1] != blob[blob_bytes - 1]) return error.BenchCopyMismatch;

    // The chunked walk must move the same bytes as one whole-blob copy. A
    // chunked evict, and an evict after a chunked restore, both equal an evict
    // in one chunk of the whole blob. The whole-blob evict runs last, because
    // a stage only grows.
    try runner.evictSlot(0, pos, blob);
    try runner.restoreSlot(0, pos, blob);
    const round = try ctx.arena.alloc(u8, blob_bytes);
    try runner.evictSlot(0, pos, round);
    runner.swap_chunk_bytes = std.math.maxInt(u64);
    try runner.evictSlot(0, pos, blob2);
    if (!std.mem.eql(u8, blob, blob2)) return error.BenchChunkMismatch;
    if (!std.mem.eql(u8, round, blob2)) return error.BenchChunkMismatch;
    try printProbe(ctx.io, ctxp, "end", w);

    const e_med = median(evict);
    const e_p99 = pct(evict, 99);
    const r_med = median(restore);
    const r_p99 = pct(restore, 99);
    const s = e_med + r_med;
    try w.print("\ndecode t: batch={d}: {d:.2} ms    batch=1: {d:.2} ms\n", .{ opt.streams, t_batch, t_one });
    try w.print("evict :   {d:.2} ms (p99 {d:.2})\n", .{ e_med, e_p99 });
    try w.print("restore:  {d:.2} ms (p99 {d:.2})\n", .{ r_med, r_p99 });
    try w.print("host cpy: {d:.2} ms (the memcpy inside each)\n", .{median(cpy)});
    try w.print("chunked:  {d} chunks of {d} MiB, evict and restore equal one whole-blob copy byte for byte\n", .{
        kv_budget.ChunkWalk.count(blob_bytes, @as(u64, opt.chunk_mib) << 20), opt.chunk_mib,
    });
    // The rate divides by the whole blob, which is what the legs move on the two
    // fused runners this bench drives. A hybrid arm must divide by the device
    // half instead, because its legs stage only the suffix layers.
    if (comptime has_split) {
        const ed = median(e_dma);
        const ec = median(e_cpy);
        const rd = median(r_dma);
        const rc = median(r_cpy);
        try w.print("legs (e): dma {d:.2} ms ({d:.1} GB/s)   memcpy {d:.2} ms ({d:.1} GB/s)\n", .{ ed, gbs(blob_bytes, ed), ec, gbs(blob_bytes, ec) });
        try w.print("legs (r): dma {d:.2} ms ({d:.1} GB/s)   memcpy {d:.2} ms ({d:.1} GB/s)\n", .{ rd, gbs(blob_bytes, rd), rc, gbs(blob_bytes, rc) });
    }
    try w.print("s = evict+restore = {d:.2} ms\n", .{s});
    try w.print("\nimplied:  swapped tick t+s = {d:.2} ms   ({d:.1}x a clean tick)\n", .{ t_batch + s, (t_batch + s) / t_batch });
    try w.print("          throughput hit at f=1 swap/tick: {d:.0}%\n", .{100.0 * s / t_batch});
    try w.flush();
}

fn timeDecode(comptime RunnerT: type, io: Io, pool: *Sessions(RunnerT), ids: []const u32, out: []u32, reps: u32) !f64 {
    std.debug.assert(reps >= 1);
    std.debug.assert(reps <= reps_max);
    for (0..warm_ticks) |_| try pool.decodeBatch(ids, out);
    var samples: [reps_max]f64 = undefined;
    for (samples[0..reps]) |*sample| {
        const t0 = Io.Clock.Timestamp.now(io, .awake);
        try pool.decodeBatch(ids, out);
        sample.* = ms(t0, Io.Clock.Timestamp.now(io, .awake));
    }
    return median(samples[0..reps]);
}

// -- process probe ----------------------------------------------------------

/// Print one probe line: the staging submit count, the usage and budget of
/// heaps 0 and 1, the resident set, and the open fd count. A leak below the
/// driver shows as a slope in one of these across the three probes.
fn printProbe(io: Io, ctxp: *const vulkan_ctx_mod.Context, label: []const u8, w: *Io.Writer) !void {
    try w.print("probe {s}: staging submits {d}", .{ label, kernels.stagingSubmitCount() });
    if (!ctxp.has_memory_budget) {
        try w.print("   heaps: VK_EXT_memory_budget absent", .{});
    } else for ([_]u32{ 0, 1 }) |heap| {
        if (ctxp.heapBudget(heap)) |b| {
            try w.print("   heap {d} used {d:.1} of {d:.1} MiB", .{ heap, mib(b.usage), mib(b.budget) });
        } else {
            try w.print("   heap {d} absent", .{heap});
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
        if (line.len <= key.len or !std.mem.startsWith(u8, line, key) or line[key.len] != ':') continue;
        const rest = std.mem.trim(u8, line[key.len + 1 ..], " \t");
        const unit = std.mem.indexOfScalar(u8, rest, ' ') orelse return null;
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

// -- timing + stats ---------------------------------------------------------

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

/// Bandwidth one leg implied, in GB/s over the blob it moved. A leg the runner
/// never ran reads as zero instead of dividing by it.
fn gbs(bytes: usize, leg_ms: f64) f64 {
    if (leg_ms <= 0) return 0;
    return @as(f64, @floatFromInt(bytes)) / (leg_ms * 1e6);
}

fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}
fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

// -- tests ------------------------------------------------------------------

test "statusKib: reads the KiB count of the named line only" {
    const text = "Name:\tzigferencer\n" ++
        "VmPeak:\t  912345 kB\n" ++
        "VmRSSX:\t       7 kB\n" ++
        "VmRSS:\t  654321 kB\n" ++
        "Threads:\t17\n";
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
