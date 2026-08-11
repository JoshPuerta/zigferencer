//! Shared values and leaf helpers for the bench-model command, which splits
//! across `bench_model_{cmd,exec,report,qwen}.zig`. The result structs are the
//! values those modules pass to each other. The helpers are the one-liners that
//! more than one of them needs. This module imports nothing from its siblings,
//! so the siblings cannot form an import cycle.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const kv_budget = @import("../core/kv_budget.zig");
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const step_stats = @import("../core/step_stats.zig");
const args = @import("args.zig");
const flags_mod = @import("flags.zig");

/// Re-exported so the bench-family modules keep one `common.writeStderr` name
/// over the single implementation in `cli/args.zig`.
pub const writeStderr = args.writeStderr;

// -- types ------------------------------------------------------------------

/// The bench-model flag set. `bench_model_cmd` parses it once and hands the
/// executors one value, which separates what the caller asked for from how the
/// run happens. The field defaults are the flag defaults, and the help states
/// them.
pub const Options = struct {
    pub_name: ?[]const u8 = null,
    prompt: []const u8 = "Hello",
    max_tokens: u32 = 32,
    runs: u32 = 3,
    chat: bool = false,
    json: bool = false,
    /// Synthesize exactly N prompt tokens instead of tokenizing `prompt`.
    /// Timing is data-independent, so this gives exact, tokenizer-agnostic
    /// prefill lengths for reproducible TTFT-vs-N sweeps.
    prompt_tokens: ?u32 = null,
    /// Force exactly max_tokens steps per run. Throughput depends only on token
    /// counts, so this gives a clean, model-agnostic ITL even when greedy
    /// decoding degenerates to early EOS.
    ignore_eos: bool = false,
    /// The bench takes no thinking knob, so the engine thinking stays `auto`.
    engine: flags_mod.EngineFlags = .{},
    /// With a --gpu-layers count: construct the hybrid even at a degenerate
    /// split (0 or all layers), which production selection never does. The
    /// cross-runner token-equality gate runs through this.
    force_hybrid: bool = false,
    /// CPU backends only, llama and qwen35: prefill the prompt unprofiled, then
    /// decode `max_tokens` steps with the per-op CPU timer
    /// (`cpu_profile.CpuProfile`) installed, print the ns + %-of-step breakdown
    /// and the parallel and serial shares, exit. The timer compiles into a
    /// hot-trace build only, so every other build refuses the flag.
    profile_cpu_decode: bool = false,
    /// qwen35 fused only: one timestamped prefill (or decode step), print the
    /// per-stage GPU breakdown, exit.
    profile_prefill: bool = false,
    profile_decode: bool = false,

    pub const docs_from = flags_mod.EngineDocs;
    pub const positional = "pub_name";
    pub const omit: []const []const u8 = &.{ "thinking", "thinking_budget" };
    pub const help: args.Help = .{
        .usage = "zigferencer-dev bench-model <pub_name> [flags]",
        .about = "Run the full forward pass repeatedly. Report TTFT, ITL, and a token-stream hash.\n",
        .footer =
        \\examples:
        \\  zigferencer-dev bench-model SmolLM2-135M-Instruct-Q8_0 --runs 5
        \\  zigferencer-dev bench-model --prompt-tokens 512 --max-tokens 64 --json
        \\
        ,
    };
    pub const docs = .{
        .prompt = args.Doc{ .group = "common", .text = "text to feed" },
        .max_tokens = args.Doc{ .group = "common", .text = "the most tokens one run generates, at least 2" },
        .runs = args.Doc{ .group = "common", .text = std.fmt.comptimePrint("repeat for stddev, at most {d}", .{limits.bench_runs_max}) },
        .chat = args.Doc{ .group = "common", .text = "apply the chat template of the model" },
        .json = args.Doc{
            .group = "common",
            .text = "machine-readable output (tools/bench_model_all.sh).\nEach run adds itl_by_depth, the mean ITL per 512 KV positions",
        },
        .prompt_tokens = args.Doc{
            .group = "prompt shape",
            .text = "synthesize N prompt tokens (TTFT-vs-N sweeps, overrides --prompt)",
        },
        .ignore_eos = args.Doc{ .group = "prompt shape", .text = "spend the full decode budget per run (clean ITL)" },
        .force_hybrid = args.Doc{
            .group = "gpu (fused-GPU build only)",
            .text = "construct the hybrid even at a degenerate split (gate\nruns, llama only). Needs a --gpu-layers count",
        },
        .profile_cpu_decode = args.Doc{
            .group = "profiling (cpu, hot-trace build only: OPT=Debug or TRACE=1)",
            .text = "per-op CPU decode timer. Print the parallel and serial shares, exit",
        },
        .profile_prefill = args.Doc{
            .group = "profiling (qwen35 fused-GPU)",
            .text = "timestamp the last prefill chunk, per-stage GPU breakdown, exit",
        },
        .profile_decode = args.Doc{
            .group = "profiling (qwen35 fused-GPU)",
            .text = "prefill, then one timestamped decode step, exit",
        },
        .gpu_layers = args.Doc{
            .group = "gpu (fused-GPU build only)",
            .text = "pin N GPU layers, bench the hybrid split runner.\nauto puts the most layers on the GPU that the free VRAM funds",
        },
    };
};

/// The KV shape a run executed under, carried to the report. Zero slots means
/// the path planned no device KV (CPU and pinned runners), and the report then
/// prints nothing.
pub const PlanInfo = struct {
    n_slots: u32 = 0,
    window: u32 = 0,
    kv_mib: u64 = 0,
    /// The dtype the plan was priced at, so the report names the run's KV
    /// element type the way the serve plan line names it.
    kv_dtype: dtype.KvDtype = .f32,

    pub fn fromPlan(plan: kv_budget.KvPlan, kv_dtype: dtype.KvDtype) PlanInfo {
        return .{
            .n_slots = plan.n_slots,
            .window = plan.window,
            .kv_mib = plan.vram.kv / (1024 * 1024),
            .kv_dtype = kv_dtype,
        };
    }
};

/// Positions per bucket of the ITL depth curve.
pub const itl_bucket_positions: u32 = 512;

/// One bucket of the ITL depth curve: the decode steps that wrote a position
/// in `pos .. pos + itl_bucket_positions`, and their summed wall time.
pub const DepthBucket = struct {
    pos: u32,
    steps: u32,
    ns: u64,

    pub fn meanMs(b: DepthBucket) f64 {
        std.debug.assert(b.steps > 0);
        return @as(f64, @floatFromInt(b.ns)) / @as(f64, @floatFromInt(b.steps)) / 1e6;
    }
};

/// The per-step decode times of one turn, summed per bucket of
/// `itl_bucket_positions`. Integer sums are exact, and the report divides once
/// per bucket, so no rounding error builds up across the steps. The buckets
/// span the engine bound, so a window at any model ceiling fits.
pub const DepthCurve = struct {
    pub const bucket_count = limits.max_model_ctx / itl_bucket_positions;
    comptime {
        std.debug.assert(limits.max_model_ctx % itl_bucket_positions == 0);
    }

    steps: [bucket_count]u32 = @splat(0),
    ns: [bucket_count]u64 = @splat(0),

    pub fn add(self: *DepthCurve, pos: u32, ns: u64) void {
        std.debug.assert(pos < limits.max_model_ctx);
        const i = pos / itl_bucket_positions;
        self.steps[i] += 1;
        self.ns[i] += ns;
    }

    /// The non-empty buckets in position order, in memory from `arena`.
    pub fn collect(self: *const DepthCurve, arena: Allocator) Allocator.Error![]const DepthBucket {
        var n: usize = 0;
        for (self.steps) |s| n += @intFromBool(s > 0);
        const out = try arena.alloc(DepthBucket, n);
        var k: usize = 0;
        for (self.steps, self.ns, 0..) |s, ns, i| {
            if (s == 0) continue;
            out[k] = .{ .pos = @as(u32, @intCast(i)) * itl_bucket_positions, .steps = s, .ns = ns };
            k += 1;
        }
        std.debug.assert(k == n);
        return out;
    }
};

pub const RunResult = struct {
    ttft_ms: f64,
    itl_ms: f64,
    tokens_hash: u64,
    tokens_generated: u32,
    itl_by_depth: []const DepthBucket,
    plan: PlanInfo = .{},
    host: step_stats.DecodeHost = .{},
};

// -- api --------------------------------------------------------------------

pub fn toMs(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) f64 {
    const ns_i96 = t0.durationTo(t1).raw.toNanoseconds();
    const ns: f64 = @floatFromInt(@as(i64, @intCast(ns_i96)));
    return ns / 1e6;
}

/// Refuse a bench whose prompt plus decode budget passes the sequence window.
/// Every bench arm frames one window, so the check lives here once. The message
/// names the count and the limit, so the CLI exits with no trace.
pub fn requireSequenceFits(io: Io, prompt_len: usize, decode_len: u32, max_seq: u32) error{PromptTooLong}!void {
    if (prompt_len + decode_len <= max_seq) return;
    var buf: [256]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    fw.interface.print("error: prompt {d} + decode {d} tokens passes the {d}-token window. Lower --prompt-tokens or --max-tokens\n", .{
        prompt_len, decode_len, max_seq,
    }) catch {};
    fw.interface.flush() catch {};
    return error.PromptTooLong;
}

/// Synthesize `n` deterministic in-range token ids for a TTFT/ITL sweep. The
/// output text is garbage, and prefill and decode timing is data-independent for
/// these kernels, so this gives exact, tokenizer-agnostic prefill lengths.
pub fn synthPromptIds(arena: Allocator, n: u32, vocab: u32) ![]u32 {
    const ids = try arena.alloc(u32, n);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 131 + 7) % vocab);
    return ids;
}

// -- tests ------------------------------------------------------------------

test "DepthCurve: bucket steps and nanoseconds sum to the stream totals exactly" {
    // Positions span the whole engine bound. A stream holds at most 8192
    // steps, so the test stays fast.
    const max_len: u32 = 8192;
    var prng: std.Random.DefaultPrng = .init(0x1d_b0c4e7);
    const rand = prng.random();
    var round: u32 = 0;
    while (round < 64) : (round += 1) {
        var curve: DepthCurve = .{};
        var seen: [DepthCurve.bucket_count]bool = @splat(false);
        // A decode writes consecutive positions, so most streams walk. A few
        // jump at random to reach every bucket boundary.
        const walk = rand.boolean();
        const len = rand.uintAtMost(u32, max_len);
        var pos = rand.uintLessThan(u32, limits.max_model_ctx);
        var total_ns: u64 = 0;
        var i: u32 = 0;
        while (i < len) : (i += 1) {
            pos = if (walk) (pos + 1) % limits.max_model_ctx else rand.uintLessThan(u32, limits.max_model_ctx);
            const ns = rand.uintAtMost(u64, 2_000_000_000);
            curve.add(pos, ns);
            seen[pos / itl_bucket_positions] = true;
            total_ns += ns;
        }

        const buckets = try curve.collect(std.testing.allocator);
        defer std.testing.allocator.free(buckets);
        var steps_sum: u64 = 0;
        var ns_sum: u64 = 0;
        var n_seen: usize = 0;
        for (seen) |b| n_seen += @intFromBool(b);
        try std.testing.expectEqual(n_seen, buckets.len);
        for (buckets, 0..) |b, k| {
            try std.testing.expect(b.steps > 0);
            try std.testing.expectEqual(@as(u32, 0), b.pos % itl_bucket_positions);
            try std.testing.expect(seen[b.pos / itl_bucket_positions]);
            if (k > 0) try std.testing.expect(buckets[k - 1].pos < b.pos);
            steps_sum += b.steps;
            ns_sum += b.ns;
        }
        try std.testing.expectEqual(@as(u64, len), steps_sum);
        try std.testing.expectEqual(total_ns, ns_sum);
    }
}

test "DepthCurve: the first and the last position of the engine bound land in the end buckets" {
    var curve: DepthCurve = .{};
    curve.add(0, 3);
    curve.add(limits.max_model_ctx - 1, 5);
    const buckets = try curve.collect(std.testing.allocator);
    defer std.testing.allocator.free(buckets);
    try std.testing.expectEqual(@as(usize, 2), buckets.len);
    try std.testing.expectEqual(@as(u32, 0), buckets[0].pos);
    try std.testing.expectEqual(@as(u64, 3), buckets[0].ns);
    try std.testing.expectEqual(limits.max_model_ctx - itl_bucket_positions, buckets[1].pos);
    try std.testing.expectEqual(@as(u64, 5), buckets[1].ns);
}
