//! The values the files of `bench-model` pass to each other, and their leaf
//! helpers.
//!
//! Comptime. An assert pins the depth buckets to divide the model context evenly. `loadAndRun`
//! turns the architecture into a family row and calls a handler generic over it.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const kv_budget = zigferencer.kv_budget;
const dtype = zigferencer.dtype;
const limits = zigferencer.limits;
const step_stats = zigferencer.step_stats;
const registry = zigferencer.registry;
const loader = zigferencer.loader;
const resolve_mod = zigferencer.resolve;
const parallel = zigferencer.parallel;
const args = zigferencer.cli.args;
const context = zigferencer.cli.context;
const flags_mod = zigferencer.cli.flag_groups;
const boot_resources = zigferencer.cli.boot_resources;
const boot_lines = zigferencer.cli.boot_lines;
const select = zigferencer.cli.select;
const family = zigferencer.family;
const build_options = @import("build_options");
const core_log = zigferencer.log;

const use_fused = build_options.use_vulkan_fused_ops;
const use_per_op = build_options.use_vulkan_ops;

pub const writeStderr = args.writeStderr;

// -- constants --------------------------------------------------------------

/// Positions per bucket of the ITL depth curve.
pub const itl_bucket_positions: u32 = 512;

// -- types ------------------------------------------------------------------

/// What every path of one bench reads. `bench_model_cmd` builds it once, after
/// the load and the resolve.
pub const Bench = struct {
    ctx: context.Context,
    opt: Options,
    model: *const registry.Model,
    loaded: *const loader.Loaded,
    boot: *const boot_resources.DeviceBoot,
    resolved: *const resolve_mod.Resolved,
    policy: parallel.Policy,

    pub fn kvDtype(self: *const Bench) dtype.KvDtype {
        return self.resolved.early.kv_dtype;
    }
};

/// The `bench-model` flag set. `bench_model_cmd` parses it once. The field defaults are the flag
/// defaults.
pub const Options = struct {
    pub_name: ?[]const u8 = null,
    prompt: []const u8 = "Hello",
    max_tokens: u32 = 32,
    runs: u32 = 3,
    chat: bool = false,
    json: bool = false,
    /// Synthesize exactly N prompt tokens instead of tokenizing `prompt`.
    prompt_tokens: ?u32 = null,
    /// Force exactly max_tokens steps per run, so ITL stays clean when greedy decoding hits EOS.
    ignore_eos: bool = false,
    /// The bench takes no thinking option, so the engine thinking stays `auto`.
    engine: flags_mod.EngineFlags = .{},
    /// CPU backends only: print the per-op decode breakdown (`bench_model_profile`), then exit.
    profile_cpu_decode: bool = false,
    /// qwen35 fused only: print the per-stage GPU breakdown of one step, then exit.
    profile_prefill: bool = false,
    profile_decode: bool = false,

    pub const docs_from = flags_mod.EngineDocs;
    pub const positional = "pub_name";
    pub const omit: []const []const u8 = &.{ "thinking", "thinking_budget" };
    pub const help: args.Help = .{
        .usage = "zigferencer-dev bench-model <pub_name> [flags]",
        .about = "Run the full forward pass repeatedly. " ++
            "Report TTFT, ITL, and a token-stream hash.\n",
        .footer =
        \\examples:
        \\  zigferencer-dev bench-model SmolLM2-135M-Instruct-Q8_0 --runs 5
        \\  zigferencer-dev bench-model --prompt-tokens 512 --max-tokens 64 --json
        \\
        ,
    };
    pub const docs = .{
        .prompt = args.Doc{ .group = "common", .text = "text to feed" },
        .max_tokens = args.Doc{
            .group = "common",
            .text = "the most tokens one run generates, at least 2",
        },
        .runs = args.Doc{
            .group = "common",
            .text = std.fmt.comptimePrint(
                "repeat for stddev, at most {d}",
                .{limits.bench_runs_max},
            ),
        },
        .chat = args.Doc{ .group = "common", .text = "apply the chat template of the model" },
        .json = args.Doc{
            .group = "common",
            .text = "machine-readable output (tools/gate.sh).\n" ++
                "Each run adds itl_by_depth, the mean ITL per 512 KV positions",
        },
        .prompt_tokens = args.Doc{
            .group = "prompt shape",
            .text = "synthesize N prompt tokens (TTFT-vs-N sweeps, overrides --prompt)",
        },
        .ignore_eos = args.Doc{
            .group = "prompt shape",
            .text = "spend the full decode budget per run (clean ITL)",
        },
        .profile_cpu_decode = args.Doc{
            .group = "profiling (cpu, trace build only: -Dtrace=true)",
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
            .text = "pin N GPU layers, bench the hybrid split runner.\n" ++
                "auto puts the most layers on the GPU that the free VRAM funds",
        },
    };
};

/// The KV shape a run executed under. Zero slots means no device KV plan, and the report skips it.
pub const PlanInfo = struct {
    n_slots: u32 = 0,
    window: u32 = 0,
    kv_mib: u64 = 0,
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
/// `itl_bucket_positions`.
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
        for (self.steps) |s| {
            n += @intFromBool(s > 0);
        }

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
    /// Generated tokens over the whole turn, prefill included.
    tok_s: f64 = 0,
    tokens_hash: u64,
    tokens_generated: u32,
    itl_by_depth: []const DepthBucket,
    plan: PlanInfo = .{},
    host: step_stats.DecodeHost = .{},
};

// -- api --------------------------------------------------------------------

/// Load the model of `opt`, resolve its placement as `kind` for `input`, and call
/// `handler(Family, &bench, extra)` with the family row of its architecture.
pub fn loadAndRun(
    ctx: context.Context,
    opt: Options,
    kind: resolve_mod.Kind,
    input: resolve_mod.Input,
    comptime handler: anytype,
    extra: anytype,
) !void {
    // A whole-model fused bench runs no CPU op, so its pool starts no workers.
    const runs_cpu_ops = !build_options.use_vulkan_fused_ops or opt.engine.gpu_layers != null;
    const flags = opt.engine.threads.policyFlags();
    const policy = try boot_resources.resolvePolicy(ctx.io, flags, runs_cpu_ops);

    const model = try select.model(ctx, opt.pub_name);
    const entry = boot_resources.entryOf(model);

    // Every refusal that needs no model file lands here, before the load.
    switch (resolve_mod.early(kind, input, entry)) {
        .ok => {},
        .refused => |r| return boot_lines.refuse(ctx.io, model.pub_name, r),
    }

    // Load the model and resolve its placement.
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    try boot_resources.checkLoaded(&loaded);
    const model_needs = boot_resources.modelNeeds(&loaded);
    const ram_free = boot_resources.hostRamFree(ctx.io);
    const resources: resolve_mod.Resources = .{ .ram_free = ram_free, .model = model_needs };
    const resolved = switch (resolve_mod.resolve(kind, input, entry, loaded.card, resources)) {
        .ok => |r| r,
        .refused => |r| return boot_lines.refuse(ctx.io, model.pub_name, r),
    };

    // Hand the bench to the family.
    const boot: boot_resources.DeviceBoot = .{
        .kind = kind,
        .input = input,
        .entry = entry,
        .card = loaded.card,
        .model = model_needs,
        .ram = .{ .read = ram_free },
    };
    const bench: Bench = .{
        .ctx = ctx,
        .opt = opt,
        .model = model,
        .loaded = &loaded,
        .boot = &boot,
        .resolved = &resolved,
        .policy = policy,
    };

    switch (loaded.card.arch) {
        inline else => |arch| return handler(family.Of(arch), &bench, extra),
    }
}

/// Refuse a bench whose prompt plus decode budget passes the sequence window.
pub fn requireSequenceFits(
    io: Io,
    prompt_len: usize,
    decode_len: u32,
    max_seq: u32,
) error{PromptTooLong}!void {
    if (prompt_len + decode_len <= max_seq) return;

    var buf: [256]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);

    fw.interface.print(
        "error: prompt {d} + decode {d} tokens passes the {d}-token window. " ++
            "Lower --prompt-tokens or --max-tokens\n",
        .{
            prompt_len, decode_len, max_seq,
        },
    ) catch {};

    fw.interface.flush() catch {};
    return error.PromptTooLong;
}

/// Synthesize `n` deterministic in-range token ids for a TTFT/ITL sweep.
pub fn synthPromptIds(arena: Allocator, n: u32, vocab: u32) ![]u32 {
    const ids = try arena.alloc(u32, n);
    for (ids, 0..) |*id, i| {
        id.* = @intCast((i * 131 + 7) % vocab);
    }

    return ids;
}

/// Refuse a flag set that this build cannot run. `bench-model` and `dispatch-list` call it.
pub fn refuseBadFlags(io: std.Io, opt: Options) !void {
    if (opt.runs == 0 or opt.runs > limits.bench_runs_max) {
        return args.fail(io, "--runs must be in 1..{d}", .{limits.bench_runs_max});
    }
    if (opt.max_tokens < 2) {
        return args.fail(io, "--max-tokens must be >= 2 (need at least 1 step for ITL)", .{});
    }

    try refuseGpuFlags(io, opt);
    try refuseCpuProfile(io, opt);
}

// -- helpers ----------------------------------------------------------------

fn refuseGpuFlags(io: std.Io, opt: Options) error{UsageError}!void {
    const gpu_layers = opt.engine.gpu_layers;

    if (!use_fused) {
        const need_fused = "requires -Dops-backend=vulkan_fused";
        if (gpu_layers != null) return args.fail(io, "--gpu-layers " ++ need_fused, .{});

        // The device profilers read device timestamps.
        if (opt.profile_prefill) return args.fail(io, "--profile-prefill " ++ need_fused, .{});
        if (opt.profile_decode) return args.fail(io, "--profile-decode " ++ need_fused, .{});
    }

    if (gpu_layers != null and (opt.profile_prefill or opt.profile_decode)) {
        return args.fail(
            io,
            "--gpu-layers benches the hybrid runner and does not combine with profile flags",
            .{},
        );
    }
}

fn refuseCpuProfile(io: std.Io, opt: Options) !void {
    if (!opt.profile_cpu_decode) return;

    // The per-op CPU timer compiles into a trace build only, so a release
    // binary holds no clock read in the forward pass.
    if (!core_log.hot) {
        return args.fail(
            io,
            "--profile-cpu-decode requires a trace build (-Dtrace=true)",
            .{},
        );
    }

    if (use_fused or use_per_op) {
        writeStderr(
            io,
            "error: --profile-cpu-decode is for CPU backends (rebuild without -Druntime=vulkan)\n",
        );
        return error.UnsupportedArchForBackend;
    }
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
            pos = if (walk) (pos + 1) % limits.max_model_ctx else rand.uintLessThan(
                u32,
                limits.max_model_ctx,
            );
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
        for (seen) |b| {
            n_seen += @intFromBool(b);
        }

        try std.testing.expectEqual(n_seen, buckets.len);
        for (buckets, 0..) |b, k| {
            try std.testing.expect(b.steps > 0);
            try std.testing.expectEqual(@as(u32, 0), b.pos % itl_bucket_positions);
            try std.testing.expect(seen[b.pos / itl_bucket_positions]);
            if (k > 0) {
                try std.testing.expect(buckets[k - 1].pos < b.pos);
            }

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
