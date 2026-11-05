//! `zigferencer-dev bench-model <pub_name>` runs the full forward pass on a prompt
//! and repeats it `--runs` times. It reports TTFT and ITL with mean, stddev, min
//! and max, and an FNV-1a 64 hash of the generated token ids.
//!
//! A stable hash across runs proves the forward pass is deterministic. A stable
//! hash across binaries built with different `-Dops-backend` values is the
//! cross-backend gate, because a backend that changes the tokens changed the math.
//! `tools/bench_model_all.sh` drives that loop.
//!
//! This file checks the flags, loads the model, and picks one path. Each path owns
//! its runner: `bench_model_host.zig` (CPU and per-op Vulkan), `bench_model_fused.zig`
//! (the whole model on the GPU), and `bench_model_placement.zig` (a CPU and GPU split).

const std = @import("std");
const build_options = @import("build_options");

const cli = @import("command.zig");
const args = @import("args.zig");
const select = @import("select.zig");
const serve_common = @import("serve_common.zig");
const common = @import("bench_model_common.zig");
const host = @import("bench_model_host.zig");
const family = @import("../engine/family.zig");
const resolve_mod = @import("../engine/resolve.zig");
const core_log = @import("../core/log.zig");
const limits = @import("../core/limits.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");

const fused_build = build_options.use_vulkan_fused_ops;
const per_op_build = build_options.use_vulkan_ops;

// The device paths resolve to `void` on a CPU build, which never analyzes them.
const fused = if (fused_build) @import("bench_model_fused.zig") else void;
const placement = if (fused_build) @import("bench_model_placement.zig") else void;

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const opt = try args.parseInto(common.Options, ctx) orelse return;
    try refuseBadFlags(ctx.io, opt);

    // A whole-model fused bench runs no CPU op, so its pool starts no workers.
    const runs_cpu_ops = !fused_build or opt.engine.gpu_layers != null;
    const flags = opt.engine.threads.policyFlags();
    const policy = try serve_common.resolvePolicy(ctx.io, flags, runs_cpu_ops);

    const model = try select.model(ctx, opt.pub_name);
    const entry = serve_common.entryOf(model);
    const input: resolve_mod.Input = .{ .engine = opt.engine, .force_hybrid = opt.force_hybrid };

    // Every refusal that needs no model file lands here, before the load.
    switch (resolve_mod.early(.bench, input, entry)) {
        .ok => {},
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    }

    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    const model_facts = try serve_common.modelFacts(ctx.gpa, &loaded);
    const ram_free = serve_common.hostRamFree(ctx.io);
    const facts: resolve_mod.Facts = .{ .ram_free = ram_free, .model = model_facts };
    const resolved = switch (resolve_mod.resolve(.bench, input, entry, loaded.card, facts)) {
        .ok => |r| r,
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    };

    const boot: serve_common.DeviceBoot = .{
        .kind = .bench,
        .input = input,
        .entry = entry,
        .card = loaded.card,
        .model = model_facts,
        .ram = .{ .read = ram_free },
    };
    const bench: common.Bench = .{
        .ctx = ctx,
        .opt = opt,
        .model = model,
        .loaded = &loaded,
        .boot = &boot,
        .resolved = &resolved,
        .policy = policy,
    };

    switch (loaded.card.arch) {
        inline else => |arch| return benchFamily(family.Of(arch), &bench),
    }
}

// -- helpers ----------------------------------------------------------------

/// Pick the path of this build and these flags.
fn benchFamily(comptime Family: type, b: *const common.Bench) !void {
    if (comptime fused_build) {
        if (b.opt.engine.gpu_layers == null) return fused.bench(Family, b);
        return placement.bench(Family, b);
    } else {
        return host.bench(Family, b);
    }
}

fn refuseBadFlags(io: std.Io, opt: common.Options) !void {
    if (opt.runs == 0 or opt.runs > limits.bench_runs_max) {
        return args.fail(io, "--runs must be in 1..{d}", .{limits.bench_runs_max});
    }
    if (opt.max_tokens < 2) {
        return args.fail(io, "--max-tokens must be >= 2 (need at least 1 step for ITL)", .{});
    }

    try refuseGpuFlags(io, opt);
    try refuseCpuProfile(io, opt);
}

fn refuseGpuFlags(io: std.Io, opt: common.Options) error{UsageError}!void {
    const gpu_layers = opt.engine.gpu_layers;

    if (!fused_build) {
        const need_fused = "requires -Dops-backend=vulkan_fused";
        if (gpu_layers != null) return args.fail(io, "--gpu-layers " ++ need_fused, .{});

        // The device profilers read device timestamps.
        if (opt.profile_prefill) return args.fail(io, "--profile-prefill " ++ need_fused, .{});
        if (opt.profile_decode) return args.fail(io, "--profile-decode " ++ need_fused, .{});
    }

    if (opt.force_hybrid) {
        const layers = gpu_layers orelse {
            return args.fail(io, "--force-hybrid requires --gpu-layers", .{});
        };

        // A forced degenerate split needs a pin, because auto picks its own split.
        if (layers == .auto) {
            return args.fail(io, "--force-hybrid requires a --gpu-layers count", .{});
        }
    }

    if (gpu_layers != null and (opt.profile_prefill or opt.profile_decode)) {
        return args.fail(
            io,
            "--gpu-layers benches the hybrid runner and does not combine with profile flags",
            .{},
        );
    }
}

fn refuseCpuProfile(io: std.Io, opt: common.Options) !void {
    if (!opt.profile_cpu_decode) return;

    // The per-op CPU timer compiles into a hot-trace build only, so a release
    // binary holds no clock read in the forward pass.
    if (!core_log.hot) {
        return args.fail(
            io,
            "--profile-cpu-decode requires a hot-trace build (OPT=Debug or TRACE=1)",
            .{},
        );
    }

    if (fused_build or per_op_build) {
        common.writeStderr(
            io,
            "error: --profile-cpu-decode is for CPU backends (rebuild without -Druntime=vulkan)\n",
        );
        return error.UnsupportedArchForBackend;
    }
}
