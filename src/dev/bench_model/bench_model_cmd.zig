//! `zigferencer-dev bench-model <pub_name>` runs the full forward pass on a prompt
//! and repeats it `--runs` times. It reports TTFT and ITL with mean, stddev, min
//! and max, and an FNV-1a 64 hash of the generated token ids.
//!
//! Comptime. Generic over the model family. The fused build option picks the GPU path.
//!
//! A stable hash across runs proves the forward pass is deterministic. A stable
//! hash across binaries built with different `-Dops-backend` values is the
//! cross-backend gate, because a backend that changes the tokens changed the math.
//! `tools/gate.sh` drives that loop.
//!
//! This file checks the flags. `common.loadAndRun` loads the model, and this file picks one
//! path. Each path owns its runner: `bench_model_host.zig` (CPU and per-op Vulkan),
//! `bench_model_fused.zig` (the whole model on the GPU), and `bench_model_hybrid.zig` (a CPU and
//! GPU split).

const std = @import("std");
const zigferencer = @import("zigferencer");
const build_options = @import("build_options");

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;
const common = @import("bench_model_common.zig");
const host = @import("bench_model_host.zig");
const resolve_mod = zigferencer.resolve;

const fused_build = build_options.use_vulkan_fused_ops;

// The device paths resolve to `void` on a CPU build, which never analyzes them.
const fused = if (fused_build) @import("bench_model_fused.zig") else void;
const hybrid_path = if (fused_build) @import("bench_model_hybrid.zig") else void;

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const opt = try args.parseInto(common.Options, ctx) orelse return;
    try common.refuseBadFlags(ctx.io, opt);

    const input: resolve_mod.Input = .{ .engine = opt.engine };
    return common.loadAndRun(ctx, opt, .bench, input, benchFamily, {});
}

// -- helpers ----------------------------------------------------------------

/// Pick the path of this build and these flags.
fn benchFamily(comptime Family: type, b: *const common.Bench, _: void) !void {
    if (comptime fused_build) {
        if (b.opt.engine.gpu_layers == null) return fused.bench(Family, b);
        return hybrid_path.bench(Family, b);
    } else {
        return host.bench(Family, b);
    }
}
