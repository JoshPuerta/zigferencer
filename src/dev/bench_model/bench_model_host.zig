//! The host path of `bench-model`: the CPU backends and the per-op Vulkan backend.
//! Each run builds a fresh runner, which starts the KV cache at zero.
//!
//! Comptime. Generic over the model family. Build options pick the per-op Vulkan paths.

const std = @import("std");
const zigferencer = @import("zigferencer");
const build_options = @import("build_options");

const common = @import("bench_model_common.zig");
const exec = @import("bench_model_exec.zig");
const profile = @import("bench_model_profile.zig");
const boot_lines = zigferencer.cli.boot_lines;
const boot_resources = zigferencer.cli.boot_resources;
const core_log = zigferencer.log;
const kv_budget = zigferencer.kv_budget;
const limits = zigferencer.limits;
const vocab_mod = zigferencer.vocab;
const parallel = zigferencer.parallel;
const Pool = zigferencer.pool.Pool;

const per_op_build = build_options.use_vulkan_ops;
const vulkan_ops_wrap = if (per_op_build) zigferencer.ops_vulkan_wrap else void;

const Bench = common.Bench;
const RunResult = common.RunResult;

// -- api --------------------------------------------------------------------

pub fn bench(comptime Family: type, b: *const Bench) !void {
    const ctx = b.ctx;

    // Refuse a family that the per-op Vulkan backend cannot run.
    if (comptime per_op_build and !Family.per_op_bench) {
        common.writeStderr(
            ctx.io,
            "error: qwen35 bench does not run on per-op vulkan. " ++
                "Build -Dops-backend=vulkan_fused (fused GPU) or a CPU backend\n",
        );
        return error.UnsupportedArchForBackend;
    }

    // Start the pool. One pool serves every CPU op of the command.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, b.policy);
    defer pool.deinit();

    // Start the per-op backend. It holds one runner and weight cache, so weights upload once.
    if (comptime per_op_build) {
        try vulkan_ops_wrap.initBackend(ctx.arena, ctx.vk_options);
    }

    defer if (comptime per_op_build) {
        vulkan_ops_wrap.deinitBackend();
    };

    // Log the placement and encode the prompt.
    const max_seq = b.resolved.window.base;
    boot_lines.logSessionWindow(b.resolved.window);
    try boot_resources.hostRamPlan(ctx.io, b.boot, b.kvDtype(), b.resolved.window, b.resolved.cpu);

    const v = try vocab_mod.buildFromParsed(ctx.arena, b.loaded.parsed);
    const prompt_ids = try exec.promptIds(b, &v, max_seq);

    // Hand a CPU profile run to its own path.
    if (comptime core_log.hot) {
        if (b.opt.profile_cpu_decode) {
            return profile.bench(Family, b, &pool, prompt_ids, max_seq);
        }
    }

    // Run the series on a fresh runner per run.
    const turn: exec.Turn = .{
        .vocab = &v,
        .prompt_ids = prompt_ids,
        .decode_len = b.opt.max_tokens,
        .ignore_eos = b.opt.ignore_eos,
    };
    const results = try ctx.arena.alloc(RunResult, b.opt.runs);
    for (results) |*result| {
        var runner = try Family.Cpu.init(ctx.arena, b.loaded, .{
            .plan = kv_budget.pinned(1, max_seq),
            .max_rows = limits.cpu_prefill_chunk,
            .pool = &pool,
            .policy = b.policy,
            .kv_dtype = b.kvDtype(),
        });
        result.* = try exec.runOnce(&runner, turn, ctx);
    }

    return exec.printReport(b, prompt_ids.len, results);
}
