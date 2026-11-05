//! The host path of `bench-model`: the CPU backends and the per-op Vulkan backend.
//! Each run builds a fresh runner, which starts the KV cache at zero.

const std = @import("std");
const build_options = @import("build_options");

const common = @import("bench_model_common.zig");
const exec = @import("bench_model_exec.zig");
const profile = @import("bench_model_profile.zig");
const serve_common = @import("serve_common.zig");
const resolve_mod = @import("../engine/resolve.zig");
const core_log = @import("../core/log.zig");
const kv_budget = @import("../core/kv_budget.zig");
const limits = @import("../core/limits.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;

const per_op_build = build_options.use_vulkan_ops;
const vulkan_ops_wrap = if (per_op_build) @import("../ops_vulkan/wrap.zig") else void;

const Bench = common.Bench;
const RunResult = common.RunResult;

// -- api --------------------------------------------------------------------

pub fn bench(comptime Family: type, b: *const Bench) !void {
    const ctx = b.ctx;

    if (comptime per_op_build and !Family.per_op_bench) {
        common.writeStderr(
            ctx.io,
            "error: qwen35 bench does not run on per-op vulkan. " ++
                "Build -Dops-backend=vulkan_fused (fused GPU) or a CPU backend\n",
        );
        return error.UnsupportedArchForBackend;
    }

    // One pool serves every CPU op of the command.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, b.policy);
    defer pool.deinit();

    // The per-op backend holds one process-wide runner and weight cache, so the
    // weights upload once across every run.
    if (comptime per_op_build) {
        try vulkan_ops_wrap.initBackend(ctx.arena, .{
            .storage_range_override = ctx.vk_storage_range,
        });
    }

    defer if (comptime per_op_build) {
        vulkan_ops_wrap.deinitBackend();
    };

    const max_seq = b.resolved.window.base;
    serve_common.logSessionWindow(b.resolved.window);
    try logHostRam(b);

    const v = try vocab_mod.buildFromParsed(ctx.arena, b.loaded.parsed);
    const prompt_ids = try exec.promptIds(b, &v, max_seq);

    if (comptime core_log.hot) {
        if (b.opt.profile_cpu_decode) {
            return profile.bench(Family, b, &pool, prompt_ids, max_seq);
        }
    }

    const turn: exec.Turn = .{
        .vocab = &v,
        .prompt_ids = prompt_ids,
        .decode_len = b.opt.max_tokens,
        .ignore_eos = b.opt.ignore_eos,
    };
    const results = try ctx.arena.alloc(RunResult, b.opt.runs);
    for (results) |*result| {
        var runner = try Family.Cpu.init(
            ctx.arena,
            b.loaded,
            kv_budget.pinned(1, max_seq),
            limits.cpu_prefill_chunk,
            ctx.io,
            &pool,
            b.policy,
            b.kvDtype(),
        );
        result.* = try exec.runOnce(&runner, turn, ctx);
    }

    return exec.printReport(b, prompt_ids.len, results);
}

// -- helpers ----------------------------------------------------------------

/// A host runner holds the KV on the host, so its quote is the CPU end of the
/// vector. The per-op Vulkan build resolved a device placement, so it resolves
/// the host placement here.
fn logHostRam(b: *const Bench) !void {
    const io = b.ctx.io;
    const resolved = b.resolved;
    const facts: resolve_mod.Facts = .{ .ram_free = b.boot.ramFree(io) };

    const host = resolved.cpu orelse
        resolve_mod.cpuArm(
            .bench,
            b.boot.input,
            b.loaded.card,
            b.kvDtype(),
            resolved.window,
            facts,
        );
    switch (host) {
        .ok => |placed| serve_common.logRamQuote(placed.quote, false),
        .refused => |r| return serve_common.refuse(io, b.model.pub_name, r),
    }
}
