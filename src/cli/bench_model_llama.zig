//! The bench-model path that only the llama family reaches: the pinned
//! placement bench. Its qwen twin lives in `bench_model_qwen.zig`, and the
//! family-independent placement body both heads call lives in
//! `bench_model_placement.zig`. The per-op CPU decode profile also lives here,
//! and it serves both CPU runner families.
//!
//! `cpuProfileBench` prefills the prompt unprofiled, so its numbers are
//! steady-state decode and not prefill. It then decodes `decode_len` steps with
//! a `cpu_profile.CpuProfile` installed, and prints the per-op-category
//! breakdown and the serial-fraction summary. The breakdown is a diagnostic and
//! not program output, so it goes to stderr under the project rule that stdout
//! carries results. The timer compiles into a `-Dtrace` build only, so
//! `bench_model_cmd.zig` names that entry behind the same comptime flag and
//! refuses the flag elsewhere. `cpu_profile.ProfileCat` is the single list of
//! categories, so a new category reaches the table without an edit here.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const cli = @import("command.zig");
const args = @import("args.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_cost = @import("../runtime/kv_cost.zig");
const limits = @import("../core/limits.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;
const sampler = @import("../runtime/sampler.zig");
const dtype = @import("../core/dtype.zig");
const llama_runner = @import("../runtime/llama_runner.zig");
const LlamaRunner = llama_runner.LlamaRunner;
const QwenRunner = @import("../runtime/qwen_runner.zig").QwenRunner;
const cpu_profile = @import("../runtime/cpu_profile.zig");
const CpuProfile = cpu_profile.CpuProfile;
const ProfileCat = cpu_profile.ProfileCat;
const common = @import("bench_model_common.zig");
const serve_common = @import("serve_common.zig");

// The placement bench needs the device stack. A CPU build resolves each alias
// to `void` and never analyzes the arm that names it, the way every other
// construction site gates.
const vulkan_ctx_mod = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/vulkan_ctx.zig")
else
    void;
const llama_fused = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/llama_fused.zig")
else
    void;
const llama_hybrid = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/llama_hybrid.zig")
else
    void;
const placement = if (build_options.use_vulkan_fused_ops)
    @import("bench_model_placement.zig")
else
    void;

// -- api --------------------------------------------------------------------

/// The llama placement bench. It hands the shared body its own hybrid runner
/// type.
pub fn llamaHybridBench(
    ctx: cli.Context,
    boot: *const serve_common.DeviceBoot,
    loaded: *const loader.Loaded,
    model: registry.Model,
    opt: common.Options,
    pool: *Pool,
    policy: parallel.Policy,
    kv_dtype: dtype.KvDtype,
) !void {
    if (comptime !build_options.use_vulkan_fused_ops) {
        return args.fail(ctx.io, "--gpu-layers requires -Dops-backend=vulkan_fused", .{});
    } else {
        if (opt.engine.gpu_layers.?.pin()) |n_gpu| try placement.refuseOverLayerCount(ctx, n_gpu, loaded.card.n_layers);
        const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
        ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
        return placement.placedBench(llama_hybrid.LlamaHybridRunner, ctx, boot, ctxp, loaded, model, opt, pool, policy, kv_dtype);
    }
}

/// `bench-model --profile-cpu-decode` (CPU backends, both families): the
/// per-op decode accounting tool. It prefills unprofiled, decodes `decode_len`
/// steps with the timer installed, and prints the category breakdown to stderr.
pub fn cpuProfileBench(
    ctx: cli.Context,
    loaded: *const loader.Loaded,
    max_seq: u32,
    model_name: []const u8,
    backend: []const u8,
    prompt_ids: []const u32,
    decode_len: u32,
    pool: *Pool,
    policy: parallel.Policy,
    kv_dtype: dtype.KvDtype,
) !void {
    const plan = kv_budget.pinned(1, max_seq);
    var prof: CpuProfile = switch (loaded.card.arch) {
        .qwen35 => blk: {
            var runner = try QwenRunner.init(ctx.arena, loaded, plan, limits.cpu_prefill_chunk, ctx.io, pool, policy, kv_dtype);
            break :blk try profileDecode(QwenRunner, &runner, .qwen, prompt_ids, decode_len, max_seq);
        },
        .llama => blk: {
            var runner = try LlamaRunner.init(ctx.arena, loaded, plan, limits.cpu_prefill_chunk, ctx.io, pool, policy, kv_dtype);
            break :blk try profileDecode(LlamaRunner, &runner, .llama, prompt_ids, decode_len, max_seq);
        },
    };
    const pos_end: u32 = @intCast(@min(prompt_ids.len, max_seq) + prof.steps);
    try reportCpuProfile(ctx.io, &prof, model_name, backend, policy, loaded.card.n_layers, prompt_ids.len, pos_end);
}

// -- helpers ----------------------------------------------------------------

/// Prefill `prompt_ids` unprofiled in batched chunks, then decode up to
/// `decode_len` greedy steps with a `family` profile installed, and return the
/// profile. The profiler stays off over the prompt because the target is
/// steady-state decode.
fn profileDecode(
    comptime Runner: type,
    runner: *Runner,
    family: cpu_profile.Family,
    prompt_ids: []const u32,
    decode_len: u32,
    max_seq: u32,
) !CpuProfile {
    std.debug.assert(prompt_ids.len > 0);
    const n_prompt: u32 = @intCast(@min(prompt_ids.len, max_seq));
    var pos: u32 = 0;
    var last: []f32 = undefined;
    while (pos < n_prompt) {
        const chunk: u32 = @min(n_prompt - pos, runner.prefill_batch_max);
        last = try runner.prefillBatch(prompt_ids[pos..][0..chunk], pos);
        pos += chunk;
    }
    var tok = sampler.greedy(last);

    var prof: CpuProfile = .{ .family = family };
    runner.profile = &prof;
    defer runner.profile = null;
    var k: u32 = 0;
    while (k < decode_len and pos < max_seq) : (k += 1) {
        const lg = try runner.step(tok, pos);
        pos += 1;
        tok = sampler.greedy(lg);
    }
    std.debug.assert(prof.steps == k);
    std.debug.assert(prof.total() > 0);
    return prof;
}

fn reportCpuProfile(
    io: Io,
    p: *const CpuProfile,
    model_name: []const u8,
    backend: []const u8,
    policy: parallel.Policy,
    n_layers: u32,
    prompt_len: usize,
    pos_end: u32,
) !void {
    const total: f64 = @floatFromInt(p.total());
    const steps: f64 = @floatFromInt(p.steps);
    const parallel_ns: f64 = @floatFromInt(p.nsParallel());
    const serial_ns = total - parallel_ns;
    const serial_frac = serial_ns / total;
    const n: f64 = @floatFromInt(@max(policy.n_threads, 1));
    // The ideal-Amdahl ceiling treats every parallel row as perfectly
    // parallel. The real ceiling is lower, because the matmuls are
    // bandwidth-bound.
    const amdahl = 1.0 / (serial_frac + (1.0 - serial_frac) / n);

    var buf: [2048]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    const w = &fw.interface;
    try w.print("[cpu-profile decode] model={s} backend={s} threads={d} flops={d}\n", .{ model_name, backend, policy.n_threads, policy.min_parallel_flops });
    try w.print("  layers={d} prompt_tokens={d} decode_steps={d} pos_end={d}\n\n", .{ n_layers, prompt_len, p.steps, pos_end });
    try w.print("  {s:<14}{s:>12}{s:>12}{s:>8}  kind\n", .{ "category", "ns/step", "ms/step", "%step" });
    try w.writeAll("  " ++ "-" ** 54 ++ "\n");
    for (std.enums.values(ProfileCat)) |cat| {
        if (!cat.usedBy(p.family)) continue;
        const ns: f64 = @floatFromInt(p.nsOf(cat));
        const ns_step = ns / steps;
        try w.print("  {s:<14}{d:>12.0}{d:>12.4}{d:>7.1}%  {s}\n", .{
            @tagName(cat),
            ns_step,
            ns_step / 1e6,
            100.0 * ns / total,
            if (cat.isParallel()) "parallel" else "serial",
        });
    }
    try w.writeAll("  " ++ "-" ** 54 ++ "\n");
    try w.print("  {s:<14}{d:>12.0}{d:>12.4}{d:>7.1}%\n", .{ "TOTAL", total / steps, total / steps / 1e6, 100.0 });
    try w.print("\n  parallel: {d:.1}%   serial: {d:.1}%\n", .{ 100.0 * (1.0 - serial_frac), 100.0 * serial_frac });
    try w.print("  ideal-Amdahl ceiling @ {d} threads: {d:.2}x  (real lower: the matmuls are bandwidth-bound)\n", .{ policy.n_threads, amdahl });
    try w.flush();
}
