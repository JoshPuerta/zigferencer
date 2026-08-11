//! Qwen3.5 fused-GPU bench modes (-Dops-backend=vulkan_fused only). The qwen35
//! hybrid GDN runner is shaped differently from the Llama LlamaFusedRunner, so it
//! gets its own bench path. Only the fused build imports this module, so its
//! GPU-only imports are never analyzed on a CPU build.

const std = @import("std");
const Io = std.Io;

const cli = @import("command.zig");
const args = @import("args.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const chat_mod = @import("../core/chat.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_cost = @import("../runtime/kv_cost.zig");
const dtype = @import("../core/dtype.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;
const sampler = @import("../runtime/sampler.zig");
const qwen_fused = @import("../runtime/qwen_fused.zig");
const qwen_hybrid = @import("../runtime/qwen_hybrid.zig");
const vulkan_ctx_mod = @import("../runtime/vulkan_ctx.zig");
const common = @import("bench_model_common.zig");
const report = @import("bench_model_report.zig");
const exec = @import("bench_model_exec.zig");
const placement = @import("bench_model_placement.zig");
const serve_common = @import("serve_common.zig");
const RunResult = common.RunResult;
const synthPromptIds = common.synthPromptIds;
const writeStderr = common.writeStderr;
const printTable = report.printTable;
const printJson = report.printJson;

// -- api --------------------------------------------------------------------

/// The Qwen3.5 placement bench. Its cost vector prices the full-attention layers
/// and the recurrent slabs at separate rates, not at one uniform price, which is
/// the whole difference from the llama head in `bench_model_llama.zig`.
pub fn qwenHybridBench(
    ctx: cli.Context,
    boot: *const serve_common.DeviceBoot,
    loaded: *const loader.Loaded,
    model: registry.Model,
    opt: common.Options,
    pool: *Pool,
    policy: parallel.Policy,
    kv_dtype: dtype.KvDtype,
) !void {
    // The qwen hybrid refuses a degenerate split by construction: the whole
    // model on one tier is a whole-model runner, and the CLI dispatches those
    // directly. So the forced construction has no qwen instance to make.
    if (opt.force_hybrid) {
        return args.fail(ctx.io, "--force-hybrid is llama-family only. The qwen hybrid holds no degenerate split, so its cross-runner gate compares the fused and CPU runners against a real split", .{});
    }
    if (opt.engine.gpu_layers.?.pin()) |n_gpu| try placement.refuseOverLayerCount(ctx, n_gpu, loaded.card.n_layers);
    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    return placement.placedBench(qwen_hybrid.QwenHybridRunner, ctx, boot, ctxp, loaded, model, opt, pool, policy, kv_dtype);
}

/// Fused single-CB GPU bench for Qwen3.5 (-Dops-backend=vulkan_fused). It uses
/// its own QwenFusedRunner, which prefills sequentially through `step()` and so
/// differs from LlamaFusedRunner. All runs share one runner. Before each run
/// after the first, the bench resets the slot state, so every run starts from
/// the state of a fresh runner and gives the same tokens.
pub fn qwenFusedBench(
    ctx: cli.Context,
    boot: *const serve_common.DeviceBoot,
    loaded: *const loader.Loaded,
    model: registry.Model,
    opt: common.Options,
    kv_dtype: dtype.KvDtype,
) !void {
    const prompt = opt.prompt;
    const decode_len = opt.max_tokens;
    const chat = opt.chat;
    const ignore_eos = opt.ignore_eos;
    const json_out = opt.json;
    const prompt_tokens = opt.prompt_tokens;
    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);

    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    defer ctxp.deinit();
    // One slot at the base window, which a default shrinks to the device
    // limits and then to the VRAM figure.
    const f = try serve_common.fusedOn(ctx.io, boot, ctxp);
    _ = try serve_common.logFused(ctx.io, model.pub_name, f, true, false, kv_dtype);
    const plan = f.plan;
    const max_seq = plan.window;
    const qr = try qwen_fused.QwenFusedRunner.init(ctx.gpa, ctxp, loaded, plan, kv_dtype);
    defer qr.deinit();

    const text = if (chat) try chat_mod.wrapAlloc(ctx.arena, model.chat_template_kind, prompt) else prompt;
    // The fit check runs first, because the synthesis allocates n ids.
    if (prompt_tokens) |n| try common.requireSequenceFits(ctx.io, n, decode_len, max_seq);
    const prompt_ids = if (prompt_tokens) |n|
        try synthPromptIds(ctx.arena, n, loaded.card.vocab_size)
    else
        try bpe.encodeAlloc(&v, ctx.arena, text, true);
    if (prompt_ids.len == 0) {
        writeStderr(ctx.io, "error: the prompt encodes to zero tokens\n");
        return error.EmptyPrompt;
    }
    try common.requireSequenceFits(ctx.io, prompt_ids.len, decode_len, max_seq);

    // --profile-prefill: prefill every chunk but the last without the profiler,
    // then one timestamped run of the last chunk, per-stage breakdown, exit. So
    // a long prompt profiles the chunk at its full depth. Profiling mutates the
    // runner's GDN and KV state, so no decode follows it.
    if (opt.profile_prefill) {
        const batch_max: usize = qr.prefill_batch_max;
        const last_len = prompt_ids.len - (prompt_ids.len - 1) / batch_max * batch_max;
        const last_start = prompt_ids.len - last_len;
        std.debug.assert(last_len >= 1);
        std.debug.assert(last_len <= batch_max);
        var off: usize = 0;
        while (off < last_start) : (off += batch_max) {
            _ = try qr.prefillBatch(prompt_ids[off..][0..batch_max], @intCast(off));
        }
        std.debug.assert(off == last_start);
        var tag_buf: [40]u8 = undefined;
        const tag = std.fmt.bufPrint(&tag_buf, "profile from {d}", .{last_start}) catch unreachable;
        try qr.profilePrefill(ctx.io, prompt_ids[last_start..], @intCast(last_start), tag);
        return;
    }

    // --profile-decode: prefill the prompt, which gives a realistic position and
    // a populated KV and state, then one timestamped decode step per stage.
    if (opt.profile_decode) {
        var pos: u32 = 0;
        var last_logits: []f32 = undefined;
        var off: usize = 0;
        while (off < prompt_ids.len and pos < qr.max_seq) {
            const room = qr.max_seq - pos;
            const chunk: u32 = @intCast(@min(@min(prompt_ids.len - off, qr.prefill_batch_max), room));
            last_logits = try qr.prefillBatch(prompt_ids[off..][0..chunk], pos);
            pos += chunk;
            off += chunk;
        }
        try qr.profileStep(ctx.io, sampler.greedy(last_logits), pos, "profile");
        return;
    }

    std.debug.assert(opt.runs >= 1);
    std.debug.assert(qr.n_slots >= 1);
    const results = try ctx.arena.alloc(RunResult, opt.runs);
    const plan_info = common.PlanInfo.fromPlan(plan, kv_dtype);
    var r: u32 = 0;
    while (r < opt.runs) : (r += 1) {
        // Each run folds the prompt into the GDN state of slot 0, and no prefill
        // rewrites that state. The scrub is the one serve uses on slot reuse.
        // Positional KV needs no scrub, because each run writes a row before
        // attention reads it.
        if (r > 0) qr.resetSlotState(0);
        results[r] = try exec.runOnce(qr, &v, prompt_ids, decode_len, ignore_eos, ctx);
        results[r].plan = plan_info;
    }

    // The GPU path has no thread count, and the report header names the backend.
    const report_policy: parallel.Policy = .{};
    if (json_out) {
        try printJson(ctx.io, model.pub_name, prompt_ids.len, decode_len, results, report_policy);
    } else {
        try printTable(ctx.io, model.pub_name, prompt, prompt_ids.len, decode_len, results, report_policy);
    }
}
