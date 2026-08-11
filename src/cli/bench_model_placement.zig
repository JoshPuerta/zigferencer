//! The family-independent half of the bench-model placement path: resolve a
//! `--gpu-layers` split, a count or auto, through the real placement planner,
//! construct the family's hybrid runner through its real init, and drive the
//! shared run body.
//! `--force-hybrid` also constructs at the degenerate splits, which production
//! selection never does, and that forced construction is the cross-runner token
//! gate.
//!
//! Two families reach this body. `resolve` prices the split of each family
//! from the facts the boot measured, and `placedBench` logs, constructs, and
//! benches from the value.

const std = @import("std");
const build_options = @import("build_options");

const cli = @import("command.zig");
const args = @import("args.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const chat_mod = @import("../core/chat.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const tensor_lookup = @import("../runtime/tensor_lookup.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;
const dtype = @import("../core/dtype.zig");
const vulkan_ctx_mod = @import("../runtime/vulkan_ctx.zig");
const serve_common = @import("serve_common.zig");

const common = @import("bench_model_common.zig");
const report = @import("bench_model_report.zig");
const exec = @import("bench_model_exec.zig");

const log = std.log.scoped(.kv);
const writeStderr = common.writeStderr;

comptime {
    if (!build_options.use_vulkan_fused_ops) {
        @compileError("cli/bench_model_placement.zig was included in a non-fused build.");
    }
}

// -- api --------------------------------------------------------------------

/// The one refusal that costs no device: a pin past the model's layer count.
/// Both family heads run it before they create a Context.
pub fn refuseOverLayerCount(ctx: cli.Context, gpu_layers: u32, n_layers: u32) error{UsageError}!void {
    if (gpu_layers > n_layers) {
        return args.fail(ctx.io, "--gpu-layers {d} exceeds the model's {d} layers", .{ gpu_layers, n_layers });
    }
}

/// Resolve the split on the context, construct `HybridT` at it, and run the
/// shared run body. The function owns `ctxp` and releases it on every path.
pub fn placedBench(
    comptime HybridT: type,
    ctx: cli.Context,
    boot: *const serve_common.DeviceBoot,
    ctxp: *vulkan_ctx_mod.Context,
    loaded: *const loader.Loaded,
    model: registry.Model,
    opt: common.Options,
    pool: *Pool,
    policy: parallel.Policy,
    kv_dtype: dtype.KvDtype,
) !void {
    // Declared first, so it runs last: the runner releases its borrow of the context before this.
    defer ctxp.deinit();
    const card = loaded.card;
    const n_layers: u32 = card.n_layers;
    const dev = try boot.resolveOn(ctx.io, ctxp);
    const p = dev.placed.?;
    if (p.clamp) |c| serve_common.logClamp(c);
    if (p.split) |s| if (p.pin == null) serve_common.logAutoSplit(s.split, n_layers);
    const h = switch (dev.arm) {
        .ok => |a| a.hybrid,
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    };
    switch (h.quote) {
        .ok => |q| serve_common.logRamQuote(q, false),
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    }
    const plan = h.plan.?;
    const max_seq: u32 = h.window.base;

    serve_common.logSessionWindow(h.window);
    var span_buf: [kv_budget.placement_text_max]u8 = undefined;
    const span = kv_budget.formatPlacement(plan, &span_buf) catch unreachable;
    log.info("{s}", .{span});
    var ledger_buf: [kv_budget.placement_ledger_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatLedger(plan, &ledger_buf) catch unreachable});
    const mib = 1024 * 1024;

    const hybrid = try HybridT.init(ctx.gpa, ctxp, loaded, plan, ctx.io, pool, policy, kv_dtype);
    defer hybrid.deinit();

    const plan_info: common.PlanInfo = .{
        .n_slots = plan.n_slots,
        .window = plan.window,
        .kv_mib = plan.kv_vram / mib,
        .kv_dtype = kv_dtype,
    };

    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);

    const text = if (opt.chat)
        try chat_mod.wrapAlloc(ctx.arena, model.chat_template_kind, opt.prompt)
    else
        opt.prompt;
    // The fit check runs first, because the synthesis allocates n ids.
    if (opt.prompt_tokens) |n| try common.requireSequenceFits(ctx.io, n, opt.max_tokens, max_seq);
    const prompt_ids = if (opt.prompt_tokens) |n|
        try common.synthPromptIds(ctx.arena, n, card.vocab_size)
    else
        try bpe.encodeAlloc(&v, ctx.arena, text, true);
    if (prompt_ids.len == 0) {
        writeStderr(ctx.io, "error: the prompt encodes to zero tokens\n");
        return error.EmptyPrompt;
    }
    try common.requireSequenceFits(ctx.io, prompt_ids.len, opt.max_tokens, max_seq);

    // Warmup: one prefill chunk and one decode step, so run 1 starts warm. That
    // is the fused-bench rule.
    {
        const chunk = @min(prompt_ids.len, @as(usize, hybrid.prefill_batch_max));
        _ = try hybrid.prefillBatch(prompt_ids[0..chunk], 0);
        _ = try hybrid.step(prompt_ids[prompt_ids.len - 1], @intCast(chunk));
    }

    const results = try ctx.arena.alloc(common.RunResult, opt.runs);
    var r: u32 = 0;
    while (r < opt.runs) : (r += 1) {
        // The warmup and every run fold the prompt into the recurrent state, and
        // no prefill rewrites it. Positional KV heals on slot reuse, and GDN
        // state does not. The scrub gives every run one clean start. Without it
        // the 9B diverges run to run, and the 2B only hides the leak.
        if (comptime @hasDecl(HybridT, "has_recurrent_state")) hybrid.resetSlotState(0);
        results[r] = try exec.doOneRun(
            hybrid,
            loaded,
            max_seq,
            &v,
            prompt_ids,
            opt.max_tokens,
            opt.ignore_eos,
            ctx,
            pool,
            policy,
            kv_dtype,
        );
        results[r].plan = plan_info;
    }

    if (opt.json) {
        try report.printJson(ctx.io, model.pub_name, prompt_ids.len, opt.max_tokens, results, policy);
    } else {
        try report.printTable(ctx.io, model.pub_name, opt.prompt, prompt_ids.len, opt.max_tokens, results, policy);
    }
}
