//! The split path of `bench-model`: `--gpu-layers` names a count or `auto`, the
//! real planner prices the split, and the real hybrid runner runs it. A split that resolves to
//! zero or all host layers refuses, because a whole-model runner serves it.
//!
//! Comptime. Generic over the model family. A block rejects a non-fused build.

const std = @import("std");
const zigferencer = @import("zigferencer");
const build_options = @import("build_options");

const common = @import("bench_model_common.zig");
const exec = @import("bench_model_exec.zig");
const boot_lines = zigferencer.cli.boot_lines;
const boot_resources = zigferencer.cli.boot_resources;
const resolve_mod = zigferencer.resolve;
const kv_budget = zigferencer.kv_budget;
const vocab_mod = zigferencer.vocab;
const parallel = zigferencer.parallel;
const Pool = zigferencer.pool.Pool;
const vulkan_ctx_mod = zigferencer.vulkan_ctx;

const log = std.log.scoped(.kv);

comptime {
    if (!build_options.use_vulkan_fused_ops) {
        @compileError("dev/bench_model/bench_model_hybrid.zig was included in a non-fused build.");
    }
}

const Bench = common.Bench;
const device_placement = resolve_mod.device_placement;

// -- constants --------------------------------------------------------------

const mib = 1024 * 1024;

// -- api --------------------------------------------------------------------

/// Resolve the split on a new device context, build the hybrid runner of `Family`
/// at it, and run the series.
pub fn bench(comptime Family: type, b: *const Bench) !void {
    const ctx = b.ctx;
    const opt = b.opt;
    const kv_dtype = b.kvDtype();

    // Refuse a bad pin, then start the pool and the device.
    const pin = opt.engine.gpu_layers.?.pin();
    try boot_resources.refusePastModel(ctx.io, pin, b.loaded.card.n_layers);

    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, b.policy);
    defer pool.deinit();

    // The context defer runs last, after the runner releases its borrow.
    const ctxp = try boot_resources.openDevice(ctx);
    defer ctxp.deinit();

    // Resolve the split and build the hybrid runner.
    const hybrid_placement = try resolveSplit(b, ctxp);
    const plan = hybrid_placement.plan.?;

    const runner = try Family.Hybrid.init(
        ctx.gpa,
        ctxp,
        b.loaded,
        .{
            .plan = plan,
            .pool = &pool,
            .policy = b.policy,
            .kv_dtype = kv_dtype,
        },
    );
    defer runner.deinit();

    // Encode the prompt and warm the runner.
    const v = try vocab_mod.buildFromParsed(ctx.arena, b.loaded.parsed);
    const prompt_ids = try exec.promptIds(b, &v, hybrid_placement.window.base);
    try exec.warmup(runner, prompt_ids);

    // Run the series.
    const turn: exec.Turn = .{
        .vocab = &v,
        .prompt_ids = prompt_ids,
        .decode_len = opt.max_tokens,
        .ignore_eos = opt.ignore_eos,
    };
    const plan_info: common.PlanInfo = .{
        .n_slots = plan.n_slots,
        .window = plan.window,
        .kv_mib = plan.kv_vram / mib,
        .kv_dtype = kv_dtype,
    };
    return exec.runSeries(runner, b, turn, plan_info, true);
}

/// Resolve the split on the context, and log the placement and its price.
pub fn resolveSplit(b: *const Bench, ctxp: *vulkan_ctx_mod.Context) !device_placement.Hybrid {
    const io = b.ctx.io;
    const name = b.model.pub_name;

    const device = try b.boot.resolveOn(io, ctxp);
    const layer_choice = device.layer_choice.?;
    if (layer_choice.clamp) |clamp| {
        boot_lines.logClamp(clamp);
    }

    if (layer_choice.layer_split) |split| {
        if (layer_choice.pin == null) {
            boot_lines.logAutoSplit(split.host_layers, b.loaded.card.n_layers);
        }
    }

    const hybrid = switch (device.placement) {
        .ok => |placement| placement.hybrid,
        .refused => |r| return boot_lines.refuse(io, name, r),
    };

    switch (hybrid.quote) {
        .ok => |quote| boot_lines.logRamQuote(quote, false),
        .refused => |r| return boot_lines.refuse(io, name, r),
    }

    const plan = hybrid.plan.?;
    boot_lines.logSessionWindow(hybrid.window);

    var span_buffer: [kv_budget.layer_split_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatLayerSplit(plan, &span_buffer)});

    var ledger_buffer: [kv_budget.layer_split_ledger_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatLedger(plan, &ledger_buffer)});

    return hybrid;
}
