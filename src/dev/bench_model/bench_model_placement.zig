//! The split path of `bench-model`: `--gpu-layers` names a count or `auto`, the
//! real planner prices the split, and the real hybrid runner runs it.
//! `--force-hybrid` also builds at a degenerate split, which production selection
//! never does. That forced construction is the cross-runner token gate.

const std = @import("std");
const zigferencer = @import("zigferencer");
const build_options = @import("build_options");

const args = zigferencer.cli.args;
const common = @import("bench_model_common.zig");
const exec = @import("bench_model_exec.zig");
const boot_lines = zigferencer.cli.boot_lines;
const resolve_mod = zigferencer.resolve;
const kv_budget = zigferencer.kv_budget;
const vocab_mod = zigferencer.vocab;
const parallel = zigferencer.parallel;
const Pool = zigferencer.pool.Pool;
const vulkan_ctx_mod = zigferencer.vulkan_ctx;

const log = std.log.scoped(.kv);

comptime {
    if (!build_options.use_vulkan_fused_ops) {
        @compileError("cli/bench_model_placement.zig was included in a non-fused build.");
    }
}

const Bench = common.Bench;
const device_arm = resolve_mod.device_arm;

// -- constants --------------------------------------------------------------

const mib = 1024 * 1024;

// -- api --------------------------------------------------------------------

/// Resolve the split on a new device context, build the hybrid runner of `Family`
/// at it, and run the series.
pub fn bench(comptime Family: type, b: *const Bench) !void {
    const ctx = b.ctx;
    const opt = b.opt;
    const kv_dtype = b.kvDtype();

    if (comptime !Family.forced_split) {
        if (opt.force_hybrid) {
            return args.fail(
                ctx.io,
                "--force-hybrid is llama-family only. The qwen hybrid holds no degenerate " ++
                    "split, so its cross-runner gate compares the fused and CPU runners " ++
                    "against a real split",
                .{},
            );
        }
    }

    try refuseOverLayerCount(b);

    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, b.policy);
    defer pool.deinit();

    // Declared first, so it runs last: the runner releases its borrow of the
    // context before this.
    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{
        .storage_range_override = ctx.vk_storage_range,
    });
    defer ctxp.deinit();

    const hybrid_placement = try resolveSplit(b, ctxp);
    const plan = hybrid_placement.plan.?;

    const runner = try Family.Hybrid.init(
        ctx.gpa,
        ctxp,
        b.loaded,
        plan,
        ctx.io,
        &pool,
        b.policy,
        kv_dtype,
    );
    defer runner.deinit();

    const v = try vocab_mod.buildFromParsed(ctx.arena, b.loaded.parsed);
    const prompt_ids = try exec.promptIds(b, &v, hybrid_placement.window.base);
    try exec.warmup(runner, prompt_ids);

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

// -- helpers ----------------------------------------------------------------

/// A pin past the layer count costs no device, so it refuses before the context.
fn refuseOverLayerCount(b: *const Bench) error{UsageError}!void {
    const gpu_layers = b.opt.engine.gpu_layers.?.pin() orelse return;
    const n_layers = b.loaded.card.n_layers;

    if (gpu_layers > n_layers) {
        const fmt = "--gpu-layers {d} exceeds the model's {d} layers";
        return args.fail(b.ctx.io, fmt, .{ gpu_layers, n_layers });
    }
}

/// Resolve the split on the context, and log the placement and its price.
fn resolveSplit(b: *const Bench, ctxp: *vulkan_ctx_mod.Context) !device_arm.Hybrid {
    const io = b.ctx.io;
    const name = b.model.pub_name;

    const device = try b.boot.resolveOn(io, ctxp);
    const placed = device.placed.?;
    if (placed.clamp) |clamp| {
        boot_lines.logClamp(clamp);
    }

    if (placed.split) |split| {
        if (placed.pin == null) {
            boot_lines.logAutoSplit(split.split, b.loaded.card.n_layers);
        }
    }

    const hybrid = switch (device.arm) {
        .ok => |arm| arm.hybrid,
        .refused => |r| return boot_lines.refuse(io, name, r),
    };

    switch (hybrid.quote) {
        .ok => |quote| boot_lines.logRamQuote(quote, false),
        .refused => |r| return boot_lines.refuse(io, name, r),
    }

    const plan = hybrid.plan.?;
    boot_lines.logSessionWindow(hybrid.window);

    var span_buffer: [kv_budget.placement_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatPlacement(plan, &span_buffer)});

    var ledger_buffer: [kv_budget.placement_ledger_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatLedger(plan, &ledger_buffer)});

    return hybrid;
}
