//! `zigferencer-dev dispatch-list --model M --out FILE` writes the Vulkan commands that the runner
//! steps record, one line per command, so `diff -u` compares the runner code of two trees.
//!
//! Steps. The file holds six sections. The short sections carry the names of the GPU labels.
//! `prefill_chunk` is one `prefillBatch` of 7 fixed tokens on slot 0. `decode` is one `step` at
//! position 7. `decode_batch` is one `stepBatch` of 3 rows over 2 slots. The `_long` sections
//! repeat the three on slot 0 past `limits.attn_split_len`: prefill chunks of 64 rows, then a step
//! at position 300. An `argmax` line follows each section.
//!
//! Placement. No `--gpu-layers` builds the fused runner. A count builds the hybrid runner of the
//! family row through the placement path of `bench-model`. The boot resolves as a serve of 2
//! sessions, because a run and a bench plan 1 slot.
//!
//! Host layers. `--host-layers K` builds the fused runner over the layers after the first `K`,
//! with the plan of `--gpu-layers L-K`: the GPU half of a hybrid. Its hidden entries take zero
//! rows, so the lists equal the GPU commands of that hybrid and the `argmax` lines differ.
//!
//! Pass-through. `--no-list` runs the same steps on the driver functions alone and writes only the
//! `argmax` lines. Equal lines with and without the list show that the wrappers change no result.
//!
//! Output. The command builds the text in memory and writes the file after the last section. A
//! refusal or an error leaves no file. The GPU profile level is off, so no timestamp command
//! records.
//!
//! Errors. A list that is full, unsupported, or missed exits with an error and the name of the
//! fault. `bench-model` checks the same flags before the load.
//!
//! Comptime. The command runs only in a fused build with `-Dtrace=true`. A guard returns on any
//! other build, so the device code is never analyzed there.

const std = @import("std");
const zigferencer = @import("zigferencer");
const build_options = @import("build_options");

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;
const boot_resources = zigferencer.cli.boot_resources;
const core_log = zigferencer.log;
const span = zigferencer.span;
const limits = zigferencer.limits;
const dtype = zigferencer.dtype;
const kv_budget = zigferencer.kv_budget;
const resolve_mod = zigferencer.resolve;
const sampler = zigferencer.sampler;
const parallel = zigferencer.parallel;
const Pool = zigferencer.pool.Pool;
const StepRow = zigferencer.step_row.StepRow;
const dispatch_list = zigferencer.dispatch_list;
const vulkan_ctx_mod = zigferencer.vulkan_ctx;
const hybrid_runner = zigferencer.hybrid_runner;
const common = @import("bench_model/bench_model_common.zig");

const list_build = build_options.use_vulkan_fused_ops and core_log.hot;

// The device paths resolve to `void` on another build, which never analyzes them.
const fused = if (list_build) @import("bench_model/bench_model_fused.zig") else void;
const hybrid_path = if (list_build) @import("bench_model/bench_model_hybrid.zig") else void;

// -- constants --------------------------------------------------------------

/// The prompt of the prefill section. Every vocabulary of the registry holds these ids.
const prompt = [_]u32{ 1, 100, 200, 300, 400, 500, 600 };

const slots: u16 = 2;

/// The long sections prefill slot 0 from `long_start` up to `long_end`, then decode at `long_end`.
/// The span passes `limits.attn_split_len` positions.
const long_start: u32 = 9;
const long_end: u32 = 300;

/// The chunk where the qwen family picks its GEMM kernel. Both families take it.
const long_chunk_rows: u32 = 64;

/// The window of the plan. It holds the last row of `decode_batch_long`.
const window: u32 = 512;

comptime {
    std.debug.assert(long_start == prompt.len + 2);
    std.debug.assert(long_end > limits.attn_split_len);
    std.debug.assert(window > long_end + 1);
}

// -- types ------------------------------------------------------------------

const Flags = struct {
    model: ?[]const u8 = null,
    gpu_layers: ?kv_budget.GpuLayers = null,
    host_layers: ?u32 = null,
    kv_dtype: ?dtype.KvDtype = null,
    out: ?[]const u8 = null,
    no_list: bool = false,

    pub const help: args.Help = .{
        .usage = "zigferencer-dev dispatch-list --model M --out FILE [flags]   " ++
            "(fused trace builds only)",
        .about = "Write the Vulkan commands of six sections, one line per command.\n" ++
            "The sections are a prefill, a decode step, and a batched step at two depths.\n" ++
            "Compare two trees with `diff -u`.\n",
    };
    pub const docs = .{
        .model = args.Doc{ .value = "M", .text = "the registry name of the model" },
        .gpu_layers = args.Doc{
            .value = "N",
            .text = "build the hybrid runner with N GPU layers.\nAbsent builds the fused runner",
        },
        .host_layers = args.Doc{
            .value = "K",
            .text = "build the fused runner over the layers after the first K,\n" ++
                "with zero rows into its hidden entries",
        },
        .kv_dtype = args.Doc{ .text = "KV element type", .default = "the model's registered one" },
        .out = args.Doc{ .value = "FILE", .text = "the list file to write" },
        .no_list = args.Doc{ .text = "run the steps without the list and write only the argmax" },
    };
};

/// Where the steps write, whether the list records, and the host layer count of a suffix.
const Out = struct { w: *std.Io.Writer, record: bool, host_layers: ?u32 };

/// The token entries of `runSteps` over the hidden entries of a suffix runner. Each entry passes
/// zero rows of width `hidden`, and drops its token ids.
fn ZeroRows(comptime Runner: type) type {
    return struct {
        const Self = @This();

        runner: *Runner,
        zeros: []const f32,
        hidden: usize,
        n_slots: u32,
        max_seq: u32,
        prefill_batch_max: u32,
        logits: []f32,

        fn rows(self: *const Self, n: usize) []const f32 {
            std.debug.assert(n >= 1);
            std.debug.assert(n * self.hidden <= self.zeros.len);

            return self.zeros[0 .. n * self.hidden];
        }

        pub fn prefillBatch(self: *const Self, tokens: []const u32, start_pos: u32) ![]f32 {
            const b: u32 = @intCast(tokens.len);
            return self.runner.prefillBatchHidden(self.rows(b), b, start_pos);
        }

        pub fn step(self: *const Self, token: u32, pos: u32) ![]f32 {
            _ = token;
            return self.runner.stepHidden(self.rows(1), pos);
        }

        pub fn stepBatch(self: *const Self, step_rows: []const StepRow) !void {
            return self.runner.stepBatchHidden(self.rows(step_rows.len), step_rows);
        }
    };
}

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;

    if (comptime !list_build) {
        args.writeStderr(
            ctx.io,
            "dispatch-list requires -Druntime=vulkan -Dops-backend=vulkan_fused -Dtrace=true\n",
        );
        return error.RequiresTraceBuild;
    }

    const out_path = flags.out orelse return args.fail(ctx.io, "--out is required", .{});
    if (flags.model == null) return args.fail(ctx.io, "--model is required", .{});

    if (flags.gpu_layers) |layers| {
        const pin = layers.pin() orelse 0;
        if (pin == 0) return args.fail(ctx.io, "--gpu-layers must be a count of 1 or more", .{});
    }

    if (flags.host_layers) |host_layers| {
        if (host_layers == 0) return args.fail(ctx.io, "--host-layers must be 1 or more", .{});
        if (flags.gpu_layers != null) {
            return args.fail(ctx.io, "--host-layers and --gpu-layers exclude each other", .{});
        }
    }

    const opt: common.Options = .{
        .pub_name = flags.model,
        .engine = .{
            .kv_dtype = flags.kv_dtype,
            .gpu_layers = flags.gpu_layers,
            .max_seq = window,
        },
    };
    try common.refuseBadFlags(ctx.io, opt);

    // The list starts before the context loads its device functions, and holds no timestamp.
    span.setGpuLevel(.off);
    const record = !flags.no_list;
    if (record) {
        try dispatch_list.enable(ctx.gpa, .default);
    }
    defer dispatch_list.deinit();

    const input: resolve_mod.Input = .{
        .engine = opt.engine,
        .serve = .{ .max_sessions = slots, .structural_slots = slots, .batch = slots },
    };
    var text: std.Io.Writer.Allocating = .init(ctx.gpa);
    defer text.deinit();

    const out: Out = .{ .w = &text.writer, .record = record, .host_layers = flags.host_layers };
    common.loadAndRun(ctx, opt, .serve, input, listFamily, out) catch |err| {
        return reportFault(ctx.io, err);
    };
    if (record) {
        dispatch_list.finish() catch |err| return reportFault(ctx.io, err);
    }

    try std.Io.Dir.cwd().writeFile(ctx.io, .{ .sub_path = out_path, .data = text.written() });
}

// -- helpers ----------------------------------------------------------------

/// Name the fault of a list error before the command returns the error.
fn reportFault(io: std.Io, err: anyerror) anyerror {
    switch (err) {
        error.DispatchListFull, error.DispatchListMissed, error.DispatchListUnsupported => {
            const reason = if (dispatch_list.faultReason()) |r| @tagName(r) else "table full";
            _ = args.fail(io, "the dispatch list is not complete ({s}: {s})", .{
                @errorName(err), reason,
            }) catch {};
        },
        else => {},
    }

    return err;
}

fn listFamily(comptime Family: type, b: *const common.Bench, out: Out) !void {
    if (out.host_layers) |host_layers| return listSuffix(Family, b, out, host_layers);
    if (b.opt.engine.gpu_layers == null) return listFused(Family, b, out);
    return listHybrid(Family, b, out);
}

/// Build the fused runner over the layers after the first `host_layers`, with the plan that the
/// hybrid of `--gpu-layers` of `n_layers - host_layers` resolves, then run the sections on zero
/// rows.
fn listSuffix(comptime Family: type, b: *const common.Bench, out: Out, host_layers: u32) !void {
    // Refuse a count that leaves no GPU layer.
    const n_layers = b.loaded.card.n_layers;
    if (host_layers >= n_layers) {
        const fmt = "--host-layers {d} leaves no GPU layer of the model's {d} layers";
        return args.fail(b.ctx.io, fmt, .{ host_layers, n_layers });
    }

    // Resolve the boot as the hybrid of the same split resolves it.
    const gpu_layers: kv_budget.GpuLayers = .{ .count = n_layers - host_layers };
    var boot = b.boot.*;
    boot.input.engine.gpu_layers = gpu_layers;
    var bench = b.*;
    bench.opt.engine.gpu_layers = gpu_layers;
    bench.boot = &boot;

    // The context defer runs last, after the runner releases its borrow.
    const ctxp = try b.ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(b.ctx.arena, b.ctx.vk_options);
    defer ctxp.deinit();

    const plan = (try hybrid_path.resolveSplit(&bench, ctxp)).plan.?;
    std.debug.assert(plan.host_layers == host_layers);
    std.debug.assert(plan.n_layers == n_layers);

    const runner = try Family.Fused.initSuffix(
        b.ctx.gpa,
        ctxp,
        b.loaded,
        hybrid_runner.suffixPlan(plan),
        host_layers,
        b.kvDtype(),
    );
    defer runner.deinit();

    // The widest section passes `long_chunk_rows` rows.
    const hidden: usize = b.loaded.card.embedding_length;
    const zeros = try b.ctx.gpa.alloc(f32, long_chunk_rows * hidden);
    defer b.ctx.gpa.free(zeros);
    @memset(zeros, 0);

    const entries: ZeroRows(@TypeOf(runner.*)) = .{
        .runner = runner,
        .zeros = zeros,
        .hidden = hidden,
        .n_slots = runner.n_slots,
        .max_seq = runner.max_seq,
        .prefill_batch_max = runner.prefill_batch_max,
        .logits = runner.logits,
    };
    return runSteps(&entries, out);
}

fn listFused(comptime Family: type, b: *const common.Bench, out: Out) !void {
    const ctxp = try b.ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(b.ctx.arena, b.ctx.vk_options);
    defer ctxp.deinit();

    const f = try fused.place(b, ctxp);
    const runner = try Family.Fused.init(b.ctx.gpa, ctxp, b.loaded, f.plan, b.kvDtype());
    defer runner.deinit();

    return runSteps(runner, out);
}

fn listHybrid(comptime Family: type, b: *const common.Bench, out: Out) !void {
    const pin = b.opt.engine.gpu_layers.?.pin();
    try boot_resources.refusePastModel(b.ctx.io, pin, b.loaded.card.n_layers);

    var pool: Pool = undefined;
    try parallel.initPool(&pool, b.ctx.io, b.policy);
    defer pool.deinit();

    // The context defer runs last, after the runner releases its borrow.
    const ctxp = try b.ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(b.ctx.arena, b.ctx.vk_options);
    defer ctxp.deinit();

    const placement = try hybrid_path.resolveSplit(b, ctxp);
    const runner = try Family.Hybrid.init(
        b.ctx.gpa,
        ctxp,
        b.loaded,
        .{
            .plan = placement.plan.?,
            .pool = &pool,
            .policy = b.policy,
            .kv_dtype = b.kvDtype(),
        },
    );
    defer runner.deinit();

    return runSteps(runner, out);
}

/// Run the six sections that the header names, each with its `argmax` line.
fn runSteps(runner: anytype, out: Out) !void {
    std.debug.assert(runner.n_slots >= slots);
    std.debug.assert(runner.max_seq >= window);

    beginSection(out, "prefill_chunk");
    const prefill = try runner.prefillBatch(&prompt, 0);
    try endSection(out, prefill);

    beginSection(out, "decode");
    const decode = try runner.step(sampler.greedy(prefill), prompt.len);
    const vocab = decode.len;
    try endSection(out, decode);

    beginSection(out, "decode_batch");
    try runner.stepBatch(&batchRows(sampler.greedy(decode), prompt.len + 1, 0));
    try endSection(out, runner.logits[0..vocab]);

    beginSection(out, "prefill_chunk_long");
    const prefill_long = try prefillLong(runner);
    try endSection(out, prefill_long);

    beginSection(out, "decode_long");
    const decode_long = try runner.step(sampler.greedy(prefill_long), long_end);
    try endSection(out, decode_long);

    beginSection(out, "decode_batch_long");
    try runner.stepBatch(&batchRows(sampler.greedy(decode_long), long_end + 1, 2));
    try endSection(out, runner.logits[0..vocab]);
}

/// Prefill slot 0 over `long_start..long_end` in chunks of `long_chunk_rows` rows or fewer. Returns
/// the logits of the last chunk.
fn prefillLong(runner: anytype) ![]f32 {
    const chunk_max = @min(long_chunk_rows, runner.prefill_batch_max);
    std.debug.assert(chunk_max >= 1);

    var tokens: [long_end - long_start]u32 = undefined;
    for (&tokens, 0..) |*t, i| {
        t.* = @intCast(1 + (i * 37) % 997);
    }

    var logits: []f32 = &.{};
    var at: u32 = 0;
    while (at < tokens.len) {
        const n: u32 = @intCast(@min(chunk_max, tokens.len - at));
        logits = try runner.prefillBatch(tokens[at..][0..n], long_start + at);
        at += n;
    }

    std.debug.assert(at == tokens.len);
    return logits;
}

/// One row on slot 0 at `pos`, then two rows on slot 1 from `slot1_pos`. The last row alone wants
/// logits, so they land in the first row of `runner.logits`.
fn batchRows(token: u32, pos: u32, slot1_pos: u32) [3]StepRow {
    return .{
        .{ .slot = 0, .token = token, .pos = pos, .want_logits = false },
        .{ .slot = 1, .token = prompt[0], .pos = slot1_pos, .want_logits = false },
        .{ .slot = 1, .token = prompt[1], .pos = slot1_pos + 1, .want_logits = true },
    };
}

fn beginSection(out: Out, name: []const u8) void {
    if (out.record) {
        dispatch_list.beginSection(name);
    }
}

/// Close the section, write its list, then the argmax of its logits.
fn endSection(out: Out, logits: []const f32) !void {
    std.debug.assert(logits.len > 0);

    if (out.record) {
        dispatch_list.endSection();
        try dispatch_list.write(out.w);
    }

    try out.w.print("argmax {d}\n", .{sampler.greedy(logits)});
}
