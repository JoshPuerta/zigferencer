//! `zigferencer gpu-selftest [gdn|attn|matmul]` bit-checks one GPU kernel against
//! its verified CPU reference on random inputs. The CPU Qwen3.5 forward is
//! bit-exact against llama.cpp, so a new kernel earns trust only when it matches
//! that reference. This command is where each kernel earns it, ahead of the
//! runner. The payoff is localization. A divergence appears here as a numeric diff
//! on one kernel, in place of garbage tokens 24 layers deep in an end-to-end run.
//! The token-hash gate there reports that something is wrong, never which kernel
//! broke.
//!
//! Each sub-command issues one short controlled GPU dispatch. A long-running
//! submit is fragile under a translation-layer driver, so every check stays small.

const std = @import("std");
const build_options = @import("build_options");
const cli = @import("command.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const gdn = @import("../ops/gdn.zig");
const naive = @import("../ops/naive.zig");
const attn_cpu = @import("../core/attn_cpu.zig");
const attn_split = @import("../core/attn_split.zig");
const attn_record = @import("../runtime/attn_record.zig");
const op_requires = @import("../core/op_requires.zig");
const limits = @import("../core/limits.zig");
const Tensor = @import("../core/tensor.zig").Tensor;

// The Vulkan imports resolve to `void` on a CPU build, so this file still
// compiles there. The kernel modules @compileError inside a non-vulkan build.
const vulkan_ctx = if (build_options.use_vulkan) @import("../runtime/vulkan_ctx.zig") else void;
const kernels = if (build_options.use_vulkan) @import("../ops_vulkan/kernels.zig") else void;

// -- types ------------------------------------------------------------------

/// The flags of `gpu-selftest`.
const Flags = struct {
    kernel: ?[]const u8 = null,

    pub const positional = "kernel";
    pub const help: args.Help = .{
        .usage = "zigferencer-dev gpu-selftest [gdn|attn|matmul]   (vulkan builds only)",
        .about =
        \\Bit-check one GPU kernel against its CPU reference on random inputs.
        \\gdn (default) the Gated-DeltaNet step, attn full attention and the
        \\attention kit, matmul the Q8_0 prefill matmuls. Exits non-zero on a
        \\numeric mismatch. Takes no flags.
        \\
        ,
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    // The parse runs first, so --help works on any build. The vulkan gate follows,
    // then the kernel dispatch. The comptime guard returns on a CPU build, so the
    // vulkan-only checks below are never analyzed.
    const flags = try args.parseInto(Flags, ctx) orelse return;
    const which = flags.kernel orelse "gdn";

    if (comptime !build_options.use_vulkan) {
        args.writeStderr(ctx.io, "gpu-selftest requires a Vulkan build (-Druntime=vulkan -Dops-backend=vulkan_fused)\n");
        return error.RequiresVulkan;
    }

    if (std.mem.eql(u8, which, "gdn")) return gdnCheck(ctx);
    if (std.mem.eql(u8, which, "attn")) return attnCheck(ctx);
    if (std.mem.eql(u8, which, "matmul")) return matmulCheck(ctx);
    return args.fail(ctx.io, "unknown kernel '{s}' (expected gdn|attn|matmul)", .{which});
}

// -- helpers ----------------------------------------------------------------

/// Check the decode attention at Qwen3.5 dims (8 q-heads, 2 kv-heads, head_dim
/// 256, GQA 4:1) against the CPU `naive.attentionRowTyped`, at both KV
/// element types. The dims cover a head_dim above 128. Each case prints its
/// maximum relative error. A split case also runs twice and fails unless the
/// two outputs match byte for byte, because the merge order is fixed. It then
/// checks the attention kit of the build with decode and prefill plans.
fn attnCheck(ctx: cli.Context) !void {
    // 7 is the small smoke shape. `split_len` is the deepest row of the
    // resident kernel, and one more is the first split row, with a last split
    // of one position. `cpu_tile` ends the CPU reference on a full tile, and
    // one more adds a tile of one position. The deeper cases walk
    // `t_len / split_len` splits.
    const split_len = limits.attn_split_len;
    const cpu_tile = limits.attn_cpu_tile;
    const t_cases = [_]u32{ 7, split_len, split_len + 1, cpu_tile, cpu_tile + 1, 4096, 8192, 32768 };

    var ctxp = try ctx.arena.create(vulkan_ctx.Context);
    ctxp.* = try vulkan_ctx.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    defer ctxp.deinit();
    var attn_f32 = try kernels.Attention.initKv(ctxp, .f32);
    defer attn_f32.deinit();
    var attn_f16 = try kernels.Attention.initKv(ctxp, .f16);
    defer attn_f16.deinit();

    var buf: [2048]u8 = undefined;
    var fw: std.Io.File.Writer = .initStreaming(.stdout(), ctx.io, &buf);
    const w = &fw.interface;
    var total_fails: usize = 0;
    for (t_cases) |t_len| {
        total_fails += try attnCheckCase(f32, ctx.gpa, &attn_f32, w, t_len);
        total_fails += try attnCheckCase(f16, ctx.gpa, &attn_f16, w, t_len);
    }
    var rig_f32 = try KitRig.init(ctxp, &attn_f32);
    defer rig_f32.deinit();
    var rig_f16 = try KitRig.init(ctxp, &attn_f16);
    defer rig_f16.deinit();
    total_fails += try attnKitCases(f32, ctx.gpa, &rig_f32, w);
    total_fails += try attnKitCases(f16, ctx.gpa, &rig_f16, w);
    try w.print("RESULT: {s}\n", .{if (total_fails == 0) "PASS" else "FAIL"});
    try w.flush();
    if (total_fails != 0) return error.BitCheckFailed;
}

/// One attention case at KV element type `T`. It returns the failure count:
/// the elements outside the tolerance, plus one when a split row gives other
/// bytes on its second run.
fn attnCheckCase(comptime T: type, gpa: std.mem.Allocator, attn: *kernels.Attention, w: *std.Io.Writer, t_len: u32) !usize {
    const n_heads: u32 = 8;
    const n_kv_heads: u32 = 2;
    const head_dim: u32 = 256;
    // Each case frees its buffers, so the deepest case alone sets the peak.
    var case_arena: std.heap.ArenaAllocator = .init(gpa);
    defer case_arena.deinit();
    const a = case_arena.allocator();

    const q = try a.alloc(f32, n_heads * head_dim);
    const k_cache = try a.alloc(T, @as(usize, t_len) * n_kv_heads * head_dim);
    const v_cache = try a.alloc(T, @as(usize, t_len) * n_kv_heads * head_dim);
    var prng = std.Random.DefaultPrng.init(0xA77E27);
    const r = prng.random();
    fillN(r, q, 1.0);
    fillTyped(T, r, k_cache, 1.0);
    fillTyped(T, r, v_cache, 1.0);

    const out_cpu = try a.alloc(f32, n_heads * head_dim);
    const parts = try a.alloc(f32, attn_cpu.partialLen(n_heads, head_dim));
    naive.attentionRowTyped(T, q, n_heads, head_dim, k_cache, v_cache, n_kv_heads, t_len, parts, out_cpu);

    const out_gpu = try a.alloc(f32, n_heads * head_dim);
    try attn.runTyped(T, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, t_len, out_gpu);

    const split = attn_split.usesSplit(t_len);
    try w.print("Attention GPU vs CPU (kv={s}, n_heads={d}, n_kv={d}, head_dim={d}, t_len={d}, {s})\n", .{
        @typeName(T), n_heads, n_kv_heads, head_dim, t_len, if (split) "split" else "resident",
    });
    try printDiff(w, "out", out_cpu, out_gpu);
    var fails = countFail(out_cpu, out_gpu, 1e-5, 1e-3);
    try w.print("  fails (atol=1e-5 rtol=1e-3): {d}\n", .{fails});
    if (split) {
        const out_again = try a.alloc(f32, n_heads * head_dim);
        try attn.runTyped(T, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, t_len, out_again);
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(out_gpu), std.mem.sliceAsBytes(out_again));
        try w.print("  second run: {s}\n", .{if (same) "same bytes" else "OTHER BYTES"});
        if (!same) fails += 1;
    }
    return fails;
}

// -- attention kit ----------------------------------------------------------

const DescriptorBufferInfo = @typeInfo(@typeInfo(@TypeOf(kernels.updateStorageBindings)).@"fn".params[2].type.?).pointer.child;

/// The score that a needle position gets, near the natural log of the deepest
/// row. The needle then holds a large share of the softmax, so a row that
/// reads one position too few or too many moves past the tolerance.
const needle_score: f32 = 9;

/// A row that no dispatch writes keeps this NaN pattern and fails the check.
const sentinel_bits: u32 = 0x7FC0_0BAD;

const KitShape = struct { n_heads: u32, n_kv_heads: u32, head_dim: u32 };

/// The Qwen3.5-2B head geometry, then the SmolLM2 one.
const kit_shapes = [_]KitShape{ .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 256 }, .{ .n_heads = 9, .n_kv_heads = 3, .head_dim = 64 } };

const KitKind = enum { decode, prefill };

/// Query rows `[start_pos, start_pos + b)` of one stream. A decode case has
/// `b` = 1 and `start_pos` = `t_len - 1`.
const KitCase = struct { kind: KitKind, start_pos: u32, b: u32 };

/// The pipelines of the attention kit at one KV element type, as a runner
/// holds them, with the sets and host buffers one case binds. `single` is
/// borrowed from the caller.
const KitRig = struct {
    ctx: *const vulkan_ctx.Context,
    single: *kernels.Attention,
    multi: kernels.AttentionMultiStream,
    passes: kernels.AttentionSplit,
    batch: kernels.AttentionBatch,
    batch_set: kernels.OneShot,
    split_set: kernels.OneShot,
    merge_set: kernels.OneShot,
    q: kernels.HostBuffer,
    k: kernels.HostBuffer,
    v: kernels.HostBuffer,
    scratch: kernels.HostBuffer,
    out: kernels.HostBuffer,

    fn init(ctx: *const vulkan_ctx.Context, single: *kernels.Attention) !KitRig {
        var multi = try kernels.AttentionMultiStream.initKv(ctx, single.kv);
        errdefer multi.deinit();
        var passes = try kernels.AttentionSplit.initKv(ctx, single.kv);
        errdefer passes.deinit();
        var batch = try kernels.AttentionBatch.initKv(ctx, single.kv);
        errdefer batch.deinit();
        const batch_set = try kernels.OneShot.init(ctx, batch.pipeline.set_layout, 4);
        errdefer batch_set.deinit(ctx);
        const split_set = try kernels.OneShot.init(ctx, passes.split.set_layout, kernels.AttentionSplit.split_bindings);
        errdefer split_set.deinit(ctx);
        const merge_set = try kernels.OneShot.init(ctx, passes.merge.set_layout, kernels.AttentionSplit.merge_bindings);
        return .{
            .ctx = ctx,
            .single = single,
            .multi = multi,
            .passes = passes,
            .batch = batch,
            .batch_set = batch_set,
            .split_set = split_set,
            .merge_set = merge_set,
            .q = .{ .ctx = ctx },
            .k = .{ .ctx = ctx },
            .v = .{ .ctx = ctx },
            .scratch = .{ .ctx = ctx },
            .out = .{ .ctx = ctx },
        };
    }

    fn deinit(self: *KitRig) void {
        self.out.deinit();
        self.scratch.deinit();
        self.v.deinit();
        self.k.deinit();
        self.q.deinit();
        self.merge_set.deinit(self.ctx);
        self.split_set.deinit(self.ctx);
        self.batch_set.deinit(self.ctx);
        self.batch.deinit();
        self.passes.deinit();
        self.multi.deinit();
    }
};

/// Check the attention kit of the build against the CPU reference with two
/// kinds of plan: a decode row through `attn_record.record`, and a prefill chunk
/// through `attn_record.recordPrefill`. The decode depths hold the last
/// resident row and two split rows. The prefill chunks straddle `split_len`
/// and sit deep in the split range. Each case fails when its maximum relative
/// error exceeds the tolerance that the kit declares.
fn attnKitCases(comptime T: type, gpa: std.mem.Allocator, rig: *KitRig, w: *std.Io.Writer) !usize {
    const split_len = limits.attn_split_len;
    const decode_t = [_]u32{ split_len - 1, split_len + 1, 4 * split_len + 3 };
    const prefill = [_]KitCase{
        .{ .kind = .prefill, .start_pos = split_len - 3, .b = 7 },
        .{ .kind = .prefill, .start_pos = 4 * split_len + 1, .b = 5 },
    };
    var fails: usize = 0;
    // Two shapes, each with three decode cases and two prefill cases.
    for (kit_shapes) |shape| {
        for (decode_t) |t_len| {
            fails += try attnKitCase(T, gpa, rig, w, shape, .{ .kind = .decode, .start_pos = t_len - 1, .b = 1 });
        }
        for (prefill) |c| fails += try attnKitCase(T, gpa, rig, w, shape, c);
    }
    return fails;
}

/// One kit case at KV element type `T`. It returns 1 when the maximum
/// relative error of any output row exceeds the kit tolerance, else 0.
fn attnKitCase(comptime T: type, gpa: std.mem.Allocator, rig: *KitRig, w: *std.Io.Writer, shape: KitShape, c: KitCase) !usize {
    const n_heads = shape.n_heads;
    const head_dim = shape.head_dim;
    const row_elems: usize = @as(usize, n_heads) * head_dim;
    const kv_row: usize = @as(usize, shape.n_kv_heads) * head_dim;
    // One position past the deepest row holds the second needle key.
    const positions = c.start_pos + c.b + 1;
    std.debug.assert(shape.n_heads % shape.n_kv_heads == 0);
    std.debug.assert(shape.n_heads / shape.n_kv_heads <= limits.attn_max_group);
    std.debug.assert(attn_split.headDimFits(head_dim));
    std.debug.assert(c.b >= 1 and positions <= limits.max_model_ctx);

    var case_arena: std.heap.ArenaAllocator = .init(gpa);
    defer case_arena.deinit();
    const a = case_arena.allocator();

    var prng = std.Random.DefaultPrng.init(0xA77E27 + @as(u64, c.start_pos) * 31 + c.b);
    const r = prng.random();
    const k_f32 = try a.alloc(f32, positions * kv_row);
    fillN(r, k_f32, 1.0);
    // V sits in [0.5, 1.5], so no output cancels toward zero and the
    // relative error stays well conditioned at depth.
    const v_f32 = try a.alloc(f32, positions * kv_row);
    for (v_f32) |*x| x.* = 0.5 + r.float(f32);
    const k = try a.alloc(T, k_f32.len);
    const v = try a.alloc(T, v_f32.len);
    for (k, k_f32) |*d, x| d.* = @floatCast(x);
    for (v, v_f32) |*d, x| d.* = @floatCast(x);
    const q = try a.alloc(f32, c.b * row_elems);
    fillNeedleRows(r, q, k_f32, shape, c);

    const out_cpu = try a.alloc(f32, c.b * row_elems);
    const parts = try a.alloc(f32, attn_cpu.partialLen(n_heads, head_dim));
    // Row i attends to `start_pos + i + 1` positions.
    for (0..c.b) |i| {
        const t_len = c.start_pos + @as(u32, @intCast(i)) + 1;
        naive.attentionRowTyped(T, q[i * row_elems ..][0..row_elems], n_heads, head_dim, k, v, shape.n_kv_heads, t_len, parts, out_cpu[i * row_elems ..][0..row_elems]);
    }

    const scratch = attn_split.Scratch.init(n_heads, head_dim, positions);
    const plan = switch (c.kind) {
        .decode => attn_split.planDecode(c.start_pos + 1),
        .prefill => attn_split.planPrefill(0, c.start_pos, c.b, scratch.max_splits),
    };
    plan.assertValid();
    std.debug.assert(plan.resident_count + plan.split_count == c.b);

    const q_bytes = try uploadSlice(f32, &rig.q, q);
    const kv_bytes = try uploadSlice(T, &rig.k, k);
    _ = try uploadSlice(T, &rig.v, v);
    const out_bytes: u64 = out_cpu.len * @sizeOf(f32);
    try rig.out.ensure(out_bytes);
    @memset(rig.out.sliceTyped(u32, out_cpu.len), sentinel_bits);
    try rig.scratch.ensure(scratch.totalBytes());
    const q_info: DescriptorBufferInfo = .{ .buffer = rig.q.buffer, .offset = 0, .range = q_bytes };
    const k_info: DescriptorBufferInfo = .{ .buffer = rig.k.buffer, .offset = 0, .range = kv_bytes };
    const v_info: DescriptorBufferInfo = .{ .buffer = rig.v.buffer, .offset = 0, .range = kv_bytes };
    const scratch_info: DescriptorBufferInfo = .{ .buffer = rig.scratch.buffer, .offset = 0, .range = scratch.totalBytes() };
    const out_info: DescriptorBufferInfo = .{ .buffer = rig.out.buffer, .offset = 0, .range = out_bytes };
    kernels.AttentionSplit.assertKBinding(k_info);
    kernels.AttentionSplit.assertVBinding(v_info);
    const resident_set = switch (c.kind) {
        .decode => rig.single.oneshot.desc_set,
        .prefill => rig.batch_set.desc_set,
    };
    kernels.updateStorageBindings(rig.ctx, resident_set, &.{ q_info, k_info, v_info, out_info });
    kernels.updateStorageBindings(rig.ctx, rig.split_set.desc_set, &.{ q_info, k_info, v_info, scratch_info });
    kernels.updateStorageBindings(rig.ctx, rig.merge_set.desc_set, &.{ scratch_info, out_info });

    const kit: kernels.AttnKit = .{ .single = rig.single, .multi = &rig.multi, .passes = &rig.passes, .batch = &rig.batch };
    const binding: kernels.AttnKit.Binding = .{
        .shape = .{
            .n_heads = n_heads,
            .n_kv_heads = shape.n_kv_heads,
            .head_dim = head_dim,
            .stream_stride_elems = @intCast(positions * kv_row),
            .max_splits = scratch.max_splits,
        },
        .resident = switch (c.kind) {
            .decode => .{ .single = resident_set },
            .prefill => .{ .batch = resident_set },
        },
        .split = rig.split_set.desc_set,
        .merge = rig.merge_set.desc_set,
    };
    const cb = rig.split_set.cmd_buffer;
    try rig.ctx.vkd.resetCommandBuffer(cb, .{});
    try rig.ctx.vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = true } });
    switch (c.kind) {
        .decode => attn_record.record(kernels.AttnKit, kit, cb, &plan, binding),
        .prefill => attn_record.recordPrefill(kernels.AttnKit, kit, cb, &plan, binding, c.start_pos),
    }
    try rig.ctx.vkd.endCommandBuffer(cb);
    const cmds = [_]@TypeOf(cb){cb};
    try rig.ctx.vkd.queueSubmit(rig.ctx.queue, &.{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }}, .null_handle);
    try rig.ctx.vkd.queueWaitIdle(rig.ctx.queue);

    const out_gpu = rig.out.sliceTyped(f32, out_cpu.len);
    const max_rel = maxRelStrict(out_cpu, out_gpu);
    const ok = op_requires.toleranceSatisfied(kernels.AttnKit.tolerance, max_rel, false);
    const verdict = if (ok) "ok" else "FAIL";
    switch (c.kind) {
        .decode => try w.print("Attention kit (kv={s}, shape={d}/{d}/{d}, decode, t_len={d}, {s}): max_rel={e:.3} {s}\n", .{
            @typeName(T),                                      n_heads, shape.n_kv_heads, head_dim, c.start_pos + 1,
            if (plan.split_count > 0) "split" else "resident", max_rel, verdict,
        }),
        .prefill => try w.print("Attention kit (kv={s}, shape={d}/{d}/{d}, prefill, start_pos={d}, b={d}, {d} resident + {d} split): max_rel={e:.3} {s}\n", .{
            @typeName(T),        n_heads,          shape.n_kv_heads, head_dim, c.start_pos, c.b,
            plan.resident_count, plan.split_count, max_rel,          verdict,
        }),
    }
    return @intFromBool(!ok);
}

/// Give each query row a needle: a score near `needle_score` at its last
/// causal position and at the position after it, for every head.
fn fillNeedleRows(r: std.Random, q: []f32, k: []const f32, shape: KitShape, c: KitCase) void {
    const head_dim: usize = shape.head_dim;
    const row_elems = @as(usize, shape.n_heads) * head_dim;
    const kv_row = @as(usize, shape.n_kv_heads) * head_dim;
    const group = shape.n_heads / shape.n_kv_heads;
    const scale = 3 * needle_score / @sqrt(@as(f32, @floatFromInt(head_dim)));
    std.debug.assert(q.len == c.b * row_elems);
    for (0..c.b) |i| {
        const p = c.start_pos + i;
        for (0..shape.n_heads) |h| {
            const kv_h = h / group;
            const at_p = k[p * kv_row + kv_h * head_dim ..][0..head_dim];
            const at_next = k[(p + 1) * kv_row + kv_h * head_dim ..][0..head_dim];
            const dst = q[i * row_elems + h * head_dim ..][0..head_dim];
            for (dst, at_p, at_next) |*x, a, n| x.* = scale * (a + n + 0.25 * (r.float(f32) - 0.5) * 2.0);
        }
    }
}

/// Copy `src` into `buf` and return the bytes that a binding covers. The f16
/// shaders read the caches as 32-bit words, so the binding covers whole words.
fn uploadSlice(comptime T: type, buf: *kernels.HostBuffer, src: []const T) !u64 {
    std.debug.assert(src.len >= 1);
    const bytes = std.mem.alignForward(u64, src.len * @sizeOf(T), 4);
    try buf.ensure(bytes);
    @memcpy(buf.sliceTyped(T, src.len), src);
    return bytes;
}

/// `maxRel` that reports a NaN element, such as an unwritten row, as infinity.
fn maxRelStrict(cpu: []const f32, gpu: []const f32) f32 {
    var mr: f32 = 0;
    for (cpu, gpu) |c, g| {
        const rel = @abs(c - g) / (@abs(c) + 1e-6);
        if (!(rel <= mr)) mr = if (std.math.isNan(rel)) std.math.inf(f32) else rel;
    }
    return mr;
}

/// Check the batched Q8_0 matmuls (the prefill kernels) against the CPU
/// `naive.linearF32Q80` reference. The shapes cover the main tile and both tails.
/// A batch tail holds a b that is not a multiple of batch_tile. An output tail
/// holds an n that is not a multiple of output_tile. The shapes also cover the
/// degenerate b=1 and a prefill-shaped chunk. The batch kernel reorders the
/// reduction against the CPU, so it carries a `.relative` tolerance. This gate is
/// therefore a relative-tolerance gate, never a bit-exact one. It guards every
/// change to linear_q8_0_batch.comp.
fn matmulCheck(ctx: cli.Context) !void {
    var ctxp = try ctx.arena.create(vulkan_ctx.Context);
    ctxp.* = try vulkan_ctx.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    defer ctxp.deinit();
    var k_batch = try kernels.LinearQ80Batch.init(ctxp);
    defer k_batch.deinit();
    var k_gemm = try kernels.LinearQ80Gemm.init(ctxp);
    defer k_gemm.deinit();

    // The shapes cover aligned tiles and the batch and output tails. The tile
    // sizes are 8 by 16 for batch and 64 by 64 for gemm. The rest cover b=1, one
    // exact gemm tile, and a prefill-shaped chunk. Both kernels run every shape.
    const shapes = [_][3]u32{
        .{ 8, 256, 64 },
        .{ 5, 64, 40 }, // b & n tails
        .{ 13, 128, 96 }, // odd b
        .{ 1, 64, 32 }, // degenerate b=1
        .{ 64, 64, 64 }, // exact one gemm tile
        .{ 70, 128, 200 }, // multi-tile with b & n tails
        .{ 128, 512, 128 }, // prefill-shaped
        // The Qwen3.5-2B projections: the widths 64 and 128 take the tiled kernel.
        .{ 64, 2048, 2048 },
        .{ 128, 2048, 6144 },
        .{ 128, 6144, 2048 },
        .{ 7, 2048, 2048 }, // a width below the tiled kernel
    };
    var all_ok = true;
    for (shapes) |s| {
        if (!try matmulCheckDims(ctx, "Batch", &k_batch, s[0], s[1], s[2])) all_ok = false;
        if (!try matmulCheckDims(ctx, "Gemm ", &k_gemm, s[0], s[1], s[2])) all_ok = false;
    }
    if (!all_ok) return error.BitCheckFailed;
}

fn matmulCheckDims(ctx: cli.Context, tag: []const u8, kernel: anytype, b: u32, k: u32, n: u32) !bool {
    std.debug.assert(k % 32 == 0);
    const a = ctx.arena;
    const blocks_per_row = k / 32;
    const row_bytes = blocks_per_row * 34;
    const weight = try a.alloc(u8, n * row_bytes);
    const input = try a.alloc(f32, @as(usize, b) * k);

    var prng = std.Random.DefaultPrng.init(0x6EA77 + @as(u64, b) * 1_000 + n);
    const r = prng.random();
    buildRandomQ80(r, weight, k, n);
    fillN(r, input, 1.0);

    // The weight Tensor carries the GGUF reversed shape {k, n}, which is
    // in_features then out_features. `linearF32Q80` derives row_bytes itself, and
    // the byte_stride field still states the true stride.
    const w_tensor: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ k, n, 0, 0 },
        .byte_stride = .{ row_bytes, 0, 0, 0 },
        .data = weight,
    };
    const out_cpu = try a.alloc(f32, @as(usize, b) * n);
    var bi: usize = 0;
    while (bi < b) : (bi += 1) {
        naive.linearF32Q80(input[bi * k ..][0..k], w_tensor, out_cpu[bi * n ..][0..n]);
    }

    const out_gpu = try a.alloc(f32, @as(usize, b) * n);
    try kernel.runOneShot(input, weight, b, k, n, out_gpu);

    var buf: [2048]u8 = undefined;
    var fw: std.Io.File.Writer = .initStreaming(.stdout(), ctx.io, &buf);
    const w = &fw.interface;
    try w.print("{s} GPU vs CPU (b={d}, k={d}, n={d}):\n", .{ tag, b, k, n });
    try printDiff(w, "out", out_cpu, out_gpu);
    const atol: f32 = 1e-4;
    const rtol: f32 = 2e-3;
    const fails = countFail(out_cpu, out_gpu, atol, rtol);
    const ok = fails == 0;
    try w.print("  fails (atol={e:.0} rtol={e:.0})={d}  RESULT: {s}\n\n", .{ atol, rtol, fails, if (ok) "PASS" else "FAIL" });
    try w.flush();
    return ok;
}

/// Build a synthetic Q8_0 weight in GGUF layout: `n` rows of out_features, each of
/// `k/32` blocks of 34 bytes. A block holds a 2-byte f16 scale and 32 i8 quants.
/// Random per-block scales in a small range, with full-range i8 quants, give a
/// realistic numeric spread. The bit-exact unit test uses an all-ones scale.
fn buildRandomQ80(r: std.Random, data: []u8, k: u32, n: u32) void {
    const blocks_per_row = k / 32;
    const row_bytes = blocks_per_row * 34;
    std.debug.assert(data.len == @as(usize, n) * row_bytes);
    var j: usize = 0;
    while (j < n) : (j += 1) {
        const row_off = j * row_bytes;
        var blk: usize = 0;
        while (blk < blocks_per_row) : (blk += 1) {
            const blk_off = row_off + blk * 34;
            const scale: f16 = @floatCast(0.005 + 0.02 * r.float(f32));
            std.mem.writeInt(u16, data[blk_off..][0..2], @bitCast(scale), .little);
            var i: usize = 0;
            while (i < 32) : (i += 1) {
                const v: i8 = @intCast(@as(i32, @intFromFloat(@round((r.float(f32) - 0.5) * 250.0))));
                data[blk_off + 2 + i] = @bitCast(v);
            }
        }
    }
}

/// Check the GDN GPU kernel against the CPU `gdn.step` for both head configs,
/// on one Context and one kernel. The configs are the 2B one (n_k = n_v = 16) and
/// the decoupled 9B one (n_k = 16, n_v = 32).
fn gdnCheck(ctx: cli.Context) !void {
    var ctxp = try ctx.arena.create(vulkan_ctx.Context);
    ctxp.* = try vulkan_ctx.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    defer ctxp.deinit();
    var k = try kernels.GdnDecode.init(ctxp);
    defer k.deinit();
    var kp = try kernels.GdnChunkParallel.init(ctxp);
    defer kp.deinit();
    var all_ok = true;
    if (!try gdnCheckDims(ctx, &k, 16, 16)) all_ok = false; // 2B  decode (1 token)
    if (!try gdnCheckDims(ctx, &k, 16, 32)) all_ok = false; // 9B  decode (1 token)
    // On the sequential chunk path for prefill, one dispatch loops T tokens. The
    // compare against the CPU sequence shows that recordChunk carries the state
    // and the conv_state.
    if (!try gdnChunkCheckDims(ctx, &k, 16, 16, 7)) all_ok = false; // 2B  chunk
    if (!try gdnChunkCheckDims(ctx, &k, 16, 32, 7)) all_ok = false; // 9B  chunk
    // The chunked-parallel prefill (a conv, a scan, and a norm dispatch) runs the
    // same compare against T CPU steps. T spans the sub-chunk size (CSUB=16), so
    // the cases cover several sub-chunks and the state carried across them.
    if (!try gdnChunkParallelCheckDims(ctx, &kp, 16, 16, 7)) all_ok = false;
    if (!try gdnChunkParallelCheckDims(ctx, &kp, 16, 32, 7)) all_ok = false;
    if (!try gdnChunkParallelCheckDims(ctx, &kp, 16, 16, 33)) all_ok = false;
    if (!try gdnChunkParallelCheckDims(ctx, &kp, 16, 32, 128)) all_ok = false; // 9B prefill-shaped
    // The runner records the 2B chunk at the widths 64 and 128. The width 100
    // leaves a tail of four tokens in the last sub-chunk.
    if (!try gdnChunkParallelCheckDims(ctx, &kp, 16, 16, 64)) all_ok = false;
    if (!try gdnChunkParallelCheckDims(ctx, &kp, 16, 16, 100)) all_ok = false;
    if (!try gdnChunkParallelCheckDims(ctx, &kp, 16, 16, 128)) all_ok = false;
    if (!all_ok) return error.BitCheckFailed;
}

/// Check the chunked-parallel prefill against the CPU step. Feed T random tokens
/// through the GPU chunked passes (GdnChunkParallel). Feed the same tokens through T
/// CPU `gdn.step` calls that carry the same state and conv_state. Then diff all T
/// readout rows, the final state, and the final conv_state. This is the parallel
/// counterpart of `gdnChunkCheckDims`.
fn gdnChunkParallelCheckDims(ctx: cli.Context, kp: *kernels.GdnChunkParallel, n_k: u32, n_v: u32, n_tokens: u32) !bool {
    const head_dim: u32 = 128;
    const conv_kernel: u32 = 4;
    const eps: f32 = 1e-6;
    const T: usize = n_tokens;
    const key_dim: usize = @as(usize, n_k) * head_dim;
    const value_dim: usize = @as(usize, n_v) * head_dim;
    const conv_dim: usize = 2 * key_dim + value_dim;

    const a = ctx.arena;
    const qkv = try a.alloc(f32, T * conv_dim);
    const z = try a.alloc(f32, T * value_dim);
    const alpha = try a.alloc(f32, T * n_v);
    const beta_in = try a.alloc(f32, T * n_v);
    const conv_w = try a.alloc(f32, @as(usize, conv_kernel) * conv_dim);
    const ssm_a = try a.alloc(f32, n_v);
    const dt_bias = try a.alloc(f32, n_v);
    const ssm_norm = try a.alloc(f32, head_dim);
    const state0 = try a.alloc(f32, value_dim * head_dim);
    const conv0 = try a.alloc(f32, @as(usize, conv_kernel - 1) * conv_dim);

    var prng = std.Random.DefaultPrng.init(0x9A5EED + @as(u64, n_v) * 31 + n_tokens);
    const r = prng.random();
    fillN(r, qkv, 1.0);
    fillN(r, z, 1.0);
    // Wide alpha (+/-3) gives softplus up to ~3, then g down to ~-8, then decay
    // ~1e-4. The cumulative decay over a sub-chunk underflows in product space.
    // The log-space scan must stay accurate in this regime. The sequential CPU
    // step is safe, because it applies S *= decay one token at a time.
    fillN(r, alpha, 3.0);
    fillN(r, beta_in, 1.0);
    fillN(r, conv_w, 0.5);
    fillN(r, dt_bias, 0.3);
    fillN(r, ssm_norm, 0.2);
    for (ssm_norm) |*v| v.* += 1.0;
    fillN(r, state0, 0.1);
    fillN(r, conv0, 0.3);
    for (ssm_a) |*v| v.* = -@exp((r.float(f32) - 0.5) * 2.0);

    // `gdn.step` mutates its qkv row, so every token gets a copy.
    const o_cpu = try a.alloc(f32, T * value_dim);
    const state_cpu = try a.dupe(f32, state0);
    const conv_cpu = try a.dupe(f32, conv0);
    const dims: gdn.Dims = .{ .n_k_heads = n_k, .n_v_heads = n_v, .head_dim = head_dim, .conv_kernel = conv_kernel };
    var ti: usize = 0;
    while (ti < T) : (ti += 1) {
        const qkv_t = try a.dupe(f32, qkv[ti * conv_dim ..][0..conv_dim]);
        gdn.step(dims, qkv_t, z[ti * value_dim ..][0..value_dim], alpha[ti * n_v ..][0..n_v], beta_in[ti * n_v ..][0..n_v], conv_w, ssm_a, dt_bias, ssm_norm, eps, state_cpu, conv_cpu, o_cpu[ti * value_dim ..][0..value_dim]);
    }

    const o_gpu = try a.alloc(f32, T * value_dim);
    const state_gpu = try a.dupe(f32, state0);
    const conv_gpu = try a.dupe(f32, conv0);
    try kp.runChunkParallel(n_tokens, n_k, n_v, head_dim, conv_kernel, qkv, z, alpha, beta_in, conv_w, ssm_a, dt_bias, ssm_norm, eps, state_gpu, conv_gpu, o_gpu);

    var buf: [2048]u8 = undefined;
    var fw: std.Io.File.Writer = .initStreaming(.stdout(), ctx.io, &buf);
    const w = &fw.interface;
    try w.print("GDN chunk-PARALLEL GPU vs CPU (n_k={d}, n_v={d}, T={d}):\n", .{ n_k, n_v, n_tokens });
    try printDiff(w, "o_out     ", o_cpu, o_gpu);
    try printDiff(w, "state     ", state_cpu, state_gpu);
    try printDiff(w, "conv_state", conv_cpu, conv_gpu);
    const atol: f32 = 1e-4;
    const rtol: f32 = 2e-3;
    const fails = countFail(o_cpu, o_gpu, atol, rtol) + countFail(state_cpu, state_gpu, atol, rtol) + countFail(conv_cpu, conv_gpu, atol, rtol);
    const ok = fails == 0;
    try w.print("  fails={d}  RESULT: {s}\n\n", .{ fails, if (ok) "PASS" else "FAIL" });
    try w.flush();
    return ok;
}

/// Check the sequential chunk path against the CPU step. Feed T random tokens
/// through the GPU `runChunk`, which is one dispatch with an internal token
/// loop. Feed the same tokens through
/// T CPU `gdn.step` calls that carry the same state and conv_state. Then diff all
/// T readout rows, the final state, and the final conv_state. This is the prefill
/// counterpart of `gdnCheckDims`.
fn gdnChunkCheckDims(ctx: cli.Context, k: *kernels.GdnDecode, n_k: u32, n_v: u32, n_tokens: u32) !bool {
    const head_dim: u32 = 128;
    const conv_kernel: u32 = 4;
    const eps: f32 = 1e-6;
    const T: usize = n_tokens;
    const key_dim: usize = @as(usize, n_k) * head_dim;
    const value_dim: usize = @as(usize, n_v) * head_dim;
    const conv_dim: usize = 2 * key_dim + value_dim;

    const a = ctx.arena;
    const qkv = try a.alloc(f32, T * conv_dim);
    const z = try a.alloc(f32, T * value_dim);
    const alpha = try a.alloc(f32, T * n_v);
    const beta_in = try a.alloc(f32, T * n_v);
    const conv_w = try a.alloc(f32, @as(usize, conv_kernel) * conv_dim);
    const ssm_a = try a.alloc(f32, n_v);
    const dt_bias = try a.alloc(f32, n_v);
    const ssm_norm = try a.alloc(f32, head_dim);
    const state0 = try a.alloc(f32, value_dim * head_dim);
    const conv0 = try a.alloc(f32, @as(usize, conv_kernel - 1) * conv_dim);

    var prng = std.Random.DefaultPrng.init(0x5EED + n_v);
    const r = prng.random();
    fillN(r, qkv, 1.0);
    fillN(r, z, 1.0);
    fillN(r, alpha, 0.5);
    fillN(r, beta_in, 1.0);
    fillN(r, conv_w, 0.5);
    fillN(r, dt_bias, 0.3);
    fillN(r, ssm_norm, 0.2);
    for (ssm_norm) |*v| v.* += 1.0;
    fillN(r, state0, 0.1);
    fillN(r, conv0, 0.3);
    for (ssm_a) |*v| v.* = -@exp((r.float(f32) - 0.5) * 2.0);

    // `gdn.step` mutates its qkv row, so every token gets a copy.
    const o_cpu = try a.alloc(f32, T * value_dim);
    const state_cpu = try a.dupe(f32, state0);
    const conv_cpu = try a.dupe(f32, conv0);
    const dims: gdn.Dims = .{ .n_k_heads = n_k, .n_v_heads = n_v, .head_dim = head_dim, .conv_kernel = conv_kernel };
    var ti: usize = 0;
    while (ti < T) : (ti += 1) {
        const qkv_t = try a.dupe(f32, qkv[ti * conv_dim ..][0..conv_dim]);
        gdn.step(dims, qkv_t, z[ti * value_dim ..][0..value_dim], alpha[ti * n_v ..][0..n_v], beta_in[ti * n_v ..][0..n_v], conv_w, ssm_a, dt_bias, ssm_norm, eps, state_cpu, conv_cpu, o_cpu[ti * value_dim ..][0..value_dim]);
    }

    const o_gpu = try a.alloc(f32, T * value_dim);
    const state_gpu = try a.dupe(f32, state0);
    const conv_gpu = try a.dupe(f32, conv0);
    try k.runChunk(n_tokens, n_k, n_v, head_dim, conv_kernel, qkv, z, alpha, beta_in, conv_w, ssm_a, dt_bias, ssm_norm, eps, state_gpu, conv_gpu, o_gpu);

    var buf: [2048]u8 = undefined;
    var fw: std.Io.File.Writer = .initStreaming(.stdout(), ctx.io, &buf);
    const w = &fw.interface;
    try w.print("GDN chunk GPU vs CPU (n_k={d}, n_v={d}, T={d}):\n", .{ n_k, n_v, n_tokens });
    try printDiff(w, "o_out     ", o_cpu, o_gpu);
    try printDiff(w, "state     ", state_cpu, state_gpu);
    try printDiff(w, "conv_state", conv_cpu, conv_gpu);
    const atol: f32 = 1e-5;
    const rtol: f32 = 1e-3;
    const fails = countFail(o_cpu, o_gpu, atol, rtol) + countFail(state_cpu, state_gpu, atol, rtol) + countFail(conv_cpu, conv_gpu, atol, rtol);
    const ok = fails == 0;
    try w.print("  fails={d}  RESULT: {s}\n\n", .{ fails, if (ok) "PASS" else "FAIL" });
    try w.flush();
    return ok;
}

fn gdnCheckDims(ctx: cli.Context, k: *kernels.GdnDecode, n_k: u32, n_v: u32) !bool {
    const head_dim: u32 = 128;
    const conv_kernel: u32 = 4;
    const eps: f32 = 1e-6;
    const key_dim: usize = @as(usize, n_k) * head_dim;
    const value_dim: usize = @as(usize, n_v) * head_dim;
    const conv_dim: usize = 2 * key_dim + value_dim;

    const a = ctx.arena;
    const qkv = try a.alloc(f32, conv_dim);
    const z = try a.alloc(f32, value_dim);
    const alpha = try a.alloc(f32, n_v);
    const beta_in = try a.alloc(f32, n_v);
    const conv_w = try a.alloc(f32, @as(usize, conv_kernel) * conv_dim);
    const ssm_a = try a.alloc(f32, n_v);
    const dt_bias = try a.alloc(f32, n_v);
    const ssm_norm = try a.alloc(f32, head_dim);
    const state0 = try a.alloc(f32, value_dim * head_dim); // n_v*D*D
    const conv0 = try a.alloc(f32, @as(usize, conv_kernel - 1) * conv_dim);

    var prng = std.Random.DefaultPrng.init(0xC0FFEE + n_v);
    const r = prng.random();
    fillN(r, qkv, 1.0);
    fillN(r, z, 1.0);
    fillN(r, alpha, 0.5);
    fillN(r, beta_in, 1.0);
    fillN(r, conv_w, 0.5);
    fillN(r, dt_bias, 0.3);
    fillN(r, ssm_norm, 0.2);
    for (ssm_norm) |*v| v.* += 1.0;
    fillN(r, state0, 0.1);
    fillN(r, conv0, 0.3);
    for (ssm_a) |*v| v.* = -@exp((r.float(f32) - 0.5) * 2.0); // -exp(A_log)

    // `gdn.step` mutates its qkv row, so the CPU side gets a copy.
    const qkv_cpu = try a.dupe(f32, qkv);
    const state_cpu = try a.dupe(f32, state0);
    const conv_cpu = try a.dupe(f32, conv0);
    const o_cpu = try a.alloc(f32, value_dim);
    const dims: gdn.Dims = .{ .n_k_heads = n_k, .n_v_heads = n_v, .head_dim = head_dim, .conv_kernel = conv_kernel };
    gdn.step(dims, qkv_cpu, z, alpha, beta_in, conv_w, ssm_a, dt_bias, ssm_norm, eps, state_cpu, conv_cpu, o_cpu);

    // The GPU side reads the raw qkv. Its state and conv start from the same
    // copies, so the diff below compares equal starting points.
    const state_gpu = try a.dupe(f32, state0);
    const conv_gpu = try a.dupe(f32, conv0);
    const o_gpu = try a.alloc(f32, value_dim);
    try k.run(n_k, n_v, head_dim, conv_kernel, qkv, z, alpha, beta_in, conv_w, ssm_a, dt_bias, ssm_norm, eps, state_gpu, conv_gpu, o_gpu);

    var buf: [2048]u8 = undefined;
    var fw: std.Io.File.Writer = .initStreaming(.stdout(), ctx.io, &buf);
    const w = &fw.interface;
    try w.print("GDN GPU vs CPU (n_k={d}, n_v={d}, head_dim={d}):\n", .{ n_k, n_v, head_dim });
    try printDiff(w, "o_out     ", o_cpu, o_gpu);
    try printDiff(w, "state     ", state_cpu, state_gpu);
    try printDiff(w, "conv_state", conv_cpu, conv_gpu);
    const atol: f32 = 1e-5;
    const rtol: f32 = 1e-3;
    const fails = countFail(o_cpu, o_gpu, atol, rtol) + countFail(state_cpu, state_gpu, atol, rtol) + countFail(conv_cpu, conv_gpu, atol, rtol);
    const ok = fails == 0;
    try w.print("  fails={d}  RESULT: {s}\n\n", .{ fails, if (ok) "PASS" else "FAIL" });
    try w.flush();
    return ok;
}

/// Count elements violating numpy-style allclose: |a-b| > atol + rtol*|a|.
fn countFail(cpu: []const f32, gpu: []const f32, atol: f32, rtol: f32) usize {
    var n: usize = 0;
    for (cpu, gpu) |c, g| {
        if (@abs(c - g) > atol + rtol * @abs(c)) n += 1;
    }
    return n;
}

fn fillN(r: std.Random, slice: []f32, scale: f32) void {
    for (slice) |*v| v.* = (r.float(f32) - 0.5) * 2.0 * scale;
}

/// `fillN` at the KV element type: each value rounds once to `T`.
fn fillTyped(comptime T: type, r: std.Random, slice: []T, scale: f32) void {
    for (slice) |*v| v.* = @floatCast((r.float(f32) - 0.5) * 2.0 * scale);
}

/// Max abs diff and max relative diff (rel = |a-b| / (|a| + tiny)).
fn maxRel(cpu: []const f32, gpu: []const f32) f32 {
    var mr: f32 = 0;
    for (cpu, gpu) |c, g| {
        const d = @abs(c - g);
        const rel = d / (@abs(c) + 1e-6);
        if (rel > mr) mr = rel;
    }
    return mr;
}

fn printDiff(w: *std.Io.Writer, name: []const u8, cpu: []const f32, gpu: []const f32) !void {
    var maxabs: f32 = 0;
    for (cpu, gpu) |c, g| {
        const d = @abs(c - g);
        if (d > maxabs) maxabs = d;
    }
    try w.print("  {s}: max_abs={e:.3}  max_rel={e:.3}  (n={d})\n", .{ name, maxabs, maxRel(cpu, gpu), cpu.len });
}
