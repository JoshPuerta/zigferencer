//! The attention checks of `gpu-selftest`: the single-pass kernels and the split kernel set,
//! against the CPU attention reference.
//!
//! Comptime. Generic over the element type, so the f32 and f16 checks share one body.

const std = @import("std");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");

const zigferencer = @import("zigferencer");

const context = zigferencer.cli.context;
const naive = zigferencer.ops;
const attn_cpu = zigferencer.attn_cpu;
const attn_split = zigferencer.attn_split;
const attn_record = zigferencer.attn_record;
const op_requires = zigferencer.op_requires;
const limits = zigferencer.limits;
// These imports resolve to `void` on a CPU build, so the file still compiles there.
const vulkan_ctx = if (build_options.use_vulkan) zigferencer.vulkan_ctx else void;
const kernels = if (build_options.use_vulkan) zigferencer.ops_vulkan else void;

const gpu_selftest_common = @import("gpu_selftest_common.zig");

// -- constants --------------------------------------------------------------

/// The score that a needle position gets, near the natural log of the deepest
/// row. The needle then holds a large share of the softmax, so a row that
/// reads one position too few or too many moves past the tolerance.
const needle_score: f32 = 9;

/// A row that no dispatch writes keeps this NaN pattern and fails the check.
const sentinel_bits: u32 = 0x7FC0_0BAD;

/// The Qwen3.5-2B head geometry, then the SmolLM2 one.
const case_shapes = [_]CaseShape{
    .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 256 },
    .{ .n_heads = 9, .n_kv_heads = 3, .head_dim = 64 },
};

// -- types ------------------------------------------------------------------

const DescriptorBufferInfo = @typeInfo(
    @typeInfo(@TypeOf(kernels.updateStorageBindings)).@"fn".params[2].type.?,
).pointer.child;

const CaseShape = struct { n_heads: u32, n_kv_heads: u32, head_dim: u32 };

const CaseKind = enum { decode, prefill };

/// Query rows `[start_pos, start_pos + b)` of one stream. A decode case has
/// `b` = 1 and `start_pos` = `t_len - 1`.
const Case = struct {
    kind: CaseKind,
    start_pos: u32,
    b: u32,

    /// One position past the deepest row holds the second needle key.
    fn positions(self: Case) u32 {
        return self.start_pos + self.b + 1;
    }
};

/// The host inputs of one case and the CPU reference output.
fn CaseInputs(comptime T: type) type {
    return struct { q: []f32, k: []T, v: []T, out_cpu: []f32 };
}

/// The pipelines of the attention kernel set at one KV element type, as a runner
/// holds them, with the sets and host buffers one case binds. `single` is
/// borrowed from the caller.
const Rig = struct {
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

    fn init(ctx: *const vulkan_ctx.Context, single: *kernels.Attention) !Rig {
        // Build the kernels.
        var multi = try kernels.AttentionMultiStream.initKv(ctx, single.kv);
        errdefer multi.deinit();

        var passes = try kernels.AttentionSplit.initKv(ctx, single.kv);
        errdefer passes.deinit();

        var batch = try kernels.AttentionBatch.initKv(ctx, single.kv);
        errdefer batch.deinit();

        // Build the one-shot descriptor sets.
        const batch_set = try kernels.OneShot.init(ctx, batch.pipeline.set_layout, 4);
        errdefer batch_set.deinit(ctx);

        const split_set = try kernels.OneShot.init(
            ctx,
            passes.split.set_layout,
            kernels.AttentionSplit.split_bindings,
        );
        errdefer split_set.deinit(ctx);

        const merge_set = try kernels.OneShot.init(
            ctx,
            passes.merge.set_layout,
            kernels.AttentionSplit.merge_bindings,
        );

        // Assemble the rig.
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

    fn deinit(self: *Rig) void {
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

// -- api --------------------------------------------------------------------

/// Check decode attention at Qwen3.5 dims (8 q-heads, 2 kv-heads, head_dim 256) at both KV types.
/// A split case runs twice and fails unless both outputs match byte for byte, because the merge
/// order is fixed. Then check the attention kernel set of the build.
pub fn attnCheck(ctx: context.Context) !void {
    // Depths: a small smoke shape, the resident limit and the first split row, then the CPU tile
    // edge and one past it.
    const split_len = limits.attn_split_len;
    const cpu_tile = limits.attn_cpu_tile;
    const t_cases = [_]u32{
        7,
        split_len,
        split_len + 1,
        cpu_tile,
        cpu_tile + 1,
        4096,
        8192,
        32768,
    };

    // Open the device and the two single-pass kernels.
    var ctxp = try ctx.arena.create(vulkan_ctx.Context);
    ctxp.* = try vulkan_ctx.Context.init(ctx.arena, ctx.vk_options);
    defer ctxp.deinit();

    var attn_f32 = try kernels.Attention.initKv(ctxp, .f32);
    defer attn_f32.deinit();

    var attn_f16 = try kernels.Attention.initKv(ctxp, .f16);
    defer attn_f16.deinit();

    // Check the single-pass kernels at every depth.
    const w = &ctx.stdout.interface;

    var total_fails: usize = 0;
    for (t_cases) |t_len| {
        total_fails += try attnCheckCase(f32, ctx.gpa, &attn_f32, w, t_len);
        total_fails += try attnCheckCase(f16, ctx.gpa, &attn_f16, w, t_len);
    }

    // Check the attention kernel set.
    var rig_f32 = try Rig.init(ctxp, &attn_f32);
    defer rig_f32.deinit();

    var rig_f16 = try Rig.init(ctxp, &attn_f16);
    defer rig_f16.deinit();

    total_fails += try attnCases(f32, ctx.gpa, &rig_f32, w);
    total_fails += try attnCases(f16, ctx.gpa, &rig_f16, w);
    try w.print("RESULT: {s}\n", .{if (total_fails == 0) "PASS" else "FAIL"});
    try w.flush();
    if (total_fails != 0) return error.BitCheckFailed;
}

// -- helpers ----------------------------------------------------------------

/// One attention case at KV element type `T`. It returns the failure count:
/// the elements outside the tolerance, plus one when a split row gives other
/// bytes on its second run.
fn attnCheckCase(
    comptime T: type,
    gpa: Allocator,
    attn: *kernels.Attention,
    w: *std.Io.Writer,
    t_len: u32,
) !usize {
    const n_heads: u32 = 8;
    const n_kv_heads: u32 = 2;
    const head_dim: u32 = 256;

    // Build the inputs. Each case frees its buffers, so the deepest case alone sets the peak.
    var case_arena: std.heap.ArenaAllocator = .init(gpa);
    defer case_arena.deinit();

    const a = case_arena.allocator();

    const q = try a.alloc(f32, n_heads * head_dim);
    const k_cache = try a.alloc(T, @as(usize, t_len) * n_kv_heads * head_dim);
    const v_cache = try a.alloc(T, @as(usize, t_len) * n_kv_heads * head_dim);
    var prng = std.Random.DefaultPrng.init(0xA77E27);
    const r = prng.random();
    gpu_selftest_common.fillN(r, q, 1.0);
    gpu_selftest_common.fillTyped(T, r, k_cache, 1.0);
    gpu_selftest_common.fillTyped(T, r, v_cache, 1.0);

    // Run the CPU reference and the GPU kernel.
    const out_cpu = try a.alloc(f32, n_heads * head_dim);
    const parts = try a.alloc(f32, attn_cpu.partialLen(n_heads, head_dim));
    naive.attentionRowTyped(
        T,
        q,
        n_heads,
        head_dim,
        k_cache,
        v_cache,
        n_kv_heads,
        t_len,
        parts,
        out_cpu,
    );

    const out_gpu = try a.alloc(f32, n_heads * head_dim);
    try attn.runTyped(T, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, t_len, out_gpu);

    // Compare, and rerun a split case to check the bytes.
    const split = attn_split.usesSplit(t_len);
    try w.print(
        "Attention GPU vs CPU (kv={s}, n_heads={d}, n_kv={d}, head_dim={d}, t_len={d}, {s})\n",
        .{
            @typeName(T), n_heads, n_kv_heads, head_dim, t_len, if (split) "split" else "resident",
        },
    );
    try gpu_selftest_common.printDiff(w, "out", out_cpu, out_gpu);

    var fails = gpu_selftest_common.countFail(out_cpu, out_gpu, 1e-5, 1e-3);
    try w.print("  fails (atol=1e-5 rtol=1e-3): {d}\n", .{fails});
    if (split) {
        const out_again = try a.alloc(f32, n_heads * head_dim);
        try attn.runTyped(T, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, t_len, out_again);
        const same = std.mem.eql(
            u8,
            std.mem.sliceAsBytes(out_gpu),
            std.mem.sliceAsBytes(out_again),
        );
        try w.print("  second run: {s}\n", .{if (same) "same bytes" else "OTHER BYTES"});
        if (!same) {
            fails += 1;
        }
    }

    return fails;
}

/// Check the attention kernel set against the CPU reference: decode rows through
/// `attn_record.record` and prefill chunks through `attn_record.recordPrefill`, around and past
/// `split_len`. A case fails when its maximum relative error exceeds the declared tolerance.
fn attnCases(comptime T: type, gpa: Allocator, rig: *Rig, w: *std.Io.Writer) !usize {
    const split_len = limits.attn_split_len;
    const decode_t = [_]u32{ split_len - 1, split_len + 1, 4 * split_len + 3 };
    const prefill = [_]Case{
        .{ .kind = .prefill, .start_pos = split_len - 3, .b = 7 },
        .{ .kind = .prefill, .start_pos = 4 * split_len + 1, .b = 5 },
    };

    var fails: usize = 0;
    for (case_shapes) |shape| {
        for (decode_t) |t_len| {
            fails += try attnCase(
                T,
                gpa,
                rig,
                w,
                shape,
                .{ .kind = .decode, .start_pos = t_len - 1, .b = 1 },
            );
        }

        for (prefill) |c| {
            fails += try attnCase(T, gpa, rig, w, shape, c);
        }
    }

    return fails;
}

/// One case at KV element type `T`. It returns 1 when the maximum
/// relative error of any output row exceeds the declared tolerance, else 0.
fn attnCase(
    comptime T: type,
    gpa: Allocator,
    rig: *Rig,
    w: *std.Io.Writer,
    shape: CaseShape,
    c: Case,
) !usize {
    // Check the shape and open a case arena.
    const positions = c.positions();
    std.debug.assert(shape.n_heads % shape.n_kv_heads == 0);
    std.debug.assert(shape.n_heads / shape.n_kv_heads <= limits.attn_max_group);
    std.debug.assert(attn_split.headDimFits(shape.head_dim));
    std.debug.assert(c.b >= 1 and positions <= limits.max_model_ctx);

    var case_arena: std.heap.ArenaAllocator = .init(gpa);
    defer case_arena.deinit();

    const a = case_arena.allocator();

    // Plan the case, bind the buffers, and run it.
    const in = try caseInputs(T, a, shape, c);

    const scratch = attn_split.Scratch.init(shape.n_heads, shape.head_dim, positions);
    const plan = switch (c.kind) {
        .decode => attn_split.planDecode(c.start_pos + 1),
        .prefill => attn_split.planPrefill(0, c.start_pos, c.b, scratch.max_splits),
    };

    plan.assertValid();
    std.debug.assert(plan.resident_count + plan.split_count == c.b);

    const binding = try bindCase(T, rig, shape, c, in, scratch);
    try submitCase(rig, c, &plan, binding);

    // Compare against the reference and print the verdict.
    const max_rel = maxRelStrict(in.out_cpu, rig.out.sliceTyped(f32, in.out_cpu.len));
    const ok = op_requires.toleranceSatisfied(kernels.AttnKernels.tolerance, max_rel, false);
    const verdict = if (ok) "ok" else "FAIL";
    switch (c.kind) {
        .decode => try w.print(
            "Attention kernel set (kv={s}, shape={d}/{d}/{d}, decode, t_len={d}, {s}): " ++
                "max_rel={e:.3} {s}\n",
            .{
                @typeName(T),
                shape.n_heads,
                shape.n_kv_heads,
                shape.head_dim,
                c.start_pos + 1,
                if (plan.split_count > 0) "split" else "resident",
                max_rel,
                verdict,
            },
        ),
        .prefill => try w.print(
            "Attention kernel set (kv={s}, shape={d}/{d}/{d}, prefill, start_pos={d}, " ++
                "b={d}, {d} resident + {d} split): max_rel={e:.3} {s}\n",
            .{
                @typeName(T),
                shape.n_heads,
                shape.n_kv_heads,
                shape.head_dim,
                c.start_pos,
                c.b,
                plan.resident_count,
                plan.split_count,
                max_rel,
                verdict,
            },
        ),
    }

    return @intFromBool(!ok);
}

/// Build the random keys, values, and queries of a case, and the CPU reference rows.
fn caseInputs(comptime T: type, a: Allocator, shape: CaseShape, c: Case) !CaseInputs(T) {
    // Draw the keys, then the values.
    const row_elems: usize = @as(usize, shape.n_heads) * shape.head_dim;
    const kv_row: usize = @as(usize, shape.n_kv_heads) * shape.head_dim;
    const positions = c.positions();

    var prng = std.Random.DefaultPrng.init(0xA77E27 + @as(u64, c.start_pos) * 31 + c.b);
    const r = prng.random();
    const k_f32 = try a.alloc(f32, positions * kv_row);
    gpu_selftest_common.fillN(r, k_f32, 1.0);

    // V sits in [0.5, 1.5], so no output cancels toward zero and the
    // relative error stays well conditioned at depth.
    const v_f32 = try a.alloc(f32, positions * kv_row);
    for (v_f32) |*x| {
        x.* = 0.5 + r.float(f32);
    }

    // Round the keys and values to `T`.
    const k = try a.alloc(T, k_f32.len);
    const v = try a.alloc(T, v_f32.len);
    for (k, k_f32) |*d, x| {
        d.* = @floatCast(x);
    }

    for (v, v_f32) |*d, x| {
        d.* = @floatCast(x);
    }

    // Build the queries and the CPU reference rows.
    const q = try a.alloc(f32, c.b * row_elems);
    fillNeedleRows(r, q, k_f32, shape, c);

    const out_cpu = try a.alloc(f32, c.b * row_elems);
    const parts = try a.alloc(f32, attn_cpu.partialLen(shape.n_heads, shape.head_dim));
    // Row i attends to `start_pos + i + 1` positions.
    for (0..c.b) |i| {
        const t_len = c.start_pos + @as(u32, @intCast(i)) + 1;
        const q_row = q[i * row_elems ..][0..row_elems];
        const out_row = out_cpu[i * row_elems ..][0..row_elems];
        naive.attentionRowTyped(
            T,
            q_row,
            shape.n_heads,
            shape.head_dim,
            k,
            v,
            shape.n_kv_heads,
            t_len,
            parts,
            out_row,
        );
    }

    return .{ .q = q, .k = k, .v = v, .out_cpu = out_cpu };
}

/// Upload the inputs of a case, point the descriptor sets at the buffers, and return the binding.
fn bindCase(
    comptime T: type,
    rig: *Rig,
    shape: CaseShape,
    c: Case,
    in: CaseInputs(T),
    scratch: attn_split.Scratch,
) !kernels.AttnKernels.Binding {
    // Upload the inputs and size the output and scratch buffers.
    const q_bytes = try uploadSlice(f32, &rig.q, in.q);
    const kv_bytes = try uploadSlice(T, &rig.k, in.k);
    _ = try uploadSlice(T, &rig.v, in.v);

    const out_bytes: u64 = in.out_cpu.len * @sizeOf(f32);
    try rig.out.ensure(out_bytes);
    @memset(rig.out.sliceTyped(u32, in.out_cpu.len), sentinel_bits);
    try rig.scratch.ensure(scratch.totalBytes());

    // Describe the buffers.
    const q_info: DescriptorBufferInfo = .{ .buffer = rig.q.buffer, .offset = 0, .range = q_bytes };
    const k_info: DescriptorBufferInfo = .{
        .buffer = rig.k.buffer,
        .offset = 0,
        .range = kv_bytes,
    };
    const v_info: DescriptorBufferInfo = .{
        .buffer = rig.v.buffer,
        .offset = 0,
        .range = kv_bytes,
    };
    const scratch_info: DescriptorBufferInfo = .{
        .buffer = rig.scratch.buffer,
        .offset = 0,
        .range = scratch.totalBytes(),
    };
    const out_info: DescriptorBufferInfo = .{
        .buffer = rig.out.buffer,
        .offset = 0,
        .range = out_bytes,
    };
    kernels.AttentionSplit.assertKBinding(k_info);
    kernels.AttentionSplit.assertVBinding(v_info);

    // Point the descriptor sets at the buffers.
    const resident_set = switch (c.kind) {
        .decode => rig.single.oneshot.desc_set,
        .prefill => rig.batch_set.desc_set,
    };

    kernels.updateStorageBindings(rig.ctx, resident_set, &.{ q_info, k_info, v_info, out_info });
    kernels.updateStorageBindings(
        rig.ctx,
        rig.split_set.desc_set,
        &.{ q_info, k_info, v_info, scratch_info },
    );
    kernels.updateStorageBindings(rig.ctx, rig.merge_set.desc_set, &.{ scratch_info, out_info });

    // Return the binding.
    return .{
        .shape = .{
            .n_heads = shape.n_heads,
            .n_kv_heads = shape.n_kv_heads,
            .head_dim = shape.head_dim,
            .stream_stride_elems = @intCast(in.k.len),
            .max_splits = scratch.max_splits,
        },
        .resident = switch (c.kind) {
            .decode => .{ .single = resident_set },
            .prefill => .{ .batch = resident_set },
        },
        .split = rig.split_set.desc_set,
        .merge = rig.merge_set.desc_set,
    };
}

/// Record the plan of a case into one command buffer, submit it, and wait.
fn submitCase(
    rig: *Rig,
    c: Case,
    plan: *const attn_split.AttnPlan,
    binding: kernels.AttnKernels.Binding,
) !void {
    const attn_kernels: kernels.AttnKernels = .{
        .single = rig.single,
        .multi = &rig.multi,
        .passes = &rig.passes,
        .batch = &rig.batch,
    };
    const cb = rig.split_set.cmd_buffer;
    try rig.ctx.vkd.resetCommandBuffer(cb, .{});
    try rig.ctx.vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = true } });
    switch (c.kind) {
        .decode => attn_record.record(kernels.AttnKernels, attn_kernels, cb, plan, binding),
        .prefill => attn_record.recordPrefill(
            kernels.AttnKernels,
            attn_kernels,
            cb,
            plan,
            binding,
            c.start_pos,
        ),
    }

    try rig.ctx.vkd.endCommandBuffer(cb);

    const cmds = [_]@TypeOf(cb){cb};
    try rig.ctx.vkd.queueSubmit(
        rig.ctx.queue,
        &.{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }},
        .null_handle,
    );
    try rig.ctx.vkd.queueWaitIdle(rig.ctx.queue);
}

/// Give each query row a needle: a score near `needle_score` at its last
/// causal position and at the position after it, for every head.
fn fillNeedleRows(r: std.Random, q: []f32, k: []const f32, shape: CaseShape, c: Case) void {
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
            for (dst, at_p, at_next) |*x, a, n| {
                x.* = scale * (a + n + 0.25 * (r.float(f32) - 0.5) * 2.0);
            }
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
        if (!(rel <= mr)) {
            mr = if (std.math.isNan(rel)) std.math.inf(f32) else rel;
        }
    }

    return mr;
}
