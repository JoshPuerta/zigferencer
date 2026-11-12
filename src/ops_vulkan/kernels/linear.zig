//! The linear kernel wrappers over the resident Q8_0 and Q4_0 weights: the decode forms of one
//! input row and the batch forms of a prefill chunk.

const std = @import("std");
const vk = @import("vulkan");
const limits = @import("../../core/limits.zig");
const op_requires = @import("../../core/op_requires.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");

const spirv = @import("spirv.zig");
const max_workgroups_x = @import("pipeline.zig").max_workgroups_x;
const Pipeline = @import("pipeline.zig").Pipeline;
const updateStorageBindings = @import("pipeline.zig").updateStorageBindings;
const HostBuffer = @import("memory.zig").HostBuffer;
const OneShot = @import("submit.zig").OneShot;
const Recorded = @import("submit.zig").Recorded;
const submitOnce = @import("submit.zig").submitOnce;
const Q80Weight = @import("weights.zig").Q80Weight;
const VecAdd = @import("elementwise.zig").VecAdd;

// -- types ------------------------------------------------------------------

/// Bit-exact Q8_0 matvec `out = input @ weight^T`, the shape of the CPU `linearF32Q80` op. `input`
/// is `[k]f32` and `out` is `[n]f32`, with `k` a multiple of 32. The caller builds a `Q80Weight`
/// once and passes it to `dispatch`.
pub const LinearQ80 = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_in: HostBuffer,
    buf_out: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // `tests/vulkan_linear_q8_0_test.zig` holds the bit-identity check.
        .tolerance = .bit_exact,
    };

    /// Must match the `Params` block of `linear_q8_0.comp`.
    const PushConsts = extern struct { k: u32, n: u32 };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ80 {
        const pipeline = try Pipeline.init(ctx, &spirv.linear_q8_0_spv, 4, @sizeOf(PushConsts));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 4);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_in = .{ .ctx = ctx },
            .buf_out = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *LinearQ80) void {
        self.buf_in.deinit();
        self.buf_out.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const DispatchError = VecAdd.RunError || error{
        InputShapeMismatch,
        OutputShapeMismatch,
    };

    pub const RunOneShotError = DispatchError || Q80Weight.InitError;

    /// `out = input @ weight^T` with the weight resident in VRAM. `input.len` must equal
    /// `weight.in_features`, and `out.len` must equal `weight.out_features`. The per-call cost is
    /// the kilobyte-scale copies, a descriptor update, and the dispatch.
    pub fn dispatch(
        self: *LinearQ80,
        weight: *const Q80Weight,
        input: []const f32,
        out: []f32,
    ) DispatchError!void {
        if (input.len != weight.in_features) return error.InputShapeMismatch;
        if (out.len != weight.out_features) return error.OutputShapeMismatch;

        // Stage the inputs in host-visible memory.
        const in_bytes: vk.DeviceSize = @intCast(input.len * @sizeOf(f32));
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_in.ensure(in_bytes);
        try self.buf_out.ensure(out_bytes);

        @memcpy(self.buf_in.sliceTyped(f32, input.len), input);

        // Bind the input, the weight, and the output.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_in.buffer, .offset = 0, .range = in_bytes },
            weight.scales,
            weight.quants,
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = out_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

        // Record the dispatch.
        try self.ctx.vkd.resetCommandBuffer(self.oneshot.cmd_buffer, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(self.oneshot.cmd_buffer, &begin);
        self.ctx.vkd.cmdBindPipeline(self.oneshot.cmd_buffer, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{self.oneshot.desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            self.oneshot.cmd_buffer,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        const pc: PushConsts = .{ .k = weight.in_features, .n = weight.out_features };
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        const group_count_x: u32 = (weight.out_features + 31) / 32;
        std.debug.assert(group_count_x <= max_workgroups_x);
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, group_count_x, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }

    /// Build a `Q80Weight`, dispatch against it, and free it. For tests and the
    /// CLI. Never call it inside a per-token loop.
    pub fn runOneShot(
        self: *LinearQ80,
        input: []const f32,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
        out: []f32,
    ) RunOneShotError!void {
        var w = try Q80Weight.init(self.ctx, weight_bytes, in_features, out_features);
        defer w.deinit();

        try self.dispatch(&w, input, out);
    }
};

/// Workgroup-cooperative Q8_0 matvec, with the shape and the storage layout of `LinearQ80`. The
/// caller builds the weight once, and `k` must be a multiple of 32.
pub const LinearQ80Reduce = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_in: HostBuffer,
    buf_out: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // The drift grows with k: 8.5e-5 at k=576 and about 1e-4 at k=14336. 2e-4 leaves headroom
        // through the widest Llama-8B shape, ffn_down at k=14336.
        .tolerance = .{ .relative = 2e-4 },
    };

    const PushConsts = extern struct { k: u32, n: u32 };

    pub const InitError = LinearQ80.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ80Reduce {
        const pipeline = try Pipeline.init(
            ctx,
            &spirv.linear_q8_0_reduce_spv,
            4,
            @sizeOf(PushConsts),
        );
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 4);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_in = .{ .ctx = ctx },
            .buf_out = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *LinearQ80Reduce) void {
        self.buf_in.deinit();
        self.buf_out.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const DispatchError = LinearQ80.DispatchError;

    /// Record one Q8_0 matmul into a command buffer the caller owns. Before the submit, the caller
    /// points `desc_set` at (input, scales, quants, output), and it inserts the barriers. The fused
    /// runners chain many such dispatches, and they run the submit and wait-idle once.
    pub fn recordInto(
        self: *LinearQ80Reduce,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        k: u32,
        n: u32,
    ) void {
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        const pc: PushConsts = .{ .k = k, .n = n };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        recordReduceGrid(self.ctx, cb, n);
    }

    pub fn dispatch(
        self: *LinearQ80Reduce,
        weight: *const Q80Weight,
        input: []const f32,
        out: []f32,
    ) DispatchError!void {
        if (input.len != weight.in_features) return error.InputShapeMismatch;
        if (out.len != weight.out_features) return error.OutputShapeMismatch;

        // Stage the inputs in host-visible memory.
        const in_bytes: vk.DeviceSize = @intCast(input.len * @sizeOf(f32));
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_in.ensure(in_bytes);
        try self.buf_out.ensure(out_bytes);

        @memcpy(self.buf_in.sliceTyped(f32, input.len), input);

        // Bind the input, the weight, and the output.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_in.buffer, .offset = 0, .range = in_bytes },
            weight.scales,
            weight.quants,
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = out_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

        // Record the dispatch.
        try self.ctx.vkd.resetCommandBuffer(self.oneshot.cmd_buffer, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(self.oneshot.cmd_buffer, &begin);
        self.ctx.vkd.cmdBindPipeline(self.oneshot.cmd_buffer, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{self.oneshot.desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            self.oneshot.cmd_buffer,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        const pc: PushConsts = .{ .k = weight.in_features, .n = weight.out_features };
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        recordReduceGrid(self.ctx, self.oneshot.cmd_buffer, weight.out_features);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// Q4_0 decode matvec. The weight comes from `Q80Weight.initQ40`, and `k` must be a multiple of 32.
pub const LinearQ40Reduce = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_in: HostBuffer,
    buf_out: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // The Q4 quantization error belongs to the data. The bench measures against the Q4_0 CPU
        // reference, so the cross-lane reduce drift is the only error the kernel adds.
        .tolerance = .{ .relative = 2e-4 },
    };

    const PushConsts = extern struct { k: u32, n: u32 };
    pub const InitError = LinearQ80.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ40Reduce {
        const pipeline = try Pipeline.init(
            ctx,
            &spirv.linear_q4_0_reduce_spv,
            4,
            @sizeOf(PushConsts),
        );
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 4);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_in = .{ .ctx = ctx },
            .buf_out = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *LinearQ40Reduce) void {
        self.buf_in.deinit();
        self.buf_out.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const DispatchError = LinearQ80.DispatchError;

    /// Record one Q4_0 matvec into a command buffer the caller owns. The caller
    /// binds the descriptor set and inserts the barriers.
    pub fn recordInto(
        self: *LinearQ40Reduce,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        k: u32,
        n: u32,
    ) void {
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        const pc: PushConsts = .{ .k = k, .n = n };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        recordReduceGrid(self.ctx, cb, n);
    }

    pub fn dispatch(
        self: *LinearQ40Reduce,
        weight: *const Q80Weight,
        input: []const f32,
        out: []f32,
    ) DispatchError!void {
        if (input.len != weight.in_features) return error.InputShapeMismatch;
        if (out.len != weight.out_features) return error.OutputShapeMismatch;

        // Stage the inputs in host-visible memory.
        const in_bytes: vk.DeviceSize = @intCast(input.len * @sizeOf(f32));
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_in.ensure(in_bytes);
        try self.buf_out.ensure(out_bytes);

        @memcpy(self.buf_in.sliceTyped(f32, input.len), input);

        // Bind the input, the weight, and the output.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_in.buffer, .offset = 0, .range = in_bytes },
            weight.scales,
            weight.quants,
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = out_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

        // Record the dispatch.
        try self.ctx.vkd.resetCommandBuffer(self.oneshot.cmd_buffer, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(self.oneshot.cmd_buffer, &begin);
        self.ctx.vkd.cmdBindPipeline(self.oneshot.cmd_buffer, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{self.oneshot.desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            self.oneshot.cmd_buffer,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );
        const pc: PushConsts = .{ .k = weight.in_features, .n = weight.out_features };
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        recordReduceGrid(self.ctx, self.oneshot.cmd_buffer, weight.out_features);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// Q8_0 matvec with staged input and a `subgroupAdd` reduction, with the shape of
/// `LinearQ80Reduce`. `weight.in_features` must be at most `lds_capacity`, and `dispatch` refuses a
/// wider weight.
pub const LinearQ80Subgroup = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_in: HostBuffer,
    buf_out: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 2e-4 },
    };

    const PushConsts = extern struct { k: u32, n: u32 };

    pub const InitError = LinearQ80.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ80Subgroup {
        const pipeline = try Pipeline.init(
            ctx,
            &spirv.linear_q8_0_subgroup_spv,
            4,
            @sizeOf(PushConsts),
        );
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 4);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_in = .{ .ctx = ctx },
            .buf_out = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *LinearQ80Subgroup) void {
        self.buf_in.deinit();
        self.buf_out.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const DispatchError = LinearQ80.DispatchError || error{InFeaturesTooLargeForLdsKernel};

    /// The LDS_CAPACITY define of `linear_q8_0_subgroup.comp`, the length of `input_lds`.
    const lds_capacity: u32 = limits.subgroup_lds_capacity;

    pub fn dispatch(
        self: *LinearQ80Subgroup,
        weight: *const Q80Weight,
        input: []const f32,
        out: []f32,
    ) DispatchError!void {
        if (input.len != weight.in_features) return error.InputShapeMismatch;
        if (out.len != weight.out_features) return error.OutputShapeMismatch;
        // The shader holds the whole input vector in shared memory. Return an
        // error rather than corrupt memory when it does not fit.
        if (weight.in_features > lds_capacity) return error.InFeaturesTooLargeForLdsKernel;

        // Stage the inputs in host-visible memory.
        const in_bytes: vk.DeviceSize = @intCast(input.len * @sizeOf(f32));
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_in.ensure(in_bytes);
        try self.buf_out.ensure(out_bytes);

        @memcpy(self.buf_in.sliceTyped(f32, input.len), input);

        // Bind the input, the weight, and the output.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_in.buffer, .offset = 0, .range = in_bytes },
            weight.scales,
            weight.quants,
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = out_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

        // Record the dispatch.
        try self.ctx.vkd.resetCommandBuffer(self.oneshot.cmd_buffer, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(self.oneshot.cmd_buffer, &begin);
        self.ctx.vkd.cmdBindPipeline(self.oneshot.cmd_buffer, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{self.oneshot.desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            self.oneshot.cmd_buffer,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        const pc: PushConsts = .{ .k = weight.in_features, .n = weight.out_features };
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        recordReduceGrid(self.ctx, self.oneshot.cmd_buffer, weight.out_features);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// Batched Q8_0 matmul, a `(B, K)` input against an `(N, K)` weight into a `(B, N)` output. `k`
/// must be a multiple of 32. This is the prefill kernel, where B is large. Decode keeps
/// `LinearQ80Reduce`, because per-call overhead dominates at B = 1.
pub const LinearQ80Batch = struct {
    /// The B_TILE and N_TILE defines of `linear_q8_0_batch.comp`.
    const batch_tile: u32 = limits.q8_batch_b_tile;
    const output_tile: u32 = limits.q8_batch_n_tile;

    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 2e-4 },
    };

    const PushConsts = extern struct { b: u32, k: u32, n: u32 };
    pub const InitError = LinearQ80.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ80Batch {
        const pipeline = try Pipeline.init(
            ctx,
            &spirv.linear_q8_0_batch_spv,
            4,
            @sizeOf(PushConsts),
        );

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *LinearQ80Batch) void {
        self.pipeline.deinit(self.ctx);
    }

    /// Record one batched Q8_0 matmul. The caller binds the set (input, scales, quants, output).
    /// `b` is the batch size of this dispatch, and the kernel covers a tail tile.
    pub fn recordInto(
        self: *LinearQ80Batch,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        b: u32,
        k: u32,
        n: u32,
    ) void {
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        const pc: PushConsts = .{ .b = b, .k = k, .n = n };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        const wg_n: u32 = (n + output_tile - 1) / output_tile;
        const wg_b: u32 = (b + batch_tile - 1) / batch_tile;
        std.debug.assert(wg_n <= max_workgroups_x);
        std.debug.assert(wg_b <= max_workgroups_x);
        self.ctx.vkd.cmdDispatch(cb, wg_n, wg_b, 1);
    }

    /// One-shot batched matmul for the correctness gate and the bench, through `matmulOneShot`.
    /// `input` is `[b, in_features]` row-major and `out` is `[b, out_features]`.
    pub fn runOneShot(
        self: *LinearQ80Batch,
        input: []const f32,
        weight_bytes: []const u8,
        b: usize,
        in_features: usize,
        out_features: usize,
        out: []f32,
    ) !void {
        return matmulOneShot(self, input, weight_bytes, b, in_features, out_features, out);
    }
};

/// Register-blocked, LDS-staged Q8_0 GEMM, the wide prefill matmul. It takes the bindings and the
/// push constants of `LinearQ80Batch`, so the runner swaps one for the other at `recordInto`. `k`
/// must be a multiple of 32.
pub const LinearQ80Gemm = struct {
    /// The BM and BN defines of `linear_q8_0_gemm.comp`.
    const tile_m: u32 = limits.gemm_m_tile;
    const tile_n: u32 = limits.gemm_n_tile;

    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 2e-4 },
    };

    const PushConsts = extern struct { b: u32, k: u32, n: u32 };
    pub const InitError = LinearQ80.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ80Gemm {
        const pipeline = try Pipeline.init(
            ctx,
            &spirv.linear_q8_0_gemm_spv,
            4,
            @sizeOf(PushConsts),
        );

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *LinearQ80Gemm) void {
        self.pipeline.deinit(self.ctx);
    }

    /// Record one tiled Q8_0 matmul. The caller binds the descriptor set
    /// (input, scales, quants, output). `b` is the batch size of this dispatch.
    /// The kernel zero-fills or skips a partial tile.
    pub fn recordInto(
        self: *LinearQ80Gemm,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        b: u32,
        k: u32,
        n: u32,
    ) void {
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        const pc: PushConsts = .{ .b = b, .k = k, .n = n };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        const wg_n: u32 = (n + tile_n - 1) / tile_n;
        const wg_m: u32 = (b + tile_m - 1) / tile_m;
        std.debug.assert(wg_n <= max_workgroups_x);
        std.debug.assert(wg_m <= max_workgroups_x);
        self.ctx.vkd.cmdDispatch(cb, wg_n, wg_m, 1);
    }

    pub fn runOneShot(
        self: *LinearQ80Gemm,
        input: []const f32,
        weight_bytes: []const u8,
        b: usize,
        in_features: usize,
        out_features: usize,
        out: []f32,
    ) !void {
        return matmulOneShot(self, input, weight_bytes, b, in_features, out_features, out);
    }
};

/// Q4_0 prefill matmul, with the bindings and the push constants of `LinearQ80Batch`. The weight
/// comes from `Q80Weight.initQ40`, and `k` must be a multiple of 32.
pub const LinearQ40Batch = struct {
    /// The B_TILE and N_TILE defines of `linear_q4_0_batch.comp`.
    const batch_tile: u32 = limits.q4_batch_b_tile;
    const output_tile: u32 = limits.q4_batch_n_tile;

    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 2e-4 },
    };

    const PushConsts = extern struct { b: u32, k: u32, n: u32 };
    pub const InitError = LinearQ80.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ40Batch {
        const pipeline = try Pipeline.init(
            ctx,
            &spirv.linear_q4_0_batch_spv,
            4,
            @sizeOf(PushConsts),
        );

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *LinearQ40Batch) void {
        self.pipeline.deinit(self.ctx);
    }

    pub fn recordInto(
        self: *LinearQ40Batch,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        b: u32,
        k: u32,
        n: u32,
    ) void {
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline.handle);
        const desc_sets = [_]vk.DescriptorSet{desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        const pc: PushConsts = .{ .b = b, .k = k, .n = n };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        const wg_n: u32 = (n + output_tile - 1) / output_tile;
        const wg_b: u32 = (b + batch_tile - 1) / batch_tile;
        std.debug.assert(wg_n <= max_workgroups_x);
        std.debug.assert(wg_b <= max_workgroups_x);
        self.ctx.vkd.cmdDispatch(cb, wg_n, wg_b, 1);
    }
};

// -- helpers ----------------------------------------------------------------

/// Record the grid of a reduce matvec over `n` outputs, one workgroup per output. The shaders read
/// the same width through their GRID_X define.
fn recordReduceGrid(ctx: *const vulkan_ctx.Context, cb: vk.CommandBuffer, n: u32) void {
    const grid = limits.reduceGrid(n);
    std.debug.assert(grid.x <= max_workgroups_x);
    std.debug.assert(grid.y <= max_workgroups_x);
    ctx.vkd.cmdDispatch(cb, grid.x, grid.y, 1);
}

/// The shared one-shot driver of `LinearQ80Batch` and `LinearQ80Gemm`, for tests and the bench.
/// `kernel` points at a kernel that exposes `.ctx`, `.pipeline`, and `recordInto(cb, set, b, k,
/// n)`.
fn matmulOneShot(
    kernel: anytype,
    input: []const f32,
    weight_bytes: []const u8,
    b: usize,
    in_features: usize,
    out_features: usize,
    out: []f32,
) !void {
    std.debug.assert(input.len == b * in_features);
    std.debug.assert(out.len == b * out_features);

    // Build the weight of this call.
    const ctx = kernel.ctx;
    var w = try Q80Weight.init(ctx, weight_bytes, in_features, out_features);
    defer w.deinit();

    // Stage the input and size the output.
    const F = @sizeOf(f32);
    var buf_in: HostBuffer = .{ .ctx = ctx };
    defer buf_in.deinit();

    var buf_out: HostBuffer = .{ .ctx = ctx };
    defer buf_out.deinit();

    try buf_in.ensure(@intCast(input.len * F));
    try buf_out.ensure(@intCast(out.len * F));
    @memcpy(buf_in.sliceTyped(f32, input.len), input);

    // Build a throwaway pool and set, and bind the four buffers.
    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{ .type = .storage_buffer, .descriptor_count = 4 },
    };
    const pool_info: vk.DescriptorPoolCreateInfo = .{
        .max_sets = 1,
        .pool_size_count = 1,
        .p_pool_sizes = &pool_sizes,
    };
    const pool = try ctx.vkd.createDescriptorPool(ctx.device, &pool_info, null);
    defer ctx.vkd.destroyDescriptorPool(ctx.device, pool, null);

    const layouts = [_]vk.DescriptorSetLayout{kernel.pipeline.set_layout};
    const ai: vk.DescriptorSetAllocateInfo = .{
        .descriptor_pool = pool,
        .descriptor_set_count = 1,
        .p_set_layouts = &layouts,
    };
    var set: vk.DescriptorSet = .null_handle;
    try ctx.vkd.allocateDescriptorSets(ctx.device, &ai, @ptrCast(&set));

    const infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = buf_in.buffer, .offset = 0, .range = buf_in.capacity },
        w.scales,
        w.quants,
        .{ .buffer = buf_out.buffer, .offset = 0, .range = buf_out.capacity },
    };
    updateStorageBindings(ctx, set, &infos);

    // Dispatch through `recordInto`, then read the result back.
    const Rec = struct {
        kernel: @TypeOf(kernel),
        set: vk.DescriptorSet,
        b: u32,
        k: u32,
        n: u32,

        pub fn record(r: @This(), cb: vk.CommandBuffer) Recorded {
            r.kernel.recordInto(cb, r.set, r.b, r.k, r.n);
            return .{};
        }
    };
    try submitOnce(
        ctx,
        .dispatch,
        Rec{
            .kernel = kernel,
            .set = set,
            .b = @intCast(b),
            .k = @intCast(in_features),
            .n = @intCast(out_features),
        },
    );
    @memcpy(out, buf_out.sliceTyped(f32, out.len));
}
