//! The element-wise and per-row kernel wrappers: the adds, the output gate, the RMS norms, SwiGLU,
//! and the QK norm of Qwen3.5 full attention.

const std = @import("std");
const vk = @import("vulkan");
const limits = @import("../../core/limits.zig");
const op_requires = @import("../../core/op_requires.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");

const spirv = @import("spirv.zig");
const Pipeline = @import("pipeline.zig").Pipeline;
const updateStorageBindings = @import("pipeline.zig").updateStorageBindings;
const HostBuffer = @import("memory.zig").HostBuffer;
const OneShot = @import("submit.zig").OneShot;

// -- types ------------------------------------------------------------------

/// Element-wise `out[i] = a[i] + b[i]`, the proof-of-life kernel. The three slices must have the
/// same length.
pub const VecAdd = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_a: HostBuffer,
    buf_b: HostBuffer,
    buf_out: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .bit_exact,
    };

    pub const InitError = Pipeline.InitError || OneShot.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!VecAdd {
        const pipeline = try Pipeline.init(ctx, &spirv.vec_add_spv, 3, @sizeOf(u32));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 3);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_a = .{ .ctx = ctx },
            .buf_b = .{ .ctx = ctx },
            .buf_out = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *VecAdd) void {
        self.buf_a.deinit();
        self.buf_b.deinit();
        self.buf_out.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const RunError = error{
        BuffersLengthMismatch,
        NoSuitableMemoryType,
    } ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError ||
        vk.DeviceWrapper.MapMemoryError ||
        vk.DeviceWrapper.BeginCommandBufferError ||
        vk.DeviceWrapper.EndCommandBufferError ||
        vk.DeviceWrapper.QueueSubmitError ||
        vk.DeviceWrapper.QueueWaitIdleError ||
        vk.DeviceWrapper.ResetCommandBufferError;

    pub fn run(
        self: *VecAdd,
        a: []const f32,
        b: []const f32,
        out: []f32,
    ) RunError!void {
        if (a.len != b.len or a.len != out.len) return error.BuffersLengthMismatch;

        // Stage the inputs in host-visible memory.
        const n: u32 = @intCast(a.len);
        const byte_size: vk.DeviceSize = @intCast(a.len * @sizeOf(f32));

        try self.buf_a.ensure(byte_size);
        try self.buf_b.ensure(byte_size);
        try self.buf_out.ensure(byte_size);

        @memcpy(self.buf_a.sliceTyped(f32, a.len), a);
        @memcpy(self.buf_b.sliceTyped(f32, b.len), b);

        // Bind the three buffers.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_a.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_b.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = byte_size },
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
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(u32),
            @ptrCast(&n),
        );
        const group_count_x: u32 = (n + 255) / 256;
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
};

/// In-place residual add `dst[i] += x[i]`. `x` and `dst` must have the same length.
pub const ResidualAdd = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_x: HostBuffer,
    buf_dst: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .bit_exact,
    };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!ResidualAdd {
        const pipeline = try Pipeline.init(ctx, &spirv.residual_add_spv, 2, @sizeOf(u32));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 2);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_x = .{ .ctx = ctx },
            .buf_dst = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *ResidualAdd) void {
        self.buf_x.deinit();
        self.buf_dst.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const RunError = VecAdd.RunError || error{LengthMismatch};

    /// Record `n_dispatches` adds into one command buffer and submit once, so `dst_final[i] =
    /// dst_init[i] + n_dispatches * x[i]`. The benchmark reads the timing: per-dispatch time falls
    /// as `n_dispatches` rises, which measures the submit that one buffer per token amortizes.
    pub fn runRepeat(
        self: *ResidualAdd,
        n_dispatches: u32,
        x: []const f32,
        dst_init: []const f32,
        dst_final: []f32,
    ) RunError!void {
        if (x.len != dst_init.len) return error.LengthMismatch;
        if (x.len != dst_final.len) return error.LengthMismatch;

        // Stage the inputs in host-visible memory.
        const n: u32 = @intCast(x.len);
        const byte_size: vk.DeviceSize = @intCast(x.len * @sizeOf(f32));

        try self.buf_x.ensure(byte_size);
        try self.buf_dst.ensure(byte_size);

        @memcpy(self.buf_x.sliceTyped(f32, x.len), x);
        @memcpy(self.buf_dst.sliceTyped(f32, dst_init.len), dst_init);

        // Bind x and dst.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_dst.buffer, .offset = 0, .range = byte_size },
        };
        var writes: [2]vk.WriteDescriptorSet = undefined;
        for (&writes, 0..) |*w, i| {
            w.* = .{
                .dst_set = self.oneshot.desc_set,
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = @ptrCast(&buf_infos[i]),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            };
        }

        self.ctx.vkd.updateDescriptorSets(self.ctx.device, &writes, null);

        // Record the dispatches, with a barrier between each pair.
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
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(u32),
            @ptrCast(&n),
        );
        const group_count_x: u32 = (n + 255) / 256;

        // Each dispatch reads the writes of the last one to `dst`. Without an explicit barrier the
        // driver is free to overlap them.
        const ssbo_barrier: vk.MemoryBarrier = .{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true },
        };
        const mem_barriers = [_]vk.MemoryBarrier{ssbo_barrier};
        var i: u32 = 0;
        while (i < n_dispatches) : (i += 1) {
            if (i > 0) {
                self.ctx.vkd.cmdPipelineBarrier(
                    self.oneshot.cmd_buffer,
                    .{ .compute_shader_bit = true },
                    .{ .compute_shader_bit = true },
                    .{},
                    &mem_barriers,
                    null,
                    null,
                );
            }

            self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, group_count_x, 1, 1);
        }

        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(dst_final, self.buf_dst.sliceTyped(f32, dst_final.len));
    }

    /// Record one `dst[i] += x[i]` dispatch into a command buffer the caller owns. `dst` carries
    /// both the input and the result. The caller binds the set (x, dst) and inserts any barriers.
    pub fn recordInto(
        self: *ResidualAdd,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
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

        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(u32),
            @ptrCast(&n),
        );
        const group_count_x: u32 = (n + 255) / 256;
        self.ctx.vkd.cmdDispatch(cb, group_count_x, 1, 1);
    }

    pub fn run(self: *ResidualAdd, x: []const f32, dst: []f32) RunError!void {
        if (x.len != dst.len) return error.LengthMismatch;

        // Stage the inputs in host-visible memory.
        const n: u32 = @intCast(x.len);
        const byte_size: vk.DeviceSize = @intCast(x.len * @sizeOf(f32));

        try self.buf_x.ensure(byte_size);
        try self.buf_dst.ensure(byte_size);

        @memcpy(self.buf_x.sliceTyped(f32, x.len), x);
        @memcpy(self.buf_dst.sliceTyped(f32, dst.len), dst);

        // Bind x and dst.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_dst.buffer, .offset = 0, .range = byte_size },
        };
        var writes: [2]vk.WriteDescriptorSet = undefined;
        for (&writes, 0..) |*w, i| {
            w.* = .{
                .dst_set = self.oneshot.desc_set,
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = @ptrCast(&buf_infos[i]),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            };
        }

        self.ctx.vkd.updateDescriptorSets(self.ctx.device, &writes, null);

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
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(u32),
            @ptrCast(&n),
        );
        const group_count_x: u32 = (n + 255) / 256;
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, group_count_x, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(dst, self.buf_dst.sliceTyped(f32, dst.len));
    }
};

/// In-place output gate of Qwen3.5 full attention. `dst` and `gate` must have the same length.
pub const SigmoidGate = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_dst: HostBuffer,
    buf_gate: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 1e-5 },
    };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!SigmoidGate {
        const pipeline = try Pipeline.init(ctx, &spirv.sigmoid_gate_spv, 2, @sizeOf(u32));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 2);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_dst = .{ .ctx = ctx },
            .buf_gate = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *SigmoidGate) void {
        self.buf_dst.deinit();
        self.buf_gate.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    /// Record one gate dispatch into a command buffer the caller owns. The
    /// caller binds the descriptor set (dst, gate).
    pub fn recordInto(
        self: *SigmoidGate,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n: u32,
    ) void {
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline.handle);
        const sets = [_]vk.DescriptorSet{desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &sets,
            null,
        );

        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(u32),
            @ptrCast(&n),
        );
        self.ctx.vkd.cmdDispatch(cb, (n + 255) / 256, 1, 1);
    }

    pub const RunError = VecAdd.RunError || error{LengthMismatch};

    pub fn run(self: *SigmoidGate, dst: []f32, gate: []const f32) RunError!void {
        if (dst.len != gate.len) return error.LengthMismatch;

        const n: u32 = @intCast(dst.len);
        const byte_size: vk.DeviceSize = @intCast(dst.len * @sizeOf(f32));
        try self.buf_dst.ensure(byte_size);
        try self.buf_gate.ensure(byte_size);
        @memcpy(self.buf_dst.sliceTyped(f32, dst.len), dst);
        @memcpy(self.buf_gate.sliceTyped(f32, gate.len), gate);

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_dst.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_gate.buffer, .offset = 0, .range = byte_size },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

        try self.ctx.vkd.resetCommandBuffer(self.oneshot.cmd_buffer, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(self.oneshot.cmd_buffer, &begin);
        self.recordInto(self.oneshot.cmd_buffer, self.oneshot.desc_set, n);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
        @memcpy(dst, self.buf_dst.sliceTyped(f32, dst.len));
    }
};

/// RMS normalization `y[i] = x[i] * inv_rms * w[i]`. The three slices must have the same length.
pub const Rmsnorm = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_x: HostBuffer,
    buf_w: HostBuffer,
    buf_y: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 5e-5 },
    };

    const PushConsts = extern struct { d: u32, eps: f32 };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!Rmsnorm {
        const pipeline = try Pipeline.init(ctx, &spirv.rmsnorm_spv, 3, @sizeOf(PushConsts));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 3);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_x = .{ .ctx = ctx },
            .buf_w = .{ .ctx = ctx },
            .buf_y = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *Rmsnorm) void {
        self.buf_x.deinit();
        self.buf_w.deinit();
        self.buf_y.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const RunError = ResidualAdd.RunError;

    /// Record one rmsnorm dispatch into a command buffer the caller owns. The
    /// caller binds the descriptor set (x, w, y) and inserts any surrounding
    /// memory barriers.
    pub fn recordInto(
        self: *Rmsnorm,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        d: u32,
        eps: f32,
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

        const pc: PushConsts = .{ .d = d, .eps = eps };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        self.ctx.vkd.cmdDispatch(cb, 1, 1, 1);
    }

    pub fn run(self: *Rmsnorm, x: []const f32, w: []const f32, eps: f32, y: []f32) RunError!void {
        if (x.len != w.len or x.len != y.len) return error.LengthMismatch;

        // Stage the inputs in host-visible memory.
        const d: u32 = @intCast(x.len);
        const byte_size: vk.DeviceSize = @intCast(x.len * @sizeOf(f32));

        try self.buf_x.ensure(byte_size);
        try self.buf_w.ensure(byte_size);
        try self.buf_y.ensure(byte_size);

        @memcpy(self.buf_x.sliceTyped(f32, x.len), x);
        @memcpy(self.buf_w.sliceTyped(f32, w.len), w);

        // Bind the three buffers.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_w.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_y.buffer, .offset = 0, .range = byte_size },
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
        const pc: PushConsts = .{ .d = d, .eps = eps };
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, 1, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(y, self.buf_y.sliceTyped(f32, y.len));
    }
};

/// SwiGLU activation `dst[i] = silu(gate[i]) * up[i]`. The three slices must have the same length.
pub const Swiglu = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_gate: HostBuffer,
    buf_up: HostBuffer,
    buf_dst: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 1e-5 },
    };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!Swiglu {
        const pipeline = try Pipeline.init(ctx, &spirv.swiglu_spv, 3, @sizeOf(u32));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 3);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_gate = .{ .ctx = ctx },
            .buf_up = .{ .ctx = ctx },
            .buf_dst = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *Swiglu) void {
        self.buf_gate.deinit();
        self.buf_up.deinit();
        self.buf_dst.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const RunError = ResidualAdd.RunError;

    /// Record one swiglu dispatch into a command buffer the caller owns. The
    /// caller binds the descriptor set (gate, up, dst).
    pub fn recordInto(
        self: *Swiglu,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
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

        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(u32),
            @ptrCast(&n),
        );
        const group_count_x: u32 = (n + 255) / 256;
        self.ctx.vkd.cmdDispatch(cb, group_count_x, 1, 1);
    }

    pub fn run(self: *Swiglu, gate: []const f32, up: []const f32, dst: []f32) RunError!void {
        if (gate.len != up.len or gate.len != dst.len) return error.LengthMismatch;

        // Stage the inputs in host-visible memory.
        const n: u32 = @intCast(gate.len);
        const byte_size: vk.DeviceSize = @intCast(gate.len * @sizeOf(f32));

        try self.buf_gate.ensure(byte_size);
        try self.buf_up.ensure(byte_size);
        try self.buf_dst.ensure(byte_size);

        @memcpy(self.buf_gate.sliceTyped(f32, gate.len), gate);
        @memcpy(self.buf_up.sliceTyped(f32, up.len), up);

        // Bind the three buffers.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_gate.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_up.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_dst.buffer, .offset = 0, .range = byte_size },
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
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(u32),
            @ptrCast(&n),
        );
        const group_count_x: u32 = (n + 255) / 256;
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, group_count_x, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(dst, self.buf_dst.sliceTyped(f32, dst.len));
    }
};

/// Batched RMS normalization, applied independently to each of `b` input rows.
pub const RmsnormBatch = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 5e-5 },
    };

    const PushConsts = extern struct { d: u32, eps: f32 };
    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!RmsnormBatch {
        const pipeline = try Pipeline.init(ctx, &spirv.rmsnorm_batch_spv, 3, @sizeOf(PushConsts));

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *RmsnormBatch) void {
        self.pipeline.deinit(self.ctx);
    }

    pub fn recordInto(
        self: *RmsnormBatch,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        b: u32,
        d: u32,
        eps: f32,
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

        const pc: PushConsts = .{ .d = d, .eps = eps };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        self.ctx.vkd.cmdDispatch(cb, b, 1, 1);
    }
};

/// Per-head QK norm of Qwen3.5 full attention, with an optional de-interleave of the q and gate
/// halves. `head_dim` must equal `head_lanes`. It mirrors the CPU `fullMixer`, and
/// the fused runner is the only caller.
pub const QwenQkNorm = struct {
    /// The `local_size_x` of `qwen_qk_norm.comp`. One lane owns one head index.
    const head_lanes: u32 = limits.qwen_qk_head_dim;

    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const InitError = Pipeline.InitError;

    const Push = extern struct {
        head_dim: u32,
        in_stride: u32,
        gate_off: u32,
        write_gate: u32,
        eps: f32,
        in_off: u32 = 0,
        out_off: u32 = 0,
    };

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!QwenQkNorm {
        const pipeline = try Pipeline.init(ctx, &spirv.qwen_qk_norm_spv, 4, @sizeOf(Push));
        return .{ .ctx = ctx, .pipeline = pipeline };
    }

    pub fn deinit(self: *QwenQkNorm) void {
        self.pipeline.deinit(self.ctx);
    }

    pub fn recordInto(
        self: *const QwenQkNorm,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n_heads: u32,
        head_dim: u32,
        in_stride: u32,
        gate_off: u32,
        write_gate: bool,
        eps: f32,
    ) void {
        self.recordRow(cb, desc_set, n_heads, head_dim, in_stride, gate_off, write_gate, eps, 0, 0);
    }

    /// Record one batched row. Prefill passes the element offsets of token t as
    /// `in_off` and `out_off`. Decode passes zero for both.
    pub fn recordRow(
        self: *const QwenQkNorm,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n_heads: u32,
        head_dim: u32,
        in_stride: u32,
        gate_off: u32,
        write_gate: bool,
        eps: f32,
        in_off: u32,
        out_off: u32,
    ) void {
        std.debug.assert(head_dim == head_lanes);

        // Bind the pipeline and the descriptor set.
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline.handle);
        const sets = [_]vk.DescriptorSet{desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &sets,
            null,
        );

        // Push the row and dispatch one workgroup per head.
        const push: Push = .{
            .head_dim = head_dim,
            .in_stride = in_stride,
            .gate_off = gate_off,
            .write_gate = @intFromBool(write_gate),
            .eps = eps,
            .in_off = in_off,
            .out_off = out_off,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(Push),
            @ptrCast(&push),
        );
        self.ctx.vkd.cmdDispatch(cb, n_heads, 1, 1);
    }
};
