//! The rotary position kernel wrappers: one vector, every head of one row, and a batch of rows.

const std = @import("std");
const vk = @import("vulkan");
const op_requires = @import("../../core/op_requires.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");
const RopeStyle = @import("../../ops/naive.zig").RopeStyle;

const spirv = @import("spirv.zig");
const Pipeline = @import("pipeline.zig").Pipeline;
const updateStorageBindings = @import("pipeline.zig").updateStorageBindings;
const HostBuffer = @import("memory.zig").HostBuffer;
const OneShot = @import("submit.zig").OneShot;
const VecAdd = @import("elementwise.zig").VecAdd;
const ResidualAdd = @import("elementwise.zig").ResidualAdd;

// -- types ------------------------------------------------------------------

/// In-place RoPE rotation of one head_dim vector. `rope_dim` must be even and at most `x.len`. Only
/// the benchmark calls this kernel.
pub const Rope = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_x: HostBuffer,
    buf_ff: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // The observed relative error is 1.6e-5 at head_dim 64, so 5e-5 leaves headroom.
        .tolerance = .{ .relative = 5e-5 },
    };

    const PushConsts = extern struct { pos: u32, rope_dim: u32, freq_base: f32, style_norm: u32 };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!Rope {
        const pipeline = try Pipeline.init(ctx, &spirv.rope_spv, 2, @sizeOf(PushConsts));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 2);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .buf_x = .{ .ctx = ctx },
            .buf_ff = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *Rope) void {
        self.buf_x.deinit();
        self.buf_ff.deinit();
        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const RunError = ResidualAdd.RunError || error{RopeDimNotEven};

    pub fn run(
        self: *Rope,
        x: []f32,
        pos: u32,
        rope_dim: usize,
        freq_base: f32,
        style: RopeStyle,
        freq_factors: ?[]const f32,
    ) RunError!void {
        if (rope_dim > x.len) return error.LengthMismatch;
        if (rope_dim % 2 != 0) return error.RopeDimNotEven;

        // Stage x and the frequency factors.
        const half = rope_dim / 2;
        if (freq_factors) |ff| std.debug.assert(ff.len >= half);

        const byte_size: vk.DeviceSize = @intCast(x.len * @sizeOf(f32));
        const ff_bytes: vk.DeviceSize = @intCast(half * @sizeOf(f32));

        try self.buf_x.ensure(byte_size);
        @memcpy(self.buf_x.sliceTyped(f32, x.len), x);
        try self.buf_ff.ensure(ff_bytes);
        const ff_dst = self.buf_ff.sliceTyped(f32, half);
        if (freq_factors) |ff| {
            @memcpy(ff_dst, ff[0..half]);
        } else {
            @memset(ff_dst, 1.0);
        }

        // Bind both buffers.
        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_ff.buffer, .offset = 0, .range = ff_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

        // Record the rotation.
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

        // Push the rotation parameters and dispatch.
        const pc: PushConsts = .{
            .pos = pos,
            .rope_dim = @intCast(rope_dim),
            .freq_base = freq_base,
            .style_norm = @intFromBool(style == .norm),
        };
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        const group_count_x: u32 = (@as(u32, @intCast(half)) + 63) / 64;
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, group_count_x, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the result back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(x, self.buf_x.sliceTyped(f32, x.len));
    }
};

/// The rotation of `Rope` on every head of an `(n_heads, head_dim)` buffer in one dispatch, 2
/// dispatches per layer on a Llama-class model against 40 per-head ones. `rope_dim` must be even.
/// Every caller records into its own command buffer, so `recordInto` is the only entry point.
pub const RopeMulti = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 1e-5 },
    };

    const PushConsts = extern struct {
        pos: u32,
        head_dim: u32,
        rope_dim: u32,
        freq_base: f32,
        style_norm: u32,
    };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!RopeMulti {
        const pipeline = try Pipeline.init(ctx, &spirv.rope_multi_spv, 2, @sizeOf(PushConsts));

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *RopeMulti) void {
        self.pipeline.deinit(self.ctx);
    }

    /// Record one multi-head rope dispatch. The caller binds the descriptor set.
    /// Binding 0 is the `(n_heads * head_dim)` x buffer. Binding 1 holds the
    /// rope_dim/2 frequency-factor divisors, all ones when the model has none.
    pub fn recordInto(
        self: *RopeMulti,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        pos: u32,
        n_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        freq_base: f32,
        style: RopeStyle,
    ) void {
        // Bind the pipeline and the descriptors.
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

        // Push the constants and dispatch.
        const pc: PushConsts = .{
            .pos = pos,
            .head_dim = head_dim,
            .rope_dim = rope_dim,
            .freq_base = freq_base,
            .style_norm = @intFromBool(style == .norm),
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        const half: u32 = rope_dim / 2;
        const group_count_x: u32 = (half + 63) / 64;
        self.ctx.vkd.cmdDispatch(cb, group_count_x, n_heads, 1);
    }
};

/// Batched multi-head RoPE over a `(B, n_heads, head_dim)` buffer. Row `bi` rotates at position
/// `start_pos + bi * pos_stride`.
pub const RopeMultiBatch = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 1e-5 },
    };

    const PushConsts = extern struct {
        start_pos: u32,
        head_dim: u32,
        rope_dim: u32,
        freq_base: f32,
        n_heads: u32,
        pos_stride: u32,
        style_norm: u32,
        row_base: u32,
    };
    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!RopeMultiBatch {
        const pipeline = try Pipeline.init(
            ctx,
            &spirv.rope_multi_batch_spv,
            2,
            @sizeOf(PushConsts),
        );

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *RopeMultiBatch) void {
        self.pipeline.deinit(self.ctx);
    }

    /// `pos_stride` is 1 for prefill and 0 for a lockstep or single-row dispatch, where every row
    /// uses `start_pos`. `row_base` offsets the buffer rows of the dispatch, so arbitrary per-row
    /// positions record as B=1 dispatches per row.
    pub fn recordInto(
        self: *RopeMultiBatch,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        start_pos: u32,
        n_heads: u32,
        head_dim: u32,
        rope_dim: u32,
        freq_base: f32,
        b: u32,
        pos_stride: u32,
        row_base: u32,
        style: RopeStyle,
    ) void {
        // Bind the pipeline and the descriptor set.
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

        // Push the row geometry and dispatch every (head, row) pair.
        const pc: PushConsts = .{
            .start_pos = start_pos,
            .head_dim = head_dim,
            .rope_dim = rope_dim,
            .freq_base = freq_base,
            .n_heads = n_heads,
            .pos_stride = pos_stride,
            .style_norm = @intFromBool(style == .norm),
            .row_base = row_base,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        const half: u32 = rope_dim / 2;
        const group_count_x: u32 = (half + 63) / 64;
        self.ctx.vkd.cmdDispatch(cb, group_count_x, n_heads, b);
    }
};
