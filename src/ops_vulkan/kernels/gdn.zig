//! The Gated DeltaNet kernel wrappers of the Qwen3.5 linear-attention layers: the decode step and
//! the chunk-parallel prefill.

const std = @import("std");
const vk = @import("vulkan");
const limits = @import("../../core/limits.zig");
const op_requires = @import("../../core/op_requires.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");

const spirv = @import("spirv.zig");
const max_workgroups_x = @import("pipeline.zig").max_workgroups_x;
const Pipeline = @import("pipeline.zig").Pipeline;
const allocateOnePrimaryCmdBuffer = @import("pipeline.zig").allocateOnePrimaryCmdBuffer;
const updateStorageBindings = @import("pipeline.zig").updateStorageBindings;
const createComputePipeline = @import("pipeline.zig").createComputePipeline;
const allocateOneSet = @import("pipeline.zig").allocateOneSet;
const HostBuffer = @import("memory.zig").HostBuffer;
const OneShot = @import("submit.zig").OneShot;

// -- types ------------------------------------------------------------------

/// Gated-DeltaNet decode kernel for one Qwen3.5 GDN layer. It mirrors `step()` in
/// `src/ops/gdn.zig`, and `run` is the standalone form of the self-test. `head_dim` must be
/// `limits.gdn_head_dim`, and `n_v_heads` must be at most `limits.gdn_max_v_per_k` whole multiples
/// of `n_k_heads`.
pub const GdnDecode = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    bufs: [11]HostBuffer,

    const n_bindings = 11;

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // The GPU `exp`, `sigmoid`, `silu`, and `inversesqrt` round differently
        // from the CPU intrinsics.
        .tolerance = .{ .relative = 1e-4 },
    };

    const Push = extern struct {
        d: u32,
        k: u32,
        n_k: u32,
        n_v: u32,
        eps: f32,
        n_tokens: u32 = 1,
        // The multi-slot form: the first batched row, and the element offsets of the slot region in
        // the state and conv buffers. All zero reproduces the single-slot dispatch bit for bit.
        row0: u32 = 0,
        state_off: u32 = 0,
        conv_off: u32 = 0,
    };

    pub const InitError = error{NoSuitableMemoryType} ||
        Pipeline.InitError || OneShot.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!GdnDecode {
        const pipeline = try Pipeline.init(ctx, &spirv.gdn_decode_spv, n_bindings, @sizeOf(Push));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, n_bindings);

        var bufs: [n_bindings]HostBuffer = undefined;
        for (&bufs) |*b| {
            b.* = .{ .ctx = ctx };
        }

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .bufs = bufs,
        };
    }

    pub fn deinit(self: *GdnDecode) void {
        for (&self.bufs) |*b| {
            b.deinit();
        }

        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    /// Record one single-token GDN decode dispatch, the fused-runner form. The caller owns the
    /// command buffer, and its set holds the 11 bindings of gdn_decode.comp on persistent device
    /// buffers.
    pub fn recordInto(
        self: *const GdnDecode,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_kernel: u32,
        eps: f32,
    ) void {
        self.recordChunk(cb, desc_set, n_k_heads, n_v_heads, head_dim, conv_kernel, eps, 1);
    }

    /// Record the GDN recurrence of a whole prefill chunk as one dispatch. The result is
    /// bit-identical to `n_tokens` per-token dispatches, and it runs the host dispatch and
    /// pipeline-barrier cost once. That is the prefill win.
    pub fn recordChunk(
        self: *const GdnDecode,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_kernel: u32,
        eps: f32,
        n_tokens: u32,
    ) void {
        self.recordRun(
            cb,
            desc_set,
            n_k_heads,
            n_v_heads,
            head_dim,
            conv_kernel,
            eps,
            n_tokens,
            0,
            0,
            0,
        );
    }

    /// Record one slot run of a mixed batch: `n_tokens` rows from batched row `row0`, on the state
    /// and conv regions of that slot at the given element offsets. Runs of different slots touch
    /// disjoint state and rows, so they need no barrier.
    pub fn recordRun(
        self: *const GdnDecode,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_kernel: u32,
        eps: f32,
        n_tokens: u32,
        row0: u32,
        state_off: u32,
        conv_off: u32,
    ) void {
        // Check the head shape: `vconv` of the shader holds one float per value head of a key head.
        std.debug.assert(head_dim == limits.gdn_head_dim);
        std.debug.assert(n_k_heads >= 1);
        std.debug.assert(n_v_heads % n_k_heads == 0);
        std.debug.assert(n_v_heads / n_k_heads <= limits.gdn_max_v_per_k);

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

        // Push the run and dispatch one workgroup per key head.
        const push: Push = .{
            .d = head_dim,
            .k = conv_kernel,
            .n_k = n_k_heads,
            .n_v = n_v_heads,
            .eps = eps,
            .n_tokens = n_tokens,
            .row0 = row0,
            .state_off = state_off,
            .conv_off = conv_off,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(Push),
            @ptrCast(&push),
        );
        self.ctx.vkd.cmdDispatch(cb, n_k_heads, 1, 1);
    }

    pub const RunError = error{ DimMismatch, NoSuitableMemoryType } ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError ||
        vk.DeviceWrapper.MapMemoryError ||
        vk.DeviceWrapper.BeginCommandBufferError ||
        vk.DeviceWrapper.EndCommandBufferError ||
        vk.DeviceWrapper.QueueSubmitError ||
        vk.DeviceWrapper.QueueWaitIdleError ||
        vk.DeviceWrapper.ResetCommandBufferError;

    /// Run one GDN decode step. `state` and `conv_state` are read in and
    /// written back. `o_out` takes the readout. The shapes and the tolerance
    /// match `gdn.step` in `src/ops/gdn.zig`.
    pub fn run(
        self: *GdnDecode,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_kernel: u32,
        qkv: []const f32,
        z: []const f32,
        alpha: []const f32,
        beta_in: []const f32,
        conv_w: []const f32,
        ssm_a: []const f32,
        dt_bias: []const f32,
        ssm_norm: []const f32,
        eps: f32,
        state: []f32,
        conv_state: []f32,
        o_out: []f32,
    ) RunError!void {
        // Check every slice against the head shape.
        const key_dim: usize = @as(usize, n_k_heads) * head_dim;
        const value_dim: usize = @as(usize, n_v_heads) * head_dim;
        const conv_dim: usize = 2 * key_dim + value_dim;
        if (head_dim != limits.gdn_head_dim or n_v_heads % n_k_heads != 0) return error.DimMismatch;
        if (n_v_heads / n_k_heads > limits.gdn_max_v_per_k) return error.DimMismatch;
        if (qkv.len != conv_dim or z.len != value_dim or o_out.len != value_dim) {
            return error.DimMismatch;
        }
        if (alpha.len != n_v_heads or beta_in.len != n_v_heads or ssm_a.len != n_v_heads or
            dt_bias.len != n_v_heads)
        {
            return error.DimMismatch;
        }
        if (conv_w.len != @as(usize, conv_kernel) * conv_dim) return error.DimMismatch;
        if (ssm_norm.len != head_dim) return error.DimMismatch;
        if (state.len != @as(usize, n_v_heads) * head_dim * head_dim) return error.DimMismatch;
        if (conv_state.len != @as(usize, conv_kernel - 1) * conv_dim) return error.DimMismatch;

        // Upload all inputs (bindings 0..10). state/conv_state are in/out.
        const inputs = [n_bindings][]const f32{
            qkv,
            z,
            alpha,
            beta_in,
            conv_w,
            ssm_a,
            dt_bias,
            ssm_norm,
            state,
            conv_state,
            o_out,
        };
        var infos: [n_bindings]vk.DescriptorBufferInfo = undefined;
        for (&self.bufs, inputs, 0..) |*buf, src, i| {
            const bytes: vk.DeviceSize = @intCast(src.len * @sizeOf(f32));
            try buf.ensure(bytes);
            if (i != 10) {
                @memcpy(buf.sliceTyped(f32, src.len), src); // o_out is write-only
            }

            infos[i] = .{ .buffer = buf.buffer, .offset = 0, .range = bytes };
        }

        updateStorageBindings(self.ctx, self.oneshot.desc_set, &infos);

        // Record the dispatch at the single-slot push defaults.
        const push: Push = .{
            .d = head_dim,
            .k = conv_kernel,
            .n_k = n_k_heads,
            .n_v = n_v_heads,
            .eps = eps,
        };

        try self.ctx.vkd.resetCommandBuffer(self.oneshot.cmd_buffer, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(self.oneshot.cmd_buffer, &begin);
        self.ctx.vkd.cmdBindPipeline(self.oneshot.cmd_buffer, .compute, self.pipeline.handle);
        const sets = [_]vk.DescriptorSet{self.oneshot.desc_set};
        self.ctx.vkd.cmdBindDescriptorSets(
            self.oneshot.cmd_buffer,
            .compute,
            self.pipeline.pipeline_layout,
            0,
            &sets,
            null,
        );
        self.ctx.vkd.cmdPushConstants(
            self.oneshot.cmd_buffer,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(Push),
            @ptrCast(&push),
        );
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, n_k_heads, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the state, the ring, and the readout back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(state, self.bufs[8].sliceTyped(f32, state.len));
        @memcpy(conv_state, self.bufs[9].sliceTyped(f32, conv_state.len));
        @memcpy(o_out, self.bufs[10].sliceTyped(f32, o_out.len));
    }

    /// Run `n_tokens` tokens through one `recordChunk` dispatch. The batched buffers (qkv, z,
    /// alpha, beta_in, o_out) hold n_tokens rows each, and the weights, the state, and conv_state
    /// hold one. The self-test compares it with n_tokens CPU `gdn.step` calls.
    pub fn runChunk(
        self: *GdnDecode,
        n_tokens: u32,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_kernel: u32,
        qkv: []const f32, // [n_tokens*conv_dim]
        z: []const f32, // [n_tokens*value_dim]
        alpha: []const f32, // [n_tokens*n_v]
        beta_in: []const f32, // [n_tokens*n_v]
        conv_w: []const f32,
        ssm_a: []const f32,
        dt_bias: []const f32,
        ssm_norm: []const f32,
        eps: f32,
        state: []f32,
        conv_state: []f32,
        o_out: []f32, // [n_tokens*value_dim]
    ) RunError!void {
        // Check every slice against the head shape.
        const t: usize = n_tokens;
        const key_dim: usize = @as(usize, n_k_heads) * head_dim;
        const value_dim: usize = @as(usize, n_v_heads) * head_dim;
        const conv_dim: usize = 2 * key_dim + value_dim;
        if (head_dim != limits.gdn_head_dim or n_v_heads % n_k_heads != 0) return error.DimMismatch;
        if (n_v_heads / n_k_heads > limits.gdn_max_v_per_k) return error.DimMismatch;
        if (qkv.len != t * conv_dim or z.len != t * value_dim or o_out.len != t * value_dim) {
            return error.DimMismatch;
        }
        if (alpha.len != t * n_v_heads or beta_in.len != t * n_v_heads) return error.DimMismatch;
        if (ssm_a.len != n_v_heads or dt_bias.len != n_v_heads) return error.DimMismatch;
        if (conv_w.len != @as(usize, conv_kernel) * conv_dim) return error.DimMismatch;
        if (ssm_norm.len != head_dim) return error.DimMismatch;
        if (state.len != value_dim * head_dim) return error.DimMismatch;
        if (conv_state.len != @as(usize, conv_kernel - 1) * conv_dim) return error.DimMismatch;

        // Upload the inputs, as in `run`.
        const inputs = [n_bindings][]const f32{
            qkv,
            z,
            alpha,
            beta_in,
            conv_w,
            ssm_a,
            dt_bias,
            ssm_norm,
            state,
            conv_state,
            o_out,
        };
        var infos: [n_bindings]vk.DescriptorBufferInfo = undefined;
        for (&self.bufs, inputs, 0..) |*buf, src, i| {
            const bytes: vk.DeviceSize = @intCast(src.len * @sizeOf(f32));
            try buf.ensure(bytes);
            if (i != 10) {
                @memcpy(buf.sliceTyped(f32, src.len), src); // o_out is write-only
            }

            infos[i] = .{ .buffer = buf.buffer, .offset = 0, .range = bytes };
        }

        updateStorageBindings(self.ctx, self.oneshot.desc_set, &infos);

        // Record one dispatch over the whole chunk.
        try self.ctx.vkd.resetCommandBuffer(self.oneshot.cmd_buffer, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(self.oneshot.cmd_buffer, &begin);
        self.recordChunk(
            self.oneshot.cmd_buffer,
            self.oneshot.desc_set,
            n_k_heads,
            n_v_heads,
            head_dim,
            conv_kernel,
            eps,
            n_tokens,
        );
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        // Submit, wait, and read the state, the ring, and the readout back.
        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(state, self.bufs[8].sliceTyped(f32, state.len));
        @memcpy(conv_state, self.bufs[9].sliceTyped(f32, conv_state.len));
        @memcpy(o_out, self.bufs[10].sliceTyped(f32, o_out.len));
    }
};

/// Chunked-parallel GDN prefill in three passes: conv, scan, and norm. `head_dim` must be
/// `limits.gdn_head_dim`. It replaces the serial-over-tokens `GdnDecode.recordChunk`, at a
/// tolerance, because the scan reorders the sums.
pub const GdnChunkParallel = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline_conv: Pipeline,
    pipeline_scan: Pipeline,
    // The norm pass shares the layout of the scan pass, so one descriptor set
    // serves both dispatches.
    norm_module: vk.ShaderModule,
    pipeline_norm: vk.Pipeline,
    // Resources for the standalone gate path only.
    desc_pool: vk.DescriptorPool,
    conv_set: vk.DescriptorSet,
    scan_set: vk.DescriptorSet,
    cmd_buffer: vk.CommandBuffer,
    bufs: [n_bufs]HostBuffer,

    // Host-buffer indices (gate path): conv reads 0..2 + writes 3..5. Scan reads
    // 3..11 + state(12) in/out + writes o_out(13).
    const n_bufs = 14;
    const conv_bindings = 6;
    const scan_bindings = 11;

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 1e-4 },
    };

    const ConvPush = extern struct { d: u32, n_k: u32, n_v: u32, k: u32, n_tokens: u32, eps: f32 };
    const ScanPush = extern struct { d: u32, n_k: u32, n_v: u32, n_tokens: u32, eps: f32 };

    pub const InitError = GdnDecode.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!GdnChunkParallel {
        // Build the conv and scan pipelines.
        const pipeline_conv = try Pipeline.init(
            ctx,
            &spirv.gdn_chunk_conv_spv,
            conv_bindings,
            @sizeOf(ConvPush),
        );
        errdefer pipeline_conv.deinit(ctx);

        const pipeline_scan = try Pipeline.init(
            ctx,
            &spirv.gdn_chunk_scan_spv,
            scan_bindings,
            @sizeOf(ScanPush),
        );
        errdefer pipeline_scan.deinit(ctx);

        // Build the norm pipeline on the scan layout.
        const norm_info: vk.ShaderModuleCreateInfo = .{
            .code_size = spirv.gdn_chunk_norm_spv.len,
            .p_code = @ptrCast(&spirv.gdn_chunk_norm_spv),
        };
        const norm_module = try ctx.vkd.createShaderModule(ctx.device, &norm_info, null);
        errdefer ctx.vkd.destroyShaderModule(ctx.device, norm_module, null);

        const pipeline_norm = try createComputePipeline(
            ctx,
            norm_module,
            pipeline_scan.pipeline_layout,
        );
        errdefer ctx.vkd.destroyPipeline(ctx.device, pipeline_norm, null);

        // Pool for the two standalone (gate) sets.
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .storage_buffer, .descriptor_count = conv_bindings + scan_bindings },
        };
        const pool_info: vk.DescriptorPoolCreateInfo = .{
            .max_sets = 2,
            .pool_size_count = 1,
            .p_pool_sizes = &pool_sizes,
        };
        const desc_pool = try ctx.vkd.createDescriptorPool(ctx.device, &pool_info, null);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, desc_pool, null);

        // Allocate the gate sets, the gate command buffer, and the empty host buffers.
        const conv_set = try allocateOneSet(ctx, desc_pool, pipeline_conv.set_layout);
        const scan_set = try allocateOneSet(ctx, desc_pool, pipeline_scan.set_layout);
        const cmd_buffer = try allocateOnePrimaryCmdBuffer(ctx);

        var bufs: [n_bufs]HostBuffer = undefined;
        for (&bufs) |*b| {
            b.* = .{ .ctx = ctx };
        }

        return .{
            .ctx = ctx,
            .pipeline_conv = pipeline_conv,
            .pipeline_scan = pipeline_scan,
            .norm_module = norm_module,
            .pipeline_norm = pipeline_norm,
            .desc_pool = desc_pool,
            .conv_set = conv_set,
            .scan_set = scan_set,
            .cmd_buffer = cmd_buffer,
            .bufs = bufs,
        };
    }

    pub fn deinit(self: *GdnChunkParallel) void {
        for (&self.bufs) |*b| {
            b.deinit();
        }

        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        self.ctx.vkd.freeCommandBuffers(self.ctx.device, self.ctx.command_pool, &cmds);
        self.ctx.vkd.destroyDescriptorPool(self.ctx.device, self.desc_pool, null);
        self.ctx.vkd.destroyPipeline(self.ctx.device, self.pipeline_norm, null);
        self.ctx.vkd.destroyShaderModule(self.ctx.device, self.norm_module, null);
        self.pipeline_scan.deinit(self.ctx);
        self.pipeline_conv.deinit(self.ctx);
    }

    /// Record the conv pass, one workgroup per KEY head. The caller binds `set`
    /// with 6 bindings: qkv, conv_w, conv_state (in and out), q_n, k_n, v_conv.
    pub fn recordConv(
        self: *const GdnChunkParallel,
        cb: vk.CommandBuffer,
        set: vk.DescriptorSet,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_kernel: u32,
        n_tokens: u32,
        eps: f32,
    ) void {
        // Check the head counts.
        std.debug.assert(head_dim == limits.gdn_head_dim);
        std.debug.assert(n_k_heads >= 1);
        std.debug.assert(n_v_heads % n_k_heads == 0);

        // Bind the pipeline and the descriptors.
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline_conv.handle);
        const sets = [_]vk.DescriptorSet{set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline_conv.pipeline_layout,
            0,
            &sets,
            null,
        );

        // Push the constants and dispatch.
        const push: ConvPush = .{
            .d = head_dim,
            .n_k = n_k_heads,
            .n_v = n_v_heads,
            .k = conv_kernel,
            .n_tokens = n_tokens,
            .eps = eps,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline_conv.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(ConvPush),
            @ptrCast(&push),
        );
        self.ctx.vkd.cmdDispatch(cb, n_k_heads, 1, 1);
    }

    /// Record the scan pass, a barrier, and the norm pass. The caller binds `set` with the 11
    /// bindings of gdn_chunk_scan.comp. It also records the barrier between `recordConv` and this
    /// call, and the barrier after this call.
    pub fn recordScan(
        self: *const GdnChunkParallel,
        cb: vk.CommandBuffer,
        set: vk.DescriptorSet,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        n_tokens: u32,
        eps: f32,
    ) void {
        // Check the head shape and the grid.
        std.debug.assert(head_dim == limits.gdn_head_dim);
        std.debug.assert(n_k_heads >= 1);
        std.debug.assert(n_v_heads % n_k_heads == 0);
        std.debug.assert(n_tokens >= 1);
        // Vulkan guarantees `max_workgroups_x` on the y axis too, where the
        // norm pass puts the token.
        std.debug.assert(n_tokens <= max_workgroups_x);

        const groups = limits.gdnScanGroups(n_v_heads);
        std.debug.assert(groups <= max_workgroups_x);
        std.debug.assert(
            limits.gdnScanLdsBytes() <=
                self.ctx.physical_device_props.limits.max_compute_shared_memory_size,
        );
        // The last lane of the last workgroup owns the last column of the last head.
        const last = limits.gdnScanCell(groups - 1, limits.gdn_scan_cols - 1);
        std.debug.assert(last.head == n_v_heads - 1);
        std.debug.assert(last.column == head_dim - 1);

        // Bind the scan pipeline and the descriptor set.
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline_scan.handle);
        const sets = [_]vk.DescriptorSet{set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.pipeline_scan.pipeline_layout,
            0,
            &sets,
            null,
        );

        // Push the chunk shape and dispatch the scan.
        const push: ScanPush = .{
            .d = head_dim,
            .n_k = n_k_heads,
            .n_v = n_v_heads,
            .n_tokens = n_tokens,
            .eps = eps,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline_scan.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(ScanPush),
            @ptrCast(&push),
        );
        self.ctx.vkd.cmdDispatch(cb, groups, 1, 1);

        // The norm pass reads the raw readout of every column of a head, and
        // it then writes the same elements in place.
        const mb = [_]vk.MemoryBarrier{.{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true },
        }};
        self.ctx.vkd.cmdPipelineBarrier(
            cb,
            .{ .compute_shader_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            &mb,
            null,
            null,
        );

        // The norm pipeline uses the scan layout, so the set and the push
        // constants stay bound across the pipeline change.
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.pipeline_norm);
        self.ctx.vkd.cmdDispatch(cb, n_v_heads, n_tokens, 1);
    }

    pub const RunError = GdnDecode.RunError;

    /// Run the whole chunked prefill over `n_tokens`, and read state, conv_state, and o_out back.
    /// It takes the inputs and writes the outputs of `GdnDecode.runChunk`, so the self-test
    /// compares both against the CPU `step` run n_tokens times.
    pub fn runChunkParallel(
        self: *GdnChunkParallel,
        n_tokens: u32,
        n_k_heads: u32,
        n_v_heads: u32,
        head_dim: u32,
        conv_kernel: u32,
        qkv: []const f32,
        z: []const f32,
        alpha: []const f32,
        beta_in: []const f32,
        conv_w: []const f32,
        ssm_a: []const f32,
        dt_bias: []const f32,
        ssm_norm: []const f32,
        eps: f32,
        state: []f32,
        conv_state: []f32,
        o_out: []f32,
    ) RunError!void {
        // Check every slice against the head shape.
        const t: usize = n_tokens;
        const key_dim: usize = @as(usize, n_k_heads) * head_dim;
        const value_dim: usize = @as(usize, n_v_heads) * head_dim;
        const conv_dim: usize = 2 * key_dim + value_dim;
        if (head_dim != limits.gdn_head_dim or n_v_heads % n_k_heads != 0) return error.DimMismatch;
        if (n_tokens == 0 or n_tokens > max_workgroups_x) return error.DimMismatch;
        if (qkv.len != t * conv_dim or z.len != t * value_dim or o_out.len != t * value_dim) {
            return error.DimMismatch;
        }
        if (alpha.len != t * n_v_heads or beta_in.len != t * n_v_heads) return error.DimMismatch;
        if (ssm_a.len != n_v_heads or dt_bias.len != n_v_heads) return error.DimMismatch;
        if (conv_w.len != @as(usize, conv_kernel) * conv_dim) return error.DimMismatch;
        if (ssm_norm.len != head_dim) return error.DimMismatch;
        if (state.len != value_dim * head_dim) return error.DimMismatch;
        if (conv_state.len != @as(usize, conv_kernel - 1) * conv_dim) return error.DimMismatch;

        // Buffer sizes by index (q_n/k_n/v_conv intermediate, o_out write-only).
        const lens = [n_bufs]usize{
            qkv.len,
            conv_w.len,
            conv_state.len,
            t * key_dim,
            t * key_dim,
            t * value_dim,
            alpha.len,
            beta_in.len,
            ssm_a.len,
            dt_bias.len,
            ssm_norm.len,
            z.len,
            state.len,
            o_out.len,
        };

        const uploads = [n_bufs]?[]const f32{
            qkv,
            conv_w,
            conv_state,
            null,
            null,
            null,
            alpha,
            beta_in,
            ssm_a,
            dt_bias,
            ssm_norm,
            z,
            state,
            null,
        };

        // Size each buffer and upload the inputs.
        var infos: [n_bufs]vk.DescriptorBufferInfo = undefined;
        for (&self.bufs, lens, uploads, 0..) |*buf, len, up, i| {
            const bytes: vk.DeviceSize = @intCast(len * @sizeOf(f32));
            try buf.ensure(bytes);
            if (up) |src| {
                @memcpy(buf.sliceTyped(f32, len), src);
            }

            infos[i] = .{ .buffer = buf.buffer, .offset = 0, .range = bytes };
        }

        // Bind the conv set: qkv(0), conv_w(1), conv_state(2), q_n(3), k_n(4), v_conv(5).
        updateStorageBindings(self.ctx, self.conv_set, infos[0..6]);

        // Bind the scan set: q_n(3), k_n(4), v_conv(5), alpha(6), beta(7), ssm_a(8), dt_bias(9),
        // ssm_norm(10), z(11), state(12), o_out(13).
        const scan_infos = [scan_bindings]vk.DescriptorBufferInfo{
            infos[3],
            infos[4],
            infos[5],
            infos[6],
            infos[7],
            infos[8],
            infos[9],
            infos[10],
            infos[11],
            infos[12],
            infos[13],
        };
        updateStorageBindings(self.ctx, self.scan_set, &scan_infos);

        // Record the conv pass, a barrier, and the scan and norm passes.
        try self.ctx.vkd.resetCommandBuffer(self.cmd_buffer, .{});
        try self.ctx.vkd.beginCommandBuffer(
            self.cmd_buffer,
            &.{ .flags = .{ .one_time_submit_bit = true } },
        );
        self.recordConv(
            self.cmd_buffer,
            self.conv_set,
            n_k_heads,
            n_v_heads,
            head_dim,
            conv_kernel,
            n_tokens,
            eps,
        );

        const mb = [_]vk.MemoryBarrier{.{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
        }};
        self.ctx.vkd.cmdPipelineBarrier(
            self.cmd_buffer,
            .{ .compute_shader_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            &mb,
            null,
            null,
        );

        self.recordScan(
            self.cmd_buffer,
            self.scan_set,
            n_k_heads,
            n_v_heads,
            head_dim,
            n_tokens,
            eps,
        );
        try self.ctx.vkd.endCommandBuffer(self.cmd_buffer);

        // Submit, wait, and read the state, the ring, and the readout back.
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        try self.ctx.vkd.queueSubmit(
            self.ctx.queue,
            &[_]vk.SubmitInfo{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }},
            .null_handle,
        );
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(conv_state, self.bufs[2].sliceTyped(f32, conv_state.len));
        @memcpy(state, self.bufs[12].sliceTyped(f32, state.len));
        @memcpy(o_out, self.bufs[13].sliceTyped(f32, o_out.len));
    }
};
