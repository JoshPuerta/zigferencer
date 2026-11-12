//! The attention kernel wrappers, the attention kernel set that `-Dattn-kernel` selects, and the
//! f16 commit of the KV rows.
//!
//! Comptime. `T` picks the KV dtype of a typed view. An assert block keeps one tolerance across
//! the kernel set and its resident kernels.
//!
//! Attention tolerance. The measured relative error is 5e-4 to 6e-4 on SmolLM2, Llama-1B, and
//! Llama-8B shapes, with a near-zero-denominator floor of 1e-5 (see `computeErrors` in
//! tools/bench.zig). The declared 1e-3 leaves a 1.5x cushion. tests/vulkan_attention_test.zig gates
//! `Attention` on the SmolLM2 shape at 128 positions. `gpu-selftest attn` gates it on the Qwen3.5
//! shape at 7 and `limits.attn_split_len` positions, at both KV types.

const std = @import("std");
const vk = @import("vulkan");
const build_options = @import("build_options");
const attn_split = @import("../../core/attn_split.zig");
const dtype = @import("../../core/dtype.zig");
const limits = @import("../../core/limits.zig");
const op_requires = @import("../../core/op_requires.zig");
const layer_op = @import("../../core/layer_op.zig");
const gpu_profile = @import("../../runtime/gpu_profile.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");
const log = std.log.scoped(.gpu);

const spirv = @import("spirv.zig");
const max_workgroups_x = @import("pipeline.zig").max_workgroups_x;
const Pipeline = @import("pipeline.zig").Pipeline;
const updateStorageBindings = @import("pipeline.zig").updateStorageBindings;
const createSingleSetPool = @import("pipeline.zig").createSingleSetPool;
const allocateOneSet = @import("pipeline.zig").allocateOneSet;
const HostBuffer = @import("memory.zig").HostBuffer;
const OneShot = @import("submit.zig").OneShot;
const Recorded = @import("submit.zig").Recorded;
const submitOnce = @import("submit.zig").submitOnce;
const VecAdd = @import("elementwise.zig").VecAdd;
const ResidualAdd = @import("elementwise.zig").ResidualAdd;

// -- types ------------------------------------------------------------------

/// Fused single-token attention. `t_len` must be at least 1 and at most `limits.max_model_ctx`, and
/// `n_kv_heads` must divide `n_heads`. `recordInto` takes a row of at most `limits.attn_split_len`
/// positions. `run` takes any depth, and it sends a longer row through `AttentionSplit`.
pub const Attention = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    /// The KV-cache element type of the pipelines, which `runTyped` checks.
    kv: dtype.KvDtype,
    /// The split path of `runTyped`, built for its first split row. The fused runners never build
    /// it.
    split: ?SplitState,
    buf_q: HostBuffer,
    buf_k: HostBuffer,
    buf_v: HostBuffer,
    buf_out: HostBuffer,
    buf_scratch: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 1e-3 },
    };

    const PushConsts = extern struct {
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
    };

    /// The split path of `runTyped`: the two passes, and one pool with the
    /// split set and the merge set. The scratch of one row is `buf_scratch`.
    /// Both paths record into `oneshot.cmd_buffer`.
    const SplitState = struct {
        passes: AttentionSplit,
        pool: vk.DescriptorPool,
        split_set: vk.DescriptorSet,
        merge_set: vk.DescriptorSet,

        const BuildError = AttentionSplit.InitError ||
            vk.DeviceWrapper.CreateDescriptorPoolError ||
            vk.DeviceWrapper.AllocateDescriptorSetsError;

        fn init(ctx: *const vulkan_ctx.Context, kv: dtype.KvDtype) BuildError!SplitState {
            var passes = try AttentionSplit.initKv(ctx, kv);
            errdefer passes.deinit();

            const pool_sizes = [_]vk.DescriptorPoolSize{.{
                .type = .storage_buffer,
                .descriptor_count = AttentionSplit.split_bindings + AttentionSplit.merge_bindings,
            }};
            const pool_info: vk.DescriptorPoolCreateInfo = .{
                .max_sets = 2,
                .pool_size_count = 1,
                .p_pool_sizes = &pool_sizes,
            };
            const pool = try ctx.vkd.createDescriptorPool(ctx.device, &pool_info, null);
            errdefer ctx.vkd.destroyDescriptorPool(ctx.device, pool, null);

            const split_set = try allocateOneSet(ctx, pool, passes.split.set_layout);
            const merge_set = try allocateOneSet(ctx, pool, passes.merge.set_layout);

            return .{
                .passes = passes,
                .pool = pool,
                .split_set = split_set,
                .merge_set = merge_set,
            };
        }

        fn deinit(self: *SplitState, ctx: *const vulkan_ctx.Context) void {
            // Destroying the pool frees the split set and the merge set.
            ctx.vkd.destroyDescriptorPool(ctx.device, self.pool, null);
            self.passes.deinit();
        }
    };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!Attention {
        return initKv(ctx, .f32);
    }

    /// Build the resident pipeline for the KV-cache element type of the process. `build.zig`
    /// compiles the f16 variant from the same source, so one selector covers the difference.
    /// `runTyped` builds the split passes for the same type at its first split row.
    pub fn initKv(ctx: *const vulkan_ctx.Context, kv: dtype.KvDtype) InitError!Attention {
        const spv: []align(@alignOf(u32)) const u8 = switch (kv) {
            .f32 => &spirv.attention_spv,
            .f16 => &spirv.attention_f16_spv,
        };

        const pipeline = try Pipeline.init(ctx, spv, 4, @sizeOf(PushConsts));
        errdefer pipeline.deinit(ctx);

        const oneshot = try OneShot.init(ctx, pipeline.set_layout, 4);

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
            .oneshot = oneshot,
            .kv = kv,
            .split = null,
            .buf_q = .{ .ctx = ctx },
            .buf_k = .{ .ctx = ctx },
            .buf_v = .{ .ctx = ctx },
            .buf_out = .{ .ctx = ctx },
            .buf_scratch = .{ .ctx = ctx },
        };
    }

    pub fn deinit(self: *Attention) void {
        self.buf_q.deinit();
        self.buf_k.deinit();
        self.buf_v.deinit();
        self.buf_out.deinit();
        self.buf_scratch.deinit();
        if (self.split) |*state| {
            state.deinit(self.ctx);
        }

        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const RunError = ResidualAdd.RunError || SplitState.BuildError;

    /// Record one fused-attention dispatch into a command buffer the caller owns. The caller binds
    /// the set (q, k_cache, v_cache, out) and keeps `t_len` within `limits.attn_split_len`. A
    /// deeper row takes the split passes through the attention kernel set.
    pub fn recordInto(
        self: *Attention,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
    ) void {
        std.debug.assert(t_len >= 1);
        std.debug.assert(t_len <= limits.max_model_ctx);
        std.debug.assert(!attn_split.usesSplit(t_len));
        std.debug.assert(n_kv_heads >= 1);
        std.debug.assert(n_heads % n_kv_heads == 0);
        std.debug.assert(head_dim >= 1);

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

        // Push the head shape and dispatch one workgroup per head.
        const pc: PushConsts = .{
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .t_len = t_len,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        self.ctx.vkd.cmdDispatch(cb, n_heads, 1, 1);
    }

    /// Run one row against f32 caches.
    pub fn run(
        self: *Attention,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
        out: []f32,
    ) RunError!void {
        return self.runTyped(f32, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, t_len, out);
    }

    /// Run one row against caches of element type `T`, which must match the KV dtype of the
    /// pipelines. A row past `limits.attn_split_len` positions runs the split pass, a shader
    /// barrier, and the merge pass. Either way the row takes one submission.
    pub fn runTyped(
        self: *Attention,
        comptime T: type,
        q: []const f32,
        k_cache: []const T,
        v_cache: []const T,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
        out: []f32,
    ) RunError!void {
        const kv: dtype.KvDtype = comptime switch (T) {
            f32 => .f32,
            f16 => .f16,
            else => @compileError("Attention.runTyped: the KV element type must be f32 or f16"),
        };

        // Check the row shape.
        std.debug.assert(self.kv == kv);
        std.debug.assert(t_len >= 1);
        std.debug.assert(t_len <= limits.max_model_ctx);
        std.debug.assert(n_kv_heads >= 1);
        std.debug.assert(n_heads % n_kv_heads == 0);
        std.debug.assert(q.len == n_heads * head_dim);
        std.debug.assert(out.len >= q.len);
        // The whole row is the one stream of the split pass, and the shaders index it in 32 bits.
        const row_elems = @as(u64, t_len) * n_kv_heads * head_dim;
        std.debug.assert(row_elems <= limits.max_kv_index_elems);
        std.debug.assert(k_cache.len >= row_elems);
        std.debug.assert(v_cache.len >= row_elems);

        // Build the split passes at the first deep row.
        const deep = attn_split.usesSplit(t_len);
        if (deep and self.split == null) {
            self.split = try SplitState.init(self.ctx, self.kv);
        }

        // Size the bindings and stage the inputs.
        const q_bytes: vk.DeviceSize = @intCast(q.len * @sizeOf(f32));
        // The f16 shaders read words, so each binding covers whole words. The split pass reads K in
        // 16-byte loads, so the K binding covers whole loads. No shader reads the padding.
        const k_bytes: vk.DeviceSize = std.mem.alignForward(
            vk.DeviceSize,
            k_cache.len * @sizeOf(T),
            16,
        );
        const v_bytes: vk.DeviceSize = std.mem.alignForward(
            vk.DeviceSize,
            v_cache.len * @sizeOf(T),
            4,
        );
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_q.ensure(q_bytes);
        try self.buf_k.ensure(k_bytes);
        try self.buf_v.ensure(v_bytes);
        try self.buf_out.ensure(out_bytes);
        @memcpy(self.buf_q.sliceTyped(f32, q.len), q);
        @memcpy(self.buf_k.sliceTyped(T, k_cache.len), k_cache);
        @memcpy(self.buf_v.sliceTyped(T, v_cache.len), v_cache);

        // Describe the four buffers.
        const q_info: vk.DescriptorBufferInfo = .{
            .buffer = self.buf_q.buffer,
            .offset = 0,
            .range = q_bytes,
        };
        const k_info: vk.DescriptorBufferInfo = .{
            .buffer = self.buf_k.buffer,
            .offset = 0,
            .range = k_bytes,
        };
        const v_info: vk.DescriptorBufferInfo = .{
            .buffer = self.buf_v.buffer,
            .offset = 0,
            .range = v_bytes,
        };
        const out_info: vk.DescriptorBufferInfo = .{
            .buffer = self.buf_out.buffer,
            .offset = 0,
            .range = out_bytes,
        };

        // Bind the split sets or the resident set. A record offset does not depend on the row
        // count, so a one-row scratch has the offsets of the first row of a runner scratch.
        const scratch: attn_split.Scratch = .{
            .rows = 1,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .max_splits = attn_split.splitCount(t_len),
        };

        if (deep) {
            const state = &self.split.?;
            const scratch_bytes: vk.DeviceSize = scratch.totalBytes();
            try self.buf_scratch.ensure(scratch_bytes);
            const scratch_info: vk.DescriptorBufferInfo = .{
                .buffer = self.buf_scratch.buffer,
                .offset = 0,
                .range = scratch_bytes,
            };
            AttentionSplit.assertKBinding(k_info);
            AttentionSplit.assertVBinding(v_info);
            const split_infos = [_]vk.DescriptorBufferInfo{ q_info, k_info, v_info, scratch_info };
            const merge_infos = [_]vk.DescriptorBufferInfo{ scratch_info, out_info };
            updateStorageBindings(self.ctx, state.split_set, &split_infos);
            updateStorageBindings(self.ctx, state.merge_set, &merge_infos);
        } else {
            const buf_infos = [_]vk.DescriptorBufferInfo{ q_info, k_info, v_info, out_info };
            updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);
        }

        // Record the split pass, a barrier, and the merge pass, or the resident kernel.
        const cb = self.oneshot.cmd_buffer;
        try self.ctx.vkd.resetCommandBuffer(cb, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(cb, &begin);
        if (deep) {
            const state = &self.split.?;
            state.passes.recordSplit(
                cb,
                state.split_set,
                n_heads,
                n_kv_heads,
                head_dim,
                t_len,
                @intCast(row_elems),
                0,
                0,
                0,
                scratch.max_splits,
            );

            // The merge reads the records that the split pass wrote.
            const records_ready = [_]vk.MemoryBarrier{.{
                .src_access_mask = .{ .shader_write_bit = true },
                .dst_access_mask = .{ .shader_read_bit = true },
            }};
            self.ctx.vkd.cmdPipelineBarrier(
                cb,
                .{ .compute_shader_bit = true },
                .{ .compute_shader_bit = true },
                .{},
                &records_ready,
                null,
                null,
            );

            state.passes.recordMerge(
                cb,
                state.merge_set,
                n_heads,
                head_dim,
                t_len,
                0,
                scratch.max_splits,
                0,
            );
        } else {
            self.recordInto(cb, self.oneshot.desc_set, n_heads, n_kv_heads, head_dim, t_len);
        }

        try self.ctx.vkd.endCommandBuffer(cb);

        // Submit, wait, and read the output back.
        const cmds = [_]vk.CommandBuffer{cb};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// Split-depth decode attention for a row past `limits.attn_split_len` positions. The kernel is
/// record-only: the caller owns the scratch, the descriptor sets, and the shader barrier between
/// the two passes. One `t_len` gives one result bit for bit.
pub const AttentionSplit = struct {
    ctx: *const vulkan_ctx.Context,
    split: Pipeline,
    merge: Pipeline,

    /// Bindings of a split set: q, the K region, the V region, the scratch.
    pub const split_bindings: u32 = 4;
    /// Bindings of a merge set: the scratch, the output.
    pub const merge_bindings: u32 = 2;
    /// Lanes of a merge workgroup, the `local_size_x` of `attention_merge.comp`.
    const merge_lanes: u32 = 64;

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // `gpu-selftest attn` prints the measured error at each depth.
        .tolerance = .{ .relative = 1e-3 },
    };

    const SplitPush = extern struct {
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
        stream_stride_elems: u32,
        q_row_base: u32,
        kv_stream_base: u32,
        scratch_row: u32,
        max_splits: u32,
    };

    const MergePush = extern struct {
        n_heads: u32,
        head_dim: u32,
        t_len: u32,
        scratch_row: u32,
        max_splits: u32,
        out_row: u32,
    };

    pub const InitError = Pipeline.InitError;

    /// Build both passes. The split pass follows the KV-cache element type, as
    /// in `Attention.initKv`. The merge pass reads f32 records at either type.
    pub fn initKv(ctx: *const vulkan_ctx.Context, kv: dtype.KvDtype) InitError!AttentionSplit {
        const split_spv: []align(@alignOf(u32)) const u8 = switch (kv) {
            .f32 => &spirv.attention_split_spv,
            .f16 => &spirv.attention_split_f16_spv,
        };

        const split = try Pipeline.init(ctx, split_spv, split_bindings, @sizeOf(SplitPush));
        errdefer split.deinit(ctx);

        const merge = try Pipeline.init(
            ctx,
            &spirv.attention_merge_spv,
            merge_bindings,
            @sizeOf(MergePush),
        );

        return .{
            .ctx = ctx,
            .split = split,
            .merge = merge,
        };
    }

    pub fn deinit(self: *AttentionSplit) void {
        self.merge.deinit(self.ctx);
        self.split.deinit(self.ctx);
    }

    /// Record the split pass of one row. The caller binds `set` with q, the K region, the V region,
    /// and the scratch. Row `q_row_base` of q reads stream `kv_stream_base`, and the records land
    /// in row `scratch_row` of a record group of `stride` splits per head.
    pub fn recordSplit(
        self: *const AttentionSplit,
        cb: vk.CommandBuffer,
        set: vk.DescriptorSet,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
        stream_stride_elems: u32,
        q_row_base: u32,
        kv_stream_base: u32,
        scratch_row: u32,
        stride: u32,
    ) void {
        // Check the row and the grid.
        std.debug.assert(attn_split.usesSplit(t_len));

        const n_splits = attn_split.splitCount(t_len);
        std.debug.assert(n_splits >= 2);
        std.debug.assert(n_splits <= stride);
        std.debug.assert(n_heads >= 1);
        std.debug.assert(n_heads <= max_workgroups_x);
        std.debug.assert(n_kv_heads >= 1);
        std.debug.assert(n_heads % n_kv_heads == 0);
        std.debug.assert(head_dim >= 1);

        // The shader stages q in LDS of `attn_max_head_dim` floats, and it reads K in whole 16-byte
        // loads. `guardHeadDim` refuses a model that breaks either rule at boot.
        std.debug.assert(head_dim <= limits.attn_max_head_dim);
        std.debug.assert(head_dim % limits.attn_head_dim_granule == 0);
        // With a 16-byte K binding offset (`assertKBinding`), every K row then
        // starts on a 16-byte boundary.
        std.debug.assert(stream_stride_elems % limits.attn_head_dim_granule == 0);
        // A stream holds its `t_len` rows, so no row reads into the next stream.
        std.debug.assert(@as(u64, t_len) * n_kv_heads * head_dim <= stream_stride_elems);
        // The shader forms the stream base and the q row as 32-bit products,
        // so the end of each must stay inside the 31-bit index.
        std.debug.assert(
            (@as(u64, kv_stream_base) + 1) * stream_stride_elems <= limits.max_kv_index_elems,
        );
        std.debug.assert(
            (@as(u64, q_row_base) + 1) * n_heads * head_dim <= limits.max_kv_index_elems,
        );
        assertRecordsIndexable(n_heads, head_dim, scratch_row, stride);

        // Bind the split pipeline and the descriptor set.
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.split.handle);
        const desc_sets = [_]vk.DescriptorSet{set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.split.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        // Push the row geometry.
        const pc: SplitPush = .{
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .t_len = t_len,
            .stream_stride_elems = stream_stride_elems,
            .q_row_base = q_row_base,
            .kv_stream_base = kv_stream_base,
            .scratch_row = scratch_row,
            .max_splits = stride,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.split.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(SplitPush),
            @ptrCast(&pc),
        );

        // One workgroup serves up to `attn_split_queries` heads of one GQA
        // group, so each workgroup reads its K and V rows once for them.
        const groups_x = n_kv_heads * attn_split.queryGroups(n_heads / n_kv_heads);
        std.debug.assert(groups_x >= n_kv_heads);
        std.debug.assert(groups_x <= n_heads);
        std.debug.assert(n_splits <= max_workgroups_x);
        self.ctx.vkd.cmdDispatch(cb, groups_x, n_splits, 1);
    }

    /// Record the merge pass of one row. The caller binds `set` with the scratch and the output,
    /// and it records a shader barrier after the split pass of the row. It passes the `scratch_row`
    /// and `stride` of that pass, and the result lands in row `out_row` of the output.
    pub fn recordMerge(
        self: *const AttentionSplit,
        cb: vk.CommandBuffer,
        set: vk.DescriptorSet,
        n_heads: u32,
        head_dim: u32,
        t_len: u32,
        scratch_row: u32,
        stride: u32,
        out_row: u32,
    ) void {
        // Check the row and the grid.
        std.debug.assert(attn_split.usesSplit(t_len));
        std.debug.assert(attn_split.splitCount(t_len) <= stride);
        std.debug.assert(n_heads >= 1);
        std.debug.assert(n_heads <= max_workgroups_x);
        std.debug.assert(head_dim >= 1);
        // The shader forms the output row as a 32-bit product.
        std.debug.assert((@as(u64, out_row) + 1) * n_heads * head_dim <= limits.max_kv_index_elems);

        assertRecordsIndexable(n_heads, head_dim, scratch_row, stride);

        // Bind the merge pipeline and the descriptor set.
        self.ctx.vkd.cmdBindPipeline(cb, .compute, self.merge.handle);
        const desc_sets = [_]vk.DescriptorSet{set};
        self.ctx.vkd.cmdBindDescriptorSets(
            cb,
            .compute,
            self.merge.pipeline_layout,
            0,
            &desc_sets,
            null,
        );

        // Push the row geometry.
        const pc: MergePush = .{
            .n_heads = n_heads,
            .head_dim = head_dim,
            .t_len = t_len,
            .scratch_row = scratch_row,
            .max_splits = stride,
            .out_row = out_row,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.merge.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(MergePush),
            @ptrCast(&pc),
        );

        // Each lane of a workgroup owns one element of the head, so the
        // chunks of a head cover every element and the last chunk is partial.
        const chunks = std.math.divCeil(u32, head_dim, merge_lanes) catch unreachable;
        std.debug.assert(chunks >= 1);
        std.debug.assert(@as(u64, chunks) * merge_lanes >= head_dim);
        std.debug.assert(@as(u64, chunks - 1) * merge_lanes < head_dim);
        std.debug.assert(chunks <= max_workgroups_x);
        self.ctx.vkd.cmdDispatch(cb, n_heads, chunks, 1);
    }

    /// Assert the K binding of a split set at the site that writes it. The split pass reads K in
    /// 16-byte loads, so the binding must start on a 16-byte boundary and hold whole loads.
    pub fn assertKBinding(k: vk.DescriptorBufferInfo) void {
        std.debug.assert(k.offset % 16 == 0);
        std.debug.assert(k.range % 16 == 0);
        std.debug.assert(k.range >= 16);
    }

    /// Assert the V binding of a split set at the site that writes it. The
    /// split pass reads V in 32-bit words, so the binding must start and end on a word.
    pub fn assertVBinding(v: vk.DescriptorBufferInfo) void {
        std.debug.assert(v.offset % 4 == 0);
        std.debug.assert(v.range % 4 == 0);
        std.debug.assert(v.range >= 4);
    }

    /// Refuse a model whose heads the split pass cannot take, before the
    /// first allocation. A GGUF file is untrusted input, and a ReleaseFast
    /// build compiles the asserts of `recordSplit` out.
    pub fn guardHeadDim(head_dim: u32) error{AttentionHeadDimUnsupported}!void {
        if (attn_split.headDimFits(head_dim)) return;
        log.err(
            "The model head_dim is {d}. " ++
                "GPU attention takes a head_dim that is a multiple of {d} and at most {d}. " ++
                "Run this model on a CPU build",
            .{ head_dim, limits.attn_head_dim_granule, limits.attn_max_head_dim },
        );
        return error.AttentionHeadDimUnsupported;
    }

    /// The shaders form each record offset as a 32-bit product, so the last
    /// element of group row `scratch_row` at `stride` splits per head must
    /// stay inside the 31-bit index.
    fn assertRecordsIndexable(n_heads: u32, head_dim: u32, scratch_row: u32, stride: u32) void {
        std.debug.assert(stride >= 1);

        const end = (@as(u64, scratch_row) + 1) * n_heads * stride * (@as(u64, head_dim) + 2);
        std.debug.assert(end <= limits.max_kv_index_elems);
    }
};

/// The summation-order class of an attention kernel set. The byte tests (the split digests, and
/// the equality of rows across paths) hold for one class only. A kernel set of another class runs
/// the reference tests at its declared tolerance and the token gates.
pub const AttnOrderClass = enum {
    /// Splits and merge both run in ascending order, so a row gets the same bytes at every batch,
    /// slot, and record group.
    split_ascending,
};

/// The head shape and the scratch geometry that every dispatch of one layer
/// shares.
pub const AttnShape = struct {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    stream_stride_elems: u32,
    max_splits: u32,
};

/// The attention kernel set of `-Dattn-kernel=split`: the resident kernels for a row inside one
/// split, and `AttentionSplit` past it. It points at the pipelines of its runner and owns none,
/// and `runtime/attn_record.zig` records a plan through it.
pub const SplitAttnKernels = struct {
    single: *Attention,
    multi: *AttentionMultiStream,
    passes: *const AttentionSplit,
    /// The resident kernel of a prefill chunk. A kernel set that records no prefill,
    /// as in the bench grid, leaves it null.
    batch: ?*AttentionBatch = null,
    /// The profile of the command buffer, which takes the pass marks. Null: no marks.
    prof: ?*gpu_profile.GpuProfile = null,

    pub const order: AttnOrderClass = .split_ascending;
    pub const tolerance: op_requires.Tolerance = AttentionSplit.requires.tolerance;
    pub const Cmd = vk.CommandBuffer;

    /// The set of the resident rows. A decode step binds the single-row set
    /// of `Attention`, a batched step binds the multi-stream set, and a
    /// prefill chunk binds the set of `AttentionBatch`.
    pub const ResidentSet = union(enum) {
        single: vk.DescriptorSet,
        multi: vk.DescriptorSet,
        batch: vk.DescriptorSet,
    };

    /// The sets and the shape of one layer.
    pub const Binding = struct {
        shape: AttnShape,
        resident: ResidentSet,
        split: vk.DescriptorSet,
        merge: vk.DescriptorSet,
    };

    comptime {
        std.debug.assert(std.meta.eql(Attention.requires.tolerance, tolerance));
        std.debug.assert(std.meta.eql(AttentionMultiStream.requires.tolerance, tolerance));
        std.debug.assert(std.meta.eql(AttentionBatch.requires.tolerance, tolerance));
    }

    pub fn recordResident(self: SplitAttnKernels, cb: Cmd, b: Binding, r: attn_split.PlanRow) void {
        std.debug.assert(!attn_split.usesSplit(r.t_len));

        const s = b.shape;
        switch (b.resident) {
            .single => |set| {
                // The single-row set binds the q row and the stream of a
                // decode step, so the row reads row 0 of stream 0.
                std.debug.assert(r.row == 0);
                std.debug.assert(r.slot == 0);

                self.single.recordInto(cb, set, s.n_heads, s.n_kv_heads, s.head_dim, r.t_len);
            },
            .multi => |set| self.multi.recordInto(
                cb,
                set,
                s.n_heads,
                s.n_kv_heads,
                s.head_dim,
                r.t_len,
                s.stream_stride_elems,
                r.row,
                r.slot,
                1,
            ),
            // A prefill chunk records its resident rows as one run.
            .batch => unreachable,
        }
    }

    /// Record the resident prefix of a prefill chunk: query rows `[0, count)`
    /// at the positions `start_pos` through `start_pos + count - 1` of the
    /// stream that the batch set binds.
    pub fn recordResidentRun(
        self: SplitAttnKernels,
        cb: Cmd,
        b: Binding,
        start_pos: u32,
        count: u32,
    ) void {
        std.debug.assert(count >= 1);
        std.debug.assert(start_pos + count <= limits.attn_split_len);

        const s = b.shape;
        const set = switch (b.resident) {
            .batch => |set| set,
            .single, .multi => unreachable,
        };

        self.batch.?.recordInto(cb, set, s.n_heads, s.n_kv_heads, s.head_dim, start_pos, count);
    }

    pub fn recordSplit(
        self: SplitAttnKernels,
        cb: Cmd,
        b: Binding,
        r: attn_split.PlanRow,
        scratch_row: u32,
        stride: u32,
    ) void {
        std.debug.assert(attn_split.usesSplit(r.t_len));

        const s = b.shape;
        assertGroupRecords(s, scratch_row, stride);
        self.passes.recordSplit(
            cb,
            b.split,
            s.n_heads,
            s.n_kv_heads,
            s.head_dim,
            r.t_len,
            s.stream_stride_elems,
            r.row,
            r.slot,
            scratch_row,
            stride,
        );
    }

    pub fn recordMerge(
        self: SplitAttnKernels,
        cb: Cmd,
        b: Binding,
        r: attn_split.PlanRow,
        scratch_row: u32,
        stride: u32,
    ) void {
        std.debug.assert(attn_split.usesSplit(r.t_len));

        const s = b.shape;
        assertGroupRecords(s, scratch_row, stride);
        self.passes.recordMerge(
            cb,
            b.merge,
            s.n_heads,
            s.head_dim,
            r.t_len,
            scratch_row,
            stride,
            r.row,
        );
    }

    /// A shader read or write waits for the shader writes before it, the
    /// barrier the runners record between two dependent dispatches.
    pub fn recordBarrier(self: SplitAttnKernels, cb: Cmd) void {
        const mb = [_]vk.MemoryBarrier{.{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true },
        }};
        self.passes.ctx.vkd.cmdPipelineBarrier(
            cb,
            .{ .compute_shader_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            &mb,
            null,
            null,
        );
    }

    /// Close the interval of the resident rows, or of the pass `part`, on the attention op.
    pub fn recordMark(self: SplitAttnKernels, cb: Cmd, part: ?layer_op.OpPart) void {
        if (self.prof) |prof| {
            prof.mark(cb, .initOne(.attention), part);
        }
    }

    /// Group row `scratch_row` at `stride` splits per head ends inside the
    /// scratch of `shape.max_splits` splits per head and row.
    fn assertGroupRecords(shape: AttnShape, scratch_row: u32, stride: u32) void {
        std.debug.assert(stride >= 1);
        std.debug.assert(stride <= shape.max_splits * limits.attn_split_rows);

        const record_elems = @as(u64, shape.head_dim) + 2;
        const end = (@as(u64, scratch_row) + 1) * shape.n_heads * stride * record_elems;
        const scratch_elems = @as(
            u64,
            limits.attn_split_rows,
        ) * shape.n_heads * shape.max_splits * record_elems;
        std.debug.assert(end <= scratch_elems);
    }
};

/// The attention kernel set that `-Dattn-kernel` selects. A new kernel set is one struct
/// with the method set of `SplitAttnKernels` and one entry in the build enum.
pub const AttnKernels = switch (build_options.attn_kernel) {
    .split => SplitAttnKernels,
};

/// Batched causal attention for the resident rows of a prefill chunk. `start_pos + b` must be at
/// most `limits.attn_split_len`, because the kernel holds every score of a row in LDS. A deeper row
/// takes `AttentionSplit`, through `attn_record.recordPrefill`.
pub const AttentionBatch = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 1e-3 },
    };

    const PushConsts = extern struct {
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: u32,
        b_count: u32,
    };
    pub const InitError = VecAdd.InitError;

    /// KV-dtype variant selector.
    pub fn initKv(ctx: *const vulkan_ctx.Context, kv: dtype.KvDtype) InitError!AttentionBatch {
        const spv: []align(@alignOf(u32)) const u8 = switch (kv) {
            .f32 => &spirv.attention_batch_spv,
            .f16 => &spirv.attention_batch_f16_spv,
        };

        const pipeline = try Pipeline.init(ctx, spv, 4, @sizeOf(PushConsts));

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *AttentionBatch) void {
        self.pipeline.deinit(self.ctx);
    }

    pub fn recordInto(
        self: *AttentionBatch,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        start_pos: u32,
        b: u32,
    ) void {
        // Check the row range and the head shape.
        std.debug.assert(b >= 1);
        // Row bi attends to positions 0 through start_pos + bi, so the last
        // row reaches start_pos + b. The LDS score row holds one split.
        std.debug.assert(@as(u64, start_pos) + b <= limits.attn_split_len);
        std.debug.assert(n_kv_heads >= 1);
        std.debug.assert(n_heads % n_kv_heads == 0);
        // The K-dot loops read an adjacent element pair per step, on both KV
        // element types, so an odd head_dim reads past the row.
        std.debug.assert(head_dim % 2 == 0);

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

        // Push the run and dispatch every (head, row) pair.
        const pc: PushConsts = .{
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .start_pos = start_pos,
            .b_count = b,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        self.ctx.vkd.cmdDispatch(cb, n_heads, b, 1);
    }
};

/// Lockstep multi-stream attention, one row per stream. Every stream shares `t_len`, and a row past
/// `limits.attn_split_len` positions takes `AttentionSplit`.
pub const AttentionMultiStream = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .{ .relative = 1e-3 },
    };

    const PushConsts = extern struct {
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
        stream_stride_elems: u32,
        q_row_base: u32,
        kv_stream_base: u32,
    };
    pub const InitError = VecAdd.InitError;

    /// KV-dtype variant selector.
    pub fn initKv(
        ctx: *const vulkan_ctx.Context,
        kv: dtype.KvDtype,
    ) InitError!AttentionMultiStream {
        const spv: []align(@alignOf(u32)) const u8 = switch (kv) {
            .f32 => &spirv.attention_multi_stream_spv,
            .f16 => &spirv.attention_multi_stream_f16_spv,
        };

        const pipeline = try Pipeline.init(ctx, spv, 4, @sizeOf(PushConsts));

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *AttentionMultiStream) void {
        self.pipeline.deinit(self.ctx);
    }

    /// One dispatch covers `n_rows` pairs, where row i is (q_row_base + i, kv_stream_base + i), and
    /// all share `t_len`. A caller with arbitrary per-row (slot, position) pairs records one
    /// single-row dispatch per row.
    pub fn recordInto(
        self: *AttentionMultiStream,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
        stream_stride_elems: u32,
        q_row_base: u32,
        kv_stream_base: u32,
        n_rows: u32,
    ) void {
        // Check the row and the stream bounds.
        std.debug.assert(t_len >= 1);
        std.debug.assert(t_len <= limits.max_model_ctx);
        std.debug.assert(!attn_split.usesSplit(t_len));
        std.debug.assert(n_kv_heads >= 1);
        std.debug.assert(n_heads % n_kv_heads == 0);
        std.debug.assert(n_rows >= 1);
        // A stream holds its `t_len` rows, so no row reads into the next stream.
        std.debug.assert(@as(u64, t_len) * n_kv_heads * head_dim <= stream_stride_elems);
        // The shader forms each stream base as a 32-bit product, so the end of
        // the last stream must stay inside the 31-bit index.
        std.debug.assert(
            (@as(u64, kv_stream_base) + n_rows) * stream_stride_elems <= limits.max_kv_index_elems,
        );

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

        // Push the stream geometry and dispatch every (head, row) pair.
        const pc: PushConsts = .{
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .t_len = t_len,
            .stream_stride_elems = stream_stride_elems,
            .q_row_base = q_row_base,
            .kv_stream_base = kv_stream_base,
        };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        self.ctx.vkd.cmdDispatch(cb, n_heads, n_rows, 1);
    }
};

/// Narrow the K and V commit scratch of one layer from f32 to f16, bit for bit as the CPU commit
/// does. The kernel is record-only: a fused runner owns the buffers, the sets, and the per-token
/// command buffer. `kvCommitF16OneShot` drives it for the rounding gate.
pub const KvCommitF16 = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        .tolerance = .bit_exact,
    };

    const PushConsts = extern struct {
        n: u32,
    };

    pub const InitError = Pipeline.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!KvCommitF16 {
        const pipeline = try Pipeline.init(ctx, &spirv.kv_commit_f16_spv, 4, @sizeOf(PushConsts));

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *KvCommitF16) void {
        self.pipeline.deinit(self.ctx);
    }

    /// Record one narrowing dispatch over `n` elements of each scratch.
    pub fn recordInto(
        self: *KvCommitF16,
        cb: vk.CommandBuffer,
        desc_set: vk.DescriptorSet,
        n: u32,
    ) void {
        std.debug.assert(n >= 1);

        const group_count_x: u32 = (n + 63) / 64;
        std.debug.assert(group_count_x <= max_workgroups_x);
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
        const pc: PushConsts = .{ .n = n };
        self.ctx.vkd.cmdPushConstants(
            cb,
            self.pipeline.pipeline_layout,
            .{ .compute_bit = true },
            0,
            @sizeOf(PushConsts),
            @ptrCast(&pc),
        );
        self.ctx.vkd.cmdDispatch(cb, group_count_x, 1, 1);
    }
};

// -- api --------------------------------------------------------------------

/// The one-shot driver of the record-only `KvCommitF16`, for the rounding gate alone. The output is
/// raw f16 bits, what the cache stores, so a bit comparison states the rounding property and keeps
/// a second conversion out of the path under test.
pub fn kvCommitF16OneShot(
    kernel: *KvCommitF16,
    k_src: []const f32,
    v_src: []const f32,
    k_out_bits: []u16,
    v_out_bits: []u16,
) !void {
    // One `n` covers all four buffers. The kernel takes the same count.
    std.debug.assert(k_src.len >= 1);
    std.debug.assert(v_src.len == k_src.len);
    std.debug.assert(k_out_bits.len == k_src.len);
    std.debug.assert(v_out_bits.len == k_src.len);

    // Stage the two inputs and size the two outputs.
    const n: u32 = @intCast(k_src.len);
    const ctx = kernel.ctx;

    const src_bytes: vk.DeviceSize = @intCast(k_src.len * @sizeOf(f32));
    const out_bytes: vk.DeviceSize = @intCast(k_src.len * @sizeOf(u16));

    var buf_k_src: HostBuffer = .{ .ctx = ctx };
    defer buf_k_src.deinit();

    var buf_v_src: HostBuffer = .{ .ctx = ctx };
    defer buf_v_src.deinit();

    var buf_k_out: HostBuffer = .{ .ctx = ctx };
    defer buf_k_out.deinit();

    var buf_v_out: HostBuffer = .{ .ctx = ctx };
    defer buf_v_out.deinit();

    try buf_k_src.ensure(src_bytes);
    try buf_v_src.ensure(src_bytes);
    try buf_k_out.ensure(out_bytes);
    try buf_v_out.ensure(out_bytes);
    @memcpy(buf_k_src.sliceTyped(f32, k_src.len), k_src);
    @memcpy(buf_v_src.sliceTyped(f32, v_src.len), v_src);

    // Build a throwaway pool and set, and bind the four buffers.
    const desc_pool = try createSingleSetPool(ctx, 4);
    defer ctx.vkd.destroyDescriptorPool(ctx.device, desc_pool, null);

    const desc_set = try allocateOneSet(ctx, desc_pool, kernel.pipeline.set_layout);

    const infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = buf_k_src.buffer, .offset = 0, .range = src_bytes },
        .{ .buffer = buf_v_src.buffer, .offset = 0, .range = src_bytes },
        .{ .buffer = buf_k_out.buffer, .offset = 0, .range = out_bytes },
        .{ .buffer = buf_v_out.buffer, .offset = 0, .range = out_bytes },
    };
    updateStorageBindings(ctx, desc_set, &infos);

    // Dispatch through `recordInto`, then read the bits back.
    const Rec = struct {
        kernel: *KvCommitF16,
        set: vk.DescriptorSet,
        n: u32,

        pub fn record(r: @This(), cb: vk.CommandBuffer) Recorded {
            r.kernel.recordInto(cb, r.set, r.n);
            return .{};
        }
    };
    try submitOnce(ctx, .dispatch, Rec{ .kernel = kernel, .set = desc_set, .n = n });

    @memcpy(k_out_bits, buf_k_out.sliceTyped(u16, k_out_bits.len));
    @memcpy(v_out_bits, buf_v_out.sliceTyped(u16, v_out_bits.len));
}
