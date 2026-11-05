//! Compute-kernel wrappers for the Vulkan op backend, one struct per shader.
//! Each wrapper owns its shader module, descriptor set layout, pipeline layout,
//! and compute pipeline, but not the buffers. A caller hands it data, and the
//! wrapper records the dispatch. The engine records many small pipelines into
//! one command buffer per token, not one fused mega-shader. Every pipeline must
//! therefore be a self-contained value that the recording side binds and
//! dispatches. Each kernel struct declares its own `requires`, because the
//! variants span more than one tolerance class.
//!
//! The per-op wrappers keep buffer handling simple: host-visible coherent
//! memory, a fresh descriptor pool per dispatch, and queue-wait-idle as the only
//! synchronization. That trades speed for clarity, which suits a correctness
//! reference and the per-op benchmarks. The fused runners in
//! `runtime/llama_fused.zig` and `runtime/qwen_fused.zig` hold device-local
//! weights, pre-bound descriptor sets, and one submit per token.

const std = @import("std");
const vk = @import("vulkan");
const build_options = @import("build_options");
const attn_split = @import("../core/attn_split.zig");
const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_index = @import("../core/kv_index.zig");
const limits = @import("../core/limits.zig");
const op_requires = @import("../core/op_requires.zig");
const core_log = @import("../core/log.zig");
const span = @import("../core/span.zig");
const gpu_profile = @import("../runtime/gpu_profile.zig");
const vulkan_ctx = @import("../runtime/vulkan_ctx.zig");
const RopeStyle = @import("../ops/naive.zig").RopeStyle;
const log = std.log.scoped(.gpu);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("ops_vulkan/kernels.zig was included in a non-vulkan build. " ++
            "Either build with -Druntime=vulkan, or fix the caller to gate on " ++
            "build_options.use_vulkan.");
    }
}

// -- constants --------------------------------------------------------------

/// Q8_0 and Q4_0 block layout, in lockstep with `src/core/dtype.zig` and the
/// CPU decoder in `ops/naive.zig`. The quant kernels assert a match.
const q8_0_block_elems: usize = 32;

const q8_0_block_bytes: usize = 2 + q8_0_block_elems; // f16 scale + 32 i8 quants
const q4_0_block_elems: usize = 32;
const q4_0_packed_bytes: usize = 16; // 32 weights at 4 bits = 16 bytes
pub const q4_0_block_bytes: usize = 2 + q4_0_packed_bytes; // f16 scale + 16 packed

/// The largest workgroup count a dispatch can ask for in one dimension. Vulkan
/// guarantees at least this in `maxComputeWorkGroupCount[0]`, so a kernel that
/// stays under it needs no device query. The narrowing kernels assert against
/// it, because their grid follows an element count the caller supplies.
const max_workgroups_x: u32 = 65535;

/// SPIR-V binary for `shaders/vec_add.comp`, produced by `glslc` at build time
/// and embedded through an anonymous import that `build.zig` declares.
/// SPIR-V is a stream of 4-byte words and `createShaderModule` casts back to
/// `[*]const u32`, so the declaration forces `align(@alignOf(u32))` on the
/// embed. A bare `@embedFile` gives a `*const [N]u8` of alignment 1. Every
/// shader embed below repeats this form.
const vec_add_spv: [
    @embedFile(
        "vec_add.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("vec_add.spv").*;

/// SPIR-V for `shaders/residual_add.comp`.
const residual_add_spv: [
    @embedFile(
        "residual_add.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("residual_add.spv").*;

/// SPIR-V for `shaders/sigmoid_gate.comp`.
const sigmoid_gate_spv: [
    @embedFile(
        "sigmoid_gate.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("sigmoid_gate.spv").*;

/// SPIR-V for `shaders/attention.comp`, once per KV-cache element type.
const attention_spv: [
    @embedFile(
        "attention.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention.spv").*;

const attention_f16_spv: [
    @embedFile(
        "attention_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_f16.spv").*;

/// SPIR-V for `shaders/attention_split.comp`, once per KV-cache element type,
/// and for `shaders/attention_merge.comp`, whose f32 records serve both types.
const attention_split_spv: [
    @embedFile(
        "attention_split.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_split.spv").*;

const attention_split_f16_spv: [
    @embedFile(
        "attention_split_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_split_f16.spv").*;
const attention_merge_spv: [
    @embedFile(
        "attention_merge.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_merge.spv").*;

/// SPIR-V for `shaders/rope.comp`.
const rope_spv: [@embedFile("rope.spv").len]u8 align(@alignOf(u32)) = @embedFile("rope.spv").*;

/// SPIR-V for `shaders/rope_multi.comp`.
const rope_multi_spv: [
    @embedFile(
        "rope_multi.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("rope_multi.spv").*;

/// SPIR-V for `shaders/rmsnorm.comp`.
const rmsnorm_spv: [
    @embedFile(
        "rmsnorm.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("rmsnorm.spv").*;

/// SPIR-V for `shaders/swiglu.comp`.
const swiglu_spv: [
    @embedFile(
        "swiglu.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("swiglu.spv").*;

/// SPIR-V for `shaders/linear_q8_0.comp`. See `vec_add_spv` for the alignment
/// rule that every shader embed here follows.
const linear_q8_0_spv: [
    @embedFile(
        "linear_q8_0.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0.spv").*;

/// SPIR-V for `shaders/linear_q8_0_reduce.comp`.
const linear_q8_0_reduce_spv: [
    @embedFile(
        "linear_q8_0_reduce.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0_reduce.spv").*;

/// SPIR-V for `shaders/linear_q4_0_reduce.comp`.
const linear_q4_0_reduce_spv: [
    @embedFile(
        "linear_q4_0_reduce.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q4_0_reduce.spv").*;

/// SPIR-V for `shaders/linear_q8_0_subgroup.comp`.
const linear_q8_0_subgroup_spv: [
    @embedFile(
        "linear_q8_0_subgroup.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0_subgroup.spv").*;

/// SPIR-V for `shaders/linear_q8_0_batch.comp`.
const linear_q8_0_batch_spv: [
    @embedFile(
        "linear_q8_0_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0_batch.spv").*;

/// SPIR-V for `shaders/linear_q8_0_gemm.comp`.
const linear_q8_0_gemm_spv: [
    @embedFile(
        "linear_q8_0_gemm.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0_gemm.spv").*;

/// SPIR-V for `shaders/linear_q4_0_batch.comp`.
const linear_q4_0_batch_spv: [
    @embedFile(
        "linear_q4_0_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q4_0_batch.spv").*;

/// SPIR-V for `shaders/rmsnorm_batch.comp`.
const rmsnorm_batch_spv: [
    @embedFile(
        "rmsnorm_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("rmsnorm_batch.spv").*;

/// SPIR-V for `shaders/rope_multi_batch.comp`.
const rope_multi_batch_spv: [
    @embedFile(
        "rope_multi_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("rope_multi_batch.spv").*;

/// SPIR-V for `shaders/attention_batch.comp`, once per KV-cache element type.
const attention_batch_spv: [
    @embedFile(
        "attention_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_batch.spv").*;

const attention_batch_f16_spv: [
    @embedFile(
        "attention_batch_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_batch_f16.spv").*;

/// SPIR-V for `shaders/attention_multi_stream.comp`, once per KV-cache element
/// type.
const attention_multi_stream_spv: [
    @embedFile(
        "attention_multi_stream.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_multi_stream.spv").*;

const attention_multi_stream_f16_spv: [
    @embedFile(
        "attention_multi_stream_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_multi_stream_f16.spv").*;

/// SPIR-V for `shaders/kv_commit_f16.comp`.
const kv_commit_f16_spv: [
    @embedFile(
        "kv_commit_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("kv_commit_f16.spv").*;

/// SPIR-V for `shaders/gdn_decode.comp`.
const gdn_decode_spv: [
    @embedFile(
        "gdn_decode.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("gdn_decode.spv").*;

/// SPIR-V for the chunked-parallel GDN prefill (three dispatches).
const gdn_chunk_conv_spv: [
    @embedFile(
        "gdn_chunk_conv.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("gdn_chunk_conv.spv").*;

const gdn_chunk_scan_spv: [
    @embedFile(
        "gdn_chunk_scan.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("gdn_chunk_scan.spv").*;
const gdn_chunk_norm_spv: [
    @embedFile(
        "gdn_chunk_norm.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("gdn_chunk_norm.spv").*;

/// SPIR-V for `shaders/qwen_qk_norm.comp`.
const qwen_qk_norm_spv: [
    @embedFile(
        "qwen_qk_norm.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("qwen_qk_norm.spv").*;

// -- types ------------------------------------------------------------------

/// The four objects of one compute-pipeline chain: the shader module, the
/// descriptor set layout, the pipeline layout, and the pipeline handle. They are
/// created and destroyed as a unit, and every kernel wrapper below embeds one.
/// Only the SPIR-V blob, the storage-buffer binding count, and the push-constant
/// size vary per kernel, so those three are the init parameters.
pub const Pipeline = struct {
    shader_module: vk.ShaderModule,
    set_layout: vk.DescriptorSetLayout,
    pipeline_layout: vk.PipelineLayout,
    handle: vk.Pipeline,

    pub const InitError = error{SpirVMisaligned} ||
        vk.DeviceWrapper.CreateShaderModuleError ||
        vk.DeviceWrapper.CreateDescriptorSetLayoutError ||
        vk.DeviceWrapper.CreatePipelineLayoutError ||
        vk.DeviceWrapper.CreateComputePipelinesError;

    pub fn init(
        ctx: *const vulkan_ctx.Context,
        spv: []align(@alignOf(u32)) const u8,
        binding_count: u32,
        push_size: u32,
    ) InitError!Pipeline {
        std.debug.assert(spv.len % 4 == 0);
        std.debug.assert(binding_count >= 1);
        std.debug.assert(push_size >= 4);
        std.debug.assert(push_size % 4 == 0);

        if (@intFromPtr(spv.ptr) % 4 != 0) return error.SpirVMisaligned;

        const shader_info: vk.ShaderModuleCreateInfo = .{
            .code_size = spv.len,
            .p_code = @ptrCast(spv.ptr),
        };
        const shader_module = try ctx.vkd.createShaderModule(ctx.device, &shader_info, null);
        errdefer ctx.vkd.destroyShaderModule(ctx.device, shader_module, null);

        const set_layout = try createStorageBufferLayout(ctx, binding_count);
        errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, set_layout, null);

        const push_range: vk.PushConstantRange = .{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = push_size,
        };
        const pipeline_layout = try createPipelineLayout(ctx, set_layout, &push_range);
        errdefer ctx.vkd.destroyPipelineLayout(ctx.device, pipeline_layout, null);

        const handle = try createComputePipeline(ctx, shader_module, pipeline_layout);

        return .{
            .shader_module = shader_module,
            .set_layout = set_layout,
            .pipeline_layout = pipeline_layout,
            .handle = handle,
        };
    }

    pub fn deinit(self: Pipeline, ctx: *const vulkan_ctx.Context) void {
        ctx.vkd.destroyPipeline(ctx.device, self.handle, null);
        ctx.vkd.destroyPipelineLayout(ctx.device, self.pipeline_layout, null);
        ctx.vkd.destroyDescriptorSetLayout(ctx.device, self.set_layout, null);
        ctx.vkd.destroyShaderModule(ctx.device, self.shader_module, null);
    }
};

/// Double-buffered host-to-device upload ring, the asynchronous counterpart to
/// `runStagingCopies`.
pub const StreamRing = @import("stream_ring.zig").StreamRing;

/// Element-wise `out[i] = a[i] + b[i]`, the proof-of-life pipeline. Three
/// storage buffers (a, b, out) and one `u32` push constant for the element
/// count. The three slices must have the same length. One lane per element at
/// local_size_x = 256, so the grid is ceil(n / 256) workgroups. The result is
/// bit-exact against the CPU reference, because one IEEE 754 add per element
/// carries no reduction.
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
        const pipeline = try Pipeline.init(ctx, &vec_add_spv, 3, @sizeOf(u32));
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

        const n: u32 = @intCast(a.len);
        const byte_size: vk.DeviceSize = @intCast(a.len * @sizeOf(f32));

        try self.buf_a.ensure(byte_size);
        try self.buf_b.ensure(byte_size);
        try self.buf_out.ensure(byte_size);

        @memcpy(self.buf_a.sliceTyped(f32, a.len), a);
        @memcpy(self.buf_b.sliceTyped(f32, b.len), b);

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_a.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_b.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = byte_size },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

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

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// In-place residual add `dst[i] += x[i]`. Two storage buffers (x readonly, dst
/// read and write) and one `u32` push constant for the length. `x` and `dst`
/// must have the same length. One lane per element at local_size_x = 256. The
/// result is bit-exact against the CPU reference, because one IEEE 754 add per
/// element carries no reduction.
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
        const pipeline = try Pipeline.init(ctx, &residual_add_spv, 2, @sizeOf(u32));
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

    /// Record `n_dispatches` back-to-back `residual_add` invocations into one
    /// command buffer and submit once. The result is
    /// `dst_final[i] = dst_init[i] + n_dispatches * x[i]`, because each dispatch
    /// adds `x` to the previous dispatch's write. The benchmark reads the timing
    /// rather than the math. Per-dispatch wall-clock falls as `n_dispatches`
    /// rises, which prices the per-call submit overhead that one command buffer
    /// per token amortizes across a whole layer.
    pub fn runRepeat(
        self: *ResidualAdd,
        n_dispatches: u32,
        x: []const f32,
        dst_init: []const f32,
        dst_final: []f32,
    ) RunError!void {
        if (x.len != dst_init.len) return error.LengthMismatch;
        if (x.len != dst_final.len) return error.LengthMismatch;

        const n: u32 = @intCast(x.len);
        const byte_size: vk.DeviceSize = @intCast(x.len * @sizeOf(f32));

        try self.buf_x.ensure(byte_size);
        try self.buf_dst.ensure(byte_size);

        @memcpy(self.buf_x.sliceTyped(f32, x.len), x);
        @memcpy(self.buf_dst.sliceTyped(f32, dst_init.len), dst_init);

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

        // SSBO read-after-write hazard between dispatches: each dispatch
        // reads the previous dispatch's writes to `dst`. Without an
        // explicit barrier the driver is free to overlap them.
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

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(dst_final, self.buf_dst.sliceTyped(f32, dst_final.len));
    }

    /// Record one `dst[i] += x[i]` dispatch into a command buffer the caller
    /// owns. `dst` carries both the input and the result. The caller binds the
    /// descriptor set (x, dst) and inserts any barriers.
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

        const n: u32 = @intCast(x.len);
        const byte_size: vk.DeviceSize = @intCast(x.len * @sizeOf(f32));

        try self.buf_x.ensure(byte_size);
        try self.buf_dst.ensure(byte_size);

        @memcpy(self.buf_x.sliceTyped(f32, x.len), x);
        @memcpy(self.buf_dst.sliceTyped(f32, dst.len), dst);

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

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(dst, self.buf_dst.sliceTyped(f32, dst.len));
    }
};

/// In-place output gate `dst[i] *= sigmoid(gate[i])` for Qwen3.5 full attention.
/// Two storage buffers (dst read and write, gate readonly) and one `u32` push
/// constant for the length. `dst` and `gate` must have the same length. One lane
/// per element at local_size_x = 256. The GPU `exp` rounds differently from the
/// CPU one, so the declared tolerance is `.relative = 1e-5`.
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
        const pipeline = try Pipeline.init(ctx, &sigmoid_gate_spv, 2, @sizeOf(u32));
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

/// Fused single-token attention: the scores, the softmax, and the V-weighted
/// sum, one workgroup per query head. Reads q, k_cache, and v_cache, and writes
/// out. `t_len` must be at least 1 and at most `limits.max_model_ctx`, and
/// `n_heads` must be a multiple of `n_kv_heads`. `recordInto` takes a row of at
/// most `limits.attn_split_len` positions, whose scores the shader holds in LDS,
/// and head_dim carries no bound. `run` takes a row of any depth, and it sends
/// a longer row through the two passes of `AttentionSplit`. The resident grid
/// is n_heads workgroups of 64 lanes. The declared tolerance is
/// `.relative = 1e-3`.
pub const Attention = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    /// The KV-cache element type of the pipelines, which `runTyped` checks.
    kv: dtype.KvDtype,
    /// The split path of `runTyped`, built for its first split row. The fused
    /// runners only call `recordInto`, so they never build it.
    split: ?SplitKit,
    buf_q: HostBuffer,
    buf_k: HostBuffer,
    buf_v: HostBuffer,
    buf_out: HostBuffer,
    buf_scratch: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // The reduction order differs from the CPU reference. LDS tree-reduces
        // carry the softmax max and sum and the per-output V-sum, and the GPU
        // `exp` rounds differently. The resident walk showed a relative error
        // of 5e-4 to 6e-4 on SmolLM2, Llama-1B, and Llama-8B shapes of up to
        // 512 positions. The bench applies a near-zero-denominator floor of
        // 1e-5 (see `computeErrors` in tools/bench.zig). 1e-3 leaves a 1.5x
        // cushion. Rows past `limits.attn_split_len` take `AttentionSplit`, so
        // tests/vulkan_attention_test.zig gates this kernel on the SmolLM2
        // shape at 128 positions. `gpu-selftest attn` gates it on the Qwen3.5
        // shape at 7 and `limits.attn_split_len` positions, at both KV types.
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
    const SplitKit = struct {
        passes: AttentionSplit,
        pool: vk.DescriptorPool,
        split_set: vk.DescriptorSet,
        merge_set: vk.DescriptorSet,

        const BuildError = AttentionSplit.InitError ||
            vk.DeviceWrapper.CreateDescriptorPoolError ||
            vk.DeviceWrapper.AllocateDescriptorSetsError;

        fn init(ctx: *const vulkan_ctx.Context, kv: dtype.KvDtype) BuildError!SplitKit {
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

        fn deinit(self: *SplitKit, ctx: *const vulkan_ctx.Context) void {
            // Destroying the pool frees the split set and the merge set.
            ctx.vkd.destroyDescriptorPool(ctx.device, self.pool, null);
            self.passes.deinit();
        }
    };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!Attention {
        return initKv(ctx, .f32);
    }

    /// Build the resident pipeline for the KV-cache element type the process
    /// resolved. The f16 variant comes from the same GLSL source under a
    /// define, so only the element type of the k_cache and v_cache bindings
    /// differs. The LDS scores buffer, the accumulators, the push constants, and
    /// the element indexing stay the same. So one selector covers the
    /// difference, and no call site below needs to know. `runTyped` builds the
    /// split passes for the same element type when it gets its first split row.
    pub fn initKv(ctx: *const vulkan_ctx.Context, kv: dtype.KvDtype) InitError!Attention {
        const spv: []align(@alignOf(u32)) const u8 = switch (kv) {
            .f32 => &attention_spv,
            .f16 => &attention_f16_spv,
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
        if (self.split) |*kit| {
            kit.deinit(self.ctx);
        }

        self.oneshot.deinit(self.ctx);
        self.pipeline.deinit(self.ctx);
    }

    pub const RunError = ResidualAdd.RunError || SplitKit.BuildError;

    /// Record one fused-attention dispatch into a command buffer the caller
    /// owns. The caller binds the descriptor set (q, k_cache, v_cache, out).
    /// One workgroup per head. The caller must keep `t_len` within
    /// `limits.attn_split_len`. A deeper row takes the split passes through
    /// the attention kit.
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

    /// Run one row against f32 caches. See `runTyped`.
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

    /// Run one row against caches of element type `T`, which must match the
    /// KV dtype of the pipelines. A row of at most `limits.attn_split_len`
    /// positions runs the resident kernel. A longer row runs the split pass, a
    /// shader barrier, and the merge pass. The first longer row builds the
    /// split passes. Either way the row takes one submission.
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

        std.debug.assert(self.kv == kv);
        std.debug.assert(t_len >= 1);
        std.debug.assert(t_len <= limits.max_model_ctx);
        std.debug.assert(n_kv_heads >= 1);
        std.debug.assert(n_heads % n_kv_heads == 0);
        std.debug.assert(q.len == n_heads * head_dim);
        std.debug.assert(out.len >= q.len);
        // The whole row is the one stream of the split pass, and the shaders
        // index it in 32 bits.
        const row_elems = @as(u64, t_len) * n_kv_heads * head_dim;
        std.debug.assert(row_elems <= limits.max_kv_index_elems);
        std.debug.assert(k_cache.len >= row_elems);
        std.debug.assert(v_cache.len >= row_elems);
        const deep = attn_split.usesSplit(t_len);
        if (deep and self.split == null) {
            self.split = try SplitKit.init(self.ctx, self.kv);
        }

        const q_bytes: vk.DeviceSize = @intCast(q.len * @sizeOf(f32));
        // The f16 shaders read the caches as 32-bit words, so each binding
        // covers whole words. The split pass reads K in 16-byte loads, so the
        // K binding covers whole loads. No shader reads the padding.
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

        // The split path writes one row of records at scratch row 0. A record
        // offset does not depend on the row count, so a one-row scratch has the
        // offsets of the first row of a runner scratch.
        const scratch: attn_split.Scratch = .{
            .rows = 1,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .max_splits = attn_split.splitCount(t_len),
        };
        if (deep) {
            const kit = &self.split.?;
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
            updateStorageBindings(self.ctx, kit.split_set, &split_infos);
            updateStorageBindings(self.ctx, kit.merge_set, &merge_infos);
        } else {
            const buf_infos = [_]vk.DescriptorBufferInfo{ q_info, k_info, v_info, out_info };
            updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);
        }

        const cb = self.oneshot.cmd_buffer;
        try self.ctx.vkd.resetCommandBuffer(cb, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(cb, &begin);
        if (deep) {
            const kit = &self.split.?;
            kit.passes.recordSplit(
                cb,
                kit.split_set,
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
            kit.passes.recordMerge(
                cb,
                kit.merge_set,
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

        const cmds = [_]vk.CommandBuffer{cb};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// Split-depth decode attention for a row of more than `limits.attn_split_len`
/// positions. The split pass runs one workgroup of 64 lanes per split and per
/// run of at most `limits.attn_split_queries` query heads of one GQA group.
/// The heads of a run share their K and V rows, so the workgroup reads each
/// row once for all of them. It writes one partial record per head into a
/// scratch that `attn_split.Scratch` lays out. The merge pass runs one
/// workgroup per query head and per 64-element chunk of the head, one lane per
/// element. It combines the records of the row in ascending split order, so
/// one `t_len` gives one result bit for bit. The geometry of a row follows its
/// `t_len` alone. The kernel is record-only: the caller owns
/// the scratch, the descriptor sets, and the shader barrier between the two
/// passes. The declared tolerance is `.relative = 1e-3`.
pub const AttentionSplit = struct {
    ctx: *const vulkan_ctx.Context,
    split: Pipeline,
    merge: Pipeline,

    /// Bindings of a split set: q, the K region, the V region, the scratch.
    pub const split_bindings: u32 = 4;
    /// Bindings of a merge set: the scratch, the output.
    pub const merge_bindings: u32 = 2;
    /// Lanes of a merge workgroup, the `local_size_x` of
    /// `attention_merge.comp`. One lane owns one element of a head.
    const merge_lanes: u32 = 64;

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // The split sums and the merge reorder the softmax and the V sums
        // against the CPU reference, as the LDS trees of `Attention` do.
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
            .f32 => &attention_split_spv,
            .f16 => &attention_split_f16_spv,
        };

        const split = try Pipeline.init(ctx, split_spv, split_bindings, @sizeOf(SplitPush));
        errdefer split.deinit(ctx);

        const merge = try Pipeline.init(
            ctx,
            &attention_merge_spv,
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

    /// Record the split pass of one row, on the grid
    /// `(n_kv_heads * attn_split.queryGroups(n_heads / n_kv_heads),
    /// attn_split.splitCount(t_len))`. The caller binds `set` with
    /// q, the K region, the V region, and the scratch. Row `q_row_base` of q
    /// reads stream `kv_stream_base` of the regions, and the records land in
    /// row `scratch_row` of a record group of `stride` splits per head. The
    /// shader reads `stride` from its `max_splits` push constant.
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
        std.debug.assert(attn_split.usesSplit(t_len));

        const n_splits = attn_split.splitCount(t_len);
        std.debug.assert(n_splits >= 2);
        std.debug.assert(n_splits <= stride);
        std.debug.assert(n_heads >= 1);
        std.debug.assert(n_heads <= max_workgroups_x);
        std.debug.assert(n_kv_heads >= 1);
        std.debug.assert(n_heads % n_kv_heads == 0);
        std.debug.assert(head_dim >= 1);
        // The shader stages q in LDS of `attn_max_head_dim` floats, and it
        // reads K in whole 16-byte loads. `guardHeadDim` refuses a model
        // that breaks either rule at boot.
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
        self.ctx.vkd.cmdDispatch(cb, groups_x, n_splits, 1);
    }

    /// Record the merge pass of one row, on the grid
    /// `(n_heads, ceil(head_dim / merge_lanes))`. The caller binds `set` with
    /// the scratch and the output, and it records a shader barrier between the
    /// split pass of the row and this pass. It
    /// reads the records at the `scratch_row` and `stride` of the split pass.
    /// The result lands in row `out_row` of the output.
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
        std.debug.assert(attn_split.usesSplit(t_len));
        std.debug.assert(attn_split.splitCount(t_len) <= stride);
        std.debug.assert(n_heads >= 1);
        std.debug.assert(n_heads <= max_workgroups_x);
        std.debug.assert(head_dim >= 1);
        // The shader forms the output row as a 32-bit product.
        std.debug.assert((@as(u64, out_row) + 1) * n_heads * head_dim <= limits.max_kv_index_elems);

        assertRecordsIndexable(n_heads, head_dim, scratch_row, stride);
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

    /// Assert the K binding of a split set at the site that writes it. The
    /// split pass reads K with 16-byte loads, so the binding must start on a
    /// 16-byte boundary and hold whole loads. `recordSplit` asserts the row
    /// strides.
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
            "The model head_dim is {d}. GPU attention takes a head_dim that is a multiple " ++
                "of {d} and at most {d}. Run this model on a CPU build",
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

/// The summation-order class of an attention kit. The byte tests (the split
/// digests, and the equality of rows across paths) hold for one class only.
/// A kit of another class runs the reference tests at its declared tolerance and
/// the token gates.
pub const AttnOrderClass = enum {
    /// Each split sums its positions in ascending order, and the merge
    /// combines the splits in ascending order. So a row gets the same bytes at
    /// every batch, slot, and record group.
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

/// The attention kit of `-Dattn-kernel=split`: the resident kernels for a row
/// inside one split, and `AttentionSplit` past it. The kit owns no pipeline.
/// It points at the pipelines of its runner, and `runtime/attn_record.zig`
/// records a plan through it.
pub const SplitAttnKit = struct {
    single: *Attention,
    multi: *AttentionMultiStream,
    passes: *const AttentionSplit,
    /// The resident kernel of a prefill chunk. A kit that records no prefill,
    /// as in the bench grid, leaves it null.
    batch: ?*AttentionBatch = null,

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

    pub fn recordResident(self: SplitAttnKit, cb: Cmd, b: Binding, r: attn_split.PlanRow) void {
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
        self: SplitAttnKit,
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
        self: SplitAttnKit,
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
        self: SplitAttnKit,
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
    pub fn recordBarrier(self: SplitAttnKit, cb: Cmd) void {
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

/// The attention kit that `-Dattn-kernel` selects. A new kit is one struct
/// with the method set of `SplitAttnKit` and one entry in the build enum.
pub const AttnKit = switch (build_options.attn_kernel) {
    .split => SplitAttnKit,
};

/// In-place RoPE rotation on one head_dim vector at the given position. Two
/// storage buffers (x read and write, the frequency factors readonly) and four
/// push constants (pos, rope_dim, freq_base, style). `rope_dim` must be even and
/// at most `x.len`. Lanes stride across the rope_dim/2 pairs at
/// local_size_x = 64. The declared tolerance is `.relative = 5e-5`. Only the
/// benchmark calls this kernel. The per-op path runs the CPU rope, and the
/// fused runners use `RopeMulti`.
pub const Rope = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_x: HostBuffer,
    buf_ff: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // Observed relative error is 1.6e-5 at head_dim 64. The GPU `pow`,
        // `cos`, and `sin` round more loosely than the CPU intrinsics.
        // 5e-5 leaves headroom.
        .tolerance = .{ .relative = 5e-5 },
    };

    const PushConsts = extern struct { pos: u32, rope_dim: u32, freq_base: f32, style_norm: u32 };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!Rope {
        const pipeline = try Pipeline.init(ctx, &rope_spv, 2, @sizeOf(PushConsts));
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

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_ff.buffer, .offset = 0, .range = ff_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

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

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(x, self.buf_x.sliceTyped(f32, x.len));
    }
};

/// Multi-head RoPE. Same rotation as `Rope`, applied to every head of an
/// `(n_heads, head_dim)` buffer in one dispatch. It replaces the per-head Rope
/// dispatch loop, which costs 40 dispatches per layer on a Llama-class model
/// against 2 here. `rope_dim` must be even. The grid is
/// (ceil(rope_dim / 2 / 64), n_heads). The declared tolerance is
/// `.relative = 1e-5`. `recordInto` is the only entry point, because every
/// caller records into its own command buffer.
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
        const pipeline = try Pipeline.init(ctx, &rope_multi_spv, 2, @sizeOf(PushConsts));

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

/// RMS normalization `y[i] = x[i] * inv_rms * w[i]`, where
/// `inv_rms = 1 / sqrt(mean(x^2) + eps)`. Three storage buffers (x, w, y) and
/// two push constants (d, eps). The three slices must have the same length. A
/// single workgroup of 256 lanes tree-reduces in LDS, then writes y across the
/// lanes. The tree-reduce sums in a different order than the scalar CPU
/// reference, so the result is not bit-exact. The declared tolerance is
/// `.relative = 5e-5`.
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
        const pipeline = try Pipeline.init(ctx, &rmsnorm_spv, 3, @sizeOf(PushConsts));
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

        const d: u32 = @intCast(x.len);
        const byte_size: vk.DeviceSize = @intCast(x.len * @sizeOf(f32));

        try self.buf_x.ensure(byte_size);
        try self.buf_w.ensure(byte_size);
        try self.buf_y.ensure(byte_size);

        @memcpy(self.buf_x.sliceTyped(f32, x.len), x);
        @memcpy(self.buf_w.sliceTyped(f32, w.len), w);

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_w.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_y.buffer, .offset = 0, .range = byte_size },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

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

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(y, self.buf_y.sliceTyped(f32, y.len));
    }
};

/// SwiGLU activation `dst[i] = silu(gate[i]) * up[i]`. Three storage buffers
/// (gate and up readonly, dst writeonly) and one `u32` push constant for the
/// length. The three slices must have the same length. One lane per element at
/// local_size_x = 256. The GPU `exp` rounds differently from the CPU
/// `std.math.exp`, so the result is not bit-exact. The declared tolerance is
/// `.relative = 1e-5`.
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
        const pipeline = try Pipeline.init(ctx, &swiglu_spv, 3, @sizeOf(u32));
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

        const n: u32 = @intCast(gate.len);
        const byte_size: vk.DeviceSize = @intCast(gate.len * @sizeOf(f32));

        try self.buf_gate.ensure(byte_size);
        try self.buf_up.ensure(byte_size);
        try self.buf_dst.ensure(byte_size);

        @memcpy(self.buf_gate.sliceTyped(f32, gate.len), gate);
        @memcpy(self.buf_up.sliceTyped(f32, up.len), up);

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_gate.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_up.buffer, .offset = 0, .range = byte_size },
            .{ .buffer = self.buf_dst.buffer, .offset = 0, .range = byte_size },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

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

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(dst, self.buf_dst.sliceTyped(f32, dst.len));
    }
};

/// A GPU-resident Q8_0 weight matrix. It owns the f16 scales buffer, the packed
/// quants buffer, and the dimensions. A caller builds it once from a CPU-side
/// weight blob, and `LinearQ80.dispatch` or `LinearQ80Reduce.dispatch` then
/// consumes it many times.
///
/// The split between the build and the dispatch is the point. A repack and
/// re-upload on every matmul is ruinous: the lm_head alone is about 27 MB, which
/// dominates the GPU share of the benchmark. So the upload leaves the per-call
/// path. Each weight becomes one `Q80Weight` at model-load time and stays in
/// VRAM for the life of the model. Every Q8_0 and Q4_0 kernel here reads the
/// same scales-plus-quants layout, so this type serves all of them.
pub const Q80Weight = struct {
    ctx: *const vulkan_ctx.Context,
    /// View into device-local memory: (buffer, offset, range). Either
    /// references arena-owned memory (when `owned` is null) or a
    /// privately-allocated DeviceBuffer (when `owned` is non-null).
    /// Shape matches `vk.DescriptorBufferInfo` so callers can write
    /// it directly into a descriptor update.
    scales: vk.DescriptorBufferInfo,
    quants: vk.DescriptorBufferInfo,
    in_features: u32,
    out_features: u32,
    /// Set when this weight allocated its own memory through `init`, and
    /// `deinit` then frees it. When null, the memory belongs to the
    /// `VramArena` values the caller owns, and `deinit` frees nothing.
    owned: ?Owned = null,

    pub const Owned = struct {
        scales_memory: vk.DeviceMemory,
        quants_memory: vk.DeviceMemory,
    };

    pub const InitError = error{
        InFeaturesNotMultipleOf32,
        WeightSizeMismatch,
        NoSuitableMemoryType,
    } ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError ||
        vk.DeviceWrapper.MapMemoryError ||
        vk.DeviceWrapper.AllocateCommandBuffersError ||
        vk.DeviceWrapper.BeginCommandBufferError ||
        vk.DeviceWrapper.EndCommandBufferError ||
        vk.DeviceWrapper.QueueSubmitError ||
        vk.DeviceWrapper.QueueWaitIdleError;

    /// Build a GPU-resident weight with its own dedicated device-memory
    /// allocations (one per buffer). Each call produces 2 VkBuffers +
    /// 2 VkDeviceMemory objects. Use this for one-off uses (bench,
    /// per-op tests, lazy wrap.zig cache). For long-lived bulk uploads
    /// (LlamaFusedRunner loading 200+ weights at init) use `initInArenas`
    /// instead. The arena version produces ~10 device allocations
    /// total instead of 2 per weight.
    pub fn init(
        ctx: *const vulkan_ctx.Context,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
    ) InitError!Q80Weight {
        if (in_features % q8_0_block_elems != 0) return error.InFeaturesNotMultipleOf32;

        const blocks_per_row = in_features / q8_0_block_elems;
        const row_bytes = blocks_per_row * q8_0_block_bytes;
        if (weight_bytes.len < out_features * row_bytes) return error.WeightSizeMismatch;

        const scales_bytes: vk.DeviceSize = @intCast(out_features * blocks_per_row * 2);
        const quants_bytes: vk.DeviceSize = @intCast(out_features * in_features);

        var staging_scales: HostBuffer = .{ .ctx = ctx };
        defer staging_scales.deinit();

        var staging_quants: HostBuffer = .{ .ctx = ctx };
        defer staging_quants.deinit();

        try staging_scales.ensure(scales_bytes);
        try staging_quants.ensure(quants_bytes);
        repackQ80ToScalesQuants(
            weight_bytes,
            in_features,
            out_features,
            staging_scales.sliceTyped(u8, @intCast(scales_bytes)),
            staging_quants.sliceTyped(u8, @intCast(quants_bytes)),
        );

        var scales_db: DeviceBuffer = .{ .ctx = ctx };
        errdefer scales_db.deinit();

        var quants_db: DeviceBuffer = .{ .ctx = ctx };
        errdefer quants_db.deinit();

        try scales_db.ensure(scales_bytes);
        try quants_db.ensure(quants_bytes);

        const copies = [_]Copy{
            .{ .src = staging_scales.buffer, .dst = scales_db.buffer, .size = scales_bytes },
            .{ .src = staging_quants.buffer, .dst = quants_db.buffer, .size = quants_bytes },
        };
        try runStagingCopies(ctx, &copies);

        return .{
            .ctx = ctx,
            .scales = .{ .buffer = scales_db.buffer, .offset = 0, .range = scales_bytes },
            .quants = .{ .buffer = quants_db.buffer, .offset = 0, .range = quants_bytes },
            .in_features = @intCast(in_features),
            .out_features = @intCast(out_features),
            .owned = .{
                .scales_memory = scales_db.memory,
                .quants_memory = quants_db.memory,
            },
        };
    }

    /// Q4_0 variant of init. Same container (scales f16 + packed quants),
    /// but the quants buffer is HALF the size (16 bytes/block vs 32) and
    /// holds 4-bit nibbles. Pair with the LinearQ40* kernels, which do the
    /// nibble extraction. `weight_bytes` is a Q4_0 tensor (18 bytes/block).
    pub fn initQ40(
        ctx: *const vulkan_ctx.Context,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
    ) InitError!Q80Weight {
        if (in_features % q4_0_block_elems != 0) return error.InFeaturesNotMultipleOf32;

        const blocks_per_row = in_features / q4_0_block_elems;
        const row_bytes = blocks_per_row * q4_0_block_bytes;
        if (weight_bytes.len < out_features * row_bytes) return error.WeightSizeMismatch;

        const scales_bytes: vk.DeviceSize = @intCast(out_features * blocks_per_row * 2);
        const quants_bytes: vk.DeviceSize = @intCast(out_features * (in_features / 2));

        var staging_scales: HostBuffer = .{ .ctx = ctx };
        defer staging_scales.deinit();

        var staging_quants: HostBuffer = .{ .ctx = ctx };
        defer staging_quants.deinit();

        try staging_scales.ensure(scales_bytes);
        try staging_quants.ensure(quants_bytes);
        repackQ40ToScalesQuants(
            weight_bytes,
            in_features,
            out_features,
            staging_scales.sliceTyped(u8, @intCast(scales_bytes)),
            staging_quants.sliceTyped(u8, @intCast(quants_bytes)),
        );

        var scales_db: DeviceBuffer = .{ .ctx = ctx };
        errdefer scales_db.deinit();

        var quants_db: DeviceBuffer = .{ .ctx = ctx };
        errdefer quants_db.deinit();

        try scales_db.ensure(scales_bytes);
        try quants_db.ensure(quants_bytes);

        const copies = [_]Copy{
            .{ .src = staging_scales.buffer, .dst = scales_db.buffer, .size = scales_bytes },
            .{ .src = staging_quants.buffer, .dst = quants_db.buffer, .size = quants_bytes },
        };
        try runStagingCopies(ctx, &copies);

        return .{
            .ctx = ctx,
            .scales = .{ .buffer = scales_db.buffer, .offset = 0, .range = scales_bytes },
            .quants = .{ .buffer = quants_db.buffer, .offset = 0, .range = quants_bytes },
            .in_features = @intCast(in_features),
            .out_features = @intCast(out_features),
            .owned = .{
                .scales_memory = scales_db.memory,
                .quants_memory = quants_db.memory,
            },
        };
    }

    /// Sub-allocate scales + quants from two caller-managed VramArenas
    /// and upload via staging. The arenas own the underlying VkBuffer
    /// + VkDeviceMemory. This Q80Weight only stores (buffer, offset,
    /// range) views into them. `deinit` is a no-op for the weight's
    /// storage. The arenas free everything when they themselves are
    /// deinit-ed.
    pub fn initInArenas(
        ctx: *const vulkan_ctx.Context,
        scales_arena: *VramArena,
        quants_arena: *VramArena,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
    ) (InitError || VramArena.AllocError)!Q80Weight {
        if (in_features % q8_0_block_elems != 0) return error.InFeaturesNotMultipleOf32;

        const blocks_per_row = in_features / q8_0_block_elems;
        const row_bytes = blocks_per_row * q8_0_block_bytes;
        if (weight_bytes.len < out_features * row_bytes) return error.WeightSizeMismatch;

        const scales_bytes: vk.DeviceSize = @intCast(out_features * blocks_per_row * 2);
        const quants_bytes: vk.DeviceSize = @intCast(out_features * in_features);

        var staging_scales: HostBuffer = .{ .ctx = ctx };
        defer staging_scales.deinit();

        var staging_quants: HostBuffer = .{ .ctx = ctx };
        defer staging_quants.deinit();

        try staging_scales.ensure(scales_bytes);
        try staging_quants.ensure(quants_bytes);
        repackQ80ToScalesQuants(
            weight_bytes,
            in_features,
            out_features,
            staging_scales.sliceTyped(u8, @intCast(scales_bytes)),
            staging_quants.sliceTyped(u8, @intCast(quants_bytes)),
        );

        const scales_view = try scales_arena.alloc(scales_bytes);
        const quants_view = try quants_arena.alloc(quants_bytes);

        const copies = [_]Copy{
            .{ .src = staging_scales.buffer, .dst = scales_view.buffer, .size = scales_bytes, .dst_offset = scales_view.offset },
            .{ .src = staging_quants.buffer, .dst = quants_view.buffer, .size = quants_bytes, .dst_offset = quants_view.offset },
        };
        try runStagingCopies(ctx, &copies);

        return .{
            .ctx = ctx,
            .scales = scales_view,
            .quants = quants_view,
            .in_features = @intCast(in_features),
            .out_features = @intCast(out_features),
            // The arenas own the memory, so `owned` stays null.
        };
    }

    /// Q4_0 arena upload (the LlamaFusedRunner path). Like initInArenas but the
    /// quants buffer is HALF size (16 bytes/block) and holds 4-bit nibbles.
    /// Pair with the LinearQ40* kernels. `weight_bytes` is a Q4_0 tensor.
    pub fn initInArenasQ40(
        ctx: *const vulkan_ctx.Context,
        scales_arena: *VramArena,
        quants_arena: *VramArena,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
    ) (InitError || VramArena.AllocError)!Q80Weight {
        if (in_features % q4_0_block_elems != 0) return error.InFeaturesNotMultipleOf32;

        const blocks_per_row = in_features / q4_0_block_elems;
        const row_bytes = blocks_per_row * q4_0_block_bytes;
        if (weight_bytes.len < out_features * row_bytes) return error.WeightSizeMismatch;

        const scales_bytes: vk.DeviceSize = @intCast(out_features * blocks_per_row * 2);
        const quants_bytes: vk.DeviceSize = @intCast(out_features * (in_features / 2));

        var staging_scales: HostBuffer = .{ .ctx = ctx };
        defer staging_scales.deinit();

        var staging_quants: HostBuffer = .{ .ctx = ctx };
        defer staging_quants.deinit();

        try staging_scales.ensure(scales_bytes);
        try staging_quants.ensure(quants_bytes);
        repackQ40ToScalesQuants(
            weight_bytes,
            in_features,
            out_features,
            staging_scales.sliceTyped(u8, @intCast(scales_bytes)),
            staging_quants.sliceTyped(u8, @intCast(quants_bytes)),
        );

        const scales_view = try scales_arena.alloc(scales_bytes);
        const quants_view = try quants_arena.alloc(quants_bytes);

        const copies = [_]Copy{
            .{ .src = staging_scales.buffer, .dst = scales_view.buffer, .size = scales_bytes, .dst_offset = scales_view.offset },
            .{ .src = staging_quants.buffer, .dst = quants_view.buffer, .size = quants_bytes, .dst_offset = quants_view.offset },
        };
        try runStagingCopies(ctx, &copies);

        return .{
            .ctx = ctx,
            .scales = scales_view,
            .quants = quants_view,
            .in_features = @intCast(in_features),
            .out_features = @intCast(out_features),
        };
    }

    pub fn deinit(self: *Q80Weight) void {
        if (self.owned) |o| {
            self.ctx.vkd.destroyBuffer(self.ctx.device, self.scales.buffer, null);
            self.ctx.vkd.freeMemory(self.ctx.device, o.scales_memory, null);
            self.ctx.vkd.destroyBuffer(self.ctx.device, self.quants.buffer, null);
            self.ctx.vkd.freeMemory(self.ctx.device, o.quants_memory, null);
            self.owned = null;
        }
    }
};

/// Q8_0 matrix-vector product `out = input @ weight^T`, the same shape as the
/// CPU `linearF32Q80` op. `input` is `[k]f32` and `out` is `[n]f32`. The weight
/// is `[n, k]` Q8_0 in the GGUF layout. Each row holds k/32 blocks of 34 bytes,
/// a 2-byte f16 scale plus 32 i8 quants. `k` must be a multiple of 32. One
/// workgroup of 32 lanes covers 32 output rows, so the grid is ceil(n / 32).
/// This struct owns only the per-dispatch resources: the descriptor pool and
/// set, the command buffer, and the host-visible input and output buffers. The
/// scales and quants live in a `Q80Weight` that the caller builds once and
/// passes to `dispatch`. The shader keeps the scalar reduction order of the CPU
/// reference, so the result is bit-exact.
pub const LinearQ80 = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_in: HostBuffer,
    buf_out: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // The linear_q8_0.comp shader keeps the scalar reduction order of the
        // CPU reference and tags every f32 in the reduction `precise`, so the
        // output is bit-identical. `tests/vulkan_linear_q8_0_test.zig` holds
        // the check.
        .tolerance = .bit_exact,
    };

    /// `cmdPushConstants` payload: must match `linear_q8_0.comp`'s
    /// `Params` block. `extern` so the field order is guaranteed.
    const PushConsts = extern struct { k: u32, n: u32 };

    pub const InitError = VecAdd.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ80 {
        const pipeline = try Pipeline.init(ctx, &linear_q8_0_spv, 4, @sizeOf(PushConsts));
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

    /// `out = input @ weight^T` with the weight already resident in VRAM.
    /// `input.len` must equal `weight.in_features`, and `out.len` must equal
    /// `weight.out_features`. The per-call cost is the kilobyte-scale input and
    /// output copies, a descriptor update, and the dispatch. The weight repack
    /// happens once, when the `Q80Weight` is built.
    pub fn dispatch(
        self: *LinearQ80,
        weight: *const Q80Weight,
        input: []const f32,
        out: []f32,
    ) DispatchError!void {
        if (input.len != weight.in_features) return error.InputShapeMismatch;
        if (out.len != weight.out_features) return error.OutputShapeMismatch;

        const in_bytes: vk.DeviceSize = @intCast(input.len * @sizeOf(f32));
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_in.ensure(in_bytes);
        try self.buf_out.ensure(out_bytes);

        @memcpy(self.buf_in.sliceTyped(f32, input.len), input);

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_in.buffer, .offset = 0, .range = in_bytes },
            weight.scales,
            weight.quants,
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = out_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

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
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, group_count_x, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

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

/// Workgroup-cooperative Q8_0 matvec, the same shape and storage layout as
/// `LinearQ80`. One workgroup of 32 lanes, one RDNA2 wave, computes a single
/// output row through partial sums in LDS. The grid is n workgroups against
/// ceil(n / 32) for `LinearQ80`, which gives 32 times more independent work
/// units. That is the throughput win when n is too small to keep every compute
/// unit busy under one invocation per output. The cross-lane reduce breaks
/// bit-exactness, so this variant declares `.relative = 2e-4`.
pub const LinearQ80Reduce = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_in: HostBuffer,
    buf_out: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // Each lane sums its own partials precisely, then lane 0 reduces the
        // 32 partials in lane order. That cross-lane step differs from the
        // scalar CPU path. The drift grows with k: 8.5e-5 at k=576 and about
        // 1e-4 at k=14336. 2e-4 leaves headroom through the widest Llama-8B
        // shape, ffn_down at k=14336.
        .tolerance = .{ .relative = 2e-4 },
    };

    const PushConsts = extern struct { k: u32, n: u32 };

    pub const InitError = LinearQ80.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ80Reduce {
        const pipeline = try Pipeline.init(ctx, &linear_q8_0_reduce_spv, 4, @sizeOf(PushConsts));
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

    /// Record one Q8_0 matmul into a command buffer the caller owns. The
    /// caller must do two things:
    ///   1. Point `desc_set` at (input, scales, quants, output) before it
    ///      submits the command buffer. The driver holds the descriptor update
    ///      until the next submit of a command buffer that reads the set.
    ///   2. Insert the memory barriers the dispatch needs.
    /// The fused runners chain many such dispatches into one command buffer and
    /// pay the submit and wait-idle cost once.
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
        self.ctx.vkd.cmdDispatch(cb, n, 1, 1);
    }

    pub fn dispatch(
        self: *LinearQ80Reduce,
        weight: *const Q80Weight,
        input: []const f32,
        out: []f32,
    ) DispatchError!void {
        if (input.len != weight.in_features) return error.InputShapeMismatch;
        if (out.len != weight.out_features) return error.OutputShapeMismatch;

        const in_bytes: vk.DeviceSize = @intCast(input.len * @sizeOf(f32));
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_in.ensure(in_bytes);
        try self.buf_out.ensure(out_bytes);

        @memcpy(self.buf_in.sliceTyped(f32, input.len), input);

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_in.buffer, .offset = 0, .range = in_bytes },
            weight.scales,
            weight.quants,
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = out_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

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
        const group_count_x: u32 = weight.out_features;
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, group_count_x, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// Q4_0 decode matvec. Same structure as `LinearQ80Reduce`: one workgroup of 32
/// lanes per output row, reducing through LDS. The weight container is a
/// `Q80Weight` built by `Q80Weight.initQ40`, with f16 scales and 16 bytes of
/// 4-bit quants per block. The shader extracts the nibbles. `k` must be a
/// multiple of 32. The declared tolerance is `.relative = 2e-4`.
pub const LinearQ40Reduce = struct {
    ctx: *const vulkan_ctx.Context,
    pipeline: Pipeline,
    oneshot: OneShot,
    buf_in: HostBuffer,
    buf_out: HostBuffer,

    pub const requires: op_requires.OpRequires = .{
        .arch = .any,
        .runtime = .vulkan,
        // The Q4 quantization error belongs to the data, not the kernel. The
        // bench measures relative error against the Q4_0 CPU reference, so the
        // cross-lane reduce drift is the only error the kernel adds. That puts
        // it in the same class as the Q8_0 reduce.
        .tolerance = .{ .relative = 2e-4 },
    };

    const PushConsts = extern struct { k: u32, n: u32 };
    pub const InitError = LinearQ80.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!LinearQ40Reduce {
        const pipeline = try Pipeline.init(ctx, &linear_q4_0_reduce_spv, 4, @sizeOf(PushConsts));
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
        self.ctx.vkd.cmdDispatch(cb, n, 1, 1);
    }

    pub fn dispatch(
        self: *LinearQ40Reduce,
        weight: *const Q80Weight,
        input: []const f32,
        out: []f32,
    ) DispatchError!void {
        if (input.len != weight.in_features) return error.InputShapeMismatch;
        if (out.len != weight.out_features) return error.OutputShapeMismatch;

        const in_bytes: vk.DeviceSize = @intCast(input.len * @sizeOf(f32));
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_in.ensure(in_bytes);
        try self.buf_out.ensure(out_bytes);

        @memcpy(self.buf_in.sliceTyped(f32, input.len), input);

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_in.buffer, .offset = 0, .range = in_bytes },
            weight.scales,
            weight.quants,
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = out_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

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
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, weight.out_features, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// Q8_0 matvec, the subgroup variant. Same shape as `LinearQ80Reduce`, one
/// workgroup of 32 lanes per output row, with two changes. The workgroup stages
/// the whole input into LDS at entry, so each invocation reads it from shared
/// memory instead of the global buffer. The cross-lane reduce is one
/// `subgroupAdd` instead of an LDS store, a barrier, and a scalar sum on lane 0.
/// `weight.in_features` must be at most `lds_capacity`. The declared tolerance
/// is `.relative = 2e-4`, the same class as the reduce variant, because the
/// lanes sum precisely and only `subgroupAdd` reorders.
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
        const pipeline = try Pipeline.init(ctx, &linear_q8_0_subgroup_spv, 4, @sizeOf(PushConsts));
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

    /// The `shared float input_lds[]` size in `linear_q8_0_subgroup.comp`. The
    /// build gives the shader the same value as the LDS_CAPACITY define.
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

        const in_bytes: vk.DeviceSize = @intCast(input.len * @sizeOf(f32));
        const out_bytes: vk.DeviceSize = @intCast(out.len * @sizeOf(f32));

        try self.buf_in.ensure(in_bytes);
        try self.buf_out.ensure(out_bytes);

        @memcpy(self.buf_in.sliceTyped(f32, input.len), input);

        const buf_infos = [_]vk.DescriptorBufferInfo{
            .{ .buffer = self.buf_in.buffer, .offset = 0, .range = in_bytes },
            weight.scales,
            weight.quants,
            .{ .buffer = self.buf_out.buffer, .offset = 0, .range = out_bytes },
        };
        updateStorageBindings(self.ctx, self.oneshot.desc_set, &buf_infos);

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
        const group_count_x: u32 = weight.out_features;
        self.ctx.vkd.cmdDispatch(self.oneshot.cmd_buffer, group_count_x, 1, 1);
        try self.ctx.vkd.endCommandBuffer(self.oneshot.cmd_buffer);

        const cmds = [_]vk.CommandBuffer{self.oneshot.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);

        @memcpy(out, self.buf_out.sliceTyped(f32, out.len));
    }
};

/// Batched Q8_0 matmul: a `(B, K)` input against an `(N, K)` weight into a
/// `(B, N)` output. Each workgroup computes a 16 by 32 output tile, which
/// amortizes the weight read across 16 batch rows. `k` must be a multiple of 32.
/// The grid is (ceil(n / 32), ceil(b / 16)), and the kernel covers a partial
/// tail tile. This is the prefill kernel, where B is large. Decode keeps
/// `LinearQ80Reduce`, because per-call overhead dominates at B = 1. The declared
/// tolerance is `.relative = 2e-4`, the same as the reduce variant: the math is
/// identical and only the tiling differs.
pub const LinearQ80Batch = struct {
    /// The build gives `linear_q8_0_batch.comp` the same two values as the
    /// B_TILE and N_TILE defines.
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
        const pipeline = try Pipeline.init(ctx, &linear_q8_0_batch_spv, 4, @sizeOf(PushConsts));

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *LinearQ80Batch) void {
        self.pipeline.deinit(self.ctx);
    }

    /// Record one batched Q8_0 matmul. The caller binds the descriptor set
    /// (input, scales, quants, output). `b` is the batch size of this dispatch.
    /// The kernel covers the tail tile when `b` is not a multiple of
    /// `batch_tile`.
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
        self.ctx.vkd.cmdDispatch(cb, wg_n, wg_b, 1);
    }

    /// One-shot batched matmul for the correctness gate and the bench. See
    /// `matmulOneShot`. `input` is `[b, in_features]` row-major and `out` is
    /// `[b, out_features]`. This is not the hot path. The runner uses
    /// `recordInto` against pre-bound device buffers.
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

/// Register-blocked, LDS-staged Q8_0 GEMM, the wide prefill matmul. It takes
/// the same storage bindings and push constants as `LinearQ80Batch`, so the
/// runner swaps one for the other at `recordInto`. A workgroup of 256 threads,
/// eight RDNA2 waves, computes a 64 by 64 output tile, and each thread owns a
/// 4 by 4 micro-tile. The input rows and the dequantized weight tile stage
/// through LDS. `k` must be a multiple of 32. The grid is
/// (ceil(n / 64), ceil(b / 64)). The declared tolerance is `.relative = 2e-4`.
pub const LinearQ80Gemm = struct {
    /// The build gives `linear_q8_0_gemm.comp` the same two values as the BM
    /// and BN defines.
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
        const pipeline = try Pipeline.init(ctx, &linear_q8_0_gemm_spv, 4, @sizeOf(PushConsts));

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

/// Q4_0 prefill matmul. It takes the same storage bindings and push constants
/// as `LinearQ80Batch`, with an 8 by 32 output tile and LDS input staging. The
/// weight container is a `Q80Weight` built by `Q80Weight.initQ40`, whose quants
/// buffer holds 16 bytes per block. The shader extracts the nibbles. `k` must be
/// a multiple of 32. The grid is (ceil(n / 32), ceil(b / 8)). The declared
/// tolerance is `.relative = 2e-4`.
pub const LinearQ40Batch = struct {
    /// The build gives `linear_q4_0_batch.comp` the same two values as the
    /// B_TILE and N_TILE defines.
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
        const pipeline = try Pipeline.init(ctx, &linear_q4_0_batch_spv, 4, @sizeOf(PushConsts));

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
        self.ctx.vkd.cmdDispatch(cb, wg_n, wg_b, 1);
    }
};

/// Batched RMS normalization, applied independently to each of B input rows.
/// One workgroup of 256 lanes per row shares the reduction, so the grid is b.
/// The arithmetic matches the decode `Rmsnorm`, and the declared tolerance is
/// `.relative = 5e-5`.
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
        const pipeline = try Pipeline.init(ctx, &rmsnorm_batch_spv, 3, @sizeOf(PushConsts));

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

/// Batched multi-head RoPE over a `(B, n_heads, head_dim)` buffer. Row `bi`
/// rotates at position `start_pos + bi * pos_stride`. One dispatch covers every
/// (head, row) pair, so the grid is (ceil(rope_dim / 2 / 64), n_heads, b). The
/// declared tolerance is `.relative = 1e-5`.
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
        const pipeline = try Pipeline.init(ctx, &rope_multi_batch_spv, 2, @sizeOf(PushConsts));

        return .{
            .ctx = ctx,
            .pipeline = pipeline,
        };
    }

    pub fn deinit(self: *RopeMultiBatch) void {
        self.pipeline.deinit(self.ctx);
    }

    /// `pos_stride`: 1 for prefill (each batch row has position
    /// `start_pos + bi`, monotonically increasing). 0 for lockstep
    /// or single-row dispatches (every row uses `start_pos`).
    /// `row_base` offsets which buffer rows the dispatch touches, so
    /// arbitrary per-row positions record as B=1 dispatches per row.
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

/// Batched causal attention for the resident rows of a prefill chunk, one
/// workgroup of 64 lanes per (head, row) pair. Query row `bi` attends to the K
/// and V positions 0 through `start_pos + bi`. The kernel holds every score of
/// a row in LDS, so `start_pos + b` must be at most `limits.attn_split_len`.
/// A deeper row takes the split passes of `AttentionSplit`, through
/// `attn_record.recordPrefill`. The grid is (n_heads, b). The declared
/// tolerance is `.relative = 1e-3`.
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

    /// KV-dtype variant selector. See `Attention.initKv`.
    pub fn initKv(ctx: *const vulkan_ctx.Context, kv: dtype.KvDtype) InitError!AttentionBatch {
        const spv: []align(@alignOf(u32)) const u8 = switch (kv) {
            .f32 => &attention_batch_spv,
            .f16 => &attention_batch_f16_spv,
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
        std.debug.assert(b >= 1);
        // Row bi attends to positions 0 through start_pos + bi, so the last
        // row reaches start_pos + b. The LDS score row holds one split.
        std.debug.assert(@as(u64, start_pos) + b <= limits.attn_split_len);
        std.debug.assert(n_kv_heads >= 1);
        std.debug.assert(n_heads % n_kv_heads == 0);
        // The K-dot loops read an adjacent element pair per step, on both KV
        // element types, so an odd head_dim reads past the row.
        std.debug.assert(head_dim % 2 == 0);

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

/// Multi-stream attention, one row per stream. Each row reads the K and V slice
/// of its own stream from the K region and the V region of one layer. Every
/// stream shares `t_len`, which is what lockstep decoding gives. The grid is
/// (n_heads, n_rows) of 64 lanes. The bounds match the decode kernel, so a row
/// past `limits.attn_split_len` positions takes `AttentionSplit`. The declared
/// tolerance is `.relative = 1e-3`.
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

    /// KV-dtype variant selector. See `Attention.initKv`.
    pub fn initKv(
        ctx: *const vulkan_ctx.Context,
        kv: dtype.KvDtype,
    ) InitError!AttentionMultiStream {
        const spv: []align(@alignOf(u32)) const u8 = switch (kv) {
            .f32 => &attention_multi_stream_spv,
            .f16 => &attention_multi_stream_f16_spv,
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

    /// One dispatch covers `n_rows` (q-row, kv-stream) pairs where row i is
    /// (q_row_base + i, kv_stream_base + i), all sharing `t_len`. A caller
    /// with arbitrary per-row (slot, position) pairs records one single-row
    /// dispatch per row instead.
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

/// Narrow the K and V commit scratch of one layer from f32 to f16. Four storage
/// buffers: f32 k_src, f32 v_src, f16 k_out, f16 v_out. One `u32` push constant
/// holds the element count of one scratch, which K and V share. One lane per
/// element at local_size_x = 64.
///
/// Record-only, like the batched attention kernel. A fused runner owns the
/// buffers, the descriptor sets, and the per-token command buffer. One dispatch
/// covers the commit of a whole layer whatever the row count, because the
/// scratch is contiguous. The scatter that follows stays a plain copy list.
///
/// The shader pins the SPIR-V `RoundingModeRTE` execution mode at width 16, so
/// the conversion rounds to nearest even by declaration and not by driver
/// default. One IEEE 754 conversion per element carries no reordering, so the
/// GPU commit and the CPU commit narrow a value bit for bit alike.
/// `tests/vulkan_kv_commit_f16_test.zig` gates that identity through
/// `kvCommitF16OneShot`.
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
        const pipeline = try Pipeline.init(ctx, &kv_commit_f16_spv, 4, @sizeOf(PushConsts));

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

/// Gated-DeltaNet decode kernel for one Qwen3.5 GDN layer, one token per call.
/// This is the standalone bit-check form: host-visible buffers, submit and
/// wait-idle, then readback. It mirrors `step()` in `src/ops/gdn.zig`. One
/// workgroup of 128 lanes runs per KEY head, so the grid is n_k_heads.
/// `head_dim` must be 128, and `n_v_heads` must be a multiple of `n_k_heads`.
/// It mutates `state` and `conv_state` and writes `o_out`. Eleven storage
/// bindings, listed in gdn_decode.comp. The declared tolerance is
/// `.relative = 1e-4`.
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
        // The multi-slot form: the first batched row, plus the ELEMENT offsets
        // of the region of the slot inside the slot-strided state and conv
        // buffers. All zero reproduces the single-slot dispatch bit for bit.
        row0: u32 = 0,
        state_off: u32 = 0,
        conv_off: u32 = 0,
    };

    pub const InitError = error{NoSuitableMemoryType} ||
        Pipeline.InitError || OneShot.InitError;

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!GdnDecode {
        const pipeline = try Pipeline.init(ctx, &gdn_decode_spv, n_bindings, @sizeOf(Push));
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

    /// Record one single-token GDN decode dispatch into a command buffer the
    /// caller owns, against a descriptor set the caller binds. That set holds
    /// the 11 bindings listed in gdn_decode.comp, pointing at persistent device
    /// buffers. This is the fused-runner form: no buffer handling and no submit.
    /// One workgroup per KEY head, and `head_dim` must be 128.
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

    /// Record the GDN recurrence for a whole prefill chunk as one dispatch. The
    /// kernel loops `n_tokens` times inside. Token t reads and writes row t of
    /// the batched projection and output buffers, and the recurrent state and
    /// the conv ring carry across t. The result is bit-identical to `n_tokens`
    /// per-token dispatches, and it pays the host dispatch and pipeline-barrier
    /// cost once rather than per token. That is the prefill win.
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

    /// Record one slot run of a mixed batch. It covers `n_tokens` rows of the
    /// same slot, starting at batched-buffer row `row0`. It reads the state and
    /// conv regions of that slot at the given ELEMENT offsets. Runs of different
    /// slots touch disjoint state and disjoint rows, so they need no barrier.
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
        const key_dim: usize = @as(usize, n_k_heads) * head_dim;
        const value_dim: usize = @as(usize, n_v_heads) * head_dim;
        const conv_dim: usize = 2 * key_dim + value_dim;
        if (head_dim != 128 or n_v_heads % n_k_heads != 0) return error.DimMismatch;
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

    /// Run the prefill path: `n_tokens` tokens through one `recordChunk`
    /// dispatch, with the recurrent state and the conv ring carried inside. The
    /// batched buffers (qkv, z, alpha, beta_in, o_out) hold n_tokens rows each,
    /// and the weights, the state, and conv_state hold one. This is the
    /// bit-check counterpart of `run` for n_tokens above 1. The self-test feeds
    /// the same token sequence through the CPU `gdn.step` n_tokens times and
    /// compares.
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
        const t: usize = n_tokens;
        const key_dim: usize = @as(usize, n_k_heads) * head_dim;
        const value_dim: usize = @as(usize, n_v_heads) * head_dim;
        const conv_dim: usize = 2 * key_dim + value_dim;
        if (head_dim != 128 or n_v_heads % n_k_heads != 0) return error.DimMismatch;
        if (qkv.len != t * conv_dim or z.len != t * value_dim or o_out.len != t * value_dim) {
            return error.DimMismatch;
        }
        if (alpha.len != t * n_v_heads or beta_in.len != t * n_v_heads) return error.DimMismatch;
        if (ssm_a.len != n_v_heads or dt_bias.len != n_v_heads) return error.DimMismatch;
        if (conv_w.len != @as(usize, conv_kernel) * conv_dim) return error.DimMismatch;
        if (ssm_norm.len != head_dim) return error.DimMismatch;
        if (state.len != value_dim * head_dim) return error.DimMismatch;
        if (conv_state.len != @as(usize, conv_kernel - 1) * conv_dim) return error.DimMismatch;

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

/// Chunked-parallel GDN prefill, a replacement for the serial-over-tokens
/// `recordChunk` of `GdnDecode`. The conv pass runs one workgroup of 128 lanes
/// per KEY head. It does the 1-D convolution, the SiLU, and the L2 norm, writes
/// q_n, k_n, and v_conv, and advances conv_state. The scan pass runs
/// `limits.gdnScanGroups` workgroups, one per block of `limits.gdn_scan_cols`
/// state columns of a VALUE head. It does the chunked matrix-form delta-rule
/// scan, which mirrors `chunkScan` in ops/gdn.zig, and writes the raw readout.
/// The norm pass runs one workgroup per (value head, token) and applies the
/// gated norm in place. `head_dim` must be `limits.gdn_head_dim`. The declared
/// tolerance is `.relative = 1e-4`.
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

    /// Refuse a model whose GDN state size the scan cannot take, before the
    /// first allocation. A ReleaseFast build compiles the asserts of
    /// `recordScan` out.
    pub fn guardHeadDim(head_dim: u32) error{GdnHeadDimUnsupported}!void {
        if (head_dim == limits.gdn_head_dim) return;
        log.err(
            "The model GDN state size is {d}. GPU prefill takes a state size of {d} only. " ++
                "Run this model on a CPU build",
            .{ head_dim, limits.gdn_head_dim },
        );
        return error.GdnHeadDimUnsupported;
    }

    pub fn init(ctx: *const vulkan_ctx.Context) InitError!GdnChunkParallel {
        const pipeline_conv = try Pipeline.init(
            ctx,
            &gdn_chunk_conv_spv,
            conv_bindings,
            @sizeOf(ConvPush),
        );
        errdefer pipeline_conv.deinit(ctx);

        const pipeline_scan = try Pipeline.init(
            ctx,
            &gdn_chunk_scan_spv,
            scan_bindings,
            @sizeOf(ScanPush),
        );
        errdefer pipeline_scan.deinit(ctx);

        const norm_info: vk.ShaderModuleCreateInfo = .{
            .code_size = gdn_chunk_norm_spv.len,
            .p_code = @ptrCast(&gdn_chunk_norm_spv),
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

    /// Record the scan pass and then the norm pass, with a barrier between
    /// them. The caller binds `set` with 11 bindings: q_n, k_n, v_conv, alpha,
    /// beta_in, ssm_a, dt_bias, ssm_norm, z, state (in and out), o_out. The
    /// caller also inserts the compute-to-compute barrier between `recordConv`
    /// and `recordScan`, and the barrier after it.
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
        const mb = [_]vk.MemoryBarrier{
            .{ .src_access_mask = .{ .shader_write_bit = true }, .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true } },
        };
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

    /// Run the whole chunked prefill, the conv pass then the scan pass, over
    /// `n_tokens`, and read state, conv_state, and o_out back. It takes the same
    /// inputs and writes the same outputs as `GdnDecode.runChunk`, so the
    /// self-test compares both against the CPU `step` run n_tokens times.
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
        var infos: [n_bufs]vk.DescriptorBufferInfo = undefined;
        for (&self.bufs, lens, uploads, 0..) |*buf, len, up, i| {
            const bytes: vk.DeviceSize = @intCast(len * @sizeOf(f32));
            try buf.ensure(bytes);
            if (up) |src| {
                @memcpy(buf.sliceTyped(f32, len), src);
            }

            infos[i] = .{ .buffer = buf.buffer, .offset = 0, .range = bytes };
        }

        // conv set: qkv(0), conv_w(1), conv_state(2), q_n(3), k_n(4), v_conv(5)
        updateStorageBindings(self.ctx, self.conv_set, infos[0..6]);
        // scan set: q_n(3), k_n(4), v_conv(5), alpha(6), beta(7), ssm_a(8),
        // dt_bias(9), ssm_norm(10), z(11), state(12), o_out(13)
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
        const mb = [_]vk.MemoryBarrier{
            .{ .src_access_mask = .{ .shader_write_bit = true }, .dst_access_mask = .{ .shader_read_bit = true } },
        };
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

/// Per-head QK norm for Qwen3.5 full attention, with an optional de-interleave
/// of the q and gate halves. Four storage bindings: in, norm_w, out, gate_out.
/// One workgroup per head, so the grid is n_heads, and `head_dim` must equal the
/// workgroup width of 256. It mirrors the CPU `fullMixer`. `recordInto` is the
/// only entry point, because the fused runner is the only caller.
pub const QwenQkNorm = struct {
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
        const pipeline = try Pipeline.init(ctx, &qwen_qk_norm_spv, 4, @sizeOf(Push));
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

/// Grow-on-demand host-visible and coherent storage buffer. It also carries
/// `transfer_src_bit`, so it serves as the staging buffer for an upload into a
/// device-local `DeviceBuffer`.
///
/// The memory type is host-cached whenever the driver offers one, because
/// an uncached host mapping slows every CPU write and read. A driver without a
/// cached type still works, and `ensure` logs the slow choice at debug level.
pub const HostBuffer = struct {
    ctx: *const vulkan_ctx.Context,
    buffer: vk.Buffer = .null_handle,
    memory: vk.DeviceMemory = .null_handle,
    capacity: vk.DeviceSize = 0,
    mapped: [*]u8 = undefined,

    pub fn deinit(self: *HostBuffer) void {
        if (self.capacity == 0) return;

        self.ctx.vkd.unmapMemory(self.ctx.device, self.memory);
        self.ctx.vkd.destroyBuffer(self.ctx.device, self.buffer, null);
        self.ctx.vkd.freeMemory(self.ctx.device, self.memory, null);
        self.capacity = 0;
    }

    pub fn ensure(self: *HostBuffer, size: vk.DeviceSize) !void {
        if (self.capacity >= size) return;

        self.deinit();

        // Host buffer flags: storage (shader-readable), transfer_src
        // (stage uploads device-ward), transfer_dst (stage readbacks
        // host-ward, e.g. logits copy or KV swap-out). Setting all
        // three is free. The driver only differentiates at use time.
        const buf_info: vk.BufferCreateInfo = .{
            .size = size,
            .usage = .{
                .storage_buffer_bit = true,
                .transfer_src_bit = true,
                .transfer_dst_bit = true,
            },
            .sharing_mode = .exclusive,
        };
        const buffer = try self.ctx.vkd.createBuffer(self.ctx.device, &buf_info, null);
        errdefer self.ctx.vkd.destroyBuffer(self.ctx.device, buffer, null);

        const reqs = self.ctx.vkd.getBufferMemoryRequirements(self.ctx.device, buffer);
        const mem_type = vulkan_ctx.pickHostType(self.ctx.mem_props, reqs.memory_type_bits) orelse
            return error.NoSuitableMemoryType;
        const chosen = self.ctx.mem_props.memory_types[mem_type];
        const chosen_heap = self.ctx.mem_props.memory_heaps[chosen.heap_index];
        log.debug(
            "host buffer: {d} bytes, memory type {d} flags 0x{x} heap {d} heap flags 0x{x}",
            .{
                size,
                mem_type,
                @as(u32, @bitCast(chosen.property_flags)),
                chosen.heap_index,
                @as(u32, @bitCast(chosen_heap.flags)),
            },
        );
        // A silent fallback hides the reason CPU access to this buffer is slow.
        if (!chosen.property_flags.host_cached_bit) {
            log.debug(
                "host buffer: no host-cached memory type, so CPU access to this host buffer " ++
                    "is uncached and slow",
                .{},
            );
        }

        const alloc_info: vk.MemoryAllocateInfo = .{
            .allocation_size = reqs.size,
            .memory_type_index = mem_type,
        };
        const memory = try self.ctx.vkd.allocateMemory(self.ctx.device, &alloc_info, null);
        errdefer self.ctx.vkd.freeMemory(self.ctx.device, memory, null);

        try self.ctx.vkd.bindBufferMemory(self.ctx.device, buffer, memory, 0);
        const mapped_opt = try self.ctx.vkd.mapMemory(self.ctx.device, memory, 0, size, .{});

        self.buffer = buffer;
        self.memory = memory;
        self.capacity = size;
        self.mapped = @ptrCast(mapped_opt.?);
    }

    pub fn sliceTyped(self: *HostBuffer, comptime T: type, count: usize) []T {
        std.debug.assert(count * @sizeOf(T) <= self.capacity);

        const raw: [*]T = @ptrCast(@alignCast(self.mapped));
        return raw[0..count];
    }
};

/// Device-local storage buffer and transfer destination. It is never mapped.
/// Write to it by copying from a `HostBuffer` staging buffer through
/// `cmdCopyBuffer`. A shader then reads device-local memory, about 500 GB/s on
/// an RDNA2 card. The host-visible path instead reads host-visible memory over
/// the virtualized host bus, about 30 GB/s.
pub const DeviceBuffer = struct {
    ctx: *const vulkan_ctx.Context,
    buffer: vk.Buffer = .null_handle,
    memory: vk.DeviceMemory = .null_handle,
    capacity: vk.DeviceSize = 0,

    pub fn deinit(self: *DeviceBuffer) void {
        if (self.capacity == 0) return;

        self.ctx.vkd.destroyBuffer(self.ctx.device, self.buffer, null);
        self.ctx.vkd.freeMemory(self.ctx.device, self.memory, null);
        self.capacity = 0;
    }

    pub fn ensure(self: *DeviceBuffer, size: vk.DeviceSize) !void {
        if (self.capacity >= size) return;

        self.deinit();

        const buf_info: vk.BufferCreateInfo = .{
            .size = size,
            .usage = .{
                .storage_buffer_bit = true,
                .transfer_dst_bit = true,
                .transfer_src_bit = true,
            },
            .sharing_mode = .exclusive,
        };
        const buffer = try self.ctx.vkd.createBuffer(self.ctx.device, &buf_info, null);
        errdefer self.ctx.vkd.destroyBuffer(self.ctx.device, buffer, null);

        const reqs = self.ctx.vkd.getBufferMemoryRequirements(self.ctx.device, buffer);
        const required: vk.MemoryPropertyFlags = .{ .device_local_bit = true };
        const mem_type = vulkan_ctx.findMemoryType(
            self.ctx.mem_props,
            reqs.memory_type_bits,
            required,
        ) orelse
            return error.NoSuitableMemoryType;

        const alloc_info: vk.MemoryAllocateInfo = .{
            .allocation_size = reqs.size,
            .memory_type_index = mem_type,
        };
        const memory = try self.ctx.vkd.allocateMemory(self.ctx.device, &alloc_info, null);
        errdefer self.ctx.vkd.freeMemory(self.ctx.device, memory, null);

        try self.ctx.vkd.bindBufferMemory(self.ctx.device, buffer, memory, 0);

        self.buffer = buffer;
        self.memory = memory;
        self.capacity = size;
    }
};

/// The device KV cache: one region per layer and tensor, in the order and the
/// packing of `kv_index`. Each segment is one device-local `DeviceBuffer` that
/// holds whole regions, so the device limits bind one region and not the whole
/// cache. The store places bytes only. The runner maps each `kv_index.fits`
/// verdict to an error before it calls `init`.
pub const KvStore = struct {
    layout: kv_index.Layout,
    packing: kv_index.Packing,
    /// The limits that the packing honors. Every descriptor from the store
    /// starts at a multiple of `offset_alignment` and binds at most
    /// `storage_range` bytes.
    lim: limits.BufferLimits,
    /// The first `packing.n_segments` entries hold the allocations in segment
    /// order. The other entries stay empty.
    segments: [max_segments]DeviceBuffer,

    /// One segment per region when no two regions fit one allocation.
    pub const max_segments: u32 = 2 * limits.max_layers;

    /// The address of one row for a copy: a buffer and a byte offset.
    pub const Row = struct { buffer: vk.Buffer, offset: u64 };

    pub const InitError = error{NoSuitableMemoryType} ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError;

    /// Allocate every segment of `layout` packed under the buffer limits of
    /// `ctx`.
    pub fn init(ctx: *const vulkan_ctx.Context, layout: kv_index.Layout) InitError!KvStore {
        const lim = ctx.buffer_limits;
        std.debug.assert(kv_index.fits(layout, lim) == .fits);
        const packing = kv_index.pack(layout, lim);
        std.debug.assert(packing.n_segments >= 1);
        std.debug.assert(packing.n_segments <= max_segments);

        var store: KvStore = .{
            .layout = layout,
            .packing = packing,
            .lim = lim,
            .segments = @splat(.{ .ctx = ctx }),
        };
        var made: u32 = 0;
        errdefer for (store.segments[0..made]) |*segment| segment.deinit();

        while (made < packing.n_segments) : (made += 1) {
            try store.segments[made].ensure(packing.segmentBytes(made));
        }

        std.debug.assert(store.deviceBytes() == layout.totalBytes());
        log.debug("kv store: {d} regions of {d} KiB in {d} allocations of up to {d} regions", .{
            packing.n_regions, packing.region_bytes >> 10, packing.n_segments, packing.per_segment,
        });
        return store;
    }

    /// Free every segment. A second call frees nothing.
    pub fn deinit(self: *KvStore) void {
        std.debug.assert(self.packing.n_segments <= max_segments);

        for (self.segments[0..self.packing.n_segments]) |*segment| {
            segment.deinit();
        }

        for (self.segments) |segment| {
            std.debug.assert(segment.capacity == 0);
        }
    }

    /// Bytes the segments hold together: the total of the layout.
    pub fn deviceBytes(self: *const KvStore) u64 {
        var sum: u64 = 0;
        for (self.segments[0..self.packing.n_segments], 0..) |segment, i| {
            std.debug.assert(segment.capacity == self.packing.segmentBytes(@intCast(i)));

            sum += segment.capacity;
        }

        std.debug.assert(sum == self.layout.totalBytes());
        return sum;
    }

    /// The binding of region `(layer, t)`: every slot of that tensor.
    pub fn region(self: *const KvStore, layer: u32, t: kv_index.Tensor) vk.DescriptorBufferInfo {
        const base = self.regionAt(layer, t);
        const range = self.packing.region_bytes;
        std.debug.assert(range == self.layout.regionBytes());
        std.debug.assert(range <= self.lim.storage_range);
        return .{ .buffer = base.buffer, .offset = base.offset, .range = range };
    }

    /// The binding of the `window` rows of `slot` in region `(layer, t)`.
    pub fn slotView(
        self: *const KvStore,
        layer: u32,
        t: kv_index.Tensor,
        slot: u32,
    ) vk.DescriptorBufferInfo {
        const base = self.regionAt(layer, t);
        const offset = base.offset + self.layout.slotByteOffset(slot);
        const range = self.layout.slotRegionBytes();
        std.debug.assert(offset % self.lim.offset_alignment == 0);
        std.debug.assert(offset + range <= base.offset + self.packing.region_bytes);
        std.debug.assert(range <= self.lim.storage_range);
        return .{ .buffer = base.buffer, .offset = offset, .range = range };
    }

    /// The address of the row at `(slot, pos)` in region `(layer, t)`.
    pub fn rowAt(self: *const KvStore, layer: u32, t: kv_index.Tensor, slot: u32, pos: u32) Row {
        const base = self.regionAt(layer, t);
        const offset = base.offset + self.layout.rowByteOffset(slot, pos);
        std.debug.assert(offset >= base.offset + self.layout.slotByteOffset(slot));
        std.debug.assert(
            offset + self.layout.rowBytes() <= base.offset + self.packing.region_bytes,
        );
        return .{ .buffer = base.buffer, .offset = offset };
    }

    /// The copy list that forks slot `source` into slot `target`: the `window`
    /// rows of the slot in every region, one copy per region. The slot rows of
    /// one region never overlap, so each copy stays inside one buffer. A copy
    /// needs no offset alignment, so the list addresses rows and not
    /// descriptor views.
    pub fn cloneCopies(self: *const KvStore, source: u32, target: u32, out: []Copy) []Copy {
        const layout = self.layout;
        std.debug.assert(source != target);
        std.debug.assert(out.len >= layout.regionCount());
        const slot_rows_bytes = layout.slotRegionBytes();
        var c: usize = 0;
        var total: u64 = 0;
        var layer: u32 = 0;
        while (layer < layout.n_layers) : (layer += 1) {
            for ([_]kv_index.Tensor{ .k, .v }) |t| {
                const from = self.rowAt(layer, t, source, 0);
                const to = self.rowAt(layer, t, target, 0);
                std.debug.assert(from.buffer == to.buffer);
                out[c] = .{
                    .src = from.buffer,
                    .dst = to.buffer,
                    .src_offset = from.offset,
                    .dst_offset = to.offset,
                    .size = slot_rows_bytes,
                };
                c += 1;
                total += slot_rows_bytes;
            }
        }

        std.debug.assert(c == layout.regionCount());
        std.debug.assert(total == layout.slotBytes());
        return out[0..c];
    }

    /// The copy list between the first `n_rows` rows of `slot` in every region
    /// and a packed host stage. The list keeps the blob order: layers
    /// ascending, K rows then V rows per layer. `to_host` true reads the device, and false writes
    /// it. A save and its restore walk one order, so their packed offsets
    /// agree.
    pub fn rowCopies(
        self: *const KvStore,
        slot: u32,
        n_rows: u32,
        stage: vk.Buffer,
        to_host: bool,
        out: []Copy,
    ) []Copy {
        const layout = self.layout;
        std.debug.assert(n_rows >= 1);
        std.debug.assert(n_rows <= layout.window);
        std.debug.assert(out.len >= layout.regionCount());
        const rows_bytes: u64 = @as(u64, n_rows) * layout.rowBytes();
        var off: u64 = 0;
        var c: usize = 0;
        var layer: u32 = 0;
        while (layer < layout.n_layers) : (layer += 1) {
            for ([_]kv_index.Tensor{ .k, .v }) |t| {
                const dev = self.rowAt(layer, t, slot, 0);
                out[c] = if (to_host)
                    .{
                        .src = dev.buffer,
                        .src_offset = dev.offset,
                        .dst = stage,
                        .dst_offset = off,
                        .size = rows_bytes,
                    }
                else
                    .{
                        .src = stage,
                        .src_offset = off,
                        .dst = dev.buffer,
                        .dst_offset = dev.offset,
                        .size = rows_bytes,
                    };
                c += 1;
                off += rows_bytes;
            }
        }

        std.debug.assert(c == layout.regionCount());
        std.debug.assert(off == layout.blobBytes(n_rows));
        return out[0..c];
    }

    /// The segment buffer of region `(layer, t)` and the byte where it starts.
    fn regionAt(self: *const KvStore, layer: u32, t: kv_index.Tensor) Row {
        const index = self.layout.regionIndex(layer, t);
        const segment = self.segments[self.packing.segmentOf(index)];
        const offset = self.packing.offsetOf(index);
        std.debug.assert(segment.capacity >= 1);
        std.debug.assert(offset + self.packing.region_bytes <= segment.capacity);
        std.debug.assert(offset % self.lim.offset_alignment == 0);
        return .{ .buffer = segment.buffer, .offset = offset };
    }
};

/// Device-local VRAM arena. It sub-allocates from a small number of large
/// buffer and memory segments. The buffer usage is a parameter, so one arena
/// type backs every device-local consumer. The consumers are the Q8_0 scales,
/// the Q8_0 quants, and the KV cache shards of a tiered cache.
///
/// The driver, not the card, sets the constraint. A Llama-8B Q8_0 model holds
/// 32 layers times 7 matmuls plus one lm_head, which is 225 Q8_0 weights. One
/// buffer and one memory object each makes 450 device allocations. That count
/// plus a 512 MB KV cache exhausts the device-local heap of the target driver.
/// It does so long before the 16 GB of the card is touched. Ten large segments
/// of about 1 GB each keep the allocation count low.
///
/// Each `alloc(size)` returns a `View` shaped like `vk.DescriptorBufferInfo`,
/// which the caller writes straight into a descriptor update. Offsets align to
/// the `minStorageBufferOffsetAlignment` of the device, so the binding stays
/// valid. Allocations are append-only and there is no `free`. An arena lives as
/// long as its use. Model weights live for the life of the runner, and a
/// per-session KV pool is reset whole between sessions.
pub const VramArena = struct {
    ctx: *const vulkan_ctx.Context,
    gpa: std.mem.Allocator,
    segments: std.ArrayList(Segment),
    seg_capacity: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
    align_bytes: vk.DeviceSize,

    pub const Segment = struct {
        buffer: vk.Buffer,
        memory: vk.DeviceMemory,
        capacity: vk.DeviceSize,
        used: vk.DeviceSize,
    };

    /// A descriptor-binding-shaped view into the arena. Identity match
    /// with `vk.DescriptorBufferInfo` so callers can write it directly
    /// into descriptor updates.
    pub const View = vk.DescriptorBufferInfo;

    pub const InitError = std.mem.Allocator.Error;

    pub const AllocError = error{
        NoSuitableMemoryType,
        OutOfMemory,
    } ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError;

    /// `seg_capacity` is the default size for each new segment. Pick it
    /// to be at least as large as the largest single allocation you
    /// expect. Otherwise `alloc` grows individual segments to fit,
    /// which works but fragments the arena. For Llama-8B Q8_0 quants
    /// the largest single weight is the lm_head at ~525 MB. 1 GB is a
    /// safe default.
    pub fn init(
        ctx: *const vulkan_ctx.Context,
        gpa: std.mem.Allocator,
        usage: vk.BufferUsageFlags,
        seg_capacity: vk.DeviceSize,
    ) InitError!VramArena {
        return .{
            .ctx = ctx,
            .gpa = gpa,
            .segments = .empty,
            .seg_capacity = seg_capacity,
            .usage = usage,
            .align_bytes = ctx.buffer_limits.offset_alignment,
        };
    }

    pub fn deinit(self: *VramArena) void {
        for (self.segments.items) |*s| {
            self.ctx.vkd.destroyBuffer(self.ctx.device, s.buffer, null);
            self.ctx.vkd.freeMemory(self.ctx.device, s.memory, null);
        }

        self.segments.deinit(self.gpa);
    }

    /// Allocate `size` bytes from the arena. Tries to fit in the most
    /// recent segment first (typical for sequential weight uploads),
    /// then any other segment, then creates a new one. The new segment
    /// is at least `seg_capacity` bytes. It grows past that only if the
    /// request itself is larger.
    pub fn alloc(self: *VramArena, size: vk.DeviceSize) AllocError!View {
        // The alloc sequence at debug scope is the VRAM map. Allocations follow
        // the deterministic weight-upload order, so a reader pairs each line
        // with the load sequence.
        log.debug(
            "arena alloc: {d} KiB ({d} MiB held before)",
            .{ size / 1024, self.totalUsed() / (1024 * 1024) },
        );
        // Try the most-recent segment first. It is likeliest to have room.
        if (self.segments.items.len > 0) {
            const last = &self.segments.items[self.segments.items.len - 1];
            if (tryFit(last, size, self.align_bytes)) |view| return view;
        }

        // Try older segments in case the latest was just exhausted.
        for (self.segments.items[0..if (self.segments.items.len ==
            0) 0 else self.segments.items.len - 1]) |*s|
        {
            if (tryFit(s, size, self.align_bytes)) |view| return view;
        }

        // Need a new segment. Size it for at least this request.
        const new_cap = @max(self.seg_capacity, size);
        try self.segments.append(self.gpa, try self.createSegment(new_cap));
        const seg = &self.segments.items[self.segments.items.len - 1];
        seg.used = size;
        return .{ .buffer = seg.buffer, .offset = 0, .range = size };
    }

    /// One debug line per segment: the coarse VRAM map. Call after init,
    /// when the arena holds its final layout.
    pub fn logSegments(self: *const VramArena, name: []const u8) void {
        const mib = 1024 * 1024;
        for (self.segments.items, 0..) |s, i| {
            log.debug(
                "arena {s} seg {d}: {d} of {d} MiB used",
                .{ name, i, s.used / mib, s.capacity / mib },
            );
        }
    }

    /// Total bytes claimed by allocations so far, the sum of `used` across the
    /// segments. The traces and the VRAM budget of the tiered KV cache read it.
    pub fn totalUsed(self: *const VramArena) vk.DeviceSize {
        var sum: vk.DeviceSize = 0;
        for (self.segments.items) |s| {
            sum += s.used;
        }

        return sum;
    }

    fn tryFit(seg: *Segment, size: vk.DeviceSize, align_bytes: vk.DeviceSize) ?View {
        const aligned = alignUp(seg.used, align_bytes);
        if (aligned + size > seg.capacity) return null;

        seg.used = aligned + size;
        return .{ .buffer = seg.buffer, .offset = aligned, .range = size };
    }

    fn createSegment(self: *VramArena, capacity: vk.DeviceSize) AllocError!Segment {
        const buf_info: vk.BufferCreateInfo = .{
            .size = capacity,
            .usage = self.usage,
            .sharing_mode = .exclusive,
        };
        const buffer = try self.ctx.vkd.createBuffer(self.ctx.device, &buf_info, null);
        errdefer self.ctx.vkd.destroyBuffer(self.ctx.device, buffer, null);

        const reqs = self.ctx.vkd.getBufferMemoryRequirements(self.ctx.device, buffer);
        const required: vk.MemoryPropertyFlags = .{ .device_local_bit = true };
        const mem_type = vulkan_ctx.findMemoryType(
            self.ctx.mem_props,
            reqs.memory_type_bits,
            required,
        ) orelse
            return error.NoSuitableMemoryType;

        const alloc_info: vk.MemoryAllocateInfo = .{
            .allocation_size = reqs.size,
            .memory_type_index = mem_type,
        };
        const memory = try self.ctx.vkd.allocateMemory(self.ctx.device, &alloc_info, null);
        errdefer self.ctx.vkd.freeMemory(self.ctx.device, memory, null);

        try self.ctx.vkd.bindBufferMemory(self.ctx.device, buffer, memory, 0);
        log.debug("vram segment {d}: {d} MiB", .{ self.segments.items.len, capacity >> 20 });
        return .{
            .buffer = buffer,
            .memory = memory,
            .capacity = capacity,
            .used = 0,
        };
    }
};

/// The standalone-submit resources a kernel owns on top of its `Pipeline`:
/// a single-set descriptor pool, the set, and a reusable primary command
/// buffer. Only kernels with a self-contained `run()` carry one. The
/// record-only kernels leave sets and command buffers to their caller.
pub const OneShot = struct {
    desc_pool: vk.DescriptorPool,
    desc_set: vk.DescriptorSet,
    cmd_buffer: vk.CommandBuffer,

    pub const InitError =
        vk.DeviceWrapper.CreateDescriptorPoolError ||
        vk.DeviceWrapper.AllocateDescriptorSetsError ||
        vk.DeviceWrapper.AllocateCommandBuffersError;

    pub fn init(
        ctx: *const vulkan_ctx.Context,
        set_layout: vk.DescriptorSetLayout,
        binding_count: u32,
    ) InitError!OneShot {
        const desc_pool = try createSingleSetPool(ctx, binding_count);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, desc_pool, null);

        const desc_set = try allocateOneSet(ctx, desc_pool, set_layout);
        const cmd_buffer = try allocateOnePrimaryCmdBuffer(ctx);

        return .{
            .desc_pool = desc_pool,
            .desc_set = desc_set,
            .cmd_buffer = cmd_buffer,
        };
    }

    pub fn deinit(self: OneShot, ctx: *const vulkan_ctx.Context) void {
        // The command buffer came from ctx.command_pool. Free it explicitly
        // so the pool stays clean for other kernels that share it.
        // Destroying the descriptor pool frees the set automatically.
        const cmds_to_free = [_]vk.CommandBuffer{self.cmd_buffer};
        ctx.vkd.freeCommandBuffers(ctx.device, ctx.command_pool, &cmds_to_free);
        ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);
    }
};

/// One staged copy: a source buffer, a destination buffer, a size, and the two
/// offsets.
pub const Copy = struct {
    src: vk.Buffer,
    dst: vk.Buffer,
    size: vk.DeviceSize,
    src_offset: vk.DeviceSize = 0,
    dst_offset: vk.DeviceSize = 0,
};

/// The phases of one one-shot submit, in order. The failure line names one.
pub const SubmitPhase = enum { reset, record, submit, wait };

/// The kind of work of one one-shot submit. The debug line prints it, so a
/// device log can tell the sites apart.
pub const SubmitLabel = enum { copy, fill, clone, dispatch };

/// What a recorder put into the command buffer, for the debug line. A
/// dispatch or a fill records no copy.
pub const Recorded = struct {
    copies: u32 = 0,
    bytes: u64 = 0,
};

pub const SubmitError = vk.DeviceWrapper.ResetCommandBufferError ||
    vk.DeviceWrapper.BeginCommandBufferError ||
    vk.DeviceWrapper.EndCommandBufferError ||
    vk.DeviceWrapper.QueueSubmitError ||
    vk.DeviceWrapper.QueueWaitIdleError;

/// Records one `cmdCopyBuffer` per copy, in list order, with no barrier.
pub const CopyRecorder = struct {
    ctx: *const vulkan_ctx.Context,
    copies: []const Copy,

    pub fn record(self: CopyRecorder, cb: vk.CommandBuffer) Recorded {
        var bytes: u64 = 0;
        for (self.copies) |c| {
            const regions = [_]vk.BufferCopy{
                .{ .src_offset = c.src_offset, .dst_offset = c.dst_offset, .size = c.size },
            };
            self.ctx.vkd.cmdCopyBuffer(cb, c.src, c.dst, &regions);
            bytes += c.size;
        }

        return .{ .copies = @intCast(self.copies.len), .bytes = bytes };
    }
};

/// The two costs one staged KV swap pays. `dma_ms` covers the
/// `runStagingCopies` submit and wait, and `memcpy_ms` covers the copy through
/// the mapped `HostBuffer`. A bench must report the two apart, because they sit
/// under different roofs and only the split names the side that costs.
pub const SwapSplit = struct {
    dma_ms: f64,
    memcpy_ms: f64,
};

/// The host side of one staged swap.
pub const StageSide = union(enum) {
    /// A save: the device bytes land in this blob.
    to_host: []u8,
    /// A restore: this blob lands on the device.
    to_device: []const u8,
};

// -- state ------------------------------------------------------------------

/// One-shot submits this process started. It numbers the debug and failure
/// lines, so a device log can tie a failure to a submit count.
var staging_submits: std.atomic.Value(u64) = .init(0);

/// True while a `submitOnce` call holds the staging buffer. It is process-wide,
/// because every caller holds its context through a `*const` pointer.
var staging_busy: std.atomic.Value(bool) = .init(false);

/// The error of a failed wait. The buffer can still be pending after it, and a
/// reset of a pending buffer is invalid, so every later call returns it. The
/// busy flag guards it.
var staging_wait_error: ?SubmitError = null;

// -- api --------------------------------------------------------------------

/// The number of one-shot submits this process started.
pub fn stagingSubmitCount() u64 {
    return staging_submits.load(.monotonic);
}

/// Submit one command buffer of work and block until the queue is idle. It
/// resets the staging buffer of `ctx`, begins it for one submit, and calls
/// `recorder.record(cb)`, which returns a `Recorded`. Then it ends the buffer,
/// submits it, and waits. Once it returns, the GPU has finished the work.
///
/// The translation layer makes a native command allocator for each command
/// buffer, and the paravirtual driver leaks part of it when the buffer is
/// freed. So every one-shot submit reuses the one buffer of the context. The
/// wait before the return keeps the buffer idle at the next entry. After a
/// failed wait, every later call returns that error. The Vulkan host thread is
/// single, and the busy flag asserts that no call nests in another.
pub fn submitOnce(
    ctx: *const vulkan_ctx.Context,
    label: SubmitLabel,
    recorder: anytype,
) SubmitError!void {
    const cmd = ctx.staging_cmd;
    std.debug.assert(cmd != .null_handle);
    const was_busy = staging_busy.swap(true, .acquire);
    std.debug.assert(!was_busy);
    defer staging_busy.store(false, .release);

    if (staging_wait_error) |err| {
        log.err("staging: submit refused, an earlier wait failed: {t}", .{err});
        return err;
    }

    const number = staging_submits.fetchAdd(1, .monotonic) + 1;
    var phase: SubmitPhase = .reset;
    errdefer |err| log.err("staging: submit {d} failed in phase {t}: {t}", .{ number, phase, err });

    var lap: [@typeInfo(SubmitPhase).@"enum".fields.len]f64 = @splat(0);

    var t = core_log.now();
    const t0 = t;
    try ctx.vkd.resetCommandBuffer(cmd, .{});
    lap[@intFromEnum(SubmitPhase.reset)] = core_log.msSince(t);

    phase = .record;
    t = core_log.now();
    const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
    try ctx.vkd.beginCommandBuffer(cmd, &begin);
    // The pool exists only at a timestamp level, so the off path records no
    // timestamp command. Both slots reset before their first write.
    const stamps = ctx.staging_query_pool;
    if (stamps != .null_handle) {
        ctx.vkd.cmdResetQueryPool(cmd, stamps, 0, 2);
        ctx.vkd.cmdWriteTimestamp(cmd, .{ .top_of_pipe_bit = true }, stamps, 0);
    }

    const recorded: Recorded = recorder.record(cmd);
    if (stamps != .null_handle) {
        ctx.vkd.cmdWriteTimestamp(cmd, .{ .bottom_of_pipe_bit = true }, stamps, 1);
    }

    try ctx.vkd.endCommandBuffer(cmd);
    lap[@intFromEnum(SubmitPhase.record)] = core_log.msSince(t);

    phase = .submit;
    t = core_log.now();
    const t_submit = t;
    const cmds = [_]vk.CommandBuffer{cmd};
    const submits = [_]vk.SubmitInfo{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }};
    try ctx.vkd.queueSubmit(ctx.queue, &submits, .null_handle);
    lap[@intFromEnum(SubmitPhase.submit)] = core_log.msSince(t);

    phase = .wait;
    t = core_log.now();
    const t_wait = t;
    const waited = ctx.vkd.queueWaitIdle(ctx.queue);
    if (waited) |_| {} else |err| {
        staging_wait_error = err;
    }

    try waited;
    const t_done = core_log.now();
    lap[@intFromEnum(SubmitPhase.wait)] = msBetween(t, t_done);
    span.record(.gpu_submit, t_submit, t_wait);
    span.record(.gpu_wait, t_wait, t_done);
    if (stamps != .null_handle) {
        stagingRecord(ctx, label, t_submit, t_done);
    }

    // The words allocate and free keep the field layout that log parsers
    // read. The reset time prints as allocate, and nothing is freed.
    const line_fmt = "staging: submit {d} {t}: {d} copies, {d} KiB, {d:.1} ms " ++
        "(allocate {d:.3}, record {d:.3}, submit {d:.3}, wait {d:.3}, free 0.000 ms)";
    log.debug(
        line_fmt,
        .{ number, label, recorded.copies, recorded.bytes >> 10, core_log.msSince(t0), lap[0], lap[1], lap[2], lap[3] },
    );
}

/// Submit one transfer that runs every copy in `copies`, for example from
/// host-visible staging buffers to device-local destinations, through
/// `submitOnce`. Once it returns, a compute dispatch can read the destination
/// buffers. An empty list is a legal submit that records nothing.
pub fn runStagingCopies(ctx: *const vulkan_ctx.Context, copies: []const Copy) SubmitError!void {
    try submitOnce(ctx, .copy, CopyRecorder{ .ctx = ctx, .copies = copies });
}

/// Move one packed blob between the device and the host through `stage`, one
/// chunk at a time. `full` lists the copies of the whole blob in blob order,
/// with `stage` as the host buffer, as `KvStore.rowCopies` builds them. Evict,
/// restore, and the prefix snapshot all walk this one loop, so the stage holds
/// one chunk of at most `chunk_bytes` and never one window.
///
/// The walk takes `kv_budget.ChunkWalk.count(blob, chunk)` trips. Each trip
/// clips `full` to its byte range, submits the clipped copies, and waits. A
/// save copies the stage out after the wait, and a restore fills the stage
/// before the submit.
pub fn walkStage(
    ctx: *const vulkan_ctx.Context,
    full: []const Copy,
    stage: *const HostBuffer,
    side: StageSide,
    chunk_bytes: u64,
) !SwapSplit {
    const to_host = side == .to_host;
    const total: u64 = switch (side) {
        inline else => |blob| blob.len,
    };

    const chunk = @min(chunk_bytes, total);
    std.debug.assert(total >= 1);
    std.debug.assert(chunk >= 1);
    std.debug.assert(full.len <= KvStore.max_segments);
    // The caller grew the stage to one chunk before it built `full`, so every
    // copy names the live stage buffer.
    std.debug.assert(stage.capacity >= chunk);
    for (full) |c| {
        std.debug.assert((if (to_host) c.dst else c.src) == stage.buffer);
    }

    var split: SwapSplit = .{ .dma_ms = 0, .memcpy_ms = 0 };
    var scratch: [KvStore.max_segments]Copy = undefined;
    var walk = kv_budget.ChunkWalk.init(total, chunk);
    var trips: u64 = 0;
    while (walk.next()) |piece| : (trips += 1) {
        const copies = kv_budget.clipCopies(Copy, full, to_host, piece.offset, piece.len, &scratch);
        const len: usize = @intCast(piece.len);
        const off: usize = @intCast(piece.offset);
        switch (side) {
            .to_host => |out| {
                const t_dma = core_log.now();
                try runStagingCopies(ctx, copies);
                split.dma_ms += core_log.msSince(t_dma);
                const t_cpy = core_log.now();
                @memcpy(out[off..][0..len], stage.mapped[0..len]);
                split.memcpy_ms += core_log.msSince(t_cpy);
            },
            .to_device => |blob| {
                const t_cpy = core_log.now();
                @memcpy(stage.mapped[0..len], blob[off..][0..len]);
                split.memcpy_ms += core_log.msSince(t_cpy);
                const t_dma = core_log.now();
                try runStagingCopies(ctx, copies);
                split.dma_ms += core_log.msSince(t_dma);
            },
        }
    }

    std.debug.assert(trips == kv_budget.ChunkWalk.count(total, chunk));
    return split;
}

/// Create a descriptor-set layout of `binding_count` storage-buffer bindings,
/// numbered 0 to `binding_count - 1`. Each binding holds one storage buffer
/// visible in the compute stage. `binding_count` must be at most 16.
pub fn createStorageBufferLayout(
    ctx: *const vulkan_ctx.Context,
    binding_count: u32,
) !vk.DescriptorSetLayout {
    var bindings: [16]vk.DescriptorSetLayoutBinding = undefined;
    std.debug.assert(binding_count <= bindings.len);
    var i: u32 = 0;
    while (i < binding_count) : (i += 1) {
        bindings[i] = .{
            .binding = i,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .compute_bit = true },
        };
    }

    const info: vk.DescriptorSetLayoutCreateInfo = .{
        .binding_count = binding_count,
        .p_bindings = &bindings,
    };
    return ctx.vkd.createDescriptorSetLayout(ctx.device, &info, null);
}

/// Allocate one buffer that lives as long as its owner. A one-shot submit uses
/// `submitOnce` and allocates nothing.
pub fn allocateOnePrimaryCmdBuffer(
    ctx: *const vulkan_ctx.Context,
) !vk.CommandBuffer {
    const info: vk.CommandBufferAllocateInfo = .{
        .command_pool = ctx.command_pool,
        .level = .primary,
        .command_buffer_count = 1,
    };
    var cmd: vk.CommandBuffer = .null_handle;
    try ctx.vkd.allocateCommandBuffers(ctx.device, &info, @ptrCast(&cmd));
    return cmd;
}

/// Bind N storage buffers at bindings 0 through N-1. `buf_infos` must outlive
/// the call, and it must hold at most 16 entries, the descriptor-set-layout cap.
pub fn updateStorageBindings(
    ctx: *const vulkan_ctx.Context,
    set: vk.DescriptorSet,
    buf_infos: []const vk.DescriptorBufferInfo,
) void {
    var writes: [16]vk.WriteDescriptorSet = undefined;
    std.debug.assert(buf_infos.len <= writes.len);
    for (buf_infos, 0..) |*info, i| {
        writes[i] = .{
            .dst_set = set,
            .dst_binding = @intCast(i),
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = @ptrCast(info),
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        };
    }

    ctx.vkd.updateDescriptorSets(ctx.device, writes[0..buf_infos.len], null);
}

/// The one-shot driver for the record-only `KvCommitF16`. It stages the two f32
/// scratch inputs in host-visible memory, builds a throwaway descriptor pool and
/// set, dispatches through `recordInto` in one `submitOnce`, and reads the two
/// narrowed outputs back as raw f16 bits. For the rounding gate alone: the
/// runners record this kernel into their own per-token command buffer.
///
/// The output is bits, not values. The bits are what the cache stores, so a bit
/// comparison states the rounding property directly and keeps a second
/// conversion out of the path under test.
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

// -- helpers ----------------------------------------------------------------

/// Repack a Q8_0 weight matrix from the interleaved GGUF layout into two flat
/// arrays the shader indexes directly. The GGUF layout holds each block of 32
/// elements as a 2-byte f16 scale plus 32 i8 quants, contiguous per row. The
/// flat arrays hold the f16 scales by `(j, blk)` and the i8 quants by `(j, p)`.
/// The copy keeps every bit pattern, so the GPU `float16_t` and `int8_t` loads
/// see the bytes the CPU kernel reads through `f16BitsToF32` and
/// `@bitCast(i8)`. The bit-exact match against the CPU kernel depends on that.
/// Do not reorder bytes here.
fn repackQ80ToScalesQuants(
    weight_data: []const u8,
    in_features: usize,
    out_features: usize,
    scales_out: []u8,
    quants_out: []u8,
) void {
    const blocks_per_row = in_features / q8_0_block_elems;
    const row_bytes = blocks_per_row * q8_0_block_bytes;
    std.debug.assert(scales_out.len >= out_features * blocks_per_row * 2);
    std.debug.assert(quants_out.len >= out_features * in_features);
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        const row_off = j * row_bytes;
        var blk: usize = 0;
        while (blk < blocks_per_row) : (blk += 1) {
            const blk_off = row_off + blk * q8_0_block_bytes;
            // 2 bytes of f16 scale -> scales_out at j * blocks_per_row + blk.
            const scales_idx = (j * blocks_per_row + blk) * 2;
            scales_out[scales_idx + 0] = weight_data[blk_off + 0];
            scales_out[scales_idx + 1] = weight_data[blk_off + 1];
            // 32 bytes of i8 quants -> quants_out at j * in_features + blk * 32.
            const quants_idx = j * in_features + blk * q8_0_block_elems;
            @memcpy(
                quants_out[quants_idx..][0..q8_0_block_elems],
                weight_data[blk_off + 2 ..][0..q8_0_block_elems],
            );
        }
    }
}

/// The Q4_0 sibling of `repackQ80ToScalesQuants`. A Q4_0 row holds `in/32`
/// blocks of 18 bytes, a 2-byte f16 scale plus 16 packed bytes. It separates the
/// scales, whose f16 layout matches Q8_0, from the 16 packed quant bytes. The
/// packed bytes keep every bit pattern. The shader extracts the nibbles in the
/// layout the CPU kernel uses.
fn repackQ40ToScalesQuants(
    weight_data: []const u8,
    in_features: usize,
    out_features: usize,
    scales_out: []u8,
    quants_out: []u8,
) void {
    const blocks_per_row = in_features / q4_0_block_elems;
    const row_bytes = blocks_per_row * q4_0_block_bytes;
    std.debug.assert(scales_out.len >= out_features * blocks_per_row * 2);
    std.debug.assert(quants_out.len >= out_features * (in_features / 2));
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        const row_off = j * row_bytes;
        var blk: usize = 0;
        while (blk < blocks_per_row) : (blk += 1) {
            const blk_off = row_off + blk * q4_0_block_bytes;
            const scales_idx = (j * blocks_per_row + blk) * 2;
            scales_out[scales_idx + 0] = weight_data[blk_off + 0];
            scales_out[scales_idx + 1] = weight_data[blk_off + 1];
            // 16 packed bytes -> quants_out at j * (in/2) + blk * 16.
            const quants_idx = j * (in_features / 2) + blk * q4_0_packed_bytes;
            @memcpy(
                quants_out[quants_idx..][0..q4_0_packed_bytes],
                weight_data[blk_off + 2 ..][0..q4_0_packed_bytes],
            );
        }
    }
}

fn alignUp(value: vk.DeviceSize, alignment: vk.DeviceSize) vk.DeviceSize {
    if (alignment == 0) return value;
    return ((value + alignment - 1) / alignment) * alignment;
}

fn msBetween(t0: std.Io.Timestamp, t1: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
}

/// Read the two stamps of the one-shot buffer after its wait and hand them to
/// the span recorder. A failed read loses one trace record, never the submit.
fn stagingRecord(
    ctx: *const vulkan_ctx.Context,
    label: SubmitLabel,
    submit: std.Io.Timestamp,
    wait: std.Io.Timestamp,
) void {
    var ts: [2]u64 = undefined;
    _ = ctx.vkd.getQueryPoolResults(
        ctx.device,
        ctx.staging_query_pool,
        0,
        2,
        @sizeOf(@TypeOf(ts)),
        &ts,
        @sizeOf(u64),
        .{ .@"64_bit" = true, .wait_bit = true },
    ) catch |err| {
        log.warn("staging: timestamp read failed: {t}", .{err});
        return;
    };
    const gpu_label: span.GpuLabel = switch (label) {
        inline else => |l| @field(span.GpuLabel, @tagName(l)),
    };

    const t = span.tickSpan(&ts, gpu_profile.tickMask(ctx.timestamp_valid_bits), null);
    span.gpu(.{
        .runner = .staging,
        .label = gpu_label,
        .gpu_start = t.start,
        .gpu_end = t.end,
        .period_ns = @floatCast(ctx.physical_device_props.limits.timestamp_period),
        .host_submit_ns = span.tsNs(submit),
        .host_wait_ns = span.tsNs(wait),
        .backsteps = t.backsteps,
    });
}

fn createPipelineLayout(
    ctx: *const vulkan_ctx.Context,
    set_layout: vk.DescriptorSetLayout,
    push_range: *const vk.PushConstantRange,
) !vk.PipelineLayout {
    const layouts = [_]vk.DescriptorSetLayout{set_layout};
    const info: vk.PipelineLayoutCreateInfo = .{
        .set_layout_count = 1,
        .p_set_layouts = &layouts,
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast(push_range),
    };
    return ctx.vkd.createPipelineLayout(ctx.device, &info, null);
}

fn createComputePipeline(
    ctx: *const vulkan_ctx.Context,
    shader_module: vk.ShaderModule,
    pipeline_layout: vk.PipelineLayout,
) !vk.Pipeline {
    // RDNA2 is natively wave32. A wave64 vector ALU or memory op executes by
    // issuing the instruction twice. So pin every compute pipeline to wave32
    // through VK_EXT_subgroup_size_control. The SmolLM2 decode benchmark over
    // 10 runs showed 31% lower inter-token latency and a 4x lower standard
    // deviation against the unpinned default. Precondition: the workgroup size
    // of every shader is a multiple of 32.
    const required_size: vk.PipelineShaderStageRequiredSubgroupSizeCreateInfo = .{
        .required_subgroup_size = 32,
    };
    const stage: vk.PipelineShaderStageCreateInfo = .{
        .p_next = @ptrCast(&required_size),
        .stage = .{ .compute_bit = true },
        .module = shader_module,
        .p_name = "main",
    };
    const infos = [_]vk.ComputePipelineCreateInfo{
        .{
            .stage = stage,
            .layout = pipeline_layout,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        },
    };
    var pipelines = [_]vk.Pipeline{.null_handle};
    _ = try ctx.vkd.createComputePipelines(ctx.device, .null_handle, &infos, null, &pipelines);
    return pipelines[0];
}

/// Pool sized for exactly one set with `descriptor_count` storage buffers.
fn createSingleSetPool(
    ctx: *const vulkan_ctx.Context,
    descriptor_count: u32,
) !vk.DescriptorPool {
    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{ .type = .storage_buffer, .descriptor_count = descriptor_count },
    };
    const info: vk.DescriptorPoolCreateInfo = .{
        .max_sets = 1,
        .pool_size_count = 1,
        .p_pool_sizes = &pool_sizes,
    };
    return ctx.vkd.createDescriptorPool(ctx.device, &info, null);
}

fn allocateOneSet(
    ctx: *const vulkan_ctx.Context,
    pool: vk.DescriptorPool,
    set_layout: vk.DescriptorSetLayout,
) !vk.DescriptorSet {
    const set_layouts = [_]vk.DescriptorSetLayout{set_layout};
    const info: vk.DescriptorSetAllocateInfo = .{
        .descriptor_pool = pool,
        .descriptor_set_count = 1,
        .p_set_layouts = &set_layouts,
    };
    var set: vk.DescriptorSet = .null_handle;
    try ctx.vkd.allocateDescriptorSets(ctx.device, &info, @ptrCast(&set));
    return set;
}

/// The shared one-shot driver for the record-only batch matmul kernels,
/// `LinearQ80Batch` and `LinearQ80Gemm`. It stages the input and output in
/// host-visible memory, builds a throwaway descriptor pool and set, dispatches
/// through `kernel.recordInto` in one `submitOnce`, and reads the result back.
/// For tests and the bench only. `kernel` points at a kernel that exposes
/// `.ctx`, `.pipeline`, and `recordInto(cb, set, b, k, n)`.
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

    const ctx = kernel.ctx;
    var w = try Q80Weight.init(ctx, weight_bytes, in_features, out_features);
    defer w.deinit();

    const F = @sizeOf(f32);
    var buf_in: HostBuffer = .{ .ctx = ctx };
    defer buf_in.deinit();

    var buf_out: HostBuffer = .{ .ctx = ctx };
    defer buf_out.deinit();

    try buf_in.ensure(@intCast(input.len * F));
    try buf_out.ensure(@intCast(out.len * F));
    @memcpy(buf_in.sliceTyped(f32, input.len), input);

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

// -- tests ------------------------------------------------------------------

/// A store over fake buffer handles. The accessors read only the handles and
/// the capacities, so the tests need no device and no context.
fn fakeKvStore(layout: kv_index.Layout, lim: limits.BufferLimits) KvStore {
    const packing = kv_index.pack(layout, lim);
    var store: KvStore = .{
        .layout = layout,
        .packing = packing,
        .lim = lim,
        .segments = @splat(.{ .ctx = undefined }),
    };
    for (store.segments[0..packing.n_segments], 0..) |*segment, i| {
        segment.buffer = @enumFromInt(i + 1);
        segment.capacity = packing.segmentBytes(@intCast(i));
    }

    return store;
}

test "KvStore: every row of every region lands once inside one allocation" {
    // Six regions of 192 bytes, two to an allocation of 384 bytes.
    const layout = kv_index.Layout.init(3, 4, 4, 3, .f32);
    const lim: limits.BufferLimits = .{
        .storage_range = 192,
        .allocation_size = 384,
        .offset_alignment = 16,
    };
    const store = fakeKvStore(layout, lim);
    try std.testing.expectEqual(@as(u32, 3), store.packing.n_segments);
    try std.testing.expectEqual(layout.totalBytes(), store.deviceBytes());

    const row_bytes = layout.rowBytes();
    const rows_per_segment = 24;
    try std.testing.expectEqual(lim.allocation_size, rows_per_segment * row_bytes);
    var seen = [_][rows_per_segment]bool{[_]bool{false} ** rows_per_segment} ** 3;
    var count: u32 = 0;
    for (0..layout.n_layers) |l| {
        const layer: u32 = @intCast(l);
        for ([_]kv_index.Tensor{ .k, .v }) |t| {
            const whole = store.region(layer, t);
            try std.testing.expectEqual(layout.regionBytes(), whole.range);
            try std.testing.expectEqual(@as(u64, 0), whole.offset % lim.offset_alignment);
            for (0..layout.n_slots) |s| {
                const slot: u32 = @intCast(s);
                const view = store.slotView(layer, t, slot);
                try std.testing.expectEqual(whole.buffer, view.buffer);
                try std.testing.expectEqual(
                    whole.offset + layout.slotByteOffset(slot),
                    view.offset,
                );
                try std.testing.expectEqual(@as(u64, layout.window) * row_bytes, view.range);
                for (0..layout.window) |p| {
                    const row = store.rowAt(layer, t, slot, @intCast(p));
                    try std.testing.expectEqual(view.buffer, row.buffer);
                    // The rows of one slot fill its view in position order.
                    try std.testing.expectEqual(view.offset + @as(u64, p) * row_bytes, row.offset);
                    const segment = @intFromEnum(row.buffer) - 1;
                    try std.testing.expect(segment < store.packing.n_segments);
                    try std.testing.expect(
                        row.offset + row_bytes <= store.segments[segment].capacity,
                    );
                    const at: usize = @intCast(row.offset / row_bytes);
                    try std.testing.expect(!seen[segment][at]);
                    seen[segment][at] = true;
                    count += 1;
                }
            }
        }
    }

    // The rows hold as many bytes as the allocations, so no allocation byte
    // stays outside a row.
    try std.testing.expectEqual(@as(u32, 3 * rows_per_segment), count);
}

test "KvStore: two 2B sessions under the raised range start slot 1 at 128 MiB in each region" {
    // Qwen3.5-2B at f16: six full-attention layers of 512-element rows, two
    // slots of 131072 positions, a storage range of 512 MiB, and allocations
    // of 2 GiB. Each region holds 256 MiB, so eight regions share the first
    // allocation and four share the second.
    const layout = kv_index.Layout.init(2, 131_072, 512, 6, .f16);
    const lim: limits.BufferLimits = .{
        .storage_range = 536_870_912,
        .allocation_size = 2_147_483_648,
        .offset_alignment = 16,
    };
    const store = fakeKvStore(layout, lim);
    const mib: u64 = 1 << 20;
    try std.testing.expectEqual(@as(u32, 2), store.packing.n_segments);
    try std.testing.expectEqual(3072 * mib, store.deviceBytes());

    const cases = [_]struct { layer: u32, t: kv_index.Tensor, segment: u64, offset: u64 }{
        .{ .layer = 0, .t = .k, .segment = 0, .offset = 0 },
        .{ .layer = 5, .t = .k, .segment = 0, .offset = 1280 * mib },
        .{ .layer = 1, .t = .v, .segment = 0, .offset = 1792 * mib },
        .{ .layer = 2, .t = .v, .segment = 1, .offset = 0 },
        .{ .layer = 5, .t = .v, .segment = 1, .offset = 768 * mib },
    };
    for (cases) |c| {
        const whole = store.region(c.layer, c.t);
        try std.testing.expectEqual(c.segment + 1, @intFromEnum(whole.buffer));
        try std.testing.expectEqual(c.offset, whole.offset);
        try std.testing.expectEqual(256 * mib, whole.range);
        const second = store.slotView(c.layer, c.t, 1);
        try std.testing.expectEqual(c.offset + 128 * mib, second.offset);
        try std.testing.expectEqual(128 * mib, second.range);
        const last = store.rowAt(c.layer, c.t, 1, 131_071);
        try std.testing.expectEqual(c.offset + 256 * mib - 1024, last.offset);
    }
}
