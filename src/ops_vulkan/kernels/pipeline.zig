//! The compute-pipeline chain that every kernel wrapper embeds, and the descriptor and
//! command-buffer helpers that the wrappers share.

const std = @import("std");
const vk = @import("vulkan");
const limits = @import("../../core/limits.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");

// -- constants --------------------------------------------------------------

/// The largest workgroup count in one dimension that needs no device query. The narrowing kernels
/// assert against it, because their grid follows an element count that the caller supplies.
pub const max_workgroups_x: u32 = limits.max_workgroups;

// -- types ------------------------------------------------------------------

/// The four objects of one compute-pipeline chain, created and destroyed as a unit. Every kernel
/// wrapper embeds one. Only the SPIR-V blob, the binding count, and the push size vary.
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

// -- api --------------------------------------------------------------------

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

pub fn createComputePipeline(
    ctx: *const vulkan_ctx.Context,
    shader_module: vk.ShaderModule,
    pipeline_layout: vk.PipelineLayout,
) !vk.Pipeline {
    // Pin wave32. Every workgroup size must be a multiple of 32. See docs/gpu-rdna2.md section 5.
    // A software device holds fewer lanes, so the size stays with its driver.
    const required_size: vk.PipelineShaderStageRequiredSubgroupSizeCreateInfo = .{
        .required_subgroup_size = limits.required_subgroup_size,
    };
    const stage: vk.PipelineShaderStageCreateInfo = .{
        .p_next = if (ctx.software) null else @ptrCast(&required_size),
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
pub fn createSingleSetPool(
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

pub fn allocateOneSet(
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

// -- helpers ----------------------------------------------------------------

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
