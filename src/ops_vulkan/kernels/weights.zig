//! The resident Q8_0 and Q4_0 weights of the Vulkan backend, and the bulk upload that fills them.
//!
//! Comptime. `UploadStage.uploadOutputs` takes the block form as a comptime value, so one function
//! uploads both forms.
//!
//! Resident weights. Each weight becomes one `Q80Weight` at model-load time and stays in VRAM for
//! the life of the model. A per-call re-upload costs too much: the lm_head alone is about 27 MB.
//! Every Q8_0 and Q4_0 kernel reads the same scales-plus-quants layout. The repack turns each GGUF
//! block (a 2-byte f16 scale plus 32 i8 quants) into two flat arrays: scales by `(j, blk)` and
//! quants by `(j, p)`. The copy keeps every bit pattern, which the bit-exact match against the CPU
//! kernel needs, so never reorder bytes.

const std = @import("std");
const vk = @import("vulkan");
const kv_budget = @import("../../core/kv_budget.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");

const HostBuffer = @import("memory.zig").HostBuffer;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const VramArena = @import("memory.zig").VramArena;
const Copy = @import("submit.zig").Copy;
const SubmitError = @import("submit.zig").SubmitError;
const runStagingCopies = @import("submit.zig").runStagingCopies;

// -- constants --------------------------------------------------------------

/// Q8_0 and Q4_0 block layout, in lockstep with `src/core/dtype.zig` and the CPU decoder in
/// `ops/naive.zig`.
const q8_0_block_elems: usize = 32;

const q8_0_block_bytes: usize = 2 + q8_0_block_elems; // f16 scale + 32 i8 quants

const q4_0_block_elems: usize = 32;

const q4_0_packed_bytes: usize = 16; // 32 weights at 4 bits = 16 bytes

pub const q4_0_block_bytes: usize = 2 + q4_0_packed_bytes; // f16 scale + 16 packed

// -- types ------------------------------------------------------------------

/// A GPU-resident Q8_0 weight matrix: the f16 scales, the packed quants, and the dimensions. A
/// caller builds it once from a CPU-side weight blob, and the matmul kernels read it many times.
pub const Q80Weight = struct {
    ctx: *const vulkan_ctx.Context,
    /// Device-local view in the shape of a descriptor write, in an arena or in the buffers of
    /// `owned`.
    scales: vk.DescriptorBufferInfo,
    quants: vk.DescriptorBufferInfo,
    in_features: u32,
    out_features: u32,
    /// The memory that `init` allocated and that `deinit` frees. Null means that the caller's
    /// arenas own it.
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
        SubmitError;

    /// Build a weight with its own device memory: 2 buffers and 2 memory objects. Use it for
    /// one-off uses: the bench, the per-op tests, and the `wrap.zig` cache. A bulk upload takes
    /// `initInArenas`, about 10 device allocations in total.
    pub fn init(
        ctx: *const vulkan_ctx.Context,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
    ) InitError!Q80Weight {
        // Refuse a weight that does not fill whole blocks.
        if (in_features % q8_0_block_elems != 0) return error.InFeaturesNotMultipleOf32;

        const blocks_per_output = in_features / q8_0_block_elems;
        const output_bytes = blocks_per_output * q8_0_block_bytes;
        if (weight_bytes.len < out_features * output_bytes) return error.WeightSizeMismatch;

        // Repack the weight into host staging.
        const scales_bytes: vk.DeviceSize = @intCast(out_features * blocks_per_output * 2);
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

        // Allocate the device buffers.
        var scales_db: DeviceBuffer = .{ .ctx = ctx };
        errdefer scales_db.deinit();

        var quants_db: DeviceBuffer = .{ .ctx = ctx };
        errdefer quants_db.deinit();

        try scales_db.ensure(scales_bytes);
        try quants_db.ensure(quants_bytes);

        // Upload the repacked bytes.
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

    /// The Q4_0 form of `init`. The quants buffer holds 16 bytes of 4-bit nibbles per block, half
    /// the Q8_0 size. `weight_bytes` is a Q4_0 tensor, and the LinearQ40 kernels extract the
    /// nibbles.
    pub fn initQ40(
        ctx: *const vulkan_ctx.Context,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
    ) InitError!Q80Weight {
        // Refuse a weight that does not fill whole blocks.
        if (in_features % q4_0_block_elems != 0) return error.InFeaturesNotMultipleOf32;

        const blocks_per_output = in_features / q4_0_block_elems;
        const output_bytes = blocks_per_output * q4_0_block_bytes;
        if (weight_bytes.len < out_features * output_bytes) return error.WeightSizeMismatch;

        // Repack the weight into host staging.
        const scales_bytes: vk.DeviceSize = @intCast(out_features * blocks_per_output * 2);
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

        // Allocate the device buffers.
        var scales_db: DeviceBuffer = .{ .ctx = ctx };
        errdefer scales_db.deinit();

        var quants_db: DeviceBuffer = .{ .ctx = ctx };
        errdefer quants_db.deinit();

        try scales_db.ensure(scales_bytes);
        try quants_db.ensure(quants_bytes);

        // Upload the repacked bytes.
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

    /// Sub-allocate scales and quants from two caller-managed arenas and upload through `stage`.
    /// This weight stores views only, so `deinit` frees nothing for it. The arenas free everything
    /// at their own `deinit`.
    pub fn initInArenas(
        ctx: *const vulkan_ctx.Context,
        stage: *UploadStage,
        scales_arena: *VramArena,
        quants_arena: *VramArena,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
    ) (InitError || VramArena.AllocError)!Q80Weight {
        // Refuse a weight that does not fill whole blocks.
        if (in_features % q8_0_block_elems != 0) return error.InFeaturesNotMultipleOf32;

        const blocks_per_output = in_features / q8_0_block_elems;
        const output_bytes = blocks_per_output * q8_0_block_bytes;
        if (weight_bytes.len < out_features * output_bytes) return error.WeightSizeMismatch;

        // Reserve the views, then repack and upload the weights of each output into them.
        const scales_bytes: vk.DeviceSize = @intCast(out_features * blocks_per_output * 2);
        const quants_bytes: vk.DeviceSize = @intCast(out_features * in_features);
        const scales_view = try scales_arena.alloc(scales_bytes);
        const quants_view = try quants_arena.alloc(quants_bytes);
        const views: UploadStage.Views = .{ .scales = scales_view, .quants = quants_view };
        try stage.uploadOutputs(ctx, .q8_0, weight_bytes, in_features, out_features, views);

        return .{
            .ctx = ctx,
            .scales = scales_view,
            .quants = quants_view,
            .in_features = @intCast(in_features),
            .out_features = @intCast(out_features),
            // The arenas own the memory, so `owned` stays null.
        };
    }

    /// The Q4_0 form of `initInArenas`, the arena upload of a fused runner. The quants buffer
    /// holds 16 bytes of 4-bit nibbles per block. `weight_bytes` is a Q4_0 tensor.
    pub fn initInArenasQ40(
        ctx: *const vulkan_ctx.Context,
        stage: *UploadStage,
        scales_arena: *VramArena,
        quants_arena: *VramArena,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
    ) (InitError || VramArena.AllocError)!Q80Weight {
        // Refuse a weight that does not fill whole blocks.
        if (in_features % q4_0_block_elems != 0) return error.InFeaturesNotMultipleOf32;

        const blocks_per_output = in_features / q4_0_block_elems;
        const output_bytes = blocks_per_output * q4_0_block_bytes;
        if (weight_bytes.len < out_features * output_bytes) return error.WeightSizeMismatch;

        // Reserve the views, then repack and upload the weights of each output into them.
        const scales_bytes: vk.DeviceSize = @intCast(out_features * blocks_per_output * 2);
        const quants_bytes: vk.DeviceSize = @intCast(out_features * (in_features / 2));
        const scales_view = try scales_arena.alloc(scales_bytes);
        const quants_view = try quants_arena.alloc(quants_bytes);
        const views: UploadStage.Views = .{ .scales = scales_view, .quants = quants_view };
        try stage.uploadOutputs(ctx, .q4_0, weight_bytes, in_features, out_features, views);

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

/// The one host buffer pair of a bulk weight upload. `small` stages the f16 scales and the f32
/// vectors, and `quants` stages the quants. The caller sizes the pair once and frees it after the
/// last weight. A matrix past the pair uploads in passes over the weights of whole outputs.
pub const UploadStage = struct {
    small: HostBuffer,
    quants: HostBuffer,

    /// The two arena views of one matrix.
    const Views = struct {
        scales: VramArena.View,
        quants: VramArena.View,
    };

    /// The block form of a repacked matrix.
    const Form = enum { q8_0, q4_0 };

    pub fn init(ctx: *const vulkan_ctx.Context) UploadStage {
        return .{ .small = .{ .ctx = ctx }, .quants = .{ .ctx = ctx } };
    }

    pub fn deinit(self: *UploadStage) void {
        self.quants.deinit();
        self.small.deinit();
    }

    /// Create both buffers. A failure leaves `deinit` to free what it created.
    pub fn ensure(self: *UploadStage, small_bytes: u64, quants_bytes: u64) !void {
        std.debug.assert(small_bytes >= 1);
        std.debug.assert(quants_bytes >= 1);

        try self.small.ensure(small_bytes);
        try self.quants.ensure(quants_bytes);
    }

    /// Repack the weights of the outputs of one matrix into the pair and copy them to `views`, one
    /// submit per pass. Output r starts at r times its byte size in each view. A pass of outputs
    /// thus lands at the offset of its first output, and the views get the bytes of a single pass.
    fn uploadOutputs(
        self: *UploadStage,
        ctx: *const vulkan_ctx.Context,
        comptime form: Form,
        weight_bytes: []const u8,
        in_features: usize,
        out_features: usize,
        views: Views,
    ) SubmitError!void {
        const block_bytes = switch (form) {
            .q8_0 => q8_0_block_bytes,
            .q4_0 => q4_0_block_bytes,
        };
        const repack = switch (form) {
            .q8_0 => repackQ80ToScalesQuants,
            .q4_0 => repackQ40ToScalesQuants,
        };
        const quant_bytes_per_output: usize = switch (form) {
            .q8_0 => in_features,
            .q4_0 => in_features / 2,
        };
        // Both forms hold 32 weights in a block.
        const blocks_per_output = in_features / q8_0_block_elems;
        const src_bytes_per_output = blocks_per_output * block_bytes;
        const scale_bytes_per_output = blocks_per_output * 2;
        std.debug.assert(views.scales.range == out_features * scale_bytes_per_output);
        std.debug.assert(views.quants.range == out_features * quant_bytes_per_output);

        // A pass holds the most whole outputs that both buffers fit.
        const outputs_max = @min(
            self.small.capacity / scale_bytes_per_output,
            self.quants.capacity / quant_bytes_per_output,
        );
        std.debug.assert(outputs_max >= 1);

        var walk = kv_budget.ChunkWalk.init(out_features, outputs_max);
        var passes: u64 = 0;
        while (walk.next()) |chunk| : (passes += 1) {
            const first: usize = @intCast(chunk.offset);
            const n: usize = @intCast(chunk.len);
            repack(
                weight_bytes[first * src_bytes_per_output ..][0 .. n * src_bytes_per_output],
                in_features,
                n,
                self.small.sliceTyped(u8, n * scale_bytes_per_output),
                self.quants.sliceTyped(u8, n * quant_bytes_per_output),
            );
            const copies = [_]Copy{
                .{
                    .src = self.small.buffer,
                    .dst = views.scales.buffer,
                    .size = n * scale_bytes_per_output,
                    .dst_offset = views.scales.offset + first * scale_bytes_per_output,
                },
                .{
                    .src = self.quants.buffer,
                    .dst = views.quants.buffer,
                    .size = n * quant_bytes_per_output,
                    .dst_offset = views.quants.offset + first * quant_bytes_per_output,
                },
            };
            try runStagingCopies(ctx, &copies);
        }

        std.debug.assert(passes == kv_budget.ChunkWalk.count(out_features, outputs_max));
    }
};

// -- helpers ----------------------------------------------------------------

/// Repack a Q8_0 weight matrix from the interleaved GGUF layout into two flat arrays. See
/// "Resident weights" in the header. Do not reorder bytes here.
fn repackQ80ToScalesQuants(
    weight_data: []const u8,
    in_features: usize,
    out_features: usize,
    scales_out: []u8,
    quants_out: []u8,
) void {
    const blocks_per_output = in_features / q8_0_block_elems;
    const output_bytes = blocks_per_output * q8_0_block_bytes;
    std.debug.assert(scales_out.len >= out_features * blocks_per_output * 2);
    std.debug.assert(quants_out.len >= out_features * in_features);
    for (0..out_features) |j| {
        const output_off = j * output_bytes;
        for (0..blocks_per_output) |blk| {
            const blk_off = output_off + blk * q8_0_block_bytes;
            // 2 bytes of f16 scale -> scales_out at j * blocks_per_output + blk.
            const scales_idx = (j * blocks_per_output + blk) * 2;
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

/// The Q4_0 form of `repackQ80ToScalesQuants`. An output holds `in/32` blocks of a 2-byte f16
/// scale plus 16 packed bytes. The packed bytes keep every bit pattern, and the shader extracts the
/// nibbles in the layout of the CPU kernel.
fn repackQ40ToScalesQuants(
    weight_data: []const u8,
    in_features: usize,
    out_features: usize,
    scales_out: []u8,
    quants_out: []u8,
) void {
    const blocks_per_output = in_features / q4_0_block_elems;
    const output_bytes = blocks_per_output * q4_0_block_bytes;
    std.debug.assert(scales_out.len >= out_features * blocks_per_output * 2);
    std.debug.assert(quants_out.len >= out_features * (in_features / 2));
    for (0..out_features) |j| {
        const output_off = j * output_bytes;
        for (0..blocks_per_output) |blk| {
            const blk_off = output_off + blk * q4_0_block_bytes;
            const scales_idx = (j * blocks_per_output + blk) * 2;
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
