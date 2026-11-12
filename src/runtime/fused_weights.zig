//! The weights of a fused runner in two device arenas, and the upload that fills them.
//!
//! Arenas. Two arenas hold the weights in a few large segments, because one allocation per tensor
//! runs a translation-layer driver out of memory on an 8B model. The upload order of the runner
//! fixes the offset of each weight in an arena. `plan` sizes each segment in a dry run of that
//! order, so the arenas hold the priced bytes plus their alignment padding.
//!
//! Upload stage. One host buffer pair carries every weight to the device, because one pair per
//! tensor creates and frees a host heap for each tensor. `plan` also finds the largest view of each
//! arena, and `stageBytes` caps the pair at one swap chunk. A weight past the pair uploads in
//! passes of whole rows, and the device gets the bytes of a whole-tensor upload.
//!
//! Heaps. The `runner` debug log prints the usage and the budget of each memory heap before and
//! after the upload. A device loss during the upload then has the heap state of its run beside it.
//!
//! Head. A Q6_K lm_head of a Q4_0 model converts to Q4_0, because no Q6_K kernel exists.
//!
//! Comptime. A build check refuses a build without Vulkan.

const std = @import("std");
const build_options = @import("build_options");
const vk = @import("vulkan");

const vulkan_ctx = @import("vulkan_ctx.zig");
const kernels = @import("../ops_vulkan/kernels.zig");
const naive = @import("../ops/naive.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const dtype = @import("../core/dtype.zig");
const layer_kernels = @import("layer_kernels.zig");
const linear_dispatch = @import("linear_dispatch.zig");

const Allocator = std.mem.Allocator;
const Weight = layer_kernels.Weight;
const WeightQuant = layer_kernels.WeightQuant;
const assert = std.debug.assert;

const log = std.log.scoped(.runner);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/fused_weights.zig was included in a non-vulkan build.");
    }
}

// -- types ------------------------------------------------------------------

/// One weight that init uploads, and where its view goes. The runner lists them in upload order.
pub const Upload = struct {
    site: Site,
    /// The GGUF name of the tensor, for the refusal line.
    name: []const u8,
    tensor: *const Tensor,

    pub const Site = union(enum) {
        /// A weight of the layer at range index `index`.
        layer: struct { index: u32, role: Weight },
        output_norm,
        lm_head,
    };

    pub fn isMatrix(self: Upload) bool {
        return switch (self.site) {
            .layer => |at| at.role.isMatrix(),
            .output_norm => false,
            .lm_head => true,
        };
    }

    /// The bytes of the view in each arena. A matrix keeps the f16 scale of each block in the
    /// small arena and the quants in the quants arena, both in the form of `quant`.
    pub fn arenaBytes(self: Upload, quant: WeightQuant) ArenaBytes {
        if (!self.isMatrix()) return .{ .small = self.tensor.data.len, .quants = 0 };

        const form: dtype.GgmlType = switch (quant) {
            .q8_0 => .q8_0,
            .q4_0 => .q4_0,
        };
        const n_in: u64 = self.tensor.shape[0];
        const n_out: u64 = self.tensor.shape[1];
        const blocks = n_out * (n_in / dtype.blockSize(form));
        const scale_bytes = @sizeOf(f16);
        return .{
            .small = blocks * scale_bytes,
            .quants = blocks * (dtype.bytesPerBlock(form) - scale_bytes),
        };
    }
};

/// The bytes of one upload in the small arena and in the quants arena.
pub const ArenaBytes = struct {
    small: u64,
    quants: u64,
};

/// The weights of a runner in two arenas: the f16 scales and the f32 vectors, then the i8 quants.
/// An arena weight frees nothing, because its arena owns the memory.
pub const WeightMemory = struct {
    ctx: *const vulkan_ctx.Context,
    small: kernels.VramArena,
    quants: kernels.VramArena,

    pub fn init(ctx: *const vulkan_ctx.Context, gpa: Allocator) WeightMemory {
        const usage: vk.BufferUsageFlags = .{
            .storage_buffer_bit = true,
            .transfer_dst_bit = true,
        };
        return .{
            .ctx = ctx,
            .small = .init(ctx, gpa, usage, kernels.VramArena.seg_bytes_small),
            .quants = .init(ctx, gpa, usage, kernels.VramArena.seg_bytes_quants),
        };
    }

    pub fn deinit(self: *WeightMemory) void {
        self.quants.deinit();
        self.small.deinit();
    }

    /// Size each segment of both arenas to its fill in a dry run of `uploads`, before the first
    /// upload. The dry run places each view as `alloc` does, so the upload fills every segment.
    /// Returns the largest view of each arena, which sizes the upload stage.
    pub fn plan(
        self: *WeightMemory,
        uploads: []const Upload,
        quant: WeightQuant,
    ) Allocator.Error!ArenaBytes {
        const Packing = kernels.VramArena.Packing;
        const gpa = self.small.gpa;
        var small: Packing = .init(self.small.packing.seg_capacity, self.small.packing.align_bytes);
        defer small.deinit(gpa);

        var quants: Packing = .init(
            self.quants.packing.seg_capacity,
            self.quants.packing.align_bytes,
        );
        defer quants.deinit(gpa);

        var largest: ArenaBytes = .{ .small = 0, .quants = 0 };
        for (uploads) |u| {
            const bytes = u.arenaBytes(quant);
            largest.small = @max(largest.small, bytes.small);
            largest.quants = @max(largest.quants, bytes.quants);
            _ = try small.place(gpa, bytes.small);
            if (u.isMatrix()) {
                _ = try quants.place(gpa, bytes.quants);
            }
        }

        try self.small.setPlan(&small);
        try self.quants.setPlan(&quants);
        return largest;
    }

    /// The bytes of the segments of both arenas.
    pub fn heldBytes(self: *const WeightMemory) u64 {
        return self.small.packing.capacityBytes() + self.quants.packing.capacityBytes();
    }

    /// The bytes of the views of both arenas.
    pub fn viewBytes(self: *const WeightMemory) u64 {
        return self.small.packing.request_bytes + self.quants.packing.request_bytes;
    }

    /// The most padding that the views of both arenas can leave.
    pub fn paddingBound(self: *const WeightMemory) u64 {
        return self.small.packing.paddingBound() + self.quants.packing.paddingBound();
    }

    /// Check both arenas after the upload against `priced`, the weight bytes of the planner.
    pub fn assertFull(self: *const WeightMemory, priced: u64) void {
        self.small.packing.assertFull();
        self.quants.packing.assertFull();

        // Each tensor view pads at most the offset alignment (256 bytes by the Vulkan limits), so
        // the padding stays under 1 MiB at `limits.max_layers`, inside the preflight reserve of
        // every row of `limits.vram_shares`.
        assert(self.viewBytes() <= priced);
        assert(self.heldBytes() <= priced + self.paddingBound());
    }

    /// Upload one matmul weight in the form of `quant`. A weight of another dtype is refused.
    pub fn matrix(
        self: *WeightMemory,
        stage: *kernels.UploadStage,
        quant: WeightQuant,
        name: []const u8,
        t: *const Tensor,
    ) !kernels.Q80Weight {
        const held: WeightQuant = switch (t.dtype) {
            .q8_0 => .q8_0,
            .q4_0 => .q4_0,
            else => {
                refuseDtype(name, t.dtype);
                return error.UnsupportedWeightDtype;
            },
        };
        if (held != quant) {
            log.err(
                "tensor {s}: dtype {t} differs from the dtype {t} of the other matrices",
                .{ name, t.dtype, quant },
            );
            return error.UnsupportedWeightDtype;
        }

        // The first dimension is the input features, and the second the output features.
        const n_in: usize = @intCast(t.shape[0]);
        const n_out: usize = @intCast(t.shape[1]);
        const ctx = self.ctx;
        const small = &self.small;
        const quants = &self.quants;
        const Q80Weight = kernels.Q80Weight;
        const weight = switch (quant) {
            .q8_0 => try Q80Weight.initInArenas(ctx, stage, small, quants, t.data, n_in, n_out),
            .q4_0 => try Q80Weight.initInArenasQ40(ctx, stage, small, quants, t.data, n_in, n_out),
        };
        assert(weight.owned == null);
        return weight;
    }

    /// Upload one f32 weight into the small arena through the small buffer of `stage`.
    pub fn vector(
        self: *WeightMemory,
        stage: *kernels.UploadStage,
        name: []const u8,
        t: *const Tensor,
    ) !vk.DescriptorBufferInfo {
        if (t.dtype != .f32) {
            refuseDtype(name, t.dtype);
            return error.UnsupportedWeightDtype;
        }

        const bytes: vk.DeviceSize = t.data.len;
        const view = try self.small.alloc(bytes);
        const copies = [_]kernels.Copy{.{
            .src = stage.small.buffer,
            .dst = view.buffer,
            .size = bytes,
            .dst_offset = view.offset,
        }};
        const side: kernels.StageSide = .{ .to_device = t.data };
        _ = try kernels.walkStage(self.ctx, &copies, &stage.small, side, stage.small.capacity);
        return view;
    }

    /// Upload the lm_head: `output.weight`, or the embedding table that a tied model reuses.
    pub fn lmHead(
        self: *WeightMemory,
        stage: *kernels.UploadStage,
        gpa: Allocator,
        quant: WeightQuant,
        name: []const u8,
        output: *const Tensor,
    ) !kernels.Q80Weight {
        if (output.dtype != .q6_k or quant != .q4_0) {
            return self.matrix(stage, quant, name, output);
        }

        // The lm_head is a small part of the weight stream of a step, so a throughput run
        // accepts the loss of the requantization.
        const q40_bytes = try convertQ6KtoQ40(gpa, output.*);
        defer gpa.free(q40_bytes);

        const in_features: usize = @intCast(output.shape[0]);
        const row_bytes: u32 = @intCast((in_features / 32) * kernels.q4_0_block_bytes);
        const t40: Tensor = .{
            .dtype = .q4_0,
            .n_dims = 2,
            .shape = output.shape,
            .byte_stride = .{ row_bytes, 0, 0, 0 },
            .data = q40_bytes,
        };
        t40.assertValid();
        return self.matrix(stage, .q4_0, name, &t40);
    }
};

// -- api --------------------------------------------------------------------

/// The buffer sizes of the upload stage: the largest view of each arena, capped at `chunk_bytes`.
pub fn stageBytes(largest: ArenaBytes, chunk_bytes: u64) ArenaBytes {
    assert(chunk_bytes >= 1);

    const out: ArenaBytes = .{
        .small = @min(largest.small, chunk_bytes),
        .quants = @min(largest.quants, chunk_bytes),
    };
    assert(out.small <= chunk_bytes);
    assert(out.quants <= chunk_bytes);
    return out;
}

/// Log the refusal of a weight that no fused kernel reads, in the line of the CPU dispatch.
pub fn refuseDtype(name: []const u8, d: dtype.GgmlType) void {
    log.err(linear_dispatch.refusal_fmt, .{ name, d });
}

/// Log the usage and the budget of each memory heap of `ctx`, one line per heap.
pub fn logHeaps(ctx: *const vulkan_ctx.Context, when: []const u8) void {
    if (!ctx.has_memory_budget) {
        log.debug("heaps {s} the weight upload: no live budget (VK_EXT_memory_budget)", .{when});
        return;
    }

    const mib = 1024 * 1024;
    const count = ctx.mem_props.memory_heap_count;
    assert(count <= vk.MAX_MEMORY_HEAPS);
    for (0..count) |i| {
        const heap: u32 = @intCast(i);
        const b = ctx.heapBudget(heap) orelse continue;
        log.debug("heap {d} {s} the weight upload: used {d} of {d} MiB, {d} MiB in the heap", .{
            heap,
            when,
            b.usage / mib,
            b.budget / mib,
            ctx.mem_props.memory_heaps[heap].size / mib,
        });
    }
}

// -- helpers ----------------------------------------------------------------

/// Convert a Q6_K tensor to new Q4_0 bytes, one 256-element superblock at a time through a small
/// f32 scratch, so the conversion needs no f32 copy of the whole lm_head. The caller owns the
/// result.
fn convertQ6KtoQ40(gpa: Allocator, t: Tensor) ![]u8 {
    assert(t.dtype == .q6_k);

    const numel: usize = @as(usize, @intCast(t.shape[0])) * @as(usize, @intCast(t.shape[1]));
    assert(numel % 256 == 0);
    const n_super = numel / 256;
    const out = try gpa.alloc(u8, (numel / 32) * kernels.q4_0_block_bytes);
    errdefer gpa.free(out);

    var scratch: [256]f32 = undefined;
    for (0..n_super) |s| {
        naive.dequantQ6K(t.data[s * 210 ..][0..210], &scratch);
        // 256 elements are 8 Q4_0 blocks of 18 bytes.
        naive.quantizeQ40(&scratch, out[s * 144 ..][0..144]);
    }

    return out;
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

test "upload stage: each buffer holds the largest view of its arena, at most one chunk" {
    const chunk: u64 = 256 << 20;
    const small = stageBytes(.{ .small = 4096, .quants = 1 << 20 }, chunk);
    try testing.expectEqual(@as(u64, 4096), small.small);
    try testing.expectEqual(@as(u64, 1 << 20), small.quants);

    // A Llama-8B Q8_0 lm_head holds 128256 rows of 4096 quants, past one chunk.
    const head: ArenaBytes = .{ .small = 128256 * 128 * 2, .quants = 128256 * 4096 };
    const capped = stageBytes(head, chunk);
    try testing.expectEqual(head.small, capped.small);
    try testing.expectEqual(chunk, capped.quants);
}
