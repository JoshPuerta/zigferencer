//! The `@Vector(8, f32)` variant of `ops/naive.zig`. The public signatures match naive exactly,
//! so the runner selects this backend by swapping one import. It takes the f32 linears from
//! `linear_simd` and the Q8_0 linears from `linear_simd.plain`, the attention part kernel and the
//! serial row from `attention_simd.zig`, `residualAdd`, `rmsnorm`, `swiglu`, and `sigmoidGate`
//! from `elementwise_simd.zig`, and the GDN key-head unit from `gdn_simd.zig`. It re-exports every
//! other op from naive unchanged.
//!
//! Width. Width 8 matches the 256-bit AVX2 ymm register, and Zig lowers it to a NEON 4+4 pair.
//!
//! Reduction order. A vector reduction sums in a different order than a scalar loop, and the
//! elementwise ops take the vector sigmoid. The f32 and Q8_0 linears add each product with one
//! fused multiply-add and add the lanes in the lane tree order of `lane_tree_simd`. The GDN step
//! takes the readout form and the lane arithmetic of `gdn_simd`. So the output differs bit for bit
//! from naive but stays inside the declared tolerance. naive stays the reference.
//!
//! Requirements. Every AVX2 target also carries F16C. Without F16C the f16 widening stays
//! correct and runs scalar, so `requires` does not list it.
//!
//! Q4 and Q6_K. The q4 CPU path exists for completeness, so it reuses the naive scalar
//! dequant-and-dot. The runner still parallelizes it across rows.
//!
//! Batch. The Q8_0 batch form is the panel path of `linear_simd`, bit-identical to per-row calls.
//! The f32 batch form runs one B = 1 call per row, so it matches the single-row call bit for bit.

const std = @import("std");
const attention_simd = @import("attention_simd.zig");
const elementwise_simd = @import("elementwise_simd.zig");
const gdn_simd = @import("gdn_simd.zig");
const linear_simd = @import("linear_simd.zig");
const naive = @import("naive.zig");
const op_requires = @import("../core/op_requires.zig");
const Tensor = @import("../core/tensor.zig").Tensor;

// -- constants --------------------------------------------------------------

/// AVX2 and FMA on x86_64. The build system rejects `-Dops-backend=simd` on a lesser target.
pub const requires: op_requires.OpRequires = .{
    .arch = .x86_64,
    .min_simd = .avx2,
    .needs_fma = true,
};

pub const swiglu_flops_per_element = elementwise_simd.swiglu_flops_per_element;

// -- api --------------------------------------------------------------------

pub const dequantQ80 = naive.dequantQ80;
pub const embedLookup = naive.embedLookup;

pub const residualAdd = elementwise_simd.residualAdd;
pub const rmsnorm = elementwise_simd.rmsnorm;
pub const swiglu = elementwise_simd.swiglu;
pub const sigmoidGate = elementwise_simd.sigmoidGate;
pub const ropeInplace = naive.ropeInplace;
pub const ropeHeads = naive.ropeHeads;
pub const RopeTable = naive.RopeTable;
pub const attentionPartTyped = attention_simd.plain.attentionPartTyped;
pub const attentionMerge = attention_simd.plain.attentionMerge;
pub const attentionRowTyped = attention_simd.plain.attentionRowTyped;
pub const gdnStepKeyHead = gdn_simd.kernels.stepKeyHead;

pub const linearF32Q40 = naive.linearF32Q40;
pub const linearF32Q40Batch = naive.linearF32Q40Batch;
pub const linearF32Q41 = naive.linearF32Q41;
pub const linearF32Q41Batch = naive.linearF32Q41Batch;
pub const linearF32Q6K = naive.linearF32Q6K;
pub const linearF32Q6KBatch = naive.linearF32Q6KBatch;
pub const linearF32Q80 = linear_simd.plain.linearF32Q80;
pub const linearF32Q80Batch = linear_simd.plain.linearF32Q80Batch;
pub const linearF32 = linear_simd.linearF32;
pub const linearF32Batch = linear_simd.linearF32Batch;

// -- tests ------------------------------------------------------------------

test "simd linearF32Q80Batch is bit-identical to per-row simd linearF32Q80" {
    const gpa = std.testing.allocator;
    const in_features: usize = 64;
    const out_features: usize = 10;
    const batch: usize = 3;
    const row_bytes = (in_features / 32) * 34;

    const inputs = try gpa.alloc(f32, batch * in_features);
    defer gpa.free(inputs);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);

    const outs = try gpa.alloc(f32, batch * out_features);
    defer gpa.free(outs);

    const out_single = try gpa.alloc(f32, out_features);
    defer gpa.free(out_single);

    for (inputs, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.04);
    }

    linear_simd.fillTestWeight(w_data, in_features, out_features, row_bytes);

    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = w_data,
    };

    linearF32Q80Batch(inputs, w, outs, batch, out_features);
    for (0..batch) |b| {
        linearF32Q80(inputs[b * in_features ..][0..in_features], w, out_single);
        for (out_single, outs[b * out_features ..][0..out_features]) |s, g| {
            try std.testing.expectEqual(@as(u32, @bitCast(s)), @as(u32, @bitCast(g)));
        }
    }
}

test "simd linearF32 matches naive within tolerance" {
    const gpa = std.testing.allocator;
    const in_features: usize = 576;
    const out_features: usize = 1536;

    const input = try gpa.alignedAlloc(f32, .of(f32), in_features);
    defer gpa.free(input);

    const w_data = try gpa.alignedAlloc(u8, .@"32", in_features * out_features * @sizeOf(f32));
    defer gpa.free(w_data);

    const out_naive = try gpa.alloc(f32, out_features);
    defer gpa.free(out_naive);

    const out_simd = try gpa.alloc(f32, out_features);
    defer gpa.free(out_simd);

    for (input, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
    }

    const w_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, w_data));
    for (w_f32, 0..) |*x, i| {
        x.* = @cos(@as(f32, @floatFromInt(i)) * 0.003);
    }

    const w: Tensor = .{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(in_features * @sizeOf(f32)), 4, 0, 0 },
        .data = w_data,
    };

    naive.linearF32(input, w, out_naive);
    linearF32(input, w, out_simd);

    // The SIMD tree rounds differently from the scalar sum over hundreds of terms. So 1e-3
    // relative is the tolerance, not 1e-7.
    for (out_naive, out_simd) |n, s| {
        try std.testing.expectApproxEqRel(n, s, 1e-3);
    }
}

test "simd linearF32Q80 matches naive within tolerance" {
    const gpa = std.testing.allocator;
    const in_features: usize = 576;
    const out_features: usize = 1536;
    const row_bytes = (in_features / 32) * 34;

    const input = try gpa.alignedAlloc(f32, .of(f32), in_features);
    defer gpa.free(input);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);

    const out_naive = try gpa.alloc(f32, out_features);
    defer gpa.free(out_naive);

    const out_simd = try gpa.alloc(f32, out_features);
    defer gpa.free(out_simd);

    for (input, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
    }

    linear_simd.fillTestWeight(w_data, in_features, out_features, row_bytes);

    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = w_data,
    };

    naive.linearF32Q80(input, w, out_naive);
    linearF32Q80(input, w, out_simd);

    // The SIMD tree rounds differently from the scalar sum over hundreds of terms. So 1e-3
    // relative is the tolerance, not 1e-7.
    for (out_naive, out_simd) |n, s| {
        try std.testing.expectApproxEqRel(n, s, 1e-3);
    }
}
