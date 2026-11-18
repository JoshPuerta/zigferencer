//! The backend that prefetches. It takes the Q8_0 linears from `linear_simd.prefetched`, the f32
//! linears from `linear_simd`, the attention kernels from `attention_simd.prefetched`,
//! `residualAdd`, `rmsnorm`, `swiglu`, and `sigmoidGate` from `elementwise_simd`, the GDN key-head
//! unit from `gdn_simd`, and every other op from `naive`. The f32 linears are the ones of `simd`,
//! with no prefetch. This is the auto default on a target with AVX2 and FMA.
//!
//! Requirements. Every AVX2 target also carries F16C. Without F16C the f16 widening stays
//! correct and runs scalar, so `requires` does not list it.
//!
//! Q4 and Q6_K. The q4 CPU path exists for completeness, so it reuses the naive scalar
//! dequant-and-dot. The runner still parallelizes it across outputs.

const std = @import("std");
const attention_simd = @import("attention_simd.zig");
const elementwise_simd = @import("elementwise_simd.zig");
const gdn_simd = @import("gdn_simd.zig");
const linear_simd = @import("linear_simd.zig");
const naive = @import("naive.zig");
const op_requires = @import("../core/op_requires.zig");
const Tensor = @import("../core/tensor.zig").Tensor;

// -- constants --------------------------------------------------------------

/// AVX2 and FMA on x86_64.
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
pub const attentionPartTyped = attention_simd.prefetched.attentionPartTyped;
pub const attentionMerge = attention_simd.prefetched.attentionMerge;
pub const attentionRowTyped = attention_simd.prefetched.attentionRowTyped;
pub const gdnStepKeyHead = gdn_simd.kernels.stepKeyHead;

pub const linearF32Q40 = naive.linearF32Q40;
pub const linearF32Q40Batch = naive.linearF32Q40Batch;
pub const linearF32Q41 = naive.linearF32Q41;
pub const linearF32Q41Batch = naive.linearF32Q41Batch;
pub const linearF32Q6K = naive.linearF32Q6K;
pub const linearF32Q6KBatch = naive.linearF32Q6KBatch;
pub const linearF32Q80 = linear_simd.prefetched.linearF32Q80;
pub const linearF32Q80Batch = linear_simd.prefetched.linearF32Q80Batch;
pub const linearF32 = linear_simd.linearF32;
pub const linearF32Batch = linear_simd.linearF32Batch;

// -- tests ------------------------------------------------------------------

fn roundtripCheck(in_features: usize, out_features: usize) !void {
    const gpa = std.testing.allocator;
    const output_bytes = (in_features / 32) * 34;

    const input = try gpa.alignedAlloc(f32, .of(f32), in_features);
    defer gpa.free(input);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * output_bytes);
    defer gpa.free(w_data);

    const out_naive = try gpa.alloc(f32, out_features);
    defer gpa.free(out_naive);

    const out_pf = try gpa.alloc(f32, out_features);
    defer gpa.free(out_pf);

    for (input, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
    }

    linear_simd.fillTestWeight(w_data, in_features, out_features, output_bytes);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(output_bytes), 0, 0, 0 },
        .data = w_data,
    };

    naive.linearF32Q80(input, w, out_naive);
    linearF32Q80(input, w, out_pf);

    for (out_naive, out_pf) |n, s| {
        try std.testing.expectApproxEqRel(n, s, 1e-3);
    }
}

test "simd_pf linearF32Q80 matches naive (out%4 == 0)" {
    try roundtripCheck(576, 1536);
}

test "simd_pf linearF32Q80 matches naive with tail (out%4 != 0)" {
    try roundtripCheck(576, 575);
}

test "simd_pf linearF32Q80Batch is bit-identical to per-row linearF32Q80" {
    const gpa = std.testing.allocator;
    const in_features: usize = 96;
    const out_features: usize = 11; // group outputs + a 3-output tail
    const batch: usize = 5;
    const out_stride: usize = 16; // > out_features: exercise the stride path
    const output_bytes = (in_features / 32) * 34;

    const inputs = try gpa.alignedAlloc(f32, .of(f32), batch * in_features);
    defer gpa.free(inputs);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * output_bytes);
    defer gpa.free(w_data);

    const outs = try gpa.alloc(f32, (batch - 1) * out_stride + out_features);
    defer gpa.free(outs);

    const out_single = try gpa.alloc(f32, out_features);
    defer gpa.free(out_single);

    for (inputs, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.013);
    }

    linear_simd.fillTestWeight(w_data, in_features, out_features, output_bytes);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(output_bytes), 0, 0, 0 },
        .data = w_data,
    };

    linearF32Q80Batch(inputs, w, outs, batch, out_stride);
    for (0..batch) |b| {
        linearF32Q80(inputs[b * in_features ..][0..in_features], w, out_single);
        for (out_single, outs[b * out_stride ..][0..out_features]) |s, g| {
            try std.testing.expectEqual(@as(u32, @bitCast(s)), @as(u32, @bitCast(g)));
        }
    }
}
