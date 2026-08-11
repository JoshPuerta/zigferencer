//! The `@Vector(8, f32)` variant of `ops/naive.zig`. The public signatures match
//! naive exactly, so the runner selects this backend by swapping one import.
//! The file overrides `linearF32` and `linearF32Q80` with vectorized kernels,
//! takes the attention part kernel and the serial row from `attention_simd.zig`,
//! and re-exports every other op from naive unchanged.
//!
//! Width 8 matches the 256-bit AVX2 ymm register, and Zig lowers it to a NEON
//! 4+4 pair. A vector reduction sums in a different order than a scalar loop, so
//! the output differs bit for bit from naive but stays inside the declared
//! tolerance. naive stays the oracle.

const std = @import("std");
const attention_simd = @import("attention_simd.zig");
const naive = @import("naive.zig");
const op_requires = @import("../core/op_requires.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const f16BitsToF32 = naive.f16BitsToF32;

// -- types ------------------------------------------------------------------

const vec_lanes = 8;
const Vec = @Vector(vec_lanes, f32);
const VecI8 = @Vector(vec_lanes, i8);

// -- api --------------------------------------------------------------------

/// AVX2 and FMA on x86_64. The build system rejects `-Dops-backend=simd` when
/// the target does not meet these.
/// Every AVX2 target also carries F16C. Without F16C the f16 widening stays
/// correct and runs scalar, so the requirement does not list it.
pub const requires: op_requires.OpRequires = .{
    .arch = .x86_64,
    .min_simd = .avx2,
    .needs_fma = true,
};

pub const dequantQ80 = naive.dequantQ80;
pub const embedLookup = naive.embedLookup;
pub const rmsnorm = naive.rmsnorm;
pub const swiglu = naive.swiglu;
pub const residualAdd = naive.residualAdd;
pub const ropeInplace = naive.ropeInplace;
pub const attentionPartTyped = attention_simd.plain.attentionPartTyped;
pub const attentionMerge = attention_simd.plain.attentionMerge;
pub const attentionRowTyped = attention_simd.plain.attentionRowTyped;
// The q4 CPU path exists for completeness, so it reuses the naive scalar
// dequant-and-dot. The runner still parallelizes it across rows.
pub const linearF32Q40 = naive.linearF32Q40;
pub const linearF32Q40Batch = naive.linearF32Q40Batch;
pub const linearF32Q41 = naive.linearF32Q41;
pub const linearF32Q41Batch = naive.linearF32Q41Batch;
pub const linearF32Q6K = naive.linearF32Q6K;
pub const linearF32Q6KBatch = naive.linearF32Q6KBatch;

pub fn linearF32(input: []const f32, weight: Tensor, out: []f32) void {
    std.debug.assert(weight.dtype == .f32);
    std.debug.assert(weight.n_dims == 2);
    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(input.len == in_features);
    std.debug.assert(out.len >= out_features);

    const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, weight.data));

    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        var acc: Vec = @splat(0);
        const row = w[j * in_features ..][0..in_features];
        var k: usize = 0;
        while (k + vec_lanes <= in_features) : (k += vec_lanes) {
            const a: Vec = input[k..][0..vec_lanes].*;
            const b: Vec = row[k..][0..vec_lanes].*;
            acc += a * b;
        }
        var sum: f32 = @reduce(.Add, acc);
        while (k < in_features) : (k += 1) sum += input[k] * row[k];
        out[j] = sum;
    }
    if (std.debug.runtime_safety) {
        for (out[0..out_features]) |x| std.debug.assert(std.math.isFinite(x));
    }
}

pub fn linearF32Q80(input: []const f32, weight: Tensor, out: []f32) void {
    std.debug.assert(weight.dtype == .q8_0);
    std.debug.assert(weight.n_dims == 2);
    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(input.len == in_features);
    std.debug.assert(out.len >= out_features);
    std.debug.assert(in_features % 32 == 0);

    const row_bytes = (in_features / 32) * 34;
    const n_blocks = in_features / 32;

    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        var acc: Vec = @splat(0);
        const row_off = j * row_bytes;

        var blk: usize = 0;
        while (blk < n_blocks) : (blk += 1) {
            const blk_off = row_off + blk * 34;
            const scale = f16BitsToF32(std.mem.readInt(u16, weight.data[blk_off..][0..2], .little));
            const scale_v: Vec = @splat(scale);
            const in_base = blk * 32;

            // One Q8_0 block holds 32 i8 values, which is 4 lanes of 8.
            inline for (0..4) |q| {
                const i8_off = blk_off + 2 + q * vec_lanes;
                const i8_arr: [vec_lanes]i8 = @bitCast(weight.data[i8_off..][0..vec_lanes].*);
                const i8_v: VecI8 = i8_arr;
                const w_v: Vec = scale_v * @as(Vec, @floatFromInt(i8_v));
                const in_v: Vec = input[in_base + q * vec_lanes ..][0..vec_lanes].*;
                acc += w_v * in_v;
            }
        }
        out[j] = @reduce(.Add, acc);
    }
    if (std.debug.runtime_safety) {
        for (out[0..out_features]) |x| std.debug.assert(std.math.isFinite(x));
    }
}

/// Batch by rows over this variant's own kernels. The result is bit-identical
/// to one B=1 call per row. No weight-reuse win exists here, because the
/// batched microkernel lives in `simd_pf`, which is the auto default.
pub fn linearF32Batch(inputs: []const f32, weight: Tensor, outs: []f32, batch: usize, out_stride: usize) void {
    naive.linearBatchPerRow(linearF32, inputs, weight, outs, batch, out_stride);
}

pub fn linearF32Q80Batch(inputs: []const f32, weight: Tensor, outs: []f32, batch: usize, out_stride: usize) void {
    naive.linearBatchPerRow(linearF32Q80, inputs, weight, outs, batch, out_stride);
}

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

    for (inputs, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.04);
    for (0..out_features) |row| {
        const off = row * row_bytes;
        var blk: usize = 0;
        while (blk < in_features / 32) : (blk += 1) {
            const blk_off = off + blk * 34;
            std.mem.writeInt(u16, w_data[blk_off..][0..2], 0x3c00, .little);
            var i: usize = 0;
            while (i < 32) : (i += 1) {
                const v: i8 = @intCast(@as(i32, @intCast((row + blk * 32 + i) % 13)) - 6);
                w_data[blk_off + 2 + i] = @bitCast(v);
            }
        }
    }
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

    for (input, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
    const w_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, w_data));
    for (w_f32, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i)) * 0.003);

    const w: Tensor = .{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(in_features * @sizeOf(f32)), 4, 0, 0 },
        .data = w_data,
    };

    naive.linearF32(input, w, out_naive);
    linearF32(input, w, out_simd);

    // An f32 dot product over hundreds of elements accumulates rounding on the
    // order of N*eps. SIMD reduces in a different tree than scalar, so 1e-3
    // relative is the right tolerance, not 1e-7.
    for (out_naive, out_simd) |n, s| try std.testing.expectApproxEqRel(n, s, 1e-3);
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

    for (input, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
    for (0..out_features) |row| {
        const off = row * row_bytes;
        var blk: usize = 0;
        while (blk < in_features / 32) : (blk += 1) {
            const blk_off = off + blk * 34;
            std.mem.writeInt(u16, w_data[blk_off..][0..2], 0x3c00, .little); // scale = 1.0
            var i: usize = 0;
            while (i < 32) : (i += 1) {
                const v_i32: i32 = @as(i32, @intCast((row + blk * 32 + i) % 13)) - 6;
                const v: i8 = @intCast(v_i32);
                w_data[blk_off + 2 + i] = @bitCast(v);
            }
        }
    }
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = w_data,
    };

    naive.linearF32Q80(input, w, out_naive);
    linearF32Q80(input, w, out_simd);

    // An f32 dot product over hundreds of elements accumulates rounding on the
    // order of N*eps. SIMD reduces in a different tree than scalar, so 1e-3
    // relative is the right tolerance, not 1e-7.
    for (out_naive, out_simd) |n, s| try std.testing.expectApproxEqRel(n, s, 1e-3);
}
