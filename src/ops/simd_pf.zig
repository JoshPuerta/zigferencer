//! A register-blocked f32 x Q8_0 microkernel with software prefetch on
//! `linearF32Q80`. Each inner block computes `mr = 4` output rows with
//! `@Vector(8)` AVX2 and FMA. Attention runs the vector part kernel of
//! `attention_simd.zig`, with a prefetch of the K and V rows 16 positions
//! ahead. The module re-exports the rest of the op API from `naive`. This is
//! the auto default on a target with AVX2 and FMA.
//!
//! Memory-bound shapes stream past L3, where plain SIMD reaches only about 30%
//! of peak DRAM bandwidth. Those shapes are `lm_head`, which streams about
//! 30 MB of Q8_0 per forward pass, and every Llama-3.1-8B matmul. Each
//! inner-block iteration prefetches the same block in the next 4-row group. It
//! uses `.locality = 0`, because those bytes are read once per pass. The
//! prefetch stages the next group's cache lines, so they are warm when the outer
//! loop reaches them. On cache-resident shapes the prefetch is plus or minus 5%
//! noise. On lm_head and the 8B matmuls the win is large.
//!
//! `linearF32Q80Batch` is the same microkernel with a batch loop per 4-row
//! group. The group's rows stay cache-resident across all batch inputs, so the
//! weight matrix streams from DRAM once per batch instead of once per row. That
//! is why the engine batches decode. The B=1 entry point delegates to it, so
//! exactly one kernel body exists. For one (row, output) pair the accumulation
//! order does not depend on the batch size. Batched output is bit-identical to
//! per-row calls, and the tests assert it.

const std = @import("std");
const attention_simd = @import("attention_simd.zig");
const naive = @import("naive.zig");
const op_requires = @import("../core/op_requires.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const f16BitsToF32 = naive.f16BitsToF32;

// -- types ------------------------------------------------------------------

const lanes = 8;
const Vec = @Vector(lanes, f32);
const VecI8 = @Vector(lanes, i8);
const mr = 4;

// -- api --------------------------------------------------------------------

/// Every AVX2 target also carries F16C. Without F16C the f16 widening stays
/// correct and runs scalar, so the requirement does not list it.
pub const requires: op_requires.OpRequires = .{
    .arch = .x86_64,
    .min_simd = .avx2,
    .needs_fma = true,
};

pub const dequantQ80 = naive.dequantQ80;
pub const embedLookup = naive.embedLookup;
pub const linearF32 = naive.linearF32;
pub const linearF32Batch = naive.linearF32Batch;
pub const rmsnorm = naive.rmsnorm;
pub const swiglu = naive.swiglu;
pub const residualAdd = naive.residualAdd;
pub const ropeInplace = naive.ropeInplace;
pub const attentionPartTyped = attention_simd.prefetched.attentionPartTyped;
pub const attentionMerge = attention_simd.prefetched.attentionMerge;
pub const attentionRowTyped = attention_simd.prefetched.attentionRowTyped;
// The q4 CPU path exists for completeness, so it reuses the naive scalar
// dequant-and-dot. The runner still parallelizes it across rows.
pub const linearF32Q40 = naive.linearF32Q40;
pub const linearF32Q40Batch = naive.linearF32Q40Batch;
pub const linearF32Q41 = naive.linearF32Q41;
pub const linearF32Q41Batch = naive.linearF32Q41Batch;
pub const linearF32Q6K = naive.linearF32Q6K;
pub const linearF32Q6KBatch = naive.linearF32Q6KBatch;

pub fn linearF32Q80(input: []const f32, weight: Tensor, out: []f32) void {
    linearF32Q80Batch(input, weight, out, 1, @intCast(weight.shape[1]));
}

/// Batched linear with Q8_0 weights. Row b reads `inputs[b * in_features ..]`
/// and writes `outs[b * out_stride ..]`, the same rule as `naive.linearF32Batch`.
/// The prefetch fires only on a group's last batch row. That keeps one
/// group-pass of lookahead, the same distance the B=1 kernel keeps.
pub fn linearF32Q80Batch(inputs: []const f32, weight: Tensor, outs: []f32, batch: usize, out_stride: usize) void {
    std.debug.assert(weight.dtype == .q8_0);
    std.debug.assert(weight.n_dims == 2);
    std.debug.assert(batch > 0);
    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(inputs.len == batch * in_features);
    std.debug.assert(out_stride >= out_features);
    std.debug.assert(outs.len >= (batch - 1) * out_stride + out_features);
    std.debug.assert(in_features % 32 == 0);

    const row_bytes = (in_features / 32) * 34;
    const n_blocks = in_features / 32;

    var j: usize = 0;
    while (j + mr <= out_features) : (j += mr) {
        const row_offs: [mr]usize = .{
            (j + 0) * row_bytes, (j + 1) * row_bytes, (j + 2) * row_bytes, (j + 3) * row_bytes,
        };
        // The prefetch targets the next 4-row group. The last outer pass has no
        // next group, so it skips the prefetch.
        const has_next = j + 2 * mr <= out_features;
        const pf_offs: [mr]usize = .{
            (j + mr + 0) * row_bytes, (j + mr + 1) * row_bytes, (j + mr + 2) * row_bytes, (j + mr + 3) * row_bytes,
        };

        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const input = inputs[b * in_features ..][0..in_features];
            const sums = groupPass(input, weight.data, row_offs, pf_offs, has_next and b + 1 == batch, n_blocks);
            outs[b * out_stride + j + 0] = sums[0];
            outs[b * out_stride + j + 1] = sums[1];
            outs[b * out_stride + j + 2] = sums[2];
            outs[b * out_stride + j + 3] = sums[3];
        }
    }

    while (j < out_features) : (j += 1) {
        const row_off = j * row_bytes;
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const input = inputs[b * in_features ..][0..in_features];
            outs[b * out_stride + j] = rowPass(input, weight.data, row_off, n_blocks);
        }
    }

    if (std.debug.runtime_safety) {
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            for (outs[b * out_stride ..][0..out_features]) |x| std.debug.assert(std.math.isFinite(x));
        }
    }
}

// -- helpers ----------------------------------------------------------------

/// One input row against one 4-row weight group, the register-blocked core.
/// `do_prefetch` stages the next group's matching block on each iteration, so
/// the `4 * n_blocks` cache-line prefetches spread evenly across the pass.
/// Locality 0 means stream once and do not pollute L1.
inline fn groupPass(
    input: []const f32,
    data: []const u8,
    row_offs: [mr]usize,
    pf_offs: [mr]usize,
    do_prefetch: bool,
    n_blocks: usize,
) [mr]f32 {
    var acc0: Vec = @splat(0);
    var acc1: Vec = @splat(0);
    var acc2: Vec = @splat(0);
    var acc3: Vec = @splat(0);

    var blk: usize = 0;
    while (blk < n_blocks) : (blk += 1) {
        const blk_off_in_row = blk * 34;

        if (do_prefetch) {
            @prefetch(&data[pf_offs[0] + blk_off_in_row], .{ .rw = .read, .locality = 0, .cache = .data });
            @prefetch(&data[pf_offs[1] + blk_off_in_row], .{ .rw = .read, .locality = 0, .cache = .data });
            @prefetch(&data[pf_offs[2] + blk_off_in_row], .{ .rw = .read, .locality = 0, .cache = .data });
            @prefetch(&data[pf_offs[3] + blk_off_in_row], .{ .rw = .read, .locality = 0, .cache = .data });
        }

        const s0v: Vec = @splat(f16BitsToF32(std.mem.readInt(u16, data[row_offs[0] + blk_off_in_row ..][0..2], .little)));
        const s1v: Vec = @splat(f16BitsToF32(std.mem.readInt(u16, data[row_offs[1] + blk_off_in_row ..][0..2], .little)));
        const s2v: Vec = @splat(f16BitsToF32(std.mem.readInt(u16, data[row_offs[2] + blk_off_in_row ..][0..2], .little)));
        const s3v: Vec = @splat(f16BitsToF32(std.mem.readInt(u16, data[row_offs[3] + blk_off_in_row ..][0..2], .little)));

        const in_base = blk * 32;

        inline for (0..4) |q| {
            const in_v: Vec = input[in_base + q * lanes ..][0..lanes].*;

            const w0_arr: [lanes]i8 = @bitCast(data[row_offs[0] + blk_off_in_row + 2 + q * lanes ..][0..lanes].*);
            const w1_arr: [lanes]i8 = @bitCast(data[row_offs[1] + blk_off_in_row + 2 + q * lanes ..][0..lanes].*);
            const w2_arr: [lanes]i8 = @bitCast(data[row_offs[2] + blk_off_in_row + 2 + q * lanes ..][0..lanes].*);
            const w3_arr: [lanes]i8 = @bitCast(data[row_offs[3] + blk_off_in_row + 2 + q * lanes ..][0..lanes].*);

            const w0v: Vec = s0v * @as(Vec, @floatFromInt(@as(VecI8, w0_arr)));
            const w1v: Vec = s1v * @as(Vec, @floatFromInt(@as(VecI8, w1_arr)));
            const w2v: Vec = s2v * @as(Vec, @floatFromInt(@as(VecI8, w2_arr)));
            const w3v: Vec = s3v * @as(Vec, @floatFromInt(@as(VecI8, w3_arr)));

            acc0 += w0v * in_v;
            acc1 += w1v * in_v;
            acc2 += w2v * in_v;
            acc3 += w3v * in_v;
        }
    }

    return .{ @reduce(.Add, acc0), @reduce(.Add, acc1), @reduce(.Add, acc2), @reduce(.Add, acc3) };
}

/// The tail rows left when `out_features % mr != 0`. One row, no prefetch, and
/// the same vector walk.
inline fn rowPass(input: []const f32, data: []const u8, row_off: usize, n_blocks: usize) f32 {
    var acc: Vec = @splat(0);
    var blk: usize = 0;
    while (blk < n_blocks) : (blk += 1) {
        const blk_off = row_off + blk * 34;
        const scale = f16BitsToF32(std.mem.readInt(u16, data[blk_off..][0..2], .little));
        const scale_v: Vec = @splat(scale);
        const in_base = blk * 32;
        inline for (0..4) |q| {
            const i8_arr: [lanes]i8 = @bitCast(data[blk_off + 2 + q * lanes ..][0..lanes].*);
            const i8_v: VecI8 = i8_arr;
            const w_v: Vec = scale_v * @as(Vec, @floatFromInt(i8_v));
            const in_v: Vec = input[in_base + q * lanes ..][0..lanes].*;
            acc += w_v * in_v;
        }
    }
    return @reduce(.Add, acc);
}

// -- tests ------------------------------------------------------------------

fn fillTestWeight(w_data: []u8, in_features: usize, out_features: usize, row_bytes: usize) void {
    for (0..out_features) |row| {
        const off = row * row_bytes;
        var blk: usize = 0;
        while (blk < in_features / 32) : (blk += 1) {
            const blk_off = off + blk * 34;
            std.mem.writeInt(u16, w_data[blk_off..][0..2], 0x3c00, .little);
            var i: usize = 0;
            while (i < 32) : (i += 1) {
                const v_i32: i32 = @as(i32, @intCast((row + blk * 32 + i) % 13)) - 6;
                const v: i8 = @intCast(v_i32);
                w_data[blk_off + 2 + i] = @bitCast(v);
            }
        }
    }
}

fn roundtripCheck(in_features: usize, out_features: usize) !void {
    const gpa = std.testing.allocator;
    const row_bytes = (in_features / 32) * 34;

    const input = try gpa.alignedAlloc(f32, .of(f32), in_features);
    defer gpa.free(input);
    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);
    const out_naive = try gpa.alloc(f32, out_features);
    defer gpa.free(out_naive);
    const out_pf = try gpa.alloc(f32, out_features);
    defer gpa.free(out_pf);

    for (input, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
    fillTestWeight(w_data, in_features, out_features, row_bytes);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = w_data,
    };

    naive.linearF32Q80(input, w, out_naive);
    linearF32Q80(input, w, out_pf);

    for (out_naive, out_pf) |n, s| try std.testing.expectApproxEqRel(n, s, 1e-3);
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
    const out_features: usize = 11; // group rows + a 3-row tail
    const batch: usize = 5;
    const out_stride: usize = 16; // > out_features: exercise the stride path
    const row_bytes = (in_features / 32) * 34;

    const inputs = try gpa.alignedAlloc(f32, .of(f32), batch * in_features);
    defer gpa.free(inputs);
    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);
    const outs = try gpa.alloc(f32, (batch - 1) * out_stride + out_features);
    defer gpa.free(outs);
    const out_single = try gpa.alloc(f32, out_features);
    defer gpa.free(out_single);

    for (inputs, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.013);
    fillTestWeight(w_data, in_features, out_features, row_bytes);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
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
