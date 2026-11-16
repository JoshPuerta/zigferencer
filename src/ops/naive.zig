//! Scalar f32 reference ops, the full set a Llama forward pass needs. That set is matmul, embed
//! lookup, RMSNorm, SwiGLU/SiLU, residual add, RoPE, and decode attention as row parts with an
//! ordered merge. Every other ops variant re-exports the remaining ops from here. It overrides
//! the linear kernels, and the SIMD variants also override the attention part kernel and the
//! serial row.
//!
//! Comptime. `E` makes the kernels generic over f32 and f16, and `kernel` binds a row kernel at
//! build time.
//!
//! Reference. This file is the correctness reference. Every other variant must match its output
//! inside a stated tolerance. No fusion, no shortcuts. Each op asserts its shape preconditions and,
//! under runtime safety, a finiteness postcondition. Boundary inputs are `Tensor`, and inner
//! slices are raw `[]f32` the caller owns.
//!
//! Linear layout. `out = input @ W.T` with `W` in `[out, in]` row-major, the PyTorch layout that
//! GGUF stores. GGUF's `weight.shape = [in, out]` is the reversed shape. The embed table is
//! PyTorch `[vocab, embed_dim]`, so GGUF `shape[0]` is `embed_dim` and `shape[1]` is the vocab.
//!
//! Quant blocks. Each row is `in / 32` blocks, except Q6_K.
//! - Q8_0: 34 bytes, an f16 scale and 32 i8 values.
//! - Q4_0: 18 bytes, an f16 scale and 16 packed bytes. Weight `i` (i < 16) is the LOW nibble of
//!   byte `i`, and weight `i + 16` is the HIGH nibble. A nibble in [0, 15] dequantizes as
//!   `(nibble - 8) * scale`.
//! - Q4_1: 20 bytes, an f16 scale `d`, an f16 min `m`, and 16 packed bytes in the Q4_0 order. A
//!   nibble dequantizes as `nibble * d + m`, with no -8 bias. The `m` term carries that bias.
//! - Q6_K: a superblock of 256 values in 210 bytes, `ql[128]` (low 4 bits) | `qh[64]` (high 2
//!   bits) | `scales[16]` (i8) | `d` (f16). Each value is `d * scale * (q6 - 32)`, where the
//!   6-bit `q6` comes from one ql nibble plus two qh bits. This mirrors ggml `block_q6_K`. The
//!   linear op dequantizes one superblock at a time into a 256-element stack buffer, so the
//!   caller needs no scratch. ggml emits Q6_K for the logit head of Q4_0 ftypes.
//!
//! Batch. A batched linear runs `batch` input rows against one weight. Row b reads
//! `inputs[b * in ..][0..in]` and writes `outs[b * out_stride ..][0..out]`. `out_stride` lets a
//! caller slice the output rows of the weight and still write into full-width rows, as
//! `parallel.linearBatch` does. The j-outer and b-inner nest streams each weight row once for all
//! inputs. For one (row, output) pair the reduction order is exactly the B=1 order, so each row
//! is bit-identical to a B=1 call. The engine batch tests pin that.
//!
//! RoPE. `RopeStyle` follows the llama.cpp rope type per architecture: the llama family is
//! `norm`, the qwen family is `neox`. The style must match how the conversion laid out the GGUF Q
//! and K weights. The wrong style scrambles attention geometry, and greedy decode degenerates into
//! loops. `freq_factors` is the per-pair frequency divisor from a `rope_freqs.weight` tensor
//! (Llama-3.1 long-context scaling). Pair i rotates by `pos * base^(-2i/D) / freq_factors[i]`.
//! Null is classic RoPE.
//!
//! Attention part. One part covers the query heads `[h_lo, h_hi)` over the positions
//! `[t_lo, t_hi)`. The kernel folds the part with the online softmax, one tile of
//! `limits.attn_cpu_tile` positions at a time. The record of head `h` starts at
//! `recs[(h - h_lo) * rec_stride]`: the value sums `a` without the division by `l`, the maximum
//! score `m`, then the sum `l` of the exponentials shifted by `m`. This is the record layout of
//! the GPU split pass. The fold state belongs to one head, so a head split gives the same records.
//!
//! Attention merge. Per head: `M = max_p m_p`, `w_p = exp(m_p - M)`, `L = sum_p l_p * w_p`, and
//! `out = (sum_p a_p * w_p) * (1 / L)`. Every sum runs in ascending part order from zero, as the
//! GPU merge pass runs it. So one row length gives one result bit for bit at any thread count.
//! The parts are the geometry of `attn_cpu`, a function of `t_len` alone.

const std = @import("std");
const attn_cpu = @import("../core/attn_cpu.zig");
const limits = @import("../core/limits.zig");
const op_requires = @import("../core/op_requires.zig");
const Tensor = @import("../core/tensor.zig").Tensor;

// -- constants --------------------------------------------------------------

const max_head_dim: usize = limits.attn_max_head_dim;
const max_group: usize = limits.attn_max_group;

/// The pairs per block of `ropeHeads`. A wider rope runs in several blocks, and the cosine and
/// sine tables take this size on the stack.
const rope_block_pairs: usize = limits.attn_max_head_dim / 2;

/// The naive variant runs anywhere.
pub const requires: op_requires.OpRequires = .{ .arch = .any };

// -- types ------------------------------------------------------------------

/// Which lanes form a RoPE rotation pair. The angle for pair i is the same either way.
/// Only the memory layout differs.
pub const RopeStyle = enum {
    /// Adjacent pairs (x[2i], x[2i+1]).
    norm,
    /// Half-split pairs (x[i], x[i + rope_dim/2]).
    neox,
};

/// The rope values of one model. `thetas` holds the base angle of each rotated pair, so the
/// rotated width is `2 * thetas.len`. `factors` is the frequency divisor of each pair, or null
/// for the plain frequencies.
pub const RopeTable = struct {
    thetas: []const f32,
    factors: ?[]const f32,
};

// -- api --------------------------------------------------------------------

/// Linear layer: `out = input @ W.T`.
pub fn linearF32(input: []const f32, weight: Tensor, out: []f32) void {
    std.debug.assert(weight.dtype == .f32);
    std.debug.assert(weight.n_dims == 2);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(input.len == in_features);
    std.debug.assert(out.len >= out_features);

    const w = std.mem.bytesAsSlice(f32, weight.data);
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        var sum: f32 = 0;
        var k: usize = 0;
        while (k < in_features) : (k += 1) {
            sum += input[k] * w[j * in_features + k];
        }

        out[j] = sum;
    }

    if (std.debug.runtime_safety) {
        for (out[0..out_features]) |x| {
            std.debug.assert(std.math.isFinite(x));
        }
    }
}

/// Linear layer with Q8_0 weights.
pub fn linearF32Q80(input: []const f32, weight: Tensor, out: []f32) void {
    std.debug.assert(weight.dtype == .q8_0);
    std.debug.assert(weight.n_dims == 2);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(input.len == in_features);
    std.debug.assert(out.len >= out_features);
    std.debug.assert(in_features % 32 == 0);

    const row_bytes = (in_features / 32) * 34;
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        const row_off = j * row_bytes;
        var sum: f32 = 0;
        var blk: usize = 0;
        while (blk < in_features / 32) : (blk += 1) {
            const blk_off = row_off + blk * 34;
            const scale = f16BitsToF32(std.mem.readInt(u16, weight.data[blk_off..][0..2], .little));
            var i: usize = 0;
            while (i < 32) : (i += 1) {
                const v: i8 = @bitCast(weight.data[blk_off + 2 + i]);
                sum += input[blk * 32 + i] * scale * @as(f32, @floatFromInt(v));
            }
        }

        out[j] = sum;
    }

    if (std.debug.runtime_safety) {
        for (out[0..out_features]) |x| {
            std.debug.assert(std.math.isFinite(x));
        }
    }
}

/// Batched linear: `batch` input rows against one f32 weight.
/// Each row is bit-identical to a B=1 `linearF32` call.
pub fn linearF32Batch(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    // Check the shapes.
    std.debug.assert(weight.dtype == .f32);
    std.debug.assert(weight.n_dims == 2);
    std.debug.assert(batch > 0);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(inputs.len == batch * in_features);
    std.debug.assert(out_stride >= out_features);
    std.debug.assert(outs.len >= (batch - 1) * out_stride + out_features);

    // Dot each batch row with each weight row.
    const w = std.mem.bytesAsSlice(f32, weight.data);
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const input = inputs[b * in_features ..][0..in_features];
            var sum: f32 = 0;
            var k: usize = 0;
            while (k < in_features) : (k += 1) {
                sum += input[k] * w[j * in_features + k];
            }

            outs[b * out_stride + j] = sum;
        }
    }

    // Check that the outputs are finite.
    if (std.debug.runtime_safety) {
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            for (outs[b * out_stride ..][0..out_features]) |x| {
                std.debug.assert(std.math.isFinite(x));
            }
        }
    }
}

/// Batched linear with Q8_0 weights.
/// The block walk and accumulation order are those of `linearF32Q80`.
pub fn linearF32Q80Batch(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    std.debug.assert(weight.dtype == .q8_0);
    std.debug.assert(weight.n_dims == 2);
    std.debug.assert(batch > 0);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(inputs.len == batch * in_features);
    std.debug.assert(out_stride >= out_features);
    std.debug.assert(outs.len >= (batch - 1) * out_stride + out_features);
    std.debug.assert(in_features % 32 == 0);

    // Walk the weight rows once, and every batch row against each.
    const row_bytes = (in_features / 32) * 34;
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        const row_off = j * row_bytes;
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            const input = inputs[b * in_features ..][0..in_features];
            var sum: f32 = 0;
            var blk: usize = 0;
            while (blk < in_features / 32) : (blk += 1) {
                const blk_off = row_off + blk * 34;
                const scale = f16BitsToF32(
                    std.mem.readInt(u16, weight.data[blk_off..][0..2], .little),
                );
                var i: usize = 0;
                while (i < 32) : (i += 1) {
                    const v: i8 = @bitCast(weight.data[blk_off + 2 + i]);
                    sum += input[blk * 32 + i] * scale * @as(f32, @floatFromInt(v));
                }
            }

            outs[b * out_stride + j] = sum;
        }
    }

    // Check that every output is finite.
    if (std.debug.runtime_safety) {
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            for (outs[b * out_stride ..][0..out_features]) |x| {
                std.debug.assert(std.math.isFinite(x));
            }
        }
    }
}

/// Per-row batch fallback over any B=1 linear kernel, bit-identical to one call per row.
/// A variant without a batched microkernel exports it over its own kernels.
pub fn linearBatchPerRow(
    comptime kernel: fn ([]const f32, Tensor, []f32) void,
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    std.debug.assert(weight.n_dims == 2);
    std.debug.assert(batch > 0);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(inputs.len == batch * in_features);
    std.debug.assert(out_stride >= out_features);
    std.debug.assert(outs.len >= (batch - 1) * out_stride + out_features);

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        kernel(
            inputs[b * in_features ..][0..in_features],
            weight,
            outs[b * out_stride ..][0..out_features],
        );
    }
}

/// Linear layer with Q4_0 weights.
pub fn linearF32Q40(input: []const f32, weight: Tensor, out: []f32) void {
    std.debug.assert(weight.dtype == .q4_0);
    std.debug.assert(weight.n_dims == 2);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(input.len == in_features);
    std.debug.assert(out.len >= out_features);
    std.debug.assert(in_features % 32 == 0);

    const row_bytes = (in_features / 32) * 18;
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        const row_off = j * row_bytes;
        var sum: f32 = 0;
        var blk: usize = 0;
        while (blk < in_features / 32) : (blk += 1) {
            const blk_off = row_off + blk * 18;
            const scale = f16BitsToF32(std.mem.readInt(u16, weight.data[blk_off..][0..2], .little));
            var i: usize = 0;
            while (i < 16) : (i += 1) {
                const byte = weight.data[blk_off + 2 + i];
                const lo: i32 = @as(i32, @intCast(byte & 0x0F)) - 8;
                const hi: i32 = @as(i32, @intCast(byte >> 4)) - 8;
                sum += input[blk * 32 + i] * scale * @as(f32, @floatFromInt(lo));
                sum += input[blk * 32 + 16 + i] * scale * @as(f32, @floatFromInt(hi));
            }
        }

        out[j] = sum;
    }

    if (std.debug.runtime_safety) {
        for (out[0..out_features]) |x| {
            std.debug.assert(std.math.isFinite(x));
        }
    }
}

/// Batched Q4_0 linear through the per-row fallback, bit-identical to `linearF32Q40`.
pub fn linearF32Q40Batch(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    std.debug.assert(weight.dtype == .q4_0);

    linearBatchPerRow(linearF32Q40, inputs, weight, outs, batch, out_stride);
}

/// Linear layer with Q4_1 weights.
pub fn linearF32Q41(input: []const f32, weight: Tensor, out: []f32) void {
    // Check the shapes.
    std.debug.assert(weight.dtype == .q4_1);
    std.debug.assert(weight.n_dims == 2);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(input.len == in_features);
    std.debug.assert(out.len >= out_features);
    std.debug.assert(in_features % 32 == 0);

    // Dot each weight row, one block at a time.
    const row_bytes = (in_features / 32) * 20;
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        const row_off = j * row_bytes;
        var sum: f32 = 0;
        var blk: usize = 0;
        while (blk < in_features / 32) : (blk += 1) {
            const blk_off = row_off + blk * 20;
            const d = f16BitsToF32(std.mem.readInt(u16, weight.data[blk_off..][0..2], .little));
            const m = f16BitsToF32(
                std.mem.readInt(u16, weight.data[blk_off + 2 ..][0..2], .little),
            );
            var i: usize = 0;
            while (i < 16) : (i += 1) {
                const byte = weight.data[blk_off + 4 + i];
                const lo: f32 = @floatFromInt(byte & 0x0F);
                const hi: f32 = @floatFromInt(byte >> 4);
                sum += input[blk * 32 + i] * (lo * d + m);
                sum += input[blk * 32 + 16 + i] * (hi * d + m);
            }
        }

        out[j] = sum;
    }

    // Check that the outputs are finite.
    if (std.debug.runtime_safety) {
        for (out[0..out_features]) |x| {
            std.debug.assert(std.math.isFinite(x));
        }
    }
}

/// Batched Q4_1 linear via the per-row fallback.
pub fn linearF32Q41Batch(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    std.debug.assert(weight.dtype == .q4_1);

    linearBatchPerRow(linearF32Q41, inputs, weight, outs, batch, out_stride);
}

/// Linear layer with Q6_K weights, with `in / 256` superblocks per row. It allocates nothing.
pub fn linearF32Q6K(input: []const f32, weight: Tensor, out: []f32) void {
    std.debug.assert(weight.dtype == .q6_k);
    std.debug.assert(weight.n_dims == 2);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(input.len == in_features);
    std.debug.assert(out.len >= out_features);
    std.debug.assert(in_features % 256 == 0);

    const n_super = in_features / 256;
    const row_bytes = n_super * 210;
    var j: usize = 0;
    while (j < out_features) : (j += 1) {
        const row_off = j * row_bytes;
        var sum: f32 = 0;
        var sb: usize = 0;
        while (sb < n_super) : (sb += 1) {
            var tmp: [256]f32 = undefined;
            dequantQ6K(weight.data[row_off + sb * 210 ..][0..210], &tmp);
            for (0..256) |i| {
                sum += input[sb * 256 + i] * tmp[i];
            }
        }

        out[j] = sum;
    }

    if (std.debug.runtime_safety) {
        for (out[0..out_features]) |x| {
            std.debug.assert(std.math.isFinite(x));
        }
    }
}

/// Batched Q6_K linear via the per-row fallback.
pub fn linearF32Q6KBatch(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    std.debug.assert(weight.dtype == .q6_k);

    linearBatchPerRow(linearF32Q6K, inputs, weight, outs, batch, out_stride);
}

/// Dequantize a Q8_0 buffer (`src.len % 34 == 0`) into f32 (`dst.len == (src.len/34)*32`).
pub fn dequantQ80(src: []const u8, dst: []f32) void {
    std.debug.assert(src.len % 34 == 0);

    const n_blocks = src.len / 34;
    std.debug.assert(dst.len == n_blocks * 32);
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const block_off = b * 34;
        const scale_bits = std.mem.readInt(u16, src[block_off..][0..2], .little);
        const scale = f16BitsToF32(scale_bits);
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            const v: i8 = @bitCast(src[block_off + 2 + i]);
            dst[b * 32 + i] = scale * @as(f32, @floatFromInt(v));
        }
    }

    if (std.debug.runtime_safety) {
        for (dst) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

pub fn f16BitsToF32(bits: u16) f32 {
    const half: f16 = @bitCast(bits);
    return @floatCast(half);
}

/// Dequantize a Q4_0 buffer (`src.len % 18 == 0`) into f32 (`dst.len == (src.len/18)*32`).
/// The nibble order matches `linearF32Q40`.
pub fn dequantQ40(src: []const u8, dst: []f32) void {
    std.debug.assert(src.len % 18 == 0);

    const n_blocks = src.len / 18;
    std.debug.assert(dst.len == n_blocks * 32);
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const off = b * 18;
        const scale = f16BitsToF32(std.mem.readInt(u16, src[off..][0..2], .little));
        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const byte = src[off + 2 + i];
            const lo: i32 = @as(i32, @intCast(byte & 0x0F)) - 8;
            const hi: i32 = @as(i32, @intCast(byte >> 4)) - 8;
            dst[b * 32 + i] = scale * @as(f32, @floatFromInt(lo));
            dst[b * 32 + 16 + i] = scale * @as(f32, @floatFromInt(hi));
        }
    }

    if (std.debug.runtime_safety) {
        for (dst) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

/// Dequantize a Q4_1 buffer (`src.len % 20 == 0`) into f32 (`dst.len == (src.len/20)*32`).
/// The nibble order matches `linearF32Q41`.
pub fn dequantQ41(src: []const u8, dst: []f32) void {
    std.debug.assert(src.len % 20 == 0);

    const n_blocks = src.len / 20;
    std.debug.assert(dst.len == n_blocks * 32);
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const off = b * 20;
        const d = f16BitsToF32(std.mem.readInt(u16, src[off..][0..2], .little));
        const m = f16BitsToF32(std.mem.readInt(u16, src[off + 2 ..][0..2], .little));
        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const byte = src[off + 4 + i];
            dst[b * 32 + i] = @as(f32, @floatFromInt(byte & 0x0F)) * d + m;
            dst[b * 32 + 16 + i] = @as(f32, @floatFromInt(byte >> 4)) * d + m;
        }
    }

    if (std.debug.runtime_safety) {
        for (dst) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

/// Quantize f32 to Q4_0 (`src.len % 32 == 0`, `dst.len == (src.len/32)*18`), as ggml
/// `quantize_row_q4_0_ref`: `d = max_signed_abs / -8`, value `clamp(floor(x / d + 8.5), 0, 15)`.
/// Load uses it only to requantize the Q6_K lm_head that QuantFactory ships, to keep GPU Q4_0.
pub fn quantizeQ40(src: []const f32, dst: []u8) void {
    std.debug.assert(src.len % 32 == 0);

    const n_blocks = src.len / 32;
    std.debug.assert(dst.len == n_blocks * 18);
    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const base = b * 32;
        var amax: f32 = 0;
        var maxv: f32 = 0;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            const v = src[base + i];
            if (@abs(v) > amax) {
                amax = @abs(v);
                maxv = v;
            }
        }

        const d: f32 = maxv / -8.0;
        const id: f32 = if (d != 0) 1.0 / d else 0.0;
        const off = b * 18;
        const d16: f16 = @floatCast(d);
        std.mem.writeInt(u16, dst[off..][0..2], @bitCast(d16), .little);
        var j: usize = 0;
        while (j < 16) : (j += 1) {
            const x0 = src[base + j] * id;
            const x1 = src[base + 16 + j] * id;
            const xi0: u8 = @intFromFloat(std.math.clamp(@floor(x0 + 8.5), 0.0, 15.0));
            const xi1: u8 = @intFromFloat(std.math.clamp(@floor(x1 + 8.5), 0.0, 15.0));
            dst[off + 2 + j] = xi0 | (xi1 << 4);
        }
    }
}

/// Dequantize a Q6_K buffer into f32 (`src.len % 210 == 0`, `dst.len == (src.len/210)*256`).
/// It mirrors ggml `dequantize_row_q6_K`.
pub fn dequantQ6K(src: []const u8, dst: []f32) void {
    std.debug.assert(src.len % 210 == 0);

    const n_super = src.len / 210;
    std.debug.assert(dst.len == n_super * 256);
    var sb: usize = 0;
    while (sb < n_super) : (sb += 1) {
        const blk = src[sb * 210 ..][0..210];
        const d = f16BitsToF32(std.mem.readInt(u16, blk[208..][0..2], .little));

        // Decode the two 128-element halves. The bases advance ql+=64, qh+=32, sc+=8.
        var half: usize = 0;
        while (half < 2) : (half += 1) {
            const ql_base = half * 64;
            const qh_base = 128 + half * 32;
            const sc_base = 192 + half * 8;
            const y_base = sb * 256 + half * 128;
            var l: usize = 0;
            while (l < 32) : (l += 1) {
                const is = l / 16; // 0 or 1
                const ql0: u8 = blk[ql_base + l];
                const ql32: u8 = blk[ql_base + l + 32];
                const qhl: u8 = blk[qh_base + l];
                const n1: u8 = (ql0 & 0x0F) | (((qhl >> 0) & 3) << 4);
                const n2: u8 = (ql32 & 0x0F) | (((qhl >> 2) & 3) << 4);
                const n3: u8 = (ql0 >> 4) | (((qhl >> 4) & 3) << 4);
                const n4: u8 = (ql32 >> 4) | (((qhl >> 6) & 3) << 4);
                const q1: f32 = @floatFromInt(@as(i32, n1) - 32);
                const q2: f32 = @floatFromInt(@as(i32, n2) - 32);
                const q3: f32 = @floatFromInt(@as(i32, n3) - 32);
                const q4: f32 = @floatFromInt(@as(i32, n4) - 32);
                const s0: f32 = @floatFromInt(@as(i8, @bitCast(blk[sc_base + is + 0])));
                const s2: f32 = @floatFromInt(@as(i8, @bitCast(blk[sc_base + is + 2])));
                const s4: f32 = @floatFromInt(@as(i8, @bitCast(blk[sc_base + is + 4])));
                const s6: f32 = @floatFromInt(@as(i8, @bitCast(blk[sc_base + is + 6])));
                dst[y_base + l + 0] = d * s0 * q1;
                dst[y_base + l + 32] = d * s2 * q2;
                dst[y_base + l + 64] = d * s4 * q3;
                dst[y_base + l + 96] = d * s6 * q4;
            }
        }
    }

    if (std.debug.runtime_safety) {
        for (dst) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

/// Look up `weight[token_id, :]` (length `dim`) into `dst`. A quantized table dequantizes.
/// `shape[0] = dim` and `shape[1] = vocab`.
pub fn embedLookup(weight: Tensor, token_id: u32, dim: usize, dst: []f32) void {
    // Check the shapes.
    std.debug.assert(weight.n_dims == 2);
    std.debug.assert(weight.shape[0] == dim);
    std.debug.assert(token_id < weight.shape[1]);
    std.debug.assert(dst.len >= dim);

    // Copy or dequantize the row of the token.
    switch (weight.dtype) {
        .f32 => {
            const view = std.mem.bytesAsSlice(f32, weight.data);
            const src_start = @as(usize, token_id) * dim;
            @memcpy(dst[0..dim], view[src_start..][0..dim]);
        },
        .q8_0 => {
            std.debug.assert(dim % 32 == 0);

            const row_bytes = (dim / 32) * 34;
            const row_start = @as(usize, token_id) * row_bytes;
            dequantQ80(weight.data[row_start..][0..row_bytes], dst[0..dim]);
        },
        .q4_0 => {
            std.debug.assert(dim % 32 == 0);

            const row_bytes = (dim / 32) * 18;
            const row_start = @as(usize, token_id) * row_bytes;
            dequantQ40(weight.data[row_start..][0..row_bytes], dst[0..dim]);
        },
        .q4_1 => {
            std.debug.assert(dim % 32 == 0);

            const row_bytes = (dim / 32) * 20;
            const row_start = @as(usize, token_id) * row_bytes;
            dequantQ41(weight.data[row_start..][0..row_bytes], dst[0..dim]);
        },
        else => unreachable, // `requireWeights` refuses any other dtype at runner init
    }

    // Check that the row is finite.
    if (std.debug.runtime_safety) {
        for (dst[0..dim]) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

/// y[i] = (x[i] / rms(x)) * w[i],  rms(x) = sqrt(mean(x^2) + eps)
pub fn rmsnorm(x: []const f32, w: []const f32, eps: f32, dst: []f32) void {
    std.debug.assert(x.len == w.len);
    std.debug.assert(dst.len >= x.len);

    var sum_sq: f32 = 0.0;
    for (x) |v| {
        sum_sq += v * v;
    }

    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    for (x, w, dst[0..x.len]) |v, wv, *d| {
        d.* = v * inv_rms * wv;
    }

    if (std.debug.runtime_safety) {
        for (dst[0..x.len]) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

/// dst[i] = silu(gate[i]) * up[i]
pub fn swiglu(gate: []const f32, up: []const f32, dst: []f32) void {
    std.debug.assert(gate.len == up.len);
    std.debug.assert(dst.len >= gate.len);

    for (gate, up, dst[0..gate.len]) |g, u, *d| {
        const e = std.math.exp(-g);
        d.* = (g / (1.0 + e)) * u;
    }

    if (std.debug.runtime_safety) {
        for (dst[0..gate.len]) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

/// dst[i] += x[i]
pub fn residualAdd(x: []const f32, dst: []f32) void {
    std.debug.assert(x.len <= dst.len);

    for (x, dst[0..x.len]) |v, *d| {
        d.* += v;
    }

    if (std.debug.runtime_safety) {
        for (dst[0..x.len]) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

/// In-place RoPE on one head_dim vector at position `pos`. Dims beyond `rope_dim` stay untouched.
/// `freq_factors` is null for classic RoPE.
pub fn ropeInplace(
    x: []f32,
    pos: u32,
    rope_dim: usize,
    freq_base: f32,
    style: RopeStyle,
    freq_factors: ?[]const f32,
) void {
    std.debug.assert(rope_dim <= x.len);
    std.debug.assert(rope_dim % 2 == 0);

    if (freq_factors) |ff| std.debug.assert(ff.len >= rope_dim / 2);

    const half = rope_dim / 2;
    const inv_dim_f: f32 = 1.0 / @as(f32, @floatFromInt(rope_dim));
    const pos_f: f32 = @floatFromInt(pos);
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const exp_v = -2.0 * @as(f32, @floatFromInt(i)) * inv_dim_f;
        const theta = std.math.pow(f32, freq_base, exp_v);
        const ff_v: f32 = if (freq_factors) |ff| ff[i] else 1.0;
        const angle = pos_f * theta / ff_v;
        const cos_v = @cos(angle);
        const sin_v = @sin(angle);
        const ia: usize = if (style == .norm) 2 * i else i;
        const ib: usize = if (style == .norm) 2 * i + 1 else i + half;
        const a = x[ia];
        const b = x[ib];
        x[ia] = a * cos_v - b * sin_v;
        x[ib] = a * sin_v + b * cos_v;
    }

    if (std.debug.runtime_safety) {
        for (x[0..rope_dim]) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

/// Fill the base angle of each rotated pair, `freq_base^(-2i/D)` with `D = 2 * thetas.len`. Each
/// entry holds the value that `ropeInplace` computes for the same pair.
pub fn fillRopeThetas(thetas: []f32, freq_base: f32) void {
    std.debug.assert(thetas.len >= 1);

    const inv_dim_f: f32 = 1.0 / @as(f32, @floatFromInt(2 * thetas.len));
    for (thetas, 0..) |*t, i| {
        const exp_v = -2.0 * @as(f32, @floatFromInt(i)) * inv_dim_f;
        t.* = std.math.pow(f32, freq_base, exp_v);
    }

    if (std.debug.runtime_safety) {
        for (thetas) |t| {
            std.debug.assert(std.math.isFinite(t));
        }
    }
}

/// The RoPE of every head of `q` and `k` for one row at `pos`. It computes the cosine and the sine
/// of each pair once, because the angle does not depend on the head. Each element gets the
/// operations of `ropeInplace`.
pub fn ropeHeads(
    q: []f32,
    k: []f32,
    head_dim: usize,
    pos: u32,
    table: RopeTable,
    style: RopeStyle,
) void {
    // Check the arguments.
    const half = table.thetas.len;
    std.debug.assert(half >= 1);
    std.debug.assert(2 * half <= head_dim);
    std.debug.assert(q.len >= head_dim);
    std.debug.assert(q.len % head_dim == 0);
    std.debug.assert(k.len % head_dim == 0);

    if (table.factors) |ff| std.debug.assert(ff.len >= half);

    // A block of pairs shares its angles across heads, so the tables hold each angle once.
    var cos_tab: [rope_block_pairs]f32 = undefined;
    var sin_tab: [rope_block_pairs]f32 = undefined;
    const pos_f: f32 = @floatFromInt(pos);
    var lo: usize = 0;
    while (lo < half) : (lo += rope_block_pairs) {
        const n = @min(rope_block_pairs, half - lo);
        for (0..n) |j| {
            const ff_v: f32 = if (table.factors) |ff| ff[lo + j] else 1.0;
            const angle = pos_f * table.thetas[lo + j] / ff_v;
            cos_tab[j] = @cos(angle);
            sin_tab[j] = @sin(angle);
        }

        // Rotate the block in every head of q, then of k.
        for ([_][]f32{ q, k }) |xs| {
            var base: usize = 0;
            while (base < xs.len) : (base += head_dim) {
                const x = xs[base..][0..head_dim];
                for (cos_tab[0..n], sin_tab[0..n], lo..) |cos_v, sin_v, i| {
                    const ia: usize = if (style == .norm) 2 * i else i;
                    const ib: usize = if (style == .norm) 2 * i + 1 else i + half;
                    const a = x[ia];
                    const b = x[ib];
                    x[ia] = a * cos_v - b * sin_v;
                    x[ib] = a * sin_v + b * cos_v;
                }
            }
        }
    }

    if (std.debug.runtime_safety) {
        for ([_][]f32{ q, k }) |xs| {
            var base: usize = 0;
            while (base < xs.len) : (base += head_dim) {
                for (xs[base..][0 .. 2 * half]) |v| {
                    std.debug.assert(std.math.isFinite(v));
                }
            }
        }
    }
}

/// One part of a decode attention row: heads `[h_lo, h_hi)` over positions `[t_lo, t_hi)`, one
/// record per head. It walks the heads of a KV head in blocks of `max_group`, so each K and V row
/// widens once per block, and the bits of a head never depend on its block.
pub fn attentionPartTyped(
    comptime E: type,
    q: []const f32,
    h_lo: usize,
    h_hi: usize,
    n_heads: usize,
    head_dim: usize,
    k_cache: []const E,
    v_cache: []const E,
    n_kv_heads: usize,
    t_lo: usize,
    t_hi: usize,
    recs: []f32,
    rec_stride: usize,
) void {
    // Check the arguments.
    comptime std.debug.assert(E == f32 or E == f16);
    const tile: usize = limits.attn_cpu_tile;
    std.debug.assert(h_lo < h_hi);
    std.debug.assert(h_hi <= n_heads);
    std.debug.assert(n_kv_heads >= 1);
    std.debug.assert(n_heads % n_kv_heads == 0);
    std.debug.assert(head_dim >= 1);
    std.debug.assert(q.len >= n_heads * head_dim);
    std.debug.assert(t_lo < t_hi);
    std.debug.assert(t_hi <= limits.max_model_ctx);
    const kv_stride = n_kv_heads * head_dim;
    std.debug.assert(k_cache.len >= t_hi * kv_stride);
    std.debug.assert(v_cache.len >= t_hi * kv_stride);
    const rec_len: usize = attn_cpu.recordElems(@intCast(head_dim));
    std.debug.assert(rec_stride >= rec_len);
    std.debug.assert(recs.len >= (h_hi - h_lo - 1) * rec_stride + rec_len);
    // The tile loop below runs at most this many times.
    std.debug.assert((t_hi - t_lo + tile - 1) / tile <= limits.max_model_ctx / tile);

    // Reserve the scratch. Row g of `scores` holds block head g over one tile, then its
    // shifted exponentials.
    const group_size = n_heads / n_kv_heads;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var row: [if (E == f32) 0 else max_head_dim]f32 = undefined;
    var scores: [max_group][limits.attn_cpu_tile]f32 = undefined;
    var s_acc: [max_group]f32 = undefined;
    var m_run: [max_group]f32 = undefined;
    var l_run: [max_group]f32 = undefined;

    // Walk each KV head, its blocks of heads, and the tiles of the part.
    var kv_h = h_lo / group_size;
    const kv_h_end = (h_hi - 1) / group_size + 1;
    while (kv_h < kv_h_end) : (kv_h += 1) {
        const g_lo = @max(h_lo, kv_h * group_size);
        const g_hi = @min(h_hi, (kv_h + 1) * group_size);
        var blk_lo = g_lo;
        while (blk_lo < g_hi) : (blk_lo += max_group) {
            const blk_hi = @min(g_hi, blk_lo + max_group);
            const blk_n = blk_hi - blk_lo;

            var tile_lo = t_lo;
            while (tile_lo < t_hi) : (tile_lo += tile) {
                const tile_hi = @min(t_hi, tile_lo + tile);
                const tile_n = tile_hi - tile_lo;

                // Score every position of the tile.
                var t = tile_lo;
                while (t < tile_hi) : (t += 1) {
                    const k_off = t * kv_stride + kv_h * head_dim;
                    @memset(s_acc[0..blk_n], 0);
                    var d0: usize = 0;
                    while (d0 < head_dim) : (d0 += max_head_dim) {
                        const dn = @min(head_dim - d0, max_head_dim);
                        const k_row: []const f32 = if (E == f32)
                            k_cache[k_off + d0 ..][0..dn]
                        else blk: {
                            widenRow(E, k_cache[k_off + d0 ..][0..dn], row[0..dn]);
                            break :blk row[0..dn];
                        };

                        var h = blk_lo;
                        while (h < blk_hi) : (h += 1) {
                            const q_off = h * head_dim + d0;
                            // The chunk continues the accumulator, so the sums match the whole row.
                            var s: f32 = s_acc[h - blk_lo];
                            var d: usize = 0;
                            while (d < dn) : (d += 1) {
                                s += q[q_off + d] * k_row[d];
                            }

                            s_acc[h - blk_lo] = s;
                        }
                    }

                    var h = blk_lo;
                    while (h < blk_hi) : (h += 1) {
                        scores[h - blk_lo][t - tile_lo] = s_acc[h - blk_lo] * scale;
                    }
                }

                // Fold the scores into the running state of each head.
                var h = blk_lo;
                while (h < blk_hi) : (h += 1) {
                    const g = h - blk_lo;
                    const e_row = scores[g][0..tile_n];
                    const acc = recs[(h - h_lo) * rec_stride ..][0..head_dim];
                    if (std.debug.runtime_safety) {
                        for (e_row) |s| {
                            std.debug.assert(std.math.isFinite(s));
                        }
                    }

                    var m_tile = e_row[0];
                    for (e_row[1..]) |s| {
                        if (s > m_tile) {
                            m_tile = s;
                        }
                    }

                    if (tile_lo == t_lo) {
                        m_run[g] = m_tile;
                        l_run[g] = 0;
                        @memset(acc, 0);
                    } else if (m_tile > m_run[g]) {
                        const c = std.math.exp(m_run[g] - m_tile);
                        for (acc) |*x| {
                            x.* *= c;
                        }

                        l_run[g] *= c;
                        m_run[g] = m_tile;
                    }

                    var l = l_run[g];
                    for (e_row) |*s| {
                        s.* = std.math.exp(s.* - m_run[g]);
                        l += s.*;
                    }

                    // The term at the maximum adds exp(0) = 1, and no term is negative. A
                    // non-finite score breaks it, so only a safety build checks it.
                    if (std.debug.runtime_safety) std.debug.assert(l >= 1);
                    l_run[g] = l;
                }

                // Add the weighted V rows to the value sums.
                t = tile_lo;
                while (t < tile_hi) : (t += 1) {
                    const v_off = t * kv_stride + kv_h * head_dim;
                    var d0: usize = 0;
                    while (d0 < head_dim) : (d0 += max_head_dim) {
                        const dn = @min(head_dim - d0, max_head_dim);
                        const v_row: []const f32 = if (E == f32)
                            v_cache[v_off + d0 ..][0..dn]
                        else blk: {
                            widenRow(E, v_cache[v_off + d0 ..][0..dn], row[0..dn]);
                            break :blk row[0..dn];
                        };

                        h = blk_lo;
                        while (h < blk_hi) : (h += 1) {
                            const w = scores[h - blk_lo][t - tile_lo];
                            const acc_off = (h - h_lo) * rec_stride + d0;
                            var d: usize = 0;
                            while (d < dn) : (d += 1) {
                                recs[acc_off + d] += w * v_row[d];
                            }
                        }
                    }
                }
            }

            // Store the running maximum and sum of each head.
            var h = blk_lo;
            while (h < blk_hi) : (h += 1) {
                const rec = recs[(h - h_lo) * rec_stride ..][0..rec_len];
                rec[head_dim] = m_run[h - blk_lo];
                rec[head_dim + 1] = l_run[h - blk_lo];
            }
        }
    }

    // Check that every record is finite.
    if (std.debug.runtime_safety) {
        var h = h_lo;
        while (h < h_hi) : (h += 1) {
            for (recs[(h - h_lo) * rec_stride ..][0..rec_len]) |x| {
                std.debug.assert(std.math.isFinite(x));
            }
        }
    }
}

/// Combine the first `n_parts` part records of every query head into the attention output.
/// The order of the sums fixes the bits.
pub fn attentionMerge(
    parts_buf: []const f32,
    n_parts: usize,
    n_heads: usize,
    head_dim: usize,
    out: []f32,
) void {
    std.debug.assert(n_parts >= 1);
    std.debug.assert(n_parts <= limits.attn_cpu_parts);

    const n_heads32: u32 = @intCast(n_heads);
    const head_dim32: u32 = @intCast(head_dim);
    std.debug.assert(parts_buf.len >= attn_cpu.partialLen(n_heads32, head_dim32));
    std.debug.assert(out.len >= n_heads * head_dim);

    var offs: [limits.attn_cpu_parts]usize = undefined;
    var w: [limits.attn_cpu_parts]f32 = undefined;
    var h: u32 = 0;
    while (h < n_heads32) : (h += 1) {
        for (offs[0..n_parts], 0..) |*off, p| {
            off.* = attn_cpu.recordOffset(n_heads32, head_dim32, h, @intCast(p));
        }

        // Find the maximum score over the parts.
        var big_m = parts_buf[offs[0] + head_dim];
        for (offs[1..n_parts]) |off| {
            if (parts_buf[off + head_dim] > big_m) {
                big_m = parts_buf[off + head_dim];
            }
        }

        // Weight the sums of the parts.
        var big_l: f32 = 0;
        for (offs[0..n_parts], w[0..n_parts]) |off, *w_p| {
            w_p.* = std.math.exp(parts_buf[off + head_dim] - big_m);
            big_l += parts_buf[off + head_dim + 1] * w_p.*;
        }

        // The part at the maximum adds its own sum, at least 1, at weight 1.
        std.debug.assert(big_l >= 1);
        // Weight the value sums and divide by the total.
        const inv_l = 1.0 / big_l;
        const out_h = out[@as(usize, h) * head_dim ..][0..head_dim];
        for (out_h, 0..) |*o, d| {
            var acc: f32 = 0;
            for (offs[0..n_parts], w[0..n_parts]) |off, w_p| {
                acc += parts_buf[off + d] * w_p;
            }

            o.* = acc * inv_l;
        }
    }

    if (std.debug.runtime_safety) {
        for (out[0 .. n_heads * head_dim]) |x| {
            std.debug.assert(std.math.isFinite(x));
        }
    }
}

/// The attention row of every query head, serial: each part in order, then `attentionMerge`.
/// `parts_buf` holds at least `attn_cpu.partialLen(n_heads, head_dim)` f32s. The row writes
/// every record it reads.
pub fn attentionRowTyped(
    comptime E: type,
    q: []const f32,
    n_heads: usize,
    head_dim: usize,
    k_cache: []const E,
    v_cache: []const E,
    n_kv_heads: usize,
    t_len: usize,
    parts_buf: []f32,
    out: []f32,
) void {
    attentionRowWith(
        @This(),
        E,
        q,
        n_heads,
        head_dim,
        k_cache,
        v_cache,
        n_kv_heads,
        t_len,
        parts_buf,
        out,
    );
}

/// The serial attention row over the part kernel of `Kernels`, a namespace that declares
/// `attentionPartTyped`. Every CPU backend runs this one row, so the part order stays one.
pub fn attentionRowWith(
    comptime Kernels: type,
    comptime E: type,
    q: []const f32,
    n_heads: usize,
    head_dim: usize,
    k_cache: []const E,
    v_cache: []const E,
    n_kv_heads: usize,
    t_len: usize,
    parts_buf: []f32,
    out: []f32,
) void {
    std.debug.assert(t_len >= 1);
    std.debug.assert(t_len <= limits.max_model_ctx);

    const n_heads32: u32 = @intCast(n_heads);
    const head_dim32: u32 = @intCast(head_dim);
    const t_len32: u32 = @intCast(t_len);
    std.debug.assert(parts_buf.len >= attn_cpu.partialLen(n_heads32, head_dim32));
    std.debug.assert(out.len >= n_heads * head_dim);

    const n_parts = attn_cpu.partCount(t_len32);
    const rec_stride = attn_cpu.recordStride(head_dim32);
    var p: u32 = 0;
    while (p < n_parts) : (p += 1) {
        const part = attn_cpu.partRange(t_len32, p);
        const rec_lo = attn_cpu.recordOffset(n_heads32, head_dim32, 0, p);
        // The part kernel finds head h at `h * rec_stride` past head 0, where the merge reads it.
        std.debug.assert(attn_cpu.recordOffset(n_heads32, head_dim32, n_heads32 - 1, p) ==
            rec_lo + (n_heads - 1) * rec_stride);
        Kernels.attentionPartTyped(
            E,
            q,
            0,
            n_heads,
            n_heads,
            head_dim,
            k_cache,
            v_cache,
            n_kv_heads,
            part.t_lo,
            part.t_hi,
            parts_buf[rec_lo..],
            rec_stride,
        );
    }

    attentionMerge(parts_buf, n_parts, n_heads, head_dim, out);
}

// -- helpers ----------------------------------------------------------------

/// Expand one storage row into f32 with packed 8-lane converts, because a scalar widen in an
/// arithmetic loop costs a `vpinsrw` and a single-lane `vcvtph2ps` per element. `noinline` is
/// load-bearing: inlined, the optimizer forwards the stores and scalarizes the converts again.
noinline fn widenRow(comptime E: type, noalias src: []const E, noalias dst: []f32) void {
    comptime std.debug.assert(E == f16);
    std.debug.assert(src.len == dst.len);
    const W = 8;
    var i: usize = 0;
    while (i + W <= src.len) : (i += W) {
        const v: @Vector(W, E) = src[i..][0..W].*;
        const w: @Vector(W, f32) = @floatCast(v);
        dst[i..][0..W].* = w;
    }

    while (i < src.len) : (i += 1) {
        dst[i] = @floatCast(src[i]);
    }
}

// -- tests ------------------------------------------------------------------

// Build a synthetic Q8_0 weight with scale 1.0. The i8 ramp varies per (row, element).
fn fillTestQ80(w_data: []u8, in_features: usize, out_features: usize) void {
    const row_bytes = (in_features / 32) * 34;
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
}

test "dequant Q8_0 single block, scale 1.0" {
    // One block: scale = 1.0 (f16 bits = 0x3c00), values 0..31.
    var src: [34]u8 = undefined;
    std.mem.writeInt(u16, src[0..2], 0x3c00, .little);
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        src[2 + i] = @intCast(i);
    }

    var dst: [32]f32 = undefined;
    dequantQ80(&src, &dst);
    for (0..32) |k| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(k)), dst[k]);
    }
}

test "Q4_0 round-trip: integer ramp in [-8,7] is exact" {
    // maxabs = 8 (the -8) -> d = -8/-8 = 1.0, so every integer in [-8,7]
    // lands on a nibble boundary and round-trips exactly.
    var src: [32]f32 = undefined;
    for (0..32) |i| {
        src[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 16)) - 8));
    }

    var packed_bytes: [18]u8 = undefined;
    quantizeQ40(&src, &packed_bytes);
    var dst: [32]f32 = undefined;
    dequantQ40(&packed_bytes, &dst);
    for (0..32) |k| {
        try std.testing.expectEqual(src[k], dst[k]);
    }
}

test "Q4_1 dequant: nibble*d + m, low/high nibble split" {
    // d = 1.0, m = 2.0. Byte 0 packs low nibble 3 (element 0) and high nibble 5
    // (element 16). The remaining nibbles are 0, decoding to m exactly.
    var src = [_]u8{0} ** 20;
    std.mem.writeInt(u16, src[0..2], 0x3c00, .little); // d = 1.0
    std.mem.writeInt(u16, src[2..4], 0x4000, .little); // m = 2.0
    src[4] = 0x53; // low nibble 3, high nibble 5
    var dst: [32]f32 = undefined;
    dequantQ41(&src, &dst);
    try std.testing.expectEqual(@as(f32, 5), dst[0]); // 3*1 + 2
    try std.testing.expectEqual(@as(f32, 7), dst[16]); // 5*1 + 2
    try std.testing.expectEqual(@as(f32, 2), dst[1]); // 0*1 + 2
    try std.testing.expectEqual(@as(f32, 2), dst[17]); // 0*1 + 2
}

test "linearF32Q41 matches dequantize-then-dot" {
    const gpa = std.testing.allocator;
    const in_features: usize = 64; // two blocks per row
    const out_features: usize = 3;
    const row_bytes = (in_features / 32) * 20;

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);

    for (0..out_features) |j| {
        for (0..in_features / 32) |blk| {
            const off = j * row_bytes + blk * 20;
            std.mem.writeInt(u16, w_data[off..][0..2], 0x3800, .little); // d = 0.5
            std.mem.writeInt(u16, w_data[off + 2 ..][0..2], 0xbc00, .little); // m = -1.0
            for (0..16) |i| {
                const lo: u8 = @intCast((i + j + blk) % 16);
                const hi: u8 = @intCast((2 * i + j) % 16);
                w_data[off + 4 + i] = lo | (hi << 4);
            }
        }
    }

    const w: Tensor = .{
        .dtype = .q4_1,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = w_data,
    };

    const input = try gpa.alloc(f32, in_features);
    defer gpa.free(input);

    for (input, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    var got: [out_features]f32 = undefined;
    linearF32Q41(input, w, &got);

    var row_f32: [in_features]f32 = undefined;
    for (0..out_features) |j| {
        dequantQ41(w_data[j * row_bytes ..][0..row_bytes], &row_f32);
        var sum: f32 = 0;
        for (0..in_features) |i| {
            sum += input[i] * row_f32[i];
        }

        try std.testing.expectApproxEqAbs(sum, got[j], 1e-5);
    }
}

test "Q6_K dequant: bit assembly, -32 bias, and per-group scale" {
    // One superblock, d = 1.0. All zero except scales[0]=2 and the first
    // ql/qh bytes, so a handful of decoded values is predictable by hand.
    var src = [_]u8{0} ** 210;
    std.mem.writeInt(u16, src[208..210], 0x3c00, .little); // d = 1.0
    for (192..208) |i| {
        src[i] = 1; // scales all 1...
    }

    src[192] = 2; //              ...except scales[0] = 2
    src[0] = 0x05; // ql[0]: low nibble 5
    src[128] = 0x03; // qh[0]: low 2 bits = 3 -> high bits of element 0
    var dst: [256]f32 = undefined;
    dequantQ6K(&src, &dst);
    // element 0: n1 = 5 | (3<<4) = 53 -> q = 21. Scale group 0 (=2) -> 42.
    try std.testing.expectEqual(@as(f32, 42), dst[0]);
    // element 1: all-zero nibbles -> q = -32. Same scale group 0 (=2) -> -64.
    try std.testing.expectEqual(@as(f32, -64), dst[1]);
    // element 32 (n2 slot), 64 (n3 slot): zero nibbles, scale 1 -> -32.
    try std.testing.expectEqual(@as(f32, -32), dst[32]);
    try std.testing.expectEqual(@as(f32, -32), dst[64]);
    // second 128-element half is untouched -> -32.
    try std.testing.expectEqual(@as(f32, -32), dst[128]);
}

test "linearF32Q6K matches dequantize-then-dot" {
    const gpa = std.testing.allocator;
    const in_features: usize = 256; // one superblock per row
    const out_features: usize = 3;
    const row_bytes = (in_features / 256) * 210;

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);

    for (0..out_features) |j| {
        const off = j * row_bytes;
        for (0..210) |b| {
            w_data[off + b] = @intCast((b * 7 + j * 13) % 251);
        }

        std.mem.writeInt(u16, w_data[off + 208 ..][0..2], 0x3000, .little); // d = 0.125
    }

    const w: Tensor = .{
        .dtype = .q6_k,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = w_data,
    };

    const input = try gpa.alloc(f32, in_features);
    defer gpa.free(input);

    for (input, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.03);
    }

    var got: [out_features]f32 = undefined;
    linearF32Q6K(input, w, &got);

    var row_f32: [in_features]f32 = undefined;
    for (0..out_features) |j| {
        dequantQ6K(w_data[j * row_bytes ..][0..row_bytes], &row_f32);
        var sum: f32 = 0;
        for (0..in_features) |i| {
            sum += input[i] * row_f32[i];
        }

        try std.testing.expectApproxEqAbs(sum, got[j], 1e-4);
    }
}

test "rmsnorm: identity weight, zero eps recovers unit-norm scaling" {
    const x = [_]f32{ 1, 2, 3, 4 };
    const w = [_]f32{ 1, 1, 1, 1 };
    var out: [4]f32 = undefined;
    rmsnorm(&x, &w, 0.0, &out);
    // mean(x^2) = (1+4+9+16)/4 = 7.5. rms = sqrt(7.5)
    const rms = @sqrt(@as(f32, 7.5));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0) / rms, out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0) / rms, out[1], 1e-6);
}

test "swiglu carries silu: silu(0) = 0 and silu(1) ~ 0.7311" {
    const gate = [_]f32{ 0.0, 1.0 };
    const up = [_]f32{ 1.0, 1.0 };
    var dst: [2]f32 = undefined;
    swiglu(&gate, &up, &dst);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dst[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), dst[1], 1e-5);
}

test "rope pos=0 is identity" {
    var x = [_]f32{ 1, 2, 3, 4 };
    const orig = x;
    ropeInplace(&x, 0, 4, 10000.0, .neox, null);
    for (x, orig) |a, b| {
        try std.testing.expectApproxEqAbs(b, a, 1e-6);
    }

    ropeInplace(&x, 0, 4, 10000.0, .norm, null);
    for (x, orig) |a, b| {
        try std.testing.expectApproxEqAbs(b, a, 1e-6);
    }
}

test "rope styles rotate the documented pairs" {
    // At pos=1 and rope_dim=4, pair 0 turns by 1.0. A one-hot input shows which lane moves.
    // The norm pairs are (x0,x1) and (x2,x3). The neox pairs are (x0,x2) and (x1,x3).
    var x_norm = [_]f32{ 1, 0, 0, 0 };
    ropeInplace(&x_norm, 1, 4, 10000.0, .norm, null);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), x_norm[0], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), x_norm[1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), x_norm[2]);

    var x_neox = [_]f32{ 1, 0, 0, 0 };
    ropeInplace(&x_neox, 1, 4, 10000.0, .neox, null);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), x_neox[0], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), x_neox[1]);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), x_neox[2], 1e-6);
}

test "rope freq_factors divide the rotation angle" {
    // factor 2 on pair 0 halves the angle: cos/sin(0.5) instead of (1.0).
    var x = [_]f32{ 1, 0, 0, 0 };
    const ff = [_]f32{ 2.0, 1.0 };
    ropeInplace(&x, 1, 4, 10000.0, .norm, &ff);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 0.5)), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 0.5)), x[1], 1e-6);
}

/// One case of `expectRopeHeadsMatch`. The check rotates seeded random heads with `ropeHeads` and
/// with `ropeInplace` per head, then compares bits.
const RopeCase = struct {
    head_dim: usize,
    n_q: usize,
    n_k: usize,
    pos: u32,
    table: RopeTable,
    freq_base: f32,
    style: RopeStyle,
};

fn expectRopeHeadsMatch(rc: RopeCase, rng: std.Random) !void {
    const head_dim = rc.head_dim;
    const n_q = rc.n_q;
    const n_k = rc.n_k;
    const pos = rc.pos;
    const table = rc.table;
    const freq_base = rc.freq_base;
    const style = rc.style;
    var q: [9 * 256]f32 = undefined;
    var k: [3 * 256]f32 = undefined;
    var q_ref: [9 * 256]f32 = undefined;
    var k_ref: [3 * 256]f32 = undefined;
    for (q[0 .. n_q * head_dim]) |*v| {
        v.* = rng.float(f32) * 4.0 - 2.0;
    }

    for (k[0 .. n_k * head_dim]) |*v| {
        v.* = rng.float(f32) * 4.0 - 2.0;
    }

    @memcpy(q_ref[0 .. n_q * head_dim], q[0 .. n_q * head_dim]);
    @memcpy(k_ref[0 .. n_k * head_dim], k[0 .. n_k * head_dim]);
    const rope_dim = 2 * table.thetas.len;
    for (0..n_q) |h| {
        const x = q_ref[h * head_dim ..][0..head_dim];
        ropeInplace(x, pos, rope_dim, freq_base, style, table.factors);
    }

    for (0..n_k) |h| {
        const x = k_ref[h * head_dim ..][0..head_dim];
        ropeInplace(x, pos, rope_dim, freq_base, style, table.factors);
    }

    ropeHeads(q[0 .. n_q * head_dim], k[0 .. n_k * head_dim], head_dim, pos, table, style);
    for (q[0 .. n_q * head_dim], q_ref[0 .. n_q * head_dim]) |a, b| {
        try std.testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
    }

    for (k[0 .. n_k * head_dim], k_ref[0 .. n_k * head_dim]) |a, b| {
        try std.testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
    }
}

test "ropeHeads is bit-identical to ropeInplace per head" {
    const Case = struct { head_dim: usize, rope_dim: usize, n_q: usize, n_k: usize };
    const cases = [_]Case{
        .{ .head_dim = 64, .rope_dim = 64, .n_q = 9, .n_k = 3 },
        .{ .head_dim = 256, .rope_dim = 64, .n_q = 8, .n_k = 2 },
        .{ .head_dim = 640, .rope_dim = 576, .n_q = 2, .n_k = 1 },
    };
    const positions = [_]u32{ 0, 1, 37, 8191, 100000 };
    const bases = [_]f32{ 10000.0, 1000000.0 };
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const widest_rope_pairs = 288;
    var thetas: [widest_rope_pairs]f32 = undefined;
    var factors: [widest_rope_pairs]f32 = undefined;
    for (&factors, 0..) |*f, i| {
        f.* = 1.0 + @as(f32, @floatFromInt(i)) / 7.0;
    }

    for (cases) |cs| {
        const pairs = cs.rope_dim / 2;
        for (bases) |base| {
            fillRopeThetas(thetas[0..pairs], base);
            for ([_]?[]const f32{ null, factors[0..pairs] }) |ff| {
                const table: RopeTable = .{ .thetas = thetas[0..pairs], .factors = ff };
                for ([_]RopeStyle{ .norm, .neox }) |style| {
                    for (positions) |pos| {
                        try expectRopeHeadsMatch(.{
                            .head_dim = cs.head_dim,
                            .n_q = cs.n_q,
                            .n_k = cs.n_k,
                            .pos = pos,
                            .table = table,
                            .freq_base = base,
                            .style = style,
                        }, prng.random());
                    }
                }
            }
        }
    }
}

test "linearF32Batch is bit-identical to per-row linearF32 (strided outs)" {
    const gpa = std.testing.allocator;
    const in_features: usize = 64;
    const out_features: usize = 7;
    const batch: usize = 3;
    const out_stride: usize = 9; // > out_features: exercise the stride path

    const inputs = try gpa.alloc(f32, batch * in_features);
    defer gpa.free(inputs);

    const w_data = try gpa.alignedAlloc(u8, .@"32", in_features * out_features * @sizeOf(f32));
    defer gpa.free(w_data);

    const outs_batch = try gpa.alloc(f32, (batch - 1) * out_stride + out_features);
    defer gpa.free(outs_batch);

    const out_single = try gpa.alloc(f32, out_features);
    defer gpa.free(out_single);

    for (inputs, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.05);
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

    linearF32Batch(inputs, w, outs_batch, batch, out_stride);
    for (0..batch) |b| {
        linearF32(inputs[b * in_features ..][0..in_features], w, out_single);
        for (out_single, outs_batch[b * out_stride ..][0..out_features]) |s, g| {
            try std.testing.expectEqual(@as(u32, @bitCast(s)), @as(u32, @bitCast(g)));
        }
    }
}

test "linearF32Q80Batch is bit-identical to per-row linearF32Q80 (strided outs)" {
    const gpa = std.testing.allocator;
    const in_features: usize = 96;
    const out_features: usize = 11;
    const batch: usize = 4;
    const out_stride: usize = 13;
    const row_bytes = (in_features / 32) * 34;

    const inputs = try gpa.alloc(f32, batch * in_features);
    defer gpa.free(inputs);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);

    const outs_batch = try gpa.alloc(f32, (batch - 1) * out_stride + out_features);
    defer gpa.free(outs_batch);

    const out_single = try gpa.alloc(f32, out_features);
    defer gpa.free(out_single);

    for (inputs, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.05);
    }

    fillTestQ80(w_data, in_features, out_features);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = w_data,
    };

    linearF32Q80Batch(inputs, w, outs_batch, batch, out_stride);
    for (0..batch) |b| {
        linearF32Q80(inputs[b * in_features ..][0..in_features], w, out_single);
        for (out_single, outs_batch[b * out_stride ..][0..out_features]) |s, g| {
            try std.testing.expectEqual(@as(u32, @bitCast(s)), @as(u32, @bitCast(g)));
        }
    }
}

test "linearBatchPerRow over linearF32Q80 is bit-identical to per-row calls" {
    const gpa = std.testing.allocator;
    const in_features: usize = 64;
    const out_features: usize = 5;
    const batch: usize = 2;
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
        x.* = @cos(@as(f32, @floatFromInt(i)) * 0.02);
    }

    fillTestQ80(w_data, in_features, out_features);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = w_data,
    };

    linearBatchPerRow(linearF32Q80, inputs, w, outs, batch, out_features);
    for (0..batch) |b| {
        linearF32Q80(inputs[b * in_features ..][0..in_features], w, out_single);
        for (out_single, outs[b * out_features ..][0..out_features]) |s, g| {
            try std.testing.expectEqual(@as(u32, @bitCast(s)), @as(u32, @bitCast(g)));
        }
    }
}

test "attention at f16 tracks the f32 reference and repeats bitwise" {
    const n_heads: usize = 2;
    const n_kv_heads: usize = 1; // group_size 2 exercises the GQA share
    const head_dim: usize = 4;
    const t_len: usize = 4;
    const kv_len = t_len * n_kv_heads * head_dim;

    var prng = std.Random.DefaultPrng.init(0x5EEDA77);
    const r = prng.random();
    var q: [n_heads * head_dim]f32 = undefined;
    var k32: [kv_len]f32 = undefined;
    var v32: [kv_len]f32 = undefined;
    for (&q) |*x| {
        x.* = r.float(f32) * 2.0 - 1.0;
    }

    for (&k32) |*x| {
        x.* = r.float(f32) * 2.0 - 1.0;
    }

    for (&v32) |*x| {
        x.* = r.float(f32) * 2.0 - 1.0;
    }

    // The f16 run reads the same values at storage precision, so the measured
    // gap is the element type alone.
    var k16: [kv_len]f16 = undefined;
    var v16: [kv_len]f16 = undefined;
    for (k32, 0..) |x, i| {
        k16[i] = @floatCast(x);
    }

    for (v32, 0..) |x, i| {
        v16[i] = @floatCast(x);
    }

    var parts: [attn_cpu.partialLen(n_heads, head_dim)]f32 = undefined;
    var out32: [n_heads * head_dim]f32 = undefined;
    var out16: [n_heads * head_dim]f32 = undefined;
    var out16_repeat: [n_heads * head_dim]f32 = undefined;
    attentionRowTyped(f32, &q, n_heads, head_dim, &k32, &v32, n_kv_heads, t_len, &parts, &out32);
    attentionRowTyped(f16, &q, n_heads, head_dim, &k16, &v16, n_kv_heads, t_len, &parts, &out16);
    attentionRowTyped(
        f16,
        &q,
        n_heads,
        head_dim,
        &k16,
        &v16,
        n_kv_heads,
        t_len,
        &parts,
        &out16_repeat,
    );

    var rel_max: f32 = 0;
    for (out32, out16) |a, b| {
        const denom = @max(@abs(a), 1.0e-2); // the outputs are averages of unit-scale values
        rel_max = @max(rel_max, @abs(a - b) / denom);
    }

    // This fixture measures 7.4e-4, so a regression must grow the error tenfold to trip.
    try std.testing.expect(rel_max < 1.0e-2);

    // Same inputs, same order, same bits: the narrower store does not cost
    // determinism.
    try std.testing.expectEqualSlices(f32, &out16, &out16_repeat);
}

const attention_ref = @import("attention_ref.zig");
const AttnShape = attention_ref.AttnShape;
const Lead = attention_ref.Lead;
const cpu_tile = attention_ref.cpu_tile;
const fillAttention = attention_ref.fillAttention;

fn expectRowNearF64(
    comptime E: type,
    r: std.Random,
    shape: AttnShape,
    t_len: usize,
    leads: []const Lead,
    with_oracle: bool,
) !void {
    try attention_ref.expectRowNearF64(@This(), E, r, shape, t_len, leads, with_oracle);
}

fn expectPartMatchesSingleHeads(
    comptime E: type,
    r: std.Random,
    shape: AttnShape,
    leads: []const Lead,
    sink_t: usize,
    t_lo: usize,
    t_hi: usize,
    h_lo: usize,
    h_hi: usize,
) !void {
    try attention_ref.expectPartMatchesSingleHeads(
        @This(),
        E,
        r,
        shape,
        leads,
        sink_t,
        t_lo,
        t_hi,
        h_lo,
        h_hi,
    );
}

test "attention row: the part row tracks the f64 reference across tile and part edges" {
    const T = cpu_tile;
    const depths = [_]usize{ 1, 7, T - 1, T, T + 1, 2 * T, 16 * T, 16 * T + 1, 4 * T + 3 };
    var prng = std.Random.DefaultPrng.init(0xA770_0001);
    const r = prng.random();
    // The deep rows take few and narrow heads, so the test stays short in Debug.
    // The widest head spans 3 tiles in 3 parts, so it crosses two tile and two part edges.
    inline for (.{ f32, f16 }) |E| {
        for (depths) |t_len| {
            try expectRowNearF64(
                E,
                r,
                .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 8 },
                t_len,
                &.{},
                true,
            );
        }

        // The widest registered head, and a group of 3 with an odd head count.
        try expectRowNearF64(
            E,
            r,
            .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 256 },
            2 * T + 1,
            &.{},
            true,
        );
        try expectRowNearF64(
            E,
            r,
            .{ .n_heads = 3, .n_kv_heads = 1, .head_dim = 8 },
            16 * T + 1,
            &.{},
            true,
        );
    }
}

test "attention row: a tail head width and a head over the stack row track the f64 reference" {
    // At width 20 the f16 widening runs two 8-lane chunks and a tail of 4. At
    // width 300 the dot walks the stack row in two chunks, and the second
    // chunk continues the accumulator.
    var prng = std.Random.DefaultPrng.init(0xA770_0008);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        try expectRowNearF64(
            E,
            r,
            .{ .n_heads = 3, .n_kv_heads = 1, .head_dim = 20 },
            cpu_tile + 5,
            &.{},
            true,
        );
        try expectRowNearF64(
            E,
            r,
            .{ .n_heads = 1, .n_kv_heads = 1, .head_dim = 300 },
            cpu_tile + 5,
            &.{},
            true,
        );
    }
}

test "attention row: eight seeded depths up to 8 tiles track the f64 reference" {
    const shapes = [_]AttnShape{
        .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 64 },
        .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 256 },
        .{ .n_heads = 9, .n_kv_heads = 3, .head_dim = 64 },
    };
    var prng = std.Random.DefaultPrng.init(0xA770_0002);
    const r = prng.random();
    // The shape cycles by 3 and the dtype by 2, so the 8 cases reach all 6 pairs.
    for (0..8) |i| {
        const t_len = 1 + r.uintLessThan(usize, 8 * cpu_tile);
        if (i % 2 == 0) {
            try expectRowNearF64(f32, r, shapes[i % 3], t_len, &.{}, false);
        } else {
            try expectRowNearF64(f16, r, shapes[i % 3], t_len, &.{}, false);
        }
    }
}

test "attention row: a sink, a ramp, and scores near 100 track the f64 reference" {
    // 34 tiles over 16 parts, so every part folds two or three tiles. The last
    // tile is short, and it still holds enough of the ramp to raise the maximum.
    const t_len = 34 * cpu_tile - 5;
    const shape: AttnShape = .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 8 };
    var prng = std.Random.DefaultPrng.init(0xA770_0003);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        for ([_]Lead{ .sink, .ramp, .large }) |lead| {
            const leads = [_]Lead{lead} ** shape.n_heads;
            try expectRowNearF64(E, r, shape, t_len, &leads, true);
        }
    }
}

test "attention part: a head range gives each head the records of a single-head call" {
    const T = cpu_tile;
    const shape: AttnShape = .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 8 };
    // The ramp heads peak in the last tile, and the sink heads peak in the
    // first tile of the part.
    const leads = [_]Lead{ .ramp, .sink, .none, .large, .sink, .none, .ramp, .large };
    var prng = std.Random.DefaultPrng.init(0xA770_0004);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        try expectPartMatchesSingleHeads(E, r, shape, &leads, T + 3, T, 5 * T + 17, 0, 8);
        // The range cuts both groups.
        try expectPartMatchesSingleHeads(E, r, shape, &leads, T + 3, T, 5 * T + 17, 1, 7);
    }
}

test "attention part: 10 heads in blocks of 8 and 2 give the records of single-head calls" {
    const shape: AttnShape = .{ .n_heads = 20, .n_kv_heads = 2, .head_dim = 8 };
    var leads = [_]Lead{.none} ** 20;
    leads[0] = .ramp;
    leads[7] = .sink;
    leads[8] = .ramp;
    leads[9] = .large;
    leads[12] = .sink;
    var prng = std.Random.DefaultPrng.init(0xA770_0005);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        try expectPartMatchesSingleHeads(E, r, shape, &leads, 0, 0, 3 * cpu_tile + 5, 0, 20);
    }
}

test "attention row: a repeated row and a row over a garbage partial buffer give the same bits" {
    const gpa = std.testing.allocator;
    const T = cpu_tile;
    const shape: AttnShape = .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 8 };
    const nh = shape.n_heads;
    const hd = shape.head_dim;
    var prng = std.Random.DefaultPrng.init(0xA770_0006);
    const r = prng.random();
    // Four parts leave 12 records per head unwritten. Seventeen tiles fill
    // every part, and the last part folds two tiles.
    inline for (.{ f32, f16 }) |E| {
        for ([_]usize{ 3 * T + 1, 16 * T + 1 }) |t_len| {
            const q = try gpa.alloc(f32, nh * hd);
            defer gpa.free(q);

            const k = try gpa.alloc(E, t_len * shape.n_kv_heads * hd);
            defer gpa.free(k);

            const v = try gpa.alloc(E, t_len * shape.n_kv_heads * hd);
            defer gpa.free(v);

            const parts_len = attn_cpu.partialLen(@intCast(nh), @intCast(hd));
            const parts = try gpa.alloc(f32, parts_len);
            defer gpa.free(parts);

            const out_first = try gpa.alloc(f32, nh * hd);
            defer gpa.free(out_first);

            const out_again = try gpa.alloc(f32, nh * hd);
            defer gpa.free(out_again);

            fillAttention(E, r, shape, t_len, &.{}, 0, q, k, v);

            @memset(parts, 0);
            attentionRowTyped(E, q, nh, hd, k, v, shape.n_kv_heads, t_len, parts, out_first);
            attentionRowTyped(E, q, nh, hd, k, v, shape.n_kv_heads, t_len, parts, out_again);
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(out_first),
                std.mem.sliceAsBytes(out_again),
            );

            for (parts) |*x| {
                x.* = r.float(f32) * 200.0 - 100.0;
            }

            attentionRowTyped(E, q, nh, hd, k, v, shape.n_kv_heads, t_len, parts, out_again);
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(out_first),
                std.mem.sliceAsBytes(out_again),
            );
        }
    }
}

test "attention merge: one part divides the value sums by the exponential sum" {
    const shape: AttnShape = .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 16 };
    const nh = shape.n_heads;
    const hd = shape.head_dim;
    const t_len = cpu_tile - 1;
    var q: [nh * hd]f32 = undefined;
    var k: [t_len * 2 * hd]f16 = undefined;
    var v: [t_len * 2 * hd]f16 = undefined;
    var parts: [attn_cpu.partialLen(nh, hd)]f32 = undefined;
    var out: [nh * hd]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xA770_0007);
    fillAttention(f16, prng.random(), shape, t_len, &.{}, 0, &q, &k, &v);
    attentionRowTyped(f16, &q, nh, hd, &k, &v, shape.n_kv_heads, t_len, &parts, &out);
    try std.testing.expectEqual(@as(u32, 1), attn_cpu.partCount(t_len));
    for (0..nh) |h| {
        const rec = parts[attn_cpu.recordOffset(
            nh,
            hd,
            @intCast(h),
            0,
        )..][0..attn_cpu.recordElems(hd)];
        const inv_l = 1.0 / rec[hd + 1];
        for (rec[0..hd], out[h * hd ..][0..hd]) |a, o| {
            try std.testing.expectEqual(@as(u32, @bitCast(a * inv_l)), @as(u32, @bitCast(o)));
        }
    }
}
