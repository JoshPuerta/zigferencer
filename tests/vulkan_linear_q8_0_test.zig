//! Bit-fidelity gate for the GPU Q8_0 matmul.
//!
//! Oracle. The gate runs `linearF32Q80` through `ops_vulkan.LinearQ80`, and the result must equal
//! the CPU naive op `zigferencer.ops.linearF32Q80` bit for bit. That op is the correctness
//! reference of the project.
//!
//! Bit equality. Bit equality is reachable, so the gate asks for it. The kernel reproduces the
//! exact f32 reduction order of the naive op: a scalar accumulate, block-major and i-minor, with
//! one invocation per output row. Vulkan mandates IEEE 754 single precision for f32 multiply and
//! add without FMA contraction unless `precise` is set.
//!
//! Faults. A divergence names one of three faults. The reduction order does not match, the f16 to
//! f32 decode of the scale differs, or the i8 byte decode differs. A tolerance absorbs all three.
//!
//! Skips. The test returns `error.SkipZigTest` when Vulkan cannot start, which covers a host with
//! no driver and a device with a missing feature.

const std = @import("std");
const zigferencer = @import("zigferencer");
const cpu_ops = zigferencer.ops;
const Tensor = zigferencer.tensor.Tensor;

// -- fixtures ---------------------------------------------------------------

const q8_0_block_elems: usize = 32;
const q8_0_block_bytes: usize = 34;

/// Fill `weight_data` with a synthetic Q8_0 weight. Every block takes the f16 scale 1.0, and the
/// quants step through the full signed i8 range with a per-row offset, which drives sign extension.
fn buildSyntheticWeight(
    weight_data: []u8,
    in_features: usize,
    out_features: usize,
) void {
    const blocks_per_row = in_features / q8_0_block_elems;
    const row_bytes = blocks_per_row * q8_0_block_bytes;
    std.debug.assert(weight_data.len == out_features * row_bytes);
    for (0..out_features) |j| {
        const row_off = j * row_bytes;
        for (0..blocks_per_row) |blk| {
            const blk_off = row_off + blk * q8_0_block_bytes;
            // The f16 value 1.0 has the bit pattern 0x3c00.
            std.mem.writeInt(u16, weight_data[blk_off..][0..2], 0x3c00, .little);
            for (0..q8_0_block_elems) |i| {
                // The row, block and element offsets make every (j, blk, i)
                // cell distinct across the full i8 range.
                const raw: i32 = @intCast((j + blk * 32 + i) % 251);
                const v: i8 = @intCast(raw - 125);
                weight_data[blk_off + 2 + i] = @bitCast(v);
            }
        }
    }
}

fn runLinearQ80Check(
    allocator: std.mem.Allocator,
    in_features: usize,
    out_features: usize,
) !void {
    const runner = zigferencer.vulkan_runner.Runner.init(allocator) catch |e| switch (e) {
        error.NoVulkanDevice,
        error.InitializationFailed,
        error.IncompatibleDriver,
        error.FeatureNotPresent,
        => return error.SkipZigTest,
        else => return e,
    };
    defer runner.deinit();

    const kernel = &runner.linear_q8_0;

    const blocks_per_row = in_features / q8_0_block_elems;
    const row_bytes = blocks_per_row * q8_0_block_bytes;
    const total_w_bytes = out_features * row_bytes;

    const input = try allocator.alloc(f32, in_features);
    defer allocator.free(input);

    const weight_bytes = try allocator.alloc(u8, total_w_bytes);
    defer allocator.free(weight_bytes);

    const out_gpu = try allocator.alloc(f32, out_features);
    defer allocator.free(out_gpu);

    const out_cpu = try allocator.alloc(f32, out_features);
    defer allocator.free(out_cpu);

    // `@sin(i)` is the input pattern of the linear bench cases in tools/bench.zig. It keeps
    // the numerics in a realistic range.
    for (input, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)));
    }

    buildSyntheticWeight(weight_bytes, in_features, out_features);

    try kernel.runOneShot(input, weight_bytes, in_features, out_features, out_gpu);

    const w_tensor: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
        .data = weight_bytes,
    };
    cpu_ops.linearF32Q80(input, w_tensor, out_cpu);

    for (0..out_features) |j| {
        if (@as(u32, @bitCast(out_gpu[j])) != @as(u32, @bitCast(out_cpu[j]))) {
            std.debug.print(
                "j={d} gpu={d} ({x:0>8}) cpu={d} ({x:0>8})  delta={d}\n",
                .{
                    j,
                    out_gpu[j],
                    @as(u32, @bitCast(out_gpu[j])),
                    out_cpu[j],
                    @as(u32, @bitCast(out_cpu[j])),
                    out_gpu[j] - out_cpu[j],
                },
            );
            return error.GpuMatmulDivergesFromCpu;
        }
    }
}

// -- tests ------------------------------------------------------------------

test "linearF32Q80: GPU bit-exact to CPU naive, k=64 n=32" {
    try runLinearQ80Check(std.testing.allocator, 64, 32);
}

test "linearF32Q80: GPU bit-exact to CPU naive, k=128 n=33" {
    // k spans more than one block pair, which covers more reduction terms.
    // n=33 forces a second, half-empty workgroup, so the kernel runs the
    // early-out path that the `j >= params.n` guard takes.
    try runLinearQ80Check(std.testing.allocator, 128, 33);
}

test "linearF32Q80: GPU bit-exact to CPU naive, k=576 n=49" {
    // The SmolLM2 lm_head has k=576. n=49 is a small slice of a 49152-wide
    // head, kept tiny so the test runs fast.
    try runLinearQ80Check(std.testing.allocator, 576, 49);
}
