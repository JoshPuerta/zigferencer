//! The matmul check of `gpu-selftest`, against the naive Q8_0 reference.

const std = @import("std");
const build_options = @import("build_options");

const zigferencer = @import("zigferencer");

const context = zigferencer.cli.context;
const naive = zigferencer.ops;
const Tensor = zigferencer.tensor.Tensor;
// These imports resolve to `void` on a CPU build, so the file still compiles there.
const vulkan_ctx = if (build_options.use_vulkan) zigferencer.vulkan_ctx else void;
const kernels = if (build_options.use_vulkan) zigferencer.ops_vulkan else void;

const gpu_selftest_common = @import("gpu_selftest_common.zig");

// -- constants --------------------------------------------------------------

/// Values per Q8_0 block, and bytes per block (a 2-byte f16 scale and the quants).
const q80_block_values = 32;
const q80_block_bytes = 34;

/// Matmul shapes as `{ b, k, n }`. They cover aligned tiles and the batch and output
/// tails. The tile sizes are 8 by 16 for batch and 64 by 64 for gemm.
const matmul_shapes = [_][3]u32{
    .{ 8, 256, 64 },
    .{ 5, 64, 40 }, // b & n tails
    .{ 13, 128, 96 }, // odd b
    .{ 1, 64, 32 }, // degenerate b=1
    .{ 64, 64, 64 }, // exact one gemm tile
    .{ 70, 128, 200 }, // multi-tile with b & n tails
    .{ 128, 512, 128 }, // prefill-shaped
    // The Qwen3.5-2B projections: the widths 64 and 128 take the tiled kernel.
    .{ 64, 2048, 2048 },
    .{ 128, 2048, 6144 },
    .{ 128, 6144, 2048 },
    .{ 7, 2048, 2048 }, // a width below the tiled kernel
};

// -- api --------------------------------------------------------------------

/// Check the batch and gemm Q8_0 kernels against `naive.linearF32Q80` on `matmul_shapes`.
/// Both kernels reorder the reduction against the CPU, so the gate is a tolerance, never bit-exact.
pub fn matmulCheck(ctx: context.Context) !void {
    var ctxp = try ctx.arena.create(vulkan_ctx.Context);
    ctxp.* = try vulkan_ctx.Context.init(ctx.arena, ctx.vk_options);
    defer ctxp.deinit();

    var k_batch = try kernels.LinearQ80Batch.init(ctxp);
    defer k_batch.deinit();

    var k_gemm = try kernels.LinearQ80Gemm.init(ctxp);
    defer k_gemm.deinit();

    var all_ok = true;
    for (matmul_shapes) |s| {
        if (!try matmulCheckDims(ctx, "Batch", &k_batch, s[0], s[1], s[2])) {
            all_ok = false;
        }

        if (!try matmulCheckDims(ctx, "Gemm ", &k_gemm, s[0], s[1], s[2])) {
            all_ok = false;
        }
    }

    if (!all_ok) return error.BitCheckFailed;
}

// -- helpers ----------------------------------------------------------------

fn matmulCheckDims(
    ctx: context.Context,
    tag: []const u8,
    kernel: anytype,
    b: u32,
    k: u32,
    n: u32,
) !bool {
    std.debug.assert(k % q80_block_values == 0);

    // Build the random weight and input.
    const a = ctx.arena;
    const blocks_per_row = k / q80_block_values;
    const row_bytes = blocks_per_row * q80_block_bytes;
    const weight = try a.alloc(u8, n * row_bytes);
    const input = try a.alloc(f32, @as(usize, b) * k);

    var prng = std.Random.DefaultPrng.init(0x6EA77 + @as(u64, b) * 1_000 + n);
    const r = prng.random();
    buildRandomQ80(r, weight, k, n);
    gpu_selftest_common.fillN(r, input, 1.0);

    // Compute the CPU reference. The GGUF shape is reversed: {k, n} is in then out features.
    const w_tensor: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ k, n, 0, 0 },
        .byte_stride = .{ row_bytes, 0, 0, 0 },
        .data = weight,
    };
    const out_cpu = try a.alloc(f32, @as(usize, b) * n);
    for (0..b) |bi| {
        naive.linearF32Q80(input[bi * k ..][0..k], w_tensor, out_cpu[bi * n ..][0..n]);
    }

    // Run the GPU kernel and compare.
    const out_gpu = try a.alloc(f32, @as(usize, b) * n);
    try kernel.runOneShot(input, weight, b, k, n, out_gpu);

    const w = &ctx.stdout.interface;
    try w.print("{s} GPU vs CPU (b={d}, k={d}, n={d}):\n", .{ tag, b, k, n });
    try gpu_selftest_common.printDiff(w, "out", out_cpu, out_gpu);

    const atol: f32 = 1e-4;
    const rtol: f32 = 2e-3;
    const fails = gpu_selftest_common.countFail(out_cpu, out_gpu, atol, rtol);
    const ok = fails == 0;
    try w.print(
        "  fails (atol={e:.0} rtol={e:.0})={d}  RESULT: {s}\n\n",
        .{ atol, rtol, fails, if (ok) "PASS" else "FAIL" },
    );
    try w.flush();
    return ok;
}

/// Build a synthetic Q8_0 weight in GGUF layout: `n` rows of `k/32` blocks, each block 34 bytes.
/// Small random scales with full-range i8 quants give a realistic numeric spread.
fn buildRandomQ80(r: std.Random, data: []u8, k: u32, n: u32) void {
    const blocks_per_row = k / q80_block_values;
    const row_bytes = blocks_per_row * q80_block_bytes;
    std.debug.assert(data.len == @as(usize, n) * row_bytes);

    for (0..n) |j| {
        const row_off = j * row_bytes;
        for (0..blocks_per_row) |blk| {
            const blk_off = row_off + blk * q80_block_bytes;
            const scale: f16 = @floatCast(0.005 + 0.02 * r.float(f32));
            std.mem.writeInt(u16, data[blk_off..][0..2], @bitCast(scale), .little);
            for (0..q80_block_values) |i| {
                const v: i8 = @intCast(@as(i32, @round((r.float(f32) - 0.5) * 250.0)));
                data[blk_off + 2 + i] = @bitCast(v);
            }
        }
    }
}
