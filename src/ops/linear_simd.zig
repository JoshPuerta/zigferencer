//! The Q8_0 linear kernels of both SIMD backends, `linearF32Q80` and `linearF32Q80Batch`. The
//! file makes two sets, `plain` and `prefetched`. `simd` exports the first and `simd_pf` the
//! second. The matmul of f32 inputs and Q8_0 weights is memory-bound, so the layout of the loop
//! decides the speed.
//!
//! Comptime. `Kernels` fixes the prefetch option, so the plain and the prefetching kernels are two
//! builds of one source. `linearTilesForCpu` picks the tile sizes of the build target.
//!
//! Bit rule. For one (row, output) pair, lane j of the accumulator adds the products of dims j,
//! j + 8, j + 16, and so on, in ascending order. Each product is
//! `w = scale * @floatFromInt(q)` followed by a separate multiply and add, never a fused one. One
//! `@reduce(.Add)` ends the sum. Every path below follows this rule. So the output is
//! bit-identical at every batch size, for every output row, with or without prefetch.
//!
//! B = 1. `groupPass` takes `mr = 4` output rows at once, one Q8_0 block at a time. The four
//! accumulators sit in registers and each input vector loads once for the four rows.
//!
//! Panel path. The B = 1 path repeats the i8 to f32 widening and the scale multiply for every
//! batch row. So with more batch rows a group widens its weights once per K tile and per pass
//! of `accumulator_rows` batch rows, and keeps them in a stack panel of `mr` rows by `k_tile`
//! weights. The panel stays in L1, and the inner loop reuses it for every batch row of the pass,
//! `b_tile` rows per register tile. The inner loop then holds loads, a multiply, and an add.
//! A batch over `accumulator_rows` widens again for each pass. Each batch row still sees the K
//! tiles in ascending order, so a pass leaves the bits unchanged. A group reads its weight bytes
//! from memory once per pass, not once per batch row.
//!
//! Prefetch. Only `prefetched` prefetches. Each block of a group prefetches the same block of the
//! next 4-row group with `.locality = 0`, because the pass reads those bytes once. So the lines
//! are warm when the outer loop arrives, and they do not pollute L1. Shapes that stream past the
//! last-level cache gain from it. Prefetch changes no bit.
//!
//! Tail. The rows left when `out_features % mr != 0` take `rowPass`: one row, no prefetch, and
//! the same vector walk.
//!
//! Tiles. No tile size enters the order of a lane, so the tiles change the speed, never a bit.
//! `linearTilesForCpu` holds one row, measured on a core with 16 vector registers of 256 bits and
//! a 32 KiB L1 data cache. Comptime asserts tie the tiles to those two figures and to
//! `limits.cpu_prefill_chunk`. A class of CPU gets its own row only with a measurement on it.
//!
//! Requirements. Every AVX2 target also carries F16C. Without F16C the f16 widening stays
//! correct and runs scalar, so the backends do not list it in `requires`.

const std = @import("std");
const builtin = @import("builtin");
const limits = @import("../core/limits.zig");
const naive = @import("naive.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const f16BitsToF32 = naive.f16BitsToF32;

// -- constants --------------------------------------------------------------

const lanes = 8;

/// Output rows per group. `groupPass` and the offsets of a group spell out four rows, so this
/// size stays out of the tile table.
const mr = 4;

/// The vector registers of the narrowest target that the SIMD backends accept: AVX2 names 16.
const vector_registers_min = 16;

/// The L1 data cache of the measured row, 32 KiB on Zen 3 (AMD publication 56665).
const l1_data_bytes = 32 * 1024;

/// The tile sizes of the build target.
const tiles = linearTilesForCpu(builtin.cpu);

/// Weights per row in one panel.
const k_tile = tiles.k_tile;

/// Batch rows per register tile.
const b_tile = tiles.b_tile;

/// The batch rows that one accumulator array holds. A larger batch runs the panel path once per
/// `accumulator_rows` rows.
const accumulator_rows = tiles.accumulator_rows;

comptime {
    // A panel tile holds whole Q8_0 blocks.
    std.debug.assert(k_tile % 32 == 0);
    // The register tile, its input vectors, and one panel vector fit with no spill.
    std.debug.assert(mr * b_tile + b_tile + 1 <= vector_registers_min);
    // The panel and the accumulators take half of L1, and the streamed input rows the rest.
    std.debug.assert(@sizeOf(Panel) + @sizeOf(Accumulators) <= l1_data_bytes / 2);
    // A chunk of the default prefill size runs whole passes.
    std.debug.assert(limits.cpu_prefill_chunk % accumulator_rows == 0);
}

/// The kernels without software prefetch. The `simd` backend exports them.
pub const plain = Kernels(.{});

/// The kernels that prefetch the next 4-row group. The `simd_pf` backend exports them. They give
/// the bits of `plain`.
pub const prefetched = Kernels(.{ .prefetch = true });

// -- types ------------------------------------------------------------------

const Vec = @Vector(lanes, f32);
const VecI8 = @Vector(lanes, i8);

/// The compile-time choices of one kernel set.
const Options = struct {
    /// Prefetch the matching block of the next 4-row group.
    prefetch: bool = false,
};

/// The tile sizes of the panel path on one class of CPU.
const Tiles = struct {
    /// Weights per row in one panel, a multiple of the 32 weights of a Q8_0 block.
    k_tile: usize,
    /// Batch rows per register tile.
    b_tile: usize,
    /// The batch rows that one accumulator array holds.
    accumulator_rows: usize,
};

/// The input width, the batch rows, and the output stride of one batched call.
const BatchShape = struct {
    in_features: usize,
    batch: usize,
    out_stride: usize,
};

/// The dequantized weights of one K tile, one row of `k_tile` weights per group row.
const Panel = [mr][k_tile]f32;

/// One accumulator vector per group row and batch row of one pass.
const Accumulators = [accumulator_rows][mr]Vec;

/// One kernel set: the two Q8_0 linear names of an ops backend.
fn Kernels(comptime opts: Options) type {
    return struct {
        pub fn linearF32Q80(input: []const f32, weight: Tensor, out: []f32) void {
            linearF32Q80Batch(input, weight, out, 1, @intCast(weight.shape[1]));
        }

        /// Batched linear with Q8_0 weights. Row b reads `inputs[b * in_features ..]` and writes
        /// `outs[b * out_stride ..]`, the same rule as `naive.linearF32Batch`.
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
            // The limit bounds the pass loop of `panelGroup`.
            std.debug.assert(batch <= limits.max_prefill_chunk);

            const in_features: usize = @intCast(weight.shape[0]);
            const out_features: usize = @intCast(weight.shape[1]);
            std.debug.assert(inputs.len == batch * in_features);
            std.debug.assert(out_stride >= out_features);
            std.debug.assert(outs.len >= (batch - 1) * out_stride + out_features);
            std.debug.assert(in_features % 32 == 0);

            const row_bytes = (in_features / 32) * 34;
            const n_blocks = in_features / 32;

            // Take the full 4-row groups.
            var j: usize = 0;
            while (j + mr <= out_features) : (j += mr) {
                const row_offs: [mr]usize = .{
                    (j + 0) * row_bytes,
                    (j + 1) * row_bytes,
                    (j + 2) * row_bytes,
                    (j + 3) * row_bytes,
                };
                // The last group has no next group to prefetch.
                const has_next = j + 2 * mr <= out_features;
                const pf_offs: [mr]usize = .{
                    (j + mr + 0) * row_bytes,
                    (j + mr + 1) * row_bytes,
                    (j + mr + 2) * row_bytes,
                    (j + mr + 3) * row_bytes,
                };

                if (batch > 1) {
                    const shape: BatchShape = .{
                        .in_features = in_features,
                        .batch = batch,
                        .out_stride = out_stride,
                    };
                    const panelFn = if (opts.prefetch) panelGroupPrefetched else panelGroupPlain;
                    panelFn(inputs, weight.data, outs, shape, j, row_offs, pf_offs, has_next);
                    continue;
                }

                const sums = groupPass(
                    opts.prefetch,
                    inputs[0..in_features],
                    weight.data,
                    row_offs,
                    pf_offs,
                    has_next,
                    n_blocks,
                );
                outs[j + 0] = sums[0];
                outs[j + 1] = sums[1];
                outs[j + 2] = sums[2];
                outs[j + 3] = sums[3];
            }

            // Take the tail rows one at a time.
            while (j < out_features) : (j += 1) {
                const row_off = j * row_bytes;
                var b: usize = 0;
                while (b < batch) : (b += 1) {
                    const input = inputs[b * in_features ..][0..in_features];
                    outs[b * out_stride + j] = rowPass(input, weight.data, row_off, n_blocks);
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
    };
}

// -- helpers ----------------------------------------------------------------

/// The tile sizes for `cpu`, a function of the target in the way of `std.atomic.cacheLineForCpu`.
/// A class with 32 vector registers, such as AVX-512VL or NEON, gets its own row only after a
/// measurement on that class. Until then every target takes the row measured on AVX2.
fn linearTilesForCpu(cpu: std.Target.Cpu) Tiles {
    _ = cpu;
    return .{ .k_tile = 512, .b_tile = 2, .accumulator_rows = 64 };
}

/// One 4-row weight group against every batch row, in passes of `accumulator_rows` rows.
inline fn panelGroup(
    comptime prefetch: bool,
    inputs: []const f32,
    data: []const u8,
    outs: []f32,
    shape: BatchShape,
    j: usize,
    row_offs: [mr]usize,
    pf_offs: [mr]usize,
    has_next: bool,
) void {
    std.debug.assert(shape.batch > 1);
    std.debug.assert(shape.batch <= limits.max_prefill_chunk);
    std.debug.assert(j + mr <= shape.out_stride);
    std.debug.assert((shape.batch - 1) * shape.out_stride + j + mr <= outs.len);

    var accs: Accumulators = undefined;

    // `dequantTile` writes the first `tile` weights of each row before `accumulateTile` reads them.
    var panel: Panel align(32) = undefined;
    var pass0: usize = 0;
    while (pass0 < shape.batch) : (pass0 += accumulator_rows) {
        const rows = @min(accumulator_rows, shape.batch - pass0);
        std.debug.assert(rows > 0);
        std.debug.assert(pass0 + rows <= shape.batch);

        for (accs[0..rows]) |*acc| {
            acc.* = .{ @splat(0), @splat(0), @splat(0), @splat(0) };
        }

        // Each batch row sees the K tiles in ascending order, so a pass leaves the bits unchanged.
        var k0: usize = 0;
        var covered: usize = 0;
        while (k0 < shape.in_features) : (k0 += k_tile) {
            const tile = @min(k_tile, shape.in_features - k0);
            std.debug.assert(tile % 32 == 0);
            covered += tile;

            dequantTile(prefetch, &panel, data, row_offs, pf_offs, has_next, k0 / 32, tile / 32);

            var b: usize = 0;
            while (b + b_tile <= rows) : (b += b_tile) {
                accumulateTile(
                    b_tile,
                    &panel,
                    inputs,
                    shape.in_features,
                    k0,
                    tile,
                    pass0,
                    b,
                    &accs,
                );
            }

            if (b < rows) {
                accumulateTile(1, &panel, inputs, shape.in_features, k0, tile, pass0, b, &accs);
            }
        }
        std.debug.assert(covered == shape.in_features);

        for (0..rows) |b| {
            inline for (0..mr) |r| {
                outs[(pass0 + b) * shape.out_stride + j + r] = @reduce(.Add, accs[b][r]);
            }
        }
    }
}

/// `panelGroup` without prefetch, as its own function. Inlined into the B = 1 entry, the panel
/// gives that entry a large stack frame.
noinline fn panelGroupPlain(
    inputs: []const f32,
    data: []const u8,
    outs: []f32,
    shape: BatchShape,
    j: usize,
    row_offs: [mr]usize,
    pf_offs: [mr]usize,
    has_next: bool,
) void {
    panelGroup(false, inputs, data, outs, shape, j, row_offs, pf_offs, has_next);
}

/// `panelGroup` with prefetch, as its own function. Inlined into the B = 1 entry, the panel gives
/// that entry a large stack frame.
noinline fn panelGroupPrefetched(
    inputs: []const f32,
    data: []const u8,
    outs: []f32,
    shape: BatchShape,
    j: usize,
    row_offs: [mr]usize,
    pf_offs: [mr]usize,
    has_next: bool,
) void {
    panelGroup(true, inputs, data, outs, shape, j, row_offs, pf_offs, has_next);
}

/// Widen `n_blk` blocks of the 4 rows into `panel`, with the same `w` expression as `groupPass`.
inline fn dequantTile(
    comptime prefetch: bool,
    panel: *Panel,
    data: []const u8,
    row_offs: [mr]usize,
    pf_offs: [mr]usize,
    has_next: bool,
    blk0: usize,
    n_blk: usize,
) void {
    std.debug.assert(n_blk * 32 <= k_tile);
    if (prefetch and has_next) {
        std.debug.assert(pf_offs[mr - 1] + (blk0 + n_blk) * 34 <= data.len);
    }
    std.debug.assert(row_offs[mr - 1] + (blk0 + n_blk) * 34 <= data.len);

    for (0..n_blk) |t| {
        const blk_off = (blk0 + t) * 34;
        inline for (0..mr) |r| {
            if (prefetch and has_next) {
                @prefetch(
                    &data[pf_offs[r] + blk_off],
                    .{ .rw = .read, .locality = 0, .cache = .data },
                );
            }

            const scale = f16BitsToF32(
                std.mem.readInt(u16, data[row_offs[r] + blk_off ..][0..2], .little),
            );
            const scale_v: Vec = @splat(scale);
            inline for (0..4) |q| {
                const w_arr: [lanes]i8 = @bitCast(
                    data[row_offs[r] + blk_off + 2 + q * lanes ..][0..lanes].*,
                );
                const w: Vec = scale_v * @as(Vec, @floatFromInt(@as(VecI8, w_arr)));
                panel[r][t * 32 + q * lanes ..][0..lanes].* = w;
            }
        }
    }
}

/// Add the products of one panel tile for `nb` batch rows of a pass into their accumulators.
inline fn accumulateTile(
    comptime nb: usize,
    panel: *const Panel,
    inputs: []const f32,
    in_features: usize,
    k0: usize,
    tile: usize,
    pass0: usize,
    b0: usize,
    accs: *Accumulators,
) void {
    std.debug.assert(tile <= k_tile);
    std.debug.assert(k0 + tile <= in_features);
    std.debug.assert(tile % lanes == 0);
    std.debug.assert((pass0 + b0 + nb) * in_features <= inputs.len);
    std.debug.assert(b0 + nb <= accs.len);

    var acc: [nb][mr]Vec = undefined;
    inline for (0..nb) |bi| {
        acc[bi] = accs[b0 + bi];
    }

    var k: usize = 0;
    while (k < tile) : (k += lanes) {
        var x: [nb]Vec = undefined;
        inline for (0..nb) |bi| {
            x[bi] = inputs[(pass0 + b0 + bi) * in_features + k0 + k ..][0..lanes].*;
        }

        inline for (0..mr) |r| {
            const w: Vec = panel[r][k..][0..lanes].*;
            inline for (0..nb) |bi| {
                acc[bi][r] = acc[bi][r] + w * x[bi];
            }
        }
    }

    inline for (0..nb) |bi| {
        accs[b0 + bi] = acc[bi];
    }
}

/// One input row against one 4-row weight group, the register-blocked core.
/// With `prefetch` and `has_next`, each iteration stages the next group's matching block.
inline fn groupPass(
    comptime prefetch: bool,
    input: []const f32,
    data: []const u8,
    row_offs: [mr]usize,
    pf_offs: [mr]usize,
    has_next: bool,
    n_blocks: usize,
) [mr]f32 {
    std.debug.assert(input.len == n_blocks * 32);
    std.debug.assert(row_offs[mr - 1] + n_blocks * 34 <= data.len);
    if (prefetch and has_next) std.debug.assert(pf_offs[mr - 1] + n_blocks * 34 <= data.len);

    var acc0: Vec = @splat(0);
    var acc1: Vec = @splat(0);
    var acc2: Vec = @splat(0);
    var acc3: Vec = @splat(0);

    var blk: usize = 0;
    while (blk < n_blocks) : (blk += 1) {
        const blk_off_in_row = blk * 34;

        // Prefetch the matching block of the next group.
        if (prefetch and has_next) {
            inline for (0..mr) |r| {
                @prefetch(
                    &data[pf_offs[r] + blk_off_in_row],
                    .{ .rw = .read, .locality = 0, .cache = .data },
                );
            }
        }

        // Load the four block scales.
        const s0v: Vec = @splat(
            f16BitsToF32(
                std.mem.readInt(u16, data[row_offs[0] + blk_off_in_row ..][0..2], .little),
            ),
        );
        const s1v: Vec = @splat(
            f16BitsToF32(
                std.mem.readInt(u16, data[row_offs[1] + blk_off_in_row ..][0..2], .little),
            ),
        );
        const s2v: Vec = @splat(
            f16BitsToF32(
                std.mem.readInt(u16, data[row_offs[2] + blk_off_in_row ..][0..2], .little),
            ),
        );
        const s3v: Vec = @splat(
            f16BitsToF32(
                std.mem.readInt(u16, data[row_offs[3] + blk_off_in_row ..][0..2], .little),
            ),
        );

        const in_base = blk * 32;

        // Accumulate the four rows, 8 lanes at a time.
        inline for (0..4) |q| {
            const in_v: Vec = input[in_base + q * lanes ..][0..lanes].*;

            const w0_arr: [lanes]i8 = @bitCast(
                data[row_offs[0] + blk_off_in_row + 2 + q * lanes ..][0..lanes].*,
            );
            const w1_arr: [lanes]i8 = @bitCast(
                data[row_offs[1] + blk_off_in_row + 2 + q * lanes ..][0..lanes].*,
            );
            const w2_arr: [lanes]i8 = @bitCast(
                data[row_offs[2] + blk_off_in_row + 2 + q * lanes ..][0..lanes].*,
            );
            const w3_arr: [lanes]i8 = @bitCast(
                data[row_offs[3] + blk_off_in_row + 2 + q * lanes ..][0..lanes].*,
            );

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

/// One tail row.
inline fn rowPass(input: []const f32, data: []const u8, row_off: usize, n_blocks: usize) f32 {
    std.debug.assert(input.len == n_blocks * 32);
    std.debug.assert(row_off + n_blocks * 34 <= data.len);

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

/// Fill `w_data` with scale 1.0 and a repeating i8 pattern. The `simd` and `simd_pf` tests share
/// it.
pub fn fillTestWeight(
    w_data: []u8,
    in_features: usize,
    out_features: usize,
    row_bytes: usize,
) void {
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

/// Fill like `fillTestWeight`, then vary the f16 scale per block, so a scale mix-up between
/// blocks changes the bits.
fn fillScaledWeight(w_data: []u8, in_features: usize, out_features: usize, row_bytes: usize) void {
    fillTestWeight(w_data, in_features, out_features, row_bytes);
    for (0..out_features) |row| {
        for (0..in_features / 32) |blk| {
            const scale_bits: u16 = @intCast(0x3c00 + (row + blk) % 64);
            const at = row * row_bytes + blk * 34;
            std.mem.writeInt(u16, w_data[at..][0..2], scale_bits, .little);
        }
    }
}

fn q80Tensor(w_data: []u8, in_features: usize, out_features: usize) Tensor {
    return .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast((in_features / 32) * 34), 0, 0, 0 },
        .data = w_data,
    };
}

/// Compare `K.linearF32Q80Batch` against per-row calls for each batch size, and check that the
/// call leaves the stride gap untouched.
fn panelCheck(
    comptime K: type,
    in_features: usize,
    out_features: usize,
    batches: []const usize,
) !void {
    const gpa = std.testing.allocator;
    const row_bytes = (in_features / 32) * 34;
    const gap: usize = 5;
    const out_stride = out_features + gap;
    const nan_bits: u32 = @bitCast(std.math.nan(f32));

    var max_batch: usize = 0;
    for (batches) |batch| {
        max_batch = @max(max_batch, batch);
    }

    const inputs = try gpa.alignedAlloc(f32, .of(f32), max_batch * in_features);
    defer gpa.free(inputs);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);

    const outs = try gpa.alloc(f32, max_batch * out_stride);
    defer gpa.free(outs);

    const out_single = try gpa.alloc(f32, out_features);
    defer gpa.free(out_single);

    var prng = std.Random.DefaultPrng.init(0x5eed);
    for (inputs) |*x| {
        x.* = prng.random().float(f32) * 2.0 - 1.0;
    }

    fillScaledWeight(w_data, in_features, out_features, row_bytes);
    const w = q80Tensor(w_data, in_features, out_features);

    for (batches) |batch| {
        @memset(outs, std.math.nan(f32));
        K.linearF32Q80Batch(inputs[0 .. batch * in_features], w, outs, batch, out_stride);
        for (0..batch) |b| {
            K.linearF32Q80(inputs[b * in_features ..][0..in_features], w, out_single);
            for (out_single, outs[b * out_stride ..][0..out_features]) |s, g| {
                try std.testing.expectEqual(@as(u32, @bitCast(s)), @as(u32, @bitCast(g)));
            }

            for (outs[b * out_stride + out_features ..][0..gap]) |g| {
                try std.testing.expectEqual(nan_bits, @as(u32, @bitCast(g)));
            }
        }
    }
}

test "linear_simd panel path equals per-row calls across K tiles" {
    // Two full K tiles and a 64-weight tail tile.
    const in_features: usize = 2 * k_tile + 64;

    // Batch 130 is two full passes and 2 rows. The out sizes leave tails of 1, 2, and 3 rows.
    const batches = [_]usize{ 1, 2, 3, accumulator_rows, 2 * accumulator_rows + 2 };
    inline for (.{ plain, prefetched }) |K| {
        for ([_]usize{ 1037, 1038, 1039 }) |out_features| {
            try panelCheck(K, in_features, out_features, &batches);
        }
    }
}

test "linear_simd panel path on a last group with a tail" {
    // One group with no next group to prefetch, and a 3-row tail.
    inline for (.{ plain, prefetched }) |K| {
        try panelCheck(K, 1088, 7, &.{ 2, 3 });
    }
}

test "linear_simd plain and prefetched give the same bits at every batch size" {
    const gpa = std.testing.allocator;
    const in_features: usize = 1088;
    const out_features: usize = 1037;
    const row_bytes = (in_features / 32) * 34;
    const batches = [_]usize{ 1, 3, 64 };

    const inputs = try gpa.alignedAlloc(f32, .of(f32), 64 * in_features);
    defer gpa.free(inputs);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);

    const outs_plain = try gpa.alloc(f32, 64 * out_features);
    defer gpa.free(outs_plain);

    const outs_pf = try gpa.alloc(f32, 64 * out_features);
    defer gpa.free(outs_pf);

    var prng = std.Random.DefaultPrng.init(0xbeef);
    for (inputs) |*x| {
        x.* = prng.random().float(f32) * 2.0 - 1.0;
    }

    fillScaledWeight(w_data, in_features, out_features, row_bytes);
    const w = q80Tensor(w_data, in_features, out_features);

    for (batches) |batch| {
        const n = batch * out_features;
        const in = inputs[0 .. batch * in_features];
        plain.linearF32Q80Batch(in, w, outs_plain, batch, out_features);
        prefetched.linearF32Q80Batch(in, w, outs_pf, batch, out_features);
        for (outs_plain[0..n], outs_pf[0..n]) |a, b| {
            try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
        }
    }
}

test "linear_simd rowPass equals the group path of linearF32Q80" {
    const gpa = std.testing.allocator;
    const in_features: usize = 1088;
    const out_features: usize = 13; // 3 groups and a 1-row tail
    const row_bytes = (in_features / 32) * 34;

    const input = try gpa.alignedAlloc(f32, .of(f32), in_features);
    defer gpa.free(input);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
    defer gpa.free(w_data);

    const out = try gpa.alloc(f32, out_features);
    defer gpa.free(out);

    var prng = std.Random.DefaultPrng.init(0xfeed);
    for (input) |*x| {
        x.* = prng.random().float(f32) * 2.0 - 1.0;
    }

    fillScaledWeight(w_data, in_features, out_features, row_bytes);
    const w = q80Tensor(w_data, in_features, out_features);

    inline for (.{ plain, prefetched }) |K| {
        K.linearF32Q80(input, w, out);
        for (out, 0..) |g, row| {
            const one = rowPass(input, w_data, row * row_bytes, in_features / 32);
            try std.testing.expectEqual(@as(u32, @bitCast(one)), @as(u32, @bitCast(g)));
        }
    }
}
