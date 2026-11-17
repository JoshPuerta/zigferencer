//! The linear kernels of both SIMD backends. The Q8_0 kernels, `linearF32Q80` and
//! `linearF32Q80Batch`, come in two sets, `plain` and `prefetched`. `simd` exports the first and
//! `simd_pf` the second. The matmul of f32 inputs and Q8_0 weights is memory-bound, so the layout
//! of the loop decides the speed. The f32 kernels, `linearF32` and `linearF32Batch`, are one set
//! that both backends export.
//!
//! Comptime. `Kernels` fixes the prefetch option, so the plain and the prefetching kernels are two
//! builds of one source. `linearTilesForCpu` picks the tile sizes of the build target.
//!
//! Bit rule. For one (row, output) pair, lane j of the accumulator adds the products of dims j,
//! j + 8, j + 16, and so on, in ascending order. Each product adds with one
//! `@mulAdd(w, x, acc)`, where `w = scale * @floatFromInt(q)`. That `w` is exact in f32, because
//! the f16 scale holds 11 significant bits and q at most 7. `lane_tree_simd` adds the 8 lanes
//! into the output in the lane tree order. A lane gets the same bits in a sum of one accumulator
//! or of four. Every path below follows this rule. So each output value has the same bits at
//! every batch size, with or without prefetch.
//!
//! B = 1. `groupPass` takes `mr = 4` output rows at once, one Q8_0 block at a time. The four
//! accumulators sit in registers and each input vector loads once for the four rows.
//!
//! Panel path. The B = 1 path repeats the i8 to f32 widening and the scale multiply for every
//! batch row. So with more batch rows a group widens its weights once per K tile and per pass
//! of `accumulator_rows` batch rows, and keeps them in a stack panel of `mr` rows by `k_tile`
//! weights. The panel stays in L1, and the inner loop reuses it for every batch row of the pass,
//! `b_tile` rows per register tile. The inner loop then holds loads and a fused multiply-add.
//! A batch over `accumulator_rows` widens again for each pass. Each batch row still sees the K
//! tiles in ascending order, so a pass leaves the bits unchanged. A group reads its weight bytes
//! from memory once per pass, not once per batch row.
//!
//! Prefetch. Only `prefetched` prefetches. Each block of a group prefetches the same block of the
//! next 4-row group with `.locality = 0`, because the pass reads those bytes once. So the lines
//! are warm when the outer loop arrives, and they do not pollute L1. Shapes that stream past the
//! last-level cache gain from it. Prefetch changes no bit.
//!
//! Tail. The outputs left when `out_features % mr != 0` take `rowPass`: one output, no prefetch,
//! the same vector walk, and the same lane tree.
//!
//! Tiles. No tile size enters the order of a lane, so the tiles change the speed, never a bit.
//! `linearTilesForCpu` holds one row, measured on a core with 16 vector registers of 256 bits and
//! a 32 KiB L1 data cache. Comptime asserts tie the tiles to those two figures and to
//! `limits.cpu_prefill_chunk`. A class of CPU gets its own row only with a measurement on it.
//!
//! F32. `linearF32` takes the bit rule with `w` read from the weight. The last `in_features % 8`
//! dims add with one vector whose other lanes are zero in `w` and in `x`. The f32 kernel takes
//! 4-output groups and tail outputs in the way of the Q8_0 kernels. It has no prefetch option,
//! because no registry model sends a 2-D f32 weight to it, and a prefetch needs a measurement.
//!
//! Requirements. Every AVX2 target also carries F16C. Without F16C the f16 widening stays
//! correct and runs scalar, so the backends do not list it in `requires`.

const std = @import("std");
const builtin = @import("builtin");
const limits = @import("../core/limits.zig");
const lane_tree = @import("lane_tree_simd.zig");
const naive = @import("naive.zig");
const Tensor = @import("../core/tensor.zig").Tensor;

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

// -- api --------------------------------------------------------------------

/// Linear layer with f32 weights: `out = input @ W.T`, under the bit rule.
pub fn linearF32(input: []const f32, weight: Tensor, out: []f32) void {
    std.debug.assert(weight.dtype == .f32);
    std.debug.assert(weight.n_dims == 2);

    const in_features: usize = @intCast(weight.shape[0]);
    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(input.len == in_features);
    std.debug.assert(out.len >= out_features);

    const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, weight.data));
    std.debug.assert(w.len >= out_features * in_features);

    // Take the full 4-output groups.
    var j: usize = 0;
    while (j + mr <= out_features) : (j += mr) {
        const sums: [lanes]f32 = lane_tree.transposeAdd(mr, f32Chains(mr, input, w, j));
        out[j..][0..mr].* = sums[0..mr].*;
    }

    // Take the tail outputs one at a time.
    while (j < out_features) : (j += 1) {
        out[j] = lane_tree.laneSum(f32Chains(1, input, w, j)[0]);
    }
    std.debug.assert(j == out_features);

    if (std.debug.runtime_safety) {
        for (out[0..out_features]) |x| {
            std.debug.assert(std.math.isFinite(x));
        }
    }
}

/// Batched linear with f32 weights, bit-identical to one `linearF32` call per row.
pub fn linearF32Batch(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    naive.linearBatchPerRow(linearF32, inputs, weight, outs, batch, out_stride);
}

// -- helpers ----------------------------------------------------------------

/// The accumulators of the `m` f32 outputs from `out0`. Each input vector loads once for the `m`
/// outputs.
inline fn f32Chains(comptime m: usize, input: []const f32, w: []const f32, out0: usize) [m]Vec {
    const in_features = input.len;
    std.debug.assert((out0 + m) * in_features <= w.len);

    var acc = [_]Vec{@splat(0)} ** m;
    var k: usize = 0;
    while (k + lanes <= in_features) : (k += lanes) {
        const x: Vec = input[k..][0..lanes].*;
        inline for (0..m) |r| {
            const wv: Vec = w[(out0 + r) * in_features + k ..][0..lanes].*;
            acc[r] = @mulAdd(Vec, wv, x, acc[r]);
        }
    }

    const tail = in_features - k;
    std.debug.assert(tail < lanes);
    if (tail > 0) {
        var x_pad: [lanes]f32 = @splat(0);
        @memcpy(x_pad[0..tail], input[k..]);
        inline for (0..m) |r| {
            var w_pad: [lanes]f32 = @splat(0);
            @memcpy(w_pad[0..tail], w[(out0 + r) * in_features + k ..][0..tail]);
            acc[r] = @mulAdd(Vec, w_pad, x_pad, acc[r]);
        }
    }

    return acc;
}

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

            // The rows under one tile take the widest tile that fits, so that each row of a
            // small batch does not sweep the panel alone.
            std.debug.assert(rows - b < b_tile);
            comptime var nb: usize = b_tile - 1;
            inline while (nb > 0) : (nb -= 1) {
                if (rows - b >= nb) {
                    const in_f = shape.in_features;
                    accumulateTile(nb, &panel, inputs, in_f, k0, tile, pass0, b, &accs);
                    b += nb;
                }
            }
            std.debug.assert(b == rows);
        }
        std.debug.assert(covered == shape.in_features);

        for (0..rows) |b| {
            const sums = lane_tree.transposeAdd(mr, accs[b]);
            inline for (0..mr) |r| {
                outs[(pass0 + b) * shape.out_stride + j + r] = sums[r];
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

/// Widen `n_blk` blocks of the 4 outputs into `panel` with `scaledQuarter`, the `w` of `blockStep`.
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

            const at = row_offs[r] + blk_off;
            const scale_v = blockScale(data, at);
            inline for (0..4) |q| {
                panel[r][t * 32 + q * lanes ..][0..lanes].* = scaledQuarter(data, at, scale_v, q);
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
                acc[bi][r] = @mulAdd(Vec, w, x[bi], acc[bi][r]);
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

    var acc = [_]Vec{@splat(0)} ** mr;

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

        var x: [4]Vec = undefined;
        inline for (0..4) |q| {
            x[q] = input[blk * 32 + q * lanes ..][0..lanes].*;
        }

        inline for (0..mr) |r| {
            acc[r] = blockStep(data, row_offs[r] + blk_off_in_row, x, acc[r]);
        }
    }

    const sums: [lanes]f32 = lane_tree.transposeAdd(mr, acc);
    return sums[0..mr].*;
}

/// One tail row.
inline fn rowPass(input: []const f32, data: []const u8, row_off: usize, n_blocks: usize) f32 {
    std.debug.assert(input.len == n_blocks * 32);
    std.debug.assert(row_off + n_blocks * 34 <= data.len);

    var acc: Vec = @splat(0);
    var blk: usize = 0;
    while (blk < n_blocks) : (blk += 1) {
        var x: [4]Vec = undefined;
        inline for (0..4) |q| {
            x[q] = input[blk * 32 + q * lanes ..][0..lanes].*;
        }
        acc = blockStep(data, row_off + blk * 34, x, acc);
    }

    return lane_tree.laneSum(acc);
}

/// Add one Q8_0 block of one output against the 32 inputs `x` into `acc`, with the `w` of
/// `scaledQuarter`.
inline fn blockStep(data: []const u8, blk_off: usize, x: [4]Vec, acc: Vec) Vec {
    std.debug.assert(blk_off + 34 <= data.len);
    const scale_v = blockScale(data, blk_off);
    var a = acc;
    inline for (0..4) |q| {
        a = @mulAdd(Vec, scaledQuarter(data, blk_off, scale_v, q), x[q], a);
    }
    return a;
}

/// The f32 scale of the Q8_0 block at `blk_off`, in every lane.
/// A scalar f16 bitcast lowers to an insert into a live register, a false dependency on an
/// accumulator. A broadcast of the u16 writes the whole register. The widening is exact.
inline fn blockScale(data: []const u8, blk_off: usize) Vec {
    const bits: @Vector(lanes, u16) = @splat(std.mem.readInt(u16, data[blk_off..][0..2], .little));
    const halves: @Vector(lanes, f16) = @bitCast(bits);
    return @floatCast(halves);
}

/// The 8 weights `w` of quarter `q` of the block at `blk_off`. The panel and the B = 1 path take
/// this one expression.
inline fn scaledQuarter(data: []const u8, blk_off: usize, scale_v: Vec, comptime q: usize) Vec {
    const arr: [lanes]i8 = @bitCast(data[blk_off + 2 + q * lanes ..][0..lanes].*);
    return scale_v * @as(Vec, @floatFromInt(@as(VecI8, arr)));
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

test "linear_simd stays inside the gamma bound of an f64 dot" {
    const gpa = std.testing.allocator;
    const out_features: usize = 37;
    const max_in: usize = 6144;
    const max_batch: usize = 64;
    const u = std.math.ldexp(@as(f64, 1), -24);
    const u_f64 = std.math.ldexp(@as(f64, 1), -53);

    const inputs = try gpa.alignedAlloc(f32, .of(f32), max_batch * max_in);
    defer gpa.free(inputs);

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * (max_in / 32) * 34);
    defer gpa.free(w_data);

    const outs = try gpa.alloc(f32, max_batch * out_features);
    defer gpa.free(outs);

    var prng = std.Random.DefaultPrng.init(0x6a33a);
    for (inputs) |*x| {
        x.* = prng.random().float(f32) * 2.0 - 1.0;
    }

    for ([_]usize{ 1088, 6144 }) |in_features| {
        const output_bytes = (in_features / 32) * 34;
        const data = w_data[0 .. out_features * output_bytes];
        fillScaledWeight(data, in_features, out_features, output_bytes);
        const w = q80Tensor(data, in_features, out_features);
        // The weight is exact. The longest path takes one fused multiply-add per product of
        // its lane and 3 adds of the lane tree, so n is in_features / 8 + 3. The f64 sum of
        // the reference adds in_features - 1 roundings of its own at 2^-53 (Higham, Accuracy
        // and Stability of Numerical Algorithms, ch. 4).
        const n: f64 = @floatFromInt(in_features / lanes + 3);
        const n64: f64 = @floatFromInt(in_features);
        const gamma = n * u / (1 - n * u) + n64 * u_f64 / (1 - n64 * u_f64);
        inline for (.{ plain, prefetched }) |K| {
            for ([_]usize{ 1, max_batch }) |batch| {
                const in = inputs[0 .. batch * in_features];
                K.linearF32Q80Batch(in, w, outs, batch, out_features);
                for (0..batch) |b| {
                    for (0..out_features) |j| {
                        const x = in[b * in_features ..][0..in_features];
                        const ref = dotF64(data, j * output_bytes, x);
                        const got: f64 = outs[b * out_features + j];
                        try std.testing.expect(@abs(got - ref.sum) <= gamma * ref.abs_sum);
                    }
                }
            }
        }
    }
}

/// The scale of the block at `blk_off` by the scalar widening, the reference for `blockScale`.
fn scalarScale(data: []const u8, blk_off: usize) f32 {
    return naive.f16BitsToF32(std.mem.readInt(u16, data[blk_off..][0..2], .little));
}

/// The f64 dot of one Q8_0 output against `x`, and the sum of the absolute products.
fn dotF64(data: []const u8, output_off: usize, x: []const f32) struct { sum: f64, abs_sum: f64 } {
    var sum: f64 = 0;
    var abs_sum: f64 = 0;
    for (0..x.len / 32) |blk| {
        const scale: f64 = scalarScale(data, output_off + blk * 34);
        for (0..32) |i| {
            const q: i8 = @bitCast(data[output_off + blk * 34 + 2 + i]);
            const p = scale * @as(f64, @floatFromInt(q)) * @as(f64, x[blk * 32 + i]);
            sum += p;
            abs_sum += @abs(p);
        }
    }
    return .{ .sum = sum, .abs_sum = abs_sum };
}

/// The bit rule in scalar f32 arithmetic: lane `i % 8` adds dim i with one fused multiply-add.
/// The result adds the 8 lanes in the lane tree order.
fn bitRuleScalar(data: []const u8, output_off: usize, x: []const f32) f32 {
    var acc: [lanes]f32 = @splat(0);
    for (0..x.len / 32) |blk| {
        const at = output_off + blk * 34;
        const scale = scalarScale(data, at);
        for (0..32) |i| {
            const q: i8 = @bitCast(data[at + 2 + i]);
            const w = scale * @as(f32, @floatFromInt(q));
            acc[i % lanes] = @mulAdd(f32, w, x[blk * 32 + i], acc[i % lanes]);
        }
    }
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

test "linear_simd gives the bits of the scalar order of its bit rule" {
    const gpa = std.testing.allocator;
    // The out size leaves a 3-output tail. Batches 3, 65 leave batch rows under one register
    // tile.
    const out_features: usize = 39;
    const max_in: usize = 2112;
    const batches = [_]usize{ 1, 2, 3, 8, 64, 65 };
    const max_batch = batches[batches.len - 1];

    const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * (max_in / 32) * 34);
    defer gpa.free(w_data);

    const inputs = try gpa.alloc(f32, max_batch * max_in);
    defer gpa.free(inputs);

    const outs = try gpa.alloc(f32, max_batch * out_features);
    defer gpa.free(outs);

    var prng = std.Random.DefaultPrng.init(0xb17);
    const r = prng.random();
    // Magnitudes from 2^-6 to 2^6 make a regrouped sum round differently.
    for (inputs) |*x| {
        x.* = (r.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), r.intRangeAtMost(i32, -6, 6));
    }

    for ([_]usize{ 1088, max_in }) |in_features| {
        const output_bytes = (in_features / 32) * 34;
        const data = w_data[0 .. out_features * output_bytes];
        // Random f16 scales and the full i8 range of q.
        for (0..out_features) |j| {
            for (0..in_features / 32) |blk| {
                const at = j * output_bytes + blk * 34;
                const scale: f16 = @floatCast(0.0005 + r.float(f32) * 0.02);
                std.mem.writeInt(u16, data[at..][0..2], @bitCast(scale), .little);
                r.bytes(data[at + 2 ..][0..32]);
            }
        }

        const w = q80Tensor(data, in_features, out_features);
        inline for (.{ plain, prefetched }) |K| {
            for (batches) |batch| {
                K.linearF32Q80Batch(inputs[0 .. batch * in_features], w, outs, batch, out_features);
                for (0..batch) |b| {
                    const x = inputs[b * in_features ..][0..in_features];
                    for (0..out_features) |j| {
                        const want: u32 = @bitCast(bitRuleScalar(data, j * output_bytes, x));
                        const got: u32 = @bitCast(outs[b * out_features + j]);
                        try std.testing.expectEqual(want, got);
                    }
                }
            }
        }
    }
}

/// The f32 shapes of the tests as (in_features, out_features). They hold an input under 8 lanes,
/// inputs off the lane grid, and output counts off the 4-output grid.
const f32_shapes = [_][2]usize{ .{ 3, 5 }, .{ 13, 9 }, .{ 64, 4 }, .{ 577, 7 }, .{ 1031, 13 } };
const f32_max_in = 1031;
const f32_max_out = 13;

fn f32Tensor(w: []const f32, in_features: usize, out_features: usize) Tensor {
    return .{
        .dtype = .f32,
        .n_dims = 2,
        .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
        .byte_stride = .{ @intCast(in_features * @sizeOf(f32)), @sizeOf(f32), 0, 0 },
        .data = std.mem.sliceAsBytes(w),
    };
}

/// Fill `x` with magnitudes from 2^-6 to 2^6, so that a regrouped sum rounds differently.
fn fillSpread(r: std.Random, x: []f32) void {
    for (x) |*xi| {
        xi.* = (r.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), r.intRangeAtMost(i32, -6, 6));
    }
}

/// The f32 bit rule in scalar arithmetic: lane `i % 8` adds dim i with one fused multiply-add.
/// The result adds the 8 lanes in the lane tree order. A pad lane adds `0 * 0` to a lane that is
/// never -0, so the scalar order leaves the pad out.
fn f32BitRuleScalar(w_out: []const f32, x: []const f32) f32 {
    var acc: [lanes]f32 = @splat(0);
    for (w_out, x, 0..) |wi, xi, i| {
        acc[i % lanes] = @mulAdd(f32, wi, xi, acc[i % lanes]);
    }
    return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

test "linear_simd linearF32 stays inside the gamma bound of an f64 dot" {
    const gpa = std.testing.allocator;
    const u = std.math.ldexp(@as(f64, 1), -24);
    const u_f64 = std.math.ldexp(@as(f64, 1), -53);

    const input = try gpa.alloc(f32, f32_max_in);
    defer gpa.free(input);

    const w_buf = try gpa.alloc(f32, f32_max_in * f32_max_out);
    defer gpa.free(w_buf);

    const out = try gpa.alloc(f32, f32_max_out);
    defer gpa.free(out);

    var prng = std.Random.DefaultPrng.init(0xf32_7013);
    const r = prng.random();
    for (f32_shapes) |shape| {
        const in_features = shape[0];
        const out_features = shape[1];
        const x = input[0..in_features];
        const w = w_buf[0 .. in_features * out_features];
        fillSpread(r, x);
        fillSpread(r, w);

        linearF32(x, f32Tensor(w, in_features, out_features), out);

        // A lane takes one fused multiply-add per vector, the padded vector included, then 3
        // adds of the lane tree. An f64 product of two f32 values is exact, and the f64 sum adds
        // in_features - 1 roundings at 2^-53 (Higham, Accuracy and Stability of Numerical
        // Algorithms, ch. 4).
        const n: f64 = @floatFromInt((in_features + lanes - 1) / lanes + 3);
        const n64: f64 = @floatFromInt(in_features);
        const gamma = n * u / (1 - n * u) + n64 * u_f64 / (1 - n64 * u_f64);
        for (0..out_features) |j| {
            var sum: f64 = 0;
            var abs_sum: f64 = 0;
            for (w[j * in_features ..][0..in_features], x) |wi, xi| {
                const p = @as(f64, wi) * @as(f64, xi);
                sum += p;
                abs_sum += @abs(p);
            }

            const got: f64 = out[j];
            try std.testing.expect(@abs(got - sum) <= gamma * abs_sum);
        }
    }
}

test "linear_simd linearF32 gives an output the same bits in a group, alone, and in a batch" {
    const gpa = std.testing.allocator;
    const batch: usize = 3;
    const gap: usize = 5;
    const nan_bits: u32 = @bitCast(std.math.nan(f32));

    const inputs = try gpa.alloc(f32, batch * f32_max_in);
    defer gpa.free(inputs);

    const w_buf = try gpa.alloc(f32, f32_max_in * f32_max_out);
    defer gpa.free(w_buf);

    const outs = try gpa.alloc(f32, batch * (f32_max_out + gap));
    defer gpa.free(outs);

    var out_single: [f32_max_out]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xf32_b175);
    const r = prng.random();
    for (f32_shapes) |shape| {
        const in_features = shape[0];
        const out_features = shape[1];
        const out_stride = out_features + gap;
        const in = inputs[0 .. batch * in_features];
        const w = w_buf[0 .. in_features * out_features];
        fillSpread(r, in);
        fillSpread(r, w);
        const weight = f32Tensor(w, in_features, out_features);

        // An output in a 4-output group gets the bits of the same output as a 1-output weight,
        // and the bits of the scalar order of the bit rule.
        const x = in[0..in_features];
        linearF32(x, weight, &out_single);
        for (0..out_features) |j| {
            const w_out = w[j * in_features ..][0..in_features];
            var alone: [1]f32 = undefined;
            linearF32(x, f32Tensor(w_out, in_features, 1), &alone);
            const got: u32 = @bitCast(out_single[j]);
            try std.testing.expectEqual(@as(u32, @bitCast(alone[0])), got);
            try std.testing.expectEqual(@as(u32, @bitCast(f32BitRuleScalar(w_out, x))), got);
        }

        // The batch gives the bits of per-row calls and leaves the stride gap untouched.
        @memset(outs, std.math.nan(f32));
        linearF32Batch(in, weight, outs, batch, out_stride);
        for (0..batch) |b| {
            linearF32(in[b * in_features ..][0..in_features], weight, &out_single);
            const row_outs = outs[b * out_stride ..];
            for (out_single[0..out_features], row_outs[0..out_features]) |s, g| {
                try std.testing.expectEqual(@as(u32, @bitCast(s)), @as(u32, @bitCast(g)));
            }

            for (row_outs[out_features..][0..gap]) |g| {
                try std.testing.expectEqual(nan_bits, @as(u32, @bitCast(g)));
            }
        }
    }
}
