//! The vector part kernel of CPU decode attention, for the AVX2 and FMA backends. It computes the
//! record of `naive.attentionPartTyped` over the same part, with `@Vector(8, f32)` lanes over the
//! head width. The tiles, the online fold, the head blocks, and the record layout are those of
//! naive. The dot products, the value sums, the exponentials, and the exponential sums run in
//! lanes, so the record bits differ from naive inside a measured tolerance. naive stays the
//! reference.
//!
//! Bit rules. The bits of one head are a function of its query, its part, and the KV rows alone.
//! Five rules hold that:
//! 1. Lane j of a score accumulator adds the products of dims j, j + 8, j + 16, and so on, one
//!    fused multiply-add each, in ascending order.
//! 2. One fixed tree folds the 8 lanes of a score:
//!    `((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))`. `transposeAdd` folds up to 8
//!    accumulators at once with this tree, and it pads a short set with zero vectors. A pad lane
//!    never meets a real lane. So the head block size, the position step, and a short last tile
//!    do not change a score.
//! 3. Lane d of a value sum adds the weighted V rows in ascending position order, one fused
//!    multiply-add each. A store and a reload between two runs of positions are exact.
//! 4. The exponential sum of a tile folds its 8 lane sums with the same tree.
//! 5. `expNeg` of `exp_simd.zig` computes each lane from the input of that lane alone, with one
//!    fixed sequence of IEEE operations. So a score gets the same exponential at any lane
//!    position, and a short last tile changes no exponential.
//!
//! So a head split, a block split, and a thread split give the same record bytes, and software
//! prefetch changes no bit.
//!
//! Head blocks. A block holds one score row per head on the stack, 64 KiB at 8 heads. That bound
//! holds per block instance. A Debug build can keep every inlined block instance in one frame.
//! The 8 block instances unroll their chunk and head loops into the one `partKernel` body.
//!
//! Register blocks. `block_accs` is 8 because a core with two FMA pipes at a latency of about 4
//! cycles needs 8 independent chains to keep both busy. AVX2 names 16 ymm registers, and the
//! other 8 hold the loaded K or V chunks. A block of n heads walks `8 / n` positions or chunks
//! per step, so 4 heads walk 2 and 1 head walks 8.
//!
//! Value runs. A run of the value pass reads one 64-byte line from each of its V rows per chunk
//! step, so a short run keeps few streams open. At depth 8192 and a tile of 256, runs of 32
//! measure 8 to 18 percent faster on one worker than runs of the whole tile, and 1 to 5 percent
//! faster with the units spread over 16 workers.
//!
//! Prefetch distance. At depth 8192 with the units spread over 16 workers, distance 16 measures
//! 6 to 26 percent faster on the three bench rows that stream from DRAM, and 2 to 4 percent
//! faster on the row that stays in the last-level cache. At a tile of 256 the same rows measured
//! 4 to 33 percent, one worker gained 21 to 38 percent, and distances 8 and 32 measured within 6
//! percent of 16.
//!
//! Inlining. `partKernel` inlines into `Kernels.attentionPartTyped`, and every helper below it
//! inlines into `partKernel`. So each kernel set holds one copy of the kernel, and
//! `tools/simd_probe.zig` reads the whole body under one name.

const std = @import("std");
const attn_cpu = @import("../core/attn_cpu.zig");
const limits = @import("../core/limits.zig");
const exp_simd = @import("exp_simd.zig");
const naive = @import("naive.zig");

const assert = std.debug.assert;
const expNeg = exp_simd.expNeg;

// -- constants --------------------------------------------------------------

const lanes = exp_simd.lanes;

const tile: usize = limits.attn_cpu_tile;

/// The widest head the vector path takes. The tests cover the registry widths
/// up to it, and a wider head runs the naive kernel.
const max_head_dim: usize = limits.attn_max_head_dim;

/// The widest head block. See "Head blocks" in the header.
const max_group: usize = limits.attn_max_group;

/// Accumulators one register block holds. See "Register blocks" in the header.
const block_accs = 8;

/// Positions one register run of the value pass walks. See "Value runs" in the header.
const value_run = 32;

/// How far ahead the prefetching kernels fetch the K and V rows, in positions.
/// See "Prefetch distance" in the header.
const prefetch_positions = 16;

comptime {
    assert(tile % lanes == 0);
    assert(tile % value_run == 0);
    assert(max_head_dim % lanes == 0);
    assert(block_accs <= lanes);
}

/// The kernels without software prefetch. The `simd` backend exports them.
pub const plain = Kernels(.{});

/// The kernels that prefetch the K and V rows `prefetch_positions` ahead. The
/// `simd_pf` backend exports them. They give the bits of `plain`.
pub const prefetched = Kernels(.{ .prefetch = prefetch_positions });

// -- types ------------------------------------------------------------------

const Vec = exp_simd.Vec;

/// The compile-time choices of one kernel set.
const Options = struct {
    /// Prefetch the K and V rows this many positions ahead. Null prefetches nothing.
    prefetch: ?usize = null,
};

/// The inputs of one head block, as its three passes read them.
fn Block(comptime E: type) type {
    return struct {
        /// The queries of the block heads, head after head.
        q: []const f32,
        head_dim: usize,
        k_cache: []const E,
        v_cache: []const E,
        kv_stride: usize,
        /// The first element of the KV head inside a KV row.
        kv_off: usize,
        t_lo: usize,
        t_hi: usize,
        /// The record of the first block head. Head g starts at `g * rec_stride`.
        recs: []f32,
        rec_stride: usize,
        scale: f32,
    };
}

/// How one run of the value pass starts its accumulators.
const Start = enum {
    /// The first tile of the part: the sums start at zero.
    zero,
    /// The first run of a later tile: the sums take the rescale factor.
    rescale,
    /// A later run of a tile: the sums continue.
    keep,
};

/// One kernel set: the part kernel, the serial row, and the merge, the three
/// attention names of an ops backend.
fn Kernels(comptime opts: Options) type {
    return struct {
        /// One part of a decode attention row, with the asserts and the record layout of
        /// `naive.attentionPartTyped`. A head width off the lane grid, or over 256, runs naive.
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
            partKernel(
                E,
                opts,
                q,
                h_lo,
                h_hi,
                n_heads,
                head_dim,
                k_cache,
                v_cache,
                n_kv_heads,
                t_lo,
                t_hi,
                recs,
                rec_stride,
            );
        }

        pub const attentionMerge = naive.attentionMerge;

        /// The attention row of every query head, serial: each part in order, then the merge.
        /// The arguments and the buffer sizes are those of `naive.attentionRowTyped`.
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
            naive.attentionRowWith(
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
    };
}

// -- helpers ----------------------------------------------------------------

/// The body of `Kernels.attentionPartTyped`. See "Inlining" in the header.
inline fn partKernel(
    comptime E: type,
    comptime opts: Options,
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
    // The block instances unroll past the default quota of 1000 comptime branches.
    @setEvalBranchQuota(20_000);

    // Check the arguments.
    comptime assert(E == f32 or E == f16);
    assert(h_lo < h_hi);
    assert(h_hi <= n_heads);
    assert(n_kv_heads >= 1);
    assert(n_heads % n_kv_heads == 0);
    assert(head_dim >= 1);
    assert(q.len >= n_heads * head_dim);
    assert(t_lo < t_hi);
    assert(t_hi <= limits.max_model_ctx);
    const kv_stride = n_kv_heads * head_dim;
    assert(k_cache.len >= t_hi * kv_stride);
    assert(v_cache.len >= t_hi * kv_stride);
    const rec_len: usize = attn_cpu.recordElems(@intCast(head_dim));
    assert(rec_stride >= rec_len);
    assert(recs.len >= (h_hi - h_lo - 1) * rec_stride + rec_len);

    // Run the naive kernel on a head width off the lane grid.
    if (head_dim % lanes != 0 or head_dim > max_head_dim) {
        naive.attentionPartTyped(
            E,
            q,
            h_lo,
            h_hi,
            n_heads,
            head_dim,
            k_cache,
            v_cache,
            n_kv_heads,
            t_lo,
            t_hi,
            recs,
            rec_stride,
        );
        return;
    }

    assert(head_dim % lanes == 0);
    assert(head_dim <= max_head_dim);
    // The tile loop of `foldBlock` runs at most this many times.
    assert((t_hi - t_lo + tile - 1) / tile <= limits.max_model_ctx / tile);

    // Cut each KV group of heads into blocks of at most `max_group`.
    const group_size = n_heads / n_kv_heads;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    var kv_h = h_lo / group_size;
    const kv_h_end = (h_hi - 1) / group_size + 1;
    while (kv_h < kv_h_end) : (kv_h += 1) {
        const g_lo = @max(h_lo, kv_h * group_size);
        const g_hi = @min(h_hi, (kv_h + 1) * group_size);
        var blk_lo = g_lo;
        while (blk_lo < g_hi) : (blk_lo += max_group) {
            const blk_hi = @min(g_hi, blk_lo + max_group);
            const blk: Block(E) = .{
                .q = q[blk_lo * head_dim .. blk_hi * head_dim],
                .head_dim = head_dim,
                .k_cache = k_cache,
                .v_cache = v_cache,
                .kv_stride = kv_stride,
                .kv_off = kv_h * head_dim,
                .t_lo = t_lo,
                .t_hi = t_hi,
                .recs = recs[(blk_lo - h_lo) * rec_stride ..],
                .rec_stride = rec_stride,
                .scale = scale,
            };
            switch (blk_hi - blk_lo) {
                inline 1...max_group => |n| foldBlock(E, n, opts, blk),
                else => unreachable,
            }
        }
    }

    // Check that every record is finite.
    if (std.debug.runtime_safety) {
        var h = h_lo;
        while (h < h_hi) : (h += 1) {
            for (recs[(h - h_lo) * rec_stride ..][0..rec_len]) |x| {
                assert(std.math.isFinite(x));
            }
        }
    }
}

/// The online fold of `n` heads of one KV head over the part, tile by tile.
/// The fold state belongs to one head and never to the block.
inline fn foldBlock(
    comptime E: type,
    comptime n: usize,
    comptime opts: Options,
    blk: Block(E),
) void {
    comptime assert(n >= 1 and n <= max_group);
    // Row g holds the scores of head g over one tile, then the shifted exponentials.
    var scores: [n][tile]f32 = undefined;
    var m_run: [n]f32 = undefined;
    var l_run: [n]f32 = undefined;
    var rescale: [n]f32 = undefined;

    // Fold the tiles in order.
    var tile_lo = blk.t_lo;
    while (tile_lo < blk.t_hi) : (tile_lo += tile) {
        const tile_n = @min(blk.t_hi - tile_lo, tile);
        const first = tile_lo == blk.t_lo;
        scoreTile(E, n, opts, blk, tile_lo, tile_n, &scores);
        for (0..n) |g| {
            rescale[g] = foldScores(&scores[g], tile_n, first, &m_run[g], &l_run[g]);
        }

        valueTile(E, n, opts, blk, tile_lo, tile_n, &scores, first, rescale);
    }

    // Store the running maximum and sum of each head.
    for (0..n) |g| {
        const rec = blk.recs[g * blk.rec_stride ..][0..attn_cpu.recordElems(
            @intCast(blk.head_dim),
        )];
        rec[blk.head_dim] = m_run[g];
        rec[blk.head_dim + 1] = l_run[g];
    }
}

/// The scaled scores of the block heads over the positions
/// `[tile_lo, tile_lo + tile_n)`.
inline fn scoreTile(
    comptime E: type,
    comptime n: usize,
    comptime opts: Options,
    blk: Block(E),
    tile_lo: usize,
    tile_n: usize,
    scores: *[n][tile]f32,
) void {
    // See "Register blocks" in the header.
    const step: usize = comptime @max(1, block_accs / n);
    var t: usize = 0;
    while (t + step <= tile_n) : (t += step) {
        scoreStep(E, n, step, opts, blk, tile_lo, t, scores);
    }

    while (t < tile_n) : (t += 1) {
        scoreStep(E, n, 1, opts, blk, tile_lo, t, scores);
    }
}

/// The scores of the block heads at `step` positions from `tile_lo + t`. One
/// K chunk load and one widening feed every head of the block.
inline fn scoreStep(
    comptime E: type,
    comptime n: usize,
    comptime step: usize,
    comptime opts: Options,
    blk: Block(E),
    tile_lo: usize,
    t: usize,
    scores: *[n][tile]f32,
) void {
    comptime assert(n * step <= lanes);
    const line_elems = 64 / @sizeOf(E);
    const pos = tile_lo + t;
    var acc = [_]Vec{@splat(0)} ** (n * step);

    // Accumulate the dot products one lane chunk at a time.
    var d: usize = 0;
    while (d < blk.head_dim) : (d += lanes) {
        if (opts.prefetch) |dist| {
            if (d % line_elems == 0 and pos + step + dist <= blk.t_hi) {
                inline for (0..step) |p| {
                    const ahead = (pos + p + dist) * blk.kv_stride + blk.kv_off + d;
                    @prefetch(
                        blk.k_cache.ptr + ahead,
                        .{ .rw = .read, .locality = 3, .cache = .data },
                    );
                }
            }
        }

        var k: [step]Vec = undefined;
        inline for (0..step) |p| {
            k[p] = loadRow(E, blk.k_cache, (pos + p) * blk.kv_stride + blk.kv_off + d);
        }

        inline for (0..n) |g| {
            const qv: Vec = blk.q[g * blk.head_dim + d ..][0..lanes].*;
            inline for (0..step) |p| {
                acc[g * step + p] = @mulAdd(Vec, qv, k[p], acc[g * step + p]);
            }
        }
    }

    // Fold the lanes and store the scaled scores.
    const sums = transposeAdd(n * step, acc) * @as(Vec, @splat(blk.scale));
    inline for (0..n) |g| {
        inline for (0..step) |p| {
            scores[g][t + p] = sums[g * step + p];
        }
    }
}

/// Fold one tile of scores into the running state of one head. The row turns into the shifted
/// exponentials. Returns the factor for the value sums: 1 when the tile does not raise the maximum.
inline fn foldScores(row: *[tile]f32, tile_n: usize, first: bool, m_run: *f32, l_run: *f32) f32 {
    assert(tile_n >= 1);
    assert(tile_n <= tile);

    // Take the maximum of the tile.
    const padded = (tile_n + lanes - 1) / lanes * lanes;
    if (std.debug.runtime_safety) {
        for (row[0..tile_n]) |s| {
            assert(std.math.isFinite(s));
        }
    }

    // A pad of -inf leaves the maximum alone. The maximum is exact in any order.
    @memset(row[tile_n..padded], -std.math.inf(f32));
    var max_v: Vec = row[0..lanes].*;
    var i: usize = lanes;
    while (i < padded) : (i += lanes) {
        max_v = @max(max_v, @as(Vec, row[i..][0..lanes].*));
    }

    const m_tile = @reduce(.Max, max_v);

    // Rescale the running sum when the maximum rises.
    var factor: f32 = 1;
    if (first) {
        m_run.* = m_tile;
        l_run.* = 0;
    } else if (m_tile > m_run.*) {
        factor = expNeg(@splat(m_run.* - m_tile))[0];
        l_run.* *= factor;
        m_run.* = m_tile;
    }

    // Turn the row into exponentials. `expNeg` maps the -inf pad to 0, which leaves the sums alone.
    const m: Vec = @splat(m_run.*);
    var sum_v: Vec = @splat(0);
    i = 0;
    while (i < padded) : (i += lanes) {
        const e = expNeg(@as(Vec, row[i..][0..lanes].*) - m);
        row[i..][0..lanes].* = e;
        sum_v += e;
    }

    if (std.debug.runtime_safety) {
        for (row[tile_n..padded]) |e| {
            assert(e == 0);
        }
    }

    // Add the tile sum to the running sum.
    const l = l_run.* + transposeAdd(1, .{sum_v})[0];
    // The term at the maximum adds exp(0) = 1, and no term is negative.
    assert(l >= 1);
    l_run.* = l;
    return factor;
}

/// Add the weighted V rows of one tile to the value sums of the block heads, in runs.
inline fn valueTile(
    comptime E: type,
    comptime n: usize,
    comptime opts: Options,
    blk: Block(E),
    tile_lo: usize,
    tile_n: usize,
    scores: *const [n][tile]f32,
    first: bool,
    rescale: [n]f32,
) void {
    // See "Register blocks" in the header.
    const chunks: usize = comptime @max(1, block_accs / n);
    var run_lo: usize = 0;
    while (run_lo < tile_n) : (run_lo += value_run) {
        const run_n = @min(value_run, tile_n - run_lo);
        const start: Start = if (run_lo > 0) .keep else if (first) .zero else .rescale;
        var d: usize = 0;
        while (d + chunks * lanes <= blk.head_dim) : (d += chunks * lanes) {
            valueStep(E, n, chunks, opts, blk, tile_lo, run_lo, run_n, scores, start, rescale, d);
        }

        while (d < blk.head_dim) : (d += lanes) {
            valueStep(E, n, 1, opts, blk, tile_lo, run_lo, run_n, scores, start, rescale, d);
        }
    }
}

/// The value sums of the block heads over `chunks` lane chunks from `d`, for one run of
/// positions. The accumulators stay in registers across the run.
inline fn valueStep(
    comptime E: type,
    comptime n: usize,
    comptime chunks: usize,
    comptime opts: Options,
    blk: Block(E),
    tile_lo: usize,
    run_lo: usize,
    run_n: usize,
    scores: *const [n][tile]f32,
    start: Start,
    rescale: [n]f32,
    d: usize,
) void {
    comptime assert(n * chunks <= lanes);
    const line_elems = 64 / @sizeOf(E);

    // Load the accumulators from the records.
    var acc: [n * chunks]Vec = undefined;
    inline for (0..n) |g| {
        const a = blk.recs[g * blk.rec_stride + d ..][0 .. chunks * lanes];
        inline for (0..chunks) |j| {
            // A factor of 1 is exact, so a tile without a new maximum leaves the sums alone.
            acc[g * chunks + j] = switch (start) {
                .zero => @splat(0),
                .rescale => @as(Vec, a[j * lanes ..][0..lanes].*) * @as(Vec, @splat(rescale[g])),
                .keep => a[j * lanes ..][0..lanes].*,
            };
        }
    }

    // Add the weighted V rows. One V chunk load and one widening feed every head.
    var t = run_lo;
    while (t < run_lo + run_n) : (t += 1) {
        const off = (tile_lo + t) * blk.kv_stride + blk.kv_off + d;
        if (opts.prefetch) |dist| {
            if (d % line_elems == 0 and tile_lo + t + dist < blk.t_hi) {
                inline for (0..chunks) |j| {
                    if (comptime (j * lanes) % line_elems == 0) {
                        @prefetch(
                            blk.v_cache.ptr + off + dist * blk.kv_stride + j * lanes,
                            .{ .rw = .read, .locality = 3, .cache = .data },
                        );
                    }
                }
            }
        }

        var v: [chunks]Vec = undefined;
        inline for (0..chunks) |j| {
            v[j] = loadRow(E, blk.v_cache, off + j * lanes);
        }

        inline for (0..n) |g| {
            const w: Vec = @splat(scores[g][t]);
            inline for (0..chunks) |j| {
                acc[g * chunks + j] = @mulAdd(Vec, w, v[j], acc[g * chunks + j]);
            }
        }
    }

    // Store the accumulators back to the records.
    inline for (0..n) |g| {
        const a = blk.recs[g * blk.rec_stride + d ..][0 .. chunks * lanes];
        inline for (0..chunks) |j| {
            a[j * lanes ..][0..lanes].* = acc[g * chunks + j];
        }
    }
}

/// Load 8 elements of a KV row as f32. An f16 row widens in registers, one
/// packed conversion per 8 lanes.
inline fn loadRow(comptime E: type, src: []const E, off: usize) Vec {
    const raw: @Vector(lanes, E) = src[off..][0..lanes].*;
    return if (E == f32) raw else @floatCast(raw);
}

/// Fold the lanes of each of `m` accumulators: lane i of the result is the sum of the lanes
/// of `accs[i]`. See bit rule 2 in the header. A lane gets the same bits at any `m`.
inline fn transposeAdd(comptime m: usize, accs: [m]Vec) Vec {
    comptime assert(m >= 1 and m <= lanes);
    var a = [_]Vec{@splat(0)} ** lanes;
    inline for (0..m) |i| {
        a[i] = accs[i];
    }

    // Lane i of `low4` holds the sum of the low 4 lanes of accumulator i, and
    // lane i + 4 the sum of its high 4 lanes. `high4` holds the same for
    // accumulators 4 to 7.
    const low4 = pairAdd(pairAdd(a[0], a[1]), pairAdd(a[2], a[3]));
    const high4 = pairAdd(pairAdd(a[4], a[5]), pairAdd(a[6], a[7]));
    const lo = @shuffle(
        f32,
        low4,
        high4,
        [lanes]i32{ 0, 1, 2, 3, ~@as(i32, 0), ~@as(i32, 1), ~@as(i32, 2), ~@as(i32, 3) },
    );
    const hi = @shuffle(
        f32,
        low4,
        high4,
        [lanes]i32{ 4, 5, 6, 7, ~@as(i32, 4), ~@as(i32, 5), ~@as(i32, 6), ~@as(i32, 7) },
    );
    return lo + hi;
}

/// The horizontal pair sums of two vectors inside each 128-bit half:
/// `[a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7]`.
inline fn pairAdd(a: Vec, b: Vec) Vec {
    const even = @shuffle(
        f32,
        a,
        b,
        [lanes]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2), 4, 6, ~@as(i32, 4), ~@as(i32, 6) },
    );
    const odd = @shuffle(
        f32,
        a,
        b,
        [lanes]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3), 5, 7, ~@as(i32, 5), ~@as(i32, 7) },
    );
    return even + odd;
}

// -- tests ------------------------------------------------------------------

const attention_ref = @import("attention_ref.zig");
const AttnShape = attention_ref.AttnShape;
const Lead = attention_ref.Lead;
const cpu_tile = attention_ref.cpu_tile;

/// The largest gap to the naive part that the part test accepts, over the value average
/// `a / l` and the log-sum-exp `m + ln(l)`. The cases measure at most 2.53e-6, so the bound
/// is 11.9 times that.
const simd_part_abs_tol: f64 = 3.0e-5;

/// Run the vector part and the naive part over one range and return the largest gap.
fn partGapToNaive(
    comptime E: type,
    r: std.Random,
    shape: AttnShape,
    t_lo: usize,
    t_hi: usize,
) !f64 {
    const gpa = std.testing.allocator;
    const nh = shape.n_heads;
    const hd = shape.head_dim;
    const q = try gpa.alloc(f32, nh * hd);
    defer gpa.free(q);

    const k = try gpa.alloc(E, t_hi * shape.n_kv_heads * hd);
    defer gpa.free(k);

    const v = try gpa.alloc(E, t_hi * shape.n_kv_heads * hd);
    defer gpa.free(v);

    const rec_len: usize = attn_cpu.recordElems(@intCast(hd));
    const recs_vec = try gpa.alloc(f32, nh * rec_len);
    defer gpa.free(recs_vec);

    const recs_ref = try gpa.alloc(f32, nh * rec_len);
    defer gpa.free(recs_ref);

    attention_ref.fillAttention(E, r, shape, t_hi, &.{}, 0, q, k, v);
    plain.attentionPartTyped(
        E,
        q,
        0,
        nh,
        nh,
        hd,
        k,
        v,
        shape.n_kv_heads,
        t_lo,
        t_hi,
        recs_vec,
        rec_len,
    );
    naive.attentionPartTyped(
        E,
        q,
        0,
        nh,
        nh,
        hd,
        k,
        v,
        shape.n_kv_heads,
        t_lo,
        t_hi,
        recs_ref,
        rec_len,
    );
    var worst: f64 = 0;
    for (0..nh) |h| {
        const a = recs_vec[h * rec_len ..][0..rec_len];
        const b = recs_ref[h * rec_len ..][0..rec_len];
        const l_a: f64 = a[hd + 1];
        const l_b: f64 = b[hd + 1];
        worst = @max(worst, @abs((a[hd] + @log(l_a)) - (b[hd] + @log(l_b))));
        for (a[0..hd], b[0..hd]) |x, y| {
            worst = @max(worst, @abs(x / l_a - y / l_b));
        }
    }

    return worst;
}

test "vector part: the records track the naive part on the same range" {
    const T = cpu_tile;
    // One position, a short tile, a full tile, a part of two tiles that ends
    // one past a tile edge, and five tiles that end in an odd short tile.
    const Range = struct { lo: usize, hi: usize };
    const ranges = [_]Range{
        .{ .lo = 0, .hi = 1 },
        .{ .lo = 0, .hi = 7 },
        .{ .lo = 0, .hi = T },
        .{ .lo = T, .hi = 2 * T + 1 },
        .{ .lo = T, .hi = 5 * T + 17 },
    };
    var prng = std.Random.DefaultPrng.init(0xA770_00A4);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        for ([_]usize{ 64, 128, 256 }) |hd| {
            for (ranges) |range| {
                // The five tiles run at one width, so the test stays short in
                // Debug. The shorter ranges run every width.
                if (range.hi - range.lo > 2 * T and hd != 64) continue;
                const gap = try partGapToNaive(
                    E,
                    r,
                    .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = hd },
                    range.lo,
                    range.hi,
                );
                try std.testing.expect(gap <= simd_part_abs_tol);
            }
        }
    }
}

test "vector row: the row tracks the f64 reference across tile and part edges" {
    const T = cpu_tile;
    const depths = [_]usize{ 1, 7, T - 1, T, T + 1, 2 * T, 16 * T, 16 * T + 1, 4 * T + 3 };
    var prng = std.Random.DefaultPrng.init(0xA770_00A5);
    const r = prng.random();
    // The deep rows take few and narrow heads, so the test stays short in
    // Debug. The wider heads span 3 tiles in 3 parts, so they cross two tile
    // edges and two part edges.
    inline for (.{ f32, f16 }) |E| {
        for (depths) |t_len| {
            try attention_ref.expectRowNearF64(
                plain,
                E,
                r,
                .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 8 },
                t_len,
                &.{},
                false,
            );
        }

        // The two wider registered heads, and a group of 3 with an odd head
        // count.
        try attention_ref.expectRowNearF64(
            plain,
            E,
            r,
            .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 128 },
            2 * T + 1,
            &.{},
            false,
        );
        try attention_ref.expectRowNearF64(
            plain,
            E,
            r,
            .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 256 },
            2 * T + 1,
            &.{},
            false,
        );
        try attention_ref.expectRowNearF64(
            plain,
            E,
            r,
            .{ .n_heads = 3, .n_kv_heads = 1, .head_dim = 8 },
            16 * T + 1,
            &.{},
            false,
        );
    }
}

test "vector row: a sink, a ramp, and scores near 100 track the f64 reference" {
    // 34 tiles over 16 parts, so every part folds two or three tiles. The last
    // tile is short, and it still holds enough of the ramp to raise the maximum.
    const t_len = 34 * cpu_tile - 5;
    const shape: AttnShape = .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 8 };
    var prng = std.Random.DefaultPrng.init(0xA770_00A6);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        for ([_]Lead{ .sink, .ramp, .large }) |lead| {
            const leads = [_]Lead{lead} ** shape.n_heads;
            try attention_ref.expectRowNearF64(plain, E, r, shape, t_len, &leads, false);
        }
    }
}

test "vector part: a head range gives each head the records of a single-head call at an odd last tile" {
    // A single head walks 8 positions per score step, a group of 4 walks 2, and
    // a group of 3 walks 2. The range ends in a tile of 17 positions, so each
    // walk ends on a lone position. The ramp heads peak in the last tile, and
    // the sink heads peak in the first tile of the part.
    const T = cpu_tile;
    const leads = [_]Lead{ .ramp, .sink, .none, .large, .sink, .none, .ramp, .large };
    var prng = std.Random.DefaultPrng.init(0xA770_00A7);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        for ([_]usize{ 64, 128, 256 }) |hd| {
            const shape: AttnShape = .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = hd };
            try attention_ref.expectPartMatchesSingleHeads(
                plain,
                E,
                r,
                shape,
                &leads,
                T + 3,
                T,
                2 * T + 17,
                0,
                8,
            );
            // The ranges cut both groups into blocks of 3, then of 2. A block
            // cut does not depend on the head width, so the cuts run at one width.
            if (hd != 64) continue;

            try attention_ref.expectPartMatchesSingleHeads(
                plain,
                E,
                r,
                shape,
                &leads,
                T + 3,
                T,
                2 * T + 17,
                1,
                7,
            );
            try attention_ref.expectPartMatchesSingleHeads(
                plain,
                E,
                r,
                shape,
                &leads,
                T + 3,
                T,
                2 * T + 17,
                2,
                6,
            );
        }

        const leads9 = [_]Lead{ .ramp, .none, .sink, .large, .none, .ramp, .sink, .none, .large };
        try attention_ref.expectPartMatchesSingleHeads(
            plain,
            E,
            r,
            .{ .n_heads = 9, .n_kv_heads = 3, .head_dim = 64 },
            &leads9,
            T + 3,
            T,
            2 * T + 17,
            0,
            9,
        );
    }
}

test "vector part: a group of 10 heads in blocks of 8 and 2 gives the records of single-head calls" {
    const shape: AttnShape = .{ .n_heads = 20, .n_kv_heads = 2, .head_dim = 8 };
    var leads = [_]Lead{.none} ** 20;
    leads[0] = .ramp;
    leads[7] = .sink;
    leads[8] = .ramp;
    leads[9] = .large;
    leads[12] = .sink;
    var prng = std.Random.DefaultPrng.init(0xA770_00A8);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        try attention_ref.expectPartMatchesSingleHeads(
            plain,
            E,
            r,
            shape,
            &leads,
            0,
            0,
            3 * cpu_tile + 5,
            0,
            20,
        );
    }
}

test "vector row: forced fan-out at 1, 2, 3, 4, and 16 threads gives the bits of the serial row, with and without prefetch" {
    const T = cpu_tile;
    // The row of 17 tiles takes narrow heads alone, so the test stays short in
    // Debug. The widest registered head runs the one-part and two-part rows.
    const shapes = [_]AttnShape{
        .{ .n_heads = 4, .n_kv_heads = 2, .head_dim = 8 },
        .{ .n_heads = 6, .n_kv_heads = 3, .head_dim = 8 },
        .{ .n_heads = 8, .n_kv_heads = 2, .head_dim = 256 },
    };
    // One part, two parts, and 17 tiles in 16 parts. At 3 threads the units
    // do not split evenly over the participants.
    const depths = [_]usize{ 5, T + 1, 16 * T + 3 };
    var prng = std.Random.DefaultPrng.init(0xA770_00A9);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        for (shapes) |s| {
            for (depths) |t_len| {
                if (s.head_dim > 8 and t_len > 2 * T) continue;
                try attention_ref.expectFanOutMatchesRow(.{ plain, prefetched }, E, r, s, t_len);
            }
        }
    }
}

test "vector part: a head width off the lane grid or over 256 runs the naive kernel bit for bit" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA770_00AA);
    const r = prng.random();
    const t_hi = cpu_tile + 9;
    inline for (.{ f32, f16 }) |E| {
        for ([_]usize{ 12, 264 }) |hd| {
            const shape: AttnShape = .{ .n_heads = 3, .n_kv_heads = 1, .head_dim = hd };
            const q = try gpa.alloc(f32, shape.n_heads * hd);
            defer gpa.free(q);

            const k = try gpa.alloc(E, t_hi * shape.n_kv_heads * hd);
            defer gpa.free(k);

            const v = try gpa.alloc(E, t_hi * shape.n_kv_heads * hd);
            defer gpa.free(v);

            const rec_len: usize = attn_cpu.recordElems(@intCast(hd));
            const recs_vec = try gpa.alloc(f32, shape.n_heads * rec_len);
            defer gpa.free(recs_vec);

            const recs_ref = try gpa.alloc(f32, shape.n_heads * rec_len);
            defer gpa.free(recs_ref);

            attention_ref.fillAttention(E, r, shape, t_hi, &.{}, 0, q, k, v);
            plain.attentionPartTyped(
                E,
                q,
                0,
                shape.n_heads,
                shape.n_heads,
                hd,
                k,
                v,
                shape.n_kv_heads,
                3,
                t_hi,
                recs_vec,
                rec_len,
            );
            naive.attentionPartTyped(
                E,
                q,
                0,
                shape.n_heads,
                shape.n_heads,
                hd,
                k,
                v,
                shape.n_kv_heads,
                3,
                t_hi,
                recs_ref,
                rec_len,
            );
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(recs_ref),
                std.mem.sliceAsBytes(recs_vec),
            );
        }
    }
}
