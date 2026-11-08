//! Shared test support for the CPU attention kernels: an f64 reference row, a scalar full-row
//! oracle, the input fillers, and the checks that the scalar and the vector kernel tests and the
//! control unit test run. Test code imports this file, and no production path reaches it.
//!
//! Comptime. `E` and `Kernels` make the checks generic, so f32 and f16 share one body.
//!
//! Kernels. A check takes the kernels as a namespace that declares `attentionPartTyped`,
//! `attentionMerge`, and `attentionRowTyped`, the shape of an ops backend. So one reference and
//! one tolerance hold every CPU backend.
//!
//! Tolerance. `row_abs_tol` is the largest absolute error against `attentionRefF64` that the row
//! tests accept. Over the cases of the scalar tests, the scalar part row measures at most
//! 3.33e-7 on the f16 sink, and the scalar oracle at most 3.07e-6 on the f32 sink. The vector
//! part row measures at most 4.81e-7 on the f16 sink. Uniform rows measure at most 7.8e-8. The
//! bound is 41.6 times the largest part row maximum.
//!
//! Leads. `fillAttention` starts every value uniform in [-1, 1]. A head with a lead reads K
//! coordinate `@intFromEnum(lead) - 1` at weight `sqrt(head_dim)`, and its other coordinates
//! shrink to a tenth. So that coordinate sets the score of the head to within about 0.1. The
//! ramp rises from -10 to 10 across the row. A ramp head zeroes its other coordinates, so its
//! score is the ramp alone, and a short last tile of a long row still raises the maximum. The
//! sink is 10 at `sink_t` and 0 elsewhere. The large lead is uniform in [90, 100], where an
//! unshifted f32 exponential overflows.

const std = @import("std");
const attn_cpu = @import("../core/attn_cpu.zig");
const limits = @import("../core/limits.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;

// -- constants --------------------------------------------------------------

pub const cpu_tile: usize = limits.attn_cpu_tile;

/// The largest absolute error against `attentionRefF64` that the row tests accept.
const row_abs_tol: f32 = 2.0e-5;

// -- types ------------------------------------------------------------------

/// A test geometry: query heads, KV heads, and head width.
pub const AttnShape = struct { n_heads: usize, n_kv_heads: usize, head_dim: usize };

/// The score profile of a test head. `none` leaves the head uniform.
pub const Lead = enum { none, ramp, sink, large };

// -- api --------------------------------------------------------------------

/// Fill one test row with random values, then shape the heads that have a lead.
pub fn fillAttention(
    comptime E: type,
    r: std.Random,
    shape: AttnShape,
    t_len: usize,
    leads: []const Lead,
    sink_t: usize,
    q: []f32,
    k: []E,
    v: []E,
) void {
    // Fill q, k, and v uniformly.
    for (q) |*x| {
        x.* = r.float(f32) * 2.0 - 1.0;
    }

    for (k) |*x| {
        x.* = @floatCast(r.float(f32) * 2.0 - 1.0);
    }

    for (v) |*x| {
        x.* = @floatCast(r.float(f32) * 2.0 - 1.0);
    }

    if (leads.len == 0) return;

    // Set the q side of each lead head.
    std.debug.assert(leads.len == shape.n_heads);
    std.debug.assert(shape.head_dim >= 4);
    const hd = shape.head_dim;
    var used = [_]bool{false} ** 3;
    for (leads, 0..) |lead, h| {
        if (lead == .none) continue;

        const dim = @intFromEnum(lead) - 1;
        used[dim] = true;
        const q_h = q[h * hd ..][0..hd];
        const rest: f32 = if (lead == .ramp) 0.0 else 0.1;
        for (q_h) |*x| {
            x.* *= rest;
        }

        @memset(q_h[0..3], 0);
        q_h[dim] = @sqrt(@as(f32, @floatFromInt(hd)));
    }

    if (used[1]) std.debug.assert(sink_t < t_len);

    // Set the k side of each lead.
    const span: f32 = @floatFromInt(@max(t_len, 2) - 1);
    for (0..t_len) |t| {
        for (0..shape.n_kv_heads) |kv_h| {
            const k_row = k[(t * shape.n_kv_heads + kv_h) * hd ..][0..hd];
            if (used[0]) {
                k_row[0] = @floatCast(20.0 * @as(f32, @floatFromInt(t)) / span - 10.0);
            }

            if (used[1]) {
                k_row[1] = if (t == sink_t) 10.0 else 0.0;
            }

            if (used[2]) {
                k_row[2] = @floatCast(90.0 + 10.0 * r.float(f32));
            }
        }
    }
}

/// Fill one row, then demand that the row of `Kernels` and, when `with_oracle` is set, the
/// scalar oracle stay inside `row_abs_tol` of the f64 reference.
pub fn expectRowNearF64(
    comptime Kernels: type,
    comptime E: type,
    r: std.Random,
    shape: AttnShape,
    t_len: usize,
    leads: []const Lead,
    with_oracle: bool,
) !void {
    const gpa = std.testing.allocator;
    const nh = shape.n_heads;
    const hd = shape.head_dim;
    const q = try gpa.alloc(f32, nh * hd);
    defer gpa.free(q);

    const kv_len = t_len * shape.n_kv_heads * hd;
    const k = try gpa.alloc(E, kv_len);
    defer gpa.free(k);

    const v = try gpa.alloc(E, kv_len);
    defer gpa.free(v);

    const parts = try gpa.alloc(f32, attn_cpu.partialLen(@intCast(nh), @intCast(hd)));
    defer gpa.free(parts);

    const out = try gpa.alloc(f32, nh * hd);
    defer gpa.free(out);

    const ref_scores = try gpa.alloc(f64, t_len);
    defer gpa.free(ref_scores);

    const ref_out = try gpa.alloc(f64, nh * hd);
    defer gpa.free(ref_out);

    // Run the kernel row and the f64 reference on the same inputs.
    fillAttention(E, r, shape, t_len, leads, 0, q, k, v);
    try expectLeadProfiles(E, q, k, shape, leads, 0, 0, t_len);
    Kernels.attentionRowTyped(E, q, nh, hd, k, v, shape.n_kv_heads, t_len, parts, out);
    attentionRefF64(E, q, nh, hd, k, v, shape.n_kv_heads, t_len, ref_scores, ref_out);

    // Compare the errors with the bound.
    const row_err = maxAbsErr(out, ref_out);
    var oracle_err: f64 = 0;
    if (with_oracle) {
        const scratch = try gpa.alloc(f32, t_len);
        defer gpa.free(scratch);

        attentionRowScalarOracle(E, q, nh, hd, k, v, shape.n_kv_heads, t_len, scratch, out);
        oracle_err = maxAbsErr(out, ref_out);
    }

    try std.testing.expect(row_err <= row_abs_tol);
    try std.testing.expect(oracle_err <= row_abs_tol);
}

/// Run one part of `Kernels` over the heads `[h_lo, h_hi)`, then each head alone, and demand
/// the same record bytes.
pub fn expectPartMatchesSingleHeads(
    comptime Kernels: type,
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
    const gpa = std.testing.allocator;
    const nh = shape.n_heads;
    const hd = shape.head_dim;
    const q = try gpa.alloc(f32, nh * hd);
    defer gpa.free(q);

    const kv_len = t_hi * shape.n_kv_heads * hd;
    const k = try gpa.alloc(E, kv_len);
    defer gpa.free(k);

    const v = try gpa.alloc(E, kv_len);
    defer gpa.free(v);

    const rec_len = attn_cpu.recordElems(@intCast(hd));
    // The row stride, so the records of the range lie apart as in a row.
    const rec_stride = attn_cpu.recordStride(@intCast(hd));
    const recs = try gpa.alloc(f32, nh * rec_stride);
    defer gpa.free(recs);

    const single = try gpa.alloc(f32, rec_len);
    defer gpa.free(single);

    // Run the whole head range at once.
    fillAttention(E, r, shape, t_hi, leads, sink_t, q, k, v);
    try expectLeadProfiles(E, q, k, shape, leads, sink_t, t_lo, t_hi);
    Kernels.attentionPartTyped(
        E,
        q,
        h_lo,
        h_hi,
        nh,
        hd,
        k,
        v,
        shape.n_kv_heads,
        t_lo,
        t_hi,
        recs,
        rec_stride,
    );

    // Run each head alone and compare the record bytes.
    for (h_lo..h_hi) |h| {
        Kernels.attentionPartTyped(
            E,
            q,
            h,
            h + 1,
            nh,
            hd,
            k,
            v,
            shape.n_kv_heads,
            t_lo,
            t_hi,
            single,
            rec_len,
        );
        const in_range = recs[(h - h_lo) * rec_stride ..][0..rec_len];
        try std.testing.expectEqualSlices(
            u8,
            std.mem.sliceAsBytes(single),
            std.mem.sliceAsBytes(in_range),
        );
    }
}

/// Fill one row, then demand that every kernel set in `kernel_sets` gives the bits of the
/// serial row of the first set, serial, on one thread, and fanned out at 2, 3, 4, and 16.
/// Garbage in the partial buffer before each call proves that the merge reads only fresh records.
pub fn expectFanOutMatchesRow(
    comptime kernel_sets: anytype,
    comptime E: type,
    r: std.Random,
    shape: AttnShape,
    t_len: usize,
) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const nh = shape.n_heads;
    const hd = shape.head_dim;
    const q = try gpa.alloc(f32, nh * hd);
    defer gpa.free(q);

    const kv_len = t_len * shape.n_kv_heads * hd;
    const k = try gpa.alloc(E, kv_len);
    defer gpa.free(k);

    const v = try gpa.alloc(E, kv_len);
    defer gpa.free(v);

    const parts = try gpa.alloc(f32, attn_cpu.partialLen(@intCast(nh), @intCast(hd)));
    defer gpa.free(parts);

    const out_row = try gpa.alloc(f32, nh * hd);
    defer gpa.free(out_row);

    const out = try gpa.alloc(f32, nh * hd);
    defer gpa.free(out);

    // Take the serial row of the first set as the expected bits.
    fillAttention(E, r, shape, t_len, &.{}, 0, q, k, v);

    kernel_sets[0].attentionRowTyped(E, q, nh, hd, k, v, shape.n_kv_heads, t_len, parts, out_row);

    // Check every set, serial first and then fanned out.
    const thread_counts = [_]u32{ 1, 2, 3, 4, 16 };
    inline for (kernel_sets) |Kernels| {
        for (parts) |*x| {
            x.* = r.float(f32) * 200.0 - 100.0;
        }

        Kernels.attentionRowTyped(E, q, nh, hd, k, v, shape.n_kv_heads, t_len, parts, out);
        try std.testing.expectEqualSlices(
            u8,
            std.mem.sliceAsBytes(out_row),
            std.mem.sliceAsBytes(out),
        );
        for (thread_counts) |n_threads| {
            for (parts) |*x| {
                x.* = r.float(f32) * 200.0 - 100.0;
            }

            @memset(out, 0);
            var pool: Pool = undefined;
            try pool.init(io, n_threads, 50_000);
            defer pool.deinit();

            const fanout: parallel.Policy = .{ .n_threads = n_threads, .min_parallel_flops = 0 };
            parallel.attentionTyped(
                E,
                Kernels,
                &pool,
                fanout,
                q,
                nh,
                hd,
                k,
                v,
                shape.n_kv_heads,
                t_len,
                parts,
                out,
            );
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(out_row),
                std.mem.sliceAsBytes(out),
            );
        }
    }
}

// -- helpers ----------------------------------------------------------------

/// Test reference: the full attention row in f64, with exact f16 widening.
/// So the gap between an f32 kernel and this reference is the kernel error.
fn attentionRefF64(
    comptime E: type,
    q: []const f32,
    n_heads: usize,
    head_dim: usize,
    k_cache: []const E,
    v_cache: []const E,
    n_kv_heads: usize,
    t_len: usize,
    scores: []f64,
    out: []f64,
) void {
    // Set the head grouping and the scale.
    const group_size = n_heads / n_kv_heads;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));
    for (0..n_heads) |h| {
        const kv_h = h / group_size;
        const q_h = q[h * head_dim ..][0..head_dim];

        // Score each position.
        for (scores[0..t_len], 0..) |*s, t| {
            const k_row = k_cache[(t * n_kv_heads + kv_h) * head_dim ..][0..head_dim];
            var dot: f64 = 0;
            for (q_h, k_row) |qd, kd| {
                dot += @as(f64, qd) * @as(f64, kd);
            }

            s.* = dot * scale;
        }

        // Softmax the scores.
        var m = scores[0];
        for (scores[1..t_len]) |s| {
            m = @max(m, s);
        }

        var sum: f64 = 0;
        for (scores[0..t_len]) |*s| {
            s.* = @exp(s.* - m);
            sum += s.*;
        }

        // Sum the values by weight.
        const out_h = out[h * head_dim ..][0..head_dim];
        @memset(out_h, 0);
        for (scores[0..t_len], 0..) |e, t| {
            const v_row = v_cache[(t * n_kv_heads + kv_h) * head_dim ..][0..head_dim];
            const w = e / sum;
            for (out_h, v_row) |*o, vd| {
                o.* += w * @as(f64, vd);
            }
        }
    }
}

/// Test reference: the full f32 attention row of every query head, one head at a time.
/// It keeps the whole score row in `scratch`, runs one softmax, and blocks no group.
/// The row tests hold it to the same f64 bound as the part row.
fn attentionRowScalarOracle(
    comptime E: type,
    q: []const f32,
    n_heads: usize,
    head_dim: usize,
    k_cache: []const E,
    v_cache: []const E,
    n_kv_heads: usize,
    t_len: usize,
    scratch: []f32,
    out: []f32,
) void {
    // Set the head grouping and the scale.
    const group_size = n_heads / n_kv_heads;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    var h: usize = 0;
    while (h < n_heads) : (h += 1) {
        const kv_h = h / group_size;
        const q_off = h * head_dim;

        // Score each position.
        var t: usize = 0;
        while (t < t_len) : (t += 1) {
            const k_off = t * n_kv_heads * head_dim + kv_h * head_dim;
            var s: f32 = 0.0;
            var d: usize = 0;
            while (d < head_dim) : (d += 1) {
                s += q[q_off + d] * @as(f32, @floatCast(k_cache[k_off + d]));
            }

            scratch[t] = s * scale;
        }

        // Softmax, then sum the values by weight.
        softmaxInplace(scratch[0..t_len]);
        const out_off = h * head_dim;
        var d: usize = 0;
        while (d < head_dim) : (d += 1) {
            out[out_off + d] = 0.0;
        }

        t = 0;
        while (t < t_len) : (t += 1) {
            const v_off = t * n_kv_heads * head_dim + kv_h * head_dim;
            const w = scratch[t];
            d = 0;
            while (d < head_dim) : (d += 1) {
                out[out_off + d] += w * @as(f32, @floatCast(v_cache[v_off + d]));
            }
        }
    }
}

/// Check that each lead head has the score profile of its lead over
/// `[t_lo, t_hi)`, so that a test reaches the fold path it names.
fn expectLeadProfiles(
    comptime E: type,
    q: []const f32,
    k: []const E,
    shape: AttnShape,
    leads: []const Lead,
    sink_t: usize,
    t_lo: usize,
    t_hi: usize,
) !void {
    for (leads, 0..) |lead, h| {
        switch (lead) {
            .none => {},
            .ramp => {
                // Every tile raises the maximum, so each tile after the first rescales.
                var before = -std.math.inf(f64);
                var tile_lo = t_lo;
                while (tile_lo < t_hi) : (tile_lo += cpu_tile) {
                    var m = -std.math.inf(f64);
                    for (tile_lo..@min(t_hi, tile_lo + cpu_tile)) |t| {
                        m = @max(m, scoreF64(E, q, k, shape, h, t));
                    }

                    try std.testing.expect(m > before);
                    before = m;
                }
            },
            .sink => {
                // The maximum lies at `sink_t`, so no later tile rescales.
                const top = scoreF64(E, q, k, shape, h, sink_t);
                for (t_lo..t_hi) |t| {
                    if (t != sink_t) {
                        try std.testing.expect(scoreF64(E, q, k, shape, h, t) < top);
                    }
                }
            },
            .large => {
                var m = -std.math.inf(f64);
                for (t_lo..t_hi) |t| {
                    m = @max(m, scoreF64(E, q, k, shape, h, t));
                }

                try std.testing.expect(m > 99.0);
                try std.testing.expect(m < 101.0);
            },
        }
    }
}

fn maxAbsErr(got: []const f32, want: []const f64) f64 {
    var worst: f64 = 0;
    for (got, want) |g, w| {
        worst = @max(worst, @abs(@as(f64, g) - w));
    }

    return worst;
}

/// In-place softmax over the whole slice, shifted by the maximum. Only the oracle runs it.
fn softmaxInplace(x: []f32) void {
    if (x.len == 0) return;

    if (std.debug.runtime_safety) {
        for (x) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }

    var max: f32 = x[0];
    for (x[1..]) |v| {
        if (v > max) {
            max = v;
        }
    }

    var sum: f32 = 0.0;
    for (x) |*v| {
        v.* = std.math.exp(v.* - max);
        sum += v.*;
    }

    const inv_sum = 1.0 / sum;
    for (x) |*v| {
        v.* *= inv_sum;
    }

    if (std.debug.runtime_safety) {
        for (x) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

fn scoreF64(
    comptime E: type,
    q: []const f32,
    k: []const E,
    shape: AttnShape,
    h: usize,
    t: usize,
) f64 {
    const hd = shape.head_dim;
    const kv_h = h / (shape.n_heads / shape.n_kv_heads);
    const k_row = k[(t * shape.n_kv_heads + kv_h) * hd ..][0..hd];
    var dot: f64 = 0;
    for (q[h * hd ..][0..hd], k_row) |qd, kd| {
        dot += @as(f64, qd) * @as(f64, kd);
    }

    return dot / @sqrt(@as(f64, @floatFromInt(hd)));
}

// -- tests ------------------------------------------------------------------

test "softmax sums to 1 and is monotone in input" {
    var x = [_]f32{ 1, 2, 3 };
    softmaxInplace(&x);
    const total = x[0] + x[1] + x[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-6);
    try std.testing.expect(x[0] < x[1]);
    try std.testing.expect(x[1] < x[2]);
}
