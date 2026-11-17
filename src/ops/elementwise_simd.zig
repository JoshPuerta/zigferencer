//! The vector elementwise ops of both SIMD backends: `residualAdd`, `rmsnorm`, `swiglu`, and
//! `sigmoidGate`.
//!
//! Bit rule. Every element takes the vector arithmetic. A tail under 8 lanes is padded with zeros
//! into one vector, and a pad lane is never stored. Each lane is a function of the inputs of its
//! own element, plus the one scale that `rmsnorm` computes per row. So an element gets the same
//! bits at any lane position, slice alignment, unit boundary, and thread count. A
//! multiply-accumulate is one `@mulAdd`, a sum of 8 lanes takes the lane tree order of
//! `lane_tree_simd`, and a sigmoid is `exp_simd.sigmoid`.
//!
//! Against naive. `residualAdd` runs the one add of the scalar loop in each lane, so it gives the
//! bits of `naive.residualAdd`. The other ops change the arithmetic. `rmsnorm` sums the squares
//! with `lane_tree_simd.sumSquares`. `swiglu` computes `(g * sigmoid(g)) * u`, and `sigmoidGate`
//! computes `o * sigmoid(g)`, both with the vector sigmoid. `naive.swiglu` calls
//! `std.math.exp`, and `naive.sigmoidGate` calls `@exp`, the `expf` that the build links. The
//! tests bound each op against an f64 form.
//!
//! Aliasing. The input and the output are the same memory or do not overlap. With a partial
//! overlap, a lane can read an element that an earlier store changed.

const std = @import("std");
const exp_simd = @import("exp_simd.zig");
const lane_tree_simd = @import("lane_tree_simd.zig");
const naive = @import("naive.zig");

const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

const lanes = exp_simd.lanes;

/// The weight of one `swiglu` element in FLOPs of the `simd_pf` Q8_0 matmul, the op of the floor
/// sweep of `parallel`. One vector element takes as long as this many FLOPs of that matmul. The
/// value is the mean rate of the `linearF32Q80` `simd_pf` lines over the mean rate of the `swiglu`
/// `simd_pf` lines of one ReleaseFast `zig build bench`, rounded to an integer.
pub const swiglu_flops_per_element: u64 = 16;

// -- types ------------------------------------------------------------------

const Vec = exp_simd.Vec;

// -- api --------------------------------------------------------------------

/// dst[i] += x[i]
pub fn residualAdd(x: []const f32, dst: []f32) void {
    assert(x.len <= dst.len);
    assert(sameOrDisjoint(x, dst[0..x.len]));

    var i: usize = 0;
    while (i + lanes <= x.len) : (i += lanes) {
        const xv: Vec = x[i..][0..lanes].*;
        const dv: Vec = dst[i..][0..lanes].*;
        dst[i..][0..lanes].* = dv + xv;
    }

    if (i < x.len) {
        storeTail(dst[0..x.len], i, loadTail(dst[0..x.len], i) + loadTail(x, i));
    }

    assertFinite(dst[0..x.len]);
}

/// y[i] = (x[i] / rms(x)) * w[i],  rms(x) = sqrt(mean(x^2) + eps). `dst` can be `x`.
pub fn rmsnorm(x: []const f32, w: []const f32, eps: f32, dst: []f32) void {
    assert(x.len > 0);
    assert(x.len == w.len);
    assert(dst.len >= x.len);
    assert(sameOrDisjoint(x, dst[0..x.len]));

    // The scale takes the expression of naive, so only the sum of squares changes order.
    const sum_sq = lane_tree_simd.sumSquares(x);
    const inv_rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    const scale: Vec = @splat(inv_rms);
    var i: usize = 0;
    while (i + lanes <= x.len) : (i += lanes) {
        const v: Vec = x[i..][0..lanes].*;
        const wv: Vec = w[i..][0..lanes].*;
        dst[i..][0..lanes].* = v * scale * wv;
    }

    if (i < x.len) {
        storeTail(dst[0..x.len], i, loadTail(x, i) * scale * loadTail(w, i));
    }

    assertFinite(dst[0..x.len]);
}

/// dst[i] = silu(gate[i]) * up[i]. `dst` can be `gate`.
pub fn swiglu(gate: []const f32, up: []const f32, dst: []f32) void {
    assert(gate.len == up.len);
    assert(dst.len >= gate.len);
    assert(sameOrDisjoint(gate, dst[0..gate.len]));
    assert(sameOrDisjoint(up, dst[0..gate.len]));

    var i: usize = 0;
    while (i + lanes <= gate.len) : (i += lanes) {
        const g: Vec = gate[i..][0..lanes].*;
        const u: Vec = up[i..][0..lanes].*;
        dst[i..][0..lanes].* = (g * exp_simd.sigmoid(g)) * u;
    }

    if (i < gate.len) {
        const g = loadTail(gate, i);
        storeTail(dst[0..gate.len], i, (g * exp_simd.sigmoid(g)) * loadTail(up, i));
    }

    assertFinite(dst[0..gate.len]);
}

/// out[i] *= sigmoid(gate[i])
pub fn sigmoidGate(gate: []const f32, out: []f32) void {
    assert(gate.len == out.len);
    assert(sameOrDisjoint(gate, out));

    var i: usize = 0;
    while (i + lanes <= out.len) : (i += lanes) {
        const g: Vec = gate[i..][0..lanes].*;
        const o: Vec = out[i..][0..lanes].*;
        out[i..][0..lanes].* = o * exp_simd.sigmoid(g);
    }

    if (i < out.len) {
        storeTail(out, i, loadTail(out, i) * exp_simd.sigmoid(loadTail(gate, i)));
    }

    assertFinite(out);
}

// -- helpers ----------------------------------------------------------------

fn sameOrDisjoint(x: []const f32, y: []const f32) bool {
    const x_lo = @intFromPtr(x.ptr);
    const y_lo = @intFromPtr(y.ptr);
    if (x_lo == y_lo) return true;

    return x_lo + x.len * @sizeOf(f32) <= y_lo or y_lo + y.len * @sizeOf(f32) <= x_lo;
}

/// The last `x.len - i` elements of `x` in the low lanes, and zeros in the pad lanes.
inline fn loadTail(x: []const f32, i: usize) Vec {
    assert(i < x.len);
    assert(x.len - i < lanes);
    var pad: [lanes]f32 = @splat(0);
    @memcpy(pad[0 .. x.len - i], x[i..]);
    return pad;
}

/// Store the low `dst.len - i` lanes of `v` at `dst[i..]`. The pad lanes stay out of memory.
inline fn storeTail(dst: []f32, i: usize, v: Vec) void {
    assert(i < dst.len);
    assert(dst.len - i < lanes);
    const all: [lanes]f32 = v;
    @memcpy(dst[i..], all[0 .. dst.len - i]);
}

fn assertFinite(x: []const f32) void {
    if (!std.debug.runtime_safety) return;

    for (x) |v| {
        assert(std.math.isFinite(v));
    }
}

// -- tests ------------------------------------------------------------------

const test_lengths = [_]usize{ 1, 7, 8, 9, 6144, 6145 };
const max_test_len = test_lengths[test_lengths.len - 1];

/// The unit roundoff of f32, `2^-24`.
const unit_roundoff = std.math.ldexp(@as(f64, 1), -24);

/// The relative error of the f64 forms of the tests. Each element takes at most 6 f64 roundings
/// plus one libm `exp` or `sqrt`, and `2^-48` covers them with a wide margin. The f64 sum of
/// squares of `rmsnorm` also rounds n times, and its test adds `n * 2^-53` for it.
const f64_slack = std.math.ldexp(@as(f64, 1), -48);

/// The relative error of one `exp_simd.sigmoid` lane. One f32 ulp of the exact result `y` is at
/// most `2^-23 * y`, which is `2u * y` with `u = 2^-24`.
const sigmoid_rel = 2 * exp_simd.sigmoid_ulp_tol * unit_roundoff;

/// The bits of `x`, so that an equality check compares bits and not values.
fn bits(x: f32) u32 {
    return @bitCast(x);
}

/// A grid of `x.len` points from -20 to 20. One point lands on -20.
fn fillGateGrid(x: []f32) void {
    const den: f32 = @floatFromInt(@max(x.len, 2) - 1);
    for (x, 0..) |*v, i| {
        v.* = -20 + 40 * @as(f32, @floatFromInt(i)) / den;
    }
}

fn fillUniform(r: std.Random, x: []f32) void {
    for (x) |*v| {
        v.* = r.float(f32) * 4 - 2;
    }
}

fn sigmoid64(x: f32) f64 {
    return 1 / (1 + @exp(-@as(f64, x)));
}

/// Check `|got - want| <= rel * |want|` for each element, with the f64 slack added to `rel`.
fn expectRelBound(got: []const f32, want: []const f64, rel: f64) !void {
    for (got, want) |g, w| {
        try std.testing.expect(@abs(@as(f64, g) - w) <= (rel + f64_slack) * @abs(w));
    }
}

test "elementwise simd: residualAdd is bit-identical to naive.residualAdd" {
    var prng = std.Random.DefaultPrng.init(0xadd);
    const rng = prng.random();
    var x: [6145]f32 = undefined;
    var got: [6145]f32 = undefined;
    var want: [6145]f32 = undefined;
    for ([_]usize{ 1, 7, 8, 9, 576, 6145 }) |n| {
        for (x[0..n], got[0..n]) |*a, *b| {
            a.* = rng.float(f32) * 4.0 - 2.0;
            b.* = rng.float(f32) * 4.0 - 2.0;
        }

        @memcpy(want[0..n], got[0..n]);
        naive.residualAdd(x[0..n], want[0..n]);
        residualAdd(x[0..n], got[0..n]);
        for (got[0..n], want[0..n]) |a, b| {
            try std.testing.expectEqual(bits(b), bits(a));
        }
    }
}

test "elementwise simd: residualAdd doubles a vector that is its own input" {
    var got: [13]f32 = undefined;
    var want: [13]f32 = undefined;
    for (&got, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.37 - 2.0;
    }

    @memcpy(&want, &got);
    naive.residualAdd(&want, &want);
    residualAdd(&got, &got);
    for (got, want) |a, b| {
        try std.testing.expectEqual(bits(b), bits(a));
    }
}

test "elementwise simd: swiglu and sigmoidGate stay inside the bound of an f64 form" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5161_6A7E);
    const r = prng.random();
    const gate = try gpa.alloc(f32, max_test_len);
    defer gpa.free(gate);
    const other = try gpa.alloc(f32, max_test_len);
    defer gpa.free(other);
    const got = try gpa.alloc(f32, max_test_len);
    defer gpa.free(got);
    const want = try gpa.alloc(f64, max_test_len);
    defer gpa.free(want);

    // The sigmoid errs by at most `sigmoid_rel`, and each multiply after it rounds once by at
    // most u. `swiglu` takes two multiplies and `sigmoidGate` takes one (Higham, chapter 2).
    // The bounds hold from the sigmoid cut up, about -87.34, and the grid stays above it. Under
    // the cut the sigmoid gives 0, so `swiglu` gives -0 where the exact result is a tiny normal.
    // That absolute error is under `92 * 2^-126 * |u|`.
    const u = unit_roundoff;
    const swiglu_rel = (1 + sigmoid_rel) * (1 + u) * (1 + u) - 1;
    const gate_rel = (1 + sigmoid_rel) * (1 + u) - 1;
    for (test_lengths) |n| {
        fillGateGrid(gate[0..n]);
        fillUniform(r, other[0..n]);

        swiglu(gate[0..n], other[0..n], got[0..n]);
        for (gate[0..n], other[0..n], want[0..n]) |g, o, *w| {
            w.* = @as(f64, g) * sigmoid64(g) * @as(f64, o);
        }

        try expectRelBound(got[0..n], want[0..n], swiglu_rel);

        @memcpy(got[0..n], other[0..n]);
        sigmoidGate(gate[0..n], got[0..n]);
        for (gate[0..n], other[0..n], want[0..n]) |g, o, *w| {
            w.* = @as(f64, o) * sigmoid64(g);
        }

        try expectRelBound(got[0..n], want[0..n], gate_rel);
    }
}

test "elementwise simd: rmsnorm stays inside the bound of an f64 form" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0A15_0A40);
    const r = prng.random();
    const x = try gpa.alloc(f32, max_test_len);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, max_test_len);
    defer gpa.free(w);
    const got = try gpa.alloc(f32, max_test_len);
    defer gpa.free(got);
    const want = try gpa.alloc(f64, max_test_len);
    defer gpa.free(want);

    const eps: f32 = 1e-6;
    const u = unit_roundoff;
    for (test_lengths) |n| {
        fillUniform(r, x[0..n]);
        fillUniform(r, w[0..n]);
        rmsnorm(x[0..n], w[0..n], eps, got[0..n]);

        var sum_sq: f64 = 0;
        for (x[0..n]) |v| {
            sum_sq += @as(f64, v) * @as(f64, v);
        }

        const inv_rms = 1 / @sqrt(sum_sq / @as(f64, @floatFromInt(n)) + @as(f64, eps));
        for (x[0..n], w[0..n], want[0..n]) |v, wv, *y| {
            y.* = @as(f64, v) * inv_rms * @as(f64, wv);
        }

        // `sumSquares` errs by at most gamma(k) with the k of its own test. The division by n
        // and the add of eps round once each, and eps only shrinks the relative error. So the
        // mean errs by at most `a`. The inverse square root turns `1 + a` into at most
        // `1 / sqrt(1 - a)`. The sqrt, the division, and the two multiplies round once each.
        const k: f64 = @floatFromInt((n + 31) / 32 + 5);
        const gamma = k * u / (1 - k * u);
        const a = (1 + gamma) * (1 + u) * (1 + u) - 1;
        const rel = (1 / @sqrt(1 - a)) * (1 + u) * (1 + u) * (1 + u) / (1 - u) - 1;
        const f64_sum = @as(f64, @floatFromInt(n)) * std.math.ldexp(@as(f64, 1), -53);
        try expectRelBound(got[0..n], want[0..n], rel + f64_sum);
    }
}

test "elementwise simd: swiglu and sigmoidGate give an element the same bits at any lane" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1A4E_5417);
    const r = prng.random();
    const len = max_test_len + lanes;
    const gate = try gpa.alloc(f32, len);
    defer gpa.free(gate);
    const other = try gpa.alloc(f32, len);
    defer gpa.free(other);
    const want = try gpa.alloc(f32, len);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, len);
    defer gpa.free(got);

    // `off` leading elements move element j of the reference call to lane `(j + off) % 8`.
    for (test_lengths) |n| {
        fillGateGrid(gate[lanes..][0..n]);
        fillUniform(r, other[lanes..][0..n]);
        fillUniform(r, gate[0..lanes]);
        fillUniform(r, other[0..lanes]);
        swiglu(gate[lanes..][0..n], other[lanes..][0..n], want[0..n]);
        for (1..lanes) |off| {
            const lo = lanes - off;
            swiglu(gate[lo..][0 .. n + off], other[lo..][0 .. n + off], got[0 .. n + off]);
            for (want[0..n], got[off..][0..n]) |a, b| {
                try std.testing.expectEqual(bits(a), bits(b));
            }
        }

        @memcpy(want[0..n], other[lanes..][0..n]);
        sigmoidGate(gate[lanes..][0..n], want[0..n]);
        for (1..lanes) |off| {
            const lo = lanes - off;
            @memcpy(got[0 .. n + off], other[lo..][0 .. n + off]);
            sigmoidGate(gate[lo..][0 .. n + off], got[0 .. n + off]);
            for (want[0..n], got[off..][0..n]) |a, b| {
                try std.testing.expectEqual(bits(a), bits(b));
            }
        }
    }
}

test "elementwise simd: rmsnorm gives the same bits at any alignment and in place" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA11C_0A40);
    const r = prng.random();
    const len = max_test_len + lanes;
    const x = try gpa.alloc(f32, max_test_len);
    defer gpa.free(x);
    const w = try gpa.alloc(f32, max_test_len);
    defer gpa.free(w);
    const want = try gpa.alloc(f32, max_test_len);
    defer gpa.free(want);
    const xs = try gpa.alloc(f32, len);
    defer gpa.free(xs);
    const ws = try gpa.alloc(f32, len);
    defer gpa.free(ws);
    const got = try gpa.alloc(f32, len);
    defer gpa.free(got);

    // A leading element changes the sum of squares, so the slice moves in memory and not in lanes.
    const eps: f32 = 1e-6;
    for (test_lengths) |n| {
        fillUniform(r, x[0..n]);
        fillUniform(r, w[0..n]);
        rmsnorm(x[0..n], w[0..n], eps, want[0..n]);
        for (1..lanes) |off| {
            @memcpy(xs[off..][0..n], x[0..n]);
            @memcpy(ws[off..][0..n], w[0..n]);
            rmsnorm(xs[off..][0..n], ws[off..][0..n], eps, got[off..][0..n]);
            for (want[0..n], got[off..][0..n]) |a, b| {
                try std.testing.expectEqual(bits(a), bits(b));
            }

            rmsnorm(xs[off..][0..n], ws[off..][0..n], eps, xs[off..][0..n]);
            for (want[0..n], xs[off..][0..n]) |a, b| {
                try std.testing.expectEqual(bits(a), bits(b));
            }
        }
    }
}
