//! The one order in which the SIMD kernels of this directory add the 8 lanes of a
//! `@Vector(8, f32)` into one f32. A kernel sums its values in a vector accumulator. Its last
//! step adds the 8 lanes of that accumulator. The order of these adds decides the low bits of the
//! result. Every kernel of `simd` and `simd_pf` that adds lanes calls this file for that step.
//! So these kernels add the lanes in one order.
//!
//! Lane tree. The order is `((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))`: neighbor lanes
//! first, then neighbor pair sums, then the two 4-lane halves. `laneSum` adds the lanes of one
//! accumulator. `transposeAdd` adds the lanes of up to 8 accumulators in one call. Lane i of its
//! result is the sum of accumulator i.
//!
//! Comptime. `transposeAdd` takes the accumulator count `m` at compile time, so each count unrolls
//! to its own shuffles, and an `m` over 8 fails the build.
//!
//! Bit rule. An accumulator gets the same sum, bit for bit, from `laneSum` and from
//! `transposeAdd` at any accumulator count. `transposeAdd` sets the unused accumulators to zero,
//! and a lane of one accumulator never adds into the sum of another.
//!
//! Bit rule of the dot and the sum of squares. `sumSquares(x)` is `dot(x, x)`. `dot` reads 8
//! elements at a time and keeps 4 accumulators. Vector number n adds into accumulator `n % 4`
//! with one fused multiply-add. `dot` pads a last vector under 8 elements with zeros, so every
//! element takes the vector arithmetic. At the end, `dot` adds the accumulators lane by lane as
//! `(acc0 + acc1) + (acc2 + acc3)` and calls `laneSum`. So the order of every add depends on
//! the length alone, and the result has the same bits at any address of the slices.
//!
//! Cost. `transposeAdd` of 8 accumulators takes 6 pair sums and one add of the two halves. So
//! the 8 sums share the shuffles and the adds.

const std = @import("std");
const exp_simd = @import("exp_simd.zig");

const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

const lanes = exp_simd.lanes;

/// The accumulators of `dot` and `sumSquares`. The count is part of the bit rule, so a change
/// moves bits. A norm runs one sum over the hidden values of one row, which are in cache, so the
/// sum is a small part of a step. More accumulators gain little there, and the count stays at 4,
/// as in XNNPACK `rsum2`.
const square_accs = 4;

// -- types ------------------------------------------------------------------

const Vec = exp_simd.Vec;

// -- api --------------------------------------------------------------------

/// Add the 8 lanes of each of the `m` accumulators in the lane tree order. Lane i of the result
/// is the sum of `accs[i]`, with the same bits at any `m`.
pub inline fn transposeAdd(comptime m: usize, accs: [m]Vec) Vec {
    comptime assert(m >= 1 and m <= lanes);
    var a = [_]Vec{@splat(0)} ** lanes;
    inline for (0..m) |i| {
        a[i] = accs[i];
    }

    // Lane i of `low4` sums the low 4 lanes of accumulator i, and lane i + 4 sums its high 4.
    // `high4` does the same for accumulators 4 to 7.
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

/// The sums of the neighbor lanes of two vectors, inside each 128-bit half:
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

/// The sum of the 8 lanes of `v` in the lane tree order.
pub inline fn laneSum(v: Vec) f32 {
    return transposeAdd(1, .{v})[0];
}

/// The sum of `x[i] * x[i]` over `x`. The order of every add depends on `x.len` alone.
pub fn sumSquares(x: []const f32) f32 {
    assert(x.len > 0);

    // The inline call keeps the whole body under the name of `sumSquares` for the probe.
    const total = @call(.always_inline, dot, .{ x, x });
    assert(!(total < 0));
    return total;
}

/// The sum of `a[i] * b[i]`. The order of every add depends on `a.len` alone.
pub fn dot(a: []const f32, b: []const f32) f32 {
    assert(a.len > 0);
    assert(b.len == a.len);

    const stride = square_accs * lanes;
    var acc = [_]Vec{@splat(0)} ** square_accs;
    var i: usize = 0;
    while (i + stride <= a.len) : (i += stride) {
        inline for (0..square_accs) |r| {
            const av: Vec = a[i + r * lanes ..][0..lanes].*;
            const bv: Vec = b[i + r * lanes ..][0..lanes].*;
            acc[r] = @mulAdd(Vec, av, bv, acc[r]);
        }
    }

    // The tail holds fewer than 4 vectors, so `r` stays under `square_accs`.
    var r: usize = 0;
    while (i + lanes <= a.len) : ({
        i += lanes;
        r += 1;
    }) {
        const av: Vec = a[i..][0..lanes].*;
        const bv: Vec = b[i..][0..lanes].*;
        acc[r] = @mulAdd(Vec, av, bv, acc[r]);
    }

    if (i < a.len) {
        assert(r < square_accs);
        var pad_a: [lanes]f32 = @splat(0);
        var pad_b: [lanes]f32 = @splat(0);
        @memcpy(pad_a[0 .. a.len - i], a[i..]);
        @memcpy(pad_b[0 .. b.len - i], b[i..]);
        acc[r] = @mulAdd(Vec, pad_a, pad_b, acc[r]);
    }

    return laneSum((acc[0] + acc[1]) + (acc[2] + acc[3]));
}

// -- tests ------------------------------------------------------------------

/// The lane tree in scalar f32 arithmetic.
fn treeSum(x: [lanes]f32) f32 {
    return ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
}

/// The bits of `x`, so that an equality check compares bits and not values.
fn bits(x: f32) u32 {
    return @bitCast(x);
}

/// Check one accumulator count: lane i of `transposeAdd` equals `laneSum(accs[i])` and the
/// scalar lane tree of `accs[i]`, bit for bit.
fn checkTransposeAdd(comptime m: usize, r: std.Random) !void {
    var accs: [m]Vec = undefined;
    for (&accs) |*acc| {
        var x: [lanes]f32 = undefined;
        // Magnitudes from 2^-12 to 2^12 make a regrouped sum round differently.
        for (&x) |*xj| {
            const scale = std.math.ldexp(@as(f32, 1), r.intRangeAtMost(i32, -12, 12));
            xj.* = (r.float(f32) * 2 - 1) * scale;
        }

        acc.* = x;
    }

    const sums: [lanes]f32 = transposeAdd(m, accs);
    for (accs, 0..) |acc, i| {
        try std.testing.expectEqual(bits(laneSum(acc)), bits(sums[i]));
        try std.testing.expectEqual(bits(treeSum(acc)), bits(sums[i]));
    }

    for (sums[m..]) |pad| {
        try std.testing.expectEqual(bits(0), bits(pad));
    }
}

test "lane tree: a lane gets the bits of laneSum and of the scalar tree at every m" {
    var prng = std.Random.DefaultPrng.init(0x1A4E_7EE0);
    const r = prng.random();
    for (0..256) |_| {
        inline for (1..lanes + 1) |m| {
            try checkTransposeAdd(m, r);
        }
    }
}

/// The order of `dot` in scalar f32 arithmetic. Element i adds into lane `i % 8` of
/// accumulator `(i / 8) % 4`. A pad element adds `0 * 0`, which changes no accumulator.
fn dotScalar(x: []const f32, y: []const f32) f32 {
    var acc = [_][lanes]f32{@splat(0)} ** square_accs;
    for (x, y, 0..) |xi, yi, i| {
        const a = (i / lanes) % square_accs;
        acc[a][i % lanes] = @mulAdd(f32, xi, yi, acc[a][i % lanes]);
    }

    var lane_sums: [lanes]f32 = undefined;
    for (&lane_sums, 0..) |*f, l| {
        f.* = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    }

    return treeSum(lane_sums);
}

const square_lengths = [_]usize{ 1, 7, 8, 9, 31, 32, 33, 128, 576, 2048, 6144, 6145 };

test "lane tree: sumSquares stays inside the gamma bound of an f64 sum" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5A0A_5E00);
    const r = prng.random();
    const buf = try gpa.alloc(f32, square_lengths[square_lengths.len - 1]);
    defer gpa.free(buf);
    const u = std.math.ldexp(@as(f64, 1), -24);
    for (square_lengths) |len| {
        const x = buf[0..len];
        var want: f64 = 0;
        for (x) |*xi| {
            xi.* = (r.float(f32) * 2 - 1) * 4;
            want += @as(f64, xi.*) * @as(f64, xi.*);
        }

        // A lane takes one rounded multiply-add per 32 elements. Then a value passes 2 adds of
        // the 4 accumulators and 3 adds of the lane tree (Higham, chapter 4).
        const n: f64 = @floatFromInt((len + square_accs * lanes - 1) / (square_accs * lanes) + 5);
        const gamma = n * u / (1 - n * u);
        const got: f64 = sumSquares(x);
        try std.testing.expect(@abs(got - want) <= gamma * want);
    }
}

test "lane tree: sumSquares gives the bits of its scalar order at any alignment" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5A0A_B175);
    const r = prng.random();
    const max_len = square_lengths[square_lengths.len - 1];
    const src = try gpa.alloc(f32, max_len);
    defer gpa.free(src);
    const dst = try gpa.alloc(f32, max_len + lanes);
    defer gpa.free(dst);
    for (src) |*s| {
        s.* = (r.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), r.intRangeAtMost(i32, -8, 8));
    }

    for (square_lengths) |len| {
        const want = bits(dotScalar(src[0..len], src[0..len]));
        for (0..lanes) |off| {
            @memcpy(dst[off..][0..len], src[0..len]);
            try std.testing.expectEqual(want, bits(sumSquares(dst[off..][0..len])));
        }
    }
}

test "lane tree: dot of two vectors gives the bits of its scalar order" {
    var prng = std.Random.DefaultPrng.init(0x6D17_5400);
    const r = prng.random();
    var x_buf: [200]f32 = undefined;
    var y_buf: [200]f32 = undefined;
    for ([_]usize{ 1, 7, 8, 12, 31, 32, 33, 41, 128, 200 }) |len| {
        const x = x_buf[0..len];
        const y = y_buf[0..len];
        for (x, y) |*xi, *yi| {
            xi.* = (r.float(f32) * 2 - 1) * 4;
            yi.* = (r.float(f32) * 2 - 1) * 4;
        }

        try std.testing.expectEqual(bits(dotScalar(x, y)), bits(dot(x, y)));
        try std.testing.expectEqual(bits(sumSquares(x)), bits(dot(x, x)));
    }
}
