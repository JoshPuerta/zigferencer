//! The vector exponential and sigmoid of the CPU datapath, `@Vector(8, f32)` lanes. `expNeg`
//! takes inputs at or below 0. CPU decode attention takes it for the online softmax fold, and the
//! sampler takes it for the full-vocab weights. `sigmoid` computes `1 / (1 + exp(-x))` for any
//! input. Each lane is a function of the input of that lane alone, so a value gets the same bits
//! at any lane position and in any caller.
//!
//! Callers. The attention fold takes `expNeg` for a score minus the running maximum, and for an
//! old maximum minus a new one. The sampler takes it for a scaled logit minus the maximum of the
//! logits. No NaN arrives. Each lane runs one fixed sequence of IEEE operations on its own input.
//!
//! Method. This is the method of Cody and Waite, with the Cephes `expf` constants of Moshier.
//! 1. Range reduction. `n = floor(x * log2(e) + 1/2)` is the integer nearest to `x / ln 2`. Then
//!    `r = x - n * ln 2` lies in `[-ln 2 / 2, ln 2 / 2]` up to one rounding, and
//!    `exp(x) = 2^n * exp(r)`.
//! 2. Two-part `ln 2`. `ln2_hi` holds 9 significant bits, so `n * ln2_hi` is exact for every `n`
//!    here, and the first step rounds nothing. `ln2_lo` holds the rest of `ln 2`, and the second
//!    step rounds once. A one-part f32 `ln 2` errs by 1.9e-9, which at `n = -126` moves the result
//!    by 2 to 4 ulps.
//! 3. The polynomial. `exp(r) = 1 + r + r^2 * P(r)`, with `P` of degree 5 in Horner form. The
//!    Cephes coefficients are a minimax fit with a relative error of at most 4.2e-9 on
//!    `[-1/2, 1/2]`. Each Horner step is one fused multiply-add, which rounds once.
//! 4. The scale. The f32 with the exponent field `n + 127` and a zero significand is `2^n`. The
//!    clamp to `exp_cut` holds `n` in `[-126, 0]`, so the field stays in `[1, 127]`. The product
//!    with `exp(r)` is then a normal f32 and exact.
//! 5. The cut. Below `exp_cut` the exact result is under the smallest normal f32, and the lane
//!    gives 0. So no result is subnormal, and `-inf` gives 0. `0` and `-0` give exactly 1, so the
//!    term at the maximum adds exactly 1.
//!
//! Error. From the cut to 0 the error against the f64 `exp` measures at most 1.0103 ulps, at
//! `0xc0bc17a1` (-5.8778844). Only 14 of those 1.1e9 inputs err by more than 1 ulp. `exp_ulp_tol`
//! rounds that maximum up.
//!
//! Sigmoid. This is the `rr2-p5-div` shape of the XNNPACK sigmoid kernels on `expNeg`, with the
//! two-constant range reduction of step 2.
//! 1. The negative half. `e = expNeg(-|x|)` and `f = e / (e + 1)` give `sigmoid(-|x|)`. No
//!    exponential overflows, and `f` lies in `[0, 1/2]`.
//! 2. The positive half. `sigmoid(x) = 1 - sigmoid(-x)`, so a lane with `x > 0` gives `1 - f`.
//! 3. The division. The quotient is a true IEEE division. A reciprocal estimate such as `vrcpps`
//!    gives other bits on another CPU vendor, so the sigmoid takes none.
//! 4. The ends. `0` and `-0` give exactly 1/2. Under `exp_cut` the exact result is under the
//!    smallest normal f32, and the lane gives 0. From 17.32868 (`0x418aa123`) up the lane gives
//!    exactly 1. A NaN lane gives its input, as the naive loop does.
//!
//! Sigmoid error. From `exp_cut` up the exact result is a normal f32. Over every f32 of that
//! domain the error against the f64 sigmoid measures at most 2.4019 ulps, at `0xc085088d`
//! (-4.157294). 6.7e6 of those 3.3e9 inputs err by more than 1 ulp. The error of `expNeg`, the
//! rounding of `e + 1`, and the rounding of the quotient add up. On the positive half the
//! rounding of `1 - f` adds to them. Under `exp_cut` the lane gives 0, and the absolute error is
//! under `2^-126`. `sigmoid_ulp_tol` rounds the maximum up.
//!
//! Monotonicity. The sigmoid is not monotone. Over every f32 in `[1, 17]` the result decreases
//! 12344 times from one input to the next, and over `(0, 1)` 75591 times. A caller must not
//! assume that a larger input gives a result that is not smaller.

const std = @import("std");
const prop = @import("../core/utils/prop.zig");

const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

pub const lanes = 8;

/// The smallest f32 whose exact exponential is a normal f32, `2^-126` times 1.0000045.
pub const exp_cut: f32 = @bitCast(@as(u32, 0xc2aeac4f));

/// The largest error of `expNeg` against the f64 `exp` that the tests accept, in ulps.
pub const exp_ulp_tol: f64 = 1.02;

/// The largest error of `sigmoid` against the f64 sigmoid that the tests accept, in ulps. One
/// sweep over every f32 from `exp_cut` up measured 2.4019 ulps, at `0xc085088d` (-4.157294).
pub const sigmoid_ulp_tol: f64 = 2.41;

// -- types ------------------------------------------------------------------

pub const Vec = @Vector(lanes, f32);

// -- api --------------------------------------------------------------------

/// `exp(x)` in each lane, for `x <= 0`. A NaN lane fails the safety check. A release build clamps
/// it to the cut, because `@max` returns the other operand, so every index stays defined.
/// The result is within `exp_ulp_tol` ulps of the f64 `exp`.
pub inline fn expNeg(x: Vec) Vec {
    const Int = @Vector(lanes, i32);
    const log2e: Vec = @splat(std.math.log2e);
    const ln2_hi: Vec = @splat(0.693359375);
    const ln2_lo: Vec = @splat(-2.12194440e-4);
    // The coefficients of `P`, the highest degree first.
    const coeffs = [_]f32{
        1.9875691500e-4,
        1.3981999507e-3,
        8.3334519073e-3,
        4.1665795894e-2,
        1.6666665459e-1,
        5.0000001201e-1,
    };
    const cut: Vec = @splat(exp_cut);
    if (std.debug.runtime_safety) assert(@reduce(.And, x <= @as(Vec, @splat(0))));

    // Reduce the input to r, with exp(x) = 2^n * exp(r).
    const xc = @max(x, cut);
    const n = @floor(@mulAdd(Vec, xc, log2e, @splat(0.5)));
    const r = @mulAdd(Vec, -n, ln2_lo, @mulAdd(Vec, -n, ln2_hi, xc));

    // Evaluate exp(r) with the Horner polynomial.
    var poly: Vec = @splat(coeffs[0]);
    inline for (coeffs[1..]) |c| {
        poly = @mulAdd(Vec, poly, r, @splat(c));
    }

    const exp_r = @mulAdd(Vec, poly, r * r, r) + @as(Vec, @splat(1));

    // Build 2^n from the exponent field, and cut the lanes under the cut.
    const n_int: Int = @intFromFloat(n);
    if (std.debug.runtime_safety) {
        assert(@reduce(.And, n_int >= @as(Int, @splat(-126))));
        assert(@reduce(.And, n_int <= @as(Int, @splat(0))));
    }

    const field: @Vector(lanes, u32) = @intCast(n_int + @as(Int, @splat(127)));
    const two_n: Vec = @bitCast(field << @splat(23));
    const y = @select(f32, x < cut, @as(Vec, @splat(0)), exp_r * two_n);
    // The attention fold and the sampler CDF need both ends of this range.
    if (std.debug.runtime_safety) {
        assert(@reduce(.And, y >= @as(Vec, @splat(0))));
        assert(@reduce(.And, y <= @as(Vec, @splat(1))));
    }

    return y;
}

/// `1 / (1 + exp(-x))` in each lane. A NaN lane fails the safety check. A release build returns
/// the NaN, as the naive loop does. From `exp_cut` up, the result is within `sigmoid_ulp_tol`
/// ulps of the f64 sigmoid. Under `exp_cut` the result is 0.
pub inline fn sigmoid(x: Vec) Vec {
    const one: Vec = @splat(1);
    if (std.debug.runtime_safety) assert(@reduce(.And, x == x));

    // The negative half is `e / (e + 1)` with `e = exp(-|x|)`, so no exponential overflows.
    const e = expNeg(-@abs(x));
    const f = e / (e + one);
    const y = @select(f32, x > @as(Vec, @splat(0)), one - f, f);
    if (std.debug.runtime_safety) {
        assert(@reduce(.And, y >= @as(Vec, @splat(0))));
        assert(@reduce(.And, y <= one));
    }

    // The clamp of `expNeg` turns a NaN lane into a number, so the select gives the NaN back.
    return @select(f32, x == x, y, x);
}

/// The error of one `expNeg` lane against the f64 `exp`, in ulps of the exact
/// result. From the cut to 0 the exact result is a normal f32.
pub fn expUlpErr(x: f32, got: f32) f64 {
    const want = @exp(@as(f64, x));
    // `want` lies in `[2^(e - 1), 2^e)`, where one f32 ulp is `2^(e - 24)`.
    const e = std.math.frexp(want).exponent;
    return @abs(@as(f64, got) - want) / std.math.ldexp(@as(f64, 1), e - 24);
}

/// The error of one `sigmoid` lane against the f64 sigmoid, in ulps of the exact result. From
/// `exp_cut` up the exact result is a normal f32.
pub fn sigmoidUlpErr(x: f32, got: f32) f64 {
    const want = 1 / (1 + @exp(-@as(f64, x)));
    // `want` lies in `[2^(e - 1), 2^e)`, where one f32 ulp is `2^(e - 24)`.
    const e = std.math.frexp(want).exponent;
    return @abs(@as(f64, got) - want) / std.math.ldexp(@as(f64, 1), e - 24);
}

// -- tests ------------------------------------------------------------------

/// The largest lane error of one `expNeg` call.
fn expWorstUlp(x: [lanes]f32) f64 {
    const y: [lanes]f32 = expNeg(x);
    var worst: f64 = 0;
    for (x, y) |xj, yj| {
        worst = @max(worst, expUlpErr(xj, yj));
    }

    return worst;
}

test "vector exp: expNeg stays inside the ulp bound from the cut to 0" {
    // The input of the largest error over every f32 of the domain.
    var worst = expWorstUlp(@splat(@bitCast(@as(u32, 0xc0bc17a1))));
    // A uniform grid of 2^20 points, both ends included.
    const n_grid: usize = 1 << 20;
    var i: usize = 0;
    while (i < n_grid) : (i += lanes) {
        var x: [lanes]f32 = undefined;
        for (&x, i..) |*xj, k| {
            xj.* = @floatCast(
                @as(f64, exp_cut) * @as(f64, @floatFromInt(k)) / @as(f64, n_grid - 1),
            );
        }

        worst = @max(worst, expWorstUlp(x));
    }

    // Every 127th f32 in [-1, 0] by bit pattern, from -0 through the
    // subnormals to -1. The odd stride varies the low significand bits.
    const neg_zero: u32 = 0x8000_0000;
    const neg_one: u32 = 0xbf80_0000;
    var b = neg_zero;
    while (b <= neg_one) {
        var x: [lanes]f32 = undefined;
        for (&x) |*xj| {
            xj.* = @bitCast(@min(b, neg_one));
            b += 127;
        }

        worst = @max(worst, expWorstUlp(x));
    }

    try std.testing.expect(worst <= exp_ulp_tol);
}

test "vector exp: 0 and -0 give 1, the cut stays normal, and every input under the cut gives 0" {
    const cut_bits: u32 = @bitCast(exp_cut);
    const above: f32 = @bitCast(cut_bits - 1);
    const below: f32 = @bitCast(cut_bits + 1);
    // The cut is tight: its exact exponential is normal, and that of the next
    // f32 down is not.
    try std.testing.expect(@exp(@as(f64, exp_cut)) >= std.math.floatMin(f32));
    try std.testing.expect(@exp(@as(f64, below)) < std.math.floatMin(f32));

    const x = [lanes]f32{ 0, -0.0, exp_cut, above, below, -88, -1000, -std.math.inf(f32) };
    const y: [lanes]f32 = expNeg(x);
    try std.testing.expectEqual(@as(f32, 1), y[0]);
    try std.testing.expectEqual(@as(f32, 1), y[1]);
    for (x[2..4], y[2..4]) |xj, yj| {
        try std.testing.expect(yj >= std.math.floatMin(f32));
        try std.testing.expect(expUlpErr(xj, yj) <= exp_ulp_tol);
    }

    for (y[4..]) |yj| {
        try std.testing.expectEqual(@as(f32, 0), yj);
    }
}

fn checkLaneBits(_: void, c: prop.Trial) anyerror!void {
    var prng = std.Random.DefaultPrng.init(c.seed);
    const r = prng.random();
    const edges = [lanes]f32{
        0,
        -0.0,
        exp_cut,
        -1.0e-40,
        -0.5 * @log(2.0),
        -88,
        -1000,
        -std.math.inf(f32),
    };
    const round = c.index;
    var src = edges;
    // Inputs from -100 to 0, so some lanes fall under the cut.
    if (round > 0) {
        for (&src) |*s| {
            s.* = -100.0 * r.float(f32);
        }
    }

    for (0..lanes) |rot| {
        var x: [lanes]f32 = undefined;
        for (&x, 0..) |*xj, j| {
            xj.* = src[(j + rot) % lanes];
        }

        const y: [lanes]f32 = expNeg(x);
        for (x, y) |xj, yj| {
            const alone: f32 = expNeg(@splat(xj))[0];
            try std.testing.expectEqual(@as(u32, @bitCast(alone)), @as(u32, @bitCast(yj)));
        }
    }
}

test "vector exp: a lane gives the bits of a splat of its input at every lane position" {
    try prop.forAll(
        .{ .site = 0xA770_00AB, .cases = 64, .max_size = 63 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkLaneBits,
    );
}

/// The largest lane error of one `sigmoid` call.
fn sigmoidWorstUlp(x: [lanes]f32) f64 {
    const y: [lanes]f32 = sigmoid(x);
    var worst: f64 = 0;
    for (x, y) |xj, yj| {
        worst = @max(worst, sigmoidUlpErr(xj, yj));
    }

    return worst;
}

test "vector sigmoid: sigmoid stays inside the ulp bound from the cut up" {
    // The input of the largest error over every f32 of the domain.
    const pinned = sigmoidWorstUlp(@splat(@bitCast(@as(u32, 0xc085088d))));
    try std.testing.expect(pinned > 2.4);
    var worst = pinned;
    // A uniform grid of 2^20 points from the cut to its mirror, both ends included.
    const n_grid: usize = 1 << 20;
    var i: usize = 0;
    while (i < n_grid) : (i += lanes) {
        var x: [lanes]f32 = undefined;
        for (&x, i..) |*xj, k| {
            const t = @as(f64, @floatFromInt(k)) / @as(f64, n_grid - 1);
            xj.* = @floatCast(@as(f64, exp_cut) * (1 - 2 * t));
        }

        worst = @max(worst, sigmoidWorstUlp(x));
    }

    try std.testing.expect(worst <= sigmoid_ulp_tol);
}

test "vector sigmoid: 0 and -0 give 1/2, the cut gives 0 below, and a large input gives 1" {
    const cut_bits: u32 = @bitCast(exp_cut);
    const below: f32 = @bitCast(cut_bits + 1);
    const inf = std.math.inf(f32);
    const x = [lanes]f32{ 0, -0.0, below, -1000, -inf, 20, 1000, inf };
    const y: [lanes]f32 = sigmoid(x);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 0.5))), @as(u32, @bitCast(y[0])));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 0.5))), @as(u32, @bitCast(y[1])));
    for (y[2..5]) |yj| {
        try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(yj)));
    }

    for (y[5..]) |yj| {
        try std.testing.expectEqual(@as(f32, 1), yj);
    }

    // The cut itself still gives a normal result inside the bound.
    const at_cut: f32 = sigmoid(@splat(exp_cut))[0];
    try std.testing.expect(at_cut >= std.math.floatMin(f32));
    try std.testing.expect(sigmoidUlpErr(exp_cut, at_cut) <= sigmoid_ulp_tol);
}

test "vector sigmoid: sigmoid(-x) and 1 - sigmoid(x) differ by one rounding of 1 - f at most" {
    var prng = std.Random.DefaultPrng.init(0x5160_1D00);
    const r = prng.random();
    // Both signs share one `f`, so only the rounding of `1 - f` separates them. That rounding
    // errs by half an ulp of a result in `[1/2, 1]`.
    const half_ulp = std.math.ldexp(@as(f64, 1), -25);
    for (0..1 << 12) |_| {
        var x: [lanes]f32 = undefined;
        for (&x) |*xj| {
            xj.* = -exp_cut * r.float(f32);
        }

        const neg_x: [lanes]f32 = -@as(Vec, x);
        const pos: [lanes]f32 = sigmoid(x);
        const neg: [lanes]f32 = sigmoid(neg_x);
        for (pos, neg) |p, n| {
            const gap = @abs((1 - @as(f64, p)) - @as(f64, n));
            try std.testing.expect(gap <= half_ulp);
        }
    }
}

fn checkSigmoidLaneBits(_: void, c: prop.Trial) anyerror!void {
    var prng = std.Random.DefaultPrng.init(c.seed);
    const r = prng.random();
    const edges = [lanes]f32{ 0, -0.0, exp_cut, -1.0e-40, 1.0e-40, -88, 17.5, std.math.inf(f32) };
    var src = edges;
    // Inputs from -100 to 100, so some lanes fall under the cut and some give exactly 1.
    if (c.index > 0) {
        for (&src) |*s| {
            s.* = 200.0 * r.float(f32) - 100.0;
        }
    }

    for (0..lanes) |rot| {
        var x: [lanes]f32 = undefined;
        for (&x, 0..) |*xj, j| {
            xj.* = src[(j + rot) % lanes];
        }

        const y: [lanes]f32 = sigmoid(x);
        for (x, y) |xj, yj| {
            const alone: f32 = sigmoid(@splat(xj))[0];
            try std.testing.expectEqual(@as(u32, @bitCast(alone)), @as(u32, @bitCast(yj)));
        }
    }
}

test "vector sigmoid: a lane gives the bits of a splat of its input at every lane position" {
    try prop.forAll(
        .{ .site = 0x5160_00AB, .cases = 64, .max_size = 63 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkSigmoidLaneBits,
    );
}
