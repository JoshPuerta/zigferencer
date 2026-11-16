//! The vector elementwise ops of both SIMD backends. The file holds `residualAdd`.
//!
//! Bit rule. Each lane runs the one add of the scalar loop, so the output is bit-identical to
//! `naive.residualAdd` at every length.
//!
//! Aliasing. The input and the output are the same memory or do not overlap. With a partial
//! overlap, a lane can read an element that an earlier store changed.

const std = @import("std");
const naive = @import("naive.zig");

// -- constants --------------------------------------------------------------

const lanes = 8;

// -- types ------------------------------------------------------------------

const Vec = @Vector(lanes, f32);

// -- api --------------------------------------------------------------------

/// dst[i] += x[i]
pub fn residualAdd(x: []const f32, dst: []f32) void {
    std.debug.assert(x.len <= dst.len);
    std.debug.assert(sameOrDisjoint(x, dst[0..x.len]));

    var i: usize = 0;
    while (i + lanes <= x.len) : (i += lanes) {
        const xv: Vec = x[i..][0..lanes].*;
        const dv: Vec = dst[i..][0..lanes].*;
        dst[i..][0..lanes].* = dv + xv;
    }

    while (i < x.len) : (i += 1) {
        dst[i] += x[i];
    }

    if (std.debug.runtime_safety) {
        for (dst[0..x.len]) |v| {
            std.debug.assert(std.math.isFinite(v));
        }
    }
}

// -- helpers ----------------------------------------------------------------

fn sameOrDisjoint(x: []const f32, y: []const f32) bool {
    const x_lo = @intFromPtr(x.ptr);
    const y_lo = @intFromPtr(y.ptr);
    if (x_lo == y_lo) return true;

    return x_lo + x.len * @sizeOf(f32) <= y_lo or y_lo + y.len * @sizeOf(f32) <= x_lo;
}

// -- tests ------------------------------------------------------------------

test "residualAdd is bit-identical to naive.residualAdd" {
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
            try std.testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
        }
    }
}

test "residualAdd doubles a vector that is its own input" {
    var got: [13]f32 = undefined;
    var want: [13]f32 = undefined;
    for (&got, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) * 0.37 - 2.0;
    }

    @memcpy(&want, &got);
    naive.residualAdd(&want, &want);
    residualAdd(&got, &got);
    for (got, want) |a, b| {
        try std.testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
    }
}
