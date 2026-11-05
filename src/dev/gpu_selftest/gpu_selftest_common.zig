//! The fill and compare helpers that every check of `gpu-selftest` shares.

const std = @import("std");

// -- api --------------------------------------------------------------------

/// Count elements violating numpy-style allclose: |a-b| > atol + rtol*|a|.
pub fn countFail(cpu: []const f32, gpu: []const f32, atol: f32, rtol: f32) usize {
    var n: usize = 0;
    for (cpu, gpu) |c, g| {
        if (@abs(c - g) > atol + rtol * @abs(c)) {
            n += 1;
        }
    }

    return n;
}

pub fn fillN(r: std.Random, slice: []f32, scale: f32) void {
    for (slice) |*v| {
        v.* = (r.float(f32) - 0.5) * 2.0 * scale;
    }
}

/// `fillN` at the KV element type: each value rounds once to `T`.
pub fn fillTyped(comptime T: type, r: std.Random, slice: []T, scale: f32) void {
    for (slice) |*v| {
        v.* = @floatCast((r.float(f32) - 0.5) * 2.0 * scale);
    }
}

pub fn printDiff(w: *std.Io.Writer, name: []const u8, cpu: []const f32, gpu: []const f32) !void {
    var maxabs: f32 = 0;
    for (cpu, gpu) |c, g| {
        const d = @abs(c - g);
        if (d > maxabs) {
            maxabs = d;
        }
    }

    try w.print(
        "  {s}: max_abs={e:.3}  max_rel={e:.3}  (n={d})\n",
        .{ name, maxabs, maxRel(cpu, gpu), cpu.len },
    );
}

// -- helpers ----------------------------------------------------------------

/// Max relative diff (rel = |a-b| / (|a| + tiny)).
fn maxRel(cpu: []const f32, gpu: []const f32) f32 {
    var mr: f32 = 0;
    for (cpu, gpu) |c, g| {
        const d = @abs(c - g);
        const rel = d / (@abs(c) + 1e-6);
        if (rel > mr) {
            mr = rel;
        }
    }

    return mr;
}
