//! Shared bench helpers: timing, error measurement, and the colored table reporter. No section
//! logic lives here. It is the common datapath that the per-op files build their rows on.
//!
//! Reporter. The reporter buffers a group of rows, the variants benched for one case, and prints
//! them together. The fastest row tints green and the slowest red. The buffer is display state
//! local to the bench tool: a single-threaded accumulator, reset after every group, never read
//! outside this file. Color stays off unless stderr is a terminal, so piped output stays plain.

const std = @import("std");
const zigferencer = @import("zigferencer");
const op_requires = zigferencer.op_requires;

// -- constants --------------------------------------------------------------

/// 2 FLOPs per multiply-accumulate (one multiply, one add).
pub const f32_flops_per_mac = 2;

const body_fmt = "{s:<34} {s:<10} {d:>10.3} {d:>10.2} {e:>12.2} {x:>18} {s:>8}";
const header_fmt = "{s:<34} {s:<10} {s:>10} {s:>10} {s:>12} {s:>18} {s:>8}\n";
const green = "\x1b[32m";
const red = "\x1b[31m";
const reset = "\x1b[0m";

// -- types ------------------------------------------------------------------

/// A matmul shape: `in` (in_features) -> `out` (out_features).
pub const Case = struct { label: []const u8, in: usize, out: usize };

pub const Timed = struct { ms: f64, iters: usize };

pub const Errors = struct { abs: f32, rel: f32 };

const Row = struct {
    label: []const u8,
    variant: []const u8,
    ms: f64,
    gops: f64,
    rel: f32,
    hash: u64,
    status: []const u8,
    ranked: bool, // false for skip rows (excluded from fastest/slowest)
};

// -- state ------------------------------------------------------------------

var group: [16]Row = undefined;
var group_len: usize = 0;
var color_on: bool = false;

// -- api --------------------------------------------------------------------

/// Warm up once, then time `pickIters` runs of the closure `ctx`. The closure must expose
/// `pub fn run(self: @This()) !void`. A CPU closure that cannot fail returns `!void` anyway.
pub fn timed(io: std.Io, iters_scale: usize, ctx: anytype) !Timed {
    try ctx.run();
    const iters: usize = pickIters(iters_scale);
    const t_start = std.Io.Clock.Timestamp.now(io, .awake);
    for (0..iters) |_| {
        try ctx.run();
    }

    const t_end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_ns: f64 = @floatFromInt(
        @as(i64, @intCast(t_start.durationTo(t_end).raw.toNanoseconds())),
    );
    return .{ .ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1e6, .iters = iters };
}

/// Wall-clock milliseconds of one `ctx.run()`, with no warm-up and no averaging.
/// A section of single blocking submissions reports the spread, which `timed` averages away.
pub fn timedOnce(io: std.Io, ctx: anytype) !f64 {
    const t_start = std.Io.Clock.Timestamp.now(io, .awake);
    try ctx.run();
    const t_end = std.Io.Clock.Timestamp.now(io, .awake);
    const elapsed_ns: f64 = @floatFromInt(
        @as(i64, @intCast(t_start.durationTo(t_end).raw.toNanoseconds())),
    );
    return elapsed_ns / 1e6;
}

/// Convert a `Timed` result and the work per call into G/s. The caller picks the
/// unit: elements for an elementwise op, FLOPs for a compute-heavy op.
pub fn rate(t: Timed, work_per_call: f64) f64 {
    return work_per_call / (t.ms * 1e6);
}

/// Wyhash of the output bytes, the token that a row carries to show bit equality.
pub fn hashOut(out: []const f32) u64 {
    const bytes = std.mem.sliceAsBytes(out);
    return std.hash.Wyhash.hash(0, bytes);
}

/// The largest absolute and the largest relative difference of `got` against `ref`.
pub fn computeErrors(ref: []const f32, got: []const f32) Errors {
    std.debug.assert(ref.len == got.len);

    // Denominator floor: an expected magnitude below it counts as zero for rel_err.
    // Without it, a near-cancellation output inflates rel_err by 4 to 5 orders of
    // magnitude, while the absolute drift stays f32 noise. 1e-5 is far below the
    // typical output of every op.
    const abs_floor: f32 = 1e-5;
    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    for (ref, got) |r, g| {
        const d = @abs(r - g);
        if (d > max_abs) {
            max_abs = d;
        }

        const denom = @abs(r);
        const denom_safe = if (denom < abs_floor) abs_floor else denom;
        const rel = d / denom_safe;
        if (rel > max_rel) {
            max_rel = rel;
        }
    }

    return .{ .abs = max_abs, .rel = max_rel };
}

/// Timed iterations for a case of `scale` work units. Larger cases run fewer.
pub fn pickIters(scale: usize) usize {
    if (scale > 100_000_000) return 4; // llama8b lm_head + ffn class
    if (scale > 10_000_000) return 16; // llama8b attn / llama1b ffn class
    if (scale > 1_000_000) return 128;
    if (scale > 100_000) return 2048;

    return 16384;
}

/// The status cell of a row: `ok` when the tolerance holds, `FAIL` otherwise.
pub fn opStatus(tol: op_requires.Tolerance, rel: f32, hash_match: bool) []const u8 {
    return if (op_requires.toleranceSatisfied(tol, rel, hash_match)) "ok" else "FAIL";
}

/// Turn ANSI color in the table on. Set once at startup from a terminal check.
pub fn setColor(on: bool) void {
    color_on = on;
}

/// Print the section title and the column header.
pub fn printSectionHeader(name: []const u8) void {
    std.debug.print("\nzigferencer bench: {s}\n", .{name});
    std.debug.print(header_fmt, .{ "case", "variant", "ms", "G/s", "rel_err", "hash", "status" });
}

/// Buffer a row that records a declined case, such as an LDS too small for the
/// variant. The ranking over fastest and slowest skips it.
pub fn skipRow(label: []const u8, variant: []const u8, reason: []const u8) void {
    add(.{
        .label = label,
        .variant = variant,
        .ms = 0,
        .gops = 0,
        .rel = 0,
        .hash = 0,
        .status = reason,
        .ranked = false,
    });
}

/// Buffer a pre-timed row, for a section that times by hand.
pub fn rawRow(
    label: []const u8,
    variant: []const u8,
    ms: f64,
    gops: f64,
    rel: f32,
    hash: u64,
    status: []const u8,
) void {
    add(.{
        .label = label,
        .variant = variant,
        .ms = ms,
        .gops = gops,
        .rel = rel,
        .hash = hash,
        .status = status,
        .ranked = true,
    });
}

/// Print the buffered group, then clear it. With more than one ranked row, the
/// fastest tints green and the slowest red.
pub fn flush() void {
    defer group_len = 0;

    var min_i: ?usize = null;
    var max_i: ?usize = null;
    for (group[0..group_len], 0..) |r, i| {
        if (!r.ranked) continue;
        if (min_i == null or r.ms < group[min_i.?].ms) {
            min_i = i;
        }

        if (max_i == null or r.ms > group[max_i.?].ms) {
            max_i = i;
        }
    }

    const rank = color_on and min_i != null and min_i.? != max_i.?;
    for (group[0..group_len], 0..) |r, i| {
        if (!r.ranked) {
            std.debug.print(header_fmt, .{ r.label, r.variant, "-", "-", "-", "-", r.status });
            continue;
        }

        const c: []const u8 = if (!rank)
            ""
        else if (i == min_i.?) green else if (i == max_i.?) red else "";

        const rs: []const u8 = if (c.len > 0) reset else "";
        std.debug.print(
            "{s}" ++ body_fmt ++ "{s}\n",
            .{ c, r.label, r.variant, r.ms, r.gops, r.rel, r.hash, r.status, rs },
        );
    }
}

/// One CPU reference row, at rel_err 0 and the reference hash. The caller
/// computes `ref_hash` from the one-shot reference output.
pub fn cpuRow(
    io: std.Io,
    scale: usize,
    label: []const u8,
    name: []const u8,
    work: f64,
    ref_hash: u64,
    ctx: anytype,
) !void {
    const t = try timed(io, scale, ctx);
    rawRow(label, name, t.ms, rate(t, work), 0.0, ref_hash, "ok");
}

/// One GPU-variant row: run once to fill `out_buf`, hash it, diff it against the
/// reference, time the closure, then buffer the row with the tolerance verdict.
pub fn gpuRow(
    io: std.Io,
    scale: usize,
    label: []const u8,
    name: []const u8,
    work: f64,
    tol: op_requires.Tolerance,
    ref_out: []const f32,
    ref_hash: u64,
    out_buf: []const f32,
    ctx: anytype,
) !void {
    try ctx.run();
    const hsh = hashOut(out_buf);
    const errs = computeErrors(ref_out, out_buf);
    const t = try timed(io, scale, ctx);
    rawRow(
        label,
        name,
        t.ms,
        rate(t, work),
        errs.rel,
        hsh,
        opStatus(tol, errs.rel, hsh == ref_hash),
    );
}

// -- helpers ----------------------------------------------------------------

fn add(row: Row) void {
    std.debug.assert(group_len < group.len);

    group[group_len] = row;
    group_len += 1;
}
