//! The property driver. `forAll` generates `opts.cases` cases and checks each one.
//!
//! Comptime. `forAll` takes the case type, the generator, and the check at comptime, so each
//! property gets typed code.
//!
//! Cases. Case i has size min(i, max_size), so the first failing case is a small
//! one. Each case draws from its own generator, seeded from the call site, the
//! test seed, and the case index. One case therefore replays alone.
//!
//! Seed. It is `std.testing.random_seed`, which the build runner passes to the
//! test binary. A plain `zig build test` takes a new seed on each run and
//! explores new cases. A gate pins `--seed`, so its cases never change.

const std = @import("std");
const assert = std.debug.assert;

// -- types ------------------------------------------------------------------

pub const Options = struct {
    /// A constant per call site, so two properties never share a case stream.
    site: u64,
    cases: u32 = 256,
    /// Case i gets size min(i, max_size), so the first failing case is a small one.
    max_size: u32 = 64,
};

/// A case that knows its number, for a property that alternates its shape by parity.
/// The number is the size, so the options must set `max_size` to `cases - 1`.
pub const Trial = struct { index: u32, seed: u64 };

const Failure = struct { index: u32, size: u32, err: anyerror };

// -- api --------------------------------------------------------------------

/// Check `check(context, gen(context, random, size))` for every case. The first
/// failing case stops the run: its index, size, seed, and site go to stderr
/// with the command that replays it, and its error returns.
pub fn forAll(
    opts: Options,
    context: anytype,
    comptime Case: type,
    comptime gen: fn (@TypeOf(context), std.Random, size: u32) Case,
    comptime check: fn (@TypeOf(context), Case) anyerror!void,
) anyerror!void {
    const failure = firstFailure(opts, context, Case, gen, check) orelse return;
    const seed = std.testing.random_seed;
    std.debug.print(
        "prop: case {d} size {d} returned {s} (seed 0x{x}, site 0x{x}). Rerun: " ++
            "zig build test --seed 0x{x}\n",
        .{ failure.index, failure.size, @errorName(failure.err), seed, opts.site, seed },
    );
    return failure.err;
}

/// A generator whose case is one seed. A check that draws a long stream of
/// values seeds its own generator from the case, so one case still replays alone.
pub fn genSeed(comptime Context: type) fn (Context, std.Random, u32) u64 {
    return struct {
        fn gen(_: Context, random: std.Random, size: u32) u64 {
            _ = size;
            return random.int(u64);
        }
    }.gen;
}

pub fn genTrial(comptime Context: type) fn (Context, std.Random, u32) Trial {
    return struct {
        fn gen(_: Context, random: std.Random, size: u32) Trial {
            return .{ .index = size, .seed = random.int(u64) };
        }
    }.gen;
}

// -- helpers ----------------------------------------------------------------

// The self-tests call this directly, because a test binary that writes to
// stderr makes the build runner print the command as failed.
fn firstFailure(
    opts: Options,
    context: anytype,
    comptime Case: type,
    comptime gen: fn (@TypeOf(context), std.Random, size: u32) Case,
    comptime check: fn (@TypeOf(context), Case) anyerror!void,
) ?Failure {
    assert(opts.cases > 0);

    const seed = std.testing.random_seed;
    var i: u32 = 0;
    while (i < opts.cases) : (i += 1) {
        const size = @min(i, opts.max_size);
        assert(size <= opts.max_size);
        var prng = std.Random.DefaultPrng.init(caseSeed(opts.site, seed, i));
        const case = gen(context, prng.random(), size);
        check(context, case) catch |err| return .{ .index = i, .size = size, .err = err };
    }

    return null;
}

fn caseSeed(site: u64, seed: u32, index: u32) u64 {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u32, bytes[0..4], seed, .little);
    std.mem.writeInt(u32, bytes[4..8], index, .little);
    return std.hash.Wyhash.hash(site, &bytes);
}

// -- tests ------------------------------------------------------------------

/// Records the size and the first draw of every generated case, and fails the
/// check from size `fail_from` up.
const Recorder = struct {
    const capacity = 32;
    sizes: [capacity]u32 = undefined,
    draws: [capacity]u64 = undefined,
    generated: u32 = 0,
    checked: u32 = 0,
    failed_at: ?u32 = null,
    fail_from: u32 = std.math.maxInt(u32),

    fn gen(r: *Recorder, random: std.Random, size: u32) u32 {
        assert(r.generated < capacity);

        r.sizes[r.generated] = size;
        r.draws[r.generated] = random.int(u64);
        r.generated += 1;
        return size;
    }

    fn check(r: *Recorder, size: u32) anyerror!void {
        r.checked += 1;
        if (size >= r.fail_from) {
            assert(r.failed_at == null);

            r.failed_at = r.checked - 1;
            return error.PropertyFalse;
        }
    }
};

test "forAll is deterministic for one site" {
    const opts: Options = .{ .site = 0x5175_0001, .cases = 24, .max_size = 9 };
    var a: Recorder = .{};
    var b: Recorder = .{};
    try forAll(opts, &a, u32, Recorder.gen, Recorder.check);
    try forAll(opts, &b, u32, Recorder.gen, Recorder.check);
    try std.testing.expectEqual(opts.cases, a.generated);
    try std.testing.expectEqualSlices(u32, a.sizes[0..a.generated], b.sizes[0..b.generated]);
    try std.testing.expectEqualSlices(u64, a.draws[0..a.generated], b.draws[0..b.generated]);
}

test "forAll sizes count up from zero and hold at max_size" {
    const opts: Options = .{ .site = 0x5175_0002, .cases = 20, .max_size = 7 };
    var r: Recorder = .{};
    try forAll(opts, &r, u32, Recorder.gen, Recorder.check);
    try std.testing.expectEqual(opts.cases, r.generated);
    for (r.sizes[0..r.generated], 0..) |size, i| {
        try std.testing.expectEqual(@min(@as(u32, @intCast(i)), opts.max_size), size);
    }
}

test "the driver stops at the first failing case and returns its error" {
    const opts: Options = .{ .site = 0x5175_0003, .cases = 20, .max_size = 9 };
    var r: Recorder = .{ .fail_from = 5 };
    const failure = firstFailure(opts, &r, u32, Recorder.gen, Recorder.check).?;
    try std.testing.expectEqual(@as(u32, 5), failure.index);
    try std.testing.expectEqual(@as(u32, 5), failure.size);
    try std.testing.expectEqual(error.PropertyFalse, failure.err);
    try std.testing.expectEqual(@as(?u32, 5), r.failed_at);
    try std.testing.expectEqual(@as(u32, 6), r.checked);
    try std.testing.expectEqual(@as(u32, 6), r.generated);
}

test "forAll gives distinct sites distinct case streams" {
    var a: Recorder = .{};
    var b: Recorder = .{};
    try forAll(.{ .site = 0x5175_0004, .cases = 16 }, &a, u32, Recorder.gen, Recorder.check);
    try forAll(.{ .site = 0x5175_0005, .cases = 16 }, &b, u32, Recorder.gen, Recorder.check);
    for (a.draws[0..a.generated], b.draws[0..b.generated]) |x, y| {
        try std.testing.expect(x != y);
    }
}
