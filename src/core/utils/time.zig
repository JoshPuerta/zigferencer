//! Clock arithmetic on timestamp pairs. One definition serves the whole tree,
//! so elapsed-time math cannot drift between callers.
//!
//! Two types. An `Io.Clock.Timestamp` comes from the application `Io`, and the bare `Io.Timestamp`
//! from the diagnostics clock of `log.now`. The `raw` helpers take the bare type. Both clocks read
//! the same monotonic OS clock, so the second timestamp of a pair never precedes the first.

const std = @import("std");
const Io = std.Io;

/// Nanoseconds from `t0` to `t1`. `t1` must not precede `t0`.
pub fn elapsedNs(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) u64 {
    return @intCast(t0.durationTo(t1).raw.toNanoseconds());
}

/// Nanoseconds from `t0` to `t1` as u64, or zero when `t1` precedes `t0`.
/// The prefix-cache admission gate uses it, so a clock step can never wrap the unsigned cost.
pub fn elapsedNsClamped(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) u64 {
    const ns: i64 = @intCast(t0.durationTo(t1).raw.toNanoseconds());
    return if (ns > 0) @intCast(ns) else 0;
}

/// Milliseconds from `t0` to `t1` as f64, the reporting unit.
pub fn msBetween(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) f64 {
    return nsToMs(elapsedNs(t0, t1));
}

/// Milliseconds from `t0` to `t1` as f32, for the f32 fields of the serve wire.
/// The signed nanoseconds convert straight to f32, because a path through f64 can round apart.
pub fn msBetweenF32(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) f32 {
    const ns: i64 = @intCast(t0.durationTo(t1).raw.toNanoseconds());
    return @as(f32, @floatFromInt(ns)) / 1e6;
}

/// Milliseconds as f64 from `t0` to the time `io` reads now on the awake clock.
pub fn msSince(io: Io, t0: Io.Clock.Timestamp) f64 {
    return msBetween(t0, Io.Clock.Timestamp.now(io, .awake));
}

/// Nanoseconds from `t0` to `t1` on bare timestamps. `t1` must not precede `t0`.
pub fn rawElapsedNs(t0: Io.Timestamp, t1: Io.Timestamp) u64 {
    std.debug.assert(t1.nanoseconds >= t0.nanoseconds);

    return @intCast(t1.nanoseconds - t0.nanoseconds);
}

/// Milliseconds as f64 from `t0` to `t1` on bare timestamps.
pub fn rawMsBetween(t0: Io.Timestamp, t1: Io.Timestamp) f64 {
    return nsToMs(rawElapsedNs(t0, t1));
}

/// Nanoseconds as milliseconds, u64 in and f64 out.
pub fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

// -- tests ------------------------------------------------------------------

fn awakeAt(ns: i96) Io.Clock.Timestamp {
    return .{ .raw = .{ .nanoseconds = ns }, .clock = .awake };
}

test "elapsedNs and msBetween on constructed timestamps" {
    const t0 = awakeAt(1_000_000);
    const t1 = awakeAt(3_500_000);
    try std.testing.expectEqual(@as(u64, 2_500_000), elapsedNs(t0, t1));
    try std.testing.expectEqual(@as(f64, 2.5), msBetween(t0, t1));
    try std.testing.expectEqual(@as(f64, 0.0), msBetween(t0, t0));
}

test "elapsedNsClamped: a forward pair is exact, a reversed pair is zero" {
    try std.testing.expectEqual(@as(u64, 25), elapsedNsClamped(awakeAt(100), awakeAt(125)));
    try std.testing.expectEqual(@as(u64, 0), elapsedNsClamped(awakeAt(125), awakeAt(100)));
    try std.testing.expectEqual(@as(u64, 0), elapsedNsClamped(awakeAt(7), awakeAt(7)));
}

test "msBetweenF32 rounds once to f32, and a reversed pair stays negative" {
    // 16_777_217 is 2^24 + 1, the first integer f32 cannot hold, so one rounding is visible.
    const t0 = awakeAt(0);
    const t1 = awakeAt(16_777_217);
    try std.testing.expectEqual(@as(f32, 16_777_216) / 1e6, msBetweenF32(t0, t1));
    try std.testing.expectEqual(@as(f32, -2.5), msBetweenF32(awakeAt(2_500_000), t0));
}

test "msSince reads a time at or after its start" {
    const io = std.testing.io;
    const t0 = Io.Clock.Timestamp.now(io, .awake);
    try std.testing.expect(msSince(io, t0) >= 0);
}

test "rawElapsedNs, rawMsBetween, and nsToMs agree with the clock helpers" {
    const t0: Io.Timestamp = .{ .nanoseconds = 100 };
    const t1: Io.Timestamp = .{ .nanoseconds = 125 };
    try std.testing.expectEqual(@as(u64, 25), rawElapsedNs(t0, t1));
    try std.testing.expectEqual(@as(u64, 0), rawElapsedNs(t1, t1));
    try std.testing.expectEqual(elapsedNs(awakeAt(100), awakeAt(125)), rawElapsedNs(t0, t1));
    const t2: Io.Timestamp = .{ .nanoseconds = 2_500_000 };
    try std.testing.expectEqual(@as(f64, 2.5), rawMsBetween(.{ .nanoseconds = 0 }, t2));
    try std.testing.expectEqual(@as(f64, 1.5), nsToMs(1_500_000));
}
