//! Clock arithmetic on `std.Io.Clock.Timestamp` pairs. One definition for
//! the whole tree, so elapsed-time math cannot drift between callers.

const std = @import("std");
const Io = std.Io;

/// Nanoseconds from `t0` to `t1`. `t1` must not precede `t0`.
pub fn elapsedNs(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) u64 {
    return @intCast(t0.durationTo(t1).raw.toNanoseconds());
}

/// Milliseconds from `t0` to `t1` as f64, the reporting unit.
pub fn msBetween(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) f64 {
    return @as(f64, @floatFromInt(elapsedNs(t0, t1))) / 1e6;
}

// -- tests ------------------------------------------------------------------

test "elapsedNs and msBetween on constructed timestamps" {
    const t0: Io.Clock.Timestamp = .{ .raw = .{ .nanoseconds = 1_000_000 }, .clock = .awake };
    const t1: Io.Clock.Timestamp = .{ .raw = .{ .nanoseconds = 3_500_000 }, .clock = .awake };
    try std.testing.expectEqual(@as(u64, 2_500_000), elapsedNs(t0, t1));
    try std.testing.expectEqual(@as(f64, 2.5), msBetween(t0, t1));
    try std.testing.expectEqual(@as(f64, 0.0), msBetween(t0, t0));
}
