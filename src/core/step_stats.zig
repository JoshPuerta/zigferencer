//! Lean per-run timers for the decode step. Values only.
//!
//! Comptime. `merge` walks the fields, so a new counter joins the sum with no edit.
//!
//! Stat. It folds one timed host step into a sum, a max, a min, and a count, so a run keeps its
//! summary in fixed space and the hot path allocates nothing.
//!
//! DecodeHost. It names the six host steps of one decode step: record, submit, speculate, wait,
//! the logits copy, and the sampler argmax. The fused runners fill the steps up to the copy and
//! the ring counts. The decode loop fills the argmax. The bench report prints the result.
//!
//! Ring counts. A hit submits a buffer that the previous step recorded. A miss records on the
//! critical path. A wasted record is a speculative record that no step submitted. The ring counts
//! it when it discards the record, so the last speculation of a run counts in the window that
//! discards it.

const std = @import("std");

// -- types ------------------------------------------------------------------

pub const Stat = struct {
    sum_ns: u64 = 0,
    max_ns: u64 = 0,
    /// Smallest sample. It reads as maxInt while the summary is empty, so use `minMs`.
    min_ns: u64 = std.math.maxInt(u64),
    count: u64 = 0,

    /// Fold one sample of `ns` nanoseconds.
    pub fn add(self: *Stat, ns: u64) void {
        self.sum_ns += ns;
        self.max_ns = @max(self.max_ns, ns);
        self.min_ns = @min(self.min_ns, ns);
        self.count += 1;
        std.debug.assert(self.max_ns <= self.sum_ns);
    }

    /// Fold another summary into this one.
    pub fn merge(self: *Stat, other: Stat) void {
        self.sum_ns += other.sum_ns;
        self.max_ns = @max(self.max_ns, other.max_ns);
        self.min_ns = @min(self.min_ns, other.min_ns);
        self.count += other.count;
    }

    /// Mean in milliseconds. An empty summary has mean zero.
    pub fn avgMs(self: Stat) f64 {
        if (self.count == 0) return 0;
        return nsToMs(self.sum_ns) / @as(f64, @floatFromInt(self.count));
    }

    pub fn maxMs(self: Stat) f64 {
        return nsToMs(self.max_ns);
    }

    /// Smallest sample in milliseconds. An empty summary has minimum zero.
    pub fn minMs(self: Stat) f64 {
        if (self.count == 0) return 0;
        return nsToMs(self.min_ns);
    }
};

/// The outcomes of the decode record ring.
pub const RingCounts = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    speculated: u64 = 0,
    wasted: u64 = 0,

    pub fn merge(self: *RingCounts, other: RingCounts) void {
        inline for (std.meta.fields(RingCounts)) |f| {
            @field(self, f.name) += @field(other, f.name);
        }
    }
};

pub const DecodeHost = struct {
    /// The record on the critical path. A ring hit adds a sample near zero.
    record: Stat = .{},
    submit: Stat = .{},
    /// The record of the predicted next step while the GPU runs this one.
    speculate: Stat = .{},
    wait: Stat = .{},
    copy: Stat = .{},
    argmax: Stat = .{},
    ring: RingCounts = .{},

    pub fn merge(self: *DecodeHost, other: DecodeHost) void {
        inline for (std.meta.fields(DecodeHost)) |f| {
            @field(self, f.name).merge(@field(other, f.name));
        }
    }

    /// True when a fused runner timed at least one step.
    pub fn hasDevice(self: DecodeHost) bool {
        return self.record.count > 0;
    }
};

// -- api --------------------------------------------------------------------

/// Nanoseconds as milliseconds.
pub fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// Nanoseconds from `t0` to `t1` on the debug-io clock the runners read.
/// The clock is monotonic, so the difference is never negative.
pub fn nsBetween(t0: std.Io.Timestamp, t1: std.Io.Timestamp) u64 {
    std.debug.assert(t1.nanoseconds >= t0.nanoseconds);

    return @intCast(t1.nanoseconds - t0.nanoseconds);
}

// -- tests ------------------------------------------------------------------

test "Stat: sum, max, count, and mean follow the samples" {
    var s: Stat = .{};
    try std.testing.expectEqual(@as(f64, 0), s.avgMs());
    s.add(1_000_000);
    s.add(3_000_000);
    s.add(2_000_000);
    try std.testing.expectEqual(@as(u64, 6_000_000), s.sum_ns);
    try std.testing.expectEqual(@as(u64, 3_000_000), s.max_ns);
    try std.testing.expectEqual(@as(u64, 3), s.count);
    try std.testing.expectEqual(@as(f64, 2.0), s.avgMs());
    try std.testing.expectEqual(@as(f64, 3.0), s.maxMs());
    try std.testing.expectEqual(@as(f64, 1.0), s.minMs());
    try std.testing.expectEqual(@as(f64, 0), (Stat{}).minMs());
}

test "DecodeHost: merge adds counts and keeps the larger max per host step" {
    var a: DecodeHost = .{};
    var b: DecodeHost = .{};
    a.record.add(4_000_000);
    b.record.add(8_000_000);
    b.argmax.add(500_000);
    try std.testing.expect(!(DecodeHost{}).hasDevice());
    a.merge(b);
    try std.testing.expect(a.hasDevice());
    try std.testing.expectEqual(@as(u64, 2), a.record.count);
    try std.testing.expectEqual(@as(u64, 8_000_000), a.record.max_ns);
    try std.testing.expectEqual(@as(f64, 6.0), a.record.avgMs());
    try std.testing.expectEqual(@as(f64, 4.0), a.record.minMs());
    try std.testing.expectEqual(@as(u64, 1), a.argmax.count);
    try std.testing.expectEqual(@as(u64, 0), a.wait.count);
    a.ring.hits = 3;
    b.ring = .{ .hits = 4, .misses = 1, .speculated = 5, .wasted = 1 };
    a.merge(b);
    try std.testing.expectEqual(
        RingCounts{ .hits = 7, .misses = 1, .speculated = 5, .wasted = 1 },
        a.ring,
    );
}

test "nsBetween: difference of two timestamps" {
    const t0: std.Io.Timestamp = .{ .nanoseconds = 10 };
    const t1: std.Io.Timestamp = .{ .nanoseconds = 35 };
    try std.testing.expectEqual(@as(u64, 25), nsBetween(t0, t1));
    try std.testing.expectEqual(@as(u64, 0), nsBetween(t1, t1));
}
