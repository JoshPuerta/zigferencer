//! The record ring of the decode step: two command buffers, each keyed by the step it holds.
//!
//! Comptime. `Slot` and `RecordRing` are generic over the key type.
//!
//! Role. While the GPU runs the submitted buffer, the runner records the predicted next step into
//! the other buffer. The next step then submits that buffer with no record on its critical path.
//! This file holds the control state alone: no Vulkan, no allocation, no I/O. The fused runners
//! own the command buffers and call these transitions around each Vulkan action.
//!
//! States. A buffer is `free`, `recorded k`, or `pending k`. `recorded k` holds the
//! commands of key `k` that the GPU has not seen. `pending k` is submitted and
//! not yet complete. The invariants, which every transition asserts:
//!
//!   1. No record writes a pending buffer.
//!   2. At most one buffer is pending.
//!   3. A step starts only when no buffer is pending.
//!   4. A hit submits a buffer whose key equals the wanted key.
//!
//! Records. A record is two transitions, `beginRecord` and `endRecord`, so a record that fails
//! half way leaves the buffer `free` and never a stale `recorded`. A key holds everything the
//! record reads that can change between steps, so equal keys give equal command streams.
//! The token is data in the staging buffer, and the descriptor sets are fixed at init. The level
//! is part of the key, so a recording made at one level never replays at another.

const std = @import("std");
const prop = @import("prop.zig");
const span = @import("span.zig");

// -- types ------------------------------------------------------------------

/// The two command buffers.
pub const Buf = enum(u1) {
    a,
    b,

    pub fn other(self: Buf) Buf {
        return switch (self) {
            .a => .b,
            .b => .a,
        };
    }
};

/// How the decode record pass frames its command buffer. The default is one-time submit at the
/// process level. Only the `bench-decode-host` hooks change `one_time`.
pub const RecordOpts = struct {
    one_time: bool = true,
    level: span.GpuLevel = .off,
};

/// The key of the decode record ring: everything a decode record reads that can change between
/// two steps.
pub const DecodeKey = struct {
    pos: u32,
    opts: RecordOpts = .{},

    /// The key of the step after this one, or null when no such step fits
    /// the window.
    pub fn next(self: DecodeKey, max_seq: u32) ?DecodeKey {
        std.debug.assert(self.pos < max_seq);
        std.debug.assert(self.opts.one_time);

        if (self.pos + 1 >= max_seq) return null;

        return .{ .pos = self.pos + 1, .opts = self.opts };
    }
};

pub fn Slot(comptime Key: type) type {
    return union(enum) {
        free,
        recorded: Key,
        pending: Key,
    };
}

/// The decision of `prepare`: the buffer to submit, and whether it already
/// holds the wanted key.
pub const Prepared = struct { buf: Buf, hit: bool };

pub fn RecordRing(comptime Key: type) type {
    return struct {
        const Self = @This();
        pub const SlotT = Slot(Key);

        slots: [2]SlotT = .{ .free, .free },

        // -- queries --------------------------------------------------------

        pub fn get(self: *const Self, c: Buf) SlotT {
            return self.slots[@intFromEnum(c)];
        }

        pub fn isPending(self: *const Self, c: Buf) bool {
            return self.get(c) == .pending;
        }

        pub fn holds(self: *const Self, c: Buf, w: Key) bool {
            return switch (self.get(c)) {
                .recorded => |k| std.meta.eql(k, w),
                else => false,
            };
        }

        pub fn pendingCount(self: *const Self) u2 {
            return @as(u2, @intFromBool(self.isPending(.a))) + @intFromBool(self.isPending(.b));
        }

        // -- transitions ----------------------------------------------------

        /// Pick the buffer for wanted key `w`. A buffer that holds `recorded w`
        /// is a hit. A miss names buffer `a`, which the caller then records.
        pub fn prepare(self: *const Self, w: Key) Prepared {
            std.debug.assert(self.pendingCount() == 0);

            for ([_]Buf{ .a, .b }) |c| {
                switch (self.get(c)) {
                    .recorded => |k| if (std.meta.eql(k, w)) return .{ .buf = c, .hit = true },
                    else => {},
                }
            }

            return .{ .buf = .a, .hit = false };
        }

        /// Free buffer `c` before the caller records into it. Returns true when
        /// this discards a recording that no step submitted.
        pub fn beginRecord(self: *Self, c: Buf) bool {
            std.debug.assert(!self.isPending(c));

            const wasted = self.get(c) == .recorded;
            self.slots[@intFromEnum(c)] = .free;
            return wasted;
        }

        /// Buffer `c` now holds the commands of key `k`.
        pub fn endRecord(self: *Self, c: Buf, k: Key) void {
            std.debug.assert(self.get(c) == .free);

            self.slots[@intFromEnum(c)] = .{ .recorded = k };
        }

        /// Submit buffer `c`, which must hold `recorded w`.
        pub fn submit(self: *Self, c: Buf, w: Key) void {
            std.debug.assert(self.pendingCount() == 0);

            switch (self.get(c)) {
                .recorded => |k| std.debug.assert(std.meta.eql(k, w)),
                else => unreachable,
            }

            self.slots[@intFromEnum(c)] = .{ .pending = w };
            std.debug.assert(self.pendingCount() == 1);
        }

        /// The buffer that can take a speculative record after a submit of `c`.
        /// The other buffer is never pending, because only `c` is.
        pub fn speculateTarget(self: *const Self, c: Buf) Buf {
            std.debug.assert(self.isPending(c));
            std.debug.assert(!self.isPending(c.other()));

            return c.other();
        }

        /// The GPU finished buffer `c`. A one-time-submit buffer is spent, so
        /// it becomes free.
        pub fn complete(self: *Self, c: Buf) void {
            std.debug.assert(self.isPending(c));

            self.slots[@intFromEnum(c)] = .free;
            std.debug.assert(self.pendingCount() == 0);
        }

        /// Free both buffers before any other submit on the runner. Returns
        /// the count of recordings that no step submitted.
        pub fn invalidate(self: *Self) u2 {
            std.debug.assert(self.pendingCount() == 0);

            var wasted: u2 = 0;
            for (&self.slots) |*s| {
                if (s.* == .recorded) {
                    wasted += 1;
                }

                s.* = .free;
            }

            return wasted;
        }

        /// Forget both buffers after a failed submit or wait. The runner reports
        /// the error, and a lost device makes every buffer unusable anyway.
        pub fn drop(self: *Self) void {
            self.slots = .{ .free, .free };
        }
    };
}

// -- api --------------------------------------------------------------------

/// The decode key of `pos` at `level`.
pub fn decodeKey(pos: u32, level: span.GpuLevel) DecodeKey {
    return .{ .pos = pos, .opts = .{ .level = level } };
}

// -- tests ------------------------------------------------------------------

const Ring = RecordRing(u32);

/// One decode step the way the runners drive the ring, without the GPU. The
/// caller completes the step before the next one, as `waitForFences` does.
fn stepOnce(r: *Ring, w: u32, next: ?u32, wasted: *u32) Prepared {
    const p = r.prepare(w);
    if (!p.hit) {
        if (r.beginRecord(p.buf)) {
            wasted.* += 1;
        }

        r.endRecord(p.buf, w);
    }

    r.submit(p.buf, w);
    if (next) |k| {
        const o = r.speculateTarget(p.buf);
        if (r.beginRecord(o)) {
            wasted.* += 1;
        }

        r.endRecord(o, k);
    }

    std.debug.assert(r.pendingCount() == 1);
    r.complete(p.buf);
    return p;
}

test "RecordRing: each transition moves one slot the way the model states" {
    var r: Ring = .{};
    try std.testing.expectEqual(@as(u2, 0), r.pendingCount());
    try std.testing.expect(!r.beginRecord(.a));
    r.endRecord(.a, 7);
    try std.testing.expectEqual(Ring.SlotT{ .recorded = 7 }, r.get(.a));
    try std.testing.expectEqual(Prepared{ .buf = .a, .hit = true }, r.prepare(7));
    try std.testing.expectEqual(Prepared{ .buf = .a, .hit = false }, r.prepare(8));
    r.submit(.a, 7);
    try std.testing.expectEqual(Ring.SlotT{ .pending = 7 }, r.get(.a));
    try std.testing.expectEqual(Buf.b, r.speculateTarget(.a));
    r.endRecord(.b, 8);
    try std.testing.expect(r.holds(.b, 8));
    try std.testing.expect(!r.holds(.b, 9));
    try std.testing.expect(!r.holds(.a, 7));
    r.complete(.a);
    try std.testing.expectEqual(Ring.SlotT.free, r.get(.a));
    try std.testing.expectEqual(Prepared{ .buf = .b, .hit = true }, r.prepare(8));
    try std.testing.expect(r.beginRecord(.b));
    try std.testing.expectEqual(Ring.SlotT.free, r.get(.b));
    r.endRecord(.b, 9);
    try std.testing.expectEqual(@as(u2, 1), r.invalidate());
    try std.testing.expectEqual(Ring{}, r);
    r.endRecord(.a, 1);
    r.submit(.a, 1);
    r.drop();
    try std.testing.expectEqual(Ring{}, r);
}

test "RecordRing: a decode run of n steps hits n - 1 times after the first miss" {
    var r: Ring = .{};
    var wasted: u32 = 0;
    const n: u32 = 64;
    var hits: u32 = 0;
    var pos: u32 = 100;
    while (pos < 100 + n) : (pos += 1) {
        const p = stepOnce(&r, pos, pos + 1, &wasted);
        if (p.hit) {
            hits += 1;
        }

        if (pos == 100) {
            try std.testing.expect(!p.hit);
        }
    }

    try std.testing.expectEqual(n - 1, hits);
    try std.testing.expectEqual(@as(u32, 0), wasted);
    // The last speculation stays recorded until something discards it.
    try std.testing.expect(r.prepare(100 + n).hit);
}

test "RecordRing: an invalidation between steps forces a miss and counts the waste" {
    var r: Ring = .{};
    var wasted: u32 = 0;
    _ = stepOnce(&r, 5, 6, &wasted);
    try std.testing.expectEqual(@as(u2, 1), r.invalidate());
    try std.testing.expect(!stepOnce(&r, 6, 7, &wasted).hit);
    try std.testing.expect(stepOnce(&r, 7, null, &wasted).hit);
    // A wanted key that the speculation did not predict also misses.
    _ = stepOnce(&r, 20, 21, &wasted);
    try std.testing.expect(!stepOnce(&r, 30, 31, &wasted).hit);
    try std.testing.expectEqual(@as(u32, 1), wasted);
}

fn checkRingModel(_: void, seed: u64) anyerror!void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const rnd = prng.random();
    var r: Ring = .{};
    var wasted: u32 = 0;
    var pos: u32 = rnd.uintLessThan(u32, 8);
    for (0..64) |_| {
        switch (rnd.uintLessThan(u8, 4)) {
            // A decode step at the next position, the common case.
            0, 1 => pos += 1,
            // A jump: a new turn or a rewind.
            2 => pos = rnd.uintLessThan(u32, 16),
            // Another submit on the runner.
            else => _ = r.invalidate(),
        }

        const before = r;
        const next: ?u32 = if (rnd.boolean()) pos + 1 else null;
        const p = stepOnce(&r, pos, next, &wasted);
        // A hit needs `recorded pos` in the chosen buffer before the step.
        if (p.hit) {
            try std.testing.expectEqual(Ring.SlotT{ .recorded = pos }, before.get(p.buf));
        } else {
            try std.testing.expect(before.get(.a) != .recorded or before.get(.a).recorded != pos);
            try std.testing.expect(before.get(.b) != .recorded or before.get(.b).recorded != pos);
        }

        // After the step the submitted buffer is free and nothing is pending.
        try std.testing.expectEqual(@as(u2, 0), r.pendingCount());
        try std.testing.expectEqual(Ring.SlotT.free, r.get(p.buf));
        if (next) |k| {
            try std.testing.expectEqual(Ring.SlotT{ .recorded = k }, r.get(p.buf.other()));
        }
    }
}

test "RecordRing: the model invariants hold over seeded random op sequences" {
    try prop.forAll(
        .{ .site = 0x5eed_0036, .cases = 200 },
        {},
        u64,
        prop.genSeed(void),
        checkRingModel,
    );
}

test "DecodeKey: a key at one level never equals the key at another" {
    const levels = std.enums.values(span.GpuLevel);
    for (levels) |a| {
        for (levels) |b| {
            const ka = decodeKey(7, a);
            const kb = decodeKey(7, b);
            try std.testing.expectEqual(a == b, std.meta.eql(ka, kb));
            // The next key keeps the level, so a speculative record never crosses levels.
            try std.testing.expectEqual(a == b, std.meta.eql(ka.next(64).?, kb.next(64).?));
        }
    }

    try std.testing.expect(decodeKey(63, .submit).next(64) == null);
}
