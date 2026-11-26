//! The record ring of the decode step: two command buffers, each keyed by the step it holds.
//!
//! Comptime. `Slot` and `RecordRing` are generic over the key type.
//!
//! Role. While the GPU runs the submitted buffer, the runner records the predicted next step into
//! the other buffer. The next step then submits that buffer with no record on its critical path.
//! This file holds the control state alone. The decode ring of a fused runner owns the command
//! buffers and calls these transitions around each Vulkan action.
//!
//! States. A buffer is `free`, `recorded k`, or `pending k`. `recorded k` holds the
//! commands of key `k` that the GPU has not seen. `pending k` is submitted and
//! not yet complete. The invariants, which every transition asserts:
//!
//!   1. No record writes a pending buffer.
//!   2. At most one buffer is pending.
//!   3. A step starts only when no buffer is pending.
//!   4. A hit submits a buffer whose key equals the wanted key.
//!   5. A pending buffer stays pending until `complete` of that buffer.
//!
//! Waits. Only a wait that reports success or a lost device ends a submission. A lost device
//! counts as complete for every object. Every other failed wait leaves the GPU free to run the
//! buffer, so the buffer stays pending. The ring then keeps the error in `wait_error`, and every
//! later step returns it. Outside a step, `wait_error` is set exactly when a buffer is pending.
//! `WaitGuard` holds the same rule for a submission that has no ring, such as a one-shot submit.
//!
//! Records. A record is two transitions, `beginRecord` and `endRecord`, so a record that fails
//! half way leaves the buffer `free` and never a stale `recorded`. A key holds everything the
//! record reads that can change between steps, so equal keys give equal command streams.
//! The token is data in the staging buffer, and the descriptor sets are fixed at init. The level
//! is part of the key, so a recording made at one level never replays at another.

const std = @import("std");
const prop = @import("utils/prop.zig");
const span = @import("span.zig");

// -- types ------------------------------------------------------------------

/// The errors of a wait on a submission. `WaitTimeout` is a wait that ended before the fence
/// signaled.
pub const WaitError = error{ OutOfHostMemory, OutOfDeviceMemory, DeviceLost, Unknown, WaitTimeout };

/// What a wait result tells about the submission: it ended, or the GPU can still run it.
pub const WaitEnd = enum { complete, pending };

/// The stored error of a failed wait on one submission that has no ring. The owner refuses every
/// later entry while the error is set, because the GPU can still run the submission.
pub const WaitGuard = struct {
    err: ?WaitError = null,

    /// Refuse an entry while the submission can be pending.
    pub fn check(self: WaitGuard) WaitError!void {
        if (self.err) |err| return err;
    }

    /// Apply the result of a wait on the submission. The caller then returns that result.
    pub fn end(self: *WaitGuard, result: WaitError!void) void {
        self.err = pendingError(result);
        std.debug.assert((self.err == null) == (waitEnd(result) == .complete));
    }

    /// Apply the result of a queue wait at a drain. A queue wait covers every submission of the
    /// queue, so it ends the submission of a guard that holds an error. A guard with no error has
    /// no pending submission, so the result does not change it.
    pub fn drain(self: *WaitGuard, result: WaitError!void) void {
        if (self.err == null) return;

        self.end(result);
    }
};

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
        /// The error of a failed wait that left a buffer pending. Every later step returns it.
        wait_error: ?WaitError = null,

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

        /// Refuse a step after a failed wait that left a buffer pending.
        pub fn checkWait(self: *const Self) WaitError!void {
            if (self.wait_error) |err| {
                std.debug.assert(self.pendingCount() == 1);
                return err;
            }
        }

        /// Pick the buffer for wanted key `w`. A buffer that holds `recorded w`
        /// is a hit. A miss names buffer `a`, which the caller then records.
        pub fn prepare(self: *const Self, w: Key) Prepared {
            std.debug.assert(self.pendingCount() == 0);
            std.debug.assert(self.wait_error == null);

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

        /// The GPU finished buffer `c`. A one-time-submit buffer cannot run again, so
        /// it becomes free. No buffer is pending after it, so no wait error stays.
        pub fn complete(self: *Self, c: Buf) void {
            std.debug.assert(self.isPending(c));

            self.slots[@intFromEnum(c)] = .free;
            self.wait_error = null;
            std.debug.assert(self.pendingCount() == 0);
        }

        /// Apply the result of the wait on pending buffer `c`, and return that result. A wait
        /// that leaves `c` pending keeps its error for every later step.
        pub fn endWait(self: *Self, c: Buf, result: WaitError!void) WaitError!void {
            std.debug.assert(self.isPending(c));
            std.debug.assert(self.wait_error == null);

            if (pendingError(result)) |err| {
                self.wait_error = err;
            } else {
                self.complete(c);
            }

            std.debug.assert((self.wait_error == null) == !self.isPending(c));
            return result;
        }

        /// Free both buffers before any other submit on the runner. Returns
        /// the count of recordings that no step submitted.
        pub fn invalidate(self: *Self) u2 {
            std.debug.assert(self.pendingCount() == 0);
            std.debug.assert(self.wait_error == null);

            var wasted: u2 = 0;
            for (&self.slots) |*s| {
                if (s.* == .recorded) {
                    wasted += 1;
                }

                s.* = .free;
            }

            return wasted;
        }

        /// Forget each recording after an error return of a decode step. A pending buffer
        /// stays pending, because the GPU can still run it.
        pub fn drop(self: *Self) void {
            for (&self.slots) |*s| {
                if (s.* != .pending) {
                    s.* = .free;
                }
            }

            std.debug.assert(self.pendingCount() <= 1);
        }
    };
}

// -- api --------------------------------------------------------------------

/// The decode key of `pos` at `level`.
pub fn decodeKey(pos: u32, level: span.GpuLevel) DecodeKey {
    return .{ .pos = pos, .opts = .{ .level = level } };
}

/// The end of the submission after a wait with `result`. A lost device counts as complete for
/// every object. An out-of-memory error changes no object. A timeout leaves the fence unsignaled,
/// and an unknown error states nothing.
pub fn waitEnd(result: WaitError!void) WaitEnd {
    result catch |err| switch (err) {
        error.DeviceLost => return .complete,
        error.OutOfHostMemory, error.OutOfDeviceMemory, error.Unknown, error.WaitTimeout => {
            return .pending;
        },
    };

    return .complete;
}

/// The error that a wait with `result` leaves stored: null when the wait ends the submission.
pub fn pendingError(result: WaitError!void) ?WaitError {
    if (waitEnd(result) == .complete) return null;

    return if (result) |_| unreachable else |err| err;
}

// -- tests ------------------------------------------------------------------

const Ring = RecordRing(u32);

/// The place where a test step fails. Each place is an error return of `DecodeRing.step`.
const Fail = union(enum) {
    none,
    /// The record of a miss fails after its `beginRecord`.
    record,
    /// The queue refuses the submit, so no buffer becomes pending.
    submit,
    /// The speculative record fails after its `beginRecord`.
    speculate,
    /// The wait on the fence returns this error.
    wait: WaitError,
};

const StepError = WaitError || error{ RecordFailed, SubmitFailed };

/// One decode step the way `DecodeRing.step` drives the ring, without the GPU. `fail` names the
/// call that fails. A failure that does not apply, such as a record failure on a hit, does not
/// happen.
fn stepOnce(r: *Ring, w: u32, next: ?u32, wasted: *u32, fail: Fail) StepError!Prepared {
    try r.checkWait();
    errdefer r.drop();

    const p = r.prepare(w);
    if (!p.hit) {
        if (r.beginRecord(p.buf)) {
            wasted.* += 1;
        }

        if (fail == .record) return error.RecordFailed;
        r.endRecord(p.buf, w);
    }

    if (fail == .submit) return error.SubmitFailed;
    r.submit(p.buf, w);
    var speculated: error{RecordFailed}!void = {};
    if (next) |k| {
        const o = r.speculateTarget(p.buf);
        if (!r.holds(o, k)) {
            if (r.beginRecord(o)) {
                wasted.* += 1;
            }

            if (fail == .speculate) {
                speculated = error.RecordFailed;
            } else {
                r.endRecord(o, k);
            }
        }
    }

    std.debug.assert(r.pendingCount() == 1);
    const waited: WaitError!void = switch (fail) {
        .wait => |err| err,
        else => {},
    };
    try r.endWait(p.buf, waited);
    try speculated;
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
    try std.testing.expectEqual(Ring.SlotT{ .pending = 1 }, r.get(.a));
    try std.testing.expectEqual(@as(u2, 1), r.pendingCount());
}

test "RecordRing: only a success or a lost device ends a submission" {
    const cases = [_]struct { result: WaitError!void, end: WaitEnd }{
        .{ .result = {}, .end = .complete },
        .{ .result = error.DeviceLost, .end = .complete },
        .{ .result = error.OutOfHostMemory, .end = .pending },
        .{ .result = error.OutOfDeviceMemory, .end = .pending },
        .{ .result = error.Unknown, .end = .pending },
        .{ .result = error.WaitTimeout, .end = .pending },
    };
    for (cases) |case| {
        try std.testing.expectEqual(case.end, waitEnd(case.result));

        // The wait returns its own result, and the ring keeps the error only while pending.
        var r: Ring = .{};
        r.endRecord(.b, 3);
        r.submit(.b, 3);
        try std.testing.expectEqual(case.result, r.endWait(.b, case.result));
        switch (case.end) {
            .complete => try std.testing.expectEqual(Ring{}, r),
            .pending => {
                try std.testing.expectEqual(Ring.SlotT{ .pending = 3 }, r.get(.b));
                try std.testing.expectError(r.wait_error.?, case.result);
            },
        }
    }
}

test "WaitGuard: a failed wait refuses every entry until a wait ends the submission" {
    const cases = [_]struct { result: WaitError!void, stays: bool }{
        .{ .result = {}, .stays = false },
        .{ .result = error.DeviceLost, .stays = false },
        .{ .result = error.OutOfHostMemory, .stays = true },
        .{ .result = error.OutOfDeviceMemory, .stays = true },
        .{ .result = error.Unknown, .stays = true },
        .{ .result = error.WaitTimeout, .stays = true },
    };
    for (cases) |first| {
        var g: WaitGuard = .{};
        g.end(first.result);
        if (first.stays) {
            const err = if (first.result) |_| unreachable else |e| e;
            try std.testing.expectError(err, g.check());
            // A second check returns the same error, because no check clears it.
            try std.testing.expectError(err, g.check());
        } else {
            try g.check();
        }

        // A drain ends the submission of a stored error only when the drain itself ends it.
        for (cases) |second| {
            var d = g;
            d.drain(second.result);
            try std.testing.expectEqual(first.stays and second.stays, d.err != null);
        }
    }
}

test "RecordRing: a failed wait refuses every later step until the complete" {
    var r: Ring = .{};
    var wasted: u32 = 0;
    _ = try stepOnce(&r, 5, 6, &wasted, .none);
    try std.testing.expectError(
        error.OutOfDeviceMemory,
        stepOnce(&r, 6, 7, &wasted, .{ .wait = error.OutOfDeviceMemory }),
    );
    // The drop forgets the speculation of key 7 and keeps the submitted buffer.
    const failed = r;
    try std.testing.expectEqual(Ring.SlotT{ .pending = 6 }, r.get(.b));
    try std.testing.expectEqual(Ring.SlotT.free, r.get(.a));
    try std.testing.expectError(error.OutOfDeviceMemory, stepOnce(&r, 7, 8, &wasted, .none));
    try std.testing.expectError(error.OutOfDeviceMemory, stepOnce(&r, 6, null, &wasted, .none));
    try std.testing.expectEqual(failed, r);
    r.complete(.b);
    try std.testing.expectEqual(Ring{}, r);
    try std.testing.expect(!(try stepOnce(&r, 7, 8, &wasted, .none)).hit);
}

test "RecordRing: a decode run of n steps hits n - 1 times after the first miss" {
    var r: Ring = .{};
    var wasted: u32 = 0;
    const n: u32 = 64;
    var hits: u32 = 0;
    var pos: u32 = 100;
    while (pos < 100 + n) : (pos += 1) {
        const p = try stepOnce(&r, pos, pos + 1, &wasted, .none);
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
    _ = try stepOnce(&r, 5, 6, &wasted, .none);
    try std.testing.expectEqual(@as(u2, 1), r.invalidate());
    try std.testing.expect(!(try stepOnce(&r, 6, 7, &wasted, .none)).hit);
    try std.testing.expect((try stepOnce(&r, 7, null, &wasted, .none)).hit);
    // A wanted key that the speculation did not predict also misses.
    _ = try stepOnce(&r, 20, 21, &wasted, .none);
    try std.testing.expect(!(try stepOnce(&r, 30, 31, &wasted, .none)).hit);
    try std.testing.expectEqual(@as(u32, 1), wasted);
}

test "RecordRing: a rewind keeps the recording that holds the predicted key" {
    var r: Ring = .{};
    var wasted: u32 = 0;
    const miss_a: Prepared = .{ .buf = .a, .hit = false };
    const hit_a: Prepared = .{ .buf = .a, .hit = true };
    const hit_b: Prepared = .{ .buf = .b, .hit = true };
    try std.testing.expectEqual(miss_a, try stepOnce(&r, 5, 6, &wasted, .none));
    try std.testing.expectEqual(hit_b, try stepOnce(&r, 6, 7, &wasted, .none));
    try std.testing.expectEqual(hit_a, try stepOnce(&r, 7, 8, &wasted, .none));
    try std.testing.expectEqual(miss_a, try stepOnce(&r, 7, 8, &wasted, .none));
    try std.testing.expectEqual(Ring.SlotT{ .recorded = 8 }, r.get(.b));
    try std.testing.expectEqual(hit_b, try stepOnce(&r, 8, 9, &wasted, .none));
    try std.testing.expectEqual(@as(u32, 0), wasted);
}

test "RecordRing: a failed speculative record leaves the ring empty after the complete" {
    var r: Ring = .{};
    r.endRecord(.a, 7);
    r.endRecord(.b, 5);
    r.submit(.a, 7);
    try std.testing.expect(r.beginRecord(r.speculateTarget(.a)));
    r.complete(.a);
    try std.testing.expectEqual(Ring{}, r);
}

/// A random failure: no failure in half the steps, else one of the four places.
fn randomFail(rnd: std.Random) Fail {
    const wait_errors = [_]WaitError{
        error.OutOfHostMemory,
        error.OutOfDeviceMemory,
        error.DeviceLost,
        error.Unknown,
        error.WaitTimeout,
    };
    return switch (rnd.uintLessThan(u8, 8)) {
        0...3 => .none,
        4 => .record,
        5 => .submit,
        6 => .speculate,
        else => .{ .wait = wait_errors[rnd.uintLessThan(usize, wait_errors.len)] },
    };
}

fn checkRingModel(_: void, seed: u64) anyerror!void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const rnd = prng.random();
    var r: Ring = .{};
    var wasted: u32 = 0;
    var pos: u32 = rnd.uintLessThan(u32, 8);
    // The buffer and key that a failed wait left pending. Only the drain below clears it.
    var owed: ?struct { buf: Buf, key: u32 } = null;
    // The reference of the wait rule, apart from `waitEnd`: a lost device is the one error that
    // ends a submission.
    const wait_ends = [_]struct { err: WaitError, ends: bool }{
        .{ .err = error.OutOfHostMemory, .ends = false },
        .{ .err = error.OutOfDeviceMemory, .ends = false },
        .{ .err = error.DeviceLost, .ends = true },
        .{ .err = error.Unknown, .ends = false },
        .{ .err = error.WaitTimeout, .ends = false },
    };
    for (0..64) |_| {
        switch (rnd.uintLessThan(u8, 4)) {
            // A decode step at the next position, the common case.
            0, 1 => pos += 1,
            // A jump: a new turn or a rewind.
            2 => pos = rnd.uintLessThan(u32, 16),
            // Another submit on the runner, or the drain of `deinit` after a failed wait.
            else => if (owed) |o| {
                r.complete(o.buf);
                owed = null;
            } else {
                _ = r.invalidate();
            },
        }

        const before = r;
        const next: ?u32 = if (rnd.boolean()) pos + 1 else null;
        const fail = randomFail(rnd);
        const stepped = stepOnce(&r, pos, next, &wasted, fail);
        if (owed) |o| {
            // A failed wait refuses the step and changes nothing.
            try std.testing.expectError(before.wait_error.?, stepped);
            try std.testing.expectEqual(before, r);
            try std.testing.expectEqual(Ring.SlotT{ .pending = o.key }, r.get(o.buf));
            continue;
        }

        try std.testing.expectEqual(@as(?WaitError, null), before.wait_error);
        try std.testing.expectEqual(@as(u2, 0), before.pendingCount());
        // A step picks the buffer that holds `recorded pos`, buffer `a` first, else buffer `a`.
        const chosen: Buf = if (!before.holds(.a, pos) and before.holds(.b, pos)) .b else .a;
        const kept = fail == .wait and for (wait_ends) |e| {
            if (e.err == fail.wait) break !e.ends;
        } else unreachable;
        const p = stepped catch |err| {
            if (kept) {
                // The wait left the submitted buffer pending, and the drop freed the other one.
                try std.testing.expectEqual(@as(anyerror, fail.wait), err);
                try std.testing.expectEqual(Ring.SlotT{ .pending = pos }, r.get(chosen));
                try std.testing.expectEqual(Ring.SlotT.free, r.get(chosen.other()));
                try std.testing.expectEqual(@as(?WaitError, fail.wait), r.wait_error);
                owed = .{ .buf = chosen, .key = pos };
            } else {
                // Every other error return ends with no buffer pending and no recording.
                try std.testing.expect(fail != .none);
                try std.testing.expectEqual(Ring{}, r);
            }

            continue;
        };

        // A hit needs `recorded pos` in the chosen buffer before the step.
        try std.testing.expectEqual(chosen, p.buf);
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
        } else {
            try std.testing.expectEqual(before.get(p.buf.other()), r.get(p.buf.other()));
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
