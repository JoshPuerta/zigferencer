//! The slot state machine of the double-buffered host-to-device upload ring in
//! `ops_vulkan/stream_ring.zig`, checked at compile time here. This file holds
//! the control table alone: no Vulkan, no allocation, no I/O. The GPU type
//! mirrors this table, and the anchor trace below replays step for step in it.
//!
//! Every state change in the ring routes through `next`. A transition the
//! table does not name is a bug in the caller, not a runtime condition, so
//! `next` returns null and the caller asserts.

const std = @import("std");

// -- types ------------------------------------------------------------------

/// The life of one slot. A slot is `free` until a producer takes it, and it
/// returns to `free` only after a consumer has read the uploaded bytes.
pub const SlotState = enum(u2) { free, filling, in_flight, ready };

/// The four operations a caller can apply to a slot.
pub const OpTag = enum { acquire, submit, wait, release };

/// One entry of a replayable trace.
pub const Step = struct { slot: u8, op: OpTag };

// -- api --------------------------------------------------------------------

/// Slot count. Two is the smallest depth that lets one upload run on the
/// device while the host fills the next one.
pub const depth: u32 = 2;

/// The whole legality rule of the ring. A null result means the operation
/// does not apply in that state.
pub fn next(s: SlotState, op: OpTag) ?SlotState {
    return switch (s) {
        .free => switch (op) {
            .acquire => .filling,
            else => null,
        },
        .filling => switch (op) {
            .submit => .in_flight,
            else => null,
        },
        .in_flight => switch (op) {
            .wait => .ready,
            else => null,
        },
        .ready => switch (op) {
            .release => .free,
            else => null,
        },
    };
}

/// Two full rotations of a depth-2 ring, ending drained. The trace fills the
/// ring three times, so it exercises the backpressure point that a shorter
/// trace never reaches.
pub const anchor_trace = [16]Step{
    .{ .slot = 0, .op = .acquire },
    .{ .slot = 0, .op = .submit },
    .{ .slot = 1, .op = .acquire },
    .{ .slot = 1, .op = .submit },
    .{ .slot = 0, .op = .wait },
    .{ .slot = 0, .op = .release },
    .{ .slot = 0, .op = .acquire },
    .{ .slot = 0, .op = .submit },
    .{ .slot = 1, .op = .wait },
    .{ .slot = 1, .op = .release },
    .{ .slot = 1, .op = .acquire },
    .{ .slot = 1, .op = .submit },
    .{ .slot = 0, .op = .wait },
    .{ .slot = 0, .op = .release },
    .{ .slot = 1, .op = .wait },
    .{ .slot = 1, .op = .release },
};

// -- tests ------------------------------------------------------------------

// The three tests below run entirely at comptime, so any drift in the table
// or in the trace fails the build instead of a test run.

test "ring_fsm: the transition table, exhaustively" {
    comptime {
        // Rows follow the `SlotState` declaration order, columns the `OpTag`
        // order. The nulls are the load-bearing half. They pin the negative
        // space that the asserts in the ring depend on.
        const expected = [4][4]?SlotState{
            .{ .filling, null, null, null },
            .{ null, .in_flight, null, null },
            .{ null, null, .ready, null },
            .{ null, null, null, .free },
        };
        if (std.enums.values(SlotState).len != 4) @compileError("SlotState gained or lost a state, so the pinned table is stale");
        if (std.enums.values(OpTag).len != 4) @compileError("OpTag gained or lost an operation, so the pinned table is stale");
        for (std.enums.values(SlotState), 0..) |s, si| {
            for (std.enums.values(OpTag), 0..) |op, oi| {
                if (next(s, op) != expected[si][oi]) @compileError(std.fmt.comptimePrint(
                    "ring_fsm.next({s}, {s}) drifted from the pinned table",
                    .{ @tagName(s), @tagName(op) },
                ));
            }
        }
    }
}

test "ring_fsm: ready is entered only by wait" {
    comptime {
        // The ring exposes a slot's bytes only in `ready`. A second door into
        // that state lets a consumer read a buffer the device still owns.
        for (std.enums.values(SlotState)) |s| {
            for (std.enums.values(OpTag)) |op| {
                if (next(s, op) == .ready and (op != .wait or s != .in_flight)) @compileError(std.fmt.comptimePrint(
                    "ring_fsm.next({s}, {s}) reaches ready without a wait on an in-flight slot",
                    .{ @tagName(s), @tagName(op) },
                ));
            }
        }
    }
}

test "ring_fsm: the anchor trace replays" {
    comptime {
        // 1-based step numbers at which the trace holds both slots in flight.
        const full_points = [3]usize{ 4, 8, 12 };

        var states = [_]SlotState{.free} ** depth;
        for (anchor_trace, 1..) |step, n| {
            if (step.slot >= depth) @compileError(std.fmt.comptimePrint("anchor_trace step {d} names a slot outside the ring", .{n}));
            const to = next(states[step.slot], step.op) orelse
                @compileError(std.fmt.comptimePrint("anchor_trace step {d} is not a legal transition", .{n}));
            states[step.slot] = to;

            var in_flight: u32 = 0;
            for (states) |s| {
                if (s == .in_flight) in_flight += 1;
            }
            if (in_flight > depth) @compileError(std.fmt.comptimePrint("anchor_trace step {d} exceeds the ring depth", .{n}));

            var expect_full = false;
            for (full_points) |p| {
                if (p == n) expect_full = true;
            }
            if (expect_full) {
                if (in_flight != depth) @compileError(std.fmt.comptimePrint("anchor_trace step {d} does not fill the ring", .{n}));
                // A full ring must refuse every acquire. That is the
                // backpressure the ring reports to its caller as a null.
                for (states) |s| {
                    if (next(s, .acquire) != null) @compileError(std.fmt.comptimePrint("anchor_trace step {d} leaves an acquirable slot in a full ring", .{n}));
                }
            } else {
                if (in_flight == depth) @compileError(std.fmt.comptimePrint("anchor_trace step {d} fills the ring outside the pinned full points", .{n}));
            }
        }

        for (states) |s| {
            if (s != .free) @compileError("anchor_trace ends without draining the ring");
        }
    }
}
