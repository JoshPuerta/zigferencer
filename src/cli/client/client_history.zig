//! The client side of compaction: the transcript with the numbers that weigh it, and one
//! compaction cycle in flight.

const std = @import("std");

const limits = @import("../../core/limits.zig");
const protocol = @import("../../serve/protocol.zig");
const client_mod = @import("../../serve/client.zig");
const transcript_mod = @import("client_transcript.zig");

const Client = client_mod.Client;
const Transcript = transcript_mod.Transcript;

// -- types ------------------------------------------------------------------

/// The client side of compaction: the transcript and the numbers that weigh
/// it.
pub const History = struct {
    t: *Transcript,
    /// The `kv_pos` of a session opened with the original system prompt. Each
    /// re-open measures it again.
    system_rows: u32,
    window: u32,
    /// The `--reserve` value, or null for `reserveFor`.
    fixed_reserve: ?u32,

    pub fn init(
        arena: std.mem.Allocator,
        system_rows: u32,
        window: u32,
        fixed_reserve: ?u32,
    ) !History {
        if (fixed_reserve) |r| std.debug.assert(r < window);

        const t = try arena.create(Transcript);
        t.init();
        return .{
            .t = t,
            .system_rows = system_rows,
            .window = window,
            .fixed_reserve = fixed_reserve,
        };
    }

    pub fn effectiveReserve(h: *const History) u32 {
        const r = h.fixed_reserve orelse transcript_mod.reserveFor(h.window, h.t.longest_cost);
        std.debug.assert(r < h.window);
        return r;
    }

    pub fn budget(h: *const History, reserve: u32) transcript_mod.Budget {
        return .{ .window = h.window, .reserve = reserve, .system_rows = h.system_rows };
    }

    pub fn plan(h: *const History, reserve: u32) transcript_mod.Plan {
        return transcript_mod.plan(h.t, h.budget(reserve));
    }

    /// Keep the system prompt only, for a cycle that cannot make progress.
    pub fn keepNothing(h: *History, cycle: *Cycle) void {
        cycle.plan = .{ .kept = 0, .dropped = cycle.plan.kept, .estimate = h.system_rows };
        h.t.clear();
    }

    pub fn retry(h: *const History, q: u32) transcript_mod.Retry {
        return transcript_mod.retryPlan(h.system_rows, h.window, h.effectiveReserve(), q);
    }

    /// Plan a compaction with `reserve`, and drop the turns that it does not
    /// keep. The window is fixed for the connection, so a dropped turn never
    /// fits again.
    pub fn shrink(h: *History, reserve: u32) transcript_mod.Plan {
        const p = h.plan(reserve);
        h.t.dropOldest(p.dropped);
        std.debug.assert(h.t.count() == p.kept);
        return p;
    }

    /// The kept pairs of the append frame that starts at turn `from`. They
    /// borrow the transcript text.
    pub fn frame(
        h: *const History,
        from: u32,
        out: *[limits.max_append_pairs]protocol.Pair,
    ) struct { end: u32, pairs: []const protocol.Pair } {
        const end = transcript_mod.frameEnd(h.t, from, h.t.count());
        for (from..end, 0..) |i, j| {
            out[j] = h.t.pair(@intCast(i));
        }

        return .{ .end = end, .pairs = out[0 .. end - from] };
    }
};

/// One compaction cycle in flight.
pub const Cycle = struct {
    plan: transcript_mod.Plan,
    /// The held turns before the first plan of the cycle, for the report.
    total: u32,
    reserve: u32,
    /// The next kept turn to send, and the end of the frame in flight.
    next: u32 = 0,
    frame_end: u32 = 0,
    base: u32 = 0,
    /// The restarts after a refused frame. Each one keeps fewer pairs, so a
    /// cycle opens at most `total + 1` sessions for them.
    refits: u32 = 0,
    /// A failed append of another kind opens the session once more.
    reopened: bool = false,
};

// -- api --------------------------------------------------------------------

/// The numbers of a `turn_too_long` refusal of one kind. Any other error, or
/// a detail the client cut, gives null.
pub fn tooLong(le: ?Client.LastError, kind: transcript_mod.TooLong.Kind) ?transcript_mod.TooLong {
    const e = le orelse return null;
    if (e.code != .turn_too_long or e.truncated) return null;

    const tl = transcript_mod.parseTooLong(e.detail()) orelse return null;
    return if (tl.kind == kind) tl else null;
}

/// The token cap that each `generate` sends. A flag of 0 asks for the whole
/// window. A flag above the window passes through, because the server stops a
/// turn at the window.
pub fn turnBudget(flag: u32, max_seq: u32) u32 {
    std.debug.assert(max_seq > 0);

    const budget = if (flag == 0) max_seq else flag;
    std.debug.assert(budget > 0);
    return budget;
}

// -- tests ------------------------------------------------------------------

test "turnBudget gives the whole window for 0 and passes a flag through" {
    try std.testing.expectEqual(@as(u32, 512), turnBudget(0, 512));
    try std.testing.expectEqual(@as(u32, 1), turnBudget(0, 1));
    try std.testing.expectEqual(@as(u32, 1), turnBudget(1, 512));
    try std.testing.expectEqual(@as(u32, 4096), turnBudget(4096, 512));
}
