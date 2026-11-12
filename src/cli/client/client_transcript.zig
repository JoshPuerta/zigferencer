//! The transcript of a client chat and the decisions that compact it. The client cannot
//! tokenize, so it weighs each turn in positions and keeps the newest turns that fit beside the
//! system rows and a reserve.
//!
//! Comptime. Asserts order the transcript byte limits, and `splitFormat` splits a format string
//! at comptime.
//!
//! Costs. A live delta errs high when the turn thought, because a kept pair drops its reasoning. A
//! refused append frame teaches a cost of at least the real one, so each restart keeps fewer pairs.
//!
//! Storage. The held turns pack from offset zero, and the pending text sits right after them. A
//! turn that the server refuses before prefill must not cost the held turns, so `begin` never
//! evicts. A growing reply evicts the oldest held turns, and the eviction moves the pending text.
//!
//! Barrier. A turn too large for the byte budget, or a reply that the caller cut, leaves a gap in
//! the record. The older turns without it misread the chat, so they leave with it.

const std = @import("std");
const protocol = @import("../../serve/protocol.zig");
const limits = @import("../../core/limits.zig");

const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

/// The pair bytes that the held turns can fill, so one held turn fits one `session_append` frame.
pub const pair_bytes_max: u32 = protocol.append_pairs_bytes_max;

const pair_overhead_bytes: u32 = protocol.pair_overhead_bytes;

/// The largest prompt of one turn, the stdin line bound of the client.
pub const prompt_bytes_max: u32 = 4096;

/// A prompt holds at least one byte, and a reply can be empty.
const min_pair_bytes: u32 = pair_overhead_bytes + 1;

const turns_max: u32 = pair_bytes_max / min_pair_bytes;

/// The extra room holds a pending prompt, because `begin` never evicts.
const text_bytes_cap: u32 = pair_bytes_max + prompt_bytes_max;

/// The divisors of the bounds of `reserveFor`.
pub const reserve_floor_divisor: u32 = 8;
pub const reserve_ceiling_divisor: u32 = 2;

const turn_pieces = splitFormat(protocol.turn_too_long_fmt);
const system_pieces = splitFormat(protocol.system_too_long_fmt);

comptime {
    assert(prompt_bytes_max < pair_bytes_max);
    assert(turns_max >= 1);
    assert(@as(u64, turns_max) * min_pair_bytes <= pair_bytes_max);
}

// -- types ------------------------------------------------------------------

/// One held turn. Its prompt and its reply sit back to back at `start`.
const Turn = struct {
    start: u32,
    prompt_len: u32,
    reply_len: u32,
    cost: u32,

    fn pairBytes(turn: Turn) u32 {
        return pair_overhead_bytes + turn.prompt_len + turn.reply_len;
    }
};

/// The turn in flight. Its text sits right after the held text.
const Pending = struct {
    prompt_len: u32,
    reply_len: u32,
    /// The held turns that the reply already pushed out.
    evicted: u32,
    /// The turn ends as a barrier: the reply outgrew the budget, or `markCut` cut it.
    overflow: bool,
};

/// What one `finish` did to the held turns.
pub const Recorded = struct {
    /// The held turns that left the transcript for this turn.
    evicted: u32,
    /// The turn and every held turn left the transcript.
    barrier: bool,
};

/// The window arithmetic of one compaction, in positions.
pub const Budget = struct {
    window: u32,
    /// The positions that stay free after the re-open, at most the window.
    reserve: u32,
    /// The `kv_pos` of a session opened with the original system prompt.
    system_rows: u32,
};

/// The split of the held turns. The `dropped` oldest turns leave, and `estimate` is the position
/// after the kept pairs.
pub const Plan = struct {
    kept: u32,
    dropped: u32,
    estimate: u32,
};

/// The numbers of a `turn_too_long` refusal.
pub const TooLong = struct {
    kind: Kind,
    tokens: u32,
    /// The positions left for a turn, or the window for a system prompt.
    room: u32,

    pub const Kind = enum { turn, system };
};

/// The answer to a refused turn.
pub const Retry = union(enum) {
    /// The prompt does not fit beside the system prompt alone.
    refuse,
    /// Compact with this reserve, re-open, and send the prompt once more.
    retry: u32,
};

/// The record of a chat in fixed storage of about 180 KiB. The caller calls `init` in place.
pub const Transcript = struct {
    /// The pair bytes of the held turns.
    held_bytes: u32,
    n: u32,
    /// The largest measured turn cost, barrier and dropped turns included. It sizes the next turn.
    longest_cost: u32,
    pending: ?Pending,
    turns: [turns_max]Turn,
    text: [text_bytes_cap]u8,

    pub fn init(t: *Transcript) void {
        t.held_bytes = 0;
        t.n = 0;
        t.longest_cost = 0;
        t.pending = null;
        t.assertHeld();
        assert(t.count() == 0);
    }

    pub fn count(t: *const Transcript) u32 {
        return t.n;
    }

    /// The positions of turn `i`, oldest first: the live `kv_pos` delta, or the exact cost once an
    /// append measured it.
    pub fn cost(t: *const Transcript, i: u32) u32 {
        assert(i < t.n);

        return t.turns[i].cost;
    }

    /// Replace the cost of turn `i` with a measured one.
    pub fn setCost(t: *Transcript, i: u32, c: u32) void {
        assert(i < t.n);
        assert(t.pending == null);

        t.turns[i].cost = c;
    }

    pub fn prompt(t: *const Transcript, i: u32) []const u8 {
        assert(i < t.n);

        const turn = t.turns[i];
        return t.text[turn.start..][0..turn.prompt_len];
    }

    pub fn reply(t: *const Transcript, i: u32) []const u8 {
        assert(i < t.n);

        const turn = t.turns[i];
        return t.text[turn.start + turn.prompt_len ..][0..turn.reply_len];
    }

    /// The kept pair of turn `i`. It borrows the transcript text.
    pub fn pair(t: *const Transcript, i: u32) protocol.Pair {
        return .{ .user = t.prompt(i), .assistant = t.reply(i) };
    }

    /// Start a turn.
    pub fn begin(t: *Transcript, prompt_text: []const u8) void {
        assert(t.pending == null);
        assert(prompt_text.len >= 1);
        assert(prompt_text.len <= prompt_bytes_max);

        const at = t.heldTextLen();
        @memcpy(t.text[at..][0..prompt_text.len], prompt_text);
        t.pending = .{
            .prompt_len = @intCast(prompt_text.len),
            .reply_len = 0,
            .evicted = 0,
            .overflow = false,
        };
        t.assertHeld();
    }

    /// Add reply text to the pending turn. Past the budget of an empty history, the turn stops
    /// growing and ends as a barrier.
    pub fn appendReply(t: *Transcript, bytes: []const u8) void {
        const p = &t.pending.?;
        if (p.overflow) return;

        const size = @as(u64, pair_overhead_bytes) + p.prompt_len + p.reply_len + bytes.len;
        if (size > pair_bytes_max) {
            p.overflow = true;
            return;
        }

        const k = t.evictionCount(size);
        t.evictOldest(k);
        p.evicted += k;

        const at = t.heldTextLen() + p.prompt_len + p.reply_len;
        @memcpy(t.text[at..][0..bytes.len], bytes);
        p.reply_len += @intCast(bytes.len);
        t.assertHeld();
        assert(t.held_bytes + size <= pair_bytes_max);
    }

    /// Mark the pending reply as cut by a buffer of the caller. A cut reply is not whole, so the
    /// turn ends as a barrier.
    pub fn markCut(t: *Transcript) void {
        t.pending.?.overflow = true;
    }

    /// Commit the pending turn with its measured cost. The reply loses its leading newlines, as
    /// after a think block. A sampled user-defined tag re-encodes as plain text and costs more
    /// positions than the live turn. A sampled control token sends no text, so it costs fewer.
    pub fn finish(t: *Transcript, turn_cost: u32) Recorded {
        t.longest_cost = @max(t.longest_cost, turn_cost);
        const p = &t.pending.?;
        if (p.overflow) return t.barrier();

        const reply_at = t.heldTextLen() + p.prompt_len;
        const raw = t.text[reply_at..][0..p.reply_len];
        const kept = std.mem.trimStart(u8, raw, "\n");
        const offset = raw.len - kept.len;
        @memmove(t.text[reply_at..][0..kept.len], raw[offset..][0..kept.len]);
        p.reply_len = @intCast(kept.len);

        const size = @as(u64, pair_overhead_bytes) + p.prompt_len + p.reply_len;
        if (size > pair_bytes_max) return t.barrier();

        // The eviction moves the pending text with the held text, so the turn stays pending until
        // the eviction ends.
        const k = t.evictionCount(size);
        t.evictOldest(k);
        p.evicted += k;

        // Each held turn takes at least `min_pair_bytes`, so the count fits.
        assert(t.n < turns_max);
        const done = p.*;

        t.turns[t.n] = .{
            .start = t.heldTextLen(),
            .prompt_len = done.prompt_len,
            .reply_len = done.reply_len,
            .cost = turn_cost,
        };
        t.pending = null;
        t.n += 1;
        t.held_bytes += @intCast(size);
        t.assertHeld();
        return .{ .evicted = done.evicted, .barrier = false };
    }

    /// Drop the pending turn. The result counts the held turns that its reply already pushed out.
    pub fn abandon(t: *Transcript) u32 {
        const p = t.pending.?;
        t.pending = null;
        t.assertHeld();
        return p.evicted;
    }

    pub fn clear(t: *Transcript) void {
        assert(t.pending == null);

        t.n = 0;
        t.held_bytes = 0;
        t.longest_cost = 0;
        t.assertHeld();
    }

    pub fn dropOldest(t: *Transcript, k: u32) void {
        assert(t.pending == null);
        assert(k <= t.n);

        const before = t.n;
        t.evictOldest(k);
        t.assertHeld();
        assert(t.n == before - k);
    }

    /// End the pending turn with every held turn.
    fn barrier(t: *Transcript) Recorded {
        const dropped = t.pending.?.evicted + t.n;
        t.pending = null;
        t.n = 0;
        t.held_bytes = 0;
        t.assertHeld();
        return .{ .evicted = dropped, .barrier = true };
    }

    /// The fewest oldest turns to evict so that `extra` pair bytes fit beside the rest. The loop
    /// runs at most `n` times.
    fn evictionCount(t: *const Transcript, extra: u64) u32 {
        assert(extra <= pair_bytes_max);

        var total: u64 = t.held_bytes;
        var k: u32 = 0;
        while (total + extra > pair_bytes_max) : (k += 1) {
            assert(k < t.n);

            total -= t.turns[k].pairBytes();
        }

        assert(total + extra <= pair_bytes_max);
        return k;
    }

    /// Remove the `k` oldest turns and move the rest of the text, the pending text too, to the
    /// front.
    fn evictOldest(t: *Transcript, k: u32) void {
        assert(k <= t.n);

        if (k == 0) return;

        const held_end = t.heldTextLen();
        const shift = if (k < t.n) t.turns[k].start else held_end;
        const used = held_end + t.pendingTextLen();
        @memmove(t.text[0 .. used - shift], t.text[shift..used]);

        var dropped: u32 = 0;
        for (t.turns[0..k]) |turn| {
            dropped += turn.pairBytes();
        }

        @memmove(t.turns[0 .. t.n - k], t.turns[k..t.n]);
        t.n -= k;
        for (t.turns[0..t.n]) |*turn| {
            turn.start -= shift;
        }

        t.held_bytes -= dropped;
    }

    fn heldTextLen(t: *const Transcript) u32 {
        if (t.n == 0) return 0;
        const last = t.turns[t.n - 1];
        return last.start + last.prompt_len + last.reply_len;
    }

    fn pendingTextLen(t: *const Transcript) u32 {
        const p = t.pending orelse return 0;
        return p.prompt_len + p.reply_len;
    }

    /// The held turns pack from offset zero, and their pair total fits the budget. The loop runs
    /// at most `n` times.
    fn assertHeld(t: *const Transcript) void {
        assert(t.n <= turns_max);
        assert(t.held_bytes <= pair_bytes_max);

        var total: u64 = 0;
        var end: u32 = 0;
        for (t.turns[0..t.n]) |turn| {
            assert(turn.start == end);
            assert(turn.prompt_len >= 1);

            end += turn.prompt_len + turn.reply_len;
            total += turn.pairBytes();
        }

        assert(total == t.held_bytes);
        assert(end + t.pendingTextLen() <= text_bytes_cap);
    }
};

/// The turn in flight at a drop, as the client saw it.
pub const InFlight = struct {
    /// The request id of its `generate`. Request ids start at 1.
    request_id: u32,
    /// The frames that reached the client and that `countsAsGenerated` takes, the end token too.
    received: u32,
};

/// What the server did with the turn in flight at a drop.
pub const Reconciled = union(enum) {
    /// The client holds every token that the server generated.
    complete,
    /// The server generated this many tokens that only the model context holds.
    cut: u32,
    /// The server never ran the turn, so the client sends it again.
    never_ran,
    /// The summary contradicts the frames that the client got. The client keeps its text.
    inconsistent,
};

/// The answer to a refused append frame. A stuck refit repeats the refusal on a restart.
pub const Refit = union(enum) {
    /// The learned costs keep fewer pairs. The dropped turns already left.
    plan: Plan,
    /// The refusal does not lower the kept pairs. The caller keeps the system prompt only.
    stuck,
};

// -- api --------------------------------------------------------------------

/// The compaction of the held turns for `budget`.
pub fn plan(t: *const Transcript, budget: Budget) Plan {
    var weights: [turns_max]u32 = undefined;
    for (0..t.n) |j| {
        weights[j] = t.cost(@intCast(t.n - 1 - j));
    }

    const p = planOver(weights[0..t.n], budget);
    assert(p.kept + p.dropped == t.n);
    return p;
}

/// The end of the `session_append` frame that starts at turn `from`: the most turns up to `to` that
/// fit `limits.max_append_pairs` pairs and `pair_bytes_max` bytes. The pair limit bounds the loop.
pub fn frameEnd(t: *const Transcript, from: u32, to: u32) u32 {
    assert(from < to);
    assert(to <= t.n);

    var bytes: u64 = 0;
    var end = from;
    while (end < to and end - from < limits.max_append_pairs) : (end += 1) {
        const b = t.turns[end].pairBytes();
        if (bytes + b > pair_bytes_max) break;
        bytes += b;
    }

    assert(end > from);
    assert(end - from <= limits.max_append_pairs);
    return end;
}

/// The cost of a finished live turn as a kept pair. A budget or cancel stop leaves the last token
/// pending, outside the `kv_pos` delta. The pair holds it, so the cost counts one more and errs
/// high by at most one per turn.
pub fn liveCost(delta: u32, finish: protocol.FinishReason, gen_tokens: u32) u32 {
    const carried = gen_tokens > 0 and (finish == .max_tokens or finish == .cancelled);
    return delta + @intFromBool(carried);
}

/// Whether a token frame counts in `gen_tokens`. The echo of the open tag of a think-open turn is
/// the only frame on that channel, and it counts as none.
pub fn countsAsGenerated(channel: protocol.Channel) bool {
    return channel != .open_tag;
}

/// Classify the turn in flight at a drop from the `resumed` summary. The server answers a resume
/// only when the session is idle, so the summary is final. It comes from the network, so a
/// contradiction is data.
pub fn reconcile(turn: InFlight, turns_done: u32, last: ?protocol.LastDone) Reconciled {
    assert(turn.request_id >= 1);

    const done = last orelse return if (turn.received == 0) .never_ran else .inconsistent;
    if (turns_done == 0) return .inconsistent;

    // Request ids only grow on one chat, so a later id names a request that the client never sent.
    if (done.request_id > turn.request_id) return .inconsistent;
    if (done.request_id < turn.request_id) {
        return if (turn.received == 0) .never_ran else .inconsistent;
    }
    if (turn.received > done.gen_tokens) return .inconsistent;

    const k = done.gen_tokens - turn.received;
    if (k == 0) return .complete;

    return .{ .cut = k };
}

/// Record the exact cost of each pair of an `appended` reply. `base` is the position before the
/// frame. Returns the position after it.
pub fn learnAppended(t: *Transcript, from: u32, base: u32, positions: []const u32) u32 {
    assert(positions.len >= 1);
    assert(from + positions.len <= t.n);

    var prev = base;
    for (positions, 0..) |pos, j| {
        assert(pos >= prev);

        t.setCost(from + @as(u32, @intCast(j)), pos - prev);
        prev = pos;
    }

    return prev;
}

/// Learn from a refused frame of turns `from..to` that the server counted as `tokens`. The refusal
/// names the sum only, so an excess over the estimates joins the oldest pair of the frame.
pub fn learnRefusal(t: *Transcript, from: u32, to: u32, tokens: u32) void {
    assert(from < to);
    assert(to <= t.n);

    var sum: u64 = 0;
    for (from..to) |j| {
        sum += t.cost(@intCast(j));
    }

    if (tokens > sum) {
        t.setCost(from, t.cost(from) + @as(u32, @intCast(tokens - sum)));
    }
}

/// Learn from a refusal, as `learnRefusal` does, and plan again. A plan that keeps as many pairs
/// as `kept_before` gives `stuck`. The refusal comes from the server, so this is data, not a broken
/// invariant.
pub fn refit(
    t: *Transcript,
    budget: Budget,
    from: u32,
    to: u32,
    tokens: u32,
    kept_before: u32,
) Refit {
    learnRefusal(t, from, to, tokens);
    const p = plan(t, budget);
    if (p.kept >= kept_before) return .stuck;

    t.dropOldest(p.dropped);
    assert(t.count() == p.kept);
    return .{ .plan = p };
}

/// Read the numbers back from the detail of a `turn_too_long` refusal. Any text other than the two
/// protocol formats gives null.
pub fn parseTooLong(detail: []const u8) ?TooLong {
    if (parseFields(turn_pieces, detail)) |f| {
        return .{ .kind = .turn, .tokens = f[0], .room = f[1] };
    }
    if (parseFields(system_pieces, detail)) |f| {
        return .{ .kind = .system, .tokens = f[0], .room = f[1] };
    }

    return null;
}

/// The answer to a refused turn of `q` positions. A compaction that keeps no turn costs exactly
/// `system_rows`, so a prompt that does not fit beside them gets a refusal.
pub fn retryPlan(system_rows: u32, window: u32, reserve: u32, q: u32) Retry {
    assert(window > 0);
    assert(reserve < window);

    if (@as(u64, system_rows) + q > window) return .refuse;

    const headroom = window -| system_rows;
    const r = @min(q +| reserve, headroom);

    // A reserve equal to the window keeps no turn.
    assert(r <= window);
    return .{ .retry = r };
}

/// The reserve of a compaction when the user names none: the longest turn so far, clamped between
/// a floor and a ceiling. The ceiling keeps at least half the window for history.
pub fn reserveFor(window: u32, longest_cost: u32) u32 {
    assert(window > 0);

    const floor = window / reserve_floor_divisor;
    const ceiling = window / reserve_ceiling_divisor;
    assert(floor <= ceiling);

    const r = std.math.clamp(longest_cost, floor, ceiling);
    assert(r >= floor);
    assert(r < window);
    return r;
}

// -- helpers ----------------------------------------------------------------

/// The count of newest turns that fit `cap`. The walk stops at the first overflow, which finds the
/// longest suffix because no weight is negative. The list length bounds the loop.
fn keepCount(cap: u64, weights_newest_first: []const u32) u32 {
    assert(weights_newest_first.len <= turns_max);

    var acc: u64 = 0;
    for (weights_newest_first, 0..) |w, i| {
        if (acc + w > cap) {
            assert(acc <= cap);

            return @intCast(i);
        }

        acc += w;
    }

    assert(acc <= cap);
    return @intCast(weights_newest_first.len);
}

/// `plan` over a weight list, newest first.
fn planOver(weights_newest_first: []const u32, budget: Budget) Plan {
    assert(budget.window > 0);
    assert(budget.reserve <= budget.window);

    const cap = (@as(u64, budget.window) - budget.reserve) -| budget.system_rows;
    const kept = keepCount(cap, weights_newest_first);

    var estimate: u64 = budget.system_rows;
    for (weights_newest_first[0..kept]) |w| {
        estimate += w;
    }

    if (kept > 0) assert(estimate + budget.reserve <= budget.window);

    const n: u32 = @intCast(weights_newest_first.len);
    assert(kept <= n);
    return .{ .kept = kept, .dropped = n - kept, .estimate = @intCast(estimate) };
}

/// The literal text around the two `{d}` fields of a format.
fn splitFormat(comptime fmt: []const u8) [3][]const u8 {
    comptime {
        var pieces: [3][]const u8 = undefined;
        var rest: []const u8 = fmt;
        for (0..2) |i| {
            const at = std.mem.find(u8, rest, "{d}").?;
            pieces[i] = rest[0..at];
            rest = rest[at + 3 ..];
        }

        assert(std.mem.findScalar(u8, rest, '{') == null);
        pieces[2] = rest;
        return pieces;
    }
}

/// Match the literal pieces around two decimal fields. Each field holds at least one digit and fits
/// a u32. `detail.len` bounds the loop over each field.
fn parseFields(pieces: [3][]const u8, detail: []const u8) ?[2]u32 {
    var fields: [2]u32 = undefined;
    var rest = detail;
    for (0..2) |i| {
        if (!std.mem.startsWith(u8, rest, pieces[i])) return null;

        rest = rest[pieces[i].len..];

        var digits: usize = 0;
        while (digits < rest.len and std.ascii.isDigit(rest[digits])) {
            digits += 1;
        }

        if (digits == 0) return null;

        fields[i] = std.fmt.parseInt(u32, rest[0..digits], 10) catch return null;
        rest = rest[digits..];
    }

    if (!std.mem.eql(u8, rest, pieces[2])) return null;

    return fields;
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

fn newTranscript() !*Transcript {
    const t = try testing.allocator.create(Transcript);
    t.init();
    return t;
}

fn record(
    t: *Transcript,
    prompt_text: []const u8,
    reply_text: []const u8,
    turn_cost: u32,
) Recorded {
    t.begin(prompt_text);

    // A turn with no reply text reaches `finish` with no `appendReply`, as in the client, so
    // `finish` does its own eviction.
    if (reply_text.len > 0) {
        t.appendReply(reply_text);
    }

    return t.finish(turn_cost);
}

/// A transcript of one-byte turns with the measured deltas `ds`.
fn weighed(ds: []const u32) !*Transcript {
    const t = try newTranscript();
    for (ds) |d| {
        _ = record(t, "q", "", d);
    }

    return t;
}

/// The costs of the turns that `plan` keeps, oldest first.
fn keptCosts(t: *const Transcript, p: Plan, out: []u32) []const u32 {
    for (0..p.kept) |j| {
        out[j] = t.cost(p.dropped + @as(u32, @intCast(j)));
    }

    return out[0..p.kept];
}

// The anchors read their numbers as system rows and live deltas, so a real
// transcript produces them. They run through `plan` and `retryPlan`.
test "anchor A' keeps the three newest turns" {
    const t = try weighed(&.{ 12, 9, 20, 7, 11 });
    defer testing.allocator.destroy(t);

    const p = plan(t, .{ .window = 64, .reserve = 8, .system_rows = 10 });
    var buf: [8]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 20, 7, 11 }, keptCosts(t, p, &buf));
    try testing.expectEqual(@as(u32, 48), p.estimate);
}

test "anchor B' keeps an exact fit" {
    const t = try weighed(&.{ 8, 38 });
    defer testing.allocator.destroy(t);

    const p = plan(t, .{ .window = 64, .reserve = 8, .system_rows = 10 });
    var buf: [8]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 8, 38 }, keptCosts(t, p, &buf));
    try testing.expectEqual(@as(u32, 56), p.estimate);
}

test "anchor C' keeps nothing when the newest turn alone is too large" {
    const t = try weighed(&.{ 5, 50 });
    defer testing.allocator.destroy(t);

    const p = plan(t, .{ .window = 64, .reserve = 8, .system_rows = 10 });
    try testing.expectEqual(Plan{ .kept = 0, .dropped = 2, .estimate = 10 }, p);
}

test "anchor D' keeps three turns at the scale of a long chat" {
    const t = try weighed(&.{ 700, 300, 900, 500 });
    defer testing.allocator.destroy(t);

    const p = plan(t, .{ .window = 2048, .reserve = 256, .system_rows = 40 });
    var buf: [8]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 300, 900, 500 }, keptCosts(t, p, &buf));
}

test "anchor E' leaves room for the refused prompt" {
    const t = try weighed(&.{ 12, 9, 20, 7, 11 });
    defer testing.allocator.destroy(t);

    const q = 30;
    const r = retryPlan(10, 64, 8, q).retry;
    try testing.expectEqual(@as(u32, @min(38, 64 - 10)), r);
    const p = plan(t, .{ .window = 64, .reserve = r, .system_rows = 10 });
    var buf: [8]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{11}, keptCosts(t, p, &buf));
    try testing.expect(p.estimate + q <= 64);
    try testing.expectEqual(@as(u32, 51), p.estimate + q);
}

test "keepCount matches the longest fitting suffix on random weights" {
    var prng = std.Random.DefaultPrng.init(0x7a17_c0de);
    const r = prng.random();
    for (0..500) |_| {
        // Draw the turns and reverse them.
        var oldest_first: [12]u32 = undefined;
        const len = r.uintAtMost(usize, oldest_first.len);
        for (oldest_first[0..len]) |*w| {
            w.* = r.uintAtMost(u32, 40);
        }

        const cap = r.uintAtMost(u64, 120);
        const turns = oldest_first[0..len];
        var newest_first: [12]u32 = undefined;
        for (turns, 0..) |w, j| {
            newest_first[len - 1 - j] = w;
        }

        // The oracle: the first start index whose suffix sum fits the cap.
        var first: usize = 0;
        while (first < len) : (first += 1) {
            var sum: u64 = 0;
            for (turns[first..]) |w| {
                sum += w;
            }

            if (sum <= cap) break;
        }

        // The count matches the oracle, and one more turn does not fit.
        const kept = keepCount(cap, newest_first[0..len]);
        try testing.expectEqual(len - first, kept);
        var sum: u64 = 0;
        for (turns[len - kept ..]) |w| {
            sum += w;
        }

        try testing.expect(sum <= cap);
        if (kept < len) {
            try testing.expect(sum + turns[len - kept - 1] > cap);
        }
    }
}

test "a refused frame teaches a cost that drops at least one kept pair" {
    // The real sum of a refused frame is past the room. The learned costs sum past it too, so the
    // restart keeps fewer pairs. That bounds the restarts of one cycle.
    var prng = std.Random.DefaultPrng.init(0x0ef1_7000);
    const rand = prng.random();
    for (0..300) |_| {
        const len = rand.intRangeAtMost(u32, 1, 10);
        const t = try newTranscript();
        defer testing.allocator.destroy(t);

        for (0..len) |_| {
            _ = record(t, "q", "", rand.uintAtMost(u32, 30));
        }

        const budget: Budget = .{
            .window = 128,
            .reserve = rand.uintAtMost(u32, 40),
            .system_rows = rand.uintAtMost(u32, 30),
        };
        const before = plan(t, budget);
        if (before.kept == 0) continue;

        // The server counts the kept pairs as one frame and finds them too large. The real sum is
        // past the room after the system rows.
        const room = budget.window - budget.system_rows;
        const real = room + 1 + rand.uintAtMost(u32, 20);
        t.dropOldest(before.dropped);
        learnRefusal(t, 0, t.count(), real);
        const after = plan(t, budget);
        try testing.expect(after.kept < before.kept);
    }
}

test "frameEnd splits the kept pairs by count and by bytes" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    for (0..limits.max_append_pairs + 5) |_| {
        _ = record(t, "q", "r", 1);
    }

    try testing.expectEqual(@as(u32, limits.max_append_pairs), frameEnd(t, 0, t.count()));
    try testing.expectEqual(t.count(), frameEnd(t, limits.max_append_pairs, t.count()));
    try testing.expectEqual(@as(u32, 3), frameEnd(t, 1, 3));

    // Two turns of about 40 KiB each take one frame each.
    const big = try newTranscript();
    defer testing.allocator.destroy(big);

    const reply_text = "x" ** (40 * 1024);
    _ = record(big, "a", reply_text, 1);
    try testing.expectEqual(
        Recorded{ .evicted = 1, .barrier = false },
        record(big, "b", reply_text, 1),
    );
    try testing.expectEqual(@as(u32, 1), big.count());
    try testing.expectEqual(@as(u32, 1), frameEnd(big, 0, 1));
    const pairs = [_]protocol.Pair{big.pair(0)};
    try testing.expect(protocol.appendPairsBytes(&pairs) <= pair_bytes_max);
}

test "refit: a refusal past the estimates drops a pair, and one at or under them is stuck" {
    const budget: Budget = .{ .window = 64, .reserve = 8, .system_rows = 10 };
    const t = try weighed(&.{ 10, 10, 10 });
    defer testing.allocator.destroy(t);

    const p = plan(t, budget);
    try testing.expectEqual(@as(u32, 3), p.kept);

    // The server counts 30 tokens, the sum of the estimates. Nothing new is learned, so a restart
    // repeats the refusal.
    try testing.expectEqual(Refit.stuck, refit(t, budget, 0, 3, 30, p.kept));
    try testing.expectEqual(Refit.stuck, refit(t, budget, 0, 3, 7, p.kept));
    try testing.expectEqual(@as(u32, 3), t.count());

    // A count past the room drops the oldest pair.
    const next = refit(t, budget, 0, 3, 55, p.kept);
    try testing.expectEqual(@as(u32, 2), next.plan.kept);
    try testing.expectEqual(@as(u32, 2), t.count());
}

test "liveCost counts the pending token of a budget or cancel stop" {
    try testing.expectEqual(@as(u32, 9), liveCost(9, .eos, 4));
    try testing.expectEqual(@as(u32, 9), liveCost(9, .context_full, 4));
    try testing.expectEqual(@as(u32, 10), liveCost(9, .max_tokens, 4));
    try testing.expectEqual(@as(u32, 10), liveCost(9, .cancelled, 1));
    try testing.expectEqual(@as(u32, 9), liveCost(9, .cancelled, 0));
}

test "learnAppended stores the exact cost of each pair" {
    const t = try weighed(&.{ 30, 30, 30 });
    defer testing.allocator.destroy(t);

    try testing.expectEqual(@as(u32, 61), learnAppended(t, 1, 40, &.{ 52, 61 }));
    try testing.expectEqual(@as(u32, 30), t.cost(0));
    try testing.expectEqual(@as(u32, 12), t.cost(1));
    try testing.expectEqual(@as(u32, 9), t.cost(2));

    // A refusal under the estimates leaves them, and one over adds the excess to the oldest pair of
    // the frame.
    learnRefusal(t, 1, 3, 15);
    try testing.expectEqual(@as(u32, 12), t.cost(1));
    learnRefusal(t, 1, 3, 25);
    try testing.expectEqual(@as(u32, 16), t.cost(1));
    try testing.expectEqual(@as(u32, 9), t.cost(2));
}

/// Reply text of any length up to the byte bound.
const filler: [pair_bytes_max]u8 = @splat('x');

/// The reply length that gives a turn with a one-byte prompt a third of the byte bound. Three such
/// turns fit, and a fourth does not.
const third_reply: u32 = pair_bytes_max / 3 - pair_overhead_bytes - 1;

test "the record evicts the oldest turns and counts them" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    const reply_text = filler[0..third_reply];
    try testing.expectEqual(
        Recorded{ .evicted = 0, .barrier = false },
        record(t, "a", reply_text, 1),
    );
    try testing.expectEqual(
        Recorded{ .evicted = 0, .barrier = false },
        record(t, "b", reply_text, 2),
    );
    try testing.expectEqual(
        Recorded{ .evicted = 0, .barrier = false },
        record(t, "c", reply_text, 3),
    );
    try testing.expectEqual(
        Recorded{ .evicted = 1, .barrier = false },
        record(t, "d", reply_text, 4),
    );
    try testing.expectEqual(@as(u32, 3), t.count());
    try testing.expectEqualStrings("b", t.prompt(0));
    try testing.expectEqual(@as(u32, 2), t.cost(0));

    // A reply that needs room pushes a turn out while it streams.
    t.begin("e");
    try testing.expectEqual(@as(u32, 3), t.count());
    t.appendReply(filler[0..1]);
    try testing.expectEqual(@as(u32, 2), t.count());
    t.appendReply(filler[0 .. third_reply - 1]);
    try testing.expectEqual(@as(u32, 2), t.count());
    try testing.expectEqual(Recorded{ .evicted = 1, .barrier = false }, t.finish(5));
    try testing.expectEqualStrings("c", t.prompt(0));
    try testing.expectEqual(@as(usize, third_reply), t.reply(2).len);
}

test "a turn past the byte budget is a barrier that drops every held turn" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    _ = record(t, "a", "0123456789", 1);
    _ = record(t, "b", "0123456789", 2);

    // The reply outgrows the budget while it streams.
    t.begin("c");
    t.appendReply(filler[0 .. pair_bytes_max / 2]);
    t.appendReply(filler[0 .. pair_bytes_max / 2]);
    try testing.expectEqual(Recorded{ .evicted = 2, .barrier = true }, t.finish(3));
    try testing.expectEqual(@as(u32, 0), t.count());
}

test "a reply that the caller cut is a barrier" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    _ = record(t, "a", "0123456789", 1);
    _ = record(t, "b", "0123456789", 2);
    t.begin("c");
    t.appendReply("part of a longer reply");
    t.markCut();
    try testing.expectEqual(Recorded{ .evicted = 2, .barrier = true }, t.finish(3));
    try testing.expectEqual(@as(u32, 0), t.count());
}

test "reserveFor clamps the longest turn between an eighth and half the window" {
    try testing.expectEqual(@as(u32, 256), reserveFor(2048, 0));
    try testing.expectEqual(@as(u32, 436), reserveFor(2048, 436));
    try testing.expectEqual(@as(u32, 1024), reserveFor(2048, 5000));
    try testing.expectEqual(@as(u32, 0), reserveFor(7, 0));
    try testing.expectEqual(@as(u32, 3), reserveFor(7, 9));
    try testing.expectEqual(@as(u32, 0), reserveFor(1, 9));
}

test "longest_cost keeps the largest turn until clear" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    _ = record(t, "a", "0123456789", 12);
    _ = record(t, "b", "0123456789", 30);
    _ = record(t, "c", "0123456789", 7);
    try testing.expectEqual(@as(u32, 30), t.longest_cost);

    // A barrier turn still measures the size of a turn.
    t.begin("d");
    t.appendReply(&filler);
    try testing.expectEqual(Recorded{ .evicted = 3, .barrier = true }, t.finish(55));
    try testing.expectEqual(@as(u32, 55), t.longest_cost);

    _ = record(t, "e", "0123456789", 9);
    t.dropOldest(1);
    t.begin("f");
    _ = t.abandon();
    try testing.expectEqual(@as(u32, 55), t.longest_cost);
    t.clear();
    try testing.expectEqual(@as(u32, 0), t.longest_cost);
}

test "begin then abandon keeps every held turn" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    _ = record(t, "a", filler[0..third_reply], 1);
    _ = record(t, "b", filler[0..third_reply], 2);
    _ = record(t, "c", filler[0..third_reply], 3);
    t.begin("p" ** 200);
    try testing.expectEqual(@as(u32, 0), t.abandon());
    try testing.expectEqual(@as(u32, 3), t.count());
    try testing.expectEqualStrings("c", t.prompt(2));

    // An abandoned reply reports the turns it already pushed out.
    t.begin("d");
    t.appendReply("z" ** 10);
    try testing.expectEqual(@as(u32, 1), t.abandon());
    try testing.expectEqual(@as(u32, 2), t.count());
}

test "finish drops only the leading newlines of the reply" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    _ = record(t, "a", "\n\n hello world \r\n", 7);
    _ = record(t, "b", "\n", 8);
    _ = record(t, "c", "ok", 9);
    try testing.expectEqualStrings(" hello world \r\n", t.reply(0));
    try testing.expectEqualStrings("", t.reply(1));
    try testing.expectEqual(
        protocol.Pair{ .user = t.prompt(2), .assistant = t.reply(2) },
        t.pair(2),
    );
    t.dropOldest(2);
    try testing.expectEqual(@as(u32, 1), t.count());
    try testing.expectEqualStrings("c", t.prompt(0));
    t.clear();
    try testing.expectEqual(@as(u32, 0), t.count());
}

test "the record holds turns_max turns of the smallest size" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    var evicted: u32 = 0;
    for (0..turns_max + 5) |_| {
        evicted += record(t, "q", "", 1).evicted;
    }

    const fit: u32 = pair_bytes_max / min_pair_bytes;
    try testing.expectEqual(fit, t.count());
    try testing.expectEqual(turns_max + 5 - fit, evicted);
}

test "parseTooLong reads back both refusal formats" {
    const values = [_]u32{ 0, 1, 999_999, std.math.maxInt(u32) };
    var buf: [128]u8 = undefined;
    for (values) |a| {
        for (values) |b| {
            const turn = try std.fmt.bufPrint(&buf, protocol.turn_too_long_fmt, .{ a, b });
            try testing.expectEqual(
                TooLong{ .kind = .turn, .tokens = a, .room = b },
                parseTooLong(turn).?,
            );
            const sys = try std.fmt.bufPrint(&buf, protocol.system_too_long_fmt, .{ a, b });
            try testing.expectEqual(
                TooLong{ .kind = .system, .tokens = a, .room = b },
                parseTooLong(sys).?,
            );
        }
    }
}

test "parseTooLong rejects foreign, truncated, and overflowing details" {
    const bad = [_][]const u8{
        "",
        "turn does not fit the window",
        "system prompt does not fit the window",
        "turn of 12 tokens, 3 positions",
        "turn of 12 tokens, 3",
        "turn of  tokens, 3 positions left",
        "turn of 12 tokens,  positions left",
        "turn of 4294967296 tokens, 0 positions left",
        "turn of 12 tokens, 3 positions left!",
        "turn of -1 tokens, 3 positions left",
        "system prompt of 12 tokens, window of 64",
    };
    for (bad) |detail| {
        try testing.expectEqual(@as(?TooLong, null), parseTooLong(detail));
    }
}

test "retryPlan refuses only when the system rows alone leave no room" {
    try testing.expectEqual(Retry{ .retry = 44 }, retryPlan(20, 64, 8, 36));
    try testing.expectEqual(Retry{ .retry = 44 }, retryPlan(20, 64, 8, 44));
    try testing.expectEqual(Retry.refuse, retryPlan(20, 64, 8, 45));
    try testing.expectEqual(Retry.refuse, retryPlan(64, 64, 0, 1));
    try testing.expectEqual(Retry{ .retry = 0 }, retryPlan(64, 64, 0, 0));

    // The prompt size comes from the network, so `q + reserve` saturates.
    const max = std.math.maxInt(u32);
    try testing.expectEqual(Retry.refuse, retryPlan(20, 64, 63, max));
    try testing.expectEqual(Retry{ .retry = max }, retryPlan(0, max, 10, max - 1));
}

test "plan keeps the newest turns that fit beside the system rows and reserve" {
    const t = try newTranscript();
    defer testing.allocator.destroy(t);

    try testing.expectEqual(
        Plan{ .kept = 0, .dropped = 0, .estimate = 20 },
        plan(t, .{ .window = 64, .reserve = 8, .system_rows = 20 }),
    );
    _ = record(t, "a", "x", 18);
    _ = record(t, "b", "x", 10);
    _ = record(t, "c", "x", 10);

    // The cap is 64 - 8 - 20 = 36, and the newest costs are 10 and 10.
    const p = plan(t, .{ .window = 64, .reserve = 8, .system_rows = 20 });
    try testing.expectEqual(Plan{ .kept = 2, .dropped = 1, .estimate = 40 }, p);
    try testing.expect(p.estimate + 8 <= 64);

    const r = retryPlan(20, 64, 8, 20).retry;
    try testing.expectEqual(@as(u32, 28), r);
    const q = plan(t, .{ .window = 64, .reserve = r, .system_rows = 20 });
    try testing.expectEqual(Plan{ .kept = 1, .dropped = 2, .estimate = 30 }, q);
    try testing.expect(q.estimate + 20 <= 64);
}

test "retryPlan leaves room for the prompt on random budgets" {
    var prng = std.Random.DefaultPrng.init(0x0e7a_11ed);
    const rand = prng.random();
    for (0..500) |_| {
        var weights: [12]u32 = undefined;
        const len = rand.uintAtMost(usize, weights.len);
        for (weights[0..len]) |*w| {
            w.* = rand.uintAtMost(u32, 40);
        }

        const window = rand.intRangeAtMost(u32, 1, 200);
        const system_rows = rand.uintAtMost(u32, 100);
        const reserve = rand.uintLessThan(u32, window);
        const q = rand.uintAtMost(u32, 120);
        switch (retryPlan(system_rows, window, reserve, q)) {
            .refuse => try testing.expect(system_rows + q > window),
            .retry => |r| {
                const p = planOver(
                    weights[0..len],
                    .{ .window = window, .reserve = r, .system_rows = system_rows },
                );
                try testing.expectEqual(len, p.kept + p.dropped);
                try testing.expect(p.estimate + q <= window);
            },
        }
    }
}

fn lastDone(request_id: u32, finish: protocol.FinishReason, gen_tokens: u32) protocol.LastDone {
    return .{
        .request_id = request_id,
        .finish = finish,
        .gen_tokens = gen_tokens,
        .reasoning_tokens = 0,
    };
}

test "reconcile: every generated token reached the client" {
    const r = reconcile(.{ .request_id = 9, .received = 12 }, 3, lastDone(9, .eos, 12));
    try testing.expectEqual(Reconciled.complete, r);
}

test "reconcile: a turn that ended exactly at the drop is complete" {
    // The last token frame arrived and the `done` did not.
    try testing.expectEqual(
        Reconciled.complete,
        reconcile(.{ .request_id = 4, .received = 7 }, 2, lastDone(4, .max_tokens, 7)),
    );
}

test "reconcile: a drop with no token in the gap cuts by zero, so the turn is complete" {
    // The drop cancelled the turn right after the last frame that the client got.
    try testing.expectEqual(
        Reconciled.complete,
        reconcile(.{ .request_id = 5, .received = 30 }, 1, lastDone(5, .cancelled, 30)),
    );

    // A drop in the turn prefill: the server ran the turn and generated nothing.
    try testing.expectEqual(
        Reconciled.complete,
        reconcile(.{ .request_id = 5, .received = 0 }, 1, lastDone(5, .cancelled, 0)),
    );
}

test "reconcile: the server generated k tokens that the client never got" {
    try testing.expectEqual(
        Reconciled{ .cut = 3 },
        reconcile(.{ .request_id = 9, .received = 17 }, 4, lastDone(9, .cancelled, 20)),
    );
    try testing.expectEqual(
        Reconciled{ .cut = 20 },
        reconcile(.{ .request_id = 9, .received = 0 }, 4, lastDone(9, .cancelled, 20)),
    );
}

test "reconcile: a turn the server never ran goes out again" {
    // has_last is 0: no turn of the session finished.
    try testing.expectEqual(
        Reconciled.never_ran,
        reconcile(.{ .request_id = 3, .received = 0 }, 0, null),
    );

    // Kept pairs count in turns_done and leave has_last at 0.
    try testing.expectEqual(
        Reconciled.never_ran,
        reconcile(.{ .request_id = 3, .received = 0 }, 5, null),
    );

    // The last finished request is an older turn.
    try testing.expectEqual(
        Reconciled.never_ran,
        reconcile(.{ .request_id = 8, .received = 0 }, 2, lastDone(6, .eos, 40)),
    );
}

test "reconcile: a summary that contradicts the client frames is inconsistent" {
    const cases = [_]struct { turn: InFlight, turns: u32, last: ?protocol.LastDone }{
        // Tokens arrived, and the server reports no finished turn.
        .{ .turn = .{ .request_id = 3, .received = 2 }, .turns = 0, .last = null },
        // Tokens arrived, and the last request is an older one.
        .{ .turn = .{ .request_id = 8, .received = 2 }, .turns = 2, .last = lastDone(6, .eos, 40) },
        // A later request id than the client ever sent.
        .{ .turn = .{ .request_id = 8, .received = 0 }, .turns = 2, .last = lastDone(9, .eos, 1) },
        // More frames than the server generated.
        .{ .turn = .{ .request_id = 8, .received = 5 }, .turns = 2, .last = lastDone(8, .eos, 4) },
        // A last turn with zero finished turns.
        .{ .turn = .{ .request_id = 8, .received = 4 }, .turns = 0, .last = lastDone(8, .eos, 4) },
    };
    for (cases) |case| {
        try testing.expectEqual(
            Reconciled.inconsistent,
            reconcile(case.turn, case.turns, case.last),
        );
    }
}

test "reconcile: on the same request the cut plus the received frames equals gen_tokens" {
    for (0..40) |g| {
        for (0..g + 1) |r| {
            const gen: u32 = @intCast(g);
            const got: u32 = @intCast(r);
            switch (reconcile(
                .{ .request_id = 1, .received = got },
                1,
                lastDone(1, .cancelled, gen),
            )) {
                .complete => try testing.expectEqual(gen, got),
                .cut => |k| {
                    try testing.expect(k > 0);
                    try testing.expectEqual(gen, got + k);
                },
                .never_ran, .inconsistent => return error.TestUnexpectedResult,
            }
        }
    }
}

/// The `received` count of the first `n` frames, the count the client keeps.
fn receivedOf(frames: []const protocol.Channel, n: usize) u32 {
    var got: u32 = 0;
    for (frames[0..n]) |ch| {
        got += @intFromBool(countsAsGenerated(ch));
    }

    return got;
}

test "reconcile: a think-open turn counts no echo, so the verdicts match the server count" {
    // The frames of a think-open turn: the echo, two think tokens, the close, the answer, and the
    // end token. The server counts five.
    const frames = [_]protocol.Channel{
        .open_tag,
        .reasoning,
        .reasoning,
        .close_tag,
        .content,
        .content,
    };
    const gen: u32 = frames.len - 1;
    const turn = InFlight{ .request_id = 4, .received = receivedOf(&frames, frames.len) };
    try testing.expectEqual(Reconciled.complete, reconcile(turn, 1, lastDone(4, .eos, gen)));

    // The drop lost the end token.
    const cut = InFlight{ .request_id = 4, .received = receivedOf(&frames, frames.len - 1) };
    try testing.expectEqual(
        Reconciled{ .cut = 1 },
        reconcile(cut, 1, lastDone(4, .cancelled, gen)),
    );

    // The drop came after the echo, during the prefill.
    const echoed = InFlight{ .request_id = 4, .received = receivedOf(&frames, 1) };
    try testing.expectEqual(Reconciled.complete, reconcile(echoed, 1, lastDone(4, .cancelled, 0)));

    // The drop came after the echo and two think tokens of five generated.
    const mid = InFlight{ .request_id = 4, .received = receivedOf(&frames, 3) };
    try testing.expectEqual(Reconciled{ .cut = 3 }, reconcile(mid, 1, lastDone(4, .cancelled, 5)));

    // A sampled id equal to the open id inside the block is reasoning, so it counts.
    try testing.expect(countsAsGenerated(.reasoning));
    try testing.expect(!countsAsGenerated(.open_tag));
}
