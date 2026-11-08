//! What the client prints: the turn summary, the compaction notes, and the shades of a reply. The
//! values here format themselves, so a test reads each line with no server.
//!
//! Comptime. `note` and `dim` take the format at comptime, so the compiler checks its arguments.
//!
//! Streams. Model text and command replies go to stdout. Prompts, diagnostics, and summaries go
//! to stderr.
//!
//! Shades. The server marks the think tags on the channel of each token event, so the client needs
//! no tag text. A tag prints nothing. Think text never enters the transcript.

const std = @import("std");
const Io = std.Io;

const style = @import("../style.zig");
const protocol = @import("../../serve/protocol.zig");
const client_mod = @import("../../serve/client.zig");
const transcript_mod = @import("client_transcript.zig");

const Client = client_mod.Client;
const Transcript = transcript_mod.Transcript;

const client_history = @import("client_history.zig");

// -- constants --------------------------------------------------------------

const preview_bytes = 60;

/// The UTF-8 tail that waits for the next token. A codepoint holds at most 4 bytes.
pub const codepoint_tail_max = 3;

const window_fmt = "window: {d}/{d} positions used";
pub const session_open_fmt = "session {d} open";

pub const no_session_note = "no session is open, quitting";
const context_full_note =
    "The window is full. The client compacts the conversation into a new session.";
const evicted_fmt = "The transcript drops its {d} oldest turns to stay inside {d} bytes of text.";
const barrier_fmt = "The turn does not fit {d} bytes of transcript text. " ++
    "The client drops it and {d} older turns.";
const abandoned_fmt = "The failed turn already dropped the {d} oldest turns from the transcript.";
pub const retry_fmt = "The turn of {d} tokens does not fit the {d} positions left. " ++
    "The client compacts and sends it once more.";
const refuse_fmt = "The prompt of {d} tokens does not fit beside the {d} system rows in the " ++
    "window of {d} positions.";
pub const refit_fmt = "The kept pairs of {d} tokens do not fit the {d} positions left. " ++
    "The client learns the cost and compacts again.";
pub const reopen_note = "The append failed. The client opens the session once more.";
pub const append_lost_note = "The append failed again. " ++
    "The client clears the transcript and keeps the system prompt only.";
pub const refit_stuck_note = "The refusal does not lower the kept pairs. " ++
    "The client clears the transcript and keeps the system prompt only.";
pub const compact_empty_note = "The transcript holds no turns, " ++
    "so a compaction has nothing to keep.";

// -- types ------------------------------------------------------------------

/// The line that ends each turn. Only `max_seq` comes from `hello`. The rest comes from `done`.
const Summary = struct {
    finish_reason: protocol.FinishReason,
    gen_tokens: u32,
    reasoning_tokens: u32,
    kv_pos: u32,
    max_seq: u32,
    prefill_ms: f32,
    decode_ms: f32,

    /// The summary of a finished turn. `end` is a `done` frame or a `Client.Turn`.
    fn of(end: anytype, max_seq: u32) Summary {
        return .{
            .finish_reason = end.finish_reason,
            .gen_tokens = end.gen_tokens,
            .reasoning_tokens = end.reasoning_tokens,
            .kv_pos = end.kv_pos,
            .max_seq = max_seq,
            .prefill_ms = end.prefill_ms,
            .decode_ms = end.decode_ms,
        };
    }

    pub fn format(s: Summary, w: *Io.Writer) Io.Writer.Error!void {
        try w.print(
            "finish_reason={t} gen_tokens={d} reasoning_tokens={d} kv_pos={d}/{d} " ++
                "prefill_ms={d:.1} decode_ms={d:.1}",
            .{
                s.finish_reason,
                s.gen_tokens,
                s.reasoning_tokens,
                s.kv_pos,
                s.max_seq,
                s.prefill_ms,
                s.decode_ms,
            },
        );
    }
};

pub const Shade = enum { reply, think, tag };

/// The think block of one turn, read from the channel of each token event.
pub const ThinkView = struct {
    /// The shade after the last event, for the bytes a turn end flushes.
    last: Shade = .reply,

    /// The shade of one event. A tag prints nothing.
    pub fn step(v: *ThinkView, channel: protocol.Channel) Shade {
        switch (channel) {
            .content => v.last = .reply,
            .reasoning => v.last = .think,
            .open_tag => v.last = .think,
            .close_tag => v.last = .reply,
        }

        return switch (channel) {
            .content => .reply,
            .reasoning => .think,
            .open_tag, .close_tag => .tag,
        };
    }
};

/// The shades of a blocking reply, the token hook of `collectTurnWith`. The hook counts the text
/// bytes as the copy does, so each offset matches the collected text.
pub const ReplyShades = struct {
    eos_id: u32,
    /// A set bit marks a byte of a think block, tags included. One bit per text buffer byte.
    dim: std.DynamicBitSetUnmanaged,
    /// A set bit marks a byte of a think tag, which prints nothing.
    tag: std.DynamicBitSetUnmanaged,
    view: ThinkView = .{},
    /// The text bytes of the reply so far, the `text_total` of the turn.
    total: usize = 0,

    pub fn init(arena: std.mem.Allocator, text_cap: usize, eos_id: u32) !ReplyShades {
        return .{
            .eos_id = eos_id,
            .dim = try .initEmpty(arena, text_cap),
            .tag = try .initEmpty(arena, text_cap),
        };
    }

    pub fn reset(s: *ReplyShades) void {
        s.dim.unsetAll();
        s.tag.unsetAll();
        s.* = .{ .eos_id = s.eos_id, .dim = s.dim, .tag = s.tag };
    }

    /// The copy skips the text of the end-of-sequence token, so the count skips it too. A byte
    /// past the text buffer has no shade to keep.
    pub fn onToken(
        s: *ReplyShades,
        token_id: u32,
        channel: protocol.Channel,
        text: []const u8,
    ) void {
        if (token_id == s.eos_id) return;

        const shade = s.view.step(channel);
        const end = @min(s.total + text.len, s.dim.bit_length);
        if (s.total < end) {
            s.dim.setRangeValue(.{ .start = s.total, .end = end }, shade != .reply);
            s.tag.setRangeValue(.{ .start = s.total, .end = end }, shade == .tag);
        }

        s.total += text.len;
    }

    /// Print each run of one shade in `text`, the kept prefix of the reply, between its color and
    /// a reset.
    pub fn write(
        s: *const ReplyShades,
        w: *Io.Writer,
        p: style.Palette,
        text: []const u8,
    ) Io.Writer.Error!void {
        std.debug.assert(text.len <= s.total);
        std.debug.assert(text.len <= s.dim.bit_length);

        var start: usize = 0;

        // Each pass prints one run. The loop ends at the kept length.
        while (start < text.len) {
            const end = s.runEnd(start, text.len);
            const dim = s.dim.isSet(start);
            if (!s.tag.isSet(start)) {
                try w.print(
                    "{s}{s}{s}",
                    .{ shadeColor(p, if (dim) .think else .reply), text[start..end], p.reset },
                );
            }

            start = end;
        }
    }

    /// Record each reply run of `text` in the transcript.
    pub fn record(s: *const ReplyShades, t: *Transcript, text: []const u8) void {
        std.debug.assert(text.len <= s.total);
        std.debug.assert(text.len <= s.dim.bit_length);

        var start: usize = 0;

        // Each pass takes one run. The loop ends at the kept length.
        while (start < text.len) {
            const end = s.runEnd(start, text.len);
            if (!s.dim.isSet(start)) {
                t.appendReply(text[start..end]);
            }

            start = end;
        }
    }

    /// The end of the run of one shade that starts at `start`.
    pub fn runEnd(s: *const ReplyShades, start: usize, len: usize) usize {
        std.debug.assert(start < len);

        const dim = s.dim.isSet(start);
        const tag = s.tag.isSet(start);
        var end = start + 1;
        while (end < len and s.dim.isSet(end) == dim and s.tag.isSet(end) == tag) {
            end += 1;
        }

        std.debug.assert(end <= len);
        return end;
    }
};

/// A blocking reply longer than one of its buffers.
const Cut = struct {
    unit: []const u8,
    total: usize,
    kept: usize,

    pub fn format(c: Cut, w: *Io.Writer) Io.Writer.Error!void {
        std.debug.assert(c.kept < c.total);

        try w.print(
            "The reply has {d} {s}. The client keeps the first {d}.",
            .{ c.total, c.unit, c.kept },
        );
    }
};

const Cuts = struct { ids: ?Cut, text: ?Cut };

/// The estimate comes from the plan, and `kv_pos` is the cost that the server measured.
pub const Compacted = struct {
    kept: u32,
    total: u32,
    reserve: u32,
    estimate: u32,
    kv_pos: u32,
    window: u32,

    pub fn format(c: Compacted, w: *Io.Writer) Io.Writer.Error!void {
        std.debug.assert(c.kept <= c.total);

        try w.print(
            "compacted: kept {d} of {d} turns, reserve {d}, estimate {d}, kv_pos {d}/{d}",
            .{
                c.kept, c.total, c.reserve, c.estimate, c.kv_pos, c.window,
            },
        );
    }

    pub fn of(cycle: *const client_history.Cycle, kv_pos: u32, window: u32) Compacted {
        return .{
            .kept = cycle.plan.kept,
            .total = cycle.total,
            .reserve = cycle.reserve,
            .estimate = cycle.plan.estimate,
            .kv_pos = kv_pos,
            .window = window,
        };
    }
};

/// The head line of `/history`.
const HistoryHead = struct {
    turns: u32,
    positions: u64,
    reserve: u32,
    window: u32,
    keeps: u32,

    pub fn format(h: HistoryHead, w: *Io.Writer) Io.Writer.Error!void {
        std.debug.assert(h.keeps <= h.turns);

        try w.print(
            "transcript: {d} turns, {d} positions, reserve {d}, window {d}. " ++
                "A compaction keeps {d}.",
            .{
                h.turns, h.positions, h.reserve, h.window, h.keeps,
            },
        );
    }
};

/// One turn of `/history`. A star marks a turn that the next compaction keeps.
const HistoryLine = struct {
    kept: bool,
    number: u32,
    positions: u32,
    prompt: []const u8,

    pub fn format(l: HistoryLine, w: *Io.Writer) Io.Writer.Error!void {
        const n = previewLen(l.prompt);
        const tail: []const u8 = if (n < l.prompt.len) "..." else "";
        try w.print("{s} {d}: {d} positions, {s}{s}", .{
            if (l.kept) "*" else " ", l.number, l.positions, l.prompt[0..n], tail,
        });
    }
};

/// Both output streams with their palettes.
pub const Out = struct {
    o: *Io.Writer,
    e: *Io.Writer,
    po: style.Palette,
    pe: style.Palette,

    pub fn note(out: Out, comptime fmt: []const u8, a: anytype) !void {
        try out.e.print(fmt ++ "\n", a);
        try out.e.flush();
    }

    pub fn dim(out: Out, comptime fmt: []const u8, a: anytype) !void {
        try out.e.writeAll(out.pe.dim);
        try out.e.print(fmt, a);
        try out.e.print("{s}\n", .{out.pe.reset});
        try out.e.flush();
    }

    pub fn marker(out: Out) !void {
        try out.e.print("{s}> {s}", .{ out.pe.bold, out.pe.reset });
        try out.e.flush();
    }

    pub fn serverError(out: Out, le: ?Client.LastError) !void {
        const e = le orelse return out.note("server error without detail", .{});
        try out.note("server error: {t}: {s}", .{ e.code, e.detail() });
    }

    pub fn turnRefused(
        out: Out,
        le: ?Client.LastError,
        session: Client.Opened,
        max_seq: u32,
    ) !void {
        try out.serverError(le);
        const e = le orelse return;
        if (e.code == .turn_too_long) {
            try out.note(window_fmt, .{ session.kv_pos, max_seq });
        }
    }

    pub fn recorded(out: Out, rec: transcript_mod.Recorded) !void {
        if (rec.barrier) {
            return out.note(barrier_fmt, .{ transcript_mod.pair_bytes_max, rec.evicted });
        }

        if (rec.evicted > 0) {
            try out.note(evicted_fmt, .{ rec.evicted, transcript_mod.pair_bytes_max });
        }
    }

    pub fn abandoned(out: Out, evicted: u32) !void {
        if (evicted > 0) {
            try out.note(abandoned_fmt, .{evicted});
        }
    }

    /// The refusal of a prompt that no compaction makes room for.
    pub fn refused(
        out: Out,
        le: ?Client.LastError,
        session: Client.Opened,
        h: *const client_history.History,
        q: u32,
    ) !void {
        try out.turnRefused(le, session, h.window);
        try out.note(refuse_fmt, .{ q, h.system_rows, h.window });
    }

    /// Step `cycle` after a failed append frame of turns `from..to`. A refused frame teaches its
    /// cost, so the plan keeps fewer pairs. Any other failure opens the session once more, and a
    /// second one clears the transcript. The caller reopens in every case.
    pub fn frameFailed(
        out: Out,
        h: *client_history.History,
        cycle: *client_history.Cycle,
        le: ?Client.LastError,
        from: u32,
        to: u32,
    ) !void {
        if (client_history.tooLong(le, .turn)) |tl| {
            try out.note(refit_fmt, .{ tl.tokens, tl.room });
            cycle.refits += 1;
            const limit = h.budget(cycle.reserve);
            switch (transcript_mod.refit(h.t, limit, from, to, tl.tokens, cycle.plan.kept)) {
                .plan => |p| cycle.plan = p,
                .stuck => {
                    try out.dim(refit_stuck_note, .{});
                    h.keepNothing(cycle);
                },
            }

            return;
        }

        try out.serverError(le);
        if (!cycle.reopened) {
            cycle.reopened = true;
            return out.note(reopen_note, .{});
        }

        try out.note(append_lost_note, .{});
        h.keepNothing(cycle);
    }

    pub fn history(out: Out, h: *const client_history.History) !void {
        const t = h.t;
        const n = t.count();
        const keeps = h.plan(h.effectiveReserve()).kept;

        var positions: u64 = 0;
        for (0..n) |j| {
            positions += t.cost(@intCast(j));
        }

        try out.o.print("{f}\n", .{HistoryHead{
            .turns = n,
            .positions = positions,
            .reserve = h.effectiveReserve(),
            .window = h.window,
            .keeps = keeps,
        }});

        // Each pass prints one turn. The loop runs `n` times.
        for (0..n) |j| {
            const i: u32 = @intCast(j);
            try out.o.print("{f}\n", .{HistoryLine{
                .kept = i >= n - keeps,
                .number = i + 1,
                .positions = t.cost(i),
                .prompt = t.prompt(i),
            }});
        }

        try out.o.flush();
    }

    pub fn finish(out: Out, end: anytype, max_seq: u32) !void {
        try out.dim("{f}", .{Summary.of(end, max_seq)});
        if (finishNote(end.finish_reason)) |note_text| {
            try out.note("{s}", .{note_text});
        }
    }

    pub fn stats(out: Out, s: Client.Stats) !void {
        try out.o.print("sessions_open={d} sessions_decoding={d} " ++
            "tokens_generated_total={d} uptime_ms={d}\n", .{
            s.sessions_open, s.sessions_decoding, s.tokens_generated_total, s.uptime_ms,
        });
        try out.o.flush();
    }

    pub fn pong(out: Out, ms: f64) !void {
        try out.o.print("pong in {d:.2} ms\n", .{ms});
        try out.o.flush();
    }
};

// -- api --------------------------------------------------------------------

pub fn shadeColor(p: style.Palette, shade: Shade) []const u8 {
    return switch (shade) {
        .reply => p.green,
        .think, .tag => p.dim,
    };
}

/// `collectTurn` fills each buffer as far as it fits and reports the true totals, so a total over
/// the kept count is a cut.
pub fn turnCuts(turn: Client.Turn, ids_cap: usize) Cuts {
    std.debug.assert(turn.text_len <= turn.text_total);

    const ids: ?Cut = if (turn.n_tokens > ids_cap)
        .{ .unit = "token ids", .total = turn.n_tokens, .kept = ids_cap }
    else
        null;

    const text: ?Cut = if (turn.text_total > turn.text_len)
        .{ .unit = "text bytes", .total = turn.text_total, .kept = turn.text_len }
    else
        null;

    return .{ .ids = ids, .text = text };
}

/// The length of the longest prefix of `bytes` that ends on a UTF-8 codepoint boundary. An invalid
/// lead byte counts as complete, so the output never stalls on bad bytes.
pub fn completePrefixLen(bytes: []const u8) usize {
    const back_max = @min(bytes.len, codepoint_tail_max);
    var i: usize = 1;
    const result = while (i <= back_max) : (i += 1) {
        const b = bytes[bytes.len - i];
        if (b & 0b1100_0000 == 0b1000_0000) continue;

        const need = std.unicode.utf8ByteSequenceLength(b) catch break bytes.len;
        break if (need > i) bytes.len - i else bytes.len;
    } else bytes.len;

    std.debug.assert(bytes.len - result <= codepoint_tail_max);
    return result;
}

// -- helpers: text ----------------------------------------------------------

/// The bytes of a prompt that a history line shows. The cut ends on a codepoint boundary.
fn previewLen(prompt_text: []const u8) usize {
    if (prompt_text.len <= preview_bytes) return prompt_text.len;

    const n = completePrefixLen(prompt_text[0..preview_bytes]);
    std.debug.assert(n <= preview_bytes);
    return n;
}

/// The line after the summary, when the finish reason asks the user to act.
fn finishNote(reason: protocol.FinishReason) ?[]const u8 {
    return switch (reason) {
        .context_full => context_full_note,
        .eos, .max_tokens, .cancelled => null,
    };
}

// -- tests ------------------------------------------------------------------

test "completePrefixLen on ASCII and on cut codepoints" {
    try std.testing.expectEqual(@as(usize, 0), completePrefixLen(""));
    try std.testing.expectEqual(@as(usize, 5), completePrefixLen("hello"));

    // "é" is C3 A9, "€" is E2 82 AC, and U+1F600 is F0 9F 98 80.
    try std.testing.expectEqual(@as(usize, 1), completePrefixLen("a\xc3"));
    try std.testing.expectEqual(@as(usize, 1), completePrefixLen("a\xe2\x82"));
    try std.testing.expectEqual(@as(usize, 1), completePrefixLen("a\xf0\x9f\x98"));
    try std.testing.expectEqual(@as(usize, 5), completePrefixLen("a\xf0\x9f\x98\x80"));
    try std.testing.expectEqual(@as(usize, 2), completePrefixLen("a\xff"));
}

test "completePrefixLen ends on a boundary at every split" {
    const s = "a\xc3\xa9\xe2\x82\xacb\xf0\x9f\x98\x80z\xe2\x82\xac";
    try std.testing.expect(std.unicode.utf8ValidateSlice(s));
    for (0..s.len) |k| {
        const result = completePrefixLen(s[0..k]);
        try std.testing.expect(result <= k);
        try std.testing.expect(k - result <= 3);
        try std.testing.expect(std.unicode.utf8ValidateSlice(s[0..result]));
        try std.testing.expect(result == s.len or s[result] & 0b1100_0000 != 0b1000_0000);
    }
}

test "the summary line and its note for each finish reason" {
    // Vary one summary for each finish reason.
    const Case = struct { s: Summary, line: []const u8, note: ?[]const u8 };
    const base: Summary = .{
        .finish_reason = .eos,
        .gen_tokens = 40,
        .reasoning_tokens = 0,
        .kv_pos = 52,
        .max_seq = 512,
        .prefill_ms = 12.5,
        .decode_ms = 800,
    };
    var capped = base;
    capped.finish_reason = .max_tokens;
    capped.reasoning_tokens = 39;
    var full = base;
    full.finish_reason = .context_full;
    full.gen_tokens = 460;
    full.reasoning_tokens = 300;
    full.kv_pos = 512;
    var cancelled = base;
    cancelled.finish_reason = .cancelled;
    // Name the line that each summary prints.
    const times = "prefill_ms=12.5 decode_ms=800.0";
    const line_base = "finish_reason=eos gen_tokens=40 reasoning_tokens=0 kv_pos=52/512 " ++ times;
    const line_capped = "finish_reason=max_tokens gen_tokens=40 reasoning_tokens=39 " ++
        "kv_pos=52/512 " ++ times;
    const line_full = "finish_reason=context_full gen_tokens=460 reasoning_tokens=300 " ++
        "kv_pos=512/512 " ++ times;
    const line_cancelled = "finish_reason=cancelled gen_tokens=40 reasoning_tokens=0 " ++
        "kv_pos=52/512 " ++ times;
    // Pair each summary with its line and its note.
    const cases = [_]Case{
        .{ .s = base, .line = line_base, .note = null },
        .{ .s = capped, .line = line_capped, .note = null },
        .{ .s = full, .line = line_full, .note = context_full_note },
        .{ .s = cancelled, .line = line_cancelled, .note = null },
    };
    comptime std.debug.assert(cases.len == std.enums.values(protocol.FinishReason).len);

    for (cases) |case| {
        try std.testing.expectFmt(case.line, "{f}", .{case.s});
        const got = finishNote(case.s.finish_reason);
        if (case.note) |want| {
            try std.testing.expectEqualStrings(want, got.?);
        } else {
            try std.testing.expect(got == null);
        }
    }
}

test "turnCuts reports each cut with both numbers" {
    const whole: Client.Turn = .{
        .finish_reason = .eos,
        .n_tokens = 8,
        .text_len = 30,
        .text_total = 30,
        .kv_pos = 20,
        .gen_tokens = 8,
        .reasoning_tokens = 0,
        .prefill_ms = 1,
        .decode_ms = 2,
    };
    const none = turnCuts(whole, 8);
    try std.testing.expect(none.ids == null);
    try std.testing.expect(none.text == null);

    var long = whole;
    long.n_tokens = 9;
    long.text_len = 65530;
    long.text_total = 70000;
    const both = turnCuts(long, 8);
    try std.testing.expectFmt(
        "The reply has 9 token ids. The client keeps the first 8.",
        "{f}",
        .{both.ids.?},
    );
    try std.testing.expectFmt(
        "The reply has 70000 text bytes. The client keeps the first 65530.",
        "{f}",
        .{both.text.?},
    );
}

/// Visible marks in place of the escape codes, so a test can read the shades.
fn refusal(tokens: u32, room: u32) Client.LastError {
    var e: Client.LastError = .{
        .code = .turn_too_long,
        .len = 0,
        .truncated = false,
        .buf = undefined,
    };
    const detail = std.fmt.bufPrint(&e.buf, protocol.turn_too_long_fmt, .{ tokens, room }) catch
        unreachable;
    e.len = @intCast(detail.len);
    return e;
}

test "frameFailed: a refusal drops pairs, a repeat clears, and another failure reopens once" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    const out: Out = .{ .o = &w, .e = &w, .po = .{}, .pe = .{} };

    var h: client_history.History = try .init(arena_state.allocator(), 10, 64, 8);
    for (0..3) |_| {
        h.t.begin("q");
        _ = h.t.finish(10);
    }

    var cycle: client_history.Cycle = .{ .plan = h.shrink(8), .total = 3, .reserve = 8 };
    try std.testing.expectEqual(@as(u32, 3), cycle.plan.kept);

    // A count past the room teaches a cost, so the plan drops the oldest pair.
    try out.frameFailed(&h, &cycle, refusal(55, 40), 0, 3);
    try std.testing.expectEqual(@as(u32, 1), cycle.refits);
    try std.testing.expectEqual(@as(u32, 2), cycle.plan.kept);
    try std.testing.expectEqual(@as(u32, 2), h.t.count());

    // A count inside the estimates teaches nothing, so the cycle keeps the system prompt only.
    try out.frameFailed(&h, &cycle, refusal(1, 40), 0, 2);
    try std.testing.expectEqual(@as(u32, 0), cycle.plan.kept);
    try std.testing.expectEqual(@as(u32, 0), h.t.count());

    // Another failure reopens once, and the second one clears.
    h.t.begin("q");
    _ = h.t.finish(10);
    try out.frameFailed(&h, &cycle, null, 0, 1);
    try std.testing.expect(cycle.reopened);
    try std.testing.expectEqual(@as(u32, 1), h.t.count());
    try out.frameFailed(&h, &cycle, null, 0, 1);
    try std.testing.expectEqual(@as(u32, 0), h.t.count());
}

const test_palette: style.Palette = .{ .reset = ")", .dim = "D(", .green = "G(" };

fn expectShades(shades: *const ReplyShades, text: []const u8, want: []const u8) !void {
    var buf: [256]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try shades.write(&w, test_palette, text);
    try std.testing.expectEqualStrings(want, w.buffered());
}

/// One token event of a test turn.
const Piece = struct { text: []const u8, channel: protocol.Channel = .content };

/// Feeds each piece to the hook under its own id, then the end-of-sequence frame, as
/// `collectTurnWith` calls it.
fn feed(shades: *ReplyShades, pieces: []const Piece) void {
    for (pieces, 0..) |p, i| {
        shades.onToken(@intCast(10 + i), p.channel, p.text);
    }

    shades.onToken(shades.eos_id, .content, "<eos>");
}

test "ThinkView: the channels split the events into reply and think shades" {
    const events = [_]protocol.Channel{
        .content,
        .open_tag,
        .reasoning,
        .reasoning,
        .close_tag,
        .content,
        .content,
    };
    const want = [_]Shade{ .reply, .tag, .think, .think, .tag, .reply, .reply };
    var v: ThinkView = .{};
    for (events, want) |e, shade| {
        try std.testing.expectEqual(shade, v.step(e));
    }

    try std.testing.expectEqual(Shade.reply, v.last);
}

test "ReplyShades: the blocking print dims the block between the tags" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var shades: ReplyShades = try .init(arena_state.allocator(), 64, 7);
    feed(&shades, &.{
        .{ .text = "Hi " },
        .{ .text = "<think>", .channel = .open_tag },
        .{ .text = "a", .channel = .reasoning },
        .{ .text = "b", .channel = .reasoning },
        .{ .text = "</think>", .channel = .close_tag },
        .{ .text = "\n\n" },
        .{ .text = "ok" },
    });
    const text = "Hi <think>ab</think>\n\nok";
    try std.testing.expectEqual(text.len, shades.total);
    try expectShades(&shades, text, "G(Hi )D(ab)G(\n\nok)");
}

test "ReplyShades: a block with no close at the finish stays dim to the end" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var shades: ReplyShades = try .init(arena_state.allocator(), 64, 7);
    feed(&shades, &.{
        .{ .text = "<think>", .channel = .open_tag },
        .{ .text = "a", .channel = .reasoning },
        .{ .text = "\n", .channel = .reasoning },
        .{ .text = "b", .channel = .reasoning },
    });
    try expectShades(&shades, "<think>a\nb", "D(a\nb)");
    try std.testing.expectEqual(Shade.think, shades.view.last);
}

test "ReplyShades: a reply with no block prints one run and counts zero" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var shades: ReplyShades = try .init(arena_state.allocator(), 64, 7);

    // Tag text on the content channel is plain text.
    feed(&shades, &.{ .{ .text = "a" }, .{ .text = "<think>" }, .{ .text = "b" } });
    try expectShades(&shades, "a<think>b", "G(a<think>b)");
}

test "ReplyShades: a cut keeps the shades of the kept bytes, and a reset starts a clean turn" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    var shades: ReplyShades = try .init(arena_state.allocator(), 8, 7);

    // The copy keeps "<think>" and stops at "abc", which does not fit.
    feed(&shades, &.{
        .{ .text = "<think>", .channel = .open_tag },
        .{ .text = "abc", .channel = .reasoning },
        .{ .text = "x", .channel = .reasoning },
    });
    try std.testing.expectEqual(@as(usize, 11), shades.total);
    try expectShades(&shades, "<think>", "");

    shades.reset();
    feed(&shades, &.{.{ .text = "ok" }});
    try expectShades(&shades, "ok", "G(ok)");
    try std.testing.expectEqual(@as(usize, 2), shades.total);
}

test "the compacted line and the history head name their numbers" {
    try std.testing.expectFmt(
        "compacted: kept 2 of 5 turns, reserve 8, estimate 51, kv_pos 47/64",
        "{f}",
        .{Compacted{
            .kept = 2,
            .total = 5,
            .reserve = 8,
            .estimate = 51,
            .kv_pos = 47,
            .window = 64,
        }},
    );
    try std.testing.expectFmt(
        "transcript: 5 turns, 130 positions, reserve 8, window 64. A compaction keeps 2.",
        "{f}",
        .{HistoryHead{
            .turns = 5,
            .positions = 130,
            .reserve = 8,
            .window = 64,
            .keeps = 2,
        }},
    );
}

test "a history line cuts its prompt at the preview bound on a codepoint boundary" {
    try std.testing.expectFmt("* 3: 41 positions, hello", "{f}", .{HistoryLine{
        .kept = true,
        .number = 3,
        .positions = 41,
        .prompt = "hello",
    }});
    const exact = "x" ** preview_bytes;
    try std.testing.expectFmt("  1: 9 positions, " ++ exact, "{f}", .{HistoryLine{
        .kept = false,
        .number = 1,
        .positions = 9,
        .prompt = exact,
    }});

    // The euro sign E2 82 AC starts one byte before the bound, so the cut keeps the bytes before
    // it.
    const long = "y" ** (preview_bytes - 1) ++ "\xe2\x82\xac tail";
    try std.testing.expectEqual(@as(usize, preview_bytes - 1), previewLen(long));
    try std.testing.expectFmt(
        "  2: 12 positions, " ++ "y" ** (preview_bytes - 1) ++ "...",
        "{f}",
        .{HistoryLine{
            .kept = false,
            .number = 2,
            .positions = 12,
            .prompt = long,
        }},
    );
    try std.testing.expectEqual(@as(usize, preview_bytes), previewLen("z" ** (preview_bytes + 1)));
}

test "ReplyShades: the transcript keeps the reply runs and drops the think block" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    var shades: ReplyShades = try .init(arena, 64, 7);
    feed(&shades, &.{
        .{ .text = "<think>", .channel = .open_tag },
        .{ .text = "a", .channel = .reasoning },
        .{ .text = "</think>", .channel = .close_tag },
        .{ .text = "\n\nok" },
    });
    const t = try arena.create(Transcript);
    t.init();
    t.begin("q");
    shades.record(t, "<think>a</think>\n\nok");
    _ = t.finish(5);
    try std.testing.expectEqualStrings("ok", t.reply(0));
}
