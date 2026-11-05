//! `zigferencer client` is an interactive chat over the binary wire. It
//! connects to a running `serve`, opens one session, and sends each typed line
//! as one turn. A line that starts with `/` is a command.
//!
//! The streaming mode runs two tasks that feed one queue: one reads stdin
//! lines, and one reads server frames. The main task is the control unit. It
//! takes one input at a time and steps a state machine, so a `/cancel` typed
//! while a turn streams reaches the server at once. `--no-stream` runs one
//! blocking loop with no tasks. A streaming turn refuses a line that arrives
//! during the turn, so piped input uses `--no-stream`.
//!
//! Both modes keep a transcript of the finished turns. When the window fills,
//! or the server refuses a turn, a compaction cycle closes the session, opens
//! a new one with the original system prompt, and appends the newest turns
//! that fit. `client_transcript.zig` holds every compaction decision.
//!
//! With `--resume-ttl`, a drop enters `reconnecting`. The control unit dials
//! at most `reconnect_attempts_max` times, then resumes with the stored token
//! and lets `reconcile` judge the turn in flight. A failed resume rebuilds the
//! session from the transcript.

const std = @import("std");
const span = @import("../core/span.zig");
const Io = std.Io;

const cli = @import("command.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const time = @import("../core/time.zig");
const limits = @import("../core/limits.zig");
const chat_mod = @import("../core/chat.zig");
const protocol = @import("../serve/protocol.zig");
const client_mod = @import("../serve/client.zig");
const transcript_mod = @import("client_transcript.zig");
const Client = client_mod.Client;
const Transcript = transcript_mod.Transcript;
const Body = client_mod.OwnedEvent.Body;

// -- constants --------------------------------------------------------------

/// A stdin line holds at most `line_bytes_max - 1` bytes plus its newline.
const line_bytes_max = transcript_mod.prompt_bytes_max;

const text_bytes_max = 64 * 1024;
const queue_len = 32;
const preview_bytes = 60;

/// The stderr writer buffer.
const stderr_buffer_bytes = 1024;
/// The UTF-8 tail that waits for the next token. A codepoint holds at most 4
/// bytes, so the tail holds at most 3.
const codepoint_tail_max = 3;

const too_long_fmt = "line exceeds {d} bytes, dropped";
const window_fmt = "window: {d}/{d} positions used";
const session_open_fmt = "session {d} open";
const startup_fmt = session_open_fmt ++ ". context {d} tokens per session. turn cap {d} tokens";
const no_session_note = "no session is open, quitting";
const context_full_note = "The window is full. The client compacts the conversation into a " ++
    "new session.";
const evicted_fmt = "The transcript drops its {d} oldest turns to stay inside {d} bytes of text.";
const barrier_fmt = "The turn does not fit {d} bytes of transcript text. The client drops " ++
    "it and {d} older turns.";
const abandoned_fmt = "The failed turn already dropped the {d} oldest turns from the transcript.";
const retry_fmt = "The turn of {d} tokens does not fit the {d} positions left. The client " ++
    "compacts and sends it once more.";
const refuse_fmt = "The prompt of {d} tokens does not fit beside the {d} system rows in the " ++
    "window of {d} positions.";
const refit_fmt = "The kept pairs of {d} tokens do not fit the {d} positions left. The " ++
    "client learns the cost and compacts again.";
const reopen_note = "The append failed. The client opens the session once more.";
const append_lost_note = "The append failed again. The client clears the transcript and " ++
    "keeps the system prompt only.";
const refit_stuck_note = "The refusal does not lower the kept pairs. The client clears the " ++
    "transcript and keeps the system prompt only.";
const compact_empty_note = "The transcript holds no turns, so a compaction has nothing to keep.";
const keep_granted_fmt = "The server keeps session {d} for {d} ms after a drop.";
const keep_off_note = "The server keeps no session after a drop. A drop ends the chat.";
const reconnect_fmt = "Reconnect attempt {d} of {d} in {d} ms.";
const reconnect_failed_fmt = "The client cannot reconnect after {d} attempts. The last " ++
    "error is {t}.";
const server_changed_note = "The new server has another model, window, or end token. The " ++
    "client ends the chat.";
const reconnecting_line_note = "The client is reconnecting and drops the line. Type it " ++
    "again after the resume.";
const resumed_fmt = "Session {d} resumes at position {d} after {d} turns.";
const no_lease_note = "The server holds no lease for the session. The client rebuilds it " ++
    "from the transcript.";
const rebuild_resend_note = "The new session does not hold the turn in flight. The client " ++
    "sends it again.";
const never_ran_note = "The server did not run the turn in flight. The client sends it again.";
const cut_reply_fmt = "The drop cut the reply. The server generated {d} {s} that the client " ++
    "did not get. The model context holds the whole reply.";
const inconsistent_fmt = "The resume summary does not match the {d} tokens that the client " ++
    "got. The client keeps its text.";
const request_lost_note = "The request in flight at the drop is lost. Send it again.";

/// At most this many dials, each after a wait that starts at the first delay
/// and doubles up to the cap.
const reconnect_attempts_max: u32 = 8;

const reconnect_delay_first_ms: u32 = 100;
const reconnect_delay_max_ms: u32 = 2000;

/// A relay that accepts and never forwards answers never, so the attempt ends
/// here.
const reconnect_greet_deadline_ms: u32 = 10_000;

/// The server answers a resume only after the prefill in flight. A cold
/// prefill of 32704 positions takes about 55 s on the fused GPU runner, so the
/// deadline leaves twice that.
const reconnect_resume_deadline_ms: u32 = 120_000;

const commands_text =
    \\  commands (typed at the prompt)
    \\  /cancel   stop the turn in flight
    \\  /stats    print the server counters
    \\  /ping     print the round trip time
    \\  /close    close the session and open a new one
    \\  /clone    fork the session and continue on the fork
    \\  /compact  keep the newest turns that fit and reopen the session
    \\  /history  print the turns the client keeps
    \\  /quit     end the connection
    \\  /help     print this command list
    \\
;

comptime {
    std.debug.assert(line_bytes_max <= std.math.maxInt(u16));
    // A prompt line fits one `generate` frame, so the client never builds an
    // oversized frame.
    std.debug.assert(line_bytes_max <= protocol.generate_text_bytes_max);
}

// -- types ------------------------------------------------------------------

const Input = union(enum) {
    line: struct { len: u16, buf: [line_bytes_max]u8 },
    line_too_long,
    frame: client_mod.OwnedEvent,
    stdin_closed,
    socket_closed: Client.NextError,
    /// The timer with this id fired. A reconnect waits on one at a time.
    timer: u32,
    /// The attempt task of this generation passed the greeting.
    greeted: u32,
    attempt_done: u32,
};

const InputQueue = Io.Queue(Input);

const Command = union(enum) {
    empty,
    prompt: []const u8,
    cancel,
    stats,
    ping,
    close,
    clone,
    compact,
    history,
    quit,
    help,
    unknown: []const u8,
};

/// The flags of `client`, and the sampling and per-turn fields they set.
const Options = struct {
    connect: ?[]const u8 = null,
    socket: ?[]const u8 = null,
    system: []const u8 = "",
    max_tokens: u32 = 0,
    /// Null takes `reserveFor` at each compaction.
    reserve: ?u32 = null,
    sampling: SamplingSource = .greedy,
    temperature: ?f32 = null,
    min_p: ?f32 = null,
    repetition_penalty: ?f32 = null,
    seed: ?u64 = null,
    thinking: chat_mod.ThinkingFlag = .auto,
    thinking_budget: ?u32 = null,
    no_stream: bool = false,
    /// The lease that the client asks for each session, in milliseconds.
    resume_ttl: u32 = 0,

    pub const help: args.Help = .{
        .usage = "zigferencer client (--connect IP:PORT | --socket PATH) [flags]",
        .about = "Chat with a running server over the binary wire. Each line you type is " ++
            "one turn.\n",
        .footer = commands_text ++
            \\
            \\examples:
            \\  zigferencer client --connect 127.0.0.1:8000
            \\  zigferencer client --socket /tmp/zigferencer.sock --system "Answer in one sentence."
            \\
        ,
    };
    pub const docs = .{
        .connect = args.Doc{
            .group = "connection",
            .value = "IP:PORT",
            .text = "connect over TCP. The host must be an IP address",
        },
        .socket = args.Doc{
            .group = "connection",
            .value = "PATH",
            .text = "connect to a server on a unix socket",
        },
        .system = args.Doc{
            .group = "session",
            .text = "system prompt of each session",
            .default = "none",
        },
        .max_tokens = args.Doc{
            .group = "session",
            .text = "the most tokens one turn generates. 0 = until the window ends",
        },
        .reserve = args.Doc{
            .group = "session",
            .text = "positions a compaction leaves free for the next turn",
            .default = std.fmt.comptimePrint(
                "the longest turn so far, from window / {d}\nto window / {d}",
                .{
                    transcript_mod.reserve_floor_divisor, transcript_mod.reserve_ceiling_divisor,
                },
            ),
        },
        .sampling = args.Doc{
            .group = "sampling",
            .text = "model for the card row of the think mode of the turns, or greedy",
        },
        .temperature = args.Doc{
            .group = "sampling",
            .text = "sampling temperature. 0 is greedy",
            .default = "0, or the card value",
        },
        .min_p = args.Doc{
            .group = "sampling",
            .text = "keep the ids with at least X times the top probability, 0..1",
            .default = "0, or the card value",
        },
        .repetition_penalty = args.Doc{
            .group = "sampling",
            .text = "factor on the ids already in the context, above 0 and at most 2.\n1 is off",
            .default = "1, or the card value",
        },
        .seed = args.Doc{
            .group = "sampling",
            .text = "sampling seed of the session. Set, every turn restarts the\nsampling " ++
                "stream at N, so a turn replays. Unset, the stream\nof the session " ++
                "continues across turns. The client has no\nfrequency penalty flag. The " ++
                "server refuses that penalty until\nit supports it",
            .default = "unset",
        },
        .thinking = args.Doc{
            .group = "thinking (models with think tags)",
            .text = "the think switch of every turn. auto sends no switch",
            .default = "auto, the server mode",
        },
        .thinking_budget = args.Doc{
            .group = "thinking (models with think tags)",
            .text = "think tokens before the server closes the block. 0 is unbounded",
            .default = "the server value",
        },
        .no_stream = args.Doc{ .group = "mode", .text = "print each reply when its turn ends" },
        .resume_ttl = args.Doc{
            .group = "reconnect (streaming mode)",
            .value = "MS",
            .text = "ask the server to keep each session for MS after a drop.\n0 keeps " ++
                "none, and a drop ends the chat. The server clamps MS\nto its own maximum. " ++
                "On a drop the client tries to reconnect\nat most 8 times, after 100 ms, " ++
                "then twice as long each time,\nat most 2 s. An attempt waits at most 10 s " ++
                "for the greeting\nand 120 s for the resume. /quit ends the wait at once. " ++
                "The\nclient then resumes the session. A turn that the server never\nran " ++
                "goes out again. If the lease is gone, the client opens a\nnew session and " ++
                "sends the kept turns",
        },
    };

    /// The think switch that every turn sends. Null sends none.
    fn thinkOpen(o: *const Options) ?bool {
        return switch (o.thinking) {
            .auto => null,
            .on => true,
            .off => false,
        };
    }

    fn turnsOpen(o: *const Options, info: Client.HelloInfo) bool {
        return o.thinkOpen() orelse (info.think_mode == .on);
    }

    /// The sampling of every session. `model` takes the card row that the
    /// server sends for the think mode of the turns.
    fn sessionSampling(o: *const Options, info: Client.HelloInfo) protocol.Sampling {
        var s = switch (o.sampling) {
            .greedy => protocol.Sampling.greedy,
            .model => info.recommendedFor(o.turnsOpen(info)),
        };

        if (o.temperature) |t| {
            s.temperature = t;
        }

        if (o.min_p) |m| {
            s.min_p = m;
        }

        if (o.repetition_penalty) |r| {
            s.repetition_penalty = r;
        }

        s.seed = o.seed orelse 0;
        return s;
    }

    /// The per-turn fields of every `generate`. With `--seed`, each turn
    /// asks for a reseed, so the turn replays.
    fn turn(
        o: *const Options,
        max_tokens: u32,
        session_sampling: protocol.Sampling,
    ) Client.TurnOptions {
        return .{
            .max_tokens = max_tokens,
            .sampling = if (o.seed != null) session_sampling else null,
            .reseed = o.seed != null,
            .think_open = o.thinkOpen(),
            .think_budget = o.thinking_budget,
        };
    }
};

const SamplingSource = enum { model, greedy };

/// The line that ends each turn. Only `max_seq` comes from `hello`, and the
/// other fields come from the server's `done`.
const Summary = struct {
    finish_reason: protocol.FinishReason,
    gen_tokens: u32,
    reasoning_tokens: u32,
    kv_pos: u32,
    max_seq: u32,
    prefill_ms: f32,
    decode_ms: f32,

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

const Shade = enum { reply, think, tag };

/// The think block of one turn, read from the channel of each token event.
/// The server marks the tags, so the client needs no tag text.
const ThinkView = struct {
    /// The shade after the last event, for the bytes a turn end flushes.
    last: Shade = .reply,

    /// The shade of one event. A tag prints nothing.
    fn step(v: *ThinkView, channel: protocol.Channel) Shade {
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

/// The shades of a blocking reply, the token hook of `collectTurnWith`. The
/// hook counts the text bytes as the copy does, so each offset matches the
/// collected text.
const ReplyShades = struct {
    eos_id: u32,
    /// A set bit marks a byte of a think block, tags included. The bit length
    /// is the size of the text buffer.
    dim: std.DynamicBitSetUnmanaged,
    /// A set bit marks a byte of a think tag, which prints nothing.
    tag: std.DynamicBitSetUnmanaged,
    view: ThinkView = .{},
    /// The text bytes of the reply so far, the `text_total` of the turn.
    total: usize = 0,

    fn init(arena: std.mem.Allocator, text_cap: usize, eos_id: u32) !ReplyShades {
        return .{
            .eos_id = eos_id,
            .dim = try .initEmpty(arena, text_cap),
            .tag = try .initEmpty(arena, text_cap),
        };
    }

    fn reset(s: *ReplyShades) void {
        s.dim.unsetAll();
        s.tag.unsetAll();
        s.* = .{ .eos_id = s.eos_id, .dim = s.dim, .tag = s.tag };
    }

    /// The copy skips the text of the end-of-sequence token, so the count
    /// skips it too. A byte past the text buffer has no shade to keep.
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

    /// Print each run of one shade in `text`, the kept prefix of the reply,
    /// between its color and a reset. A tag run prints nothing.
    fn write(
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

    /// Record each reply run of `text` in the transcript. Think text never
    /// enters it.
    fn record(s: *const ReplyShades, t: *Transcript, text: []const u8) void {
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
    fn runEnd(s: *const ReplyShades, start: usize, len: usize) usize {
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

/// The client side of compaction: the transcript and the numbers that weigh
/// it.
const History = struct {
    t: *Transcript,
    /// The `kv_pos` of a session opened with the original system prompt. Each
    /// re-open measures it again.
    system_rows: u32,
    window: u32,
    /// The `--reserve` value, or null for `reserveFor`.
    fixed_reserve: ?u32,

    fn init(arena: std.mem.Allocator, system_rows: u32, window: u32, fixed_reserve: ?u32) !History {
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

    fn effectiveReserve(h: *const History) u32 {
        const r = h.fixed_reserve orelse transcript_mod.reserveFor(h.window, h.t.longest_cost);
        std.debug.assert(r < h.window);
        return r;
    }

    fn budget(h: *const History, reserve: u32) transcript_mod.Budget {
        return .{ .window = h.window, .reserve = reserve, .system_rows = h.system_rows };
    }

    fn plan(h: *const History, reserve: u32) transcript_mod.Plan {
        return transcript_mod.plan(h.t, h.budget(reserve));
    }

    /// Keep the system prompt only, for a cycle that cannot make progress.
    fn keepNothing(h: *History, cycle: *Cycle) void {
        cycle.plan = .{ .kept = 0, .dropped = cycle.plan.kept, .estimate = h.system_rows };
        h.t.clear();
    }

    fn retry(h: *const History, q: u32) transcript_mod.Retry {
        return transcript_mod.retryPlan(h.system_rows, h.window, h.effectiveReserve(), q);
    }

    /// Plan a compaction with `reserve`, and drop the turns that it does not
    /// keep. The window is fixed for the connection, so a dropped turn never
    /// fits again.
    fn shrink(h: *History, reserve: u32) transcript_mod.Plan {
        const p = h.plan(reserve);
        h.t.dropOldest(p.dropped);
        std.debug.assert(h.t.count() == p.kept);
        return p;
    }

    /// The kept pairs of the append frame that starts at turn `from`. They
    /// borrow the transcript text.
    fn frame(
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
const Cycle = struct {
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

/// The estimate comes from the plan, and `kv_pos` is the cost that the server
/// measured.
const Compacted = struct {
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

    fn of(cycle: *const Cycle, kv_pos: u32, window: u32) Compacted {
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
            "transcript: {d} turns, {d} positions, reserve {d}, window {d}. A compaction " ++
                "keeps {d}.",
            .{
                h.turns, h.positions, h.reserve, h.window, h.keeps,
            },
        );
    }
};

/// One turn of `/history`. A star marks a turn that a compaction keeps now.
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

/// Both output streams with their palettes. Model text and command replies go
/// to stdout. Prompts, diagnostics, and summaries go to stderr.
const Out = struct {
    o: *Io.Writer,
    e: *Io.Writer,
    po: style.Palette,
    pe: style.Palette,

    fn note(out: Out, comptime fmt: []const u8, a: anytype) !void {
        try out.e.print(fmt ++ "\n", a);
        try out.e.flush();
    }

    fn dim(out: Out, comptime fmt: []const u8, a: anytype) !void {
        try out.e.writeAll(out.pe.dim);
        try out.e.print(fmt, a);
        try out.e.print("{s}\n", .{out.pe.reset});
        try out.e.flush();
    }

    fn marker(out: Out) !void {
        try out.e.print("{s}> {s}", .{ out.pe.bold, out.pe.reset });
        try out.e.flush();
    }

    fn serverError(out: Out, le: ?Client.LastError) !void {
        const e = le orelse return out.note("server error without detail", .{});
        try out.note("server error: {t}: {s}", .{ e.code, e.detail() });
    }

    fn turnRefused(out: Out, le: ?Client.LastError, session: Client.Opened, max_seq: u32) !void {
        try out.serverError(le);
        const e = le orelse return;
        if (e.code == .turn_too_long) {
            try out.note(window_fmt, .{ session.kv_pos, max_seq });
        }
    }

    fn recorded(out: Out, rec: transcript_mod.Recorded) !void {
        if (rec.barrier) {
            return out.note(barrier_fmt, .{ transcript_mod.pair_bytes_max, rec.evicted });
        }

        if (rec.evicted > 0) {
            try out.note(evicted_fmt, .{ rec.evicted, transcript_mod.pair_bytes_max });
        }
    }

    fn abandoned(out: Out, evicted: u32) !void {
        if (evicted > 0) {
            try out.note(abandoned_fmt, .{evicted});
        }
    }

    /// The refusal of a prompt that no compaction makes room for.
    fn refused(
        out: Out,
        le: ?Client.LastError,
        session: Client.Opened,
        h: *const History,
        q: u32,
    ) !void {
        try out.turnRefused(le, session, h.window);
        try out.note(refuse_fmt, .{ q, h.system_rows, h.window });
    }

    fn history(out: Out, h: *const History) !void {
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

    fn finish(out: Out, s: Summary) !void {
        try out.dim("{f}", .{s});
        if (finishNote(s.finish_reason)) |note_text| {
            try out.note("{s}", .{note_text});
        }
    }

    fn stats(out: Out, s: Client.Stats) !void {
        try out.o.print("sessions_open={d} sessions_decoding={d} " ++
            "tokens_generated_total={d} uptime_ms={d}\n", .{
            s.sessions_open, s.sessions_decoding, s.tokens_generated_total, s.uptime_ms,
        });
        try out.o.flush();
    }

    fn pong(out: Out, ms: f64) !void {
        try out.o.print("pong in {d:.2} ms\n", .{ms});
        try out.o.flush();
    }
};

/// The states of the streaming control unit. `keeping` waits on the `kept`
/// reply of a new session. `reconnecting` holds the main task while it
/// dials and resumes, so no input reaches the machine in that state.
const State = enum { opening, appending, keeping, idle, streaming, reconnecting, quitting };

/// Parsed once, so each reconnect dials the same server.
const Target = union(enum) {
    tcp: Io.net.IpAddress,
    unix: []const u8,
};

/// The lease of the first session. A zero `ttl_ms` keeps no session.
const Keep = struct { ttl_ms: u32, token: ?protocol.Token };

/// The values that startup resolves once and that both loops read.
const Startup = struct {
    out: Out,
    opts: *const Options,
    client: *Client,
    info: Client.HelloInfo,
    sampling: protocol.Sampling,
    session: Client.Opened,
    hist: *History,
};

/// The streaming control unit. Only the main task touches it.
const Chat = struct {
    io: Io,
    client: *Client,
    out: Out,
    opts: *const Options,
    /// The text of the `info.eos_id` token frame is not part of the reply.
    info: Client.HelloInfo,
    sampling: protocol.Sampling,
    state: State = .idle,
    session: ?Client.Opened,
    /// The request id of the `generate`, `session_open`, `session_clone`, or
    /// `session_append` that the current state waits on.
    pending_id: u32 = 0,
    /// A token can end inside a codepoint. The incomplete tail waits here for
    /// the next token.
    hold: [codepoint_tail_max]u8 = undefined,
    hold_len: usize = 0,
    think: ThinkView = .{},
    ping: ?struct { id: u32, t0: Io.Clock.Timestamp } = null,
    /// Replies to `/stats` and `session_close` that have not arrived.
    replies_due: u32 = 0,
    /// `/quit` and EOF wait for the turn in flight and every due reply, so
    /// piped input sees its whole output before the hang-up.
    quit_pending: bool = false,
    lost_session: bool = false,
    hist: *History,
    cycle: ?Cycle = null,
    /// The prompt of the turn in flight, kept so a refused turn can go out
    /// once more.
    prompt_buf: [line_bytes_max]u8 = undefined,
    prompt_len: u32 = 0,
    /// The prompt in `prompt_buf` waits for the open of a compaction.
    retry_waiting: bool = false,
    /// The turn in flight is a retry, so a second refusal is final.
    retried: bool = false,
    target: Target,
    /// The lease that each new session asks for. 0 keeps no session.
    keep_ttl: u32,
    /// The token of the lease on `session`. Null falls back to a rebuild.
    lease: ?protocol.Token,
    /// The token frames of the turn in flight that the server counts, for
    /// `reconcile` after a drop.
    received: u32 = 0,
    /// The allocator of each new client.
    gpa: std.mem.Allocator,
    /// The generation of the attempt task in flight, and the id of the newest
    /// timer. A marker of an older one is stale.
    gen: u32 = 0,
    timer_gen: u32 = 0,
    /// The sources of `/clone`, which stay open on this connection. The quit
    /// closes them. A source past the bound has no lease, so the hang-up
    /// frees it.
    sources: [limits.max_logical_sessions]u32 = undefined,
    sources_len: u32 = 0,
};

/// What one reconnect found on the server.
const Outcome = union(enum) {
    resumed: Client.ResumedSession,
    /// The client held no lease, or the server refused the resume.
    no_lease,
    /// `/quit` or the end of stdin arrived while the client reconnected.
    quit,
};

/// The values one attempt task needs. They are copies, so the task shares
/// no state with the control unit.
const AttemptJob = struct {
    gpa: std.mem.Allocator,
    io: Io,
    target: Target,
    /// The `hello_ack` of the first server. A new server must match it.
    expect: Client.HelloInfo,
    next_request_id: u32,
    token: ?protocol.Token,
    gen: u32,
    /// The last pass of the reconnect loop. A retryable refusal ends in a
    /// rebuild there, because no later pass can retry.
    final: bool,
};

/// The result of one attempt task. A client that it returns is connected
/// and greeted, and the control unit owns it.
const Attempt = struct {
    client: ?Client = null,
    info: ?Client.HelloInfo = null,
    outcome: union(enum) {
        resumed: Client.ResumedSession,
        /// The resume was refused, with the reason, or the client held no
        /// token.
        no_lease: ?Client.LastError,
        /// The new server serves another model or window.
        changed,
        failed: anyerror,
    },
};

/// What ends one wait of the `reconnecting` state.
const Wake = enum { timer, greeted, attempt, quit };

const ResumeRefusal = enum {
    /// The server holds no such lease, so the session must be rebuilt.
    no_lease,
    /// The token can still be valid, so the next attempt tries it again.
    retry,
};

/// The state of the blocking loop.
const Blocking = struct {
    out: Out,
    opts: *const Options,
    client: *Client,
    info: Client.HelloInfo,
    sampling: protocol.Sampling,
    hist: *History,
    session: Client.Opened,
    ids: []u32,
    text: []u8,
    shades: *ReplyShades,
};

/// A failed append frame of turns `from..to`, with the numbers of a
/// `turn_too_long` refusal when it was one.
const AppendFailure = struct { from: u32, to: u32, too_long: ?transcript_mod.TooLong };

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const opts = try parseOptions(ctx) orelse return;

    var out_buf: [cli.stdout_buffer_bytes]u8 = undefined;
    var err_buf: [stderr_buffer_bytes]u8 = undefined;
    var fo: Io.File.Writer = .initStreaming(.stdout(), ctx.io, &out_buf);
    var fe: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &err_buf);
    const out: Out = .{
        .o = &fo.interface,
        .e = &fe.interface,
        .po = style.Palette.init(ctx.color),
        .pe = style.Palette.init(ctx.color_err),
    };

    const target = try parseTarget(ctx, &opts);
    var client = try connect(ctx, target, &opts, out);
    defer client.deinit();

    const info = client.hello() catch |err| return startupFailed(out, &client, err);
    try checkServer(out, &opts, info);

    const sampling = opts.sessionSampling(info);
    if (opts.sampling == .model) {
        try printSampling(out, &opts, info, sampling);
    }

    const session = client.openSessionInfo(opts.system, sampling) catch |err|
        return startupFailed(out, &client, err);
    const hist = try ctx.arena.create(History);
    hist.* = try .init(ctx.arena, session.kv_pos, info.max_seq, opts.reserve);
    try printStartup(out, &opts, info, session.id);

    const keep: Keep = if (opts.resume_ttl > 0)
        try startKeep(out, &client, info, session.id, opts.resume_ttl)
    else
        .{ .ttl_ms = 0, .token = null };

    const st: Startup = .{
        .out = out,
        .opts = &opts,
        .client = &client,
        .info = info,
        .sampling = sampling,
        .session = session,
        .hist = hist,
    };
    if (!opts.no_stream) return runStreaming(ctx, st, target, keep);

    std.debug.assert(keep.ttl_ms == 0);
    return runBlocking(ctx, st);
}

// -- helpers: startup -------------------------------------------------------

/// Null means `--help` printed and the command has nothing more to do.
fn parseOptions(ctx: cli.Context) !?Options {
    const o = try args.parseInto(Options, ctx) orelse return null;
    if ((o.connect == null) == (o.socket == null)) {
        return args.fail(ctx.io, "give exactly one of --connect or --socket", .{});
    }

    if (o.temperature) |t| {
        if (!(t >= 0) or !std.math.isFinite(t)) {
            return args.fail(ctx.io, "--temperature must be 0 or more", .{});
        }
    }

    if (o.thinking_budget) |b| {
        if (b > limits.max_model_ctx) {
            return args.fail(
                ctx.io,
                "--thinking-budget must be in 0..{d}",
                .{limits.max_model_ctx},
            );
        }
    }

    if (o.min_p) |m| {
        if (!(m >= 0 and m <= 1)) return args.fail(ctx.io, "--min-p must be in 0..1", .{});
    }

    if (o.repetition_penalty) |r| {
        if (!(r > 0 and r <= limits.max_repetition_penalty)) {
            return args.fail(
                ctx.io,
                "--repetition-penalty must be above 0 and at most {d}",
                .{limits.max_repetition_penalty},
            );
        }
    }

    if (o.resume_ttl > limits.max_resume_ttl_ms) {
        return args.fail(
            ctx.io,
            "--resume-ttl must be in 0..{d} ms, the lease ceiling",
            .{limits.max_resume_ttl_ms},
        );
    }
    if (o.resume_ttl > 0 and o.no_stream) {
        return args.fail(
            ctx.io,
            "--resume-ttl needs the streaming mode, so it does not go with --no-stream",
            .{},
        );
    }
    if (o.system.len > protocol.system_prompt_bytes_max) {
        return args.fail(
            ctx.io,
            "--system holds {d} bytes. One session_open frame carries at most {d}.",
            .{
                o.system.len, protocol.system_prompt_bytes_max,
            },
        );
    }

    return o;
}

fn parseTarget(ctx: cli.Context, opts: *const Options) !Target {
    if (opts.socket) |path| return .{ .unix = path };

    const hostport = opts.connect.?;
    const colon = std.mem.lastIndexOfScalar(u8, hostport, ':') orelse
        return args.fail(ctx.io, "--connect wants IP:PORT", .{});
    const port = std.fmt.parseInt(u16, hostport[colon + 1 ..], 10) catch
        return args.fail(ctx.io, "--connect wants IP:PORT", .{});
    const address = Io.net.IpAddress.parse(hostport[0..colon], port) catch
        return args.fail(ctx.io, "--connect wants an IP address and a port", .{});
    return .{ .tcp = address };
}

/// Refuse a server whose `hello_ack` contradicts the options. The window
/// comes from the network, and the blocking mode sizes its id buffer from it.
fn checkServer(out: Out, opts: *const Options, info: Client.HelloInfo) !void {
    if (info.max_seq == 0 or info.max_seq > limits.max_model_ctx) {
        try out.note(
            "The server reports a window of {d} positions. The client accepts 1 to {d}.",
            .{
                info.max_seq, limits.max_model_ctx,
            },
        );
        return error.WindowOutOfRange;
    }

    if (opts.reserve) |reserve| {
        if (reserve >= info.max_seq) {
            try out.note(
                "--reserve {d} is at or past the window of {d} positions.",
                .{ reserve, info.max_seq },
            );
            return error.ReserveOutOfRange;
        }
    }

    if (info.think_mode == .none and (opts.thinking == .on or opts.thinking_budget != null)) {
        try out.note(
            "The server model has no think tags, so --thinking on and --thinking-budget do " ++
                "not apply.",
            .{},
        );
        return error.ThinkingUnsupported;
    }
}

fn printSampling(
    out: Out,
    opts: *const Options,
    info: Client.HelloInfo,
    sampling: protocol.Sampling,
) !void {
    try out.dim(
        "sampling: think mode {t}, turns open {}, think budget {d}, temperature {d:.2}, " ++
            "top_k {d}, top_p {d:.2}, min_p {d:.2}, presence_penalty {d:.2}, " ++
            "repetition_penalty {d:.2}",
        .{
            info.think_mode,
            opts.turnsOpen(
                info,
            ),
            opts.thinking_budget orelse info.think_budget,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.min_p,
            sampling.presence_penalty,
            sampling.repetition_penalty,
        },
    );
}

fn printStartup(out: Out, opts: *const Options, info: Client.HelloInfo, session_id: u32) !void {
    const cap = turnBudget(opts.max_tokens, info.max_seq);
    if (opts.reserve) |reserve| {
        try out.dim(
            startup_fmt ++ ", reserve {d} positions",
            .{ session_id, info.max_seq, cap, reserve },
        );
    } else {
        try out.dim(startup_fmt ++ ", reserve adaptive, from {d} to {d} positions", .{
            session_id,                                 info.max_seq, cap,
            transcript_mod.reserveFor(info.max_seq, 0),
            transcript_mod.reserveFor(
                info.max_seq,
                std.math.maxInt(u32),
            ),
        });
    }
}

fn dial(ctx: cli.Context, target: Target) !Client {
    return switch (target) {
        .unix => |path| Client.connectUnix(ctx.gpa, ctx.io, path),
        .tcp => |address| Client.connectTcp(ctx.gpa, ctx.io, address),
    };
}

fn connect(ctx: cli.Context, target: Target, opts: *const Options, out: Out) !Client {
    return dial(ctx, target) catch |err| {
        try out.note("cannot connect to {s}: {t}", .{ opts.socket orelse opts.connect.?, err });
        return err;
    };
}

/// Keep the first session for `ttl_ms`. A server with resumption off, or a
/// refused keep, gives no lease, and the chat goes on without one.
fn startKeep(
    out: Out,
    client: *Client,
    info: Client.HelloInfo,
    session_id: u32,
    ttl_ms: u32,
) !Keep {
    std.debug.assert(ttl_ms > 0);

    const off: Keep = .{ .ttl_ms = 0, .token = null };

    if (info.resume_ttl_max_ms == 0) {
        try out.note(keep_off_note, .{});
        return off;
    }

    const k = client.keepSession(session_id, ttl_ms) catch |err| {
        if (err != error.ServerError) return err;

        try out.serverError(client.last_error);
        try out.note(keep_off_note, .{});
        return off;
    };
    if (k.granted_ms == 0) {
        try out.note(keep_off_note, .{});
        return off;
    }

    try out.dim(keep_granted_fmt, .{ session_id, k.granted_ms });
    return .{ .ttl_ms = k.granted_ms, .token = k.token };
}

/// A failed write of the report is dropped, so the startup error itself
/// reaches `main`.
fn startupFailed(out: Out, client: *const Client, err: anyerror) anyerror {
    if (err == error.ServerError) {
        out.serverError(client.last_error) catch {};
    } else {
        out.note("startup failed: {t}", .{err}) catch {};
    }

    return err;
}

// -- helpers: streaming loop ------------------------------------------------

fn runStreaming(ctx: cli.Context, st: Startup, target: Target, keep: Keep) !void {
    const io = ctx.io;
    const buf = try ctx.arena.alloc(Input, queue_len);
    var queue: InputQueue = .init(buf);

    // A reconnect awaits the old socket task and starts a new one, so the
    // future is null between the two.
    var socket_future: ?Io.Future(void) = try io.concurrent(socketTask, .{ io, st.client, &queue });
    var stdin_future: ?Io.Future(void) = null;
    // The queue closes first, so a producer blocked on a full queue returns.
    // The hang-up then ends the socket task, and the cancel ends a stdin read.
    defer {
        queue.close(io);
        st.client.hangUp();
        if (socket_future) |*f| {
            f.await(io);
        }

        if (stdin_future) |*f| {
            f.cancel(io);
        }
    }

    stdin_future = try io.concurrent(stdinTask, .{ io, &queue });

    var chat: Chat = .{
        .io = io,
        .client = st.client,
        .out = st.out,
        .opts = st.opts,
        .info = st.info,
        .sampling = st.sampling,
        .session = st.session,
        .hist = st.hist,
        .target = target,
        .keep_ttl = keep.ttl_ms,
        .lease = keep.token,
        .gpa = ctx.gpa,
    };
    try enterIdle(&chat);

    // Each pass takes one input. The loop ends only in `quitting`.
    while (chat.state != .quitting) {
        const input = try queue.getOne(io);
        switch (input) {
            .line => |*l| try onLine(&chat, l.buf[0..l.len]),
            .line_too_long => try st.out.note(too_long_fmt, .{line_bytes_max - 1}),
            .frame => |*ev| try onFrame(&chat, ev),
            .stdin_closed => chat.quit_pending = true,
            .socket_closed => |err| try onDrop(&chat, &queue, &socket_future, err),
            // A marker of a reconnect that already ended.
            .timer, .greeted, .attempt_done => {},
        }

        settleQuit(&chat);
    }

    if (chat.lost_session) return error.SessionLost;

    closeAll(&chat);
}

/// The connection dropped. Without a lease, or during a quit, the chat ends
/// with an error. Else the chat reconnects and starts a new socket task.
fn onDrop(
    c: *Chat,
    queue: *InputQueue,
    socket_future: *?Io.Future(void),
    err: Client.NextError,
) !void {
    try c.out.note("connection closed: {t}", .{err});
    if (c.keep_ttl == 0 or c.quit_pending) return error.ConnectionClosed;

    // The socket task puts `socket_closed` last and returns, so no frame of
    // the old connection follows it.
    socket_future.*.?.await(c.io);
    socket_future.* = null;

    const prior = c.state;
    const outcome = try reconnect(c, queue);
    if (outcome == .quit) {
        c.state = .quitting;
        return;
    }

    socket_future.* = try c.io.concurrent(socketTask, .{ c.io, c.client, queue });
    try afterReconnect(c, prior, outcome);
}

/// A quit closes every session on purpose, so none waits out a lease. The
/// closes go out before the hang-up on the same carrier, and the hang-up ends
/// the chat whether they arrive or not.
fn closeAll(c: *Chat) void {
    std.debug.assert(c.state == .quitting);

    for (c.sources[0..c.sources_len]) |sid| {
        _ = c.client.send(.session_close, sid) catch return;
    }

    c.sources_len = 0;

    if (c.lease != null) {
        _ = c.client.send(.session_close, c.session.?.id) catch return;
    }
}

fn stdinTask(io: Io, queue: *InputQueue) void {
    var buf: [line_bytes_max]u8 = undefined;
    var reader = Io.File.stdin().readerStreaming(io, &buf);
    const r = &reader.interface;

    // Each pass reads one line. The loop ends at EOF, a read error, a cancel
    // (seen as a read error), or a closed queue.
    while (true) {
        const maybe = r.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                queue.putOne(io, .line_too_long) catch return;
                _ = r.discardDelimiterInclusive('\n') catch break;
                continue;
            },
            error.ReadFailed => break,
        };
        const line = maybe orelse break;

        var in: Input = .{ .line = .{ .len = @intCast(line.len), .buf = undefined } };
        @memcpy(in.line.buf[0..line.len], line);
        queue.putOne(io, in) catch return;
    }

    queue.putOne(io, .stdin_closed) catch {};
}

fn socketTask(io: Io, client: *Client, queue: *InputQueue) void {
    span.nameThread("zf-client");
    // Each pass reads one frame. The loop ends at the first read error, and
    // the hang-up at teardown forces one.
    while (true) {
        const ev = client.next() catch |err| {
            queue.putOne(io, .{ .socket_closed = err }) catch {};
            return;
        };
        // The span starts after the frame arrives, so it holds no idle wait.
        const sp = span.begin(.client_stream);
        defer sp.end();

        queue.putOne(io, .{ .frame = client_mod.toOwned(ev) }) catch return;
    }
}

fn enterIdle(c: *Chat) !void {
    c.state = .idle;
    if (!c.quit_pending) {
        try c.out.marker();
    }
}

fn settleQuit(c: *Chat) void {
    if (c.quit_pending and c.state == .idle and c.replies_due == 0 and c.ping == null) {
        c.state = .quitting;
    }
}

fn onLine(c: *Chat, line: []const u8) !void {
    if (c.quit_pending) return;

    const cmd = parseLine(line);
    switch (cmd) {
        .empty => {},
        .help => try printCommands(c.out.e),
        .quit => c.quit_pending = true,
        .unknown => |word| try c.out.note("unknown command: /{s}", .{word}),
        .history => try c.out.history(c.hist),
        .stats => {
            _ = try c.client.send(.stats, 0);
            c.replies_due += 1;
        },
        .ping => {
            if (c.ping != null) return c.out.note("ping in flight, wait", .{});
            const t0 = Io.Clock.Timestamp.now(c.io, .awake);
            c.ping = .{ .id = try c.client.send(.ping, 0), .t0 = t0 };
        },
        .prompt, .cancel, .close, .clone, .compact => switch (c.state) {
            .idle => try idleLine(c, cmd),
            .streaming => try streamingLine(c, cmd),
            .opening, .appending, .keeping => try c.out.note("session opening, wait", .{}),
            .reconnecting, .quitting => unreachable,
        },
    }
}

fn idleLine(c: *Chat, cmd: Command) !void {
    std.debug.assert(c.state == .idle);

    const sid = c.session.?.id;
    const sampling = c.sampling;
    switch (cmd) {
        .prompt => |text| {
            std.debug.assert(text.len < line_bytes_max);

            @memcpy(c.prompt_buf[0..text.len], text);
            c.prompt_len = @intCast(text.len);
            try sendPrompt(c, false);
        },
        .cancel => try c.out.note("no turn in flight", .{}),
        .close => {
            c.hist.t.clear();
            _ = try c.client.send(.session_close, sid);
            c.replies_due += 1;
            c.session = null;
            c.lease = null;
            c.pending_id = try c.client.send(.{ .session_open = .{
                .sampling = sampling,
                .system_prompt = c.opts.system,
            } }, 0);
            c.state = .opening;
        },
        .clone => {
            c.pending_id = try c.client.send(.{ .session_clone = .{ .sampling = sampling } }, sid);
            c.state = .opening;
        },
        .compact => {
            if (c.hist.t.count() == 0) return c.out.note(compact_empty_note, .{});
            try startCompaction(c, c.hist.effectiveReserve(), false);
        },
        else => unreachable,
    }
}

fn streamingLine(c: *Chat, cmd: Command) !void {
    std.debug.assert(c.state == .streaming);

    switch (cmd) {
        .cancel => _ = try c.client.cancel(c.session.?.id),
        else => try c.out.note("turn in progress, /cancel first", .{}),
    }
}

fn sendPrompt(c: *Chat, retried: bool) !void {
    std.debug.assert(c.session != null);
    std.debug.assert(c.prompt_len > 0);

    const text = c.prompt_buf[0..c.prompt_len];
    c.hist.t.begin(text);
    const budget = turnBudget(c.opts.max_tokens, c.info.max_seq);
    c.pending_id = try c.client.generateWith(c.session.?.id, text, c.opts.turn(budget, c.sampling));

    c.received = 0;
    c.hold_len = 0;
    c.think = .{};
    c.retried = retried;
    c.retry_waiting = false;
    c.state = .streaming;
}

// -- helpers: compaction ----------------------------------------------------

/// Start a compaction cycle. The replies step it in `onFrame` and `onErr`.
fn startCompaction(c: *Chat, reserve: u32, retry: bool) !void {
    std.debug.assert(c.session != null);
    std.debug.assert(c.cycle == null);
    std.debug.assert(reserve <= c.hist.window);

    const total = c.hist.t.count();
    c.cycle = .{ .plan = c.hist.shrink(reserve), .total = total, .reserve = reserve };
    c.retry_waiting = retry;
    try reopen(c);
}

/// Close the open session and open one with the original system prompt. The
/// cycle restarts at its first kept pair.
fn reopen(c: *Chat) !void {
    if (c.session) |s| {
        _ = try c.client.send(.session_close, s.id);
        c.replies_due += 1;
        c.session = null;
        c.lease = null;
    }

    if (c.cycle) |*cy| {
        cy.next = 0;
    }

    c.pending_id = try c.client.send(.{ .session_open = .{
        .sampling = c.sampling,
        .system_prompt = c.opts.system,
    } }, 0);
    c.state = .opening;
}

/// Send the next append frame of the cycle, or end the cycle when every kept
/// pair is in.
fn continueCycle(c: *Chat) !void {
    const cy = &c.cycle.?;
    std.debug.assert(cy.next <= c.hist.t.count());

    if (cy.next == c.hist.t.count()) {
        try c.out.dim("{f}", .{Compacted.of(cy, c.session.?.kv_pos, c.hist.window)});
        c.cycle = null;
        return sessionReady(c);
    }

    var buf: [limits.max_append_pairs]protocol.Pair = undefined;
    const f = c.hist.frame(cy.next, &buf);
    c.pending_id = try c.client.sendAppend(c.session.?.id, f.pairs);
    cy.frame_end = f.end;
    cy.base = c.session.?.kv_pos;
    c.state = .appending;
}

/// The chat leaves a clone source for its fork. The source drops its lease,
/// so a drop frees it, and the quit closes it.
fn leaveSource(c: *Chat, source_id: u32) !void {
    std.debug.assert(c.state == .opening);
    std.debug.assert(c.cycle == null);

    if (c.lease != null) {
        _ = try c.client.send(.{ .session_keep = .{ .ttl_ms = 0 } }, source_id);
        c.replies_due += 1;
    }

    // Past the bound, the source has no lease, and the hang-up frees it.
    if (c.sources_len < c.sources.len) {
        c.sources[c.sources_len] = source_id;
        c.sources_len += 1;
    }
}

/// With resumption on, the client keeps the new session before the next turn,
/// so a drop at any later point finds a lease.
fn sessionReady(c: *Chat) !void {
    std.debug.assert(c.cycle == null);
    std.debug.assert(c.lease == null);

    if (c.keep_ttl == 0) return afterKeep(c);

    c.pending_id = try c.client.send(
        .{ .session_keep = .{ .ttl_ms = c.keep_ttl } },
        c.session.?.id,
    );
    c.state = .keeping;
}

/// The session is ready, with or without a lease. A prompt that waits for a
/// compaction goes out now.
fn afterKeep(c: *Chat) !void {
    if (c.retry_waiting) return sendPrompt(c, true);
    return enterIdle(c);
}

/// A refused frame teaches its cost, and the cycle restarts with a new plan
/// that keeps fewer pairs. Any other failure opens the session once more,
/// and a second one clears the transcript.
fn appendRefused(c: *Chat, e: *const Client.LastError) !void {
    const cy = &c.cycle.?;

    if (tooLong(e.*, .turn)) |tl| {
        try c.out.note(refit_fmt, .{ tl.tokens, tl.room });
        cy.refits += 1;
        switch (transcript_mod.refit(
            c.hist.t,
            c.hist.budget(cy.reserve),
            cy.next,
            cy.frame_end,
            tl.tokens,
            cy.plan.kept,
        )) {
            .plan => |p| cy.plan = p,
            .stuck => {
                try c.out.dim(refit_stuck_note, .{});
                c.hist.keepNothing(cy);
            },
        }

        return reopen(c);
    }

    try c.out.serverError(e.*);
    if (!cy.reopened) {
        cy.reopened = true;
        try c.out.note(reopen_note, .{});
        return reopen(c);
    }

    try c.out.note(append_lost_note, .{});
    c.hist.keepNothing(cy);
    return reopen(c);
}

// -- helpers: frames --------------------------------------------------------

fn onFrame(c: *Chat, ev: *const client_mod.OwnedEvent) !void {
    const mine = ev.header.request_id == c.pending_id;
    switch (ev.body) {
        .token => |*t| if (c.state == .streaming and mine) return onToken(c, t),
        .done => |d| if (c.state == .streaming and mine) return onDone(c, d),
        .session_opened => |m| if (c.state == .opening and mine) {
            return onOpened(c, ev.header.session_id, m.kv_pos);
        },
        .kept => |k| return onKept(c, k, mine),
        .appended => |*a| if (c.state == .appending and mine) return onAppended(c, a),
        .err => |*e| return onErr(c, ev.header.request_id, e),
        .pong => if (c.ping) |pg| if (pg.id == ev.header.request_id) {
            c.ping = null;
            return c.out.pong(time.msBetween(pg.t0, Io.Clock.Timestamp.now(c.io, .awake)));
        },
        .stats_ack => |s| {
            c.replies_due -|= 1;
            return c.out.stats(s);
        },
        .session_closed => {
            c.replies_due -|= 1;
            return;
        },
        .hello_ack, .shutdown_ack, .resumed => {},
    }

    try c.out.note("unexpected frame: {t}", .{ev.body});
}

fn onToken(c: *Chat, t: *const @FieldType(Body, "token")) !void {
    if (transcript_mod.countsAsGenerated(t.channel)) {
        c.received += 1;
    }

    if (t.token_id == c.info.eos_id) return;

    const text = t.text[0..t.len];
    const shade = c.think.step(t.channel);
    if (shade == .reply) {
        c.hist.t.appendReply(text);
    }

    if (shade == .tag) return;

    return emitText(c, text, shade);
}

fn onDone(c: *Chat, d: @FieldType(Body, "done")) !void {
    try endTurn(c);

    const before = c.session.?.kv_pos;
    std.debug.assert(d.kv_pos >= before);
    const rec = c.hist.t.finish(
        transcript_mod.liveCost(d.kv_pos - before, d.finish_reason, d.gen_tokens),
    );
    c.session.?.kv_pos = d.kv_pos;

    try c.out.finish(.{
        .finish_reason = d.finish_reason,
        .gen_tokens = d.gen_tokens,
        .reasoning_tokens = d.reasoning_tokens,
        .kv_pos = d.kv_pos,
        .max_seq = c.info.max_seq,
        .prefill_ms = d.prefill_ms,
        .decode_ms = d.decode_ms,
    });
    try c.out.recorded(rec);

    if (d.finish_reason == .context_full) {
        return startCompaction(c, c.hist.effectiveReserve(), false);
    }

    return enterIdle(c);
}

fn onOpened(c: *Chat, session_id: u32, kv_pos: u32) !void {
    if (c.session) |source| {
        try leaveSource(c, source.id);
    }

    c.session = .{ .id = session_id, .kv_pos = kv_pos };
    c.lease = null;

    if (c.cycle != null) {
        // The original system prompt opens each cycle, so the open measures
        // the system rows exactly.
        c.hist.system_rows = kv_pos;
        return continueCycle(c);
    }

    try c.out.dim(session_open_fmt, .{session_id});
    return sessionReady(c);
}

fn onKept(c: *Chat, k: @FieldType(Body, "kept"), mine: bool) !void {
    // The other `kept` reply is the release of a clone source.
    if (!(c.state == .keeping and mine)) {
        c.replies_due -|= 1;
        return;
    }

    // A grant of 0 is a server that turned resumption off.
    c.lease = if (k.granted_ms > 0) k.token else null;
    return afterKeep(c);
}

fn onAppended(c: *Chat, a: *const protocol.Appended) !void {
    const cy = &c.cycle.?;
    if (a.n != cy.frame_end - cy.next) return c.out.note("unexpected appended count {d}", .{a.n});

    c.session.?.kv_pos = transcript_mod.learnAppended(c.hist.t, cy.next, cy.base, a.positions());
    cy.next = cy.frame_end;
    return continueCycle(c);
}

fn onErr(c: *Chat, request_id: u32, e: *const Client.LastError) !void {
    const mine = request_id == c.pending_id;

    if (c.state == .streaming and mine) {
        try endTurn(c);
        try c.out.abandoned(c.hist.t.abandon());

        const refusal = if (c.retried) null else tooLong(e.*, .turn);
        if (refusal) |tl| {
            switch (c.hist.retry(tl.tokens)) {
                .retry => |r| {
                    try c.out.note(retry_fmt, .{ tl.tokens, tl.room });
                    return startCompaction(c, r, true);
                },
                .refuse => {
                    try c.out.refused(e.*, c.session.?, c.hist, tl.tokens);
                    return enterIdle(c);
                },
            }
        }

        try c.out.turnRefused(e.*, c.session.?, c.info.max_seq);
        return enterIdle(c);
    }

    if (c.state == .appending and mine) return appendRefused(c, e);

    if (c.state == .keeping and mine) {
        // The session lives on with no lease. A drop then rebuilds it.
        try c.out.serverError(e.*);
        return afterKeep(c);
    }

    if (c.state == .opening and mine) {
        c.cycle = null;
        try c.out.serverError(e.*);
        if (c.session != null) return enterIdle(c);

        try c.out.note(no_session_note, .{});
        c.lost_session = true;
        c.state = .quitting;
        return;
    }

    if (c.ping) |pg| {
        if (pg.id == request_id) {
            c.ping = null;
        } else {
            c.replies_due -|= 1;
        }
    } else {
        c.replies_due -|= 1;
    }

    try c.out.serverError(e.*);
}

fn emitText(c: *Chat, text: []const u8, shade: Shade) !void {
    var buf: [codepoint_tail_max + limits.max_token_text_bytes]u8 = undefined;
    std.debug.assert(c.hold_len <= codepoint_tail_max);
    std.debug.assert(text.len <= limits.max_token_text_bytes);

    @memcpy(buf[0..c.hold_len], c.hold[0..c.hold_len]);
    @memcpy(buf[c.hold_len..][0..text.len], text);
    const all = buf[0 .. c.hold_len + text.len];
    const n = completePrefixLen(all);

    const o = c.out.o;
    try o.print("{s}{s}{s}", .{ shadeColor(c.out.po, shade), all[0..n], c.out.po.reset });
    try o.flush();

    c.hold_len = all.len - n;
    @memcpy(c.hold[0..c.hold_len], all[n..]);
}

fn endTurn(c: *Chat) !void {
    const o = c.out.o;
    try o.print(
        "{s}{s}{s}\n",
        .{ shadeColor(c.out.po, c.think.last), c.hold[0..c.hold_len], c.out.po.reset },
    );
    try o.flush();
    c.hold_len = 0;
}

// -- helpers: reconnect -----------------------------------------------------

/// The `reconnecting` state. Each pass waits out the backoff, then runs one
/// attempt task under a deadline, so the loop runs at most
/// `reconnect_attempts_max` passes. Stdin stays live, so `/quit` and the end
/// of stdin end the wait at once.
fn reconnect(c: *Chat, queue: *InputQueue) !Outcome {
    c.state = .reconnecting;
    var last_err: anyerror = error.ConnectionClosed;
    for (0..reconnect_attempts_max) |i| {
        const attempt: u32 = @intCast(i);
        const ms = reconnectDelayMs(attempt);
        try c.out.note(reconnect_fmt, .{ attempt + 1, reconnect_attempts_max, ms });
        if (try waitTimer(c, queue, ms) == .quit) return quitReconnect(c);

        c.gen += 1;
        var task = try c.io.concurrent(attemptTask, .{ AttemptJob{
            .gpa = c.gpa,
            .io = c.io,
            .target = c.target,
            .expect = c.info,
            .next_request_id = c.client.next_request_id,
            .token = c.lease,
            .gen = c.gen,
            .final = i + 1 == reconnect_attempts_max,
        }, queue });

        const wake = try waitAttempt(c, queue);
        var result = if (wake == .attempt) task.await(c.io) else task.cancel(c.io);
        if (wake == .quit) {
            if (result.client) |*fresh| {
                fresh.deinit();
            }

            return quitReconnect(c);
        }

        // A task cut at its deadline reports the cancel, and the attempt
        // failed on time.
        if (wake == .timer and result.outcome == .failed and result.outcome.failed ==
            error.Canceled)
        {
            result.outcome = .{ .failed = error.Timeout };
        }

        switch (result.outcome) {
            .failed => |err| {
                std.debug.assert(result.client == null);

                last_err = err;
                continue;
            },
            .changed => {
                std.debug.assert(result.client == null);

                try c.out.note(server_changed_note, .{});
                return error.ServerChanged;
            },
            .resumed, .no_lease => {},
        }

        return adoptAttempt(c, result);
    }

    try c.out.note(reconnect_failed_fmt, .{ reconnect_attempts_max, last_err });
    return error.ReconnectFailed;
}

/// Install the client of a finished attempt. The attempt is `resumed` or
/// `no_lease`.
fn adoptAttempt(c: *Chat, result: Attempt) !Outcome {
    c.client.deinit();
    c.client.* = result.client.?;

    // The new server can differ in its think mode and card rows, and the
    // rebuild opens its session from them.
    c.info = result.info.?;
    c.sampling = c.opts.sessionSampling(c.info);
    if (c.info.resume_ttl_max_ms == 0 and c.keep_ttl > 0) {
        c.keep_ttl = 0;
        try c.out.note(keep_off_note, .{});
    }

    return switch (result.outcome) {
        .resumed => |r| .{ .resumed = r },
        .no_lease => |le| {
            if (le) |e| {
                try c.out.serverError(e);
            }

            return .no_lease;
        },
        .failed, .changed => unreachable,
    };
}

/// A quit while the client reconnects. No server hears a close now, so the
/// lease waits out its time to live.
fn quitReconnect(c: *Chat) Outcome {
    c.quit_pending = true;
    c.lease = null;
    c.sources_len = 0;
    return .quit;
}

/// Wait `ms` and read stdin meanwhile.
fn waitTimer(c: *Chat, queue: *InputQueue, ms: u32) !Wake {
    c.timer_gen += 1;
    var timer = try c.io.concurrent(timerTask, .{ c.io, queue, ms, c.timer_gen });
    defer timer.cancel(c.io);

    return waitFor(c, queue, c.timer_gen);
}

/// Wait for the attempt task of `c.gen` under its deadlines: one for the
/// dial and the greeting, then one for the resume.
fn waitAttempt(c: *Chat, queue: *InputQueue) !Wake {
    c.timer_gen += 1;
    var timer = try c.io.concurrent(
        timerTask,
        .{ c.io, queue, reconnect_greet_deadline_ms, c.timer_gen },
    );
    const first = waitFor(c, queue, c.timer_gen) catch |err| {
        timer.cancel(c.io);
        return err;
    };
    timer.cancel(c.io);
    if (first != .greeted) return first;

    c.timer_gen += 1;
    var resume_timer = try c.io.concurrent(
        timerTask,
        .{ c.io, queue, reconnect_resume_deadline_ms, c.timer_gen },
    );
    defer resume_timer.cancel(c.io);

    const second = try waitFor(c, queue, c.timer_gen);
    // The task sends one greeting marker per attempt.
    std.debug.assert(second != .greeted);
    return second;
}

/// Each pass takes one input. The loop ends at the timer `timer_id`, a
/// marker of the attempt `c.gen`, a quit, or the end of stdin. No socket
/// task runs in this state, so no frame arrives.
fn waitFor(c: *Chat, queue: *InputQueue, timer_id: u32) !Wake {
    while (true) {
        const input = try queue.getOne(c.io);
        switch (input) {
            .timer => |id| if (id == timer_id) return .timer,
            .greeted => |g| if (g == c.gen) return .greeted,
            .attempt_done => |g| if (g == c.gen) return .attempt,
            .stdin_closed => return .quit,
            .line => |*l| {
                if (parseLine(l.buf[0..l.len]) == .quit) return .quit;
                try c.out.note(reconnecting_line_note, .{});
            },
            .line_too_long => try c.out.note(too_long_fmt, .{line_bytes_max - 1}),
            .frame, .socket_closed => unreachable,
        }
    }
}

fn timerTask(io: Io, queue: *InputQueue, ms: u32, id: u32) void {
    io.sleep(.fromMilliseconds(ms), .awake) catch return;
    queue.putOne(io, .{ .timer = id }) catch {};
}

/// One dial, greeting, and resume on a new connection. A cancel at the
/// deadline ends a blocked read with an error, and the errdefer closes the
/// connection. A resume that fails on the carrier keeps the token, because
/// the server can still hold it.
fn attemptTask(job: AttemptJob, queue: *InputQueue) Attempt {
    defer queue.putOne(job.io, .{ .attempt_done = job.gen }) catch {};

    return attemptRun(job, queue) catch |err| .{ .outcome = .{ .failed = err } };
}

fn attemptRun(job: AttemptJob, queue: *InputQueue) !Attempt {
    var fresh = switch (job.target) {
        .unix => |path| try Client.connectUnix(job.gpa, job.io, path),
        .tcp => |address| try Client.connectTcp(job.gpa, job.io, address),
    };

    errdefer fresh.deinit();

    // Request ids keep growing across connections, so `reconcile` never
    // takes an id of the old connection for a new request.
    fresh.next_request_id = job.next_request_id;
    const info = try fresh.hello();
    if (!sameServer(&job.expect, &info)) {
        fresh.deinit();
        return .{ .outcome = .changed };
    }

    try queue.putOne(job.io, .{ .greeted = job.gen });

    const token = job.token orelse return .{
        .client = fresh,
        .info = info,
        .outcome = .{ .no_lease = null },
    };
    const r = fresh.resumeSession(token) catch |err| {
        if (err != error.ServerError) return err;

        const refusal = fresh.last_error.?;
        // A retryable refusal leaves the token valid, because the server
        // rotates it only after a successful resume.
        if (classifyResumeRefusal(refusal.code) == .retry and !job.final) return error.ServerError;

        return .{ .client = fresh, .info = info, .outcome = .{ .no_lease = refusal } };
    };

    return .{ .client = fresh, .info = info, .outcome = .{ .resumed = r } };
}

/// Sort the refusal of a `session_resume`. A new `ErrorCode` fails to compile
/// here until someone classifies it.
fn classifyResumeRefusal(code: protocol.ErrorCode) ResumeRefusal {
    return switch (code) {
        .unknown_session, .forbidden => .no_lease,
        .malformed_frame,
        .unsupported_version,
        .unknown_msg_type,
        .frame_too_large,
        .sessions_exhausted,
        .session_busy,
        .turn_too_long,
        .connections_exhausted,
        .internal,
        .cancelled,
        => .retry,
    };
}

/// The transcript costs and the token ids hold only on the same model with
/// the same window.
fn sameServer(old: *const Client.HelloInfo, new: *const Client.HelloInfo) bool {
    return old.max_seq == new.max_seq and old.eos_id == new.eos_id and
        std.mem.eql(u8, old.modelName(), new.modelName());
}

/// The step out of `reconnecting`. A resume continues the state before the
/// drop. No lease rebuilds the session.
fn afterReconnect(c: *Chat, prior: State, outcome: Outcome) !void {
    std.debug.assert(c.state == .reconnecting);
    std.debug.assert(prior != .reconnecting);
    std.debug.assert(prior != .quitting);

    // The replies due on the old connection never arrive. The clone sources
    // held no lease, so the drop freed them.
    c.replies_due = 0;
    c.ping = null;
    c.sources_len = 0;

    switch (outcome) {
        .resumed => |r| return resumeChat(c, prior, r),
        .no_lease => return rebuild(c, prior),
        .quit => unreachable,
    }
}

/// A lease exists only on an open session with no cycle in flight, so a
/// resume reaches only the states that hold one.
fn resumeChat(c: *Chat, prior: State, r: Client.ResumedSession) !void {
    std.debug.assert(c.session != null);
    std.debug.assert(c.cycle == null);

    const before = c.session.?.kv_pos;
    c.session = .{ .id = r.id, .kv_pos = r.state.kv_pos };
    c.lease = r.state.token;
    try c.out.dim(resumed_fmt, .{ r.id, r.state.kv_pos, r.state.turns_done });
    switch (prior) {
        .idle => return enterIdle(c),
        .streaming => return reconcileTurn(c, before, r.state),
        // Only a clone waits in `opening` with a lease, on its source.
        .opening => {
            try c.out.note(request_lost_note, .{});
            return enterIdle(c);
        },
        .appending, .keeping, .reconnecting, .quitting => unreachable,
    }
}

/// The turn in flight at the drop takes the verdict of `reconcile`. Only a
/// turn that the server never ran goes out again.
fn reconcileTurn(c: *Chat, before: u32, s: protocol.Resumed) !void {
    try endTurn(c);

    const verdict = transcript_mod.reconcile(
        .{ .request_id = c.pending_id, .received = c.received },
        s.turns_done,
        s.last,
    );
    switch (verdict) {
        .never_ran => {
            try c.out.abandoned(c.hist.t.abandon());
            try c.out.note(never_ran_note, .{});
            return sendPrompt(c, c.retried);
        },
        .cut => |k| try c.out.note(cut_reply_fmt, .{ k, if (k == 1) "token" else "tokens" }),
        .inconsistent => try c.out.note(inconsistent_fmt, .{c.received}),
        .complete => {},
    }

    // An inconsistent summary names no usable finish, so the turn counts as
    // cut at the frames the client got.
    const finish: protocol.FinishReason = if (verdict == .inconsistent)
        .cancelled
    else
        s.last.?.finish;

    const gen_tokens = if (verdict == .inconsistent) c.received else s.last.?.gen_tokens;
    const rec = c.hist.t.finish(transcript_mod.liveCost(s.kv_pos -| before, finish, gen_tokens));
    try c.out.recorded(rec);

    if (finish == .context_full) return startCompaction(c, c.hist.effectiveReserve(), false);

    return enterIdle(c);
}

/// Open a new session and append the kept pairs, as a compaction does. The
/// new session lacks the turn in flight, so that turn goes out again.
fn rebuild(c: *Chat, prior: State) !void {
    try c.out.note(no_lease_note, .{});
    c.lease = null;

    var retry = c.retry_waiting;
    if (prior == .streaming) {
        try endTurn(c);
        try c.out.abandoned(c.hist.t.abandon());
        try c.out.note(rebuild_resend_note, .{});
        retry = true;
    }

    c.session = null;
    c.cycle = null;
    const reserve = c.hist.effectiveReserve();
    const total = c.hist.t.count();
    c.cycle = .{ .plan = c.hist.shrink(reserve), .total = total, .reserve = reserve };
    c.retry_waiting = retry;
    return reopen(c);
}

/// The wait before reconnect attempt `attempt`, counted from 0. It starts at
/// the first delay and doubles up to the cap.
fn reconnectDelayMs(attempt: u32) u32 {
    std.debug.assert(attempt < reconnect_attempts_max);

    const ms = @min(
        @as(u64, reconnect_delay_first_ms) << @intCast(attempt),
        reconnect_delay_max_ms,
    );
    std.debug.assert(ms >= reconnect_delay_first_ms);
    return @intCast(ms);
}

// -- helpers: blocking loop -------------------------------------------------

fn runBlocking(ctx: cli.Context, st: Startup) !void {
    const out = st.out;
    const client = st.client;
    const ids = try ctx.arena.alloc(u32, st.info.max_seq);
    const text = try ctx.arena.alloc(u8, text_bytes_max);
    var shades: ReplyShades = try .init(ctx.arena, text.len, st.info.eos_id);

    var buf: [line_bytes_max]u8 = undefined;
    var reader = Io.File.stdin().readerStreaming(ctx.io, &buf);
    const r = &reader.interface;

    var b: Blocking = .{
        .out = out,
        .opts = st.opts,
        .client = client,
        .info = st.info,
        .sampling = st.sampling,
        .hist = st.hist,
        .session = st.session,
        .ids = ids,
        .text = text,
        .shades = &shades,
    };

    // Each pass reads one line. The loop ends at `/quit`, EOF, or a read error.
    while (true) {
        try out.marker();
        const maybe = r.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                try out.note(too_long_fmt, .{line_bytes_max - 1});
                _ = r.discardDelimiterInclusive('\n') catch break;
                continue;
            },
            error.ReadFailed => break,
        };
        const line = maybe orelse break;
        const cmd = parseLine(line);
        if (cmd == .quit) break;

        blockingStep(&b, cmd) catch |err| {
            if (err != error.ServerError) return err;
            try out.serverError(client.last_error);
        };
    }

    // The marker of the last pass stays open, so the quit ends its line.
    try out.e.writeAll("\n");
    try out.e.flush();

    client.closeSession(b.session.id) catch |err| {
        if (err != error.ServerError) return err;
        try out.serverError(client.last_error);
    };
}

fn blockingStep(b: *Blocking, cmd: Command) !void {
    const out = b.out;
    const client = b.client;

    switch (cmd) {
        .empty, .quit => {},
        .help => try printCommands(out.e),
        .unknown => |word| try out.note("unknown command: /{s}", .{word}),
        .cancel => try out.note("no streaming turn to cancel", .{}),
        .stats => try out.stats(try client.stats()),
        .history => try out.history(b.hist),
        .ping => {
            const t0 = Io.Clock.Timestamp.now(client.io, .awake);
            try client.ping();
            try out.pong(time.msBetween(t0, Io.Clock.Timestamp.now(client.io, .awake)));
        },
        .close => {
            b.hist.t.clear();
            try client.closeSession(b.session.id);
            b.session = client.openSessionInfo(b.opts.system, b.sampling) catch |err|
                return sessionLost(out, client, err);
            try out.dim(session_open_fmt, .{b.session.id});
        },
        .clone => {
            b.session = try client.cloneSessionInfo(b.session.id, b.sampling);
            try out.dim(session_open_fmt, .{b.session.id});
        },
        .compact => {
            if (b.hist.t.count() == 0) return out.note(compact_empty_note, .{});
            try blockingCompact(b, b.hist.effectiveReserve());
        },
        .prompt => |p| try blockingTurn(b, p),
    }
}

/// One typed prompt. A refusal that a compaction can answer runs the prompt
/// once more, so the loop runs at most two passes.
fn blockingTurn(b: *Blocking, prompt_text: []const u8) !void {
    const t = b.hist.t;
    var retried = false;

    while (true) {
        t.begin(prompt_text);
        _ = try b.client.generateWith(
            b.session.id,
            prompt_text,
            b.opts.turn(turnBudget(b.opts.max_tokens, b.info.max_seq), b.sampling),
        );
        b.shades.reset();

        const turn = b.client.collectTurnWith(b.ids, b.text, b.shades) catch |err| {
            const evicted = t.abandon();
            if (err != error.ServerError) return err;

            try b.out.abandoned(evicted);

            const le = b.client.last_error;
            const refusal = if (retried) null else tooLong(le, .turn);
            const tl = refusal orelse return b.out.turnRefused(le, b.session, b.info.max_seq);
            switch (b.hist.retry(tl.tokens)) {
                .refuse => return b.out.refused(le, b.session, b.hist, tl.tokens),
                .retry => |r| {
                    try b.out.note(retry_fmt, .{ tl.tokens, tl.room });
                    try blockingCompact(b, r);
                    retried = true;
                    continue;
                },
            }
        };
        return blockingRecord(b, turn);
    }
}

/// A turn that filled the window compacts at once, so the next prompt finds
/// room.
fn blockingRecord(b: *Blocking, turn: Client.Turn) !void {
    const out = b.out;
    const t = b.hist.t;
    std.debug.assert(b.shades.total == turn.text_total);

    const before = b.session.kv_pos;
    std.debug.assert(turn.kv_pos >= before);
    b.session.kv_pos = turn.kv_pos;

    const kept = b.text[0..turn.text_len];
    try b.shades.write(out.o, out.po, kept);
    try out.o.writeAll("\n");
    try out.o.flush();

    const cuts = turnCuts(turn, b.ids.len);
    if (cuts.ids) |cut| {
        try out.note("{f}", .{cut});
    }

    if (cuts.text) |cut| {
        try out.note("{f}", .{cut});
    }

    b.shades.record(t, kept);
    // A cut reply is not whole, so the transcript must not keep it as one.
    if (cuts.text != null) {
        t.markCut();
    }

    const rec = t.finish(
        transcript_mod.liveCost(turn.kv_pos - before, turn.finish_reason, turn.gen_tokens),
    );
    try out.finish(.{
        .finish_reason = turn.finish_reason,
        .gen_tokens = turn.gen_tokens,
        .reasoning_tokens = turn.reasoning_tokens,
        .kv_pos = turn.kv_pos,
        .max_seq = b.info.max_seq,
        .prefill_ms = turn.prefill_ms,
        .decode_ms = turn.decode_ms,
    });
    try out.recorded(rec);

    if (turn.finish_reason == .context_full) {
        try blockingCompact(b, b.hist.effectiveReserve());
    }
}

/// One compaction cycle in the blocking mode. Each pass opens one session, so
/// the loop runs at most `total + 3` passes.
fn blockingCompact(b: *Blocking, reserve: u32) !void {
    std.debug.assert(reserve <= b.hist.window);

    const out = b.out;
    const client = b.client;
    const total = b.hist.t.count();
    var cycle: Cycle = .{ .plan = b.hist.shrink(reserve), .total = total, .reserve = reserve };

    while (true) {
        try client.closeSession(b.session.id);
        b.session = client.openSessionInfo(b.opts.system, b.sampling) catch |err|
            return sessionLost(out, client, err);
        b.hist.system_rows = b.session.kv_pos;

        const failure = try blockingAppend(b) orelse
            return out.dim("{f}", .{Compacted.of(&cycle, b.session.kv_pos, b.hist.window)});

        if (failure.too_long) |tl| {
            try out.note(refit_fmt, .{ tl.tokens, tl.room });
            cycle.refits += 1;
            switch (transcript_mod.refit(
                b.hist.t,
                b.hist.budget(reserve),
                failure.from,
                failure.to,
                tl.tokens,
                cycle.plan.kept,
            )) {
                .plan => |p| cycle.plan = p,
                .stuck => {
                    try out.dim(refit_stuck_note, .{});
                    b.hist.keepNothing(&cycle);
                },
            }

            continue;
        }

        try out.serverError(client.last_error);
        if (!cycle.reopened) {
            cycle.reopened = true;
            try out.note(reopen_note, .{});
            continue;
        }

        try out.note(append_lost_note, .{});
        b.hist.keepNothing(&cycle);
    }
}

/// Append every kept pair in frames. Each pass sends one frame of at least
/// one pair, so the loop runs at most `count` times. Null means every pair
/// is in.
fn blockingAppend(b: *Blocking) !?AppendFailure {
    const t = b.hist.t;
    var next: u32 = 0;

    while (next < t.count()) {
        var buf: [limits.max_append_pairs]protocol.Pair = undefined;
        const f = b.hist.frame(next, &buf);
        const a = b.client.append(b.session.id, f.pairs) catch |err| {
            if (err != error.ServerError) return err;
            return .{ .from = next, .to = f.end, .too_long = tooLong(b.client.last_error, .turn) };
        };
        b.session.kv_pos = transcript_mod.learnAppended(t, next, b.session.kv_pos, a.positions());
        next = f.end;
    }

    return null;
}

/// The report of an open that failed with no session left. The quit follows.
fn sessionLost(out: Out, client: *const Client, err: anyerror) anyerror {
    if (err == error.ServerError) {
        out.serverError(client.last_error) catch |e| return e;
    }

    out.note(no_session_note, .{}) catch |e| return e;
    return error.SessionLost;
}

// -- helpers: text ----------------------------------------------------------

fn shadeColor(p: style.Palette, shade: Shade) []const u8 {
    return switch (shade) {
        .reply => p.green,
        .think, .tag => p.dim,
    };
}

/// The numbers of a `turn_too_long` refusal of one kind. Any other error, or
/// a detail the client cut, gives null.
fn tooLong(le: ?Client.LastError, kind: transcript_mod.TooLong.Kind) ?transcript_mod.TooLong {
    const e = le orelse return null;
    if (e.code != .turn_too_long or e.truncated) return null;

    const tl = transcript_mod.parseTooLong(e.detail()) orelse return null;
    return if (tl.kind == kind) tl else null;
}

/// The bytes of a prompt that a history line shows. The cut ends on a
/// codepoint boundary.
fn previewLen(prompt_text: []const u8) usize {
    if (prompt_text.len <= preview_bytes) return prompt_text.len;

    const n = completePrefixLen(prompt_text[0..preview_bytes]);
    std.debug.assert(n <= preview_bytes);
    return n;
}

/// The token cap that each `generate` sends. A flag of 0 asks for the whole
/// window. A flag above the window passes through, because the server stops a
/// turn at the window.
fn turnBudget(flag: u32, max_seq: u32) u32 {
    std.debug.assert(max_seq > 0);

    const budget = if (flag == 0) max_seq else flag;
    std.debug.assert(budget > 0);
    return budget;
}

/// The line after the summary, when the finish reason asks the user to act.
fn finishNote(reason: protocol.FinishReason) ?[]const u8 {
    return switch (reason) {
        .context_full => context_full_note,
        .eos, .max_tokens, .cancelled => null,
    };
}

/// `collectTurn` fills each buffer as far as it fits and reports the true
/// totals, so a total above the kept count is a cut.
fn turnCuts(turn: Client.Turn, ids_cap: usize) Cuts {
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

/// The command a typed line names. Only a trailing `\r` is trimmed.
fn parseLine(raw: []const u8) Command {
    const line = if (std.mem.endsWith(u8, raw, "\r")) raw[0 .. raw.len - 1] else raw;
    if (line.len == 0) return .empty;
    if (line[0] != '/') return .{ .prompt = line };

    const word = line[1..];
    const table = [_]struct { []const u8, Command }{
        .{ "cancel", .cancel }, .{ "stats", .stats },     .{ "ping", .ping },
        .{ "close", .close },   .{ "clone", .clone },     .{ "quit", .quit },
        .{ "help", .help },     .{ "compact", .compact }, .{ "history", .history },
    };
    for (table) |entry| {
        if (std.mem.eql(u8, word, entry[0])) return entry[1];
    }

    return .{ .unknown = word };
}

/// The length of the longest prefix of `bytes` that ends on a UTF-8 codepoint
/// boundary. An invalid lead byte counts as complete, so the output never
/// stalls on bad bytes.
fn completePrefixLen(bytes: []const u8) usize {
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

fn printCommands(e: *Io.Writer) !void {
    try e.writeAll(commands_text);
    try e.flush();
}

// -- tests ------------------------------------------------------------------

test "parseLine maps each command word" {
    try std.testing.expectEqual(Command.cancel, parseLine("/cancel"));
    try std.testing.expectEqual(Command.stats, parseLine("/stats"));
    try std.testing.expectEqual(Command.ping, parseLine("/ping"));
    try std.testing.expectEqual(Command.close, parseLine("/close"));
    try std.testing.expectEqual(Command.clone, parseLine("/clone"));
    try std.testing.expectEqual(Command.quit, parseLine("/quit"));
    try std.testing.expectEqual(Command.help, parseLine("/help"));
    try std.testing.expectEqual(Command.compact, parseLine("/compact"));
    try std.testing.expectEqual(Command.history, parseLine("/history"));
}

test "parseLine keeps an unknown word and a bare prompt" {
    try std.testing.expectEqualStrings("x", parseLine("/x").unknown);
    try std.testing.expectEqualStrings("hello there", parseLine("hello there").prompt);
    try std.testing.expectEqualStrings(" padded ", parseLine(" padded ").prompt);
}

test "parseLine trims only a trailing carriage return" {
    try std.testing.expectEqual(Command.empty, parseLine(""));
    try std.testing.expectEqual(Command.empty, parseLine("\r"));
    try std.testing.expectEqual(Command.quit, parseLine("/quit\r"));
    try std.testing.expectEqualStrings("hi", parseLine("hi\r").prompt);
}

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

test "turnBudget gives the whole window for 0 and passes a flag through" {
    try std.testing.expectEqual(@as(u32, 512), turnBudget(0, 512));
    try std.testing.expectEqual(@as(u32, 1), turnBudget(0, 1));
    try std.testing.expectEqual(@as(u32, 1), turnBudget(1, 512));
    try std.testing.expectEqual(@as(u32, 4096), turnBudget(4096, 512));
}

test "the summary line and its note for each finish reason" {
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
    const cases = [_]Case{
        .{ .s = base, .line = "finish_reason=eos gen_tokens=40 reasoning_tokens=0 kv_pos=52/512 prefill_ms=12.5 decode_ms=800.0", .note = null },
        .{ .s = capped, .line = "finish_reason=max_tokens gen_tokens=40 reasoning_tokens=39 kv_pos=52/512 prefill_ms=12.5 decode_ms=800.0", .note = null },
        .{ .s = full, .line = "finish_reason=context_full gen_tokens=460 reasoning_tokens=300 kv_pos=512/512 prefill_ms=12.5 decode_ms=800.0", .note = context_full_note },
        .{ .s = cancelled, .line = "finish_reason=cancelled gen_tokens=40 reasoning_tokens=0 kv_pos=52/512 prefill_ms=12.5 decode_ms=800.0", .note = null },
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
const test_palette: style.Palette = .{ .reset = ")", .dim = "D(", .green = "G(" };

fn expectShades(shades: *const ReplyShades, text: []const u8, want: []const u8) !void {
    var buf: [256]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try shades.write(&w, test_palette, text);
    try std.testing.expectEqualStrings(want, w.buffered());
}

/// One token event of a test turn.
const Piece = struct { text: []const u8, channel: protocol.Channel = .content };

/// Feeds each piece to the hook under its own id, then the end-of-sequence
/// frame, as `collectTurnWith` calls it.
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
    // The euro sign E2 82 AC starts one byte before the bound, so the cut keeps
    // the bytes before it.
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

test "reconnectDelayMs starts at 100 ms, doubles, and stops at 2 s" {
    const want = [reconnect_attempts_max]u32{ 100, 200, 400, 800, 1600, 2000, 2000, 2000 };
    var total: u64 = 0;
    for (want, 0..) |ms, i| {
        try std.testing.expectEqual(ms, reconnectDelayMs(@intCast(i)));
        total += ms;
    }

    // The whole reconnect waits at most this long before it gives up.
    try std.testing.expectEqual(@as(u64, 9100), total);
}

test "classifyResumeRefusal: only a missing or forbidden lease rebuilds at once" {
    var no_lease: u32 = 0;
    inline for (@typeInfo(protocol.ErrorCode).@"enum".fields) |f| {
        const code: protocol.ErrorCode = @enumFromInt(f.value);
        const want: ResumeRefusal = switch (code) {
            .unknown_session, .forbidden => .no_lease,
            else => .retry,
        };

        try std.testing.expectEqual(want, classifyResumeRefusal(code));
        if (want == .no_lease) {
            no_lease += 1;
        }
    }

    try std.testing.expectEqual(@as(u32, 2), no_lease);
}

test "Options.sessionSampling follows a new hello_ack, and an explicit flag keeps its value" {
    var old = std.mem.zeroes(Client.HelloInfo);
    old.think_mode = .off;
    old.recommended_plain = protocol.Sampling.greedy;
    old.recommended_plain.temperature = 0.7;
    old.recommended_think = protocol.Sampling.greedy;
    old.recommended_think.temperature = 1.0;
    var new = old;
    new.think_mode = .on;
    new.recommended_think.temperature = 0.6;

    const o: Options = .{ .sampling = .model };
    try std.testing.expect(!o.turnsOpen(old));
    try std.testing.expect(o.turnsOpen(new));
    try std.testing.expectEqual(@as(f32, 0.7), o.sessionSampling(old).temperature);
    try std.testing.expectEqual(@as(f32, 0.6), o.sessionSampling(new).temperature);

    const pinned: Options = .{ .sampling = .model, .thinking = .off, .temperature = 0.3 };
    try std.testing.expect(!pinned.turnsOpen(new));
    try std.testing.expectEqual(@as(f32, 0.3), pinned.sessionSampling(new).temperature);
}
