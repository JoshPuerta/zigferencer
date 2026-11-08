//! The streaming mode: the main task takes one input at a time and steps `Chat`, so a `/cancel`
//! typed while a turn streams reaches the server at once. This file holds the line, frame, and
//! compaction transitions, and the steps after a reconnect.
//!
//! Teardown. The queue closes first, so a producer blocked on a full queue returns. The hang-up
//! then ends the socket task, and the cancel ends a stdin read.
//!
//! Leases. With resumption on, the client keeps each new session before the next turn, so a drop at
//! any later point finds a lease. A lease exists only on an open session with no cycle in flight,
//! so a resume reaches only the states that hold one.
//!
//! Drops. The replies due on the old connection never arrive. A resume continues the state before
//! the drop, and only a turn that the server never ran goes out again. With no lease, the client
//! opens a new session and appends the kept pairs, as a compaction does. That session lacks the
//! turn in flight, so that turn goes out again.

const std = @import("std");
const Io = std.Io;

const span = @import("../../core/span.zig");
const context = @import("../context.zig");
const time = @import("../../core/time.zig");
const limits = @import("../../core/limits.zig");
const protocol = @import("../../serve/protocol.zig");
const client_mod = @import("../../serve/client.zig");
const transcript_mod = @import("client_transcript.zig");

const Client = client_mod.Client;
const Body = client_mod.OwnedEvent.Body;

const client_chat = @import("client_chat.zig");
const client_history = @import("client_history.zig");
const client_input = @import("client_input.zig");
const client_output = @import("client_output.zig");
const client_reconnect = @import("client_reconnect.zig");
const client_startup = @import("client_startup.zig");

const Chat = client_chat.Chat;

// -- constants --------------------------------------------------------------

const queue_len = 32;

const resumed_fmt = "Session {d} resumes at position {d} after {d} turns.";
const no_lease_note = "The server holds no lease for the session. " ++
    "The client rebuilds it from the transcript.";
const rebuild_resend_note = "The new session does not hold the turn in flight. " ++
    "The client sends it again.";
const never_ran_note = "The server did not run the turn in flight. The client sends it again.";
const cut_reply_fmt = "The drop cut the reply. " ++
    "The server generated {d} {s} that the client did not get. " ++
    "The model context holds the whole reply.";
const inconsistent_fmt = "The resume summary does not match the {d} tokens that the client " ++
    "got. The client keeps its text.";
const request_lost_note = "The request in flight at the drop is lost. Send it again.";

// -- api --------------------------------------------------------------------

pub fn runStreaming(
    ctx: context.Context,
    st: client_startup.Startup,
    target: client_startup.Target,
    keep: client_startup.Keep,
) !void {
    const io = ctx.io;

    // Create the queue that feeds the control unit.
    const buf = try ctx.arena.alloc(client_input.Input, queue_len);
    var queue: client_input.InputQueue = .init(buf);

    // A reconnect awaits the old socket task and starts a new one, so the future is null between
    // the two.
    var socket_future: ?Io.Future(void) = try io.concurrent(socketTask, .{ io, st.client, &queue });
    var stdin_future: ?Io.Future(void) = null;

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

    // Start the stdin task under that teardown.
    stdin_future = try io.concurrent(client_input.stdinTask, .{ io, &queue });

    // Build the control unit in `idle`.
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
            .line_too_long => try st.out.note(
                client_input.too_long_fmt,
                .{client_input.line_bytes_max - 1},
            ),
            .frame => |*ev| try onFrame(&chat, ev),
            .stdin_closed => chat.quit_pending = true,
            .socket_closed => |err| try onDrop(&chat, &queue, &socket_future, err),
            // A marker of a reconnect that already ended.
            .timer, .greeted, .attempt_done => {},
        }

        settleQuit(&chat);
    }

    // Close what stays open, unless the chat lost its session.
    if (chat.lost_session) return error.SessionLost;

    closeAll(&chat);
}

// -- helpers: streaming loop ------------------------------------------------

/// The connection dropped. With resumption off, or during a quit, the chat ends with an error.
/// Else the chat reconnects and starts a new socket task.
fn onDrop(
    c: *Chat,
    queue: *client_input.InputQueue,
    socket_future: *?Io.Future(void),
    err: Client.NextError,
) !void {
    try c.out.note("connection closed: {t}", .{err});
    if (c.keep_ttl == 0 or c.quit_pending) return error.ConnectionClosed;

    // The socket task puts `socket_closed` last and returns, so no frame of the old connection
    // follows it.
    socket_future.*.?.await(c.io);
    socket_future.* = null;

    const prior = c.state;
    const outcome = try client_reconnect.reconnect(c, queue);
    if (outcome == .quit) {
        c.state = .quitting;
        return;
    }

    socket_future.* = try c.io.concurrent(socketTask, .{ c.io, c.client, queue });
    try afterReconnect(c, prior, outcome);
}

/// A quit closes every session on purpose, so none waits out a lease. The closes go out before
/// the hang-up on the same carrier, and the hang-up ends the chat whether they arrive or not.
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

fn socketTask(io: Io, client: *Client, queue: *client_input.InputQueue) void {
    span.nameThread("zf-client");

    // Each pass reads one frame. The loop ends at the first read error, and the hang-up at teardown
    // forces one.
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

    const cmd = client_input.parseLine(line);
    switch (cmd) {
        .empty => {},
        .help => try client_input.printCommands(c.out.e),
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

fn idleLine(c: *Chat, cmd: client_input.Command) !void {
    std.debug.assert(c.state == .idle);

    const sid = c.session.?.id;
    const sampling = c.sampling;
    switch (cmd) {
        .prompt => |text| {
            std.debug.assert(text.len < client_input.line_bytes_max);

            @memcpy(c.prompt.buf[0..text.len], text);
            c.prompt.len = @intCast(text.len);
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
            if (c.hist.t.count() == 0) return c.out.note(client_output.compact_empty_note, .{});
            try startCompaction(c, c.hist.effectiveReserve(), false);
        },
        else => unreachable,
    }
}

fn streamingLine(c: *Chat, cmd: client_input.Command) !void {
    std.debug.assert(c.state == .streaming);

    switch (cmd) {
        .cancel => _ = try c.client.cancel(c.session.?.id),
        else => try c.out.note("turn in progress, /cancel first", .{}),
    }
}

fn sendPrompt(c: *Chat, retried: bool) !void {
    std.debug.assert(c.session != null);
    std.debug.assert(c.prompt.len > 0);

    const text = c.prompt.buf[0..c.prompt.len];
    c.hist.t.begin(text);
    const budget = client_history.turnBudget(c.opts.max_tokens, c.info.max_seq);
    c.pending_id = try c.client.generateWith(c.session.?.id, text, c.opts.turn(budget, c.sampling));

    c.stream = .{};
    c.prompt.retried = retried;
    c.prompt.waiting = false;
    c.state = .streaming;
}

// -- helpers: compaction ----------------------------------------------------

/// Start a compaction cycle. The replies step it in `onFrame` and `onErr`. A rebuild starts one
/// with no session, so `reopen` closes nothing.
fn startCompaction(c: *Chat, reserve: u32, retry: bool) !void {
    std.debug.assert(c.cycle == null);
    std.debug.assert(reserve <= c.hist.window);

    const total = c.hist.t.count();
    c.cycle = .{ .plan = c.hist.shrink(reserve), .total = total, .reserve = reserve };
    c.prompt.waiting = retry;
    return reopen(c);
}

/// Close the open session and open one with the original system prompt. The cycle restarts at its
/// first kept pair.
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

/// Send the next append frame of the cycle, or end the cycle when every kept pair is in.
fn continueCycle(c: *Chat) !void {
    const cy = &c.cycle.?;
    std.debug.assert(cy.next <= c.hist.t.count());

    if (cy.next == c.hist.t.count()) {
        try c.out.dim("{f}", .{client_output.Compacted.of(cy, c.session.?.kv_pos, c.hist.window)});
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

/// The chat leaves a clone source for its fork. The source drops its lease, so a drop frees it,
/// and the quit closes it.
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

/// Ask for a lease on the new session.
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

/// The session is ready, with or without a lease. A prompt that waits for a compaction goes out.
fn afterKeep(c: *Chat) !void {
    if (c.prompt.waiting) return sendPrompt(c, true);
    return enterIdle(c);
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
        c.stream.received += 1;
    }

    if (t.token_id == c.info.eos_id) return;

    const text = t.text[0..t.len];
    const shade = c.stream.think.step(t.channel);
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

    try c.out.finish(d, c.info.max_seq);
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
        // The original system prompt opens each cycle, so the open measures the system rows
        // exactly.
        c.hist.system_rows = kv_pos;
        return continueCycle(c);
    }

    try c.out.dim(client_output.session_open_fmt, .{session_id});
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
    if (request_id == c.pending_id) {
        switch (c.state) {
            .streaming => return turnFailed(c, e),
            .appending => {
                const cy = &c.cycle.?;
                try c.out.frameFailed(c.hist, cy, e.*, cy.next, cy.frame_end);
                return reopen(c);
            },
            .keeping => {
                // The session lives on with no lease. A drop then rebuilds it.
                try c.out.serverError(e.*);
                return afterKeep(c);
            },
            .opening => return openFailed(c, e),
            .idle, .reconnecting, .quitting => {},
        }
    }

    // The error answers a ping or a due reply.
    const answers_ping = if (c.ping) |pg| pg.id == request_id else false;
    if (answers_ping) {
        c.ping = null;
    } else {
        c.replies_due -|= 1;
    }

    try c.out.serverError(e.*);
}

/// Close a refused turn. A compaction can answer the first refusal of a prompt.
fn turnFailed(c: *Chat, e: *const Client.LastError) !void {
    try endTurn(c);
    try c.out.abandoned(c.hist.t.abandon());

    const refusal = if (c.prompt.retried) null else client_history.tooLong(e.*, .turn);
    if (refusal) |tl| {
        switch (c.hist.retry(tl.tokens)) {
            .retry => |r| {
                try c.out.note(client_output.retry_fmt, .{ tl.tokens, tl.room });
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

/// A failed open ends the cycle. With no session left, it ends the chat.
fn openFailed(c: *Chat, e: *const Client.LastError) !void {
    c.cycle = null;
    try c.out.serverError(e.*);
    if (c.session != null) return enterIdle(c);

    try c.out.note(client_output.no_session_note, .{});
    c.lost_session = true;
    c.state = .quitting;
}

fn emitText(c: *Chat, text: []const u8, shade: client_output.Shade) !void {
    var buf: [client_output.codepoint_tail_max + limits.max_token_text_bytes]u8 = undefined;
    const s = &c.stream;
    std.debug.assert(s.hold_len <= client_output.codepoint_tail_max);
    std.debug.assert(text.len <= limits.max_token_text_bytes);

    @memcpy(buf[0..s.hold_len], s.hold[0..s.hold_len]);
    @memcpy(buf[s.hold_len..][0..text.len], text);
    const all = buf[0 .. s.hold_len + text.len];
    const n = client_output.completePrefixLen(all);

    const o = c.out.o;
    const color = client_output.shadeColor(c.out.po, shade);
    try o.print("{s}{s}{s}", .{ color, all[0..n], c.out.po.reset });
    try o.flush();

    s.hold_len = all.len - n;
    @memcpy(s.hold[0..s.hold_len], all[n..]);
}

fn endTurn(c: *Chat) !void {
    const o = c.out.o;
    const s = &c.stream;
    const color = client_output.shadeColor(c.out.po, s.think.last);
    try o.print("{s}{s}{s}\n", .{ color, s.hold[0..s.hold_len], c.out.po.reset });
    try o.flush();
    s.hold_len = 0;
}

// -- helpers: reconnect -----------------------------------------------------

/// The step out of `reconnecting`.
fn afterReconnect(c: *Chat, prior: client_chat.State, outcome: client_reconnect.Outcome) !void {
    std.debug.assert(c.state == .reconnecting);
    std.debug.assert(prior != .reconnecting);
    std.debug.assert(prior != .quitting);

    // The clone sources held no lease, so the drop freed them.
    c.replies_due = 0;
    c.ping = null;
    c.sources_len = 0;

    switch (outcome) {
        .resumed => |r| return resumeChat(c, prior, r),
        .no_lease => return rebuild(c, prior),
        .quit => unreachable,
    }
}

/// Continue the state before the drop.
fn resumeChat(c: *Chat, prior: client_chat.State, r: Client.ResumedSession) !void {
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

/// The turn in flight at the drop takes the verdict of `reconcile`.
fn reconcileTurn(c: *Chat, before: u32, s: protocol.Resumed) !void {
    try endTurn(c);

    const verdict = transcript_mod.reconcile(
        .{ .request_id = c.pending_id, .received = c.stream.received },
        s.turns_done,
        s.last,
    );
    switch (verdict) {
        .never_ran => {
            try c.out.abandoned(c.hist.t.abandon());
            try c.out.note(never_ran_note, .{});
            return sendPrompt(c, c.prompt.retried);
        },
        .cut => |k| try c.out.note(cut_reply_fmt, .{ k, if (k == 1) "token" else "tokens" }),
        .inconsistent => try c.out.note(inconsistent_fmt, .{c.stream.received}),
        .complete => {},
    }

    // An inconsistent summary names no usable finish, so the turn counts as cut at the frames that
    // the client got.
    const finish: protocol.FinishReason = if (verdict == .inconsistent)
        .cancelled
    else
        s.last.?.finish;

    const gen_tokens = if (verdict == .inconsistent) c.stream.received else s.last.?.gen_tokens;
    const rec = c.hist.t.finish(transcript_mod.liveCost(s.kv_pos -| before, finish, gen_tokens));
    try c.out.recorded(rec);

    if (finish == .context_full) return startCompaction(c, c.hist.effectiveReserve(), false);

    return enterIdle(c);
}

/// Open a new session and append the kept pairs.
fn rebuild(c: *Chat, prior: client_chat.State) !void {
    try c.out.note(no_lease_note, .{});
    c.lease = null;

    var retry = c.prompt.waiting;
    if (prior == .streaming) {
        try endTurn(c);
        try c.out.abandoned(c.hist.t.abandon());
        try c.out.note(rebuild_resend_note, .{});
        retry = true;
    }

    c.session = null;
    c.cycle = null;
    return startCompaction(c, c.hist.effectiveReserve(), retry);
}
