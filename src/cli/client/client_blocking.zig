//! The `--no-stream` mode: one blocking loop with no tasks, for piped input.
//!
//! Errors. A server error ends one step, and the loop prints it and reads the next line. Any other
//! error ends the chat.

const std = @import("std");
const Io = std.Io;

const context = @import("../context.zig");
const time = @import("../../core/utils/time.zig");
const limits = @import("../../core/limits.zig");
const protocol = @import("../../serve/protocol.zig");
const client_mod = @import("../../serve/client.zig");
const transcript_mod = @import("client_transcript.zig");

const Client = client_mod.Client;

const client_history = @import("client_history.zig");
const client_input = @import("client_input.zig");
const client_output = @import("client_output.zig");
const client_startup = @import("client_startup.zig");

// -- constants --------------------------------------------------------------

const text_bytes_max = 64 * 1024;

// -- types ------------------------------------------------------------------

/// The state of the blocking loop.
const Blocking = struct {
    out: client_output.Out,
    opts: *const client_startup.Options,
    client: *Client,
    info: Client.HelloInfo,
    sampling: protocol.Sampling,
    hist: *client_history.History,
    session: Client.Opened,
    ids: []u32,
    text: []u8,
    shades: *client_output.ReplyShades,
};

/// A failed append frame of turns `from..to`.
const AppendFailure = struct { from: u32, to: u32 };

// -- api --------------------------------------------------------------------

pub fn runBlocking(ctx: context.Context, st: client_startup.Startup) !void {
    const out = st.out;
    const client = st.client;

    // Allocate the reply buffers once, for every turn of the chat.
    const ids = try ctx.arena.alloc(u32, st.info.max_seq);
    const text = try ctx.arena.alloc(u8, text_bytes_max);
    var shades: client_output.ReplyShades = try .init(ctx.arena, text.len, st.info.eos_id);

    // Read stdin through a buffer of one line.
    var buf: [client_input.line_bytes_max]u8 = undefined;
    var reader = Io.File.stdin().readerStreaming(ctx.io, &buf);
    const r = &reader.interface;

    // Gather the values that each step reads.
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
                try out.note(client_input.too_long_fmt, .{client_input.line_bytes_max - 1});
                _ = r.discardDelimiterInclusive('\n') catch break;
                continue;
            },
            error.ReadFailed => break,
        };
        const line = maybe orelse break;
        const cmd = client_input.parseLine(line);
        if (cmd == .quit) break;

        blockingStep(&b, cmd) catch |err| {
            if (err != error.ServerError) return err;
            try out.serverError(client.last_error);
        };
    }

    // The marker of the last pass stays open, so the quit ends its line.
    try out.e.writeAll("\n");
    try out.e.flush();

    // Close the session. A server error here does not fail the chat.
    client.closeSession(b.session.id) catch |err| {
        if (err != error.ServerError) return err;
        try out.serverError(client.last_error);
    };
}

// -- helpers: blocking loop -------------------------------------------------

fn blockingStep(b: *Blocking, cmd: client_input.Command) !void {
    const out = b.out;
    const client = b.client;

    switch (cmd) {
        .empty, .quit => {},
        .help => try client_input.printCommands(out.e),
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
            try out.dim(client_output.session_open_fmt, .{b.session.id});
        },
        .clone => {
            b.session = try client.cloneSessionInfo(b.session.id, b.sampling);
            try out.dim(client_output.session_open_fmt, .{b.session.id});
        },
        .compact => {
            if (b.hist.t.count() == 0) return out.note(client_output.compact_empty_note, .{});
            try blockingCompact(b, b.hist.effectiveReserve());
        },
        .prompt => |p| try blockingTurn(b, p),
    }
}

/// One typed prompt. A refusal that a compaction can answer runs the prompt once more, so the
/// loop runs at most two passes.
fn blockingTurn(b: *Blocking, prompt_text: []const u8) !void {
    const t = b.hist.t;
    var retried = false;

    while (true) {
        t.begin(prompt_text);
        _ = try b.client.generateWith(
            b.session.id,
            prompt_text,
            b.opts.turn(client_history.turnBudget(b.opts.max_tokens, b.info.max_seq), b.sampling),
        );
        b.shades.reset();

        const turn = b.client.collectTurnWith(b.ids, b.text, b.shades) catch |err| {
            const evicted = t.abandon();
            if (err != error.ServerError) return err;

            try b.out.abandoned(evicted);

            const le = b.client.last_error;
            const refusal = if (retried) null else client_history.tooLong(le, .turn);
            const tl = refusal orelse return b.out.turnRefused(le, b.session, b.info.max_seq);
            switch (b.hist.retry(tl.tokens)) {
                .refuse => return b.out.refused(le, b.session, b.hist, tl.tokens),
                .retry => |r| {
                    try b.out.note(client_output.retry_fmt, .{ tl.tokens, tl.room });
                    try blockingCompact(b, r);
                    retried = true;
                    continue;
                },
            }
        };
        return blockingRecord(b, turn);
    }
}

/// A turn that filled the window compacts at once, so the next prompt finds room.
fn blockingRecord(b: *Blocking, turn: Client.Turn) !void {
    const out = b.out;
    const t = b.hist.t;
    std.debug.assert(b.shades.total == turn.text_total);

    // Move the session to the position of the server.
    const before = b.session.kv_pos;
    std.debug.assert(turn.kv_pos >= before);
    b.session.kv_pos = turn.kv_pos;

    // Print the kept text with its shades.
    const kept = b.text[0..turn.text_len];
    try b.shades.write(out.o, out.po, kept);
    try out.o.writeAll("\n");
    try out.o.flush();

    // Report each buffer that cut the reply.
    const cuts = client_output.turnCuts(turn, b.ids.len);
    if (cuts.ids) |cut| {
        try out.note("{f}", .{cut});
    }

    if (cuts.text) |cut| {
        try out.note("{f}", .{cut});
    }

    // A cut reply is not whole, so the transcript must not keep it as one.
    b.shades.record(t, kept);
    if (cuts.text != null) {
        t.markCut();
    }

    // Commit the turn and print its summary.
    const rec = t.finish(
        transcript_mod.liveCost(turn.kv_pos - before, turn.finish_reason, turn.gen_tokens),
    );
    try out.finish(turn, b.info.max_seq);
    try out.recorded(rec);

    if (turn.finish_reason == .context_full) {
        try blockingCompact(b, b.hist.effectiveReserve());
    }
}

/// One compaction cycle in the blocking mode. Each pass opens one session, so the loop runs at
/// most `total + 3` passes.
fn blockingCompact(b: *Blocking, reserve: u32) !void {
    std.debug.assert(reserve <= b.hist.window);

    const out = b.out;
    const client = b.client;

    // Plan the cycle. The turns that it drops leave the transcript at once.
    const total = b.hist.t.count();
    var cycle: client_history.Cycle = .{
        .plan = b.hist.shrink(reserve),
        .total = total,
        .reserve = reserve,
    };

    while (true) {
        // Replace the session with one that holds only the system prompt.
        try client.closeSession(b.session.id);
        b.session = client.openSessionInfo(b.opts.system, b.sampling) catch |err|
            return sessionLost(out, client, err);
        b.hist.system_rows = b.session.kv_pos;

        // Send the kept pairs. When every pair is in, the cycle ends.
        const failure = try blockingAppend(b) orelse
            return out.dim(
                "{f}",
                .{client_output.Compacted.of(&cycle, b.session.kv_pos, b.hist.window)},
            );

        // Step the cycle. The next pass opens a new session.
        try out.frameFailed(b.hist, &cycle, client.last_error, failure.from, failure.to);
    }
}

/// Append every kept pair in frames. Each pass sends at least one pair, so the loop runs at most
/// `count` times. Null means every pair is in.
fn blockingAppend(b: *Blocking) !?AppendFailure {
    const t = b.hist.t;
    var next: u32 = 0;

    while (next < t.count()) {
        var buf: [limits.max_append_pairs]protocol.Pair = undefined;
        const f = b.hist.frame(next, &buf);
        const a = b.client.append(b.session.id, f.pairs) catch |err| {
            if (err != error.ServerError) return err;
            return .{ .from = next, .to = f.end };
        };
        b.session.kv_pos = transcript_mod.learnAppended(t, next, b.session.kv_pos, a.positions());
        next = f.end;
    }

    return null;
}

/// The report of an open that failed with no session left. The quit follows.
fn sessionLost(out: client_output.Out, client: *const Client, err: anyerror) anyerror {
    if (err == error.ServerError) {
        out.serverError(client.last_error) catch |e| return e;
    }

    out.note(client_output.no_session_note, .{}) catch |e| return e;
    return error.SessionLost;
}
