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
const Io = std.Io;

const context = @import("../context.zig");
const style = @import("../style.zig");

const client_blocking = @import("client_blocking.zig");
const client_history = @import("client_history.zig");
const client_output = @import("client_output.zig");
const client_startup = @import("client_startup.zig");
const client_streaming = @import("client_streaming.zig");

// -- constants --------------------------------------------------------------

/// The stderr writer buffer.
const stderr_buffer_bytes = 1024;

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const opts = try client_startup.parseOptions(ctx) orelse return;

    var err_buf: [stderr_buffer_bytes]u8 = undefined;
    var fe: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &err_buf);
    const out: client_output.Out = .{
        .o = &ctx.stdout.interface,
        .e = &fe.interface,
        .po = style.Palette.init(ctx.color),
        .pe = style.Palette.init(ctx.color_err),
    };

    const target = try client_startup.parseTarget(ctx, &opts);
    var client = try client_startup.connect(ctx, target, &opts, out);
    defer client.deinit();

    const info = client.hello() catch |err| return client_startup.startupFailed(out, &client, err);
    try client_startup.checkServer(out, &opts, info);

    const sampling = opts.sessionSampling(info);
    if (opts.sampling == .model) {
        try client_startup.printSampling(out, &opts, info, sampling);
    }

    const session = client.openSessionInfo(opts.system, sampling) catch |err|
        return client_startup.startupFailed(out, &client, err);
    const hist = try ctx.arena.create(client_history.History);
    hist.* = try .init(ctx.arena, session.kv_pos, info.max_seq, opts.reserve);
    try client_startup.printStartup(out, &opts, info, session.id);

    const keep: client_startup.Keep = if (opts.resume_ttl > 0)
        try client_startup.startKeep(out, &client, info, session.id, opts.resume_ttl)
    else
        .{ .ttl_ms = 0, .token = null };

    const st: client_startup.Startup = .{
        .out = out,
        .opts = &opts,
        .client = &client,
        .info = info,
        .sampling = sampling,
        .session = session,
        .hist = hist,
    };
    if (!opts.no_stream) return client_streaming.runStreaming(ctx, st, target, keep);

    std.debug.assert(keep.ttl_ms == 0);
    return client_blocking.runBlocking(ctx, st);
}
