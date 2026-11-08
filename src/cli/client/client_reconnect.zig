//! The `reconnecting` state: the backoff, one attempt task per pass, and the resume. Stdin stays
//! live, so `/quit` and the end of stdin end the wait at once.
//!
//! Backoff. The client dials at most `reconnect_attempts_max` times. Each dial follows a wait that
//! starts at the first delay and doubles up to the cap.
//!
//! Deadlines. A relay that accepts and never forwards answers never, so the greeting deadline ends
//! that attempt. The server answers a resume only after the prefill in flight. A cold prefill of
//! 32704 positions takes about 55 s on the fused GPU runner, so the resume deadline leaves twice
//! that.
//!
//! Attempts. An attempt task works on copies, so it shares no state with the control unit. A cancel
//! at the deadline ends a blocked read with an error, and the errdefer closes the connection. A
//! client that a task returns is connected and greeted, and the control unit owns it.
//!
//! Tokens. A resume that fails on the carrier keeps the token, because the server can still hold
//! it. A retryable refusal also keeps it, because the server rotates it only after a successful
//! resume.

const std = @import("std");
const Io = std.Io;

const protocol = @import("../../serve/protocol.zig");
const client_mod = @import("../../serve/client.zig");

const Client = client_mod.Client;

const client_chat = @import("client_chat.zig");
const client_input = @import("client_input.zig");
const client_startup = @import("client_startup.zig");

const Chat = client_chat.Chat;
const InputQueue = client_input.InputQueue;

// -- constants --------------------------------------------------------------

const reconnect_fmt = "Reconnect attempt {d} of {d} in {d} ms.";
const reconnect_failed_fmt = "The client cannot reconnect after {d} attempts. " ++
    "The last error is {t}.";
const server_changed_note = "The new server has another model, window, or end token. " ++
    "The client ends the chat.";
const reconnecting_line_note = "The client is reconnecting and drops the line. " ++
    "Type it again after the resume.";

const reconnect_attempts_max: u32 = 8;

const reconnect_delay_first_ms: u32 = 100;
const reconnect_delay_max_ms: u32 = 2000;

const reconnect_greet_deadline_ms: u32 = 10_000;

const reconnect_resume_deadline_ms: u32 = 120_000;

// -- types ------------------------------------------------------------------

/// What one reconnect found on the server.
pub const Outcome = union(enum) {
    resumed: Client.ResumedSession,
    /// The client held no lease, or the server refused the resume.
    no_lease,
    /// `/quit` or the end of stdin arrived while the client reconnected.
    quit,
};

/// The values that one attempt task copies.
const AttemptJob = struct {
    gpa: std.mem.Allocator,
    io: Io,
    target: client_startup.Target,
    /// The `hello_ack` of the first server. A new server must match it.
    expect: Client.HelloInfo,
    next_request_id: u32,
    token: ?protocol.Token,
    gen: u32,
    /// The last pass of the loop. No later pass can retry, so a retryable refusal rebuilds.
    final: bool,
};

/// The result of one attempt task.
const Attempt = struct {
    client: ?Client = null,
    info: ?Client.HelloInfo = null,
    outcome: union(enum) {
        resumed: Client.ResumedSession,
        /// The reason that the server refused the resume, or null with no token to send.
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

// -- api --------------------------------------------------------------------

/// The `reconnecting` state. Each pass waits out the backoff, then runs one attempt task under a
/// deadline, so the loop runs at most `reconnect_attempts_max` passes.
pub fn reconnect(c: *Chat, queue: *InputQueue) !Outcome {
    c.state = .reconnecting;

    var last_err: anyerror = error.ConnectionClosed;
    for (0..reconnect_attempts_max) |i| {
        // Wait out the backoff.
        const attempt: u32 = @intCast(i);
        const ms = reconnectDelayMs(attempt);
        try c.out.note(reconnect_fmt, .{ attempt + 1, reconnect_attempts_max, ms });
        if (try waitTimer(c, queue, ms) == .quit) return quitReconnect(c);

        // Start the attempt of a new generation.
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

        // Collect the attempt. A quit also frees the client that the task returns.
        const wake = try waitAttempt(c, queue);
        var result = if (wake == .attempt) task.await(c.io) else task.cancel(c.io);
        if (wake == .quit) {
            if (result.client) |*fresh| {
                fresh.deinit();
            }

            return quitReconnect(c);
        }

        // A task cut at its deadline reports the cancel, and the attempt failed on time.
        if (wake == .timer and result.outcome == .failed and result.outcome.failed ==
            error.Canceled)
        {
            result.outcome = .{ .failed = error.Timeout };
        }

        // Dial again after a failure. End the chat on a changed server.
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

// -- helpers: reconnect -----------------------------------------------------

/// Install the client of a finished attempt. The attempt is `resumed` or `no_lease`.
fn adoptAttempt(c: *Chat, result: Attempt) !Outcome {
    c.client.deinit();
    c.client.* = result.client.?;

    // The new server can differ in its think mode and card rows, and the rebuild opens its session
    // from them.
    c.info = result.info.?;
    c.sampling = c.opts.sessionSampling(c.info);
    if (c.info.resume_ttl_max_ms == 0 and c.keep_ttl > 0) {
        c.keep_ttl = 0;
        try c.out.note(client_startup.keep_off_note, .{});
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

/// A quit while the client reconnects. No server hears a close, so the lease waits out its time
/// to live.
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

/// Wait for the attempt task of `c.gen` under its deadlines: one for the dial and the greeting,
/// then one for the resume.
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

/// Each pass takes one input. The loop ends at the timer `timer_id`, a marker of the attempt
/// `c.gen`, a quit, or the end of stdin. No socket task runs in this state, so no frame arrives.
fn waitFor(c: *Chat, queue: *InputQueue, timer_id: u32) !Wake {
    while (true) {
        const input = try queue.getOne(c.io);
        switch (input) {
            .timer => |id| if (id == timer_id) return .timer,
            .greeted => |g| if (g == c.gen) return .greeted,
            .attempt_done => |g| if (g == c.gen) return .attempt,
            .stdin_closed => return .quit,
            .line => |*l| {
                if (client_input.parseLine(l.buf[0..l.len]) == .quit) return .quit;
                try c.out.note(reconnecting_line_note, .{});
            },
            .line_too_long => try c.out.note(
                client_input.too_long_fmt,
                .{client_input.line_bytes_max - 1},
            ),
            .frame, .socket_closed => unreachable,
        }
    }
}

fn timerTask(io: Io, queue: *InputQueue, ms: u32, id: u32) void {
    io.sleep(.fromMilliseconds(ms), .awake) catch return;
    queue.putOne(io, .{ .timer = id }) catch {};
}

/// One dial, greeting, and resume on a new connection.
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

    // Request ids keep growing across connections, so `reconcile` never takes an id of the old
    // connection for a new request.
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
        if (classifyResumeRefusal(refusal.code) == .retry and !job.final) return error.ServerError;

        return .{ .client = fresh, .info = info, .outcome = .{ .no_lease = refusal } };
    };

    return .{ .client = fresh, .info = info, .outcome = .{ .resumed = r } };
}

/// Sort the refusal of a `session_resume`. A new `ErrorCode` fails to compile here until someone
/// classifies it.
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

/// The transcript costs and the token ids hold only on the same model with the same window.
fn sameServer(old: *const Client.HelloInfo, new: *const Client.HelloInfo) bool {
    return old.max_seq == new.max_seq and old.eos_id == new.eos_id and
        std.mem.eql(u8, old.modelName(), new.modelName());
}

/// The wait before reconnect attempt `attempt`, counted from 0.
fn reconnectDelayMs(attempt: u32) u32 {
    std.debug.assert(attempt < reconnect_attempts_max);

    const ms = @min(
        @as(u64, reconnect_delay_first_ms) << @intCast(attempt),
        reconnect_delay_max_ms,
    );
    std.debug.assert(ms >= reconnect_delay_first_ms);
    return @intCast(ms);
}

// -- tests ------------------------------------------------------------------

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
