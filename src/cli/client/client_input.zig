//! What the client reads: the stdin lines, the slash commands, and the queue that feeds the
//! streaming control unit.
//!
//! Comptime. An assert keeps a prompt line inside one `generate` frame.

const std = @import("std");
const Io = std.Io;

const protocol = @import("../../serve/protocol.zig");
const client_mod = @import("../../serve/client.zig");
const transcript_mod = @import("client_transcript.zig");

const Client = client_mod.Client;

// -- constants --------------------------------------------------------------

/// A stdin line holds at most `line_bytes_max - 1` bytes plus its newline.
pub const line_bytes_max = transcript_mod.prompt_bytes_max;

pub const too_long_fmt = "line exceeds {d} bytes, dropped";

pub const commands_text =
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

    // A prompt line fits one `generate` frame, so the client never builds an oversized frame.
    std.debug.assert(line_bytes_max <= protocol.generate_text_bytes_max);
}

// -- types ------------------------------------------------------------------

pub const Input = union(enum) {
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

pub const InputQueue = Io.Queue(Input);

pub const Command = union(enum) {
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

// -- api --------------------------------------------------------------------

pub fn stdinTask(io: Io, queue: *InputQueue) void {
    var buf: [line_bytes_max]u8 = undefined;
    var reader = Io.File.stdin().readerStreaming(io, &buf);
    const r = &reader.interface;

    // Each pass reads one line. The loop ends at EOF, a read error, a cancel (seen as a read
    // error), or a closed queue.
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

/// The command a typed line names. Only a trailing `\r` is trimmed.
pub fn parseLine(raw: []const u8) Command {
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

pub fn printCommands(e: *Io.Writer) !void {
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
