//! The OS calls of the serve sockets, with their result types and capability constants.
//! `serve/timed_stream.zig` builds its reader and writer on them.
//!
//! Platforms. One file holds the calls of each OS, and this file selects one at compile time. The
//! files are `socket_wait_linux.zig`, `socket_wait_windows.zig`, and `socket_wait_none.zig` for an
//! OS with no deadline. Each OS file states how it makes the calls.
//!
//! Comptime. `assertImpl` fails the build when the selected file lacks a function or a capability
//! constant, or declares one with another type.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const net = Io.net;

comptime {
    assertImpl(Impl);
}

// -- types ------------------------------------------------------------------

const Impl = switch (builtin.os.tag) {
    .linux => @import("socket_wait_linux.zig"),
    .windows => @import("socket_wait_windows.zig"),
    else => @import("socket_wait_none.zig"),
};

/// The end of a wait: the socket is ready, the deadline passed, or the wait call failed.
pub const Wait = enum { ready, timed_out, failed };

/// The end of a send with a deadline: the bytes that it sent, the deadline, or a failure.
pub const TimedSendResult = union(enum) { sent: usize, timed_out, failed };

// -- api --------------------------------------------------------------------

/// True when the OS file has the deadline, the probe, the keepalive, and the drain.
pub const has_deadline: bool = Impl.has_deadline;

/// True where input that arrives after a close makes the peer lose the reply that it has not read.
/// A refusal and a fatal reply then drop the input of the client until a deadline.
pub const late_input_drops_reply: bool = Impl.late_input_drops_reply;

/// True where a unix socket takes an absolute path only.
pub const unix_path_absolute_only: bool = Impl.unix_path_absolute_only;

/// Read once into the buffers of `dest`. Null when the read failed. Zero is the end of the stream.
pub const read = Impl.read;

/// Send bytes from the front of `buffered`, then `data`, then the last slice of `data` `splat`
/// times, as std `netWrite` does. Return the count that it sent, or null when the write failed.
pub const write = Impl.write;

/// End every read and write that waits on the socket, and fail the later ones.
pub const wake = Impl.wake;

/// Wait until the socket is readable, or until the deadline. A hang-up counts as ready, so the
/// next read reports it. An error counts as ready or fails the wait.
pub const waitReadable = Impl.waitReadable;

/// Send some of `bytes`, and wait for room until the deadline. A wake ends the wait. After
/// `timed_out` or `failed`, some of the bytes can have reached the peer, so the caller must end
/// the connection. Call it only when `has_deadline` is true.
pub const sendUntil = Impl.sendUntil;

/// Read and drop input until the end of the stream, until `max_bytes`, or until the deadline. A
/// deadline that passed drops only the arrived input. The byte bound keeps the caller from looping.
/// True when a receive gave the end of the stream. On Linux a receive after `wake` gives it too.
pub const drainUntil = Impl.drainUntil;

/// True when the peer has closed the connection or its send side.
pub const peerClosed = Impl.peerClosed;

/// Turn on the TCP keepalive of a socket with an idle time, a probe interval, and a probe count.
/// A refused option leaves the socket as it is.
pub const enableKeepalive = Impl.enableKeepalive;

// -- helpers ----------------------------------------------------------------

fn assertImpl(comptime T: type) void {
    const Deadline = Io.Clock.Timestamp;
    const expected = .{
        .{ "read", fn (Io, net.Stream, [][]u8) ?usize },
        .{ "write", fn (Io, net.Stream, []const u8, []const []const u8, usize) ?usize },
        .{ "wake", fn (Io, net.Stream) void },
        .{ "waitReadable", fn (Io, net.Stream, Deadline) Wait },
        .{ "sendUntil", fn (Io, net.Stream, []const u8, Deadline) TimedSendResult },
        .{ "drainUntil", fn (Io, net.Stream, usize, Deadline) bool },
        .{ "peerClosed", fn (Io, net.Stream) bool },
        .{ "enableKeepalive", fn (Io, net.Stream, c_int, c_int, c_int) void },
    };
    const flags = .{ "has_deadline", "late_input_drops_reply", "unix_path_absolute_only" };
    inline for (flags) |name| {
        if (!@hasDecl(T, name) or @TypeOf(@field(T, name)) != bool) {
            @compileError("socket wait file " ++ @typeName(T) ++ " lacks " ++ name ++ ": bool");
        }
    }
    // The drop of late input uses `drainUntil`, which needs the deadline.
    if (T.late_input_drops_reply and !T.has_deadline) {
        @compileError("socket wait file " ++ @typeName(T) ++ " drops replies with no deadline");
    }
    inline for (expected) |entry| {
        const name = entry[0];
        if (!@hasDecl(T, name)) {
            @compileError("socket wait file " ++ @typeName(T) ++ " lacks " ++ name);
        }
        if (@TypeOf(@field(T, name)) != entry[1]) {
            @compileError("socket wait file " ++ @typeName(T) ++ " declares " ++ name ++
                " with type " ++ @typeName(@TypeOf(@field(T, name))));
        }
    }
}
