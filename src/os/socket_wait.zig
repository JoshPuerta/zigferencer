//! The OS calls of the serve sockets. They are the read, the write, the wake, the wait with a
//! deadline, the send and the drain that do not block, the hang-up probe, and the TCP keepalive.
//! `serve/timed_stream.zig` builds its reader and writer on them.
//!
//! Platforms. One file holds the calls of each OS, and this file selects one at compile time.
//! Linux uses `socket_wait_linux.zig`. Windows uses `socket_wait_windows.zig`: its own read, write,
//! and wake, and no deadline. Every other OS uses `socket_wait_none.zig`: no deadline, a probe that
//! reports no hang-up, no keepalive, and drains that drop nothing.
//!
//! Comptime. A check below asserts that the selected file declares every function with the type
//! that this file expects, so an incomplete OS file fails the build.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const net = Io.net;

comptime {
    assertImpl(Impl);
}

// -- types ------------------------------------------------------------------

/// The file that holds the calls of the target OS.
const Impl = switch (builtin.os.tag) {
    .linux => @import("socket_wait_linux.zig"),
    .windows => @import("socket_wait_windows.zig"),
    else => @import("socket_wait_none.zig"),
};

/// The end of a wait: the socket is ready, the deadline passed, or the wait call failed.
pub const Wait = enum { ready, timed_out, failed };

/// The end of a send that does not block: the bytes that it sent, a full send buffer, or a failure.
pub const SendResult = union(enum) { sent: usize, again, failed };

// -- api --------------------------------------------------------------------

/// True when the OS file has the deadline, the probe, the keepalive, and the drains.
pub const has_deadline: bool = Impl.has_deadline;

/// Read once into the buffers of `dest`. Null when the read failed. Zero is the end of the stream.
pub const read = Impl.read;

/// Send bytes from the front of `buffered`, then `data`, then the last slice of `data` `splat`
/// times, as std `netWrite` does. Return the count that it sent, or null when the write failed.
pub const write = Impl.write;

/// End every read and write that waits on the socket, and fail the later ones.
pub const wake = Impl.wake;

/// Wait until the socket is readable, or until the deadline. A hang-up or an error counts as ready,
/// so the next read reports it.
pub const waitReadable = Impl.waitReadable;

/// Wait until the socket has room to send, or until the deadline. A hang-up or an error counts as
/// ready, so the next write reports it.
pub const waitWritable = Impl.waitWritable;

/// Send what fits now, with no wait. Call it only when `has_deadline` is true.
pub const sendNow = Impl.sendNow;

/// Read and drop at most `max_bytes` that the peer has sent so far, with no wait.
pub const drainNow = Impl.drainNow;

/// Read and drop input until the end of the stream, until `max_bytes`, or until the deadline.
pub const drainUntil = Impl.drainUntil;

/// True when the peer has closed the connection or its send side.
pub const peerClosed = Impl.peerClosed;

/// Turn on the TCP keepalive of a socket with an idle time, a probe interval, and a probe count.
/// A refused option leaves the socket as it is.
pub const enableKeepalive = Impl.enableKeepalive;

// -- helpers ----------------------------------------------------------------

/// An OS file must declare each function with the type of the Linux file. One that lacks a
/// function, or declares it with another type, fails the build.
fn assertImpl(comptime T: type) void {
    const Deadline = Io.Clock.Timestamp;
    const expected = .{
        .{ "read", fn (Io, net.Stream, [][]u8) ?usize },
        .{ "write", fn (Io, net.Stream, []const u8, []const []const u8, usize) ?usize },
        .{ "wake", fn (Io, net.Stream) void },
        .{ "waitReadable", fn (Io, net.Stream, Deadline) Wait },
        .{ "waitWritable", fn (Io, net.Stream, Deadline) Wait },
        .{ "sendNow", fn (net.Stream, []const u8) SendResult },
        .{ "drainNow", fn (net.Stream, usize) void },
        .{ "drainUntil", fn (Io, net.Stream, usize, Deadline) void },
        .{ "peerClosed", fn (net.Stream) bool },
        .{ "enableKeepalive", fn (net.Stream, c_int, c_int, c_int) void },
    };
    if (!@hasDecl(T, "has_deadline") or @TypeOf(T.has_deadline) != bool) {
        @compileError("socket wait file " ++ @typeName(T) ++ " lacks has_deadline: bool");
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
