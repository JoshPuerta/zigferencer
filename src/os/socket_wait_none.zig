//! The serve socket calls on an OS with no implementation. The wait returns at once, the drains
//! drop nothing, the probe reports no hang-up, and the keepalive does nothing. The reader and the
//! writer then use the std socket path with no deadline.

const std = @import("std");
const socket_wait = @import("socket_wait.zig");
const Io = std.Io;
const net = Io.net;

const Wait = socket_wait.Wait;
const SendResult = socket_wait.SendResult;

// -- constants --------------------------------------------------------------

pub const has_deadline: bool = false;

// -- api --------------------------------------------------------------------

pub fn read(io: Io, stream: net.Stream, dest: [][]u8) ?usize {
    return io.vtable.netRead(io.userdata, stream.socket.handle, dest) catch null;
}

pub fn write(
    io: Io,
    stream: net.Stream,
    buffered: []const u8,
    data: []const []const u8,
    splat: usize,
) ?usize {
    const handle = stream.socket.handle;
    return io.vtable.netWrite(io.userdata, handle, buffered, data, splat) catch null;
}

pub fn wake(io: Io, stream: net.Stream) void {
    stream.shutdown(io, .both) catch {};
}

pub fn waitReadable(io: Io, stream: net.Stream, deadline: Io.Clock.Timestamp) Wait {
    _ = io;
    _ = stream;
    _ = deadline;
    return .ready;
}

pub fn waitWritable(io: Io, stream: net.Stream, deadline: Io.Clock.Timestamp) Wait {
    _ = io;
    _ = stream;
    _ = deadline;
    return .ready;
}

/// The caller sends through `write` when `has_deadline` is false, so no call reaches this body.
pub fn sendNow(stream: net.Stream, bytes: []const u8) SendResult {
    _ = stream;
    _ = bytes;
    unreachable;
}

pub fn drainNow(stream: net.Stream, max_bytes: usize) void {
    _ = stream;
    _ = max_bytes;
}

pub fn drainUntil(io: Io, stream: net.Stream, max_bytes: usize, deadline: Io.Clock.Timestamp) void {
    _ = io;
    _ = stream;
    _ = max_bytes;
    _ = deadline;
}

pub fn peerClosed(stream: net.Stream) bool {
    _ = stream;
    return false;
}

pub fn enableKeepalive(stream: net.Stream, idle_s: c_int, interval_s: c_int, probes: c_int) void {
    _ = stream;
    _ = idle_s;
    _ = interval_s;
    _ = probes;
}
