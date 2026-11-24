//! The serve socket calls on an OS with no deadline. The read, the write, and the wake are the std
//! calls. The wait returns at once, the drain drops nothing, the probe reports no hang-up, and the
//! keepalive does nothing.

const std = @import("std");
const socket_wait = @import("socket_wait.zig");
const Io = std.Io;
const net = Io.net;

const Wait = socket_wait.Wait;
const TimedSendResult = socket_wait.TimedSendResult;

// -- constants --------------------------------------------------------------

pub const has_deadline: bool = false;
pub const late_input_drops_reply: bool = false;
pub const unix_path_absolute_only: bool = false;

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

/// The caller sends through `write` when `has_deadline` is false, so no call reaches this body.
pub fn sendUntil(
    io: Io,
    stream: net.Stream,
    bytes: []const u8,
    deadline: Io.Clock.Timestamp,
) TimedSendResult {
    _ = io;
    _ = stream;
    _ = bytes;
    _ = deadline;
    unreachable;
}

pub fn drainUntil(io: Io, stream: net.Stream, max_bytes: usize, deadline: Io.Clock.Timestamp) bool {
    _ = io;
    _ = stream;
    _ = max_bytes;
    _ = deadline;
    return false;
}

pub fn peerClosed(io: Io, stream: net.Stream) bool {
    _ = io;
    _ = stream;
    return false;
}

pub fn enableKeepalive(
    io: Io,
    stream: net.Stream,
    idle_s: c_int,
    interval_s: c_int,
    probes: c_int,
) void {
    _ = io;
    _ = stream;
    _ = idle_s;
    _ = interval_s;
    _ = probes;
}
