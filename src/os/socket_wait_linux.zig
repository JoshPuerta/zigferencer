//! The serve socket calls on Linux. The read, the write, and the wake are the std calls. The wait
//! polls the socket with a deadline. The send with a deadline joins that wait and a send that does
//! not block. A blocking send on a full buffer waits with no bound. Std `Io.Threaded` reports
//! `EAGAIN` from its own socket read as a bug, so a socket receive timeout is no option.
//!
//! Deadline. The wait measures the deadline on the monotonic clock, so a signal that restarts the
//! wait does not extend it.

const std = @import("std");
const socket_wait = @import("socket_wait.zig");
const Io = std.Io;
const net = Io.net;
const linux = std.os.linux;
const posix = std.posix;

const Wait = socket_wait.Wait;
const TimedSendResult = socket_wait.TimedSendResult;

// -- constants --------------------------------------------------------------

pub const has_deadline: bool = true;
pub const late_input_drops_reply: bool = false;
pub const unix_path_absolute_only: bool = false;

/// Bytes of one receive of a drain.
const drain_chunk_bytes: usize = 4096;

/// The passes of one loop that a signal or an empty wait can restart. The bound turns a storm of
/// them into a failure.
const signal_passes_max: u32 = 64;

// -- types ------------------------------------------------------------------

/// The end of a send that does not block: the bytes that it sent, no room at the time of the call,
/// or a failure. Room can come back at once after `again`.
const SendResult = union(enum) { sent: usize, again, failed };

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
    return waitEvents(io, stream, posix.POLL.IN, deadline);
}

pub fn sendUntil(
    io: Io,
    stream: net.Stream,
    bytes: []const u8,
    deadline: Io.Clock.Timestamp,
) TimedSendResult {
    // A wait that returns ready can still meet a full buffer, so the pass repeats. The bound ends
    // a storm of such passes with a failure.
    for (0..signal_passes_max) |_| {
        switch (sendNow(stream, bytes)) {
            .sent => |n| return .{ .sent = n },
            .again => switch (waitWritable(io, stream, deadline)) {
                .ready => {},
                .timed_out => return .timed_out,
                .failed => return .failed,
            },
            .failed => return .failed,
        }
    }

    return .failed;
}

pub fn drainUntil(io: Io, stream: net.Stream, max_bytes: usize, deadline: Io.Clock.Timestamp) bool {
    const fd = stream.socket.handle;
    var buf: [drain_chunk_bytes]u8 = undefined;
    var dropped: usize = 0;
    // A pass drops at least one byte, waits for input, or restarts after a signal. One wait can
    // come before each byte, so the byte bound gives two passes for each byte. The other passes
    // end a storm of signals or of empty waits with no byte.
    const passes_max = 2 * max_bytes + signal_passes_max;
    for (0..passes_max) |_| {
        if (dropped >= max_bytes) return false;

        const want = @min(buf.len, max_bytes - dropped);
        const rc = linux.recvfrom(fd, &buf, want, linux.MSG.DONTWAIT, null, null);
        switch (linux.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return true;
                dropped += rc;
            },
            .AGAIN => if (waitReadable(io, stream, deadline) != .ready) return false,
            .INTR => {},
            else => return false,
        }
    }

    return false;
}

/// Readable data alone is no hang-up, because a client can send its next request before it reads
/// the reply.
pub fn peerClosed(io: Io, stream: net.Stream) bool {
    _ = io;
    // Linux defines POLLRDHUP with the bit of EPOLLRDHUP, and std names only the second one.
    const rdhup: i16 = linux.EPOLL.RDHUP;
    var fds = [_]posix.pollfd{.{ .fd = stream.socket.handle, .events = rdhup, .revents = 0 }};
    const n = posix.poll(&fds, 0) catch return false;
    if (n == 0) return false;

    return fds[0].revents & (rdhup | posix.POLL.HUP | posix.POLL.ERR) != 0;
}

/// A unix socket has no keepalive, so this reports nothing.
pub fn enableKeepalive(
    io: Io,
    stream: net.Stream,
    idle_s: c_int,
    interval_s: c_int,
    probes: c_int,
) void {
    _ = io;
    const fd = stream.socket.handle;
    const tcp = posix.IPPROTO.TCP;
    const options = [_]struct { level: i32, name: u32, value: c_int }{
        .{ .level = posix.SOL.SOCKET, .name = posix.SO.KEEPALIVE, .value = 1 },
        .{ .level = tcp, .name = posix.TCP.KEEPIDLE, .value = idle_s },
        .{ .level = tcp, .name = posix.TCP.KEEPINTVL, .value = interval_s },
        .{ .level = tcp, .name = posix.TCP.KEEPCNT, .value = probes },
    };
    for (options) |o| {
        posix.setsockopt(fd, o.level, o.name, std.mem.asBytes(&o.value)) catch return;
    }
}

// -- helpers ----------------------------------------------------------------

fn sendNow(stream: net.Stream, bytes: []const u8) SendResult {
    const flags = linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL;
    const fd = stream.socket.handle;
    // A signal before any byte moves restarts the send.
    for (0..signal_passes_max) |_| {
        const rc = linux.sendto(fd, bytes.ptr, bytes.len, flags, null, 0);
        switch (linux.errno(rc)) {
            .SUCCESS => return .{ .sent = rc },
            .INTR => continue,
            .AGAIN => return .again,
            else => return .failed,
        }
    }

    return .failed;
}

fn waitWritable(io: Io, stream: net.Stream, deadline: Io.Clock.Timestamp) Wait {
    return waitEvents(io, stream, posix.POLL.OUT, deadline);
}

/// A failed `poll` is `failed`, so no caller falls into a blocking call.
fn waitEvents(io: Io, stream: net.Stream, events: i16, deadline: Io.Clock.Timestamp) Wait {
    // A signal ends `poll` early, and the next pass waits the rest.
    for (0..signal_passes_max) |_| {
        const left_ms = deadline.durationFromNow(io).raw.toMilliseconds();
        if (left_ms <= 0) return .timed_out;

        var fds = [_]posix.pollfd{.{ .fd = stream.socket.handle, .events = events, .revents = 0 }};
        const wait: i32 = @intCast(@min(left_ms, std.math.maxInt(i32)));
        const rc = posix.system.poll(&fds, @intCast(fds.len), wait);
        switch (posix.errno(rc)) {
            .SUCCESS => if (rc > 0) return .ready else return .timed_out,
            .INTR => continue,
            else => return .failed,
        }
    }

    return .failed;
}
