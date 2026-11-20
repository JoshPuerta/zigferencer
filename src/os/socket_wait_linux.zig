//! The serve socket calls on Linux. The wait polls the socket with a deadline, and the send and
//! the drain do not block, because a blocking send on a full buffer waits with no bound. Std
//! `Io.Threaded` reports `EAGAIN` from its own socket read as a bug, so a socket receive timeout is
//! no option.
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
const SendResult = socket_wait.SendResult;

// -- constants --------------------------------------------------------------

pub const has_deadline: bool = true;

/// Bytes of one receive of a drain.
const drain_chunk_bytes: usize = 4096;

/// The passes of one call that a signal can restart. The bound turns a signal storm into a failure.
const signal_passes_max: u32 = 64;

// -- api --------------------------------------------------------------------

pub fn waitReadable(io: Io, stream: net.Stream, deadline: Io.Clock.Timestamp) Wait {
    return waitEvents(io, stream, posix.POLL.IN, deadline);
}

pub fn waitWritable(io: Io, stream: net.Stream, deadline: Io.Clock.Timestamp) Wait {
    return waitEvents(io, stream, posix.POLL.OUT, deadline);
}

pub fn sendNow(stream: net.Stream, bytes: []const u8) SendResult {
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

/// The bound keeps the acceptor from looping on a peer that keeps sending.
pub fn drainNow(stream: net.Stream, max_bytes: usize) void {
    var buf: [drain_chunk_bytes]u8 = undefined;
    const reads_max = (max_bytes + buf.len - 1) / buf.len;
    for (0..reads_max) |_| {
        const fd = stream.socket.handle;
        const rc = linux.recvfrom(fd, &buf, buf.len, linux.MSG.DONTWAIT, null, null);
        switch (linux.errno(rc)) {
            .SUCCESS => if (rc == 0) return,
            else => return,
        }
    }
}

pub fn drainUntil(io: Io, stream: net.Stream, max_bytes: usize, deadline: Io.Clock.Timestamp) void {
    const fd = stream.socket.handle;
    var buf: [drain_chunk_bytes]u8 = undefined;
    var dropped: usize = 0;
    // A pass drops at least one byte, or it finds no byte after a wake. The byte bound and the
    // deadline end the loop, and the pass bound also ends a storm of empty wakes.
    const passes_max = max_bytes + signal_passes_max;
    for (0..passes_max) |_| {
        if (dropped >= max_bytes) return;
        if (waitReadable(io, stream, deadline) != .ready) return;

        const want = @min(buf.len, max_bytes - dropped);
        const rc = linux.recvfrom(fd, &buf, want, linux.MSG.DONTWAIT, null, null);
        switch (linux.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return;
                dropped += rc;
            },
            .AGAIN, .INTR => {},
            else => return,
        }
    }
}

/// Readable data alone is no hang-up, because a client can send its next request before it reads
/// the reply.
pub fn peerClosed(stream: net.Stream) bool {
    // Linux defines POLLRDHUP with the bit of EPOLLRDHUP, and std names only the second one.
    const rdhup: i16 = linux.EPOLL.RDHUP;
    var fds = [_]posix.pollfd{.{ .fd = stream.socket.handle, .events = rdhup, .revents = 0 }};
    const n = posix.poll(&fds, 0) catch return false;
    if (n == 0) return false;

    return fds[0].revents & (rdhup | posix.POLL.HUP | posix.POLL.ERR) != 0;
}

/// A unix socket has no keepalive, so this reports nothing.
pub fn enableKeepalive(stream: net.Stream, idle_s: c_int, interval_s: c_int, probes: c_int) void {
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
