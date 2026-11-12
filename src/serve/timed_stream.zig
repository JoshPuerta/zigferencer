//! A socket reader and writer with a deadline per read and per write, a hang-up probe, the TCP
//! keepalive of an accepted socket, and the one send of a refusal. Both frontends read and write a
//! connection through these types, so `--io-timeout` bounds every socket wait of a connection task.
//!
//! Deadline. Each read waits at most `timeout_ms` for the socket to become readable. Each write
//! call waits at most `timeout_ms` in total for room in the send buffer. The deadline is per call,
//! so a client that sends one byte per deadline keeps its connection. Zero means no deadline. An
//! expired wait or a failed wait fails the read or the write, and the task then exits through the
//! normal path. The wait measures the deadline on the monotonic clock, so a signal that restarts
//! the wait does not extend it.
//!
//! Wake. A task cancel does not wake a waiting `poll`. Teardown wakes it by a socket shutdown,
//! which makes the socket readable and writable.
//!
//! Platforms. Linux has the deadline, the probe, and the keepalive. The read waits in `poll`, and
//! the write uses a send that does not block, because a blocking send on a full buffer waits with
//! no bound. Std `Io.Threaded` reports `EAGAIN` from its own socket read as a bug, so a socket
//! receive timeout is no option. The drain of input (`endOutput`, `dropArrived`, `lingerOutput`)
//! and the `sendOnce` that does not wait also exist on Linux only. On the other platforms the
//! reader and the writer call the std socket path with no deadline, the probe reports no hang-up,
//! and the keepalive does nothing. These platforms drop no input in the drain functions, and
//! `sendOnce` is the blocking std write.
//!
//! Threads. A reader or a writer belongs to one connection task. The value must not move after
//! `init`, because the interface finds its parent by address.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const net = Io.net;
const posix = std.posix;
const limits = @import("../core/limits.zig");

// -- constants --------------------------------------------------------------

pub const has_deadline = builtin.os.tag == .linux;

/// The iovec count of one read. The std socket reader uses the same small fixed array.
const read_iovecs_max = 8;

// -- types ------------------------------------------------------------------

const Wait = enum { ready, timed_out, failed };

const SendResult = union(enum) { sent: usize, again, failed };

pub const Reader = struct {
    io: Io,
    stream: net.Stream,
    timeout_ms: u32,
    /// True after a read that failed at its deadline.
    timed_out: bool = false,
    interface: Io.Reader,

    pub fn init(stream: net.Stream, io: Io, buffer: []u8, timeout_ms: u32) Reader {
        return .{
            .io = io,
            .stream = stream,
            .timeout_ms = timeout_ms,
            .interface = .{
                .vtable = &.{ .stream = streamImpl, .readVec = readVec },
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
        };
    }

    fn streamImpl(io_r: *Io.Reader, io_w: *Io.Writer, limit: Io.Limit) Io.Reader.StreamError!usize {
        const dest = limit.slice(try io_w.writableSliceGreedy(1));
        var data: [1][]u8 = .{dest};
        const n = try readVec(io_r, &data);
        io_w.advance(n);
        return n;
    }

    fn readVec(io_r: *Io.Reader, data: [][]u8) Io.Reader.Error!usize {
        const r: *Reader = @alignCast(@fieldParentPtr("interface", io_r));
        var iovecs: [read_iovecs_max][]u8 = undefined;
        const dest_n, const data_size = try io_r.writableVector(&iovecs, data);
        const dest = iovecs[0..dest_n];
        std.debug.assert(dest[0].len > 0);

        if (r.timeout_ms > 0) {
            switch (waitReady(r.io, r.stream, .readable, deadlineAfter(r.io, r.timeout_ms))) {
                .ready => {},
                .timed_out => {
                    r.timed_out = true;
                    return error.ReadFailed;
                },
                .failed => return error.ReadFailed,
            }
        }

        const io = r.io;
        const n = io.vtable.netRead(io.userdata, r.stream.socket.handle, dest) catch {
            return error.ReadFailed;
        };
        if (n == 0) return error.EndOfStream;
        if (n > data_size) {
            io_r.end += n - data_size;
            return data_size;
        }

        return n;
    }
};

pub const Writer = struct {
    io: Io,
    stream: net.Stream,
    timeout_ms: u32,
    /// True after a write that failed at its deadline.
    timed_out: bool = false,
    interface: Io.Writer,

    pub fn init(stream: net.Stream, io: Io, buffer: []u8, timeout_ms: u32) Writer {
        return .{
            .io = io,
            .stream = stream,
            .timeout_ms = timeout_ms,
            .interface = .{ .vtable = &.{ .drain = drain }, .buffer = buffer },
        };
    }

    fn drain(io_w: *Io.Writer, data: []const []const u8, splat: usize) Io.Writer.Error!usize {
        const w: *Writer = @alignCast(@fieldParentPtr("interface", io_w));
        if (!has_deadline) return w.drainBlocking(data, splat);
        if (w.timeout_ms == 0) return w.drainBlocking(data, splat);

        // Each call sends one slice and waits at most one deadline. The caller loops.
        const buffered = io_w.buffered();
        if (buffered.len > 0) return io_w.consume(try w.send(buffered));
        for (data[0 .. data.len - 1]) |bytes| {
            if (bytes.len > 0) return w.send(bytes);
        }

        const pattern = data[data.len - 1];
        if (splat == 0 or pattern.len == 0) return 0;
        return w.send(pattern);
    }

    fn drainBlocking(w: *Writer, data: []const []const u8, splat: usize) Io.Writer.Error!usize {
        const io = w.io;
        const handle = w.stream.socket.handle;
        const buffered = w.interface.buffered();
        const n = io.vtable.netWrite(io.userdata, handle, buffered, data, splat) catch {
            return error.WriteFailed;
        };
        return w.interface.consume(n);
    }

    /// Send what fits now. A full send buffer waits for room, and all waits of one call end at
    /// one deadline.
    fn send(w: *Writer, bytes: []const u8) Io.Writer.Error!usize {
        std.debug.assert(has_deadline);
        std.debug.assert(bytes.len > 0);
        const deadline = deadlineAfter(w.io, w.timeout_ms);
        // A pass sends, fails, or waits for room. A wait that returns ready can still meet a full
        // buffer, so the passes repeat until the deadline. The bound turns a signal storm into a
        // failure.
        const passes_max = 64;
        var pass: u32 = 0;
        while (pass < passes_max) : (pass += 1) {
            switch (sendNow(w.stream, bytes)) {
                .sent => |n| return n,
                .again => switch (waitReady(w.io, w.stream, .writable, deadline)) {
                    .ready => {},
                    .timed_out => {
                        w.timed_out = true;
                        return error.WriteFailed;
                    },
                    .failed => return error.WriteFailed,
                },
                .failed => return error.WriteFailed,
            }
        }

        return error.WriteFailed;
    }
};

// -- api --------------------------------------------------------------------

/// True when the peer has closed the connection or its send side. Readable data alone is no
/// hang-up, because a client can send its next request before it reads the reply.
pub fn peerClosed(stream: net.Stream) bool {
    if (!has_deadline) return false;

    // Linux defines POLLRDHUP with the bit of EPOLLRDHUP, and std names only the second one.
    const rdhup: i16 = std.os.linux.EPOLL.RDHUP;
    var fds = [_]posix.pollfd{.{ .fd = stream.socket.handle, .events = rdhup, .revents = 0 }};
    const n = posix.poll(&fds, 0) catch return false;
    if (n == 0) return false;

    return fds[0].revents & (rdhup | posix.POLL.HUP | posix.POLL.ERR) != 0;
}

/// Turn on the TCP keepalive of an accepted socket, so a peer that vanishes without a close frees
/// its connection. A unix socket has no keepalive, and a refused option leaves the socket as it
/// is, so this reports nothing.
pub fn enableKeepalive(stream: net.Stream) void {
    if (!has_deadline) return;

    const fd = stream.socket.handle;
    const tcp = posix.IPPROTO.TCP;
    const options = [_]struct { level: i32, name: u32, value: c_int }{
        .{ .level = posix.SOL.SOCKET, .name = posix.SO.KEEPALIVE, .value = 1 },
        .{ .level = tcp, .name = posix.TCP.KEEPIDLE, .value = limits.keepalive_idle_s },
        .{ .level = tcp, .name = posix.TCP.KEEPINTVL, .value = limits.keepalive_interval_s },
        .{ .level = tcp, .name = posix.TCP.KEEPCNT, .value = limits.keepalive_probes },
    };
    for (options) |o| {
        posix.setsockopt(fd, o.level, o.name, std.mem.asBytes(&o.value)) catch return;
    }
}

/// Send `bytes` once with no wait. A short send drops the rest, because a refusal must not stall
/// the acceptor. Off Linux this is the std blocking write, so a peer that never reads can hold it.
pub fn sendOnce(io: Io, stream: net.Stream, bytes: []const u8) void {
    if (!has_deadline) {
        var w = stream.writer(io, &.{});
        w.interface.writeAll(bytes) catch {};
        return;
    }

    _ = sendNow(stream, bytes);
}

/// Close a refused socket in the order of `endOutput`.
pub fn closeRefused(io: Io, stream: *net.Stream) void {
    endOutput(io, stream.*);
    stream.close(io);
}

/// Shut the send side, then drop at most `limits.close_drain_bytes` of arrived input, with no
/// wait. Call it before the close, because a close over unread input sends a reset that can drop
/// the last reply. Input that arrives later can still cause the reset.
pub fn endOutput(io: Io, stream: net.Stream) void {
    stream.shutdown(io, .send) catch {};
    drainNow(stream);
}

/// Drop at most `limits.close_drain_bytes` of arrived input, with no wait and no shutdown. Call it
/// before a close that must send its end of stream only at the close itself.
pub fn dropArrived(stream: net.Stream) void {
    drainNow(stream);
}

/// Shut the send side, then read and drop input until the end of the stream, until
/// `limits.close_drain_bytes`, or for `wait_ms`, whichever comes first. A connection task calls it
/// after a refusal, because the client can still be sending its request.
pub fn lingerOutput(io: Io, stream: net.Stream, wait_ms: u32) void {
    stream.shutdown(io, .send) catch {};
    if (!has_deadline) return;

    const linux = std.os.linux;
    const deadline = deadlineAfter(io, wait_ms);
    const fd = stream.socket.handle;
    var buf: [4096]u8 = undefined;
    var dropped: usize = 0;
    // A pass drops at least one byte, or it finds no byte after a wake. The byte bound and the
    // deadline end the loop, and the pass bound also ends a storm of empty wakes.
    const passes_max = limits.close_drain_bytes + 64;
    for (0..passes_max) |_| {
        if (dropped >= limits.close_drain_bytes) return;
        if (waitReady(io, stream, .readable, deadline) != .ready) return;

        const want = @min(buf.len, limits.close_drain_bytes - dropped);
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

// -- helpers ----------------------------------------------------------------

/// Read and drop what the peer has sent so far, with no wait. The bound keeps the acceptor from
/// looping on a peer that keeps sending.
fn drainNow(stream: net.Stream) void {
    if (!has_deadline) return;

    const linux = std.os.linux;
    var buf: [4096]u8 = undefined;
    const reads_max = (limits.close_drain_bytes + buf.len - 1) / buf.len;
    for (0..reads_max) |_| {
        const fd = stream.socket.handle;
        const rc = linux.recvfrom(fd, &buf, buf.len, linux.MSG.DONTWAIT, null, null);
        switch (linux.errno(rc)) {
            .SUCCESS => if (rc == 0) return,
            else => return,
        }
    }
}

fn sendNow(stream: net.Stream, bytes: []const u8) SendResult {
    std.debug.assert(has_deadline);
    const linux = std.os.linux;
    const flags = linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL;
    const fd = stream.socket.handle;
    // A signal before any byte moves restarts the send. The bound turns a signal storm into a
    // failure.
    const passes_max = 64;
    for (0..passes_max) |_| {
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

fn deadlineAfter(io: Io, timeout_ms: u32) Io.Clock.Timestamp {
    const now = Io.Clock.Timestamp.now(io, .awake);
    return now.addDuration(.{ .raw = .fromMilliseconds(timeout_ms), .clock = .awake });
}

/// Wait until the socket is ready, or until `deadline`. A hang-up or an error counts as ready, so
/// the next read or write reports it. A failed `poll` is `failed`, so no caller falls into a
/// blocking call.
fn waitReady(
    io: Io,
    stream: net.Stream,
    ready: enum { readable, writable },
    deadline: Io.Clock.Timestamp,
) Wait {
    if (!has_deadline) return .ready;

    const events: i16 = switch (ready) {
        .readable => posix.POLL.IN,
        .writable => posix.POLL.OUT,
    };
    // A signal ends `poll` early, and the next pass waits the rest. The bound turns a signal
    // storm into a failure.
    const passes_max = 64;
    for (0..passes_max) |_| {
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

// -- tests ------------------------------------------------------------------

const testing = std.testing;

const Pair = struct {
    server: net.Server,
    client: net.Stream,
    conn: net.Stream,

    fn open(io: Io) !Pair {
        const addr = try net.IpAddress.parse("127.0.0.1", 0);
        var server = try addr.listen(io, .{ .reuse_address = true });
        errdefer server.deinit(io);
        const client = try server.socket.address.connect(io, .{ .mode = .stream });
        errdefer client.close(io);
        const conn = try server.accept(io);
        return .{ .server = server, .client = client, .conn = conn };
    }

    fn close(p: *Pair, io: Io) void {
        p.conn.close(io);
        p.client.close(io);
        p.server.deinit(io);
    }
};

test "timed_stream: a silent peer fails the read at the deadline, and data passes" {
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    var buf: [64]u8 = undefined;
    var r: Reader = .init(p.conn, io, &buf, 20);
    try testing.expectError(error.ReadFailed, r.interface.takeByte());
    try testing.expect(r.timed_out);

    var cw = p.client.writer(io, &.{});
    try cw.interface.writeAll("ab");
    r.timed_out = false;
    try testing.expectEqual(@as(u8, 'a'), try r.interface.takeByte());
    try testing.expect(!r.timed_out);
}

test "timed_stream: a peer that never reads fails the write at the deadline" {
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    var buf: [4096]u8 = undefined;
    var w: Writer = .init(p.conn, io, &buf, 20);
    const chunk: [65536]u8 = @splat('x');
    // Loopback buffers hold a few MiB, so the bound passes them with a wide margin.
    const chunks_max = 4096;
    var sent: u32 = 0;
    while (sent < chunks_max) : (sent += 1) {
        w.interface.writeAll(&chunk) catch break;
    }

    try testing.expect(sent < chunks_max);
    try testing.expect(w.timed_out);
}

test "timed_stream: the probe tells a close from a pipelined request" {
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.server.deinit(io);
    defer p.conn.close(io);

    try testing.expect(!peerClosed(p.conn));
    var cw = p.client.writer(io, &.{});
    try cw.interface.writeAll("GET / HTTP/1.1\r\n");
    try testing.expect(!peerClosed(p.conn));

    p.client.close(io);
    const fd = p.conn.socket.handle;
    var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
    // The close reaches the socket as an event, so this wait returns at once.
    _ = try posix.poll(&fds, -1);
    try testing.expect(peerClosed(p.conn));
}
