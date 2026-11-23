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
//! Wake. A task cancel does not wake a waiting `poll`. A shutdown alone does not end a pending
//! read or write on Windows. So teardown, a writer that stops, and a client that hangs up call
//! `wake`. This one socket call ends every read and write that waits on the socket. It also fails
//! the later ones. Linux shuts the socket down. Windows also cancels the I/O requests of the
//! socket.
//!
//! Platforms. `os/socket_wait.zig` makes the OS calls. Linux has the deadline, the probe, and the
//! keepalive. The read waits in `poll`, and the write uses a send that does not block. The drain
//! of input (`endOutput`, `dropArrived`, `lingerOutput`) and the `sendOnce` that does not wait
//! also exist on Linux only. On the other platforms the reader and the writer have no deadline,
//! the probe reports no hang-up, and the keepalive does nothing. These platforms drop no input in
//! the drain functions, and `sendOnce` is a write that blocks. Windows sends the reads and the
//! writes as I/O requests of its own, so the wake can cancel them. Every other OS calls the std
//! socket path.
//!
//! Threads. A reader or a writer belongs to one connection task. The interface finds its parent
//! by address. So a caller must not copy the interface out of its value. A caller also must not
//! move the value while a pointer to its interface is in use.

const std = @import("std");
const Io = std.Io;
const net = Io.net;
const limits = @import("../core/limits.zig");
const socket_wait = @import("../os/socket_wait.zig");

// -- constants --------------------------------------------------------------

pub const has_deadline = socket_wait.has_deadline;

/// The iovec count of one read. The std socket reader uses the same small fixed array.
const read_iovecs_max = 8;

// -- types ------------------------------------------------------------------

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
            switch (socket_wait.waitReadable(r.io, r.stream, deadlineAfter(r.io, r.timeout_ms))) {
                .ready => {},
                .timed_out => {
                    r.timed_out = true;
                    return error.ReadFailed;
                },
                .failed => return error.ReadFailed,
            }
        }

        const n = socket_wait.read(r.io, r.stream, dest) orelse return error.ReadFailed;
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
        const buffered = w.interface.buffered();
        const n = socket_wait.write(w.io, w.stream, buffered, data, splat) orelse {
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
            switch (socket_wait.sendNow(w.stream, bytes)) {
                .sent => |n| return n,
                .again => switch (socket_wait.waitWritable(w.io, w.stream, deadline)) {
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
    return socket_wait.peerClosed(stream);
}

/// Turn on the TCP keepalive of an accepted socket, so a peer that vanishes without a close frees
/// its connection. A unix socket has no keepalive, and a refused option leaves the socket as it
/// is, so this reports nothing.
pub fn enableKeepalive(stream: net.Stream) void {
    socket_wait.enableKeepalive(
        stream,
        limits.keepalive_idle_s,
        limits.keepalive_interval_s,
        limits.keepalive_probes,
    );
}

/// Send `bytes` once with no wait. A short send drops the rest, because a refusal must not stall
/// the acceptor. Off Linux this is a write that blocks, so a peer that never reads can hold it.
pub fn sendOnce(io: Io, stream: net.Stream, bytes: []const u8) void {
    if (!has_deadline) {
        var w: Writer = .init(stream, io, &.{}, 0);
        w.interface.writeAll(bytes) catch {};
        return;
    }

    _ = socket_wait.sendNow(stream, bytes);
}

/// End every read and write that waits on the socket, and fail the later ones. The handle must
/// stay open until the wake returns. Every read and write of the socket must go through this
/// file, because on Windows the wake cancels them and std does not accept a cancelled request.
pub fn wake(io: Io, stream: net.Stream) void {
    socket_wait.wake(io, stream);
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
    socket_wait.drainNow(stream, limits.close_drain_bytes);
}

/// Drop at most `limits.close_drain_bytes` of arrived input, with no wait and no shutdown. Call it
/// before a close that must send its end of stream only at the close itself.
pub fn dropArrived(stream: net.Stream) void {
    socket_wait.drainNow(stream, limits.close_drain_bytes);
}

/// Shut the send side, then read and drop input until the end of the stream, until
/// `limits.close_drain_bytes`, or for `wait_ms`, whichever comes first. A connection task calls it
/// after a refusal, because the client can still be sending its request.
pub fn lingerOutput(io: Io, stream: net.Stream, wait_ms: u32) void {
    stream.shutdown(io, .send) catch {};
    socket_wait.drainUntil(io, stream, limits.close_drain_bytes, deadlineAfter(io, wait_ms));
}

// -- helpers ----------------------------------------------------------------

fn deadlineAfter(io: Io, timeout_ms: u32) Io.Clock.Timestamp {
    const now = Io.Clock.Timestamp.now(io, .awake);
    return now.addDuration(.{ .raw = .fromMilliseconds(timeout_ms), .clock = .awake });
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

/// The deadline of a wait that a close of the peer ends at once.
const close_wait_ms: u32 = 60_000;

/// The bound for tests from a wake to the return of the last read or write that it ends, on the
/// `.awake` clock. The TCP timer of a half-closed connection ends a waiting read at 120 s, which is
/// also the limit of the test runner. Without this bound, a regression fails only at that limit.
/// The tests of this file and the loopback test of `server.zig` read it.
pub const wake_bound_ms: i64 = 10_000;

/// The passes of the order in which the read waits first. Without the handshake of the test, the
/// read already waited at the wake in most passes. With it, a few passes are enough.
const wake_read_passes = 4;

/// Loopback buffers hold a few MiB, so the chunk bound passes them with a wide margin.
const wake_chunks_max = 4096;
const wake_chunk = [_]u8{'x'} ** (64 * 1024);

/// The end of one read or write task of the wake test, on the `.awake` clock.
const WakeEnd = struct { failed: bool, at: Io.Clock.Timestamp };

/// The ends of one order of the wake test, from the time of the wake.
const WakeOrder = struct { woke_at: Io.Clock.Timestamp, read: WakeEnd, write: ?WakeEnd };

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

fn readUntilWake(io: Io, stream: net.Stream) WakeEnd {
    var buf: [64]u8 = undefined;
    var r: Reader = .init(stream, io, &buf, 0);
    const failed = if (r.interface.takeByte()) |_| false else |_| true;
    return .{ .failed = failed, .at = .now(io, .awake) };
}

fn writeUntilWake(io: Io, stream: net.Stream) WakeEnd {
    var buf: [4096]u8 = undefined;
    var w: Writer = .init(stream, io, &buf, 0);
    var sent: u32 = 0;
    while (sent < wake_chunks_max) : (sent += 1) {
        w.interface.writeAll(&wake_chunk) catch break;
    }

    return .{ .failed = sent < wake_chunks_max, .at = .now(io, .awake) };
}

/// Read the one byte of the peer, set `got_byte`, and read again until the wake. A first read that
/// fails also sets `got_byte`, so the test never waits for it in vain.
fn readByteThenWait(io: Io, stream: net.Stream, got_byte: *Io.Event) WakeEnd {
    var buf: [64]u8 = undefined;
    var r: Reader = .init(stream, io, &buf, 0);
    const first = r.interface.takeByte();
    got_byte.set(io);
    const first_ok = if (first) |byte| byte == 'a' else |_| false;
    const second_failed = if (r.interface.takeByte()) |_| false else |_| true;
    return .{ .failed = first_ok and second_failed, .at = .now(io, .awake) };
}

/// Order A: the wake comes first, and then a read and a write start on the socket.
fn wakeThenCall(io: Io, stream: net.Stream) !WakeOrder {
    const woke_at: Io.Clock.Timestamp = .now(io, .awake);
    wake(io, stream);
    var reader = try io.concurrent(readUntilWake, .{ io, stream });
    errdefer _ = reader.cancel(io);
    var writer = try io.concurrent(writeUntilWake, .{ io, stream });
    return .{ .woke_at = woke_at, .read = reader.await(io), .write = writer.await(io) };
}

/// Order B: the read consumes the byte of the peer and reads again, and then the wake comes.
fn readThenWake(io: Io, p: *Pair) !WakeOrder {
    var cw = p.client.writer(io, &.{});
    try cw.interface.writeAll("a");

    var got_byte: Io.Event = .unset;
    var reader = try io.concurrent(readByteThenWait, .{ io, p.conn, &got_byte });
    errdefer _ = reader.cancel(io);
    try got_byte.wait(io);

    const woke_at: Io.Clock.Timestamp = .now(io, .awake);
    wake(io, p.conn);
    return .{ .woke_at = woke_at, .read = reader.await(io), .write = null };
}

fn expectWakeEnds(order: WakeOrder) !void {
    try testing.expect(order.read.failed);
    const read_ms = order.woke_at.durationTo(order.read.at).raw.toMilliseconds();
    try testing.expect(read_ms < wake_bound_ms);
    if (order.write) |write| {
        try testing.expect(write.failed);
        const write_ms = order.woke_at.durationTo(write.at).raw.toMilliseconds();
        try testing.expect(write_ms < wake_bound_ms);
    }
}

test "timed_stream: a wake ends a later read and write, and a read that waits" {
    // Order A shows that a call after the wake fails. In order B the event proves that the reader
    // task runs and consumed the byte, so its second read starts at once. That read can still
    // start a few instructions after the wake, and the wake ends a read in either order.
    //
    // A write that already waits is not covered here. It waits only after megabytes fill the
    // buffers, and nothing tells the test when that happened. A failed bound still awaits the
    // tasks, and the TCP timer ends a read that waits.
    const io = testing.io;
    {
        var p = try Pair.open(io);
        defer p.close(io);

        try expectWakeEnds(try wakeThenCall(io, p.conn));
    }

    for (0..wake_read_passes) |_| {
        var p = try Pair.open(io);
        defer p.close(io);

        try expectWakeEnds(try readThenWake(io, &p));
    }
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
    // The close reaches the socket as an event, so this wait returns long before its deadline.
    const wait = socket_wait.waitReadable(io, p.conn, deadlineAfter(io, close_wait_ms));
    try testing.expectEqual(socket_wait.Wait.ready, wait);
    try testing.expect(peerClosed(p.conn));
}
