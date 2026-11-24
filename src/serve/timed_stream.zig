//! A socket reader and writer with a deadline per read and per write, a hang-up probe, the TCP
//! keepalive of an accepted socket, and the one send of a refusal. Both frontends read and write a
//! connection through these types, so `--io-timeout` bounds every socket wait of a connection task.
//!
//! Deadline. Each read waits at most `timeout_ms` for input. Each write call waits at most
//! `timeout_ms` in total until the OS takes some of its bytes. The deadline is per call, so a
//! client that sends one byte per deadline keeps its connection. Zero means no deadline. An expired
//! wait or a failed wait fails the read or the write, and the task then exits through the normal
//! path. The wait measures the deadline on the monotonic clock, so a wait that ends early and
//! starts again does not extend it.
//!
//! Failed write. After a write fails, some of its bytes can have reached the peer. So a failed
//! write must end the connection, and the peer then sees a cut stream.
//!
//! Wake. A task cancel does not end a wait with a deadline. So teardown, a writer that stops, and
//! a client that hangs up call `wake`. This one call ends every read, write, and wait on the
//! socket. It also fails the later ones.
//!
//! Platforms. `os/socket_wait.zig` makes the OS calls, and each OS file states how it waits. Where
//! `has_deadline` is false, the reader and the writer have no deadline, the probe reports no
//! hang-up, and the keepalive does nothing. There the drain functions drop no input, and
//! `sendOnce` is a write that blocks.
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
pub const late_input_drops_reply = socket_wait.late_input_drops_reply;

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

    /// Send some of `bytes`. Every wait of one call ends at one deadline.
    fn send(w: *Writer, bytes: []const u8) Io.Writer.Error!usize {
        std.debug.assert(has_deadline);
        std.debug.assert(bytes.len > 0);
        const deadline = deadlineAfter(w.io, w.timeout_ms);
        switch (socket_wait.sendUntil(w.io, w.stream, bytes, deadline)) {
            .sent => |n| return n,
            .timed_out => {
                w.timed_out = true;
                return error.WriteFailed;
            },
            .failed => return error.WriteFailed,
        }
    }
};

// -- api --------------------------------------------------------------------

/// True when the peer has closed the connection or its send side. Readable data alone is no
/// hang-up, because a client can send its next request before it reads the reply.
pub fn peerClosed(io: Io, stream: net.Stream) bool {
    return socket_wait.peerClosed(io, stream);
}

/// Turn on the TCP keepalive of an accepted socket, so a peer that vanishes without a close frees
/// its connection. A unix socket has no keepalive, and a refused option leaves the socket as it
/// is, so this reports nothing.
pub fn enableKeepalive(io: Io, stream: net.Stream) void {
    socket_wait.enableKeepalive(
        io,
        stream,
        limits.keepalive_idle_s,
        limits.keepalive_interval_s,
        limits.keepalive_probes,
    );
}

/// Send `bytes` once with no wait. A short send drops the rest, because a refusal must not stall
/// the acceptor. Without the deadline this is a write that blocks, so a peer that never reads can
/// hold it.
pub fn sendOnce(io: Io, stream: net.Stream, bytes: []const u8) void {
    std.debug.assert(bytes.len > 0);
    if (!has_deadline) {
        var w: Writer = .init(stream, io, &.{}, 0);
        w.interface.writeAll(bytes) catch {};
        return;
    }

    _ = socket_wait.sendUntil(io, stream, bytes, .now(io, .awake));
}

/// End every read and write that waits on the socket, and fail the later ones. The handle must
/// stay open until the wake returns. Every read and write of the socket must go through this file.
/// On Windows the wake cancels each request, and std reaches `unreachable` on one of its own.
pub fn wake(io: Io, stream: net.Stream) void {
    socket_wait.wake(io, stream);
}

/// Close a refused socket. Where late input drops the unread refusal, the close waits for the end
/// of the client input, at most `limits.refuse_linger_ms`. Elsewhere it waits for nothing.
pub fn closeRefused(io: Io, stream: *net.Stream) void {
    if (late_input_drops_reply) {
        lingerOutput(io, stream.*, limits.refuse_linger_ms);
    } else {
        endOutput(io, stream.*);
    }
    stream.close(io);
}

/// Shut the send side, then drop at most `limits.close_drain_bytes` of arrived input, with no
/// wait. Call it before the close, because a close over unread input sends a reset that can drop
/// the last reply. Input that arrives later can still cause the reset.
pub fn endOutput(io: Io, stream: net.Stream) void {
    stream.shutdown(io, .send) catch {};
    _ = socket_wait.drainUntil(io, stream, limits.close_drain_bytes, .now(io, .awake));
}

/// Drop at most `limits.close_drain_bytes` of arrived input, with no wait and no shutdown. Call it
/// before a close that must send its end of stream only at the close itself.
pub fn dropArrived(io: Io, stream: net.Stream) void {
    _ = socket_wait.drainUntil(io, stream, limits.close_drain_bytes, .now(io, .awake));
}

/// Shut the send side, then read and drop input until the end of the stream or until
/// `limits.close_drain_bytes`. The drop waits for more input for at most `wait_ms`. The client of a
/// refused request or of a refused connection can still be sending.
pub fn lingerOutput(io: Io, stream: net.Stream, wait_ms: u32) void {
    stream.shutdown(io, .send) catch {};
    _ = socket_wait.drainUntil(io, stream, limits.close_drain_bytes, deadlineAfter(io, wait_ms));
}

/// Read and drop input with no shutdown until its end or `limits.close_drain_bytes`. The drop
/// waits for more input for at most `wait_ms`, and `wake` ends it with either result. True at the
/// end of the input. A reader calls it after a fatal reply, and after false the client can lose it.
pub fn dropInputUntilEnd(io: Io, stream: net.Stream, wait_ms: u32) bool {
    return socket_wait.drainUntil(io, stream, limits.close_drain_bytes, deadlineAfter(io, wait_ms));
}

// -- helpers ----------------------------------------------------------------

fn deadlineAfter(io: Io, timeout_ms: u32) Io.Clock.Timestamp {
    const now = Io.Clock.Timestamp.now(io, .awake);
    return now.addDuration(.{ .raw = .fromMilliseconds(timeout_ms), .clock = .awake });
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

/// The deadline of a wait that waiting input or a close of the peer ends at once.
const close_wait_ms: u32 = 60_000;

/// The bound for tests from a wake to the return of the last read or write that it ends, on the
/// `.awake` clock. Without it, a wake that ends nothing fails only at the limit of the test runner.
/// The tests of this file and the loopback test of `server.zig` read it. One drop test reads it
/// as the bound of a drop whose input has ended.
pub const wake_bound_ms: i64 = 10_000;

/// The deadline of a read or a write that a wake ends. It is far over `wake_bound_ms`, so only the
/// wake can end the call inside that bound.
const wake_deadline_ms: u32 = 60_000;

/// The passes of order B. The second read can start after the wake, so the order repeats.
const wake_read_passes = 4;

/// The bound is far over what a loopback send path takes before it is full.
const wake_chunks_max = 4096;
const wake_chunk = [_]u8{'x'} ** (64 * 1024);

/// The end of one read or write task of the wake test, on the `.awake` clock.
const WakeEnd = struct { failed: bool, timed_out: bool, at: Io.Clock.Timestamp };

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
    // The bound is far over what a loopback send path takes before it is full.
    const chunks_max = 4096;
    var sent: u32 = 0;
    while (sent < chunks_max) : (sent += 1) {
        w.interface.writeAll(&chunk) catch break;
    }

    try testing.expect(sent < chunks_max);
    try testing.expect(w.timed_out);
}

fn readUntilWake(io: Io, stream: net.Stream, timeout_ms: u32) WakeEnd {
    var buf: [64]u8 = undefined;
    var r: Reader = .init(stream, io, &buf, timeout_ms);
    const failed = if (r.interface.takeByte()) |_| false else |_| true;
    return .{ .failed = failed, .timed_out = r.timed_out, .at = .now(io, .awake) };
}

fn writeUntilWake(io: Io, stream: net.Stream, timeout_ms: u32) WakeEnd {
    var buf: [4096]u8 = undefined;
    var w: Writer = .init(stream, io, &buf, timeout_ms);
    var sent: u32 = 0;
    while (sent < wake_chunks_max) : (sent += 1) {
        w.interface.writeAll(&wake_chunk) catch break;
    }

    return .{ .failed = sent < wake_chunks_max, .timed_out = w.timed_out, .at = .now(io, .awake) };
}

fn signalThenWrite(io: Io, stream: net.Stream, started: *Io.Event) WakeEnd {
    started.set(io);
    return writeUntilWake(io, stream, wake_deadline_ms);
}

/// Send `bytes_total` bytes and then the end of the stream. False when a send or the shutdown
/// failed.
fn sendThenShut(io: Io, stream: net.Stream, bytes_total: usize) bool {
    var w: Writer = .init(stream, io, &.{}, wake_deadline_ms);
    var left = bytes_total;
    // Each pass sends one chunk, so the chunk count of `bytes_total` bounds the loop.
    while (left > 0) {
        const n = @min(left, wake_chunk.len);
        w.interface.writeAll(wake_chunk[0..n]) catch return false;
        left -= n;
    }
    stream.shutdown(io, .send) catch return false;
    return true;
}

/// Read the one byte of the peer, set `got_byte`, and read again until the wake. A first read that
/// fails also sets `got_byte`, so the test never waits for it in vain.
fn readByteThenWait(io: Io, stream: net.Stream, timeout_ms: u32, got_byte: *Io.Event) WakeEnd {
    var buf: [64]u8 = undefined;
    var r: Reader = .init(stream, io, &buf, timeout_ms);
    const first = r.interface.takeByte();
    got_byte.set(io);
    const first_ok = if (first) |byte| byte == 'a' else |_| false;
    const second_failed = if (r.interface.takeByte()) |_| false else |_| true;
    return .{
        .failed = first_ok and second_failed,
        .timed_out = r.timed_out,
        .at = .now(io, .awake),
    };
}

/// Order A: the wake comes first, and then a read and a write start on the socket.
fn wakeThenCall(io: Io, stream: net.Stream, timeout_ms: u32) !WakeOrder {
    const woke_at: Io.Clock.Timestamp = .now(io, .awake);
    wake(io, stream);
    var reader = try io.concurrent(readUntilWake, .{ io, stream, timeout_ms });
    errdefer _ = reader.cancel(io);
    var writer = try io.concurrent(writeUntilWake, .{ io, stream, timeout_ms });
    return .{ .woke_at = woke_at, .read = reader.await(io), .write = writer.await(io) };
}

/// Order B: the read consumes the byte of the peer and reads again, and then the wake comes.
fn readThenWake(io: Io, p: *Pair, timeout_ms: u32) !WakeOrder {
    var cw = p.client.writer(io, &.{});
    try cw.interface.writeAll("a");

    var got_byte: Io.Event = .unset;
    const args = .{ io, p.conn, timeout_ms, &got_byte };
    var reader = try io.concurrent(readByteThenWait, args);
    errdefer _ = reader.cancel(io);
    try got_byte.wait(io);

    const woke_at: Io.Clock.Timestamp = .now(io, .awake);
    wake(io, p.conn);
    return .{ .woke_at = woke_at, .read = reader.await(io), .write = null };
}

/// A wake ends each call inside the bound, and no call reports a deadline.
fn expectWakeEnds(order: WakeOrder) !void {
    try expectWokenAt(order.woke_at, order.read);
    if (order.write) |write| {
        try expectWokenAt(order.woke_at, write);
    }
}

fn expectWokenAt(woke_at: Io.Clock.Timestamp, end: WakeEnd) !void {
    try testing.expect(end.failed);
    try testing.expect(!end.timed_out);
    try testing.expect(woke_at.durationTo(end.at).raw.toMilliseconds() < wake_bound_ms);
}

/// Send chunks with a deadline of now until a send finds no room. Room can come back at once.
fn fillSendPath(io: Io, stream: net.Stream) !void {
    for (0..wake_chunks_max) |_| {
        switch (socket_wait.sendUntil(io, stream, &wake_chunk, .now(io, .awake))) {
            .sent => {},
            .timed_out => return,
            .failed => return error.SendFailed,
        }
    }

    return error.SendPathNeverFull;
}

/// The byte at `offset` of the stream of the prefix test. A prime period makes a lost or a doubled
/// chunk of any power-of-two length visible.
fn patternByte(offset: u64) u8 {
    return @intCast(offset % 251);
}

/// Read until the end of the stream. The first `reported` bytes must be the pattern in order.
fn expectPrefixToEnd(io: Io, stream: net.Stream, reported: u64) !void {
    var buf: [16 * 1024]u8 = undefined;
    var received: u64 = 0;
    // A read returns at least one byte or the end. A cancelled send adds at most one chunk.
    const reads_max = reported + wake_chunk.len + 1;
    for (0..reads_max) |_| {
        // The wait fails a lost end of stream at its deadline, so the test does not hang.
        const wait = socket_wait.waitReadable(io, stream, deadlineAfter(io, close_wait_ms));
        try testing.expectEqual(socket_wait.Wait.ready, wait);
        var dest = [_][]u8{&buf};
        const n = socket_wait.read(io, stream, &dest) orelse return error.ReadFailed;
        if (n == 0) {
            try testing.expect(received >= reported);
            return;
        }

        for (buf[0..n], received..) |byte, offset| {
            if (offset < reported) {
                try testing.expectEqual(patternByte(offset), byte);
            }
        }
        received += n;
    }

    return error.TooManyBytes;
}

test "timed_stream: a wake ends a later read and write, and a read that waits" {
    // Order A shows that a call after the wake fails. In order B the event proves that the reader
    // task runs and consumed the byte, so its second read starts at once. That read can still
    // start a few instructions after the wake, and the wake ends a read in either order.
    const io = testing.io;
    {
        var p = try Pair.open(io);
        defer p.close(io);

        try expectWakeEnds(try wakeThenCall(io, p.conn, 0));
    }

    for (0..wake_read_passes) |_| {
        var p = try Pair.open(io);
        defer p.close(io);

        try expectWakeEnds(try readThenWake(io, &p, 0));
    }
}

test "timed_stream: a wake ends a read and a write with a deadline, before the deadline" {
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    {
        var p = try Pair.open(io);
        defer p.close(io);

        try expectWakeEnds(try wakeThenCall(io, p.conn, wake_deadline_ms));
    }

    for (0..wake_read_passes) |_| {
        var p = try Pair.open(io);
        defer p.close(io);

        try expectWakeEnds(try readThenWake(io, &p, wake_deadline_ms));
    }
}

test "timed_stream: a wake ends a write that waits for room" {
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    // The writer sends until the path is full, then it waits. In the first order the wake comes
    // before the write starts. `timed_out` means only that the send found no room at that time, so
    // this order covers a full path in most runs, not in all.
    {
        var full = try Pair.open(io);
        defer full.close(io);

        try fillSendPath(io, full.conn);
        const woke_at: Io.Clock.Timestamp = .now(io, .awake);
        wake(io, full.conn);
        try expectWokenAt(woke_at, writeUntilWake(io, full.conn, wake_deadline_ms));
    }

    try fillSendPath(io, p.conn);
    var started: Io.Event = .unset;
    var writer = try io.concurrent(signalThenWrite, .{ io, p.conn, &started });
    started.wait(io) catch |err| {
        _ = writer.cancel(io);
        return err;
    };

    const woke_at: Io.Clock.Timestamp = .now(io, .awake);
    wake(io, p.conn);
    const end = writer.await(io);
    try expectWokenAt(woke_at, end);
}

test "timed_stream: the reported bytes reach the peer in order, also after a deadline" {
    // The bytes of the writes that succeeded are a prefix of what the peer receives. A write that
    // ends at its deadline can add bytes after that prefix, but it cannot move or drop one in it.
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    var chunk: [wake_chunk.len]u8 = undefined;
    var reported: u64 = 0;
    var w: Writer = .init(p.conn, io, &.{}, 20);
    for (0..wake_chunks_max) |_| {
        for (&chunk, reported..) |*byte, offset| {
            byte.* = patternByte(offset);
        }
        reported += w.interface.write(&chunk) catch break;
    }
    try testing.expect(w.timed_out);

    try p.conn.shutdown(io, .send);
    try expectPrefixToEnd(io, p.client, reported);
}

test "timed_stream: after the drop of input to its end, a reply before a wake reaches the peer" {
    // The peer sends input that the server never reads and then its end. Without the drop, the
    // wake finds unread input, and where late input drops a reply the peer loses the reply.
    // Where late input drops no reply, the reply arrives also without the drop. So a pass there
    // does not prove the drop.
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    switch (socket_wait.sendUntil(io, p.client, &wake_chunk, .now(io, .awake))) {
        .sent => |n| try testing.expect(n > 0),
        .timed_out, .failed => return error.SendFailed,
    }
    try p.client.shutdown(io, .send);

    var w: Writer = .init(p.conn, io, &.{}, wake_deadline_ms);
    try w.interface.writeAll("reply");
    const drop_at: Io.Clock.Timestamp = .now(io, .awake);
    try testing.expect(dropInputUntilEnd(io, p.conn, wake_deadline_ms));
    // A drop that finds the end late still returns true, so the bound fails it.
    const took = drop_at.durationTo(.now(io, .awake));
    try testing.expect(took.raw.toMilliseconds() < wake_bound_ms);
    wake(io, p.conn);

    var buf: [64]u8 = undefined;
    var r: Reader = .init(p.client, io, &buf, close_wait_ms);
    try testing.expectEqualStrings("reply", try r.interface.takeArray(5));
    try testing.expectError(error.EndOfStream, r.interface.takeByte());
}

test "timed_stream: a drop of input that starts after a wake ends before its deadline" {
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    const woke_at: Io.Clock.Timestamp = .now(io, .awake);
    wake(io, p.conn);
    _ = dropInputUntilEnd(io, p.conn, wake_deadline_ms);
    const took = woke_at.durationTo(.now(io, .awake));
    try testing.expect(took.raw.toMilliseconds() < wake_bound_ms);
}

test "timed_stream: a drop of input that stops at its byte bound reports no end" {
    // The peer sends more than the bound and then its end, so the drop stops with input unread.
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    const sent_bytes = limits.close_drain_bytes + wake_chunk.len;
    var sender = try io.concurrent(sendThenShut, .{ io, p.client, sent_bytes });
    const ended = dropInputUntilEnd(io, p.conn, wake_deadline_ms);
    // The rest of the input lets the sender finish, so the test awaits it before any check.
    var buf: [4096]u8 = undefined;
    var r: Reader = .init(p.conn, io, &buf, close_wait_ms);
    const rest = r.interface.discardRemaining();
    const sent = sender.await(io);

    try testing.expect(!ended);
    try testing.expect(sent);
    try testing.expectEqual(sent_bytes - limits.close_drain_bytes, try rest);
}

test "timed_stream: the drop of arrived input takes the bytes that wait" {
    // The peer stays open and silent after its bytes. A drop that leaves a byte makes the read
    // below return that byte.
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    var cw = p.client.writer(io, &.{});
    try cw.interface.writeAll("GET / HTTP/1.1\r\n");
    // Loopback delivers the bytes of one small send together, so a ready wait sees all of them.
    const wait = socket_wait.waitReadable(io, p.conn, deadlineAfter(io, close_wait_ms));
    try testing.expectEqual(socket_wait.Wait.ready, wait);

    dropArrived(io, p.conn);
    var buf: [64]u8 = undefined;
    var r: Reader = .init(p.conn, io, &buf, 20);
    try testing.expectError(error.ReadFailed, r.interface.takeByte());
    try testing.expect(r.timed_out);
}

test "timed_stream: the probe tells a close from a pipelined request" {
    if (!has_deadline) return error.SkipZigTest;
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.server.deinit(io);
    defer p.conn.close(io);

    try testing.expect(!peerClosed(io, p.conn));
    var cw = p.client.writer(io, &.{});
    try cw.interface.writeAll("GET / HTTP/1.1\r\n");
    try testing.expect(!peerClosed(io, p.conn));

    p.client.close(io);
    // The bytes of the request make this wait return at once. So only the order of the loopback
    // delivery puts the close before the probe below.
    const wait = socket_wait.waitReadable(io, p.conn, deadlineAfter(io, close_wait_ms));
    try testing.expectEqual(socket_wait.Wait.ready, wait);
    try testing.expect(peerClosed(io, p.conn));
}
