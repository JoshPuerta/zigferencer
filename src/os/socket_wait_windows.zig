//! The serve socket calls on Windows. The read and the write send one `AFD.RECEIVE` or one
//! `AFD.SEND` request through `io.operate`, and the wake cancels every request of the socket. The
//! wait for input is one receive of zero bytes, and the send with a deadline is one send. This file
//! sends both itself and cancels each one at its deadline. A deadline that passed cancels the
//! request at once, so such a call waits only for the end of that cancel. The probe is an
//! `AFD.POLL` on the socket with a timeout of zero.
//!
//! Wake. A shutdown alone does not end a pending receive or send on Windows. A pending receive
//! then waits for the TCP timer of the half-closed connection, and a pending send waits with no
//! bound. So the wake also cancels every request of the socket handle. A receive or a send that
//! starts after the wake fails at once. The status of a cancelled request comes back to this file,
//! which turns it into a failed call.
//!
//! Requests. Every call that can wait is a request of this file on the socket handle itself, so
//! the cancel of the wake reaches it. Std reaches `unreachable` when a cancel ends its own socket
//! read, write, or accept. A poll cannot be the wait. A read poll that starts after the wake waits
//! its whole timeout, and so does a send poll on a full socket: the wake sets no poll event.
//!
//! Deadline. One send takes the whole slice or waits. So the deadline of a write covers the time
//! until the send path takes the whole slice. A send that a cancel ends reports 0 bytes, and some
//! of its bytes can still reach the peer. That send ends the connection, so the peer sees a cut
//! stream.
//!
//! Cancel. A task cancel of std signals once. When a request of this file takes it, the request
//! calls `io.recancel` and then fails. So the next cancel point of the task returns
//! `error.Canceled`.
//!
//! Lifetime. The status block, the flag, and the buffers of a request are locals of the function
//! that sends it. That function returns only after the completion routine ran. The routine runs in
//! an alertable wait of the thread that sent the request, so the `Io` of the process must be
//! `Io.Threaded`, where a task stays on one thread. Std `Io.Batch` holds the same request, but
//! `batchCancel` waits for a routine or an alert before it sends the cancel. Its start call also
//! reaches `unreachable` when the wake cancels the request inside that call.

const std = @import("std");
const socket_wait = @import("socket_wait.zig");
const kernel32 = @import("windows_kernel32_declarations.zig");
const afd = @import("windows_afd_declarations.zig");
const Io = std.Io;
const net = Io.net;
const windows = std.os.windows;
const ntdll = windows.ntdll;
const AFD = windows.AFD;

const Wait = socket_wait.Wait;
const TimedSendResult = socket_wait.TimedSendResult;

// -- constants --------------------------------------------------------------

pub const has_deadline: bool = true;

/// Input that arrives after the close or the wake makes the next read of the peer fail, and the
/// peer loses the bytes that it did not read.
pub const late_input_drops_reply: bool = true;

/// Std returns `error.Unexpected` for a listen on a relative unix socket path.
pub const unix_path_absolute_only: bool = true;

/// The buffer count of one receive. Std uses the same bound for its own socket read.
const read_buffers_max = Io.Threaded.max_iovecs_len;

/// The flags that std passes with each socket request.
const afd_flags: AFD = .{ .NO_FAST_IO = true, .OVERLAPPED = true };

/// Bytes of one receive of a drain.
const drain_chunk_bytes: usize = 4096;

/// The events of a hang-up. Input alone is no hang-up.
const hang_up_events = afd.poll_disconnect | afd.poll_abort;

/// The sleeps of one timed request. While the request is pending, only an alert or a routine of
/// another request ends a sleep early. Std sends at most one late alert here, so the bound turns
/// an alert storm into a failure.
const wait_passes_max: u32 = 64;

// -- types ------------------------------------------------------------------

/// One request that this file sends with its own completion routine and waits for until a
/// deadline. The value must stay in place from the start until `finish` returns. The kernel writes
/// the status block and reads the buffers and the input block, and the routine writes the flag.
const TimedRequest = struct {
    iosb: windows.IO_STATUS_BLOCK,
    done: bool,
    /// The status that the start call returned. An error status queues no routine.
    status_at_call: windows.NTSTATUS,
    no_bytes: [1]u8,
    receive_buffers: [1]AFD.WSABUF(.@"var"),
    send_buffers: [1]AFD.WSABUF(.@"const"),
    info: union { receive: AFD.RECV_INFO, send: AFD.SEND_INFO },

    fn startReceive(r: *TimedRequest, handle: windows.HANDLE) windows.NTSTATUS {
        r.receive_buffers = .{.{ .buf = &r.no_bytes, .len = 0 }};
        r.info = .{ .receive = .{
            .BufferArray = &r.receive_buffers,
            .BufferCount = r.receive_buffers.len,
            .AfdFlags = afd_flags,
            .TdiFlags = .{ .NORMAL = true },
        } };
        return r.start(handle, windows.IOCTL.AFD.RECEIVE, std.mem.asBytes(&r.info.receive));
    }

    fn startSend(r: *TimedRequest, handle: windows.HANDLE, bytes: []const u8) windows.NTSTATUS {
        std.debug.assert(bytes.len > 0);
        r.send_buffers = .{.{ .buf = bytes.ptr, .len = clampLen(bytes.len) }};
        r.info = .{ .send = .{
            .BufferArray = &r.send_buffers,
            .BufferCount = r.send_buffers.len,
            .AfdFlags = afd_flags,
            .TdiFlags = .{},
        } };
        return r.start(handle, windows.IOCTL.AFD.SEND, std.mem.asBytes(&r.info.send));
    }

    /// No completion routine runs after an error status. The caller must check that the status is
    /// an error before it returns with no `finish`.
    fn start(
        r: *TimedRequest,
        handle: windows.HANDLE,
        code: windows.CTL_CODE,
        in: []const u8,
    ) windows.NTSTATUS {
        r.done = false;
        r.status_at_call = ntdll.NtDeviceIoControlFile(
            handle,
            null,
            markDone,
            &r.done,
            &r.iosb,
            code,
            in.ptr,
            @intCast(in.len),
            null,
            0,
        );
        return r.status_at_call;
    }

    fn finishReceive(
        r: *TimedRequest,
        io: Io,
        handle: windows.HANDLE,
        deadline: Io.Clock.Timestamp,
    ) Wait {
        const expired = r.finish(io, handle, deadline);
        return switch (r.iosb.u.Status) {
            .SUCCESS => ready: {
                std.debug.assert(r.iosb.Information == 0);
                break :ready .ready;
            },
            // A cancel before the deadline comes from the wake or from the pass bound, so
            // `timed_out` means the deadline.
            .CANCELLED => if (expired) .timed_out else .failed,
            else => .failed,
        };
    }

    /// A send that completed while the cancel was on its way keeps the count of the kernel.
    fn finishSend(
        r: *TimedRequest,
        io: Io,
        handle: windows.HANDLE,
        deadline: Io.Clock.Timestamp,
    ) TimedSendResult {
        const expired = r.finish(io, handle, deadline);
        return switch (r.iosb.u.Status) {
            .SUCCESS => sent: {
                std.debug.assert(r.iosb.Information <= r.send_buffers[0].len);
                break :sent .{ .sent = r.iosb.Information };
            },
            .CANCELLED => if (expired) .timed_out else .failed,
            else => .failed,
        };
    }

    /// Wait for the completion routine, and cancel the request at the deadline. True when the
    /// deadline passed before the routine ran.
    fn finish(r: *TimedRequest, io: Io, handle: windows.HANDLE, deadline: Io.Clock.Timestamp) bool {
        // A request with an error status at the call has no routine, so this wait never ends.
        std.debug.assert(!isErrorStatus(r.status_at_call));
        var expired = false;
        var passes: u32 = 0;
        while (!r.done) : (passes += 1) {
            const left = hundredNanosUntil(io, deadline);
            if (left <= 0 or passes == wait_passes_max) {
                expired = left <= 0;
                r.cancel(handle);
                break;
            }

            const interval: windows.LARGE_INTEGER = -left;
            _ = ntdll.NtDelayExecution(.TRUE, &interval);
        }

        std.debug.assert(r.done);
        return expired;
    }

    /// The kernel holds pointers into this value until the routine ran, so this wait has no pass
    /// bound.
    fn cancel(r: *TimedRequest, handle: windows.HANDLE) void {
        var cancel_iosb: windows.IO_STATUS_BLOCK = undefined;
        // NOT_FOUND means that the request completed first, and its routine still runs.
        _ = ntdll.NtCancelIoFileEx(handle, &r.iosb, &cancel_iosb);
        while (!r.done) {
            Io.Threaded.waitForApcOrAlert();
        }
    }
};

// -- api --------------------------------------------------------------------

pub fn read(io: Io, stream: net.Stream, dest: [][]u8) ?usize {
    var buffers: [read_buffers_max]AFD.WSABUF(.@"var") = undefined;
    var count: u32 = 0;
    for (dest) |buf| {
        if (count == buffers.len) break;
        if (buf.len == 0) continue;

        buffers[count] = .{ .buf = buf.ptr, .len = clampLen(buf.len) };
        count += 1;
        // The bytes past a clamped buffer must not land in the next one.
        if (buf.len > std.math.maxInt(u32)) break;
    }

    // A receive with no buffer fails, so a caller must pass room for one byte.
    std.debug.assert(count > 0);
    const info: AFD.RECV_INFO = .{
        .BufferArray = &buffers,
        .BufferCount = count,
        .AfdFlags = afd_flags,
        .TdiFlags = .{ .NORMAL = true },
    };
    return request(io, stream, windows.IOCTL.AFD.RECEIVE, std.mem.asBytes(&info));
}

/// One call sends one slice, because the caller loops until it sent all of them.
pub fn write(
    io: Io,
    stream: net.Stream,
    buffered: []const u8,
    data: []const []const u8,
    splat: usize,
) ?usize {
    std.debug.assert(data.len > 0);

    const bytes = firstSlice(buffered, data, splat) orelse return 0;
    const buffers = [1]AFD.WSABUF(.@"const"){.{ .buf = bytes.ptr, .len = clampLen(bytes.len) }};
    const info: AFD.SEND_INFO = .{
        .BufferArray = &buffers,
        .BufferCount = buffers.len,
        .AfdFlags = afd_flags,
        .TdiFlags = .{},
    };
    return request(io, stream, windows.IOCTL.AFD.SEND, std.mem.asBytes(&info));
}

pub fn wake(io: Io, stream: net.Stream) void {
    // The shutdown is a cancel point of std. A task cancel there skips the shutdown and is lost.
    const protect = io.swapCancelProtection(.blocked);
    defer _ = io.swapCancelProtection(protect);

    stream.shutdown(io, .both) catch {};
    // FALSE with no request to cancel is the normal case, so the result needs no check.
    _ = kernel32.CancelIoEx(stream.socket.handle, null);
}

/// The wait is one receive of zero bytes. It carries no data, so a cancel at the deadline can lose
/// no byte.
pub fn waitReadable(io: Io, stream: net.Stream, deadline: Io.Clock.Timestamp) Wait {
    const handle = stream.socket.handle;
    var input: TimedRequest = undefined;
    if (isErrorStatus(input.startReceive(handle))) return .failed;

    return input.finishReceive(io, handle, deadline);
}

pub fn sendUntil(
    io: Io,
    stream: net.Stream,
    bytes: []const u8,
    deadline: Io.Clock.Timestamp,
) TimedSendResult {
    const handle = stream.socket.handle;
    var output: TimedRequest = undefined;
    if (isErrorStatus(output.startSend(handle, bytes))) return .failed;

    return output.finishSend(io, handle, deadline);
}

pub fn drainUntil(io: Io, stream: net.Stream, max_bytes: usize, deadline: Io.Clock.Timestamp) bool {
    var buf: [drain_chunk_bytes]u8 = undefined;
    var dropped: usize = 0;
    // A pass drops at least one byte or returns, so the byte bound is also the pass bound.
    for (0..max_bytes) |_| {
        if (dropped >= max_bytes) return false;
        if (waitReadable(io, stream, deadline) != .ready) return false;

        const want = @min(buf.len, max_bytes - dropped);
        var dest = [_][]u8{buf[0..want]};
        const n = read(io, stream, &dest) orelse return false;
        if (n == 0) return true;
        dropped += n;
    }

    return false;
}

/// A failed poll reports no hang-up.
pub fn peerClosed(io: Io, stream: net.Stream) bool {
    return pollSocket(io, stream, hang_up_events, 0);
}

/// A unix socket refuses the first option. Its peer is on the same host and cannot vanish with no
/// close, so this reports nothing.
pub fn enableKeepalive(
    io: Io,
    stream: net.Stream,
    idle_s: c_int,
    interval_s: c_int,
    probes: c_int,
) void {
    const ws2_32 = windows.ws2_32;
    const tcp = ws2_32.IPPROTO.TCP;
    // Std names the idle time `TCP.KEEPALIVE`, the number of `TCP_KEEPIDLE` in the Windows headers.
    const options = [_]struct { level: i32, name: u32, value: c_int }{
        .{ .level = ws2_32.SOL.SOCKET, .name = ws2_32.SO.KEEPALIVE, .value = 1 },
        .{ .level = tcp, .name = ws2_32.TCP.KEEPALIVE, .value = idle_s },
        .{ .level = tcp, .name = ws2_32.TCP.KEEPINTVL, .value = interval_s },
        .{ .level = tcp, .name = ws2_32.TCP.KEEPCNT, .value = probes },
    };
    for (options) |o| {
        if (!setOption(io, stream, o.level, o.name, @intCast(o.value))) return;
    }
}

// -- helpers ----------------------------------------------------------------

/// One socket-option request. Std declares the request and its record, but not a public call.
fn setOption(io: Io, stream: net.Stream, level: i32, name: u32, value: u32) bool {
    const info: AFD.SOCKOPT_INFO = .{
        .mode = .set,
        .level = level,
        .optname = name,
        .optval = &value,
        .optlen = @sizeOf(u32),
    };
    return request(io, stream, windows.IOCTL.AFD.SOCKOPT, std.mem.asBytes(&info)) != null;
}

/// One poll on the socket itself, so the wake reaches it. A timeout of zero never waits. True when
/// an event of the mask came before the timeout. A failed request reports no event.
fn pollSocket(io: Io, stream: net.Stream, events: windows.ULONG, hundred_nanos: i64) bool {
    std.debug.assert(hundred_nanos >= 0);
    std.debug.assert(events != 0);

    const handle = stream.socket.handle;
    var info: afd.AFD_POLL_INFO = .{
        .Timeout = -hundred_nanos,
        .NumberOfHandles = 1,
        .Exclusive = 0,
        .Handles = .{.{ .Handle = handle, .Events = events, .Status = .SUCCESS }},
    };
    const bytes = std.mem.asBytes(&info);
    const result = io.operate(.{ .device_io_control = .{
        .file = .{ .handle = handle, .flags = .{ .nonblocking = true } },
        .code = windows.IOCTL.AFD.POLL,
        .in = bytes,
        .out = bytes,
    } }) catch |err| switch (err) {
        error.Canceled => {
            io.recancel();
            return false;
        },
    };
    // The kernel writes the events only when it reports one handle.
    const status = result.device_io_control.u.Status;
    return status == .SUCCESS and info.NumberOfHandles > 0;
}

/// Every status other than success fails the call, a cancel included.
fn request(io: Io, stream: net.Stream, code: windows.CTL_CODE, in: []const u8) ?usize {
    const result = io.operate(.{ .device_io_control = .{
        .file = .{ .handle = stream.socket.handle, .flags = .{ .nonblocking = true } },
        .code = code,
        .in = in,
    } }) catch |err| switch (err) {
        error.Canceled => {
            io.recancel();
            return null;
        },
    };
    const iosb = result.device_io_control;
    if (iosb.u.Status != .SUCCESS) return null;

    return iosb.Information;
}

/// An NT status with both top bits set is an error. A status with a lower severity can still
/// queue the completion routine.
fn isErrorStatus(status: windows.NTSTATUS) bool {
    return @intFromEnum(status) >> 30 == 0b11;
}

fn markDone(
    context: ?*anyopaque,
    _: *windows.IO_STATUS_BLOCK,
    _: windows.ULONG,
) align(Io.Threaded.apc_align) callconv(.winapi) void {
    const done: *bool = @ptrCast(context);
    done.* = true;
}

/// The time to the deadline in the unit of the NT calls. It rounds up, so the request never gets
/// less time than the caller gave.
fn hundredNanosUntil(io: Io, deadline: Io.Clock.Timestamp) i64 {
    const nanos = deadline.durationFromNow(io).raw.toNanoseconds();
    if (nanos <= 0) return 0;

    const hundreds = @divFloor(nanos + 99, 100);
    return @intCast(@min(hundreds, std.math.maxInt(i64)));
}

/// The first slice that is not empty, in the order of std `netWrite`.
fn firstSlice(buffered: []const u8, data: []const []const u8, splat: usize) ?[]const u8 {
    if (buffered.len > 0) return buffered;
    for (data[0 .. data.len - 1]) |bytes| {
        if (bytes.len > 0) return bytes;
    }

    const pattern = data[data.len - 1];
    if (splat == 0 or pattern.len == 0) return null;
    return pattern;
}

/// A request takes at most `maxInt(u32)` bytes for each buffer. The caller reads or sends the
/// rest in a later call.
fn clampLen(len: usize) u32 {
    return @intCast(@min(len, std.math.maxInt(u32)));
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

/// The deadline of a wait that an event ends. A regression then fails at this deadline, not at the
/// limit of the test runner.
const event_wait_ms: u32 = 60_000;

/// The deadline of a wait that no event ends.
const silent_wait_ms: u32 = 20;

/// The sends that one test queues on a socket whose peer reads nothing. Their bytes are far over
/// what a loopback send path takes, so the last send still waits for room at the wake.
const queued_sends = 1024;
const send_chunk = [_]u8{'x'} ** (64 * 1024);

const Pair = struct {
    server: net.Server,
    client: ?net.Stream,
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

    fn closeClient(p: *Pair, io: Io) void {
        if (p.client) |client| {
            client.close(io);
        }
        p.client = null;
    }

    fn close(p: *Pair, io: Io) void {
        p.conn.close(io);
        p.closeClient(io);
        p.server.deinit(io);
    }
};

fn deadlineAfter(io: Io, wait_ms: u32) Io.Clock.Timestamp {
    const now = Io.Clock.Timestamp.now(io, .awake);
    return now.addDuration(.{ .raw = .fromMilliseconds(wait_ms), .clock = .awake });
}

fn waitInput(io: Io, stream: net.Stream) Wait {
    return waitReadable(io, stream, deadlineAfter(io, event_wait_ms));
}

fn sendAll(io: Io, stream: net.Stream, bytes: []const u8) !void {
    const n = write(io, stream, "", &.{bytes}, 1) orelse return error.WriteFailed;
    try testing.expectEqual(bytes.len, n);
}

test "socket_wait_windows: the wait for input ends at its deadline and consumes no input" {
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    const deadline = deadlineAfter(io, silent_wait_ms);
    try testing.expectEqual(Wait.timed_out, waitReadable(io, p.conn, deadline));
    // A wait that reports the deadline must not end before it.
    try testing.expect(deadline.durationFromNow(io).raw.nanoseconds <= 0);
    try testing.expectEqual(Wait.timed_out, waitReadable(io, p.conn, deadline));

    // The end of the stream after the bytes keeps the read below from a wait. So a wait for input
    // that consumed a byte fails the test and does not hang it.
    try sendAll(io, p.client.?, "ab");
    try p.client.?.shutdown(io, .send);
    try testing.expectEqual(Wait.ready, waitInput(io, p.conn));
    try testing.expectEqual(Wait.ready, waitInput(io, p.conn));

    var buf: [8]u8 = undefined;
    var dest = [_][]u8{&buf};
    try testing.expectEqual(@as(?usize, 2), read(io, p.conn, &dest));
    try testing.expectEqualStrings("ab", buf[0..2]);
}

test "socket_wait_windows: an end of stream makes the wait for input ready, and a reset fails it" {
    const io = testing.io;
    var buf: [8]u8 = undefined;
    var dest = [_][]u8{&buf};
    {
        var p = try Pair.open(io);
        defer p.close(io);

        try p.client.?.shutdown(io, .send);
        try testing.expectEqual(Wait.ready, waitInput(io, p.conn));
        try testing.expectEqual(@as(?usize, 0), read(io, p.conn, &dest));
    }
    {
        var p = try Pair.open(io);
        defer p.close(io);

        p.closeClient(io);
        try testing.expectEqual(Wait.ready, waitInput(io, p.conn));
        try testing.expectEqual(@as(?usize, 0), read(io, p.conn, &dest));
    }
    {
        var p = try Pair.open(io);
        defer p.close(io);

        // A close over input that the client did not read resets the connection.
        try sendAll(io, p.conn, "unread");
        const client = p.client.?;
        try testing.expectEqual(Wait.ready, waitInput(io, client));
        p.closeClient(io);
        try testing.expectEqual(Wait.failed, waitInput(io, p.conn));
    }
}

test "socket_wait_windows: a wake ends a wait for input in both orders" {
    const io = testing.io;
    {
        var p = try Pair.open(io);
        defer p.close(io);

        wake(io, p.conn);
        try testing.expectEqual(Wait.failed, waitInput(io, p.conn));
    }
    {
        var p = try Pair.open(io);
        defer p.close(io);

        // On an idle pair `PENDING` proves that the request waits in the kernel when the wake
        // comes. No `try` comes between the start and the finish, because the kernel holds
        // pointers into `input` until the routine ran.
        const handle = p.conn.socket.handle;
        var input: TimedRequest = undefined;
        const status = input.startReceive(handle);
        if (status != .PENDING and status != .SUCCESS) {
            std.debug.assert(isErrorStatus(status));
            return error.ReceiveRefused;
        }
        wake(io, p.conn);
        const end = input.finishReceive(io, handle, deadlineAfter(io, event_wait_ms));
        try testing.expectEqual(windows.NTSTATUS.PENDING, status);
        try testing.expectEqual(Wait.failed, end);
    }
}

/// The states of the peer that the hang-up probe must tell apart.
const PeerState = enum { idle, bytes, shut, closed, bytes_shut, bytes_closed, reset, bytes_reset };

/// Bring the peer into `state`, and wait for the event of that state.
fn enterPeerState(io: Io, p: *Pair, state: PeerState) !void {
    const client = p.client.?;
    const has_input = switch (state) {
        .bytes, .bytes_shut, .bytes_closed, .bytes_reset => true,
        .idle, .shut, .closed, .reset => false,
    };
    if (has_input) {
        try sendAll(io, client, "GET / HTTP/1.1\r\n");
    }
    switch (state) {
        .idle, .bytes => {},
        .shut, .bytes_shut => try client.shutdown(io, .send),
        .closed, .bytes_closed => p.closeClient(io),
        .reset, .bytes_reset => {
            // A close over input that the client did not read resets the connection.
            try sendAll(io, p.conn, "unread");
            try testing.expectEqual(Wait.ready, waitInput(io, client));
            p.closeClient(io);
        },
    }

    const deadline = deadlineAfter(io, event_wait_ms);
    switch (state) {
        .idle => {},
        .bytes => try testing.expectEqual(Wait.ready, waitReadable(io, p.conn, deadline)),
        else => try testing.expect(pollHangUp(io, p.conn, deadline)),
    }
}

fn pollHangUp(io: Io, stream: net.Stream, deadline: Io.Clock.Timestamp) bool {
    return pollSocket(io, stream, hang_up_events, hundredNanosUntil(io, deadline));
}

/// Drain on a task of its own. A drain that waits then fails the test at `event_wait_ms`, and the
/// wake ends it.
fn drainWithin(io: Io, stream: net.Stream) !void {
    var done: Io.Event = .unset;
    var task = try io.concurrent(drainThenSet, .{ io, stream, &done });
    const deadline = deadlineAfter(io, event_wait_ms);
    // A wait can end with no event, so the loop reads the event again. The deadline ends the loop.
    while (!done.isSet() and deadline.durationFromNow(io).raw.nanoseconds > 0) {
        done.waitTimeout(io, .{ .deadline = deadline }) catch |err| switch (err) {
            error.Timeout => {},
            error.Canceled => |e| {
                wake(io, stream);
                task.await(io);
                return e;
            },
        };
    }

    const drained = done.isSet();
    if (!drained) {
        wake(io, stream);
    }
    task.await(io);
    try testing.expect(drained);
}

fn drainThenSet(io: Io, stream: net.Stream, done: *Io.Event) void {
    _ = drainUntil(io, stream, drain_chunk_bytes, .now(io, .awake));
    done.set(io);
}

test "socket_wait_windows: the probe tells a hang-up from waiting input in eight peer states" {
    const io = testing.io;
    for (std.enums.values(PeerState)) |state| {
        var p = try Pair.open(io);
        defer p.close(io);

        try enterPeerState(io, &p, state);
        const hung_up = switch (state) {
            .idle, .bytes => false,
            .shut, .closed, .bytes_shut, .bytes_closed, .reset, .bytes_reset => true,
        };
        try testing.expectEqual(hung_up, peerClosed(io, p.conn));
        if (!hung_up) {
            const short = deadlineAfter(io, silent_wait_ms);
            try testing.expect(!pollHangUp(io, p.conn, short));
        }

        // The drain never waits, and it leaves no input behind.
        try drainWithin(io, p.conn);
        const after_drain = waitReadable(io, p.conn, deadlineAfter(io, silent_wait_ms));
        const want: Wait = switch (state) {
            .idle, .bytes => .timed_out,
            .shut, .closed, .bytes_shut, .bytes_closed => .ready,
            .reset, .bytes_reset => .failed,
        };
        try testing.expectEqual(want, after_drain);
    }
}

/// A boolean option fills one byte of the value.
fn getOption(io: Io, stream: net.Stream, level: i32, name: u32) !u32 {
    var value: u32 = 0;
    const info: AFD.SOCKOPT_INFO = .{
        .mode = .get,
        .level = level,
        .optname = name,
        .optval = &value,
        .optlen = @sizeOf(u32),
    };
    const result = try io.operate(.{ .device_io_control = .{
        .file = .{ .handle = stream.socket.handle, .flags = .{ .nonblocking = true } },
        .code = windows.IOCTL.AFD.SOCKOPT,
        .in = std.mem.asBytes(&info),
        .out = std.mem.asBytes(&value),
    } });
    try testing.expectEqual(windows.NTSTATUS.SUCCESS, result.device_io_control.u.Status);
    return value;
}

test "socket_wait_windows: the keepalive options hold the values that enableKeepalive set" {
    // A refused option ends the call with no report, so only a read of each option shows it.
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    enableKeepalive(io, p.conn, 60, 10, 6);
    const ws2_32 = windows.ws2_32;
    const tcp = ws2_32.IPPROTO.TCP;
    try testing.expectEqual(1, try getOption(io, p.conn, ws2_32.SOL.SOCKET, ws2_32.SO.KEEPALIVE));
    try testing.expectEqual(60, try getOption(io, p.conn, tcp, ws2_32.TCP.KEEPALIVE));
    try testing.expectEqual(10, try getOption(io, p.conn, tcp, ws2_32.TCP.KEEPINTVL));
    try testing.expectEqual(6, try getOption(io, p.conn, tcp, ws2_32.TCP.KEEPCNT));
}

/// Take the cancel of the test in a wait, call `io.recancel`, and run the keepalive requests. The
/// next cancel point must still return `error.Canceled`.
fn keepaliveUnderCancel(io: Io, stream: net.Stream) Io.Event.WaitTimeoutError!void {
    var never: Io.Event = .unset;
    // Only the cancel of the test ends this wait.
    never.wait(io) catch io.recancel();
    enableKeepalive(io, stream, 60, 10, 6);

    const bound: Io.Timeout = .{ .duration = .{
        .raw = .fromMilliseconds(event_wait_ms),
        .clock = .awake,
    } };
    return never.waitTimeout(io, bound);
}

test "socket_wait_windows: a task cancel that a keepalive request takes still ends the task" {
    // The acceptor turns on the keepalive outside its cancel protection. A stop that lands there
    // must still end the next accept.
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    var task = try io.concurrent(keepaliveUnderCancel, .{ io, p.conn });
    try testing.expectError(error.Canceled, task.cancel(io));
}

/// How the queued sends of one test ended.
const SendEnds = struct {
    sent: usize = 0,
    cancelled: usize = 0,
    timed_out: usize = 0,
    other: usize = 0,
};

test "socket_wait_windows: a wake ends a send that waits for room" {
    // A send that found no room does not prove a full path, because room can come back at once.
    // So no send here has a deadline, and all of them queue on the socket. The peer reads nothing.
    // The path takes the first sends, and the last one waits for room until the wake.
    const io = testing.io;
    var p = try Pair.open(io);
    defer p.close(io);

    const sends = try testing.allocator.alloc(TimedRequest, queued_sends);
    defer testing.allocator.free(sends);
    const handle = p.conn.socket.handle;
    // The kernel holds pointers into a started request until its routine ran. So no path returns
    // between the first start and the last finish.
    var started: usize = 0;
    for (sends) |*send| {
        if (isErrorStatus(send.startSend(handle, &send_chunk))) break;
        started += 1;
    }

    wake(io, p.conn);
    const deadline = deadlineAfter(io, event_wait_ms);
    var ends: SendEnds = .{};
    for (sends[0..started]) |*send| {
        switch (send.finishSend(io, handle, deadline)) {
            .sent => ends.sent += 1,
            .timed_out => ends.timed_out += 1,
            .failed => if (send.iosb.u.Status == .CANCELLED) {
                ends.cancelled += 1;
            } else {
                ends.other += 1;
            },
        }
    }

    errdefer std.debug.print(
        "{d} of {d} sends started: {d} sent, {d} cancelled, {d} at the deadline, {d} other\n",
        .{ started, queued_sends, ends.sent, ends.cancelled, ends.timed_out, ends.other },
    );
    try testing.expectEqual(queued_sends, started);
    // A send at the deadline waited through the wake.
    try testing.expectEqual(0, ends.timed_out);
    try testing.expectEqual(windows.NTSTATUS.CANCELLED, sends[started - 1].iosb.u.Status);
}
