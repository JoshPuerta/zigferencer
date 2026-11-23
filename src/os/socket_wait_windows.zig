//! The serve socket calls on Windows. The read and the write send one `AFD.RECEIVE` or one
//! `AFD.SEND` request through `io.operate`, and the wake cancels every request of the socket. The
//! deadline, the probe, the keepalive, and the drains keep the bodies of `socket_wait_none.zig`.
//!
//! Wake. A shutdown alone does not end a pending receive or send on Windows. A pending receive
//! then waits for the TCP timer of the half-closed connection, and a pending send waits with no
//! bound. So the wake also cancels the requests of the socket. The status of a cancelled request
//! comes back to this file, which turns it into a failed call.
//!
//! Requests. Std reaches `unreachable` when a cancel ends its own socket read, write, or accept.
//! So every read and write of a connection socket must be a request of this file.

const std = @import("std");
const none = @import("socket_wait_none.zig");
const kernel32 = @import("windows_kernel32_declarations.zig");
const Io = std.Io;
const net = Io.net;
const windows = std.os.windows;
const AFD = windows.AFD;

// -- constants --------------------------------------------------------------

pub const has_deadline: bool = false;

/// The buffer count of one receive. Std uses the same bound for its own socket read.
const read_buffers_max = Io.Threaded.max_iovecs_len;

/// The flags that std passes with each socket request.
const afd_flags: AFD = .{ .NO_FAST_IO = true, .OVERLAPPED = true };

// -- api --------------------------------------------------------------------

pub const waitReadable = none.waitReadable;
pub const waitWritable = none.waitWritable;
pub const sendNow = none.sendNow;
pub const drainNow = none.drainNow;
pub const drainUntil = none.drainUntil;
pub const peerClosed = none.peerClosed;
pub const enableKeepalive = none.enableKeepalive;

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

    // A receive with no buffer succeeds with 0 bytes, which reads as the end of the stream.
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
    stream.shutdown(io, .both) catch {};
    // FALSE with no request to cancel is the normal case, so the result needs no check.
    _ = kernel32.CancelIoEx(stream.socket.handle, null);
}

// -- helpers ----------------------------------------------------------------

/// Every status other than success fails the call, a cancel included.
fn request(io: Io, stream: net.Stream, code: windows.CTL_CODE, in: []const u8) ?usize {
    const result = io.operate(.{ .device_io_control = .{
        .file = .{ .handle = stream.socket.handle, .flags = .{ .nonblocking = true } },
        .code = code,
        .in = in,
    } }) catch return null;
    const iosb = result.device_io_control;
    if (iosb.u.Status != .SUCCESS) return null;

    return iosb.Information;
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
