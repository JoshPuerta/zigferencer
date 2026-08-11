//! The ZMTP 3.1 codec for a ZMQ peer of the serve layer. A ZMQ DEALER socket
//! connects to the server, and the server answers as a ROUTER. This module
//! encodes and decodes the greeting, the frame head, and the commands of the
//! NULL handshake and the heartbeat. ZMTP integers are big-endian, unlike the
//! little-endian `protocol.zig` frames that the message frames carry.
//!
//! Each peer opens with a 64-byte greeting:
//!
//!     off  len  field      rule
//!     0    1    signature  0xFF
//!     1    8    padding    zeros on write, ignored on read
//!     9    1    signature  0x7F
//!     10   1    major      3, a ZMTP 2.0 peer puts 0x01 here and is refused
//!     11   1    minor      1 on write, any value on read
//!     12   20   security   "NULL", padded with zero bytes
//!     32   1    as-server  0 on write, ignored on read
//!     33   31   filler     zeros on write, ignored on read
//!
//! Every frame then starts with a 2-byte or 9-byte head:
//!
//!     off  len  field  rule
//!     0    1    flags  bit 0 MORE, bit 1 LONG, bit 2 COMMAND, bits 3-7 zero
//!     1    1    size   when LONG is clear, a writer uses it up to 255
//!     1    8    size   when LONG is set, big-endian
//!
//! The size counts the frame body and excludes the head. A command frame
//! never sets MORE. A command body is a name length byte, a name of 1 to 255
//! bytes, and the command data:
//!
//!     READY  properties, each a name length byte, the name, a 4-byte
//!            big-endian value length, and the value
//!     ERROR  a reason length byte and the reason
//!     PING   a 2-byte big-endian ttl and a context of up to 16 bytes
//!     PONG   the context of the PING that it answers
//!
//! Property names compare without regard to letter case, and a decode skips
//! an unknown property. A command with another name decodes to `.other`.
//!
//! This module is pure. It imports no I/O and it allocates nothing. Decoded
//! commands borrow their slices from the body, and the caller copies what it
//! keeps. Every decode checks every length and returns a typed error. The
//! socket is a trust boundary, so a decode never panics on input.

const std = @import("std");
const protocol = @import("protocol.zig");

// -- types ------------------------------------------------------------------

pub const greeting_bytes: usize = 64;

/// Ceiling on a READY command body. The READY of a libzmq DEALER holds 41
/// bytes. The ceiling leaves room for more properties and bounds the property
/// walk.
pub const max_ready_bytes: usize = 1024;

/// ZMTP 3.1 caps the PING context at 16 bytes.
pub const max_ping_context_bytes: usize = 16;

/// A one-byte length prefixes the ERROR reason.
pub const max_error_reason_bytes: usize = 255;

/// Ceiling on a message frame body, the size of the largest `protocol.zig`
/// frame.
pub const max_message_bytes: usize = protocol.header_bytes + protocol.max_frame_payload_bytes;

pub const socket_type_router = "ROUTER";
pub const socket_type_dealer = "DEALER";

/// `decodeCommand` returns `MessageTooLarge` for a READY body above
/// `max_ready_bytes`. The caller maps a frame size above `max_message_bytes`
/// to `MessageTooLarge` and a false `peerAccepted` to `PeerTypeRefused`.
pub const Error = error{
    BadSignature,
    UnsupportedVersion,
    MechanismUnsupported,
    Malformed,
    PeerTypeRefused,
    MessageTooLarge,
};

pub const FrameHead = struct {
    more: bool,
    long: bool,
    command: bool,
    size: u64,
};

/// `socket_type` borrows the body. It is empty when the peer sends no
/// Socket-Type property.
pub const Ready = struct {
    socket_type: []const u8,
    has_identity: bool,
};

/// `context` borrows the body.
pub const Ping = struct {
    ttl: u16,
    context: []const u8,
};

/// `pong` holds the context and `err` holds the reason. Both borrow the body.
pub const Command = union(enum) {
    ready: Ready,
    ping: Ping,
    pong: []const u8,
    err: []const u8,
    other,
};

const signature_first: u8 = 0xFF;
const signature_last: u8 = 0x7F;
const signature_last_offset: usize = 9;
const major_offset: usize = 10;
const minor_offset: usize = 11;
const security_offset: usize = 12;
const security_bytes: usize = 20;
const version_major: u8 = 3;
const version_minor: u8 = 1;
const security_null: [security_bytes]u8 = "NULL".* ++ [_]u8{0} ** (security_bytes - 4);

const flag_more: u8 = 0x01;
const flag_long: u8 = 0x02;
const flag_command: u8 = 0x04;
const flags_reserved: u8 = 0xF8;
const short_head_bytes: usize = 2;
const long_head_bytes: usize = 9;
const short_size_max: u64 = std.math.maxInt(u8);

const command_ready = "READY";
const command_error = "ERROR";
const command_ping = "PING";
const command_pong = "PONG";
const property_socket_type = "Socket-Type";
const property_identity = "Identity";

/// A name length byte, a name of one byte, and a 4-byte value length.
const min_property_bytes: usize = 1 + 1 + 4;
const ready_fixed_bytes: usize = 1 + command_ready.len + 1 + property_socket_type.len + 4;

/// Bounds-checked big-endian reader over a command body. A `take` past the
/// end returns `Malformed`, so a body parses exactly or not at all.
const Cursor = struct {
    buf: []const u8,
    i: usize = 0,

    fn take(c: *Cursor, comptime T: type) Error!T {
        std.debug.assert(c.i <= c.buf.len);
        const n = @sizeOf(T);
        if (c.buf.len - c.i < n) return error.Malformed;
        defer c.i += n;
        return std.mem.readInt(T, c.buf[c.i..][0..n], .big);
    }

    fn takeSlice(c: *Cursor, len: u64) Error![]const u8 {
        std.debug.assert(c.i <= c.buf.len);
        const remaining: u64 = c.buf.len - c.i;
        if (len > remaining) return error.Malformed;
        const n: usize = @intCast(len);
        defer c.i += n;
        return c.buf[c.i..][0..n];
    }

    fn takeRest(c: *Cursor) []const u8 {
        std.debug.assert(c.i <= c.buf.len);
        defer c.i = c.buf.len;
        return c.buf[c.i..];
    }

    fn done(c: *const Cursor) bool {
        std.debug.assert(c.i <= c.buf.len);
        return c.i == c.buf.len;
    }
};

/// Asserting big-endian writer. The caller sizes every encode buffer, so an
/// overflow is a programmer error and not a runtime one.
const Writer = struct {
    buf: []u8,
    i: usize = 0,

    fn put(w: *Writer, comptime T: type, v: T) void {
        const n = @sizeOf(T);
        std.debug.assert(w.buf.len - w.i >= n);
        std.mem.writeInt(T, w.buf[w.i..][0..n], v, .big);
        w.i += n;
    }

    fn putSlice(w: *Writer, bytes: []const u8) void {
        std.debug.assert(w.buf.len - w.i >= bytes.len);
        @memcpy(w.buf[w.i..][0..bytes.len], bytes);
        w.i += bytes.len;
    }

    fn putShort(w: *Writer, bytes: []const u8) void {
        std.debug.assert(bytes.len <= std.math.maxInt(u8));
        w.put(u8, @intCast(bytes.len));
        w.putSlice(bytes);
    }
};

// -- api --------------------------------------------------------------------

pub fn encodeGreeting(out: *[greeting_bytes]u8) void {
    @memset(out, 0);
    out[0] = signature_first;
    out[signature_last_offset] = signature_last;
    out[major_offset] = version_major;
    out[minor_offset] = version_minor;
    out[security_offset..][0..security_bytes].* = security_null;
}

/// The padding, the minor version, the as-server byte, and the filler take
/// any value, so a peer with a later 3.x minor version passes.
pub fn decodeGreeting(bytes: *const [greeting_bytes]u8) Error!void {
    if (bytes[0] != signature_first) return error.BadSignature;
    if (bytes[signature_last_offset] != signature_last) return error.BadSignature;
    if (bytes[major_offset] != version_major) return error.UnsupportedVersion;
    const security = bytes[security_offset..][0..security_bytes];
    if (!std.mem.eql(u8, security, &security_null)) return error.MechanismUnsupported;
}

pub fn headLen(flags: u8) usize {
    return if (flags & flag_long != 0) long_head_bytes else short_head_bytes;
}

/// Decodes the head at the start of `bytes`. The caller reads
/// `headLen(bytes[0])` bytes first, and a shorter slice returns `Malformed`.
/// The caller also bounds `size`, because the right ceiling depends on the
/// frame kind.
pub fn decodeFrameHead(bytes: []const u8) Error!FrameHead {
    if (bytes.len == 0) return error.Malformed;
    const flags = bytes[0];
    if (flags & flags_reserved != 0) return error.Malformed;
    const n = headLen(flags);
    if (bytes.len < n) return error.Malformed;
    const head: FrameHead = .{
        .more = flags & flag_more != 0,
        .long = flags & flag_long != 0,
        .command = flags & flag_command != 0,
        .size = if (n == long_head_bytes) std.mem.readInt(u64, bytes[1..9], .big) else bytes[1],
    };
    if (head.command and head.more) return error.Malformed;
    return head;
}

/// Writes the head of a frame with a body of `size` bytes and returns the
/// head length. The head never sets MORE.
pub fn encodeFrameHead(size: u64, command: bool, out: *[long_head_bytes]u8) usize {
    std.debug.assert(size <= max_message_bytes);
    const kind: u8 = if (command) flag_command else 0;
    const n: usize = if (size <= short_size_max) blk: {
        out[0] = kind;
        out[1] = @intCast(size);
        break :blk short_head_bytes;
    } else blk: {
        out[0] = kind | flag_long;
        std.mem.writeInt(u64, out[1..9], size, .big);
        break :blk long_head_bytes;
    };
    std.debug.assert(headLen(out[0]) == n);
    return n;
}

/// Decodes a command body, the bytes after the frame head. Slices borrow from
/// `body`. An unknown name returns `.other`. Trailing bytes after a known
/// command return `Malformed`.
pub fn decodeCommand(body: []const u8) Error!Command {
    var c: Cursor = .{ .buf = body };
    const name = try c.takeSlice(try c.take(u8));
    if (name.len == 0) return error.Malformed;
    if (std.mem.eql(u8, name, command_ready)) {
        if (body.len > max_ready_bytes) return error.MessageTooLarge;
        return .{ .ready = try takeReady(&c) };
    }
    const command: Command = if (std.mem.eql(u8, name, command_ping))
        .{ .ping = .{ .ttl = try c.take(u16), .context = try takeContext(&c) } }
    else if (std.mem.eql(u8, name, command_pong))
        .{ .pong = try takeContext(&c) }
    else if (std.mem.eql(u8, name, command_error))
        .{ .err = try c.takeSlice(try c.take(u8)) }
    else
        return .other;
    if (!c.done()) return error.Malformed;
    return command;
}

/// Writes a READY body with the Socket-Type property alone and returns its
/// length. The body excludes the frame head.
pub fn encodeReady(socket_type: []const u8, out: []u8) usize {
    const len = ready_fixed_bytes + socket_type.len;
    std.debug.assert(len <= max_ready_bytes);
    std.debug.assert(out.len >= len);
    var w: Writer = .{ .buf = out };
    w.putShort(command_ready);
    w.putShort(property_socket_type);
    w.put(u32, @intCast(socket_type.len));
    w.putSlice(socket_type);
    std.debug.assert(w.i == len);
    return w.i;
}

/// Writes a PONG body that echoes `context` and returns its length.
pub fn encodePong(context: []const u8, out: *[32]u8) usize {
    std.debug.assert(context.len <= max_ping_context_bytes);
    var w: Writer = .{ .buf = out };
    w.putShort(command_pong);
    w.putSlice(context);
    std.debug.assert(w.i == 1 + command_pong.len + context.len);
    return w.i;
}

/// Writes an ERROR body and returns its length.
pub fn encodeError(reason: []const u8, out: []u8) usize {
    std.debug.assert(reason.len <= max_error_reason_bytes);
    std.debug.assert(out.len >= 1 + command_error.len + 1 + reason.len);
    var w: Writer = .{ .buf = out };
    w.putShort(command_error);
    w.putShort(reason);
    return w.i;
}

/// True for a DEALER in any letter case, and for a peer that sends no
/// Socket-Type property.
pub fn peerAccepted(socket_type: []const u8) bool {
    if (socket_type.len == 0) return true;
    return std.ascii.eqlIgnoreCase(socket_type, socket_type_dealer);
}

// -- helpers ----------------------------------------------------------------

fn takeReady(c: *Cursor) Error!Ready {
    std.debug.assert(c.buf.len <= max_ready_bytes);
    var ready: Ready = .{ .socket_type = "", .has_identity = false };
    var has_socket_type = false;
    var properties: usize = 0;
    // Each pass consumes at least `min_property_bytes`, so the body bound
    // caps the passes at max_ready_bytes / min_property_bytes.
    while (!c.done()) : (properties += 1) {
        std.debug.assert(properties <= max_ready_bytes / min_property_bytes);
        const name = try c.takeSlice(try c.take(u8));
        if (name.len == 0) return error.Malformed;
        const value = try c.takeSlice(try c.take(u32));
        // A repeated property has no single meaning, so the decode refuses it.
        if (std.ascii.eqlIgnoreCase(name, property_socket_type)) {
            if (has_socket_type) return error.Malformed;
            has_socket_type = true;
            ready.socket_type = value;
        } else if (std.ascii.eqlIgnoreCase(name, property_identity)) {
            if (ready.has_identity) return error.Malformed;
            ready.has_identity = true;
        }
    }
    return ready;
}

fn takeContext(c: *Cursor) Error![]const u8 {
    const context = c.takeRest();
    if (context.len > max_ping_context_bytes) return error.Malformed;
    return context;
}

// -- tests ------------------------------------------------------------------

/// The READY command frame that a libzmq 4.3 DEALER sends, head included.
const dealer_ready_frame: [43]u8 = ("\x04\x29" ++
    "\x05READY" ++
    "\x0bSocket-Type\x00\x00\x00\x06DEALER" ++
    "\x08Identity\x00\x00\x00\x00").*;

/// The body of `dealer_ready_frame` up to the Identity property.
const socket_type_only_bytes: usize = 28;

fn decodeFrame(frame: []const u8) Error!Command {
    const head = try decodeFrameHead(frame);
    const n = headLen(frame[0]);
    const available: u64 = frame.len - n;
    const body_len: usize = @intCast(@min(head.size, available));
    return decodeCommand(frame[n..][0..body_len]);
}

fn borrowsFrom(frame: []const u8, command: Command) bool {
    const slice: []const u8 = switch (command) {
        .ready => |r| r.socket_type,
        .ping => |p| p.context,
        .pong, .err => |s| s,
        .other => return true,
    };
    if (slice.len == 0) return true;
    const lo = @intFromPtr(frame.ptr);
    const at = @intFromPtr(slice.ptr);
    return at >= lo and at + slice.len <= lo + frame.len;
}

test "greeting round trip" {
    var greeting: [greeting_bytes]u8 = undefined;
    encodeGreeting(&greeting);
    const expected = [_]u8{0xFF} ++ [_]u8{0} ** 8 ++ [_]u8{ 0x7F, 3, 1 } ++
        [_]u8{ 'N', 'U', 'L', 'L' } ++ [_]u8{0} ** 48;
    try std.testing.expectEqualSlices(u8, &expected, &greeting);
    try decodeGreeting(&greeting);
}

test "greeting guards: signature, major version, security method" {
    var good: [greeting_bytes]u8 = undefined;
    encodeGreeting(&good);

    var bad = good;
    bad[0] = 0xFE;
    try std.testing.expectError(error.BadSignature, decodeGreeting(&bad));

    bad = good;
    bad[9] = 0x7E;
    try std.testing.expectError(error.BadSignature, decodeGreeting(&bad));

    // A ZMTP 2.0 peer sends revision 0x01 at byte 10.
    bad = good;
    bad[10] = 0x01;
    try std.testing.expectError(error.UnsupportedVersion, decodeGreeting(&bad));

    bad = good;
    bad[10] = 4;
    try std.testing.expectError(error.UnsupportedVersion, decodeGreeting(&bad));

    bad = good;
    @memcpy(bad[12..17], "PLAIN");
    try std.testing.expectError(error.MechanismUnsupported, decodeGreeting(&bad));

    bad = good;
    bad[16] = 'X';
    try std.testing.expectError(error.MechanismUnsupported, decodeGreeting(&bad));

    var lax = good;
    lax[1] = 0xAA;
    lax[11] = 0;
    lax[32] = 1;
    lax[63] = 0x55;
    try decodeGreeting(&lax);
}

test "frame head round trip at the short and long boundaries" {
    const sizes = [_]u64{ 0, 1, 255, 256, 65536, max_message_bytes };
    for (sizes) |size| {
        for ([_]bool{ false, true }) |command| {
            var out: [long_head_bytes]u8 = undefined;
            const n = encodeFrameHead(size, command, &out);
            try std.testing.expectEqual(size > 255, out[0] & 0x02 != 0);
            try std.testing.expectEqual(@as(usize, if (size <= 255) 2 else 9), n);
            const head = try decodeFrameHead(out[0..n]);
            try std.testing.expectEqual(FrameHead{
                .more = false,
                .long = size > 255,
                .command = command,
                .size = size,
            }, head);
        }
    }

    var out: [long_head_bytes]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 2), encodeFrameHead(255, false, &out));
    try std.testing.expectEqualSlices(u8, &.{ 0x00, 0xFF }, out[0..2]);
    try std.testing.expectEqual(@as(usize, 9), encodeFrameHead(256, true, &out));
    try std.testing.expectEqualSlices(u8, &.{ 0x06, 0, 0, 0, 0, 0, 0, 0x01, 0x00 }, &out);

    const long_small = [_]u8{ flag_long, 0, 0, 0, 0, 0, 0, 0, 5 };
    try std.testing.expectEqual(@as(u64, 5), (try decodeFrameHead(&long_small)).size);
}

test "frame head guards: reserved bits, command with MORE, short buffer" {
    for (3..8) |bit| {
        const shift: u3 = @intCast(bit);
        const head = [_]u8{ @as(u8, 1) << shift, 0 };
        try std.testing.expectError(error.Malformed, decodeFrameHead(&head));
    }
    const command_more = [_]u8{ flag_command | flag_more, 0 };
    try std.testing.expectError(error.Malformed, decodeFrameHead(&command_more));
    try std.testing.expectError(error.Malformed, decodeFrameHead(&[_]u8{}));
    try std.testing.expectError(error.Malformed, decodeFrameHead(&.{0x00}));
    try std.testing.expectError(error.Malformed, decodeFrameHead(&.{ flag_long, 0, 0, 0, 0 }));

    const more = try decodeFrameHead(&.{ flag_more, 3 });
    try std.testing.expect(more.more and !more.command);
    try std.testing.expectEqual(@as(u64, 3), more.size);
}

test "the libzmq DEALER READY frame decodes to DEALER with an identity" {
    const body_len = (1 + command_ready.len) +
        (1 + property_socket_type.len + 4 + socket_type_dealer.len) +
        (1 + property_identity.len + 4);
    try std.testing.expectEqual(@as(usize, 41), body_len);
    try std.testing.expectEqual(dealer_ready_frame.len, short_head_bytes + body_len);

    const head = try decodeFrameHead(&dealer_ready_frame);
    try std.testing.expectEqual(FrameHead{
        .more = false,
        .long = false,
        .command = true,
        .size = body_len,
    }, head);

    const command = try decodeCommand(dealer_ready_frame[short_head_bytes..]);
    try std.testing.expect(command == .ready);
    try std.testing.expectEqualStrings("DEALER", command.ready.socket_type);
    try std.testing.expect(command.ready.has_identity);
    try std.testing.expect(peerAccepted(command.ready.socket_type));
}

const CommandWant = union(enum) {
    err: Error,
    other,
    ready: struct { socket_type: []const u8, has_identity: bool },
};

const CommandCase = struct { name: []const u8, body: []const u8, want: CommandWant };

fn checkCommand(c: CommandCase) !void {
    switch (c.want) {
        .err => |err| try std.testing.expectError(err, decodeCommand(c.body)),
        .other => try std.testing.expect((try decodeCommand(c.body)) == .other),
        .ready => |r| {
            const command = try decodeCommand(c.body);
            try std.testing.expect(command == .ready);
            try std.testing.expectEqualStrings(r.socket_type, command.ready.socket_type);
            try std.testing.expectEqual(r.has_identity, command.ready.has_identity);
        },
    }
}

/// A READY body of exactly `len` bytes whose single property carries the padding.
fn paddedReady(comptime len: usize) [len]u8 {
    const fixed = "\x05READY" ++ "\x01X";
    return (fixed ++ std.mem.toBytes(std.mem.nativeToBig(u32, len - fixed.len - 4)) ++ ("v" ** (len - fixed.len - 4))).*;
}

test "commands: every malformed body is refused with its error, and the edges decode" {
    const malformed: CommandWant = .{ .err = error.Malformed };
    const body = dealer_ready_frame[short_head_bytes..];
    const ready_at_bound = paddedReady(max_ready_bytes);
    const ready_past_bound = paddedReady(max_ready_bytes + 1);
    const cases = [_]CommandCase{
        .{ .name = "empty body", .body = "", .want = malformed },
        .{ .name = "empty name", .body = "\x00", .want = malformed },
        .{ .name = "short name", .body = "\x05READ", .want = malformed },
        .{ .name = "ERROR with no reason length", .body = "\x05ERROR", .want = malformed },
        .{ .name = "ERROR reason cut", .body = "\x05ERROR\x03ab", .want = malformed },
        .{ .name = "ERROR trailing byte", .body = "\x05ERROR\x02ab!", .want = malformed },
        .{ .name = "unknown name", .body = "\x09SUBSCRIBEtopic", .want = .other },
        .{ .name = "name case matters", .body = "\x05ready", .want = .other },
        .{ .name = "PING with no ttl", .body = "\x04PING", .want = malformed },
        .{ .name = "PING with a short ttl", .body = "\x04PING\x00", .want = malformed },
        .{ .name = "PING context past the bound", .body = "\x04PING\x00\x0a" ++ "x" ** 17, .want = malformed },
        .{ .name = "PONG context past the bound", .body = "\x04PONG" ++ "x" ** 17, .want = malformed },
        .{ .name = "READY cut inside a value length", .body = body[0..20], .want = malformed },
        .{ .name = "READY value past the end", .body = body[0 .. body.len - 1] ++ [_]u8{1}, .want = malformed },
        .{ .name = "READY empty property name", .body = "\x05READY\x00\x00\x00\x00\x00", .want = malformed },
        .{ .name = "Socket-Type twice", .body = body[0..socket_type_only_bytes] ++ body[6..socket_type_only_bytes], .want = malformed },
        .{ .name = "Identity twice", .body = body ++ body[socket_type_only_bytes..], .want = malformed },
        .{ .name = "READY at the size bound", .body = &ready_at_bound, .want = .{ .ready = .{ .socket_type = "", .has_identity = false } } },
        .{ .name = "READY past the size bound", .body = &ready_past_bound, .want = .{ .err = error.MessageTooLarge } },
        .{ .name = "READY with only Socket-Type", .body = body[0..socket_type_only_bytes], .want = .{ .ready = .{ .socket_type = "DEALER", .has_identity = false } } },
        .{ .name = "READY with no property", .body = "\x05READY", .want = .{ .ready = .{ .socket_type = "", .has_identity = false } } },
        .{
            .name = "READY skips an unknown property and ignores name case",
            .body = "\x05READY" ++
                "\x08Resource\x00\x00\x00\x04/zmq" ++
                "\x0bsocket-type\x00\x00\x00\x06dealer" ++
                "\x08IDENTITY\x00\x00\x00\x02id",
            .want = .{ .ready = .{ .socket_type = "dealer", .has_identity = true } },
        },
    };
    for (cases) |c| {
        errdefer std.debug.print("command case: {s}\n", .{c.name});
        try checkCommand(c);
    }
}

test "a READY body prefix decodes only at a property boundary" {
    const body = dealer_ready_frame[short_head_bytes..];
    for (0..body.len + 1) |len| {
        const boundary = len == 1 + command_ready.len or
            len == socket_type_only_bytes or len == body.len;
        if (decodeCommand(body[0..len])) |command| {
            try std.testing.expect(boundary);
            try std.testing.expect(command == .ready);
        } else |err| {
            try std.testing.expect(!boundary);
            try std.testing.expect(err == error.Malformed);
        }
    }
}

test "peerAccepted takes DEALER in any letter case and an absent socket type" {
    try std.testing.expect(peerAccepted("DEALER"));
    try std.testing.expect(peerAccepted("dealer"));
    try std.testing.expect(peerAccepted("Dealer"));
    try std.testing.expect(peerAccepted(""));
    try std.testing.expect(!peerAccepted("REQ"));
    try std.testing.expect(!peerAccepted("ROUTER"));
    try std.testing.expect(!peerAccepted("PUB"));
    try std.testing.expect(!peerAccepted("DEALERS"));
}

test "PING decode then PONG encode echoes the context" {
    const contexts = [_][]const u8{ "", "ctx-7", "0123456789abcdef" };
    for (contexts) |context| {
        var body: [32]u8 = undefined;
        var w: Writer = .{ .buf = &body };
        w.putShort(command_ping);
        w.put(u16, 300);
        w.putSlice(context);

        const ping = try decodeCommand(body[0..w.i]);
        try std.testing.expect(ping == .ping);
        try std.testing.expectEqual(@as(u16, 300), ping.ping.ttl);
        try std.testing.expectEqualSlices(u8, context, ping.ping.context);

        var out: [32]u8 = undefined;
        const n = encodePong(ping.ping.context, &out);
        try std.testing.expectEqual(1 + command_pong.len + context.len, n);
        try std.testing.expectEqualSlices(u8, "\x04PONG", out[0 .. 1 + command_pong.len]);
        try std.testing.expectEqualSlices(u8, context, out[1 + command_pong.len .. n]);

        const pong = try decodeCommand(out[0..n]);
        try std.testing.expect(pong == .pong);
        try std.testing.expectEqualSlices(u8, context, pong.pong);
    }
}

test "ERROR round trip" {
    const longest = [_]u8{'r'} ** max_error_reason_bytes;
    const reasons = [_][]const u8{ "", "peer type refused", &longest };
    for (reasons) |reason| {
        var out: [1 + command_error.len + 1 + max_error_reason_bytes]u8 = undefined;
        const n = encodeError(reason, &out);
        try std.testing.expectEqual(1 + command_error.len + 1 + reason.len, n);
        const command = try decodeCommand(out[0..n]);
        try std.testing.expect(command == .err);
        try std.testing.expectEqualSlices(u8, reason, command.err);
    }
}

test "encodeReady round trip and the libzmq byte image" {
    var body: [max_ready_bytes]u8 = undefined;
    var n = encodeReady(socket_type_router, &body);
    try std.testing.expectEqual(ready_fixed_bytes + socket_type_router.len, n);
    const command = try decodeCommand(body[0..n]);
    try std.testing.expect(command == .ready);
    try std.testing.expectEqualStrings(socket_type_router, command.ready.socket_type);
    try std.testing.expect(!command.ready.has_identity);

    var head: [long_head_bytes]u8 = undefined;
    const head_len = encodeFrameHead(n, true, &head);
    try std.testing.expectEqualSlices(u8, &.{ flag_command, @intCast(n) }, head[0..head_len]);

    n = encodeReady(socket_type_dealer, &body);
    const expected = dealer_ready_frame[short_head_bytes..][0..socket_type_only_bytes];
    try std.testing.expectEqualSlices(u8, expected, body[0..n]);
}

test "property: every single-byte mutation of the READY frame decodes or returns a typed error" {
    var decoded: u32 = 0;
    var refused: u32 = 0;
    for (0..dealer_ready_frame.len) |at| {
        for (0..256) |value| {
            var frame = dealer_ready_frame;
            frame[at] = @intCast(value);
            if (decodeFrame(&frame)) |command| {
                decoded += 1;
                try std.testing.expect(borrowsFrom(&frame, command));
            } else |_| {
                refused += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(u32, dealer_ready_frame.len * 256), decoded + refused);
    try std.testing.expect(decoded > 0);
    try std.testing.expect(refused > 0);
}

/// The decode of a frame of at most three bytes, as the head and command
/// rules state it. No known command name fits in two bytes, so a body that
/// holds its name decodes to `other`.
fn expectShortFrame(frame: []const u8) !void {
    std.debug.assert(frame.len <= 3);
    const head_ok = frame.len > 0 and frame[0] & flags_reserved == 0 and
        frame.len >= (if (frame[0] & 0x02 != 0) @as(usize, 9) else 2) and
        !(frame[0] & flag_command != 0 and frame[0] & flag_more != 0);
    if (decodeFrameHead(frame)) |head| {
        try std.testing.expect(head_ok);
        try std.testing.expect(!head.long);
        try std.testing.expectEqual(@as(u64, frame[1]), head.size);
        try std.testing.expectEqual(frame[0] & flag_more != 0, head.more);
        try std.testing.expectEqual(frame[0] & flag_command != 0, head.command);
    } else |err| {
        try std.testing.expect(!head_ok);
        try std.testing.expectEqual(error.Malformed, err);
    }
    const name_fits = frame.len > 0 and frame[0] > 0 and 1 + @as(usize, frame[0]) <= frame.len;
    if (decodeCommand(frame)) |command| {
        try std.testing.expect(name_fits);
        try std.testing.expect(command == .other);
    } else |err| {
        try std.testing.expect(!name_fits);
        try std.testing.expectEqual(error.Malformed, err);
    }
}

test "exhaustive: every frame of at most three bytes decodes as the head and command rules state" {
    var frame: [3]u8 = undefined;
    try expectShortFrame(frame[0..0]);
    for (0..256) |a| {
        frame[0] = @intCast(a);
        try expectShortFrame(frame[0..1]);
        for (0..256) |b| {
            frame[1] = @intCast(b);
            try expectShortFrame(frame[0..2]);
            for (0..256) |c| {
                frame[2] = @intCast(c);
                try expectShortFrame(frame[0..3]);
            }
        }
    }
}

test "round trip: an ERROR frame of every reason length crosses the short and long head boundary" {
    const body_max = 1 + command_error.len + 1 + max_error_reason_bytes;
    var reason: [max_error_reason_bytes]u8 = undefined;
    for (&reason, 0..) |*b, i| b.* = @intCast(i);
    var longs: u32 = 0;
    for (0..max_error_reason_bytes + 1) |len| {
        var frame: [long_head_bytes + body_max]u8 = undefined;
        var head: [long_head_bytes]u8 = undefined;
        var body: [body_max]u8 = undefined;
        const body_len = encodeError(reason[0..len], &body);
        const head_len = encodeFrameHead(body_len, true, &head);
        @memcpy(frame[0..head_len], head[0..head_len]);
        @memcpy(frame[head_len..][0..body_len], body[0..body_len]);
        const n = head_len + body_len;

        const decoded = try decodeFrameHead(frame[0..n]);
        try std.testing.expectEqual(body_len > short_size_max, decoded.long);
        try std.testing.expectEqual(@as(u64, body_len), decoded.size);
        longs += @intFromBool(decoded.long);
        const command = try decodeFrame(frame[0..n]);
        try std.testing.expect(command == .err);
        try std.testing.expectEqualSlices(u8, reason[0..len], command.err);
        // A frame cut by one byte cuts the reason, so its length no longer fits.
        try std.testing.expectError(error.Malformed, decodeFrame(frame[0 .. n - 1]));
    }
    // Bodies of 256 to 262 bytes take the long head.
    try std.testing.expectEqual(@as(u32, 1 + command_error.len + 1 + max_error_reason_bytes - short_size_max), longs);
}
