//! The body of an HTTP request: the framing that the head declares, and a strict decoder of the
//! chunked transfer coding. The std HTTP server parses the head. Its body readers accept a lax
//! chunked form, and one chunk size overflows them, so every body byte of this server passes
//! through this file instead.
//!
//! Role. `framing` reads the raw lines of a head before any body byte moves. It returns no body,
//! a body of a declared length, a chunked body, or a refusal with its status. Only the raw lines
//! count. The std head parser accepts a sign and digit separators in a length. In
//! `Transfer-Encoding` it refuses an unknown coding name, but it reads a content-coding name such
//! as `gzip` as a compression and `none` as no coding. It also skips a field line that it cannot
//! read, and it accepts a bare LF or CR inside a line. `readChunked` decodes a chunked body from
//! a reader.
//!
//! Rules. Every line of a head is valid, or the head is refused, so a proxy and this server read
//! one set of fields. A request line holds no control byte. A field line is a name of token
//! characters, a colon right after it, and a value with no control byte but tab. A request must
//! not carry both `Content-Length` and `Transfer-Encoding`. The only transfer coding is
//! `chunked`, and HTTP/1.0 has none. A length is ASCII digits only. A method that takes no body
//! must not declare one. The only expectation is `100-continue`. A chunk size line holds hex
//! digits, an optional extension of optional white space, a semicolon, and value bytes, and a
//! CRLF. Every line ends with CRLF. A trailer line is a field line, and the decoder skips it.
//!
//! Limits. A body holds at most `limits.openai_max_request_bytes`, and the decoder checks the cap
//! before it reads the data of a chunk. The framing bytes of one chunked body hold at most
//! `limits.openai_max_chunk_overhead_bytes`, so a decode consumes at most `wire_bytes_max`. One
//! extension holds at most `limits.openai_max_chunk_ext_bytes`, and the trailers at most
//! `limits.openai_max_header_bytes`, the cap of a head.
//!
//! Comptime. Asserts tie the digit bound of a chunk size to the body cap.

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;
const Allocator = std.mem.Allocator;
const http = std.http;
const limits = @import("../../core/limits.zig");

// -- constants --------------------------------------------------------------

/// The body cap. A chunk size and a length fit this type.
pub const body_bytes_max: u32 = limits.openai_max_request_bytes;
const overhead_bytes_max: u32 = limits.openai_max_chunk_overhead_bytes;
const ext_bytes_max: u32 = limits.openai_max_chunk_ext_bytes;
const trailer_bytes_max: u32 = limits.openai_max_header_bytes;

/// The most bytes that one chunked decode consumes from its reader.
pub const wire_bytes_max: u32 = body_bytes_max + overhead_bytes_max;

/// The most hex digits of a chunk size: the digits of a `u32`, so the size never overflows. The
/// cap fits, so a size of these digits can name any body that the cap passes.
const size_digits_max = @bitSizeOf(u32) / 4;

comptime {
    assert(body_bytes_max < std.math.maxInt(u32) - overhead_bytes_max);
    assert(std.math.log2_int_ceil(u64, @as(u64, body_bytes_max) + 1) <= 4 * size_digits_max);
    assert(ext_bytes_max < overhead_bytes_max);
    assert(trailer_bytes_max < overhead_bytes_max);
}

// -- types ------------------------------------------------------------------

/// A head that the framing refuses. The task answers with the status and closes.
pub const Refusal = enum {
    length_and_coding,
    coding_unsupported,
    coding_on_http10,
    length_invalid,
    too_large,
    body_not_allowed,
    expectation_failed,
    line_invalid,

    pub fn status(r: Refusal) u16 {
        return switch (r) {
            .length_and_coding,
            .coding_on_http10,
            .length_invalid,
            .body_not_allowed,
            .line_invalid,
            => 400,
            // RFC 9112 section 6.1 names 501 for a transfer coding that the server does not know.
            .coding_unsupported => 501,
            .too_large => 413,
            .expectation_failed => 417,
        };
    }

    pub fn message(r: Refusal) []const u8 {
        return switch (r) {
            .length_and_coding => "a request must not carry both Content-Length and " ++
                "Transfer-Encoding",
            .coding_unsupported => "the only transfer coding is chunked",
            .coding_on_http10 => "an HTTP/1.0 request has no Transfer-Encoding",
            .length_invalid => "Content-Length must be one value of ASCII digits",
            .too_large => "request body too large",
            .body_not_allowed => "this method takes no request body",
            .expectation_failed => "the only expectation is 100-continue",
            .line_invalid => "a line of the request head is malformed",
        };
    }
};

/// The body form of a request, decided from its head.
pub const Framing = union(enum) {
    none,
    /// A declared length, more than 0 and at most the cap.
    length: u32,
    chunked,
    refused: Refusal,
};

pub const ChunkError = error{
    /// A size line holds no digit, a byte that is not a hex digit, or too many digits.
    SizeInvalid,
    ExtensionTooLong,
    /// An extension holds a control byte, or white space after the size is not followed by `;`.
    ExtensionInvalid,
    /// A trailer line is no field line.
    TrailerInvalid,
    /// A line ends with a bare LF, or a CR is not followed by LF.
    LineEndInvalid,
    /// The data of a chunk is not followed by CRLF.
    DataEndInvalid,
    TrailerTooLong,
    /// The body passes the cap.
    TooLarge,
    /// The framing bytes pass their bound.
    OverheadTooLarge,
    /// The stream ended inside the body.
    Truncated,
    ReadFailed,
    OutOfMemory,
};

/// The framing state of one chunked decode. Every framing byte passes `take`, so the overhead
/// bound covers the size lines, the extensions, the line ends, and the trailers.
const Decoder = struct {
    in: *Io.Reader,
    overhead: u32 = 0,

    fn take(d: *Decoder) ChunkError!u8 {
        if (d.overhead >= overhead_bytes_max) return error.OverheadTooLarge;
        const c = d.in.takeByte() catch |err| return switch (err) {
            error.EndOfStream => error.Truncated,
            error.ReadFailed => error.ReadFailed,
        };
        d.overhead += 1;
        return c;
    }

    /// Read one size line through its CRLF, and return the size.
    fn sizeLine(d: *Decoder) ChunkError!u32 {
        var size: u32 = 0;
        var digits: u32 = 0;
        var c = try d.take();
        while (hexValue(c)) |v| : (c = try d.take()) {
            if (digits == size_digits_max) return error.SizeInvalid;
            size = size * 16 + v;
            digits += 1;
        }
        if (digits == 0) return error.SizeInvalid;

        // An extension is optional white space, a semicolon, and value bytes up to the CR.
        var ext: u32 = 0;
        while (c == ' ' or c == '\t') : (c = try d.take()) {
            if (ext == ext_bytes_max) return error.ExtensionTooLong;
            ext += 1;
        }
        if (ext > 0 and c != ';') return error.ExtensionInvalid;
        if (c == ';') {
            while (c != '\r') : (c = try d.take()) {
                if (c == '\n') return error.LineEndInvalid;
                if (!isValueByte(c)) return error.ExtensionInvalid;
                if (ext == ext_bytes_max) return error.ExtensionTooLong;
                ext += 1;
            }
        }

        if (c == '\n') return error.LineEndInvalid;
        if (c != '\r') return error.SizeInvalid;
        if (try d.take() != '\n') return error.LineEndInvalid;
        return size;
    }

    /// Skip the trailer lines and the empty line that ends the body. `trailer_bytes_max` counts
    /// the name, colon, and value bytes of the lines, and no line end.
    fn trailers(d: *Decoder) ChunkError!void {
        var bytes: u32 = 0;
        for (0..trailer_bytes_max + 1) |_| {
            var c = try d.take();
            if (c == '\r') {
                if (try d.take() != '\n') return error.LineEndInvalid;
                return;
            }

            // A name of token characters, then the colon.
            var name: u32 = 0;
            while (c != ':') : (c = try d.take()) {
                if (c == '\n') return error.LineEndInvalid;
                if (!isTokenChar(c)) return error.TrailerInvalid;
                if (bytes == trailer_bytes_max) return error.TrailerTooLong;
                bytes += 1;
                name += 1;
            }
            if (name == 0) return error.TrailerInvalid;

            // The colon, then the value bytes up to the CR.
            while (c != '\r') : (c = try d.take()) {
                if (c == '\n') return error.LineEndInvalid;
                if (!isValueByte(c)) return error.TrailerInvalid;
                if (bytes == trailer_bytes_max) return error.TrailerTooLong;
                bytes += 1;
            }
            if (try d.take() != '\n') return error.LineEndInvalid;
        }

        // Each pass skips at least one counted byte, so the bound ends the loop first.
        unreachable;
    }
};

// -- api --------------------------------------------------------------------

/// The body form that `head_bytes` declares. `head_bytes` is a head that the std parser accepted.
/// That parser accepts a bare LF or CR inside a line, so this function checks every line itself.
pub fn framing(method: http.Method, version: http.Version, head_bytes: []const u8) Framing {
    var lengths: u32 = 0;
    var length: ?[]const u8 = null;
    var codings: u32 = 0;
    var coding: []const u8 = "";
    var expects: u32 = 0;
    var expect: []const u8 = "";

    // Check every line, and collect the three headers. The head cap bounds the lines.
    var lines = std.mem.splitSequence(u8, head_bytes, "\r\n");
    if (!requestLineValid(lines.first())) return .{ .refused = .line_invalid };
    while (lines.next()) |line| {
        if (line.len == 0) break;

        const colon = fieldColon(line) orelse return .{ .refused = .line_invalid };
        const name = line[0..colon];
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            lengths += 1;
            length = value;
        } else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) {
            codings += 1;
            coding = value;
        } else if (std.ascii.eqlIgnoreCase(name, "expect")) {
            expects += 1;
            expect = value;
        }
    }

    // Decide the form. A syntax refusal comes before the method rule and the cap.
    if (lengths > 0 and codings > 0) return .{ .refused = .length_and_coding };
    if (codings > 0) {
        if (version == .@"HTTP/1.0") return .{ .refused = .coding_on_http10 };
        if (codings > 1) return .{ .refused = .coding_unsupported };
        if (!std.ascii.eqlIgnoreCase(coding, "chunked")) return .{ .refused = .coding_unsupported };
    }
    const declared: ?u64 = if (length) |text| blk: {
        if (lengths > 1) return .{ .refused = .length_invalid };
        break :blk parseLength(text) orelse return .{ .refused = .length_invalid };
    } else null;
    const has_body = codings > 0 or (declared orelse 0) > 0;
    if (has_body and !method.requestHasBody()) return .{ .refused = .body_not_allowed };
    if ((declared orelse 0) > body_bytes_max) return .{ .refused = .too_large };
    if (expects > 1) return .{ .refused = .expectation_failed };
    if (expects == 1 and !std.ascii.eqlIgnoreCase(expect, "100-continue")) {
        return .{ .refused = .expectation_failed };
    }

    if (codings > 0) return .chunked;
    const n = declared orelse 0;
    if (n == 0) return .none;
    return .{ .length = @intCast(n) };
}

/// True when the head asks for an interim reply. Call it only on a head that `framing` accepted.
pub fn expectsContinue(head_bytes: []const u8) bool {
    var lines = std.mem.splitSequence(u8, head_bytes, "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        if (line.len == 0) break;

        const colon = std.mem.findScalar(u8, line, ':') orelse continue;
        if (std.ascii.eqlIgnoreCase(line[0..colon], "expect")) return true;
    }

    return false;
}

/// Decode a chunked body from `in` into `arena`. The result holds at most `body_bytes_max`
/// bytes, and the decode consumes at most `wire_bytes_max` bytes of `in`.
pub fn readChunked(in: *Io.Reader, arena: Allocator) ChunkError![]u8 {
    var d: Decoder = .{ .in = in };
    var body: std.ArrayList(u8) = .empty;

    // Each chunk before the last one adds at least one byte, so the cap bounds the chunks.
    for (0..body_bytes_max + 1) |_| {
        const size = try d.sizeLine();
        if (size == 0) {
            try d.trailers();
            assert(body.items.len <= body_bytes_max);
            assert(d.overhead <= overhead_bytes_max);
            return body.items;
        }

        if (size > body_bytes_max - body.items.len) return error.TooLarge;
        const data = try body.addManyAsSlice(arena, size);
        in.readSliceAll(data) catch |err| return switch (err) {
            error.EndOfStream => error.Truncated,
            error.ReadFailed => error.ReadFailed,
        };
        if (try d.take() != '\r') return error.DataEndInvalid;
        if (try d.take() != '\n') return error.DataEndInvalid;
    }

    unreachable;
}

// -- helpers ----------------------------------------------------------------

/// The value of a length of ASCII digits, or null for any other text. A value past the cap
/// saturates to `maxInt(u64)`, so the caller refuses it as too large.
fn parseLength(text: []const u8) ?u64 {
    if (text.len == 0) return null;

    var value: u64 = 0;
    for (text) |c| {
        if (c < '0' or c > '9') return null;
        value = std.math.mul(u64, value, 10) catch std.math.maxInt(u64);
        value = std.math.add(u64, value, c - '0') catch std.math.maxInt(u64);
    }

    return value;
}

/// True when no byte of the request line is a control byte. The std parser checks the rest of it.
fn requestLineValid(line: []const u8) bool {
    for (line) |c| {
        if (c < ' ' or c == 0x7f) return false;
    }

    return true;
}

/// The index of the colon of a field line, or null when the line is no field line: a name of
/// one or more token characters, the colon right after it, and value bytes after the colon.
/// A line that starts with white space, the obsolete line folding, fails the name.
fn fieldColon(line: []const u8) ?usize {
    const colon = std.mem.findScalar(u8, line, ':') orelse return null;
    if (colon == 0) return null;

    for (line[0..colon]) |c| {
        if (!isTokenChar(c)) return null;
    }
    for (line[colon + 1 ..]) |c| {
        if (!isValueByte(c)) return null;
    }

    return colon;
}

/// The `tchar` set of RFC 9110 section 5.6.2.
fn isTokenChar(c: u8) bool {
    return switch (c) {
        'a'...'z', 'A'...'Z', '0'...'9' => true,
        '!', '#', '$', '%', '&', '\'', '*', '+', '-', '.', '^', '_', '`', '|', '~' => true,
        else => false,
    };
}

/// A byte of a field value: any byte but a control byte, where tab counts as white space.
fn isValueByte(c: u8) bool {
    if (c == '\t') return true;
    return c >= ' ' and c != 0x7f;
}

fn hexValue(c: u8) ?u32 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => null,
    };
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;
const prop = @import("../../core/utils/prop.zig");

fn decode(arena: Allocator, bytes: []const u8) ChunkError![]u8 {
    var in: Io.Reader = .fixed(bytes);
    return readChunked(&in, arena);
}

/// The framing of `head`, or null when the std parser refuses the head first.
fn headFraming(head: []const u8) ?Framing {
    const parsed = http.Server.Request.Head.parse(head) catch return null;
    return framing(parsed.method, parsed.version, head);
}

test "body: the framing table" {
    const post = "POST /v1/chat/completions HTTP/1.1\r\n";
    const big = std.fmt.comptimePrint("{d}", .{body_bytes_max + 1});
    const cap = std.fmt.comptimePrint("{d}", .{body_bytes_max});
    // A null want is a head that the std parser refuses, and the server answers it with 400.
    const Row = struct { head: []const u8, want: ?Framing };
    const rows = [_]Row{
        .{ .head = post ++ "\r\n", .want = .none },
        .{ .head = post ++ "Content-Length: 0\r\n\r\n", .want = .none },
        .{ .head = post ++ "Content-Length: 12\r\n\r\n", .want = .{ .length = 12 } },
        .{ .head = post ++ "Content-Length: 007\r\n\r\n", .want = .{ .length = 7 } },
        .{ .head = post ++ "Content-Length: " ++ cap ++ "\r\n\r\n", .want = .{ .length = cap_n } },
        .{ .head = post ++ "transfer-encoding: chunked\r\n\r\n", .want = .chunked },
        .{ .head = post ++ "Transfer-Encoding: Chunked\r\n\r\n", .want = null },
        .{ .head = post ++ "Content-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n", .want = .{
            .refused = .length_and_coding,
        } },
        .{ .head = post ++ "Transfer-Encoding: gzip\r\n\r\n", .want = .{
            .refused = .coding_unsupported,
        } },
        .{ .head = post ++ "Transfer-Encoding: gzip, chunked\r\n\r\n", .want = .{
            .refused = .coding_unsupported,
        } },
        .{ .head = post ++ "Transfer-Encoding: none\r\n\r\n", .want = .{
            .refused = .coding_unsupported,
        } },
        .{ .head = "POST / HTTP/1.0\r\nTransfer-Encoding: chunked\r\n\r\n", .want = .{
            .refused = .coding_on_http10,
        } },
        .{ .head = post ++ "Content-Length: +10\r\n\r\n", .want = .{ .refused = .length_invalid } },
        .{ .head = post ++ "Content-Length: 1_0\r\n\r\n", .want = .{ .refused = .length_invalid } },
        .{ .head = post ++ "Content-Length:\r\n\r\n", .want = null },
        .{ .head = post ++ "Content-Length: " ++ big ++ "\r\n\r\n", .want = .{
            .refused = .too_large,
        } },
        .{ .head = post ++ "Content-Length: 99999999999999999999999\r\n\r\n", .want = null },
        .{ .head = "GET /v1/models HTTP/1.1\r\nContent-Length: 0\r\n\r\n", .want = .none },
        .{ .head = "GET /v1/models HTTP/1.1\r\nContent-Length: 3\r\n\r\n", .want = .{
            .refused = .body_not_allowed,
        } },
        .{ .head = "HEAD / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", .want = .{
            .refused = .body_not_allowed,
        } },
        .{ .head = post ++ "Content-Length: 2\r\nExpect: 100-Continue\r\n\r\n", .want = .{
            .length = 2,
        } },
        .{ .head = post ++ "Content-Length: 2\r\nExpect: foo\r\n\r\n", .want = .{
            .refused = .expectation_failed,
        } },
        .{
            .head = post ++ "Expect: 100-continue\r\nContent-Length: " ++ big ++ "\r\n\r\n",
            .want = .{ .refused = .too_large },
        },
    };
    for (rows) |row| {
        testing.expectEqualDeep(row.want, headFraming(row.head)) catch |err| {
            std.debug.print("head: {s}\n", .{row.head});
            return err;
        };
    }

    // The framing does not lean on the std parser for the heads that the std parser refuses.
    const v11: http.Version = .@"HTTP/1.1";
    const empty = post ++ "Content-Length:\r\n\r\n";
    try testing.expectEqual(Refusal.length_invalid, framing(.POST, v11, empty).refused);
    const huge = post ++ "Content-Length: 99999999999999999999999\r\n\r\n";
    try testing.expectEqual(Refusal.too_large, framing(.POST, v11, huge).refused);
    const twice = post ++ "Content-Length: 1\r\nContent-Length: 1\r\n\r\n";
    try testing.expectEqual(Refusal.length_invalid, framing(.POST, v11, twice).refused);
    const mixed = post ++ "Transfer-Encoding: Chunked\r\n\r\n";
    try testing.expectEqual(Framing.chunked, framing(.POST, v11, mixed));
}

const cap_n: u32 = body_bytes_max;

test "body: the std head parser refuses two lengths before the framing sees them" {
    const head = "POST / HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n";
    try testing.expectError(error.HttpHeadersInvalid, http.Server.Request.Head.parse(head));
}

test "body: a chunked body decodes, with every accepted form of size line and trailer" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const Row = struct { wire: []const u8, want: []const u8 };
    const rows = [_]Row{
        .{ .wire = "0\r\n\r\n", .want = "" },
        .{ .wire = "5\r\nhello\r\n0\r\n\r\n", .want = "hello" },
        .{ .wire = "0005\r\nhello\r\n000\r\n\r\n", .want = "hello" },
        .{
            .wire = "A\r\n0123456789\r\na\r\nabcdefghij\r\n0\r\n\r\n",
            .want = "0123456789abcdefghij",
        },
        .{ .wire = "2;name=value\r\nhi\r\n0 ; last\r\n\r\n", .want = "hi" },
        .{ .wire = "2\r\nhi\r\n0\r\nX-One: 1\r\nX-Two: 2\r\n\r\n", .want = "hi" },
    };
    for (rows) |row| {
        try testing.expectEqualStrings(row.want, try decode(arena, row.wire));
    }

    // Many small chunks.
    var wire: std.ArrayList(u8) = .empty;
    const chunks = 1000;
    for (0..chunks) |_| {
        try wire.appendSlice(arena, "3\r\nabc\r\n");
    }
    try wire.appendSlice(arena, "0\r\n\r\n");
    const body = try decode(arena, wire.items);
    try testing.expectEqual(@as(usize, 3 * chunks), body.len);
    try testing.expectEqualStrings("abcabc", body[0..6]);
}

test "body: the cap passes exactly, and one byte more is refused before the data" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const size = std.fmt.comptimePrint("{x}\r\n", .{body_bytes_max});
    const wire = try std.mem.concat(arena, u8, &.{
        size,
        try arena.alloc(u8, body_bytes_max),
        "\r\n0\r\n\r\n",
    });
    @memset(wire[size.len..][0..body_bytes_max], 'x');
    try testing.expectEqual(@as(usize, body_bytes_max), (try decode(arena, wire)).len);

    // The size line alone is refused, so no data byte needs to arrive.
    const over = std.fmt.comptimePrint("{x}\r\n", .{body_bytes_max + 1});
    try testing.expectError(error.TooLarge, decode(arena, over));
    // Two chunks that pass the cap together.
    const half = std.fmt.comptimePrint("{x}\r\n", .{body_bytes_max / 2 + 1});
    const two = try std.mem.concat(arena, u8, &.{
        half,
        try arena.alloc(u8, body_bytes_max / 2 + 1),
        "\r\n",
        half,
    });
    @memset(two[half.len..][0 .. body_bytes_max / 2 + 1], 'x');
    try testing.expectError(error.TooLarge, decode(arena, two));
}

test "body: each malformed framing gets its typed error" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const ext = "1;" ++ "a" ** (ext_bytes_max + 1);
    const trailer = "0\r\nX: " ++ "a" ** trailer_bytes_max ++ "\r\n\r\n";
    const Row = struct { wire: []const u8, want: ChunkError };
    const rows = [_]Row{
        .{ .wire = "ffffffffffffffff\r\n", .want = error.SizeInvalid },
        .{ .wire = "1g\r\n", .want = error.SizeInvalid },
        .{ .wire = "+1\r\n", .want = error.SizeInvalid },
        .{ .wire = "\r\n", .want = error.SizeInvalid },
        .{ .wire = "1\nx\r\n0\r\n\r\n", .want = error.LineEndInvalid },
        .{ .wire = "1\rx", .want = error.LineEndInvalid },
        .{ .wire = "1;x\nx\r\n", .want = error.LineEndInvalid },
        .{ .wire = "5 junk\r\nhello\r\n0\r\n\r\n", .want = error.ExtensionInvalid },
        .{ .wire = "5;a\x01b\r\n", .want = error.ExtensionInvalid },
        .{
            .wire = "0\r\nGET /v1/models HTTP/1.1\r\nHost: a\r\n\r\n",
            .want = error.TrailerInvalid,
        },
        .{ .wire = "0\r\n: a\r\n\r\n", .want = error.TrailerInvalid },
        .{ .wire = "0\r\n X: a\r\n\r\n", .want = error.TrailerInvalid },
        .{ .wire = "0\r\nX: a\x00\r\n\r\n", .want = error.TrailerInvalid },
        .{ .wire = ext, .want = error.ExtensionTooLong },
        .{ .wire = "1\r\nxy\r\n", .want = error.DataEndInvalid },
        .{ .wire = "1\r\nx\n0\r\n\r\n", .want = error.DataEndInvalid },
        .{ .wire = "0\r\nX: 1\n\r\n", .want = error.LineEndInvalid },
        .{ .wire = "0\r\n\n", .want = error.LineEndInvalid },
        .{ .wire = trailer, .want = error.TrailerTooLong },
        .{ .wire = "5\r\nhel", .want = error.Truncated },
        .{ .wire = "5\r\nhello\r\n", .want = error.Truncated },
        .{ .wire = "0\r\n", .want = error.Truncated },
    };
    for (rows) |row| {
        testing.expectError(row.want, decode(arena, row.wire)) catch |err| {
            std.debug.print("wire: {s}\n", .{row.wire[0..@min(row.wire.len, 40)]});
            return err;
        };
    }

    // A chunk of one byte costs five framing bytes, so these chunks pass the overhead bound
    // before the body cap.
    var wire: std.ArrayList(u8) = .empty;
    for (0..overhead_bytes_max / 5 + 1) |_| {
        try wire.appendSlice(arena, "1\r\nx\r\n");
    }
    try testing.expectError(error.OverheadTooLarge, decode(arena, wire.items));
}

test "body: every line of a head must be valid, or the head is refused" {
    const get = "GET /v1/models HTTP/1.1\r\n";
    const v11: http.Version = .@"HTTP/1.1";
    const valid = [_][]const u8{
        "X: a",
        "X-Y:a",
        "x:",
        "X: a\tb ",
        "X: \x80\xff",
        "!#$%&'*+-.^_`|~09az: v",
    };
    for (valid) |line| {
        const head = try std.mem.concat(testing.allocator, u8, &.{ get, line, "\r\n\r\n" });
        defer testing.allocator.free(head);
        try testing.expectEqual(Framing.none, framing(.GET, v11, head));
    }

    const invalid = [_][]const u8{
        "Content-Length : 5",
        " X: a",
        "\tX: a",
        "X a",
        "X: a\nContent-Length: 5",
        "X: a\rB: c",
        "X: a\x00",
        ": a",
        "X\x01: a",
        "X: \x7f",
        "X(: a",
    };
    for (invalid) |line| {
        const head = try std.mem.concat(testing.allocator, u8, &.{ get, line, "\r\n\r\n" });
        defer testing.allocator.free(head);
        testing.expectEqual(Refusal.line_invalid, framing(.GET, v11, head).refused) catch |err| {
            std.debug.print("line: {s}\n", .{line});
            return err;
        };
    }

    const bad_target = "GET /\nContent-Length: 5 HTTP/1.1\r\n\r\n";
    try testing.expectEqual(Refusal.line_invalid, framing(.GET, v11, bad_target).refused);
}

test "body: the bounds pass at their value and refuse one past it" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    // An extension of exactly the bound passes, and one byte more is refused.
    const ext_at = "1;" ++ "a" ** (ext_bytes_max - 1) ++ "\r\nx\r\n0\r\n\r\n";
    try testing.expectEqualStrings("x", try decode(arena, ext_at));
    const ext_past = "1;" ++ "a" ** ext_bytes_max ++ "\r\nx\r\n0\r\n\r\n";
    try testing.expectError(error.ExtensionTooLong, decode(arena, ext_past));

    // Trailers of exactly the bound pass, and one byte more is refused.
    const trailer_at = "0\r\nX:" ++ "a" ** (trailer_bytes_max - 2) ++ "\r\n\r\n";
    try testing.expectEqualStrings("", try decode(arena, trailer_at));
    const trailer_past = "0\r\nX:" ++ "a" ** (trailer_bytes_max - 1) ++ "\r\n\r\n";
    try testing.expectError(error.TrailerTooLong, decode(arena, trailer_past));

    // One-byte chunks cost five framing bytes each, and the last chunk five more. An extension
    // pads the framing to exactly the bound, and one more pad byte is refused.
    const chunks = (overhead_bytes_max - 5) / 5;
    const pad = overhead_bytes_max - 5 - 5 * chunks;
    for ([_]u32{ pad, pad + 1 }) |ext| {
        var wire: std.ArrayList(u8) = .empty;
        try wire.appendSlice(arena, "1");
        if (ext > 0) {
            try wire.append(arena, ';');
            try wire.appendNTimes(arena, 'p', ext - 1);
        }
        try wire.appendSlice(arena, "\r\nx\r\n");
        for (1..chunks) |_| {
            try wire.appendSlice(arena, "1\r\nx\r\n");
        }
        try wire.appendSlice(arena, "0\r\n\r\n");
        if (ext == pad) {
            try testing.expectEqual(@as(usize, chunks), (try decode(arena, wire.items)).len);
        } else {
            try testing.expectError(error.OverheadTooLarge, decode(arena, wire.items));
        }
    }
}

// -- tests: properties --------------------------------------------------------

/// A request that follows each body, so a test can see where the body read stopped.
const sentinel = "GET /sentinel HTTP/1.1\r\n\r\n";

/// The two valid base requests, each followed by the sentinel.
const base_body = "{\"a\":1}";
const base_length = "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n" ++
    "Content-Length: 7\r\nExpect: 100-continue\r\n\r\n" ++ base_body ++ sentinel;
const base_chunked = "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n" ++
    "Transfer-Encoding: chunked\r\nExpect: 100-continue\r\n\r\n" ++
    "4;ext=1\r\n{\"a\"\r\n3\r\n:1}\r\n0\r\nX-Trailer: t\r\n\r\n" ++ sentinel;

/// Field values that sit at a boundary of a rule.
const field_values = [_][]const u8{
    "",                 "0",         "+1",                "1_0",
    "ffffffffffffffff", "g",         "40000",             "40001",
    "00000004",         "000000004", "chunked",           "gzip, chunked",
    "100-continue",     "foo",       "99999999999999999", "\n",
};

/// The fields that a mutation can rewrite: a header value or a size line.
const field_targets = [_][]const u8{
    "chunked", "100-continue", "4;ext=1", "3\r\n", "0\r\nX", "Content-Length: 7",
};

/// Spellings of a body header that a lenient parser can still read as that header.
const lenient_names = [_][]const u8{
    "Content-Length",     "content-length",     "CONTENT-LENGTH ",
    " Content-Length",    "\tcontent-length\t", "Transfer-Encoding",
    "transfer-encoding ", " TRANSFER-ENCODING",
};
const lenient_breaks = [_][]const u8{ "\r\n", "\n", "\r", "\x00" };
const lenient_values = [_][]const u8{ "5", " 5", "chunked", "+5", "7" };

const Mutated = struct {
    bytes: [base_chunked.len + 128]u8,
    len: usize,
    /// The request bytes are a base request with no change.
    intact: bool,
};

fn mutate(random: std.Random) Mutated {
    var out: Mutated = .{ .bytes = undefined, .len = 0, .intact = false };
    var w: Io.Writer = .fixed(&out.bytes);
    const base = if (random.boolean()) base_length else base_chunked;
    const at = random.uintLessThan(usize, base.len);
    switch (random.uintLessThan(u8, 6)) {
        0 => {
            w.writeAll(base) catch unreachable;
            out.intact = true;
        },
        // One byte changes, goes, or comes.
        1 => {
            w.writeAll(base) catch unreachable;
            out.bytes[at] = random.int(u8);
        },
        2 => {
            w.writeAll(base[0..at]) catch unreachable;
            w.writeAll(base[at + 1 ..]) catch unreachable;
        },
        3 => {
            w.writeAll(base[0..at]) catch unreachable;
            w.writeByte(random.int(u8)) catch unreachable;
            w.writeAll(base[at..]) catch unreachable;
        },
        // One field takes a boundary value.
        4 => {
            const target = field_targets[random.uintLessThan(usize, field_targets.len)];
            const value = field_values[random.uintLessThan(usize, field_values.len)];
            const start = std.mem.find(u8, base, target) orelse {
                w.writeAll(base) catch unreachable;
                out.intact = true;
                out.len = w.end;
                return out;
            };
            w.writeAll(base[0..start]) catch unreachable;
            w.writeAll(value) catch unreachable;
            w.writeAll(base[start + target.len ..]) catch unreachable;
        },
        // A body header in a lenient spelling goes in at a random byte of the head.
        else => {
            const head_len = std.mem.find(u8, base, "\r\n\r\n").?;
            const cut = random.uintLessThan(usize, head_len);
            const pick = struct {
                fn one(r: std.Random, list: []const []const u8) []const u8 {
                    return list[r.uintLessThan(usize, list.len)];
                }
            }.one;
            w.writeAll(base[0..cut]) catch unreachable;
            w.writeAll(pick(random, &lenient_breaks)) catch unreachable;
            w.writeAll(pick(random, &lenient_names)) catch unreachable;
            w.writeAll(":") catch unreachable;
            w.writeAll(pick(random, &lenient_values)) catch unreachable;
            w.writeAll(base[cut..]) catch unreachable;
        },
    }
    out.len = w.end;
    return out;
}

/// True when a lenient parser finds a body header in `head`: it ends a line at any CR, LF, or
/// NUL, and it trims and case-folds the name.
fn lenientBodyHeader(head: []const u8) bool {
    var lines = std.mem.tokenizeAny(u8, head, "\r\n\x00");
    while (lines.next()) |line| {
        const colon = std.mem.findScalar(u8, line, ':') orelse continue;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) return true;
        // A length of zeros declares no body to every parser.
        const zero = value.len > 0 and std.mem.allEqual(u8, value, '0');
        if (std.ascii.eqlIgnoreCase(name, "content-length") and !zero) return true;
    }

    return false;
}

fn checkMutation(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const m = mutate(prng.random());
    const bytes = m.bytes[0..m.len];

    // Find the end of the head as the server does, then let the std parser take it.
    var head_parser: http.HeadParser = .{};
    const head_end = head_parser.feed(bytes);
    if (head_parser.state != .finished) return;
    const head = bytes[0..head_end];
    const parsed = http.Server.Request.Head.parse(head) catch return;

    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    const form = framing(parsed.method, parsed.version, head);
    // A body header that a lenient proxy reads must never read as no body here.
    if (lenientBodyHeader(head)) {
        try testing.expect(form != .none);
    }

    var in: Io.Reader = .fixed(bytes[head_end..]);
    const body: []const u8 = switch (form) {
        .none, .refused => return,
        .length => |n| read: {
            try testing.expect(n > 0);
            try testing.expect(n <= body_bytes_max);
            const data = try arena_state.allocator().alloc(u8, n);
            in.readSliceAll(data) catch return;
            break :read data;
        },
        .chunked => readChunked(&in, arena_state.allocator()) catch |err| {
            try testing.expect(in.seek <= wire_bytes_max);
            return switch (err) {
                error.ReadFailed, error.OutOfMemory => err,
                else => {},
            };
        },
    };
    try testing.expect(body.len <= body_bytes_max);
    try testing.expect(in.seek <= wire_bytes_max);

    // A base request reads its body and stops at the first byte of the sentinel.
    if (m.intact) {
        try testing.expectEqualStrings(base_body, body);
        try testing.expectEqualStrings(sentinel, bytes[head_end + in.seek ..]);
    }
}

test "body: a request with one byte or one field changed is refused or parsed within the cap" {
    const opts: prop.Options = .{ .site = 0x424f_4459_0001, .cases = 4000 };
    try prop.forAll(opts, {}, u64, prop.genSeed(void), checkMutation);
}

/// Encode `data` as a chunked body with random chunk sizes, size spellings, extensions, and
/// trailers, all within the bounds.
fn encodeChunked(arena: Allocator, random: std.Random, data: []const u8) ![]u8 {
    var wire: std.ArrayList(u8) = .empty;
    var rest = data;
    // Chunks of 64 bytes or more keep the framing below its bound at any body size.
    while (rest.len > 0) {
        const n = @min(rest.len, random.intRangeAtMost(usize, 64, 8192));
        const zeros = random.uintLessThan(usize, 3);
        try wire.appendNTimes(arena, '0', zeros);
        if (random.boolean()) {
            try wire.print(arena, "{x}", .{n});
        } else {
            try wire.print(arena, "{X}", .{n});
        }
        if (random.uintLessThan(u8, 8) == 0) {
            try wire.appendSlice(arena, " ;name=\"v\"");
        }
        try wire.appendSlice(arena, "\r\n");
        try wire.appendSlice(arena, rest[0..n]);
        try wire.appendSlice(arena, "\r\n");
        rest = rest[n..];
    }
    try wire.appendSlice(arena, "0\r\n");
    for (0..random.uintLessThan(usize, 3)) |_| {
        try wire.appendSlice(arena, "X-Sum: 1\r\n");
    }
    try wire.appendSlice(arena, "\r\n");
    return wire.items;
}

fn checkRoundTrip(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    // Sizes up to the cap, with the cap itself in a share of the cases.
    const len: usize = switch (random.uintLessThan(u8, 4)) {
        0 => body_bytes_max,
        1 => random.uintAtMost(usize, 64),
        else => random.uintAtMost(usize, body_bytes_max),
    };
    const data = try arena.alloc(u8, len);
    random.bytes(data);

    const chunked = random.boolean();
    const wire = if (chunked) try encodeChunked(arena, random, data) else data;
    const all = try std.mem.concat(arena, u8, &.{ wire, sentinel });
    var in: Io.Reader = .fixed(all);
    const body = if (chunked) try readChunked(&in, arena) else read: {
        const out = try arena.alloc(u8, len);
        try in.readSliceAll(out);
        break :read out;
    };
    try testing.expectEqualSlices(u8, data, body);
    try testing.expectEqual(wire.len, in.seek);
}

test "body: a valid body of any size up to the cap reads whole and stops at the next request" {
    const opts: prop.Options = .{ .site = 0x424f_4459_0002, .cases = 64 };
    try prop.forAll(opts, {}, u64, prop.genSeed(void), checkRoundTrip);
}
