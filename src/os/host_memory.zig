//! The free RAM and the total RAM of the host. The RAM plan checks a model against the free
//! figure, and `info` prints the total.
//!
//! Platforms. Linux reads `/proc/meminfo`. Windows calls `GlobalMemoryStatusEx`. Another OS gives
//! null, and the RAM plan then skips its check.

const std = @import("std");
const builtin = @import("builtin");
const kernel32 = @import("windows_kernel32_declarations.zig");
const Io = std.Io;

// -- constants --------------------------------------------------------------

/// Bytes `/proc/meminfo` holds. The file is about 1.5 KiB, and both fields the
/// engine reads sit in its first lines.
const meminfo_text_max: usize = 4096;

// -- api --------------------------------------------------------------------

/// The RAM that a new allocation can get now without swap, or null when the read fails.
pub const availableBytes = switch (builtin.os.tag) {
    .linux => availableBytesLinux,
    .windows => availableBytesWindows,
    else => bytesNone,
};

/// The physical RAM of the host, or null when the read fails.
pub const totalBytes = switch (builtin.os.tag) {
    .linux => totalBytesLinux,
    .windows => totalBytesWindows,
    else => bytesNone,
};

// -- helpers ----------------------------------------------------------------

/// `MemAvailable:` in `/proc/meminfo`.
fn availableBytesLinux(io: Io) ?u64 {
    var buf: [meminfo_text_max]u8 = undefined;
    const text = Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return null;
    return meminfoBytes(text, "MemAvailable");
}

fn totalBytesLinux(io: Io) ?u64 {
    var buf: [meminfo_text_max]u8 = undefined;
    const text = Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return null;
    return meminfoBytes(text, "MemTotal");
}

/// Bytes of one `Field:   N kB` line of `/proc/meminfo` text. The match is at a
/// line start, so a field name never matches inside a longer name. A missing
/// field, a malformed line, or a count past a `u64` gives null.
fn meminfoBytes(text: []const u8, field: []const u8) ?u64 {
    std.debug.assert(field.len >= 1);
    std.debug.assert(std.mem.findScalar(u8, field, ':') == null);

    var lines = std.mem.splitScalar(u8, text, '\n');
    // Each pass reads one line, so the text length bounds the loop.
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, field)) continue;

        const rest = line[field.len..];
        if (rest.len == 0 or rest[0] != ':') continue;

        var fields = std.mem.tokenizeAny(u8, rest[1..], " \t\r");
        const count = fields.next() orelse return null;
        const unit = fields.next() orelse return null;
        if (fields.next() != null) return null;
        if (!std.mem.eql(u8, unit, "kB")) return null;

        const n_kb = std.fmt.parseInt(u64, count, 10) catch return null;
        return std.math.mul(u64, n_kb, 1024) catch null;
    }

    return null;
}

/// The lower of the free physical RAM and the free commit.
fn availableBytesWindows(io: Io) ?u64 {
    _ = io;
    const status = memoryStatusWindows() orelse return null;

    // Windows refuses an allocation at the commit limit, and the free commit can be below the
    // free physical RAM.
    const available = @min(status.ullAvailPhys, status.ullAvailPageFile);
    std.debug.assert(available <= status.ullAvailPhys);
    return available;
}

fn totalBytesWindows(io: Io) ?u64 {
    _ = io;
    const status = memoryStatusWindows() orelse return null;
    return status.ullTotalPhys;
}

fn memoryStatusWindows() ?kernel32.MEMORYSTATUSEX {
    var status: kernel32.MEMORYSTATUSEX = std.mem.zeroes(kernel32.MEMORYSTATUSEX);
    status.dwLength = @sizeOf(kernel32.MEMORYSTATUSEX);
    if (!kernel32.GlobalMemoryStatusEx(&status).toBool()) return null;
    return status;
}

fn bytesNone(io: Io) ?u64 {
    _ = io;
    return null;
}

// -- tests ------------------------------------------------------------------

test "meminfoBytes: a fixed meminfo sample, a missing field, and malformed lines" {
    const sample =
        \\MemTotal:       24606612 kB
        \\MemFree:        22687856 kB
        \\MemAvailable:   23555960 kB
        \\Buffers:           86880 kB
        \\HugePages_Total:       0
        \\
    ;
    try std.testing.expectEqual(@as(?u64, 24606612 * 1024), meminfoBytes(sample, "MemTotal"));
    try std.testing.expectEqual(@as(?u64, 23555960 * 1024), meminfoBytes(sample, "MemAvailable"));
    try std.testing.expectEqual(@as(?u64, 86880 * 1024), meminfoBytes(sample, "Buffers"));
    // A field name that is a prefix of another line never matches that line.
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes(sample, "Mem"));
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes(sample, "SwapTotal"));
    // A line without a unit, with a wrong unit, or with no digits is malformed.
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes(sample, "HugePages_Total"));
    try std.testing.expectEqual(
        @as(?u64, null),
        meminfoBytes("MemAvailable: 12 MB\n", "MemAvailable"),
    );
    try std.testing.expectEqual(
        @as(?u64, null),
        meminfoBytes("MemAvailable: x1 kB\n", "MemAvailable"),
    );
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes("MemAvailable:\n", "MemAvailable"));
    // A count whose byte figure passes a `u64` gives null, not a wrapped value.
    try std.testing.expectEqual(
        @as(?u64, null),
        meminfoBytes("MemAvailable: 18446744073709551615 kB", "MemAvailable"),
    );
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes("", "MemAvailable"));
}

test "host_memory: each figure is nonzero, and the free RAM stays inside the total" {
    if (builtin.os.tag != .linux and builtin.os.tag != .windows) return error.SkipZigTest;

    const available = availableBytes(std.testing.io) orelse return error.TestUnexpectedResult;
    const total = totalBytes(std.testing.io) orelse return error.TestUnexpectedResult;
    try std.testing.expect(available > 0);
    try std.testing.expect(available <= total);
}
