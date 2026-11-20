//! Figures of the whole process: the name of the OS, the process id, the resident memory, and the
//! open-file count.
//!
//! Platforms. Linux reads the process id from the kernel, and the resident memory and the open
//! files from `/proc/self`. Another OS has process id 1 and gives null for the last two figures.
//!
//! Comptime. `openFileCount` takes its cap at comptime, so each caller states its own bound.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

// -- constants --------------------------------------------------------------

/// Bytes of `/proc/self/status` that the read keeps. The file is about 1.5 KiB.
const status_text_max: usize = 4096;

// -- api --------------------------------------------------------------------

/// The name of the OS that this binary targets, for `info`.
pub const os_name: []const u8 = @tagName(builtin.os.tag);

/// The id of this process.
pub fn processId() u32 {
    return switch (builtin.os.tag) {
        .linux => @intCast(std.os.linux.getpid()),
        else => 1,
    };
}

/// The resident memory of this process in KiB, or null when the read fails.
pub const residentKib = switch (builtin.os.tag) {
    .linux => residentKibLinux,
    else => residentKibNone,
};

/// The open file descriptors of this process, at most `count_max`, or null when the walk fails.
/// The walk holds one descriptor of its own, which the count leaves out.
pub fn openFileCount(io: Io, comptime count_max: u32) ?u32 {
    comptime std.debug.assert(count_max >= 1);

    return switch (builtin.os.tag) {
        .linux => openFileCountLinux(io, count_max),
        else => null,
    };
}

// -- helpers ----------------------------------------------------------------

fn residentKibLinux(io: Io) ?u64 {
    var buf: [status_text_max]u8 = undefined;
    const text = Io.Dir.cwd().readFile(io, "/proc/self/status", &buf) catch return null;
    return statusKib(text, "VmRSS");
}

fn residentKibNone(io: Io) ?u64 {
    _ = io;
    return null;
}

/// The value of the `key:` line of a `/proc/<pid>/status` text, in KiB. The
/// kernel prints these lines as `VmRSS:\t   1234 kB`. `null` when no line has
/// the key or the value is not a KiB count.
fn statusKib(text: []const u8, key: []const u8) ?u64 {
    std.debug.assert(key.len >= 1);

    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len <= key.len) continue;
        if (!std.mem.startsWith(u8, line, key)) continue;
        if (line[key.len] != ':') continue;

        const rest = std.mem.trim(u8, line[key.len + 1 ..], " \t");
        const unit = std.mem.findScalar(u8, rest, ' ') orelse return null;
        if (!std.mem.eql(u8, std.mem.trimStart(u8, rest[unit..], " "), "kB")) return null;

        return std.fmt.parseInt(u64, rest[0..unit], 10) catch null;
    }

    return null;
}

fn openFileCountLinux(io: Io, comptime count_max: u32) ?u32 {
    var dir = Io.Dir.cwd().openDir(io, "/proc/self/fd", .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var it = dir.iterate();
    var count: u32 = 0;
    while (count < count_max + 1) {
        const entry = (it.next(io) catch return null) orelse break;
        _ = entry;
        count += 1;
    }

    std.debug.assert(count >= 1);
    return @min(count - 1, count_max);
}

// -- tests ------------------------------------------------------------------

test "statusKib: reads the KiB count of the named line only" {
    const text = "Name:\tzigferencer\nVmPeak:\t  912345 kB\nVmRSSX:\t       7 kB\n" ++
        "VmRSS:\t  654321 kB\nThreads:\t17\n";
    try std.testing.expectEqual(@as(?u64, 654321), statusKib(text, "VmRSS"));
    try std.testing.expectEqual(@as(?u64, 912345), statusKib(text, "VmPeak"));
    // A line without the kB unit, a missing key, and a key prefix all read as null.
    try std.testing.expectEqual(@as(?u64, null), statusKib(text, "Threads"));
    try std.testing.expectEqual(@as(?u64, null), statusKib(text, "VmSwap"));
    try std.testing.expectEqual(@as(?u64, null), statusKib(text, "VmR"));
    try std.testing.expectEqual(@as(?u64, null), statusKib("VmRSS:\t12x kB\n", "VmRSS"));
    try std.testing.expectEqual(@as(?u64, null), statusKib("", "VmRSS"));
}

test "residentKib: reads the live status of this process" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    var buf: [status_text_max]u8 = undefined;
    const text = try Io.Dir.cwd().readFile(std.testing.io, "/proc/self/status", &buf);
    try std.testing.expect(text.len < buf.len);
    const kib = residentKib(std.testing.io) orelse return error.TestUnexpectedResult;
    try std.testing.expect(kib >= 1);
}

test "openFileCount: counts open descriptors and leaves out its own" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    const io = std.testing.io;
    const count_max = 1 << 16;
    const before = openFileCount(io, count_max) orelse return error.TestUnexpectedResult;
    var dir = try Io.Dir.cwd().openDir(io, "/proc", .{});
    const during = openFileCount(io, count_max) orelse return error.TestUnexpectedResult;
    dir.close(io);
    const after = openFileCount(io, count_max) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(before + 1, during);
    try std.testing.expectEqual(before, after);
}
