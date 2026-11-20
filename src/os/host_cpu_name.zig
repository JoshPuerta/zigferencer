//! The CPU name that the OS states, for a CPU with no CPUID name.
//!
//! Platforms. Linux reads the first `model name` value of `/proc/cpuinfo`, best effort. Another OS
//! gives null.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;

// -- constants --------------------------------------------------------------

/// Bytes of `/proc/cpuinfo` that the read keeps.
const cpuinfo_text_max: usize = 64 * 1024;

// -- api --------------------------------------------------------------------

/// The CPU name, copied to `arena`, or null when the OS states none or the read fails.
pub const read = switch (builtin.os.tag) {
    .linux => readLinux,
    else => readNone,
};

// -- helpers ----------------------------------------------------------------

fn readLinux(io: Io, arena: Allocator) ?[]const u8 {
    var buf: [cpuinfo_text_max]u8 = undefined;
    const cwd = Io.Dir.cwd();
    const slice = cwd.readFile(io, "/proc/cpuinfo", &buf) catch return null;
    const needle = "model name";
    const idx = std.mem.find(u8, slice, needle) orelse return null;
    const after_needle = slice[idx + needle.len ..];
    const colon = std.mem.findScalar(u8, after_needle, ':') orelse return null;
    const after_colon = after_needle[colon + 1 ..];
    const end = std.mem.findScalar(u8, after_colon, '\n') orelse return null;
    const trimmed = std.mem.trim(u8, after_colon[0..end], " \t");
    return arena.dupe(u8, trimmed) catch null;
}

fn readNone(io: Io, arena: Allocator) ?[]const u8 {
    _ = io;
    _ = arena;
    return null;
}
