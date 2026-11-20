//! The settings of a thread on each OS: the thread name that perf and the trace show.
//!
//! Platforms. Only Linux names a thread in the kernel.

const std = @import("std");
const builtin = @import("builtin");

// -- api --------------------------------------------------------------------

/// Give the calling thread `name` in the kernel, for the `comm` column that perf prints. The
/// kernel keeps at most 15 bytes. A name that the kernel refuses changes only the perf label.
pub const nameCurrentThread = switch (builtin.os.tag) {
    .linux => nameCurrentThreadLinux,
    else => nameCurrentThreadNone,
};

// -- helpers ----------------------------------------------------------------

fn nameCurrentThreadLinux(name: []const u8) void {
    var z: [16:0]u8 = @splat(0);
    const n = @min(name.len, z.len - 1);
    @memcpy(z[0..n], name[0..n]);
    _ = std.posix.prctl(.SET_NAME, .{@intFromPtr(&z)}) catch {};
}

fn nameCurrentThreadNone(name: []const u8) void {
    _ = name;
}
