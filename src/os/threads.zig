//! The settings of a thread on each OS: the stack size of a pool worker, and the thread name that
//! perf and the trace show.
//!
//! Platforms. Only Linux names a thread in the kernel. Windows gives a pool worker a smaller stack
//! than the std default.

const std = @import("std");
const builtin = @import("builtin");

// -- constants --------------------------------------------------------------

/// The stack of one pool worker. Windows commits the whole stack when a thread starts, and the std
/// default of 16 MiB then costs 16 MiB of commit for each worker. 1 MiB is the default stack of a
/// Windows thread, and the deepest worker frame is the 64 KiB score row of an attention part
/// kernel. Another OS keeps the std default, because it backs a stack page at the first touch.
pub const pool_worker_stack_bytes: usize = switch (builtin.os.tag) {
    .windows => 1 << 20,
    else => std.Thread.SpawnConfig.default_stack_size,
};

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
