//! The value every command handler receives, and the buffer size its stdout writer uses.

const std = @import("std");
const limits = @import("../core/limits.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;

// -- constants --------------------------------------------------------------

/// The buffer of the one stdout writer.
pub const stdout_buffer_bytes = 4096;

// -- types ------------------------------------------------------------------

pub const Context = struct {
    io: Io,
    arena: Allocator,
    gpa: Allocator,
    args: []const [:0]const u8,
    /// The one stdout writer of the process. `entry` owns it, so its address stays stable. A
    /// command flushes it at its own points. `entry` reads the cause of a failed write here.
    stdout: *Io.File.Writer,
    /// Resolved once from `$ZIGFERENCER_MODELS_DIR`.
    models_root: []const u8,
    /// Resolved once from `$ZIGFERENCER_CACHE_DIR`.
    cache_dir: []const u8,
    /// The device choices of the environment. A command passes them to every Vulkan context.
    vk_options: limits.DeviceOptions,
    /// True when stdout takes ANSI color.
    color: bool,
    /// The same for stderr, which redirects apart from stdout.
    color_err: bool,
};
