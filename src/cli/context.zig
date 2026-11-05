//! The value every command handler receives, and the buffer size its stdout writer uses.

const std = @import("std");
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
    /// Resolved once from `$ZIGFERENCER_MODELS_DIR`. See core/registry.zig.
    models_root: []const u8,
    /// Resolved once from `$ZIGFERENCER_CACHE_DIR`. See runtime/prompt_cache.zig.
    cache_dir: []const u8,
    /// Replaces the storage-buffer range that the device reports. Every command that opens a
    /// Vulkan context passes it.
    vk_storage_range: ?u64,
    /// True when stdout takes ANSI color. Commands build a `style.Palette` from it.
    color: bool,
    /// The same for stderr, which redirects apart from stdout. The pickers key their palette on it.
    color_err: bool,
};
