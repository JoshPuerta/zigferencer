//! What a read-only map of a file needs on each OS, beside the map that the standard library
//! makes: the populate flag, the test for a heap copy, the prefault call, and the refusal text.
//!
//! Platforms. On Linux the populate flag prefaults, the heap-copy test is false, and the
//! prefault call does nothing. On Windows the populate flag commits the section, a heap copy in
//! place of a map is a refusal, and the prefault is one `PrefetchVirtualMemory` call. Another OS
//! behaves as Linux.

const std = @import("std");
const builtin = @import("builtin");
const kernel32 = @import("windows_kernel32_declarations.zig");
const File = std.Io.File;

const log = std.log.scoped(.gguf);

// -- constants --------------------------------------------------------------

/// The text of the refusal for a heap copy. A caller logs it after the file name.
pub const not_mapped_message = "Windows cannot memory-map the file. A network share is the " ++
    "usual cause. Copy the file to a local disk.";

// -- api --------------------------------------------------------------------

/// The `populate` option of the map, for a caller that asks for a prefault or not.
pub fn populate(prefault_wanted: bool) bool {
    return switch (builtin.os.tag) {
        // The flag means commit, the default attribute of a file map, and an empty attribute set
        // can fail.
        .windows => true,
        else => prefault_wanted,
    };
}

/// True when std gave a heap copy of the file in place of a map. The RAM plan does not count that
/// memory, so the caller refuses it.
pub fn isHeapCopy(mm: File.MemoryMap) bool {
    return switch (builtin.os.tag) {
        .windows => mm.section == null,
        else => false,
    };
}

/// Read the whole map into memory before the first touch, where the populate flag does not.
pub const prefault = switch (builtin.os.tag) {
    .windows => prefaultWindows,
    else => prefaultNone,
};

// -- helpers ----------------------------------------------------------------

/// Ask Windows to read the whole map into the file cache. A failure costs only speed, because
/// the pages then fault in on first touch.
fn prefaultWindows(path: []const u8, memory: []const u8) void {
    if (memory.len == 0) return;

    const windows = std.os.windows;
    const ranges = [_]kernel32.WIN32_MEMORY_RANGE_ENTRY{
        .{ .VirtualAddress = memory.ptr, .NumberOfBytes = memory.len },
    };
    const ok = kernel32.PrefetchVirtualMemory(windows.current_process, ranges.len, &ranges, 0);
    if (!ok.toBool()) {
        log.warn("{s}: the prefetch of the map failed ({t}), so pages fault in on first touch", .{
            std.Io.Dir.path.basename(path), windows.GetLastError(),
        });
    }
}

fn prefaultNone(path: []const u8, memory: []const u8) void {
    _ = path;
    _ = memory;
}
