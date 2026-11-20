//! Map a file read-only into memory with no copy. Two modes match the two
//! access patterns, and the difference is when the pages fault in.
//!
//!   `openReadOnly` prefaults every page at map time. The kernels stream the
//!   whole weight file once per token, so the faults cost less at load time
//!   than scattered through the decode loop.
//!
//!   `openReadOnlyLazy` skips the prefault. Pages fault in on first touch. A
//!   metadata-only reader touches only the descriptor pages, so it opens a
//!   multi-gigabyte file in milliseconds.
//!
//! Platforms. The standard library maps the file, and `os/file_map.zig` adds additional rules
//! for each OS. A heap copy in place of a map is a refusal, `error.ModelFileNotMapped`, because
//! the RAM plan does not count that memory.

const std = @import("std");
const file_map = @import("../os/file_map.zig");
const Io = std.Io;
const File = std.Io.File;

// -- types ------------------------------------------------------------------

pub const OpenError = File.OpenError || File.LengthError || File.MemoryMap.CreateError ||
    error{ModelFileNotMapped};

pub const Mapped = struct {
    io: Io,
    file: File,
    mm: File.MemoryMap,

    pub fn bytes(self: *const Mapped) []const u8 {
        return self.mm.memory;
    }

    pub fn close(self: *Mapped) void {
        self.mm.destroy(self.io);
        self.file.close(self.io);
    }
};

// -- api --------------------------------------------------------------------

/// Map `path` read-only and prefault every page. Use it for a reader that
/// consumes the whole file, such as inference weights or a cache snapshot.
pub fn openReadOnly(io: Io, path: []const u8) OpenError!Mapped {
    return open(io, path, true);
}

/// Map `path` read-only without a prefault. Pages fault in on first access.
/// Use it for a metadata-only reader such as GGUF inspection, where only the
/// descriptor pages fault in and a multi-gigabyte file opens in milliseconds.
pub fn openReadOnlyLazy(io: Io, path: []const u8) OpenError!Mapped {
    return open(io, path, false);
}

// -- helpers ----------------------------------------------------------------

fn open(io: Io, path: []const u8, prefault: bool) OpenError!Mapped {
    var cwd = Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    errdefer file.close(io);

    const len = try file.length(io);
    var mm = try File.MemoryMap.create(io, file, .{
        .len = @intCast(len),
        .protection = .{ .read = true, .write = false },
        .populate = file_map.populate(prefault),
    });
    errdefer mm.destroy(io);

    // The map starts at offset 0, because std ignores the offset on Windows.
    std.debug.assert(mm.offset == 0);
    std.debug.assert(mm.memory.len == len);

    if (file_map.isHeapCopy(mm)) return error.ModelFileNotMapped;
    if (prefault) {
        file_map.prefault(path, mm.memory);
    }

    return .{ .io = io, .file = file, .mm = mm };
}

// -- tests ------------------------------------------------------------------

test "both open modes map the bytes that a plain read returns" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Three pages and a partial page, so the map has a tail past a page boundary.
    var content: [3 * std.heap.page_size_min + 123]u8 = undefined;
    for (&content, 0..) |*byte, i| {
        byte.* = @truncate(i *% 31 +% 7);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "map.bin", .data = &content });

    var pathbuf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&pathbuf, ".zig-cache/tmp/{s}/map.bin", .{&tmp.sub_path});
    const read = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(read);

    const opens = [_]*const fn (Io, []const u8) OpenError!Mapped{ openReadOnly, openReadOnlyLazy };
    for (opens) |openFn| {
        var map = try openFn(io, path);
        defer map.close();

        try std.testing.expect(map.mm.section != null);
        try std.testing.expectEqualSlices(u8, read, map.bytes());
    }
}
