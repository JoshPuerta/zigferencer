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
//! Portability belongs to the standard library. On a platform without mmap the
//! Io vtable picks the fallback and this surface does not change.

const std = @import("std");
const Io = std.Io;
const File = std.Io.File;

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

/// Map `path` read-only and prefault every page. Use it for a reader that
/// consumes the whole file, such as inference weights or a cache snapshot.
pub fn openReadOnly(io: Io, path: []const u8) !Mapped {
    return open(io, path, true);
}

/// Map `path` read-only without a prefault. Pages fault in on first access.
/// Use it for a metadata-only reader such as GGUF inspection, where only the
/// descriptor pages fault in and a multi-gigabyte file opens in milliseconds.
pub fn openReadOnlyLazy(io: Io, path: []const u8) !Mapped {
    return open(io, path, false);
}

fn open(io: Io, path: []const u8, populate: bool) !Mapped {
    var cwd = Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    errdefer file.close(io);
    const len = try file.length(io);
    var mm = try File.MemoryMap.create(io, file, .{
        .len = @intCast(len),
        .protection = .{ .read = true, .write = false },
        .populate = populate,
    });
    errdefer mm.destroy(io);
    return .{ .io = io, .file = file, .mm = mm };
}
