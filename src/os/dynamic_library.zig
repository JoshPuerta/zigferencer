//! A shared library that a run opens by name: open it, and find a symbol as an untyped pointer.
//! The file names no library and no symbol of its own, so the caller holds both.
//!
//! Platforms. Linux opens through `std.DynLib`, which calls `dlopen` when the binary links libc.
//! Without libc, `std.DynLib` maps the file with its own ELF reader. That reader runs no
//! initializer and resolves no dependency, so such a build refuses every open. Windows calls
//! `LoadLibraryExW` with a search of the System32 directory alone. So a file in the application
//! directory or the current directory cannot take the place of a system library. Another OS
//! refuses every open.
//!
//! Lifetime. An open library stays open until the process ends, so a pointer from `lookup` stays
//! valid. A second open of one name gives the same library at the same address.
//!
//! Comptime. `open` takes its name at comptime, so the Windows wide name is a constant and no
//! name fails a conversion at run time.

const std = @import("std");
const builtin = @import("builtin");
const kernel32 = @import("windows_kernel32_declarations.zig");

// -- constants --------------------------------------------------------------

/// True when `std.DynLib` calls `dlopen` on Linux: libc, and not a static musl link. The test of
/// the musl ABI family is at least as strict as the `abi == .musl` test of `std.DynLib`.
const linux_dlopen = builtin.link_libc and
    !(builtin.abi.isMusl() and builtin.link_mode == .static);

pub const can_open: bool = switch (builtin.os.tag) {
    .linux => linux_dlopen,
    .windows => true,
    else => false,
};

// -- types ------------------------------------------------------------------

pub const OpenError = error{
    /// No library of this name opens.
    LibraryNotFound,
    /// This build opens no library.
    LibraryOpenUnsupported,
};

/// An open library.
pub const Library = struct {
    handle: Handle,
};

const Handle = switch (builtin.os.tag) {
    .linux => if (linux_dlopen) @FieldType(std.DynLib, "inner") else void,
    .windows => std.os.windows.HMODULE,
    else => void,
};

// -- api --------------------------------------------------------------------

/// Open the library `name`. Linux searches as `dlopen` does, and Windows searches System32 alone.
pub fn open(comptime name: [:0]const u8) OpenError!Library {
    comptime std.debug.assert(name.len > 0);
    if (!can_open) return error.LibraryOpenUnsupported;

    switch (builtin.os.tag) {
        .linux => {
            const handle = Handle.openZ(name.ptr) catch |err| switch (err) {
                error.FileNotFound, error.NameTooLong => return error.LibraryNotFound,
            };
            return .{ .handle = handle };
        },
        .windows => {
            const handle = kernel32.LoadLibraryExW(
                std.unicode.utf8ToUtf16LeStringLiteral(name),
                null,
                kernel32.load_library_search_system32,
            ) orelse return error.LibraryNotFound;
            return .{ .handle = handle };
        },
        else => comptime unreachable,
    }
}

/// The address of the symbol `name` in `library`, or null when the library has no such symbol.
pub fn lookup(library: *Library, name: [:0]const u8) ?*const anyopaque {
    std.debug.assert(name.len > 0);
    // No open gives a library in such a build.
    if (!can_open) unreachable;

    switch (builtin.os.tag) {
        .linux => return library.handle.lookup(*const anyopaque, name),
        .windows => {
            const address = kernel32.GetProcAddress(library.handle, name.ptr) orelse return null;
            return @ptrCast(address);
        },
        else => comptime unreachable,
    }
}

// -- tests ------------------------------------------------------------------

test "dynamic_library: a system library opens, and its symbol resolves" {
    // A Linux binary without libc opens no library, so this case needs the libc build.
    if (!can_open) return error.SkipZigTest;

    const name, const symbol = switch (builtin.os.tag) {
        .linux => .{ "libc.so.6", "getpid" },
        .windows => .{ "kernel32.dll", "GetCurrentProcessId" },
        else => comptime unreachable,
    };
    var library = try open(name);
    try std.testing.expect(lookup(&library, symbol) != null);
    try std.testing.expect(lookup(&library, "no_such_symbol_in_this_library") == null);

    var again = try open(name);
    try std.testing.expectEqual(lookup(&library, symbol), lookup(&again, symbol));
}

test "dynamic_library: a missing library fails with a typed error" {
    const want: OpenError = if (can_open) error.LibraryNotFound else error.LibraryOpenUnsupported;
    try std.testing.expectError(want, open("no-such-library-of-this-test-1"));
}
