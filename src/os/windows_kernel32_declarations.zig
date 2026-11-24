//! The kernel32 functions that this project calls on Windows, with their structs and constants.
//! The Zig standard library declares none of them, so this file is the one place to check a
//! signature against the Windows documentation.
//!
//! Names. A function, a struct field, and a parameter keep the name of the Windows documentation,
//! so a reader can compare each one with its page. A constant has a snake_case name, and its doc
//! comment gives the Windows name.
//!
//! Comptime. Each struct has a size check below it. A wrong layout fails the build, not a run.
//!
//! Use. A caller reaches a function only under `builtin.os.tag == .windows`. A build for another
//! OS never analyzes an `extern` here, so it needs no kernel32.

const std = @import("std");
const windows = std.os.windows;

// -- constants --------------------------------------------------------------

/// `RelationProcessorCore`: one record for each physical core.
pub const relation_processor_core: u32 = 0;

/// `LOAD_LIBRARY_SEARCH_SYSTEM32`: search the System32 directory alone, for the library and for
/// the libraries that it loads.
pub const load_library_search_system32: windows.DWORD = 0x0000_0800;

// -- types ------------------------------------------------------------------

/// `MEMORYSTATUSEX`. The caller sets `dwLength` to the struct size before the call.
pub const MEMORYSTATUSEX = extern struct {
    dwLength: windows.DWORD,
    dwMemoryLoad: windows.DWORD,
    ullTotalPhys: u64,
    ullAvailPhys: u64,
    ullTotalPageFile: u64,
    ullAvailPageFile: u64,
    ullTotalVirtual: u64,
    ullAvailVirtual: u64,
    ullAvailExtendedVirtual: u64,
};

comptime {
    std.debug.assert(@sizeOf(MEMORYSTATUSEX) == 64);
}

/// The first two fields of `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`. `Size` is the byte count of
/// the whole record, so a reader walks the buffer by it and needs no other field to count records.
pub const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX_HEADER = extern struct {
    Relationship: u32,
    Size: windows.DWORD,
};

comptime {
    std.debug.assert(@sizeOf(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX_HEADER) == 8);
}

/// `WIN32_MEMORY_RANGE_ENTRY`.
pub const WIN32_MEMORY_RANGE_ENTRY = extern struct {
    VirtualAddress: *const anyopaque,
    NumberOfBytes: windows.SIZE_T,
};

comptime {
    std.debug.assert(@sizeOf(WIN32_MEMORY_RANGE_ENTRY) == 2 * @sizeOf(usize));
}

// -- api --------------------------------------------------------------------

/// A null `lpOverlapped` cancels every request of `hFile` from every thread of the process.
pub extern "kernel32" fn CancelIoEx(
    hFile: windows.HANDLE,
    lpOverlapped: ?*anyopaque,
) callconv(.winapi) windows.BOOL;

pub extern "kernel32" fn GlobalMemoryStatusEx(
    lpBuffer: *MEMORYSTATUSEX,
) callconv(.winapi) windows.BOOL;

/// `Buffer` holds records of variable size. A call with a buffer that is too small fails, and
/// `ReturnedLength` then holds the byte count that the records need.
pub extern "kernel32" fn GetLogicalProcessorInformationEx(
    RelationshipType: u32,
    Buffer: ?[*]align(@alignOf(u64)) u8,
    ReturnedLength: *windows.DWORD,
) callconv(.winapi) windows.BOOL;

pub extern "kernel32" fn PrefetchVirtualMemory(
    hProcess: windows.HANDLE,
    NumberOfEntries: windows.ULONG_PTR,
    VirtualAddresses: [*]const WIN32_MEMORY_RANGE_ENTRY,
    Flags: windows.ULONG,
) callconv(.winapi) windows.BOOL;

pub extern "kernel32" fn LoadLibraryExW(
    lpLibFileName: [*:0]const u16,
    hFile: ?windows.HANDLE,
    dwFlags: windows.DWORD,
) callconv(.winapi) ?windows.HMODULE;

pub extern "kernel32" fn GetProcAddress(
    hModule: windows.HMODULE,
    lpProcName: [*:0]const u8,
) callconv(.winapi) ?windows.FARPROC;
