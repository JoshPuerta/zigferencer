//! The AFD layout of the poll request, which the Zig standard library does not declare: the poll
//! record, its handle record, and the event bits. Std declares the request codes and the other AFD
//! structs that the serve socket calls send.
//!
//! Names. A struct and its fields keep the names in common use for the AFD driver, in the form of
//! `windows_kernel32_declarations.zig`. A constant has a snake_case name, and its doc comment gives
//! the uppercase name.
//!
//! Comptime. Each struct has a size check below it. A wrong layout fails the build, not a run. The
//! sizes are those of the 64-bit layout, so a 32-bit target fails at the check.

const std = @import("std");
const windows = std.os.windows;

// -- constants --------------------------------------------------------------

/// `AFD_POLL_DISCONNECT`: the peer closed the connection or its send side.
pub const poll_disconnect: windows.ULONG = 0x0008;

/// `AFD_POLL_ABORT`: the peer reset the connection.
pub const poll_abort: windows.ULONG = 0x0010;

// -- types ------------------------------------------------------------------

/// `AFD_POLL_HANDLE_INFO`. The kernel writes the events that it found into `Events`.
pub const AFD_POLL_HANDLE_INFO = extern struct {
    Handle: windows.HANDLE,
    Events: windows.ULONG,
    Status: windows.NTSTATUS,
};

comptime {
    std.debug.assert(@sizeOf(AFD_POLL_HANDLE_INFO) == 16);
}

/// `AFD_POLL_INFO` with one handle. A negative `Timeout` is a count of 100 ns from now, and a
/// positive one is an absolute time. The kernel sets `NumberOfHandles` to the handles with events.
pub const AFD_POLL_INFO = extern struct {
    Timeout: windows.LARGE_INTEGER,
    NumberOfHandles: windows.ULONG,
    Exclusive: windows.ULONG,
    Handles: [1]AFD_POLL_HANDLE_INFO,
};

comptime {
    std.debug.assert(@sizeOf(AFD_POLL_INFO) == 32);
}
