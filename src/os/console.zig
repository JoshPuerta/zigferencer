//! The output code page of a console. The standard library writes console bytes raw, and the model
//! text is UTF-8, so a Windows console shows it wrong unless its output code page is UTF-8.
//!
//! Platforms. Only Windows has a code page. Both functions do nothing on another OS.
//!
//! Console. Each request names stdout. The request fails when stdout is no console of this
//! process, so a redirected stdout keeps its code page.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

// -- constants --------------------------------------------------------------

/// `CP_UTF8`: the Windows code page number of UTF-8.
const utf8_code_page: u32 = 65001;

// -- api --------------------------------------------------------------------

/// Set the output code page to UTF-8 when stdout is a console. Return the old code page. Return
/// null when no change took place: stdout is no console, a request failed, the code page was
/// already UTF-8, or the OS has no code page.
pub const setUtf8Output = switch (builtin.os.tag) {
    .windows => setUtf8OutputWindows,
    else => setUtf8OutputNone,
};

/// Set the output code page back to `old_code_page`, when it is not null.
pub const restoreOutput = switch (builtin.os.tag) {
    .windows => restoreOutputWindows,
    else => restoreOutputNone,
};

// -- helpers ----------------------------------------------------------------

fn setUtf8OutputWindows(io: Io) ?u32 {
    var get_cp = std.os.windows.CONSOLE.USER_IO.GET_CP(.Output);
    const get_status = get_cp.operate(io, Io.File.stdout()) catch return null;
    if (get_status != .SUCCESS) return null;

    const old_code_page: u32 = get_cp.Data.CodePage;
    if (old_code_page == utf8_code_page) return null;

    var set_cp = std.os.windows.CONSOLE.USER_IO.SET_CP(.Output, utf8_code_page);
    const set_status = set_cp.operate(io, Io.File.stdout()) catch return null;
    if (set_status != .SUCCESS) return null;

    return old_code_page;
}

fn restoreOutputWindows(io: Io, old_code_page: ?u32) void {
    const code_page = old_code_page orelse return;
    var set_cp = std.os.windows.CONSOLE.USER_IO.SET_CP(.Output, code_page);

    // The process exits next, so a failure leaves nothing to repair.
    _ = set_cp.operate(io, Io.File.stdout()) catch {};
}

fn setUtf8OutputNone(io: Io) ?u32 {
    _ = io;
    return null;
}

fn restoreOutputNone(io: Io, old_code_page: ?u32) void {
    _ = io;
    _ = old_code_page;
}

// -- tests ------------------------------------------------------------------

test "setUtf8Output: a test run changes nothing off Windows" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;

    try std.testing.expectEqual(@as(?u32, null), setUtf8Output(std.testing.io));
    restoreOutput(std.testing.io, null);
}
