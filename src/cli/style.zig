//! Terminal styling for program output. The palette holds ANSI codes that are empty strings when
//! color is off, so a call site prints `{s}+{s}` with `p.green` and `p.reset` and needs no `if`.

const std = @import("std");
const Io = std.Io;

// -- types ------------------------------------------------------------------

pub const Palette = struct {
    reset: []const u8 = "",
    bold: []const u8 = "",
    dim: []const u8 = "",
    green: []const u8 = "",
    cyan: []const u8 = "",

    pub fn init(on: bool) Palette {
        return if (on) .{
            .reset = "\x1b[0m",
            .bold = "\x1b[1m",
            .dim = "\x1b[2m",
            .green = "\x1b[32m",
            .cyan = "\x1b[36m",
        } else .{};
    }
};

// -- api --------------------------------------------------------------------

/// True when stdout gets color: stdout is a terminal and `$NO_COLOR` is unset. The no-color.org
/// rule disables color at any value, even an empty one, so `no_color_set` means "the key is
/// present". The test comes first, so a run without color leaves the console mode alone.
pub fn stdoutColor(io: Io, no_color_set: bool) bool {
    if (no_color_set) return false;
    return enablesAnsi(Io.File.stdout(), io);
}

/// True when stderr gets color, by the rule of `stdoutColor`. The two streams redirect
/// independently, so the pickers key their palette on this value.
pub fn stderrColor(io: Io, no_color_set: bool) bool {
    if (no_color_set) return false;
    return enablesAnsi(Io.File.stderr(), io);
}

// -- helpers ----------------------------------------------------------------

/// True when `file` reads ANSI codes. On Windows the call also switches virtual terminal
/// processing on, which a console needs before it reads a code. Elsewhere it succeeds when the
/// file is a terminal.
fn enablesAnsi(file: Io.File, io: Io) bool {
    file.enableAnsiEscapeCodes(io) catch return false;
    return true;
}

// -- tests ------------------------------------------------------------------

test "Palette: off is all-empty, on carries codes" {
    const off = Palette.init(false);
    try std.testing.expectEqualStrings("", off.green);
    try std.testing.expectEqualStrings("", off.reset);

    const on = Palette.init(true);
    try std.testing.expect(on.green.len > 0);
    try std.testing.expect(on.reset.len > 0);
}
