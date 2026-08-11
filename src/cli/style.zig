//! Terminal styling for program output. The palette holds ANSI codes that become
//! empty strings when color is off (piped output, $NO_COLOR, or a non-TTY), so
//! call sites stay branch-free.
//!
//! `w.print("{s}+{s}", .{ p.green, p.reset })` is correct in both modes. With
//! color off the codes are absent, and no span needs an `if`. Color is a
//! presentation choice. The CLI resolves it once and carries it on the Context.

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

/// True when stdout gets color: stdout is a TTY and `$NO_COLOR` is unset. This
/// follows the no-color.org rule. The variable disables color at any value, even
/// an empty one, so `no_color_set` means "the key is present".
pub fn stdoutColor(io: Io, no_color_set: bool) bool {
    if (no_color_set) return false;
    return Io.File.stdout().isTty(io) catch false;
}

/// True when stderr gets color, by the rule of `stdoutColor`. The CLI resolves
/// the two streams separately because they redirect independently. `inspect >
/// card.txt` leaves the picker on a color terminal while stdout is a file, so the
/// picker keys its palette on this value.
pub fn stderrColor(io: Io, no_color_set: bool) bool {
    if (no_color_set) return false;
    return Io.File.stderr().isTty(io) catch false;
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
