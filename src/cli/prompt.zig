//! Interactive CLI shell: the stdin and stderr prompts the subcommands share.
//!
//! Prompts and their framing go to stderr, next to the other diagnostics. Stdout
//! stays reserved for program output, so `inspect > card.txt` still shows its
//! picker on the terminal. Input is one line from stdin. EOF, a read error, and a
//! non-interactive stdin all resolve to "no" or "no selection". A piped
//! invocation therefore never blocks on a prompt that nothing can answer.
//!
//! The reads are the imperative shell. The parse of the typed line is a pure
//! function (`isAffirmative`, `parseChoice`), so the tests cover the decision
//! logic without a terminal.

const std = @import("std");
const Io = std.Io;

const style = @import("style.zig");

// -- api --------------------------------------------------------------------

/// True when stdin is an interactive terminal. A caller checks this gate before
/// it prompts. A pipe, a file, or /dev/null reads as false.
pub fn stdinIsTty(io: Io) bool {
    return Io.File.stdin().isTty(io) catch false;
}

/// Print `question` and ` [y/N] ` to stderr, then read a yes or no answer. A line
/// that does not start with 'y' or 'Y' is false, and so are EOF and a read error.
/// False is the safe default ahead of a destructive action. A caller that needs a
/// distinct message on a non-TTY checks `stdinIsTty` first.
pub fn confirm(io: Io, p: style.Palette, question: []const u8) bool {
    var out_buf: [256]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &out_buf);
    fw.interface.print("{s}[?]{s} {s} {s}[y/N]{s} ", .{ p.bold, p.reset, question, p.dim, p.reset }) catch return false;
    fw.interface.flush() catch return false;

    var in_buf: [64]u8 = undefined;
    var fr: Io.File.Reader = Io.File.stdin().reader(io, &in_buf);
    const line = fr.interface.takeDelimiterExclusive('\n') catch return false;
    return isAffirmative(line);
}

/// Read a 1-based menu choice from stdin and return its 0-based index, or null to
/// abort. An abort is EOF, a read error, or an out-of-range or non-numeric answer.
/// An empty line (a bare Enter) selects `default_index`. That default can be null,
/// which turns Enter into an abort for a destructive action. The caller prints the
/// numbered list, because the items are its own. This function reads the answer.
pub fn chooseIndex(io: Io, count: usize, default_index: ?usize) ?usize {
    var in_buf: [64]u8 = undefined;
    var fr: Io.File.Reader = Io.File.stdin().reader(io, &in_buf);
    const line = fr.interface.takeDelimiterExclusive('\n') catch return null;
    return parseChoice(line, count, default_index);
}

// -- helpers ----------------------------------------------------------------

/// Pure parse of a yes or no line. A leading 'y' or 'Y' after the trim is a yes.
fn isAffirmative(line: []const u8) bool {
    const a = std.mem.trim(u8, line, " \t\r");
    return a.len > 0 and (a[0] == 'y' or a[0] == 'Y');
}

/// Pure parse of a picker answer. A trimmed empty line gives `default_index`. A
/// decimal in 1..=count gives its 0-based index. Anything else gives null.
fn parseChoice(line: []const u8, count: usize, default_index: ?usize) ?usize {
    const s = std.mem.trim(u8, line, " \t\r");
    if (s.len == 0) return default_index;
    const choice = std.fmt.parseInt(usize, s, 10) catch return null;
    if (choice < 1 or choice > count) return null;
    return choice - 1;
}

// -- tests ------------------------------------------------------------------

test "isAffirmative: only a leading y/Y is yes" {
    try std.testing.expect(isAffirmative("y"));
    try std.testing.expect(isAffirmative("Y"));
    try std.testing.expect(isAffirmative("  yes \n"));
    try std.testing.expect(!isAffirmative(""));
    try std.testing.expect(!isAffirmative("n"));
    try std.testing.expect(!isAffirmative("no"));
}

test "parseChoice: empty line falls back to the default" {
    try std.testing.expectEqual(@as(?usize, 0), parseChoice("", 3, 0));
    try std.testing.expectEqual(@as(?usize, 2), parseChoice("  \r", 3, 2));
    try std.testing.expectEqual(@as(?usize, null), parseChoice("", 3, null));
}

test "parseChoice: a valid 1-based choice maps to a 0-based index" {
    try std.testing.expectEqual(@as(?usize, 0), parseChoice("1", 3, null));
    try std.testing.expectEqual(@as(?usize, 2), parseChoice(" 3 ", 3, null));
}

test "parseChoice: out-of-range or non-numeric is a null (abort)" {
    try std.testing.expectEqual(@as(?usize, null), parseChoice("0", 3, 1));
    try std.testing.expectEqual(@as(?usize, null), parseChoice("4", 3, 1));
    try std.testing.expectEqual(@as(?usize, null), parseChoice("x", 3, 1));
}
