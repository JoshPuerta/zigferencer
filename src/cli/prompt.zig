//! The stdin and stderr prompts the commands share. Prompts go to stderr, so stdout stays program
//! output. EOF, a read error, and a non-TTY stdin all resolve to "no" or "no selection", so a piped
//! run never blocks on a prompt.

const std = @import("std");
const Io = std.Io;

const style = @import("style.zig");

// -- constants --------------------------------------------------------------

/// The stack buffer of the confirm question.
const question_buffer_bytes = 256;

/// The stack buffer of one answer line. A longer line is not an answer.
const answer_buffer_bytes = 64;

// -- api --------------------------------------------------------------------

/// True when stdin is an interactive terminal. A caller checks this before it prompts.
pub fn stdinIsTty(io: Io) bool {
    return Io.File.stdin().isTty(io) catch false;
}

/// Print `question` and ` [y/N] ` to stderr, then read a yes or no answer. Anything but a leading
/// 'y' or 'Y' is false, because false is the safe default ahead of a destructive action.
pub fn confirm(io: Io, p: style.Palette, question: []const u8) bool {
    var stderr_buffer: [question_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    stderr_writer.interface.print("{s}[?]{s} {s} {s}[y/N]{s} ", .{
        p.bold, p.reset, question, p.dim, p.reset,
    }) catch return false;
    stderr_writer.interface.flush() catch return false;

    var stdin_buffer: [answer_buffer_bytes]u8 = undefined;
    var stdin_reader: Io.File.Reader = Io.File.stdin().reader(io, &stdin_buffer);
    const line = stdin_reader.interface.takeDelimiterExclusive('\n') catch return false;
    return isAffirmative(line);
}

/// Read a 1-based menu choice from stdin and return its 0-based index, or null to abort. A bare
/// Enter selects `default_index`, and a null default turns Enter into an abort. The caller prints
/// the list.
pub fn chooseIndex(io: Io, count: usize, default_index: ?usize) ?usize {
    var stdin_buffer: [answer_buffer_bytes]u8 = undefined;
    var stdin_reader: Io.File.Reader = Io.File.stdin().reader(io, &stdin_buffer);
    const line = stdin_reader.interface.takeDelimiterExclusive('\n') catch return null;
    return parseChoice(line, count, default_index);
}

// -- helpers ----------------------------------------------------------------

/// A leading 'y' or 'Y' after the trim is a yes.
fn isAffirmative(line: []const u8) bool {
    const a = std.mem.trim(u8, line, " \t\r");
    return a.len > 0 and (a[0] == 'y' or a[0] == 'Y');
}

/// An empty line gives `default_index`, a decimal in 1..=count gives its 0-based index, and
/// anything else gives null.
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
