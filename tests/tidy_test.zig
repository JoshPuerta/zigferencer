//! Repo-wide rules that no single module owns. The build runs this test with
//! the build root as the working directory.
//!
//! 1. No test waits on a clock. A test drives a clock pin or an injected
//!    clock, because a sleep makes the result depend on machine load. A
//!    source file that imports `std.testing` has no sleep call outside its
//!    test blocks either, because a test helper can hide one there.
//! 2. Non-test code does not print to stderr. Output goes through the log
//!    scopes. Two test-support files print a failing seed or a missing fixture.

const std = @import("std");
const Allocator = std.mem.Allocator;

// -- types ------------------------------------------------------------------

const Rule = enum { sleep, debug_print };

const Violation = struct { rule: Rule, line: usize };

const max_file_bytes = 4 << 20;

const sleep_token = "sleep" ++ "(";
const testing_token = "std." ++ "testing";
const print_token = "std.debug." ++ "print(";

/// Files that import `std.testing` and sleep in production code on purpose.
const sleep_allowed = [_][]const u8{ "cli/client_cmd.zig", "core/span.zig" };

/// Files whose job is to print on a test failure.
const print_allowed = [_][]const u8{ "core/prop.zig", "core/test_fixtures.zig" };

// -- helpers ----------------------------------------------------------------

fn firstViolation(source: []const u8, whole_file_is_test: bool, print_ok: bool, sleep_ok: bool) ?Violation {
    const imports_testing = std.mem.indexOf(u8, source, testing_token) != null;
    var in_test = whole_file_is_test;
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (std.mem.startsWith(u8, line, "test ")) in_test = true;
        defer if (!whole_file_is_test and std.mem.eql(u8, line, "}")) {
            in_test = false;
        };
        const code = std.mem.trimStart(u8, line, " ");
        if (std.mem.startsWith(u8, code, "//")) continue;
        const sleep_scope = in_test or (imports_testing and !sleep_ok);
        if (sleep_scope and hasToken(line, sleep_token)) return .{ .rule = .sleep, .line = line_no };
        if (!in_test and !print_ok and std.mem.indexOf(u8, line, print_token) != null)
            return .{ .rule = .debug_print, .line = line_no };
    }
    return null;
}

/// True when `token` starts at a word boundary, so `awaitAllAsleep(` is no sleep call.
fn hasToken(line: []const u8, token: []const u8) bool {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, line, from, token)) |at| : (from = at + 1) {
        if (at == 0 or !std.ascii.isAlphabetic(line[at - 1])) return true;
    }
    return false;
}

fn checkTree(gpa: Allocator, root: []const u8, whole_file_is_test: bool) !void {
    const io = std.testing.io;
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(gpa);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".zig")) continue;
        if (std.mem.eql(u8, entry.path, "tidy_test.zig")) continue;
        const source = try dir.readFileAlloc(io, entry.path, gpa, .limited(max_file_bytes));
        defer gpa.free(source);
        const print_ok = for (print_allowed) |allowed| {
            if (std.mem.eql(u8, entry.path, allowed)) break true;
        } else false;
        const sleep_ok = for (sleep_allowed) |allowed| {
            if (std.mem.eql(u8, entry.path, allowed)) break true;
        } else false;
        if (firstViolation(source, whole_file_is_test, print_ok, sleep_ok)) |v| {
            std.debug.print("tidy: {s}/{s}:{d} breaks rule {s}\n", .{ root, entry.path, v.line, @tagName(v.rule) });
            return error.TidyViolation;
        }
    }
}

// -- tests ------------------------------------------------------------------

test "the scanner finds each rule and ignores comments, words, and test blocks" {
    const Case = struct { src: []const u8, whole_test: bool, print_ok: bool, want: ?Rule, sleep_ok: bool = false };
    const cases = [_]Case{
        .{ .src = "test \"t\" {\n    io." ++ sleep_token ++ ")\n}\n", .whole_test = false, .print_ok = false, .want = .sleep },
        .{ .src = "fn f() void {\n    io." ++ sleep_token ++ ")\n}\n", .whole_test = false, .print_ok = false, .want = null },
        .{ .src = "fn f() void { x." ++ sleep_token ++ ") }\n", .whole_test = true, .print_ok = false, .want = .sleep },
        .{ .src = "fn awaitAllAsleep(p: *P) void {}\ntest \"t\" {\n    awaitAllAsleep(&p);\n}\n", .whole_test = false, .print_ok = false, .want = null },
        .{ .src = "// io." ++ sleep_token ++ ")\n", .whole_test = true, .print_ok = false, .want = null },
        .{ .src = "fn f() void {\n    " ++ print_token ++ "\"x\", .{});\n}\n", .whole_test = false, .print_ok = false, .want = .debug_print },
        .{ .src = "fn f() void {\n    " ++ print_token ++ "\"x\", .{});\n}\n", .whole_test = false, .print_ok = true, .want = null },
        .{ .src = "test \"t\" {\n    " ++ print_token ++ "\"x\", .{});\n}\n", .whole_test = false, .print_ok = false, .want = null },
        .{ .src = "/// Like " ++ print_token ++ ")\nfn f() void {}\n", .whole_test = false, .print_ok = false, .want = null },
        .{ .src = "const t = " ++ testing_token ++ ";\nfn f() void {\n    io." ++ sleep_token ++ ")\n}\n", .whole_test = false, .print_ok = false, .want = .sleep },
        .{ .src = "const t = " ++ testing_token ++ ";\nfn f() void {\n    io." ++ sleep_token ++ ")\n}\n", .whole_test = false, .print_ok = false, .want = null, .sleep_ok = true },
        .{ .src = "const t = " ++ testing_token ++ ";\nfn awaitAllAsleep(p: *P) void {}\n", .whole_test = false, .print_ok = false, .want = null },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("tidy scanner case {d}\n", .{i});
        const got = firstViolation(c.src, c.whole_test, c.print_ok, c.sleep_ok);
        try std.testing.expectEqual(c.want, if (got) |v| v.rule else null);
    }
}

test "no test sleeps and no non-test code prints" {
    try checkTree(std.testing.allocator, "tests", true);
    try checkTree(std.testing.allocator, "src", false);
}
