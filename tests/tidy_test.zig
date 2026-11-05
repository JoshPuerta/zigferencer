//! Repo-wide rules that no single module owns. The build runs this test with
//! the build root as the working directory.
//!
//! 1. No test waits on a clock. A test drives a clock pin or an injected
//!    clock, because a sleep makes the result depend on machine load. A
//!    source file that imports `std.testing` has no sleep call outside its
//!    test blocks either, because a test helper can hide one there.
//! 2. Non-test code does not print to stderr. Output goes through the log
//!    scopes. Two test-support files print a failing seed or a missing fixture.
//! 3. Every loop takes braces. An `if` statement takes braces, unless it is a
//!    guard: one line, no `else`, and a body that only leaves or asserts.
//! 4. A line holds at most 100 columns. A test name, a row of a data table, a
//!    URL, and a string of escaped JSON cannot wrap, so they are exempt.
//! 5. A file keeps its sections in the order `constants`, `types`, `state`,
//!    `api`, `helpers`, `tests`. A function in `api` is public, a function in
//!    `helpers` is private, and neither section declares a type.
//! 6. Every script keeps the rules that tools/README.md states and a program can check.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ast = std.zig.Ast;

// -- constants --------------------------------------------------------------

const max_file_bytes = 4 << 20;
const line_columns_max = 100;

const sleep_token = "sleep" ++ "(";
const testing_token = "std." ++ "testing";
const print_token = "std.debug." ++ "print(";

/// Files that import `std.testing` and sleep in production code on purpose.
const sleep_allowed = [_][]const u8{ "cli/client_cmd.zig", "core/span.zig" };

/// Files whose job is to print on a test failure.
const print_allowed = [_][]const u8{ "core/prop.zig", "core/test_fixtures.zig" };

// -- types ------------------------------------------------------------------

const Rule = enum { sleep, debug_print, braces, width, section, script };

/// The sections in file order. Every label under `tests` counts as `tests`.
const Section = enum { constants, types, state, api, helpers, tests };

const Violation = struct { rule: Rule, line: usize };

// -- helpers ----------------------------------------------------------------

fn firstViolation(
    source: []const u8,
    whole_file_is_test: bool,
    print_ok: bool,
    sleep_ok: bool,
) ?Violation {
    const imports_testing = std.mem.indexOf(u8, source, testing_token) != null;
    var in_test = whole_file_is_test;
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (std.mem.startsWith(u8, line, "test ")) {
            in_test = true;
        }

        defer if (!whole_file_is_test and std.mem.eql(u8, line, "}")) {
            in_test = false;
        };

        const code = std.mem.trimStart(u8, line, " ");
        if (std.mem.startsWith(u8, code, "//")) continue;

        const sleep_scope = in_test or (imports_testing and !sleep_ok);
        if (sleep_scope and hasToken(line, sleep_token)) {
            return .{ .rule = .sleep, .line = line_no };
        }
        if (!in_test and !print_ok and std.mem.indexOf(u8, line, print_token) != null) {
            return .{ .rule = .debug_print, .line = line_no };
        }
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

/// The first break of rules 3, 4, and 5 in `source`, a whole Zig file.
fn shapeViolation(gpa: Allocator, source: []const u8) !?Violation {
    if (widthViolation(source)) |v| return v;
    if (sectionViolation(source)) |v| return v;

    const terminated = try gpa.dupeZ(u8, source);
    defer gpa.free(terminated);

    var tree = try Ast.parse(gpa, terminated, .zig);
    defer tree.deinit(gpa);

    return bracesViolation(&tree);
}

fn widthViolation(source: []const u8) ?Violation {
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (line.len <= line_columns_max) continue;
        if (widthExempt(std.mem.trimStart(u8, line, " "))) continue;

        return .{ .rule = .width, .line = line_no };
    }

    return null;
}

fn widthExempt(code: []const u8) bool {
    if (std.mem.startsWith(u8, code, "test \"")) return true;
    if (std.mem.startsWith(u8, code, ".{") and std.mem.endsWith(u8, code, "},")) return true;
    if (std.mem.indexOf(u8, code, "https://") != null) return true;
    return std.mem.indexOf(u8, code, "\\\"") != null;
}

fn sectionViolation(source: []const u8) ?Violation {
    var section: ?Section = null;
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (sectionOf(line)) |next| {
            if (section) |current| {
                const backwards = @intFromEnum(next) < @intFromEnum(current);
                if (backwards) return .{ .rule = .section, .line = line_no };
            }
            section = next;
            continue;
        }

        const current = section orelse continue;
        if (current == .api and std.mem.startsWith(u8, line, "fn ")) {
            return .{ .rule = .section, .line = line_no };
        }
        if (current == .helpers and std.mem.startsWith(u8, line, "pub fn ")) {
            return .{ .rule = .section, .line = line_no };
        }
        if ((current == .api or current == .helpers) and declaresType(line)) {
            return .{ .rule = .section, .line = line_no };
        }
    }

    return null;
}

/// The section a banner line opens. A label such as `helpers: parse` names a
/// group of `helpers`, and a fixture banner opens the test part of the file.
fn sectionOf(line: []const u8) ?Section {
    const banner = "// -- ";
    if (!std.mem.startsWith(u8, line, banner)) return null;

    const label = line[banner.len..];
    const end = std.mem.indexOfAny(u8, label, ": ") orelse label.len;
    const word = label[0..end];
    if (std.mem.eql(u8, word, "fixtures")) return .tests;
    return std.meta.stringToEnum(Section, word);
}

fn declaresType(line: []const u8) bool {
    const code = if (std.mem.startsWith(u8, line, "pub ")) line["pub ".len..] else line;
    if (!std.mem.startsWith(u8, code, "const ")) return false;

    const kinds = [_][]const u8{ " = struct", " = enum", " = union", " = extern struct" };
    for (kinds) |kind| {
        if (std.mem.indexOf(u8, code, kind) != null) return true;
    }

    return false;
}

fn bracesViolation(tree: *const Ast) ?Violation {
    for (0..tree.nodes.len) |i| {
        const node: Ast.Node.Index = @enumFromInt(i);
        var buffer: [2]Ast.Node.Index = undefined;
        const statements = tree.blockStatements(&buffer, node) orelse continue;
        for (statements) |statement| {
            if (statementViolation(tree, statement)) |v| return v;
        }
    }

    return null;
}

fn statementViolation(tree: *const Ast, node: Ast.Node.Index) ?Violation {
    if (tree.fullFor(node)) |full| return bodyViolation(tree, full.ast.then_expr);
    if (tree.fullWhile(node)) |full| return bodyViolation(tree, full.ast.then_expr);

    const full = tree.fullIf(node) orelse return null;
    const then_body = full.ast.then_expr;
    if (full.ast.else_expr.unwrap()) |else_body| {
        if (bodyViolation(tree, then_body)) |v| return v;
        if (tree.fullIf(else_body) != null) return statementViolation(tree, else_body);
        return bodyViolation(tree, else_body);
    }

    if (isGuard(tree, then_body) and onOneLine(tree, node)) return null;
    return bodyViolation(tree, then_body);
}

fn bodyViolation(tree: *const Ast, body: Ast.Node.Index) ?Violation {
    switch (tree.nodeTag(body)) {
        .block, .block_two, .block_semicolon, .block_two_semicolon => return null,
        else => return .{ .rule = .braces, .line = lineOf(tree, tree.firstToken(body)) },
    }
}

fn isGuard(tree: *const Ast, body: Ast.Node.Index) bool {
    switch (tree.nodeTag(body)) {
        .@"return", .@"break", .@"continue", .unreachable_literal => return true,
        .builtin_call_two, .builtin_call_two_comma, .builtin_call, .builtin_call_comma => {
            const name = tree.tokenSlice(tree.nodeMainToken(body));
            return std.mem.eql(u8, name, "@compileError") or std.mem.eql(u8, name, "@panic");
        },
        else => {},
    }

    var buffer: [1]Ast.Node.Index = undefined;
    const call = tree.fullCall(&buffer, body) orelse return false;
    const callee = tree.tokenSlice(tree.lastToken(call.ast.fn_expr));
    return std.mem.eql(u8, callee, "assert");
}

fn onOneLine(tree: *const Ast, node: Ast.Node.Index) bool {
    return lineOf(tree, tree.firstToken(node)) == lineOf(tree, tree.lastToken(node));
}

fn lineOf(tree: *const Ast, token: Ast.TokenIndex) usize {
    return tree.tokenLocation(0, token).line + 1;
}

fn checkShapes(gpa: Allocator, root: []const u8) !void {
    const io = std.testing.io;
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(gpa);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".zig")) continue;

        const source = try dir.readFileAlloc(io, entry.path, gpa, .limited(max_file_bytes));
        defer gpa.free(source);

        if (try shapeViolation(gpa, source)) |v| {
            std.debug.print(
                "tidy: {s}/{s}:{d} breaks rule {s}\n",
                .{ root, entry.path, v.line, @tagName(v.rule) },
            );
            return error.TidyViolation;
        }
    }
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
            std.debug.print(
                "tidy: {s}/{s}:{d} breaks rule {s}\n",
                .{ root, entry.path, v.line, @tagName(v.rule) },
            );
            return error.TidyViolation;
        }
    }
}

/// The script rules of `tools/README.md` that a program can check: the width of every line, the
/// strict mode of a bash script, and the `main` of a Python script.
fn scriptViolation(path: []const u8, source: []const u8) ?Violation {
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (line.len > line_columns_max) return .{ .rule = .width, .line = line_no };
    }

    const bash = std.mem.startsWith(u8, source, "#!/usr/bin/env bash\n");
    if (bash and std.mem.indexOf(u8, source, "\nset -euo pipefail\n") == null) {
        return .{ .rule = .script, .line = 1 };
    }

    const python = std.mem.endsWith(u8, path, ".py");
    if (python and std.mem.indexOf(u8, source, "\ndef main(") == null) {
        return .{ .rule = .script, .line = 1 };
    }

    return null;
}

fn checkScript(gpa: Allocator, dir: std.Io.Dir, path: []const u8) !void {
    const io = std.testing.io;
    const source = try dir.readFileAlloc(io, path, gpa, .limited(max_file_bytes));
    defer gpa.free(source);

    if (scriptViolation(path, source)) |v| {
        std.debug.print("tidy: {s}:{d} breaks rule {s}\n", .{ path, v.line, @tagName(v.rule) });
        return error.TidyViolation;
    }
}

// -- tests ------------------------------------------------------------------

test "the scanner finds each rule and ignores comments, words, and test blocks" {
    const Case = struct {
        src: []const u8,
        whole_test: bool,
        print_ok: bool,
        want: ?Rule,
        sleep_ok: bool = false,
    };
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

test "the shape rules find a bare loop body, a long line, and a section out of place" {
    const long = "x" ** (line_columns_max + 1);
    const Case = struct { src: [:0]const u8, want: ?Rule };
    const cases = [_]Case{
        .{ .src = "fn f(a: u32) void {\n    for (0..a) |i| g(i);\n}\n", .want = .braces },
        .{ .src = "fn f(a: u32) void {\n    while (a) g();\n}\n", .want = .braces },
        .{ .src = "fn f(a: bool) void {\n    if (a) g();\n}\n", .want = .braces },
        .{ .src = "fn f(a: bool) void {\n    if (a) g() else h();\n}\n", .want = .braces },
        .{ .src = "fn f(a: bool) void {\n    if (a)\n        return;\n}\n", .want = .braces },
        .{ .src = "fn f(a: bool) void {\n    if (a) return;\n}\n", .want = null },
        .{ .src = "fn f(a: bool, b: bool) void {\n    if (a) assert(b);\n}\n", .want = null },
        .{ .src = "fn f(a: bool) void {\n    if (a) {\n        g();\n    }\n}\n", .want = null },
        .{ .src = "fn f(a: bool) u32 {\n    return if (a) 1 else 2;\n}\n", .want = null },
        .{ .src = "// " ++ long ++ "\n", .want = .width },
        .{ .src = "test \"" ++ long ++ "\" {}\n", .want = null },
        .{ .src = "// -- api --\nfn f() void {}\n", .want = .section },
        .{ .src = "// -- helpers --\npub fn f() void {}\n", .want = .section },
        .{ .src = "// -- helpers: parse --\nconst T = struct {};\n", .want = .section },
        .{ .src = "// -- api --\n// -- types --\n", .want = .section },
        .{ .src = "// -- types --\npub fn T() type {}\n// -- api --\npub fn f() void {}\n", .want = null },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("tidy shape case {d}\n", .{i});

        const got = try shapeViolation(std.testing.allocator, c.src);
        try std.testing.expectEqual(c.want, if (got) |v| v.rule else null);
    }
}

test "every source file keeps the shape rules" {
    const gpa = std.testing.allocator;
    try checkShapes(gpa, "src");
    try checkShapes(gpa, "tests");
    try checkShapes(gpa, "tools");

    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    const build_zig = try cwd.readFileAlloc(io, "build.zig", gpa, .limited(max_file_bytes));
    defer gpa.free(build_zig);

    try std.testing.expectEqual(@as(?Violation, null), try shapeViolation(gpa, build_zig));
}

test "no test sleeps and no non-test code prints" {
    try checkTree(std.testing.allocator, "tests", true);
    try checkTree(std.testing.allocator, "src", false);
}

test "the script rules find a long line, a missing strict mode, and a missing main" {
    const long = "x" ** (line_columns_max + 1);
    const Case = struct { path: []const u8, src: []const u8, want: ?Rule };
    const cases = [_]Case{
        .{ .path = "Makefile", .src = long ++ "\n", .want = .width },
        .{ .path = "a.sh", .src = "#!/usr/bin/env bash\nset -euo pipefail\n", .want = null },
        .{ .path = "a.sh", .src = "#!/usr/bin/env bash\nset -e\n", .want = .script },
        .{ .path = "common.sh", .src = "note() { :; }\n", .want = null },
        .{ .path = "a.py", .src = "#!/usr/bin/env python3\n\ndef main() -> int:\n", .want = null },
        .{ .path = "a.py", .src = "#!/usr/bin/env python3\nprint(1)\n", .want = .script },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("tidy script case {d}\n", .{i});

        const got = scriptViolation(c.path, c.src);
        try std.testing.expectEqual(c.want, if (got) |v| v.rule else null);
    }
}

test "every script and the Makefile keep the script rules" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();

    try checkScript(gpa, cwd, "Makefile");

    var dir = try cwd.openDir(io, "tools", .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(gpa);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.path, ".zig")) continue;
        if (std.mem.endsWith(u8, entry.path, ".md")) continue;

        try checkScript(gpa, dir, entry.path);
    }
}
