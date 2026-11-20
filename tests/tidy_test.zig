//! Repo-wide rules that no single module owns. The build runs this test with
//! the build root as the working directory. Each test reports every break as
//! `file:line: rule`, and then fails.
//!
//! 1. No test waits on a clock. A test drives a clock pin or an injected
//!    clock, because a sleep makes the result depend on machine load. A
//!    source file that imports `std.testing` has no sleep call outside its
//!    test blocks either, because a test helper can hide one there.
//! 2. Non-test code does not print to stderr. Output goes through the log
//!    scopes. Two test-support files print a failing seed or a missing fixture.
//!    An engine file opens no stream either, because a host of the library owns
//!    them. Three files write to one on purpose.
//! 3. Every loop takes braces. An `if` statement takes braces, unless it is a
//!    guard: one line, no `else`, and a body that only leaves or asserts.
//! 4. A line holds at most 100 columns. Only two kinds of line can run past:
//!    a line with a URL, and a line with one string literal alone, because
//!    grep must find a sentence whole. That literal is a `"..."` with an
//!    optional `++` or comma, or one `\\` line.
//! 5. A file keeps its sections in the order `constants`, `types`, `state`,
//!    `api`, `helpers`, `tests`. A function in `api` is public, a function in
//!    `helpers` is private, and neither section declares a type.
//! 6. Every script keeps the rules that tools/README.md states and a program can check.
//! 7. An import points down the layers. A file in `src/os/`, the lowest layer,
//!    reads only `std`, `builtin`, and the files of `src/os/`. The engine reads
//!    no CLI file. The CLI building blocks read no command. No command reads
//!    another command. No file outside `src/dev/` reads a developer file.
//! 8. An interior comment block in `src/` and `tools/` holds at most 6 lines.
//!    Test blocks are exempt.
//! 9. A doc comment in `src/` and `tools/` holds at most 3 lines on a function,
//!    2 on a field, and 4 on a type or a constant. Test blocks are exempt.
//! 10. A `//!` header in `src/` and `tools/` holds at most 100 lines.
//! 11. No import alias of one or two letters, except `vk`. This rule covers
//!     `src/`, `tests/`, and `tools/`.
//! 12. A file that holds the keyword `comptime` states what it does in a
//!     `Comptime.` paragraph of its `//!` header. This rule covers `src/`,
//!     `tests/`, `tools/`, and `build.zig`.
//! 13. An OS condition lives only in `src/os/`: no other file names
//!     `builtin.os.tag`, `builtin.target.os`, `std.os.linux`, `std.os.windows`,
//!     or `std.posix`. This rule covers `src/`, `tests/`, and `tools/`. Test
//!     blocks are exempt. Four files make raw socket calls or branch on
//!     purpose. `build.zig` makes the build decisions of each OS, so the rule
//!     does not read it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Ast = std.zig.Ast;

// -- constants --------------------------------------------------------------

const max_file_bytes = 4 << 20;
const line_columns_max = 100;
const block_lines_max = 6;
const fn_doc_lines_max = 3;
const field_doc_lines_max = 2;
const decl_doc_lines_max = 4;
const header_lines_max = 100;
const alias_letters_max = 2;
const comptime_label = "//! Comptime.";

const sleep_token = "sleep" ++ "(";
const testing_token = "std." ++ "testing";
const print_token = "std.debug." ++ "print(";

/// Files that import `std.testing` and sleep in production code on purpose.
const sleep_allowed = [_][]const u8{ "cli/client/client_reconnect.zig", "core/span.zig" };

/// Files whose job is to print on a test failure.
const print_allowed = [_][]const u8{ "core/utils/prop.zig", "core/test_fixtures.zig" };

/// The directories of `src/` that a host of the library can read.
const engine_dirs = [_][]const u8{
    "core/",
    "resolve/",
    "io/",
    "ops/",
    "ops_vulkan/",
    "runtime/",
    "serve/",
};

/// The engine files that write to a stream on purpose: the log, the progress bar of a download,
/// and the table of a GPU profile.
const stream_allowed = [_][]const u8{
    "core/utils/log.zig",
    "io/download.zig",
    "runtime/gpu_profile.zig",
};

const stream_tokens = [_][]const u8{ ".stdout" ++ "()", ".stderr" ++ "()", "lockStderr" ++ "(" };

/// The files that hold a command table. They name every command on purpose.
const table_files = [_][]const u8{ "cli/command.zig", "dev/dev_command.zig" };

/// The CLI files that every command reads. Each one sits below all commands.
const building_blocks = [_][]const u8{
    "cli/args.zig",
    "cli/context.zig",
    "cli/flag_groups.zig",
    "cli/prompt.zig",
    "cli/select.zig",
    "cli/style.zig",
};

/// The directory of `src/` that holds every OS condition. It is the lowest layer.
const os_dir = "os/";

/// The names that make an OS condition. Each one is split, so this file holds none of them.
const os_tokens = [_][]const u8{
    "builtin" ++ ".os.tag",
    "builtin" ++ ".target.os",
    "std.os" ++ ".linux",
    "std.os" ++ ".windows",
    "std" ++ ".posix",
};

/// The files outside `src/os/` that make raw socket calls on purpose: the serve tests and the
/// serve demo.
const os_allowed = [_][]const u8{
    "tests/serve_harness.zig",
    "tests/zmq_interop_test.zig",
    "tests/serve_integration_test.zig",
    "src/dev/serve_demo_cmd.zig",
};

/// The declaration qualifiers that sit between a doc comment and the keyword it documents.
const qualifiers = [_][]const u8{ "pub ", "export ", "inline ", "noinline ", "extern " };

// -- types ------------------------------------------------------------------

const Rule = enum {
    sleep,
    debug_print,
    stream,
    braces,
    width,
    section,
    script,
    layer,
    block,
    doc,
    header,
    alias,
    comptime_note,
    os_condition,
};

/// The sections in file order. Every label under `tests` counts as `tests`.
const Section = enum { constants, types, state, api, helpers, tests };

const Violation = struct { rule: Rule, line: usize };

const Found = std.ArrayList(Violation);

/// A Zig file that a tree walk reads. `path` is relative to `root`.
const File = struct { root: []const u8, path: []const u8, source: []const u8 };

/// What a doc comment documents, which sets the length limit of the comment.
const DocTarget = enum { function, declaration, field };

const LineKind = enum { header, doc, interior, other };

/// Each check of a tree reads one file at a time and returns the number of breaks it printed.
const ShapeCheck = struct {
    comment_rules: bool,

    fn visit(self: ShapeCheck, gpa: Allocator, file: File) !usize {
        var found: Found = .empty;
        defer found.deinit(gpa);

        try shapeViolations(gpa, &found, file.source, self.comment_rules);
        return report(file.root, file.path, &found);
    }
};

const LayerCheck = struct {
    fn visit(_: LayerCheck, gpa: Allocator, file: File) !usize {
        var found: Found = .empty;
        defer found.deinit(gpa);

        try layerViolations(gpa, &found, file.path, file.source);
        return report(file.root, file.path, &found);
    }
};

const OsCheck = struct {
    fn visit(_: OsCheck, gpa: Allocator, file: File) !usize {
        const in_os_dir = std.mem.eql(u8, file.root, "src") and
            std.mem.startsWith(u8, file.path, os_dir);
        if (in_os_dir) return 0;

        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const joined = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ file.root, file.path });
        if (containsString(&os_allowed, joined)) return 0;

        var found: Found = .empty;
        defer found.deinit(gpa);

        try osViolations(gpa, &found, file.source);
        return report(file.root, file.path, &found);
    }
};

const ClockCheck = struct {
    whole_file_is_test: bool,

    fn visit(self: ClockCheck, gpa: Allocator, file: File) !usize {
        if (std.mem.eql(u8, file.path, "tidy_test.zig")) return 0;

        var found: Found = .empty;
        defer found.deinit(gpa);

        const print_ok = containsString(&print_allowed, file.path);
        const sleep_ok = containsString(&sleep_allowed, file.path);
        try clockViolations(gpa, &found, file.source, self.whole_file_is_test, print_ok, sleep_ok);
        if (!self.whole_file_is_test and isEngineFile(file.path)) {
            try streamViolations(gpa, &found, file.source);
        }

        return report(file.root, file.path, &found);
    }
};

// -- helpers ----------------------------------------------------------------

/// Print every break of one file as `file:line: rule` in line order, and return their number.
fn report(root: []const u8, path: []const u8, found: *Found) usize {
    std.mem.sort(Violation, found.items, {}, lineLessThan);
    const separator = if (root.len > 0) "/" else "";
    for (found.items) |v| {
        const location = .{ root, separator, path, v.line, @tagName(v.rule) };
        std.debug.print("{s}{s}{s}:{d}: {s}\n", location);
    }

    return found.items.len;
}

fn lineLessThan(_: void, a: Violation, b: Violation) bool {
    return a.line < b.line;
}

/// Run `context.visit` on every Zig file under `root`, and return the total of breaks.
fn walkTree(gpa: Allocator, root: []const u8, context: anytype) !usize {
    const io = std.testing.io;
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(gpa);
    defer walker.deinit();

    var total: usize = 0;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".zig")) continue;

        const source = try dir.readFileAlloc(io, entry.path, gpa, .limited(max_file_bytes));
        defer gpa.free(source);

        total += try context.visit(gpa, .{ .root = root, .path = entry.path, .source = source });
    }

    return total;
}

fn clockViolations(
    gpa: Allocator,
    found: *Found,
    source: []const u8,
    whole_file_is_test: bool,
    print_ok: bool,
    sleep_ok: bool,
) !void {
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
            try found.append(gpa, .{ .rule = .sleep, .line = line_no });
        }
        if (!in_test and !print_ok and std.mem.indexOf(u8, line, print_token) != null) {
            try found.append(gpa, .{ .rule = .debug_print, .line = line_no });
        }
    }
}

fn isEngineFile(path: []const u8) bool {
    if (containsString(&stream_allowed, path)) return false;

    for (engine_dirs) |dir| {
        if (std.mem.startsWith(u8, path, dir)) return true;
    }

    return false;
}

/// Find each stream that a line of non-test code opens.
fn streamViolations(gpa: Allocator, found: *Found, source: []const u8) !void {
    var in_test = false;
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (std.mem.startsWith(u8, line, "test ")) {
            in_test = true;
        }

        defer if (std.mem.eql(u8, line, "}")) {
            in_test = false;
        };

        const code = std.mem.trimStart(u8, line, " ");
        if (in_test or std.mem.startsWith(u8, code, "//")) continue;

        for (stream_tokens) |token| {
            if (std.mem.indexOf(u8, line, token) != null) {
                try found.append(gpa, .{ .rule = .stream, .line = line_no });
            }
        }
    }
}

/// Find each OS condition in a line of non-test code.
fn osViolations(gpa: Allocator, found: *Found, source: []const u8) !void {
    var in_test = false;
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (std.mem.startsWith(u8, line, "test ")) {
            in_test = true;
        }

        defer if (std.mem.eql(u8, line, "}")) {
            in_test = false;
        };

        const code = std.mem.trimStart(u8, line, " ");
        if (in_test or std.mem.startsWith(u8, code, "//")) continue;

        for (os_tokens) |token| {
            if (std.mem.indexOf(u8, line, token) != null) {
                try found.append(gpa, .{ .rule = .os_condition, .line = line_no });
                break;
            }
        }
    }
}

/// True when `token` starts at a word boundary, so `awaitAllAsleep(` is no sleep call.
fn hasToken(line: []const u8, token: []const u8) bool {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, line, from, token)) |at| : (from = at + 1) {
        if (at == 0 or !std.ascii.isAlphabetic(line[at - 1])) return true;
    }

    return false;
}

/// Every break of rules 3 to 5 and 8 to 12 in `source`, a whole Zig file. The comment rules
/// (8 to 10) cover `src/` and `tools/` only.
fn shapeViolations(gpa: Allocator, found: *Found, source: []const u8, comment_rules: bool) !void {
    try widthViolations(gpa, found, source);
    try sectionViolations(gpa, found, source);
    try aliasViolations(gpa, found, source);
    if (comment_rules) {
        try commentViolations(gpa, found, source);
    }

    const terminated = try gpa.dupeZ(u8, source);
    defer gpa.free(terminated);

    var tree = try Ast.parse(gpa, terminated, .zig);
    defer tree.deinit(gpa);

    try bracesViolations(gpa, found, &tree);
    try comptimeViolations(gpa, found, terminated);
}

fn widthViolations(gpa: Allocator, found: *Found, source: []const u8) !void {
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (line.len <= line_columns_max) continue;
        if (widthExempt(std.mem.trimStart(u8, line, " "))) continue;

        try found.append(gpa, .{ .rule = .width, .line = line_no });
    }
}

/// A long line is legal when it holds a URL, or one string literal alone.
fn widthExempt(code: []const u8) bool {
    if (std.mem.indexOf(u8, code, "https://") != null) return true;
    return isLoneString(code);
}

/// True for one `\\` line, or for one `"..."` with an optional `++` and an optional comma.
fn isLoneString(code: []const u8) bool {
    if (std.mem.startsWith(u8, code, "\\\\")) return true;

    var body = std.mem.trimEnd(u8, code, ",");
    body = std.mem.trimEnd(u8, body, " ");
    body = std.mem.trimEnd(u8, body, "+");
    body = std.mem.trimEnd(u8, body, " ");
    if (body.len < 2 or body[0] != '"' or body[body.len - 1] != '"') return false;

    var at: usize = 1;
    while (at < body.len - 1) : (at += 1) {
        if (body[at] == '\\') {
            at += 1;
        } else if (body[at] == '"') {
            return false;
        }
    }

    return true;
}

fn sectionViolations(gpa: Allocator, found: *Found, source: []const u8) !void {
    var section: ?Section = null;
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (sectionOf(line)) |next| {
            if (section) |current| {
                const backwards = @intFromEnum(next) < @intFromEnum(current);
                if (backwards) {
                    try found.append(gpa, .{ .rule = .section, .line = line_no });
                }
            }

            section = next;
            continue;
        }

        const current = section orelse continue;
        const private_fn_in_api = current == .api and std.mem.startsWith(u8, line, "fn ");
        const public_fn_in_helpers = current == .helpers and
            std.mem.startsWith(u8, line, "pub fn ");
        const type_in_code = (current == .api or current == .helpers) and declaresType(line);
        if (private_fn_in_api or public_fn_in_helpers or type_in_code) {
            try found.append(gpa, .{ .rule = .section, .line = line_no });
        }
    }
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

/// Rule 11: `const z = @import("z");` names its module with one or two letters.
fn aliasViolations(gpa: Allocator, found: *Found, source: []const u8) !void {
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;

        var code = std.mem.trimStart(u8, line, " ");
        if (std.mem.startsWith(u8, code, "pub ")) {
            code = code["pub ".len..];
        }

        if (!std.mem.startsWith(u8, code, "const ")) continue;

        const rest = code["const ".len..];
        const equals = std.mem.indexOf(u8, rest, " = @import(") orelse continue;
        const name = rest[0..equals];
        if (name.len <= alias_letters_max and !std.mem.eql(u8, name, "vk")) {
            try found.append(gpa, .{ .rule = .alias, .line = line_no });
        }
    }
}

/// Rule 12: a `comptime` keyword token needs a `Comptime.` paragraph in the `//!` header. The
/// tokenizer skips comments and strings, so only the keyword counts.
fn comptimeViolations(gpa: Allocator, found: *Found, source: [:0]const u8) !void {
    var tokens = std.zig.Tokenizer.init(source);
    const holds_keyword = while (true) {
        const token = tokens.next();
        if (token.tag == .eof) break false;
        if (token.tag == .keyword_comptime) break true;
    };
    if (!holds_keyword) return;

    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "//!")) break;
        if (std.mem.startsWith(u8, line, comptime_label)) return;
    }

    try found.append(gpa, .{ .rule = .comptime_note, .line = 1 });
}

/// Rules 8 to 10 on the lines of `source`. Test blocks are exempt from rules 8 and 9.
fn commentViolations(gpa: Allocator, found: *Found, source: []const u8) !void {
    var in_test = false;
    var run_kind: LineKind = .other;
    var run_len: usize = 0;
    var run_line: usize = 0;
    var run_in_test = false;
    var header_len: usize = 0;
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (std.mem.startsWith(u8, line, "test ") and !std.mem.endsWith(u8, line, "}")) {
            in_test = true;
        }

        const kind = lineKind(line);
        if (kind == .header and header_len == line_no - 1) {
            header_len += 1;
        }

        if (run_len > 0 and kind != run_kind) {
            if (!run_in_test) {
                try runViolation(gpa, found, run_kind, run_len, run_line, line);
            }

            run_len = 0;
        }

        if (kind == .doc or kind == .interior) {
            if (run_len == 0) {
                run_kind = kind;
                run_line = line_no;
                run_in_test = in_test;
            }

            run_len += 1;
        }

        if (in_test and std.mem.eql(u8, line, "}")) {
            in_test = false;
        }
    }

    if (header_len > header_lines_max) {
        try found.append(gpa, .{ .rule = .header, .line = 1 });
    }
}

fn lineKind(line: []const u8) LineKind {
    const code = std.mem.trimStart(u8, line, " ");
    const indent = line.len - code.len;
    if (std.mem.startsWith(u8, code, "//!")) return .header;
    if (std.mem.startsWith(u8, code, "///")) return .doc;
    if (std.mem.startsWith(u8, code, "// --")) return .other;
    if (std.mem.startsWith(u8, code, "//") and indent >= 4) return .interior;
    return .other;
}

/// Check a finished run of `len` comment lines. `next` is the line that follows the run.
fn runViolation(
    gpa: Allocator,
    found: *Found,
    kind: LineKind,
    len: usize,
    line: usize,
    next: []const u8,
) !void {
    const limit: usize = switch (kind) {
        .interior => block_lines_max,
        .doc => switch (docTarget(next)) {
            .function => fn_doc_lines_max,
            .declaration => decl_doc_lines_max,
            .field => field_doc_lines_max,
        },
        else => unreachable,
    };
    if (len <= limit) return;

    const rule: Rule = if (kind == .interior) .block else .doc;
    try found.append(gpa, .{ .rule = rule, .line = line });
}

/// The kind of declaration that follows a doc comment. `noinline fn` is a function.
fn docTarget(next: []const u8) DocTarget {
    var code = std.mem.trimStart(u8, next, " ");
    for (0..qualifiers.len) |_| {
        const qualifier = for (qualifiers) |q| {
            if (std.mem.startsWith(u8, code, q)) break q;
        } else break;

        code = code[qualifier.len..];
    }

    if (std.mem.startsWith(u8, code, "fn ")) return .function;
    if (std.mem.startsWith(u8, code, "const ")) return .declaration;
    return .field;
}

fn bracesViolations(gpa: Allocator, found: *Found, tree: *const Ast) !void {
    for (0..tree.nodes.len) |i| {
        const node: Ast.Node.Index = @enumFromInt(i);
        var buffer: [2]Ast.Node.Index = undefined;
        const statements = tree.blockStatements(&buffer, node) orelse continue;
        for (statements) |statement| {
            if (statementViolation(tree, statement)) |v| {
                try found.append(gpa, v);
            }
        }
    }
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

/// Every import of `source` that points up the layers. `path` is relative to `src/`.
fn layerViolations(gpa: Allocator, found: *Found, path: []const u8, source: []const u8) !void {
    const opener = "@import(\"";
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (std.mem.startsWith(u8, std.mem.trimStart(u8, line, " "), "//")) continue;

        const at = std.mem.indexOf(u8, line, opener) orelse continue;
        const rest = line[at + opener.len ..];
        const end = std.mem.indexOfScalar(u8, rest, '"') orelse continue;
        if (importBreaksLayer(path, rest[0..end])) {
            try found.append(gpa, .{ .rule = .layer, .line = line_no });
        }
    }
}

fn importBreaksLayer(path: []const u8, target: []const u8) bool {
    if (std.mem.startsWith(u8, path, os_dir)) {
        const names_std = std.mem.eql(u8, target, "std") or std.mem.eql(u8, target, "builtin");
        const names_sibling = std.mem.endsWith(u8, target, ".zig") and
            std.mem.indexOfScalar(u8, target, '/') == null;
        return !names_std and !names_sibling;
    }

    const in_dev = std.mem.startsWith(u8, path, "dev/");
    const names_dev = std.mem.indexOf(u8, target, "dev/") != null or
        std.mem.indexOf(u8, target, "dev_") != null or
        std.mem.endsWith(u8, target, "-dev");
    if (!in_dev and names_dev) return true;

    const names_cli = std.mem.indexOf(u8, target, "cli/") != null;
    if (startsWithAny(path, &engine_dirs) and names_cli) return true;

    if (containsString(&table_files, path)) return false;

    const names_command = std.mem.endsWith(u8, target, "_cmd.zig") or
        std.mem.endsWith(u8, target, "command.zig");
    if (std.mem.endsWith(u8, path, "_cmd.zig") and names_command) {
        return !std.mem.endsWith(u8, path, std.fs.path.basename(target));
    }

    const below_commands = containsString(&building_blocks, path) or
        std.mem.startsWith(u8, path, "cli/boot/");
    if (below_commands) return names_command or std.mem.indexOf(u8, target, "client/") != null;

    return false;
}

fn startsWithAny(path: []const u8, prefixes: []const []const u8) bool {
    for (prefixes) |prefix| {
        if (std.mem.startsWith(u8, path, prefix)) return true;
    }

    return false;
}

fn containsString(list: []const []const u8, item: []const u8) bool {
    for (list) |entry| {
        if (std.mem.eql(u8, entry, item)) return true;
    }

    return false;
}

/// The script rules of `tools/README.md` that a program can check: the width of every line, the
/// strict mode of a bash script, and the `main` of a Python script.
fn scriptViolations(gpa: Allocator, found: *Found, path: []const u8, source: []const u8) !void {
    var line_no: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        line_no += 1;
        if (line.len > line_columns_max) {
            try found.append(gpa, .{ .rule = .width, .line = line_no });
        }
    }

    const bash = std.mem.startsWith(u8, source, "#!/usr/bin/env bash\n");
    if (bash and std.mem.indexOf(u8, source, "\nset -euo pipefail\n") == null) {
        try found.append(gpa, .{ .rule = .script, .line = 1 });
    }

    const python = std.mem.endsWith(u8, path, ".py");
    if (python and std.mem.indexOf(u8, source, "\ndef main(") == null) {
        try found.append(gpa, .{ .rule = .script, .line = 1 });
    }
}

fn checkScript(gpa: Allocator, dir: std.Io.Dir, root: []const u8, path: []const u8) !usize {
    const io = std.testing.io;
    const source = try dir.readFileAlloc(io, path, gpa, .limited(max_file_bytes));
    defer gpa.free(source);

    var found: Found = .empty;
    defer found.deinit(gpa);

    try scriptViolations(gpa, &found, path, source);
    return report(root, path, &found);
}

/// The rule of the first break in `found`, or null when it holds none.
fn firstRule(found: Found) ?Rule {
    return if (found.items.len > 0) found.items[0].rule else null;
}

// -- tests ------------------------------------------------------------------

test "the stream rule finds an engine file that opens a stream" {
    const gpa = std.testing.allocator;
    const Case = struct { src: []const u8, want: usize };
    const cases = [_]Case{
        .{ .src = "fn f() void {\n    _ = Io.File" ++ stream_tokens[0] ++ ";\n}\n", .want = 1 },
        .{ .src = "fn f() void {\n    _ = Io.File" ++ stream_tokens[1] ++ ";\n}\n", .want = 1 },
        .{ .src = "test \"t\" {\n    _ = Io.File" ++ stream_tokens[1] ++ ";\n}\n", .want = 0 },
        .{ .src = "// Io.File" ++ stream_tokens[0] ++ "\n", .want = 0 },
    };
    for (cases) |case| {
        var found: Found = .empty;
        defer found.deinit(gpa);

        try streamViolations(gpa, &found, case.src);
        try std.testing.expectEqual(case.want, found.items.len);
    }

    try std.testing.expect(isEngineFile("serve/server.zig"));
    try std.testing.expect(!isEngineFile("core/utils/log.zig"));
    try std.testing.expect(!isEngineFile("cli/run_cmd.zig"));
}

test "the scanner finds each rule and ignores comments, words, and test blocks" {
    const sleep_call = "io." ++ sleep_token ++ ")\n";
    const print_call = print_token ++ "\"x\", .{});\n";
    const imports = "const t = " ++ testing_token ++ ";\n";
    const Case = struct {
        src: []const u8,
        want: ?Rule,
        whole_test: bool = false,
        print_ok: bool = false,
        sleep_ok: bool = false,
    };
    const cases = [_]Case{
        .{ .src = "test \"t\" {\n    " ++ sleep_call ++ "}\n", .want = .sleep },
        .{ .src = "fn f() void {\n    " ++ sleep_call ++ "}\n", .want = null },
        .{ .src = "fn f() void { " ++ sleep_call ++ " }\n", .want = .sleep, .whole_test = true },
        .{ .src = "fn awaitAllAsleep(p: *P) void {}\n", .want = null, .whole_test = true },
        .{ .src = "// " ++ sleep_call, .want = null, .whole_test = true },
        .{ .src = "fn f() void {\n    " ++ print_call ++ "}\n", .want = .debug_print },
        .{ .src = "fn f() void {\n    " ++ print_call ++ "}\n", .want = null, .print_ok = true },
        .{ .src = "test \"t\" {\n    " ++ print_call ++ "}\n", .want = null },
        .{ .src = "/// Like " ++ print_token ++ ")\nfn f() void {}\n", .want = null },
        .{ .src = imports ++ "fn f() void {\n    " ++ sleep_call ++ "}\n", .want = .sleep },
        .{
            .src = imports ++ "fn f() void {\n    " ++ sleep_call ++ "}\n",
            .want = null,
            .sleep_ok = true,
        },
        .{ .src = imports ++ "fn awaitAllAsleep(p: *P) void {}\n", .want = null },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("tidy scanner case {d}\n", .{i});

        var found: Found = .empty;
        defer found.deinit(std.testing.allocator);

        try clockViolations(
            std.testing.allocator,
            &found,
            c.src,
            c.whole_test,
            c.print_ok,
            c.sleep_ok,
        );
        try std.testing.expectEqual(c.want, firstRule(found));
    }
}

test "the shape rules find a bare loop body, a long line, and a section out of place" {
    const long = "x" ** (line_columns_max + 1);
    const Case = struct { src: []const u8, want: ?Rule };
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
        .{ .src = "test \"" ++ long ++ "\" {}\n", .want = .width },
        .{ .src = "const entries = .{\n    .{ " ++ long ++ " },\n};\n", .want = .width },
        .{ .src = "const j = f(\"{\\\"" ++ long ++ "\\\"}\");\n", .want = .width },
        .{ .src = "// https://" ++ long ++ "\n", .want = null },
        .{ .src = "const s = .{\n    \"" ++ long ++ "\",\n};\n", .want = null },
        .{ .src = "const s = \n    \"" ++ long ++ "\" ++\n    \"b\";\n", .want = null },
        .{ .src = "const s = .{\n    \"a\\\"" ++ long ++ "\\\"\"\n};\n", .want = null },
        .{ .src = "const s =\n    \\\\" ++ long ++ "\n;\n", .want = null },
        .{ .src = "const s = .{\n    \"" ++ long ++ "\", \"b\",\n};\n", .want = .width },
        .{ .src = "// -- api --\nfn f() void {}\n", .want = .section },
        .{ .src = "// -- helpers --\npub fn f() void {}\n", .want = .section },
        .{ .src = "// -- helpers: parse --\nconst T = struct {};\n", .want = .section },
        .{ .src = "// -- api --\n// -- types --\n", .want = .section },
        .{
            .src = "// -- types --\npub fn T() type {}\n// -- api --\npub fn f() void {}\n",
            .want = null,
        },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("tidy shape case {d}\n", .{i});

        var found: Found = .empty;
        defer found.deinit(std.testing.allocator);

        try shapeViolations(std.testing.allocator, &found, c.src, false);
        try std.testing.expectEqual(c.want, firstRule(found));
    }
}

test "the comment rules find a long block, a long doc, a long header, and a short alias" {
    const line = "    // x\n";
    const fn_open = "fn f() void {\n";
    const Case = struct { src: []const u8, want: ?Rule, comments: bool = true };
    const cases = [_]Case{
        .{ .src = fn_open ++ line ** 7 ++ "}\n", .want = .block },
        .{ .src = fn_open ++ line ** 6 ++ "}\n", .want = null },
        .{ .src = "test \"t\" {\n" ++ line ** 7 ++ "}\n", .want = null },
        .{ .src = fn_open ++ line ** 7 ++ "}\n", .want = null, .comments = false },
        .{ .src = "/// x\n" ** 4 ++ "fn f() void {}\n", .want = .doc },
        .{ .src = "/// x\n" ** 3 ++ "pub fn f() void {}\n", .want = null },
        .{ .src = "/// x\n" ** 3 ++ "noinline fn f() void {}\n", .want = null },
        .{ .src = "/// x\n" ** 3 ++ "a: u32,\n", .want = .doc },
        .{ .src = "/// x\n" ** 2 ++ "a: u32,\n", .want = null },
        .{ .src = "/// x\n" ** 5 ++ "const a = 1;\n", .want = .doc },
        .{ .src = "/// x\n" ** 4 ++ "pub const a = 1;\n", .want = null },
        .{ .src = "test \"t\" {\n" ++ "/// x\n" ** 4 ++ "fn f() void {}\n}\n", .want = null },
        .{ .src = "//! x\n" ** 101 ++ "\nconst a = 1;\n", .want = .header },
        .{ .src = "//! x\n" ** 100 ++ "\nconst a = 1;\n", .want = null },
        .{ .src = "const z = @import(\"z\");\n", .want = .alias },
        .{ .src = "pub const bu = @import(\"bench_util.zig\");\n", .want = .alias },
        .{ .src = "const vk = @import(\"vulkan\");\n", .want = null },
        .{ .src = "const ops = @import(\"ops\");\n", .want = null },
        .{ .src = "const z = @import(\"z\");\n", .want = .alias, .comments = false },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("tidy comment case {d}\n", .{i});

        var found: Found = .empty;
        defer found.deinit(std.testing.allocator);

        try shapeViolations(std.testing.allocator, &found, c.src, c.comments);
        try std.testing.expectEqual(c.want, firstRule(found));
    }
}

test "the comptime rule needs a Comptime paragraph in the header" {
    const header = "//! A file.\n//!\n";
    const note = "//! Comptime. Asserts pin the sizes.\n";
    const use = "\nfn f(comptime n: usize) void {\n    _ = n;\n}\n";
    const Case = struct { src: []const u8, want: ?Rule };
    const cases = [_]Case{
        .{ .src = header ++ use, .want = .comptime_note },
        .{ .src = header ++ note ++ use, .want = null },
        .{ .src = header ++ "\ncomptime {}\n", .want = .comptime_note },
        .{ .src = header ++ "\n// A comptime word.\nconst s = \"comptime\";\n", .want = null },
        .{ .src = "const a = 1;\n//! Comptime. Late.\n" ++ use, .want = .comptime_note },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("tidy comptime case {d}\n", .{i});

        var found: Found = .empty;
        defer found.deinit(std.testing.allocator);

        try shapeViolations(std.testing.allocator, &found, c.src, false);
        try std.testing.expectEqual(c.want, firstRule(found));
    }
}

test "every source file keeps the shape rules" {
    const gpa = std.testing.allocator;
    var breaks: usize = 0;
    breaks += try walkTree(gpa, "src", ShapeCheck{ .comment_rules = true });
    breaks += try walkTree(gpa, "tests", ShapeCheck{ .comment_rules = false });
    breaks += try walkTree(gpa, "tools", ShapeCheck{ .comment_rules = true });

    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    const build_zig = try cwd.readFileAlloc(io, "build.zig", gpa, .limited(max_file_bytes));
    defer gpa.free(build_zig);

    var found: Found = .empty;
    defer found.deinit(gpa);

    try shapeViolations(gpa, &found, build_zig, false);
    breaks += report("", "build.zig", &found);
    try std.testing.expectEqual(@as(usize, 0), breaks);
}

test "the layer rule finds an import that points up" {
    const bad = [_]struct { path: []const u8, target: []const u8 }{
        .{ .path = "cli/run_cmd.zig", .target = "../dev/tokens_cmd.zig" },
        .{ .path = "cli/run_cmd.zig", .target = "zigferencer-dev" },
        .{ .path = "serve/server.zig", .target = "../cli/args.zig" },
        .{ .path = "cli/args.zig", .target = "command.zig" },
        .{ .path = "cli/select.zig", .target = "run_cmd.zig" },
        .{ .path = "cli/boot/boot_lines.zig", .target = "../serve_cmd.zig" },
        .{ .path = "cli/boot/serve_boot.zig", .target = "../client/client_cmd.zig" },
        .{ .path = "cli/run_cmd.zig", .target = "serve_cmd.zig" },
        .{ .path = "os/process.zig", .target = "../core/limits.zig" },
        .{ .path = "os/console.zig", .target = "zigferencer" },
        .{ .path = "os/threads.zig", .target = "build_options" },
    };
    for (bad) |case| {
        try std.testing.expect(importBreaksLayer(case.path, case.target));
    }

    const good = [_]struct { path: []const u8, target: []const u8 }{
        .{ .path = "cli/command.zig", .target = "run_cmd.zig" },
        .{ .path = "dev/dev_command.zig", .target = "tokens_cmd.zig" },
        .{ .path = "dev/tokens_cmd.zig", .target = "zigferencer" },
        .{ .path = "cli/run_cmd.zig", .target = "boot/boot_resources.zig" },
        .{ .path = "cli/boot/boot_resources.zig", .target = "../args.zig" },
        .{ .path = "serve/server.zig", .target = "client.zig" },
        .{ .path = "os/socket_wait.zig", .target = "socket_wait_linux.zig" },
        .{ .path = "os/console.zig", .target = "std" },
        .{ .path = "os/console.zig", .target = "builtin" },
        .{ .path = "core/span.zig", .target = "../os/threads.zig" },
    };
    for (good) |case| {
        try std.testing.expect(!importBreaksLayer(case.path, case.target));
    }
}

test "every import points down the layers" {
    try std.testing.expectEqual(
        @as(usize, 0),
        try walkTree(std.testing.allocator, "src", LayerCheck{}),
    );
}

test "the OS rule finds an OS condition outside test blocks" {
    const open = "fn f() void {\n    _ = ";
    const close = ";\n}\n";
    const Case = struct { src: []const u8, want: ?Rule };
    const cases = [_]Case{
        .{ .src = open ++ os_tokens[0] ++ close, .want = .os_condition },
        .{ .src = open ++ os_tokens[1] ++ close, .want = .os_condition },
        .{ .src = open ++ os_tokens[2] ++ close, .want = .os_condition },
        .{ .src = open ++ os_tokens[3] ++ close, .want = .os_condition },
        .{ .src = open ++ os_tokens[4] ++ close, .want = .os_condition },
        .{ .src = "test \"t\" {\n    _ = " ++ os_tokens[0] ++ close, .want = null },
        .{ .src = "// " ++ os_tokens[2] ++ "\n", .want = null },
        .{
            .src = "const os = @tagName(" ++ os_tokens[0] ++ ");\n",
            .want = .os_condition,
        },
        .{ .src = open ++ "std.os.argv" ++ close, .want = null },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("tidy OS case {d}\n", .{i});

        var found: Found = .empty;
        defer found.deinit(std.testing.allocator);

        try osViolations(std.testing.allocator, &found, c.src);
        try std.testing.expectEqual(c.want, firstRule(found));
    }
}

test "no file outside src/os/ holds an OS condition" {
    const gpa = std.testing.allocator;
    var breaks: usize = 0;
    breaks += try walkTree(gpa, "src", OsCheck{});
    breaks += try walkTree(gpa, "tests", OsCheck{});
    breaks += try walkTree(gpa, "tools", OsCheck{});
    try std.testing.expectEqual(@as(usize, 0), breaks);
}

test "no test sleeps and no non-test code prints" {
    const gpa = std.testing.allocator;
    var breaks: usize = 0;
    breaks += try walkTree(gpa, "tests", ClockCheck{ .whole_file_is_test = true });
    breaks += try walkTree(gpa, "src", ClockCheck{ .whole_file_is_test = false });
    try std.testing.expectEqual(@as(usize, 0), breaks);
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

        var found: Found = .empty;
        defer found.deinit(std.testing.allocator);

        try scriptViolations(std.testing.allocator, &found, c.path, c.src);
        try std.testing.expectEqual(c.want, firstRule(found));
    }
}

test "every script and the Makefile keep the script rules" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();

    var breaks = try checkScript(gpa, cwd, "", "Makefile");

    var dir = try cwd.openDir(io, "tools", .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(gpa);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.path, ".zig")) continue;
        if (std.mem.endsWith(u8, entry.path, ".md")) continue;
        // The `py_compile` check of the script rules leaves a bytecode cache beside the script.
        if (std.mem.endsWith(u8, entry.path, ".pyc")) continue;

        breaks += try checkScript(gpa, dir, "tools", entry.path);
    }

    try std.testing.expectEqual(@as(usize, 0), breaks);
}
