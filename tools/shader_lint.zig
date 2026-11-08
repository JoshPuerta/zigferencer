//! Style and truth rules for the GLSL kernels in `src/ops_vulkan/shaders`.
//!
//! Comptime. `add` and `addAt` take the format string at comptime, so the compiler checks each
//! finding message.
//!
//! `glslc` and `spirv-val` own the semantics. This tool owns what they cannot see: the shape of
//! the header, the comments, the blank lines, and the claims a header states about the code. It
//! reads text line by line and runs no compiler, so it needs no Vulkan SDK.
//!
//! Findings go to stderr, one per line, in the shape of a `glslc` diagnostic, so one problem
//! matcher reads both:
//!
//!     <path>:<line>:<column>: error: <rule>: <message>
//!
//! A count follows the last finding. A passing run prints nothing.
//!
//! Usage: `shader_lint <shader dir> <kernels.zig>`. The exit code is 1 when a finding exists.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

// -- constants --------------------------------------------------------------

const line_columns_max = 100;
const banner_columns = 78;
const comment_block_lines_max = 2;
const header_lines_max = 100;
const file_bytes_max = 1 << 20;
const output_buffer_bytes = 1 << 16;

/// Calls whose evaluation order the GLSL specification leaves open. A chain that must keep its
/// rounding writes the sum out.
const banned_calls = [_][]const u8{ "dot", "fma", "length", "distance", "normalize", "mix" };

/// Builtins that differ between the lanes of one workgroup.
const lane_builtins = [_][]const u8{
    "gl_LocalInvocationID",
    "gl_LocalInvocationIndex",
    "gl_GlobalInvocationID",
    "gl_SubgroupInvocationID",
};

const body_banners = [_][]const u8{ "configuration", "bindings", "helpers", "main" };

/// The rules that read one raw line alone. Each check returns the column of the defect.
const text_checks = [_]TextCheck{
    .{ .rule = .width, .check = overWidth, .message = "line is wider than 100 columns" },
    .{ .rule = .tab, .check = findTab, .message = "line holds a tab" },
    .{ .rule = .trailing_space, .check = trailingSpace, .message = "line ends in a space" },
    .{ .rule = .ascii, .check = nonAscii, .message = "line holds a byte past ASCII" },
    .{ .rule = .block_comment, .check = blockComment, .message = "use // comments only" },
    .{ .rule = .comment_backslash, .check = commentBackslash, .message = "a final backslash " ++
        "joins the next line" },
};

// -- types ------------------------------------------------------------------

const Rule = enum {
    width,
    tab,
    trailing_space,
    ascii,
    final_newline,
    block_comment,
    comment_backslash,
    title,
    section_order,
    section_missing,
    section_length,
    header_length,
    version_line,
    banner_order,
    comment_block,
    step_label,
    blank_run,
    blank_after_guard,
    blank_after_block,
    blank_after_barrier,
    braces,
    banned_call,
    bit_exact_precise,
    tolerance,
    uniform_guard,
    binding_row,
};

const TextCheck = struct {
    rule: Rule,
    check: *const fn ([]const u8) ?usize,
    message: []const u8,
};

/// The sections of a header, in the required order.
const Section = enum { What, Math, Schedule, Memory, Hardware, Numerics, Bindings, Defines };

const section_count = @typeInfo(Section).@"enum".fields.len;

/// The most lines a section holds, label included, in the order of `Section`.
const section_lines_max = [section_count]usize{ 6, 22, 28, 12, 16, 8, 18, 9 };

const Span = struct { first: usize, end: usize };

const Kind = enum { blank, comment, directive, code };

const Line = struct {
    /// The bytes of the line, without the newline.
    raw: []const u8,
    /// The line without a trailing comment and without edge spaces.
    code: []const u8,
    kind: Kind,
};

const Finding = struct { rule: Rule, line: usize, column: usize, message: []const u8 };

/// One shader and the host file that its header makes a claim about.
const Input = struct {
    stem: []const u8,
    source: []const u8,
    kernels_zig: []const u8,
};

const Report = struct {
    arena: Allocator,
    findings: std.ArrayList(Finding) = .empty,

    fn add(self: *Report, rule: Rule, line: usize, comptime fmt: []const u8, args: anytype) !void {
        try self.addAt(rule, line, 1, fmt, args);
    }

    fn addAt(
        self: *Report,
        rule: Rule,
        line: usize,
        column: usize,
        comptime fmt: []const u8,
        args: anytype,
    ) !void {
        const message = try std.fmt.allocPrint(self.arena, fmt, args);
        try self.findings.append(self.arena, .{
            .rule = rule,
            .line = line,
            .column = column,
            .message = message,
        });
    }

    fn lessThan(_: void, a: Finding, b: Finding) bool {
        if (a.line != b.line) return a.line < b.line;

        return @intFromEnum(a.rule) < @intFromEnum(b.rule);
    }
};

/// The values of one shader that more than one rule reads.
const Shader = struct {
    input: Input,
    lines: []const Line,
    /// Index of the `#version` line. The header is every line before it.
    version: usize,
    sections: [section_count]?Span,
    main_first: usize,
    main_end: usize,

    fn span(self: *const Shader, section: Section) ?Span {
        return self.sections[@intFromEnum(section)];
    }

    fn has(self: *const Shader, needle: []const u8) bool {
        for (self.lines[self.version..]) |line| {
            if (line.kind == .code and containsWord(line.code, needle)) return true;
        }

        return false;
    }
};

// -- api --------------------------------------------------------------------

/// Run every rule on one shader. The findings come back sorted by line.
pub fn lint(arena: Allocator, input: Input) ![]const Finding {
    var report: Report = .{ .arena = arena };
    const lines = try classify(arena, input.source);
    var shader = indexShader(input, lines);
    try parseHeader(&shader, &report);

    try textRules(&shader, &report);
    try headerRules(arena, &shader, &report);
    try bannerRules(&shader, &report);
    try commentRules(&shader, &report);
    try blankRules(&shader, &report);
    try braceRules(&shader, &report);
    try numericRules(arena, &shader, &report);
    try uniformRules(arena, &shader, &report);
    try bindingRows(&shader, &report);

    std.mem.sort(Finding, report.findings.items, {}, Report.lessThan);

    return report.findings.items;
}

pub fn main(init: std.process.Init) !void {
    // Read the arguments and the kernel source.
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 3) return error.Usage;

    const cwd = std.Io.Dir.cwd();
    const kernels_zig = try cwd.readFileAlloc(io, args[2], arena, .limited(file_bytes_max * 8));

    var buffer: [output_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &buffer);
    const w = &stderr_writer.interface;

    // List the shader files in order.
    var dir = try cwd.openDir(io, args[1], .{ .iterate = true });
    defer dir.close(io);

    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);

    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".comp")) continue;

        try names.append(gpa, try arena.dupe(u8, entry.name));
    }

    std.mem.sort([]const u8, names.items, {}, nameLessThan);

    // Lint each shader and print its findings.
    var total: usize = 0;
    for (names.items) |name| {
        const source = try dir.readFileAlloc(io, name, arena, .limited(file_bytes_max));
        const stem = name[0 .. name.len - ".comp".len];
        const input: Input = .{ .stem = stem, .source = source, .kernels_zig = kernels_zig };
        const findings = try lint(arena, input);
        for (findings) |f| {
            const fmt = "{s}/{s}:{d}:{d}: error: {s}: {s}\n";
            try w.print(fmt, .{ args[1], name, f.line, f.column, @tagName(f.rule), f.message });
        }

        total += findings.len;
    }

    // Report the total, and exit nonzero on findings.
    if (total != 0) {
        try w.print("shader-lint: {d} findings in {d} files\n", .{ total, names.items.len });
    }

    try w.flush();
    if (total != 0) {
        std.process.exit(1);
    }
}

// -- helpers: scan ----------------------------------------------------------

fn nameLessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn classify(arena: Allocator, source: []const u8) ![]const Line {
    var lines: std.ArrayList(Line) = .empty;
    var continued = false;
    var it = std.mem.splitScalar(u8, source, '\n');
    while (it.next()) |raw| {
        const trimmed = std.mem.trim(u8, raw, " \t");
        const comment_at = std.mem.indexOf(u8, trimmed, "//");
        const code = std.mem.trimEnd(u8, trimmed[0 .. comment_at orelse trimmed.len], " \t");

        var kind: Kind = .code;
        if (continued) {
            kind = .directive;
            continued = std.mem.endsWith(u8, trimmed, "\\");
        } else if (trimmed.len == 0) {
            kind = .blank;
        } else if (comment_at != null and code.len == 0) {
            kind = .comment;
        } else if (trimmed[0] == '#') {
            kind = .directive;
            continued = std.mem.endsWith(u8, trimmed, "\\");
        }

        try lines.append(arena, .{ .raw = raw, .code = code, .kind = kind });
    }

    return lines.toOwnedSlice(arena);
}

fn indexShader(input: Input, lines: []const Line) Shader {
    var version: usize = 0;
    var main_first: usize = lines.len;
    var main_end: usize = lines.len;
    for (lines, 0..) |line, i| {
        if (version == 0 and std.mem.startsWith(u8, line.code, "#version")) {
            version = i;
        }
        if (std.mem.startsWith(u8, line.code, "void main()")) {
            main_first = i;
        }
        if (i > main_first and main_end == lines.len and std.mem.eql(u8, line.raw, "}")) {
            main_end = i;
        }
    }

    return .{
        .input = input,
        .lines = lines,
        .version = version,
        .sections = @splat(null),
        .main_first = main_first,
        .main_end = main_end,
    };
}

/// Find the section labels of the header. A label line is `// ` and a section name.
fn parseHeader(shader: *Shader, report: *Report) !void {
    var open: ?Section = null;
    for (shader.lines[0..shader.version], 0..) |line, i| {
        const text = if (std.mem.startsWith(u8, line.raw, "// ")) line.raw[3..] else "";
        const section = std.meta.stringToEnum(Section, text) orelse continue;
        if (open) |previous| {
            shader.sections[@intFromEnum(previous)].?.end = i;
            if (@intFromEnum(section) <= @intFromEnum(previous)) {
                const fmt = "{s} sits after {s}";
                const names = .{ @tagName(section), @tagName(previous) };
                try report.add(.section_order, i + 1, fmt, names);
            }
        }

        shader.sections[@intFromEnum(section)] = .{ .first = i, .end = i };
        open = section;
    }

    // The blank line before `#version` ends the header. It belongs to no section.
    if (open) |last| {
        const blank = shader.lines[shader.version - 1].kind == .blank;
        shader.sections[@intFromEnum(last)].?.end = shader.version - @intFromBool(blank);
    }
}

fn isWordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn containsWord(haystack: []const u8, word: []const u8) bool {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, from, word)) |at| : (from = at + 1) {
        const left = at == 0 or !isWordByte(haystack[at - 1]);
        const end = at + word.len;
        const right = end == haystack.len or !isWordByte(haystack[end]);
        if (left and right) return true;
    }

    return false;
}

fn mentions(code: []const u8, names: []const []const u8) bool {
    for (names) |name| {
        if (containsWord(code, name)) return true;
    }

    return false;
}

fn isGuardBody(body: []const u8) bool {
    const words = [_][]const u8{ "return", "continue", "break" };
    for (words) |word| {
        if (std.mem.startsWith(u8, body, word)) return true;
    }

    return false;
}

fn isGuard(code: []const u8) bool {
    if (!std.mem.startsWith(u8, code, "if (")) return false;

    const close = closingParen(code, 3) orelse return false;

    return isGuardBody(std.mem.trim(u8, code[close + 1 ..], " "));
}

/// Index of the `)` that closes the `(` at or after `from`, or null when the line ends first.
fn closingParen(text: []const u8, from: usize) ?usize {
    var depth: usize = 0;
    for (text[from..], from..) |c, i| {
        if (c == '(') {
            depth += 1;
        }
        if (c == ')') {
            depth -= 1;
            if (depth == 0) return i;
        }
    }

    return null;
}

// -- helpers: text ----------------------------------------------------------

fn overWidth(raw: []const u8) ?usize {
    return if (raw.len > line_columns_max) line_columns_max + 1 else null;
}

fn findTab(raw: []const u8) ?usize {
    const at = std.mem.indexOfScalar(u8, raw, '\t') orelse return null;

    return at + 1;
}

fn trailingSpace(raw: []const u8) ?usize {
    return if (raw.len > 0 and raw[raw.len - 1] == ' ') raw.len else null;
}

fn nonAscii(raw: []const u8) ?usize {
    for (raw, 1..) |c, column| {
        if (c >= 0x80) return column;
    }

    return null;
}

fn blockComment(raw: []const u8) ?usize {
    const at = std.mem.indexOf(u8, raw, "/*") orelse return null;

    return at + 1;
}

fn commentBackslash(raw: []const u8) ?usize {
    const comment = std.mem.indexOf(u8, raw, "//") != null;

    return if (comment and std.mem.endsWith(u8, raw, "\\")) raw.len else null;
}

fn textRules(shader: *const Shader, report: *Report) !void {
    for (shader.lines, 1..) |line, n| {
        for (text_checks) |c| {
            if (c.check(line.raw)) |column| {
                try report.addAt(c.rule, n, column, "{s}", .{c.message});
            }
        }
    }

    const source = shader.input.source;
    if (!std.mem.endsWith(u8, source, "\n") or std.mem.endsWith(u8, source, "\n\n")) {
        try report.add(.final_newline, shader.lines.len, "file must end in one newline", .{});
    }
}

// -- helpers: header --------------------------------------------------------

fn headerRules(arena: Allocator, shader: *const Shader, report: *Report) !void {
    const lines = shader.lines;
    if (shader.version < 2) {
        try report.add(.version_line, 1, "a header comment comes before #version", .{});
        return;
    }

    const header_lines = shader.version - 1;
    if (header_lines > header_lines_max) {
        const fmt = "header holds {d} lines, the limit is {d}";
        try report.add(.header_length, 1, fmt, .{ header_lines, header_lines_max });
    }
    if (lines[shader.version - 1].kind != .blank or lines[shader.version - 2].kind != .comment) {
        try report.add(.version_line, shader.version + 1, "one blank line ends the header", .{});
    }

    const prefix = try std.fmt.allocPrint(arena, "// {s}: ", .{shader.input.stem});
    if (!std.mem.startsWith(u8, lines[0].raw, prefix)) {
        try report.add(.title, 1, "line 1 must start with `{s}`", .{prefix});
    }

    for (shader.sections) |maybe| {
        if (maybe != null) break;
    } else {
        try report.add(.section_order, 1, "the header holds no section label", .{});
        return;
    }

    try requiredSections(shader, report);
    try sectionLengths(shader, report);
}

fn requiredSections(shader: *const Shader, report: *Report) !void {
    const always = [_]Section{ .What, .Hardware, .Numerics, .Bindings };
    for (always) |section| {
        if (shader.span(section) == null) {
            try report.add(.section_missing, 1, "the header needs {s}", .{@tagName(section)});
        }
    }

    const cooperates = shader.has("barrier");
    if (cooperates and shader.span(.Math) == null) {
        try report.add(.section_missing, 1, "a kernel with a barrier needs Math", .{});
    }

    const has_grid = cooperates or shader.has("gl_WorkGroupID");
    if (has_grid and shader.span(.Schedule) == null) {
        try report.add(.section_missing, 1, "a kernel with a grid needs Schedule", .{});
    }

    for (shader.lines[shader.version..]) |line| {
        if (!std.mem.startsWith(u8, line.code, "shared ")) continue;

        if (shader.span(.Memory) == null) {
            try report.add(.section_missing, 1, "a kernel with shared memory needs Memory", .{});
        }

        break;
    }
}

fn sectionLengths(shader: *const Shader, report: *Report) !void {
    for (std.enums.values(Section)) |section| {
        const s = shader.span(section) orelse continue;
        var used = s.end - s.first;
        if (shader.lines[s.end - 1].raw.len == 2) {
            used -= 1;
        }
        if (used > section_lines_max[@intFromEnum(section)]) {
            const fmt = "{s} holds {d} lines, the limit is {d}";
            try report.add(.section_length, s.first + 1, fmt, .{
                @tagName(section),
                used,
                section_lines_max[@intFromEnum(section)],
            });
        }
    }
}

// -- helpers: body layout and comments --------------------------------------

/// The banners come in the order of `body_banners`, each one 78 columns wide, and `local_size`
/// comes before the bindings.
fn bannerRules(shader: *const Shader, report: *Report) !void {
    var rank: usize = 0;
    var seen_bindings = false;
    var seen_main = false;
    for (shader.lines, 1..) |line, n| {
        if (n <= shader.version) continue;

        const is_size = std.mem.startsWith(u8, line.code, "layout(local_size");
        if (is_size and seen_bindings) {
            try report.add(.banner_order, n, "local_size comes before the bindings banner", .{});
        }

        if (!std.mem.startsWith(u8, line.raw, "// -- ")) continue;

        const label_end = std.mem.indexOfScalarPos(u8, line.raw, 6, ' ') orelse line.raw.len;
        const label = line.raw[6..label_end];
        const dashes = std.mem.trimEnd(u8, line.raw, "-");
        if (line.raw.len != banner_columns or dashes.len == line.raw.len) {
            try report.add(.banner_order, n, "a banner ends in dashes at column 78", .{});
        }

        const at = for (body_banners, 0..) |banner, i| {
            if (std.mem.eql(u8, banner, label)) break i;
        } else null;

        if (at == null or at.? < rank) {
            try report.add(.banner_order, n, "banner `{s}` is out of order or unknown", .{label});
            continue;
        }

        rank = at.?;
        seen_bindings = seen_bindings or rank >= 1;
        seen_main = seen_main or rank == 3;
    }

    if (!seen_main) {
        try report.add(.banner_order, shader.version + 1, "the file needs a main banner", .{});
    }
}

fn isBanner(line: Line) bool {
    return std.mem.startsWith(u8, std.mem.trimStart(u8, line.raw, " "), "// -- ");
}

/// True for `1. Load`, `1-2. Load`, and the like, which `Pass N:` replaces.
fn isNumberedStep(text: []const u8) bool {
    const end = std.mem.indexOfNone(u8, text, "0123456789-") orelse return false;

    return end > 0 and text[end] == '.' and end + 1 < text.len and text[end + 1] == ' ';
}

fn passNumber(text: []const u8) ?usize {
    const prefix = "Pass ";
    if (!std.mem.startsWith(u8, text, prefix)) return null;

    const colon = std.mem.indexOfScalar(u8, text, ':') orelse return null;

    return std.fmt.parseInt(usize, text[prefix.len..colon], 10) catch null;
}

fn commentRules(shader: *const Shader, report: *Report) !void {
    const lines = shader.lines;
    var next_pass: usize = 1;
    var i = shader.version;
    while (i < lines.len) : (i += 1) {
        if (lines[i].kind != .comment or isBanner(lines[i])) continue;

        const first = i;
        while (i + 1 < lines.len and lines[i + 1].kind == .comment and !isBanner(lines[i + 1])) {
            i += 1;
        }

        const count = i - first + 1;
        if (count > comment_block_lines_max) {
            const fmt = "comment block of {d} lines, the limit is {d}";
            try report.add(.comment_block, first + 1, fmt, .{ count, comment_block_lines_max });
        }

        const head = std.mem.trimStart(u8, lines[first].raw, " /");
        if (isNumberedStep(head)) {
            try report.add(.step_label, first + 1, "a step label reads `Pass N:`", .{});
        }

        const number = passNumber(head) orelse continue;
        if (number != next_pass) {
            const fmt = "pass {d} comes where pass {d} belongs";
            try report.add(.step_label, first + 1, fmt, .{ number, next_pass });
        }

        next_pass = number + 1;
    }
}

// -- helpers: paragraphs and braces -----------------------------------------

fn continuesBlock(code: []const u8) bool {
    if (std.mem.eql(u8, code, "barrier();")) return true;

    return std.mem.startsWith(u8, code, "else") or std.mem.startsWith(u8, code, "#e");
}

/// Blank lines end the paragraphs of `main`: a guard group, a block, and a barrier each end one.
fn blankRules(shader: *const Shader, report: *Report) !void {
    const lines = shader.lines;
    for (lines, 0..) |line, i| {
        const run = i > 0 and line.kind == .blank and lines[i - 1].kind == .blank;
        if (run and i + 1 < lines.len) {
            try report.add(.blank_run, i + 1, "two blank lines in a row", .{});
        }
    }

    if (shader.main_first + 1 >= shader.main_end) return;

    for (shader.main_first + 1..shader.main_end) |i| {
        const line = lines[i];
        const next = lines[i + 1];
        const n = i + 1;
        if (line.kind != .code or next.kind == .blank) continue;

        if (next.code.len > 0 and next.code[0] == '}') continue;

        if (isGuard(line.code) and !isGuard(next.code)) {
            try report.add(.blank_after_guard, n + 1, "a blank line ends a guard group", .{});
        }

        if (std.mem.eql(u8, line.code, "}") and !continuesBlock(next.code)) {
            try report.add(.blank_after_block, n + 1, "a blank line follows a block", .{});
        }

        const is_barrier = std.mem.eql(u8, line.code, "barrier();");
        const is_rung = std.mem.startsWith(u8, next.code, "if (local < ");
        if (is_barrier and !is_rung and next.kind != .directive) {
            try report.add(.blank_after_barrier, n + 1, "a barrier ends a paragraph", .{});
        }
    }
}

fn braceRules(shader: *const Shader, report: *Report) !void {
    const heads = [_][]const u8{ "for (", "while (", "if (", "else if (", "} else if (" };
    const lines = shader.lines;
    for (lines[shader.version .. lines.len - 1], shader.version..) |line, i| {
        if (line.kind != .code) continue;

        const head = for (heads) |candidate| {
            if (std.mem.startsWith(u8, line.code, candidate)) break candidate;
        } else continue;

        const open = std.mem.indexOfScalar(u8, line.code, '(').?;
        const close = closingParen(line.code, open) orelse continue;
        const body = std.mem.trim(u8, line.code[close + 1 ..], " ");
        if (std.mem.eql(u8, body, "{")) continue;

        // A guard and a barrier rung keep their one-line form.
        const is_if = !std.mem.eql(u8, head, "for (") and !std.mem.eql(u8, head, "while (");
        const rung = is_if and std.mem.eql(u8, lines[i + 1].code, "barrier();");
        const guard = is_if and isGuardBody(body);
        const keeps_form = body.len > 0 and !std.mem.startsWith(u8, body, "{");
        if (keeps_form and (rung or guard)) continue;

        try report.add(.braces, i + 1, "the body needs braces on its own lines", .{});
    }
}

// -- helpers: numerics ------------------------------------------------------

fn flatHeader(arena: Allocator, shader: *const Shader) ![]const u8 {
    var flat: std.ArrayList(u8) = .empty;
    for (shader.lines[0..shader.version]) |line| {
        try flat.appendSlice(arena, std.mem.trim(u8, line.raw, "/ "));
        try flat.append(arena, ' ');
    }

    return flat.toOwnedSlice(arena);
}

fn numericRules(arena: Allocator, shader: *const Shader, report: *Report) !void {
    const flat = try flatHeader(arena, shader);
    const bit_exact = std.mem.indexOf(u8, flat, "Bit-exact") != null;
    for (shader.lines[shader.version..], shader.version + 1..) |line, n| {
        if (line.kind != .code) continue;

        for (banned_calls) |call| {
            const at = std.mem.indexOf(u8, line.code, call) orelse continue;
            const end = at + call.len;
            const called = end < line.code.len and line.code[end] == '(';
            if (called and containsWord(line.code, call)) {
                try report.add(.banned_call, n, "`{s}` leaves the evaluation order open", .{call});
            }
        }

        const in_main = n > shader.main_first + 1 and n <= shader.main_end;
        if (bit_exact and in_main and std.mem.startsWith(u8, line.code, "float ")) {
            try report.add(.bit_exact_precise, n, "a bit-exact kernel marks floats precise", .{});
        }
    }

    if (!bit_exact) {
        try toleranceRule(arena, shader, flat, report);
    }
}

/// Compare the tolerance a header claims against the `requires` of its Zig wrapper.
fn toleranceRule(arena: Allocator, shader: *const Shader, flat: []const u8, report: *Report) !void {
    if (std.mem.indexOf(u8, flat, "declares no tolerance") != null) return;

    const key = "`.relative = ";
    const at = std.mem.indexOf(u8, flat, key) orelse {
        try report.add(.tolerance, 1, "the header names no tolerance", .{});
        return;
    };
    const value_end = std.mem.indexOfScalarPos(u8, flat, at + key.len, '`') orelse return;
    const value = flat[at + key.len .. value_end];

    const name_end = std.mem.lastIndexOfScalar(u8, flat[0..at], '`') orelse return;
    const name_start = std.mem.lastIndexOfScalar(u8, flat[0..name_end], '`') orelse return;
    const wrapper = flat[name_start + 1 .. name_end];

    const needle = try std.fmt.allocPrint(arena, "pub const {s} = struct", .{wrapper});
    const kernels = shader.input.kernels_zig;
    const struct_at = std.mem.indexOf(u8, kernels, needle) orelse {
        try report.add(.tolerance, 1, "no wrapper `{s}` in kernels.zig", .{wrapper});
        return;
    };

    const tolerance_key = ".tolerance = .{ .relative = ";
    const decl_at = std.mem.indexOfPos(u8, kernels, struct_at, tolerance_key) orelse return;
    const decl_value = kernels[decl_at + tolerance_key.len ..];
    const declared = decl_value[0 .. std.mem.indexOfScalar(u8, decl_value, ' ') orelse return];
    if (!std.mem.eql(u8, declared, value)) {
        const fmt = "header says {s}, `{s}` declares {s}";
        try report.add(.tolerance, 1, fmt, .{ value, wrapper, declared });
    }
}

// -- helpers: uniform control flow ------------------------------------------

/// A barrier needs uniform control flow, so a lane must not skip one. The rule marks every name
/// that derives from a lane builtin, then walks the blocks of `main`.
fn uniformRules(arena: Allocator, shader: *const Shader, report: *Report) !void {
    if (!shader.has("barrier")) return;

    const lines = shader.lines[shader.main_first..shader.main_end];
    var varying: std.ArrayList([]const u8) = .empty;
    try varying.appendSlice(arena, &lane_builtins);

    // Two passes reach a name that derives from a name declared later in the text.
    for (0..2) |_| {
        for (lines) |line| {
            if (line.kind == .code) {
                try markVarying(arena, line.code, &varying);
            }
        }
    }

    var last_barrier: usize = 0;
    for (lines, 0..) |line, i| {
        if (std.mem.eql(u8, line.code, "barrier();")) {
            last_barrier = i;
        }
    }

    // One entry per open block: true when its condition differs between lanes.
    var stack: std.ArrayList(bool) = .empty;
    var last_closed = false;
    for (lines, shader.main_first + 1..) |line, n| {
        if (line.kind != .code) continue;

        const inside = std.mem.indexOfScalar(bool, stack.items, true) != null;
        if (inside and containsWord(line.code, "barrier")) {
            try report.add(.uniform_guard, n, "a barrier sits under a lane condition", .{});
        }

        const offset = n - shader.main_first - 1;
        const leaves = containsWord(line.code, "return");
        const varies = isGuard(line.code) and mentions(line.code, varying.items);
        if (leaves and (inside or varies) and offset < last_barrier) {
            try report.add(.uniform_guard, n, "a lane exit skips a later barrier", .{});
        }

        if (std.mem.startsWith(u8, line.code, "}")) {
            if (stack.pop()) |closed| {
                last_closed = closed;
            }
        }
        if (std.mem.endsWith(u8, line.code, "{")) {
            const is_else = std.mem.startsWith(u8, line.code, "} else");
            const lane_test = mentions(line.code, varying.items);
            try stack.append(arena, lane_test or (is_else and last_closed));
        }
    }
}

/// Add the name that `code` assigns when its value derives from a varying name.
fn markVarying(arena: Allocator, code: []const u8, varying: *std.ArrayList([]const u8)) !void {
    const equals = std.mem.indexOf(u8, code, " = ") orelse return;
    if (!mentions(code[equals..], varying.items)) return;

    var head = code[0..equals];
    if (std.mem.startsWith(u8, head, "for (")) {
        head = head["for (".len..];
    }

    const start = (std.mem.lastIndexOfScalar(u8, head, ' ') orelse return) + 1;
    const name = head[start..];
    if (name.len == 0 or !isWordByte(name[0])) return;

    if (!mentions(name, varying.items)) {
        try varying.append(arena, name);
    }
}

// -- helpers: header truth --------------------------------------------------

fn isAccessWord(word: []const u8) bool {
    const words = [_][]const u8{ "readonly", "writeonly", "readwrite", "unused" };
    for (words) |candidate| {
        if (std.mem.eql(u8, candidate, word)) return true;
    }

    return false;
}

/// Every `binding = N` of the code has a row in Bindings, and every row has a declaration unless
/// the row says `unused`.
fn bindingRows(shader: *const Shader, report: *Report) !void {
    const s = shader.span(.Bindings) orelse return;
    var rows: [32]bool = @splat(false);
    var unused: [32]bool = @splat(false);
    for (shader.lines[s.first..s.end]) |line| {
        var it = std.mem.tokenizeAny(u8, line.raw, "/ ");
        const word = it.next() orelse continue;
        const index = std.fmt.parseInt(usize, word, 10) catch continue;
        const access = it.next() orelse continue;
        if (index >= rows.len or !isAccessWord(access)) continue;

        rows[index] = true;
        unused[index] = std.mem.indexOf(u8, line.raw, "unused") != null;
    }

    var declared: [32]bool = @splat(false);
    const key = "binding = ";
    for (shader.lines[shader.version..], shader.version + 1..) |line, n| {
        const at = std.mem.indexOf(u8, line.code, key) orelse continue;
        var it = std.mem.tokenizeAny(u8, line.code[at + key.len ..], ") ");
        const index = std.fmt.parseInt(usize, it.next() orelse continue, 10) catch continue;
        if (index >= declared.len) continue;

        declared[index] = true;
        if (!rows[index]) {
            try report.add(.binding_row, n, "binding {d} has no row in Bindings", .{index});
        }
    }

    for (rows, 0..) |row, index| {
        if (row and !declared[index] and !unused[index]) {
            try report.add(.binding_row, s.first + 1, "row {d} has no declaration", .{index});
        }
    }
}

// -- tests ------------------------------------------------------------------

const clean_prefix =
    \\// demo: a kernel of the tests.
    \\//
    \\// What
    \\//   It proves the rules.
    \\//
    \\// Hardware
    \\//   Lane i owns element i.
    \\//
    \\// Numerics
    \\//   Tolerance: `Demo` declares `.relative = 1e-3`.
    \\//
    \\// Bindings
    \\//   0     readonly  f32 x[n]
    \\//   push  uint n
    \\
    \\#version 450
    \\
    \\layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
    \\
    \\// -- bindings ---------------------------------------------------------------
    \\layout(std430, set = 0, binding = 0) readonly buffer XBuf {
    \\    float data[];
    \\} x_buf;
    \\
    \\layout(push_constant) uniform Params {
    \\    uint n;
    \\} params;
    \\
    \\// -- main -------------------------------------------------------------------
    \\void main() {
    \\
;

const demo_kernels = "pub const Demo = struct {\n.tolerance = .{ .relative = 1e-3 },\n};\n";

fn rulesOfSource(arena: Allocator, source: []const u8) ![]Rule {
    const input: Input = .{ .stem = "demo", .source = source, .kernels_zig = demo_kernels };
    const findings = try lint(arena, input);
    const rules = try arena.alloc(Rule, findings.len);
    for (findings, rules) |f, *rule| {
        rule.* = f.rule;
    }

    return rules;
}

fn hasRule(body: []const u8, rule: Rule) !bool {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const source = try std.mem.concat(arena, u8, &.{ clean_prefix, body, "}\n" });

    return std.mem.indexOfScalar(Rule, try rulesOfSource(arena, source), rule) != null;
}

fn expectRule(body: []const u8, want: Rule) !void {
    try std.testing.expect(try hasRule(body, want));
}

fn expectNoRule(body: []const u8, absent: Rule) !void {
    try std.testing.expect(!try hasRule(body, absent));
}

test "a clean kernel holds no finding" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const body = "    uint i = gl_GlobalInvocationID.x;\n    if (i >= params.n) return;\n\n" ++
        "    x_buf.data[i] = 0.0;\n}\n";
    const source = try std.mem.concat(arena, u8, &.{ clean_prefix, body });
    try std.testing.expectEqualSlices(Rule, &.{}, try rulesOfSource(arena, source));
}

test "each text rule fires on its shape" {
    const Case = struct { body: []const u8, want: Rule };
    const cases = [_]Case{
        .{ .body = "    // " ++ "a" ** 100 ++ ".\n", .want = .width },
        .{ .body = "\tx;\n", .want = .tab },
        .{ .body = "    x; \n", .want = .trailing_space },
        .{ .body = "    // caf\xc3\xa9.\n", .want = .ascii },
        .{ .body = "    /* x */\n", .want = .block_comment },
        .{ .body = "    // A diagram \\\n    y;\n", .want = .comment_backslash },
        .{ .body = "    x;\n\n\n    y;\n", .want = .blank_run },
    };
    for (cases) |c| {
        try expectRule(c.body, c.want);
    }
}

test "each paragraph rule fires on its shape" {
    const Case = struct { body: []const u8, want: Rule };
    const cases = [_]Case{
        .{ .body = "    if (i) return;\n    x = 1;\n", .want = .blank_after_guard },
        .{ .body = "    for (a; b; c) {\n        x;\n    }\n    y;\n", .want = .blank_after_block },
        .{ .body = "    barrier();\n    y;\n", .want = .blank_after_barrier },
    };
    for (cases) |c| {
        try expectRule(c.body, c.want);
    }
}

test "a ladder of rungs is one paragraph" {
    const body = "    barrier();\n    if (local < 2u) z;\n    barrier();\n" ++
        "    if (local < 1u) z;\n    barrier();\n";
    try expectNoRule(body, .blank_after_barrier);
    try expectNoRule(body, .braces);
}

test "the brace rule keeps guards and rungs on one line" {
    const Case = struct { body: []const u8, want: Rule };
    const cases = [_]Case{
        .{ .body = "    for (a; b; c) x;\n", .want = .braces },
        .{ .body = "    if (a) x;\n    y;\n", .want = .braces },
        .{ .body = "    if (a) { x; }\n", .want = .braces },
    };
    for (cases) |c| {
        try expectRule(c.body, c.want);
    }

    try expectNoRule("    if (a) return;\n", .braces);
    try expectNoRule("    if (local < 2u) x;\n    barrier();\n", .braces);
}

test "the comment rules read one block" {
    const Case = struct { body: []const u8, want: Rule };
    const cases = [_]Case{
        .{ .body = "    // One.\n    // Two.\n    // Three.\n    x;\n", .want = .comment_block },
        .{ .body = "    // 1. Load the row.\n    x;\n", .want = .step_label },
        .{ .body = "    // Pass 2: x.\n    y;\n", .want = .step_label },
    };
    for (cases) |c| {
        try expectRule(c.body, c.want);
    }

    try expectNoRule("    // Pass 1: x.\n    y;\n\n    // Pass 2: y.\n    z;\n", .step_label);
}

test "a lane must not skip a barrier" {
    const lane = "    uint l = gl_LocalInvocationID.x;\n";
    try expectRule(lane ++ "    if (l >= 4u) return;\n\n    barrier();\n", .uniform_guard);
    try expectRule(lane ++ "\n    if (l < 4u) {\n        barrier();\n    }\n", .uniform_guard);

    // A second name that derives from a lane name counts.
    const derived = lane ++ "    uint d = l * 2u;\n";
    try expectRule(derived ++ "\n    if (d >= 4u) return;\n\n    barrier();\n", .uniform_guard);

    // A workgroup id is uniform, and an exit past the last barrier skips none.
    const group = "    uint g = gl_WorkGroupID.x;\n    if (g >= 4u) return;\n\n    barrier();\n";
    try expectNoRule(group, .uniform_guard);
    try expectNoRule(lane ++ "    barrier();\n\n    if (l >= 4u) return;\n", .uniform_guard);
}

test "the historic tail lane bug fails and the clamp does not" {
    const head = "    uint col = gl_GlobalInvocationID.x;\n";
    const tail_returns = head ++ "    if (col >= params.n) return;\n\n    shared_sum();\n" ++
        "    barrier();\n\n    x_buf.data[col] = 0.0;\n";
    try expectRule(tail_returns, .uniform_guard);

    // The tail lane stays in the workgroup, reaches the barrier, and skips only the store.
    const tail_stays = head ++ "    bool live = col < params.n;\n\n    shared_sum();\n" ++
        "    barrier();\n\n    if (live) {\n        x_buf.data[col] = 0.0;\n    }\n";
    try expectNoRule(tail_stays, .uniform_guard);
}

test "a banned call fails and a longer name does not" {
    try expectRule("    float s = dot(a, b);\n", .banned_call);
    try expectNoRule("    float length_of = 1.0;\n", .banned_call);
}

test "a header that lies about a binding or a tolerance fails" {
    const Case = struct { from: []const u8, to: []const u8, want: Rule };
    const cases = [_]Case{
        .{ .from = "//   0     readonly", .to = "//   1     readonly", .want = .binding_row },
        .{ .from = "`.relative = 1e-3`", .to = "`.relative = 1e-4`", .want = .tolerance },
        .{ .from = "// demo: ", .to = "// other: ", .want = .title },
    };
    for (cases) |c| {
        var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena_state.deinit();

        const arena = arena_state.allocator();
        const tail = "    x_buf.data[0] = 0.0;\n}\n";
        const whole = try std.mem.concat(arena, u8, &.{ clean_prefix, tail });
        const edited = try std.mem.replaceOwned(u8, arena, whole, c.from, c.to);
        const got = try rulesOfSource(arena, edited);
        try std.testing.expect(std.mem.indexOfScalar(Rule, got, c.want) != null);
    }
}
