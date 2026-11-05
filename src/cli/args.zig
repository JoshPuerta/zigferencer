//! The flag parser of every command. A field `snake_case`
//! is the flag `--snake-case`, and the field default is the flag default. The
//! same struct generates the flag lines of the `--help` text, so the parse and
//! the help cannot drift.
//!
//! A command type declares `help` (the usage line, the prose, the footer) and
//! `docs` (one `Doc` per flag). It can declare `positional`, the name of one
//! `?[]const u8` field that takes the one bare argument, and `omit`, nested
//! fields that the command does not take. A doc of the command replaces the doc
//! of a nested type for one field, so a shared option keeps one name and can
//! state a default of its own command. A type can also name `docs_from`, a
//! type whose `docs` cover nested fields that carry no help of their own.
//!
//! Every error prints one `error:` line to stderr and returns
//! `error.UsageError`. `main` exits cleanly on that error, with no developer
//! stack trace. The full usage block belongs to `--help`.

const std = @import("std");
const Io = std.Io;
const style = @import("style.zig");

// -- constants --------------------------------------------------------------

const diag_buffer_bytes = 256;
const help_buffer_bytes = 1024;
const stderr_buffer_bytes = 512;

/// The compile-time walk over the flag structs of the largest command needs this budget.
const eval_branch_quota = 200_000;

/// The text column of the flag lines. A longer label puts its text on the next line.
const label_column_max = 26;

/// The right margin of a flag line, after which a default moves down.
const line_width_max = 88;

// -- types ------------------------------------------------------------------

/// The help text of one flag.
pub const Doc = struct {
    /// A `\n` starts a continuation line under the text column.
    text: []const u8,
    /// The header the flag prints under. The groups print in the order of their first flag.
    group: []const u8 = "",
    /// The value placeholder. Null derives it from the type.
    value: ?[]const u8 = null,
    /// The text of `[default: X]`. Null derives it from the field default.
    /// An empty string prints no default.
    default: ?[]const u8 = null,
    /// A second spelling of the flag, such as `-f`.
    alias: ?[]const u8 = null,
};

/// The text around the generated flag lines of `--help`.
pub const Help = struct {
    /// The usage line after `usage: `.
    usage: []const u8,
    /// The prose under the usage line. It ends with a newline.
    about: []const u8,
    /// The text after the flag lines, such as the examples. It ends with a newline.
    footer: []const u8 = "",
};

/// A flag that can occur up to `cap` times. The parser refuses one more.
pub fn Repeated(comptime T: type, comptime cap: usize) type {
    comptime std.debug.assert(cap > 0);

    return struct {
        const Self = @This();
        pub const Item = T;
        pub const capacity = cap;

        /// Only `items[0..len]` is defined.
        items: [cap]T = undefined,
        len: usize = 0,

        pub fn slice(self: *const Self) []const T {
            std.debug.assert(self.len <= cap);

            return self.items[0..self.len];
        }
    };
}

/// The text of the last parse error, so the parse itself writes nothing.
pub const Diag = struct {
    buf: [diag_buffer_bytes]u8 = undefined,
    len: usize = 0,

    pub fn text(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }

    fn fail(self: *Diag, comptime fmt: []const u8, fmtargs: anytype) error{UsageError} {
        var w: Io.Writer = .fixed(&self.buf);
        // A message past the buffer keeps its head, which names the flag.
        w.print(fmt, fmtargs) catch {};
        self.len = w.end;
        std.debug.assert(self.len <= self.buf.len);
        return error.UsageError;
    }
};

/// How the parser reads a flag type and how the help prints it.
const Kind = enum { boolean, int, float, string, enumeration, custom, repeated, nested };

/// One flag of a command, as the parse refusals and the help need it.
const Entry = struct {
    /// The long spelling, such as `--max-seq`.
    name: []const u8,
    /// The left column of the flag line.
    label: []const u8,
    doc: Doc,
    /// The text of `[default: X]`. An empty string prints no default.
    default: []const u8,
    boolean: bool,
};

/// A read position over `argv`. `Argv` is a slice or an array pointer of byte slices.
fn Cursor(comptime Argv: type) type {
    return struct {
        const Self = @This();

        argv: Argv,
        index: usize = 0,

        fn next(self: *Self) ?[]const u8 {
            if (self.index >= self.argv.len) return null;

            const token: []const u8 = self.argv[self.index];
            self.index += 1;
            return token;
        }
    };
}

/// One bit per flag of `Top`, set when the flag has been given.
fn SeenSet(comptime Top: type) type {
    @setEvalBranchQuota(eval_branch_quota);
    return std.StaticBitSet(flatten(&.{Top}).len);
}

/// The type of a flag value, which an optional field wraps.
fn Base(comptime F: type) type {
    return switch (@typeInfo(F)) {
        .optional => |o| o.child,
        else => F,
    };
}

// -- api --------------------------------------------------------------------

/// Parse `ctx.args` into a value of `T`. A `--help` or `-h` prints the help of
/// `T` to stdout and returns null. A bad invocation prints one `error:` line
/// and returns `error.UsageError`. `ctx` is a `cli.Context`.
pub fn parseInto(comptime T: type, ctx: anytype) error{UsageError}!?T {
    var diag: Diag = .{};
    return parse(T, ctx.args, &diag) catch |err| switch (err) {
        error.Help => {
            printHelp(T, ctx.io, .stdout(), style.Palette.init(ctx.color));
            return null;
        },
        error.UsageError => fail(ctx.io, "{s}", .{diag.text()}),
    };
}

/// Fill a value of `T` from `argv`, a slice of byte slices. The parse writes
/// nothing. An error leaves its message in `diag`.
pub fn parse(comptime T: type, argv: anytype, diag: *Diag) error{ UsageError, Help }!T {
    comptime validate(T);

    var out: T = .{};
    var seen: SeenSet(T) = .initEmpty();
    var positional_seen = false;
    var cursor: Cursor(@TypeOf(argv)) = .{ .argv = argv };

    // Each pass consumes at least one token, so the loop ends in argv.len passes.
    while (cursor.next()) |token| {
        if (std.mem.eql(u8, token, "--help") or std.mem.eql(u8, token, "-h")) return error.Help;

        if (std.mem.startsWith(u8, token, "-")) {
            const taken = try setFlag(&.{T}, &out, token, &cursor, &seen, diag);
            if (!taken) return failUnknown(T, token, diag);
            continue;
        }

        if (comptime !@hasDecl(T, "positional")) {
            return diag.fail("unexpected argument '{s}'", .{token});
        }
        if (positional_seen) return diag.fail("unexpected extra argument '{s}'", .{token});

        @field(out, T.positional) = token;
        positional_seen = true;
    }

    std.debug.assert(cursor.index == argv.len);
    return out;
}

/// The whole `--help` text of `T` after the usage line, built at compile time.
pub fn helpBody(comptime T: type) []const u8 {
    comptime {
        validate(T);
        @setEvalBranchQuota(eval_branch_quota);

        const entries = flatten(&.{T});
        const width = textColumn(entries);

        var out: []const u8 = T.help.about;
        var groups: []const []const u8 = &.{};
        for (entries) |e| {
            if (contains(groups, e.doc.group)) continue;

            groups = groups ++ .{e.doc.group};
            out = out ++ groupLines(entries, e.doc.group, width);
        }

        if (T.help.footer.len > 0) {
            out = out ++ "\n" ++ T.help.footer;
        }

        return out;
    }
}

/// Print the help of `T` to `file`: stdout when the user asked for it, stderr
/// when a usage error shows it. A failed write has nowhere to go.
pub fn printHelp(comptime T: type, io: Io, file: Io.File, p: style.Palette) void {
    const body = comptime helpBody(T);

    var buffer: [help_buffer_bytes]u8 = undefined;
    var file_writer: Io.File.Writer = .initStreaming(file, io, &buffer);
    const w = &file_writer.interface;

    w.print("{s}usage:{s} " ++ T.help.usage ++ "\n\n", .{ p.bold, p.reset }) catch return;
    w.writeAll(body) catch return;
    w.flush() catch return;
}

/// Write `bytes` to stderr for usage text and diagnostics. The function discards a
/// failed write, like `std.debug.print`. A broken stderr must not turn a usage
/// line into a `WriteFailed` that the caller carries back up to `main`.
pub fn writeStderr(io: Io, bytes: []const u8) void {
    var stderr_buffer: [stderr_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    w.writeAll(bytes) catch return;
    w.flush() catch {};
}

/// Print `error: <msg>` to stderr and return `error.UsageError`. The parser and
/// the validation of each command both call it, so every refusal reads the same.
pub fn fail(io: Io, comptime fmt: []const u8, fmtargs: anytype) error{UsageError} {
    var stderr_buffer: [stderr_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    w.print("error: " ++ fmt ++ "\n", fmtargs) catch {};
    w.flush() catch {};
    return error.UsageError;
}

// -- helpers: parse ---------------------------------------------------------

/// Set the field whose flag is `token`. False means no field takes it.
/// `scopes` runs from the command type down to the type of `out`.
fn setFlag(
    comptime scopes: []const type,
    out: *scopes[scopes.len - 1],
    token: []const u8,
    cursor: anytype,
    seen: *SeenSet(scopes[0]),
    diag: *Diag,
) error{UsageError}!bool {
    const Container = scopes[scopes.len - 1];

    inline for (comptime std.meta.fields(Container)) |f| {
        if (comptime isPositional(Container, f.name)) continue;

        const slot = &@field(out, f.name);
        if (comptime isNested(f.type)) {
            if (try setFlag(scopes ++ .{f.type}, slot, token, cursor, seen, diag)) return true;
            continue;
        }

        if (comptime omitted(scopes[0], f.name)) continue;

        const doc = comptime docFor(scopes, f.name);
        const name = comptime flagName(f.name);
        if (matches(token, name, doc.alias)) {
            try refuseRepeat(scopes[0], f.type, name, seen, diag);
            try setValue(f.type, name, slot, cursor, diag);
            return true;
        }
    }

    return false;
}

fn matches(token: []const u8, name: []const u8, alias: ?[]const u8) bool {
    if (std.mem.eql(u8, token, name)) return true;

    const spelling = alias orelse return false;
    return std.mem.eql(u8, token, spelling);
}

/// Refuse a second use of the flag `name` of `Top`, because the last value of a
/// scalar flag wins in silence. A bool is idempotent, and `Repeated` has its own cap.
fn refuseRepeat(
    comptime Top: type,
    comptime F: type,
    comptime name: []const u8,
    seen: *SeenSet(Top),
    diag: *Diag,
) error{UsageError}!void {
    const kind = comptime kindOf(Base(F));
    if (kind == .repeated or kind == .boolean) return;

    const bit = comptime entryIndex(Top, name);
    if (seen.isSet(bit)) return diag.fail(name ++ " given twice", .{});

    seen.set(bit);
}

/// The position of the flag `name` in the flag list of `Top`.
fn entryIndex(comptime Top: type, comptime name: []const u8) usize {
    for (flatten(&.{Top}), 0..) |e, k| {
        if (std.mem.eql(u8, e.name, name)) return k;
    }

    @compileError("flag " ++ name ++ " is not in " ++ @typeName(Top));
}

fn setValue(
    comptime F: type,
    comptime name: []const u8,
    slot: *F,
    cursor: anytype,
    diag: *Diag,
) error{UsageError}!void {
    const B = Base(F);
    const kind = comptime kindOf(B);
    if (kind == .boolean) {
        slot.* = true;
        return;
    }

    const raw = cursor.next() orelse return diag.fail(name ++ " requires a value", .{});
    if (kind == .repeated) {
        if (slot.len >= B.capacity) {
            return diag.fail(name ++ " accepts at most {d} values", .{B.capacity});
        }

        slot.items[slot.len] = try parseValue(B.Item, name, raw, diag);
        slot.len += 1;
        return;
    }

    slot.* = try parseValue(B, name, raw, diag);
}

fn parseValue(
    comptime B: type,
    comptime name: []const u8,
    raw: []const u8,
    diag: *Diag,
) error{UsageError}!B {
    const not_number = name ++ ": '{s}' is not a number";
    const not_valid = name ++ " '{s}' is not a valid value. Use ";

    return switch (comptime kindOf(B)) {
        .int => std.fmt.parseInt(B, raw, 10) catch |err| switch (err) {
            error.Overflow => diag.fail(name ++ ": '{s}' is out of range", .{raw}),
            error.InvalidCharacter => diag.fail(not_number, .{raw}),
        },
        .float => std.fmt.parseFloat(B, raw) catch diag.fail(not_number, .{raw}),
        .string => raw,
        .enumeration => std.meta.stringToEnum(B, raw) orelse
            diag.fail(not_valid ++ comptime tagChoices(B), .{raw}),
        .custom => B.parseFlag(raw) orelse diag.fail(not_valid ++ B.flag_choices, .{raw}),
        .boolean, .repeated, .nested => unreachable,
    };
}

/// The words of the refusal of a bad enum value: "a or b", or "a, b, or c".
fn tagChoices(comptime E: type) []const u8 {
    comptime {
        const fields = std.meta.fields(E);
        var out: []const u8 = "";
        for (fields, 0..) |f, i| {
            out = out ++ choiceSeparator(i, fields.len) ++ f.name;
        }

        return out;
    }
}

/// The words before choice `index` of a list of `count` choices.
fn choiceSeparator(comptime index: usize, comptime count: usize) []const u8 {
    if (index == 0) return "";
    if (index + 1 < count) return ", ";
    return if (count == 2) " or " else ", or ";
}

/// The refusal of a token that no flag takes. A known flag with `=value` gets
/// the spelling that works, because the parser reads the value as the next token.
fn failUnknown(comptime T: type, token: []const u8, diag: *Diag) error{UsageError} {
    const eq = std.mem.indexOfScalar(u8, token, '=') orelse
        return diag.fail("unknown flag '{s}'", .{token});

    const head = token[0..eq];
    const value = token[eq + 1 ..];
    inline for (comptime flatten(&.{T})) |e| {
        if (matches(head, e.name, e.doc.alias)) {
            if (e.boolean) return diag.fail("{s} takes no value", .{head});
            return diag.fail("unknown flag '{s}'. Write '{s} {s}'", .{ token, head, value });
        }
    }

    return diag.fail("unknown flag '{s}'", .{token});
}

// -- helpers: help text -----------------------------------------------------

fn textColumn(comptime entries: []const Entry) usize {
    var width: usize = 0;
    for (entries) |e| {
        width = @max(width, e.label.len);
    }

    return @min(width + 2, label_column_max);
}

/// The blank line, the header, and the flag lines of one group.
fn groupLines(
    comptime entries: []const Entry,
    comptime group: []const u8,
    comptime width: usize,
) []const u8 {
    comptime {
        var out: []const u8 = "\n";
        if (group.len > 0) {
            out = out ++ "  " ++ group ++ "\n";
        }

        for (entries) |e| {
            if (std.mem.eql(u8, e.doc.group, group)) {
                out = out ++ flagLines(e, width);
            }
        }

        return out;
    }
}

/// The label in the left column, then the text with its default in the right column.
fn flagLines(comptime e: Entry, comptime width: usize) []const u8 {
    comptime {
        const pad = " " ** (width + 2);

        var out: []const u8 = "  " ++ e.label;
        if (e.label.len + 2 > width) {
            out = out ++ "\n" ++ pad;
        } else {
            out = out ++ " " ** (width - e.label.len);
        }

        var lines = std.mem.splitScalar(u8, e.doc.text ++ defaultSuffix(e, width), '\n');
        out = out ++ lines.first();
        while (lines.next()) |line| {
            out = out ++ "\n" ++ pad ++ line;
        }

        return out ++ "\n";
    }
}

/// The `[default: X]` text after the description. A default that runs past the
/// right margin starts its own line.
fn defaultSuffix(comptime e: Entry, comptime width: usize) []const u8 {
    comptime {
        if (e.default.len == 0) return "";

        const gap = "   ";
        const open = "[default: ";
        const close = "]";

        const text = e.doc.text;
        const text_last_start = if (std.mem.lastIndexOfScalar(u8, text, '\n')) |k| k + 1 else 0;
        const text_last_len = text.len - text_last_start;
        const default_head_len = std.mem.indexOfScalar(u8, e.default, '\n') orelse e.default.len;

        const suffix_len = gap.len + open.len + default_head_len + close.len;
        const line_end = width + 2 + text_last_len + suffix_len;
        const joint = if (line_end > line_width_max) "\n" else gap;
        return joint ++ open ++ e.default ++ close;
    }
}

fn valueName(comptime F: type) []const u8 {
    return switch (kindOf(F)) {
        .boolean => "",
        .int => "N",
        .float => "X",
        .string => "TEXT",
        .enumeration => tagList(F),
        .custom => F.flag_value,
        .repeated => valueName(F.Item),
        .nested => unreachable,
    };
}

/// The placeholder of an enum flag: "a|b|c".
fn tagList(comptime E: type) []const u8 {
    comptime {
        var out: []const u8 = "";
        for (std.meta.fields(E), 0..) |f, i| {
            out = out ++ (if (i == 0) "" else "|") ++ f.name;
        }

        return out;
    }
}

fn defaultText(comptime f: std.builtin.Type.StructField) []const u8 {
    const value = f.defaultValue() orelse
        @compileError("flag field '" ++ f.name ++ "' needs a default");

    return switch (@typeInfo(f.type)) {
        .optional => if (value) |v| valueText(@TypeOf(v), v) else "",
        .bool => "",
        else => valueText(f.type, value),
    };
}

fn valueText(comptime F: type, comptime v: F) []const u8 {
    return switch (kindOf(F)) {
        .int, .float => std.fmt.comptimePrint("{d}", .{v}),
        .string => if (v.len == 0) "" else "\"" ++ v ++ "\"",
        .enumeration => @tagName(v),
        .boolean, .custom, .repeated, .nested => "",
    };
}

// -- helpers: struct reflection ---------------------------------------------

/// Every flag of the last type of `scopes`, nested fields flattened, in field order.
fn flatten(comptime scopes: []const type) []const Entry {
    comptime {
        const Container = scopes[scopes.len - 1];

        var out: []const Entry = &.{};
        for (std.meta.fields(Container)) |f| {
            if (isPositional(Container, f.name)) continue;

            if (isNested(f.type)) {
                out = out ++ flatten(scopes ++ .{f.type});
                continue;
            }

            if (omitted(scopes[0], f.name)) continue;

            out = out ++ .{entryOf(f, docFor(scopes, f.name))};
        }

        return out;
    }
}

fn entryOf(comptime f: std.builtin.Type.StructField, comptime doc: Doc) Entry {
    const flag = flagName(f.name);
    const spelled = if (doc.alias) |a| a ++ ", " ++ flag else flag;
    const value = doc.value orelse valueName(Base(f.type));

    return .{
        .name = flag,
        .label = if (value.len > 0) spelled ++ " " ++ value else spelled,
        .doc = doc,
        .default = doc.default orelse defaultText(f),
        .boolean = kindOf(Base(f.type)) == .boolean,
    };
}

fn kindOf(comptime F: type) Kind {
    const unsupported = "flag type " ++ @typeName(F) ++ " is not supported";

    return switch (@typeInfo(F)) {
        .bool => .boolean,
        .int => .int,
        .float => .float,
        .@"enum" => .enumeration,
        .pointer => if (F == []const u8) .string else @compileError(unsupported),
        .@"struct", .@"union" => compositeKind(F),
        else => @compileError(unsupported),
    };
}

/// A struct declares its kind first, because `Repeated` and a custom flag are structs too.
fn compositeKind(comptime F: type) Kind {
    if (@hasDecl(F, "parseFlag")) return .custom;
    if (@hasDecl(F, "Item") and @hasDecl(F, "capacity")) return .repeated;
    if (@typeInfo(F) == .@"struct") return .nested;

    @compileError("flag type " ++ @typeName(F) ++ " needs parseFlag");
}

fn isNested(comptime F: type) bool {
    return @typeInfo(F) != .optional and kindOf(F) == .nested;
}

fn flagName(comptime field: []const u8) []const u8 {
    comptime {
        var name = field[0..field.len].*;
        std.mem.replaceScalar(u8, &name, '_', '-');
        const final = name;
        return "--" ++ &final;
    }
}

fn contains(comptime list: []const []const u8, comptime name: []const u8) bool {
    for (list) |item| {
        if (std.mem.eql(u8, item, name)) return true;
    }

    return false;
}

fn omitted(comptime Top: type, comptime name: []const u8) bool {
    return @hasDecl(Top, "omit") and contains(Top.omit, name);
}

fn isPositional(comptime Container: type, comptime name: []const u8) bool {
    return @hasDecl(Container, "positional") and std.mem.eql(u8, Container.positional, name);
}

/// The doc of field `name`. `scopes` runs from the command type down to the
/// type that holds the field, and the outermost doc wins, so a command can
/// restate a shared option.
fn docFor(comptime scopes: []const type, comptime name: []const u8) Doc {
    for (scopes) |S| {
        if (@hasDecl(S, "docs") and @hasField(@TypeOf(S.docs), name)) {
            return @field(S.docs, name);
        }
        if (@hasDecl(S, "docs_from") and @hasField(@TypeOf(S.docs_from.docs), name)) {
            return @field(S.docs_from.docs, name);
        }
    }

    const holder = @typeName(scopes[scopes.len - 1]);
    @compileError("flag field '" ++ name ++ "' of " ++ holder ++ " has no doc");
}

// -- helpers: compile-time checks -------------------------------------------

/// The compile-time checks of a command type: unique flag names, one doc per
/// flag and no doc for a missing flag, a real positional field, and real omit
/// names.
fn validate(comptime T: type) void {
    comptime {
        @setEvalBranchQuota(eval_branch_quota);

        if (!@hasDecl(T, "help")) {
            @compileError(@typeName(T) ++ " needs a help declaration");
        }

        validateNames(T);
        validateDocs(T);
        if (@hasDecl(T, "omit")) {
            validateOmit(T);
        }

        if (@hasDecl(T, "positional")) {
            validatePositional(T);
        }
    }
}

fn validateNames(comptime T: type) void {
    comptime {
        var names: []const []const u8 = &.{};
        for (flatten(&.{T})) |e| {
            if (std.mem.eql(u8, e.name, "--help")) {
                @compileError("--help is reserved");
            }

            names = uniqueName(T, names, e.name);
            if (e.doc.alias) |a| {
                names = uniqueName(T, names, a);
            }
        }
    }
}

fn uniqueName(
    comptime T: type,
    comptime names: []const []const u8,
    comptime name: []const u8,
) []const []const u8 {
    if (contains(names, name)) {
        @compileError("flag " ++ name ++ " repeats in " ++ @typeName(T));
    }

    return names ++ .{name};
}

/// Every doc of `Container` and of its nested types names one of their fields.
fn validateDocs(comptime Container: type) void {
    if (@hasDecl(Container, "docs")) {
        for (std.meta.fields(@TypeOf(Container.docs))) |d| {
            if (!hasFlatField(Container, d.name)) {
                const holder = @typeName(Container);
                @compileError("doc '" ++ d.name ++ "' of " ++ holder ++ " names no field");
            }
        }
    }

    for (std.meta.fields(Container)) |f| {
        if (isNested(f.type)) {
            validateDocs(f.type);
        }
    }
}

fn validateOmit(comptime T: type) void {
    for (T.omit) |o| {
        if (!hasFlatField(T, o)) {
            @compileError("omit '" ++ o ++ "' of " ++ @typeName(T) ++ " names no field");
        }
    }
}

fn validatePositional(comptime T: type) void {
    if (!@hasField(T, T.positional)) {
        @compileError("positional '" ++ T.positional ++ "' names no field");
    }
    if (@FieldType(T, T.positional) != ?[]const u8) {
        @compileError("positional field must be ?[]const u8");
    }
}

fn hasFlatField(comptime Container: type, comptime name: []const u8) bool {
    for (std.meta.fields(Container)) |f| {
        if (std.mem.eql(u8, f.name, name)) return true;
        if (isNested(f.type) and hasFlatField(f.type, name)) return true;
    }

    return false;
}

// -- tests ------------------------------------------------------------------

const Mode = enum { fast, slow };
const Pick = enum { a, b, c };

const Inner = struct {
    depth: u16 = 4,
    label: ?[]const u8 = null,

    pub const docs = .{
        .depth = Doc{ .text = "inner depth", .group = "inner" },
        .label = Doc{ .text = "inner label", .group = "inner" },
    };
};

/// The test struct: every field kind the parser takes.
const Sample = struct {
    name: ?[]const u8 = null,
    count: u32 = 7,
    small: i8 = -3,
    ratio: f32 = 0.5,
    verbose: bool = false,
    mode: Mode = .fast,
    pick: ?Pick = null,
    limit: ?u64 = null,
    text: []const u8 = "hi",
    tags: Repeated([]const u8, 3) = .{},
    inner: Inner = .{},
    hidden: u8 = 1,

    pub const positional = "name";
    pub const omit: []const []const u8 = &.{"hidden"};
    pub const help: Help = .{ .usage = "sample <name> [flags]", .about = "A test command.\n" };
    pub const docs = .{
        .count = Doc{ .text = "a count" },
        .small = Doc{ .text = "a signed count" },
        .ratio = Doc{ .text = "a ratio" },
        .verbose = Doc{ .text = "say more", .alias = "-v" },
        .mode = Doc{ .text = "the mode" },
        .pick = Doc{ .text = "a pick" },
        .limit = Doc{ .text = "a limit" },
        .text = Doc{ .text = "some text\nover two lines" },
        .tags = Doc{ .text = "a tag, repeatable" },
        .depth = Doc{ .text = "the depth the command states", .group = "inner" },
    };
};

fn parseTest(comptime T: type, argv: []const []const u8) error{ UsageError, Help }!T {
    var diag: Diag = .{};
    return parse(T, argv, &diag);
}

test "parse: every unknown or malformed flag is a usage error" {
    const bad = [_][]const []const u8{
        &.{"--nope"},
        &.{"-x"},
        &.{"--hidden"},
        &.{"--count"},
        &.{ "--count", "x" },
        &.{ "--count", "-1" },
        &.{ "--count", "4294967296" },
        &.{ "--small", "128" },
        &.{ "--ratio", "fast" },
        &.{ "--mode", "medium" },
        &.{ "--pick", "d" },
        &.{"--text"},
        &.{ "--tag", "a" },
        &.{ "--tags", "a", "--tags", "b", "--tags", "c", "--tags", "d" },
        &.{ "one", "two" },
        &.{"--verbose=true"},
        &.{ "--count", "1", "--count", "2" },
        &.{ "--count", "1", "--count", "1" },
        &.{ "--mode", "fast", "--mode", "slow" },
        &.{ "--text", "a", "--text", "b" },
        &.{"--count=5"},
    };
    for (bad) |argv| {
        try std.testing.expectError(error.UsageError, parseTest(Sample, argv));
    }
}

test "parse: the messages name the flag and the value" {
    var diag: Diag = .{};
    try std.testing.expectError(error.UsageError, parse(Sample, &[_][]const u8{"--nope"}, &diag));
    try std.testing.expectEqualStrings("unknown flag '--nope'", diag.text());
    try std.testing.expectError(error.UsageError, parse(Sample, &[_][]const u8{"--count"}, &diag));
    try std.testing.expectEqualStrings("--count requires a value", diag.text());
    try std.testing.expectError(
        error.UsageError,
        parse(Sample, &[_][]const u8{ "--count", "x" }, &diag),
    );
    try std.testing.expectEqualStrings("--count: 'x' is not a number", diag.text());
    try std.testing.expectError(
        error.UsageError,
        parse(Sample, &[_][]const u8{ "--pick", "d" }, &diag),
    );
    try std.testing.expectEqualStrings(
        "--pick 'd' is not a valid value. Use a, b, or c",
        diag.text(),
    );
    try std.testing.expectError(error.UsageError, parse(Sample, &[_][]const u8{ "a", "b" }, &diag));
    try std.testing.expectEqualStrings("unexpected extra argument 'b'", diag.text());
    const messages = [_]struct { argv: []const []const u8, want: []const u8 }{
        .{ .argv = &.{ "--count", "1", "--count", "2" }, .want = "--count given twice" },
        .{ .argv = &.{"--count=5"}, .want = "unknown flag '--count=5'. Write '--count 5'" },
        .{ .argv = &.{"--verbose=true"}, .want = "--verbose takes no value" },
        .{ .argv = &.{"--nope=5"}, .want = "unknown flag '--nope=5'" },
        .{ .argv = &.{ "--count", "4294967296" }, .want = "--count: '4294967296' is out of range" },
        .{ .argv = &.{ "--small", "-129" }, .want = "--small: '-129' is out of range" },
    };
    for (messages) |m| {
        try std.testing.expectError(error.UsageError, parse(Sample, m.argv, &diag));
        try std.testing.expectEqualStrings(m.want, diag.text());
    }

    // A token past the buffer keeps the head of the message.
    const long = "--" ++ "x" ** 400;
    try std.testing.expectError(error.UsageError, parse(Sample, &[_][]const u8{long}, &diag));
    try std.testing.expect(std.mem.startsWith(u8, diag.text(), "unknown flag '--xxx"));
    try std.testing.expectEqual(diag.buf.len, diag.text().len);
}

test "parse: every field kind reads its value, and a repeated bool is idempotent" {
    const s = try parseTest(Sample, &.{
        "m",      "--count", "9",      "--small", "-8",      "--ratio", "1.5",    "-v", "--verbose",
        "--mode", "slow",    "--pick", "c",       "--limit", "5",       "--tags", "a",  "--tags",
        "b",      "--depth", "2",
    });
    try std.testing.expectEqualStrings("m", s.name.?);
    try std.testing.expectEqual(@as(u32, 9), s.count);
    try std.testing.expectEqual(@as(i8, -8), s.small);
    try std.testing.expectEqual(@as(f32, 1.5), s.ratio);
    try std.testing.expect(s.verbose);
    try std.testing.expectEqual(Mode.slow, s.mode);
    try std.testing.expectEqual(@as(?Pick, .c), s.pick);
    try std.testing.expectEqual(@as(?u64, 5), s.limit);
    try std.testing.expectEqual(@as(usize, 2), s.tags.len);
    try std.testing.expectEqual(@as(u16, 2), s.inner.depth);
}

test "parse: help, alias, and a value that starts with a dash" {
    try std.testing.expectError(error.Help, parseTest(Sample, &.{ "--count", "1", "-h" }));
    try std.testing.expectError(error.Help, parseTest(Sample, &.{"--help"}));
    const s = try parseTest(Sample, &.{ "-v", "--text", "-dash", "m" });
    try std.testing.expect(s.verbose);
    try std.testing.expectEqualStrings("-dash", s.text);
    try std.testing.expectEqualStrings("m", s.name.?);
}

test "parse: a refusal keeps the alias spelling and joins two choices with or" {
    const messages = [_]struct { argv: []const []const u8, want: []const u8 }{
        .{ .argv = &.{"-v=1"}, .want = "-v takes no value" },
        .{
            .argv = &.{ "--mode", "medium" },
            .want = "--mode 'medium' is not a valid value. Use fast or slow",
        },
    };
    var diag: Diag = .{};
    for (messages) |m| {
        try std.testing.expectError(error.UsageError, parse(Sample, m.argv, &diag));
        try std.testing.expectEqualStrings(m.want, diag.text());
    }
}

test "helpBody: a long label and a default past the margin move to their own lines" {
    const Wide = struct {
        label_past_the_text_column: u32 = 1,
        short: u32 = 2,

        pub const help: Help = .{ .usage = "wide", .about = "" };
        pub const docs = .{
            .label_past_the_text_column = Doc{ .text = "x" ** 60 },
            .short = Doc{ .text = "y" },
        };
    };
    const pad = " " ** 28;
    try std.testing.expectEqualStrings(
        "\n  --label-past-the-text-column N\n" ++
            pad ++ "x" ** 60 ++ "\n" ++
            pad ++ "[default: 1]\n" ++
            "  --short N" ++ " " ** 17 ++ "y   [default: 2]\n",
        comptime helpBody(Wide),
    );
}

test "helpBody: one line per flag, groups in first-flag order, defaults stated" {
    const body = comptime helpBody(Sample);
    try std.testing.expect(std.mem.startsWith(u8, body, "A test command.\n"));
    try std.testing.expect(std.mem.indexOf(u8, body, "--count N") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "a count   [default: 7]") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "-v, --verbose") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "--mode fast|slow") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "over two lines   [default: \"hi\"]") != null);
    try std.testing.expect(
        std.mem.indexOf(u8, body, "the depth the command states   [default: 4]") != null,
    );
    try std.testing.expect(std.mem.indexOf(u8, body, "--hidden") == null);
    // The ungrouped flags print first, then the inner group under its header.
    try std.testing.expect(
        std.mem.indexOf(u8, body, "--count").? < std.mem.indexOf(u8, body, "  inner\n").?,
    );
    try std.testing.expect(
        std.mem.indexOf(u8, body, "  inner\n").? < std.mem.indexOf(u8, body, "--depth").?,
    );
}
