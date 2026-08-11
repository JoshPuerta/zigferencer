//! The flag parser of every subcommand. A command states its flags as one
//! struct, and `parseInto` fills a value of it from argv. A field `snake_case`
//! is the flag `--snake-case`, and the field default is the flag default. The
//! same struct generates the flag lines of the `--help` text, so the parse and
//! the help cannot drift. The shape follows the `flags` module of TigerBeetle.
//!
//! The field types and their argv forms:
//!   - `bool`: a bare flag that sets true.
//!   - an integer or a float: one number. The type bounds the range, and the
//!     command checks a tighter bound after the parse when it has one.
//!   - `[]const u8`: one token, taken as it is.
//!   - an enum: one tag name.
//!   - a type with `parseFlag`, `flag_value`, and `flag_choices`: one token that
//!     the type parses. The two names feed the help and the refusal text.
//!   - `Repeated(T, n)`: a flag that can occur at most n times.
//!   - `?T`: the form of T. An absent flag leaves null.
//!   - a nested struct: its fields flatten into the flags of the command.
//!
//! A command type declares `help` (the usage line, the prose, the footer) and
//! `docs` (one `Doc` per flag). It can declare `positional`, the name of one
//! `?[]const u8` field that takes the one bare argument, and `omit`, nested
//! fields that the command does not take. A doc of the command replaces the doc
//! of a nested type for one field, so a shared knob keeps one name and can
//! state a default of its own command. A type can also name `docs_from`, a
//! type whose `docs` cover nested fields that carry no help of their own.
//!
//! Every error prints one `error:` line to stderr and returns
//! `error.UsageError`. `main` exits cleanly on that error, with no developer
//! stack trace. The full usage block belongs to `--help`. The CLI does not
//! reprint it after a flag slip.

const std = @import("std");
const Io = std.Io;
const style = @import("style.zig");

// -- types ------------------------------------------------------------------

/// The help text of one flag.
pub const Doc = struct {
    /// The description. A `\n` starts a continuation line under the text column.
    text: []const u8,
    /// The group header the flag prints under. The groups print in the order of
    /// their first flag.
    group: []const u8 = "",
    /// The value placeholder. Null derives it from the type.
    value: ?[]const u8 = null,
    /// The text of `[default: X]`. Null derives it from the field default, and
    /// an empty string prints no default.
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
    /// The text after the flag lines, such as the examples. It ends with a
    /// newline.
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
    buf: [256]u8 = undefined,
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
    var positional_seen = false;
    var seen: SeenSet(T) = .initEmpty();
    var i: usize = 0;
    // Each pass consumes at least one token, so the loop ends in argv.len passes.
    while (i < argv.len) {
        const token: []const u8 = argv[i];
        i += 1;
        if (std.mem.eql(u8, token, "--help") or std.mem.eql(u8, token, "-h")) return error.Help;
        if (std.mem.startsWith(u8, token, "-")) {
            if (!try setFlag(&.{T}, T, &out, token, argv, &i, diag, &seen)) {
                return failUnknown(T, token, diag);
            }
            continue;
        }
        if (comptime !@hasDecl(T, "positional")) {
            return diag.fail("unexpected argument '{s}'", .{token});
        }
        if (positional_seen) return diag.fail("unexpected extra argument '{s}'", .{token});
        @field(out, T.positional) = token;
        positional_seen = true;
    }
    std.debug.assert(i == argv.len);
    return out;
}

/// The refusal of a token that no flag takes. A known flag with `=value` gets
/// the spelling that works, because the parser reads the value as the next token.
fn failUnknown(comptime T: type, token: []const u8, diag: *Diag) error{UsageError} {
    const eq = std.mem.indexOfScalar(u8, token, '=') orelse return diag.fail("unknown flag '{s}'", .{token});
    const head = token[0..eq];
    inline for (comptime flatten(&.{T})) |e| {
        const hit = std.mem.eql(u8, head, e.name) or (e.doc.alias != null and std.mem.eql(u8, head, e.doc.alias.?));
        if (hit) {
            if (e.boolean) return diag.fail("{s} takes no value", .{head});
            return diag.fail("unknown flag '{s}'. Write '{s} {s}'", .{ token, head, token[eq + 1 ..] });
        }
    }
    return diag.fail("unknown flag '{s}'", .{token});
}

/// The whole `--help` text of `T` after the usage line, built at compile time.
pub fn helpBody(comptime T: type) []const u8 {
    comptime {
        validate(T);
        @setEvalBranchQuota(200_000);
        const entries = flatten(&.{T});
        var width: usize = 0;
        for (entries) |e| width = @max(width, e.label.len);
        width = @min(width + 2, label_column_max);
        var out: []const u8 = T.help.about;
        var groups: []const []const u8 = &.{};
        for (entries) |e| {
            if (contains(groups, e.doc.group)) continue;
            groups = groups ++ .{e.doc.group};
            out = out ++ "\n";
            if (e.doc.group.len > 0) out = out ++ "  " ++ e.doc.group ++ "\n";
            for (entries) |g| {
                if (std.mem.eql(u8, g.doc.group, e.doc.group)) out = out ++ flagLines(g, width);
            }
        }
        if (T.help.footer.len > 0) out = out ++ "\n" ++ T.help.footer;
        return out;
    }
}

/// Print the help of `T` to `file`: stdout when the user asked for it, stderr
/// when a usage error shows it. A failed write has nowhere to go.
pub fn printHelp(comptime T: type, io: Io, file: Io.File, p: style.Palette) void {
    const body = comptime helpBody(T);
    var buf: [1024]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(file, io, &buf);
    const w = &fw.interface;
    w.print("{s}usage:{s} " ++ T.help.usage ++ "\n\n", .{ p.bold, p.reset }) catch return;
    w.writeAll(body) catch return;
    w.flush() catch return;
}

/// Write `bytes` to stderr for usage text and diagnostics. The function discards a
/// failed write, like `std.debug.print`. A broken stderr must not turn a usage
/// line into a `WriteFailed` that the caller carries back up to `main`.
pub fn writeStderr(io: Io, bytes: []const u8) void {
    var buf: [512]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    fw.interface.writeAll(bytes) catch return;
    fw.interface.flush() catch {};
}

/// Print `error: <msg>` to stderr and return `error.UsageError`. This is the one
/// place that reports a bad invocation. The parser and the per-command
/// validation both call it, so flag errors and value-range errors read the same
/// everywhere.
pub fn fail(io: Io, comptime fmt: []const u8, fmtargs: anytype) error{UsageError} {
    var buf: [512]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    fw.interface.print("error: " ++ fmt ++ "\n", fmtargs) catch {};
    fw.interface.flush() catch {};
    return error.UsageError;
}

// -- helpers ----------------------------------------------------------------

/// The text column of the flag lines. A longer label puts its text on the next
/// line.
const label_column_max = 26;

/// The right margin of a flag line, after which a default moves down.
const line_width_max = 88;

const Kind = enum { boolean, int, float, string, enumeration, custom, repeated, nested };

fn kindOf(comptime F: type) Kind {
    return switch (@typeInfo(F)) {
        .bool => .boolean,
        .int => .int,
        .float => .float,
        .@"enum" => .enumeration,
        .pointer => if (F == []const u8) .string else @compileError("flag type " ++ @typeName(F) ++ " is not supported"),
        .@"struct", .@"union" => if (@hasDecl(F, "parseFlag"))
            .custom
        else if (@hasDecl(F, "Item") and @hasDecl(F, "capacity"))
            .repeated
        else if (@typeInfo(F) == .@"struct")
            .nested
        else
            @compileError("flag type " ++ @typeName(F) ++ " needs parseFlag"),
        else => @compileError("flag type " ++ @typeName(F) ++ " is not supported"),
    };
}

fn Base(comptime F: type) type {
    return switch (@typeInfo(F)) {
        .optional => |o| o.child,
        else => F,
    };
}

fn isNested(comptime F: type) bool {
    return @typeInfo(F) != .optional and kindOf(F) == .nested;
}

fn flagName(comptime field: []const u8) []const u8 {
    comptime {
        var name: [field.len]u8 = undefined;
        for (field, &name) |c, *n| n.* = if (c == '_') '-' else c;
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
/// restate a shared knob.
fn docFor(comptime scopes: []const type, comptime name: []const u8) Doc {
    for (scopes) |S| {
        if (@hasDecl(S, "docs") and @hasField(@TypeOf(S.docs), name)) return @field(S.docs, name);
        if (@hasDecl(S, "docs_from") and @hasField(@TypeOf(S.docs_from.docs), name)) return @field(S.docs_from.docs, name);
    }
    @compileError("flag field '" ++ name ++ "' of " ++ @typeName(scopes[scopes.len - 1]) ++ " has no doc");
}

const Entry = struct {
    name: []const u8,
    label: []const u8,
    doc: Doc,
    default: []const u8,
    boolean: bool,
};

/// Every flag of the last type of `scopes`, nested fields flattened, in field
/// order.
fn flatten(comptime scopes: []const type) []const Entry {
    comptime {
        const Top = scopes[0];
        const Container = scopes[scopes.len - 1];
        var out: []const Entry = &.{};
        for (std.meta.fields(Container)) |f| {
            if (isPositional(Container, f.name)) continue;
            if (isNested(f.type)) {
                out = out ++ flatten(scopes ++ .{f.type});
                continue;
            }
            if (omitted(Top, f.name)) continue;
            const doc = docFor(scopes, f.name);
            const value = doc.value orelse valueName(Base(f.type));
            const flag = flagName(f.name);
            const spelled = if (doc.alias) |a| a ++ ", " ++ flag else flag;
            const label = if (value.len > 0) spelled ++ " " ++ value else spelled;
            out = out ++ .{Entry{
                .name = flag,
                .label = label,
                .doc = doc,
                .default = doc.default orelse defaultText(f),
                .boolean = kindOf(Base(f.type)) == .boolean,
            }};
        }
        return out;
    }
}

fn valueName(comptime F: type) []const u8 {
    return switch (kindOf(F)) {
        .boolean => "",
        .int => "N",
        .float => "X",
        .string => "TEXT",
        .enumeration => tagList(F, "|"),
        .custom => F.flag_value,
        .repeated => valueName(F.Item),
        .nested => unreachable,
    };
}

fn tagList(comptime E: type, comptime sep: []const u8) []const u8 {
    comptime {
        var out: []const u8 = "";
        for (std.meta.fields(E), 0..) |f, i| out = out ++ (if (i == 0) "" else sep) ++ f.name;
        return out;
    }
}

/// The words of the refusal of a bad enum value: "a, b, or c".
fn tagChoices(comptime E: type) []const u8 {
    comptime {
        const fields = std.meta.fields(E);
        var out: []const u8 = "";
        for (fields, 0..) |f, i| {
            const sep = if (i == 0) "" else if (i + 1 == fields.len) (if (fields.len == 2) " or " else ", or ") else ", ";
            out = out ++ sep ++ f.name;
        }
        return out;
    }
}

fn defaultText(comptime f: std.builtin.Type.StructField) []const u8 {
    const value = f.defaultValue() orelse @compileError("flag field '" ++ f.name ++ "' needs a default");
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

fn flagLines(comptime e: Entry, comptime width: usize) []const u8 {
    comptime {
        const pad = " " ** (width + 2);
        const last = if (std.mem.lastIndexOfScalar(u8, e.doc.text, '\n')) |k| e.doc.text.len - k - 1 else e.doc.text.len;
        const head = if (std.mem.indexOfScalar(u8, e.default, '\n')) |k| k else e.default.len;
        // A default that runs past the right margin starts its own line.
        const joint = if (width + 2 + last + 3 + head + 11 > line_width_max) "\n" else "   ";
        const tail = if (e.default.len > 0) joint ++ "[default: " ++ e.default ++ "]" else "";
        const text = e.doc.text ++ tail;
        var out: []const u8 = "  " ++ e.label;
        if (e.label.len + 2 > width) {
            out = out ++ "\n" ++ pad;
        } else {
            out = out ++ " " ** (width - e.label.len);
        }
        var lines = std.mem.splitScalar(u8, text, '\n');
        var first = true;
        while (lines.next()) |line| {
            if (!first) out = out ++ "\n" ++ pad;
            out = out ++ line;
            first = false;
        }
        return out ++ "\n";
    }
}

/// The compile-time checks of a command type: unique flag names, one doc per
/// flag and no doc for a missing flag, a real positional field, and real omit
/// names.
fn validate(comptime T: type) void {
    comptime {
        @setEvalBranchQuota(200_000);
        if (!@hasDecl(T, "help")) @compileError(@typeName(T) ++ " needs a help declaration");
        const entries = flatten(&.{T});
        var names: []const []const u8 = &.{};
        for (entries) |e| {
            if (std.mem.eql(u8, e.name, "--help")) @compileError("--help is reserved");
            if (contains(names, e.name)) @compileError("flag " ++ e.name ++ " repeats in " ++ @typeName(T));
            names = names ++ .{e.name};
            if (e.doc.alias) |a| {
                if (contains(names, a)) @compileError("flag " ++ a ++ " repeats in " ++ @typeName(T));
                names = names ++ .{a};
            }
        }
        validateDocs(T);
        if (@hasDecl(T, "omit")) {
            for (T.omit) |o| {
                if (!hasFlatField(T, o)) @compileError("omit '" ++ o ++ "' of " ++ @typeName(T) ++ " names no field");
            }
        }
        if (@hasDecl(T, "positional")) {
            if (!@hasField(T, T.positional)) @compileError("positional '" ++ T.positional ++ "' names no field");
            if (@FieldType(T, T.positional) != ?[]const u8) @compileError("positional field must be ?[]const u8");
        }
    }
}

/// Every doc of `Container` and of its nested types names one of their fields.
fn validateDocs(comptime Container: type) void {
    if (@hasDecl(Container, "docs")) {
        for (std.meta.fields(@TypeOf(Container.docs))) |d| {
            if (!hasFlatField(Container, d.name)) @compileError("doc '" ++ d.name ++ "' of " ++ @typeName(Container) ++ " names no field");
        }
    }
    for (std.meta.fields(Container)) |f| {
        if (isNested(f.type)) validateDocs(f.type);
    }
}

fn hasFlatField(comptime Container: type, comptime name: []const u8) bool {
    for (std.meta.fields(Container)) |f| {
        if (std.mem.eql(u8, f.name, name)) return true;
        if (isNested(f.type) and hasFlatField(f.type, name)) return true;
    }
    return false;
}

/// Set the field whose flag is `token`. False means no field of `Container`
/// takes it.
fn setFlag(
    comptime scopes: []const type,
    comptime Container: type,
    out: *Container,
    token: []const u8,
    argv: anytype,
    i: *usize,
    diag: *Diag,
    seen: anytype,
) error{UsageError}!bool {
    inline for (comptime std.meta.fields(Container)) |f| {
        if (comptime isPositional(Container, f.name)) continue;
        if (comptime isNested(f.type)) {
            if (try setFlag(scopes ++ .{f.type}, f.type, &@field(out, f.name), token, argv, i, diag, seen)) return true;
        } else if (comptime !omitted(scopes[0], f.name)) {
            const doc = comptime docFor(scopes, f.name);
            const name = comptime flagName(f.name);
            const hit = std.mem.eql(u8, token, name) or (doc.alias != null and std.mem.eql(u8, token, doc.alias.?));
            if (hit) {
                // Without this check a repeated scalar flag keeps the last value in silence. A bool is idempotent.
                if (comptime kindOf(Base(f.type)) != .repeated and kindOf(Base(f.type)) != .boolean) {
                    const slot = comptime entryIndex(scopes[0], name);
                    if (seen.isSet(slot)) return diag.fail(name ++ " given twice", .{});
                    seen.set(slot);
                }
                try setValue(f.type, name, &@field(out, f.name), argv, i, diag);
                return true;
            }
        }
    }
    return false;
}

/// One bit per flag of `Top`, set when the flag has been given.
fn SeenSet(comptime Top: type) type {
    @setEvalBranchQuota(200_000);
    return std.StaticBitSet(flatten(&.{Top}).len);
}

/// The position of the flag `name` in the flag list of `Top`.
fn entryIndex(comptime Top: type, comptime name: []const u8) usize {
    for (flatten(&.{Top}), 0..) |e, k| {
        if (std.mem.eql(u8, e.name, name)) return k;
    }
    @compileError("flag " ++ name ++ " is not in " ++ @typeName(Top));
}

fn setValue(comptime F: type, comptime name: []const u8, slot: *F, argv: anytype, i: *usize, diag: *Diag) error{UsageError}!void {
    const B = Base(F);
    if (comptime kindOf(B) == .boolean) {
        slot.* = true;
        return;
    }
    if (i.* >= argv.len) return diag.fail(name ++ " requires a value", .{});
    const raw: []const u8 = argv[i.*];
    i.* += 1;
    if (comptime kindOf(B) == .repeated) {
        if (slot.len >= B.capacity) return diag.fail(name ++ " accepts at most {d} values", .{B.capacity});
        slot.items[slot.len] = try parseValue(B.Item, name, raw, diag);
        slot.len += 1;
        return;
    }
    slot.* = try parseValue(B, name, raw, diag);
}

fn parseValue(comptime B: type, comptime name: []const u8, raw: []const u8, diag: *Diag) error{UsageError}!B {
    return switch (comptime kindOf(B)) {
        .int => std.fmt.parseInt(B, raw, 10) catch |err| switch (err) {
            error.Overflow => diag.fail(name ++ ": '{s}' is out of range", .{raw}),
            error.InvalidCharacter => diag.fail(name ++ ": '{s}' is not a number", .{raw}),
        },
        .float => std.fmt.parseFloat(B, raw) catch diag.fail(name ++ ": '{s}' is not a number", .{raw}),
        .string => raw,
        .enumeration => std.meta.stringToEnum(B, raw) orelse
            diag.fail(name ++ " '{s}' is not a valid value. Use " ++ comptime tagChoices(B), .{raw}),
        .custom => B.parseFlag(raw) orelse
            diag.fail(name ++ " '{s}' is not a valid value. Use " ++ B.flag_choices, .{raw}),
        .boolean, .repeated, .nested => unreachable,
    };
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
    try std.testing.expectError(error.UsageError, parse(Sample, &[_][]const u8{ "--count", "x" }, &diag));
    try std.testing.expectEqualStrings("--count: 'x' is not a number", diag.text());
    try std.testing.expectError(error.UsageError, parse(Sample, &[_][]const u8{ "--pick", "d" }, &diag));
    try std.testing.expectEqualStrings("--pick 'd' is not a valid value. Use a, b, or c", diag.text());
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

test "helpBody: one line per flag, groups in first-flag order, defaults stated" {
    const body = comptime helpBody(Sample);
    try std.testing.expect(std.mem.startsWith(u8, body, "A test command.\n"));
    try std.testing.expect(std.mem.indexOf(u8, body, "--count N") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "a count   [default: 7]") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "-v, --verbose") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "--mode fast|slow") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "over two lines   [default: \"hi\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "the depth the command states   [default: 4]") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "--hidden") == null);
    // The ungrouped flags print first, then the inner group under its header.
    try std.testing.expect(std.mem.indexOf(u8, body, "--count").? < std.mem.indexOf(u8, body, "  inner\n").?);
    try std.testing.expect(std.mem.indexOf(u8, body, "  inner\n").? < std.mem.indexOf(u8, body, "--depth").?);
}
