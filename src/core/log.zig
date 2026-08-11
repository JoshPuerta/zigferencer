//! Process logging: `std.log` routed to stderr, gated at runtime by
//! `$ZIGFERENCER_LOG`. Quiet, louder, or per-subsystem is one env var away,
//! with no rebuild. Diagnostics only: generated text stays on stdout.
//!
//! The comptime ceiling (`std_options.log_level` in `main.zig`) is `debug` in
//! *every* build, so this module is the only gate and a release binary stays
//! verbose on demand. Flow traces sit at boundaries, never per element, and
//! the gate runs only for messages that cleared the comptime ceiling. Keep
//! log-arg tuples to already-computed scalars: the args expression evaluates
//! before the gate. The three hot scopes (`layer`, `dispatch`, `token`) are
//! the exception. Their call sites sit inside `if (comptime hot)`, so
//! `-Dtrace` decides at build time whether they exist at all.
//!
//! Use `info` for the milestones a user waits through, `debug` for the rest.
//! Each line opens with `+{ms} `, the time since startup:
//!
//!     const log = std.log.scoped(.gguf);
//!     log.info("loaded {d} tensors in {d:.0} ms", .{ n, ms });
//!
//! `$ZIGFERENCER_LOG` grammar (comma-separated terms, left to right):
//!
//!     debug | info | warn | err | off   the global ceiling, as a bare level
//!     gguf                   scope `gguf` at debug. A value with no bare level
//!                            drops the global to `off` ("only these")
//!     gguf=info | gguf=off   scope `gguf` at an explicit ceiling, or silenced
//!     info,parallel=debug    everyone at info, plus `parallel` at debug
//!     debug,parallel=off     everyone at debug, except the noisy `parallel`
//!
//! An unknown level word reads as a scope name. A mistyped `scope=level` term
//! is skipped, so the default stands and the run does not fail.

const std = @import("std");
const build_options = @import("build_options");

// -- types ------------------------------------------------------------------

/// A per-scope ceiling. `level == null` means the scope is silenced (`=off`).
const ScopeCeiling = struct { name: []const u8, level: ?std.log.Level };

const LevelSpec = union(enum) { off, level: std.log.Level, unknown };

// -- state ------------------------------------------------------------------

/// Bound on distinct scopes nameable in one `$ZIGFERENCER_LOG`. The engine
/// defines 16 scopes. This leaves headroom without unbounding the surface.
const max_log_scopes = 24;

/// Per-scope ceilings parsed from the env once at startup. `name` borrows the env
/// string, which lives for the process, so no copy. Bounded by `max_log_scopes`.
var scope_ceilings: [max_log_scopes]ScopeCeiling = undefined;
var scope_count: usize = 0;

/// Global ceiling for any scope not named in the table. `null` = silent. Default
/// `.info` preserves the behavior of an unset or bare-level env var.
var threshold: ?std.log.Level = .info;

/// The raw `$ZIGFERENCER_LOG` as given, kept so `zigferencer info` can echo what
/// the user actually set (not the parsed interpretation). `null` = unset.
var raw_env: ?[]const u8 = null;

/// Process start, for the `+{ms}` line prefix. `setFromEnv` anchors it in
/// `main` before any thread exists. The lazy capture in `elapsedNs` covers
/// roots that never call `setFromEnv` (the bench and probe tools).
var start: ?std.Io.Timestamp = null;

// -- api --------------------------------------------------------------------

/// Comptime gate for hot-path traces (the `layer`, `dispatch`, and `token`
/// scopes). `-Dtrace` sets it: on in Debug, off in release unless forced. A
/// call site wraps the whole trace, including any clock read, in
/// `if (comptime hot)`, so a release build compiles the site out entirely.
pub const hot = build_options.trace_hot;

/// Monotonic timestamp from the diagnostics clock. `std.Options.debug_io` is
/// the singleton `std.log.defaultLog` writes stderr through, independent of
/// the application `Io`. Public so a hot-trace site with no `io` in reach
/// (the fused runners) can time itself.
pub fn now() std.Io.Timestamp {
    return std.Io.Clock.now(.awake, std.Options.debug_io);
}

/// Milliseconds from `t0` to now, for hot-trace args.
pub fn msSince(t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(now().nanoseconds - t0.nanoseconds)) / 1e6;
}

/// Apply `$ZIGFERENCER_LOG`. An unset value keeps the defaults. An unknown
/// level word reads as a scope name, and an unknown `scope=level` term is
/// skipped, so a mistake never fails a run.
pub fn setFromEnv(value: ?[]const u8) void {
    if (start == null) start = now();
    const v = value orelse return;
    raw_env = v;
    scope_count = 0;
    threshold = .info; // Re-establish the default. The terms below can override it.
    var saw_scope = false;
    var saw_level = false;

    var it = std.mem.tokenizeScalar(u8, v, ',');
    while (it.next()) |term| {
        std.debug.assert(term.len > 0); // tokenizeScalar drops empties. Defend anyway.
        if (std.mem.indexOfScalar(u8, term, '=')) |eq_at| {
            const name = term[0..eq_at];
            if (name.len == 0) continue; // "=debug" malformed -> skip
            switch (parseLevel(term[eq_at + 1 ..])) {
                .off => {
                    addScope(name, null);
                    saw_scope = true;
                },
                .level => |l| {
                    addScope(name, l);
                    saw_scope = true;
                },
                .unknown => {}, // mistyped level -> skip the term
            }
        } else switch (parseLevel(term)) {
            .off => {
                threshold = null;
                saw_level = true;
            },
            .level => |l| {
                threshold = l;
                saw_level = true;
            },
            .unknown => {
                addScope(term, .debug); // a bare word is a scope name, at debug
                saw_scope = true;
            },
        }
    }
    // Scopes named without an explicit global level mean "only these": quiet the rest.
    if (saw_scope and !saw_level) threshold = null;
    std.debug.assert(scope_count <= max_log_scopes);
}

/// `std.options.logFn`: drop messages above the scope's runtime ceiling,
/// otherwise write the dim `+{ms}` prefix and defer to std's colored stderr
/// formatter (`LEVEL(scope): message`). The lock-and-protect body mirrors
/// `std.log.defaultLog`. `err`/`warn` are the lowest enum values, so a higher
/// `@intFromEnum` means more verbose.
pub fn logFn(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    const ceiling = ceilingFor(scope) orelse return;
    if (@intFromEnum(level) > @intFromEnum(ceiling)) return;
    const ns = elapsedNs();
    const io = std.Options.debug_io;
    const prev = io.swapCancelProtection(.blocked);
    defer _ = io.swapCancelProtection(prev);
    var buffer: [64]u8 = undefined;
    const stderr = std.debug.lockStderr(&buffer).terminal();
    defer std.debug.unlockStderr();
    var prefix_buf: [24]u8 = undefined;
    stderr.setColor(.dim) catch {};
    stderr.writer.writeAll(prefixText(&prefix_buf, ns)) catch {};
    stderr.setColor(.reset) catch {};
    std.log.defaultLogFileTerminal(level, scope, format, args, stderr) catch {};
}

/// Runtime pre-check for a hot-trace site whose args need a clock read: skip
/// the read when the scope cannot pass the gate anyway. Flow traces do not
/// need this. Their args are precomputed scalars (see the header).
pub fn enabled(comptime scope: @EnumLiteral(), level: std.log.Level) bool {
    const ceiling = ceilingFor(scope) orelse return false;
    return @intFromEnum(level) <= @intFromEnum(ceiling);
}

/// The raw `$ZIGFERENCER_LOG` value, for `zigferencer info` to report verbatim.
pub fn rawEnv() ?[]const u8 {
    return raw_env;
}

// -- helpers ----------------------------------------------------------------

fn elapsedNs() u64 {
    const t0 = start orelse blk: {
        const t = now();
        start = t;
        break :blk t;
    };
    return @intCast(now().nanoseconds - t0.nanoseconds);
}

/// Render the line prefix into `buf`: `+` then right-aligned milliseconds with
/// three decimals, then `ms `. It is pure, so the format has a test.
fn prefixText(buf: []u8, ns: u64) []const u8 {
    const ms = @as(f64, @floatFromInt(ns)) / 1e6;
    return std.fmt.bufPrint(buf, "+{d:>9.3}ms ", .{ms}) catch buf[0..0];
}

/// Map a level word (with the err/error, warn/warning aliases) to a spec. An
/// unrecognized word is `.unknown`. The caller decides whether that means "a
/// scope name" (bare term) or "skip this term" (`scope=level` form).
fn parseLevel(word: []const u8) LevelSpec {
    if (std.mem.eql(u8, word, "off")) return .off;
    if (std.mem.eql(u8, word, "err") or std.mem.eql(u8, word, "error")) return .{ .level = .err };
    if (std.mem.eql(u8, word, "warn") or std.mem.eql(u8, word, "warning")) return .{ .level = .warn };
    if (std.mem.eql(u8, word, "info")) return .{ .level = .info };
    if (std.mem.eql(u8, word, "debug")) return .{ .level = .debug };
    return .unknown;
}

fn addScope(name: []const u8, level: ?std.log.Level) void {
    std.debug.assert(name.len > 0);
    if (scope_count >= max_log_scopes) return; // bounded: a term past the cap drops
    std.debug.assert(scope_count < max_log_scopes);
    scope_ceilings[scope_count] = .{ .name = name, .level = level };
    scope_count += 1;
}

/// Resolve the runtime ceiling for `scope`: its per-scope entry when the env
/// names it, else the global threshold. `@tagName(scope)` is comptime, so the
/// only runtime work is the bounded scan, and only for messages that already
/// cleared the comptime ceiling.
fn ceilingFor(comptime scope: @EnumLiteral()) ?std.log.Level {
    std.debug.assert(scope_count <= max_log_scopes);
    const name = @tagName(scope);
    var i: usize = 0;
    while (i < scope_count) : (i += 1) {
        if (std.mem.eql(u8, scope_ceilings[i].name, name)) return scope_ceilings[i].level;
    }
    return threshold;
}

// -- tests ------------------------------------------------------------------

const Level = ?std.log.Level;

const EnvCase = struct {
    env: []const u8,
    threshold: Level,
    scopes: usize,
    /// The expected ceiling of three probe scopes.
    gguf: Level,
    run: Level,
    parallel: Level,
};

/// One env value gives one global threshold and one ceiling per scope, and
/// `enabled` passes a level exactly when it does not exceed that ceiling.
fn checkEnv(c: EnvCase) !void {
    setFromEnv(c.env);
    try std.testing.expectEqual(c.threshold, threshold);
    try std.testing.expectEqual(c.scopes, scope_count);
    try std.testing.expectEqual(c.gguf, ceilingFor(.gguf));
    try std.testing.expectEqual(c.run, ceilingFor(.run));
    try std.testing.expectEqual(c.parallel, ceilingFor(.parallel));
    for (std.enums.values(std.log.Level)) |level| {
        for ([_]Level{ c.gguf, c.run, c.parallel }, 0..) |want_ceiling, k| {
            const want = if (want_ceiling) |ceiling| @intFromEnum(level) <= @intFromEnum(ceiling) else false;
            const got = switch (k) {
                0 => enabled(.gguf, level),
                1 => enabled(.run, level),
                else => enabled(.parallel, level),
            };
            try std.testing.expectEqual(want, got);
        }
    }
}

test "setFromEnv: the grammar gives the threshold and the scope ceilings" {
    const cases = [_]EnvCase{
        // A bare level is the global threshold.
        .{ .env = "debug", .threshold = .debug, .scopes = 0, .gguf = .debug, .run = .debug, .parallel = .debug },
        .{ .env = "off", .threshold = null, .scopes = 0, .gguf = null, .run = null, .parallel = null },
        .{ .env = "info", .threshold = .info, .scopes = 0, .gguf = .info, .run = .info, .parallel = .info },
        // A bare scope silences the rest.
        .{ .env = "gguf", .threshold = null, .scopes = 1, .gguf = .debug, .run = null, .parallel = null },
        // A scope=level term overrides the global level for that scope.
        .{ .env = "info,parallel=debug", .threshold = .info, .scopes = 1, .gguf = .info, .run = .info, .parallel = .debug },
        .{ .env = "debug,parallel=off", .threshold = .debug, .scopes = 1, .gguf = .debug, .run = .debug, .parallel = null },
        // A mistyped level skips the term, and the default stands.
        .{ .env = "gguf=verbose", .threshold = .info, .scopes = 0, .gguf = .info, .run = .info, .parallel = .info },
    };
    for (cases) |c| {
        errdefer std.debug.print("env case: {s}\n", .{c.env});
        try checkEnv(c);
    }
}

test "setFromEnv: the scope table is bounded" {
    const many = comptime blk: {
        var s: []const u8 = "";
        var i: usize = 0;
        while (i < max_log_scopes + 5) : (i += 1) s = s ++ std.fmt.comptimePrint("s{d},", .{i});
        break :blk s;
    };
    setFromEnv(many);
    try std.testing.expectEqual(@as(usize, max_log_scopes), scope_count);
}

test "prefixText: fixed-width milliseconds with three decimals" {
    var buf: [24]u8 = undefined;
    try std.testing.expectEqualStrings("+    0.000ms ", prefixText(&buf, 0));
    try std.testing.expectEqualStrings("+    1.235ms ", prefixText(&buf, 1_234_567));
    try std.testing.expectEqualStrings("+12345.679ms ", prefixText(&buf, 12_345_678_900));
}
