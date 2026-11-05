//! Command dispatch: argv to a command handler, the one entry point from `main.zig`.
//! Each command is a `{ name, summary, handler }` row of a table. `zigferencer` dispatches
//! `release`, and `zigferencer-dev` dispatches `release` plus the table of
//! `src/dev/dev_command.zig`. The release binary names the dev commands only as strings, because
//! its module holds no path to a dev file. The usage screen reads the same `summary`, so help and
//! dispatch cannot drift.
//!
//! Help-text dialect, the same in every `--help`. `args.helpBody` generates the flag lines:
//!   - A `summary` is an imperative fragment of 55 characters or less.
//!   - A `--help` prints `usage:`, one sentence, the flags in groups under short lowercase headers
//!     (for example tuning, kv, gpu), then `examples:`.
//!   - A default reads as a trailing `[default: X]`, never as inline prose.
//!   - One qualifier per flag line.
//!   - A backend caveat goes in a group header (`gpu (fused-GPU build only)`), not on a line.
//!   - A shared option keeps one name and one wording, because
//!     `flag_groups.EngineFlags` and `flag_groups.ServeFlags` declare it once.
//!   - A flag that the build cannot honor stays listed, and the command rejects it at use.

const std = @import("std");
const Io = std.Io;

const fetch_cmd = @import("fetch_cmd.zig");
const list_cmd = @import("list_cmd.zig");
const inspect_cmd = @import("inspect_cmd.zig");
const run_cmd = @import("run_cmd.zig");
const serve_cmd = @import("serve_cmd.zig");
const serve_openai_cmd = @import("serve_openai_cmd.zig");
const info_cmd = @import("info/info_cmd.zig");
const client_cmd = @import("client/client_cmd.zig");
const style = @import("style.zig");
const context = @import("context.zig");
const registry = @import("../core/registry.zig");
const limits = @import("../core/limits.zig");
const prompt_cache = @import("../runtime/prompt_cache.zig");

comptime {
    // A name in both tables makes the dispatch order pick the handler.
    for (release) |r| {
        for (dev_names) |d| {
            if (std.mem.eql(u8, r.name, d)) {
                @compileError("command '" ++ d ++ "' sits in both tables");
            }
        }
    }
}

// -- constants --------------------------------------------------------------

/// The stack buffer of one error message on stderr.
const error_buffer_bytes = 256;

/// Spaces for the name column of the usage screen. The widest command name must fit.
const pad = " " ** 32;

/// The commands of the release binary.
pub const release = [_]Subcommand{
    .{ .name = "list", .summary = "List registered models (and which are on disk)", .handler = list_cmd.run },
    .{ .name = "fetch", .summary = "Download a model (registry name or HF path)", .handler = fetch_cmd.run },
    .{ .name = "inspect", .summary = "Print a downloaded GGUF's metadata", .handler = inspect_cmd.run },
    .{ .name = "run", .summary = "Generate text from a prompt (greedy)", .handler = run_cmd.run },
    .{ .name = "serve", .summary = "Serve sessions over TCP/unix, batched decode", .handler = serve_cmd.run },
    .{ .name = "serve-openai", .summary = "Serve an OpenAI-compatible HTTP API", .handler = serve_openai_cmd.run },
    .{ .name = "client", .summary = "Chat with a running server over the binary wire", .handler = client_cmd.run },
    .{ .name = "info", .summary = "Print the build, host card, and support matrix", .handler = info_cmd.run },
};

/// The dev command names as strings, for the hint of the release binary. This module cannot read
/// the dev table, so the list repeats the names. `dev_command.zig` checks the two lists.
pub const dev_names = [_][]const u8{
    "bench-model",
    "bench-serve",
    "bench-swap",
    "bench-decode-host",
    "gpu-selftest",
    "tokens",
    "serve-demo",
};

// -- types ------------------------------------------------------------------

// The handlers hold different error sets, and one function-pointer table shares one return type.
pub const Handler = *const fn (context.Context) anyerror!void;

pub const Subcommand = struct {
    name: []const u8,
    summary: []const u8,
    handler: Handler,
};

// -- api --------------------------------------------------------------------

/// Dispatch argv over the release commands.
pub fn dispatchRelease(init: std.process.Init, stdout: *std.Io.File.Writer) !void {
    return dispatchOver(init, stdout, &release, "zigferencer");
}

/// True when `arg` asks for the top-level usage screen.
pub fn isHelpWord(arg: []const u8) bool {
    const words = [_][]const u8{ "--help", "-h", "help" };
    for (words) |w| {
        if (std.mem.eql(u8, arg, w)) return true;
    }

    return false;
}

/// Resolve `args[1]` against `table` and run its handler.
pub fn dispatchOver(
    init: std.process.Init,
    stdout: *std.Io.File.Writer,
    comptime table: []const Subcommand,
    comptime binary_name: []const u8,
) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    std.debug.assert(args.len >= 1);

    const no_color = init.environ_map.get("NO_COLOR") != null;
    const color = style.stdoutColor(init.io, no_color);
    const color_err = style.stderrColor(init.io, no_color);

    if (args.len < 2) {
        printUsage(init.io, .stderr(), style.Palette.init(color_err), table, binary_name);
        return;
    }

    // Help answers before any environment check, so a bad variable cannot hide it.
    if (isHelpWord(args[1])) {
        printUsage(init.io, .stdout(), style.Palette.init(color), table, binary_name);
        return;
    }

    const models_root = registry.resolveRoot(init.environ_map.get(registry.models_dir_env));
    const cache_dir = prompt_cache.resolveCacheDir(
        init.environ_map.get(prompt_cache.cache_dir_env),
    );
    const vk_storage_range = try resolveStorageRange(init);

    const sub = args[1];
    for (table) |s| {
        if (std.mem.eql(u8, s.name, sub)) {
            return s.handler(.{
                .io = init.io,
                .arena = arena,
                .gpa = init.gpa,
                .args = args[2..],
                .stdout = stdout,
                .models_root = models_root,
                .cache_dir = cache_dir,
                .vk_storage_range = vk_storage_range,
                .color = color,
                .color_err = color_err,
            });
        }
    }

    // The dev binary holds every dev name, so only the release binary reaches the hint.
    for (dev_names) |name| {
        if (std.mem.eql(u8, name, sub)) {
            printDevCommand(init.io, sub);
            return error.UnknownSubcommand;
        }
    }

    printUnknownSubcommand(init.io, sub, binary_name);
    return error.UnknownSubcommand;
}

// -- helpers ----------------------------------------------------------------

fn resolveStorageRange(init: std.process.Init) !?u64 {
    const text = init.environ_map.get(limits.storage_range_env) orelse return null;
    return limits.parseStorageRange(text) catch |err| {
        printBadStorageRange(init.io, text);
        return err;
    };
}

// The printers below discard a failed write, like `args.writeStderr`, because the error has
// nowhere to go.
fn printUsage(
    io: Io,
    file: Io.File,
    p: style.Palette,
    comptime table: []const Subcommand,
    comptime binary_name: []const u8,
) void {
    var usage_buffer: [context.stdout_buffer_bytes]u8 = undefined;
    var usage_writer: Io.File.Writer = .initStreaming(file, io, &usage_buffer);
    const w = &usage_writer.interface;

    w.print(
        "{s}" ++ binary_name ++ "{s}: a small LLM inference engine (Zig 0.16)\n\n",
        .{ p.bold, p.reset },
    ) catch return;
    w.print(
        "{s}usage:{s} " ++ binary_name ++ " <command> [args]\n\n",
        .{ p.bold, p.reset },
    ) catch return;
    w.print("{s}commands:{s}\n", .{ p.bold, p.reset }) catch return;

    const width = comptime nameWidth(table);
    for (table) |s| {
        w.print(
            "  {s}{s}{s}{s} {s}\n",
            .{ p.green, s.name, p.reset, pad[0 .. width - s.name.len], s.summary },
        ) catch return;
    }

    w.print(
        "\n{s}'" ++ binary_name ++ " <command> --help' lists a command's flags.{s}\n",
        .{ p.dim, p.reset },
    ) catch return;
    w.flush() catch {};
}

/// The longest command name of `table`, the width of the name column.
fn nameWidth(comptime table: []const Subcommand) usize {
    var width: usize = 0;
    for (table) |s| {
        width = @max(width, s.name.len);
    }

    std.debug.assert(width <= pad.len);
    return width;
}

fn printBadStorageRange(io: Io, text: []const u8) void {
    var stderr_buffer: [error_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    w.print(
        "error: " ++ limits.storage_range_env ++ " must be a byte count in 1..{d}. The " ++
            "value is '{s}'\n",
        .{ limits.storage_range_max, text },
    ) catch return;
    w.flush() catch {};
}

fn printDevCommand(io: Io, sub: []const u8) void {
    var stderr_buffer: [error_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    w.print(
        "error: '{s}' is a developer command. Run it with zigferencer-dev.\n",
        .{sub},
    ) catch return;
    w.flush() catch {};
}

fn printUnknownSubcommand(io: Io, sub: []const u8, comptime binary_name: []const u8) void {
    var stderr_buffer: [error_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    w.print("error: unknown command '{s}'\n", .{sub}) catch return;
    w.writeAll("run '" ++ binary_name ++ "' with no arguments for usage\n") catch return;
    w.flush() catch {};
}

// -- tests ------------------------------------------------------------------

test "isHelpWord: the three help spellings and nothing else" {
    const cases = [_]struct { arg: []const u8, want: bool }{
        .{ .arg = "--help", .want = true },
        .{ .arg = "-h", .want = true },
        .{ .arg = "help", .want = true },
        .{ .arg = "list", .want = false },
        .{ .arg = "--helpme", .want = false },
        .{ .arg = "", .want = false },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c.want, isHelpWord(c.arg));
    }
}
