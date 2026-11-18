//! Command dispatch: argv to a command handler, the entry point from `main.zig`.
//!
//! Comptime. The command tables are comptime parameters, and a check rejects a command name in
//! both tables.
//!
//! Tables. Each command is a `{ name, summary, handler }` entry. `zigferencer` dispatches
//! `release`. `zigferencer-dev` dispatches `release` plus the table of `src/dev/dev_command.zig`.
//! The usage screen reads the same `summary`, so help and dispatch cannot drift.
//!
//! Dev names. The release binary names the dev commands only as strings, because its module holds
//! no path to a dev file. `dev_command.zig` checks that the strings match its table.

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
    entry("list", "List registered models (and which are on disk)", list_cmd.run),
    entry("fetch", "Download a model (registry name or HF path)", fetch_cmd.run),
    entry("inspect", "Print a downloaded GGUF's metadata", inspect_cmd.run),
    entry("run", "Generate text from a prompt (greedy)", run_cmd.run),
    entry("serve", "Serve sessions over TCP/unix, batched decode", serve_cmd.run),
    entry("serve-openai", "Serve an OpenAI-compatible HTTP API", serve_openai_cmd.run),
    entry("client", "Chat with a running server over the binary wire", client_cmd.run),
    entry("info", "Print the build, host card, and support matrix", info_cmd.run),
};

/// The dev command names as strings, for the hint of the release binary.
pub const dev_names = [_][]const u8{
    "bench-model",
    "bench-serve",
    "bench-swap",
    "bench-decode-host",
    "gpu-selftest",
    "tokens",
    "serve-demo",
    "dispatch-list",
    "compare-logits",
};

// -- types ------------------------------------------------------------------

// The handlers hold different error sets, so one function-pointer table needs `anyerror`.
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
    // Read argv and the color choice.
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

    // Resolve the environment that every handler reads.
    const models_root = registry.resolveRoot(init.environ_map.get(registry.models_dir_env));
    const cache_dir = prompt_cache.resolveCacheDir(
        init.environ_map.get(prompt_cache.cache_dir_env),
    );
    const vk_options: limits.DeviceOptions = .{
        .storage_range_override = try resolveStorageRange(init),
        .software = try resolveSoftwareDevice(init),
    };

    // Run the handler that matches the name.
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
                .vk_options = vk_options,
                .color = color,
                .color_err = color_err,
            });
        }
    }

    // Refuse the name. Only the release binary reaches the dev hint, because the dev table holds
    // every dev name.
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

fn entry(comptime name: []const u8, comptime summary: []const u8, handler: Handler) Subcommand {
    return .{ .name = name, .summary = summary, .handler = handler };
}

fn resolveStorageRange(init: std.process.Init) !?u64 {
    const text = init.environ_map.get(limits.storage_range_env) orelse return null;
    return limits.parseStorageRange(text) catch |err| {
        printBadStorageRange(init.io, text);
        return err;
    };
}

fn resolveSoftwareDevice(init: std.process.Init) !bool {
    const text = init.environ_map.get(limits.software_device_env) orelse return false;
    return limits.parseSoftwareDevice(text) catch |err| {
        printBadSoftwareDevice(init.io, text);
        return err;
    };
}

// The printers below discard a failed write, like `args.writeStderr`, because it has nowhere to go.
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

fn printBadSoftwareDevice(io: Io, text: []const u8) void {
    var stderr_buffer: [error_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    w.print(
        "error: " ++ limits.software_device_env ++ " must be 1 or 0. The value is '{s}'\n",
        .{text},
    ) catch return;
    w.flush() catch {};
}

fn printBadStorageRange(io: Io, text: []const u8) void {
    var stderr_buffer: [error_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    w.print(
        "error: " ++ limits.storage_range_env ++ " must be a byte count in 1..{d}. " ++
            "The value is '{s}'\n",
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
