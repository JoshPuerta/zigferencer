//! Subcommand dispatch: argv to a sub-handler call, the one entry point from
//! `main.zig`. Each subcommand is a `{ name, summary, handler }` row in one of
//! two closed tables. `release` holds the commands a user of the engine needs.
//! `dev` holds the benches, the kernel checks, the demo, and the tokenizer dump.
//! The release binary dispatches over `release`, and `zigferencer-dev`
//! dispatches over `release ++ dev`. The release binary names the dev commands
//! only as strings, so it never analyzes a dev handler. The same `summary`
//! string feeds the no-args usage screen, so help and dispatch cannot drift.
//!
//! Help-text dialect, uniform across every `--help` so a reader learns it once.
//! `args.helpBody` generates the flag lines from the flag struct of a command:
//!   - A `summary` is an imperative fragment of about 55 characters or less.
//!   - A `--help` prints a `usage:` line, then one sentence, then the flags in
//!     groups under short lowercase headers (for example tuning, kv, gpu),
//!     then `examples:`.
//!   - A default reads as a trailing `[default: X]`, never as inline prose.
//!   - One qualifier per flag line.
//!   - A backend caveat goes in a group header (`gpu (fused-GPU build only)`),
//!     never in a per-line suffix.
//!   - A shared knob keeps one name and one wording everywhere, because
//!     `flags.EngineFlags` and `flags.ServeFlags` declare it once.
//!   - A flag that the build cannot honor stays listed, so it stays
//!     discoverable. The command rejects it at use with a clear message.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const tokens_cmd = @import("tokens_cmd.zig");
const fetch_cmd = @import("fetch_cmd.zig");
const list_cmd = @import("list_cmd.zig");
const inspect_cmd = @import("inspect_cmd.zig");
const run_cmd = @import("run_cmd.zig");
const serve_cmd = @import("serve_cmd.zig");
const serve_openai_cmd = @import("serve_openai_cmd.zig");
const info_cmd = @import("info_cmd.zig");
const bench_model_cmd = @import("bench_model_cmd.zig");
const bench_serve_cmd = @import("bench_serve_cmd.zig");
const bench_swap_cmd = @import("bench_swap_cmd.zig");
const bench_decode_host_cmd = @import("bench_decode_host_cmd.zig");
const serve_demo_cmd = @import("serve_demo_cmd.zig");
const client_cmd = @import("client_cmd.zig");
const gpu_selftest_cmd = @import("gpu_selftest_cmd.zig");
const style = @import("style.zig");
const registry = @import("../core/registry.zig");
const limits = @import("../core/limits.zig");
const prompt_cache = @import("../runtime/prompt_cache.zig");

// -- types ------------------------------------------------------------------

pub const Context = struct {
    io: Io,
    arena: Allocator,
    gpa: Allocator,
    args: []const [:0]const u8,
    /// Where models are stored on disk. Resolved once from
    /// `$ZIGFERENCER_MODELS_DIR` (default `models/`). See core/registry.zig.
    models_root: []const u8,
    /// Where the prompt cache keeps snapshots. Resolved once from
    /// `$ZIGFERENCER_CACHE_DIR` (default `.prompt_cache`). See runtime/prompt_cache.zig.
    cache_dir: []const u8,
    /// The storage-buffer range that replaces the one the device reports.
    /// Resolved once from `$ZIGFERENCER_VK_STORAGE_RANGE` (default: unset, the
    /// reported range). Every command that opens a Vulkan context passes it.
    vk_storage_range: ?u64,
    /// True when stdout takes ANSI color. Resolved once: stdout is a TTY and
    /// `$NO_COLOR` is unset. Commands build a `style.Palette` from it. See
    /// cli/style.zig.
    color: bool,
    /// True when stderr takes ANSI color. The CLI resolves it apart from `color`
    /// because the two streams redirect independently. The interactive pickers and
    /// the confirms write to stderr and key their palette on this.
    color_err: bool,
};

// The handlers hold heterogeneous error sets (HTTP, GGUF parse, Vulkan), and one
// function-pointer table can share only one return type.
const Handler = *const fn (Context) anyerror!void;

const Subcommand = struct {
    name: []const u8,
    summary: []const u8,
    handler: Handler,
};

/// The commands of the release binary.
const release = [_]Subcommand{
    .{ .name = "list", .summary = "List registered models (and which are on disk)", .handler = list_cmd.run },
    .{ .name = "fetch", .summary = "Download a model (registry name or HF path)", .handler = fetch_cmd.run },
    .{ .name = "inspect", .summary = "Print a downloaded GGUF's metadata", .handler = inspect_cmd.run },
    .{ .name = "run", .summary = "Generate text from a prompt (greedy)", .handler = run_cmd.run },
    .{ .name = "serve", .summary = "Serve sessions over TCP/unix, batched decode", .handler = serve_cmd.run },
    .{ .name = "serve-openai", .summary = "Serve an OpenAI-compatible HTTP API", .handler = serve_openai_cmd.run },
    .{ .name = "client", .summary = "Chat with a running server over the binary wire", .handler = client_cmd.run },
    .{ .name = "info", .summary = "Print the build, host card, and support matrix", .handler = info_cmd.run },
};

/// The developer commands. Only `zigferencer-dev` dispatches them.
const dev = [_]Subcommand{
    .{ .name = "bench-model", .summary = "Benchmark a model: TTFT, ITL, hash", .handler = bench_model_cmd.run },
    .{ .name = "bench-serve", .summary = "Benchmark serve throughput vs batch size", .handler = bench_serve_cmd.run },
    .{ .name = "bench-swap", .summary = "Benchmark tiered-KV swap cost (go/no-go)", .handler = bench_swap_cmd.run },
    .{ .name = "bench-decode-host", .summary = "Time the host legs of one decode step", .handler = bench_decode_host_cmd.run },
    .{ .name = "gpu-selftest", .summary = "Bit-check a GPU kernel vs its CPU reference", .handler = gpu_selftest_cmd.run },
    .{ .name = "tokens", .summary = "Print a prompt's token ids", .handler = tokens_cmd.run },
    .{ .name = "serve-demo", .summary = "Demo the serve connector end-to-end", .handler = serve_demo_cmd.run },
};

/// The dev command names as strings, for the hint of the release binary. A read
/// of `dev` takes the address of every dev handler and analyzes it, so the list
/// repeats the names. `dispatchDev` checks the two lists against each other.
const dev_names = [_][]const u8{
    "bench-model",
    "bench-serve",
    "bench-swap",
    "bench-decode-host",
    "gpu-selftest",
    "tokens",
    "serve-demo",
};

comptime {
    // A name in both tables makes the dispatch order decide the handler.
    for (release) |r| {
        for (dev_names) |d| {
            if (std.mem.eql(u8, r.name, d)) @compileError("command '" ++ d ++ "' sits in both tables");
        }
    }
}

// -- api --------------------------------------------------------------------

/// Dispatch argv over the release commands.
pub fn dispatchRelease(init: std.process.Init) !void {
    return dispatchOver(init, &release, "zigferencer");
}

/// Dispatch argv over the release and the dev commands.
pub fn dispatchDev(init: std.process.Init) !void {
    comptime {
        std.debug.assert(dev.len == dev_names.len);
        for (dev, dev_names) |d, name| {
            if (!std.mem.eql(u8, d.name, name)) @compileError("dev_names diverges from dev at '" ++ d.name ++ "'");
        }
    }
    return dispatchOver(init, &(release ++ dev), "zigferencer-dev");
}

/// True when `arg` asks for the top-level usage screen.
pub fn isHelpWord(arg: []const u8) bool {
    const words = [_][]const u8{ "--help", "-h", "help" };
    for (words) |w| {
        if (std.mem.eql(u8, arg, w)) return true;
    }
    return false;
}

// -- helpers ----------------------------------------------------------------

/// Resolve `args[1]` against `table` and run its handler.
fn dispatchOver(init: std.process.Init, comptime table: []const Subcommand, comptime binary_name: []const u8) !void {
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
    const cache_dir = prompt_cache.resolveCacheDir(init.environ_map.get(prompt_cache.cache_dir_env));
    const vk_storage_range: ?u64 = if (init.environ_map.get(limits.storage_range_env)) |text|
        limits.parseStorageRange(text) catch |err| {
            printBadStorageRange(init.io, text);
            return err;
        }
    else
        null;
    const sub = args[1];
    for (table) |s| {
        if (std.mem.eql(u8, s.name, sub)) {
            return s.handler(.{
                .io = init.io,
                .arena = arena,
                .gpa = init.gpa,
                .args = args[2..],
                .models_root = models_root,
                .cache_dir = cache_dir,
                .vk_storage_range = vk_storage_range,
                .color = color,
                .color_err = color_err,
            });
        }
    }
    // The dev binary holds every dev name, so only the release binary reaches
    // the hint.
    for (dev_names) |name| {
        if (std.mem.eql(u8, name, sub)) {
            printDevCommand(init.io, sub);
            return error.UnknownSubcommand;
        }
    }
    printUnknownSubcommand(init.io, sub, binary_name);
    return error.UnknownSubcommand;
}

// The usage and unknown-subcommand paths discard a failed write, like
// `args.writeStderr`. A failure there has nowhere useful to go.
fn printUsage(io: Io, file: Io.File, p: style.Palette, comptime table: []const Subcommand, comptime binary_name: []const u8) void {
    var buf: [4096]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(file, io, &buf);
    const w = &fw.interface;
    w.print("{s}" ++ binary_name ++ "{s}: a small LLM inference engine (Zig 0.16)\n\n", .{ p.bold, p.reset }) catch return;
    w.print("{s}usage:{s} " ++ binary_name ++ " <command> [args]\n\n", .{ p.bold, p.reset }) catch return;
    w.print("{s}commands:{s}\n", .{ p.bold, p.reset }) catch return;
    const width = comptime nameWidth(table);
    for (table) |s| {
        w.print("  {s}{s}{s}{s} {s}\n", .{ p.green, s.name, p.reset, pad[0 .. width - s.name.len], s.summary }) catch return;
    }
    w.print("\n{s}'" ++ binary_name ++ " <command> --help' lists a command's flags.{s}\n", .{ p.dim, p.reset }) catch return;
    w.flush() catch {};
}

/// The longest command name of `table`, the width of the name column.
fn nameWidth(comptime table: []const Subcommand) usize {
    var width: usize = 0;
    for (table) |s| width = @max(width, s.name.len);
    std.debug.assert(width <= pad.len);
    return width;
}

const pad = " " ** 32;

fn printBadStorageRange(io: Io, text: []const u8) void {
    var buf: [256]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    const w = &fw.interface;
    w.print("error: " ++ limits.storage_range_env ++ " must be a byte count in 1..{d}. The value is '{s}'\n", .{ limits.storage_range_max, text }) catch return;
    w.flush() catch {};
}

fn printDevCommand(io: Io, sub: []const u8) void {
    var buf: [256]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    const w = &fw.interface;
    w.print("error: '{s}' is a developer command. Run it with zigferencer-dev.\n", .{sub}) catch return;
    w.flush() catch {};
}

fn printUnknownSubcommand(io: Io, sub: []const u8, comptime binary_name: []const u8) void {
    var buf: [256]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    const w = &fw.interface;
    w.print("error: unknown subcommand '{s}'\n", .{sub}) catch return;
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
    for (cases) |c| try std.testing.expectEqual(c.want, isHelpWord(c.arg));
}
