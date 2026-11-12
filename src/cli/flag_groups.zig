//! The flag values that several commands share. `EngineFlags` holds the options of an engine boot.
//! `run`, `serve`, `serve-openai`, and the benches nest it, so each option has one flag name and
//! one help text. A command that does not take an option names it in `omit`. `ServeFlags` adds the
//! options of `serve` and `serve-openai`. The docs follow the help dialect in `command.zig`.
//!
//! Comptime. Help text builds at comptime from the flag structs and the limits, so the numbers
//! cannot drift.

const std = @import("std");

const args = @import("args.zig");
const limits = @import("../core/limits.zig");
const dtype = @import("../core/dtype.zig");
const chat = @import("../core/chat.zig");
const kv_budget = @import("../core/kv_budget.zig");
const parallel = @import("../runtime/parallel.zig");
const engine_flags = @import("../resolve/flags.zig");

const Doc = args.Doc;

// -- types ------------------------------------------------------------------

pub const ThreadFlags = engine_flags.ThreadFlags;
pub const EngineFlags = engine_flags.EngineFlags;

/// The help of every `EngineFlags` option. A command that nests `EngineFlags` names this type in
/// `docs_from`.
pub const EngineDocs = struct {
    pub const docs = .{
        .threads = Doc{
            .group = "tuning",
            .text = "pool threads. 0 = auto, 1 = serial",
            .default = std.fmt.comptimePrint(
                "min(physical cores, {d})",
                .{parallel.max_default_threads},
            ),
        },
        .min_parallel_flops = Doc{
            .group = "tuning",
            .value = "F",
            .text = "ops below this stay serial",
            .default = groupDigits(parallel.default_min_parallel_flops),
        },
        .spin_us = Doc{
            .group = "tuning",
            .text = std.fmt.comptimePrint(
                "microseconds a pool worker spins before it sleeps\n(0 to {s})",
                .{groupDigits(parallel.max_spin_us)},
            ),
            .default = std.fmt.comptimePrint(
                "{d}",
                .{parallel.default_spin_ns / std.time.ns_per_us},
            ),
        },
        .kv_dtype = Doc{
            .group = "kv",
            .text = "KV element type. f16 halves every KV size",
            .default = "the model's registered one",
        },
        .max_seq = Doc{
            .group = "kv",
            .text = "KV positions per session, 1..the model ceiling. A window larger\n" ++
                "than VRAM needs --gpu-layers, which keeps the KV of the host\nlayers in RAM",
            .default = windowDefault(limits.run_session_kv_bytes),
        },
        .gpu_layers = Doc{
            .group = "gpu (fused-GPU build only)",
            .text = "pin N GPU layers, counted from the top (llama, qwen35).\n0 = CPU " ++
                "runner. A mid split runs the hybrid runner.\nauto puts the most layers on " ++
                "the GPU that the free VRAM holds",
        },
        .thinking = Doc{
            .group = "thinking (qwen35)",
            .text = "thinking mode of the chat template",
            .default = "auto, the model's own",
        },
        .thinking_budget = Doc{
            .group = "thinking (qwen35)",
            .text = "think tokens before the server closes the block",
            .default = "0, unbounded",
        },
    };
};

/// The options that `serve` and `serve-openai` share.
pub const ServeFlags = struct {
    engine: EngineFlags = .{},
    max_sessions: u16 = 4,
    max_connections: u32 = 8,
    /// The default is the default of the llama.cpp server option `--timeout`.
    io_timeout: u32 = 600,
    max_batch: ?u16 = null,
    prefill_chunk: ?u32 = null,
    vram_slots: ?u16 = null,
    kv_vram: ?u32 = null,
    kv_ram: ?u32 = null,
    prefix_disk: ?u32 = null,
    preload: args.Repeated([]const u8, limits.serve_preload_max) = .{},

    pub const docs_from = EngineDocs;
    pub const docs = .{
        .max_seq = Doc{
            .group = "kv",
            .text = "KV positions per session. A window larger than VRAM needs\n" ++
                "--gpu-layers, which keeps the KV of the host layers in RAM",
            .default = windowDefault(limits.serve_session_kv_bytes),
        },
        .gpu_layers = Doc{
            .group = "gpu (fused-GPU build only)",
            .text = "pin N GPU layers, counted from the top (llama, qwen35).\n0 = CPU " ++
                "runner. A mid split serves the hybrid runner,\nwhich holds at most 8 VRAM " ++
                "KV slots.\nauto puts the most layers on the GPU that the free VRAM holds",
        },
        .max_sessions = Doc{
            .group = "sessions",
            .text = std.fmt.comptimePrint(
                "logical conversations. Cap {d} CPU, {d} fused-GPU",
                .{ limits.max_batch, limits.max_logical_sessions },
            ),
        },
        .max_connections = Doc{
            .group = "sessions",
            .text = std.fmt.comptimePrint(
                "live client connections, cap {d}",
                .{limits.max_connections},
            ),
        },
        .io_timeout = Doc{
            .group = "sessions",
            .value = "S",
            .text = std.fmt.comptimePrint(
                "The seconds that one socket read or write can wait,\n" ++
                    "cap {d}. 0 sets no deadline. The deadline works on\n" ++
                    "Linux only. A binary client that holds a session\n" ++
                    "keeps its connection when idle.",
                .{limits.max_io_timeout_s},
            ),
        },
        .max_batch = Doc{
            .group = "sessions",
            .text = std.fmt.comptimePrint(
                "sessions advanced per decode tick, cap {d}",
                .{limits.max_batch},
            ),
            .default = std.fmt.comptimePrint("min(slots, {d})", .{limits.max_batch}),
        },
        .prefill_chunk = Doc{
            .group = "sessions",
            .text = "prompt tokens prefilled per pass. Unset, a pass with no decode\n" ++
                "row takes a full runner row width and a pass with decode rows\n" ++
                "takes the spare ones. 0 = always the spare decode rows\n" ++
                "(max-batch minus decode rows)",
            .default = "unset",
        },
        .vram_slots = Doc{
            .group = "gpu (fused-GPU build only)",
            .text = std.fmt.comptimePrint(
                "physical VRAM KV slots, 1..{d}, sized weights-aware. Below\n" ++
                    "--max-sessions it swaps idle KV to RAM",
                .{limits.max_vram_slots},
            ),
            .default = "--max-sessions",
        },
        .kv_vram = Doc{
            .group = "kv",
            .value = "MiB",
            .text = "VRAM for the KV slots. It sets the slot count, so it\n" ++
                "excludes --vram-slots. fused-GPU build only",
        },
        .kv_ram = Doc{
            .group = "kv",
            .value = "MiB",
            .text = "RAM ceiling for evicted KV blobs. Over it, a turn parks\nuntil a slot " ++
                "frees. Unset, the blob tier takes its full\nneed, or the boot refuses and " ++
                "names the --max-seq and\n--max-sessions that fit",
        },
        .prefix_disk = Doc{
            .group = "kv",
            .value = "MiB",
            .text = "disk ceiling for prefix-cache entries. They survive\n" ++
                "restarts ($ZIGFERENCER_CACHE_DIR)",
            .default = "off",
        },
        .preload = Doc{
            .group = "kv",
            .value = "FILE",
            .text = std.fmt.comptimePrint(
                "prefill the text of FILE (a system prompt) into the prefix\n" ++
                    "cache at boot, so the first session that opens with it\nstarts warm. " ++
                    "Repeatable, up to {d} files",
                .{limits.serve_preload_max},
            ),
        },
    };
};

// -- helpers ----------------------------------------------------------------

/// The default text of `--max-seq` for a session budget of `bytes`.
fn windowDefault(comptime bytes: u64) []const u8 {
    return std.fmt.comptimePrint(
        "the {d} MiB session budget, capped by the model ceiling.\n" ++
            "On the fused-GPU build, a default shrinks to fit the device\n" ++
            "buffer limits and the free VRAM",
        .{bytes >> 20},
    );
}

/// The decimal digits of `v` in groups of three, joined by `_`, as Zig source writes them.
fn groupDigits(comptime v: u64) []const u8 {
    comptime {
        const plain = std.fmt.comptimePrint("{d}", .{v});
        const len = plain.len + (plain.len - 1) / 3;
        var buf: [len]u8 = undefined;
        var j: usize = len;
        var i: usize = plain.len;
        var in_group: usize = 0;
        while (i > 0) {
            i -= 1;
            if (in_group == 3) {
                j -= 1;
                buf[j] = '_';
                in_group = 0;
            }

            j -= 1;
            buf[j] = plain[i];
            in_group += 1;
        }

        std.debug.assert(j == 0);
        const final = buf;
        return &final;
    }
}

// -- tests ------------------------------------------------------------------

test "EngineFlags: the thread help states the parallel defaults" {
    const Probe = struct {
        engine: EngineFlags = .{},
        pub const docs_from = EngineDocs;
        pub const help: args.Help = .{ .usage = "probe", .about = "" };
    };
    const body = comptime args.helpBody(Probe);

    const threads = std.fmt.comptimePrint(
        "[default: min(physical cores, {d})]",
        .{parallel.max_default_threads},
    );
    try std.testing.expect(std.mem.find(u8, body, threads) != null);
    const floor = "[default: " ++ comptime groupDigits(
        parallel.default_min_parallel_flops,
    ) ++ "]";
    try std.testing.expect(std.mem.find(u8, body, floor) != null);
    const spin = std.fmt.comptimePrint(
        "[default: {d}]",
        .{parallel.default_spin_ns / std.time.ns_per_us},
    );
    try std.testing.expect(std.mem.find(u8, body, spin) != null);
    try std.testing.expect(std.mem.find(u8, body, "--spin-us N") != null);
    try std.testing.expect(std.mem.find(u8, body, "--gpu-layers N|auto") != null);
    try std.testing.expect(std.mem.find(u8, body, "--thinking auto|on|off") != null);
}

test "ServeFlags: every shared serve flag reaches its field" {
    const Cmd = struct {
        serve: ServeFlags = .{},
        pub const help: args.Help = .{ .usage = "probe", .about = "" };
    };
    const argv = [_][]const u8{
        "--max-sessions",    "6",
        "--max-connections", "9",
        "--io-timeout",      "30",
        "--max-batch",       "3",
        "--prefill-chunk",   "0",
        "--vram-slots",      "2",
        "--kv-vram",         "100",
        "--kv-ram",          "64",
        "--prefix-disk",     "5",
        "--preload",         "a",
        "--preload",         "b",
        "--max-seq",         "512",
        "--kv-dtype",        "f16",
        "--gpu-layers",      "auto",
        "--threads",         "2",
        "--thinking",        "off",
        "--thinking-budget", "7",
    };
    var diag: args.Diag = .{};
    const cmd = try args.parse(Cmd, &argv, &diag);

    const p = cmd.serve;
    try std.testing.expectEqual(@as(u16, 6), p.max_sessions);
    try std.testing.expectEqual(@as(u32, 9), p.max_connections);
    try std.testing.expectEqual(@as(u32, 30), p.io_timeout);
    try std.testing.expectEqual(@as(?u16, 3), p.max_batch);
    try std.testing.expectEqual(@as(?u32, 0), p.prefill_chunk);
    try std.testing.expectEqual(@as(?u16, 2), p.vram_slots);
    try std.testing.expectEqual(@as(?u32, 100), p.kv_vram);
    try std.testing.expectEqual(@as(?u32, 64), p.kv_ram);
    try std.testing.expectEqual(@as(?u32, 5), p.prefix_disk);
    try std.testing.expectEqual(@as(usize, 2), p.preload.slice().len);
    try std.testing.expectEqualStrings("b", p.preload.slice()[1]);
    try std.testing.expectEqual(@as(?u32, 512), p.engine.max_seq);
    try std.testing.expectEqual(@as(?dtype.KvDtype, .f16), p.engine.kv_dtype);
    try std.testing.expectEqual(kv_budget.GpuLayers.auto, p.engine.gpu_layers.?);
    try std.testing.expectEqual(@as(?u32, 2), p.engine.threads.threads);
    try std.testing.expectEqual(chat.ThinkingFlag.off, p.engine.thinking);
    try std.testing.expectEqual(@as(u32, 7), p.engine.thinking_budget);
}

test "ServeFlags: the help names the io deadline, its cap, and the llama.cpp default" {
    const Probe = struct {
        serve: ServeFlags = .{},
        pub const help: args.Help = .{ .usage = "probe", .about = "" };
    };
    const body = comptime args.helpBody(Probe);

    try std.testing.expect(std.mem.find(u8, body, "--io-timeout S") != null);
    const cap = std.fmt.comptimePrint("cap {d}", .{limits.max_io_timeout_s});
    try std.testing.expect(std.mem.find(u8, body, cap) != null);
    try std.testing.expect(std.mem.find(u8, body, "Linux only") != null);
    const flag = std.mem.find(u8, body, "--io-timeout S").?;
    const default = std.mem.find(u8, body[flag..], "[default: 600]").?;
    // The default closes the entry of the flag, before the next flag starts.
    try std.testing.expect(std.mem.find(u8, body[flag..][0..default], "\n  --") == null);
}

test "groupDigits: groups of three from the right, as the Zig source writes them" {
    try std.testing.expectEqualStrings("0", comptime groupDigits(0));
    try std.testing.expectEqualStrings("8", comptime groupDigits(8));
    try std.testing.expectEqualStrings("999", comptime groupDigits(999));
    try std.testing.expectEqualStrings("1_000", comptime groupDigits(1_000));
    try std.testing.expectEqualStrings("250_000", comptime groupDigits(250_000));
    try std.testing.expectEqualStrings("12_345_678", comptime groupDigits(12_345_678));
}
