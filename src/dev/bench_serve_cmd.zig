//! `zigferencer bench-serve <pub_name>` draws the batching-win curve. For each session count it
//! starts a fresh in-process server, drives that many concurrent sessions of one turn through the
//! reference client, and reports the throughput and the inter-token tails. The turn is raw text,
//! because an instruct template can hit EOS early.
//!
//! Sizing. The bench plans what `serve` plans: the KV type, the window, and the placement come
//! from `resolve`. The last row is the most sessions of the model: the largest count that holds
//! every session in a slot. Past `limits.max_batch` the sessions share the decode rows of a tick.
//!
//! Rates. A row counts the generated tokens over its whole wall time, so prefill lowers the rate.
//! The prefix cache stays off, so every session pays for its own prompt.
//!
//! Comptime. Generic over the model family and the server type. The fused build option picks
//! the GPU curve.
//!
//! Decoding is greedy and the prompts are identical, so every session generates the
//! same stream at every B, and the rows differ only in concurrency.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;
const common = @import("bench_model/bench_model_common.zig");
const flags_mod = zigferencer.cli.flag_groups;
const select = zigferencer.cli.select;
const boot_resources = zigferencer.cli.boot_resources;
const boot_lines = zigferencer.cli.boot_lines;
const family = zigferencer.family;
const bpe = zigferencer.bpe;
const dtype = zigferencer.dtype;
const kv_budget = zigferencer.kv_budget;
const kv_index = zigferencer.kv_index;
const limits = zigferencer.limits;
const registry = zigferencer.registry;
const resolve_mod = zigferencer.resolve;
const vocab_mod = zigferencer.vocab;
const loader = zigferencer.loader;
const parallel = zigferencer.parallel;
const Pool = zigferencer.pool.Pool;
const Sessions = zigferencer.session.Sessions;
const Client = zigferencer.serve_client.Client;
const event = zigferencer.serve_event;
const prefill_chunk = zigferencer.serve_prefill_chunk;
const protocol = zigferencer.serve_protocol;
const server_mod = zigferencer.serve_server;

const log = std.log.scoped(.bench_serve);

const fused_build = build_options.use_vulkan_fused_ops;

// -- constants --------------------------------------------------------------

/// A session holds one connection here, and a CPU boot holds one decode row per session.
const session_cap: u16 = if (fused_build)
    @intCast(@min(limits.max_connections, limits.max_logical_sessions))
else
    @intCast(limits.max_batch);

/// The rows of a curve: the powers of two below the most sessions, then the most sessions.
const rows_max: usize = std.math.log2_int_ceil(u32, limits.max_connections) + 2;

// -- types ------------------------------------------------------------------

const Row = struct {
    tokens: u64,
    wall_s: f64,
    agg_tok_s: f64,
    per_sess: f64,
    ttft_ms: f64,
    p50_ms: f64,
    p99_ms: f64,
    /// Token arrivals past a session's store of `store` entries, summed over
    /// the row. The ITL tails omit them, and the row says so.
    unstored: u64,
    store: u32,
    /// Sessions that ended before `store` tokens, because the model reached EOS.
    short: u32,
};

/// One session's turn. `arrivals` holds `--max-tokens` entries, and
/// `n_tokens` counts every token event, stored or not.
const ClientResult = struct {
    ok: bool = false,
    n_tokens: u32 = 0,
    t_sent: Io.Clock.Timestamp = undefined,
    arrivals: []Io.Clock.Timestamp,
};

/// The flags of `bench-serve`.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    prompt: []const u8 = "The market square fills with traders, and",
    prompt_tokens: ?u32 = null,
    max_tokens: u32 = 48,
    max_sessions: ?u16 = null,
    max_only: bool = false,
    vram_budget: ?u32 = null,
    engine: flags_mod.EngineFlags = .{},

    pub const docs_from = flags_mod.EngineDocs;
    pub const positional = "pub_name";
    pub const omit: []const []const u8 = &.{
        "gpu_layers",
        "thinking",
        "thinking_budget",
    };
    pub const help: args.Help = .{
        .usage = "zigferencer-dev bench-serve <pub_name> [flags]",
        .about = "Drive 1, 2, 4, ... concurrent sessions up to the most the model holds, " ++
            "fresh server per row. Report tok/s + ITL tails.\n",
        .footer =
        \\examples:
        \\  zigferencer-dev bench-serve SmolLM2-135M-Instruct-Q8_0 --max-tokens 64
        \\  zigferencer-dev bench-serve Qwen3.5-2B-Q8_0 --prompt-tokens 512 --max-tokens 128 \
        \\      --vram-budget 10240 --max-only
        \\
        ,
    };
    pub const docs = .{
        .prompt = args.Doc{
            .text = "the turn every session sends",
            .default = "a fixed continuation",
        },
        .prompt_tokens = args.Doc{
            .text = "repeat the prompt, then cut it at N tokens as the server encodes them",
        },
        .max_tokens = args.Doc{ .text = "the most tokens one turn generates, 1..max-seq" },
        .max_sessions = args.Doc{
            .text = "the session count of the last row, a cap on what the model holds",
            .default = "the most the model holds",
        },
        .max_only = args.Doc{ .text = "run only the last row of the curve" },
        .vram_budget = args.Doc{
            .text = "MiB of VRAM the plan reads, at most the live budget. Fused build only.\n" ++
                "A fixed figure keeps the session count the same from run to run",
            .default = "the live budget",
        },
        .max_seq = args.Doc{
            .group = "kv",
            .text = "KV positions per session",
            .default = std.fmt.comptimePrint(
                "the {d} MiB session budget at the KV type, capped by\nthe model ceiling",
                .{limits.serve_session_kv_bytes >> 20},
            ),
        },
    };
};

/// What every row of one curve reads.
const Curve = struct {
    ctx: context.Context,
    w: *Io.Writer,
    loaded: *const loader.Loaded,
    codec: *const event.Codec,
    pool: *Pool,
    policy: parallel.Policy,
    flags: *const Flags,
    boot: *const boot_resources.DeviceBoot,
    kv_dtype: dtype.KvDtype,
    max_seq: u32,
    prompt: []const u8,
    prompt_tokens: u32,
    max_tokens: u32,
};

fn ServerTask(comptime ServerT: type) type {
    return struct {
        fn run(server: *ServerT) void {
            server.run() catch |err| log.err("the bench server stopped: {t}", .{err});
        }
    };
}

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    // Parse the flags. The fused build runs no CPU op, so its pool starts no workers.
    const flags = try args.parseInto(Flags, ctx) orelse return;
    try checkFlags(ctx.io, &flags);
    const thread_flags = flags.engine.threads.policyFlags();
    const policy = try boot_resources.resolvePolicy(ctx.io, thread_flags, !fused_build);

    // Load the model, and fix the window and the KV type as `serve` does.
    const model = try select.model(ctx, flags.pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    const entry = boot_resources.entryOf(model);
    const input: resolve_mod.Input = .{ .engine = flags.engine };
    const first = resolve_mod.spendVram(input, 1);
    const boot = try boot_resources.resolveLoaded(ctx, .serve, first, entry, &loaded);
    const max_seq = boot.values.window.base;

    // A turn ends at the window, so the window bounds the tokens a turn emits.
    if (flags.max_tokens == 0 or flags.max_tokens > max_seq) {
        return args.fail(
            ctx.io,
            "--max-tokens must be in 1..{d}, the --max-seq window",
            .{max_seq},
        );
    }

    // One pool serves every CPU op of the command.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, policy);
    defer pool.deinit();

    // Build the codec and print the intro.
    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);
    const pieces = try decodePieces(ctx.arena, &v);
    const codec: event.Codec = .{
        .vocab = &v,
        .kind = .none,
        .pieces = pieces,
        .eos_id = v.eos_id,
    };

    const prompt = if (flags.prompt_tokens) |n|
        try textOfTokens(ctx.arena, &v, flags.prompt, n)
    else
        flags.prompt;
    const prompt_tokens = (try plainIds(ctx.arena, &v, prompt, true)).len;

    const w = &ctx.stdout.interface;
    try printIntro(w, model.pub_name, policy, prompt_tokens, flags.max_tokens);

    if (comptime build_options.use_vulkan_ops) {
        log.err("bench-serve does not run on the per-op vulkan backend. " ++
            "Build -Dops-backend=vulkan_fused or a CPU backend", .{});
        return error.UnsupportedServeBackend;
    }

    // Run the curve for the family.
    const curve: Curve = .{
        .ctx = ctx,
        .w = w,
        .loaded = &loaded,
        .codec = &codec,
        .pool = &pool,
        .policy = policy,
        .flags = &flags,
        .boot = &boot.device,
        .kv_dtype = boot.values.early.kv_dtype,
        .max_seq = max_seq,
        .prompt = prompt,
        .prompt_tokens = @intCast(prompt_tokens),
        .max_tokens = flags.max_tokens,
    };
    switch (loaded.card.arch) {
        inline else => |arch| return benchFamily(family.Of(arch), &curve),
    }
}

// -- helpers ----------------------------------------------------------------

fn checkFlags(io: Io, flags: *const Flags) !void {
    if (flags.max_sessions) |n| {
        if (n == 0 or n > session_cap) {
            return args.fail(io, "--max-sessions must be in 1..{d}", .{session_cap});
        }
    }
    if (flags.prompt_tokens) |n| {
        if (n == 0) return args.fail(io, "--prompt-tokens must be at least 1", .{});
    }
    if (flags.vram_budget) |mib| {
        if (!fused_build) {
            return args.fail(io, "--vram-budget applies only to the fused-GPU build", .{});
        }
        if (mib == 0) return args.fail(io, "--vram-budget must be at least 1", .{});
    }
}

/// The ids of `text` as the server encodes request text, as plain text with no special token.
fn plainIds(
    arena: Allocator,
    v: *const vocab_mod.Vocab,
    text: []const u8,
    sequence_start: bool,
) ![]u32 {
    const whole: []const bpe.PlainRange = &.{.{ .start = 0, .end = text.len }};
    return bpe.encodePromptAlloc(v, arena, text, whole, sequence_start);
}

/// A text that the server encodes to `n` tokens: `seed` repeated, then cut at a token. A cut can
/// move a token boundary, so the text encodes once more, and a miss fails the bench, because a
/// wrong prompt length is not the measurement.
fn textOfTokens(arena: Allocator, v: *const vocab_mod.Vocab, seed: []const u8, n: u32) ![]const u8 {
    std.debug.assert(n >= 1);

    const seed_tokens = (try plainIds(arena, v, seed, false)).len;
    if (seed_tokens == 0) return error.EmptyPrompt;

    // Repeat past `n` tokens, and leave room for the start of the sequence.
    const start_tokens: u32 = @intCast((try plainIds(arena, v, "", true)).len);
    if (n <= start_tokens) return error.PromptTokensMissed;

    const repeats = n / seed_tokens + 2;
    var long: std.ArrayList(u8) = .empty;
    for (0..repeats) |_| {
        try long.appendSlice(arena, seed);
        try long.append(arena, ' ');
    }

    const ids = try plainIds(arena, v, long.items, false);
    const keep = n - start_tokens;
    if (ids.len < keep) return error.PromptTokensMissed;

    const text = try bpe.decodeAlloc(v, arena, ids[0..keep]);
    const back = try plainIds(arena, v, text, true);
    if (back.len != n) return error.PromptTokensMissed;

    return text;
}

/// The rows of a curve into `out`: the powers of two below `most`, then `most`. `max_only` keeps
/// the last row alone.
fn sessionRows(most: u16, max_only: bool, out: *[rows_max]u16) []const u16 {
    std.debug.assert(most >= 1);
    std.debug.assert(most <= limits.max_connections);

    var n: usize = 0;
    if (!max_only) {
        var row: u16 = 1;
        while (row < most) : (row *= 2) {
            out[n] = row;
            n += 1;
        }
    }

    out[n] = most;
    return out[0 .. n + 1];
}

fn decodePieces(arena: Allocator, v: *const vocab_mod.Vocab) ![]const []const u8 {
    const pieces = try arena.alloc([]const u8, v.tokens.len);
    for (pieces, 0..) |*piece, id| {
        piece.* = try bpe.decodeAlloc(v, arena, &[_]u32{@intCast(id)});
        if (piece.len > limits.max_token_text_bytes) return error.TokenPieceTooLong;
    }

    return pieces;
}

fn printIntro(
    w: *Io.Writer,
    model_name: []const u8,
    policy: parallel.Policy,
    prompt_tokens: usize,
    max_tokens: u32,
) !void {
    try w.print("model:    {s}\n", .{model_name});
    try w.print("backend:  {s}  threads: {d}  (min_parallel_flops={d})\n", .{
        build_options.ops_backend, policy.n_threads, policy.min_parallel_flops,
    });
    try w.print("turn:     {d} prompt tokens -> up to {d} tokens, greedy, no template\n", .{
        prompt_tokens, max_tokens,
    });
    try w.flush();
}

fn benchFamily(comptime Family: type, c: *const Curve) !void {
    if (comptime fused_build) return fusedCurve(Family, c);
    return cpuCurve(Family, c);
}

/// A fresh runner and engine per row give clean KV pools.
fn cpuCurve(comptime Family: type, c: *const Curve) !void {
    const Runner = Family.Cpu;
    const ctx = c.ctx;
    const cap = c.flags.max_sessions orelse session_cap;

    const resources: resolve_mod.Resources = .{
        .ram_free = c.boot.ramFree(ctx.io),
        .model = c.boot.model,
    };
    const input = c.boot.input;
    const most = resolve_mod.mostSessions(input, c.boot.entry, c.boot.card, resources, cap) orelse {
        return args.fail(ctx.io, "the host RAM holds no session at {d} positions", .{c.max_seq});
    };

    try common.requireSequenceFits(ctx.io, c.prompt_tokens, c.max_tokens, c.max_seq);
    try c.w.print("sessions: {d} at most, kv {t}, limit {d}\n", .{ most, c.kv_dtype, cap });
    try printTableHead(c.w, c.max_seq);

    var rows_buffer: [rows_max]u16 = undefined;
    for (sessionRows(most, c.flags.max_only, &rows_buffer)) |b| {
        var round_arena_state = std.heap.ArenaAllocator.init(ctx.gpa);
        defer round_arena_state.deinit();

        const arena = round_arena_state.allocator();

        var runner = try Runner.init(arena, c.loaded, .{
            .plan = kv_budget.pinned(b, c.max_seq),
            .max_rows = prefill_chunk.stepRows(prefill_chunk.cpu_rule, b, b),
            .pool = c.pool,
            .policy = c.policy,
            .kv_dtype = c.kv_dtype,
        });
        const EngineT = Sessions(Runner);
        var engine = try EngineT.init(arena, &runner);
        const rule = prefill_chunk.cpu_rule;
        const row = try benchOne(EngineT, c, &engine, rule, c.max_seq, b, b);
        try printRow(c.w, b, row);
    }
}

/// One engine holds the whole curve, so the weights upload once, and the plan takes
/// the widest row. The rows reuse slots: a session closes at the end of a row, and
/// `create` re-zeros any recurrent state.
fn fusedCurve(comptime Family: type, c: *const Curve) !void {
    const ctx = c.ctx;

    const ctxp = try boot_resources.openDevice(ctx);
    defer ctxp.deinit();

    // Plan what `serve` plans for the most sessions of the model.
    const live = boot_resources.vramFigure(ctxp);
    const budget: ?u64 = if (c.flags.vram_budget) |mib| @as(u64, mib) << 20 else null;
    const vram = @min(live, budget orelse live);
    const device: resolve_mod.DeviceCapacity = .{
        .vram = vram,
        .driver = ctxp.driver,
        .buffer_limits = ctxp.buffer_limits,
    };
    const resources: resolve_mod.Resources = .{
        .ram_free = c.boot.ramFree(ctx.io),
        .model = c.boot.model,
        .device = device,
    };
    const cap = c.flags.max_sessions orelse session_cap;
    const entry = c.boot.entry;
    const card = c.boot.card;
    const input = c.boot.input;
    const most = resolve_mod.mostSessions(input, entry, card, resources, cap) orelse {
        return args.fail(ctx.io, "{d} MiB of VRAM holds no session of {s}", .{
            vram >> 20, entry.pub_name,
        });
    };

    const shaped = resolve_mod.spendVram(input, most);
    const boot = switch (resolve_mod.resolve(.serve, shaped, entry, card, resources)) {
        .ok => |r| r,
        .refused => |r| return boot_lines.refuse(ctx.io, entry.pub_name, r),
    };
    const placed = boot.device.?.placement.ok.fused;
    const plan = placed.plan;
    boot_lines.logPlan(plan, c.kv_dtype);
    try common.requireSequenceFits(ctx.io, c.prompt_tokens, c.max_tokens, plan.window);
    try printDevicePlan(c, ctxp, placed, vram, live, most);

    const fused = try Family.Fused.init(ctx.gpa, ctxp, c.loaded, plan, c.kv_dtype);
    defer fused.deinit();

    const EngineT = Sessions(Family.Fused);
    var engine = try EngineT.init(ctx.arena, fused);

    try printTableHead(c.w, fused.max_seq);
    var rows_buffer: [rows_max]u16 = undefined;
    for (sessionRows(most, c.flags.max_only, &rows_buffer)) |b| {
        const rule: prefill_chunk.Rule = .{ .derived = engine.runner.maxStepRows() };
        const width: u16 = @intCast(@min(b, plan.n_streams));
        const row = try benchOne(EngineT, c, &engine, rule, fused.max_seq, b, width);
        try printRow(c.w, b, row);
    }
}

/// What bound the session count: the VRAM, the device buffers, and the connections of the bench.
fn printDevicePlan(
    c: *const Curve,
    ctxp: anytype,
    placed: anytype,
    vram: u64,
    live: u64,
    most: u16,
) !void {
    const plan = placed.plan;
    const slot_bytes = plan.vram.kv / plan.n_slots;
    const by_vram = kv_budget.derivedX(placed.caps) / slot_bytes;
    const by_device = kv_index.maxSlotsForLimits(
        plan.window,
        c.boot.card.kvShape().kv_row_elems,
        c.kv_dtype,
        ctxp.buffer_limits,
    );

    try c.w.print("vram:     plan reads {d} MiB of {d} MiB live\n", .{ vram >> 20, live >> 20 });
    try c.w.print("sessions: {d} at most, window {d}, kv {t}, slot {d} MiB\n", .{
        most, plan.window, c.kv_dtype, slot_bytes >> 20,
    });
    try c.w.print("limits:   vram {d}, device buffers {d}, connections {d}\n", .{
        by_vram, by_device, session_cap,
    });
    try c.w.flush();
}

/// The table head names the window the engine holds.
fn printTableHead(w: *Io.Writer, max_seq: u32) !void {
    try w.print("max_seq:  {d}\n", .{max_seq});
    try w.print("note:     agg_tok/s is generated tokens over the wall time, with prefill\n", .{});
    try w.print("note:     prefill runs on decode ticks. " ++
        "Concurrency shows in ttft, not itl tails\n\n", .{});
    try w.print("  sessions   tokens   wall_s   agg_tok/s   per_session   ttft_ms   " ++
        "itl_p50_ms   itl_p99_ms\n", .{});
    try w.flush();
}

fn printRow(w: *Io.Writer, b: u16, row: Row) !void {
    try w.print("  {d:>8}   {d:>6}   {d:>6.1}   {d:>9.2}   {d:>11.2}   " ++
        "{d:>7.0}   {d:>10.1}   {d:>10.1}\n", .{
        b,            row.tokens,  row.wall_s, row.agg_tok_s,
        row.per_sess, row.ttft_ms, row.p50_ms, row.p99_ms,
    });
    if (row.unstored > 0) {
        try w.print("      cut: {d} token arrivals passed the store of {d} per session. " ++
            "The ITL tails omit them\n", .{ row.unstored, row.store });
    }
    if (row.short > 0) {
        try w.print("          short: {d} sessions reached EOS before {d} tokens\n", .{
            row.short, row.store,
        });
    }

    try w.flush();
}

/// One row: a fresh server, `b` concurrent clients, and the server stop. A fresh
/// server gives clean queues and a clean port, and it lives in a round arena.
fn benchOne(
    comptime EngineT: type,
    c: *const Curve,
    engine: *EngineT,
    rule: prefill_chunk.Rule,
    max_seq: u32,
    b: u16,
    width: u16,
) !Row {
    const ServerT = server_mod.Server(EngineT, event.Codec);
    const ctx = c.ctx;
    std.debug.assert(b <= limits.max_connections);
    std.debug.assert(width >= 1);
    std.debug.assert(width <= b);
    std.debug.assert(width <= limits.max_batch);

    var round_arena_state = std.heap.ArenaAllocator.init(ctx.gpa);
    defer round_arena_state.deinit();

    const arena = round_arena_state.allocator();

    // Allocate the stores before the threads start, because the arena is not thread-safe.
    var stores: [limits.max_connections][]Io.Clock.Timestamp = undefined;
    for (stores[0..b]) |*store| {
        store.* = try arena.alloc(Io.Clock.Timestamp, c.max_tokens);
    }

    const itl = try arena.alloc(f64, @as(usize, b) * c.max_tokens);

    // Start the server and drive the clients.
    var server = try ServerT.init(arena, ctx.gpa, ctx.io, engine, c.codec, .{
        .max_sessions = b,
        .max_batch = width,
        .max_connections = b,
        .max_seq = max_seq,
        .max_step_rows = engine.runner.maxStepRows(),
        .prefill = rule,
        .allow_shutdown = true,
        .model_name = "bench",
        .vocab_size = c.loaded.card.vocab_size,
        .eos_id = c.codec.eos_id,
    }, .{ .tcp = .{} });
    var server_future = try ctx.io.concurrent(ServerTask(ServerT).run, .{&server});
    const addr = server.boundAddress();

    const t0 = Io.Clock.Timestamp.now(ctx.io, .awake);
    var results: [limits.max_connections]ClientResult = undefined;
    try driveClients(c, addr, stores[0..b], results[0..b]);
    const t1 = Io.Clock.Timestamp.now(ctx.io, .awake);

    // Stop the server through the protocol.
    var control = try Client.connectTcp(ctx.gpa, ctx.io, addr);
    try control.shutdown();
    control.deinit();
    server_future.await(ctx.io);

    return summarize(results[0..b], itl, ms(t0, t1) / 1000.0, c.max_tokens);
}

/// Run one client per store to the end of its turn.
fn driveClients(
    c: *const Curve,
    addr: Io.net.IpAddress,
    stores: []const []Io.Clock.Timestamp,
    results: []ClientResult,
) !void {
    const ctx = c.ctx;
    std.debug.assert(stores.len == results.len);
    std.debug.assert(stores.len <= limits.max_connections);

    var futures_buffer: [limits.max_connections]Io.Future(ClientResult) = undefined;
    const futures = futures_buffer[0..stores.len];
    for (futures, stores) |*future, store| {
        future.* = try ctx.io.concurrent(clientTask, .{
            ctx.gpa, ctx.io, addr, c.prompt, c.max_tokens, store,
        });
    }

    for (futures, results) |*future, *result| {
        result.* = future.await(ctx.io);
    }
}

/// The figures of one row. `itl` is a scratch of `results.len * max_tokens` gaps.
fn summarize(results: []const ClientResult, itl: []f64, wall_s: f64, max_tokens: u32) !Row {
    var total: u64 = 0;
    var unstored: u64 = 0;
    var short: u32 = 0;
    var ttft_sum: f64 = 0;
    var n_itl: usize = 0;
    for (results) |*r| {
        if (!r.ok) return error.BenchClientFailed;

        total += r.n_tokens;
        if (r.n_tokens < max_tokens) {
            short += 1;
        }

        unstored += r.n_tokens - @min(r.n_tokens, r.arrivals.len);
        ttft_sum += ms(r.t_sent, r.arrivals[0]);
        n_itl += storedGaps(r.arrivals, r.n_tokens, itl[n_itl..]);
    }

    std.debug.assert(n_itl <= itl.len);
    std.mem.sort(f64, itl[0..n_itl], {}, std.sort.asc(f64));

    const n_clients: f64 = @floatFromInt(results.len);
    const total_tokens: f64 = @floatFromInt(total);
    return .{
        .tokens = total,
        .wall_s = wall_s,
        .agg_tok_s = total_tokens / wall_s,
        .per_sess = total_tokens / wall_s / n_clients,
        .ttft_ms = ttft_sum / n_clients,
        .p50_ms = percentile(itl[0..n_itl], 50),
        .p99_ms = percentile(itl[0..n_itl], 99),
        .unstored = unstored,
        .store = max_tokens,
        .short = short,
    };
}

/// One session, one turn, and the per-token arrival timestamps. A failure
/// surfaces as `ok = false`. The row then fails, because a silent partial row
/// reports a number that is not the measurement.
fn clientTask(
    gpa: Allocator,
    io: Io,
    addr: Io.net.IpAddress,
    prompt: []const u8,
    max_tokens: u32,
    store: []Io.Clock.Timestamp,
) ClientResult {
    std.debug.assert(store.len >= 1);

    var r: ClientResult = .{ .arrivals = store };
    var client = Client.connectTcp(gpa, io, addr) catch return r;
    defer client.deinit();

    const sid = client.openSession("", protocol.Sampling.greedy) catch return r;
    r.t_sent = Io.Clock.Timestamp.now(io, .awake);
    _ = client.generate(sid, prompt, max_tokens, null) catch return r;

    // The server ends every turn with `.done`, and a failed read ends the loop.
    while (true) {
        const ev = client.next() catch return r;
        switch (ev.msg) {
            .token => {
                if (r.n_tokens < r.arrivals.len) {
                    r.arrivals[r.n_tokens] = Io.Clock.Timestamp.now(io, .awake);
                }

                r.n_tokens += 1;
            },
            .done => {
                r.ok = r.n_tokens > 0;
                return r;
            },
            else => return r,
        }
    }
}

/// Write the gaps between the stored arrivals of one session to `out`, and
/// return the gap count. The reads stop at the store, whatever `n_tokens` says.
fn storedGaps(arrivals: []const Io.Clock.Timestamp, n_tokens: u32, out: []f64) usize {
    const stored = @min(n_tokens, arrivals.len);
    if (stored < 2) return 0;

    const n_gaps = stored - 1;
    std.debug.assert(out.len >= n_gaps);

    for (arrivals[1..stored], arrivals[0..n_gaps], out[0..n_gaps]) |t_now, t_prev, *gap| {
        gap.* = ms(t_prev, t_now);
    }

    return n_gaps;
}

fn percentile(sorted: []const f64, pct: usize) f64 {
    if (sorted.len == 0) return 0;

    const idx = @min(sorted.len - 1, (sorted.len * pct) / 100);
    return sorted[idx];
}

fn ms(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) f64 {
    const ns: i64 = @intCast(t0.durationTo(t1).raw.toNanoseconds());
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

// -- tests ------------------------------------------------------------------

fn atMs(t: i96) Io.Clock.Timestamp {
    return .{ .raw = .{ .nanoseconds = t * std.time.ns_per_ms }, .clock = .awake };
}

test "textOfTokens: the server encoding of the text holds the asked count" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const seed = "The market square fills with traders, and";
    for ([_][]const u8{ "SmolLM2-135M-Instruct-Q8_0", "Qwen3.5-2B-Q8_0" }) |name| {
        const path = try zigferencer.test_fixtures.path(arena, name);
        var loaded = try loader.load(std.testing.io, arena, path);
        defer loaded.close();

        var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
        for ([_]u32{ 2, 16, 512 }) |n| {
            const text = try textOfTokens(arena, &v, seed, n);
            const ids = try plainIds(arena, &v, text, true);
            try std.testing.expectEqual(@as(usize, n), ids.len);
        }
    }
}

test "sessionRows: the powers of two below the most, then the most" {
    var out: [rows_max]u16 = undefined;

    try std.testing.expectEqualSlices(u16, &.{1}, sessionRows(1, false, &out));
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 4, 5 }, sessionRows(5, false, &out));
    try std.testing.expectEqualSlices(u16, &.{ 1, 2, 4, 8 }, sessionRows(8, false, &out));
    try std.testing.expectEqualSlices(u16, &.{12}, sessionRows(12, true, &out));

    const cap: u16 = @intCast(limits.max_connections);
    const curve = sessionRows(cap, false, &out);
    try std.testing.expectEqual(cap, curve[curve.len - 1]);
    try std.testing.expect(curve.len <= rows_max);
}

test "storedGaps: reads stop at the store when more tokens arrived" {
    const arrivals = [_]Io.Clock.Timestamp{ atMs(10), atMs(11), atMs(13) };
    var out: [8]f64 = @splat(-1);

    // Five tokens arrived and the store kept three, so two gaps and no read
    // past the store.
    try std.testing.expectEqual(@as(usize, 2), storedGaps(&arrivals, 5, &out));
    try std.testing.expectEqualSlices(f64, &.{ 1, 2 }, out[0..2]);
    try std.testing.expectEqual(@as(f64, -1), out[2]);

    try std.testing.expectEqual(@as(usize, 1), storedGaps(&arrivals, 2, &out));
    try std.testing.expectEqual(@as(usize, 0), storedGaps(&arrivals, 1, &out));
    try std.testing.expectEqual(@as(usize, 0), storedGaps(&arrivals, 0, &out));
}
