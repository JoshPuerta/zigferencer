//! `zigferencer bench-serve <pub_name>` draws the batching-win curve. For each batch
//! size B it starts a fresh in-process server, drives B concurrent sessions of one
//! turn through the reference client, and reports the throughput and the inter-token
//! tails. The turn is raw text, because an instruct template can hit EOS early.
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
const flags_mod = zigferencer.cli.flag_groups;
const select = zigferencer.cli.select;
const boot_facts = zigferencer.cli.boot_facts;
const boot_lines = zigferencer.cli.boot_lines;
const family = zigferencer.family;
const bpe = zigferencer.bpe;
const kv_budget = zigferencer.kv_budget;
const limits = zigferencer.limits;
const registry = zigferencer.registry;
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

const batch_sizes = [_]u16{ 1, 2, 4, 8 };

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
    max_tokens: u32 = 48,
    engine: flags_mod.EngineFlags = .{},

    pub const docs_from = flags_mod.EngineDocs;
    pub const positional = "pub_name";
    pub const omit: []const []const u8 = &.{
        "kv_dtype",
        "gpu_layers",
        "thinking",
        "thinking_budget",
    };
    pub const help: args.Help = .{
        .usage = "zigferencer-dev bench-serve <pub_name> [flags]",
        .about = "Drive B in {1, 2, 4, 8} concurrent sessions, fresh server per row. " ++
            "Report tok/s + ITL tails.\n",
        .footer =
        \\examples:
        \\  zigferencer-dev bench-serve SmolLM2-135M-Instruct-Q8_0 --max-tokens 64
        \\
        ,
    };
    pub const docs = .{
        .prompt = args.Doc{
            .text = "the turn every session sends",
            .default = "a fixed continuation",
        },
        .max_tokens = args.Doc{ .text = "the most tokens one turn generates, 1..max-seq" },
        .max_seq = args.Doc{
            .group = "kv",
            .text = "KV positions per session",
            .default = std.fmt.comptimePrint(
                "the {d} MiB session budget at f32, capped by the\nmodel ceiling",
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
    ceiling: u32,
    max_seq: u32,
    prompt: []const u8,
    max_tokens: u32,
};

fn ServerTask(comptime ServerT: type) type {
    return struct {
        fn run(server: *ServerT) void {
            server.run() catch {};
        }
    };
}

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;

    // The fused build benches the whole-model fused runners alone, which run
    // no CPU op, so its pool starts no workers.
    const thread_flags = flags.engine.threads.policyFlags();
    const policy = try boot_facts.resolvePolicy(ctx.io, thread_flags, !fused_build);

    const model = try select.model(ctx, flags.pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    const ceiling = limits.contextCeiling(loaded.card.context_length, model.context_limit);
    const max_seq = try turnWindow(ctx.io, &flags, loaded.card, ceiling);

    // One pool serves every CPU op of the command.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, policy);
    defer pool.deinit();

    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);
    const pieces = try decodePieces(ctx.arena, &v);
    const codec: event.Codec = .{
        .vocab = &v,
        .kind = .none,
        .pieces = pieces,
        .eos_id = v.eos_id,
    };

    const w = &ctx.stdout.interface;
    try printIntro(w, model.pub_name, policy, &flags);

    if (comptime build_options.use_vulkan_ops) {
        log.err("bench-serve does not run on the per-op vulkan backend. " ++
            "Build -Dops-backend=vulkan_fused or a CPU backend", .{});
        return error.UnsupportedServeBackend;
    }

    const curve: Curve = .{
        .ctx = ctx,
        .w = w,
        .loaded = &loaded,
        .codec = &codec,
        .pool = &pool,
        .policy = policy,
        .ceiling = ceiling,
        .max_seq = max_seq,
        .prompt = flags.prompt,
        .max_tokens = flags.max_tokens,
    };
    switch (loaded.card.arch) {
        inline else => |arch| return benchFamily(family.Of(arch), &curve),
    }
}

// -- helpers ----------------------------------------------------------------

/// The window of every row: the flag, or the serve default of the model. The
/// command has no `--kv-dtype` flag, so the default prices the f32 identity.
fn turnWindow(io: Io, flags: *const Flags, card: loader.ModelCard, ceiling: u32) !u32 {
    const cost = kv_budget.costModel(card.kvShape(), .f32);
    const serve_default = kv_budget.defaultWindows(cost, ceiling).serve;
    const flag = flags.engine.max_seq;
    const max_seq = boot_facts.baseWindow(flag, ceiling, serve_default) catch |err| switch (err) {
        error.WindowOutOfRange => {
            return args.fail(io, boot_lines.window_out_of_range_fmt, .{ceiling});
        },
    };

    // A turn ends at the window, so the window bounds the tokens a turn emits.
    if (flags.max_tokens == 0 or flags.max_tokens > max_seq) {
        return args.fail(io, "--max-tokens must be in 1..{d}, the --max-seq window", .{max_seq});
    }

    return max_seq;
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
    flags: *const Flags,
) !void {
    try w.print("model:    {s}\n", .{model_name});
    try w.print("backend:  {s}  threads: {d}  (min_parallel_flops={d})\n", .{
        build_options.ops_backend, policy.n_threads, policy.min_parallel_flops,
    });
    try w.print("turn:     \"{s}\" -> up to {d} tokens, greedy, no template\n", .{
        flags.prompt, flags.max_tokens,
    });
    try w.flush();
}

fn benchFamily(comptime Family: type, c: *const Curve) !void {
    if (comptime fused_build) return fusedCurve(Family, c);
    return cpuCurve(Family, c);
}

/// A fresh runner and engine per row give clean KV pools and no stacking.
fn cpuCurve(comptime Family: type, c: *const Curve) !void {
    const Runner = Family.Cpu;
    const ctx = c.ctx;

    try printTableHead(c.w, c.max_seq);
    for (batch_sizes) |b| {
        if (b > limits.max_batch) break;

        var round_arena_state = std.heap.ArenaAllocator.init(ctx.gpa);
        defer round_arena_state.deinit();

        const arena = round_arena_state.allocator();

        var runner = try Runner.init(
            arena,
            c.loaded,
            kv_budget.pinned(b, c.max_seq),
            prefill_chunk.stepRows(prefill_chunk.cpu_rule, b, b),
            ctx.io,
            c.pool,
            c.policy,
            .f32,
        );
        const EngineT = Sessions(Runner);
        var engine = try EngineT.init(arena, &runner);
        const row = try benchOne(EngineT, c, &engine, prefill_chunk.cpu_rule, c.max_seq, b);
        try printRow(c.w, b, row);
    }
}

/// One engine holds the whole curve, so the weights upload once, and the plan takes
/// the widest row. The rows reuse slots: a session closes at the end of a row, and
/// `create` re-zeros any recurrent state.
fn fusedCurve(comptime Family: type, c: *const Curve) !void {
    const ctx = c.ctx;
    const card = c.loaded.card;

    const ctxp = try boot_facts.openDevice(ctx);
    defer ctxp.deinit();

    const n_slots: u32 = @max(Family.fused_min_slots, limits.max_batch);
    const caps = try Family.kvCaps(ctx.gpa, c.loaded, c.ceiling, 0, n_slots, .f32);
    const plan = try kv_budget.kvPlan(caps, Family.Fused.kvCostModel(card, .f32), .{
        .window = c.max_seq,
        .slots = n_slots,
        .batch = limits.max_batch,
    });
    boot_lines.logPlan(plan, .f32);

    const fused = try Family.initFused(ctx.gpa, ctxp, c.loaded, plan, .f32);
    defer fused.deinit();

    const EngineT = Sessions(Family.Fused);
    var engine = try EngineT.init(ctx.arena, fused);

    try printTableHead(c.w, fused.max_seq);
    for (batch_sizes) |b| {
        if (b > limits.max_batch) break;

        const rule: prefill_chunk.Rule = .{ .derived = engine.runner.maxStepRows() };
        const row = try benchOne(EngineT, c, &engine, rule, fused.max_seq, b);
        try printRow(c.w, b, row);
    }
}

/// The table head names the window the engine holds.
fn printTableHead(w: *Io.Writer, max_seq: u32) !void {
    try w.print("max_seq:  {d}\n", .{max_seq});
    try w.print("note:     prefill runs on decode ticks. " ++
        "Concurrency shows in ttft, not itl tails\n\n", .{});
    try w.print("  B   tokens   wall_s   agg_tok/s   per_session   ttft_ms   " ++
        "itl_p50_ms   itl_p99_ms\n", .{});
    try w.flush();
}

fn printRow(w: *Io.Writer, b: u16, row: Row) !void {
    try w.print("  {d}   {d:>6}   {d:>6.1}   {d:>9.2}   {d:>11.2}   " ++
        "{d:>7.0}   {d:>10.1}   {d:>10.1}\n", .{
        b,            row.tokens,  row.wall_s, row.agg_tok_s,
        row.per_sess, row.ttft_ms, row.p50_ms, row.p99_ms,
    });
    if (row.unstored > 0) {
        try w.print("      cut: {d} token arrivals passed the store of {d} per session. " ++
            "The ITL tails omit them\n", .{ row.unstored, row.store });
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
) !Row {
    const ServerT = server_mod.Server(EngineT, event.Codec);
    const ctx = c.ctx;
    std.debug.assert(b <= limits.max_batch);

    var round_arena_state = std.heap.ArenaAllocator.init(ctx.gpa);
    defer round_arena_state.deinit();

    const arena = round_arena_state.allocator();

    // The stores come from the round arena before the server and the clients
    // start, because the arena is not thread-safe.
    var stores: [limits.max_batch][]Io.Clock.Timestamp = undefined;
    for (stores[0..b]) |*store| {
        store.* = try arena.alloc(Io.Clock.Timestamp, c.max_tokens);
    }

    const itl = try arena.alloc(f64, @as(usize, b) * c.max_tokens);

    var server = try ServerT.init(arena, ctx.gpa, ctx.io, engine, c.codec, .{
        .max_sessions = b,
        .max_batch = b,
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
    var results: [limits.max_batch]ClientResult = undefined;
    try driveClients(c, addr, stores[0..b], results[0..b]);
    const t1 = Io.Clock.Timestamp.now(ctx.io, .awake);

    // The server stops through the protocol, the way any client stops it.
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
    std.debug.assert(stores.len <= limits.max_batch);

    var futures_buffer: [limits.max_batch]Io.Future(ClientResult) = undefined;
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
    var ttft_sum: f64 = 0;
    var n_itl: usize = 0;
    for (results) |*r| {
        if (!r.ok) return error.BenchClientFailed;

        total += r.n_tokens;
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
