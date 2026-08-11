//! `zigferencer bench-serve <pub_name>` draws the batching-win curve. For each
//! batch size B it starts a fresh in-process server on an ephemeral loopback
//! port. It then drives B concurrent sessions, one connection and one turn each,
//! through the reference client. It reports the aggregate throughput and the
//! inter-token tail latency. Decoding is greedy and the prompts are identical, so
//! every session generates the same stream at every B under the engine's
//! batch-equivalence invariant. The rows therefore differ only in concurrency,
//! and the curve isolates the weight-streaming amortization.
//!
//! The turn is a raw continuation and carries no chat template. An
//! instruct-template turn can hit EOS after a few tokens and truncate the
//! measurement.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");

const cli = @import("command.zig");
const args = @import("args.zig");
const flags_mod = @import("flags.zig");
const style = @import("style.zig");
const select = @import("select.zig");
const serve_common = @import("serve_common.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;
const LlamaRunner = @import("../runtime/llama_runner.zig").LlamaRunner;
const QwenRunner = @import("../runtime/qwen_runner.zig").QwenRunner;
const Sessions = @import("../runtime/session.zig").Sessions;
const prefill_chunk = @import("../serve/prefill_chunk.zig");
const event = @import("../serve/event.zig");
const protocol = @import("../serve/protocol.zig");
const server_mod = @import("../serve/server.zig");
const Client = @import("../serve/client.zig").Client;
const log = std.log.scoped(.bench_serve);

const llama_fused = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/llama_fused.zig")
else
    void;
const qwen_fused = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/qwen_fused.zig")
else
    void;
const vulkan_ctx_mod = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/vulkan_ctx.zig")
else
    void;

// -- types ------------------------------------------------------------------

const batch_sizes = [_]u16{ 1, 2, 4, 8 };

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
    pub const omit: []const []const u8 = &.{ "kv_dtype", "gpu_layers", "thinking", "thinking_budget" };
    pub const help: args.Help = .{
        .usage = "zigferencer-dev bench-serve <pub_name> [flags]",
        .about = "Drive B in {1, 2, 4, 8} concurrent sessions, fresh server per row. Report tok/s + ITL tails.\n",
        .footer =
        \\examples:
        \\  zigferencer-dev bench-serve SmolLM2-135M-Instruct-Q8_0 --max-tokens 64
        \\
        ,
    };
    pub const docs = .{
        .prompt = args.Doc{ .text = "the turn every session sends", .default = "a fixed continuation" },
        .max_tokens = args.Doc{ .text = "the most tokens one turn generates, 1..max-seq" },
        .max_seq = args.Doc{
            .group = "kv",
            .text = "KV positions per session",
            .default = std.fmt.comptimePrint("the {d} MiB session budget at f32, capped by the\nmodel ceiling", .{limits.serve_session_kv_bytes >> 20}),
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    const prompt = flags.prompt;
    const max_tokens = flags.max_tokens;
    const max_seq_flag = flags.engine.max_seq;
    // The fused build benches the whole-model fused runners alone, which run
    // no CPU op, so its pool starts no workers.
    const policy = try serve_common.resolvePolicy(ctx.io, flags.engine.threads.policyFlags(), !build_options.use_vulkan_fused_ops);
    const model = try select.model(ctx, flags.pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();
    if (loaded.card.arch != .llama and loaded.card.arch != .qwen35) {
        log.err("bench-serve drives llama-family and qwen3.5 models (model is {t})", .{loaded.card.arch});
        return error.UnsupportedServeArch;
    }
    // One ceiling per boot, and the serve default of the model when no flag
    // names the window, the way serve resolves both. bench-serve carries no
    // --kv-dtype flag, so the default prices the f32 identity every arm runs.
    const ceiling = limits.contextCeiling(loaded.card.context_length, model.context_limit);
    const serve_default = kv_budget.defaultWindows(
        kv_budget.costModel(loaded.card.kvShape(), .f32),
        ceiling,
    ).serve;
    const max_seq = serve_common.baseWindow(max_seq_flag, ceiling, serve_default) catch |err| switch (err) {
        error.WindowOutOfRange => return args.fail(ctx.io, serve_common.window_out_of_range_fmt, .{ceiling}),
    };
    // A turn ends at the window, so the window bounds the tokens a turn emits.
    if (max_tokens == 0 or max_tokens > max_seq) {
        return args.fail(ctx.io, "--max-tokens must be in 1..{d}, the --max-seq window", .{max_seq});
    }

    // One pool serves every CPU op of the command. Its workers live until
    // the command returns.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, policy);
    defer pool.deinit();

    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);
    const pieces = try ctx.arena.alloc([]const u8, v.tokens.len);
    for (pieces, 0..) |*piece, id| {
        piece.* = try bpe.decodeAlloc(&v, ctx.arena, &[_]u32{@intCast(id)});
        if (piece.len > limits.max_token_text_bytes) return error.TokenPieceTooLong;
    }
    const codec: event.Codec = .{ .vocab = &v, .kind = .none, .pieces = pieces, .eos_id = v.eos_id };

    var stdout_buf: [4096]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stdout(), ctx.io, &stdout_buf);
    const w = &fw.interface;
    try w.print("model:    {s}\n", .{model.pub_name});
    try w.print("backend:  {s}  threads: {d}  (min_parallel_flops={d})\n", .{
        build_options.ops_backend, policy.n_threads, policy.min_parallel_flops,
    });
    try w.print("turn:     \"{s}\" -> up to {d} tokens, greedy, no template\n", .{ prompt, max_tokens });
    try w.flush();

    // The engine matrix matches serve. A CPU build benches the CPU runners with a
    // fresh engine per row. The fused build benches the GPU engines with one
    // engine, so the weights upload once, and a fresh server per row. Per-op
    // vulkan serves nothing, so it benches nothing.
    if (comptime build_options.use_vulkan_ops) {
        log.err("bench-serve does not run on the per-op vulkan backend. Build -Dops-backend=vulkan_fused or a CPU backend", .{});
        return error.UnsupportedServeBackend;
    }
    if (comptime build_options.use_vulkan_fused_ops) {
        const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
        ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
        defer ctxp.deinit();
        // The curve holds one engine for every row, so it plans the widest row.
        // Both engines take the one window above. Neither reads a device
        // budget. bench-serve carries no --kv-dtype flag, so both arms price
        // the f32 identity.
        if (loaded.card.arch == .qwen35) {
            const caps = qwen_fused.QwenFusedRunner.kvCaps(&loaded, ceiling, 0, limits.max_batch, .f32);
            const plan = try kv_budget.kvPlan(caps, qwen_fused.QwenFusedRunner.kvCostModel(loaded.card, .f32), .{
                .window = max_seq,
                .slots = limits.max_batch,
                .batch = limits.max_batch,
            });
            serve_common.logPlan(plan, .f32);
            const fused = try qwen_fused.QwenFusedRunner.init(ctx.gpa, ctxp, &loaded, plan, .f32);
            defer fused.deinit();
            var engine = try Sessions(qwen_fused.QwenFusedRunner).init(ctx.arena, fused);
            return benchCurveOver(Sessions(qwen_fused.QwenFusedRunner), ctx, w, &engine, &codec, loaded.card.vocab_size, fused.max_seq, prompt, max_tokens);
        }
        // The llama runner keeps a spare stream, so it never plans fewer than
        // two slots. The window is the one above, as in the qwen arm.
        const n_slots: u32 = @max(2, limits.max_batch);
        const caps = try llama_fused.LlamaFusedRunner.kvCaps(ctx.gpa, &loaded, ceiling, 0, n_slots, .f32);
        const plan = try kv_budget.kvPlan(caps, llama_fused.LlamaFusedRunner.kvCostModel(loaded.card, .f32), .{
            .window = max_seq,
            .slots = n_slots,
            .batch = limits.max_batch,
        });
        serve_common.logPlan(plan, .f32);
        const fused = try llama_fused.LlamaFusedRunner.initFromLoaded(ctx.gpa, ctxp, &loaded, plan, .f32);
        defer fused.deinit();
        var engine = try Sessions(llama_fused.LlamaFusedRunner).init(ctx.arena, fused);
        return benchCurveOver(Sessions(llama_fused.LlamaFusedRunner), ctx, w, &engine, &codec, loaded.card.vocab_size, fused.max_seq, prompt, max_tokens);
    }
    if (loaded.card.arch == .qwen35) {
        return benchCurve(QwenRunner, ctx, w, &loaded, &codec, &pool, policy, max_seq, prompt, max_tokens);
    }
    return benchCurve(LlamaRunner, ctx, w, &loaded, &codec, &pool, policy, max_seq, prompt, max_tokens);
}

// -- helpers ----------------------------------------------------------------

/// The CPU curve builds a fresh runner and engine per row, which gives clean KV
/// pools and no stacking.
fn benchCurve(
    comptime RunnerT: type,
    ctx: cli.Context,
    w: *Io.Writer,
    loaded: *const loader.Loaded,
    codec: *const event.Codec,
    pool: *Pool,
    policy: parallel.Policy,
    max_seq: u32,
    prompt: []const u8,
    max_tokens: u32,
) !void {
    try printTableHead(w, max_seq);
    for (batch_sizes) |b| {
        if (b > limits.max_batch) break;
        var round_arena_state = std.heap.ArenaAllocator.init(ctx.gpa);
        defer round_arena_state.deinit();
        var runner = try RunnerT.init(round_arena_state.allocator(), loaded, kv_budget.pinned(b, max_seq), prefill_chunk.stepRows(prefill_chunk.cpu_rule, b, b), ctx.io, pool, policy, .f32);
        var engine = try Sessions(RunnerT).init(round_arena_state.allocator(), &runner);
        const row = try benchOne(Sessions(RunnerT), ctx, &engine, prefill_chunk.cpu_rule, codec, loaded.card.vocab_size, max_seq, b, prompt, max_tokens);
        try printRow(w, b, row);
    }
}

/// The GPU curve holds one engine for the whole curve, so the weights upload
/// once. The rows reuse slots: a session closes at the end of a row, and `create`
/// re-zeros any recurrent state, which is the same slot-reuse rule serve relies
/// on.
fn benchCurveOver(
    comptime EngineT: type,
    ctx: cli.Context,
    w: *Io.Writer,
    engine: *EngineT,
    codec: *const event.Codec,
    vocab_size: u32,
    max_seq: u32,
    prompt: []const u8,
    max_tokens: u32,
) !void {
    try printTableHead(w, max_seq);
    for (batch_sizes) |b| {
        if (b > limits.max_batch) break;
        const row = try benchOne(EngineT, ctx, engine, .{ .derived = engine.runner.maxStepRows() }, codec, vocab_size, max_seq, b, prompt, max_tokens);
        try printRow(w, b, row);
    }
}

/// The table head names the window the engine holds, which the fused arms
/// read from the runner.
fn printTableHead(w: *Io.Writer, max_seq: u32) !void {
    try w.print("max_seq:  {d}\n", .{max_seq});
    try w.print("note:     prefill runs on decode ticks. Concurrency shows in ttft, not itl tails\n\n", .{});
    try w.print("  B   tokens   wall_s   agg_tok/s   per_session   ttft_ms   itl_p50_ms   itl_p99_ms\n", .{});
    try w.flush();
}

fn printRow(w: *Io.Writer, b: u16, row: Row) !void {
    try w.print("  {d}   {d:>6}   {d:>6.1}   {d:>9.2}   {d:>11.2}   {d:>7.0}   {d:>10.1}   {d:>10.1}\n", .{
        b,            row.tokens,  row.wall_s, row.agg_tok_s,
        row.per_sess, row.ttft_ms, row.p50_ms, row.p99_ms,
    });
    if (row.unstored > 0) {
        try w.print("      cut: {d} token arrivals passed the store of {d} per session. The ITL tails omit them\n", .{ row.unstored, row.store });
    }
    try w.flush();
}

fn benchOne(
    comptime EngineT: type,
    ctx: cli.Context,
    engine: *EngineT,
    rule: prefill_chunk.Rule,
    codec: *const event.Codec,
    vocab_size: u32,
    max_seq: u32,
    b: u16,
    prompt: []const u8,
    max_tokens: u32,
) !Row {
    const ServerT = server_mod.Server(EngineT, event.Codec);
    // A fresh server per row gives clean queues and a clean port, and it lives in
    // a round arena.
    var round_arena_state = std.heap.ArenaAllocator.init(ctx.gpa);
    defer round_arena_state.deinit();
    const arena = round_arena_state.allocator();

    // The stores come from the round arena before the server and the clients
    // start, because the arena is not thread-safe.
    std.debug.assert(b <= limits.max_batch);
    var stores: [limits.max_batch][]Io.Clock.Timestamp = undefined;
    for (stores[0..b]) |*store| store.* = try arena.alloc(Io.Clock.Timestamp, max_tokens);
    const itl = try arena.alloc(f64, @as(usize, b) * max_tokens);

    var server = try ServerT.init(arena, ctx.gpa, ctx.io, engine, codec, .{
        .max_sessions = b,
        .max_batch = b,
        .max_seq = max_seq,
        .max_step_rows = engine.runner.maxStepRows(),
        .prefill = rule,
        .allow_shutdown = true,
        .model_name = "bench",
        .vocab_size = vocab_size,
        .eos_id = codec.eos_id,
    }, .{ .tcp = .{} });
    var server_future = try ctx.io.concurrent(ServerTask(ServerT).run, .{&server});
    const addr = server.boundAddress();

    const t0 = Io.Clock.Timestamp.now(ctx.io, .awake);
    var futures: [limits.max_batch]Io.Future(ClientResult) = undefined;
    var i: usize = 0;
    while (i < b) : (i += 1) {
        futures[i] = try ctx.io.concurrent(clientTask, .{ ctx.gpa, ctx.io, addr, prompt, max_tokens, stores[i] });
    }
    var results: [limits.max_batch]ClientResult = undefined;
    i = 0;
    while (i < b) : (i += 1) results[i] = futures[i].await(ctx.io);
    const t1 = Io.Clock.Timestamp.now(ctx.io, .awake);

    // The server stops through the protocol, the way any client stops it.
    var control = try Client.connectTcp(ctx.gpa, ctx.io, addr);
    try control.shutdown();
    control.deinit();
    server_future.await(ctx.io);

    var total: u64 = 0;
    var unstored: u64 = 0;
    var ttft_sum: f64 = 0;
    var n_itl: usize = 0;
    for (results[0..b]) |*r| {
        if (!r.ok) return error.BenchClientFailed;
        total += r.n_tokens;
        unstored += r.n_tokens - @min(r.n_tokens, r.arrivals.len);
        ttft_sum += ms(r.t_sent, r.arrivals[0]);
        n_itl += storedGaps(r.arrivals, r.n_tokens, itl[n_itl..]);
    }
    std.debug.assert(n_itl <= itl.len);
    std.mem.sort(f64, itl[0..n_itl], {}, std.sort.asc(f64));
    const wall_s = ms(t0, t1) / 1000.0;
    return .{
        .tokens = total,
        .wall_s = wall_s,
        .agg_tok_s = @as(f64, @floatFromInt(total)) / wall_s,
        .per_sess = @as(f64, @floatFromInt(total)) / wall_s / @as(f64, @floatFromInt(b)),
        .ttft_ms = ttft_sum / @as(f64, @floatFromInt(b)),
        .p50_ms = percentile(itl[0..n_itl], 50),
        .p99_ms = percentile(itl[0..n_itl], 99),
        .unstored = unstored,
        .store = max_tokens,
    };
}

fn ServerTask(comptime ServerT: type) type {
    return struct {
        fn run(server: *ServerT) void {
            server.run() catch {};
        }
    };
}

/// One session, one turn, and the per-token arrival timestamps. A failure
/// surfaces as `ok = false`. The row then fails, because a silent partial row
/// reports a number that is not the measurement.
fn clientTask(gpa: Allocator, io: Io, addr: Io.net.IpAddress, prompt: []const u8, max_tokens: u32, store: []Io.Clock.Timestamp) ClientResult {
    std.debug.assert(store.len >= 1);
    var r: ClientResult = .{ .arrivals = store };
    var client = Client.connectTcp(gpa, io, addr) catch return r;
    defer client.deinit();
    const sid = client.openSession("", protocol.Sampling.greedy) catch return r;
    r.t_sent = Io.Clock.Timestamp.now(io, .awake);
    _ = client.generate(sid, prompt, max_tokens, null) catch return r;
    while (true) {
        const ev = client.next() catch return r;
        switch (ev.msg) {
            .token => {
                if (r.n_tokens < r.arrivals.len) r.arrivals[r.n_tokens] = Io.Clock.Timestamp.now(io, .awake);
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
