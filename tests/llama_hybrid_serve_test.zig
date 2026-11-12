//! Serve gate for the llama hybrid runner under the real serve stack. The session pool is
//! `Residency(Sessions(LlamaHybrid))`, the same type `serve_boot` builds for `--gpu-layers
//! N`. The driver is the reference client over an in-process server on an ephemeral TCP port.
//!
//! Claim. The load-bearing assertion is the one `serve_integration_test` makes of the CPU runner,
//! and it is exact. A turn that the server runs while another session decodes in the same batch
//! produces the same token stream the turn produces alone.
//!
//! Failure modes. A two-tier runner has two ways to break that claim. A batched pass can address
//! the wrong tier for a row. An over-subscribed pool can lose a region across the evict and restore
//! round trip.
//!
//! Tests. The first test pins the batched pass at full residency. The second over-subscribes the
//! pool and takes a second turn on every open session. Two slots cannot hold three sessions, so at
//! least one turn decodes over a restored blob, and the swap counters prove the swap really ran.
//! The second test runs once per KV dtype, because f16 halves every region the swap moves. Each
//! dtype states its claim against its own solo references, and no byte or token crosses the two
//! runs.
//!
//! Hardware. Every runner here builds at a true mid split, so a single-tier runner cannot pass this
//! file. The tests need a Vulkan device and the canonical SmolLM2-135M-Instruct-Q8_0 fixture from
//! `test_fixtures.path`. They return `error.SkipZigTest` when the device is absent, or when the
//! device budget cannot fund the placement.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");

const bpe = zigferencer.bpe;
const dtype = zigferencer.dtype;
const event = zigferencer.serve_event;
const kv_budget = zigferencer.kv_budget;
const limits = zigferencer.limits;
const loader = zigferencer.loader;
const protocol = zigferencer.serve_protocol;
const server_mod = zigferencer.serve_server;
const tensor_lookup = zigferencer.tensor_lookup;
const vocab_mod = zigferencer.vocab;
const vulkan_ctx = zigferencer.vulkan_ctx;

const Client = zigferencer.serve_client.Client;
const LlamaHybrid = zigferencer.family.Of(.llama).Hybrid;
const Residency = zigferencer.serve_residency.Residency;
const Sessions = zigferencer.session.Sessions;

const SessionPool = Residency(Sessions(LlamaHybrid));
const HybridServer = server_mod.Server(SessionPool, event.Codec);

// -- fixtures ---------------------------------------------------------------

/// Two VRAM slots and 64 positions hold a short prompt plus its decode. Both
/// stay small, because this gate runs on a fragile GPU stack.
const n_slots: u32 = 2;
const window: u32 = 64;
const turn_tokens_max: u32 = 12;

const prompts = [_][]const u8{
    "The dragon flew over",
    "Tell me a story about",
    "Once upon a time",
};

/// The second turn every session takes in the over-subscription gate. One
/// continuation text serves all three, because the first turns already made
/// the conversations distinct.
const second_prompt: []const u8 = " and then";

/// One thread everywhere, so the hybrid's host prefix runs the same work split
/// on every stream this file compares.
const cpu_cfg: test_policy.Policy = .{ .n_threads = 1 };

fn initCtx(arena: std.mem.Allocator) !*vulkan_ctx.Context {
    const ctxp = try arena.create(vulkan_ctx.Context);
    ctxp.* = vulkan_ctx.Context.init(arena, .{}) catch |e| switch (e) {
        error.NoVulkanDevice,
        error.NoComputeQueue,
        error.InitializationFailed,
        error.IncompatibleDriver,
        error.FeatureNotPresent,
        => return error.SkipZigTest,
        else => return e,
    };
    return ctxp;
}

/// Build a hybrid at a true mid split through the production planner. A budget refusal skips.
/// The KV dtype prices the caps, the plan, and the runner together, because the runner preflight
/// asserts that the plan states its own reservations.
fn initHybrid(
    gpa: std.mem.Allocator,
    ctxp: *const vulkan_ctx.Context,
    loaded: *const loader.Loaded,
    cpu_pool: *test_policy.Pool,
    kv_dtype: dtype.KvDtype,
) !*LlamaHybrid {
    const card = loaded.card;
    const n_layers: u32 = card.n_layers;
    std.debug.assert(n_layers >= 2); // a mid split needs a layer on each side
    const caps = zigferencer.family.Of(.llama).Fused.kvCaps(
        loaded,
        limits.contextCeiling(loaded.card.context_length, null),
        ctxp.deviceLocalAvailableBytes(),
        ctxp.driver,
        n_slots,
        kv_dtype,
    );
    const kv_costs = zigferencer.kv_cost.llamaCostVector(card, kv_dtype);
    var table: [limits.max_layers]u64 = undefined;
    const plan = kv_budget.planLayerSplit(
        caps,
        tensor_lookup.layerWeightBytes(loaded.tensors, n_layers, &table),
        tensor_lookup.headWeightBytes(loaded.tensors),
        .{
            .n_slots = n_slots,
            .window = window,
            .costs = kv_costs,
        },
        // These runners evict and restore, so the plan prices both swap
        // stages. The RAM check stays off, because the test fixes no host.
        .{ .ram_split = .{ .blobs = 0, .prefix = 0 }, .residency = true },
        n_layers / 2,
    ) catch return error.SkipZigTest;
    const hybrid = LlamaHybrid.init(
        gpa,
        ctxp,
        loaded,
        .{
            .plan = plan,
            .pool = cpu_pool,
            .policy = cpu_cfg,
            .kv_dtype = kv_dtype,
        },
    ) catch |e| switch (e) {
        error.VramBudgetExceeded => return error.SkipZigTest,
        else => return e,
    };
    std.debug.assert(hybrid.host_layers > 0);
    std.debug.assert(hybrid.host_layers < hybrid.n_layers);
    return hybrid;
}

/// id -> displayable bytes for the whole vocab, the codec's text side.
fn buildPieces(arena: std.mem.Allocator, v: *const vocab_mod.Vocab) ![]const []const u8 {
    const pieces = try arena.alloc([]const u8, v.tokens.len);
    for (pieces, 0..) |*piece, id| {
        piece.* = try bpe.decodeAlloc(v, arena, &[_]u32{@intCast(id)});
        try std.testing.expect(piece.len <= limits.max_token_text_bytes);
    }

    return pieces;
}

fn runServerTask(server: *HybridServer) void {
    server.run() catch {};
}

const Stream = struct {
    ids: [turn_tokens_max]u32 = undefined,
    n: usize = 0,
    reason: protocol.FinishReason = .eos,
};

/// One whole turn on an otherwise-quiet connection.
fn oneTurn(client: *Client, sid: u32, prompt: []const u8) !Stream {
    _ = try client.generate(sid, prompt, turn_tokens_max, null);
    var s: Stream = .{};
    var text_buf: [4096]u8 = undefined;
    const t = try client.collectTurn(s.ids[0..], &text_buf);
    s.n = @min(t.n_tokens, s.ids.len);
    s.reason = t.finish_reason;
    return s;
}

/// Take solo turns per prompt on a quiet server and close each session after it.
/// `out2 = null` takes one turn per session. Otherwise a second turn with `second_prompt`
/// follows the first on the same session.
fn soloReference(client: *Client, out: []Stream, out2: ?[]Stream) !void {
    for (out, 0..) |*ref, i| {
        const sid = try client.openSession("", protocol.Sampling.greedy);
        ref.* = try oneTurn(client, sid, prompts[i]);
        try std.testing.expect(ref.n > 0);
        if (out2) |second| {
            second[i] = try oneTurn(client, sid, second_prompt);
            try std.testing.expect(second[i].n > 0);
        }

        _ = try client.send(.session_close, sid);
        const ev = try client.next();
        try std.testing.expect(ev.msg == .session_closed);
    }
}

/// Put every prompt in flight at once and demultiplex the events by session id.
/// The sessions stay open, and `sids` reports them, so a caller can take more
/// turns on the same conversations.
fn interleaved(client: *Client, out: []Stream, sids: []u32) !void {
    std.debug.assert(out.len <= prompts.len);
    std.debug.assert(sids.len == out.len);

    for (sids) |*sid| {
        sid.* = try client.openSession("", protocol.Sampling.greedy);
    }

    for (sids, 0..) |sid, i| {
        _ = try client.generate(sid, prompts[i], turn_tokens_max, null);
    }

    for (out) |*s| {
        s.* = .{};
    }

    var done_count: usize = 0;
    while (done_count < out.len) {
        const ev = try client.next();
        const slot: usize = for (sids, 0..) |sid, i| {
            if (ev.header.session_id == sid) break i;
        } else return error.UnknownSession;

        switch (ev.msg) {
            .token => |t| {
                try std.testing.expect(out[slot].n < out[slot].ids.len);
                out[slot].ids[out[slot].n] = t.token_id;
                out[slot].n += 1;
            },
            .done => |d| {
                out[slot].reason = d.finish_reason;
                done_count += 1;
            },
            else => return error.UnexpectedReply,
        }
    }
}

fn expectSameStreams(reference: []const Stream, got: []const Stream) !void {
    for (reference, got) |ref, g| {
        try std.testing.expectEqual(ref.reason, g.reason);
        try std.testing.expectEqualSlices(u32, ref.ids[0..ref.n], g.ids[0..g.n]);
    }
}

/// The over-subscription round trip at one KV dtype. A swapped stream must equal an unswapped
/// stream, against the solo references of the same dtype.
fn overSubscribedRoundTrip(kv_dtype: dtype.KvDtype) !void {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const codec: event.Codec = .{
        .vocab = &v,
        .kind = .none,
        .pieces = try buildPieces(arena, &v),
        .eos_id = v.eos_id,
    };

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, cpu_cfg);
    defer cpu_pool.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, &cpu_pool, kv_dtype);
    defer hybrid.deinit();

    // Three logical sessions over two VRAM slots: the third forces a swap on every tick.
    // The blob budget funds the evicted session plus the transient blob of a rotation.
    var sessions = try Sessions(LlamaHybrid).init(arena, hybrid);
    const blob_budget: u64 = 2 * hybrid.evictSlotBytesFor(hybrid.max_seq);
    var pool = try SessionPool.init(arena, gpa, &sessions, prompts.len, blob_budget);
    defer pool.deinit();

    var server = HybridServer.init(arena, gpa, io, &pool, &codec, .{
        .max_sessions = prompts.len,
        .max_batch = @intCast(n_slots),
        .max_seq = hybrid.max_seq,
        .max_step_rows = @min(n_slots + 1, hybrid.maxStepRows()),
        .allow_shutdown = true,
        .model_name = "fixture",
        .vocab_size = loaded.card.vocab_size,
        .eos_id = v.eos_id,
    }, .{ .tcp = .{} }) catch return error.SkipZigTest;

    var run_future = try io.concurrent(runServerTask, .{&server});
    var client = try Client.connectTcp(gpa, io, server.boundAddress());
    defer client.deinit();
    defer {
        client.shutdown() catch blk: {
            var fallback = Client.connectTcp(gpa, io, server.boundAddress()) catch break :blk;
            defer fallback.deinit();

            fallback.shutdown() catch {};
        };
        run_future.await(io);
    }

    _ = try client.hello();

    // The solo conversations close each session, so at most one sits resident
    // and the reference streams cost no swap.
    var reference: [prompts.len]Stream = undefined;
    var reference2: [prompts.len]Stream = undefined;
    try soloReference(&client, &reference, &reference2);
    const quiet = pool.stats();

    var got: [prompts.len]Stream = undefined;
    var sids: [prompts.len]u32 = undefined;
    try interleaved(&client, &got, &sids);
    try expectSameStreams(&reference, &got);

    // A second turn on every open session. Two slots cannot hold three, so the pigeonhole
    // forces at least one restored blob, whatever the tick timing.
    var got2: [prompts.len]Stream = undefined;
    for (&got2, sids) |*s, sid| {
        s.* = try oneTurn(&client, sid, second_prompt);
    }

    try expectSameStreams(&reference2, &got2);

    // Without a swap the comparisons above say nothing, so read the counters.
    const busy = pool.stats();
    try std.testing.expect(busy.evicts > quiet.evicts);
    try std.testing.expect(busy.restores > quiet.restores);
}

// -- tests ------------------------------------------------------------------

test "llama hybrid serve: an interleaved turn matches its solo stream" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const codec: event.Codec = .{
        .vocab = &v,
        .kind = .none,
        .pieces = try buildPieces(arena, &v),
        .eos_id = v.eos_id,
    };

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, cpu_cfg);
    defer cpu_pool.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, &cpu_pool, .f32);
    defer hybrid.deinit();

    // Full residency: no swap can run, so the batched pass alone answers for the streams.
    var sessions = try Sessions(LlamaHybrid).init(arena, hybrid);
    var pool = try SessionPool.init(arena, gpa, &sessions, 2, null);
    defer pool.deinit();

    var server = HybridServer.init(arena, gpa, io, &pool, &codec, .{
        .max_sessions = 2,
        .max_batch = @intCast(n_slots),
        .max_seq = hybrid.max_seq,
        .max_step_rows = @min(n_slots + 1, hybrid.maxStepRows()),
        .allow_shutdown = true,
        .model_name = "fixture",
        .vocab_size = loaded.card.vocab_size,
        .eos_id = v.eos_id,
    }, .{ .tcp = .{} }) catch return error.SkipZigTest;

    var run_future = try io.concurrent(runServerTask, .{&server});
    var client = try Client.connectTcp(gpa, io, server.boundAddress());
    defer client.deinit();
    // Shut the server down on every exit path, so a failed assertion cannot block the process.
    // Use a fresh connection if the main one is the casualty.
    defer {
        client.shutdown() catch blk: {
            var fallback = Client.connectTcp(gpa, io, server.boundAddress()) catch break :blk;
            defer fallback.deinit();

            fallback.shutdown() catch {};
        };
        run_future.await(io);
    }

    _ = try client.hello();

    var reference: [2]Stream = undefined;
    try soloReference(&client, &reference, null);
    var got: [2]Stream = undefined;
    var sids: [2]u32 = undefined;
    try interleaved(&client, &got, &sids);
    try expectSameStreams(&reference, &got);
}

test "llama hybrid serve: an over-subscribed pool evicts and restores" {
    try overSubscribedRoundTrip(.f32);
}

test "llama hybrid serve: an over-subscribed pool evicts and restores at f16" {
    // Half-width KV goes through the same two-tier swap. A dropped or mis-sized region
    // diverges a second turn from its solo reference.
    try overSubscribedRoundTrip(.f16);
}
