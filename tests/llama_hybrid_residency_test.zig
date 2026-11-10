//! Swap gate for the llama hybrid runner. The tiered KV swap must restore a conversation byte for
//! byte, and the provenance tag must refuse a blob that another host layer count or dtype wrote.
//!
//! Two tiers. A hybrid slot lives in two tiers. The host prefix holds layers `[0..k)` and the
//! device suffix holds layers `[k..L)`, so two walks write one blob. The state model proves that
//! algebra at anchor scale, and these tests replay it at model scale through the public surface of
//! the runner.
//!
//! Witnesses. The blob size never moves with the host layers, and "the blob is split-independent"
//! witnesses that across a CPU runner and a hybrid runner. The two walks together write the
//! whole-model framing, and "one blob carries the whole-model framing" witnesses it three ways:
//!
//!   1. The host half is byte-equal to the half of the CPU runner, because the prefix boundary runs
//!      the same ops over the same weights.
//!   2. A CPU blob goes through the H2D walk of the hybrid in and its D2H walk out byte for byte.
//!      The transport therefore writes and recovers every region. The shipped fingerprint forbids
//!      this cross-placement restore, and the test does it on purpose, because only a foreign blob
//!      proves that the walks address the whole-model framing and not merely each other.
//!   3. A greedy decode over the restored rows continues on the path of the CPU runner, so the
//!      regions sit where attention expects them. A region that the walk misplaces garbles the
//!      logits here, even though the transport check cannot see it.
//!
//! Bits. The device halves hold the same values in the same places, but not the same bits. The
//! GPU-computed suffix KV differs from the CPU-computed suffix KV in the low mantissa, because the
//! reduce kernel sums in a different order. Byte equality across runners is therefore not a
//! guarantee this engine makes, and that guarantee is the reason the snapshot fingerprint folds the
//! host layers. A blob must not restore under a runner that did not write it.
//!
//! Interior count. Every runner here builds with an interior count of host layers. A whole-model
//! count lets a single-tier runner pass this whole file, so `initHybrid` asserts the count it got.
//!
//! Hardware. The hardware tests need a Vulkan device and the canonical SmolLM2-135M-Instruct-Q8_0
//! fixture from `test_fixtures.path`. They return `error.SkipZigTest` when the device is absent.
//! The two provenance tests are pure, so they run wherever this file compiles.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");

const bpe = zigferencer.bpe;
const kv_budget = zigferencer.kv_budget;
const limits = zigferencer.limits;
const loader = zigferencer.loader;
const prompt_cache = zigferencer.prompt_cache;
const tensor_lookup = zigferencer.tensor_lookup;
const vocab_mod = zigferencer.vocab;
const vulkan_ctx = zigferencer.vulkan_ctx;

const LlamaRow = zigferencer.family.Of(.llama);
const LlamaHybrid = LlamaRow.Hybrid;
const LlamaCpu = zigferencer.family.Of(.llama).Cpu;
const Sessions = zigferencer.session.Sessions;

// -- fixtures ---------------------------------------------------------------

/// Two slots carry the round trip and the fork. 64 positions hold a short prompt and its decode.
/// Both stay small, because this gate runs on a fragile GPU stack.
const n_slots: u32 = 2;
const window: u32 = 64;

const prompts = [_][]const u8{ "The dragon flew over", "Tell me a story about" };
const n_decode: usize = 4;
const n_tokens = n_decode + 1;

/// One thread everywhere, so a byte comparison reads the same work split on the
/// reference runner and on the hybrid's prefix runner.
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

/// Build a hybrid through the production planner, half the layers on each tier. A degenerate count
/// lets a single-tier runner pass the swap gates, so the helper asserts the count. A device that
/// cannot fund the placement skips the gate on its budget refusal.
fn initHybrid(
    gpa: std.mem.Allocator,
    ctxp: *const vulkan_ctx.Context,
    loaded: *const loader.Loaded,
    thread_pool: *test_policy.Pool,
) !*LlamaHybrid {
    const card = loaded.card;
    const n_layers: u32 = card.n_layers;
    const gpu_layers: u32 = n_layers / 2;
    std.debug.assert(1 <= gpu_layers);
    std.debug.assert(gpu_layers < n_layers);
    const caps = zigferencer.family.Of(.llama).Fused.kvCaps(
        loaded,
        limits.contextCeiling(loaded.card.context_length, null),
        ctxp.deviceLocalAvailableBytes(),
        n_slots,
        .f32,
    );
    const kv_costs = zigferencer.kv_cost.llamaCostVector(card, .f32);
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
        gpu_layers,
    ) catch return error.SkipZigTest;
    const hybrid = LlamaHybrid.init(
        gpa,
        ctxp,
        loaded,
        .{
            .plan = plan,
            .pool = thread_pool,
            .policy = cpu_cfg,
            .kv_dtype = .f32,
        },
    ) catch |e| switch (e) {
        error.VramBudgetExceeded => return error.SkipZigTest,
        else => return e,
    };
    std.debug.assert(hybrid.host_layers > 0);
    std.debug.assert(hybrid.host_layers < hybrid.n_layers);
    return hybrid;
}

/// Bytes the host prefix layers hold inside an `n_rows` blob, by one layer's
/// share of the whole-model figure times the host layers.
fn hostBlobBytes(cpu: *const LlamaCpu, hybrid: *const LlamaHybrid, n_rows: u32) usize {
    return @intCast(cpu.evictSlotBytesFor(n_rows) / hybrid.n_layers * hybrid.host_layers);
}

fn argmaxOf(logits: []const f32) usize {
    var best: usize = 0;
    for (logits[1..], 1..) |v, i| {
        if (v > logits[best]) {
            best = i;
        }
    }

    return best;
}

// -- tests ------------------------------------------------------------------

test "llama hybrid swap: the blob is split-independent" {
    const io = std.testing.io;
    var thread_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, &thread_pool);
    defer hybrid.deinit();

    const cpu = try LlamaCpu.init(arena, &loaded, .{
        .plan = kv_budget.pinned(n_slots, window),
        .max_rows = 0,
        .pool = &thread_pool,
        .policy = cpu_cfg,
        .kv_dtype = .f32,
    });

    // The residency tier sizes a RAM blob before it knows which runner holds
    // the slot, so the two figures must agree at every row count.
    for ([_]u32{ 1, 2, window }) |n_rows| {
        try std.testing.expectEqual(
            cpu.evictSlotBytesFor(n_rows),
            hybrid.evictSlotBytesFor(n_rows),
        );
    }
}

test "llama hybrid swap: one blob carries the whole-model framing" {
    const io = std.testing.io;
    var thread_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[0], true);

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, &thread_pool);
    defer hybrid.deinit();

    var cpu = try LlamaCpu.init(arena, &loaded, .{
        .plan = kv_budget.pinned(n_slots, window),
        .max_rows = 0,
        .pool = &thread_pool,
        .policy = cpu_cfg,
        .kv_dtype = .f32,
    });

    // Four positions cover one KV write and three KV reads on each tier.
    // Each runner takes its own entry point, as in a residency swap.
    const n_rows: u32 = 4;
    std.debug.assert(ids.len >= n_rows);
    for (ids[0..n_rows], 0..) |token, i| {
        const pos: u32 = @intCast(i);
        _ = try cpu.step(token, pos);
        try hybrid.stepBatch(&.{.{ .slot = 0, .token = token, .pos = pos, .want_logits = true }});
    }

    const blob_bytes: usize = @intCast(cpu.evictSlotBytesFor(n_rows));
    const cpu_blob = try arena.alloc(u8, blob_bytes);
    const hybrid_blob = try arena.alloc(u8, blob_bytes);
    try cpu.evictSlot(0, n_rows, cpu_blob);
    try hybrid.evictSlot(0, n_rows, hybrid_blob);

    const host_bytes = hostBlobBytes(&cpu, hybrid, n_rows);
    // Witness 1: the host half holds layers [0..k) and both runners compute it alike.
    try std.testing.expectEqualSlices(u8, cpu_blob[0..host_bytes], hybrid_blob[0..host_bytes]);

    // Witness 2: the CPU blob goes in through the H2D walk and comes back through the D2H walk.
    const round_blob = try arena.alloc(u8, blob_bytes);
    try hybrid.restoreSlot(0, n_rows, cpu_blob);
    try hybrid.evictSlot(0, n_rows, round_blob);
    try std.testing.expectEqualSlices(u8, cpu_blob, round_blob);

    // Witness 3: the hybrid holds the CPU state, so the greedy path must match the CPU runner.
    var pos: u32 = n_rows;
    while (pos < n_rows + 3) : (pos += 1) {
        const cpu_logits = try cpu.step(ids[0], pos);
        try hybrid.stepBatch(&.{.{ .slot = 0, .token = ids[0], .pos = pos, .want_logits = true }});
        try std.testing.expectEqual(
            argmaxOf(cpu_logits),
            argmaxOf(hybrid.logits[0..cpu_logits.len]),
        );
    }
}

test "llama hybrid swap: a restored slot decodes as if it never left" {
    const io = std.testing.io;
    var thread_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    var prompt_ids: [prompts.len][]const u32 = undefined;
    for (prompts, 0..) |p, i| {
        prompt_ids[i] = try bpe.encodeAlloc(&v, arena, p, true);
    }

    // The reference: both prompts on one two-slot CPU runner, decoded together.
    var reference: [prompts.len][n_tokens]u32 = undefined;
    {
        var cpu = try LlamaCpu.init(arena, &loaded, .{
            .plan = kv_budget.pinned(n_slots, window),
            .max_rows = 0,
            .pool = &thread_pool,
            .policy = cpu_cfg,
            .kv_dtype = .f32,
        });
        var cpu_pool = try Sessions(LlamaCpu).init(arena, &cpu);
        var cpu_ids: [prompts.len]u32 = undefined;
        for (prompt_ids, 0..) |ids, p| {
            cpu_ids[p] = try cpu_pool.create(0, .{});
            reference[p][0] = (try cpu_pool.feed(cpu_ids[p], ids, true)).?;
        }

        for (1..n_tokens) |k| {
            var out: [prompts.len]u32 = undefined;
            try cpu_pool.decodeBatch(&cpu_ids, &out);
            for (0..prompts.len) |p| {
                reference[p][k] = out[p];
            }
        }
    }

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, &thread_pool);
    defer hybrid.deinit();

    var pool = try Sessions(LlamaHybrid).init(arena, hybrid);

    var sids: [prompts.len]u32 = undefined;
    for (prompt_ids, 0..) |ids, p| {
        sids[p] = try pool.create(0, .{});
        try std.testing.expectEqual(reference[p][0], (try pool.feed(sids[p], ids, true)).?);
    }

    // Each tick evicts slot 1, overwrites it with the state of slot 0 on both tiers, and restores.
    // A wrong tier or region diverges slot 1. A swap that bleeds diverges slot 0 as well.
    const blob_max = try arena.alloc(u8, @intCast(hybrid.evictSlotBytesFor(hybrid.max_seq)));
    for (1..n_tokens) |k| {
        const n_rows = pool.posOf(sids[1]);
        const blob = blob_max[0..@intCast(hybrid.evictSlotBytesFor(n_rows))];
        try hybrid.evictSlot(1, n_rows, blob);
        hybrid.cloneSlotState(0, 1, n_rows);
        try hybrid.restoreSlot(1, n_rows, blob);
        var out: [prompts.len]u32 = undefined;
        try pool.decodeBatch(&sids, &out);
        for (0..prompts.len) |p| {
            try std.testing.expectEqual(reference[p][k], out[p]);
        }
    }
}

test "llama hybrid swap: a fork decodes beside its source" {
    const io = std.testing.io;
    var thread_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[0], true);

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, &thread_pool);
    defer hybrid.deinit();

    var pool = try Sessions(LlamaHybrid).init(arena, hybrid);

    const source = try pool.create(0, .{});
    _ = (try pool.feed(source, ids, true)).?;

    // The fork and its source must emit the same token every tick.
    // A KV region that the copy misses on either tier diverges them.
    const fork = try pool.clone(source, 0, .{});
    var pair: [2]u32 = undefined;
    for (0..4) |_| {
        try pool.decodeBatch(&.{ source, fork }, &pair);
        try std.testing.expectEqual(pair[0], pair[1]);
    }
}

test "llama hybrid provenance: the tag names the split, the quant, and the dtype" {
    var buf_15: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_16: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_q4: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_f16: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    const at_15 = LlamaRow.hybridBackendTag(&buf_15, 15, .q8_0, .f32);
    const at_16 = LlamaRow.hybridBackendTag(&buf_16, 16, .q8_0, .f32);
    const at_15_q4 = LlamaRow.hybridBackendTag(&buf_q4, 15, .q4_0, .f32);
    const at_15_f16 = LlamaRow.hybridBackendTag(&buf_f16, 15, .q8_0, .f16);

    // The split, the quant and the KV dtype are the three parts of a tag,
    // and each one alone must move the tag.
    try std.testing.expect(!std.mem.eql(u8, at_15, at_16));
    try std.testing.expect(!std.mem.eql(u8, at_15, at_15_q4));
    try std.testing.expect(!std.mem.eql(u8, at_15, at_15_f16));

    // The f32 tag stays the shipped literal byte for byte, so the dtype fold
    // invalidates no cached blob.
    try std.testing.expectEqualStrings("vulkan_hybrid:k=15:q8_0", at_15);
    try std.testing.expectEqualStrings("vulkan_hybrid:k=15:q8_0:kv=f16", at_15_f16);

    // The provenance of a snapshot is per tag. Every distinct tag must fold to a
    // distinct namespace, and none can land on the namespace of a single-tier family.
    const model_fp = prompt_cache.modelFingerprint("smollm2-135m", 145_000_000);
    const fps = [_]u64{
        prompt_cache.backendFingerprint(model_fp, at_15),
        prompt_cache.backendFingerprint(model_fp, at_16),
        prompt_cache.backendFingerprint(model_fp, at_15_q4),
        prompt_cache.backendFingerprint(model_fp, at_15_f16),
        prompt_cache.backendFingerprint(model_fp, "vulkan_fused"),
        prompt_cache.backendFingerprint(model_fp, "vulkan_fused:kv=f16"),
        prompt_cache.backendFingerprint(model_fp, "cpu"),
        prompt_cache.backendFingerprint(model_fp, "cpu:kv=f16"),
    };
    for (fps, 0..) |a, i| {
        for (fps[i + 1 ..]) |b| {
            try std.testing.expect(a != b);
        }
    }
}

test "llama hybrid provenance: a foreign tag is another namespace" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dirbuf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dirbuf, ".zig-cache/tmp/{s}/hybrid", .{&tmp.sub_path});

    var buf_mine: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_theirs: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_half: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    const model_fp = prompt_cache.modelFingerprint("smollm2-135m", 145_000_000);
    const mine = prompt_cache.backendFingerprint(
        model_fp,
        LlamaRow.hybridBackendTag(&buf_mine, 15, .q8_0, .f32),
    );
    const theirs = prompt_cache.backendFingerprint(
        model_fp,
        LlamaRow.hybridBackendTag(&buf_theirs, 16, .q8_0, .f32),
    );
    // The same split at the other KV dtype. A blob that crosses the dtype lands wrong bytes.
    const half = prompt_cache.backendFingerprint(
        model_fp,
        LlamaRow.hybridBackendTag(&buf_half, 15, .q8_0, .f16),
    );

    const seq = [_]u32{ 1, 2, 3, 4 };
    const blob = [_]u8{7} ** 64;
    const logits = [_]f32{ 0, 1 };
    {
        var cache = prompt_cache.PromptCache.init(gpa, 1 << 20);
        defer cache.deinit();

        try cache.enableDisk(io, dir, 1 << 20, mine);
        try std.testing.expect(
            cache.admit(
                prompt_cache.prefixKey(mine, &seq),
                &seq,
                &blob,
                &logits,
                100_000_000,
                1_000_000,
            ),
        );
        try std.testing.expect(cache.last_admit_disk);
    }

    // The f16 entry of the same split, prefix, and directory. Each scan below must pick its own
    // namespace out of a populated one.
    {
        var cache = prompt_cache.PromptCache.init(gpa, 1 << 20);
        defer cache.deinit();

        try cache.enableDisk(io, dir, 1 << 20, half);
        try std.testing.expect(
            cache.admit(
                prompt_cache.prefixKey(half, &seq),
                &seq,
                blob[0 .. blob.len / 2],
                &logits,
                100_000_000,
                1_000_000,
            ),
        );
        try std.testing.expect(cache.last_admit_disk);
    }

    // The other split scans the same directory and indexes nothing, because the
    // fingerprint every file there carries is not its own.
    var other = prompt_cache.PromptCache.init(gpa, 1 << 20);
    defer other.deinit();

    try other.enableDisk(io, dir, 1 << 20, theirs);
    other.loadDiskIndex();
    try std.testing.expectEqual(@as(usize, 0), other.diskIndexLen());

    // The same split and dtype indexes its own entry, so the refusal above is provenance.
    // The count is one, because the f16 sibling is another namespace.
    var same = prompt_cache.PromptCache.init(gpa, 1 << 20);
    defer same.deinit();

    try same.enableDisk(io, dir, 1 << 20, mine);
    same.loadDiskIndex();
    try std.testing.expectEqual(@as(usize, 1), same.diskIndexLen());

    // The dtype refusal runs both ways. f16 indexes its own entry alone.
    var same_half = prompt_cache.PromptCache.init(gpa, 1 << 20);
    defer same_half.deinit();

    try same_half.enableDisk(io, dir, 1 << 20, half);
    same_half.loadDiskIndex();
    try std.testing.expectEqual(@as(usize, 1), same_half.diskIndexLen());
}
