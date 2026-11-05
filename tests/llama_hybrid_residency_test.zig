//! Swap gate for the llama hybrid runner. The tiered KV swap must restore a
//! conversation byte for byte, and the provenance tag must refuse a blob that
//! another placement wrote.
//!
//! A hybrid slot lives in two tiers. The host prefix holds layers `[0..k)` and
//! the device suffix holds layers `[k..L)`, so two walks write one blob. The
//! state model proves that algebra at anchor scale, and these tests replay it
//! at model scale through the public surface of the runner. The blob size never
//! moves with the split, and "the blob is split-independent" witnesses that
//! across a CPU runner and a hybrid runner. The two walks together write the
//! whole-model framing, and "one blob carries the whole-model framing"
//! witnesses it three ways. The host half is byte-equal to the half of the CPU
//! runner, because the prefix boundary runs the same ops over the same weights. A
//! CPU blob goes through the H2D walk of the hybrid in and its D2H walk out byte for
//! byte. The transport therefore places and recovers every region. A greedy
//! decode over the restored rows then continues on the path of the CPU runner,
//! so the regions sit where attention expects them.
//!
//! The device halves hold the same values in the same places, but not the same
//! bits. The GPU-computed suffix KV differs from the CPU-computed suffix KV in
//! the low mantissa, because the reduce kernel sums in a different order. Byte
//! equality across executors is therefore not a guarantee this engine makes,
//! and that fact is the reason the snapshot fingerprint folds the split. A blob
//! must not restore under an executor that did not write it.
//!
//! Every runner here builds at a true mid split, except the split-0 case that
//! pins the fused stream. A degenerate split lets a
//! single-tier runner pass this whole file, so `initHybrid` asserts the split
//! it got.
//!
//! The hardware tests need a Vulkan device and the canonical
//! SmolLM2-135M-Instruct-Q8_0 fixture from `test_fixtures.path`. They return
//! `error.SkipZigTest` when the device is absent. The two provenance tests are
//! pure, so they run wherever this file compiles.

const std = @import("std");
const z = @import("zigferencer");
const tp = @import("test_policy.zig");

const bpe = z.bpe;
const kv_budget = z.kv_budget;
const limits = z.limits;
const loader = z.loader;
const prompt_cache = z.prompt_cache;
const tensor_lookup = z.tensor_lookup;
const vocab_mod = z.vocab;
const vulkan_ctx = z.vulkan_ctx;
const llama_fused = z.llama_fused;

const LlamaHybridRunner = z.llama_hybrid.LlamaHybridRunner;
const LlamaRunner = z.llama_runner.LlamaRunner;
const Sessions = z.session.Sessions;

// -- fixtures ---------------------------------------------------------------

/// Two slots carry the eviction round trip and the fork, and 64 positions hold
/// a short prompt plus its decode. Both stay small, because this gate runs on a
/// fragile GPU stack.
const n_slots: u32 = 2;
const window: u32 = 64;

const prompts = [_][]const u8{ "The dragon flew over", "Tell me a story about" };
const n_decode: usize = 4;
const n_tokens = n_decode + 1;

/// Tokens the split-0 identity gate compares. A drift in one layer body shows
/// within a few greedy steps, and 16 leaves the window room for the prompt.
const n_identity_tokens: usize = 16;

/// One thread everywhere, so a byte comparison reads the same work split on the
/// reference runner and on the hybrid's prefix runner.
const cpu_cfg: tp.Policy = .{ .n_threads = 1 };

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

/// Build a hybrid through the production planner with `gpu_layers` on the
/// device. A device that cannot fund that placement cannot run this gate, so a
/// budget refusal skips.
fn initHybridAt(
    gpa: std.mem.Allocator,
    ctxp: *const vulkan_ctx.Context,
    loaded: *const loader.Loaded,
    io: std.Io,
    thread_pool: *tp.Pool,
    gpu_layers: u32,
) !*LlamaHybridRunner {
    const card = loaded.card;
    const n_layers: u32 = card.n_layers;
    std.debug.assert(gpu_layers <= n_layers);
    const caps = try llama_fused.LlamaFusedRunner.kvCaps(
        gpa,
        loaded,
        limits.contextCeiling(loaded.card.context_length, null),
        ctxp.deviceLocalAvailableBytes(),
        n_slots,
        .f32,
    );
    const kv_costs = z.kv_cost.llamaCostVector(card, .f32);
    var table: [limits.max_layers]u64 = undefined;
    const plan = kv_budget.placementPlan(
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
    return LlamaHybridRunner.init(
        gpa,
        ctxp,
        loaded,
        plan,
        io,
        thread_pool,
        cpu_cfg,
        .f32,
    ) catch |e| switch (e) {
        error.VramBudgetExceeded => return error.SkipZigTest,
        else => return e,
    };
}

/// The true mid split the swap gates need: half the layers on the device, the
/// rest on the host. A degenerate split lets a single-tier runner pass those
/// gates, so this helper asserts the split it got.
fn initHybrid(
    gpa: std.mem.Allocator,
    ctxp: *const vulkan_ctx.Context,
    loaded: *const loader.Loaded,
    io: std.Io,
    thread_pool: *tp.Pool,
) !*LlamaHybridRunner {
    const n_layers: u32 = loaded.card.n_layers;
    std.debug.assert(n_layers >= 2); // a mid split needs a layer on each side
    const hybrid = try initHybridAt(gpa, ctxp, loaded, io, thread_pool, n_layers / 2);
    std.debug.assert(hybrid.split > 0);
    std.debug.assert(hybrid.split < hybrid.n_layers);
    return hybrid;
}

/// The fused runner over the whole model, at the slot count and window the
/// hybrid takes. A budget refusal skips, like the hybrid's.
fn initFused(
    gpa: std.mem.Allocator,
    ctxp: *const vulkan_ctx.Context,
    loaded: *const loader.Loaded,
) !*llama_fused.LlamaFusedRunner {
    const caps = try llama_fused.LlamaFusedRunner.kvCaps(
        gpa,
        loaded,
        limits.contextCeiling(loaded.card.context_length, null),
        ctxp.deviceLocalAvailableBytes(),
        n_slots,
        .f32,
    );
    const plan = kv_budget.kvPlan(
        caps,
        llama_fused.LlamaFusedRunner.kvCostModel(loaded.card, .f32),
        .{
            .window = window,
            .slots = n_slots,
            .batch = n_slots,
        },
    ) catch return error.SkipZigTest;
    return llama_fused.LlamaFusedRunner.initFromLoaded(
        gpa,
        ctxp,
        loaded,
        plan,
        .f32,
    ) catch |e| switch (e) {
        error.VramBudgetExceeded => return error.SkipZigTest,
        else => return e,
    };
}

/// Bytes the host prefix layers hold inside an `n_rows` blob, by one layer's
/// share of the whole-model figure times the split.
fn hostBlobBytes(cpu: *const LlamaRunner, hybrid: *const LlamaHybridRunner, n_rows: u32) usize {
    return @intCast(cpu.evictSlotBytesFor(n_rows) / hybrid.n_layers * hybrid.split);
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
    var thread_pool: tp.Pool = undefined;
    try tp.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, io, &thread_pool);
    defer hybrid.deinit();

    const cpu = try LlamaRunner.init(
        arena,
        &loaded,
        kv_budget.pinned(n_slots, window),
        0,
        io,
        &thread_pool,
        cpu_cfg,
        .f32,
    );

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
    var thread_pool: tp.Pool = undefined;
    try tp.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[0], true);

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, io, &thread_pool);
    defer hybrid.deinit();

    var cpu = try LlamaRunner.init(
        arena,
        &loaded,
        kv_budget.pinned(n_slots, window),
        0,
        io,
        &thread_pool,
        cpu_cfg,
        .f32,
    );

    // Four positions cover one KV write and three KV reads on each tier. Each
    // runner takes its own entry point, so the blobs come out of the two code
    // paths a residency swap really uses.
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
    // The host half holds layers [0..k). Both runners run these layers through
    // the same CPU ops over the same weights, so the prefix boundary carries them.
    try std.testing.expectEqualSlices(u8, cpu_blob[0..host_bytes], hybrid_blob[0..host_bytes]);

    // The transport check: the CPU blob goes in through the H2D walk of the
    // hybrid and comes back through its D2H walk unchanged. The shipped
    // fingerprint forbids this cross-executor restore. The test does it on
    // purpose, because only a foreign blob proves the walks address the
    // whole-model framing and not merely each other.
    const round_blob = try arena.alloc(u8, blob_bytes);
    try hybrid.restoreSlot(0, n_rows, cpu_blob);
    try hybrid.evictSlot(0, n_rows, round_blob);
    try std.testing.expectEqualSlices(u8, cpu_blob, round_blob);

    // The meaning check: the hybrid now holds the state of the CPU runner, so a
    // greedy continuation must follow the path of the CPU runner. A region the
    // walk misplaced sits where attention reads it, so it garbles the logits
    // here even though the transport check above cannot see it.
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
    var thread_pool: tp.Pool = undefined;
    try tp.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

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
        var cpu = try LlamaRunner.init(
            arena,
            &loaded,
            kv_budget.pinned(n_slots, window),
            0,
            io,
            &thread_pool,
            cpu_cfg,
            .f32,
        );
        var cpu_pool = try Sessions(LlamaRunner).init(arena, &cpu);
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

    const hybrid = try initHybrid(gpa, ctxp, &loaded, io, &thread_pool);
    defer hybrid.deinit();

    var pool = try Sessions(LlamaHybridRunner).init(arena, hybrid);

    var sids: [prompts.len]u32 = undefined;
    for (prompt_ids, 0..) |ids, p| {
        sids[p] = try pool.create(0, .{});
        try std.testing.expectEqual(reference[p][0], (try pool.feed(sids[p], ids, true)).?);
    }

    // Every tick round-trips slot 1 through a RAM blob with a deliberate
    // corruption in between. The clone overwrites slot 1 with the state of slot
    // 0 on both tiers. Only a restore that lands every region puts the stream
    // back. A swap that addresses the wrong tier, or the wrong region
    // inside one, diverges slot 1 from the reference. A swap that bleeds
    // diverges slot 0 as well. One buffer holds the widest blob, sliced each
    // round to the committed rows.
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
    var thread_pool: tp.Pool = undefined;
    try tp.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[0], true);

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    const hybrid = try initHybrid(gpa, ctxp, &loaded, io, &thread_pool);
    defer hybrid.deinit();

    var pool = try Sessions(LlamaHybridRunner).init(arena, hybrid);

    const source = try pool.create(0, .{});
    _ = (try pool.feed(source, ids, true)).?;

    // Greedy sampling over a faithful two-tier copy has one outcome. The fork
    // and its source must emit the same token every tick. A KV region the copy
    // misses on either tier diverges them within a few steps.
    const fork = try pool.clone(source, 0, .{});
    var pair: [2]u32 = undefined;
    for (0..4) |_| {
        try pool.decodeBatch(&.{ source, fork }, &pair);
        try std.testing.expectEqual(pair[0], pair[1]);
    }
}

test "llama hybrid dispatch: split 0 decodes the stream of the fused runner" {
    const io = std.testing.io;
    var thread_pool: tp.Pool = undefined;
    try tp.startPool(&thread_pool, io, cpu_cfg);
    defer thread_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[0], true);

    const ctxp = try initCtx(arena);
    defer ctxp.deinit();

    // Every layer on the device: the placement the runner header calls
    // dispatch-identical to the fused runner. A shell gate proves that claim
    // over the whole split ladder. This case proves the one end of it here.
    const hybrid = try initHybridAt(gpa, ctxp, &loaded, io, &thread_pool, loaded.card.n_layers);
    defer hybrid.deinit();

    try std.testing.expectEqual(@as(u32, 0), hybrid.split);

    const fused = try initFused(gpa, ctxp, &loaded);
    defer fused.deinit();

    var hybrid_pool = try Sessions(LlamaHybridRunner).init(arena, hybrid);
    var fused_pool = try Sessions(llama_fused.LlamaFusedRunner).init(arena, fused);
    const h_sid = try hybrid_pool.create(0, .{});
    const f_sid = try fused_pool.create(0, .{});

    // One greedy stream per runner over one prompt. Two runners that record the
    // same dispatches over the same weights emit one stream, so the first token
    // that differs names the layer body that drifted.
    const h_first = (try hybrid_pool.feed(h_sid, ids, true)).?;
    const f_first = (try fused_pool.feed(f_sid, ids, true)).?;
    try std.testing.expectEqual(f_first, h_first);
    for (1..n_identity_tokens) |_| {
        var h_out: [1]u32 = undefined;
        var f_out: [1]u32 = undefined;
        try hybrid_pool.decodeBatch(&.{h_sid}, &h_out);
        try fused_pool.decodeBatch(&.{f_sid}, &f_out);
        try std.testing.expectEqual(f_out[0], h_out[0]);
    }
}

test "llama hybrid provenance: the tag names the placement" {
    var buf_15: [LlamaHybridRunner.backend_tag_buf_len]u8 = undefined;
    var buf_16: [LlamaHybridRunner.backend_tag_buf_len]u8 = undefined;
    var buf_q4: [LlamaHybridRunner.backend_tag_buf_len]u8 = undefined;
    var buf_f16: [LlamaHybridRunner.backend_tag_buf_len]u8 = undefined;
    const at_15 = LlamaHybridRunner.backendTag(&buf_15, 15, .q8_0, .f32);
    const at_16 = LlamaHybridRunner.backendTag(&buf_16, 16, .q8_0, .f32);
    const at_15_q4 = LlamaHybridRunner.backendTag(&buf_q4, 15, .q4_0, .f32);
    const at_15_f16 = LlamaHybridRunner.backendTag(&buf_f16, 15, .q8_0, .f16);

    // The split, the quant and the KV dtype are the three parts of a placement,
    // and each one alone must move the tag.
    try std.testing.expect(!std.mem.eql(u8, at_15, at_16));
    try std.testing.expect(!std.mem.eql(u8, at_15, at_15_q4));
    try std.testing.expect(!std.mem.eql(u8, at_15, at_15_f16));

    // The f32 tag stays the shipped literal byte for byte, so the dtype fold
    // invalidates no cached blob.
    try std.testing.expectEqualStrings("vulkan_hybrid:k=15:q8_0", at_15);
    try std.testing.expectEqualStrings("vulkan_hybrid:k=15:q8_0:kv=f16", at_15_f16);

    // The numeric provenance of a snapshot is per placement. Every distinct tag
    // must fold to a distinct namespace, and no hybrid placement can land on
    // the namespace of a single-tier family.
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

test "llama hybrid provenance: a foreign placement is another namespace" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dirbuf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dirbuf, ".zig-cache/tmp/{s}/hybrid", .{&tmp.sub_path});

    var buf_mine: [LlamaHybridRunner.backend_tag_buf_len]u8 = undefined;
    var buf_theirs: [LlamaHybridRunner.backend_tag_buf_len]u8 = undefined;
    var buf_half: [LlamaHybridRunner.backend_tag_buf_len]u8 = undefined;
    const model_fp = prompt_cache.modelFingerprint("smollm2-135m", 145_000_000);
    const mine = prompt_cache.backendFingerprint(
        model_fp,
        LlamaHybridRunner.backendTag(&buf_mine, 15, .q8_0, .f32),
    );
    const theirs = prompt_cache.backendFingerprint(
        model_fp,
        LlamaHybridRunner.backendTag(&buf_theirs, 16, .q8_0, .f32),
    );
    // The same placement at the other KV dtype. Its rows are half as wide, so a
    // blob that crosses the dtype lands the wrong bytes in every region.
    const half = prompt_cache.backendFingerprint(
        model_fp,
        LlamaHybridRunner.backendTag(&buf_half, 15, .q8_0, .f16),
    );

    const seq = [_]u32{ 1, 2, 3, 4 };
    const blob = [_]u8{7} ** 64;
    const logits = [_]f32{ 0, 1 };
    {
        var cache = prompt_cache.PromptCache.init(gpa, 1 << 20);
        defer cache.deinit();

        try cache.enableDisk(io, dir, 1 << 20, mine);
        try std.testing.expect(
            try cache.admit(
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

    // The f16 entry of the same split, over the same prefix, in the same
    // directory. Every scan below must now pick its own namespace out of a
    // populated one, and a half-width blob states the size the dtype moves.
    {
        var cache = prompt_cache.PromptCache.init(gpa, 1 << 20);
        defer cache.deinit();

        try cache.enableDisk(io, dir, 1 << 20, half);
        try std.testing.expect(
            try cache.admit(
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

    // The same split at the same dtype does index its own entry, so the refusal
    // above is provenance and not an empty directory. The count is one and not
    // two, because the f16 sibling beside it is another namespace.
    var same = prompt_cache.PromptCache.init(gpa, 1 << 20);
    defer same.deinit();

    try same.enableDisk(io, dir, 1 << 20, mine);
    same.loadDiskIndex();
    try std.testing.expectEqual(@as(usize, 1), same.diskIndexLen());

    // The dtype refusal runs both ways. f16 indexes its own entry alone, so
    // neither dtype can read the rows of the other in either direction of the swap.
    var same_half = prompt_cache.PromptCache.init(gpa, 1 << 20);
    defer same_half.deinit();

    try same_half.enableDisk(io, dir, 1 << 20, half);
    same_half.loadDiskIndex();
    try std.testing.expectEqual(@as(usize, 1), same_half.diskIndexLen());
}
