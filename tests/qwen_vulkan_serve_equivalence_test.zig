//! Batching gate for the qwen fused runner on the GPU. The greedy token streams that the multi-slot
//! `stepBatch` produces must equal the reference streams of the CPU runner exactly.
//!
//! Equality. Greedy tokens are exact across backends. The per-kernel numerics stay under the
//! tolerance each kernel declares.
//!
//! Batched decode. Two sessions decode interleaved through `Sessions(QwenFused)`. This drives
//! the same-slot runs of chunked prefill through the per-run GDN dispatches and the per-row rope
//! and attention. The test then clones one session. The fork must continue the stream of its
//! source, which pins the device-side slot copy.
//!
//! Residency. One physical VRAM slot carries two logical sessions. The residency layer evicts the
//! KV and the GDN recurrent state of one session to a RAM blob and restores the other on every
//! interleaved step. This is the worst-case thrash, and the streams must still match the CPU
//! reference, because a faithful swap is invisible at the boundary. The layer drives the real fused
//! runner.
//!
//! Prefix restore. This is the prefix-cache path of the serve scheduler. A cold session feeds the
//! whole prompt. A warm session restores the first `n` rows from a blob and prefills the one-token
//! suffix only. The qwen blob holds the recurrent state and the conv ring of every GDN layer beside
//! the KV rows. Both are point-in-time, so the restore is faithful only at the position of the
//! snapshot. Warm must equal cold. The batched decode test already pins the GPU stream against the
//! CPU reference, so this test does not repeat it.
//!
//! The Qwen3.5-2B fixture comes from `test_fixtures.path`. The tests return `error.SkipZigTest`
//! without a Vulkan device.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;
const QwenCpu = zigferencer.family.Of(.qwen35).Cpu;
const Sessions = zigferencer.session.Sessions;
const Residency = zigferencer.serve_residency.Residency;
const limits = zigferencer.limits;
const vulkan_ctx = zigferencer.vulkan_ctx;
const QwenFused = zigferencer.family.Of(.qwen35).Fused;

// -- fixtures ---------------------------------------------------------------

const prompts = [_][]const u8{ "The dragon flew over", "Tell me a story about" };
const n_decode: usize = 11;
const n_tokens = n_decode + 1;

const Stream = struct { tokens: [n_tokens]u32 };

/// Build the plan for the fused runner: `n_slots` slots, the narrower test
/// window past one slot, and no device budget.
fn fusedPlan(
    loaded: *const loader.Loaded,
    n_slots: u32,
    kv_dtype: zigferencer.dtype.KvDtype,
) !zigferencer.kv_budget.KvPlan {
    const fakes = zigferencer.test_fakes;
    const window = if (n_slots <= 1) fakes.test_window else fakes.test_serve_window;
    return zigferencer.kv_budget.kvPlan(
        QwenFused.kvCaps(
            loaded,
            limits.contextCeiling(loaded.card.context_length, null),
            0,
            .other,
            n_slots,
            kv_dtype,
        ),
        QwenFused.kvCostModel(loaded.card, kv_dtype),
        .{
            .window = @min(loaded.card.context_length, window),
            .slots = n_slots,
            .batch = n_slots,
        },
    );
}

fn batchedDecodeGate(kv_dtype: zigferencer.dtype.KvDtype) !void {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    var prompt_ids: [prompts.len][]const u32 = undefined;
    for (prompts, 0..) |p, i| {
        prompt_ids[i] = try bpe.encodeAlloc(&v, arena, p, true);
    }

    // -- cpu reference: each prompt alone on the batched CPU runner ---------
    var reference: [prompts.len]Stream = undefined;
    for (prompt_ids, 0..) |ids, p| {
        var cpu_pool: test_policy.Pool = undefined;
        try test_policy.startPool(&cpu_pool, io, test_policy.policy());
        defer cpu_pool.deinit();

        var runner = try QwenCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(1, zigferencer.test_fakes.test_window),
            .max_rows = 0,
            .pool = &cpu_pool,
            .policy = test_policy.policy(),
            .kv_dtype = kv_dtype,
        });
        var pool = try Sessions(QwenCpu).init(arena, &runner);
        const sid = try pool.create(0, .{});
        reference[p].tokens[0] = (try pool.feed(sid, ids, true)).?;
        for (1..n_tokens) |k| {
            var out: [1]u32 = undefined;
            try pool.decodeBatch(&.{sid}, &out);
            reference[p].tokens[k] = out[0];
        }
    }

    // -- gpu run: both sessions interleaved on one 3-slot fused runner ------
    const ctxp = try arena.create(vulkan_ctx.Context);
    ctxp.* = vulkan_ctx.Context.init(arena, .{}) catch return error.SkipZigTest;
    defer ctxp.deinit();

    const fused = QwenFused.init(
        gpa,
        ctxp,
        &loaded,
        try fusedPlan(&loaded, 3, kv_dtype),
        kv_dtype,
    ) catch
        return error.SkipZigTest;
    defer fused.deinit();

    var pool = try Sessions(QwenFused).init(arena, fused);

    var sids: [prompts.len]u32 = undefined;
    for (prompt_ids, 0..) |ids, p| {
        sids[p] = try pool.create(0, .{});
        const first = (try pool.feed(sids[p], ids, true)).?;
        try std.testing.expectEqual(reference[p].tokens[0], first);
    }

    // Each tick evicts slot 1, corrupts it with the state of slot 0, and restores the blob.
    // One buffer holds the widest blob, and each round slices it to the committed rows.
    const blob_max = try arena.alloc(u8, @intCast(fused.evictSlotBytesFor(fused.max_seq)));
    for (1..n_tokens) |k| {
        const rows = pool.posOf(sids[1]);
        const blob = blob_max[0..@intCast(fused.evictSlotBytesFor(rows))];
        try fused.evictSlot(1, rows, blob);
        fused.cloneSlotState(0, 1, rows); // corrupt slot 1
        try fused.restoreSlot(1, rows, blob); // must fully undo it
        var out: [prompts.len]u32 = undefined;
        try pool.decodeBatch(&sids, &out);
        for (0..prompts.len) |p| {
            try std.testing.expectEqual(reference[p].tokens[k], out[p]);
        }
    }

    // -- clone: the fork and its source continue identically ----------------
    // The pair must emit the same token every tick. A missed state region diverges them.
    const target = try pool.clone(sids[0], 0, .{});
    var pair: [2]u32 = undefined;
    for (0..4) |_| {
        try pool.decodeBatch(&.{ sids[0], target }, &pair);
        try std.testing.expectEqual(pair[0], pair[1]);
    }
}

// -- tests ------------------------------------------------------------------

test "qwen GPU batched decode matches the CPU reference streams (greedy, exact)" {
    try batchedDecodeGate(.f32);
}

test "qwen GPU batched decode at kv f16 matches the f16 CPU reference" {
    // KV bytes differ across backends by design, so the gate reads tokens and never cache bits.
    try batchedDecodeGate(.f16);
}

test "qwen GPU residency over-subscription: 2 sessions, 1 VRAM slot, streams match" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    var prompt_ids: [prompts.len][]const u32 = undefined;
    for (prompts, 0..) |p, i| {
        prompt_ids[i] = try bpe.encodeAlloc(&v, arena, p, true);
    }

    // CPU reference: each prompt alone.
    var reference: [prompts.len]Stream = undefined;
    for (prompt_ids, 0..) |ids, p| {
        var cpu_pool: test_policy.Pool = undefined;
        try test_policy.startPool(&cpu_pool, io, test_policy.policy());
        defer cpu_pool.deinit();

        var runner = try QwenCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(1, zigferencer.test_fakes.test_window),
            .max_rows = 0,
            .pool = &cpu_pool,
            .policy = test_policy.policy(),
            .kv_dtype = .f32,
        });
        var pool = try Sessions(QwenCpu).init(arena, &runner);
        const sid = try pool.create(0, .{});
        reference[p].tokens[0] = (try pool.feed(sid, ids, true)).?;
        for (1..n_tokens) |k| {
            var out: [1]u32 = undefined;
            try pool.decodeBatch(&.{sid}, &out);
            reference[p].tokens[k] = out[0];
        }
    }

    // Two logical sessions share one VRAM slot.
    const ctxp = try arena.create(vulkan_ctx.Context);
    ctxp.* = vulkan_ctx.Context.init(arena, .{}) catch return error.SkipZigTest;
    defer ctxp.deinit();

    const fused = QwenFused.init(
        gpa,
        ctxp,
        &loaded,
        try fusedPlan(&loaded, 1, .f32),
        .f32,
    ) catch
        return error.SkipZigTest;
    defer fused.deinit();

    var sessions = try Sessions(QwenFused).init(arena, fused);
    var res = try Residency(
        Sessions(QwenFused),
    ).init(arena, gpa, &sessions, 2, null);
    defer res.deinit();

    var sids: [prompts.len]u32 = undefined;
    for (prompt_ids, 0..) |ids, p| {
        sids[p] = try res.create(0, .{});
        const first = (try res.feed(sids[p], ids, true)).?; // the second feed evicts the first
        try std.testing.expectEqual(reference[p].tokens[0], first);
    }

    for (1..n_tokens) |k| {
        for (0..prompts.len) |p| {
            var out: [1]u32 = undefined;
            try res.decodeBatch(&.{sids[p]}, &out); // one slot: a swap each step
            try std.testing.expectEqual(reference[p].tokens[k], out[0]);
        }
    }

    res.destroy(sids[0]);
    res.destroy(sids[1]);
}

test "qwen GPU prefix restore: a warm-restored prefix continues bit-exactly" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[0], true);
    std.debug.assert(ids.len >= 2); // the split needs a prefix row and a suffix row
    const n: u32 = @intCast(ids.len - 1);

    const ctxp = try arena.create(vulkan_ctx.Context);
    ctxp.* = vulkan_ctx.Context.init(arena, .{}) catch return error.SkipZigTest;
    defer ctxp.deinit();

    const fused = QwenFused.init(
        gpa,
        ctxp,
        &loaded,
        try fusedPlan(&loaded, 2, .f32),
        .f32,
    ) catch
        return error.SkipZigTest;
    defer fused.deinit();

    var pool = try Sessions(QwenFused).init(arena, fused);

    // Cold: feed the prefix without logits, snapshot those rows, then feed the suffix.
    var cold_stream: Stream = undefined;
    const cold = try pool.create(0, .{});
    _ = try pool.feed(cold, ids[0..n], false);
    const blob = try arena.alloc(u8, @intCast(pool.snapshotBytesFor(n)));
    try pool.snapshotPrefix(cold, n, blob);
    cold_stream.tokens[0] = (try pool.feed(cold, ids[n..], true)).?;
    const cold_pos = pool.posOf(cold);
    for (1..n_tokens) |k| {
        var out: [1]u32 = undefined;
        try pool.decodeBatch(&.{cold}, &out);
        cold_stream.tokens[k] = out[0];
    }

    // Warm: a fresh session seeded from the blob, then the identical suffix.
    var warm_stream: Stream = undefined;
    const warm = try pool.create(0, .{});
    try pool.restorePrefix(warm, n, blob);
    warm_stream.tokens[0] = (try pool.feed(warm, ids[n..], true)).?;
    try std.testing.expectEqual(cold_pos, pool.posOf(warm));
    for (1..n_tokens) |k| {
        var out: [1]u32 = undefined;
        try pool.decodeBatch(&.{warm}, &out);
        warm_stream.tokens[k] = out[0];
    }

    try std.testing.expectEqualSlices(u32, &cold_stream.tokens, &warm_stream.tokens);
}
