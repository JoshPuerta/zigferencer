//! Batching gate for the qwen fused runner on the GPU. The greedy token streams
//! that the multi-slot `stepBatch` produces must equal the CPU `QwenRunner`
//! reference streams exactly.
//!
//! That equality is the cross-backend rule of this engine. Greedy tokens are
//! exact, and the per-kernel numerics stay under the tolerance each kernel
//! declares. Two sessions decode interleaved through
//! `Sessions(QwenFusedRunner)`, which drives the same-slot runs of chunked
//! prefill through the per-run GDN dispatches and the per-row rope and
//! attention. The test then clones one session, and the fork must continue the
//! stream of its source, which pins the device-side slot copy.
//!
//! The Qwen3.5-2B fixture comes from `test_fixtures.path`. The tests return
//! `error.SkipZigTest` without a Vulkan device.

const std = @import("std");
const z = @import("zigferencer");
const tp = @import("test_policy.zig");
const loader = z.loader;
const vocab_mod = z.vocab;
const bpe = z.bpe;
const QwenRunner = z.qwen_runner.QwenRunner;
const Sessions = z.session.Sessions;
const Residency = z.serve_residency.Residency;
const limits = z.limits;
const vulkan_ctx = z.vulkan_ctx;
const qwen_fused = z.qwen_fused;

// -- fixtures ---------------------------------------------------------------

const prompts = [_][]const u8{ "The dragon flew over", "Tell me a story about" };
const n_decode: usize = 11;
const n_tokens = n_decode + 1;

const Stream = struct { tokens: [n_tokens]u32 };

/// Build the plan for the fused runner: `n_slots` slots, the narrower test
/// window past one slot, and no device budget.
fn fusedPlan(loaded: *const loader.Loaded, n_slots: u32, kv_dtype: z.dtype.KvDtype) !z.kv_budget.KvPlan {
    const window = if (n_slots <= 1) z.test_fakes.test_window else z.test_fakes.test_serve_window;
    return z.kv_budget.kvPlan(
        qwen_fused.QwenFusedRunner.kvCaps(loaded, limits.contextCeiling(loaded.card.context_length, null), 0, n_slots, kv_dtype),
        qwen_fused.QwenFusedRunner.kvCostModel(loaded.card, kv_dtype),
        .{
            .window = @min(loaded.card.context_length, window),
            .slots = n_slots,
            .batch = n_slots,
        },
    );
}

fn batchedDecodeGate(kv_dtype: z.dtype.KvDtype) !void {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();
    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    var prompt_ids: [prompts.len][]const u32 = undefined;
    for (prompts, 0..) |p, i| prompt_ids[i] = try bpe.encodeAlloc(&v, arena, p, true);

    // -- cpu reference: each prompt alone on the batched CPU runner ---------
    var reference: [prompts.len]Stream = undefined;
    for (prompt_ids, 0..) |ids, p| {
        var cpu_pool: tp.Pool = undefined;
        try tp.startPool(&cpu_pool, io, tp.policy());
        defer cpu_pool.deinit();
        var runner = try QwenRunner.init(arena, &loaded, z.kv_budget.pinned(1, z.test_fakes.test_window), 0, io, &cpu_pool, tp.policy(), kv_dtype);
        var pool = try Sessions(QwenRunner).init(arena, &runner);
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
    const fused = qwen_fused.QwenFusedRunner.init(gpa, ctxp, &loaded, try fusedPlan(&loaded, 3, kv_dtype), kv_dtype) catch
        return error.SkipZigTest;
    defer fused.deinit();
    var pool = try Sessions(qwen_fused.QwenFusedRunner).init(arena, fused);

    var sids: [prompts.len]u32 = undefined;
    for (prompt_ids, 0..) |ids, p| {
        sids[p] = try pool.create(0, .{});
        const first = (try pool.feed(sids[p], ids, true)).?;
        try std.testing.expectEqual(reference[p].tokens[0], first);
    }
    // Every tick round-trips slot 1 through a RAM blob with a deliberate
    // corruption in between. The round evicts the slot, overwrites it with the
    // state of slot 0, and restores the blob. An evict or a restore that
    // addresses the wrong slot region diverges the stream of slot 1 from the
    // reference. A round that bleeds diverges slot 0 as well. One buffer holds
    // the widest blob, sliced each round to the rows the session committed.
    const blob_max = try arena.alloc(u8, @intCast(fused.evictSlotBytesFor(fused.max_seq)));
    for (1..n_tokens) |k| {
        const rows = pool.posOf(sids[1]);
        const blob = blob_max[0..@intCast(fused.evictSlotBytesFor(rows))];
        try fused.evictSlot(1, rows, blob);
        fused.cloneSlotState(0, 1, rows); // corrupt slot 1
        try fused.restoreSlot(1, rows, blob); // must fully undo it
        var out: [prompts.len]u32 = undefined;
        try pool.decodeBatch(&sids, &out);
        for (0..prompts.len) |p| try std.testing.expectEqual(reference[p].tokens[k], out[p]);
    }

    // -- clone: the fork and its source continue identically ----------------
    // Greedy sampling over a faithful device-side state copy has one outcome.
    // The pair must emit the same token every tick. A missed region, whether
    // the state, the conv ring or a KV layer, diverges them within a few steps.
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
    // Both backends frame KV at f16, and greedy tokens must still agree
    // exactly. KV bytes differ across backends by design, so the gate reads the
    // token stream and never the cache bits.
    try batchedDecodeGate(.f16);
}

test "qwen GPU residency over-subscription: 2 logical sessions, 1 VRAM slot, streams match reference" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();
    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    var prompt_ids: [prompts.len][]const u32 = undefined;
    for (prompts, 0..) |p, i| prompt_ids[i] = try bpe.encodeAlloc(&v, arena, p, true);

    // CPU reference: each prompt alone, the in-repo oracle.
    var reference: [prompts.len]Stream = undefined;
    for (prompt_ids, 0..) |ids, p| {
        var cpu_pool: tp.Pool = undefined;
        try tp.startPool(&cpu_pool, io, tp.policy());
        defer cpu_pool.deinit();
        var runner = try QwenRunner.init(arena, &loaded, z.kv_budget.pinned(1, z.test_fakes.test_window), 0, io, &cpu_pool, tp.policy(), .f32);
        var pool = try Sessions(QwenRunner).init(arena, &runner);
        const sid = try pool.create(0, .{});
        reference[p].tokens[0] = (try pool.feed(sid, ids, true)).?;
        for (1..n_tokens) |k| {
            var out: [1]u32 = undefined;
            try pool.decodeBatch(&.{sid}, &out);
            reference[p].tokens[k] = out[0];
        }
    }

    // One physical VRAM slot carries two logical sessions. The residency must
    // evict the KV and the GDN recurrent state of one session to a RAM blob. It
    // must restore the other on every interleaved step. That is the worst-case
    // thrash. The greedy streams must still match the CPU reference, because a
    // faithful swap is invisible at the boundary. The residency layer here
    // drives the real fused runner and not a fake.
    const ctxp = try arena.create(vulkan_ctx.Context);
    ctxp.* = vulkan_ctx.Context.init(arena, .{}) catch return error.SkipZigTest;
    defer ctxp.deinit();
    const fused = qwen_fused.QwenFusedRunner.init(gpa, ctxp, &loaded, try fusedPlan(&loaded, 1, .f32), .f32) catch
        return error.SkipZigTest;
    defer fused.deinit();
    var sessions = try Sessions(qwen_fused.QwenFusedRunner).init(arena, fused);
    var res = try Residency(Sessions(qwen_fused.QwenFusedRunner)).init(arena, gpa, &sessions, 2, null);
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
    // This is the prefix-cache path of the serve scheduler on the qwen device
    // runner. A cold session feeds the whole prompt. A warm session restores
    // the first `n` rows of the prompt out of a blob and then prefills the
    // one-token suffix only. Greedy tokens are exact on this backend, so the
    // two streams must agree token for token.
    //
    // The qwen blob carries the recurrent state and the conv ring of every GDN
    // layer beside the KV rows. Both are point-in-time rather than a prefix.
    // The restore is therefore faithful only at the position the snapshot took,
    // which is what this test pins. It feeds `n` non-final tokens and then the
    // same continuation from a fresh session.
    //
    // The rule under test is that warm equals cold. The first test in this file
    // already pins the GPU stream of this prompt against the CPU reference.
    // That reference block lives inside that test, so this test does not repeat
    // it.
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();
    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[0], true);
    std.debug.assert(ids.len >= 2); // the split needs a prefix row and a suffix row
    const n: u32 = @intCast(ids.len - 1);

    const ctxp = try arena.create(vulkan_ctx.Context);
    ctxp.* = vulkan_ctx.Context.init(arena, .{}) catch return error.SkipZigTest;
    defer ctxp.deinit();
    const fused = qwen_fused.QwenFusedRunner.init(gpa, ctxp, &loaded, try fusedPlan(&loaded, 2, .f32), .f32) catch
        return error.SkipZigTest;
    defer fused.deinit();
    var pool = try Sessions(qwen_fused.QwenFusedRunner).init(arena, fused);

    // Cold: feed the prefix without asking for logits, snapshot those rows,
    // then feed the suffix and decode on. The snapshot is a read, so the cold
    // session goes on running.
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
