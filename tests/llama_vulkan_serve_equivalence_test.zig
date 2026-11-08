//! Batching gate for the llama fused runner on the GPU. The greedy token streams that the batched
//! multi-stream path produces must equal the CPU reference streams exactly.
//!
//! Equality. Greedy tokens are exact across backends. The per-op numerics stay under the tolerance
//! each kernel declares.
//!
//! Batched decode. Two sessions decode interleaved through `Sessions(LlamaFusedRunner)`. This also
//! drives the same-slot ascending rows of chunked prefill through the per-row rope and attention
//! dispatches. The test compares each stream against a solo CPU `LlamaRunner` run.
//!
//! Composed advance. This is the exact path of the serve scheduler and the hardest compact-logits
//! case. One final chunk carries a whole prompt while another session decodes, inside one fused
//! `stepBatch`. The want-logits suffix is the last row of the chunk plus the decode row, so `first`
//! is chunk_len-1 above 0 and `n_logits` is 2 above 1. The suffix compacts through the buf_x
//! bounce. It lands at logits[0], the first token of the late session, and logits[1], the decode of
//! the front session, exactly where `advance` reads them.
//!
//! Prefix restore. This is the prefix-cache path of the serve scheduler. A cold session feeds the
//! whole prompt. A warm session restores the first `n` rows from a blob and prefills the one-token
//! suffix only. A blob is whole or nothing at its row count, so a snapshot taken at `n` restores at
//! `n`. Warm must equal cold. The batched decode test already pins the GPU stream against the CPU
//! reference, so this test does not repeat it.
//!
//! The test reads the canonical SmolLM2 Q8_0 from `test_fixtures.path`. It returns
//! `error.SkipZigTest` without a Vulkan device.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;
const LlamaRunner = zigferencer.llama_runner.LlamaRunner;
const Sessions = zigferencer.session.Sessions;
const limits = zigferencer.limits;
const LlamaFusedRunner = zigferencer.llama_fused.LlamaFusedRunner;

// -- fixtures ---------------------------------------------------------------

const prompts = [_][]const u8{ "The dragon flew over", "Tell me a story about" };
const n_decode: usize = 12;
const n_tokens = n_decode + 1;

/// Build the plan for the fused runner: `n_slots` streams at the multi-slot
/// test window, and no device budget.
fn fusedPlan(
    gpa: std.mem.Allocator,
    loaded: *const loader.Loaded,
    n_slots: u32,
) !zigferencer.kv_budget.KvPlan {
    const caps = try LlamaFusedRunner.kvCaps(
        gpa,
        loaded,
        limits.contextCeiling(loaded.card.context_length, null),
        0,
        n_slots,
        .f32,
    );
    return zigferencer.kv_budget.kvPlan(caps, LlamaFusedRunner.kvCostModel(loaded.card, .f32), .{
        .window = @min(loaded.card.context_length, zigferencer.test_fakes.test_serve_window),
        .slots = n_slots,
        .batch = n_slots,
    });
}

// -- tests ------------------------------------------------------------------

test "llama GPU batched decode matches the CPU reference streams (greedy, exact)" {
    const io = std.testing.io;
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

    // The CPU reference runs each prompt alone.
    var reference: [prompts.len][n_tokens]u32 = undefined;
    {
        const max_seq = @min(loaded.card.context_length, zigferencer.test_fakes.test_window);
        for (prompt_ids, 0..) |ids, p| {
            var cpu_pool: test_policy.Pool = undefined;
            try test_policy.startPool(&cpu_pool, io, test_policy.policy());
            defer cpu_pool.deinit();

            var runner = try LlamaRunner.init(
                arena,
                &loaded,
                zigferencer.kv_budget.pinned(1, max_seq),
                0,
                &cpu_pool,
                test_policy.policy(),
                .f32,
            );
            var pool = try Sessions(LlamaRunner).init(arena, &runner);
            const sid = try pool.create(0, .{});
            reference[p][0] = (try pool.feed(sid, ids, true)).?;
            for (1..n_tokens) |k| {
                var out: [1]u32 = undefined;
                try pool.decodeBatch(&.{sid}, &out);
                reference[p][k] = out[0];
            }
        }
    }

    // GPU batched: one fused runner, two sessions, interleaved ticks.
    const ctxp = try arena.create(zigferencer.vulkan_ctx.Context);
    ctxp.* = zigferencer.vulkan_ctx.Context.init(arena, .{}) catch return error.SkipZigTest;
    defer ctxp.deinit();

    const fused = try LlamaFusedRunner.initFromLoaded(
        gpa,
        ctxp,
        &loaded,
        try fusedPlan(gpa, &loaded, prompts.len + 1),
        .f32,
    );
    defer fused.deinit();

    var pool = try Sessions(LlamaFusedRunner).init(arena, fused);
    var sids: [prompts.len]u32 = undefined;
    var got: [prompts.len][n_tokens]u32 = undefined;
    for (prompt_ids, 0..) |ids, p| {
        sids[p] = try pool.create(0, .{});
        got[p][0] = (try pool.feed(sids[p], ids, true)).?;
    }

    for (1..n_tokens) |k| {
        var out: [prompts.len]u32 = undefined;
        try pool.decodeBatch(&sids, &out);
        for (0..prompts.len) |p| {
            got[p][k] = out[p];
        }
    }

    for (reference, got) |ref, g| {
        try std.testing.expectEqualSlices(u32, &ref, &g);
    }

    // Fork session 0 on the device. The pair must emit the same token every tick.
    const target = try pool.clone(sids[0], 0, .{});
    var pair: [2]u32 = undefined;
    for (0..4) |_| {
        try pool.decodeBatch(&.{ sids[0], target }, &pair);
        try std.testing.expectEqual(pair[0], pair[1]);
    }
}

test "llama GPU composed advance: a multi-row prefill chunk riding a decode row stays bit-exact" {
    const io = std.testing.io;
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

    // The CPU solo references: the first decode token of the front session and
    // the first token of the late session.
    var ref_front1: u32 = undefined; // prompt 0, decode token after prefill
    var ref_late0: u32 = undefined; // prompt 1, first token after prefill
    {
        const max_seq = @min(loaded.card.context_length, zigferencer.test_fakes.test_window);
        var cpu_pool: test_policy.Pool = undefined;
        try test_policy.startPool(&cpu_pool, io, test_policy.policy());
        defer cpu_pool.deinit();

        var r0 = try LlamaRunner.init(
            arena,
            &loaded,
            zigferencer.kv_budget.pinned(1, max_seq),
            0,
            &cpu_pool,
            test_policy.policy(),
            .f32,
        );
        var p0 = try Sessions(LlamaRunner).init(arena, &r0);
        const s0 = try p0.create(0, .{});
        _ = (try p0.feed(s0, prompt_ids[0], true)).?;
        var o0: [1]u32 = undefined;
        try p0.decodeBatch(&.{s0}, &o0);
        ref_front1 = o0[0];

        var r1 = try LlamaRunner.init(
            arena,
            &loaded,
            zigferencer.kv_budget.pinned(1, max_seq),
            0,
            &cpu_pool,
            test_policy.policy(),
            .f32,
        );
        var p1 = try Sessions(LlamaRunner).init(arena, &r1);
        const s1 = try p1.create(0, .{});
        ref_late0 = (try p1.feed(s1, prompt_ids[1], true)).?;
    }

    const ctxp = try arena.create(zigferencer.vulkan_ctx.Context);
    ctxp.* = zigferencer.vulkan_ctx.Context.init(arena, .{}) catch return error.SkipZigTest;
    defer ctxp.deinit();

    const fused = try LlamaFusedRunner.initFromLoaded(
        gpa,
        ctxp,
        &loaded,
        try fusedPlan(gpa, &loaded, prompts.len + 1),
        .f32,
    );
    defer fused.deinit();

    var pool = try Sessions(LlamaFusedRunner).init(arena, fused);
    const front = try pool.create(0, .{});
    _ = (try pool.feed(front, prompt_ids[0], true)).?; // front ready, pending its first token

    // The whole prompt of the late session goes in as one final chunk, composed
    // with the decode step of the front session.
    const late = try pool.create(0, .{});
    var out: [1]u32 = undefined;
    const first_late = try pool.advance(
        .{ .id = late, .tokens = prompt_ids[1], .final = true },
        &.{front},
        &out,
    );
    try std.testing.expectEqual(ref_front1, out[0]);
    try std.testing.expectEqual(ref_late0, first_late.?);
}

test "llama GPU prefix restore: a warm-restored prefix continues bit-exactly" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[0], true);
    std.debug.assert(ids.len >= 2); // the split needs a prefix row and a suffix row
    const n: u32 = @intCast(ids.len - 1);

    const ctxp = try arena.create(zigferencer.vulkan_ctx.Context);
    ctxp.* = zigferencer.vulkan_ctx.Context.init(arena, .{}) catch return error.SkipZigTest;
    defer ctxp.deinit();

    const fused = try LlamaFusedRunner.initFromLoaded(
        gpa,
        ctxp,
        &loaded,
        try fusedPlan(gpa, &loaded, 2),
        .f32,
    );
    defer fused.deinit();

    var pool = try Sessions(LlamaFusedRunner).init(arena, fused);

    // Cold: feed the prefix without logits, snapshot those rows, then feed the suffix.
    var cold_stream: [n_tokens]u32 = undefined;
    const cold = try pool.create(0, .{});
    _ = try pool.feed(cold, ids[0..n], false);
    const blob = try arena.alloc(u8, @intCast(pool.snapshotBytesFor(n)));
    try pool.snapshotPrefix(cold, n, blob);
    cold_stream[0] = (try pool.feed(cold, ids[n..], true)).?;
    const cold_pos = pool.posOf(cold);
    for (1..n_tokens) |k| {
        var out: [1]u32 = undefined;
        try pool.decodeBatch(&.{cold}, &out);
        cold_stream[k] = out[0];
    }

    // Warm: a fresh session seeded from the blob, then the identical suffix.
    var warm_stream: [n_tokens]u32 = undefined;
    const warm = try pool.create(0, .{});
    try pool.restorePrefix(warm, n, blob);
    warm_stream[0] = (try pool.feed(warm, ids[n..], true)).?;
    try std.testing.expectEqual(cold_pos, pool.posOf(warm));
    for (1..n_tokens) |k| {
        var out: [1]u32 = undefined;
        try pool.decodeBatch(&.{warm}, &out);
        warm_stream[k] = out[0];
    }

    try std.testing.expectEqualSlices(u32, &cold_stream, &warm_stream);
}
