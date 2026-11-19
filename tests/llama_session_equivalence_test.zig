//! Batch-equivalence gate for the llama family. Batched decode must produce tokens and logits that
//! are bit-identical to running each session alone.
//!
//! Passes. Three prompts run six ways against a real llama-family GGUF.
//!
//!   a. alone: one 1-slot runner per prompt, the single-sequence shape.
//!   b. batched: one 3-slot runner, whole prompts, one `decodeBatch` tick per token.
//!   c. chunk-fed: pass b with each prompt split over several `feed` calls, the chunked-prefill
//!      shape.
//!   d. continuous: a late prompt prefills through `advance` chunks that share the decode ticks of
//!      the other two sessions, the composed-tick shape of the serve scheduler.
//!   e. clone: a session forked partway through a conversation continues bit-identically to its
//!      source, which pins the KV row copy.
//!   f. forced parallelism: pass b with the output-parallel matmuls and the attention units forced
//!      on. A unit is one KV group over one part of an attention row.
//!
//! Exactness. The token streams and the FNV-1a hashes of the raw logits bytes must match pass a
//! exactly, and not within a tolerance. The batch linear kernels repeat the per-(row, output)
//! reduction order of their B=1 siblings. Every per-row op reads exactly the inputs a lone call
//! produces. Greedy sampling uses no RNG. Nothing is left to differ.
//!
//! Compact logits. A single-session feed lands its final row at logits[0]. In an `advance` tick the
//! want-logits rows pack at logits[0..]. The decode rows follow the final row of the chunk only
//! when that row took an lm_head. So they sit at base+p, where base is 1 on the final chunk and 0
//! otherwise.
//!
//! Forced parallelism. `min_parallel_flops = 0` spreads the output-parallel matmuls and the
//! attention units (parallel.attentionRowsTyped) across `n_threads`. The matmul partition keeps the
//! reduction order of every output. An attention unit writes the same records on any worker, and
//! the merge adds them in part order. So multi-slot and multi-thread output is bit-exact. This pins
//! the determinism of the parallel paths that the serve runner uses.
//!
//! Prefix restore. This is the prefix-cache path of the serve scheduler on the CPU runner. The GPU
//! equivalence file pins the twin path for the fused runner. A cold session feeds the whole prompt
//! and snapshots its first `n` rows. A warm session restores those rows from the blob and prefills
//! the one-token suffix only. Warm must equal cold token for token.
//!
//! The tests read the canonical SmolLM2-135M-Instruct-Q8_0 from `test_fixtures.path`.
const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;
const LlamaCpu = zigferencer.family.Of(.llama).Cpu;
const Sessions = zigferencer.session.Sessions;

// -- fixtures ---------------------------------------------------------------

const prompts = [_][]const u8{ "Hello", "The dragon flew over", "Tell me a story about" };
const n_prompts = prompts.len;
const n_decode: usize = 15; // ticks after the first sampled token: 16 tokens total
const n_tokens = n_decode + 1;

const Stream = struct {
    tokens: [n_tokens]u32,
    logits_hash: [n_tokens]u64,
};

fn hashLogitsRow(runner: *const LlamaCpu, row: usize) u64 {
    const vocab = runner.logits.len / runner.n_slots;
    return std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(runner.logits[row * vocab ..][0..vocab]));
}

// -- tests ------------------------------------------------------------------

test "llama batched and chunk-fed decode are bit-identical to single-session runs" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const cfg: test_policy.Policy = .{ .n_threads = 1 };
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, cfg);
    defer cpu_pool.deinit();

    const max_seq = @min(loaded.card.context_length, zigferencer.test_fakes.test_window);

    var prompt_ids: [n_prompts][]const u32 = undefined;
    for (prompts, 0..) |p, i| {
        prompt_ids[i] = try bpe.encodeAlloc(&v, arena, p, true);
    }

    // -- pass a: each prompt alone on a fresh 1-slot runner -----------------
    var reference: [n_prompts]Stream = undefined;
    for (prompt_ids, 0..) |ids, p| {
        var runner = try LlamaCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(1, max_seq),
            .max_rows = 0,
            .pool = &cpu_pool,
            .policy = cfg,
            .kv_dtype = .f32,
        });
        var pool = try Sessions(LlamaCpu).init(arena, &runner);
        const sid = try pool.create(0, .{});
        reference[p].tokens[0] = (try pool.feed(sid, ids, true)).?;
        reference[p].logits_hash[0] = hashLogitsRow(&runner, (ids.len - 1) % runner.n_slots);
        for (1..n_tokens) |k| {
            var out: [1]u32 = undefined;
            try pool.decodeBatch(&.{sid}, &out);
            reference[p].tokens[k] = out[0];
            reference[p].logits_hash[k] = hashLogitsRow(&runner, 0);
        }
    }

    // -- pass b: one 3-slot runner, whole prompts, batched ticks ------------
    {
        var runner = try LlamaCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(n_prompts, max_seq),
            .max_rows = 0,
            .pool = &cpu_pool,
            .policy = cfg,
            .kv_dtype = .f32,
        });
        var pool = try Sessions(LlamaCpu).init(arena, &runner);
        var sids: [n_prompts]u32 = undefined;
        for (prompt_ids, 0..) |ids, p| {
            sids[p] = try pool.create(0, .{});
            const first = (try pool.feed(sids[p], ids, true)).?;
            try std.testing.expectEqual(reference[p].tokens[0], first);
            try std.testing.expectEqual(reference[p].logits_hash[0], hashLogitsRow(&runner, 0));
        }

        for (1..n_tokens) |k| {
            var out: [n_prompts]u32 = undefined;
            try pool.decodeBatch(&sids, &out);
            for (0..n_prompts) |p| {
                try std.testing.expectEqual(reference[p].tokens[k], out[p]);
                try std.testing.expectEqual(reference[p].logits_hash[k], hashLogitsRow(&runner, p));
            }
        }
    }

    // -- pass c: chunk-fed prefill, 1 token, then 2 tokens, then the rest ---
    {
        var runner = try LlamaCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(n_prompts, max_seq),
            .max_rows = 0,
            .pool = &cpu_pool,
            .policy = cfg,
            .kv_dtype = .f32,
        });
        var pool = try Sessions(LlamaCpu).init(arena, &runner);
        var sids: [n_prompts]u32 = undefined;
        for (prompt_ids, 0..) |ids, p| {
            sids[p] = try pool.create(0, .{});
            var fed: usize = 0;
            var chunk_len: usize = 1;
            var first: ?u32 = null;
            while (fed < ids.len) : (chunk_len += 1) {
                const n = @min(chunk_len, ids.len - fed);
                const final = fed + n == ids.len;
                first = try pool.feed(sids[p], ids[fed..][0..n], final);
                fed += n;
            }

            try std.testing.expectEqual(reference[p].tokens[0], first.?);
        }

        for (1..n_tokens) |k| {
            var out: [n_prompts]u32 = undefined;
            try pool.decodeBatch(&sids, &out);
            for (0..n_prompts) |p| {
                try std.testing.expectEqual(reference[p].tokens[k], out[p]);
                try std.testing.expectEqual(reference[p].logits_hash[k], hashLogitsRow(&runner, p));
            }
        }
    }

    // -- pass d: continuous batching ----------------------------------------
    // Prompt 2 arrives late and prefills through `advance` chunks that share the decode ticks.
    // The decode rows must stay exactly the solo streams.
    {
        var runner = try LlamaCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(n_prompts, max_seq),
            .max_rows = 0,
            .pool = &cpu_pool,
            .policy = cfg,
            .kv_dtype = .f32,
        });
        var pool = try Sessions(LlamaCpu).init(arena, &runner);
        var sids: [2]u32 = undefined;
        for (0..2) |p| {
            sids[p] = try pool.create(0, .{});
            const first = (try pool.feed(sids[p], prompt_ids[p], true)).?;
            try std.testing.expectEqual(reference[p].tokens[0], first);
        }

        const late = try pool.create(0, .{});
        const ids_late = prompt_ids[2];
        var k: usize = 1;
        var fed: usize = 0;
        var first_late: ?u32 = null;
        while (fed < ids_late.len) : (fed += 1) {
            var out: [2]u32 = undefined;
            const final = fed + 1 == ids_late.len;
            first_late = try pool.advance(.{
                .id = late,
                .tokens = ids_late[fed..][0..1],
                .final = final,
            }, &sids, &out);
            // The decode rows sit at base+p.
            const base: usize = if (final) 1 else 0;
            for (0..2) |p| {
                try std.testing.expectEqual(reference[p].tokens[k], out[p]);
                try std.testing.expectEqual(
                    reference[p].logits_hash[k],
                    hashLogitsRow(&runner, base + p),
                );
            }

            k += 1;
        }

        try std.testing.expectEqual(reference[2].tokens[0], first_late.?);
        try std.testing.expectEqual(reference[2].logits_hash[0], hashLogitsRow(&runner, 0));

        // All three decode together for the remaining ticks.
        const all: [3]u32 = .{ sids[0], sids[1], late };
        var k_late: usize = 1;
        while (k < n_tokens) : ({
            k += 1;
            k_late += 1;
        }) {
            var out: [3]u32 = undefined;
            try pool.decodeBatch(&all, &out);
            for (0..2) |p| {
                try std.testing.expectEqual(reference[p].tokens[k], out[p]);
                try std.testing.expectEqual(reference[p].logits_hash[k], hashLogitsRow(&runner, p));
            }

            try std.testing.expectEqual(reference[2].tokens[k_late], out[2]);
            try std.testing.expectEqual(
                reference[2].logits_hash[k_late],
                hashLogitsRow(&runner, 2),
            );
        }
    }

    // -- pass e: clone ------------------------------------------------------
    // The greedy continuation of the fork must equal the solo stream of the source.
    // The source must stay undisturbed.
    {
        var runner = try LlamaCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(2, max_seq),
            .max_rows = 0,
            .pool = &cpu_pool,
            .policy = cfg,
            .kv_dtype = .f32,
        });
        var pool = try Sessions(LlamaCpu).init(arena, &runner);
        const source = try pool.create(0, .{});
        try std.testing.expectEqual(
            reference[0].tokens[0],
            (try pool.feed(source, prompt_ids[0], true)).?,
        );
        for (1..4) |k| {
            var out: [1]u32 = undefined;
            try pool.decodeBatch(&.{source}, &out);
            try std.testing.expectEqual(reference[0].tokens[k], out[0]);
        }

        const target = try pool.clone(source, 0, .{});
        for (4..n_tokens) |k| {
            var out: [2]u32 = undefined;
            try pool.decodeBatch(&.{ source, target }, &out);
            try std.testing.expectEqual(reference[0].tokens[k], out[0]);
            try std.testing.expectEqual(reference[0].tokens[k], out[1]);
            try std.testing.expectEqual(reference[0].logits_hash[k], hashLogitsRow(&runner, 0));
            try std.testing.expectEqual(reference[0].logits_hash[k], hashLogitsRow(&runner, 1));
        }
    }

    // -- pass f: forced parallelism -----------------------------------------
    // Batched decode under forced parallelism must equal the serial reference.
    {
        const cfg_par: test_policy.Policy = .{ .n_threads = 4, .min_parallel_flops = 0 };
        var par_cpu_pool: test_policy.Pool = undefined;
        try test_policy.startPool(&par_cpu_pool, io, cfg_par);
        defer par_cpu_pool.deinit();

        var runner = try LlamaCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(n_prompts, max_seq),
            .max_rows = 0,
            .pool = &par_cpu_pool,
            .policy = cfg_par,
            .kv_dtype = .f32,
        });
        var pool = try Sessions(LlamaCpu).init(arena, &runner);
        var sids: [n_prompts]u32 = undefined;
        for (prompt_ids, 0..) |ids, p| {
            sids[p] = try pool.create(0, .{});
            const first = (try pool.feed(sids[p], ids, true)).?;
            try std.testing.expectEqual(reference[p].tokens[0], first);
            try std.testing.expectEqual(reference[p].logits_hash[0], hashLogitsRow(&runner, 0));
        }

        for (1..n_tokens) |k| {
            var out: [n_prompts]u32 = undefined;
            try pool.decodeBatch(&sids, &out);
            for (0..n_prompts) |p| {
                try std.testing.expectEqual(reference[p].tokens[k], out[p]);
                try std.testing.expectEqual(reference[p].logits_hash[k], hashLogitsRow(&runner, p));
            }
        }
    }
}

test "llama CPU prefix restore: a warm-restored prefix continues bit-exactly" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const cfg: test_policy.Policy = .{ .n_threads = 1 };
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, cfg);
    defer cpu_pool.deinit();

    const max_seq = @min(loaded.card.context_length, zigferencer.test_fakes.test_window);
    const ids = try bpe.encodeAlloc(&v, arena, prompts[1], true);
    std.debug.assert(ids.len >= 2); // the split needs a prefix row and a suffix row
    const n: u32 = @intCast(ids.len - 1);

    var runner = try LlamaCpu.init(arena, &loaded, .{
        .plan = zigferencer.kv_budget.pinned(2, max_seq),
        .max_rows = 0,
        .pool = &cpu_pool,
        .policy = cfg,
        .kv_dtype = .f32,
    });
    var pool = try Sessions(LlamaCpu).init(arena, &runner);

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
