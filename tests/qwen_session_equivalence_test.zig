//! Session gate for the qwen CPU runner. Time-sliced multi-session decode must be bit-identical to
//! running each session alone.
//!
//! Routing. Rows execute one at a time through the exact single-row forward, against the KV and the
//! GDN state of their own slot. Nothing numeric is left to differ, so this test pins the state
//! routing. A slot that reads the recurrent state of another slot changes the tokens at once.
//!
//! Passes. Seven passes state the claim.
//!
//!   A. each prompt alone on a 1-slot runner, the `run` shape.
//!   B. one 2-slot runner with both sessions interleaved per tick.
//!   C. chunk-fed prefill on the 2-slot runner.
//!   D. slot reuse. The pass destroys a session, creates a new one on the same slot and replays
//!      prompt 0, which pins `resetSlotState`. The GDN scan is a running accumulator, so stale
//!      state corrupts a fresh conversation.
//!   E. continuous. Prompt 1 re-prefills through `advance` chunks that share the decode ticks of
//!      prompt 0, the composed-tick shape of the serve scheduler. Mixed rows must not disturb the
//!      GDN state routing of either slot.
//!   F. clone. A session forked partway through a conversation continues bit-identically to its
//!      source, which pins the KV row copy and the GDN slab copy.
//!   G. forced parallelism. Pass B with the row-parallel matmuls and the attention units forced
//!      on. A unit is one KV group over one part of an attention row.
//!
//! Exactness. The token streams and the logits-row hashes must equal pass A exactly. Pass G sets
//! `min_parallel_flops = 0`, which spreads the row-parallel matmuls and the attention units
//! (parallel.attentionTyped) across 4 threads, even at short rows. The matmul partition keeps the
//! reduction order of every output. An attention unit writes the same records on any worker, and
//! the merge adds them in part order.
//!
//! Compact logits. A single-session feed lands its final row at logits[0]. In an `advance` tick the
//! decode row of the front session follows the final row of the chunk only when that row took an
//! lm_head. It sits at base, which is 1 on the final chunk and 0 otherwise.
//!
//! Prefix restore. This is the CPU twin of the warm-equals-cold rule that the hybrid runner
//! carries. The blob holds the full-attention KV rows and the whole recurrent and conv slabs of
//! every GDN layer, so the test pins the slab capture. A missed or misplaced region diverges the
//! greedy streams at once.
//!
//! Running. The Qwen3.5-2B fixture comes from `test_fixtures.path`. The model is 2 GB, and a Debug
//! test binary runs these tests for minutes. Run them in ReleaseSafe:
//!   zig build test -Doptimize=ReleaseSafe

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;
const QwenRunner = zigferencer.qwen_runner.QwenRunner;
const Sessions = zigferencer.session.Sessions;
const KvDtype = zigferencer.dtype.KvDtype;

// -- fixtures ---------------------------------------------------------------

const prompts = [_][]const u8{ "The dragon flew over", "Tell me a story about" };
const n_decode: usize = 11;
const n_tokens = n_decode + 1;

const Stream = struct {
    tokens: [n_tokens]u32,
    logits_hash: [n_tokens]u64,
};

fn hashLogitsRow(runner: *const QwenRunner, row: usize) u64 {
    const vocab = runner.logits.len / runner.n_slots;
    return std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(runner.logits[row * vocab ..][0..vocab]));
}

/// The thread policy of every runner here but pass G: the `-Dtest-threads`
/// pool with a floor of zero.
const default_policy: test_policy.Policy = test_policy.policy();

/// A CPU runner over the fixture at a stated KV dtype and thread policy. Every
/// test here builds through this one, so the dtype and the policy reach the
/// constructor at a single place.
fn cpuRunner(
    arena: std.mem.Allocator,
    io: std.Io,
    cpu_pool: *test_policy.Pool,
    loaded: *const loader.Loaded,
    n_slots: u32,
    kv_dtype: KvDtype,
    policy: test_policy.Policy,
) !QwenRunner {
    return QwenRunner.init(
        arena,
        loaded,
        zigferencer.kv_budget.pinned(n_slots, zigferencer.test_fakes.test_window),
        0,
        io,
        cpu_pool,
        policy,
        kv_dtype,
    );
}

fn soloStream(
    arena: std.mem.Allocator,
    io: std.Io,
    loaded: *const loader.Loaded,
    ids: []const u32,
) !Stream {
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, default_policy);
    defer cpu_pool.deinit();

    var runner = try cpuRunner(arena, io, &cpu_pool, loaded, 1, .f32, default_policy);
    var pool = try Sessions(QwenRunner).init(arena, &runner);
    const sid = try pool.create(0, .{});
    var s: Stream = undefined;
    s.tokens[0] = (try pool.feed(sid, ids, true)).?;
    s.logits_hash[0] = hashLogitsRow(&runner, (ids.len - 1) % runner.n_slots);
    for (1..n_tokens) |k| {
        var out: [1]u32 = undefined;
        try pool.decodeBatch(&.{sid}, &out);
        s.tokens[k] = out[0];
        s.logits_hash[k] = hashLogitsRow(&runner, 0);
    }

    return s;
}

/// Decode `out.len` greedy tokens from `ids` on a fresh runner at `kv_dtype`.
fn greedyProbe(
    arena: std.mem.Allocator,
    io: std.Io,
    loaded: *const loader.Loaded,
    ids: []const u32,
    kv_dtype: KvDtype,
    out: []u32,
) !void {
    std.debug.assert(out.len >= 1);

    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, default_policy);
    defer cpu_pool.deinit();

    var runner = try cpuRunner(arena, io, &cpu_pool, loaded, 1, kv_dtype, default_policy);
    var pool = try Sessions(QwenRunner).init(arena, &runner);
    const sid = try pool.create(0, .{});
    out[0] = (try pool.feed(sid, ids, true)).?;
    for (1..out.len) |k| {
        var one: [1]u32 = undefined;
        try pool.decodeBatch(&.{sid}, &one);
        out[k] = one[0];
    }
}

// -- tests ------------------------------------------------------------------

test "qwen time-sliced sessions are bit-identical to solo runs, and slot reuse starts clean" {
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

    var reference: [prompts.len]Stream = undefined;
    for (prompt_ids, 0..) |ids, p| {
        reference[p] = try soloStream(arena, io, &loaded, ids);
    }

    // -- B: interleaved on one 2-slot runner --------------------------------
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, default_policy);
    defer cpu_pool.deinit();

    var runner = try cpuRunner(arena, io, &cpu_pool, &loaded, prompts.len, .f32, default_policy);
    var pool = try Sessions(QwenRunner).init(arena, &runner);
    var sids: [prompts.len]u32 = undefined;
    for (prompt_ids, 0..) |ids, p| {
        sids[p] = try pool.create(0, .{});
        const first = (try pool.feed(sids[p], ids, true)).?;
        try std.testing.expectEqual(reference[p].tokens[0], first);
        try std.testing.expectEqual(reference[p].logits_hash[0], hashLogitsRow(&runner, 0));
    }

    for (1..n_tokens) |k| {
        var out: [prompts.len]u32 = undefined;
        try pool.decodeBatch(&sids, &out);
        for (0..prompts.len) |p| {
            try std.testing.expectEqual(reference[p].tokens[k], out[p]);
            try std.testing.expectEqual(reference[p].logits_hash[k], hashLogitsRow(&runner, p));
        }
    }

    // -- C: chunk-fed prefill on fresh sessions -----------------------------
    for (0..prompts.len) |p| {
        pool.destroy(sids[p]);
    }

    for (prompt_ids, 0..) |ids, p| {
        sids[p] = try pool.create(0, .{});
        var fed: usize = 0;
        var chunk_len: usize = 1;
        var first: ?u32 = null;
        while (fed < ids.len) : (chunk_len += 1) {
            const n = @min(chunk_len, ids.len - fed);
            first = try pool.feed(sids[p], ids[fed..][0..n], fed + n == ids.len);
            fed += n;
        }

        try std.testing.expectEqual(reference[p].tokens[0], first.?);
    }

    for (1..n_tokens) |k| {
        var out: [prompts.len]u32 = undefined;
        try pool.decodeBatch(&sids, &out);
        for (0..prompts.len) |p| {
            try std.testing.expectEqual(reference[p].tokens[k], out[p]);
        }
    }

    // -- D: slot reuse must not inherit the previous session's GDN state ----
    pool.destroy(sids[0]);
    const reused = try pool.create(0, .{});
    try std.testing.expectEqual(sids[0], reused); // same slot, fresh conversation
    var again: Stream = undefined;
    again.tokens[0] = (try pool.feed(reused, prompt_ids[0], true)).?;
    for (1..n_tokens) |k| {
        var out: [1]u32 = undefined;
        try pool.decodeBatch(&.{reused}, &out);
        again.tokens[k] = out[0];
    }

    try std.testing.expectEqualSlices(u32, &reference[0].tokens, &again.tokens);

    // -- E: continuous batching ---------------------------------------------
    // Prompt 1 re-prefills through `advance` chunks that share the decode ticks
    // of prompt 0. Both streams must stay exactly the solo streams.
    pool.destroy(reused);
    pool.destroy(sids[1]);
    const front = try pool.create(0, .{});
    try std.testing.expectEqual(
        reference[0].tokens[0],
        (try pool.feed(front, prompt_ids[0], true)).?,
    );

    const late = try pool.create(0, .{});
    var k: usize = 1;
    var fed: usize = 0;
    var first_late: ?u32 = null;
    while (fed < prompt_ids[1].len) : (fed += 1) {
        var out: [1]u32 = undefined;
        const final = fed + 1 == prompt_ids[1].len;
        first_late = try pool.advance(.{
            .id = late,
            .tokens = prompt_ids[1][fed..][0..1],
            .final = final,
        }, &.{front}, &out);
        // The decode row sits at base.
        const base: usize = if (final) 1 else 0;
        try std.testing.expectEqual(reference[0].tokens[k], out[0]);
        try std.testing.expectEqual(reference[0].logits_hash[k], hashLogitsRow(&runner, base));
        k += 1;
    }

    try std.testing.expectEqual(reference[1].tokens[0], first_late.?);
    try std.testing.expectEqual(reference[1].logits_hash[0], hashLogitsRow(&runner, 0));

    var k_late: usize = 1;
    while (k < n_tokens) : ({
        k += 1;
        k_late += 1;
    }) {
        var out: [2]u32 = undefined;
        try pool.decodeBatch(&.{ front, late }, &out);
        try std.testing.expectEqual(reference[0].tokens[k], out[0]);
        try std.testing.expectEqual(reference[1].tokens[k_late], out[1]);
    }

    // -- F: clone -----------------------------------------------------------
    // The greedy continuation of the fork must equal the solo stream of the source.
    pool.destroy(front);
    pool.destroy(late);
    const source = try pool.create(0, .{});
    try std.testing.expectEqual(
        reference[0].tokens[0],
        (try pool.feed(source, prompt_ids[0], true)).?,
    );
    for (1..4) |k2| {
        var out: [1]u32 = undefined;
        try pool.decodeBatch(&.{source}, &out);
        try std.testing.expectEqual(reference[0].tokens[k2], out[0]);
    }

    const target = try pool.clone(source, 0, .{});
    for (4..n_tokens) |k2| {
        var out: [2]u32 = undefined;
        try pool.decodeBatch(&.{ source, target }, &out);
        try std.testing.expectEqual(reference[0].tokens[k2], out[0]);
        try std.testing.expectEqual(reference[0].tokens[k2], out[1]);
    }

    // -- G: forced parallelism ----------------------------------------------
    // Both streams must equal pass A exactly.
    {
        const forced: test_policy.Policy = .{ .n_threads = 4, .min_parallel_flops = 0 };
        var par_cpu_pool: test_policy.Pool = undefined;
        try test_policy.startPool(&par_cpu_pool, io, forced);
        defer par_cpu_pool.deinit();

        var par_runner = try cpuRunner(
            arena,
            io,
            &par_cpu_pool,
            &loaded,
            prompts.len,
            .f32,
            forced,
        );
        var par_pool = try Sessions(QwenRunner).init(arena, &par_runner);
        var par_sids: [prompts.len]u32 = undefined;
        for (prompt_ids, 0..) |ids, p| {
            par_sids[p] = try par_pool.create(0, .{});
            const first = (try par_pool.feed(par_sids[p], ids, true)).?;
            try std.testing.expectEqual(reference[p].tokens[0], first);
            try std.testing.expectEqual(reference[p].logits_hash[0], hashLogitsRow(&par_runner, 0));
        }

        for (1..n_tokens) |k2| {
            var out: [prompts.len]u32 = undefined;
            try par_pool.decodeBatch(&par_sids, &out);
            for (0..prompts.len) |p| {
                try std.testing.expectEqual(reference[p].tokens[k2], out[p]);
                try std.testing.expectEqual(
                    reference[p].logits_hash[k2],
                    hashLogitsRow(&par_runner, p),
                );
            }
        }
    }
}

test "qwen prefix restore: a warm-restored prefix continues bit-exactly" {
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

    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, default_policy);
    defer cpu_pool.deinit();

    var runner = try cpuRunner(arena, io, &cpu_pool, &loaded, 2, .f32, default_policy);
    var pool = try Sessions(QwenRunner).init(arena, &runner);

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

test "qwen kv f16: the kv slab halves and the blob round-trips" {
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

    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, default_policy);
    defer cpu_pool.deinit();

    var wide = try cpuRunner(arena, io, &cpu_pool, &loaded, 1, .f32, default_policy);
    var narrow = try cpuRunner(arena, io, &cpu_pool, &loaded, 1, .f16, default_policy);

    // The slab prices whole rows, so f16 halves the storage exactly.
    try std.testing.expectEqual(wide.kv_slots[0].sizeBytes(), 2 * narrow.kv_slots[0].sizeBytes());

    // The blob narrows its KV half only. The GDN state stays f32, so both blobs share it.
    const n_rows: u32 = @intCast(ids.len);
    const per_step: u32 = wide.n_kv_heads * wide.head_dim;
    const n_full: u32 = wide.kv_slots[0].n_layers;
    const kv_wide: u64 = @as(u64, 2) * n_full * n_rows * KvDtype.f32.rowBytes(per_step);
    const kv_narrow: u64 = @as(u64, 2) * n_full * n_rows * KvDtype.f16.rowBytes(per_step);
    const bytes_wide = wide.evictSlotBytesFor(n_rows);
    const bytes_narrow = narrow.evictSlotBytesFor(n_rows);
    try std.testing.expectEqual(kv_wide - kv_narrow, bytes_wide - bytes_narrow);
    try std.testing.expectEqual(bytes_wide - kv_wide, bytes_narrow - kv_narrow);
    try std.testing.expect(bytes_wide - kv_wide > 0); // the GDN remainder is real

    // An evict, a restore and a second evict must reproduce the same bytes at
    // f16. A byte range that misframes the dtype shifts every region after it.
    var pool = try Sessions(QwenRunner).init(arena, &narrow);
    const sid = try pool.create(0, .{});
    _ = try pool.feed(sid, ids, true);
    const first = try arena.alloc(u8, @intCast(bytes_narrow));
    const second = try arena.alloc(u8, @intCast(bytes_narrow));
    try narrow.evictSlot(0, n_rows, first);
    try narrow.restoreSlot(0, n_rows, first);
    try narrow.evictSlot(0, n_rows, second);
    try std.testing.expectEqualSlices(u8, first, second);
    try std.testing.expect(!std.mem.allEqual(u8, first, 0));
}

test "qwen kv f16: greedy divergence against f32 stays coherent" {
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

    const n_probe = 32;
    var wide_toks: [n_probe]u32 = undefined;
    var narrow_toks: [n_probe]u32 = undefined;
    try greedyProbe(arena, io, &loaded, ids, .f32, &wide_toks);
    try greedyProbe(arena, io, &loaded, ids, .f16, &narrow_toks);

    var diverge: usize = n_probe;
    for (wide_toks, narrow_toks, 0..) |a, b, i| {
        if (a != b) {
            diverge = i;
            break;
        }
    }

    // A narrower cache can diverge later, because an argmax tie flips on the last bit.
    // A divergence at token 0 means the commit or the attention read the wrong bytes.
    // Print only on failure, because the build runner labels any stderr write as a failure.
    if (diverge == 0) {
        std.debug.print(
            "qwen kv f16 vs f32: greedy diverged at index {d} of {d} (f32 {d}, f16 {d})\n",
            .{ diverge, n_probe, wide_toks[0], narrow_toks[0] },
        );
        return error.KvF16DivergedAtFirstToken;
    }
}
