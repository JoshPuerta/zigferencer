//! Snapshot gate: the first N greedy-decoded token ids from a fixed prompt against the real
//! SmolLM2-135M-Instruct GGUFs (Q8_0 and Q4_0).
//!
//! Comptime. `checkSnapshot` takes the id count at comptime, so the expected array has a checked
//! size.
//!
//! Anchor. The ids must match the snapshot exactly, because the snapshot is the determinism anchor
//! of the suite. A regression in the GGUF parser, the tokenizer, any op, the KV cache, the
//! scheduler or the sampler surfaces here.
//!
//! Q4_0 file. The Q4_0 case also drives the q4_0 and q4_1 CPU dequant-and-dot kernels. That file is
//! a mixed quant: a q4_0 body, a few q4_1 ffn_down rows, and a q8_0 tied embedding.
//!
//! Capture. Both snapshots come from x86_64-linux with f32 activations and naive scalar kernels,
//! and with the BOS handling of `vocab.addBos`. The ids match the llama.cpp reference for the Q4_0
//! file byte for byte. The Q4_0 path tracks the reference over a long greedy prefix, before f32
//! near-tie drift flips a token.
//!
//! Changes. A divergence is a signal to investigate before anyone regenerates the ids. A snapshot
//! can change for a good reason, such as a new quantization. Copy the new ids into `expected` and
//! give the reason in the commit message.
//!
//! Files. The tests find the files through `test_fixtures.path`.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;
const LlamaRunner = zigferencer.llama_runner.LlamaRunner;
const sampler = zigferencer.sampler;

// -- fixtures ---------------------------------------------------------------

const name_q8_0 = "SmolLM2-135M-Instruct-Q8_0";
const name_q4_0 = "SmolLM2-135M-Instruct-Q4_0";

const prompt = "Hello";

/// Greedy-decoded first 5 token ids for "Hello" against SmolLM2-135M-Instruct-Q8_0.
const expected_q8_0 = [_]u32{ 198, 198, 57, 5248, 18948 };

/// Greedy-decoded first 8 token ids for "Hello" against the mixed-quant SmolLM2-135M-Instruct-Q4_0.
const expected_q4_0 = [_]u32{ 260, 1919, 506, 746, 30, 198, 198, 57 };

/// Greedy-decode `expected.len` tokens from `prompt` against the registry model `name` under
/// the policy `cfg`, and compare them to `expected`. The ids do not depend on the policy.
fn checkSnapshot(
    comptime n: usize,
    name: []const u8,
    expected: *const [n]u32,
    cfg: test_policy.Policy,
) !void {
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const path = try zigferencer.test_fixtures.path(arena, name);

    var loaded = try loader.load(io, arena, path);
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, cfg);
    defer cpu_pool.deinit();

    const window = @min(loaded.card.context_length, zigferencer.test_fakes.test_window);
    var runner = try LlamaRunner.init(
        arena,
        &loaded,
        zigferencer.kv_budget.pinned(1, window),
        0, // max_rows: single-sequence snapshot, no wide prefill chunk
        &cpu_pool,
        cfg,
        .f32,
    );

    const prompt_ids = try bpe.encodeAlloc(&v, arena, prompt, true);

    var pos: u32 = 0;
    var last_logits: []f32 = undefined;
    for (prompt_ids) |tid| {
        last_logits = try runner.step(tid, pos);
        pos += 1;
    }

    var generated: [n]u32 = undefined;
    for (&generated) |*g| {
        const id = sampler.greedy(last_logits);
        g.* = id;
        last_logits = try runner.step(id, pos);
        pos += 1;
    }

    if (!std.mem.eql(u32, expected, &generated)) {
        std.debug.print("\nSNAPSHOT MISMATCH for prompt {s} ({s})\n", .{ prompt, path });
        std.debug.print("  expected: {{ ", .{});
        for (expected) |id| {
            std.debug.print("{d}, ", .{id});
        }

        std.debug.print("}}\n  actual:   {{ ", .{});
        for (generated) |id| {
            std.debug.print("{d}, ", .{id});
        }

        std.debug.print("}}\n", .{});
        return error.SnapshotMismatch;
    }
}

// -- tests ------------------------------------------------------------------

test "SmolLM2-Q8_0 greedy first-5 IDs snapshot" {
    // The default policy of the host builds this runner, so the suite runs the
    // shipped thread count and floor at least once.
    try checkSnapshot(
        expected_q8_0.len,
        name_q8_0,
        &expected_q8_0,
        test_policy.Policy.forHost(std.testing.io),
    );
}

test "SmolLM2-Q4_0 greedy first-8 IDs snapshot" {
    try checkSnapshot(expected_q4_0.len, name_q4_0, &expected_q4_0, test_policy.policy());
}

test "SmolLM2-Q8_0 at kv f16: the slab halves and the blob round-trips" {
    // Within one dtype the f16 CPU placement frames half the bytes, and its
    // transport moves them byte for byte. Token equality across dtypes is not
    // the rule under test, so this test states no expected stream.
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const path = try zigferencer.test_fixtures.path(arena, name_q8_0);

    var loaded = try loader.load(io, arena, path);
    defer loaded.close();

    const cfg = test_policy.policy();
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, cfg);
    defer cpu_pool.deinit();

    const plan = zigferencer.kv_budget.pinned(1, 64);
    var wide = try LlamaRunner.init(arena, &loaded, plan, 0, &cpu_pool, cfg, .f32);
    var narrow = try LlamaRunner.init(arena, &loaded, plan, 0, &cpu_pool, cfg, .f16);
    var restored = try LlamaRunner.init(arena, &loaded, plan, 0, &cpu_pool, cfg, .f16);

    const rows: u32 = 8;
    try std.testing.expectEqual(wide.kv_slots[0].sizeBytes() / 2, narrow.kv_slots[0].sizeBytes());
    try std.testing.expectEqual(wide.evictSlotBytesFor(rows) / 2, narrow.evictSlotBytesFor(rows));

    // Commit real rows, so the blob carries converted values and not zeros.
    var pos: u32 = 0;
    while (pos < rows) : (pos += 1) {
        _ = try narrow.step(pos + 1, pos);
    }

    const blob = try arena.alloc(u8, @intCast(narrow.evictSlotBytesFor(rows)));
    const again = try arena.alloc(u8, blob.len);
    try narrow.evictSlot(0, rows, blob);
    try restored.restoreSlot(0, rows, blob);
    try restored.evictSlot(0, rows, again);
    try std.testing.expectEqualSlices(u8, blob, again);

    // A zeroed blob passes the round-trip above, so some byte must be
    // non-zero.
    var any: u8 = 0;
    for (blob) |b| {
        any |= b;
    }

    try std.testing.expect(any != 0);
}
