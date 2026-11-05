//! Chunked-prefill gate for the CPU runners. A prompt fed through
//! `prefillBatch` in chunks must give the same last-row logits, bit for bit,
//! and the same greedy continuation as the prompt fed one token at a time
//! through `step`.
//!
//! The reference path runs `step` per token at one thread with `max_rows = 0`,
//! the shape of the per-token prefill. The chunked paths run chunks of 1, 17,
//! and 64 rows over a 70-token prompt, so a chunk boundary falls inside and
//! outside the 4-row groups of the batch kernel. Each chunked path runs at one
//! thread and at four threads with the fan-out forced on.
//!
//! The SmolLM2 and Qwen3.5-2B fixtures come from `test_fixtures.path`.

const std = @import("std");
const z = @import("zigferencer");
const tp = @import("test_policy.zig");
const loader = z.loader;
const vocab_mod = z.vocab;
const bpe = z.bpe;
const LlamaRunner = z.llama_runner.LlamaRunner;
const QwenRunner = z.qwen_runner.QwenRunner;

// -- fixtures ---------------------------------------------------------------

const prompt_text = "The dragon flew over the quiet valley at dawn, and the people of the " ++
    "old town looked up from their fields. Tell me a story about what they saw " ++
    "next, and why the river turned silver before the bells rang at noon. The " ++
    "miller counted the boats, the children ran to the bridge, and nobody spoke. " ++
    "At dusk the dragon came back, and it carried a lantern in its claws.";
const n_prompt: usize = 70;
const n_decode: usize = 8;
const max_seq: u32 = 256;
const chunk_sizes = [_]u32{ 1, 17, 64 };
const serial_policy: tp.Policy = .{ .n_threads = 1 };
const forced_policy: tp.Policy = .{ .n_threads = 4, .min_parallel_flops = 0 };

comptime {
    // The widest chunk here is the production chunk, and the prompt spans
    // more than one of them, so a short tail attends to a full chunk.
    std.debug.assert(chunk_sizes[chunk_sizes.len - 1] == z.limits.cpu_prefill_chunk);
    std.debug.assert(n_prompt > z.limits.cpu_prefill_chunk);
    std.debug.assert(n_prompt + n_decode <= max_seq);
}

/// The first `n_prompt` ids of the prompt text under the fixture vocabulary.
fn promptIds(arena: std.mem.Allocator, loaded: *const loader.Loaded) ![]const u32 {
    var v = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    const ids = try bpe.encodeAlloc(&v, arena, prompt_text, true);
    try std.testing.expect(ids.len >= n_prompt);
    return ids[0..n_prompt];
}

const Outcome = struct {
    last_logits: []const f32,
    tokens: [n_decode]u32,
};

/// Greedy-decode `n_decode` tokens from `logits` at `pos` through `step`.
fn decodeTail(
    comptime Runner: type,
    runner: *Runner,
    logits: []const f32,
    pos: u32,
    out: *[n_decode]u32,
) !void {
    var tok = z.sampler.greedy(logits);
    for (out, 0..) |*o, k| {
        o.* = tok;
        const lg = try runner.step(tok, pos + @as(u32, @intCast(k)));
        tok = z.sampler.greedy(lg);
    }
}

/// The per-token reference: `step` over every prompt id at one thread.
fn referenceLeg(
    comptime Runner: type,
    arena: std.mem.Allocator,
    loaded: *const loader.Loaded,
    ids: []const u32,
) !Outcome {
    const io = std.testing.io;
    var cpu_pool: tp.Pool = undefined;
    try tp.startPool(&cpu_pool, io, serial_policy);
    defer cpu_pool.deinit();

    var runner = try Runner.init(
        arena,
        loaded,
        z.kv_budget.pinned(1, max_seq),
        0,
        io,
        &cpu_pool,
        serial_policy,
        .f32,
    );
    var last: []f32 = undefined;
    for (ids, 0..) |id, i| {
        last = try runner.step(id, @intCast(i));
    }

    var o: Outcome = .{ .last_logits = try arena.dupe(f32, last), .tokens = undefined };
    try decodeTail(Runner, &runner, last, @intCast(ids.len), &o.tokens);
    return o;
}

/// The chunked path: `prefillBatch` in chunks of `chunk` rows on a runner that
/// holds the production row scratch.
fn chunkedLeg(
    comptime Runner: type,
    arena: std.mem.Allocator,
    loaded: *const loader.Loaded,
    ids: []const u32,
    chunk: u32,
    policy: tp.Policy,
) !Outcome {
    const io = std.testing.io;
    var cpu_pool: tp.Pool = undefined;
    try tp.startPool(&cpu_pool, io, policy);
    defer cpu_pool.deinit();

    var runner = try Runner.init(
        arena,
        loaded,
        z.kv_budget.pinned(1, max_seq),
        z.limits.cpu_prefill_chunk,
        io,
        &cpu_pool,
        policy,
        .f32,
    );
    try std.testing.expectEqual(z.limits.cpu_prefill_chunk, runner.prefill_batch_max);
    var pos: u32 = 0;
    var last: []f32 = undefined;
    var calls: u32 = 0;
    while (pos < ids.len) {
        const n: u32 = @min(@as(u32, @intCast(ids.len)) - pos, chunk);
        last = try runner.prefillBatch(ids[pos..][0..n], pos);
        pos += n;
        calls += 1;
    }

    try std.testing.expectEqual(std.math.divCeil(usize, ids.len, chunk) catch unreachable, calls);
    var o: Outcome = .{ .last_logits = try arena.dupe(f32, last), .tokens = undefined };
    try decodeTail(Runner, &runner, last, pos, &o.tokens);
    return o;
}

/// Every chunk size at both thread policies against the per-token reference.
fn expectChunksMatch(
    comptime Runner: type,
    arena: std.mem.Allocator,
    loaded: *const loader.Loaded,
) !void {
    const ids = try promptIds(arena, loaded);
    const ref = try referenceLeg(Runner, arena, loaded, ids);
    for ([_]tp.Policy{ serial_policy, forced_policy }) |policy| {
        for (chunk_sizes) |chunk| {
            const got = try chunkedLeg(Runner, arena, loaded, ids, chunk, policy);
            // Bytes, not floats: the claim is bit identity, NaN included.
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(ref.last_logits),
                std.mem.sliceAsBytes(got.last_logits),
            );
            try std.testing.expectEqualSlices(u32, &ref.tokens, &got.tokens);
        }
    }
}

// -- tests ------------------------------------------------------------------

test "llama chunked prefill is bit-identical to per-token step" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");
    var loaded = try loader.load(std.testing.io, arena, fixture);
    defer loaded.close();

    try expectChunksMatch(LlamaRunner, arena, &loaded);
}

test "qwen chunked prefill is bit-identical to per-token step" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");
    var loaded = try loader.load(std.testing.io, arena, fixture);
    defer loaded.close();

    try expectChunksMatch(QwenRunner, arena, &loaded);
}

test "llama stepBatch and prefillBatch refuse rows past the row scratch" {
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");
    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    // One slot and `max_rows = 0` hold one row of scratch.
    var cpu_pool: tp.Pool = undefined;
    try tp.startPool(&cpu_pool, io, serial_policy);
    defer cpu_pool.deinit();

    var runner = try LlamaRunner.init(
        arena,
        &loaded,
        z.kv_budget.pinned(1, max_seq),
        0,
        io,
        &cpu_pool,
        serial_policy,
        .f32,
    );
    try std.testing.expectEqual(@as(u32, 1), runner.max_rows);
    try std.testing.expectEqual(@as(u32, 1), runner.prefill_batch_max);
    const two = [_]z.step_row.StepRow{
        .{ .slot = 0, .token = 1, .pos = 0, .want_logits = false },
        .{ .slot = 0, .token = 1, .pos = 1, .want_logits = true },
    };
    try std.testing.expectError(error.TooManyRows, runner.stepBatch(&two));
    try std.testing.expectError(error.TooManyRows, runner.prefillBatch(&.{ 1, 1 }, 0));
    // The refusal leaves the runner usable: one row still steps.
    _ = try runner.prefillBatch(&.{1}, 0);
}

test "qwen stepBatch and prefillBatch refuse rows past the row scratch" {
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try z.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");
    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    var cpu_pool: tp.Pool = undefined;
    try tp.startPool(&cpu_pool, io, serial_policy);
    defer cpu_pool.deinit();

    var runner = try QwenRunner.init(
        arena,
        &loaded,
        z.kv_budget.pinned(1, max_seq),
        0,
        io,
        &cpu_pool,
        serial_policy,
        .f32,
    );
    try std.testing.expectEqual(@as(u32, 1), runner.max_rows);
    const two = [_]z.step_row.StepRow{
        .{ .slot = 0, .token = 1, .pos = 0, .want_logits = false },
        .{ .slot = 0, .token = 1, .pos = 1, .want_logits = true },
    };
    try std.testing.expectError(error.TooManyRows, runner.stepBatch(&two));
    try std.testing.expectError(error.TooManyRows, runner.prefillBatch(&.{ 1, 1 }, 0));
    _ = try runner.prefillBatch(&.{1}, 0);
}
