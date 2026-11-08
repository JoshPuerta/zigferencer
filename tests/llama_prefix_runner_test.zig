//! Equivalence gate for the llama prefix boundary. A runner that `initPrefix` builds at the layer
//! count of the model runs the same layers as the runner that `init` builds. Both runners use the
//! same weights and the same KV.
//!
//! Method. Two runners share one loaded model. The whole-model runner steps through `stepBatch`. A
//! composing runner drives the prefix runner in two calls, which embed into `x` and then call
//! `layerForward` per layer. The residual streams must match byte for byte. The final norm and the
//! lm_head write `x_norm`, so `x` still holds the output of the layer loop when `stepBatch`
//! returns.
//!
//! Exactness. Nothing here has a tolerance. Both runners call the same kernels in the same order
//! over the same bytes. A difference means the layer-count parameter moved something it must not
//! move.
//!
//! Fixture. The test needs a real GGUF, so it reads the canonical SmolLM2-135M-Instruct-Q8_0 from
//! `test_fixtures.path`.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const LlamaRunner = zigferencer.llama_runner.LlamaRunner;
const StepRow = zigferencer.step_row.StepRow;

// -- fixtures ---------------------------------------------------------------

/// Two positions cover a KV write and a KV read, so no runner here needs a
/// wider window than this.
const window: u32 = 16;

/// Drive one row through a runner from the outside. The call does the
/// embedding and then every layer the runner holds. `embedLookup` is the naive
/// function on every op backend, so this is the exact call `stepBatch` makes.
fn driveRow(runner: *LlamaRunner, row: StepRow) !void {
    const d = runner.card.embedding_length;
    zigferencer.ops.embedLookup(runner.embed.*, row.token, d, runner.x[0..d]);
    var layer: u32 = 0;
    while (layer < runner.n_layers) : (layer += 1) {
        runner.layerForward(layer, &.{row});
    }
}

/// Row 0 of the residual stream, as bytes, so the comparison is bitwise.
fn residual(runner: *const LlamaRunner) []const u8 {
    return std.mem.sliceAsBytes(runner.x[0..runner.card.embedding_length]);
}

// -- tests ------------------------------------------------------------------

test "llama initPrefix at the full layer count reproduces the whole-model residual stream" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    const policy: test_policy.Policy = .{ .n_threads = 1 };
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, policy);
    defer cpu_pool.deinit();

    const plan = zigferencer.kv_budget.pinned(1, window);

    var full = try LlamaRunner.init(arena, &loaded, plan, 0, &cpu_pool, policy, .f32);
    var prefix = try LlamaRunner.initPrefix(
        arena,
        &loaded,
        plan,
        loaded.card.n_layers,
        0,
        &cpu_pool,
        policy,
        .f32,
    );
    try std.testing.expectEqual(loaded.card.n_layers, full.n_layers);
    try std.testing.expectEqual(full.n_layers, prefix.n_layers);

    _ = try full.step(1, 0);
    try driveRow(&prefix, .{ .slot = 0, .token = 1, .pos = 0, .want_logits = false });
    try std.testing.expectEqualSlices(u8, residual(&full), residual(&prefix));

    // A second token at position 1 attends over the row both runners wrote, so
    // it covers the KV write and the KV read.
    _ = try full.step(2, 1);
    try driveRow(&prefix, .{ .slot = 0, .token = 2, .pos = 1, .want_logits = false });
    try std.testing.expectEqualSlices(u8, residual(&full), residual(&prefix));
}

test "llama initPrefix sizes the weight slices and the KV to its own layer count" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    const policy: test_policy.Policy = .{ .n_threads = 1 };
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, policy);
    defer cpu_pool.deinit();

    const plan = zigferencer.kv_budget.pinned(1, window);
    const n_layers = loaded.card.n_layers;
    std.debug.assert(n_layers >= 2); // the split needs a layer on each side

    const half: u32 = n_layers / 2;
    var full = try LlamaRunner.init(arena, &loaded, plan, 0, &cpu_pool, policy, .f32);
    var short = try LlamaRunner.initPrefix(
        arena,
        &loaded,
        plan,
        half,
        0,
        &cpu_pool,
        policy,
        .f32,
    );
    try std.testing.expectEqual(half, short.n_layers);
    try std.testing.expectEqual(@as(usize, half), short.attn_q_w.len);
    try std.testing.expectEqual(@as(usize, half), short.ffn_down_w.len);

    // A KV blob holds the layers the runner executes, so a half-height runner
    // carries a proportional share of the whole-model blob.
    const rows: u32 = 4;
    try std.testing.expectEqual(
        full.evictSlotBytesFor(rows) / n_layers * half,
        short.evictSlotBytesFor(rows),
    );

    // The narrowed runner runs its own layers, over its own KV slabs.
    try driveRow(&short, .{ .slot = 0, .token = 1, .pos = 0, .want_logits = false });
}
