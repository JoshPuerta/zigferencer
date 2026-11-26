//! Equivalence gate for the qwen prefix boundary. A runner that `initPrefix` builds at `k` runs
//! layers `[0..k)` exactly as the whole-model runner runs them. Both runners use the same weights,
//! the same KV rows and the same GDN slabs.
//!
//! Method. Two runners share one loaded model, and a composing runner drives both the same way,
//! which embeds into `x` and then calls `layerForward` per layer. The residual streams must match
//! byte for byte. Nothing here has a tolerance. Both runners call the same kernels in the same
//! order over the same bytes. A difference means the layer-count parameter moved something it must
//! not move.
//!
//! Blob. The blob check is the load-bearing one for the composed runner. The snapshot walks layers
//! in model order. The whole blob of the prefix must therefore equal the leading span of the blob
//! of the whole model. That is the boundary offset the hybrid needs, stated as a test.
//!
//! Fixtures. The Qwen3.5-2B fixture and the SmolLM2 fixture of the one non-qwen case come from
//! `test_fixtures.path`.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const QwenCpu = zigferencer.family.Of(.qwen35).Cpu;
const StepRow = zigferencer.step_row.StepRow;
const KvDtype = zigferencer.dtype.KvDtype;

// -- fixtures ---------------------------------------------------------------

/// Two positions cover a KV write, a KV read, and two GDN steps, so no runner
/// here needs a wider window than this.
const window: u32 = 16;

/// Every runner here runs serially.
const serial: test_policy.Policy = .{ .n_threads = 1 };

/// The splits under test. 6 holds one full-attention layer (absolute index 3),
/// 12 holds three. Both land mid-model on the 2B, where the compaction index and
/// the GDN slab vector both narrow.
const splits = [_]u32{ 6, 12 };

/// Two token ids every qwen vocabulary holds. The values do not matter. The
/// two runners must agree on whatever they compute.
const tokens = [_]u32{ 1, 2 };

/// A prefix runner over the fixture. Every runner here builds through this one,
/// so the plan and the thread count reach the constructor at a single place.
fn prefixRunner(
    arena: std.mem.Allocator,
    cpu_pool: *test_policy.Pool,
    loaded: *const loader.Loaded,
    n_layers: u32,
    n_slots: u32,
    kv_dtype: KvDtype,
) !QwenCpu {
    return QwenCpu.initPrefix(arena, loaded, n_layers, .{
        .plan = zigferencer.kv_budget.pinned(n_slots, window),
        .max_rows = 0,
        .pool = cpu_pool,
        .policy = serial,
        .kv_dtype = kv_dtype,
    });
}

/// Drive one row through the first `k` layers of a runner from the outside. `embedLookup` is the
/// naive function on every op backend, so this is the exact call `stepBatch` makes.
fn driveRow(runner: *QwenCpu, k: u32, row: StepRow) !void {
    const d = runner.card.embedding_length;
    zigferencer.ops.embedLookup(runner.embed.*, row.token, d, runner.scratch.get(.x)[0..d]);
    var layer: u32 = 0;
    while (layer < k) : (layer += 1) {
        runner.layerForward(layer, &.{row});
    }
}

/// Row 0 of the residual stream, as bytes, so the comparison is bitwise.
fn residual(runner: *const QwenCpu) []const u8 {
    return std.mem.sliceAsBytes(runner.scratch.get(.x)[0..runner.card.embedding_length]);
}

/// The GDN shape of the card, derived here, so the slab sizes have a check outside the runner.
fn gdnDims(card: loader.ModelCard) zigferencer.gdn.Dims {
    const g = card.gdn.?;
    return .{
        .n_k_heads = g.group_count,
        .n_v_heads = g.time_step_rank,
        .head_dim = g.state_size,
        .conv_kernel = g.conv_kernel,
    };
}

// -- tests ------------------------------------------------------------------

test "qwen initPrefix reproduces the whole-model layers, KV, and GDN state" {
    const io = std.testing.io;
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, serial);
    defer cpu_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    const n_layers = loaded.card.n_layers;

    for (splits) |k| {
        std.debug.assert(k < n_layers); // a prefix test needs a layer left over

        // Each split gets a fresh pair, because the GDN state is a running accumulator.
        var full = try QwenCpu.init(arena, &loaded, .{
            .plan = zigferencer.kv_budget.pinned(1, window),
            .max_rows = 0,
            .pool = &cpu_pool,
            .policy = serial,
            .kv_dtype = .f32,
        });
        var prefix = try prefixRunner(arena, &cpu_pool, &loaded, k, 1, .f32);
        try std.testing.expectEqual(@as(usize, n_layers), full.layers.len);
        try std.testing.expectEqual(@as(usize, k), prefix.layers.len);

        for (tokens, 0..) |token, pos| {
            const row: StepRow = .{
                .slot = 0,
                .token = token,
                .pos = @intCast(pos),
                .want_logits = false,
            };
            try driveRow(&full, k, row);
            try driveRow(&prefix, k, row);
            try std.testing.expectEqualSlices(u8, residual(&full), residual(&prefix));
        }

        // The GDN slabs, layer by layer. A full-attention layer holds empty
        // slices on both sides, so the comparison spans every layer kind.
        for (0..k) |li| {
            try std.testing.expectEqualSlices(f32, full.gdn_state[li], prefix.gdn_state[li]);
            try std.testing.expectEqualSlices(
                f32,
                full.gdn_conv_state[li],
                prefix.gdn_conv_state[li],
            );
        }

        // The whole snapshot of the prefix is the leading span of the snapshot
        // of the whole model, which is the boundary offset the hybrid needs.
        const n_rows: u32 = tokens.len;
        const prefix_bytes: usize = @intCast(prefix.evictSlotBytesFor(n_rows));
        const full_bytes: usize = @intCast(full.evictSlotBytesFor(n_rows));
        try std.testing.expect(prefix_bytes < full_bytes);
        const prefix_blob = try arena.alloc(u8, prefix_bytes);
        const full_blob = try arena.alloc(u8, full_bytes);
        try prefix.evictSlot(0, n_rows, prefix_blob);
        try full.evictSlot(0, n_rows, full_blob);
        try std.testing.expectEqualSlices(u8, full_blob[0..prefix_bytes], prefix_blob);
        try std.testing.expect(!std.mem.allEqual(u8, prefix_blob, 0));
    }
}

test "qwen prefix shapes: KV height, GDN slabs, and the compacted kv_index" {
    const io = std.testing.io;
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, serial);
    defer cpu_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    const g = loaded.card.gdn orelse return error.SkipZigTest;
    // The counts below hold for the interval-4 layout. A fixture at another
    // interval needs its own counts, so the assert states the assumption.
    try std.testing.expectEqual(@as(u32, 4), g.full_attention_interval);

    const k: u32 = 6;
    const n_slots: u32 = 2;
    const prefix = try prefixRunner(arena, &cpu_pool, &loaded, k, n_slots, .f32);

    try std.testing.expectEqual(@as(usize, n_slots), prefix.kv_slots.len);
    try std.testing.expectEqual(@as(u32, 1), prefix.kv_slots[0].n_layers);
    try std.testing.expectEqual(@as(usize, n_slots) * k, prefix.gdn_state.len);
    try std.testing.expectEqual(@as(usize, n_slots) * k, prefix.gdn_conv_state.len);

    var n_gdn: usize = 0;
    for (0..k) |li| {
        if (prefix.kv_index[li]) |index| {
            // Layer 3 is the only full-attention layer in the range, and it
            // takes the only index in the cache.
            try std.testing.expectEqual(@as(usize, 3), li);
            try std.testing.expectEqual(@as(u32, 0), index);
            for (0..n_slots) |s| {
                try std.testing.expectEqual(@as(usize, 0), prefix.gdn_state[s * k + li].len);
                try std.testing.expectEqual(
                    @as(usize, 0),
                    prefix.gdn_conv_state[s * k + li].len,
                );
            }
        } else {
            n_gdn += 1;
            for (0..n_slots) |s| {
                try std.testing.expectEqual(
                    gdnDims(prefix.card).stateLen(),
                    prefix.gdn_state[s * k + li].len,
                );
                try std.testing.expectEqual(
                    gdnDims(prefix.card).convStateLen(),
                    prefix.gdn_conv_state[s * k + li].len,
                );
            }
        }
    }

    try std.testing.expectEqual(@as(usize, 5), n_gdn);
}

test "qwen prefix below the attention interval holds no KV and still steps" {
    const io = std.testing.io;
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, serial);
    defer cpu_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    // k = 3 is the boundary value, because a prefix of 3 layers holds zero
    // full-attention layers at the interval-4 layout. The slot vector is empty,
    // so every path that indexes a cache must stay behind a layer-kind branch.
    const k: u32 = 3;
    var prefix = try prefixRunner(arena, &cpu_pool, &loaded, k, 2, .f32);
    try std.testing.expectEqual(@as(usize, 0), prefix.kv_slots.len);
    for (0..k) |li| {
        try std.testing.expectEqual(@as(?u32, null), prefix.kv_index[li]);
    }

    for (tokens, 0..) |token, pos| {
        try driveRow(
            &prefix,
            k,
            .{ .slot = 0, .token = token, .pos = @intCast(pos), .want_logits = false },
        );
    }

    // The blob is pure GDN state, so its size does not move with the position
    // count, and it round-trips.
    const dims = gdnDims(prefix.card);
    const per_gdn: u64 = (dims.stateLen() + dims.convStateLen()) * @sizeOf(f32);
    try std.testing.expectEqual(@as(u64, k) * per_gdn, prefix.evictSlotBytesFor(1));
    try std.testing.expectEqual(@as(u64, k) * per_gdn, prefix.evictSlotBytesFor(tokens.len));
    const first = try arena.alloc(u8, @intCast(prefix.evictSlotBytesFor(1)));
    const second = try arena.alloc(u8, first.len);
    try prefix.evictSlot(0, 1, first);
    try prefix.restoreSlot(0, 1, first);
    try prefix.evictSlot(0, 1, second);
    try std.testing.expectEqualSlices(u8, first, second);
    try std.testing.expect(!std.mem.allEqual(u8, first, 0));

    // A fork copies the GDN slabs alone, because there is no KV to copy.
    try prefix.cloneSlotState(0, 1, 1);
    for (0..k) |li| {
        try std.testing.expectEqualSlices(f32, prefix.gdn_state[li], prefix.gdn_state[k + li]);
        try std.testing.expectEqualSlices(
            f32,
            prefix.gdn_conv_state[li],
            prefix.gdn_conv_state[k + li],
        );
    }
}

test "qwen prefix resetSlotState zeroes its own slot's slabs only" {
    const io = std.testing.io;
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, serial);
    defer cpu_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    const k: u32 = 6;
    const n_slots: u32 = 2;
    var prefix = try prefixRunner(arena, &cpu_pool, &loaded, k, n_slots, .f32);
    for (0..n_slots) |s| {
        try driveRow(
            &prefix,
            k,
            .{ .slot = @intCast(s), .token = tokens[0], .pos = 0, .want_logits = false },
        );
    }

    // Keep the slabs of slot 1 before the scrub, so the comparison after the
    // scrub is bitwise.
    const keep = try arena.alloc([]f32, k);
    for (0..k) |li| {
        keep[li] = try arena.alloc(f32, prefix.gdn_state[k + li].len);
        @memcpy(keep[li], prefix.gdn_state[k + li]);
    }

    var live: usize = 0;
    for (0..k) |li| {
        if (!std.mem.allEqual(f32, prefix.gdn_state[li], 0)) {
            live += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 5), live); // every GDN layer ran

    try prefix.resetSlotState(0);
    for (0..k) |li| {
        try std.testing.expect(std.mem.allEqual(f32, prefix.gdn_state[li], 0));
        try std.testing.expect(std.mem.allEqual(f32, prefix.gdn_conv_state[li], 0));
        try std.testing.expectEqualSlices(f32, keep[li], prefix.gdn_state[k + li]);
    }

    try std.testing.expect(!std.mem.allEqual(f32, prefix.gdn_state[k + 0], 0));
}

test "qwen prefix init reports an allocation failure and leaves nothing behind" {
    const io = std.testing.io;
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, serial);
    defer cpu_pool.deinit();

    const gpa = std.testing.allocator;
    var load_arena = std.heap.ArenaAllocator.init(gpa);
    defer load_arena.deinit();

    const fixture = try zigferencer.test_fixtures.path(load_arena.allocator(), "Qwen3.5-2B-Q8_0");
    var loaded = try loader.load(io, load_arena.allocator(), fixture);
    defer loaded.close();

    // The arena of the runner draws from a failing allocator, so the constructor meets
    // OutOfMemory partway. The arena deinit must leave no byte behind. That is the whole leak duty.
    var fail_index: usize = 0;
    while (fail_index < 3) : (fail_index += 1) {
        var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = fail_index });
        var runner_arena = std.heap.ArenaAllocator.init(failing.allocator());
        defer runner_arena.deinit();

        try std.testing.expectError(error.OutOfMemory, prefixRunner(
            runner_arena.allocator(),
            &cpu_pool,
            &loaded,
            6,
            1,
            .f32,
        ));
    }
}

test "qwen prefix init refuses a non-qwen model before it sizes anything" {
    const io = std.testing.io;
    var cpu_pool: test_policy.Pool = undefined;
    try test_policy.startPool(&cpu_pool, io, serial);
    defer cpu_pool.deinit();

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    // The arch gate stays ahead of the range asserts, so a llama card reaches a
    // typed error and never the layer walk.
    try std.testing.expectError(
        error.UnsupportedArchitecture,
        prefixRunner(arena, &cpu_pool, &loaded, 1, 1, .f32),
    );
}
