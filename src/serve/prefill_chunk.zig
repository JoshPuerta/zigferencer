//! The prefill chunk rule of the serve scheduler: how many prompt rows one forward pass takes from
//! the turn that feeds. The scheduler owns the state (`fed`, the decode set) and passes the
//! numbers in.
//!
//! Rules. `derived` is the default. A pass with no decode row takes a full row width of prompt, so
//! one session prefills in ceil(N / width) passes. A pass with decode rows takes the spare decode
//! rows, so a decoding neighbor keeps its tick cadence. `spare_rows` takes the spare decode rows
//! always. `fixed` takes an explicit number of rows.
//!
//! Scratch. A runner sizes its row scratch with `stepRows`. The decode rows are at most `batch`.
//! The `batch + 1` floor keeps the one-row prefill legal when decode saturates. The slot count is
//! a floor too, so every session fits in one pass.

const std = @import("std");
const limits = @import("../core/limits.zig");
const prop = @import("../core/utils/prop.zig");

// -- constants --------------------------------------------------------------

/// The CPU runners hold `limits.cpu_prefill_chunk` prompt rows per pass.
pub const cpu_rule: Rule = .{ .derived = limits.cpu_prefill_chunk };

// -- types ------------------------------------------------------------------

pub const Rule = union(enum) {
    /// The static row width of the runner: the rows its scratch holds for one prompt chunk.
    derived: u32,
    /// The spare decode rows: `max_batch - n_decode`, with a floor of one.
    spare_rows,
    /// An explicit number of rows, in 1..limits.max_prefill_chunk.
    fixed: u32,
};

// -- api --------------------------------------------------------------------

/// Map the `--prefill-chunk` value to a rule. Unset selects `derived` with the
/// row width of the runner. Zero selects `spare_rows`.
pub fn fromFlag(flag: ?u32, width: u32) Rule {
    const n = flag orelse return .{ .derived = width };
    if (n == 0) return .spare_rows;
    return .{ .fixed = n };
}

/// The prompt rows one pass takes, given the rows that the feeding turn has left (`remaining`).
/// `step_cap` is the most rows the runner takes in one pass. The result is zero only when decode
/// rows fill the pass. The caller then composes the pass without a chunk.
pub fn chunk(
    rule: Rule,
    remaining: usize,
    n_decode: usize,
    max_batch: usize,
    step_cap: usize,
) usize {
    std.debug.assert(remaining >= 1);
    std.debug.assert(n_decode <= max_batch);
    std.debug.assert(n_decode <= step_cap);

    const room = step_cap - n_decode;
    const spare = @max(1, max_batch - n_decode);
    const budget: usize = switch (rule) {
        .spare_rows => spare,
        .fixed => |n| n,
        .derived => |width| blk: {
            std.debug.assert(width >= 1);

            break :blk if (n_decode == 0) width else @min(spare, width);
        },
    };

    const result = @min(remaining, @min(room, budget));
    std.debug.assert(result <= remaining);
    std.debug.assert(result + n_decode <= step_cap);
    if (room >= 1) std.debug.assert(result >= 1);

    return result;
}

/// The rows a runner needs for its row scratch, before the runner clamps the request to its own
/// width.
pub fn stepRows(rule: Rule, slots: u32, batch: u32) u32 {
    const prompt_rows: u32 = switch (rule) {
        .spare_rows => 0,
        .fixed => |n| n,
        .derived => |width| width,
    };

    return @max(slots, @max(batch + 1, batch + prompt_rows));
}

// -- tests ------------------------------------------------------------------

const ChunkCase = struct {
    rule: Rule,
    remaining: usize,
    n_decode: usize,
    max_batch: usize,
    step_cap: usize,
    want: usize,
};

/// A test case. The fields follow the order of `chunk`, then the wanted rows.
fn chunkCase(
    rule: Rule,
    remaining: usize,
    n_decode: usize,
    max_batch: usize,
    step_cap: usize,
    want: usize,
) ChunkCase {
    return .{
        .rule = rule,
        .remaining = remaining,
        .n_decode = n_decode,
        .max_batch = max_batch,
        .step_cap = step_cap,
        .want = want,
    };
}

test "chunk: decode rows narrow the chunk to the spare rows, and a full pass takes none" {
    const derived: Rule = .{ .derived = 64 };
    const cases = [_]ChunkCase{
        chunkCase(derived, 440, 1, 4, 68, 3),
        chunkCase(derived, 440, 4, 4, 68, 1),
        chunkCase(derived, 1, 2, 4, 68, 1),
        chunkCase(.spare_rows, 440, 2, 4, 5, 2),
        chunkCase(.spare_rows, 440, 4, 4, 5, 1),
        // The row cap wins over the budget, and a full pass leaves no room.
        chunkCase(.spare_rows, 9, 2, 2, 2, 0),
        chunkCase(.spare_rows, 9, 1, 2, 2, 1),
        chunkCase(.{ .fixed = 4 }, 5, 3, 3, 6, 3),
        chunkCase(.{ .fixed = 4 }, 5, 6, 6, 6, 0),
    };
    for (cases) |c| {
        try std.testing.expectEqual(
            c.want,
            chunk(c.rule, c.remaining, c.n_decode, c.max_batch, c.step_cap),
        );
    }
}

test "flag maps to a rule" {
    try std.testing.expectEqual(Rule{ .derived = 128 }, fromFlag(null, 128));
    try std.testing.expectEqual(Rule.spare_rows, fromFlag(0, 128));
    try std.testing.expectEqual(Rule{ .fixed = 32 }, fromFlag(32, 128));
}

test "stepRows keeps the original size for spare_rows and holds a full width for derived" {
    try std.testing.expectEqual(@as(u32, 2), stepRows(.spare_rows, 1, 1));
    try std.testing.expectEqual(@as(u32, 9), stepRows(.spare_rows, 9, 4));
    try std.testing.expectEqual(@as(u32, 8), stepRows(.{ .fixed = 4 }, 2, 4));
    try std.testing.expectEqual(@as(u32, 65), stepRows(cpu_rule, 1, 1));
    try std.testing.expectEqual(@as(u32, 68), stepRows(cpu_rule, 3, 4));
}

const TileCase = struct { rule: Rule, batch: u32, slots: u32, n: usize, all_idle: bool, seed: u64 };

fn genTile(_: void, random: std.Random, size: u32) TileCase {
    _ = size;
    const width = random.intRangeAtMost(u32, 1, 64);
    const batch = random.intRangeAtMost(u32, 1, 16);
    return .{
        .rule = switch (random.uintLessThan(u8, 3)) {
            0 => .{ .derived = width },
            1 => .spare_rows,
            else => .{ .fixed = random.intRangeAtMost(u32, 1, 64) },
        },
        .batch = batch,
        .slots = random.intRangeAtMost(u32, batch, batch + 4),
        .n = random.intRangeAtMost(usize, 1, 700),
        .all_idle = random.boolean(),
        .seed = random.int(u64),
    };
}

fn checkTile(_: void, c: TileCase) anyerror!void {
    var prng = std.Random.DefaultPrng.init(c.seed);
    const rng = prng.random();
    const step_cap = stepRows(c.rule, c.slots, c.batch);
    var fed: usize = 0;
    var steps: usize = 0;
    while (fed < c.n) : (steps += 1) {
        // Every pass moves at least one row, so the walk ends within n passes.
        try std.testing.expect(steps < c.n);
        const d: usize = if (c.all_idle) 0 else rng.intRangeAtMost(usize, 0, c.batch);
        const k = chunk(c.rule, c.n - fed, d, c.batch, step_cap);
        try std.testing.expect(k >= 1);
        try std.testing.expect(k <= c.n - fed);
        try std.testing.expect(k + d <= step_cap);
        switch (c.rule) {
            .derived => |w| try std.testing.expect(k <= w),
            .spare_rows => try std.testing.expect(k <= @max(1, c.batch - d)),
            .fixed => |f| try std.testing.expect(k <= f),
        }

        fed += k;
    }

    try std.testing.expectEqual(c.n, fed);
    try std.testing.expect(steps <= c.n);
    if (c.all_idle) {
        switch (c.rule) {
            .derived => |w| try std.testing.expectEqual((c.n + w - 1) / w, steps),
            .spare_rows => try std.testing.expectEqual((c.n + c.batch - 1) / c.batch, steps),
            .fixed => |f| try std.testing.expectEqual((c.n + f - 1) / f, steps),
        }
    }
}

test "property: the chunks of one prompt tile it in order under any decode sequence" {
    try prop.forAll(.{ .site = 0x5eed_0014, .cases = 500 }, {}, TileCase, genTile, checkTile);
}
