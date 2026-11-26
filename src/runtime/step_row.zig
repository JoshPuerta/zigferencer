//! The batch row every runner steps, and the row rule they all hold. A runner of either
//! family can depend on this file without depending on another runner.
//!
//! Comptime. `chunkedPrefill` reads a runner declaration at comptime, so no runner needs a flag.
//!
//! Row rule. Rows that share a slot are adjacent and ascend by exactly one position. The
//! `want_logits` rows form a suffix of the batch, so their hidden states are contiguous. The
//! lm_head then runs as one batched projection, and prompt rows skip it. A batch row sees exactly
//! the inputs of its solo call.
//!
//! Error split. A bad slot, a broken adjacency, or a broken suffix is a programmer error (assert).
//! A position past the window is runtime data (typed error). A row count past `rows_max` is a
//! typed error in every build mode, because a ReleaseFast overflow writes past the row scratch.

const std = @import("std");
const Allocator = std.mem.Allocator;

// -- types ------------------------------------------------------------------

pub const Error = error{
    MissingTensor,
    UnsupportedWeightDtype,
    PositionOutOfRange,
    TooManyRows,
} || Allocator.Error;

/// One row of a batched forward step. The header states the row rule.
pub const StepRow = struct {
    /// KV slot to read and write.
    slot: u32,
    token: u32,
    /// Position to write at. Attention spans `0..=pos`.
    pos: u32,
    /// Compute the final norm and lm_head for this row. The j-th `want_logits` row of a step lands
    /// at `logits[j * vocab_size ..]`.
    want_logits: bool,
};

/// The logits that one prompt chunk asks for.
pub const ChunkLogits = enum {
    /// The last row of the chunk wants logits: the chunk ends the prompt.
    last_row,
    /// No row wants logits: a later chunk of the same prompt follows.
    none,
};

// -- api --------------------------------------------------------------------

/// Check the row rule that every runner shares, so they cannot drift. Returns the count of
/// `want_logits` rows. The header splits the asserts from the typed errors.
pub fn validateRows(
    rows: []const StepRow,
    slot_count: u32,
    rows_max: u32,
    max_seq: u32,
) error{ PositionOutOfRange, TooManyRows }!u32 {
    std.debug.assert(rows.len >= 1);

    if (rows.len > rows_max) return error.TooManyRows;

    var seen_logits = false;
    var n_logits: u32 = 0;
    // Check each row against its neighbors and count the logits suffix.
    for (rows, 0..) |r, i| {
        std.debug.assert(r.slot < slot_count);

        if (r.pos >= max_seq) return error.PositionOutOfRange;

        if (i > 0 and rows[i - 1].slot == r.slot) {
            // The chunked-prefill shape: consecutive ascending positions.
            std.debug.assert(rows[i - 1].pos + 1 == r.pos);
        } else if (i > 0) {
            // A slot's rows are adjacent. It must not reappear later.
            for (rows[0 .. i - 1]) |prev| {
                std.debug.assert(prev.slot != r.slot);
            }
        }

        if (seen_logits) std.debug.assert(r.want_logits);

        if (r.want_logits) {
            seen_logits = true;
            n_logits += 1;
        }
    }

    return n_logits;
}

/// The rows of a prompt chunk on slot 0: `tokens` at `start_pos` onward. With `.last_row`, the
/// last row wants logits. With `.none`, no row does. `out` must hold `tokens.len` rows.
pub fn chunkRows(
    out: []StepRow,
    tokens: []const u32,
    start_pos: u32,
    logits: ChunkLogits,
) []const StepRow {
    std.debug.assert(tokens.len >= 1);
    std.debug.assert(tokens.len <= out.len);

    // A position past the `u32` range saturates, so `validateRows` refuses it as past the window.
    for (tokens, out[0..tokens.len], 0..) |token, *r, i| {
        r.* = .{
            .slot = 0,
            .token = token,
            .pos = start_pos +| @as(u32, @intCast(i)),
            .want_logits = logits == .last_row and i == tokens.len - 1,
        };
    }

    return out[0..tokens.len];
}

/// Start of the `want_logits` suffix. The lm_head runs for that suffix only, so the prompt rows of
/// a prefill chunk skip the costliest matmul in the model.
pub fn wantLogitsFirst(rows: []const StepRow) u32 {
    var first: u32 = @intCast(rows.len);
    while (first > 0 and rows[first - 1].want_logits) {
        first -= 1;
    }

    return first;
}

/// True when the runner behind `RunnerPtr` declares `chunked_prefill = true`. Such a runner
/// prefills in chunks, `prefillBatchNoLogits` for each chunk but the last and `prefillBatch` for
/// the last. Every other runner keeps the per-token loop.
pub fn chunkedPrefill(comptime RunnerPtr: type) bool {
    const R = @typeInfo(RunnerPtr).pointer.child;
    return @hasDecl(R, "chunked_prefill") and R.chunked_prefill;
}

// -- tests ------------------------------------------------------------------

test "chunkedPrefill: reads the runner declaration, absent means false" {
    const On = struct {
        pub const chunked_prefill: bool = true;
    };
    const Off = struct {
        pub const chunked_prefill: bool = false;
    };
    const Bare = struct {};
    try std.testing.expect(chunkedPrefill(*On));
    try std.testing.expect(chunkedPrefill(*const On));
    try std.testing.expect(!chunkedPrefill(*Off));
    try std.testing.expect(!chunkedPrefill(*Bare));
}

test "validateRows: a row count past the cap is a typed error" {
    const rows = [_]StepRow{
        .{ .slot = 0, .token = 1, .pos = 0, .want_logits = false },
        .{ .slot = 0, .token = 1, .pos = 1, .want_logits = true },
    };
    try std.testing.expectError(error.TooManyRows, validateRows(&rows, 1, 1, 16));
    try std.testing.expectEqual(@as(u32, 1), try validateRows(&rows, 1, 2, 16));
}

test "chunkRows: a chunk past the window is a typed error, at the u32 ceiling too" {
    var buf: [4]StepRow = undefined;
    const tokens = [_]u32{ 7, 8, 9 };
    const rows = chunkRows(&buf, &tokens, 13, .last_row);
    try std.testing.expectEqual(@as(u32, 1), try validateRows(rows, 1, 4, 16));
    try std.testing.expectEqual(@as(u32, 15), rows[2].pos);
    try std.testing.expect(!rows[1].want_logits);
    try std.testing.expect(rows[2].want_logits);

    try std.testing.expectError(error.PositionOutOfRange, validateRows(rows, 1, 4, 15));
    const top = chunkRows(&buf, &tokens, std.math.maxInt(u32) - 1, .last_row);
    try std.testing.expectError(error.PositionOutOfRange, validateRows(top, 1, 4, 16));
}

test "chunkRows: a chunk with no logits holds no want_logits row" {
    var buf: [4]StepRow = undefined;
    const tokens = [_]u32{ 7, 8, 9 };
    const rows = chunkRows(&buf, &tokens, 0, .none);
    try std.testing.expectEqual(@as(u32, 0), try validateRows(rows, 1, 4, 16));
    try std.testing.expectEqual(@as(u32, 3), wantLogitsFirst(rows));
    for (rows, 0..) |r, i| {
        try std.testing.expect(!r.want_logits);
        try std.testing.expectEqual(@as(u32, @intCast(i)), r.pos);
    }
}
