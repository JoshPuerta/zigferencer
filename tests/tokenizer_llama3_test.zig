//! Golden-id gate for the llama3 (`llama-bpe`) pre-tokenizer against the real
//! Meta-Llama-3.1-8B-Instruct-Q8_0 GGUF.
//!
//! The expected ids come from the llama.cpp `llama-tokenize --ids` oracle, so
//! the ids must match it exactly. The gate pins the regex pre-split and
//! `ignore_merges` to that reference. The digit run is the case of interest,
//! because the pre-split turns "2147" into "214" and "7". The test finds the
//! fixture through `test_fixtures.path`.

const std = @import("std");
const zigferencer = @import("zigferencer");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;

// -- fixtures ---------------------------------------------------------------

const Case = struct { text: []const u8, ids: []const u32 };

// The ids come from `llama-tokenize --ids` and carry BOS 128000. The digit run
// is the case the pre-split fixes: 2147 becomes 11584 ("214") plus 22 ("7").
const golden = [_]Case{
    .{ .text = "The year is 2147.", .ids = &.{ 128000, 791, 1060, 374, 220, 11584, 22, 13 } },
    .{ .text = "I don't think it's ready", .ids = &.{ 128000, 40, 1541, 956, 1781, 433, 596, 5644 } },
    .{ .text = "func(x, y) { return x+y; }", .ids = &.{ 128000, 2900, 2120, 11, 379, 8, 314, 471, 865, 44110, 26, 335 } },
    .{ .text = "12345 67", .ids = &.{ 128000, 4513, 1774, 220, 3080 } },
};

// -- tests ------------------------------------------------------------------

test "llama3 pre-tokenizer matches the llama.cpp oracle (golden ids)" {
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const path = try zigferencer.test_fixtures.path(arena, "Meta-Llama-3.1-8B-Instruct-Q8_0");

    // The gate reads metadata and vocab only, so it does not prefault the
    // 8 GB of weights.
    var loaded = try loader.loadWith(io, arena, path, .{ .prefault = false });
    defer loaded.close();
    var vocab = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    try std.testing.expectEqual(vocab_mod.PreType.llama3, vocab.pre);

    for (golden) |c| {
        const ids = try bpe.encodeAlloc(&vocab, gpa, c.text, true);
        defer gpa.free(ids);
        try std.testing.expectEqualSlices(u32, c.ids, ids);
    }
}
