//! Golden-id gate for the Qwen (`qwen35`) pre-tokenizer against the real
//! Qwen3.5-2B-Q8_0 GGUF.
//!
//! The expected ids come from the llama.cpp `llama-tokenize --ids` oracle, so
//! the ids must match it exactly. The Qwen regex is the llama3 regex with
//! single-digit runs and no `ignore_merges`. So "2147" becomes 2, 1, 4 and 7,
//! and a multi-space run keeps its grouping. The added tokens of the vocab
//! (`<think>`, `<tool_call>`) encode to one id each. The tests find the
//! fixture through `test_fixtures.path`.

const std = @import("std");
const zigferencer = @import("zigferencer");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;

// -- fixtures ---------------------------------------------------------------

const Case = struct { text: []const u8, ids: []const u32 };

// The ids come from `llama-tokenize --ids`. Qwen3.5 adds no BOS. "2147" splits
// into single digits (17, 16, 19, 22), and the spaced case keeps its
// \s+(?!\S) grouping.
const golden = [_]Case{
    .{ .text = "The year is 2147.", .ids = &.{ 760, 1007, 369, 220, 17, 16, 19, 22, 13 } },
    .{ .text = "  spaced   out  ", .ids = &.{ 220, 61674, 256, 680, 256 } },
    .{ .text = "I don't think it's ready", .ids = &.{ 40, 1459, 914, 1683, 424, 579, 5354 } },
};

// -- tests ------------------------------------------------------------------

test "qwen pre-tokenizer matches the llama.cpp oracle (golden ids)" {
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const path = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.loadWith(io, arena, path, .{ .prefault = false });
    defer loaded.close();
    var vocab = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    try std.testing.expectEqual(vocab_mod.PreType.qwen, vocab.pre);

    for (golden) |c| {
        const ids = try bpe.encodeAlloc(&vocab, gpa, c.text, true);
        defer gpa.free(ids);
        try std.testing.expectEqualSlices(u32, c.ids, ids);
    }
}

test "qwen added tokens encode to one id each and split the text around them" {
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const path = try zigferencer.test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

    var loaded = try loader.loadWith(io, arena, path, .{ .prefault = false });
    defer loaded.close();
    var vocab = try vocab_mod.buildFromParsed(arena, loaded.parsed);

    const think_id: u32 = 248068;
    const added = [_]struct { text: []const u8, id: u32 }{
        .{ .text = "<think>", .id = think_id },
        .{ .text = "</think>", .id = 248069 },
        .{ .text = "<tool_call>", .id = 248058 },
    };
    for (added) |c| {
        const ids = try bpe.encodeAlloc(&vocab, arena, c.text, true);
        try std.testing.expectEqualSlices(u32, &.{c.id}, ids);
        // The piece of the id is the literal tag text.
        try std.testing.expectEqualStrings(c.text, try bpe.decodeAlloc(&vocab, arena, ids));
    }

    const a = try bpe.encodeAlloc(&vocab, arena, "a", true);
    const b = try bpe.encodeAlloc(&vocab, arena, "b", false);
    const want = try std.mem.concat(arena, u32, &.{ a, &.{think_id}, b });
    try std.testing.expectEqualSlices(u32, want, try bpe.encodeAlloc(&vocab, arena, "a<think>b", true));
}
