//! Tokenizer gate for the byte-level BPE path against the real
//! SmolLM2-135M-Instruct-Q8_0 GGUF.
//!
//! Two invariants hold without an oracle, and both must hold exactly.
//! `decode(encode(s))` returns `s` for short ASCII inputs, and the SmolLM2
//! `<|im_start|>` special token encodes as one id. A third test pins the
//! pre-split to the llama.cpp `llama-tokenize --ids` reference. The tests find the
//! file through `test_fixtures.path`.

const std = @import("std");
const zigferencer = @import("zigferencer");
const loader = zigferencer.loader;
const parser_mod = zigferencer.parser;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;

// -- tests ------------------------------------------------------------------

test "BPE round-trip on SmolLM2 vocab" {
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const test_path = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, test_path);
    defer loaded.close();

    var vocab = try vocab_mod.buildFromParsed(arena, loaded.parsed);

    const cases = [_][]const u8{
        "Hello",
        "Hello, world",
        "The quick brown fox jumps over the lazy dog.",
        "a b c d e f g",
        "<|im_start|>user\nHi<|im_end|>",
    };

    for (cases) |s| {
        const ids = try bpe.encodeAlloc(&vocab, gpa, s, true);
        defer gpa.free(ids);

        try std.testing.expect(ids.len > 0);
        for (ids) |id| {
            try std.testing.expect(id < vocab.tokens.len);
        }

        const back = try bpe.decodeAlloc(&vocab, gpa, ids);
        defer gpa.free(back);

        try std.testing.expectEqualStrings(s, back);
    }
}

test "smollm pre-tokenizer matches the llama.cpp oracle (golden ids)" {
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const test_path = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.loadWith(io, arena, test_path, .{ .prefault = false });
    defer loaded.close();

    var vocab = try vocab_mod.buildFromParsed(arena, loaded.parsed);
    try std.testing.expectEqual(vocab_mod.PreType.smollm, vocab.pre);

    // The ids come from `llama-tokenize --ids`. The GPT-2 family splits a
    // lowercase contraction only, keeps single digits, and groups a
    // multi-space run with \s+(?!\S). Whole-span BPE gets that grouping wrong.
    const Case = struct { text: []const u8, ids: []const u32 };
    const golden = [_]Case{
        .{ .text = "  spaced   out  ", .ids = &.{ 216, 23861, 256, 578, 256 } },
        .{ .text = "DON'T don't", .ids = &.{ 52, 2154, 23, 68, 1326, 982 } },
        .{ .text = "The year is 2147.", .ids = &.{ 504, 713, 314, 216, 34, 33, 36, 39, 30 } },
    };
    for (golden) |c| {
        const ids = try bpe.encodeAlloc(&vocab, gpa, c.text, true);
        defer gpa.free(ids);

        try std.testing.expectEqualSlices(u32, c.ids, ids);
    }
}

test "<|im_start|> tokenizes as one special id" {
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const test_path = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, test_path);
    defer loaded.close();

    var vocab = try vocab_mod.buildFromParsed(arena, loaded.parsed);

    const ids = try bpe.encodeAlloc(&vocab, gpa, "<|im_start|>", true);
    defer gpa.free(ids);

    try std.testing.expectEqual(@as(usize, 1), ids.len);
    // `specials_sorted` is the one special-token form the encoder reads, so the
    // assertion goes through it.
    var found = false;
    for (vocab.specials_sorted) |s| {
        if (s.id == ids[0]) {
            found = true;
        }
    }

    try std.testing.expect(found);
}
