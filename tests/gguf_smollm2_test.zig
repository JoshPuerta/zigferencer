//! Load gate for the real SmolLM2-135M-Instruct-Q8_0 GGUF.
//!
//! The test reads about 145 MB of real data through the whole mmap, parser and
//! loader path. Every model-card field must match the file exactly, because a
//! synthetic fixture cannot catch schema drift in the GGUF metadata. The test
//! finds the file through `test_fixtures.path`.

const std = @import("std");
const zigferencer = @import("zigferencer");
const loader = zigferencer.loader;

// -- tests ------------------------------------------------------------------

test "SmolLM2-135M-Instruct-Q8_0 loads and reports expected ModelCard" {
    const io = std.testing.io;

    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const test_path = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

    var loaded = try loader.load(io, arena, test_path);
    defer loaded.close();

    const c = loaded.card;
    try std.testing.expectEqualStrings("llama", c.architecture);
    try std.testing.expectEqual(@as(u32, 30), c.n_layers);
    try std.testing.expectEqual(@as(u32, 9), c.n_heads);
    try std.testing.expectEqual(@as(u32, 3), c.n_kv_heads);
    try std.testing.expectEqual(@as(u32, 64), c.head_dim);
    try std.testing.expectEqual(@as(u32, 576), c.embedding_length);
    try std.testing.expectEqual(@as(u32, 1536), c.feed_forward_length);
    try std.testing.expectEqual(@as(u32, 8192), c.context_length);
    try std.testing.expectEqual(@as(u32, 49152), c.vocab_size);
    try std.testing.expectEqual(@as(u32, 1), c.bos_token_id);
    try std.testing.expectEqual(@as(u32, 2), c.eos_token_id);

    // The table holds 30 layers at 9 weights each plus 2 global tensors.
    try std.testing.expectEqual(@as(usize, 272), loaded.tensors.len);

    // The token embedding stores [embedding_length, vocab_size], not the
    // transpose.
    const tok_embd = blk: {
        for (loaded.tensors) |ref| {
            if (std.mem.eql(u8, ref.name, "token_embd.weight")) break :blk ref;
        }
        return error.MissingTokenEmbedding;
    };
    try std.testing.expectEqual(.q8_0, tok_embd.tensor.dtype);
    try std.testing.expectEqual(@as(u32, 576), tok_embd.tensor.shape[0]);
    try std.testing.expectEqual(@as(u32, 49152), tok_embd.tensor.shape[1]);

    // The loader must keep the card inside the engine bounds.
    try std.testing.expect(c.n_layers <= 64);
    try std.testing.expect(c.vocab_size <= 200_000);
}
