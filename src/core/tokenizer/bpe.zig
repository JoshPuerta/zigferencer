//! Byte-level BPE encode and decode over a `Vocab`. It is a pure datapath: no
//! Vocab mutation, no I/O, and a `gpa` taken only for the output slice.
//! `Vocab` (vocab.zig) holds the state. The split keeps encode and decode small
//! and testable on their own.
//!
//! Encode: greedy-match the longest special token at the cursor. On a match,
//! flush the pending segment through BPE merge and emit the special id.
//! Otherwise consume one byte into `pending` through `vocab.byte_to_id[b]`.
//! Flush at the end. The flush repeatedly merges the lowest-rank adjacent pair
//! until none remain. Decode concatenates each token's bytes, walking codepoint
//! by codepoint to undo the GPT-2 byte-level mapping.

const std = @import("std");
const Allocator = std.mem.Allocator;
const span = @import("../span.zig");
const v = @import("vocab.zig");
const Vocab = v.Vocab;

// -- types ------------------------------------------------------------------

pub const EncodeError = error{UntokenizableByte} || Allocator.Error;

const Match = struct { id: u32, len: usize };

const CpClass = struct { letter: bool, number: bool, whitespace: bool, cr_or_lf: bool };

const Cp = struct { cp: u21, len: usize };

// -- api --------------------------------------------------------------------

/// Encode `text` to token ids (specials matched greedily, the rest through BPE
/// merges). BOS is prepended when the model's tokenizer metadata says so
/// (`vocab.add_bos`) AND `sequence_start` is true, so callers pass text and
/// never prepend it themselves. Text appended mid-conversation (a follow-up
/// turn) passes false, so BOS appears only at position 0.
pub fn encodeAlloc(vocab: *const Vocab, gpa: Allocator, text: []const u8, sequence_start: bool) EncodeError![]u32 {
    const sp = span.begin(.tokenize);
    defer sp.end();
    std.debug.assert(vocab.tokens.len > 0);
    // Non-default tokenizers (llama3 / qwen / smollm) split the text on a
    // GPT-2-style regex before BPE. The default path BPE-merges the whole
    // inter-special span. Branch first so the default path stays byte-identical.
    if (vocab.pre != .default) {
        const pre_ids = try encodePreSplit(vocab, gpa, text, sequence_start);
        if (std.debug.runtime_safety) for (pre_ids) |id| std.debug.assert(id < vocab.tokens.len);
        return pre_ids;
    }

    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    if (sequence_start and vocab.add_bos) try ids.append(gpa, vocab.bos_id);

    var pending: std.ArrayList(u32) = .empty;
    defer pending.deinit(gpa);

    var pos: usize = 0;
    while (pos < text.len) {
        const matched = matchSpecial(vocab, text, pos);
        if (matched) |m| {
            try flushSegment(vocab, gpa, &pending, &ids);
            try ids.append(gpa, m.id);
            pos += m.len;
            continue;
        }
        const id = vocab.byte_to_id[text[pos]];
        if (id == v.no_token) return error.UntokenizableByte;
        try pending.append(gpa, id);
        pos += 1;
    }
    try flushSegment(vocab, gpa, &pending, &ids);
    const out = try ids.toOwnedSlice(gpa);
    if (std.debug.runtime_safety) for (out) |id| std.debug.assert(id < vocab.tokens.len);
    return out;
}

/// Decode token ids back to UTF-8 text (inverse of the GPT-2 byte mapping).
pub fn decodeAlloc(vocab: *const Vocab, gpa: Allocator, ids: []const u32) Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);

    for (ids) |id| {
        if (id >= vocab.tokens.len) continue;
        const token_bytes = vocab.tokens[id];
        var i: usize = 0;
        while (i < token_bytes.len) {
            const seq_len: usize = std.unicode.utf8ByteSequenceLength(token_bytes[i]) catch {
                try buf.append(gpa, token_bytes[i]);
                i += 1;
                continue;
            };
            if (i + seq_len > token_bytes.len) {
                try buf.append(gpa, token_bytes[i]);
                i += 1;
                continue;
            }
            const cp = std.unicode.utf8Decode(token_bytes[i..][0..seq_len]) catch {
                try buf.append(gpa, token_bytes[i]);
                i += 1;
                continue;
            };
            if (vocab.codepoint_to_byte.get(cp)) |byte| {
                try buf.append(gpa, byte);
            } else {
                try buf.appendSlice(gpa, token_bytes[i..][0..seq_len]);
            }
            i += seq_len;
        }
    }
    return buf.toOwnedSlice(gpa);
}

// -- helpers ----------------------------------------------------------------

fn matchSpecial(vocab: *const Vocab, text: []const u8, pos: usize) ?Match {
    for (vocab.specials_sorted) |sp| {
        if (sp.bytes.len == 0) continue;
        if (pos + sp.bytes.len > text.len) continue;
        if (std.mem.eql(u8, text[pos..][0..sp.bytes.len], sp.bytes)) {
            return .{ .id = sp.id, .len = sp.bytes.len };
        }
    }
    return null;
}

fn flushSegment(
    vocab: *const Vocab,
    gpa: Allocator,
    pending: *std.ArrayList(u32),
    out: *std.ArrayList(u32),
) Allocator.Error!void {
    if (pending.items.len == 0) return;

    while (pending.items.len >= 2) {
        var best_rank: u32 = std.math.maxInt(u32);
        var best_idx: ?usize = null;
        var i: usize = 0;
        while (i + 1 < pending.items.len) : (i += 1) {
            const key = v.pairKey(pending.items[i], pending.items[i + 1]);
            if (vocab.merge_rank.get(key)) |rank| {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                }
            }
        }
        if (best_idx == null) break;
        const idx = best_idx.?;
        const merged = vocab.merge_to_id.get(v.pairKey(pending.items[idx], pending.items[idx + 1])).?;
        pending.items[idx] = merged;
        // Shift left to drop the consumed right-hand element.
        std.mem.copyForwards(u32, pending.items[idx + 1 .. pending.items.len - 1], pending.items[idx + 2 ..]);
        pending.shrinkRetainingCapacity(pending.items.len - 1);
    }

    try out.appendSlice(gpa, pending.items);
    pending.clearRetainingCapacity();
}

// -- pre-split path: regex pre-tokenize, then BPE per pre-token -------------
//
// llama.cpp pre-tokenizes on a GPT-2-family regex before BPE. Merges therefore
// never cross a pre-token boundary. For example, llama3 splits "2147" to
// ["214","7"] (digit runs of 1-3) instead of merging across. `preTokenLen`
// dispatches the per-family regex (`preSplitLlama` for llama3/qwen,
// `preSplitGpt2` for smollm).
// `ignore_merges` (llama3 only) emits a whole pre-token directly when it is a
// vocab token.
//
// Scope: ASCII-exact classification of \p{L}/\p{N}/\s. Non-ASCII codepoints are
// treated as letters. That covers Latin/CJK words and qwen's `\p{M}` marks.
// Non-ASCII digits and punctuation are a documented gap, because full Unicode
// tables are absent.

fn encodePreSplit(vocab: *const Vocab, gpa: Allocator, text: []const u8, sequence_start: bool) EncodeError![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    if (sequence_start and vocab.add_bos) try ids.append(gpa, vocab.bos_id);

    var pending: std.ArrayList(u32) = .empty;
    defer pending.deinit(gpa);
    var mapped: std.ArrayList(u8) = .empty; // byte-level pre-token, for ignore_merges
    defer mapped.deinit(gpa);

    var pos: usize = 0;
    while (pos < text.len) {
        // Special tokens are extracted before the regex (llama.cpp does the same).
        // `capAtSpecial` keeps a pre-token from swallowing a special's start.
        if (matchSpecial(vocab, text, pos)) |m| {
            try ids.append(gpa, m.id);
            pos += m.len;
            continue;
        }
        const seg = capAtSpecial(vocab, text, pos, preTokenLen(vocab, text, pos));
        const bytes = text[pos..][0..seg];

        if (vocab.ignore_merges) {
            mapped.clearRetainingCapacity();
            for (bytes) |b| try mapped.appendSlice(gpa, v.byteLevel(b));
            if (vocab.token_to_id.get(mapped.items)) |id| {
                try ids.append(gpa, id);
                pos += seg;
                continue;
            }
        }
        for (bytes) |b| {
            const id = vocab.byte_to_id[b];
            if (id == v.no_token) return error.UntokenizableByte;
            try pending.append(gpa, id);
        }
        try flushSegment(vocab, gpa, &pending, &ids); // merges within this pre-token, then clears
        pos += seg;
    }
    return ids.toOwnedSlice(gpa);
}

/// First special token starting strictly inside `(from, from+seg)` caps the
/// pre-token there, so the next loop turn matches it at its start. Returns the
/// (possibly shortened) length.
fn capAtSpecial(vocab: *const Vocab, text: []const u8, from: usize, seg: usize) usize {
    var q = from + 1;
    const end = from + seg;
    while (q < end) : (q += 1) {
        if (matchSpecial(vocab, text, q) != null) return q - from;
    }
    return seg;
}

/// ASCII-exact. Any non-ASCII codepoint counts as a letter (see the scope
/// note above).
fn classify(cp: u21) CpClass {
    if (cp < 128) {
        const c: u8 = @intCast(cp);
        return .{
            .letter = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'),
            .number = c >= '0' and c <= '9',
            .whitespace = c == ' ' or (c >= 0x09 and c <= 0x0d), // space, \t \n \v \f \r
            .cr_or_lf = c == '\r' or c == '\n',
        };
    }
    return .{ .letter = true, .number = false, .whitespace = false, .cr_or_lf = false };
}

fn nextCp(text: []const u8, i: usize) Cp {
    const n = std.unicode.utf8ByteSequenceLength(text[i]) catch return .{ .cp = text[i], .len = 1 };
    if (i + n > text.len) return .{ .cp = text[i], .len = 1 };
    const cp = std.unicode.utf8Decode(text[i..][0..n]) catch return .{ .cp = text[i], .len = 1 };
    return .{ .cp = cp, .len = n };
}

fn lowerAscii(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}

/// Byte length of the next pre-token for `vocab.pre` at `from`. Dispatches the
/// per-family regex. `.default` never reaches here (gated in `encodeAlloc`).
fn preTokenLen(vocab: *const Vocab, text: []const u8, from: usize) usize {
    return switch (vocab.pre) {
        .llama3 => preSplitLlama(text, from, 3),
        .qwen => preSplitLlama(text, from, 1),
        .smollm => preSplitGpt2(text, from),
        .default => unreachable,
    };
}

/// Byte length of the next llama3/qwen pre-token at `from` (`from < text.len`).
/// Always >= 1. The regex alternatives are tried in order, and the first that
/// matches wins. `digit_max` is the `\p{N}{1,N}` run cap (3 for llama3, 1 for
/// qwen).
fn preSplitLlama(text: []const u8, from: usize, digit_max: usize) usize {
    const len = text.len;
    const first = nextCp(text, from);
    const fc = classify(first.cp);

    // 1. contractions: '(s|t|m|d) -> 2, '(re|ve|ll) -> 3  (ASCII, case-insensitive)
    if (text[from] == '\'') {
        if (from + 1 < len) {
            const a = lowerAscii(text[from + 1]);
            if (a == 's' or a == 't' or a == 'm' or a == 'd') return 2;
            if (from + 2 < len) {
                const b = lowerAscii(text[from + 2]);
                if ((a == 'r' and b == 'e') or (a == 'v' and b == 'e') or (a == 'l' and b == 'l')) return 3;
            }
        }
    }

    // 2. [^\r\n\p{L}\p{N}]? \p{L}+
    if (fc.letter) {
        return lettersRun(text, from + first.len) - from;
    }
    if (!fc.cr_or_lf and !fc.number) { // first is a valid optional leading char
        const after = from + first.len;
        if (after < len and classify(nextCp(text, after).cp).letter) {
            return lettersRun(text, after) - from;
        }
    }

    // 3. \p{N}{1,digit_max}
    if (fc.number) {
        var cnt: usize = 1;
        var j = from + first.len;
        while (cnt < digit_max and j < len) {
            const nc = nextCp(text, j);
            if (!classify(nc.cp).number) break;
            j += nc.len;
            cnt += 1;
        }
        return j - from;
    }

    // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*   (optional literal space, then symbols, then CR/LF)
    {
        var k = from;
        if (first.cp == ' ' and from + 1 < len and isSymbol(nextCp(text, from + 1).cp)) {
            k = from + 1; // the optional leading space, with symbols after it
        }
        if (isSymbol(nextCp(text, k).cp)) {
            var j = k;
            while (j < len and isSymbol(nextCp(text, j).cp)) j += nextCp(text, j).len;
            while (j < len and classify(nextCp(text, j).cp).cr_or_lf) j += nextCp(text, j).len;
            return j - from;
        }
    }

    // 5/6/7. whitespace tail (reached only when `first` is whitespace).
    var j = from;
    var last_nl_end: usize = 0; // byte after the last CR/LF in the run (0 = none)
    var last_ws_start: usize = from;
    while (j < len) {
        const nc = nextCp(text, j);
        const ncl = classify(nc.cp);
        if (!ncl.whitespace) break;
        last_ws_start = j;
        j += nc.len;
        if (ncl.cr_or_lf) last_nl_end = j;
    }
    if (j == from) return first.len; // safety: nothing classified (unreachable)
    if (last_nl_end > from) return last_nl_end - from; // 5. \s*[\r\n]+
    if (last_ws_start > from and j < len) return last_ws_start - from; // 6. \s+(?!\S)
    return j - from; // 7. \s+
}

/// Byte length of the next smollm (GPT-2) pre-token at `from`. Always >= 1. The
/// llama.cpp `smollm` regex is `{ "\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|
///  ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)" }`. The leading `\p{N}` pass isolates
/// every digit, so digits are single pre-tokens and a digit also acts as a
/// boundary for the trailing-whitespace lookahead. It differs from llama3/qwen:
/// lowercase contractions, and the optional leading char is a literal space
/// only (not any non-L/N). Symbols carry no trailing `[\r\n]*`, and there is no
/// `\s*[\r\n]+` or final `\s+` (an unmatched char falls back to one codepoint).
fn preSplitGpt2(text: []const u8, from: usize) usize {
    const len = text.len;
    const first = nextCp(text, from);
    const fc = classify(first.cp);

    // 1. contractions, LOWERCASE only: 's 't 'm 'd -> 2, 're 've 'll -> 3
    if (text[from] == '\'' and from + 1 < len) {
        const a = text[from + 1];
        if (a == 's' or a == 't' or a == 'm' or a == 'd') return 2;
        if (from + 2 < len) {
            const b = text[from + 2];
            if ((a == 'r' and b == 'e') or (a == 'v' and b == 'e') or (a == 'l' and b == 'l')) return 3;
        }
    }

    // 2.  ?\p{L}+  (optional ONE literal space, then letters)
    if (fc.letter) return lettersRun(text, from + first.len) - from;
    if (first.cp == ' ' and from + 1 < len and classify(nextCp(text, from + 1).cp).letter) {
        return lettersRun(text, from + 1) - from;
    }

    // 3. single digit (the leading \p{N} pass makes every digit its own token)
    if (fc.number) return first.len;

    // 4.  ?[^\s\p{L}\p{N}]+  (optional ONE literal space, then symbols, no [\r\n]*)
    {
        var k = from;
        if (first.cp == ' ' and from + 1 < len and isSymbol(nextCp(text, from + 1).cp)) k = from + 1;
        if (isSymbol(nextCp(text, k).cp)) {
            var j = k;
            while (j < len and isSymbol(nextCp(text, j).cp)) j += nextCp(text, j).len;
            return j - from;
        }
    }

    // 5. \s+(?!\S): the lookahead also holds before a digit (the \p{N} pass
    //    splits there) and at EOF. A lone non-space whitespace before \S that
    //    nothing else matched falls back to one codepoint.
    var j = from;
    var last_ws_start: usize = from;
    while (j < len) {
        const nc = nextCp(text, j);
        if (!classify(nc.cp).whitespace) break;
        last_ws_start = j;
        j += nc.len;
    }
    if (j == from) return first.len; // unmatched (e.g. a control char) -> 1 codepoint
    const boundary = j >= len or classify(nextCp(text, j).cp).number; // EOF or digit
    if (boundary) return j - from; // whole run
    if (last_ws_start > from) return last_ws_start - from; // give back the last ws before \S
    return j - from; // a lone whitespace before \S (its own token)
}

/// Byte index past the maximal run of letters starting at `at`.
fn lettersRun(text: []const u8, at: usize) usize {
    var j = at;
    while (j < text.len) {
        const nc = nextCp(text, j);
        if (!classify(nc.cp).letter) break;
        j += nc.len;
    }
    return j;
}

/// `[^\s\p{L}\p{N}]`: not whitespace, letter, or number.
fn isSymbol(cp: u21) bool {
    const c = classify(cp);
    return !c.whitespace and !c.letter and !c.number;
}

// -- tests ------------------------------------------------------------------

const parser = @import("../../io/gguf/parser.zig");
const prop = @import("../prop.zig");

/// The merges of the synthetic vocab, in rank order, as byte-level strings.
/// They cover letters, a leading space (U+0120), digits, the two bytes of
/// U+00E9, and a newline pair (U+010A).
const test_merges = [_][]const u8{ "a b", "ab ab", "\u{120} a", "\u{120}a b", "1 2", "\u{C3} \u{A9}", "\u{10A} \u{10A}" };
const test_specials = [_][]const u8{ "<|s|>", "<think>" };

/// A vocab of the 256 byte-level characters, one token per merge, and two
/// specials, built through the GGUF key path with no model file.
fn testVocab(arena: Allocator, pre: []const u8) !Vocab {
    var tokens: std.ArrayList([]const u8) = .empty;
    var types: std.ArrayList(i32) = .empty;
    // The GPT-2 map: a printable byte is its own codepoint, and the other
    // bytes take the codepoints from U+0100 up, in byte order.
    var escape: u21 = 0x100;
    for (0..256) |b| {
        const printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174;
        const cp: u21 = if (printable) @intCast(b) else escape;
        if (!printable) escape += 1;
        var utf8: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(cp, &utf8);
        try tokens.append(arena, try arena.dupe(u8, utf8[0..len]));
        try types.append(arena, 1);
    }
    for (test_merges) |m| {
        const space = std.mem.indexOfScalar(u8, m, ' ').?;
        try tokens.append(arena, try std.mem.concat(arena, u8, &.{ m[0..space], m[space + 1 ..] }));
        try types.append(arena, 1);
    }
    for (test_specials, [_]i32{ 3, 4 }) |s, t| {
        try tokens.append(arena, s);
        try types.append(arena, t);
    }
    const kvs = try arena.dupe(parser.Kv, &.{
        .{ .key = "tokenizer.ggml.tokens", .value = .{ .array = .{ .strings = tokens.items } } },
        .{ .key = "tokenizer.ggml.merges", .value = .{ .array = .{ .strings = &test_merges } } },
        .{ .key = "tokenizer.ggml.token_type", .value = .{ .array = .{ .i32s = types.items } } },
        .{ .key = "tokenizer.ggml.eos_token_id", .value = .{ .u32_val = 256 } },
        .{ .key = "tokenizer.ggml.pre", .value = .{ .string = pre } },
    });
    return v.buildFromParsed(arena, .{ .header = undefined, .kvs = kvs, .tensors = &.{}, .data_offset = 0, .alignment = 1 });
}

const TestVocabs = [4]Vocab;

/// Text pieces that hit the merges, the specials, the pre-split classes, and
/// invalid UTF-8: a lone lead byte, a cut 3-byte sequence, and 0xFF.
const test_pieces = [_][]const u8{ "a", "b", "ab", " ", " a", "1", "12", "\n", "\n\n", "\xc3\xa9", "<|s|>", "<think>", "<|s", "'s", "\t", "!", "\xc3", "\xe2\x82", "\xff", "\xf0\x9f\x98\x80" };

/// Fills `text` with pieces and raw bytes, valid and invalid UTF-8, and
/// returns the length.
fn drawText(r: std.Random, text: *[256]u8) usize {
    var n: usize = 0;
    while (n < 200) {
        if (r.uintLessThan(u8, 4) == 0) break;
        if (r.boolean()) {
            const piece = test_pieces[r.uintLessThan(usize, test_pieces.len)];
            @memcpy(text[n..][0..piece.len], piece);
            n += piece.len;
        } else {
            text[n] = r.int(u8);
            n += 1;
        }
    }
    return n;
}

/// Over random byte strings, valid and invalid UTF-8, under every pre-split
/// family: decode(encode(s)) == s.
fn checkBpeRoundTrip(vocabs: *const TestVocabs, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    var text: [256]u8 = undefined;
    const n = drawText(prng.random(), &text);
    const gpa = std.testing.allocator;
    for (vocabs) |*vocab| {
        const ids = try encodeAlloc(vocab, gpa, text[0..n], false);
        defer gpa.free(ids);
        const back = try decodeAlloc(vocab, gpa, ids);
        defer gpa.free(back);
        try std.testing.expectEqualSlices(u8, text[0..n], back);
    }
}

test "property: decode(encode(s)) == s over random byte strings for every pre-split family" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var vocabs: TestVocabs = undefined;
    for (&vocabs, [_][]const u8{ "", "llama-bpe", "qwen2", "smollm" }) |*vocab, pre| vocab.* = try testVocab(arena, pre);
    for (vocabs, [_]v.PreType{ .default, .llama3, .qwen, .smollm }) |vocab, pre| {
        try std.testing.expectEqual(pre, vocab.pre);
        try std.testing.expectEqual(@as(usize, 2), vocab.specials_sorted.len);
        for (vocab.byte_to_id) |id| try std.testing.expect(id != v.no_token);
    }
    try prop.forAll(.{ .site = 0xB9E0_0001, .cases = 600 }, @as(*const TestVocabs, &vocabs), u64, prop.genSeed(*const TestVocabs), checkBpeRoundTrip);
}

// The naive model. It holds the merge list as literal strings and applies
// the lowest-rank merge to a list of strings, with no id and no hash map.

const ByteLevel = struct { bytes: [4]u8, len: u3 };

/// The byte-level spelling of every byte: a printable byte is its own
/// codepoint and the others take U+0100 up in byte order.
fn byteLevelTable() [256]ByteLevel {
    var table: [256]ByteLevel = undefined;
    var escape: u21 = 0x100;
    for (&table, 0..) |*entry, b| {
        const printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174;
        const cp: u21 = if (printable) @intCast(b) else escape;
        if (!printable) escape += 1;
        entry.len = @intCast(std.unicode.utf8Encode(cp, &entry.bytes) catch unreachable);
    }
    return table;
}

/// The rank of the merge line `left ++ " " ++ right`, or null. A byte-level
/// spelling holds no ASCII space, so the line splits at its only space.
fn naiveRank(left: []const u8, right: []const u8) ?usize {
    for (test_merges, 0..) |line, rank| {
        const space = std.mem.indexOfScalar(u8, line, ' ').?;
        if (std.mem.eql(u8, line[0..space], left) and std.mem.eql(u8, line[space + 1 ..], right)) return rank;
    }
    return null;
}

/// The id of a final symbol. Byte tokens come first, then one token per
/// merge line in order.
fn naiveId(table: *const [256]ByteLevel, symbol: []const u8) u32 {
    for (test_merges, 0..) |line, rank| {
        const space = std.mem.indexOfScalar(u8, line, ' ').?;
        const left = line[0..space];
        const right = line[space + 1 ..];
        if (symbol.len == left.len + right.len and
            std.mem.startsWith(u8, symbol, left) and std.mem.endsWith(u8, symbol, right)) return @intCast(256 + rank);
    }
    for (table, 0..) |entry, b| {
        if (std.mem.eql(u8, symbol, entry.bytes[0..entry.len])) return @intCast(b);
    }
    unreachable;
}

/// The ids of `text` under no pre-split: a special is the leftmost literal
/// match, and the bytes between two specials merge by lowest rank first,
/// leftmost on a tie.
fn naiveEncode(arena: Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
    const table = byteLevelTable();
    var symbols: std.ArrayList([]const u8) = .empty;
    var pos: usize = 0;
    while (true) {
        const special = for (test_specials, 0..) |sp, i| {
            if (std.mem.startsWith(u8, text[pos..], sp)) break i;
        } else null;
        if (pos == text.len or special != null) {
            while (true) {
                var best: ?usize = null;
                var best_rank: usize = std.math.maxInt(usize);
                for (0..symbols.items.len -| 1) |i| {
                    const rank = naiveRank(symbols.items[i], symbols.items[i + 1]) orelse continue;
                    if (rank < best_rank) {
                        best_rank = rank;
                        best = i;
                    }
                }
                const i = best orelse break;
                symbols.items[i] = try std.mem.concat(arena, u8, &.{ symbols.items[i], symbols.items[i + 1] });
                _ = symbols.orderedRemove(i + 1);
            }
            for (symbols.items) |symbol| try out.append(arena, naiveId(&table, symbol));
            symbols.clearRetainingCapacity();
            if (pos == text.len) return;
            try out.append(arena, @intCast(256 + test_merges.len + special.?));
            pos += test_specials[special.?].len;
            continue;
        }
        const entry = &table[text[pos]];
        try symbols.append(arena, entry.bytes[0..entry.len]);
        pos += 1;
    }
}

/// The default family encodes to the ids of the naive lowest-rank-first
/// merge over the same ranks.
fn checkBpeAgainstNaive(vocab: *const Vocab, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    var text: [256]u8 = undefined;
    const n = drawText(prng.random(), &text);
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    var want: std.ArrayList(u32) = .empty;
    try naiveEncode(arena_state.allocator(), text[0..n], &want);
    const got = try encodeAlloc(vocab, std.testing.allocator, text[0..n], false);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualSlices(u32, want.items, got);
}

test "property: the default family encodes to the ids of a naive lowest-rank-first merge" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const vocab = try testVocab(arena_state.allocator(), "");
    try prop.forAll(.{ .site = 0xB9E0_0002, .cases = 600 }, @as(*const Vocab, &vocab), u64, prop.genSeed(*const Vocab), checkBpeAgainstNaive);
}

test "each special piece encodes to its single id under every pre-split family" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const gpa = std.testing.allocator;
    for ([_][]const u8{ "", "llama-bpe", "qwen2", "smollm" }) |pre| {
        const vocab = try testVocab(arena, pre);
        for (test_specials, 0..) |special, i| {
            const id: u32 = @intCast(256 + test_merges.len + i);
            const alone = try encodeAlloc(&vocab, gpa, special, false);
            defer gpa.free(alone);
            try std.testing.expectEqualSlices(u32, &.{id}, alone);
            // Text on both sides must not swallow the special.
            const text = try std.mem.concat(arena, u8, &.{ "a", special, "b" });
            const between = try encodeAlloc(&vocab, gpa, text, false);
            defer gpa.free(between);
            try std.testing.expectEqualSlices(u32, &.{ 'a', id, 'b' }, between);
        }
    }
}
