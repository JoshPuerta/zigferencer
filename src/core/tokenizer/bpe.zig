//! Byte-level BPE encode and decode over a `Vocab`. It is a pure datapath: no Vocab mutation, no
//! I/O, and a `gpa` taken only for the output slice. `Vocab` (vocab.zig) holds the state. The split
//! keeps encode and decode small and testable on their own.
//!
//! Encode. It greedy-matches the longest special token at the cursor. On a match, it flushes the
//! pending segment through BPE merge and emits the special id. Otherwise it consumes one byte into
//! `pending` through `vocab.byte_to_id[b]`. The flush at the end repeatedly merges the lowest-rank
//! adjacent pair until none remain.
//!
//! Plain ranges. `encodePromptAlloc` takes the byte ranges of the text that came from a request. A
//! special token matches only when its whole span lies outside every plain range, so request text
//! encodes as ordinary bytes. With no range, the cursor matches exactly what `matchSpecial`
//! matches, and `encodeAlloc` is that case. A merge or a whole pre-token never yields a special
//! id, so a special id comes only from a match.
//!
//! Decode. It concatenates the bytes of each token, walking codepoint by codepoint to undo the
//! GPT-2 byte-level mapping.
//!
//! Pre-split path. A non-default tokenizer (llama3, qwen, smollm) splits the text on a GPT-2-family
//! regex before BPE, as llama.cpp does, so merges never cross a pre-token boundary. For example,
//! llama3 splits "2147" into "214" and "7" (digit runs of 1-3). The default path BPE-merges the
//! whole inter-special span, and it stays byte-identical because `encodePromptAlloc` branches
//! first.
//! `preTokenLen` dispatches the per-family regex: `preSplitLlama` for llama3 and qwen,
//! `preSplitGpt2` for smollm. `ignore_merges` (llama3 only) emits a whole pre-token directly when
//! it is a vocab token. A special token is extracted before the regex runs, and `capAtSpecial`
//! keeps a pre-token from swallowing the start of a special.
//!
//! Classification scope. It is ASCII-exact for \p{L}, \p{N}, and \s. Non-ASCII codepoints count as
//! letters, which covers Latin and CJK words and the `\p{M}` marks of qwen. Non-ASCII digits and
//! punctuation are a documented gap, because the full Unicode tables are absent.
//!
//! Smollm regex. The llama.cpp `smollm` regex is `{ "\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|
//!  ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)" }`. The leading `\p{N}` pass isolates every digit, so
//! digits are single pre-tokens and a digit also bounds the trailing-whitespace lookahead. It
//! differs from llama3 and qwen in three ways: lowercase contractions, an optional leading char
//! that is a literal space only, and no `[\r\n]*` after symbols. It has no `\s*[\r\n]+` and no
//! final `\s+`, so an unmatched char falls back to one codepoint.

const std = @import("std");
const Allocator = std.mem.Allocator;
const span = @import("../span.zig");
const vocab_mod = @import("vocab.zig");
const Vocab = vocab_mod.Vocab;

// -- types ------------------------------------------------------------------

pub const EncodeError = error{UntokenizableByte} || Allocator.Error;

/// A byte range `[start, end)` of a text in which no special token can match.
pub const PlainRange = struct { start: usize, end: usize };

const Match = struct { id: u32, len: usize };

/// The cursor of one encode over its plain ranges. The position of a call never decreases, so
/// the cursor passes each range once.
const Specials = struct {
    plain: []const PlainRange,
    next: usize = 0,

    /// The longest special token at `pos` whose span lies outside every plain range.
    fn match(self: *Specials, vocab: *const Vocab, text: []const u8, pos: usize) ?Match {
        std.debug.assert(pos < text.len);

        while (self.next < self.plain.len) : (self.next += 1) {
            const r = self.plain[self.next];
            if (r.start < r.end and r.end > pos) break;
        }

        if (self.next == self.plain.len) return matchSpecial(vocab, text, pos);

        // A special must end at or before the start of the next plain range.
        const r = self.plain[self.next];
        if (r.start <= pos) return null;

        const m = matchSpecial(vocab, text[0..r.start], pos) orelse return null;
        std.debug.assert(pos + m.len <= r.start);
        return m;
    }
};

const CpClass = struct { letter: bool, number: bool, whitespace: bool, cr_or_lf: bool };

const Cp = struct { cp: u21, len: usize };

// -- api --------------------------------------------------------------------

/// Encode `text` to token ids: specials greedily, the rest through BPE merges. A caller never
/// prepends BOS: this function does, when `vocab.add_bos` and `sequence_start` hold. A follow-up
/// turn passes false, so BOS appears only at position 0.
pub fn encodeAlloc(
    vocab: *const Vocab,
    gpa: Allocator,
    text: []const u8,
    sequence_start: bool,
) EncodeError![]u32 {
    return encodePromptAlloc(vocab, gpa, text, &.{}, sequence_start);
}

/// Encode `text` to token ids, and match no special token that touches a plain range. The ranges
/// must be sorted and disjoint. With no range, every special that `matchSpecial` finds matches.
pub fn encodePromptAlloc(
    vocab: *const Vocab,
    gpa: Allocator,
    text: []const u8,
    plain: []const PlainRange,
    sequence_start: bool,
) EncodeError![]u32 {
    const sp = span.begin(.tokenize);
    defer sp.end();

    std.debug.assert(vocab.tokens.len > 0);
    for (plain, 0..) |r, i| {
        std.debug.assert(r.start <= r.end);
        std.debug.assert(r.end <= text.len);
        if (i > 0) std.debug.assert(plain[i - 1].end <= r.start);
    }

    // Take the pre-split path for a non-default tokenizer.
    var specials: Specials = .{ .plain = plain };
    if (vocab.pre != .default) {
        const pre_ids = try encodePreSplit(vocab, gpa, text, &specials, sequence_start);
        if (std.debug.runtime_safety) {
            for (pre_ids) |id| {
                std.debug.assert(id < vocab.tokens.len);
            }
        }

        return pre_ids;
    }

    // Scan the text: specials emit their id, other bytes join the pending segment.
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);

    if (sequence_start and vocab.add_bos) {
        try ids.append(gpa, vocab.bos_id);
    }

    var pending: std.ArrayList(u32) = .empty;
    defer pending.deinit(gpa);

    var pos: usize = 0;
    while (pos < text.len) {
        const matched = specials.match(vocab, text, pos);
        if (matched) |m| {
            try flushSegment(vocab, gpa, &pending, &ids);
            try ids.append(gpa, m.id);
            pos += m.len;
            continue;
        }

        const id = vocab.byte_to_id[text[pos]];
        if (id == vocab_mod.no_token) return error.UntokenizableByte;

        try pending.append(gpa, id);
        pos += 1;
    }

    try flushSegment(vocab, gpa, &pending, &ids);
    const out = try ids.toOwnedSlice(gpa);
    if (std.debug.runtime_safety) {
        for (out) |id| {
            std.debug.assert(id < vocab.tokens.len);
        }
    }

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
            const key = vocab_mod.pairKey(pending.items[i], pending.items[i + 1]);
            if (vocab.merge_rank.get(key)) |rank| {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                }
            }
        }

        if (best_idx == null) break;

        const idx = best_idx.?;
        const merged = vocab.merge_to_id.get(
            vocab_mod.pairKey(pending.items[idx], pending.items[idx + 1]),
        ).?;
        pending.items[idx] = merged;
        // Drop the consumed right-hand element.
        @memmove(
            pending.items[idx + 1 .. pending.items.len - 1],
            pending.items[idx + 2 ..],
        );
        pending.shrinkRetainingCapacity(pending.items.len - 1);
    }

    try out.appendSlice(gpa, pending.items);
    pending.clearRetainingCapacity();
}

// -- pre-split path: regex pre-tokenize, then BPE per pre-token -------------

fn encodePreSplit(
    vocab: *const Vocab,
    gpa: Allocator,
    text: []const u8,
    specials: *Specials,
    sequence_start: bool,
) EncodeError![]u32 {
    // Open the output with the BOS token.
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);

    if (sequence_start and vocab.add_bos) {
        try ids.append(gpa, vocab.bos_id);
    }

    var pending: std.ArrayList(u32) = .empty;
    defer pending.deinit(gpa);

    var mapped: std.ArrayList(u8) = .empty; // byte-level pre-token, for ignore_merges
    defer mapped.deinit(gpa);

    // Walk the text one pre-token at a time.
    var pos: usize = 0;
    while (pos < text.len) {
        if (specials.match(vocab, text, pos)) |m| {
            try ids.append(gpa, m.id);
            pos += m.len;
            continue;
        }

        const seg = capAtSpecial(vocab, text, specials.*, pos, preTokenLen(vocab, text, pos));
        const bytes = text[pos..][0..seg];

        // Emit a whole pre-token that the vocabulary holds.
        if (vocab.ignore_merges) {
            mapped.clearRetainingCapacity();
            for (bytes) |b| {
                try mapped.appendSlice(gpa, vocab_mod.byteLevel(b));
            }

            // A special spelled by a pre-token can come only from a plain range, so it stays bytes.
            const hit = vocab.token_to_id.get(mapped.items);
            if (hit != null and !vocab.is_special[hit.?]) {
                try ids.append(gpa, hit.?);
                pos += seg;
                continue;
            }
        }

        // Map the bytes and merge the pre-token.
        for (bytes) |b| {
            const id = vocab.byte_to_id[b];
            if (id == vocab_mod.no_token) return error.UntokenizableByte;
            try pending.append(gpa, id);
        }

        try flushSegment(vocab, gpa, &pending, &ids); // merges within this pre-token, then clears
        pos += seg;
    }

    return ids.toOwnedSlice(gpa);
}

/// The pre-token length `seg`, cut at the first special that starts strictly inside it. The
/// cursor is a copy, so the scan ahead leaves the cursor of the caller at `from`.
fn capAtSpecial(
    vocab: *const Vocab,
    text: []const u8,
    specials: Specials,
    from: usize,
    seg: usize,
) usize {
    var ahead = specials;
    var q = from + 1;
    const end = from + seg;
    while (q < end) : (q += 1) {
        if (ahead.match(vocab, text, q) != null) return q - from;
    }

    return seg;
}

/// ASCII-exact. Any non-ASCII codepoint counts as a letter.
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

/// Byte length of the next pre-token for `vocab.pre` at `from`. `.default` never reaches here.
fn preTokenLen(vocab: *const Vocab, text: []const u8, from: usize) usize {
    return switch (vocab.pre) {
        .llama3 => preSplitLlama(text, from, 3),
        .qwen => preSplitLlama(text, from, 1),
        .smollm => preSplitGpt2(text, from),
        .default => unreachable,
    };
}

/// Byte length of the next llama3 or qwen pre-token at `from`, always at least 1. The first regex
/// alternative that matches wins. `digit_max` is the `\p{N}{1,N}` run cap, 3 for llama3, 1 for
/// qwen.
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
                if ((a == 'r' and b == 'e') or (a == 'v' and b == 'e') or (a == 'l' and b == 'l')) {
                    return 3;
                }
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
            while (j < len and isSymbol(nextCp(text, j).cp)) {
                j += nextCp(text, j).len;
            }

            while (j < len and classify(nextCp(text, j).cp).cr_or_lf) {
                j += nextCp(text, j).len;
            }

            return j - from;
        }
    }

    // 5/6/7. Whitespace tail. Only a whitespace `first` reaches it.
    var j = from;
    var last_nl_end: usize = 0; // byte after the last CR/LF in the run (0 = none)
    var last_ws_start: usize = from;
    while (j < len) {
        const nc = nextCp(text, j);
        const ncl = classify(nc.cp);
        if (!ncl.whitespace) break;

        last_ws_start = j;
        j += nc.len;
        if (ncl.cr_or_lf) {
            last_nl_end = j;
        }
    }

    if (j == from) return first.len; // safety: nothing classified (unreachable)
    if (last_nl_end > from) return last_nl_end - from; // 5. \s*[\r\n]+
    if (last_ws_start > from and j < len) return last_ws_start - from; // 6. \s+(?!\S)

    return j - from; // 7. \s+
}

/// Byte length of the next smollm (GPT-2) pre-token at `from`, always at least 1.
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
            if ((a == 'r' and b == 'e') or (a == 'v' and b == 'e') or (a == 'l' and b == 'l')) {
                return 3;
            }
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
        if (first.cp == ' ' and from + 1 < len and isSymbol(nextCp(text, from + 1).cp)) {
            k = from + 1;
        }

        if (isSymbol(nextCp(text, k).cp)) {
            var j = k;
            while (j < len and isSymbol(nextCp(text, j).cp)) {
                j += nextCp(text, j).len;
            }

            return j - from;
        }
    }

    // 5. \s+(?!\S). The lookahead also holds before a digit and at EOF. A lone whitespace
    //    before \S that nothing else matched falls back to one codepoint.
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

/// The merges of the synthetic vocab, in rank order, as byte-level strings. They cover letters, a
/// leading space (U+0120), digits, the two bytes of U+00E9, and a newline pair (U+010A). The last
/// merge spells the special `<s`, so it adds no token of its own.
const test_merges = [_][]const u8{
    "a b",
    "ab ab",
    "\u{120} a",
    "\u{120}a b",
    "1 2",
    "\u{C3} \u{A9}",
    "\u{10A} \u{10A}",
    "< s",
};
const test_merge_tokens = test_merges.len - 1;

/// Longest first, as the naive model reads them. `<|s` is a prefix of `<|s|>`, and `<s` is one
/// pre-token under the llama rules.
const test_specials = [_][]const u8{ "<think>", "<|s|>", "<|s", "<s" };
const test_special_types = [_]i32{ 4, 3, 3, 3 };

fn testSpecialId(i: usize) u32 {
    std.debug.assert(i < test_specials.len);
    return @intCast(256 + test_merge_tokens + i);
}

/// A vocab of the 256 byte-level characters, one token per merge, and the
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
        if (!printable) {
            escape += 1;
        }

        var utf8: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(cp, &utf8);
        try tokens.append(arena, try arena.dupe(u8, utf8[0..len]));
        try types.append(arena, 1);
    }

    for (test_merges[0..test_merge_tokens]) |m| {
        const space = std.mem.findScalar(u8, m, ' ').?;
        try tokens.append(arena, try std.mem.concat(arena, u8, &.{ m[0..space], m[space + 1 ..] }));
        try types.append(arena, 1);
    }

    for (test_specials, test_special_types) |s, t| {
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
    return vocab_mod.buildFromParsed(
        arena,
        .{ .header = undefined, .kvs = kvs, .tensors = &.{}, .data_offset = 0, .alignment = 1 },
    );
}

const TestVocabs = [4]Vocab;

/// The synthetic vocab under each pre-split family, in the order of `PreType`.
fn testVocabs(arena: Allocator) !TestVocabs {
    var vocabs: TestVocabs = undefined;
    for (&vocabs, [_][]const u8{ "", "llama-bpe", "qwen2", "smollm" }) |*vocab, pre| {
        vocab.* = try testVocab(arena, pre);
    }

    return vocabs;
}

/// Text pieces that hit the merges, the specials, the pre-split classes, and
/// invalid UTF-8: a lone lead byte, a cut 3-byte sequence, and 0xFF.
const test_pieces = [_][]const u8{
    "a",
    "b",
    "ab",
    " ",
    " a",
    "1",
    "12",
    "\n",
    "\n\n",
    "\xc3\xa9",
    "<|s|>",
    "<think>",
    "<|s",
    "<s",
    "'s",
    "\t",
    "!",
    "\xc3",
    "\xe2\x82",
    "\xff",
    "\xf0\x9f\x98\x80",
};

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

    const vocabs = try testVocabs(arena_state.allocator());

    for (vocabs, [_]vocab_mod.PreType{ .default, .llama3, .qwen, .smollm }) |vocab, pre| {
        try std.testing.expectEqual(pre, vocab.pre);
        try std.testing.expectEqual(test_specials.len, vocab.specials_sorted.len);
        for (vocab.byte_to_id) |id| {
            try std.testing.expect(id != vocab_mod.no_token);
        }
    }

    try prop.forAll(
        .{ .site = 0xB9E0_0001, .cases = 600 },
        @as(*const TestVocabs, &vocabs),
        u64,
        prop.genSeed(*const TestVocabs),
        checkBpeRoundTrip,
    );
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
        if (!printable) {
            escape += 1;
        }

        entry.len = @intCast(std.unicode.utf8Encode(cp, &entry.bytes) catch unreachable);
    }

    return table;
}

/// The rank of the merge line `left ++ " " ++ right`, or null. A byte-level
/// spelling holds no ASCII space, so the line splits at its only space.
fn naiveRank(left: []const u8, right: []const u8) ?usize {
    for (test_merges, 0..) |line, rank| {
        const space = std.mem.findScalar(u8, line, ' ').?;
        if (std.mem.eql(u8, line[0..space], left) and std.mem.eql(u8, line[space + 1 ..], right)) {
            return rank;
        }
    }

    return null;
}

/// The id of a final symbol. Byte tokens come first, then one token per
/// merge line in order.
fn naiveId(table: *const [256]ByteLevel, symbol: []const u8) u32 {
    for (test_merges, 0..) |line, rank| {
        const space = std.mem.findScalar(u8, line, ' ').?;
        const left = line[0..space];
        const right = line[space + 1 ..];
        if (symbol.len == left.len + right.len and
            std.mem.startsWith(u8, symbol, left) and std.mem.endsWith(u8, symbol, right))
        {
            std.debug.assert(rank < test_merge_tokens);
            return @intCast(256 + rank);
        }
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

    // Read the text to its end.
    while (true) {
        const special = for (test_specials, 0..) |sp, i| {
            if (std.mem.startsWith(u8, text[pos..], sp)) break i;
        } else null;

        // Merge by lowest rank at a special token or at the end.
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
                symbols.items[i] = try std.mem.concat(
                    arena,
                    u8,
                    &.{ symbols.items[i], symbols.items[i + 1] },
                );
                _ = symbols.orderedRemove(i + 1);
            }

            for (symbols.items) |symbol| {
                try out.append(arena, naiveId(&table, symbol));
            }

            symbols.clearRetainingCapacity();
            if (pos == text.len) return;

            try out.append(arena, testSpecialId(special.?));
            pos += test_specials[special.?].len;
            continue;
        }

        // Add the next byte as a symbol.
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
    try prop.forAll(
        .{ .site = 0xB9E0_0002, .cases = 600 },
        @as(*const Vocab, &vocab),
        u64,
        prop.genSeed(*const Vocab),
        checkBpeAgainstNaive,
    );
}

test "each special piece encodes to its single id under every pre-split family" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const gpa = std.testing.allocator;
    for ([_][]const u8{ "", "llama-bpe", "qwen2", "smollm" }) |pre| {
        const vocab = try testVocab(arena, pre);
        for (test_specials, 0..) |special, i| {
            const id = testSpecialId(i);
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

fn countId(ids: []const u32, id: u32) usize {
    return std.mem.count(u32, ids, &.{id});
}

test "a special token that touches a plain range encodes as bytes" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const gpa = std.testing.allocator;
    const text = "a<|s|>b<think> <sc<s";
    // The count of each special in `test_specials` order: <think>, <|s|>, <|s, <s.
    const Case = struct { plain: []const PlainRange, want: [test_specials.len]usize };
    const cases = [_]Case{
        .{ .plain = &.{}, .want = .{ 1, 1, 0, 2 } },
        // An empty range touches no byte.
        .{ .plain = &.{.{ .start = 3, .end = 3 }}, .want = .{ 1, 1, 0, 2 } },
        .{ .plain = &.{.{ .start = 0, .end = text.len }}, .want = .{ 0, 0, 0, 0 } },
        // A range inside a special blocks it and each longer special around it.
        .{ .plain = &.{.{ .start = 3, .end = 4 }}, .want = .{ 1, 0, 0, 2 } },
        // A range past the end of a short special lets the short one match.
        .{ .plain = &.{.{ .start = 4, .end = 5 }}, .want = .{ 1, 0, 1, 2 } },
        // A range that ends inside a special blocks it.
        .{ .plain = &.{.{ .start = 6, .end = 8 }}, .want = .{ 0, 1, 0, 2 } },
        // Ranges that end where a special starts and start where it ends leave it whole.
        .{
            .plain = &.{ .{ .start = 0, .end = 1 }, .{ .start = 6, .end = 7 } },
            .want = .{ 1, 1, 0, 2 },
        },
        // A special that is one whole pre-token, or one merge, stays bytes in a range.
        .{ .plain = &.{.{ .start = 14, .end = 18 }}, .want = .{ 1, 1, 0, 1 } },
        .{ .plain = &.{.{ .start = 18, .end = 20 }}, .want = .{ 1, 1, 0, 1 } },
    };
    for ([_][]const u8{ "", "llama-bpe", "qwen2", "smollm" }) |pre| {
        const vocab = try testVocab(arena, pre);
        for (cases) |c| {
            const ids = try encodePromptAlloc(&vocab, gpa, text, c.plain, false);
            defer gpa.free(ids);

            for (c.want, 0..) |want, i| {
                try std.testing.expectEqual(want, countId(ids, testSpecialId(i)));
            }

            const back = try decodeAlloc(&vocab, gpa, ids);
            defer gpa.free(back);

            try std.testing.expectEqualStrings(text, back);
        }
    }
}

const max_prompt_segments = 8;
const max_segment_pieces = 6;

/// A random prompt: segments of template text and plain text, with the range of each plain one.
const PromptCase = struct {
    text: [max_prompt_segments * max_segment_pieces * 8]u8 = undefined,
    len: usize = 0,
    plain: [max_prompt_segments]PlainRange = undefined,
    plain_len: usize = 0,

    fn draw(seed: u64, plain_specials: bool) PromptCase {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        var c: PromptCase = .{};
        for (0..r.uintLessThan(usize, max_prompt_segments + 1)) |_| {
            const is_plain = r.boolean();
            const start = c.len;
            for (0..r.uintLessThan(usize, max_segment_pieces + 1)) |_| {
                if (r.boolean()) {
                    const piece = test_pieces[r.uintLessThan(usize, test_pieces.len)];
                    if (is_plain and !plain_specials and isSpecial(piece)) continue;

                    @memcpy(c.text[c.len..][0..piece.len], piece);
                    c.len += piece.len;
                } else {
                    c.text[c.len] = r.int(u8);
                    c.len += 1;
                }
            }

            if (is_plain) {
                c.plain[c.plain_len] = .{ .start = start, .end = c.len };
                c.plain_len += 1;
            }
        }

        return c;
    }

    /// True when some special string overlaps a plain range, so the two encodes can differ.
    fn touched(c: *const PromptCase) bool {
        for (test_specials) |sp| {
            var at: usize = 0;
            while (std.mem.findPos(u8, c.text[0..c.len], at, sp)) |i| : (at = i + 1) {
                for (c.plain[0..c.plain_len]) |r| {
                    if (r.start < r.end and r.start < i + sp.len and i < r.end) return true;
                }
            }
        }

        return false;
    }
};

fn isSpecial(piece: []const u8) bool {
    for (test_specials) |sp| {
        if (std.mem.eql(u8, sp, piece)) return true;
    }

    return false;
}

/// With no special string in or across a plain range, the prompt encode equals `encodeAlloc`.
fn checkPromptEqualsPlainEncode(vocabs: *const TestVocabs, seed: u64) anyerror!void {
    const c = PromptCase.draw(seed, false);
    if (c.touched()) return;

    const gpa = std.testing.allocator;
    for (vocabs) |*vocab| {
        const want = try encodeAlloc(vocab, gpa, c.text[0..c.len], true);
        defer gpa.free(want);

        const plain = c.plain[0..c.plain_len];
        const got = try encodePromptAlloc(vocab, gpa, c.text[0..c.len], plain, true);
        defer gpa.free(got);

        try std.testing.expectEqualSlices(u32, want, got);
    }
}

/// Every special id of a prompt encode spans bytes outside every plain range, and the ids decode
/// to the text.
fn checkPromptPlainHoldsNoSpecial(vocabs: *const TestVocabs, seed: u64) anyerror!void {
    const c = PromptCase.draw(seed, true);
    const text = c.text[0..c.len];
    const gpa = std.testing.allocator;
    for (vocabs) |*vocab| {
        const ids = try encodePromptAlloc(vocab, gpa, text, c.plain[0..c.plain_len], false);
        defer gpa.free(ids);

        const back = try decodeAlloc(vocab, gpa, ids);
        defer gpa.free(back);

        try std.testing.expectEqualSlices(u8, text, back);

        // Walk the ids with their byte offsets in the text.
        var at: usize = 0;
        for (ids) |id| {
            const bytes = try decodeAlloc(vocab, gpa, &.{id});
            defer gpa.free(bytes);

            const is_special = for (vocab.specials_sorted) |sp| {
                if (sp.id == id) break true;
            } else false;
            if (is_special) {
                for (c.plain[0..c.plain_len]) |r| {
                    try std.testing.expect(r.start == r.end or at + bytes.len <= r.start or
                        r.end <= at);
                }
            }

            at += bytes.len;
        }

        try std.testing.expectEqual(text.len, at);
    }
}

test "property: a prompt with no special in a plain range encodes as encodeAlloc does" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const vocabs = try testVocabs(arena_state.allocator());

    try prop.forAll(
        .{ .site = 0xB9E0_0003, .cases = 600 },
        @as(*const TestVocabs, &vocabs),
        u64,
        prop.genSeed(*const TestVocabs),
        checkPromptEqualsPlainEncode,
    );
}

test "property: no special id comes from a plain range, and the ids decode to the text" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const vocabs = try testVocabs(arena_state.allocator());

    try prop.forAll(
        .{ .site = 0xB9E0_0004, .cases = 600 },
        @as(*const TestVocabs, &vocabs),
        u64,
        prop.genSeed(*const TestVocabs),
        checkPromptPlainHoldsNoSpecial,
    );
}
