//! `Vocab`, a boundary value derived from the GGUF tokenizer KVs. It is built
//! once at load time and consumed by `bpe.encodeAlloc` and `bpe.decodeAlloc`,
//! which stay pure over Vocab with storage on the caller's arena. GPT-2
//! byte-level BPE: every byte 0..255 maps to a printable Unicode codepoint and
//! the model only ever sees those. The byte-to-codepoint table is fixed and
//! lives in `comptime` (see `gpt2_byte_table`). The encode-time speed lives in
//! the data, not the code: `byte_to_id` is a flat [256]u32 lookup, the merge
//! maps are u64-keyed, and `specials_sorted` is length-descending so the
//! encoder greedy-matches the longest special token first.

const std = @import("std");
const Allocator = std.mem.Allocator;
const parser = @import("../../io/gguf/parser.zig");

// -- constants --------------------------------------------------------------

/// GGUF `tokenizer.ggml.token_type` values of the added tokens that the
/// encoder matches whole. Type 1 is a normal token.
const token_type_control: i32 = 3;

const token_type_user_defined: i32 = 4;

/// Fixed bijection: byte 0..255 -> GPT-2 byte-level codepoint and its UTF-8 bytes.
/// Pair-validation: asserts exactly 68 escape codepoints, which is what the
/// GPT-2 byte-to-unicode algorithm produces.
const gpt2_byte_table: [256]ByteEntry = blk: {
    @setEvalBranchQuota(10_000);
    var t: [256]ByteEntry = undefined;
    var unmapped: u21 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or
            (b >= 174 and b <= 255);
        const cp: u21 = if (printable) @intCast(b) else escape: {
            const c = 256 + unmapped;
            unmapped += 1;
            break :escape c;
        };

        var buf: [4]u8 = .{ 0, 0, 0, 0 };
        const n = std.unicode.utf8Encode(cp, &buf) catch unreachable;
        t[b] = .{ .cp = cp, .bytes = buf, .len = n };
    }

    if (unmapped != 68) @compileError("GPT-2 byte map must produce exactly 68 escape codepoints");

    break :blk t;
};

/// Sentinel for `Vocab.byte_to_id` slots whose GPT-2 char does not exist in the
/// vocab. `bpe.encodeAlloc` returns `error.UntokenizableByte` if encountered.
pub const no_token: u32 = std.math.maxInt(u32);

// -- types ------------------------------------------------------------------

pub const Error = error{
    MissingTokenizerKey,
    WrongTokenizerKeyType,
    InvalidMerge,
    TokenStringTooLong,
} || Allocator.Error;

pub const Special = struct {
    id: u32,
    bytes: []const u8,
};

/// Pre-tokenization regime, from `tokenizer.ggml.pre`, mirroring llama.cpp's
/// per-family regexes. `.default` BPE-merges the whole inter-special span as one
/// segment (only correct for a tokenizer that truly does no pre-split). The rest
/// split the text into pre-tokens first so merges never cross a boundary:
///   - `.llama3` (`llama-bpe`): GPT-2/llama3 regex, digit runs of 1-3, plus
///     `ignore_merges` (a whole pre-token that is a token is emitted directly).
///   - `.qwen` (`qwen35`/`qwen2`): the same regex but single-digit runs, no
///     `ignore_merges`.
///   - `.smollm` (`smollm`): the classic GPT-2 regex, with a space-only leading
///     char, lowercase contractions, single-digit runs, simpler whitespace.
pub const PreType = enum { default, llama3, qwen, smollm };

pub const Vocab = struct {
    /// id -> token bytes (in GPT-2 byte-level encoding for normal tokens, raw
    /// UTF-8 for special tokens like `<|im_start|>`). Backed by the GGUF mmap.
    tokens: []const []const u8,
    /// (left_id << 32 | right_id) -> rank. Lower = earlier merge.
    merge_rank: std.AutoHashMap(u64, u32),
    /// (left_id << 32 | right_id) -> merged token id.
    merge_to_id: std.AutoHashMap(u64, u32),
    /// byte 0..255 -> vocab id of that byte's GPT-2 byte-level character.
    byte_to_id: [256]u32,
    /// Codepoint 0..0x143ish -> original byte. Used during decode to undo the
    /// byte-level encoding.
    codepoint_to_byte: std.AutoHashMap(u21, u8),
    /// Special tokens sorted by length DESC for longest-first greedy match.
    specials_sorted: []const Special,

    bos_id: u32,
    eos_id: u32,
    /// Whether `bpe.encodeAlloc` prepends `bos_id`. It comes from the model's
    /// tokenizer metadata, not from the caller (see `addBos`).
    add_bos: bool,
    /// Pre-tokenization regime (see `PreType`). Selects the encode path in `bpe`.
    pre: PreType,
    /// llama3: emit a whole pre-token directly when it is itself a vocab token,
    /// rather than BPE-merging its bytes. Mirrors llama.cpp `ignore_merges`.
    ignore_merges: bool,
    /// Byte-level token string -> id. Built for merge resolution either way and
    /// kept for the llama3 whole-pre-token (`ignore_merges`) lookup.
    token_to_id: std.StringHashMap(u32),
};

/// One byte's GPT-2 byte-level entry: its codepoint and the UTF-8 bytes that
/// encode it. Computed entirely at comptime (see `gpt2_byte_table`).
const ByteEntry = struct {
    cp: u21,
    bytes: [4]u8,
    len: u3,
};

// -- api --------------------------------------------------------------------

/// Pack a (left, right) token-id pair into a u64 hashmap key.
pub fn pairKey(left: u32, right: u32) u64 {
    return (@as(u64, left) << 32) | @as(u64, right);
}

/// The UTF-8 bytes of byte `b`'s GPT-2 byte-level character (e.g. space -> codepoint U+0120).
/// `bpe` assembles a pre-token's byte-level string from these for the
/// `ignore_merges` whole-pre-token lookup against `token_to_id`.
pub fn byteLevel(b: u8) []const u8 {
    const e = &gpt2_byte_table[b];
    return e.bytes[0..e.len];
}

/// Build a Vocab from a parsed GGUF. Allocations live on `arena`.
pub fn buildFromParsed(arena: Allocator, p: parser.Parsed) Error!Vocab {
    const tokens = try lookupStringArray(p, "tokenizer.ggml.tokens");
    const merges = try lookupStringArray(p, "tokenizer.ggml.merges");
    const token_types = try lookupI32Array(p, "tokenizer.ggml.token_type");
    // BOS id is optional: ChatML models (Qwen3.5) set add_bos_token=false and
    // can omit it. Whether the encoder prepends it is the model's call (addBos).
    const bos = lookupU32(p, "tokenizer.ggml.bos_token_id") catch 0;
    const eos = try lookupU32(p, "tokenizer.ggml.eos_token_id");
    const add_bos = addBos(p);
    const pre = preType(p);
    // llama.cpp enables ignore_merges for the llama3 (llama-bpe) pre-tokenizer.
    const ignore_merges = (pre == .llama3);

    std.debug.assert(tokens.len == token_types.len);
    std.debug.assert(tokens.len > 0);

    // 1. String -> id (used for merge & byte resolution).
    var string_to_id = std.StringHashMap(u32).init(arena);
    try string_to_id.ensureTotalCapacity(@intCast(tokens.len));
    for (tokens, 0..) |t, i| {
        // `putNoClobber` faults on a duplicate token, which is a GGUF defect.
        // The loader tolerates the rare duplicate by keeping the *first* id.
        if (!string_to_id.contains(t)) {
            try string_to_id.put(t, @intCast(i));
        }
    }

    // 2. Special-flag list, local to this build. Step 5 turns it into
    //    `specials_sorted`, the one form the encoder reads. The reference
    //    tokenizer matches user-defined added tokens such as `<think>` as
    //    specials, so each one encodes to one id.
    const is_special = try arena.alloc(bool, tokens.len);
    for (token_types, is_special) |tt, *b| {
        b.* = (tt == token_type_control or tt == token_type_user_defined);
    }

    // 3. Walk merges in order, and record rank + the merged id.
    var merge_rank = std.AutoHashMap(u64, u32).init(arena);
    var merge_to_id = std.AutoHashMap(u64, u32).init(arena);
    try merge_rank.ensureTotalCapacity(@intCast(merges.len));
    try merge_to_id.ensureTotalCapacity(@intCast(merges.len));

    var key_buf: [512]u8 = undefined;
    for (merges, 0..) |m, rank| {
        const space = std.mem.indexOfScalar(u8, m, ' ') orelse return error.InvalidMerge;
        const left_str = m[0..space];
        const right_str = m[space + 1 ..];
        const left_id = string_to_id.get(left_str) orelse return error.InvalidMerge;
        const right_id = string_to_id.get(right_str) orelse return error.InvalidMerge;

        const total = left_str.len + right_str.len;
        if (total > key_buf.len) return error.TokenStringTooLong;

        @memcpy(key_buf[0..left_str.len], left_str);
        @memcpy(key_buf[left_str.len..total], right_str);
        const merged_id = string_to_id.get(key_buf[0..total]) orelse return error.InvalidMerge;

        const key = pairKey(left_id, right_id);
        try merge_rank.put(key, @intCast(rank));
        try merge_to_id.put(key, merged_id);
    }

    // 4. byte->id and codepoint->byte tables. The byte->codepoint map and its
    //    UTF-8 encoding are both fixed math and live in `comptime` (see
    //    `gpt2_byte_table`). At runtime only the vocab string lookup remains.
    var byte_to_id: [256]u32 = undefined;
    var codepoint_to_byte = std.AutoHashMap(u21, u8).init(arena);
    try codepoint_to_byte.ensureTotalCapacity(256);

    inline for (gpt2_byte_table, 0..) |entry, b| {
        const utf8 = entry.bytes[0..entry.len];
        // Tolerate vocab gaps: SmolLM2 drops a handful of rare bytes from the
        // 256-char base set. Mark them, so `bpe.encodeAlloc` fails loud on one.
        byte_to_id[b] = string_to_id.get(utf8) orelse no_token;
        try codepoint_to_byte.put(entry.cp, @intCast(b));
    }

    // 5. specials_sorted (longest first).
    var n_specials: usize = 0;
    for (is_special) |s| {
        if (s) {
            n_specials += 1;
        }
    }

    const specials = try arena.alloc(Special, n_specials);
    var k: usize = 0;
    for (tokens, is_special, 0..) |t, sp, i| {
        if (sp) {
            specials[k] = .{ .id = @intCast(i), .bytes = t };
            k += 1;
        }
    }

    std.sort.pdq(Special, specials, {}, lenDescLessThan);

    std.debug.assert(bos < tokens.len);
    std.debug.assert(eos < tokens.len);

    return .{
        .tokens = tokens,
        .merge_rank = merge_rank,
        .merge_to_id = merge_to_id,
        .byte_to_id = byte_to_id,
        .codepoint_to_byte = codepoint_to_byte,
        .specials_sorted = specials,
        .bos_id = bos,
        .eos_id = eos,
        .add_bos = add_bos,
        .pre = pre,
        .ignore_merges = ignore_merges,
        .token_to_id = string_to_id,
    };
}

// -- helpers ----------------------------------------------------------------

/// Whether the encoder prepends BOS, mirroring llama.cpp (`llama-vocab.cpp`).
/// The pre-tokenizer type sets the default, true for the llama3 family and
/// false otherwise, and an explicit `tokenizer.ggml.add_bos_token` wins. The
/// default is load-bearing: Llama-3 GGUFs ship the `llama-bpe` pre type with no
/// explicit key, and without their BOS anchor short raw prompts decode to
/// garbage.
fn addBos(p: parser.Parsed) bool {
    if (lookupBool(p, "tokenizer.ggml.add_bos_token")) |explicit| return explicit;
    return preType(p) == .llama3;
}

/// Map `tokenizer.ggml.pre` to a `PreType`, mirroring llama.cpp's name -> pre-type
/// table for the families this engine supports. An unrecognized name maps
/// to `.default` (whole-span BPE) rather than guessing a regex.
fn preType(p: parser.Parsed) PreType {
    const pre = lookupString(p, "tokenizer.ggml.pre") orelse return .default;
    const table = [_]struct { name: []const u8, t: PreType }{
        .{ .name = "llama3", .t = .llama3 },
        .{ .name = "llama-v3", .t = .llama3 },
        .{ .name = "llama-bpe", .t = .llama3 },
        .{ .name = "qwen35", .t = .qwen },
        .{ .name = "qwen2", .t = .qwen },
        .{ .name = "smollm", .t = .smollm },
    };
    for (table) |e| {
        if (std.mem.eql(u8, pre, e.name)) return e.t;
    }

    return .default;
}

fn lenDescLessThan(_: void, a: Special, b: Special) bool {
    return a.bytes.len > b.bytes.len;
}

fn lookupKv(p: parser.Parsed, key: []const u8) ?*const parser.Value {
    for (p.kvs) |*kv| {
        if (std.mem.eql(u8, kv.key, key)) return &kv.value;
    }

    return null;
}

fn lookupStringArray(p: parser.Parsed, key: []const u8) Error![]const []const u8 {
    const v = lookupKv(p, key) orelse return error.MissingTokenizerKey;
    return switch (v.*) {
        .array => |a| switch (a) {
            .strings => |s| s,
            else => error.WrongTokenizerKeyType,
        },
        else => error.WrongTokenizerKeyType,
    };
}

fn lookupI32Array(p: parser.Parsed, key: []const u8) Error![]const i32 {
    const v = lookupKv(p, key) orelse return error.MissingTokenizerKey;
    return switch (v.*) {
        .array => |a| switch (a) {
            .i32s => |s| s,
            else => error.WrongTokenizerKeyType,
        },
        else => error.WrongTokenizerKeyType,
    };
}

fn lookupBool(p: parser.Parsed, key: []const u8) ?bool {
    const v = lookupKv(p, key) orelse return null;
    return switch (v.*) {
        .bool_val => |b| b,
        else => null,
    };
}

fn lookupString(p: parser.Parsed, key: []const u8) ?[]const u8 {
    const v = lookupKv(p, key) orelse return null;
    return switch (v.*) {
        .string => |s| s,
        else => null,
    };
}

fn lookupU32(p: parser.Parsed, key: []const u8) Error!u32 {
    const v = lookupKv(p, key) orelse return error.MissingTokenizerKey;
    return switch (v.*) {
        .u32_val => |u| u,
        .u64_val => |u| @intCast(u),
        .i32_val => |i| @intCast(i),
        .i64_val => |i| @intCast(i),
        else => error.WrongTokenizerKeyType,
    };
}

// -- tests ------------------------------------------------------------------

fn parsedWithKvs(kvs: []const parser.Kv) parser.Parsed {
    return .{ .header = undefined, .kvs = kvs, .tensors = &.{}, .data_offset = 0, .alignment = 1 };
}

test "pairKey packs left/right into u64" {
    try std.testing.expectEqual(@as(u64, (1 << 32) | 2), pairKey(1, 2));
    try std.testing.expectEqual(@as(u64, (0xFF << 32) | 0xAB), pairKey(0xFF, 0xAB));
}

test "addBos: llama3-family pre defaults true, an explicit key wins, else false" {
    const llama = [_]parser.Kv{
        .{ .key = "tokenizer.ggml.pre", .value = .{ .string = "llama-bpe" } },
    };
    const llama_explicit_off = [_]parser.Kv{
        .{ .key = "tokenizer.ggml.pre", .value = .{ .string = "llama-bpe" } },
        .{ .key = "tokenizer.ggml.add_bos_token", .value = .{ .bool_val = false } },
    };
    const smollm = [_]parser.Kv{
        .{ .key = "tokenizer.ggml.pre", .value = .{ .string = "smollm" } },
    };
    try std.testing.expect(addBos(parsedWithKvs(&llama)));
    try std.testing.expect(!addBos(parsedWithKvs(&llama_explicit_off)));
    try std.testing.expect(!addBos(parsedWithKvs(&smollm)));
    try std.testing.expect(!addBos(parsedWithKvs(&.{})));
}
