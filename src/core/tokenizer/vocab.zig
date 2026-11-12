//! `Vocab`, a boundary value derived from the GGUF tokenizer metadata. It is built once at load
//! time and consumed by `bpe.encodeAlloc` and `bpe.decodeAlloc`, with storage on the caller's
//! arena.
//!
//! Comptime. The GPT-2 byte table (`gpt2_byte_table`) is built at comptime.
//!
//! Byte level. GPT-2 byte-level BPE maps every byte 0..255 to a printable Unicode codepoint, and
//! the model only ever sees those. The byte-to-codepoint table is fixed and lives in `comptime`.
//! The table must produce exactly 68 escape codepoints, which is what the GPT-2 byte-to-unicode
//! algorithm produces.
//!
//! Encode speed. It lives in the data, not the code: `byte_to_id` is a flat [256]u32 lookup, the
//! merge maps are u64-keyed, and `specials_sorted` is length-descending so the encoder
//! greedy-matches the longest special token first.
//!
//! Pre-tokenization. `tokenizer.ggml.pre` selects a `PreType` that mirrors the per-family regexes
//! of llama.cpp. `.default` BPE-merges the whole inter-special span as one segment, which is
//! correct only for a tokenizer that does no pre-split. The other types split the text into
//! pre-tokens first, so merges never cross a boundary. `.llama3` (`llama-bpe`) uses the GPT-2 and
//! llama3 regex with digit runs of 1-3, plus `ignore_merges`: a whole pre-token that is a token
//! is emitted directly. `.qwen` (`qwen35`, `qwen2`) uses the same regex with single-digit runs and
//! no `ignore_merges`. `.smollm` uses the classic GPT-2 regex with a space-only leading char,
//! lowercase contractions, single-digit runs, and simpler whitespace.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const parser = @import("../../io/gguf/parser.zig");

const log = std.log.scoped(.gguf);

// -- constants --------------------------------------------------------------

/// GGUF `tokenizer.ggml.token_type` values of the added tokens that the encoder matches whole.
const token_type_control: i32 = 3;

const token_type_user_defined: i32 = 4;

/// The test runner fails a test that logs an error, and a refusal test expects these lines.
const refusal_level: std.log.Level = if (builtin.is_test) .debug else .err;

/// Fixed bijection: byte 0..255 to its GPT-2 byte-level codepoint and the UTF-8 bytes of it.
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

/// Sentinel for a `Vocab.byte_to_id` slot whose GPT-2 char is not in the vocab. `bpe.encodeAlloc`
/// returns `error.UntokenizableByte` on it.
pub const no_token: u32 = std.math.maxInt(u32);

// -- types ------------------------------------------------------------------

pub const Error = error{
    MissingTokenizerKey,
    WrongTokenizerKeyType,
    InvalidMerge,
    TokenStringTooLong,
    /// A tokenizer value disagrees with the token list: a count or a special id.
    BadTokenizerValue,
} || Allocator.Error;

pub const Special = struct {
    id: u32,
    bytes: []const u8,
};

/// Pre-tokenization regime, from `tokenizer.ggml.pre`.
pub const PreType = enum { default, llama3, qwen, smollm };

pub const Vocab = struct {
    /// id to token bytes: byte-level for normal tokens, raw UTF-8 for specials. Backed by the mmap.
    tokens: []const []const u8,
    /// (left_id << 32 | right_id) -> rank. Lower = earlier merge.
    merge_rank: std.AutoHashMap(u64, u32),
    /// (left_id << 32 | right_id) -> merged token id.
    merge_to_id: std.AutoHashMap(u64, u32),
    /// byte 0..255 -> vocab id of that byte's GPT-2 byte-level character.
    byte_to_id: [256]u32,
    /// Codepoint to original byte. Decode uses it to undo the byte-level encoding.
    codepoint_to_byte: std.AutoHashMap(u21, u8),
    /// Special tokens sorted by length DESC for longest-first greedy match.
    specials_sorted: []const Special,
    /// id -> true for a control or user-defined token, the tokens in `specials_sorted`.
    is_special: []const bool,
    /// id -> true for a control token only. The serve pieces give a control token no text.
    is_control: []const bool,

    bos_id: u32,
    eos_id: u32,
    /// Whether `bpe.encodeAlloc` prepends `bos_id`. It comes from the model metadata.
    add_bos: bool,
    /// Pre-tokenization regime. Selects the encode path in `bpe`.
    pre: PreType,
    /// llama3: emit a whole pre-token directly when it is a vocab token. Mirrors `ignore_merges`.
    ignore_merges: bool,
    /// Byte-level token string to id. Merge resolution and the `ignore_merges` lookup use it.
    token_to_id: std.StringHashMap(u32),
};

/// One byte's GPT-2 byte-level entry: its codepoint and the UTF-8 bytes that
/// encode it. Computed entirely at comptime.
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

/// The UTF-8 bytes of the GPT-2 byte-level character of byte `b`, such as U+0120 for a space.
/// `bpe` builds the byte-level string of a pre-token from these.
pub fn byteLevel(b: u8) []const u8 {
    const e = &gpt2_byte_table[b];
    return e.bytes[0..e.len];
}

/// Build a Vocab from a parsed GGUF. Allocations live on `arena`.
pub fn buildFromParsed(arena: Allocator, p: parser.Parsed) Error!Vocab {
    const tokens = try lookupStringArray(p, "tokenizer.ggml.tokens");
    const merges = try lookupStringArray(p, "tokenizer.ggml.merges");
    const token_types = try lookupI32Array(p, "tokenizer.ggml.token_type");
    // The BOS id is optional, because ChatML models such as Qwen3.5 can omit it.
    const bos = try lookupU32Or(p, "tokenizer.ggml.bos_token_id", 0);
    const eos = try lookupU32(p, "tokenizer.ggml.eos_token_id");
    const add_bos = addBos(p);
    const pre = preType(p);
    if (pre == .default) {
        if (lookupString(p, "tokenizer.ggml.pre")) |name| {
            log.info("the tokenizer pre-type {s} is not known, so the encoder splits no " ++
                "pre-tokens and merges each span between specials whole", .{name});
        }
    }
    // llama.cpp enables ignore_merges for the llama3 pre-tokenizer.
    const ignore_merges = (pre == .llama3);

    // Dev tools call this function directly, so it checks the file values the loader also checks.
    if (tokens.len == 0 or tokens.len != token_types.len) {
        refuse("the tokenizer holds {d} tokens and {d} token types", .{
            tokens.len, token_types.len,
        });
        return error.BadTokenizerValue;
    }
    if (bos >= tokens.len or eos >= tokens.len) {
        refuse("the BOS id {d} or the EOS id {d} is not below the token count {d}", .{
            bos, eos, tokens.len,
        });
        return error.BadTokenizerValue;
    }

    // Map each token string to its id.
    var string_to_id = std.StringHashMap(u32).init(arena);
    try string_to_id.ensureTotalCapacity(@intCast(tokens.len));
    for (tokens, 0..) |t, i| {
        // A duplicate token is a GGUF defect. The first id wins.
        if (!string_to_id.contains(t)) {
            string_to_id.putAssumeCapacity(t, @intCast(i));
        }
    }

    // Flag the specials. The reference tokenizer matches user-defined added tokens such as
    // `<think>` as specials, so each one encodes to one id.
    const is_special = try arena.alloc(bool, tokens.len);
    const is_control = try arena.alloc(bool, tokens.len);
    for (token_types, is_special, is_control) |tt, *special, *control| {
        special.* = (tt == token_type_control or tt == token_type_user_defined);
        control.* = (tt == token_type_control);
    }

    // Record the rank and the merged id of each merge, in order.
    var merge_rank = std.AutoHashMap(u64, u32).init(arena);
    var merge_to_id = std.AutoHashMap(u64, u32).init(arena);
    try merge_rank.ensureTotalCapacity(@intCast(merges.len));
    try merge_to_id.ensureTotalCapacity(@intCast(merges.len));

    var key_buf: [512]u8 = undefined;
    for (merges, 0..) |m, rank| {
        const space = std.mem.findScalar(u8, m, ' ') orelse return invalidMerge(rank);
        const left_str = m[0..space];
        const right_str = m[space + 1 ..];
        const left_id = string_to_id.get(left_str) orelse return invalidMerge(rank);
        const right_id = string_to_id.get(right_str) orelse return invalidMerge(rank);

        const total = left_str.len + right_str.len;
        if (total > key_buf.len) {
            refuse("merge {d} joins {d} bytes, past the limit of {d}", .{
                rank, total, key_buf.len,
            });
            return error.TokenStringTooLong;
        }

        @memcpy(key_buf[0..left_str.len], left_str);
        @memcpy(key_buf[left_str.len..total], right_str);
        const merged_id = string_to_id.get(key_buf[0..total]) orelse return invalidMerge(rank);
        // A merge never makes a special. Outside a plain range the encoder matches the special
        // before any merge, so only a plain range can reach such a merge, and it must stay bytes.
        if (is_special[merged_id]) continue;

        const key = pairKey(left_id, right_id);
        merge_rank.putAssumeCapacity(key, @intCast(rank));
        merge_to_id.putAssumeCapacity(key, merged_id);
    }

    // Build the byte to id and codepoint to byte tables. Only the vocab lookup runs at runtime.
    var byte_to_id: [256]u32 = undefined;
    var codepoint_to_byte = std.AutoHashMap(u21, u8).init(arena);
    try codepoint_to_byte.ensureTotalCapacity(256);

    inline for (gpt2_byte_table, 0..) |entry, b| {
        const utf8 = entry.bytes[0..entry.len];
        // SmolLM2 drops a few rare bytes from the 256-char base set. Mark the gap, so
        // `bpe.encodeAlloc` fails loud on one.
        byte_to_id[b] = string_to_id.get(utf8) orelse no_token;
        codepoint_to_byte.putAssumeCapacity(entry.cp, @intCast(b));
    }

    // Sort the specials, longest first.
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

    std.debug.assert(specials.len <= tokens.len);

    return .{
        .tokens = tokens,
        .merge_rank = merge_rank,
        .merge_to_id = merge_to_id,
        .byte_to_id = byte_to_id,
        .codepoint_to_byte = codepoint_to_byte,
        .specials_sorted = specials,
        .is_special = is_special,
        .is_control = is_control,
        .bos_id = bos,
        .eos_id = eos,
        .add_bos = add_bos,
        .pre = pre,
        .ignore_merges = ignore_merges,
        .token_to_id = string_to_id,
    };
}

// -- helpers ----------------------------------------------------------------

/// Whether the encoder prepends BOS, as in llama.cpp. The pre type sets the default, true for
/// llama3. An explicit `tokenizer.ggml.add_bos_token` wins. Llama-3 GGUFs carry no explicit key,
/// and without their BOS anchor short raw prompts decode to garbage.
fn addBos(p: parser.Parsed) bool {
    if (lookupBool(p, "tokenizer.ggml.add_bos_token")) |explicit| return explicit;
    return preType(p) == .llama3;
}

/// Map `tokenizer.ggml.pre` to a `PreType`, as the llama.cpp table does for the supported
/// families. An unrecognized name maps to `.default` rather than guessing a regex.
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

fn refuse(comptime format: []const u8, args: anytype) void {
    switch (refusal_level) {
        .err => log.err(format, args),
        else => log.debug(format, args),
    }
}

fn lenDescLessThan(_: void, a: Special, b: Special) bool {
    return a.bytes.len > b.bytes.len;
}

fn findMetadata(p: parser.Parsed, key: []const u8) ?*const parser.Value {
    for (p.metadata) |*pair| {
        if (std.mem.eql(u8, pair.key, key)) return &pair.value;
    }

    return null;
}

/// A merge that is not two tokens of the list whose join is a token of the list.
fn invalidMerge(rank: usize) Error {
    refuse("merge {d} is not two tokens that join to a token", .{rank});
    return error.InvalidMerge;
}

fn requiredKv(p: parser.Parsed, key: []const u8) Error!*const parser.Value {
    return findMetadata(p, key) orelse {
        refuse("the tokenizer key {s} is missing", .{key});
        return error.MissingTokenizerKey;
    };
}

fn wrongType(key: []const u8, v: *const parser.Value) Error {
    refuse("the tokenizer key {s} holds a value of type {t}", .{ key, v.* });
    return error.WrongTokenizerKeyType;
}

fn lookupStringArray(p: parser.Parsed, key: []const u8) Error![]const []const u8 {
    const v = try requiredKv(p, key);
    return switch (v.*) {
        .array => |a| switch (a) {
            .strings => |s| s,
            else => wrongType(key, v),
        },
        else => wrongType(key, v),
    };
}

fn lookupI32Array(p: parser.Parsed, key: []const u8) Error![]const i32 {
    const v = try requiredKv(p, key);
    return switch (v.*) {
        .array => |a| switch (a) {
            .i32s => |s| s,
            else => wrongType(key, v),
        },
        else => wrongType(key, v),
    };
}

fn lookupBool(p: parser.Parsed, key: []const u8) ?bool {
    const v = findMetadata(p, key) orelse return null;
    return switch (v.*) {
        .bool_val => |b| b,
        else => null,
    };
}

fn lookupString(p: parser.Parsed, key: []const u8) ?[]const u8 {
    const v = findMetadata(p, key) orelse return null;
    return switch (v.*) {
        .string => |s| s,
        else => null,
    };
}

/// A value that does not fit a `u32` is a bad value, not a narrowing cast.
fn lookupU32(p: parser.Parsed, key: []const u8) Error!u32 {
    const v = try requiredKv(p, key);
    const wide: i128 = switch (v.*) {
        .u32_val => |u| u,
        .u64_val => |u| u,
        .i32_val => |i| i,
        .i64_val => |i| i,
        else => return wrongType(key, v),
    };

    return std.math.cast(u32, wide) orelse {
        refuse("the tokenizer key {s} holds {d}, which is not a 32-bit id", .{ key, wide });
        return error.BadTokenizerValue;
    };
}

/// The value of an optional key. Only an absent key takes `default`, so a key of the wrong
/// type fails and does not fall back in silence.
fn lookupU32Or(p: parser.Parsed, key: []const u8, default: u32) Error!u32 {
    if (findMetadata(p, key) == null) return default;
    return lookupU32(p, key);
}

// -- tests ------------------------------------------------------------------

fn parsedWithMetadata(metadata: []const parser.MetadataPair) parser.Parsed {
    return .{
        .header = undefined,
        .metadata = metadata,
        .tensors = &.{},
        .data_offset = 0,
        .alignment = 1,
    };
}

test "pairKey packs left/right into u64" {
    try std.testing.expectEqual(@as(u64, (1 << 32) | 2), pairKey(1, 2));
    try std.testing.expectEqual(@as(u64, (0xFF << 32) | 0xAB), pairKey(0xFF, 0xAB));
}

test "buildFromParsed: an absent BOS key takes 0, and a bad BOS key refuses" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const tokens = [_][]const u8{ "a", "b" };
    const types = [_]i32{ 1, 1 };
    const required = [_]parser.MetadataPair{
        .{ .key = "tokenizer.ggml.tokens", .value = .{ .array = .{ .strings = &tokens } } },
        .{ .key = "tokenizer.ggml.merges", .value = .{ .array = .{ .strings = &.{} } } },
        .{ .key = "tokenizer.ggml.token_type", .value = .{ .array = .{ .i32s = &types } } },
        .{ .key = "tokenizer.ggml.eos_token_id", .value = .{ .u32_val = 1 } },
    };
    const vocab = try buildFromParsed(arena, parsedWithMetadata(&required));
    try std.testing.expectEqual(0, vocab.bos_id);

    const wrong_type = required ++ [_]parser.MetadataPair{
        .{ .key = "tokenizer.ggml.bos_token_id", .value = .{ .string = "1" } },
    };
    try std.testing.expectError(
        error.WrongTokenizerKeyType,
        buildFromParsed(arena, parsedWithMetadata(&wrong_type)),
    );

    const negative = required ++ [_]parser.MetadataPair{
        .{ .key = "tokenizer.ggml.bos_token_id", .value = .{ .i32_val = -1 } },
    };
    try std.testing.expectError(
        error.BadTokenizerValue,
        buildFromParsed(arena, parsedWithMetadata(&negative)),
    );
}

test "buildFromParsed: a token type count or a special id that the token list breaks refuses" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const two = [_][]const u8{ "a", "b" };
    const Case = struct {
        tokens: []const []const u8 = &two,
        types: []const i32 = &.{ 1, 1 },
        bos: u32 = 0,
        eos: u32 = 1,
        want: ?Error = null,
    };
    const bad = error.BadTokenizerValue;
    const cases = [_]Case{
        .{},
        .{ .types = &.{1}, .want = bad },
        .{ .types = &.{ 1, 1, 1 }, .want = bad },
        .{ .tokens = &.{}, .types = &.{}, .eos = 0, .want = bad },
        .{ .bos = 2, .want = bad },
        .{ .eos = 2, .want = bad },
    };
    for (cases, 0..) |c, i| {
        const metadata = [_]parser.MetadataPair{
            .{ .key = "tokenizer.ggml.tokens", .value = .{ .array = .{ .strings = c.tokens } } },
            .{ .key = "tokenizer.ggml.merges", .value = .{ .array = .{ .strings = &.{} } } },
            .{ .key = "tokenizer.ggml.token_type", .value = .{ .array = .{ .i32s = c.types } } },
            .{ .key = "tokenizer.ggml.bos_token_id", .value = .{ .u32_val = c.bos } },
            .{ .key = "tokenizer.ggml.eos_token_id", .value = .{ .u32_val = c.eos } },
        };
        const got = buildFromParsed(arena, parsedWithMetadata(&metadata));
        errdefer std.debug.print("vocab case {d} failed\n", .{i});
        if (c.want) |err| {
            try std.testing.expectError(err, got);
        } else {
            _ = try got;
        }
    }
}

test "addBos: llama3-family pre defaults true, an explicit key wins, else false" {
    const llama = [_]parser.MetadataPair{
        .{ .key = "tokenizer.ggml.pre", .value = .{ .string = "llama-bpe" } },
    };
    const llama_explicit_off = [_]parser.MetadataPair{
        .{ .key = "tokenizer.ggml.pre", .value = .{ .string = "llama-bpe" } },
        .{ .key = "tokenizer.ggml.add_bos_token", .value = .{ .bool_val = false } },
    };
    const smollm = [_]parser.MetadataPair{
        .{ .key = "tokenizer.ggml.pre", .value = .{ .string = "smollm" } },
    };
    try std.testing.expect(addBos(parsedWithMetadata(&llama)));
    try std.testing.expect(!addBos(parsedWithMetadata(&llama_explicit_off)));
    try std.testing.expect(!addBos(parsedWithMetadata(&smollm)));
    try std.testing.expect(!addBos(parsedWithMetadata(&.{})));
}
