//! The vocabulary values a serve frontend needs: the token pieces of its codec and the think
//! tags of the scheduler.
//!
//! Think tags. A vocab that breaks a rule of `thinkTagsFrom` refuses the boot, which keeps model
//! data away from the scheduler asserts. Each tag is one id. The forced ids start with the close,
//! number at most `think_force_ids_max`, and hold no end-of-sequence id.

const std = @import("std");

const bpe = @import("../../core/tokenizer/bpe.zig");
const chat = @import("../../core/chat.zig");
const limits = @import("../../core/limits.zig");
const parser = @import("../../io/gguf/parser.zig");
const vocab_mod = @import("../../core/tokenizer/vocab.zig");
const scheduler_mod = @import("../../serve/scheduler.zig");

const log = std.log.scoped(.serve);

// -- types ------------------------------------------------------------------

/// The ingredients a frontend needs to build its own codec.
pub const CodecParts = struct {
    vocab: *const vocab_mod.Vocab,
    pieces: []const []const u8,
    /// The registry kind with the `--thinking` flag applied.
    kind: chat.Kind,
    eos_id: u32,
    /// The think tags as ids, for a kind that has a thinking mode.
    think_ids: ?chat.ThinkIds = null,
};

/// The think tags as the scheduler reads them: the two tags and the ids that a full budget forces.
pub const ThinkTags = struct {
    ids: chat.ThinkIds,
    force: [limits.think_force_ids_max]u32,
    force_len: u8,
};

// -- api --------------------------------------------------------------------

/// Decode the displayable bytes of every id once at startup. A piece past the serve cap fails the
/// boot, never a silent truncation. A control token gets the empty piece. A user-defined token
/// keeps its text.
pub fn buildPieces(arena: std.mem.Allocator, v: *const vocab_mod.Vocab) ![]const []const u8 {
    std.debug.assert(v.is_control.len == v.tokens.len);

    const pieces = try arena.alloc([]const u8, v.tokens.len);
    for (pieces, v.is_control, 0..) |*piece, control, id| {
        if (control) {
            std.debug.assert(v.is_special[id]);
            piece.* = "";
            continue;
        }

        piece.* = try bpe.decodeAlloc(v, arena, &[_]u32{@intCast(id)});
        if (piece.len > limits.max_token_text_bytes) {
            log.err("token {d} decodes to {d} bytes (serve cap {d})", .{
                id, piece.len, limits.max_token_text_bytes,
            });
            return error.TokenPieceTooLong;
        }
    }

    log.debug("pieces: {d} tokens materialized", .{pieces.len});
    return pieces;
}

/// The think tags of `kind`, or null for a kind with no thinking mode.
pub fn thinkTagsOf(
    arena: std.mem.Allocator,
    v: *const vocab_mod.Vocab,
    kind: chat.Kind,
) !?ThinkTags {
    const text = chat.thinkText(kind) orelse return null;
    return try thinkTags(arena, v, text);
}

/// The scheduler think config of `kind`. A kind with no tags gets the empty config, so the
/// scheduler refuses every open turn and budget.
pub fn schedulerThink(tags: ?ThinkTags, kind: chat.Kind, budget: u32) scheduler_mod.Think {
    const t = tags orelse return .{};
    return .{
        .open_default = chat.thinkOpen(kind),
        .open_id = t.ids.open,
        .close_id = t.ids.close,
        .budget = budget,
        .force = t.force,
        .force_len = t.force_len,
    };
}

pub fn codecParts(
    v: *const vocab_mod.Vocab,
    pieces: []const []const u8,
    kind: chat.Kind,
    think_tags: ?ThinkTags,
) CodecParts {
    return .{
        .vocab = v,
        .pieces = pieces,
        .kind = kind,
        .eos_id = v.eos_id,
        .think_ids = if (think_tags) |t| t.ids else null,
    };
}

// -- helpers ----------------------------------------------------------------

/// Encode the think tags of a model with a thinking mode.
fn thinkTags(
    arena: std.mem.Allocator,
    v: *const vocab_mod.Vocab,
    text: chat.ThinkText,
) !ThinkTags {
    const open = try bpe.encodeAlloc(v, arena, text.open, false);
    const close = try bpe.encodeAlloc(v, arena, text.close, false);

    // The forced ids are the close and the separator that the template writes after a block.
    const force_text = try std.mem.concat(arena, u8, &.{ text.close, text.separator });
    const force = try bpe.encodeAlloc(v, arena, force_text, false);

    const tags = thinkTagsFrom(open, close, force, v.eos_id) catch |err| {
        log.err("think tags: {s} encodes to {any}, {s} to {any}, the forced close to {any}", .{
            text.open, open, text.close, close, force,
        });
        log.err("think tags: each tag must be one id. " ++
            "The forced close must start with {s} and hold at most {d} ids, " ++
            "none of them end-of-sequence {d}", .{
            text.close, limits.think_force_ids_max, v.eos_id,
        });
        return err;
    };
    log.debug("think tags: open {d}, close {d}, forced {any}", .{
        tags.ids.open, tags.ids.close, tags.force[0..tags.force_len],
    });
    return tags;
}

/// The rules that the scheduler asserts at init, checked on the encoded ids.
fn thinkTagsFrom(
    open: []const u32,
    close: []const u32,
    force: []const u32,
    eos_id: u32,
) error{ThinkTagsUnusable}!ThinkTags {
    if (open.len != 1 or close.len != 1) return error.ThinkTagsUnusable;
    if (force.len == 0 or force.len > limits.think_force_ids_max) return error.ThinkTagsUnusable;
    if (force[0] != close[0]) return error.ThinkTagsUnusable;
    if (close[0] == eos_id) return error.ThinkTagsUnusable;

    for (force) |id| {
        if (id == eos_id) return error.ThinkTagsUnusable;
    }

    var tags: ThinkTags = .{
        .ids = .{ .open = open[0], .close = close[0] },
        .force = @splat(scheduler_mod.no_token),
        .force_len = @intCast(force.len),
    };
    @memcpy(tags.force[0..force.len], force);
    std.debug.assert(tags.force[0] == tags.ids.close);
    return tags;
}

// -- tests ------------------------------------------------------------------

test "thinkTagsFrom: one id per tag, the close first, and no end-of-sequence id" {
    // The Qwen3.5 ids: the two tags, and the `\n\n` separator of the forced close.
    const open = [_]u32{248068};
    const close = [_]u32{248069};
    const sep: u32 = 271;
    const eos: u32 = 248046;
    const tags = try thinkTagsFrom(&open, &close, &.{ 248069, sep }, eos);
    try std.testing.expectEqual(@as(u32, 248068), tags.ids.open);
    try std.testing.expectEqual(@as(u32, 248069), tags.ids.close);
    try std.testing.expectEqualSlices(u32, &.{ 248069, sep }, tags.force[0..tags.force_len]);

    // Each refusal below breaks one rule.
    const bad = error.ThinkTagsUnusable;
    try std.testing.expectError(bad, thinkTagsFrom(&.{ 27, 28 }, &close, &.{ 248069, sep }, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &.{ 27, 28 }, &.{ 27, 28 }, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &close, &.{ sep, 248069 }, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &close, &.{}, eos));
    try std.testing.expectError(
        bad,
        thinkTagsFrom(&open, &close, &.{ 248069, sep, sep, sep, sep }, eos),
    );
    try std.testing.expectError(bad, thinkTagsFrom(&open, &.{eos}, &.{eos}, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &close, &.{ 248069, eos }, eos));
    // The bound itself is legal.
    const widest = try thinkTagsFrom(&open, &close, &.{ 248069, sep, sep, sep }, eos);
    try std.testing.expectEqual(limits.think_force_ids_max, widest.force_len);
}

test "buildPieces: a control token has the empty piece, and every other token has its text" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const tokens = [_][]const u8{ "a", "<|end|>", "<tool>", "b", "<|ctl|>" };
    // The GGUF token types: 1 normal, 3 control, 4 user-defined.
    const types = [_]i32{ 1, 3, 4, 1, 3 };
    const metadata = [_]parser.MetadataPair{
        .{ .key = "tokenizer.ggml.tokens", .value = .{ .array = .{ .strings = &tokens } } },
        .{ .key = "tokenizer.ggml.merges", .value = .{ .array = .{ .strings = &.{} } } },
        .{ .key = "tokenizer.ggml.token_type", .value = .{ .array = .{ .i32s = &types } } },
        .{ .key = "tokenizer.ggml.eos_token_id", .value = .{ .u32_val = 1 } },
    };
    const v = try vocab_mod.buildFromParsed(arena, .{
        .header = undefined,
        .metadata = &metadata,
        .tensors = &.{},
        .data_offset = 0,
        .alignment = 1,
    });
    const pieces = try buildPieces(arena, &v);
    try std.testing.expectEqual(tokens.len, pieces.len);
    try std.testing.expectEqualStrings("", pieces[1]);
    try std.testing.expectEqualStrings("<tool>", pieces[2]);

    // The piece is empty exactly when the token is a control token.
    for (pieces, types, tokens) |piece, token_type, token| {
        try std.testing.expectEqual(token_type == 3, piece.len == 0);
        if (token_type != 3) {
            try std.testing.expectEqualStrings(token, piece);
        }
    }
}
