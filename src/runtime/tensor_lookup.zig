//! Tensor-by-name lookup over a `loader.TensorRef` list. Every forward pass
//! shares it to resolve the weights of a loaded GGUF model.
//!
//! One home keeps the four runners (CPU Llama, CPU Qwen, GPU Llama, GPU Qwen)
//! on one found/not-found rule. An absence is either an optional (`find`) or an
//! `error.MissingTensor` (`findRequired`). The call site picks the form it
//! needs, and the runner that holds the call site does not change the answer.

const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const limits = @import("../core/limits.zig");
const loader = @import("../io/gguf/loader.zig");

// -- types ------------------------------------------------------------------

pub const Error = error{MissingTensor};

// -- api --------------------------------------------------------------------

/// The tensor named `name`, or null. This is the optional primitive that
/// `findRequired` wraps. A call site with a legitimate absence (a tied lm_head,
/// optional rope factors) uses it directly.
pub fn find(refs: []const loader.TensorRef, name: []const u8) ?*const Tensor {
    for (refs) |*r| {
        if (std.mem.eql(u8, r.name, name)) return &r.tensor;
    }

    return null;
}

/// The tensor named `name`, or `error.MissingTensor`. A forward pass uses this
/// form for a weight it cannot run without.
pub fn findRequired(refs: []const loader.TensorRef, name: []const u8) Error!*const Tensor {
    return find(refs, name) orelse error.MissingTensor;
}

/// `findRequired` with the name formatted into `buf` (e.g. `"blk.{d}.attn_q.weight"`).
/// This function treats a `bufPrint` overflow as a miss: a name that cannot fit
/// cannot exist.
pub fn findByFmt(
    refs: []const loader.TensorRef,
    buf: []u8,
    comptime fmt: []const u8,
    args: anytype,
) Error!*const Tensor {
    const name = std.fmt.bufPrint(buf, fmt, args) catch return error.MissingTensor;
    return findRequired(refs, name);
}

/// The weight bytes each transformer block holds, keyed by the `blk.{i}.`
/// prefix every per-layer GGUF tensor carries. The non-layer tensors
/// (`token_embd`, `output`, `output_norm`) sit outside a per-layer split, so
/// this function leaves them out. `out` holds one entry per layer, and the
/// result aliases its first `n_layers` entries.
pub fn layerWeightBytes(refs: []const loader.TensorRef, n_layers: u32, out: []u64) []u64 {
    std.debug.assert(n_layers >= 1);
    std.debug.assert(n_layers <= limits.max_layers);
    std.debug.assert(out.len >= n_layers);

    @memset(out[0..n_layers], 0);
    for (refs) |*r| {
        const idx = blockIndex(r.name) orelse continue;
        // The loader validates the declared block count, so a higher index
        // means a malformed model. The release build skips it rather than
        // writing past the vector.
        std.debug.assert(idx < n_layers);
        if (idx >= n_layers) continue;

        out[idx] += r.tensor.data.len;
    }

    return out[0..n_layers];
}

/// The weight bytes outside the per-layer split: the final norm plus the
/// lm_head. A model that ties its lm_head to the embedding table ships no
/// `output.weight`. The un-embedding then reads `token_embd.weight`, so the
/// tied case charges the embedding table here. An absent tensor charges
/// nothing, the way `layerWeightBytes` charges nothing for an empty layer.
pub fn headWeightBytes(refs: []const loader.TensorRef) u64 {
    var total: u64 = 0;
    if (find(refs, "output_norm.weight")) |t| {
        total += t.data.len;
    }

    const head = find(refs, "output.weight") orelse find(refs, "token_embd.weight");
    if (head) |t| {
        total += t.data.len;
    }

    return total;
}

/// Reinterpret the bytes of an f32 tensor as `[]const f32`. The loader refuses
/// a tensor whose start breaks the alignment of its element type, with
/// `error.TensorDataUnaligned`. So the `@alignCast` holds for every loaded
/// tensor.
pub fn fp32View(t: Tensor) []const f32 {
    std.debug.assert(t.dtype == .f32);

    return @alignCast(std.mem.bytesAsSlice(f32, t.data));
}

// -- helpers ----------------------------------------------------------------

/// The block index a tensor name carries, or null when the name names no block.
/// `blk.13.attn_q.weight` yields 13.
fn blockIndex(name: []const u8) ?u32 {
    const prefix = "blk.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;

    const rest = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(u32, rest[0..dot], 10) catch null;
}

// -- tests ------------------------------------------------------------------

test "find / findRequired / findByFmt resolve and report misses" {
    const data align(@alignOf(f32)) = [_]u8{0} ** 16;
    const t: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 4, 0, 0, 0 },
        .byte_stride = .{ 16, 0, 0, 0 },
        .data = &data,
    };
    const refs = [_]loader.TensorRef{
        .{ .name = "blk.0.attn_q.weight", .tensor = t },
        .{ .name = "token_embd.weight", .tensor = t },
    };
    try std.testing.expect(find(&refs, "token_embd.weight") != null);
    try std.testing.expect(find(&refs, "absent") == null);
    try std.testing.expectError(error.MissingTensor, findRequired(&refs, "absent"));

    var buf: [64]u8 = undefined;
    _ = try findByFmt(&refs, &buf, "blk.{d}.attn_q.weight", .{0});
    try std.testing.expectError(
        error.MissingTensor,
        findByFmt(&refs, &buf, "blk.{d}.attn_q.weight", .{9}),
    );
    try std.testing.expectEqual(@as(usize, 4), fp32View(t).len);
}

test "layerWeightBytes: sums each block and leaves the non-layer tensors out" {
    const data align(@alignOf(f32)) = [_]u8{0} ** 16;
    const t: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 4, 0, 0, 0 },
        .byte_stride = .{ 16, 0, 0, 0 },
        .data = &data,
    };
    const refs = [_]loader.TensorRef{
        .{ .name = "token_embd.weight", .tensor = t },
        .{ .name = "blk.0.attn_q.weight", .tensor = t },
        .{ .name = "blk.0.ffn_up.weight", .tensor = t },
        .{ .name = "blk.1.attn_q.weight", .tensor = t },
        .{ .name = "blk.not-a-number.weight", .tensor = t },
        .{ .name = "output_norm.weight", .tensor = t },
        .{ .name = "output.weight", .tensor = t },
    };
    var table: [limits.max_layers]u64 = undefined;

    const two = layerWeightBytes(&refs, 2, &table);
    try std.testing.expectEqual(@as(usize, 2), two.len);
    try std.testing.expectEqual(@as(u64, 32), two[0]);
    try std.testing.expectEqual(@as(u64, 16), two[1]);

    // A declared layer with no tensor of its own weighs nothing.
    const three = layerWeightBytes(&refs, 3, &table);
    try std.testing.expectEqual(@as(u64, 0), three[2]);
}

test "headWeightBytes: the final norm plus the lm_head, tied and untied" {
    const embed_data align(@alignOf(f32)) = [_]u8{0} ** 64;
    const norm_data align(@alignOf(f32)) = [_]u8{0} ** 16;
    const head_data align(@alignOf(f32)) = [_]u8{0} ** 32;
    const embed: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 16, 0, 0, 0 },
        .byte_stride = .{ 64, 0, 0, 0 },
        .data = &embed_data,
    };
    const norm: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 4, 0, 0, 0 },
        .byte_stride = .{ 16, 0, 0, 0 },
        .data = &norm_data,
    };
    const head: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 8, 0, 0, 0 },
        .byte_stride = .{ 32, 0, 0, 0 },
        .data = &head_data,
    };

    // Tied: no `output.weight`, so the embedding table pays for the lm_head.
    const tied = [_]loader.TensorRef{
        .{ .name = "token_embd.weight", .tensor = embed },
        .{ .name = "output_norm.weight", .tensor = norm },
        .{ .name = "blk.0.attn_q.weight", .tensor = head },
    };
    try std.testing.expectEqual(@as(u64, 80), headWeightBytes(&tied));

    // Untied: `output.weight` wins, and the per-layer tensors stay out.
    const untied = tied ++ [_]loader.TensorRef{
        .{ .name = "output.weight", .tensor = head },
    };
    try std.testing.expectEqual(@as(u64, 48), headWeightBytes(&untied));
}
