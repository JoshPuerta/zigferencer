//! Tensor-by-name lookup over a `loader.TensorRef` list. Every forward pass shares it to resolve
//! the weights of a loaded GGUF model.
//!
//! Comptime. `roleName` and `findRole` take the layer family at comptime, because the family names
//! each weight role with a comptime format, so a bad format fails the build.
//!
//! Absence. One home keeps the four runners (CPU Llama, CPU Qwen, GPU Llama, GPU Qwen) on one
//! found/not-found rule. An absence is either an optional (`find`) or an `error.MissingTensor`
//! (`findRequired`). The call site picks the form it needs, and the runner does not change the
//! answer.
//!
//! Weight bytes. `layerWeightBytes` keys each block by its `blk.{i}.` prefix and leaves the
//! non-layer tensors out. `headWeightBytes` sums the final norm plus the lm_head. A model that
//! ties its lm_head to the embedding ships no `output.weight`, so the embedding table covers the
//! head. An absent tensor adds nothing.

const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const limits = @import("../core/limits.zig");
const loader = @import("../io/gguf/loader.zig");
const Weight = @import("layer_kernels.zig").Weight;

// -- constants --------------------------------------------------------------

/// Fits every `blk.{d}.*` tensor name of both families at every layer of `limits.max_layers`.
pub const role_name_bytes_max = 80;

// -- types ------------------------------------------------------------------

pub const Error = error{MissingTensor};

// -- api --------------------------------------------------------------------

/// The tensor named `name`, or null. A call site with a legitimate absence uses it directly.
pub fn find(refs: []const loader.TensorRef, name: []const u8) ?*const Tensor {
    const ref = findRef(refs, name) orelse return null;
    return &ref.tensor;
}

/// The entry named `name`, or null. Its name outlives a formatted lookup key.
pub fn findRef(refs: []const loader.TensorRef, name: []const u8) ?*const loader.TensorRef {
    for (refs) |*r| {
        if (std.mem.eql(u8, r.name, name)) return r;
    }

    return null;
}

/// The tensor named `name`, or `error.MissingTensor`. Use it for a weight the pass needs.
pub fn findRequired(refs: []const loader.TensorRef, name: []const u8) Error!*const Tensor {
    return find(refs, name) orelse error.MissingTensor;
}

/// The GGUF name of `role` in the model layer `layer`, written into `buf`. Every role of
/// `Family.layerWeights` has a name.
pub fn roleName(
    comptime Family: type,
    buf: *[role_name_bytes_max]u8,
    role: Weight,
    layer: u32,
) []const u8 {
    std.debug.assert(layer < limits.max_layers);

    switch (role) {
        inline else => |r| {
            if (comptime Family.tensorName(r)) |fmt| {
                // The longest format plus the widest layer index fits the buffer.
                return std.fmt.bufPrint(buf, fmt, .{layer}) catch unreachable;
            }

            unreachable; // Every role of `layerWeights` has a GGUF name.
        },
    }
}

/// The entry of `role` in the model layer `layer`, or `error.MissingTensor`.
pub fn findRole(
    comptime Family: type,
    refs: []const loader.TensorRef,
    role: Weight,
    layer: u32,
) Error!*const loader.TensorRef {
    var buf: [role_name_bytes_max]u8 = undefined;
    return findRef(refs, roleName(Family, &buf, role, layer)) orelse error.MissingTensor;
}

/// The weight bytes of each transformer block. `out` holds one entry per layer, and the result
/// aliases its first `n_layers` entries.
pub fn layerWeightBytes(refs: []const loader.TensorRef, n_layers: u32, out: []u64) []u64 {
    std.debug.assert(n_layers >= 1);
    std.debug.assert(n_layers <= limits.max_layers);
    std.debug.assert(out.len >= n_layers);

    @memset(out[0..n_layers], 0);
    for (refs) |*r| {
        const idx = blockIndex(r.name) orelse continue;
        // The index comes from a file name, so the walk skips a block past the declared layers.
        if (idx >= n_layers) continue;

        out[idx] += r.tensor.data.len;
    }

    return out[0..n_layers];
}

/// The weight bytes outside the per-layer split: the final norm plus the lm_head.
pub fn headWeightBytes(refs: []const loader.TensorRef) u64 {
    var total: u64 = 0;
    if (find(refs, loader.output_norm_tensor)) |t| {
        total += t.data.len;
    }

    const head = find(refs, loader.head_tensor) orelse find(refs, loader.embed_tensor);
    if (head) |t| {
        total += t.data.len;
    }

    return total;
}

/// The block index a tensor name carries, or null when the name names no block.
/// `blk.13.attn_q.weight` yields 13.
pub fn blockIndex(name: []const u8) ?u32 {
    const prefix = "blk.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;

    const rest = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(u32, rest[0..dot], 10) catch null;
}

/// Reinterpret the bytes of an f32 tensor as `[]const f32`. The loader refuses an unaligned
/// tensor with `error.TensorDataUnaligned`, so the `@alignCast` holds for every loaded tensor.
pub fn fp32View(t: Tensor) []const f32 {
    std.debug.assert(t.dtype == .f32);

    return @alignCast(std.mem.bytesAsSlice(f32, t.data));
}

// -- tests ------------------------------------------------------------------

test "find / findRequired / findRole resolve and report misses" {
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

    const llama = @import("llama_layer.zig");
    const ref = try findRole(llama, &refs, .attn_q, 0);
    try std.testing.expectEqualStrings("blk.0.attn_q.weight", ref.name);
    try std.testing.expectError(error.MissingTensor, findRole(llama, &refs, .attn_q, 9));
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

    // The walk skips a block past the declared layers, and the sum of block 0 holds.
    const one = layerWeightBytes(&refs, 1, &table);
    try std.testing.expectEqual(@as(usize, 1), one.len);
    try std.testing.expectEqual(@as(u64, 32), one[0]);
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

    // Tied: no `output.weight`, so the embedding table covers the lm_head.
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
