//! The KV size of a model card, one entry per layer. The layer split planner cuts this vector at
//! the split, and the RAM plan sizes every serve placement from it. It reads the card alone and
//! touches no device. A CPU build sizes the same figures as a fused-GPU build. Every caller reads
//! the size here, so no runner states a second size.
//!
//! Layer kinds. A Llama layer holds the same KV and no recurrent state. A Qwen3.5 full-attention
//! layer holds KV rows a token. A GDN layer holds one recurrent state and one conv ring a slot,
//! which stay f32 under any KV dtype. One product over the layer count cannot size this mix.

const std = @import("std");
const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const loader = @import("../io/gguf/loader.zig");

// -- api --------------------------------------------------------------------

/// The per-layer size of the model the card states: Qwen3.5 with GDN layers, else Llama.
pub fn costVector(card: loader.ModelCard, kv_dtype: dtype.KvDtype) kv_budget.KvCostVector {
    return if (card.gdn != null) qwenCostVector(
        card,
        kv_dtype,
    ) else llamaCostVector(card, kv_dtype);
}

/// The Llama vector is uniform, and its fixed column is zero.
pub fn llamaCostVector(card: loader.ModelCard, kv_dtype: dtype.KvDtype) kv_budget.KvCostVector {
    std.debug.assert(card.gdn == null);

    const vector = kv_budget.KvCostVector.uniform(
        card.n_layers,
        @as(u64, 2) * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim),
    );
    validate(vector, card, kv_dtype);
    return vector;
}

/// The Qwen3.5 vector: KV bytes on full-attention layers, fixed state bytes on GDN layers.
pub fn qwenCostVector(card: loader.ModelCard, kv_dtype: dtype.KvDtype) kv_budget.KvCostVector {
    const g = card.gdn orelse return kv_budget.KvCostVector.uniform(card.n_layers, 0);
    const key_dim: u64 = @as(u64, g.group_count) * g.state_size;
    const value_dim: u64 = @as(u64, g.time_step_rank) * g.state_size;
    const state_elems: u64 = @as(u64, g.time_step_rank) * g.state_size * g.state_size;
    const conv_elems: u64 = @as(u64, g.conv_kernel - 1) * (2 * key_dim + value_dim);
    const attn_bytes: u64 = @as(u64, 2) * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
    const gdn_bytes: u64 = (state_elems + conv_elems) * @sizeOf(f32);
    var vector = kv_budget.KvCostVector.uniform(card.n_layers, 0);
    var li: u32 = 0;
    while (li < card.n_layers) : (li += 1) {
        if (g.isFullAttention(li)) {
            vector.set(li, attn_bytes, 0);
        } else {
            vector.set(li, 0, gdn_bytes);
        }
    }

    validate(vector, card, kv_dtype);
    return vector;
}

// -- helpers ----------------------------------------------------------------

/// A family states its size twice: as this vector and as the scalar of `kv_budget.costModel`.
/// The two must agree, so this check catches a mistake at the source.
fn validate(vector: kv_budget.KvCostVector, card: loader.ModelCard, kv_dtype: dtype.KvDtype) void {
    const whole = vector.total();
    const scalar = kv_budget.costModel(card.kvShape(), kv_dtype);
    std.debug.assert(vector.n_layers == card.n_layers);
    std.debug.assert(whole.kv_bytes_per_token == scalar.kv_bytes_per_token);
    std.debug.assert(whole.fixed_state_bytes_per_slot == scalar.fixed_state_bytes_per_slot);
}

// -- tests ------------------------------------------------------------------

const testCard = @import("test_fakes.zig").testCard;

test "costVector: the 8B card counts 4096 f16 bytes a token on every layer" {
    const vector = costVector(testCard(.llama), .f16);
    try std.testing.expectEqual(@as(u32, 32), vector.n_layers);
    for (vector.layers()) |entry| {
        try std.testing.expectEqual(@as(u32, 4_096), entry.kv_bytes_per_token);
        try std.testing.expectEqual(@as(u32, 0), entry.fixed_bytes_per_slot);
    }

    // f32 doubles the token size.
    try std.testing.expectEqual(
        @as(u64, 2 * 32 * 4_096),
        costVector(testCard(.llama), .f32).total().kv_bytes_per_token,
    );
}

test "costVector: the 2B card sizes attention and GDN apart, and the dtype moves only tokens" {
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const vector = costVector(testCard(.qwen35), kv_dtype);
        const attn_bytes: u64 = 2 * kv_dtype.rowBytes(2 * 256);
        for (vector.layers(), 0..) |entry, i| {
            if ((i + 1) % 4 == 0) {
                try std.testing.expectEqual(attn_bytes, entry.kv_bytes_per_token);
                try std.testing.expectEqual(@as(u32, 0), entry.fixed_bytes_per_slot);
            } else {
                try std.testing.expectEqual(@as(u32, 0), entry.kv_bytes_per_token);
                try std.testing.expectEqual(@as(u32, 1_122_304), entry.fixed_bytes_per_slot);
            }
        }
    }
}
