//! The KV price of a model card, one entry per layer. The planner splits this vector at a
//! placement, and the RAM quote prices every serve arm from it. It reads the card alone and touches
//! no device. A CPU build prices the same figures as a fused-GPU build. Every caller reads the
//! price here, so no runner states a second price.
//!
//! Layer kinds. A Llama layer holds the same KV and no recurrent state. A Qwen3.5 full-attention
//! layer charges KV rows a token. A GDN layer holds one recurrent state and one conv ring a slot,
//! which stay f32 under any KV dtype. One product over the layer count cannot price this mix.

const std = @import("std");
const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const loader = @import("../io/gguf/loader.zig");

// -- api --------------------------------------------------------------------

/// The per-layer price of the model the card states: Qwen3.5 with GDN layers, else Llama.
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

/// A family states its price twice: as this vector and as the scalar of `kv_budget.costModel`.
/// The two must agree, so this check catches a mistake at the source.
fn validate(vector: kv_budget.KvCostVector, card: loader.ModelCard, kv_dtype: dtype.KvDtype) void {
    const whole = vector.total();
    const scalar = kv_budget.costModel(card.kvShape(), kv_dtype);
    std.debug.assert(vector.n_layers == card.n_layers);
    std.debug.assert(whole.kv_bytes_per_token == scalar.kv_bytes_per_token);
    std.debug.assert(whole.fixed_state_bytes_per_slot == scalar.fixed_state_bytes_per_slot);
}

// -- tests ------------------------------------------------------------------

/// A Llama-3.1-8B-shaped card: 32 layers of 8 KV heads x 128 head_dim.
fn llamaCard() loader.ModelCard {
    return .{
        .arch = .llama,
        .architecture = "llama",
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .embedding_length = 4096,
        .feed_forward_length = 14336,
        .context_length = 131072,
        .vocab_size = 128256,
        .rope_freq_base = 500000.0,
        .rope_dim_count = 128,
        .rms_norm_eps = 1e-5,
        .bos_token_id = 128000,
        .eos_token_id = 128009,
        .chat_template = "",
    };
}

/// A Qwen3.5-2B-shaped card: 24 layers at interval 4, so 18 GDN layers and 6
/// full-attention layers.
fn qwenCard() loader.ModelCard {
    var card = llamaCard();
    card.arch = .qwen35;
    card.architecture = "qwen3.5";
    card.n_layers = 24;
    card.n_heads = 16;
    card.n_kv_heads = 2;
    card.head_dim = 256;
    card.gdn = .{
        .conv_kernel = 4,
        .state_size = 128,
        .group_count = 16,
        .inner_size = 2048,
        .time_step_rank = 16,
        .full_attention_interval = 4,
    };
    return card;
}

test "costVector: the 8B card charges 4096 f16 bytes a token on every layer" {
    const vector = costVector(llamaCard(), .f16);
    try std.testing.expectEqual(@as(u32, 32), vector.n_layers);
    for (vector.layers()) |entry| {
        try std.testing.expectEqual(@as(u32, 4_096), entry.kv_bytes_per_token);
        try std.testing.expectEqual(@as(u32, 0), entry.fixed_bytes_per_slot);
    }

    // f32 doubles the token price.
    try std.testing.expectEqual(
        @as(u64, 2 * 32 * 4_096),
        costVector(llamaCard(), .f32).total().kv_bytes_per_token,
    );
}

test "costVector: the 2B card prices attention and GDN apart, and the dtype moves only tokens" {
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const vector = costVector(qwenCard(), kv_dtype);
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
