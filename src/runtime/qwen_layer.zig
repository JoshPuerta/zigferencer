//! The Qwen3.5 decoder layer: its forward through a layer kernel set, and the rules of the family
//! that a runner reads to hold and bind the layer.
//!
//! Comptime. `forward` takes the kernel set type and checks it with `assertLayerKernels`.
//! `tensorName` takes the role at comptime, because `tensor_lookup.roleName` takes a comptime
//! format.
//!
//! Mixer. A layer is full attention or Gated DeltaNet (GDN). The kind is a property of the model,
//! so every rule takes the model index of the layer, and `forward` passes it to each call.
//!
//! KV index. A runner of the layers from `first` holds KV regions for its full-attention layers
//! only. The KV index of a full layer counts the full layers in `[first, layer)`. A model index
//! used in place of the KV index reads the KV of another layer, and no bounds check catches it.
//!
//! Tests. The logging kernel set of `layer_kernels` records the calls of `forward`. The tests check
//! the op order of both layer kinds, and that each op occurs at most once in a layer, which the
//! binding table needs.

const std = @import("std");
const loader = @import("../io/gguf/loader.zig");
const layer_kernels = @import("layer_kernels.zig");
const cpu_profile = @import("cpu_profile.zig");
const qwen_gpu_guard = @import("qwen_gpu_guard.zig");

const Activation = layer_kernels.Activation;
const GpuRunner = @import("../core/span.zig").GpuRunner;
const LayerOp = layer_kernels.LayerOp;
const Weight = layer_kernels.Weight;
const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

/// The architecture of the family. A runner refuses a card of another one.
pub const arch: loader.Arch = .qwen35;

/// A GDN layer holds recurrent state in each slot, so the session surface needs the slot hooks.
pub const has_recurrent_state = true;

/// Qwen3.5 rotates at the plain frequencies, so it names no rope factor tensor.
pub const rope_freqs: ?[]const u8 = null;

/// The op categories of the CPU profile of the family.
pub const profile_family: cpu_profile.Family = .qwen;

/// The weight quants of the fused runner.
pub const fused_quants: []const layer_kernels.WeightQuant = &.{.q8_0};

/// From `gemm_min_batch` rows, the fused runner records the tiled GEMM.
pub const fused_gemm = true;

/// The rows of one fused step. This count sizes every per-row scratch buffer of the fused
/// runner, and it bounds the time of the one submit of the step.
pub const fused_rows_max: u32 = 128;

/// The GPU records of the fused runner: the whole model, and the GPU layers of a hybrid.
pub const fused_gpu_runner: GpuRunner = .qwen_fused;
pub const hybrid_gpu_runner: GpuRunner = .qwen_hybrid;

/// The weight roles of a full-attention layer.
const full_weights = [_]Weight{
    .attn_norm,
    .attn_q,
    .attn_k,
    .attn_v,
    .attn_q_norm,
    .attn_k_norm,
    .attn_o,
    .ffn_norm,
    .ffn_gate,
    .ffn_up,
    .ffn_down,
};

/// The weight roles of a GDN layer.
const gdn_weights = [_]Weight{
    .attn_norm,
    .ssm_qkv,
    .ssm_z,
    .ssm_alpha,
    .ssm_beta,
    .ssm_conv1d,
    .ssm_a,
    .ssm_dt,
    .ssm_norm,
    .ssm_out,
    .ffn_norm,
    .ffn_gate,
    .ffn_up,
    .ffn_down,
};

// -- api --------------------------------------------------------------------

/// One Qwen3.5 decoder layer over the step rows that `kernels` holds. `layer` is the model index,
/// because the mixer of a layer is a property of the model.
pub fn forward(comptime Kernels: type, kernels: *Kernels, layer: u32) void {
    comptime layer_kernels.assertLayerKernels(Kernels);
    assert(layer < kernels.card.n_layers);

    // Norm the layer input, then run the mixer of this layer.
    kernels.norm(layer, .attn_norm, .attn_norm);
    if (isFullAttention(kernels.card, layer)) {
        kernels.project(layer, .attn_proj, .x_norm, &.{
            .{ .weight = .attn_q, .out = .q_gate },
            .{ .weight = .attn_k, .out = .k },
            .{ .weight = .attn_v, .out = .v },
        });
        kernels.qkNorm(layer);
        kernels.rope(layer, .neox);
        kernels.kvCommit(layer);
        kernels.attention(layer);
        kernels.gate(layer);
        kernels.project(layer, .o_proj, .attn_out, &.{.{ .weight = .attn_o, .out = .mix_out }});
    } else {
        kernels.project(layer, .gdn_proj, .x_norm, &.{
            .{ .weight = .ssm_qkv, .out = .qkv },
            .{ .weight = .ssm_z, .out = .z },
            .{ .weight = .ssm_alpha, .out = .alpha },
            .{ .weight = .ssm_beta, .out = .beta },
        });
        kernels.gdn(layer);
        kernels.project(layer, .gdn_out, .mix_o, &.{.{ .weight = .ssm_out, .out = .mix_out }});
    }

    // Add the mixer residual, then run the FFN and add its residual.
    kernels.residual(layer, .mixer_residual, .mix_out);
    kernels.norm(layer, .ffn_norm, .ffn_norm);
    kernels.project(layer, .gate_up_proj, .x_norm, &.{
        .{ .weight = .ffn_gate, .out = .ffn_gate },
        .{ .weight = .ffn_up, .out = .ffn_up },
    });
    kernels.swiglu(layer);
    kernels.project(layer, .down_proj, .ffn_gate, &.{.{ .weight = .ffn_down, .out = .mix_out }});
    kernels.residual(layer, .ffn_residual, .mix_out);
}

/// True for a full-attention layer, false for a GDN layer. `layer` is the model index.
pub fn isFullAttention(card: *const loader.ModelCard, layer: u32) bool {
    assert(card.arch == .qwen35);
    assert(layer < card.n_layers);

    return card.gdn.?.isFullAttention(layer);
}

/// The GGUF name format of a weight role, with the layer index as its one argument. Null: the
/// family holds no such role.
pub fn tensorName(comptime weight: Weight) ?[]const u8 {
    return switch (weight) {
        .attn_norm => "blk.{d}.attn_norm.weight",
        .ffn_norm => "blk.{d}.post_attention_norm.weight",
        .attn_q => "blk.{d}.attn_q.weight",
        .attn_k => "blk.{d}.attn_k.weight",
        .attn_v => "blk.{d}.attn_v.weight",
        .attn_o => "blk.{d}.attn_output.weight",
        .attn_q_norm => "blk.{d}.attn_q_norm.weight",
        .attn_k_norm => "blk.{d}.attn_k_norm.weight",
        .ssm_qkv => "blk.{d}.attn_qkv.weight",
        .ssm_z => "blk.{d}.attn_gate.weight",
        .ssm_alpha => "blk.{d}.ssm_alpha.weight",
        .ssm_beta => "blk.{d}.ssm_beta.weight",
        .ssm_out => "blk.{d}.ssm_out.weight",
        .ssm_conv1d => "blk.{d}.ssm_conv1d.weight",
        .ssm_a => "blk.{d}.ssm_a",
        .ssm_dt => "blk.{d}.ssm_dt.bias",
        .ssm_norm => "blk.{d}.ssm_norm.weight",
        .ffn_gate => "blk.{d}.ffn_gate.weight",
        .ffn_up => "blk.{d}.ffn_up.weight",
        .ffn_down => "blk.{d}.ffn_down.weight",
    };
}

/// The weight roles of the model layer `layer`, by its kind.
pub fn layerWeights(card: *const loader.ModelCard, layer: u32) []const Weight {
    return if (isFullAttention(card, layer)) &full_weights else &gdn_weights;
}

/// The elements of `act` in one step row. Zero: the family holds no such buffer.
pub fn activationWidth(card: *const loader.ModelCard, act: Activation) u32 {
    const g = card.gdn.?;
    const key_dim = g.group_count * g.state_size;
    const value_dim = g.time_step_rank * g.state_size;
    const q_dim = card.n_heads * card.head_dim;
    return switch (act) {
        .x, .x_norm, .mix_out => card.embedding_length,
        // The q projection interleaves q and the attention gate per head.
        .q_gate => 2 * q_dim,
        .q, .gate, .attn_out => q_dim,
        .k, .v => card.n_kv_heads * card.head_dim,
        .qkv => 2 * key_dim + value_dim,
        .z, .mix_o => value_dim,
        .alpha, .beta => g.time_step_rank,
        .ffn_gate, .ffn_up => card.feed_forward_length,
        .ffn_down => 0,
    };
}

/// The dims that the card requires of a layer weight, in the order of the GGUF shape: the input
/// width, then the output width. A vector holds 0 as its second entry. Zero: no such weight.
pub fn weightDims(card: *const loader.ModelCard, weight: Weight) [2]u32 {
    assert(card.arch == .qwen35);

    const g = card.gdn.?;
    const emb = card.embedding_length;
    const q_dim = activationWidth(card, .q);
    const kv_dim = activationWidth(card, .k);
    const qkv = activationWidth(card, .qkv);
    const value_dim = activationWidth(card, .z);
    const ffn = card.feed_forward_length;
    return switch (weight) {
        .attn_norm, .ffn_norm => .{ emb, 0 },
        .attn_q => .{ emb, activationWidth(card, .q_gate) },
        .attn_k, .attn_v => .{ emb, kv_dim },
        .attn_q_norm, .attn_k_norm => .{ card.head_dim, 0 },
        .attn_o => .{ q_dim, emb },
        .ssm_qkv => .{ emb, qkv },
        .ssm_z => .{ emb, value_dim },
        .ssm_alpha, .ssm_beta => .{ emb, g.time_step_rank },
        .ssm_out => .{ value_dim, emb },
        // The conv weight holds the taps of one channel next to each other.
        .ssm_conv1d => .{ g.conv_kernel, qkv },
        .ssm_a, .ssm_dt => .{ g.time_step_rank, 0 },
        .ssm_norm => .{ g.state_size, 0 },
        .ffn_gate, .ffn_up => .{ emb, ffn },
        .ffn_down => .{ ffn, emb },
    };
}

/// The KV region of the model layer `layer` in a runner of the layers from `first`. Null for a
/// GDN layer, which holds no KV.
pub fn kvIndex(card: *const loader.ModelCard, first: u32, layer: u32) ?u32 {
    assert(first <= layer);
    assert(layer < card.n_layers);

    if (!isFullAttention(card, layer)) return null;

    // Layer l is full when l + 1 is a multiple of the interval, so `[0, x)` holds x / interval.
    const interval = card.gdn.?.full_attention_interval;
    const index = layer / interval - first / interval;
    assert(index <= layer - first);
    return index;
}

/// The f32 elements of one slot of one GDN layer: the recurrent state and the conv ring.
pub fn recurrentElems(card: *const loader.ModelCard) layer_kernels.RecurrentElems {
    const g = card.gdn.?;
    const conv_dim = activationWidth(card, .qkv);
    return .{
        .state = g.time_step_rank * g.state_size * g.state_size,
        .conv = (g.conv_kernel - 1) * conv_dim,
    };
}

/// Refuse a shape past the arrays of the Qwen GPU kernels, before the fused runner allocates.
pub fn fusedGuard(card: *const loader.ModelCard) qwen_gpu_guard.Error!void {
    assert(card.arch == .qwen35);

    return qwen_gpu_guard.requireShape(card);
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;
const naive = @import("../ops/naive.zig");
const kv_cost = @import("kv_cost.zig");
const test_fakes = @import("test_fakes.zig");

const LogKernels = layer_kernels.LogKernels;

fn logLayer(card: *const loader.ModelCard, layer: u32, log: *LogKernels.Log) void {
    var kernels: LogKernels = .{ .card = card, .log = log };
    forward(LogKernels, &kernels, layer);
}

test {
    // The GPU load gate of the family has no other test reference in a CPU build.
    _ = qwen_gpu_guard;
}

test "forward: a full layer and a GDN layer make their ops in the order of the model" {
    const card = test_fakes.testCard(.qwen35);
    var full: LogKernels.Log = .{};
    logLayer(&card, 3, &full);
    try testing.expectEqualSlices(LayerOp, &.{
        .attn_norm,    .attn_proj, .qk_norm,   .rope,           .kv_commit,
        .attention,    .attn_gate, .o_proj,    .mixer_residual, .ffn_norm,
        .gate_up_proj, .swiglu,    .down_proj, .ffn_residual,
    }, full.calls());
    try testing.expectEqual(@as(?naive.RopeStyle, .neox), full.rope_style);

    var gdn: LogKernels.Log = .{};
    logLayer(&card, 0, &gdn);
    try testing.expectEqualSlices(LayerOp, &.{
        .attn_norm, .gdn_proj,     .gdn,    .gdn_out,   .mixer_residual,
        .ffn_norm,  .gate_up_proj, .swiglu, .down_proj, .ffn_residual,
    }, gdn.calls());
}

test "forward: each op occurs at most once, on exactly the weights and buffers of the layer" {
    const card = test_fakes.testCard(.qwen35);
    var names_seen: std.EnumSet(Weight) = .initEmpty();
    var layer: u32 = 0;
    while (layer < card.n_layers) : (layer += 1) {
        var log: LogKernels.Log = .{};
        logLayer(&card, layer, &log);

        // The binding table keys a layer by op, so no op repeats.
        var ops: std.EnumSet(LayerOp) = .initEmpty();
        for (log.calls()) |op| {
            try testing.expect(!ops.contains(op));
            ops.insert(op);
        }

        // The calls touch every role of the layer and no other, and each role has a name.
        try testing.expect(log.weights.eql(.initMany(layerWeights(&card, layer))));
        var weights = log.weights.iterator();
        while (weights.next()) |w| {
            names_seen.insert(w);
        }

        // Every buffer that a call touches has a width.
        var acts = log.acts.iterator();
        while (acts.next()) |act| {
            try testing.expect(activationWidth(&card, act) > 0);
        }

        try testing.expect(!log.acts.contains(.ffn_down));
    }

    // The two layer kinds cover every role, and every role has its own GGUF name.
    const roles = comptime std.meta.tags(Weight);
    const names = comptime names: {
        var out: [roles.len][]const u8 = undefined;
        for (roles, &out) |role, *name| {
            name.* = tensorName(role).?;
        }

        break :names out;
    };
    try testing.expectEqual(@as(usize, roles.len), names_seen.count());
    for (names, 0..) |a, i| {
        try testing.expect(std.mem.startsWith(u8, a, "blk.{d}."));
        for (names[i + 1 ..]) |b| {
            try testing.expect(!std.mem.eql(u8, a, b));
        }
    }
}

test "forward: the two layer kinds call each Qwen CPU row of a layer, and no other op" {
    const card = test_fakes.testCard(.qwen35);
    var called: std.EnumSet(LayerOp) = .initEmpty();
    for ([_]u32{ 0, 3 }) |layer| {
        var log: LogKernels.Log = .{};
        logLayer(&card, layer, &log);
        for (log.calls()) |op| {
            called.insert(op);
        }
    }

    // The runner owns the embed and the head, so only their ops stay outside.
    for (std.enums.values(LayerOp)) |op| {
        const runner = op == .embed or op == .final_norm or op == .lm_head;
        const used = cpu_profile.usedBy(op, profile_family);
        try testing.expectEqual(used and !runner, called.contains(op));
    }
}

test "kvIndex: the closed form counts the full layers of the range before the layer" {
    const card = test_fakes.testCard(.qwen35);
    var first: u32 = 0;
    while (first < card.n_layers) : (first += 1) {
        var walked: u32 = 0;
        var layer: u32 = first;
        while (layer < card.n_layers) : (layer += 1) {
            const index = kvIndex(&card, first, layer);
            if (!isFullAttention(&card, layer)) {
                try testing.expectEqual(@as(?u32, null), index);
                continue;
            }

            try testing.expectEqual(@as(?u32, walked), index);
            walked += 1;
        }
    }
}

test "recurrentElems: one GDN slot holds the fixed bytes that the cost vector sizes" {
    const card = test_fakes.testCard(.qwen35);
    const elems = recurrentElems(&card);
    const vector = kv_cost.qwenCostVector(card, .f32);
    for (vector.layers(), 0..) |entry, li| {
        const fixed: u64 = if (isFullAttention(&card, @intCast(li)))
            0
        else
            (@as(u64, elems.state) + elems.conv) * @sizeOf(f32);
        try testing.expectEqual(fixed, entry.fixed_bytes_per_slot);
    }
}

test "forward: a projection multiplies exactly the matrix roles of the layer" {
    // The fused runner uploads a role as a matrix or as a vector by `isMatrix`, and the bind pass
    // binds a projection weight as a matrix.
    const card = test_fakes.testCard(.qwen35);
    var layer: u32 = 0;
    while (layer < card.n_layers) : (layer += 1) {
        var log: LogKernels.Log = .{};
        logLayer(&card, layer, &log);
        for (layerWeights(&card, layer)) |role| {
            try testing.expectEqual(role.isMatrix(), log.projected.contains(role));
        }
    }
}
