//! The Llama decoder layer: its forward through a layer kernel set, and the rules of the family
//! that a runner reads to hold and bind the layer.
//!
//! Comptime. `forward` takes the kernel set type and checks it with `assertLayerKernels`.
//! `tensorName` takes the role at comptime, because `tensor_lookup.roleName` takes a comptime
//! format.
//!
//! Layer. Every layer holds the same nine weights: grouped-query attention with adjacent-pair
//! rope, then the SwiGLU FFN. The o projection writes `x_norm`, because no op of the layer reads
//! the normed input after the q, k, and v projections. The down projection writes `ffn_down`. Each
//! residual adds the buffer of the projection before it to `x`.
//!
//! KV index. Every layer holds KV, so a runner of the layers from `first` keeps the KV region of
//! the model layer `layer` at `layer - first`. No layer holds recurrent state.
//!
//! Rope factors. A Llama 3.1 file can hold one factor tensor that divides the frequency of each
//! rotated pair in every layer. A file without it rotates at the plain frequencies.
//!
//! Tests. The logging kernel set of `layer_kernels` records the calls of `forward`. The tests check
//! the op order, that each op occurs at most once in a layer, that each residual adds what the
//! projection before it wrote, and that the ops charge the Llama categories of the CPU profile.

const std = @import("std");
const loader = @import("../io/gguf/loader.zig");
const layer_kernels = @import("layer_kernels.zig");
const cpu_profile = @import("cpu_profile.zig");
const dispatch_guard = @import("dispatch_guard.zig");

const Activation = layer_kernels.Activation;
const GpuRunner = @import("../core/span.zig").GpuRunner;
const LayerOp = layer_kernels.LayerOp;
const Weight = layer_kernels.Weight;
const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

/// The architecture of the family. A runner refuses a card of another one.
pub const arch: loader.Arch = .llama;

/// No Llama layer holds recurrent state, so the session surface needs no slot hooks.
pub const has_recurrent_state = false;

/// The GGUF name of the rope factor tensor, one tensor for all layers.
pub const rope_freqs: ?[]const u8 = loader.rope_freqs_tensor;

/// The op categories of the CPU profile of the family.
pub const profile_family: cpu_profile.Family = .llama;

/// The weight quants of the fused runner. A Q4_0 model can hold a Q6_K lm_head, which the runner
/// converts to Q4_0.
pub const fused_quants: []const layer_kernels.WeightQuant = &.{ .q8_0, .q4_0 };

/// The fused runner records the batch kernel at every row count past one.
pub const fused_gemm = false;

/// The rows of one fused step, which bound the time of its submit. A multi-second submit
/// destabilizes a translation-layer driver.
pub const fused_rows_max: u32 = 256;

/// The GPU records of the fused runner: the whole model, and the GPU layers of a hybrid.
pub const fused_gpu_runner: GpuRunner = .llama_fused;
pub const hybrid_gpu_runner: GpuRunner = .llama_hybrid;

/// The weight roles of every layer.
const layer_weights = [_]Weight{
    .attn_norm,
    .attn_q,
    .attn_k,
    .attn_v,
    .attn_o,
    .ffn_norm,
    .ffn_gate,
    .ffn_up,
    .ffn_down,
};

// -- api --------------------------------------------------------------------

/// One Llama decoder layer over the step rows that `kernels` holds. `layer` is the model index.
pub fn forward(comptime Kernels: type, kernels: *Kernels, layer: u32) void {
    comptime layer_kernels.assertLayerKernels(Kernels);
    assert(layer < kernels.card.n_layers);
    assert(kernels.card.gdn == null);

    // Norm the layer input, attend over the KV of each step row, and add the attention residual.
    kernels.norm(layer, .attn_norm, .attn_norm);
    kernels.project(layer, .attn_proj, .x_norm, &.{
        .{ .weight = .attn_q, .out = .q },
        .{ .weight = .attn_k, .out = .k },
        .{ .weight = .attn_v, .out = .v },
    });
    kernels.rope(layer, .norm);
    kernels.kvCommit(layer);
    kernels.attention(layer);
    kernels.project(layer, .o_proj, .attn_out, &.{.{ .weight = .attn_o, .out = .x_norm }});
    kernels.residual(layer, .mixer_residual, .x_norm);

    // Run the FFN and add its residual.
    kernels.norm(layer, .ffn_norm, .ffn_norm);
    kernels.project(layer, .gate_up_proj, .x_norm, &.{
        .{ .weight = .ffn_gate, .out = .ffn_gate },
        .{ .weight = .ffn_up, .out = .ffn_up },
    });
    kernels.swiglu(layer);
    kernels.project(layer, .down_proj, .ffn_gate, &.{.{ .weight = .ffn_down, .out = .ffn_down }});
    kernels.residual(layer, .ffn_residual, .ffn_down);
}

/// The GGUF name format of a weight role, with the layer index as its one argument. Null: the
/// family holds no such role.
pub fn tensorName(comptime weight: Weight) ?[]const u8 {
    return switch (weight) {
        .attn_norm => "blk.{d}.attn_norm.weight",
        .ffn_norm => "blk.{d}.ffn_norm.weight",
        .attn_q => "blk.{d}.attn_q.weight",
        .attn_k => "blk.{d}.attn_k.weight",
        .attn_v => "blk.{d}.attn_v.weight",
        .attn_o => "blk.{d}.attn_output.weight",
        .ffn_gate => "blk.{d}.ffn_gate.weight",
        .ffn_up => "blk.{d}.ffn_up.weight",
        .ffn_down => "blk.{d}.ffn_down.weight",
        .attn_q_norm,
        .attn_k_norm,
        .ssm_qkv,
        .ssm_z,
        .ssm_alpha,
        .ssm_beta,
        .ssm_out,
        .ssm_conv1d,
        .ssm_a,
        .ssm_dt,
        .ssm_norm,
        => null,
    };
}

/// The weight roles of the model layer `layer`: the same nine in every layer.
pub fn layerWeights(card: *const loader.ModelCard, layer: u32) []const Weight {
    assert(card.arch == .llama);
    assert(layer < card.n_layers);

    return &layer_weights;
}

/// The elements of `act` in one step row. Zero: the family holds no such buffer.
pub fn activationWidth(card: *const loader.ModelCard, act: Activation) u32 {
    assert(card.arch == .llama);

    const q_dim = card.n_heads * card.head_dim;
    return switch (act) {
        .x, .x_norm, .ffn_down => card.embedding_length,
        .q, .attn_out => q_dim,
        .k, .v => card.n_kv_heads * card.head_dim,
        .ffn_gate, .ffn_up => card.feed_forward_length,
        .q_gate, .gate, .mix_out, .qkv, .z, .alpha, .beta, .mix_o => 0,
    };
}

/// The dims that the card requires of a layer weight, in the order of the GGUF shape: the input
/// width, then the output width. A vector holds 0 as its second entry. Zero: no such weight.
pub fn weightDims(card: *const loader.ModelCard, weight: Weight) [2]u32 {
    assert(card.arch == .llama);

    const emb = card.embedding_length;
    const q_dim = activationWidth(card, .q);
    const kv_dim = activationWidth(card, .k);
    const ffn = card.feed_forward_length;
    return switch (weight) {
        .attn_norm, .ffn_norm => .{ emb, 0 },
        .attn_q => .{ emb, q_dim },
        .attn_k, .attn_v => .{ emb, kv_dim },
        .attn_o => .{ q_dim, emb },
        .ffn_gate, .ffn_up => .{ emb, ffn },
        .ffn_down => .{ ffn, emb },
        .attn_q_norm,
        .attn_k_norm,
        .ssm_qkv,
        .ssm_z,
        .ssm_alpha,
        .ssm_beta,
        .ssm_out,
        .ssm_conv1d,
        .ssm_a,
        .ssm_dt,
        .ssm_norm,
        => .{ 0, 0 },
    };
}

/// The KV region of the model layer `layer` in a runner of the layers from `first`.
pub fn kvIndex(card: *const loader.ModelCard, first: u32, layer: u32) ?u32 {
    assert(first <= layer);
    assert(layer < card.n_layers);

    return layer - first;
}

/// The f32 elements of one slot of one recurrent layer: none.
pub fn recurrentElems(card: *const loader.ModelCard) layer_kernels.RecurrentElems {
    assert(card.gdn == null);

    return .{ .state = 0, .conv = 0 };
}

/// Refuse a head count past one axis of a fused dispatch. The fused kernels take every other
/// Llama shape that the checks of all families pass.
pub fn fusedGuard(card: *const loader.ModelCard) dispatch_guard.Error!void {
    assert(card.arch == .llama);

    return dispatch_guard.requireHeads(card);
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

test "forward: every layer makes its ops in the order of the model, each op once" {
    const card = test_fakes.testCard(.llama);
    var layer: u32 = 0;
    while (layer < card.n_layers) : (layer += 1) {
        var log: LogKernels.Log = .{};
        logLayer(&card, layer, &log);
        try testing.expectEqualSlices(LayerOp, &.{
            .attn_norm, .attn_proj,      .rope,     .kv_commit,    .attention,
            .o_proj,    .mixer_residual, .ffn_norm, .gate_up_proj, .swiglu,
            .down_proj, .ffn_residual,
        }, log.calls());

        // Each residual adds the output of the projection right before it, and the rope rotates
        // adjacent pairs.
        try testing.expectEqual(@as(u32, 2), log.residuals_matched);
        try testing.expectEqual(@as(?naive.RopeStyle, .norm), log.rope_style);
    }
}

test "forward: the calls touch the weights and the buffers that the rules hold" {
    const card = test_fakes.testCard(.llama);
    var log: LogKernels.Log = .{};
    logLayer(&card, 0, &log);

    // The calls touch every role of the layer and no other.
    try testing.expect(log.weights.eql(.initMany(layerWeights(&card, 0))));

    // A buffer has a width exactly when a call touches it.
    for (std.enums.values(Activation)) |act| {
        try testing.expectEqual(log.acts.contains(act), activationWidth(&card, act) > 0);
    }

    // Every role of the layer has its own GGUF name, and no other role has one.
    const roles = comptime std.meta.tags(Weight);
    inline for (roles, 0..) |a, i| {
        const held = log.weights.contains(a);
        try testing.expectEqual(held, tensorName(a) != null);
        if (tensorName(a)) |name_a| {
            try testing.expect(std.mem.startsWith(u8, name_a, "blk.{d}."));
            inline for (roles[i + 1 ..]) |b| {
                if (tensorName(b)) |name_b| {
                    try testing.expect(!std.mem.eql(u8, name_a, name_b));
                }
            }
        }
    }
}

test "forward: a layer calls each Llama CPU row of a layer, and no other op" {
    const card = test_fakes.testCard(.llama);
    var log: LogKernels.Log = .{};
    logLayer(&card, 0, &log);

    var called: std.EnumSet(LayerOp) = .initEmpty();
    for (log.calls()) |op| {
        called.insert(op);
    }

    // The runner owns the embed and the head, so only their ops stay outside.
    for (std.enums.values(LayerOp)) |op| {
        const runner = op == .embed or op == .final_norm or op == .lm_head;
        const used = cpu_profile.usedBy(op, profile_family);
        try testing.expectEqual(used and !runner, called.contains(op));
    }
}

test "kvIndex: each layer of a range keeps its KV region at its range index" {
    const card = test_fakes.testCard(.llama);
    var first: u32 = 0;
    while (first < card.n_layers) : (first += 1) {
        var layer: u32 = first;
        while (layer < card.n_layers) : (layer += 1) {
            try testing.expectEqual(@as(?u32, layer - first), kvIndex(&card, first, layer));
        }
    }
}

test "recurrentElems: a Llama slot holds only the KV that the cost vector prices" {
    const card = test_fakes.testCard(.llama);
    const elems = recurrentElems(&card);
    try testing.expectEqual(@as(u32, 0), elems.state);
    try testing.expectEqual(@as(u32, 0), elems.conv);
    try testing.expect(!has_recurrent_state);

    // Each layer prices one K row and one V row a token, and no fixed bytes.
    const vector = kv_cost.llamaCostVector(card, .f32);
    const row_bytes: u64 = activationWidth(&card, .k) * @sizeOf(f32);
    for (vector.layers()) |entry| {
        try testing.expectEqual(@as(u64, 0), entry.fixed_bytes_per_slot);
        try testing.expectEqual(2 * row_bytes, @as(u64, entry.kv_bytes_per_token));
    }
}

test "forward: a projection multiplies exactly the matrix roles of the layer" {
    // The fused runner uploads a role as a matrix or as a vector by `isMatrix`, and the bind pass
    // binds a projection weight as a matrix.
    const card = test_fakes.testCard(.llama);
    var layer: u32 = 0;
    while (layer < card.n_layers) : (layer += 1) {
        var log: LogKernels.Log = .{};
        logLayer(&card, layer, &log);
        for (layerWeights(&card, layer)) |role| {
            try testing.expectEqual(role.isMatrix(), log.projected.contains(role));
        }
    }
}
