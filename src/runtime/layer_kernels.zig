//! The vocabulary of a decoder layer that every family and every layer kernel set shares: the ops
//! of a layer, the weight roles, the activations, and the build-time check of a kernel set.
//!
//! Layer kernel set. A type with ten methods: `norm`, `project`, `qkNorm`, `rope`, `kvCommit`,
//! `attention`, `gate`, `residual`, `swiglu`, and `gdn`. Each call computes or records one
//! `LayerOp` over the step rows that the value holds. A decoder layer calls only these methods. It
//! reads the layer kind from the `card` field, and it passes the model index of the layer.
//!
//! Comptime. `assertLayerKernels` checks the methods and the `card` field of a kernel set, and
//! `assertFamily` and `assertFusedFamily` check the declarations of a layer file.
//!
//! Called ops. `LogKernels` computes nothing and logs each call of one layer. `calledOps` runs the
//! forward of every model layer through it, so a runner builds only the kernels of the ops that a
//! family calls. The layer tests read the same log.
//!
//! Family. A layer file (`llama_layer`, `qwen_layer`) is the family parameter of a runner. It
//! declares the same names with the same types in every family. Zig has no conditional
//! declaration, so a family without a feature states a value for it. Only the fused runner reads
//! the `fused_` values.
//!
//! Bindings. `LayerBinding` holds the descriptor sets of one layer, keyed by `LayerOp`, because a
//! layer makes each op call at most once. A layer op holds one set unless `Set order` names it,
//! and an op of the runner holds none. A null handle marks an unused entry. A CPU build binds no
//! set, so its binding holds no handle.
//!
//! Set order. The sets of one op keep a fixed order. `project` follows its `outs` order, and
//! `qkNorm` and `rope` put q before k. `attention` holds the resident set of a decode step or a
//! prefill chunk, then the multi-stream set, then the split set. `gdn` holds the recurrence set,
//! then the conv set, then the scan set.
//!
//! Ops. `LayerOp` comes from `core/layer_op.zig`, the one list of op names. A CPU kernel set opens
//! the span of each op, and a fused kernel set marks each op, so both profiles read one list.

const std = @import("std");
const build_options = @import("build_options");
const loader = @import("../io/gguf/loader.zig");
const naive = @import("../ops/naive.zig");
const core_span = @import("../core/span.zig");
const layer_op = @import("../core/layer_op.zig");
const limits = @import("../core/limits.zig");
const cpu_profile = @import("cpu_profile.zig");

const assert = std.debug.assert;

const DescriptorSet = if (build_options.use_vulkan) @import("vulkan").DescriptorSet else void;

// -- constants --------------------------------------------------------------

/// The most weights of one `project` call: the four GDN input projections.
pub const projections_max = 4;

/// The most descriptor sets of one op: the four sets of the GDN input projections.
pub const op_sets_max = 4;

// -- types ------------------------------------------------------------------

/// The runner entry that a step records for. A fused layer kernel set takes one as its kind.
pub const StepKind = enum { decode, decode_batch, prefill_chunk };

/// One op of a forward pass. A layer op keys the binding of one call.
pub const LayerOp = layer_op.LayerOp;

/// One activation buffer that a layer op reads or writes. A family holds a subset.
pub const Activation = enum {
    x,
    x_norm,
    q_gate,
    q,
    gate,
    k,
    v,
    attn_out,
    mix_out,
    qkv,
    z,
    alpha,
    beta,
    mix_o,
    ffn_gate,
    ffn_up,
    ffn_down,
};

/// The role of one weight of a decoder layer, over all families. The `tensorName` of a family
/// maps a role to its GGUF name.
pub const Weight = enum {
    attn_norm,
    ffn_norm,
    attn_q,
    attn_k,
    attn_v,
    attn_o,
    attn_q_norm,
    attn_k_norm,
    ssm_qkv,
    ssm_z,
    ssm_alpha,
    ssm_beta,
    ssm_out,
    ssm_conv1d,
    ssm_a,
    ssm_dt,
    ssm_norm,
    ffn_gate,
    ffn_up,
    ffn_down,

    /// True for a matrix that a `project` call multiplies. False for an f32 vector that a norm,
    /// the QK norm, or the GDN reads.
    pub fn isMatrix(self: Weight) bool {
        return switch (self) {
            .attn_q,
            .attn_k,
            .attn_v,
            .attn_o,
            .ssm_qkv,
            .ssm_z,
            .ssm_alpha,
            .ssm_beta,
            .ssm_out,
            .ffn_gate,
            .ffn_up,
            .ffn_down,
            => true,
            .attn_norm,
            .ffn_norm,
            .attn_q_norm,
            .attn_k_norm,
            .ssm_conv1d,
            .ssm_a,
            .ssm_dt,
            .ssm_norm,
            => false,
        };
    }
};

/// The quantization of the matmul weights of a model. A fused runner detects it once, because the
/// matmuls of every layer are uniform. It picks the matmul kernels and the weight upload.
pub const WeightQuant = enum { q8_0, q4_0 };

/// One weight of a `project` call and the activation it writes.
pub const Projection = struct { weight: Weight, out: Activation };

/// The f32 elements of one slot of one recurrent layer: the state and the conv ring.
pub const RecurrentElems = struct { state: u32, conv: u32 };

/// The descriptor sets of each op of one layer, in the order of the header.
pub const LayerBinding = std.EnumArray(LayerOp, [op_sets_max]DescriptorSet);

/// A kernel set that computes nothing. It logs each call of one layer, with the weights and the
/// activations that the call touches. A fixed-operand method logs the operands that its fused and
/// CPU forms read and write.
pub const LogKernels = struct {
    card: *const loader.ModelCard,
    log: *Log,

    pub const Log = struct {
        ops: [std.meta.tags(LayerOp).len]LayerOp = undefined,
        len: u32 = 0,
        layer: ?u32 = null,
        weights: std.EnumSet(Weight) = .initEmpty(),
        /// The weights that a `project` call multiplies.
        projected: std.EnumSet(Weight) = .initEmpty(),
        acts: std.EnumSet(Activation) = .initEmpty(),
        rope_style: ?naive.RopeStyle = null,
        /// The output of the last projection call, and the count of residuals that added it.
        last_out: ?Activation = null,
        residuals_matched: u32 = 0,

        /// The ops of the layer in call order.
        pub fn calls(self: *const Log) []const LayerOp {
            return self.ops[0..self.len];
        }

        fn add(self: *Log, layer: u32, op: LayerOp) void {
            if (self.layer) |l| assert(l == layer);
            assert(self.len < self.ops.len);

            self.layer = layer;
            self.ops[self.len] = op;
            self.len += 1;
        }

        fn touch(self: *Log, weights: []const Weight, acts: []const Activation) void {
            for (weights) |w| {
                self.weights.insert(w);
            }

            for (acts) |a| {
                self.acts.insert(a);
            }
        }
    };

    pub fn norm(self: *LogKernels, layer: u32, op: LayerOp, weight: Weight) void {
        self.log.add(layer, op);
        self.log.touch(&.{weight}, &.{ .x, .x_norm });
    }

    pub fn project(
        self: *LogKernels,
        layer: u32,
        op: LayerOp,
        in: Activation,
        outs: []const Projection,
    ) void {
        assert(outs.len >= 1);
        assert(outs.len <= projections_max);

        self.log.add(layer, op);
        self.log.touch(&.{}, &.{in});
        for (outs) |p| {
            self.log.touch(&.{p.weight}, &.{p.out});
            self.log.projected.insert(p.weight);
        }

        self.log.last_out = outs[outs.len - 1].out;
    }

    pub fn qkNorm(self: *LogKernels, layer: u32) void {
        self.log.add(layer, .qk_norm);
        self.log.touch(&.{ .attn_q_norm, .attn_k_norm }, &.{ .q_gate, .k, .q, .gate });
    }

    pub fn rope(self: *LogKernels, layer: u32, style: naive.RopeStyle) void {
        self.log.add(layer, .rope);
        self.log.touch(&.{}, &.{ .q, .k });
        self.log.rope_style = style;
    }

    pub fn kvCommit(self: *LogKernels, layer: u32) void {
        self.log.add(layer, .kv_commit);
        self.log.touch(&.{}, &.{ .k, .v });
    }

    pub fn attention(self: *LogKernels, layer: u32) void {
        self.log.add(layer, .attention);
        self.log.touch(&.{}, &.{ .q, .attn_out });
    }

    pub fn gate(self: *LogKernels, layer: u32) void {
        self.log.add(layer, .attn_gate);
        self.log.touch(&.{}, &.{ .gate, .attn_out });
    }

    pub fn residual(self: *LogKernels, layer: u32, op: LayerOp, src: Activation) void {
        self.log.add(layer, op);
        self.log.touch(&.{}, &.{ .x, src });
        if (self.log.last_out == src) {
            self.log.residuals_matched += 1;
        }
    }

    pub fn swiglu(self: *LogKernels, layer: u32) void {
        self.log.add(layer, .swiglu);
        self.log.touch(&.{}, &.{ .ffn_gate, .ffn_up });
    }

    pub fn gdn(self: *LogKernels, layer: u32) void {
        self.log.add(layer, .gdn);
        self.log.touch(
            &.{ .ssm_conv1d, .ssm_a, .ssm_dt, .ssm_norm },
            &.{ .qkv, .z, .alpha, .beta, .mix_o },
        );
    }
};

// -- api --------------------------------------------------------------------

/// Check that `K` declares the ten methods with their exact signatures and the `card` field. A
/// kernel set that lacks one fails the build.
pub fn assertLayerKernels(comptime K: type) void {
    const methods = .{
        .{ "norm", fn (*K, u32, LayerOp, Weight) void },
        .{ "project", fn (*K, u32, LayerOp, Activation, []const Projection) void },
        .{ "qkNorm", fn (*K, u32) void },
        .{ "rope", fn (*K, u32, naive.RopeStyle) void },
        .{ "kvCommit", fn (*K, u32) void },
        .{ "attention", fn (*K, u32) void },
        .{ "gate", fn (*K, u32) void },
        .{ "residual", fn (*K, u32, LayerOp, Activation) void },
        .{ "swiglu", fn (*K, u32) void },
        .{ "gdn", fn (*K, u32) void },
    };
    const set = "layer kernel set " ++ @typeName(K);
    inline for (methods) |m| {
        if (!@hasDecl(K, m[0])) @compileError(set ++ " lacks " ++ m[0]);

        const Have = @TypeOf(@field(K, m[0]));
        if (Have != m[1]) @compileError(set ++ ": " ++ m[0] ++ " is " ++ @typeName(Have));
    }

    if (!@hasField(K, "card")) @compileError(set ++ " lacks the field card");
    if (@FieldType(K, "card") != *const loader.ModelCard) @compileError(set ++ ": bad card type");
}

/// Check that the layer file `F` declares the values and the rules that every runner reads, with
/// their exact types. A family that lacks one fails the build.
pub fn assertFamily(comptime F: type) void {
    const Card = *const loader.ModelCard;
    assertDecls(F, .{
        .{ "arch", loader.Arch },
        .{ "has_recurrent_state", bool },
        .{ "rope_freqs", ?[]const u8 },
        .{ "profile_family", cpu_profile.Family },
        .{ "tensorName", fn (comptime Weight) ?[]const u8 },
        .{ "layerWeights", fn (Card, u32) []const Weight },
        .{ "activationWidth", fn (Card, Activation) u32 },
        .{ "kvIndex", fn (Card, u32, u32) ?u32 },
        .{ "recurrentElems", fn (Card) RecurrentElems },
    });

    // `forward` is generic over the kernel set, so the check reads its shape.
    const family = "layer family " ++ @typeName(F);
    if (!@hasDecl(F, "forward")) @compileError(family ++ " lacks forward");

    const forward = @typeInfo(@TypeOf(F.forward));
    const shaped = forward == .@"fn" and forward.@"fn".params.len == 3 and
        (forward.@"fn".return_type orelse noreturn) == void;
    if (!shaped) @compileError(family ++ ": forward is " ++ @typeName(@TypeOf(F.forward)));
}

/// Check that the layer file `F` declares the `fused_` values and the GPU records that only the
/// fused runners read, with their exact types, and a `fusedGuard` of the right shape.
pub fn assertFusedFamily(comptime F: type) void {
    const Card = *const loader.ModelCard;
    assertDecls(F, .{
        .{ "fused_quants", []const WeightQuant },
        .{ "fused_gemm", bool },
        .{ "fused_rows_max", u32 },
        .{ "fused_gpu_runner", core_span.GpuRunner },
        .{ "hybrid_gpu_runner", core_span.GpuRunner },
    });

    // `fusedGuard` returns the refusals of its own family, so the check reads its shape.
    const family = "layer family " ++ @typeName(F);
    if (!@hasDecl(F, "fusedGuard")) @compileError(family ++ " lacks fusedGuard");

    const guard = @typeInfo(@TypeOf(F.fusedGuard));
    const guard_shaped = guard == .@"fn" and guard.@"fn".params.len == 1 and
        guard.@"fn".params[0].type == Card and
        @typeInfo(guard.@"fn".return_type.?) == .error_union and
        @typeInfo(guard.@"fn".return_type.?).error_union.payload == void;
    if (!guard_shaped) {
        @compileError(family ++ ": fusedGuard is " ++ @typeName(@TypeOf(F.fusedGuard)));
    }
}

/// The ops that the forward of the layer file `F` calls over every layer of the model. A fused
/// runner builds the kernels of an op that a family can lack only when this set holds the op.
pub fn calledOps(comptime F: type, card: *const loader.ModelCard) std.EnumSet(LayerOp) {
    assert(card.n_layers >= 1);
    assert(card.n_layers <= limits.max_layers);

    var ops: std.EnumSet(LayerOp) = .initEmpty();
    var layer: u32 = 0;
    while (layer < card.n_layers) : (layer += 1) {
        var log: LogKernels.Log = .{};
        var kernels: LogKernels = .{ .card = card, .log = &log };
        F.forward(LogKernels, &kernels, layer);
        for (log.calls()) |op| {
            ops.insert(op);
        }
    }

    // Every layer norms its input and adds the residual of its FFN.
    assert(ops.contains(.attn_norm));
    assert(ops.contains(.ffn_residual));
    return ops;
}

// -- helpers ----------------------------------------------------------------

/// Check that `F` declares each `{ name, type }` pair of `decls` with exactly that type.
fn assertDecls(comptime F: type, comptime decls: anytype) void {
    const family = "layer family " ++ @typeName(F);
    inline for (decls) |d| {
        if (!@hasDecl(F, d[0])) @compileError(family ++ " lacks " ++ d[0]);

        const Have = @TypeOf(@field(F, d[0]));
        if (Have != d[1]) @compileError(family ++ ": " ++ d[0] ++ " is " ++ @typeName(Have));
    }
}
