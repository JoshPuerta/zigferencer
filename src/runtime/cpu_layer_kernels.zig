//! The layer kernel set of a CPU runner. Each method computes one layer op over all step rows of
//! the value, through the op backend that the build selected.
//!
//! Holds. A value points at what its runner owns and owns nothing: the weights of each layer, the
//! rope factors, the scratch of each activation, the KV slots, the recurrent slabs, the pool, and
//! the step rows. The runner builds one value for each step call. The file reads no runner, so both
//! families build it.
//!
//! Rows. Each op runs over all step rows before the next op starts. A row writes only its own
//! stripe of each scratch buffer, and its attention reads only the KV of its slot up to its
//! position. So the order of the ops across the rows changes no bit of a result.
//!
//! Pool. `project` calls the matmul dispatch once per weight, and `gdn` and `attention` call their
//! pool function once per row. Every other op takes no pool.
//!
//! Comptime. `use_vulkan_ops` selects the op backend and the GPU forms of the recurrence and the
//! attention.
//!
//! Layer index. A method takes the model index of the layer. A CPU runner runs a prefix of the
//! model, so the model index is also the index of its tables.

const std = @import("std");
const build_options = @import("build_options");
const span = @import("../core/span.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const dtype = @import("../core/dtype.zig");
const loader = @import("../io/gguf/loader.zig");
const naive = @import("../ops/naive.zig");
const GdnDims = @import("../ops/gdn.zig").Dims;
const KvCache = @import("kv_cache.zig").KvCache;
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const StepRow = @import("step_row.zig").StepRow;
const linear_dispatch = @import("linear_dispatch.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const layer_kernels = @import("layer_kernels.zig");

const Activation = layer_kernels.Activation;
const LayerOp = layer_kernels.LayerOp;
const Projection = layer_kernels.Projection;
const Weight = layer_kernels.Weight;
const assert = std.debug.assert;

// The op backend that the build resolved, the same module that the CPU runners select.
const ops = if (build_options.use_vulkan_ops)
    @import("../ops_vulkan/wrap.zig")
else if (build_options.cpu_ops_simd_pf)
    @import("../ops/simd_pf.zig")
else if (build_options.cpu_ops_simd)
    @import("../ops/simd.zig")
else
    @import("../ops/naive.zig");

// -- types ------------------------------------------------------------------

/// The layer kernel set of a CPU runner over the step rows of one call.
pub const CpuLayerKernels = struct {
    card: *const loader.ModelCard,
    pool: *Pool,
    policy: parallel.Policy,
    rows: []const StepRow,
    /// The weights of each layer of the prefix, keyed by role. Null: the layer holds no such role.
    layers: []const std.EnumArray(Weight, ?*const Tensor),
    /// The frequency divisor of each rotated pair in every layer. Null: the plain frequencies.
    rope_factors: ?[]const f32,
    /// The KV region of each layer of the prefix. Null: the layer holds no KV.
    kv_index: []const ?u32,
    /// The elements of each activation in one step row. Zero: no buffer.
    widths: *const std.EnumArray(Activation, u32),
    /// The scratch of each activation, one stripe per step row at the width of the activation.
    scratch: std.EnumArray(Activation, []f32),
    /// The part records of one attention row. The rows of a step attend one at a time.
    attn_parts: []f32,
    kv_slots: []KvCache,
    kv_dtype: dtype.KvDtype,
    /// The recurrent state and the conv ring of each (slot, layer), at `slot * layers.len + layer`.
    gdn_state: []const []f32,
    gdn_conv_state: []const []f32,

    // -- api ----------------------------------------------------------------

    /// RMS-norm each row of `x` into `x_norm` with the norm weight `weight`.
    pub fn norm(self: *CpuLayerKernels, layer: u32, op: LayerOp, weight: Weight) void {
        assert(op == .attn_norm or op == .ffn_norm);
        assert(weight == .attn_norm or weight == .ffn_norm);

        const w = self.f32Weight(layer, weight);
        const eps = self.card.rms_norm_eps;
        const sp = span.begin(.ofOps(.initOne(op)));
        defer sp.end();

        for (0..self.rows.len) |i| {
            ops.rmsnorm(self.stripe(.x, i), w, eps, self.stripe(.x_norm, i));
        }
    }

    /// Multiply the rows of `in` by each weight of `outs`, one matmul dispatch per weight.
    pub fn project(
        self: *CpuLayerKernels,
        layer: u32,
        op: LayerOp,
        in: Activation,
        outs: []const Projection,
    ) void {
        assert(outs.len >= 1);
        assert(outs.len <= layer_kernels.projections_max);
        assert(op.class() == .matmul);

        const n = self.rows.len;
        const inputs = self.scratch.get(in)[0 .. n * self.width(in)];
        for (outs) |p| {
            const w = self.tensor(layer, p.weight);
            assert(w.shape[0] == self.width(in));
            assert(w.shape[1] == self.width(p.out));

            const sp = span.begin(.ofOps(.initOne(op)));
            defer sp.end();

            const out = self.scratch.get(p.out);
            linear_dispatch.linearBatchDispatch(self.pool, self.policy, inputs, w.*, out, n);
        }
    }

    /// Split `q_gate` of each row into q and the attention gate, then RMS-norm each head of q and
    /// of k in place.
    pub fn qkNorm(self: *CpuLayerKernels, layer: u32) void {
        // Read the norm weights and the head shape.
        const c = self.card;
        const hd = c.head_dim;
        const q_norm = self.f32Weight(layer, .attn_q_norm);
        const k_norm = self.f32Weight(layer, .attn_k_norm);
        const eps = c.rms_norm_eps;
        assert(self.width(.q_gate) == 2 * self.width(.q));
        assert(self.width(.q) == c.n_heads * hd);
        assert(self.width(.k) == c.n_kv_heads * hd);

        const sp = span.begin(.ofOps(.initOne(.qk_norm)));
        defer sp.end();

        for (0..self.rows.len) |i| {
            // Each head of `q_gate` holds its query, then its gate, `head_dim` elements each.
            const q = self.stripe(.q, i);
            const gate_row = self.stripe(.gate, i);
            const qg = self.stripe(.q_gate, i);
            var h: u32 = 0;
            while (h < c.n_heads) : (h += 1) {
                @memcpy(q[h * hd ..][0..hd], qg[h * 2 * hd ..][0..hd]);
                @memcpy(gate_row[h * hd ..][0..hd], qg[h * 2 * hd + hd ..][0..hd]);
            }

            // Norm each head of q and of k over `head_dim`.
            h = 0;
            while (h < c.n_heads) : (h += 1) {
                const qh = q[h * hd ..][0..hd];
                ops.rmsnorm(qh, q_norm, eps, qh);
            }

            const k = self.stripe(.k, i);
            h = 0;
            while (h < c.n_kv_heads) : (h += 1) {
                const kh = k[h * hd ..][0..hd];
                ops.rmsnorm(kh, k_norm, eps, kh);
            }
        }
    }

    /// Rotate the first `rope_dim` elements of each head of q and k at the position of the row.
    pub fn rope(self: *CpuLayerKernels, layer: u32, style: naive.RopeStyle) void {
        assert(layer < self.layers.len);

        const c = self.card;
        const hd = c.head_dim;
        const sp = span.begin(.ofOps(.initOne(.rope)));
        defer sp.end();

        for (self.rows, 0..) |r, i| {
            const q = self.stripe(.q, i);
            var h: u32 = 0;
            while (h < c.n_heads) : (h += 1) {
                ops.ropeInplace(
                    q[h * hd ..][0..hd],
                    r.pos,
                    c.rope_dim_count,
                    c.rope_freq_base,
                    style,
                    self.rope_factors,
                );
            }

            const k = self.stripe(.k, i);
            h = 0;
            while (h < c.n_kv_heads) : (h += 1) {
                ops.ropeInplace(
                    k[h * hd ..][0..hd],
                    r.pos,
                    c.rope_dim_count,
                    c.rope_freq_base,
                    style,
                    self.rope_factors,
                );
            }
        }
    }

    /// Store k and v of each row at the slot and the position of the row. The narrowing to the KV
    /// dtype happens here, at the one conversion boundary.
    pub fn kvCommit(self: *CpuLayerKernels, layer: u32) void {
        const index = self.kvIndex(layer);
        const kvd = self.width(.k);
        assert(self.width(.v) == kvd);

        const sp = span.begin(.ofOps(.initOne(.kv_commit)));
        defer sp.end();

        for (self.rows, 0..) |r, i| {
            const kv = &self.kv_slots[r.slot];
            const k_src = self.stripe(.k, i);
            const v_src = self.stripe(.v, i);
            switch (self.kv_dtype) {
                .f32 => {
                    @memcpy(kv.kDst(index, r.pos), k_src);
                    @memcpy(kv.vDst(index, r.pos), v_src);
                },
                .f16 => {
                    const k_dst = kv.kDstAs(f16, index, r.pos);
                    const v_dst = kv.vDstAs(f16, index, r.pos);
                    for (0..kvd) |e| {
                        k_dst[e] = @floatCast(k_src[e]);
                        v_dst[e] = @floatCast(v_src[e]);
                    }
                },
            }
        }
    }

    /// Attend q of each row over the positions `0..=pos` of its slot, into `attn_out`.
    pub fn attention(self: *CpuLayerKernels, layer: u32) void {
        // Read the KV region of the layer and the head shape.
        const index = self.kvIndex(layer);
        const c = self.card;
        const hd: usize = c.head_dim;

        // Attend each row on the GPU kernel of the per-op build, or on the pool at the KV dtype.
        for (self.rows, 0..) |r, i| {
            const sp = span.begin(.ofOps(.initOne(.attention)));
            defer sp.end();

            const kv = &self.kv_slots[r.slot];
            const q = self.stripe(.q, i);
            const out = self.stripe(.attn_out, i);
            const t_len: usize = @as(usize, r.pos) + 1;
            if (comptime build_options.use_vulkan_ops) {
                // Flag resolution refuses every dtype but f32 on this build,
                // because `wrap.attentionSingle` uploads the slab as f32.
                assert(self.kv_dtype == .f32);

                ops.attentionSingle(
                    q,
                    c.n_heads,
                    hd,
                    kv.kSlice(index, r.pos),
                    kv.vSlice(index, r.pos),
                    c.n_kv_heads,
                    t_len,
                    out,
                );
            } else {
                switch (self.kv_dtype) {
                    .f32 => parallel.attentionTyped(
                        f32,
                        ops,
                        self.pool,
                        self.policy,
                        q,
                        c.n_heads,
                        hd,
                        kv.kSlice(index, r.pos),
                        kv.vSlice(index, r.pos),
                        c.n_kv_heads,
                        t_len,
                        self.attn_parts,
                        out,
                    ),
                    .f16 => parallel.attentionTyped(
                        f16,
                        ops,
                        self.pool,
                        self.policy,
                        q,
                        c.n_heads,
                        hd,
                        kv.kSliceAs(f16, index, r.pos),
                        kv.vSliceAs(f16, index, r.pos),
                        c.n_kv_heads,
                        t_len,
                        self.attn_parts,
                        out,
                    ),
                }
            }
        }
    }

    /// Multiply `attn_out` of each row by the sigmoid of its attention gate, in place.
    pub fn gate(self: *CpuLayerKernels, layer: u32) void {
        assert(layer < self.layers.len);
        assert(self.width(.gate) == self.width(.attn_out));

        const sp = span.begin(.ofOps(.initOne(.attn_gate)));
        defer sp.end();

        for (0..self.rows.len) |i| {
            for (self.stripe(.attn_out, i), self.stripe(.gate, i)) |*o, gv| {
                o.* *= 1.0 / (1.0 + @exp(-gv));
            }
        }
    }

    /// Add `src` of each row to `x` of the row, in place.
    pub fn residual(self: *CpuLayerKernels, layer: u32, op: LayerOp, src: Activation) void {
        assert(layer < self.layers.len);
        assert(op == .mixer_residual or op == .ffn_residual);
        assert(self.width(src) == self.width(.x));

        const sp = span.begin(.ofOps(.initOne(op)));
        defer sp.end();

        for (0..self.rows.len) |i| {
            ops.residualAdd(self.stripe(src, i), self.stripe(.x, i));
        }
    }

    /// Write `silu(ffn_gate) * ffn_up` of each row into `ffn_gate`.
    pub fn swiglu(self: *CpuLayerKernels, layer: u32) void {
        assert(layer < self.layers.len);
        assert(self.width(.ffn_up) == self.width(.ffn_gate));

        const sp = span.begin(.ofOps(.initOne(.swiglu)));
        defer sp.end();

        for (0..self.rows.len) |i| {
            const g = self.stripe(.ffn_gate, i);
            ops.swiglu(g, self.stripe(.ffn_up, i), g);
        }
    }

    /// Run the conv, the delta rule, and the gated norm of each row into `mix_o`, against the
    /// recurrent state of the slot of the row. Row order is time order for the rows of one slot.
    pub fn gdn(self: *CpuLayerKernels, layer: u32) void {
        // Read the shape and the f32 weights of a recurrent layer, which holds no KV.
        assert(self.kv_index[layer] == null);
        const d = self.gdnDims();
        assert(self.width(.qkv) == d.convDim());
        assert(self.width(.z) == d.valueDim());
        assert(self.width(.alpha) == d.n_v_heads);
        assert(self.width(.mix_o) == d.valueDim());
        const conv1d = self.f32Weight(layer, .ssm_conv1d);
        const ssm_a = self.f32Weight(layer, .ssm_a);
        const dt_bias = self.f32Weight(layer, .ssm_dt);
        const ssm_norm = self.f32Weight(layer, .ssm_norm);
        const eps = self.card.rms_norm_eps;

        const sp = span.begin(.ofOps(.initOne(.gdn)));
        defer sp.end();

        // Step each row against the slabs of its slot.
        for (self.rows, 0..) |r, i| {
            const slab = @as(usize, r.slot) * self.layers.len + layer;
            const state = self.gdn_state[slab];
            const conv_state = self.gdn_conv_state[slab];
            assert(state.len == d.stateLen());
            assert(conv_state.len == d.convStateLen());

            const qkv_row = self.stripe(.qkv, i);
            const z_row = self.stripe(.z, i);
            const a_row = self.stripe(.alpha, i);
            const b_row = self.stripe(.beta, i);
            const o_row = self.stripe(.mix_o, i);
            if (comptime build_options.use_vulkan_ops) {
                // The GdnDecode kernel, bit-checked against `gdn.step` by `gpu-selftest`.
                ops.gdnStep(
                    @intCast(d.n_k_heads),
                    @intCast(d.n_v_heads),
                    @intCast(d.head_dim),
                    @intCast(d.conv_kernel),
                    qkv_row,
                    z_row,
                    a_row,
                    b_row,
                    conv1d,
                    ssm_a,
                    dt_bias,
                    ssm_norm,
                    eps,
                    state,
                    conv_state,
                    o_row,
                );
            } else {
                parallel.gdnStep(self.pool, self.policy, d, qkv_row, conv1d, conv_state, .{
                    .qkv = qkv_row,
                    .z = z_row,
                    .a = a_row,
                    .b = b_row,
                    .ssm_a = ssm_a,
                    .dt_bias = dt_bias,
                    .ssm_norm = ssm_norm,
                    .rms_eps = eps,
                    .state = state,
                    .o_out = o_row,
                });
            }
        }
    }

    // -- helpers ------------------------------------------------------------

    /// The tensor of `weight` in the model layer `layer`.
    fn tensor(self: *const CpuLayerKernels, layer: u32, weight: Weight) *const Tensor {
        assert(layer < self.card.n_layers);
        assert(layer < self.layers.len);

        return self.layers[layer].get(weight).?;
    }

    fn f32Weight(self: *const CpuLayerKernels, layer: u32, weight: Weight) []const f32 {
        return tensor_lookup.fp32View(self.tensor(layer, weight).*);
    }

    /// The KV region of a full-attention layer.
    fn kvIndex(self: *const CpuLayerKernels, layer: u32) u32 {
        assert(layer < self.kv_index.len);
        assert(self.kv_slots.len >= 1);

        const index = self.kv_index[layer].?;
        assert(index < self.kv_slots[0].n_layers);
        return index;
    }

    fn width(self: *const CpuLayerKernels, act: Activation) usize {
        const w = self.widths.get(act);
        assert(w >= 1);
        return w;
    }

    /// The stripe of step row `i` in the scratch of `act`.
    fn stripe(self: *const CpuLayerKernels, act: Activation, i: usize) []f32 {
        assert(i < self.rows.len);

        const w = self.width(act);
        return self.scratch.get(act)[i * w ..][0..w];
    }

    /// The GDN shape of the model card, as the runner derives it at init.
    fn gdnDims(self: *const CpuLayerKernels) GdnDims {
        const g = self.card.gdn.?;
        return .{
            .n_k_heads = g.group_count,
            .n_v_heads = g.time_step_rank,
            .head_dim = g.state_size,
            .conv_kernel = g.conv_kernel,
        };
    }
};
