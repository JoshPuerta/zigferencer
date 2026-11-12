//! The layer kernel set of a fused runner. Each method records the dispatches of one layer op into
//! the command buffer of one step, so a decoder layer records a whole layer through it.
//!
//! Holds. A value points at what its runner owns and owns nothing: the command buffer, the profile,
//! the pipelines, the `OpResources`, the binding table, the KV store, the commit scratch, and the
//! attention kernel set with the plan of the step. The runner builds one value for each record
//! call, so no value keeps step state from one recording to the next. The file reads no runner, so
//! both families build it.
//!
//! Op resources. `OpResources` owns the kernels and the scratch of the ops that a family can lack:
//! the QK norm, the attention gate, and the GDN. It builds each for an op that the forward of the
//! family calls, so the runner names none of these ops.
//!
//! Barriers. An op records its dispatches, then one barrier, then its mark. The KV commit keeps the
//! barrier between its narrowing and its copies, and it closes with a transfer-to-shader barrier.
//! The attention keeps the barriers of `attn_record` between its split passes and merges. The GDN
//! of a prefill chunk records a barrier between its conv and its scan.
//!
//! Marks. An op of one pass marks its op. An op of two passes marks each pass with its `OpPart`,
//! after the barrier that closes the pass. Three ops can hold two passes: the KV commit at a narrow
//! KV dtype, the GDN of a prefill chunk, and the attention of a row past one split. `attn_record`
//! marks the attention passes up to the merges of the last record group.
//!
//! Step kinds. `kind` picks the step input: one position, the step rows, or a prefill chunk.
//! Decode records the commands of a batch of one step row at slot 0, except in `norm`, `rope`, and
//! `attention`. So those three switch on the kind, and `gdn` and `kvCommit` switch for a prefill
//! chunk alone. The other methods loop over the step rows.
//!
//! Matmul. The row count picks the kernel: the reduce kernel for one row, the GEMM from
//! `gemm_min_batch` rows when the value holds one, and the batch kernel for the other counts. The
//! weight quant picks the Q8_0 or the Q4_0 form of the reduce and the batch kernel. The GEMM has
//! no Q4_0 form, so a Q4_0 value holds no GEMM.
//!
//! Bind pass. At init, `bindLayers` runs the forward of each layer through a private kernel set.
//! Each of its methods makes the sets of its op for all three step kinds. A count pass over the
//! same methods sizes the descriptor pool first, so the pool holds the sets exactly. The ops that
//! bind no layer buffer share one set across the layers: the rope, the gate, the residual, and the
//! SwiGLU. The bind pass writes and the record pass reads the sets of each op through one check.
//!
//! Comptime. `kind` selects the step input and the prongs of the methods that differ by kind. The
//! bind pass takes the decoder layer file, whose `forward` it runs.
//!
//! Layer index. Each method takes the model index of the layer, and it reads the tables of the
//! value at `layer - first_layer`.

const std = @import("std");
const build_options = @import("build_options");
const vk = @import("vulkan");

const vulkan_ctx = @import("vulkan_ctx.zig");
const loader = @import("../io/gguf/loader.zig");
const naive = @import("../ops/naive.zig");
const kernels = @import("../ops_vulkan/kernels.zig");
const attn_split = @import("../core/attn_split.zig");
const attn_record = @import("attn_record.zig");
const gpu_profile = @import("gpu_profile.zig");
const fused_commands = @import("fused_commands.zig");
const layer_kernels = @import("layer_kernels.zig");
const layer_op = @import("../core/layer_op.zig");
const core_log = @import("../core/utils/log.zig");
const StepRow = @import("step_row.zig").StepRow;
const KvTensor = @import("../core/kv_index.zig").Tensor;

const Activation = layer_kernels.Activation;
const LayerBinding = layer_kernels.LayerBinding;
const LayerOp = layer_kernels.LayerOp;
const OpPart = layer_op.OpPart;
const Projection = layer_kernels.Projection;
const StepKind = layer_kernels.StepKind;
const Weight = layer_kernels.Weight;
const WeightQuant = layer_kernels.WeightQuant;
const op_sets_max = layer_kernels.op_sets_max;
const assert = std.debug.assert;

const log_dispatch = std.log.scoped(.dispatch);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/fused_layer_kernels.zig was included in a non-vulkan build.");
    }
}

// -- constants --------------------------------------------------------------

/// The row count where the tiled GEMM beats the single-wave batch kernel. The GEMM computes a full
/// 64-row tile, so it amortizes its weight reads only once the rows fill that height.
pub const gemm_min_batch: u32 = 64;

comptime {
    // One set binds each projection of a `project` call.
    assert(layer_kernels.projections_max <= op_sets_max);
}

// -- types ------------------------------------------------------------------

/// The layer kernel set of the step entry `kind`.
pub fn FusedLayerKernels(comptime kind: StepKind) type {
    return struct {
        const Self = @This();

        /// The input of one runner call of `kind`.
        pub const Step = switch (kind) {
            .decode => struct { pos: u32 },
            .decode_batch => struct { rows: []const StepRow },
            .prefill_chunk => struct { b: u32, start_pos: u32 },
        };

        card: *const loader.ModelCard,
        vkd: *const vk.DeviceWrapper,
        cb: vk.CommandBuffer,
        prof: *gpu_profile.GpuProfile,
        /// The model index of `bindings[0]`.
        first_layer: u32,
        /// The sets of each layer of the range, keyed by op.
        bindings: []const LayerBinding,
        /// The KV region of each layer of the range. Null: the layer holds no KV.
        kv_index: []const ?u32,
        /// The elements of each activation in one step row. Zero: no buffer.
        widths: *const std.EnumArray(Activation, u32),
        /// The state and conv elements of one slot of a recurrent layer.
        recurrent: layer_kernels.RecurrentElems,

        /// The quant of every matmul weight of the range.
        quant: WeightQuant,
        k_linear: *kernels.LinearQ80Reduce,
        k_linear_batch: *kernels.LinearQ80Batch,
        /// Null in a family without Q4_0 weights.
        k_linear_q4: ?*kernels.LinearQ40Reduce,
        k_linear_batch_q4: ?*kernels.LinearQ40Batch,
        /// Null: the batch kernel serves every row count past one.
        k_linear_gemm: ?*kernels.LinearQ80Gemm,
        k_rmsnorm: *kernels.Rmsnorm,
        k_rmsnorm_batch: *kernels.RmsnormBatch,
        k_rope: *kernels.RopeMulti,
        k_rope_batch: *kernels.RopeMultiBatch,
        k_swiglu: *kernels.Swiglu,
        k_residual: *kernels.ResidualAdd,
        /// The kernels of the QK norm, the gate, and the GDN.
        op_resources: *OpResources,

        kv: *const kernels.KvStore,
        /// The narrowing kernel and the scratch pairs of the KV commit.
        commit: fused_commands.KvCommit,
        attn: kernels.AttnKernels,
        /// The one merge set of the split attention.
        attn_merge: vk.DescriptorSet,
        /// The attention plan of the step. Every full-attention layer records it.
        attn_plan: attn_split.AttnPlan,
        attn_shape: kernels.AttnShape,
        step: Step,

        /// Record the RMS norm of `x` into `x_norm` with the norm weight that the set of `op`
        /// binds.
        pub fn norm(self: *Self, layer: u32, op: LayerOp, weight: Weight) void {
            assert(op == .attn_norm or op == .ffn_norm);
            assert(weight == .attn_norm or weight == .ffn_norm);

            const set = self.oneSet(layer, op);
            const eps = self.card.rms_norm_eps;
            const hidden = self.width(.x);
            switch (kind) {
                .decode => self.k_rmsnorm.recordInto(self.cb, set, hidden, eps),
                .decode_batch, .prefill_chunk => {
                    self.k_rmsnorm_batch.recordInto(self.cb, set, self.rowCount(), hidden, eps);
                },
            }

            self.closeShader(op, null);
        }

        /// Record one matmul for each projection of `outs`, all of them over `in`.
        pub fn project(
            self: *Self,
            layer: u32,
            op: LayerOp,
            in: Activation,
            outs: []const Projection,
        ) void {
            assert(outs.len >= 1);
            assert(outs.len <= layer_kernels.projections_max);

            const sets = self.opSets(layer, op, outs.len);
            for (outs, sets[0..outs.len]) |p, set| {
                self.recordMatmul(p.weight, set, self.width(in), self.width(p.out));
            }

            self.closeShader(op, null);
        }

        /// Record the RMS norm of each head of q and k for each step row. The q pass splits
        /// `q_gate` into q and the attention gate, and the k pass norms in place.
        pub fn qkNorm(self: *Self, layer: u32) void {
            // Read the q set, the k set, and the row widths of the q pass and the k pass.
            const sets = self.opSets(layer, .qk_norm, 2);
            const k_qknorm = &self.op_resources.qk_norm.?;
            const c = self.card;
            const q_in = self.width(.q_gate);
            const q_out = self.width(.q);
            const k_row = self.width(.k);
            assert(q_in == 2 * q_out);

            // Record the q pass, then the k pass, of each step row at the offsets of the row.
            var t: u32 = 0;
            while (t < self.rowCount()) : (t += 1) {
                k_qknorm.recordRow(
                    self.cb,
                    sets[0],
                    c.n_heads,
                    c.head_dim,
                    2 * c.head_dim,
                    c.head_dim,
                    true,
                    c.rms_norm_eps,
                    t * q_in,
                    t * q_out,
                );
                k_qknorm.recordRow(
                    self.cb,
                    sets[1],
                    c.n_kv_heads,
                    c.head_dim,
                    c.head_dim,
                    0,
                    false,
                    c.rms_norm_eps,
                    t * k_row,
                    t * k_row,
                );
            }

            self.closeShader(.qk_norm, null);
        }

        /// Rotate q and k of each step row at the position of the row.
        pub fn rope(self: *Self, layer: u32, style: naive.RopeStyle) void {
            const sets = self.opSets(layer, .rope, 2);
            const c = self.card;
            const heads = [_]u32{ c.n_heads, c.n_kv_heads };

            // Record q and k by step kind: one position, one per row, or a stride over the chunk.
            switch (kind) {
                .decode => {
                    const pos = self.step.pos;
                    for (heads, sets[0..2]) |n_heads, set| {
                        self.k_rope.recordInto(
                            self.cb,
                            set,
                            pos,
                            n_heads,
                            c.head_dim,
                            c.rope_dim_count,
                            c.rope_freq_base,
                            style,
                        );
                    }
                },
                .decode_batch => {
                    // Each row holds its own position, so each row records q and then k as a
                    // one-row dispatch at buffer row `i`.
                    var i: u32 = 0;
                    while (i < self.rowCount()) : (i += 1) {
                        for (heads, sets[0..2]) |n_heads, set| {
                            self.k_rope_batch.recordInto(
                                self.cb,
                                set,
                                self.rowPos(i),
                                n_heads,
                                c.head_dim,
                                c.rope_dim_count,
                                c.rope_freq_base,
                                1,
                                0,
                                i,
                                style,
                            );
                        }
                    }
                },
                .prefill_chunk => {
                    // The position stride 1 puts chunk row `i` at `start_pos + i`.
                    for (heads, sets[0..2]) |n_heads, set| {
                        self.k_rope_batch.recordInto(
                            self.cb,
                            set,
                            self.step.start_pos,
                            n_heads,
                            c.head_dim,
                            c.rope_dim_count,
                            c.rope_freq_base,
                            self.rowCount(),
                            1,
                            0,
                            style,
                        );
                    }
                },
            }

            // Close the pass with a barrier and a profile mark.
            self.closeShader(.rope, null);
        }

        /// Narrow k and v of each step row to the KV dtype. Then copy each row to the row of its
        /// slot and position in the KV regions of the layer.
        pub fn kvCommit(self: *Self, layer: u32) void {
            const r = self.rangeIndex(layer);
            assert(self.kv_index[r] != null);

            const index = self.kv_index[r].?;
            const layout = self.kv.layout;
            const row_bytes: vk.DeviceSize = layout.rowBytes();
            const n = self.rowCount();
            assert(index < layout.n_layers);
            assert(row_bytes == self.commit.kv_dtype.rowBytes(self.commit.row_elems));

            // The narrowing closes on its own barrier, so the copies form a second pass.
            const src = fused_commands.recordKvCommit(self.vkd.*, self.cb, self.commit, n);
            const narrowed = self.commit.kv_dtype != .f32;
            if (narrowed) {
                self.prof.mark(self.cb, .initOne(.kv_commit), .narrow);
            }

            switch (kind) {
                .decode, .decode_batch => {
                    // Each row lands at the position of its own slot, so each tensor copies one
                    // region per row. The regions of a tensor lie in one buffer.
                    self.scatterRows(src.k, index, .k);
                    self.scatterRows(src.v, index, .v);
                },
                .prefill_chunk => {
                    // The chunk rows sit at consecutive positions of one slot, so one copy per
                    // tensor moves them all.
                    const slot = self.rowSlot(0);
                    const pos = self.rowPos(0);
                    assert(slot < layout.n_slots);
                    assert(self.rowPos(n - 1) < layout.window);

                    const bytes = @as(vk.DeviceSize, n) * row_bytes;
                    const k_rows = self.kv.rowAt(index, .k, slot, pos);
                    const v_rows = self.kv.rowAt(index, .v, slot, pos);
                    self.copyRows(src.k, 0, k_rows, bytes);
                    self.copyRows(src.v, 0, v_rows, bytes);
                },
            }

            fused_commands.barrierTransferToShader(self.vkd.*, self.cb);
            self.prof.mark(self.cb, .initOne(.kv_commit), if (narrowed) .copy else null);
        }

        /// Record the attention of each step row over the positions `0..=pos` of its slot.
        pub fn attention(self: *Self, layer: u32) void {
            assert(self.kv_index[self.rangeIndex(layer)] != null);
            assert(self.attn_merge != .null_handle);

            // The resident rows read the set of the kind. A row past one split reads its slot
            // through the whole-region set in every kind, so one split shader serves all three.
            const sets = self.opSets(layer, .attention, 3);
            const binding: kernels.AttnKernels.Binding = .{
                .shape = self.attn_shape,
                .resident = switch (kind) {
                    .decode => .{ .single = sets[0] },
                    .decode_batch => .{ .multi = sets[1] },
                    .prefill_chunk => .{ .batch = sets[0] },
                },
                .split = sets[2],
                .merge = self.attn_merge,
            };

            const Attn = kernels.AttnKernels;
            const plan = &self.attn_plan;
            switch (kind) {
                .decode, .decode_batch => {
                    attn_record.record(Attn, self.attn, self.cb, plan, binding);
                },
                .prefill_chunk => {
                    const start = self.step.start_pos;
                    attn_record.recordPrefill(Attn, self.attn, self.cb, plan, binding, start);
                },
            }

            // The recorder marks every pass but the merges of the last record group.
            self.closeShader(.attention, if (plan.group_count > 0) .merge else null);
        }

        /// Multiply `attn_out` by the sigmoid of the attention gate, in place.
        pub fn gate(self: *Self, layer: u32) void {
            const set = self.oneSet(layer, .attn_gate);
            const width_out = self.width(.attn_out);
            assert(width_out == self.width(.gate));

            self.op_resources.gate.?.recordInto(self.cb, set, self.rowCount() * width_out);
            self.closeShader(.attn_gate, null);
        }

        /// Add `src` to `x`, in place.
        pub fn residual(self: *Self, layer: u32, op: LayerOp, src: Activation) void {
            assert(op == .mixer_residual or op == .ffn_residual);
            assert(self.width(src) == self.width(.x));

            const set = self.oneSet(layer, op);
            self.k_residual.recordInto(self.cb, set, self.rowCount() * self.width(.x));
            self.closeShader(op, null);
        }

        /// Write `silu(ffn_gate) * ffn_up` into `ffn_gate`.
        pub fn swiglu(self: *Self, layer: u32) void {
            const set = self.oneSet(layer, .swiglu);
            const width_ffn = self.width(.ffn_gate);
            assert(width_ffn == self.width(.ffn_up));

            self.k_swiglu.recordInto(self.cb, set, self.rowCount() * width_ffn);
            self.closeShader(.swiglu, null);
        }

        /// Run the conv, the delta rule, and the gated norm of a GDN layer into `mix_o`. The
        /// state and the conv ring of each slot update in place.
        pub fn gdn(self: *Self, layer: u32) void {
            // Read the sets of a recurrent layer, which holds no KV.
            assert(self.kv_index[self.rangeIndex(layer)] == null);
            const sets = self.opSets(layer, .gdn, 3);
            const g = self.card.gdn.?;
            const eps = self.card.rms_norm_eps;
            const n = self.rowCount();

            // Record by step kind: runs of one slot for decode, the conv and the scan for prefill.
            switch (kind) {
                .decode, .decode_batch => {
                    // Record one dispatch per same-slot run. Runs touch disjoint state and rows,
                    // so no barrier sits between them. The run loop ends after `rowCount` rows.
                    const k_gdn = &self.op_resources.gdn.?;
                    var start: u32 = 0;
                    while (start < n) {
                        const slot = self.rowSlot(start);
                        var end: u32 = start + 1;
                        while (end < n and self.rowSlot(end) == slot) {
                            end += 1;
                        }

                        k_gdn.recordRun(
                            self.cb,
                            sets[0],
                            g.group_count,
                            g.time_step_rank,
                            g.state_size,
                            g.conv_kernel,
                            eps,
                            end - start,
                            start,
                            slot * self.recurrent.state,
                            slot * self.recurrent.conv,
                        );
                        start = end;
                    }
                },
                .prefill_chunk => {
                    // The scan reads the q, k, and v that the conv writes, so a barrier and the
                    // conv mark sit between them.
                    const k_chunk = &self.op_resources.gdn_chunk.?;
                    k_chunk.recordConv(
                        self.cb,
                        sets[1],
                        g.group_count,
                        g.time_step_rank,
                        g.state_size,
                        g.conv_kernel,
                        n,
                        eps,
                    );
                    fused_commands.barrierShader(self.vkd.*, self.cb);
                    self.prof.mark(self.cb, .initOne(.gdn), .conv);
                    k_chunk.recordScan(
                        self.cb,
                        sets[2],
                        g.group_count,
                        g.time_step_rank,
                        g.state_size,
                        n,
                        eps,
                    );
                },
            }

            // Close the pass with a barrier and a profile mark.
            self.closeShader(.gdn, if (kind == .prefill_chunk) .scan else null);
        }

        /// The step rows of the value.
        fn rowCount(self: *const Self) u32 {
            const n: u32 = switch (kind) {
                .decode => 1,
                .decode_batch => @intCast(self.step.rows.len),
                .prefill_chunk => self.step.b,
            };
            assert(n >= 1);
            assert(n <= attn_split.max_plan_rows);
            return n;
        }

        /// The slot of step row `i`. Decode and a prefill chunk run on slot 0.
        fn rowSlot(self: *const Self, i: u32) u32 {
            assert(i < self.rowCount());

            return switch (kind) {
                .decode, .prefill_chunk => 0,
                .decode_batch => self.step.rows[i].slot,
            };
        }

        /// The position of step row `i`.
        fn rowPos(self: *const Self, i: u32) u32 {
            assert(i < self.rowCount());

            return switch (kind) {
                .decode => self.step.pos,
                .decode_batch => self.step.rows[i].pos,
                .prefill_chunk => self.step.start_pos + i,
            };
        }

        /// The index of the model layer `layer` in the tables of the range.
        fn rangeIndex(self: *const Self, layer: u32) u32 {
            assert(layer >= self.first_layer);
            assert(layer < self.card.n_layers);
            assert(self.kv_index.len == self.bindings.len);

            const r = layer - self.first_layer;
            assert(r < self.bindings.len);
            return r;
        }

        /// The sets of `op` in `layer`, which must hold `count` sets.
        fn opSets(
            self: *const Self,
            layer: u32,
            op: LayerOp,
            count: usize,
        ) *const [op_sets_max]vk.DescriptorSet {
            const sets = self.bindings[self.rangeIndex(layer)].getPtrConst(op);
            assertSets(sets, count);
            return sets;
        }

        /// The set of an op that binds one set.
        fn oneSet(self: *const Self, layer: u32, op: LayerOp) vk.DescriptorSet {
            return self.opSets(layer, op, 1)[0];
        }

        fn width(self: *const Self, act: Activation) u32 {
            const w = self.widths.get(act);
            assert(w >= 1);
            return w;
        }

        /// Record one matmul over the step rows. The row count picks the kernel, and the quant
        /// picks its form.
        fn recordMatmul(self: *Self, weight: Weight, set: vk.DescriptorSet, k: u32, n: u32) void {
            const rows = self.rowCount();
            const gemm = rows >= gemm_min_batch and self.k_linear_gemm != null;
            if (self.quant == .q4_0) assert(self.k_linear_gemm == null);
            if (comptime core_log.hot) {
                const name = if (rows == 1) "reduce" else if (gemm) "gemm" else "batch";
                const args = .{ weight, name, self.quant, rows, k, n };
                log_dispatch.debug("  matmul {t} {s} ({t}): b {d} k {d} n {d}", args);
            }

            if (rows == 1) {
                switch (self.quant) {
                    .q8_0 => self.k_linear.recordInto(self.cb, set, k, n),
                    .q4_0 => self.k_linear_q4.?.recordInto(self.cb, set, k, n),
                }
            } else if (gemm) {
                self.k_linear_gemm.?.recordInto(self.cb, set, rows, k, n);
            } else {
                switch (self.quant) {
                    .q8_0 => self.k_linear_batch.recordInto(self.cb, set, rows, k, n),
                    .q4_0 => self.k_linear_batch_q4.?.recordInto(self.cb, set, rows, k, n),
                }
            }
        }

        /// Copy row `i` of `src` to the row of step row `i` in the KV region `(index, t)`, with one
        /// copy command that holds one region per step row.
        fn scatterRows(self: *Self, src: vk.Buffer, index: u32, t: KvTensor) void {
            const layout = self.kv.layout;
            const row_bytes: vk.DeviceSize = layout.rowBytes();
            const dst = self.kv.region(index, t).buffer;
            var regions_buf: [attn_split.max_plan_rows]vk.BufferCopy = undefined;
            const regions = regions_buf[0..self.rowCount()];
            for (regions, 0..) |*region, i| {
                const slot = self.rowSlot(@intCast(i));
                const pos = self.rowPos(@intCast(i));
                assert(slot < layout.n_slots);
                assert(pos < layout.window);

                const row = self.kv.rowAt(index, t, slot, pos);
                assert(row.buffer == dst);
                region.* = .{
                    .src_offset = @as(vk.DeviceSize, i) * row_bytes,
                    .dst_offset = row.offset,
                    .size = row_bytes,
                };
            }

            self.vkd.cmdCopyBuffer(self.cb, src, dst, regions);
        }

        fn copyRows(
            self: *Self,
            src: vk.Buffer,
            src_offset: vk.DeviceSize,
            dst: kernels.KvStore.Row,
            bytes: vk.DeviceSize,
        ) void {
            fused_commands.copyDeviceToDevice(
                self.vkd.*,
                self.cb,
                src,
                src_offset,
                dst.buffer,
                dst.offset,
                bytes,
            );
        }

        /// Close an op, or its last pass `part`: one compute barrier, then the mark.
        fn closeShader(self: *Self, op: LayerOp, part: ?OpPart) void {
            fused_commands.barrierShader(self.vkd.*, self.cb);
            self.prof.mark(self.cb, .initOne(op), part);
        }
    };
}

/// The kernels and the scratch of the layer ops that a family can lack: the QK norm, the attention
/// gate, and the GDN. `init` builds the kernels of an op that the forward of the family calls.
pub const OpResources = struct {
    /// Null: the forward of the family calls no QK norm.
    qk_norm: ?kernels.QwenQkNorm,
    /// Null: the forward of the family calls no attention gate.
    gate: ?kernels.SigmoidGate,
    /// The recurrence of the step rows. Null: the forward calls no GDN.
    gdn: ?kernels.GdnDecode,
    /// The conv and the scan of a prefill chunk. Null: the forward calls no GDN.
    gdn_chunk: ?kernels.GdnChunkParallel,
    /// The q, k, and v rows that the conv of a GDN prefill chunk writes for its scan. Empty when
    /// the forward calls no GDN.
    chunk_rows: [3]kernels.DeviceBuffer,

    pub const InitError = kernels.QwenQkNorm.InitError ||
        kernels.SigmoidGate.InitError ||
        kernels.GdnDecode.InitError ||
        kernels.GdnChunkParallel.InitError ||
        error{NoSuitableMemoryType} ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError;

    /// Build the kernels of the ops that the forward of `DecoderLayer` calls on `card`, and the
    /// GDN chunk rows at `rows` step rows.
    pub fn init(
        comptime DecoderLayer: type,
        ctx: *const vulkan_ctx.Context,
        card: *const loader.ModelCard,
        rows: u32,
    ) InitError!OpResources {
        assert(rows >= 1);
        assert(rows <= attn_split.max_plan_rows);

        const ops = layer_kernels.calledOps(DecoderLayer, card);
        var self: OpResources = .{
            .qk_norm = null,
            .gate = null,
            .gdn = null,
            .gdn_chunk = null,
            .chunk_rows = @splat(.{ .ctx = ctx }),
        };
        errdefer self.deinit();

        if (ops.contains(.qk_norm)) {
            self.qk_norm = try kernels.QwenQkNorm.init(ctx);
        }

        if (ops.contains(.attn_gate)) {
            self.gate = try kernels.SigmoidGate.init(ctx);
        }

        if (ops.contains(.gdn)) {
            self.gdn = try kernels.GdnDecode.init(ctx);
            self.gdn_chunk = try kernels.GdnChunkParallel.init(ctx);
            for (&self.chunk_rows, chunkWidths(card)) |*buffer, width| {
                try buffer.ensure(@as(u64, rows) * width * @sizeOf(f32));
            }
        }

        return self;
    }

    /// Free every kernel and buffer, in reverse construction order. An empty entry frees nothing.
    pub fn deinit(self: *OpResources) void {
        for (&self.chunk_rows) |*buffer| {
            buffer.deinit();
        }

        if (self.gdn_chunk) |*k| {
            k.deinit();
        }

        if (self.gdn) |*k| {
            k.deinit();
        }

        if (self.gate) |*k| {
            k.deinit();
        }

        if (self.qk_norm) |*k| {
            k.deinit();
        }
    }

    /// The device bytes of the scratch that `init` allocates at `rows` step rows. The runner adds
    /// them to the scratch price of the planner.
    pub fn scratchBytes(
        comptime DecoderLayer: type,
        card: *const loader.ModelCard,
        rows: u64,
    ) u64 {
        if (!layer_kernels.calledOps(DecoderLayer, card).contains(.gdn)) return 0;

        var per_row: u64 = 0;
        for (chunkWidths(card)) |width| {
            per_row += width;
        }

        return rows * per_row * @sizeOf(f32);
    }

    /// The views of the chunk rows that the conv writes and the scan reads: q, k, then v.
    fn chunkViews(self: *const OpResources) [3]vk.DescriptorBufferInfo {
        var views: [3]vk.DescriptorBufferInfo = undefined;
        for (&views, &self.chunk_rows) |*view, *buffer| {
            assert(buffer.capacity >= 1);
            view.* = .{ .buffer = buffer.buffer, .offset = 0, .range = buffer.capacity };
        }

        return views;
    }
};

/// The buffers of one weight that a set binds: the scales and the quants of a matrix, or the one
/// f32 buffer of a vector.
pub const WeightBuffers = union(enum) {
    matrix: *const kernels.Q80Weight,
    vector: vk.DescriptorBufferInfo,
};

/// The buffers of one layer of the range that its sets bind.
pub const LayerBuffers = struct {
    /// Null: the layer holds no such weight.
    weights: std.EnumArray(Weight, ?WeightBuffers),
    /// The recurrent state and the conv ring of every slot. Null: the layer holds no such state.
    state: ?vk.DescriptorBufferInfo,
    conv: ?vk.DescriptorBufferInfo,
};

/// What the bind pass binds: the buffers and the pipelines of one fused runner, which owns them.
pub const BindInput = struct {
    card: *const loader.ModelCard,
    ctx: *const vulkan_ctx.Context,
    /// The model index of `layers[0]`.
    first_layer: u32,
    /// The buffers of each layer of the range.
    layers: []const LayerBuffers,
    /// The KV region of each layer of the range. Null: the layer holds no KV.
    kv_index: []const ?u32,
    /// The elements of each activation in one step row. Zero: no buffer.
    widths: *const std.EnumArray(Activation, u32),
    /// The buffer of each activation. Null: no buffer.
    acts: std.EnumArray(Activation, ?vk.DescriptorBufferInfo),
    /// The divisor of each rotated pair, binding 1 of the rope shaders.
    rope_freqs: vk.DescriptorBufferInfo,
    /// The part records of the split attention.
    attn_records: vk.DescriptorBufferInfo,
    kv: *const kernels.KvStore,

    k_linear: *const kernels.LinearQ80Reduce,
    k_rmsnorm: *const kernels.Rmsnorm,
    k_rope: *const kernels.RopeMulti,
    k_swiglu: *const kernels.Swiglu,
    k_residual: *const kernels.ResidualAdd,
    /// The kernels of the QK norm, the gate, and the GDN, and the GDN chunk rows.
    op_resources: *const OpResources,
    attn: kernels.AttnKernels,
};

/// The descriptor sets and the storage descriptors that a pool must hold.
pub const SetCount = struct {
    sets: u32 = 0,
    descriptors: u32 = 0,
};

const AllocError = vk.DeviceWrapper.AllocateDescriptorSetsError;

/// The kernel set of the bind pass. Each method makes the sets of its op for all three step kinds,
/// then keys them in the binding of the layer. The count pass holds no pool, so it counts the sets
/// and makes none.
const BindKernels = struct {
    card: *const loader.ModelCard,
    input: *const BindInput,
    /// Null on the count pass.
    pool: ?vk.DescriptorPool,
    /// Empty on the count pass.
    bindings: []LayerBinding,
    made: SetCount = .{},
    /// The first failed allocation. The methods return void, so the pass returns it at the end.
    failed: ?AllocError = null,
    /// The sets of the ops that bind no layer buffer, made at their first layer and then shared.
    rope_sets: ?[2]vk.DescriptorSet = null,
    gate_set: ?vk.DescriptorSet = null,
    swiglu_set: ?vk.DescriptorSet = null,
    /// The residual sets, keyed by the activation that the op adds to `x`.
    residual_sets: std.EnumArray(Activation, ?vk.DescriptorSet) = .initFill(null),

    pub fn norm(self: *BindKernels, layer: u32, op: LayerOp, weight: Weight) void {
        assert(op == .attn_norm or op == .ffn_norm);

        const layout = self.input.k_rmsnorm.pipeline.set_layout;
        const w = self.vector(layer, weight);
        self.put(layer, op, &.{self.make(layout, &.{ self.act(.x), w, self.act(.x_norm) })});
    }

    pub fn project(
        self: *BindKernels,
        layer: u32,
        op: LayerOp,
        in: Activation,
        outs: []const Projection,
    ) void {
        assert(outs.len >= 1);
        assert(outs.len <= layer_kernels.projections_max);

        // The record pass reads the matmul shape from the widths, so each weight must hold it.
        const layout = self.input.k_linear.pipeline.set_layout;
        var sets: [layer_kernels.projections_max]vk.DescriptorSet = undefined;
        for (outs, sets[0..outs.len]) |p, *set| {
            const w = self.matrix(layer, p.weight);
            assert(w.in_features == self.input.widths.get(in));
            assert(w.out_features == self.input.widths.get(p.out));
            set.* = self.make(layout, &.{ self.act(in), w.scales, w.quants, self.act(p.out) });
        }

        self.put(layer, op, sets[0..outs.len]);
    }

    pub fn qkNorm(self: *BindKernels, layer: u32) void {
        // The k pass writes no gate, but its layout holds the gate binding too.
        const layout = self.input.op_resources.qk_norm.?.pipeline.set_layout;
        const gate_out = self.act(.gate);
        const q_norm = self.vector(layer, .attn_q_norm);
        const k_norm = self.vector(layer, .attn_k_norm);
        self.put(layer, .qk_norm, &.{
            self.make(layout, &.{ self.act(.q_gate), q_norm, self.act(.q), gate_out }),
            self.make(layout, &.{ self.act(.k), k_norm, self.act(.k), gate_out }),
        });
    }

    pub fn rope(self: *BindKernels, layer: u32, style: naive.RopeStyle) void {
        // A push constant picks the style, so the sets serve both styles.
        _ = style;
        if (self.rope_sets == null) {
            const layout = self.input.k_rope.pipeline.set_layout;
            const freqs = self.input.rope_freqs;
            self.rope_sets = .{
                self.make(layout, &.{ self.act(.q), freqs }),
                self.make(layout, &.{ self.act(.k), freqs }),
            };
        }

        self.put(layer, .rope, &self.rope_sets.?);
    }

    pub fn kvCommit(self: *BindKernels, layer: u32) void {
        // The commit narrows through the one commit set of the runner, so a layer binds no set.
        assert(self.kvIndex(layer) < self.input.kv.layout.n_layers);
    }

    pub fn attention(self: *BindKernels, layer: u32) void {
        // A set binds a KV region from its base, and a descriptor offset must be a multiple of
        // the offset alignment. So a region is a whole number of alignment units.
        const index = self.kvIndex(layer);
        const kv = self.input.kv;
        assert(index < kv.layout.n_layers);
        assert(kv.layout.regionBytes() % kv.lim.offset_alignment == 0);

        // Bind the slot 0 view of a decode step or a prefill chunk, the whole regions of the step
        // rows (their shader strides by slot), and the split passes.
        const k = kv.region(index, .k);
        const v = kv.region(index, .v);
        kernels.AttentionSplit.assertKBinding(k);
        kernels.AttentionSplit.assertVBinding(v);
        const q = self.act(.q);
        const out = self.act(.attn_out);
        const attn = self.input.attn;
        const k_slot = kv.slotView(index, .k, 0);
        const v_slot = kv.slotView(index, .v, 0);
        self.put(layer, .attention, &.{
            self.make(attn.single.pipeline.set_layout, &.{ q, k_slot, v_slot, out }),
            self.make(attn.multi.pipeline.set_layout, &.{ q, k, v, out }),
            self.make(attn.passes.split.set_layout, &.{ q, k, v, self.input.attn_records }),
        });
    }

    pub fn gate(self: *BindKernels, layer: u32) void {
        if (self.gate_set == null) {
            const layout = self.input.op_resources.gate.?.pipeline.set_layout;
            self.gate_set = self.make(layout, &.{ self.act(.attn_out), self.act(.gate) });
        }

        self.put(layer, .attn_gate, &.{self.gate_set.?});
    }

    pub fn residual(self: *BindKernels, layer: u32, op: LayerOp, src: Activation) void {
        assert(op == .mixer_residual or op == .ffn_residual);

        const set = self.residual_sets.getPtr(src);
        if (set.* == null) {
            const layout = self.input.k_residual.pipeline.set_layout;
            set.* = self.make(layout, &.{ self.act(src), self.act(.x) });
        }

        self.put(layer, op, &.{set.*.?});
    }

    pub fn swiglu(self: *BindKernels, layer: u32) void {
        if (self.swiglu_set == null) {
            const layout = self.input.k_swiglu.pipeline.set_layout;
            const ffn_gate = self.act(.ffn_gate);
            self.swiglu_set = self.make(layout, &.{ ffn_gate, self.act(.ffn_up), ffn_gate });
        }

        self.put(layer, .swiglu, &.{self.swiglu_set.?});
    }

    pub fn gdn(self: *BindKernels, layer: u32) void {
        // Read the buffers of a recurrent layer, which holds no KV.
        const r = self.rangeIndex(layer);
        assert(self.input.kv_index[r] == null);
        const buffers = &self.input.layers[r];
        const state = buffers.state.?;
        const conv = buffers.conv.?;
        const resources = self.input.op_resources;
        const chunk = resources.chunkViews();
        const conv1d = self.vector(layer, .ssm_conv1d);
        const a = self.vector(layer, .ssm_a);
        const dt = self.vector(layer, .ssm_dt);
        const norm_w = self.vector(layer, .ssm_norm);
        const qkv = self.act(.qkv);
        const z = self.act(.z);
        const alpha = self.act(.alpha);
        const beta = self.act(.beta);
        const mix_o = self.act(.mix_o);

        // Bind the step recurrence, then the chunk conv and scan. The chunk sets bind the state and
        // the conv ring of the step set, so a chunk leaves both as a step expects.
        const step_layout = resources.gdn.?.pipeline.set_layout;
        const k_chunk = &resources.gdn_chunk.?;
        self.put(layer, .gdn, &.{
            self.make(step_layout, &.{
                qkv, z, alpha, beta, conv1d, a, dt, norm_w, state, conv, mix_o,
            }),
            self.make(k_chunk.pipeline_conv.set_layout, &.{
                qkv, conv1d, conv, chunk[0], chunk[1], chunk[2],
            }),
            self.make(k_chunk.pipeline_scan.set_layout, &.{
                chunk[0], chunk[1], chunk[2], alpha, beta, a, dt, norm_w, z, state, mix_o,
            }),
        });
    }

    /// Make one set of `layout` over `infos`, and count it. The count pass makes none.
    fn make(
        self: *BindKernels,
        layout: vk.DescriptorSetLayout,
        infos: []const vk.DescriptorBufferInfo,
    ) vk.DescriptorSet {
        assert(layout != .null_handle);
        assert(infos.len >= 1);
        for (infos) |info| {
            assert(info.buffer != .null_handle);
            assert(info.range >= 1);
        }

        self.made.sets += 1;
        self.made.descriptors += @intCast(infos.len);
        const pool = self.pool orelse return .null_handle;
        if (self.failed != null) return .null_handle;

        const ctx = self.input.ctx;
        const layouts = [_]vk.DescriptorSetLayout{layout};
        const info: vk.DescriptorSetAllocateInfo = .{
            .descriptor_pool = pool,
            .descriptor_set_count = 1,
            .p_set_layouts = &layouts,
        };
        var set: vk.DescriptorSet = .null_handle;
        ctx.vkd.allocateDescriptorSets(ctx.device, &info, @ptrCast(&set)) catch |err| {
            self.failed = err;
            return .null_handle;
        };

        assert(set != .null_handle);
        kernels.updateStorageBindings(ctx, set, infos);
        return set;
    }

    /// Key `sets` as the sets of `op` in the binding of `layer`. The count pass keys none.
    fn put(self: *BindKernels, layer: u32, op: LayerOp, sets: []const vk.DescriptorSet) void {
        assert(sets.len >= 1);
        assert(sets.len <= op_sets_max);

        const r = self.rangeIndex(layer);
        if (self.pool == null or self.failed != null) return;

        // Forward makes each op call once in a layer, so the entry is still empty.
        const entry = self.bindings[r].getPtr(op);
        assertSets(entry, 0);
        @memcpy(entry[0..sets.len], sets);
        assertSets(entry, sets.len);
    }

    fn rangeIndex(self: *const BindKernels, layer: u32) u32 {
        assert(layer >= self.input.first_layer);

        const r = layer - self.input.first_layer;
        assert(r < self.input.layers.len);
        return r;
    }

    /// The KV region of a layer that holds KV.
    fn kvIndex(self: *const BindKernels, layer: u32) u32 {
        return self.input.kv_index[self.rangeIndex(layer)].?;
    }

    fn act(self: *const BindKernels, a: Activation) vk.DescriptorBufferInfo {
        assert(self.input.widths.get(a) >= 1);
        return self.input.acts.get(a).?;
    }

    fn vector(self: *const BindKernels, layer: u32, weight: Weight) vk.DescriptorBufferInfo {
        return self.input.layers[self.rangeIndex(layer)].weights.get(weight).?.vector;
    }

    fn matrix(self: *const BindKernels, layer: u32, weight: Weight) *const kernels.Q80Weight {
        return self.input.layers[self.rangeIndex(layer)].weights.get(weight).?.matrix;
    }
};

// -- api --------------------------------------------------------------------

/// Count the sets that `bindLayers` makes over the layers of `input`. The runner sizes its pool
/// with this count.
pub fn countLayerSets(comptime DecoderLayer: type, input: *const BindInput) SetCount {
    var binder: BindKernels = .{
        .card = input.card,
        .input = input,
        .pool = null,
        .bindings = &.{},
    };
    walk(DecoderLayer, &binder);

    assert(binder.failed == null);
    assert(binder.made.sets >= input.layers.len);
    return binder.made;
}

/// Make the sets of every layer of `input` from `pool`, and key them by op in `bindings`. The
/// count of the sets it made equals the count of `countLayerSets`.
pub fn bindLayers(
    comptime DecoderLayer: type,
    input: *const BindInput,
    pool: vk.DescriptorPool,
    bindings: []LayerBinding,
) AllocError!SetCount {
    assert(pool != .null_handle);
    assert(bindings.len == input.layers.len);

    // Check the buffers and the KV index of each layer against the rules of the family.
    for (input.layers, input.kv_index, 0..) |*buffers, index, r| {
        const layer = input.first_layer + @as(u32, @intCast(r));
        const rule = DecoderLayer.kvIndex(input.card, input.first_layer, layer);
        assert(std.meta.eql(index, rule));
        assertWeights(buffers, DecoderLayer.layerWeights(input.card, layer));
    }

    for (bindings) |*binding| {
        binding.* = .initFill(@splat(.null_handle));
    }

    var binder: BindKernels = .{
        .card = input.card,
        .input = input,
        .pool = pool,
        .bindings = bindings,
    };
    walk(DecoderLayer, &binder);
    if (binder.failed) |err| return err;

    return binder.made;
}

// -- helpers ----------------------------------------------------------------

/// The widths of the q, k, and v rows that the conv of a GDN prefill chunk writes for its scan.
fn chunkWidths(card: *const loader.ModelCard) [3]u32 {
    const g = card.gdn.?;
    const key_dim = g.group_count * g.state_size;
    return .{ key_dim, key_dim, g.time_step_rank * g.state_size };
}

/// Run the forward of each layer of the range through `binder`.
fn walk(comptime DecoderLayer: type, binder: *BindKernels) void {
    const input = binder.input;
    assert(input.layers.len == input.kv_index.len);
    assert(input.first_layer + input.layers.len <= input.card.n_layers);

    for (0..input.layers.len) |r| {
        DecoderLayer.forward(BindKernels, binder, input.first_layer + @as(u32, @intCast(r)));
    }
}

/// Check that a layer holds the buffers of the weight roles of its family rule, and of no other.
fn assertWeights(buffers: *const LayerBuffers, roles: []const Weight) void {
    var held: usize = 0;
    for (buffers.weights.values) |w| {
        if (w != null) {
            held += 1;
        }
    }

    assert(held == roles.len);
    for (roles) |role| {
        assert(buffers.weights.get(role) != null);
    }
}

/// Check that `sets` holds `count` sets, then null handles. The bind pass writes each op and the
/// record pass reads each op through this check, so the two agree on the sets of every op.
fn assertSets(sets: *const [op_sets_max]vk.DescriptorSet, count: usize) void {
    assert(count <= op_sets_max);
    for (sets, 0..) |set, i| {
        if (i < count) {
            assert(set != .null_handle);
        } else {
            assert(set == .null_handle);
        }
    }
}
