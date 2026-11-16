//! `CpuRunner(Family)` runs the forward pass of one family on the CPU over a bounded set of KV
//! slots. `Family` is the decoder layer file of the family: `llama_layer` or `qwen_layer`. The
//! runner holds the weights of each layer, the KV slots, the recurrent slabs, the scratch, and the
//! bookkeeping of each step. Every numerical op writes into a slice that the runner owns.
//!
//! Comptime. `assertFamily` checks `Family`, `findRole` names each weight role, and
//! `core_log.hot` removes the trace code.
//!
//! Family. What differs between two families comes from `Family`: the weight roles and their GGUF
//! names, the width of each activation, the KV index of each layer, the size of the recurrent
//! slabs, the rope factor tensor, and the forward of a layer. No line of this file reads the
//! architecture, except the refusal of a card of another family.
//!
//! Backend. `-Dops-backend=naive|simd|simd_pf|vulkan|auto` binds the op backend at build time. The
//! default `auto` picks the strongest eligible CPU variant. The runner and `CpuLayerKernels` call
//! the resolved module directly, so the hot path has no dispatch. The `vulkan` backend wraps the
//! GPU kernels behind the same op API. On a fused build this runner takes the CPU variant the build
//! resolved.
//!
//! Batched step. `stepBatch` advances a batch of rows, each row a (slot, token, pos) triple: decode
//! rows of independent sessions, or consecutive rows of one session (chunked prefill). Each layer
//! op runs over all rows before the next op starts. So a layer writes the KV of every row before
//! any row attends, and a projection streams each weight once per step. Every per-row op takes the
//! inputs of the solo call of its row, so a batch gives the bits of each row run alone.
//!
//! Recurrent slabs. `gdn_state` and `gdn_conv_state` hold one slice per (slot, layer), at
//! `slot * n_layers + layer`, with the sizes of `Family.recurrentElems`. A KV layer gets an empty
//! slice, and a family without recurrent state holds no slice. The slabs have no position axis, and
//! zero means a fresh sequence. A recurrent layer steps the state of the slot of its row. The row
//! rule keeps the rows of one slot adjacent and ascending, so batch order is time order.
//!
//! Prefix. A prefix runner holds the layers `[0..n_layers)` and no head: no final norm, no lm_head,
//! and no logits rows, because its rows never want logits. The family rules take the model index,
//! and a prefix starts at 0, so the model index is also the index of every table. The KV index of a
//! layer counts the KV layers below it. A prefix without a KV layer holds no KV, and its slot
//! vector is empty, because `KvCache` refuses a slab of zero layers.
//!
//! Scratch. Init sizes one buffer for each activation of the family at `max_rows` rows, from
//! `Family.activationWidth`. Row i of a step owns the i-th stripe of each. `max_rows` is decoupled
//! from `n_slots`, so a prefill chunk can be wider than the session count, and it is never
//! narrower. One row of attention part records serves every row, because the rows attend one at a
//! time. The per-op GPU attention keeps its own records, so that build holds none.
//!
//! Compact logits. Only the `want_logits` rows get an lm_head: the decode rows and the last row of
//! a prefill chunk. They number at most `n_slots`, so `logits` has `n_slots` rows. The j-th such
//! row lands at `logits[j * vocab_size ..]`, and `Sessions.advance` reads it at the same index.
//!
//! KV commit. The f32 scratch pipeline never learns the KV dtype. One conversion boundary narrows
//! the finished K and V rows at commit. The compiler vectorizes the narrowing into
//! hardware round-to-nearest-even with F16C. The recurrent slabs stay f32 under every KV
//! dtype, so the planner sizes them as one constant per slot.
//!
//! Blobs. A prefix snapshot moves the state of a slot to and from a host blob by memcpy. The blob
//! walks the layers in model order: the K rows, then the V rows of a KV layer, and the recurrent
//! slab, then the conv slab of a recurrent layer. A fused runner walks the same regions in the same
//! order, so a blob restores through another runner at the same KV dtype. The values can differ in
//! bits between two runners. The presence of `evictSlot` lights `Sessions.can_snapshot` and
//! `Residency.can_evict` for this runner.

const std = @import("std");
const build_options = @import("build_options");
const span = @import("../core/span.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const attn_cpu = @import("../core/attn_cpu.zig");
const kv_budget = @import("../core/kv_budget.zig");
const core_log = @import("../core/utils/log.zig");
const time = @import("../core/utils/time.zig");
const loader = @import("../io/gguf/loader.zig");
const KvCache = @import("kv_cache.zig").KvCache;
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const step_row = @import("step_row.zig");
const linear_dispatch = @import("linear_dispatch.zig");
const weight_check = @import("weight_check.zig");
const naive = @import("../ops/naive.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const layer_kernels = @import("layer_kernels.zig");
const CpuLayerKernels = @import("cpu_layer_kernels.zig").CpuLayerKernels;

const Allocator = std.mem.Allocator;
const Activation = layer_kernels.Activation;
const LayerOp = layer_kernels.LayerOp;
const StepRow = step_row.StepRow;
const Weight = layer_kernels.Weight;
const assert = std.debug.assert;

const fp32View = tensor_lookup.fp32View;
const findTensor = tensor_lookup.findRequired;
const findOptional = tensor_lookup.find;

/// The op backend that the build resolved. The header describes the choices.
const ops = if (build_options.use_vulkan_ops)
    @import("../ops_vulkan/wrap.zig")
else if (build_options.cpu_ops_simd_pf)
    @import("../ops/simd_pf.zig")
else if (build_options.cpu_ops_simd)
    @import("../ops/simd.zig")
else
    @import("../ops/naive.zig");

const log = std.log.scoped(.runner);
const log_layer = std.log.scoped(.layer);

// -- types ------------------------------------------------------------------

/// The weight tensors of one layer, keyed by role. Null: the layer holds no such role.
const LayerTensors = std.EnumArray(Weight, ?*const Tensor);

/// The forward-pass state of the family `Family` over `n_slots` independent KV slots. One slot is
/// the CLI shape.
pub fn CpuRunner(comptime Family: type) type {
    return struct {
        const Self = @This();

        card: loader.ModelCard,
        /// Row-parallel matmul and attention fan-out. The caller owns the pool and must outlive it.
        pool: *Pool,
        policy: parallel.Policy,

        embed: *const Tensor,
        /// The final norm. Empty in a prefix runner.
        output_norm: []const f32,
        /// The lm_head: the same pointer as `embed` when the model ties its embeddings. Null in a
        /// prefix runner.
        output: ?*const Tensor,
        /// The frequency divisor of each rotated pair in every layer. Null: the plain frequencies.
        rope_factors: ?[]const f32,
        /// The base angle of each rotated pair, `rope_dim_count / 2` entries.
        rope_thetas: []const f32,

        /// Layers this runner executes. Only a prefix runner narrows it below the card count.
        n_layers: u32,
        /// The weight tensors of each layer, keyed by role.
        layers: []LayerTensors,
        /// The KV region of each layer. Null: a recurrent layer.
        kv_index: []?u32,
        /// The elements of each activation buffer in one step row.
        activation_widths: std.EnumArray(Activation, u32),

        /// One KV slab per slot. Empty when the range holds no KV layer.
        kv_slots: []KvCache,
        /// The element type every KV row is framed at. One process holds one dtype.
        kv_dtype: dtype.KvDtype,
        n_slots: u32,
        /// Max rows in one `stepBatch`, the row-scratch capacity. At least `n_slots`.
        max_rows: u32,
        /// Max prompt rows in one `prefillBatch`: `max_rows`, clamped to the stack row array.
        prefill_batch_max: u32,
        max_seq: u32,

        // The recurrent slabs of each (slot, layer). The header describes the indexing.
        gdn_state: [][]f32,
        gdn_conv_state: [][]f32,

        /// The scratch of each activation, one stripe per step row. Empty: the family holds no such
        /// buffer.
        scratch: std.EnumArray(Activation, []f32),
        attn_parts: []f32,
        /// The compact logits rows, `n_slots` rows. Empty in a prefix runner.
        logits: []f32,

        /// What `init` and `initPrefix` take beside the arena, the model, and the layer count.
        pub const Options = struct {
            /// The slot count and the window.
            plan: kv_budget.KvPlan,
            /// The widest `stepBatch` of the caller. The row scratch holds at least the slots.
            max_rows: u32,
            /// The caller owns the pool, which must outlive the runner.
            pool: *Pool,
            policy: parallel.Policy,
            /// Flag resolution has gated the dtype against the model.
            kv_dtype: dtype.KvDtype,
        };

        pub const Error = step_row.Error || weight_check.Error;

        /// A reused slot inherits the previous conversation unless it is scrubbed. The session
        /// surface reads this marker and then needs `resetSlotState` and `cloneSlotState`.
        pub const has_recurrent_state: bool = Family.has_recurrent_state;

        /// The per-op GPU build keeps the per-token prompt loop that its token-hash gates anchor.
        pub const chunked_prefill: bool = !build_options.use_vulkan_ops;

        comptime {
            layer_kernels.assertFamily(Family);
        }

        // -- init -----------------------------------------------------------

        /// The whole-model runner: every layer, then the final norm and the lm_head.
        pub fn init(arena: Allocator, loaded: *const loader.Loaded, options: Options) Error!Self {
            return build(arena, loaded, loaded.card.n_layers, options, .with_head);
        }

        /// A prefix runner executes layers `[0..n_layers)` with weights, KV, and recurrent state
        /// sized to match, and no head. The hybrid runner composes through it.
        pub fn initPrefix(
            arena: Allocator,
            loaded: *const loader.Loaded,
            n_layers: u32,
            options: Options,
        ) Error!Self {
            return build(arena, loaded, n_layers, options, .no_head);
        }

        /// Every init funnels here. `.with_head` builds the final norm, the lm_head, and the logits
        /// rows.
        fn build(
            arena: Allocator,
            loaded: *const loader.Loaded,
            n_layers: u32,
            options: Options,
            parts: enum { with_head, no_head },
        ) Error!Self {
            const with_head = parts == .with_head;
            // Check the limits. Refuse a card of another family, a dtype no kernel reads, and a
            // weight that the card does not describe.
            const plan = options.plan;
            const n_slots: u32 = plan.n_slots;
            assert(n_slots >= 1);
            assert(n_slots <= limits.max_batch);
            assert(options.pool.threads() == @max(options.policy.n_threads, 1));

            const c = loaded.card;
            try weight_check.requireFamily(Family, &c);

            assert(n_layers >= 1);
            assert(n_layers <= c.n_layers);
            if (with_head) assert(n_layers == c.n_layers);
            try linear_dispatch.requireWeights(loaded.tensors, n_layers);
            try weight_check.checkWeights(Family, &c, loaded.tensors);
            assert(c.rope_dim_count >= 2);
            assert(c.rope_dim_count % 2 == 0);

            // The row scratch holds the widest stepBatch, and never fewer rows than the slots.
            const rows_cap: u32 = @max(n_slots, options.max_rows);
            assert(rows_cap <= limits.max_batch + limits.max_prefill_chunk);

            // Find the weights that no layer owns, then the weights of each layer.
            const refs = loaded.tensors;
            const embed = try findTensor(refs, loader.embed_tensor);
            const norm_name = loader.output_norm_tensor;
            const output_norm = if (with_head) try findTensor(refs, norm_name) else null;
            const layers = try arena.alloc(LayerTensors, n_layers);
            const kv_index = try arena.alloc(?u32, n_layers);
            const n_kv = try findLayers(refs, &c, layers, kv_index);

            // Allocate the recurrent slabs and the KV slots.
            const kv_dtype = options.kv_dtype;
            const elems = Family.recurrentElems(&c);
            const gdn_state = try allocSlabs(arena, n_slots, kv_index, elems.state);
            const gdn_conv_state = try allocSlabs(arena, n_slots, kv_index, elems.conv);
            const kv_slots = try allocKv(arena, &c, n_kv, plan, kv_dtype);

            // Size the scratch of each activation, and the logits rows of a runner with a head.
            const widths = activationWidths(&c);
            const logits_len: usize = if (with_head) @as(usize, n_slots) * c.vocab_size else 0;
            const rope_thetas = try arena.alloc(f32, c.rope_dim_count / 2);
            naive.fillRopeThetas(rope_thetas, c.rope_freq_base);
            const runner: Self = .{
                .card = c,
                .pool = options.pool,
                .policy = options.policy,
                .embed = embed,
                .output_norm = if (output_norm) |t| fp32View(t.*) else &.{},
                .output = if (with_head) headOf(refs, embed) else null,
                .rope_factors = ropeFactors(refs, &c),
                .rope_thetas = rope_thetas,
                .n_layers = n_layers,
                .layers = layers,
                .kv_index = kv_index,
                .activation_widths = widths,
                .kv_slots = kv_slots,
                .kv_dtype = kv_dtype,
                .n_slots = n_slots,
                .max_rows = rows_cap,
                .prefill_batch_max = @min(rows_cap, limits.max_prefill_chunk),
                .max_seq = plan.window,
                .gdn_state = gdn_state,
                .gdn_conv_state = gdn_conv_state,
                .scratch = try allocScratch(arena, &widths, rows_cap),
                .attn_parts = try arena.alloc(f32, attnPartsLen(&c)),
                .logits = try arena.alloc(f32, logits_len),
            };
            runner.logInit(n_kv);
            return runner;
        }

        /// Max rows one `stepBatch` accepts. The scheduler fills decode rows plus a prefill chunk.
        pub fn maxStepRows(self: *const Self) u32 {
            return self.max_rows;
        }

        /// The layers this runner holds: all of them, or the host layers of a hybrid.
        pub fn layerCount(self: *const Self) u32 {
            return self.n_layers;
        }

        // -- step -----------------------------------------------------------

        /// One forward step on slot 0: consume `token_id` at `pos`, and return `vocab_size`
        /// logits. The runner owns the slice and reuses it on the next call.
        pub fn step(self: *Self, token_id: u32, pos: u32) Error![]f32 {
            const row: StepRow = .{ .slot = 0, .token = token_id, .pos = pos, .want_logits = true };
            try self.stepBatch(&.{row});
            return self.logits[0..self.card.vocab_size];
        }

        /// Consume `ids` on slot 0 at `pos..pos + ids.len` in one batched step, and return the
        /// logits of the last row only. Callers chunk a longer prompt by `prefill_batch_max`.
        pub fn prefillBatch(self: *Self, ids: []const u32, pos: u32) Error![]f32 {
            assert(ids.len >= 1);

            if (ids.len > self.prefill_batch_max) return error.TooManyRows;

            var rows_buf: [limits.max_prefill_chunk]StepRow = undefined;
            try self.stepBatch(step_row.chunkRows(&rows_buf, ids, pos));
            const out = self.logits[0..self.card.vocab_size];
            assert(out.len == self.card.vocab_size);
            if (std.debug.runtime_safety) {
                for (out) |x| {
                    assert(std.math.isFinite(x));
                }
            }

            return out;
        }

        /// Advance every row one position: embed, all layers, then the final norm and the lm_head
        /// for the `want_logits` suffix only. The header describes the compact logits.
        pub fn stepBatch(self: *Self, rows: []const StepRow) Error!void {
            _ = try step_row.validateRows(rows, self.n_slots, self.max_rows, self.max_seq);
            log.debug(
                "step: {d} rows, slot {d} pos {d}..{d}",
                .{ rows.len, rows[0].slot, rows[0].pos, rows[rows.len - 1].pos },
            );

            // Embed each token.
            const d = self.card.embedding_length;
            const x = self.scratch.get(.x);
            const sp_embed = span.begin(.ofOps(.initOne(.embed)));
            for (rows, 0..) |r, i| {
                ops.embedLookup(self.embed.*, r.token, d, x[i * d ..][0..d]);
            }

            sp_embed.end();

            // Run all layers through one kernel set.
            const live = (comptime core_log.hot) and core_log.enabled(.layer, .debug);
            var kernels = self.layerKernels(rows);
            var layer: u32 = 0;
            while (layer < self.n_layers) : (layer += 1) {
                var t_layer: std.Io.Timestamp = undefined;
                if (live) {
                    t_layer = core_log.now();
                }

                runLayer(&kernels, layer);
                if (live) {
                    const kv = self.kv_index[layer] != null;
                    const kind: []const u8 = if (kv) "kv" else "recurrent";
                    const args = .{ layer, kind, rows.len, core_log.msSince(t_layer) };
                    log_layer.debug("layer {d} ({s}): {d} rows, {d:.2} ms", args);
                }
            }

            self.head(rows);
        }

        /// One decoder layer over all rows. It writes the KV of every row before any row attends,
        /// so consecutive rows of one slot see exactly their causal prefix. Public for the hybrid.
        pub fn layerForward(self: *Self, layer: u32, rows: []const StepRow) void {
            assert(layer < self.n_layers);

            var kernels = self.layerKernels(rows);
            runLayer(&kernels, layer);
        }

        /// The final norm and the lm_head of the `want_logits` suffix of `rows`, if any.
        fn head(self: *Self, rows: []const StepRow) void {
            var first: usize = rows.len;
            while (first > 0 and rows[first - 1].want_logits) {
                first -= 1;
            }

            if (first == rows.len) return;

            // Only a runner with a head takes a want-logits row.
            const output = self.output.?;
            assert(self.logits.len >= (rows.len - first) * @as(usize, self.card.vocab_size));

            // Norm each logits row, then project them in one batched pass.
            const d = self.card.embedding_length;
            const x = self.scratch.get(.x);
            const x_norm = self.scratch.get(.x_norm);
            const sp_norm = span.begin(.ofOps(.initOne(.final_norm)));
            for (rows[first..], first..) |_, i| {
                ops.rmsnorm(
                    x[i * d ..][0..d],
                    self.output_norm,
                    self.card.rms_norm_eps,
                    x_norm[i * d ..][0..d],
                );
            }

            sp_norm.end();
            const n_logits = rows.len - first;
            const sp_head = span.begin(.ofOps(.initOne(.lm_head)));
            linear_dispatch.linearBatchDispatch(
                self.pool,
                self.policy,
                x_norm[first * d .. rows.len * d],
                output.*,
                self.logits[0..],
                n_logits,
            );
            sp_head.end();
            assert(first + n_logits == rows.len);
        }

        /// The layer kernel set of one call over `rows`. It points at the state of the runner.
        fn layerKernels(self: *Self, rows: []const StepRow) CpuLayerKernels {
            assert(rows.len >= 1);
            assert(rows.len <= self.max_rows);

            return .{
                .card = &self.card,
                .pool = self.pool,
                .policy = self.policy,
                .rows = rows,
                .layers = self.layers,
                .rope_factors = self.rope_factors,
                .rope_thetas = self.rope_thetas,
                .kv_index = self.kv_index,
                .widths = &self.activation_widths,
                .scratch = self.scratch,
                .attn_parts = self.attn_parts,
                .kv_slots = self.kv_slots,
                .kv_dtype = self.kv_dtype,
                .gdn_state = self.gdn_state,
                .gdn_conv_state = self.gdn_conv_state,
            };
        }

        // -- slot state -----------------------------------------------------

        /// Zero the recurrent slabs of a slot for a fresh conversation. KV needs no scrub, because
        /// attention reads only rewritten positions. A family without recurrent state does nothing.
        pub fn resetSlotState(self: *Self, slot: u32) void {
            assert(slot < self.n_slots);

            if (comptime !has_recurrent_state) return;

            const base = @as(usize, slot) * self.n_layers;
            for (self.gdn_state[base..][0..self.n_layers]) |s| {
                @memset(s, 0);
            }

            for (self.gdn_conv_state[base..][0..self.n_layers]) |s| {
                @memset(s, 0);
            }
        }

        /// Copy slot state for a session fork: KV rows `0..pos` of the KV layers, and the WHOLE
        /// recurrent slabs, because they have no position axis. Rows past `pos` stay stale,
        /// because attention reads only `0..=pos`.
        pub fn cloneSlotState(self: *Self, source_slot: u32, target_slot: u32, pos: u32) void {
            assert(source_slot < self.n_slots);
            assert(target_slot < self.n_slots);
            assert(source_slot != target_slot);
            assert(pos <= self.max_seq);

            // A runner without a KV layer carries no KV to fork.
            if (self.kv_slots.len > 0) {
                self.kv_slots[target_slot].copyFrom(&self.kv_slots[source_slot], pos);
            }

            if (comptime !has_recurrent_state) return;

            const source_base = @as(usize, source_slot) * self.n_layers;
            const target_base = @as(usize, target_slot) * self.n_layers;
            for (0..self.n_layers) |li| {
                @memcpy(self.gdn_state[target_base + li], self.gdn_state[source_base + li]);
                @memcpy(
                    self.gdn_conv_state[target_base + li],
                    self.gdn_conv_state[source_base + li],
                );
            }
        }

        // -- tiered kv ------------------------------------------------------

        /// Bytes that the state of one slot occupies in a host blob at `n_rows` committed
        /// positions: two KV rows a position for a KV layer, the two slabs for a recurrent layer.
        pub fn evictSlotBytesFor(self: *const Self, n_rows: u32) u64 {
            const c = &self.card;
            const row_bytes: u64 = self.kv_dtype.rowBytes(c.n_kv_heads * c.head_dim);
            const elems = Family.recurrentElems(c);
            const slab_bytes: u64 = (@as(u64, elems.state) + elems.conv) * @sizeOf(f32);
            var total: u64 = 0;
            for (self.kv_index) |index| {
                total += if (index != null) @as(u64, 2) * n_rows * row_bytes else slab_bytes;
            }

            return total;
        }

        /// Copy the state of `slot` over its first `n_rows` positions into `out`.
        pub fn evictSlot(self: *Self, slot: u32, n_rows: u32, out: []u8) !void {
            assert(slot < self.n_slots);
            assert(n_rows >= 1); // a blob must hold rows to restore from
            assert(n_rows <= self.max_seq);
            assert(out.len == @as(usize, @intCast(self.evictSlotBytesFor(n_rows))));

            var off: usize = 0;
            var layer: u32 = 0;
            while (layer < self.n_layers) : (layer += 1) {
                for (self.slotRegions(slot, layer, n_rows)) |bytes| {
                    @memcpy(out[off..][0..bytes.len], bytes);
                    off += bytes.len;
                }
            }

            assert(off == out.len);
        }

        /// Restore a blob of `n_rows` rows that `evictSlot` wrote into `slot`.
        pub fn restoreSlot(self: *Self, slot: u32, n_rows: u32, blob: []const u8) !void {
            assert(slot < self.n_slots);
            assert(n_rows >= 1); // a blob must hold rows to restore from
            assert(n_rows <= self.max_seq);
            assert(blob.len == @as(usize, @intCast(self.evictSlotBytesFor(n_rows))));

            var off: usize = 0;
            var layer: u32 = 0;
            while (layer < self.n_layers) : (layer += 1) {
                for (self.slotRegions(slot, layer, n_rows)) |bytes| {
                    @memcpy(bytes, blob[off..][0..bytes.len]);
                    off += bytes.len;
                }
            }

            assert(off == blob.len);
        }

        /// The two regions of `layer` in a blob of `slot`, in walk order: the K rows, then the V
        /// rows of a KV layer, or the recurrent slab, then the conv slab of a recurrent layer.
        fn slotRegions(self: *Self, slot: u32, layer: u32, n_rows: u32) [2][]u8 {
            assert(slot < self.n_slots);
            assert(layer < self.n_layers);

            if (self.kv_index[layer]) |index| {
                const kv = &self.kv_slots[slot];
                return .{ kv.kRowsBytesMut(index, n_rows), kv.vRowsBytesMut(index, n_rows) };
            }

            // Only a family with recurrent state holds a layer without KV.
            assert(has_recurrent_state);
            const slab = @as(usize, slot) * self.n_layers + layer;
            return .{
                std.mem.sliceAsBytes(self.gdn_state[slab]),
                std.mem.sliceAsBytes(self.gdn_conv_state[slab]),
            };
        }

        // -- helpers --------------------------------------------------------

        /// One decoder layer through `kernels`, under the span of the layer.
        fn runLayer(kernels: *CpuLayerKernels, layer: u32) void {
            const sp = span.beginArg(.layer_forward, layer);
            defer sp.end();

            Family.forward(CpuLayerKernels, kernels, layer);
        }

        /// Find the weights of each role in each layer, and give each layer its KV index. It
        /// returns the count of KV layers.
        fn findLayers(
            refs: []const loader.TensorRef,
            card: *const loader.ModelCard,
            layers: []LayerTensors,
            kv_index: []?u32,
        ) Error!u32 {
            assert(layers.len == kv_index.len);
            assert(layers.len <= card.n_layers);

            var n_kv: u32 = 0;
            for (layers, kv_index, 0..) |*tensors, *index, li| {
                const layer: u32 = @intCast(li);
                tensors.* = .initFill(null);
                for (Family.layerWeights(card, layer)) |role| {
                    const ref = try tensor_lookup.findRole(Family, refs, role, layer);
                    tensors.set(role, &ref.tensor);
                }

                index.* = Family.kvIndex(card, 0, layer);
                if (index.*) |kv| {
                    assert(kv == n_kv); // the rule counts what the walk counts
                    n_kv += 1;
                }
            }

            return n_kv;
        }

        /// The lm_head: `output.weight`, or the embedding table of a model that ties its head.
        fn headOf(refs: []const loader.TensorRef, embed: *const Tensor) *const Tensor {
            return findOptional(refs, loader.head_tensor) orelse embed;
        }

        /// One KV slab of `n_kv` layers per slot. No KV layer means no slot vector, because
        /// `KvCache` refuses a slab of zero layers.
        fn allocKv(
            arena: Allocator,
            card: *const loader.ModelCard,
            n_kv: u32,
            plan: kv_budget.KvPlan,
            kv_dtype: dtype.KvDtype,
        ) Allocator.Error![]KvCache {
            const slots = try arena.alloc(KvCache, if (n_kv == 0) 0 else plan.n_slots);
            for (slots) |*slot| {
                slot.* = try KvCache.init(
                    arena,
                    n_kv,
                    card.n_kv_heads,
                    card.head_dim,
                    plan.window,
                    kv_dtype,
                );
            }

            return slots;
        }

        /// One zeroed slab of `len` floats for each (slot, recurrent layer), and an empty slice for
        /// each KV layer. A family without recurrent state gets no slice.
        fn allocSlabs(
            arena: Allocator,
            n_slots: u32,
            kv_index: []const ?u32,
            len: u32,
        ) Allocator.Error![][]f32 {
            const n_layers = kv_index.len;
            const n_slabs = if (has_recurrent_state) n_slots * n_layers else 0;
            const slabs = try arena.alloc([]f32, n_slabs);
            for (slabs, 0..) |*slab, i| {
                if (kv_index[i % n_layers] != null) {
                    slab.* = &.{};
                } else {
                    slab.* = try arena.alloc(f32, len);
                    @memset(slab.*, 0);
                }
            }

            return slabs;
        }

        /// The width of each activation of the family in one step row.
        fn activationWidths(card: *const loader.ModelCard) std.EnumArray(Activation, u32) {
            var widths: std.EnumArray(Activation, u32) = .initUndefined();
            var entries = widths.iterator();
            while (entries.next()) |entry| {
                entry.value.* = Family.activationWidth(card, entry.key);
            }

            assert(widths.get(.x) == card.embedding_length);
            assert(widths.get(.x_norm) == card.embedding_length);
            return widths;
        }

        /// The rope factors of the family tensor. Null: the family names none, or the file holds
        /// none.
        fn ropeFactors(
            refs: []const loader.TensorRef,
            card: *const loader.ModelCard,
        ) ?[]const f32 {
            const name = Family.rope_freqs orelse return null;
            const t = findOptional(refs, name) orelse return null;
            const factors = fp32View(t.*);
            assert(factors.len >= card.rope_dim_count / 2);
            return factors;
        }

        /// The elements of the attention part records of one row. The per-op GPU attention keeps
        /// its own records.
        fn attnPartsLen(card: *const loader.ModelCard) usize {
            if (build_options.use_vulkan_ops) return 0;

            return attn_cpu.partialLen(card.n_heads, card.head_dim);
        }

        fn logInit(self: *const Self, n_kv: u32) void {
            const row_bytes = self.kv_dtype.rowBytes(self.card.n_kv_heads * self.card.head_dim);
            const kv_bytes: u64 = 2 * @as(u64, self.n_slots) * n_kv * self.max_seq * row_bytes;
            const head_word: []const u8 = if (self.output != null) "with" else "no";
            log.debug(
                "init {t}: {d} layers ({d} kv), {s} head, {d} slots, max_seq {d}, kv {d:.0} MiB",
                .{
                    Family.arch,
                    self.n_layers,
                    n_kv,
                    head_word,
                    self.n_slots,
                    self.max_seq,
                    @as(f64, @floatFromInt(kv_bytes)) / (1024.0 * 1024.0),
                },
            );
        }
    };
}

// -- helpers ----------------------------------------------------------------

/// The scratch of each activation at `rows` step rows. A zero width gets an empty slice.
fn allocScratch(
    arena: Allocator,
    widths: *const std.EnumArray(Activation, u32),
    rows: u32,
) Allocator.Error!std.EnumArray(Activation, []f32) {
    assert(rows >= 1);

    var scratch: std.EnumArray(Activation, []f32) = .initUndefined();
    var entries = scratch.iterator();
    while (entries.next()) |entry| {
        entry.value.* = try arena.alloc(f32, @as(usize, rows) * widths.get(entry.key));
    }

    return scratch;
}

// -- tests ------------------------------------------------------------------

const builtin = @import("builtin");
const cpu_profile = @import("cpu_profile.zig");
const kv_cost = @import("kv_cost.zig");
const test_fakes = @import("test_fakes.zig");
const test_fixtures = @import("../core/test_fixtures.zig");
const llama_layer = @import("llama_layer.zig");
const qwen_layer = @import("qwen_layer.zig");

/// The families of the tests below.
const test_families = .{ llama_layer, qwen_layer };

/// The smallest model of a family that the gates run.
fn testFixture(comptime Family: type) []const u8 {
    return switch (Family.arch) {
        .llama => "SmolLM2-135M-Instruct-Q8_0",
        .qwen35 => "Qwen3.5-2B-Q8_0",
    };
}

test {
    // The shared timer module has no other test reference in a CPU build.
    _ = cpu_profile;
}

test "cpu profile: the ops of each family partition one decode step" {
    // The instrument exists only in a `-Dtrace` build. The GPU per-op backend touches a device.
    if (comptime !core_log.hot or build_options.use_vulkan_ops) {
        return error.SkipZigTest;
    } else {
        inline for (test_families) |Family| {
            try expectProfilePartition(Family);
        }
    }
}

fn expectProfilePartition(comptime Family: type) !void {
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    // Load the fixture and build a one-thread runner.
    const arena = arena_state.allocator();
    const fixture = try test_fixtures.path(arena, testFixture(Family));

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    const max_seq = @min(loaded.card.context_length, test_fakes.test_window);
    var pool: Pool = undefined;
    try pool.init(io, 1, 0);
    defer pool.deinit();

    const plan = kv_budget.pinned(1, max_seq);
    var r = try CpuRunner(Family).init(arena, &loaded, .{
        .plan = plan,
        .max_rows = 0,
        .pool = &pool,
        .policy = .{ .n_threads = 1 },
        .kv_dtype = .f32,
    });

    // One unprofiled step faults in the weight pages, so the measured step sees no first touch.
    _ = try r.step(1, 0);

    // Time one profiled step against the wall clock.
    const family = Family.profile_family;
    var prof: cpu_profile.CpuProfile = .{ .family = family };
    span.installSums(&prof.sums);
    const t0 = std.Io.Clock.Timestamp.now(io, .awake);
    _ = try r.step(2, 1);
    const t1 = std.Io.Clock.Timestamp.now(io, .awake);
    span.installSums(null);
    prof.steps = 1;
    const wall = time.elapsedNs(t0, t1);

    try std.testing.expectEqual(@as(u64, 1), prof.steps);
    // Cover: one decode step times every op of the family, and no other op key.
    for (std.enums.values(LayerOp)) |op| {
        if (cpu_profile.usedBy(op, family)) {
            try std.testing.expect(prof.nsOf(op) > 0);
        } else {
            try std.testing.expectEqual(@as(u64, 0), prof.nsOf(op));
        }
    }

    // Sum: the times of all keys add to the total.
    var attributed: u64 = 0;
    for (prof.sums.ns) |ns| {
        attributed += ns;
    }

    try std.testing.expectEqual(prof.total(), attributed);

    // Disjoint: two overlapping brackets attribute one nanosecond twice, past the wall time.
    try std.testing.expect(prof.total() <= wall);
    // Whole: the uncounted remainder is row validation and loop bookkeeping.
    try std.testing.expect(prof.total() * 2 >= wall);
    // The pool share is a subset of the whole, because the step also runs serial ops.
    try std.testing.expect(prof.nsParallel() > 0);
    try std.testing.expect(prof.nsParallel() < prof.total());
}

test "pool joins: a layer forks per weight, per row of its stateful op, and once for swiglu" {
    // A Debug build runs the forward too slowly for the unit run. The GPU per-op backend submits
    // the matmuls, the recurrent steps, and the attention rows without the pool.
    const skip = builtin.mode == .Debug or builtin.single_threaded or build_options.use_vulkan_ops;
    if (comptime skip) {
        return error.SkipZigTest;
    } else {
        inline for (test_families) |Family| {
            try expectPoolJoins(Family);
        }
    }
}

fn expectPoolJoins(comptime Family: type) !void {
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try test_fixtures.path(arena, testFixture(Family));

    var loaded = try loader.load(io, arena, fixture);
    defer loaded.close();

    // The zero floor forks every op of two units or more, and every op here holds two.
    const policy: parallel.Policy = .{ .n_threads = 4, .min_parallel_flops = 0 };
    var pool: Pool = undefined;
    try parallel.initPool(&pool, io, policy);
    defer pool.deinit();

    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const n_prompt: u32 = prompt.len;
    const plan = kv_budget.pinned(1, 16);
    var r = try CpuRunner(Family).init(arena, &loaded, .{
        .plan = plan,
        .max_rows = n_prompt,
        .pool = &pool,
        .policy = policy,
        .kv_dtype = .f32,
    });

    // Count the epochs of a 7-row prefill chunk, then of one decode step.
    const seq_prompt = pool.seq;
    _ = try r.prefillBatch(&prompt, 0);
    try std.testing.expectEqual(joinsOf(r.kv_index, n_prompt), pool.seq - seq_prompt);

    const seq_decode = pool.seq;
    _ = try r.step(8, n_prompt);
    try std.testing.expectEqual(joinsOf(r.kv_index, 1), pool.seq - seq_decode);
}

/// The fork-joins of one step of `n` rows with one logits row: one per weight of each layer, one
/// per row of the attention or the recurrence of each layer, one for the `swiglu` of each layer,
/// and one for the lm_head.
fn joinsOf(kv_index: []const ?u32, n: u32) u32 {
    var joins: u32 = 1;
    for (kv_index) |index| {
        // A KV layer of either family projects q, k, v, o, gate, up, and down. A GDN layer
        // projects qkv, z, alpha, beta, out, gate, up, and down.
        const weights: u32 = if (index != null) 7 else 8;
        joins += weights + n + 1;
    }

    return joins;
}

test "blob walk: each layer frames its regions at its cost entry, in model order" {
    inline for (test_families) |Family| {
        for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
            try expectBlobWalk(Family, kv_dtype);
        }
    }
}

/// Fill slot 1 of a whole-model runner and of two prefix runners with one byte value per region,
/// then check the blob of each against the cost vector and against the others.
fn expectBlobWalk(comptime Family: type, kv_dtype: dtype.KvDtype) !void {
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const fixture = try test_fixtures.path(arena, testFixture(Family));

    // The walk reads no weight, so the map skips the prefault.
    var loaded = try loader.loadWith(io, arena, fixture, .{ .prefault = false, .quiet = true });
    defer loaded.close();

    var pool: Pool = undefined;
    try pool.init(io, 1, 0);
    defer pool.deinit();

    // Slot 1 of two, so each region must come from the slot that the walk names.
    const plan = kv_budget.pinned(2, 16);
    const policy: parallel.Policy = .{ .n_threads = 1 };
    const n_rows: u32 = 5;
    const kv: kv_budget.KvStructure = .{
        .n_slots = 1,
        .window = n_rows,
        .costs = kv_cost.costVector(loaded.card, kv_dtype),
    };

    // The whole walk: each layer spans its cost entry, and each region holds its own value.
    var whole = try CpuRunner(Family).init(arena, &loaded, .{
        .plan = plan,
        .max_rows = 0,
        .pool = &pool,
        .policy = policy,
        .kv_dtype = kv_dtype,
    });
    const blob = try fillAndEvict(Family, arena, &whole, n_rows);
    try std.testing.expectEqual(kv_budget.slotSpanBytes(kv, 0, whole.n_layers), blob.len);
    var off: usize = 0;
    var layer: u32 = 0;
    while (layer < whole.n_layers) : (layer += 1) {
        const start = off;
        for (whole.slotRegions(1, layer, n_rows), 0..) |region, ri| {
            const value = regionValue(layer, ri);
            try std.testing.expect(std.mem.allEqual(u8, blob[off..][0..region.len], value));
            off += region.len;
        }

        try std.testing.expectEqual(kv_budget.slotSpanBytes(kv, layer, layer + 1), off - start);
    }

    // A restore into a zeroed slot gives the same blob back, and slot 0 stays zero.
    for (0..whole.n_layers) |li| {
        for (whole.slotRegions(1, @intCast(li), n_rows)) |region| {
            @memset(region, 0);
        }
    }

    try whole.restoreSlot(1, n_rows, blob);
    const again = try arena.alloc(u8, blob.len);
    try whole.evictSlot(1, n_rows, again);
    try std.testing.expectEqualSlices(u8, blob, again);
    try whole.evictSlot(0, n_rows, again);
    try std.testing.expect(std.mem.allEqual(u8, again, 0));

    // A prefix walk is the head of the whole walk, so the two halves of a hybrid blob frame it.
    for ([_]u32{ 1, whole.n_layers / 2 }) |k| {
        var prefix = try CpuRunner(Family).initPrefix(arena, &loaded, k, .{
            .plan = plan,
            .max_rows = 0,
            .pool = &pool,
            .policy = policy,
            .kv_dtype = kv_dtype,
        });
        const head_blob = try fillAndEvict(Family, arena, &prefix, n_rows);
        try std.testing.expectEqual(kv_budget.slotSpanBytes(kv, 0, k), head_blob.len);
        try std.testing.expectEqualSlices(u8, blob[0..head_blob.len], head_blob);
    }
}

/// The byte value of region `ri` of `layer` in the walk test. Each layer has two regions.
fn regionValue(layer: u32, ri: usize) u8 {
    return @truncate(2 * layer + ri + 1);
}

/// Fill each region of slot 1 with its value, and return the blob of that slot.
fn fillAndEvict(
    comptime Family: type,
    arena: Allocator,
    r: *CpuRunner(Family),
    n_rows: u32,
) ![]u8 {
    var layer: u32 = 0;
    while (layer < r.n_layers) : (layer += 1) {
        for (r.slotRegions(1, layer, n_rows), 0..) |region, ri| {
            @memset(region, regionValue(layer, ri));
        }
    }

    const blob = try arena.alloc(u8, @intCast(r.evictSlotBytesFor(n_rows)));
    try r.evictSlot(1, n_rows, blob);
    return blob;
}
