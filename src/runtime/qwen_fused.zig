//! `QwenFusedRunner` records the whole Qwen3.5 hybrid forward of one token into one Vulkan command
//! buffer and submits it once. Weights stage to VRAM once at
//! init, and activations stay in VRAM end to end. The kernel set and the tensor layout come from
//! the CPU `QwenRunner`. The GPU kernels GdnDecode, Attention@256, QwenQkNorm, and SigmoidGate are
//! tolerance-checked (see `ops/gdn.zig`).
//!
//! Comptime. `core_log.hot` removes the trace code from builds without hot tracing.
//!
//! Entry shapes. `step` advances one token on slot 0, the CLI decode. `prefillBatch` runs one
//! sequence chunk on slot 0 through the chunk-parallel GDN kernels. `stepBatch` advances a bounded
//! batch of (slot, token, pos) rows over per-slot KV and GDN-state regions. Each shape has a hidden
//! form that reads rows the caller computed. `qwen_hybrid.zig` feeds a layer suffix the host
//! residual stream. Both forms share one staging buffer and one record body.
//!
//! Layer range. `isFullAttention` keeps the absolute layer index, because the mixer of a layer is a
//! property of the model. Every tensor is found by that index. `kv_slot` compacts inside the range:
//! it counts the full-attention layers below its layer inside `[first..n_layers)`. A model-global
//! index against a range-sized KV store reads another layer and fails no Vulkan bound. Only an
//! index past `n_full` trips `regionIndex`. So every KV site checks its index against `n_full` and
//! its row span against the window. Init checks `kvSlotInRange` against the walking counter.
//!
//! KV dtype. The dtype moves the per-token KV price, the scratch reservation, and the two default
//! windows. The GDN recurrent state is not KV and stays f32 under any dtype. The planner stays
//! monotone in cost only while that fixed state is equal across dtypes.
//!
//! KV commit. `KvCommitF16` narrows `buf_k` and `buf_v` into `buf_k_kv` and `buf_v_kv` under the
//! rule of "KV commit" in `llama_fused.zig`. The barrier before `recordKvCommit` covers the compute
//! stage only.
//!
//! Recurrent state. The GDN state and conv rings live in persistent device buffers. They are a
//! running accumulator with no position axis. So a reused slot inherits the previous conversation
//! unless it is scrubbed. The KV needs no scrub, because attention reads only rewritten positions.
//!
//! Scratch. Every activation buffer holds `prefill_batch_max` rows, and decode is row 0. The same
//! sets serve decode and prefill, because the batch kernels read [b, dim] through a `b` push.
//! `buf_logits` alone holds `logit_rows_max` rows: the decode width plus the final prefill row of a
//! composed tick. The split record box ignores the layer range, so a suffix at any split allocates
//! what the whole model prices.
//!
//! Matmul kernels. All three Q8_0 matmul kernels share the 4-storage-buffer descriptor layout, so
//! the pre-bound sets work with any of them. The row count `b` alone picks the kernel:
//!
//!     b == 1    Reduce mat-vec, the single-stream decode kernel. It reads each weight row once and
//!               runs memory-bound (docs/gpu-rdna2.md section 4). The 16-row tile of LinearQ80Batch
//!               computes ~16x the work for one row, and makes that row about 8x slower.
//!     b < 64    Single-wave batch. It streams the weight once and computes b columns, for the
//!               multi-row decode and continuous prefill.
//!     b >= 64   Tiled GEMM. It wins at the b=128 standalone prefill (9B: 564 ms down to 379 ms).
//!
//! Prefill. `prefillBatch` turns per-token mat-vec into mat-mat, which amortizes weight reads
//! across the batch. QK-norm stays per-token because it reads row offsets. The GDN chain runs conv
//! and L2-norm per key head, then the matrix-form delta-rule scan per block of state columns. The
//! gated norm runs per value head and token. `gdn_scan` books the scan and the norm. `gpu-selftest
//! gdn` checks the chain against the CPU step at a tolerance, because the scan reorders the sums.
//!
//! Batch step. `stepBatch` follows the row rules of `QwenRunner.stepBatch`. Projections run
//! batched. The GDN recurrence runs one dispatch per same-slot run, so batch order is time order
//! and runs touch disjoint state. Rope and attention dispatch per row. Every K and V row lands
//! before any row attends. Row i reads only 0..=pos_i of its slot, so the layout gives causality. A
//! wide prefill chunk is one slot and many rows, so the batch width can exceed `n_slots`.
//! `logit_rows_max` bounds the lm_head suffix.
//!
//! Compact logits. The lm_head runs for the `want_logits` suffix only. So the prompt rows of a
//! prefill chunk skip the costliest matmul of the model. The suffix compacts to `buf_x_norm[0..]`
//! for the offset-0 lm_head set. The copy bounces through `buf_x`, which is free after the norm,
//! because `cmdCopyBuffer` forbids in-place overlap. Logits land at `buf_logits[0..]` and
//! `logits[0..]` in suffix order, the layout of `Sessions.advance`. `n_logits` can be 0: an opening
//! system-prompt chunk, or a mid-prompt chunk with no decode row.
//!
//! Snapshots and eviction. A blob is the whole forward state of one slot after `n` tokens. It holds
//! the KV rows [0..n] and the recurrent state and conv ring of every GDN layer. The GDN part is
//! valid only at that n. Layers come in ascending order, with K rows before V rows. The
//! prompt-cache snapshot takes slot 0, and residency evicts any slot at its committed row count.
//! The blob holds storage bytes. So a restore equals a cold prefill and a warm continuation matches
//! the cold token hash.
//!
//! Stages. Every save reads back through `kv_readback_stage`, and every restore uploads out of
//! `snap_stage`. A readback resource serves as a copy
//! destination only on the target driver. So the stage choice decides correctness.
//!
//! Tests. Every test here is device-free arithmetic. The hardware gates are the equivalence tests
//! in `tests/`.

const std = @import("std");
const vk = @import("vulkan");
const Tensor = @import("../core/tensor.zig").Tensor;
const loader = @import("../io/gguf/loader.zig");
const vulkan_ctx = @import("vulkan_ctx.zig");
const kernels = @import("../ops_vulkan/kernels.zig");
const naive = @import("../ops/naive.zig");
const limits = @import("../core/limits.zig");
const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_cost = @import("kv_cost.zig");
const kv_index = @import("../core/kv_index.zig");
const attn_split = @import("../core/attn_split.zig");
const attn_record = @import("attn_record.zig");
const time = @import("../core/time.zig");
const step_row = @import("step_row.zig");
const linear_dispatch = @import("linear_dispatch.zig");
const qwen_gpu_guard = @import("qwen_gpu_guard.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const gpu_profile = @import("gpu_profile.zig");
const core_log = @import("../core/log.zig");
const span = @import("../core/span.zig");
const step_stats = @import("../core/step_stats.zig");
const decode_ring = @import("decode_ring.zig");
const Allocator = std.mem.Allocator;
const StepRow = step_row.StepRow;
// `findTensor` fails when the named tensor is missing, which every site needs.
const findTensor = tensor_lookup.findRequired;
const findByFmt = tensor_lookup.findByFmt;

const log = std.log.scoped(.runner);
const log_dispatch = std.log.scoped(.dispatch);

// -- constants --------------------------------------------------------------

/// Pool capacity for all pre-bound sets, generous on purpose.
const descriptor_sets_max = 1024;
const storage_descriptors_max = 8192;
/// Fits every `blk.{d}.*` tensor name of the family.
const tensor_name_bytes_max = 80;

// -- types ------------------------------------------------------------------

const GdnWeights = struct {
    qkv: kernels.Q80Weight,
    zgate: kernels.Q80Weight,
    alpha: kernels.Q80Weight,
    beta: kernels.Q80Weight,
    ssm_out: kernels.Q80Weight,
    conv1d: kernels.HostBuffer, // f32
    ssm_a: kernels.HostBuffer,
    dt_bias: kernels.HostBuffer,
    ssm_norm: kernels.HostBuffer,
    state: kernels.DeviceBuffer, // persistent recurrent state
    conv_state: kernels.DeviceBuffer,
    // The descriptor sets below bind once at init and never change.
    set_qkv: vk.DescriptorSet,
    set_zgate: vk.DescriptorSet,
    set_alpha: vk.DescriptorSet,
    set_beta: vk.DescriptorSet,
    set_gdn: vk.DescriptorSet, // decode (sequential, 1 token)
    set_gdn_conv: vk.DescriptorSet, // prefill chunked: conv front
    set_gdn_scan: vk.DescriptorSet, // prefill chunked: scan
    set_ssm_out: vk.DescriptorSet,
};

const FullWeights = struct {
    q: kernels.Q80Weight,
    k: kernels.Q80Weight,
    v: kernels.Q80Weight,
    o: kernels.Q80Weight,
    q_norm: kernels.HostBuffer, // f32
    k_norm: kernels.HostBuffer,
    kv_slot: u32,
    set_q: vk.DescriptorSet,
    set_k: vk.DescriptorSet,
    set_v: vk.DescriptorSet,
    set_qknorm_q: vk.DescriptorSet,
    set_qknorm_k: vk.DescriptorSet,
    set_rope_q: vk.DescriptorSet,
    set_rope_k: vk.DescriptorSet,
    set_attention: vk.DescriptorSet,
    set_attention_multi: vk.DescriptorSet, // per-row (slot, pos) dispatches in stepBatch
    /// A row past one split: q, the whole K and V regions, and the records.
    set_attention_split: vk.DescriptorSet,
    set_sigmoid: vk.DescriptorSet,
    set_o: vk.DescriptorSet,
};

const Layer = struct {
    attn_norm: kernels.HostBuffer, // f32
    post_norm: kernels.HostBuffer,
    ffn_gate: kernels.Q80Weight,
    ffn_up: kernels.Q80Weight,
    ffn_down: kernels.Q80Weight,
    set_attn_norm: vk.DescriptorSet,
    set_post_norm: vk.DescriptorSet,
    set_residual_attn: vk.DescriptorSet,
    set_gate: vk.DescriptorSet,
    set_up: vk.DescriptorSet,
    set_swiglu: vk.DescriptorSet,
    set_down: vk.DescriptorSet,
    set_residual_ffn: vk.DescriptorSet,
    kind: union(enum) { gdn: GdnWeights, full: FullWeights },
};

pub const QwenFusedRunner = struct {
    gpa: Allocator,
    ctx: *const vulkan_ctx.Context,

    /// Layers this runner holds, `layers.len`. The model layer count only at `first_layer == 0`.
    n_layers: u32,
    /// The absolute model index of `layers[0]`. Zero on a whole-model runner.
    first_layer: u32,
    /// Full-attention layers in the range, the layer count of `kv.layout`. The per-layer record
    /// loop reads it once per layer per token, so it is a field and not a rescan.
    n_full: u32,
    hidden: u32,
    ffn: u32,
    vocab: u32,
    max_seq: u32,
    /// Resident conversations: per-slot rows in every KV region and per-slot GDN state regions.
    /// One slot is the CLI shape. A KV budget buys breadth past `limits.max_batch`.
    n_slots: u32,
    /// Decode width: the most rows one tick advances. At most `n_slots` and `limits.max_batch`.
    n_streams_max: u32,
    /// The most logits rows of one call. Sizes `buf_logits`, `stage_out`, and `logits`.
    logit_rows_max: u32,
    rms_eps: f32,
    // GDN dims (decoupled: n_v_heads >= n_k_heads, and 2B is n_v==n_k)
    n_k_heads: u32, // ssm.group_count (key heads)
    n_v_heads: u32, // ssm.time_step_rank (value heads)
    gdn_head_dim: u32, // ssm.state_size
    key_dim: u32, // n_k_heads * head_dim (q,k width)
    value_dim: u32, // n_v_heads * head_dim (v,z,mix_o,ssm_out width = inner_size)
    conv_kernel: u32,
    conv_dim: u32, // 2*key_dim + value_dim
    // full-attn dims
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    rope_base: f32,
    per_step: u32, // n_kv_heads * head_dim
    /// The element of every full-attention KV row.
    kv_dtype: dtype.KvDtype,

    embed: *const Tensor, // CPU embed lookup (tied lm_head also staged below)

    // kernels (decode = single-row, batch = B>1 prefill, sharing descriptor sets)
    k_linear: kernels.LinearQ80Reduce, // 1-row mat-vec (decode)
    k_rmsnorm: kernels.Rmsnorm,
    k_gdn: kernels.GdnDecode, // decode (1 token) path
    k_gdn_chunk: kernels.GdnChunkParallel, // chunked-parallel prefill path
    k_attn: kernels.Attention,
    k_rope: kernels.RopeMulti,
    k_swiglu: kernels.Swiglu,
    k_residual: kernels.ResidualAdd,
    k_gate: kernels.SigmoidGate,
    k_qknorm: kernels.QwenQkNorm,
    k_linear_gemm: kernels.LinearQ80Gemm, // tiled GEMM: b=128 standalone prefill
    k_linear_batch: kernels.LinearQ80Batch, // single-wave: low-b decode / continuous-prefill ticks
    k_rmsnorm_batch: kernels.RmsnormBatch,
    k_rope_batch: kernels.RopeMultiBatch,
    k_attn_batch: kernels.AttentionBatch,
    k_attn_multi: kernels.AttentionMultiStream, // per-row (slot, pos) attention in stepBatch
    /// Split attention passes. A decode row past one split merges partial records in split order.
    k_attn_split: kernels.AttentionSplit,
    /// The record box of the split rows in flight, sized at the declared context.
    attn_scratch: attn_split.Scratch,
    /// The one merge set: the records and `buf_attn_out`. Both buffers never move.
    set_attention_merge: vk.DescriptorSet,
    /// The commit narrowing kernel, present only at a non-f32 `kv_dtype`.
    k_kv_commit: ?kernels.KvCommitF16,
    /// The one narrowing set: f32 scratch sources, narrowed destinations. The buffers never move.
    set_kv_commit: vk.DescriptorSet,
    prefill_batch_max: u32,

    // scratch (VRAM)
    buf_x: kernels.DeviceBuffer,
    buf_x_norm: kernels.DeviceBuffer,
    buf_mix_out: kernels.DeviceBuffer,
    buf_qkv: kernels.DeviceBuffer,
    buf_q_n: kernels.DeviceBuffer, // chunked GDN: conv+normed q (per key head)
    buf_k_n: kernels.DeviceBuffer, // chunked GDN: conv+normed k
    buf_v_conv: kernels.DeviceBuffer, // chunked GDN: the convolved v, per value head
    buf_z: kernels.DeviceBuffer,
    buf_alpha: kernels.DeviceBuffer,
    buf_beta: kernels.DeviceBuffer,
    buf_mix_o: kernels.DeviceBuffer,
    buf_q_gate: kernels.DeviceBuffer,
    buf_q: kernels.DeviceBuffer,
    /// RoPE per-pair freq divisors (rope_dim/2 f32), all ones: Qwen3.5 has no `rope_freqs.weight`.
    /// Binding 1 of the rope sets, to match the shared rope shaders.
    buf_rope_ff: kernels.HostBuffer,
    buf_gate: kernels.DeviceBuffer,
    buf_k: kernels.DeviceBuffer,
    buf_v: kernels.DeviceBuffer,
    /// The narrowed commit scratch.
    buf_k_kv: kernels.DeviceBuffer,
    buf_v_kv: kernels.DeviceBuffer,
    buf_attn_out: kernels.DeviceBuffer,
    buf_attn_split: kernels.DeviceBuffer,
    buf_ffn_gate: kernels.DeviceBuffer,
    buf_ffn_up: kernels.DeviceBuffer,
    buf_logits: kernels.DeviceBuffer,
    /// The KV cache: one region per layer and tensor, each `[slot][max_seq][per_step]` of
    /// `kv_dtype`.
    kv: kernels.KvStore,

    output_norm: kernels.HostBuffer, // f32
    lm_head: kernels.Q80Weight, // tied to embed
    set_output_norm: vk.DescriptorSet,
    set_lm_head: vk.DescriptorSet,

    layers: []Layer,

    desc_pool: vk.DescriptorPool,
    /// Every submit except a decode step: prefill, `stepBatch`, the profiler, the bench probes.
    cmd_buffer: vk.CommandBuffer,
    /// The two buffers of `step` and `stepHidden`. Every other submit invalidates the ring first.
    decode: decode_ring.DecodeRing,
    /// True after init binds the last set. A ring recording stays valid only while no set changes.
    sets_fixed: bool,
    stage_in: kernels.HostBuffer,
    stage_out: kernels.HostBuffer,
    snap_stage: kernels.HostBuffer, // host-visible upload stage (H2D restores)
    /// The stage of every device-to-host copy.
    kv_readback_stage: kernels.HostBuffer,
    /// The chunk one staged copy moves.
    swap_chunk_bytes: u64,
    /// Row i of a `stepBatch` call sits at `logits[i * vocab ..]`. `step` and `prefillBatch` return
    /// row 0.
    logits: []f32,

    /// The timestamps of `cmd_buffer`. A profile command raises the level for one submit.
    prof: gpu_profile.GpuProfile,
    /// The runner kind of the GPU records. A suffix runner records for the hybrid that owns it.
    gpu_runner: span.GpuRunner,

    /// Host time of the single-row decode host steps. `generate.decode` clears it and reads it.
    host: step_stats.DecodeHost,

    /// True while `cmd_buffer` holds a `bench-decode-host` recording that can run again.
    probe_recorded: bool,

    /// The session surface reads this marker and then needs both slot hooks from this runner.
    pub const has_recurrent_state = true;

    /// Full-attention layers inside the absolute layer range `[from..to)`.
    fn fullCountInRange(g: loader.GdnParams, from: u32, to: u32) u32 {
        std.debug.assert(from <= to);
        std.debug.assert(to <= limits.max_layers);

        var n_full: u32 = 0;
        var li: u32 = from;
        while (li < to) : (li += 1) {
            if (g.isFullAttention(li)) {
                n_full += 1;
            }
        }

        std.debug.assert(n_full <= to - from);
        return n_full;
    }

    /// The KV index of the full-attention layer at absolute index `li`, in a range that starts at
    /// `from`.
    fn kvSlotInRange(g: loader.GdnParams, from: u32, li: u32) u32 {
        std.debug.assert(from <= li);
        std.debug.assert(g.isFullAttention(li));

        return fullCountInRange(g, from, li);
    }

    /// What one KV slot costs at `kv_dtype`. The planner states the price, so the CLI reads the
    /// same figure before a runner exists.
    pub fn kvCostModel(card: loader.ModelCard, kv_dtype: dtype.KvDtype) kv_budget.KvCostModel {
        std.debug.assert(card.gdn != null);

        return kv_budget.costModel(card.kvShape(), kv_dtype);
    }

    /// What one slot holds over a layer range, in the parts the preflight logs. The preflight
    /// asserts that `total` equals the `kv_budget.slotSpanBytes` of the same range.
    pub const RangeSlotBytes = struct {
        kv: u64,
        state: u64,
        conv: u64,

        pub fn total(self: RangeSlotBytes) u64 {
            return self.kv + self.state + self.conv;
        }
    };

    /// The device bytes of one slot over the absolute layer range `[from..to)`. It is the runner
    /// side of the per-layer cost vector, so the preflight compares two derivations of one range.
    pub fn rangeSlotBytes(
        card: loader.ModelCard,
        kv_dtype: dtype.KvDtype,
        window: u32,
        from: u32,
        to: u32,
    ) RangeSlotBytes {
        std.debug.assert(from <= to);
        std.debug.assert(to <= card.n_layers);
        std.debug.assert(window >= 1);

        const g = card.gdn orelse return .{ .kv = 0, .state = 0, .conv = 0 };
        const n_full: u64 = fullCountInRange(g, from, to);
        const n_gdn: u64 = @as(u64, to - from) - n_full;
        const key_dim: u64 = @as(u64, g.group_count) * g.state_size;
        const value_dim: u64 = @as(u64, g.time_step_rank) * g.state_size;
        const state_elems: u64 = @as(u64, g.time_step_rank) * g.state_size * g.state_size;
        const conv_elems: u64 = @as(u64, g.conv_kernel - 1) * (2 * key_dim + value_dim);
        const kv_row: u64 = kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        return .{
            .kv = 2 * n_full * @as(u64, window) * kv_row,
            .state = n_gdn * state_elems * @sizeOf(f32),
            .conv = n_gdn * conv_elems * @sizeOf(f32),
        };
    }

    /// Rows one prefill dispatch holds, and the batch factor of the activation
    /// scratch.
    pub const prefill_rows_max: u32 = 128;

    /// The device bytes of the activation scratch. It must match the `initSuffix` allocation list.
    /// A card without GDN metadata never builds this runner, so it prices nothing.
    pub fn scratchVramBytes(card: loader.ModelCard, kv_dtype: dtype.KvDtype) u64 {
        const g = card.gdn orelse return 0;
        const key_dim: u64 = @as(u64, g.group_count) * g.state_size;
        const value_dim: u64 = @as(u64, g.time_step_rank) * g.state_size;
        const conv_dim: u64 = 2 * key_dim + value_dim;
        const q_dim: u64 = @as(u64, card.n_heads) * card.head_dim;
        const per_step: u64 = @as(u64, card.n_kv_heads) * card.head_dim;
        const per_row: u64 = 3 * @as(u64, card.embedding_length) + conv_dim +
            2 * key_dim + 3 * value_dim + 2 * @as(u64, g.time_step_rank) +
            5 * q_dim + 2 * per_step + 2 * @as(u64, card.feed_forward_length);
        const f32_bytes: u64 = @as(u64, prefill_rows_max) * per_row * @sizeOf(f32) +
            attnSplitScratch(card).totalBytes();
        if (kv_dtype == .f32) return f32_bytes;

        return f32_bytes + @as(u64, prefill_rows_max) * 2 * kv_dtype.rowBytes(@intCast(per_step));
    }

    /// The split attention records of this model. The pricing and the allocation read this one
    /// value.
    fn attnSplitScratch(card: loader.ModelCard) attn_split.Scratch {
        return attn_split.Scratch.init(card.n_heads, card.head_dim, card.context_length);
    }

    /// The device bytes the uploaded weights occupy: every tensor `init` stages
    /// to VRAM for the whole model.
    pub fn weightsVramBytes(loaded: *const loader.Loaded) u64 {
        return weightsVramBytesRange(loaded.tensors, loaded.card.n_layers, 0, loaded.card.n_layers);
    }

    /// `weightsVramBytes` over the absolute layer range `[from..to)`, plus the head. A suffix ends
    /// at the last layer and runs the head itself, so the head goes with any range. `token_embd`
    /// stays host side, so `headWeightBytes` charges it only as the tied lm_head.
    pub fn weightsVramBytesRange(
        refs: []const loader.TensorRef,
        n_layers: u32,
        from: u32,
        to: u32,
    ) u64 {
        std.debug.assert(from <= to);
        std.debug.assert(to <= n_layers);

        var table: [limits.max_layers]u64 = undefined;
        const per_layer = tensor_lookup.layerWeightBytes(refs, n_layers, &table);
        var total: u64 = tensor_lookup.headWeightBytes(refs);
        for (per_layer[from..to]) |bytes| {
            total += bytes;
        }

        return total;
    }

    /// The device bytes the per-slot logits rows occupy.
    pub fn logitsVramBytes(vocab_size: u32, n_slots: u32) u64 {
        return @as(u64, n_slots) * vocab_size * @sizeOf(f32);
    }

    /// The planner inputs of this model at `vram_free`, the measured free device-local bytes. Zero
    /// keeps the planner off the budget. `ceiling` comes from `limits.contextCeiling`. `logit_rows`
    /// must cover the `min(n_slots, n_streams + 1)` rows of `init`.
    pub fn kvCaps(
        loaded: *const loader.Loaded,
        ceiling: u32,
        vram_free: u64,
        logit_rows: u32,
        kv_dtype: dtype.KvDtype,
    ) kv_budget.Caps {
        std.debug.assert(ceiling >= 1);
        std.debug.assert(ceiling <= loaded.card.context_length);

        const windows = kv_budget.defaultWindows(kvCostModel(loaded.card, kv_dtype), ceiling);
        return .{
            .vram_free = vram_free,
            .weights = weightsVramBytes(loaded),
            .logits = logitsVramBytes(loaded.card.vocab_size, logit_rows),
            .scratch = scratchVramBytes(loaded.card, kv_dtype),
            .model_ctx = ceiling,
            .serve_default = windows.serve,
            .default_ceil = windows.run,
        };
    }

    // -- init ---------------------------------------------------------------

    /// The whole-model constructor. The caller states the KV dtype.
    pub fn init(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        kv_dtype: dtype.KvDtype,
    ) !*QwenFusedRunner {
        return initSuffix(gpa, ctx, loaded, plan, 0, kv_dtype);
    }

    /// Build a runner of layers `[first..n_layers)`, with weights, KV, GDN state, and sets sized to
    /// match. Every init path ends here, so one body sizes and frames every buffer.
    pub fn initSuffix(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        first: u32,
        kv_dtype: dtype.KvDtype,
    ) !*QwenFusedRunner {
        // Refuse a card of another family and a model with a missing weight.
        const c = loaded.card;
        if (c.arch != .qwen35) return error.NotQwen;

        const g = c.gdn orelse return error.NotQwen;
        const refs = loaded.tensors;
        try linear_dispatch.requireWeights(refs, c.n_layers);
        try qwen_gpu_guard.requireShape(c);

        // Slots bound breadth, streams bound width, and a row owns a slot.
        std.debug.assert(plan.n_slots >= 1);
        std.debug.assert(plan.n_slots <= limits.max_vram_slots);
        std.debug.assert(plan.n_streams >= 1);
        std.debug.assert(plan.n_streams <= limits.max_batch);
        std.debug.assert(plan.n_streams <= plan.n_slots);

        // The range ends at the last layer of the model, so the runner always owns the head. An
        // empty range owns a head with nothing to feed it.
        std.debug.assert(first < c.n_layers);
        const n_end: u32 = c.n_layers;

        const self = try gpa.create(QwenFusedRunner);
        errdefer gpa.destroy(self);

        // Copy the model geometry.
        self.gpa = gpa;
        self.ctx = ctx;
        self.first_layer = first;
        self.n_layers = n_end - first;
        self.hidden = c.embedding_length;
        self.ffn = c.feed_forward_length;
        self.vocab = c.vocab_size;

        // The caller plans the window and the slot count from the byte budget
        // (`core/kv_budget.zig`), because the same arithmetic serves the plan and the preflight.
        self.max_seq = plan.window;
        self.n_slots = plan.n_slots;
        self.n_streams_max = plan.n_streams;

        // Every decode row and the final prefill row of a composed tick want logits, and each such
        // row occupies its own slot.
        self.logit_rows_max = @min(plan.n_slots, plan.n_streams + 1);
        self.rms_eps = c.rms_norm_eps;

        self.n_k_heads = g.group_count;
        self.n_v_heads = g.time_step_rank;
        self.gdn_head_dim = g.state_size;
        self.key_dim = g.group_count * g.state_size;
        self.value_dim = g.time_step_rank * g.state_size; // = inner_size
        self.conv_kernel = g.conv_kernel;
        self.conv_dim = 2 * self.key_dim + self.value_dim;
        self.n_heads = c.n_heads;
        self.n_kv_heads = c.n_kv_heads;
        self.head_dim = c.head_dim;
        self.rope_dim = c.rope_dim_count;
        self.rope_base = c.rope_freq_base;
        self.per_step = c.n_kv_heads * c.head_dim;
        self.kv_dtype = kv_dtype;

        // The f16 batch shader unpacks word pairs (`KV_LOAD2`), so a row must hold an even element
        // count. Every registered qwen model does (head_dim 256).
        if (kv_dtype != .f32) std.debug.assert(self.per_step % 2 == 0);

        self.embed = try findTensor(refs, "token_embd.weight");

        // Count the full-attention layers inside the range, the only layers that hold a KV cache.
        // Every KV address rebases on this count, so it lands before the first buffer sizes itself.
        self.n_full = fullCountInRange(g, first, n_end);

        // The layer count is a multiple of the attention interval, so a range to the last layer
        // holds a full-attention layer. A KV layout of no layers holds no region.
        std.debug.assert(self.n_full >= 1);
        std.debug.assert(self.n_full <= self.n_layers);

        // Refuse a KV region past the device limits, before the first allocation.
        const kv_layout = kv_index.Layout.init(
            self.n_slots,
            self.max_seq,
            self.per_step,
            self.n_full,
            kv_dtype,
        );
        try ctx.guardKvLayout(kv_layout);

        // Guard the split attention records, which the plan prices at every split. The range holds
        // a full-attention layer, so a split set binds them.
        try kernels.AttentionSplit.guardHeadDim(c.head_dim);
        try kernels.GdnChunkParallel.guardHeadDim(g.state_size);
        const attn_scratch = attnSplitScratch(c);
        std.debug.assert(attn_split.splitCount(self.max_seq) <= attn_scratch.max_splits);
        try ctx.guardAttnSplitScratch(attn_scratch);
        self.attn_scratch = attn_scratch;

        // Refuse a plan over the live device budget. See "The GPU checks" in docs/architecture.md.
        // Every figure prices the range, so a suffix reserves what a suffix uploads.
        {
            const mib = 1024 * 1024;
            const per_slot = rangeSlotBytes(c, kv_dtype, self.max_seq, first, n_end);

            // The cost vector span over this range and the allocation sizes below are two
            // derivations of one number.
            const kv_struct: kv_budget.KvStructure = .{
                .n_slots = self.n_slots,
                .window = self.max_seq,
                .costs = kv_cost.qwenCostVector(c, kv_dtype),
            };
            std.debug.assert(per_slot.total() == kv_budget.slotSpanBytes(kv_struct, first, n_end));
            std.debug.assert(per_slot.kv == kv_layout.slotBytes());

            // The regions hold exactly the priced KV, and the plan adds the fixed recurrent state
            // of every slot on top.
            const kv_full: u64 = @as(u64, self.n_slots) * per_slot.kv;
            std.debug.assert(kv_layout.totalBytes() == kv_full);
            const gdn_state: u64 = @as(u64, self.n_slots) * per_slot.state;
            const gdn_conv: u64 = @as(u64, self.n_slots) * per_slot.conv;
            const logits_bytes: u64 = logitsVramBytes(self.vocab, self.logit_rows_max);
            const weights_bytes: u64 = weightsVramBytesRange(refs, c.n_layers, first, n_end);
            const scratch_bytes: u64 = scratchVramBytes(c, kv_dtype);

            // The logits figure of the plan can exceed the rows here: a budget caller reserves the
            // width before the planner resolves the slot count.
            std.debug.assert(plan.vram.kv == kv_full + gdn_state + gdn_conv);
            std.debug.assert(plan.vram.weights == weights_bytes);
            std.debug.assert(plan.vram.logits >= logits_bytes);
            std.debug.assert(plan.vram.scratch == scratch_bytes);

            // Compare the sum with the live budget.
            const planned: u64 = weights_bytes + kv_full + gdn_state + gdn_conv + logits_bytes +
                scratch_bytes;
            const available: u64 = ctx.deviceLocalAvailableBytes();
            log.debug(
                "vram preflight: planned {d} MiB (weights {d}, kv {d}, gdn {d}, scratch " ++
                    "{d}), available {d} MiB",
                .{
                    planned / mib,
                    weights_bytes / mib,
                    kv_full / mib,
                    (gdn_state + gdn_conv) / mib,
                    scratch_bytes / mib,
                    available / mib,
                },
            );
            if (!limits.vramPlanFits(planned, available)) {
                // Name the largest window that fits, from the same budget and reservations.
                const reserved: u64 = weights_bytes + logits_bytes + scratch_bytes;
                const cost: kv_budget.KvCostModel = .{
                    .kv_bytes_per_token = kv_layout.tokenBytes(),
                    .fixed_state_bytes_per_slot = per_slot.state + per_slot.conv,
                };
                std.debug.assert(cost.kv_bytes_per_token * self.max_seq == per_slot.kv);

                var refusal_buf: [kv_budget.vram_refusal_text_max]u8 = undefined;
                log.err("{s}", .{kv_budget.formatVramRefusal(&refusal_buf, .{
                    .planned = planned,
                    .available = available,
                    .weights = weights_bytes,
                    .kv = kv_full,
                    .recurrent_state = gdn_state + gdn_conv,
                    .n_slots = self.n_slots,
                    .window = self.max_seq,
                    .fit = kv_budget.largestFittingWindow(
                        available,
                        reserved,
                        cost,
                        self.n_slots,
                        self.max_seq,
                    ),
                })});
                return error.VramBudgetExceeded;
            }
        }

        // Build the single-row kernels. Each resource takes its own errdefer, because pipelines
        // and pool handles stay undefined until their line runs, so `deinit` cannot serve here.
        self.k_linear = try kernels.LinearQ80Reduce.init(ctx);
        errdefer self.k_linear.deinit();

        self.k_rmsnorm = try kernels.Rmsnorm.init(ctx);
        errdefer self.k_rmsnorm.deinit();

        self.k_gdn = try kernels.GdnDecode.init(ctx);
        errdefer self.k_gdn.deinit();

        self.k_gdn_chunk = try kernels.GdnChunkParallel.init(ctx);
        errdefer self.k_gdn_chunk.deinit();

        // The four attention kernels read the KV regions, so each takes the storage element.
        self.k_attn = try kernels.Attention.initKv(ctx, kv_dtype);
        errdefer self.k_attn.deinit();

        self.k_rope = try kernels.RopeMulti.init(ctx);
        errdefer self.k_rope.deinit();

        self.k_swiglu = try kernels.Swiglu.init(ctx);
        errdefer self.k_swiglu.deinit();

        self.k_residual = try kernels.ResidualAdd.init(ctx);
        errdefer self.k_residual.deinit();

        self.k_gate = try kernels.SigmoidGate.init(ctx);
        errdefer self.k_gate.deinit();

        self.k_qknorm = try kernels.QwenQkNorm.init(ctx);
        errdefer self.k_qknorm.deinit();

        // Build the batch kernels.
        self.k_linear_gemm = try kernels.LinearQ80Gemm.init(ctx);
        errdefer self.k_linear_gemm.deinit();

        self.k_linear_batch = try kernels.LinearQ80Batch.init(ctx);
        errdefer self.k_linear_batch.deinit();

        self.k_rmsnorm_batch = try kernels.RmsnormBatch.init(ctx);
        errdefer self.k_rmsnorm_batch.deinit();

        self.k_rope_batch = try kernels.RopeMultiBatch.init(ctx);
        errdefer self.k_rope_batch.deinit();

        self.k_attn_batch = try kernels.AttentionBatch.initKv(ctx, kv_dtype);
        errdefer self.k_attn_batch.deinit();

        self.k_attn_multi = try kernels.AttentionMultiStream.initKv(ctx, kv_dtype);
        errdefer self.k_attn_multi.deinit();

        self.k_attn_split = try kernels.AttentionSplit.initKv(ctx, kv_dtype);
        errdefer self.k_attn_split.deinit();

        // Build the commit narrowing kernel, which only a non-f32 dtype needs.
        self.k_kv_commit = null;
        self.set_kv_commit = .null_handle;
        if (kv_dtype != .f32) {
            self.k_kv_commit = try kernels.KvCommitF16.init(ctx);
        }

        errdefer if (self.k_kv_commit) |*k| k.deinit();

        self.prefill_batch_max = prefill_rows_max;

        // Allocate the activation scratch.
        const F = @sizeOf(f32);
        const pb: u64 = self.prefill_batch_max;
        self.buf_x = try devBuf(ctx, pb * self.hidden * F);
        errdefer self.buf_x.deinit();

        self.buf_x_norm = try devBuf(ctx, pb * self.hidden * F);
        errdefer self.buf_x_norm.deinit();

        self.buf_mix_out = try devBuf(ctx, pb * self.hidden * F);
        errdefer self.buf_mix_out.deinit();

        // Allocate the GDN scratch.
        self.buf_qkv = try devBuf(ctx, pb * self.conv_dim * F);
        errdefer self.buf_qkv.deinit();

        self.buf_q_n = try devBuf(ctx, pb * self.key_dim * F);
        errdefer self.buf_q_n.deinit();

        self.buf_k_n = try devBuf(ctx, pb * self.key_dim * F);
        errdefer self.buf_k_n.deinit();

        self.buf_v_conv = try devBuf(ctx, pb * self.value_dim * F);
        errdefer self.buf_v_conv.deinit();

        self.buf_z = try devBuf(ctx, pb * self.value_dim * F);
        errdefer self.buf_z.deinit();

        self.buf_alpha = try devBuf(ctx, pb * self.n_v_heads * F);
        errdefer self.buf_alpha.deinit();

        self.buf_beta = try devBuf(ctx, pb * self.n_v_heads * F);
        errdefer self.buf_beta.deinit();

        self.buf_mix_o = try devBuf(ctx, pb * self.value_dim * F);
        errdefer self.buf_mix_o.deinit();

        // Allocate the attention input scratch and the rope divisors.
        self.buf_q_gate = try devBuf(ctx, pb * self.n_heads * self.head_dim * 2 * F);
        errdefer self.buf_q_gate.deinit();

        self.buf_q = try devBuf(ctx, pb * self.n_heads * self.head_dim * F);
        errdefer self.buf_q.deinit();

        self.buf_rope_ff = .{ .ctx = ctx };
        try self.buf_rope_ff.ensure(@as(vk.DeviceSize, self.rope_dim / 2) * F);
        errdefer self.buf_rope_ff.deinit();

        @memset(self.buf_rope_ff.sliceTyped(f32, self.rope_dim / 2), 1.0);
        self.buf_gate = try devBuf(ctx, pb * self.n_heads * self.head_dim * F);
        errdefer self.buf_gate.deinit();

        self.buf_k = try devBuf(ctx, pb * self.per_step * F);
        errdefer self.buf_k.deinit();

        self.buf_v = try devBuf(ctx, pb * self.per_step * F);
        errdefer self.buf_v.deinit();

        // Allocate the narrowed commit scratch, only where a dispatch narrows into it.
        self.buf_k_kv = .{ .ctx = ctx };
        self.buf_v_kv = .{ .ctx = ctx };
        errdefer self.buf_k_kv.deinit();
        errdefer self.buf_v_kv.deinit();

        if (kv_dtype != .f32) {
            const kv_scratch: vk.DeviceSize = pb * self.kvRowBytes();
            try self.buf_k_kv.ensure(kv_scratch);
            try self.buf_v_kv.ensure(kv_scratch);
        }

        // Allocate the attention output, the split records, the FFN scratch, and the logits rows.
        self.buf_attn_out = try devBuf(ctx, pb * self.n_heads * self.head_dim * F);
        errdefer self.buf_attn_out.deinit();

        self.buf_attn_split = try devBuf(ctx, attn_scratch.totalBytes());
        errdefer self.buf_attn_split.deinit();

        self.buf_ffn_gate = try devBuf(ctx, pb * self.ffn * F);
        errdefer self.buf_ffn_gate.deinit();

        self.buf_ffn_up = try devBuf(ctx, pb * self.ffn * F);
        errdefer self.buf_ffn_up.deinit();

        self.buf_logits = try devBuf(ctx, @as(vk.DeviceSize, self.logit_rows_max) * self.vocab * F);
        errdefer self.buf_logits.deinit();

        // Allocate the KV cache. The commit copies move scratch rows of `kvRowBytes()` into layout
        // rows.
        std.debug.assert(kv_layout.rowBytes() == self.kvRowBytes());
        self.kv = try kernels.KvStore.init(ctx, kv_layout);
        errdefer self.kv.deinit();

        const kv_mib: f64 = @as(f64, @floatFromInt(self.kv.deviceBytes())) / (1024.0 * 1024.0);
        log.debug(
            "init qwen fused: layers [{d}..{d}) ({d} full, {d} gdn), {d} slots, max_seq " ++
                "{d}, kv {d:.0} MiB",
            .{
                self.first_layer,
                self.first_layer + self.n_layers,
                self.n_full,
                self.n_layers - self.n_full,
                self.n_slots,
                self.max_seq,
                kv_mib,
            },
        );

        // Allocate the host stages and the host logits.
        self.stage_in = .{ .ctx = ctx };
        try self.stage_in.ensure(pb * self.hidden * F); // up to prefill_batch_max embeddings
        errdefer self.stage_in.deinit();

        self.stage_out = .{ .ctx = ctx };
        try self.stage_out.ensure(@as(vk.DeviceSize, self.logit_rows_max) * self.vocab * F);
        errdefer self.stage_out.deinit();

        self.snap_stage = .{ .ctx = ctx }; // sized on the first restore
        self.kv_readback_stage = .{ .ctx = ctx }; // sized on the first save
        self.swap_chunk_bytes = limits.swap_stage_chunk_bytes;
        self.host = .{};
        self.probe_recorded = false;
        self.sets_fixed = false;
        self.logits = try gpa.alloc(f32, @as(usize, self.logit_rows_max) * self.vocab);
        errdefer gpa.free(self.logits);

        // Create the descriptor pool.
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .storage_buffer, .descriptor_count = storage_descriptors_max },
        };
        const pool_info: vk.DescriptorPoolCreateInfo = .{
            .max_sets = descriptor_sets_max,
            .pool_size_count = 1,
            .p_pool_sizes = &pool_sizes,
        };
        self.desc_pool = try ctx.vkd.createDescriptorPool(ctx.device, &pool_info, null);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

        // Upload the head: a separate `output.weight` (9B), or `token_embd` tied (2B).
        self.lm_head = try q80(ctx, findTensor(refs, "output.weight") catch self.embed);
        errdefer self.lm_head.deinit();

        self.output_norm = try f32buf(ctx, try findTensor(refs, "output_norm.weight"));
        errdefer self.output_norm.deinit();

        // Bind the head sets.
        self.set_output_norm = try self.mkSet(
            self.k_rmsnorm.pipeline.set_layout,
            &.{ bi(self.buf_x), bi(self.output_norm), bi(self.buf_x_norm) },
        );
        self.set_lm_head = try self.mkSet(
            self.k_linear.pipeline.set_layout,
            &.{
                bi(self.buf_x_norm),
                self.lm_head.scales,
                self.lm_head.quants,
                bi(self.buf_logits),
            },
        );

        // Bind the commit set. The sources are the f32 rope scratch, and the destinations hold the
        // same rows at the storage element.
        if (self.k_kv_commit) |*k| {
            self.set_kv_commit = try self.mkSet(
                k.pipeline.set_layout,
                &.{ bi(self.buf_k), bi(self.buf_v), bi(self.buf_k_kv), bi(self.buf_v_kv) },
            );
        }

        // Bind the split merge set. The guard above proved that the records fit one binding.
        std.debug.assert(self.buf_attn_split.capacity == attn_scratch.totalBytes());
        std.debug.assert(self.buf_attn_split.capacity <= ctx.buffer_limits.storage_range);
        self.set_attention_merge = try self.mkSet(
            self.k_attn_split.merge.set_layout,
            &.{ bi(self.buf_attn_split), bi(self.buf_attn_out) },
        );

        // Allocate the layers.
        self.layers = try gpa.alloc(Layer, self.n_layers);
        errdefer gpa.free(self.layers);

        // Free the complete layers on a failure. The per-field errdefers of the loop body cover
        // the interrupted layer and die with the iteration, so the two never overlap.
        var layers_built: usize = 0;
        errdefer {
            var j = layers_built;
            while (j > 0) {
                j -= 1;
                deinitLayer(&self.layers[j]);
            }
        }

        // Build each layer. `kv_next` counts inside the range, and the closed form checks it.
        var name_buf: [tensor_name_bytes_max]u8 = undefined;
        var kv_next: u32 = 0;
        for (self.layers, 0..) |*layer, i| {
            // The absolute model index. Every tensor name and every `isFullAttention` question
            // takes it, and only `kv_slot` rebases.
            const li: u32 = first + @as(u32, @intCast(i));

            // Upload the norms and the FFN weights.
            layer.attn_norm = try f32buf(
                ctx,
                try findByFmt(refs, &name_buf, "blk.{d}.attn_norm.weight", .{li}),
            );
            errdefer layer.attn_norm.deinit();

            layer.post_norm = try f32buf(
                ctx,
                try findByFmt(refs, &name_buf, "blk.{d}.post_attention_norm.weight", .{li}),
            );
            errdefer layer.post_norm.deinit();

            layer.ffn_gate = try q80(
                ctx,
                try findByFmt(refs, &name_buf, "blk.{d}.ffn_gate.weight", .{li}),
            );
            errdefer layer.ffn_gate.deinit();

            layer.ffn_up = try q80(
                ctx,
                try findByFmt(refs, &name_buf, "blk.{d}.ffn_up.weight", .{li}),
            );
            errdefer layer.ffn_up.deinit();

            layer.ffn_down = try q80(
                ctx,
                try findByFmt(refs, &name_buf, "blk.{d}.ffn_down.weight", .{li}),
            );
            errdefer layer.ffn_down.deinit();

            // Bind the norm and residual sets.
            layer.set_attn_norm = try self.mkSet(
                self.k_rmsnorm.pipeline.set_layout,
                &.{ bi(self.buf_x), bi(layer.attn_norm), bi(self.buf_x_norm) },
            );
            layer.set_post_norm = try self.mkSet(
                self.k_rmsnorm.pipeline.set_layout,
                &.{ bi(self.buf_x), bi(layer.post_norm), bi(self.buf_x_norm) },
            );
            layer.set_residual_attn = try self.mkSet(
                self.k_residual.pipeline.set_layout,
                &.{ bi(self.buf_mix_out), bi(self.buf_x) },
            );

            // Bind the FFN sets.
            layer.set_gate = try self.mkSet(
                self.k_linear.pipeline.set_layout,
                &.{
                    bi(self.buf_x_norm),
                    layer.ffn_gate.scales,
                    layer.ffn_gate.quants,
                    bi(self.buf_ffn_gate),
                },
            );
            layer.set_up = try self.mkSet(
                self.k_linear.pipeline.set_layout,
                &.{
                    bi(self.buf_x_norm),
                    layer.ffn_up.scales,
                    layer.ffn_up.quants,
                    bi(self.buf_ffn_up),
                },
            );
            layer.set_swiglu = try self.mkSet(
                self.k_swiglu.pipeline.set_layout,
                &.{ bi(self.buf_ffn_gate), bi(self.buf_ffn_up), bi(self.buf_ffn_gate) },
            );

            layer.set_down = try self.mkSet(
                self.k_linear.pipeline.set_layout,
                &.{
                    bi(self.buf_ffn_gate),
                    layer.ffn_down.scales,
                    layer.ffn_down.quants,
                    bi(self.buf_mix_out),
                },
            );
            layer.set_residual_ffn = try self.mkSet(
                self.k_residual.pipeline.set_layout,
                &.{ bi(self.buf_mix_out), bi(self.buf_x) },
            );

            if (g.isFullAttention(li)) {
                // Upload the attention weights.
                var w: FullWeights = undefined;
                w.q = try q80(ctx, try findByFmt(refs, &name_buf, "blk.{d}.attn_q.weight", .{li}));
                errdefer w.q.deinit();

                w.k = try q80(ctx, try findByFmt(refs, &name_buf, "blk.{d}.attn_k.weight", .{li}));
                errdefer w.k.deinit();

                w.v = try q80(ctx, try findByFmt(refs, &name_buf, "blk.{d}.attn_v.weight", .{li}));
                errdefer w.v.deinit();

                w.o = try q80(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.attn_output.weight", .{li}),
                );
                errdefer w.o.deinit();

                w.q_norm = try f32buf(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.attn_q_norm.weight", .{li}),
                );
                errdefer w.q_norm.deinit();

                w.k_norm = try f32buf(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.attn_k_norm.weight", .{li}),
                );
                errdefer w.k_norm.deinit();

                // Assign the KV index. The walking counter and the closed form must name one index,
                // inside the KV height of this runner.
                std.debug.assert(kv_next == kvSlotInRange(g, first, li));
                std.debug.assert(kv_next < self.n_full);
                w.kv_slot = kv_next;
                kv_next += 1;

                // A KV binding starts at a region base, and a descriptor offset must be a multiple
                // of the offset alignment. So a region is a whole number of alignment units.
                std.debug.assert(w.kv_slot < self.kv.layout.n_layers);
                std.debug.assert(self.kv.layout.regionBytes() % self.kv.lim.offset_alignment == 0);

                // Bind the projection sets.
                w.set_q = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{ bi(self.buf_x_norm), w.q.scales, w.q.quants, bi(self.buf_q_gate) },
                );
                w.set_k = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{ bi(self.buf_x_norm), w.k.scales, w.k.quants, bi(self.buf_k) },
                );
                w.set_v = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{ bi(self.buf_x_norm), w.v.scales, w.v.quants, bi(self.buf_v) },
                );

                // Bind the QK-norm and rope sets.
                w.set_qknorm_q = try self.mkSet(
                    self.k_qknorm.pipeline.set_layout,
                    &.{ bi(self.buf_q_gate), bi(w.q_norm), bi(self.buf_q), bi(self.buf_gate) },
                );
                w.set_qknorm_k = try self.mkSet(
                    self.k_qknorm.pipeline.set_layout,
                    &.{ bi(self.buf_k), bi(w.k_norm), bi(self.buf_k), bi(self.buf_gate) },
                );
                w.set_rope_q = try self.mkSet(
                    self.k_rope.pipeline.set_layout,
                    &.{ bi(self.buf_q), bi(self.buf_rope_ff) },
                );
                w.set_rope_k = try self.mkSet(
                    self.k_rope.pipeline.set_layout,
                    &.{ bi(self.buf_k), bi(self.buf_rope_ff) },
                );

                // Bind the attention sets.
                w.set_attention = try self.mkSet(
                    self.k_attn.pipeline.set_layout,
                    &.{
                        bi(self.buf_q),
                        self.kv.slotView(w.kv_slot, .k, 0),
                        self.kv.slotView(w.kv_slot, .v, 0),
                        bi(self.buf_attn_out),
                    },
                );
                // Bind the whole K and V regions of this layer, because the shader strides by slot.
                w.set_attention_multi = try self.mkSet(self.k_attn_multi.pipeline.set_layout, &.{
                    bi(self.buf_q),
                    self.kv.region(w.kv_slot, .k),
                    self.kv.region(w.kv_slot, .v),
                    bi(self.buf_attn_out),
                });
                kernels.AttentionSplit.assertKBinding(self.kv.region(w.kv_slot, .k));
                kernels.AttentionSplit.assertVBinding(self.kv.region(w.kv_slot, .v));
                w.set_attention_split = try self.mkSet(self.k_attn_split.split.set_layout, &.{
                    bi(self.buf_q),
                    self.kv.region(w.kv_slot, .k),
                    self.kv.region(w.kv_slot, .v),
                    bi(self.buf_attn_split),
                });

                // Bind the gate and the output projection.
                w.set_sigmoid = try self.mkSet(
                    self.k_gate.pipeline.set_layout,
                    &.{ bi(self.buf_attn_out), bi(self.buf_gate) },
                );
                w.set_o = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{ bi(self.buf_attn_out), w.o.scales, w.o.quants, bi(self.buf_mix_out) },
                );
                layer.kind = .{ .full = w };
            } else {
                var w: GdnWeights = undefined;
                // Upload the GDN weights.
                w.qkv = try q80(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.attn_qkv.weight", .{li}),
                );
                errdefer w.qkv.deinit();

                w.zgate = try q80(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.attn_gate.weight", .{li}),
                );
                errdefer w.zgate.deinit();

                w.alpha = try q80(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.ssm_alpha.weight", .{li}),
                );
                errdefer w.alpha.deinit();

                w.beta = try q80(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.ssm_beta.weight", .{li}),
                );
                errdefer w.beta.deinit();

                w.ssm_out = try q80(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.ssm_out.weight", .{li}),
                );
                errdefer w.ssm_out.deinit();

                w.conv1d = try f32buf(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.ssm_conv1d.weight", .{li}),
                );
                errdefer w.conv1d.deinit();

                w.ssm_a = try f32buf(ctx, try findByFmt(refs, &name_buf, "blk.{d}.ssm_a", .{li}));
                errdefer w.ssm_a.deinit();

                w.dt_bias = try f32buf(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.ssm_dt.bias", .{li}),
                );
                errdefer w.dt_bias.deinit();

                w.ssm_norm = try f32buf(
                    ctx,
                    try findByFmt(refs, &name_buf, "blk.{d}.ssm_norm.weight", .{li}),
                );
                errdefer w.ssm_norm.deinit();

                // Allocate the slot-strided recurrent state. Slot s starts at `s *
                // stateElemsPerSlot()` and at `s * convElemsPerSlot()`.
                const slots: vk.DeviceSize = self.n_slots;
                w.state = try devBuf(
                    ctx,
                    slots * self.n_v_heads * self.gdn_head_dim * self.gdn_head_dim * F,
                );
                errdefer w.state.deinit();

                w.conv_state = try devBuf(ctx, slots * (self.conv_kernel - 1) * self.conv_dim * F);
                errdefer w.conv_state.deinit();

                try zeroDevBuf(ctx, &w.state);
                try zeroDevBuf(ctx, &w.conv_state);

                // Bind the GDN projection sets.
                w.set_qkv = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{ bi(self.buf_x_norm), w.qkv.scales, w.qkv.quants, bi(self.buf_qkv) },
                );
                w.set_zgate = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{ bi(self.buf_x_norm), w.zgate.scales, w.zgate.quants, bi(self.buf_z) },
                );
                w.set_alpha = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{ bi(self.buf_x_norm), w.alpha.scales, w.alpha.quants, bi(self.buf_alpha) },
                );
                w.set_beta = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{ bi(self.buf_x_norm), w.beta.scales, w.beta.quants, bi(self.buf_beta) },
                );

                // Bind the decode GDN set.
                w.set_gdn = try self.mkSet(
                    self.k_gdn.pipeline.set_layout,
                    &.{
                        bi(self.buf_qkv),
                        bi(self.buf_z),
                        bi(self.buf_alpha),
                        bi(self.buf_beta),
                        bi(w.conv1d),
                        bi(w.ssm_a),
                        bi(w.dt_bias),
                        bi(w.ssm_norm),
                        bi(w.state),
                        bi(w.conv_state),
                        bi(self.buf_mix_o),
                    },
                );

                // Chunked prefill: conv front (6) and scan (11).
                w.set_gdn_conv = try self.mkSet(
                    self.k_gdn_chunk.pipeline_conv.set_layout,
                    &.{
                        bi(self.buf_qkv),
                        bi(w.conv1d),
                        bi(w.conv_state),
                        bi(self.buf_q_n),
                        bi(self.buf_k_n),
                        bi(self.buf_v_conv),
                    },
                );

                w.set_gdn_scan = try self.mkSet(
                    self.k_gdn_chunk.pipeline_scan.set_layout,
                    &.{
                        bi(self.buf_q_n),
                        bi(self.buf_k_n),
                        bi(self.buf_v_conv),
                        bi(self.buf_alpha),
                        bi(self.buf_beta),
                        bi(w.ssm_a),
                        bi(w.dt_bias),
                        bi(w.ssm_norm),
                        bi(self.buf_z),
                        bi(w.state),
                        bi(self.buf_mix_o),
                    },
                );

                // Bind the output projection.
                w.set_ssm_out = try self.mkSet(
                    self.k_linear.pipeline.set_layout,
                    &.{
                        bi(self.buf_mix_o),
                        w.ssm_out.scales,
                        w.ssm_out.quants,
                        bi(self.buf_mix_out),
                    },
                );
                layer.kind = .{ .gdn = w };
            }

            layers_built = i + 1;
        }

        // The walk filled the KV height of the store, so no layer reads the rows of another layer
        // and no region sits unused.
        std.debug.assert(kv_next == self.n_full);
        std.debug.assert(kv_next == self.kv.layout.n_layers);

        // Build the command buffer, the decode ring, and the profile.
        self.cmd_buffer = try kernels.allocateOnePrimaryCmdBuffer(ctx);
        errdefer ctx.vkd.freeCommandBuffers(
            ctx.device,
            ctx.command_pool,
            &[_]vk.CommandBuffer{self.cmd_buffer},
        );

        self.gpu_runner = if (first == 0) .qwen_fused else .qwen_hybrid;
        self.decode = try decode_ring.DecodeRing.init(ctx, self.gpu_runner);
        errdefer self.decode.deinit();

        self.sets_fixed = true;

        self.prof = try gpu_profile.GpuProfile.init(ctx, span.gpuLevel(), span.gpuLevel());
        return self;
    }

    fn mkSet(
        self: *QwenFusedRunner,
        layout: vk.DescriptorSetLayout,
        infos: []const vk.DescriptorBufferInfo,
    ) !vk.DescriptorSet {
        std.debug.assert(!self.sets_fixed);

        const layouts = [_]vk.DescriptorSetLayout{layout};
        const ai: vk.DescriptorSetAllocateInfo = .{
            .descriptor_pool = self.desc_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = &layouts,
        };
        var set: vk.DescriptorSet = .null_handle;
        try self.ctx.vkd.allocateDescriptorSets(self.ctx.device, &ai, @ptrCast(&set));
        kernels.updateStorageBindings(self.ctx, set, infos);
        return set;
    }

    /// Bytes one KV row occupies at `kv_dtype`. It sizes the commit scratch and
    /// every commit copy, and init asserts that it equals the row of
    /// `kv.layout`. The GDN state regions never route through it and stay f32.
    fn kvRowBytes(self: *const QwenFusedRunner) vk.DeviceSize {
        return self.kv_dtype.rowBytes(self.per_step);
    }

    /// The two buffers a KV commit copy reads.
    fn kvCommitSrc(self: *const QwenFusedRunner) struct { k: vk.Buffer, v: vk.Buffer } {
        if (self.kv_dtype == .f32) return .{ .k = self.buf_k.buffer, .v = self.buf_v.buffer };
        return .{ .k = self.buf_k_kv.buffer, .v = self.buf_v_kv.buffer };
    }

    /// Record the narrowing of the `rows` committed rows of one layer, plus the barrier to the copy
    /// that follows. At f32 no dispatch runs.
    fn recordKvCommit(self: *QwenFusedRunner, cb: vk.CommandBuffer, rows: u32) void {
        std.debug.assert(rows >= 1);
        std.debug.assert(rows <= self.prefill_batch_max);

        if (self.k_kv_commit) |*k| {
            if (comptime core_log.hot) {
                log_dispatch.debug("  kv commit ({t}): rows {d}", .{ self.kv_dtype, rows });
            }

            k.recordInto(cb, self.set_kv_commit, rows * self.per_step);
        }

        barrierShaderToTransfer(self.ctx.vkd, cb);
    }

    /// The attention kernel set over the attention pipelines of the runner.
    fn attnKernels(self: *QwenFusedRunner) kernels.AttnKernels {
        return .{
            .single = &self.k_attn,
            .multi = &self.k_attn_multi,
            .passes = &self.k_attn_split,
            .batch = &self.k_attn_batch,
        };
    }

    fn attnShape(self: *const QwenFusedRunner, stream_stride_elems: u32) kernels.AttnShape {
        return .{
            .n_heads = self.n_heads,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
            .stream_stride_elems = stream_stride_elems,
            .max_splits = self.attn_scratch.max_splits,
        };
    }

    // -- step ---------------------------------------------------------------
    // Each entry shape has a token form and a hidden form.

    /// One decode step of `token_id` on slot 0. Returns the logits row, valid until the next call.
    pub fn step(self: *QwenFusedRunner, token_id: u32, pos: u32) ![]f32 {
        // The callers guard this too, but a public entry point checks its own precondition.
        if (pos >= self.max_seq) return error.PositionOutOfRange;

        log.debug("step: token {d}, pos {d}", .{ token_id, pos });
        naive.embedLookup(
            self.embed.*,
            token_id,
            self.hidden,
            self.stage_in.sliceTyped(f32, self.hidden),
        );
        return self.stepStaged(pos);
    }

    /// `step` from one activation row instead of one token id.
    pub fn stepHidden(self: *QwenFusedRunner, x: []const f32, pos: u32) ![]f32 {
        std.debug.assert(x.len == self.hidden);

        if (pos >= self.max_seq) return error.PositionOutOfRange;

        log.debug("step hidden: pos {d}", .{pos});
        @memcpy(self.stage_in.sliceTyped(f32, self.hidden), x);
        return self.stepStaged(pos);
    }

    /// The decode body: the staging buffer already holds this step's row.
    fn stepStaged(self: *QwenFusedRunner, pos: u32) ![]f32 {
        return self.stepStagedKey(gpu_profile.decodeKey(pos, self.decode.level()));
    }

    /// One decode step through the record ring, then the logits copy. The ring
    /// records `key` on a miss and the next key while the GPU runs.
    fn stepStagedKey(self: *QwenFusedRunner, key: gpu_profile.DecodeKey) ![]f32 {
        std.debug.assert(self.sets_fixed);

        try self.decode.step(key, key.next(self.max_seq), DecodeRec{ .runner = self }, &self.host);
        const t0 = core_log.now();
        @memcpy(self.logits[0..self.vocab], self.stage_out.sliceTyped(f32, self.vocab));
        self.host.copy.add(step_stats.nsBetween(t0, core_log.now()));
        return self.logits[0..self.vocab];
    }

    /// The record callback of the decode ring.
    const DecodeRec = struct {
        runner: *QwenFusedRunner,

        pub fn recordDecode(
            self: DecodeRec,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            key: gpu_profile.DecodeKey,
        ) !void {
            return self.runner.recordOpts(cb, prof, key.pos, key.opts);
        }
    };

    /// Free the decode ring before a submit that is not a decode step.
    pub fn invalidateDecode(self: *QwenFusedRunner) void {
        self.decode.invalidate(&self.host);
    }

    /// Max rows one `stepBatch` accepts: `prefill_batch_max`, past `n_slots` and `logit_rows_max`.
    pub fn maxStepRows(self: *const QwenFusedRunner) u32 {
        return self.prefill_batch_max;
    }

    /// Rows of the compact logits buffers. `Sessions` derives the vocab stride
    /// from this, so it must state the allocated row count exactly.
    pub fn logitRowsMax(self: *const QwenFusedRunner) u32 {
        return self.logit_rows_max;
    }

    /// Advance a batch of rows, each a (slot, token, pos) over an independent KV and GDN-state
    /// slot. Logits land compactly at `logits[0..]`.
    pub fn stepBatch(self: *QwenFusedRunner, rows: []const StepRow) !void {
        // `n_logits` can be 0.
        const n_logits = try step_row.validateRows(
            rows,
            self.n_slots,
            self.prefill_batch_max,
            self.max_seq,
        );
        const n: u32 = @intCast(rows.len);
        log.debug(
            "step_batch: {d} rows, slot {d} pos {d}..{d}",
            .{ n, rows[0].slot, rows[0].pos, rows[n - 1].pos },
        );
        const x_host = self.stage_in.sliceTyped(f32, @as(usize, n) * self.hidden);
        for (rows, 0..) |r, i| {
            naive.embedLookup(
                self.embed.*,
                r.token,
                self.hidden,
                x_host[i * self.hidden ..][0..self.hidden],
            );
        }

        return self.stepBatchStaged(rows, n_logits);
    }

    /// `stepBatch` from one activation row per row instead of one token per row.
    pub fn stepBatchHidden(self: *QwenFusedRunner, x: []const f32, rows: []const StepRow) !void {
        const n_logits = try step_row.validateRows(
            rows,
            self.n_slots,
            self.prefill_batch_max,
            self.max_seq,
        );
        std.debug.assert(x.len == rows.len * @as(usize, self.hidden));
        log.debug(
            "step_batch hidden: {d} rows, slot {d} pos {d}..{d}",
            .{ rows.len, rows[0].slot, rows[0].pos, rows[rows.len - 1].pos },
        );
        @memcpy(self.stage_in.sliceTyped(f32, x.len), x);
        return self.stepBatchStaged(rows, n_logits);
    }

    /// The batch body: the staging buffer already holds one row per step row.
    fn stepBatchStaged(self: *QwenFusedRunner, rows: []const StepRow, n_logits: u32) !void {
        std.debug.assert(n_logits <= self.logit_rows_max);

        const n: u32 = @intCast(rows.len);
        const live = (comptime core_log.hot) and core_log.enabled(.dispatch, .debug);
        var t_record: std.Io.Timestamp = undefined;
        if (live) {
            t_record = core_log.now();
        }

        try self.recordStepBatch(rows);
        var record_ms: f64 = 0;
        var t_submit: std.Io.Timestamp = undefined;
        if (live) {
            record_ms = core_log.msSince(t_record);
            t_submit = core_log.now();
        }

        try self.submitWait(.decode_batch);
        if (live) {
            log_dispatch.debug(
                "step_batch {d} rows: record {d:.1} ms, submit+wait {d:.1} ms",
                .{ n, record_ms, core_log.msSince(t_submit) },
            );
        }

        // The GPU computed and read back only the want_logits suffix, compacted
        // to the front.
        const n_floats = @as(usize, n_logits) * self.vocab;
        @memcpy(self.logits[0..n_floats], self.stage_out.sliceTyped(f32, n_floats));
    }

    /// Batch size where the tiled GEMM beats the single-wave batch kernel. The GEMM computes a full
    /// 64-row tile, so it amortizes its weight reads only once `b` fills that height.
    const gemm_min_batch: u32 = 64;

    /// Record one matmul over `b` rows. `b` alone picks the kernel.
    fn recordBatchMatmul(
        self: *QwenFusedRunner,
        cb: vk.CommandBuffer,
        set: vk.DescriptorSet,
        b: u32,
        k: u32,
        n: u32,
        op: []const u8,
    ) void {
        if (comptime core_log.hot) {
            const kind: []const u8 = if (b == 1)
                "reduce"
            else if (b >= gemm_min_batch) "gemm" else "batch";

            log_dispatch.debug("  matmul {s} {s}: b {d} k {d} n {d}", .{ op, kind, b, k, n });
        }

        if (b == 1) {
            self.k_linear.recordInto(cb, set, k, n);
        } else if (b >= gemm_min_batch) {
            self.k_linear_gemm.recordInto(cb, set, b, k, n);
        } else {
            self.k_linear_batch.recordInto(cb, set, b, k, n);
        }
    }

    /// Run a batched prefill of `tokens` from `start_pos` in one submit. Returns the logits of the
    /// last token. `tokens.len` must be at most `prefill_batch_max`.
    pub fn prefillBatch(self: *QwenFusedRunner, tokens: []const u32, start_pos: u32) ![]f32 {
        const b: u32 = @intCast(tokens.len);
        std.debug.assert(b >= 1 and b <= self.prefill_batch_max);
        log.debug("prefill: {d} tokens, batch {d}, start_pos {d}", .{ tokens.len, b, start_pos });
        const x_host = self.stage_in.sliceTyped(f32, @as(usize, b) * self.hidden);
        for (tokens, 0..) |tid, i| {
            naive.embedLookup(
                self.embed.*,
                tid,
                self.hidden,
                x_host[i * self.hidden ..][0..self.hidden],
            );
        }

        return self.prefillStaged(b, start_pos);
    }

    /// `prefillBatch` from `b` activation rows instead of `b` token ids.
    pub fn prefillBatchHidden(
        self: *QwenFusedRunner,
        x: []const f32,
        b: u32,
        start_pos: u32,
    ) ![]f32 {
        std.debug.assert(b >= 1 and b <= self.prefill_batch_max);
        std.debug.assert(x.len == @as(usize, b) * self.hidden);

        log.debug("prefill hidden: batch {d}, start_pos {d}", .{ b, start_pos });
        @memcpy(self.stage_in.sliceTyped(f32, x.len), x);
        return self.prefillStaged(b, start_pos);
    }

    /// The prefill body: the staging buffer already holds the chunk's rows.
    fn prefillStaged(self: *QwenFusedRunner, b: u32, start_pos: u32) ![]f32 {
        const live = (comptime core_log.hot) and core_log.enabled(.dispatch, .debug);
        var t_record: std.Io.Timestamp = undefined;
        if (live) {
            t_record = core_log.now();
        }

        try self.recordPrefillStep(b, start_pos);
        var record_ms: f64 = 0;
        var t_submit: std.Io.Timestamp = undefined;
        if (live) {
            record_ms = core_log.msSince(t_record);
            t_submit = core_log.now();
        }

        try self.submitWait(.prefill_chunk);
        if (live) {
            log_dispatch.debug(
                "prefill b={d}: record {d:.1} ms, submit+wait {d:.1} ms",
                .{ b, record_ms, core_log.msSince(t_submit) },
            );
        }

        @memcpy(self.logits[0..self.vocab], self.stage_out.sliceTyped(f32, self.vocab));
        return self.logits[0..self.vocab];
    }

    /// Submit `cmd_buffer`, wait for the queue, and read its timestamps back.
    /// The host times exist only when the buffer wrote timestamps or the span
    /// recorder runs, so the off path reads no clock.
    fn submitWait(self: *QwenFusedRunner, label: span.GpuLabel) !void {
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        self.invalidateDecode();
        const timed = self.prof.level != .off or span.enabled();
        const t0: std.Io.Timestamp = if (timed) core_log.now() else undefined;
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        const t1: std.Io.Timestamp = if (timed) core_log.now() else undefined;
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
        if (!timed) return;

        const t2 = core_log.now();
        span.record(.gpu_submit, t0, t1);
        span.record(.gpu_wait, t1, t2);
        _ = try self.prof.collectInto(self.gpu_runner, label, t0, t2);
    }

    /// Record one prefill chunk of `b` rows from `start_pos` into `cmd_buffer`.
    fn recordPrefillStep(self: *QwenFusedRunner, b: u32, start_pos: u32) !void {
        // Open the command buffer.
        const vkd = self.ctx.vkd;
        const cb = self.cmd_buffer;
        try vkd.resetCommandBuffer(cb, .{});
        try vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = true } });
        const F: vk.DeviceSize = @sizeOf(f32);
        const kv_row: vk.DeviceSize = self.kvRowBytes();
        const n_full = self.n_full;

        // Plan the attention once: every full-attention layer records the same plan. The chunk rows
        // sit in slot 0.
        std.debug.assert(b <= attn_split.max_plan_rows);
        const prefill_plan = attn_split.planPrefill(0, start_pos, b, self.attn_scratch.max_splits);
        const prefill_shape = self.attnShape(self.kv.layout.streamStrideElems());

        // Move the input rows to the device.
        self.prof.begin(cb);
        copyHostToDevice(
            vkd,
            cb,
            self.stage_in.buffer,
            self.buf_x.buffer,
            @as(vk.DeviceSize, b) * self.hidden * F,
        );
        barrierTransferToShader(vkd, cb);
        self.prof.mark(cb, .copy);

        for (self.layers, 0..) |*layer, li| {
            if (comptime core_log.hot) {
                log_dispatch.debug("record layer {d}", .{li});
            }

            // Norm the layer input, then run the mixer of this layer.
            self.k_rmsnorm_batch.recordInto(cb, layer.set_attn_norm, b, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);
            switch (layer.kind) {
                .gdn => |*w| {
                    // Project the input into the GDN streams.
                    self.recordBatchMatmul(cb, w.set_qkv, b, self.hidden, self.conv_dim, "gdn_qkv");
                    self.recordBatchMatmul(
                        cb,
                        w.set_zgate,
                        b,
                        self.hidden,
                        self.value_dim,
                        "zgate",
                    );
                    self.recordBatchMatmul(
                        cb,
                        w.set_alpha,
                        b,
                        self.hidden,
                        self.n_v_heads,
                        "alpha",
                    );
                    self.recordBatchMatmul(cb, w.set_beta, b, self.hidden, self.n_v_heads, "beta");
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .matmul_gdn_in);

                    // Run the chunked GDN chain. It shares `w.state` and `w.conv_state` with the
                    // decode `k_gdn`, so it leaves both as decode expects.
                    self.k_gdn_chunk.recordConv(
                        cb,
                        w.set_gdn_conv,
                        self.n_k_heads,
                        self.n_v_heads,
                        self.gdn_head_dim,
                        self.conv_kernel,
                        b,
                        self.rms_eps,
                    );
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .gdn_conv);
                    self.k_gdn_chunk.recordScan(
                        cb,
                        w.set_gdn_scan,
                        self.n_k_heads,
                        self.n_v_heads,
                        self.gdn_head_dim,
                        b,
                        self.rms_eps,
                    );
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .gdn_scan);

                    // Project the mix to the hidden width.
                    self.recordBatchMatmul(
                        cb,
                        w.set_ssm_out,
                        b,
                        self.value_dim,
                        self.hidden,
                        "ssm_out",
                    );
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .matmul_out);
                },
                .full => |*w| {
                    // Project q, k, and v.
                    self.recordBatchMatmul(
                        cb,
                        w.set_q,
                        b,
                        self.hidden,
                        self.n_heads * self.head_dim * 2,
                        "q",
                    );
                    self.recordBatchMatmul(cb, w.set_k, b, self.hidden, self.per_step, "k");
                    self.recordBatchMatmul(cb, w.set_v, b, self.hidden, self.per_step, "v");
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .matmul_attn_in);

                    // Norm q and k row by row, because the kernel reads row offsets.
                    var t: u32 = 0;
                    while (t < b) : (t += 1) {
                        self.k_qknorm.recordRow(
                            cb,
                            w.set_qknorm_q,
                            self.n_heads,
                            self.head_dim,
                            self.head_dim * 2,
                            self.head_dim,
                            true,
                            self.rms_eps,
                            t * self.n_heads * self.head_dim * 2,
                            t * self.n_heads * self.head_dim,
                        );

                        self.k_qknorm.recordRow(
                            cb,
                            w.set_qknorm_k,
                            self.n_kv_heads,
                            self.head_dim,
                            self.head_dim,
                            0,
                            false,
                            self.rms_eps,
                            t * self.per_step,
                            t * self.per_step,
                        );
                    }

                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .qknorm);

                    // Rotate q and k.
                    self.k_rope_batch.recordInto(
                        cb,
                        w.set_rope_q,
                        start_pos,
                        self.n_heads,
                        self.head_dim,
                        self.rope_dim,
                        self.rope_base,
                        b,
                        1,
                        0,
                        .neox,
                    );

                    self.k_rope_batch.recordInto(
                        cb,
                        w.set_rope_k,
                        start_pos,
                        self.n_kv_heads,
                        self.head_dim,
                        self.rope_dim,
                        self.rope_base,
                        b,
                        1,
                        0,
                        .neox,
                    );
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .rope);

                    // Commit K and V to the rebased KV address: the index inside the range, and the
                    // B rows of slot 0.
                    std.debug.assert(w.kv_slot < n_full);
                    std.debug.assert(@as(u64, start_pos) + b <= self.kv.layout.window);
                    const k_row = self.kv.rowAt(w.kv_slot, .k, 0, start_pos);
                    const v_row = self.kv.rowAt(w.kv_slot, .v, 0, start_pos);
                    // Narrow the rows.
                    self.recordKvCommit(cb, b);
                    const src = self.kvCommitSrc();
                    copyDeviceToDevice(
                        vkd,
                        cb,
                        src.k,
                        0,
                        k_row.buffer,
                        k_row.offset,
                        @as(vk.DeviceSize, b) * kv_row,
                    );
                    copyDeviceToDevice(
                        vkd,
                        cb,
                        src.v,
                        0,
                        v_row.buffer,
                        v_row.offset,
                        @as(vk.DeviceSize, b) * kv_row,
                    );
                    barrierTransferToShader(vkd, cb);
                    self.prof.mark(cb, .copy);

                    // Attend. The resident prefix runs the batch kernel on slot 0. A row past one
                    // split reads the whole-region set, so its bytes match a decode row.
                    attn_record.recordPrefill(
                        kernels.AttnKernels,
                        self.attnKernels(),
                        cb,
                        &prefill_plan,
                        .{
                            .shape = prefill_shape,
                            .resident = .{ .batch = w.set_attention },
                            .split = w.set_attention_split,
                            .merge = self.set_attention_merge,
                        },
                        start_pos,
                    );
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .attn);

                    // Gate the attention output, and project it to the hidden width.
                    self.k_gate.recordInto(cb, w.set_sigmoid, b * self.n_heads * self.head_dim);
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .elementwise);
                    self.recordBatchMatmul(
                        cb,
                        w.set_o,
                        b,
                        self.n_heads * self.head_dim,
                        self.hidden,
                        "o",
                    );
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .matmul_out);
                },
            }

            // Add the mixer residual, run the FFN, and add its residual.
            self.k_residual.recordInto(cb, layer.set_residual_attn, b * self.hidden);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
            self.k_rmsnorm_batch.recordInto(cb, layer.set_post_norm, b, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);
            self.recordBatchMatmul(cb, layer.set_gate, b, self.hidden, self.ffn, "gate");
            self.recordBatchMatmul(cb, layer.set_up, b, self.hidden, self.ffn, "up");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_up);
            self.k_swiglu.recordInto(cb, layer.set_swiglu, b * self.ffn);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
            self.recordBatchMatmul(cb, layer.set_down, b, self.ffn, self.hidden, "down");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_down);
            self.k_residual.recordInto(cb, layer.set_residual_ffn, b * self.hidden);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
        }

        // Only the LAST token's logits are needed. Bring row b-1 to row 0, then
        // the decode-shaped final norm + lm_head (which read row 0).
        if (b > 1) {
            barrierShaderToTransfer(vkd, cb);
            copyDeviceToDevice(
                vkd,
                cb,
                self.buf_x.buffer,
                @as(vk.DeviceSize, b - 1) * self.hidden * F,
                self.buf_x.buffer,
                0,
                self.hidden * F,
            );
            barrierTransferToShader(vkd, cb);
        }

        // Run the final norm and the lm_head, and copy the logits to the host.
        self.k_rmsnorm.recordInto(cb, self.set_output_norm, self.hidden, self.rms_eps);
        barrierShader(vkd, cb);
        self.prof.mark(cb, .norm);
        self.k_linear.recordInto(cb, self.set_lm_head, self.hidden, self.vocab);
        barrierShader(vkd, cb);
        self.prof.mark(cb, .lmhead);
        barrierShaderToTransfer(vkd, cb);
        copyDeviceToHost(
            vkd,
            cb,
            self.buf_logits.buffer,
            self.stage_out.buffer,
            @as(vk.DeviceSize, self.vocab) * F,
        );
        self.prof.mark(cb, .copy);
        self.prof.end(cb);
        try vkd.endCommandBuffer(cb);
    }

    /// One decoder pass over a mixed batch: batched projections, per-run GDN
    /// recurrence, per-row rope and attention against each row's slot. It keeps
    /// the dispatch and barrier order of `recordPrefillStep`.
    fn recordStepBatch(self: *QwenFusedRunner, rows: []const StepRow) !void {
        // Open the command buffer.
        const vkd = self.ctx.vkd;
        const cb = self.cmd_buffer;
        try vkd.resetCommandBuffer(cb, .{});
        try vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = true } });
        self.prof.begin(cb);
        const F: vk.DeviceSize = @sizeOf(f32);
        const n: u32 = @intCast(rows.len);
        const kv_row: vk.DeviceSize = self.kvRowBytes();
        const n_full = self.n_full;

        // The stride is in elements: the shader indexes elements, and its f16 load macro owns the
        // word addressing. Each set binds one region, and the shader reads `slot * stride` into it.
        const stream_stride_elems: u32 = self.kv.layout.streamStrideElems();
        // The push constant and the copy offsets state one slot stride in two
        // units, so a layout that moved only one of them fails here.
        std.debug.assert(
            self.kv_dtype.rowBytes(stream_stride_elems) == self.kv.layout.slotRegionBytes(),
        );

        // Plan the attention once: every full-attention layer records the same plan.
        std.debug.assert(n <= self.prefill_batch_max);
        const attn_plan = attn_record.planStep(rows, self.attn_scratch.max_splits);
        const attn_shape = self.attnShape(stream_stride_elems);

        // Move the input rows to the device.
        copyHostToDevice(
            vkd,
            cb,
            self.stage_in.buffer,
            self.buf_x.buffer,
            @as(vk.DeviceSize, n) * self.hidden * F,
        );
        barrierTransferToShader(vkd, cb);
        self.prof.mark(cb, .copy);

        for (self.layers, 0..) |*layer, li| {
            if (comptime core_log.hot) {
                log_dispatch.debug("record layer {d}", .{li});
            }

            // Norm the layer input, then run the mixer of this layer.
            self.k_rmsnorm_batch.recordInto(cb, layer.set_attn_norm, n, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);
            switch (layer.kind) {
                .gdn => |*w| {
                    // Project the input into the GDN streams.
                    self.recordBatchMatmul(cb, w.set_qkv, n, self.hidden, self.conv_dim, "gdn_qkv");
                    self.recordBatchMatmul(
                        cb,
                        w.set_zgate,
                        n,
                        self.hidden,
                        self.value_dim,
                        "zgate",
                    );
                    self.recordBatchMatmul(
                        cb,
                        w.set_alpha,
                        n,
                        self.hidden,
                        self.n_v_heads,
                        "alpha",
                    );
                    self.recordBatchMatmul(cb, w.set_beta, n, self.hidden, self.n_v_heads, "beta");
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .matmul_gdn_in);

                    // Run one GDN dispatch per same-slot run. Runs touch disjoint state and rows,
                    // so no barrier sits between them.
                    var start: u32 = 0;
                    while (start < n) {
                        var end: u32 = start + 1;
                        while (end < n and rows[end].slot == rows[start].slot) {
                            end += 1;
                        }

                        const slot = rows[start].slot;
                        self.k_gdn.recordRun(
                            cb,
                            w.set_gdn,
                            self.n_k_heads,
                            self.n_v_heads,
                            self.gdn_head_dim,
                            self.conv_kernel,
                            self.rms_eps,
                            end - start,
                            start,
                            slot * self.stateElemsPerSlot(),
                            slot * self.convElemsPerSlot(),
                        );
                        start = end;
                    }

                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .gdn_scan);

                    // Project the mix to the hidden width.
                    self.recordBatchMatmul(
                        cb,
                        w.set_ssm_out,
                        n,
                        self.value_dim,
                        self.hidden,
                        "ssm_out",
                    );
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .matmul_out);
                },
                .full => |*w| {
                    // Project q, k, and v.
                    self.recordBatchMatmul(
                        cb,
                        w.set_q,
                        n,
                        self.hidden,
                        self.n_heads * self.head_dim * 2,
                        "q",
                    );
                    self.recordBatchMatmul(cb, w.set_k, n, self.hidden, self.per_step, "k");
                    self.recordBatchMatmul(cb, w.set_v, n, self.hidden, self.per_step, "v");
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .matmul_attn_in);

                    // Norm q and k row by row, because the kernel reads row offsets.
                    for (0..rows.len) |i| {
                        const t: u32 = @intCast(i);
                        self.k_qknorm.recordRow(
                            cb,
                            w.set_qknorm_q,
                            self.n_heads,
                            self.head_dim,
                            self.head_dim * 2,
                            self.head_dim,
                            true,
                            self.rms_eps,
                            t * self.n_heads * self.head_dim * 2,
                            t * self.n_heads * self.head_dim,
                        );

                        self.k_qknorm.recordRow(
                            cb,
                            w.set_qknorm_k,
                            self.n_kv_heads,
                            self.head_dim,
                            self.head_dim,
                            0,
                            false,
                            self.rms_eps,
                            t * self.per_step,
                            t * self.per_step,
                        );
                    }

                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .qknorm);

                    // Rotate q and k at the position of each row.
                    for (rows, 0..) |r, i| {
                        self.k_rope_batch.recordInto(
                            cb,
                            w.set_rope_q,
                            r.pos,
                            self.n_heads,
                            self.head_dim,
                            self.rope_dim,
                            self.rope_base,
                            1,
                            0,
                            @intCast(i),
                            .neox,
                        );

                        self.k_rope_batch.recordInto(
                            cb,
                            w.set_rope_k,
                            r.pos,
                            self.n_kv_heads,
                            self.head_dim,
                            self.rope_dim,
                            self.rope_base,
                            1,
                            0,
                            @intCast(i),
                            .neox,
                        );
                    }

                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .rope);

                    // Narrow the rows. Every row lands before any row attends.
                    self.recordKvCommit(cb, n);
                    const src = self.kvCommitSrc();

                    // Copy each row to the rebased KV address: the index inside the range, and the
                    // row of its slot.
                    std.debug.assert(w.kv_slot < n_full);
                    for (rows, 0..) |r, i| {
                        std.debug.assert(r.slot < self.kv.layout.n_slots);
                        std.debug.assert(r.pos < self.kv.layout.window);

                        const k_row = self.kv.rowAt(w.kv_slot, .k, r.slot, r.pos);
                        const v_row = self.kv.rowAt(w.kv_slot, .v, r.slot, r.pos);
                        copyDeviceToDevice(
                            vkd,
                            cb,
                            src.k,
                            @as(vk.DeviceSize, @intCast(i)) * kv_row,
                            k_row.buffer,
                            k_row.offset,
                            kv_row,
                        );
                        copyDeviceToDevice(
                            vkd,
                            cb,
                            src.v,
                            @as(vk.DeviceSize, @intCast(i)) * kv_row,
                            v_row.buffer,
                            v_row.offset,
                            kv_row,
                        );
                    }

                    barrierTransferToShader(vkd, cb);
                    self.prof.mark(cb, .copy);

                    // Attend at the (slot, pos) of each row.
                    attn_record.record(kernels.AttnKernels, self.attnKernels(), cb, &attn_plan, .{
                        .shape = attn_shape,
                        .resident = .{ .multi = w.set_attention_multi },
                        .split = w.set_attention_split,
                        .merge = self.set_attention_merge,
                    });
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .attn);

                    // Gate the attention output, and project it to the hidden width.
                    self.k_gate.recordInto(cb, w.set_sigmoid, n * self.n_heads * self.head_dim);
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .elementwise);
                    self.recordBatchMatmul(
                        cb,
                        w.set_o,
                        n,
                        self.n_heads * self.head_dim,
                        self.hidden,
                        "o",
                    );
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .matmul_out);
                },
            }

            // Add the mixer residual, run the FFN, and add its residual.
            self.k_residual.recordInto(cb, layer.set_residual_attn, n * self.hidden);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
            self.k_rmsnorm_batch.recordInto(cb, layer.set_post_norm, n, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);
            self.recordBatchMatmul(cb, layer.set_gate, n, self.hidden, self.ffn, "gate");
            self.recordBatchMatmul(cb, layer.set_up, n, self.hidden, self.ffn, "up");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_up);
            self.k_swiglu.recordInto(cb, layer.set_swiglu, n * self.ffn);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
            self.recordBatchMatmul(cb, layer.set_down, n, self.ffn, self.hidden, "down");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_down);
            self.k_residual.recordInto(cb, layer.set_residual_ffn, n * self.hidden);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
        }

        // Stop before the lm_head when no row wants logits. The KV and state writes are the whole
        // product, and a zero-size `cmdCopyBuffer` is not valid Vulkan.
        const first = step_row.wantLogitsFirst(rows);
        if (first == n) {
            self.prof.end(cb);
            try vkd.endCommandBuffer(cb);
            return;
        }

        // Run the final norm over all rows.
        self.k_rmsnorm_batch.recordInto(cb, self.set_output_norm, n, self.hidden, self.rms_eps);

        // Compact the `want_logits` suffix to the front, through `buf_x`.
        const n_logits: u32 = n - first;
        const hidden_row: vk.DeviceSize = @as(vk.DeviceSize, self.hidden) * F;

        if (first > 0) {
            barrierShaderToTransfer(vkd, cb);
            const suffix_bytes: vk.DeviceSize = @as(vk.DeviceSize, n_logits) * hidden_row;
            copyDeviceToDevice(
                vkd,
                cb,
                self.buf_x_norm.buffer,
                @as(vk.DeviceSize, first) * hidden_row,
                self.buf_x.buffer,
                0,
                suffix_bytes,
            );
            barrierTransferToTransfer(vkd, cb);
            copyDeviceToDevice(
                vkd,
                cb,
                self.buf_x.buffer,
                0,
                self.buf_x_norm.buffer,
                0,
                suffix_bytes,
            );
            barrierTransferToShader(vkd, cb);
        } else {
            barrierShader(vkd, cb);
        }

        // Run the lm_head over the suffix, and copy the logits to the host.
        self.recordBatchMatmul(cb, self.set_lm_head, n_logits, self.hidden, self.vocab, "lmhead");
        barrierShaderToTransfer(vkd, cb);
        self.prof.mark(cb, .lmhead);
        copyDeviceToHost(
            vkd,
            cb,
            self.buf_logits.buffer,
            self.stage_out.buffer,
            @as(vk.DeviceSize, n_logits) * self.vocab * F,
        );
        self.prof.mark(cb, .copy);
        self.prof.end(cb);
        try vkd.endCommandBuffer(cb);
    }

    /// Run one prefill chunk with GPU timestamps and print a breakdown by `gpu_profile.Phase`, the
    /// diagnostic behind `--profile-prefill`. It changes the GDN state and the KV as a real prefill
    /// does, so the runner is dirty afterwards.
    pub fn profilePrefill(
        self: *QwenFusedRunner,
        io: std.Io,
        tokens: []const u32,
        start_pos: u32,
        tag: []const u8,
    ) !void {
        const b: u32 = @intCast(tokens.len);
        std.debug.assert(b >= 1 and b <= self.prefill_batch_max);

        // Raise the profile for this submit. The pool grows on the first probe, so level off
        // creates none. No submit of this buffer is pending here.
        const level = self.prof.level;
        try self.prof.reserve(.phases);
        self.prof.setLevel(.phases);
        defer self.prof.setLevel(level);

        // Embed the chunk.
        const x_host = self.stage_in.sliceTyped(f32, @as(usize, b) * self.hidden);
        for (tokens, 0..) |tid, i| {
            naive.embedLookup(
                self.embed.*,
                tid,
                self.hidden,
                x_host[i * self.hidden ..][0..self.hidden],
            );
        }

        // Record, submit, and wait.
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        try self.recordPrefillStep(b, start_pos);
        const t_rec = std.Io.Clock.Timestamp.now(io, .awake);
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        self.invalidateDecode();
        try self.ctx.vkd.queueSubmit(
            self.ctx.queue,
            &[_]vk.SubmitInfo{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }},
            .null_handle,
        );
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
        const t_wait = std.Io.Clock.Timestamp.now(io, .awake);

        // Read the timestamps and print the breakdown.
        const r = (try self.prof.collectInto(
            self.gpu_runner,
            .prefill_chunk,
            t_rec.raw,
            t_wait.raw,
        )).?;
        try r.report(
            io,
            "[qwen prefill {s}] b={d}",
            .{ tag, b },
            time.msBetween(t0, t_rec),
            time.msBetween(t_rec, t_wait),
        );
    }

    /// Record one decode step at `pos` into `cb`.
    fn recordOpts(
        self: *QwenFusedRunner,
        cb: vk.CommandBuffer,
        prof: *gpu_profile.GpuProfile,
        pos: u32,
        opts: gpu_profile.RecordOpts,
    ) !void {
        // Open the command buffer.
        const vkd = self.ctx.vkd;
        try vkd.resetCommandBuffer(cb, .{});
        try vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = opts.one_time } });

        // Move the input row to the device.
        prof.begin(cb);
        const hidden_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.hidden) * @sizeOf(f32);
        const kv_row: vk.DeviceSize = self.kvRowBytes();
        copyHostToDevice(vkd, cb, self.stage_in.buffer, self.buf_x.buffer, hidden_bytes);
        barrierTransferToShader(vkd, cb);
        prof.mark(cb, .copy);

        // Plan the attention once: every full-attention layer records the same plan.
        const attn_plan = attn_split.planDecode(pos + 1);
        const attn_shape = self.attnShape(self.kv.layout.streamStrideElems());

        for (self.layers, 0..) |*layer, li| {
            if (comptime core_log.hot) {
                log_dispatch.debug("record layer {d}", .{li});
            }

            // Norm the layer input, then run the mixer of this layer.
            self.k_rmsnorm.recordInto(cb, layer.set_attn_norm, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            prof.mark(cb, .norm);
            switch (layer.kind) {
                .gdn => |*w| {
                    // Project the input into the GDN streams.
                    self.k_linear.recordInto(cb, w.set_qkv, self.hidden, self.conv_dim);
                    self.k_linear.recordInto(cb, w.set_zgate, self.hidden, self.value_dim);
                    self.k_linear.recordInto(cb, w.set_alpha, self.hidden, self.n_v_heads);
                    self.k_linear.recordInto(cb, w.set_beta, self.hidden, self.n_v_heads);
                    barrierShader(vkd, cb);
                    prof.mark(cb, .matmul_gdn_in);

                    // Run the sequential GDN over the one row.
                    self.k_gdn.recordInto(
                        cb,
                        w.set_gdn,
                        self.n_k_heads,
                        self.n_v_heads,
                        self.gdn_head_dim,
                        self.conv_kernel,
                        self.rms_eps,
                    );
                    barrierShader(vkd, cb);
                    prof.mark(cb, .gdn_scan);

                    // Project the mix to the hidden width.
                    self.k_linear.recordInto(cb, w.set_ssm_out, self.value_dim, self.hidden);
                    barrierShader(vkd, cb);
                    prof.mark(cb, .matmul_out);
                },
                .full => |*w| {
                    // Project q, k, and v.
                    self.k_linear.recordInto(
                        cb,
                        w.set_q,
                        self.hidden,
                        self.n_heads * self.head_dim * 2,
                    );
                    self.k_linear.recordInto(cb, w.set_k, self.hidden, self.per_step);
                    self.k_linear.recordInto(cb, w.set_v, self.hidden, self.per_step);
                    barrierShader(vkd, cb);
                    prof.mark(cb, .matmul_attn_in);

                    // Norm q and k. The q pass de-interleaves into the gate buffer, and the k pass
                    // norms in place.
                    self.k_qknorm.recordInto(
                        cb,
                        w.set_qknorm_q,
                        self.n_heads,
                        self.head_dim,
                        self.head_dim * 2,
                        self.head_dim,
                        true,
                        self.rms_eps,
                    );
                    self.k_qknorm.recordInto(
                        cb,
                        w.set_qknorm_k,
                        self.n_kv_heads,
                        self.head_dim,
                        self.head_dim,
                        0,
                        false,
                        self.rms_eps,
                    );
                    barrierShader(vkd, cb);
                    prof.mark(cb, .qknorm);

                    // Rotate q and k.
                    self.k_rope.recordInto(
                        cb,
                        w.set_rope_q,
                        pos,
                        self.n_heads,
                        self.head_dim,
                        self.rope_dim,
                        self.rope_base,
                        .neox,
                    );
                    self.k_rope.recordInto(
                        cb,
                        w.set_rope_k,
                        pos,
                        self.n_kv_heads,
                        self.head_dim,
                        self.rope_dim,
                        self.rope_base,
                        .neox,
                    );
                    barrierShader(vkd, cb);
                    prof.mark(cb, .rope);

                    // Commit K and V to the rebased KV address: the index inside the range, and the
                    // row of slot 0.
                    std.debug.assert(w.kv_slot < self.n_full);
                    std.debug.assert(pos < self.kv.layout.window);
                    const k_row = self.kv.rowAt(w.kv_slot, .k, 0, pos);
                    const v_row = self.kv.rowAt(w.kv_slot, .v, 0, pos);
                    // Narrow the row.
                    self.recordKvCommit(cb, 1);
                    const src = self.kvCommitSrc();
                    copyDeviceToDevice(vkd, cb, src.k, 0, k_row.buffer, k_row.offset, kv_row);
                    copyDeviceToDevice(vkd, cb, src.v, 0, v_row.buffer, v_row.offset, kv_row);
                    barrierTransferToShader(vkd, cb);
                    prof.mark(cb, .copy);

                    // Attend. A row past one split reads slot 0 through the whole-region set, so it
                    // takes the split shader of a batch row.
                    attn_record.record(kernels.AttnKernels, self.attnKernels(), cb, &attn_plan, .{
                        .shape = attn_shape,
                        .resident = .{ .single = w.set_attention },
                        .split = w.set_attention_split,
                        .merge = self.set_attention_merge,
                    });
                    barrierShader(vkd, cb);
                    prof.mark(cb, .attn);

                    // Gate the attention output, and project it to the hidden width.
                    self.k_gate.recordInto(cb, w.set_sigmoid, self.n_heads * self.head_dim);
                    barrierShader(vkd, cb);
                    prof.mark(cb, .elementwise);
                    self.k_linear.recordInto(
                        cb,
                        w.set_o,
                        self.n_heads * self.head_dim,
                        self.hidden,
                    );
                    barrierShader(vkd, cb);
                    prof.mark(cb, .matmul_out);
                },
            }

            // Add the mixer residual, run the FFN, and add its residual.
            self.k_residual.recordInto(cb, layer.set_residual_attn, self.hidden);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);
            self.k_rmsnorm.recordInto(cb, layer.set_post_norm, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            prof.mark(cb, .norm);
            self.k_linear.recordInto(cb, layer.set_gate, self.hidden, self.ffn);
            self.k_linear.recordInto(cb, layer.set_up, self.hidden, self.ffn);
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_ffn_up);
            self.k_swiglu.recordInto(cb, layer.set_swiglu, self.ffn);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);
            self.k_linear.recordInto(cb, layer.set_down, self.ffn, self.hidden);
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_ffn_down);
            self.k_residual.recordInto(cb, layer.set_residual_ffn, self.hidden);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);
        }

        // Run the final norm and the lm_head, and copy the logits to the host.
        self.k_rmsnorm.recordInto(cb, self.set_output_norm, self.hidden, self.rms_eps);
        barrierShader(vkd, cb);
        prof.mark(cb, .norm);
        self.k_linear.recordInto(cb, self.set_lm_head, self.hidden, self.vocab);
        barrierShader(vkd, cb);
        prof.mark(cb, .lmhead);
        barrierShaderToTransfer(vkd, cb);
        copyDeviceToHost(
            vkd,
            cb,
            self.buf_logits.buffer,
            self.stage_out.buffer,
            @as(vk.DeviceSize, self.vocab) * @sizeOf(f32),
        );
        prof.mark(cb, .copy);
        prof.end(cb);
        try vkd.endCommandBuffer(cb);
    }

    // -- bench hooks --------------------------------------------------------
    // The `bench-decode-host` command times the pieces of one decode step.
    // None of these hooks runs in a normal decode.

    /// Record the decode command buffer for `pos` and stop: no submit. The
    /// profile must stay at the process level. Then the timing is the record cost.
    pub fn probeRecord(self: *QwenFusedRunner, pos: u32) !void {
        std.debug.assert(self.prof.level == span.gpuLevel());
        std.debug.assert(pos < self.max_seq);

        try self.recordOpts(self.cmd_buffer, &self.prof, pos, .{});
        self.probe_recorded = false;
    }

    /// Record once without the one-time-submit flag, so `probeSubmitWait` can
    /// submit the same buffer again. Resubmission repeats the KV row write and
    /// the GDN state update, so the output after it has no meaning.
    pub fn probeRecordReusable(self: *QwenFusedRunner, pos: u32) !void {
        std.debug.assert(self.prof.level == span.gpuLevel());
        std.debug.assert(pos < self.max_seq);

        try self.recordOpts(self.cmd_buffer, &self.prof, pos, .{ .one_time = false });
        self.probe_recorded = true;
    }

    /// Submit the buffer that `probeRecordReusable` recorded, and wait for the queue.
    pub fn probeSubmitWait(self: *QwenFusedRunner) !void {
        std.debug.assert(self.probe_recorded);

        self.invalidateDecode();
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
    }

    /// Set the timestamp level of the decode ring. Only the gap mode of
    /// `bench-decode-host` calls it. It drops every recording first.
    pub fn setDecodeLevel(self: *QwenFusedRunner, level: gpu_profile.Level) !void {
        try self.decode.setLevel(level, &self.host);
    }

    /// The timestamps of the last decode step, or null at level `off`.
    pub fn lastDecodeReadout(self: *const QwenFusedRunner) ?gpu_profile.Readout {
        return self.decode.last;
    }

    /// Run one decode step with GPU timestamps and print a breakdown by `gpu_profile.Phase`. It
    /// changes the recurrent state and the KV as a real step does, so the runner is dirty
    /// afterwards.
    pub fn profileStep(
        self: *QwenFusedRunner,
        io: std.Io,
        token_id: u32,
        pos: u32,
        tag: []const u8,
    ) !void {
        self.invalidateDecode();
        const level = self.prof.level;
        // The pool grows on the first probe, so level off creates none. No
        // submit of this buffer is pending here.
        try self.prof.reserve(.phases);
        self.prof.setLevel(.phases);
        defer self.prof.setLevel(level);

        const x_host = self.stage_in.sliceTyped(f32, self.hidden);
        naive.embedLookup(self.embed.*, token_id, self.hidden, x_host);
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        try self.recordOpts(self.cmd_buffer, &self.prof, pos, .{});
        const t_rec = std.Io.Clock.Timestamp.now(io, .awake);
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        try self.ctx.vkd.queueSubmit(
            self.ctx.queue,
            &[_]vk.SubmitInfo{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }},
            .null_handle,
        );
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
        const t_wait = std.Io.Clock.Timestamp.now(io, .awake);

        const r = (try self.prof.collectInto(self.gpu_runner, .decode, t_rec.raw, t_wait.raw)).?;
        try r.report(
            io,
            "[qwen decode {s}] pos={d}",
            .{ tag, pos },
            time.msBetween(t0, t_rec),
            time.msBetween(t_rec, t_wait),
        );
    }

    /// One slot's GDN recurrent-state region, in f32 elements.
    fn stateElemsPerSlot(self: *const QwenFusedRunner) u32 {
        return self.n_v_heads * self.gdn_head_dim * self.gdn_head_dim;
    }

    /// One slot's conv-ring region, in f32 elements.
    fn convElemsPerSlot(self: *const QwenFusedRunner) u32 {
        return (self.conv_kernel - 1) * self.conv_dim;
    }

    // -- session slots ------------------------------------------------------

    /// Zero the GDN state and conv regions of a slot for a fresh conversation. `Sessions.create`
    /// calls this on slot reuse. A swallowed failure corrupts the next conversation, so this
    /// panics.
    pub fn resetSlotState(self: *QwenFusedRunner, slot: u32) void {
        std.debug.assert(slot < self.n_slots);

        self.resetSlotStateInner(slot) catch |e|
            std.debug.panic("qwen gpu slot reset failed: {t}", .{e});
    }

    fn resetSlotStateInner(self: *QwenFusedRunner, slot: u32) !void {
        // Size the regions of one slot.
        const F: vk.DeviceSize = @sizeOf(f32);
        const state_bytes = @as(vk.DeviceSize, self.stateElemsPerSlot()) * F;
        const conv_bytes = @as(vk.DeviceSize, self.convElemsPerSlot()) * F;

        // Fill those regions with zeros on every GDN layer.
        const Rec = struct {
            runner: *const QwenFusedRunner,
            slot: u32,
            state_bytes: vk.DeviceSize,
            conv_bytes: vk.DeviceSize,

            pub fn record(r: @This(), cb: vk.CommandBuffer) kernels.Recorded {
                const vkd = r.runner.ctx.vkd;
                for (r.runner.layers) |*layer| {
                    switch (layer.kind) {
                        .gdn => |*w| {
                            vkd.cmdFillBuffer(
                                cb,
                                w.state.buffer,
                                r.slot * r.state_bytes,
                                r.state_bytes,
                                0,
                            );
                            vkd.cmdFillBuffer(
                                cb,
                                w.conv_state.buffer,
                                r.slot * r.conv_bytes,
                                r.conv_bytes,
                                0,
                            );
                        },
                        .full => {},
                    }
                }

                return .{};
            }
        };

        // Free the decode ring, then submit the fill.
        self.invalidateDecode();
        try kernels.submitOnce(
            self.ctx,
            .fill,
            Rec{
                .runner = self,
                .slot = slot,
                .state_bytes = state_bytes,
                .conv_bytes = conv_bytes,
            },
        );
    }

    /// Fork slot state on the device: every GDN region in full, plus the source rows in every KV
    /// region. Rows past `pos` copy stale bytes that attention never reads. `Sessions.clone` calls
    /// this. It panics for the reason `resetSlotState` states.
    pub fn cloneSlotState(
        self: *QwenFusedRunner,
        source_slot: u32,
        target_slot: u32,
        pos: u32,
    ) void {
        std.debug.assert(source_slot < self.n_slots);
        std.debug.assert(target_slot < self.n_slots);
        std.debug.assert(source_slot != target_slot);
        std.debug.assert(pos <= self.max_seq);

        self.cloneSlotStateInner(source_slot, target_slot) catch |e|
            std.debug.panic("qwen gpu slot clone failed: {t}", .{e});
    }

    fn cloneSlotStateInner(self: *QwenFusedRunner, source_slot: u32, target_slot: u32) !void {
        // Size the GDN regions, and list the KV copies.
        const F: vk.DeviceSize = @sizeOf(f32);
        const state_bytes = @as(vk.DeviceSize, self.stateElemsPerSlot()) * F;
        const conv_bytes = @as(vk.DeviceSize, self.convElemsPerSlot()) * F;
        var kv_scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
        const kv_copies = self.kv.cloneCopies(source_slot, target_slot, &kv_scratch);
        std.debug.assert(kv_copies.len == 2 * @as(usize, self.n_full));

        // Copy the GDN regions of every layer, then the KV rows.
        const Rec = struct {
            runner: *const QwenFusedRunner,
            source_slot: u32,
            target_slot: u32,
            state_bytes: vk.DeviceSize,
            conv_bytes: vk.DeviceSize,
            kv_copies: []const kernels.Copy,

            pub fn record(r: @This(), cb: vk.CommandBuffer) kernels.Recorded {
                const vkd = r.runner.ctx.vkd;
                var out: kernels.Recorded = .{};
                for (r.runner.layers) |*layer| {
                    switch (layer.kind) {
                        .gdn => |*w| {
                            copyDeviceToDevice(
                                vkd,
                                cb,
                                w.state.buffer,
                                r.source_slot * r.state_bytes,
                                w.state.buffer,
                                r.target_slot * r.state_bytes,
                                r.state_bytes,
                            );

                            copyDeviceToDevice(
                                vkd,
                                cb,
                                w.conv_state.buffer,
                                r.source_slot * r.conv_bytes,
                                w.conv_state.buffer,
                                r.target_slot * r.conv_bytes,
                                r.conv_bytes,
                            );
                            out.copies += 2;
                            out.bytes += r.state_bytes + r.conv_bytes;
                        },
                        .full => {},
                    }
                }

                for (r.kv_copies) |c| {
                    copyDeviceToDevice(vkd, cb, c.src, c.src_offset, c.dst, c.dst_offset, c.size);
                    out.bytes += c.size;
                }

                out.copies += @intCast(r.kv_copies.len);
                return out;
            }
        };

        // Free the decode ring, then submit the copies.
        self.invalidateDecode();
        try kernels.submitOnce(self.ctx, .clone, Rec{
            .runner = self,
            .source_slot = source_slot,
            .target_slot = target_slot,
            .state_bytes = state_bytes,
            .conv_bytes = conv_bytes,
            .kv_copies = kv_copies,
        });
    }

    // -- prompt-cache snapshot and restore ----------------------------------

    /// Bytes a snapshot at `n` tokens occupies. Per full layer, K[0..n]+V[0..n]
    /// (each `n` rows of `kvRowBytes`). Per GDN layer, `state` and
    /// `conv_state`, which hold a fixed size and stay f32 under any dtype.
    pub fn snapshotBytes(self: *const QwenFusedRunner, n: u32) usize {
        const kv_bytes: usize = 2 * @as(usize, n) * @as(usize, @intCast(self.kvRowBytes()));
        const gdn_elems: usize = @as(usize, self.stateElemsPerSlot()) + self.convElemsPerSlot();
        var total: usize = 0;
        for (self.layers) |*layer| {
            switch (layer.kind) {
                .full => total += kv_bytes,
                // A blob holds the regions of one slot, not the whole slot-strided buffers.
                .gdn => total += gdn_elems * @sizeOf(f32),
            }
        }

        return total;
    }

    /// Fill `out` with the copy list of the state of `slot`, and return the count: the first `n` KV
    /// positions and every GDN region. `to_host` true saves into `stage`, and false restores out of
    /// it. Save and restore share this layout, so the host offsets match.
    fn slotCopies(
        self: *QwenFusedRunner,
        slot: u32,
        n: u32,
        to_host: bool,
        stage: vk.Buffer,
        out: []kernels.Copy,
    ) usize {
        // The direction fixes the legal stage.
        std.debug.assert(to_host == (stage == self.kv_readback_stage.buffer));
        std.debug.assert(slot < self.kv.layout.n_slots);
        std.debug.assert(n <= self.kv.layout.window);

        // Walk the layers in blob order, and advance the host offset per copy.
        const F: vk.DeviceSize = @sizeOf(f32);
        const n_full = self.n_full;
        const kv_bytes: vk.DeviceSize = @as(vk.DeviceSize, n) * self.kv.layout.rowBytes();
        const state_slot: vk.DeviceSize = @as(vk.DeviceSize, self.stateElemsPerSlot()) * F;
        const conv_slot: vk.DeviceSize = @as(vk.DeviceSize, self.convElemsPerSlot()) * F;
        const state_at: vk.DeviceSize = @as(vk.DeviceSize, slot) * state_slot;
        const conv_at: vk.DeviceSize = @as(vk.DeviceSize, slot) * conv_slot;
        var off: vk.DeviceSize = 0;
        var kv_total: u64 = 0;
        var c: usize = 0;
        for (self.layers) |*layer| {
            switch (layer.kind) {
                .full => |*w| {
                    // One K and one V copy at the rebased KV address: the index inside the range,
                    // and the first `n` rows of the slot.
                    std.debug.assert(w.kv_slot < n_full);

                    for ([_]kv_index.Tensor{ .k, .v }) |t| {
                        const dev = self.kv.rowAt(w.kv_slot, t, slot, 0);
                        out[c] = stagedCopy(to_host, stage, off, dev.buffer, dev.offset, kv_bytes);
                        c += 1;
                        off += kv_bytes;
                        kv_total += kv_bytes;
                    }
                },
                .gdn => |*w| {
                    out[c] = stagedCopy(to_host, stage, off, w.state.buffer, state_at, state_slot);
                    off += state_slot;
                    const conv_buf = w.conv_state.buffer;
                    out[c + 1] = stagedCopy(to_host, stage, off, conv_buf, conv_at, conv_slot);
                    off += conv_slot;
                    c += 2;
                },
            }
        }

        // The KV half is the blob the layout frames, and the whole walk fills
        // the blob `snapshotBytes` prices, so a cached blob keeps its framing.
        std.debug.assert(kv_total == self.kv.layout.blobBytes(n));
        std.debug.assert(off == self.snapshotBytes(n));
        return c;
    }

    /// Save the post-`n`-token state of slot 0 into the caller's `out`, which
    /// holds `snapshotBytes(n)` bytes. The copy goes through the readback stage,
    /// which the CPU reads at full rate.
    pub fn snapshotInto(self: *QwenFusedRunner, n: u32, out: []u8) !void {
        std.debug.assert(n <= self.max_seq);
        std.debug.assert(out.len == self.snapshotBytes(n));

        try self.stageCopies(0, n, .{ .to_host = out });
    }

    /// Restore from a caller-held `snapshotInto(n)` blob of this geometry, which the cache
    /// fingerprint of the model guards. Goes through `snap_stage`. Returns the resume position,
    /// which equals `n`.
    pub fn restoreFromBytes(self: *QwenFusedRunner, blob: []const u8, n: u32) !u32 {
        std.debug.assert(n <= self.max_seq);
        std.debug.assert(blob.len == self.snapshotBytes(n));

        log.debug("restore: {d} positions, slot 0", .{n});
        try self.stageCopies(0, n, .{ .to_device = blob });
        return n;
    }

    /// Move the state of `slot` over its first `n` positions through the stage of the direction.
    fn stageCopies(self: *QwenFusedRunner, slot: u32, n: u32, side: kernels.StageSide) !void {
        self.invalidateDecode();
        const stage = switch (side) {
            .to_host => &self.kv_readback_stage,
            .to_device => &self.snap_stage,
        };

        try stage.ensure(@min(self.swap_chunk_bytes, self.snapshotBytes(n)));
        kv_budget.assertStageCap(
            stage.capacity,
            self.swap_chunk_bytes,
            self.snapshotBytes(self.max_seq),
        );
        var copies: [kernels.KvStore.max_segments]kernels.Copy = undefined;
        const count = self.slotCopies(slot, n, side == .to_host, stage.buffer, &copies);
        _ = try kernels.walkStage(self.ctx, copies[0..count], stage, side, self.swap_chunk_bytes);
    }

    // -- tiered kv ----------------------------------------------------------

    /// Bytes one slot's state occupies in a host blob at `n_rows` committed
    /// positions (KV over `n_rows` + every GDN layer's state + conv).
    pub fn evictSlotBytesFor(self: *const QwenFusedRunner, n_rows: u32) u64 {
        return self.snapshotBytes(n_rows);
    }

    /// Copy slot `slot`'s state over its first `n_rows` positions out of VRAM
    /// into `out` (D2H via the readback stage).
    pub fn evictSlot(self: *QwenFusedRunner, slot: u32, n_rows: u32, out: []u8) !void {
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);

        const bytes: usize = @intCast(self.evictSlotBytesFor(n_rows));
        std.debug.assert(out.len == bytes);
        try self.stageCopies(slot, n_rows, .{ .to_host = out });
    }

    /// Restore a previously-evicted `n_rows`-row blob into slot `slot` (H2D).
    pub fn restoreSlot(self: *QwenFusedRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);

        const bytes: usize = @intCast(self.evictSlotBytesFor(n_rows));
        std.debug.assert(blob.len == bytes);
        try self.stageCopies(slot, n_rows, .{ .to_device = blob });
    }

    /// Free the weights and state of one layer, in reverse construction order. Every `Q80Weight`
    /// comes from `Q80Weight.init` and not from a VRAM arena, so each `deinit` is a real free.
    /// The sets stay, because destroying `desc_pool` releases them all.
    fn deinitLayer(layer: *Layer) void {
        switch (layer.kind) {
            .full => |*w| {
                w.k_norm.deinit();
                w.q_norm.deinit();
                w.o.deinit();
                w.v.deinit();
                w.k.deinit();
                w.q.deinit();
            },
            .gdn => |*w| {
                w.conv_state.deinit();
                w.state.deinit();
                w.ssm_norm.deinit();
                w.dt_bias.deinit();
                w.ssm_a.deinit();
                w.conv1d.deinit();
                w.ssm_out.deinit();
                w.beta.deinit();
                w.alpha.deinit();
                w.zgate.deinit();
                w.qkv.deinit();
            },
        }

        layer.ffn_down.deinit();
        layer.ffn_up.deinit();
        layer.ffn_gate.deinit();
        layer.post_norm.deinit();
        layer.attn_norm.deinit();
    }

    pub fn deinit(self: *QwenFusedRunner) void {
        // Free in reverse construction order: the command buffers and the layers first.
        const ctx = self.ctx;
        log.debug(
            "deinit: layers [{d}..{d}), {d} slots",
            .{ self.first_layer, self.first_layer + self.n_layers, self.n_slots },
        );
        self.prof.deinit();
        self.decode.deinit();
        const cmds_to_free = [_]vk.CommandBuffer{self.cmd_buffer};
        ctx.vkd.freeCommandBuffers(ctx.device, ctx.command_pool, &cmds_to_free);

        var i = self.layers.len;
        while (i > 0) {
            i -= 1;
            deinitLayer(&self.layers[i]);
        }

        self.gpa.free(self.layers);

        // Free the head and the descriptor pool.
        self.output_norm.deinit();
        self.lm_head.deinit();
        ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

        // Free the host stages and the host logits.
        self.gpa.free(self.logits);
        self.kv_readback_stage.deinit();
        self.snap_stage.deinit();
        self.stage_out.deinit();
        self.stage_in.deinit();

        // Free the KV cache and the scratch.
        self.kv.deinit();
        self.buf_logits.deinit();
        self.buf_ffn_up.deinit();
        self.buf_ffn_gate.deinit();
        self.buf_attn_split.deinit();
        self.buf_attn_out.deinit();
        self.buf_v_kv.deinit();
        self.buf_k_kv.deinit();
        self.buf_v.deinit();
        self.buf_k.deinit();
        self.buf_gate.deinit();
        self.buf_rope_ff.deinit();

        self.buf_q.deinit();
        self.buf_q_gate.deinit();
        self.buf_mix_o.deinit();
        self.buf_beta.deinit();
        self.buf_alpha.deinit();
        self.buf_z.deinit();
        self.buf_v_conv.deinit();
        self.buf_k_n.deinit();
        self.buf_q_n.deinit();
        self.buf_qkv.deinit();
        self.buf_mix_out.deinit();
        self.buf_x_norm.deinit();
        self.buf_x.deinit();

        // Free the kernels.
        if (self.k_kv_commit) |*k| {
            k.deinit();
        }

        self.k_attn_split.deinit();
        self.k_attn_multi.deinit();
        self.k_attn_batch.deinit();
        self.k_rope_batch.deinit();
        self.k_rmsnorm_batch.deinit();
        self.k_linear_batch.deinit();
        self.k_linear_gemm.deinit();
        self.k_qknorm.deinit();
        self.k_gate.deinit();
        self.k_residual.deinit();
        self.k_swiglu.deinit();
        self.k_rope.deinit();
        self.k_attn.deinit();
        self.k_gdn_chunk.deinit();
        self.k_gdn.deinit();
        self.k_rmsnorm.deinit();
        self.k_linear.deinit();

        self.gpa.destroy(self);
    }
};

// -- helpers ----------------------------------------------------------------

fn devBuf(ctx: *const vulkan_ctx.Context, bytes: vk.DeviceSize) !kernels.DeviceBuffer {
    var b: kernels.DeviceBuffer = .{ .ctx = ctx };
    try b.ensure(bytes);
    return b;
}

/// Zero a device buffer with one `cmdFillBuffer` in one `submitOnce`.
fn zeroDevBuf(ctx: *const vulkan_ctx.Context, b: *kernels.DeviceBuffer) !void {
    const Rec = struct {
        buf: *const kernels.DeviceBuffer,

        pub fn record(r: @This(), cb: vk.CommandBuffer) kernels.Recorded {
            r.buf.ctx.vkd.cmdFillBuffer(cb, r.buf.buffer, 0, r.buf.capacity, 0);
            return .{};
        }
    };
    try kernels.submitOnce(ctx, .fill, Rec{ .buf = b });
}

fn q80(ctx: *const vulkan_ctx.Context, t: *const Tensor) !kernels.Q80Weight {
    if (t.dtype != .q8_0) return error.UnsupportedWeightDtype;
    return kernels.Q80Weight.init(ctx, t.data, @intCast(t.shape[0]), @intCast(t.shape[1]));
}

fn f32buf(ctx: *const vulkan_ctx.Context, t: *const Tensor) !kernels.HostBuffer {
    std.debug.assert(t.dtype == .f32);

    var b: kernels.HostBuffer = .{ .ctx = ctx };
    try b.ensure(t.data.len);
    @memcpy(b.mapped[0..t.data.len], t.data);
    return b;
}

fn bi(b: anytype) vk.DescriptorBufferInfo {
    return .{ .buffer = b.buffer, .offset = 0, .range = b.capacity };
}

// A barrier or copy helper name states both ends of the dependency. `llama_fused.zig` uses the
// same names, so one name never covers two different hazards across the fused runners.

/// Shader-write -> shader-read/write.
fn barrierShader(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mb = [_]vk.MemoryBarrier{.{
        .src_access_mask = .{ .shader_write_bit = true },
        .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true },
    }};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .compute_shader_bit = true },
        .{ .compute_shader_bit = true },
        .{},
        &mb,
        null,
        null,
    );
}

/// Transfer-write -> shader-read.
fn barrierTransferToShader(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mb = [_]vk.MemoryBarrier{.{
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_access_mask = .{ .shader_read_bit = true },
    }};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .transfer_bit = true },
        .{ .compute_shader_bit = true },
        .{},
        &mb,
        null,
        null,
    );
}

/// Shader-write -> transfer-read.
fn barrierShaderToTransfer(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mb = [_]vk.MemoryBarrier{.{
        .src_access_mask = .{ .shader_write_bit = true },
        .dst_access_mask = .{ .transfer_read_bit = true },
    }};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .compute_shader_bit = true },
        .{ .transfer_bit = true },
        .{},
        &mb,
        null,
        null,
    );
}

/// Transfer-write -> transfer-read (the two-hop bounce that compacts a
/// buffer's suffix to its front).
fn barrierTransferToTransfer(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mb = [_]vk.MemoryBarrier{.{
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_access_mask = .{ .transfer_read_bit = true },
    }};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .transfer_bit = true },
        .{ .transfer_bit = true },
        .{},
        &mb,
        null,
        null,
    );
}

fn copyHostToDevice(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    dst: vk.Buffer,
    bytes: vk.DeviceSize,
) void {
    vkd.cmdCopyBuffer(
        cb,
        src,
        dst,
        &[_]vk.BufferCopy{.{ .src_offset = 0, .dst_offset = 0, .size = bytes }},
    );
}

/// Pure full-range copy. The caller records the shader-write ->
/// transfer-read barrier first (`barrierShaderToTransfer`).
fn copyDeviceToHost(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    dst: vk.Buffer,
    bytes: vk.DeviceSize,
) void {
    vkd.cmdCopyBuffer(
        cb,
        src,
        dst,
        &[_]vk.BufferCopy{.{ .src_offset = 0, .dst_offset = 0, .size = bytes }},
    );
}

/// One copy of `size` bytes between `stage` at `stage_off` and `buf` at `buf_off`.
fn stagedCopy(
    to_host: bool,
    stage: vk.Buffer,
    stage_off: vk.DeviceSize,
    buf: vk.Buffer,
    buf_off: vk.DeviceSize,
    size: vk.DeviceSize,
) kernels.Copy {
    if (to_host) {
        return .{
            .src = buf,
            .src_offset = buf_off,
            .dst = stage,
            .dst_offset = stage_off,
            .size = size,
        };
    }

    return .{
        .src = stage,
        .src_offset = stage_off,
        .dst = buf,
        .dst_offset = buf_off,
        .size = size,
    };
}

fn copyDeviceToDevice(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    src_off: vk.DeviceSize,
    dst: vk.Buffer,
    dst_off: vk.DeviceSize,
    bytes: vk.DeviceSize,
) void {
    vkd.cmdCopyBuffer(
        cb,
        src,
        dst,
        &[_]vk.BufferCopy{.{ .src_offset = src_off, .dst_offset = dst_off, .size = bytes }},
    );
}

// -- tests ------------------------------------------------------------------

/// A 2B-shaped card: 24 layers at interval 4 (18 GDN + 6 full attention). The
/// qwen family reasons about one geometry, so `qwen_hybrid.zig` takes this card
/// instead of restating it. That is why the fixture is public.
pub fn testCard() loader.ModelCard {
    return .{
        .arch = .qwen35,
        .architecture = "qwen3.5",
        .n_layers = 24,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
        .embedding_length = 2048,
        .feed_forward_length = 8192,
        .context_length = 262144,
        .vocab_size = 248320,
        .rope_freq_base = 5000000.0,
        .rope_dim_count = 64,
        .rms_norm_eps = 1e-6,
        .bos_token_id = 0,
        .eos_token_id = 0,
        .chat_template = "",
        .gdn = .{
            .conv_kernel = 4,
            .state_size = 128,
            .group_count = 16,
            .inner_size = 2048,
            .time_step_rank = 16,
            .full_attention_interval = 4,
        },
    };
}

test "qwen kv cost: the dtype prices the token and never the fixed state" {
    const card = testCard();
    const c32 = QwenFusedRunner.kvCostModel(card, .f32);
    const c16 = QwenFusedRunner.kvCostModel(card, .f16);
    // The planner stays monotone in cost under three conditions. The fixed state is equal across
    // dtypes. The token price is no larger. The token price is at least one byte.
    try std.testing.expectEqual(c32.fixed_state_bytes_per_slot, c16.fixed_state_bytes_per_slot);
    try std.testing.expectEqual(c32.kv_bytes_per_token, 2 * c16.kv_bytes_per_token);
    try std.testing.expect(c16.kv_bytes_per_token >= 1);
}

test "qwen kv cost: the vector prices attention and recurrent layers apart" {
    const card = testCard();
    // 24 layers at interval 4: layers 3, 7, 11, 15, 19, and 23 are full
    // attention, and the other 18 hold the GDN state and conv slabs.
    const gdn_bytes: u64 = 1_122_304;
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const vector = kv_cost.qwenCostVector(card, kv_dtype);
        const attn_bytes: u64 = @as(u64, 2) * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        try std.testing.expectEqual(@as(u32, 24), vector.n_layers);
        for (vector.layers(), 0..) |entry, i| {
            if ((i + 1) % 4 == 0) {
                try std.testing.expectEqual(attn_bytes, @as(u64, entry.kv_bytes_per_token));
                try std.testing.expectEqual(@as(u32, 0), entry.fixed_bytes_per_slot);
            } else {
                try std.testing.expectEqual(@as(u32, 0), entry.kv_bytes_per_token);
                try std.testing.expectEqual(gdn_bytes, @as(u64, entry.fixed_bytes_per_slot));
            }
        }

        const whole = vector.total();
        try std.testing.expectEqual(6 * attn_bytes, whole.kv_bytes_per_token);
        try std.testing.expectEqual(18 * gdn_bytes, whole.fixed_state_bytes_per_slot);
        try std.testing.expectEqual(QwenFusedRunner.kvCostModel(card, kv_dtype), whole);
    }

    // The dtype halves the per-token column and leaves the fixed column equal,
    // which is what the planner's monotonicity in cost needs.
    const wide_vector = kv_cost.qwenCostVector(card, .f32);
    const half_vector = kv_cost.qwenCostVector(card, .f16);
    const wide = wide_vector.total();
    const half = half_vector.total();
    try std.testing.expectEqual(@as(u64, 24_576), wide.kv_bytes_per_token);
    try std.testing.expectEqual(@as(u64, 12_288), half.kv_bytes_per_token);
    try std.testing.expectEqual(@as(u64, 20_201_472), wide.fixed_state_bytes_per_slot);
    try std.testing.expectEqual(wide.fixed_state_bytes_per_slot, half.fixed_state_bytes_per_slot);
}

test "qwen range: the full-attention count compacts inside a layer range" {
    const g = testCard().gdn.?;
    // 24 layers at interval 4: layers 3, 7, 11, 15, 19, and 23 are full
    // attention. A range holds the ones its own bounds contain.
    try std.testing.expectEqual(@as(u32, 6), QwenFusedRunner.fullCountInRange(g, 0, 24));
    try std.testing.expectEqual(@as(u32, 5), QwenFusedRunner.fullCountInRange(g, 6, 24));
    try std.testing.expectEqual(@as(u32, 3), QwenFusedRunner.fullCountInRange(g, 12, 24));
    try std.testing.expectEqual(@as(u32, 1), QwenFusedRunner.fullCountInRange(g, 23, 24));
    // Negative space: a range below the attention interval holds none, and an
    // empty range holds none wherever it sits.
    try std.testing.expectEqual(@as(u32, 0), QwenFusedRunner.fullCountInRange(g, 0, 3));
    try std.testing.expectEqual(@as(u32, 0), QwenFusedRunner.fullCountInRange(g, 12, 12));
    // The counts of the two halves partition the whole, at every cut.
    var k: u32 = 0;
    while (k <= 24) : (k += 1) {
        const head = QwenFusedRunner.fullCountInRange(g, 0, k);
        const tail = QwenFusedRunner.fullCountInRange(g, k, 24);
        try std.testing.expectEqual(@as(u32, 6), head + tail);
    }
}

test "qwen range: kv_slot rebases on the range and not on the model" {
    const g = testCard().gdn.?;
    // The suffix [12..24) holds three full-attention layers, at KV indices 0, 1, and 2. A
    // whole-model runner gives the same three layers the model-global indices 3, 4, and 5.
    const suffix = [_]u32{ 15, 19, 23 };
    for (suffix, 0..) |li, expect| {
        try std.testing.expectEqual(
            @as(u32, @intCast(expect)),
            QwenFusedRunner.kvSlotInRange(g, 12, li),
        );
        try std.testing.expectEqual(
            @as(u32, @intCast(expect)) + 3,
            QwenFusedRunner.kvSlotInRange(g, 0, li),
        );
    }

    // The whole-model range compacts the full-attention layers to 0..5 in model order.
    var seen: u32 = 0;
    var li: u32 = 0;
    while (li < 24) : (li += 1) {
        if (!g.isFullAttention(li)) continue;
        try std.testing.expectEqual(seen, QwenFusedRunner.kvSlotInRange(g, 0, li));
        seen += 1;
    }

    try std.testing.expectEqual(@as(u32, 6), seen);
    // Every index a range assigns stays inside that range's KV height, which is
    // the bound the V-half address depends on.
    const first: u32 = 12;
    const height = QwenFusedRunner.fullCountInRange(g, first, 24);
    li = first;
    while (li < 24) : (li += 1) {
        if (g.isFullAttention(li)) {
            try std.testing.expect(QwenFusedRunner.kvSlotInRange(g, first, li) < height);
        }
    }
}

test "qwen range: the preflight price equals the planner's span" {
    const card = testCard();
    const ranges = [_][2]u32{
        .{ 0, 24 },
        .{ 6, 24 },
        .{ 12, 24 },
        .{ 23, 24 },
        .{ 0, 12 },
        .{ 12, 12 },
    };
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        for ([_]u32{ 512, 8192 }) |window| {
            const kv: kv_budget.KvStructure = .{
                .n_slots = 4,
                .window = window,
                .costs = kv_cost.qwenCostVector(card, kv_dtype),
            };
            for (ranges) |r| {
                const per_slot = QwenFusedRunner.rangeSlotBytes(card, kv_dtype, window, r[0], r[1]);
                try std.testing.expectEqual(
                    kv_budget.slotSpanBytes(kv, r[0], r[1]),
                    per_slot.total(),
                );
                // The KV half is what the regions of the range hold for one slot. The fixed half
                // is what the GDN layers hold at any position.
                const n_full = QwenFusedRunner.fullCountInRange(card.gdn.?, r[0], r[1]);
                const layer_bytes: u64 = @as(u64, window) * kv_dtype.rowBytes(
                    card.n_kv_heads * card.head_dim,
                );
                try std.testing.expectEqual(@as(u64, 2) * n_full * layer_bytes, per_slot.kv);
                // A blob of n rows holds the KV half of the n-row price. A range without full
                // attention holds no regions.
                if (n_full == 0) continue;

                const layout = kv_index.Layout.init(
                    kv.n_slots,
                    window,
                    card.n_kv_heads * card.head_dim,
                    n_full,
                    kv_dtype,
                );
                try std.testing.expectEqual(kv.n_slots * per_slot.kv, layout.totalBytes());
                try std.testing.expectEqual(per_slot.kv, layout.slotBytes());
                for ([_]u32{ 1, 37, window }) |n_rows| {
                    const blob = QwenFusedRunner.rangeSlotBytes(card, kv_dtype, n_rows, r[0], r[1]);
                    try std.testing.expectEqual(blob.kv, layout.blobBytes(n_rows));
                }
            }

            // The two halves of a cut partition the whole model's slot price.
            const head = QwenFusedRunner.rangeSlotBytes(card, kv_dtype, window, 0, 12);
            const tail = QwenFusedRunner.rangeSlotBytes(card, kv_dtype, window, 12, 24);
            const whole = QwenFusedRunner.rangeSlotBytes(card, kv_dtype, window, 0, 24);
            try std.testing.expectEqual(whole.total(), head.total() + tail.total());
        }
    }
}

test "qwen suffix weights: the whole model is the range at first zero" {
    const data align(@alignOf(f32)) = [_]u8{0} ** 64;
    // Four layers of 16 bytes each, plus a 32-byte head pair.
    const layer_t: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 4, 0, 0, 0 },
        .byte_stride = .{ 16, 0, 0, 0 },
        .data = data[0..16],
    };
    const head_t: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 8, 0, 0, 0 },
        .byte_stride = .{ 32, 0, 0, 0 },
        .data = data[0..32],
    };
    const tied = [_]loader.TensorRef{
        .{ .name = "blk.0.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.1.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.2.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.3.ffn_down.weight", .tensor = layer_t },
        .{ .name = "output_norm.weight", .tensor = layer_t },
        .{ .name = "token_embd.weight", .tensor = head_t },
    };
    // The tied head charges the embedding table, which init uploads as the lm_head. So the figure
    // is every tensor the model carries.
    const whole = QwenFusedRunner.weightsVramBytesRange(&tied, 4, 0, 4);
    try std.testing.expectEqual(@as(u64, 4 * 16 + 16 + 32), whole);
    var every_tensor: u64 = 0;
    for (tied) |r| {
        every_tensor += r.tensor.data.len;
    }

    try std.testing.expectEqual(every_tensor, whole);
    // A suffix drops the prefix layers and keeps the head whole, because the
    // suffix runs the head itself.
    try std.testing.expectEqual(
        @as(u64, 2 * 16 + 48),
        QwenFusedRunner.weightsVramBytesRange(&tied, 4, 2, 4),
    );
    try std.testing.expectEqual(
        @as(u64, 48),
        QwenFusedRunner.weightsVramBytesRange(&tied, 4, 4, 4),
    );
    // The layer halves of a cut partition the whole, once the head is removed
    // from the double count.
    const head_bytes: u64 = QwenFusedRunner.weightsVramBytesRange(&tied, 4, 0, 0);
    var k: u32 = 0;
    while (k <= 4) : (k += 1) {
        const halves = QwenFusedRunner.weightsVramBytesRange(&tied, 4, 0, k) +
            QwenFusedRunner.weightsVramBytesRange(&tied, 4, k, 4);
        try std.testing.expectEqual(whole + head_bytes, halves);
    }

    // Untied: `output.weight` is the head, and the embedding table stays host side, so it charges
    // nothing.
    const untied = [_]loader.TensorRef{
        .{ .name = "blk.0.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.1.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.2.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.3.ffn_down.weight", .tensor = layer_t },
        .{ .name = "output_norm.weight", .tensor = layer_t },
        .{ .name = "token_embd.weight", .tensor = head_t },
        .{ .name = "output.weight", .tensor = layer_t },
    };
    var untied_every: u64 = 0;
    for (untied) |r| {
        untied_every += r.tensor.data.len;
    }

    const untied_whole = QwenFusedRunner.weightsVramBytesRange(&untied, 4, 0, 4);
    try std.testing.expectEqual(@as(u64, 4 * 16 + 16 + 16), untied_whole);
    try std.testing.expectEqual(untied_every - head_t.data.len, untied_whole);
}

test "qwen scratch: only a narrower dtype adds the commit pair" {
    const card = testCard();
    const s32 = QwenFusedRunner.scratchVramBytes(card, .f32);
    const s16 = QwenFusedRunner.scratchVramBytes(card, .f16);
    const per_step: u32 = card.n_kv_heads * card.head_dim;
    const rows: u64 = QwenFusedRunner.prefill_rows_max;
    const pair: u64 = rows * 2 * dtype.KvDtype.f16.rowBytes(per_step);
    try std.testing.expectEqual(s32 + pair, s16);
}

test "qwen scratch: the split attention records follow the declared context" {
    const deep = testCard();
    var shallow = testCard();
    shallow.context_length = limits.attn_split_len;
    const records_deep = QwenFusedRunner.attnSplitScratch(deep).totalBytes();
    const records_shallow = QwenFusedRunner.attnSplitScratch(shallow).totalBytes();
    // The records are the only context term, and they are f32 at every dtype.
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const scratch_deep = QwenFusedRunner.scratchVramBytes(deep, kv_dtype);
        const scratch_shallow = QwenFusedRunner.scratchVramBytes(shallow, kv_dtype);
        try std.testing.expectEqual(records_deep - records_shallow, scratch_deep - scratch_shallow);
    }

    // The anchors: 4 rows x 16 heads x 1024 splits x 258 elements of 4 bytes,
    // on top of 128 activation rows of 60448 f32 elements.
    try std.testing.expectEqual(@as(u64, 67_633_152), records_deep);
    try std.testing.expectEqual(
        @as(u64, 30_949_376 + 67_633_152),
        QwenFusedRunner.scratchVramBytes(deep, .f32),
    );
    // Negative space: a card without GDN metadata never builds this runner,
    // so it prices no records either.
    var dense = testCard();
    dense.gdn = null;
    try std.testing.expectEqual(@as(u64, 0), QwenFusedRunner.scratchVramBytes(dense, .f32));
}
