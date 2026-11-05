//! `QwenFusedRunner` records the whole Qwen3.5 hybrid forward for one token into
//! one Vulkan command buffer and submits it once. It is the Qwen counterpart of
//! the Llama-only `llama_fused.LlamaFusedRunner`. Activations stay in VRAM end to
//! end. The GDN recurrent state lives in persistent device buffers, where it
//! does for the recurrent layers what the KV cache does for attention. Weights
//! stage to VRAM once at init, and the descriptor sets bind at the same time.
//!
//! One command buffer per token removes the host round-trips. The per-op path
//! (`ops_vulkan/wrap.zig`, used by `QwenRunner`) submits and waits about 400
//! times per token. Weights stage through `Q80Weight.init` and not through a
//! VRAM arena. `limits.vramPlanFits` refuses an over-budget configuration at
//! init, before the first device allocation.
//!
//! Three entry shapes exist. `step` advances one token on slot 0, which is the
//! CLI decode. `prefillBatch` runs one sequence chunk on slot 0 through the
//! chunk-parallel GDN kernels. `stepBatch` advances a bounded batch of
//! (slot, token, pos) rows over per-slot KV and GDN-state regions: the
//! projections batched, the GDN recurrence one dispatch per same-slot run, and
//! rope and attention per row.
//!
//! Each shape also takes a hidden form, which reads activation rows the caller
//! already computed. A runner built at `first_layer > 0` holds a layer suffix,
//! and `qwen_hybrid.zig` feeds it the residual stream of a host layer prefix.
//!
//! The kernel set and the tensor layout come from the CPU `QwenRunner` and the
//! tolerance-checked GPU kernels GdnDecode, Attention@256, QwenQkNorm, and SigmoidGate
//! (see `ops/gdn.zig`).

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
    /// The split pass of a row past one split binds q, the whole K region, the
    /// whole V region, and the records, as the multi-stream set does.
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

    /// Layers this runner HOLDS, which is `layers.len`. A suffix runner holds
    /// `[first_layer .. first_layer + n_layers)` of the model, so this is the
    /// model's layer count only at `first_layer == 0`.
    n_layers: u32,
    /// The absolute model index of `layers[0]`. A hybrid gives the GPU a suffix
    /// and keeps the layer prefix on the host. Zero is the whole-model runner
    /// every shipped entry point builds.
    first_layer: u32,
    /// Full-attention layers inside this runner's range: the layer count of
    /// `kv.layout`. It is a field and not a rescan, because the per-layer
    /// record loop reads it once per layer per token.
    n_full: u32,
    hidden: u32,
    ffn: u32,
    vocab: u32,
    max_seq: u32,
    /// Session slots: per-slot rows in every full-attn KV region + per-slot
    /// GDN state regions inside the slot-strided buffers. 1 == the
    /// single-sequence CLI shape. A value above 1 is the serve engine
    /// (`Sessions(QwenFusedRunner)`). Decoupled from the decode width, so a KV
    /// budget buys breadth past `limits.max_batch`.
    n_slots: u32,
    /// Decode width: the most rows one tick advances. At most `n_slots` and at
    /// most `limits.max_batch`.
    n_streams_max: u32,
    /// The most logits rows one call produces: the decode rows plus the final
    /// prefill row of a composed tick, capped by the slot count. Sizes
    /// `buf_logits`, `stage_out`, and `logits`.
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
    /// The element type every full-attention KV row is framed at. `kvRowBytes`
    /// sizes the commit scratch, and `kv.layout` places every KV row. The GDN
    /// recurrent state is not KV and stays f32 under any dtype.
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
    /// The split attention passes. A decode row past one split walks its depth
    /// in splits and merges the partial records in split order.
    k_attn_split: kernels.AttentionSplit,
    /// The record box of the split rows in flight, sized at the declared
    /// context, so every row the window admits fits it.
    attn_scratch: attn_split.Scratch,
    /// The one merge set: the records and `buf_attn_out`. Both buffers never
    /// move, so one set serves every full-attention layer.
    set_attention_merge: vk.DescriptorSet,
    /// The commit narrowing kernel, present only at a non-f32 `kv_dtype`. One
    /// dispatch per layer per entry point narrows that layer's whole K and V
    /// commit scratch, whatever the row count.
    k_kv_commit: ?kernels.KvCommitF16,
    /// The one descriptor set the narrowing dispatch binds: the two f32
    /// scratch buffers as sources, the two narrowed ones as destinations. The
    /// buffers never move, so one set serves every layer and every entry point.
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
    /// RoPE per-pair freq divisors (rope_dim/2 f32). Qwen3.5 has no
    /// `rope_freqs.weight`, so this stays all-ones. It is bound at binding 1
    /// of the rope sets to match the shared rope shaders.
    buf_rope_ff: kernels.HostBuffer,
    buf_gate: kernels.DeviceBuffer,
    buf_k: kernels.DeviceBuffer,
    buf_v: kernels.DeviceBuffer,
    /// The narrowed commit scratch, allocated only at a non-f32 `kv_dtype`.
    /// `KvCommitF16` reads the two f32 rope scratch buffers above and writes
    /// these, and the commit copy lists then scatter dtype-sized regions out of
    /// them. They are zero-capacity buffers at f32, where no dispatch runs.
    buf_k_kv: kernels.DeviceBuffer,
    buf_v_kv: kernels.DeviceBuffer,
    buf_attn_out: kernels.DeviceBuffer,
    buf_attn_split: kernels.DeviceBuffer,
    buf_ffn_gate: kernels.DeviceBuffer,
    buf_ffn_up: kernels.DeviceBuffer,
    buf_logits: kernels.DeviceBuffer,
    /// The full-attention KV cache: one region per full-attention layer of
    /// the range and tensor, each holding `[slot][max_seq][per_step]` of
    /// `kv_dtype`. `kv.layout` computes every host offset and stride. The
    /// single-stream paths address slot 0.
    kv: kernels.KvStore,

    output_norm: kernels.HostBuffer, // f32
    lm_head: kernels.Q80Weight, // tied to embed
    set_output_norm: vk.DescriptorSet,
    set_lm_head: vk.DescriptorSet,

    layers: []Layer,

    desc_pool: vk.DescriptorPool,
    /// The command buffer of every submit except a decode step: prefill,
    /// `stepBatch`, the profiler, and the bench probes.
    cmd_buffer: vk.CommandBuffer,
    /// The decode steps of `step` and `stepHidden`, over two buffers of their
    /// own. Every other submit invalidates it first.
    decode: decode_ring.DecodeRing,
    /// True after init binds the last descriptor set. A recording in the ring
    /// stays valid only while no set changes, so no set changes after init.
    sets_fixed: bool,
    stage_in: kernels.HostBuffer,
    stage_out: kernels.HostBuffer,
    snap_stage: kernels.HostBuffer, // host-visible upload stage (H2D restores)
    /// Separate staging for every device-to-host copy (eviction and the
    /// prompt-cache snapshot). Both stages use host-cached memory when the
    /// driver offers it. Two stages and not one, because a readback must not
    /// wait for a restore that still uses `snap_stage`.
    kv_readback_stage: kernels.HostBuffer,
    /// The chunk one staged copy moves. The planner prices
    /// `limits.swap_stage_chunk_bytes`. Only the `bench-swap` chunk sweep sets
    /// another value, and a value past the blob moves the blob in one copy.
    /// The hybrid runners hold no such field: they pass
    /// `limits.swap_stage_chunk_bytes` to `kernels.walkStage` directly.
    /// `init` sets it, because this runner assigns its fields one by one.
    swap_chunk_bytes: u64,
    /// Heap logits: row i of a `stepBatch` call at `logits[i * vocab ..]`, the
    /// layout the session surface in `runtime/session.zig` reads back. `step`
    /// and `prefillBatch` return row 0.
    logits: []f32,

    /// The timestamps of `cmd_buffer`, at the process level. The profile
    /// commands raise it to `phases` for one submit.
    prof: gpu_profile.GpuProfile,
    /// The runner kind of the GPU records: a suffix runner records for the
    /// hybrid runner that owns it.
    gpu_runner: span.GpuRunner,

    /// Host time of the single-row decode host steps, summed since the last reset.
    /// `generate.decode` clears it before a run and reads it after.
    host: step_stats.DecodeHost,

    /// True while `cmd_buffer` holds a decode recording that `probeSubmitWait`
    /// can resubmit. Only the `bench-decode-host` hooks set it.
    probe_recorded: bool,

    /// The GDN state and conv buffers are a running accumulator with no position
    /// axis, so a reused slot inherits the previous conversation unless it is
    /// scrubbed. The session surface reads this marker and then needs both
    /// `resetSlotState` and `cloneSlotState` from this runner.
    pub const has_recurrent_state = true;

    /// Full-attention layers inside the ABSOLUTE layer range `[from..to)`.
    /// `isFullAttention` takes the absolute index at every call, because the
    /// mixer a layer runs is a property of the model and never of the range.
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

    /// The KV index the full-attention layer at absolute index `li` takes
    /// inside a range that starts at `from`: the full-attention layers below it
    /// and inside the range. This is the compaction invariant a rebased runner
    /// lives by. A model-global index against a range-sized KV store reads the
    /// region of another layer, and only an index past `n_full` trips the
    /// `regionIndex` bound. So the closed form is stated once here and checked
    /// against the walking counter at init.
    fn kvSlotInRange(g: loader.GdnParams, from: u32, li: u32) u32 {
        std.debug.assert(from <= li);
        std.debug.assert(g.isFullAttention(li));

        return fullCountInRange(g, from, li);
    }

    /// What one KV slot costs on this model: KV bytes on the full-attention
    /// layers only, plus the recurrent state and conv ring every GDN layer holds
    /// per slot, which no window can change. The caller states the KV dtype,
    /// which prices the per-token KV rows alone. The fixed per-slot state stays
    /// `@sizeOf(f32)` under any dtype. The planner stays monotone in cost only
    /// while that fixed state is equal across dtypes, so a dtype that moved it
    /// breaks the argument. The planner states the price, so the CLI reads the
    /// same figure before a runner exists.
    pub fn kvCostModel(card: loader.ModelCard, kv_dtype: dtype.KvDtype) kv_budget.KvCostModel {
        std.debug.assert(card.gdn != null);

        return kv_budget.costModel(card.kvShape(), kv_dtype);
    }

    /// What one slot holds over a layer range, split the way the preflight logs
    /// it. `total` is the figure `kv_budget.slotSpanBytes` prices for the same
    /// range, and the preflight asserts the two agree.
    pub const RangeSlotBytes = struct {
        kv: u64,
        state: u64,
        conv: u64,

        pub fn total(self: RangeSlotBytes) u64 {
            return self.kv + self.state + self.conv;
        }
    };

    /// The device bytes one slot occupies over the ABSOLUTE layer range
    /// `[from..to)`: KV rows on the full-attention layers, plus the recurrent
    /// state and conv ring every GDN layer holds whatever the position. It is
    /// the runner side of the per-layer cost vector, so the planner and the
    /// preflight price one range through two derivations and compare them.
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
        return .{
            .kv = 2 * n_full * @as(u64, window) * kv_dtype.rowBytes(
                card.n_kv_heads * card.head_dim,
            ),
            .state = n_gdn * state_elems * @sizeOf(f32),
            .conv = n_gdn * conv_elems * @sizeOf(f32),
        };
    }

    /// Rows one prefill dispatch holds, and the batch factor of the activation
    /// scratch.
    pub const prefill_rows_max: u32 = 128;

    /// The device bytes the activation scratch occupies: the per-row forward
    /// buffers `init` allocates, `prefill_rows_max` rows each. Must match that
    /// allocation list, so the planner's reservation is the preflight's sum. A
    /// non-f32 KV dtype adds the narrowed commit scratch, one K row and one V
    /// row of the storage element beside the f32 pair the matmuls write, so the
    /// plan prices what that dtype really allocates. The split attention
    /// records add `attnSplitScratch`, which is f32 at every KV dtype. A card
    /// without GDN metadata never builds this runner, so it prices nothing.
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

    /// The split attention records of this model. The box is sized at the
    /// declared context and not at the window, because the planner subtracts
    /// the scratch before it derives the window. It does not depend on the
    /// layer range, so a suffix at any split allocates what the whole model
    /// prices. The pricing and the allocation both read this one value.
    fn attnSplitScratch(card: loader.ModelCard) attn_split.Scratch {
        return attn_split.Scratch.init(card.n_heads, card.head_dim, card.context_length);
    }

    /// The device bytes the uploaded weights occupy: every tensor `init` stages
    /// to VRAM for the whole model.
    pub fn weightsVramBytes(loaded: *const loader.Loaded) u64 {
        return weightsVramBytesRange(loaded.tensors, loaded.card.n_layers, 0, loaded.card.n_layers);
    }

    /// `weightsVramBytes` over the ABSOLUTE layer range `[from..to)`: the range
    /// walk `init` uploads, plus the head. The head goes with any range, because a
    /// suffix ends at the last layer and runs the final norm and the lm_head
    /// itself. `token_embd` stays host side for the embedding lookup, so
    /// `headWeightBytes` charges it only where it doubles as the tied lm_head,
    /// which is the tensor `init` uploads there.
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

    /// The planner inputs this model implies: the reservations `init` is about
    /// to make, and the free device-local bytes the caller measured. A caller
    /// that measured none states zero, which keeps the planner off the budget.
    /// `logit_rows` is the logits reservation in rows. `init` allocates
    /// `min(n_slots, n_streams + 1)` rows, so the caller reserves at least
    /// that (the decode width plus the composed tick's final prefill row). The
    /// KV dtype moves the scratch reservation and the two default windows. The
    /// KV price itself lives in `kvCostModel`, which the planner takes
    /// separately. `ceiling` is the model ceiling the caller resolved through
    /// `limits.contextCeiling`.
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

    /// A suffix runner executes layers `[first..n_layers)` with weights, KV, GDN
    /// state, and descriptor bindings sized to match, which is what
    /// `qwen_hybrid.zig` composes through. The whole init chain funnels here,
    /// so one body sizes and frames every buffer.
    ///
    /// Two range facts hold this constructor together. `isFullAttention` keeps
    /// the ABSOLUTE layer index, so a suffix layer runs the mixer the whole
    /// model gives it, and every tensor is found by that index. `kv_slot`
    /// compacts WITHIN the range, so it counts the full-attention layers below
    /// its layer inside `[first..n_layers)`. A model-global index against a
    /// range-sized KV store addresses another layer's region and never fails a
    /// Vulkan bound, so every KV site pairs its index against `self.n_full` and
    /// its row span against the window.
    pub fn initSuffix(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        first: u32,
        kv_dtype: dtype.KvDtype,
    ) !*QwenFusedRunner {
        const c = loaded.card;
        if (c.arch != .qwen35) return error.NotQwen;

        const g = c.gdn orelse return error.NotQwen;
        const refs = loaded.tensors;
        // Slots bound breadth (resident conversations), streams bound width
        // (rows per tick), and a row owns a slot.
        std.debug.assert(plan.n_slots >= 1);
        std.debug.assert(plan.n_slots <= limits.max_vram_slots);
        std.debug.assert(plan.n_streams >= 1);
        std.debug.assert(plan.n_streams <= limits.max_batch);
        std.debug.assert(plan.n_streams <= plan.n_slots);
        // The range ends at the model's last layer, so the runner always owns
        // the head. An empty range owns a head with nothing to feed it.
        std.debug.assert(first < c.n_layers);
        const n_end: u32 = c.n_layers;

        const self = try gpa.create(QwenFusedRunner);
        errdefer gpa.destroy(self);

        self.gpa = gpa;
        self.ctx = ctx;
        self.first_layer = first;
        self.n_layers = n_end - first;
        self.hidden = c.embedding_length;
        self.ffn = c.feed_forward_length;
        self.vocab = c.vocab_size;
        // The window and the slot count arrive resolved: the caller plans them
        // from the byte budget (`core/kv_budget.zig`), because the same
        // arithmetic has to serve the plan and the preflight below.
        self.max_seq = plan.window;
        self.n_slots = plan.n_slots;
        self.n_streams_max = plan.n_streams;
        // Logits rows: every decode row plus the final prefill row of a
        // composed tick wants logits, and each such row occupies its own slot.
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
        // The f16 batch shader unpacks word pairs (`KV_LOAD2`), so a row must
        // hold an even element count. Every registered qwen model does
        // (head_dim 256), and the assert states the bound.
        if (kv_dtype != .f32) std.debug.assert(self.per_step % 2 == 0);

        self.embed = try findTensor(refs, "token_embd.weight");

        // Full-attention layer count INSIDE the range: only these layers hold a
        // KV cache (the GDN layers keep recurrent state instead). Every KV
        // address in this runner rebases on it, so it lands in a field before
        // the first buffer sizes itself.
        self.n_full = fullCountInRange(g, first, n_end);
        // A range that ends at the last layer holds at least one full-attention
        // layer on every registered qwen, because the layer count is a multiple
        // of the attention interval. A KV layout of no layers holds no region,
        // so the bound is stated rather than assumed.
        std.debug.assert(self.n_full >= 1);
        std.debug.assert(self.n_full <= self.n_layers);

        // Every full-attention layer of the range holds one K region and one V
        // region, and the recurrent state lives in other buffers. The device
        // limits bind one region, so the guard refuses before the first
        // allocation.
        const kv_layout = kv_index.Layout.init(
            self.n_slots,
            self.max_seq,
            self.per_step,
            self.n_full,
            kv_dtype,
        );
        try ctx.guardKvLayout(kv_layout);
        // The plan prices the split attention records at every split, so they
        // are allocated and checked at every split too. The range holds at
        // least one full-attention layer (asserted above), so a split set
        // binds the records.
        try kernels.AttentionSplit.guardHeadDim(c.head_dim);
        try kernels.GdnChunkParallel.guardHeadDim(g.state_size);
        const attn_scratch = attnSplitScratch(c);
        std.debug.assert(attn_split.splitCount(self.max_seq) <= attn_scratch.max_splits);
        try ctx.guardAttnSplitScratch(attn_scratch);
        self.attn_scratch = attn_scratch;

        // VRAM preflight: refuse a configuration that does not fit the live
        // device-local budget before the first allocation, with an actionable
        // message. An over-budget plan otherwise drives vkAllocateMemory to
        // OutOfDeviceMemory mid-init, which removes the device on the target
        // driver. Every figure prices the RANGE, so a suffix reserves what a
        // suffix uploads.
        {
            const mib = 1024 * 1024;
            const per_slot = rangeSlotBytes(c, kv_dtype, self.max_seq, first, n_end);
            // The planner and the runner must price one arithmetic: the cost
            // vector's span over this range, and the sizes the allocations
            // below take, are two derivations of one number.
            const kv_struct: kv_budget.KvStructure = .{
                .n_slots = self.n_slots,
                .window = self.max_seq,
                .costs = kv_cost.qwenCostVector(c, kv_dtype),
            };
            std.debug.assert(per_slot.total() == kv_budget.slotSpanBytes(kv_struct, first, n_end));
            std.debug.assert(per_slot.kv == kv_layout.slotBytes());
            const kv_full: u64 = @as(u64, self.n_slots) * per_slot.kv;
            // The regions hold exactly the priced KV, and the plan adds the
            // fixed recurrent state of every slot on top.
            std.debug.assert(kv_layout.totalBytes() == kv_full);
            const gdn_state: u64 = @as(u64, self.n_slots) * per_slot.state;
            const gdn_conv: u64 = @as(u64, self.n_slots) * per_slot.conv;
            const logits_bytes: u64 = logitsVramBytes(self.vocab, self.logit_rows_max);
            const weights_bytes: u64 = weightsVramBytesRange(refs, c.n_layers, first, n_end);
            const scratch_bytes: u64 = scratchVramBytes(c, kv_dtype);
            // The plan states the reservations this preflight measures. Its
            // logits figure can exceed the rows allocated here, because a
            // budget caller reserves the width before the planner resolves the
            // slot count that can cap it.
            std.debug.assert(plan.vram.kv == kv_full + gdn_state + gdn_conv);
            std.debug.assert(plan.vram.weights == weights_bytes);
            std.debug.assert(plan.vram.logits >= logits_bytes);
            std.debug.assert(plan.vram.scratch == scratch_bytes);
            const planned: u64 = weights_bytes + kv_full + gdn_state + gdn_conv + logits_bytes +
                scratch_bytes;
            const available: u64 = ctx.deviceLocalAvailableBytes();
            log.debug(
                "vram preflight: planned {d} MiB (weights {d}, kv {d}, gdn {d}, scratch " ++
                    "{d}), available {d} MiB",
                .{ planned / mib, weights_bytes / mib, kv_full / mib, (gdn_state + gdn_conv) / mib, scratch_bytes / mib, available / mib },
            );
            if (!limits.vramPlanFits(planned, available)) {
                // The fit reads the same live budget and the same reservations
                // as the check above, priced over this runner's range.
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

        // Kernels. Every resource below carries its own errdefer, because
        // `deinit` cannot serve a partial construction: kernel pipelines and
        // pool handles are undefined until their line runs.
        self.k_linear = try kernels.LinearQ80Reduce.init(ctx);
        errdefer self.k_linear.deinit();

        self.k_rmsnorm = try kernels.Rmsnorm.init(ctx);
        errdefer self.k_rmsnorm.deinit();

        self.k_gdn = try kernels.GdnDecode.init(ctx);
        errdefer self.k_gdn.deinit();

        self.k_gdn_chunk = try kernels.GdnChunkParallel.init(ctx);
        errdefer self.k_gdn_chunk.deinit();

        // The four attention kernels read the KV regions, so each takes the
        // storage element. At f32 the selector picks the shipped pipeline.
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

        self.k_kv_commit = null;
        self.set_kv_commit = .null_handle;
        if (kv_dtype != .f32) {
            self.k_kv_commit = try kernels.KvCommitF16.init(ctx);
        }

        errdefer if (self.k_kv_commit) |*k| k.deinit();

        self.prefill_batch_max = prefill_rows_max;

        // Scratch buffers, sized for up to prefill_batch_max rows (decode is
        // row 0). The same descriptor sets serve decode (b=1) and prefill
        // (b<=pb), because the batch kernels read [b, dim] through a `b` push.
        // `buf_logits` is the one exception: it holds `logit_rows_max` rows,
        // which is what the compact logits suffix needs.
        const F = @sizeOf(f32);
        const pb: u64 = self.prefill_batch_max;
        self.buf_x = try devBuf(ctx, pb * self.hidden * F);
        errdefer self.buf_x.deinit();

        self.buf_x_norm = try devBuf(ctx, pb * self.hidden * F);
        errdefer self.buf_x_norm.deinit();

        self.buf_mix_out = try devBuf(ctx, pb * self.hidden * F);
        errdefer self.buf_mix_out.deinit();

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

        // The narrowed commit scratch: the same row count at the storage
        // element. It exists only where a dispatch narrows into it, so the f32
        // path allocates nothing new.
        self.buf_k_kv = .{ .ctx = ctx };
        self.buf_v_kv = .{ .ctx = ctx };
        errdefer self.buf_k_kv.deinit();
        errdefer self.buf_v_kv.deinit();

        if (kv_dtype != .f32) {
            const kv_scratch: vk.DeviceSize = pb * self.kvRowBytes();
            try self.buf_k_kv.ensure(kv_scratch);
            try self.buf_v_kv.ensure(kv_scratch);
        }

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

        // KV cache, one region per full-attention layer of the RANGE and
        // tensor. The commit copies move scratch rows of `kvRowBytes()` into
        // rows of the layout.
        std.debug.assert(kv_layout.rowBytes() == self.kvRowBytes());
        self.kv = try kernels.KvStore.init(ctx, kv_layout);
        errdefer self.kv.deinit();

        const kv_mib: f64 = @as(f64, @floatFromInt(self.kv.deviceBytes())) / (1024.0 * 1024.0);
        log.debug(
            "init qwen fused: layers [{d}..{d}) ({d} full, {d} gdn), {d} slots, max_seq " ++
                "{d}, kv {d:.0} MiB",
            .{ self.first_layer, self.first_layer + self.n_layers, self.n_full, self.n_layers - self.n_full, self.n_slots, self.max_seq, kv_mib },
        );

        self.stage_in = .{ .ctx = ctx };
        try self.stage_in.ensure(pb * self.hidden * F); // up to prefill_batch_max embeddings
        errdefer self.stage_in.deinit();

        self.stage_out = .{ .ctx = ctx };
        try self.stage_out.ensure(@as(vk.DeviceSize, self.logit_rows_max) * self.vocab * F);
        errdefer self.stage_out.deinit();

        self.snap_stage = .{ .ctx = ctx }; // sized lazily on first snapshot
        self.kv_readback_stage = .{ .ctx = ctx }; // sized lazily on first evict
        self.swap_chunk_bytes = limits.swap_stage_chunk_bytes;
        self.host = .{};
        self.probe_recorded = false;
        self.sets_fixed = false;
        self.logits = try gpa.alloc(f32, @as(usize, self.logit_rows_max) * self.vocab);
        errdefer gpa.free(self.logits);

        // Descriptor pool sized generously for all pre-bound sets.
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .storage_buffer, .descriptor_count = 8192 },
        };
        const pool_info: vk.DescriptorPoolCreateInfo = .{
            .max_sets = 1024,
            .pool_size_count = 1,
            .p_pool_sizes = &pool_sizes,
        };
        self.desc_pool = try ctx.vkd.createDescriptorPool(ctx.device, &pool_info, null);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

        // The lm_head is a separate `output.weight` (9B) or tied to
        // `token_embd` (2B).
        self.lm_head = try q80(ctx, findTensor(refs, "output.weight") catch self.embed);
        errdefer self.lm_head.deinit();

        self.output_norm = try f32buf(ctx, try findTensor(refs, "output_norm.weight"));
        errdefer self.output_norm.deinit();

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

        // kv commit: (k, v, k_kv, v_kv). The sources are the f32 rope scratch,
        // the destinations the same rows at the storage element. Bound once,
        // because the buffers never move.
        if (self.k_kv_commit) |*k| {
            self.set_kv_commit = try self.mkSet(
                k.pipeline.set_layout,
                &.{ bi(self.buf_k), bi(self.buf_v), bi(self.buf_k_kv), bi(self.buf_v_kv) },
            );
        }

        // Split merge: (the records, attn_out). The guard above proved that
        // the records fit one binding, and every set binds them whole.
        std.debug.assert(self.buf_attn_split.capacity == attn_scratch.totalBytes());
        std.debug.assert(self.buf_attn_split.capacity <= ctx.buffer_limits.storage_range);
        self.set_attention_merge = try self.mkSet(
            self.k_attn_split.merge.set_layout,
            &.{ bi(self.buf_attn_split), bi(self.buf_attn_out) },
        );

        self.layers = try gpa.alloc(Layer, self.n_layers);
        errdefer gpa.free(self.layers);

        // The watermark walk frees the layers a failure leaves complete. The
        // per-field errdefers inside the loop body cover the layer the failure
        // interrupts, and they die with the iteration, so the two never
        // overlap.
        var layers_built: usize = 0;
        errdefer {
            var j = layers_built;
            while (j > 0) {
                j -= 1;
                deinitLayer(&self.layers[j]);
            }
        }

        var name_buf: [80]u8 = undefined;
        // The KV index the next full-attention layer takes. It counts inside the
        // RANGE, and the closed form checks it at every assignment.
        var kv_next: u32 = 0;
        for (self.layers, 0..) |*layer, i| {
            // The ABSOLUTE model index: every tensor name and every
            // `isFullAttention` question takes it, and only `kv_slot` rebases.
            const li: u32 = first + @as(u32, @intCast(i));
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

                // The compaction invariant, producer side: the walking counter
                // and the closed form must name one index, and that index must
                // sit inside the KV height this runner allocated.
                std.debug.assert(kv_next == kvSlotInRange(g, first, li));
                std.debug.assert(kv_next < self.n_full);
                w.kv_slot = kv_next;
                kv_next += 1;
                // Every KV binding starts at a region base, which the store
                // packs at whole regions. A storage-buffer descriptor offset
                // must be a multiple of the device offset alignment, so a
                // region must be a whole number of alignment units.
                std.debug.assert(w.kv_slot < self.kv.layout.n_layers);
                std.debug.assert(self.kv.layout.regionBytes() % self.kv.lim.offset_alignment == 0);
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
                w.set_attention = try self.mkSet(
                    self.k_attn.pipeline.set_layout,
                    &.{
                        bi(self.buf_q),
                        self.kv.slotView(w.kv_slot, .k, 0),
                        self.kv.slotView(w.kv_slot, .v, 0),
                        bi(self.buf_attn_out),
                    },
                );
                // Multi-slot attention: bind the whole K region and the whole
                // V region of this layer. The shader strides by slot.
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

                // Slot-strided recurrent state: slot s's regions start at
                // s * stateElemsPerSlot() / s * convElemsPerSlot(). The
                // single-stream paths use slot 0 (offset zero).
                w.state = try devBuf(
                    ctx,
                    @as(
                        vk.DeviceSize,
                        self.n_slots,
                    ) * self.n_v_heads * self.gdn_head_dim * self.gdn_head_dim * F,
                );
                errdefer w.state.deinit();

                w.conv_state = try devBuf(
                    ctx,
                    @as(vk.DeviceSize, self.n_slots) * (self.conv_kernel - 1) * self.conv_dim * F,
                );
                errdefer w.conv_state.deinit();

                try zeroDevBuf(ctx, &w.state);
                try zeroDevBuf(ctx, &w.conv_state);
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
                // Chunked prefill: conv front (6) and scan (11). It shares the
                // persistent `w.state` and `w.conv_state` with the decode
                // `set_gdn`, so a chunked prefill leaves both in the state the
                // decode path expects.
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

        // The walk filled the KV height the store was sized for, so no layer
        // reads rows another layer owns and no region sits unused.
        std.debug.assert(kv_next == self.n_full);
        std.debug.assert(kv_next == self.kv.layout.n_layers);

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

    /// The two buffers a KV commit copy reads. At f32 they are the
    /// rope scratch itself, and the whole path is the shipped one. At a
    /// narrower dtype they are the narrowed scratch `recordKvCommit` just
    /// wrote. Either way one row is `kvRowBytes()`, so every offset and every
    /// region size in the copy sites holds unchanged.
    fn kvCommitSrc(self: *const QwenFusedRunner) struct { k: vk.Buffer, v: vk.Buffer } {
        if (self.kv_dtype == .f32) return .{ .k = self.buf_k.buffer, .v = self.buf_v.buffer };
        return .{ .k = self.buf_k_kv.buffer, .v = self.buf_v_kv.buffer };
    }

    /// Record the narrowing of one layer's `rows` committed rows, plus the
    /// barrier that hands the result to the copy that follows. One dispatch
    /// covers the whole commit whatever the row count, because the scratch is
    /// contiguous. At f32 no dispatch runs and the caller's copy reads the
    /// rope scratch directly. Both dtypes need the barrier, because the copy
    /// reads in the transfer stage what a shader just wrote.
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

    /// The attention kit over the attention pipelines of the runner.
    fn attnKit(self: *QwenFusedRunner) kernels.AttnKit {
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
    // Each entry shape below comes in two forms. The token form embeds and
    // delegates. The hidden form takes activation rows the caller already
    // computed, which is how `qwen_hybrid.zig` hands over the residual stream
    // of its host layer prefix. Both forms fill the one staging buffer and then
    // share one record body, so a suffix runner records the same dispatches
    // whichever way its rows arrived.
    pub fn step(self: *QwenFusedRunner, token_id: u32, pos: u32) ![]f32 {
        // Defense in depth: the lone caller (`generate.decode`) guards this. The
        // public single-row entry point still checks its own precondition, the
        // way the batched `validateRows` and the CPU runner do.
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

    /// Max rows one `stepBatch` accepts. Scratch / matmul buffers are sized for
    /// `prefill_batch_max`, so a composed tick (decode rows plus a wide prefill
    /// chunk) can use up to this many rows. That is past `n_slots` and
    /// `logit_rows_max`, which bound only the slot index and the logits suffix
    /// the lm_head runs for.
    pub fn maxStepRows(self: *const QwenFusedRunner) u32 {
        return self.prefill_batch_max;
    }

    /// Rows of the compact logits buffers. `Sessions` derives the vocab stride
    /// from this, so it must state the allocated row count exactly.
    pub fn logitRowsMax(self: *const QwenFusedRunner) u32 {
        return self.logit_rows_max;
    }

    /// Advance a batch of rows, each a (slot, token, pos) over an independent
    /// KV and GDN-state slot. This is the GPU sibling of `QwenRunner.stepBatch`
    /// under the same row rules. Projections run batched. The GDN recurrence
    /// runs one dispatch per same-slot run (the in-shader sequential loop, so
    /// batch order is time order and runs touch disjoint state). Rope and
    /// attention dispatch per row at the row's (slot, pos). The lm_head runs
    /// for the `want_logits` SUFFIX only, so a prefill chunk's prompt rows skip
    /// the model's costliest matmul. Those logits land COMPACTLY at
    /// `logits[0..]` in suffix order, the layout `Sessions.advance` reads back.
    pub fn stepBatch(self: *QwenFusedRunner, rows: []const StepRow) !void {
        // The shared row rules. The batch width can exceed the slot count,
        // because a wide prefill chunk is one slot and many rows. `n_slots`
        // bounds the slot indices, and `logit_rows_max` bounds the suffix the
        // lm_head runs for. Compact logits land at `buf_logits[0..]` and
        // `logits[0..]`, both sized for the decode width plus the composed
        // tick's one final-prefill row.
        // `n_logits` can be 0: an opening system-prompt chunk, or a mid-prompt
        // chunk with no decode row, wants no logits. The record path then skips
        // the lm_head tail and the readback stays empty.
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

    /// Batch size at/above which the tiled GEMM beats the single-wave batch
    /// kernel: the GEMM computes a full 64-row output tile no matter what, so
    /// it only amortizes its weight reads once `b` fills that tile height.
    const gemm_min_batch: u32 = 64;

    /// All three Q8_0 matmul kernels share the 4-storage-buffer descriptor
    /// layout, so the pre-bound sets work with any of them. The row count `b`
    /// alone picks the kernel:
    ///   b == 1   -> Reduce mat-vec: each weight row read once, no wasted
    ///              lanes. It is the single-stream decode kernel, and it runs
    ///              memory-bound on the 9B (matmul 18.7 ms against the
    ///              9.5 GB over 512 GB/s floor). The 16-row tile of
    ///              LinearQ80Batch makes one decode row about 8x slower.
    ///   b <  64  -> single-wave batch: streams the weight once and computes b
    ///              columns. The multi-row decode and continuous-prefill path.
    ///   b >= 64  -> tiled GEMM: the win at the b=128 standalone prefill
    ///              (9B: 564 ms down to 379 ms).
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

    /// Batched prefill of `tokens` starting at `start_pos` (one CB, one submit).
    /// Turns per-token mat-vec into mat-mat (amortizes weight reads across the
    /// batch) for the TTFT win. The GDN layers run the chunked-parallel conv
    /// and scan, which the self-test checks at a tolerance. QK-norm stays
    /// per-token because it reads row offsets. Everything weight-bound is
    /// batched. Returns the LAST token's logits.
    /// tokens.len <= prefill_batch_max.
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

    fn recordPrefillStep(self: *QwenFusedRunner, b: u32, start_pos: u32) !void {
        const vkd = self.ctx.vkd;
        const cb = self.cmd_buffer;
        try vkd.resetCommandBuffer(cb, .{});
        try vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = true } });
        const F: vk.DeviceSize = @sizeOf(f32);
        const kv_row: vk.DeviceSize = self.kvRowBytes();
        const n_full = self.n_full;
        // Every full-attention layer records the same plan. The prefill rows
        // of a chunk sit in slot 0.
        std.debug.assert(b <= attn_split.max_plan_rows);
        const prefill_plan = attn_split.planPrefill(0, start_pos, b, self.attn_scratch.max_splits);
        const prefill_shape = self.attnShape(self.kv.layout.streamStrideElems());

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

            self.k_rmsnorm_batch.recordInto(cb, layer.set_attn_norm, b, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);
            switch (layer.kind) {
                .gdn => |*w| {
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
                    // Chunked-parallel GDN: conv and L2-norm per key head,
                    // then the matrix-form delta-rule scan per block of state
                    // columns, then the gated norm per value head and token.
                    // `gdn_scan` books the scan and the norm. The chain
                    // replaces the serial-over-tokens recurrence, and
                    // `gpu-selftest gdn` checks it against the CPU step at a
                    // tolerance, because the scan reorders the sums. The chunk
                    // kernels share `w.state` and `w.conv_state` with the
                    // decode path, which keeps the sequential `k_gdn`.
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
                    // The rebased KV address, prefill path: the index inside the
                    // range's height, and the B rows inside slot 0's rows of
                    // each region.
                    std.debug.assert(w.kv_slot < n_full);
                    std.debug.assert(@as(u64, start_pos) + b <= self.kv.layout.window);
                    const k_row = self.kv.rowAt(w.kv_slot, .k, 0, start_pos);
                    const v_row = self.kv.rowAt(w.kv_slot, .v, 0, start_pos);
                    // The narrowing dispatch (a no-op at f32) plus the barrier:
                    // the copy reads in the transfer stage what a shader just
                    // wrote, and the barrier above covers the compute stage only.
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
                    // The resident prefix of the chunk runs the batch
                    // kernel on the slot-0 set. A row past one split reads
                    // slot 0 through the whole-region set, as a decode row
                    // does, so it gets the bytes of that decode row.
                    attn_record.recordPrefill(kernels.AttnKit, self.attnKit(), cb, &prefill_plan, .{
                        .shape = prefill_shape,
                        .resident = .{ .batch = w.set_attention },
                        .split = w.set_attention_split,
                        .merge = self.set_attention_merge,
                    }, start_pos);
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .attn);
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
        const vkd = self.ctx.vkd;
        const cb = self.cmd_buffer;
        try vkd.resetCommandBuffer(cb, .{});
        try vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = true } });
        self.prof.begin(cb);
        const F: vk.DeviceSize = @sizeOf(f32);
        const n: u32 = @intCast(rows.len);
        const kv_row: vk.DeviceSize = self.kvRowBytes();
        const n_full = self.n_full;
        // In ELEMENTS, not bytes: the multi-stream shader indexes elements and
        // its f16 load macro owns the word addressing, so the stride does not
        // take `kvRowBytes`. Each multi-stream descriptor binds one whole
        // region, and the shader reads `slot * stride` into it.
        const stream_stride_elems: u32 = self.kv.layout.streamStrideElems();
        // The push constant and the copy offsets state one slot stride in two
        // units, so a layout that moved only one of them fails here.
        std.debug.assert(
            self.kv_dtype.rowBytes(stream_stride_elems) == self.kv.layout.slotRegionBytes(),
        );
        // Every full-attention layer records the same plan.
        std.debug.assert(n <= self.prefill_batch_max);
        const attn_plan = attn_record.planStep(rows, self.attn_scratch.max_splits);
        const attn_shape = self.attnShape(stream_stride_elems);

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

            self.k_rmsnorm_batch.recordInto(cb, layer.set_attn_norm, n, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);
            switch (layer.kind) {
                .gdn => |*w| {
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
                    // Rows of one slot are adjacent and ascending, so batch
                    // order is time order. Runs touch disjoint state regions
                    // and disjoint rows, so no barrier sits between them.
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
                    // Every row's K/V lands before any row attends. Row i then
                    // reads only its slot's 0..=pos_i (causality by layout).
                    // The narrowing dispatch (a no-op at f32) plus the barrier:
                    // the copies read in the transfer stage what a shader just
                    // wrote, and the barrier above covers the compute stage only.
                    self.recordKvCommit(cb, n);
                    const src = self.kvCommitSrc();
                    // The rebased KV address, batch path: the index inside the
                    // range's height, and each row's write inside its slot's
                    // rows of each region.
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
                    attn_record.record(kernels.AttnKit, self.attnKit(), cb, &attn_plan, .{
                        .shape = attn_shape,
                        .resident = .{ .multi = w.set_attention_multi },
                        .split = w.set_attention_split,
                        .merge = self.set_attention_merge,
                    });
                    barrierShader(vkd, cb);
                    self.prof.mark(cb, .attn);
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

        // A chunk with no want-logits row (see `stepBatch`) ends here, because
        // the KV and state writes above are the whole product. A zero-size
        // lm_head or copy must not reach the command buffer. A zero-size
        // `cmdCopyBuffer` is not valid Vulkan.
        const first = step_row.wantLogitsFirst(rows);
        if (first == n) {
            self.prof.end(cb);
            try vkd.endCommandBuffer(cb);
            return;
        }

        // Final norm over all rows (elementwise, low cost), then lm_head for the
        // want_logits SUFFIX only. The suffix is compacted to buf_x_norm[0..], so
        // the offset-0 lm_head descriptor sees it. The copy bounces through
        // `buf_x` (free after the norm) to avoid the in-place overlap
        // `cmdCopyBuffer` forbids. Logits land compactly at `buf_logits[0..]`,
        // the layout `Sessions.advance` reads.
        self.k_rmsnorm_batch.recordInto(cb, self.set_output_norm, n, self.hidden, self.rms_eps);

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

    /// Run ONE prefill chunk with GPU timestamps and print a time breakdown by
    /// `gpu_profile.Phase` (matmul, gdn, qknorm, attn, and the rest). It is the
    /// diagnostic behind `--profile-prefill`. It mutates the GDN state, the
    /// conv ring, and the KV like a real prefill, so the caller must treat the
    /// runner as dirty afterwards.
    pub fn profilePrefill(
        self: *QwenFusedRunner,
        io: std.Io,
        tokens: []const u32,
        start_pos: u32,
        tag: []const u8,
    ) !void {
        const b: u32 = @intCast(tokens.len);
        std.debug.assert(b >= 1 and b <= self.prefill_batch_max);
        const level = self.prof.level;
        // The pool grows on the first probe, so level off creates none. No
        // submit of this buffer is pending here.
        try self.prof.reserve(.phases);
        self.prof.setLevel(.phases);
        defer self.prof.setLevel(level);

        const x_host = self.stage_in.sliceTyped(f32, @as(usize, b) * self.hidden);
        for (tokens, 0..) |tid, i| {
            naive.embedLookup(
                self.embed.*,
                tid,
                self.hidden,
                x_host[i * self.hidden ..][0..self.hidden],
            );
        }

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

    fn recordOpts(
        self: *QwenFusedRunner,
        cb: vk.CommandBuffer,
        prof: *gpu_profile.GpuProfile,
        pos: u32,
        opts: gpu_profile.RecordOpts,
    ) !void {
        const vkd = self.ctx.vkd;
        try vkd.resetCommandBuffer(cb, .{});
        try vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = opts.one_time } });

        prof.begin(cb);

        const hidden_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.hidden) * @sizeOf(f32);
        const kv_row: vk.DeviceSize = self.kvRowBytes();
        copyHostToDevice(vkd, cb, self.stage_in.buffer, self.buf_x.buffer, hidden_bytes);
        barrierTransferToShader(vkd, cb);
        prof.mark(cb, .copy);

        const attn_plan = attn_split.planDecode(pos + 1);
        const attn_shape = self.attnShape(self.kv.layout.streamStrideElems());
        for (self.layers, 0..) |*layer, li| {
            if (comptime core_log.hot) {
                log_dispatch.debug("record layer {d}", .{li});
            }

            self.k_rmsnorm.recordInto(cb, layer.set_attn_norm, self.hidden, self.rms_eps);
            barrierShader(vkd, cb);
            prof.mark(cb, .norm);
            switch (layer.kind) {
                .gdn => |*w| {
                    self.k_linear.recordInto(cb, w.set_qkv, self.hidden, self.conv_dim);
                    self.k_linear.recordInto(cb, w.set_zgate, self.hidden, self.value_dim);
                    self.k_linear.recordInto(cb, w.set_alpha, self.hidden, self.n_v_heads);
                    self.k_linear.recordInto(cb, w.set_beta, self.hidden, self.n_v_heads);
                    barrierShader(vkd, cb);
                    prof.mark(cb, .matmul_gdn_in);
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
                    self.k_linear.recordInto(cb, w.set_ssm_out, self.value_dim, self.hidden);
                    barrierShader(vkd, cb);
                    prof.mark(cb, .matmul_out);
                },
                .full => |*w| {
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
                    // The q pass de-interleaves into the gate buffer, and
                    // the k pass norms in place.
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
                    // The rebased KV address, decode path: the index inside
                    // the range's height, and the row inside slot 0's rows of
                    // each region.
                    std.debug.assert(w.kv_slot < self.n_full);
                    std.debug.assert(pos < self.kv.layout.window);
                    const k_row = self.kv.rowAt(w.kv_slot, .k, 0, pos);
                    const v_row = self.kv.rowAt(w.kv_slot, .v, 0, pos);
                    // The narrowing dispatch (a no-op at f32) plus the barrier:
                    // the copy reads in the transfer stage what a shader just
                    // wrote, and the barrier above covers the compute stage only.
                    self.recordKvCommit(cb, 1);
                    const src = self.kvCommitSrc();
                    copyDeviceToDevice(vkd, cb, src.k, 0, k_row.buffer, k_row.offset, kv_row);
                    copyDeviceToDevice(vkd, cb, src.v, 0, v_row.buffer, v_row.offset, kv_row);
                    barrierTransferToShader(vkd, cb);
                    prof.mark(cb, .copy);
                    // A row past one split reads slot 0 through the
                    // whole-region set, so it takes the same split shader as a
                    // batch row.
                    attn_record.record(kernels.AttnKit, self.attnKit(), cb, &attn_plan, .{
                        .shape = attn_shape,
                        .resident = .{ .single = w.set_attention },
                        .split = w.set_attention_split,
                        .merge = self.set_attention_merge,
                    });
                    barrierShader(vkd, cb);
                    prof.mark(cb, .attn);
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

    /// Submit the buffer `probeRecordReusable` recorded and wait for the queue,
    /// the way a normal step does.
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

    /// Run ONE decode step with GPU timestamps and print a breakdown by
    /// `gpu_profile.Phase`, the decode counterpart of `profilePrefill`. It
    /// mutates the recurrent state and the KV like a real step, so the runner
    /// is dirty afterwards.
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

    /// Zero a slot's GDN recurrent + conv regions for a fresh conversation
    /// (KV needs no scrub: attention reads only rewritten positions).
    /// `Sessions.create` calls this on slot reuse. A swallowed failure silently
    /// corrupts the next conversation, so this panics instead.
    pub fn resetSlotState(self: *QwenFusedRunner, slot: u32) void {
        std.debug.assert(slot < self.n_slots);

        self.resetSlotStateInner(slot) catch |e|
            std.debug.panic("qwen gpu slot reset failed: {t}", .{e});
    }

    fn resetSlotStateInner(self: *QwenFusedRunner, slot: u32) !void {
        const F: vk.DeviceSize = @sizeOf(f32);
        const state_bytes = @as(vk.DeviceSize, self.stateElemsPerSlot()) * F;
        const conv_bytes = @as(vk.DeviceSize, self.convElemsPerSlot()) * F;
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

    /// Fork slot state on the device: every GDN layer's recurrent and conv
    /// regions in full, plus the source's rows in every KV region. Rows past
    /// `pos` copy stale bytes that attention never reads, which is the
    /// slot-reuse argument. `Sessions.clone` calls this. It panics rather than
    /// swallow a failure, for the reason `resetSlotState` states.
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
        const F: vk.DeviceSize = @sizeOf(f32);
        const state_bytes = @as(vk.DeviceSize, self.stateElemsPerSlot()) * F;
        const conv_bytes = @as(vk.DeviceSize, self.convElemsPerSlot()) * F;
        var kv_scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
        const kv_copies = self.kv.cloneCopies(source_slot, target_slot, &kv_scratch);
        std.debug.assert(kv_copies.len == 2 * @as(usize, self.n_full));
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
    // A snapshot is the whole forward state after `n` tokens: the full-attention
    // KV rows [0..n], which are per-position, plus every GDN layer's recurrent
    // state and conv ring, which are point-in-time and valid only at this n.
    // The blob holds storage bytes on both halves, so a restore is numerically
    // identical to a cold prefill. A warm continuation matches the cold token
    // hash exactly.

    /// Bytes a snapshot at `n` tokens occupies. Per full layer, K[0..n]+V[0..n]
    /// (each `n` rows of `kvRowBytes`). Per GDN layer, `state` and
    /// `conv_state`, which hold a fixed size and stay f32 under any dtype.
    pub fn snapshotBytes(self: *const QwenFusedRunner, n: u32) usize {
        const F = @sizeOf(f32);
        var total: usize = 0;
        for (self.layers) |*layer| {
            switch (layer.kind) {
                .full => total += 2 * @as(usize, n) * @as(usize, @intCast(self.kvRowBytes())),
                // Snapshots are the single-sequence (slot 0) shape: one slot's
                // state + conv regions, not the whole slot-strided buffers.
                .gdn => total += (@as(
                    usize,
                    self.stateElemsPerSlot(),
                ) + self.convElemsPerSlot()) * F,
            }
        }

        return total;
    }

    /// Build the (device<->host) copy list for `slot`'s state into the caller's
    /// `out` array: the first `n` KV positions, plus every GDN layer's full
    /// recurrent and conv regions. `to_host` true = save (D2H into `stage`),
    /// false = restore (H2D out of `stage`). Save and restore share this layout
    /// so the running host offsets match. The prompt cache calls it at slot 0
    /// with a prefix `n`. The residency layer's evict/restore call it at any
    /// slot with the session's committed row count. The caller names `stage`,
    /// and the direction fixes which stage is legal: every D2H copy reads back
    /// through `kv_readback_stage`, every H2D copy uploads out of `snap_stage`.
    /// Returns the copy count.
    fn slotCopies(
        self: *QwenFusedRunner,
        slot: u32,
        n: u32,
        to_host: bool,
        stage: vk.Buffer,
        out: []kernels.Copy,
    ) usize {
        // A readback resource serves as a copy destination only on the target
        // driver (see the `kv_readback_stage` field), so the stage choice is a
        // correctness fact and not a convention.
        std.debug.assert(to_host == (stage == self.kv_readback_stage.buffer));
        std.debug.assert(slot < self.kv.layout.n_slots);
        std.debug.assert(n <= self.kv.layout.window);

        const F: vk.DeviceSize = @sizeOf(f32);
        const n_full = self.n_full;
        const kv_bytes: vk.DeviceSize = @as(vk.DeviceSize, n) * self.kv.layout.rowBytes();
        const state_slot: vk.DeviceSize = @as(vk.DeviceSize, self.stateElemsPerSlot()) * F;
        const conv_slot: vk.DeviceSize = @as(vk.DeviceSize, self.convElemsPerSlot()) * F;
        var off: vk.DeviceSize = 0;
        var kv_total: u64 = 0;
        var c: usize = 0;
        for (self.layers) |*layer| {
            switch (layer.kind) {
                .full => |*w| {
                    // The rebased KV address, snapshot path: the index inside the
                    // range's height, and the first `n` rows of the slot in each
                    // region. The walk keeps the blob order: layers ascending, K
                    // rows then V rows, between the GDN slabs.
                    std.debug.assert(w.kv_slot < n_full);

                    for ([_]kv_index.Tensor{ .k, .v }) |t| {
                        const dev = self.kv.rowAt(w.kv_slot, t, slot, 0);
                        out[c] = if (to_host)
                            .{
                                .src = dev.buffer,
                                .src_offset = dev.offset,
                                .dst = stage,
                                .dst_offset = off,
                                .size = kv_bytes,
                            }
                        else
                            .{
                                .src = stage,
                                .src_offset = off,
                                .dst = dev.buffer,
                                .dst_offset = dev.offset,
                                .size = kv_bytes,
                            };
                        c += 1;
                        off += kv_bytes;
                        kv_total += kv_bytes;
                    }
                },
                .gdn => |*w| {
                    const regions = [_]struct {
                        buf: vk.Buffer,
                        dev: vk.DeviceSize,
                        sz: vk.DeviceSize,
                    }{
                        .{ .buf = w.state.buffer, .dev = @as(vk.DeviceSize, slot) * state_slot, .sz = state_slot },
                        .{ .buf = w.conv_state.buffer, .dev = @as(vk.DeviceSize, slot) * conv_slot, .sz = conv_slot },
                    };
                    for (regions) |r| {
                        out[c] = if (to_host)
                            .{
                                .src = r.buf,
                                .src_offset = r.dev,
                                .dst = stage,
                                .dst_offset = off,
                                .size = r.sz,
                            }
                        else
                            .{
                                .src = stage,
                                .src_offset = off,
                                .dst = r.buf,
                                .dst_offset = r.dev,
                                .size = r.sz,
                            };
                        c += 1;
                        off += r.sz;
                    }
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

    /// Restore from a caller-held snapshot `blob` (e.g. a prompt-cache RAM/disk
    /// entry) through `snap_stage`. Returns the resume position, which equals `n`.
    /// The blob must be a `snapshotInto(n)` of this runner's geometry. The cache
    /// fingerprint of the model guards that.
    pub fn restoreFromBytes(self: *QwenFusedRunner, blob: []const u8, n: u32) !u32 {
        std.debug.assert(n <= self.max_seq);
        std.debug.assert(blob.len == self.snapshotBytes(n));

        log.debug("restore: {d} positions, slot 0", .{n});
        try self.stageCopies(0, n, .{ .to_device = blob });
        return n;
    }

    /// Move the state of `slot` over its first `n` positions through the stage
    /// of the direction: every save reads back through `kv_readback_stage`, and
    /// every restore uploads out of `snap_stage`. The walk moves one chunk at a
    /// time, so a stage holds one chunk and never one window.
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
    // Evict a slot to a host blob, and restore it back. The serve residency
    // layer keeps active KV and GDN state in VRAM, and it spills idle sessions
    // to RAM. The prompt-cache snapshot takes slot 0 and a prefix `n`, while an
    // eviction takes ANY slot. Both take the same prefix shape: the KV rows the
    // session wrote, plus every GDN layer's whole recurrent state and conv
    // ring, which are point-in-time and never a prefix.

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

    /// Zero every GDN layer's recurrent state + conv ring so the next forward
    /// starts a fresh sequence. The full-attn KV is overwritten as it is
    /// filled, so it needs no reset. Used to start a new conversation on the
    /// persistent runner, and by the `--kv-experiment` break-even sweep.
    pub fn resetState(self: *QwenFusedRunner) !void {
        self.invalidateDecode();
        for (self.layers) |*layer| {
            switch (layer.kind) {
                .gdn => |*w| {
                    try zeroDevBuf(self.ctx, &w.state);
                    try zeroDevBuf(self.ctx, &w.conv_state);
                },
                .full => {},
            }
        }
    }

    /// Free one layer's weights and state, reverse of the construction order
    /// in `init`. Unlike `llama_fused`, this runner has no VRAM arena: every
    /// `Q80Weight` here is individually allocated, so each `deinit` is a real
    /// free and not a kept-for-symmetry no-op. Descriptor sets need no per-set
    /// free, because destroying `desc_pool` releases them all.
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

        self.output_norm.deinit();
        self.lm_head.deinit();
        ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

        self.gpa.free(self.logits);
        self.kv_readback_stage.deinit();
        self.snap_stage.deinit();
        self.stage_out.deinit();
        self.stage_in.deinit();

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

// The barrier and copy helper names state both ends of the dependency, the way
// `llama_fused.zig` names them. One name never covers two different hazards
// across the two fused runners.
/// Shader-write -> shader-read/write.
fn barrierShader(vkd: anytype, cb: vk.CommandBuffer) void {
    const mb = [_]vk.MemoryBarrier{
        .{ .src_access_mask = .{ .shader_write_bit = true }, .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true } },
    };
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
fn barrierTransferToShader(vkd: anytype, cb: vk.CommandBuffer) void {
    const mb = [_]vk.MemoryBarrier{
        .{ .src_access_mask = .{ .transfer_write_bit = true }, .dst_access_mask = .{ .shader_read_bit = true } },
    };
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
fn barrierShaderToTransfer(vkd: anytype, cb: vk.CommandBuffer) void {
    const mb = [_]vk.MemoryBarrier{
        .{ .src_access_mask = .{ .shader_write_bit = true }, .dst_access_mask = .{ .transfer_read_bit = true } },
    };
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
fn barrierTransferToTransfer(vkd: anytype, cb: vk.CommandBuffer) void {
    const mb = [_]vk.MemoryBarrier{
        .{ .src_access_mask = .{ .transfer_write_bit = true }, .dst_access_mask = .{ .transfer_read_bit = true } },
    };
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
    vkd: anytype,
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
    vkd: anytype,
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

fn copyDeviceToDevice(
    vkd: anytype,
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

// Every test below is device-free cost-model arithmetic. The hardware gates for
// this runner are the equivalence tests in `tests/`.
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
    // The planner stays monotone in cost under three facts: a fixed state equal
    // across dtypes, a token price no larger, and a token price of at least one
    // byte. The exact halving pins the token price at the designed ratio.
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
        const halves = QwenFusedRunner.fullCountInRange(g, 0, k) + QwenFusedRunner.fullCountInRange(
            g,
            k,
            24,
        );
        try std.testing.expectEqual(@as(u32, 6), halves);
    }
}

test "qwen range: kv_slot rebases on the range and not on the model" {
    const g = testCard().gdn.?;
    // The suffix [12..24) holds three full-attention layers, and they take the
    // KV indices 0, 1, and 2. A whole-model runner gives the same three layers
    // the model-global indices 3, 4, and 5.
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

    // The whole-model range reproduces the compaction the shipped runner used:
    // the full-attention layers take 0..5 in model order.
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
                // The KV half is what the regions of the range hold for one
                // slot. The fixed half is what the GDN layers hold at any
                // position.
                const n_full = QwenFusedRunner.fullCountInRange(card.gdn.?, r[0], r[1]);
                const layer_bytes: u64 = @as(u64, window) * kv_dtype.rowBytes(
                    card.n_kv_heads * card.head_dim,
                );
                try std.testing.expectEqual(@as(u64, 2) * n_full * layer_bytes, per_slot.kv);
                // The regions of the range hold exactly the priced KV half, and
                // a blob of n rows holds the KV half of the n-row price. A
                // range without full attention holds no regions.
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
    // The tied head charges the embedding table, which init uploads as the
    // lm_head. The whole-model figure is every layer plus that head, which is
    // every tensor the model carries.
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

    // Untied: `output.weight` is the head, and the embedding table stays host
    // side for the lookup, so it charges nothing. That is the subtraction the
    // whole-model form makes by hand.
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
    const pair: u64 = @as(
        u64,
        QwenFusedRunner.prefill_rows_max,
    ) * 2 * dtype.KvDtype.f16.rowBytes(per_step);
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
        const gained = QwenFusedRunner.scratchVramBytes(
            deep,
            kv_dtype,
        ) - QwenFusedRunner.scratchVramBytes(
            shallow,
            kv_dtype,
        );
        try std.testing.expectEqual(records_deep - records_shallow, gained);
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
