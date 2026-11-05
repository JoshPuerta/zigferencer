//! `LlamaFusedRunner` records the entire forward pass for one token into a single
//! Vulkan command buffer and submits it once. It owns every piece of
//! device-local state that inference needs. That state is the scratch buffers
//! (x, x_norm, q, k/v scratch, attn_out, the three FFN buffers, logits) and the
//! KV cache. The KV cache is a `kernels.KvStore`: one region per layer and
//! tensor, [K/V][n_layers] regions of [slot][max_seq][n_kv_heads*head_dim]. At
//! one slot the bytes are the CPU `KvCache` layout, and the single-sequence
//! paths use slot 0. It also owns all model weights (Q8_0 plus f32 norms),
//! uploaded once at init, and the per-call descriptor sets, bound once at init.
//!
//! The file exists to remove submit cost. The per-op path
//! (`ops_vulkan/wrap.zig`) submits and waits once per operation. On Llama-8B
//! that is ~450 ops/token at ~0.27 ms of host round-trip each, so ~120 ms/token
//! of pure synchronization runs before any math does. One submit per token,
//! instead of one per op, removes that cost:
//!
//!   Per-op path (wrap.zig)            Fused path (this file)
//!     for each of ~450 ops/token:       record all ops into one CB:
//!       record 1 dispatch                 [ dispatch | barrier | dispatch | ... ]
//!       submit + waitIdle (~0.27 ms)      submit + fence wait (once per token)
//!     = ~450 host round-trips/token     = 1 host round-trip/token
//!
//! Three choices keep the cost of single-command-buffer recording low. The
//! runner pre-binds descriptor sets and never rebinds them. Init points each
//! call site's set at its (input, weight, output) buffers once, and the command
//! buffer then binds the right set per call, because Vulkan sets are persistent.
//! KV cache writes use `cmdCopyBuffer`, a transfer command, rather than a
//! shader. The copy is small (n_kv_heads*head_dim floats per layer-step), and a
//! shader adds nothing but the same memcpy. A memory barrier sits between every
//! dependent pair of commands. Two dependent dispatches take a
//! compute-to-compute barrier, and a dispatch that feeds a copy takes a
//! compute-to-transfer one. One token records ~450 dispatches, ~64 copies, and
//! ~500 barriers, all GPU-side with no host synchronization until the final
//! fence wait.

const std = @import("std");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");
const vk = @import("vulkan");

const vulkan_ctx = @import("vulkan_ctx.zig");
const kernels = @import("../ops_vulkan/kernels.zig");
const naive = @import("../ops/naive.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const dtype = @import("../core/dtype.zig");
const loader = @import("../io/gguf/loader.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_cost = @import("kv_cost.zig");
const kv_index = @import("../core/kv_index.zig");
const attn_split = @import("../core/attn_split.zig");
const attn_record = @import("attn_record.zig");
const step_row = @import("step_row.zig");
const linear_dispatch = @import("linear_dispatch.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const gpu_profile = @import("gpu_profile.zig");
const core_log = @import("../core/log.zig");
const span = @import("../core/span.zig");
const step_stats = @import("../core/step_stats.zig");
const decode_ring = @import("decode_ring.zig");
const time = @import("../core/time.zig");

// This runner consumes the optional lookup directly: a missing `output.weight`
// ties to `embed`, and a missing `rope_freqs` is null.
const findTensor = tensor_lookup.find;
const findByFmt = tensor_lookup.findByFmt;

const log = std.log.scoped(.runner);
const log_kv = std.log.scoped(.kv);
const log_dispatch = std.log.scoped(.dispatch);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/llama_fused.zig was included in a non-vulkan build.");
    }
}

// -- types ------------------------------------------------------------------

/// All the per-layer and final weight tensor handles that the fused runner
/// needs at init time. The field names mirror `runtime.LlamaRunner`, so a caller
/// passes `LlamaRunner.attn_q_w[layer]` and its siblings straight through.
pub const WeightHandles = struct {
    attn_norm_w: []const *const Tensor, // per layer, f32
    attn_q_w: []const *const Tensor, // per layer, Q8_0
    attn_k_w: []const *const Tensor,
    attn_v_w: []const *const Tensor,
    attn_o_w: []const *const Tensor,
    ffn_norm_w: []const *const Tensor, // per layer, f32
    ffn_gate_w: []const *const Tensor, // per layer, Q8_0
    ffn_up_w: []const *const Tensor,
    ffn_down_w: []const *const Tensor,
    output_norm: *const Tensor, // f32
    output: *const Tensor, // Q8_0, tied to the embedding in some models
    embed: *const Tensor, // f32 or Q8_0
    /// `rope_freqs.weight` (Llama-3.1 per-pair freq divisors). It is null when
    /// the model has none, and rope then uses an all-ones factor buffer.
    rope_freqs: ?*const Tensor = null,
};

/// Quantization of a model's matmul weights. Init detects it once. The whole
/// model uses one, because the per-layer matmuls are uniform. It selects the
/// matmul kernel and the weight-upload path.
pub const WeightQuant = enum { q8_0, q4_0 };

pub const LlamaFusedRunner = struct {
    gpa: Allocator,
    ctx: *const vulkan_ctx.Context,

    // Model dimensions
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    hidden_dim: u32,
    ffn_dim: u32,
    vocab_size: u32,
    max_seq: u32,
    rms_norm_eps: f32,
    rope_dim: u32,
    rope_freq_base: f32,
    per_step_elems: u32, // n_kv_heads * head_dim
    /// The element type every KV row is framed at. `kvRowBytes` sizes the
    /// commit scratch, and `kv.layout` places every KV row.
    kv_dtype: dtype.KvDtype,

    // Embed tensor stays CPU-side for the lookup step (one row per token).
    embed: *const Tensor,

    // Each kernel owns its pipeline and set layout. The runner holds two sets:
    // decode kernels (B=1, memory-bandwidth-bound single-token decode) and
    // batched kernels (B>1, prefill, where compute dominates).
    //
    // `quant` comes from the first attention weight at init. It picks the Q8_0
    // or the Q4_0 kernel at every record site.
    quant: WeightQuant,
    k_linear: kernels.LinearQ80Reduce,
    // Q4_0 decode matmul, used instead of k_linear when quant == .q4_0.
    k_linear_q4: kernels.LinearQ40Reduce,
    k_rmsnorm: kernels.Rmsnorm,
    k_swiglu: kernels.Swiglu,
    k_residual: kernels.ResidualAdd,
    k_attention: kernels.Attention,
    k_rope_multi: kernels.RopeMulti,
    /// The commit narrowing kernel, present only at a non-f32 `kv_dtype`. One
    /// dispatch per layer per entry point narrows that layer's whole K and V
    /// commit scratch, whatever the row count.
    k_kv_commit: ?kernels.KvCommitF16,
    /// The one descriptor set the narrowing dispatch binds: the two f32 scratch
    /// buffers as sources, the two narrowed ones as destinations. The buffers
    /// never move, so one set serves every layer and every entry point.
    set_kv_commit: vk.DescriptorSet,
    // The batched kernels share the scratch buffers and the descriptor sets.
    // Their pipelines read those buffers as (B, ...) instead of (1, ...).
    // Swiglu and residual_add are elementwise, so the batched paths reuse the
    // decode kernels with `n = B * per_row`.
    k_linear_batch: kernels.LinearQ80Batch,
    // Q4_0 prefill matmul, used instead of k_linear_batch when quant == .q4_0.
    k_linear_batch_q4: kernels.LinearQ40Batch,
    k_rmsnorm_batch: kernels.RmsnormBatch,
    k_rope_multi_batch: kernels.RopeMultiBatch,
    k_attention_batch: kernels.AttentionBatch,

    // Scratch device buffers (per-token state)
    buf_x: kernels.DeviceBuffer,
    buf_x_norm: kernels.DeviceBuffer,
    buf_q: kernels.DeviceBuffer,
    buf_k_scratch: kernels.DeviceBuffer,
    /// RoPE per-pair freq divisors (rope_dim/2 f32), from `rope_freqs.weight`
    /// or all-ones. They are bound at binding 1 of the rope sets.
    buf_rope_ff: kernels.HostBuffer,
    buf_v_scratch: kernels.DeviceBuffer,
    /// The narrowed commit scratch, allocated only at a non-f32 `kv_dtype`.
    /// `KvCommitF16` reads the two f32 scratch buffers above and writes these,
    /// and the commit copy list then scatters halved regions out of them. They
    /// stay zero-capacity at f32, where no dispatch runs.
    buf_k_scratch_kv: kernels.DeviceBuffer,
    buf_v_scratch_kv: kernels.DeviceBuffer,
    buf_attn_out: kernels.DeviceBuffer,
    buf_ffn_gate: kernels.DeviceBuffer,
    buf_ffn_up: kernels.DeviceBuffer,
    buf_ffn_down: kernels.DeviceBuffer,
    buf_logits: kernels.DeviceBuffer,

    // Weight storage arenas: they consolidate ~448 small VkDeviceMemory
    // allocations (one per scales or quants buffer) into ~10 large segments.
    // The driver's per-allocation overhead and commit limits ran the device
    // out of memory before the 16 GB card was full.
    //   - arena_small holds f16 scales (~250 MB on Llama-8B) AND f32 norm
    //     weights (~1 MB). Same usage flags, same lifetime, so one arena
    //     serves both.
    //   - arena_quants holds i8 quants only (~7.6 GB on Llama-8B). It stays
    //     separate because quants dominate VRAM, and a dedicated arena keeps
    //     the budget readable: VRAM headroom for the KV cache = card -
    //     arena_quants - arena_small - scratch - driver overhead.
    arena_small: kernels.VramArena,
    arena_quants: kernels.VramArena,

    // Weight views into the arenas. Each Q80Weight is a (scales, quants) pair
    // of buffer-offset-range views. The norm views sit beside them because
    // they share arena_small.
    attn_norm_w: []vk.DescriptorBufferInfo,
    ffn_norm_w: []vk.DescriptorBufferInfo,
    output_norm_w: vk.DescriptorBufferInfo,
    attn_q_w: []kernels.Q80Weight,
    attn_k_w: []kernels.Q80Weight,
    attn_v_w: []kernels.Q80Weight,
    attn_o_w: []kernels.Q80Weight,
    ffn_gate_w: []kernels.Q80Weight,
    ffn_up_w: []kernels.Q80Weight,
    ffn_down_w: []kernels.Q80Weight,
    output_w: kernels.Q80Weight,

    // Host staging (for token-in / logits-out round trips)
    stage_in: kernels.HostBuffer,
    stage_out: kernels.HostBuffer,
    // The GPU writes stage_out into host-coherent memory across the bus, which
    // the CPU reads slowly (one cache snoop per line). One stream copy per step
    // lands the logits here, in ordinary cached heap memory, which
    // `sampler.greedy` then reads at full speed.
    logits_heap: []f32,

    // Descriptor pool + per-call-site descriptor sets (all pre-bound at init)
    desc_pool: vk.DescriptorPool,
    set_rmsnorm_attn: []vk.DescriptorSet, // [n_layers]
    set_rmsnorm_ffn: []vk.DescriptorSet, // [n_layers]
    set_rmsnorm_final: vk.DescriptorSet,
    set_q: []vk.DescriptorSet, // [n_layers]
    set_k: []vk.DescriptorSet,
    set_v: []vk.DescriptorSet,
    set_o: []vk.DescriptorSet,
    set_gate: []vk.DescriptorSet,
    set_up: []vk.DescriptorSet,
    set_down: []vk.DescriptorSet,
    set_lm_head: vk.DescriptorSet,
    set_swiglu: vk.DescriptorSet,
    set_residual_attn: vk.DescriptorSet, // (x_norm, x)
    set_residual_ffn: vk.DescriptorSet, // (ffn_down, x)
    set_attention: []vk.DescriptorSet, // [n_layers]
    set_rope_q: vk.DescriptorSet,
    set_rope_k: vk.DescriptorSet,

    /// Maximum batch dimension for one prefill dispatch. Scratch
    /// buffers and descriptor bindings are sized to this value.
    /// Callers must chunk longer prompts.
    prefill_batch_max: u32,
    /// Decode width: the most rows one tick advances. It is at most `n_slots`,
    /// because a decode row owns a slot, and at most `limits.max_batch`.
    n_streams_max: u32,
    /// The session-pool boundary (`runtime/session.Sessions`) reads slot count,
    /// `max_seq`, and `logits` by these names on every runner. Physical KV
    /// slots: the outer dimension of each KV region. Decoupled from the decode
    /// width, so a KV budget buys breadth past `max_batch`.
    n_slots: u32,
    /// The KV cache: one region per layer and tensor, each holding
    /// `[slot][max_seq][per_step_elems]` of `kv_dtype`. `kv.layout` computes
    /// every host offset and stride. The single-stream paths (`step` and
    /// `prefillBatch`) address slot 0.
    kv: kernels.KvStore,
    /// The most logits rows one call produces: the decode rows plus the final
    /// prefill row of a composed tick, capped by the slot count. Sizes
    /// `buf_logits`, `stage_out`, and `logits`.
    logit_rows_max: u32,
    /// Multi-stream attention kernel, with one descriptor set per layer. A set
    /// binds the K region and the V region of that layer, which cover ALL
    /// slots. The shader indexes per slot with `stream_stride_elems` inside the
    /// bound range.
    k_attention_multi: kernels.AttentionMultiStream,
    set_attention_multi: []vk.DescriptorSet, // [n_layers]
    /// The split attention passes. A decode row past one split walks its
    /// depth in splits and merges the partial records in split order.
    k_attention_split: kernels.AttentionSplit,
    /// The record box of the split rows in flight, sized at the declared
    /// context, so every row the window admits fits it.
    attn_scratch: attn_split.Scratch,
    buf_attn_split: kernels.DeviceBuffer,
    /// One split set per layer binds q, the whole K region, the whole V
    /// region, and the records, as the multi-stream sets do.
    set_attention_split: []vk.DescriptorSet, // [n_layers]
    /// The one merge set: the records and `buf_attn_out`. Both buffers never
    /// move, so one set serves every layer.
    set_attention_merge: vk.DescriptorSet,
    /// Heap-side per-row logits buffer (vocab_size * logit_rows_max).
    /// stepBatch writes here after streaming the GPU output
    /// through host-coherent stage_out memory.
    logits: []f32,

    /// The command buffer of every submit except a decode step: prefill,
    /// `stepBatch`, the profiler, and the bench probes.
    cmd_buffer: vk.CommandBuffer,
    /// The decode steps of `step`, over two buffers of their own. Every other
    /// submit invalidates it first.
    decode: decode_ring.DecodeRing,
    /// True after init binds the last descriptor set. A recording in the ring
    /// stays valid only while no set changes, so no set changes after init.
    sets_fixed: bool = false,

    // Host-to-device staging for the KV restore path. The first restore
    // allocates it, so a run that never restores pays nothing.
    kv_swap_stage: kernels.HostBuffer,

    /// Separate staging for the device-to-host copy. Both stages use host-cached
    /// memory when the driver offers it. Two stages and not one, because an
    /// eviction readback must not wait for an upload that still uses
    /// `kv_swap_stage`.
    kv_readback_stage: kernels.HostBuffer,

    /// Telemetry only: the DMA and copy split of the last evict or restore, so a bench
    /// aggregates the DMA and the mapped-staging memcpy without parsing logs.
    last_swap_split: kernels.SwapSplit,

    /// The chunk one staged copy moves. The planner prices
    /// `limits.swap_stage_chunk_bytes`. Only the `bench-swap` chunk sweep sets
    /// another value, and a value past the blob moves the blob in one copy.
    /// The hybrid runners hold no such field: they pass
    /// `limits.swap_stage_chunk_bytes` to `kernels.walkStage` directly.
    swap_chunk_bytes: u64 = limits.swap_stage_chunk_bytes,

    /// The timestamps of `cmd_buffer`, at the process level. The profile
    /// commands raise it to `phases` for one submit.
    prof: gpu_profile.GpuProfile,

    /// Host time of the single-row decode host steps, summed since the last reset.
    /// `generate.decode` clears it before a run and reads it after.
    host: step_stats.DecodeHost = .{},

    /// True while `cmd_buffer` holds a decode recording that `probeSubmitWait`
    /// can resubmit. Only the `bench-decode-host` hooks set it.
    probe_recorded: bool = false,

    /// The `run` command batches the prompt through `prefillBatch`, as serve
    /// and bench-model do.
    pub const chunked_prefill: bool = true;

    // -- weight discovery ---------------------------------------------------

    /// What a lookup of the weights of a loaded model returns.
    pub const FindError = Allocator.Error || error{ MissingTensor, UnsupportedWeightDtype };

    pub const InitError = error{
        OutOfMemory,
        NoSuitableMemoryType,
        UnsupportedWeightDtype,
        VramBudgetExceeded,
        KvRegionPastDeviceLimit,
        AttentionScratchPastDeviceLimit,
        AttentionHeadDimUnsupported,
    } || vulkan_ctx.InitError ||
        kernels.LinearQ80Reduce.InitError ||
        kernels.Q80Weight.InitError ||
        kernels.KvStore.InitError ||
        vk.DeviceWrapper.CreateDescriptorPoolError ||
        vk.DeviceWrapper.AllocateDescriptorSetsError ||
        vk.DeviceWrapper.AllocateCommandBuffersError;

    /// The weight handles a loaded model yields, plus the one allocation that
    /// backs their nine per-layer arrays. `init` reads the handles and keeps no
    /// reference, so the caller releases the backing right after `init` returns.
    pub const FoundWeights = struct {
        handles: WeightHandles,
        backing: []*const Tensor,

        pub fn deinit(self: FoundWeights, gpa: Allocator) void {
            gpa.free(self.backing);
        }
    };

    /// Discover every weight tensor by name, mirroring the lookup logic in
    /// `runtime.LlamaRunner.init`. `initFromLoaded` and `weightsVramBytesFromLoaded`
    /// share it, so the byte sum a call site plans with cannot name a different
    /// tensor set than the upload.
    pub fn handlesFromLoaded(
        gpa: Allocator,
        loaded: *const loader.Loaded,
    ) FindError!FoundWeights {
        const c = loaded.card;
        try linear_dispatch.requireWeights(loaded.tensors, c.n_layers);
        const embed = findTensor(
            loaded.tensors,
            "token_embd.weight",
        ) orelse return error.MissingTensor;
        const output_norm = findTensor(
            loaded.tensors,
            "output_norm.weight",
        ) orelse return error.MissingTensor;
        const output = findTensor(loaded.tensors, "output.weight") orelse embed;

        // One allocation carved into the nine per-layer arrays: a partial
        // failure then has nothing to unwind.
        const per_layer_arrays = 9;
        const n: usize = c.n_layers;
        const backing = try gpa.alloc(*const Tensor, per_layer_arrays * n);
        errdefer gpa.free(backing);

        const attn_norm_w = backing[0 * n ..][0..n];
        const attn_q_w = backing[1 * n ..][0..n];
        const attn_k_w = backing[2 * n ..][0..n];
        const attn_v_w = backing[3 * n ..][0..n];
        const attn_o_w = backing[4 * n ..][0..n];
        const ffn_norm_w = backing[5 * n ..][0..n];
        const ffn_gate_w = backing[6 * n ..][0..n];
        const ffn_up_w = backing[7 * n ..][0..n];
        const ffn_down_w = backing[8 * n ..][0..n];

        var name_buf: [64]u8 = undefined;
        var i: u32 = 0;
        while (i < c.n_layers) : (i += 1) {
            attn_norm_w[i] = try findByFmt(
                loaded.tensors,
                &name_buf,
                "blk.{d}.attn_norm.weight",
                .{i},
            );
            attn_q_w[i] = try findByFmt(loaded.tensors, &name_buf, "blk.{d}.attn_q.weight", .{i});
            attn_k_w[i] = try findByFmt(loaded.tensors, &name_buf, "blk.{d}.attn_k.weight", .{i});
            attn_v_w[i] = try findByFmt(loaded.tensors, &name_buf, "blk.{d}.attn_v.weight", .{i});
            attn_o_w[i] = try findByFmt(
                loaded.tensors,
                &name_buf,
                "blk.{d}.attn_output.weight",
                .{i},
            );
            ffn_norm_w[i] = try findByFmt(
                loaded.tensors,
                &name_buf,
                "blk.{d}.ffn_norm.weight",
                .{i},
            );
            ffn_gate_w[i] = try findByFmt(
                loaded.tensors,
                &name_buf,
                "blk.{d}.ffn_gate.weight",
                .{i},
            );
            ffn_up_w[i] = try findByFmt(loaded.tensors, &name_buf, "blk.{d}.ffn_up.weight", .{i});
            ffn_down_w[i] = try findByFmt(
                loaded.tensors,
                &name_buf,
                "blk.{d}.ffn_down.weight",
                .{i},
            );
        }

        return .{
            .handles = .{
                .attn_norm_w = attn_norm_w,
                .attn_q_w = attn_q_w,
                .attn_k_w = attn_k_w,
                .attn_v_w = attn_v_w,
                .attn_o_w = attn_o_w,
                .ffn_norm_w = ffn_norm_w,
                .ffn_gate_w = ffn_gate_w,
                .ffn_up_w = ffn_up_w,
                .ffn_down_w = ffn_down_w,
                .output_norm = output_norm,
                .output = output,
                .embed = embed,
                .rope_freqs = findTensor(loaded.tensors, loader.rope_freqs_tensor),
            },
            .backing = backing,
        };
    }

    // -- budgets ------------------------------------------------------------

    /// What one KV slot costs on this model. Every layer holds a KV cache, and
    /// no layer holds recurrent state, so the per-token figure is exact. The
    /// planner states the price, so the CLI reads the same figure before a
    /// runner exists.
    pub fn kvCostModel(card: loader.ModelCard, kv_dtype: dtype.KvDtype) kv_budget.KvCostModel {
        std.debug.assert(card.gdn == null);

        return kv_budget.costModel(card.kvShape(), kv_dtype);
    }

    /// Rows one prefill dispatch holds, and the batch factor of the activation
    /// scratch. The value bounds the duration of one submit. A multi-second
    /// submit destabilizes a translation-layer driver, and 512 rows reached
    /// 2.6 s on Llama-8B.
    pub const prefill_rows_max: u32 = 256;

    /// The device bytes the activation scratch occupies: the per-row forward
    /// buffers `init` allocates, `prefill_rows_max` rows each. Must match that
    /// allocation list, so the planner's reservation is the preflight's sum.
    /// A non-f32 KV dtype adds the narrowed commit scratch, one K row and one V
    /// row of the storage element beside the f32 pair the matmuls write, so the
    /// plan prices what that dtype really allocates. The split attention
    /// records add `attnSplitScratch`, which is f32 at every KV dtype.
    pub fn scratchVramBytes(card: loader.ModelCard, kv_dtype: dtype.KvDtype) u64 {
        const q_dim: u64 = @as(u64, card.n_heads) * card.head_dim;
        const per_step: u64 = @as(u64, card.n_kv_heads) * card.head_dim;
        const per_row: u64 = 3 * @as(u64, card.embedding_length) + 2 * q_dim +
            2 * per_step + 2 * @as(u64, card.feed_forward_length);
        const f32_bytes: u64 = @as(u64, prefill_rows_max) * per_row * @sizeOf(f32) +
            attnSplitScratch(card).totalBytes();
        if (kv_dtype == .f32) return f32_bytes;

        return f32_bytes + @as(u64, prefill_rows_max) * 2 * kv_dtype.rowBytes(@intCast(per_step));
    }

    /// The split attention records of this model. The box is sized at the
    /// declared context and not at the window, because the planner subtracts
    /// the scratch before it derives the window. The pricing and the
    /// allocation both read this one value.
    pub fn attnSplitScratch(card: loader.ModelCard) attn_split.Scratch {
        return attn_split.Scratch.init(card.n_heads, card.head_dim, card.context_length);
    }

    /// The device bytes the uploaded weights occupy: every matmul and norm.
    /// `token_embd` stays host side for the embedding lookup, so it is not
    /// counted, unless it doubles as the tied lm_head, which is uploaded.
    pub fn weightsVramBytes(weights: WeightHandles, n_layers: u32) u64 {
        var total: u64 = weights.output_norm.data.len + weights.output.data.len;
        for (0..n_layers) |li| {
            total +=
                weights.attn_norm_w[li].data.len + weights.ffn_norm_w[li].data.len +
                weights.attn_q_w[li].data.len + weights.attn_k_w[li].data.len +
                weights.attn_v_w[li].data.len + weights.attn_o_w[li].data.len +
                weights.ffn_gate_w[li].data.len + weights.ffn_up_w[li].data.len +
                weights.ffn_down_w[li].data.len;
        }

        return total;
    }

    /// The same sum, for a call site that plans before it holds the handles.
    pub fn weightsVramBytesFromLoaded(
        gpa: Allocator,
        loaded: *const loader.Loaded,
    ) FindError!u64 {
        const found = try handlesFromLoaded(gpa, loaded);
        defer found.deinit(gpa);

        return weightsVramBytes(found.handles, loaded.card.n_layers);
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
    /// that (the decode width plus the composed tick's final prefill row).
    /// The caller states the KV dtype, which moves the scratch reservation and
    /// the two default windows. The KV price itself lives in `kvCostModel`,
    /// which the planner takes separately. `ceiling` is the model ceiling the
    /// caller resolved through `limits.contextCeiling`.
    pub fn kvCaps(
        gpa: Allocator,
        loaded: *const loader.Loaded,
        ceiling: u32,
        vram_free: u64,
        logit_rows: u32,
        kv_dtype: dtype.KvDtype,
    ) FindError!kv_budget.Caps {
        std.debug.assert(ceiling >= 1);
        std.debug.assert(ceiling <= loaded.card.context_length);

        const windows = kv_budget.defaultWindows(kvCostModel(loaded.card, kv_dtype), ceiling);
        return .{
            .vram_free = vram_free,
            .weights = try weightsVramBytesFromLoaded(gpa, loaded),
            .logits = logitsVramBytes(loaded.card.vocab_size, logit_rows),
            .scratch = scratchVramBytes(loaded.card, kv_dtype),
            .model_ctx = ceiling,
            .serve_default = windows.serve,
            .default_ceil = windows.run,
        };
    }

    // -- init ---------------------------------------------------------------

    /// Build a LlamaFusedRunner straight from a loaded model. The caller states the
    /// KV dtype. The flag resolves one dtype per process, and the resolution
    /// gates it against the model before this runs.
    pub fn initFromLoaded(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        kv_dtype: dtype.KvDtype,
    ) (InitError || error{MissingTensor})!*LlamaFusedRunner {
        const found = try handlesFromLoaded(gpa, loaded);
        defer found.deinit(gpa);

        return init(gpa, ctx, loaded.card, found.handles, plan, kv_dtype);
    }

    /// The whole init chain funnels here, so one body sizes and frames every KV
    /// buffer. The caller states the KV dtype, because the flag resolves one
    /// dtype per process.
    pub fn init(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        card: loader.ModelCard,
        weights: WeightHandles,
        plan: kv_budget.KvPlan,
        kv_dtype: dtype.KvDtype,
    ) InitError!*LlamaFusedRunner {
        const self = try gpa.create(LlamaFusedRunner);
        errdefer gpa.destroy(self);

        const hidden_dim: u32 = @intCast(card.embedding_length);
        const ffn_dim: u32 = @intCast(card.feed_forward_length);
        const vocab_size: u32 = @intCast(card.vocab_size);
        // The window, the slot count, and the decode width arrive resolved:
        // the caller plans them from the byte budget (`core/kv_budget.zig`),
        // because the same arithmetic has to serve the plan and the preflight
        // below. Slots bound breadth (resident conversations), streams bound
        // width (rows per tick), and a row owns a slot.
        const n_slots: u32 = plan.n_slots;
        const n_streams: u32 = plan.n_streams;
        const max_seq: u32 = plan.window;
        std.debug.assert(n_slots >= 1);
        std.debug.assert(n_slots <= limits.max_vram_slots);
        std.debug.assert(n_streams >= 1);
        std.debug.assert(n_streams <= limits.max_batch);
        std.debug.assert(n_streams <= n_slots);
        // Logits rows: every decode row plus the final prefill row of a
        // composed tick wants logits, and each such row occupies its own slot.
        const logit_rows: u32 = @min(n_slots, n_streams + 1);
        const n_layers: u32 = card.n_layers;
        const n_heads: u32 = card.n_heads;
        const n_kv_heads: u32 = card.n_kv_heads;
        const head_dim: u32 = card.head_dim;
        const per_step_elems: u32 = n_kv_heads * head_dim;

        // Every layer holds one K region and one V region, and the plan priced
        // exactly those bytes. The device limits bind one region, so the guard
        // refuses before the first allocation.
        const kv_layout = kv_index.Layout.init(
            n_slots,
            max_seq,
            per_step_elems,
            n_layers,
            kv_dtype,
        );
        std.debug.assert(kv_layout.totalBytes() == plan.vram.kv);
        try ctx.guardKvLayout(kv_layout);
        try kernels.AttentionSplit.guardHeadDim(head_dim);
        const attn_scratch = attnSplitScratch(card);
        std.debug.assert(attn_split.splitCount(max_seq) <= attn_scratch.max_splits);
        try ctx.guardAttnSplitScratch(attn_scratch);

        // VRAM preflight: refuse a config that does not fit the live device-local
        // budget before the first allocation, with an actionable message. An
        // over-budget plan otherwise drives vkAllocateMemory to
        // OutOfDeviceMemory mid-init, and the driver then removes the device.
        // Weights: every uploaded matmul and norm. token_embd stays host-side
        // for the embedding lookup, so it is not counted. The lm_head `output`
        // is counted.
        {
            const mib = 1024 * 1024;
            const kv_one: u64 = kv_layout.slotBytes();
            const kv_total: u64 = kv_layout.totalBytes();
            const weights_bytes: u64 = weightsVramBytes(weights, n_layers);
            const logits_bytes: u64 = logitsVramBytes(vocab_size, logit_rows);
            const scratch_bytes: u64 = scratchVramBytes(card, kv_dtype);
            // The plan states the reservations this preflight measures. Its
            // logits figure can exceed the rows allocated here, because a
            // budget caller reserves the width before the planner resolves the
            // slot count that can cap it.
            std.debug.assert(plan.vram.weights == weights_bytes);
            std.debug.assert(plan.vram.kv == kv_total);
            std.debug.assert(plan.vram.logits >= logits_bytes);
            std.debug.assert(plan.vram.scratch == scratch_bytes);
            const planned: u64 = weights_bytes + kv_total + logits_bytes + scratch_bytes;
            const available: u64 = ctx.deviceLocalAvailableBytes();
            log.debug(
                "vram preflight: planned {d} MiB (weights {d}, kv {d}, scratch {d}), " ++
                    "available {d} MiB",
                .{ planned / mib, weights_bytes / mib, kv_total / mib, scratch_bytes / mib, available / mib },
            );
            if (!limits.vramPlanFits(planned, available)) {
                // The fit reads the same live budget and the same reservations
                // as the check above.
                const reserved: u64 = weights_bytes + logits_bytes + scratch_bytes;
                const cost = kvCostModel(card, kv_dtype);
                std.debug.assert(cost.kv_bytes_per_token * max_seq == kv_one);
                var refusal_buf: [kv_budget.vram_refusal_text_max]u8 = undefined;
                log.err("{s}", .{kv_budget.formatVramRefusal(&refusal_buf, .{
                    .planned = planned,
                    .available = available,
                    .weights = weights_bytes,
                    .kv = kv_total,
                    .n_slots = n_slots,
                    .window = max_seq,
                    .fit = kv_budget.largestFittingWindow(
                        available,
                        reserved,
                        cost,
                        n_slots,
                        max_seq,
                    ),
                })});
                return error.VramBudgetExceeded;
            }
        }

        self.* = .{
            .gpa = gpa,
            .ctx = ctx,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .hidden_dim = hidden_dim,
            .ffn_dim = ffn_dim,
            .vocab_size = vocab_size,
            .max_seq = max_seq,
            .rms_norm_eps = card.rms_norm_eps,
            .rope_dim = card.rope_dim_count,
            .rope_freq_base = card.rope_freq_base,
            .per_step_elems = per_step_elems,
            .kv_dtype = kv_dtype,
            .embed = weights.embed,
            .quant = if (weights.attn_q_w[0].dtype == .q4_0) .q4_0 else .q8_0,
            .k_linear = undefined,
            .k_linear_q4 = undefined,
            .k_rmsnorm = undefined,
            .k_swiglu = undefined,
            .k_residual = undefined,
            .k_attention = undefined,
            .k_rope_multi = undefined,
            .k_kv_commit = null,
            .set_kv_commit = .null_handle,
            .k_linear_batch = undefined,
            .k_linear_batch_q4 = undefined,
            .k_rmsnorm_batch = undefined,
            .k_rope_multi_batch = undefined,
            .k_attention_batch = undefined,
            .prefill_batch_max = prefill_rows_max,
            .n_streams_max = n_streams,
            .n_slots = n_slots,
            .logit_rows_max = logit_rows,
            .k_attention_multi = undefined,
            .set_attention_multi = undefined,
            .k_attention_split = undefined,
            .attn_scratch = attn_scratch,
            .buf_attn_split = .{ .ctx = ctx },
            .set_attention_split = undefined,
            .set_attention_merge = .null_handle,
            .logits = undefined,
            .buf_x = .{ .ctx = ctx },
            .buf_x_norm = .{ .ctx = ctx },
            .buf_q = .{ .ctx = ctx },
            .buf_rope_ff = .{ .ctx = ctx },
            .buf_k_scratch = .{ .ctx = ctx },
            .buf_v_scratch = .{ .ctx = ctx },
            .buf_k_scratch_kv = .{ .ctx = ctx },
            .buf_v_scratch_kv = .{ .ctx = ctx },
            .buf_attn_out = .{ .ctx = ctx },
            .buf_ffn_gate = .{ .ctx = ctx },
            .buf_ffn_up = .{ .ctx = ctx },
            .buf_ffn_down = .{ .ctx = ctx },
            .buf_logits = .{ .ctx = ctx },
            .kv = undefined, // set below, after the scratch buffers
            .attn_norm_w = undefined,
            .ffn_norm_w = undefined,
            .output_norm_w = undefined,
            .attn_q_w = undefined,
            .attn_k_w = undefined,
            .attn_v_w = undefined,
            .attn_o_w = undefined,
            .ffn_gate_w = undefined,
            .ffn_up_w = undefined,
            .ffn_down_w = undefined,
            .output_w = undefined,
            .arena_small = undefined,
            .arena_quants = undefined,
            .stage_in = .{ .ctx = ctx },
            .stage_out = .{ .ctx = ctx },
            .logits_heap = undefined,
            .desc_pool = .null_handle,
            .set_rmsnorm_attn = undefined,
            .set_rmsnorm_ffn = undefined,
            .set_rmsnorm_final = .null_handle,
            .set_q = undefined,
            .set_k = undefined,
            .set_v = undefined,
            .set_o = undefined,
            .set_gate = undefined,
            .set_up = undefined,
            .set_down = undefined,
            .set_lm_head = .null_handle,
            .set_swiglu = .null_handle,
            .set_residual_attn = .null_handle,
            .set_residual_ffn = .null_handle,
            .set_attention = undefined,
            .set_rope_q = .null_handle,
            .set_rope_k = .null_handle,
            .cmd_buffer = .null_handle,
            .decode = undefined, // set below, after the descriptor sets
            .kv_swap_stage = .{ .ctx = ctx },
            .kv_readback_stage = .{ .ctx = ctx },
            .last_swap_split = .{ .dma_ms = 0, .memcpy_ms = 0 },
            .prof = undefined, // set below, after the context is stable
        };

        // Per-row state x batch_max, so the same buffers serve decode
        // (B=1, uses first row) and prefill (B<=batch_max). Logits stay
        // at single row: sampling reads only the last prefill row.
        const f32_size = @sizeOf(f32);
        const bm: vk.DeviceSize = @intCast(self.prefill_batch_max);
        try self.buf_x.ensure(bm * @as(vk.DeviceSize, hidden_dim) * f32_size);
        try self.buf_x_norm.ensure(bm * @as(vk.DeviceSize, hidden_dim) * f32_size);
        try self.buf_q.ensure(bm * @as(vk.DeviceSize, n_heads * head_dim) * f32_size);
        {
            const half = self.rope_dim / 2;
            try self.buf_rope_ff.ensure(@as(vk.DeviceSize, half) * f32_size);
            const ff_dst = self.buf_rope_ff.sliceTyped(f32, half);
            if (weights.rope_freqs) |t| {
                const src: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, t.data));
                @memcpy(ff_dst, src[0..half]);
            } else {
                @memset(ff_dst, 1.0);
            }
        }

        try self.buf_k_scratch.ensure(bm * @as(vk.DeviceSize, per_step_elems) * f32_size);
        try self.buf_v_scratch.ensure(bm * @as(vk.DeviceSize, per_step_elems) * f32_size);
        // The narrowed commit scratch: the same row count at the storage
        // element. It exists only where a dispatch narrows into it, so the f32
        // path allocates nothing new.
        if (kv_dtype != .f32) {
            const kv_scratch: vk.DeviceSize = bm * @as(vk.DeviceSize, self.kvRowBytes());
            try self.buf_k_scratch_kv.ensure(kv_scratch);
            try self.buf_v_scratch_kv.ensure(kv_scratch);
        }

        try self.buf_attn_out.ensure(bm * @as(vk.DeviceSize, n_heads * head_dim) * f32_size);
        try self.buf_attn_split.ensure(attn_scratch.totalBytes());
        errdefer self.buf_attn_split.deinit();

        try self.buf_ffn_gate.ensure(bm * @as(vk.DeviceSize, ffn_dim) * f32_size);
        try self.buf_ffn_up.ensure(bm * @as(vk.DeviceSize, ffn_dim) * f32_size);
        try self.buf_ffn_down.ensure(bm * @as(vk.DeviceSize, hidden_dim) * f32_size);
        // buf_logits is sized for logit_rows_max rows. B=1 decode uses
        // the first row, a batched tick the first n_logits rows,
        // and prefill is also single-row because sampling reads only
        // the last token.
        try self.buf_logits.ensure(
            @as(vk.DeviceSize, self.logit_rows_max) * @as(vk.DeviceSize, vocab_size) * f32_size,
        );

        // KV cache, one region per layer and tensor. The commit copies move
        // scratch rows of `kvRowBytes()` into rows of the layout.
        std.debug.assert(kv_layout.rowBytes() == self.kvRowBytes());
        self.kv = try kernels.KvStore.init(ctx, kv_layout);
        errdefer self.kv.deinit();

        // Host staging. stage_in carries one token's embedding for
        // decode, or up to `prefill_batch_max` rows for prefill.
        // stage_out sized for logit_rows_max rows so batched
        // readback fits.
        try self.stage_in.ensure(bm * @as(vk.DeviceSize, hidden_dim) * f32_size);
        try self.stage_out.ensure(
            @as(vk.DeviceSize, self.logit_rows_max) * @as(vk.DeviceSize, vocab_size) * f32_size,
        );
        self.logits_heap = try gpa.alloc(f32, vocab_size);

        self.k_linear = try kernels.LinearQ80Reduce.init(ctx);
        self.k_linear_q4 = try kernels.LinearQ40Reduce.init(ctx);
        self.k_rmsnorm = try kernels.Rmsnorm.init(ctx);
        self.k_swiglu = try kernels.Swiglu.init(ctx);
        self.k_residual = try kernels.ResidualAdd.init(ctx);
        // The four attention kernels read the KV regions, so each takes the
        // storage element. At f32 the selector picks the f32 pipeline.
        self.k_attention = try kernels.Attention.initKv(ctx, kv_dtype);
        self.k_rope_multi = try kernels.RopeMulti.init(ctx);
        self.k_linear_batch = try kernels.LinearQ80Batch.init(ctx);
        self.k_linear_batch_q4 = try kernels.LinearQ40Batch.init(ctx);
        self.k_rmsnorm_batch = try kernels.RmsnormBatch.init(ctx);
        self.k_rope_multi_batch = try kernels.RopeMultiBatch.init(ctx);
        self.k_attention_batch = try kernels.AttentionBatch.initKv(ctx, kv_dtype);
        self.k_attention_multi = try kernels.AttentionMultiStream.initKv(ctx, kv_dtype);
        self.k_attention_split = try kernels.AttentionSplit.initKv(ctx, kv_dtype);
        errdefer self.k_attention_split.deinit();

        if (kv_dtype != .f32) {
            self.k_kv_commit = try kernels.KvCommitF16.init(ctx);
        }

        // The 1 GB quants segment holds the largest single weight on Llama-8B
        // Q8_0, the lm_head quants at ~525 MB. The whole model then takes ~9
        // segments instead of 448 separate allocations.
        const arena_usage: vk.BufferUsageFlags = .{
            .storage_buffer_bit = true,
            .transfer_dst_bit = true,
        };
        self.arena_small = try kernels.VramArena.init(ctx, gpa, arena_usage, 256 * 1024 * 1024);
        errdefer self.arena_small.deinit();

        self.arena_quants = try kernels.VramArena.init(ctx, gpa, arena_usage, 1024 * 1024 * 1024);
        errdefer self.arena_quants.deinit();

        self.logits = try gpa.alloc(f32, @as(usize, self.logit_rows_max) * vocab_size);

        self.attn_norm_w = try gpa.alloc(vk.DescriptorBufferInfo, n_layers);
        self.ffn_norm_w = try gpa.alloc(vk.DescriptorBufferInfo, n_layers);
        self.attn_q_w = try gpa.alloc(kernels.Q80Weight, n_layers);
        self.attn_k_w = try gpa.alloc(kernels.Q80Weight, n_layers);
        self.attn_v_w = try gpa.alloc(kernels.Q80Weight, n_layers);
        self.attn_o_w = try gpa.alloc(kernels.Q80Weight, n_layers);
        self.ffn_gate_w = try gpa.alloc(kernels.Q80Weight, n_layers);
        self.ffn_up_w = try gpa.alloc(kernels.Q80Weight, n_layers);
        self.ffn_down_w = try gpa.alloc(kernels.Q80Weight, n_layers);

        var layer: u32 = 0;
        while (layer < n_layers) : (layer += 1) {
            self.attn_norm_w[layer] = try uploadFp32NormInArena(
                ctx,
                weights.attn_norm_w[layer].*,
                &self.arena_small,
            );
            self.ffn_norm_w[layer] = try uploadFp32NormInArena(
                ctx,
                weights.ffn_norm_w[layer].*,
                &self.arena_small,
            );
            self.attn_q_w[layer] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.attn_q_w[layer].*,
            );
            self.attn_k_w[layer] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.attn_k_w[layer].*,
            );
            self.attn_v_w[layer] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.attn_v_w[layer].*,
            );
            self.attn_o_w[layer] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.attn_o_w[layer].*,
            );
            self.ffn_gate_w[layer] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.ffn_gate_w[layer].*,
            );
            self.ffn_up_w[layer] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.ffn_up_w[layer].*,
            );
            self.ffn_down_w[layer] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.ffn_down_w[layer].*,
            );
        }

        self.output_norm_w = try uploadFp32NormInArena(
            ctx,
            weights.output_norm.*,
            &self.arena_small,
        );
        self.output_w = try makeOutputW(
            ctx,
            gpa,
            &self.arena_small,
            &self.arena_quants,
            self.quant,
            weights.output.*,
        );

        // Descriptor pool. Total descriptor slots conservatively sized:
        //   matmul: n_layers*7 + 1 sets, 4 storage_buffer per set
        //   rmsnorm: n_layers*2 + 1 sets, 3 per set
        //   attention (decode/prefill): n_layers sets, 4 per set
        //   attention (multi-stream): n_layers sets, 4 per set
        //   attention (split): n_layers sets, 4 per set
        //   attention (merge): 1 set, 2 per set
        //   residual: 2 sets, 2 per set
        //   swiglu: 1 set, 3 per set
        //   rope_multi: 2 sets, 2 per set (x buffer + freq factors)
        //   kv commit: 1 set, 4 per set, at a non-f32 KV dtype only
        const kv_commit_sets: u32 = if (kv_dtype == .f32) 0 else 1;
        const split_bindings = kernels.AttentionSplit.split_bindings;
        const merge_bindings = kernels.AttentionSplit.merge_bindings;
        const total_sets: u32 = (n_layers * 7 + 1) + (n_layers * 2 + 1) + n_layers + n_layers +
            n_layers + 1 + 2 + 1 + 2 + kv_commit_sets;
        const total_storage: u32 = (n_layers * 7 + 1) * 4 + (n_layers * 2 + 1) * 3 + n_layers * 4 +
            n_layers * 4 +
            n_layers * split_bindings + merge_bindings + 2 * 2 + 3 + 2 * 2 + kv_commit_sets * 4;
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .storage_buffer, .descriptor_count = total_storage },
        };
        const pool_info: vk.DescriptorPoolCreateInfo = .{
            .max_sets = total_sets,
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = &pool_sizes,
        };
        self.desc_pool = try ctx.vkd.createDescriptorPool(ctx.device, &pool_info, null);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

        self.set_rmsnorm_attn = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_rmsnorm_ffn = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_q = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_k = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_v = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_o = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_gate = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_up = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_down = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_attention = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_attention_multi = try gpa.alloc(vk.DescriptorSet, n_layers);
        self.set_attention_split = try gpa.alloc(vk.DescriptorSet, n_layers);

        try self.allocateAndBindDescriptorSets();

        self.cmd_buffer = try kernels.allocateOnePrimaryCmdBuffer(ctx);
        errdefer ctx.vkd.freeCommandBuffers(
            ctx.device,
            ctx.command_pool,
            &[_]vk.CommandBuffer{self.cmd_buffer},
        );

        self.decode = try decode_ring.DecodeRing.init(ctx, .llama_fused);
        errdefer self.decode.deinit();

        self.sets_fixed = true;

        self.prof = try gpu_profile.GpuProfile.init(ctx, span.gpuLevel(), span.gpuLevel());
        errdefer self.prof.deinit();

        const mib: f64 = 1024.0 * 1024.0;
        const weights_mib: f64 = @as(
            f64,
            @floatFromInt(self.arena_quants.totalUsed() + self.arena_small.totalUsed()),
        ) / mib;
        const kv_mib: f64 = @as(f64, @floatFromInt(self.kv.deviceBytes())) / mib;
        log.debug(
            "init llama fused: {d} layers, {d} slots x {d} streams, max_seq {d}, quant {t}, " ++
                "weights {d:.0} MiB, kv {d:.0} MiB",
            .{ n_layers, self.n_slots, self.n_streams_max, max_seq, self.quant, weights_mib, kv_mib },
        );
        self.arena_small.logSegments("small");
        self.arena_quants.logSegments("quants");

        return self;
    }

    pub fn deinit(self: *LlamaFusedRunner) void {
        const ctx = self.ctx;
        const freed_mib: f64 = @as(
            f64,
            @floatFromInt(
                self.arena_quants.totalUsed() + self.arena_small.totalUsed() +
                    self.kv.layout.totalBytes(),
            ),
        ) / (1024.0 * 1024.0);
        log.debug("deinit: freeing {d:.0} MiB weights+kv", .{freed_mib});
        self.prof.deinit();
        self.decode.deinit();
        const cmds_to_free = [_]vk.CommandBuffer{self.cmd_buffer};
        ctx.vkd.freeCommandBuffers(ctx.device, ctx.command_pool, &cmds_to_free);
        ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

        self.gpa.free(self.set_attention_split);
        self.gpa.free(self.set_attention_multi);
        self.gpa.free(self.set_attention);
        self.gpa.free(self.set_down);
        self.gpa.free(self.set_up);
        self.gpa.free(self.set_gate);
        self.gpa.free(self.set_o);
        self.gpa.free(self.set_v);
        self.gpa.free(self.set_k);
        self.gpa.free(self.set_q);
        self.gpa.free(self.set_rmsnorm_ffn);
        self.gpa.free(self.set_rmsnorm_attn);

        // Q80Weight.deinit is a no-op for arena-backed weights (the
        // arenas own the underlying VkBuffer + VkDeviceMemory). The
        // call is still correct, and it keeps the symmetry.
        self.output_w.deinit();
        var layer = self.n_layers;
        while (layer > 0) {
            layer -= 1;
            self.ffn_down_w[layer].deinit();
            self.ffn_up_w[layer].deinit();
            self.ffn_gate_w[layer].deinit();
            self.attn_o_w[layer].deinit();
            self.attn_v_w[layer].deinit();
            self.attn_k_w[layer].deinit();
            self.attn_q_w[layer].deinit();
        }

        self.gpa.free(self.ffn_down_w);
        self.gpa.free(self.ffn_up_w);
        self.gpa.free(self.ffn_gate_w);
        self.gpa.free(self.attn_o_w);
        self.gpa.free(self.attn_v_w);
        self.gpa.free(self.attn_k_w);
        self.gpa.free(self.attn_q_w);
        self.gpa.free(self.ffn_norm_w);
        self.gpa.free(self.attn_norm_w);

        self.arena_quants.deinit();
        self.arena_small.deinit();

        if (self.k_kv_commit) |*k| {
            k.deinit();
        }

        self.k_attention_split.deinit();
        self.k_attention_multi.deinit();
        self.k_attention_batch.deinit();
        self.k_rope_multi_batch.deinit();
        self.k_rmsnorm_batch.deinit();
        self.k_linear_batch_q4.deinit();
        self.k_linear_batch.deinit();
        self.k_rope_multi.deinit();
        self.k_attention.deinit();
        self.k_residual.deinit();
        self.k_swiglu.deinit();
        self.k_rmsnorm.deinit();
        self.k_linear_q4.deinit();
        self.k_linear.deinit();

        self.gpa.free(self.logits);
        self.gpa.free(self.logits_heap);
        self.kv_swap_stage.deinit();
        self.kv_readback_stage.deinit();
        self.stage_out.deinit();
        self.stage_in.deinit();
        self.kv.deinit();
        self.buf_logits.deinit();
        self.buf_ffn_down.deinit();
        self.buf_ffn_up.deinit();
        self.buf_ffn_gate.deinit();
        self.buf_attn_split.deinit();
        self.buf_attn_out.deinit();
        self.buf_v_scratch_kv.deinit();
        self.buf_k_scratch_kv.deinit();
        self.buf_v_scratch.deinit();
        self.buf_k_scratch.deinit();
        self.buf_rope_ff.deinit();
        self.buf_q.deinit();
        self.buf_x_norm.deinit();
        self.buf_x.deinit();

        self.gpa.destroy(self);
    }

    // -- kv layout ----------------------------------------------------------

    /// Bytes one KV row occupies at `kv_dtype`. It sizes the commit scratch and
    /// every commit copy, and init asserts that it equals the row of
    /// `kv.layout`.
    fn kvRowBytes(self: *const LlamaFusedRunner) vk.DeviceSize {
        return self.kv_dtype.rowBytes(self.per_step_elems);
    }

    /// The two buffers a commit copy list reads. At f32 they are the rope
    /// scratch itself. At a narrower dtype they are the narrowed scratch that
    /// `recordKvCommit` just wrote. Either way one row is `kvRowBytes()`, so
    /// every offset and every region size in the copy lists holds unchanged.
    fn kvCommitSrc(self: *const LlamaFusedRunner) struct { k: vk.Buffer, v: vk.Buffer } {
        if (self.kv_dtype == .f32) {
            return .{ .k = self.buf_k_scratch.buffer, .v = self.buf_v_scratch.buffer };
        }

        return .{ .k = self.buf_k_scratch_kv.buffer, .v = self.buf_v_scratch_kv.buffer };
    }

    /// Record the narrowing of one layer's `rows` committed rows, plus the
    /// barrier that hands the result to the copy list that follows. One
    /// dispatch covers the whole commit whatever the row count, because the
    /// scratch is contiguous. At f32 no dispatch runs and the caller's copy
    /// list reads the rope scratch directly. Both dtypes need the barrier,
    /// because the copy reads in the transfer stage what a shader just wrote.
    fn recordKvCommit(self: *LlamaFusedRunner, cb: vk.CommandBuffer, rows: u32) void {
        std.debug.assert(rows >= 1);
        std.debug.assert(rows <= self.prefill_batch_max);

        if (self.k_kv_commit) |*k| {
            if (comptime core_log.hot) {
                log_dispatch.debug("  kv commit ({t}): rows {d}", .{ self.kv_dtype, rows });
            }

            k.recordInto(cb, self.set_kv_commit, rows * self.per_step_elems);
        }

        barrierShaderToTransfer(self.ctx.vkd, cb);
    }

    // -- slot clone ---------------------------------------------------------

    /// Fork slot state for a session clone: copy the source slot's rows of
    /// every region (`kv.layout.slotBytes()` in all) to the target slot. Rows
    /// past `pos` copy stale bytes that attention never reads, the same
    /// slot-reuse argument as `Sessions.destroy`. Llama has no recurrent state,
    /// so KV is all there is (qwen's clone adds the GDN slabs).
    /// `Sessions.clone` calls this. The presence of this method is what makes
    /// `Sessions(LlamaFusedRunner).can_clone` true, so a swallowed failure
    /// corrupts the fork in silence. It panics instead.
    pub fn cloneSlotState(
        self: *LlamaFusedRunner,
        source_slot: u32,
        target_slot: u32,
        pos: u32,
    ) void {
        std.debug.assert(source_slot < self.n_slots);
        std.debug.assert(target_slot < self.n_slots);
        std.debug.assert(source_slot != target_slot);
        std.debug.assert(pos <= self.max_seq);

        const stride = self.kv.layout.slotBytes();
        log_kv.debug(
            "clone slot {d} -> {d}, pos {d}, {d:.0} MiB",
            .{ source_slot, target_slot, pos, @as(f64, @floatFromInt(stride)) / (1024.0 * 1024.0) },
        );
        var scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
        const copies = self.kv.cloneCopies(source_slot, target_slot, &scratch);
        self.invalidateDecode();
        kernels.runStagingCopies(self.ctx, copies) catch |e|
            std.debug.panic("llama gpu slot clone failed: {t}", .{e});
    }

    // -- tiered kv ----------------------------------------------------------
    // The serve residency layer (serve/residency.zig) keeps active KV in VRAM
    // and spills idle sessions to RAM. A slot's KV is its rows in every region.
    // Eviction copies the rows the session wrote into a caller-owned blob, and
    // a restore writes them back byte for byte. Attention reads only positions
    // below `pos`, so a prefix of `n_rows` rows is the whole live state. A
    // session decoded after a round trip is bit-identical to one that never
    // left VRAM.

    /// Bytes one slot's first `n_rows` KV positions occupy in a host blob.
    pub fn evictSlotBytesFor(self: *const LlamaFusedRunner, n_rows: u32) u64 {
        return self.kv.layout.blobBytes(n_rows);
    }

    /// Build the device<->host copy list for `slot`'s first `n_rows` KV
    /// positions into `out`, in the blob order of `KvStore.rowCopies`.
    /// `to_host` true = save (D2H into the readback stage), false = restore
    /// (H2D out of the swap stage).
    fn slotCopies(
        self: *LlamaFusedRunner,
        slot: u32,
        n_rows: u32,
        to_host: bool,
        out: []kernels.Copy,
    ) []kernels.Copy {
        const stage = if (to_host) self.kv_readback_stage.buffer else self.kv_swap_stage.buffer;
        const copies = self.kv.rowCopies(slot, n_rows, stage, to_host, out);
        std.debug.assert(copies.len == 2 * @as(usize, self.n_layers));
        return copies;
    }

    /// Copy the first `n_rows` KV positions of slot `slot` out of VRAM into
    /// `out` (D2H via the readback stage).
    pub fn evictSlot(self: *LlamaFusedRunner, slot: u32, n_rows: u32, out: []u8) !void {
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);

        const bytes: usize = @intCast(self.evictSlotBytesFor(n_rows));
        std.debug.assert(out.len == bytes);
        const t0 = core_log.now();
        try self.kv_readback_stage.ensure(@min(self.swap_chunk_bytes, bytes));
        kv_budget.assertStageCap(
            self.kv_readback_stage.capacity,
            self.swap_chunk_bytes,
            self.evictSlotBytesFor(self.max_seq),
        );
        var scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
        const copies = self.slotCopies(slot, n_rows, true, &scratch);
        self.invalidateDecode();
        const split = try kernels.walkStage(
            self.ctx,
            copies,
            &self.kv_readback_stage,
            .{ .to_host = out },
            self.swap_chunk_bytes,
        );
        self.last_swap_split = split;
        log_kv.debug(
            "evict slot {d}: {d} rows, {d:.0} MiB, {d:.0} ms ({d:.1} dma + {d:.1} memcpy)",
            .{ slot, n_rows, @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), core_log.msSince(t0), split.dma_ms, split.memcpy_ms },
        );
    }

    /// Restore a previously-evicted `n_rows`-row blob into slot `slot` (H2D).
    pub fn restoreSlot(self: *LlamaFusedRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);

        const bytes: usize = @intCast(self.evictSlotBytesFor(n_rows));
        std.debug.assert(blob.len == bytes);
        const t0 = core_log.now();
        try self.kv_swap_stage.ensure(@min(self.swap_chunk_bytes, bytes));
        kv_budget.assertStageCap(
            self.kv_swap_stage.capacity,
            self.swap_chunk_bytes,
            self.evictSlotBytesFor(self.max_seq),
        );
        var scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
        const copies = self.slotCopies(slot, n_rows, false, &scratch);
        self.invalidateDecode();
        const split = try kernels.walkStage(
            self.ctx,
            copies,
            &self.kv_swap_stage,
            .{ .to_device = blob },
            self.swap_chunk_bytes,
        );
        self.last_swap_split = split;
        log_kv.debug(
            "restore slot {d}: {d} rows, {d:.0} MiB, {d:.0} ms ({d:.1} dma + {d:.1} memcpy)",
            .{ slot, n_rows, @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), core_log.msSince(t0), split.dma_ms, split.memcpy_ms },
        );
    }

    // -- decode step --------------------------------------------------------

    /// Run one decode step. Returns a logits slice owned by `logits_heap`,
    /// valid until the next call to `step`.
    /// The record ring records the step on a miss and the next step while the
    /// GPU runs.
    pub fn step(self: *LlamaFusedRunner, token_id: u32, pos: u32) ![]f32 {
        // Defense in depth: the lone caller (`generate.decode`) guards this.
        // The public single-row entry point still checks its own precondition,
        // as the batched `validateRows` and the CPU runner do.
        if (pos >= self.max_seq) return error.PositionOutOfRange;

        log.debug("step: token {d} at pos {d}", .{ token_id, pos });
        // 1. Embed lookup on CPU into the host-mapped stage_in buffer.
        const x_host = self.stage_in.sliceTyped(f32, self.hidden_dim);
        naive.embedLookup(self.embed.*, token_id, self.hidden_dim, x_host);

        // 2. Submit the forward pass once and wait on the fence.
        std.debug.assert(self.sets_fixed);
        const key = gpu_profile.decodeKey(pos, self.decode.level());
        try self.decode.step(key, key.next(self.max_seq), DecodeRec{ .runner = self }, &self.host);
        const t3 = core_log.now();

        // 4. Stream-copy stage_out into logits_heap once. The host-coherent
        //    mapping behind stage_out is host-visible memory over the
        //    virtualized bus, where a CPU read pays a cache-line snoop. One
        //    bulk memcpy beats the sampler's scattered reads.
        @memcpy(self.logits_heap, self.stage_out.sliceTyped(f32, self.vocab_size));
        self.host.copy.add(step_stats.nsBetween(t3, core_log.now()));
        return self.logits_heap;
    }

    /// The record callback of the decode ring.
    const DecodeRec = struct {
        runner: *LlamaFusedRunner,

        pub fn recordDecode(
            self: DecodeRec,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            key: gpu_profile.DecodeKey,
        ) !void {
            return self.runner.recordDecodeStepOpts(cb, prof, key.pos, key.opts);
        }
    };

    /// Free the decode ring before a submit that is not a decode step.
    pub fn invalidateDecode(self: *LlamaFusedRunner) void {
        self.decode.invalidate(&self.host);
    }

    /// Record a single-row (decode-shape) matmul with the kernel matching
    /// this model's weight quant. Same call shape as LinearQ80Reduce.recordInto.
    fn recordDecodeMatmul(
        self: *LlamaFusedRunner,
        cb: vk.CommandBuffer,
        set: vk.DescriptorSet,
        k: u32,
        n: u32,
        op: []const u8,
    ) void {
        if (comptime core_log.hot) {
            log_dispatch.debug("  matmul {s} reduce ({t}): k {d} n {d}", .{ op, self.quant, k, n });
        }

        switch (self.quant) {
            .q8_0 => self.k_linear.recordInto(cb, set, k, n),
            .q4_0 => self.k_linear_q4.recordInto(cb, set, k, n),
        }
    }

    /// Record a batched (prefill or multi-stream) matmul with the kernel
    /// matching this model's weight quant. A single row is a mat-vec. The batch
    /// kernel's 16-row tile computes and wastes ~16x the work for one decode
    /// row, which turns a memory-bound B=1 serve tick into a compute-bound one.
    /// That trap made the qwen GPU engine ~8x slow at B=1. So B=1 takes the
    /// Reduce kernel, the same path single-stream decode takes.
    fn recordBatchMatmul(
        self: *LlamaFusedRunner,
        cb: vk.CommandBuffer,
        set: vk.DescriptorSet,
        b: u32,
        k: u32,
        n: u32,
        op: []const u8,
    ) void {
        if (b == 1) return self.recordDecodeMatmul(cb, set, k, n, op);
        if (comptime core_log.hot) {
            log_dispatch.debug(
                "  matmul {s} batch ({t}): b {d} k {d} n {d}",
                .{ op, self.quant, b, k, n },
            );
        }

        switch (self.quant) {
            .q8_0 => self.k_linear_batch.recordInto(cb, set, b, k, n),
            .q4_0 => self.k_linear_batch_q4.recordInto(cb, set, b, k, n),
        }
    }

    /// The attention kit over the attention pipelines of the runner.
    fn attnKit(self: *LlamaFusedRunner) kernels.AttnKit {
        return .{
            .single = &self.k_attention,
            .multi = &self.k_attention_multi,
            .passes = &self.k_attention_split,
            .batch = &self.k_attention_batch,
        };
    }

    fn attnShape(self: *const LlamaFusedRunner, stream_stride_elems: u32) kernels.AttnShape {
        return .{
            .n_heads = self.n_heads,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
            .stream_stride_elems = stream_stride_elems,
            .max_splits = self.attn_scratch.max_splits,
        };
    }

    fn recordDecodeStepOpts(
        self: *LlamaFusedRunner,
        cb: vk.CommandBuffer,
        prof: *gpu_profile.GpuProfile,
        pos: u32,
        opts: gpu_profile.RecordOpts,
    ) !void {
        const vkd = self.ctx.vkd;
        try vkd.resetCommandBuffer(cb, .{});
        const begin: vk.CommandBufferBeginInfo = .{
            .flags = .{ .one_time_submit_bit = opts.one_time },
        };
        try vkd.beginCommandBuffer(cb, &begin);

        prof.begin(cb);

        const hidden_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.hidden_dim) * @sizeOf(f32);
        const kv_row_bytes: vk.DeviceSize = self.kvRowBytes();

        // 0. Upload x: stage_in -> buf_x.
        copyHostToDevice(vkd, cb, self.stage_in.buffer, self.buf_x.buffer, hidden_bytes);
        barrierTransferToShader(vkd, cb);
        prof.mark(cb, .copy);

        const attn_plan = attn_split.planDecode(pos + 1);
        const attn_shape = self.attnShape(self.kv.layout.streamStrideElems());
        var layer: u32 = 0;
        while (layer < self.n_layers) : (layer += 1) {
            if (comptime core_log.hot) {
                log_dispatch.debug("record layer {d}", .{layer});
            }

            // -- attention block --------------------------------------------
            // x -> x_norm via rmsnorm
            self.k_rmsnorm.recordInto(
                cb,
                self.set_rmsnorm_attn[layer],
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .norm);

            // x_norm @ Q/K/V -> q, k_scratch, v_scratch
            self.recordDecodeMatmul(
                cb,
                self.set_q[layer],
                self.hidden_dim,
                self.n_heads * self.head_dim,
                "q",
            );
            self.recordDecodeMatmul(
                cb,
                self.set_k[layer],
                self.hidden_dim,
                self.per_step_elems,
                "k",
            );
            self.recordDecodeMatmul(
                cb,
                self.set_v[layer],
                self.hidden_dim,
                self.per_step_elems,
                "v",
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_attn_in);

            // Multi-head RoPE on q and k_scratch.
            self.k_rope_multi.recordInto(
                cb,
                self.set_rope_q,
                pos,
                self.n_heads,
                self.head_dim,
                self.rope_dim,
                self.rope_freq_base,
                .norm,
            );
            self.k_rope_multi.recordInto(
                cb,
                self.set_rope_k,
                pos,
                self.n_kv_heads,
                self.head_dim,
                self.rope_dim,
                self.rope_freq_base,
                .norm,
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .rope);

            // Commit: narrow the roped scratch to the storage element (nothing
            // at f32), then scatter it. One row, so `rows` is 1.
            self.recordKvCommit(cb, 1);

            // KV cache writes via cmdCopyBuffer, into slot 0 at `pos`.
            const src = self.kvCommitSrc();
            const k_row = self.kv.rowAt(layer, .k, 0, pos);
            const v_row = self.kv.rowAt(layer, .v, 0, pos);
            copyDeviceToDevice(vkd, cb, src.k, 0, k_row.buffer, k_row.offset, kv_row_bytes);
            copyDeviceToDevice(vkd, cb, src.v, 0, v_row.buffer, v_row.offset, kv_row_bytes);
            barrierTransferToShader(vkd, cb);
            prof.mark(cb, .copy);

            // Fused attention: (q, k_cache[layer], v_cache[layer], attn_out).
            // A row past one split reads slot 0 through the whole-region set,
            // so it takes the same split shader as a batch row.
            attn_record.record(kernels.AttnKit, self.attnKit(), cb, &attn_plan, .{
                .shape = attn_shape,
                .resident = .{ .single = self.set_attention[layer] },
                .split = self.set_attention_split[layer],
                .merge = self.set_attention_merge,
            });
            barrierShader(vkd, cb);
            prof.mark(cb, .attn);

            // O projection: attn_out @ o_w -> x_norm (reused).
            self.recordDecodeMatmul(
                cb,
                self.set_o[layer],
                self.n_heads * self.head_dim,
                self.hidden_dim,
                "o",
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_out);

            // Residual: x += x_norm.
            self.k_residual.recordInto(cb, self.set_residual_attn, self.hidden_dim);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);

            // -- ffn block --------------------------------------------------
            self.k_rmsnorm.recordInto(
                cb,
                self.set_rmsnorm_ffn[layer],
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .norm);

            self.recordDecodeMatmul(
                cb,
                self.set_gate[layer],
                self.hidden_dim,
                self.ffn_dim,
                "gate",
            );
            self.recordDecodeMatmul(cb, self.set_up[layer], self.hidden_dim, self.ffn_dim, "up");
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_ffn_up);

            // swiglu writes back to ffn_gate (in-place gate = silu(gate) * up).
            self.k_swiglu.recordInto(cb, self.set_swiglu, self.ffn_dim);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);

            // Down projection: ffn_gate @ down_w -> ffn_down.
            self.recordDecodeMatmul(
                cb,
                self.set_down[layer],
                self.ffn_dim,
                self.hidden_dim,
                "down",
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_ffn_down);

            // Residual: x += ffn_down.
            self.k_residual.recordInto(cb, self.set_residual_ffn, self.hidden_dim);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);
        }

        // Final norm + lm_head.
        self.k_rmsnorm.recordInto(cb, self.set_rmsnorm_final, self.hidden_dim, self.rms_norm_eps);
        barrierShader(vkd, cb);
        prof.mark(cb, .norm);
        self.recordDecodeMatmul(cb, self.set_lm_head, self.hidden_dim, self.vocab_size, "lmhead");
        barrierShaderToTransfer(vkd, cb);
        prof.mark(cb, .lmhead);

        // Logits readback: buf_logits -> stage_out.
        const logits_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.vocab_size) * @sizeOf(f32);
        copyDeviceToHost(vkd, cb, self.buf_logits.buffer, self.stage_out.buffer, logits_bytes);
        prof.mark(cb, .copy);

        prof.end(cb);
        try vkd.endCommandBuffer(cb);
    }

    // -- bench hooks --------------------------------------------------------
    // The `bench-decode-host` command times the pieces of one decode step.
    // None of these hooks runs in a normal decode.

    /// Record the decode command buffer for `pos` and stop: no submit. The
    /// profile must stay at the process level. Then the timing is the record cost.
    pub fn probeRecord(self: *LlamaFusedRunner, pos: u32) !void {
        std.debug.assert(self.prof.level == span.gpuLevel());
        std.debug.assert(pos < self.max_seq);

        try self.recordDecodeStepOpts(self.cmd_buffer, &self.prof, pos, .{});
        self.probe_recorded = false;
    }

    /// Record once without the one-time-submit flag, so `probeSubmitWait` can
    /// submit the same buffer again. Resubmission repeats the KV row write, so
    /// the output after it has no meaning.
    pub fn probeRecordReusable(self: *LlamaFusedRunner, pos: u32) !void {
        std.debug.assert(self.prof.level == span.gpuLevel());
        std.debug.assert(pos < self.max_seq);

        try self.recordDecodeStepOpts(self.cmd_buffer, &self.prof, pos, .{ .one_time = false });
        self.probe_recorded = true;
    }

    /// Submit the buffer `probeRecordReusable` recorded and wait for the queue,
    /// the way a normal step does.
    pub fn probeSubmitWait(self: *LlamaFusedRunner) !void {
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
    pub fn setDecodeLevel(self: *LlamaFusedRunner, level: gpu_profile.Level) !void {
        try self.decode.setLevel(level, &self.host);
    }

    /// The timestamps of the last decode step, or null at level `off`.
    pub fn lastDecodeReadout(self: *const LlamaFusedRunner) ?gpu_profile.Readout {
        return self.decode.last;
    }

    // -- batched step -------------------------------------------------------

    /// Rows of the compact logits buffers. `Sessions` derives the vocab stride
    /// from this, so it must state the allocated row count exactly.
    pub fn logitRowsMax(self: *const LlamaFusedRunner) u32 {
        return self.logit_rows_max;
    }

    /// Max rows one `stepBatch` accepts. The batch matmul and KV-write buffers
    /// are sized for `prefill_batch_max`, so a composed tick (decode rows plus
    /// a wide prefill chunk) can use up to this many rows. That is well past
    /// `n_slots` and `logit_rows_max`, which bound only the slot index and the
    /// logits suffix that lm_head runs on.
    pub fn maxStepRows(self: *const LlamaFusedRunner) u32 {
        return self.prefill_batch_max;
    }

    /// Advance a batch of rows, each row a (slot, token, pos) triple over an
    /// independent KV slot of every region. It is the GPU sibling of the CPU
    /// `LlamaRunner.stepBatch` and takes the same row rule: same-slot rows adjacent
    /// and ascending, positions independent otherwise. Weight matmuls run
    /// batched. Rope and attention record single-row dispatches per row, for
    /// arbitrary per-row positions. lm_head runs for the `want_logits` SUFFIX
    /// only, so a prefill chunk's prompt rows skip the model's most costly
    /// projection. Those logits are written COMPACTLY to `logits[0..]` in
    /// suffix order, the same layout `Sessions.advance` reads back.
    pub fn stepBatch(self: *LlamaFusedRunner, rows: []const step_row.StepRow) !void {
        // The shared row rule. The batch width can exceed the slot count,
        // because a wide prefill chunk is one slot and many rows. `n_slots`
        // bounds the slot indices, and `logit_rows_max` bounds the suffix
        // lm_head runs on. Compact logits land at buf_logits[0..] and
        // self.logits[0..], both sized for that many rows: the decode width
        // plus the composed tick's one final-prefill row.
        //
        // n_logits can be 0. An opening (system-prompt) prefill chunk, or a
        // mid-prompt chunk with no decode rows, wants no logits. The record
        // path then skips the lm_head tail and the readback stays empty.
        const n_logits = try step_row.validateRows(
            rows,
            self.n_slots,
            self.prefill_batch_max,
            self.max_seq,
        );
        std.debug.assert(n_logits <= self.logit_rows_max);
        const n: u32 = @intCast(rows.len);
        log.debug(
            "step_batch: {d} rows, slot {d} pos {d}..{d}",
            .{ n, rows[0].slot, rows[0].pos, rows[n - 1].pos },
        );

        // Embed lookup: one row per batch index, CPU-side into stage_in.
        const x_host = self.stage_in.sliceTyped(f32, @as(usize, n) * self.hidden_dim);
        for (rows, 0..) |r, i| {
            const dst_off = @as(usize, i) * self.hidden_dim;
            naive.embedLookup(
                self.embed.*,
                r.token,
                self.hidden_dim,
                x_host[dst_off..][0..self.hidden_dim],
            );
        }

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

        // Stream-copy stage_out into ordinary heap memory, for the same reason
        // the decode and prefill paths do. The GPU computed and read back only
        // the want_logits suffix, already compacted to the front.
        const n_floats = @as(usize, n_logits) * self.vocab_size;
        @memcpy(self.logits[0..n_floats], self.stage_out.sliceTyped(f32, n_floats));
    }

    /// Submit `cmd_buffer`, wait for the queue, and read its timestamps back.
    /// The host times exist only when the buffer wrote timestamps or the span
    /// recorder runs, so the off path reads no clock.
    fn submitWait(self: *LlamaFusedRunner, label: span.GpuLabel) !void {
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
        _ = try self.prof.collectInto(.llama_fused, label, t0, t2);
    }

    fn recordStepBatch(self: *LlamaFusedRunner, rows: []const step_row.StepRow) !void {
        const n: u32 = @intCast(rows.len);
        const vkd = self.ctx.vkd;
        const cb = self.cmd_buffer;
        try vkd.resetCommandBuffer(cb, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try vkd.beginCommandBuffer(cb, &begin);
        self.prof.begin(cb);

        const f32_size: vk.DeviceSize = @sizeOf(f32);
        const hidden_row_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.hidden_dim) * f32_size;
        const total_x_bytes: vk.DeviceSize = @as(vk.DeviceSize, n) * hidden_row_bytes;
        const kv_row_bytes: vk.DeviceSize = self.kvRowBytes();
        // The attention_multi shader reads `k_buf[slot * stream_stride +
        // offset in the slot]`, and each descriptor binds one whole region.
        const stream_stride_elems: u32 = self.kv.layout.streamStrideElems();
        // Every layer records the same attention plan.
        std.debug.assert(n <= self.prefill_batch_max);
        const attn_plan = attn_record.planStep(rows, self.attn_scratch.max_splits);
        const attn_shape = self.attnShape(stream_stride_elems);

        copyHostToDevice(vkd, cb, self.stage_in.buffer, self.buf_x.buffer, total_x_bytes);
        barrierTransferToShader(vkd, cb);
        self.prof.mark(cb, .copy);

        var layer: u32 = 0;
        while (layer < self.n_layers) : (layer += 1) {
            if (comptime core_log.hot) {
                log_dispatch.debug("record layer {d}", .{layer});
            }

            // attn pre-norm: per-stream rmsnorm
            self.k_rmsnorm_batch.recordInto(
                cb,
                self.set_rmsnorm_attn[layer],
                n,
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);

            // Q/K/V matmuls: B = n streams, output goes to per-row
            // slots in buf_q / buf_k_scratch / buf_v_scratch.
            self.recordBatchMatmul(
                cb,
                self.set_q[layer],
                n,
                self.hidden_dim,
                self.n_heads * self.head_dim,
                "q",
            );
            self.recordBatchMatmul(
                cb,
                self.set_k[layer],
                n,
                self.hidden_dim,
                self.per_step_elems,
                "k",
            );
            self.recordBatchMatmul(
                cb,
                self.set_v[layer],
                n,
                self.hidden_dim,
                self.per_step_elems,
                "v",
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_attn_in);

            // Multi-head RoPE: one single-row dispatch per row at the row's
            // OWN position (row_base aims the dispatch, pos_stride 0).
            for (rows, 0..) |r, i| {
                const row: u32 = @intCast(i);
                self.k_rope_multi_batch.recordInto(
                    cb,
                    self.set_rope_q,
                    r.pos,
                    self.n_heads,
                    self.head_dim,
                    self.rope_dim,
                    self.rope_freq_base,
                    1,
                    0,
                    row,
                    .norm,
                );
                self.k_rope_multi_batch.recordInto(
                    cb,
                    self.set_rope_k,
                    r.pos,
                    self.n_kv_heads,
                    self.head_dim,
                    self.rope_dim,
                    self.rope_freq_base,
                    1,
                    0,
                    row,
                    .norm,
                );
            }

            barrierShader(vkd, cb);
            self.prof.mark(cb, .rope);

            // Commit: narrow all n roped rows in one dispatch (nothing at
            // f32), then scatter them.
            self.recordKvCommit(cb, n);

            // KV cache writes: one copy region per row, at its OWN position in
            // its SLOT's rows of this layer's K region and V region. Each region
            // lives in one buffer, so each tensor takes one copy command.
            const src = self.kvCommitSrc();
            const k_dst = self.kv.region(layer, .k).buffer;
            const v_dst = self.kv.region(layer, .v).buffer;
            var k_regions_buf: [
                limits.max_batch +
                    limits.max_prefill_chunk
            ]vk.BufferCopy = undefined;
            var v_regions_buf: [
                limits.max_batch +
                    limits.max_prefill_chunk
            ]vk.BufferCopy = undefined;
            const k_regions = k_regions_buf[0..n];
            const v_regions = v_regions_buf[0..n];
            for (rows, 0..) |r, i| {
                const k_row = self.kv.rowAt(layer, .k, r.slot, r.pos);
                const v_row = self.kv.rowAt(layer, .v, r.slot, r.pos);
                std.debug.assert(k_row.buffer == k_dst);
                std.debug.assert(v_row.buffer == v_dst);
                k_regions[i] = .{
                    .src_offset = @as(vk.DeviceSize, i) * kv_row_bytes,
                    .dst_offset = k_row.offset,
                    .size = kv_row_bytes,
                };
                v_regions[i] = .{
                    .src_offset = @as(vk.DeviceSize, i) * kv_row_bytes,
                    .dst_offset = v_row.offset,
                    .size = kv_row_bytes,
                };
            }

            vkd.cmdCopyBuffer(cb, src.k, k_dst, k_regions);
            vkd.cmdCopyBuffer(cb, src.v, v_dst, v_regions);
            barrierTransferToShader(vkd, cb);
            self.prof.mark(cb, .copy);

            // Attention: single-row dispatches per row, each with its own
            // t_len and its own KV stream. The copies above committed every
            // row's K and V, so consecutive rows of one slot see exactly their
            // causal prefix.
            attn_record.record(kernels.AttnKit, self.attnKit(), cb, &attn_plan, .{
                .shape = attn_shape,
                .resident = .{ .multi = self.set_attention_multi[layer] },
                .split = self.set_attention_split[layer],
                .merge = self.set_attention_merge,
            });
            barrierShader(vkd, cb);
            self.prof.mark(cb, .attn);

            self.recordBatchMatmul(
                cb,
                self.set_o[layer],
                n,
                self.n_heads * self.head_dim,
                self.hidden_dim,
                "o",
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_out);
            self.k_residual.recordInto(cb, self.set_residual_attn, n * self.hidden_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);

            self.k_rmsnorm_batch.recordInto(
                cb,
                self.set_rmsnorm_ffn[layer],
                n,
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);
            self.recordBatchMatmul(
                cb,
                self.set_gate[layer],
                n,
                self.hidden_dim,
                self.ffn_dim,
                "gate",
            );
            self.recordBatchMatmul(cb, self.set_up[layer], n, self.hidden_dim, self.ffn_dim, "up");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_up);
            self.k_swiglu.recordInto(cb, self.set_swiglu, n * self.ffn_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
            self.recordBatchMatmul(
                cb,
                self.set_down[layer],
                n,
                self.ffn_dim,
                self.hidden_dim,
                "down",
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_down);
            self.k_residual.recordInto(cb, self.set_residual_ffn, n * self.hidden_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
        }

        // A chunk with no want-logits row (see stepBatch) is done here, because
        // the KV writes above are its whole product. A zero-size lm_head or
        // copy must not reach the command buffer, because a zero-size
        // cmdCopyBuffer is not valid Vulkan.
        const first = step_row.wantLogitsFirst(rows);
        if (first == n) {
            self.prof.end(cb);
            try vkd.endCommandBuffer(cb);
            return;
        }

        // Final norm over all rows (elementwise, low cost), then lm_head for the
        // want_logits SUFFIX only. The suffix is compacted to buf_x_norm[0..] so
        // the offset-0 lm_head descriptor sees it. Logits then land compactly at
        // buf_logits[0..n_logits], the layout Sessions.advance reads.
        self.k_rmsnorm_batch.recordInto(
            cb,
            self.set_rmsnorm_final,
            n,
            self.hidden_dim,
            self.rms_norm_eps,
        );

        const n_logits: u32 = n - first;
        if (first > 0) {
            // Compact the suffix to the front, bouncing through buf_x (free after
            // the final norm) so the in-place src/dst ranges do not overlap.
            // cmdCopyBuffer forbids overlap, and a small chunk can have
            // first < n_logits.
            barrierShaderToTransfer(vkd, cb);
            const suffix_bytes: vk.DeviceSize = @as(vk.DeviceSize, n_logits) * hidden_row_bytes;
            copyDeviceToDevice(
                vkd,
                cb,
                self.buf_x_norm.buffer,
                @as(vk.DeviceSize, first) * hidden_row_bytes,
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

        // Batched lm_head: B = n_logits want-logits rows. Output is (n_logits, vocab).
        self.recordBatchMatmul(
            cb,
            self.set_lm_head,
            n_logits,
            self.hidden_dim,
            self.vocab_size,
            "lmhead",
        );
        barrierShaderToTransfer(vkd, cb);
        self.prof.mark(cb, .lmhead);

        const logits_bytes: vk.DeviceSize = @as(vk.DeviceSize, n_logits) * @as(
            vk.DeviceSize,
            self.vocab_size,
        ) * f32_size;
        copyDeviceToHost(vkd, cb, self.buf_logits.buffer, self.stage_out.buffer, logits_bytes);
        self.prof.mark(cb, .copy);
        self.prof.end(cb);
        try vkd.endCommandBuffer(cb);
    }

    // -- prefill ------------------------------------------------------------

    /// Process `b` prompt tokens through the model in ONE batched forward pass.
    /// Returns logits for the LAST token only, which is enough to sample the
    /// first decode token after prefill. `start_pos` is the absolute sequence
    /// position of token 0 in `tokens`. It is 0 for a fresh conversation, and
    /// for multi-turn it continues where the previous turn ended. The call
    /// updates KV cache rows [start_pos, start_pos + b).
    ///
    /// `b` must be at most `prefill_batch_max`. Callers chunk longer prompts.
    /// The returned slice is the heap-cached logits, valid until the next
    /// `step` or `prefillBatch` call.
    pub fn prefillBatch(self: *LlamaFusedRunner, tokens: []const u32, start_pos: u32) ![]f32 {
        std.debug.assert(tokens.len > 0);
        std.debug.assert(tokens.len <= self.prefill_batch_max);

        const b: u32 = @intCast(tokens.len);
        log.debug("prefill: {d} tokens at start_pos {d}", .{ b, start_pos });

        // 1. Embedding lookup on CPU into the host-mapped stage_in
        //    buffer. One row per prompt token.
        const x_host = self.stage_in.sliceTyped(f32, @as(usize, b) * self.hidden_dim);
        var i: u32 = 0;
        while (i < b) : (i += 1) {
            const dst_off = @as(usize, i) * self.hidden_dim;
            naive.embedLookup(
                self.embed.*,
                tokens[i],
                self.hidden_dim,
                x_host[dst_off..][0..self.hidden_dim],
            );
        }

        // 2. Record + submit the batched forward pass.
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

        // 3. Copy logits into ordinary heap memory, for the same reason the
        //    decode step does.
        @memcpy(self.logits_heap, self.stage_out.sliceTyped(f32, self.vocab_size));
        return self.logits_heap;
    }

    fn recordPrefillStep(self: *LlamaFusedRunner, b: u32, start_pos: u32) !void {
        const vkd = self.ctx.vkd;
        const cb = self.cmd_buffer;
        try vkd.resetCommandBuffer(cb, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try vkd.beginCommandBuffer(cb, &begin);

        self.prof.begin(cb);

        const f32_size: vk.DeviceSize = @sizeOf(f32);
        const hidden_row_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.hidden_dim) * f32_size;
        const total_x_bytes: vk.DeviceSize = @as(vk.DeviceSize, b) * hidden_row_bytes;
        const kv_row_bytes: vk.DeviceSize = self.kvRowBytes();
        const total_kv_rows_bytes: vk.DeviceSize = @as(vk.DeviceSize, b) * kv_row_bytes;
        // The B rows land contiguous in slot 0, so they stay inside its rows.
        std.debug.assert(@as(u64, start_pos) + b <= self.max_seq);
        // Every layer records the same plan.
        std.debug.assert(b <= attn_split.max_plan_rows);
        const prefill_plan = attn_split.planPrefill(0, start_pos, b, self.attn_scratch.max_splits);
        const prefill_shape = self.attnShape(self.kv.layout.streamStrideElems());

        copyHostToDevice(vkd, cb, self.stage_in.buffer, self.buf_x.buffer, total_x_bytes);
        barrierTransferToShader(vkd, cb);
        self.prof.mark(cb, .copy);

        var layer: u32 = 0;
        while (layer < self.n_layers) : (layer += 1) {
            if (comptime core_log.hot) {
                log_dispatch.debug("record layer {d}", .{layer});
            }

            // attn pre-norm: batched
            self.k_rmsnorm_batch.recordInto(
                cb,
                self.set_rmsnorm_attn[layer],
                b,
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);

            // Q/K/V projections: batched matmul, (B, hidden) -> (B, *)
            self.recordBatchMatmul(
                cb,
                self.set_q[layer],
                b,
                self.hidden_dim,
                self.n_heads * self.head_dim,
                "q",
            );
            self.recordBatchMatmul(
                cb,
                self.set_k[layer],
                b,
                self.hidden_dim,
                self.per_step_elems,
                "k",
            );
            self.recordBatchMatmul(
                cb,
                self.set_v[layer],
                b,
                self.hidden_dim,
                self.per_step_elems,
                "v",
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_attn_in);

            // Multi-head RoPE: per-row position = start_pos + row_idx.
            self.k_rope_multi_batch.recordInto(
                cb,
                self.set_rope_q,
                start_pos,
                self.n_heads,
                self.head_dim,
                self.rope_dim,
                self.rope_freq_base,
                b,
                1,
                0,
                .norm,
            );
            self.k_rope_multi_batch.recordInto(
                cb,
                self.set_rope_k,
                start_pos,
                self.n_kv_heads,
                self.head_dim,
                self.rope_dim,
                self.rope_freq_base,
                b,
                1,
                0,
                .norm,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .rope);

            // Commit: narrow all B roped rows in one dispatch (nothing at
            // f32), then copy them.
            self.recordKvCommit(cb, b);

            // KV cache writes: B contiguous rows of slot 0 starting at
            // start_pos.
            const src = self.kvCommitSrc();
            const k_row = self.kv.rowAt(layer, .k, 0, start_pos);
            const v_row = self.kv.rowAt(layer, .v, 0, start_pos);
            copyDeviceToDevice(vkd, cb, src.k, 0, k_row.buffer, k_row.offset, total_kv_rows_bytes);
            copyDeviceToDevice(vkd, cb, src.v, 0, v_row.buffer, v_row.offset, total_kv_rows_bytes);
            barrierTransferToShader(vkd, cb);
            self.prof.mark(cb, .copy);

            // Causal attention: each query row attends to positions
            // [0, start_pos + row_idx]. The resident prefix runs the batch
            // kernel on the slot-0 set. A row past one split reads slot 0
            // through the whole-region set, as a decode row does, so it gets
            // the bytes of that decode row.
            attn_record.recordPrefill(kernels.AttnKit, self.attnKit(), cb, &prefill_plan, .{
                .shape = prefill_shape,
                .resident = .{ .batch = self.set_attention[layer] },
                .split = self.set_attention_split[layer],
                .merge = self.set_attention_merge,
            }, start_pos);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .attn);

            // O projection (N-outer layout).
            self.recordBatchMatmul(
                cb,
                self.set_o[layer],
                b,
                self.n_heads * self.head_dim,
                self.hidden_dim,
                "o",
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_out);

            // Residual: x += x_norm. Elementwise over B*hidden floats.
            self.k_residual.recordInto(cb, self.set_residual_attn, b * self.hidden_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);

            // FFN pre-norm.
            self.k_rmsnorm_batch.recordInto(
                cb,
                self.set_rmsnorm_ffn[layer],
                b,
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);

            self.recordBatchMatmul(
                cb,
                self.set_gate[layer],
                b,
                self.hidden_dim,
                self.ffn_dim,
                "gate",
            );
            self.recordBatchMatmul(cb, self.set_up[layer], b, self.hidden_dim, self.ffn_dim, "up");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_up);

            // SwiGLU: elementwise over B*ffn_dim entries.
            self.k_swiglu.recordInto(cb, self.set_swiglu, b * self.ffn_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);

            // Down projection.
            self.recordBatchMatmul(
                cb,
                self.set_down[layer],
                b,
                self.ffn_dim,
                self.hidden_dim,
                "down",
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_down);

            // Residual: x += ffn_down. Elementwise.
            self.k_residual.recordInto(cb, self.set_residual_ffn, b * self.hidden_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
        }

        // Final norm.
        self.k_rmsnorm_batch.recordInto(
            cb,
            self.set_rmsnorm_final,
            b,
            self.hidden_dim,
            self.rms_norm_eps,
        );
        barrierShader(vkd, cb);
        self.prof.mark(cb, .norm);

        // Only the LAST batch row's logits matter, because they sample the
        // first decode token. Copy the last row of buf_x_norm to its first row,
        // so the per-row lm_head descriptor bound at offset 0 sees the right
        // input. cmdCopyBuffer forbids overlapping ranges within one buffer.
        // At B >= 2 the source range starts past offset hidden_row_bytes, so it
        // does not overlap [0..hidden_row_bytes).
        if (b > 1) {
            const src_off: vk.DeviceSize = @as(vk.DeviceSize, b - 1) * hidden_row_bytes;
            copyDeviceToDevice(
                vkd,
                cb,
                self.buf_x_norm.buffer,
                src_off,
                self.buf_x_norm.buffer,
                0,
                hidden_row_bytes,
            );
            barrierTransferToShader(vkd, cb);
        }

        // lm_head: per-row decode matmul. The reduce variant is the best B=1
        // kernel, because a large N makes the dispatch compute-bound.
        self.recordDecodeMatmul(cb, self.set_lm_head, self.hidden_dim, self.vocab_size, "lmhead");
        barrierShaderToTransfer(vkd, cb);
        self.prof.mark(cb, .lmhead);

        const logits_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.vocab_size) * f32_size;
        copyDeviceToHost(vkd, cb, self.buf_logits.buffer, self.stage_out.buffer, logits_bytes);
        self.prof.mark(cb, .copy);
        self.prof.end(cb);
        try vkd.endCommandBuffer(cb);
    }

    // -- profiling ----------------------------------------------------------

    /// Run ONE batched prefill with GPU timestamps and print a per-`Phase`
    /// breakdown. It is the diagnostic behind `--profile-prefill`. It mutates
    /// the KV like a real prefill, so the caller must treat the runner as dirty
    /// afterwards.
    pub fn profilePrefill(
        self: *LlamaFusedRunner,
        io: std.Io,
        tokens: []const u32,
        start_pos: u32,
        tag: []const u8,
    ) !void {
        std.debug.assert(tokens.len > 0 and tokens.len <= self.prefill_batch_max);

        const level = self.prof.level;
        // The pool grows on the first probe, so level off creates none. No
        // submit of this buffer is pending here.
        try self.prof.reserve(.phases);
        self.prof.setLevel(.phases);
        defer self.prof.setLevel(level);

        const b: u32 = @intCast(tokens.len);

        const x_host = self.stage_in.sliceTyped(f32, @as(usize, b) * self.hidden_dim);
        var i: u32 = 0;
        while (i < b) : (i += 1) {
            const dst_off = @as(usize, i) * self.hidden_dim;
            naive.embedLookup(
                self.embed.*,
                tokens[i],
                self.hidden_dim,
                x_host[dst_off..][0..self.hidden_dim],
            );
        }

        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        try self.recordPrefillStep(b, start_pos);
        const t_rec = std.Io.Clock.Timestamp.now(io, .awake);
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        self.invalidateDecode();
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
        const t_wait = std.Io.Clock.Timestamp.now(io, .awake);

        const r = (try self.prof.collectInto(
            .llama_fused,
            .prefill_chunk,
            t_rec.raw,
            t_wait.raw,
        )).?;
        try r.report(
            io,
            "[llama prefill {s}] b={d}",
            .{ tag, b },
            time.msBetween(t0, t_rec),
            time.msBetween(t_rec, t_wait),
        );
    }

    /// Run ONE decode step with GPU timestamps and print a per-`Phase`
    /// breakdown, the decode twin of `profilePrefill`. It mutates the KV like a
    /// real step, so the runner is dirty afterwards.
    pub fn profileStep(
        self: *LlamaFusedRunner,
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

        const x_host = self.stage_in.sliceTyped(f32, self.hidden_dim);
        naive.embedLookup(self.embed.*, token_id, self.hidden_dim, x_host);
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        try self.recordDecodeStepOpts(self.cmd_buffer, &self.prof, pos, .{});
        const t_rec = std.Io.Clock.Timestamp.now(io, .awake);
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
        const t_wait = std.Io.Clock.Timestamp.now(io, .awake);
        @memcpy(self.logits_heap, self.stage_out.sliceTyped(f32, self.vocab_size));

        const r = (try self.prof.collectInto(.llama_fused, .decode, t_rec.raw, t_wait.raw)).?;
        try r.report(
            io,
            "[llama decode {s}] pos={d}",
            .{ tag, pos },
            time.msBetween(t0, t_rec),
            time.msBetween(t_rec, t_wait),
        );
    }

    // -- descriptor sets ----------------------------------------------------

    fn allocateAndBindDescriptorSets(self: *LlamaFusedRunner) !void {
        std.debug.assert(!self.sets_fixed);

        const ctx = self.ctx;
        const layouts = struct {
            rmsnorm: vk.DescriptorSetLayout,
            matmul: vk.DescriptorSetLayout,
            attention: vk.DescriptorSetLayout,
            residual: vk.DescriptorSetLayout,
            swiglu: vk.DescriptorSetLayout,
            rope_multi: vk.DescriptorSetLayout,
        }{
            .rmsnorm = self.k_rmsnorm.pipeline.set_layout,
            .matmul = self.k_linear.pipeline.set_layout,
            .attention = self.k_attention.pipeline.set_layout,
            .residual = self.k_residual.pipeline.set_layout,
            .swiglu = self.k_swiglu.pipeline.set_layout,
            .rope_multi = self.k_rope_multi.pipeline.set_layout,
        };

        // Helper that allocates one descriptor set from self.desc_pool.
        const allocOne = struct {
            fn f(
                ctx2: *const vulkan_ctx.Context,
                pool: vk.DescriptorPool,
                layout: vk.DescriptorSetLayout,
            ) !vk.DescriptorSet {
                const layouts_arr = [_]vk.DescriptorSetLayout{layout};
                const alloc_info: vk.DescriptorSetAllocateInfo = .{
                    .descriptor_pool = pool,
                    .descriptor_set_count = 1,
                    .p_set_layouts = &layouts_arr,
                };
                var set: vk.DescriptorSet = .null_handle;
                try ctx2.vkd.allocateDescriptorSets(ctx2.device, &alloc_info, @ptrCast(&set));
                return set;
            }
        }.f;

        // Descriptor ranges cover the MAX batched extent. Decode
        // (B=1) reads only the first row. Batched prefill (B<=batch_max)
        // reads B rows. The shader uses a `b` push constant to decide
        // the actual extent. Norm and KV cache ranges stay per-row /
        // per-layer because they are not batched.
        const bm: vk.DeviceSize = @intCast(self.prefill_batch_max);
        const hidden_bytes: vk.DeviceSize = bm * @as(vk.DeviceSize, self.hidden_dim) * @sizeOf(f32);
        const qkv_q_bytes: vk.DeviceSize = bm * @as(
            vk.DeviceSize,
            self.n_heads * self.head_dim,
        ) * @sizeOf(f32);
        const kv_scratch_bytes: vk.DeviceSize = bm * @as(
            vk.DeviceSize,
            self.per_step_elems,
        ) * @sizeOf(f32);
        const ffn_bytes: vk.DeviceSize = bm * @as(vk.DeviceSize, self.ffn_dim) * @sizeOf(f32);
        // The logits descriptor range covers logit_rows_max rows, the full
        // batched output of a composed tick. Single-row decode and prefill read
        // only the first row.
        const logits_bytes: vk.DeviceSize = @as(
            vk.DeviceSize,
            self.logit_rows_max,
        ) * @as(vk.DeviceSize, self.vocab_size) * @sizeOf(f32);
        // Every attention binding below starts at a region base, which the
        // store packs at whole regions. A storage-buffer descriptor offset
        // must be a multiple of the device offset alignment, so a region must
        // be a whole number of alignment units.
        std.debug.assert(self.kv.layout.regionBytes() % self.kv.lim.offset_alignment == 0);
        // The guard in `init` proved that the records fit one binding.
        const split_bytes: vk.DeviceSize = self.attn_scratch.totalBytes();
        std.debug.assert(split_bytes <= self.buf_attn_split.capacity);
        std.debug.assert(split_bytes <= self.ctx.buffer_limits.storage_range);

        // Per-layer rmsnorm + matmul + attention sets.
        var layer: u32 = 0;
        while (layer < self.n_layers) : (layer += 1) {
            // rmsnorm attn-pre: (x, attn_norm_w[L], x_norm)
            self.set_rmsnorm_attn[layer] = try allocOne(ctx, self.desc_pool, layouts.rmsnorm);
            bindRmsnorm(
                ctx,
                self.set_rmsnorm_attn[layer],
                self.buf_x.buffer,
                self.attn_norm_w[layer],
                self.buf_x_norm.buffer,
                hidden_bytes,
            );

            // rmsnorm ffn-pre: (x, ffn_norm_w[L], x_norm)
            self.set_rmsnorm_ffn[layer] = try allocOne(ctx, self.desc_pool, layouts.rmsnorm);
            bindRmsnorm(
                ctx,
                self.set_rmsnorm_ffn[layer],
                self.buf_x.buffer,
                self.ffn_norm_w[layer],
                self.buf_x_norm.buffer,
                hidden_bytes,
            );

            // matmul Q: (x_norm, q_w.scales, q_w.quants, q)
            self.set_q[layer] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_q[layer],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.attn_q_w[layer],
                self.buf_q.buffer,
                qkv_q_bytes,
            );

            // matmul K: (x_norm, k_w, k_scratch)
            self.set_k[layer] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_k[layer],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.attn_k_w[layer],
                self.buf_k_scratch.buffer,
                kv_scratch_bytes,
            );

            // matmul V: (x_norm, v_w, v_scratch)
            self.set_v[layer] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_v[layer],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.attn_v_w[layer],
                self.buf_v_scratch.buffer,
                kv_scratch_bytes,
            );

            // matmul O: (attn_out, o_w, x_norm). It writes back to x_norm,
            // which the residual then consumes.
            self.set_o[layer] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_o[layer],
                self.buf_attn_out.buffer,
                qkv_q_bytes,
                &self.attn_o_w[layer],
                self.buf_x_norm.buffer,
                hidden_bytes,
            );

            // matmul gate: (x_norm, gate_w, ffn_gate)
            self.set_gate[layer] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_gate[layer],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.ffn_gate_w[layer],
                self.buf_ffn_gate.buffer,
                ffn_bytes,
            );

            // matmul up: (x_norm, up_w, ffn_up)
            self.set_up[layer] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_up[layer],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.ffn_up_w[layer],
                self.buf_ffn_up.buffer,
                ffn_bytes,
            );

            // matmul down: (ffn_gate, down_w, ffn_down). swiglu wrote silu(gate)*up back into
            // ffn_gate.
            self.set_down[layer] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_down[layer],
                self.buf_ffn_gate.buffer,
                ffn_bytes,
                &self.ffn_down_w[layer],
                self.buf_ffn_down.buffer,
                hidden_bytes,
            );

            // attention: (q, slot 0 of k_cache[L], slot 0 of v_cache[L],
            // attn_out)
            self.set_attention[layer] = try allocOne(ctx, self.desc_pool, layouts.attention);
            bindAttention(
                ctx,
                self.set_attention[layer],
                self.buf_q.buffer,
                qkv_q_bytes,
                self.kv.slotView(layer, .k, 0),
                self.kv.slotView(layer, .v, 0),
                self.buf_attn_out.buffer,
                qkv_q_bytes,
            );

            // Multi-slot attention descriptor: binds the whole K region and
            // the whole V region of this layer. The shader strides by slot
            // (`stream_stride_elems`).
            self.set_attention_multi[layer] = try allocOne(
                ctx,
                self.desc_pool,
                self.k_attention_multi.pipeline.set_layout,
            );
            bindAttention(
                ctx,
                self.set_attention_multi[layer],
                self.buf_q.buffer,
                qkv_q_bytes,
                self.kv.region(layer, .k),
                self.kv.region(layer, .v),
                self.buf_attn_out.buffer,
                qkv_q_bytes,
            );

            // Split attention: (q, the whole K region, the whole V region,
            // the records).
            self.set_attention_split[layer] = try allocOne(
                ctx,
                self.desc_pool,
                self.k_attention_split.split.set_layout,
            );
            kernels.AttentionSplit.assertKBinding(self.kv.region(layer, .k));
            kernels.AttentionSplit.assertVBinding(self.kv.region(layer, .v));
            bindAttention(
                ctx,
                self.set_attention_split[layer],
                self.buf_q.buffer,
                qkv_q_bytes,
                self.kv.region(layer, .k),
                self.kv.region(layer, .v),
                self.buf_attn_split.buffer,
                split_bytes,
            );
        }

        // Split merge: (the records, attn_out).
        self.set_attention_merge = try allocOne(
            ctx,
            self.desc_pool,
            self.k_attention_split.merge.set_layout,
        );
        kernels.updateStorageBindings(ctx, self.set_attention_merge, &.{
            .{ .buffer = self.buf_attn_split.buffer, .offset = 0, .range = split_bytes },
            .{ .buffer = self.buf_attn_out.buffer, .offset = 0, .range = qkv_q_bytes },
        });

        // Final rmsnorm: (x, output_norm_w, x_norm)
        self.set_rmsnorm_final = try allocOne(ctx, self.desc_pool, layouts.rmsnorm);
        bindRmsnorm(
            ctx,
            self.set_rmsnorm_final,
            self.buf_x.buffer,
            self.output_norm_w,
            self.buf_x_norm.buffer,
            hidden_bytes,
        );

        // lm_head: (x_norm, output_w, logits)
        self.set_lm_head = try allocOne(ctx, self.desc_pool, layouts.matmul);
        bindMatmul(
            ctx,
            self.set_lm_head,
            self.buf_x_norm.buffer,
            hidden_bytes,
            &self.output_w,
            self.buf_logits.buffer,
            logits_bytes,
        );

        // swiglu: (ffn_gate, ffn_up, ffn_gate). dst = ffn_gate (in-place).
        self.set_swiglu = try allocOne(ctx, self.desc_pool, layouts.swiglu);
        kernels.updateStorageBindings(ctx, self.set_swiglu, &.{
            .{ .buffer = self.buf_ffn_gate.buffer, .offset = 0, .range = ffn_bytes },
            .{ .buffer = self.buf_ffn_up.buffer, .offset = 0, .range = ffn_bytes },
            .{ .buffer = self.buf_ffn_gate.buffer, .offset = 0, .range = ffn_bytes },
        });

        // residual_attn: (x_norm, x). It adds x_norm into x.
        self.set_residual_attn = try allocOne(ctx, self.desc_pool, layouts.residual);
        kernels.updateStorageBindings(ctx, self.set_residual_attn, &.{
            .{ .buffer = self.buf_x_norm.buffer, .offset = 0, .range = hidden_bytes },
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = hidden_bytes },
        });

        // residual_ffn: (ffn_down, x). It adds ffn_down into x.
        self.set_residual_ffn = try allocOne(ctx, self.desc_pool, layouts.residual);
        kernels.updateStorageBindings(ctx, self.set_residual_ffn, &.{
            .{ .buffer = self.buf_ffn_down.buffer, .offset = 0, .range = hidden_bytes },
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = hidden_bytes },
        });

        // rope_multi Q: (q_buf, freq factors)
        const ff_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.rope_dim / 2) * @sizeOf(f32);
        self.set_rope_q = try allocOne(ctx, self.desc_pool, layouts.rope_multi);
        kernels.updateStorageBindings(ctx, self.set_rope_q, &.{
            .{ .buffer = self.buf_q.buffer, .offset = 0, .range = qkv_q_bytes },
            .{ .buffer = self.buf_rope_ff.buffer, .offset = 0, .range = ff_bytes },
        });

        // rope_multi K: (k_scratch_buf, freq factors)
        self.set_rope_k = try allocOne(ctx, self.desc_pool, layouts.rope_multi);
        kernels.updateStorageBindings(ctx, self.set_rope_k, &.{
            .{ .buffer = self.buf_k_scratch.buffer, .offset = 0, .range = kv_scratch_bytes },
            .{ .buffer = self.buf_rope_ff.buffer, .offset = 0, .range = ff_bytes },
        });

        // kv commit: (k_scratch, v_scratch, k_scratch_kv, v_scratch_kv). The
        // sources are the f32 rope scratch, the destinations the same rows at
        // the storage element. Bound once, because the buffers never move.
        if (self.k_kv_commit) |*k| {
            const kv_narrow_bytes: vk.DeviceSize = bm * self.kvRowBytes();
            self.set_kv_commit = try allocOne(ctx, self.desc_pool, k.pipeline.set_layout);
            kernels.updateStorageBindings(ctx, self.set_kv_commit, &.{
                .{ .buffer = self.buf_k_scratch.buffer, .offset = 0, .range = kv_scratch_bytes },
                .{ .buffer = self.buf_v_scratch.buffer, .offset = 0, .range = kv_scratch_bytes },
                .{ .buffer = self.buf_k_scratch_kv.buffer, .offset = 0, .range = kv_narrow_bytes },
                .{ .buffer = self.buf_v_scratch_kv.buffer, .offset = 0, .range = kv_narrow_bytes },
            });
        }
    }
};

// -- helpers ----------------------------------------------------------------

/// Suballocate space in `arena` for the norm weight, stage the upload
/// through a transient HostBuffer, and return a view into the arena
/// that callers can use in descriptor writes.
fn uploadFp32NormInArena(
    ctx: *const vulkan_ctx.Context,
    t: Tensor,
    arena: *kernels.VramArena,
) !vk.DescriptorBufferInfo {
    if (t.dtype != .f32) return error.UnsupportedWeightDtype;

    const bytes: vk.DeviceSize = @intCast(t.data.len);

    var staging: kernels.HostBuffer = .{ .ctx = ctx };
    defer staging.deinit();

    try staging.ensure(bytes);
    @memcpy(staging.sliceTyped(u8, @intCast(bytes)), t.data);

    const view = try arena.alloc(bytes);
    const copies = [_]kernels.Copy{.{
        .src = staging.buffer,
        .dst = view.buffer,
        .size = bytes,
        .dst_offset = view.offset,
    }};
    try kernels.runStagingCopies(ctx, &copies);
    return view;
}

fn makeQ80InArenas(
    ctx: *const vulkan_ctx.Context,
    scales_arena: *kernels.VramArena,
    quants_arena: *kernels.VramArena,
    t: Tensor,
) !kernels.Q80Weight {
    if (t.dtype != .q8_0) return error.UnsupportedWeightDtype;

    const in_features: usize = @intCast(t.shape[0]);
    const out_features: usize = @intCast(t.shape[1]);
    return try kernels.Q80Weight.initInArenas(
        ctx,
        scales_arena,
        quants_arena,
        t.data,
        in_features,
        out_features,
    );
}

fn makeQ40InArenas(
    ctx: *const vulkan_ctx.Context,
    scales_arena: *kernels.VramArena,
    quants_arena: *kernels.VramArena,
    t: Tensor,
) !kernels.Q80Weight {
    if (t.dtype != .q4_0) return error.UnsupportedWeightDtype;

    const in_features: usize = @intCast(t.shape[0]);
    const out_features: usize = @intCast(t.shape[1]);
    return try kernels.Q80Weight.initInArenasQ40(
        ctx,
        scales_arena,
        quants_arena,
        t.data,
        in_features,
        out_features,
    );
}

/// Upload the output/lm_head weight. QuantFactory's "Q4_0" Llama-3.1-8B keeps
/// this one projection at Q6_K (the higher-precision logit head ggml emits for
/// Q4_0 ftypes). No Q6_K kernel exists here. For a Q4 model this function
/// dequantizes that single weight to f32 and requantizes to Q4_0. The GPU
/// path then stays uniformly Q4. The lm_head is ~10% of the per-step weight stream, so
/// the precision drop is acceptable for a throughput measurement. An 8B model
/// stays coherent with a Q4_0 head. Any other dtype routes through the normal
/// dispatch.
fn makeOutputW(
    ctx: *const vulkan_ctx.Context,
    gpa: Allocator,
    scales_arena: *kernels.VramArena,
    quants_arena: *kernels.VramArena,
    quant: WeightQuant,
    t: Tensor,
) !kernels.Q80Weight {
    if (t.dtype != .q6_k) return makeMatmulW(ctx, scales_arena, quants_arena, quant, t);

    std.debug.assert(quant == .q4_0); // only the Q4 model ships a Q6_K head here
    const q40_bytes = try convertQ6KtoQ40(gpa, t);
    defer gpa.free(q40_bytes);

    const in_features: usize = @intCast(t.shape[0]);
    const row_bytes: u32 = @intCast((in_features / 32) * kernels.q4_0_block_bytes);
    const t40: Tensor = .{
        .dtype = .q4_0,
        .n_dims = 2,
        .shape = t.shape,
        .byte_stride = .{ row_bytes, 0, 0, 0 },
        .data = q40_bytes,
    };
    t40.assertValid();
    return makeQ40InArenas(ctx, scales_arena, quants_arena, t40);
}

/// Convert a Q6_K tensor to freshly-allocated Q4_0 bytes, streaming one
/// 256-element superblock at a time through a small f32 scratch. That avoids a
/// multi-GB f32 staging copy of the whole lm_head. Caller owns the result.
fn convertQ6KtoQ40(gpa: Allocator, t: Tensor) ![]u8 {
    std.debug.assert(t.dtype == .q6_k);

    const numel: usize = @as(usize, @intCast(t.shape[0])) * @as(usize, @intCast(t.shape[1]));
    std.debug.assert(numel % 256 == 0);
    const n_super = numel / 256;
    const out = try gpa.alloc(u8, (numel / 32) * kernels.q4_0_block_bytes);
    errdefer gpa.free(out);

    var scratch: [256]f32 = undefined;
    var s: usize = 0;
    while (s < n_super) : (s += 1) {
        naive.dequantQ6K(t.data[s * 210 ..][0..210], &scratch);
        // 256 elements = 8 Q4_0 blocks = 8 * 18 = 144 bytes.
        naive.quantizeQ40(&scratch, out[s * 144 ..][0..144]);
    }

    return out;
}

/// Upload one matmul weight using the kernel matching `quant`. Q4 models
/// route to the half-size Q4_0 path. Everything else stays Q8_0.
fn makeMatmulW(
    ctx: *const vulkan_ctx.Context,
    scales_arena: *kernels.VramArena,
    quants_arena: *kernels.VramArena,
    quant: WeightQuant,
    t: Tensor,
) !kernels.Q80Weight {
    return switch (quant) {
        .q8_0 => makeQ80InArenas(ctx, scales_arena, quants_arena, t),
        .q4_0 => makeQ40InArenas(ctx, scales_arena, quants_arena, t),
    };
}

fn bindRmsnorm(
    ctx: *const vulkan_ctx.Context,
    set: vk.DescriptorSet,
    x: vk.Buffer,
    w: vk.DescriptorBufferInfo,
    y: vk.Buffer,
    n_bytes: vk.DeviceSize,
) void {
    const infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = x, .offset = 0, .range = n_bytes },
        w,
        .{ .buffer = y, .offset = 0, .range = n_bytes },
    };
    kernels.updateStorageBindings(ctx, set, &infos);
}

fn bindMatmul(
    ctx: *const vulkan_ctx.Context,
    set: vk.DescriptorSet,
    in_buf: vk.Buffer,
    in_bytes: vk.DeviceSize,
    weight: *const kernels.Q80Weight,
    out_buf: vk.Buffer,
    out_bytes: vk.DeviceSize,
) void {
    const infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = in_buf, .offset = 0, .range = in_bytes },
        weight.scales,
        weight.quants,
        .{ .buffer = out_buf, .offset = 0, .range = out_bytes },
    };
    kernels.updateStorageBindings(ctx, set, &infos);
}

fn bindAttention(
    ctx: *const vulkan_ctx.Context,
    set: vk.DescriptorSet,
    q: vk.Buffer,
    q_bytes: vk.DeviceSize,
    k: vk.DescriptorBufferInfo,
    v: vk.DescriptorBufferInfo,
    out: vk.Buffer,
    out_bytes: vk.DeviceSize,
) void {
    std.debug.assert(k.range == v.range);

    const infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = q, .offset = 0, .range = q_bytes },
        k,
        v,
        .{ .buffer = out, .offset = 0, .range = out_bytes },
    };
    kernels.updateStorageBindings(ctx, set, &infos);
}

// -- command recording ---------------------------------------------------------
fn copyHostToDevice(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    dst: vk.Buffer,
    size: vk.DeviceSize,
) void {
    const regions = [_]vk.BufferCopy{.{ .src_offset = 0, .dst_offset = 0, .size = size }};
    vkd.cmdCopyBuffer(cb, src, dst, &regions);
}

fn copyDeviceToDevice(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    src_off: vk.DeviceSize,
    dst: vk.Buffer,
    dst_off: vk.DeviceSize,
    size: vk.DeviceSize,
) void {
    const regions = [_]vk.BufferCopy{
        .{ .src_offset = src_off, .dst_offset = dst_off, .size = size },
    };
    vkd.cmdCopyBuffer(cb, src, dst, &regions);
}

fn copyDeviceToHost(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    dst: vk.Buffer,
    size: vk.DeviceSize,
) void {
    const regions = [_]vk.BufferCopy{.{ .src_offset = 0, .dst_offset = 0, .size = size }};
    vkd.cmdCopyBuffer(cb, src, dst, &regions);
}

/// Barrier between two SSBO-touching dispatches (shader write -> shader read/write).
fn barrierShader(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mem: vk.MemoryBarrier = .{
        .src_access_mask = .{ .shader_write_bit = true },
        .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true },
    };
    const mems = [_]vk.MemoryBarrier{mem};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .compute_shader_bit = true },
        .{ .compute_shader_bit = true },
        .{},
        &mems,
        null,
        null,
    );
}

/// Barrier between two transfers (cmdCopyBuffer write -> cmdCopyBuffer read),
/// for example the two-hop bounce that compacts a buffer's suffix to its front.
fn barrierTransferToTransfer(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mem: vk.MemoryBarrier = .{
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_access_mask = .{ .transfer_read_bit = true },
    };
    const mems = [_]vk.MemoryBarrier{mem};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .transfer_bit = true },
        .{ .transfer_bit = true },
        .{},
        &mems,
        null,
        null,
    );
}

/// Barrier from a transfer-write (cmdCopyBuffer) to a shader read.
fn barrierTransferToShader(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mem: vk.MemoryBarrier = .{
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true },
    };
    const mems = [_]vk.MemoryBarrier{mem};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .transfer_bit = true },
        .{ .compute_shader_bit = true },
        .{},
        &mems,
        null,
        null,
    );
}

/// Barrier from a shader write to a transfer read (preparing for readback).
fn barrierShaderToTransfer(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mem: vk.MemoryBarrier = .{
        .src_access_mask = .{ .shader_write_bit = true },
        .dst_access_mask = .{ .transfer_read_bit = true },
    };
    const mems = [_]vk.MemoryBarrier{mem};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .compute_shader_bit = true },
        .{ .transfer_bit = true },
        .{},
        &mems,
        null,
        null,
    );
}

// -- tests ------------------------------------------------------------------

// The cost-model anchors below need no Vulkan device.
/// A Llama-3.1-8B-shaped card: 32 layers stepping 8 KV heads x 128 head_dim.
fn testCard() loader.ModelCard {
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

test "llama kv cost: the vector is n_layers equal entries and sums to the model" {
    const card = testCard();
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const vector = kv_cost.llamaCostVector(card, kv_dtype);
        const per_layer: u64 = @as(u64, 2) * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        try std.testing.expectEqual(card.n_layers, vector.n_layers);
        try std.testing.expectEqual(@as(usize, 32), vector.layers().len);
        for (vector.layers()) |entry| {
            try std.testing.expectEqual(per_layer, @as(u64, entry.kv_bytes_per_token));
            // No Llama layer holds recurrent state.
            try std.testing.expectEqual(@as(u32, 0), entry.fixed_bytes_per_slot);
        }

        const whole = vector.total();
        try std.testing.expectEqual(LlamaFusedRunner.kvCostModel(card, kv_dtype), whole);
        try std.testing.expectEqual(@as(u64, card.n_layers) * per_layer, whole.kv_bytes_per_token);
    }

    // The anchors: 262144 bytes a token at f32, and half that at f16.
    const wide_vector = kv_cost.llamaCostVector(card, .f32);
    const half_vector = kv_cost.llamaCostVector(card, .f16);
    const wide = wide_vector.total();
    const half = half_vector.total();
    try std.testing.expectEqual(@as(u64, 262_144), wide.kv_bytes_per_token);
    try std.testing.expectEqual(@as(u64, 131_072), half.kv_bytes_per_token);
}

test "llama scratch: the split attention records follow the declared context" {
    const deep = testCard();
    var shallow = testCard();
    shallow.context_length = limits.attn_split_len;
    const records_deep = LlamaFusedRunner.attnSplitScratch(deep).totalBytes();
    const records_shallow = LlamaFusedRunner.attnSplitScratch(shallow).totalBytes();
    // The records are the only context term, and they are f32 at every dtype.
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const gained = LlamaFusedRunner.scratchVramBytes(
            deep,
            kv_dtype,
        ) - LlamaFusedRunner.scratchVramBytes(
            shallow,
            kv_dtype,
        );
        try std.testing.expectEqual(records_deep - records_shallow, gained);
    }

    // The anchors: 4 rows x 32 heads x 512 splits x 130 elements of 4 bytes,
    // on top of 256 activation rows of 51200 f32 elements.
    try std.testing.expectEqual(@as(u64, 34_078_720), records_deep);
    try std.testing.expectEqual(
        @as(u64, 52_428_800 + 34_078_720),
        LlamaFusedRunner.scratchVramBytes(deep, .f32),
    );
}
