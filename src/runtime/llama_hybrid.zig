//! `LlamaHybridRunner` splits one Llama forward pass across both executors: the CPU
//! runs the layer prefix `[0..split)` over the mmap-backed weights and a host
//! KV cache, and the GPU runs the layer suffix `[split..n_layers)` plus the
//! final norm and the lm_head against uploaded weight views and a suffix-sized
//! KV store. The residual stream crosses the host-device boundary once per
//! call: the prefix leaves its activations in `stage_in`, and the recorded
//! command buffer starts with that one upload. The budget planner picks `split`
//! so the GPU suffix fits the device budget, which funds the middle between a
//! model that fits VRAM and a model that runs CPU-only.
//!
//! This runner records its own GPU layer bodies. They stay dispatch-identical
//! to the fused runner's at `split == 0`, so the cross-runner token gate
//! compares like against like.
//!
//! Prefill is chunk-interleaved: each chunk of at most `prefill_batch_max` rows
//! runs batched through the CPU prefix, crosses once, and runs batched through
//! the GPU suffix. The one alternative, a whole-prompt CPU prefix first, needs
//! prompt-length activation storage, which is unbounded.

const std = @import("std");
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
const kv_index = @import("../core/kv_index.zig");
const attn_split = @import("../core/attn_split.zig");
const attn_record = @import("attn_record.zig");
const llama_runner = @import("llama_runner.zig");
const step_row = @import("step_row.zig");
const llama_fused = @import("llama_fused.zig");
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const core_log = @import("../core/log.zig");
const span = @import("../core/span.zig");
const step_stats = @import("../core/step_stats.zig");
const gpu_profile = @import("gpu_profile.zig");
const decode_ring = @import("decode_ring.zig");
const Allocator = std.mem.Allocator;
const WeightHandles = llama_fused.WeightHandles;
const WeightQuant = llama_fused.WeightQuant;

const log = std.log.scoped(.runner);
const log_dispatch = std.log.scoped(.dispatch);
const log_kv = std.log.scoped(.kv);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/llama_hybrid.zig was included in a non-vulkan build.");
    }
}

// -- types ------------------------------------------------------------------

pub const LlamaHybridRunner = struct {
    gpa: Allocator,
    ctx: *const vulkan_ctx.Context,

    // Model dimensions
    split: u32, // host layers [0..split)
    n_suffix: u32, // GPU layers [split..n_layers)
    n_layers: u32, // model total
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
    /// The element type every KV row is framed at, on both tiers. The prefix
    /// LlamaRunner frames its host half the same way, and `kv.layout` places
    /// every device row.
    kv_dtype: dtype.KvDtype,

    // Host prefix: a k-layer CPU runner over the mmap weights and a host KV
    // sized for exactly the prefix layers. Null when `split == 0`. Its arena
    // is owned here, so deinit reclaims the prefix in one call.
    prefix: ?llama_runner.LlamaRunner,
    prefix_arena: std.heap.ArenaAllocator,

    embed: *const Tensor,

    quant: WeightQuant,
    k_linear: kernels.LinearQ80Reduce,
    k_linear_q4: kernels.LinearQ40Reduce,
    k_rmsnorm: kernels.Rmsnorm,
    k_swiglu: kernels.Swiglu,
    k_residual: kernels.ResidualAdd,
    k_attention: kernels.Attention,
    k_rope_multi: kernels.RopeMulti,
    k_linear_batch: kernels.LinearQ80Batch,
    k_linear_batch_q4: kernels.LinearQ40Batch,
    k_rmsnorm_batch: kernels.RmsnormBatch,
    k_rope_multi_batch: kernels.RopeMultiBatch,
    k_attention_batch: kernels.AttentionBatch,
    k_attention_multi: kernels.AttentionMultiStream,
    /// The split attention passes. A decode row past one split walks its
    /// depth in splits and merges the partial records in split order.
    k_attention_split: kernels.AttentionSplit,
    /// The record box of the split rows in flight, sized at the declared
    /// context, so every row the window admits fits it.
    attn_scratch: attn_split.Scratch,
    /// The commit narrowing kernel, present only at a non-f32 `kv_dtype`. One
    /// dispatch per suffix layer per entry point narrows that layer's whole K
    /// and V commit scratch, whatever the row count.
    k_kv_commit: ?kernels.KvCommitF16,
    /// The one descriptor set the narrowing dispatch binds: the two f32 scratch
    /// buffers as sources, the two narrowed ones as destinations.
    set_kv_commit: vk.DescriptorSet,

    // Scratch device buffers, `prefill_batch_max` rows each, plus the split
    // attention records. They are the exact allocation list
    // `llama_fused.scratchVramBytes` prices, so the plan's scratch reservation
    // equals this init's sum.
    buf_x: kernels.DeviceBuffer,
    buf_x_norm: kernels.DeviceBuffer,
    buf_q: kernels.DeviceBuffer,
    buf_rope_ff: kernels.HostBuffer,
    buf_k_scratch: kernels.DeviceBuffer,
    buf_v_scratch: kernels.DeviceBuffer,
    /// The narrowed commit scratch, allocated only at a non-f32 `kv_dtype`.
    buf_k_scratch_kv: kernels.DeviceBuffer,
    buf_v_scratch_kv: kernels.DeviceBuffer,
    buf_attn_out: kernels.DeviceBuffer,
    buf_attn_split: kernels.DeviceBuffer,
    buf_ffn_gate: kernels.DeviceBuffer,
    buf_ffn_up: kernels.DeviceBuffer,
    buf_ffn_down: kernels.DeviceBuffer,
    buf_logits: kernels.DeviceBuffer,

    arena_small: kernels.VramArena,
    arena_quants: kernels.VramArena,

    // Suffix weight views, index j = model layer `split + j`.
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

    stage_in: kernels.HostBuffer,
    stage_out: kernels.HostBuffer,
    logits_heap: []f32,

    desc_pool: vk.DescriptorPool,
    set_rmsnorm_attn: []vk.DescriptorSet, // [n_suffix]
    set_rmsnorm_ffn: []vk.DescriptorSet,
    set_rmsnorm_final: vk.DescriptorSet,
    set_q: []vk.DescriptorSet,
    set_k: []vk.DescriptorSet,
    set_v: []vk.DescriptorSet,
    set_o: []vk.DescriptorSet,
    set_gate: []vk.DescriptorSet,
    set_up: []vk.DescriptorSet,
    set_down: []vk.DescriptorSet,
    set_lm_head: vk.DescriptorSet,
    set_swiglu: vk.DescriptorSet,
    set_residual_attn: vk.DescriptorSet,
    set_residual_ffn: vk.DescriptorSet,
    set_attention: []vk.DescriptorSet, // [n_suffix]
    set_attention_multi: []vk.DescriptorSet, // [n_suffix]
    set_attention_split: []vk.DescriptorSet, // [n_suffix]
    /// The one merge set: the records and `buf_attn_out`.
    set_attention_merge: vk.DescriptorSet,
    set_rope_q: vk.DescriptorSet,
    set_rope_k: vk.DescriptorSet,

    prefill_batch_max: u32,
    n_slots: u32,
    /// Suffix KV: one region per suffix layer and tensor, each holding
    /// `[slot][max_seq][per_step_elems]` of `kv_dtype`. Suffix layer j holds
    /// model layer `split + j`. It is null when `n_suffix == 0`, where the whole
    /// KV lives in the host prefix.
    kv: ?kernels.KvStore,
    logit_rows_max: u32,
    logits: []f32,

    /// The command buffer of every submit except a decode step.
    cmd_buffer: vk.CommandBuffer,
    /// The timestamps of `cmd_buffer`, at the process level.
    prof: gpu_profile.GpuProfile,
    /// The decode steps of `step`, over two buffers of their own. Every other
    /// submit invalidates it first.
    decode: decode_ring.DecodeRing,
    /// True after init binds the last descriptor set. A recording in the ring
    /// stays valid only while no set changes, so no set changes after init.
    sets_fixed: bool = false,
    /// Host time of the decode host steps of the device suffix, summed since the
    /// last reset. `generate.decode` clears it before a run and reads it after.
    host: step_stats.DecodeHost = .{},

    /// Staging for the device half of a KV swap, sized to one chunk of the
    /// suffix rows a call moves. Lazily allocated on the first swap, so a
    /// bench that never evicts pays nothing.
    kv_swap_stage: kernels.HostBuffer,

    /// Separate staging for the device-to-host copy. Both stages use host-cached
    /// memory when the driver offers it. Two stages and not one, because an
    /// eviction readback must not wait for an upload that still uses
    /// `kv_swap_stage`.
    kv_readback_stage: kernels.HostBuffer,

    /// True when the plan priced both swap stages. A plan without residency
    /// prices zero RAM for them, so this runner must then never grow them.
    swap_priced: bool,

    /// Telemetry only: the DMA and copy split of the last evict or restore, so a bench
    /// aggregates the DMA and the mapped-staging memcpy without parsing logs.
    /// It reports the device half alone, because the host half is a plain
    /// memcpy inside the prefix LlamaRunner.
    last_swap_split: kernels.SwapSplit,

    /// The `run` command batches the prompt through `prefillBatch`, as serve
    /// and bench-model do.
    pub const chunked_prefill: bool = true;

    pub const InitError = error{
        NoSuitableMemoryType,
        VramBudgetExceeded,
        KvRegionPastDeviceLimit,
        AttentionScratchPastDeviceLimit,
        AttentionHeadDimUnsupported,
    } || step_row.Error ||
        kernels.LinearQ80Reduce.InitError ||
        kernels.Q80Weight.InitError ||
        kernels.KvStore.InitError ||
        vk.DeviceWrapper.CreateDescriptorPoolError ||
        vk.DeviceWrapper.AllocateDescriptorSetsError ||
        vk.DeviceWrapper.AllocateCommandBuffersError;

    /// The literal frame of a placement tag, shared by the bound and the
    /// format so the two stay one fact.
    const tag_prefix = "vulkan_hybrid:k=";

    /// Bytes the widest tag occupies: the literal frame, the widest u32 split,
    /// the one separator, the longest `WeightQuant` name, and the widest KV
    /// dtype component.
    pub const backend_tag_buf_len: usize = tag_prefix.len +
        std.fmt.comptimePrint("{d}", .{std.math.maxInt(u32)}).len + 1 +
        dtype.kv_tag_suffix_max + widest: {
        var widest: usize = 0;
        for (@typeInfo(WeightQuant).@"enum".fields) |f| {
            widest = @max(widest, f.name.len);
        }

        break :widest widest;
    };

    /// The snapshot-provenance tag for a hybrid placement. It folds the split,
    /// the weight quant, and the KV dtype, so a blob can never restore across a
    /// different placement or a different storage element. The f32 tag is the
    /// shipped literal byte for byte, because the dtype component names only a
    /// non-identity dtype. A caller must size its buffer at
    /// `backend_tag_buf_len`, which is the bound that makes the format here
    /// infallible. Serve folds this tag into its backend fingerprint at the
    /// hybrid construction arm.
    pub fn backendTag(
        buf: []u8,
        split: u32,
        quant: WeightQuant,
        kv_dtype: dtype.KvDtype,
    ) []const u8 {
        std.debug.assert(buf.len >= backend_tag_buf_len);

        return std.fmt.bufPrint(
            buf,
            tag_prefix ++ "{d}:{t}{s}",
            .{ split, quant, kv_dtype.tagSuffix() },
        ) catch unreachable;
    }

    // -- init ---------------------------------------------------------------

    /// The caller states the KV dtype. Both tiers frame their rows the same
    /// way, so the one value reaches the prefix LlamaRunner and the device suffix
    /// from here.
    pub fn init(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        loaded: *const loader.Loaded,
        plan: kv_budget.PlacementPlan,
        io: std.Io,
        pool: *Pool,
        policy: parallel.Policy,
        kv_dtype: dtype.KvDtype,
    ) InitError!*LlamaHybridRunner {
        const card = loaded.card;
        std.debug.assert(plan.n_layers == card.n_layers);
        std.debug.assert(plan.split <= plan.n_layers);
        std.debug.assert(plan.n_slots >= 1);
        std.debug.assert(plan.n_slots <= limits.max_batch);

        const found = try llama_fused.LlamaFusedRunner.handlesFromLoaded(gpa, loaded);
        defer found.deinit(gpa);

        const weights = found.handles;

        const self = try gpa.create(LlamaHybridRunner);
        errdefer gpa.destroy(self);

        const split: u32 = plan.split;
        const n_layers: u32 = card.n_layers;
        const n_suffix: u32 = n_layers - split;
        const hidden_dim: u32 = @intCast(card.embedding_length);
        const ffn_dim: u32 = @intCast(card.feed_forward_length);
        const vocab_size: u32 = @intCast(card.vocab_size);
        const n_slots: u32 = plan.n_slots;
        const max_seq: u32 = plan.window;
        const n_heads: u32 = card.n_heads;
        const n_kv_heads: u32 = card.n_kv_heads;
        const head_dim: u32 = card.head_dim;
        const per_step_elems: u32 = n_kv_heads * head_dim;
        const logit_rows: u32 = n_slots;
        const f32_size = @sizeOf(f32);

        // Every suffix layer holds one K region and one V region, and the plan
        // priced exactly those bytes. The all-host split holds no device KV.
        // The device limits bind one region, so the guard refuses before the
        // first allocation.
        const kv_layout: ?kv_index.Layout = if (n_suffix > 0)
            kv_index.Layout.init(n_slots, max_seq, per_step_elems, n_suffix, kv_dtype)
        else
            null;

        if (kv_layout) |layout| {
            std.debug.assert(layout.totalBytes() == plan.kv_vram);

            try ctx.guardKvLayout(layout);
        } else {
            std.debug.assert(plan.kv_vram == 0);
        }

        // The plan prices the split attention records at every split, so
        // they are allocated and checked at every split too.
        try kernels.AttentionSplit.guardHeadDim(head_dim);
        const attn_scratch = llama_fused.LlamaFusedRunner.attnSplitScratch(card);
        std.debug.assert(attn_split.splitCount(max_seq) <= attn_scratch.max_splits);
        try ctx.guardAttnSplitScratch(attn_scratch);

        // Plan-is-a-value: every priced component must equal the plan's named
        // field. The planner priced layers by a `blk.{d}.*` name scan
        // (`tensor_lookup.layerWeightBytes`) and this init sums the nine handle
        // tensors. The two agree because a Llama block holds exactly those
        // nine, and the assert below is what makes that family fact
        // load-bearing. The hybrid uploads the head at every split, because the
        // lm_head stays on the GPU. Its resident figure therefore carries the
        // head term even at `split == n_layers`, where only the forced bench
        // gate constructs.
        const stage_one: u64 = swapBytes(kv_dtype, n_suffix, max_seq, per_step_elems);
        {
            var suffix_bytes: u64 = 0;
            for (split..n_layers) |li| {
                suffix_bytes +=
                    weights.attn_norm_w[li].data.len + weights.ffn_norm_w[li].data.len +
                    weights.attn_q_w[li].data.len + weights.attn_k_w[li].data.len +
                    weights.attn_v_w[li].data.len + weights.attn_o_w[li].data.len +
                    weights.ffn_gate_w[li].data.len + weights.ffn_up_w[li].data.len +
                    weights.ffn_down_w[li].data.len;
            }

            const head_bytes: u64 = weights.output_norm.data.len + weights.output.data.len;
            const kv_layer_bytes: u64 = @as(u64, max_seq) * kv_dtype.rowBytes(per_step_elems);
            const kv_one: u64 = @as(u64, 2) * n_suffix * kv_layer_bytes;
            const kv_vram: u64 = @as(u64, n_slots) * kv_one;
            const kv_host: u64 = @as(u64, n_slots) * 2 * split * kv_layer_bytes;
            std.debug.assert(plan.weight_resident == suffix_bytes + head_bytes);
            std.debug.assert(plan.kv_vram == kv_vram);
            std.debug.assert(plan.kv_host == kv_host);
            // A residency plan prices both staging buffers at one chunk of the
            // device half of one slot, or the whole half when that is smaller.
            // A plan without residency prices zero, and this runner then never
            // swaps. The runner allocates no blob and no prefix cache, so it
            // checks neither of those two fields.
            std.debug.assert(plan.ram.host_kv == kv_host);
            std.debug.assert(
                plan.ram.swap_stages == 0 or plan.ram.swap_stages == 2 * kv_budget.stageBytes(
                    stage_one,
                ),
            );
            std.debug.assert(plan.ram.swap_stages <= 2 * limits.swap_stage_chunk_bytes);
            std.debug.assert(plan.ram.swap_stages * n_slots <= 2 * kv_vram);
            std.debug.assert(
                plan.scratch == llama_fused.LlamaFusedRunner.scratchVramBytes(card, kv_dtype),
            );
            std.debug.assert(
                plan.logits >= llama_fused.LlamaFusedRunner.logitsVramBytes(vocab_size, logit_rows),
            );

            // Preflight sums the fields directly: `plan.spend()` reads zero at
            // the degenerate all-host split, and the forced gate still
            // allocates the head, the scratch, and the logits there.
            const mib = 1024 * 1024;
            const planned: u64 = plan.weight_resident + plan.kv_vram + plan.logits + plan.scratch;
            const available: u64 = ctx.deviceLocalAvailableBytes();
            log.debug(
                "vram preflight: planned {d} MiB (weights {d}, kv {d}, scratch {d}), " ++
                    "available {d} MiB",
                .{ planned / mib, plan.weight_resident / mib, kv_vram / mib, plan.scratch / mib, available / mib },
            );
            if (!limits.vramPlanFits(planned, available)) {
                // The fit reads the same live budget and the same reservations
                // as the check above, with the KV of the device suffix alone.
                const reserved: u64 = plan.weight_resident + plan.logits + plan.scratch;
                // The all-host split holds no device KV, so a token costs no
                // device bytes there.
                const cost: kv_budget.KvCostModel = .{
                    .kv_bytes_per_token = if (kv_layout) |layout| layout.tokenBytes() else 0,
                };
                std.debug.assert(cost.kv_bytes_per_token * max_seq == kv_one);
                var refusal_buf: [kv_budget.vram_refusal_text_max]u8 = undefined;
                log.err("{s}", .{kv_budget.formatVramRefusal(&refusal_buf, .{
                    .planned = planned,
                    .available = available,
                    .weights = plan.weight_resident,
                    .kv = kv_vram,
                    .n_slots = n_slots,
                    .window = max_seq,
                    .fit = kv_budget.largestFittingWindow(
                        available,
                        reserved,
                        cost,
                        n_slots,
                        max_seq,
                    ),
                    .suffix = true,
                })});
                return error.VramBudgetExceeded;
            }
        }

        self.* = .{
            .gpa = gpa,
            .ctx = ctx,
            .split = split,
            .n_suffix = n_suffix,
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
            .prefix = null,
            .prefix_arena = std.heap.ArenaAllocator.init(gpa),
            .embed = weights.embed,
            .quant = if (weights.attn_q_w[0].dtype == .q4_0) .q4_0 else .q8_0,
            .k_linear = undefined,
            .k_linear_q4 = undefined,
            .k_rmsnorm = undefined,
            .k_swiglu = undefined,
            .k_residual = undefined,
            .k_attention = undefined,
            .k_rope_multi = undefined,
            .k_linear_batch = undefined,
            .k_linear_batch_q4 = undefined,
            .k_rmsnorm_batch = undefined,
            .k_rope_multi_batch = undefined,
            .k_attention_batch = undefined,
            .k_attention_multi = undefined,
            .k_attention_split = undefined,
            .attn_scratch = attn_scratch,
            .k_kv_commit = null,
            .set_kv_commit = .null_handle,
            .buf_x = .{ .ctx = ctx },
            .buf_x_norm = .{ .ctx = ctx },
            .buf_q = .{ .ctx = ctx },
            .buf_rope_ff = .{ .ctx = ctx },
            .buf_k_scratch = .{ .ctx = ctx },
            .buf_v_scratch = .{ .ctx = ctx },
            .buf_k_scratch_kv = .{ .ctx = ctx },
            .buf_v_scratch_kv = .{ .ctx = ctx },
            .buf_attn_out = .{ .ctx = ctx },
            .buf_attn_split = .{ .ctx = ctx },
            .buf_ffn_gate = .{ .ctx = ctx },
            .buf_ffn_up = .{ .ctx = ctx },
            .buf_ffn_down = .{ .ctx = ctx },
            .buf_logits = .{ .ctx = ctx },
            .kv = null, // set below, after the scratch buffers
            .arena_small = undefined,
            .arena_quants = undefined,
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
            .set_attention_multi = undefined,
            .set_attention_split = undefined,
            .set_attention_merge = .null_handle,
            .set_rope_q = .null_handle,
            .set_rope_k = .null_handle,
            .prefill_batch_max = llama_fused.LlamaFusedRunner.prefill_rows_max,
            .n_slots = n_slots,
            .logit_rows_max = logit_rows,
            .logits = undefined,
            .cmd_buffer = .null_handle,
            .decode = undefined, // set below, after the descriptor sets
            .prof = undefined, // set below, with the decode ring
            .kv_swap_stage = .{ .ctx = ctx },
            .kv_readback_stage = .{ .ctx = ctx },
            .swap_priced = plan.ram.swap_stages == 2 * kv_budget.stageBytes(stage_one),
            .last_swap_split = .{ .dma_ms = 0, .memcpy_ms = 0 },
        };
        errdefer self.prefix_arena.deinit();

        if (split > 0) {
            // The host prefix frames its KV the way the device suffix does, so
            // one dtype crosses the boundary and the blob keeps one row stride.
            self.prefix = try llama_runner.LlamaRunner.initPrefix(
                self.prefix_arena.allocator(),
                loaded,
                kv_budget.pinned(n_slots, max_seq),
                split,
                self.prefill_batch_max,
                io,
                pool,
                policy,
                kv_dtype,
            );
        }

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
        try self.buf_logits.ensure(
            @as(vk.DeviceSize, self.logit_rows_max) * @as(vk.DeviceSize, vocab_size) * f32_size,
        );

        if (kv_layout) |layout| {
            // The commit copies move scratch rows of `kvRowBytes()` into rows
            // of the layout.
            std.debug.assert(layout.rowBytes() == self.kvRowBytes());

            self.kv = try kernels.KvStore.init(ctx, layout);
        }

        errdefer if (self.kv) |*kv| kv.deinit();

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
        // storage element. At f32 the selector picks the shipped pipeline.
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

        const arena_usage: vk.BufferUsageFlags = .{
            .storage_buffer_bit = true,
            .transfer_dst_bit = true,
        };
        self.arena_small = try kernels.VramArena.init(ctx, gpa, arena_usage, 256 * 1024 * 1024);
        errdefer self.arena_small.deinit();

        self.arena_quants = try kernels.VramArena.init(ctx, gpa, arena_usage, 1024 * 1024 * 1024);
        errdefer self.arena_quants.deinit();

        self.logits = try gpa.alloc(f32, @as(usize, self.logit_rows_max) * vocab_size);

        self.attn_norm_w = try gpa.alloc(vk.DescriptorBufferInfo, n_suffix);
        self.ffn_norm_w = try gpa.alloc(vk.DescriptorBufferInfo, n_suffix);
        self.attn_q_w = try gpa.alloc(kernels.Q80Weight, n_suffix);
        self.attn_k_w = try gpa.alloc(kernels.Q80Weight, n_suffix);
        self.attn_v_w = try gpa.alloc(kernels.Q80Weight, n_suffix);
        self.attn_o_w = try gpa.alloc(kernels.Q80Weight, n_suffix);
        self.ffn_gate_w = try gpa.alloc(kernels.Q80Weight, n_suffix);
        self.ffn_up_w = try gpa.alloc(kernels.Q80Weight, n_suffix);
        self.ffn_down_w = try gpa.alloc(kernels.Q80Weight, n_suffix);

        var j: u32 = 0;
        while (j < n_suffix) : (j += 1) {
            const li = split + j;
            self.attn_norm_w[j] = try uploadFp32NormInArena(
                ctx,
                weights.attn_norm_w[li].*,
                &self.arena_small,
            );
            self.ffn_norm_w[j] = try uploadFp32NormInArena(
                ctx,
                weights.ffn_norm_w[li].*,
                &self.arena_small,
            );
            self.attn_q_w[j] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.attn_q_w[li].*,
            );
            self.attn_k_w[j] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.attn_k_w[li].*,
            );
            self.attn_v_w[j] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.attn_v_w[li].*,
            );
            self.attn_o_w[j] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.attn_o_w[li].*,
            );
            self.ffn_gate_w[j] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.ffn_gate_w[li].*,
            );
            self.ffn_up_w[j] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.ffn_up_w[li].*,
            );
            self.ffn_down_w[j] = try makeMatmulW(
                ctx,
                &self.arena_small,
                &self.arena_quants,
                self.quant,
                weights.ffn_down_w[li].*,
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

        const ns = n_suffix;
        // The fused runner's list, over the suffix layers: matmul, rmsnorm,
        // attention, multi-stream attention, split attention, the one merge
        // set, residual, swiglu, rope_multi, and kv commit. The kv-commit set
        // exists at a non-f32 KV dtype only.
        const kv_commit_sets: u32 = if (kv_dtype == .f32) 0 else 1;
        const split_bindings = kernels.AttentionSplit.split_bindings;
        const merge_bindings = kernels.AttentionSplit.merge_bindings;
        const total_sets: u32 = (ns * 7 + 1) + (ns * 2 + 1) + ns + ns + ns + 1 + 2 + 1 + 2 +
            kv_commit_sets;
        const total_storage: u32 = (ns * 7 + 1) * 4 + (ns * 2 + 1) * 3 + ns * 4 + ns * 4 +
            ns * split_bindings + merge_bindings + 2 * 2 + 3 + 2 * 2 + kv_commit_sets * 4;
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

        self.set_rmsnorm_attn = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_rmsnorm_ffn = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_q = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_k = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_v = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_o = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_gate = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_up = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_down = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_attention = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_attention_multi = try gpa.alloc(vk.DescriptorSet, ns);
        self.set_attention_split = try gpa.alloc(vk.DescriptorSet, ns);

        try self.allocateAndBindDescriptorSets();

        self.cmd_buffer = try kernels.allocateOnePrimaryCmdBuffer(ctx);
        errdefer ctx.vkd.freeCommandBuffers(
            ctx.device,
            ctx.command_pool,
            &[_]vk.CommandBuffer{self.cmd_buffer},
        );

        self.decode = try decode_ring.DecodeRing.init(ctx, .llama_hybrid);
        errdefer self.decode.deinit();

        self.prof = try gpu_profile.GpuProfile.init(ctx, span.gpuLevel(), span.gpuLevel());
        errdefer self.prof.deinit();

        self.sets_fixed = true;

        const mibf: f64 = 1024.0 * 1024.0;
        const weights_mib: f64 = @as(
            f64,
            @floatFromInt(self.arena_quants.totalUsed() + self.arena_small.totalUsed()),
        ) / mibf;
        log.debug(
            "init llama hybrid: split {d} of {d} layers ({d} on gpu), {d} slots, max_seq " ++
                "{d}, quant {t}, gpu weights {d:.0} MiB",
            .{ split, n_layers, n_suffix, n_slots, max_seq, self.quant, weights_mib },
        );
        self.arena_small.logSegments("small");
        self.arena_quants.logSegments("quants");

        return self;
    }

    pub fn deinit(self: *LlamaHybridRunner) void {
        const ctx = self.ctx;
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

        self.output_w.deinit();
        var j = self.n_suffix;
        while (j > 0) {
            j -= 1;
            self.ffn_down_w[j].deinit();
            self.ffn_up_w[j].deinit();
            self.ffn_gate_w[j].deinit();
            self.attn_o_w[j].deinit();
            self.attn_v_w[j].deinit();
            self.attn_k_w[j].deinit();
            self.attn_q_w[j].deinit();
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
        if (self.kv) |*kv| {
            kv.deinit();
        }

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

        self.prefix_arena.deinit();
        self.gpa.destroy(self);
    }

    pub fn logitRowsMax(self: *const LlamaHybridRunner) u32 {
        return self.logit_rows_max;
    }

    pub fn maxStepRows(self: *const LlamaHybridRunner) u32 {
        return self.prefill_batch_max;
    }

    // -- kv framing ---------------------------------------------------------

    /// Bytes one KV row occupies at `kv_dtype`. It sizes the commit scratch and
    /// every commit copy, and init asserts that it equals the row of
    /// `kv.layout`.
    fn kvRowBytes(self: *const LlamaHybridRunner) vk.DeviceSize {
        return self.kv_dtype.rowBytes(self.per_step_elems);
    }

    /// The device KV store. Only a runner with a device suffix holds one, so
    /// the callers reach it inside a suffix-layer loop or behind a suffix
    /// check.
    fn deviceKv(self: *const LlamaHybridRunner) *const kernels.KvStore {
        std.debug.assert(self.n_suffix > 0);
        std.debug.assert(self.kv != null);

        return &self.kv.?;
    }

    /// The two buffers a commit copy list reads. At f32 they are the
    /// rope scratch itself, and the whole path is the shipped one. At a
    /// narrower dtype they are the narrowed scratch `recordKvCommit` just
    /// wrote. Either way one row is `kvRowBytes()`, so every offset and every
    /// region size in the copy lists holds unchanged.
    fn kvCommitSrc(self: *const LlamaHybridRunner) struct { k: vk.Buffer, v: vk.Buffer } {
        if (self.kv_dtype == .f32) {
            return .{ .k = self.buf_k_scratch.buffer, .v = self.buf_v_scratch.buffer };
        }

        return .{ .k = self.buf_k_scratch_kv.buffer, .v = self.buf_v_scratch_kv.buffer };
    }

    /// Record the narrowing of one suffix layer's `rows` committed rows, plus
    /// the barrier that hands the result to the copy list that follows. One
    /// dispatch covers the whole commit whatever the row count, because the
    /// scratch is contiguous. At f32 no dispatch runs and the caller's copy
    /// list reads the rope scratch directly. Both dtypes need the barrier,
    /// because the copy reads in the transfer stage what a shader just wrote.
    fn recordKvCommit(self: *LlamaHybridRunner, cb: vk.CommandBuffer, rows: u32) void {
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

    // -- slot state ---------------------------------------------------------

    /// Fork slot state for a session clone across both tiers. The prefix
    /// LlamaRunner forks its host layers, and one device-to-device copy per
    /// suffix region forks the suffix slot. Rows past `pos` copy stale bytes
    /// that attention never reads, which is the same slot-reuse argument as
    /// `Sessions.destroy`. One slot id spans both tiers, because the prefix
    /// carries the same slot count.
    /// `Sessions.clone` calls this. The presence of this method is what makes
    /// `Sessions(LlamaHybridRunner).can_clone` true, so a swallowed failure silently
    /// corrupts the fork. It panics instead.
    pub fn cloneSlotState(
        self: *LlamaHybridRunner,
        source_slot: u32,
        target_slot: u32,
        pos: u32,
    ) void {
        std.debug.assert(source_slot < self.n_slots);
        std.debug.assert(target_slot < self.n_slots);
        std.debug.assert(source_slot != target_slot);
        std.debug.assert(pos <= self.max_seq);

        if (self.prefix) |*p| {
            p.cloneSlotState(source_slot, target_slot, pos);
        }

        if (self.n_suffix == 0) return;

        const kv = self.deviceKv();
        const stride: vk.DeviceSize = kv.layout.slotBytes();
        log_kv.debug(
            "clone slot {d} -> {d}, pos {d}, {d:.0} MiB suffix",
            .{ source_slot, target_slot, pos, @as(f64, @floatFromInt(stride)) / (1024.0 * 1024.0) },
        );
        var scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
        const copies = kv.cloneCopies(source_slot, target_slot, &scratch);
        self.invalidateDecode();
        kernels.runStagingCopies(self.ctx, copies) catch |e|
            std.debug.panic("hybrid gpu slot clone failed: {t}", .{e});
    }

    // -- tiered kv ----------------------------------------------------------
    // Evict a slot to a host blob, and restore it back. The serve residency
    // layer (`serve/residency.zig`) spills idle sessions to RAM. A hybrid slot
    // lives in two tiers, so one blob holds the host prefix layers first and
    // the device suffix layers after. That is the whole-model order (ascending
    // layers, K rows then V rows per layer), so a blob holds the same bytes a
    // single-tier runner writes for the same slot. Attention reads only
    // positions below `pos`, so a prefix of `n_rows` rows is the whole live
    // state.

    /// Bytes one slot's first `n_rows` KV positions occupy in a host blob. The
    /// figure covers the whole model, so it does not move with the split.
    pub fn evictSlotBytesFor(self: *const LlamaHybridRunner, n_rows: u32) u64 {
        return swapBytes(self.kv_dtype, self.n_layers, n_rows, self.per_step_elems);
    }

    /// Bytes the host prefix layers take in that blob. The device half starts
    /// at exactly this offset.
    fn hostSwapBytes(self: *const LlamaHybridRunner, n_rows: u32) usize {
        return @intCast(swapBytes(self.kv_dtype, self.split, n_rows, self.per_step_elems));
    }

    /// Bytes the device suffix layers take in that blob.
    fn deviceSwapBytes(self: *const LlamaHybridRunner, n_rows: u32) usize {
        return @intCast(swapBytes(self.kv_dtype, self.n_suffix, n_rows, self.per_step_elems));
    }

    /// Build the device<->host copy list for the suffix half of `slot`'s first
    /// `n_rows` KV positions, in the blob order of `KvStore.rowCopies`.
    /// `to_host` true = save (D2H into the readback stage), false = restore
    /// (H2D out of the swap stage).
    fn slotCopies(
        self: *LlamaHybridRunner,
        slot: u32,
        n_rows: u32,
        to_host: bool,
        out: []kernels.Copy,
    ) []kernels.Copy {
        const stage = if (to_host) self.kv_readback_stage.buffer else self.kv_swap_stage.buffer;
        const copies = self.deviceKv().rowCopies(slot, n_rows, stage, to_host, out);
        std.debug.assert(copies.len == 2 * @as(usize, self.n_suffix));
        // The copies fill exactly the device half of the blob.
        std.debug.assert(self.deviceKv().layout.blobBytes(n_rows) == self.deviceSwapBytes(n_rows));
        return copies;
    }

    /// Copy the first `n_rows` KV positions of `slot` into `out`: the prefix
    /// LlamaRunner copies the host half, then a D2H staging pass appends the suffix
    /// half.
    pub fn evictSlot(self: *LlamaHybridRunner, slot: u32, n_rows: u32, out: []u8) !void {
        // The stage this call grows is RAM the plan must have priced.
        std.debug.assert(self.swap_priced);
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);

        const bytes: usize = @intCast(self.evictSlotBytesFor(n_rows));
        std.debug.assert(out.len == bytes);
        const host_bytes = self.hostSwapBytes(n_rows);
        const device_bytes = self.deviceSwapBytes(n_rows);
        std.debug.assert(host_bytes + device_bytes == bytes);
        const t0 = core_log.now();
        if (self.prefix) |*p| {
            try p.evictSlot(slot, n_rows, out[0..host_bytes]);
        }

        var dma_ms: f64 = 0;
        var memcpy_ms: f64 = 0;
        if (self.n_suffix > 0) {
            try self.kv_readback_stage.ensure(kv_budget.stageBytes(device_bytes));
            kv_budget.assertStageCap(
                self.kv_readback_stage.capacity,
                limits.swap_stage_chunk_bytes,
                self.deviceSwapBytes(self.max_seq),
            );
            var scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
            const copies = self.slotCopies(slot, n_rows, true, &scratch);
            self.invalidateDecode();
            const split = try kernels.walkStage(
                self.ctx,
                copies,
                &self.kv_readback_stage,
                .{ .to_host = out[host_bytes..] },
                limits.swap_stage_chunk_bytes,
            );
            dma_ms = split.dma_ms;
            memcpy_ms = split.memcpy_ms;
        }

        self.last_swap_split = .{ .dma_ms = dma_ms, .memcpy_ms = memcpy_ms };
        log_kv.debug(
            "evict slot {d}: {d} rows, {d:.0} MiB ({d} host + {d} gpu layers), {d:.0} ms " ++
                "({d:.1} dma + {d:.1} memcpy)",
            .{ slot, n_rows, @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), self.split, self.n_suffix, core_log.msSince(t0), dma_ms, memcpy_ms },
        );
    }

    /// Restore a previously-evicted `n_rows`-row blob into `slot`, the mirror
    /// of `evictSlot`: the host half goes back through the prefix LlamaRunner, and
    /// the suffix half goes through the stage into VRAM.
    pub fn restoreSlot(self: *LlamaHybridRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        // The stage this call grows is RAM the plan must have priced.
        std.debug.assert(self.swap_priced);
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);

        const bytes: usize = @intCast(self.evictSlotBytesFor(n_rows));
        std.debug.assert(blob.len == bytes);
        const host_bytes = self.hostSwapBytes(n_rows);
        const device_bytes = self.deviceSwapBytes(n_rows);
        std.debug.assert(host_bytes + device_bytes == bytes);
        const t0 = core_log.now();
        if (self.prefix) |*p| {
            try p.restoreSlot(slot, n_rows, blob[0..host_bytes]);
        }

        var dma_ms: f64 = 0;
        var memcpy_ms: f64 = 0;
        if (self.n_suffix > 0) {
            try self.kv_swap_stage.ensure(kv_budget.stageBytes(device_bytes));
            kv_budget.assertStageCap(
                self.kv_swap_stage.capacity,
                limits.swap_stage_chunk_bytes,
                self.deviceSwapBytes(self.max_seq),
            );
            var scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
            const copies = self.slotCopies(slot, n_rows, false, &scratch);
            self.invalidateDecode();
            const split = try kernels.walkStage(
                self.ctx,
                copies,
                &self.kv_swap_stage,
                .{ .to_device = blob[host_bytes..] },
                limits.swap_stage_chunk_bytes,
            );
            dma_ms = split.dma_ms;
            memcpy_ms = split.memcpy_ms;
        }

        self.last_swap_split = .{ .dma_ms = dma_ms, .memcpy_ms = memcpy_ms };
        log_kv.debug(
            "restore slot {d}: {d} rows, {d:.0} MiB ({d} host + {d} gpu layers), {d:.0} ms " ++
                "({d:.1} dma + {d:.1} memcpy)",
            .{ slot, n_rows, @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), self.split, self.n_suffix, core_log.msSince(t0), dma_ms, memcpy_ms },
        );
    }

    // -- step ---------------------------------------------------------------

    /// Run the CPU prefix over `rows` and leave the crossing activations in
    /// `stage_in`: embed into the prefix's residual stream, run layers
    /// `[0..split)`, then copy the stream once into the mapped staging
    /// buffer. At `split == 0` the embeddings go straight to `stage_in`,
    /// which is exactly the fused runner's entry.
    fn hostPrefix(self: *LlamaHybridRunner, rows: []const step_row.StepRow) step_row.Error!void {
        const sp = span.beginHot(.host_prefix);
        defer sp.end();

        const d = self.hidden_dim;
        const x_host = self.stage_in.sliceTyped(f32, rows.len * d);
        if (self.prefix) |*p| {
            for (rows, 0..) |r, i| {
                naive.embedLookup(self.embed.*, r.token, d, p.x[i * d ..][0..d]);
            }

            var l: u32 = 0;
            while (l < self.split) : (l += 1) {
                p.layerForward(l, rows);
            }

            @memcpy(x_host, p.x[0 .. rows.len * d]);
        } else {
            for (rows, 0..) |r, i| {
                naive.embedLookup(self.embed.*, r.token, d, x_host[i * d ..][0..d]);
            }
        }
    }

    /// One decode step on slot 0. Returns the heap logits slice, valid until
    /// the next call.
    pub fn step(self: *LlamaHybridRunner, token_id: u32, pos: u32) ![]f32 {
        if (pos >= self.max_seq) return error.PositionOutOfRange;

        log.debug("step: token {d} at pos {d} (split {d})", .{ token_id, pos, self.split });
        const row = [_]step_row.StepRow{
            .{ .slot = 0, .token = token_id, .pos = pos, .want_logits = true },
        };
        try self.hostPrefix(&row);

        // The record ring records the suffix on a miss and the next step while
        // the GPU runs. The host prefix of the next step needs its token, so
        // only the record overlaps.
        std.debug.assert(self.sets_fixed);
        const key = gpu_profile.decodeKey(pos, self.decode.level());
        try self.decode.step(key, key.next(self.max_seq), DecodeRec{ .runner = self }, &self.host);

        const t0 = core_log.now();
        @memcpy(self.logits_heap, self.stage_out.sliceTyped(f32, self.vocab_size));
        self.host.copy.add(step_stats.nsBetween(t0, core_log.now()));
        return self.logits_heap;
    }

    /// The record callback of the decode ring.
    const DecodeRec = struct {
        runner: *LlamaHybridRunner,

        pub fn recordDecode(
            self: DecodeRec,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            key: gpu_profile.DecodeKey,
        ) !void {
            std.debug.assert(key.opts.one_time);

            return self.runner.recordDecodeStep(cb, prof, key.pos);
        }
    };

    /// Free the decode ring before a submit that is not a decode step.
    fn invalidateDecode(self: *LlamaHybridRunner) void {
        self.decode.invalidate(&self.host);
    }

    /// Advance a batch of rows, the GPU sibling of `LlamaRunner.stepBatch`: the same
    /// row rules, and compact logits for the `want_logits` suffix. The CPU
    /// prefix runs the whole batch before the one crossing, so KV
    /// write-before-attend holds on both sides.
    pub fn stepBatch(self: *LlamaHybridRunner, rows: []const step_row.StepRow) !void {
        const n_logits = try step_row.validateRows(
            rows,
            self.n_slots,
            self.prefill_batch_max,
            self.max_seq,
        );
        std.debug.assert(n_logits <= self.logit_rows_max);
        const n: u32 = @intCast(rows.len);
        log.debug(
            "step_batch: {d} rows, slot {d} pos {d}..{d} (split {d})",
            .{ n, rows[0].slot, rows[0].pos, rows[n - 1].pos, self.split },
        );
        try self.hostPrefix(rows);

        try self.recordStepBatch(rows);
        try self.submitWait(.decode_batch);

        const n_floats = @as(usize, n_logits) * self.vocab_size;
        @memcpy(self.logits[0..n_floats], self.stage_out.sliceTyped(f32, n_floats));
    }

    /// Process `b` prompt tokens in one chunk-interleaved pass on slot 0:
    /// batched CPU prefix, one upload, batched GPU suffix. Returns logits for
    /// the last token only. Callers chunk longer prompts.
    pub fn prefillBatch(self: *LlamaHybridRunner, tokens: []const u32, start_pos: u32) ![]f32 {
        std.debug.assert(tokens.len > 0);
        std.debug.assert(tokens.len <= self.prefill_batch_max);

        const b: u32 = @intCast(tokens.len);
        log.debug(
            "prefill: {d} tokens at start_pos {d} (split {d})",
            .{ b, start_pos, self.split },
        );

        // `init` sets `prefill_batch_max` to this row bound, so the bound must fit the stack array.
        comptime std.debug.assert(
            llama_fused.LlamaFusedRunner.prefill_rows_max <= limits.max_prefill_chunk,
        );

        var rows_buf: [limits.max_prefill_chunk]step_row.StepRow = undefined;
        for (tokens, 0..) |t, i| {
            rows_buf[i] = .{
                .slot = 0,
                .token = t,
                .pos = start_pos + @as(u32, @intCast(i)),
                .want_logits = i == tokens.len - 1,
            };
        }

        try self.hostPrefix(rows_buf[0..b]);

        try self.recordPrefillStep(b, start_pos);
        try self.submitWait(.prefill_chunk);

        @memcpy(self.logits_heap, self.stage_out.sliceTyped(f32, self.vocab_size));
        return self.logits_heap;
    }

    /// Submit `cmd_buffer`, wait for the queue, and read its timestamps back.
    /// The host times exist only when the buffer wrote timestamps or the span
    /// recorder runs, so the off path reads no clock.
    fn submitWait(self: *LlamaHybridRunner, label: span.GpuLabel) !void {
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
        _ = try self.prof.collectInto(.llama_hybrid, label, t0, t2);
    }

    fn recordDecodeMatmul(
        self: *LlamaHybridRunner,
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

    fn recordBatchMatmul(
        self: *LlamaHybridRunner,
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
    fn attnKit(self: *LlamaHybridRunner) kernels.AttnKit {
        return .{
            .single = &self.k_attention,
            .multi = &self.k_attention_multi,
            .passes = &self.k_attention_split,
            .batch = &self.k_attention_batch,
        };
    }

    fn attnShape(self: *const LlamaHybridRunner, stream_stride_elems: u32) kernels.AttnShape {
        return .{
            .n_heads = self.n_heads,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
            .stream_stride_elems = stream_stride_elems,
            .max_splits = self.attn_scratch.max_splits,
        };
    }

    fn recordDecodeStep(
        self: *LlamaHybridRunner,
        cb: vk.CommandBuffer,
        prof: *gpu_profile.GpuProfile,
        pos: u32,
    ) !void {
        const vkd = self.ctx.vkd;
        try vkd.resetCommandBuffer(cb, .{});
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try vkd.beginCommandBuffer(cb, &begin);
        prof.begin(cb);

        const hidden_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.hidden_dim) * @sizeOf(f32);
        const kv_row_bytes: vk.DeviceSize = self.kvRowBytes();

        copyHostToDevice(vkd, cb, self.stage_in.buffer, self.buf_x.buffer, hidden_bytes);
        barrierTransferToShader(vkd, cb);
        prof.mark(cb, .copy);

        // The plan reads no device state. The shape reads the device KV, which a
        // split with no GPU layer lacks, so only a suffix layer reads the shape.
        const attn_plan = attn_split.planDecode(pos + 1);
        var j: u32 = 0;
        while (j < self.n_suffix) : (j += 1) {
            if (comptime core_log.hot) {
                log_dispatch.debug("record suffix layer {d} (model {d})", .{ j, self.split + j });
            }

            self.k_rmsnorm.recordInto(
                cb,
                self.set_rmsnorm_attn[j],
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .norm);

            self.recordDecodeMatmul(
                cb,
                self.set_q[j],
                self.hidden_dim,
                self.n_heads * self.head_dim,
                "q",
            );
            self.recordDecodeMatmul(cb, self.set_k[j], self.hidden_dim, self.per_step_elems, "k");
            self.recordDecodeMatmul(cb, self.set_v[j], self.hidden_dim, self.per_step_elems, "v");
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_attn_in);

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

            const src = self.kvCommitSrc();
            const k_row = self.deviceKv().rowAt(j, .k, 0, pos);
            const v_row = self.deviceKv().rowAt(j, .v, 0, pos);
            copyDeviceToDevice(vkd, cb, src.k, 0, k_row.buffer, k_row.offset, kv_row_bytes);
            copyDeviceToDevice(vkd, cb, src.v, 0, v_row.buffer, v_row.offset, kv_row_bytes);
            barrierTransferToShader(vkd, cb);
            prof.mark(cb, .copy);

            // A row past one split reads slot 0 through the whole-region set,
            // so it takes the same split shader as a batch row.
            attn_record.record(kernels.AttnKit, self.attnKit(), cb, &attn_plan, .{
                .shape = self.attnShape(self.deviceKv().layout.streamStrideElems()),
                .resident = .{ .single = self.set_attention[j] },
                .split = self.set_attention_split[j],
                .merge = self.set_attention_merge,
            });
            barrierShader(vkd, cb);
            prof.mark(cb, .attn);

            self.recordDecodeMatmul(
                cb,
                self.set_o[j],
                self.n_heads * self.head_dim,
                self.hidden_dim,
                "o",
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_out);

            self.k_residual.recordInto(cb, self.set_residual_attn, self.hidden_dim);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);

            self.k_rmsnorm.recordInto(
                cb,
                self.set_rmsnorm_ffn[j],
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            prof.mark(cb, .norm);

            self.recordDecodeMatmul(cb, self.set_gate[j], self.hidden_dim, self.ffn_dim, "gate");
            self.recordDecodeMatmul(cb, self.set_up[j], self.hidden_dim, self.ffn_dim, "up");
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_ffn_up);

            self.k_swiglu.recordInto(cb, self.set_swiglu, self.ffn_dim);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);

            self.recordDecodeMatmul(cb, self.set_down[j], self.ffn_dim, self.hidden_dim, "down");
            barrierShader(vkd, cb);
            prof.mark(cb, .matmul_ffn_down);

            self.k_residual.recordInto(cb, self.set_residual_ffn, self.hidden_dim);
            barrierShader(vkd, cb);
            prof.mark(cb, .elementwise);
        }

        self.k_rmsnorm.recordInto(cb, self.set_rmsnorm_final, self.hidden_dim, self.rms_norm_eps);
        barrierShader(vkd, cb);
        prof.mark(cb, .norm);
        self.recordDecodeMatmul(cb, self.set_lm_head, self.hidden_dim, self.vocab_size, "lmhead");
        barrierShaderToTransfer(vkd, cb);
        prof.mark(cb, .lmhead);

        const logits_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.vocab_size) * @sizeOf(f32);
        copyDeviceToHost(vkd, cb, self.buf_logits.buffer, self.stage_out.buffer, logits_bytes);
        prof.mark(cb, .copy);

        prof.end(cb);
        try vkd.endCommandBuffer(cb);
    }

    fn recordStepBatch(self: *LlamaHybridRunner, rows: []const step_row.StepRow) !void {
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
        // Every suffix layer records the same attention plan.
        std.debug.assert(n <= self.prefill_batch_max);
        const attn_plan = attn_record.planStep(rows, self.attn_scratch.max_splits);

        copyHostToDevice(vkd, cb, self.stage_in.buffer, self.buf_x.buffer, total_x_bytes);
        barrierTransferToShader(vkd, cb);
        self.prof.mark(cb, .copy);

        var j: u32 = 0;
        while (j < self.n_suffix) : (j += 1) {
            if (comptime core_log.hot) {
                log_dispatch.debug("record suffix layer {d} (model {d})", .{ j, self.split + j });
            }

            self.k_rmsnorm_batch.recordInto(
                cb,
                self.set_rmsnorm_attn[j],
                n,
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);

            self.recordBatchMatmul(
                cb,
                self.set_q[j],
                n,
                self.hidden_dim,
                self.n_heads * self.head_dim,
                "q",
            );
            self.recordBatchMatmul(cb, self.set_k[j], n, self.hidden_dim, self.per_step_elems, "k");
            self.recordBatchMatmul(cb, self.set_v[j], n, self.hidden_dim, self.per_step_elems, "v");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_attn_in);

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

            // One copy region per row into its slot's rows at its own
            // position. Each region lives in one buffer, so each tensor takes
            // one copy command.
            const src = self.kvCommitSrc();
            const kv = self.deviceKv();
            const k_dst = kv.region(j, .k).buffer;
            const v_dst = kv.region(j, .v).buffer;
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
                const k_row = kv.rowAt(j, .k, r.slot, r.pos);
                const v_row = kv.rowAt(j, .v, r.slot, r.pos);
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

            // The attention_multi shader reads `k_buf[slot * stream_stride +
            // offset in the slot]`, and each descriptor binds one whole region.
            attn_record.record(kernels.AttnKit, self.attnKit(), cb, &attn_plan, .{
                .shape = self.attnShape(self.deviceKv().layout.streamStrideElems()),
                .resident = .{ .multi = self.set_attention_multi[j] },
                .split = self.set_attention_split[j],
                .merge = self.set_attention_merge,
            });
            barrierShader(vkd, cb);
            self.prof.mark(cb, .attn);

            self.recordBatchMatmul(
                cb,
                self.set_o[j],
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
                self.set_rmsnorm_ffn[j],
                n,
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);
            self.recordBatchMatmul(cb, self.set_gate[j], n, self.hidden_dim, self.ffn_dim, "gate");
            self.recordBatchMatmul(cb, self.set_up[j], n, self.hidden_dim, self.ffn_dim, "up");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_up);
            self.k_swiglu.recordInto(cb, self.set_swiglu, n * self.ffn_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
            self.recordBatchMatmul(cb, self.set_down[j], n, self.ffn_dim, self.hidden_dim, "down");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_down);
            self.k_residual.recordInto(cb, self.set_residual_ffn, n * self.hidden_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
        }

        // A chunk with no want-logits row ends here: a zero-size lm_head or
        // copy must not reach the command buffer.
        const first = step_row.wantLogitsFirst(rows);
        if (first == n) {
            self.prof.end(cb);
            try vkd.endCommandBuffer(cb);
            return;
        }

        self.k_rmsnorm_batch.recordInto(
            cb,
            self.set_rmsnorm_final,
            n,
            self.hidden_dim,
            self.rms_norm_eps,
        );

        const n_logits: u32 = n - first;
        if (first > 0) {
            // Compact the want-logits suffix to the front through buf_x, free
            // after the final norm, because cmdCopyBuffer forbids overlap.
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

    fn recordPrefillStep(self: *LlamaHybridRunner, b: u32, start_pos: u32) !void {
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
        // Every suffix layer records the same plan.
        std.debug.assert(b <= attn_split.max_plan_rows);
        const prefill_plan = attn_split.planPrefill(0, start_pos, b, self.attn_scratch.max_splits);

        copyHostToDevice(vkd, cb, self.stage_in.buffer, self.buf_x.buffer, total_x_bytes);
        barrierTransferToShader(vkd, cb);
        self.prof.mark(cb, .copy);

        var j: u32 = 0;
        while (j < self.n_suffix) : (j += 1) {
            if (comptime core_log.hot) {
                log_dispatch.debug("record suffix layer {d} (model {d})", .{ j, self.split + j });
            }

            self.k_rmsnorm_batch.recordInto(
                cb,
                self.set_rmsnorm_attn[j],
                b,
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);

            self.recordBatchMatmul(
                cb,
                self.set_q[j],
                b,
                self.hidden_dim,
                self.n_heads * self.head_dim,
                "q",
            );
            self.recordBatchMatmul(cb, self.set_k[j], b, self.hidden_dim, self.per_step_elems, "k");
            self.recordBatchMatmul(cb, self.set_v[j], b, self.hidden_dim, self.per_step_elems, "v");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_attn_in);

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

            const src = self.kvCommitSrc();
            const k_row = self.deviceKv().rowAt(j, .k, 0, start_pos);
            const v_row = self.deviceKv().rowAt(j, .v, 0, start_pos);
            copyDeviceToDevice(vkd, cb, src.k, 0, k_row.buffer, k_row.offset, total_kv_rows_bytes);
            copyDeviceToDevice(vkd, cb, src.v, 0, v_row.buffer, v_row.offset, total_kv_rows_bytes);
            barrierTransferToShader(vkd, cb);
            self.prof.mark(cb, .copy);

            // The resident prefix runs the batch kernel on the slot-0 set. A
            // row past one split reads slot 0 through the whole-region set,
            // as a decode row does, so it gets the bytes of that decode row.
            // The stride read stays inside the loop, because an all-host split
            // holds no device KV.
            attn_record.recordPrefill(kernels.AttnKit, self.attnKit(), cb, &prefill_plan, .{
                .shape = self.attnShape(self.deviceKv().layout.streamStrideElems()),
                .resident = .{ .batch = self.set_attention[j] },
                .split = self.set_attention_split[j],
                .merge = self.set_attention_merge,
            }, start_pos);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .attn);

            self.recordBatchMatmul(
                cb,
                self.set_o[j],
                b,
                self.n_heads * self.head_dim,
                self.hidden_dim,
                "o",
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_out);

            self.k_residual.recordInto(cb, self.set_residual_attn, b * self.hidden_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);

            self.k_rmsnorm_batch.recordInto(
                cb,
                self.set_rmsnorm_ffn[j],
                b,
                self.hidden_dim,
                self.rms_norm_eps,
            );
            barrierShader(vkd, cb);
            self.prof.mark(cb, .norm);

            self.recordBatchMatmul(cb, self.set_gate[j], b, self.hidden_dim, self.ffn_dim, "gate");
            self.recordBatchMatmul(cb, self.set_up[j], b, self.hidden_dim, self.ffn_dim, "up");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_up);

            self.k_swiglu.recordInto(cb, self.set_swiglu, b * self.ffn_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);

            self.recordBatchMatmul(cb, self.set_down[j], b, self.ffn_dim, self.hidden_dim, "down");
            barrierShader(vkd, cb);
            self.prof.mark(cb, .matmul_ffn_down);

            self.k_residual.recordInto(cb, self.set_residual_ffn, b * self.hidden_dim);
            barrierShader(vkd, cb);
            self.prof.mark(cb, .elementwise);
        }

        self.k_rmsnorm_batch.recordInto(
            cb,
            self.set_rmsnorm_final,
            b,
            self.hidden_dim,
            self.rms_norm_eps,
        );
        barrierShader(vkd, cb);
        self.prof.mark(cb, .norm);

        // Only the last row's logits matter. Copy it to row 0 so the lm_head
        // descriptor bound at offset 0 sees it. Non-overlapping for b >= 2.
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

        self.recordDecodeMatmul(cb, self.set_lm_head, self.hidden_dim, self.vocab_size, "lmhead");
        barrierShaderToTransfer(vkd, cb);
        self.prof.mark(cb, .lmhead);

        const logits_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.vocab_size) * f32_size;
        copyDeviceToHost(vkd, cb, self.buf_logits.buffer, self.stage_out.buffer, logits_bytes);
        self.prof.mark(cb, .copy);
        self.prof.end(cb);
        try vkd.endCommandBuffer(cb);
    }

    // -- descriptor sets ----------------------------------------------------

    fn allocateAndBindDescriptorSets(self: *LlamaHybridRunner) !void {
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
        const logits_bytes: vk.DeviceSize = @as(
            vk.DeviceSize,
            self.logit_rows_max,
        ) * @as(vk.DeviceSize, self.vocab_size) * @sizeOf(f32);
        // Every attention binding below starts at a region base, which the
        // store packs at whole regions. A storage-buffer descriptor offset
        // must be a multiple of the device offset alignment, so a region must
        // be a whole number of alignment units.
        if (self.kv) |*kv| {
            std.debug.assert(kv.layout.regionBytes() % kv.lim.offset_alignment == 0);
        }

        // The guard in `init` proved that the records fit one binding.
        const split_bytes: vk.DeviceSize = self.attn_scratch.totalBytes();
        std.debug.assert(split_bytes <= self.buf_attn_split.capacity);
        std.debug.assert(split_bytes <= self.ctx.buffer_limits.storage_range);

        var j: u32 = 0;
        while (j < self.n_suffix) : (j += 1) {
            self.set_rmsnorm_attn[j] = try allocOne(ctx, self.desc_pool, layouts.rmsnorm);
            bindRmsnorm(
                ctx,
                self.set_rmsnorm_attn[j],
                self.buf_x.buffer,
                self.attn_norm_w[j],
                self.buf_x_norm.buffer,
                hidden_bytes,
            );

            self.set_rmsnorm_ffn[j] = try allocOne(ctx, self.desc_pool, layouts.rmsnorm);
            bindRmsnorm(
                ctx,
                self.set_rmsnorm_ffn[j],
                self.buf_x.buffer,
                self.ffn_norm_w[j],
                self.buf_x_norm.buffer,
                hidden_bytes,
            );

            self.set_q[j] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_q[j],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.attn_q_w[j],
                self.buf_q.buffer,
                qkv_q_bytes,
            );

            self.set_k[j] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_k[j],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.attn_k_w[j],
                self.buf_k_scratch.buffer,
                kv_scratch_bytes,
            );

            self.set_v[j] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_v[j],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.attn_v_w[j],
                self.buf_v_scratch.buffer,
                kv_scratch_bytes,
            );

            self.set_o[j] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_o[j],
                self.buf_attn_out.buffer,
                qkv_q_bytes,
                &self.attn_o_w[j],
                self.buf_x_norm.buffer,
                hidden_bytes,
            );

            self.set_gate[j] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_gate[j],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.ffn_gate_w[j],
                self.buf_ffn_gate.buffer,
                ffn_bytes,
            );

            self.set_up[j] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_up[j],
                self.buf_x_norm.buffer,
                hidden_bytes,
                &self.ffn_up_w[j],
                self.buf_ffn_up.buffer,
                ffn_bytes,
            );

            self.set_down[j] = try allocOne(ctx, self.desc_pool, layouts.matmul);
            bindMatmul(
                ctx,
                self.set_down[j],
                self.buf_ffn_gate.buffer,
                ffn_bytes,
                &self.ffn_down_w[j],
                self.buf_ffn_down.buffer,
                hidden_bytes,
            );

            // KV regions in suffix coordinates: suffix layer j holds model
            // layer `split + j`. The single-row kernels bind slot 0 of each
            // region, and the multi-stream kernel binds the whole region.
            const kv = self.deviceKv();
            self.set_attention[j] = try allocOne(ctx, self.desc_pool, layouts.attention);
            bindAttention(
                ctx,
                self.set_attention[j],
                self.buf_q.buffer,
                qkv_q_bytes,
                kv.slotView(j, .k, 0),
                kv.slotView(j, .v, 0),
                self.buf_attn_out.buffer,
                qkv_q_bytes,
            );

            self.set_attention_multi[j] = try allocOne(
                ctx,
                self.desc_pool,
                self.k_attention_multi.pipeline.set_layout,
            );
            bindAttention(
                ctx,
                self.set_attention_multi[j],
                self.buf_q.buffer,
                qkv_q_bytes,
                kv.region(j, .k),
                kv.region(j, .v),
                self.buf_attn_out.buffer,
                qkv_q_bytes,
            );

            // Split attention: (q, the whole K region, the whole V region,
            // the records).
            self.set_attention_split[j] = try allocOne(
                ctx,
                self.desc_pool,
                self.k_attention_split.split.set_layout,
            );
            kernels.AttentionSplit.assertKBinding(kv.region(j, .k));
            kernels.AttentionSplit.assertVBinding(kv.region(j, .v));
            bindAttention(
                ctx,
                self.set_attention_split[j],
                self.buf_q.buffer,
                qkv_q_bytes,
                kv.region(j, .k),
                kv.region(j, .v),
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

        self.set_rmsnorm_final = try allocOne(ctx, self.desc_pool, layouts.rmsnorm);
        bindRmsnorm(
            ctx,
            self.set_rmsnorm_final,
            self.buf_x.buffer,
            self.output_norm_w,
            self.buf_x_norm.buffer,
            hidden_bytes,
        );

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

        self.set_swiglu = try allocOne(ctx, self.desc_pool, layouts.swiglu);
        kernels.updateStorageBindings(ctx, self.set_swiglu, &.{
            .{ .buffer = self.buf_ffn_gate.buffer, .offset = 0, .range = ffn_bytes },
            .{ .buffer = self.buf_ffn_up.buffer, .offset = 0, .range = ffn_bytes },
            .{ .buffer = self.buf_ffn_gate.buffer, .offset = 0, .range = ffn_bytes },
        });

        self.set_residual_attn = try allocOne(ctx, self.desc_pool, layouts.residual);
        kernels.updateStorageBindings(ctx, self.set_residual_attn, &.{
            .{ .buffer = self.buf_x_norm.buffer, .offset = 0, .range = hidden_bytes },
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = hidden_bytes },
        });

        self.set_residual_ffn = try allocOne(ctx, self.desc_pool, layouts.residual);
        kernels.updateStorageBindings(ctx, self.set_residual_ffn, &.{
            .{ .buffer = self.buf_ffn_down.buffer, .offset = 0, .range = hidden_bytes },
            .{ .buffer = self.buf_x.buffer, .offset = 0, .range = hidden_bytes },
        });

        const ff_bytes: vk.DeviceSize = @as(vk.DeviceSize, self.rope_dim / 2) * @sizeOf(f32);
        self.set_rope_q = try allocOne(ctx, self.desc_pool, layouts.rope_multi);
        kernels.updateStorageBindings(ctx, self.set_rope_q, &.{
            .{ .buffer = self.buf_q.buffer, .offset = 0, .range = qkv_q_bytes },
            .{ .buffer = self.buf_rope_ff.buffer, .offset = 0, .range = ff_bytes },
        });

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

// The upload, binding, and recording helpers below repeat the ones in
// `llama_fused.zig`. Each file keeps its own copy, so a change to one never
// moves the dispatches the other records.
/// Bytes a KV walk over `n_layers` layers writes for `n_rows` positions of one
/// slot: two regions per layer, and `per_step_elems` values of `kv_dtype` in
/// each row. This is the one arithmetic behind the host half, the device half,
/// and the plan's staging reservation.
fn swapBytes(kv_dtype: dtype.KvDtype, n_layers: u32, n_rows: u32, per_step_elems: u32) u64 {
    return @as(u64, 2) * n_layers * n_rows * kv_dtype.rowBytes(per_step_elems);
}

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

/// A Q6_K lm_head (the higher-precision head ggml emits for Q4_0 ftypes)
/// requantizes to Q4_0 so the GPU path stays uniform, the fused-runner rule.
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

/// Convert a Q6_K tensor to freshly-allocated Q4_0 bytes, one 256-element
/// superblock at a time, so no multi-GB f32 staging copy exists.
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
