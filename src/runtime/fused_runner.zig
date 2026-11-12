//! `FusedRunner(Family)` records the forward pass of one family into one Vulkan command buffer per
//! step and submits it once. `Family` is the decoder layer file of the family: `llama_layer` or
//! `qwen_layer`. The runner owns every device resource of its layers: the weights, the activation
//! scratch, the KV store, the recurrent state, and the descriptor sets.
//!
//! Comptime. `assertFamily` and `assertFusedFamily` check the declarations of `Family` when the
//! type is built, and `layerKernels` takes the step kind at comptime. A build check refuses a
//! build without Vulkan.
//!
//! Family. What differs between two families comes from `Family`: the weight roles and their
//! names, the width of each activation, the KV index and the recurrent slabs of each layer, the
//! rope factor tensor, the forward of a layer, and the `fused_` values. No line of this file reads
//! the architecture, except the refusal of a card of another family.
//!
//! Kernels. A kernel exists only where the family can record it: the Q4_0 kernels for a family
//! with Q4_0 weights, the GEMM by `fused_gemm`, and the narrowing at a narrow KV dtype. The
//! `OpResources` of the layer kernel set build the kernels of the ops that a family can lack, and
//! `Family.fusedGuard` refuses a shape past their shader arrays.
//!
//! Layer range. The runner holds the model layers `[first_layer, L)`, the final norm, and the
//! lm_head. `initSuffix` builds the GPU half of a hybrid. Every family rule and tensor name takes
//! the model index, and the KV regions, the sets, and the weight tables take the range index. The
//! hidden entries take the residual rows of the host layers in place of tokens.
//!
//! Recording. Each entry records each layer through `Family.forward` and one `FusedLayerKernels`
//! value of its step kind. The entry records the input copy and the head: the final norm, the
//! lm_head, and the logits readback, and it marks each as its op (`stage_in`, `final_norm`,
//! `lm_head`, `readback`). The decode ring records the next position while the GPU runs.
//!
//! Sets. Vulkan sets persist, so init points each set at its buffers once. The bind pass of
//! `fused_layer_kernels` makes the sets of every layer from a pool that its count pass sizes.
//!
//! Weights. `fused_weights` holds the weights in two arenas and uploads them through one host
//! stage. The upload order of this file fixes the offset of each weight in an arena.
//!
//! Scratch. Every activation buffer holds `prefill_batch_max` rows, and decode reads row 0. The
//! batch kernels read [b, width] through a push constant, so one set serves every entry.
//! `scratchVramBytes` sizes the buffers of `allocBuffers`, so the reservation of the planner is
//! the sum of the preflight.
//!
//! KV commit. At a narrower KV dtype, one dispatch per layer narrows the f32 K and V rows into the
//! narrowed pair, and the copies read that pair. At f32 the copies read the f32 rows. The
//! recurrent state stays f32 under every KV dtype.
//!
//! Logits. `logit_rows_max` is the most logits rows of one call: the decode rows and the final
//! prefill row of a composed tick. Row i of a step lands at `logits[i * vocab ..]`. One bulk copy
//! per step moves the host-coherent stage into cached heap memory, because the CPU reads the stage
//! slowly across the virtualized bus.
//!
//! Blobs. A blob is the state of one slot after `n` rows: the K rows, then the V rows of a KV
//! layer, or the recurrent slab, then the conv slab of a recurrent layer, in model order. The CPU
//! runner walks the same regions, so a blob restores through either runner at one KV dtype. A
//! save reads back through `kv_readback_stage`, and a restore uploads through `kv_swap_stage`.
//!
//! Tests. Every test here is device-free arithmetic for both families. The equivalence tests in
//! `tests/` and the hybrid gate run on a device.

const std = @import("std");
const build_options = @import("build_options");
const vk = @import("vulkan");

const vulkan_ctx = @import("vulkan_ctx.zig");
const kernels = @import("../ops_vulkan/kernels.zig");
const naive = @import("../ops/naive.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_index = @import("../core/kv_index.zig");
const attn_split = @import("../core/attn_split.zig");
const core_log = @import("../core/utils/log.zig");
const span = @import("../core/span.zig");
const step_stats = @import("../core/step_stats.zig");
const time = @import("../core/utils/time.zig");
const loader = @import("../io/gguf/loader.zig");
const attn_record = @import("attn_record.zig");
const fused_commands = @import("fused_commands.zig");
const fused_weights = @import("fused_weights.zig");
const fused_layer_kernels = @import("fused_layer_kernels.zig");
const gpu_profile = @import("gpu_profile.zig");
const kv_cost = @import("kv_cost.zig");
const layer_kernels = @import("layer_kernels.zig");
const linear_dispatch = @import("linear_dispatch.zig");
const weight_check = @import("weight_check.zig");
const step_row = @import("step_row.zig");
const tensor_lookup = @import("tensor_lookup.zig");

const Allocator = std.mem.Allocator;
const Activation = layer_kernels.Activation;
const FusedLayerKernels = fused_layer_kernels.FusedLayerKernels;
const OpResources = fused_layer_kernels.OpResources;
const StepRow = step_row.StepRow;
const Upload = fused_weights.Upload;
const Weight = layer_kernels.Weight;
const WeightMemory = fused_weights.WeightMemory;
const refuseDtype = fused_weights.refuseDtype;
const WeightQuant = layer_kernels.WeightQuant;
const assert = std.debug.assert;

const copyHostToDevice = fused_commands.copyHostToDevice;
const copyDeviceToDevice = fused_commands.copyDeviceToDevice;
const copyDeviceToHost = fused_commands.copyDeviceToHost;
const barrierShader = fused_commands.barrierShader;
const barrierTransferToTransfer = fused_commands.barrierTransferToTransfer;
const barrierTransferToShader = fused_commands.barrierTransferToShader;
const barrierShaderToTransfer = fused_commands.barrierShaderToTransfer;
const findOptional = tensor_lookup.find;
const findTensor = tensor_lookup.findRequired;

const log = std.log.scoped(.runner);
const log_kv = std.log.scoped(.kv);
const log_dispatch = std.log.scoped(.dispatch);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/fused_runner.zig was included in a non-vulkan build.");
    }
}

// -- constants --------------------------------------------------------------

/// The row count where the tiled GEMM wins. The layer kernel set reads the same threshold.
const gemm_min_batch = fused_layer_kernels.gemm_min_batch;

// -- types ------------------------------------------------------------------

/// The model of `initRange`: the loaded model and the first layer of the range. The range ends at
/// the last layer, so the runner always owns the head.
const ModelRange = struct {
    loaded: *const loader.Loaded,
    first: u32,
};

/// The device bytes of one slot over a layer range.
const RangeSlotBytes = struct {
    /// The K and V rows of the KV layers at the window.
    kv: u64,
    /// The recurrent slab and the conv slab of each recurrent layer.
    recurrent: u64,

    fn total(self: RangeSlotBytes) u64 {
        return self.kv + self.recurrent;
    }
};

/// The kernels of a runner, each with its pipeline and its set layout. Null: the family records no
/// such kernel, or the KV dtype needs no narrowing.
const Pipelines = struct {
    linear: kernels.LinearQ80Reduce,
    linear_batch: kernels.LinearQ80Batch,
    linear_q4: ?kernels.LinearQ40Reduce,
    linear_batch_q4: ?kernels.LinearQ40Batch,
    linear_gemm: ?kernels.LinearQ80Gemm,
    rmsnorm: kernels.Rmsnorm,
    rmsnorm_batch: kernels.RmsnormBatch,
    rope: kernels.RopeMulti,
    rope_batch: kernels.RopeMultiBatch,
    swiglu: kernels.Swiglu,
    residual: kernels.ResidualAdd,
    attention: kernels.Attention,
    attention_batch: kernels.AttentionBatch,
    attention_multi: kernels.AttentionMultiStream,
    attention_split: kernels.AttentionSplit,
    kv_commit: ?kernels.KvCommitF16,

    /// Free every kernel, in reverse construction order.
    fn deinit(self: *Pipelines) void {
        if (self.kv_commit) |*k| {
            k.deinit();
        }

        self.attention_split.deinit();
        self.attention_multi.deinit();
        self.attention_batch.deinit();
        self.attention.deinit();
        self.residual.deinit();
        self.swiglu.deinit();
        self.rope_batch.deinit();
        self.rope.deinit();
        self.rmsnorm_batch.deinit();
        self.rmsnorm.deinit();
        if (self.linear_gemm) |*k| {
            k.deinit();
        }

        if (self.linear_batch_q4) |*k| {
            k.deinit();
        }

        if (self.linear_q4) |*k| {
            k.deinit();
        }

        self.linear_batch.deinit();
        self.linear.deinit();
    }

    /// The attention kernel set over the attention pipelines. It marks its passes in `prof`, and
    /// the bind pass, which records nothing, passes null.
    fn attnKernels(self: *Pipelines, prof: ?*gpu_profile.GpuProfile) kernels.AttnKernels {
        return .{
            .single = &self.attention,
            .multi = &self.attention_multi,
            .passes = &self.attention_split,
            .batch = &self.attention_batch,
            .prof = prof,
        };
    }
};

/// The device scratch of a runner. An empty buffer frees nothing, so a partial allocation frees
/// cleanly.
const Buffers = struct {
    /// One buffer for each activation that the family holds, `prefill_batch_max` rows each.
    acts: std.EnumArray(Activation, kernels.DeviceBuffer),
    /// The K and V rows of the storage element, written by the narrowing.
    k_narrowed: kernels.DeviceBuffer,
    v_narrowed: kernels.DeviceBuffer,
    /// The part records of the split attention.
    attn_records: kernels.DeviceBuffer,
    /// The logits rows of one step, `logit_rows_max` rows.
    logits: kernels.DeviceBuffer,
    /// The frequency divisor of each rotated pair, binding 1 of the rope shaders.
    rope_factors: kernels.HostBuffer,

    fn empty(ctx: *const vulkan_ctx.Context) Buffers {
        return .{
            .acts = .initFill(.{ .ctx = ctx }),
            .k_narrowed = .{ .ctx = ctx },
            .v_narrowed = .{ .ctx = ctx },
            .attn_records = .{ .ctx = ctx },
            .logits = .{ .ctx = ctx },
            .rope_factors = .{ .ctx = ctx },
        };
    }

    fn deinit(self: *Buffers) void {
        self.rope_factors.deinit();
        self.logits.deinit();
        self.attn_records.deinit();
        self.v_narrowed.deinit();
        self.k_narrowed.deinit();
        for (&self.acts.values) |*buffer| {
            buffer.deinit();
        }
    }
};

/// The weights and the recurrent state of one layer of the range. An empty entry frees nothing.
const LayerStore = struct {
    /// The matmul weights, keyed by role. Null: the layer holds no such matrix.
    matrices: std.EnumArray(Weight, ?kernels.Q80Weight),
    /// The arena view of each f32 weight. Null: the layer holds no such vector.
    vectors: std.EnumArray(Weight, ?vk.DescriptorBufferInfo),
    /// The recurrent slab and the conv slab of every slot. Empty on a KV layer.
    state: kernels.DeviceBuffer,
    conv: kernels.DeviceBuffer,

    fn empty(ctx: *const vulkan_ctx.Context) LayerStore {
        return .{
            .matrices = .initFill(null),
            .vectors = .initFill(null),
            .state = .{ .ctx = ctx },
            .conv = .{ .ctx = ctx },
        };
    }

    /// Free the recurrent slabs. The arenas of `WeightMemory` own the weights.
    fn deinit(self: *LayerStore) void {
        self.conv.deinit();
        self.state.deinit();
    }

    /// The buffers that the sets of the layer bind, keyed by weight role.
    fn buffers(self: *const LayerStore) fused_layer_kernels.LayerBuffers {
        var out: fused_layer_kernels.LayerBuffers = .{
            .weights = .initFill(null),
            .state = if (self.state.capacity > 0) bi(self.state) else null,
            .conv = if (self.conv.capacity > 0) bi(self.conv) else null,
        };
        for (std.enums.values(Weight)) |role| {
            if (self.matrices.getPtrConst(role).*) |*weight| {
                out.weights.set(role, .{ .matrix = weight });
            } else if (self.vectors.get(role)) |view| {
                out.weights.set(role, .{ .vector = view });
            }
        }

        return out;
    }
};

/// The final norm and the lm_head, both in the arenas of `WeightMemory`.
const Head = struct {
    norm: ?vk.DescriptorBufferInfo,
    lm_head: ?kernels.Q80Weight,

    const empty: Head = .{ .norm = null, .lm_head = null };
};

/// The fused runner of the family `Family` over the model layers `[first_layer, L)`.
pub fn FusedRunner(comptime Family: type) type {
    return struct {
        const Self = @This();

        gpa: Allocator,
        /// The command buffer, the decode ring, the stages, the profile, and the decode host times.
        commands: fused_commands.FusedCommands,
        /// The card of the whole model. The family rules and the layer kernel set read it.
        card: loader.ModelCard,
        /// The model index of range layer 0. Zero for the whole model.
        first_layer: u32,
        /// The layers of the range, `L - first_layer`.
        n_layers: u32,
        max_seq: u32,
        /// Resident conversations. A KV budget holds breadth past the decode width.
        n_slots: u32,
        /// The most logits rows of one call. It sizes the logits buffer, `stage_out`, and `logits`.
        logit_rows_max: u32,
        /// The rows of one step, and of every scratch buffer.
        prefill_batch_max: u32,
        /// The element of every KV row.
        kv_dtype: dtype.KvDtype,
        /// The embedding table, which the host reads for the lookup of each token.
        embed: *const Tensor,
        /// The quant of every matmul weight. It picks the kernel form at every record site.
        quant: WeightQuant,

        pipelines: Pipelines,
        /// The kernels and the scratch of the ops that a family can lack.
        op_resources: OpResources,
        buffers: Buffers,
        /// The KV regions of the KV layers of the range.
        kv: kernels.KvStore,
        /// The record box of the split rows in flight, sized at the declared context.
        attn_scratch: attn_split.Scratch,
        memory: WeightMemory,
        /// The weights and the recurrent state of each layer of the range.
        layers: []LayerStore,
        head: Head,

        desc_pool: vk.DescriptorPool,
        set_output_norm: vk.DescriptorSet,
        set_lm_head: vk.DescriptorSet,
        /// The one merge set of the split attention: the records and the attention output.
        set_attention_merge: vk.DescriptorSet,
        /// The one narrowing set. A null handle at f32.
        set_kv_commit: vk.DescriptorSet,
        /// The sets of each layer, keyed by op. They bind once at init and never change.
        bindings: []layer_kernels.LayerBinding,
        /// The KV region of each layer. Null: a recurrent layer.
        kv_index: []?u32,
        /// The elements of each activation buffer in one step row.
        activation_widths: std.EnumArray(Activation, u32),
        /// True after init binds the last set. A ring recording stays valid only while no set
        /// changes.
        sets_fixed: bool,
        /// The logits rows of the last call, compact in step row order.
        logits: []f32,
        /// Telemetry only: the DMA and memcpy split of the last evict or restore, for a bench.
        last_swap_split: kernels.SwapSplit,

        /// The session surface reads this marker and then needs both slot hooks.
        pub const has_recurrent_state: bool = Family.has_recurrent_state;

        /// The `run` command batches the prompt through `prefillBatch`, as serve and bench-model
        /// do.
        pub const chunked_prefill: bool = true;

        /// Rows of one step: the size of the scratch and the descriptor bindings.
        pub const prefill_rows_max: u32 = Family.fused_rows_max;

        pub const InitError = Allocator.Error || weight_check.Error || error{
            StepBufferPastDeviceLimit,
            UnsupportedArchitecture,
            UnsupportedWeightDtype,
            VramBudgetExceeded,
            KvRegionPastDeviceLimit,
            AttentionScratchPastDeviceLimit,
            AttentionHeadDimUnsupported,
            BindCountMismatch,
            NoKvLayerInRange,
            DeviceFailedAtInit,
        } || tensor_lookup.Error || GuardError || OpResources.InitError ||
            vulkan_ctx.InitError ||
            kernels.LinearQ80Reduce.InitError ||
            kernels.Q80Weight.InitError ||
            kernels.VramArena.AllocError ||
            kernels.KvStore.InitError ||
            kernels.SubmitError ||
            vk.DeviceWrapper.CreateDescriptorPoolError ||
            vk.DeviceWrapper.AllocateDescriptorSetsError ||
            vk.DeviceWrapper.AllocateCommandBuffersError;

        /// The refusals of the shape guard of the family. A family without a check has none.
        const GuardError = @typeInfo(
            @typeInfo(@TypeOf(Family.fusedGuard)).@"fn".return_type.?,
        ).error_union.error_set;

        /// The family holds Q4_0 weights, so the runner builds the Q4_0 matmul kernels.
        const has_q4 = std.mem.indexOfScalar(WeightQuant, Family.fused_quants, .q4_0) != null;

        comptime {
            layer_kernels.assertFamily(Family);
            layer_kernels.assertFusedFamily(Family);
            assert(Family.fused_quants.len >= 1);
            assert(prefill_rows_max >= 1);
            assert(prefill_rows_max <= limits.max_prefill_chunk);
        }

        // -- budgets --------------------------------------------------------

        /// What one KV slot costs on this model. The planner states the size, so the CLI reads the
        /// same figure before a runner exists.
        pub fn kvCostModel(card: loader.ModelCard, kv_dtype: dtype.KvDtype) kv_budget.KvCostModel {
            assert(card.arch == Family.arch);

            return kv_budget.costModel(card.kvShape(), kv_dtype);
        }

        /// The device bytes of the scratch of `allocBuffers` and of the `OpResources`:
        /// `prefill_rows_max` rows of each activation and of the op scratch, the split records,
        /// and the narrowed commit pair.
        pub fn scratchVramBytes(card: loader.ModelCard, kv_dtype: dtype.KvDtype) u64 {
            var per_row: u64 = 0;
            for (std.enums.values(Activation)) |act| {
                per_row += Family.activationWidth(&card, act);
            }

            const rows: u64 = prefill_rows_max;
            const f32_bytes = rows * per_row * @sizeOf(f32) +
                OpResources.scratchBytes(Family, &card, rows) +
                attnSplitScratch(card).totalBytes();
            if (kv_dtype == .f32) return f32_bytes;

            return f32_bytes + rows * 2 * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        }

        /// The device bytes of every weight that `init` uploads for the whole model.
        pub fn weightsVramBytes(loaded: *const loader.Loaded) u64 {
            const n_layers = loaded.card.n_layers;
            return weightsVramBytesRange(loaded.tensors, n_layers, 0, n_layers);
        }

        /// `weightsVramBytes` over the model layers `[from..to)`, plus the head, which every range
        /// runs. `token_embd` stays on the host, so it counts only as the tied lm_head.
        pub fn weightsVramBytesRange(
            refs: []const loader.TensorRef,
            n_layers: u32,
            from: u32,
            to: u32,
        ) u64 {
            assert(from <= to);
            assert(to <= n_layers);

            var table: [limits.max_layers]u64 = undefined;
            const per_layer = tensor_lookup.layerWeightBytes(refs, n_layers, &table);
            var total: u64 = tensor_lookup.headWeightBytes(refs);
            for (per_layer[from..to]) |bytes| {
                total += bytes;
            }

            return total;
        }

        /// The device bytes of `n_slots` logits rows.
        pub fn logitsVramBytes(vocab_size: u32, n_slots: u32) u64 {
            return @as(u64, n_slots) * vocab_size * @sizeOf(f32);
        }

        /// The planner inputs of this model at `vram_free`, the measured free device-local bytes,
        /// on `driver`. Zero keeps the planner off the budget. `logit_rows` must cover the rows of
        /// `init`.
        pub fn kvCaps(
            loaded: *const loader.Loaded,
            ceiling: u32,
            vram_free: u64,
            driver: limits.Driver,
            logit_rows: u32,
            kv_dtype: dtype.KvDtype,
        ) kv_budget.Caps {
            assert(ceiling >= 1);
            assert(ceiling <= loaded.card.context_length);

            const windows = kv_budget.defaultWindows(kvCostModel(loaded.card, kv_dtype), ceiling);
            return .{
                .vram_free = vram_free,
                .driver = driver,
                .weights = weightsVramBytes(loaded),
                .logits = logitsVramBytes(loaded.card.vocab_size, logit_rows),
                .scratch = scratchVramBytes(loaded.card, kv_dtype),
                .model_ctx = ceiling,
                .serve_default = windows.serve,
                .default_ceil = windows.run,
            };
        }

        // -- init -----------------------------------------------------------

        /// The whole-model runner. The flag resolution has gated `kv_dtype` against the model.
        pub fn init(
            gpa: Allocator,
            ctx: *const vulkan_ctx.Context,
            loaded: *const loader.Loaded,
            plan: kv_budget.KvPlan,
            kv_dtype: dtype.KvDtype,
        ) InitError!*Self {
            return initRange(gpa, ctx, .{ .loaded = loaded, .first = 0 }, plan, kv_dtype);
        }

        /// The runner of the model layers `[first..L)`: the GPU half of a hybrid. The plan sizes
        /// the range, and `n_streams = n_slots` lets every slot step in one tick.
        pub fn initSuffix(
            gpa: Allocator,
            ctx: *const vulkan_ctx.Context,
            loaded: *const loader.Loaded,
            plan: kv_budget.KvPlan,
            first: u32,
            kv_dtype: dtype.KvDtype,
        ) InitError!*Self {
            assert(first < loaded.card.n_layers);

            return initRange(gpa, ctx, .{ .loaded = loaded, .first = first }, plan, kv_dtype);
        }

        /// Every init funnels here, so one place names the retry after a device failure.
        fn initRange(
            gpa: Allocator,
            ctx: *const vulkan_ctx.Context,
            range: ModelRange,
            plan: kv_budget.KvPlan,
            kv_dtype: dtype.KvDtype,
        ) InitError!*Self {
            return buildRange(gpa, ctx, range, plan, kv_dtype) catch |err| {
                const text = deviceFailureText(err) orelse return err;
                log.err("{s}", .{text});
                return error.DeviceFailedAtInit;
            };
        }

        /// One body sizes and frames every buffer of every init.
        fn buildRange(
            gpa: Allocator,
            ctx: *const vulkan_ctx.Context,
            range: ModelRange,
            plan: kv_budget.KvPlan,
            kv_dtype: dtype.KvDtype,
        ) InitError!*Self {
            // Refuse a model that the kernels cannot run, then a plan past the device.
            const refs = range.loaded.tensors;
            const quant = try checkModel(range.loaded);
            const kv_layout = try checkPlan(ctx, range, plan, kv_dtype);

            // Reserve the runner, and fill the fields that need no device allocation.
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);

            self.* = try initFields(gpa, ctx, range, plan, kv_dtype, quant);
            // Generic callers read the field, and the scratch and the bindings hold the constant.
            assert(self.prefill_batch_max == prefill_rows_max);
            errdefer self.commands.deinit();
            errdefer self.memory.deinit();

            // Build the kernels, the op resources, the device scratch, and the KV store.
            self.pipelines = try initPipelines(ctx, kv_dtype);
            errdefer self.pipelines.deinit();

            self.op_resources = try .init(Family, ctx, &self.card, self.prefill_batch_max);
            errdefer self.op_resources.deinit();

            errdefer self.buffers.deinit();
            try self.allocBuffers(refs);

            self.kv = try kernels.KvStore.init(ctx, kv_layout);
            errdefer self.kv.deinit();

            // Allocate the host stages, the host logits, and the tables of the range.
            errdefer self.freeHost();
            try self.allocHost();

            // Upload the weights of each layer and of the head, then zero the recurrent slabs.
            errdefer self.freeSlabs();
            try self.uploadWeights(refs);
            try self.allocSlabs();

            // Make the descriptor pool at its exact size, then the head sets and the layer sets.
            try self.createSets();
            errdefer ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

            // Allocate the command buffers, the decode ring, and the profile.
            try self.commands.initRecording();
            self.sets_fixed = true;
            self.logInit(refs);
            return self;
        }

        /// Free every resource of the runner, then the runner itself.
        pub fn deinit(self: *Self) void {
            const ctx = self.commands.ctx;
            log.debug("deinit {t}: layers [{d}..{d}), {d} slots", .{
                self.commands.gpu_runner,
                self.first_layer,
                self.first_layer + self.n_layers,
                self.n_slots,
            });

            // Free the command buffers and the stages, then the pool, which frees every set.
            self.commands.deinit();
            ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

            // Free the recurrent slabs, the host tables, then the device memory.
            self.freeSlabs();
            self.freeHost();
            self.memory.deinit();
            self.kv.deinit();
            self.buffers.deinit();
            self.op_resources.deinit();
            self.pipelines.deinit();
            self.gpa.destroy(self);
        }

        // -- decode step ----------------------------------------------------

        /// One decode step of `token_id` on slot 0. Returns the logits row, valid until the next
        /// call. The ring records the step on a miss and the next step while the GPU runs.
        pub fn step(self: *Self, token_id: u32, pos: u32) ![]f32 {
            // The callers guard this too, but a public entry point checks its own precondition.
            if (pos >= self.max_seq) return error.PositionOutOfRange;

            log.debug("step: token {d} at pos {d}", .{ token_id, pos });
            self.embedRow(0, token_id);
            return self.stepStaged(pos);
        }

        /// `step` from one activation row instead of one token id. The decode ring key never reads
        /// `stage_in`, so this entry records what `step` records.
        pub fn stepHidden(self: *Self, x: []const f32, pos: u32) ![]f32 {
            assert(x.len == self.card.embedding_length);

            if (pos >= self.max_seq) return error.PositionOutOfRange;

            log.debug("step hidden: pos {d}", .{pos});
            @memcpy(self.commands.stage_in.sliceTyped(f32, x.len), x);
            return self.stepStaged(pos);
        }

        /// The decode body: `stage_in` already holds the row of this step.
        fn stepStaged(self: *Self, pos: u32) ![]f32 {
            assert(pos < self.max_seq);
            assert(self.sets_fixed);

            // Submit the step through the ring and wait on its fence.
            const key = gpu_profile.decodeKey(pos, self.commands.decode.level());
            const rec: DecodeRec = .{ .runner = self };
            try self.commands.decode.step(key, key.next(self.max_seq), rec, &self.commands.host);

            // Stream-copy the stage into the heap logits once.
            const t0 = core_log.now();
            const vocab = self.card.vocab_size;
            @memcpy(self.logits[0..vocab], self.commands.stage_out.sliceTyped(f32, vocab));
            self.commands.host.copy.add(time.rawElapsedNs(t0, core_log.now()));
            return self.logits[0..vocab];
        }

        /// The record callback of the decode ring.
        const DecodeRec = struct {
            runner: *Self,

            pub fn recordDecode(
                self: DecodeRec,
                cb: vk.CommandBuffer,
                prof: *gpu_profile.GpuProfile,
                key: gpu_profile.DecodeKey,
            ) !void {
                return self.runner.recordDecode(cb, prof, key.pos, key.opts);
            }
        };

        /// Record one decode step at `pos` into `cb`.
        fn recordDecode(
            self: *Self,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            pos: u32,
            opts: gpu_profile.RecordOpts,
        ) !void {
            try self.openRecord(cb, prof, 1, opts.one_time);
            self.recordLayers(.decode, cb, prof, .{ .pos = pos });
            self.recordRowHead(cb, prof);
            prof.end(cb);
            try self.commands.ctx.vkd.endCommandBuffer(cb);
        }

        // -- bench hooks ----------------------------------------------------
        // The `bench-decode-host` command times the pieces of one decode step. None of these
        // hooks runs in a normal decode.

        /// Record the decode command buffer for `pos` and stop: no submit. The profile must stay
        /// at the process level, so the timing is the record cost.
        pub fn probeRecord(self: *Self, pos: u32) !void {
            assert(self.commands.prof.level == span.gpuLevel());
            assert(pos < self.max_seq);

            try self.recordDecode(self.commands.cmd_buffer, &self.commands.prof, pos, .{});
            self.commands.probe_recorded = false;
        }

        /// Record once without the one-time-submit flag, so `probeSubmitWait` can submit the same
        /// buffer again. A resubmit repeats the KV and state writes, so its output has no meaning.
        pub fn probeRecordReusable(self: *Self, pos: u32) !void {
            assert(self.commands.prof.level == span.gpuLevel());
            assert(pos < self.max_seq);

            const cb = self.commands.cmd_buffer;
            try self.recordDecode(cb, &self.commands.prof, pos, .{ .one_time = false });
            self.commands.probe_recorded = true;
        }

        /// Submit the buffer that `probeRecordReusable` recorded, and wait for the queue.
        pub fn probeSubmitWait(self: *Self) !void {
            assert(self.commands.probe_recorded);

            try self.commands.submitWaitUntimed();
        }

        /// Set the timestamp level of the decode ring. Only the gap mode of `bench-decode-host`
        /// calls it. It drops every recording first.
        pub fn setDecodeLevel(self: *Self, level: gpu_profile.Level) !void {
            try self.commands.decode.setLevel(level, &self.commands.host);
        }

        /// The timestamps of the last decode step, or null at level `off`.
        pub fn lastDecodeReadout(self: *const Self) ?gpu_profile.Readout {
            return self.commands.decode.last;
        }

        // -- batched step ---------------------------------------------------

        /// Rows of the compact logits buffers. `Sessions` derives the vocab stride from this, so it
        /// states the allocated row count exactly.
        pub fn logitRowsMax(self: *const Self) u32 {
            return self.logit_rows_max;
        }

        /// Max rows one `stepBatch` accepts: `prefill_batch_max`, the rows of the scratch. The slot
        /// count and `logit_rows_max` bound only the slot index and the logits suffix.
        pub fn maxStepRows(self: *const Self) u32 {
            return self.prefill_batch_max;
        }

        /// The decode host times that `generate.decode` clears before a run and reads after it.
        pub fn decodeHost(self: *Self) *step_stats.DecodeHost {
            return &self.commands.host;
        }

        /// Advance a batch of rows, each a (slot, token, pos) triple over an independent slot.
        /// Logits land compact at `logits[0..]`.
        pub fn stepBatch(self: *Self, rows: []const StepRow) !void {
            // Validate the shared row rule. `n_logits` can be 0.
            const n_logits = try step_row.validateRows(
                rows,
                self.n_slots,
                self.prefill_batch_max,
                self.max_seq,
            );
            log.debug(
                "step_batch: {d} rows, slot {d} pos {d}..{d}",
                .{ rows.len, rows[0].slot, rows[0].pos, rows[rows.len - 1].pos },
            );

            for (rows, 0..) |r, i| {
                self.embedRow(i, r.token);
            }

            return self.stepBatchStaged(rows, n_logits);
        }

        /// `stepBatch` from one activation row per step row instead of one token per row.
        pub fn stepBatchHidden(self: *Self, x: []const f32, rows: []const StepRow) !void {
            const n_logits = try step_row.validateRows(
                rows,
                self.n_slots,
                self.prefill_batch_max,
                self.max_seq,
            );
            assert(x.len == rows.len * @as(usize, self.card.embedding_length));

            log.debug(
                "step_batch hidden: {d} rows, slot {d} pos {d}..{d}",
                .{ rows.len, rows[0].slot, rows[0].pos, rows[rows.len - 1].pos },
            );
            @memcpy(self.commands.stage_in.sliceTyped(f32, x.len), x);
            return self.stepBatchStaged(rows, n_logits);
        }

        /// The batch body: `stage_in` already holds one row per step row.
        fn stepBatchStaged(self: *Self, rows: []const StepRow, n_logits: u32) !void {
            assert(n_logits <= self.logit_rows_max);
            assert(rows.len <= self.prefill_batch_max);

            // Record, submit, and wait. Only a live dispatch log reads the clock.
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

            try self.commands.submitWait(.decode_batch);
            if (live) {
                log_dispatch.debug(
                    "step_batch {d} rows: record {d:.1} ms, submit+wait {d:.1} ms",
                    .{ rows.len, record_ms, core_log.msSince(t_submit) },
                );
            }

            // The GPU compacted the want_logits suffix to the front.
            const n_floats = @as(usize, n_logits) * self.card.vocab_size;
            @memcpy(self.logits[0..n_floats], self.commands.stage_out.sliceTyped(f32, n_floats));
        }

        /// Record one step over `rows`: batched projections, one recurrence per same-slot run,
        /// and rope and attention per row against the slot of the row.
        fn recordStepBatch(self: *Self, rows: []const StepRow) !void {
            const cb = self.commands.cmd_buffer;
            const prof = &self.commands.prof;
            const n: u32 = @intCast(rows.len);
            try self.openRecord(cb, prof, n, true);
            self.recordLayers(.decode_batch, cb, prof, .{ .rows = rows });

            // A step with no want-logits row ends here, because its KV and state writes are its
            // whole product. A zero-size `cmdCopyBuffer` is not valid Vulkan.
            const first = step_row.wantLogitsFirst(rows);
            if (first < n) {
                self.recordBatchHead(cb, prof, n, first);
            }

            prof.end(cb);
            try self.commands.ctx.vkd.endCommandBuffer(cb);
        }

        // -- prefill --------------------------------------------------------

        /// Run `tokens` on slot 0 from `start_pos` in one batched step, and return the logits of
        /// the last token, valid until the next call. Callers chunk a longer prompt.
        pub fn prefillBatch(self: *Self, tokens: []const u32, start_pos: u32) ![]f32 {
            assert(tokens.len >= 1);

            // The row rule refuses a chunk past the scratch or past the window in every build mode.
            if (tokens.len > self.prefill_batch_max) return error.TooManyRows;

            var rows_buf: [prefill_rows_max]StepRow = undefined;
            const rows = step_row.chunkRows(&rows_buf, tokens, start_pos);
            _ = try step_row.validateRows(rows, self.n_slots, self.prefill_batch_max, self.max_seq);

            const b: u32 = @intCast(tokens.len);
            log.debug("prefill: {d} tokens at start_pos {d}", .{ b, start_pos });
            for (tokens, 0..) |token, i| {
                self.embedRow(i, token);
            }

            return self.prefillStaged(b, start_pos);
        }

        /// `prefillBatch` from `b` activation rows instead of `b` token ids.
        pub fn prefillBatchHidden(self: *Self, x: []const f32, b: u32, start_pos: u32) ![]f32 {
            assert(b >= 1);
            assert(b <= self.prefill_batch_max);
            assert(x.len == @as(usize, b) * self.card.embedding_length);

            log.debug("prefill hidden: {d} rows at start_pos {d}", .{ b, start_pos });
            @memcpy(self.commands.stage_in.sliceTyped(f32, x.len), x);
            return self.prefillStaged(b, start_pos);
        }

        /// The prefill body: `stage_in` already holds the rows of the chunk.
        fn prefillStaged(self: *Self, b: u32, start_pos: u32) ![]f32 {
            assert(b >= 1);
            assert(b <= self.prefill_batch_max);

            // Record, submit, and wait. Only a live dispatch log reads the clock.
            const live = (comptime core_log.hot) and core_log.enabled(.dispatch, .debug);
            var t_record: std.Io.Timestamp = undefined;
            if (live) {
                t_record = core_log.now();
            }

            try self.recordPrefill(b, start_pos);
            var record_ms: f64 = 0;
            var t_submit: std.Io.Timestamp = undefined;
            if (live) {
                record_ms = core_log.msSince(t_record);
                t_submit = core_log.now();
            }

            try self.commands.submitWait(.prefill_chunk);
            if (live) {
                log_dispatch.debug(
                    "prefill b={d}: record {d:.1} ms, submit+wait {d:.1} ms",
                    .{ b, record_ms, core_log.msSince(t_submit) },
                );
            }

            // Copy the logits into heap memory, for the same reason the decode step does.
            const vocab = self.card.vocab_size;
            @memcpy(self.logits[0..vocab], self.commands.stage_out.sliceTyped(f32, vocab));
            return self.logits[0..vocab];
        }

        /// Record one prefill chunk of `b` rows from `start_pos` into `cmd_buffer`.
        fn recordPrefill(self: *Self, b: u32, start_pos: u32) !void {
            const cb = self.commands.cmd_buffer;
            const prof = &self.commands.prof;
            try self.openRecord(cb, prof, b, true);
            self.recordLayers(.prefill_chunk, cb, prof, .{ .b = b, .start_pos = start_pos });

            // The head reads row 0 and only the last row samples, so copy row b-1 of `x` to row 0.
            // The source starts at row 1 or later, so the ranges cannot overlap.
            if (b > 1) {
                const vkd = self.commands.ctx.vkd;
                const x = self.buffers.acts.get(.x).buffer;
                const row_bytes = self.hiddenRowBytes();
                barrierShaderToTransfer(vkd, cb);
                copyDeviceToDevice(vkd, cb, x, (b - 1) * row_bytes, x, 0, row_bytes);
                barrierTransferToShader(vkd, cb);
            }

            self.recordRowHead(cb, prof);
            prof.end(cb);
            try self.commands.ctx.vkd.endCommandBuffer(cb);
        }

        // -- profiling ------------------------------------------------------

        /// Run one prefill chunk with GPU timestamps and print a breakdown by op, for
        /// `--profile-prefill`. It changes the KV and the recurrent state as a real prefill does,
        /// so the runner is dirty afterwards.
        pub fn profilePrefill(
            self: *Self,
            io: std.Io,
            tokens: []const u32,
            start_pos: u32,
            tag: []const u8,
        ) !void {
            assert(tokens.len >= 1);
            assert(tokens.len <= self.prefill_batch_max);

            // Raise the timestamps to `phases` for this call. The pool grows on the first probe,
            // so level off creates none. No submit of this buffer is pending here.
            const level = self.commands.prof.level;
            try self.commands.prof.reserve(.phases);
            self.commands.prof.setLevel(.phases);
            defer self.commands.prof.setLevel(level);

            // Embed the chunk, then record, submit, and wait.
            const b: u32 = @intCast(tokens.len);
            for (tokens, 0..) |token, i| {
                self.embedRow(i, token);
            }

            const t0 = std.Io.Clock.Timestamp.now(io, .awake);
            try self.recordPrefill(b, start_pos);
            const t_rec = std.Io.Clock.Timestamp.now(io, .awake);
            try self.commands.submitWaitUntimed();
            const t_wait = std.Io.Clock.Timestamp.now(io, .awake);

            // Read the timestamps back and print the breakdown.
            const r = (try self.commands.prof.collectInto(
                self.commands.gpu_runner,
                .prefill_chunk,
                t_rec.raw,
                t_wait.raw,
            )).?;
            try r.report(
                io,
                "[{t} prefill {s}] b={d}",
                .{ Family.arch, tag, b },
                time.msBetween(t0, t_rec),
                time.msBetween(t_rec, t_wait),
            );
        }

        /// Run one decode step with GPU timestamps and print a breakdown by op. It changes the
        /// KV and the recurrent state as a real step does, so the runner is dirty afterwards.
        pub fn profileStep(
            self: *Self,
            io: std.Io,
            token_id: u32,
            pos: u32,
            tag: []const u8,
        ) !void {
            // Raise the timestamps to `phases` for this call, as `profilePrefill` does.
            self.commands.invalidateDecode();
            const level = self.commands.prof.level;
            try self.commands.prof.reserve(.phases);
            self.commands.prof.setLevel(.phases);
            defer self.commands.prof.setLevel(level);

            // Embed the token, then record, submit, and wait.
            self.embedRow(0, token_id);
            const t0 = std.Io.Clock.Timestamp.now(io, .awake);
            try self.recordDecode(self.commands.cmd_buffer, &self.commands.prof, pos, .{});
            const t_rec = std.Io.Clock.Timestamp.now(io, .awake);
            try self.commands.submitWaitUntimed();
            const t_wait = std.Io.Clock.Timestamp.now(io, .awake);
            const vocab = self.card.vocab_size;
            @memcpy(self.logits[0..vocab], self.commands.stage_out.sliceTyped(f32, vocab));

            // Read the timestamps back and print the breakdown.
            const r = (try self.commands.prof.collectInto(
                self.commands.gpu_runner,
                .decode,
                t_rec.raw,
                t_wait.raw,
            )).?;
            try r.report(
                io,
                "[{t} decode {s}] pos={d}",
                .{ Family.arch, tag, pos },
                time.msBetween(t0, t_rec),
                time.msBetween(t_rec, t_wait),
            );
        }

        // -- slot state -----------------------------------------------------

        /// Zero the recurrent slabs of a slot for a fresh conversation. `Sessions.create` calls
        /// this on slot reuse. A family without recurrent state does nothing. A failure panics,
        /// because a swallowed one corrupts the next conversation.
        pub fn resetSlotState(self: *Self, slot: u32) void {
            assert(slot < self.n_slots);

            if (comptime !has_recurrent_state) return;

            self.zeroSlot(slot) catch |err| std.debug.panic("fused slot reset failed: {t}", .{err});
        }

        /// Fork slot state for a session clone: every KV region, and the whole recurrent slabs,
        /// which have no position axis. Attention never reads the stale rows past `pos`. A
        /// failure panics for the reason `resetSlotState` states.
        pub fn cloneSlotState(self: *Self, source_slot: u32, target_slot: u32, pos: u32) void {
            assert(source_slot < self.n_slots);
            assert(target_slot < self.n_slots);
            assert(source_slot != target_slot);
            assert(pos <= self.max_seq);

            log_kv.debug("clone slot {d} -> {d}, pos {d}", .{ source_slot, target_slot, pos });
            self.copySlot(source_slot, target_slot) catch |err| {
                std.debug.panic("fused slot clone failed: {t}", .{err});
            };
        }

        // -- blobs ----------------------------------------------------------

        /// Bytes of a blob at `n` rows: two KV rows a position for a KV layer, and the two slabs of
        /// one slot for a recurrent layer, which stay f32 under every KV dtype.
        pub fn snapshotBytes(self: *const Self, n: u32) usize {
            const kv_bytes: usize = 2 * @as(usize, n) * @as(usize, @intCast(self.kvRowBytes()));
            const slab = self.slabBytes();
            const slab_bytes: usize = @intCast(slab.state + slab.conv);
            var total: usize = 0;
            for (self.kv_index) |index| {
                total += if (index != null) kv_bytes else slab_bytes;
            }

            return total;
        }

        /// Save the state of slot 0 after `n` rows into `out`, which holds `snapshotBytes(n)`
        /// bytes.
        pub fn snapshotInto(self: *Self, n: u32, out: []u8) !void {
            assert(n <= self.max_seq);
            assert(out.len == self.snapshotBytes(n));

            _ = try self.stageSlot(0, n, .{ .to_host = out });
        }

        /// Restore slot 0 from a `snapshotInto(n)` blob of this geometry, which the cache
        /// fingerprint of the model guards. Returns the resume position, which equals `n`.
        pub fn restoreFromBytes(self: *Self, blob: []const u8, n: u32) !u32 {
            assert(n <= self.max_seq);
            assert(blob.len == self.snapshotBytes(n));

            log.debug("restore: {d} positions, slot 0", .{n});
            _ = try self.stageSlot(0, n, .{ .to_device = blob });
            return n;
        }

        /// Bytes the state of one slot occupies in a host blob at `n_rows` committed positions.
        pub fn evictSlotBytesFor(self: *const Self, n_rows: u32) u64 {
            return self.snapshotBytes(n_rows);
        }

        /// Copy the state of `slot` over its first `n_rows` positions out of VRAM into `out`.
        pub fn evictSlot(self: *Self, slot: u32, n_rows: u32, out: []u8) !void {
            assert(slot < self.n_slots);
            assert(n_rows >= 1); // a blob must hold rows to restore from
            assert(n_rows <= self.max_seq);
            assert(out.len == self.snapshotBytes(n_rows));

            const t0 = core_log.now();
            self.last_swap_split = try self.stageSlot(slot, n_rows, .{ .to_host = out });
            self.logSwap("evict", slot, n_rows, t0);
        }

        /// Restore a blob of `n_rows` rows that `evictSlot` wrote into `slot`.
        pub fn restoreSlot(self: *Self, slot: u32, n_rows: u32, blob: []const u8) !void {
            assert(slot < self.n_slots);
            assert(n_rows >= 1); // a blob must hold rows to restore from
            assert(n_rows <= self.max_seq);
            assert(blob.len == self.snapshotBytes(n_rows));

            const t0 = core_log.now();
            self.last_swap_split = try self.stageSlot(slot, n_rows, .{ .to_device = blob });
            self.logSwap("restore", slot, n_rows, t0);
        }

        // -- helpers: init --------------------------------------------------

        /// Refuse a card of another family, a weight that no kernel reads or that the card does
        /// not describe, and a shape past the shader arrays. Returns the quant of the matmul
        /// weights.
        fn checkModel(loaded: *const loader.Loaded) InitError!WeightQuant {
            const card = &loaded.card;
            try weight_check.requireFamily(Family, card);
            try linear_dispatch.requireWeights(loaded.tensors, card.n_layers);
            try weight_check.checkWeights(Family, card, loaded.tensors);
            try kernels.AttentionSplit.guardHeadDim(card.head_dim);
            try Family.fusedGuard(card);
            return detectQuant(loaded);
        }

        /// The quant of the matmul weights: the dtype of the first matrix of model layer 0. Each
        /// upload checks its tensor against it, so one model holds one quant.
        fn detectQuant(loaded: *const loader.Loaded) InitError!WeightQuant {
            for (Family.layerWeights(&loaded.card, 0)) |role| {
                if (!role.isMatrix()) continue;

                const ref = try tensor_lookup.findRole(Family, loaded.tensors, role, 0);
                const quant: WeightQuant = switch (ref.tensor.dtype) {
                    .q8_0 => .q8_0,
                    .q4_0 => .q4_0,
                    else => {
                        refuseDtype(ref.name, ref.tensor.dtype);
                        return error.UnsupportedWeightDtype;
                    },
                };
                const known = std.mem.indexOfScalar(WeightQuant, Family.fused_quants, quant);
                if (known == null) {
                    refuseDtype(ref.name, ref.tensor.dtype);
                    return error.UnsupportedWeightDtype;
                }

                return quant;
            }

            unreachable; // Every layer of a family holds a matrix.
        }

        /// Check the plan, and refuse a KV layout, a record box, or a plan past the device before
        /// the first allocation. Returns the KV layout of the range.
        fn checkPlan(
            ctx: *const vulkan_ctx.Context,
            range: ModelRange,
            plan: kv_budget.KvPlan,
            kv_dtype: dtype.KvDtype,
        ) InitError!kv_index.Layout {
            // Slots bound breadth, and streams bound width. A suffix takes the slot ceiling of the
            // hybrid, because its host layers cannot evict a slot.
            const card = &range.loaded.card;
            const slots_max = if (range.first == 0) limits.max_vram_slots else limits.max_batch;
            assert(range.first < card.n_layers);
            assert(plan.n_slots >= 1);
            assert(plan.n_slots <= slots_max);
            assert(plan.n_streams >= 1);
            assert(plan.n_streams <= limits.max_batch);
            assert(plan.n_streams <= plan.n_slots);

            // The f16 batch shader unpacks word pairs, so a narrowed row holds an even count.
            const row_elems = card.n_kv_heads * card.head_dim;
            if (kv_dtype != .f32) assert(row_elems % 2 == 0);

            // Refuse a range without a KV layer, then a KV layout or a record box past the device
            // limits.
            const n_kv = countKvLayers(card, range.first) catch |err| {
                log.err("The layers [{d}..{d}) hold no attention layer, and a fused runner " ++
                    "needs one. Run this model on a CPU build", .{ range.first, card.n_layers });
                return err;
            };
            const layout = kv_index.Layout.init(
                plan.n_slots,
                plan.window,
                row_elems,
                n_kv,
                kv_dtype,
            );
            try ctx.guardKvLayout(layout);

            // The kernels index the recurrent slabs and the activation rows with 32-bit products.
            if (comptime has_recurrent_state) {
                const slot = Family.recurrentElems(card);
                try ctx.guardStepBuffer("GDN state slab", plan.n_slots, slot.state);
                try ctx.guardStepBuffer("GDN conv slab", plan.n_slots, slot.conv);
            }

            var widest: u32 = 0;
            for (std.enums.values(layer_kernels.Activation)) |act| {
                widest = @max(widest, Family.activationWidth(card, act));
            }
            try ctx.guardStepBuffer("activation scratch", prefill_rows_max, widest);

            const records = attnSplitScratch(card.*);
            assert(attn_split.splitCount(plan.window) <= records.max_splits);
            try ctx.guardAttnSplitScratch(records);

            // Refuse a plan past the live device-local budget.
            const vram = vramOf(range, plan, layout, kv_dtype);
            const available = ctx.deviceLocalAvailableBytes();
            try fused_commands.preflightVram(plan, vram, ctx.driver, available);
            return layout;
        }

        /// The device bytes that init allocates for `range`. The slot size from the family rules
        /// equals the span of the cost vector, so two derivations of one range agree.
        fn vramOf(
            range: ModelRange,
            plan: kv_budget.KvPlan,
            layout: kv_index.Layout,
            kv_dtype: dtype.KvDtype,
        ) fused_commands.VramPreflight {
            // Size one slot of the range two ways.
            const card = &range.loaded.card;
            const first = range.first;
            const end = card.n_layers;
            const per_slot = rangeSlotBytes(card, kv_dtype, plan.window, first, end);
            const kv_struct: kv_budget.KvStructure = .{
                .n_slots = plan.n_slots,
                .window = plan.window,
                .costs = kv_cost.costVector(card.*, kv_dtype),
            };
            assert(per_slot.total() == kv_budget.slotSpanBytes(kv_struct, first, end));
            assert(per_slot.kv == layout.slotBytes());

            // Every figure sizes the range, so a suffix reserves what a suffix uploads.
            const whole = weightsVramBytes(range.loaded);
            const weights = weightsVramBytesRange(range.loaded.tensors, end, first, end);
            if (range.first == 0) assert(weights == whole);
            if (range.first > 0) assert(weights < whole);

            const recurrent = @as(u64, plan.n_slots) * per_slot.recurrent;
            return .{
                .weights = weights,
                .kv = layout.totalBytes(),
                .recurrent_state = if (has_recurrent_state) recurrent else null,
                .logits = logitsVramBytes(card.vocab_size, @min(plan.n_slots, plan.n_streams + 1)),
                .scratch = scratchVramBytes(card.*, kv_dtype),
                .cost = .{
                    .kv_bytes_per_token = layout.tokenBytes(),
                    .fixed_state_bytes_per_slot = per_slot.recurrent,
                },
                .n_slots = plan.n_slots,
                .window = plan.window,
                .suffix = range.first > 0,
            };
        }

        /// The runner value before its device resources: the plain fields, and every resource
        /// empty. `initRange` builds `pipelines` and `kv` right after.
        fn initFields(
            gpa: Allocator,
            ctx: *const vulkan_ctx.Context,
            range: ModelRange,
            plan: kv_budget.KvPlan,
            kv_dtype: dtype.KvDtype,
            quant: WeightQuant,
        ) InitError!Self {
            const card = range.loaded.card;
            const embed = try findTensor(range.loaded.tensors, loader.embed_tensor);
            const tag = if (range.first == 0) Family.fused_gpu_runner else Family.hybrid_gpu_runner;
            return .{
                .gpa = gpa,
                .commands = .init(ctx, tag),
                .card = card,
                .first_layer = range.first,
                .n_layers = card.n_layers - range.first,
                .max_seq = plan.window,
                .n_slots = plan.n_slots,
                .logit_rows_max = @min(plan.n_slots, plan.n_streams + 1),
                .prefill_batch_max = prefill_rows_max,
                .kv_dtype = kv_dtype,
                .embed = embed,
                .quant = quant,
                .pipelines = undefined, // built by `initRange` right after this value
                .op_resources = undefined, // built by `initRange` after the pipelines
                .buffers = .empty(ctx),
                .kv = undefined, // built by `initRange` after the scratch
                .attn_scratch = attnSplitScratch(card),
                .memory = .init(ctx, gpa),
                .layers = &.{},
                .head = .empty,
                .desc_pool = .null_handle,
                .set_output_norm = .null_handle,
                .set_lm_head = .null_handle,
                .set_attention_merge = .null_handle,
                .set_kv_commit = .null_handle,
                .bindings = &.{},
                .kv_index = &.{},
                .activation_widths = activationWidths(&card),
                .sets_fixed = false,
                .logits = &.{},
                .last_swap_split = .{ .dma_ms = 0, .memcpy_ms = 0 },
            };
        }

        /// Build the kernels that the family can record. The header names what brings each
        /// optional kernel.
        fn initPipelines(
            ctx: *const vulkan_ctx.Context,
            kv_dtype: dtype.KvDtype,
        ) InitError!Pipelines {
            var p: Pipelines = undefined;

            // Build the matmul kernels.
            p.linear = try kernels.LinearQ80Reduce.init(ctx);
            errdefer p.linear.deinit();

            p.linear_batch = try kernels.LinearQ80Batch.init(ctx);
            errdefer p.linear_batch.deinit();

            p.linear_q4 = if (has_q4) try kernels.LinearQ40Reduce.init(ctx) else null;
            errdefer if (p.linear_q4) |*k| k.deinit();

            p.linear_batch_q4 = if (has_q4) try kernels.LinearQ40Batch.init(ctx) else null;
            errdefer if (p.linear_batch_q4) |*k| k.deinit();

            p.linear_gemm = if (Family.fused_gemm) try kernels.LinearQ80Gemm.init(ctx) else null;
            errdefer if (p.linear_gemm) |*k| k.deinit();

            // Build the norm, rope, and elementwise kernels.
            p.rmsnorm = try kernels.Rmsnorm.init(ctx);
            errdefer p.rmsnorm.deinit();

            p.rmsnorm_batch = try kernels.RmsnormBatch.init(ctx);
            errdefer p.rmsnorm_batch.deinit();

            p.rope = try kernels.RopeMulti.init(ctx);
            errdefer p.rope.deinit();

            p.rope_batch = try kernels.RopeMultiBatch.init(ctx);
            errdefer p.rope_batch.deinit();

            p.swiglu = try kernels.Swiglu.init(ctx);
            errdefer p.swiglu.deinit();

            p.residual = try kernels.ResidualAdd.init(ctx);
            errdefer p.residual.deinit();

            // Build the four attention kernels, which read the KV regions at the storage element.
            p.attention = try kernels.Attention.initKv(ctx, kv_dtype);
            errdefer p.attention.deinit();

            p.attention_batch = try kernels.AttentionBatch.initKv(ctx, kv_dtype);
            errdefer p.attention_batch.deinit();

            p.attention_multi = try kernels.AttentionMultiStream.initKv(ctx, kv_dtype);
            errdefer p.attention_multi.deinit();

            p.attention_split = try kernels.AttentionSplit.initKv(ctx, kv_dtype);
            errdefer p.attention_split.deinit();

            // Build the narrowing of a narrow KV dtype.
            p.kv_commit = if (kv_dtype != .f32) try kernels.KvCommitF16.init(ctx) else null;
            return p;
        }

        /// Allocate the device scratch at `prefill_batch_max` rows, the list that
        /// `scratchVramBytes` sizes, then the logits rows and the rope factors. The caller frees
        /// a partial allocation.
        fn allocBuffers(self: *Self, refs: []const loader.TensorRef) InitError!void {
            const b = &self.buffers;
            const rows: u64 = self.prefill_batch_max;
            const f32_size: u64 = @sizeOf(f32);

            // One buffer for each activation that the family holds.
            var acts = b.acts.iterator();
            while (acts.next()) |entry| {
                const width = self.activation_widths.get(entry.key);
                if (width > 0) {
                    try entry.value.ensure(rows * width * f32_size);
                }
            }

            // The narrowed commit pair, which only a narrowing dispatch writes.
            if (self.kv_dtype != .f32) {
                try b.k_narrowed.ensure(rows * self.kvRowBytes());
                try b.v_narrowed.ensure(rows * self.kvRowBytes());
            }

            // The split records, then the logits rows of one call.
            try b.attn_records.ensure(self.attn_scratch.totalBytes());
            try b.logits.ensure(@as(u64, self.logit_rows_max) * self.card.vocab_size * f32_size);

            // The divisor of each rotated pair: the factor tensor of the family, or ones.
            const half = self.card.rope_dim_count / 2;
            try b.rope_factors.ensure(@as(u64, half) * f32_size);
            const factors = b.rope_factors.sliceTyped(f32, half);
            const held = if (Family.rope_freqs) |name| findOptional(refs, name) else null;
            if (held) |tensor| {
                const src = tensor_lookup.fp32View(tensor.*);
                assert(src.len >= half);
                @memcpy(factors, src[0..half]);
            } else {
                @memset(factors, 1.0);
            }
        }

        /// Allocate the host stages, the host logits, and the tables of the range: an empty store,
        /// a binding, and a KV index for each layer. The caller frees a partial allocation.
        fn allocHost(self: *Self) InitError!void {
            // The input stage holds one row per step row, and the output stage the logits rows.
            const rows: u64 = self.prefill_batch_max;
            const logits_elems = @as(usize, self.logit_rows_max) * self.card.vocab_size;
            const logits_bytes = @as(u64, logits_elems) * @sizeOf(f32);
            try self.commands.allocStages(rows * self.hiddenRowBytes(), logits_bytes);
            self.logits = try self.gpa.alloc(f32, logits_elems);

            // Allocate the tables of the range.
            self.layers = try self.gpa.alloc(LayerStore, self.n_layers);
            @memset(self.layers, .empty(self.commands.ctx));
            self.bindings = try self.gpa.alloc(layer_kernels.LayerBinding, self.n_layers);
            self.kv_index = try self.gpa.alloc(?u32, self.n_layers);

            // Give each layer its KV index. The rule counts what the walk counts, and the walk
            // fills the KV height of the store.
            var n_kv: u32 = 0;
            for (self.kv_index, 0..) |*index, j| {
                const layer = self.first_layer + @as(u32, @intCast(j));
                index.* = Family.kvIndex(&self.card, self.first_layer, layer);
                if (index.*) |kv| {
                    assert(kv == n_kv);
                    n_kv += 1;
                }
            }

            assert(n_kv == self.kv.layout.n_layers);
        }

        /// Free the host logits and the tables of `allocHost`. An empty slice frees nothing, and
        /// `commands` frees the stages.
        fn freeHost(self: *Self) void {
            self.gpa.free(self.kv_index);
            self.gpa.free(self.bindings);
            self.gpa.free(self.layers);
            self.gpa.free(self.logits);
        }

        /// Upload every weight of the range in the order of `uploadOrder`. A dry run of the same
        /// order sizes each arena segment first, so the arenas hold the views plus their padding.
        /// One upload stage carries every weight. The stage frees its buffers before `allocSlabs`.
        fn uploadWeights(self: *Self, refs: []const loader.TensorRef) InitError!void {
            const uploads = try self.uploadOrder(refs);
            defer self.gpa.free(uploads);

            const largest = try self.memory.plan(uploads, self.quant);
            const bytes = fused_weights.stageBytes(largest, self.commands.swap_chunk_bytes);
            fused_weights.logHeaps(self.commands.ctx, "before");
            var stage: kernels.UploadStage = .init(self.commands.ctx);
            defer stage.deinit();

            try stage.ensure(bytes.small, bytes.quants);
            for (uploads) |u| {
                try self.upload(&stage, u);
            }

            fused_weights.logHeaps(self.commands.ctx, "after");

            const end = self.first_layer + self.n_layers;
            const planned = weightsVramBytesRange(refs, self.card.n_layers, self.first_layer, end);
            self.memory.assertFull(planned);
        }

        /// Every weight of the range in upload order: each layer with its vectors, then its
        /// matrices, then the final norm and the lm_head. The dry run and the upload both walk this
        /// list, because the order fixes the arena offset of each weight. The caller frees it.
        fn uploadOrder(self: *const Self, refs: []const loader.TensorRef) InitError![]Upload {
            // A layer uploads each role once at most, and the head adds two weights.
            const roles_max = std.enums.values(Weight).len;
            var list: std.ArrayList(Upload) = try .initCapacity(
                self.gpa,
                self.n_layers * roles_max + 2,
            );
            errdefer list.deinit(self.gpa);

            for (0..self.n_layers) |j| {
                const index: u32 = @intCast(j);
                const layer = self.first_layer + index;
                const roles = Family.layerWeights(&self.card, layer);
                assert(roles.len <= roles_max);

                // The vectors first, then the matrices.
                for ([_]bool{ false, true }) |matrices| {
                    for (roles) |role| {
                        if (role.isMatrix() != matrices) continue;

                        const ref = try tensor_lookup.findRole(Family, refs, role, layer);
                        list.appendAssumeCapacity(.{
                            .site = .{ .layer = .{ .index = index, .role = role } },
                            .name = ref.name,
                            .tensor = &ref.tensor,
                        });
                    }
                }
            }

            // A tied model reuses the embedding table as the lm_head.
            const output = findOptional(refs, loader.head_tensor);
            const norm_name = loader.output_norm_tensor;
            const norm = try findTensor(refs, norm_name);
            list.appendAssumeCapacity(.{ .site = .output_norm, .name = norm_name, .tensor = norm });
            list.appendAssumeCapacity(.{
                .site = .lm_head,
                .name = if (output == null) loader.embed_tensor else loader.head_tensor,
                .tensor = output orelse self.embed,
            });
            return list.toOwnedSlice(self.gpa);
        }

        /// Upload one weight, and keep its view where the sets of its layer or of the head find
        /// it. Each view holds the bytes that the dry run of `WeightMemory.plan` placed for it.
        fn upload(self: *Self, stage: *kernels.UploadStage, u: Upload) InitError!void {
            const bytes = u.arenaBytes(self.quant);
            if (!u.isMatrix()) {
                const view = try self.memory.vector(stage, u.name, u.tensor);
                assert(view.range == bytes.small);
                switch (u.site) {
                    .layer => |at| self.layers[at.index].vectors.set(at.role, view),
                    .output_norm => self.head.norm = view,
                    .lm_head => unreachable, // The lm_head is a matrix.
                }
                return;
            }

            const weight = switch (u.site) {
                .lm_head => try self.memory.lmHead(stage, self.gpa, self.quant, u.name, u.tensor),
                else => try self.memory.matrix(stage, self.quant, u.name, u.tensor),
            };
            assert(weight.scales.range == bytes.small);
            assert(weight.quants.range == bytes.quants);
            switch (u.site) {
                .layer => |at| self.layers[at.index].matrices.set(at.role, weight),
                .lm_head => self.head.lm_head = weight,
                .output_norm => unreachable, // The final norm is a vector.
            }
        }

        /// Allocate the zeroed recurrent slabs of each layer without KV. The caller frees a partial
        /// allocation.
        fn allocSlabs(self: *Self) InitError!void {
            const ctx = self.commands.ctx;
            const slab = self.slabBytes();
            for (self.layers, self.kv_index) |*store, index| {
                if (index != null) continue;

                // Slot s of a slab starts at `s` times the bytes of one slot.
                try store.state.ensure(self.n_slots * slab.state);
                try store.conv.ensure(self.n_slots * slab.conv);
                try zeroDeviceBuffer(ctx, &store.state);
                try zeroDeviceBuffer(ctx, &store.conv);
            }
        }

        /// Free the recurrent slabs of each layer. An empty store frees nothing.
        fn freeSlabs(self: *Self) void {
            for (self.layers) |*store| {
                store.deinit();
            }
        }

        fn logInit(self: *const Self, refs: []const loader.TensorRef) void {
            const mib: f64 = 1024.0 * 1024.0;
            const end = self.first_layer + self.n_layers;
            const weights = weightsVramBytesRange(refs, self.card.n_layers, self.first_layer, end);
            log.debug(
                "init {t}: layers [{d}..{d}) ({d} kv), {d} slots, max_seq {d}, quant {t}, " ++
                    "weights {d:.0} MiB, kv {d:.0} MiB",
                .{
                    self.commands.gpu_runner,
                    self.first_layer,
                    end,
                    self.kv.layout.n_layers,
                    self.n_slots,
                    self.max_seq,
                    self.quant,
                    @as(f64, @floatFromInt(weights)) / mib,
                    @as(f64, @floatFromInt(self.kv.deviceBytes())) / mib,
                },
            );
            const m = &self.memory;
            log.debug(
                "weight arenas: small {d} B in {d} segments, quants {d} B in {d} segments, " ++
                    "held {d} B = views {d} B + padding {d} B (bound {d} B), planned {d} B, " ++
                    "device memory {d} B",
                .{
                    m.small.packing.capacityBytes(),
                    m.small.packing.fills.items.len,
                    m.quants.packing.capacityBytes(),
                    m.quants.packing.fills.items.len,
                    m.heldBytes(),
                    m.viewBytes(),
                    m.heldBytes() - m.viewBytes(),
                    m.paddingBound(),
                    weights,
                    m.small.memory_bytes + m.quants.memory_bytes,
                },
            );
            m.small.logSegments("small");
            m.quants.logSegments("quants");
        }

        // -- helpers: descriptor sets ---------------------------------------

        /// Make the descriptor pool at the exact count of the sets, then the head sets, then the
        /// sets of every layer through the bind pass. Destroying the pool frees every set.
        fn createSets(self: *Self) InitError!void {
            assert(!self.sets_fixed);

            // The guards of `checkPlan` proved that the split records fit one binding.
            const ctx = self.commands.ctx;
            const b = &self.buffers;
            assert(b.attn_records.capacity == self.attn_scratch.totalBytes());
            assert(b.attn_records.capacity <= ctx.buffer_limits.storage_range);

            // Describe the buffers of each layer and of the runner to the bind pass.
            const views = try self.gpa.alloc(fused_layer_kernels.LayerBuffers, self.n_layers);
            defer self.gpa.free(views);

            for (views, self.layers) |*view, *store| {
                view.* = store.buffers();
            }

            const input = self.bindInput(views);

            // State the head sets: the final norm, the lm_head, the split merge, and the narrowing
            // of the commit, from the f32 rows to the storage element.
            const x = bi(b.acts.get(.x));
            const x_norm = bi(b.acts.get(.x_norm));
            const lm_head = self.head.lm_head.?;
            const norm_infos = [_]vk.DescriptorBufferInfo{ x, self.head.norm.?, x_norm };
            const lm_head_infos = [_]vk.DescriptorBufferInfo{
                x_norm,
                lm_head.scales,
                lm_head.quants,
                bi(b.logits),
            };
            const merge_infos = [_]vk.DescriptorBufferInfo{
                bi(b.attn_records),
                bi(b.acts.get(.attn_out)),
            };
            const commit_infos = [_]vk.DescriptorBufferInfo{
                bi(b.acts.get(.k)),
                bi(b.acts.get(.v)),
                bi(b.k_narrowed),
                bi(b.v_narrowed),
            };
            const commits: u32 = @intFromBool(self.pipelines.kv_commit != null);
            const always: u32 = norm_infos.len + lm_head_infos.len + merge_infos.len;
            const head: fused_layer_kernels.SetCount = .{
                .sets = 3 + commits,
                .descriptors = always + commits * @as(u32, commit_infos.len),
            };

            // Size the pool to the head sets and to the layer sets that the count pass counts.
            const layer_sets = fused_layer_kernels.countLayerSets(Family, &input);
            try self.createPool(head, layer_sets);
            errdefer ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);

            // Make the head sets, then the layer sets, which equal the count of the count pass.
            const p = &self.pipelines;
            self.set_output_norm = try self.mkSet(p.rmsnorm.pipeline.set_layout, &norm_infos);
            self.set_lm_head = try self.mkSet(p.linear.pipeline.set_layout, &lm_head_infos);
            const merge_layout = p.attention_split.merge.set_layout;
            self.set_attention_merge = try self.mkSet(merge_layout, &merge_infos);
            if (p.kv_commit) |*k| {
                self.set_kv_commit = try self.mkSet(k.pipeline.set_layout, &commit_infos);
            }

            // The pool holds exactly the counted sets, so a bind pass that makes another count
            // leaves a layer unbound or fails an allocation.
            const pool = self.desc_pool;
            const bound = try fused_layer_kernels.bindLayers(Family, &input, pool, self.bindings);
            if (!std.meta.eql(bound, layer_sets)) return error.BindCountMismatch;
        }

        /// Create the descriptor pool for exactly the head sets and the layer sets.
        fn createPool(
            self: *Self,
            head: fused_layer_kernels.SetCount,
            layer_sets: fused_layer_kernels.SetCount,
        ) InitError!void {
            assert(head.sets >= 1);
            assert(layer_sets.sets >= self.n_layers);

            const pool_sizes = [_]vk.DescriptorPoolSize{.{
                .type = .storage_buffer,
                .descriptor_count = head.descriptors + layer_sets.descriptors,
            }};
            const pool_info: vk.DescriptorPoolCreateInfo = .{
                .max_sets = head.sets + layer_sets.sets,
                .pool_size_count = pool_sizes.len,
                .p_pool_sizes = &pool_sizes,
            };
            const ctx = self.commands.ctx;
            self.desc_pool = try ctx.vkd.createDescriptorPool(ctx.device, &pool_info, null);
            log.debug("descriptor pool: {d} head and {d} layer sets, {d} storage descriptors", .{
                head.sets,
                layer_sets.sets,
                pool_sizes[0].descriptor_count,
            });
        }

        fn mkSet(
            self: *Self,
            layout: vk.DescriptorSetLayout,
            infos: []const vk.DescriptorBufferInfo,
        ) InitError!vk.DescriptorSet {
            assert(!self.sets_fixed);

            const layouts = [_]vk.DescriptorSetLayout{layout};
            const ai: vk.DescriptorSetAllocateInfo = .{
                .descriptor_pool = self.desc_pool,
                .descriptor_set_count = 1,
                .p_set_layouts = &layouts,
            };
            var set: vk.DescriptorSet = .null_handle;
            const ctx = self.commands.ctx;
            try ctx.vkd.allocateDescriptorSets(ctx.device, &ai, @ptrCast(&set));
            kernels.updateStorageBindings(ctx, set, infos);
            return set;
        }

        /// The buffers and the pipelines of the runner, as the bind pass reads them.
        fn bindInput(
            self: *Self,
            views: []const fused_layer_kernels.LayerBuffers,
        ) fused_layer_kernels.BindInput {
            assert(views.len == self.n_layers);

            // Name the buffer of each activation that the family holds.
            const b = &self.buffers;
            var acts: std.EnumArray(Activation, ?vk.DescriptorBufferInfo) = .initFill(null);
            var entries = acts.iterator();
            while (entries.next()) |entry| {
                const buffer = b.acts.get(entry.key);
                assert((buffer.capacity > 0) == (self.activation_widths.get(entry.key) > 0));
                if (buffer.capacity > 0) {
                    entry.value.* = bi(buffer);
                }
            }

            // Point the input at the buffers and the pipelines of the runner.
            const p = &self.pipelines;
            return .{
                .card = &self.card,
                .ctx = self.commands.ctx,
                .first_layer = self.first_layer,
                .layers = views,
                .kv_index = self.kv_index,
                .widths = &self.activation_widths,
                .acts = acts,
                .rope_freqs = bi(b.rope_factors),
                .attn_records = bi(b.attn_records),
                .kv = &self.kv,
                .k_linear = &p.linear,
                .k_rmsnorm = &p.rmsnorm,
                .k_rope = &p.rope,
                .k_swiglu = &p.swiglu,
                .k_residual = &p.residual,
                .op_resources = &self.op_resources,
                .attn = p.attnKernels(null),
            };
        }

        // -- helpers: recording ---------------------------------------------

        /// The layer kernel set of one record call of `kind`. It reads `input` alone, so the decode
        /// ring builds it from its key. Every KV layer records the one attention plan of the step.
        fn layerKernels(
            self: *Self,
            comptime kind: layer_kernels.StepKind,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            input: FusedLayerKernels(kind).Step,
        ) FusedLayerKernels(kind) {
            assert(self.sets_fixed);

            // Plan the attention of the step. The rows of a decode step and of a prefill chunk sit
            // in slot 0, and the rows of a chunk stay inside its window.
            const max_splits = self.attn_scratch.max_splits;
            const attn_plan = switch (kind) {
                .decode => attn_split.planDecode(input.pos + 1),
                .decode_batch => plan: {
                    assert(input.rows.len <= self.prefill_batch_max);
                    break :plan attn_record.planStep(input.rows, max_splits);
                },
                .prefill_chunk => plan: {
                    assert(input.b <= self.prefill_batch_max);
                    assert(input.b <= attn_split.max_plan_rows);
                    assert(@as(u64, input.start_pos) + input.b <= self.max_seq);
                    break :plan attn_split.planPrefill(0, input.start_pos, input.b, max_splits);
                },
            };

            // The shader reads `slot * stride` elements into the region of a set, and the commit
            // copies step a slot in bytes. Both units must name one slot.
            const stride: u32 = self.kv.layout.streamStrideElems();
            assert(self.kv_dtype.rowBytes(stride) == self.kv.layout.slotRegionBytes());

            // Point the value at the state of the runner.
            const p = &self.pipelines;
            return .{
                .card = &self.card,
                .vkd = &self.commands.ctx.vkd,
                .cb = cb,
                .prof = prof,
                .first_layer = self.first_layer,
                .bindings = self.bindings,
                .kv_index = self.kv_index,
                .widths = &self.activation_widths,
                .recurrent = Family.recurrentElems(&self.card),
                .quant = self.quant,
                .k_linear = &p.linear,
                .k_linear_batch = &p.linear_batch,
                .k_linear_q4 = if (p.linear_q4) |*k| k else null,
                .k_linear_batch_q4 = if (p.linear_batch_q4) |*k| k else null,
                .k_linear_gemm = self.gemmKernel(),
                .k_rmsnorm = &p.rmsnorm,
                .k_rmsnorm_batch = &p.rmsnorm_batch,
                .k_rope = &p.rope,
                .k_rope_batch = &p.rope_batch,
                .k_swiglu = &p.swiglu,
                .k_residual = &p.residual,
                .op_resources = &self.op_resources,
                .kv = &self.kv,
                .commit = self.kvCommit(),
                .attn = p.attnKernels(prof),
                .attn_merge = self.set_attention_merge,
                .attn_plan = attn_plan,
                .attn_shape = self.attnShape(stride),
                .step = input,
            };
        }

        /// The GEMM of the runner at its quant. Null: the family records no GEMM, or the weights
        /// are Q4_0, because the GEMM has no Q4_0 form.
        fn gemmKernel(self: *Self) ?*kernels.LinearQ80Gemm {
            if (self.quant != .q8_0) return null;

            return if (self.pipelines.linear_gemm) |*k| k else null;
        }

        /// The commit kernel, its set, and the two scratch pairs, for `recordKvCommit`.
        fn kvCommit(self: *Self) fused_commands.KvCommit {
            const b = &self.buffers;
            return .{
                .kernel = if (self.pipelines.kv_commit) |*k| k else null,
                .set = self.set_kv_commit,
                .kv_dtype = self.kv_dtype,
                .row_elems = self.card.n_kv_heads * self.card.head_dim,
                .rows_max = self.prefill_batch_max,
                .scratch = .{ .k = b.acts.get(.k).buffer, .v = b.acts.get(.v).buffer },
                .narrowed = .{ .k = b.k_narrowed.buffer, .v = b.v_narrowed.buffer },
            };
        }

        fn attnShape(self: *const Self, stream_stride_elems: u32) kernels.AttnShape {
            return .{
                .n_heads = self.card.n_heads,
                .n_kv_heads = self.card.n_kv_heads,
                .head_dim = self.card.head_dim,
                .stream_stride_elems = stream_stride_elems,
                .max_splits = self.attn_scratch.max_splits,
            };
        }

        /// Open `cb` and its timestamps, then copy `rows` input rows from `stage_in` to `x`.
        fn openRecord(
            self: *Self,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            rows: u32,
            one_time: bool,
        ) !void {
            const vkd = self.commands.ctx.vkd;
            try vkd.resetCommandBuffer(cb, .{});
            try vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = one_time } });
            prof.begin(cb);

            const stage_in = self.commands.stage_in.buffer;
            const x = self.buffers.acts.get(.x).buffer;
            copyHostToDevice(vkd, cb, stage_in, x, rows * self.hiddenRowBytes());
            barrierTransferToShader(vkd, cb);
            prof.mark(cb, .initOne(.stage_in), null);
        }

        /// Record each layer of the range through one kernel set, built from the step input.
        fn recordLayers(
            self: *Self,
            comptime kind: layer_kernels.StepKind,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            input: FusedLayerKernels(kind).Step,
        ) void {
            var kernel_set = self.layerKernels(kind, cb, prof, input);
            for (0..self.n_layers) |j| {
                if (comptime core_log.hot) {
                    log_dispatch.debug("record layer {d}", .{j});
                }

                const layer = self.first_layer + @as(u32, @intCast(j));
                Family.forward(FusedLayerKernels(kind), &kernel_set, layer);
            }
        }

        /// Record the head of row 0: the final norm, the lm_head, and the logits readback. The
        /// shader-to-transfer barrier covers the readback.
        fn recordRowHead(self: *Self, cb: vk.CommandBuffer, prof: *gpu_profile.GpuProfile) void {
            const vkd = self.commands.ctx.vkd;
            const card = &self.card;
            const set = self.set_output_norm;
            self.pipelines.rmsnorm.recordInto(cb, set, card.embedding_length, card.rms_norm_eps);
            barrierShader(vkd, cb);
            prof.mark(cb, .initOne(.final_norm), null);
            self.recordHeadMatmul(cb, 1);
            barrierShaderToTransfer(vkd, cb);
            prof.mark(cb, .initOne(.lm_head), null);
            self.recordReadback(cb, prof, 1);
        }

        /// Record the head of a step of `n` rows: the final norm of every row, then the lm_head and
        /// the readback of the want-logits suffix from row `first`.
        fn recordBatchHead(
            self: *Self,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            n: u32,
            first: u32,
        ) void {
            assert(first < n);

            // Norm every row.
            const vkd = self.commands.ctx.vkd;
            const card = &self.card;
            const set = self.set_output_norm;
            const hidden = card.embedding_length;
            self.pipelines.rmsnorm_batch.recordInto(cb, set, n, hidden, card.rms_norm_eps);

            // Compact the suffix to the front through `x`, which is free after the final norm.
            // `cmdCopyBuffer` forbids overlap, and a small step can have first < n_logits.
            const n_logits = n - first;
            if (first > 0) {
                const x = self.buffers.acts.get(.x).buffer;
                const x_norm = self.buffers.acts.get(.x_norm).buffer;
                const row_bytes = self.hiddenRowBytes();
                const suffix_bytes = n_logits * row_bytes;
                barrierShaderToTransfer(vkd, cb);
                copyDeviceToDevice(vkd, cb, x_norm, first * row_bytes, x, 0, suffix_bytes);
                barrierTransferToTransfer(vkd, cb);
                copyDeviceToDevice(vkd, cb, x, 0, x_norm, 0, suffix_bytes);
                barrierTransferToShader(vkd, cb);
            } else {
                barrierShader(vkd, cb);
            }

            // Run the lm_head over the suffix, and read the logits back.
            prof.mark(cb, .initOne(.final_norm), null);
            self.recordHeadMatmul(cb, n_logits);
            barrierShaderToTransfer(vkd, cb);
            prof.mark(cb, .initOne(.lm_head), null);
            self.recordReadback(cb, prof, n_logits);
        }

        /// Record the lm_head over `rows` rows. The row count picks the kernel and the quant picks
        /// its form, as in the layer kernel set.
        fn recordHeadMatmul(self: *Self, cb: vk.CommandBuffer, rows: u32) void {
            const p = &self.pipelines;
            const set = self.set_lm_head;
            const k = self.card.embedding_length;
            const n = self.card.vocab_size;
            const gemm = if (rows >= gemm_min_batch) self.gemmKernel() else null;
            if (comptime core_log.hot) {
                const name = if (rows == 1) "reduce" else if (gemm != null) "gemm" else "batch";
                const args = .{ name, self.quant, rows, k, n };
                log_dispatch.debug("  matmul lm_head {s} ({t}): b {d} k {d} n {d}", args);
            }

            if (rows == 1) {
                switch (self.quant) {
                    .q8_0 => p.linear.recordInto(cb, set, k, n),
                    .q4_0 => p.linear_q4.?.recordInto(cb, set, k, n),
                }
            } else if (gemm) |kernel| {
                kernel.recordInto(cb, set, rows, k, n);
            } else {
                switch (self.quant) {
                    .q8_0 => p.linear_batch.recordInto(cb, set, rows, k, n),
                    .q4_0 => p.linear_batch_q4.?.recordInto(cb, set, rows, k, n),
                }
            }
        }

        /// Copy `n_logits` logits rows to `stage_out`. The caller recorded the barrier.
        fn recordReadback(
            self: *Self,
            cb: vk.CommandBuffer,
            prof: *gpu_profile.GpuProfile,
            n_logits: u32,
        ) void {
            const bytes = @as(vk.DeviceSize, n_logits) * self.card.vocab_size * @sizeOf(f32);
            const logits = self.buffers.logits.buffer;
            const stage_out = self.commands.stage_out.buffer;
            copyDeviceToHost(self.commands.ctx.vkd, cb, logits, stage_out, bytes);
            prof.mark(cb, .initOne(.readback), null);
        }

        // -- helpers: slots -------------------------------------------------

        /// Fill the slabs of `slot` with zeros in every recurrent layer, in one submit.
        fn zeroSlot(self: *Self, slot: u32) !void {
            const Rec = struct {
                runner: *const Self,
                slot: u32,

                pub fn record(r: @This(), cb: vk.CommandBuffer) kernels.Recorded {
                    const vkd = r.runner.commands.ctx.vkd;
                    const slab = r.runner.slabBytes();
                    for (r.runner.layers, r.runner.kv_index) |*store, index| {
                        if (index != null) continue;

                        const state = store.state.buffer;
                        const conv = store.conv.buffer;
                        vkd.cmdFillBuffer(cb, state, r.slot * slab.state, slab.state, 0);
                        vkd.cmdFillBuffer(cb, conv, r.slot * slab.conv, slab.conv, 0);
                    }

                    return .{};
                }
            };

            // Free the decode ring, then submit the fill.
            self.commands.invalidateDecode();
            try kernels.submitOnce(self.commands.ctx, .fill, Rec{ .runner = self, .slot = slot });
        }

        /// Copy the slabs of `source` into `target` in every recurrent layer, then every KV
        /// region, in one submit.
        fn copySlot(self: *Self, source: u32, target: u32) !void {
            var kv_scratch: [kernels.KvStore.max_segments]kernels.Copy = undefined;
            const kv_copies = self.kv.cloneCopies(source, target, &kv_scratch);
            assert(kv_copies.len == 2 * @as(usize, self.kv.layout.n_layers));

            const Rec = struct {
                runner: *const Self,
                source: u32,
                target: u32,
                kv_copies: []const kernels.Copy,

                pub fn record(r: @This(), cb: vk.CommandBuffer) kernels.Recorded {
                    const vkd = r.runner.commands.ctx.vkd;
                    const slab = r.runner.slabBytes();
                    var out: kernels.Recorded = .{};
                    for (r.runner.layers, r.runner.kv_index) |*store, index| {
                        if (index != null) continue;

                        r.copyRegion(vkd, cb, store.state.buffer, slab.state);
                        r.copyRegion(vkd, cb, store.conv.buffer, slab.conv);
                        out.copies += 2;
                        out.bytes += slab.state + slab.conv;
                    }

                    for (r.kv_copies) |c| {
                        const size = c.size;
                        copyDeviceToDevice(vkd, cb, c.src, c.src_offset, c.dst, c.dst_offset, size);
                        out.bytes += size;
                    }

                    out.copies += @intCast(r.kv_copies.len);
                    return out;
                }

                /// Copy the region of the source slot in `buffer` to the region of the target.
                fn copyRegion(
                    r: @This(),
                    vkd: vk.DeviceWrapper,
                    cb: vk.CommandBuffer,
                    buffer: vk.Buffer,
                    bytes: vk.DeviceSize,
                ) void {
                    const from = r.source * bytes;
                    const to = r.target * bytes;
                    copyDeviceToDevice(vkd, cb, buffer, from, buffer, to, bytes);
                }
            };

            // Free the decode ring, then submit the copies.
            self.commands.invalidateDecode();
            try kernels.submitOnce(self.commands.ctx, .clone, Rec{
                .runner = self,
                .source = source,
                .target = target,
                .kv_copies = kv_copies,
            });
        }

        /// Move the state of `slot` over its first `n` rows through the stage of the direction.
        fn stageSlot(self: *Self, slot: u32, n: u32, side: kernels.StageSide) !kernels.SwapSplit {
            const stage = try self.commands.slotStage(side, self.snapshotBytes(self.max_seq));
            var copies: [kernels.KvStore.max_segments]kernels.Copy = undefined;
            const list = self.slotCopies(slot, n, side == .to_host, stage.buffer, &copies);
            return self.commands.walkSlot(stage, list, side);
        }

        /// The copy list of the state of `slot` over its first `n` rows, in blob order. `to_host`
        /// true saves into `stage`, and false restores out of it, so the host offsets agree.
        fn slotCopies(
            self: *Self,
            slot: u32,
            n: u32,
            to_host: bool,
            stage: vk.Buffer,
            out: []kernels.Copy,
        ) []kernels.Copy {
            // The direction fixes the legal stage.
            assert(to_host == (stage == self.commands.kv_readback_stage.buffer));
            assert(slot < self.kv.layout.n_slots);
            assert(n <= self.kv.layout.window);
            assert(out.len >= 2 * @as(usize, self.n_layers));

            // Walk the layers in model order, and advance the host offset per copy.
            const kv_bytes: vk.DeviceSize = @as(vk.DeviceSize, n) * self.kv.layout.rowBytes();
            const slab = self.slabBytes();
            var off: vk.DeviceSize = 0;
            var kv_total: u64 = 0;
            var c: usize = 0;
            for (self.layers, self.kv_index) |*store, index| {
                if (index) |kv_layer| {
                    // One K and one V copy of the first `n` rows of the slot.
                    for ([_]kv_index.KvKind{ .k, .v }) |t| {
                        const dev = self.kv.rowAt(kv_layer, t, slot, 0);
                        out[c] = stagedCopy(to_host, stage, off, dev.buffer, dev.offset, kv_bytes);
                        c += 1;
                        off += kv_bytes;
                        kv_total += kv_bytes;
                    }
                } else {
                    // The region of the slot in the recurrent slab, then in the conv slab.
                    const state = store.state.buffer;
                    const conv = store.conv.buffer;
                    out[c] = stagedCopy(to_host, stage, off, state, slot * slab.state, slab.state);
                    off += slab.state;
                    out[c + 1] = stagedCopy(to_host, stage, off, conv, slot * slab.conv, slab.conv);
                    off += slab.conv;
                    c += 2;
                }
            }

            // The KV half is the blob that the layout frames, and the walk fills the whole blob.
            assert(kv_total == self.kv.layout.blobBytes(n));
            assert(off == self.snapshotBytes(n));
            return out[0..c];
        }

        fn logSwap(
            self: *const Self,
            verb: []const u8,
            slot: u32,
            n_rows: u32,
            t0: std.Io.Timestamp,
        ) void {
            const split = self.last_swap_split;
            const bytes: f64 = @floatFromInt(self.snapshotBytes(n_rows));
            log_kv.debug(
                "{s} slot {d}: {d} rows, {d:.0} MiB, {d:.0} ms ({d:.1} dma + {d:.1} memcpy)",
                .{
                    verb,
                    slot,
                    n_rows,
                    bytes / (1024.0 * 1024.0),
                    core_log.msSince(t0),
                    split.dma_ms,
                    split.memcpy_ms,
                },
            );
        }

        // -- helpers: rules -------------------------------------------------

        /// The KV layers of the range `[first, L)`. The KV store and the attention plan of a step
        /// need one, so a range without one is refused.
        fn countKvLayers(card: *const loader.ModelCard, first: u32) error{NoKvLayerInRange}!u32 {
            assert(first < card.n_layers);

            var n_kv: u32 = 0;
            var layer = first;
            while (layer < card.n_layers) : (layer += 1) {
                if (Family.kvIndex(card, first, layer) != null) {
                    n_kv += 1;
                }
            }

            if (n_kv == 0) return error.NoKvLayerInRange;

            assert(n_kv <= card.n_layers - first);
            return n_kv;
        }

        /// The device bytes of one slot over the model layers `[from..to)` at `window` rows, from
        /// the family rules.
        fn rangeSlotBytes(
            card: *const loader.ModelCard,
            kv_dtype: dtype.KvDtype,
            window: u32,
            from: u32,
            to: u32,
        ) RangeSlotBytes {
            assert(from <= to);
            assert(to <= card.n_layers);
            assert(window >= 1);

            const row: u64 = kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
            const elems = Family.recurrentElems(card);
            const slab: u64 = (@as(u64, elems.state) + elems.conv) * @sizeOf(f32);
            var out: RangeSlotBytes = .{ .kv = 0, .recurrent = 0 };
            var layer = from;
            while (layer < to) : (layer += 1) {
                if (Family.kvIndex(card, from, layer) != null) {
                    out.kv += 2 * @as(u64, window) * row;
                } else {
                    out.recurrent += slab;
                }
            }

            return out;
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

        /// The split attention records of this model. The sizing and the allocation read this one
        /// value.
        fn attnSplitScratch(card: loader.ModelCard) attn_split.Scratch {
            return attn_split.Scratch.init(card.n_heads, card.head_dim, card.context_length);
        }

        /// The bytes of one slot in the recurrent slab and in the conv slab of a recurrent layer.
        fn slabBytes(self: *const Self) struct { state: vk.DeviceSize, conv: vk.DeviceSize } {
            const elems = Family.recurrentElems(&self.card);
            return .{
                .state = @as(vk.DeviceSize, elems.state) * @sizeOf(f32),
                .conv = @as(vk.DeviceSize, elems.conv) * @sizeOf(f32),
            };
        }

        /// Bytes of one KV row at `kv_dtype`. It sizes the commit scratch and every commit copy.
        fn kvRowBytes(self: *const Self) vk.DeviceSize {
            return self.kv_dtype.rowBytes(self.card.n_kv_heads * self.card.head_dim);
        }

        /// Bytes of one f32 activation row of `x`.
        fn hiddenRowBytes(self: *const Self) vk.DeviceSize {
            return @as(vk.DeviceSize, self.card.embedding_length) * @sizeOf(f32);
        }

        /// Look up the embedding of `token` into row `row` of `stage_in`.
        fn embedRow(self: *Self, row: usize, token: u32) void {
            const d = self.card.embedding_length;
            const x_host = self.commands.stage_in.sliceTyped(f32, (row + 1) * d);
            naive.embedLookup(self.embed.*, token, d, x_host[row * d ..][0..d]);
        }
    };
}

// -- helpers ----------------------------------------------------------------

fn bi(b: anytype) vk.DescriptorBufferInfo {
    return .{ .buffer = b.buffer, .offset = 0, .range = b.capacity };
}

/// Fill a device buffer with zeros in one `submitOnce`.
fn zeroDeviceBuffer(ctx: *const vulkan_ctx.Context, b: *const kernels.DeviceBuffer) !void {
    const Rec = struct {
        buf: *const kernels.DeviceBuffer,

        pub fn record(r: @This(), cb: vk.CommandBuffer) kernels.Recorded {
            r.buf.ctx.vkd.cmdFillBuffer(cb, r.buf.buffer, 0, r.buf.capacity, 0);
            return .{};
        }
    };
    try kernels.submitOnce(ctx, .fill, Rec{ .buf = b });
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

/// The refusal of a device failure during a fused init, or null for another error. The plan
/// passed the preflight, so the text names the retry and not a smaller plan.
fn deviceFailureText(err: anyerror) ?[]const u8 {
    const retry = " during the init of the fused runner, after the plan passed the VRAM " ++
        "preflight. Run the command again. When the log shows a device removal, " ++
        "first restart the virtual machine of the driver, for example with `wsl --shutdown`";
    return switch (err) {
        error.DeviceLost => "device: the driver removed the device" ++ retry,
        error.OutOfDeviceMemory => "device: the driver refused a device allocation" ++ retry,
        else => null,
    };
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;
const llama_layer = @import("llama_layer.zig");
const qwen_layer = @import("qwen_layer.zig");

/// The families of the tests below.
const test_families = .{ llama_layer, qwen_layer };

const testCard = @import("test_fakes.zig").testCard;

test "kv cost: the bytes of a slot are the total of the cost vector of the family" {
    inline for (test_families) |Family| {
        const Runner = FusedRunner(Family);
        const card = testCard(Family.arch);
        for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
            const vector = kv_cost.costVector(card, kv_dtype);
            try testing.expectEqual(vector.total(), Runner.kvCostModel(card, kv_dtype));
        }

        // A narrower dtype halves the token size and keeps the fixed state, so the planner
        // stays monotone in cost.
        const wide = Runner.kvCostModel(card, .f32);
        const half = Runner.kvCostModel(card, .f16);
        try testing.expectEqual(wide.kv_bytes_per_token, 2 * half.kv_bytes_per_token);
        try testing.expectEqual(wide.fixed_state_bytes_per_slot, half.fixed_state_bytes_per_slot);
        try testing.expect(half.kv_bytes_per_token >= 1);
    }

    // The anchors at f32: 262144 token bytes on the Llama card, and 24576 token bytes plus
    // 20201472 fixed bytes a slot on the Qwen card.
    const llama = FusedRunner(llama_layer).kvCostModel(testCard(.llama), .f32);
    const qwen = FusedRunner(qwen_layer).kvCostModel(testCard(.qwen35), .f32);
    try testing.expectEqual(@as(u64, 262_144), llama.kv_bytes_per_token);
    try testing.expectEqual(@as(u64, 0), llama.fixed_state_bytes_per_slot);
    try testing.expectEqual(@as(u64, 24_576), qwen.kv_bytes_per_token);
    try testing.expectEqual(@as(u64, 20_201_472), qwen.fixed_state_bytes_per_slot);
}

test "range: the slot bytes of every range equal the span of the planner" {
    inline for (test_families) |Family| {
        const Runner = FusedRunner(Family);
        const card = testCard(Family.arch);
        const row_elems = card.n_kv_heads * card.head_dim;
        for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
            for ([_]u32{ 512, 8192 }) |window| {
                const kv: kv_budget.KvStructure = .{
                    .n_slots = 4,
                    .window = window,
                    .costs = kv_cost.costVector(card, kv_dtype),
                };
                try expectSuffixBytes(Runner, &card, kv, kv_dtype, row_elems);

                // The two halves of every cut partition the slot size of the model.
                const whole = Runner.rangeSlotBytes(&card, kv_dtype, window, 0, card.n_layers);
                var cut: u32 = 0;
                while (cut <= card.n_layers) : (cut += 1) {
                    const head = Runner.rangeSlotBytes(&card, kv_dtype, window, 0, cut);
                    const tail = Runner.rangeSlotBytes(&card, kv_dtype, window, cut, card.n_layers);
                    try testing.expectEqual(whole.total(), head.total() + tail.total());
                }
            }
        }
    }
}

/// Check every suffix `[first, L)` against the planner span, the KV layout, and the blob of n
/// rows.
fn expectSuffixBytes(
    comptime Runner: type,
    card: *const loader.ModelCard,
    kv: kv_budget.KvStructure,
    kv_dtype: dtype.KvDtype,
    row_elems: u32,
) !void {
    var first: u32 = 0;
    while (first < card.n_layers) : (first += 1) {
        // The family rules and the cost vector size one range alike.
        const slot = Runner.rangeSlotBytes(card, kv_dtype, kv.window, first, card.n_layers);
        try testing.expectEqual(kv_budget.slotSpanBytes(kv, first, card.n_layers), slot.total());

        // The KV half is what the regions of the range hold for one slot. A blob of n rows holds
        // the KV half of the size at n rows.
        const n_kv = try Runner.countKvLayers(card, first);
        const layout = kv_index.Layout.init(kv.n_slots, kv.window, row_elems, n_kv, kv_dtype);
        try testing.expectEqual(layout.slotBytes(), slot.kv);
        try testing.expectEqual(kv.n_slots * slot.kv, layout.totalBytes());
        for ([_]u32{ 1, 37, kv.window }) |n_rows| {
            const blob = Runner.rangeSlotBytes(card, kv_dtype, n_rows, first, card.n_layers);
            try testing.expectEqual(layout.blobBytes(n_rows), blob.kv);
        }
    }
}

test "range: a Qwen range past the last attention layer is refused" {
    // Layer 24 of a 25-layer card at interval 4 is a GDN layer, so the range [24, 25) holds no KV.
    const Qwen = FusedRunner(qwen_layer);
    var card = testCard(.qwen35);
    card.n_layers = 25;
    try testing.expectError(error.NoKvLayerInRange, Qwen.countKvLayers(&card, 24));
    try testing.expectEqual(@as(u32, 1), try Qwen.countKvLayers(&card, 23));
    try testing.expectEqual(@as(u32, 6), try Qwen.countKvLayers(&card, 0));
}

test "weights: the whole model is the range at first zero" {
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
    const untied = [_]loader.TensorRef{
        .{ .name = "blk.0.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.1.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.2.ffn_down.weight", .tensor = layer_t },
        .{ .name = "blk.3.ffn_down.weight", .tensor = layer_t },
        .{ .name = "output_norm.weight", .tensor = layer_t },
        .{ .name = "token_embd.weight", .tensor = head_t },
        .{ .name = "output.weight", .tensor = layer_t },
    };

    inline for (test_families) |Family| {
        const Runner = FusedRunner(Family);

        // The tied head includes the embedding table, which init uploads as the lm_head. So the
        // figure is every tensor the model carries.
        const whole = Runner.weightsVramBytesRange(&tied, 4, 0, 4);
        try testing.expectEqual(@as(u64, 4 * 16 + 16 + 32), whole);

        // A suffix drops the layers before it and keeps the head, which the suffix runs.
        const suffix = Runner.weightsVramBytesRange(&tied, 4, 2, 4);
        try testing.expectEqual(@as(u64, 2 * 16 + 48), suffix);
        try testing.expectEqual(@as(u64, 48), Runner.weightsVramBytesRange(&tied, 4, 4, 4));

        // The layer halves of a cut partition the whole, once the head leaves the double count.
        const head_bytes = Runner.weightsVramBytesRange(&tied, 4, 0, 0);
        var k: u32 = 0;
        while (k <= 4) : (k += 1) {
            const halves = Runner.weightsVramBytesRange(&tied, 4, 0, k) +
                Runner.weightsVramBytesRange(&tied, 4, k, 4);
            try testing.expectEqual(whole + head_bytes, halves);
        }

        // Untied: `output.weight` is the head, and the embedding table on the host adds
        // nothing.
        const untied_whole = Runner.weightsVramBytesRange(&untied, 4, 0, 4);
        try testing.expectEqual(@as(u64, 4 * 16 + 16 + 16), untied_whole);
    }
}

test "scratch: a narrower dtype adds only the narrowed commit pair" {
    inline for (test_families) |Family| {
        const Runner = FusedRunner(Family);
        const card = testCard(Family.arch);
        const rows: u64 = Runner.prefill_rows_max;
        const pair: u64 = rows * 2 * dtype.KvDtype.f16.rowBytes(card.n_kv_heads * card.head_dim);
        const wide = Runner.scratchVramBytes(card, .f32);
        try testing.expectEqual(wide + pair, Runner.scratchVramBytes(card, .f16));
    }
}

test "scratch: the split records follow the declared context" {
    inline for (test_families) |Family| {
        // The records are the only context term, and they are f32 at every dtype.
        const Runner = FusedRunner(Family);
        const deep = testCard(Family.arch);
        var shallow = deep;
        shallow.context_length = limits.attn_split_len;
        const records_deep = Runner.attnSplitScratch(deep).totalBytes();
        const records_shallow = Runner.attnSplitScratch(shallow).totalBytes();
        for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
            const scratch_deep = Runner.scratchVramBytes(deep, kv_dtype);
            const scratch_shallow = Runner.scratchVramBytes(shallow, kv_dtype);
            try testing.expectEqual(records_deep - records_shallow, scratch_deep - scratch_shallow);
        }
    }

    // The Llama anchors: 4 rows x 32 heads x 512 splits x 130 elements of 4 bytes, on top of 256
    // activation rows of 51200 f32 elements.
    const Llama = FusedRunner(llama_layer);
    const llama_records = Llama.attnSplitScratch(testCard(.llama)).totalBytes();
    try testing.expectEqual(@as(u64, 34_078_720), llama_records);
    try testing.expectEqual(
        @as(u64, 52_428_800 + 34_078_720),
        Llama.scratchVramBytes(testCard(.llama), .f32),
    );

    // The Qwen anchors: 4 rows x 16 heads x 1024 splits x 258 elements of 4 bytes, on top of 128
    // activation rows of 60448 f32 elements, the GDN chunk rows included.
    const Qwen = FusedRunner(qwen_layer);
    const qwen_records = Qwen.attnSplitScratch(testCard(.qwen35)).totalBytes();
    try testing.expectEqual(@as(u64, 67_633_152), qwen_records);
    try testing.expectEqual(
        @as(u64, 30_949_376 + 67_633_152),
        Qwen.scratchVramBytes(testCard(.qwen35), .f32),
    );
}

test "init errors: each family names the refusals of its own guard and no other" {
    const gdn_refusals = [_][]const u8{
        "GdnHeadsUnsupported",
        "QwenHeadDimUnsupported",
        "GdnHeadDimUnsupported",
    };
    inline for (gdn_refusals) |name| {
        try testing.expect(!comptime hasError(FusedRunner(llama_layer).InitError, name));
        try testing.expect(comptime hasError(FusedRunner(qwen_layer).InitError, name));
    }
}

/// True when the error set `E` holds the error `name`.
fn hasError(comptime E: type, comptime name: []const u8) bool {
    for (@typeInfo(E).error_set.?) |err| {
        if (std.mem.eql(u8, err.name, name)) return true;
    }

    return false;
}

test "a device failure at init names the retry, and another error names nothing" {
    for ([_]anyerror{ error.DeviceLost, error.OutOfDeviceMemory }) |err| {
        const text = deviceFailureText(err) orelse return error.TestExpectedText;
        try testing.expect(std.mem.indexOf(u8, text, "Run the command again") != null);
        try testing.expect(std.mem.indexOf(u8, text, "wsl --shutdown") != null);
    }

    try testing.expectEqual(@as(?[]const u8, null), deviceFailureText(error.OutOfMemory));
    try testing.expectEqual(@as(?[]const u8, null), deviceFailureText(error.VramBudgetExceeded));
}
