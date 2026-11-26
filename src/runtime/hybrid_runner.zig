//! `HybridRunner(Family)` runs one forward pass of a family on two tiers. The CPU runner of the
//! family runs the host layers `[0..host_layers)`. The fused runner runs the GPU layers
//! `[host_layers..n_layers)`, the final norm, and the lm_head. The planner picks `host_layers` so
//! the GPU layers fit the VRAM budget.
//!
//! Comptime. `Family` is an entry of `resolve/family.zig`, and the runner names only its `Cpu` and
//! `Fused` runners. A build check stops the compiler when a build without fused ops includes it.
//!
//! Composition. `Family.Cpu.initPrefix` builds the host layers, and the hidden-row entries of
//! `Family.Fused` take the crossing rows. This file records no dispatch and holds no layer body.
//!
//! Crossing. The residual stream crosses the host-device boundary once per call. The CPU runner
//! leaves its rows in its own `x`, and the fused runner reads them in place of an embedding
//! lookup. The CPU runner runs the whole batch before the crossing. So each KV row is written
//! before attention reads it, and the GDN recurrence sees time order on both tiers.
//!
//! Interior splits. Zero host layers is the fused runner, and all host layers is the CPU runner.
//! The placement route sends both ends to those runners, so this runner accepts only
//! `1 <= host_layers < n_layers`. A second construction of a whole-model runner has no gate and no
//! token hash.
//!
//! Slot state. The state of a slot spans both tiers, so every slot hook drives both runners. The
//! CPU runner of a family with recurrent state sets `has_recurrent_state`, and this runner copies
//! the value. `resetSlotState` exists for every family and forwards only when the value is true.
//!
//! Blob. One blob is one walk in model order: the host regions first, the GPU regions after. A KV
//! layer writes its rows, and a GDN layer writes its whole recurrent and conv slabs. So the two
//! walks together write the bytes of a single-tier runner. The blob boundary is the byte count of
//! the host walk, and each call checks it against what the two runners report.
//!
//! Disk namespace. The CPU and GPU halves of a blob are not bit-identical. So a hybrid blob lives
//! in its own disk namespace. The tag of each family entry names the family, the host layer count,
//! the weight quant, and the KV dtype.
//!
//! Swap stages. A swap stage holds one chunk of the GPU half of a slot blob, or that whole half
//! when it is smaller. A plan with residency sizes two stages. A plan without residency sizes
//! zero, and the runner never swaps.
//!
//! Tests. The tests below need no device: the blob arithmetic of both families. The equivalence
//! tests in `tests/` and the hybrid gate run on a device.

const std = @import("std");
const build_options = @import("build_options");
const span = @import("../core/span.zig");
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const step_stats = @import("../core/step_stats.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const loader = @import("../io/gguf/loader.zig");
const naive = @import("../ops/naive.zig");
const vulkan_ctx = @import("vulkan_ctx.zig");
const kv_cost = @import("kv_cost.zig");
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const step_row = @import("step_row.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const Allocator = std.mem.Allocator;
const StepRow = step_row.StepRow;

const log = std.log.scoped(.runner);
const log_kv = std.log.scoped(.kv);

comptime {
    if (!build_options.use_vulkan_fused_ops) {
        @compileError("runtime/hybrid_runner.zig was included in a build without fused ops.");
    }
}

// -- types ------------------------------------------------------------------

/// The hybrid placement of one family: its CPU runner over the host layers, and its fused runner
/// over the GPU layers.
pub fn HybridRunner(comptime Family: type) type {
    return struct {
        const Self = @This();
        const Cpu = Family.Cpu;
        const Fused = Family.Fused;

        gpa: Allocator,

        /// The host layers `[0..host_layers)`. They live in `prefix_arena`, so `deinit` reclaims
        /// them in one call.
        prefix: Cpu,
        prefix_arena: std.heap.ArenaAllocator,
        /// The GPU layers `[host_layers..n_layers)`, the final norm, and the lm_head.
        suffix: *Fused,

        host_layers: u32,
        n_layers: u32,
        hidden: u32,
        card: loader.ModelCard,
        /// The element of every KV row on both tiers, so the blob keeps one row stride.
        kv_dtype: dtype.KvDtype,

        // -- session surface ------------------------------------------------
        // The session layer in `runtime/session.zig` reads the members below from
        // every runner family, and serve reads the backend tag.

        /// The compact logits rows of the fused runner, aliased: it runs the lm_head and owns them.
        logits: []f32,
        n_slots: u32,
        max_seq: u32,
        /// Rows one `prefillBatch` accepts. The bench and the serve chunker read this field on
        /// every runner.
        prefill_batch_max: u32,
        /// True when the plan sized both swap stages. A plan without them means this runner never
        /// swaps.
        swap_planned: bool,

        /// The value of the CPU runner. A false value keeps the session surface free of the
        /// recurrent hooks.
        pub const has_recurrent_state: bool = Cpu.has_recurrent_state;

        /// The `run` command batches the prompt through `prefillBatch`, as serve and bench-model
        /// do.
        pub const chunked_prefill: bool = true;

        // The row buffer of `prefillBatch` holds `Fused.prefill_rows_max` rows, and the serve
        // chunker never offers more than `max_prefill_chunk`.
        comptime {
            std.debug.assert(Fused.prefill_rows_max <= limits.max_prefill_chunk);
        }

        // A slot reset that reaches one tier only leaks the state of the other tier.
        comptime {
            const fused_recurrent = @hasDecl(Fused, "has_recurrent_state") and
                Fused.has_recurrent_state;
            if (has_recurrent_state != fused_recurrent) {
                @compileError("hybrid runner: " ++ @typeName(Cpu) ++ " and " ++
                    @typeName(Fused) ++ " must both hold recurrent state, or neither");
            }
        }

        // -- init -----------------------------------------------------------

        /// The errors of both runners and of the allocations of this file.
        pub const InitError = Allocator.Error || ErrorSetOf(Fused.initSuffix) ||
            ErrorSetOf(Cpu.initPrefix) || error{BlobLengthMismatch};

        /// What `checkPlan` proves about a plan, for the checks after both runners exist.
        const PlanCheck = struct {
            /// Bytes one slot occupies on the host layers.
            host_span: u64,
            /// Bytes one slot occupies on the GPU layers.
            device_span: u64,
            /// True when the plan sized both swap stages.
            swap_planned: bool,
        };

        /// What `init` takes beside the allocator, the context, and the model.
        pub const Options = struct {
            /// The layer split, the slots, the window, and the KV reservation of each tier.
            plan: kv_budget.LayerSplit,
            /// The caller owns the pool, which must outlive the runner.
            pool: *Pool,
            policy: parallel.Policy,
            /// Flag resolution has gated the dtype against the model. Both tiers share it.
            kv_dtype: dtype.KvDtype,
        };

        /// Build the fused runner and the CPU runner at the KV dtype of the caller, which both
        /// tiers share. The `logits` reservation must cover `n_slots` rows, because every decode
        /// row of a tick wants logits.
        pub fn init(
            gpa: Allocator,
            ctx: *const vulkan_ctx.Context,
            loaded: *const loader.Loaded,
            options: Options,
        ) InitError!*Self {
            const plan = options.plan;
            const kv_dtype = options.kv_dtype;

            // Check the plan before the first allocation.
            const card = loaded.card;
            const host_layers: u32 = plan.host_layers;
            const n_layers: u32 = plan.n_layers;
            const checked = checkPlan(card, plan, kv_dtype);

            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);

            // Build the fused runner first: its VRAM preflight refuses an over-budget placement
            // before the CPU runner uses host memory.
            const suffix = try Fused.initSuffix(
                gpa,
                ctx,
                loaded,
                suffixPlan(plan),
                host_layers,
                kv_dtype,
            );
            errdefer suffix.deinit();

            // Fill the fields. `prefix` stays undefined until the init below.
            self.* = .{
                .gpa = gpa,
                .prefix = undefined,
                .prefix_arena = .init(gpa),
                .suffix = suffix,
                .host_layers = host_layers,
                .n_layers = n_layers,
                .hidden = card.embedding_length,
                .card = card,
                .kv_dtype = kv_dtype,
                .logits = suffix.logits,
                .n_slots = plan.n_slots,
                .max_seq = plan.window,
                .prefill_batch_max = suffix.maxStepRows(),
                .swap_planned = checked.swap_planned,
            };
            errdefer self.prefix_arena.deinit();

            // Build the CPU runner. Every row of a call crosses, so it holds row scratch for the
            // widest fused step.
            self.prefix = try Cpu.initPrefix(self.prefix_arena.allocator(), loaded, host_layers, .{
                .plan = kv_budget.pinned(plan.n_slots, plan.window),
                .max_rows = suffix.maxStepRows(),
                .pool = options.pool,
                .policy = options.policy,
                .kv_dtype = kv_dtype,
            });

            // Check the built runners against the plan.
            try self.checkBuilt(checked);
            self.logBuilt();
            return self;
        }

        /// Each runner allocated what the planner sized for its layers: one slot at the window.
        /// A blob length that drifts from the plan is an error in every build mode.
        fn checkBuilt(self: *const Self, checked: PlanCheck) error{BlobLengthMismatch}!void {
            const window = self.max_seq;
            const host_held = self.prefix.evictSlotBytesFor(window);
            const device_held = self.suffix.evictSlotBytesFor(window);
            if (host_held != checked.host_span) return error.BlobLengthMismatch;
            if (device_held != checked.device_span) return error.BlobLengthMismatch;

            std.debug.assert(self.prefix.maxStepRows() >= self.suffix.maxStepRows());
            std.debug.assert(self.prefix.n_slots == self.n_slots);
            std.debug.assert(self.prefix.max_seq == self.max_seq);
            std.debug.assert(self.prefix.layerCount() == self.host_layers);
        }

        fn logBuilt(self: *const Self) void {
            log.debug(
                "init {s} hybrid: {d} of {d} layers on the host ({d} on gpu), {d} slots, " ++
                    "max_seq {d}, kv {t}",
                .{
                    Family.name,
                    self.host_layers,
                    self.n_layers,
                    self.n_layers - self.host_layers,
                    self.n_slots,
                    self.max_seq,
                    self.kv_dtype,
                },
            );
        }

        /// Check the plan against the per-layer cost vector. A plan built from another vector
        /// fails here, before the first allocation.
        fn checkPlan(
            card: loader.ModelCard,
            plan: kv_budget.LayerSplit,
            kv_dtype: dtype.KvDtype,
        ) PlanCheck {
            const host_layers: u32 = plan.host_layers;
            const n_layers: u32 = plan.n_layers;
            std.debug.assert(n_layers == card.n_layers);

            // Zero host layers or all of them belongs to a whole-model runner.
            std.debug.assert(host_layers >= 1);
            std.debug.assert(host_layers < n_layers);

            // The CPU runner indexes its slots by the decode width, the ceiling `hybridSlots`
            // enforces.
            std.debug.assert(plan.n_slots >= 1);
            std.debug.assert(plan.n_slots <= limits.max_batch);
            std.debug.assert(plan.window >= 1);

            // The KV bytes of each tier.
            const kv_struct: kv_budget.KvStructure = .{
                .n_slots = plan.n_slots,
                .window = plan.window,
                .costs = kv_cost.costVector(card, kv_dtype),
            };
            const host_span = kv_budget.slotSpanBytes(kv_struct, 0, host_layers);
            const device_span = kv_budget.slotSpanBytes(kv_struct, host_layers, n_layers);
            std.debug.assert(plan.kv_host == @as(u64, plan.n_slots) * host_span);
            std.debug.assert(plan.kv_vram == @as(u64, plan.n_slots) * device_span);
            std.debug.assert(
                host_span + device_span == kv_budget.slotSpanBytes(kv_struct, 0, n_layers),
            );
            std.debug.assert(plan.ram.host_kv == plan.kv_host);

            // The swap stages.
            const stage_one = blobSpanBytes(card, kv_dtype, plan.window, host_layers, n_layers);
            std.debug.assert(stage_one == device_span);
            const swap_planned = plan.ram.swap_stages == 2 * kv_budget.stageBytes(stage_one);
            std.debug.assert(swap_planned or plan.ram.swap_stages == 0);
            std.debug.assert(plan.ram.swap_stages <= 2 * limits.swap_stage_chunk_bytes);
            std.debug.assert(plan.ram.swap_stages * plan.n_slots <= 2 * plan.kv_vram);

            return .{
                .host_span = host_span,
                .device_span = device_span,
                .swap_planned = swap_planned,
            };
        }

        pub fn deinit(self: *Self) void {
            self.suffix.deinit();
            self.prefix_arena.deinit();
            self.gpa.destroy(self);
        }

        /// Rows of the compact logits buffer. The fused runner runs the lm_head, so the figure is
        /// its figure.
        pub fn logitRowsMax(self: *const Self) u32 {
            return self.suffix.logitRowsMax();
        }

        /// Max rows one `stepBatch` accepts. Both tiers hold row scratch for it.
        pub fn maxStepRows(self: *const Self) u32 {
            return self.suffix.maxStepRows();
        }

        /// The decode host times of the fused runner. They cover its GPU submits and leave out the
        /// CPU host layers.
        pub fn decodeHost(self: *Self) *step_stats.DecodeHost {
            return self.suffix.decodeHost();
        }

        // -- step -----------------------------------------------------------

        /// Run the CPU runner over `rows`: embed, then layers `[0..host_layers)`. The crossing
        /// rows stay in the `x` scratch of `prefix`. `embedLookup` is the naive function on every
        /// op backend, as in the `stepBatch` of the CPU runner.
        fn hostPrefix(self: *Self, rows: []const StepRow) void {
            const sp = span.begin(.host_prefix);
            defer sp.end();

            std.debug.assert(rows.len >= 1);
            std.debug.assert(rows.len <= self.prefix.max_rows);
            const d = self.hidden;
            const x = self.prefix.scratch.get(.x);
            for (rows, 0..) |r, i| {
                naive.embedLookup(self.prefix.embed.*, r.token, d, x[i * d ..][0..d]);
            }

            var l: u32 = 0;
            while (l < self.host_layers) : (l += 1) {
                self.prefix.layerForward(l, rows);
            }
        }

        /// The residual rows the CPU runner just wrote: the value that crosses the host-device
        /// boundary.
        fn crossing(self: *const Self, n_rows: usize) []const f32 {
            std.debug.assert(n_rows >= 1);
            std.debug.assert(n_rows <= self.prefix.max_rows);

            return self.prefix.scratch.get(.x)[0 .. n_rows * @as(usize, self.hidden)];
        }

        /// One decode step on slot 0. Returns the logits row, valid until the next call.
        pub fn step(self: *Self, token_id: u32, pos: u32) ![]f32 {
            if (pos >= self.max_seq) return error.PositionOutOfRange;

            const row = [_]StepRow{
                .{ .slot = 0, .token = token_id, .pos = pos, .want_logits = true },
            };
            self.hostPrefix(&row);
            return self.suffix.stepHidden(self.crossing(1), pos);
        }

        /// Advance a batch of rows under the row rules of the CPU runner, with compact logits for
        /// the `want_logits` suffix.
        pub fn stepBatch(self: *Self, rows: []const StepRow) !void {
            // The row rules bind the host layers too, so the check runs before the first of them.
            _ = try step_row.validateRows(rows, self.n_slots, self.maxStepRows(), self.max_seq);
            self.hostPrefix(rows);
            return self.suffix.stepBatchHidden(self.crossing(rows.len), rows);
        }

        /// Process `b` prompt tokens in one pass on slot 0: the batched host layers, one
        /// crossing, the batched GPU layers. Returns logits for the last token only. Callers chunk
        /// longer prompts.
        pub fn prefillBatch(self: *Self, tokens: []const u32, start_pos: u32) ![]f32 {
            std.debug.assert(tokens.len >= 1);

            // The row rules bind the host layers too, so the check runs before the first of them.
            if (tokens.len > self.maxStepRows()) return error.TooManyRows;

            var rows_buf: [Fused.prefill_rows_max]StepRow = undefined;
            const rows = step_row.chunkRows(&rows_buf, tokens, start_pos, .last_row);
            _ = try step_row.validateRows(rows, self.n_slots, self.maxStepRows(), self.max_seq);

            const b: u32 = @intCast(tokens.len);
            self.hostPrefix(rows);
            return self.suffix.prefillBatchHidden(self.crossing(b), b, start_pos);
        }

        /// The method of the CPU runner for a chunk that a later chunk follows. The GPU suffix
        /// keeps its head, so its dispatch list stays the one of `prefillBatch`.
        pub fn prefillBatchNoLogits(self: *Self, tokens: []const u32, start_pos: u32) !void {
            _ = try self.prefillBatch(tokens, start_pos);
        }

        // -- slot state -----------------------------------------------------
        // One slot spans both tiers, so every hook below drives both runners.

        /// Zero the recurrent state of a slot on both tiers for a fresh conversation.
        /// `Sessions.create` calls this on slot reuse. The KV needs no scrub, because attention
        /// reads only rewritten positions.
        pub fn resetSlotState(self: *Self, slot: u32) !void {
            std.debug.assert(slot < self.n_slots);

            if (comptime has_recurrent_state) {
                try self.prefix.resetSlotState(slot);
                try self.suffix.resetSlotState(slot);
            }
        }

        /// Fork slot state for a session clone across both tiers. One slot id spans them, because
        /// both carry the same slot count. `Sessions.clone` calls this.
        pub fn cloneSlotState(self: *Self, source_slot: u32, target_slot: u32, pos: u32) !void {
            std.debug.assert(source_slot < self.n_slots);
            std.debug.assert(target_slot < self.n_slots);
            std.debug.assert(source_slot != target_slot);
            std.debug.assert(pos <= self.max_seq);

            try self.prefix.cloneSlotState(source_slot, target_slot, pos);
            try self.suffix.cloneSlotState(source_slot, target_slot, pos);
        }

        // -- tiered kv ------------------------------------------------------
        // Evict a slot to a host blob, and restore it back.

        /// Bytes the first `n_rows` positions of one slot occupy in a host blob. The figure covers
        /// the whole model, so it does not move with `host_layers`.
        pub fn evictSlotBytesFor(self: *const Self, n_rows: u32) u64 {
            return blobSpanBytes(self.card, self.kv_dtype, n_rows, 0, self.n_layers);
        }

        /// The offset where the GPU half of a blob starts. It checks the figure of each runner and
        /// the blob length against the whole-model walk, in every build mode, so neither direction
        /// can drift from the other.
        fn blobBoundary(
            self: *const Self,
            n_rows: u32,
            blob_len: usize,
        ) error{BlobLengthMismatch}!usize {
            const costs = kv_cost.costVector(self.card, self.kv_dtype);
            const host = hostBlobBytes(costs, n_rows, self.host_layers);
            const device =
                blobSpanBytes(self.card, self.kv_dtype, n_rows, self.host_layers, self.n_layers);
            if (host != self.prefix.evictSlotBytesFor(n_rows)) return error.BlobLengthMismatch;
            if (device != self.suffix.evictSlotBytesFor(n_rows)) return error.BlobLengthMismatch;
            if (host + device != self.evictSlotBytesFor(n_rows)) return error.BlobLengthMismatch;
            if (blob_len != host + device) return error.BlobLengthMismatch;

            // A layer holds KV rows or a recurrent slab, and each tier holds at least one layer.
            std.debug.assert(host >= 1);
            std.debug.assert(device >= 1);
            return @intCast(host);
        }

        /// Copy the first `n_rows` positions of `slot` into `out`: the CPU runner writes the host
        /// half, then the fused runner appends the GPU half.
        pub fn evictSlot(self: *Self, slot: u32, n_rows: u32, out: []u8) !void {
            // The GPU half moves through a stage that the plan must have sized.
            std.debug.assert(self.swap_planned);
            std.debug.assert(slot < self.n_slots);
            std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
            std.debug.assert(n_rows <= self.max_seq);

            const at = try self.blobBoundary(n_rows, out.len);
            try self.prefix.evictSlot(slot, n_rows, out[0..at]);
            try self.suffix.evictSlot(slot, n_rows, out[at..]);
            log_kv.debug("evict slot {d}: {d} rows, {d} KiB ({d} host + {d} gpu layers)", .{
                slot, n_rows, out.len / 1024, self.host_layers, self.n_layers - self.host_layers,
            });
        }

        /// Restore an evicted `n_rows`-row blob into `slot`, the mirror of `evictSlot`.
        pub fn restoreSlot(self: *Self, slot: u32, n_rows: u32, blob: []const u8) !void {
            // The GPU half moves through a stage that the plan must have sized.
            std.debug.assert(self.swap_planned);
            std.debug.assert(slot < self.n_slots);
            std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
            std.debug.assert(n_rows <= self.max_seq);

            const at = try self.blobBoundary(n_rows, blob.len);
            try self.prefix.restoreSlot(slot, n_rows, blob[0..at]);
            try self.suffix.restoreSlot(slot, n_rows, blob[at..]);
            log_kv.debug("restore slot {d}: {d} rows, {d} KiB ({d} host + {d} gpu layers)", .{
                slot, n_rows, blob.len / 1024, self.host_layers, self.n_layers - self.host_layers,
            });
        }
    };
}

// -- api --------------------------------------------------------------------

/// Bytes the state of one slot occupies in a blob over the model layers `[from..to)` at `n_rows`
/// committed positions. It is the `slotSpanBytes` of the planner with the row count in the place of
/// the window.
pub fn blobSpanBytes(
    card: loader.ModelCard,
    kv_dtype: dtype.KvDtype,
    n_rows: u32,
    from: u32,
    to: u32,
) u64 {
    const kv: kv_budget.KvStructure = .{
        .n_slots = 1,
        .window = n_rows,
        .costs = kv_cost.costVector(card, kv_dtype),
    };
    return kv_budget.slotSpanBytes(kv, from, to);
}

/// Bytes of the host half of a blob, which is the offset where the GPU half starts. A cost vector
/// in the place of a card lets a test frame a walk of any layer mix.
pub fn hostBlobBytes(costs: kv_budget.KvCostVector, n_rows: u32, host_layers: u32) u64 {
    const kv: kv_budget.KvStructure = .{ .n_slots = 1, .window = n_rows, .costs = costs };
    return kv_budget.slotSpanBytes(kv, 0, host_layers);
}

/// The fused plan of the GPU layers of a hybrid. A decode row occupies a slot, and the slot count
/// of a hybrid sits under the decode ceiling. So every slot can step in one tick.
pub fn suffixPlan(plan: kv_budget.LayerSplit) kv_budget.KvPlan {
    std.debug.assert(plan.host_layers < plan.n_layers);
    std.debug.assert(plan.n_slots <= limits.max_batch);

    return .{
        .vram = .{
            .weights = plan.weight_resident,
            .kv = plan.kv_vram,
            .logits = plan.logits,
            .scratch = plan.scratch,
        },
        .ram = .{ .blobs = plan.ram.blobs, .prefix = plan.ram.prefix },
        .n_slots = plan.n_slots,
        .n_streams = plan.n_slots,
        .window = plan.window,
    };
}

// -- helpers ----------------------------------------------------------------

/// The error set of a function that returns an error union.
fn ErrorSetOf(comptime f: anytype) type {
    return @typeInfo(@typeInfo(@TypeOf(f)).@"fn".return_type.?).error_union.error_set;
}

// -- tests ------------------------------------------------------------------

// The blob arithmetic and the tags of both families, all device-free.

/// The two families that run a hybrid.
const test_arches = [_]loader.Arch{ .llama, .qwen35 };

/// The anchor card of a family. The fused runner tests read the same cards, so the cost models here
/// read the same geometry.
const testCard = @import("test_fakes.zig").testCard;
const qwen_layer = @import("qwen_layer.zig");

/// Zero bytes the synthetic tensors below view. A weight census reads lengths alone, so one
/// static buffer backs every entry.
const test_bytes align(@alignOf(f32)) = [_]u8{0} ** 4096;

/// A tensor census of `n_layers` equal decoder layers, the final norm, and a tied `token_embd`
/// head. It is what `layerWeightBytes` and `headWeightBytes` read.
fn testTensors(comptime n_layers: u32) [n_layers + 2]loader.TensorRef {
    const layer_t: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 256, 0, 0, 0 },
        .byte_stride = .{ 1024, 0, 0, 0 },
        .data = test_bytes[0..1024],
    };
    const head_t: Tensor = .{
        .dtype = .f32,
        .n_dims = 1,
        .shape = .{ 1024, 0, 0, 0 },
        .byte_stride = .{ 4096, 0, 0, 0 },
        .data = test_bytes[0..4096],
    };
    var refs: [n_layers + 2]loader.TensorRef = undefined;
    // A comptime name per layer, because a `TensorRef` borrows its name.
    inline for (0..n_layers) |i| {
        refs[i] = .{
            .name = std.fmt.comptimePrint("blk.{d}.ffn_down.weight", .{i}),
            .tensor = layer_t,
        };
    }

    refs[n_layers] = .{ .name = "output_norm.weight", .tensor = layer_t };
    refs[n_layers + 1] = .{ .name = "token_embd.weight", .tensor = head_t };
    return refs;
}

/// One GDN layer of the 2B card: the recurrent state plus the conv ring, f32 under every KV
/// dtype.
const qwen_gdn_bytes: u64 = 1_122_304;

test "hybrid qwen: the cost vector sizes six attention layers and eighteen slabs" {
    const card = testCard(.qwen35);
    // The slab of one GDN layer is the recurrent state plus the conv ring of the family rule.
    const elems = qwen_layer.recurrentElems(&card);
    const slab_bytes = (@as(u64, elems.state) + elems.conv) * @sizeOf(f32);
    try std.testing.expectEqual(qwen_gdn_bytes, slab_bytes);

    // Layers 3, 7, 11, 15, 19, and 23 hold a KV row pair a token, and the other eighteen hold a
    // fixed slab.
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const row_pair: u64 = 2 * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        const costs = kv_cost.costVector(card, kv_dtype);
        try std.testing.expectEqual(@as(u32, 24), costs.n_layers);
        for (costs.layers(), 0..) |entry, i| {
            const full = (i + 1) % 4 == 0;
            try std.testing.expectEqual(
                if (full) row_pair else 0,
                @as(u64, entry.kv_bytes_per_token),
            );
            try std.testing.expectEqual(
                if (full) 0 else qwen_gdn_bytes,
                @as(u64, entry.fixed_bytes_per_slot),
            );
        }

        // A uniform per-layer size gives a different number at every layer count, so the
        // divisibility accident on this model cannot stand in for the vector.
        try std.testing.expectEqual(6 * row_pair, costs.total().kv_bytes_per_token);
        try std.testing.expectEqual(18 * qwen_gdn_bytes, costs.total().fixed_state_bytes_per_slot);
    }
}

test "hybrid plan: every interior plan sizes each tier by the blob span of its layers" {
    inline for (test_arches) |arch| {
        const card = testCard(arch);
        const n_layers = card.n_layers;
        const refs = testTensors(comptime testCard(arch).n_layers);
        const caps: kv_budget.Caps = .{
            .vram_free = 1 << 40,
            .weights = 0,
            .logits = 0,
            .model_ctx = card.context_length,
            // The layer split planner reads neither default window.
            .serve_default = limits.min_kv_window,
            .default_ceil = limits.min_kv_window,
        };
        const kv_dtype: dtype.KvDtype = .f32;
        const kv: kv_budget.KvStructure = .{
            .n_slots = 2,
            .window = 512,
            .costs = kv_cost.costVector(card, kv_dtype),
        };
        var table: [limits.max_layers]u64 = undefined;
        const weights = tensor_lookup.layerWeightBytes(&refs, n_layers, &table);
        var host_layers: u32 = 1;
        while (host_layers < n_layers) : (host_layers += 1) {
            const plan = try kv_budget.planLayerSplit(
                caps,
                weights,
                tensor_lookup.headWeightBytes(&refs),
                kv,
                .{ .ram_split = .{ .blobs = 0, .prefix = 0 }, .residency = true },
                n_layers - host_layers,
            );
            // The blob boundary at the window is the span the planner sized, which is the
            // equality `init` asserts before it allocates.
            const host = blobSpanBytes(card, kv_dtype, kv.window, 0, host_layers);
            const device = blobSpanBytes(card, kv_dtype, kv.window, host_layers, n_layers);
            try std.testing.expectEqual(host_layers, plan.host_layers);
            try std.testing.expectEqual(@as(u64, kv.n_slots) * host, plan.kv_host);
            try std.testing.expectEqual(@as(u64, kv.n_slots) * device, plan.kv_vram);
            // One swap stage is one chunk of the GPU half of one slot blob at the window, or the
            // whole half when that is smaller.
            try std.testing.expectEqual(2 * kv_budget.stageBytes(device), plan.ram.swap_stages);
            // The resident weights are the GPU layers plus the head the fused runner runs.
            try std.testing.expectEqual(
                @as(u64, n_layers - host_layers) * 1024 + 4096 + 1024,
                plan.weight_resident,
            );
        }

        // The closed form at the middle: Llama sizes sixteen equal KV layers a side. Qwen
        // sizes three attention layers and nine slabs a side.
        const half = blobSpanBytes(card, kv_dtype, kv.window, 0, n_layers / 2);
        const row_pair: u64 = 2 * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        const want: u64 = switch (arch) {
            .llama => @as(u64, kv.window) * 16 * row_pair,
            .qwen35 => @as(u64, kv.window) * 3 * row_pair + 9 * qwen_gdn_bytes,
        };
        try std.testing.expectEqual(want, half);
    }
}

test "hybrid blob: the blob boundary partitions the whole-model walk" {
    inline for (test_arches) |arch| {
        const card = testCard(arch);
        const n_layers = card.n_layers;
        for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
            for ([_]u32{ 1, 2, 17, 512 }) |n_rows| {
                const whole = blobSpanBytes(card, kv_dtype, n_rows, 0, n_layers);
                // Every host layer count, the two ends included.
                var k: u32 = 0;
                while (k <= n_layers) : (k += 1) {
                    const host = blobSpanBytes(card, kv_dtype, n_rows, 0, k);
                    const device = blobSpanBytes(card, kv_dtype, n_rows, k, n_layers);
                    try std.testing.expectEqual(whole, host + device);
                    // The boundary only moves forward, so a wider host half never shrinks.
                    if (k > 0) {
                        try std.testing.expect(
                            host >= blobSpanBytes(card, kv_dtype, n_rows, 0, k - 1),
                        );
                    }
                }

                // Zero host layers leave the host half empty, all of them leave the GPU half empty,
                // and every interior count leaves both halves non-empty.
                try std.testing.expectEqual(
                    @as(u64, 0),
                    blobSpanBytes(card, kv_dtype, n_rows, 0, 0),
                );
                try std.testing.expectEqual(
                    @as(u64, 0),
                    blobSpanBytes(card, kv_dtype, n_rows, n_layers, n_layers),
                );
                k = 1;
                while (k < n_layers) : (k += 1) {
                    try std.testing.expect(blobSpanBytes(card, kv_dtype, n_rows, 0, k) >= 1);
                    try std.testing.expect(
                        blobSpanBytes(card, kv_dtype, n_rows, k, n_layers) >= 1,
                    );
                }
            }
        }
    }
}

test "hybrid blob: a Llama blob is the row pairs of its layers" {
    const card = testCard(.llama);
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const row_pair: u64 = 2 * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        for ([_]u32{ 1, 2, 17, 512 }) |n_rows| {
            // The CPU runner and the fused runner both write this figure for their layers.
            var k: u32 = 1;
            while (k < card.n_layers) : (k += 1) {
                try std.testing.expectEqual(
                    @as(u64, k) * n_rows * row_pair,
                    blobSpanBytes(card, kv_dtype, n_rows, 0, k),
                );
            }
        }
    }
}

test "hybrid blob: Qwen rows size the attention layers and never the slabs" {
    const card = testCard(.qwen35);
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const row_bytes: u64 = 2 * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        // A host half below the attention interval holds no KV at all, so its blob does not move
        // with the row count.
        const thin_one = blobSpanBytes(card, kv_dtype, 1, 0, 3);
        try std.testing.expectEqual(3 * qwen_gdn_bytes, thin_one);
        try std.testing.expectEqual(thin_one, blobSpanBytes(card, kv_dtype, 512, 0, 3));
        // Twelve host layers hold three attention layers on each side.
        for ([_]u32{ 1, 2, 17, 512 }) |n_rows| {
            const host = blobSpanBytes(card, kv_dtype, n_rows, 0, 12);
            try std.testing.expectEqual(9 * qwen_gdn_bytes + 3 * n_rows * row_bytes, host);
            const device = blobSpanBytes(card, kv_dtype, n_rows, 12, 24);
            try std.testing.expectEqual(9 * qwen_gdn_bytes + 3 * n_rows * row_bytes, device);
        }

        // The dtype halves the row-sized part and leaves the slabs equal.
        try std.testing.expectEqual(
            18 * qwen_gdn_bytes + 6 * @as(u64, 512) * row_bytes,
            blobSpanBytes(card, kv_dtype, 512, 0, 24),
        );
    }
}

test "hybrid blob: the walk of mixed layers splits at the byte count of the host walk" {
    // The anchor of the blob framing: two KV layers of 24-byte K and V rows at one row, and two
    // GDN layers of a 100-byte recurrent slab and an 8-byte conv slab.
    const regions = [4][2]u64{ .{ 24, 24 }, .{ 100, 8 }, .{ 100, 8 }, .{ 24, 24 } };
    const want = [8][2]u64{
        .{ 0, 24 },    .{ 24, 24 }, .{ 48, 100 }, .{ 148, 8 },
        .{ 156, 100 }, .{ 256, 8 }, .{ 264, 24 }, .{ 288, 24 },
    };
    var costs = kv_budget.KvCostVector.uniform(4, 0);
    costs.set(0, 48, 0);
    costs.set(1, 0, 108);
    costs.set(2, 0, 108);
    costs.set(3, 48, 0);
    const kv: kv_budget.KvStructure = .{ .n_slots = 1, .window = 1, .costs = costs };

    // The whole walk from offset 0 is the anchor, and each layer spans its cost entry.
    var got: [8][2]u64 = undefined;
    var base: u64 = 0;
    for (regions, 0..) |layer, li| {
        const layer_bytes = layer[0] + layer[1];
        try std.testing.expectEqual(
            layer_bytes,
            kv_budget.slotSpanBytes(kv, @intCast(li), @intCast(li + 1)),
        );
        for (layer, 0..) |size, ri| {
            got[2 * li + ri] = .{ base, size };
            base += size;
        }
    }

    try std.testing.expectEqualSlices([2]u64, &want, &got);

    // Two host layers end at byte 156. The GPU walk from there is the tail of the anchor.
    const boundary = hostBlobBytes(costs, 1, 2);
    try std.testing.expectEqual(@as(u64, 156), boundary);
    try std.testing.expectEqual(boundary, want[4][0]);
    try std.testing.expectEqual(
        kv_budget.slotSpanBytes(kv, 0, 4),
        boundary + kv_budget.slotSpanBytes(kv, 2, 4),
    );
    try std.testing.expectEqual(@as(u64, 312), kv_budget.slotSpanBytes(kv, 0, 4));
}
