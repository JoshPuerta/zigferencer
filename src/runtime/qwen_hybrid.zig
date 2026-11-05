//! `QwenHybridRunner` splits one Qwen3.5 forward pass across both executors. A `QwenRunner` runs
//! the layer prefix `[0..split)` on the CPU. A `QwenFusedRunner` runs the suffix
//! `[split..n_layers)`, the final norm, and the lm_head on the GPU. The budget planner picks
//! `split` so the device suffix fits the VRAM budget.
//!
//! Composition. `QwenRunner.initPrefix` builds the host prefix, and the hidden-row entry points of
//! `QwenFusedRunner` take the crossing rows. This file records no dispatch and copies no layer
//! body.
//!
//! Crossing. The residual stream crosses the host-device boundary once per call. The prefix leaves
//! its rows in its own `x`. The suffix reads that stream in place of an embedding lookup, so no
//! third buffer exists. The CPU prefix runs the whole batch before the one crossing. So the GDN
//! recurrence sees time order on both tiers.
//!
//! Split bounds. The two degenerate placements are whole-model runners, and the CLI dispatches them
//! to `QwenRunner` and `QwenFusedRunner` directly. A hybrid that accepted them holds a second
//! construction of each. No gate reaches it, and no token hash covers it.
//!
//! Slot state. The state of a slot spans both tiers, so every slot hook does too. Both tiers hold
//! GDN recurrent slabs, so the session surface needs both slot hooks from this runner. Without the
//! `has_recurrent_state` marker, a hybrid that omits `resetSlotState` still compiles, and it leaks
//! that state across a slot reuse.
//!
//! Blob. One blob stays one walk in whole-model order: prefix regions first, suffix regions after.
//! Both walks emit layers in model order. A full-attention layer emits KV rows, and a GDN layer
//! emits its whole recurrent and conv slabs. So the concatenation is the order that a single-tier
//! runner writes. The boundary offset is the byte count of the prefix. `blobSpanBytes` states that
//! partition, and every call checks it against what the two runners report.
//!
//! Disk namespace. The CPU and GPU halves of one blob are not bit-identical. So a hybrid blob lives
//! in its own disk namespace. The split and the KV dtype qualify the tag further.
//!
//! Swap stages. A swap stage holds one chunk of the device half of a slot blob, or that half when
//! smaller. A residency plan prices two stages. A plan without residency prices zero and never
//! swaps.
//!
//! Tests. Nothing in this file runs without a device. The tests below cover the blob framing and
//! the backend tag, which are device-free arithmetic. The hardware gates are the equivalence tests
//! in `tests/`.

const std = @import("std");
const span = @import("../core/span.zig");
const build_options = @import("build_options");
const vulkan_ctx = @import("vulkan_ctx.zig");
const loader = @import("../io/gguf/loader.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_cost = @import("kv_cost.zig");
const naive = @import("../ops/naive.zig");
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const step_row = @import("step_row.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const qwen_runner = @import("qwen_runner.zig");
const qwen_fused = @import("qwen_fused.zig");
const Allocator = std.mem.Allocator;
const StepRow = step_row.StepRow;
const QwenRunner = qwen_runner.QwenRunner;
const QwenFusedRunner = qwen_fused.QwenFusedRunner;

const log = std.log.scoped(.runner);
const log_kv = std.log.scoped(.kv);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/qwen_hybrid.zig was included in a non-vulkan build.");
    }
}

// -- types ------------------------------------------------------------------

pub const QwenHybridRunner = struct {
    gpa: Allocator,

    /// The host layers `[0..split)`, which live in `prefix_arena`, so `deinit` reclaims them in one
    /// call.
    prefix: QwenRunner,
    prefix_arena: std.heap.ArenaAllocator,
    /// The device layers `[split..n_layers)`, the final norm, and the lm_head.
    suffix: *QwenFusedRunner,

    split: u32,
    n_layers: u32,
    hidden: u32,
    card: loader.ModelCard,
    /// The element of every full-attention KV row on both tiers, so the blob keeps one row stride.
    kv_dtype: dtype.KvDtype,

    // -- session surface ----------------------------------------------------
    // The session layer in `runtime/session.zig` reads the members below from
    // every runner family, and serve reads the backend tag.

    /// The compact logits rows of the suffix, aliased: the suffix runs the lm_head and owns them.
    logits: []f32,
    n_slots: u32,
    max_seq: u32,
    /// Rows one `prefillBatch` accepts. The bench and the serve chunker read this field on every
    /// runner.
    prefill_batch_max: u32,
    /// True when the plan priced both swap stages. A plan without them means this runner never
    /// swaps.
    swap_priced: bool,

    /// Both tiers hold GDN slabs. See "Slot state" in the header.
    pub const has_recurrent_state = true;

    /// The `run` command batches the prompt through `prefillBatch`, as serve and bench-model do.
    pub const chunked_prefill: bool = true;

    /// The literal frame of a placement tag, which the bound and the format share.
    const tag_prefix = "vulkan_hybrid_qwen:k=";

    /// The one weight element the qwen family stages: `qwen_fused` refuses any other dtype at
    /// upload.
    const quant_tag = "q8_0";

    /// Bytes the widest tag occupies: the literal frame, the widest u32 split,
    /// the one separator, the quant name, and the widest KV dtype component.
    pub const backend_tag_buf_len: usize = tag_prefix.len +
        std.fmt.comptimePrint("{d}", .{std.math.maxInt(u32)}).len + 1 +
        quant_tag.len + dtype.kv_tag_suffix_max;

    /// The snapshot-provenance tag of a qwen hybrid placement. See "Disk namespace" in the header.
    /// `buf` must hold `backend_tag_buf_len` bytes, the bound that makes the format infallible.
    pub fn backendTag(buf: []u8, split: u32, kv_dtype: dtype.KvDtype) []const u8 {
        std.debug.assert(buf.len >= backend_tag_buf_len);

        return std.fmt.bufPrint(
            buf,
            tag_prefix ++ "{d}:" ++ quant_tag ++ "{s}",
            .{ split, kv_dtype.tagSuffix() },
        ) catch unreachable;
    }

    // -- init ---------------------------------------------------------------

    /// Build the suffix and the prefix at the KV dtype of the caller, which both tiers share. The
    /// `logits` reservation must cover `n_slots` rows, because every decode row of a tick wants
    /// logits.
    pub fn init(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        loaded: *const loader.Loaded,
        plan: kv_budget.PlacementPlan,
        io: std.Io,
        pool: *Pool,
        policy: parallel.Policy,
        kv_dtype: dtype.KvDtype,
    ) !*QwenHybridRunner {
        // Name the placement.
        const card = loaded.card;
        const split: u32 = plan.split;
        const n_layers: u32 = plan.n_layers;
        std.debug.assert(n_layers == card.n_layers);

        // A degenerate split belongs to a whole-model runner. See "Split bounds" in the header.
        std.debug.assert(split >= 1);
        std.debug.assert(split < n_layers);

        // The host prefix indexes its slots by the decode width, the ceiling `hybridSlots`
        // enforces.
        std.debug.assert(plan.n_slots >= 1);
        std.debug.assert(plan.n_slots <= limits.max_batch);
        std.debug.assert(plan.window >= 1);

        // Check the plan against the per-layer cost vector, before the first allocation. A plan
        // built from another vector fails here.
        const kv_struct: kv_budget.KvStructure = .{
            .n_slots = plan.n_slots,
            .window = plan.window,
            .costs = kv_cost.qwenCostVector(card, kv_dtype),
        };
        const host_span = kv_budget.slotSpanBytes(kv_struct, 0, split);
        const device_span = kv_budget.slotSpanBytes(kv_struct, split, n_layers);
        std.debug.assert(plan.kv_host == @as(u64, plan.n_slots) * host_span);
        std.debug.assert(plan.kv_vram == @as(u64, plan.n_slots) * device_span);
        std.debug.assert(
            host_span + device_span == kv_budget.slotSpanBytes(kv_struct, 0, n_layers),
        );
        std.debug.assert(plan.ram.host_kv == plan.kv_host);

        // Check the swap stages. See "Swap stages" in the header.
        const stage_one = blobSpanBytes(card, kv_dtype, plan.window, split, n_layers);
        std.debug.assert(stage_one == device_span);
        const swap_priced = plan.ram.swap_stages == 2 * kv_budget.stageBytes(stage_one);
        std.debug.assert(swap_priced or plan.ram.swap_stages == 0);
        std.debug.assert(plan.ram.swap_stages <= 2 * limits.swap_stage_chunk_bytes);
        std.debug.assert(plan.ram.swap_stages * plan.n_slots <= 2 * plan.kv_vram);

        const self = try gpa.create(QwenHybridRunner);
        errdefer gpa.destroy(self);

        // Build the suffix first: its VRAM preflight refuses an over-budget placement before the
        // prefix spends host memory.
        const suffix = try QwenFusedRunner.initSuffix(gpa, ctx, loaded, .{
            .vram = .{
                .weights = plan.weight_resident,
                .kv = plan.kv_vram,
                .logits = plan.logits,
                .scratch = plan.scratch,
            },
            .ram = .{ .blobs = plan.ram.blobs, .prefix = plan.ram.prefix },
            .n_slots = plan.n_slots,
            // A decode row occupies a slot, and the slot count sits under the decode ceiling. So
            // every slot can step in one tick.
            .n_streams = plan.n_slots,
            .window = plan.window,
        }, split, kv_dtype);
        errdefer suffix.deinit();

        // Fill the fields. `prefix` stays undefined until the init below.
        self.* = .{
            .gpa = gpa,
            .prefix = undefined,
            .prefix_arena = .init(gpa),
            .suffix = suffix,
            .split = split,
            .n_layers = n_layers,
            .hidden = card.embedding_length,
            .card = card,
            .kv_dtype = kv_dtype,
            .logits = suffix.logits,
            .n_slots = plan.n_slots,
            .max_seq = plan.window,
            .prefill_batch_max = suffix.maxStepRows(),
            .swap_priced = swap_priced,
        };
        errdefer self.prefix_arena.deinit();

        // Build the prefix. Every row of a call crosses, so it holds row scratch for the widest
        // suffix step.
        self.prefix = try QwenRunner.initPrefix(
            self.prefix_arena.allocator(),
            loaded,
            kv_budget.pinned(plan.n_slots, plan.window),
            split,
            suffix.maxStepRows(),
            io,
            pool,
            policy,
            kv_dtype,
        );

        // The prefix allocated what the planner priced for `[0..split)`: one slot's KV at the
        // window plus its GDN slabs. The suffix ran the same check on its own range.
        std.debug.assert(self.prefix.evictSlotBytesFor(plan.window) == host_span);
        std.debug.assert(self.prefix.maxStepRows() >= suffix.maxStepRows());
        std.debug.assert(self.prefix.n_slots == self.n_slots);
        std.debug.assert(self.prefix.max_seq == self.max_seq);
        std.debug.assert(self.prefix.layers.len == split);

        log.debug(
            "init qwen hybrid: split {d} of {d} layers ({d} on gpu), {d} slots, max_seq " ++
                "{d}, kv {t}",
            .{
                split, n_layers, n_layers - split, plan.n_slots, plan.window, kv_dtype,
            },
        );
        return self;
    }

    pub fn deinit(self: *QwenHybridRunner) void {
        self.suffix.deinit();
        self.prefix_arena.deinit();
        self.gpa.destroy(self);
    }

    /// Rows of the compact logits buffer. The suffix runs the lm_head, so the
    /// figure is the suffix's.
    pub fn logitRowsMax(self: *const QwenHybridRunner) u32 {
        return self.suffix.logitRowsMax();
    }

    /// Max rows one `stepBatch` accepts. Both tiers hold row scratch for it.
    pub fn maxStepRows(self: *const QwenHybridRunner) u32 {
        return self.suffix.maxStepRows();
    }

    // -- step ---------------------------------------------------------------

    /// Run the CPU prefix over `rows`: embed, then layers `[0..split)`. The crossing activations
    /// stay in `prefix.x`. `embedLookup` is the naive function on every op backend, as in
    /// `QwenRunner.stepBatch`.
    fn hostPrefix(self: *QwenHybridRunner, rows: []const StepRow) qwen_runner.Error!void {
        const sp = span.beginHot(.host_prefix);
        defer sp.end();

        std.debug.assert(rows.len >= 1);
        std.debug.assert(rows.len <= self.prefix.max_rows);
        const d = self.hidden;
        for (rows, 0..) |r, i| {
            naive.embedLookup(self.prefix.embed.*, r.token, d, self.prefix.x[i * d ..][0..d]);
        }

        var l: u32 = 0;
        while (l < self.split) : (l += 1) {
            self.prefix.layerForward(l, rows);
        }
    }

    /// The residual rows the prefix just wrote: the value that crosses the
    /// host-device boundary.
    fn crossing(self: *const QwenHybridRunner, n_rows: usize) []const f32 {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(n_rows <= self.prefix.max_rows);

        return self.prefix.x[0 .. n_rows * @as(usize, self.hidden)];
    }

    /// One decode step on slot 0. Returns the logits row, valid until the next
    /// call.
    pub fn step(self: *QwenHybridRunner, token_id: u32, pos: u32) ![]f32 {
        if (pos >= self.max_seq) return error.PositionOutOfRange;

        const row = [_]StepRow{.{ .slot = 0, .token = token_id, .pos = pos, .want_logits = true }};
        try self.hostPrefix(&row);
        return self.suffix.stepHidden(self.crossing(1), pos);
    }

    /// Advance a batch of rows under the rules of `QwenRunner.stepBatch`, with compact logits for
    /// the `want_logits` suffix. See "Crossing" in the header.
    pub fn stepBatch(self: *QwenHybridRunner, rows: []const StepRow) !void {
        // The row rules bind the prefix drive too, so the check runs ahead of the first host layer.
        _ = try step_row.validateRows(rows, self.n_slots, self.maxStepRows(), self.max_seq);
        try self.hostPrefix(rows);
        return self.suffix.stepBatchHidden(self.crossing(rows.len), rows);
    }

    /// Process `b` prompt tokens in one chunk-interleaved pass on slot 0:
    /// batched CPU prefix, one crossing, batched GPU suffix. Returns logits for
    /// the last token only. Callers chunk longer prompts.
    pub fn prefillBatch(self: *QwenHybridRunner, tokens: []const u32, start_pos: u32) ![]f32 {
        const b: u32 = @intCast(tokens.len);
        std.debug.assert(b >= 1);
        std.debug.assert(b <= self.maxStepRows());
        var rows_buf: [QwenFusedRunner.prefill_rows_max]StepRow = undefined;
        for (tokens, 0..) |t, i| {
            rows_buf[i] = .{
                .slot = 0,
                .token = t,
                .pos = start_pos + @as(u32, @intCast(i)),
                .want_logits = i == tokens.len - 1,
            };
        }

        try self.hostPrefix(rows_buf[0..b]);
        return self.suffix.prefillBatchHidden(self.crossing(b), b, start_pos);
    }

    // -- slot state ---------------------------------------------------------
    // One slot spans both tiers, so every hook below drives both runners.

    /// Zero the GDN state of a slot on both tiers for a fresh conversation. `Sessions.create` calls
    /// this on slot reuse. The KV needs no scrub, because attention reads only rewritten positions.
    pub fn resetSlotState(self: *QwenHybridRunner, slot: u32) void {
        std.debug.assert(slot < self.n_slots);

        self.prefix.resetSlotState(slot);
        self.suffix.resetSlotState(slot);
    }

    /// Fork slot state for a session clone across both tiers. One slot id spans them, because both
    /// carry the same slot count. `Sessions.clone` calls this.
    pub fn cloneSlotState(
        self: *QwenHybridRunner,
        source_slot: u32,
        target_slot: u32,
        pos: u32,
    ) void {
        std.debug.assert(source_slot < self.n_slots);
        std.debug.assert(target_slot < self.n_slots);
        std.debug.assert(source_slot != target_slot);
        std.debug.assert(pos <= self.max_seq);

        self.prefix.cloneSlotState(source_slot, target_slot, pos);
        self.suffix.cloneSlotState(source_slot, target_slot, pos);
    }

    // -- tiered kv ----------------------------------------------------------
    // Evict a slot to a host blob, and restore it back. See "Blob" in the header.

    /// Bytes the state of one slot occupies in a blob over the absolute layer range `[from..to)` at
    /// `n_rows` committed positions. It is the `slotSpanBytes` of the planner with the row count in
    /// the place of the window.
    pub fn blobSpanBytes(
        card: loader.ModelCard,
        kv_dtype: dtype.KvDtype,
        n_rows: u32,
        from: u32,
        to: u32,
    ) u64 {
        return QwenFusedRunner.rangeSlotBytes(card, kv_dtype, n_rows, from, to).total();
    }

    /// Bytes one slot's first `n_rows` positions occupy in a host blob. The
    /// figure covers the whole model, so it does not move with the split.
    pub fn evictSlotBytesFor(self: *const QwenHybridRunner, n_rows: u32) u64 {
        return blobSpanBytes(self.card, self.kv_dtype, n_rows, 0, self.n_layers);
    }

    /// The blob boundary: the offset where the device half starts. It checks the figure of each
    /// runner against the whole-model walk, so neither direction can drift from the other.
    fn seam(self: *const QwenHybridRunner, n_rows: u32, blob_len: usize) usize {
        const host: usize = @intCast(
            blobSpanBytes(self.card, self.kv_dtype, n_rows, 0, self.split),
        );
        const device: usize = @intCast(
            blobSpanBytes(self.card, self.kv_dtype, n_rows, self.split, self.n_layers),
        );
        std.debug.assert(host == @as(usize, @intCast(self.prefix.evictSlotBytesFor(n_rows))));
        std.debug.assert(device == @as(usize, @intCast(self.suffix.evictSlotBytesFor(n_rows))));
        std.debug.assert(host + device == @as(usize, @intCast(self.evictSlotBytesFor(n_rows))));
        std.debug.assert(blob_len == host + device);
        // A split of at least one layer holds state of one kind or the other.
        std.debug.assert(host >= 1);
        std.debug.assert(device >= 1);
        return host;
    }

    /// Copy the first `n_rows` positions of `slot` into `out`: the prefix runner
    /// writes the host half, then the suffix appends the device half.
    pub fn evictSlot(self: *QwenHybridRunner, slot: u32, n_rows: u32, out: []u8) !void {
        // The device half moves through a stage that the plan must have priced.
        std.debug.assert(self.swap_priced);
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);

        const at = self.seam(n_rows, out.len);
        try self.prefix.evictSlot(slot, n_rows, out[0..at]);
        try self.suffix.evictSlot(slot, n_rows, out[at..]);
        log_kv.debug("evict slot {d}: {d} rows, {d} KiB ({d} host + {d} gpu layers)", .{
            slot, n_rows, out.len / 1024, self.split, self.n_layers - self.split,
        });
    }

    /// Restore an evicted `n_rows`-row blob into `slot`, the mirror of `evictSlot`.
    pub fn restoreSlot(self: *QwenHybridRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        // The device half moves through a stage that the plan must have priced.
        std.debug.assert(self.swap_priced);
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);

        const at = self.seam(n_rows, blob.len);
        try self.prefix.restoreSlot(slot, n_rows, blob[0..at]);
        try self.suffix.restoreSlot(slot, n_rows, blob[at..]);
        log_kv.debug("restore slot {d}: {d} rows, {d} KiB ({d} host + {d} gpu layers)", .{
            slot, n_rows, blob.len / 1024, self.split, self.n_layers - self.split,
        });
    }
};

// -- tests ------------------------------------------------------------------

// Blob framing and tag anchors, all device-free arithmetic.
/// The 2B anchor card. Both files reason about one Qwen3.5 geometry, so one
/// declaration holds it and drift cannot open between the two cost models.
const testCard = qwen_fused.testCard;

/// Zero bytes the synthetic tensors below view. A weight census reads lengths
/// alone, so one static buffer backs every entry.
const test_bytes align(@alignOf(f32)) = [_]u8{0} ** 4096;

/// The 2B tensor census: 24 equal decoder layers, the final norm, and a tied
/// `token_embd` head. It is what `layerWeightBytes` and `headWeightBytes` read.
fn testTensors() [26]loader.TensorRef {
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
    var refs: [26]loader.TensorRef = undefined;
    // A comptime name per layer, because a `TensorRef` borrows its name.
    inline for (0..24) |i| {
        refs[i] = .{
            .name = std.fmt.comptimePrint("blk.{d}.ffn_down.weight", .{i}),
            .tensor = layer_t,
        };
    }

    refs[24] = .{ .name = "output_norm.weight", .tensor = layer_t };
    refs[25] = .{ .name = "token_embd.weight", .tensor = head_t };
    return refs;
}

test "qwen split: the cost vector prices six attention layers and eighteen slabs" {
    const card = testCard();
    // The closed form of the 2B stack, `qwen2bCosts` in `core/kv_budget.zig`: layers 3, 7, 11, 15,
    // 19, and 23 charge a KV row pair a token, and the other eighteen hold a fixed slab.
    const gdn_bytes: u64 = 1_122_304;
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const row_pair: u64 = 2 * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        const costs = kv_cost.qwenCostVector(card, kv_dtype);
        try std.testing.expectEqual(@as(u32, 24), costs.n_layers);
        for (costs.layers(), 0..) |entry, i| {
            const full = (i + 1) % 4 == 0;
            try std.testing.expectEqual(
                if (full) row_pair else 0,
                @as(u64, entry.kv_bytes_per_token),
            );
            try std.testing.expectEqual(
                if (full) 0 else gdn_bytes,
                @as(u64, entry.fixed_bytes_per_slot),
            );
        }

        // A uniform per-layer price charges a different number at every split, so the divisibility
        // accident on this model cannot stand in for the vector.
        try std.testing.expectEqual(6 * row_pair, costs.total().kv_bytes_per_token);
        try std.testing.expectEqual(18 * gdn_bytes, costs.total().fixed_state_bytes_per_slot);
    }
}

test "qwen split: a mid split prices each tier by the layers it holds" {
    const card = testCard();
    const refs = testTensors();
    const caps: kv_budget.Caps = .{
        .vram_free = 1 << 32,
        .weights = 0,
        .logits = 0,
        .model_ctx = card.context_length,
        // The placement planner reads neither default window.
        .serve_default = limits.min_kv_window,
        .default_ceil = limits.min_kv_window,
    };
    const kv_dtype: dtype.KvDtype = .f32;
    const kv: kv_budget.KvStructure = .{
        .n_slots = 2,
        .window = 512,
        .costs = kv_cost.qwenCostVector(card, kv_dtype),
    };
    var table: [limits.max_layers]u64 = undefined;
    // Twelve GPU layers leave the split at 12, and each half holds three
    // attention layers and nine recurrent ones.
    const plan = try kv_budget.placementPlan(
        caps,
        tensor_lookup.layerWeightBytes(&refs, card.n_layers, &table),
        tensor_lookup.headWeightBytes(&refs),
        kv,
        .{
            .ram_split = .{ .blobs = 0, .prefix = 0 },
            .residency = true,
        },
        12,
    );
    const half: u64 = @as(u64, kv.n_slots) * (@as(u64, kv.window) * 3 * 4_096 + 9 * 1_122_304);
    try std.testing.expectEqual(@as(u32, 12), plan.split);
    try std.testing.expectEqual(half, plan.kv_vram);
    try std.testing.expectEqual(half, plan.kv_host);
    // The boundary the runner walks is the same span the planner priced, which is
    // the equality `init` asserts before it allocates.
    try std.testing.expectEqual(
        kv_budget.slotSpanBytes(kv, 0, plan.split),
        QwenHybridRunner.blobSpanBytes(card, kv_dtype, kv.window, 0, plan.split),
    );
    // The resident weights are the suffix layers plus the head the suffix runs.
    try std.testing.expectEqual(@as(u64, 12 * 1024 + 4096 + 1024), plan.weight_resident);
    // One swap stage is one chunk of the device half of one slot blob at the window, or the whole
    // half when that is smaller.
    try std.testing.expectEqual(
        2 * kv_budget.stageBytes(
            QwenHybridRunner.blobSpanBytes(card, kv_dtype, kv.window, plan.split, 24),
        ),
        plan.ram.swap_stages,
    );
}

test "qwen hybrid blob: the seam offset partitions the whole-model walk" {
    const card = testCard();
    const H = QwenHybridRunner;
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        for ([_]u32{ 1, 2, 17, 512 }) |n_rows| {
            const whole = H.blobSpanBytes(card, kv_dtype, n_rows, 0, 24);
            // Every split this runner accepts, plus the two ends the CLI keeps
            // for the whole-model runners.
            var k: u32 = 0;
            while (k <= 24) : (k += 1) {
                const host = H.blobSpanBytes(card, kv_dtype, n_rows, 0, k);
                const device = H.blobSpanBytes(card, kv_dtype, n_rows, k, 24);
                try std.testing.expectEqual(whole, host + device);
                // The boundary only moves forward, so a wider prefix never shrinks
                // the host half.
                if (k > 0) {
                    try std.testing.expect(
                        host >= H.blobSpanBytes(card, kv_dtype, n_rows, 0, k - 1),
                    );
                }
            }

            // The ends empty the half they leave behind, and every accepted
            // split leaves both halves non-empty.
            try std.testing.expectEqual(@as(u64, 0), H.blobSpanBytes(card, kv_dtype, n_rows, 0, 0));
            try std.testing.expectEqual(whole, H.blobSpanBytes(card, kv_dtype, n_rows, 0, 24));
            k = 1;
            while (k < 24) : (k += 1) {
                try std.testing.expect(H.blobSpanBytes(card, kv_dtype, n_rows, 0, k) >= 1);
                try std.testing.expect(H.blobSpanBytes(card, kv_dtype, n_rows, k, 24) >= 1);
            }
        }
    }
}

test "qwen hybrid blob: rows price the attention layers and never the slabs" {
    const card = testCard();
    const H = QwenHybridRunner;
    // One GDN layer's fixed slabs: the recurrent state plus the conv ring,
    // which stay f32 under every KV dtype.
    const gdn_bytes: u64 = 1_122_304;
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        const row_bytes: u64 = 2 * kv_dtype.rowBytes(card.n_kv_heads * card.head_dim);
        // A prefix below the attention interval holds no KV at all, so its blob
        // does not move with the row count.
        const thin_one = H.blobSpanBytes(card, kv_dtype, 1, 0, 3);
        try std.testing.expectEqual(3 * gdn_bytes, thin_one);
        try std.testing.expectEqual(thin_one, H.blobSpanBytes(card, kv_dtype, 512, 0, 3));
        // A split at 12 holds three attention layers on each side.
        for ([_]u32{ 1, 2, 17, 512 }) |n_rows| {
            const host = H.blobSpanBytes(card, kv_dtype, n_rows, 0, 12);
            try std.testing.expectEqual(9 * gdn_bytes + 3 * n_rows * row_bytes, host);
            const device = H.blobSpanBytes(card, kv_dtype, n_rows, 12, 24);
            try std.testing.expectEqual(9 * gdn_bytes + 3 * n_rows * row_bytes, device);
        }

        // The dtype halves the row-priced half and leaves the slabs equal.
        try std.testing.expectEqual(
            18 * gdn_bytes + 6 * @as(u64, 512) * row_bytes,
            H.blobSpanBytes(card, kv_dtype, 512, 0, 24),
        );
    }
}

test "qwen hybrid blob: the seam equals the planner's prefix span" {
    const card = testCard();
    const H = QwenHybridRunner;
    // The planner prices a slot at the window, and the blob prices it at the committed rows. At
    // `n_rows == window` the two are one number.
    for ([_]dtype.KvDtype{ .f32, .f16 }) |kv_dtype| {
        for ([_]u32{ 16, 512, 8192 }) |window| {
            const kv: kv_budget.KvStructure = .{
                .n_slots = 4,
                .window = window,
                .costs = kv_cost.qwenCostVector(card, kv_dtype),
            };
            var k: u32 = 1;
            while (k < 24) : (k += 1) {
                try std.testing.expectEqual(
                    kv_budget.slotSpanBytes(kv, 0, k),
                    H.blobSpanBytes(card, kv_dtype, window, 0, k),
                );
                try std.testing.expectEqual(
                    kv_budget.slotSpanBytes(kv, k, 24),
                    H.blobSpanBytes(card, kv_dtype, window, k, 24),
                );
            }
        }
    }
}

test "qwen hybrid tag: the split and the dtype each get their own namespace" {
    const H = QwenHybridRunner;
    var buf_a: [H.backend_tag_buf_len]u8 = undefined;
    var buf_b: [H.backend_tag_buf_len]u8 = undefined;
    var buf_c: [H.backend_tag_buf_len]u8 = undefined;
    const at_6 = H.backendTag(&buf_a, 6, .f32);
    const at_12 = H.backendTag(&buf_b, 12, .f32);
    const at_6_f16 = H.backendTag(&buf_c, 6, .f16);
    try std.testing.expectEqualStrings("vulkan_hybrid_qwen:k=6:q8_0", at_6);
    try std.testing.expectEqualStrings("vulkan_hybrid_qwen:k=12:q8_0", at_12);
    try std.testing.expectEqualStrings("vulkan_hybrid_qwen:k=6:q8_0:kv=f16", at_6_f16);
    // The family name leads, so a qwen hybrid blob can never restore into the
    // llama hybrid's namespace whatever the split.
    try std.testing.expect(!std.mem.startsWith(u8, at_6, "vulkan_hybrid:"));
    // The widest tag fits the stated bound.
    var buf_max: [H.backend_tag_buf_len]u8 = undefined;
    const widest = H.backendTag(&buf_max, std.math.maxInt(u32), .f16);
    try std.testing.expectEqual(H.backend_tag_buf_len, widest.len);
}
