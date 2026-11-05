//! Qwen3.5 hybrid forward (CPU reference) over a bounded set of session slots.
//! 3 of every 4 layers are Gated DeltaNet: linear attention with a recurrent
//! state, and the op lives in `ops/gdn.zig`. Every 4th layer is full softmax
//! attention with QK-norm and output gating. FFN is SwiGLU, and the embeddings
//! are tied (lm_head = token_embd).
//!
//! This file is the Control Unit. It holds the per-slot GDN recurrent state,
//! the full-attention KV caches, and the shared scratch. The numerical ops
//! (`ops.*`, `gdn.*`) are pure datapath. This file is also the coherence
//! reference, validated before any GPU kernel.
//!
//! `stepBatch` advances a batch of rows in the llama `LlamaRunner` shape. Every q8_0
//! projection (GDN qkv/gate/alpha/beta/out, full-attention q/k/v/o, FFN,
//! lm_head) runs as one batched pass per weight, so the weight streams from
//! DRAM once per call instead of once per row. Everything stateful stays per
//! row. The GDN recurrence is sequential in TIME, not across sessions. Each row
//! steps the state of its own slot, and the row rule keeps same-slot rows
//! (chunked prefill) adjacent and ascending. Batch order is therefore time
//! order, and every row sees exactly the state of its solo run. Batched output
//! is bit-identical to running each row alone.

const std = @import("std");
const span = @import("../core/span.zig");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");
const Tensor = @import("../core/tensor.zig").Tensor;
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const attn_cpu = @import("../core/attn_cpu.zig");
const kv_budget = @import("../core/kv_budget.zig");
const core_log = @import("../core/log.zig");
const gdn = @import("../ops/gdn.zig");
const loader = @import("../io/gguf/loader.zig");
const KvCache = @import("kv_cache.zig").KvCache;
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const step_row = @import("step_row.zig");
const linear_dispatch = @import("linear_dispatch.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const cpu_profile = @import("cpu_profile.zig");
const ProfileCat = cpu_profile.ProfileCat;

const StepRow = step_row.StepRow;

// Shared tensor-by-name lookup. `findTensor` and `findByFmt` are the required
// forms. This runner stores weights as `*const Tensor`, so the `fp32View`
// helper below adapts the pointer to the shared value-taking view.
const findTensor = tensor_lookup.findRequired;
const findByFmt = tensor_lookup.findByFmt;

// `wrap` is the GPU per-op backend. It re-exports the whole CPU op API, backs
// the heavy ops (matmul, rmsnorm, swiglu, residual, attention) with the GPU,
// and adds gdnStep. The GDN recurrence and the full-attention-at-256 GPU
// kernels are bit-checked against this CPU path (`gpu-selftest`).
// `-Dops-backend=vulkan` selects it. On a vulkan_fused build this runner is the
// qwen SESSION engine, because the fused GPU runner is single-sequence, so it
// takes the `cpu_ops` variant the build resolved.
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

pub const Error = error{
    MissingTensor,
    UnsupportedWeightDtype,
    NotQwen,
    PositionOutOfRange,
    TooManyRows,
} || Allocator.Error;

const GdnWeights = struct {
    qkv: *const Tensor, // attn_qkv [hidden, 3*inner]  q8_0
    gate: *const Tensor, // attn_gate [hidden, inner]  q8_0 (z gate)
    alpha: *const Tensor, // ssm_alpha [hidden, n_heads]  q8_0
    beta: *const Tensor, // ssm_beta  [hidden, n_heads]  q8_0
    out: *const Tensor, // ssm_out   [inner, hidden]  q8_0
    conv1d: []const f32, // ssm_conv1d [K, convDim]  f32
    ssm_a: []const f32, // ssm_a (A_log) [n_heads]  f32
    dt_bias: []const f32, // ssm_dt.bias [n_heads]  f32
    ssm_norm: []const f32, // ssm_norm [head_dim]  f32
};

const FullWeights = struct {
    q: *const Tensor, // attn_q [hidden, n_heads*head_dim*2]  (query|gate)  q8_0
    k: *const Tensor, // attn_k [hidden, n_kv_heads*head_dim]  q8_0
    v: *const Tensor, // attn_v [hidden, n_kv_heads*head_dim]  q8_0
    o: *const Tensor, // attn_output [n_heads*head_dim, hidden]  q8_0
    q_norm: []const f32, // attn_q_norm [head_dim]  f32 (QK-norm)
    k_norm: []const f32, // attn_k_norm [head_dim]  f32
    kv_slot: u32, // index into the (full-attn-only) KV cache
};

const Layer = struct {
    attn_norm: []const f32, // pre-mixer RMSNorm
    post_attn_norm: []const f32, // pre-FFN RMSNorm (named post_attention_norm)
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
    kind: union(enum) {
        gdn: GdnWeights,
        full: FullWeights,
    },
};

/// Qwen3.5 hybrid forward-pass state over `n_slots` session slots (the
/// Control Unit). `n_slots = 1` is the single-sequence CLI shape.
pub const QwenRunner = struct {
    card: loader.ModelCard,
    /// The clock source of the CPU timer.
    io: std.Io,
    /// The pool and the policy of the row-parallel matmuls and the attention
    /// fan-out, the same fan-out `LlamaRunner` uses. A pool of one thread runs
    /// fully serial, with zero overhead. The pool must outlive the runner.
    pool: *Pool,
    policy: parallel.Policy,

    /// Per-op CPU timer, present only in a `-Dtrace` build. Null on every
    /// production path. A bench command installs one to measure the serial-op
    /// fraction. See `cpu_profile.CpuProfile`.
    profile: cpu_profile.Slot = cpu_profile.slot_off,

    gdn_dims: gdn.Dims,
    n_heads: u32, // full-attn query heads
    n_kv_heads: u32,
    head_dim: u32, // full-attn head dim (256)
    rope_dim: u32,
    rope_base: f32,
    rms_eps: f32,
    hidden: u32,
    ffn_dim: u32,

    embed: *const Tensor, // embed lookup
    lm_head: *const Tensor, // output.weight (9B) or tied to embed (2B)
    output_norm: []const f32,
    layers: []Layer,

    /// One full-attn KV cache per session slot. Empty when the runner holds no
    /// full-attention layer, which only a prefix shorter than the attention
    /// interval reaches.
    kv_slots: []KvCache,
    /// The element type every full-attention KV row is framed at. The GDN
    /// recurrent state stays f32 under every dtype, so this dtype prices the
    /// attention half of a slot only.
    kv_dtype: dtype.KvDtype,
    n_slots: u32,
    /// Max rows in one `stepBatch` (row-scratch capacity), decoupled from the
    /// KV-slot count so a prefill chunk can exceed the session count. `>= n_slots`.
    max_rows: u32,
    /// Max prompt rows in one `prefillBatch`: `max_rows`, clamped to the stack
    /// row array of `prefillBatch`.
    prefill_batch_max: u32,
    /// Sequence cap, mirrored flat so every runner exposes the same field
    /// (LlamaRunner and LlamaFusedRunner already do, so callers read one name).
    max_seq: u32,
    // Per-(slot, layer) recurrent and conv state, indexed `slot * layers.len +
    // layer`. That is the layer count of this runner, which a prefix narrows,
    // and a full-attention layer gets an empty slice. Zeroed at init = fresh
    // sequence. `Sessions` relies on rewrite-before-read for slot reuse of the
    // KV, but it resets the GDN state explicitly (see `resetSlotState`).
    gdn_state: [][]f32,
    gdn_conv_state: [][]f32,

    // Scratch buffers, preallocated at init and sized x n_slots: row i of a
    // batched call owns the i-th stripe of each. One arena allocation each.
    x: []f32,
    x_norm: []f32,
    qkv: []f32, // [convDim] gdn q|k|v / conv buffer, per row
    z: []f32, // [inner] gdn gate, per row
    gdn_a: []f32, // [n_heads] per row
    gdn_b: []f32, // [n_heads] per row
    mix_o: []f32, // [inner] gdn readout (pre ssm_out), per row
    q_gate: []f32, // [n_heads*head_dim*2] full query|gate (per-head interleaved)
    q_full: []f32, // [n_heads*head_dim] de-interleaved query, per row
    gate_full: []f32, // [n_heads*head_dim] de-interleaved gate, per row
    k_buf: []f32,
    v_buf: []f32,
    attn_out: []f32,
    attn_parts: []f32,
    mix_out: []f32, // [hidden] mixer output for residual, per row
    ffn_g: []f32,
    ffn_u: []f32,
    logits: []f32,

    /// The GDN slabs are a running accumulator with no position axis, so a
    /// reused slot inherits the previous conversation unless it is scrubbed.
    /// The session surface reads this marker and then needs `resetSlotState`
    /// and `cloneSlotState` from this runner by name.
    pub const has_recurrent_state = true;

    /// The per-op GPU build keeps the per-token prompt loop that its token-hash
    /// gates anchor.
    pub const chunked_prefill: bool = !build_options.use_vulkan_ops;

    // -- init ---------------------------------------------------------------

    /// The whole-model constructor. The caller states the KV dtype. The flag
    /// resolves one dtype per process, and the resolution gates it against the
    /// model before this runs.
    pub fn init(
        arena: Allocator,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        max_rows: u32,
        io: std.Io,
        pool: *Pool,
        policy: parallel.Policy,
        kv_dtype: dtype.KvDtype,
    ) Error!QwenRunner {
        return initPrefix(
            arena,
            loaded,
            plan,
            loaded.card.n_layers,
            max_rows,
            io,
            pool,
            policy,
            kv_dtype,
        );
    }

    /// A prefix runner executes layers `[0..n_layers)` with weights, KV, and
    /// GDN state sized to match. The hybrid runner composes through it. The
    /// whole init chain funnels here, so one body sizes and frames every slab.
    ///
    /// Two range facts hold this constructor together. `isFullAttention` keeps
    /// the ABSOLUTE layer index, so a prefix layer runs the mixer the whole
    /// model gives it. `kv_slot` stays compacted WITHIN the range, so it counts
    /// the full-attention layers below its layer inside `[0..n_layers)`. That
    /// count coincides with the model-global one, because a prefix starts at 0.
    pub fn initPrefix(
        arena: Allocator,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        n_layers: u32,
        max_rows: u32,
        io: std.Io,
        pool: *Pool,
        policy: parallel.Policy,
        kv_dtype: dtype.KvDtype,
    ) Error!QwenRunner {
        const n_slots: u32 = plan.n_slots;
        const max_seq: u32 = plan.window;
        std.debug.assert(n_slots >= 1);
        std.debug.assert(n_slots <= limits.max_batch);
        std.debug.assert(pool.threads() == @max(policy.n_threads, 1));
        const c = loaded.card;
        if (c.arch != .qwen35) return error.NotQwen;

        const g = c.gdn orelse return error.NotQwen;
        std.debug.assert(n_layers >= 1);
        std.debug.assert(n_layers <= c.n_layers);

        const gd: gdn.Dims = .{
            .n_k_heads = g.group_count,
            .n_v_heads = g.time_step_rank, // 9B is decoupled (n_v=32 > n_k=16)
            .head_dim = g.state_size,
            .conv_kernel = g.conv_kernel,
        };
        const value_dim: u32 = @intCast(gd.valueDim()); // z / mix_o width (= inner_size)
        const conv_dim: u32 = @intCast(gd.convDim());

        const refs = loaded.tensors;
        const embed = try findTensor(refs, "token_embd.weight");
        // lm_head: separate output.weight when present (9B), else tied to embed (2B).
        const lm_head = findTensor(refs, "output.weight") catch embed;
        const output_norm = fp32View(try findTensor(refs, "output_norm.weight"));

        const layers = try arena.alloc(Layer, n_layers);
        const gdn_state = try arena.alloc([]f32, @as(usize, n_slots) * n_layers);
        const gdn_conv_state = try arena.alloc([]f32, @as(usize, n_slots) * n_layers);

        // The KV height the range asks for, counted before the walk assigns the
        // first `kv_slot`: the compaction assert needs the total, and the walk
        // produces it one layer too late.
        var n_full_range: u32 = 0;
        {
            var li: u32 = 0;
            while (li < n_layers) : (li += 1) {
                if (g.isFullAttention(li)) {
                    n_full_range += 1;
                }
            }
        }

        var n_full: u32 = 0;
        var name_buf: [80]u8 = undefined;
        var i: u32 = 0;
        while (i < n_layers) : (i += 1) {
            const attn_norm = fp32View(
                try findByFmt(refs, &name_buf, "blk.{d}.attn_norm.weight", .{i}),
            );
            const post_attn_norm = fp32View(
                try findByFmt(refs, &name_buf, "blk.{d}.post_attention_norm.weight", .{i}),
            );
            const ffn_gate = try findByFmt(refs, &name_buf, "blk.{d}.ffn_gate.weight", .{i});
            const ffn_up = try findByFmt(refs, &name_buf, "blk.{d}.ffn_up.weight", .{i});
            const ffn_down = try findByFmt(refs, &name_buf, "blk.{d}.ffn_down.weight", .{i});

            if (g.isFullAttention(i)) {
                std.debug.assert(n_full < n_full_range); // the index this layer takes

                layers[i] = .{
                    .attn_norm = attn_norm,
                    .post_attn_norm = post_attn_norm,
                    .ffn_gate = ffn_gate,
                    .ffn_up = ffn_up,
                    .ffn_down = ffn_down,
                    .kind = .{ .full = .{
                        .q = try findByFmt(refs, &name_buf, "blk.{d}.attn_q.weight", .{i}),
                        .k = try findByFmt(refs, &name_buf, "blk.{d}.attn_k.weight", .{i}),
                        .v = try findByFmt(refs, &name_buf, "blk.{d}.attn_v.weight", .{i}),
                        .o = try findByFmt(refs, &name_buf, "blk.{d}.attn_output.weight", .{i}),
                        .q_norm = fp32View(
                            try findByFmt(refs, &name_buf, "blk.{d}.attn_q_norm.weight", .{i}),
                        ),
                        .k_norm = fp32View(
                            try findByFmt(refs, &name_buf, "blk.{d}.attn_k_norm.weight", .{i}),
                        ),
                        .kv_slot = n_full,
                    } },
                };
                n_full += 1;
                var s: u32 = 0;
                while (s < n_slots) : (s += 1) {
                    gdn_state[@as(usize, s) * n_layers + i] = &.{};
                    gdn_conv_state[@as(usize, s) * n_layers + i] = &.{};
                }
            } else {
                layers[i] = .{
                    .attn_norm = attn_norm,
                    .post_attn_norm = post_attn_norm,
                    .ffn_gate = ffn_gate,
                    .ffn_up = ffn_up,
                    .ffn_down = ffn_down,
                    .kind = .{ .gdn = .{
                        .qkv = try findByFmt(refs, &name_buf, "blk.{d}.attn_qkv.weight", .{i}),
                        .gate = try findByFmt(refs, &name_buf, "blk.{d}.attn_gate.weight", .{i}),
                        .alpha = try findByFmt(refs, &name_buf, "blk.{d}.ssm_alpha.weight", .{i}),
                        .beta = try findByFmt(refs, &name_buf, "blk.{d}.ssm_beta.weight", .{i}),
                        .out = try findByFmt(refs, &name_buf, "blk.{d}.ssm_out.weight", .{i}),
                        .conv1d = fp32View(
                            try findByFmt(refs, &name_buf, "blk.{d}.ssm_conv1d.weight", .{i}),
                        ),
                        .ssm_a = fp32View(try findByFmt(refs, &name_buf, "blk.{d}.ssm_a", .{i})),
                        .dt_bias = fp32View(
                            try findByFmt(refs, &name_buf, "blk.{d}.ssm_dt.bias", .{i}),
                        ),
                        .ssm_norm = fp32View(
                            try findByFmt(refs, &name_buf, "blk.{d}.ssm_norm.weight", .{i}),
                        ),
                    } },
                };
                var s: u32 = 0;
                while (s < n_slots) : (s += 1) {
                    const idx = @as(usize, s) * n_layers + i;
                    gdn_state[idx] = try arena.alloc(f32, gd.stateLen());
                    gdn_conv_state[idx] = try arena.alloc(f32, gd.convStateLen());
                    @memset(gdn_state[idx], 0); // S_0 = 0
                    @memset(gdn_conv_state[idx], 0);
                }
            }
        }

        std.debug.assert(n_full == n_full_range); // the walk filled the height it priced
        // A range below the attention interval holds no full-attention layer,
        // so it holds no KV at all. `KvCache` refuses a zero-layer slab, and an
        // empty slot vector states that absence, where a zero-height cache
        // hides it.
        const kv_slots = try arena.alloc(KvCache, if (n_full == 0) 0 else n_slots);
        for (kv_slots) |*slot| {
            slot.* = try KvCache.init(arena, n_full, c.n_kv_heads, c.head_dim, max_seq, kv_dtype);
        }

        const ns: usize = n_slots;
        // Row-scratch holds the widest stepBatch (decode + a prefill chunk). The
        // per-slot GDN state and logits below stay sized by the slot count.
        const rows_cap: u32 = @max(n_slots, max_rows);
        std.debug.assert(rows_cap <= limits.max_batch + limits.max_prefill_chunk);
        const kv_bytes: u64 = 2 * @as(u64, n_slots) * n_full * max_seq * kv_dtype.rowBytes(
            c.n_kv_heads * c.head_dim,
        );
        log.debug(
            "init qwen35: {d} layers ({d} full-attn), {d} slots, max_seq {d}, kv {d:.0} MiB",
            .{
                n_layers,                                              n_full, n_slots, max_seq,
                @as(f64, @floatFromInt(kv_bytes)) / (1024.0 * 1024.0),
            },
        );
        return .{
            .card = c,
            .io = io,
            .pool = pool,
            .policy = policy,
            .gdn_dims = gd,
            .n_heads = c.n_heads,
            .n_kv_heads = c.n_kv_heads,
            .head_dim = c.head_dim,
            .rope_dim = c.rope_dim_count,
            .rope_base = c.rope_freq_base,
            .rms_eps = c.rms_norm_eps,
            .hidden = c.embedding_length,
            .ffn_dim = c.feed_forward_length,
            .embed = embed,
            .lm_head = lm_head,
            .output_norm = output_norm,
            .layers = layers,
            .kv_slots = kv_slots,
            .kv_dtype = kv_dtype,
            .n_slots = n_slots,
            .max_rows = rows_cap,
            .prefill_batch_max = @min(rows_cap, limits.max_prefill_chunk),
            .max_seq = max_seq,
            .gdn_state = gdn_state,
            .gdn_conv_state = gdn_conv_state,
            .x = try arena.alloc(f32, rows_cap * c.embedding_length),
            .x_norm = try arena.alloc(f32, rows_cap * c.embedding_length),
            .qkv = try arena.alloc(f32, rows_cap * conv_dim),
            .z = try arena.alloc(f32, rows_cap * value_dim),
            .gdn_a = try arena.alloc(f32, rows_cap * gd.n_v_heads),
            .gdn_b = try arena.alloc(f32, rows_cap * gd.n_v_heads),
            .mix_o = try arena.alloc(f32, rows_cap * value_dim),
            .q_gate = try arena.alloc(f32, rows_cap * c.n_heads * c.head_dim * 2),
            .q_full = try arena.alloc(f32, rows_cap * c.n_heads * c.head_dim),
            .gate_full = try arena.alloc(f32, rows_cap * c.n_heads * c.head_dim),
            .k_buf = try arena.alloc(f32, rows_cap * c.n_kv_heads * c.head_dim),
            .v_buf = try arena.alloc(f32, rows_cap * c.n_kv_heads * c.head_dim),
            .attn_out = try arena.alloc(f32, rows_cap * c.n_heads * c.head_dim),
            // The part records of one attention row. The rows attend one at a
            // time, and each call ends before the next starts, so one row of
            // records serves every row of a step. The per-op GPU attention
            // keeps its own records, so that build holds none here.
            .attn_parts = try arena.alloc(
                f32,
                if (build_options.use_vulkan_ops) 0 else attn_cpu.partialLen(c.n_heads, c.head_dim),
            ),
            .mix_out = try arena.alloc(f32, rows_cap * c.embedding_length),
            .ffn_g = try arena.alloc(f32, rows_cap * c.feed_forward_length),
            .ffn_u = try arena.alloc(f32, rows_cap * c.feed_forward_length),
            // Logits: only want_logits rows (<= n_slots), written compactly.
            .logits = try arena.alloc(f32, ns * c.vocab_size),
        };
    }

    /// Max rows one `stepBatch` accepts (the row-scratch capacity).
    pub fn maxStepRows(self: *const QwenRunner) u32 {
        return self.max_rows;
    }

    // -- step ---------------------------------------------------------------

    /// One forward step on slot 0, the single-sequence shape, a thin wrapper
    /// over `stepBatch`. It returns the `vocab_size` logits row. The runner
    /// owns that row and reuses it on the next call.
    pub fn step(self: *QwenRunner, token_id: u32, pos: u32) Error![]f32 {
        try self.stepBatch(&.{.{ .slot = 0, .token = token_id, .pos = pos, .want_logits = true }});
        return self.logits[0..self.card.vocab_size];
    }

    /// Consume `ids` on slot 0 at positions `pos..pos + ids.len` in one batched
    /// step, and return the logits of the last row only. The runner owns the
    /// slice and reuses it on the next call. Callers chunk a longer prompt by
    /// `prefill_batch_max`.
    pub fn prefillBatch(self: *QwenRunner, ids: []const u32, pos: u32) Error![]f32 {
        std.debug.assert(ids.len >= 1);

        if (ids.len > self.prefill_batch_max) return error.TooManyRows;

        var rows_buf: [limits.max_prefill_chunk]StepRow = undefined;
        for (ids, rows_buf[0..ids.len], 0..) |id, *r, i| {
            r.* = .{
                .slot = 0,
                .token = id,
                .pos = pos + @as(u32, @intCast(i)),
                .want_logits = i == ids.len - 1,
            };
        }

        try self.stepBatch(rows_buf[0..ids.len]);
        const out = self.logits[0..self.card.vocab_size];
        std.debug.assert(out.len == self.card.vocab_size);
        if (std.debug.runtime_safety) {
            for (out) |x| {
                std.debug.assert(std.math.isFinite(x));
            }
        }

        return out;
    }

    /// Advance every row one position: embed, all layers, then final norm +
    /// lm_head for the `want_logits` suffix only (prompt rows skip the most
    /// costly projection in the model). Projections run batched across
    /// rows. The GDN scan and attention run per row against the row's slot
    /// state. The logits are compact: the j-th `want_logits` row lands at
    /// `logits[j * vocab_size ..][0..vocab_size]`.
    pub fn stepBatch(self: *QwenRunner, rows: []const StepRow) Error!void {
        _ = try step_row.validateRows(rows, self.n_slots, self.max_rows, self.max_seq);
        log.debug(
            "step: {d} rows, slot {d} pos {d}..{d}",
            .{ rows.len, rows[0].slot, rows[0].pos, rows[rows.len - 1].pos },
        );
        const d = self.hidden;
        if (comptime core_log.hot) {
            if (self.profile) |p| {
                p.steps += 1;
            }
        }

        const t_embed = self.tBegin();
        for (rows, 0..) |r, i| {
            ops.embedLookup(self.embed.*, r.token, d, self.x[i * d ..][0..d]);
        }

        self.tEnd(t_embed, .embed);

        const live = (comptime core_log.hot) and core_log.enabled(.layer, .debug);
        const n_layers: u32 = @intCast(self.layers.len);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) {
            var t_layer: std.Io.Timestamp = undefined;
            if (live) {
                t_layer = core_log.now();
            }

            try self.layerForward(li, rows);
            if (live) {
                log_layer.debug("layer {d} ({t}): {d} rows, {d:.2} ms", .{
                    li,
                    std.meta.activeTag(
                        self.layers[li].kind,
                    ),
                    rows.len,
                    core_log.msSince(t_layer),
                });
            }
        }

        var first: usize = rows.len;
        while (first > 0 and rows[first - 1].want_logits) {
            first -= 1;
        }

        if (first == rows.len) return;

        const t_norm = self.tBegin();
        for (rows[first..], first..) |_, i| {
            ops.rmsnorm(
                self.x[i * d ..][0..d],
                self.output_norm,
                self.rms_eps,
                self.x_norm[i * d ..][0..d],
            );
        }

        self.tEnd(t_norm, .rmsnorm);
        const n_logits = rows.len - first;
        // Compact: want-logits rows land at logits[0..], so the buffer needs only
        // n_slots rows, not max_rows (Sessions.advance reads the same index).
        try self.linearBatch(
            self.x_norm[first * d .. rows.len * d],
            self.lm_head.*, // output.weight (9B) or tied embed (2B)
            self.logits[0..],
            n_logits,
            .lm_head,
        );
        std.debug.assert(first + n_logits == rows.len);
    }

    /// One decoder layer over all rows: pre-mixer norm per row, the layer's
    /// mixer (GDN or full attention), then residual. The SwiGLU FFN follows,
    /// with batched projections. It is public so a composing runner can drive
    /// the layers itself.
    pub fn layerForward(self: *QwenRunner, layer: u32, rows: []const StepRow) Error!void {
        const sp = span.beginHot(.layer_forward);
        defer sp.end();

        const d = self.hidden;
        const fd = self.ffn_dim;
        const n = rows.len;
        std.debug.assert(layer < self.layers.len);
        std.debug.assert(n <= self.max_rows);
        const lw = &self.layers[layer];

        // -- mixer block ----------------------------------------------------
        const t_an = self.tBegin();
        for (0..n) |i| {
            ops.rmsnorm(
                self.x[i * d ..][0..d],
                lw.attn_norm,
                self.rms_eps,
                self.x_norm[i * d ..][0..d],
            );
        }

        self.tEnd(t_an, .rmsnorm);
        switch (lw.kind) {
            .gdn => |w| try self.gdnMixer(w, rows, layer),
            .full => |w| try self.fullMixer(w, rows),
        }

        const t_res1 = self.tBegin();
        for (0..n) |i| {
            ops.residualAdd(self.mix_out[i * d ..][0..d], self.x[i * d ..][0..d]);
        }

        self.tEnd(t_res1, .residual);

        // -- FFN / SwiGLU block ---------------------------------------------
        const t_fn = self.tBegin();
        for (0..n) |i| {
            ops.rmsnorm(
                self.x[i * d ..][0..d],
                lw.post_attn_norm,
                self.rms_eps,
                self.x_norm[i * d ..][0..d],
            );
        }

        self.tEnd(t_fn, .rmsnorm);
        try self.linearBatch(self.x_norm[0 .. n * d], lw.ffn_gate.*, self.ffn_g, n, .gate_up_proj);
        try self.linearBatch(self.x_norm[0 .. n * d], lw.ffn_up.*, self.ffn_u, n, .gate_up_proj);
        const t_sg = self.tBegin();
        for (0..n) |i| {
            ops.swiglu(
                self.ffn_g[i * fd ..][0..fd],
                self.ffn_u[i * fd ..][0..fd],
                self.ffn_g[i * fd ..][0..fd],
            );
        }

        self.tEnd(t_sg, .swiglu);
        try self.linearBatch(self.ffn_g[0 .. n * fd], lw.ffn_down.*, self.mix_out, n, .down_proj);
        const t_res2 = self.tBegin();
        for (0..n) |i| {
            ops.residualAdd(self.mix_out[i * d ..][0..d], self.x[i * d ..][0..d]);
        }

        self.tEnd(t_res2, .residual);
    }

    // -- slot state ---------------------------------------------------------

    /// Zero the GDN recurrent state of a slot for a fresh conversation. KV
    /// needs no scrub, because attention reads only rewritten positions. The
    /// GDN scan is a running accumulator, so a reused slot inherits the state
    /// of the previous session without this. `Sessions.create` calls it when
    /// present.
    pub fn resetSlotState(self: *QwenRunner, slot: u32) void {
        std.debug.assert(slot < self.n_slots);

        const base = @as(usize, slot) * self.layers.len;
        for (self.gdn_state[base..][0..self.layers.len]) |s| {
            @memset(s, 0);
        }

        for (self.gdn_conv_state[base..][0..self.layers.len]) |s| {
            @memset(s, 0);
        }
    }

    /// Copy slot state for a session fork: KV rows `0..pos` of the
    /// full-attention layers, plus the WHOLE GDN recurrent and conv slabs. The
    /// scan state has no position axis, so the accumulator copies in full.
    /// `Sessions.clone` calls this when present.
    pub fn cloneSlotState(self: *QwenRunner, source_slot: u32, target_slot: u32, pos: u32) void {
        std.debug.assert(source_slot < self.n_slots);
        std.debug.assert(target_slot < self.n_slots);
        std.debug.assert(source_slot != target_slot);
        std.debug.assert(pos <= self.max_seq);

        // A runner without a full-attention layer carries no KV to fork, so the
        // GDN slabs below are the whole state.
        if (self.kv_slots.len > 0) {
            self.kv_slots[target_slot].copyFrom(&self.kv_slots[source_slot], pos);
        }

        const source_base = @as(usize, source_slot) * self.layers.len;
        const target_base = @as(usize, target_slot) * self.layers.len;
        for (0..self.layers.len) |li| {
            @memcpy(self.gdn_state[target_base + li], self.gdn_state[source_base + li]);
            @memcpy(self.gdn_conv_state[target_base + li], self.gdn_conv_state[source_base + li]);
        }
    }

    // -- tiered kv ----------------------------------------------------------

    // Prefix snapshots move the state of a slot to and from a host blob. The
    // CPU state already lives in host memory, so both directions are memcpys.
    // The blob walks the layers in model order: KV rows for a full-attention
    // layer, the whole recurrent and conv slabs for a GDN layer, because the
    // scan state has no position axis. That order matches the bytes of the
    // fused GPU runner exactly, so a blob holds the same bytes whichever
    // backend wrote it at the same KV dtype. The presence of `evictSlot` is
    // what lights `Sessions.can_snapshot` and `Residency.can_evict` for this
    // runner.

    /// Bytes that the state of one slot occupies in a host blob at `n_rows`
    /// committed positions.
    pub fn evictSlotBytesFor(self: *const QwenRunner, n_rows: u32) u64 {
        const row_bytes: u64 = self.kv_dtype.rowBytes(self.n_kv_heads * self.head_dim);
        var total: u64 = 0;
        for (self.layers) |*layer| {
            switch (layer.kind) {
                .full => total += @as(u64, 2) * n_rows * row_bytes,
                // The GDN state keeps f32 under every KV dtype. The planner prices
                // the per-slot fixed state as one constant, so a dtype-varying slab
                // breaks that pricing.
                .gdn => total += (@as(
                    u64,
                    self.gdn_dims.stateLen(),
                ) + self.gdn_dims.convStateLen()) * @sizeOf(f32),
            }
        }

        return total;
    }

    /// Copy slot `slot`'s state over its first `n_rows` positions into `out`.
    pub fn evictSlot(self: *QwenRunner, slot: u32, n_rows: u32, out: []u8) !void {
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);
        std.debug.assert(out.len == @as(usize, @intCast(self.evictSlotBytesFor(n_rows))));

        const base = @as(usize, slot) * self.layers.len;
        var off: usize = 0;
        for (self.layers, 0..) |*layer, li| {
            switch (layer.kind) {
                // The cache binding lives in this arm, because a runner without a
                // full-attention layer has no slot vector to index.
                .full => |*w| {
                    const kv = &self.kv_slots[slot];
                    for ([_][]const u8{
                        kv.kRowsBytes(w.kv_slot, n_rows),
                        kv.vRowsBytes(w.kv_slot, n_rows),
                    }) |bytes| {
                        @memcpy(out[off..][0..bytes.len], bytes);
                        off += bytes.len;
                    }
                },
                .gdn => {
                    for ([_][]const f32{
                        self.gdn_state[base + li],
                        self.gdn_conv_state[base + li],
                    }) |slab| {
                        const bytes = std.mem.sliceAsBytes(slab);
                        @memcpy(out[off..][0..bytes.len], bytes);
                        off += bytes.len;
                    }
                },
            }
        }

        std.debug.assert(off == out.len);
    }

    /// Restore a previously-saved `n_rows`-row blob into slot `slot`.
    pub fn restoreSlot(self: *QwenRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);
        std.debug.assert(blob.len == @as(usize, @intCast(self.evictSlotBytesFor(n_rows))));

        const base = @as(usize, slot) * self.layers.len;
        var off: usize = 0;
        for (self.layers, 0..) |*layer, li| {
            switch (layer.kind) {
                // The cache binding lives in this arm, because a runner without a
                // full-attention layer has no slot vector to index.
                .full => |*w| {
                    const kv = &self.kv_slots[slot];
                    for ([_][]u8{
                        kv.kRowsBytesMut(w.kv_slot, n_rows),
                        kv.vRowsBytesMut(w.kv_slot, n_rows),
                    }) |bytes| {
                        @memcpy(bytes, blob[off..][0..bytes.len]);
                        off += bytes.len;
                    }
                },
                .gdn => {
                    for ([_][]f32{
                        self.gdn_state[base + li],
                        self.gdn_conv_state[base + li],
                    }) |slab| {
                        const bytes = std.mem.sliceAsBytes(slab);
                        @memcpy(bytes, blob[off..][0..bytes.len]);
                        off += bytes.len;
                    }
                },
            }
        }

        std.debug.assert(off == blob.len);
    }

    // -- mixers -------------------------------------------------------------

    /// Gated DeltaNet layer: project q|k|v, gate z, decay a, and beta b, each
    /// as one batched pass over all rows. Then run the recurrent op PER ROW
    /// against the slot state of that row, and project the readouts back to
    /// hidden. Row order is time order for same-slot rows (validateRows), so
    /// the scan of each row sees exactly the state of its solo run. Inside a
    /// row, the value heads fan out on the pool.
    fn gdnMixer(self: *QwenRunner, w: GdnWeights, rows: []const StepRow, li: usize) Error!void {
        const d = self.gdn_dims;
        const n = rows.len;
        const hidden = self.hidden;
        const cd = d.convDim();
        const vd = d.valueDim();
        const nv = d.n_v_heads;
        try self.linearBatch(self.x_norm[0 .. n * hidden], w.qkv.*, self.qkv, n, .gdn_proj);
        try self.linearBatch(self.x_norm[0 .. n * hidden], w.gate.*, self.z, n, .gdn_proj);
        try self.linearBatch(self.x_norm[0 .. n * hidden], w.alpha.*, self.gdn_a, n, .gdn_proj);
        try self.linearBatch(self.x_norm[0 .. n * hidden], w.beta.*, self.gdn_b, n, .gdn_proj);

        const t_step = self.tBegin();
        for (rows, 0..) |r, i| {
            const state_idx = @as(usize, r.slot) * self.layers.len + li;
            const qkv_row = self.qkv[i * cd ..][0..cd];
            const z_row = self.z[i * vd ..][0..vd];
            const a_row = self.gdn_a[i * nv ..][0..nv];
            const b_row = self.gdn_b[i * nv ..][0..nv];
            const o_row = self.mix_o[i * vd ..][0..vd];
            if (comptime build_options.use_vulkan_ops) {
                // GPU: GdnDecode kernel (bit-checked vs gdn.step via gpu-selftest).
                ops.gdnStep(
                    @intCast(d.n_k_heads),
                    @intCast(d.n_v_heads),
                    @intCast(d.head_dim),
                    @intCast(d.conv_kernel),
                    qkv_row,
                    z_row,
                    a_row,
                    b_row,
                    w.conv1d,
                    w.ssm_a,
                    w.dt_bias,
                    w.ssm_norm,
                    self.rms_eps,
                    self.gdn_state[state_idx],
                    self.gdn_conv_state[state_idx],
                    o_row,
                );
            } else {
                parallel.gdnStep(
                    self.pool,
                    self.policy,
                    d,
                    qkv_row,
                    w.conv1d,
                    self.gdn_conv_state[state_idx],
                    .{
                        .qkv = qkv_row,
                        .z = z_row,
                        .a = a_row,
                        .b = b_row,
                        .ssm_a = w.ssm_a,
                        .dt_bias = w.dt_bias,
                        .ssm_norm = w.ssm_norm,
                        .rms_eps = self.rms_eps,
                        .state = self.gdn_state[state_idx],
                        .o_out = o_row,
                    },
                );
            }
        }

        self.tEnd(t_step, .gdn_step);
        try self.linearBatch(self.mix_o[0 .. n * vd], w.out.*, self.mix_out, n, .gdn_out);
    }

    /// Full softmax attention. q/k/v run as batched passes. Then per row:
    /// query|gate split, per-head QK-norm, partial RoPE, K/V commit. This
    /// mixer writes the K/V of every row before any row attends. Row i reads
    /// the `0..=pos_i` range of its own slot, so causality holds by
    /// construction, the same argument as `LlamaRunner.layerForward`. Then GQA
    /// attention plus a sigmoid output gate per row, and one batched output
    /// projection.
    fn fullMixer(self: *QwenRunner, w: FullWeights, rows: []const StepRow) Error!void {
        // The compaction invariant, consumer side: a full-attention layer
        // implies a cache, and its `kv_slot` addresses a layer inside that
        // cache. A prefix sizes the cache to its own range, so a model-global
        // index reads the rows of another layer and never fails a bound.
        std.debug.assert(self.kv_slots.len == self.n_slots);
        std.debug.assert(w.kv_slot < self.kv_slots[0].n_layers);

        const hd = self.head_dim;
        const n = rows.len;
        const hidden = self.hidden;
        const qd = self.n_heads * hd;
        const kvd = self.n_kv_heads * hd;
        try self.linearBatch(self.x_norm[0 .. n * hidden], w.q.*, self.q_gate, n, .attn_proj);
        try self.linearBatch(self.x_norm[0 .. n * hidden], w.k.*, self.k_buf, n, .attn_proj);
        try self.linearBatch(self.x_norm[0 .. n * hidden], w.v.*, self.v_buf, n, .attn_proj);

        for (rows, 0..) |r, i| {
            const t_qk = self.tBegin();
            // De-interleave: attn_q is reshaped [head_dim*2, n_head] => per head
            // [query(head_dim) | gate(head_dim)]. Pull into contiguous buffers.
            const q = self.q_full[i * qd ..][0..qd];
            const gate = self.gate_full[i * qd ..][0..qd];
            const qg = self.q_gate[i * qd * 2 ..][0 .. qd * 2];
            var dh: u32 = 0;
            while (dh < self.n_heads) : (dh += 1) {
                @memcpy(q[dh * hd ..][0..hd], qg[dh * 2 * hd ..][0..hd]);
                @memcpy(gate[dh * hd ..][0..hd], qg[dh * 2 * hd + hd ..][0..hd]);
            }

            // QK-norm (per head, over head_dim) then RoPE (partial, GGUF rope_dim).
            var h: u32 = 0;
            while (h < self.n_heads) : (h += 1) {
                const qh = q[h * hd ..][0..hd];
                ops.rmsnorm(qh, w.q_norm, self.rms_eps, qh);
                ops.ropeInplace(qh, r.pos, self.rope_dim, self.rope_base, .neox, null);
            }

            h = 0;
            while (h < self.n_kv_heads) : (h += 1) {
                const kh = self.k_buf[i * kvd + h * hd ..][0..hd];
                ops.rmsnorm(kh, w.k_norm, self.rms_eps, kh);
                ops.ropeInplace(kh, r.pos, self.rope_dim, self.rope_base, .neox, null);
            }

            self.tEnd(t_qk, .qk_norm_rope);
            const t_kv = self.tBegin();
            // The one conversion boundary: the f32 scratch pipeline never learns
            // the dtype, and the commit narrows the finished rows. Zen 3 carries
            // F16C, so the compiler vectorizes this into hardware
            // round-to-nearest-even.
            switch (self.kv_dtype) {
                .f32 => {
                    @memcpy(
                        self.kv_slots[r.slot].kDst(w.kv_slot, r.pos),
                        self.k_buf[i * kvd ..][0..kvd],
                    );
                    @memcpy(
                        self.kv_slots[r.slot].vDst(w.kv_slot, r.pos),
                        self.v_buf[i * kvd ..][0..kvd],
                    );
                },
                .f16 => {
                    const k_dst = self.kv_slots[r.slot].kDstAs(f16, w.kv_slot, r.pos);
                    const v_dst = self.kv_slots[r.slot].vDstAs(f16, w.kv_slot, r.pos);
                    const k_src = self.k_buf[i * kvd ..][0..kvd];
                    const v_src = self.v_buf[i * kvd ..][0..kvd];
                    for (0..kvd) |e| {
                        k_dst[e] = @floatCast(k_src[e]);
                        v_dst[e] = @floatCast(v_src[e]);
                    }
                },
            }

            self.tEnd(t_kv, .kv_commit);
        }

        for (rows, 0..) |r, i| {
            const t_attn = self.tBegin();
            const t_len: usize = @intCast(r.pos + 1);
            if (comptime build_options.use_vulkan_ops) {
                // Flag resolution refuses every dtype but f32 on this build arm,
                // because `wrap.attentionSingle` uploads the slab as f32.
                std.debug.assert(self.kv_dtype == .f32);

                ops.attentionSingle(
                    self.q_full[i * qd ..][0..qd],
                    self.n_heads,
                    hd,
                    self.kv_slots[r.slot].kSlice(w.kv_slot, r.pos),
                    self.kv_slots[r.slot].vSlice(w.kv_slot, r.pos),
                    self.n_kv_heads,
                    t_len,
                    self.attn_out[i * qd ..][0..qd],
                );
            } else {
                switch (self.kv_dtype) {
                    .f32 => parallel.attentionTyped(
                        f32,
                        ops,
                        self.pool,
                        self.policy,
                        self.q_full[i * qd ..][0..qd],
                        self.n_heads,
                        hd,
                        self.kv_slots[r.slot].kSlice(w.kv_slot, r.pos),
                        self.kv_slots[r.slot].vSlice(w.kv_slot, r.pos),
                        self.n_kv_heads,
                        t_len,
                        self.attn_parts,
                        self.attn_out[i * qd ..][0..qd],
                    ),
                    .f16 => parallel.attentionTyped(
                        f16,
                        ops,
                        self.pool,
                        self.policy,
                        self.q_full[i * qd ..][0..qd],
                        self.n_heads,
                        hd,
                        self.kv_slots[r.slot].kSliceAs(f16, w.kv_slot, r.pos),
                        self.kv_slots[r.slot].vSliceAs(f16, w.kv_slot, r.pos),
                        self.n_kv_heads,
                        t_len,
                        self.attn_parts,
                        self.attn_out[i * qd ..][0..qd],
                    ),
                }
            }

            self.tEnd(t_attn, .attention);
            // Sigmoid output gate.
            const t_gate = self.tBegin();
            const out_row = self.attn_out[i * qd ..][0..qd];
            const gate = self.gate_full[i * qd ..][0..qd];
            for (out_row, gate) |*o, gv| {
                o.* *= 1.0 / (1.0 + @exp(-gv));
            }

            self.tEnd(t_gate, .attn_gate);
        }

        try self.linearBatch(self.attn_out[0 .. n * qd], w.o.*, self.mix_out, n, .o_proj);
    }

    // -- matmul dispatch ----------------------------------------------------

    /// Dispatch the right batched linear kernel by weight dtype, with optional
    /// row-parallel fan-out. See the shared `linear_dispatch.linearBatchDispatch`.
    /// The method binds the pool and the policy of this runner. It is
    /// also the one interface every matmul of the step passes through, so the CPU
    /// timer charges every matmul category here.
    fn linearBatch(
        self: *const QwenRunner,
        inputs: []const f32,
        w: Tensor,
        outs: []f32,
        batch: usize,
        cat: ProfileCat,
    ) Error!void {
        std.debug.assert(cat.isMatmul());

        const t0 = self.tBegin();
        defer self.tEnd(t0, cat);

        return linear_dispatch.linearBatchDispatch(self.pool, self.policy, inputs, w, outs, batch);
    }

    // -- timing -------------------------------------------------------------

    /// Open a timing span. See `cpu_profile.begin`.
    inline fn tBegin(self: *const QwenRunner) cpu_profile.Span {
        return cpu_profile.begin(self.profile, self.io);
    }

    /// Close a timing span opened by `tBegin`, charging the elapsed ns to `cat`.
    inline fn tEnd(self: *const QwenRunner, t0: cpu_profile.Span, cat: ProfileCat) void {
        cpu_profile.end(self.profile, self.io, t0, cat);
    }
};

// -- helpers ----------------------------------------------------------------

/// The shared `tensor_lookup.fp32View`, adapted to the `*const Tensor` form
/// this runner stores its weights in.
fn fp32View(t: *const Tensor) []const f32 {
    return tensor_lookup.fp32View(t.*);
}

// -- tests ------------------------------------------------------------------

const test_fakes = @import("test_fakes.zig");
const test_fixtures = @import("../core/test_fixtures.zig");

test {
    // The shared timer module has no other test reference.
    _ = cpu_profile;
}

test "cpu profile: the qwen categories partition one decode step" {
    // The instrument exists only in a `-Dtrace` build, and the GPU per-op
    // backend dispatches to a device this test must not touch.
    if (comptime !core_log.hot or build_options.use_vulkan_ops) {
        return error.SkipZigTest;
    } else {
        const io = std.testing.io;
        var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena_state.deinit();

        const arena = arena_state.allocator();
        const fixture = try test_fixtures.path(arena, "Qwen3.5-2B-Q8_0");

        var loaded = try loader.load(io, arena, fixture);
        defer loaded.close();

        const max_seq = @min(loaded.card.context_length, test_fakes.test_window);
        var pool: Pool = undefined;
        try pool.init(io, 1, 0);
        defer pool.deinit();

        var r = try QwenRunner.init(
            arena,
            &loaded,
            kv_budget.pinned(1, max_seq),
            0,
            io,
            &pool,
            .{ .n_threads = 1 },
            .f32,
        );

        // One unprofiled step faults in the weight pages, so the measured step
        // pays no first-touch cost the brackets cannot see.
        _ = try r.step(1, 0);

        var prof: cpu_profile.CpuProfile = .{ .family = .qwen };
        r.profile = &prof;
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        _ = try r.step(2, 1);
        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        r.profile = null;
        const wall: u64 = @intCast(t0.durationTo(t1).raw.toNanoseconds());

        try std.testing.expectEqual(@as(u64, 1), prof.steps);
        // Cover: one decode step charges every qwen category, so no category
        // is dead and no op of the step escapes the partition. A llama
        // category stays at zero.
        for (std.enums.values(ProfileCat)) |cat| {
            if (cat.usedBy(.qwen)) {
                try std.testing.expect(prof.nsOf(cat) > 0);
            } else {
                try std.testing.expectEqual(@as(u64, 0), prof.nsOf(cat));
            }
        }

        // Disjoint: two overlapping brackets charge one nanosecond twice, which
        // pushes the sum past the wall time of the step.
        try std.testing.expect(prof.total() <= wall);
        // Whole: the uncounted remainder is row validation and loop
        // bookkeeping, so the categories carry the greater part of the step.
        try std.testing.expect(prof.total() * 2 >= wall);
        // The pool share is a subset of the whole, because the step also
        // runs serial categories.
        try std.testing.expect(prof.nsParallel() > 0);
        try std.testing.expect(prof.nsParallel() < prof.total());
    }
}
