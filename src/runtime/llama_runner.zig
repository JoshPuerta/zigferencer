//! Llama-family forward pass over a bounded set of KV slots. There is no Graph IR: the layer is
//! inlined as straight-line Zig. This file is the Control Unit. It holds the KV slots, the scratch
//! buffers, and the per-call row bookkeeping. Every numerical op it calls (`ops.*`) is pure and
//! writes into a slice the runner owns.
//!
//! Comptime. `core_log.hot` removes the trace code, and `use_vulkan_ops` picks the Vulkan kernels.
//!
//! Backend. `-Dops-backend=naive|simd|simd_pf|vulkan|auto` binds the op backend at build time. The
//! default `auto` picks the strongest eligible CPU variant. The runner imports the resolved
//! variant module and calls it directly, so the hot path has no dispatch. The `vulkan` backend
//! needs `-Druntime=vulkan`. It wraps the GPU kernels behind the same op API, and it is the
//! integration boundary for end-to-end GPU inference. On any other GPU build this runner rides
//! along as CPU capability, so it takes the `cpu_ops` variant the build resolved.
//!
//! Batched step. `stepBatch` advances a batch of rows, each row a (slot, token, pos) triple: B
//! decode rows from B independent sessions, or several consecutive rows of ONE session (chunked
//! prefill). Within a layer the runner writes the K/V of every row before any row attends, and row
//! i attends over `0..=pos_i`, so causality holds by construction. The weight projections run as
//! one batched pass per matrix, so a weight streams from DRAM once per call instead of once per
//! row. Every per-row step (norms, rope, attention) takes exactly the inputs of the solo call for
//! that row, so batched output is bit-identical to running each row alone. `step` is the B=1
//! wrapper over that one forward path.
//!
//! Scratch. Init preallocates the scratch buffers, sized by `max_rows`. Row i of a batched call
//! owns the i-th stripe of each. `max_rows` is decoupled from `n_slots`, so a prefill chunk can be
//! wider than the session count, and it is never narrower. One row of attention part records
//! serves every row of a step, because the rows attend one at a time. The per-op GPU attention
//! keeps its own records, so that build holds none.
//!
//! Compact logits. Only the `want_logits` rows get an lm_head: the decode rows and the last row of
//! a prefill chunk. They number at most `n_slots`, so `logits` has `n_slots` rows. The j-th such
//! row lands at `logits[j * vocab_size ..]`, and `Sessions.advance` reads it at the same index.
//!
//! KV commit. The f32 scratch pipeline never learns the KV dtype. One conversion boundary narrows
//! the finished K/V rows at commit. Zen 3 carries F16C, so the compiler vectorizes the narrowing
//! into hardware round-to-nearest-even.
//!
//! Blobs. Prefix snapshots move the rows of a slot to and from a host blob. The CPU cache already
//! lives in host memory, so both directions are memcpys. The layout, the region order, and the
//! sizes are identical whichever backend wrote the blob: ascending layers, K rows then V rows per
//! layer. A blob therefore restores through any runner at the same dtype. The bytes themselves are
//! executor-specific. The presence of `evictSlot` lights `Sessions.can_snapshot` and
//! `Residency.can_evict` for this runner.
//!
//! Layer. One decoder layer, pre-norm residual form. `x` is the residual stream, the
//! top line is the bypass, the lower path is the transform, and they rejoin at +:
//!
//!     x ---------------------------------------------------+ (bypass)
//!     |                                                     |
//!     +-> rmsnorm -> Wq,Wk,Wv -> RoPE -> attention -> Wo ->(+)-> x'
//!                                            ^
//!                                        KV slot
//!
//!     x'--------------------------------------------------+ (bypass)
//!     |                                                    |
//!     +-> rmsnorm -> Wgate,Wup -> SwiGLU -> Wdown -------->(+)-> x''
//!
//!   Whole pass: x = embed[token]; x = layer(x) for each of n_layers;
//!               logits = lm_head @ rmsnorm(x)   (only for rows that ask).

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
const loader = @import("../io/gguf/loader.zig");
const KvCache = @import("kv_cache.zig").KvCache;
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const step_row = @import("step_row.zig");
const linear_dispatch = @import("linear_dispatch.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const cpu_profile = @import("cpu_profile.zig");
const ProfileCat = cpu_profile.ProfileCat;

const Error = step_row.Error;
const StepRow = step_row.StepRow;

// The four runners share the lookup: `findTensor` is required, `findOptional` is optional.
const fp32View = tensor_lookup.fp32View;
const findTensor = tensor_lookup.findRequired;
const findOptional = tensor_lookup.find;
const findByFmt = tensor_lookup.findByFmt;

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

/// Llama forward-pass state over `n_slots` independent KV slots. One slot is the CLI shape.
pub const LlamaRunner = struct {
    card: loader.ModelCard,

    /// The clock source of the CPU timer.
    io: std.Io,
    /// Row-parallel matmul and attention fan-out. The caller owns the pool and must outlive it.
    pool: *Pool,
    policy: parallel.Policy,

    /// Per-op CPU timer, present only in a `-Dtrace` build. A bench command installs one.
    profile: cpu_profile.Slot = cpu_profile.slot_off,

    embed: *const Tensor,
    output_norm: *const Tensor,
    /// The lm_head. The same pointer as `embed` when the model ties embeddings.
    output: *const Tensor,
    /// Per-pair RoPE frequency divisors from `rope_freqs.weight`. Null when the model ships none.
    rope_factors: ?[]const f32,

    /// Layers this runner executes. Only a prefix runner narrows it below the card count.
    /// The weight slices and the KV slabs use this count.
    n_layers: u32,

    attn_norm_w: []const *const Tensor,
    attn_q_w: []const *const Tensor,
    attn_k_w: []const *const Tensor,
    attn_v_w: []const *const Tensor,
    attn_o_w: []const *const Tensor,
    ffn_norm_w: []const *const Tensor,
    ffn_gate_w: []const *const Tensor,
    ffn_up_w: []const *const Tensor,
    ffn_down_w: []const *const Tensor,

    /// One independent KV slab per slot. A session owns a slot for its lifetime.
    kv_slots: []KvCache,
    /// The element type every KV row is framed at. One process holds one dtype.
    kv_dtype: dtype.KvDtype,
    n_slots: u32,
    /// Max rows in one `stepBatch`, the row-scratch capacity. At least `n_slots`.
    max_rows: u32,
    /// Max prompt rows in one `prefillBatch`: `max_rows`, clamped to the stack row array.
    prefill_batch_max: u32,
    max_seq: u32,

    // Scratch buffers. The header describes their layout.
    x: []f32,
    x_norm: []f32,
    q: []f32,
    k_scratch: []f32,
    v_scratch: []f32,
    attn_out: []f32,
    attn_parts: []f32,
    ffn_gate: []f32,
    ffn_up: []f32,
    ffn_down: []f32,
    logits: []f32,

    /// The per-op GPU build keeps the per-token prompt loop that its token-hash
    /// gates anchor.
    pub const chunked_prefill: bool = !build_options.use_vulkan_ops;

    // -- init ---------------------------------------------------------------

    /// The whole-model constructor. Flag resolution has already gated `kv_dtype` against the model.
    pub fn init(
        arena: Allocator,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        max_rows: u32,
        io: std.Io,
        pool: *Pool,
        policy: parallel.Policy,
        kv_dtype: dtype.KvDtype,
    ) Error!LlamaRunner {
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

    /// A prefix runner executes layers `[0..n_layers)` with weights and KV sized to match.
    /// The hybrid runner composes through it. Every init funnels here, so one body sizes each slab.
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
    ) Error!LlamaRunner {
        // Check the limits and refuse a dtype that no kernel reads.
        const n_slots: u32 = plan.n_slots;
        const max_seq: u32 = plan.window;
        std.debug.assert(n_slots >= 1);
        std.debug.assert(n_slots <= limits.max_batch);
        std.debug.assert(pool.threads() == @max(policy.n_threads, 1));

        // The row scratch holds the widest stepBatch, and never fewer rows than the slots.
        const rows_cap: u32 = @max(n_slots, max_rows);
        std.debug.assert(rows_cap <= limits.max_batch + limits.max_prefill_chunk);

        const c = loaded.card;
        std.debug.assert(n_layers >= 1);
        std.debug.assert(n_layers <= c.n_layers);
        try linear_dispatch.requireWeights(loaded.tensors, n_layers);

        // Find the weights that no layer owns.
        const embed = try findTensor(loaded.tensors, "token_embd.weight");
        const output_norm = try findTensor(loaded.tensors, "output_norm.weight");
        const output = findOptional(loaded.tensors, "output.weight") orelse embed;
        const rope_factors: ?[]const f32 = if (findOptional(
            loaded.tensors,
            loader.rope_freqs_tensor,
        )) |t|
            fp32View(t.*)
        else
            null;

        if (rope_factors) |ff| std.debug.assert(ff.len >= c.rope_dim_count / 2);

        // Find the weights of each layer.
        const attn_norm_w = try arena.alloc(*const Tensor, n_layers);
        const attn_q_w = try arena.alloc(*const Tensor, n_layers);
        const attn_k_w = try arena.alloc(*const Tensor, n_layers);
        const attn_v_w = try arena.alloc(*const Tensor, n_layers);
        const attn_o_w = try arena.alloc(*const Tensor, n_layers);
        const ffn_norm_w = try arena.alloc(*const Tensor, n_layers);
        const ffn_gate_w = try arena.alloc(*const Tensor, n_layers);
        const ffn_up_w = try arena.alloc(*const Tensor, n_layers);
        const ffn_down_w = try arena.alloc(*const Tensor, n_layers);

        var name_buf: [64]u8 = undefined;
        var i: u32 = 0;
        while (i < n_layers) : (i += 1) {
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

        // Allocate one KV slab per slot.
        const kv_slots = try arena.alloc(KvCache, n_slots);
        for (kv_slots) |*slot| {
            slot.* = try KvCache.init(arena, n_layers, c.n_kv_heads, c.head_dim, max_seq, kv_dtype);
        }

        // Log the KV size.
        const kv_bytes: u64 = 2 * @as(u64, n_slots) * n_layers * max_seq * kv_dtype.rowBytes(
            c.n_kv_heads * c.head_dim,
        );
        log.debug("init llama: {d} layers, {d} slots, max_seq {d}, kv {d:.0} MiB", .{
            n_layers,                                              n_slots, max_seq,
            @as(f64, @floatFromInt(kv_bytes)) / (1024.0 * 1024.0),
        });

        // Allocate the scratch and build the runner.
        return .{
            .card = c,
            .io = io,
            .pool = pool,
            .policy = policy,
            .embed = embed,
            .output_norm = output_norm,
            .output = output,
            .rope_factors = rope_factors,
            .n_layers = n_layers,
            .attn_norm_w = attn_norm_w,
            .attn_q_w = attn_q_w,
            .attn_k_w = attn_k_w,
            .attn_v_w = attn_v_w,
            .attn_o_w = attn_o_w,
            .ffn_norm_w = ffn_norm_w,
            .ffn_gate_w = ffn_gate_w,
            .ffn_up_w = ffn_up_w,
            .ffn_down_w = ffn_down_w,
            .kv_slots = kv_slots,
            .kv_dtype = kv_dtype,
            .n_slots = n_slots,
            .max_rows = rows_cap,
            .prefill_batch_max = @min(rows_cap, limits.max_prefill_chunk),
            .max_seq = max_seq,
            .x = try arena.alloc(f32, rows_cap * c.embedding_length),
            .x_norm = try arena.alloc(f32, rows_cap * c.embedding_length),
            .q = try arena.alloc(f32, rows_cap * c.n_heads * c.head_dim),
            .k_scratch = try arena.alloc(f32, rows_cap * c.n_kv_heads * c.head_dim),
            .v_scratch = try arena.alloc(f32, rows_cap * c.n_kv_heads * c.head_dim),
            .attn_out = try arena.alloc(f32, rows_cap * c.n_heads * c.head_dim),
            .attn_parts = try arena.alloc(
                f32,
                if (build_options.use_vulkan_ops) 0 else attn_cpu.partialLen(c.n_heads, c.head_dim),
            ),
            .ffn_gate = try arena.alloc(f32, rows_cap * c.feed_forward_length),
            .ffn_up = try arena.alloc(f32, rows_cap * c.feed_forward_length),
            .ffn_down = try arena.alloc(f32, rows_cap * c.embedding_length),
            .logits = try arena.alloc(f32, n_slots * c.vocab_size),
        };
    }

    /// Max rows one `stepBatch` accepts. The scheduler fills decode rows plus a prefill chunk.
    pub fn maxStepRows(self: *const LlamaRunner) u32 {
        return self.max_rows;
    }

    // -- timing -------------------------------------------------------------

    /// Open a timing span. See `cpu_profile.begin`.
    inline fn tBegin(self: *const LlamaRunner) cpu_profile.Span {
        return cpu_profile.begin(self.profile, self.io);
    }

    /// Close a timing span opened by `tBegin`, charging the elapsed ns to `cat`.
    inline fn tEnd(self: *const LlamaRunner, t0: cpu_profile.Span, cat: ProfileCat) void {
        cpu_profile.end(self.profile, self.io, t0, cat);
    }

    // -- step ---------------------------------------------------------------

    /// One forward step on slot 0: consume `token_id` at `pos`, and return `vocab_size` logits.
    /// The runner owns the slice and reuses it on the next call.
    pub fn step(self: *LlamaRunner, token_id: u32, pos: u32) Error![]f32 {
        try self.stepBatch(&.{.{ .slot = 0, .token = token_id, .pos = pos, .want_logits = true }});
        return self.logits[0..self.card.vocab_size];
    }

    /// Consume `ids` on slot 0 at `pos..pos + ids.len` in one batched step, and return the
    /// logits of the last row only. Callers chunk a longer prompt by `prefill_batch_max`.
    pub fn prefillBatch(self: *LlamaRunner, ids: []const u32, pos: u32) Error![]f32 {
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

    /// Advance every row one position: embed, all layers, then final norm and lm_head for the
    /// `want_logits` suffix only. The header describes the compact logits.
    pub fn stepBatch(self: *LlamaRunner, rows: []const StepRow) Error!void {
        _ = try step_row.validateRows(rows, self.n_slots, self.max_rows, self.max_seq);
        log.debug(
            "step: {d} rows, slot {d} pos {d}..{d}",
            .{ rows.len, rows[0].slot, rows[0].pos, rows[rows.len - 1].pos },
        );
        const c = self.card;
        const d = c.embedding_length;
        if (comptime core_log.hot) {
            if (self.profile) |p| {
                p.steps += 1;
            }
        }

        // Embed each token.
        const t_embed = self.tBegin();
        for (rows, 0..) |r, i| {
            ops.embedLookup(self.embed.*, r.token, d, self.x[i * d ..][0..d]);
        }

        self.tEnd(t_embed, .embed);

        // Run all layers.
        const live = (comptime core_log.hot) and core_log.enabled(.layer, .debug);
        var layer: u32 = 0;
        while (layer < self.n_layers) : (layer += 1) {
            var t_layer: std.Io.Timestamp = undefined;
            if (live) {
                t_layer = core_log.now();
            }

            self.layerForward(layer, rows);
            if (live) {
                log_layer.debug(
                    "layer {d}: {d} rows, {d:.2} ms",
                    .{ layer, rows.len, core_log.msSince(t_layer) },
                );
            }
        }

        // Project the want-logits suffix, if any.
        var first: usize = rows.len;
        while (first > 0 and rows[first - 1].want_logits) {
            first -= 1;
        }

        if (first == rows.len) return;

        const t_norm = self.tBegin();
        for (rows[first..], first..) |_, i| {
            ops.rmsnorm(
                self.x[i * d ..][0..d],
                fp32View(self.output_norm.*),
                c.rms_norm_eps,
                self.x_norm[i * d ..][0..d],
            );
        }

        self.tEnd(t_norm, .rmsnorm);
        const n_logits = rows.len - first;
        self.linearBatch(
            self.x_norm[first * d .. rows.len * d],
            self.output.*,
            self.logits[0..],
            n_logits,
            .lm_head,
        );
        std.debug.assert(first + n_logits == rows.len);
    }

    /// One decoder layer over all rows. It writes KV for every row before any row attends, so
    /// consecutive rows of one slot see exactly their causal prefix. Public for a composing runner.
    pub fn layerForward(self: *LlamaRunner, layer: u32, rows: []const StepRow) void {
        const sp = span.beginHot(.layer_forward);
        defer sp.end();

        std.debug.assert(layer < self.n_layers);
        std.debug.assert(rows.len <= self.max_rows);

        self.attentionBlock(layer, rows);
        self.ffnBlock(layer, rows.len);
    }

    /// The first half of a layer: norm, QKV, RoPE, KV commit, attention, and the residual add.
    fn attentionBlock(self: *LlamaRunner, layer: u32, rows: []const StepRow) void {
        const c = self.card;
        const d = c.embedding_length;
        const qd = c.n_heads * c.head_dim;
        const kvd = c.n_kv_heads * c.head_dim;
        const n = rows.len;

        // Normalize the residual stream.
        const t_an = self.tBegin();
        for (0..n) |i| {
            ops.rmsnorm(
                self.x[i * d ..][0..d],
                fp32View(self.attn_norm_w[layer].*),
                c.rms_norm_eps,
                self.x_norm[i * d ..][0..d],
            );
        }

        self.tEnd(t_an, .rmsnorm);

        // Project Q, K, and V.
        self.linearBatch(self.x_norm[0 .. n * d], self.attn_q_w[layer].*, self.q, n, .qkv_proj);
        self.linearBatch(
            self.x_norm[0 .. n * d],
            self.attn_k_w[layer].*,
            self.k_scratch,
            n,
            .qkv_proj,
        );
        self.linearBatch(
            self.x_norm[0 .. n * d],
            self.attn_v_w[layer].*,
            self.v_scratch,
            n,
            .qkv_proj,
        );

        // Rotate each row at its own position, then commit its K/V to its slot.
        const t_rope = self.tBegin();
        for (rows, 0..) |r, i| {
            var h: u32 = 0;
            while (h < c.n_heads) : (h += 1) {
                const off = i * qd + h * c.head_dim;
                ops.ropeInplace(
                    self.q[off..][0..c.head_dim],
                    r.pos,
                    c.rope_dim_count,
                    c.rope_freq_base,
                    .norm,
                    self.rope_factors,
                );
            }

            h = 0;
            while (h < c.n_kv_heads) : (h += 1) {
                const off = i * kvd + h * c.head_dim;
                ops.ropeInplace(
                    self.k_scratch[off..][0..c.head_dim],
                    r.pos,
                    c.rope_dim_count,
                    c.rope_freq_base,
                    .norm,
                    self.rope_factors,
                );
            }

            // The one conversion boundary. The header describes the KV commit.
            switch (self.kv_dtype) {
                .f32 => {
                    @memcpy(
                        self.kv_slots[r.slot].kDst(layer, r.pos),
                        self.k_scratch[i * kvd ..][0..kvd],
                    );
                    @memcpy(
                        self.kv_slots[r.slot].vDst(layer, r.pos),
                        self.v_scratch[i * kvd ..][0..kvd],
                    );
                },
                .f16 => {
                    const k_dst = self.kv_slots[r.slot].kDstAs(f16, layer, r.pos);
                    const v_dst = self.kv_slots[r.slot].vDstAs(f16, layer, r.pos);
                    const k_src = self.k_scratch[i * kvd ..][0..kvd];
                    const v_src = self.v_scratch[i * kvd ..][0..kvd];
                    for (0..kvd) |e| {
                        k_dst[e] = @floatCast(k_src[e]);
                        v_dst[e] = @floatCast(v_src[e]);
                    }
                },
            }
        }

        self.tEnd(t_rope, .rope);

        // Attend causally per row over positions 0..=pos. The CPU path fans the parts over threads.
        const t_attn = self.tBegin();
        for (rows, 0..) |r, i| {
            const t_len: usize = @intCast(r.pos + 1);
            if (comptime build_options.use_vulkan_ops) {
                // Flag resolution refuses every dtype but f32 on this build,
                // because `wrap.attentionSingle` uploads the slab as f32.
                std.debug.assert(self.kv_dtype == .f32);

                ops.attentionSingle(
                    self.q[i * qd ..][0..qd],
                    c.n_heads,
                    c.head_dim,
                    self.kv_slots[r.slot].kSlice(layer, r.pos),
                    self.kv_slots[r.slot].vSlice(layer, r.pos),
                    c.n_kv_heads,
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
                        self.q[i * qd ..][0..qd],
                        c.n_heads,
                        c.head_dim,
                        self.kv_slots[r.slot].kSlice(layer, r.pos),
                        self.kv_slots[r.slot].vSlice(layer, r.pos),
                        c.n_kv_heads,
                        t_len,
                        self.attn_parts,
                        self.attn_out[i * qd ..][0..qd],
                    ),
                    .f16 => parallel.attentionTyped(
                        f16,
                        ops,
                        self.pool,
                        self.policy,
                        self.q[i * qd ..][0..qd],
                        c.n_heads,
                        c.head_dim,
                        self.kv_slots[r.slot].kSliceAs(f16, layer, r.pos),
                        self.kv_slots[r.slot].vSliceAs(f16, layer, r.pos),
                        c.n_kv_heads,
                        t_len,
                        self.attn_parts,
                        self.attn_out[i * qd ..][0..qd],
                    ),
                }
            }
        }

        self.tEnd(t_attn, .attention);

        // Project the attention output and add the residual.
        self.linearBatch(
            self.attn_out[0 .. n * qd],
            self.attn_o_w[layer].*,
            self.x_norm,
            n,
            .o_proj,
        );
        const t_res1 = self.tBegin();
        for (0..n) |i| {
            ops.residualAdd(self.x_norm[i * d ..][0..d], self.x[i * d ..][0..d]);
        }

        self.tEnd(t_res1, .residual);
    }

    /// The second half of a layer: norm, SwiGLU feed-forward, and the residual add.
    fn ffnBlock(self: *LlamaRunner, layer: u32, n: usize) void {
        const c = self.card;
        const d = c.embedding_length;

        // Normalize for the feed-forward block.
        const t_fn = self.tBegin();
        for (0..n) |i| {
            ops.rmsnorm(
                self.x[i * d ..][0..d],
                fp32View(self.ffn_norm_w[layer].*),
                c.rms_norm_eps,
                self.x_norm[i * d ..][0..d],
            );
        }

        self.tEnd(t_fn, .rmsnorm);

        // Gate and up projections, then SwiGLU.
        self.linearBatch(
            self.x_norm[0 .. n * d],
            self.ffn_gate_w[layer].*,
            self.ffn_gate,
            n,
            .gate_up_proj,
        );
        self.linearBatch(
            self.x_norm[0 .. n * d],
            self.ffn_up_w[layer].*,
            self.ffn_up,
            n,
            .gate_up_proj,
        );
        const fd = c.feed_forward_length;
        const t_sg = self.tBegin();
        for (0..n) |i| {
            ops.swiglu(
                self.ffn_gate[i * fd ..][0..fd],
                self.ffn_up[i * fd ..][0..fd],
                self.ffn_gate[i * fd ..][0..fd],
            );
        }

        self.tEnd(t_sg, .swiglu);

        // Down projection and the second residual add.
        self.linearBatch(
            self.ffn_gate[0 .. n * fd],
            self.ffn_down_w[layer].*,
            self.ffn_down,
            n,
            .down_proj,
        );
        const t_res2 = self.tBegin();
        for (0..n) |i| {
            ops.residualAdd(self.ffn_down[i * d ..][0..d], self.x[i * d ..][0..d]);
        }

        self.tEnd(t_res2, .residual);
    }

    /// Copy KV rows `0..pos` of the source slot into the target slot, for a session fork.
    /// Rows past `pos` stay stale, because attention reads only `0..=pos`.
    pub fn cloneSlotState(self: *LlamaRunner, source_slot: u32, target_slot: u32, pos: u32) void {
        std.debug.assert(source_slot < self.n_slots);
        std.debug.assert(target_slot < self.n_slots);
        std.debug.assert(source_slot != target_slot);
        std.debug.assert(pos <= self.max_seq);

        self.kv_slots[target_slot].copyFrom(&self.kv_slots[source_slot], pos);
    }

    // -- tiered kv ----------------------------------------------------------

    /// Bytes that the first `n_rows` KV positions of one slot occupy in a host blob.
    pub fn evictSlotBytesFor(self: *const LlamaRunner, n_rows: u32) u64 {
        const c = self.card;
        const row_bytes: u64 = self.kv_dtype.rowBytes(c.n_kv_heads * c.head_dim);
        return @as(u64, 2) * self.n_layers * n_rows * row_bytes;
    }

    /// Copy the first `n_rows` KV positions of `slot` into `out`.
    pub fn evictSlot(self: *LlamaRunner, slot: u32, n_rows: u32, out: []u8) !void {
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);
        std.debug.assert(out.len == @as(usize, @intCast(self.evictSlotBytesFor(n_rows))));

        const kv = &self.kv_slots[slot];
        var off: usize = 0;
        var layer: u32 = 0;
        while (layer < self.n_layers) : (layer += 1) {
            for ([_][]const u8{
                kv.kRowsBytes(layer, n_rows),
                kv.vRowsBytes(layer, n_rows),
            }) |bytes| {
                @memcpy(out[off..][0..bytes.len], bytes);
                off += bytes.len;
            }
        }

        std.debug.assert(off == out.len);
    }

    /// Restore a previously-saved `n_rows`-row blob into `slot`.
    pub fn restoreSlot(self: *LlamaRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        std.debug.assert(slot < self.n_slots);
        std.debug.assert(n_rows >= 1); // a blob must hold rows to restore from
        std.debug.assert(n_rows <= self.max_seq);
        std.debug.assert(blob.len == @as(usize, @intCast(self.evictSlotBytesFor(n_rows))));

        const kv = &self.kv_slots[slot];
        var off: usize = 0;
        var layer: u32 = 0;
        while (layer < self.n_layers) : (layer += 1) {
            for ([_][]u8{
                kv.kRowsBytesMut(layer, n_rows),
                kv.vRowsBytesMut(layer, n_rows),
            }) |bytes| {
                @memcpy(bytes, blob[off..][0..bytes.len]);
                off += bytes.len;
            }
        }

        std.debug.assert(off == blob.len);
    }

    // -- matmul dispatch ----------------------------------------------------

    /// Bind the pool and the policy to `linear_dispatch.linearBatchDispatch`. Every matmul of the
    /// step passes through here, so the CPU timer charges the matmul categories in one place.
    fn linearBatch(
        r: *const LlamaRunner,
        inputs: []const f32,
        w: Tensor,
        outs: []f32,
        batch: usize,
        cat: ProfileCat,
    ) void {
        std.debug.assert(cat.isMatmul());

        const t0 = r.tBegin();
        defer r.tEnd(t0, cat);

        linear_dispatch.linearBatchDispatch(r.pool, r.policy, inputs, w, outs, batch);
    }
};

// -- tests ------------------------------------------------------------------

const test_fakes = @import("test_fakes.zig");
const test_fixtures = @import("../core/test_fixtures.zig");

test "cpu profile: the categories partition one decode step" {
    // The instrument exists only in a `-Dtrace` build. The GPU per-op backend touches a device.
    if (comptime !core_log.hot or build_options.use_vulkan_ops) {
        return error.SkipZigTest;
    } else {
        const io = std.testing.io;
        var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena_state.deinit();

        const arena = arena_state.allocator();
        const fixture = try test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

        var loaded = try loader.load(io, arena, fixture);
        defer loaded.close();

        const max_seq = @min(loaded.card.context_length, test_fakes.test_window);
        var pool: Pool = undefined;
        try pool.init(io, 1, 0);
        defer pool.deinit();

        var r = try LlamaRunner.init(
            arena,
            &loaded,
            kv_budget.pinned(1, max_seq),
            0,
            io,
            &pool,
            .{ .n_threads = 1 },
            .f32,
        );

        // One unprofiled step faults in the weight pages, so the measured step pays no first touch.
        _ = try r.step(1, 0);

        var prof: cpu_profile.CpuProfile = .{ .family = .llama };
        r.profile = &prof;
        const t0 = std.Io.Clock.Timestamp.now(io, .awake);
        _ = try r.step(2, 1);
        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        r.profile = null;
        const wall: u64 = @intCast(t0.durationTo(t1).raw.toNanoseconds());

        try std.testing.expectEqual(@as(u64, 1), prof.steps);
        // Cover: one decode step charges every llama category, and a qwen category stays at zero.
        for (std.enums.values(ProfileCat)) |cat| {
            if (cat.usedBy(.llama)) {
                try std.testing.expect(prof.nsOf(cat) > 0);
            } else {
                try std.testing.expectEqual(@as(u64, 0), prof.nsOf(cat));
            }
        }

        // Disjoint: two overlapping brackets charge one nanosecond twice, past the wall time.
        try std.testing.expect(prof.total() <= wall);
        // Whole: the uncounted remainder is row validation and loop bookkeeping.
        try std.testing.expect(prof.total() * 2 >= wall);
        // The pool share is a subset of the whole, because the step also runs serial categories.
        try std.testing.expect(prof.nsParallel() > 0);
        try std.testing.expect(prof.nsParallel() < prof.total());
    }
}
