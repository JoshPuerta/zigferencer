//! Backs the CPU op API (`linearF32Q80`, `rmsnorm`, `swiglu`, `residualAdd`)
//! with the GPU kernels in `kernels.zig`. The inference Runner then drives the
//! GPU through the exact op signatures it already calls on the CPU. The CPU op
//! API has no whole-row attention op, so this module exports its own
//! `attentionSingle` for the per-op branch of the runners. The backing is hybrid. The heavy ops run
//! on the GPU. The small or
//! fragmented ones stay on the CPU: the embedding lookup, the per-head RoPE, and
//! the f32 norms. Their microsecond cost beats the fixed overhead of a dispatch.
//!
//! This is the simple end-to-end path. Every op call is its own submit and
//! wait-idle, roughly 0.25 ms times hundreds of ops per Llama-8B token, which is
//! the known floor. The fused decode path in `runtime/llama_fused.zig` records
//! one command buffer per token and removes that floor. The simple boundary
//! earns its place beside the fused one, because the per-op benchmarks and the
//! GPU-against-CPU correctness checks run against it.
//!
//! Which side each op lands on:
//!
//!   embedLookup                       CPU. A pure memcpy loses to a dispatch.
//!   ropeInplace                       CPU. One dispatch per head, 32 per layer,
//!                                     costs more than the CPU microseconds.
//!   linearF32                         CPU. Only the small f32 norm weights.
//!   rmsnorm, swiglu, residualAdd,     GPU. The heavy ops that repay a dispatch.
//!   attentionSingle, linearF32Q80

const std = @import("std");
const build_options = @import("build_options");
const op_requires = @import("../core/op_requires.zig");
const naive = @import("../ops/naive.zig");
const kernels = @import("kernels.zig");
const vulkan_ctx = @import("../runtime/vulkan_ctx.zig");
const vulkan_runner = @import("../runtime/vulkan_runner.zig");
const Tensor = @import("../core/tensor.zig").Tensor;

comptime {
    if (!build_options.use_vulkan) {
        @compileError("ops_vulkan/wrap.zig was included in a non-vulkan build. " ++
            "It can only be selected as the ops backend when -Druntime=vulkan.");
    }
}

// -- constants --------------------------------------------------------------

pub const requires: op_requires.OpRequires = .{
    .arch = .any,
    .runtime = .vulkan,
    // The per-op tolerances vary: the matmul near 2e-4, rmsnorm near 1e-6, and
    // attention near 1e-3. 2e-3 is the loosest of those with a 2x cushion. It
    // holds end to end, because no op in the chain widens the drift past what
    // its own `requires.tolerance` already permits.
    .tolerance = .{ .relative = 2e-3 },
};

// -- types ------------------------------------------------------------------

pub const InitError = vulkan_runner.Runner.InitError;

// -- state ------------------------------------------------------------------

//
// These globals are deliberate. They hold the Runner that owns the GPU context
// and the kernels, the VRAM weight cache, and the lazily built GdnDecode
// pipeline. The deliberation is worth recording, because the engine otherwise
// keeps no globals at all.
//
// The op signatures set the constraint. This module backs the CPU op signatures
// with GPU kernels. The inference Runner then drives the GPU through the exact
// functions it already calls on the CPU. Those signatures are
// `linearF32Q80(input, weight, out)`, `rmsnorm(...)`, and so on. Every one takes
// values and returns values, with no handle to thread a `*Runner` through.
// Adding one puts a GPU object into the op signatures that every backend
// compiles against, which is the coupling the value-at-the-boundary and
// encapsulate-state rules forbid. A parallel GPU-only op API duplicates the same
// signatures the runner is built against. So the GPU lifetime stays confined
// here instead: `initBackend` creates it, `deinitBackend` tears it down, and
// nothing outside this file observes it. The contrast is the fused path in
// `runtime/llama_fused.zig`. Its unit of work is a whole forward pass, owned by
// one explicit LlamaFusedRunner value. A per-op function has no such natural owner.
var g_runner: ?*vulkan_runner.Runner = null;
var g_weight_cache: std.AutoHashMap(usize, kernels.Q80Weight) = undefined;
var g_initialized: bool = false;

/// The GPU Gated-DeltaNet pipeline, built on the first `gdnStep` call. It lives
/// here rather than in `vulkan_runner.Runner`, so the shared per-op bundle
/// carries no Qwen-only pipeline.
var g_gdn: ?kernels.GdnDecode = null;

// -- api --------------------------------------------------------------------

// The CPU passthroughs. Each one re-exports the naive op unchanged.
pub const dequantQ80 = naive.dequantQ80;
pub const embedLookup = naive.embedLookup;
pub const linearF32 = naive.linearF32;
pub const linearF32Batch = naive.linearF32Batch;
pub const ropeInplace = naive.ropeInplace;

// -- lifecycle --------------------------------------------------------------
pub fn initBackend(alloc: std.mem.Allocator, options: vulkan_ctx.Options) InitError!void {
    if (g_initialized) return;

    g_runner = try vulkan_runner.Runner.initWith(alloc, options);
    g_weight_cache = std.AutoHashMap(usize, kernels.Q80Weight).init(alloc);
    g_initialized = true;
}

pub fn deinitBackend() void {
    if (!g_initialized) return;

    if (g_gdn) |*g| {
        g.deinit();
        g_gdn = null;
    }

    var it = g_weight_cache.valueIterator();
    while (it.next()) |w| {
        w.deinit();
    }

    g_weight_cache.deinit();
    g_runner.?.deinit();
    g_runner = null;
    g_initialized = false;
}

// -- gpu ops ----------------------------------------------------------------
pub fn linearF32Q80(input: []const f32, weight: Tensor, out: []f32) void {
    std.debug.assert(weight.dtype == .q8_0);
    std.debug.assert(weight.n_dims == 2);

    const r = runner();

    // VRAM weight cache. The key is the pointer of the slice, so a weight that
    // is mapped once and referenced many times stages exactly once. The cache
    // never evicts, because weights live in VRAM for the life of the program.
    const key: usize = @intFromPtr(weight.data.ptr);
    const gop = g_weight_cache.getOrPut(key) catch @panic("ops_vulkan.wrap: weight cache OOM");
    if (!gop.found_existing) {
        const in_features: usize = @intCast(weight.shape[0]);
        const out_features: usize = @intCast(weight.shape[1]);
        gop.value_ptr.* = kernels.Q80Weight.init(
            &r.ctx,
            weight.data,
            in_features,
            out_features,
        ) catch |e|
            std.debug.panic("ops_vulkan.wrap: Q80Weight.init failed: {t}", .{e});
    }

    r.linear_q8_0_reduce.dispatch(gop.value_ptr, input, out) catch |e|
        std.debug.panic("ops_vulkan.wrap: linear_q8_0_reduce.dispatch failed: {t}", .{e});
}

/// Per-row batch over the GPU matvec: one dispatch per row, with no weight-reuse
/// win beyond the VRAM cache it already has. The GPU batching of the engine
/// lives in the fused multi-stream path, not in this per-op backend.
pub fn linearF32Q80Batch(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    naive.linearBatchPerRow(linearF32Q80, inputs, weight, outs, batch, out_stride);
}

pub fn rmsnorm(x: []const f32, w: []const f32, eps: f32, dst: []f32) void {
    runner().rmsnorm.run(x, w, eps, dst) catch |e|
        std.debug.panic("ops_vulkan.wrap: rmsnorm failed: {t}", .{e});
}

pub fn swiglu(gate: []const f32, up: []const f32, dst: []f32) void {
    runner().swiglu.run(gate, up, dst) catch |e|
        std.debug.panic("ops_vulkan.wrap: swiglu failed: {t}", .{e});
}

pub fn residualAdd(x: []const f32, dst: []f32) void {
    runner().residual_add.run(x, dst) catch |e|
        std.debug.panic("ops_vulkan.wrap: residualAdd failed: {t}", .{e});
}

pub fn attentionSingle(
    q: []const f32,
    n_heads: usize,
    head_dim: usize,
    k_cache: []const f32,
    v_cache: []const f32,
    n_kv_heads: usize,
    t_len: usize,
    out: []f32,
) void {
    runner().attention.run(
        q,
        k_cache,
        v_cache,
        @intCast(n_heads),
        @intCast(n_kv_heads),
        @intCast(head_dim),
        @intCast(t_len),
        out,
    ) catch |e| std.debug.panic("ops_vulkan.wrap: attention failed: {t}", .{e});
}

/// Run one Gated-DeltaNet decode step on the GPU, the Qwen3.5 linear-attention
/// layer. It mirrors `gdn.step`: it mutates `state` and `conv_state` and writes
/// `o_out`. `head_dim` must be 128. It panics on a GPU failure like the other
/// ops here. The kernel is bit-checked against `gdn.step`.
pub fn gdnStep(
    n_k_heads: u32,
    n_v_heads: u32,
    head_dim: u32,
    conv_kernel: u32,
    qkv: []const f32,
    z: []const f32,
    alpha: []const f32,
    beta_in: []const f32,
    conv_w: []const f32,
    ssm_a: []const f32,
    dt_bias: []const f32,
    ssm_norm: []const f32,
    eps: f32,
    state: []f32,
    conv_state: []f32,
    o_out: []f32,
) void {
    if (g_gdn == null) {
        g_gdn = kernels.GdnDecode.init(&runner().ctx) catch |e|
            std.debug.panic("ops_vulkan.wrap: GdnDecode.init failed: {t}", .{e});
    }

    g_gdn.?.run(
        n_k_heads,
        n_v_heads,
        head_dim,
        conv_kernel,
        qkv,
        z,
        alpha,
        beta_in,
        conv_w,
        ssm_a,
        dt_bias,
        ssm_norm,
        eps,
        state,
        conv_state,
        o_out,
    ) catch |e|
        std.debug.panic("ops_vulkan.wrap: gdnStep failed: {t}", .{e});
}

// -- helpers ----------------------------------------------------------------

fn runner() *vulkan_runner.Runner {
    return g_runner orelse @panic("ops_vulkan.wrap: initBackend must be called before any op. " ++
        "Hook bench_model_cmd / run_cmd to call it at process start.");
}
