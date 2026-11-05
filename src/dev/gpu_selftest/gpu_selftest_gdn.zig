//! The Gated DeltaNet checks of `gpu-selftest`: the decode step and the chunked prefill scan,
//! against the CPU reference in `ops/gdn.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");

const zigferencer = @import("zigferencer");

const context = zigferencer.cli.context;
const gdn = zigferencer.gdn;
// These imports resolve to `void` on a CPU build, so the file still compiles there.
const vulkan_ctx = if (build_options.use_vulkan) zigferencer.vulkan_ctx else void;
const kernels = if (build_options.use_vulkan) zigferencer.ops_vulkan else void;

const gpu_selftest_common = @import("gpu_selftest_common.zig");

// -- constants --------------------------------------------------------------

/// GDN head counts as `{ n_k, n_v }`: the 2B config, then the decoupled 9B config.
const gdn_decode_heads = [_][2]u32{ .{ 16, 16 }, .{ 16, 32 } };

/// GDN chunk cases as `{ n_k, n_v, n_tokens }`.
const gdn_chunk_cases = [_][3]u32{ .{ 16, 16, 7 }, .{ 16, 32, 7 } };

/// Chunked-parallel cases as `{ n_k, n_v, n_tokens }`. The token counts span the
/// sub-chunk size of 16. The runner records the 2B chunk at the widths 64 and 128,
/// and the width 100 leaves a tail of four tokens in the last sub-chunk.
const gdn_parallel_cases = [_][3]u32{
    .{ 16, 16, 7 },
    .{ 16, 32, 7 },
    .{ 16, 16, 33 },
    .{ 16, 32, 128 },
    .{ 16, 16, 64 },
    .{ 16, 16, 100 },
    .{ 16, 16, 128 },
};

const gdn_head_dim: u32 = 128;
const gdn_conv_kernel: u32 = 4;
const gdn_eps: f32 = 1e-6;

// -- types ------------------------------------------------------------------

/// The random inputs of one GDN check. `n_tokens` is 1 for the decode step.
const GdnInputs = struct {
    n_k: u32,
    n_v: u32,
    n_tokens: u32,
    qkv: []f32,
    z: []f32,
    alpha: []f32,
    beta_in: []f32,
    conv_w: []f32,
    ssm_a: []f32,
    dt_bias: []f32,
    ssm_norm: []f32,
    state0: []f32,
    conv0: []f32,
};

/// What one GDN run produces: every readout row, the final state, and the final conv_state.
const GdnOutputs = struct { o: []f32, state: []f32, conv: []f32 };

// -- api --------------------------------------------------------------------

/// Check the GDN GPU kernels against the CPU `gdn.step`: the decode step, the
/// sequential chunk path, and the chunked-parallel prefill, on one Context.
pub fn gdnCheck(ctx: context.Context) !void {
    var ctxp = try ctx.arena.create(vulkan_ctx.Context);
    ctxp.* = try vulkan_ctx.Context.init(
        ctx.arena,
        .{ .storage_range_override = ctx.vk_storage_range },
    );
    defer ctxp.deinit();

    var k = try kernels.GdnDecode.init(ctxp);
    defer k.deinit();

    var kp = try kernels.GdnChunkParallel.init(ctxp);
    defer kp.deinit();

    var all_ok = true;
    for (gdn_decode_heads) |h| {
        if (!try gdnCheckDims(ctx, &k, h[0], h[1])) {
            all_ok = false;
        }
    }

    // One dispatch loops T tokens, so the compare shows that `recordChunk` carries
    // the state and the conv_state.
    for (gdn_chunk_cases) |c| {
        if (!try gdnChunkCheckDims(ctx, &k, c[0], c[1], c[2])) {
            all_ok = false;
        }
    }

    for (gdn_parallel_cases) |c| {
        if (!try gdnChunkParallelCheckDims(ctx, &kp, c[0], c[1], c[2])) {
            all_ok = false;
        }
    }

    if (!all_ok) return error.BitCheckFailed;
}

// -- helpers ----------------------------------------------------------------

/// Check the chunked-parallel prefill (a conv, a scan, and a norm dispatch) against
/// T CPU `gdn.step` calls that carry the same state and conv_state. This is the
/// parallel counterpart of `gdnChunkCheckDims`.
fn gdnChunkParallelCheckDims(
    ctx: context.Context,
    kp: *kernels.GdnChunkParallel,
    n_k: u32,
    n_v: u32,
    n_tokens: u32,
) !bool {
    // Wide alpha (+/-3) gives softplus up to ~3, then g down to ~-8, then decay
    // ~1e-4. The cumulative decay over a sub-chunk underflows in product space.
    // The log-space scan must stay accurate in this regime. The sequential CPU
    // step is safe, because it applies S *= decay one token at a time.
    const seed = 0x9A5EED + @as(u64, n_v) * 31 + n_tokens;
    const in = try gdnInputs(ctx.arena, n_k, n_v, n_tokens, seed, 3.0);
    const cpu = try gdnCpuReference(ctx.arena, in);
    const gpu = try gdnOutputsFor(ctx.arena, in);
    try kp.runChunkParallel(
        n_tokens,
        n_k,
        n_v,
        gdn_head_dim,
        gdn_conv_kernel,
        in.qkv,
        in.z,
        in.alpha,
        in.beta_in,
        in.conv_w,
        in.ssm_a,
        in.dt_bias,
        in.ssm_norm,
        gdn_eps,
        gpu.state,
        gpu.conv,
        gpu.o,
    );

    const w = &ctx.stdout.interface;
    try w.print(
        "GDN chunk-PARALLEL GPU vs CPU (n_k={d}, n_v={d}, T={d}):\n",
        .{ n_k, n_v, n_tokens },
    );
    return gdnCompare(w, cpu, gpu, 1e-4, 2e-3);
}

/// Check the sequential chunk path against T CPU `gdn.step` calls that carry the
/// same state and conv_state. The GPU `runChunk` is one dispatch with an internal
/// token loop. This is the prefill counterpart of `gdnCheckDims`.
fn gdnChunkCheckDims(
    ctx: context.Context,
    k: *kernels.GdnDecode,
    n_k: u32,
    n_v: u32,
    n_tokens: u32,
) !bool {
    const in = try gdnInputs(ctx.arena, n_k, n_v, n_tokens, 0x5EED + n_v, 0.5);
    const cpu = try gdnCpuReference(ctx.arena, in);
    const gpu = try gdnOutputsFor(ctx.arena, in);
    try k.runChunk(
        n_tokens,
        n_k,
        n_v,
        gdn_head_dim,
        gdn_conv_kernel,
        in.qkv,
        in.z,
        in.alpha,
        in.beta_in,
        in.conv_w,
        in.ssm_a,
        in.dt_bias,
        in.ssm_norm,
        gdn_eps,
        gpu.state,
        gpu.conv,
        gpu.o,
    );

    const w = &ctx.stdout.interface;
    try w.print("GDN chunk GPU vs CPU (n_k={d}, n_v={d}, T={d}):\n", .{ n_k, n_v, n_tokens });
    return gdnCompare(w, cpu, gpu, 1e-5, 1e-3);
}

fn gdnCheckDims(ctx: context.Context, k: *kernels.GdnDecode, n_k: u32, n_v: u32) !bool {
    const in = try gdnInputs(ctx.arena, n_k, n_v, 1, 0xC0FFEE + n_v, 0.5);
    const cpu = try gdnCpuReference(ctx.arena, in);
    const gpu = try gdnOutputsFor(ctx.arena, in);
    try k.run(
        n_k,
        n_v,
        gdn_head_dim,
        gdn_conv_kernel,
        in.qkv,
        in.z,
        in.alpha,
        in.beta_in,
        in.conv_w,
        in.ssm_a,
        in.dt_bias,
        in.ssm_norm,
        gdn_eps,
        gpu.state,
        gpu.conv,
        gpu.o,
    );

    const w = &ctx.stdout.interface;
    try w.print("GDN GPU vs CPU (n_k={d}, n_v={d}, head_dim={d}):\n", .{ n_k, n_v, gdn_head_dim });
    return gdnCompare(w, cpu, gpu, 1e-5, 1e-3);
}

/// Random inputs for `n_tokens` GDN tokens. `alpha_scale` sets the range of the decay input.
fn gdnInputs(
    a: Allocator,
    n_k: u32,
    n_v: u32,
    n_tokens: u32,
    seed: u64,
    alpha_scale: f32,
) !GdnInputs {
    const T: usize = n_tokens;
    const key_dim: usize = @as(usize, n_k) * gdn_head_dim;
    const value_dim: usize = @as(usize, n_v) * gdn_head_dim;
    const conv_dim: usize = 2 * key_dim + value_dim;

    const in: GdnInputs = .{
        .n_k = n_k,
        .n_v = n_v,
        .n_tokens = n_tokens,
        .qkv = try a.alloc(f32, T * conv_dim),
        .z = try a.alloc(f32, T * value_dim),
        .alpha = try a.alloc(f32, T * n_v),
        .beta_in = try a.alloc(f32, T * n_v),
        .conv_w = try a.alloc(f32, @as(usize, gdn_conv_kernel) * conv_dim),
        .ssm_a = try a.alloc(f32, n_v),
        .dt_bias = try a.alloc(f32, n_v),
        .ssm_norm = try a.alloc(f32, gdn_head_dim),
        .state0 = try a.alloc(f32, value_dim * gdn_head_dim),
        .conv0 = try a.alloc(f32, @as(usize, gdn_conv_kernel - 1) * conv_dim),
    };

    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    gpu_selftest_common.fillN(r, in.qkv, 1.0);
    gpu_selftest_common.fillN(r, in.z, 1.0);
    gpu_selftest_common.fillN(r, in.alpha, alpha_scale);
    gpu_selftest_common.fillN(r, in.beta_in, 1.0);
    gpu_selftest_common.fillN(r, in.conv_w, 0.5);
    gpu_selftest_common.fillN(r, in.dt_bias, 0.3);
    gpu_selftest_common.fillN(r, in.ssm_norm, 0.2);
    for (in.ssm_norm) |*v| {
        v.* += 1.0;
    }

    gpu_selftest_common.fillN(r, in.state0, 0.1);
    gpu_selftest_common.fillN(r, in.conv0, 0.3);
    for (in.ssm_a) |*v| {
        v.* = -@exp((r.float(f32) - 0.5) * 2.0); // -exp(A_log)
    }

    return in;
}

/// Output buffers that start from the same state and conv_state as the inputs.
fn gdnOutputsFor(a: Allocator, in: GdnInputs) !GdnOutputs {
    const value_dim: usize = @as(usize, in.n_v) * gdn_head_dim;
    return .{
        .o = try a.alloc(f32, @as(usize, in.n_tokens) * value_dim),
        .state = try a.dupe(f32, in.state0),
        .conv = try a.dupe(f32, in.conv0),
    };
}

/// Run the tokens through `gdn.step` one at a time.
fn gdnCpuReference(a: Allocator, in: GdnInputs) !GdnOutputs {
    const key_dim: usize = @as(usize, in.n_k) * gdn_head_dim;
    const value_dim: usize = @as(usize, in.n_v) * gdn_head_dim;
    const conv_dim: usize = 2 * key_dim + value_dim;
    const n_v: usize = in.n_v;
    const out = try gdnOutputsFor(a, in);
    const dims: gdn.Dims = .{
        .n_k_heads = in.n_k,
        .n_v_heads = in.n_v,
        .head_dim = gdn_head_dim,
        .conv_kernel = gdn_conv_kernel,
    };

    for (0..in.n_tokens) |ti| {
        // `gdn.step` mutates its qkv row, so every token gets a copy.
        const qkv_t = try a.dupe(f32, in.qkv[ti * conv_dim ..][0..conv_dim]);
        gdn.step(
            dims,
            qkv_t,
            in.z[ti * value_dim ..][0..value_dim],
            in.alpha[ti * n_v ..][0..n_v],
            in.beta_in[ti * n_v ..][0..n_v],
            in.conv_w,
            in.ssm_a,
            in.dt_bias,
            in.ssm_norm,
            gdn_eps,
            out.state,
            out.conv,
            out.o[ti * value_dim ..][0..value_dim],
        );
    }

    return out;
}

/// Print the three diffs and the verdict of one GDN check, and return whether it passed.
fn gdnCompare(w: *std.Io.Writer, cpu: GdnOutputs, gpu: GdnOutputs, atol: f32, rtol: f32) !bool {
    try gpu_selftest_common.printDiff(w, "o_out     ", cpu.o, gpu.o);
    try gpu_selftest_common.printDiff(w, "state     ", cpu.state, gpu.state);
    try gpu_selftest_common.printDiff(w, "conv_state", cpu.conv, gpu.conv);

    const fails = gpu_selftest_common.countFail(cpu.o, gpu.o, atol, rtol) +
        gpu_selftest_common.countFail(cpu.state, gpu.state, atol, rtol) +
        gpu_selftest_common.countFail(cpu.conv, gpu.conv, atol, rtol);
    const ok = fails == 0;
    try w.print("  fails={d}  RESULT: {s}\n\n", .{ fails, if (ok) "PASS" else "FAIL" });
    try w.flush();
    return ok;
}
