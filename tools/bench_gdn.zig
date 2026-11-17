//! Gated-DeltaNet (Qwen3.5 hybrid) benchmarks. The recurrent GDN layer is the one op of the
//! hybrid that no other section covers, and its two paths cost very different amounts.
//! `gdn.step` is the reference of the CPU op. The naive CPU Qwen runner drives the GDN layers
//! with its key-head unit, and its line runs on every build. The `simd_pf` line runs the lane step
//! that the SIMD backends drive, `gdn_simd.kernels.step`, against that reference. The GPU lines
//! for `GdnDecode` (one token) and `GdnChunkParallel` (a whole chunk of prefill) run under vulkan.
//!
//! Lane error. The rel_err column of the `simd_pf` line holds the normwise error
//! `max |ref - got| / max |ref|`. The gated norm multiplies by `silu(z)`, so some readouts lie
//! near zero, and the largest per-element relative error measures their rounding, not the step.
//!
//! Comptime. `GdnCpu` takes its `gdn.Kernels` type at compile time, so the naive and the `simd_pf`
//! lines share one body. `build_options.use_vulkan` gates the GPU code, so a CPU build never reads
//! it.
//!
//! Not covered. The Qwen full-attention helpers that exist only as `recordInto` (QkNorm and the
//! batched rmsnorm, rope, and attention) go into the fused command buffer, so no section here
//! runs them standalone. `bench-model <qwen> --profile-prefill` and `--profile-decode` time
//! them instead.
//!
//! Inputs. A deterministic pattern fills the inputs, so the reference hash stays stable. One CPU
//! reference serves both sections: decode runs it at one token, and prefill at 64 tokens.
//!
//! Chunk tolerance. The chunked-parallel scan reorders the recurrence (log-space cumulative
//! decays) against the sequential CPU product-space reference. So the readout drifts about
//! 1.5e-2 under a pure-relative metric. `gpu-selftest gdn` gates it more tightly with a combined
//! abs+rel tolerance. Here the token hash is the exact-snapshot signal, and 2e-2 is the rough
//! gate.

const std = @import("std");
const build_options = @import("build_options");
const zigferencer = @import("zigferencer");
const op_requires = zigferencer.op_requires;

const bench_util = @import("bench_util.zig");
const cpuRow = bench_util.cpuRow;
const gpuRow = bench_util.gpuRow;
const flush = bench_util.flush;
const printSectionHeader = bench_util.printSectionHeader;
const hashOut = bench_util.hashOut;

// -- constants --------------------------------------------------------------

const head_dim: u32 = 128; // GdnDecode requires head_dim == 128.
const conv_kernel: u32 = 4;
const eps: f32 = 1e-6;
const chunk_tokens: usize = 64;

// -- types ------------------------------------------------------------------

const Cfg = struct { label: []const u8, n_k: u32, n_v: u32 };

/// The inputs of a run of `tokens` GDN steps at one model size. A decode row uses one token.
const GdnInputs = struct {
    dims: zigferencer.gdn.Dims,
    n_k: u32,
    n_v: u32,
    tokens: usize,
    conv_dim: usize,
    value_dim: usize,
    state_len: usize,
    conv_state_len: usize,
    qkv: []f32,
    zg: []f32,
    alpha: []f32,
    beta: []f32,
    conv_w: []f32,
    ssm_a: []f32,
    dt_bias: []f32,
    ssm_norm: []f32,
    state0: []f32,
    conv0: []f32,

    /// Allocate from `arena` and fill with the deterministic pattern.
    fn init(arena: std.mem.Allocator, tokens: usize, cf: Cfg) !GdnInputs {
        // Size the buffers from the config.
        const key_dim = @as(usize, cf.n_k) * head_dim;
        const value_dim = @as(usize, cf.n_v) * head_dim;
        const conv_dim = 2 * key_dim + value_dim;
        const state_len = value_dim * head_dim;
        const conv_state_len = @as(usize, conv_kernel - 1) * conv_dim;
        const n_v: usize = cf.n_v;

        // Allocate the buffers.
        const self: GdnInputs = .{
            .dims = .{
                .n_k_heads = cf.n_k,
                .n_v_heads = cf.n_v,
                .head_dim = head_dim,
                .conv_kernel = conv_kernel,
            },
            .n_k = cf.n_k,
            .n_v = cf.n_v,
            .tokens = tokens,
            .conv_dim = conv_dim,
            .value_dim = value_dim,
            .state_len = state_len,
            .conv_state_len = conv_state_len,
            .qkv = try arena.alloc(f32, tokens * conv_dim),
            .zg = try arena.alloc(f32, tokens * value_dim),
            .alpha = try arena.alloc(f32, tokens * n_v),
            .beta = try arena.alloc(f32, tokens * n_v),
            .conv_w = try arena.alloc(f32, @as(usize, conv_kernel) * conv_dim),
            .ssm_a = try arena.alloc(f32, n_v),
            .dt_bias = try arena.alloc(f32, n_v),
            .ssm_norm = try arena.alloc(f32, head_dim),
            .state0 = try arena.alloc(f32, state_len),
            .conv0 = try arena.alloc(f32, conv_state_len),
        };

        // Fill the buffers with test patterns.
        fillGdn(
            self.qkv,
            self.zg,
            self.alpha,
            self.beta,
            self.conv_w,
            self.ssm_a,
            self.dt_bias,
            self.ssm_norm,
            self.state0,
            self.conv0,
        );
        return self;
    }
};

/// The reference of one row: the CPU output and its hash, the work, and the timing scale.
const GdnRef = struct {
    label: []const u8,
    scale: usize,
    work: f64,
    out: []const f32,
    hash: u64,
};

/// One CPU line: `tokens` sequential `Kernels.step` calls that carry state and conv forward.
/// Each call mutates its qkv row, so the scratch copies keep the inputs intact.
fn GdnCpu(comptime Kernels: type) type {
    return struct {
        inputs: *const GdnInputs,
        sr: []f32,
        cr: []f32,
        qr: []f32,
        o: []f32,

        /// Allocate the scratch from `arena`, before the timed loop.
        fn init(arena: std.mem.Allocator, inputs: *const GdnInputs, o: []f32) !@This() {
            return .{
                .inputs = inputs,
                .sr = try arena.alloc(f32, inputs.state_len),
                .cr = try arena.alloc(f32, inputs.conv_state_len),
                .qr = try arena.alloc(f32, inputs.conv_dim),
                .o = o,
            };
        }

        pub fn run(self: @This()) !void {
            const in = self.inputs;
            @memcpy(self.sr, in.state0);
            @memcpy(self.cr, in.conv0);
            for (0..in.tokens) |ti| {
                @memcpy(self.qr, in.qkv[ti * in.conv_dim ..][0..in.conv_dim]);
                Kernels.step(
                    in.dims,
                    self.qr,
                    in.zg[ti * in.value_dim ..][0..in.value_dim],
                    in.alpha[ti * in.n_v ..][0..in.n_v],
                    in.beta[ti * in.n_v ..][0..in.n_v],
                    in.conv_w,
                    in.ssm_a,
                    in.dt_bias,
                    in.ssm_norm,
                    eps,
                    self.sr,
                    self.cr,
                    self.o[ti * in.value_dim ..][0..in.value_dim],
                );
            }
        }
    };
}

// -- api --------------------------------------------------------------------

pub fn benchGdn(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("gdn decode step (gated-deltanet, 1 token)");
    const cfgs = [_]Cfg{
        .{ .label = "gdn 2B (16k/16v d128)", .n_k = 16, .n_v = 16 },
        .{ .label = "gdn 9B (16k/32v d128)", .n_k = 16, .n_v = 32 },
    };

    for (cfgs) |cf| {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();

        const arena = arena_state.allocator();
        const inputs = try GdnInputs.init(arena, 1, cf);
        const ref = try gdnCpuRow(arena, io, &inputs, cf.label);

        if (comptime build_options.use_vulkan) {
            if (runner) |r| {
                try gdnDecodeGpuRow(arena, io, r, &inputs, ref);
            }
        }

        flush();
    }
}

pub fn benchGdnChunked(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("gdn chunked prefill (gated-deltanet, T tokens)");
    const cfgs = [_]Cfg{
        .{ .label = "gdn 2B T=64 (16k/16v)", .n_k = 16, .n_v = 16 },
        .{ .label = "gdn 9B T=64 (16k/32v)", .n_k = 16, .n_v = 32 },
    };

    for (cfgs) |cf| {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();

        const arena = arena_state.allocator();
        const inputs = try GdnInputs.init(arena, chunk_tokens, cf);
        const ref = try gdnCpuRow(arena, io, &inputs, cf.label);

        if (comptime build_options.use_vulkan) {
            if (runner) |r| {
                try gdnChunkGpuRow(arena, io, r, &inputs, ref);
            }
        }

        flush();
    }
}

// -- helpers ----------------------------------------------------------------

/// Run the CPU reference once, print the `naive` line and the `simd_pf` line, and return the
/// reference for the GPU line. The scratch is allocated before the timed loop, which keeps
/// allocation out of the measurement.
fn gdnCpuRow(
    arena: std.mem.Allocator,
    io: std.Io,
    inputs: *const GdnInputs,
    label: []const u8,
) !GdnRef {
    const o_ref = try arena.alloc(f32, inputs.tokens * inputs.value_dim);
    const cpu = try GdnCpu(zigferencer.gdn.reference).init(arena, inputs, o_ref);
    try cpu.run(); // populate o_ref as the reference
    // The work is the dxd recurrent state, which is also what GDN time scales with.
    const scale = inputs.tokens * inputs.state_len;
    const ref: GdnRef = .{
        .label = label,
        .scale = scale,
        .work = @floatFromInt(scale),
        .out = o_ref,
        .hash = hashOut(o_ref),
    };
    try cpuRow(io, ref.scale, label, "naive", ref.work, ref.hash, cpu);

    // The lane step of the SIMD backends, against the reference.
    const o_lane = try arena.alloc(f32, inputs.tokens * inputs.value_dim);
    const lane = try GdnCpu(zigferencer.gdn_simd.kernels).init(arena, inputs, o_lane);
    try lane.run();
    const norm = bench_util.normwiseError(o_ref, o_lane);
    const hash = hashOut(o_lane);
    const tol = zigferencer.ops_simd_pf.requires.tolerance;
    const t = try bench_util.timed(io, ref.scale, lane);
    const status = bench_util.opStatus(tol, norm, hash == ref.hash);
    bench_util.rawRow(label, "simd_pf", t.ms, bench_util.rate(t, ref.work), norm, hash, status);
    return ref;
}

/// The GPU row of the one-token decode kernel.
fn gdnDecodeGpuRow(
    arena: std.mem.Allocator,
    io: std.Io,
    runner: anytype,
    inputs: *const GdnInputs,
    ref: GdnRef,
) !void {
    // Build the GPU kernel and its output.
    var kern = try zigferencer.ops_vulkan.GdnDecode.init(&runner.ctx);
    defer kern.deinit();

    const g_o = try arena.alloc(f32, inputs.value_dim);

    // Wrap the GPU call as a bench case.
    const Gpu = struct {
        k: *zigferencer.ops_vulkan.GdnDecode,
        inputs: *const GdnInputs,
        state_w: []f32,
        conv_wk: []f32,
        o: []f32,

        pub fn run(self: @This()) !void {
            const in = self.inputs;
            @memcpy(self.state_w, in.state0);
            @memcpy(self.conv_wk, in.conv0);
            try self.k.run(
                in.n_k,
                in.n_v,
                head_dim,
                conv_kernel,
                in.qkv,
                in.zg,
                in.alpha,
                in.beta,
                in.conv_w,
                in.ssm_a,
                in.dt_bias,
                in.ssm_norm,
                eps,
                self.state_w,
                self.conv_wk,
                self.o,
            );
        }
    };

    // Run the GPU row against the reference.
    const tol: op_requires.Tolerance = .{ .relative = 5e-3 };
    try gpuRow(io, ref.scale, ref.label, "vk_gdn", ref.work, tol, ref.out, ref.hash, g_o, Gpu{
        .k = &kern,
        .inputs = inputs,
        .state_w = try arena.alloc(f32, inputs.state_len),
        .conv_wk = try arena.alloc(f32, inputs.conv_state_len),
        .o = g_o,
    });
}

/// The GPU row of the chunk-parallel prefill kernel.
fn gdnChunkGpuRow(
    arena: std.mem.Allocator,
    io: std.Io,
    runner: anytype,
    inputs: *const GdnInputs,
    ref: GdnRef,
) !void {
    var kern = try zigferencer.ops_vulkan.GdnChunkParallel.init(&runner.ctx);
    defer kern.deinit();

    const g_o = try arena.alloc(f32, inputs.tokens * inputs.value_dim);
    const Gpu = struct {
        k: *zigferencer.ops_vulkan.GdnChunkParallel,
        inputs: *const GdnInputs,
        state_w: []f32,
        conv_wk: []f32,
        o: []f32,

        pub fn run(self: @This()) !void {
            const in = self.inputs;
            @memcpy(self.state_w, in.state0);
            @memcpy(self.conv_wk, in.conv0);
            try self.k.runChunkParallel(
                @intCast(in.tokens),
                in.n_k,
                in.n_v,
                head_dim,
                conv_kernel,
                in.qkv,
                in.zg,
                in.alpha,
                in.beta,
                in.conv_w,
                in.ssm_a,
                in.dt_bias,
                in.ssm_norm,
                eps,
                self.state_w,
                self.conv_wk,
                self.o,
            );
        }
    };
    // The chunked scan reorders the recurrence, so the readout drifts about 1.5e-2.
    // The token hash is the exact signal, and 2e-2 is the rough gate.
    const tol: op_requires.Tolerance = .{ .relative = 2e-2 };
    try gpuRow(io, ref.scale, ref.label, "vk_gdn_chunk", ref.work, tol, ref.out, ref.hash, g_o, Gpu{
        .k = &kern,
        .inputs = inputs,
        .state_w = try arena.alloc(f32, inputs.state_len),
        .conv_wk = try arena.alloc(f32, inputs.conv_state_len),
        .o = g_o,
    });
}

fn fillGdn(
    qkv: []f32,
    zg: []f32,
    alpha: []f32,
    beta: []f32,
    conv_w: []f32,
    ssm_a: []f32,
    dt_bias: []f32,
    ssm_norm: []f32,
    state0: []f32,
    conv0: []f32,
) void {
    // Fill the projections with smooth waves.
    for (qkv, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.011);
    }

    for (zg, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)) * 0.013);
    }

    for (alpha, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    for (beta, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)) * 0.1) * 0.5 + 0.5;
    }

    for (conv_w, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.02) * 0.5;
    }

    // Fill the decay and the norm parameters.
    for (ssm_a) |*v| {
        v.* = -1.0; // -exp(A_log): negative decay
    }

    for (dt_bias, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i))) * 0.3;
    }

    for (ssm_norm, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i))) * 0.2 + 1.0;
    }

    // Fill the recurrent and conv states.
    for (state0, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.005) * 0.1;
    }

    for (conv0, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)) * 0.03) * 0.3;
    }
}
