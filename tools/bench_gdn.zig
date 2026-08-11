//! Gated-DeltaNet (Qwen3.5 hybrid) benchmarks. The recurrent GDN layer is the
//! one op of the hybrid that no other section covers, and its two paths cost
//! very different amounts. `gdn.step` is a real CPU op, because the CPU Qwen
//! runner drives the GDN layers with it, so its reference row runs on every
//! build. The GPU rows for `GdnDecode` (one token) and `GdnChunkParallel` (a
//! whole chunk of prefill) run under vulkan.
//!
//! The Qwen full-attention helpers that exist only as `recordInto` (QkNorm and
//! the batched rmsnorm, rope, and attention) go into the fused command buffer,
//! so no section here runs them standalone. `bench-model <qwen>
//! --profile-prefill` and `--profile-decode` price them instead.

const std = @import("std");
const build_options = @import("build_options");
const z = @import("zigferencer");
const op_requires = z.op_requires;

const h = @import("bench_util.zig");
const cpuRow = h.cpuRow;
const gpuRow = h.gpuRow;
const flush = h.flush;
const printSectionHeader = h.printSectionHeader;
const hashOut = h.hashOut;

// -- api --------------------------------------------------------------------

pub fn benchGdn(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("gdn decode step (gated-deltanet, 1 token)");
    const head_dim: u32 = 128; // GdnDecode requires head_dim == 128.
    const conv_kernel: u32 = 4;
    const eps: f32 = 1e-6;
    const Cfg = struct { label: []const u8, n_k: u32, n_v: u32 };
    const cfgs = [_]Cfg{
        .{ .label = "gdn 2B (16k/16v d128)", .n_k = 16, .n_v = 16 },
        .{ .label = "gdn 9B (16k/32v d128)", .n_k = 16, .n_v = 32 },
    };

    for (cfgs) |cf| {
        const key_dim = @as(usize, cf.n_k) * head_dim;
        const value_dim = @as(usize, cf.n_v) * head_dim;
        const conv_dim = 2 * key_dim + value_dim;
        const state_len = value_dim * head_dim;
        const conv_state_len = @as(usize, conv_kernel - 1) * conv_dim;

        const qkv0 = try gpa.alloc(f32, conv_dim);
        defer gpa.free(qkv0);
        const zg = try gpa.alloc(f32, value_dim);
        defer gpa.free(zg);
        const alpha = try gpa.alloc(f32, cf.n_v);
        defer gpa.free(alpha);
        const beta = try gpa.alloc(f32, cf.n_v);
        defer gpa.free(beta);
        const conv_w = try gpa.alloc(f32, @as(usize, conv_kernel) * conv_dim);
        defer gpa.free(conv_w);
        const ssm_a = try gpa.alloc(f32, cf.n_v);
        defer gpa.free(ssm_a);
        const dt_bias = try gpa.alloc(f32, cf.n_v);
        defer gpa.free(dt_bias);
        const ssm_norm = try gpa.alloc(f32, head_dim);
        defer gpa.free(ssm_norm);
        const state0 = try gpa.alloc(f32, state_len);
        defer gpa.free(state0);
        const conv0 = try gpa.alloc(f32, conv_state_len);
        defer gpa.free(conv0);
        fillGdn(qkv0, zg, alpha, beta, conv_w, ssm_a, dt_bias, ssm_norm, state0, conv0);

        // The CPU reference runs one token. `gdn.step` mutates qkv, state, and
        // conv, so the reference works on copies.
        const dims: z.gdn.Dims = .{ .n_k_heads = cf.n_k, .n_v_heads = cf.n_v, .head_dim = head_dim, .conv_kernel = conv_kernel };
        const o_ref = try gpa.alloc(f32, value_dim);
        defer gpa.free(o_ref);
        {
            const qr = try gpa.dupe(f32, qkv0);
            defer gpa.free(qr);
            const sr = try gpa.dupe(f32, state0);
            defer gpa.free(sr);
            const cr = try gpa.dupe(f32, conv0);
            defer gpa.free(cr);
            z.gdn.step(dims, qr, zg, alpha, beta, conv_w, ssm_a, dt_bias, ssm_norm, eps, sr, cr, o_ref);
        }
        const ref_hash = hashOut(o_ref);
        const work: f64 = @floatFromInt(state_len); // throughput over the dxd state.
        const scale = state_len; // GDN work scales with the dxd recurrent state.

        // CPU timing closure: reset the mutated buffers each call.
        const qkv_w = try gpa.alloc(f32, conv_dim);
        defer gpa.free(qkv_w);
        const state_w = try gpa.alloc(f32, state_len);
        defer gpa.free(state_w);
        const conv_wk = try gpa.alloc(f32, conv_state_len);
        defer gpa.free(conv_wk);
        const o_w = try gpa.alloc(f32, value_dim);
        defer gpa.free(o_w);
        const Cpu = struct {
            d: z.gdn.Dims,
            eps: f32,
            qkv0: []const f32,
            zg: []const f32,
            alpha: []const f32,
            beta: []const f32,
            conv_w: []const f32,
            ssm_a: []const f32,
            dt_bias: []const f32,
            ssm_norm: []const f32,
            state0: []const f32,
            conv0: []const f32,
            qkv_w: []f32,
            state_w: []f32,
            conv_wk: []f32,
            o: []f32,
            pub fn run(self: @This()) !void {
                @memcpy(self.qkv_w, self.qkv0);
                @memcpy(self.state_w, self.state0);
                @memcpy(self.conv_wk, self.conv0);
                z.gdn.step(self.d, self.qkv_w, self.zg, self.alpha, self.beta, self.conv_w, self.ssm_a, self.dt_bias, self.ssm_norm, self.eps, self.state_w, self.conv_wk, self.o);
            }
        };
        try cpuRow(io, scale, cf.label, "naive", work, ref_hash, Cpu{
            .d = dims,
            .eps = eps,
            .qkv0 = qkv0,
            .zg = zg,
            .alpha = alpha,
            .beta = beta,
            .conv_w = conv_w,
            .ssm_a = ssm_a,
            .dt_bias = dt_bias,
            .ssm_norm = ssm_norm,
            .state0 = state0,
            .conv0 = conv0,
            .qkv_w = qkv_w,
            .state_w = state_w,
            .conv_wk = conv_wk,
            .o = o_w,
        });

        if (comptime build_options.use_vulkan) if (runner) |r| {
            var kern = try z.ops_vulkan.GdnDecode.init(&r.ctx);
            defer kern.deinit();
            const g_state = try gpa.alloc(f32, state_len);
            defer gpa.free(g_state);
            const g_conv = try gpa.alloc(f32, conv_state_len);
            defer gpa.free(g_conv);
            const g_o = try gpa.alloc(f32, value_dim);
            defer gpa.free(g_o);
            const Gpu = struct {
                k: *z.ops_vulkan.GdnDecode,
                n_k: u32,
                n_v: u32,
                head_dim: u32,
                conv_kernel: u32,
                eps: f32,
                qkv0: []const f32,
                zg: []const f32,
                alpha: []const f32,
                beta: []const f32,
                conv_w: []const f32,
                ssm_a: []const f32,
                dt_bias: []const f32,
                ssm_norm: []const f32,
                state0: []const f32,
                conv0: []const f32,
                state_w: []f32,
                conv_wk: []f32,
                o: []f32,
                pub fn run(self: @This()) !void {
                    @memcpy(self.state_w, self.state0);
                    @memcpy(self.conv_wk, self.conv0);
                    try self.k.run(self.n_k, self.n_v, self.head_dim, self.conv_kernel, self.qkv0, self.zg, self.alpha, self.beta, self.conv_w, self.ssm_a, self.dt_bias, self.ssm_norm, self.eps, self.state_w, self.conv_wk, self.o);
                }
            };
            const tol: op_requires.Tolerance = .{ .relative = 5e-3 };
            try gpuRow(io, scale, cf.label, "vk_gdn", work, tol, o_ref, ref_hash, g_o, Gpu{
                .k = &kern,
                .n_k = cf.n_k,
                .n_v = cf.n_v,
                .head_dim = head_dim,
                .conv_kernel = conv_kernel,
                .eps = eps,
                .qkv0 = qkv0,
                .zg = zg,
                .alpha = alpha,
                .beta = beta,
                .conv_w = conv_w,
                .ssm_a = ssm_a,
                .dt_bias = dt_bias,
                .ssm_norm = ssm_norm,
                .state0 = state0,
                .conv0 = conv0,
                .state_w = g_state,
                .conv_wk = g_conv,
                .o = g_o,
            });
        };
        flush();
    }
}

pub fn benchGdnChunked(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("gdn chunked prefill (gated-deltanet, T tokens)");
    const head_dim: u32 = 128;
    const conv_kernel: u32 = 4;
    const eps: f32 = 1e-6;
    const T: usize = 64;
    const Cfg = struct { label: []const u8, n_k: u32, n_v: u32 };
    const cfgs = [_]Cfg{
        .{ .label = "gdn 2B T=64 (16k/16v)", .n_k = 16, .n_v = 16 },
        .{ .label = "gdn 9B T=64 (16k/32v)", .n_k = 16, .n_v = 32 },
    };

    for (cfgs) |cf| {
        const key_dim = @as(usize, cf.n_k) * head_dim;
        const value_dim = @as(usize, cf.n_v) * head_dim;
        const conv_dim = 2 * key_dim + value_dim;
        const state_len = value_dim * head_dim;
        const conv_state_len = @as(usize, conv_kernel - 1) * conv_dim;
        const n_v: usize = cf.n_v;

        const qkv = try gpa.alloc(f32, T * conv_dim);
        defer gpa.free(qkv);
        const zg = try gpa.alloc(f32, T * value_dim);
        defer gpa.free(zg);
        const alpha = try gpa.alloc(f32, T * n_v);
        defer gpa.free(alpha);
        const beta = try gpa.alloc(f32, T * n_v);
        defer gpa.free(beta);
        const conv_w = try gpa.alloc(f32, @as(usize, conv_kernel) * conv_dim);
        defer gpa.free(conv_w);
        const ssm_a = try gpa.alloc(f32, n_v);
        defer gpa.free(ssm_a);
        const dt_bias = try gpa.alloc(f32, n_v);
        defer gpa.free(dt_bias);
        const ssm_norm = try gpa.alloc(f32, head_dim);
        defer gpa.free(ssm_norm);
        const state0 = try gpa.alloc(f32, state_len);
        defer gpa.free(state0);
        const conv0 = try gpa.alloc(f32, conv_state_len);
        defer gpa.free(conv0);
        // The filler writes the per-token inputs across the whole T*width
        // buffers. The per-layer inputs reuse the same filler.
        fillGdn(qkv, zg, alpha, beta, conv_w, ssm_a, dt_bias, ssm_norm, state0, conv0);

        const dims: z.gdn.Dims = .{ .n_k_heads = cf.n_k, .n_v_heads = cf.n_v, .head_dim = head_dim, .conv_kernel = conv_kernel };

        // The CPU reference runs T sequential `gdn.step` calls that carry state
        // and conv forward. Each call mutates its qkv row, so the reference
        // works on a scratch copy. The scratch is allocated before the timed
        // loop, which keeps allocation out of the measurement.
        const sr = try gpa.alloc(f32, state_len);
        defer gpa.free(sr);
        const cr = try gpa.alloc(f32, conv_state_len);
        defer gpa.free(cr);
        const qr = try gpa.alloc(f32, conv_dim);
        defer gpa.free(qr);
        const o_ref = try gpa.alloc(f32, T * value_dim);
        defer gpa.free(o_ref);

        const Cpu = struct {
            d: z.gdn.Dims,
            eps: f32,
            t: usize,
            conv_dim: usize,
            value_dim: usize,
            n_v: usize,
            qkv: []const f32,
            zg: []const f32,
            alpha: []const f32,
            beta: []const f32,
            conv_w: []const f32,
            ssm_a: []const f32,
            dt_bias: []const f32,
            ssm_norm: []const f32,
            state0: []const f32,
            conv0: []const f32,
            sr: []f32,
            cr: []f32,
            qr: []f32,
            o: []f32,
            pub fn run(self: @This()) !void {
                @memcpy(self.sr, self.state0);
                @memcpy(self.cr, self.conv0);
                var ti: usize = 0;
                while (ti < self.t) : (ti += 1) {
                    @memcpy(self.qr, self.qkv[ti * self.conv_dim ..][0..self.conv_dim]);
                    z.gdn.step(self.d, self.qr, self.zg[ti * self.value_dim ..][0..self.value_dim], self.alpha[ti * self.n_v ..][0..self.n_v], self.beta[ti * self.n_v ..][0..self.n_v], self.conv_w, self.ssm_a, self.dt_bias, self.ssm_norm, self.eps, self.sr, self.cr, self.o[ti * self.value_dim ..][0..self.value_dim]);
                }
            }
        };
        const cpu = Cpu{
            .d = dims,
            .eps = eps,
            .t = T,
            .conv_dim = conv_dim,
            .value_dim = value_dim,
            .n_v = n_v,
            .qkv = qkv,
            .zg = zg,
            .alpha = alpha,
            .beta = beta,
            .conv_w = conv_w,
            .ssm_a = ssm_a,
            .dt_bias = dt_bias,
            .ssm_norm = ssm_norm,
            .state0 = state0,
            .conv0 = conv0,
            .sr = sr,
            .cr = cr,
            .qr = qr,
            .o = o_ref,
        };
        try cpu.run(); // populate o_ref as the reference
        const ref_hash = hashOut(o_ref);
        const work: f64 = @floatFromInt(T * state_len);
        const scale = T * state_len;
        try cpuRow(io, scale, cf.label, "naive", work, ref_hash, cpu);

        if (comptime build_options.use_vulkan) if (runner) |r| {
            var kern = try z.ops_vulkan.GdnChunkParallel.init(&r.ctx);
            defer kern.deinit();
            const g_state = try gpa.alloc(f32, state_len);
            defer gpa.free(g_state);
            const g_conv = try gpa.alloc(f32, conv_state_len);
            defer gpa.free(g_conv);
            const g_o = try gpa.alloc(f32, T * value_dim);
            defer gpa.free(g_o);
            const Gpu = struct {
                k: *z.ops_vulkan.GdnChunkParallel,
                t: u32,
                n_k: u32,
                n_v: u32,
                head_dim: u32,
                conv_kernel: u32,
                eps: f32,
                qkv: []const f32,
                zg: []const f32,
                alpha: []const f32,
                beta: []const f32,
                conv_w: []const f32,
                ssm_a: []const f32,
                dt_bias: []const f32,
                ssm_norm: []const f32,
                state0: []const f32,
                conv0: []const f32,
                state_w: []f32,
                conv_wk: []f32,
                o: []f32,
                pub fn run(self: @This()) !void {
                    @memcpy(self.state_w, self.state0);
                    @memcpy(self.conv_wk, self.conv0);
                    try self.k.runChunkParallel(self.t, self.n_k, self.n_v, self.head_dim, self.conv_kernel, self.qkv, self.zg, self.alpha, self.beta, self.conv_w, self.ssm_a, self.dt_bias, self.ssm_norm, self.eps, self.state_w, self.conv_wk, self.o);
                }
            };
            // The chunked-parallel scan reorders the recurrence (log-space
            // cumulative decays) vs the sequential CPU product-space reference, so
            // the readout drifts ~1.5e-2 under a pure-relative metric. `gpu-selftest
            // gdn` gates it more tightly with a combined abs+rel tolerance. Here the
            // token hash is the exact-snapshot signal and 2e-2 is the rough gate.
            const tol: op_requires.Tolerance = .{ .relative = 2e-2 };
            try gpuRow(io, scale, cf.label, "vk_gdn_chunk", work, tol, o_ref, ref_hash, g_o, Gpu{
                .k = &kern,
                .t = @intCast(T),
                .n_k = cf.n_k,
                .n_v = cf.n_v,
                .head_dim = head_dim,
                .conv_kernel = conv_kernel,
                .eps = eps,
                .qkv = qkv,
                .zg = zg,
                .alpha = alpha,
                .beta = beta,
                .conv_w = conv_w,
                .ssm_a = ssm_a,
                .dt_bias = dt_bias,
                .ssm_norm = ssm_norm,
                .state0 = state0,
                .conv0 = conv0,
                .state_w = g_state,
                .conv_wk = g_conv,
                .o = g_o,
            });
        };
        flush();
    }
}

// -- helpers ----------------------------------------------------------------

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
    for (qkv, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.011);
    for (zg, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.013);
    for (alpha, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.1);
    for (beta, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.1) * 0.5 + 0.5;
    for (conv_w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.02) * 0.5;
    for (ssm_a) |*v| v.* = -1.0; // -exp(A_log): negative decay
    for (dt_bias, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i))) * 0.3;
    for (ssm_norm, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i))) * 0.2 + 1.0;
    for (state0, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.005) * 0.1;
    for (conv0, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.03) * 0.3;
}
