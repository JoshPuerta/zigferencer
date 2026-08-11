//! Per-layer op benchmarks: the elementwise ops (residual_add, swiglu, rmsnorm,
//! rope), fused single-head attention, the dispatch-amortization demo,
//! and the Qwen full-attention sigmoid gate. These ops sit between the matmuls
//! on the forward pass, so a regression here hides behind the matmul rows unless
//! it gets its own table. Each section runs the CPU reference on every build and
//! the GPU variant under vulkan. The attention section also walks a
//! context-depth grid on the two qwen full-attention shapes, and prices the f16
//! KV cache that those models default to.

const std = @import("std");
const build_options = @import("build_options");
const z = @import("zigferencer");
const op_requires = z.op_requires;

const h = @import("bench_util.zig");
const cpuRow = h.cpuRow;
const gpuRow = h.gpuRow;
const rawRow = h.rawRow;
const flush = h.flush;
const printSectionHeader = h.printSectionHeader;
const hashOut = h.hashOut;
const computeErrors = h.computeErrors;
const opStatus = h.opStatus;
const timed = h.timed;
const rate = h.rate;

// -- types ------------------------------------------------------------------

/// The fan-out of every threaded attention row. The thread count is fixed and
/// not detected, so the rows of two hosts compare.
const bench_threaded: z.parallel.Policy = .{ .n_threads = 16, .min_parallel_flops = 0 };

// -- api --------------------------------------------------------------------

/// The elementwise sections and the attention section. Every section has the
/// same shape: it prints a header, then for each case it allocates the fixtures,
/// runs the CPU naive reference, times the CPU loop and, under vulkan, the GPU
/// loop, and prints one row per variant.
pub fn benchElementwise(
    gpa: std.mem.Allocator,
    io: std.Io,
    runner: anytype,
) !void {
    printSectionHeader("residual_add");
    // Vector lengths covering hidden_dim across the three matmul bench scales,
    // plus the FFN width for Llama-8B (the largest residual in practice).
    const resid_cases = [_]struct { label: []const u8, n: usize }{
        .{ .label = "ew n=576    (hidden)", .n = 576 },
        .{ .label = "ew n=2048   (hidden)", .n = 2048 },
        .{ .label = "ew n=4096   (hidden)", .n = 4096 },
        .{ .label = "ew n=14336  (ffn)", .n = 14336 },
    };
    for (resid_cases) |c| {
        const x = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(x);
        const dst_init = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_init);
        const dst_ref = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_ref);
        const dst_work = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_work);
        for (x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)));
        for (dst_init, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)));

        @memcpy(dst_ref, dst_init);
        z.ops.residualAdd(x, dst_ref);
        const ref_hash = hashOut(dst_ref);
        const work: f64 = @floatFromInt(c.n);

        const Cpu = struct {
            x: []const f32,
            dst_init: []const f32,
            dst_work: []f32,
            pub fn run(self: @This()) !void {
                @memcpy(self.dst_work, self.dst_init);
                z.ops.residualAdd(self.x, self.dst_work);
            }
        };
        try cpuRow(io, c.n, c.label, "naive", work, ref_hash, Cpu{ .x = x, .dst_init = dst_init, .dst_work = dst_work });

        if (comptime build_options.use_vulkan) if (runner) |r| {
            const Gpu = struct {
                r: *z.vulkan_runner.Runner,
                x: []const f32,
                dst_init: []const f32,
                dst_work: []f32,
                pub fn run(self: @This()) !void {
                    @memcpy(self.dst_work, self.dst_init);
                    try self.r.residual_add.run(self.x, self.dst_work);
                }
            };
            try gpuRow(io, c.n, c.label, "vk_resid", work, z.ops_vulkan.ResidualAdd.requires.tolerance, dst_ref, ref_hash, dst_work, Gpu{ .r = r, .x = x, .dst_init = dst_init, .dst_work = dst_work });
        };
        flush();
    }

    printSectionHeader("swiglu");
    // FFN widths: SmolLM2 (1536), Llama-1B (8192), Llama-8B (14336).
    const swiglu_cases = [_]struct { label: []const u8, n: usize }{
        .{ .label = "swiglu n=1536  (ffn)", .n = 1536 },
        .{ .label = "swiglu n=8192  (ffn)", .n = 8192 },
        .{ .label = "swiglu n=14336 (ffn)", .n = 14336 },
    };
    for (swiglu_cases) |c| {
        const gate = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(gate);
        const up = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(up);
        const dst_ref = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_ref);
        const dst_work = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_work);
        for (gate, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)));
        for (up, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)));

        z.ops.swiglu(gate, up, dst_ref);
        const ref_hash = hashOut(dst_ref);
        const work: f64 = @floatFromInt(c.n);

        const Cpu = struct {
            gate: []const f32,
            up: []const f32,
            dst: []f32,
            pub fn run(self: @This()) !void {
                z.ops.swiglu(self.gate, self.up, self.dst);
            }
        };
        try cpuRow(io, c.n, c.label, "naive", work, ref_hash, Cpu{ .gate = gate, .up = up, .dst = dst_work });

        if (comptime build_options.use_vulkan) if (runner) |r| {
            const Gpu = struct {
                r: *z.vulkan_runner.Runner,
                gate: []const f32,
                up: []const f32,
                dst: []f32,
                pub fn run(self: @This()) !void {
                    try self.r.swiglu.run(self.gate, self.up, self.dst);
                }
            };
            try gpuRow(io, c.n, c.label, "vk_swiglu", work, z.ops_vulkan.Swiglu.requires.tolerance, dst_ref, ref_hash, dst_work, Gpu{ .r = r, .gate = gate, .up = up, .dst = dst_work });
        };
        flush();
    }

    printSectionHeader("rmsnorm");
    const rms_cases = [_]struct { label: []const u8, d: usize }{
        .{ .label = "rmsnorm d=576   (hidden)", .d = 576 },
        .{ .label = "rmsnorm d=2048  (hidden)", .d = 2048 },
        .{ .label = "rmsnorm d=4096  (hidden)", .d = 4096 },
    };
    const eps: f32 = 1e-5;
    for (rms_cases) |c| {
        const x = try gpa.alignedAlloc(f32, .of(f32), c.d);
        defer gpa.free(x);
        const w = try gpa.alignedAlloc(f32, .of(f32), c.d);
        defer gpa.free(w);
        const y_ref = try gpa.alignedAlloc(f32, .of(f32), c.d);
        defer gpa.free(y_ref);
        const y_work = try gpa.alignedAlloc(f32, .of(f32), c.d);
        defer gpa.free(y_work);
        for (x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)));
        for (w, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i))) * 0.5 + 1.0;

        z.ops.rmsnorm(x, w, eps, y_ref);
        const ref_hash = hashOut(y_ref);
        // 3 FLOPs/elem (mul, sq+add via FMA, plus the scale pass mul).
        const work: f64 = @as(f64, @floatFromInt(c.d)) * 3.0;

        const Cpu = struct {
            x: []const f32,
            w: []const f32,
            eps: f32,
            y: []f32,
            pub fn run(self: @This()) !void {
                z.ops.rmsnorm(self.x, self.w, self.eps, self.y);
            }
        };
        try cpuRow(io, c.d, c.label, "naive", work, ref_hash, Cpu{ .x = x, .w = w, .eps = eps, .y = y_work });

        if (comptime build_options.use_vulkan) if (runner) |r| {
            const Gpu = struct {
                r: *z.vulkan_runner.Runner,
                x: []const f32,
                w: []const f32,
                eps: f32,
                y: []f32,
                pub fn run(self: @This()) !void {
                    try self.r.rmsnorm.run(self.x, self.w, self.eps, self.y);
                }
            };
            try gpuRow(io, c.d, c.label, "vk_rmsnorm", work, z.ops_vulkan.Rmsnorm.requires.tolerance, y_ref, ref_hash, y_work, Gpu{ .r = r, .x = x, .w = w, .eps = eps, .y = y_work });
        };
        flush();
    }

    printSectionHeader("rope");
    const rope_cases = [_]struct { label: []const u8, head_dim: usize }{
        .{ .label = "rope head_dim=64", .head_dim = 64 },
        .{ .label = "rope head_dim=128", .head_dim = 128 },
        .{ .label = "rope head_dim=256", .head_dim = 256 }, // Qwen3.5 full-attn
    };
    const pos: u32 = 17;
    const freq_base: f32 = 10000.0;
    for (rope_cases) |c| {
        const x_init = try gpa.alignedAlloc(f32, .of(f32), c.head_dim);
        defer gpa.free(x_init);
        const x_ref = try gpa.alignedAlloc(f32, .of(f32), c.head_dim);
        defer gpa.free(x_ref);
        const x_work = try gpa.alignedAlloc(f32, .of(f32), c.head_dim);
        defer gpa.free(x_work);
        for (x_init, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)));

        @memcpy(x_ref, x_init);
        z.ops.ropeInplace(x_ref, pos, c.head_dim, freq_base, .neox, null);
        const ref_hash = hashOut(x_ref);
        const work: f64 = @floatFromInt(c.head_dim);

        const Cpu = struct {
            x_init: []const f32,
            x_work: []f32,
            pos: u32,
            head_dim: usize,
            freq_base: f32,
            pub fn run(self: @This()) !void {
                @memcpy(self.x_work, self.x_init);
                z.ops.ropeInplace(self.x_work, self.pos, self.head_dim, self.freq_base, .neox, null);
            }
        };
        try cpuRow(io, c.head_dim, c.label, "naive", work, ref_hash, Cpu{
            .x_init = x_init,
            .x_work = x_work,
            .pos = pos,
            .head_dim = c.head_dim,
            .freq_base = freq_base,
        });

        if (comptime build_options.use_vulkan) if (runner) |r| {
            const Gpu = struct {
                r: *z.vulkan_runner.Runner,
                x_init: []const f32,
                x_work: []f32,
                pos: u32,
                head_dim: usize,
                freq_base: f32,
                pub fn run(self: @This()) !void {
                    @memcpy(self.x_work, self.x_init);
                    try self.r.rope.run(self.x_work, self.pos, self.head_dim, self.freq_base, .neox, null);
                }
            };
            try gpuRow(io, c.head_dim, c.label, "vk_rope", work, z.ops_vulkan.Rope.requires.tolerance, x_ref, ref_hash, x_work, Gpu{
                .r = r,
                .x_init = x_init,
                .x_work = x_work,
                .pos = pos,
                .head_dim = c.head_dim,
                .freq_base = freq_base,
            });
        };
        flush();
    }

    printSectionHeader("fused attention");
    const Atn = struct {
        label: []const u8,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
        kv_f16: bool = false,
    };
    // The last ten cases walk a context-depth grid on the two qwen
    // full-attention shapes. Qwen3.5-2B states 8 query heads over 2 kv heads,
    // and Qwen3.5-9B states 16 over 4, both at head_dim 256. Attention is the
    // one per-token cost that grows with the context, so the grid is what
    // prices the CPU against the GPU at depth. Their f16 rows read the KV cache
    // in the dtype the qwen registry entries default to. Every attention case
    // also runs a 16-thread row through the attention control unit, and that
    // row must give the bits of the serial row. The `simd` rows run the vector
    // part kernel of the `simd` backend, and the `simd_pf` rows the prefetching
    // kernel of the `simd_pf` backend. Their rel_err column holds the error
    // against the naive row of the same dtype, and every threaded vector row
    // must give the bits of the serial `simd` row.
    // The GPU rows stay f32, because the per-op runner builds `Attention` for
    // f32 caches and the bench calls `Attention.run`. `Attention.runTyped`
    // takes f16 caches on an f16 pipeline, and the kernel tests and
    // `gpu-selftest attn` use it.
    const atn_cases = [_]Atn{
        .{ .label = "attn 9h,3kv,d64,t128", .n_heads = 9, .n_kv_heads = 3, .head_dim = 64, .t_len = 128 },
        .{ .label = "attn 32h,8kv,d64,t512", .n_heads = 32, .n_kv_heads = 8, .head_dim = 64, .t_len = 512 },
        .{ .label = "attn 32h,8kv,d128,t512", .n_heads = 32, .n_kv_heads = 8, .head_dim = 128, .t_len = 512 },
        // This shape is the widest GQA group the kernel blocks for. No loaded
        // model states it, so the row prices the block bound alone.
        .{ .label = "attn 16h,2kv,d256,t512 group 8", .n_heads = 16, .n_kv_heads = 2, .head_dim = 256, .t_len = 512 },
        .{ .label = "attn 8h,2kv,d256,t512", .n_heads = 8, .n_kv_heads = 2, .head_dim = 256, .t_len = 512, .kv_f16 = true },
        .{ .label = "attn 8h,2kv,d256,t1024", .n_heads = 8, .n_kv_heads = 2, .head_dim = 256, .t_len = 1024, .kv_f16 = true },
        .{ .label = "attn 8h,2kv,d256,t2048", .n_heads = 8, .n_kv_heads = 2, .head_dim = 256, .t_len = 2048, .kv_f16 = true },
        .{ .label = "attn 8h,2kv,d256,t4096", .n_heads = 8, .n_kv_heads = 2, .head_dim = 256, .t_len = 4096, .kv_f16 = true },
        .{ .label = "attn 8h,2kv,d256,t8192", .n_heads = 8, .n_kv_heads = 2, .head_dim = 256, .t_len = 8192, .kv_f16 = true },
        .{ .label = "attn 16h,4kv,d256,t512", .n_heads = 16, .n_kv_heads = 4, .head_dim = 256, .t_len = 512, .kv_f16 = true },
        .{ .label = "attn 16h,4kv,d256,t1024", .n_heads = 16, .n_kv_heads = 4, .head_dim = 256, .t_len = 1024, .kv_f16 = true },
        .{ .label = "attn 16h,4kv,d256,t2048", .n_heads = 16, .n_kv_heads = 4, .head_dim = 256, .t_len = 2048, .kv_f16 = true },
        .{ .label = "attn 16h,4kv,d256,t4096", .n_heads = 16, .n_kv_heads = 4, .head_dim = 256, .t_len = 4096, .kv_f16 = true },
        .{ .label = "attn 16h,4kv,d256,t8192", .n_heads = 16, .n_kv_heads = 4, .head_dim = 256, .t_len = 8192, .kv_f16 = true },
    };
    // One pool serves every threaded row of the section.
    var pool: z.pool.Pool = undefined;
    try z.parallel.initPool(&pool, io, bench_threaded);
    defer pool.deinit();
    for (atn_cases) |c| {
        const q_len: usize = @as(usize, c.n_heads) * @as(usize, c.head_dim);
        const kv_len: usize = @as(usize, c.t_len) * @as(usize, c.n_kv_heads) * @as(usize, c.head_dim);
        const q = try gpa.alignedAlloc(f32, .of(f32), q_len);
        defer gpa.free(q);
        const k_cache = try gpa.alignedAlloc(f32, .of(f32), kv_len);
        defer gpa.free(k_cache);
        const v_cache = try gpa.alignedAlloc(f32, .of(f32), kv_len);
        defer gpa.free(v_cache);
        const out_ref = try gpa.alignedAlloc(f32, .of(f32), q_len);
        defer gpa.free(out_ref);
        const out_work = try gpa.alignedAlloc(f32, .of(f32), q_len);
        defer gpa.free(out_work);
        const out_ref16 = try gpa.alignedAlloc(f32, .of(f32), q_len);
        defer gpa.free(out_ref16);
        const parts = try gpa.alloc(f32, z.attn_cpu.partialLen(c.n_heads, c.head_dim));
        defer gpa.free(parts);

        // Deterministic small-magnitude inputs so softmax stays numeric.
        for (q, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.07);
        for (k_cache, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.03);
        for (v_cache, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.05);

        // The f16 caches hold the same values at storage precision, so the gap
        // to the f32 row is the element type alone. A zero-length allocation
        // costs the f32-only cases nothing.
        const k16 = try gpa.alignedAlloc(f16, .of(f16), if (c.kv_f16) kv_len else 0);
        defer gpa.free(k16);
        const v16 = try gpa.alignedAlloc(f16, .of(f16), if (c.kv_f16) kv_len else 0);
        defer gpa.free(v16);
        for (k16, 0..) |*e, i| e.* = @floatCast(k_cache[i]);
        for (v16, 0..) |*e, i| e.* = @floatCast(v_cache[i]);

        z.ops.attentionRowTyped(f32, q, c.n_heads, c.head_dim, k_cache, v_cache, c.n_kv_heads, c.t_len, parts, out_ref);
        const ref_hash = hashOut(out_ref);
        // 4 FLOPs per (head * t * d): scores dot, exp, weighted-V dot, and the
        // normalize multiply.
        const total_work_elems = @as(usize, c.t_len) * @as(usize, c.head_dim) * @as(usize, c.n_heads);
        const work: f64 = @as(f64, @floatFromInt(total_work_elems)) * 4.0;

        const cpu32: AtnCpu(f32, z.ops) = .{
            .q = q,
            .n_heads = c.n_heads,
            .head_dim = c.head_dim,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .n_kv_heads = c.n_kv_heads,
            .t_len = c.t_len,
            .parts = parts,
            .out = out_work,
        };
        try cpuRow(io, total_work_elems, c.label, "naive", work, ref_hash, cpu32);
        var cpu32_mt = cpu32;
        cpu32_mt.pool = &pool;
        _ = try cpuCheckedRow(io, total_work_elems, c.label, "naive_mt", work, out_ref, ref_hash, cpu32_mt);
        try simdRows(f32, io, &pool, total_work_elems, c.label, "", work, out_ref, cpu32);

        if (c.kv_f16) {
            const cpu16: AtnCpu(f16, z.ops) = .{
                .q = q,
                .n_heads = c.n_heads,
                .head_dim = c.head_dim,
                .k_cache = k16,
                .v_cache = v16,
                .n_kv_heads = c.n_kv_heads,
                .t_len = c.t_len,
                .parts = parts,
                .out = out_work,
            };
            // The f16 row carries its own hash, because a narrower store moves
            // the output bits. A gate against the f32 hash claims an equality
            // that the dtype does not promise.
            try cpu16.run();
            @memcpy(out_ref16, out_work);
            const hash16 = hashOut(out_ref16);
            try cpuRow(io, total_work_elems, c.label, "naive_f16", work, hash16, cpu16);
            var cpu16_mt = cpu16;
            cpu16_mt.pool = &pool;
            _ = try cpuCheckedRow(io, total_work_elems, c.label, "naive_f16_mt", work, out_ref16, hash16, cpu16_mt);
            try simdRows(f16, io, &pool, total_work_elems, c.label, "_f16", work, out_ref16, cpu16);
        }

        if (comptime build_options.use_vulkan) if (runner) |r| {
            const Gpu = struct {
                r: *z.vulkan_runner.Runner,
                q: []const f32,
                k_cache: []const f32,
                v_cache: []const f32,
                n_heads: u32,
                n_kv_heads: u32,
                head_dim: u32,
                t_len: u32,
                out: []f32,
                pub fn run(self: @This()) !void {
                    try self.r.attention.run(self.q, self.k_cache, self.v_cache, self.n_heads, self.n_kv_heads, self.head_dim, self.t_len, self.out);
                }
            };
            try gpuRow(io, total_work_elems, c.label, "vk_atn", work, z.ops_vulkan.Attention.requires.tolerance, out_ref, ref_hash, out_work, Gpu{
                .r = r,
                .q = q,
                .k_cache = k_cache,
                .v_cache = v_cache,
                .n_heads = c.n_heads,
                .n_kv_heads = c.n_kv_heads,
                .head_dim = c.head_dim,
                .t_len = c.t_len,
                .out = out_work,
            });
        };
        flush();
    }

    if (comptime build_options.use_vulkan) if (runner) |r| {
        try benchAttnKitGrid(gpa, io, r);
        try benchDispatchAmortization(gpa, io, r);
    };
}

/// Qwen3.5 full-attention output gate: `dst *= sigmoid(gate)`. GPU-only kernel
/// with an inline CPU reference (there is no standalone CPU sigmoid-gate op).
pub fn benchSigmoidGate(gpa: std.mem.Allocator, io: std.Io, runner: *z.vulkan_runner.Runner) !void {
    printSectionHeader("sigmoid_gate (qwen full-attn output gate)");
    var kern = try z.ops_vulkan.SigmoidGate.init(&runner.ctx);
    defer kern.deinit();
    const cases = [_]struct { label: []const u8, n: usize }{
        .{ .label = "sigmoid n=2048 (qwen attn out)", .n = 2048 },
        .{ .label = "sigmoid n=4096", .n = 4096 },
    };
    for (cases) |c| {
        const gate = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(gate);
        const dst_init = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_init);
        const dst_ref = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_ref);
        const dst_work = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_work);
        for (gate, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
        for (dst_init, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.01);

        @memcpy(dst_ref, dst_init);
        for (dst_ref, gate) |*d, g| d.* *= 1.0 / (1.0 + @exp(-g));
        const ref_hash = hashOut(dst_ref);
        const work: f64 = @floatFromInt(c.n);

        const Gpu = struct {
            k: *z.ops_vulkan.SigmoidGate,
            gate: []const f32,
            dst_init: []const f32,
            dst_work: []f32,
            pub fn run(self: @This()) !void {
                @memcpy(self.dst_work, self.dst_init);
                try self.k.run(self.dst_work, self.gate);
            }
        };
        const tol: op_requires.Tolerance = .{ .relative = 1e-5 };
        try gpuRow(io, c.n, c.label, "vk_sigmoid", work, tol, dst_ref, ref_hash, dst_work, Gpu{ .k = &kern, .gate = gate, .dst_init = dst_init, .dst_work = dst_work });
        flush();
    }
}

// -- helpers ----------------------------------------------------------------

/// One CPU attention closure per KV storage type and kernel set. The two dtypes
/// differ only in the cache element type, so a single generic body keeps the
/// f32 and f16 rows from drifting apart. `Kernels` is an ops backend, and
/// `z.ops` is the naive reference. Without `pool` the closure runs the serial
/// row. With `pool` it runs the attention control unit over the kernels at
/// `bench_threaded`.
fn AtnCpu(comptime E: type, comptime Kernels: type) type {
    return struct {
        pool: ?*z.pool.Pool = null,
        q: []const f32,
        n_heads: u32,
        head_dim: u32,
        k_cache: []const E,
        v_cache: []const E,
        n_kv_heads: u32,
        t_len: u32,
        parts: []f32,
        out: []f32,
        pub fn run(self: @This()) !void {
            if (self.pool) |pool| {
                z.parallel.attentionTyped(E, Kernels, pool, bench_threaded, self.q, self.n_heads, self.head_dim, self.k_cache, self.v_cache, self.n_kv_heads, self.t_len, self.parts, self.out);
            } else {
                Kernels.attentionRowTyped(E, self.q, self.n_heads, self.head_dim, self.k_cache, self.v_cache, self.n_kv_heads, self.t_len, self.parts, self.out);
            }
        }

        /// The same inputs and output over the kernel set `K`, serial when
        /// `pool` is null.
        pub fn over(self: @This(), comptime K: type, pool: ?*z.pool.Pool) AtnCpu(E, K) {
            return .{
                .pool = pool,
                .q = self.q,
                .n_heads = self.n_heads,
                .head_dim = self.head_dim,
                .k_cache = self.k_cache,
                .v_cache = self.v_cache,
                .n_kv_heads = self.n_kv_heads,
                .t_len = self.t_len,
                .parts = self.parts,
                .out = self.out,
            };
        }
    };
}

/// The vector rows of one attention case: the serial `simd` row, then the
/// 16-thread rows of the `simd` and the `simd_pf` kernels. `naive_ctx` holds
/// the inputs and the output, and `ref_out` holds the naive row of the same
/// dtype.
fn simdRows(
    comptime E: type,
    io: std.Io,
    pool: *z.pool.Pool,
    scale: usize,
    label: []const u8,
    comptime suffix: []const u8,
    work: f64,
    ref_out: []const f32,
    naive_ctx: AtnCpu(E, z.ops),
) !void {
    const simd_hash = try cpuCheckedRow(io, scale, label, "simd" ++ suffix, work, ref_out, null, naive_ctx.over(z.ops_simd, null));
    _ = try cpuCheckedRow(io, scale, label, "simd" ++ suffix ++ "_mt", work, ref_out, simd_hash, naive_ctx.over(z.ops_simd, pool));
    // The prefetch changes the timing alone, so its row must give the same
    // bits.
    _ = try cpuCheckedRow(io, scale, label, "simd_pf" ++ suffix ++ "_mt", work, ref_out, simd_hash, naive_ctx.over(z.ops_simd_pf, pool));
}

/// One CPU row with a checked output. The row runs once before the timing, and
/// its error against `ref_out` fills the rel_err column. With `want_hash`, a
/// different output hash prints FAIL. A threaded row passes the hash of its
/// serial row, because the part geometry does not depend on the thread count.
/// Returns the output hash.
fn cpuCheckedRow(
    io: std.Io,
    scale: usize,
    label: []const u8,
    name: []const u8,
    work: f64,
    ref_out: []const f32,
    want_hash: ?u64,
    ctx: anytype,
) !u64 {
    try ctx.run();
    const got = hashOut(ctx.out);
    const errs = computeErrors(ref_out, ctx.out);
    const t = try timed(io, scale, ctx);
    const ok = if (want_hash) |want| got == want else true;
    rawRow(label, name, t.ms, rate(t, work), errs.rel, got, if (ok) "ok" else "FAIL");
    return got;
}

/// Dispatch amortization: how far the per-dispatch wall clock drops when N
/// dispatches share one command buffer and one submit. It runs
/// `ResidualAdd.runRepeat(N)` for a few N on one shape and prints the ms per
/// dispatch. The N=1 row is the per-call overhead of the `run()` path. A later
/// row shows how much of that overhead many ops amortize between them.
fn benchDispatchAmortization(
    gpa: std.mem.Allocator,
    io: std.Io,
    runner: *z.vulkan_runner.Runner,
) !void {
    printSectionHeader("fused dispatch amortization (residual_add)");
    const n_elems: usize = 4096; // Llama-8B hidden size, representative
    const x = try gpa.alignedAlloc(f32, .of(f32), n_elems);
    defer gpa.free(x);
    const dst_init = try gpa.alignedAlloc(f32, .of(f32), n_elems);
    defer gpa.free(dst_init);
    const dst_work = try gpa.alignedAlloc(f32, .of(f32), n_elems);
    defer gpa.free(dst_work);
    const dst_ref = try gpa.alignedAlloc(f32, .of(f32), n_elems);
    defer gpa.free(dst_ref);
    for (x, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)));
    for (dst_init, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)));

    const dispatch_counts = [_]u32{ 1, 4, 16, 64 };
    for (dispatch_counts) |n_dispatches| {
        // The CPU reference repeats the exact accumulation pattern of the GPU: N
        // sequential `dst += x` adds, and never `dst + N*x`. The two forms are
        // equal in exact arithmetic, but they differ at the last bit under f32
        // rounding, and the bit_exact rule needs one accumulation order.
        @memcpy(dst_ref, dst_init);
        var k: u32 = 0;
        while (k < n_dispatches) : (k += 1) z.ops.residualAdd(x, dst_ref);
        const ref_hash = hashOut(dst_ref);

        const Gpu = struct {
            r: *z.vulkan_runner.Runner,
            n: u32,
            x: []const f32,
            dst_init: []const f32,
            dst_work: []f32,
            pub fn run(self: @This()) !void {
                try self.r.residual_add.runRepeat(self.n, self.x, self.dst_init, self.dst_work);
            }
        };
        const gpu = Gpu{ .r = runner, .n = n_dispatches, .x = x, .dst_init = dst_init, .dst_work = dst_work };
        try gpu.run();
        const gpu_hash = hashOut(dst_work);
        const gpu_errs = computeErrors(dst_ref, dst_work);
        const gpu_t = try timed(io, n_elems, gpu);
        // ms_per_dispatch = total_ms / n_dispatches.
        const ms_per_dispatch = gpu_t.ms / @as(f64, @floatFromInt(n_dispatches));
        // "G/s" here is per-dispatch element throughput.
        const gops = @as(f64, @floatFromInt(n_elems)) / (ms_per_dispatch * 1e6);
        var label_buf: [40]u8 = undefined;
        const label = std.fmt.bufPrint(&label_buf, "n_dispatches={d} (n={d})", .{ n_dispatches, n_elems }) catch unreachable;
        const tol = z.ops_vulkan.ResidualAdd.requires.tolerance;
        rawRow(label, "vk_resid_x", ms_per_dispatch, gops, gpu_errs.rel, gpu_hash, opStatus(tol, gpu_errs.rel, gpu_hash == ref_hash));
    }
    flush();
}

/// One row of the kit grid: a head geometry, the KV type of its registry
/// default, and the deepest row its declared context holds.
const KitShape = struct {
    label: []const u8,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_f16: bool,
    max_depth: u32,
};

/// The attention kit of the build over a grid of (shape, depth, rows). Each
/// point records `rows` consecutive query rows of one stream that end at
/// `depth`, through `attn_record` as a runner records them, and times one
/// submission. The G/s column is the unique KV bytes of the stream, K and V,
/// over the time. The rel_err column holds the error of the deepest row
/// against the CPU reference. So a kernel experiment runs here without a
/// model.
pub fn benchAttnKitGrid(gpa: std.mem.Allocator, io: std.Io, runner: *z.vulkan_runner.Runner) !void {
    const vkops = z.ops_vulkan;
    const OneShot = vkops.OneShot;
    const HostBuffer = vkops.HostBuffer;
    // The bench imports no Vulkan binding, so it reads the descriptor type
    // from the binding helper.
    const DescriptorBufferInfo = @typeInfo(@typeInfo(@TypeOf(vkops.updateStorageBindings)).@"fn".params[2].type.?).pointer.child;
    const ctx = &runner.ctx;
    printSectionHeader("fused attention kit grid (-Dattn-kernel=" ++ @tagName(build_options.attn_kernel) ++ ")");

    const shapes = [_]KitShape{
        .{ .label = "qwen35-2b 8h,2kv,d256", .n_heads = 8, .n_kv_heads = 2, .head_dim = 256, .kv_f16 = true, .max_depth = 262_144 },
        .{ .label = "llama-8b 32h,8kv,d128", .n_heads = 32, .n_kv_heads = 8, .head_dim = 128, .kv_f16 = true, .max_depth = 131_072 },
        .{ .label = "qwen35-9b 16h,4kv,d256", .n_heads = 16, .n_kv_heads = 4, .head_dim = 256, .kv_f16 = true, .max_depth = 262_144 },
        .{ .label = "smollm2 9h,3kv,d64", .n_heads = 9, .n_kv_heads = 3, .head_dim = 64, .kv_f16 = false, .max_depth = 8192 },
    };
    const depths = [_]u32{ 2048, 8192, 32_704 };
    // One decode row, one row tile of eight rows, and one prefill chunk.
    const row_counts = [_]u32{ 1, 8, 128 };

    for (shapes) |sh| {
        const kv: z.dtype.KvDtype = if (sh.kv_f16) .f16 else .f32;
        var single = try vkops.Attention.initKv(ctx, kv);
        defer single.deinit();
        var multi = try vkops.AttentionMultiStream.initKv(ctx, kv);
        defer multi.deinit();
        var passes = try vkops.AttentionSplit.initKv(ctx, kv);
        defer passes.deinit();
        const single_set = try OneShot.init(ctx, single.pipeline.set_layout, 4);
        defer single_set.deinit(ctx);
        const multi_set = try OneShot.init(ctx, multi.pipeline.set_layout, 4);
        defer multi_set.deinit(ctx);
        const split_set = try OneShot.init(ctx, passes.split.set_layout, vkops.AttentionSplit.split_bindings);
        defer split_set.deinit(ctx);
        const merge_set = try OneShot.init(ctx, passes.merge.set_layout, vkops.AttentionSplit.merge_bindings);
        defer merge_set.deinit(ctx);
        var buf_q: HostBuffer = .{ .ctx = ctx };
        defer buf_q.deinit();
        var buf_k: HostBuffer = .{ .ctx = ctx };
        defer buf_k.deinit();
        var buf_v: HostBuffer = .{ .ctx = ctx };
        defer buf_v.deinit();
        var buf_out: HostBuffer = .{ .ctx = ctx };
        defer buf_out.deinit();
        // The runners keep q, KV, and the outputs in device-local memory, so
        // the timed kernels read the same memory type.
        var dev_q: vkops.DeviceBuffer = .{ .ctx = ctx };
        defer dev_q.deinit();
        var dev_k: vkops.DeviceBuffer = .{ .ctx = ctx };
        defer dev_k.deinit();
        var dev_v: vkops.DeviceBuffer = .{ .ctx = ctx };
        defer dev_v.deinit();
        var dev_scratch: vkops.DeviceBuffer = .{ .ctx = ctx };
        defer dev_scratch.deinit();
        var dev_out: vkops.DeviceBuffer = .{ .ctx = ctx };
        defer dev_out.deinit();

        const row_elems: usize = @as(usize, sh.n_heads) * sh.head_dim;
        const kv_elems: usize = @as(usize, sh.n_kv_heads) * sh.head_dim;
        const elem_bytes: usize = if (sh.kv_f16) 2 else 4;
        for (depths) |depth| {
            if (depth > sh.max_depth) continue;
            const stream_elems = @as(usize, depth) * kv_elems;
            // The f16 shaders read the caches as 32-bit words, so each binding
            // covers whole words.
            const kv_bytes = std.mem.alignForward(usize, stream_elems * elem_bytes, 4);
            if (kv_bytes > ctx.buffer_limits.storage_range) {
                h.skipRow(sh.label, "kit", "range");
                continue;
            }
            const k_host = try gpa.alloc(f32, stream_elems);
            defer gpa.free(k_host);
            const v_host = try gpa.alloc(f32, stream_elems);
            defer gpa.free(v_host);
            for (k_host, 0..) |*x, i| x.* = @cos(@as(f32, @floatFromInt(i % 65_521)) * 0.03);
            for (v_host, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i % 65_519)) * 0.05);
            try buf_k.ensure(kv_bytes);
            try buf_v.ensure(kv_bytes);
            if (sh.kv_f16) {
                for (buf_k.sliceTyped(f16, stream_elems), k_host) |*d, x| d.* = @floatCast(x);
                for (buf_v.sliceTyped(f16, stream_elems), v_host) |*d, x| d.* = @floatCast(x);
                // The reference reads the values the cache stores.
                for (k_host) |*x| x.* = @as(f16, @floatCast(x.*));
                for (v_host) |*x| x.* = @as(f16, @floatCast(x.*));
            } else {
                @memcpy(buf_k.sliceTyped(f32, stream_elems), k_host);
                @memcpy(buf_v.sliceTyped(f32, stream_elems), v_host);
            }
            try dev_k.ensure(kv_bytes);
            try dev_v.ensure(kv_bytes);
            try vkops.runStagingCopies(ctx, &.{
                .{ .src = buf_k.buffer, .dst = dev_k.buffer, .size = kv_bytes },
                .{ .src = buf_v.buffer, .dst = dev_v.buffer, .size = kv_bytes },
            });
            const scratch = z.attn_split.Scratch.init(sh.n_heads, sh.head_dim, depth);
            try dev_scratch.ensure(scratch.totalBytes());
            // The reporter keeps the label slices until the flush below.
            var label_bufs: [row_counts.len][48]u8 = undefined;

            for (row_counts, 0..) |rows, ri| {
                const q_host = try gpa.alloc(f32, rows * row_elems);
                defer gpa.free(q_host);
                for (q_host, 0..) |*x, i| x.* = @sin(@as(f32, @floatFromInt(i)) * 0.07);
                const q_bytes = q_host.len * @sizeOf(f32);
                try buf_q.ensure(q_bytes);
                @memcpy(buf_q.sliceTyped(f32, q_host.len), q_host);
                try dev_q.ensure(q_bytes);
                try vkops.runStagingCopies(ctx, &.{.{ .src = buf_q.buffer, .dst = dev_q.buffer, .size = q_bytes }});
                const out_bytes = q_host.len * @sizeOf(f32);
                try buf_out.ensure(out_bytes);
                try dev_out.ensure(out_bytes);

                const q_info: DescriptorBufferInfo = .{ .buffer = dev_q.buffer, .offset = 0, .range = q_bytes };
                const k_info: DescriptorBufferInfo = .{ .buffer = dev_k.buffer, .offset = 0, .range = kv_bytes };
                const v_info: DescriptorBufferInfo = .{ .buffer = dev_v.buffer, .offset = 0, .range = kv_bytes };
                const scratch_info: DescriptorBufferInfo = .{ .buffer = dev_scratch.buffer, .offset = 0, .range = scratch.totalBytes() };
                const out_info: DescriptorBufferInfo = .{ .buffer = dev_out.buffer, .offset = 0, .range = out_bytes };
                vkops.updateStorageBindings(ctx, single_set.desc_set, &.{ q_info, k_info, v_info, out_info });
                vkops.updateStorageBindings(ctx, multi_set.desc_set, &.{ q_info, k_info, v_info, out_info });
                vkops.AttentionSplit.assertKBinding(k_info);
                vkops.AttentionSplit.assertVBinding(v_info);
                vkops.updateStorageBindings(ctx, split_set.desc_set, &.{ q_info, k_info, v_info, scratch_info });
                vkops.updateStorageBindings(ctx, merge_set.desc_set, &.{ scratch_info, out_info });

                // Row i attends to the positions `[0, depth - rows + 1 + i)`,
                // so the last row reads the whole stream.
                var sites: [z.attn_split.max_plan_rows]z.attn_split.RowSite = undefined;
                for (sites[0..rows], 0..) |*site, i| site.* = .{ .slot = 0, .t_len = depth - rows + 1 + @as(u32, @intCast(i)) };
                const plan = z.attn_split.planRows(sites[0..rows], scratch.max_splits);
                const binding: vkops.AttnKit.Binding = .{
                    .shape = .{
                        .n_heads = sh.n_heads,
                        .n_kv_heads = sh.n_kv_heads,
                        .head_dim = sh.head_dim,
                        .stream_stride_elems = @intCast(stream_elems),
                        .max_splits = scratch.max_splits,
                    },
                    // A decode step binds the single-row set, and a batched
                    // step the multi-stream set.
                    .resident = if (rows == 1) .{ .single = single_set.desc_set } else .{ .multi = multi_set.desc_set },
                    .split = split_set.desc_set,
                    .merge = merge_set.desc_set,
                };
                const Step = struct {
                    ctx: @TypeOf(ctx),
                    kit: vkops.AttnKit,
                    cb: vkops.AttnKit.Cmd,
                    plan: *const z.attn_split.AttnPlan,
                    binding: vkops.AttnKit.Binding,
                    pub fn run(self: @This()) !void {
                        try self.ctx.vkd.resetCommandBuffer(self.cb, .{});
                        try self.ctx.vkd.beginCommandBuffer(self.cb, &.{ .flags = .{ .one_time_submit_bit = true } });
                        z.attn_record.record(vkops.AttnKit, self.kit, self.cb, self.plan, self.binding);
                        try self.ctx.vkd.endCommandBuffer(self.cb);
                        const cmds = [_]vkops.AttnKit.Cmd{self.cb};
                        try self.ctx.vkd.queueSubmit(self.ctx.queue, &.{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }}, .null_handle);
                        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
                    }
                };
                const step: Step = .{
                    .ctx = ctx,
                    .kit = .{ .single = &single, .multi = &multi, .passes = &passes },
                    .cb = single_set.cmd_buffer,
                    .plan = &plan,
                    .binding = binding,
                };
                try step.run();
                try vkops.runStagingCopies(ctx, &.{.{ .src = dev_out.buffer, .dst = buf_out.buffer, .size = out_bytes }});
                const out = buf_out.sliceTyped(f32, q_host.len);
                const hsh = hashOut(out);

                // The deepest row against the CPU reference.
                const last_q = q_host[(rows - 1) * row_elems ..][0..row_elems];
                const ref = try gpa.alloc(f32, row_elems);
                defer gpa.free(ref);
                const parts = try gpa.alloc(f32, z.attn_cpu.partialLen(sh.n_heads, sh.head_dim));
                defer gpa.free(parts);
                z.ops.attentionRowTyped(f32, last_q, sh.n_heads, sh.head_dim, k_host, v_host, sh.n_kv_heads, depth, parts, ref);
                const errs = computeErrors(ref, out[(rows - 1) * row_elems ..][0..row_elems]);

                const t = try timed(io, rows * depth * row_elems, step);
                const unique_kv_bytes: f64 = @floatFromInt(2 * stream_elems * elem_bytes);
                const gbs = unique_kv_bytes / (t.ms * 1e6);
                const label = std.fmt.bufPrint(&label_bufs[ri], "{s} t{d} r{d}", .{ sh.label, depth, rows }) catch unreachable;
                const variant: []const u8 = if (sh.kv_f16) "kit_f16" else "kit_f32";
                const tol = vkops.AttnKit.tolerance;
                rawRow(label, variant, t.ms, gbs, errs.rel, hsh, opStatus(tol, errs.rel, false));
            }
            flush();
        }
    }
}
