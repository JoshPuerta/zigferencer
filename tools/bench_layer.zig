//! Per-layer op benchmarks: the elementwise ops (residual_add, swiglu, rmsnorm, rope), fused
//! single-head attention, the dispatch-amortization demo, and the Qwen full-attention sigmoid
//! gate. These ops sit between the matmuls on the forward pass, so a regression here hides
//! behind the matmul rows unless it gets its own table. Each section runs the CPU reference on
//! every build and the GPU variant under vulkan.
//!
//! Comptime. `build_options.use_vulkan` gates the GPU rows, so a CPU build never reads them. The
//! element type and kernel set are parameters.
//!
//! Attention grid. The attention section also walks a context-depth grid on the two qwen
//! full-attention shapes, the last ten cases. Qwen3.5-2B states 8 query heads over 2 kv heads,
//! and Qwen3.5-9B states 16 over 4, both at head_dim 256. Attention is the one per-token cost
//! that grows with the context, so the grid measures the CPU against the GPU at depth. The f16
//! rows read the KV cache in the dtype that the qwen registry entries default to. The f16 caches
//! hold the same values at storage precision, so the gap to the f32 row is the element type
//! alone.
//!
//! Threaded rows. Every attention case also runs a 16-thread row through the attention control
//! unit. That row must give the bits of the serial row, because the part geometry does not
//! depend on the thread count.
//!
//! Vector rows. The `simd` rows run the vector part kernel of the `simd` backend, and the
//! `simd_pf` rows run the prefetching kernel of the `simd_pf` backend. Their rel_err column
//! holds the error against the naive row of the same dtype. Every threaded vector row must give
//! the bits of the serial `simd` row. The prefetch changes the timing alone.
//!
//! f16 hash. The f16 naive row carries its own hash, because a narrower store moves the output
//! bits. A gate against the f32 hash claims an equality that the dtype does not promise.
//!
//! GPU attention rows. They stay f32, because the per-op runner builds `Attention` for f32
//! caches and the bench calls `Attention.run`. `Attention.runTyped` takes f16 caches on an f16
//! pipeline, and the kernel tests and `gpu-selftest attn` use it.
//!
//! Kernel set grid. `benchAttnGrid` records `rows` consecutive query rows of one stream that end
//! at `depth`, through `attn_record` as a runner records them, and times one submission. The G/s
//! column is the unique KV bytes of the stream, K and V, over the time. The rel_err column holds
//! the error of the deepest row against the CPU reference. So a kernel experiment runs here
//! without a model.
//!
//! Dispatch amortization. The section shows how far the per-dispatch wall clock drops when N
//! dispatches share one command buffer and one submit. It runs `ResidualAdd.runRepeat(N)` for a
//! few N on one shape and prints the ms per dispatch. The N=1 row is the per-call overhead of
//! the `run()` path. A later row shows how much of that overhead many ops amortize between them.

const std = @import("std");
const build_options = @import("build_options");
const zigferencer = @import("zigferencer");
const op_requires = zigferencer.op_requires;
const vkops = zigferencer.ops_vulkan;
const Runner = zigferencer.vulkan_runner.Runner;

const bench_util = @import("bench_util.zig");
const cpuRow = bench_util.cpuRow;
const gpuRow = bench_util.gpuRow;
const rawRow = bench_util.rawRow;
const flush = bench_util.flush;
const printSectionHeader = bench_util.printSectionHeader;
const hashOut = bench_util.hashOut;
const computeErrors = bench_util.computeErrors;
const opStatus = bench_util.opStatus;
const timed = bench_util.timed;
const rate = bench_util.rate;

// -- constants --------------------------------------------------------------

/// The fan-out of every threaded attention row. The count is fixed, so the rows of two hosts
/// compare.
const bench_threaded: zigferencer.parallel.Policy = .{ .n_threads = 16, .min_parallel_flops = 0 };

const atn_cases = [_]AtnCase{
    atnCase("attn 9h,3kv,d64,t128", 9, 3, 64, 128, false),
    atnCase("attn 32h,8kv,d64,t512", 32, 8, 64, 512, false),
    atnCase("attn 32h,8kv,d128,t512", 32, 8, 128, 512, false),
    // This shape is the widest GQA group the kernel blocks for. No loaded
    // model states it, so the row measures the block bound alone.
    atnCase("attn 16h,2kv,d256,t512 group 8", 16, 2, 256, 512, false),
    atnCase("attn 8h,2kv,d256,t512", 8, 2, 256, 512, true),
    atnCase("attn 8h,2kv,d256,t1024", 8, 2, 256, 1024, true),
    atnCase("attn 8h,2kv,d256,t2048", 8, 2, 256, 2048, true),
    atnCase("attn 8h,2kv,d256,t4096", 8, 2, 256, 4096, true),
    atnCase("attn 8h,2kv,d256,t8192", 8, 2, 256, 8192, true),
    atnCase("attn 16h,4kv,d256,t512", 16, 4, 256, 512, true),
    atnCase("attn 16h,4kv,d256,t1024", 16, 4, 256, 1024, true),
    atnCase("attn 16h,4kv,d256,t2048", 16, 4, 256, 2048, true),
    atnCase("attn 16h,4kv,d256,t4096", 16, 4, 256, 4096, true),
    atnCase("attn 16h,4kv,d256,t8192", 16, 4, 256, 8192, true),
};

const grid_shapes = [_]GridShape{
    gridShape("qwen35-2b 8h,2kv,d256", 8, 2, 256, true, 262_144),
    gridShape("llama-8b 32h,8kv,d128", 32, 8, 128, true, 131_072),
    gridShape("qwen35-9b 16h,4kv,d256", 16, 4, 256, true, 262_144),
    gridShape("smollm2 9h,3kv,d64", 9, 3, 64, false, 8192),
};

const grid_depths = [_]u32{ 2048, 8192, 32_704 };

// One decode row, one row tile of eight rows, and one prefill chunk.
const grid_row_counts = [_]u32{ 1, 8, 128 };

// -- types ------------------------------------------------------------------

/// One CPU attention closure per KV storage type `E` and kernel set `Kernels`. One generic body
/// keeps the f32 and f16 rows from drifting. Without `pool` it runs the serial row, and with
/// `pool` the attention control unit at `bench_threaded`.
fn AtnCpu(comptime E: type, comptime Kernels: type) type {
    return struct {
        pool: ?*zigferencer.pool.Pool = null,
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
                zigferencer.parallel.attentionTyped(
                    E,
                    Kernels,
                    pool,
                    bench_threaded,
                    self.q,
                    self.n_heads,
                    self.head_dim,
                    self.k_cache,
                    self.v_cache,
                    self.n_kv_heads,
                    self.t_len,
                    self.parts,
                    self.out,
                );
            } else {
                Kernels.attentionRowTyped(
                    E,
                    self.q,
                    self.n_heads,
                    self.head_dim,
                    self.k_cache,
                    self.v_cache,
                    self.n_kv_heads,
                    self.t_len,
                    self.parts,
                    self.out,
                );
            }
        }

        /// The same inputs and output over the kernel set `K`, serial when
        /// `pool` is null.
        pub fn over(self: @This(), comptime K: type, pool: ?*zigferencer.pool.Pool) AtnCpu(E, K) {
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

/// One row of the kernel set grid: a head geometry, the KV type of its registry
/// default, and the deepest row its declared context holds.
const GridShape = struct {
    label: []const u8,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_f16: bool,
    max_depth: u32,
};

/// One CPU attention case: a head geometry, a context depth, and the KV storage type.
const AtnCase = struct {
    label: []const u8,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    t_len: u32,
    kv_f16: bool,
};

/// The input slices that the f32 and the f16 rows of one attention case share.
const AtnData = struct {
    q: []const f32,
    k_cache: []const f32,
    v_cache: []const f32,
    out_ref: []const f32,
    out_work: []f32,
};

/// The f16 copies of the KV cache, with their scratch and the f16 reference output.
const AtnF16 = struct {
    k: []const f16,
    v: []const f16,
    parts: []f32,
    out_ref: []f32,
};

const GridCtx = @TypeOf(@as(Runner, undefined).ctx);

// The bench imports no Vulkan binding, so it reads the descriptor type from the binding helper.
const DescriptorBufferInfo = @typeInfo(
    @typeInfo(@TypeOf(vkops.updateStorageBindings)).@"fn".params[2].type.?,
).pointer.child;

/// The kernels, descriptor sets, and buffers that every row of one shape shares.
const GridRig = struct {
    ctx: *GridCtx,
    single: *vkops.Attention,
    multi: *vkops.AttentionMultiStream,
    passes: *vkops.AttentionSplit,
    single_set: vkops.OneShot,
    multi_set: vkops.OneShot,
    split_set: vkops.OneShot,
    merge_set: vkops.OneShot,
    buf_q: *vkops.HostBuffer,
    buf_k: *vkops.HostBuffer,
    buf_v: *vkops.HostBuffer,
    buf_out: *vkops.HostBuffer,
    dev_q: *vkops.DeviceBuffer,
    dev_k: *vkops.DeviceBuffer,
    dev_v: *vkops.DeviceBuffer,
    dev_scratch: *vkops.DeviceBuffer,
    dev_out: *vkops.DeviceBuffer,
};

/// One KV stream at one depth: the host values, their byte size, and the split scratch.
const GridStream = struct {
    depth: u32,
    stream_elems: usize,
    kv_bytes: usize,
    elem_bytes: usize,
    k_host: []f32,
    v_host: []f32,
    scratch: zigferencer.attn_split.Scratch,
};

/// One submission of the recorded kernel set: the timed closure of a grid point.
const GridStep = struct {
    ctx: *GridCtx,
    kernels: vkops.AttnKernels,
    cb: vkops.AttnKernels.Cmd,
    plan: *const zigferencer.attn_split.AttnPlan,
    binding: vkops.AttnKernels.Binding,

    pub fn run(self: @This()) !void {
        try self.ctx.vkd.resetCommandBuffer(self.cb, .{});
        try self.ctx.vkd.beginCommandBuffer(
            self.cb,
            &.{ .flags = .{ .one_time_submit_bit = true } },
        );
        zigferencer.attn_record.record(
            vkops.AttnKernels,
            self.kernels,
            self.cb,
            self.plan,
            self.binding,
        );
        try self.ctx.vkd.endCommandBuffer(self.cb);
        const cmds = [_]vkops.AttnKernels.Cmd{self.cb};
        try self.ctx.vkd.queueSubmit(
            self.ctx.queue,
            &.{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }},
            .null_handle,
        );
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
    }
};

// -- api --------------------------------------------------------------------

/// The elementwise sections and the attention sections, in print order. Each section prints a
/// header, then one row per variant for each case.
pub fn benchElementwise(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    try benchResidualAdd(gpa, io, runner);
    try benchSwiglu(gpa, io, runner);
    try benchRmsnorm(gpa, io, runner);
    try benchRope(gpa, io, runner);
    try benchFusedAttention(gpa, io, runner);
    if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            try benchAttnGrid(gpa, io, r);
            try benchDispatchAmortization(gpa, io, r);
        }
    }
}

/// Qwen3.5 full-attention output gate: `dst *= sigmoid(gate)`. GPU-only kernel
/// with an inline CPU reference (there is no standalone CPU sigmoid-gate op).
pub fn benchSigmoidGate(gpa: std.mem.Allocator, io: std.Io, runner: *Runner) !void {
    printSectionHeader("sigmoid_gate (qwen full-attn output gate)");
    var kern = try zigferencer.ops_vulkan.SigmoidGate.init(&runner.ctx);
    defer kern.deinit();

    const cases = [_]struct { label: []const u8, n: usize }{
        .{ .label = "sigmoid n=2048 (qwen attn out)", .n = 2048 },
        .{ .label = "sigmoid n=4096", .n = 4096 },
    };
    for (cases) |c| {
        // Allocate and fill the inputs.
        const gate = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(gate);

        const dst_init = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_init);

        const dst_ref = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_ref);

        const dst_work = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_work);

        for (gate, 0..) |*v, i| {
            v.* = @sin(@as(f32, @floatFromInt(i)) * 0.01);
        }

        for (dst_init, 0..) |*v, i| {
            v.* = @cos(@as(f32, @floatFromInt(i)) * 0.01);
        }

        // Compute the reference.
        @memcpy(dst_ref, dst_init);
        for (dst_ref, gate) |*d, g| {
            d.* *= 1.0 / (1.0 + @exp(-g));
        }

        const ref_hash = hashOut(dst_ref);
        const work: f64 = @floatFromInt(c.n);

        // Time the GPU row.
        const Gpu = struct {
            k: *zigferencer.ops_vulkan.SigmoidGate,
            gate: []const f32,
            dst_init: []const f32,
            dst_work: []f32,

            pub fn run(self: @This()) !void {
                @memcpy(self.dst_work, self.dst_init);
                try self.k.run(self.dst_work, self.gate);
            }
        };
        const tol: op_requires.Tolerance = .{ .relative = 1e-5 };
        try gpuRow(
            io,
            c.n,
            c.label,
            "vk_sigmoid",
            work,
            tol,
            dst_ref,
            ref_hash,
            dst_work,
            Gpu{ .k = &kern, .gate = gate, .dst_init = dst_init, .dst_work = dst_work },
        );
        flush();
    }
}

/// The attention kernel set of the build over a grid of (shape, depth, rows), one timed submission
/// per point.
pub fn benchAttnGrid(gpa: std.mem.Allocator, io: std.Io, runner: *Runner) !void {
    printSectionHeader(
        "fused attention kernel set grid (-Dattn-kernel=" ++ @tagName(
            build_options.attn_kernel,
        ) ++ ")",
    );

    for (grid_shapes) |sh| {
        try benchGridShape(gpa, io, &runner.ctx, sh);
    }
}

// -- helpers ----------------------------------------------------------------

fn gridShape(
    label: []const u8,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_f16: bool,
    max_depth: u32,
) GridShape {
    return .{
        .label = label,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .kv_f16 = kv_f16,
        .max_depth = max_depth,
    };
}

fn atnCase(
    label: []const u8,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    t_len: u32,
    kv_f16: bool,
) AtnCase {
    return .{
        .label = label,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .t_len = t_len,
        .kv_f16 = kv_f16,
    };
}

fn benchResidualAdd(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
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
        // Allocate and fill the inputs.
        const x = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(x);

        const dst_init = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_init);

        const dst_ref = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_ref);

        const dst_work = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_work);

        for (x, 0..) |*v, i| {
            v.* = @sin(@as(f32, @floatFromInt(i)));
        }

        for (dst_init, 0..) |*v, i| {
            v.* = @cos(@as(f32, @floatFromInt(i)));
        }

        // Compute the reference.
        @memcpy(dst_ref, dst_init);
        zigferencer.ops.residualAdd(x, dst_ref);
        const ref_hash = hashOut(dst_ref);
        const work: f64 = @floatFromInt(c.n);

        // Time the CPU row.
        const Cpu = struct {
            x: []const f32,
            dst_init: []const f32,
            dst_work: []f32,

            pub fn run(self: @This()) !void {
                @memcpy(self.dst_work, self.dst_init);
                zigferencer.ops.residualAdd(self.x, self.dst_work);
            }
        };
        try cpuRow(
            io,
            c.n,
            c.label,
            "naive",
            work,
            ref_hash,
            Cpu{ .x = x, .dst_init = dst_init, .dst_work = dst_work },
        );

        // Time the GPU row.
        if (comptime build_options.use_vulkan) {
            if (runner) |r| {
                const Gpu = struct {
                    r: *Runner,
                    x: []const f32,
                    dst_init: []const f32,
                    dst_work: []f32,

                    pub fn run(self: @This()) !void {
                        @memcpy(self.dst_work, self.dst_init);
                        try self.r.residual_add.run(self.x, self.dst_work);
                    }
                };
                try gpuRow(
                    io,
                    c.n,
                    c.label,
                    "vk_resid",
                    work,
                    zigferencer.ops_vulkan.ResidualAdd.requires.tolerance,
                    dst_ref,
                    ref_hash,
                    dst_work,
                    Gpu{ .r = r, .x = x, .dst_init = dst_init, .dst_work = dst_work },
                );
            }
        }

        flush();
    }
}

fn benchSwiglu(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("swiglu");
    // FFN widths: SmolLM2 (1536), Llama-1B (8192), Llama-8B (14336).
    const swiglu_cases = [_]struct { label: []const u8, n: usize }{
        .{ .label = "swiglu n=1536  (ffn)", .n = 1536 },
        .{ .label = "swiglu n=8192  (ffn)", .n = 8192 },
        .{ .label = "swiglu n=14336 (ffn)", .n = 14336 },
    };
    for (swiglu_cases) |c| {
        // Allocate and fill the inputs.
        const gate = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(gate);

        const up = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(up);

        const dst_ref = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_ref);

        const dst_work = try gpa.alignedAlloc(f32, .of(f32), c.n);
        defer gpa.free(dst_work);

        for (gate, 0..) |*v, i| {
            v.* = @sin(@as(f32, @floatFromInt(i)));
        }

        for (up, 0..) |*v, i| {
            v.* = @cos(@as(f32, @floatFromInt(i)));
        }

        // Compute the reference.
        zigferencer.ops.swiglu(gate, up, dst_ref);
        const ref_hash = hashOut(dst_ref);
        const work: f64 = @floatFromInt(c.n);

        // Time the CPU row.
        const Cpu = struct {
            gate: []const f32,
            up: []const f32,
            dst: []f32,

            pub fn run(self: @This()) !void {
                zigferencer.ops.swiglu(self.gate, self.up, self.dst);
            }
        };
        try cpuRow(
            io,
            c.n,
            c.label,
            "naive",
            work,
            ref_hash,
            Cpu{ .gate = gate, .up = up, .dst = dst_work },
        );

        // Time the GPU row.
        if (comptime build_options.use_vulkan) {
            if (runner) |r| {
                const Gpu = struct {
                    r: *Runner,
                    gate: []const f32,
                    up: []const f32,
                    dst: []f32,

                    pub fn run(self: @This()) !void {
                        try self.r.swiglu.run(self.gate, self.up, self.dst);
                    }
                };
                try gpuRow(
                    io,
                    c.n,
                    c.label,
                    "vk_swiglu",
                    work,
                    zigferencer.ops_vulkan.Swiglu.requires.tolerance,
                    dst_ref,
                    ref_hash,
                    dst_work,
                    Gpu{ .r = r, .gate = gate, .up = up, .dst = dst_work },
                );
            }
        }

        flush();
    }
}

fn benchRmsnorm(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("rmsnorm");
    const rms_cases = [_]struct { label: []const u8, d: usize }{
        .{ .label = "rmsnorm d=576   (hidden)", .d = 576 },
        .{ .label = "rmsnorm d=2048  (hidden)", .d = 2048 },
        .{ .label = "rmsnorm d=4096  (hidden)", .d = 4096 },
    };
    const eps: f32 = 1e-5;
    for (rms_cases) |c| {
        // Allocate and fill the inputs.
        const x = try gpa.alignedAlloc(f32, .of(f32), c.d);
        defer gpa.free(x);

        const w = try gpa.alignedAlloc(f32, .of(f32), c.d);
        defer gpa.free(w);

        const y_ref = try gpa.alignedAlloc(f32, .of(f32), c.d);
        defer gpa.free(y_ref);

        const y_work = try gpa.alignedAlloc(f32, .of(f32), c.d);
        defer gpa.free(y_work);

        for (x, 0..) |*v, i| {
            v.* = @sin(@as(f32, @floatFromInt(i)));
        }

        for (w, 0..) |*v, i| {
            v.* = @cos(@as(f32, @floatFromInt(i))) * 0.5 + 1.0;
        }

        // Compute the reference.
        zigferencer.ops.rmsnorm(x, w, eps, y_ref);
        const ref_hash = hashOut(y_ref);
        // 3 FLOPs per element: the square, the add via FMA, and the scale multiply.
        const work: f64 = @as(f64, @floatFromInt(c.d)) * 3.0;

        // Time the CPU row.
        const Cpu = struct {
            x: []const f32,
            w: []const f32,
            eps: f32,
            y: []f32,

            pub fn run(self: @This()) !void {
                zigferencer.ops.rmsnorm(self.x, self.w, self.eps, self.y);
            }
        };
        try cpuRow(
            io,
            c.d,
            c.label,
            "naive",
            work,
            ref_hash,
            Cpu{ .x = x, .w = w, .eps = eps, .y = y_work },
        );

        // Time the GPU row.
        if (comptime build_options.use_vulkan) {
            if (runner) |r| {
                const Gpu = struct {
                    r: *Runner,
                    x: []const f32,
                    w: []const f32,
                    eps: f32,
                    y: []f32,

                    pub fn run(self: @This()) !void {
                        try self.r.rmsnorm.run(self.x, self.w, self.eps, self.y);
                    }
                };
                try gpuRow(
                    io,
                    c.d,
                    c.label,
                    "vk_rmsnorm",
                    work,
                    zigferencer.ops_vulkan.Rmsnorm.requires.tolerance,
                    y_ref,
                    ref_hash,
                    y_work,
                    Gpu{ .r = r, .x = x, .w = w, .eps = eps, .y = y_work },
                );
            }
        }

        flush();
    }
}

fn benchRope(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("rope");
    const rope_cases = [_]struct { label: []const u8, head_dim: usize }{
        .{ .label = "rope head_dim=64", .head_dim = 64 },
        .{ .label = "rope head_dim=128", .head_dim = 128 },
        .{ .label = "rope head_dim=256", .head_dim = 256 }, // Qwen3.5 full-attn
    };
    const pos: u32 = 17;
    const freq_base: f32 = 10000.0;
    for (rope_cases) |c| {
        // Allocate and fill the inputs.
        const x_init = try gpa.alignedAlloc(f32, .of(f32), c.head_dim);
        defer gpa.free(x_init);

        const x_ref = try gpa.alignedAlloc(f32, .of(f32), c.head_dim);
        defer gpa.free(x_ref);

        const x_work = try gpa.alignedAlloc(f32, .of(f32), c.head_dim);
        defer gpa.free(x_work);

        for (x_init, 0..) |*v, i| {
            v.* = @sin(@as(f32, @floatFromInt(i)));
        }

        // Compute the reference.
        @memcpy(x_ref, x_init);
        zigferencer.ops.ropeInplace(x_ref, pos, c.head_dim, freq_base, .neox, null);
        const ref_hash = hashOut(x_ref);
        const work: f64 = @floatFromInt(c.head_dim);

        // Time the CPU row.
        const Cpu = struct {
            x_init: []const f32,
            x_work: []f32,
            pos: u32,
            head_dim: usize,
            freq_base: f32,

            pub fn run(self: @This()) !void {
                @memcpy(self.x_work, self.x_init);
                zigferencer.ops.ropeInplace(
                    self.x_work,
                    self.pos,
                    self.head_dim,
                    self.freq_base,
                    .neox,
                    null,
                );
            }
        };
        try cpuRow(io, c.head_dim, c.label, "naive", work, ref_hash, Cpu{
            .x_init = x_init,
            .x_work = x_work,
            .pos = pos,
            .head_dim = c.head_dim,
            .freq_base = freq_base,
        });

        // Time the GPU row.
        if (comptime build_options.use_vulkan) {
            if (runner) |r| {
                const Gpu = struct {
                    r: *Runner,
                    x_init: []const f32,
                    x_work: []f32,
                    pos: u32,
                    head_dim: usize,
                    freq_base: f32,

                    pub fn run(self: @This()) !void {
                        @memcpy(self.x_work, self.x_init);
                        try self.r.rope.run(
                            self.x_work,
                            self.pos,
                            self.head_dim,
                            self.freq_base,
                            .neox,
                            null,
                        );
                    }
                };
                try gpuRow(
                    io,
                    c.head_dim,
                    c.label,
                    "vk_rope",
                    work,
                    zigferencer.ops_vulkan.Rope.requires.tolerance,
                    x_ref,
                    ref_hash,
                    x_work,
                    Gpu{
                        .r = r,
                        .x_init = x_init,
                        .x_work = x_work,
                        .pos = pos,
                        .head_dim = c.head_dim,
                        .freq_base = freq_base,
                    },
                );
            }
        }

        flush();
    }
}

fn benchFusedAttention(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("fused attention");
    // One pool serves every threaded row of the section.
    var pool: zigferencer.pool.Pool = undefined;
    try zigferencer.parallel.initPool(&pool, io, bench_threaded);
    defer pool.deinit();

    for (atn_cases) |c| {
        try attentionCase(gpa, io, runner, &pool, c);
    }
}

/// The f16 rows of one attention case. They carry their own reference hash.
fn attentionF16Rows(
    io: std.Io,
    pool: *zigferencer.pool.Pool,
    c: AtnCase,
    d: AtnData,
    k16: AtnF16,
    total_work_elems: usize,
    work: f64,
) !void {
    const cpu16: AtnCpu(f16, zigferencer.ops) = .{
        .q = d.q,
        .n_heads = c.n_heads,
        .head_dim = c.head_dim,
        .k_cache = k16.k,
        .v_cache = k16.v,
        .n_kv_heads = c.n_kv_heads,
        .t_len = c.t_len,
        .parts = k16.parts,
        .out = d.out_work,
    };
    // Hash the f16 output on its own, since an f16 store moves the bits.
    try cpu16.run();
    @memcpy(k16.out_ref, d.out_work);
    const hash16 = hashOut(k16.out_ref);
    try cpuRow(io, total_work_elems, c.label, "naive_f16", work, hash16, cpu16);
    var cpu16_mt = cpu16;
    cpu16_mt.pool = pool;
    _ = try cpuCheckedRow(
        io,
        total_work_elems,
        c.label,
        "naive_f16_mt",
        work,
        k16.out_ref,
        hash16,
        cpu16_mt,
    );
    try simdRows(f16, io, pool, total_work_elems, c.label, "_f16", work, k16.out_ref, cpu16);
}

/// The GPU row of one attention case, f32 caches only.
fn attentionGpuRow(
    io: std.Io,
    r: *Runner,
    c: AtnCase,
    d: AtnData,
    total_work_elems: usize,
    work: f64,
    ref_hash: u64,
) !void {
    // Wrap the GPU call as a bench case.
    const Gpu = struct {
        r: *Runner,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        t_len: u32,
        out: []f32,

        pub fn run(self: @This()) !void {
            try self.r.attention.run(
                self.q,
                self.k_cache,
                self.v_cache,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.t_len,
                self.out,
            );
        }
    };

    // Run the GPU row against the reference.
    try gpuRow(
        io,
        total_work_elems,
        c.label,
        "vk_atn",
        work,
        zigferencer.ops_vulkan.Attention.requires.tolerance,
        d.out_ref,
        ref_hash,
        d.out_work,
        Gpu{
            .r = r,
            .q = d.q,
            .k_cache = d.k_cache,
            .v_cache = d.v_cache,
            .n_heads = c.n_heads,
            .n_kv_heads = c.n_kv_heads,
            .head_dim = c.head_dim,
            .t_len = c.t_len,
            .out = d.out_work,
        },
    );
}

fn attentionCase(
    gpa: std.mem.Allocator,
    io: std.Io,
    runner: anytype,
    pool: *zigferencer.pool.Pool,
    c: AtnCase,
) !void {
    // Allocate the buffers.
    const q_len: usize = @as(usize, c.n_heads) * @as(usize, c.head_dim);
    const kv_len: usize = @as(usize, c.t_len) * @as(
        usize,
        c.n_kv_heads,
    ) * @as(usize, c.head_dim);
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

    const parts = try gpa.alloc(f32, zigferencer.attn_cpu.partialLen(c.n_heads, c.head_dim));
    defer gpa.free(parts);

    // Fill the inputs with small magnitudes, so softmax stays numeric.
    for (q, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.07);
    }

    for (k_cache, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)) * 0.03);
    }

    for (v_cache, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)) * 0.05);
    }

    // Copy the values into f16 caches. A zero-length allocation costs an f32-only case nothing.
    const k16 = try gpa.alignedAlloc(f16, .of(f16), if (c.kv_f16) kv_len else 0);
    defer gpa.free(k16);

    const v16 = try gpa.alignedAlloc(f16, .of(f16), if (c.kv_f16) kv_len else 0);
    defer gpa.free(v16);

    for (k16, 0..) |*e, i| {
        e.* = @floatCast(k_cache[i]);
    }

    for (v16, 0..) |*e, i| {
        e.* = @floatCast(v_cache[i]);
    }

    // Compute the f32 reference.
    zigferencer.ops.attentionRowTyped(
        f32,
        q,
        c.n_heads,
        c.head_dim,
        k_cache,
        v_cache,
        c.n_kv_heads,
        c.t_len,
        parts,
        out_ref,
    );
    const ref_hash = hashOut(out_ref);
    // 4 FLOPs per (head * t * d): scores dot, exp, weighted-V dot, and the normalize multiply.
    const total_work_elems = @as(usize, c.t_len) * @as(
        usize,
        c.head_dim,
    ) * @as(usize, c.n_heads);
    const work: f64 = @as(f64, @floatFromInt(total_work_elems)) * 4.0;

    // Time the f32 CPU rows.
    const cpu32: AtnCpu(f32, zigferencer.ops) = .{
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
    cpu32_mt.pool = pool;
    _ = try cpuCheckedRow(
        io,
        total_work_elems,
        c.label,
        "naive_mt",
        work,
        out_ref,
        ref_hash,
        cpu32_mt,
    );
    try simdRows(f32, io, pool, total_work_elems, c.label, "", work, out_ref, cpu32);

    // Time the f16 rows and the GPU row.
    const data: AtnData = .{
        .q = q,
        .k_cache = k_cache,
        .v_cache = v_cache,
        .out_ref = out_ref,
        .out_work = out_work,
    };
    if (c.kv_f16) {
        const f16_data: AtnF16 = .{ .k = k16, .v = v16, .parts = parts, .out_ref = out_ref16 };
        try attentionF16Rows(io, pool, c, data, f16_data, total_work_elems, work);
    }

    if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            try attentionGpuRow(io, r, c, data, total_work_elems, work, ref_hash);
        }
    }

    flush();
}

/// All depths and row counts of one shape.
fn benchGridShape(gpa: std.mem.Allocator, io: std.Io, ctx: *GridCtx, sh: GridShape) !void {
    // Build the kernels and their descriptor sets.
    const kv: zigferencer.dtype.KvDtype = if (sh.kv_f16) .f16 else .f32;
    var single = try vkops.Attention.initKv(ctx, kv);
    defer single.deinit();

    var multi = try vkops.AttentionMultiStream.initKv(ctx, kv);
    defer multi.deinit();

    var passes = try vkops.AttentionSplit.initKv(ctx, kv);
    defer passes.deinit();

    const single_set = try vkops.OneShot.init(ctx, single.pipeline.set_layout, 4);
    defer single_set.deinit(ctx);

    const multi_set = try vkops.OneShot.init(ctx, multi.pipeline.set_layout, 4);
    defer multi_set.deinit(ctx);

    const split_set = try vkops.OneShot.init(
        ctx,
        passes.split.set_layout,
        vkops.AttentionSplit.split_bindings,
    );
    defer split_set.deinit(ctx);

    const merge_set = try vkops.OneShot.init(
        ctx,
        passes.merge.set_layout,
        vkops.AttentionSplit.merge_bindings,
    );
    defer merge_set.deinit(ctx);

    // Declare the host and device buffers.
    var buf_q: vkops.HostBuffer = .{ .ctx = ctx };
    defer buf_q.deinit();

    var buf_k: vkops.HostBuffer = .{ .ctx = ctx };
    defer buf_k.deinit();

    var buf_v: vkops.HostBuffer = .{ .ctx = ctx };
    defer buf_v.deinit();

    var buf_out: vkops.HostBuffer = .{ .ctx = ctx };
    defer buf_out.deinit();

    // The runners keep q, KV, and the outputs in device-local memory, so the timed kernels
    // read the same memory type.
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

    // Bundle the rig, then walk the depths.
    const rig: GridRig = .{
        .ctx = ctx,
        .single = &single,
        .multi = &multi,
        .passes = &passes,
        .single_set = single_set,
        .multi_set = multi_set,
        .split_set = split_set,
        .merge_set = merge_set,
        .buf_q = &buf_q,
        .buf_k = &buf_k,
        .buf_v = &buf_v,
        .buf_out = &buf_out,
        .dev_q = &dev_q,
        .dev_k = &dev_k,
        .dev_v = &dev_v,
        .dev_scratch = &dev_scratch,
        .dev_out = &dev_out,
    };

    const kv_elems: usize = @as(usize, sh.n_kv_heads) * sh.head_dim;
    const elem_bytes: usize = if (sh.kv_f16) 2 else 4;
    for (grid_depths) |depth| {
        if (depth > sh.max_depth) continue;

        const stream_elems = @as(usize, depth) * kv_elems;
        // The f16 shaders read the caches as 32-bit words, so each binding covers whole words.
        const kv_bytes = std.mem.alignForward(usize, stream_elems * elem_bytes, 4);
        if (kv_bytes > ctx.buffer_limits.storage_range) {
            bench_util.skipRow(sh.label, "attn", "range");
            continue;
        }

        // Fill the stream and upload it.
        const k_host = try gpa.alloc(f32, stream_elems);
        defer gpa.free(k_host);

        const v_host = try gpa.alloc(f32, stream_elems);
        defer gpa.free(v_host);

        const stream: GridStream = .{
            .depth = depth,
            .stream_elems = stream_elems,
            .kv_bytes = kv_bytes,
            .elem_bytes = elem_bytes,
            .k_host = k_host,
            .v_host = v_host,
            .scratch = zigferencer.attn_split.Scratch.init(sh.n_heads, sh.head_dim, depth),
        };
        try gridUploadStream(rig, sh, stream);
        try dev_scratch.ensure(stream.scratch.totalBytes());

        // Time every row count. The reporter keeps the label slices until the flush.
        var label_bufs: [grid_row_counts.len][48]u8 = undefined;

        for (grid_row_counts, 0..) |rows, ri| {
            try benchGridPoint(gpa, io, rig, sh, stream, rows, &label_bufs[ri]);
        }

        flush();
    }
}

/// Fill the host stream, copy it to the device buffers, and quantize the reference to f16.
fn gridUploadStream(rig: GridRig, sh: GridShape, stream: GridStream) !void {
    const k_host = stream.k_host;
    const v_host = stream.v_host;
    const stream_elems = stream.stream_elems;
    for (k_host, 0..) |*x, i| {
        x.* = @cos(@as(f32, @floatFromInt(i % 65_521)) * 0.03);
    }

    for (v_host, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i % 65_519)) * 0.05);
    }

    try rig.buf_k.ensure(stream.kv_bytes);
    try rig.buf_v.ensure(stream.kv_bytes);
    if (sh.kv_f16) {
        for (rig.buf_k.sliceTyped(f16, stream_elems), k_host) |*d, x| {
            d.* = @floatCast(x);
        }

        for (rig.buf_v.sliceTyped(f16, stream_elems), v_host) |*d, x| {
            d.* = @floatCast(x);
        }

        // The reference reads the values the cache stores.
        for (k_host) |*x| {
            x.* = @as(f16, @floatCast(x.*));
        }

        for (v_host) |*x| {
            x.* = @as(f16, @floatCast(x.*));
        }
    } else {
        @memcpy(rig.buf_k.sliceTyped(f32, stream_elems), k_host);
        @memcpy(rig.buf_v.sliceTyped(f32, stream_elems), v_host);
    }

    try rig.dev_k.ensure(stream.kv_bytes);
    try rig.dev_v.ensure(stream.kv_bytes);
    try vkops.runStagingCopies(rig.ctx, &.{
        .{ .src = rig.buf_k.buffer, .dst = rig.dev_k.buffer, .size = stream.kv_bytes },
        .{ .src = rig.buf_v.buffer, .dst = rig.dev_v.buffer, .size = stream.kv_bytes },
    });
}

/// One grid point: record `rows` query rows that end at the stream depth, time one submission,
/// and print the row.
fn benchGridPoint(
    gpa: std.mem.Allocator,
    io: std.Io,
    rig: GridRig,
    sh: GridShape,
    stream: GridStream,
    rows: u32,
    label_buf: *[48]u8,
) !void {
    const ctx = rig.ctx;
    const attn_split = zigferencer.attn_split;
    const depth = stream.depth;
    const row_elems: usize = @as(usize, sh.n_heads) * sh.head_dim;

    // Upload the query rows.
    const q_host = try gpa.alloc(f32, rows * row_elems);
    defer gpa.free(q_host);

    for (q_host, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)) * 0.07);
    }

    const q_bytes = q_host.len * @sizeOf(f32);
    try rig.buf_q.ensure(q_bytes);
    @memcpy(rig.buf_q.sliceTyped(f32, q_host.len), q_host);
    try rig.dev_q.ensure(q_bytes);
    try vkops.runStagingCopies(
        ctx,
        &.{.{ .src = rig.buf_q.buffer, .dst = rig.dev_q.buffer, .size = q_bytes }},
    );
    const out_bytes = q_host.len * @sizeOf(f32);
    try rig.buf_out.ensure(out_bytes);
    try rig.dev_out.ensure(out_bytes);
    gridBindSets(rig, stream, q_bytes, out_bytes);

    // Plan the rows. Row i attends to positions `[0, depth - rows + 1 + i)`, so the last row
    // reads the whole stream.
    var sites: [attn_split.max_plan_rows]attn_split.RowSite = undefined;
    for (sites[0..rows], 0..) |*site, i| {
        site.* = .{ .slot = 0, .t_len = depth - rows + 1 + @as(u32, @intCast(i)) };
    }

    const plan = attn_split.planRows(sites[0..rows], stream.scratch.max_splits);
    const binding: vkops.AttnKernels.Binding = .{
        .shape = .{
            .n_heads = sh.n_heads,
            .n_kv_heads = sh.n_kv_heads,
            .head_dim = sh.head_dim,
            .stream_stride_elems = @intCast(stream.stream_elems),
            .max_splits = stream.scratch.max_splits,
        },
        // A decode step binds the single-row set, and a batched step the multi-stream set.
        .resident = if (rows == 1) .{
            .single = rig.single_set.desc_set,
        } else .{ .multi = rig.multi_set.desc_set },
        .split = rig.split_set.desc_set,
        .merge = rig.merge_set.desc_set,
    };

    // Run once and read the output back.
    const step: GridStep = .{
        .ctx = ctx,
        .kernels = .{ .single = rig.single, .multi = rig.multi, .passes = rig.passes },
        .cb = rig.single_set.cmd_buffer,
        .plan = &plan,
        .binding = binding,
    };
    try step.run();
    try vkops.runStagingCopies(
        ctx,
        &.{.{ .src = rig.dev_out.buffer, .dst = rig.buf_out.buffer, .size = out_bytes }},
    );
    const out = rig.buf_out.sliceTyped(f32, q_host.len);
    const hsh = hashOut(out);

    // Compare the deepest row with the CPU reference.
    const last_q = q_host[(rows - 1) * row_elems ..][0..row_elems];
    const deepest = out[(rows - 1) * row_elems ..][0..row_elems];
    const errs = try gridReferenceErrors(gpa, sh, stream, last_q, deepest);

    // Time the submission and print the row.
    const t = try timed(io, rows * depth * row_elems, step);
    const unique_kv_bytes: f64 = @floatFromInt(2 * stream.stream_elems * stream.elem_bytes);
    const gbs = unique_kv_bytes / (t.ms * 1e6);
    const label = std.fmt.bufPrint(
        label_buf,
        "{s} t{d} r{d}",
        .{ sh.label, depth, rows },
    ) catch unreachable;
    const variant: []const u8 = if (sh.kv_f16) "attn_f16" else "attn_f32";
    const tol = vkops.AttnKernels.tolerance;
    rawRow(label, variant, t.ms, gbs, errs.rel, hsh, opStatus(tol, errs.rel, false));
}

/// Point the four descriptor sets at the device buffers of one grid point.
fn gridBindSets(rig: GridRig, stream: GridStream, q_bytes: usize, out_bytes: usize) void {
    // Describe the range of each buffer.
    const ctx = rig.ctx;
    const q_info: DescriptorBufferInfo = .{
        .buffer = rig.dev_q.buffer,
        .offset = 0,
        .range = q_bytes,
    };
    const k_info: DescriptorBufferInfo = .{
        .buffer = rig.dev_k.buffer,
        .offset = 0,
        .range = stream.kv_bytes,
    };
    const v_info: DescriptorBufferInfo = .{
        .buffer = rig.dev_v.buffer,
        .offset = 0,
        .range = stream.kv_bytes,
    };
    const scratch_info: DescriptorBufferInfo = .{
        .buffer = rig.dev_scratch.buffer,
        .offset = 0,
        .range = stream.scratch.totalBytes(),
    };
    const out_info: DescriptorBufferInfo = .{
        .buffer = rig.dev_out.buffer,
        .offset = 0,
        .range = out_bytes,
    };

    // Bind the sets of each kernel.
    vkops.updateStorageBindings(
        ctx,
        rig.single_set.desc_set,
        &.{ q_info, k_info, v_info, out_info },
    );
    vkops.updateStorageBindings(
        ctx,
        rig.multi_set.desc_set,
        &.{ q_info, k_info, v_info, out_info },
    );
    vkops.AttentionSplit.assertKBinding(k_info);
    vkops.AttentionSplit.assertVBinding(v_info);
    vkops.updateStorageBindings(
        ctx,
        rig.split_set.desc_set,
        &.{ q_info, k_info, v_info, scratch_info },
    );
    vkops.updateStorageBindings(ctx, rig.merge_set.desc_set, &.{ scratch_info, out_info });
}

/// The error of one device row against the CPU attention over the same stream.
fn gridReferenceErrors(
    gpa: std.mem.Allocator,
    sh: GridShape,
    stream: GridStream,
    last_q: []const f32,
    device_row: []const f32,
) !bench_util.Errors {
    const row_elems: usize = @as(usize, sh.n_heads) * sh.head_dim;
    const ref = try gpa.alloc(f32, row_elems);
    defer gpa.free(ref);

    const parts = try gpa.alloc(
        f32,
        zigferencer.attn_cpu.partialLen(sh.n_heads, sh.head_dim),
    );
    defer gpa.free(parts);

    zigferencer.ops.attentionRowTyped(
        f32,
        last_q,
        sh.n_heads,
        sh.head_dim,
        stream.k_host,
        stream.v_host,
        sh.n_kv_heads,
        stream.depth,
        parts,
        ref,
    );
    return computeErrors(ref, device_row);
}

/// The vector rows of one attention case: the serial `simd` row, then the 16-thread rows of the
/// `simd` and `simd_pf` kernels. `ref_out` holds the naive row of the same dtype.
fn simdRows(
    comptime E: type,
    io: std.Io,
    pool: *zigferencer.pool.Pool,
    scale: usize,
    label: []const u8,
    comptime suffix: []const u8,
    work: f64,
    ref_out: []const f32,
    naive_ctx: AtnCpu(E, zigferencer.ops),
) !void {
    // Time the serial `simd` row, which sets the hash.
    const simd_hash = try cpuCheckedRow(
        io,
        scale,
        label,
        "simd" ++ suffix,
        work,
        ref_out,
        null,
        naive_ctx.over(zigferencer.ops_simd, null),
    );
    // Time the threaded `simd` row.
    _ = try cpuCheckedRow(
        io,
        scale,
        label,
        "simd" ++ suffix ++ "_mt",
        work,
        ref_out,
        simd_hash,
        naive_ctx.over(zigferencer.ops_simd, pool),
    );
    // Time the threaded prefetch row. It must give the same bits as the serial row.
    _ = try cpuCheckedRow(
        io,
        scale,
        label,
        "simd_pf" ++ suffix ++ "_mt",
        work,
        ref_out,
        simd_hash,
        naive_ctx.over(zigferencer.ops_simd_pf, pool),
    );
}

/// One CPU row with a checked output: its error against `ref_out` fills rel_err, and a hash
/// that differs from `want_hash` prints FAIL. Returns the output hash.
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

/// Dispatch amortization of `ResidualAdd.runRepeat(N)` for a few N on one shape.
fn benchDispatchAmortization(
    gpa: std.mem.Allocator,
    io: std.Io,
    runner: *Runner,
) !void {
    printSectionHeader("fused dispatch amortization (residual_add)");
    // Allocate and fill the inputs.
    const n_elems: usize = 4096; // Llama-8B hidden size, representative
    const x = try gpa.alignedAlloc(f32, .of(f32), n_elems);
    defer gpa.free(x);

    const dst_init = try gpa.alignedAlloc(f32, .of(f32), n_elems);
    defer gpa.free(dst_init);

    const dst_work = try gpa.alignedAlloc(f32, .of(f32), n_elems);
    defer gpa.free(dst_work);

    const dst_ref = try gpa.alignedAlloc(f32, .of(f32), n_elems);
    defer gpa.free(dst_ref);

    for (x, 0..) |*v, i| {
        v.* = @sin(@as(f32, @floatFromInt(i)));
    }

    for (dst_init, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)));
    }

    // Time each dispatch count.
    const dispatch_counts = [_]u32{ 1, 4, 16, 64 };
    for (dispatch_counts) |n_dispatches| {
        // Compute the reference as N sequential `dst += x` adds, never `dst + N*x`. The two forms
        // differ at the last bit under f32 rounding, and the bit_exact rule needs one order.
        @memcpy(dst_ref, dst_init);
        var k: u32 = 0;
        while (k < n_dispatches) : (k += 1) {
            zigferencer.ops.residualAdd(x, dst_ref);
        }

        const ref_hash = hashOut(dst_ref);

        // Run the GPU repeat kernel and print the row.
        const Gpu = struct {
            r: *Runner,
            n: u32,
            x: []const f32,
            dst_init: []const f32,
            dst_work: []f32,

            pub fn run(self: @This()) !void {
                try self.r.residual_add.runRepeat(self.n, self.x, self.dst_init, self.dst_work);
            }
        };
        const gpu = Gpu{
            .r = runner,
            .n = n_dispatches,
            .x = x,
            .dst_init = dst_init,
            .dst_work = dst_work,
        };
        try gpu.run();
        const gpu_hash = hashOut(dst_work);
        const gpu_errs = computeErrors(dst_ref, dst_work);
        const gpu_t = try timed(io, n_elems, gpu);
        const ms_per_dispatch = gpu_t.ms / @as(f64, @floatFromInt(n_dispatches));
        // "G/s" here is per-dispatch element throughput.
        const gops = @as(f64, @floatFromInt(n_elems)) / (ms_per_dispatch * 1e6);
        var label_buf: [40]u8 = undefined;
        const label = std.fmt.bufPrint(
            &label_buf,
            "n_dispatches={d} (n={d})",
            .{ n_dispatches, n_elems },
        ) catch unreachable;
        const tol = zigferencer.ops_vulkan.ResidualAdd.requires.tolerance;
        rawRow(
            label,
            "vk_resid_x",
            ms_per_dispatch,
            gops,
            gpu_errs.rel,
            gpu_hash,
            opStatus(tol, gpu_errs.rel, gpu_hash == ref_hash),
        );
    }

    flush();
}
