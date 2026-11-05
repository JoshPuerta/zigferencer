//! Matmul benchmarks: the Q8_0 decode kernel and its CPU and GPU variants, the
//! Q4_0 decode kernel, and the batched (prefill) Q8_0 kernels. The matmul is the
//! bandwidth-bound op that decides decode speed, so every variant reports
//! GFLOP/s next to its relative error against `naive`, the scalar reference.
//! `runQ80` owns the variant table and the eligibility report that the decode
//! matmul rows share.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const z = @import("zigferencer");
const Tensor = z.tensor.Tensor;
const op_requires = z.op_requires;

const h = @import("bench_util.zig");
const Case = h.Case;
const cpuRow = h.cpuRow;
const gpuRow = h.gpuRow;
const rawRow = h.rawRow;
const skipRow = h.skipRow;
const flush = h.flush;
const hashOut = h.hashOut;
const computeErrors = h.computeErrors;
const Errors = h.Errors;
const printSectionHeader = h.printSectionHeader;
const pickIters = h.pickIters;
const f32_flops_per_mac = h.f32_flops_per_mac;

// -- types ------------------------------------------------------------------

/// Where the kernel lives. CPU side is a function pointer with the signature
/// `linearF32Q80`. GPU side is one of the `*LinearQ80*` kernel structs on the
/// heap-owned Runner. The GPU branches consume a `*const Q80Weight` built once
/// per case.
const Kernel = union(enum) {
    cpu: *const fn ([]const f32, Tensor, []f32) void,
    gpu_q8_0: if (build_options.use_vulkan) *z.ops_vulkan.LinearQ80 else void,
    gpu_q8_0_reduce: if (build_options.use_vulkan) *z.ops_vulkan.LinearQ80Reduce else void,
    gpu_q8_0_subgroup: if (build_options.use_vulkan) *z.ops_vulkan.LinearQ80Subgroup else void,
};

const Variant = struct {
    name: []const u8,
    desc: []const u8,
    requires: op_requires.OpRequires,
    kernel: Kernel,
};

const BenchResult = struct { ms: f64, gflops: f64, hash: u64 };

// -- api --------------------------------------------------------------------

/// Q8_0 decode matmul across three shape scales, every eligible variant per
/// case. Builds the variant table (borrowing GPU kernels from `runner`), prints
/// the eligibility report, then walks the cases.
pub fn runQ80(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    var variants_buf: [16]Variant = undefined;
    var n: usize = 0;
    variants_buf[n] = .{
        .name = "naive",
        .desc = "scalar f32 reference",
        .requires = z.ops.requires,
        .kernel = .{ .cpu = z.ops.linearF32Q80 },
    };
    n += 1;
    variants_buf[n] = .{
        .name = "simd",
        .desc = "@Vector(8), AVX2 + FMA",
        .requires = z.ops_simd.requires,
        .kernel = .{ .cpu = z.ops_simd.linearF32Q80 },
    };
    n += 1;
    variants_buf[n] = .{
        .name = "simd_pf",
        .desc = "register-blocked (MR=4) + prefetch",
        .requires = z.ops_simd_pf.requires,
        .kernel = .{ .cpu = z.ops_simd_pf.linearF32Q80 },
    };
    n += 1;
    if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            variants_buf[n] = .{
                .name = "vk_q8_0",
                .desc = "GPU: 1 invocation per output row",
                .requires = z.ops_vulkan.LinearQ80.requires,
                .kernel = .{ .gpu_q8_0 = &r.linear_q8_0 },
            };
            n += 1;
            variants_buf[n] = .{
                .name = "vk_q8_0_re",
                .desc = "GPU: workgroup-per-row LDS reduce",
                .requires = z.ops_vulkan.LinearQ80Reduce.requires,
                .kernel = .{ .gpu_q8_0_reduce = &r.linear_q8_0_reduce },
            };
            n += 1;
            variants_buf[n] = .{
                .name = "vk_q8_0_sg",
                .desc = "GPU: subgroup reduce, LDS-staged input",
                .requires = z.ops_vulkan.LinearQ80Subgroup.requires,
                .kernel = .{ .gpu_q8_0_subgroup = &r.linear_q8_0_subgroup },
            };
            n += 1;
        }
    }

    const variants = variants_buf[0..n];

    printEligibility(variants);
    printSectionHeader("linearF32Q80 (matmul)");

    // Three hidden-dim scales span the supported model families: small (576),
    // medium (2048), and large (4096). SmolLM2 is about 576. Llama-3.2-1B and
    // Qwen3.5-2B are about 2048. Llama-3.1-8B and Qwen3.5-9B are about 4096.
    // Each scale covers a square (attn_q), a wide-out rectangle (ffn_gate), a
    // wide-in rectangle (ffn_down), and the very wide lm_head. The four shapes
    // expose how each variant scales: a narrow shape underuses the parallelism
    // of the GPU, and a wide one lets it stretch. Each label states the shape
    // alone, because the kernel reads (in, out) and not the source model.
    const cases = [_]Case{
        .{ .label = "attn_q    576 -> 576", .in = 576, .out = 576 },
        .{ .label = "ffn_gate  576 -> 1536", .in = 576, .out = 1536 },
        .{ .label = "ffn_down  1536 -> 576", .in = 1536, .out = 576 },
        .{ .label = "lm_head   576 -> 49152", .in = 576, .out = 49152 },
        .{ .label = "attn_q    2048 -> 2048", .in = 2048, .out = 2048 },
        .{ .label = "ffn_gate  2048 -> 8192", .in = 2048, .out = 8192 },
        .{ .label = "ffn_down  8192 -> 2048", .in = 8192, .out = 2048 },
        .{ .label = "lm_head   2048 -> 128256", .in = 2048, .out = 128256 },
        .{ .label = "attn_q    4096 -> 4096", .in = 4096, .out = 4096 },
        .{ .label = "ffn_gate  4096 -> 14336", .in = 4096, .out = 14336 },
        .{ .label = "ffn_down  14336 -> 4096", .in = 14336, .out = 4096 },
        .{ .label = "lm_head   4096 -> 128256", .in = 4096, .out = 128256 },
    };

    for (cases) |c| {
        // Per-case fixtures: input, CPU-layout weight bytes, the Tensor wrapper
        // (for CPU kernels), and a `Q80Weight` upload (for GPU kernels). Built
        // once and reused across every eligible variant in this case.
        std.debug.assert(c.in % 32 == 0);

        const row_bytes = (c.in / 32) * 34;
        const total_w_bytes = c.out * row_bytes;

        const input = try gpa.alignedAlloc(f32, .of(f32), c.in);
        defer gpa.free(input);

        const w_bytes = try gpa.alignedAlloc(u8, .@"32", total_w_bytes);
        defer gpa.free(w_bytes);

        fillSyntheticInputAndWeight(input, w_bytes, c.in, c.out);

        const w_tensor: Tensor = .{
            .dtype = .q8_0,
            .n_dims = 2,
            .shape = .{ @intCast(c.in), @intCast(c.out), 0, 0 },
            .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
            .data = w_bytes,
        };

        const GpuWeightOpt = if (build_options.use_vulkan)
            ?z.ops_vulkan.Q80Weight
        else
            void;

        var gpu_weight: GpuWeightOpt = if (comptime build_options.use_vulkan) null else {};
        defer if (comptime build_options.use_vulkan) {
            if (gpu_weight) |*w| {
                w.deinit();
            }
        };

        if (comptime build_options.use_vulkan) {
            if (runner != null) {
                gpu_weight = try z.ops_vulkan.Q80Weight.init(&runner.?.ctx, w_bytes, c.in, c.out);
            }
        }

        std.debug.assert(std.mem.eql(u8, variants[0].name, "naive"));
        const ref_out = try gpa.alignedAlloc(f32, .of(f32), c.out);
        defer gpa.free(ref_out);

        for (variants, 0..) |v, i| {
            if (!isEligible(v)) continue;

            const is_naive = i == 0;
            const out = if (is_naive) ref_out else try gpa.alignedAlloc(f32, .of(f32), c.out);
            defer if (!is_naive) gpa.free(out);

            const r = runVariant(
                io,
                input,
                w_tensor,
                gpu_weight,
                v.kernel,
                out,
            ) catch |e| switch (e) {
                // A variant declines a case it cannot serve, such as an
                // LDS-staged input kernel at k past the LDS capacity. The bench
                // buffers a "skip(<reason>)" row and goes on to the next row.
                error.InFeaturesTooLargeForLdsKernel => {
                    skipRow(c.label, v.name, "skip(lds)");
                    continue;
                },
                else => return e,
            };
            const errs: Errors = if (is_naive)
                .{ .abs = 0.0, .rel = 0.0 }
            else
                computeErrors(ref_out, out);

            const ok = op_requires.toleranceSatisfied(
                v.requires.tolerance,
                errs.rel,
                r.hash == hashOut(ref_out),
            );
            const status: []const u8 = if (ok) "ok" else "FAIL";
            rawRow(c.label, v.name, r.ms, r.gflops, errs.rel, r.hash, status);
        }

        flush(); // colors the fastest/slowest variant for this shape
    }
}

/// Q4_0 decode-matvec section: it measures the Q4_0 reduce kernel against the
/// naive Q4_0 CPU reference, at the shapes the Q8_0 section uses. Decode is
/// bandwidth-bound, and Q4_0 reads about half the weight bytes of Q8_0. On the
/// large shapes the Q4_0 GFLOP/s therefore reaches about 2x the Q8_0 GFLOP/s.
pub fn benchQ40(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    printSectionHeader("linearF32Q40 (compare GFLOP/s vs the q8_0 section)");
    const cases = [_]Case{
        .{ .label = "ffn_down  1536 -> 576", .in = 1536, .out = 576 },
        .{ .label = "attn_q    4096 -> 4096", .in = 4096, .out = 4096 },
        .{ .label = "ffn_gate  4096 -> 14336", .in = 4096, .out = 14336 },
        .{ .label = "ffn_down  14336 -> 4096", .in = 14336, .out = 4096 },
        .{ .label = "lm_head   4096 -> 128256", .in = 4096, .out = 128256 },
    };

    for (cases) |c| {
        std.debug.assert(c.in % 32 == 0);

        const row_bytes = (c.in / 32) * 18;
        const total_w_bytes = c.out * row_bytes;

        const input = try gpa.alignedAlloc(f32, .of(f32), c.in);
        defer gpa.free(input);

        const w_bytes = try gpa.alignedAlloc(u8, .@"32", total_w_bytes);
        defer gpa.free(w_bytes);

        fillSyntheticInputAndWeightQ4(input, w_bytes, c.in, c.out);

        const w_tensor: Tensor = .{
            .dtype = .q4_0,
            .n_dims = 2,
            .shape = .{ @intCast(c.in), @intCast(c.out), 0, 0 },
            .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
            .data = w_bytes,
        };

        const ref_out = try gpa.alignedAlloc(f32, .of(f32), c.out);
        defer gpa.free(ref_out);

        const work = @as(f64, @floatFromInt(c.in * c.out * f32_flops_per_mac));
        const scale = c.in * c.out;

        // The CPU reference row (naive_q4) runs on every build.
        z.ops.linearF32Q40(input, w_tensor, ref_out);
        const ref_hash = hashOut(ref_out);
        const CpuQ4 = struct {
            input: []const f32,
            w: Tensor,
            out: []f32,

            pub fn run(self: @This()) !void {
                z.ops.linearF32Q40(self.input, self.w, self.out);
            }
        };
        try cpuRow(
            io,
            scale,
            c.label,
            "naive_q4",
            work,
            ref_hash,
            CpuQ4{ .input = input, .w = w_tensor, .out = ref_out },
        );

        if (comptime build_options.use_vulkan) {
            if (runner) |r| {
                var gpu_weight = try z.ops_vulkan.Q80Weight.initQ40(&r.ctx, w_bytes, c.in, c.out);
                defer gpu_weight.deinit();

                const gpu_out = try gpa.alignedAlloc(f32, .of(f32), c.out);
                defer gpa.free(gpu_out);

                const tol = z.ops_vulkan.LinearQ40Reduce.requires.tolerance;
                const GpuQ4 = struct {
                    r: *z.vulkan_runner.Runner,
                    w: *const z.ops_vulkan.Q80Weight,
                    input: []const f32,
                    out: []f32,

                    pub fn run(self: @This()) !void {
                        try self.r.linear_q4_0_reduce.dispatch(self.w, self.input, self.out);
                    }
                };
                try gpuRow(
                    io,
                    scale,
                    c.label,
                    "vk_q4_0_re",
                    work,
                    tol,
                    ref_out,
                    ref_hash,
                    gpu_out,
                    GpuQ4{ .r = r, .w = &gpu_weight, .input = input, .out = gpu_out },
                );
            }
        }

        flush();
    }
}

/// Batched (prefill) Q8_0 matmul: B input rows per dispatch, so the weight is
/// streamed once for B outputs. Compare GFLOP/s against the decode `vk_q8_0_re`
/// to see the prefill amortization. GPU-only (no batched CPU op). The CPU
/// reference is the naive kernel looped over the B rows.
pub fn benchBatchedMatmul(
    gpa: std.mem.Allocator,
    io: std.Io,
    runner: *z.vulkan_runner.Runner,
) !void {
    printSectionHeader("linearF32Q80 batched (prefill: B rows / dispatch)");
    const B: usize = 64;
    const cases = [_]Case{
        .{ .label = "ffn_gate  2048 -> 8192", .in = 2048, .out = 8192 },
        .{ .label = "attn_q    4096 -> 4096", .in = 4096, .out = 4096 },
        .{ .label = "ffn_down  14336 -> 4096", .in = 14336, .out = 4096 },
    };
    var batch = try z.ops_vulkan.LinearQ80Batch.init(&runner.ctx);
    defer batch.deinit();

    var gemm = try z.ops_vulkan.LinearQ80Gemm.init(&runner.ctx);
    defer gemm.deinit();

    for (cases) |c| {
        std.debug.assert(c.in % 32 == 0);

        const row_bytes = (c.in / 32) * 34;
        const input = try gpa.alignedAlloc(f32, .of(f32), B * c.in);
        defer gpa.free(input);

        const w_bytes = try gpa.alignedAlloc(u8, .@"32", c.out * row_bytes);
        defer gpa.free(w_bytes);

        // Fill the weight (first arg gets one input row, overwritten below).
        fillSyntheticInputAndWeight(input[0..c.in], w_bytes, c.in, c.out);
        for (input, 0..) |*v, i| {
            v.* = @sin(@as(f32, @floatFromInt(i)) * 0.001);
        }

        const w_tensor: Tensor = .{
            .dtype = .q8_0,
            .n_dims = 2,
            .shape = .{ @intCast(c.in), @intCast(c.out), 0, 0 },
            .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
            .data = w_bytes,
        };

        const ref_out = try gpa.alignedAlloc(f32, .of(f32), B * c.out);
        defer gpa.free(ref_out);

        for (0..B) |bi| {
            z.ops.linearF32Q80(
                input[bi * c.in ..][0..c.in],
                w_tensor,
                ref_out[bi * c.out ..][0..c.out],
            );
        }

        const ref_hash = hashOut(ref_out);
        const work = @as(f64, @floatFromInt(B * c.in * c.out * f32_flops_per_mac));
        const scale = B * c.in * c.out;
        // The batched matmul fuses its multiply and add over the full K, so it
        // drifts from the unfused naive reference more than a decode kernel
        // does. The tolerance against naive is therefore 5e-3.
        const tol: op_requires.Tolerance = .{ .relative = 5e-3 };

        const gpu_out = try gpa.alignedAlloc(f32, .of(f32), B * c.out);
        defer gpa.free(gpu_out);

        const Batch = struct {
            k: *z.ops_vulkan.LinearQ80Batch,
            input: []const f32,
            w: []const u8,
            b: usize,
            in: usize,
            out: usize,
            o: []f32,

            pub fn run(self: @This()) !void {
                try self.k.runOneShot(self.input, self.w, self.b, self.in, self.out, self.o);
            }
        };
        try gpuRow(
            io,
            scale,
            c.label,
            "vk_q8_batch",
            work,
            tol,
            ref_out,
            ref_hash,
            gpu_out,
            Batch{
                .k = &batch,
                .input = input,
                .w = w_bytes,
                .b = B,
                .in = c.in,
                .out = c.out,
                .o = gpu_out,
            },
        );

        const Gemm = struct {
            k: *z.ops_vulkan.LinearQ80Gemm,
            input: []const f32,
            w: []const u8,
            b: usize,
            in: usize,
            out: usize,
            o: []f32,

            pub fn run(self: @This()) !void {
                try self.k.runOneShot(self.input, self.w, self.b, self.in, self.out, self.o);
            }
        };
        try gpuRow(
            io,
            scale,
            c.label,
            "vk_q8_gemm",
            work,
            tol,
            ref_out,
            ref_hash,
            gpu_out,
            Gemm{
                .k = &gemm,
                .input = input,
                .w = w_bytes,
                .b = B,
                .in = c.in,
                .out = c.out,
                .o = gpu_out,
            },
        );
        flush();
    }
}

// -- helpers ----------------------------------------------------------------

fn isEligible(v: Variant) bool {
    if (!op_requires.isMetBy(v.requires, builtin.cpu)) return false;
    if (build_options.use_vulkan) {
        return op_requires.runtimeMatches(
            v.requires,
            .vulkan,
        ) or op_requires.runtimeMatches(v.requires, .cpu);
    } else {
        return op_requires.runtimeMatches(v.requires, .cpu);
    }
}

fn printEligibility(variants: []const Variant) void {
    std.debug.print(
        "\nlinearF32Q80 variants for this build target ([+] eligible, [-] not):\n",
        .{},
    );
    for (variants) |v| {
        const mark: []const u8 = if (isEligible(v)) "+" else "-";
        std.debug.print(
            "  [{s}] {s:<11} {s:<7} {s}\n",
            .{ mark, v.name, @tagName(v.requires.runtime), v.desc },
        );
    }
}

/// Warm-up + `pickIters` timed iterations of `kernel`, against fixtures built
/// once for the case. Returns ms/G/s/hash.
fn runVariant(
    io: std.Io,
    input: []const f32,
    w_tensor: Tensor,
    gpu_weight: anytype, // ?Q80Weight under vulkan, void otherwise
    kernel: Kernel,
    out: []f32,
) !BenchResult {
    const in_features = input.len;
    const out_features = out.len;

    try callKernel(kernel, input, w_tensor, gpu_weight, out); // warm-up

    const iters: usize = pickIters(in_features * out_features);
    const t_start = std.Io.Clock.Timestamp.now(io, .awake);
    var k: usize = 0;
    while (k < iters) : (k += 1) {
        try callKernel(kernel, input, w_tensor, gpu_weight, out);
    }

    const t_end = std.Io.Clock.Timestamp.now(io, .awake);

    const elapsed_ns_i96 = t_start.durationTo(t_end).raw.toNanoseconds();
    const elapsed_ns: f64 = @floatFromInt(@as(i64, @intCast(elapsed_ns_i96)));
    const per_call_ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1e6;
    const flops_per_call = @as(f64, @floatFromInt(in_features * out_features * f32_flops_per_mac));
    const gflops = (flops_per_call * @as(f64, @floatFromInt(iters))) / elapsed_ns;

    return .{ .ms = per_call_ms, .gflops = gflops, .hash = hashOut(out) };
}

fn callKernel(
    kernel: Kernel,
    input: []const f32,
    w_tensor: Tensor,
    gpu_weight: anytype,
    out: []f32,
) !void {
    switch (kernel) {
        .cpu => |fn_ptr| fn_ptr(input, w_tensor, out),
        .gpu_q8_0 => |kp| {
            if (comptime !build_options.use_vulkan) unreachable;
            try kp.dispatch(&gpu_weight.?, input, out);
        },
        .gpu_q8_0_reduce => |kp| {
            if (comptime !build_options.use_vulkan) unreachable;
            try kp.dispatch(&gpu_weight.?, input, out);
        },
        .gpu_q8_0_subgroup => |kp| {
            if (comptime !build_options.use_vulkan) unreachable;
            try kp.dispatch(&gpu_weight.?, input, out);
        },
    }
}

fn fillSyntheticInputAndWeight(
    input: []f32,
    w_bytes: []u8,
    in_features: usize,
    out_features: usize,
) void {
    for (input, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)));
    }

    const row_bytes = (in_features / 32) * 34;
    for (0..out_features) |row| {
        const off = row * row_bytes;
        var blk: usize = 0;
        while (blk < in_features / 32) : (blk += 1) {
            const blk_off = off + blk * 34;
            std.mem.writeInt(u16, w_bytes[blk_off..][0..2], 0x3c00, .little);
            var i: usize = 0;
            while (i < 32) : (i += 1) {
                const v_i32: i32 = @as(i32, @intCast((row + blk * 32 + i) % 13)) - 6;
                const v: i8 = @intCast(v_i32);
                w_bytes[blk_off + 2 + i] = @bitCast(v);
            }
        }
    }
}

fn fillSyntheticInputAndWeightQ4(
    input: []f32,
    w_bytes: []u8,
    in_features: usize,
    out_features: usize,
) void {
    for (input, 0..) |*x, i| {
        x.* = @sin(@as(f32, @floatFromInt(i)));
    }

    const row_bytes = (in_features / 32) * 18;
    for (0..out_features) |row| {
        const off = row * row_bytes;
        var blk: usize = 0;
        while (blk < in_features / 32) : (blk += 1) {
            const blk_off = off + blk * 18;
            std.mem.writeInt(u16, w_bytes[blk_off..][0..2], 0x3c00, .little); // fp16 1.0
            var i: usize = 0;
            while (i < 16) : (i += 1) {
                // Two 4-bit nibbles per byte. Pattern in [0,15].
                const lo: u8 = @intCast((row + blk * 16 + i) % 16);
                const hi: u8 = @intCast((row + blk * 16 + i + 7) % 16);
                w_bytes[blk_off + 2 + i] = lo | (hi << 4);
            }
        }
    }
}
