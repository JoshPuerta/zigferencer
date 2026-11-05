//! Matmul benchmarks: the Q8_0 decode kernel and its CPU and GPU variants, the Q4_0 decode
//! kernel, and the batched (prefill) Q8_0 kernels. The matmul is the bandwidth-bound op that
//! decides decode speed, so every variant reports GFLOP/s next to its relative error against
//! `naive`, the scalar reference. `runQ80` owns the variant table and the eligibility report
//! that the decode matmul rows share.
//!
//! Cases. Three hidden-dim scales span the supported model families: 576 (SmolLM2), 2048
//! (Llama-3.2-1B and Qwen3.5-2B), and 4096 (Llama-3.1-8B and Qwen3.5-9B). Each scale covers a
//! square (attn_q), a wide-out rectangle (ffn_gate), a wide-in rectangle (ffn_down), and the
//! very wide lm_head. The four shapes show how a variant scales: a narrow shape underuses the
//! parallelism of the GPU, and a wide one lets it stretch. The kernel reads (in, out) and not
//! the source model, so each label states the shape alone.
//!
//! Skipped variants. A variant declines a case it cannot serve, such as an LDS-staged input
//! kernel at k past the LDS capacity. The bench buffers a `skip(<reason>)` row and goes on.
//!
//! Q4_0. Decode is bandwidth-bound, and Q4_0 reads about half the weight bytes of Q8_0. On the
//! large shapes the Q4_0 GFLOP/s therefore reaches about 2x the Q8_0 GFLOP/s.
//!
//! Batched. `benchBatchedMatmul` runs B input rows per dispatch, so the weight streams once for
//! B outputs. Compare its GFLOP/s against the decode `vk_q8_0_re` row to see the prefill
//! amortization. It is GPU-only, because no batched CPU op exists. The CPU reference is the
//! naive kernel looped over the B rows.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const zigferencer = @import("zigferencer");
const Tensor = zigferencer.tensor.Tensor;
const op_requires = zigferencer.op_requires;
const ops_vulkan = zigferencer.ops_vulkan;

const bench_util = @import("bench_util.zig");
const Case = bench_util.Case;
const cpuRow = bench_util.cpuRow;
const gpuRow = bench_util.gpuRow;
const rawRow = bench_util.rawRow;
const skipRow = bench_util.skipRow;
const flush = bench_util.flush;
const hashOut = bench_util.hashOut;
const computeErrors = bench_util.computeErrors;
const Errors = bench_util.Errors;
const printSectionHeader = bench_util.printSectionHeader;
const pickIters = bench_util.pickIters;
const f32_flops_per_mac = bench_util.f32_flops_per_mac;

// -- types ------------------------------------------------------------------

/// Where the kernel lives: a function pointer with the `linearF32Q80` signature, or a
/// `*LinearQ80*` kernel struct on the heap-owned Runner. GPU branches use a `Q80Weight` built
/// once per case.
const Kernel = union(enum) {
    cpu: *const fn ([]const f32, Tensor, []f32) void,
    gpu_q8_0: if (build_options.use_vulkan) *ops_vulkan.LinearQ80 else void,
    gpu_q8_0_reduce: if (build_options.use_vulkan) *ops_vulkan.LinearQ80Reduce else void,
    gpu_q8_0_subgroup: if (build_options.use_vulkan) *ops_vulkan.LinearQ80Subgroup else void,
};

const Variant = struct {
    name: []const u8,
    desc: []const u8,
    requires: op_requires.OpRequires,
    kernel: Kernel,
};

const BenchResult = struct { ms: f64, gflops: f64, hash: u64 };

// -- api --------------------------------------------------------------------

/// Q8_0 decode matmul across three shape scales, every eligible variant per case. It borrows
/// the GPU kernels from `runner`.
pub fn runQ80(gpa: std.mem.Allocator, io: std.Io, runner: anytype) !void {
    // Build the variant table with the CPU variants.
    var variants_buf: [16]Variant = undefined;
    var n: usize = 0;
    variants_buf[n] = .{
        .name = "naive",
        .desc = "scalar f32 reference",
        .requires = zigferencer.ops.requires,
        .kernel = .{ .cpu = zigferencer.ops.linearF32Q80 },
    };
    n += 1;
    variants_buf[n] = .{
        .name = "simd",
        .desc = "@Vector(8), AVX2 + FMA",
        .requires = zigferencer.ops_simd.requires,
        .kernel = .{ .cpu = zigferencer.ops_simd.linearF32Q80 },
    };
    n += 1;
    variants_buf[n] = .{
        .name = "simd_pf",
        .desc = "register-blocked (MR=4) + prefetch",
        .requires = zigferencer.ops_simd_pf.requires,
        .kernel = .{ .cpu = zigferencer.ops_simd_pf.linearF32Q80 },
    };
    n += 1;

    // Add the GPU variants.
    if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            variants_buf[n] = .{
                .name = "vk_q8_0",
                .desc = "GPU: 1 invocation per output row",
                .requires = zigferencer.ops_vulkan.LinearQ80.requires,
                .kernel = .{ .gpu_q8_0 = &r.linear_q8_0 },
            };
            n += 1;
            variants_buf[n] = .{
                .name = "vk_q8_0_re",
                .desc = "GPU: workgroup-per-row LDS reduce",
                .requires = zigferencer.ops_vulkan.LinearQ80Reduce.requires,
                .kernel = .{ .gpu_q8_0_reduce = &r.linear_q8_0_reduce },
            };
            n += 1;
            variants_buf[n] = .{
                .name = "vk_q8_0_sg",
                .desc = "GPU: subgroup reduce, LDS-staged input",
                .requires = zigferencer.ops_vulkan.LinearQ80Subgroup.requires,
                .kernel = .{ .gpu_q8_0_subgroup = &r.linear_q8_0_subgroup },
            };
            n += 1;
        }
    }

    const variants = variants_buf[0..n];

    // Print the eligibility report and the table header.
    printEligibility(variants);
    printSectionHeader("linearF32Q80 (matmul)");

    // Walk the shape cases.
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
        // Build the fixtures once per case: input, weight bytes, the Tensor wrapper for CPU
        // kernels, and the `Q80Weight` upload for GPU kernels.
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
            ?zigferencer.ops_vulkan.Q80Weight
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
                gpu_weight = try ops_vulkan.Q80Weight.init(
                    &runner.?.ctx,
                    w_bytes,
                    c.in,
                    c.out,
                );
            }
        }

        // Time every eligible variant against the naive reference.
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

        flush();
    }
}

/// Q4_0 decode matvec: the Q4_0 reduce kernel against the naive Q4_0 CPU reference, at shapes
/// of the Q8_0 section.
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
        // Build the fixtures.
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

        const work: f64 = @floatFromInt(c.in * c.out * f32_flops_per_mac);
        const scale = c.in * c.out;

        // Time the CPU reference row, which every build runs.
        zigferencer.ops.linearF32Q40(input, w_tensor, ref_out);
        const ref_hash = hashOut(ref_out);
        const CpuQ4 = struct {
            input: []const f32,
            w: Tensor,
            out: []f32,

            pub fn run(self: @This()) !void {
                zigferencer.ops.linearF32Q40(self.input, self.w, self.out);
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

        // Time the GPU row.
        if (comptime build_options.use_vulkan) {
            if (runner) |r| {
                var gpu_weight = try ops_vulkan.Q80Weight.initQ40(
                    &r.ctx,
                    w_bytes,
                    c.in,
                    c.out,
                );
                defer gpu_weight.deinit();

                const gpu_out = try gpa.alignedAlloc(f32, .of(f32), c.out);
                defer gpa.free(gpu_out);

                const tol = zigferencer.ops_vulkan.LinearQ40Reduce.requires.tolerance;
                const GpuQ4 = struct {
                    r: *zigferencer.vulkan_runner.Runner,
                    w: *const zigferencer.ops_vulkan.Q80Weight,
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

/// Batched (prefill) Q8_0 matmul: B input rows per dispatch, GPU-only.
pub fn benchBatchedMatmul(
    gpa: std.mem.Allocator,
    io: std.Io,
    runner: *zigferencer.vulkan_runner.Runner,
) !void {
    printSectionHeader("linearF32Q80 batched (prefill: B rows / dispatch)");
    const B: usize = 64;
    const cases = [_]Case{
        .{ .label = "ffn_gate  2048 -> 8192", .in = 2048, .out = 8192 },
        .{ .label = "attn_q    4096 -> 4096", .in = 4096, .out = 4096 },
        .{ .label = "ffn_down  14336 -> 4096", .in = 14336, .out = 4096 },
    };
    var batch = try zigferencer.ops_vulkan.LinearQ80Batch.init(&runner.ctx);
    defer batch.deinit();

    var gemm = try zigferencer.ops_vulkan.LinearQ80Gemm.init(&runner.ctx);
    defer gemm.deinit();

    for (cases) |c| {
        // Build the fixtures.
        std.debug.assert(c.in % 32 == 0);

        const row_bytes = (c.in / 32) * 34;
        const input = try gpa.alignedAlloc(f32, .of(f32), B * c.in);
        defer gpa.free(input);

        const w_bytes = try gpa.alignedAlloc(u8, .@"32", c.out * row_bytes);
        defer gpa.free(w_bytes);

        // Fill the weight. The input rows get their own values below.
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

        // Compute the CPU reference over the B rows.
        for (0..B) |bi| {
            zigferencer.ops.linearF32Q80(
                input[bi * c.in ..][0..c.in],
                w_tensor,
                ref_out[bi * c.out ..][0..c.out],
            );
        }

        const ref_hash = hashOut(ref_out);
        const work: f64 = @floatFromInt(B * c.in * c.out * f32_flops_per_mac);
        const scale = B * c.in * c.out;
        // The batched matmul fuses multiply and add over the full K, so it drifts from the
        // unfused naive reference more than a decode kernel. The tolerance is 5e-3.
        const tol: op_requires.Tolerance = .{ .relative = 5e-3 };

        const gpu_out = try gpa.alignedAlloc(f32, .of(f32), B * c.out);
        defer gpa.free(gpu_out);

        // Time the batch kernel.
        const Batch = struct {
            k: *zigferencer.ops_vulkan.LinearQ80Batch,
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

        // Time the gemm kernel.
        const Gemm = struct {
            k: *zigferencer.ops_vulkan.LinearQ80Gemm,
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

/// Warm up `kernel`, then time `pickIters` runs against the fixtures of the case. Returns the
/// ms per call, the GFLOP/s, and the output hash.
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
    for (0..iters) |_| {
        try callKernel(kernel, input, w_tensor, gpu_weight, out);
    }

    const t_end = std.Io.Clock.Timestamp.now(io, .awake);

    const elapsed_ns_i96 = t_start.durationTo(t_end).raw.toNanoseconds();
    const elapsed_ns: f64 = @floatFromInt(@as(i64, @intCast(elapsed_ns_i96)));
    const per_call_ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1e6;
    const flops_per_call: f64 = @floatFromInt(
        in_features * out_features * f32_flops_per_mac,
    );
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

/// Fill `input` with a sine and `w_bytes` with q8_0 blocks: scale 1.0 and a period-13 pattern.
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
        for (0..in_features / 32) |blk| {
            const blk_off = off + blk * 34;
            std.mem.writeInt(u16, w_bytes[blk_off..][0..2], 0x3c00, .little);
            for (0..32) |i| {
                const v_i32: i32 = @as(i32, @intCast((row + blk * 32 + i) % 13)) - 6;
                const v: i8 = @intCast(v_i32);
                w_bytes[blk_off + 2 + i] = @bitCast(v);
            }
        }
    }
}

/// Fill `input` with a sine and `w_bytes` with q4_0 blocks: scale 1.0 and a period-16 pattern.
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
        for (0..in_features / 32) |blk| {
            const blk_off = off + blk * 18;
            std.mem.writeInt(u16, w_bytes[blk_off..][0..2], 0x3c00, .little); // fp16 1.0
            for (0..16) |i| {
                // Pack two 4-bit nibbles per byte, each in [0, 15].
                const lo: u8 = @intCast((row + blk * 16 + i) % 16);
                const hi: u8 = @intCast((row + blk * 16 + i + 7) % 16);
                w_bytes[blk_off + 2 + i] = lo | (hi << 4);
            }
        }
    }
}
