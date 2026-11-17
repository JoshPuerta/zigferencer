//! The compute and cache roofs of the host. `zig build bench -- peak` runs this section alone. Each
//! line times one kernel on 1, 8, and 16 threads and prints the total rate of all threads.
//!
//! Lines. `fmaChains` runs independent `@mulAdd` chains of `Vec`. `mulAddChains` runs the same
//! chains as a separate multiply and add, the form that strict float mode emits. In both, the
//! multiply is on the chain. `fmaDotChains` and `mulAddDotChains` run the dot form of the matmul
//! kernels, `acc + w * y`, where only the add is on the chain. `int8DotChains` runs chains of
//! `vpmaddubsw`, `vpmaddwd`, and `vpaddd`, the int8 dot step of AVX2. `readSum` sums a per-thread
//! buffer that fits in L1, in L2, or in L3. The DRAM roof is the `benchBandwidth` table of
//! `bench_parallel`, which `bench.zig` prints after this section.
//!
//! Method. Each thread runs the same kernel on its own accumulators or its own buffer, fanned out
//! on an `Io.Group` as `benchBandwidth` does. The rate is the work of all threads over the wall
//! time of one run, after one shorter warm-up run. Every kernel is a `noinline` function with no
//! comptime parameter, so the disassembly of the bench binary shows its body under its own name.
//!
//! Chains. A chain is a loop-carried dependency, so the number of chains sets how many operations
//! the core can have in flight. The multiply and add chains converge to the fixed point 2 of
//! `x = 0.5 * x + 1`, so no lane goes subnormal or infinite. The dot chains add `w * y` and then
//! `w * -y`, so each lane returns to its start value and stays bounded.
//!
//! Comptime. An `inline fn` with a comptime chain count unrolls each chain set into registers. The
//! non-generic kernel picks the body with a `switch`. A comptime check of x86_64 with AVX2 gates
//! the inline assembly of the int8 line, and another target prints that line as skipped.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

const Vec = @Vector(8, f32);
const VecI32 = @Vector(8, i32);
const VecI16 = @Vector(16, i16);
const VecI8 = @Vector(32, i8);
const VecU8 = @Vector(32, u8);

const lanes = 8;
const flop_per_lane = 2;
const products_per_maddubs = 32;
const read_accumulators = 8;
const read_floats_per_step = lanes * read_accumulators;

const thread_counts = [_]u32{ 1, 8, 16 };
const float_chain_counts = [_]u32{ 8, 10, 12 };
const add_dot_chain_counts = [_]u32{ 6, 8, 10, 12 };
const int8_chain_counts = [_]u32{ 8, 10 };
const read_buffer_bytes = [_]usize{ 16 << 10, 256 << 10, 2 << 20 };
const read_buffer_names = [_][]const u8{ "16 KiB", "256 KiB", "2 MiB" };

/// Vector steps per thread in one run. At 2 steps per cycle (AMD publication 56665, two FMA
/// pipes) and the 4.9 GHz boost of the CPU of the reference setup (its AMD product page), one
/// run takes more than 0.2 s.
const steps_per_run: u64 = 2_400_000_000;

/// Bytes read per thread in one run. The core loads at most two 32-byte vectors per cycle
/// (publication 56665), so one run at 4.9 GHz takes more than 0.2 s at every level.
const read_bytes_per_run: u64 = 64 << 30;

/// The warm-up run does this fraction of the work. It starts the threads and the clock ramp.
const warm_up_divisor: u64 = 8;

const has_avx2_int8 = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

// -- types ------------------------------------------------------------------

const Kernel = enum { fma, mul_add, fma_dot, mul_add_dot, int8_dot, read_sum };

/// The work of one thread. `steps` counts steps per chain, or passes over the buffer for
/// `read_sum`. `buffers` holds one slice of `per_thread` floats for each thread.
const Job = struct {
    kernel: Kernel,
    chains: u32,
    steps: u64,
    buffers: []const f32,
    per_thread: usize,
};

/// One run: the same job on `threads` tasks of one `Io.Group`, joined by one await.
const FanOut = struct {
    io: std.Io,
    job: Job,
    threads: u32,

    pub fn run(self: @This()) !void {
        var group: std.Io.Group = .init;
        var t: u32 = 0;
        while (t < self.threads) : (t += 1) {
            group.async(self.io, runJob, .{ self.job, t });
        }

        try group.await(self.io);
    }
};

// -- api --------------------------------------------------------------------

pub fn benchPeak(gpa: std.mem.Allocator, io: std.Io) !void {
    const flop_per_step = flop_per_lane * lanes;
    printTitle("FMA roof, dependent form x = x * m + a (GFLOP/s, 2 flop per lane)");
    try printChainTable(io, .fma, &float_chain_counts, flop_per_step);

    printTitle("multiply and add roof, dependent form x = x * m + a (GFLOP/s)");
    try printChainTable(io, .mul_add, &float_chain_counts, flop_per_step);

    printTitle("FMA roof, dot form acc = acc + w * y (GFLOP/s, 2 flop per lane)");
    try printChainTable(io, .fma_dot, &float_chain_counts, flop_per_step);

    printTitle("multiply and add roof, dot form acc = acc + w * y (GFLOP/s)");
    try printChainTable(io, .mul_add_dot, &add_dot_chain_counts, flop_per_step);

    printTitle("int8 dot roof (G int8 products/s, 32 per step)");
    if (comptime has_avx2_int8) {
        try printChainTable(io, .int8_dot, &int8_chain_counts, products_per_maddubs);
    } else {
        std.debug.print("{s:<12}{s:>10}{s:>9}{s:>12}\n", .{ "int8_dot", "-", "-", "skipped" });
    }

    printTitle("cache read roof (GB/s, per-thread buffer)");
    try printReadTable(gpa, io);
}

// -- helpers ----------------------------------------------------------------

fn printTitle(title: []const u8) void {
    std.debug.print("\nzigferencer bench: {s}\n", .{title});
}

fn printChainTable(
    io: std.Io,
    kernel: Kernel,
    chain_counts: []const u32,
    work_per_step: u64,
) !void {
    assert(kernel != .read_sum);
    assert(chain_counts.len > 0);

    std.debug.print("{s:<12}{s:>10}{s:>9}{s:>12}\n", .{ "kernel", "chains", "threads", "rate" });
    for (chain_counts) |chains| {
        assert(steps_per_run % chains == 0);
        const job: Job = .{
            .kernel = kernel,
            .chains = chains,
            .steps = steps_per_run / chains,
            .buffers = &.{},
            .per_thread = 0,
        };
        for (thread_counts) |threads| {
            const seconds = try timeJob(io, job, threads);
            const work: f64 = @floatFromInt(threads * steps_per_run * work_per_step);
            std.debug.print(
                "{s:<12}{d:>10}{d:>9}{d:>12.1}\n",
                .{ @tagName(kernel), chains, threads, work / seconds / 1e9 },
            );
        }
    }
}

fn printReadTable(gpa: std.mem.Allocator, io: std.Io) !void {
    const max_threads = thread_counts[thread_counts.len - 1];
    std.debug.print("{s:<12}{s:>10}{s:>9}{s:>12}\n", .{ "kernel", "size", "threads", "rate" });
    for (read_buffer_bytes, read_buffer_names) |bytes, name| {
        const per_thread = bytes / @sizeOf(f32);
        assert(per_thread % read_floats_per_step == 0);
        assert(read_bytes_per_run % bytes == 0);

        const buffers = try gpa.alignedAlloc(f32, .@"64", per_thread * max_threads);
        defer gpa.free(buffers);

        for (buffers, 0..) |*x, i| {
            x.* = @floatFromInt(i & 0xff);
        }

        const job: Job = .{
            .kernel = .read_sum,
            .chains = read_accumulators,
            .steps = read_bytes_per_run / bytes,
            .buffers = buffers,
            .per_thread = per_thread,
        };
        for (thread_counts) |threads| {
            const seconds = try timeJob(io, job, threads);
            const total: f64 = @floatFromInt(threads * read_bytes_per_run);
            std.debug.print(
                "{s:<12}{s:>10}{d:>9}{d:>12.1}\n",
                .{ "read_sum", name, threads, total / seconds / 1e9 },
            );
        }
    }
}

/// The wall seconds of one run of `job` on `threads` tasks, after a shorter warm-up run.
fn timeJob(io: std.Io, job: Job, threads: u32) !f64 {
    assert(threads > 0);
    assert(job.steps >= warm_up_divisor);

    var warm_up = job;
    warm_up.steps = job.steps / warm_up_divisor;
    try (FanOut{ .io = io, .job = warm_up, .threads = threads }).run();

    const t_start = std.Io.Clock.Timestamp.now(io, .awake);
    try (FanOut{ .io = io, .job = job, .threads = threads }).run();
    const t_end = std.Io.Clock.Timestamp.now(io, .awake);
    const ns: i64 = @intCast(t_start.durationTo(t_end).raw.toNanoseconds());
    assert(ns > 0);
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn runJob(job: Job, t: u32) std.Io.Cancelable!void {
    switch (job.kernel) {
        .fma => std.mem.doNotOptimizeAway(fmaChains(job.chains, job.steps, 0.5, 1.0)),
        .mul_add => std.mem.doNotOptimizeAway(mulAddChains(job.chains, job.steps, 0.5, 1.0)),
        .fma_dot => std.mem.doNotOptimizeAway(fmaDotChains(job.chains, job.steps, 0.5, 0.25)),
        .mul_add_dot => {
            std.mem.doNotOptimizeAway(mulAddDotChains(job.chains, job.steps, 0.5, 0.25));
        },
        .int8_dot => if (comptime has_avx2_int8) {
            std.mem.doNotOptimizeAway(int8DotChains(job.chains, job.steps));
        } else unreachable,
        .read_sum => {
            const start = t * job.per_thread;
            const buffer = job.buffers[start..][0..job.per_thread];
            std.mem.doNotOptimizeAway(readSum(buffer, job.steps));
        },
    }
}

noinline fn fmaChains(chains: u32, steps: u64, m: f32, a: f32) f32 {
    assert(steps > 0);
    return switch (chains) {
        8 => fmaBody(8, steps, m, a),
        10 => fmaBody(10, steps, m, a),
        12 => fmaBody(12, steps, m, a),
        else => unreachable,
    };
}

inline fn fmaBody(comptime chains: u32, steps: u64, m: f32, a: f32) f32 {
    const mv: Vec = @splat(m);
    const av: Vec = @splat(a);
    var acc: [chains]Vec = undefined;
    inline for (&acc, 0..) |*x, c| {
        x.* = @splat(@floatFromInt(c));
    }

    var step: u64 = 0;
    while (step < steps) : (step += 1) {
        inline for (&acc) |*x| {
            x.* = @mulAdd(Vec, x.*, mv, av);
        }
    }

    return sumLanes(chains, acc);
}

noinline fn mulAddChains(chains: u32, steps: u64, m: f32, a: f32) f32 {
    assert(steps > 0);
    return switch (chains) {
        8 => mulAddBody(8, steps, m, a),
        10 => mulAddBody(10, steps, m, a),
        12 => mulAddBody(12, steps, m, a),
        else => unreachable,
    };
}

inline fn mulAddBody(comptime chains: u32, steps: u64, m: f32, a: f32) f32 {
    const mv: Vec = @splat(m);
    const av: Vec = @splat(a);
    var acc: [chains]Vec = undefined;
    inline for (&acc, 0..) |*x, c| {
        x.* = @splat(@floatFromInt(c));
    }

    var step: u64 = 0;
    while (step < steps) : (step += 1) {
        inline for (&acc) |*x| {
            x.* = x.* * mv + av;
        }
    }

    return sumLanes(chains, acc);
}

noinline fn fmaDotChains(chains: u32, steps: u64, w: f32, y: f32) f32 {
    assert(steps > 0);
    assert(steps % 2 == 0);
    return switch (chains) {
        8 => dotBody(8, true, steps, w, y),
        10 => dotBody(10, true, steps, w, y),
        12 => dotBody(12, true, steps, w, y),
        else => unreachable,
    };
}

noinline fn mulAddDotChains(chains: u32, steps: u64, w: f32, y: f32) f32 {
    assert(steps > 0);
    assert(steps % 2 == 0);
    return switch (chains) {
        6 => dotBody(6, false, steps, w, y),
        8 => dotBody(8, false, steps, w, y),
        10 => dotBody(10, false, steps, w, y),
        12 => dotBody(12, false, steps, w, y),
        else => unreachable,
    };
}

/// The empty `asm volatile` makes `w` opaque before each product, so the compiler cannot hoist
/// or share the product, and it emits no instruction. One `w` register serves all chains.
inline fn dotBody(comptime chains: u32, comptime fused: bool, steps: u64, w: f32, y: f32) f32 {
    var wv: Vec = @splat(w);
    const ys = [2]Vec{ @splat(y), @splat(-y) };
    var acc: [chains]Vec = undefined;
    inline for (&acc, 0..) |*x, c| {
        x.* = @splat(@floatFromInt(c));
    }

    var step: u64 = 0;
    while (step < steps) : (step += 2) {
        inline for (ys) |yv| {
            inline for (&acc) |*x| {
                asm volatile (""
                    : [wv] "+x" (wv),
                );
                if (fused) {
                    x.* = @mulAdd(Vec, wv, yv, x.*);
                } else {
                    x.* = x.* + wv * yv;
                }
            }
        }
    }

    return sumLanes(chains, acc);
}

noinline fn int8DotChains(chains: u32, steps: u64) i32 {
    assert(steps > 0);
    return switch (chains) {
        8 => int8DotBody(8, steps),
        10 => int8DotBody(10, steps),
        else => unreachable,
    };
}

/// The accumulator is an input and an output of each block, so no block can leave the loop.
/// The sums wrap, which `vpaddd` does by definition.
inline fn int8DotBody(comptime chains: u32, steps: u64) i32 {
    const unsigned: VecU8 = @splat(1);
    const signed: VecI8 = @splat(1);
    const ones: VecI16 = @splat(1);
    var acc: [chains]VecI32 = @splat(@splat(0));
    var step: u64 = 0;
    while (step < steps) : (step += 1) {
        inline for (&acc) |*x| {
            var product: VecI32 = undefined;
            x.* = asm (
                \\vpmaddubsw %[s], %[u], %[p]
                \\vpmaddwd %[ones], %[p], %[p]
                \\vpaddd %[p], %[acc], %[acc]
                : [acc] "=x" (-> VecI32),
                  [p] "=&x" (product),
                : [_] "0" (x.*),
                  [u] "x" (unsigned),
                  [s] "x" (signed),
                  [ones] "x" (ones),
            );
        }
    }

    var total: VecI32 = @splat(0);
    for (acc) |x| {
        total +%= x;
    }

    return @reduce(.Add, total);
}

noinline fn readSum(buffer: []const f32, passes: u64) f32 {
    assert(buffer.len > 0);
    assert(buffer.len % read_floats_per_step == 0);
    assert(passes > 0);

    var acc: [read_accumulators]Vec = @splat(@splat(0));
    var pass: u64 = 0;
    while (pass < passes) : (pass += 1) {
        var i: usize = 0;
        while (i < buffer.len) : (i += read_floats_per_step) {
            inline for (&acc, 0..) |*x, k| {
                const v: Vec = buffer[i + k * lanes ..][0..lanes].*;
                x.* += v;
            }
        }
    }

    return sumLanes(read_accumulators, acc);
}

inline fn sumLanes(comptime chains: u32, acc: [chains]Vec) f32 {
    var total: Vec = @splat(0);
    for (acc) |x| {
        total += x;
    }

    return @reduce(.Add, total);
}
