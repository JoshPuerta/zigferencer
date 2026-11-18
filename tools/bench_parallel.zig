//! Micro-benchmarks for the three costs that bound the CPU threading defaults:
//!
//!   benchFanoutCost    the cost of one pool dispatch inside
//!                      `parallel.linearBatch`. It times a near-empty kernel
//!                      forced over the fan-out floor against the serial
//!                      inline path (n_threads = 1). This cost sets the
//!                      smallest matmul that gains from a fan-out.
//!   benchDispatchCost  the cost of one dispatch of empty units, one unit per
//!                      participant, on an `Io.Group` and on the fork-join
//!                      pool. The pool lines run with a spin budget of 0 and of
//!                      50 µs, so they show the sleep path and the spin path.
//!   benchBandwidth     the DRAM read bandwidth of the host against thread
//!                      count. It sum-reduces a buffer far larger than L3,
//!                      fanned out the way the matmul kernels are, which gives
//!                      the practical ceiling that the derived decode GB/s is
//!                      measured against. The bench derives bandwidth from
//!                      wall time, because a hardware counter is not always
//!                      available.
//!
//! Output. Every section prints a plain table to stdout. No section has a rel_err or a status
//! column, so no section uses the shared reporter. All run on the CPU and take no GPU runner.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Tensor = zigferencer.tensor.Tensor;
const parallel = zigferencer.parallel;
const Pool = zigferencer.pool.Pool;
const bench_util = @import("bench_util.zig");

// -- types ------------------------------------------------------------------

/// The context of one empty pool dispatch. The unit body is empty, so the
/// time is the cost of the publish, the wake, the claims, and the join.
const PoolCtx = struct {
    pool: *Pool,
    parts: u32,

    pub fn run(self: @This()) !void {
        const units: EmptyUnits = .{ .sink = 0 };
        self.pool.run(EmptyUnits, &units, emptyUnit, self.parts, self.parts);
    }
};

/// The context of one empty `Io.Group` fan-out: one task per participant and
/// one await, the shape of the matmul fan-out.
const GroupCtx = struct {
    io: std.Io,
    parts: u32,

    pub fn run(self: @This()) !void {
        var group: std.Io.Group = .init;
        var t: u32 = 0;
        while (t < self.parts) : (t += 1) {
            group.async(self.io, emptyTask, .{});
        }

        try group.await(self.io);
    }
};

const EmptyUnits = struct { sink: u32 };

// -- api --------------------------------------------------------------------

pub fn benchFanoutCost(gpa: std.mem.Allocator, io: std.Io) !void {
    // Build a weight that gives two units or more at every pool size up to 64.
    // The shape is n = 256 outputs with k = 32, one q8_0 block of 34 bytes per output.
    const k: u32 = 32;
    const n: u32 = 256;
    const output_bytes: usize = 34;
    const w_data = try gpa.alloc(u8, n * output_bytes);
    defer gpa.free(w_data);

    @memset(w_data, 0);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ k, n, 0, 0 },
        .byte_stride = .{ @intCast(output_bytes), 0, 0, 0 },
        .data = w_data,
    };
    const input = try gpa.alloc(f32, k);
    defer gpa.free(input);

    @memset(input, 1.0);
    const out = try gpa.alloc(f32, n);
    defer gpa.free(out);

    // Define the closure that times one pool dispatch of the near-empty kernel.
    const Ctx = struct {
        pool: *Pool,
        policy: parallel.Policy,
        input: []const f32,
        w: Tensor,
        out: []f32,

        pub fn run(self: @This()) !void {
            parallel.linearBatch(
                self.pool,
                self.policy,
                tinyKernel,
                self.input,
                self.w,
                self.out,
                1,
            );
        }
    };

    // Print the table header.
    std.debug.print(
        "\nzigferencer bench: parallel dispatch cost of linearBatch (pool path, " ++
            "near-empty kernel)\n",
        .{},
    );
    std.debug.print(
        "{s:>8}{s:>14}{s:>16}{s:>14}\n",
        .{ "threads", "us/call", "overhead_us", "us/thread" },
    );

    // Time each thread count against the one-thread line.
    const threads = [_]u32{ 1, 2, 4, 8, 16, 32 };
    var base_us: f64 = 0;
    for (threads) |t| {
        const policy: parallel.Policy = .{ .n_threads = t, .min_parallel_flops = 0 };
        var pool: Pool = undefined;
        try parallel.initPool(&pool, io, policy);
        defer pool.deinit();

        const timed = try bench_util.timed(
            io,
            1000,
            Ctx{ .pool = &pool, .policy = policy, .input = input, .w = w, .out = out },
        );
        const us = timed.ms * 1000.0;
        if (t == 1) {
            base_us = us;
        }

        const overhead = us - base_us;
        const per_thread = if (t > 1) overhead / @as(f64, @floatFromInt(t)) else 0;
        std.debug.print("{d:>8}{d:>14.3}{d:>16.3}{d:>14.3}\n", .{ t, us, overhead, per_thread });
    }
}

/// The pool participants are the owner plus its workers, so a pool of `parts`
/// threads serves a dispatch of `parts` participants.
pub fn benchDispatchCost(io: std.Io) !void {
    // Print the table header.
    std.debug.print(
        "\nzigferencer bench: dispatch cost of empty units (Io.Group against the pool)\n",
        .{},
    );
    std.debug.print(
        "{s:>8}{s:>14}{s:>16}{s:>16}\n",
        .{ "parts", "group_us", "pool_spin0_us", "pool_spin50_us" },
    );

    const participants = [_]u32{ 2, 4, 8, 16 };
    for (participants) |parts| {
        const group = try bench_util.timed(io, 0, GroupCtx{ .io = io, .parts = parts });
        var pool_us: [2]f64 = undefined;
        for ([_]u64{ 0, parallel.default_spin_ns }, &pool_us) |spin_ns, *us| {
            var p: Pool = undefined;
            try p.init(io, parts, spin_ns);
            defer p.deinit();

            const timed = try bench_util.timed(io, 0, PoolCtx{ .pool = &p, .parts = parts });
            us.* = timed.ms * 1000.0;
        }

        std.debug.print(
            "{d:>8}{d:>14.3}{d:>16.3}{d:>16.3}\n",
            .{ parts, group.ms * 1000.0, pool_us[0], pool_us[1] },
        );
    }
}

pub fn benchBandwidth(gpa: std.mem.Allocator, io: std.Io) !void {
    // Allocate a buffer far past the 32 MiB L3, so every read misses to DRAM. That is the
    // regime of streaming the weights of a multi-GB model once per decode step.
    const n_floats: usize = (2 << 30) / @sizeOf(f32);
    const buf = gpa.alloc(f32, n_floats) catch {
        std.debug.print("\nparallel bandwidth: skipped (the 2 GiB allocation failed)\n", .{});
        return;
    };
    defer gpa.free(buf);

    for (buf, 0..) |*x, i| {
        x.* = @floatFromInt(i & 0xff); // first-touch every page
    }

    // Define the closure that sums the buffer in `n_threads` chunks.
    const bytes: f64 = @floatFromInt(buf.len * @sizeOf(f32));
    const max_threads: usize = 32;
    var partials: [max_threads]f64 = undefined;

    const Ctx = struct {
        buf: []const f32,
        partials: []f64,
        io: std.Io,
        n_threads: usize,

        pub fn run(self: @This()) !void {
            var group: std.Io.Group = .init;
            var t: usize = 0;
            while (t < self.n_threads) : (t += 1) {
                const start = t * self.buf.len / self.n_threads;
                const end = (t + 1) * self.buf.len / self.n_threads;
                group.async(self.io, sumChunk, .{ self.buf[start..end], &self.partials[t] });
            }

            group.await(self.io) catch |e| std.debug.panic("bandwidth await: {t}", .{e});
        }
    };

    // Print the table header.
    std.debug.print(
        "\nzigferencer bench: DRAM read bandwidth (2 GiB sum-reduce, Io.Group fan-out)\n",
        .{},
    );
    std.debug.print("{s:>8}{s:>12}{s:>10}{s:>10}\n", .{ "threads", "ms", "GB/s", "scaling" });

    // Time each thread count against the one-thread line.
    const threads = [_]usize{ 1, 2, 4, 8, 16, 32 };
    var base_gbs: f64 = 0;
    var sink: f64 = 0;
    for (threads) |t| {
        const timed = try bench_util.timed(
            io,
            1_000_000_000,
            Ctx{ .buf = buf, .partials = &partials, .io = io, .n_threads = t },
        );
        for (partials[0..t]) |p| {
            sink += p;
        }

        const gbs = bytes / (timed.ms * 1e6);
        if (t == 1) {
            base_gbs = gbs;
        }

        std.debug.print(
            "{d:>8}{d:>12.3}{d:>10.1}{d:>9.2}x\n",
            .{ t, timed.ms, gbs, gbs / base_gbs },
        );
    }

    std.mem.doNotOptimizeAway(sink);
}

// -- helpers ----------------------------------------------------------------

/// Near-empty batched linear kernel: one write per call keeps the dispatch alive, so a timed
/// `parallel.linearBatch` at batch 1 shows the pool fan-out cost. The fan-out still slices
/// `weight`.
fn tinyKernel(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    _ = weight;
    _ = batch;
    _ = out_stride;
    if (outs.len > 0) {
        outs[0] = inputs[0];
    }
}

fn sumChunk(slice: []const f32, out: *f64) std.Io.Cancelable!void {
    var s: f64 = 0;
    for (slice) |x| {
        s += x;
    }

    out.* = s;
}

fn emptyUnit(units: *const EmptyUnits, unit: u32) void {
    _ = units;
    _ = unit;
}

fn emptyTask() std.Io.Cancelable!void {}
