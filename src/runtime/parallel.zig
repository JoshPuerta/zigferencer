//! Row-parallel matmul dispatcher over the fork-join `Pool`. It wraps a
//! linear-kernel function pointer. When serial (one pool thread), below the
//! FLOPs floor, or with fewer than two units, it calls the kernel directly, at
//! zero overhead: no pool touch, no alloc, no sync. Otherwise it splits the
//! output rows into units and runs them on the pool.
//!
//! The dispatcher is bit-deterministic across thread counts. Exactly one unit
//! computes each output row in the reduction order of the kernel. The bench
//! token hash therefore holds across thread counts. `--threads N`,
//! `--min-parallel-flops F`, and `--spin-us N` override the defaults.
//!
//! The module also holds the control units of CPU attention and of the GDN
//! step. They spread the work units of one row over the same pool, with the
//! same gate, `Policy.parts`.
//!
//! To fan out a new op on the pool, do these steps:
//!
//! 1. Divide the op into units. The result of a unit must not change with the
//!    worker that runs it.
//! 2. Write a context value and a unit function for `Pool.run`.
//! 3. Count the FLOPs of the op.
//! 4. Get the participant count from `Policy.parts`. If it is 1, run the
//!    units inline on the caller.
//! 5. Add a test that the fan-out at several thread counts gives the bits of
//!    the serial path.

const std = @import("std");
const builtin = @import("builtin");
const attn_cpu = @import("../core/attn_cpu.zig");
const gdn = @import("../ops/gdn.zig");
const limits = @import("../core/limits.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;

const log = std.log.scoped(.parallel);

// -- constants --------------------------------------------------------------

/// The default fan-out floor in FLOPs. One floor serves the matmuls, the
/// attention, and the Gated-DeltaNet heads. A sweep of the floor measured
/// 250 kFLOP best on a small model and on a 2 GB model. A dispatch on the
/// pool costs about half a microsecond at 8 participants, so an op of this
/// size gains from a split.
pub const default_min_parallel_flops: u64 = 250_000;

/// The default spin budget of a pool worker. A worker that sleeps adds a
/// futex wake to the next dispatch, and a 50 us spin covers the gap between
/// the matmuls of one decode step.
pub const default_spin_ns: u64 = 50_000;

/// The cap on `--spin-us`, one second. A longer spin keeps idle workers on
/// the cores and gives no gain on the next dispatch.
pub const max_spin_us: u64 = 1_000_000;

/// The cap on the default thread count. Decode saturates the memory
/// bandwidth of a dual-channel desktop between 4 and 8 threads, and prefill
/// keeps gaining to 8. The cap keeps sustained load off the rest of the
/// cores. `--threads N` raises it.
pub const max_default_threads: u32 = 8;

/// The cap on the logical CPUs that `hostCores` reads. A host with more
/// CPUs falls back to the logical estimate, and the log names the cause.
const max_host_cpus: u32 = 512;

/// The longest `thread_siblings_list` text that `hostCores` keeps.
const sibling_text_max: usize = 48;

// -- types ------------------------------------------------------------------

/// A batched linear kernel: `out = input @ weight.T` with weight in `[in, out]`
/// (GGUF row-major) layout, over `batch` input rows against one weight. Row b
/// reads `inputs[b*in ..]` and writes `outs[b*out_stride ..][0..out]`.
/// `out_stride` lets a kernel given a weight-row *slice* still write into
/// full-width rows. The kernel is opaque to this module. `llama_runner.zig` passes
/// the build-selected variant in.
pub const LinearBatchKernel = *const fn (
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void;

/// The threading policy of one process: the pool size and the gates that
/// decide when an op runs on the pool. `forHost` gives the defaults, and the
/// command flags replace single fields through `resolve`.
pub const Policy = struct {
    /// The pool size, the calling thread included. 1 (or 0) = single-threaded.
    n_threads: u32 = 1,
    /// An op with fewer FLOPs than this runs inline on the caller.
    min_parallel_flops: u64 = default_min_parallel_flops,
    /// The time a pool worker spins on the epoch word before it sleeps.
    spin_ns: u64 = default_spin_ns,

    /// The default policy of the host: `defaultThreads` of the physical cores,
    /// the default floor, and the default spin budget.
    pub fn forHost(io: std.Io) Policy {
        const policy: Policy = .{ .n_threads = defaultThreads(hostCores(io).physical) };
        std.debug.assert(policy.n_threads >= 1);
        std.debug.assert(policy.n_threads <= max_default_threads);
        return policy;
    }

    /// The participant count of one op on a pool of `threads` threads: 1 runs
    /// the op inline on the caller. An op runs inline when the pool has one
    /// thread, when it has fewer than two units, or when its FLOPs fall under
    /// `min_parallel_flops`. Otherwise `min(threads, n_units)` participants
    /// claim the units. Every fan-out decision comes here, so a new rule, such
    /// as a cap for each bound of an op or a floor for each op class, goes here.
    pub fn parts(self: Policy, threads: u32, n_units: usize, flops: u64) u32 {
        std.debug.assert(threads >= 1);

        const result: u32 = if (threads <= 1 or n_units < 2 or flops < self.min_parallel_flops)
            1
        else
            @intCast(@min(threads, n_units));

        std.debug.assert(result >= 1);
        std.debug.assert(result <= threads);
        std.debug.assert(result <= @max(n_units, 1));
        return result;
    }
};

/// The physical core count of the host, and where it came from.
pub const HostCores = struct {
    physical: u32,
    /// True when the count comes from the sysfs topology, false when it is
    /// the estimate of half the logical CPUs.
    from_sysfs: bool,
};

/// Where a field of a resolved policy came from.
pub const Source = enum { auto, flag, placement };

/// The thread flags of a command. A null field keeps the host default.
pub const Flags = struct {
    /// `--threads N`. Zero keeps the default.
    threads: ?u32 = null,
    /// `--min-parallel-flops F`.
    min_parallel_flops: ?u64 = null,
    /// `--spin-us N`, in microseconds, at most `max_spin_us`.
    spin_us: ?u64 = null,
};

/// A policy with the source of each field, for the log line of a command.
pub const Resolved = struct {
    policy: Policy,
    threads_from: Source,
    flops_from: Source,
    spin_from: Source,
};

pub const PoolError = pool_mod.InitError || error{TooManyThreads};

/// The flag values that `resolve` refuses.
pub const ResolveError = error{ TooManyThreads, SpinTooLong };

/// The inputs of one matmul as its units read them. Unit `u` covers the
/// output rows `[u * rows, min(n, (u + 1) * rows))`.
const MatmulUnits = struct {
    kernel: LinearBatchKernel,
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    n: usize,
    rows: usize,

    fn unit(self: *const MatmulUnits, u: u32) void {
        const start = @as(usize, u) * self.rows;
        std.debug.assert(start < self.n);
        const end = @min(self.n, start + self.rows);
        const slice = sliceWeightRows(self.weight, start, end);
        self.kernel(self.inputs, slice, self.outs[start..], self.batch, self.n);
    }
};

/// The inputs of one GDN step as its units read them. Unit `u` is value
/// head `u`.
const GdnUnits = struct {
    d: gdn.Dims,
    heads: gdn.Heads,

    fn unit(self: *const GdnUnits, u: u32) void {
        std.debug.assert(u < self.d.n_v_heads);

        gdn.stepHeads(self.d, self.heads, u, @as(usize, u) + 1);
    }
};

/// The inputs of one attention row as its units read them. The pool erases
/// the type of the context, so the element type and the kernels bind through
/// this type.
fn Units(comptime E: type, comptime Kernels: type) type {
    return struct {
        q: []const f32,
        n_heads: usize,
        head_dim: usize,
        k_cache: []const E,
        v_cache: []const E,
        n_kv_heads: usize,
        t_len: u32,
        n_parts: u32,
        parts_buf: []f32,

        const Self = @This();

        /// Run the units `[u_lo, u_hi)` in ascending order.
        fn run(self: *const Self, u_lo: usize, u_hi: usize) void {
            std.debug.assert(u_lo < u_hi);
            std.debug.assert(u_hi <= self.n_kv_heads * self.n_parts);

            const n_heads32: u32 = @intCast(self.n_heads);
            const head_dim32: u32 = @intCast(self.head_dim);
            const group_size = self.n_heads / self.n_kv_heads;
            const rec_stride = attn_cpu.recordStride(head_dim32);
            // The part kernel finds head h at `(h - h_lo) * rec_stride` past
            // the record of head h_lo, where the merge reads it.
            std.debug.assert(attn_cpu.recordOffset(n_heads32, head_dim32, n_heads32 - 1, 0) ==
                (self.n_heads - 1) * rec_stride);
            var u = u_lo;
            while (u < u_hi) : (u += 1) {
                const g = u / self.n_parts;
                const p: u32 = @intCast(u % self.n_parts);
                const part = attn_cpu.partRange(self.t_len, p);
                const h_lo = g * group_size;
                const rec_lo = attn_cpu.recordOffset(n_heads32, head_dim32, @intCast(h_lo), p);
                Kernels.attentionPartTyped(
                    E,
                    self.q,
                    h_lo,
                    h_lo + group_size,
                    self.n_heads,
                    self.head_dim,
                    self.k_cache,
                    self.v_cache,
                    self.n_kv_heads,
                    part.t_lo,
                    part.t_hi,
                    self.parts_buf[rec_lo..],
                    rec_stride,
                );
            }
        }

        fn unit(self: *const Self, u: u32) void {
            self.run(u, @as(usize, u) + 1);
        }
    };
}

// -- api --------------------------------------------------------------------

/// Start `pool` with the threads and the spin budget of `policy`. A policy over
/// `limits.max_threads` threads returns `error.TooManyThreads`, because the
/// pool holds a fixed array of workers.
pub fn initPool(pool: *Pool, io: std.Io, policy: Policy) PoolError!void {
    const n_threads = @max(policy.n_threads, 1);
    if (n_threads > limits.max_threads) return error.TooManyThreads;

    try pool.init(io, n_threads, policy.spin_ns);
    std.debug.assert(pool.threads() == n_threads);
}

/// The default thread count for `physical` cores.
pub fn defaultThreads(physical: u32) u32 {
    std.debug.assert(physical >= 1);

    const threads = @min(physical, max_default_threads);
    std.debug.assert(threads >= 1);
    return threads;
}

/// Apply the flags of a command to the host policy. A command whose
/// placement runs no CPU op gets one thread, so its pool starts no workers.
/// A `--threads` value over `limits.max_threads` returns
/// `error.TooManyThreads`. A `--spin-us` value over `max_spin_us` returns
/// `error.SpinTooLong`.
pub fn resolve(host: Policy, flags: Flags, runs_cpu_ops: bool) ResolveError!Resolved {
    std.debug.assert(host.n_threads >= 1);

    var out: Resolved = .{
        .policy = host,
        .threads_from = .auto,
        .flops_from = .auto,
        .spin_from = .auto,
    };
    if (flags.threads) |t| {
        if (t > limits.max_threads) return error.TooManyThreads;
        if (t != 0) {
            out.policy.n_threads = t;
            out.threads_from = .flag;
        }
    }

    if (flags.min_parallel_flops) |f| {
        out.policy.min_parallel_flops = f;
        out.flops_from = .flag;
    }

    if (flags.spin_us) |us| {
        if (us > max_spin_us) return error.SpinTooLong;
        out.policy.spin_ns = us * std.time.ns_per_us;
        out.spin_from = .flag;
    }

    if (!runs_cpu_ops) {
        out.policy.n_threads = 1;
        out.threads_from = .placement;
    }

    std.debug.assert(out.policy.n_threads >= 1);
    std.debug.assert(out.policy.n_threads <= limits.max_threads);
    if (flags.spin_us != null) {
        std.debug.assert(out.policy.spin_ns <= max_spin_us * std.time.ns_per_us);
    }

    return out;
}

/// Log the resolved policy of a command, with the source of each field.
pub fn logResolved(r: Resolved) void {
    log.info("threads {d} ({t}), floor {d} FLOPs ({t}), spin {d} ns ({t})", .{
        r.policy.n_threads,
        r.threads_from,
        r.policy.min_parallel_flops,
        r.flops_from,
        r.policy.spin_ns,
        r.spin_from,
    });
}

/// The physical cores of the host. The count is the number of distinct
/// `thread_siblings_list` texts in the sysfs topology, because SMT siblings
/// share one text. Without a readable topology the count is half the
/// logical CPUs, the SMT-2 estimate.
pub fn hostCores(io: std.Io) HostCores {
    if (readSysfsCores(io)) |physical| {
        std.debug.assert(physical >= 1);

        return .{ .physical = physical, .from_sysfs = true };
    }

    const logical: u32 = @intCast(@max(1, @min(std.Thread.getCpuCount() catch 1, max_host_cpus)));
    const physical: u32 = @max(1, logical / 2);
    std.debug.assert(1 <= physical);
    std.debug.assert(physical <= logical);
    return .{ .physical = physical, .from_sysfs = false };
}

/// The number of distinct sibling-list texts in `lists`, with surrounding
/// white space ignored. Each physical core has one text, and every logical
/// CPU of the core repeats it.
pub fn countCores(lists: []const []const u8) u32 {
    std.debug.assert(lists.len >= 1);
    std.debug.assert(lists.len <= max_host_cpus);

    var distinct: u32 = 0;
    for (lists, 0..) |list, i| {
        const text = std.mem.trim(u8, list, " \t\r\n");
        const seen = for (lists[0..i]) |earlier| {
            if (std.mem.eql(u8, std.mem.trim(u8, earlier, " \t\r\n"), text)) break true;
        } else false;

        if (!seen) {
            distinct += 1;
        }
    }

    std.debug.assert(distinct >= 1);
    std.debug.assert(distinct <= lists.len);
    return distinct;
}

/// Run a linear matmul over `batch` input rows, parallelized when the work is
/// large enough. The serial path is a direct kernel call: no pool touch, no
/// allocation, no synchronization.
///
/// The partition stays over OUTPUT rows, not the batch. Each unit streams a
/// disjoint weight chunk once for ALL rows, which is bandwidth-optimal. A batch
/// partition streams the full weight per thread, the exact traffic that
/// batching exists to remove. Exactly one unit computes each (row, output)
/// element in the reduction order of the kernel, so the result is bit-identical
/// across thread counts. `outs` must hold `batch` rows at stride
/// `weight.shape[1]`.
///
/// A unit is a chunk of `matmulUnitRows` output rows. The pool has about four
/// units per participant, so a participant that finishes early claims more.
pub fn linearBatch(
    pool: *Pool,
    policy: Policy,
    kernel: LinearBatchKernel,
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
) void {
    const k: usize = @intCast(weight.shape[0]);
    const n: usize = @intCast(weight.shape[1]);
    std.debug.assert(batch > 0);
    std.debug.assert(n > 0);
    std.debug.assert(inputs.len == batch * k);
    std.debug.assert(outs.len >= batch * n);
    const threads = pool.threads();
    std.debug.assert(@max(policy.n_threads, 1) == threads);
    const flops: u64 = 2 * @as(u64, k) * @as(u64, n) * @as(u64, batch);
    const rows = matmulUnitRows(n, threads);
    const n_units = std.math.divCeil(usize, n, rows) catch unreachable;

    const parts = policy.parts(threads, n_units, flops);
    if (parts == 1) {
        log.debug(
            "linearBatch serial: flops {d}, floor {d}, n {d}, batch {d}",
            .{ flops, policy.min_parallel_flops, n, batch },
        );
        kernel(inputs, weight, outs, batch, n);
        return;
    }

    log.debug(
        "linearBatch fanout: {d} parts, {d} units of {d} rows over n {d}, batch {d}, flops {d}",
        .{ parts, n_units, rows, n, batch, flops },
    );
    const ctx: MatmulUnits = .{
        .kernel = kernel,
        .inputs = inputs,
        .weight = weight,
        .outs = outs,
        .batch = batch,
        .n = n,
        .rows = rows,
    };
    pool.run(MatmulUnits, &ctx, MatmulUnits.unit, @intCast(n_units), parts);
}

/// The output rows of one matmul unit: `ceil(n / (4 * threads))` rounded up to
/// a multiple of 4. The size is a function of the shape and the pool size.
/// It never changes a bit, because the kernels give a row the same bits at
/// every chunk boundary.
pub fn matmulUnitRows(n: usize, threads: u32) usize {
    std.debug.assert(n > 0);
    std.debug.assert(threads >= 1);

    const per_unit = std.math.divCeil(usize, n, 4 * @as(usize, threads)) catch unreachable;
    const rows = std.mem.alignForward(usize, per_unit, 4);
    std.debug.assert(rows >= 4);
    std.debug.assert(rows % 4 == 0);
    return rows;
}

/// The control unit of one CPU decode attention row. The geometry of
/// `attn_cpu` splits the row into `c = attn_cpu.partCount(t_len)` parts. A work
/// unit is one KV group over one part, so the row holds `n_kv_heads * c` units.
/// Unit `u` is group `u / c` and part `u % c`. A unit writes the records of its
/// group heads for its part, and no two units write one record. After every
/// unit ends, `Kernels.attentionMerge` combines the records on the calling
/// thread.
///
/// Any assignment of units to participants gives the same bits. The geometry
/// is a function of `t_len` alone. A unit runs the same kernel on the same
/// inputs on every participant. The merge adds the parts in ascending order.
/// So the token hash holds across thread counts, the same guarantee as
/// `linearBatch`.
///
/// `Policy.parts` decides between the serial row and the fan-out.
///
/// `Kernels` is an ops namespace that declares `attentionPartTyped` and
/// `attentionMerge`. `E` is the KV element type, and it reaches the kernel
/// unchanged. `parts_buf` holds at least `attn_cpu.partialLen(n_heads,
/// head_dim)` f32s, a length that the window does not change.
pub fn attentionTyped(
    comptime E: type,
    comptime Kernels: type,
    pool: *Pool,
    policy: Policy,
    q: []const f32,
    n_heads: usize,
    head_dim: usize,
    k_cache: []const E,
    v_cache: []const E,
    n_kv_heads: usize,
    t_len: usize,
    parts_buf: []f32,
    out: []f32,
) void {
    comptime std.debug.assert(E == f32 or E == f16);
    std.debug.assert(n_kv_heads >= 1);
    std.debug.assert(n_heads >= n_kv_heads);
    std.debug.assert(n_heads % n_kv_heads == 0);
    std.debug.assert(head_dim >= 1);
    std.debug.assert(t_len >= 1);
    std.debug.assert(t_len <= limits.max_model_ctx);
    std.debug.assert(q.len == n_heads * head_dim);
    std.debug.assert(out.len == n_heads * head_dim);
    std.debug.assert(k_cache.len >= t_len * n_kv_heads * head_dim);
    std.debug.assert(v_cache.len >= t_len * n_kv_heads * head_dim);
    const n_heads32: u32 = @intCast(n_heads);
    const head_dim32: u32 = @intCast(head_dim);
    std.debug.assert(parts_buf.len >= attn_cpu.partialLen(n_heads32, head_dim32));
    const threads = pool.threads();
    std.debug.assert(@max(policy.n_threads, 1) == threads);

    const units: Units(E, Kernels) = .{
        .q = q,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .k_cache = k_cache,
        .v_cache = v_cache,
        .n_kv_heads = n_kv_heads,
        .t_len = @intCast(t_len),
        .n_parts = attn_cpu.partCount(@intCast(t_len)),
        .parts_buf = parts_buf,
    };
    const n_units: usize = n_kv_heads * units.n_parts;
    std.debug.assert(n_units <= n_kv_heads * limits.attn_cpu_parts);
    // 4 FLOP per (head, position, dim): the score dot plus the weighted-V sum.
    const flops: u64 = 4 * @as(u64, n_heads) * @as(u64, head_dim) * @as(u64, t_len);

    const parts = policy.parts(threads, n_units, flops);
    if (parts == 1) {
        log.debug(
            "attention serial: flops {d}, floor {d}, t_len {d}, units {d}",
            .{ flops, policy.min_parallel_flops, t_len, n_units },
        );
        units.run(0, n_units);
    } else {
        log.debug(
            "attention fanout: {d} parts over {d} units, t_len {d}",
            .{ parts, n_units, t_len },
        );
        pool.run(Units(E, Kernels), &units, Units(E, Kernels).unit, @intCast(n_units), parts);
    }

    Kernels.attentionMerge(parts_buf, units.n_parts, n_heads, head_dim, out);
}

/// The control unit of one GDN step of one row. The caller thread runs
/// `gdn.stepPrologue`, then the value heads run on the pool, one head per
/// unit. A head writes only its own state and readout, so any assignment of
/// heads to participants gives the bits of `gdn.step`. The finite check runs
/// after every head ends. `heads.qkv` must be `qkv`, the buffer that the
/// prologue writes.
///
/// `Policy.parts` decides between the serial heads and the fan-out.
pub fn gdnStep(
    pool: *Pool,
    policy: Policy,
    d: gdn.Dims,
    qkv: []f32,
    conv_w: []const f32,
    conv_state: []f32,
    heads: gdn.Heads,
) void {
    std.debug.assert(heads.qkv.ptr == qkv.ptr);
    std.debug.assert(heads.qkv.len == qkv.len);
    std.debug.assert(d.n_v_heads >= 1);

    const threads = pool.threads();
    std.debug.assert(@max(policy.n_threads, 1) == threads);
    // 7 FLOP per state element: decay 1, key read 2, rank-1 write 2, query read 2.
    const flops: u64 = 7 * @as(u64, d.n_v_heads) * d.head_dim * d.head_dim;

    gdn.stepPrologue(d, qkv, conv_w, conv_state, heads.rms_eps);
    const n_units: u32 = @intCast(d.n_v_heads);
    const parts = policy.parts(threads, n_units, flops);
    if (parts == 1) {
        log.debug(
            "gdn serial: flops {d}, floor {d}, heads {d}",
            .{ flops, policy.min_parallel_flops, d.n_v_heads },
        );
        gdn.stepHeads(d, heads, 0, d.n_v_heads);
    } else {
        log.debug("gdn fanout: {d} parts over {d} heads, flops {d}", .{ parts, n_units, flops });
        const ctx: GdnUnits = .{ .d = d, .heads = heads };
        pool.run(GdnUnits, &ctx, GdnUnits.unit, n_units, parts);
    }

    gdn.assertFinite(d, heads.o_out);
}

// -- helpers ----------------------------------------------------------------

/// Read the sibling list of each logical CPU from sysfs and count the
/// physical cores. The read stops at the first CPU without a topology file.
/// It returns null when no CPU has one, when a read fails, or when a list or
/// the CPU count passes its cap. A cap that cuts the read logs the cause.
fn readSysfsCores(io: std.Io) ?u32 {
    if (builtin.os.tag != .linux) return null;

    var texts: [max_host_cpus][sibling_text_max]u8 = undefined;
    var lists: [max_host_cpus][]const u8 = undefined;
    var n: u32 = 0;
    const cwd = std.Io.Dir.cwd();
    while (n < max_host_cpus) : (n += 1) {
        var path_buf: [96]u8 = undefined;
        const path = std.fmt.bufPrint(
            &path_buf,
            "/sys/devices/system/cpu/cpu{d}/topology/thread_siblings_list",
            .{n},
        ) catch unreachable;
        const text = cwd.readFile(io, path, &texts[n]) catch |err| switch (err) {
            error.FileNotFound => break,
            else => return null,
        };
        if (text.len == sibling_text_max) {
            log.info(
                "cpu{d} sibling list is longer than {d} bytes. The thread default uses the " ++
                    "logical estimate",
                .{ n, sibling_text_max },
            );
            return null;
        }

        lists[n] = text;
    } else {
        log.info(
            "the host has more than {d} CPUs. The thread default uses the logical estimate",
            .{max_host_cpus},
        );
        return null;
    }

    if (n == 0) return null;

    return countCores(lists[0..n]);
}

/// Build a Tensor view of `weight` over the row range `[start, end)` along the
/// output dimension (`shape[1]` in GGUF reversed-shape convention).
///
/// Rationale for `data.len / out_features`: the `Tensor.byte_stride` field
/// carries per-axis strides. The kernels instead compute per-output-row bytes
/// from the total data length divided by the output count (`shape[1]`). The
/// same convention here keeps the slice math consistent with the kernels, and
/// it works uniformly for f32, Q8_0, and any dtype added later.
fn sliceWeightRows(weight: Tensor, start: usize, end: usize) Tensor {
    std.debug.assert(weight.n_dims == 2);

    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(out_features > 0);
    std.debug.assert(weight.data.len % out_features == 0);
    const row_bytes: usize = weight.data.len / out_features;
    std.debug.assert(end <= out_features);
    return .{
        .dtype = weight.dtype,
        .n_dims = weight.n_dims,
        .shape = .{
            weight.shape[0],
            @intCast(end - start),
            weight.shape[2],
            weight.shape[3],
        },
        .byte_stride = weight.byte_stride,
        .data = weight.data[start * row_bytes .. end * row_bytes],
    };
}

// -- tests ------------------------------------------------------------------

test "countCores: SMT pairs, single threads, and comma lists" {
    // An SMT-2 host lists each pair twice, once per logical CPU.
    const smt = [_][]const u8{ "0-1\n", "0-1\n", "2-3\n", "2-3\n", "4-5\n", "4-5\n" };
    try std.testing.expectEqual(@as(u32, 3), countCores(&smt));
    // A host without SMT lists each CPU alone.
    const solo = [_][]const u8{ "0\n", "1\n", "2\n", "3\n" };
    try std.testing.expectEqual(@as(u32, 4), countCores(&solo));
    // A host that numbers the siblings apart writes a comma list.
    const comma = [_][]const u8{ "0,16\n", "1,17\n", "0,16\n", "1,17\n" };
    try std.testing.expectEqual(@as(u32, 2), countCores(&comma));
}

test "forHost: min(physical cores, 8), the default floor and spin" {
    const cores = hostCores(std.testing.io);
    try std.testing.expect(cores.physical >= 1);
    const policy = Policy.forHost(std.testing.io);
    try std.testing.expectEqual(@min(cores.physical, max_default_threads), policy.n_threads);
    try std.testing.expectEqual(default_min_parallel_flops, policy.min_parallel_flops);
    try std.testing.expectEqual(default_spin_ns, policy.spin_ns);
    try std.testing.expectEqual(@as(u32, 1), defaultThreads(1));
    try std.testing.expectEqual(@as(u32, 8), defaultThreads(16));
}

test "Policy.parts: every branch of the fan-out decision" {
    const policy: Policy = .{ .n_threads = 8, .min_parallel_flops = 1000 };
    // One thread runs inline at any size.
    try std.testing.expectEqual(@as(u32, 1), policy.parts(1, 64, 1_000_000));
    // One unit, or none, runs inline.
    try std.testing.expectEqual(@as(u32, 1), policy.parts(8, 1, 1_000_000));
    try std.testing.expectEqual(@as(u32, 1), policy.parts(8, 0, 1_000_000));
    // Under the floor runs inline, and at the floor fans out.
    try std.testing.expectEqual(@as(u32, 1), policy.parts(8, 64, 999));
    try std.testing.expectEqual(@as(u32, 8), policy.parts(8, 64, 1000));
    // Fewer units than threads caps the participants at the units.
    try std.testing.expectEqual(@as(u32, 3), policy.parts(8, 3, 1_000_000));
    // More units than threads takes every thread.
    try std.testing.expectEqual(@as(u32, 8), policy.parts(8, 1000, 1_000_000));
    // Floor 0 fans out any op with two units.
    const forced: Policy = .{ .n_threads = 4, .min_parallel_flops = 0 };
    try std.testing.expectEqual(@as(u32, 2), forced.parts(4, 2, 0));
}

test "resolve: zero keeps auto, a flag replaces, the bound refuses, fused gets 1" {
    const host: Policy = .{ .n_threads = 8 };
    const auto = try resolve(host, .{ .threads = 0 }, true);
    try std.testing.expectEqual(@as(u32, 8), auto.policy.n_threads);
    try std.testing.expectEqual(Source.auto, auto.threads_from);
    const flagged = try resolve(host, .{ .threads = 3, .min_parallel_flops = 7 }, true);
    try std.testing.expectEqual(@as(u32, 3), flagged.policy.n_threads);
    try std.testing.expectEqual(@as(u64, 7), flagged.policy.min_parallel_flops);
    try std.testing.expectEqual(Source.flag, flagged.flops_from);
    try std.testing.expectEqual(Source.auto, flagged.spin_from);
    try std.testing.expectEqual(default_spin_ns, flagged.policy.spin_ns);
    const spun = try resolve(host, .{ .spin_us = 200 }, true);
    try std.testing.expectEqual(@as(u64, 200_000), spun.policy.spin_ns);
    try std.testing.expectEqual(Source.flag, spun.spin_from);
    const no_spin = try resolve(host, .{ .spin_us = 0 }, true);
    try std.testing.expectEqual(@as(u64, 0), no_spin.policy.spin_ns);
    try std.testing.expectEqual(
        max_spin_us * std.time.ns_per_us,
        (try resolve(host, .{ .spin_us = max_spin_us }, true)).policy.spin_ns,
    );
    try std.testing.expectError(
        error.SpinTooLong,
        resolve(host, .{ .spin_us = max_spin_us + 1 }, true),
    );
    try std.testing.expectEqual(
        limits.max_threads,
        (try resolve(host, .{ .threads = limits.max_threads }, true)).policy.n_threads,
    );
    try std.testing.expectError(
        error.TooManyThreads,
        resolve(host, .{ .threads = limits.max_threads + 1 }, true),
    );
    const fused = try resolve(host, .{ .threads = 6 }, false);
    try std.testing.expectEqual(@as(u32, 1), fused.policy.n_threads);
    try std.testing.expectEqual(Source.placement, fused.threads_from);
    logResolved(fused);
}

test "sliceWeightRows preserves layout, slices data" {
    // 4 output rows x 34 bytes each (Q8_0 with in_features=32 -> 1 block of 34).
    var fake_bytes: [4 * 34]u8 = .{0} ** (4 * 34);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ 32, 4, 0, 0 }, // in=32, out=4
        .byte_stride = .{ 34, 0, 0, 0 },
        .data = &fake_bytes,
    };
    const slice = sliceWeightRows(w, 1, 3);
    try std.testing.expectEqual(.q8_0, slice.dtype);
    try std.testing.expectEqual(@as(u32, 32), slice.shape[0]);
    try std.testing.expectEqual(@as(u32, 2), slice.shape[1]);
    try std.testing.expectEqual(@as(usize, 68), slice.data.len); // 2 output rows x 34 bytes
}

/// A q8_0 matmul fixture: seeded inputs and a weight with distinct rows.
const Q80Fixture = struct {
    inputs: []f32,
    w_data: []align(32) u8,
    weight: Tensor,

    fn init(
        gpa: std.mem.Allocator,
        in_features: usize,
        out_features: usize,
        batch: usize,
    ) !Q80Fixture {
        std.debug.assert(in_features % 32 == 0);

        const row_bytes = (in_features / 32) * 34;
        const inputs = try gpa.alloc(f32, batch * in_features);
        errdefer gpa.free(inputs);

        const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * row_bytes);
        for (inputs, 0..) |*x, i| {
            x.* = @sin(@as(f32, @floatFromInt(i)) * 0.02);
        }

        for (0..out_features) |row| {
            const off = row * row_bytes;
            var blk: usize = 0;
            while (blk < in_features / 32) : (blk += 1) {
                const blk_off = off + blk * 34;
                std.mem.writeInt(u16, w_data[blk_off..][0..2], 0x3c00, .little);
                var i: usize = 0;
                while (i < 32) : (i += 1) {
                    const v: i8 = @intCast(@as(i32, @intCast((row + blk * 32 + i) % 13)) - 6);
                    w_data[blk_off + 2 + i] = @bitCast(v);
                }
            }
        }

        return .{
            .inputs = inputs,
            .w_data = w_data,
            .weight = .{
                .dtype = .q8_0,
                .n_dims = 2,
                .shape = .{ @intCast(in_features), @intCast(out_features), 0, 0 },
                .byte_stride = .{ @intCast(row_bytes), 0, 0, 0 },
                .data = w_data,
            },
        };
    }

    fn deinit(self: Q80Fixture, gpa: std.mem.Allocator) void {
        gpa.free(self.inputs);
        gpa.free(self.w_data);
    }
};

/// Run the fixture matmul serially and on a pool of each thread count with
/// floor 0, and expect the bits of the serial call every time.
fn expectFanOutMatchesSerial(out_features: usize, batch: usize, thread_counts: []const u32) !void {
    const naive = @import("../ops/naive.zig");
    const gpa = std.testing.allocator;
    const fx = try Q80Fixture.init(gpa, 64, out_features, batch);
    defer fx.deinit(gpa);

    const outs_serial = try gpa.alloc(f32, batch * out_features);
    defer gpa.free(outs_serial);

    const outs_fanout = try gpa.alloc(f32, batch * out_features);
    defer gpa.free(outs_fanout);

    naive.linearF32Q80Batch(fx.inputs, fx.weight, outs_serial, batch, out_features);
    for (thread_counts) |n_threads| {
        var pool: Pool = undefined;
        try pool.init(std.testing.io, n_threads, 50_000);
        defer pool.deinit();

        const policy: Policy = .{ .n_threads = n_threads, .min_parallel_flops = 0 };
        // Every pool above one thread splits the rows into two units or more.
        if (n_threads > 1) {
            try std.testing.expect(out_features > matmulUnitRows(out_features, n_threads));
        }

        @memset(outs_fanout, std.math.nan(f32));
        linearBatch(
            &pool,
            policy,
            naive.linearF32Q80Batch,
            fx.inputs,
            fx.weight,
            outs_fanout,
            batch,
        );
        for (outs_serial, outs_fanout) |s, f| {
            try std.testing.expectEqual(@as(u32, @bitCast(s)), @as(u32, @bitCast(f)));
        }
    }
}

test "linearBatch: bits-equal across thread counts (q8_0, forced fan-out)" {
    try expectFanOutMatchesSerial(32, 3, &.{ 1, 4 });
}

test "linearBatch: 37 rows over 3, 5, and 7 threads cut odd chunk boundaries" {
    // 37 rows give 4-row units at 3 and 5 threads and a 1-row tail unit.
    try std.testing.expectEqual(@as(usize, 4), matmulUnitRows(37, 3));
    try std.testing.expectEqual(@as(usize, 4), matmulUnitRows(37, 5));
    try std.testing.expectEqual(@as(usize, 4), matmulUnitRows(37, 7));
    try expectFanOutMatchesSerial(37, 3, &.{ 3, 5, 7 });
}

test "matmulUnitRows: about four units per thread, a multiple of 4" {
    try std.testing.expectEqual(@as(usize, 4), matmulUnitRows(1, 1));
    try std.testing.expectEqual(@as(usize, 128), matmulUnitRows(2048, 4));
    try std.testing.expectEqual(@as(usize, 64), matmulUnitRows(2048, 8));
    try std.testing.expectEqual(@as(usize, 4748), matmulUnitRows(151_936, 8));
}

test "attention: forced fan-out at 1, 2, 3, 4, and 16 threads gives the bits of the serial row" {
    const naive = @import("../ops/naive.zig");
    const attention_ref = @import("../ops/attention_ref.zig");
    const T = limits.attn_cpu_tile;
    const shapes = [_]attention_ref.AttnShape{
        .{ .n_heads = 4, .n_kv_heads = 2, .head_dim = 8 },
        .{ .n_heads = 6, .n_kv_heads = 3, .head_dim = 8 },
    };
    // One part, two parts, and 17 tiles in 16 parts. At 3 threads the units
    // do not split evenly over the participants.
    const depths = [_]usize{ 5, T + 1, 16 * T + 3 };
    var prng = std.Random.DefaultPrng.init(0xA770_00A3);
    const r = prng.random();
    inline for (.{ f32, f16 }) |E| {
        for (shapes) |s| {
            for (depths) |t_len| {
                try attention_ref.expectFanOutMatchesRow(.{naive}, E, r, s, t_len);
            }
        }
    }
}

test "gdnStep: forced fan-out at 1, 2, 3, 4, and 16 threads gives the bits of gdn.step" {
    const gpa = std.testing.allocator;
    // The 2B (16 value heads) and the 9B (32 value heads). At 3 threads the
    // heads do not split evenly over the participants.
    const shapes = [_]gdn.Dims{
        .{ .n_k_heads = 16, .n_v_heads = 16, .head_dim = 128, .conv_kernel = 4 },
        .{ .n_k_heads = 16, .n_v_heads = 32, .head_dim = 128, .conv_kernel = 4 },
    };
    const thread_counts = [_]u32{ 1, 2, 3, 4, 16 };
    const tokens = 2;
    var prng = std.Random.DefaultPrng.init(0x6D17_00A5);
    const r = prng.random();
    for (shapes) |d| {
        const cd = d.convDim();
        const vd = d.valueDim();
        const nv = d.n_v_heads;
        const inputs = try gpa.alloc(
            f32,
            tokens * cd + vd + 4 * nv + d.conv_kernel * cd + d.head_dim,
        );
        defer gpa.free(inputs);

        for (inputs) |*x| {
            x.* = r.float(f32) * 2.0 - 1.0;
        }

        const qkv_in = inputs[0 .. tokens * cd];
        var at: usize = tokens * cd;
        const z = inputs[at..][0..vd];
        at += vd;
        const a = inputs[at..][0..nv];
        const b = inputs[at + nv ..][0..nv];
        const ssm_a = inputs[at + 2 * nv ..][0..nv];
        const dt_bias = inputs[at + 3 * nv ..][0..nv];
        at += 4 * nv;
        const conv_w = inputs[at..][0 .. d.conv_kernel * cd];
        at += d.conv_kernel * cd;
        const ssm_norm = inputs[at..][0..d.head_dim];
        for (ssm_a) |*x| {
            x.* = -@abs(x.*) - 0.1;
        }

        const state0 = try gpa.alloc(f32, d.stateLen());
        defer gpa.free(state0);

        for (state0) |*x| {
            x.* = (r.float(f32) - 0.5) * 0.2;
        }

        const conv0 = try gpa.alloc(f32, d.convStateLen());
        defer gpa.free(conv0);

        for (conv0) |*x| {
            x.* = r.float(f32) - 0.5;
        }

        // The serial reference: `gdn.step` over the tokens.
        const state_ref = try gpa.dupe(f32, state0);
        defer gpa.free(state_ref);

        const conv_ref = try gpa.dupe(f32, conv0);
        defer gpa.free(conv_ref);

        const qkv_ref = try gpa.dupe(f32, qkv_in);
        defer gpa.free(qkv_ref);

        const o_ref = try gpa.alloc(f32, tokens * vd);
        defer gpa.free(o_ref);

        for (0..tokens) |t| {
            gdn.step(
                d,
                qkv_ref[t * cd ..][0..cd],
                z,
                a,
                b,
                conv_w,
                ssm_a,
                dt_bias,
                ssm_norm,
                1e-6,
                state_ref,
                conv_ref,
                o_ref[t * vd ..][0..vd],
            );
        }

        const state = try gpa.alloc(f32, d.stateLen());
        defer gpa.free(state);

        const conv = try gpa.alloc(f32, d.convStateLen());
        defer gpa.free(conv);

        const qkv = try gpa.alloc(f32, tokens * cd);
        defer gpa.free(qkv);

        const o = try gpa.alloc(f32, tokens * vd);
        defer gpa.free(o);

        for (thread_counts) |n_threads| {
            @memcpy(state, state0);
            @memcpy(conv, conv0);
            @memcpy(qkv, qkv_in);
            @memset(o, std.math.nan(f32));
            var pool: Pool = undefined;
            try pool.init(std.testing.io, n_threads, 50_000);
            defer pool.deinit();

            const policy: Policy = .{ .n_threads = n_threads, .min_parallel_flops = 0 };
            for (0..tokens) |t| {
                const qkv_row = qkv[t * cd ..][0..cd];
                gdnStep(&pool, policy, d, qkv_row, conv_w, conv, .{
                    .qkv = qkv_row,
                    .z = z,
                    .a = a,
                    .b = b,
                    .ssm_a = ssm_a,
                    .dt_bias = dt_bias,
                    .ssm_norm = ssm_norm,
                    .rms_eps = 1e-6,
                    .state = state,
                    .o_out = o[t * vd ..][0..vd],
                });
            }

            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(o_ref),
                std.mem.sliceAsBytes(o),
            );
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(state_ref),
                std.mem.sliceAsBytes(state),
            );
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(conv_ref),
                std.mem.sliceAsBytes(conv),
            );
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(qkv_ref),
                std.mem.sliceAsBytes(qkv),
            );
        }
    }
}
