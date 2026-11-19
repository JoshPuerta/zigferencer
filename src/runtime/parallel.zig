//! Output-parallel matmul dispatcher over the fork-join `Pool`. It wraps a linear-kernel function
//! pointer. When serial (one pool thread), below the FLOPs floor, or with fewer than two units, it
//! calls the kernel directly, at zero overhead: no pool touch, no alloc, no sync. Otherwise it
//! splits the outputs into units and runs them on the pool.
//!
//! Comptime. `E` and `Kernels` make the fan-out generic, so f32 and f16 share one body.
//!
//! The module also holds the control units of CPU attention, of the GDN step, and of `swiglu`. They
//! spread their work units over the same pool, with the same gate, `Policy.parts`.
//!
//! Determinism. The dispatcher is bit-deterministic across thread counts. Exactly one unit
//! computes each output in the reduction order of the kernel. The bench token hash therefore
//! holds across thread counts. `--threads N`, `--min-parallel-flops F`, and `--spin-us N` override
//! the defaults.
//!
//! Fan-out gate. An op runs inline when the pool has one thread, when it has fewer than two units,
//! or when its FLOPs fall under `min_parallel_flops`. Otherwise `min(threads, n_units)`
//! participants claim the units. Every fan-out decision comes through `Policy.parts`, so a new
//! rule, such as a cap for each bound of an op or a floor for each op class, goes there.
//!
//! Defaults. One floor serves the matmuls, the attention, the Gated-DeltaNet heads, and `swiglu`. A
//! sweep of the floor measured 250 kFLOP best on a small model and on a 2 GB model. A dispatch on
//! the pool costs about half a microsecond at 8 participants, so an op of this size gains from a
//! split. A worker that sleeps adds a futex wake to the next dispatch, and a 50 us spin covers the
//! gap between the matmuls of one decode step. A longer spin keeps idle workers on the cores and
//! gives no gain, so `--spin-us` stops at one second. Decode saturates the memory bandwidth of a
//! dual-channel desktop between 4 and 8 threads, and prefill keeps gaining to 8. The default thread
//! count stops at 8 to keep sustained load off the rest of the cores. `--threads N` raises it.
//!
//! Matmul units. The partition stays over outputs, not the batch. Each unit streams a disjoint
//! weight chunk once for all rows, which is bandwidth-optimal. A batch partition streams the full
//! weight per thread, the traffic that batching exists to remove. A unit is a chunk of
//! `matmulUnitOutputs` outputs, about four units per participant, so a participant that finishes
//! early claims more. The chunk size is a function of the shape and the pool size. It never
//! changes a bit, because the kernels give an output the same bits at every chunk boundary.
//!
//! Attention units. `attentionRowsTyped` makes one fork for all attention rows of a step. The
//! geometry of `attn_cpu` splits a row into `c = attn_cpu.partCount(t_len)` parts. A unit is one KV
//! group of one row over one part, so a row holds `n_kv_heads * c` units. The units of a row follow
//! the units of the row before it. Inside a row, unit `u` is group `u / c` and part `u % c`. A unit
//! writes the records of its group heads for its part, inside the records of its row, so no two
//! units write one record. After every unit ends, `Kernels.attentionMerge` combines the records of
//! each row on the calling thread, adding the parts in ascending order. The geometry is a function
//! of `t_len` alone, so any assignment of units to participants gives the bits of the serial row,
//! `Kernels.attentionRowTyped`.
//!
//! GDN units. `gdnRows` makes one fork for all GDN rows of a step. A unit is one key head, and it
//! steps every row in row order with `Kernels.gdnStepKeyHead`. For each row it runs the conv, the
//! SiLU, and the norms of its q and k channels, then the conv and the recurrence of its value
//! heads. Each channel, each conv ring entry, and each state belongs to one key head. The rows of
//! one slot are in time order in a step, and one unit keeps that order. So any assignment of units
//! to participants gives the bits of the serial step of the backend, row by row: `gdn.step` for
//! naive, and `gdn_simd.kernels.step` for the SIMD backends. The finite check runs after every unit
//! ends.
//!
//! Swiglu units. The op is elementwise, so a unit is a range of `elementwiseUnitLen` elements of
//! the flat gate buffer. Any partition gives the bits of the serial op.
//!
//! Weight slices. `Tensor.byte_stride` carries per-axis strides, but the kernels compute the bytes
//! of one output as the data length divided by the output count (`shape[1]`). The slice math
//! uses the same rule, so it works for f32, Q8_0, and any dtype added later.
//!
//! To fan out a new op on the pool, do these steps:
//!
//! 1. Divide the op into units. The result of a unit must not change with the worker that runs it.
//! 2. Write a context value and a unit function for `Pool.run`.
//! 3. Count the FLOPs of the op.
//! 4. Get the participant count from `Policy.parts`. If it is 1, run the units inline on the
//!    caller.
//! 5. Add a test that the fan-out at several thread counts gives the bits of the serial path.

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

/// The default fan-out floor in FLOPs.
pub const default_min_parallel_flops: u64 = 250_000;

/// The default spin budget of a pool worker.
pub const default_spin_ns: u64 = 50_000;

/// The cap on `--spin-us`, one second.
pub const max_spin_us: u64 = 1_000_000;

/// The cap on the default thread count. `--threads N` raises it.
pub const max_default_threads: u32 = 8;

/// The cap on the logical CPUs that `hostCores` reads. A host with more CPUs logs the cause.
const max_host_cpus: u32 = 512;

/// The longest `thread_siblings_list` text that `hostCores` keeps.
const sibling_text_max: usize = 48;

// -- types ------------------------------------------------------------------

/// A batched linear kernel: `out = input @ weight.T` with weight in `[in, out]` (GGUF row-major).
/// Row b reads `inputs[b*in ..]` and writes `outs[b*out_stride ..][0..out]`.
/// `out_stride` lets a kernel that gets a slice of the outputs still write full-width rows.
pub const LinearBatchKernel = *const fn (
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void;

/// The threading policy of one process: the pool size and the gates for the pool.
/// `forHost` gives the defaults, and the command flags replace single fields through `resolve`.
pub const Policy = struct {
    /// The pool size, the calling thread included. 1 (or 0) = single-threaded.
    n_threads: u32 = 1,
    /// An op with fewer FLOPs than this runs inline on the caller.
    min_parallel_flops: u64 = default_min_parallel_flops,
    /// The time a pool worker spins on the epoch word before it sleeps.
    spin_ns: u64 = default_spin_ns,

    /// The default policy of the host: `defaultThreads` of the physical cores, the default floor,
    /// and the default spin budget.
    pub fn forHost(io: std.Io) Policy {
        const policy: Policy = .{ .n_threads = defaultThreads(hostCores(io).physical) };
        std.debug.assert(policy.n_threads >= 1);
        std.debug.assert(policy.n_threads <= max_default_threads);
        return policy;
    }

    /// The participant count of one op on a pool of `threads` threads. 1 runs the op inline on
    /// the caller.
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
    /// True for the sysfs topology, false for the estimate of half the logical CPUs.
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

/// The inputs of one matmul. Unit `u` covers the outputs
/// `[u * outputs_per_unit, min(n, (u + 1) * outputs_per_unit))`.
const MatmulUnits = struct {
    kernel: LinearBatchKernel,
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    n: usize,
    outputs_per_unit: usize,

    fn unit(self: *const MatmulUnits, u: u32) void {
        const start = @as(usize, u) * self.outputs_per_unit;
        std.debug.assert(start < self.n);
        const end = @min(self.n, start + self.outputs_per_unit);
        const slice = sliceWeightOutputs(self.weight, start, end);
        self.kernel(self.inputs, slice, self.outs[start..], self.batch, self.n);
    }
};

/// The inputs of one `swiglu` call. Unit `u` covers the elements
/// `[u * len, min(n, (u + 1) * len))`.
fn SwigluUnits(comptime Kernels: type) type {
    return struct {
        gate: []const f32,
        up: []const f32,
        dst: []f32,
        len: usize,
        n: usize,

        const Self = @This();

        fn unit(self: *const Self, u: u32) void {
            const start = @as(usize, u) * self.len;
            std.debug.assert(start < self.n);
            const end = @min(self.n, start + self.len);
            Kernels.swiglu(self.gate[start..end], self.up[start..end], self.dst[start..end]);
        }
    };
}

/// The inputs of one attention row of a step, over the part records of that row.
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
            // The part kernel finds head h at `(h - h_lo) * rec_stride` past head h_lo.
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
    };
}

/// The inputs of one attention row of a step.
pub fn AttentionRow(comptime E: type) type {
    return struct {
        q: []const f32,
        k_cache: []const E,
        v_cache: []const E,
        t_len: u32,
        out: []f32,
    };
}

/// The inputs of the attention rows of one step. The pool erases the context type, so this type
/// binds `E` and the kernels. The units of row `i` are `[unit_ends[i - 1], unit_ends[i])`, in the
/// order of `Units` for that row.
fn AttentionRowsUnits(comptime E: type, comptime Kernels: type) type {
    return struct {
        rows: []const AttentionRow(E),
        unit_ends: []const u32,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
        parts_buf: []f32,
        /// The elements of the part records of one row.
        row_parts_len: usize,

        const Self = @This();

        fn unit(self: *const Self, u: u32) void {
            const i = self.rowOf(u);
            const unit_lo: u32 = if (i == 0) 0 else self.unit_ends[i - 1];
            std.debug.assert(u >= unit_lo);
            std.debug.assert(u < self.unit_ends[i]);

            const units = self.rowUnits(i);
            const local: usize = u - unit_lo;
            units.run(local, local + 1);
        }

        /// The row that owns unit `u`: the first row whose units end past `u`.
        fn rowOf(self: *const Self, u: u32) usize {
            std.debug.assert(u < self.unit_ends[self.unit_ends.len - 1]);

            var lo: usize = 0;
            var hi: usize = self.unit_ends.len - 1;
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (self.unit_ends[mid] > u) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            std.debug.assert(self.unit_ends[lo] > u);
            return lo;
        }

        /// The unit context of row `i`, over its own part records.
        fn rowUnits(self: *const Self, i: usize) Units(E, Kernels) {
            const r = self.rows[i];
            return .{
                .q = r.q,
                .n_heads = self.n_heads,
                .head_dim = self.head_dim,
                .k_cache = r.k_cache,
                .v_cache = r.v_cache,
                .n_kv_heads = self.n_kv_heads,
                .t_len = r.t_len,
                .n_parts = attn_cpu.partCount(r.t_len),
                .parts_buf = self.parts_buf[i * self.row_parts_len ..][0..self.row_parts_len],
            };
        }
    };
}

/// The weights of one GDN layer and the epsilon of its gated norm, shared by every row.
pub const GdnWeights = struct {
    conv_w: []const f32,
    ssm_a: []const f32,
    dt_bias: []const f32,
    ssm_norm: []const f32,
    rms_eps: f32,
};

/// The inputs of one GDN row of a step: its stripes and the slabs of its slot.
pub const GdnRow = struct {
    qkv: []f32,
    z: []const f32,
    a: []const f32,
    b: []const f32,
    state: []f32,
    conv_state: []f32,
    o_out: []f32,
};

/// The inputs of the GDN rows of one step. Unit `u` is key head `u` over every row, in row order.
fn GdnRowsUnits(comptime Kernels: type) type {
    return struct {
        d: gdn.Dims,
        rows: []const GdnRow,
        w: GdnWeights,

        const Self = @This();

        fn unit(self: *const Self, u: u32) void {
            std.debug.assert(u < self.d.n_k_heads);

            self.keyHead(u);
        }

        /// Step key head `kh` over every row in row order. The rows of one slot are in time
        /// order, and a key head owns its channels, its ring entries, and its value heads.
        fn keyHead(self: *const Self, kh: usize) void {
            for (self.rows) |r| {
                Kernels.gdnStepKeyHead(self.d, r.qkv, self.w.conv_w, r.conv_state, .{
                    .qkv = r.qkv,
                    .z = r.z,
                    .a = r.a,
                    .b = r.b,
                    .ssm_a = self.w.ssm_a,
                    .dt_bias = self.w.dt_bias,
                    .ssm_norm = self.w.ssm_norm,
                    .rms_eps = self.w.rms_eps,
                    .state = r.state,
                    .o_out = r.o_out,
                }, kh);
            }
        }
    };
}

// -- api --------------------------------------------------------------------

/// Start `pool` with the threads and the spin budget of `policy`.
/// A policy over `limits.max_threads` threads returns `error.TooManyThreads`.
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

/// Apply the flags of a command to the host policy. A placement that runs no CPU op gets one
/// thread. A `--threads` value over `limits.max_threads` returns `error.TooManyThreads`.
/// A `--spin-us` value over `max_spin_us` returns `error.SpinTooLong`.
pub fn resolve(host: Policy, flags: Flags, runs_cpu_ops: bool) ResolveError!Resolved {
    // Start from the host policy.
    std.debug.assert(host.n_threads >= 1);

    var out: Resolved = .{
        .policy = host,
        .threads_from = .auto,
        .flops_from = .auto,
        .spin_from = .auto,
    };

    // Apply the flags.
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

    // Pin one thread when no CPU op runs.
    if (!runs_cpu_ops) {
        out.policy.n_threads = 1;
        out.threads_from = .placement;
    }

    // Check the result.
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

/// The physical cores of the host: the distinct `thread_siblings_list` texts in sysfs, because
/// SMT siblings share one text. Without a readable topology, half the logical CPUs (SMT-2).
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

/// The number of distinct sibling-list texts in `lists`, with surrounding white space ignored.
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

/// Run a linear matmul over `batch` input rows, on the pool when the work is large enough.
/// The serial path is a direct kernel call.
/// `outs` must hold `batch` rows at stride `weight.shape[1]`.
pub fn linearBatch(
    pool: *Pool,
    policy: Policy,
    kernel: LinearBatchKernel,
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
) void {
    // Check the shapes.
    const k: usize = @intCast(weight.shape[0]);
    const n: usize = @intCast(weight.shape[1]);
    std.debug.assert(batch > 0);
    std.debug.assert(n > 0);
    std.debug.assert(inputs.len == batch * k);
    std.debug.assert(outs.len >= batch * n);

    // Size the units and pick the participants.
    const threads = pool.threads();
    std.debug.assert(@max(policy.n_threads, 1) == threads);
    const flops: u64 = 2 * @as(u64, k) * @as(u64, n) * @as(u64, batch);
    const outputs_per_unit = matmulUnitOutputs(n, threads);
    const n_units = std.math.divCeil(usize, n, outputs_per_unit) catch unreachable;

    // Call the kernel directly when one participant is enough.
    const parts = policy.parts(threads, n_units, flops);
    if (parts == 1) {
        log.debug(
            "linearBatch serial: flops {d}, floor {d}, n {d}, batch {d}",
            .{ flops, policy.min_parallel_flops, n, batch },
        );
        kernel(inputs, weight, outs, batch, n);
        return;
    }

    // Run the output chunks on the pool.
    log.debug(
        "linearBatch fanout: {d} parts, {d} units of {d} outputs over n {d}, batch {d}, flops {d}",
        .{ parts, n_units, outputs_per_unit, n, batch, flops },
    );
    const ctx: MatmulUnits = .{
        .kernel = kernel,
        .inputs = inputs,
        .weight = weight,
        .outs = outs,
        .batch = batch,
        .n = n,
        .outputs_per_unit = outputs_per_unit,
    };
    pool.run(MatmulUnits, &ctx, MatmulUnits.unit, @intCast(n_units), parts);
}

/// The outputs of one matmul unit: `ceil(n / (4 * threads))` rounded up to a multiple of 4.
pub fn matmulUnitOutputs(n: usize, threads: u32) usize {
    std.debug.assert(n > 0);
    std.debug.assert(threads >= 1);

    const per_unit = std.math.divCeil(usize, n, 4 * @as(usize, threads)) catch unreachable;
    const outputs = std.mem.alignForward(usize, per_unit, 4);
    std.debug.assert(outputs >= 4);
    std.debug.assert(outputs % 4 == 0);
    return outputs;
}

/// One fork over the attention units of every row of a step. Each row gets the bits of the serial
/// row `Kernels.attentionRowTyped`. `parts_buf` holds `attn_cpu.partialLen` f32s for each row. The
/// caller stores the K and V of every row first. `E` is the KV element type.
pub fn attentionRowsTyped(
    comptime E: type,
    comptime Kernels: type,
    pool: *Pool,
    policy: Policy,
    rows: []const AttentionRow(E),
    n_heads: usize,
    head_dim: usize,
    n_kv_heads: usize,
    parts_buf: []f32,
) void {
    // Check the shapes and buffers.
    comptime std.debug.assert(E == f32 or E == f16);
    std.debug.assert(rows.len >= 1);
    std.debug.assert(rows.len <= limits.step_rows_max);
    std.debug.assert(n_kv_heads >= 1);
    std.debug.assert(n_heads >= n_kv_heads);
    std.debug.assert(n_heads % n_kv_heads == 0);
    std.debug.assert(head_dim >= 1);
    const row_parts_len = attn_cpu.partialLen(@intCast(n_heads), @intCast(head_dim));
    std.debug.assert(parts_buf.len >= rows.len * row_parts_len);
    const threads = pool.threads();
    std.debug.assert(@max(policy.n_threads, 1) == threads);

    // Count the units of each row and the FLOPs of the step.
    var unit_ends_buf: [limits.step_rows_max]u32 = undefined;
    const unit_ends = unit_ends_buf[0..rows.len];
    var units_sum: usize = 0;
    var flops: u64 = 0;
    for (rows, unit_ends) |r, *end| {
        std.debug.assert(r.t_len >= 1);
        std.debug.assert(r.t_len <= limits.max_model_ctx);
        std.debug.assert(r.q.len == n_heads * head_dim);
        std.debug.assert(r.out.len == n_heads * head_dim);
        std.debug.assert(r.k_cache.len >= @as(usize, r.t_len) * n_kv_heads * head_dim);
        std.debug.assert(r.v_cache.len >= @as(usize, r.t_len) * n_kv_heads * head_dim);

        units_sum += n_kv_heads * attn_cpu.partCount(r.t_len);
        end.* = @intCast(units_sum);
        // 4 FLOP per (head, position, dim): the score dot plus the weighted-V sum.
        flops += 4 * @as(u64, n_heads) * @as(u64, head_dim) * r.t_len;
    }
    std.debug.assert(units_sum <= limits.step_rows_max * n_kv_heads * limits.attn_cpu_parts);
    std.debug.assert(units_sum <= std.math.maxInt(u32));
    const n_units: u32 = @intCast(units_sum);

    const units: AttentionRowsUnits(E, Kernels) = .{
        .rows = rows,
        .unit_ends = unit_ends,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .n_kv_heads = n_kv_heads,
        .parts_buf = parts_buf,
        .row_parts_len = row_parts_len,
    };

    // Run the units serially or on the pool.
    const parts = policy.parts(threads, n_units, flops);
    if (parts == 1) {
        log.debug(
            "attention rows serial: flops {d}, floor {d}, rows {d}, units {d}",
            .{ flops, policy.min_parallel_flops, rows.len, n_units },
        );
        for (0..rows.len) |i| {
            const row_units = units.rowUnits(i);
            row_units.run(0, n_kv_heads * row_units.n_parts);
        }
    } else {
        log.debug(
            "attention rows fanout: {d} parts over {d} units, rows {d}",
            .{ parts, n_units, rows.len },
        );
        const Ctx = AttentionRowsUnits(E, Kernels);
        pool.run(Ctx, &units, Ctx.unit, n_units, parts);
    }

    // Merge the part records of each row on the calling thread.
    for (rows, 0..) |r, i| {
        const row_units = units.rowUnits(i);
        Kernels.attentionMerge(row_units.parts_buf, row_units.n_parts, n_heads, head_dim, r.out);
    }
}

/// The elements of one `swiglu` unit: `ceil(n / (4 * threads))` rounded up to a multiple of 16.
/// 16 f32 fill one 64-byte cache line, so two units never share a line of an aligned buffer.
pub fn elementwiseUnitLen(n: usize, threads: u32) usize {
    std.debug.assert(n > 0);
    std.debug.assert(threads >= 1);

    const per_unit = std.math.divCeil(usize, n, 4 * @as(usize, threads)) catch unreachable;
    const len = std.mem.alignForward(usize, per_unit, 16);
    std.debug.assert(len >= 16);
    std.debug.assert(len % 16 == 0);
    return len;
}

/// Run `dst = silu(gate) * up` over the flat buffers, on the pool when the work is large enough.
/// `dst` can be `gate`. `Kernels` declares `swiglu` and `swiglu_flops_per_element`.
pub fn swiglu(
    comptime Kernels: type,
    pool: *Pool,
    policy: Policy,
    gate: []const f32,
    up: []const f32,
    dst: []f32,
) void {
    const n = gate.len;
    std.debug.assert(n > 0);
    std.debug.assert(up.len == n);
    std.debug.assert(dst.len >= n);
    const threads = pool.threads();
    std.debug.assert(@max(policy.n_threads, 1) == threads);

    // Size the units and pick the participants.
    const len = elementwiseUnitLen(n, threads);
    const n_units = std.math.divCeil(usize, n, len) catch unreachable;
    const flops: u64 = @as(u64, n) * Kernels.swiglu_flops_per_element;
    const parts = policy.parts(threads, n_units, flops);
    if (parts == 1) {
        log.debug(
            "swiglu serial: flops {d}, floor {d}, n {d}",
            .{ flops, policy.min_parallel_flops, n },
        );
        Kernels.swiglu(gate, up, dst);
        return;
    }

    log.debug(
        "swiglu fanout: {d} parts, {d} units of {d} elements over n {d}",
        .{ parts, n_units, len, n },
    );
    const Ctx = SwigluUnits(Kernels);
    const ctx: Ctx = .{ .gate = gate, .up = up, .dst = dst, .len = len, .n = n };
    pool.run(Ctx, &ctx, Ctx.unit, @intCast(n_units), parts);
}

/// One fork over the key heads of the GDN rows of a step, each head over every row in row order.
/// Each row gets the bits of the serial step of the backend. Each row holds the slabs of its own
/// slot, and the rows of one slot must be in time order.
pub fn gdnRows(
    comptime Kernels: type,
    pool: *Pool,
    policy: Policy,
    d: gdn.Dims,
    rows: []const GdnRow,
    w: GdnWeights,
) void {
    std.debug.assert(rows.len >= 1);
    std.debug.assert(rows.len <= limits.step_rows_max);
    std.debug.assert(d.n_k_heads >= 1);
    std.debug.assert(d.n_v_heads >= 1);
    for (rows) |r| {
        std.debug.assert(r.qkv.len == d.convDim());
        std.debug.assert(r.o_out.len == d.valueDim());
        std.debug.assert(r.state.len == d.stateLen());
        std.debug.assert(r.conv_state.len == d.convStateLen());
    }

    const threads = pool.threads();
    std.debug.assert(@max(policy.n_threads, 1) == threads);
    // 7 FLOP per state element of each row: decay 1, key read 2, rank-1 write 2, query read 2.
    const flops: u64 = 7 * @as(u64, d.n_v_heads) * d.head_dim * d.head_dim * rows.len;

    const Ctx = GdnRowsUnits(Kernels);
    const ctx: Ctx = .{
        .d = d,
        .rows = rows,
        .w = w,
    };

    // Run the key heads serially or on the pool.
    const n_units: u32 = @intCast(d.n_k_heads);
    const parts = policy.parts(threads, n_units, flops);
    if (parts == 1) {
        log.debug(
            "gdn rows serial: flops {d}, floor {d}, rows {d}",
            .{ flops, policy.min_parallel_flops, rows.len },
        );
        for (0..d.n_k_heads) |kh| {
            ctx.keyHead(kh);
        }
    } else {
        log.debug(
            "gdn rows fanout: {d} parts over {d} key heads, rows {d}, flops {d}",
            .{ parts, n_units, rows.len, flops },
        );
        pool.run(Ctx, &ctx, Ctx.unit, n_units, parts);
    }

    // Check the readout of each row after every head ends.
    for (rows) |r| {
        gdn.assertFinite(d, r.o_out);
    }
}

// -- helpers ----------------------------------------------------------------

/// Count the physical cores from the sibling list of each logical CPU in sysfs. It returns null
/// when no CPU has a topology file, when a read fails, or when a cap cuts the read, which logs.
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
                "cpu{d} sibling list is longer than {d} bytes. " ++
                    "The thread default uses the logical estimate",
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

/// Build a Tensor view of `weight` over the outputs `[start, end)` (`shape[1]` in GGUF
/// reversed-shape order).
fn sliceWeightOutputs(weight: Tensor, start: usize, end: usize) Tensor {
    std.debug.assert(weight.n_dims == 2);

    const out_features: usize = @intCast(weight.shape[1]);
    std.debug.assert(out_features > 0);
    std.debug.assert(weight.data.len % out_features == 0);
    const output_bytes: usize = weight.data.len / out_features;
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
        .data = weight.data[start * output_bytes .. end * output_bytes],
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

test "sliceWeightOutputs preserves layout, slices data" {
    // 4 outputs x 34 bytes each (Q8_0 with in_features=32 -> 1 block of 34).
    var fake_bytes: [4 * 34]u8 = .{0} ** (4 * 34);
    const w: Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ 32, 4, 0, 0 }, // in=32, out=4
        .byte_stride = .{ 34, 0, 0, 0 },
        .data = &fake_bytes,
    };
    const slice = sliceWeightOutputs(w, 1, 3);
    try std.testing.expectEqual(.q8_0, slice.dtype);
    try std.testing.expectEqual(@as(u32, 32), slice.shape[0]);
    try std.testing.expectEqual(@as(u32, 2), slice.shape[1]);
    try std.testing.expectEqual(@as(usize, 68), slice.data.len); // 2 outputs x 34 bytes
}

/// A q8_0 matmul fixture: seeded inputs and a weight with distinct outputs.
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

        // Allocate the inputs and the weights.
        const output_bytes = (in_features / 32) * 34;
        const inputs = try gpa.alloc(f32, batch * in_features);
        errdefer gpa.free(inputs);

        const w_data = try gpa.alignedAlloc(u8, .@"32", out_features * output_bytes);

        // Fill the inputs and the quants of each output.
        for (inputs, 0..) |*x, i| {
            x.* = @sin(@as(f32, @floatFromInt(i)) * 0.02);
        }

        for (0..out_features) |j| {
            const off = j * output_bytes;
            for (0..in_features / 32) |blk| {
                const blk_off = off + blk * 34;
                std.mem.writeInt(u16, w_data[blk_off..][0..2], 0x3c00, .little);
                for (0..32) |i| {
                    const v: i8 = @intCast(@as(i32, @intCast((j + blk * 32 + i) % 13)) - 6);
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
                .byte_stride = .{ @intCast(output_bytes), 0, 0, 0 },
                .data = w_data,
            },
        };
    }

    fn deinit(self: Q80Fixture, gpa: std.mem.Allocator) void {
        gpa.free(self.inputs);
        gpa.free(self.w_data);
    }
};

/// Run the fixture matmul serially and on a pool of each thread count with floor 0, and expect
/// the bits of the serial call every time.
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
        // Every pool above one thread splits the outputs into two units or more.
        if (n_threads > 1) {
            try std.testing.expect(out_features > matmulUnitOutputs(out_features, n_threads));
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

test "linearBatch: 37 outputs over 3, 5, and 7 threads cut odd chunk boundaries" {
    // 37 outputs give 4-output units at 3 and 5 threads and a 1-output tail unit.
    try std.testing.expectEqual(@as(usize, 4), matmulUnitOutputs(37, 3));
    try std.testing.expectEqual(@as(usize, 4), matmulUnitOutputs(37, 5));
    try std.testing.expectEqual(@as(usize, 4), matmulUnitOutputs(37, 7));
    try expectFanOutMatchesSerial(37, 3, &.{ 3, 5, 7 });
}

test "matmulUnitOutputs: about four units per thread, a multiple of 4" {
    try std.testing.expectEqual(@as(usize, 4), matmulUnitOutputs(1, 1));
    try std.testing.expectEqual(@as(usize, 128), matmulUnitOutputs(2048, 4));
    try std.testing.expectEqual(@as(usize, 64), matmulUnitOutputs(2048, 8));
    try std.testing.expectEqual(@as(usize, 4748), matmulUnitOutputs(151_936, 8));
}

test "attention: forced fan-out at 1, 2, 3, 4, and 16 threads gives the bits of the serial row" {
    const naive = @import("../ops/naive.zig");
    const attention_ref = @import("../ops/attention_ref.zig");
    const T = limits.attn_cpu_tile;
    const shapes = [_]attention_ref.AttnShape{
        .{ .n_heads = 4, .n_kv_heads = 2, .head_dim = 8 },
        .{ .n_heads = 6, .n_kv_heads = 3, .head_dim = 8 },
    };
    // One part, two parts, and 17 tiles in 16 parts. At 3 threads the units split unevenly.
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

test "elementwiseUnitLen: about four units per thread, a multiple of 16" {
    try std.testing.expectEqual(@as(usize, 16), elementwiseUnitLen(1, 1));
    try std.testing.expectEqual(@as(usize, 16), elementwiseUnitLen(64, 1));
    try std.testing.expectEqual(@as(usize, 32), elementwiseUnitLen(65, 1));
    try std.testing.expectEqual(@as(usize, 1536), elementwiseUnitLen(49152, 8));
    try std.testing.expectEqual(@as(usize, 16), elementwiseUnitLen(1000, 16));
}

test "swiglu: forced fan-out at 1, 2, 3, 4, and 16 threads gives the bits of the serial op" {
    // A unit holds a multiple of 16 elements, so each element keeps its lane. The tail of the
    // last unit is the tail of the serial call.
    inline for (.{ @import("../ops/naive.zig"), @import("../ops/simd_pf.zig") }) |Kernels| {
        try checkSwigluFanout(Kernels);
    }
}

fn checkSwigluFanout(comptime Kernels: type) !void {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xd34d_b33f);
    const r = prng.random();
    for ([_]usize{ 1, 17, 6144 * 3 + 5 }) |n| {
        const gate = try gpa.alloc(f32, n);
        defer gpa.free(gate);

        const up = try gpa.alloc(f32, n);
        defer gpa.free(up);

        for (gate, up) |*g, *u| {
            g.* = r.float(f32) * 8.0 - 4.0;
            u.* = r.float(f32) * 8.0 - 4.0;
        }

        const want = try gpa.alloc(f32, n);
        defer gpa.free(want);

        Kernels.swiglu(gate, up, want);

        const got = try gpa.alloc(f32, n);
        defer gpa.free(got);

        const in_place = try gpa.alloc(f32, n);
        defer gpa.free(in_place);

        for ([_]u32{ 1, 2, 3, 4, 16 }) |n_threads| {
            var pool: Pool = undefined;
            try pool.init(std.testing.io, n_threads, 50_000);
            defer pool.deinit();

            const policy: Policy = .{ .n_threads = n_threads, .min_parallel_flops = 0 };
            @memset(got, std.math.nan(f32));
            swiglu(Kernels, &pool, policy, gate, up, got);
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(want),
                std.mem.sliceAsBytes(got),
            );

            @memcpy(in_place, gate);
            swiglu(Kernels, &pool, policy, in_place, up, in_place);
            try std.testing.expectEqualSlices(
                u8,
                std.mem.sliceAsBytes(want),
                std.mem.sliceAsBytes(in_place),
            );
        }
    }
}

/// The row counts and the thread counts of the step fan-out tests. The last count is the CPU
/// prefill chunk of `run` and `bench-model`.
const step_case_rows = [_]usize{ 1, 2, 5, limits.cpu_prefill_chunk };
const step_case_threads = [_]u32{ 1, 2, 3, 4, 16 };

const AttnShape = @import("../ops/attention_ref.zig").AttnShape;

/// The slot of row `i` in a step of two slots: every third row is slot 1. The runner keeps the
/// rows of one slot adjacent. These rows mix the slots, a stricter case for the disjoint writes.
fn stepCaseSlot(i: usize) u32 {
    return if (i % 3 == 1) 1 else 0;
}

test "attentionRowsTyped: one fork per step gives the bits of the serial row kernel" {
    const naive = @import("../ops/naive.zig");
    const simd_pf = @import("../ops/simd_pf.zig");
    const tile = limits.attn_cpu_tile;
    const shapes = [_]AttnShape{
        .{ .n_heads = 4, .n_kv_heads = 2, .head_dim = 8 },
        .{ .n_heads = 6, .n_kv_heads = 3, .head_dim = 16 },
    };
    // At the prefill chunk, slot 0 crosses from 1 part to 2, and slot 1 from 3 parts to 4.
    const starts = [2]u32{ tile - 20, 3 * tile - 10 };
    const chunk = limits.cpu_prefill_chunk;
    const slot1_rows = (chunk + 1) / 3;
    try std.testing.expectEqual(@as(u32, 1), attn_cpu.partCount(starts[0] + 1));
    try std.testing.expectEqual(@as(u32, 2), attn_cpu.partCount(starts[0] + chunk - slot1_rows));
    try std.testing.expectEqual(@as(u32, 3), attn_cpu.partCount(starts[1] + 1));
    try std.testing.expectEqual(@as(u32, 4), attn_cpu.partCount(starts[1] + slot1_rows));
    var prng = std.Random.DefaultPrng.init(0xA770_57E9);
    const r = prng.random();
    inline for (.{ naive, simd_pf }) |Kernels| {
        inline for (.{ f32, f16 }) |E| {
            for (shapes) |s| {
                for (step_case_rows) |n_rows| {
                    const threads = &step_case_threads;
                    try expectAttentionRowsMatch(E, Kernels, r, s, starts, n_rows, threads);
                }
            }
        }
    }
}

test "attentionRowsTyped: the widest serve step gives the bits of the serial row kernel" {
    const simd_pf = @import("../ops/simd_pf.zig");
    // A small shape at shallow depths keeps the widest step short.
    const s: AttnShape = .{ .n_heads = 2, .n_kv_heads = 1, .head_dim = 8 };
    var prng = std.Random.DefaultPrng.init(0xA770_0108);
    const r = prng.random();
    try expectAttentionRowsMatch(f32, simd_pf, r, s, .{ 0, 0 }, limits.step_rows_max, &.{ 1, 4 });
}

/// Check that `attentionRowsTyped` over a step of `n_rows` rows of two slots gives the bits of
/// `Kernels.attentionRowTyped` row by row, at each thread count. Slot `j` starts at the depth
/// `starts[j]`, and its rows are in time order.
fn expectAttentionRowsMatch(
    comptime E: type,
    comptime Kernels: type,
    r: std.Random,
    s: AttnShape,
    starts: [2]u32,
    n_rows: usize,
    thread_counts: []const u32,
) !void {
    const gpa = std.testing.allocator;
    const kv_row = s.n_kv_heads * s.head_dim;
    const width = s.n_heads * s.head_dim;

    // Fill the KV of both slots and the q of every row.
    var kv: [2][2][]E = undefined;
    for (&kv, starts) |*slot, start| {
        for (slot) |*buf| {
            buf.* = try gpa.alloc(E, (start + n_rows) * kv_row);
            for (buf.*) |*x| {
                x.* = @floatCast(r.float(f32) * 2.0 - 1.0);
            }
        }
    }
    defer for (kv) |slot| {
        for (slot) |buf| {
            gpa.free(buf);
        }
    };

    const q = try gpa.alloc(f32, n_rows * width);
    defer gpa.free(q);

    for (q) |*x| {
        x.* = r.float(f32) * 2.0 - 1.0;
    }

    // Lay out the rows: each slot in time order, the slots mixed.
    const out_want = try gpa.alloc(f32, n_rows * width);
    defer gpa.free(out_want);

    const out_got = try gpa.alloc(f32, n_rows * width);
    defer gpa.free(out_got);

    const rows = try gpa.alloc(AttentionRow(E), n_rows);
    defer gpa.free(rows);

    var next_pos = starts;
    for (rows, 0..) |*row, i| {
        const slot = stepCaseSlot(i);
        const t_len = next_pos[slot] + 1;
        next_pos[slot] += 1;
        row.* = .{
            .q = q[i * width ..][0..width],
            .k_cache = kv[slot][0][0 .. t_len * kv_row],
            .v_cache = kv[slot][1][0 .. t_len * kv_row],
            .t_len = t_len,
            .out = out_got[i * width ..][0..width],
        };
    }

    // Run the serial row kernel for the expected bits.
    const row_parts_len = attn_cpu.partialLen(@intCast(s.n_heads), @intCast(s.head_dim));
    const parts = try gpa.alloc(f32, n_rows * row_parts_len);
    defer gpa.free(parts);

    for (rows, 0..) |row, i| {
        Kernels.attentionRowTyped(
            E,
            row.q,
            s.n_heads,
            s.head_dim,
            row.k_cache,
            row.v_cache,
            s.n_kv_heads,
            row.t_len,
            parts[0..row_parts_len],
            out_want[i * width ..][0..width],
        );
    }

    // Run the step fork at each thread count over stale records.
    for (thread_counts) |n_threads| {
        for (parts) |*x| {
            x.* = r.float(f32) * 200.0 - 100.0;
        }

        @memset(out_got, std.math.nan(f32));
        var pool: Pool = undefined;
        try pool.init(std.testing.io, n_threads, 50_000);
        defer pool.deinit();

        const policy: Policy = .{ .n_threads = n_threads, .min_parallel_flops = 0 };
        const nh = s.n_heads;
        attentionRowsTyped(E, Kernels, &pool, policy, rows, nh, s.head_dim, s.n_kv_heads, parts);
        try std.testing.expectEqualSlices(
            u8,
            std.mem.sliceAsBytes(out_want),
            std.mem.sliceAsBytes(out_got),
        );
    }
}

test "gdnRows: one fork per step gives the bits of the serial step" {
    const naive = @import("../ops/naive.zig");
    const simd_pf = @import("../ops/simd_pf.zig");
    const gdn_simd = @import("../ops/gdn_simd.zig");
    const simd = @import("../ops/simd.zig");
    // Both SIMD backends run one GDN unit, so the simd_pf run covers simd too.
    comptime std.debug.assert(simd.gdnStepKeyHead == simd_pf.gdnStepKeyHead);
    // The 2B, the 9B, and a small shape of 2 key heads. At 3 threads the key heads split unevenly.
    const small: gdn.Dims = .{ .n_k_heads = 2, .n_v_heads = 6, .head_dim = 40, .conv_kernel = 4 };
    const qwen_2b: gdn.Dims = .{
        .n_k_heads = 16,
        .n_v_heads = 16,
        .head_dim = 128,
        .conv_kernel = 4,
    };
    const qwen_9b: gdn.Dims = .{
        .n_k_heads = 16,
        .n_v_heads = 32,
        .head_dim = 128,
        .conv_kernel = 4,
    };
    var prng = std.Random.DefaultPrng.init(0x6D17_57E9);
    const r = prng.random();
    // Each backend with the serial step of its own arithmetic.
    inline for (.{ .{ naive, gdn.reference }, .{ simd_pf, gdn_simd.kernels } }) |set| {
        for (step_case_rows) |n_rows| {
            try expectGdnRowsMatch(set[0], set[1], r, small, n_rows, &step_case_threads);
        }
        // The large shapes at a few rows keep the test short.
        try expectGdnRowsMatch(set[0], set[1], r, qwen_2b, 5, &step_case_threads);
        try expectGdnRowsMatch(set[0], set[1], r, qwen_9b, 2, &step_case_threads);
        try expectGdnRowsMatch(set[0], set[1], r, small, limits.step_rows_max, &.{ 1, 4 });
    }
}

/// Check that `gdnRows` over `Kernels` and a step of `n_rows` rows of two slots gives the bits of
/// `Serial.step` row by row, at each thread count: the readouts, the states, the conv rings, and
/// the qkv stripes.
fn expectGdnRowsMatch(
    comptime Kernels: type,
    comptime Serial: type,
    r: std.Random,
    d: gdn.Dims,
    n_rows: usize,
    thread_counts: []const u32,
) !void {
    const gpa = std.testing.allocator;
    const cd = d.convDim();
    const nv = d.n_v_heads;

    // Draw the stripes of every row, the weights, and the start slabs of both slots.
    const stripes0 = try gpa.alloc(f32, n_rows * gdnStripeLen(d));
    defer gpa.free(stripes0);

    for (stripes0) |*x| {
        x.* = r.float(f32) * 2.0 - 1.0;
    }

    const weights = try gpa.alloc(f32, gdnWeightsLen(d));
    defer gpa.free(weights);

    for (weights) |*x| {
        x.* = r.float(f32) * 2.0 - 1.0;
    }

    const conv_w = weights[0 .. d.conv_kernel * cd];
    const ssm_a = weights[d.conv_kernel * cd ..][0..nv];
    for (ssm_a) |*x| {
        x.* = -@abs(x.*) - 0.1;
    }

    const w: GdnWeights = .{
        .conv_w = conv_w,
        .ssm_a = ssm_a,
        .dt_bias = weights[d.conv_kernel * cd + nv ..][0..nv],
        .ssm_norm = weights[d.conv_kernel * cd + 2 * nv ..][0..d.head_dim],
        .rms_eps = 1e-6,
    };
    const slabs0 = try gpa.alloc(f32, 2 * gdnSlabLen(d));
    defer gpa.free(slabs0);

    for (slabs0) |*x| {
        x.* = (r.float(f32) - 0.5) * 0.2;
    }

    // Each run works on copies: the stripes, the slabs, and the readouts.
    const want = try gpa.alloc(f32, gdnCaseLen(d, n_rows));
    defer gpa.free(want);

    const got = try gpa.alloc(f32, want.len);
    defer gpa.free(got);

    // Run the serial step of the backend row by row for the expected bits.
    {
        const rows = try gdnCaseRows(gpa, d, want, stripes0, slabs0, n_rows);
        defer gpa.free(rows);

        for (rows) |row| {
            Serial.step(
                d,
                row.qkv,
                row.z,
                row.a,
                row.b,
                w.conv_w,
                w.ssm_a,
                w.dt_bias,
                w.ssm_norm,
                w.rms_eps,
                row.state,
                row.conv_state,
                row.o_out,
            );
        }
    }

    // Run the step fork at each thread count and compare every buffer.
    for (thread_counts) |n_threads| {
        var pool: Pool = undefined;
        try pool.init(std.testing.io, n_threads, 50_000);
        defer pool.deinit();

        const policy: Policy = .{ .n_threads = n_threads, .min_parallel_flops = 0 };
        const rows = try gdnCaseRows(gpa, d, got, stripes0, slabs0, n_rows);
        defer gpa.free(rows);

        gdnRows(Kernels, &pool, policy, d, rows, w);
        try std.testing.expectEqualSlices(
            u8,
            std.mem.sliceAsBytes(want),
            std.mem.sliceAsBytes(got),
        );
    }
}

test "gdnRows: one participant steps each key head over the rows in row order" {
    if (builtin.single_threaded) return error.SkipZigTest;

    const d: gdn.Dims = .{ .n_k_heads = 2, .n_v_heads = 6, .head_dim = 40, .conv_kernel = 4 };
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6D17_0D3E);
    const r = prng.random();
    for ([_]usize{ 2, 5, 64 }) |n_rows| {
        const stripes0 = try gpa.alloc(f32, n_rows * gdnStripeLen(d));
        defer gpa.free(stripes0);

        const slabs0 = try gpa.alloc(f32, 2 * gdnSlabLen(d));
        defer gpa.free(slabs0);

        const weights = try gpa.alloc(f32, gdnWeightsLen(d));
        defer gpa.free(weights);

        for ([_][]f32{ stripes0, slabs0, weights }) |buf| {
            for (buf) |*x| {
                x.* = (r.float(f32) - 0.5) * 0.2;
            }
        }

        const cw_len = d.conv_kernel * d.convDim();
        for (weights[cw_len..][0..d.n_v_heads]) |*x| {
            x.* = -@abs(x.*) - 0.1;
        }

        const w: GdnWeights = .{
            .conv_w = weights[0..cw_len],
            .ssm_a = weights[cw_len..][0..d.n_v_heads],
            .dt_bias = weights[cw_len + d.n_v_heads ..][0..d.n_v_heads],
            .ssm_norm = weights[cw_len + 2 * d.n_v_heads ..][0..d.head_dim],
            .rms_eps = 1e-6,
        };
        const buf = try gpa.alloc(f32, gdnCaseLen(d, n_rows));
        defer gpa.free(buf);

        for ([_]u32{ 2, 4, 16 }) |n_threads| {
            var pool: Pool = undefined;
            try pool.init(std.testing.io, n_threads, 50_000);
            defer pool.deinit();

            const rows = try gdnCaseRows(gpa, d, buf, stripes0, slabs0, n_rows);
            defer gpa.free(rows);

            GdnOrderKernels.reset(rows[0].o_out.ptr, d.valueDim());
            const policy: Policy = .{ .n_threads = n_threads, .min_parallel_flops = 0 };
            gdnRows(GdnOrderKernels, &pool, policy, d, rows, w);
            try std.testing.expect(!GdnOrderKernels.gate_timed_out.load(.acquire));
            try std.testing.expect(!GdnOrderKernels.out_of_order.load(.acquire));
            for (GdnOrderKernels.next_row[0..d.n_k_heads]) |*next| {
                try std.testing.expectEqual(@as(u32, @intCast(n_rows)), next.load(.acquire));
            }
        }
    }
}

/// The naive key head with a check that each key head steps the rows one at a time, in row order.
/// The pool runs unit 0 on the caller and unit 1 on a worker. Row 0 of key head 0 waits until a
/// later row starts. Key head 1 ends the wait, but a unit split over rows ends it with a later row
/// of key head 0, out of order.
const GdnOrderKernels = struct {
    const kh_max = 2;
    /// The bound of the wait of row 0. Each yield lets another participant run.
    const gate_yields_max = 1 << 22;

    var o_base: usize = 0;
    var o_stride: usize = 0;
    var next_row: [kh_max]std.atomic.Value(u32) = @splat(.init(0));
    var later_started: std.atomic.Value(bool) = .init(false);
    var out_of_order: std.atomic.Value(bool) = .init(false);
    var gate_timed_out: std.atomic.Value(bool) = .init(false);

    fn reset(o_first: [*]const f32, value_dim: usize) void {
        o_base = @intFromPtr(o_first);
        o_stride = value_dim * @sizeOf(f32);
        for (&next_row) |*next| {
            next.store(0, .release);
        }

        later_started.store(false, .release);
        out_of_order.store(false, .release);
        gate_timed_out.store(false, .release);
    }

    pub fn gdnStepKeyHead(
        d: gdn.Dims,
        qkv: []f32,
        conv_w: []const f32,
        conv_state: []f32,
        heads: gdn.Heads,
        kh: usize,
    ) void {
        std.debug.assert(kh < kh_max);
        const row: u32 = @intCast((@intFromPtr(heads.o_out.ptr) - o_base) / o_stride);
        if (next_row[kh].load(.acquire) != row) {
            out_of_order.store(true, .release);
        }

        if (row >= 1) {
            later_started.store(true, .release);
        }

        if (kh == 0 and row == 0) {
            for (0..gate_yields_max) |_| {
                if (later_started.load(.acquire)) break;
                std.Thread.yield() catch {};
            } else gate_timed_out.store(true, .release);
        }

        gdn.stepKeyHead(d, qkv, conv_w, conv_state, heads, kh);
        next_row[kh].store(row + 1, .release);
    }
};

test "attentionRowsTyped and gdnRows: the floor counts the FLOPs of the whole step" {
    const naive = @import("../ops/naive.zig");
    const gpa = std.testing.allocator;
    var pool: Pool = undefined;
    try pool.init(std.testing.io, 4, 50_000);
    defer pool.deinit();

    // Three attention rows of one slot at depths 1, 2, and 3. The step holds
    // 4 * n_heads * head_dim * (1 + 2 + 3) FLOPs, and each row alone holds less.
    const s: AttnShape = .{ .n_heads = 4, .n_kv_heads = 2, .head_dim = 8 };
    const width = s.n_heads * s.head_dim;
    const kv = try gpa.alloc(f32, 2 * 3 * s.n_kv_heads * s.head_dim);
    defer gpa.free(kv);

    const io_buf = try gpa.alloc(f32, 2 * 3 * width);
    defer gpa.free(io_buf);

    const row_parts_len = attn_cpu.partialLen(@intCast(s.n_heads), @intCast(s.head_dim));
    const parts = try gpa.alloc(f32, 3 * row_parts_len);
    defer gpa.free(parts);

    @memset(kv, 0.25);
    @memset(io_buf, 0.5);
    const kv_len = 3 * s.n_kv_heads * s.head_dim;
    var rows: [3]AttentionRow(f32) = undefined;
    for (&rows, 0..) |*row, i| {
        row.* = .{
            .q = io_buf[i * width ..][0..width],
            .k_cache = kv[0..kv_len],
            .v_cache = kv[kv_len..],
            .t_len = @intCast(i + 1),
            .out = io_buf[(3 + i) * width ..][0..width],
        };
    }

    const attn_flops: u64 = 4 * s.n_heads * s.head_dim * (1 + 2 + 3);
    for ([_]u64{ attn_flops, attn_flops + 1 }, [_]u32{ 1, 0 }) |floor, forks| {
        const policy: Policy = .{ .n_threads = 4, .min_parallel_flops = floor };
        const seq = pool.seq;
        attentionRowsTyped(f32, naive, &pool, policy, &rows, s.n_heads, s.head_dim, 2, parts);
        try std.testing.expectEqual(forks, pool.seq - seq);
    }

    // Three GDN rows. The step holds 7 * n_v_heads * head_dim^2 * 3 FLOPs.
    const d: gdn.Dims = .{ .n_k_heads = 2, .n_v_heads = 6, .head_dim = 40, .conv_kernel = 4 };
    const stripes0 = try gpa.alloc(f32, 3 * gdnStripeLen(d));
    defer gpa.free(stripes0);

    const slabs0 = try gpa.alloc(f32, 2 * gdnSlabLen(d));
    defer gpa.free(slabs0);

    const weights = try gpa.alloc(f32, gdnWeightsLen(d));
    defer gpa.free(weights);

    @memset(stripes0, 0.1);
    @memset(slabs0, 0.01);
    @memset(weights, -0.2);
    const cw_len = d.conv_kernel * d.convDim();
    const w: GdnWeights = .{
        .conv_w = weights[0..cw_len],
        .ssm_a = weights[cw_len..][0..d.n_v_heads],
        .dt_bias = weights[cw_len + d.n_v_heads ..][0..d.n_v_heads],
        .ssm_norm = weights[cw_len + 2 * d.n_v_heads ..][0..d.head_dim],
        .rms_eps = 1e-6,
    };
    const buf = try gpa.alloc(f32, gdnCaseLen(d, 3));
    defer gpa.free(buf);

    const gdn_flops: u64 = 7 * d.n_v_heads * d.head_dim * d.head_dim * 3;
    for ([_]u64{ gdn_flops, gdn_flops + 1 }, [_]u32{ 1, 0 }) |floor, forks| {
        const policy: Policy = .{ .n_threads = 4, .min_parallel_flops = floor };
        const gdn_rows = try gdnCaseRows(gpa, d, buf, stripes0, slabs0, 3);
        defer gpa.free(gdn_rows);

        const seq = pool.seq;
        gdnRows(naive, &pool, policy, d, gdn_rows, w);
        try std.testing.expectEqual(forks, pool.seq - seq);
    }
}

/// The f32s of the weights of one GDN layer: the conv, `ssm_a`, `dt_bias`, and `ssm_norm`.
fn gdnWeightsLen(d: gdn.Dims) usize {
    return d.conv_kernel * d.convDim() + 2 * d.n_v_heads + d.head_dim;
}

/// The f32s of the stripes of one GDN row: qkv, z, alpha, and beta.
fn gdnStripeLen(d: gdn.Dims) usize {
    return d.convDim() + d.valueDim() + 2 * d.n_v_heads;
}

/// The f32s of the slabs of one slot: the state and the conv ring.
fn gdnSlabLen(d: gdn.Dims) usize {
    return d.stateLen() + d.convStateLen();
}

/// The f32s of one GDN case of `n_rows` rows: the stripes, the slabs of two slots, the readouts.
fn gdnCaseLen(d: gdn.Dims, n_rows: usize) usize {
    return n_rows * gdnStripeLen(d) + 2 * gdnSlabLen(d) + n_rows * d.valueDim();
}

/// Copy the start stripes and slabs into `buf`, fill the readouts with NaN, and carve the rows of
/// a step of two slots. The caller frees the rows.
fn gdnCaseRows(
    gpa: std.mem.Allocator,
    d: gdn.Dims,
    buf: []f32,
    stripes0: []const f32,
    slabs0: []const f32,
    n_rows: usize,
) ![]GdnRow {
    const cd = d.convDim();
    const vd = d.valueDim();
    const nv = d.n_v_heads;
    const stripe_len = gdnStripeLen(d);
    const slab_len = gdnSlabLen(d);
    std.debug.assert(stripes0.len == n_rows * stripe_len);
    std.debug.assert(slabs0.len == 2 * slab_len);
    std.debug.assert(buf.len == gdnCaseLen(d, n_rows));

    @memcpy(buf[0..stripes0.len], stripes0);
    @memcpy(buf[stripes0.len..][0..slabs0.len], slabs0);
    const slabs = buf[stripes0.len..][0..slabs0.len];
    const outs = buf[stripes0.len + slabs0.len ..];
    @memset(outs, std.math.nan(f32));

    const rows = try gpa.alloc(GdnRow, n_rows);
    for (rows, 0..) |*row, i| {
        const stripe = buf[i * stripe_len ..][0..stripe_len];
        const slab = slabs[stepCaseSlot(i) * slab_len ..][0..slab_len];
        row.* = .{
            .qkv = stripe[0..cd],
            .z = stripe[cd..][0..vd],
            .a = stripe[cd + vd ..][0..nv],
            .b = stripe[cd + vd + nv ..][0..nv],
            .state = slab[0..d.stateLen()],
            .conv_state = slab[d.stateLen()..],
            .o_out = outs[i * vd ..][0..vd],
        };
    }
    return rows;
}
