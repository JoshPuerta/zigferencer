//! GPU timestamps for every GPU command buffer of both model families. Each command buffer holds
//! one `GpuProfile`: the two decode ring buffers, the batch and prefill buffer of each runner, and
//! the one-shot staging buffer (through `vulkan_ctx.Context.staging_query_pool`).
//!
//! Levels. The level comes from `$ZIGFERENCER_TRACE_GPU` (`span.gpuLevel`). At `off`, the default,
//! `begin`, `mark`, and `end` emit no command, so the command stream is byte-identical to an
//! unprofiled one. At `submit`, a buffer writes two timestamps, start and end. At `phases`, it
//! writes every `mark` of the record functions, and `report` buckets the intervals by `Phase`.
//! `mark` also carries the `.dispatch` hot trace, which `-Dtrace` gates at build time.
//!
//! Query rules. `begin` resets every slot the buffer can write, in the same command buffer, before
//! the first write. `collect` reads the pool only after the wait that the submit path already does,
//! so the pool is never read while its buffer is pending.
//!
//! Full bound. `prof_max` bounds the marks of one submit. `dropped` counts the marks past it, and
//! `report` prints the count, so a full pool never deflates the per-phase percentages in silence.
//! Each profile warns once, which keeps a long run from writing one line per record.

const std = @import("std");
const vk = @import("vulkan");
const vulkan_ctx = @import("vulkan_ctx.zig");
const core_log = @import("../core/log.zig");
const span = @import("../core/span.zig");
const record_ring = @import("../core/record_ring.zig");

const log_dispatch = std.log.scoped(.dispatch);
const log = std.log.scoped(.gpu);

// -- constants --------------------------------------------------------------

/// Bound on timestamps per profiled submit (64 layers x ~14 marks + tail).
pub const prof_max: u32 = 1024;
/// Stdout buffer of one `report`.
const report_buffer_bytes = 1024;
/// The longest row label of `report`: a phase name plus its indent.
const report_label_bytes = 24;

// -- types ------------------------------------------------------------------

pub const Level = span.GpuLevel;

/// Op-group labels. The Llama runner uses no `gdn_*`, no `matmul_gdn_in`, and
/// no `qknorm`. The Qwen decode step runs conv and scan in one kernel and
/// books it under `gdn_scan`.
pub const Phase = span.GpuPhase;

pub const RecordOpts = record_ring.RecordOpts;
pub const DecodeKey = record_ring.DecodeKey;

/// One read-back command buffer. The ticks are raw device ticks.
pub const Readout = struct {
    gpu_start: u64,
    gpu_end: u64,
    period_ns: f64,
    marks: u32,
    dropped: u32,
    /// The intervals where the device counter stepped back (see `span.tickSpan`).
    backsteps: u32,
    /// The interval sums by phase, at the `phases` level only.
    phase_ns: ?[span.gpu_phase_count]f64,

    /// The GPU record of this readout for the span recorder.
    pub fn record(
        self: Readout,
        runner: span.GpuRunner,
        label: span.GpuLabel,
        submit: std.Io.Timestamp,
        wait: std.Io.Timestamp,
    ) span.GpuRecord {
        var rec: span.GpuRecord = .{
            .runner = runner,
            .label = label,
            .gpu_start = self.gpu_start,
            .gpu_end = self.gpu_end,
            .period_ns = self.period_ns,
            .host_submit_ns = span.tsNs(submit),
            .host_wait_ns = span.tsNs(wait),
            .backsteps = self.backsteps,
        };
        if (self.phase_ns) |ph| {
            var out: [span.gpu_phase_count]u64 = undefined;
            for (ph, &out) |ns, *o| {
                o.* = @intFromFloat(ns);
            }

            rec.phases = out;
        }

        return rec;
    }

    /// Print the per-`Phase` table on stdout, because the output of an opt-in profiler is program
    /// output. `hdr` labels the run, as in "[qwen prefill warmup] b=411". The two host times are
    /// the record step and the submit-plus-wait step.
    pub fn report(
        self: Readout,
        io: std.Io,
        comptime hdr: []const u8,
        hdr_args: anytype,
        record_ms: f64,
        wait_ms: f64,
    ) !void {
        // Print the run line.
        const bucket = self.phase_ns.?;
        const total_ns = @as(f64, @floatFromInt(self.gpu_end - self.gpu_start)) * self.period_ns;

        var buf: [report_buffer_bytes]u8 = undefined;
        var fw: std.Io.File.Writer = .initStreaming(.stdout(), io, &buf);
        const w = &fw.interface;
        try w.print(hdr, hdr_args);
        try w.print(
            " marks={d} dropped={d} backsteps={d} gpu_total={d:.2}ms (host record={d:.2}ms " ++
                "submit+wait={d:.2}ms)\n",
            .{ self.marks, self.dropped, self.backsteps, total_ns / 1e6, record_ms, wait_ms },
        );

        // Sum the group totals. The summary lines `matmul` and `gdn` keep their names for scripts.
        var matmul_ns: f64 = 0;
        var gdn_ns: f64 = 0;
        inline for (std.meta.fields(Phase)) |fld| {
            const ph = @field(Phase, fld.name);
            if (isMatmul(ph)) {
                matmul_ns += bucket[@intFromEnum(ph)];
            }

            if (isGdn(ph)) {
                gdn_ns += bucket[@intFromEnum(ph)];
            }
        }

        // Print each group total, then its indented parts.
        try printLine(w, "matmul", matmul_ns, total_ns, 0);
        inline for (std.meta.fields(Phase)) |fld| {
            const ph = @field(Phase, fld.name);
            if (comptime isMatmul(ph)) {
                try printLine(w, fld.name, bucket[@intFromEnum(ph)], total_ns, 2);
            }
        }

        try printLine(w, "gdn", gdn_ns, total_ns, 0);
        inline for (std.meta.fields(Phase)) |fld| {
            const ph = @field(Phase, fld.name);
            if (comptime isGdn(ph)) {
                try printLine(w, fld.name, bucket[@intFromEnum(ph)], total_ns, 2);
            }
        }

        // Print the phases outside both groups.
        inline for (std.meta.fields(Phase)) |fld| {
            const ph = @field(Phase, fld.name);
            if (comptime !isMatmul(ph) and !isGdn(ph)) {
                try printLine(w, fld.name, bucket[@intFromEnum(ph)], total_ns, 0);
            }
        }

        try w.flush();
    }
};

/// The timestamps of one command buffer.
pub const GpuProfile = struct {
    /// Stable for the lifetime of the runner that owns this profile.
    ctx: *const vulkan_ctx.Context,
    /// Null while the level is `off` and the caller asked for no pool.
    query_pool: vk.QueryPool = .null_handle,
    capacity: u32 = 0,
    period_ns: f64,
    /// The valid bits of a tick, from the queue family.
    tick_mask: u64,
    level: Level,
    /// Timestamps the last record wrote. Zero means nothing to read back.
    n: u32 = 0,
    /// Marks past the `prof_max` bound since `begin`. See "Full bound" in the header.
    dropped: u32 = 0,
    /// True after the first warning of a full bound.
    warned_full: bool = false,
    labels: [prof_max]Phase = undefined,

    /// A profile at `level`. Its pool holds the slots of `pool_for`, or of `level` when that is
    /// larger, so a later `setLevel` up to `pool_for` creates no pool. `off` and `pool_for == .off`
    /// hold no pool.
    pub fn init(ctx: *const vulkan_ctx.Context, level: Level, pool_for: Level) !GpuProfile {
        var self: GpuProfile = .{
            .ctx = ctx,
            .period_ns = @floatCast(ctx.physical_device_props.limits.timestamp_period),
            .tick_mask = tickMask(ctx.timestamp_valid_bits),
            .level = level,
        };
        // `Context.init` drops the level to off on a device that cannot time.
        if (level != .off) std.debug.assert(self.canTime());

        try self.ensurePool(@max(slotsFor(level), slotsFor(pool_for)));
        return self;
    }

    pub fn deinit(self: *GpuProfile) void {
        if (self.query_pool != .null_handle) {
            self.ctx.vkd.destroyQueryPool(self.ctx.device, self.query_pool, null);
        }

        self.* = undefined;
    }

    /// Grow the pool to the slots of `level`. The buffer of this profile must not be pending, and
    /// no recording that names the old pool can be submitted again. A device that cannot time
    /// refuses every level above `off`.
    pub fn reserve(self: *GpuProfile, level: Level) !void {
        if (level != .off and !self.canTime()) {
            log.err(
                "this GPU queue writes no usable timestamps ({d} valid bits, a period of " ++
                    "{d} ns), so a GPU profile cannot run. Run without the profile flag",
                .{ self.ctx.timestamp_valid_bits, self.period_ns },
            );
            return error.TimestampsUnsupported;
        }

        try self.ensurePool(slotsFor(level));
        std.debug.assert(self.capacity >= slotsFor(level));
    }

    /// True when the queue can time a command buffer.
    pub fn canTime(self: *const GpuProfile) bool {
        return self.tick_mask != 0 and self.period_ns > 0;
    }

    /// Change the level within the pool. The next record writes at it.
    pub fn setLevel(self: *GpuProfile, level: Level) void {
        std.debug.assert(slotsFor(level) <= self.capacity);

        self.level = level;
        self.n = 0;
    }

    /// At command-buffer start: reset every slot this record can write and
    /// write the start. Index 0 is the pass start, so no interval precedes it.
    pub fn begin(self: *GpuProfile, cb: vk.CommandBuffer) void {
        if (self.level == .off) return;

        const slots = slotsFor(self.level);
        std.debug.assert(slots <= self.capacity);
        self.ctx.vkd.cmdResetQueryPool(cb, self.query_pool, 0, slots);
        self.ctx.vkd.cmdWriteTimestamp(cb, .{ .top_of_pipe_bit = true }, self.query_pool, 0);
        self.labels[0] = .copy;
        self.n = 1;
        self.dropped = 0;
    }

    /// Tag the interval that just ended with its `Phase`. The `.dispatch` trace
    /// fires on every record pass. The timestamp writes only at `phases`.
    pub fn mark(self: *GpuProfile, cb: vk.CommandBuffer, phase: Phase) void {
        if (comptime core_log.hot) {
            log_dispatch.debug("mark: {t}", .{phase});
        }

        if (self.level != .phases) return;

        std.debug.assert(self.n >= 1);
        if (self.n >= prof_max) {
            if (!self.warned_full) {
                log_dispatch.warn(
                    "profile: the {d}-timestamp bound is full, so the rest of each such " ++
                        "submit goes untimed. The readout counts the dropped marks",
                    .{prof_max},
                );
            }

            self.warned_full = true;
            self.dropped += 1;
            return;
        }

        self.labels[self.n] = phase;
        self.ctx.vkd.cmdWriteTimestamp(
            cb,
            .{ .bottom_of_pipe_bit = true },
            self.query_pool,
            self.n,
        );
        self.n += 1;
    }

    /// Before `endCommandBuffer`. At `submit` it writes the end stamp. At
    /// `phases` the last mark is the end, so it writes nothing.
    pub fn end(self: *GpuProfile, cb: vk.CommandBuffer) void {
        switch (self.level) {
            .off => {},
            .submit => {
                std.debug.assert(self.n == 1);

                self.ctx.vkd.cmdWriteTimestamp(
                    cb,
                    .{ .bottom_of_pipe_bit = true },
                    self.query_pool,
                    1,
                );
                self.n = 2;
            },
            .phases => std.debug.assert(self.n >= 2),
        }
    }

    /// Read back the last record of this buffer, or null when it wrote no
    /// timestamp. Call it only after the wait of the submit, so the results
    /// are available and the wait flag never blocks.
    pub fn collect(self: *GpuProfile) !?Readout {
        if (self.n == 0) return null;

        const n = self.n;
        std.debug.assert(n >= 2);
        std.debug.assert(n <= self.capacity);
        self.n = 0;

        // Read the ticks back.
        var ts: [prof_max]u64 = undefined;
        _ = try self.ctx.vkd.getQueryPoolResults(
            self.ctx.device,
            self.query_pool,
            0,
            n,
            n * @sizeOf(u64),
            &ts,
            @sizeOf(u64),
            .{ .@"64_bit" = true, .wait_bit = true },
        );

        // Turn the ticks into intervals, and bucket the intervals at `phases`.
        var deltas: [prof_max - 1]u64 = undefined;
        const t = span.tickSpan(ts[0..n], self.tick_mask, deltas[0 .. n - 1]);
        var r: Readout = .{
            .gpu_start = t.start,
            .gpu_end = t.end,
            .period_ns = self.period_ns,
            .marks = n,
            .dropped = self.dropped,
            .backsteps = t.backsteps,
            .phase_ns = null,
        };
        if (self.level == .phases) {
            var bucket: [span.gpu_phase_count]f64 = @splat(0);
            for (deltas[0 .. n - 1], self.labels[1..n]) |dt, label| {
                bucket[@intFromEnum(label)] += @as(f64, @floatFromInt(dt)) * self.period_ns;
            }

            r.phase_ns = bucket;
        }

        return r;
    }

    /// `collect`, then hand the record to the span recorder.
    pub fn collectInto(
        self: *GpuProfile,
        runner: span.GpuRunner,
        label: span.GpuLabel,
        submit: std.Io.Timestamp,
        wait: std.Io.Timestamp,
    ) !?Readout {
        const r = try self.collect() orelse return null;
        span.gpu(r.record(runner, label, submit, wait));
        return r;
    }

    fn ensurePool(self: *GpuProfile, slots: u32) !void {
        if (slots <= self.capacity) return;

        const info: vk.QueryPoolCreateInfo = .{ .query_type = .timestamp, .query_count = slots };
        const pool = try self.ctx.vkd.createQueryPool(self.ctx.device, &info, null);
        if (self.query_pool != .null_handle) {
            self.ctx.vkd.destroyQueryPool(self.ctx.device, self.query_pool, null);
        }

        self.query_pool = pool;
        self.capacity = slots;
    }
};

// -- api --------------------------------------------------------------------

pub const decodeKey = record_ring.decodeKey;

/// The mask of the valid bits of a tick. Zero bits give a zero mask.
pub fn tickMask(valid_bits: u32) u64 {
    std.debug.assert(valid_bits <= 64);

    if (valid_bits == 64) return std.math.maxInt(u64);
    return (@as(u64, 1) << @intCast(valid_bits)) - 1;
}

/// The query slots one record writes at `level`.
pub fn slotsFor(level: Level) u32 {
    return switch (level) {
        .off => 0,
        .submit => 2,
        .phases => prof_max,
    };
}

// -- helpers ----------------------------------------------------------------

/// True when the phase counts toward the `matmul` summary line.
fn isMatmul(ph: Phase) bool {
    return switch (ph) {
        .matmul_gdn_in, .matmul_attn_in, .matmul_out, .matmul_ffn_up, .matmul_ffn_down => true,
        else => false,
    };
}

/// True when the phase counts toward the `gdn` summary line.
fn isGdn(ph: Phase) bool {
    return ph == .gdn_conv or ph == .gdn_scan;
}

/// One report row. A part row sits `indent` spaces in and keeps the columns.
fn printLine(w: *std.Io.Writer, name: []const u8, ns: f64, total_ns: f64, indent: usize) !void {
    var label: [report_label_bytes]u8 = undefined;
    std.debug.assert(indent + name.len <= label.len);
    @memset(label[0..indent], ' ');
    @memcpy(label[indent..][0..name.len], name);
    try w.print(
        "  {s:<18} {d:>8.2} ms  ({d:>5.1}%)\n",
        .{ label[0 .. indent + name.len], ns / 1e6, 100.0 * ns / total_ns },
    );
}

// -- tests ------------------------------------------------------------------

test "every phase counts in at most one report group" {
    var matmul: usize = 0;
    var gdn: usize = 0;
    inline for (std.meta.fields(Phase)) |fld| {
        const ph = @field(Phase, fld.name);
        if (isMatmul(ph)) {
            matmul += 1;
        }

        if (isGdn(ph)) {
            gdn += 1;
        }

        try std.testing.expect(!(isMatmul(ph) and isGdn(ph)));
    }

    try std.testing.expectEqual(@as(usize, 5), matmul);
    try std.testing.expectEqual(@as(usize, 2), gdn);
}

test "a part row keeps the summary columns" {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try printLine(&w, "matmul", 2e6, 4e6, 0);
    try printLine(&w, "matmul_out", 1e6, 4e6, 2);
    try std.testing.expectEqualStrings(
        "  matmul                 2.00 ms  ( 50.0%)\n" ++
            "    matmul_out           1.00 ms  ( 25.0%)\n",
        w.buffered(),
    );
}
