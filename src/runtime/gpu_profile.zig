//! GPU timestamps for every GPU command buffer of both model families. Each command buffer holds
//! one `GpuProfile`: the two decode ring buffers, the batch and prefill buffer of each runner, and
//! the one-shot staging buffer (through `vulkan_ctx.Context.staging_query_pool`).
//!
//! Comptime. `core_log.hot` removes the trace code from builds without hot tracing, and the mark
//! bound comes from the limits.
//!
//! Levels. The level comes from `$ZIGFERENCER_TRACE_GPU` (`span.gpuLevel`). At `off`, the default,
//! `begin`, `mark`, and `end` emit no command, so the command stream is byte-identical to an
//! unprofiled one. At `submit`, a buffer writes two timestamps, start and end. At `phases`, it
//! writes every `mark` of the record functions, and `collect` sums the intervals by `OpKey`.
//! `mark` also carries the `.dispatch` hot trace, which `-Dtrace` gates at build time.
//!
//! Marks. A mark closes the interval of one op set, or of one pass of a split op. `report` groups
//! the sums by `OpClass`, then by op, and it prints each pass under its op.
//!
//! Query rules. `begin` resets every slot the buffer can write, in the same command buffer, before
//! the first write. `collect` reads the pool only after the wait that the submit path already does,
//! so the pool is never read while its buffer is pending.
//!
//! Readback. `collect` reads first without the wait flag, because with the flag a driver can make
//! one blocking call for each query. A result that is not ready yet counts in `not_ready` and
//! reads again with the flag.
//!
//! Full bound. `prof_max` bounds the marks of one submit: the attention of the widest prefill
//! chunk marks two passes per record group. `dropped` counts the marks past it, and `report`
//! prints the count, so a full pool never deflates the percentages in silence. Each profile warns
//! once, which keeps a long run from writing one line per record.

const std = @import("std");
const vk = @import("vulkan");
const vulkan_ctx = @import("vulkan_ctx.zig");
const core_log = @import("../core/utils/log.zig");
const layer_op = @import("../core/layer_op.zig");
const limits = @import("../core/limits.zig");
const span = @import("../core/span.zig");
const record_ring = @import("../core/record_ring.zig");

const OpClass = layer_op.OpClass;
const OpKey = layer_op.OpKey;
const OpPart = layer_op.OpPart;
const OpSet = layer_op.OpSet;
const assert = std.debug.assert;

const log_dispatch = std.log.scoped(.dispatch);
const log = std.log.scoped(.gpu);

// -- constants --------------------------------------------------------------

/// The most record groups of the split attention in one step: the widest prefill chunk, with at
/// least `attn_split_rows` split rows in each group.
const attn_groups_max: u32 = std.math.divCeil(
    u32,
    limits.max_prefill_chunk,
    limits.attn_split_rows,
) catch unreachable;

/// The most marks of one layer: one per op of the list, a second pass of the KV commit and of the
/// GDN, and the split mark and the merge mark of each attention record group.
const layer_marks_max: u32 = layer_op.op_count + 2 + 2 * attn_groups_max;

/// The input copy, the final norm, the lm_head, and the readback.
const head_marks = 4;

/// Bound on timestamps per profiled submit: the start, the marks of `limits.max_layers` layers,
/// and the marks of the input copy and of the head.
pub const prof_max: u32 = 1 + limits.max_layers * layer_marks_max + head_marks;

/// Stdout buffer of one `report`.
const report_buffer_bytes = 4096;

/// The deepest indent of a `report` line: a pass under its op under its class.
const report_indent_max = 4;

/// The longest line label of `report`: the longest key name plus the deepest indent.
const report_label_bytes = report_indent_max + keyNameMax();

// -- types ------------------------------------------------------------------

pub const Level = span.GpuLevel;

pub const RecordOpts = record_ring.RecordOpts;
pub const DecodeKey = record_ring.DecodeKey;

/// One read-back command buffer. The ticks are raw device ticks.
pub const Readout = struct {
    gpu_start: u64,
    gpu_end: u64,
    period_ns: f64,
    marks: u32,
    dropped: u32,
    /// The intervals where the device counter stepped back.
    backsteps: u32,
    /// The interval sums by op key, at the `phases` level only.
    op_ns: ?[OpKey.count]f64,

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
        if (self.op_ns) |sums| {
            var out: [OpKey.count]u64 = undefined;
            for (sums, &out) |ns, *o| {
                o.* = @intFromFloat(ns);
            }

            rec.op_ns = out;
        }

        return rec;
    }

    /// Print the table of the op sums on stdout, because the output of an opt-in profiler is
    /// program output. `hdr` labels the run, as in "[qwen prefill warmup] b=411". The two host
    /// times are the record step and the submit-plus-wait step. A line with no time stays out.
    pub fn report(
        self: Readout,
        io: std.Io,
        comptime hdr: []const u8,
        hdr_args: anytype,
        record_ms: f64,
        wait_ms: f64,
    ) !void {
        // Print the run line.
        const sums = &self.op_ns.?;
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

        // Print each class, then its ops, then the passes of each op.
        for (std.enums.values(OpClass)) |class| {
            try printClass(w, sums, class, total_ns);
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
    /// Marks past the `prof_max` bound since `begin`.
    dropped: u32 = 0,
    /// True after the first warning of a full bound.
    warned_full: bool = false,
    /// Readbacks that found a result not ready after the wait and read again with the wait flag.
    not_ready: u32 = 0,
    /// The key of the interval that each timestamp closes. Index 0 closes none.
    labels: [prof_max]OpKey = undefined,

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
        if (level != .off) assert(self.canTime());

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
                "this GPU queue writes no usable timestamps ({d} valid bits, " ++
                    "a period of {d} ns), so a GPU profile cannot run. " ++
                    "Run without the profile flag",
                .{ self.ctx.timestamp_valid_bits, self.period_ns },
            );
            return error.TimestampsUnsupported;
        }

        try self.ensurePool(slotsFor(level));
        assert(self.capacity >= slotsFor(level));
    }

    /// True when the queue can time a command buffer.
    pub fn canTime(self: *const GpuProfile) bool {
        return self.tick_mask != 0 and self.period_ns > 0;
    }

    /// Change the level within the pool. The next record writes at it.
    pub fn setLevel(self: *GpuProfile, level: Level) void {
        assert(slotsFor(level) <= self.capacity);

        self.level = level;
        self.n = 0;
    }

    /// At command-buffer start: reset every slot this record can write and
    /// write the start. Index 0 is the pass start, so no interval precedes it.
    pub fn begin(self: *GpuProfile, cb: vk.CommandBuffer) void {
        if (self.level == .off) return;

        const slots = slotsFor(self.level);
        assert(slots <= self.capacity);
        self.ctx.vkd.cmdResetQueryPool(cb, self.query_pool, 0, slots);
        self.ctx.vkd.cmdWriteTimestamp(cb, .{ .top_of_pipe_bit = true }, self.query_pool, 0);
        self.n = 1;
        self.dropped = 0;
    }

    /// Close the interval that just ended on `ops`, or on the pass `part` of the one op of
    /// `ops`. The `.dispatch` trace fires on every record pass. The timestamp writes only at
    /// `phases`.
    pub fn mark(self: *GpuProfile, cb: vk.CommandBuffer, ops: OpSet, part: ?OpPart) void {
        const key: OpKey = .of(ops, part);
        if (comptime core_log.hot) {
            log_dispatch.debug("mark: {s}", .{key.name()});
        }

        if (self.level != .phases) return;

        assert(self.n >= 1);
        if (self.n >= prof_max) {
            if (!self.warned_full) {
                log_dispatch.warn(
                    "profile: the {d}-timestamp bound is full, " ++
                        "so the rest of each such submit goes untimed. " ++
                        "The readout counts the dropped marks",
                    .{prof_max},
                );
            }

            self.warned_full = true;
            self.dropped += 1;
            return;
        }

        self.labels[self.n] = key;
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
                assert(self.n == 1);

                self.ctx.vkd.cmdWriteTimestamp(
                    cb,
                    .{ .bottom_of_pipe_bit = true },
                    self.query_pool,
                    1,
                );
                self.n = 2;
            },
            .phases => assert(self.n >= 2),
        }
    }

    /// Read back the last record of this buffer, or null when it wrote no timestamp. Call it only
    /// after the wait of the submit: a read with the wait flag blocks until the submit runs.
    pub fn collect(self: *GpuProfile) !?Readout {
        if (self.n == 0) return null;

        const n = self.n;
        assert(n >= 2);
        assert(n <= self.capacity);
        self.n = 0;

        var ts: [prof_max]u64 = undefined;
        try self.readAll(ts[0..n]);

        // Turn the ticks into intervals, and sum the intervals by key at `phases`.
        var deltas: [prof_max - 1]u64 = undefined;
        const t = span.tickSpan(ts[0..n], self.tick_mask, deltas[0 .. n - 1]);
        var r: Readout = .{
            .gpu_start = t.start,
            .gpu_end = t.end,
            .period_ns = self.period_ns,
            .marks = n,
            .dropped = self.dropped,
            .backsteps = t.backsteps,
            .op_ns = null,
        };
        if (self.level == .phases) {
            var sums: [OpKey.count]f64 = @splat(0);
            for (deltas[0 .. n - 1], self.labels[1..n]) |dt, key| {
                sums[@intFromEnum(key)] += @as(f64, @floatFromInt(dt)) * self.period_ns;
            }

            r.op_ns = sums;
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

    /// Read the first `ts.len` slots, and read again with the wait flag only when a result is not
    /// ready yet.
    fn readAll(self: *GpuProfile, ts: []u64) !void {
        const sp = span.begin(.gpu_collect);
        defer sp.end();

        if (try self.readTicks(ts, false) == .success) return;

        self.not_ready +|= 1;
        if (self.not_ready == 1) {
            log.warn(
                "profile: a timestamp was not ready after the wait, " ++
                    "so this profile reads again with the wait flag when that happens",
                .{},
            );
        }

        const again = try self.readTicks(ts, true);
        assert(again == .success);
    }

    /// Read the first `ts.len` slots. With `wait`, the driver blocks until each result is ready.
    fn readTicks(self: *GpuProfile, ts: []u64, wait: bool) !vk.Result {
        assert(ts.len >= 2);
        assert(ts.len <= self.capacity);

        const n: u32 = @intCast(ts.len);
        return self.ctx.vkd.getQueryPoolResults(
            self.ctx.device,
            self.query_pool,
            0,
            n,
            n * @sizeOf(u64),
            ts.ptr,
            @sizeOf(u64),
            .{ .@"64_bit" = true, .wait_bit = wait },
        );
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
    assert(valid_bits <= 64);

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

/// The sum of every key of `class`.
fn classNs(sums: *const [OpKey.count]f64, class: OpClass) f64 {
    var ns: f64 = 0;
    for (sums, 0..) |key_ns, i| {
        const key: OpKey = @enumFromInt(i);
        if (key.class() == class) {
            ns += key_ns;
        }
    }

    return ns;
}

/// The time of the line of an op key: its own sum, plus the sum of each pass of its op.
fn lineNs(sums: *const [OpKey.count]f64, key: OpKey) f64 {
    assert(key.part() == null);

    var ns = sums[@intFromEnum(key)];
    if (key.ops().count() > 1) return ns;

    for (std.enums.values(OpPart)) |p| {
        if (p.op() == key.op()) {
            ns += sums[@intFromEnum(OpKey.of(key.ops(), p))];
        }
    }

    return ns;
}

/// The lines of `class` in a report: the class, each op with time, and each pass with time.
fn printClass(
    w: *std.Io.Writer,
    sums: *const [OpKey.count]f64,
    class: OpClass,
    total_ns: f64,
) !void {
    const class_ns = classNs(sums, class);
    if (class_ns == 0) return;

    try printLine(w, @tagName(class), class_ns, total_ns, 0);
    for (0..OpKey.count) |i| {
        const key: OpKey = @enumFromInt(i);
        if (key.part() != null or key.class() != class) continue;

        const line_ns = lineNs(sums, key);
        if (line_ns == 0) continue;

        try printLine(w, key.name(), line_ns, total_ns, 2);
        if (key.ops().count() > 1) continue;

        for (std.enums.values(OpPart)) |p| {
            if (p.op() != key.op()) continue;

            const ns = sums[@intFromEnum(OpKey.of(key.ops(), p))];
            if (ns > 0) {
                try printLine(w, @tagName(p), ns, total_ns, 4);
            }
        }
    }
}

/// The longest key name, which sizes the label of a report line.
fn keyNameMax() usize {
    var most: usize = 0;
    for (0..OpKey.count) |i| {
        const key: OpKey = @enumFromInt(i);
        most = @max(most, key.name().len);
    }

    return most;
}

/// One report line. A nested line sits `indent` spaces in and keeps the columns.
fn printLine(w: *std.Io.Writer, name: []const u8, ns: f64, total_ns: f64, indent: usize) !void {
    assert(indent <= report_indent_max);
    assert(indent + name.len <= report_label_bytes);

    var label: [report_label_bytes]u8 = @splat(' ');
    @memcpy(label[indent..][0..name.len], name);
    try w.print("  {s} {d:>8.2} ms  ({d:>5.1}%)\n", .{ &label, ns / 1e6, 100.0 * ns / total_ns });
}

// -- tests ------------------------------------------------------------------

test "lineNs: an op line sums the op and each of its passes, and a class sums its keys" {
    var sums: [OpKey.count]f64 = @splat(0);
    const attention: OpSet = .initOne(.attention);
    sums[@intFromEnum(OpKey.ofOp(.attention))] = 1;
    sums[@intFromEnum(OpKey.of(attention, .split))] = 2;
    sums[@intFromEnum(OpKey.of(attention, .merge))] = 4;
    sums[@intFromEnum(OpKey.of(.initOne(.gdn), .conv))] = 8;
    sums[@intFromEnum(OpKey.ofOp(.attn_proj))] = 16;
    try std.testing.expectEqual(@as(f64, 7), lineNs(&sums, .ofOp(.attention)));
    try std.testing.expectEqual(@as(f64, 8), lineNs(&sums, .ofOp(.gdn)));
    try std.testing.expectEqual(@as(f64, 7), classNs(&sums, .attention));
    try std.testing.expectEqual(@as(f64, 8), classNs(&sums, .recurrent));
    try std.testing.expectEqual(@as(f64, 16), classNs(&sums, .matmul));
    try std.testing.expectEqual(@as(f64, 0), classNs(&sums, .norm));
}

test "prof_max: the widest prefill chunk of the deepest stack fits one submit" {
    // The richest layer is a Qwen full-attention layer at a narrow KV dtype: twelve ops of one
    // pass, the two KV passes, the resident rows, and two marks per attention record group.
    const richest_layer = 12 + 2 + 1 + 2 * attn_groups_max;
    try std.testing.expectEqual(@as(u32, 64), attn_groups_max);
    try std.testing.expect(layer_marks_max >= richest_layer);
    try std.testing.expect(prof_max >= 1 + limits.max_layers * richest_layer + head_marks);
}

test "a nested line keeps the columns of its class line" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try printLine(&w, "attention", 2e6, 4e6, 0);
    try printLine(&w, "attention", 2e6, 4e6, 2);
    try printLine(&w, "split", 1e6, 4e6, 4);
    try std.testing.expectEqualStrings(
        "  attention                2.00 ms  ( 50.0%)\n" ++
            "    attention              2.00 ms  ( 50.0%)\n" ++
            "      split                1.00 ms  ( 25.0%)\n",
        w.buffered(),
    );
}
