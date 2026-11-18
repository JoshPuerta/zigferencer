//! The GPU submit records: their vocabulary, their mapping onto the host clock, and the folded
//! stacks of their op sums. Pure functions only. `span.zig` holds the ring and the writer.
//!
//! Comptime. An assert pins the size of `GpuRecord`, which sizes the GPU ring.
//!
//! GPU clock. The true offset of a record, host time minus device time, lies at or below
//! `host_wait - gpu_end` of that record. A device counter that pauses while the GPU idles only lets
//! the host clock run ahead, so the offset never falls from one record to a later one. The bound of
//! a later record therefore bounds this record too, and the minimum over the record and the records
//! after it comes closest to the true offset.
//!
//! Clamp. Records arrive in host order within a few positions, because two threads can push in
//! the reverse order of their waits. The forward window and the clamp absorb that inversion. A
//! clamp moves the record inside `[host_submit, host_wait]` when a counter breaks that rule, and
//! keeps its duration when it fits.
//!
//! Device ticks. A stamp delta wraps at the valid bits of the queue, so `tickSpan` masks each
//! delta. A masked delta above `mask / 2` is a step back of the counter, not a near-full wrap. It
//! counts as zero ticks and as one backstep. The end is the start plus the sum of the deltas, so
//! `end - start` and the op sums agree. A sum past `u64` saturates.
//!
//! Folded. A stack reads runner, label, class, op, then the pass of a split op, so a flame graph
//! nests each pass under its op.

const std = @import("std");
const layer_op = @import("layer_op.zig");

const OpKey = layer_op.OpKey;
const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

/// Records that the local offset of one record reads: itself and the records after it.
pub const anchor_window: u32 = 16;

// -- types ------------------------------------------------------------------

/// The GPU timestamp level, from `$ZIGFERENCER_TRACE_GPU`. `submit` writes a start and an end
/// stamp per command buffer. `phases` also writes every op mark of the record functions.
pub const GpuLevel = enum(u8) { off, submit, phases };

/// The runner that recorded a GPU command buffer. `staging` is `kernels.submitOnce`, which no
/// runner owns.
pub const GpuRunner = enum(u8) { llama_fused, qwen_fused, llama_hybrid, qwen_hybrid, staging };

/// What a GPU command buffer did. The last four are the labels of `kernels.SubmitLabel`.
pub const GpuLabel = enum(u8) { decode, decode_batch, prefill_chunk, copy, fill, clone, dispatch };

/// One GPU command buffer, read back after the wait its path already does. The device ticks stay
/// raw, and `period_ns` converts them. The host times are `.awake` nanoseconds at submit and wait.
pub const GpuRecord = struct {
    runner: GpuRunner,
    label: GpuLabel,
    gpu_start: u64,
    gpu_end: u64,
    period_ns: f64,
    host_submit_ns: u64,
    host_wait_ns: u64,
    /// The sums by op key of a `phases` record. A `submit` record has none.
    op_ns: ?[OpKey.count]u64 = null,
    /// The intervals where the device counter stepped back. Each counts as zero ticks.
    backsteps: u32 = 0,

    comptime {
        // The ring stores one ready flag beside each record, so an entry costs 281 bytes.
        // `docs/profiling.md` sizes the GPU ring with it.
        assert(@sizeOf(GpuRecord) == 280);
    }

    pub fn gpuStartNs(self: GpuRecord) i64 {
        return ticksToNs(self.gpu_start, self.period_ns);
    }

    pub fn gpuEndNs(self: GpuRecord) i64 {
        return ticksToNs(self.gpu_end, self.period_ns);
    }
};

/// Where one GPU record sits on the host clock, in nanoseconds.
pub const HostInterval = struct {
    start_ns: i64,
    end_ns: i64,
    /// How far the clamp moved the record into its host window.
    clamp_ns: u64,
};

/// The device times of one command buffer, in raw ticks.
pub const TickSpan = struct {
    start: u64,
    end: u64,
    backsteps: u32,
};

// -- api --------------------------------------------------------------------

/// Map record `i` of `records` on the host clock.
pub fn hostInterval(records: []const GpuRecord, i: usize) HostInterval {
    assert(i < records.len);

    const lo = i;
    const hi = @min(records.len, i + anchor_window);
    var offset: i64 = std.math.maxInt(i64);
    for (records[lo..hi]) |rec| {
        offset = @min(offset, @as(i64, @intCast(rec.host_wait_ns)) - rec.gpuEndNs());
    }

    const rec = records[i];
    const submit: i64 = @intCast(rec.host_submit_ns);
    const wait: i64 = @intCast(rec.host_wait_ns);
    assert(submit <= wait);
    const dur = @min(rec.gpuEndNs() - rec.gpuStartNs(), wait - submit);
    assert(dur >= 0);
    const want = rec.gpuStartNs() + offset;
    const start = std.math.clamp(want, submit, wait - dur);
    const p: HostInterval = .{
        .start_ns = start,
        .end_ns = start + dur,
        .clamp_ns = @abs(start - want),
    };
    assert(p.start_ns >= submit);
    assert(p.end_ns <= wait);
    return p;
}

/// Read the start, the end, and the interval deltas of `ts`, the raw stamps of one command buffer
/// in write order.
pub fn tickSpan(ts: []const u64, mask: u64, deltas: ?[]u64) TickSpan {
    assert(ts.len >= 2);
    assert(mask != 0);

    if (deltas) |d| assert(d.len == ts.len - 1);

    var sum: u64 = 0;
    var backsteps: u32 = 0;
    for (ts[0 .. ts.len - 1], ts[1..], 0..) |prev, cur, i| {
        var dt = (cur -% prev) & mask;
        if (dt > mask / 2) {
            dt = 0;
            backsteps += 1;
        }

        if (deltas) |d| {
            d[i] = dt;
        }

        sum +|= dt;
    }

    const start = ts[0] & mask;
    const out: TickSpan = .{ .start = start, .end = start +| sum, .backsteps = backsteps };
    assert(out.start <= out.end);
    assert(out.backsteps < ts.len);
    return out;
}

/// Write one folded-stack line per runner kind, label, and op key with the summed nanoseconds, for
/// `flamegraph.pl`. A `submit` record adds its whole span to its runner and label line.
pub fn writeFolded(w: *std.Io.Writer, records: []const GpuRecord) std.Io.Writer.Error!void {
    const n_runner = std.meta.fields(GpuRunner).len;
    const n_label = std.meta.fields(GpuLabel).len;
    // The last key column holds the records without op sums.
    var sum: [n_runner][n_label][OpKey.count + 1]u64 = @splat(@splat(@splat(0)));
    for (records) |rec| {
        const cell = &sum[@intFromEnum(rec.runner)][@intFromEnum(rec.label)];
        if (rec.op_ns) |sums| {
            for (sums, 0..) |ns, k| {
                cell[k] += ns;
            }
        } else {
            cell[OpKey.count] += @intCast(rec.gpuEndNs() - rec.gpuStartNs());
        }
    }

    for (sum, 0..) |per_runner, ri| {
        for (per_runner, 0..) |cell, li| {
            for (cell, 0..) |ns, k| {
                if (ns == 0) continue;

                const runner: GpuRunner = @enumFromInt(ri);
                const label: GpuLabel = @enumFromInt(li);
                try w.print("{t};{t}", .{ runner, label });
                if (k < OpKey.count) {
                    try writeKeyFrames(w, @enumFromInt(k));
                }

                try w.print(" {d}\n", .{ns});
            }
        }
    }
}

/// Nanoseconds of a raw device tick count.
pub fn ticksToNs(ticks: u64, period_ns: f64) i64 {
    assert(period_ns > 0);

    return @intFromFloat(@as(f64, @floatFromInt(ticks)) * period_ns);
}

// -- helpers ----------------------------------------------------------------

/// The frames of one op key: the class, the op or the fused set, then the pass of a split op.
fn writeKeyFrames(w: *std.Io.Writer, key: OpKey) std.Io.Writer.Error!void {
    if (key.part()) |p| {
        try w.print(";{t};{t};{t}", .{ key.class(), key.op(), p });
    } else {
        try w.print(";{t};{s}", .{ key.class(), key.name() });
    }
}

// -- tests ------------------------------------------------------------------

const prop = @import("utils/prop.zig");
const testing = std.testing;

/// GPU records whose device clock drifts against the host clock: a counter that pauses (the offset
/// grows) or runs fast (the offset falls). `true_start` holds the host time of each GPU start.
const AnchorCase = struct {
    records: [48]GpuRecord,
    true_start: [48]i64,
    n: u32,
    /// True when the offset never falls from one record to the next.
    monotone: bool,
};

const jitter_max_ns: u64 = 2_000;

fn genAnchor(_: void, random: std.Random, size: u32) AnchorCase {
    // Pick the case size and the drift mode.
    var c: AnchorCase = .{
        .records = undefined,
        .true_start = undefined,
        .n = @as(u32, @min(size, 47)) + 1,
        .monotone = true,
    };
    const mode = random.uintLessThan(u32, 3);
    c.monotone = mode != 2;
    const rate: i64 = @intCast(random.uintLessThan(u64, 60_000));
    var offset: i64 = @intCast(1_000_000_000 + random.uintLessThan(u64, 1_000_000));
    var gpu_t: u64 = 10_000_000 + random.uintLessThan(u64, 10_000);

    // Generate each record from the drifting clock.
    for (c.records[0..c.n], 0..) |*r, i| {
        const change: i64 = switch (mode) {
            0 => rate,
            1 => if (random.uintLessThan(u32, 8) == 0) @intCast(
                1_000 + random.uintLessThan(u64, 5_000_000),
            ) else 0,
            else => -@divTrunc(rate, 8),
        };

        if (i > 0) {
            offset += change;
        }

        const busy = 1_000 + random.uintLessThan(u64, 100_000);
        const lead = 5_000 + random.uintLessThan(u64, 5_000);
        const host_start: i64 = @as(i64, @intCast(gpu_t)) + offset;
        c.true_start[i] = host_start;
        r.* = .{
            .runner = .qwen_fused,
            .label = .decode,
            .gpu_start = gpu_t,
            .gpu_end = gpu_t + busy,
            .period_ns = 1.0,
            .host_submit_ns = @intCast(host_start - @as(i64, @intCast(lead))),
            .host_wait_ns = @intCast(
                host_start + @as(i64, @intCast(busy + random.uintLessThan(u64, jitter_max_ns))),
            ),
        };
        gpu_t += busy + random.uintLessThan(u64, 500);
    }

    return c;
}

fn checkAnchor(_: void, c: AnchorCase) anyerror!void {
    const records = c.records[0..c.n];
    for (records, 0..) |rec, i| {
        const p = hostInterval(records, i);
        // Every record sits inside its host window, at its device duration.
        try testing.expect(p.start_ns >= @as(i64, @intCast(rec.host_submit_ns)));
        try testing.expect(p.end_ns <= @as(i64, @intCast(rec.host_wait_ns)));
        try testing.expect(p.end_ns - p.start_ns == rec.gpuEndNs() - rec.gpuStartNs());
        // With an offset that never falls, the interval is late by less than the wait jitter.
        if (c.monotone) {
            const err = p.start_ns - c.true_start[i];
            try testing.expect(err >= 0);
            try testing.expect(err < @as(i64, @intCast(jitter_max_ns)));
        }
    }
}

test "anchor: each record sits in its host window and follows a drifting device clock" {
    try prop.forAll(
        .{ .site = 0x5350_0003, .max_size = 47 },
        {},
        AnchorCase,
        genAnchor,
        checkAnchor,
    );
}

test "folded: one line per runner, label, and op key with the summed time" {
    var ph: [OpKey.count]u64 = @splat(0);
    ph[@intFromEnum(OpKey.ofOp(.attention))] = 300;
    ph[@intFromEnum(OpKey.of(.initOne(.attention), .split))] = 20;
    ph[@intFromEnum(OpKey.ofOp(.lm_head))] = 50;
    const recs = [_]GpuRecord{
        .{
            .runner = .llama_fused,
            .label = .decode,
            .gpu_start = 0,
            .gpu_end = 350,
            .period_ns = 1,
            .host_submit_ns = 1,
            .host_wait_ns = 400,
            .op_ns = ph,
        },
        .{
            .runner = .llama_fused,
            .label = .decode,
            .gpu_start = 400,
            .gpu_end = 750,
            .period_ns = 1,
            .host_submit_ns = 1,
            .host_wait_ns = 800,
            .op_ns = ph,
        },
        .{
            .runner = .staging,
            .label = .copy,
            .gpu_start = 0,
            .gpu_end = 40,
            .period_ns = 2,
            .host_submit_ns = 1,
            .host_wait_ns = 100,
        },
    };
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeFolded(&w, &recs);
    try testing.expectEqualStrings(
        "llama_fused;decode;attention;attention 600\n" ++
            "llama_fused;decode;matmul;lm_head 100\n" ++
            "llama_fused;decode;attention;attention;split 40\n" ++
            "staging;copy 80\n",
        w.buffered(),
    );
}

test "tickSpan: wrap, step back, equal ticks, and normal order" {
    const narrow: u64 = (1 << 36) - 1;
    const wide: u64 = std.math.maxInt(u64);
    const Case = struct {
        name: []const u8,
        ts: []const u64,
        mask: u64,
        start: u64,
        end: u64,
        backsteps: u32,
        deltas: []const u64,
    };
    const cases = [_]Case{
        .{
            .name = "normal order",
            .ts = &.{ 100, 150, 400 },
            .mask = wide,
            .start = 100,
            .end = 400,
            .backsteps = 0,
            .deltas = &.{ 50, 250 },
        },
        .{
            .name = "equal ticks",
            .ts = &.{ 7, 7, 7 },
            .mask = wide,
            .start = 7,
            .end = 7,
            .backsteps = 0,
            .deltas = &.{ 0, 0 },
        },
        // A 36-bit counter wraps between the stamps. The high bits are noise.
        .{
            .name = "wrap on a narrow counter",
            .ts = &.{ (0xab << 36) | (narrow - 9), (0xcd << 36) | 5, 20 },
            .mask = narrow,
            .start = narrow - 9,
            .end = narrow - 9 + 30,
            .backsteps = 0,
            .deltas = &.{ 15, 15 },
        },
        // The middle stamp reads 10 ticks before the first one.
        .{
            .name = "step back on a 64-bit counter",
            .ts = &.{ 1000, 990, 1100 },
            .mask = wide,
            .start = 1000,
            .end = 1110,
            .backsteps = 1,
            .deltas = &.{ 0, 110 },
        },
        // The end stamp reads before the start, which makes the raw delta wrap.
        .{
            .name = "end before start on a 64-bit counter",
            .ts = &.{ 1000, 990 },
            .mask = wide,
            .start = 1000,
            .end = 1000,
            .backsteps = 1,
            .deltas = &.{0},
        },
        .{
            .name = "step back on a narrow counter",
            .ts = &.{ 50, 40 },
            .mask = narrow,
            .start = 50,
            .end = 50,
            .backsteps = 1,
            .deltas = &.{0},
        },
    };
    for (cases) |case| {
        errdefer std.debug.print("tickSpan case: {s}\n", .{case.name});

        var deltas: [4]u64 = undefined;
        const got = tickSpan(case.ts, case.mask, deltas[0 .. case.ts.len - 1]);
        try testing.expectEqual(case.start, got.start);
        try testing.expectEqual(case.end, got.end);
        try testing.expectEqual(case.backsteps, got.backsteps);
        try testing.expectEqualSlices(u64, case.deltas, deltas[0 .. case.ts.len - 1]);
        // The staging record passes no delta buffer and reads the same span.
        try testing.expectEqual(got, tickSpan(case.ts, case.mask, null));
    }
}
