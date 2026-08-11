//! Host and device transfer cost, in two sections. `benchCrossingLatency`
//! measures the fixed price of one crossing at residual-row payloads.
//! `benchStagedCopy` measures staged-copy bandwidth: each row times one
//! `runStagingCopies` submission per repetition, in both directions, across the
//! transfer sizes that the weight uploads and the KV swaps move. The file runs
//! copies alone, with no model, no kernels, and no dispatch. Both sections need
//! a device, so `main` calls them under the runner guard.

const std = @import("std");
const z = @import("zigferencer");

const bu = @import("bench_util.zig");

// -- types ------------------------------------------------------------------

/// Transfer sizes in MiB: 266 MiB is about one Llama-8B-Q8 layer, and 128 and
/// 512 MiB are about one KV slot at window 512 and 2048.
const transfer_mib = [_]u32{ 8, 32, 64, 128, 266, 512 };

/// The sizes that also run as `batch_copies` regions in one submission, which is
/// the shape a KV slot swap emits.
const batched_mib = [_]u32{ 128, 512 };

const n_reps = 5;
const batch_copies = 64;
const bytes_per_mib: u64 = 1 << 20;

/// The single pair of buffers covers the largest row, so the sweep allocates
/// once and every row reuses them.
const max_mib: u32 = blk: {
    var m: u32 = 0;
    for (transfer_mib) |v| m = @max(m, v);
    break :blk m;
};

/// Crossing payloads in bytes: 8 KiB is one Qwen3.5-2B f32 hidden row (2048
/// floats) and 16 KiB is one 9B row (4096 floats).
const crossing_bytes = [_]u32{ 8 << 10, 16 << 10 };

/// The crossing section reports a spread rather than an average, so it needs
/// enough repetitions for a tail to appear.
const n_crossing_reps = 9;

/// The crossing section owns 64 KiB buffers of its own, because it must still
/// report on a host where the staged-copy sweep declines its 512 MiB pair.
const crossing_capacity: u64 = 64 << 10;

/// Where the crossing rows read the device buffer back. A submission that
/// records both directions puts no barrier between them, so the read region must
/// not overlap the written one.
const readback_offset: u64 = crossing_capacity / 2;

const max_crossing_bytes: u32 = blk: {
    var m: u32 = 0;
    for (crossing_bytes) |v| m = @max(m, v);
    break :blk m;
};

comptime {
    std.debug.assert(readback_offset + max_crossing_bytes <= crossing_capacity);
}

const crossing_header_fmt = "{s:<18}{s:>12}{s:>12}{s:>12}\n";
const crossing_row_fmt = "{s:<18}{d:>12.5}{d:>12.5}{d:>12.5}\n";

const Dir = enum { h2d, d2h };

/// One `runStagingCopies` submission as a `bu.timedOnce` closure. An empty
/// `copies` slice is a legal submission that records nothing.
const Submit = struct {
    ctx: *const z.vulkan_ctx.Context,
    copies: []const z.ops_vulkan.Copy,
    pub fn run(self: @This()) !void {
        try z.ops_vulkan.runStagingCopies(self.ctx, self.copies);
    }
};

/// One host-side `@memcpy` as a `bu.timedOnce` closure: the leg of a crossing
/// that never reaches the queue.
const MemCopy = struct {
    dst: []u8,
    src: []const u8,
    pub fn run(self: @This()) !void {
        @memcpy(self.dst, self.src);
    }
};

// -- api --------------------------------------------------------------------

pub fn benchStagedCopy(io: std.Io, runner: *z.vulkan_runner.Runner) !void {
    const max_bytes: u64 = @as(u64, max_mib) * bytes_per_mib;
    var host: z.ops_vulkan.HostBuffer = .{ .ctx = &runner.ctx };
    defer host.deinit();
    var device: z.ops_vulkan.DeviceBuffer = .{ .ctx = &runner.ctx };
    defer device.deinit();
    host.ensure(max_bytes) catch |e| {
        std.debug.print("\nstaged copy: skipped (host buffer {d} MiB: {t})\n", .{ max_mib, e });
        return;
    };
    device.ensure(max_bytes) catch |e| {
        std.debug.print("\nstaged copy: skipped (device buffer {d} MiB: {t})\n", .{ max_mib, e });
        return;
    };
    // The transfer never reads the payload, so one fill before the sweep keeps
    // every row moving the same bytes.
    const payload = host.sliceTyped(u8, @intCast(max_bytes));
    for (payload, 0..) |*b, i| b.* = @truncate(i);

    std.debug.print("\nzigferencer bench: host<->device staged copy bandwidth (GB/s per submission)\n", .{});
    std.debug.print("{s:>10}{s:>6}{s:>8}", .{ "size", "dir", "copies" });
    for (1..n_reps + 1) |i| {
        var label_buf: [8]u8 = undefined;
        const label = std.fmt.bufPrint(&label_buf, "rep{d}", .{i}) catch unreachable;
        std.debug.print("{s:>10}", .{label});
    }
    std.debug.print("{s:>10}\n", .{"median"});

    for (transfer_mib) |m| {
        try copyRow(io, &runner.ctx, &host, &device, m, .h2d, 1);
        try copyRow(io, &runner.ctx, &host, &device, m, .d2h, 1);
    }
    for (batched_mib) |m| {
        try copyRow(io, &runner.ctx, &host, &device, m, .h2d, batch_copies);
        try copyRow(io, &runner.ctx, &host, &device, m, .d2h, batch_copies);
    }
}

/// The crossing price at residual-row payloads: what one leg between host and
/// device costs before any bandwidth term appears. Every staged-copy row carries
/// 8 MiB or more, where the transfer time hides the fixed submit-and-wait price.
/// These rows carry one hidden row, so the floor is what is left. `main` calls
/// this section before `benchStagedCopy`, which returns early when its 512 MiB
/// pair does not fit.
///
/// A persistent-command-buffer variant of `submit_empty` (reset, record, and
/// submit per repetition) is absent. Recording a copy outside the engine needs
/// the `vk` types, and the bench module imports the engine alone. Every
/// submission row therefore holds the transient allocate-and-free cost that
/// `runStagingCopies` pays, and each one ends on `queueWaitIdle` rather than on
/// a fence.
pub fn benchCrossingLatency(io: std.Io, runner: *z.vulkan_runner.Runner) !void {
    const ctx = &runner.ctx;
    var up: z.ops_vulkan.HostBuffer = .{ .ctx = ctx };
    defer up.deinit();
    var down: z.ops_vulkan.HostBuffer = .{ .ctx = ctx };
    defer down.deinit();
    var device: z.ops_vulkan.DeviceBuffer = .{ .ctx = ctx };
    defer device.deinit();
    up.ensure(crossing_capacity) catch |e| return crossingSkip("host upload buffer", e);
    down.ensure(crossing_capacity) catch |e| return crossingSkip("host readback buffer", e);
    device.ensure(crossing_capacity) catch |e| return crossingSkip("device buffer", e);

    // No row reads a payload for its value, so one fill covers every row.
    const upload = up.sliceTyped(u8, @intCast(crossing_capacity));
    for (upload, 0..) |*b, i| b.* = @truncate(i);
    // One full-capacity upload leaves defined bytes across the device buffer, so
    // no readback row moves memory that was never written.
    const prime = [_]z.ops_vulkan.Copy{.{ .src = up.buffer, .dst = device.buffer, .size = crossing_capacity }};
    try z.ops_vulkan.runStagingCopies(ctx, &prime);

    // The host end of a crossing is an ordinary CPU allocation. Its address
    // escapes here, so the compiler keeps the map-row stores instead of eliding
    // the whole leg.
    var scratch: [max_crossing_bytes]u8 = undefined;
    for (&scratch, 0..) |*b, i| b.* = @truncate(i);
    std.mem.doNotOptimizeAway(&scratch);

    std.debug.print(
        "\nzigferencer bench: host<->device crossing latency (ms per operation, {d} reps)\n",
        .{n_crossing_reps},
    );
    std.debug.print(crossing_header_fmt, .{ "row", "median", "min", "max" });

    try crossingRow(io, "submit_empty", Submit{ .ctx = ctx, .copies = &.{} });

    for (crossing_bytes) |n| {
        var label_buf: [24]u8 = undefined;
        const to_dev = [_]z.ops_vulkan.Copy{.{ .src = up.buffer, .dst = device.buffer, .size = n }};
        try crossingRow(io, crossingLabel(&label_buf, "h2d", n), Submit{ .ctx = ctx, .copies = &to_dev });
        const to_host = [_]z.ops_vulkan.Copy{.{ .src = device.buffer, .dst = down.buffer, .size = n, .src_offset = readback_offset }};
        try crossingRow(io, crossingLabel(&label_buf, "d2h", n), Submit{ .ctx = ctx, .copies = &to_host });
    }

    // Both directions in one submission: the shape a fused segment boundary
    // emits when it hands a row down and takes the answer back.
    var fused_label: [24]u8 = undefined;
    const both = [_]z.ops_vulkan.Copy{
        .{ .src = up.buffer, .dst = device.buffer, .size = crossing_bytes[0] },
        .{ .src = device.buffer, .dst = down.buffer, .size = crossing_bytes[0], .src_offset = readback_offset },
    };
    try crossingRow(io, crossingLabel(&fused_label, "h2d+d2h", crossing_bytes[0]), Submit{ .ctx = ctx, .copies = &both });

    for (crossing_bytes) |n| {
        var label_buf: [24]u8 = undefined;
        const write: MemCopy = .{ .dst = up.sliceTyped(u8, n), .src = scratch[0..n] };
        try crossingRow(io, crossingLabel(&label_buf, "map_write", n), write);
        const read: MemCopy = .{ .dst = scratch[0..n], .src = down.sliceTyped(u8, n) };
        try crossingRow(io, crossingLabel(&label_buf, "map_read", n), read);
    }
}

// -- helpers ----------------------------------------------------------------

/// One row: `chunks` copies covering `mib` MiB in a single submission, warmed
/// once and then timed `n_reps` times.
fn copyRow(
    io: std.Io,
    ctx: *const z.vulkan_ctx.Context,
    host: *z.ops_vulkan.HostBuffer,
    device: *z.ops_vulkan.DeviceBuffer,
    mib: u32,
    dir: Dir,
    chunks: u32,
) !void {
    std.debug.assert(chunks >= 1 and chunks <= batch_copies);
    const size: u64 = @as(u64, mib) * bytes_per_mib;
    std.debug.assert(size % chunks == 0);
    const chunk: u64 = size / chunks;

    var copies: [batch_copies]z.ops_vulkan.Copy = undefined;
    for (copies[0..chunks], 0..) |*c, i| {
        const off: u64 = @as(u64, i) * chunk;
        c.* = switch (dir) {
            .h2d => .{ .src = host.buffer, .dst = device.buffer, .size = chunk, .src_offset = off, .dst_offset = off },
            .d2h => .{ .src = device.buffer, .dst = host.buffer, .size = chunk, .src_offset = off, .dst_offset = off },
        };
    }
    const submit: Submit = .{ .ctx = ctx, .copies = copies[0..chunks] };

    try submit.run();
    var gbs: [n_reps]f64 = undefined;
    for (&gbs) |*g| {
        // `runStagingCopies` waits for the queue to go idle, so the wall clock
        // around it holds the whole transfer.
        const ms = try bu.timedOnce(io, submit);
        g.* = @as(f64, @floatFromInt(size)) / (ms * 1e6);
    }
    var sorted = gbs;
    std.mem.sort(f64, &sorted, {}, std.sort.asc(f64));

    std.debug.print("{d:>6} MiB{s:>6}{d:>8}", .{ mib, @tagName(dir), chunks });
    for (gbs) |g| std.debug.print("{d:>10.2}", .{g});
    std.debug.print("{d:>10.2}\n", .{sorted[n_reps / 2]});
}

/// Every crossing row needs all three buffers, so one failure retires the whole
/// section.
fn crossingSkip(what: []const u8, e: anyerror) void {
    std.debug.print("\ncrossing latency: skipped ({s}: {t})\n", .{ what, e });
}

/// "<name> <n> KiB" into `buf`. The row print consumes the label before the next
/// call reuses the buffer.
fn crossingLabel(buf: []u8, name: []const u8, bytes: u32) []const u8 {
    return std.fmt.bufPrint(buf, "{s} {d} KiB", .{ name, bytes >> 10 }) catch unreachable;
}

/// One crossing row: warm `op` once, then time it `n_crossing_reps` times, one
/// operation per repetition. `bu.timedOnce` keeps the spread that an averaging
/// timer folds away, and the spread is the point of this section.
fn crossingRow(io: std.Io, label: []const u8, op: anytype) !void {
    try op.run();
    var ms: [n_crossing_reps]f64 = undefined;
    for (&ms) |*m| m.* = try bu.timedOnce(io, op);
    var sorted = ms;
    std.mem.sort(f64, &sorted, {}, std.sort.asc(f64));
    std.debug.print(crossing_row_fmt, .{ label, sorted[n_crossing_reps / 2], sorted[0], sorted[n_crossing_reps - 1] });
}
