//! One-shot submits on the staging command buffer of the context, and the staged walk of a KV
//! swap.
//!
//! One-shot submits. The paravirtual driver leaks part of a command buffer when it is freed. So
//! every one-shot submit reuses the one staging buffer of the context. The wait before
//! `submitOnce` returns keeps that buffer idle at the next entry. A failed wait can leave the
//! buffer pending, and a reset of a pending buffer is invalid, so every later call returns that
//! error. The Vulkan host thread is single, and the busy flag asserts that no call nests.
//!
//! Staged swaps. Evict, restore, and the prefix snapshot all walk one loop, `walkStage`. The stage
//! holds one chunk of at most `chunk_bytes`. The walk takes `kv_budget.ChunkWalk.count` trips.
//! Each trip clips the full copy list to its byte range, submits, and waits. A save copies
//! the stage out after the wait, and a restore fills the stage before the submit.

const std = @import("std");
const vk = @import("vulkan");
const kv_budget = @import("../../core/kv_budget.zig");
const core_log = @import("../../core/log.zig");
const span = @import("../../core/span.zig");
const gpu_profile = @import("../../runtime/gpu_profile.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");
const log = std.log.scoped(.gpu);

const allocateOnePrimaryCmdBuffer = @import("pipeline.zig").allocateOnePrimaryCmdBuffer;
const createSingleSetPool = @import("pipeline.zig").createSingleSetPool;
const allocateOneSet = @import("pipeline.zig").allocateOneSet;
const HostBuffer = @import("memory.zig").HostBuffer;
const KvStore = @import("memory.zig").KvStore;

// -- types ------------------------------------------------------------------

/// The standalone-submit resources a kernel owns on top of its `Pipeline`: a single-set descriptor
/// pool, the set, and a reusable primary command buffer. Only a kernel with a self-contained
/// `run()` carries one.
pub const OneShot = struct {
    desc_pool: vk.DescriptorPool,
    desc_set: vk.DescriptorSet,
    cmd_buffer: vk.CommandBuffer,

    pub const InitError =
        vk.DeviceWrapper.CreateDescriptorPoolError ||
        vk.DeviceWrapper.AllocateDescriptorSetsError ||
        vk.DeviceWrapper.AllocateCommandBuffersError;

    pub fn init(
        ctx: *const vulkan_ctx.Context,
        set_layout: vk.DescriptorSetLayout,
        binding_count: u32,
    ) InitError!OneShot {
        const desc_pool = try createSingleSetPool(ctx, binding_count);
        errdefer ctx.vkd.destroyDescriptorPool(ctx.device, desc_pool, null);

        const desc_set = try allocateOneSet(ctx, desc_pool, set_layout);
        const cmd_buffer = try allocateOnePrimaryCmdBuffer(ctx);

        return .{
            .desc_pool = desc_pool,
            .desc_set = desc_set,
            .cmd_buffer = cmd_buffer,
        };
    }

    pub fn deinit(self: OneShot, ctx: *const vulkan_ctx.Context) void {
        // The command buffer came from ctx.command_pool, so free it for the other kernels of that
        // pool. Destroying the descriptor pool frees the set.
        const cmds_to_free = [_]vk.CommandBuffer{self.cmd_buffer};
        ctx.vkd.freeCommandBuffers(ctx.device, ctx.command_pool, &cmds_to_free);
        ctx.vkd.destroyDescriptorPool(ctx.device, self.desc_pool, null);
    }
};

/// One staged copy: a source buffer, a destination buffer, a size, and the two
/// offsets.
pub const Copy = struct {
    src: vk.Buffer,
    dst: vk.Buffer,
    size: vk.DeviceSize,
    src_offset: vk.DeviceSize = 0,
    dst_offset: vk.DeviceSize = 0,
};

/// The phases of one one-shot submit, in order. The failure line names one.
pub const SubmitPhase = enum { reset, record, submit, wait };

/// The kind of work of one one-shot submit. The debug line prints it, so a
/// device log can tell the sites apart.
pub const SubmitLabel = enum { copy, fill, clone, dispatch };

/// What a recorder put into the command buffer, for the debug line. A
/// dispatch or a fill records no copy.
pub const Recorded = struct {
    copies: u32 = 0,
    bytes: u64 = 0,
};

pub const SubmitError = vk.DeviceWrapper.ResetCommandBufferError ||
    vk.DeviceWrapper.BeginCommandBufferError ||
    vk.DeviceWrapper.EndCommandBufferError ||
    vk.DeviceWrapper.QueueSubmitError ||
    vk.DeviceWrapper.QueueWaitIdleError;

/// Records one `cmdCopyBuffer` per copy, in list order, with no barrier.
pub const CopyRecorder = struct {
    ctx: *const vulkan_ctx.Context,
    copies: []const Copy,

    pub fn record(self: CopyRecorder, cb: vk.CommandBuffer) Recorded {
        var bytes: u64 = 0;
        for (self.copies) |c| {
            const regions = [_]vk.BufferCopy{
                .{ .src_offset = c.src_offset, .dst_offset = c.dst_offset, .size = c.size },
            };
            self.ctx.vkd.cmdCopyBuffer(cb, c.src, c.dst, &regions);
            bytes += c.size;
        }

        return .{ .copies = @intCast(self.copies.len), .bytes = bytes };
    }
};

/// The two costs of one staged KV swap. `dma_ms` covers the `runStagingCopies` submit and wait, and
/// `memcpy_ms` covers the copy through the mapped `HostBuffer`. A bench reports the two apart,
/// because they sit under different roofs and only the split names the side that costs.
pub const SwapSplit = struct {
    dma_ms: f64,
    memcpy_ms: f64,
};

/// The host side of one staged swap.
pub const StageSide = union(enum) {
    /// A save: the device bytes land in this blob.
    to_host: []u8,
    /// A restore: this blob lands on the device.
    to_device: []const u8,
};

// -- state ------------------------------------------------------------------

/// One-shot submits this process started. It numbers the debug and failure
/// lines, so a device log can tie a failure to a submit count.
var staging_submits: std.atomic.Value(u64) = .init(0);

/// True while a `submitOnce` call holds the staging buffer. It is process-wide,
/// because every caller holds its context through a `*const` pointer.
var staging_busy: std.atomic.Value(bool) = .init(false);

/// The error of a failed wait. Every later submit returns it.
var staging_wait_error: ?SubmitError = null;

// -- api --------------------------------------------------------------------

/// The number of one-shot submits this process started.
pub fn stagingSubmitCount() u64 {
    return staging_submits.load(.monotonic);
}

/// Submit one command buffer of work on the staging buffer of `ctx`, and block until the queue is
/// idle. `recorder.record(cb)` records the work and returns a `Recorded`. On return the GPU has
/// finished.
pub fn submitOnce(
    ctx: *const vulkan_ctx.Context,
    label: SubmitLabel,
    recorder: anytype,
) SubmitError!void {
    // Take the staging buffer, or refuse after a failed wait.
    const cmd = ctx.staging_cmd;
    std.debug.assert(cmd != .null_handle);
    const was_busy = staging_busy.swap(true, .acquire);
    std.debug.assert(!was_busy);
    defer staging_busy.store(false, .release);

    if (staging_wait_error) |err| {
        log.err("staging: submit refused, an earlier wait failed: {t}", .{err});
        return err;
    }

    // Number this submit and start the laps.
    const number = staging_submits.fetchAdd(1, .monotonic) + 1;
    var phase: SubmitPhase = .reset;
    errdefer |err| log.err("staging: submit {d} failed in phase {t}: {t}", .{ number, phase, err });

    var lap: [@typeInfo(SubmitPhase).@"enum".fields.len]f64 = @splat(0);

    // Reset the staging buffer.
    var t = core_log.now();
    const t0 = t;
    try ctx.vkd.resetCommandBuffer(cmd, .{});
    lap[@intFromEnum(SubmitPhase.reset)] = core_log.msSince(t);

    // Record the work, between timestamps when the pool exists.
    phase = .record;
    t = core_log.now();
    const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
    try ctx.vkd.beginCommandBuffer(cmd, &begin);
    // The pool exists only at a timestamp level, so the off path records no
    // timestamp command. Both slots reset before their first write.
    const stamps = ctx.staging_query_pool;
    if (stamps != .null_handle) {
        ctx.vkd.cmdResetQueryPool(cmd, stamps, 0, 2);
        ctx.vkd.cmdWriteTimestamp(cmd, .{ .top_of_pipe_bit = true }, stamps, 0);
    }

    const recorded: Recorded = recorder.record(cmd);
    if (stamps != .null_handle) {
        ctx.vkd.cmdWriteTimestamp(cmd, .{ .bottom_of_pipe_bit = true }, stamps, 1);
    }

    try ctx.vkd.endCommandBuffer(cmd);
    lap[@intFromEnum(SubmitPhase.record)] = core_log.msSince(t);

    // Hand the buffer to the queue.
    phase = .submit;
    t = core_log.now();
    const t_submit = t;
    const cmds = [_]vk.CommandBuffer{cmd};
    const submits = [_]vk.SubmitInfo{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }};
    try ctx.vkd.queueSubmit(ctx.queue, &submits, .null_handle);
    lap[@intFromEnum(SubmitPhase.submit)] = core_log.msSince(t);

    // Wait for the queue, and keep a failed wait for every later call.
    phase = .wait;
    t = core_log.now();
    const t_wait = t;
    const waited = ctx.vkd.queueWaitIdle(ctx.queue);
    if (waited) |_| {} else |err| {
        staging_wait_error = err;
    }

    try waited;

    // Close the laps and record the spans.
    const t_done = core_log.now();
    lap[@intFromEnum(SubmitPhase.wait)] = msBetween(t, t_done);
    span.record(.gpu_submit, t_submit, t_wait);
    span.record(.gpu_wait, t_wait, t_done);
    if (stamps != .null_handle) {
        stagingRecord(ctx, label, t_submit, t_done);
    }

    // The words allocate and free keep the field layout that log parsers
    // read. The reset time prints as allocate, and nothing is freed.
    const line_fmt = "staging: submit {d} {t}: " ++
        "{d} copies, {d} KiB, {d:.1} ms (allocate {d:.3}, record {d:.3}, submit {d:.3}, " ++
        "wait {d:.3}, free 0.000 ms)";
    log.debug(
        line_fmt,
        .{
            number,
            label,
            recorded.copies,
            recorded.bytes >> 10,
            core_log.msSince(t0),
            lap[0],
            lap[1],
            lap[2],
            lap[3],
        },
    );
}

/// Submit one transfer that runs every copy in `copies`, through `submitOnce`. Once it returns, a
/// compute dispatch can read the destination buffers. An empty list is a legal submit that records
/// nothing.
pub fn runStagingCopies(ctx: *const vulkan_ctx.Context, copies: []const Copy) SubmitError!void {
    try submitOnce(ctx, .copy, CopyRecorder{ .ctx = ctx, .copies = copies });
}

/// Move one packed blob between the device and the host through `stage`, one chunk at a time.
/// `full` lists the copies of the whole blob in blob order, with `stage` as the host buffer, as
/// `KvStore.rowCopies` builds them.
pub fn walkStage(
    ctx: *const vulkan_ctx.Context,
    full: []const Copy,
    stage: *const HostBuffer,
    side: StageSide,
    chunk_bytes: u64,
) !SwapSplit {
    // Check the chunk and the stage.
    const to_host = side == .to_host;
    const total: u64 = switch (side) {
        inline else => |blob| blob.len,
    };

    const chunk = @min(chunk_bytes, total);
    std.debug.assert(total >= 1);
    std.debug.assert(chunk >= 1);
    std.debug.assert(full.len <= KvStore.max_segments);
    // The caller grew the stage to one chunk before it built `full`, so every
    // copy names the live stage buffer.
    std.debug.assert(stage.capacity >= chunk);
    for (full) |c| {
        std.debug.assert((if (to_host) c.dst else c.src) == stage.buffer);
    }

    // Walk the blob, one submit per chunk.
    var split: SwapSplit = .{ .dma_ms = 0, .memcpy_ms = 0 };
    var scratch: [KvStore.max_segments]Copy = undefined;
    var walk = kv_budget.ChunkWalk.init(total, chunk);
    var trips: u64 = 0;
    while (walk.next()) |piece| : (trips += 1) {
        const copies = kv_budget.clipCopies(Copy, full, to_host, piece.offset, piece.len, &scratch);
        const len: usize = @intCast(piece.len);
        const off: usize = @intCast(piece.offset);
        switch (side) {
            .to_host => |out| {
                const t_dma = core_log.now();
                try runStagingCopies(ctx, copies);
                split.dma_ms += core_log.msSince(t_dma);
                const t_cpy = core_log.now();
                @memcpy(out[off..][0..len], stage.mapped[0..len]);
                split.memcpy_ms += core_log.msSince(t_cpy);
            },
            .to_device => |blob| {
                const t_cpy = core_log.now();
                @memcpy(stage.mapped[0..len], blob[off..][0..len]);
                split.memcpy_ms += core_log.msSince(t_cpy);
                const t_dma = core_log.now();
                try runStagingCopies(ctx, copies);
                split.dma_ms += core_log.msSince(t_dma);
            },
        }
    }

    std.debug.assert(trips == kv_budget.ChunkWalk.count(total, chunk));
    return split;
}

// -- helpers ----------------------------------------------------------------

fn msBetween(t0: std.Io.Timestamp, t1: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
}

/// Read the two stamps of the one-shot buffer after its wait and hand them to
/// the span recorder. A failed read loses one trace record, never the submit.
fn stagingRecord(
    ctx: *const vulkan_ctx.Context,
    label: SubmitLabel,
    submit: std.Io.Timestamp,
    wait: std.Io.Timestamp,
) void {
    var ts: [2]u64 = undefined;
    _ = ctx.vkd.getQueryPoolResults(
        ctx.device,
        ctx.staging_query_pool,
        0,
        2,
        @sizeOf(@TypeOf(ts)),
        &ts,
        @sizeOf(u64),
        .{ .@"64_bit" = true, .wait_bit = true },
    ) catch |err| {
        log.warn("staging: timestamp read failed: {t}", .{err});
        return;
    };
    const gpu_label: span.GpuLabel = switch (label) {
        inline else => |l| @field(span.GpuLabel, @tagName(l)),
    };

    const t = span.tickSpan(&ts, gpu_profile.tickMask(ctx.timestamp_valid_bits), null);
    span.gpu(.{
        .runner = .staging,
        .label = gpu_label,
        .gpu_start = t.start,
        .gpu_end = t.end,
        .period_ns = @floatCast(ctx.physical_device_props.limits.timestamp_period),
        .host_submit_ns = span.tsNs(submit),
        .host_wait_ns = span.tsNs(wait),
        .backsteps = t.backsteps,
    });
}
