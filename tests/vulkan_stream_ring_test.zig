//! GPU gate for the streaming upload ring. The ring owns a fixed set of host
//! segments and rotates them, so the host can fill one segment while the device
//! still reads another.
//!
//! The ring is the first place in the engine that uses a VkFence, so the
//! property under test is the synchronization and not any arithmetic. Each test
//! uploads a byte pattern the host knows, then reads the device segment back
//! through the ordinary staging path and compares. A missing barrier, a
//! mis-sized copy, or a slot whose bytes the next rotation overwrote all show
//! up as a byte mismatch.
//!
//! `core/ring_fsm.zig` already proves the state table at comptime. Only
//! hardware can prove that the fence really covers the transfer, so the first
//! test replays the same anchor trace against a live device.
//!
//! The ring here holds two 4 KiB segments. The whole test costs well under
//! 32 KiB of device memory, which keeps it safe to run beside anything else.
//!
//! The tests return `error.SkipZigTest` when Vulkan cannot start.

const std = @import("std");
const z = @import("zigferencer");

const fsm = z.ring_fsm;
const StreamRing = z.ops_vulkan.StreamRing;

// -- fixtures ---------------------------------------------------------------

const segment_capacity: u64 = 4096;

fn initCtx() !z.vulkan_ctx.Context {
    return z.vulkan_ctx.Context.init(std.testing.allocator, .{}) catch |e| switch (e) {
        error.NoVulkanDevice,
        error.NoComputeQueue,
        error.InitializationFailed,
        error.IncompatibleDriver,
        error.FeatureNotPresent,
        => return error.SkipZigTest,
        else => return e,
    };
}

/// Byte `j` of upload `i`. The multiplier keeps consecutive uploads apart at
/// every offset, so a stale segment cannot pass as a fresh one.
fn patternByte(upload: usize, j: usize) u8 {
    return @truncate(upload *% 37 +% j);
}

/// Check the view of a `ready` slot. Then read the device bytes back and
/// compare them with the pattern of the upload the slot holds.
fn checkSegment(
    ctx: *const z.vulkan_ctx.Context,
    ring: *StreamRing,
    scratch: *z.ops_vulkan.HostBuffer,
    slot: StreamRing.Slot,
    upload: usize,
    len: u64,
) !void {
    const v = ring.view(slot);
    try std.testing.expectEqual(ring.device_buf.buffer, v.buffer);
    try std.testing.expectEqual(@as(u64, slot) * ring.stride, v.offset);
    try std.testing.expectEqual(len, v.range);

    const copies = [_]z.ops_vulkan.Copy{.{
        .src = v.buffer,
        .dst = scratch.buffer,
        .size = len,
        .src_offset = v.offset,
    }};
    try z.ops_vulkan.runStagingCopies(ctx, &copies);

    const got = scratch.mapped[0..@as(usize, @intCast(len))];
    for (got, 0..) |b, j| {
        const want = patternByte(upload, j);
        if (b != want) {
            std.debug.print(
                "slot {d} upload {d}: byte {d} is {d}, want {d}\n",
                .{ slot, upload, j, b, want },
            );
            return error.StreamRingReadbackMismatch;
        }
    }
}

// -- tests ------------------------------------------------------------------

test "stream_ring: the anchor trace, uploaded and read back" {
    var ctx = try initCtx();
    defer ctx.deinit();

    var scratch: z.ops_vulkan.HostBuffer = .{ .ctx = &ctx };
    defer scratch.deinit();

    try scratch.ensure(segment_capacity);

    var ring = try StreamRing.init(&ctx, segment_capacity);

    // The trace submits four times: a full segment, a partial one, a full one
    // again, and a single byte. The last two land in slots the earlier uploads
    // already used, so the readback proves the overwrite.
    const lens = [4]u64{ 4096, 1000, 4096, 1 };
    var next_upload: usize = 0;
    var slot_upload = [_]usize{ 0, 0 };

    for (fsm.anchor_trace, 1..) |step, n| {
        switch (step.op) {
            .acquire => {
                const acq = ring.acquire() orelse return error.StreamRingRefusedAcquire;
                try std.testing.expectEqual(step.slot, acq.slot);
                try std.testing.expectEqual(@as(usize, @intCast(segment_capacity)), acq.bytes.len);
                for (acq.bytes, 0..) |*b, j| {
                    b.* = patternByte(next_upload, j);
                }

                slot_upload[acq.slot] = next_upload;
                next_upload += 1;
            },
            .submit => try ring.submit(step.slot, lens[slot_upload[step.slot]]),
            .wait => {
                try ring.wait(step.slot);
                const upload = slot_upload[step.slot];
                try checkSegment(&ctx, &ring, &scratch, step.slot, upload, lens[upload]);
            },
            .release => ring.release(step.slot),
        }

        // The three points at which the trace fills the ring. The FSM test
        // pins them at comptime, and here the live ring must agree.
        if (n == 4 or n == 8 or n == 12) {
            try std.testing.expect(ring.acquire() == null);
            try std.testing.expectEqual(@as(u32, 2), ring.inFlightCount());
        }
    }

    try std.testing.expectEqual(@as(usize, 4), next_upload);
    try std.testing.expectEqual(@as(u32, 0), ring.inFlightCount());
    ring.deinit();
    ring.deinit();
}

test "stream_ring: deinit drains from any state" {
    var ctx = try initCtx();
    defer ctx.deinit();

    // Nothing ever acquired.
    var idle = try StreamRing.init(&ctx, segment_capacity);
    idle.deinit();
    idle.deinit();

    // One slot mid-fill. Nothing was recorded, so no fence is pending.
    var filling = try StreamRing.init(&ctx, segment_capacity);
    const held = filling.acquire() orelse return error.StreamRingRefusedAcquire;
    try std.testing.expectEqual(@as(StreamRing.Slot, 0), held.slot);
    filling.deinit();
    filling.deinit();

    // One slot in flight and one mid-fill. The drain must fence-wait slot 0
    // and leave slot 1 alone.
    var mixed = try StreamRing.init(&ctx, segment_capacity);
    const first = mixed.acquire() orelse return error.StreamRingRefusedAcquire;
    for (first.bytes, 0..) |*b, j| {
        b.* = patternByte(0, j);
    }

    try mixed.submit(first.slot, segment_capacity);
    const second = mixed.acquire() orelse return error.StreamRingRefusedAcquire;
    try std.testing.expectEqual(@as(StreamRing.Slot, 1), second.slot);
    try std.testing.expectEqual(@as(u32, 1), mixed.inFlightCount());
    mixed.deinit();
    mixed.deinit();
}

test "stream_ring: overlap does not corrupt" {
    var ctx = try initCtx();
    defer ctx.deinit();

    var scratch: z.ops_vulkan.HostBuffer = .{ .ctx = &ctx };
    defer scratch.deinit();

    try scratch.ensure(segment_capacity);

    var ring = try StreamRing.init(&ctx, segment_capacity);
    defer ring.deinit();

    // The ring exists for this case. The host fills the second segment while
    // the first upload is still outstanding. Only per-slot offsets keep the two
    // apart.
    const first = ring.acquire() orelse return error.StreamRingRefusedAcquire;
    for (first.bytes, 0..) |*b, j| {
        b.* = patternByte(0, j);
    }

    try ring.submit(first.slot, segment_capacity);

    const second = ring.acquire() orelse return error.StreamRingRefusedAcquire;
    for (second.bytes, 0..) |*b, j| {
        b.* = patternByte(1, j);
    }

    try ring.submit(second.slot, 2048);

    try ring.wait(first.slot);
    try ring.wait(second.slot);

    try checkSegment(&ctx, &ring, &scratch, first.slot, 0, segment_capacity);
    try checkSegment(&ctx, &ring, &scratch, second.slot, 1, 2048);

    ring.release(first.slot);
    ring.release(second.slot);
}
