//! A double-buffered host-to-device ring for plain byte segments. The producer
//! fills one mapped segment while the device uploads the other, and the ring
//! hands each finished segment back as a descriptor view. The ring knows nothing
//! about weights, layers, or quantization. A segment is bytes, so any consumer
//! that moves bytes device-ward can drive it.
//!
//! This is the only place in the engine that owns fences. Every other Vulkan
//! path submits with a null fence and blocks on `queueWaitIdle`, so no other
//! path tracks a submission. A ring cannot submit that way, because its whole
//! point is to leave one upload outstanding. So the ring carries its own fences
//! and its own command pool. The pool matters for the same reason:
//! `ctx.command_pool` is externally synchronized and shared with the wait-idle
//! paths, and recording into it while an upload is pending couples the two. The
//! ring drains every outstanding fence at `deinit`, so it destroys no handle
//! that the device still reads.
//!
//! `core/ring_fsm.zig` holds the slot state table. Every state change here
//! routes through it, so the ring and the table cannot drift apart.

const std = @import("std");
const vk = @import("vulkan");
const build_options = @import("build_options");
const ring_fsm = @import("../core/ring_fsm.zig");
const vulkan_ctx = @import("../runtime/vulkan_ctx.zig");
const kernels = @import("kernels.zig");

comptime {
    if (!build_options.use_vulkan) {
        @compileError("ops_vulkan/stream_ring.zig was included in a non-vulkan build. " ++
            "Either build with -Druntime=vulkan, or fix the caller to gate on " ++
            "build_options.use_vulkan.");
    }
}

// -- types ------------------------------------------------------------------

pub const StreamRing = struct {
    ctx: *const vulkan_ctx.Context,
    command_pool: vk.CommandPool,
    cmd_bufs: [depth]vk.CommandBuffer,
    fences: [depth]vk.Fence,
    states: [depth]ring_fsm.SlotState,
    lens: [depth]vk.DeviceSize,
    /// Zero marks a deinitialized ring, which is what makes `deinit`
    /// idempotent.
    segment_capacity: vk.DeviceSize,
    stride: vk.DeviceSize,
    host: kernels.HostBuffer,
    device_buf: kernels.DeviceBuffer,

    pub const depth = ring_fsm.depth;
    pub const Slot = u8;

    /// One acquired slot: its index, and the mapped host bytes to fill.
    pub const Acquired = struct { slot: Slot, bytes: []u8 };

    pub const InitError = error{NoSuitableMemoryType} ||
        vk.DeviceWrapper.CreateCommandPoolError ||
        vk.DeviceWrapper.AllocateCommandBuffersError ||
        vk.DeviceWrapper.CreateFenceError ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError ||
        vk.DeviceWrapper.MapMemoryError;

    pub const SubmitError = vk.DeviceWrapper.BeginCommandBufferError ||
        vk.DeviceWrapper.EndCommandBufferError ||
        vk.DeviceWrapper.QueueSubmitError;

    pub const WaitError = vk.DeviceWrapper.WaitForFencesError ||
        vk.DeviceWrapper.ResetFencesError;

    /// `segment_capacity` is the largest upload one slot can carry. The ring
    /// allocates `depth` segments of that size up front and never grows, so
    /// pick the capacity from the consumer's largest expected chunk.
    pub fn init(
        ctx: *const vulkan_ctx.Context,
        segment_capacity: vk.DeviceSize,
    ) InitError!StreamRing {
        std.debug.assert(segment_capacity > 0);

        // Each slot starts at a descriptor-bindable offset, so a consumer can
        // plug `view()` straight into a storage-buffer write.
        const alignment = ctx.buffer_limits.offset_alignment;
        const stride = alignUp(segment_capacity, alignment);
        const total: vk.DeviceSize = @as(vk.DeviceSize, depth) * stride;

        const pool_info: vk.CommandPoolCreateInfo = .{
            .flags = .{ .transient_bit = true, .reset_command_buffer_bit = true },
            .queue_family_index = ctx.queue_family,
        };
        const command_pool = try ctx.vkd.createCommandPool(ctx.device, &pool_info, null);
        errdefer ctx.vkd.destroyCommandPool(ctx.device, command_pool, null);

        var cmd_bufs: [depth]vk.CommandBuffer = @splat(.null_handle);
        const cmd_info: vk.CommandBufferAllocateInfo = .{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = depth,
        };
        try ctx.vkd.allocateCommandBuffers(ctx.device, &cmd_info, &cmd_bufs);

        var fences: [depth]vk.Fence = @splat(.null_handle);
        var created: usize = 0;
        errdefer for (fences[0..created]) |f| ctx.vkd.destroyFence(ctx.device, f, null);

        // Unsignaled at birth. The invariant the ring keeps is that a fence is
        // signaled or pending exactly when its slot is `in_flight`.
        const fence_info: vk.FenceCreateInfo = .{};
        while (created < depth) : (created += 1) {
            fences[created] = try ctx.vkd.createFence(ctx.device, &fence_info, null);
        }

        // One `ensure` each, for the whole life of the ring. `ensure` grows by
        // destroying and remapping, which invalidates a slot the device reads.
        // So the ring never calls it twice.
        var host: kernels.HostBuffer = .{ .ctx = ctx };
        errdefer host.deinit();

        try host.ensure(total);

        var device_buf: kernels.DeviceBuffer = .{ .ctx = ctx };
        errdefer device_buf.deinit();

        try device_buf.ensure(total);

        std.debug.assert(stride >= segment_capacity);
        if (alignment != 0) std.debug.assert(stride % alignment == 0);

        return .{
            .ctx = ctx,
            .command_pool = command_pool,
            .cmd_bufs = cmd_bufs,
            .fences = fences,
            .states = @splat(.free),
            .lens = @splat(0),
            .segment_capacity = segment_capacity,
            .stride = stride,
            .host = host,
            .device_buf = device_buf,
        };
    }

    /// Take the next free slot and return its mapped bytes. A null means every
    /// slot is busy. Backpressure is the caller's policy, so the ring reports
    /// the condition and does not block.
    pub fn acquire(self: *StreamRing) ?Acquired {
        std.debug.assert(self.segment_capacity > 0);

        for (self.states, 0..) |s, i| {
            if (s != .free) continue;

            const slot: Slot = @intCast(i);
            self.transition(slot, .acquire);
            const base: usize = @intCast(@as(vk.DeviceSize, slot) * self.stride);
            const bytes = self.host.mapped[base .. base + @as(
                usize,
                @intCast(self.segment_capacity),
            )];
            std.debug.assert(bytes.len == self.segment_capacity);
            return .{ .slot = slot, .bytes = bytes };
        }

        for (self.states) |s| {
            std.debug.assert(s != .free);
        }

        return null;
    }

    /// Record and submit the upload of the first `len` bytes of `slot`. The call
    /// returns as soon as the queue accepts the work. Only `wait` makes the
    /// bytes readable.
    pub fn submit(self: *StreamRing, slot: Slot, len: vk.DeviceSize) SubmitError!void {
        std.debug.assert(self.segment_capacity > 0);
        std.debug.assert(slot < depth);
        std.debug.assert(self.states[slot] == .filling);
        std.debug.assert(len > 0);
        std.debug.assert(len <= self.segment_capacity);

        const offset: vk.DeviceSize = @as(vk.DeviceSize, slot) * self.stride;
        const cmd = self.cmd_bufs[slot];

        // The pool carries `reset_command_buffer_bit`, so beginning the buffer
        // again implicitly resets last rotation's recording.
        const begin: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
        try self.ctx.vkd.beginCommandBuffer(cmd, &begin);

        const regions = [_]vk.BufferCopy{
            .{ .src_offset = offset, .dst_offset = offset, .size = len },
        };
        self.ctx.vkd.cmdCopyBuffer(cmd, self.host.buffer, self.device_buf.buffer, &regions);

        // This barrier replaces the `queueWaitIdle` the rest of the engine
        // leans on. The transfer-read part covers a consumer that stages the
        // segment back host-ward instead of dispatching on it.
        const mb = [_]vk.MemoryBarrier{.{
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true, .transfer_read_bit = true },
        }};
        self.ctx.vkd.cmdPipelineBarrier(
            cmd,
            .{ .transfer_bit = true },
            .{ .compute_shader_bit = true, .transfer_bit = true },
            .{},
            &mb,
            null,
            null,
        );
        try self.ctx.vkd.endCommandBuffer(cmd);

        const cmds = [_]vk.CommandBuffer{cmd};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, self.fences[slot]);

        // Only a queue that took the work leaves a fence pending. An error
        // above keeps the slot `filling`, which needs no drain at `deinit`.
        self.lens[slot] = len;
        self.transition(slot, .submit);
        std.debug.assert(self.states[slot] == .in_flight);
        std.debug.assert(self.inFlightCount() <= depth);
    }

    /// Block until the device finishes `slot`, then reset its fence and open
    /// the slot for reading.
    pub fn wait(self: *StreamRing, slot: Slot) WaitError!void {
        std.debug.assert(self.segment_capacity > 0);
        std.debug.assert(slot < depth);
        std.debug.assert(self.states[slot] == .in_flight);

        const waited = [_]vk.Fence{self.fences[slot]};
        const result = try self.ctx.vkd.waitForFences(
            self.ctx.device,
            &waited,
            .true,
            std.math.maxInt(u64),
        );
        // An infinite timeout cannot report `timeout`, so any result but
        // success means the wrapper or the driver broke the Vulkan rule.
        std.debug.assert(result == .success);
        try self.ctx.vkd.resetFences(self.ctx.device, &waited);

        self.transition(slot, .wait);
        std.debug.assert(self.states[slot] == .ready);
    }

    /// The uploaded bytes of `slot`, shaped for a descriptor write. The state
    /// assert is the no-consume-before-signal obligation: a slot is readable
    /// only after its fence has signaled.
    ///
    /// A consumer that overlaps its dispatch with a later reuse of the slot
    /// must record its own compute-to-transfer barrier. Every submit-and-wait
    /// consumer on this one-queue stack satisfies that obligation trivially.
    pub fn view(self: *const StreamRing, slot: Slot) vk.DescriptorBufferInfo {
        std.debug.assert(self.segment_capacity > 0);
        std.debug.assert(slot < depth);
        std.debug.assert(self.states[slot] == .ready);
        std.debug.assert(self.lens[slot] > 0);

        return .{
            .buffer = self.device_buf.buffer,
            .offset = @as(vk.DeviceSize, slot) * self.stride,
            .range = self.lens[slot],
        };
    }

    /// Give the slot back to the producer. The length goes with it, because a
    /// free slot has no view.
    pub fn release(self: *StreamRing, slot: Slot) void {
        std.debug.assert(self.segment_capacity > 0);
        std.debug.assert(slot < depth);
        std.debug.assert(self.states[slot] == .ready);

        self.transition(slot, .release);
        self.lens[slot] = 0;
        std.debug.assert(self.states[slot] == .free);
    }

    pub fn inFlightCount(self: *const StreamRing) u32 {
        var n: u32 = 0;
        for (self.states) |s| {
            if (s == .in_flight) {
                n += 1;
            }
        }

        std.debug.assert(n <= depth);
        return n;
    }

    pub fn deinit(self: *StreamRing) void {
        if (self.segment_capacity == 0) return;

        // A `free`, `filling`, or `ready` slot has no pending fence. The
        // submit path guarantees it: a failed submit leaves the slot
        // `filling` with nothing recorded.
        drain: for (self.states, 0..) |s, i| {
            if (s != .in_flight) continue;
            const waited = [_]vk.Fence{self.fences[i]};
            _ = self.ctx.vkd.waitForFences(
                self.ctx.device,
                &waited,
                .true,
                std.math.maxInt(u64),
            ) catch {
                // A lost device fails every later fence wait too. One
                // queue-wide drain is the last chance to let outstanding work
                // settle before the handles go.
                self.ctx.vkd.queueWaitIdle(self.ctx.queue) catch {};
                break :drain;
            };
        }

        var slot: Slot = 0;
        while (slot < depth) : (slot += 1) {
            if (self.states[slot] == .in_flight) {
                self.transition(slot, .wait);
            }
        }

        std.debug.assert(self.inFlightCount() == 0);

        for (self.fences) |f| {
            self.ctx.vkd.destroyFence(self.ctx.device, f, null);
        }

        // Destroying the pool frees the command buffers allocated from it.
        self.ctx.vkd.destroyCommandPool(self.ctx.device, self.command_pool, null);
        self.device_buf.deinit();
        self.host.deinit();
        self.segment_capacity = 0;
    }

    /// The single door to `states`. `core/ring_fsm.zig` decides what is legal,
    /// and an illegal move here is a caller bug rather than a runtime
    /// condition.
    fn transition(self: *StreamRing, slot: Slot, op: ring_fsm.OpTag) void {
        std.debug.assert(slot < depth);

        const from = self.states[slot];
        const to = ring_fsm.next(from, op);
        std.debug.assert(to != null);
        self.states[slot] = to.?;
        std.debug.assert(self.states[slot] != from);
    }
};

// -- helpers ----------------------------------------------------------------

fn alignUp(value: vk.DeviceSize, alignment: vk.DeviceSize) vk.DeviceSize {
    if (alignment == 0) return value;
    return ((value + alignment - 1) / alignment) * alignment;
}
