//! The decode record ring of a fused runner: two decode command buffers, one persistent fence, and
//! the `core/record_ring.zig` state over them.
//!
//! Comptime. `core_log.hot` removes the trace code from builds without hot tracing.
//!
//! Flow. A step submits the buffer that holds its key, or records the key first on a miss. While
//! the GPU runs that buffer, the step records the predicted next key into the other buffer. Then it
//! waits on the fence. The next step at the predicted key submits with no record on its critical
//! path.
//!
//! Record safety. A record writes only its command buffer. The runners bind every descriptor set at
//! init, so a record of the next step touches no memory that the running step reads. Both buffers
//! come from the context pool, which carries the reset-command-buffer flag, so a reset of one
//! buffer is legal while the other is pending.
//!
//! Submit once. Each record keeps the one-time-submit flag, and the ring submits each recording
//! once, so a hit submits the command stream that a miss records.
//!
//! Invalidation. The runner owns every other submit, on its own command buffer, and calls
//! `invalidate` before each. So a recording never outlives a change to what the record reads.
//!
//! Profiles. Each buffer holds its own `GpuProfile`, at the index of the buffer. A record writes
//! only the profile of its buffer, and `step` reads that profile back right after the fence wait.
//! The ring invariant keeps at most one buffer pending, so no record resets a pool that a pending
//! buffer writes.
//!
//! Failed waits. A failed fence wait with a live device leaves the buffer pending. The ring keeps
//! it pending and refuses every later step. An object of a pending submission must not go, so
//! the owner drains the queue before `deinit`, and a failed drain skips `deinit`.

const std = @import("std");
const vk = @import("vulkan");
const vulkan_ctx = @import("vulkan_ctx.zig");
const kernels = @import("../ops_vulkan/kernels.zig");
const gpu_profile = @import("gpu_profile.zig");
const record_ring = @import("../core/record_ring.zig");
const step_stats = @import("../core/step_stats.zig");
const core_log = @import("../core/utils/log.zig");
const span = @import("../core/span.zig");
const time = @import("../core/utils/time.zig");

const log_dispatch = std.log.scoped(.dispatch);

pub const Key = gpu_profile.DecodeKey;

pub const DecodeRing = struct {
    ctx: *const vulkan_ctx.Context,
    cbs: [2]vk.CommandBuffer,
    fence: vk.Fence,
    ring: record_ring.RecordRing(Key) = .{},
    /// The timestamps of each buffer, at the index of the buffer.
    profs: [2]gpu_profile.GpuProfile,
    runner: span.GpuRunner,
    /// The readout of the last step, or null when it wrote no timestamp.
    last: ?gpu_profile.Readout = null,

    pub fn init(ctx: *const vulkan_ctx.Context, runner: span.GpuRunner) !DecodeRing {
        const a = try kernels.allocateOnePrimaryCmdBuffer(ctx);
        errdefer ctx.vkd.freeCommandBuffers(ctx.device, ctx.command_pool, &[_]vk.CommandBuffer{a});

        const b = try kernels.allocateOnePrimaryCmdBuffer(ctx);
        errdefer ctx.vkd.freeCommandBuffers(ctx.device, ctx.command_pool, &[_]vk.CommandBuffer{b});

        const fence = try ctx.vkd.createFence(ctx.device, &.{}, null);
        errdefer ctx.vkd.destroyFence(ctx.device, fence, null);

        const at = span.gpuLevel();
        var pa = try gpu_profile.GpuProfile.init(ctx, at, at);
        errdefer pa.deinit();

        const pb = try gpu_profile.GpuProfile.init(ctx, at, at);
        return .{
            .ctx = ctx,
            .cbs = .{ a, b },
            .fence = fence,
            .profs = .{ pa, pb },
            .runner = runner,
        };
    }

    pub fn deinit(self: *DecodeRing) void {
        std.debug.assert(self.ring.pendingCount() == 0);
        std.debug.assert(self.ring.wait_error == null);

        for (&self.profs) |*p| {
            p.deinit();
        }

        self.ctx.vkd.destroyFence(self.ctx.device, self.fence, null);
        self.ctx.vkd.freeCommandBuffers(self.ctx.device, self.ctx.command_pool, &self.cbs);
    }

    /// Free both buffers. The runner calls this before any submit that is not a ring step, after
    /// `checkWait`. The discarded recordings count as wasted.
    pub fn invalidate(self: *DecodeRing, host: *step_stats.DecodeHost) void {
        host.ring.wasted += self.ring.invalidate();
    }

    /// Refuse every call after a failed wait that left a ring buffer pending.
    pub fn checkWait(self: *const DecodeRing) record_ring.WaitError!void {
        return self.ring.checkWait();
    }

    /// Apply the result of a queue wait at a drain. The queue wait covers the pending buffer, so a
    /// result that ends a submission completes it.
    pub fn drain(self: *DecodeRing, waited: record_ring.WaitError!void) void {
        if (record_ring.pendingError(waited) != null) return;

        for ([_]record_ring.Buf{ .a, .b }) |c| {
            if (self.ring.isPending(c)) {
                self.ring.complete(c);
            }
        }

        std.debug.assert(self.ring.pendingCount() == 0);
    }

    /// The level the next decode key carries.
    pub fn level(self: *const DecodeRing) gpu_profile.Level {
        std.debug.assert(self.profs[0].level == self.profs[1].level);

        return self.profs[0].level;
    }

    /// Change the timestamp level of both buffers. It drops every recording first, so no buffer is
    /// pending and no recording at the old level can be submitted. Only `bench-decode-host` calls
    /// it.
    pub fn setLevel(self: *DecodeRing, new: gpu_profile.Level, host: *step_stats.DecodeHost) !void {
        try self.checkWait();
        self.invalidate(host);
        std.debug.assert(self.ring.pendingCount() == 0);
        // Both pools grow before either level moves, so a failed reserve leaves the old level.
        for (&self.profs) |*p| {
            try p.reserve(new);
        }

        for (&self.profs) |*p| {
            p.setLevel(new);
        }

        self.last = null;
    }

    /// One decode step at `key`. `rec.recordDecode(cb, prof, key)` records into `cb` with the
    /// timestamps of `prof`. On return the GPU is done and the other buffer holds `next`, if any.
    /// The host steps and the ring counts fold into `host`.
    pub fn step(
        self: *DecodeRing,
        key: Key,
        next: ?Key,
        rec: anytype,
        host: *step_stats.DecodeHost,
    ) !void {
        try self.checkWait();
        std.debug.assert(self.ring.pendingCount() == 0);

        if (next) |k| std.debug.assert(!std.meta.eql(k, key));

        std.debug.assert(key.opts.level == self.level());
        // After a failed call no recording is trusted. A pending buffer stays pending.
        errdefer self.ring.drop();

        const vkd = self.ctx.vkd;

        // Prepare the buffer: reuse a hit, or record a miss.
        const t0 = core_log.now();
        const p = self.ring.prepare(key);
        if (p.hit) {
            host.ring.hits += 1;
        } else {
            host.ring.misses += 1;
            try self.recordInto(p.buf, key, rec, host);
        }

        const t1 = core_log.now();

        // Submit the buffer and set the fence. A failed submit leaves no buffer pending, so the
        // ring marks the buffer pending only after the queue took it.
        const fences = [_]vk.Fence{self.fence};
        try vkd.resetFences(self.ctx.device, &fences);
        const cmds = [_]vk.CommandBuffer{self.cbs[@intFromEnum(p.buf)]};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        try vkd.queueSubmit(self.ctx.queue, &submits, self.fence);
        self.ring.submit(p.buf, key);
        const t2 = core_log.now();

        // Record the predicted next key into the other buffer, which can hold it after a rewind.
        const speculated = if (next) |k| blk: {
            const c = self.ring.speculateTarget(p.buf);
            if (self.ring.holds(c, k)) break :blk {};

            host.ring.speculated += 1;
            break :blk self.recordInto(c, k, rec, host);
        } else {};

        const t3 = core_log.now();

        // Wait on the fence even when the speculative record failed. A timeout is a broken
        // driver at an infinite timeout, and the fence did not signal, so the buffer stays pending.
        const waited = vkd.waitForFences(self.ctx.device, &fences, .true, std.math.maxInt(u64));
        const result: record_ring.WaitError!void = if (waited) |r| switch (r) {
            .success => {},
            .timeout => error.WaitTimeout,
            // The wrapper returns every other result code as an error.
            else => unreachable,
        } else |err| err;
        try self.ring.endWait(p.buf, result);
        const t4 = core_log.now();
        try speculated;
        self.last = try self.profs[@intFromEnum(p.buf)].collectInto(self.runner, .decode, t1, t4);

        // Report the four host steps.
        host.record.add(time.rawElapsedNs(t0, t1));
        host.submit.add(time.rawElapsedNs(t1, t2));
        host.speculate.add(time.rawElapsedNs(t2, t3));
        host.wait.add(time.rawElapsedNs(t3, t4));
        span.record(.gpu_record, t0, t1);
        span.record(.gpu_submit, t1, t2);
        span.record(.gpu_speculate, t2, t3);
        span.record(.gpu_wait, t3, t4);
        if ((comptime core_log.hot) and core_log.enabled(.dispatch, .debug)) {
            log_dispatch.debug("step pos {d} ({s}): record {d:.1} ms, submit+wait {d:.1} ms", .{
                key.pos,
                if (p.hit) "hit" else "miss",
                time.nsToMs(time.rawElapsedNs(t0, t1)),
                time.nsToMs(time.rawElapsedNs(t1, t4)),
            });
        }

        std.debug.assert(self.ring.pendingCount() == 0);
    }

    fn recordInto(
        self: *DecodeRing,
        c: record_ring.Buf,
        key: Key,
        rec: anytype,
        host: *step_stats.DecodeHost,
    ) !void {
        if (self.ring.beginRecord(c)) {
            host.ring.wasted += 1;
        }

        try rec.recordDecode(self.cbs[@intFromEnum(c)], &self.profs[@intFromEnum(c)], key);
        self.ring.endRecord(c, key);
    }
};
