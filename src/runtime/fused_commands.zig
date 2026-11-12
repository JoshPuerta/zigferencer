//! The command state of one fused runner, and the barriers and copies that every fused runner
//! records.
//!
//! Holds. `FusedCommands` holds the command buffer of every submit except a decode step, the decode
//! ring, the four host stages, the GPU profile, and the decode host times. It allocates and frees
//! the four stages. The plain functions record a barrier, a copy, or the KV commit, and check a
//! plan against the device budget.
//!
//! One value. A fused runner and its layer kernel set record into one command buffer through the
//! same barriers and copies. So one value holds that state, and one file holds those functions. A
//! barrier or copy name states both ends of the dependency, so one name never covers two hazards.
//!
//! Decode ring. A submit of another command buffer invalidates the speculative decode record. So
//! every submit here frees the ring first. A recording never outlives a change to what it reads.
//!
//! Comptime. A build gate refuses non-Vulkan builds, and `core_log.hot` removes the trace code from
//! builds without hot tracing.

const std = @import("std");
const build_options = @import("build_options");
const vk = @import("vulkan");

const vulkan_ctx = @import("vulkan_ctx.zig");
const kernels = @import("../ops_vulkan/kernels.zig");
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const core_log = @import("../core/log.zig");
const span = @import("../core/span.zig");
const step_stats = @import("../core/step_stats.zig");
const gpu_profile = @import("gpu_profile.zig");
const decode_ring = @import("decode_ring.zig");

const log = std.log.scoped(.runner);
const log_dispatch = std.log.scoped(.dispatch);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/fused_commands.zig was included in a non-vulkan build.");
    }
}

// -- types ------------------------------------------------------------------

pub const FusedCommands = struct {
    ctx: *const vulkan_ctx.Context,
    /// The buffer of every submit except a decode step: prefill, `stepBatch`, profiler, probes.
    cmd_buffer: vk.CommandBuffer,
    /// The decode steps, over two own buffers. Every other submit invalidates it first.
    decode: decode_ring.DecodeRing,
    /// The host rows of one step: token embeddings or the residual rows of a hybrid.
    stage_in: kernels.HostBuffer,
    /// The logits rows of one step.
    stage_out: kernels.HostBuffer,
    /// The host-to-device stage of a slot restore. The first restore sizes it.
    kv_swap_stage: kernels.HostBuffer,
    /// The device-to-host stage of a slot save. The first save sizes it.
    kv_readback_stage: kernels.HostBuffer,
    /// The chunk of one staged copy.
    swap_chunk_bytes: u64,
    /// The timestamps of `cmd_buffer`. The profile commands raise it to `phases` for one submit.
    prof: gpu_profile.GpuProfile,
    /// The runner kind of the GPU records. A suffix runner records for the hybrid that owns it.
    gpu_runner: span.GpuRunner,
    /// Host time of single-row decode. `generate.decode` clears it before a run, reads it after.
    host: step_stats.DecodeHost,
    /// True while `cmd_buffer` holds a resubmittable decode recording. Only bench hooks set it.
    probe_recorded: bool,

    /// The value before `initRecording`: empty stages and no command buffer.
    pub fn init(ctx: *const vulkan_ctx.Context, gpu_runner: span.GpuRunner) FusedCommands {
        return .{
            .ctx = ctx,
            .cmd_buffer = .null_handle,
            .decode = undefined, // set by initRecording, after the descriptor sets
            .stage_in = .{ .ctx = ctx },
            .stage_out = .{ .ctx = ctx },
            .kv_swap_stage = .{ .ctx = ctx },
            .kv_readback_stage = .{ .ctx = ctx },
            .swap_chunk_bytes = limits.swap_stage_chunk_bytes,
            .prof = undefined, // set by initRecording
            .gpu_runner = gpu_runner,
            .host = .{},
            .probe_recorded = false,
        };
    }

    /// Allocate the stage of the step rows and the stage of the logits rows. A failure leaves
    /// `deinit` to free what it allocated.
    pub fn allocStages(self: *FusedCommands, in_bytes: u64, out_bytes: u64) !void {
        std.debug.assert(in_bytes >= 1);
        std.debug.assert(out_bytes >= 1);

        try self.stage_in.ensure(in_bytes);
        try self.stage_out.ensure(out_bytes);
    }

    /// Allocate the command buffer, the decode ring, and the profile. The runner calls it after
    /// it binds the last descriptor set, because a ring recording reads the sets.
    pub fn initRecording(self: *FusedCommands) !void {
        std.debug.assert(self.cmd_buffer == .null_handle);

        // Fill the fields only when all three exist, so `deinit` reads a whole recording or none.
        const ctx = self.ctx;
        const cmd_buffer = try kernels.allocateOnePrimaryCmdBuffer(ctx);
        errdefer ctx.vkd.freeCommandBuffers(
            ctx.device,
            ctx.command_pool,
            &[_]vk.CommandBuffer{cmd_buffer},
        );

        var decode = try decode_ring.DecodeRing.init(ctx, self.gpu_runner);
        errdefer decode.deinit();

        self.prof = try gpu_profile.GpuProfile.init(ctx, span.gpuLevel(), span.gpuLevel());
        self.decode = decode;
        self.cmd_buffer = cmd_buffer;
        std.debug.assert(self.cmd_buffer != .null_handle);
    }

    /// Free the four stages, and what `initRecording` allocated when it ran. An empty stage frees
    /// nothing, so the error path of a runner init calls it at any point.
    pub fn deinit(self: *FusedCommands) void {
        const ctx = self.ctx;
        if (self.cmd_buffer != .null_handle) {
            self.prof.deinit();
            self.decode.deinit();
            const cmds_to_free = [_]vk.CommandBuffer{self.cmd_buffer};
            ctx.vkd.freeCommandBuffers(ctx.device, ctx.command_pool, &cmds_to_free);
            self.cmd_buffer = .null_handle;
        }

        self.kv_readback_stage.deinit();
        self.kv_swap_stage.deinit();
        self.stage_out.deinit();
        self.stage_in.deinit();
    }

    /// Free the decode ring before a submit that is not a decode step.
    pub fn invalidateDecode(self: *FusedCommands) void {
        self.decode.invalidate(&self.host);
    }

    /// Submit `cmd_buffer`, wait for the queue, and read its timestamps back.
    /// The host times exist only when the buffer wrote timestamps or the span
    /// recorder runs, so the off path reads no clock.
    pub fn submitWait(self: *FusedCommands, label: span.GpuLabel) !void {
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        self.invalidateDecode();
        const timed = self.prof.level != .off or span.enabled();
        const t0: std.Io.Timestamp = if (timed) core_log.now() else undefined;
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        const t1: std.Io.Timestamp = if (timed) core_log.now() else undefined;
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
        if (!timed) return;

        const t2 = core_log.now();
        span.record(.gpu_submit, t0, t1);
        span.record(.gpu_wait, t1, t2);
        _ = try self.prof.collectInto(self.gpu_runner, label, t0, t2);
    }

    /// The stage of the direction of `side`, sized for its blob. `blob_bytes_max` is the blob at
    /// the whole window, which bounds the stage.
    pub fn slotStage(
        self: *FusedCommands,
        side: kernels.StageSide,
        blob_bytes_max: u64,
    ) !*kernels.HostBuffer {
        const stage = switch (side) {
            .to_host => &self.kv_readback_stage,
            .to_device => &self.kv_swap_stage,
        };
        const bytes: u64 = switch (side) {
            .to_host => |blob| blob.len,
            .to_device => |blob| blob.len,
        };
        std.debug.assert(bytes <= blob_bytes_max);

        try stage.ensure(@min(self.swap_chunk_bytes, bytes));
        kv_budget.assertStageCap(stage.capacity, self.swap_chunk_bytes, blob_bytes_max);
        return stage;
    }

    /// Move one slot blob through `stage` in chunks of `swap_chunk_bytes`. `copies` lists the
    /// whole blob in blob order, with `stage` as the host buffer.
    pub fn walkSlot(
        self: *FusedCommands,
        stage: *const kernels.HostBuffer,
        copies: []const kernels.Copy,
        side: kernels.StageSide,
    ) !kernels.SwapSplit {
        switch (side) {
            .to_host => std.debug.assert(stage == &self.kv_readback_stage),
            .to_device => std.debug.assert(stage == &self.kv_swap_stage),
        }

        self.invalidateDecode();
        return kernels.walkStage(self.ctx, copies, stage, side, self.swap_chunk_bytes);
    }

    /// Submit `cmd_buffer` and wait for the queue, with no timestamp readback. The profile
    /// commands and the bench probes read their own clocks.
    pub fn submitWaitUntimed(self: *FusedCommands) !void {
        const cmds = [_]vk.CommandBuffer{self.cmd_buffer};
        const submits = [_]vk.SubmitInfo{
            .{ .command_buffer_count = 1, .p_command_buffers = &cmds },
        };
        self.invalidateDecode();
        try self.ctx.vkd.queueSubmit(self.ctx.queue, &submits, .null_handle);
        try self.ctx.vkd.queueWaitIdle(self.ctx.queue);
    }
};

/// The K and V buffers of one commit side.
pub const KvPair = struct { k: vk.Buffer, v: vk.Buffer };

/// What `recordKvCommit` reads from a runner. The runner builds it for each record.
pub const KvCommit = struct {
    /// Narrows the f32 scratch, only at a non-f32 `kv_dtype`.
    kernel: ?*kernels.KvCommitF16,
    /// The one narrowing set: the f32 scratch in, the narrowed scratch out.
    set: vk.DescriptorSet,
    kv_dtype: dtype.KvDtype,
    /// Elements of one KV row: `n_kv_heads * head_dim`.
    row_elems: u32,
    /// Rows of the commit scratch: `prefill_batch_max`.
    rows_max: u32,
    /// The f32 rows that rope and the V matmul write.
    scratch: KvPair,
    /// The rows of `kv_dtype` that the kernel writes. Zero capacity at f32.
    narrowed: KvPair,
};

/// The device bytes that one fused runner allocates, for `preflightVram`. Every figure prices
/// the layer range of the runner.
pub const VramPreflight = struct {
    weights: u64,
    /// The KV regions of every slot.
    kv: u64,
    /// The recurrent state of a hybrid family. Null for a family whose every layer holds KV.
    recurrent_state: ?u64 = null,
    logits: u64,
    scratch: u64,
    /// The cost of one slot, which names the largest window that fits.
    cost: kv_budget.KvCostModel,
    n_slots: u32,
    window: u32,
    /// True when the range starts past layer 0. The refusal then names the suffix weights.
    suffix: bool = false,
};

// -- api --------------------------------------------------------------------

pub fn copyHostToDevice(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    dst: vk.Buffer,
    size: vk.DeviceSize,
) void {
    const regions = [_]vk.BufferCopy{.{ .src_offset = 0, .dst_offset = 0, .size = size }};
    vkd.cmdCopyBuffer(cb, src, dst, &regions);
}

pub fn copyDeviceToDevice(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    src_off: vk.DeviceSize,
    dst: vk.Buffer,
    dst_off: vk.DeviceSize,
    size: vk.DeviceSize,
) void {
    const regions = [_]vk.BufferCopy{
        .{ .src_offset = src_off, .dst_offset = dst_off, .size = size },
    };
    vkd.cmdCopyBuffer(cb, src, dst, &regions);
}

/// A full-range copy. The caller records `barrierShaderToTransfer` first.
pub fn copyDeviceToHost(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    dst: vk.Buffer,
    size: vk.DeviceSize,
) void {
    const regions = [_]vk.BufferCopy{.{ .src_offset = 0, .dst_offset = 0, .size = size }};
    vkd.cmdCopyBuffer(cb, src, dst, &regions);
}

/// Barrier between two SSBO-touching dispatches (shader write -> shader read/write).
pub fn barrierShader(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mem: vk.MemoryBarrier = .{
        .src_access_mask = .{ .shader_write_bit = true },
        .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true },
    };
    const mems = [_]vk.MemoryBarrier{mem};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .compute_shader_bit = true },
        .{ .compute_shader_bit = true },
        .{},
        &mems,
        null,
        null,
    );
}

/// Barrier between two transfers, for example the two-hop bounce that compacts a buffer's suffix to
/// its front.
pub fn barrierTransferToTransfer(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mem: vk.MemoryBarrier = .{
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_access_mask = .{ .transfer_read_bit = true },
    };
    const mems = [_]vk.MemoryBarrier{mem};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .transfer_bit = true },
        .{ .transfer_bit = true },
        .{},
        &mems,
        null,
        null,
    );
}

/// Barrier from a transfer-write (cmdCopyBuffer) to a shader read.
pub fn barrierTransferToShader(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mem: vk.MemoryBarrier = .{
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_access_mask = .{ .shader_read_bit = true, .shader_write_bit = true },
    };
    const mems = [_]vk.MemoryBarrier{mem};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .transfer_bit = true },
        .{ .compute_shader_bit = true },
        .{},
        &mems,
        null,
        null,
    );
}

/// Barrier from a shader write to a transfer read (preparing for readback).
pub fn barrierShaderToTransfer(vkd: vk.DeviceWrapper, cb: vk.CommandBuffer) void {
    const mem: vk.MemoryBarrier = .{
        .src_access_mask = .{ .shader_write_bit = true },
        .dst_access_mask = .{ .transfer_read_bit = true },
    };
    const mems = [_]vk.MemoryBarrier{mem};
    vkd.cmdPipelineBarrier(
        cb,
        .{ .compute_shader_bit = true },
        .{ .transfer_bit = true },
        .{},
        &mems,
        null,
        null,
    );
}

/// Record the narrowing of the `rows` committed rows of one layer, and the barrier to the copies.
/// Returns the pair that the copies read: the f32 scratch at f32, and the narrowed one otherwise.
pub fn recordKvCommit(
    vkd: vk.DeviceWrapper,
    cb: vk.CommandBuffer,
    commit: KvCommit,
    rows: u32,
) KvPair {
    std.debug.assert(rows >= 1);
    std.debug.assert(rows <= commit.rows_max);
    std.debug.assert((commit.kernel == null) == (commit.kv_dtype == .f32));

    if (commit.kernel) |k| {
        if (comptime core_log.hot) {
            log_dispatch.debug("  kv commit ({t}): rows {d}", .{ commit.kv_dtype, rows });
        }

        k.recordInto(cb, commit.set, rows * commit.row_elems);
    }

    barrierShaderToTransfer(vkd, cb);
    if (commit.kv_dtype == .f32) {
        return commit.scratch;
    }

    return commit.narrowed;
}

/// Refuse a plan past the usable share of `available`, the live device-local budget, on `driver`
/// before the first allocation. The refusal names the largest window that fits, from the same
/// budget and reservations.
pub fn preflightVram(
    plan: kv_budget.KvPlan,
    vram: VramPreflight,
    driver: limits.Driver,
    available: u64,
) error{VramBudgetExceeded}!void {
    // The plan states these reservations. Its logits figure can exceed the rows here: a budget
    // caller reserves the width before the planner resolves the slot count.
    const recurrent: u64 = vram.recurrent_state orelse 0;
    std.debug.assert(plan.vram.weights == vram.weights);
    std.debug.assert(plan.vram.kv == vram.kv + recurrent);
    std.debug.assert(plan.vram.logits >= vram.logits);
    std.debug.assert(plan.vram.scratch == vram.scratch);
    std.debug.assert(vram.cost.kv_bytes_per_token * vram.window * vram.n_slots == vram.kv);

    const mib = 1024 * 1024;
    const planned: u64 = vram.weights + vram.kv + recurrent + vram.logits + vram.scratch;
    log.debug(
        "vram preflight: planned {d} MiB (weights {d}, kv {d}, recurrent state {d}, " ++
            "scratch {d}), available {d} MiB, usable {d} MiB ({t} row)",
        .{
            planned / mib,
            vram.weights / mib,
            vram.kv / mib,
            recurrent / mib,
            vram.scratch / mib,
            available / mib,
            limits.vramUsable(driver, available) / mib,
            driver,
        },
    );
    if (limits.vramPlanFits(driver, planned, available)) return;

    const reserved: u64 = vram.weights + vram.logits + vram.scratch;
    var refusal_buf: [kv_budget.vram_refusal_text_max]u8 = undefined;
    log.err("{s}", .{kv_budget.formatVramRefusal(&refusal_buf, .{
        .planned = planned,
        .available = available,
        .driver = driver,
        .weights = vram.weights,
        .kv = vram.kv,
        .recurrent_state = vram.recurrent_state,
        .n_slots = vram.n_slots,
        .window = vram.window,
        .fit = kv_budget.largestFittingWindow(
            driver,
            available,
            reserved,
            vram.cost,
            vram.n_slots,
            vram.window,
        ),
        .suffix = vram.suffix,
    })});
    return error.VramBudgetExceeded;
}

// -- tests ------------------------------------------------------------------

/// A plan of one family with no recurrent state, and the preflight value that prices it.
fn testPreflight() struct { plan: kv_budget.KvPlan, vram: VramPreflight } {
    const vram: VramPreflight = .{
        .weights = 1000,
        .kv = 2 * 64 * 4,
        .logits = 100,
        .scratch = 50,
        .cost = .{ .kv_bytes_per_token = 4 },
        .n_slots = 2,
        .window = 64,
    };
    const plan: kv_budget.KvPlan = .{
        .vram = .{ .weights = 1000, .kv = 2 * 64 * 4, .logits = 100, .scratch = 50 },
        .ram = .{ .blobs = 0, .prefix = 0 },
        .n_slots = 2,
        .n_streams = 1,
        .window = 64,
    };
    return .{ .plan = plan, .vram = vram };
}

test "preflightVram accepts a plan at the usable share of the budget" {
    const t = testPreflight();
    const planned: u64 = 1000 + 2 * 64 * 4 + 100 + 50;

    // The smallest budget whose usable share holds the plan, on each driver.
    for (std.enums.values(limits.Driver)) |driver| {
        const share = limits.vram_shares.get(driver);
        const per_unit = std.math.divCeil(u64, planned, share.num) catch unreachable;
        const available: u64 = per_unit * share.den;
        try std.testing.expect(limits.vramPlanFits(driver, planned, available));
        try preflightVram(t.plan, t.vram, driver, available);
    }
}

test "preflightVram counts the recurrent state on top of the KV regions" {
    var t = testPreflight();
    t.vram.recurrent_state = 300;
    t.plan.vram.kv += 300;
    const planned: u64 = 1000 + 2 * 64 * 4 + 300 + 100 + 50;
    const available: u64 = planned * 2;
    try std.testing.expect(limits.vramPlanFits(.other, planned, available));
    try preflightVram(t.plan, t.vram, .other, available);
}
