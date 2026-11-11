//! The single owning handle for the Vulkan objects a compute workload needs.
//! Those are the instance, the chosen physical device, the logical device, the
//! compute queue, one command pool, and the one command buffer of every
//! one-shot submit. Every kernel and runner that issues Vulkan calls borrows
//! from this `Context`. Initialization is one shot, and teardown reverses it
//! in order.
//!
//! Comptime. A build gate stops the compiler when a non-Vulkan build includes this file. A trace
//! build loads the device functions through the dispatch list loader, and no other build holds it.
//!
//! The module compiles only under `-Druntime=vulkan`. The runtime is a
//! build-time axis, gated by `build_options.use_vulkan` and the conditional
//! re-export in `main.zig`, so CPU builds never link libvulkan or the Vulkan
//! bindings.
//!
//! Device selection (`scoreDeviceType`) ranks DISCRETE > INTEGRATED > VIRTUAL
//! and rejects `.cpu`, so a software rasterizer like llvmpipe never wins the
//! tiebreak on a system that also has real hardware.
//!
//! Software device. `Options.software` turns that rule around: it takes a CPU device and no other,
//! so a debug run with no GPU is a named choice. The subgroup-size pin stays off there, because
//! such a device holds fewer lanes. Only `linear_q8_0_subgroup.comp` reads the lanes of a subgroup,
//! and only `tools/bench_matmul.zig` runs it. Timings on this device mean nothing.
//!
//! The engine targets a translation-layer driver, and four properties of that
//! driver shape this file. It advertises Vulkan 1.2, so the code stays inside
//! 1.2. It reports subgroup sizes 32 to 64, so `init` pins 32 on every compute
//! pipeline through the subgroup-size-control extension. It exposes neither
//! 8-bit storage nor `shaderInt8`, so the quant kernels pack four quants to a
//! `uint`. It reports the spec minimum for the storage-buffer range, so a
//! caller can replace that one limit through `Options`.
//!
//! Device features. The Q8_0 kernels read f16 weight scales out of storage buffers. That needs
//! 16-bit storage and float16 arithmetic, which are core in Vulkan 1.1 and 1.2, so a
//! Vulkan11/12Features chain enables them. 8-bit storage and `shaderInt8` stay off, because the
//! driver exposes neither on RDNA2. The kernels pack the quants four to a `uint` and decode them
//! with `bitfieldExtract`, at the price of a few ALU ops. `VK_EXT_subgroup_size_control` pins a
//! required subgroup size of 32 on every compute pipeline. The driver reports 32 to 64, and
//! without the pin its shader compiler can choose wave64. RDNA2 issues a wave64 vector op in two
//! passes, so the pin forces the native wave32.
//!
//! Memory budget. `VK_EXT_memory_budget` is optional and exposes the free device-local VRAM,
//! which sizes the multi-stream KV window to the hardware. When a driver lacks it,
//! `deviceLocalBudget` returns null and the runner keeps its default window.
//!
//! One-shot submit. The translation layer makes a native command allocator for each command
//! buffer, and the paravirtual driver leaks part of it when the buffer is freed. So a one-shot
//! submit never allocates. It resets `staging_cmd`, and every submit waits before it returns, so
//! the buffer is never pending at the next reset.

const std = @import("std");
const build_options = @import("build_options");
const vk = @import("vulkan");
const limits = @import("../core/limits.zig");
const span = @import("../core/span.zig");
const kv_index = @import("../core/kv_index.zig");
const attn_split = @import("../core/attn_split.zig");
const core_log = @import("../core/log.zig");
const dispatch_list = if (core_log.hot) @import("../ops_vulkan/dispatch_list.zig") else void;

const log = std.log.scoped(.gpu);

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/vulkan_ctx.zig was included in a non-vulkan build. " ++
            "Either build with -Druntime=vulkan, " ++
            "or fix the caller to gate on build_options.use_vulkan.");
    }
}

/// The one symbol the dynamic linker resolves. Every other Vulkan function pointer loads
/// through it. `wireVulkan` in `build.zig` links `libvulkan.so.1` (or the platform equivalent).
extern fn vkGetInstanceProcAddr(
    instance: vk.Instance,
    name: [*:0]const u8,
) callconv(vk.vulkan_call_conv) vk.PfnVoidFunction;

// -- constants --------------------------------------------------------------

/// The advice of a KV region refusal, with the widest window that fits. A
/// region past the storage-buffer range also names the override.
const kv_advice_fmt = "Use --max-seq {d} or less, or fewer sessions";

const kv_range_advice_fmt = "Use --max-seq {d} or less, fewer sessions, or set " ++
    "" ++ limits.storage_range_env;
const kv_advice_max: usize = kv_range_advice_fmt.len + std.fmt.count(
    "{d}",
    .{std.math.maxInt(u32)},
);

/// The flags `pickHostType` asks for first.
pub const host_cached_flags: vk.MemoryPropertyFlags = .{
    .host_visible_bit = true,
    .host_coherent_bit = true,
    .host_cached_bit = true,
};

/// The flags `pickHostType` asks for second.
pub const host_coherent_flags: vk.MemoryPropertyFlags = .{
    .host_visible_bit = true,
    .host_coherent_bit = true,
};

// -- types ------------------------------------------------------------------

pub const InitError = error{
    NoVulkanDevice,
    NoComputeQueue,
    DeviceBufferLimitsInvalid,
    BaseFunctionsMissing,
    InstanceFunctionsMissing,
    DeviceFunctionsMissing,
} ||
    vk.BaseWrapper.CreateInstanceError ||
    vk.InstanceWrapper.EnumeratePhysicalDevicesError ||
    vk.InstanceWrapper.EnumeratePhysicalDevicesAllocError ||
    vk.InstanceWrapper.CreateDeviceError ||
    vk.DeviceWrapper.CreateCommandPoolError ||
    std.mem.Allocator.Error;

/// What a caller states when it opens a context.
pub const Options = limits.DeviceOptions;

/// The live usage and budget of one memory heap, from VK_EXT_memory_budget.
pub const HeapBudget = struct {
    usage: u64,
    budget: u64,
};

pub const PickedDevice = struct {
    device: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    /// See `scoreDeviceType`. A negative score rejects the device.
    score: i32,
};

pub const Context = struct {
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    physical_device_props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    /// True when VK_EXT_memory_budget is enabled, which makes `deviceLocalBudget` live.
    has_memory_budget: bool,
    /// True on a CPU device. A pipeline then leaves the subgroup size to the driver.
    software: bool,
    /// The limits every buffer check reads. The storage range is the override when set.
    buffer_limits: limits.BufferLimits,
    /// The storage range the driver reports, kept for the boot log and `info`.
    reported_storage_range: u64,
    device: vk.Device,
    queue: vk.Queue,
    queue_family: u32,
    /// Valid bits of a queue timestamp. A tick difference is taken modulo this width.
    timestamp_valid_bits: u32,
    command_pool: vk.CommandPool,
    /// The one command buffer of every one-shot submit.
    staging_cmd: vk.CommandBuffer,
    /// The start and end timestamp slots of `staging_cmd`. A null handle unless timestamps are on.
    staging_query_pool: vk.QueryPool,
    vkb: vk.BaseWrapper,
    vki: vk.InstanceWrapper,
    vkd: vk.DeviceWrapper,

    pub fn init(allocator: std.mem.Allocator, options: Options) InitError!Context {
        if (options.storage_range_override) |r| {
            std.debug.assert(r >= 1);
            std.debug.assert(r <= limits.storage_range_max);
        }

        // Create the instance.
        const vkb = vk.BaseWrapper.load(vkGetInstanceProcAddr);
        if (vkb.dispatch.vkCreateInstance == null) return error.BaseFunctionsMissing;

        const app_info: vk.ApplicationInfo = .{
            .p_application_name = "zigferencer",
            .application_version = vk.makeApiVersion(0, 0, 1, 0).toU32(),
            .p_engine_name = "zigferencer",
            .engine_version = vk.makeApiVersion(0, 0, 1, 0).toU32(),
            .api_version = vk.makeApiVersion(0, 1, 2, 0).toU32(),
        };
        const instance_info: vk.InstanceCreateInfo = .{
            .p_application_info = &app_info,
        };
        const instance = try vkb.createInstance(&instance_info, null);

        const vki = vk.InstanceWrapper.load(instance, vkb.dispatch.vkGetInstanceProcAddr.?);
        if (vki.dispatch.vkDestroyInstance == null) return error.InstanceFunctionsMissing;

        errdefer vki.destroyInstance(instance, null);

        // Pick the device and its compute queue family.
        const picked = try pickPhysicalDevice(vki, instance, allocator, options.software);
        if (options.software) {
            log.warn("the GPU paths run on the software device {s}. Timings mean nothing", .{
                std.mem.sliceTo(&picked.props.device_name, 0),
            });
        }

        const family = try findComputeQueueFamily(vki, picked.device, allocator);
        const queue_family = family.index;

        // Turn GPU timestamps off when the device cannot time a command buffer.
        if (span.gpuLevel() != .off and
            (family.timestamp_valid_bits == 0 or !(picked.props.limits.timestamp_period > 0)))
        {
            log.warn(
                "GPU timestamps off: the queue has {d} valid timestamp bits and a period of {d} ns",
                .{ family.timestamp_valid_bits, picked.props.limits.timestamp_period },
            );
            span.setGpuLevel(.off);
        }

        // Describe the one compute queue.
        const queue_priorities = [_]f32{1.0};
        const queue_create_infos = [_]vk.DeviceQueueCreateInfo{
            .{
                .queue_family_index = queue_family,
                .queue_count = 1,
                .p_queue_priorities = &queue_priorities,
            },
        };

        // Enable f16 storage, and pin wave32.
        var sgsc_features: vk.PhysicalDeviceSubgroupSizeControlFeatures = .{
            .subgroup_size_control = .true,
        };
        var v11_features: vk.PhysicalDeviceVulkan11Features = .{
            .storage_buffer_16_bit_access = .true,
            .p_next = @ptrCast(&sgsc_features),
        };
        var v12_features: vk.PhysicalDeviceVulkan12Features = .{
            .shader_float_16 = .true,
            .p_next = @ptrCast(&v11_features),
        };

        // Probe the optional extensions.
        const avail_exts = vki.enumerateDeviceExtensionPropertiesAlloc(
            picked.device,
            null,
            allocator,
        ) catch
            &[_]vk.ExtensionProperties{};
        defer allocator.free(avail_exts);

        var has_memory_budget = false;
        var has_maintenance4 = apiAtLeast13(picked.props.api_version);
        for (avail_exts) |*ext| {
            const name = std.mem.sliceTo(&ext.extension_name, 0);
            if (std.mem.eql(u8, name, vk.extensions.ext_memory_budget.name)) {
                has_memory_budget = true;
            }

            if (std.mem.eql(u8, name, vk.extensions.khr_maintenance_4.name)) {
                has_maintenance4 = true;
            }
        }

        log.debug(
            "ext: {d} available, memory_budget {}, maintenance4 {}",
            .{ avail_exts.len, has_memory_budget, has_maintenance4 },
        );

        // Read the buffer limits, and apply the storage-range override.
        const reported = try readBufferLimits(vki, picked.device, picked.props, has_maintenance4);
        var buffer_limits = reported;
        if (options.storage_range_override) |r| {
            buffer_limits.storage_range = r;
        }

        std.debug.assert(buffer_limits.allocation_size == reported.allocation_size);
        std.debug.assert(buffer_limits.offset_alignment == reported.offset_alignment);
        if (options.storage_range_override != null) {
            log.info(
                "storage-buffer range {d} bytes (reported {d}, set by " ++
                    "" ++ limits.storage_range_env ++ ")",
                .{ buffer_limits.storage_range, reported.storage_range },
            );
        }

        log.debug(
            "buffer limits: storage-buffer range {d} bytes, allocation size {d} bytes, " ++
                "offset alignment {d} bytes",
            .{
                buffer_limits.storage_range,
                buffer_limits.allocation_size,
                buffer_limits.offset_alignment,
            },
        );

        // Create the logical device.
        var ext_names: [2][*:0]const u8 = undefined;
        var ext_count: u32 = 0;
        ext_names[ext_count] = vk.extensions.ext_subgroup_size_control.name;
        ext_count += 1;
        if (has_memory_budget) {
            ext_names[ext_count] = vk.extensions.ext_memory_budget.name;
            ext_count += 1;
        }

        const device_info: vk.DeviceCreateInfo = .{
            .p_next = @ptrCast(&v12_features),
            .queue_create_info_count = 1,
            .p_queue_create_infos = &queue_create_infos,
            .enabled_extension_count = ext_count,
            .pp_enabled_extension_names = &ext_names,
        };
        const device = try vki.createDevice(picked.device, &device_info, null);

        const vkd = if (comptime core_log.hot) blk: {
            dispatch_list.setRealLoader(vki.dispatch.vkGetDeviceProcAddr.?);
            break :blk vk.DeviceWrapper.load(device, dispatch_list.loader);
        } else vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);
        if (vkd.dispatch.vkDestroyDevice == null) return error.DeviceFunctionsMissing;

        errdefer vkd.destroyDevice(device, null);

        // Create the command pool, the one-shot command buffer, and the timestamp pool.
        const queue = vkd.getDeviceQueue(device, queue_family, 0);

        const pool_info: vk.CommandPoolCreateInfo = .{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_family,
        };
        const command_pool = try vkd.createCommandPool(device, &pool_info, null);
        errdefer vkd.destroyCommandPool(device, command_pool, null);

        // Destroying the pool on an error exit frees this buffer too.
        var staging_cmd: vk.CommandBuffer = .null_handle;
        const cmd_info: vk.CommandBufferAllocateInfo = .{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        };
        try vkd.allocateCommandBuffers(device, &cmd_info, @ptrCast(&staging_cmd));
        std.debug.assert(staging_cmd != .null_handle);

        var staging_query_pool: vk.QueryPool = .null_handle;
        if (span.gpuLevel() != .off) {
            const query_info: vk.QueryPoolCreateInfo = .{
                .query_type = .timestamp,
                .query_count = 2,
            };
            staging_query_pool = try vkd.createQueryPool(device, &query_info, null);
        }

        errdefer if (staging_query_pool != .null_handle) vkd.destroyQueryPool(
            device,
            staging_query_pool,
            null,
        );

        // Log the memory layout.
        const mem_props = vki.getPhysicalDeviceMemoryProperties(picked.device);
        for (mem_props.memory_types[0..mem_props.memory_type_count], 0..) |t, type_index| {
            const t_heap = mem_props.memory_heaps[t.heap_index];
            log.debug("memory type {d}: flags 0x{x} heap {d} ({d} MiB, heap flags 0x{x})", .{
                type_index,
                @as(u32, @bitCast(t.property_flags)),
                t.heap_index,
                t_heap.size / (1024 * 1024),
                @as(u32, @bitCast(t_heap.flags)),
            });
        }

        var vram_mib: f64 = 0;
        for (mem_props.memory_heaps[0..mem_props.memory_heap_count]) |heap| {
            if (heap.flags.device_local_bit) {
                vram_mib = @as(f64, @floatFromInt(heap.size)) / (1024.0 * 1024.0);
                break;
            }
        }

        log.debug("init done: queue family {d}, vram {d:.0} MiB", .{ queue_family, vram_mib });

        // Hand the handles to the context.
        return .{
            .instance = instance,
            .physical_device = picked.device,
            .physical_device_props = picked.props,
            .mem_props = mem_props,
            .has_memory_budget = has_memory_budget,
            .software = options.software,
            .buffer_limits = buffer_limits,
            .reported_storage_range = reported.storage_range,
            .device = device,
            .queue = queue,
            .queue_family = queue_family,
            .timestamp_valid_bits = family.timestamp_valid_bits,
            .command_pool = command_pool,
            .staging_cmd = staging_cmd,
            .staging_query_pool = staging_query_pool,
            .vkb = vkb,
            .vki = vki,
            .vkd = vkd,
        };
    }

    pub fn deinit(self: *Context) void {
        std.debug.assert(self.staging_cmd != .null_handle);

        const cmds = [_]vk.CommandBuffer{self.staging_cmd};
        self.vkd.freeCommandBuffers(self.device, self.command_pool, &cmds);
        self.staging_cmd = .null_handle;
        if (self.staging_query_pool != .null_handle) {
            self.vkd.destroyQueryPool(self.device, self.staging_query_pool, null);
        }

        self.vkd.destroyCommandPool(self.device, self.command_pool, null);
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroyInstance(self.instance, null);
    }

    /// Budget and usage of the device-local heap, from the live memory-budget query.
    /// Null when the extension is absent. Read once at runner init, so the KV window stays fixed.
    pub fn deviceLocalBudget(self: *const Context) ?struct { budget: u64, usage: u64 } {
        if (!self.has_memory_budget) return null;

        var budget_props: vk.PhysicalDeviceMemoryBudgetPropertiesEXT = .{
            .heap_budget = undefined,
            .heap_usage = undefined,
        };
        var props2: vk.PhysicalDeviceMemoryProperties2 = .{
            .memory_properties = undefined,
            .p_next = @ptrCast(&budget_props),
        };
        self.vki.getPhysicalDeviceMemoryProperties2(self.physical_device, &props2);
        const mp = &props2.memory_properties;
        var i: u32 = 0;
        while (i < mp.memory_heap_count) : (i += 1) {
            if (mp.memory_heaps[i].flags.device_local_bit) {
                return .{
                    .budget = budget_props.heap_budget[i],
                    .usage = budget_props.heap_usage[i],
                };
            }
        }

        return null;
    }

    /// The live usage and budget of heap `heap`. `null` when the extension is
    /// absent or the device has no such heap.
    pub fn heapBudget(self: *const Context, heap: u32) ?HeapBudget {
        if (!self.has_memory_budget) return null;

        var budget_props: vk.PhysicalDeviceMemoryBudgetPropertiesEXT = .{
            .heap_budget = undefined,
            .heap_usage = undefined,
        };
        var props2: vk.PhysicalDeviceMemoryProperties2 = .{
            .memory_properties = undefined,
            .p_next = @ptrCast(&budget_props),
        };
        self.vki.getPhysicalDeviceMemoryProperties2(self.physical_device, &props2);
        const count = props2.memory_properties.memory_heap_count;
        std.debug.assert(count <= vk.MAX_MEMORY_HEAPS);
        if (heap >= count) return null;

        return .{
            .usage = budget_props.heap_usage[heap],
            .budget = budget_props.heap_budget[heap],
        };
    }

    /// Device-local bytes free now: budget - usage when live, else the static heap size.
    /// The heap size overstates the free bytes on a GPU that a desktop shares.
    pub fn deviceLocalAvailableBytes(self: *const Context) u64 {
        if (self.deviceLocalBudget()) |b| return b.budget -| b.usage;
        for (self.mem_props.memory_heaps[0..self.mem_props.memory_heap_count]) |heap| {
            if (heap.flags.device_local_bit) return heap.size;
        }

        return 0;
    }

    /// Refuse a step buffer of `n` blocks of `block_elems` f32 elements that breaks a device buffer
    /// limit, before any allocation. `what` names the buffer in the log line.
    pub fn guardStepBuffer(
        self: *const Context,
        what: []const u8,
        n: u32,
        block_elems: u32,
    ) error{StepBufferPastDeviceLimit}!void {
        const lim = self.buffer_limits;
        const verdict = limits.blocksFit(n, block_elems, lim);
        if (verdict == .fits) return;

        const limit = kv_index.limitOf(verdict, lim);
        log.err(
            "The {s} holds {d} blocks of {d} f32 elements, past the {s} of {d} {t}. " ++
                "Run this model on a CPU build",
            .{ what, n, block_elems, limit.name, limit.value, limit.unit },
        );
        return error.StepBufferPastDeviceLimit;
    }

    /// Refuse a KV layout whose region breaks a device buffer limit, before any allocation.
    /// The log line names the limit, the region size, and the widest window that fits.
    pub fn guardKvLayout(
        self: *const Context,
        layout: kv_index.Layout,
    ) error{KvRegionPastDeviceLimit}!void {
        const lim = self.buffer_limits;
        const verdict = kv_index.fits(layout, lim);
        if (verdict == .fits) return;

        // Name the failed limit.
        const bytes = layout.regionBytes();
        const n_slots = layout.n_slots;
        const window = layout.window;
        const limit = kv_index.limitOf(verdict, lim);
        if (verdict == .misaligned) {
            // A window that is a multiple of `unit` keeps the region whole in alignment units.
            const one_position = kv_index.Layout.init(
                n_slots,
                1,
                layout.per_step,
                layout.n_layers,
                layout.kv_dtype,
            );
            const unit = lim.offset_alignment / std.math.gcd(
                lim.offset_alignment,
                one_position.regionBytes(),
            );
            std.debug.assert(unit >= 2);
            log.err(
                "KV region of {d} bytes for {d} slots x {d} positions is not a multiple of " ++
                    "the {s} of {d} {t}. Use a --max-seq value that is a multiple of {d}",
                .{ bytes, n_slots, window, limit.name, limit.value, limit.unit, unit },
            );
            return error.KvRegionPastDeviceLimit;
        }

        // Pick the advice, from the widest window that fits.
        const fit = kv_index.maxGranuleWindow(n_slots, layout.per_step, layout.kv_dtype, lim);
        std.debug.assert(fit < window);
        var advice_buf: [kv_advice_max]u8 = undefined;
        const range = verdict == .past_storage_range;
        const advice: []const u8 = if (fit == 0)
            (if (range) "No window fits this many slots. Use fewer sessions, or set " ++
                "" ++ limits.storage_range_env else "No window fits this many slots. " ++
                "Use fewer sessions")
        else if (range)
            std.fmt.bufPrint(&advice_buf, kv_range_advice_fmt, .{fit}) catch unreachable
        else
            std.fmt.bufPrint(&advice_buf, kv_advice_fmt, .{fit}) catch unreachable;

        // Log the refusal in the unit of the failed limit.
        switch (limit.unit) {
            .elements => log.err(
                "KV region of {d} bytes for {d} slots x {d} positions holds {d} elements, " ++
                    "but the {s} is {d} {t}. {s}",
                .{
                    bytes,
                    n_slots,
                    window,
                    layout.regionElems(),
                    limit.name,
                    limit.value,
                    limit.unit,
                    advice,
                },
            ),
            .bytes => log.err(
                "KV region of {d} bytes for {d} slots x {d} positions is larger than the " ++
                    "{s} of {d} {t}. {s}",
                .{ bytes, n_slots, window, limit.name, limit.value, limit.unit, advice },
            ),
        }

        return error.KvRegionPastDeviceLimit;
    }

    /// Refuse a split attention scratch that breaks a device buffer limit, before any allocation.
    pub fn guardAttnSplitScratch(
        self: *const Context,
        scratch: attn_split.Scratch,
    ) error{AttentionScratchPastDeviceLimit}!void {
        const lim = self.buffer_limits;
        const verdict = attn_split.fits(scratch, lim);
        if (verdict == .fits) return;

        const bytes = scratch.totalBytes();
        const limit = kv_index.limitOf(verdict, lim);
        switch (verdict) {
            // The scratch binds whole at offset 0, so the alignment never binds it.
            .fits, .misaligned => unreachable,
            .index_past_31_bits => log.err(
                "Split attention scratch of {d} bytes holds {d} elements, but the {s} is {d} {t}",
                .{ bytes, scratch.totalElems(), limit.name, limit.value, limit.unit },
            ),
            .past_allocation_size => log.err(
                "Split attention scratch of {d} bytes is larger than the {s} of {d} {t}",
                .{ bytes, limit.name, limit.value, limit.unit },
            ),
            .past_storage_range => log.err(
                "Split attention scratch of {d} bytes is larger than the {s} of {d} {t}. " ++
                    "Set " ++ limits.storage_range_env ++
                    " to a range that this device reads correctly",
                .{ bytes, limit.name, limit.value, limit.unit },
            ),
        }

        return error.AttentionScratchPastDeviceLimit;
    }
};

const QueueFamily = struct { index: u32, timestamp_valid_bits: u32 };

// -- api --------------------------------------------------------------------

pub fn pickPhysicalDevice(
    vki: vk.InstanceWrapper,
    instance: vk.Instance,
    allocator: std.mem.Allocator,
    software: bool,
) InitError!PickedDevice {
    const devices = try vki.enumeratePhysicalDevicesAlloc(instance, allocator);
    defer allocator.free(devices);

    if (devices.len == 0) return error.NoVulkanDevice;

    // Keep the best score. The first device wins a tie.
    var best: ?PickedDevice = null;
    for (devices) |dev| {
        const props = vki.getPhysicalDeviceProperties(dev);
        const score = scoreDeviceType(props.device_type, software);
        if (score < 0) continue;

        if (best == null or score > best.?.score) {
            best = .{ .device = dev, .props = props, .score = score };
        }
    }

    const picked = best orelse {
        if (software) {
            log.err(
                limits.software_device_env ++ " takes a CPU Vulkan device, and none exists",
                .{},
            );
        }

        return error.NoVulkanDevice;
    };

    const name = std.mem.sliceTo(&picked.props.device_name, 0);
    log.debug(
        "select: {s} type {s}, score {d}, from {d} adapters",
        .{ name, @tagName(picked.props.device_type), picked.score, devices.len },
    );
    return picked;
}

/// Find a memory type that the `type_bits` mask allows and that has every flag in `required_flags`.
/// Returns null when none qualifies. The caller turns that into `error.NoSuitableMemoryType`.
pub fn findMemoryType(
    mem_props: vk.PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk.MemoryPropertyFlags,
) ?u32 {
    var i: u32 = 0;
    while (i < mem_props.memory_type_count) : (i += 1) {
        const supported = (type_bits & (@as(u32, 1) << @intCast(i))) != 0;
        if (!supported) continue;

        const flags = mem_props.memory_types[i].property_flags;
        if (flags.contains(required_flags)) return i;
    }

    return null;
}

/// Pick the memory type for a buffer that the CPU writes and reads. A cached type wins, because
/// an uncached mapping runs at a small fraction of memory speed. The coherent fallback is slow.
/// Returns null when no allowed type is host-visible and coherent.
pub fn pickHostType(mem_props: vk.PhysicalDeviceMemoryProperties, type_bits: u32) ?u32 {
    return findMemoryType(mem_props, type_bits, host_cached_flags) orelse
        findMemoryType(mem_props, type_bits, host_coherent_flags);
}

// -- helpers ----------------------------------------------------------------

/// True when a device reports Vulkan 1.3 or later, where `maxBufferSize` is core.
fn apiAtLeast13(api_version: u32) bool {
    const v: vk.Version = @bitCast(api_version);
    return v.major > 1 or (v.major == 1 and v.minor >= 3);
}

/// Read the storage-buffer limits of one device. The allocation size takes the lower
/// `maxBufferSize` where the device reports one. A zero limit or an odd alignment fails the open.
fn readBufferLimits(
    vki: vk.InstanceWrapper,
    dev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    has_maintenance4: bool,
) error{DeviceBufferLimitsInvalid}!limits.BufferLimits {
    var maintenance4: vk.PhysicalDeviceMaintenance4Properties = .{ .max_buffer_size = 0 };
    var maintenance3: vk.PhysicalDeviceMaintenance3Properties = .{
        .max_per_set_descriptors = 0,
        .max_memory_allocation_size = 0,
        .p_next = if (has_maintenance4) @ptrCast(&maintenance4) else null,
    };
    var props2: vk.PhysicalDeviceProperties2 = .{
        .properties = undefined,
        .p_next = @ptrCast(&maintenance3),
    };
    vki.getPhysicalDeviceProperties2(dev, &props2);
    var allocation_size: u64 = maintenance3.max_memory_allocation_size;
    if (has_maintenance4) {
        allocation_size = @min(allocation_size, maintenance4.max_buffer_size);
    }

    const lim: limits.BufferLimits = .{
        .storage_range = props.limits.max_storage_buffer_range,
        .allocation_size = allocation_size,
        .offset_alignment = props.limits.min_storage_buffer_offset_alignment,
    };
    if (lim.storage_range == 0 or lim.allocation_size == 0) return error.DeviceBufferLimitsInvalid;
    if (lim.offset_alignment == 0 or !std.math.isPowerOfTwo(lim.offset_alignment)) {
        return error.DeviceBufferLimitsInvalid;
    }

    std.debug.assert(lim.storage_range <= limits.storage_range_max);
    return lim;
}

/// The rank of a device type: 3 = discrete, 2 = integrated, 1 = virtual, 0 = other, -1 = rejected.
/// A software run scores a CPU device 0 and rejects every other type.
fn scoreDeviceType(t: vk.PhysicalDeviceType, software: bool) i32 {
    if (software) return if (t == .cpu) 0 else -1;

    return switch (t) {
        .discrete_gpu => 3,
        .integrated_gpu => 2,
        .virtual_gpu => 1,
        .other => 0,
        .cpu => -1,
        else => -1,
    };
}

fn findComputeQueueFamily(
    vki: vk.InstanceWrapper,
    dev: vk.PhysicalDevice,
    allocator: std.mem.Allocator,
) InitError!QueueFamily {
    const families = try vki.getPhysicalDeviceQueueFamilyPropertiesAlloc(dev, allocator);
    defer allocator.free(families);

    for (families, 0..) |fam, i| {
        if (fam.queue_flags.compute_bit) {
            log.debug(
                "queue: compute family {d} of {d}, {d} timestamp bits",
                .{ i, families.len, fam.timestamp_valid_bits },
            );
            std.debug.assert(fam.timestamp_valid_bits <= 64);
            return .{ .index = @intCast(i), .timestamp_valid_bits = fam.timestamp_valid_bits };
        }
    }

    return error.NoComputeQueue;
}

// -- tests ------------------------------------------------------------------

/// Build a fabricated memory table from per-type flag words. Every type sits
/// on heap 0, which the picker never reads.
fn fakeMemProps(flag_words: []const u32) vk.PhysicalDeviceMemoryProperties {
    var props = std.mem.zeroes(vk.PhysicalDeviceMemoryProperties);
    props.memory_type_count = @intCast(flag_words.len);
    for (flag_words, 0..) |w, i| {
        props.memory_types[i] = .{ .property_flags = @bitCast(w), .heap_index = 0 };
    }

    return props;
}

const all_types: u32 = 0xffff_ffff;

test "scoreDeviceType: a CPU device needs the software option, which rejects a GPU" {
    try std.testing.expectEqual(@as(i32, 3), scoreDeviceType(.discrete_gpu, false));
    try std.testing.expectEqual(@as(i32, -1), scoreDeviceType(.cpu, false));
    try std.testing.expectEqual(@as(i32, 0), scoreDeviceType(.cpu, true));
    try std.testing.expectEqual(@as(i32, -1), scoreDeviceType(.discrete_gpu, true));
    try std.testing.expectEqual(@as(i32, -1), scoreDeviceType(.integrated_gpu, true));
}

test "pickHostType: a cached type wins when the buffer allows it" {
    const props = fakeMemProps(&.{ 0x0, 0x6, 0xe, 0x1 });
    try std.testing.expectEqual(@as(?u32, 2), pickHostType(props, all_types));
}

test "pickHostType: masking out the cached type picks the coherent type" {
    const props = fakeMemProps(&.{ 0x0, 0x6, 0xe, 0x1 });
    try std.testing.expectEqual(@as(?u32, 1), pickHostType(props, all_types & ~@as(u32, 1 << 2)));
}

test "pickHostType: without a cached type it picks the coherent type" {
    const props = fakeMemProps(&.{ 0x0, 0x1, 0x6 });
    try std.testing.expectEqual(@as(?u32, 2), pickHostType(props, all_types));
}

test "pickHostType: without a host-visible type it returns null" {
    const props = fakeMemProps(&.{ 0x0, 0x1, 0x1 });
    try std.testing.expectEqual(@as(?u32, null), pickHostType(props, all_types));
    // A host-visible type that the buffer masks out does not count.
    const masked = fakeMemProps(&.{ 0x0, 0x6 });
    try std.testing.expectEqual(@as(?u32, null), pickHostType(masked, 1 << 0));
}

test "pickHostType: the cached type wins before and after a coherent-only type" {
    const cached_first = fakeMemProps(&.{ 0xe, 0x6 });
    try std.testing.expectEqual(@as(?u32, 0), pickHostType(cached_first, all_types));
    const cached_last = fakeMemProps(&.{ 0x6, 0xe });
    try std.testing.expectEqual(@as(?u32, 1), pickHostType(cached_last, all_types));
}
