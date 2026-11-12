//! The buffers of the Vulkan backend: the host-visible buffer, the device-local buffer, the device
//! KV store, and the VRAM arena.
//!
//! Comptime. `HostBuffer.sliceTyped` takes the element type of its mapped view.
//!
//! VRAM arenas. The driver sets the constraint. A Llama-8B Q8_0 model holds 225 Q8_0 weights, so
//! one buffer and one memory object each makes 450 device allocations. That count plus a 512 MB KV
//! cache exhausts the device-local heap of the target driver long before the 16 GB fill up. Ten
//! segments of about 1 GB each keep the count low. Buffer usage is a parameter, so one arena type
//! backs the scales, the quants, and the KV shards. Offsets align to
//! `minStorageBufferOffsetAlignment`. Allocation is append-only: weights live as long as the
//! runner, and a per-session KV pool resets whole between sessions.

const std = @import("std");
const vk = @import("vulkan");
const kv_index = @import("../../core/kv_index.zig");
const limits = @import("../../core/limits.zig");
const vulkan_ctx = @import("../../runtime/vulkan_ctx.zig");
const log = std.log.scoped(.gpu);

const Copy = @import("submit.zig").Copy;

// -- types ------------------------------------------------------------------

/// Grow-on-demand host-visible and coherent storage buffer, and the staging buffer of a
/// `DeviceBuffer`. The memory type is host-cached when the driver offers one, because an uncached
/// mapping slows every CPU write and read.
pub const HostBuffer = struct {
    ctx: *const vulkan_ctx.Context,
    buffer: vk.Buffer = .null_handle,
    memory: vk.DeviceMemory = .null_handle,
    capacity: vk.DeviceSize = 0,
    mapped: [*]u8 = undefined,

    pub fn deinit(self: *HostBuffer) void {
        if (self.capacity == 0) return;

        self.ctx.vkd.unmapMemory(self.ctx.device, self.memory);
        self.ctx.vkd.destroyBuffer(self.ctx.device, self.buffer, null);
        self.ctx.vkd.freeMemory(self.ctx.device, self.memory, null);
        self.capacity = 0;
    }

    pub fn ensure(self: *HostBuffer, size: vk.DeviceSize) !void {
        if (self.capacity >= size) return;

        self.deinit();

        // Storage for shader reads, plus staged uploads and staged readbacks such as logits or a KV
        // swap-out. All three flags are free, because the driver differs only at use time.
        const buf_info: vk.BufferCreateInfo = .{
            .size = size,
            .usage = .{
                .storage_buffer_bit = true,
                .transfer_src_bit = true,
                .transfer_dst_bit = true,
            },
            .sharing_mode = .exclusive,
        };
        const buffer = try self.ctx.vkd.createBuffer(self.ctx.device, &buf_info, null);
        errdefer self.ctx.vkd.destroyBuffer(self.ctx.device, buffer, null);

        // Pick the host memory type.
        const reqs = self.ctx.vkd.getBufferMemoryRequirements(self.ctx.device, buffer);
        const mem_type = vulkan_ctx.pickHostType(self.ctx.mem_props, reqs.memory_type_bits) orelse
            return error.NoSuitableMemoryType;
        const chosen = self.ctx.mem_props.memory_types[mem_type];
        const chosen_heap = self.ctx.mem_props.memory_heaps[chosen.heap_index];
        log.debug(
            "host buffer: {d} bytes, memory type {d} flags 0x{x} heap {d} heap flags 0x{x}",
            .{
                size,
                mem_type,
                @as(u32, @bitCast(chosen.property_flags)),
                chosen.heap_index,
                @as(u32, @bitCast(chosen_heap.flags)),
            },
        );
        // A silent fallback hides the reason CPU access to this buffer is slow.
        if (!chosen.property_flags.host_cached_bit) {
            log.debug(
                "host buffer: no host-cached memory type, " ++
                    "so CPU access to this host buffer is uncached and slow",
                .{},
            );
        }

        // Allocate, bind, and map the memory.
        const alloc_info: vk.MemoryAllocateInfo = .{
            .allocation_size = reqs.size,
            .memory_type_index = mem_type,
        };
        const memory = try self.ctx.vkd.allocateMemory(self.ctx.device, &alloc_info, null);
        errdefer self.ctx.vkd.freeMemory(self.ctx.device, memory, null);

        try self.ctx.vkd.bindBufferMemory(self.ctx.device, buffer, memory, 0);
        const mapped_opt = try self.ctx.vkd.mapMemory(self.ctx.device, memory, 0, size, .{});

        // Commit the new buffer.
        self.buffer = buffer;
        self.memory = memory;
        self.capacity = size;
        self.mapped = @ptrCast(mapped_opt.?);
        _ = host_buffers_created.fetchAdd(1, .monotonic);
    }

    pub fn sliceTyped(self: *HostBuffer, comptime T: type, count: usize) []T {
        std.debug.assert(count * @sizeOf(T) <= self.capacity);

        const raw: [*]T = @ptrCast(@alignCast(self.mapped));
        return raw[0..count];
    }
};

/// Device-local storage buffer and transfer destination, written by a copy from a `HostBuffer` and
/// never mapped. A shader reads this memory at about 500 GB/s on an RDNA2 card, against about 30
/// GB/s for host-visible memory over the virtualized host bus.
pub const DeviceBuffer = struct {
    ctx: *const vulkan_ctx.Context,
    buffer: vk.Buffer = .null_handle,
    memory: vk.DeviceMemory = .null_handle,
    capacity: vk.DeviceSize = 0,

    pub fn deinit(self: *DeviceBuffer) void {
        if (self.capacity == 0) return;

        self.ctx.vkd.destroyBuffer(self.ctx.device, self.buffer, null);
        self.ctx.vkd.freeMemory(self.ctx.device, self.memory, null);
        self.capacity = 0;
    }

    pub fn ensure(self: *DeviceBuffer, size: vk.DeviceSize) !void {
        if (self.capacity >= size) return;

        self.deinit();

        const buf_info: vk.BufferCreateInfo = .{
            .size = size,
            .usage = .{
                .storage_buffer_bit = true,
                .transfer_dst_bit = true,
                .transfer_src_bit = true,
            },
            .sharing_mode = .exclusive,
        };
        const buffer = try self.ctx.vkd.createBuffer(self.ctx.device, &buf_info, null);
        errdefer self.ctx.vkd.destroyBuffer(self.ctx.device, buffer, null);

        const reqs = self.ctx.vkd.getBufferMemoryRequirements(self.ctx.device, buffer);
        const required: vk.MemoryPropertyFlags = .{ .device_local_bit = true };
        const mem_type = vulkan_ctx.findMemoryType(
            self.ctx.mem_props,
            reqs.memory_type_bits,
            required,
        ) orelse
            return error.NoSuitableMemoryType;

        const alloc_info: vk.MemoryAllocateInfo = .{
            .allocation_size = reqs.size,
            .memory_type_index = mem_type,
        };
        const memory = try self.ctx.vkd.allocateMemory(self.ctx.device, &alloc_info, null);
        errdefer self.ctx.vkd.freeMemory(self.ctx.device, memory, null);

        try self.ctx.vkd.bindBufferMemory(self.ctx.device, buffer, memory, 0);

        self.buffer = buffer;
        self.memory = memory;
        self.capacity = size;
    }
};

/// The device KV cache: one region per layer and K or V, in the order and the packing of
/// `kv_index`. Each segment holds whole regions, so the device limits bind one region and not the
/// whole cache. The runner maps each `kv_index.fits` verdict to an error before `init`.
pub const KvStore = struct {
    layout: kv_index.Layout,
    packing: kv_index.Packing,
    /// The limits that the packing honors: each descriptor starts at a multiple of
    /// `offset_alignment` and binds at most `storage_range` bytes.
    lim: limits.BufferLimits,
    /// The first `packing.n_segments` entries hold the allocations in order. The others stay empty.
    segments: [max_segments]DeviceBuffer,

    /// One segment per region when no two regions fit one allocation.
    pub const max_segments: u32 = 2 * limits.max_layers;

    /// The address of one row for a copy: a buffer and a byte offset.
    pub const Row = struct { buffer: vk.Buffer, offset: u64 };

    pub const InitError = error{NoSuitableMemoryType} ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError;

    /// Allocate every segment of `layout` packed under the buffer limits of
    /// `ctx`.
    pub fn init(ctx: *const vulkan_ctx.Context, layout: kv_index.Layout) InitError!KvStore {
        const lim = ctx.buffer_limits;
        std.debug.assert(kv_index.fits(layout, lim) == .fits);
        const packing = kv_index.pack(layout, lim);
        std.debug.assert(packing.n_segments >= 1);
        std.debug.assert(packing.n_segments <= max_segments);

        var store: KvStore = .{
            .layout = layout,
            .packing = packing,
            .lim = lim,
            .segments = @splat(.{ .ctx = ctx }),
        };
        var made: u32 = 0;
        errdefer for (store.segments[0..made]) |*segment| segment.deinit();

        while (made < packing.n_segments) : (made += 1) {
            try store.segments[made].ensure(packing.segmentBytes(made));
        }

        std.debug.assert(store.deviceBytes() == layout.totalBytes());
        log.debug("kv store: {d} regions of {d} KiB in {d} allocations of up to {d} regions", .{
            packing.n_regions, packing.region_bytes >> 10, packing.n_segments, packing.per_segment,
        });
        return store;
    }

    /// Free every segment. A second call frees nothing.
    pub fn deinit(self: *KvStore) void {
        std.debug.assert(self.packing.n_segments <= max_segments);

        for (self.segments[0..self.packing.n_segments]) |*segment| {
            segment.deinit();
        }

        for (self.segments) |segment| {
            std.debug.assert(segment.capacity == 0);
        }
    }

    /// Bytes the segments hold together: the total of the layout.
    pub fn deviceBytes(self: *const KvStore) u64 {
        var sum: u64 = 0;
        for (self.segments[0..self.packing.n_segments], 0..) |segment, i| {
            std.debug.assert(segment.capacity == self.packing.segmentBytes(@intCast(i)));

            sum += segment.capacity;
        }

        std.debug.assert(sum == self.layout.totalBytes());
        return sum;
    }

    /// The binding of region `(layer, t)`: every slot of that kind.
    pub fn region(self: *const KvStore, layer: u32, t: kv_index.KvKind) vk.DescriptorBufferInfo {
        const base = self.regionAt(layer, t);
        const range = self.packing.region_bytes;
        std.debug.assert(range == self.layout.regionBytes());
        std.debug.assert(range <= self.lim.storage_range);
        return .{ .buffer = base.buffer, .offset = base.offset, .range = range };
    }

    /// The binding of the `window` rows of `slot` in region `(layer, t)`.
    pub fn slotView(
        self: *const KvStore,
        layer: u32,
        t: kv_index.KvKind,
        slot: u32,
    ) vk.DescriptorBufferInfo {
        const base = self.regionAt(layer, t);
        const offset = base.offset + self.layout.slotByteOffset(slot);
        const range = self.layout.slotRegionBytes();
        std.debug.assert(offset % self.lim.offset_alignment == 0);
        std.debug.assert(offset + range <= base.offset + self.packing.region_bytes);
        std.debug.assert(range <= self.lim.storage_range);
        return .{ .buffer = base.buffer, .offset = offset, .range = range };
    }

    /// The address of the row at `(slot, pos)` in region `(layer, t)`.
    pub fn rowAt(self: *const KvStore, layer: u32, t: kv_index.KvKind, slot: u32, pos: u32) Row {
        const base = self.regionAt(layer, t);
        const offset = base.offset + self.layout.rowByteOffset(slot, pos);
        std.debug.assert(offset >= base.offset + self.layout.slotByteOffset(slot));
        std.debug.assert(
            offset + self.layout.rowBytes() <= base.offset + self.packing.region_bytes,
        );
        return .{ .buffer = base.buffer, .offset = offset };
    }

    /// The copy list that forks slot `source` into slot `target`, one copy per region. The slot
    /// rows of one region never overlap, so each copy stays inside one buffer. A copy needs no
    /// offset alignment, so the list addresses rows and not descriptor views.
    pub fn cloneCopies(self: *const KvStore, source: u32, target: u32, out: []Copy) []Copy {
        const layout = self.layout;
        std.debug.assert(source != target);
        std.debug.assert(out.len >= layout.regionCount());
        const slot_rows_bytes = layout.slotRegionBytes();
        var c: usize = 0;
        var total: u64 = 0;
        var layer: u32 = 0;
        while (layer < layout.n_layers) : (layer += 1) {
            for ([_]kv_index.KvKind{ .k, .v }) |t| {
                const from = self.rowAt(layer, t, source, 0);
                const to = self.rowAt(layer, t, target, 0);
                std.debug.assert(from.buffer == to.buffer);
                out[c] = .{
                    .src = from.buffer,
                    .dst = to.buffer,
                    .src_offset = from.offset,
                    .dst_offset = to.offset,
                    .size = slot_rows_bytes,
                };
                c += 1;
                total += slot_rows_bytes;
            }
        }

        std.debug.assert(c == layout.regionCount());
        std.debug.assert(total == layout.slotBytes());
        return out[0..c];
    }

    /// The copy list between the first `n_rows` rows of `slot` in every region and a packed host
    /// stage, in blob order: layers ascending, K rows then V rows. `to_host` true reads the device,
    /// and false writes it. A save and its restore walk one order, so their offsets agree.
    pub fn rowCopies(
        self: *const KvStore,
        slot: u32,
        n_rows: u32,
        stage: vk.Buffer,
        to_host: bool,
        out: []Copy,
    ) []Copy {
        const layout = self.layout;
        std.debug.assert(n_rows >= 1);
        std.debug.assert(n_rows <= layout.window);
        std.debug.assert(out.len >= layout.regionCount());

        // Walk the regions in blob order, one copy per region.
        const rows_bytes: u64 = @as(u64, n_rows) * layout.rowBytes();
        var off: u64 = 0;
        var c: usize = 0;
        var layer: u32 = 0;
        while (layer < layout.n_layers) : (layer += 1) {
            for ([_]kv_index.KvKind{ .k, .v }) |t| {
                const dev = self.rowAt(layer, t, slot, 0);
                out[c] = if (to_host)
                    .{
                        .src = dev.buffer,
                        .src_offset = dev.offset,
                        .dst = stage,
                        .dst_offset = off,
                        .size = rows_bytes,
                    }
                else
                    .{
                        .src = stage,
                        .src_offset = off,
                        .dst = dev.buffer,
                        .dst_offset = dev.offset,
                        .size = rows_bytes,
                    };
                c += 1;
                off += rows_bytes;
            }
        }

        // Check that the walk covered the blob.
        std.debug.assert(c == layout.regionCount());
        std.debug.assert(off == layout.blobBytes(n_rows));
        return out[0..c];
    }

    /// The segment buffer of region `(layer, t)` and the byte where it starts.
    fn regionAt(self: *const KvStore, layer: u32, t: kv_index.KvKind) Row {
        const index = self.layout.regionIndex(layer, t);
        const segment = self.segments[self.packing.segmentOf(index)];
        const offset = self.packing.offsetOf(index);
        std.debug.assert(segment.capacity >= 1);
        std.debug.assert(offset + self.packing.region_bytes <= segment.capacity);
        std.debug.assert(offset % self.lim.offset_alignment == 0);
        return .{ .buffer = segment.buffer, .offset = offset };
    }
};

/// Device-local VRAM arena. It sub-allocates from a small number of large buffer and memory
/// segments, and it never frees one allocation. A plan from a dry run of the same requests sizes
/// each segment to the requests it holds, so no segment keeps an unused tail.
pub const VramArena = struct {
    ctx: *const vulkan_ctx.Context,
    gpa: std.mem.Allocator,
    usage: vk.BufferUsageFlags,
    /// The segment and the offset of each request, and the capacity and the fill of each segment.
    packing: Packing,
    /// The buffer and the memory of each segment of `packing`, in the same order.
    segments: std.ArrayList(Segment),
    /// The device memory of every segment, which the driver can round up past its capacity.
    memory_bytes: vk.DeviceSize,

    pub const Segment = struct {
        buffer: vk.Buffer,
        memory: vk.DeviceMemory,
    };

    /// A view of the arena, the shape that a descriptor update takes.
    pub const View = vk.DescriptorBufferInfo;

    /// The segment size of the f16 scale and f32 norm arena.
    pub const seg_bytes_small: vk.DeviceSize = 256 * 1024 * 1024;

    /// The segment size of the i8 quants arena. The largest Llama-8B Q8_0 quants weight is the
    /// lm_head at about 525 MB, so 1 GB is a safe default.
    pub const seg_bytes_quants: vk.DeviceSize = 1024 * 1024 * 1024;

    pub const AllocError = error{
        NoSuitableMemoryType,
        OutOfMemory,
    } ||
        vk.DeviceWrapper.CreateBufferError ||
        vk.DeviceWrapper.AllocateMemoryError ||
        vk.DeviceWrapper.BindBufferMemoryError;

    /// The placement rule of an arena without its device memory, so a dry run over the requests
    /// of an upload finds the fill of each segment before the arena allocates one. A request lands
    /// in the most recent segment, then in an older one, then in a new segment.
    pub const Packing = struct {
        /// The capacity of a new segment without a plan. A larger request takes its own size.
        seg_capacity: vk.DeviceSize,
        align_bytes: vk.DeviceSize,
        /// The capacity and the fill of each segment, in the order they open.
        fills: std.ArrayList(Fill),
        /// The capacity of each segment in the order they open. Null: no plan.
        plan: ?[]const vk.DeviceSize,
        requests: u64,
        request_bytes: u64,

        pub const Fill = struct {
            capacity: vk.DeviceSize,
            used: vk.DeviceSize,
        };

        /// The segment and the offset of one request. `opens`: the request opens the segment.
        pub const Place = struct {
            segment: u32,
            offset: vk.DeviceSize,
            opens: bool,
        };

        pub fn init(seg_capacity: vk.DeviceSize, align_bytes: vk.DeviceSize) Packing {
            std.debug.assert(seg_capacity >= 1);
            std.debug.assert(std.math.isPowerOfTwo(align_bytes));

            return .{
                .seg_capacity = seg_capacity,
                .align_bytes = align_bytes,
                .fills = .empty,
                .plan = null,
                .requests = 0,
                .request_bytes = 0,
            };
        }

        /// Free the fills and the plan, which `gpa` holds too.
        pub fn deinit(self: *Packing, gpa: std.mem.Allocator) void {
            if (self.plan) |plan| {
                gpa.free(plan);
            }

            self.fills.deinit(gpa);
        }

        /// Take the fill of each segment of `dry`, a dry run of the next requests in their order,
        /// as its capacity. A planned segment is no larger than in the dry run and holds what the
        /// dry run put in it, so each request lands where the dry run put it.
        pub fn setPlan(
            self: *Packing,
            gpa: std.mem.Allocator,
            dry: *const Packing,
        ) std.mem.Allocator.Error!void {
            std.debug.assert(self.fills.items.len == 0);
            std.debug.assert(self.plan == null);
            std.debug.assert(dry.plan == null);
            std.debug.assert(dry.seg_capacity == self.seg_capacity);
            std.debug.assert(dry.align_bytes == self.align_bytes);

            const plan = try gpa.alloc(vk.DeviceSize, dry.fills.items.len);
            for (plan, dry.fills.items) |*capacity, fill| {
                capacity.* = fill.used;
            }

            self.plan = plan;
        }

        /// Place a request of `size` bytes in the first segment with room, or in a new segment.
        pub fn place(
            self: *Packing,
            gpa: std.mem.Allocator,
            size: vk.DeviceSize,
        ) std.mem.Allocator.Error!Place {
            std.debug.assert(size >= 1);

            const at = self.fit(size) orelse try self.open(gpa, size);
            self.requests += 1;
            self.request_bytes += size;
            std.debug.assert(at.offset % self.align_bytes == 0);
            std.debug.assert(at.offset + size <= self.fills.items[at.segment].capacity);
            return at;
        }

        /// The bytes of every segment.
        pub fn capacityBytes(self: *const Packing) u64 {
            var sum: u64 = 0;
            for (self.fills.items) |fill| {
                sum += fill.capacity;
            }

            return sum;
        }

        /// The bytes of every segment up to the end of its last request.
        pub fn usedBytes(self: *const Packing) u64 {
            var sum: u64 = 0;
            for (self.fills.items) |fill| {
                sum += fill.used;
            }

            std.debug.assert(sum >= self.request_bytes);
            return sum;
        }

        /// The most padding that the requests so far can leave: one alignment for each, because
        /// a request starts less than one alignment past the fill before it.
        pub fn paddingBound(self: *const Packing) u64 {
            return self.requests * self.align_bytes;
        }

        /// Every segment of the plan is open and full, so the segments hold the requests plus
        /// their padding and nothing more.
        pub fn assertFull(self: *const Packing) void {
            std.debug.assert(self.plan != null);
            const plan = self.plan.?;
            std.debug.assert(self.fills.items.len == plan.len);
            for (self.fills.items, plan) |fill, capacity| {
                std.debug.assert(fill.capacity == capacity);
                std.debug.assert(fill.used == fill.capacity);
            }

            std.debug.assert(self.usedBytes() - self.request_bytes <= self.paddingBound());
        }

        /// Fit `size` in the most recent segment, which is likeliest to have room, then in an
        /// older one.
        fn fit(self: *Packing, size: vk.DeviceSize) ?Place {
            const n = self.fills.items.len;
            if (n == 0) return null;
            if (self.fitIn(n - 1, size)) |at| return at;

            for (0..n - 1) |i| {
                if (self.fitIn(i, size)) |at| return at;
            }

            return null;
        }

        fn fitIn(self: *Packing, index: usize, size: vk.DeviceSize) ?Place {
            const fill = &self.fills.items[index];
            const offset = alignUp(fill.used, self.align_bytes);
            if (offset + size > fill.capacity) return null;

            fill.used = offset + size;
            return .{ .segment = @intCast(index), .offset = offset, .opens = false };
        }

        /// Open a segment for `size`. A plan holds one capacity for each segment of its dry run.
        fn open(
            self: *Packing,
            gpa: std.mem.Allocator,
            size: vk.DeviceSize,
        ) std.mem.Allocator.Error!Place {
            const n = self.fills.items.len;
            const capacity = if (self.plan) |plan| blk: {
                std.debug.assert(n < plan.len);
                break :blk plan[n];
            } else @max(self.seg_capacity, size);
            std.debug.assert(size <= capacity);

            try self.fills.append(gpa, .{ .capacity = capacity, .used = size });
            return .{ .segment = @intCast(n), .offset = 0, .opens = true };
        }
    };

    /// `seg_capacity` is the default size of a new segment. Pick at least the largest single
    /// allocation, or `alloc` grows a segment to fit, which fragments the arena.
    pub fn init(
        ctx: *const vulkan_ctx.Context,
        gpa: std.mem.Allocator,
        usage: vk.BufferUsageFlags,
        seg_capacity: vk.DeviceSize,
    ) VramArena {
        return .{
            .ctx = ctx,
            .gpa = gpa,
            .usage = usage,
            .packing = .init(seg_capacity, ctx.buffer_limits.offset_alignment),
            .segments = .empty,
            .memory_bytes = 0,
        };
    }

    pub fn deinit(self: *VramArena) void {
        for (self.segments.items) |s| {
            self.ctx.vkd.destroyBuffer(self.ctx.device, s.buffer, null);
            self.ctx.vkd.freeMemory(self.ctx.device, s.memory, null);
        }

        self.segments.deinit(self.gpa);
        self.packing.deinit(self.gpa);
    }

    /// Size each segment from `dry`, a dry run of the requests that `alloc` takes next, in the
    /// same order. Call it before the first `alloc`.
    pub fn setPlan(self: *VramArena, dry: *const Packing) std.mem.Allocator.Error!void {
        std.debug.assert(self.segments.items.len == 0);

        try self.packing.setPlan(self.gpa, dry);
    }

    /// Allocate `size` bytes where `Packing.place` puts them. A request that opens a segment
    /// allocates the segment at its capacity. After an error, the caller must free the arena.
    pub fn alloc(self: *VramArena, size: vk.DeviceSize) AllocError!View {
        // The alloc sequence at debug scope is the VRAM map. Allocations follow the deterministic
        // weight-upload order, so a reader pairs each line with the load sequence.
        log.debug(
            "arena alloc: {d} KiB ({d} MiB held before)",
            .{ size / 1024, self.totalUsed() / (1024 * 1024) },
        );
        try self.segments.ensureUnusedCapacity(self.gpa, 1);
        const at = try self.packing.place(self.gpa, size);
        if (at.opens) {
            const capacity = self.packing.fills.items[at.segment].capacity;
            self.segments.appendAssumeCapacity(try self.createSegment(capacity));
        }

        std.debug.assert(self.segments.items.len == self.packing.fills.items.len);
        const segment = self.segments.items[at.segment];
        return .{ .buffer = segment.buffer, .offset = at.offset, .range = size };
    }

    /// One debug line per segment: the coarse VRAM map. Call after init,
    /// when the arena holds its final layout.
    pub fn logSegments(self: *const VramArena, name: []const u8) void {
        const mib = 1024 * 1024;
        for (self.packing.fills.items, 0..) |fill, i| {
            log.debug(
                "arena {s} seg {d}: {d} of {d} MiB used",
                .{ name, i, fill.used / mib, fill.capacity / mib },
            );
        }
    }

    /// Total bytes claimed by allocations so far, the sum of `used` across the
    /// segments. The traces read it.
    pub fn totalUsed(self: *const VramArena) vk.DeviceSize {
        return self.packing.usedBytes();
    }

    fn createSegment(self: *VramArena, capacity: vk.DeviceSize) AllocError!Segment {
        const buf_info: vk.BufferCreateInfo = .{
            .size = capacity,
            .usage = self.usage,
            .sharing_mode = .exclusive,
        };
        const buffer = try self.ctx.vkd.createBuffer(self.ctx.device, &buf_info, null);
        errdefer self.ctx.vkd.destroyBuffer(self.ctx.device, buffer, null);

        const reqs = self.ctx.vkd.getBufferMemoryRequirements(self.ctx.device, buffer);
        const required: vk.MemoryPropertyFlags = .{ .device_local_bit = true };
        const mem_type = vulkan_ctx.findMemoryType(
            self.ctx.mem_props,
            reqs.memory_type_bits,
            required,
        ) orelse
            return error.NoSuitableMemoryType;

        const alloc_info: vk.MemoryAllocateInfo = .{
            .allocation_size = reqs.size,
            .memory_type_index = mem_type,
        };
        const memory = try self.ctx.vkd.allocateMemory(self.ctx.device, &alloc_info, null);
        errdefer self.ctx.vkd.freeMemory(self.ctx.device, memory, null);

        try self.ctx.vkd.bindBufferMemory(self.ctx.device, buffer, memory, 0);
        log.debug("vram segment {d}: {d} MiB", .{ self.segments.items.len, capacity >> 20 });
        self.memory_bytes += reqs.size;
        return .{ .buffer = buffer, .memory = memory };
    }
};

// -- state ------------------------------------------------------------------

/// Host buffers this process created. A test bounds the count of one runner init with it.
var host_buffers_created: std.atomic.Value(u64) = .init(0);

// -- api --------------------------------------------------------------------

/// The number of host buffers this process created.
pub fn hostBufferCount() u64 {
    return host_buffers_created.load(.monotonic);
}

// -- helpers ----------------------------------------------------------------

fn alignUp(value: vk.DeviceSize, alignment: vk.DeviceSize) vk.DeviceSize {
    if (alignment == 0) return value;
    return ((value + alignment - 1) / alignment) * alignment;
}

// -- tests ------------------------------------------------------------------

/// A store over fake buffer handles. The accessors read only the handles and
/// the capacities, so the tests need no device and no context.
fn fakeKvStore(layout: kv_index.Layout, lim: limits.BufferLimits) KvStore {
    const packing = kv_index.pack(layout, lim);
    var store: KvStore = .{
        .layout = layout,
        .packing = packing,
        .lim = lim,
        .segments = @splat(.{ .ctx = undefined }),
    };
    for (store.segments[0..packing.n_segments], 0..) |*segment, i| {
        segment.buffer = @enumFromInt(i + 1);
        segment.capacity = packing.segmentBytes(@intCast(i));
    }

    return store;
}

test "KvStore: every row of every region lands once inside one allocation" {
    // Six regions of 192 bytes, two to an allocation of 384 bytes.
    const layout = kv_index.Layout.init(3, 4, 4, 3, .f32);
    const lim: limits.BufferLimits = .{
        .storage_range = 192,
        .allocation_size = 384,
        .offset_alignment = 16,
    };
    const store = fakeKvStore(layout, lim);
    try std.testing.expectEqual(@as(u32, 3), store.packing.n_segments);
    try std.testing.expectEqual(layout.totalBytes(), store.deviceBytes());

    // Walk every row, and mark each row position that it lands on.
    const row_bytes = layout.rowBytes();
    const rows_per_segment = 24;
    try std.testing.expectEqual(lim.allocation_size, rows_per_segment * row_bytes);
    var seen = [_][rows_per_segment]bool{[_]bool{false} ** rows_per_segment} ** 3;
    var count: u32 = 0;
    for (0..layout.n_layers) |l| {
        const layer: u32 = @intCast(l);
        for ([_]kv_index.KvKind{ .k, .v }) |t| {
            const whole = store.region(layer, t);
            try std.testing.expectEqual(layout.regionBytes(), whole.range);
            try std.testing.expectEqual(@as(u64, 0), whole.offset % lim.offset_alignment);

            for (0..layout.n_slots) |s| {
                const slot: u32 = @intCast(s);
                const view = store.slotView(layer, t, slot);
                try std.testing.expectEqual(whole.buffer, view.buffer);
                try std.testing.expectEqual(
                    whole.offset + layout.slotByteOffset(slot),
                    view.offset,
                );
                try std.testing.expectEqual(@as(u64, layout.window) * row_bytes, view.range);

                for (0..layout.window) |p| {
                    const row = store.rowAt(layer, t, slot, @intCast(p));
                    try std.testing.expectEqual(view.buffer, row.buffer);
                    // The rows of one slot fill its view in position order.
                    try std.testing.expectEqual(view.offset + @as(u64, p) * row_bytes, row.offset);
                    const segment = @intFromEnum(row.buffer) - 1;
                    try std.testing.expect(segment < store.packing.n_segments);
                    try std.testing.expect(
                        row.offset + row_bytes <= store.segments[segment].capacity,
                    );
                    const at: usize = @intCast(row.offset / row_bytes);
                    try std.testing.expect(!seen[segment][at]);
                    seen[segment][at] = true;
                    count += 1;
                }
            }
        }
    }

    // The rows hold as many bytes as the allocations, so no allocation byte
    // stays outside a row.
    try std.testing.expectEqual(@as(u32, 3 * rows_per_segment), count);
}

test "KvStore: two 2B sessions under the raised range start slot 1 at 128 MiB in each region" {
    // Qwen3.5-2B at f16: six full-attention layers, two slots, and 2 GiB allocations. Each region
    // holds 256 MiB, so eight regions share the first allocation and four the second.
    const layout = kv_index.Layout.init(2, 131_072, 512, 6, .f16);
    const lim: limits.BufferLimits = .{
        .storage_range = 536_870_912,
        .allocation_size = 2_147_483_648,
        .offset_alignment = 16,
    };
    const store = fakeKvStore(layout, lim);
    const mib: u64 = 1 << 20;
    try std.testing.expectEqual(@as(u32, 2), store.packing.n_segments);
    try std.testing.expectEqual(3072 * mib, store.deviceBytes());

    const cases = [_]struct { layer: u32, t: kv_index.KvKind, segment: u64, offset: u64 }{
        .{ .layer = 0, .t = .k, .segment = 0, .offset = 0 },
        .{ .layer = 5, .t = .k, .segment = 0, .offset = 1280 * mib },
        .{ .layer = 1, .t = .v, .segment = 0, .offset = 1792 * mib },
        .{ .layer = 2, .t = .v, .segment = 1, .offset = 0 },
        .{ .layer = 5, .t = .v, .segment = 1, .offset = 768 * mib },
    };
    for (cases) |c| {
        const whole = store.region(c.layer, c.t);
        try std.testing.expectEqual(c.segment + 1, @intFromEnum(whole.buffer));
        try std.testing.expectEqual(c.offset, whole.offset);
        try std.testing.expectEqual(256 * mib, whole.range);
        const second = store.slotView(c.layer, c.t, 1);
        try std.testing.expectEqual(c.offset + 128 * mib, second.offset);
        try std.testing.expectEqual(128 * mib, second.range);
        const last = store.rowAt(c.layer, c.t, 1, 131_071);
        try std.testing.expectEqual(c.offset + 256 * mib - 1024, last.offset);
    }
}

test "VramArena: a plan sizes each segment to its fill and keeps every offset" {
    // A synthetic upload order into segments of 100 bytes at a 16-byte alignment. The 150-byte
    // request opens a segment of its own, and three requests fill the tail of an older segment.
    const gpa = std.testing.allocator;
    const sizes = [_]vk.DeviceSize{ 40, 30, 50, 150, 20, 70, 10, 33, 64, 8, 5 };
    const segments = [_]u32{ 0, 0, 1, 2, 0, 3, 3, 1, 4, 4, 4 };
    const offsets = [_]vk.DeviceSize{ 0, 48, 0, 0, 80, 0, 80, 64, 0, 64, 80 };
    const request_bytes: u64 = 480;

    // The dry run takes the default capacity, so segments 1, 3, and 4 keep a tail.
    var dry: VramArena.Packing = .init(100, 16);
    defer dry.deinit(gpa);
    for (sizes, segments, offsets) |size, segment, offset| {
        const at = try dry.place(gpa, size);
        try std.testing.expectEqual(segment, at.segment);
        try std.testing.expectEqual(offset, at.offset);
    }
    try std.testing.expectEqual(@as(u64, 550), dry.capacityBytes());
    try std.testing.expectEqual(@as(u64, 522), dry.usedBytes());
    try std.testing.expectEqual(request_bytes, dry.request_bytes);

    // The plan opens each segment at its fill, and every request lands where the dry run put it.
    var planned: VramArena.Packing = .init(100, 16);
    defer planned.deinit(gpa);
    try planned.setPlan(gpa, &dry);
    for (sizes, segments, offsets) |size, segment, offset| {
        const at = try planned.place(gpa, size);
        try std.testing.expectEqual(segment, at.segment);
        try std.testing.expectEqual(offset, at.offset);
    }
    planned.assertFull();

    // Five segments hold the requests plus 42 bytes of padding, under one alignment per request.
    const capacities = [_]vk.DeviceSize{ 100, 97, 150, 90, 85 };
    try std.testing.expectEqual(capacities.len, planned.fills.items.len);
    for (capacities, planned.fills.items) |capacity, fill| {
        try std.testing.expectEqual(capacity, fill.capacity);
    }
    try std.testing.expectEqual(request_bytes + 42, planned.capacityBytes());
    try std.testing.expectEqual(@as(u64, 11 * 16), planned.paddingBound());
    try std.testing.expect(planned.capacityBytes() <= request_bytes + planned.paddingBound());
}
