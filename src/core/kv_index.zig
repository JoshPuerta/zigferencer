//! The index algebra of the device KV cache. The cache holds one region per
//! layer and tensor. Region `t * n_layers + layer` holds the rows of every slot
//! in row-major order: slot, then position, then element. An element offset is
//! therefore a mixed-radix numeral with the radices `window` and `per_step`
//! (Knuth, TAOCP vol 1, section 2.2.6). So each coordinate of the index box
//! gets its own element, and every element stays below the region size. The
//! device limits bind per region, and the regions pack into few allocations.
//! The module is pure: no allocation, no device, and no I/O.

const std = @import("std");
const prop = @import("prop.zig");
const dtype = @import("dtype.zig");
const limits = @import("limits.zig");

const assert = std.debug.assert;
const KvDtype = dtype.KvDtype;

// -- constants --------------------------------------------------------------

/// The widest element of any admitted KV dtype.
const max_elem_bytes: u64 = blk: {
    var widest: u64 = 0;
    for (std.enums.values(KvDtype)) |d| {
        widest = @max(widest, elemBytes(d));
    }

    break :blk widest;
};

comptime {
    // The widest layout the bounds admit keeps its byte total inside a `u64`.
    const widest: u128 = 2 * @as(u128, limits.max_layers) * limits.max_vram_slots *
        limits.max_model_ctx * limits.max_kv_row_elems * max_elem_bytes;
    assert(widest <= std.math.maxInt(u64));
}

// -- types ------------------------------------------------------------------

/// The two tensors of one layer. The value orders the regions: every K region
/// comes before every V region.
pub const Tensor = enum(u1) { k = 0, v = 1 };

/// The shape of the device KV cache: `n_slots` slots of `window` positions,
/// with rows of `per_step` elements, in `n_layers` layers that hold KV.
pub const Layout = struct {
    n_slots: u32,
    window: u32,
    per_step: u32,
    n_layers: u32,
    kv_dtype: KvDtype,

    pub fn init(n_slots: u32, window: u32, per_step: u32, n_layers: u32, kv_dtype: KvDtype) Layout {
        const layout: Layout = .{
            .n_slots = n_slots,
            .window = window,
            .per_step = per_step,
            .n_layers = n_layers,
            .kv_dtype = kv_dtype,
        };
        layout.assertValid();
        assert(layout.regionBytes() >= 1);
        return layout;
    }

    /// The regions of the cache: one K region and one V region per layer.
    pub fn regionCount(self: Layout) u32 {
        self.assertValid();
        const count = 2 * self.n_layers;
        assert(count >= 2);
        assert(count <= 2 * limits.max_layers);
        return count;
    }

    /// The index of the region that holds tensor `t` of `layer`.
    pub fn regionIndex(self: Layout, layer: u32, t: Tensor) u32 {
        assert(layer < self.n_layers);

        const region = @as(u32, @intFromEnum(t)) * self.n_layers + layer;
        assert(region < self.regionCount());
        return region;
    }

    /// Bytes one row of `per_step` elements occupies.
    pub fn rowBytes(self: Layout) u64 {
        self.assertValid();
        const bytes = self.kv_dtype.rowBytes(self.per_step);
        assert(bytes >= 1);
        assert(bytes == @as(u64, self.per_step) * elemBytes(self.kv_dtype));
        return bytes;
    }

    /// Elements one region holds: every row of every slot.
    pub fn regionElems(self: Layout) u64 {
        self.assertValid();
        const elems = @as(u64, self.n_slots) * self.window * self.per_step;
        assert(elems >= 1);
        return elems;
    }

    /// Bytes one region holds.
    pub fn regionBytes(self: Layout) u64 {
        const bytes = @as(u64, self.n_slots) * self.window * self.rowBytes();
        assert(bytes == self.regionElems() * elemBytes(self.kv_dtype));
        return bytes;
    }

    /// Bytes the `window` rows of one slot hold in one region.
    pub fn slotRegionBytes(self: Layout) u64 {
        const bytes = @as(u64, self.window) * self.rowBytes();
        assert(bytes * self.n_slots == self.regionBytes());
        return bytes;
    }

    /// Bytes one position of one slot holds across every region: the KV
    /// price of one token.
    pub fn tokenBytes(self: Layout) u64 {
        const bytes = @as(u64, self.regionCount()) * self.rowBytes();
        assert(bytes >= 2);
        return bytes;
    }

    /// The element index of `(slot, pos, elem)` inside a region.
    pub fn offset(self: Layout, slot: u32, pos: u32, elem: u32) u64 {
        assert(slot < self.n_slots);
        assert(pos < self.window);
        assert(elem < self.per_step);

        const index = (@as(u64, slot) * self.window + pos) * self.per_step + elem;
        assert(index < self.regionElems());
        return index;
    }

    /// The byte offset of the row at `(slot, pos)` inside a region.
    pub fn rowByteOffset(self: Layout, slot: u32, pos: u32) u64 {
        assert(slot < self.n_slots);
        assert(pos < self.window);

        const row_bytes = self.rowBytes();
        const at = (@as(u64, slot) * self.window + pos) * row_bytes;
        assert(at == self.offset(slot, pos, 0) * elemBytes(self.kv_dtype));
        assert(at + row_bytes <= self.regionBytes());
        return at;
    }

    /// The byte offset of the first row of `slot` inside a region.
    pub fn slotByteOffset(self: Layout, slot: u32) u64 {
        assert(slot < self.n_slots);

        const at = @as(u64, slot) * self.slotRegionBytes();
        assert(at == self.rowByteOffset(slot, 0));
        return at;
    }

    /// Elements between two slots inside a region: the shader stride from one
    /// slot to the next.
    pub fn streamStrideElems(self: Layout) u32 {
        // The shader multiplies this stride by a slot index in 32 bits, so the
        // whole region must stay inside the 31-bit index before the narrow.
        assert(self.regionElems() <= limits.max_kv_index_elems);

        const stride = @as(u64, self.window) * self.per_step;
        assert(stride * self.n_slots == self.regionElems());
        return std.math.cast(u32, stride) orelse unreachable;
    }

    /// Bytes one slot holds across every region.
    pub fn slotBytes(self: Layout) u64 {
        const bytes = @as(u64, self.regionCount()) * self.slotRegionBytes();
        assert(bytes == @as(u64, self.window) * self.tokenBytes());
        assert(bytes * self.n_slots == self.totalBytes());
        return bytes;
    }

    /// Bytes every region holds together.
    pub fn totalBytes(self: Layout) u64 {
        const region_bytes = self.regionBytes();
        const bytes = @as(u64, self.regionCount()) * region_bytes;
        assert(bytes >= 2 * region_bytes);
        return bytes;
    }

    /// Bytes of one slot's first `n_rows` rows in every region: the blob an
    /// evict writes.
    pub fn blobBytes(self: Layout, n_rows: u32) u64 {
        assert(n_rows <= self.window);

        const bytes = @as(u64, self.regionCount()) * n_rows * self.rowBytes();
        assert(bytes <= self.slotBytes());
        if (n_rows == self.window) assert(bytes == self.slotBytes());

        return bytes;
    }

    fn assertValid(self: Layout) void {
        assert(self.n_slots >= 1);
        assert(self.n_slots <= limits.max_vram_slots);
        assert(self.window >= 1);
        assert(self.window <= limits.max_model_ctx);
        assert(self.per_step >= 1);
        assert(self.per_step <= limits.max_kv_row_elems);

        // The f16 shaders read a row in 32-bit word pairs.
        if (self.kv_dtype == .f16) assert(self.per_step % 2 == 0);

        assert(self.n_layers >= 1);
        assert(self.n_layers <= limits.max_layers);
    }
};

/// Equal-size regions packed into allocations: region `r` lives in segment
/// `r / per_segment` at byte `(r % per_segment) * region_bytes`.
pub const Packing = struct {
    region_bytes: u64,
    n_regions: u32,
    per_segment: u32,
    n_segments: u32,

    /// The allocation that holds `region`.
    pub fn segmentOf(self: Packing, region: u32) u32 {
        assert(region < self.n_regions);
        assert(self.per_segment >= 1);

        const segment = region / self.per_segment;
        assert(segment < self.n_segments);
        return segment;
    }

    /// The byte offset of `region` inside its allocation.
    pub fn offsetOf(self: Packing, region: u32) u64 {
        assert(region < self.n_regions);

        const at = @as(u64, region % self.per_segment) * self.region_bytes;
        assert(at + self.region_bytes <= self.segmentBytes(self.segmentOf(region)));
        return at;
    }

    /// Bytes of one allocation. The last one holds the regions left over.
    pub fn segmentBytes(self: Packing, segment: u32) u64 {
        assert(segment < self.n_segments);

        const first = segment * self.per_segment;
        assert(first < self.n_regions);
        const count = @min(self.per_segment, self.n_regions - first);
        assert(count >= 1);
        return @as(u64, count) * self.region_bytes;
    }
};

/// The device limit a verdict names: the words a log line uses for it, its
/// value, and the unit of that value.
pub const Limit = struct {
    name: []const u8,
    value: u64,
    unit: Unit,

    pub const Unit = enum { elements, bytes };
};

// -- api --------------------------------------------------------------------

/// The first device limit one region breaks. There is one home for the
/// verdict: `limits.bufferFits`.
pub fn fits(layout: Layout, lim: limits.BufferLimits) limits.BufferVerdict {
    const elems = layout.regionElems();
    const bytes = layout.regionBytes();
    const verdict = limits.bufferFits(elems, bytes, lim);
    if (verdict == .fits) {
        assert(elems <= limits.max_kv_index_elems);
        assert(bytes <= lim.allocation_size);
        assert(bytes <= lim.storage_range);
    }

    return verdict;
}

/// The deepest window whose region of `n_slots` slots passes the 31-bit index,
/// the allocation size, and the storage range. It is not capped by any model
/// ceiling, so a caller takes the minimum with its own window. It leaves the
/// alignment out: a row of whole alignment units keeps every window aligned.
/// Zero means that no window fits.
pub fn maxWindowForLimits(
    n_slots: u32,
    per_step: u32,
    kv_dtype: KvDtype,
    lim: limits.BufferLimits,
) u32 {
    assert(n_slots >= 1);
    assert(n_slots <= limits.max_vram_slots);
    assert(per_step >= 1);
    assert(per_step <= limits.max_kv_row_elems);

    const per_position: u64 = @as(u64, n_slots) * per_step;
    const cap = regionElemCap(kv_dtype, lim);
    const window = std.math.cast(u32, cap / per_position) orelse unreachable;
    assert(@as(u64, window) * per_position <= cap);
    // One more position breaks a limit.
    assert((@as(u64, window) + 1) * per_position > cap);
    return window;
}

/// The most slots whose region of `window` positions passes the 31-bit index,
/// the allocation size, and the storage range: the dual of
/// `maxWindowForLimits`. It is not capped by the slot ceiling, so a caller
/// takes the minimum with its own. It leaves the alignment out for the same
/// reason. Zero means that one slot already breaks a limit.
pub fn maxSlotsForLimits(
    window: u32,
    per_step: u32,
    kv_dtype: KvDtype,
    lim: limits.BufferLimits,
) u32 {
    assert(window >= 1);
    assert(window <= limits.max_model_ctx);
    assert(per_step >= 1);
    assert(per_step <= limits.max_kv_row_elems);

    const per_slot: u64 = @as(u64, window) * per_step;
    const cap = regionElemCap(kv_dtype, lim);
    const slots = std.math.cast(u32, cap / per_slot) orelse unreachable;
    assert(@as(u64, slots) * per_slot <= cap);
    // One more slot breaks a limit.
    assert((@as(u64, slots) + 1) * per_slot > cap);
    return slots;
}

/// The deepest window in whole `limits.window_granule` units whose region of
/// `n_slots` slots passes the device limits. Zero means that no granule fits.
pub fn maxGranuleWindow(
    n_slots: u32,
    per_step: u32,
    kv_dtype: KvDtype,
    lim: limits.BufferLimits,
) u32 {
    const deepest = maxWindowForLimits(n_slots, per_step, kv_dtype, lim);
    const window: u32 = @intCast(limits.floorToGranule(deepest));
    assert(window <= deepest);
    assert(window % limits.window_granule == 0);
    return window;
}

/// The limit that `verdict` names under `lim`. Every log line that names a
/// broken limit reads its words and its value here.
pub fn limitOf(verdict: limits.BufferVerdict, lim: limits.BufferLimits) Limit {
    assert(verdict != .fits);

    const limit: Limit = switch (verdict) {
        .fits => unreachable,
        .index_past_31_bits => .{
            .name = "shader index limit",
            .value = limits.max_kv_index_elems,
            .unit = .elements,
        },
        .past_allocation_size => .{
            .name = "device allocation limit",
            .value = lim.allocation_size,
            .unit = .bytes,
        },
        .past_storage_range => .{
            .name = "storage-buffer range",
            .value = lim.storage_range,
            .unit = .bytes,
        },
        .misaligned => .{
            .name = "storage-buffer offset alignment",
            .value = lim.offset_alignment,
            .unit = .bytes,
        },
    };

    assert(limit.value >= 1);
    if (limit.unit == .elements) assert(verdict == .index_past_31_bits);

    return limit;
}

/// Pack the regions of `layout` into allocations of at most
/// `lim.allocation_size` bytes, in region order. Every region must pass the
/// limits first.
pub fn pack(layout: Layout, lim: limits.BufferLimits) Packing {
    assert(fits(layout, lim) == .fits);

    const region_bytes = layout.regionBytes();
    const n_regions = layout.regionCount();
    // A region inside the allocation size leaves room for one per allocation.
    const by_size = lim.allocation_size / region_bytes;
    assert(by_size >= 1);
    const per_segment: u32 = @intCast(@min(@as(u64, n_regions), by_size));
    const packing: Packing = .{
        .region_bytes = region_bytes,
        .n_regions = n_regions,
        .per_segment = per_segment,
        .n_segments = (n_regions + per_segment - 1) / per_segment,
    };
    assert(packing.n_segments >= 1);
    assert(packing.n_segments <= n_regions);
    // At most `2 * limits.max_layers` segments, one per region at worst.
    var sum: u64 = 0;
    var segment: u32 = 0;
    while (segment < packing.n_segments) : (segment += 1) {
        const bytes = packing.segmentBytes(segment);
        assert(bytes <= lim.allocation_size);
        sum += bytes;
    }

    assert(sum == layout.totalBytes());
    return packing;
}

// -- helpers ----------------------------------------------------------------

/// The most elements one region holds under the 31-bit index, the allocation
/// size, and the storage range. The byte limits hold whole elements, so one
/// element past the cap breaks one of the three limits.
fn regionElemCap(kv_dtype: KvDtype, lim: limits.BufferLimits) u64 {
    assert(lim.storage_range >= 1);
    assert(lim.allocation_size >= 1);

    const elem_bytes = elemBytes(kv_dtype);
    const by_bytes = @min(lim.allocation_size, lim.storage_range) / elem_bytes;
    const cap = @min(limits.max_kv_index_elems, by_bytes);
    assert(cap <= limits.max_kv_index_elems);
    assert(cap * elem_bytes <= lim.allocation_size);
    assert(cap * elem_bytes <= lim.storage_range);
    return cap;
}

/// Bytes one element of `kv_dtype` occupies. Every admitted KV dtype has a
/// block of one element, and `Layout.rowBytes` asserts that a row is a whole
/// number of these.
fn elemBytes(kv_dtype: KvDtype) u64 {
    const bytes = kv_dtype.rowBytes(1);
    assert(bytes >= 1);
    return bytes;
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

/// Limits with every bound far above any test region, so a test moves one.
const wide_limits: limits.BufferLimits = .{
    .storage_range = 1 << 40,
    .allocation_size = 1 << 40,
    .offset_alignment = 16,
};

/// A driver that reports the full 32-bit storage range.
const full_range_limits: limits.BufferLimits = .{
    .storage_range = 4_294_967_295,
    .allocation_size = 4_294_967_292,
    .offset_alignment = 4,
};

/// A translation-layer driver that reports the spec minimum storage range.
const spec_min_limits: limits.BufferLimits = .{
    .storage_range = 134_217_728,
    .allocation_size = 2_147_483_648,
    .offset_alignment = 16,
};

/// The same driver with the storage range raised to 512 MiB by the
/// `limits.storage_range_env` override.
const raised_range_limits: limits.BufferLimits = .{
    .storage_range = 536_870_912,
    .allocation_size = 2_147_483_648,
    .offset_alignment = 16,
};

test "offset: every coordinate of a small box gets its own element below the region" {
    for ([_]KvDtype{ .f32, .f16 }) |kv_dtype| {
        var n_slots: u32 = 1;
        while (n_slots <= 3) : (n_slots += 1) {
            var window: u32 = 1;
            while (window <= 5) : (window += 1) {
                var per_step: u32 = 1;
                while (per_step <= 4) : (per_step += 1) {
                    if (kv_dtype == .f16 and per_step % 2 != 0) continue;

                    const layout = Layout.init(n_slots, window, per_step, 1, kv_dtype);
                    var seen = [_]bool{false} ** 64;
                    var count: u64 = 0;
                    for (0..n_slots) |s| {
                        for (0..window) |p| {
                            for (0..per_step) |e| {
                                const at = layout.offset(@intCast(s), @intCast(p), @intCast(e));
                                try testing.expect(at < layout.regionElems());
                                try testing.expect(!seen[@intCast(at)]);
                                seen[@intCast(at)] = true;
                                count += 1;
                            }
                        }
                    }

                    // The box and the region have one size, so the offset
                    // covers every element of the region.
                    try testing.expectEqual(layout.regionElems(), count);
                    for (0..n_slots) |s| {
                        const slot: u32 = @intCast(s);
                        try testing.expectEqual(
                            layout.rowByteOffset(slot, 0),
                            layout.slotByteOffset(slot),
                        );
                        for (0..window) |p| {
                            const at = layout.rowByteOffset(slot, @intCast(p));
                            try testing.expectEqual(
                                layout.offset(slot, @intCast(p), 0) * elemBytes(kv_dtype),
                                at,
                            );
                        }
                    }
                }
            }
        }
    }
}

test "fits: each limit passes at its bound and fails one position past it" {
    // The storage range, at 1024 bytes a row with every other bound wide.
    var lim = wide_limits;
    lim.storage_range = 4096 * 1024;
    try testing.expectEqual(
        limits.BufferVerdict.fits,
        fits(Layout.init(1, 4096, 512, 1, .f16), lim),
    );
    try testing.expectEqual(
        limits.BufferVerdict.past_storage_range,
        fits(Layout.init(1, 4097, 512, 1, .f16), lim),
    );
    // The allocation size, with the storage range wide.
    lim = wide_limits;
    lim.allocation_size = 2 * 4096 * 1024;
    try testing.expectEqual(
        limits.BufferVerdict.fits,
        fits(Layout.init(2, 4096, 512, 1, .f16), lim),
    );
    try testing.expectEqual(
        limits.BufferVerdict.past_allocation_size,
        fits(Layout.init(2, 4097, 512, 1, .f16), lim),
    );
    // The 31-bit index: 16 slots of 131072 positions of 1024 elements hold
    // exactly 2^31 elements.
    try testing.expectEqual(
        limits.BufferVerdict.fits,
        fits(Layout.init(16, 131_072, 1024, 1, .f32), wide_limits),
    );
    try testing.expectEqual(
        limits.BufferVerdict.index_past_31_bits,
        fits(Layout.init(16, 131_073, 1024, 1, .f32), wide_limits),
    );
    // The alignment: a 4-byte row fills a 16-byte unit every fourth position.
    try testing.expectEqual(
        limits.BufferVerdict.fits,
        fits(Layout.init(1, 4, 1, 1, .f32), wide_limits),
    );
    try testing.expectEqual(
        limits.BufferVerdict.misaligned,
        fits(Layout.init(1, 5, 1, 1, .f32), wide_limits),
    );
}

test "anchors: the region sizes and verdicts of the two registry shapes" {
    // Qwen3.5-2B at f16, one slot at the 262144 model ceiling: six
    // full-attention layers of 512-element rows.
    const qwen = Layout.init(1, 262_144, 512, 6, .f16);
    try testing.expectEqual(@as(u64, 268_435_456), qwen.regionBytes());
    try testing.expectEqual(@as(u64, 3_221_225_472), qwen.totalBytes());
    // Llama-3.1-8B at f16, one slot at the 131072 model ceiling.
    const llama = Layout.init(1, 131_072, 1024, 32, .f16);
    try testing.expectEqual(@as(u64, 268_435_456), llama.regionBytes());
    try testing.expectEqual(@as(u64, 17_179_869_184), llama.totalBytes());
    // At one slot the slot rows of a region are the whole region. A token
    // costs one K row and one V row per layer.
    try testing.expectEqual(qwen.regionBytes(), qwen.slotRegionBytes());
    try testing.expectEqual(@as(u64, 12_288), qwen.tokenBytes());
    try testing.expectEqual(@as(u64, 131_072), llama.tokenBytes());
    // The full 32-bit range accepts both regions.
    try testing.expectEqual(limits.BufferVerdict.fits, fits(qwen, full_range_limits));
    try testing.expectEqual(limits.BufferVerdict.fits, fits(llama, full_range_limits));
    // The 128 MiB storage range accepts neither, and the deepest windows it
    // accepts at one slot are 131072 and 65536.
    try testing.expectEqual(limits.BufferVerdict.past_storage_range, fits(qwen, spec_min_limits));
    try testing.expectEqual(limits.BufferVerdict.past_storage_range, fits(llama, spec_min_limits));
    try testing.expectEqual(@as(u32, 131_072), maxWindowForLimits(1, 512, .f16, spec_min_limits));
    try testing.expectEqual(@as(u32, 65_536), maxWindowForLimits(1, 1024, .f16, spec_min_limits));
    // The 512 MiB override accepts both.
    try testing.expectEqual(limits.BufferVerdict.fits, fits(qwen, raised_range_limits));
    try testing.expectEqual(limits.BufferVerdict.fits, fits(llama, raised_range_limits));
}

test "pack: two 2B sessions at 131072 under the override take two allocations" {
    // Each region holds `2 * 131072 * 1024` bytes, 256 MiB. A 2 GiB allocation
    // holds eight of the twelve, so the KV takes 2 GiB and 1 GiB.
    const layout = Layout.init(2, 131_072, 512, 6, .f16);
    try testing.expectEqual(limits.BufferVerdict.past_storage_range, fits(layout, spec_min_limits));
    const packing = pack(layout, raised_range_limits);
    try testing.expectEqual(@as(u32, 8), packing.per_segment);
    try testing.expectEqual(@as(u32, 2), packing.n_segments);
    try testing.expectEqual(@as(u64, 2_147_483_648), packing.segmentBytes(0));
    try testing.expectEqual(@as(u64, 1_073_741_824), packing.segmentBytes(1));
    try testing.expectEqual(@as(u64, 3_221_225_472), layout.totalBytes());
    // Slot 1 starts 128 MiB into each region, past the reported range.
    try testing.expectEqual(@as(u64, 134_217_728), layout.slotByteOffset(1));
}

fn checkPackRegions(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const kv_dtype: KvDtype = if (r.boolean()) .f16 else .f32;
    // A multiple of 8 elements keeps a row a whole number of 16-byte units.
    const per_step = 8 * r.intRangeAtMost(u32, 1, 128);
    const layout = Layout.init(
        r.intRangeAtMost(u32, 1, 8),
        r.intRangeAtMost(u32, 1, 4096),
        per_step,
        r.intRangeAtMost(u32, 1, limits.max_layers),
        kv_dtype,
    );
    const region_bytes = layout.regionBytes();
    var lim = wide_limits;
    lim.allocation_size = region_bytes * r.intRangeAtMost(u64, 1, layout.regionCount() + 2) +
        r.intRangeLessThan(u64, 0, region_bytes);
    try testing.expectEqual(limits.BufferVerdict.fits, fits(layout, lim));
    const packing = pack(layout, lim);
    const n_regions = layout.regionCount();
    try testing.expectEqual(
        @min(@as(u64, n_regions), lim.allocation_size / region_bytes),
        packing.per_segment,
    );
    try testing.expectEqual(
        (n_regions + packing.per_segment - 1) / packing.per_segment,
        packing.n_segments,
    );
    var sum: u64 = 0;
    for (0..packing.n_segments) |s| {
        const bytes = packing.segmentBytes(@intCast(s));
        try testing.expect(bytes <= lim.allocation_size);
        sum += bytes;
    }

    try testing.expectEqual(layout.totalBytes(), sum);
    for (0..n_regions) |i| {
        const region: u32 = @intCast(i);
        const segment = packing.segmentOf(region);
        const at = packing.offsetOf(region);
        try testing.expect(at + region_bytes <= packing.segmentBytes(segment));
        try testing.expect(at % lim.offset_alignment == 0);
        // The segment and the slot inside it name the region back, so no
        // two regions share a place.
        try testing.expectEqual(
            region,
            segment * packing.per_segment + @as(u32, @intCast(at / region_bytes)),
        );
    }
}

test "pack: every region lands once, aligned, inside an allocation" {
    try prop.forAll(
        .{ .site = 0x4B56_1D8C, .cases = 500 },
        {},
        u64,
        prop.genSeed(void),
        checkPackRegions,
    );
}

test "Layout: at one slot the regions reproduce the offsets of the CPU KvCache slab" {
    // The CPU `KvCache` slab holds layer `l` of tensor `t` at `(t * L + l) * W`
    // rows, and position `p` one row further on. The slab holds `2 * L * W`
    // rows.
    const Case = struct {
        window: u32,
        per_step: u32,
        n_layers: u32,
        kv_dtype: KvDtype,
        allocation_size: u64,
    };
    const cases = [_]Case{
        // SmolLM2-135M at f32 in one allocation.
        .{ .window = 2048, .per_step = 192, .n_layers = 30, .kv_dtype = .f32, .allocation_size = 1 << 40 },
        // Llama-3.1-8B at f16, split over allocations of five regions.
        .{ .window = 512, .per_step = 1024, .n_layers = 32, .kv_dtype = .f16, .allocation_size = 5 * 512 * 2048 },
    };
    for (cases) |c| {
        const layout = Layout.init(1, c.window, c.per_step, c.n_layers, c.kv_dtype);
        const row_bytes = c.kv_dtype.rowBytes(c.per_step);
        try testing.expectEqual(
            2 * @as(u64, c.n_layers) * c.window * row_bytes,
            layout.slotBytes(),
        );
        try testing.expectEqual(layout.slotBytes(), layout.totalBytes());
        var lim = wide_limits;
        lim.allocation_size = c.allocation_size;
        const packing = pack(layout, lim);
        // Each segment starts where the one before it ends.
        var segment_base = [_]u64{0} ** (2 * limits.max_layers);
        for (1..packing.n_segments) |s| {
            segment_base[s] = segment_base[s - 1] + packing.segmentBytes(@intCast(s - 1));
        }

        for (0..c.n_layers) |l| {
            const layer: u32 = @intCast(l);
            try testing.expectEqual(layer, layout.regionIndex(layer, .k));
            try testing.expectEqual(c.n_layers + layer, layout.regionIndex(layer, .v));
            for ([_]Tensor{ .k, .v }) |t| {
                const region = layout.regionIndex(layer, t);
                const base = segment_base[packing.segmentOf(region)] + packing.offsetOf(region);
                for ([_]u32{ 0, 1, c.window / 2, c.window - 1 }) |pos| {
                    const single = (@as(
                        u64,
                        @intFromEnum(t),
                    ) * c.n_layers + layer) * c.window * row_bytes +
                        @as(u64, pos) * row_bytes;
                    try testing.expectEqual(single, base + layout.rowByteOffset(0, pos));
                }
            }
        }
    }
}

test "kvPlan conservation: the regions plus the fixed state are the plan's KV" {
    const kv_budget = @import("kv_budget.zig");
    const qwen_2b: kv_budget.KvShape.Recurrent = .{
        .full_attention_interval = 4,
        .conv_kernel = 4,
        .state_size = 128,
        .group_count = 16,
        .time_step_rank = 16,
    };
    var qwen_9b = qwen_2b;
    qwen_9b.time_step_rank = 32;
    const Shape = struct { shape: kv_budget.KvShape, kv_dtype: KvDtype, model_ctx: u32 };
    const shapes = [_]Shape{
        .{ .shape = .{ .n_layers = 30, .kv_row_elems = 192 }, .kv_dtype = .f32, .model_ctx = 8192 },
        .{ .shape = .{ .n_layers = 32, .kv_row_elems = 1024 }, .kv_dtype = .f16, .model_ctx = 131_072 },
        .{ .shape = .{ .n_layers = 32, .kv_row_elems = 1024 }, .kv_dtype = .f32, .model_ctx = 131_072 },
        .{ .shape = .{ .n_layers = 24, .kv_row_elems = 512, .recurrent = qwen_2b }, .kv_dtype = .f16, .model_ctx = 262_144 },
        .{ .shape = .{ .n_layers = 32, .kv_row_elems = 1024, .recurrent = qwen_9b }, .kv_dtype = .f16, .model_ctx = 262_144 },
    };
    // The two anchors: the 2B at its ceiling adds 20201472 bytes of recurrent
    // state to 3 GiB of regions. The 8B at its ceiling is 16 GiB exactly.
    const anchor_2b = kv_budget.costModel(shapes[3].shape, .f16);
    try testing.expectEqual(@as(u64, 12_288), anchor_2b.kv_bytes_per_token);
    try testing.expectEqual(@as(u64, 20_201_472), anchor_2b.fixed_state_bytes_per_slot);
    var prng = std.Random.DefaultPrng.init(0xC0_5E_4F_A7);
    const r = prng.random();
    for (shapes, 0..) |s, i| {
        const cost = kv_budget.costModel(s.shape, s.kv_dtype);
        const n_kv_layers = if (s.shape.recurrent) |rec|
            s.shape.n_layers / rec.full_attention_interval
        else
            s.shape.n_layers;

        const caps: kv_budget.Caps = .{
            .vram_free = 0,
            .weights = 0,
            .logits = 0,
            .model_ctx = s.model_ctx,
            .serve_default = 512,
            .default_ceil = 8192,
        };
        var trial: u32 = 0;
        while (trial < 64) : (trial += 1) {
            const req: kv_budget.Request = switch (trial) {
                // The ceiling anchor for every shape.
                0 => .{ .window = s.model_ctx },
                else => if (r.boolean())
                    .{
                        .window = r.intRangeAtMost(u32, 1, s.model_ctx),
                        .slots = r.intRangeAtMost(u32, 1, 16),
                    }
                else
                    .{ .x_bytes = r.intRangeAtMost(u64, 1 << 20, 64 << 30), .window = r.intRangeAtMost(u32, 1, s.model_ctx) },
            };

            const plan = kv_budget.kvPlan(caps, cost, req) catch |err| {
                try testing.expectEqual(error.KvVramBudgetTooSmall, err);
                continue;
            };
            const layout = Layout.init(
                plan.n_slots,
                plan.window,
                s.shape.kv_row_elems,
                n_kv_layers,
                s.kv_dtype,
            );
            // The per-token cost is one K row and one V row per KV layer.
            try testing.expectEqual(cost.kv_bytes_per_token, layout.tokenBytes());
            try testing.expectEqual(cost.kv_bytes_per_token * plan.window, layout.slotBytes());
            try testing.expectEqual(
                plan.vram.kv,
                layout.totalBytes() + plan.n_slots * cost.fixed_state_bytes_per_slot,
            );
            if (trial == 0 and i == 3) {
                try testing.expectEqual(@as(u64, 3_241_426_944), plan.vram.kv);
            }

            if (trial == 0 and i == 1) {
                try testing.expectEqual(@as(u64, 17_179_869_184), plan.vram.kv);
            }
        }
    }
}

test "blobBytes: the walk over layers, K then V, sums to the blob" {
    const layouts = [_]Layout{
        Layout.init(1, 2048, 192, 30, .f32),
        Layout.init(4, 512, 1024, 32, .f16),
        Layout.init(3, 1000, 512, 6, .f16),
    };
    for (layouts) |layout| {
        for ([_]u32{ 0, 1, layout.window / 2, layout.window }) |n_rows| {
            // One blob region per device region, in the evict order: layers
            // ascending, K before V.
            var walked: u64 = 0;
            for (0..layout.n_layers) |_| {
                for ([_]Tensor{ .k, .v }) |_| {
                    walked += @as(u64, n_rows) * layout.rowBytes();
                }
            }

            try testing.expectEqual(walked, layout.blobBytes(n_rows));
        }

        try testing.expectEqual(layout.slotBytes(), layout.blobBytes(layout.window));
    }
}

fn checkWindowForLimits(layout_checks: *u32, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const kv_dtype: KvDtype = if (r.boolean()) .f16 else .f32;
    const n_slots = r.intRangeAtMost(u32, 1, limits.max_vram_slots);
    // Whole 16-byte rows, so the alignment never decides a verdict.
    const per_step = 8 * r.intRangeAtMost(u32, 1, 512);
    const lim: limits.BufferLimits = .{
        .storage_range = r.intRangeAtMost(u64, 16, limits.storage_range_max),
        .allocation_size = r.intRangeAtMost(u64, 16, 1 << 36),
        .offset_alignment = 16,
    };
    const window = maxWindowForLimits(n_slots, per_step, kv_dtype, lim);
    const row_bytes = kv_dtype.rowBytes(per_step);
    // The arithmetic form holds at every window, with no model ceiling.
    const per_position = @as(u64, n_slots) * per_step;
    const here = limits.bufferFits(window * per_position, window * n_slots * row_bytes, lim);
    try testing.expectEqual(limits.BufferVerdict.fits, here);
    const next = @as(u64, window) + 1;
    const past = limits.bufferFits(next * per_position, next * n_slots * row_bytes, lim);
    try testing.expect(
        past == .index_past_31_bits or past == .past_allocation_size or past == .past_storage_range,
    );
    // The layout form, where the window is a legal window.
    if (window >= 1 and window < limits.max_model_ctx) {
        try testing.expectEqual(
            limits.BufferVerdict.fits,
            fits(Layout.init(n_slots, window, per_step, 1, kv_dtype), lim),
        );
        const verdict = fits(Layout.init(n_slots, window + 1, per_step, 1, kv_dtype), lim);
        try testing.expect(
            verdict == .index_past_31_bits or verdict == .past_allocation_size or verdict ==
                .past_storage_range,
        );
        layout_checks.* += 1;
    }
}

test "maxWindowForLimits: the window passes the limits and one more position breaks one" {
    var layout_checks: u32 = 0;
    try prop.forAll(
        .{ .site = 0x3A1D_0B17, .cases = 4000 },
        &layout_checks,
        u64,
        prop.genSeed(*u32),
        checkWindowForLimits,
    );
    // The draw reaches the layout form often, not only the arithmetic form.
    try testing.expect(layout_checks >= 1000);
}

test "maxSlotsForLimits: the registry shapes under the three limit instances" {
    // Qwen3.5-2B at f16: the serve default of 10752 positions takes 10.5 MiB
    // a slot, so the 128 MiB range holds 12 slots and not 13.
    try testing.expectEqual(@as(u32, 12), maxSlotsForLimits(10_752, 512, .f16, spec_min_limits));
    try testing.expectEqual(
        limits.BufferVerdict.fits,
        fits(Layout.init(12, 10_752, 512, 6, .f16), spec_min_limits),
    );
    try testing.expectEqual(
        limits.BufferVerdict.past_storage_range,
        fits(Layout.init(13, 10_752, 512, 6, .f16), spec_min_limits),
    );
    // One slot at 131072 fills the range exactly, and one at the 262144
    // ceiling breaks it, so no slot fits there.
    try testing.expectEqual(@as(u32, 1), maxSlotsForLimits(131_072, 512, .f16, spec_min_limits));
    try testing.expectEqual(@as(u32, 0), maxSlotsForLimits(262_144, 512, .f16, spec_min_limits));
    // The 512 MiB override holds two such slots. The full 32-bit range holds
    // 15, because a sixteenth passes the allocation size by 4 bytes.
    try testing.expectEqual(
        @as(u32, 2),
        maxSlotsForLimits(262_144, 512, .f16, raised_range_limits),
    );
    try testing.expectEqual(@as(u32, 15), maxSlotsForLimits(262_144, 512, .f16, full_range_limits));
    // Llama-3.1-8B at the f16 serve default of 1024 positions: 2 MiB a slot.
    try testing.expectEqual(@as(u32, 64), maxSlotsForLimits(1024, 1024, .f16, spec_min_limits));
    // With the byte limits wide, the 31-bit index binds.
    try testing.expectEqual(@as(u32, 16), maxSlotsForLimits(131_072, 1024, .f32, wide_limits));
}

fn checkSlotsMatchWindow(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const kv_dtype: KvDtype = if (r.boolean()) .f16 else .f32;
    const per_step = 8 * r.intRangeAtMost(u32, 1, 512);
    const lim: limits.BufferLimits = .{
        .storage_range = r.intRangeAtMost(u64, 16, limits.storage_range_max),
        .allocation_size = r.intRangeAtMost(u64, 16, 1 << 36),
        .offset_alignment = 16,
    };
    const n_slots = r.intRangeAtMost(u32, 1, limits.max_vram_slots);
    const window = r.intRangeAtMost(u32, 1, limits.max_model_ctx);
    // Both bounds solve `slots * window * per_step` against the same
    // limits, so each one admits the pair exactly when the other does.
    const by_slots = n_slots <= maxSlotsForLimits(window, per_step, kv_dtype, lim);
    const by_window = window <= maxWindowForLimits(n_slots, per_step, kv_dtype, lim);
    try testing.expectEqual(by_window, by_slots);
    if (by_slots) {
        try testing.expectEqual(
            limits.BufferVerdict.fits,
            fits(Layout.init(n_slots, window, per_step, 1, kv_dtype), lim),
        );
    }
}

test "maxSlotsForLimits: a slot count fits exactly when the window of that count does" {
    try prop.forAll(
        .{ .site = 0x51_07_D0_A1, .cases = 4000 },
        {},
        u64,
        prop.genSeed(void),
        checkSlotsMatchWindow,
    );
}

test "maxGranuleWindow: the registry shapes under the spec-minimum storage range" {
    const lim = spec_min_limits;
    // SmolLM2-135M at f32: 3 KV heads x 64, 768 bytes a row. The range holds
    // 174762 rows of one slot, so the granule floor is 174592.
    try testing.expectEqual(@as(u32, 174_592), maxGranuleWindow(1, 192, .f32, lim));
    try testing.expectEqual(@as(u32, 43_520), maxGranuleWindow(4, 192, .f32, lim));
    // Qwen3.5-2B at f16: 2 KV heads x 256, 1024 bytes a row.
    try testing.expectEqual(@as(u32, 131_072), maxGranuleWindow(1, 512, .f16, lim));
    try testing.expectEqual(@as(u32, 65_536), maxGranuleWindow(2, 512, .f16, lim));
    try testing.expectEqual(@as(u32, 32_768), maxGranuleWindow(4, 512, .f16, lim));
    // Llama-3.1-8B: 8 KV heads x 128, at f16 and at f32.
    try testing.expectEqual(@as(u32, 65_536), maxGranuleWindow(1, 1024, .f16, lim));
    try testing.expectEqual(@as(u32, 32_768), maxGranuleWindow(1, 1024, .f32, lim));
    // The slot ceiling of the 8B at f32 leaves one granule, and a 64 MiB range
    // leaves none.
    try testing.expectEqual(
        limits.window_granule,
        maxGranuleWindow(limits.max_vram_slots, 1024, .f32, lim),
    );
    var narrow = lim;
    narrow.storage_range = 64 << 20;
    try testing.expectEqual(
        @as(u32, 0),
        maxGranuleWindow(limits.max_vram_slots, 1024, .f32, narrow),
    );
    // With the range and the allocation wide, the 31-bit index binds: 16 slots
    // of 1024 elements hold 2^31 elements at 131072 positions.
    try testing.expectEqual(@as(u32, 131_072), maxGranuleWindow(16, 1024, .f32, wide_limits));
}

test "limitOf: each broken limit names its words, its value, and its unit" {
    const lim = spec_min_limits;
    const index = limitOf(.index_past_31_bits, lim);
    try testing.expectEqualStrings("shader index limit", index.name);
    try testing.expectEqual(limits.max_kv_index_elems, index.value);
    try testing.expectEqual(Limit.Unit.elements, index.unit);
    const allocation = limitOf(.past_allocation_size, lim);
    try testing.expectEqualStrings("device allocation limit", allocation.name);
    try testing.expectEqual(lim.allocation_size, allocation.value);
    try testing.expectEqual(Limit.Unit.bytes, allocation.unit);
    const range = limitOf(.past_storage_range, lim);
    try testing.expectEqualStrings("storage-buffer range", range.name);
    try testing.expectEqual(lim.storage_range, range.value);
    try testing.expectEqual(Limit.Unit.bytes, range.unit);
    const alignment = limitOf(.misaligned, lim);
    try testing.expectEqualStrings("storage-buffer offset alignment", alignment.name);
    try testing.expectEqual(lim.offset_alignment, alignment.value);
    try testing.expectEqual(Limit.Unit.bytes, alignment.unit);
}
