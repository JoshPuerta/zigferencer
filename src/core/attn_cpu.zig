//! The geometry of CPU attention. The module is pure: no allocation, no threads, and no I/O.
//!
//! Parts. A row of `t_len` positions splits into tiles of `limits.attn_cpu_tile` positions. The
//! tiles group into `min(limits.attn_cpu_parts, tiles)` contiguous parts. Part `p` of `c` parts
//! owns the tiles `[p * n / c, (p + 1) * n / c)` of the `n` tiles. The part count never passes the
//! tile count, so every tile lands in one part and no part is empty. The geometry is a function of
//! `t_len` alone. So a row gets the same parts and the same bits at every thread count, window,
//! batch, and slot.
//!
//! Records. Each part writes one record per query head: the value sums, then the maximum score,
//! then the sum of the shifted exponentials. A merge adds the records in part order.
//!
//! Partial buffer. It holds one record per query head and part. A record offset is a mixed-radix
//! numeral with the radices `n_heads`, `attn_cpu_parts`, and the record width (Knuth, TAOCP vol 1,
//! section 2.2.6). So the records of one head lie next to each other in part order, and the merge
//! reads them as one run. The tiles are the split geometry of `attn_split.zig` at a split length of
//! `attn_cpu_tile`.

const std = @import("std");
const limits = @import("limits.zig");

const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

/// Positions one tile holds.
const tile: u32 = limits.attn_cpu_tile;

/// The part ceiling of one row.
const max_parts: u32 = limits.attn_cpu_parts;

/// Bytes one partial element occupies. The records are f32 at every KV dtype.
const elem_bytes: u64 = @sizeOf(f32);

comptime {
    // `tileCount` adds `tile - 1` to a row length inside 32 bits.
    assert(@as(u64, limits.max_model_ctx) + tile - 1 <= std.math.maxInt(u32));
}

// -- types ------------------------------------------------------------------

/// The positions `[t_lo, t_hi)` that one part of a row owns.
pub const Part = struct {
    t_lo: u32,
    t_hi: u32,
};

// -- api --------------------------------------------------------------------

/// The parts a row of `t_len` positions splits into:
/// `min(attn_cpu_parts, tileCount(t_len))`.
pub fn partCount(t_len: u32) u32 {
    const tiles = tileCount(t_len);
    const count = @min(max_parts, tiles);
    assert(count >= 1);
    assert(count <= max_parts);
    // Every part holds at least one tile.
    assert(count <= tiles);
    return count;
}

/// The positions that part `p` of a row of `t_len` positions owns.
pub fn partRange(t_len: u32, p: u32) Part {
    const tiles = tileCount(t_len);
    const count = partCount(t_len);
    assert(p < count);
    const tile_lo = partStart(tiles, count, p);
    const tile_hi = partStart(tiles, count, p + 1);
    assert(tile_lo < tile_hi);
    const part: Part = .{
        .t_lo = tile_lo * tile,
        .t_hi = @min(tile_hi * tile, t_len),
    };
    assert(part.t_lo < part.t_hi);
    assert(part.t_hi <= t_len);
    // The first part starts the row, and the last part ends it.
    if (p == 0) assert(part.t_lo == 0);
    if (p == count - 1) assert(part.t_hi == t_len);

    return part;
}

/// Elements one record holds: the value sums, then the maximum score, then
/// the exponential sum.
pub fn recordElems(head_dim: u32) u32 {
    assert(head_dim >= 1);
    // The maximum and the sum fit beside the value sums inside 32 bits.
    assert(head_dim <= std.math.maxInt(u32) - 2);

    const elems = head_dim + 2;
    assert(elems > head_dim);
    return elems;
}

/// Elements between the records of two adjacent query heads in one part. A
/// part kernel walks the heads of one part at this stride.
pub fn recordStride(head_dim: u32) usize {
    const stride = @as(usize, max_parts) * recordElems(head_dim);
    assert(stride >= recordElems(head_dim));
    return stride;
}

/// The first element of the record of query head `h` and part `p`. The
/// records of one head lie next to each other in part order.
pub fn recordOffset(n_heads: u32, head_dim: u32, h: u32, p: u32) usize {
    assert(h < n_heads);
    assert(p < max_parts);

    const record = recordElems(head_dim);
    const offset = (@as(usize, h) * max_parts + p) * record;
    // The whole record lies inside the partial buffer.
    assert(offset + record <= partialLen(n_heads, head_dim));
    return offset;
}

/// Elements the partial buffer of one row holds: one record per query head
/// and part. The window does not change it.
pub fn partialLen(n_heads: u32, head_dim: u32) usize {
    assert(n_heads >= 1);

    const record = recordElems(head_dim);
    // The engine sets no bound on the head count or width, so check the byte total in 128 bits.
    const widest = @as(u128, n_heads) * max_parts * record * elem_bytes;
    assert(widest <= std.math.maxInt(u64));
    const len: usize = @intCast(@as(u64, n_heads) * max_parts * record);
    assert(len >= @as(usize, max_parts) * 3);
    return len;
}

// -- helpers ----------------------------------------------------------------

/// The tiles a row of `t_len` positions walks: `ceil(t_len / attn_cpu_tile)`.
fn tileCount(t_len: u32) u32 {
    assert(t_len >= 1);
    assert(t_len <= limits.max_model_ctx);

    const count = (t_len + tile - 1) / tile;
    assert(count >= 1);
    // The last tile starts inside the row, so no tile is empty.
    assert(@as(u64, count - 1) * tile < t_len);
    // The tiles cover the row.
    assert(@as(u64, count) * tile >= t_len);
    return count;
}

/// The first tile of part `p` of `count` parts over `tiles` tiles.
fn partStart(tiles: u32, count: u32, p: u32) u32 {
    assert(count >= 1);
    assert(count <= tiles);
    assert(p <= count);

    const start = @as(u64, p) * tiles / count;
    assert(start <= tiles);
    if (p == 0) assert(start == 0);
    if (p == count) assert(start == tiles);

    return @intCast(start);
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;
const T = limits.attn_cpu_tile;
const P = limits.attn_cpu_parts;

/// The geometry tests walk every row up to `dense_tiles` tiles. Past that they
/// walk each tile at a fixed stride and add both of its edges, so every tile
/// count up to `test_tiles` gets rows, and a Debug run stays short.
const dense_tiles: u32 = 3;
const test_tiles: u32 = 40;
const sparse_stride: u32 = T / 8 + 1;

comptime {
    // The walk reaches every tile count up to at least one past P + 1.
    assert(test_tiles >= P + 2);
    // The stride is shorter than a tile, so no tile count gets skipped.
    assert(sparse_stride < T);
}

/// The next row length of the geometry walk after `t_len`.
fn nextTestLen(t_len: u32) u32 {
    assert(t_len >= 1);

    if (t_len < dense_tiles * T or t_len % T == 0) return t_len + 1;

    const next_edge = (t_len + T - 1) / T * T;
    const next = @min(t_len + sparse_stride, next_edge);
    assert(next > t_len);
    return next;
}

test "tileCount and partCount: one tile up to T positions, one part per tile up to P tiles, then P parts" {
    try testing.expectEqual(@as(u32, 1), tileCount(1));
    try testing.expectEqual(@as(u32, 1), partCount(1));
    try testing.expectEqual(@as(u32, 1), tileCount(T));
    try testing.expectEqual(@as(u32, 1), partCount(T));
    try testing.expectEqual(@as(u32, 2), tileCount(T + 1));
    try testing.expectEqual(@as(u32, 2), partCount(T + 1));
    try testing.expectEqual(P, tileCount(P * T));
    try testing.expectEqual(P, partCount(P * T));
    try testing.expectEqual(P + 1, tileCount(P * T + 1));
    try testing.expectEqual(P, partCount(P * T + 1));
    try testing.expectEqual(limits.max_model_ctx / T, tileCount(limits.max_model_ctx));
    try testing.expectEqual(P, partCount(limits.max_model_ctx));
}

test "partRange: the parts of every walked row up to 40 tiles are non-empty, contiguous, and balanced, and they cover the row" {
    var tiles_seen = [_]bool{false} ** (test_tiles + 1);
    var t_len: u32 = 1;
    var prev_len: u32 = 0;
    while (t_len <= test_tiles * T) : (t_len = nextTestLen(t_len)) {
        const tiles = tileCount(t_len);
        const count = partCount(t_len);
        tiles_seen[tiles] = true;
        // A longer row never has fewer tiles or parts.
        if (prev_len >= 1) {
            try testing.expect(tileCount(prev_len) <= tiles);
            try testing.expect(partCount(prev_len) <= count);
        }

        prev_len = t_len;
        var next_lo: u32 = 0;
        var p: u32 = 0;
        while (p < count) : (p += 1) {
            const part = partRange(t_len, p);
            // Each part starts on a tile edge, where the part before it ends.
            try testing.expectEqual(next_lo, part.t_lo);
            try testing.expectEqual(@as(u32, 0), part.t_lo % T);
            try testing.expect(part.t_lo < part.t_hi);
            // A part holds floor(n / c) or ceil(n / c) tiles, so two parts
            // differ by one tile at most.
            const part_tiles = (part.t_hi - part.t_lo + T - 1) / T;
            try testing.expect(part_tiles >= tiles / count);
            try testing.expect(part_tiles <= (tiles + count - 1) / count);
            next_lo = part.t_hi;
        }

        try testing.expectEqual(t_len, next_lo);
    }

    // The walk ends on the last edge, and it reaches every tile count.
    try testing.expectEqual(test_tiles * T, prev_len);
    for (tiles_seen[1..]) |seen| {
        try testing.expect(seen);
    }
}

test "partRange: every position of a walked row lies in exactly one part" {
    const owners = try testing.allocator.alloc(u8, test_tiles * T);
    defer testing.allocator.free(owners);

    var t_len: u32 = 1;
    while (t_len <= test_tiles * T) : (t_len = nextTestLen(t_len)) {
        @memset(owners[0..t_len], 0);
        var p: u32 = 0;
        while (p < partCount(t_len)) : (p += 1) {
            const part = partRange(t_len, p);
            for (owners[part.t_lo..part.t_hi]) |*owner| {
                owner.* += 1;
            }
        }

        for (owners[0..t_len]) |owner| {
            try testing.expectEqual(@as(u8, 1), owner);
        }
    }
}

test "recordOffset: every head, part, and element of a small box gets its own element below partialLen" {
    var n_heads: u32 = 1;
    while (n_heads <= 3) : (n_heads += 1) {
        var head_dim: u32 = 1;
        while (head_dim <= 3) : (head_dim += 1) {
            const len = partialLen(n_heads, head_dim);
            const record = recordElems(head_dim);
            // The largest box holds 3 * P * 5 elements.
            var seen = [_]bool{false} ** (3 * P * 5);
            var count: usize = 0;
            for (0..n_heads) |h| {
                for (0..P) |p| {
                    const at = recordOffset(n_heads, head_dim, @intCast(h), @intCast(p));
                    // The same part of the next head lies one stride on.
                    if (h > 0) {
                        const above = recordOffset(n_heads, head_dim, @intCast(h - 1), @intCast(p));
                        try testing.expectEqual(above + recordStride(head_dim), at);
                    }

                    // The records of one head follow each other in part order.
                    if (p > 0) {
                        const before = recordOffset(
                            n_heads,
                            head_dim,
                            @intCast(h),
                            @intCast(p - 1),
                        );
                        try testing.expectEqual(before + record, at);
                    }

                    for (0..record) |e| {
                        try testing.expect(at + e < len);
                        try testing.expect(!seen[at + e]);
                        seen[at + e] = true;
                        count += 1;
                    }
                }
            }

            // The box and the buffer have one size, so the last record ends
            // the buffer.
            try testing.expectEqual(len, count);
            try testing.expectEqual(
                len,
                recordOffset(n_heads, head_dim, n_heads - 1, P - 1) + record,
            );
        }
    }
}

test "anchors: the tiles, the parts, and the records at T = 2048 and P = 16" {
    try testing.expectEqual(@as(u32, 2048), T);
    try testing.expectEqual(@as(u32, 16), P);
    const Anchor = struct { t_len: u32, tiles: u32, parts: u32, first: Part, last: Part };
    const anchors = [_]Anchor{
        .{ .t_len = 1, .tiles = 1, .parts = 1, .first = .{ .t_lo = 0, .t_hi = 1 }, .last = .{ .t_lo = 0, .t_hi = 1 } },
        .{ .t_len = 2048, .tiles = 1, .parts = 1, .first = .{ .t_lo = 0, .t_hi = 2048 }, .last = .{ .t_lo = 0, .t_hi = 2048 } },
        .{ .t_len = 2049, .tiles = 2, .parts = 2, .first = .{ .t_lo = 0, .t_hi = 2048 }, .last = .{ .t_lo = 2048, .t_hi = 2049 } },
        .{ .t_len = 4000, .tiles = 2, .parts = 2, .first = .{ .t_lo = 0, .t_hi = 2048 }, .last = .{ .t_lo = 2048, .t_hi = 4000 } },
        .{ .t_len = 4032, .tiles = 2, .parts = 2, .first = .{ .t_lo = 0, .t_hi = 2048 }, .last = .{ .t_lo = 2048, .t_hi = 4032 } },
        .{ .t_len = 8064, .tiles = 4, .parts = 4, .first = .{ .t_lo = 0, .t_hi = 2048 }, .last = .{ .t_lo = 6144, .t_hi = 8064 } },
        // Seventeen tiles over sixteen parts: the last part holds two tiles.
        .{ .t_len = 32_769, .tiles = 17, .parts = 16, .first = .{ .t_lo = 0, .t_hi = 2048 }, .last = .{ .t_lo = 30_720, .t_hi = 32_769 } },
        .{ .t_len = 262_144, .tiles = 128, .parts = 16, .first = .{ .t_lo = 0, .t_hi = 16_384 }, .last = .{ .t_lo = 245_760, .t_hi = 262_144 } },
    };
    for (anchors) |a| {
        try testing.expectEqual(a.tiles, tileCount(a.t_len));
        try testing.expectEqual(a.parts, partCount(a.t_len));
        try testing.expectEqual(a.first, partRange(a.t_len, 0));
        try testing.expectEqual(a.last, partRange(a.t_len, a.parts - 1));
    }

    // Qwen3.5-2B: 8 query heads of 256.
    try testing.expectEqual(@as(u32, 258), recordElems(256));
    try testing.expectEqual(@as(usize, 33_024), partialLen(8, 256));
    try testing.expectEqual(@as(u64, 132_096), partialLen(8, 256) * elem_bytes);
    try testing.expectEqual(@as(usize, 13_674), recordOffset(8, 256, 3, 5));
    // Llama-3.1-8B: 32 query heads of 128. The last record ends the buffer.
    try testing.expectEqual(@as(usize, 66_560), partialLen(32, 128));
    try testing.expectEqual(@as(u64, 266_240), partialLen(32, 128) * elem_bytes);
    try testing.expectEqual(partialLen(32, 128), recordOffset(32, 128, 31, 15) + recordElems(128));
}
