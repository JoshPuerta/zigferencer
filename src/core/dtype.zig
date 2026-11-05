//! GGUF `ggml_type` enum plus per-type block geometry: how many bytes a tensor of N elements
//! occupies on disk and in memory.
//!
//! Kernel coverage. The kernels read F32 and Q8_0 directly and dequantize Q4_0 and Q6_K at load.
//! The enum lists the other types so the parser can recognize a mixed GGUF. `blockSize` and
//! `bytesPerBlock` panic on a type no kernel handles, because a loud failure beats a silently wrong
//! one. The k-quant sizes exist so the loader can map a mixed GGUF. Only q6_k has kernels: a CPU
//! dequant-and-dot linear for the Q6_K logit head of a Q4_0 file, and a load-time requantize to
//! q4_0 for the GPU path.
//!
//! Block geometry. It is the unit every kernel iterates over. A row of N weights is N/32
//! contiguous blocks. An f16 scale `d` leads each block:
//!
//!   Q8_0   32 weights -> 34 bytes          value[i] = d * q[i]
//!     +--------+-----------------------------------+
//!     | f16 d  | i8 q[0] q[1] ............... q[31] |
//!     +--------+-----------------------------------+
//!       2 B              32 B
//!
//!   Q4_0   32 weights -> 18 bytes          value[i]    = (lo(b_i) - 8) * d
//!     +--------+--------------------+       value[i+16] = (hi(b_i) - 8) * d
//!     | f16 d  | u4x2 b[0] .. b[15] |       byte b_i packs element i in its
//!     +--------+--------------------+       low nibble, i+16 in its high one
//!       2 B          16 B

const std = @import("std");

// -- constants --------------------------------------------------------------

/// Bytes the widest `tagSuffix` occupies. A tag buffer adds this to its frame, so the format cannot
/// fail.
pub const kv_tag_suffix_max: usize = blk: {
    var widest: usize = 0;
    for (std.enums.values(KvDtype)) |d| {
        widest = @max(widest, d.tagSuffix().len);
    }

    break :blk widest;
};

// -- types ------------------------------------------------------------------

/// Open enum mirroring ggml's wire type ids (unknown ids stay representable).
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    _,
};

/// Element type of the KV cache. One value per process, resolved once at flag
/// time, so every runner in the process frames its rows the same way. `q8_0` is
/// reserved and not admitted.
pub const KvDtype = enum {
    f32,
    f16,

    /// Bytes one KV row of `per_step_elems` elements occupies. Every KV size derives from this
    /// product.
    pub fn rowBytes(self: KvDtype, per_step_elems: u32) u64 {
        const t: GgmlType = switch (self) {
            .f32 => .f32,
            .f16 => .f16,
        };

        return bytesForNumel(t, per_step_elems);
    }

    /// The component a backend provenance tag folds in. Empty at f32, so shipped blob namespaces
    /// stay byte identical. Otherwise it names the dtype, so no blob restores across dtypes.
    pub fn tagSuffix(self: KvDtype) []const u8 {
        return switch (self) {
            .f32 => "",
            .f16 => ":kv=f16",
        };
    }

    /// The dtype token a plan line folds in. Empty at f32, so the f32 plan line stays byte
    /// identical.
    pub fn planToken(self: KvDtype) []const u8 {
        return switch (self) {
            .f32 => "",
            .f16 => " (f16)",
        };
    }
};

// -- api --------------------------------------------------------------------

/// True when `blockSize` and `bytesPerBlock` both name `t`. A GGUF can carry a
/// wire type id with no block geometry here, so a caller at the trust boundary
/// must ask this before it sizes a tensor.
pub fn hasBlockGeometry(t: GgmlType) bool {
    return switch (t) {
        .f32,
        .f16,
        .bf16,
        .f64,
        .i8,
        .i16,
        .i32,
        .i64,
        .q4_0,
        .q4_1,
        .q5_0,
        .q5_1,
        .q8_0,
        .q8_1,
        .q2_k,
        .q3_k,
        .q4_k,
        .q5_k,
        .q6_k,
        .q8_k,
        => true,
        else => false,
    };
}

/// Elements per quant block (1 for scalars, 32 for q*_0/_1, 256 for k-quants).
/// The caller must gate on `hasBlockGeometry` first.
pub fn blockSize(t: GgmlType) usize {
    return switch (t) {
        .f32, .f16, .bf16, .f64, .i8, .i16, .i32, .i64 => 1,
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
        else => @panic("blockSize: unsupported dtype"),
    };
}

/// Encoded bytes per quant block.
pub fn bytesPerBlock(t: GgmlType) usize {
    return switch (t) {
        .f32, .i32 => 4,
        .f16, .bf16, .i16 => 2,
        .f64, .i64 => 8,
        .i8 => 1,
        .q4_0 => 18,
        .q4_1 => 20,
        .q5_0 => 22,
        .q5_1 => 24,
        .q8_0 => 34,
        .q8_1 => 36,
        // k-quant superblocks of 256 elements. See "Kernel coverage" in the header.
        .q2_k => 84,
        .q3_k => 110,
        .q4_k => 144,
        .q5_k => 176,
        .q6_k => 210,
        .q8_k => 292,
        else => @panic("bytesPerBlock: unsupported dtype"),
    };
}

/// Encoded byte size of a `numel`-element tensor of type `t`. The caller
/// holds a size the engine already bounds. A size from a file goes through
/// `bytesForNumelChecked`.
pub fn bytesForNumel(t: GgmlType, numel: usize) usize {
    const bytes = bytesForNumelChecked(t, numel) orelse unreachable;
    return @intCast(bytes);
}

/// Encoded byte size of a `numel`-element tensor of type `t`, or null when
/// the size passes 64 bits. The caller must gate on `hasBlockGeometry` and
/// give a whole number of blocks.
pub fn bytesForNumelChecked(t: GgmlType, numel: u64) ?u64 {
    const bs = blockSize(t);
    std.debug.assert(numel % bs == 0);
    const blocks = numel / bs;
    const bytes = std.math.mul(u64, blocks, bytesPerBlock(t)) catch return null;
    std.debug.assert(bytes >= blocks);
    return bytes;
}

// -- tests ------------------------------------------------------------------

/// The literal block geometry of each wire tag. Null is a tag that the engine
/// cannot size. The values are the ggml type traits, typed out by hand so a
/// change to a geometry switch fails here without a model file.
const geometry_rows = [_]struct { t: GgmlType, geometry: ?struct { block: usize, bytes: usize } }{
    .{ .t = .f32, .geometry = .{ .block = 1, .bytes = 4 } },
    .{ .t = .f16, .geometry = .{ .block = 1, .bytes = 2 } },
    .{ .t = .q4_0, .geometry = .{ .block = 32, .bytes = 18 } },
    .{ .t = .q4_1, .geometry = .{ .block = 32, .bytes = 20 } },
    .{ .t = .q5_0, .geometry = .{ .block = 32, .bytes = 22 } },
    .{ .t = .q5_1, .geometry = .{ .block = 32, .bytes = 24 } },
    .{ .t = .q8_0, .geometry = .{ .block = 32, .bytes = 34 } },
    .{ .t = .q8_1, .geometry = .{ .block = 32, .bytes = 36 } },
    .{ .t = .q2_k, .geometry = .{ .block = 256, .bytes = 84 } },
    .{ .t = .q3_k, .geometry = .{ .block = 256, .bytes = 110 } },
    .{ .t = .q4_k, .geometry = .{ .block = 256, .bytes = 144 } },
    .{ .t = .q5_k, .geometry = .{ .block = 256, .bytes = 176 } },
    .{ .t = .q6_k, .geometry = .{ .block = 256, .bytes = 210 } },
    .{ .t = .q8_k, .geometry = .{ .block = 256, .bytes = 292 } },
    .{ .t = .iq2_xxs, .geometry = null },
    .{ .t = .iq2_xs, .geometry = null },
    .{ .t = .iq3_xxs, .geometry = null },
    .{ .t = .iq1_s, .geometry = null },
    .{ .t = .iq4_nl, .geometry = null },
    .{ .t = .iq3_s, .geometry = null },
    .{ .t = .iq2_s, .geometry = null },
    .{ .t = .iq4_xs, .geometry = null },
    .{ .t = .i8, .geometry = .{ .block = 1, .bytes = 1 } },
    .{ .t = .i16, .geometry = .{ .block = 1, .bytes = 2 } },
    .{ .t = .i32, .geometry = .{ .block = 1, .bytes = 4 } },
    .{ .t = .i64, .geometry = .{ .block = 1, .bytes = 8 } },
    .{ .t = .f64, .geometry = .{ .block = 1, .bytes = 8 } },
    .{ .t = .iq1_m, .geometry = null },
    .{ .t = .bf16, .geometry = .{ .block = 1, .bytes = 2 } },
    .{ .t = .tq1_0, .geometry = null },
    .{ .t = .tq2_0, .geometry = null },
    .{ .t = .mxfp4, .geometry = null },
};

test "geometry: each wire tag has its literal block and block bytes, or no geometry" {
    const tags = std.enums.values(GgmlType);
    try std.testing.expectEqual(tags.len, geometry_rows.len);
    for (tags, geometry_rows) |t, row| {
        try std.testing.expectEqual(t, row.t);
        const g = row.geometry orelse {
            try std.testing.expect(!hasBlockGeometry(t));
            continue;
        };
        try std.testing.expect(hasBlockGeometry(t));
        try std.testing.expectEqual(g.block, blockSize(t));
        try std.testing.expectEqual(g.bytes, bytesPerBlock(t));
    }

    // An id outside the enum stays representable, and it has no geometry.
    try std.testing.expect(!hasBlockGeometry(@enumFromInt(9999)));
}

test "geometry: the size products of a KV row and a tensor, and the checked edge" {
    try std.testing.expectEqual(@as(u64, 2048), KvDtype.f32.rowBytes(512));
    try std.testing.expectEqual(@as(u64, 1024), KvDtype.f16.rowBytes(512));
    try std.testing.expectEqual(@as(u64, 4), KvDtype.f32.rowBytes(1));
    const data: [68]u8 = @splat(0);
    const t: @import("tensor.zig").Tensor = .{
        .dtype = .q8_0,
        .n_dims = 2,
        .shape = .{ 32, 2, 0, 0 },
        .byte_stride = .{ 34, 0, 0, 0 },
        .data = &data,
    };
    try std.testing.expectEqual(@as(usize, 68), t.bytesRequired());
    // The checked form refuses the first size past 64 bits, and only that.
    const max_f32: u64 = std.math.maxInt(u64) / 4;
    try std.testing.expectEqual(@as(?u64, max_f32 * 4), bytesForNumelChecked(.f32, max_f32));
    try std.testing.expectEqual(@as(?u64, null), bytesForNumelChecked(.f32, max_f32 + 1));
    try std.testing.expectEqual(
        @as(?u64, null),
        bytesForNumelChecked(.q8_0, (std.math.maxInt(u64) / 32) * 32),
    );
}

test "KvDtype: the plan token names only a non-identity dtype" {
    // The empty token is what keeps every f32 plan line byte identical.
    try std.testing.expectEqualStrings("", KvDtype.f32.planToken());
    try std.testing.expectEqualStrings(" (f16)", KvDtype.f16.planToken());
}
