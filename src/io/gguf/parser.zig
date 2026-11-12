//! Role. GGUF v3 binary parser. The file is a datapath: a byte slice goes in and a
//! structured `Parsed` value comes out, so a plain buffer tests it. The parser checks
//! the "GGUF" identifier, the version, the alignment, and the alignment of each tensor offset. It
//! rejects truncated or malformed input with a typed `Error`. The loader one level up joins the
//! map, this parser, and the architecture-specific semantic checks.
//!
//! Comptime. The integer reader takes its type at comptime, so one body reads every width.
//!
//! File layout (v3).
//!     magic                "GGUF" (4 bytes)
//!     version              u32 LE  (asserted == 3)
//!     tensor_count         u64 LE
//!     metadata_kv_count    u64 LE
//!     metadata_kv_pairs    repeated { key_len, key, value_type, value }
//!     tensor_info          repeated { name_len, name, n_dims, dims..., ggml_type, offset }
//!     pad to general.alignment (default 32)
//!     tensor_data          raw blob (offsets in TensorInfo are relative to here)

const std = @import("std");
const dtype = @import("../../core/dtype.zig");
const Allocator = std.mem.Allocator;
const prop = @import("../../core/utils/prop.zig");

const log = std.log.scoped(.gguf);

// -- constants --------------------------------------------------------------

pub const magic: [4]u8 = .{ 'G', 'G', 'U', 'F' };
const default_alignment: u64 = 32;
const max_tensor_name_len: usize = 64;
const supported_version: u32 = 3;

// -- types ------------------------------------------------------------------

pub const Error = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    BadValueType,
    NestedArray,
    NameTooLong,
    BadAlignment,
    BadTensorDims,
    /// A header count (metadata, tensor, or array length) is larger than the remaining bytes can
    /// hold.
    CountExceedsFile,
    /// Tensor offsets do not rise monotonically. The loader needs that order.
    TensorOffsetsUnordered,
    /// A tensor offset is not a multiple of `general.alignment`. The loader views data in place.
    TensorOffsetUnaligned,
} || Allocator.Error;

const ValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
    _,
};

pub const Value = union(enum) {
    u8_val: u8,
    i8_val: i8,
    u16_val: u16,
    i16_val: i16,
    u32_val: u32,
    i32_val: i32,
    f32_val: f32,
    bool_val: bool,
    string: []const u8,
    array: Array,
    u64_val: u64,
    i64_val: i64,
    f64_val: f64,
};

pub const Array = union(enum) {
    u8s: []const u8,
    i8s: []const i8,
    u16s: []const u16,
    i16s: []const i16,
    u32s: []const u32,
    i32s: []const i32,
    f32s: []const f32,
    bools: []const bool,
    strings: []const []const u8,
    u64s: []const u64,
    i64s: []const i64,
    f64s: []const f64,
};

/// One `gguf_metadata_kv_t` of the spec: a key and its typed value.
pub const MetadataPair = struct {
    key: []const u8,
    value: Value,
};

pub const TensorInfo = struct {
    name: []const u8,
    dtype: dtype.GgmlType,
    n_dims: u8,
    shape: [4]u64,
    /// Byte offset within the *tensor data blob*, not the file. Add
    /// `Parsed.data_offset` to get the absolute offset.
    offset_in_data: u64,
};

const Header = struct {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

pub const Parsed = struct {
    header: Header,
    metadata: []const MetadataPair,
    tensors: []const TensorInfo,
    /// Absolute byte offset where the tensor data blob starts. It is a
    /// multiple of `alignment`.
    data_offset: u64,
    /// A power of two. Every `TensorInfo.offset_in_data` is a multiple of it.
    alignment: u64,
};

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn ensure(r: *const Reader, n: usize) Error!void {
        std.debug.assert(r.pos <= r.bytes.len);

        if (n > r.bytes.len - r.pos) return error.Truncated;
    }

    fn remaining(r: *const Reader) usize {
        return r.bytes.len - r.pos;
    }

    /// Reject a header count before the parser allocates for it. Each element takes at least
    /// `min_each` bytes, so the allocation never exceeds the file size.
    fn boundCount(r: *const Reader, count: u64, min_each: usize) Error!void {
        std.debug.assert(min_each >= 1);

        if (count > r.remaining() / min_each) return error.CountExceedsFile;
    }

    fn skip(r: *Reader, n: usize) Error!void {
        try r.ensure(n);
        r.pos += n;
    }

    fn read(r: *Reader, comptime T: type) Error!T {
        const sz = @sizeOf(T);
        try r.ensure(sz);
        const slice = r.bytes[r.pos..][0..sz];
        r.pos += sz;
        return switch (@typeInfo(T)) {
            .int => std.mem.readInt(T, slice, .little),
            .float => |fi| blk: {
                const IntT = std.meta.Int(.unsigned, fi.bits);
                const bits = std.mem.readInt(IntT, slice, .little);
                break :blk @bitCast(bits);
            },
            else => @compileError("Reader.read: only int/float"),
        };
    }

    fn readBool(r: *Reader) Error!bool {
        try r.ensure(1);
        const v = r.bytes[r.pos] != 0;
        r.pos += 1;
        return v;
    }

    fn readString(r: *Reader) Error![]const u8 {
        const n_u64 = try r.read(u64);
        const n = std.math.cast(usize, n_u64) orelse return error.Truncated;
        try r.ensure(n);
        const s = r.bytes[r.pos..][0..n];
        r.pos += n;
        return s;
    }
};

// -- api --------------------------------------------------------------------

/// Parse a GGUF v3 byte buffer into a `Parsed` value. Every read checks its
/// bounds first.
pub fn parse(arena: Allocator, bytes: []const u8) Error!Parsed {
    var r: Reader = .{ .bytes = bytes };

    // Read the fixed header.
    try r.ensure(4 + 4 + 8 + 8);
    if (!std.mem.eql(u8, bytes[0..4], &magic)) return error.BadMagic;

    r.pos = 4;

    const version = try r.read(u32);
    if (version != supported_version) return error.UnsupportedVersion;

    const tensor_count = try r.read(u64);
    const pair_count = try r.read(u64);
    log.debug(
        "header: v{d}, {d} tensors, {d} metadata pairs",
        .{ version, tensor_count, pair_count },
    );

    // Read the metadata pairs. The floor per pair: 8 (key length) + 4 (value type) + 1 (value).
    try r.boundCount(pair_count, 13);
    const metadata = try arena.alloc(MetadataPair, @intCast(pair_count));
    for (metadata) |*pair| {
        pair.key = try r.readString();
        pair.value = try readValue(&r, arena);
    }

    // Read the tensor table.
    try r.boundCount(
        tensor_count,
        32,
    ); // floor per tensor: name len + n_dims + 1 dim + dtype + offset
    const tensors = try arena.alloc(TensorInfo, @intCast(tensor_count));
    var last_offset: u64 = 0;
    for (tensors) |*ti| {
        const name = try r.readString();
        if (name.len > max_tensor_name_len) return error.NameTooLong;

        ti.name = name;
        const n_dims_u32 = try r.read(u32);
        if (n_dims_u32 == 0 or n_dims_u32 > 4) return error.BadTensorDims;

        ti.n_dims = @intCast(n_dims_u32);
        ti.shape = .{ 1, 1, 1, 1 };
        for (0..ti.n_dims) |i| {
            ti.shape[i] = try r.read(u64);
        }

        ti.dtype = @enumFromInt(try r.read(u32));
        ti.offset_in_data = try r.read(u64);
        if (ti.offset_in_data < last_offset) return error.TensorOffsetsUnordered;

        last_offset = ti.offset_in_data;
    }

    // Find the alignment and check each tensor offset against it.
    var alignment: u64 = default_alignment;
    for (metadata) |pair| {
        if (std.mem.eql(u8, pair.key, "general.alignment")) {
            alignment = switch (pair.value) {
                .u32_val => |u| u,
                .u64_val => |u| u,
                else => return error.BadAlignment,
            };
            break;
        }
    }

    // A power of two keeps every aligned offset aligned for each smaller power of two.
    if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.BadAlignment;

    for (tensors) |ti| {
        if (ti.offset_in_data % alignment != 0) {
            log.debug(
                "tensor {s} starts at data offset {d}, " ++
                    "which is not a multiple of the alignment {d}",
                .{ ti.name, ti.offset_in_data, alignment },
            );
            return error.TensorOffsetUnaligned;
        }
    }

    // Skip the padding to the start of the tensor data.
    const pad = (alignment - (r.pos % alignment)) % alignment;
    try r.skip(std.math.cast(usize, pad) orelse return error.Truncated);
    log.debug(
        "parsed: metadata done, tensor table done, data at {d} (align {d})",
        .{ r.pos, alignment },
    );
    std.debug.assert(r.pos % alignment == 0);

    return .{
        .header = .{
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = pair_count,
        },
        .metadata = metadata,
        .tensors = tensors,
        .data_offset = r.pos,
        .alignment = alignment,
    };
}

// -- helpers ----------------------------------------------------------------

/// Smallest number of file bytes that one array element of `t` can consume.
/// A string floor is its 8-byte length. A nested array fails later, so 1 is a loose floor.
fn elemMinBytes(t: ValueType) usize {
    return switch (t) {
        .u8, .i8, .bool => 1,
        .u16, .i16 => 2,
        .u32, .i32, .f32 => 4,
        .u64, .i64, .f64, .string => 8,
        .array, _ => 1,
    };
}

fn readArray(r: *Reader, arena: Allocator) Error!Array {
    // Read the element type and the bounded count.
    const elem_type_raw = try r.read(u32);
    const elem_type: ValueType = @enumFromInt(elem_type_raw);
    const n_u64 = try r.read(u64);
    try r.boundCount(n_u64, elemMinBytes(elem_type));
    const n: usize = @intCast(n_u64);

    // Read the elements of the array.
    return switch (elem_type) {
        .u8 => blk: {
            const buf = try arena.alloc(u8, n);
            for (buf) |*x| {
                x.* = try r.read(u8);
            }

            break :blk .{ .u8s = buf };
        },
        .i8 => blk: {
            const buf = try arena.alloc(i8, n);
            for (buf) |*x| {
                x.* = try r.read(i8);
            }

            break :blk .{ .i8s = buf };
        },
        .u16 => blk: {
            const buf = try arena.alloc(u16, n);
            for (buf) |*x| {
                x.* = try r.read(u16);
            }

            break :blk .{ .u16s = buf };
        },
        .i16 => blk: {
            const buf = try arena.alloc(i16, n);
            for (buf) |*x| {
                x.* = try r.read(i16);
            }

            break :blk .{ .i16s = buf };
        },
        .u32 => blk: {
            const buf = try arena.alloc(u32, n);
            for (buf) |*x| {
                x.* = try r.read(u32);
            }

            break :blk .{ .u32s = buf };
        },
        .i32 => blk: {
            const buf = try arena.alloc(i32, n);
            for (buf) |*x| {
                x.* = try r.read(i32);
            }

            break :blk .{ .i32s = buf };
        },
        .f32 => blk: {
            const buf = try arena.alloc(f32, n);
            for (buf) |*x| {
                x.* = try r.read(f32);
            }

            break :blk .{ .f32s = buf };
        },
        .bool => blk: {
            const buf = try arena.alloc(bool, n);
            for (buf) |*x| {
                x.* = try r.readBool();
            }

            break :blk .{ .bools = buf };
        },
        .string => blk: {
            const buf = try arena.alloc([]const u8, n);
            for (buf) |*x| {
                x.* = try r.readString();
            }

            break :blk .{ .strings = buf };
        },
        .u64 => blk: {
            const buf = try arena.alloc(u64, n);
            for (buf) |*x| {
                x.* = try r.read(u64);
            }

            break :blk .{ .u64s = buf };
        },
        .i64 => blk: {
            const buf = try arena.alloc(i64, n);
            for (buf) |*x| {
                x.* = try r.read(i64);
            }

            break :blk .{ .i64s = buf };
        },
        .f64 => blk: {
            const buf = try arena.alloc(f64, n);
            for (buf) |*x| {
                x.* = try r.read(f64);
            }

            break :blk .{ .f64s = buf };
        },
        .array => return error.NestedArray,
        _ => return error.BadValueType,
    };
}

fn readValue(r: *Reader, arena: Allocator) Error!Value {
    const vt_raw = try r.read(u32);
    const vt: ValueType = @enumFromInt(vt_raw);
    return switch (vt) {
        .u8 => .{ .u8_val = try r.read(u8) },
        .i8 => .{ .i8_val = try r.read(i8) },
        .u16 => .{ .u16_val = try r.read(u16) },
        .i16 => .{ .i16_val = try r.read(i16) },
        .u32 => .{ .u32_val = try r.read(u32) },
        .i32 => .{ .i32_val = try r.read(i32) },
        .f32 => .{ .f32_val = try r.read(f32) },
        .bool => .{ .bool_val = try r.readBool() },
        .string => .{ .string = try r.readString() },
        .array => .{ .array = try readArray(r, arena) },
        .u64 => .{ .u64_val = try r.read(u64) },
        .i64 => .{ .i64_val = try r.read(i64) },
        .f64 => .{ .f64_val = try r.read(f64) },
        _ => return error.BadValueType,
    };
}

// -- tests ------------------------------------------------------------------

fn tInt(al: Allocator, list: *std.ArrayList(u8), comptime T: type, v: T) !void {
    var b: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &b, v, .little);
    try list.appendSlice(al, &b);
}

test "magic bytes" {
    try std.testing.expectEqualSlices(u8, "GGUF", &magic);
}

test "hardening: a header count larger than the file is rejected before allocating" {
    // tensor_count=0, metadata_kv_count=2^40, no body: the count cannot fit in 0 trailing
    // bytes, so `parse` rejects it early instead of trying to alloc ~2^40 pairs.
    var buf: [24]u8 = undefined;
    @memcpy(buf[0..4], "GGUF");
    std.mem.writeInt(u32, buf[4..8], supported_version, .little);
    std.mem.writeInt(u64, buf[8..16], 0, .little);
    std.mem.writeInt(u64, buf[16..24], @as(u64, 1) << 40, .little);
    try std.testing.expectError(error.CountExceedsFile, parse(std.testing.allocator, &buf));
}

test "hardening: unordered tensor offsets are a typed error, not a panic" {
    const al = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(al);
    defer arena.deinit();

    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(al);

    try list.appendSlice(al, "GGUF");
    try tInt(al, &list, u32, supported_version);
    try tInt(al, &list, u64, 2); // tensor_count
    try tInt(al, &list, u64, 0); // metadata_kv_count
    inline for (.{ .{ "a", 100 }, .{ "b", 50 } }) |t| { // second offset < first
        try tInt(al, &list, u64, @as(u64, t[0].len));
        try list.appendSlice(al, t[0]);
        try tInt(al, &list, u32, 1); // n_dims
        try tInt(al, &list, u64, 1); // dim0
        try tInt(al, &list, u32, 0); // dtype (ggml F32)
        try tInt(al, &list, u64, @as(u64, t[1])); // offset
    }

    try std.testing.expectError(error.TensorOffsetsUnordered, parse(arena.allocator(), list.items));
}

/// A small valid GGUF: three metadata pairs (the alignment, a string, a string array),
/// two tensors, and a data blob. The `table_end` bytes before the padding are
/// the header, the metadata section, and the tensor table.
pub const Sample = struct {
    /// The loader views f32 tensor data in place, so the buffer keeps the
    /// file alignment that a page-aligned map gives.
    bytes: [sample_capacity]u8 align(16),
    len: usize,
    table_end: usize,
};

pub const sample_capacity = 512;

pub fn sampleGguf() !Sample {
    var s: Sample = .{ .bytes = undefined, .len = 0, .table_end = 0 };
    var fba = std.heap.FixedBufferAllocator.init(&s.bytes);
    const al = fba.allocator();
    var list = try std.ArrayList(u8).initCapacity(al, sample_capacity);
    list.appendSliceAssumeCapacity("GGUF");
    try tInt(al, &list, u32, supported_version);
    try tInt(al, &list, u64, 2); // tensor_count
    try tInt(al, &list, u64, 3); // metadata_kv_count
    const key_strings = [_][]const u8{ "general.alignment", "general.name", "tok.list" };
    try tInt(al, &list, u64, key_strings[0].len);
    list.appendSliceAssumeCapacity(key_strings[0]);
    try tInt(al, &list, u32, @intFromEnum(ValueType.u32));
    try tInt(al, &list, u32, 16);
    try tInt(al, &list, u64, key_strings[1].len);
    list.appendSliceAssumeCapacity(key_strings[1]);
    try tInt(al, &list, u32, @intFromEnum(ValueType.string));
    try tInt(al, &list, u64, 3);
    list.appendSliceAssumeCapacity("abc");
    try tInt(al, &list, u64, key_strings[2].len);
    list.appendSliceAssumeCapacity(key_strings[2]);
    try tInt(al, &list, u32, @intFromEnum(ValueType.array));
    try tInt(al, &list, u32, @intFromEnum(ValueType.string));
    try tInt(al, &list, u64, 2);
    inline for (.{ "a", "bc" }) |item| {
        try tInt(al, &list, u64, item.len);
        list.appendSliceAssumeCapacity(item);
    }

    inline for (.{ .{ "t0", 0 }, .{ "t1", 16 } }) |t| {
        try tInt(al, &list, u64, t[0].len);
        list.appendSliceAssumeCapacity(t[0]);
        try tInt(al, &list, u32, 2); // n_dims
        try tInt(al, &list, u64, 2);
        try tInt(al, &list, u64, 2);
        try tInt(al, &list, u32, @intFromEnum(dtype.GgmlType.f32));
        try tInt(al, &list, u64, t[1]);
    }

    const table_end = list.items.len;
    list.appendNTimesAssumeCapacity(0, (16 - table_end % 16) % 16);
    list.appendNTimesAssumeCapacity(0x5A, 32);
    // The list grew inside `s.bytes` from its start, so the bytes are in place.
    std.debug.assert(@intFromPtr(list.items.ptr) == @intFromPtr(&s.bytes));
    s.len = list.items.len;
    s.table_end = table_end;
    return s;
}

fn inside(buf: []const u8, slice: []const u8) bool {
    if (slice.len == 0) return true;

    const lo = @intFromPtr(buf.ptr);
    const at = @intFromPtr(slice.ptr);
    return at >= lo and at + slice.len <= lo + buf.len;
}

/// `parse` returns a typed error, or a result whose every borrowed slice and
/// whose data offset lie inside the buffer. The parser bounds no tensor data
/// range, because the loader checks each range against the file.
fn expectParseInside(bytes: []const u8) !void {
    var heap: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&heap);
    const p = parse(fba.allocator(), bytes) catch |err| {
        // The heap holds every count that the file size admits.
        try std.testing.expect(err != error.OutOfMemory);
        return;
    };
    try std.testing.expect(p.data_offset <= bytes.len);
    try std.testing.expect(std.math.isPowerOfTwo(p.alignment));
    try std.testing.expectEqual(@as(u64, 0), p.data_offset % p.alignment);
    for (p.metadata) |pair| {
        try std.testing.expect(inside(bytes, pair.key));
        switch (pair.value) {
            .string => |s| try std.testing.expect(inside(bytes, s)),
            .array => |a| if (a == .strings) for (a.strings) |s| try std.testing.expect(
                inside(bytes, s),
            ),
            else => {},
        }
    }

    var last: u64 = 0;
    for (p.tensors) |t| {
        try std.testing.expect(inside(bytes, t.name));
        try std.testing.expect(t.name.len <= max_tensor_name_len);
        try std.testing.expect(t.n_dims >= 1 and t.n_dims <= 4);
        try std.testing.expect(t.offset_in_data >= last);
        try std.testing.expectEqual(@as(u64, 0), t.offset_in_data % p.alignment);
        last = t.offset_in_data;
    }
}

test "exhaustive: every single-byte change of the file head parses in bounds or fails typed" {
    const sample = try sampleGguf();
    const original = sample.bytes[0..sample.len];
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const p = try parse(arena.allocator(), original);
    try std.testing.expectEqual(@as(usize, 2), p.tensors.len);
    try std.testing.expectEqual(@as(u64, 16), p.alignment);
    try std.testing.expectEqual(sample.len - 32, p.data_offset);

    var bytes: [sample_capacity]u8 = undefined;
    for (0..sample.table_end) |at| {
        for (0..256) |value| {
            @memcpy(bytes[0..sample.len], original);
            bytes[at] = @intCast(value);
            try expectParseInside(bytes[0..sample.len]);
        }
    }
}

/// A random mutation: up to eight byte writes, one cut, and one 8-byte
/// field forced to an edge value.
fn checkGgufMutation(sample: *const Sample, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var bytes: [sample_capacity]u8 = undefined;
    @memcpy(bytes[0..sample.len], sample.bytes[0..sample.len]);
    for (0..1 + r.uintLessThan(usize, 8)) |_| {
        bytes[r.uintLessThan(usize, sample.len)] = r.int(u8);
    }

    if (r.boolean()) {
        const edges = [_]u64{
            0,
            1,
            0x7F,
            0xFF,
            1 << 32,
            std.math.maxInt(u32),
            std.math.maxInt(u63),
            std.math.maxInt(u64),
            std.math.maxInt(u64) - 7,
        };
        const at = r.uintAtMost(usize, sample.table_end - 8);
        std.mem.writeInt(u64, bytes[at..][0..8], edges[r.uintLessThan(usize, edges.len)], .little);
    }

    const len = if (r.boolean()) sample.len else r.uintAtMost(usize, sample.len);
    try expectParseInside(bytes[0..len]);
}

test "property: random multi-byte mutations and cuts parse inside the buffer or fail typed" {
    const sample = try sampleGguf();
    try prop.forAll(
        .{ .site = 0x6767_7566, .cases = 4000 },
        &sample,
        u64,
        prop.genSeed(*const Sample),
        checkGgufMutation,
    );
}

test "hardening: a key length of maxInt(u64) is a typed error, not an overflow" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var bytes: [37]u8 = @splat(0);
    @memcpy(bytes[0..4], "GGUF");
    std.mem.writeInt(u32, bytes[4..8], supported_version, .little);
    std.mem.writeInt(u64, bytes[24..32], std.math.maxInt(u64), .little);
    std.mem.writeInt(u64, bytes[16..24], 1, .little); // metadata_kv_count
    try std.testing.expectError(error.Truncated, parse(arena.allocator(), &bytes));
}

test "hardening: an alignment near maxInt(u64) is a typed error, not an overflow" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var list: std.ArrayList(u8) = .empty;
    const a = arena.allocator();
    try list.appendSlice(a, "GGUF");
    try tInt(a, &list, u32, supported_version);
    try tInt(a, &list, u64, 0);
    try tInt(a, &list, u64, 1);
    try tInt(a, &list, u64, "general.alignment".len);
    try list.appendSlice(a, "general.alignment");
    try tInt(a, &list, u32, @intFromEnum(ValueType.u64));
    for ([_]u64{ std.math.maxInt(u64), std.math.maxInt(u64) - 7, 1 << 63 }) |align_value| {
        list.items.len = 4 + 4 + 8 + 8 + 8 + "general.alignment".len + 4;
        try tInt(a, &list, u64, align_value);
        const want: Error = if (std.math.isPowerOfTwo(
            align_value,
        )) error.Truncated else error.BadAlignment;

        try std.testing.expectError(want, parse(a, list.items));
    }
}

/// The sample with its alignment value and the offset of tensor `t1`
/// rewritten. Both fields sit at fixed places in the sample layout.
fn sampleWith(alignment: u32, t1_offset: u64) !Sample {
    var s = try sampleGguf();
    const align_at = 4 + 4 + 8 + 8 + 8 + "general.alignment".len + 4;
    std.mem.writeInt(u32, s.bytes[align_at..][0..4], alignment, .little);
    std.mem.writeInt(u64, s.bytes[s.table_end - 8 ..][0..8], t1_offset, .little);
    return s;
}

test "alignment: the parse takes a power of two and offsets on it, and refuses the rest" {
    const cases = [_]struct { alignment: u32, t1_offset: u64, want: ?Error }{
        .{ .alignment = 16, .t1_offset = 16, .want = null },
        .{ .alignment = 1, .t1_offset = 16, .want = null },
        .{ .alignment = 4, .t1_offset = 20, .want = null },
        .{ .alignment = 0, .t1_offset = 16, .want = error.BadAlignment },
        .{ .alignment = 3, .t1_offset = 18, .want = error.BadAlignment },
        .{ .alignment = 48, .t1_offset = 48, .want = error.BadAlignment },
        .{ .alignment = 16, .t1_offset = 17, .want = error.TensorOffsetUnaligned },
        .{ .alignment = 16, .t1_offset = 24, .want = error.TensorOffsetUnaligned },
        .{ .alignment = 4, .t1_offset = 18, .want = error.TensorOffsetUnaligned },
    };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    for (cases) |c| {
        const s = try sampleWith(c.alignment, c.t1_offset);
        const got = parse(arena.allocator(), s.bytes[0..s.len]);
        if (c.want) |err| {
            try std.testing.expectError(err, got);
        } else {
            const p = try got;
            try std.testing.expectEqual(@as(u64, c.alignment), p.alignment);
            try std.testing.expectEqual(c.t1_offset, p.tensors[1].offset_in_data);
            try std.testing.expectEqual(@as(u64, 0), p.data_offset % c.alignment);
        }
    }
}
