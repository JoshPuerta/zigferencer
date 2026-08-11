//! Static-allocation KV cache for positional attention. Both families store
//! their attention rows in it. One flat byte slab holds both halves (K then
//! V), each shaped `[n_layers][max_seq][n_kv_heads * head_dim]` and addressed
//! by slice math.
//! `init` sizes the slab once from (n_layers, n_kv_heads, head_dim, max_seq,
//! kv_dtype), so there is no per-step churn. Ops never see the struct. They get
//! plain typed K/V slices for the current decode step, because boundaries are
//! values.
//!
//! The slab is bytes tagged with the element type it was framed at. The `As`
//! accessors take that element type and hand back a typed view. The f32-named
//! accessors are the same views pinned to f32.
//!
//! Slab layout: one flat slab, K half then V half. Each (layer, position) cell
//! is one row of `n_kv_heads * head_dim` elements.
//!
//!     |<-------------- K half -------------->|<-------------- V half ------------>|
//!     +----------------+----------------+-...-+----------------+-...
//!     |    layer 0     |    layer 1     |     |    layer 0     |
//!     | p0 p1 .. pmax  | p0 p1 .. pmax  |     | p0 p1 .. pmax  |
//!     +----------------+----------------+     +----------------+
//!
//!   per_step_elems  = n_kv_heads * head_dim          (one K or V row)
//!   per_layer_elems = max_seq * per_step_elems
//!   kBase(L)        = L * per_layer_elems
//!   vBase(L)        = n_layers * per_layer_elems + L * per_layer_elems

const std = @import("std");
const Allocator = std.mem.Allocator;
const limits = @import("../core/limits.zig");
const dtype = @import("../core/dtype.zig");

const log = std.log.scoped(.kv);

// -- api --------------------------------------------------------------------

pub const KvCache = struct {
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    /// Storage: `[2][n_layers][max_seq][n_kv_heads][head_dim]` elements of
    /// `kv_dtype`, flattened to bytes. The 4-byte alignment covers every
    /// element type the tag can name, so a typed view never needs a fixup.
    storage: []align(4) u8,
    /// The element type the rows are framed at. A view at any other type is a
    /// framing bug, so the accessors assert against this tag.
    kv_dtype: dtype.KvDtype,
    /// Elements per (layer, seq_pos): n_kv_heads * head_dim.
    per_step_elems: u32,
    /// Elements per layer: max_seq * per_step_elems.
    per_layer_elems: u32,

    // -- init ---------------------------------------------------------------

    pub fn init(
        arena: Allocator,
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: u32,
        kv_dtype: dtype.KvDtype,
    ) Allocator.Error!KvCache {
        std.debug.assert(n_layers > 0);
        std.debug.assert(n_kv_heads > 0);
        std.debug.assert(head_dim > 0);
        std.debug.assert(max_seq > 0);
        std.debug.assert(max_seq <= limits.max_model_ctx);
        std.debug.assert(n_layers <= limits.max_layers);

        const per_step_elems = n_kv_heads * head_dim;
        const per_layer_wide: u64 = @as(u64, max_seq) * per_step_elems;
        std.debug.assert(per_layer_wide <= std.math.maxInt(u32));
        const per_layer_elems: u32 = @intCast(per_layer_wide);
        // Rows, not elements: the slab is a K row and a V row for every
        // (layer, position), and the dtype prices one row.
        const n_rows: u64 = 2 * @as(u64, n_layers) * max_seq;
        const total_bytes: usize = @intCast(n_rows * kv_dtype.rowBytes(per_step_elems));
        const storage = try arena.alignedAlloc(u8, .@"4", total_bytes);
        @memset(storage, 0);
        return .{
            .n_layers = n_layers,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_seq = max_seq,
            .storage = storage,
            .kv_dtype = kv_dtype,
            .per_step_elems = per_step_elems,
            .per_layer_elems = per_layer_elems,
        };
    }

    pub fn sizeBytes(self: *const KvCache) usize {
        return self.storage.len;
    }

    // -- typed views --------------------------------------------------------

    /// Slice of K-cache rows for `layer` covering positions `0..=pos`, viewed at
    /// `E`. Each row is `n_kv_heads * head_dim` elements. Returned length =
    /// `(pos + 1) * per_step_elems`.
    pub fn kSliceAs(self: *const KvCache, comptime E: type, layer: u32, pos: u32) []const E {
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(pos < self.max_seq);
        return self.viewConst(E, self.kBase(layer), (@as(usize, pos) + 1) * self.per_step_elems);
    }

    /// Slice of V-cache rows for `layer` covering positions `0..=pos`, viewed at `E`.
    pub fn vSliceAs(self: *const KvCache, comptime E: type, layer: u32, pos: u32) []const E {
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(pos < self.max_seq);
        return self.viewConst(E, self.vBase(layer), (@as(usize, pos) + 1) * self.per_step_elems);
    }

    /// Mutable K destination row for (layer, pos), viewed at `E`. Length =
    /// per_step_elems.
    pub fn kDstAs(self: *KvCache, comptime E: type, layer: u32, pos: u32) []E {
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(pos < self.max_seq);
        const off = self.kBase(layer) + @as(usize, pos) * self.per_step_elems;
        return self.viewMut(E, off, self.per_step_elems);
    }

    /// Mutable V destination row for (layer, pos), viewed at `E`.
    pub fn vDstAs(self: *KvCache, comptime E: type, layer: u32, pos: u32) []E {
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(pos < self.max_seq);
        const off = self.vBase(layer) + @as(usize, pos) * self.per_step_elems;
        return self.viewMut(E, off, self.per_step_elems);
    }

    /// Slice of K-cache rows for `layer` covering positions `0..=pos`.
    pub fn kSlice(self: *const KvCache, layer: u32, pos: u32) []const f32 {
        return self.kSliceAs(f32, layer, pos);
    }

    /// Slice of V-cache rows for `layer` covering positions `0..=pos`.
    pub fn vSlice(self: *const KvCache, layer: u32, pos: u32) []const f32 {
        return self.vSliceAs(f32, layer, pos);
    }

    /// Mutable destination row to write K for (layer, pos). Length = per_step_elems.
    pub fn kDst(self: *KvCache, layer: u32, pos: u32) []f32 {
        return self.kDstAs(f32, layer, pos);
    }

    /// Mutable destination row to write V for (layer, pos). Length = per_step_elems.
    pub fn vDst(self: *KvCache, layer: u32, pos: u32) []f32 {
        return self.vDstAs(f32, layer, pos);
    }

    // -- byte ranges --------------------------------------------------------

    /// Byte range of positions `0..n_rows` of one layer's K half. The transport
    /// family reads bytes because a blob is storage bytes. Evict, restore, and
    /// the prefix snapshots move whatever the rows are framed at, so they must
    /// never name an element type.
    pub fn kRowsBytes(self: *const KvCache, layer: u32, n_rows: u32) []const u8 {
        std.debug.assert(layer < self.n_layers);
        return self.rangeConst(self.kBase(layer), n_rows);
    }

    /// Byte range of positions `0..n_rows` of one layer's V half.
    pub fn vRowsBytes(self: *const KvCache, layer: u32, n_rows: u32) []const u8 {
        std.debug.assert(layer < self.n_layers);
        return self.rangeConst(self.vBase(layer), n_rows);
    }

    /// Mutable byte range of positions `0..n_rows` of one layer's K half.
    pub fn kRowsBytesMut(self: *KvCache, layer: u32, n_rows: u32) []u8 {
        std.debug.assert(layer < self.n_layers);
        return self.rangeMut(self.kBase(layer), n_rows);
    }

    /// Mutable byte range of positions `0..n_rows` of one layer's V half.
    pub fn vRowsBytesMut(self: *KvCache, layer: u32, n_rows: u32) []u8 {
        std.debug.assert(layer < self.n_layers);
        return self.rangeMut(self.vBase(layer), n_rows);
    }

    // -- clone --------------------------------------------------------------

    /// Copy the `n_pos` leading positions of every layer's K and V rows from
    /// `source` (a session fork). Rows past `n_pos` keep their old bytes.
    /// Attention reads only the `0..=pos` range of the owning session, the same
    /// rewrite-before-read argument that lets slot reuse skip the scrub.
    /// The copy is dtype-blind, because both slabs frame their rows the same way.
    pub fn copyFrom(self: *KvCache, source: *const KvCache, n_pos: u32) void {
        std.debug.assert(self.storage.len == source.storage.len);
        std.debug.assert(self.per_step_elems == source.per_step_elems);
        std.debug.assert(self.n_layers == source.n_layers);
        std.debug.assert(self.kv_dtype == source.kv_dtype);
        std.debug.assert(n_pos <= self.max_seq);
        if (n_pos == 0) return;
        log.debug("clone: {d} positions across {d} layers", .{ n_pos, self.n_layers });
        var layer: u32 = 0;
        while (layer < self.n_layers) : (layer += 1) {
            @memcpy(self.kRowsBytesMut(layer, n_pos), source.kRowsBytes(layer, n_pos));
            @memcpy(self.vRowsBytesMut(layer, n_pos), source.vRowsBytes(layer, n_pos));
        }
    }

    // -- addressing ---------------------------------------------------------

    fn kBase(self: *const KvCache, layer: u32) usize {
        // K-half starts at offset 0. The layer block stride is per_layer_elems.
        return @as(usize, layer) * self.per_layer_elems;
    }

    fn vBase(self: *const KvCache, layer: u32) usize {
        // V-half starts after all K-layers.
        return @as(usize, self.n_layers) * self.per_layer_elems + @as(usize, layer) * self.per_layer_elems;
    }

    /// Bytes one stored element occupies: 4 at f32, 2 at f16. It comes from the
    /// one row-cost function, so this file carries no element width of its own.
    fn elemBytes(self: *const KvCache) usize {
        return @intCast(self.kv_dtype.rowBytes(1));
    }

    /// Byte offset of element index `elems`. Every byte address here comes from
    /// this one product, which keeps the element-stride vocabulary above intact.
    fn byteOff(self: *const KvCache, elems: usize) usize {
        return elems * self.elemBytes();
    }

    /// The byte range of `n_rows` rows starting at element base `base`. It
    /// shares `byteOff` with the typed views, so a transport range and a typed
    /// view of the same rows address the same bytes by construction.
    fn rangeConst(self: *const KvCache, base: usize, n_rows: u32) []const u8 {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(n_rows <= self.max_seq);
        return self.storage[self.byteOff(base)..][0..self.byteOff(@as(usize, n_rows) * self.per_step_elems)];
    }

    fn rangeMut(self: *KvCache, base: usize, n_rows: u32) []u8 {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(n_rows <= self.max_seq);
        return self.storage[self.byteOff(base)..][0..self.byteOff(@as(usize, n_rows) * self.per_step_elems)];
    }

    /// The one tag check. Every typed view funnels through `viewConst` or
    /// `viewMut`, so no accessor can hand out a view the tag does not name.
    fn assertTag(self: *const KvCache, comptime E: type) void {
        std.debug.assert(self.kv_dtype == tagOf(E));
    }

    fn viewConst(self: *const KvCache, comptime E: type, elem_off: usize, elem_len: usize) []const E {
        self.assertTag(E);
        const bytes = self.storage[self.byteOff(elem_off)..][0..self.byteOff(elem_len)];
        return @alignCast(std.mem.bytesAsSlice(E, bytes));
    }

    fn viewMut(self: *KvCache, comptime E: type, elem_off: usize, elem_len: usize) []E {
        self.assertTag(E);
        const bytes = self.storage[self.byteOff(elem_off)..][0..self.byteOff(elem_len)];
        return @alignCast(std.mem.bytesAsSlice(E, bytes));
    }
};

// -- helpers ----------------------------------------------------------------

/// The tag a view element type names. Any other element type fails to compile,
/// which keeps the storage set and the view set one set.
fn tagOf(comptime E: type) dtype.KvDtype {
    if (E == f32) return .f32;
    if (E == f16) return .f16;
    @compileError("KvCache view: element type must be f32 or f16");
}

// -- tests ------------------------------------------------------------------

test "KvCache shapes and slice math" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var c = try KvCache.init(arena, 2, 3, 4, 8, .f32);
    try std.testing.expectEqual(@as(u32, 12), c.per_step_elems);
    try std.testing.expectEqual(@as(u32, 96), c.per_layer_elems);
    try std.testing.expectEqual(@as(usize, 384 * 4), c.storage.len);

    const k_dst = c.kDst(1, 2);
    for (k_dst, 0..) |*x, i| x.* = @floatFromInt(i + 100);

    const k_view = c.kSlice(1, 2);
    try std.testing.expectEqual(@as(usize, 3 * 12), k_view.len); // pos 0,1,2
    try std.testing.expectEqual(@as(f32, 100), k_view[2 * 12]); // pos 2 starts at offset 24
}

// A view at the wrong element type trips the `assertTag` precondition, the same
// way an out-of-range layer trips its own. An assert failure is not a testable
// outcome here, so that check stays a precondition and gets no case.
test "KvCache at f16 halves the slab and round-trips a row" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var c = try KvCache.init(arena, 2, 3, 4, 8, .f16);
    try std.testing.expectEqual(@as(u32, 12), c.per_step_elems);
    try std.testing.expectEqual(@as(u32, 96), c.per_layer_elems);
    try std.testing.expectEqual(@as(usize, 384 * 2), c.storage.len);

    const k_dst = c.kDstAs(f16, 1, 2);
    try std.testing.expectEqual(@as(usize, 12), k_dst.len);
    for (k_dst, 0..) |*x, i| x.* = @floatFromInt(i + 100);

    const k_view = c.kSliceAs(f16, 1, 2);
    try std.testing.expectEqual(@as(usize, 3 * 12), k_view.len);
    try std.testing.expectEqual(@as(f16, 100), k_view[2 * 12]);
    try std.testing.expectEqual(@as(f16, 111), k_view[2 * 12 + 11]);
    // Positions the write did not touch keep the zeroed slab.
    try std.testing.expectEqual(@as(f16, 0), k_view[0]);
}

test "KvCache byte ranges price the rows at the storage dtype" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The transport family sizes a blob from these ranges, so a range must
    // halve with the dtype and must cover exactly the rows the typed view does.
    var wide = try KvCache.init(arena, 2, 3, 4, 8, .f32);
    var narrow = try KvCache.init(arena, 2, 3, 4, 8, .f16);
    try std.testing.expectEqual(@as(usize, 3 * 12 * 4), wide.kRowsBytes(1, 3).len);
    try std.testing.expectEqual(@as(usize, 3 * 12 * 2), narrow.kRowsBytes(1, 3).len);
    try std.testing.expectEqual(wide.kRowsBytes(1, 3).len, wide.vRowsBytes(1, 3).len);

    const bytes = narrow.kRowsBytesMut(1, 3);
    for (bytes, 0..) |*b, i| b.* = @truncate(i + 1);
    const rows = narrow.kSliceAs(f16, 1, 2);
    try std.testing.expectEqualSlices(u8, bytes, std.mem.sliceAsBytes(rows));
}
