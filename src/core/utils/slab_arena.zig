//! An arena that gives each large request its own pages. A small request goes to the standard
//! arena. Nothing is free before `deinit`, as in any arena.
//!
//! Why. The standard arena asks its backing allocator for the request plus its last buffer, plus
//! half of that sum. Linux backs a page at the first touch, so the extra half costs nothing
//! there. Windows commits every byte of a request at once. A 2 GiB KV slab from the standard
//! arena then commits 3 GiB, and the RAM plan counts 2 GiB.
//!
//! Invariant. A block is a slab exactly when its length is at least `slab_min_bytes`. `alloc`
//! selects by the length, and no resize moves a length across the limit. So `resize`, `remap`,
//! and `free` know the owner of a block from its length alone.
//!
//! Layout of a slab: the header, padding up to the alignment of the request, then the bytes of
//! the caller. The backing allocator gets one request of exactly that size.
//!
//! Threads. Any thread can allocate. The slab list takes a new slab with a compare-and-swap, and
//! the standard arena is safe for threads. `deinit` runs on one thread, after the last use.
//!
//! Comptime. An assert keeps the limit at or above one page, so a slab never shares a page.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;
const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

/// A request of this size or more is a slab. Below it, the slack of the standard arena stays
/// small, because that arena then never holds a large buffer to grow from.
pub const slab_min_bytes: usize = 1 << 20;

comptime {
    assert(slab_min_bytes >= std.heap.page_size_max);
}

// -- types ------------------------------------------------------------------

pub const SlabArena = struct {
    backing: Allocator,
    small: std.heap.ArenaAllocator,
    first_slab: std.atomic.Value(?*SlabHeader),

    pub fn init(backing: Allocator) SlabArena {
        return .{
            .backing = backing,
            .small = .init(backing),
            .first_slab = .init(null),
        };
    }

    /// Free every slab and every small block. No other thread can use the arena now.
    pub fn deinit(self: *SlabArena) void {
        var next = self.first_slab.swap(null, .acquire);
        // The list holds one entry for each slab request, so the count of slabs bounds the loop.
        while (next) |header| {
            next = header.next;
            const raw: [*]u8 = @ptrCast(header);
            self.backing.rawFree(raw[0..header.raw_len], header.raw_alignment, @returnAddress());
        }

        self.small.deinit();
        self.* = undefined;
    }

    pub fn allocator(self: *SlabArena) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    /// The bytes that the backing allocator gave for the slabs, headers included.
    pub fn slabBytes(self: *const SlabArena) usize {
        var total: usize = 0;
        var next = self.first_slab.load(.acquire);
        while (next) |header| : (next = header.next) {
            total += header.raw_len;
        }

        return total;
    }

    fn alloc(ctx: *anyopaque, n: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
        const self: *SlabArena = @ptrCast(@alignCast(ctx));
        assert(n > 0);

        if (n < slab_min_bytes) {
            return self.small.allocator().rawAlloc(n, alignment, ret_addr);
        }

        return self.allocSlab(n, alignment, ret_addr);
    }

    fn allocSlab(self: *SlabArena, n: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
        assert(n >= slab_min_bytes);

        const raw_alignment = alignment.max(.of(SlabHeader));
        const header_span = alignment.forward(@sizeOf(SlabHeader));
        const raw_len = std.math.add(usize, header_span, n) catch return null;
        const raw = self.backing.rawAlloc(raw_len, raw_alignment, ret_addr) orelse return null;

        const header: *SlabHeader = @ptrCast(@alignCast(raw));
        header.* = .{
            .next = self.first_slab.load(.acquire),
            .raw_len = raw_len,
            .raw_alignment = raw_alignment,
        };
        // Another thread can push its own slab first. Then this slab takes the new head as next.
        while (self.first_slab.cmpxchgWeak(header.next, header, .release, .acquire)) |head| {
            header.next = head;
        }

        const bytes = raw + header_span;
        assert(alignment.check(@intFromPtr(bytes)));
        return bytes;
    }

    /// A slab keeps its pages, so it can only shrink, and only while it stays a slab.
    fn resize(
        ctx: *anyopaque,
        memory: []u8,
        alignment: Alignment,
        new_len: usize,
        ret_addr: usize,
    ) bool {
        const self: *SlabArena = @ptrCast(@alignCast(ctx));
        if (memory.len >= slab_min_bytes) {
            assert(self.ownsSlab(memory.ptr));
            return new_len >= slab_min_bytes and new_len <= memory.len;
        }
        if (new_len >= slab_min_bytes) return false;

        return self.small.allocator().rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remap(
        ctx: *anyopaque,
        memory: []u8,
        alignment: Alignment,
        new_len: usize,
        ret_addr: usize,
    ) ?[*]u8 {
        const self: *SlabArena = @ptrCast(@alignCast(ctx));
        if (memory.len >= slab_min_bytes) {
            return if (resize(ctx, memory, alignment, new_len, ret_addr)) memory.ptr else null;
        }
        if (new_len >= slab_min_bytes) return null;

        return self.small.allocator().rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
        const self: *SlabArena = @ptrCast(@alignCast(ctx));
        if (memory.len >= slab_min_bytes) {
            assert(self.ownsSlab(memory.ptr));
            return;
        }

        self.small.allocator().rawFree(memory, alignment, ret_addr);
    }

    /// True when `bytes` is the start of the caller bytes of one slab in the list.
    fn ownsSlab(self: *const SlabArena, bytes: [*]const u8) bool {
        const address = @intFromPtr(bytes);
        var next = self.first_slab.load(.acquire);
        while (next) |header| : (next = header.next) {
            const raw = @intFromPtr(header);
            if (address > raw and address < raw + header.raw_len) return true;
        }

        return false;
    }
};

/// The first bytes of a slab. The list of headers is the record that `deinit` frees from.
const SlabHeader = struct {
    next: ?*SlabHeader,
    raw_len: usize,
    raw_alignment: Alignment,
};

// -- tests ------------------------------------------------------------------

/// A backing allocator that records each request, so a test can read what the arena asked for.
const RecordingAllocator = struct {
    child: Allocator,
    requests: [requests_max]usize = undefined,
    count: usize = 0,

    const requests_max = 16;

    fn allocator(self: *RecordingAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = recordAlloc,
                .resize = recordResize,
                .remap = recordRemap,
                .free = recordFree,
            },
        };
    }

    fn recordAlloc(ctx: *anyopaque, n: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
        const self: *RecordingAllocator = @ptrCast(@alignCast(ctx));
        assert(self.count < requests_max);
        self.requests[self.count] = n;
        self.count += 1;
        return self.child.rawAlloc(n, alignment, ret_addr);
    }

    fn recordResize(ctx: *anyopaque, m: []u8, a: Alignment, n: usize, ret_addr: usize) bool {
        const self: *RecordingAllocator = @ptrCast(@alignCast(ctx));
        return self.child.rawResize(m, a, n, ret_addr);
    }

    fn recordRemap(ctx: *anyopaque, m: []u8, a: Alignment, n: usize, ret_addr: usize) ?[*]u8 {
        const self: *RecordingAllocator = @ptrCast(@alignCast(ctx));
        return self.child.rawRemap(m, a, n, ret_addr);
    }

    fn recordFree(ctx: *anyopaque, m: []u8, a: Alignment, ret_addr: usize) void {
        const self: *RecordingAllocator = @ptrCast(@alignCast(ctx));
        self.child.rawFree(m, a, ret_addr);
    }
};

test "SlabArena: a large request costs its own size plus one header, never half again" {
    var recording: RecordingAllocator = .{ .child = std.testing.allocator };
    var arena: SlabArena = .init(recording.allocator());
    defer arena.deinit();

    const a = arena.allocator();
    const first = try a.alignedAlloc(u8, .@"64", 3 * slab_min_bytes);
    const second = try a.alignedAlloc(u8, .@"64", 5 * slab_min_bytes);
    @memset(first, 1);
    @memset(second, 2);

    // Two requests reached the backing allocator, each within one alignment of its size.
    try std.testing.expectEqual(@as(usize, 2), recording.count);
    try std.testing.expect(recording.requests[0] >= first.len);
    try std.testing.expect(recording.requests[0] <= first.len + 64);
    try std.testing.expect(recording.requests[1] >= second.len);
    try std.testing.expect(recording.requests[1] <= second.len + 64);
    try std.testing.expectEqual(recording.requests[0] + recording.requests[1], arena.slabBytes());

    try std.testing.expect(std.mem.isAligned(@intFromPtr(first.ptr), 64));
    try std.testing.expect(std.mem.isAligned(@intFromPtr(second.ptr), 64));
    try std.testing.expectEqual(@as(u8, 1), first[first.len - 1]);
    try std.testing.expectEqual(@as(u8, 2), second[0]);
}

test "SlabArena: a small request never becomes a slab, and a slab never becomes small" {
    var arena: SlabArena = .init(std.testing.allocator);
    defer arena.deinit();

    const a = arena.allocator();

    // A small block that grows past the limit moves to a slab, with its bytes.
    var small = try a.alloc(u8, 4096);
    @memset(small, 7);
    try std.testing.expectEqual(@as(usize, 0), arena.slabBytes());
    try std.testing.expect(!a.resize(small, slab_min_bytes));

    small = try a.realloc(small, 2 * slab_min_bytes);
    try std.testing.expect(arena.slabBytes() >= 2 * slab_min_bytes);
    try std.testing.expectEqual(@as(u8, 7), small[4095]);

    // A slab shrinks in place down to the limit, and no further.
    try std.testing.expect(a.resize(small, slab_min_bytes));
    const shrunk: []u8 = small[0..slab_min_bytes];
    try std.testing.expect(!a.resize(shrunk, slab_min_bytes - 1));
    try std.testing.expect(!a.resize(shrunk, 2 * slab_min_bytes));

    // A free keeps the slab until `deinit`, and the leak check of the test allocator proves
    // that `deinit` frees it.
    const before = arena.slabBytes();
    a.free(shrunk);
    try std.testing.expectEqual(before, arena.slabBytes());
}

test "SlabArena: threads allocate slabs at the same time and none is lost" {
    const threads_n = 4;
    const slabs_per_thread = 8;

    var arena: SlabArena = .init(std.testing.allocator);
    defer arena.deinit();

    const Worker = struct {
        fn run(a: Allocator, failed: *std.atomic.Value(bool)) void {
            for (0..slabs_per_thread) |_| {
                const block = a.alloc(u8, slab_min_bytes) catch {
                    failed.store(true, .release);
                    return;
                };
                block[0] = 1;
            }
        }
    };

    var failed: std.atomic.Value(bool) = .init(false);
    var threads: [threads_n]std.Thread = undefined;
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, Worker.run, .{ arena.allocator(), &failed });
    }
    for (threads) |thread| {
        thread.join();
    }

    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expect(arena.slabBytes() >= threads_n * slabs_per_thread * slab_min_bytes);

    var count: usize = 0;
    var next = arena.first_slab.load(.acquire);
    while (next) |header| : (next = header.next) {
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, threads_n * slabs_per_thread), count);
}
