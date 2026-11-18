//! Role. The dispatch list of a trace build: the Vulkan commands that one runner step records, as
//! text lines. Two trees that write equal lists record the same commands, so `diff -u` of two list
//! files checks a change of the runner code without a device hash.
//!
//! Loader. `loader` is the device function loader of the context in a trace build. While the list
//! is on at load time, it returns a wrapper for each command of `Real`. A wrapper calls the driver
//! function with the same arguments, then records the command. Otherwise the loader returns the
//! driver function itself, so a trace run without the list holds no wrapper.
//!
//! Off by default. The flag `enabled` is off until `enable`. While it is off, `loader` returns the
//! driver function, so no wrapper runs. The threads of a traced test suite then share only the
//! driver loader, which `setRealLoader` stores once with an atomic compare-exchange. While the
//! list is on, every wrapper must run on the thread that called `enable`. A wrapper on another
//! thread records nothing and sets a fault.
//!
//! Lists. Each command buffer holds one list, which `vkBeginCommandBuffer` clears. Each submit
//! inside an open section appends a copy of the list of its command buffer to that section. The
//! decode ring submits a buffer again without a new record, so a list outlives its submit. A list
//! that no submit reads goes away at the next begin of its buffer.
//!
//! Entries. A dispatch names the shader of the bound pipeline and its group counts. Its push bytes
//! since the bind and the bindings of each bound set follow as indented lines. A binding states the
//! buffer size, the offset, and the range, but never a handle, because handles change per run. The
//! size comes from the buffer table at the time of the descriptor update.
//!
//! Push window. A pipeline bind resets the push window of its command buffer. The recorder relies
//! on the kernels to push after the bind, because the push bytes of a dispatch are the bytes
//! pushed since the last bind.
//!
//! Omitted. The list holds no `vkCmdWriteTimestamp`, `vkCmdResetQueryPool`, or
//! `vkResetCommandBuffer`. The dev command sets the GPU profile level to off, so no kernel records
//! a timestamp command. A command buffer that the driver resets without a begin keeps its list.
//!
//! Descriptor sets. An allocate wrapper clears the entry of each returned set, because a driver can
//! return a handle that an old set held. A destroy or a reset of a pool clears the sets of the
//! pool, and a free clears the freed sets.
//!
//! Bounds. `Caps` sizes every table once in `enable`, and `memory_bytes_max` bounds their sum. A
//! wrapper of a `vkCmd` command returns void, so a full table sets the flag `full`, which stays
//! set. `finish` then returns `error.DispatchListFull`. No table drops an entry without the flag.
//!
//! Faults. A wrapper never asserts on a command that the list cannot state, because an assert
//! vanishes in a release build. The wrapper calls the driver, then stores the first `Fault` in a
//! sticky field and returns. An unsupported command (a dynamic offset, an image barrier, a
//! descriptor copy, a descriptor that is no storage buffer, a null handle, a wrapper on another
//! thread) gives `error.DispatchListUnsupported`. A command that names an unknown shader, set,
//! buffer, or command buffer gives `error.DispatchListMissed`. `faultReason` names the reason.
//!
//! Comptime. A build check refuses a build without Vulkan or without trace. A block checks each
//! wrapper against the function type of its command.

const std = @import("std");
const vk = @import("vulkan");
const build_options = @import("build_options");
const core_log = @import("../core/utils/log.zig");
const kernels = @import("kernels.zig");

const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

comptime {
    if (!build_options.use_vulkan or !core_log.hot) {
        @compileError("ops_vulkan/dispatch_list.zig needs -Druntime=vulkan and -Dtrace=true.");
    }
}

// -- constants --------------------------------------------------------------

/// Vulkan guarantees at least 128 bytes of push constants, and no kernel pushes more.
pub const push_bytes_max: u32 = 128;

/// The push bytes of one entry line.
pub const push_chunk_bytes: u32 = 16;

/// The cap of `kernels.updateStorageBindings`, the only set layout of the kernels.
pub const bindings_max: u32 = 16;

/// Vulkan guarantees at least four bound sets.
pub const sets_max: u32 = 4;

pub const sections_max: u32 = 8;

/// The sum of the tables of the default `Caps` stays under this bound.
pub const memory_bytes_max: u64 = 64 * 1024 * 1024;

comptime {
    assert(push_bytes_max % push_chunk_bytes == 0);
    assert(bindings_max <= @bitSizeOf(u16));
    assert(Caps.default.bytes() <= memory_bytes_max);
}

// -- types ------------------------------------------------------------------

/// Why a list is not complete. The first fault of a recording stays.
pub const Fault = enum {
    dynamic_offsets,
    image_barriers,
    descriptor_copies,
    descriptor_type,
    null_handle,
    other_thread,
    driver_mismatch,
    unknown_shader,
    unknown_set,
    unknown_buffer,
    unknown_command_buffer,

    fn isMissed(fault: Fault) bool {
        return switch (fault) {
            .unknown_shader, .unknown_set, .unknown_buffer, .unknown_command_buffer => true,
            else => false,
        };
    }
};

pub const FinishError = error{ DispatchListFull, DispatchListUnsupported, DispatchListMissed };

/// The capacity of each table. A run that fills one gets `error.DispatchListFull` from `finish`.
pub const Caps = struct {
    command_buffers: u32,
    /// The entries of one command buffer list.
    list_entries: u32,
    /// The entries of all sections together.
    section_entries: u32,
    buffers: u32,
    sets: u32,
    modules: u32,
    pipelines: u32,

    pub const default: Caps = .{
        .command_buffers = 16,
        .list_entries = 32 * 1024,
        .section_entries = 256 * 1024,
        .buffers = 4096,
        .sets = 4096,
        .modules = 1024,
        .pipelines = 1024,
    };

    pub fn bytes(caps: Caps) u64 {
        const entry: u64 = @sizeOf(Entry);
        const lists = @as(u64, caps.command_buffers) * (caps.list_entries * entry +
            @sizeOf(CommandList));
        const sections = caps.section_entries * entry;
        const buffers = @as(u64, caps.buffers) * (@sizeOf(u64) * 2);
        const sets = @as(u64, caps.sets) * (@sizeOf(u64) + @sizeOf(SetBindings));
        const names = @as(u64, caps.modules + caps.pipelines) * (@sizeOf(u64) + @sizeOf([]u8));
        return lists + sections + buffers + sets + names;
    }
};

/// One line of a list. A dispatch, a barrier, and a copy carry their parts as the lines after.
pub const Entry = union(enum) {
    /// A submit of a command buffer that no begin recorded has a null count.
    submit: ?u32,
    dispatch: struct { shader: []const u8, groups: [3]u32 },
    push: struct { offset: u8, len: u8, bytes: [push_chunk_bytes]u8 },
    /// A set handle that no update named has a null count.
    set: struct { index: u8, bindings: ?u8 },
    binding: struct { index: u8, size: ?u64, offset: u64, range: u64 },
    barrier: struct { src_stage: u32, dst_stage: u32, dependency: u32 },
    memory: struct { src_access: u32, dst_access: u32 },
    buffer: struct { src_access: u32, dst_access: u32, offset: u64, size: u64 },
    copy: struct { regions: u32, src_size: ?u64, dst_size: ?u64 },
    region: struct { size: u64, src_offset: u64, dst_offset: u64 },
    fill: struct { offset: u64, size: u64, value: u32 },
};

/// The buffer of one descriptor binding, as the last update wrote it. The size is the buffer size
/// at the update, and null names a buffer that no create recorded.
const Binding = struct { buffer: u64, size: ?u64, offset: u64, range: u64 };

const SetBindings = struct {
    /// The pool that allocated the set, or zero for a set that no allocate named.
    pool: u64 = 0,
    written: u16 = 0,
    bindings: [bindings_max]Binding = undefined,
};

/// The list and the bind state of one command buffer.
const CommandList = struct {
    key: u64,
    entries: []Entry,
    len: u32 = 0,
    shader: []const u8 = shader_none,
    sets: [sets_max]u64 = @splat(0),
    set_count: u32 = 0,
    push: [push_bytes_max]u8 = @splat(0),
    push_lo: u32 = push_bytes_max,
    push_hi: u32 = 0,
};

/// The names of a dispatch whose shader no table gave.
const shader_none = "none";
const shader_unknown = "unknown";

const no_bindings: SetBindings = .{};

const Section = struct { name: []const u8, start: u32, len: u32 = 0 };

/// A map from a handle to a value with a fixed capacity and a linear search.
fn Table(comptime V: type) type {
    return struct {
        const Self = @This();

        keys: []u64,
        values: []V,
        len: u32 = 0,

        fn init(gpa: Allocator, cap: u32) Allocator.Error!Self {
            assert(cap > 0);

            const keys = try gpa.alloc(u64, cap);
            errdefer gpa.free(keys);
            return .{ .keys = keys, .values = try gpa.alloc(V, cap) };
        }

        fn deinit(self: *Self, gpa: Allocator) void {
            gpa.free(self.keys);
            gpa.free(self.values);
        }

        fn get(self: *Self, key: u64) ?*V {
            for (self.keys[0..self.len], 0..) |k, i| {
                if (k == key) return &self.values[i];
            }

            return null;
        }

        /// The value of `key`, new or old. A new key in a full table returns null.
        fn put(self: *Self, key: u64, value: V) ?*V {
            assert(self.len <= self.keys.len);

            if (self.get(key)) |old| {
                old.* = value;
                return old;
            }

            if (self.len == self.keys.len) return null;

            self.keys[self.len] = key;
            self.values[self.len] = value;
            self.len += 1;
            return &self.values[self.len - 1];
        }

        fn remove(self: *Self, key: u64) void {
            for (self.keys[0..self.len], 0..) |k, i| {
                if (k != key) continue;

                self.len -= 1;
                self.keys[i] = self.keys[self.len];
                self.values[i] = self.values[self.len];
                return;
            }
        }
    };
}

/// The tables and the lists of one recording. The wrappers drive the one instance in `global`, and
/// a test drives one of its own with plain handle values.
pub const DispatchList = struct {
    gpa: Allocator,
    pool: []Entry,
    lists: []CommandList,
    list_count: u32 = 0,
    section_pool: []Entry,
    section_len: u32 = 0,
    sections: [sections_max]Section = undefined,
    section_count: u32 = 0,
    section_open: bool = false,
    sections_written: u32 = 0,
    buffers: Table(u64),
    sets: Table(SetBindings),
    modules: Table([]const u8),
    pipelines: Table([]const u8),
    full: bool = false,
    fault: ?Fault = null,

    pub fn init(gpa: Allocator, caps: Caps) Allocator.Error!DispatchList {
        assert(caps.command_buffers > 0);
        assert(caps.list_entries > 0);
        assert(caps.section_entries > 0);

        const pool = try gpa.alloc(Entry, @as(usize, caps.command_buffers) * caps.list_entries);
        errdefer gpa.free(pool);
        const lists = try gpa.alloc(CommandList, caps.command_buffers);
        errdefer gpa.free(lists);
        const section_pool = try gpa.alloc(Entry, caps.section_entries);
        errdefer gpa.free(section_pool);
        var buffers: Table(u64) = try .init(gpa, caps.buffers);
        errdefer buffers.deinit(gpa);
        var sets: Table(SetBindings) = try .init(gpa, caps.sets);
        errdefer sets.deinit(gpa);
        var modules: Table([]const u8) = try .init(gpa, caps.modules);
        errdefer modules.deinit(gpa);
        const pipelines: Table([]const u8) = try .init(gpa, caps.pipelines);

        for (lists, 0..) |*list, i| {
            list.* = .{ .key = 0, .entries = pool[i * caps.list_entries ..][0..caps.list_entries] };
        }

        return .{
            .gpa = gpa,
            .pool = pool,
            .lists = lists,
            .section_pool = section_pool,
            .buffers = buffers,
            .sets = sets,
            .modules = modules,
            .pipelines = pipelines,
        };
    }

    pub fn deinit(self: *DispatchList) void {
        self.gpa.free(self.pool);
        self.gpa.free(self.lists);
        self.gpa.free(self.section_pool);
        self.buffers.deinit(self.gpa);
        self.sets.deinit(self.gpa);
        self.modules.deinit(self.gpa);
        self.pipelines.deinit(self.gpa);
        self.* = undefined;
    }

    /// Store `reason` when no earlier fault stands.
    pub fn fail(self: *DispatchList, reason: Fault) void {
        if (self.fault == null) {
            self.fault = reason;
        }
    }

    pub fn createModule(self: *DispatchList, module: u64, code: *const anyopaque) void {
        if (module == 0) return self.fail(.null_handle);

        if (self.modules.put(module, shaderName(code)) == null) {
            self.full = true;
        }
    }

    pub fn createPipeline(self: *DispatchList, pipeline: u64, module: u64) void {
        if (pipeline == 0) return self.fail(.null_handle);

        const name: []const u8 = if (self.modules.get(module)) |n| n.* else shader_unknown;
        if (self.pipelines.put(pipeline, name) == null) {
            self.full = true;
        }
    }

    pub fn createBuffer(self: *DispatchList, buffer: u64, size: u64) void {
        if (buffer == 0) return self.fail(.null_handle);

        if (self.buffers.put(buffer, size) == null) {
            self.full = true;
        }
    }

    pub fn destroyBuffer(self: *DispatchList, buffer: u64) void {
        self.buffers.remove(buffer);
        assert(self.buffers.get(buffer) == null);
    }

    /// Record the buffer of one binding with its size at this update.
    pub fn writeBinding(
        self: *DispatchList,
        set: u64,
        index: u32,
        buffer: u64,
        offset: u64,
        range: u64,
    ) void {
        if (set == 0) return self.fail(.null_handle);

        if (index >= bindings_max) {
            self.full = true;
            return;
        }

        const entry = self.sets.get(set) orelse self.sets.put(set, .{}) orelse {
            self.full = true;
            return;
        };
        const size = self.bufferSize(buffer);
        if (size == null) {
            self.fail(.unknown_buffer);
        }

        entry.bindings[index] = .{
            .buffer = buffer,
            .size = size,
            .offset = offset,
            .range = range,
        };
        entry.written |= @as(u16, 1) << @intCast(index);
    }

    /// A new set holds no binding, even when the driver reuses the handle of a freed set.
    pub fn allocateSet(self: *DispatchList, set: u64, pool: u64) void {
        if (set == 0 or pool == 0) return self.fail(.null_handle);

        if (self.sets.put(set, .{ .pool = pool }) == null) {
            self.full = true;
        }
    }

    pub fn freeSet(self: *DispatchList, set: u64) void {
        self.sets.remove(set);
    }

    /// Clear each set that the pool allocated.
    pub fn clearPool(self: *DispatchList, pool: u64) void {
        var i: u32 = 0;
        while (i < self.sets.len) {
            if (self.sets.values[i].pool == pool) {
                self.sets.remove(self.sets.keys[i]);
            } else {
                i += 1;
            }
        }
    }

    /// Clear the list of `cb`, or claim a list for a new command buffer.
    pub fn begin(self: *DispatchList, cb: u64) void {
        if (cb == 0) return self.fail(.null_handle);

        const list = self.listOf(cb) orelse blk: {
            if (self.list_count == self.lists.len) {
                self.full = true;
                return;
            }

            self.list_count += 1;
            break :blk &self.lists[self.list_count - 1];
        };
        list.* = .{ .key = cb, .entries = list.entries };
        assert(list.len == 0);
    }

    pub fn bindPipeline(self: *DispatchList, cb: u64, pipeline: u64) void {
        const list = self.listOf(cb) orelse return self.noList();
        list.shader = if (self.pipelines.get(pipeline)) |n| n.* else shader_unknown;
        list.push_lo = push_bytes_max;
        list.push_hi = 0;
    }

    pub fn bindSets(self: *DispatchList, cb: u64, first: u32, sets: []const u64) void {
        const list = self.listOf(cb) orelse return self.noList();
        if (first + sets.len > sets_max) {
            self.full = true;
            return;
        }

        for (sets, first..) |set, i| {
            list.sets[i] = set;
        }

        list.set_count = @max(list.set_count, first + @as(u32, @intCast(sets.len)));
    }

    pub fn push(self: *DispatchList, cb: u64, offset: u32, bytes: []const u8) void {
        const list = self.listOf(cb) orelse return self.noList();
        if (offset + bytes.len > push_bytes_max) {
            self.full = true;
            return;
        }

        const end: u32 = offset + @as(u32, @intCast(bytes.len));
        @memcpy(list.push[offset..end], bytes);
        list.push_lo = @min(list.push_lo, offset);
        list.push_hi = @max(list.push_hi, end);
    }

    /// Append the dispatch line, then its push bytes, then each bound set and its bindings.
    pub fn dispatch(self: *DispatchList, cb: u64, groups: [3]u32) void {
        const list = self.listOf(cb) orelse return self.noList();
        if (std.mem.eql(u8, list.shader, shader_none) or
            std.mem.eql(u8, list.shader, shader_unknown))
        {
            self.fail(.unknown_shader);
        }

        self.append(list, .{ .dispatch = .{ .shader = list.shader, .groups = groups } });

        var at = list.push_lo;
        while (at < list.push_hi) : (at += push_chunk_bytes) {
            const len = @min(push_chunk_bytes, list.push_hi - at);
            var chunk: [push_chunk_bytes]u8 = @splat(0);
            @memcpy(chunk[0..len], list.push[at..][0..len]);
            const line: Entry = .{
                .push = .{ .offset = @intCast(at), .len = @intCast(len), .bytes = chunk },
            };
            self.append(list, line);
        }

        for (list.sets[0..list.set_count], 0..) |set, s| {
            self.appendSet(list, @intCast(s), set);
        }
    }

    pub fn barrier(self: *DispatchList, cb: u64, entry: Entry) void {
        assert(entry == .barrier or entry == .memory or entry == .buffer);

        const list = self.listOf(cb) orelse return self.noList();
        self.append(list, entry);
    }

    /// Append the copy line with the sizes of both buffers, then one line per region.
    pub fn copy(
        self: *DispatchList,
        cb: u64,
        src: u64,
        dst: u64,
        regions: []const vk.BufferCopy,
    ) void {
        const list = self.listOf(cb) orelse return self.noList();
        self.append(list, .{ .copy = .{
            .regions = @intCast(regions.len),
            .src_size = self.bufferSize(src),
            .dst_size = self.bufferSize(dst),
        } });

        for (regions) |r| {
            const line: Entry = .{ .region = .{
                .size = r.size,
                .src_offset = r.src_offset,
                .dst_offset = r.dst_offset,
            } };
            self.append(list, line);
        }
    }

    pub fn fill(self: *DispatchList, cb: u64, offset: u64, size: u64, value: u32) void {
        const list = self.listOf(cb) orelse return self.noList();
        self.append(list, .{ .fill = .{ .offset = offset, .size = size, .value = value } });
    }

    /// Append a copy of the list of `cb` to the open section. Outside a section a submit adds
    /// nothing.
    pub fn submit(self: *DispatchList, cb: u64) void {
        if (!self.section_open) return;

        const section = &self.sections[self.section_count - 1];
        const list = self.listOf(cb) orelse {
            self.appendSection(section, &.{.{ .submit = null }});
            return;
        };

        self.appendSection(section, &.{.{ .submit = list.len }});
        self.appendSection(section, list.entries[0..list.len]);
    }

    pub fn beginSection(self: *DispatchList, name: []const u8) void {
        assert(!self.section_open);
        assert(name.len > 0);

        if (self.section_count == sections_max) {
            self.full = true;
            return;
        }

        self.sections[self.section_count] = .{ .name = name, .start = self.section_len };
        self.section_count += 1;
        self.section_open = true;
    }

    pub fn endSection(self: *DispatchList) void {
        if (self.full and !self.section_open) return;

        assert(self.section_open);
        self.section_open = false;
    }

    pub fn finish(self: *const DispatchList) FinishError!void {
        if (self.full) return error.DispatchListFull;

        const fault = self.fault orelse return;
        if (fault.isMissed()) return error.DispatchListMissed;

        return error.DispatchListUnsupported;
    }

    /// Write each closed section that no earlier call wrote.
    pub fn write(self: *DispatchList, w: *std.Io.Writer) (FinishError || std.Io.Writer.Error)!void {
        try self.finish();

        const closed = self.section_count - @intFromBool(self.section_open);
        for (self.sections[self.sections_written..closed]) |section| {
            try w.print("section {s}\n", .{section.name});
            for (self.section_pool[section.start..][0..section.len]) |entry| {
                try writeEntry(w, entry);
            }
        }

        self.sections_written = closed;
    }

    fn listOf(self: *DispatchList, cb: u64) ?*CommandList {
        for (self.lists[0..self.list_count]) |*list| {
            if (list.key == cb) return list;
        }

        return null;
    }

    /// A command of a buffer that got no list. A full list table leaves a buffer without one, and
    /// the flag `full` states that. Any other buffer without a list was never begun.
    fn noList(self: *DispatchList) void {
        if (!self.full) {
            self.fail(.unknown_command_buffer);
        }
    }

    fn append(self: *DispatchList, list: *CommandList, entry: Entry) void {
        if (list.len == list.entries.len) {
            self.full = true;
            return;
        }

        list.entries[list.len] = entry;
        list.len += 1;
    }

    fn appendSection(self: *DispatchList, section: *Section, entries: []const Entry) void {
        assert(section.start + section.len == self.section_len);

        if (self.section_len + entries.len > self.section_pool.len) {
            self.full = true;
            return;
        }

        @memcpy(self.section_pool[self.section_len..][0..entries.len], entries);
        self.section_len += @intCast(entries.len);
        section.len += @intCast(entries.len);
    }

    fn appendSet(self: *DispatchList, list: *CommandList, index: u8, set: u64) void {
        const bindings: *const SetBindings = self.sets.get(set) orelse &no_bindings;
        if (bindings.written == 0) {
            self.fail(.unknown_set);
            self.append(list, .{ .set = .{ .index = index, .bindings = null } });
            return;
        }

        const count: u8 = @popCount(bindings.written);
        self.append(list, .{ .set = .{ .index = index, .bindings = count } });
        for (bindings.bindings, 0..) |b, i| {
            if (bindings.written & (@as(u16, 1) << @intCast(i)) == 0) continue;

            self.append(list, .{ .binding = .{
                .index = @intCast(i),
                .size = b.size,
                .offset = b.offset,
                .range = b.range,
            } });
        }
    }

    fn bufferSize(self: *DispatchList, buffer: u64) ?u64 {
        const size = self.buffers.get(buffer) orelse return null;
        return size.*;
    }
};

/// The driver functions behind the wrappers. Each field name is a command name, and a wrapper of
/// the same name in this file stands in for it.
const Real = struct {
    vkBeginCommandBuffer: ?vk.PfnBeginCommandBuffer = null,
    vkCmdBindPipeline: ?vk.PfnCmdBindPipeline = null,
    vkCmdBindDescriptorSets: ?vk.PfnCmdBindDescriptorSets = null,
    vkCmdPushConstants: ?vk.PfnCmdPushConstants = null,
    vkCmdDispatch: ?vk.PfnCmdDispatch = null,
    vkCmdPipelineBarrier: ?vk.PfnCmdPipelineBarrier = null,
    vkCmdCopyBuffer: ?vk.PfnCmdCopyBuffer = null,
    vkCmdFillBuffer: ?vk.PfnCmdFillBuffer = null,
    vkQueueSubmit: ?vk.PfnQueueSubmit = null,
    vkCreateShaderModule: ?vk.PfnCreateShaderModule = null,
    vkCreateComputePipelines: ?vk.PfnCreateComputePipelines = null,
    vkUpdateDescriptorSets: ?vk.PfnUpdateDescriptorSets = null,
    vkAllocateDescriptorSets: ?vk.PfnAllocateDescriptorSets = null,
    vkFreeDescriptorSets: ?vk.PfnFreeDescriptorSets = null,
    vkResetDescriptorPool: ?vk.PfnResetDescriptorPool = null,
    vkDestroyDescriptorPool: ?vk.PfnDestroyDescriptorPool = null,
    vkCreateBuffer: ?vk.PfnCreateBuffer = null,
    vkDestroyBuffer: ?vk.PfnDestroyBuffer = null,
};

comptime {
    for (std.meta.fields(Real)) |f| {
        const Pfn = @typeInfo(f.type).optional.child;
        if (@TypeOf(&@field(@This(), f.name)) != Pfn) {
            @compileError("the wrapper " ++ f.name ++ " differs from the type of its command");
        }
    }
}

// -- state ------------------------------------------------------------------

/// The address of the driver loader, or zero before `setRealLoader`.
var real_get_device_proc_addr: std.atomic.Value(usize) = .init(0);
var real: Real = .{};

/// The first fault that a wrapper cannot store in `global`, as a `Fault` plus one, or zero.
var late_fault: std.atomic.Value(u8) = .init(0);

var enabled: std.atomic.Value(bool) = .init(false);
var owner: std.Thread.Id = undefined;
var global: DispatchList = undefined;

// -- api --------------------------------------------------------------------

/// Set the driver loader that `loader` asks. A second context must name the same loader. The
/// first call stores it, and a call with another loader sets the fault `driver_mismatch`.
pub fn setRealLoader(get: vk.PfnGetDeviceProcAddr) void {
    const address = @intFromPtr(get);
    const old = real_get_device_proc_addr.cmpxchgStrong(0, address, .acq_rel, .acquire) orelse {
        return;
    };
    if (old != address) {
        setLateFault(.driver_mismatch);
    }
}

/// The loader argument of `vk.DeviceWrapper.load`. The header states when it returns a wrapper.
pub fn loader(device: vk.Device, name: [*:0]const u8) vk.PfnVoidFunction {
    const get: vk.PfnGetDeviceProcAddr = @ptrFromInt(real_get_device_proc_addr.load(.acquire));
    const driver = get(device, name) orelse return null;
    if (!enabled.load(.acquire)) return driver;

    const command = std.mem.span(name);
    inline for (std.meta.fields(Real)) |f| {
        if (std.mem.eql(u8, command, f.name)) {
            const typed: f.type = @ptrCast(driver);
            // The wrappers hold one driver function each, so a second device must load the same.
            if (@field(real, f.name)) |old| {
                if (old != typed.?) {
                    setLateFault(.driver_mismatch);
                }
            }

            @field(real, f.name) = typed;
            return @ptrCast(&@field(@This(), f.name));
        }
    }

    return driver;
}

/// Start the recording on this thread. Call it before the context loads its device functions.
pub fn enable(gpa: Allocator, caps: Caps) Allocator.Error!void {
    assert(!enabled.load(.acquire));
    assert(caps.bytes() <= memory_bytes_max);

    global = try .init(gpa, caps);
    late_fault.store(0, .release);
    owner = std.Thread.getCurrentId();
    enabled.store(true, .release);
}

/// Stop the recording and free the tables. The wrappers then only call the driver.
pub fn deinit() void {
    if (!enabled.load(.acquire)) return;

    assert(std.Thread.getCurrentId() == owner);
    enabled.store(false, .release);
    global.deinit();
}

pub fn beginSection(name: []const u8) void {
    recording().?.beginSection(name);
}

pub fn endSection() void {
    recording().?.endSection();
}

pub fn finish() FinishError!void {
    try lateFinish();
    return recording().?.finish();
}

pub fn write(w: *std.Io.Writer) (FinishError || std.Io.Writer.Error)!void {
    try lateFinish();
    return recording().?.write(w);
}

/// The first fault of the recording, or null. Call it after `finish` returns an error.
pub fn faultReason() ?Fault {
    if (late()) |reason| return reason;

    return if (enabled.load(.acquire)) global.fault else null;
}

/// The name of the SPIR-V blob at `code`, or `unknown` for an address outside the table.
pub fn shaderName(code: *const anyopaque) []const u8 {
    for (kernels.spv_names) |spv_name| {
        if (spv_name.code == code) return spv_name.name;
    }

    return shader_unknown;
}

// -- helpers ----------------------------------------------------------------

fn recording() ?*DispatchList {
    if (!enabled.load(.acquire)) return null;

    if (std.Thread.getCurrentId() != owner) {
        setLateFault(.other_thread);
        return null;
    }

    return &global;
}

fn setLateFault(reason: Fault) void {
    _ = late_fault.cmpxchgStrong(0, @as(u8, @intFromEnum(reason)) + 1, .acq_rel, .acquire);
}

fn late() ?Fault {
    const stored = late_fault.load(.acquire);
    if (stored == 0) return null;

    return @enumFromInt(stored - 1);
}

fn lateFinish() FinishError!void {
    const reason = late() orelse return;
    if (reason.isMissed()) return error.DispatchListMissed;

    return error.DispatchListUnsupported;
}

fn writeEntry(w: *std.Io.Writer, entry: Entry) std.Io.Writer.Error!void {
    switch (entry) {
        .submit => |n| {
            if (n) |count| return w.print("submit entries={d}\n", .{count});
            return w.writeAll("submit unrecorded\n");
        },
        .dispatch => |d| try w.print("dispatch {s} groups={d},{d},{d}\n", .{
            d.shader, d.groups[0], d.groups[1], d.groups[2],
        }),
        .push => |p| try w.print("  push offset={d} bytes={x}\n", .{ p.offset, p.bytes[0..p.len] }),
        .set => |s| {
            if (s.bindings) |n| return w.print("  set {d} bindings={d}\n", .{ s.index, n });
            return w.print("  set {d} unknown\n", .{s.index});
        },
        .binding => |b| {
            try w.print("    binding {d} size=", .{b.index});
            try writeSize(w, b.size);
            try w.print(" offset={d} range=", .{b.offset});
            try writeSize(w, b.range);
            try w.writeAll("\n");
        },
        .barrier => |b| try w.print("barrier src_stage=0x{x:0>8} dst_stage=0x{x:0>8} " ++
            "dependency=0x{x}\n", .{ b.src_stage, b.dst_stage, b.dependency }),
        .memory => |m| try w.print("  memory src_access=0x{x:0>8} dst_access=0x{x:0>8}\n", .{
            m.src_access, m.dst_access,
        }),
        .buffer => |b| try w.print("  buffer src_access=0x{x:0>8} dst_access=0x{x:0>8} " ++
            "offset={d} size={d}\n", .{ b.src_access, b.dst_access, b.offset, b.size }),
        .copy => |c| {
            try w.print("copy regions={d} src_size=", .{c.regions});
            try writeSize(w, c.src_size);
            try w.writeAll(" dst_size=");
            try writeSize(w, c.dst_size);
            try w.writeAll("\n");
        },
        .region => |r| try w.print("  region size={d} src_offset={d} dst_offset={d}\n", .{
            r.size, r.src_offset, r.dst_offset,
        }),
        .fill => |f| try w.print("fill offset={d} size={d} value=0x{x:0>8}\n", .{
            f.offset, f.size, f.value,
        }),
    }
}

/// A size of `vk.WHOLE_SIZE` reads `whole`, and a buffer that no create recorded reads `unknown`.
fn writeSize(w: *std.Io.Writer, size: ?u64) std.Io.Writer.Error!void {
    const n = size orelse return w.writeAll("unknown");
    if (n == vk.WHOLE_SIZE) return w.writeAll("whole");

    return w.print("{d}", .{n});
}

fn handleKey(handle: anytype) u64 {
    return @intFromEnum(handle);
}

fn flags(mask: anytype) u32 {
    return @bitCast(mask);
}

// -- wrappers ---------------------------------------------------------------

// Each wrapper calls the driver first, so a create wrapper records only a new handle.

fn vkBeginCommandBuffer(
    cb: vk.CommandBuffer,
    info: *const vk.CommandBufferBeginInfo,
) callconv(vk.vulkan_call_conv) vk.Result {
    const result = real.vkBeginCommandBuffer.?(cb, info);
    if (recording()) |list| {
        if (result == .success) {
            list.begin(handleKey(cb));
        }
    }

    return result;
}

fn vkCmdBindPipeline(
    cb: vk.CommandBuffer,
    bind_point: vk.PipelineBindPoint,
    pipeline: vk.Pipeline,
) callconv(vk.vulkan_call_conv) void {
    real.vkCmdBindPipeline.?(cb, bind_point, pipeline);
    if (recording()) |list| {
        list.bindPipeline(handleKey(cb), handleKey(pipeline));
    }
}

fn vkCmdBindDescriptorSets(
    cb: vk.CommandBuffer,
    bind_point: vk.PipelineBindPoint,
    layout: vk.PipelineLayout,
    first_set: u32,
    set_count: u32,
    sets: [*]const vk.DescriptorSet,
    dynamic_offset_count: u32,
    dynamic_offsets: ?[*]const u32,
) callconv(vk.vulkan_call_conv) void {
    real.vkCmdBindDescriptorSets.?(
        cb,
        bind_point,
        layout,
        first_set,
        set_count,
        sets,
        dynamic_offset_count,
        dynamic_offsets,
    );
    const list = recording() orelse return;
    if (dynamic_offset_count != 0) return list.fail(.dynamic_offsets);

    var keys: [sets_max]u64 = undefined;
    const n = @min(set_count, sets_max);
    for (sets[0..n], keys[0..n]) |set, *k| {
        k.* = handleKey(set);
    }

    if (set_count > sets_max) {
        list.full = true;
        return;
    }

    list.bindSets(handleKey(cb), first_set, keys[0..n]);
}

fn vkCmdPushConstants(
    cb: vk.CommandBuffer,
    layout: vk.PipelineLayout,
    stages: vk.ShaderStageFlags,
    offset: u32,
    size: u32,
    values: *const anyopaque,
) callconv(vk.vulkan_call_conv) void {
    real.vkCmdPushConstants.?(cb, layout, stages, offset, size, values);
    const list = recording() orelse return;

    const bytes: [*]const u8 = @ptrCast(values);
    list.push(handleKey(cb), offset, bytes[0..size]);
}

fn vkCmdDispatch(
    cb: vk.CommandBuffer,
    x: u32,
    y: u32,
    z: u32,
) callconv(vk.vulkan_call_conv) void {
    real.vkCmdDispatch.?(cb, x, y, z);
    if (recording()) |list| {
        list.dispatch(handleKey(cb), .{ x, y, z });
    }
}

fn vkCmdPipelineBarrier(
    cb: vk.CommandBuffer,
    src_stage: vk.PipelineStageFlags,
    dst_stage: vk.PipelineStageFlags,
    dependency: vk.DependencyFlags,
    memory_count: u32,
    memory: ?[*]const vk.MemoryBarrier,
    buffer_count: u32,
    buffers: ?[*]const vk.BufferMemoryBarrier,
    image_count: u32,
    images: ?[*]const vk.ImageMemoryBarrier,
) callconv(vk.vulkan_call_conv) void {
    real.vkCmdPipelineBarrier.?(
        cb,
        src_stage,
        dst_stage,
        dependency,
        memory_count,
        memory,
        buffer_count,
        buffers,
        image_count,
        images,
    );
    // Refuse an image barrier, which the list cannot print.
    const list = recording() orelse return;

    if (image_count != 0) return list.fail(.image_barriers);

    // Record the barrier line, then one line per memory barrier and per buffer barrier.
    const cb_key = handleKey(cb);
    list.barrier(cb_key, .{ .barrier = .{
        .src_stage = flags(src_stage),
        .dst_stage = flags(dst_stage),
        .dependency = flags(dependency),
    } });

    if (memory) |m| {
        for (m[0..memory_count]) |b| {
            const line: Entry = .{ .memory = .{
                .src_access = flags(b.src_access_mask),
                .dst_access = flags(b.dst_access_mask),
            } };
            list.barrier(cb_key, line);
        }
    }

    if (buffers) |bs| {
        for (bs[0..buffer_count]) |b| {
            list.barrier(cb_key, .{ .buffer = .{
                .src_access = flags(b.src_access_mask),
                .dst_access = flags(b.dst_access_mask),
                .offset = b.offset,
                .size = b.size,
            } });
        }
    }
}

fn vkCmdCopyBuffer(
    cb: vk.CommandBuffer,
    src: vk.Buffer,
    dst: vk.Buffer,
    region_count: u32,
    regions: [*]const vk.BufferCopy,
) callconv(vk.vulkan_call_conv) void {
    real.vkCmdCopyBuffer.?(cb, src, dst, region_count, regions);
    const list = recording() orelse return;
    list.copy(handleKey(cb), handleKey(src), handleKey(dst), regions[0..region_count]);
}

fn vkCmdFillBuffer(
    cb: vk.CommandBuffer,
    dst: vk.Buffer,
    offset: vk.DeviceSize,
    size: vk.DeviceSize,
    value: u32,
) callconv(vk.vulkan_call_conv) void {
    real.vkCmdFillBuffer.?(cb, dst, offset, size, value);
    if (recording()) |list| {
        list.fill(handleKey(cb), offset, size, value);
    }
}

fn vkQueueSubmit(
    queue: vk.Queue,
    submit_count: u32,
    submits: ?[*]const vk.SubmitInfo,
    fence: vk.Fence,
) callconv(vk.vulkan_call_conv) vk.Result {
    const result = real.vkQueueSubmit.?(queue, submit_count, submits, fence);
    if (result != .success) return result;

    const list = recording() orelse return result;
    const infos = submits orelse return result;

    for (infos[0..submit_count]) |info| {
        const cbs = info.p_command_buffers orelse continue;
        for (cbs[0..info.command_buffer_count]) |cb| {
            list.submit(handleKey(cb));
        }
    }

    return result;
}

fn vkCreateShaderModule(
    device: vk.Device,
    info: *const vk.ShaderModuleCreateInfo,
    allocator: ?*const vk.AllocationCallbacks,
    module: *vk.ShaderModule,
) callconv(vk.vulkan_call_conv) vk.Result {
    const result = real.vkCreateShaderModule.?(device, info, allocator, module);
    if (recording()) |list| {
        if (result == .success) {
            list.createModule(handleKey(module.*), info.p_code);
        }
    }

    return result;
}

fn vkCreateComputePipelines(
    device: vk.Device,
    cache: vk.PipelineCache,
    info_count: u32,
    infos: [*]const vk.ComputePipelineCreateInfo,
    allocator: ?*const vk.AllocationCallbacks,
    pipelines: [*]vk.Pipeline,
) callconv(vk.vulkan_call_conv) vk.Result {
    const result = real.vkCreateComputePipelines.?(
        device,
        cache,
        info_count,
        infos,
        allocator,
        pipelines,
    );
    const list = recording() orelse return result;
    if (result != .success) return result;

    for (infos[0..info_count], pipelines[0..info_count]) |info, pipeline| {
        list.createPipeline(handleKey(pipeline), handleKey(info.stage.module));
    }

    return result;
}

fn vkUpdateDescriptorSets(
    device: vk.Device,
    write_count: u32,
    writes: ?[*]const vk.WriteDescriptorSet,
    copy_count: u32,
    copies: ?[*]const vk.CopyDescriptorSet,
) callconv(vk.vulkan_call_conv) void {
    real.vkUpdateDescriptorSets.?(device, write_count, writes, copy_count, copies);
    const list = recording() orelse return;

    if (copy_count != 0) return list.fail(.descriptor_copies);

    const all = writes orelse return;
    for (all[0..write_count]) |w| {
        if (w.descriptor_type != .storage_buffer) {
            list.fail(.descriptor_type);
            continue;
        }

        for (w.p_buffer_info[0..w.descriptor_count], 0..) |info, j| {
            const index = w.dst_binding + w.dst_array_element + @as(u32, @intCast(j));
            list.writeBinding(
                handleKey(w.dst_set),
                index,
                handleKey(info.buffer),
                info.offset,
                info.range,
            );
        }
    }
}

fn vkAllocateDescriptorSets(
    device: vk.Device,
    info: *const vk.DescriptorSetAllocateInfo,
    sets: [*]vk.DescriptorSet,
) callconv(vk.vulkan_call_conv) vk.Result {
    const result = real.vkAllocateDescriptorSets.?(device, info, sets);
    if (result != .success) return result;

    const list = recording() orelse return result;
    for (sets[0..info.descriptor_set_count]) |set| {
        list.allocateSet(handleKey(set), handleKey(info.descriptor_pool));
    }

    return result;
}

fn vkFreeDescriptorSets(
    device: vk.Device,
    pool: vk.DescriptorPool,
    set_count: u32,
    sets: [*]const vk.DescriptorSet,
) callconv(vk.vulkan_call_conv) vk.Result {
    const result = real.vkFreeDescriptorSets.?(device, pool, set_count, sets);
    if (result != .success) return result;

    const list = recording() orelse return result;
    for (sets[0..set_count]) |set| {
        list.freeSet(handleKey(set));
    }

    return result;
}

fn vkResetDescriptorPool(
    device: vk.Device,
    pool: vk.DescriptorPool,
    reset_flags: vk.DescriptorPoolResetFlags,
) callconv(vk.vulkan_call_conv) vk.Result {
    const result = real.vkResetDescriptorPool.?(device, pool, reset_flags);
    if (result != .success) return result;

    if (recording()) |list| {
        list.clearPool(handleKey(pool));
    }

    return result;
}

fn vkDestroyDescriptorPool(
    device: vk.Device,
    pool: vk.DescriptorPool,
    allocator: ?*const vk.AllocationCallbacks,
) callconv(vk.vulkan_call_conv) void {
    real.vkDestroyDescriptorPool.?(device, pool, allocator);
    if (recording()) |list| {
        list.clearPool(handleKey(pool));
    }
}

fn vkCreateBuffer(
    device: vk.Device,
    info: *const vk.BufferCreateInfo,
    allocator: ?*const vk.AllocationCallbacks,
    buffer: *vk.Buffer,
) callconv(vk.vulkan_call_conv) vk.Result {
    const result = real.vkCreateBuffer.?(device, info, allocator, buffer);
    if (recording()) |list| {
        if (result == .success) {
            list.createBuffer(handleKey(buffer.*), info.size);
        }
    }

    return result;
}

fn vkDestroyBuffer(
    device: vk.Device,
    buffer: vk.Buffer,
    allocator: ?*const vk.AllocationCallbacks,
) callconv(vk.vulkan_call_conv) void {
    real.vkDestroyBuffer.?(device, buffer, allocator);
    if (recording()) |list| {
        list.destroyBuffer(handleKey(buffer));
    }
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

const test_caps: Caps = .{
    .command_buffers = 2,
    .list_entries = 64,
    .section_entries = 128,
    .buffers = 8,
    .sets = 4,
    .modules = 4,
    .pipelines = 4,
};

fn expectLines(list: *DispatchList, expected: []const u8) !void {
    var buffer: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    try list.write(&w);
    try testing.expectEqualStrings(expected, w.buffered());
}

test "every entry kind writes its line, and no handle appears" {
    var list: DispatchList = try .init(testing.allocator, test_caps);
    defer list.deinit();

    // Build the tables: one module, one pipeline, two buffers, one set with two bindings.
    const code = kernels.spv_names[0];
    list.createModule(0x51, code.code);
    list.createPipeline(0x61, 0x51);
    list.createBuffer(0x71, 4096);
    list.createBuffer(0x72, 256);
    list.writeBinding(0x81, 0, 0x71, 0, 4096);
    list.writeBinding(0x81, 1, 0x72, 64, vk.WHOLE_SIZE);

    // Record one of each command, then submit the buffer twice inside one section.
    list.begin(0x91);
    list.bindPipeline(0x91, 0x61);
    list.bindSets(0x91, 0, &.{0x81});
    list.push(0x91, 0, &.{ 0x0a, 0, 0, 0, 0xcd, 0xcc, 0x4c, 0x3e });
    list.dispatch(0x91, .{ 3, 1, 1 });
    const stages: Entry = .{
        .barrier = .{ .src_stage = 0x800, .dst_stage = 0x1800, .dependency = 0 },
    };
    list.barrier(0x91, stages);
    list.barrier(0x91, .{ .memory = .{ .src_access = 0x40, .dst_access = 0x60 } });
    list.copy(0x91, 0x71, 0x99, &.{.{ .src_offset = 8, .dst_offset = 16, .size = 32 }});
    list.fill(0x91, 0, 256, 0);
    list.beginSection("decode");
    list.submit(0x91);
    list.submit(0x92);
    list.endSection();
    try list.finish();

    const name = code.name;
    const expected = "section decode\n" ++
        "submit entries=10\n" ++
        "dispatch " ++ name ++ " groups=3,1,1\n" ++
        "  push offset=0 bytes=0a000000cdcc4c3e\n" ++
        "  set 0 bindings=2\n" ++
        "    binding 0 size=4096 offset=0 range=4096\n" ++
        "    binding 1 size=256 offset=64 range=whole\n" ++
        "barrier src_stage=0x00000800 dst_stage=0x00001800 dependency=0x0\n" ++
        "  memory src_access=0x00000040 dst_access=0x00000060\n" ++
        "copy regions=1 src_size=4096 dst_size=unknown\n" ++
        "  region size=32 src_offset=8 dst_offset=16\n" ++
        "fill offset=0 size=256 value=0x00000000\n" ++
        "submit unrecorded\n";
    try expectLines(&list, expected);
    try expectLines(&list, "");
}

test "a list outlives its submit, and a begin clears it" {
    var list: DispatchList = try .init(testing.allocator, test_caps);
    defer list.deinit();

    list.begin(0x91);
    list.fill(0x91, 0, 4, 1);
    list.beginSection("a");
    list.submit(0x91);
    list.submit(0x91);
    list.endSection();

    list.begin(0x91);
    list.beginSection("b");
    list.submit(0x91);
    list.endSection();
    list.submit(0x91);

    const fill = "fill offset=0 size=4 value=0x00000001\n";
    const expected = "section a\nsubmit entries=1\n" ++ fill ++ "submit entries=1\n" ++ fill ++
        "section b\nsubmit entries=0\n";
    try expectLines(&list, expected);
}

test "a full table sets the flag, and finish and write return the error" {
    var list: DispatchList = try .init(testing.allocator, test_caps);
    defer list.deinit();

    list.begin(0x91);
    for (0..test_caps.list_entries) |_| {
        list.fill(0x91, 0, 4, 0);
    }

    try list.finish();
    list.fill(0x91, 0, 4, 0);
    try testing.expectError(error.DispatchListFull, list.finish());

    var buffer: [16]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    try testing.expectError(error.DispatchListFull, list.write(&w));

    // A third command buffer finds no list, and its commands keep the flag.
    var other: DispatchList = try .init(testing.allocator, test_caps);
    defer other.deinit();
    other.begin(1);
    other.begin(2);
    try other.finish();
    other.begin(3);
    other.fill(3, 0, 4, 0);
    try testing.expectError(error.DispatchListFull, other.finish());
}

/// A list with one module, one pipeline, one buffer, and one begun command buffer.
fn knownList() !DispatchList {
    var list: DispatchList = try .init(testing.allocator, test_caps);
    list.createModule(0x51, kernels.spv_names[0].code);
    list.createPipeline(0x61, 0x51);
    list.createBuffer(0x71, 4096);
    list.begin(0x91);
    return list;
}

test "a dispatch of an unknown shader gives the missed error" {
    var list = try knownList();
    defer list.deinit();

    list.bindPipeline(0x91, 0x61);
    list.dispatch(0x91, .{ 1, 1, 1 });
    try list.finish();

    list.bindPipeline(0x91, 0x62);
    list.dispatch(0x91, .{ 1, 1, 1 });
    try testing.expectError(error.DispatchListMissed, list.finish());
    try testing.expectEqual(Fault.unknown_shader, list.fault.?);
}

test "a bound set or a binding buffer that no table knows gives the missed error" {
    var unknown_set = try knownList();
    defer unknown_set.deinit();

    unknown_set.bindPipeline(0x91, 0x61);
    unknown_set.bindSets(0x91, 0, &.{0x81});
    unknown_set.dispatch(0x91, .{ 1, 1, 1 });
    try testing.expectError(error.DispatchListMissed, unknown_set.finish());
    try testing.expectEqual(Fault.unknown_set, unknown_set.fault.?);

    var unknown_buffer = try knownList();
    defer unknown_buffer.deinit();

    unknown_buffer.writeBinding(0x81, 0, 0x72, 0, 16);
    try testing.expectError(error.DispatchListMissed, unknown_buffer.finish());
    try testing.expectEqual(Fault.unknown_buffer, unknown_buffer.fault.?);
}

test "a command that the list cannot state gives the unsupported error, and the first stays" {
    var list = try knownList();
    defer list.deinit();

    list.fail(.dynamic_offsets);
    list.fail(.unknown_buffer);
    try testing.expectError(error.DispatchListUnsupported, list.finish());
    try testing.expectEqual(Fault.dynamic_offsets, list.fault.?);

    var null_handle = try knownList();
    defer null_handle.deinit();

    null_handle.createBuffer(0, 4);
    try testing.expectError(error.DispatchListUnsupported, null_handle.finish());
    try testing.expectEqual(Fault.null_handle, null_handle.fault.?);
}

test "an allocate, a free, and a pool clear drop the bindings of an old set" {
    var list = try knownList();
    defer list.deinit();

    list.bindPipeline(0x91, 0x61);
    list.bindSets(0x91, 0, &.{0x81});
    list.writeBinding(0x81, 0, 0x71, 0, 16);
    list.allocateSet(0x81, 0x41);
    try testing.expectEqual(@as(u16, 0), list.sets.get(0x81).?.written);

    list.writeBinding(0x81, 0, 0x71, 0, 16);
    list.writeBinding(0x82, 0, 0x71, 0, 16);
    list.freeSet(0x82);
    try testing.expect(list.sets.get(0x82) == null);

    list.allocateSet(0x83, 0x41);
    list.allocateSet(0x84, 0x42);
    list.clearPool(0x41);
    try testing.expect(list.sets.get(0x81) == null);
    try testing.expect(list.sets.get(0x83) == null);
    try testing.expect(list.sets.get(0x84) != null);
    try list.finish();
}

test "a binding keeps the buffer size of its update after the buffer goes away" {
    var list = try knownList();
    defer list.deinit();

    list.bindPipeline(0x91, 0x61);
    list.bindSets(0x91, 0, &.{0x81});
    list.writeBinding(0x81, 0, 0x71, 0, 16);
    list.destroyBuffer(0x71);
    list.dispatch(0x91, .{ 1, 1, 1 });
    list.beginSection("a");
    list.submit(0x91);
    list.endSection();

    var buffer: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    try list.write(&w);
    const line = "    binding 0 size=4096 offset=0 range=16\n";
    try testing.expect(std.mem.indexOf(u8, w.buffered(), line) != null);
}

test "a known blob address gives its name, and an unknown address gives unknown" {
    try testing.expect(kernels.spv_names.len > 0);

    for (kernels.spv_names) |spv_name| {
        try testing.expectEqualStrings(spv_name.name, shaderName(spv_name.code));
        try testing.expect(!std.mem.eql(u8, spv_name.name, "unknown"));
    }

    const elsewhere: u32 = 0;
    try testing.expectEqualStrings("unknown", shaderName(&elsewhere));
}

test "the default capacities stay inside the memory bound" {
    try testing.expect(Caps.default.bytes() <= memory_bytes_max);
    try testing.expect(test_caps.bytes() < Caps.default.bytes());
}
