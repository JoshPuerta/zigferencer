//! The Perfetto protobuf encoder of the span trace, and the nesting walk that turns the intervals
//! of one track into balanced slice events. Datapath, std only.
//!
//! Comptime. Asserts check that the worst packet of each kind fits `packet_bytes_max`. `Nest` takes
//! the interval type at comptime.
//!
//! Wire. A trace file is a run of `Trace.packet` fields (field 1, length-delimited). A key is
//! `(field << 3) | wire_type`. Varint fields use LEB128, and a flow id is a little-endian
//! `fixed64`. Each packet is built in a stack buffer of `packet_bytes_max` bytes, and the encoder
//! asserts that bound before each write. Then the writer emits the key, the length, and the bytes.
//!
//! Lengths. A nested message reserves two length bytes, because no packet reaches 2^14 bytes. When
//! the body ends under 128 bytes, the encoder moves the body back by one byte.
//!
//! Sequence. The writer emits the clock snapshot, then the header with the interned names, then
//! the descriptors and the events, all on one sequence. Interned state accumulates on a sequence,
//! so the header splits the names over as many packets as the bound needs.
//!
//! Nesting. `Nest` is the parenthesis match on intervals sorted by start, the longer first on a
//! tie. Before it opens an item, it closes each open item that ends at or before the start. An
//! item that ends past the innermost open item, or that finds the stack full, is skipped and
//! counted in `misnested`, so the event stream stays balanced. Each pass opens or skips one item,
//! or closes one, so a walk takes at most `2 * items.len` steps.

const std = @import("std");

const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

pub const clock_monotonic: u32 = 3;
pub const clock_boottime: u32 = 6;

/// The bound of one encoded packet, without the `Trace.packet` key and length.
pub const packet_bytes_max: usize = 1024;

/// The deepest stack of open slices that `Nest` holds on one track.
pub const nest_depth_max: u32 = 64;

/// The longest name that a descriptor or an interned entry accepts.
pub const name_bytes_max: usize = 128;

const varint_bytes_max = 10;

/// A nested message reserves this many length bytes, which hold any length under 2^14.
const length_reserve = 2;

const seq_cleared: u32 = 1;
const seq_needs_state: u32 = 2;

/// The largest packet of fixed fields before the interned names: sequence, flags, defaults.
const header_fixed_bytes = 32;

comptime {
    assert(packet_bytes_max < 1 << (7 * length_reserve));
    assert(header_fixed_bytes + internBytesMax(name_bytes_max) <= packet_bytes_max);
    assert(descriptorBytesMax() <= packet_bytes_max);
}

// -- types ------------------------------------------------------------------

/// The field numbers, read from the v59.1 protos.
const field = struct {
    const trace_packet = 1;

    const timestamp = 8;
    const sequence_id = 10;
    const track_event = 11;
    const interned_data = 12;
    const sequence_flags = 13;
    const clock_snapshot = 6;
    const packet_defaults = 59;
    const track_descriptor = 60;

    const snapshot_clocks = 1;
    const snapshot_primary = 2;
    const clock_id = 1;
    const clock_timestamp = 2;

    const defaults_clock = 58;

    const interned_event_names = 2;
    const interned_arg_names = 3;
    const interned_iid = 1;
    const interned_name = 2;

    const track_uuid = 1;
    const track_name = 2;
    const track_process = 3;
    const track_thread = 4;
    const track_parent = 5;
    const process_pid = 1;
    const process_name = 6;
    const thread_pid = 1;
    const thread_tid = 2;
    const thread_name = 5;

    const event_args = 4;
    const event_type = 9;
    const event_name_iid = 10;
    const event_track = 11;
    const event_flows = 47;
    const event_terminating_flows = 48;
    const arg_name_iid = 1;
    const arg_uint = 3;
};

pub const EventType = enum(u8) { slice_begin = 1, slice_end = 2, instant = 3 };

/// One debug annotation: an interned name and an unsigned value.
pub const Arg = struct { name_iid: u64, value: u64 };

/// One track event. A `slice_end` carries no name (`name_iid == 0`).
pub const Event = struct {
    ts_ns: u64,
    track_uuid: u64,
    type: EventType,
    name_iid: u64 = 0,
    args: []const Arg = &.{},
    flow_ids: []const u64 = &.{},
    terminating_flow_ids: []const u64 = &.{},
};

pub const Writer = struct {
    out: *std.Io.Writer,
    seq: u32,
    stage: Stage = .snapshot,

    /// The packet order of one sequence.
    const Stage = enum { snapshot, header, body };

    pub fn init(out: *std.Io.Writer, seq: u32) Writer {
        assert(seq != 0);

        return .{ .out = out, .seq = seq };
    }

    /// TracePacket.clock_snapshot: MONOTONIC and BOOTTIME read back to back, primary MONOTONIC.
    /// Without BOOTTIME, trace_processor cannot relate the MONOTONIC domain and drops every event.
    pub fn clockSnapshot(self: *Writer, monotonic_ns: u64, boottime_ns: u64) !void {
        assert(self.stage == .snapshot);

        var enc: Enc = .{};
        enc.uint(field.sequence_id, self.seq);
        const snap = enc.open(field.clock_snapshot);
        const clocks = [_][2]u64{
            .{ clock_monotonic, monotonic_ns },
            .{ clock_boottime, boottime_ns },
        };
        for (clocks) |c| {
            const clock = enc.open(field.snapshot_clocks);
            enc.uint(field.clock_id, c[0]);
            enc.uint(field.clock_timestamp, c[1]);
            enc.close(clock);
        }

        enc.uint(field.snapshot_primary, clock_monotonic);
        enc.close(snap);
        try self.emit(&enc);
        self.stage = .header;
    }

    /// SEQ_INCREMENTAL_STATE_CLEARED, the MONOTONIC clock default, and the interned event names
    /// and debug annotation names (iid = index + 1). A packet that fills up continues in the next.
    pub fn header(
        self: *Writer,
        event_names: []const []const u8,
        arg_names: []const []const u8,
    ) !void {
        assert(self.stage == .header);

        // The first packet clears the state and sets the clock default.
        var enc: Enc = .{};
        enc.uint(field.sequence_id, self.seq);
        enc.uint(field.sequence_flags, seq_cleared);
        const defaults = enc.open(field.packet_defaults);
        enc.uint(field.defaults_clock, clock_monotonic);
        enc.close(defaults);
        assert(enc.len <= header_fixed_bytes);
        var interned = enc.open(field.interned_data);

        // Each entry goes into the open packet, or into a new one when it does not fit.
        const lists = [_]struct { u32, []const []const u8 }{
            .{ field.interned_event_names, event_names },
            .{ field.interned_arg_names, arg_names },
        };
        for (lists) |list| {
            for (list[1], 1..) |name, iid| {
                assert(name.len <= name_bytes_max);

                if (enc.len + internBytesMax(name.len) > packet_bytes_max) {
                    enc.close(interned);
                    try self.emit(&enc);
                    enc = .{};
                    enc.uint(field.sequence_id, self.seq);
                    enc.uint(field.sequence_flags, seq_needs_state);
                    interned = enc.open(field.interned_data);
                }

                const entry = enc.open(list[0]);
                enc.uint(field.interned_iid, iid);
                enc.string(field.interned_name, name);
                enc.close(entry);
            }
        }

        enc.close(interned);
        try self.emit(&enc);
        self.stage = .body;
    }

    pub fn process(self: *Writer, uuid: u64, pid: u32, name: []const u8) !void {
        var enc = self.descriptorStart(name);
        const desc = enc.open(field.track_descriptor);
        enc.uint(field.track_uuid, uuid);
        const proc = enc.open(field.track_process);
        enc.uint(field.process_pid, pid);
        enc.string(field.process_name, name);
        enc.close(proc);
        enc.close(desc);
        try self.emit(&enc);
    }

    pub fn thread(
        self: *Writer,
        uuid: u64,
        parent_uuid: u64,
        pid: u32,
        tid: u64,
        name: []const u8,
    ) !void {
        assert(uuid != parent_uuid);

        var enc = self.descriptorStart(name);
        const desc = enc.open(field.track_descriptor);
        enc.uint(field.track_uuid, uuid);
        enc.uint(field.track_parent, parent_uuid);
        const thr = enc.open(field.track_thread);
        enc.uint(field.thread_pid, pid);
        enc.uint(field.thread_tid, tid);
        enc.string(field.thread_name, name);
        enc.close(thr);
        enc.close(desc);
        try self.emit(&enc);
    }

    pub fn track(self: *Writer, uuid: u64, parent_uuid: u64, name: []const u8) !void {
        assert(uuid != parent_uuid);

        var enc = self.descriptorStart(name);
        const desc = enc.open(field.track_descriptor);
        enc.uint(field.track_uuid, uuid);
        enc.string(field.track_name, name);
        enc.uint(field.track_parent, parent_uuid);
        enc.close(desc);
        try self.emit(&enc);
    }

    /// One track event on SEQ_NEEDS_INCREMENTAL_STATE. Its size bound must fit the packet.
    pub fn event(self: *Writer, e: Event) !void {
        assert(self.stage == .body);
        assert((e.type == .slice_end) == (e.name_iid == 0));
        const flows = e.flow_ids.len + e.terminating_flow_ids.len;
        assert(eventBytesMax(e.args.len, flows) <= packet_bytes_max);

        var enc: Enc = .{};
        enc.uint(field.timestamp, e.ts_ns);
        enc.uint(field.sequence_id, self.seq);
        enc.uint(field.sequence_flags, seq_needs_state);
        const te = enc.open(field.track_event);
        enc.uint(field.event_type, @intFromEnum(e.type));
        enc.uint(field.event_track, e.track_uuid);
        if (e.name_iid != 0) {
            enc.uint(field.event_name_iid, e.name_iid);
        }

        for (e.args) |a| {
            assert(a.name_iid != 0);
            const arg = enc.open(field.event_args);
            enc.uint(field.arg_name_iid, a.name_iid);
            enc.uint(field.arg_uint, a.value);
            enc.close(arg);
        }

        for (e.flow_ids) |id| {
            enc.fixed64(field.event_flows, id);
        }

        for (e.terminating_flow_ids) |id| {
            enc.fixed64(field.event_terminating_flows, id);
        }

        enc.close(te);
        try self.emit(&enc);
    }

    fn descriptorStart(self: *Writer, name: []const u8) Enc {
        assert(self.stage == .body);
        assert(name.len <= name_bytes_max);

        var enc: Enc = .{};
        enc.uint(field.sequence_id, self.seq);
        enc.uint(field.sequence_flags, seq_needs_state);
        return enc;
    }

    fn emit(self: *Writer, enc: *const Enc) std.Io.Writer.Error!void {
        assert(enc.len > 0);
        assert(enc.len <= packet_bytes_max);

        var head: [1 + varint_bytes_max]u8 = undefined;
        head[0] = @intCast(keyOf(field.trace_packet, .len));
        const n = putVarint(head[1..], enc.len);
        try self.out.writeAll(head[0 .. 1 + n]);
        try self.out.writeAll(enc.buf[0..enc.len]);
    }
};

/// The nesting walk over the intervals of one track. `T` has `start_ns: u64` and `end_ns: u64`,
/// and the items are sorted by start, the longer first on a tie.
pub fn Nest(comptime T: type) type {
    return struct {
        items: []const T,
        cursor: u32 = 0,
        stack: [nest_depth_max]u32 = undefined,
        depth: u32 = 0,
        last_begin: ?u32 = null,
        /// The items skipped for a partial overlap or a full stack.
        misnested: u32 = 0,

        const Self = @This();

        pub const Kind = enum { begin, end };
        pub const Step = struct { kind: Kind, index: u32 };

        pub fn init(items: []const T) Self {
            assert(items.len < std.math.maxInt(u32));

            return .{ .items = items };
        }

        pub fn next(self: *Self) ?Step {
            // Each pass returns or skips one item, so the loop ends within `items.len + 1` passes.
            const n: u32 = @intCast(self.items.len);
            var passes: u32 = 0;
            while (passes <= n) : (passes += 1) {
                if (self.cursor == n) return self.pop();

                const item = self.items[self.cursor];
                assert(item.start_ns <= item.end_ns);
                if (self.cursor > 0) assert(!before(item, self.items[self.cursor - 1]));

                if (self.depth > 0) {
                    const top = self.items[self.stack[self.depth - 1]];
                    if (top.end_ns <= item.start_ns) return self.pop();
                    if (item.end_ns > top.end_ns) {
                        self.skip();
                        continue;
                    }
                }

                if (self.depth == nest_depth_max) {
                    self.skip();
                    continue;
                }

                return self.push();
            }

            unreachable;
        }

        fn push(self: *Self) Step {
            assert(self.depth < nest_depth_max);
            const i = self.cursor;
            if (self.last_begin) |last| assert(i > last);

            self.stack[self.depth] = i;
            self.depth += 1;
            self.cursor += 1;
            self.last_begin = i;
            return .{ .kind = .begin, .index = i };
        }

        fn pop(self: *Self) ?Step {
            if (self.depth == 0) return null;

            self.depth -= 1;
            const i = self.stack[self.depth];
            assert(i < self.cursor);
            return .{ .kind = .end, .index = i };
        }

        fn skip(self: *Self) void {
            self.misnested += 1;
            self.cursor += 1;
        }

        fn before(a: T, b: T) bool {
            if (a.start_ns != b.start_ns) return a.start_ns < b.start_ns;
            return a.end_ns > b.end_ns;
        }
    };
}

const Wire = enum(u3) { varint = 0, fixed64 = 1, len = 2 };

/// One packet under construction, in a fixed buffer.
const Enc = struct {
    buf: [packet_bytes_max]u8 = undefined,
    len: usize = 0,

    fn bytes(self: *Enc, b: []const u8) void {
        assert(self.len + b.len <= packet_bytes_max);

        @memcpy(self.buf[self.len..][0..b.len], b);
        self.len += b.len;
    }

    fn varint(self: *Enc, v: u64) void {
        var tmp: [varint_bytes_max]u8 = undefined;
        self.bytes(tmp[0..putVarint(&tmp, v)]);
    }

    fn uint(self: *Enc, num: u32, v: u64) void {
        self.varint(keyOf(num, .varint));
        self.varint(v);
    }

    fn fixed64(self: *Enc, num: u32, v: u64) void {
        self.varint(keyOf(num, .fixed64));
        var tmp: [8]u8 = undefined;
        std.mem.writeInt(u64, &tmp, v, .little);
        self.bytes(&tmp);
    }

    fn string(self: *Enc, num: u32, s: []const u8) void {
        self.varint(keyOf(num, .len));
        self.varint(s.len);
        self.bytes(s);
    }

    /// Open a nested message: the key, then two reserved length bytes. Returns the body start.
    fn open(self: *Enc, num: u32) usize {
        self.varint(keyOf(num, .len));
        self.bytes(&@as([length_reserve]u8, @splat(0)));
        return self.len;
    }

    fn close(self: *Enc, body: usize) void {
        assert(body >= length_reserve);
        assert(body <= self.len);

        const n = self.len - body;
        assert(n < 1 << (7 * length_reserve));
        if (n < 0x80) {
            self.buf[body - 2] = @intCast(n);
            std.mem.copyForwards(u8, self.buf[body - 1 ..][0..n], self.buf[body..][0..n]);
            self.len -= 1;
        } else {
            self.buf[body - 2] = @as(u8, @truncate(n)) | 0x80;
            self.buf[body - 1] = @intCast(n >> 7);
        }
    }
};

// -- api --------------------------------------------------------------------

/// Write `v` as LEB128 into `out` and return the byte count, at most 10.
pub fn putVarint(out: []u8, v: u64) usize {
    assert(out.len >= varintLen(v));

    var x = v;
    var n: usize = 0;
    while (x >= 0x80) : (x >>= 7) {
        out[n] = @as(u8, @truncate(x)) | 0x80;
        n += 1;
    }

    out[n] = @intCast(x);
    n += 1;
    assert(n <= varint_bytes_max);
    return n;
}

pub fn varintLen(v: u64) usize {
    const bits: usize = 64 - @clz(v | 1);
    return (bits + 6) / 7;
}

/// The peak bytes of one event packet with `args` annotations and `flows` flow ids.
pub fn eventBytesMax(args: usize, flows: usize) usize {
    // Timestamp, sequence, flags, the track event key and length, type, track, name.
    const fixed = (1 + 10) + (1 + 5) + (1 + 1) + (1 + length_reserve) + (1 + 1) +
        (1 + 10) + (1 + 10);
    const arg = 1 + length_reserve + (1 + 10) + (1 + 10);
    const flow = 2 + 8;
    return fixed + args * arg + flows * flow;
}

// -- helpers ----------------------------------------------------------------

fn keyOf(num: u32, wire: Wire) u64 {
    assert(num != 0);

    return (@as(u64, num) << 3) | @intFromEnum(wire);
}

/// The peak bytes of one interned entry: key, reserve, iid, and the name with its length.
fn internBytesMax(name_len: usize) usize {
    return 1 + length_reserve + (1 + 10) + (1 + 2 + name_len);
}

/// The peak bytes of the largest descriptor: a thread with its ids and a longest name.
fn descriptorBytesMax() usize {
    return (1 + 5) + (1 + 1) + (2 + length_reserve) + (1 + 10) + (1 + 10) +
        (1 + length_reserve) + (1 + 5) + (1 + 10) + (1 + 2 + name_bytes_max);
}

// -- tests ------------------------------------------------------------------

const prop = @import("prop.zig");
const testing = std.testing;

/// The bounds of the test decoder.
pub const decode_args_max = 32;
pub const decode_flows_max = 64;
pub const interned_max = 256;

/// The names that the header packets intern, by iid.
pub const TestInterned = struct {
    event_names: [interned_max][]const u8 = @splat(""),
    arg_names: [interned_max][]const u8 = @splat(""),
};

pub const TestEvent = struct {
    type: u64 = 0,
    track_uuid: u64 = 0,
    name_iid: u64 = 0,
    args: [decode_args_max]Arg = undefined,
    n_args: u32 = 0,
    flows: [decode_flows_max]u64 = undefined,
    n_flows: u32 = 0,
    terms: [decode_flows_max]u64 = undefined,
    n_terms: u32 = 0,
};

pub const TestTrack = struct {
    uuid: u64 = 0,
    parent_uuid: u64 = 0,
    name: []const u8 = "",
    pid: ?u64 = null,
    tid: ?u64 = null,
};

/// The fields of one packet that the tests read.
pub const TestPacket = struct {
    bytes: usize = 0,
    timestamp: ?u64 = null,
    seq: u64 = 0,
    flags: u64 = 0,
    defaults_clock: u64 = 0,
    clocks: [2]u64 = @splat(0),
    n_clocks: u32 = 0,
    primary_clock: u64 = 0,
    event: ?TestEvent = null,
    track: ?TestTrack = null,
};

const TestField = struct { num: u64, wire: u64, int: u64 = 0, bytes: []const u8 = "" };

/// A walk over the fields of one message.
const TestFields = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn varint(self: *TestFields) !u64 {
        var v: u64 = 0;
        var i: u32 = 0;
        while (i < varint_bytes_max) : (i += 1) {
            if (self.pos >= self.bytes.len) return error.Malformed;
            const b = self.bytes[self.pos];
            self.pos += 1;
            v |= @as(u64, b & 0x7f) << @intCast(7 * i);
            if (b < 0x80) return v;
        }

        return error.Malformed;
    }

    fn next(self: *TestFields) !?TestField {
        if (self.pos == self.bytes.len) return null;

        const k = try self.varint();
        var f: TestField = .{ .num = k >> 3, .wire = k & 7 };
        switch (f.wire) {
            0 => f.int = try self.varint(),
            1 => {
                if (self.pos + 8 > self.bytes.len) return error.Malformed;
                f.int = std.mem.readInt(u64, self.bytes[self.pos..][0..8], .little);
                self.pos += 8;
            },
            2 => {
                const n = try self.varint();
                if (n > self.bytes.len - self.pos) return error.Malformed;
                f.bytes = self.bytes[self.pos..][0..@intCast(n)];
                self.pos += @intCast(n);
            },
            else => return error.Malformed,
        }

        return f;
    }
};

/// A walk over the `Trace.packet` fields of a trace file.
pub const TestReader = struct {
    fields: TestFields,

    pub fn init(bytes: []const u8) TestReader {
        return .{ .fields = .{ .bytes = bytes } };
    }

    pub fn next(self: *TestReader, interned: *TestInterned) !?TestPacket {
        const f = try self.fields.next() orelse return null;
        if (f.num != field.trace_packet or f.wire != 2) return error.Malformed;

        return try decodePacket(f.bytes, interned);
    }
};

fn decodePacket(bytes: []const u8, interned: *TestInterned) !TestPacket {
    var p: TestPacket = .{ .bytes = bytes.len };
    var it: TestFields = .{ .bytes = bytes };
    while (try it.next()) |f| {
        switch (f.num) {
            field.timestamp => p.timestamp = f.int,
            field.sequence_id => p.seq = f.int,
            field.sequence_flags => p.flags = f.int,
            field.clock_snapshot => try decodeSnapshot(f.bytes, &p),
            field.packet_defaults => {
                var d: TestFields = .{ .bytes = f.bytes };
                while (try d.next()) |g| {
                    if (g.num == field.defaults_clock) {
                        p.defaults_clock = g.int;
                    }
                }
            },
            field.interned_data => try decodeInterned(f.bytes, interned),
            field.track_event => p.event = try decodeEvent(f.bytes),
            field.track_descriptor => p.track = try decodeTrack(f.bytes),
            else => return error.UnknownField,
        }
    }

    return p;
}

fn decodeSnapshot(bytes: []const u8, p: *TestPacket) !void {
    var it: TestFields = .{ .bytes = bytes };
    while (try it.next()) |f| {
        if (f.num == field.snapshot_primary) {
            p.primary_clock = f.int;
            continue;
        }

        var c: TestFields = .{ .bytes = f.bytes };
        while (try c.next()) |g| {
            if (g.num == field.clock_id and p.n_clocks < p.clocks.len) {
                p.clocks[p.n_clocks] = g.int;
                p.n_clocks += 1;
            }
        }
    }
}

fn decodeInterned(bytes: []const u8, interned: *TestInterned) !void {
    var it: TestFields = .{ .bytes = bytes };
    while (try it.next()) |f| {
        var iid: u64 = 0;
        var name: []const u8 = "";
        var e: TestFields = .{ .bytes = f.bytes };
        while (try e.next()) |g| {
            if (g.num == field.interned_iid) {
                iid = g.int;
            }
            if (g.num == field.interned_name) {
                name = g.bytes;
            }
        }

        if (iid == 0 or iid >= interned_max) return error.Malformed;
        const table = switch (f.num) {
            field.interned_event_names => &interned.event_names,
            field.interned_arg_names => &interned.arg_names,
            else => return error.UnknownField,
        };
        table[@intCast(iid)] = name;
    }
}

fn decodeEvent(bytes: []const u8) !TestEvent {
    var e: TestEvent = .{};
    var it: TestFields = .{ .bytes = bytes };
    while (try it.next()) |f| {
        switch (f.num) {
            field.event_type => e.type = f.int,
            field.event_track => e.track_uuid = f.int,
            field.event_name_iid => e.name_iid = f.int,
            field.event_args => {
                if (e.n_args == decode_args_max) return error.Malformed;
                var a: Arg = .{ .name_iid = 0, .value = 0 };
                var g: TestFields = .{ .bytes = f.bytes };
                while (try g.next()) |h| {
                    if (h.num == field.arg_name_iid) {
                        a.name_iid = h.int;
                    }
                    if (h.num == field.arg_uint) {
                        a.value = h.int;
                    }
                }
                e.args[e.n_args] = a;
                e.n_args += 1;
            },
            field.event_flows => {
                if (f.wire != 1 or e.n_flows == decode_flows_max) return error.Malformed;
                e.flows[e.n_flows] = f.int;
                e.n_flows += 1;
            },
            field.event_terminating_flows => {
                if (f.wire != 1 or e.n_terms == decode_flows_max) return error.Malformed;
                e.terms[e.n_terms] = f.int;
                e.n_terms += 1;
            },
            else => return error.UnknownField,
        }
    }

    return e;
}

fn decodeTrack(bytes: []const u8) !TestTrack {
    var t: TestTrack = .{};
    var it: TestFields = .{ .bytes = bytes };
    while (try it.next()) |f| {
        switch (f.num) {
            field.track_uuid => t.uuid = f.int,
            field.track_name => t.name = f.bytes,
            field.track_parent => t.parent_uuid = f.int,
            field.track_process, field.track_thread => {
                var g: TestFields = .{ .bytes = f.bytes };
                while (try g.next()) |h| {
                    // The pid has field 1 in both descriptors.
                    if (h.num == field.process_pid) {
                        t.pid = h.int;
                    }
                    if (f.num == field.track_thread and h.num == field.thread_tid) {
                        t.tid = h.int;
                    }
                    if (f.num == field.track_thread and h.num == field.thread_name) {
                        t.name = h.bytes;
                    }
                    if (f.num == field.track_process and h.num == field.process_name) {
                        t.name = h.bytes;
                    }
                }
            },
            else => return error.UnknownField,
        }
    }

    return t;
}

fn genVarint(_: void, random: std.Random, size: u32) u64 {
    // Mask to a random width, so the short encodings get cases too.
    const bits: u6 = @intCast(random.uintLessThan(u32, 64));
    _ = size;
    return random.int(u64) >> bits;
}

fn checkVarint(_: void, x: u64) anyerror!void {
    var buf: [varint_bytes_max]u8 = undefined;
    const n = putVarint(&buf, x);
    try testing.expect(n <= varint_bytes_max);
    try testing.expectEqual(varintLen(x), n);
    var it: TestFields = .{ .bytes = buf[0..n] };
    try testing.expectEqual(x, try it.varint());
    try testing.expectEqual(n, it.pos);
}

test "varint: decode of encode is the identity, in at most 10 bytes" {
    try prop.forAll(.{ .site = 0x5046_0001 }, {}, u64, genVarint, checkVarint);
    try checkVarint({}, 0);
    try checkVarint({}, std.math.maxInt(u64));
}

const Interval = struct { start_ns: u64, end_ns: u64 };

const nest_items_max = 96;

const NestCase = struct {
    items: [nest_items_max]Interval,
    n: u32,
};

fn intervalLess(_: void, a: Interval, b: Interval) bool {
    if (a.start_ns != b.start_ns) return a.start_ns < b.start_ns;
    return a.end_ns > b.end_ns;
}

/// A random tree as a parenthesis string: each step opens a child of the innermost open interval
/// or closes it. Zero-width steps make ties and empty intervals.
fn genTree(_: void, random: std.Random, size: u32) NestCase {
    var c: NestCase = .{ .items = undefined, .n = 0 };
    var open: [nest_depth_max]u32 = undefined;
    var depth: u32 = 0;
    var t: u64 = 1 + random.uintLessThan(u64, 100);
    const want = @min(size + 1, nest_items_max);
    var steps: u32 = 0;
    while (steps < 2 * nest_items_max) : (steps += 1) {
        if (c.n == want and depth == 0) break;

        t += random.uintLessThan(u64, 3);
        const room = c.n < want and depth < nest_depth_max - 1;
        const opens = room and (depth == 0 or random.boolean());
        if (opens) {
            c.items[c.n] = .{ .start_ns = t, .end_ns = 0 };
            open[depth] = c.n;
            depth += 1;
            c.n += 1;
        } else {
            depth -= 1;
            c.items[open[depth]].end_ns = t;
        }
    }

    assert(depth == 0);
    std.mem.sort(Interval, c.items[0..c.n], {}, intervalLess);
    return c;
}

/// Random intervals with any overlap.
fn genAny(_: void, random: std.Random, size: u32) NestCase {
    var c: NestCase = .{ .items = undefined, .n = @min(size + 1, nest_items_max) };
    for (c.items[0..c.n]) |*it| {
        const s = 1 + random.uintLessThan(u64, 200);
        it.* = .{ .start_ns = s, .end_ns = s + random.uintLessThan(u64, 80) };
    }

    std.mem.sort(Interval, c.items[0..c.n], {}, intervalLess);
    return c;
}

/// What a walk did: the parent of each emitted item, and the counts.
const Walk = struct {
    parent: [nest_items_max]?u32 = @splat(null),
    begun: [nest_items_max]u32 = @splat(0),
    ended: [nest_items_max]u32 = @splat(0),
    emitted: u32 = 0,
    misnested: u32 = 0,
};

/// Walk the case and check that the step stream is balanced and that ends come in LIFO order.
fn walk(items: []const Interval) !Walk {
    var w: Walk = .{};
    var stack: [nest_depth_max]u32 = undefined;
    var depth: u32 = 0;
    var nest: Nest(Interval) = .init(items);
    while (nest.next()) |step| {
        switch (step.kind) {
            .begin => {
                w.parent[step.index] = if (depth > 0) stack[depth - 1] else null;
                w.begun[step.index] += 1;
                stack[depth] = step.index;
                depth += 1;
                w.emitted += 1;
            },
            .end => {
                try testing.expect(depth > 0);
                try testing.expectEqual(stack[depth - 1], step.index);
                depth -= 1;
                w.ended[step.index] += 1;
            },
        }
    }

    try testing.expectEqual(@as(u32, 0), depth);
    w.misnested = nest.misnested;
    return w;
}

fn contains(outer: Interval, inner: Interval) bool {
    return outer.start_ns <= inner.start_ns and inner.end_ns <= outer.end_ns;
}

fn checkTree(_: void, c: NestCase) anyerror!void {
    const items = c.items[0..c.n];
    const w = try walk(items);
    try testing.expectEqual(@as(u32, 0), w.misnested);
    for (items, 0..) |_, i| {
        try testing.expectEqual(@as(u32, 1), w.begun[i]);
        try testing.expectEqual(@as(u32, 1), w.ended[i]);
    }
}

fn checkAny(_: void, c: NestCase) anyerror!void {
    const items = c.items[0..c.n];
    const w = try walk(items);
    try testing.expectEqual(c.n, w.emitted + w.misnested);
    for (items, 0..) |it, i| {
        try testing.expectEqual(w.begun[i], w.ended[i]);
        try testing.expect(w.begun[i] <= 1);
        const p = w.parent[i] orelse continue;
        try testing.expect(contains(items[p], it));
    }
}

test "nest: a well-nested set walks with no misnested item, each item once" {
    const opts: prop.Options = .{ .site = 0x5046_0002, .max_size = nest_items_max - 1 };
    try prop.forAll(opts, {}, NestCase, genTree, checkTree);
}

test "nest: any overlap walks balanced, each item inside its parent, and every item counted" {
    const opts: prop.Options = .{ .site = 0x5046_0003, .max_size = nest_items_max - 1 };
    try prop.forAll(opts, {}, NestCase, genAny, checkAny);
}

test "nest: a chain past the depth bound skips each item past it" {
    var items: [nest_depth_max + 6]Interval = undefined;
    for (&items, 0..) |*it, i| {
        it.* = .{ .start_ns = 10 + i, .end_ns = 1000 - i };
    }

    const w = try walk(&items);
    try testing.expectEqual(@as(u32, 6), w.misnested);
    try testing.expectEqual(nest_depth_max, w.emitted);
}

test "writer: the packets decode, the names resolve, and each packet keeps its bound" {
    var names: [interned_max - 1][]const u8 = undefined;
    var name_bufs: [interned_max - 1][24]u8 = undefined;
    for (&names, &name_bufs, 0..) |*n, *b, i| {
        n.* = std.fmt.bufPrint(b, "event_name_number_{d}", .{i}) catch unreachable;
    }

    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();

    var w: Writer = .init(&out.writer, 1);
    try w.clockSnapshot(1_000, 1_005);
    try w.header(&names, &.{ "layer", "epoch" });
    try w.process(42, 42, "proc");
    try w.thread(7, 42, 42, 43, "main");
    try w.track(9, 42, "GPU queue x");
    try w.event(.{ .ts_ns = 2_000, .track_uuid = 7, .type = .slice_begin, .name_iid = 3 });
    try w.event(.{
        .ts_ns = 2_100,
        .track_uuid = 7,
        .type = .instant,
        .name_iid = interned_max - 1,
        .args = &.{.{ .name_iid = 2, .value = 99 }},
        .flow_ids = &.{ 5, 6 },
        .terminating_flow_ids = &.{7},
    });
    try w.event(.{ .ts_ns = 2_200, .track_uuid = 7, .type = .slice_end });

    var interned: TestInterned = .{};
    var reader: TestReader = .init(out.written());
    var packets: u32 = 0;
    var header_packets: u32 = 0;
    var events: u32 = 0;
    var tracks: u32 = 0;
    while (try reader.next(&interned)) |p| {
        try testing.expect(p.bytes <= packet_bytes_max);
        try testing.expectEqual(@as(u64, 1), p.seq);
        if (packets == 0) {
            try testing.expectEqual(@as(u32, 2), p.n_clocks);
            try testing.expectEqual(@as(u64, clock_monotonic), p.clocks[0]);
            try testing.expectEqual(@as(u64, clock_boottime), p.clocks[1]);
            try testing.expectEqual(@as(u64, clock_monotonic), p.primary_clock);
        } else if (packets == 1) {
            try testing.expectEqual(@as(u64, seq_cleared), p.flags);
            try testing.expectEqual(@as(u64, clock_monotonic), p.defaults_clock);
        } else {
            try testing.expectEqual(@as(u64, seq_needs_state), p.flags);
        }

        if (packets >= 1 and p.event == null and p.track == null) {
            header_packets += 1;
        }

        if (p.track != null) {
            tracks += 1;
        }

        if (p.event) |e| {
            events += 1;
            try testing.expect(p.timestamp != null);
            if (e.type == @intFromEnum(EventType.instant)) {
                try testing.expectEqual(@as(u32, 1), e.n_args);
                try testing.expectEqual(@as(u64, 99), e.args[0].value);
                try testing.expectEqualSlices(u64, &.{ 5, 6 }, e.flows[0..e.n_flows]);
                try testing.expectEqualSlices(u64, &.{7}, e.terms[0..e.n_terms]);
            }
        }

        packets += 1;
    }

    // The names fill more than one packet, and every one resolves at its iid.
    try testing.expect(header_packets > 1);
    for (names, 1..) |n, iid| {
        try testing.expectEqualStrings(n, interned.event_names[iid]);
    }

    try testing.expectEqualStrings("epoch", interned.arg_names[2]);
    try testing.expectEqual(@as(u32, 3), tracks);
    try testing.expectEqual(@as(u32, 3), events);
}

// The spike encoder writes the debug annotation after the flows. These bytes follow this encoder,
// which writes it before them, so the bytes differ from the spike only in that order.
const golden_hex =
    "0a14500132100a05080310e8070a05080610ed0710030a2a50016801da0303d00303621e120a0801" ++
    "1206615f6e616d65120508021201621a09080112056c617965720a1750016802e203100807282a22" ++
    "0a082a102b2a046d61696e0a2940d00f500168025a20480158075001220408011803f9024d000000" ++
    "0000000081034e000000000000000a0d40b410500168025a04480258070a0f409811500168025a06" ++
    "4803582a5002";

test "writer: the bytes equal those of an independent encoder that trace_processor reads" {
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();

    var w: Writer = .init(&out.writer, 1);
    try w.clockSnapshot(1000, 1005);
    try w.header(&.{ "a_name", "b" }, &.{"layer"});
    try w.thread(7, 42, 42, 43, "main");
    try w.event(.{
        .ts_ns = 2000,
        .track_uuid = 7,
        .type = .slice_begin,
        .name_iid = 1,
        .args = &.{.{ .name_iid = 1, .value = 3 }},
        .flow_ids = &.{77},
        .terminating_flow_ids = &.{78},
    });
    try w.event(.{ .ts_ns = 2100, .track_uuid = 7, .type = .slice_end });
    try w.event(.{ .ts_ns = 2200, .track_uuid = 42, .type = .instant, .name_iid = 2 });

    var want: [golden_hex.len / 2]u8 = undefined;
    _ = try std.fmt.hexToBytes(&want, golden_hex);
    try testing.expectEqualSlices(u8, &want, out.written());
}

test "enc: a nested message of any body size up to the bound decodes at its own length" {
    const fill: [packet_bytes_max]u8 = @splat(0xa5);
    // The key and the two reserved length bytes take the rest of the packet.
    const body_max = packet_bytes_max - 1 - length_reserve;
    var n: usize = 0;
    while (n <= body_max) : (n += 1) {
        var enc: Enc = .{};
        const body = enc.open(field.track_event);
        enc.bytes(fill[0..n]);
        enc.close(body);
        try testing.expectEqual(1 + varintLen(n) + n, enc.len);

        var it: TestFields = .{ .bytes = enc.buf[0..enc.len] };
        const f = (try it.next()).?;
        try testing.expectEqual(@as(u64, field.track_event), f.num);
        try testing.expectEqual(n, f.bytes.len);
        try testing.expectEqualSlices(u8, fill[0..n], f.bytes);
        try testing.expect(try it.next() == null);
    }
}

test "writer: a descriptor name of exactly the name bound decodes whole" {
    const name: [name_bytes_max]u8 = @splat('n');
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();

    var w: Writer = .init(&out.writer, 1);
    try w.clockSnapshot(1, 2);
    try w.header(&.{&name}, &.{});
    try w.thread(7, 42, 42, 43, &name);

    var interned: TestInterned = .{};
    var reader: TestReader = .init(out.written());
    var tracks: u32 = 0;
    while (try reader.next(&interned)) |p| {
        const t = p.track orelse continue;
        try testing.expectEqualSlices(u8, &name, t.name);
        tracks += 1;
    }

    try testing.expectEqual(@as(u32, 1), tracks);
    try testing.expectEqualSlices(u8, &name, interned.event_names[1]);
}
