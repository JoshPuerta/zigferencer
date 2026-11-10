//! The layout of the span trace file: the span vocabulary, the tracks, the slices, the flows,
//! and the `trace_meta` counts, encoded with `perfetto.zig`. Pure functions only. `span.zig`
//! collects the spans and calls `writeTrace`.
//!
//! Comptime. The name tables come from the enums at comptime. Asserts pin the size of `Span` and
//! the bound of each event packet.
//!
//! Tracks. The process track has the process id as its uuid. A thread track has `thread_uuid_tag`,
//! its ring index, and its tid, so two rings of one reused tid get two tracks. A GPU queue track
//! has `gpu_uuid_tag` and its runner.
//!
//! Shares. A part span is a `pool_run` of `part` participants or the share of one worker, and
//! `Name.partFits` states its shape. A `pool_run` starts the flows of workers `1..part`, and a
//! worker share ends the flow of its own index. A flow id is the epoch word above the worker index.
//!
//! Args. Bit 0 of `Span.flags` marks a set arg, so an arg of 0 still reads as a value.
//!
//! Ops. An op span takes a `Name` past the tags, one for each `OpKey`, and its slice reads the key
//! name. A GPU slice carries one `<key>_ns` arg for each op key with time.

const std = @import("std");
const limits = @import("limits.zig");
const gpu_record = @import("gpu_record.zig");
const layer_op = @import("layer_op.zig");
const perfetto = @import("perfetto.zig");

const OpKey = layer_op.OpKey;
const OpSet = layer_op.OpSet;
const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

const trace_sequence: u32 = 1;
const process_name = "zigferencer";

/// The uuid tags of the thread tracks and the GPU queue tracks, past any process id.
const thread_uuid_tag: u64 = 1 << 40;
const gpu_uuid_tag: u64 = 2 << 40;

/// The low bits of a flow id hold the worker index. The bits above hold the epoch word.
const flow_worker_bits = 8;

/// A thread uuid holds the ring index above the tid.
const ring_shift = 32;

const gpu_track_name_bytes = 32;

/// The args of one GPU slice: the device time, the host submit and wait, and each op sum.
const gpu_args_max = 3 + OpKey.count;

/// The span names: the tags, then one op name per key.
const name_count = Name.op_base + OpKey.count;
const label_count = std.meta.fields(GpuLabel).len;
const key_count = std.meta.fields(ArgKey).len;

/// The interned event names: the span names (iid = value + 1), the GPU labels, and `trace_meta`.
const event_names = blk: {
    var out: [name_count + label_count + 1][]const u8 = undefined;
    for (std.meta.fields(Name), 0..) |f, i| {
        out[i] = f.name;
    }

    for (0..OpKey.count) |k| {
        const key: OpKey = @enumFromInt(k);
        out[Name.op_base + k] = key.name();
    }

    for (std.meta.fields(GpuLabel), 0..) |f, i| {
        out[name_count + i] = f.name;
    }

    out[name_count + label_count] = "trace_meta";
    break :blk out;
};

const meta_iid: u64 = name_count + label_count + 1;

/// The interned arg names: the `ArgKey` tags (iid = tag + 1), then one `<key>_ns` per op key.
const arg_names = blk: {
    var out: [key_count + OpKey.count][]const u8 = undefined;
    for (std.meta.fields(ArgKey), 0..) |f, i| {
        out[i] = f.name;
    }

    for (0..OpKey.count) |k| {
        const key: OpKey = @enumFromInt(k);
        out[key_count + k] = key.name() ++ "_ns";
    }

    break :blk out;
};

comptime {
    // A `pool_run` slice carries two args and one flow per worker.
    assert(perfetto.eventBytesMax(2, limits.max_threads - 1) <= perfetto.packet_bytes_max);
    assert(perfetto.eventBytesMax(gpu_args_max, 0) <= perfetto.packet_bytes_max);
    assert(perfetto.eventBytesMax(key_count, 0) <= perfetto.packet_bytes_max);
    assert(limits.max_threads < 1 << flow_worker_bits);
    assert(thread_uuid_tag | (@as(u64, limits.span_threads_max) << ring_shift) < gpu_uuid_tag);
}

// -- types ------------------------------------------------------------------

/// The closed span vocabulary. The interned event name is the tag name. An op span takes the
/// value `op_base` plus its `OpKey`, and its event name reads as the key.
pub const Name = enum(u16) {
    model_load,
    tokenize,
    prefill_chunk,
    decode_step,
    sample,
    gpu_record,
    gpu_submit,
    gpu_speculate,
    gpu_wait,
    sched_tick,
    composed_step,
    kv_evict,
    kv_restore,
    socket_read,
    socket_write,
    http_chat,
    client_stream,
    pool_run,
    run_share,
    layer_forward,
    host_prefix,
    worker_spin,
    worker_sleep,
    gpu_collect,
    _,

    /// The value of the first op span: one past the last tag.
    pub const op_base: u16 = std.meta.fields(Name).len;

    /// The name of a span over `ops`.
    pub fn ofOps(ops: OpSet) Name {
        return ofKey(.of(ops, null));
    }

    pub fn ofKey(key: OpKey) Name {
        return @enumFromInt(op_base + @as(u16, @intFromEnum(key)));
    }

    /// The key of an op span. Null for a tag.
    pub fn opKey(self: Name) ?OpKey {
        const n = @intFromEnum(self);
        if (n < op_base) return null;

        assert(n - op_base < OpKey.count);
        return @enumFromInt(n - op_base);
    }

    /// True for an op span, the span that a `Sums` counts.
    pub fn isOp(self: Name) bool {
        return self.opKey() != null;
    }

    /// The shape of a part span: a `pool_run` of 2 or more participants, or a worker share.
    /// Participant 0 runs its share as a `run_share` of part 0.
    pub fn partFits(self: Name, part: u8) bool {
        const worker = part >= 1 and part < limits.max_threads;
        if (self == .pool_run) return part >= 2 and part <= limits.max_threads;
        if (self == .run_share) return part == 0 or worker;
        return self.isOp() and worker;
    }
};

/// One closed span. The times are nanoseconds on the `.awake` clock. `part` is the participant
/// count of a `pool_run`, or the worker index of a share, and 0 elsewhere.
pub const Span = extern struct {
    start_ns: u64,
    end_ns: u64,
    name: Name,
    part: u8 = 0,
    flags: u8 = 0,
    arg: u32 = 0,

    /// The bit of `flags` that marks a set `arg`.
    pub const flag_arg: u8 = 1;

    comptime {
        assert(@sizeOf(Span) == 24);
        // The ring ceiling in `limits` prices one thread ring at 96 MiB.
        assert(@as(u64, limits.spans_per_thread_max) * @sizeOf(Span) == 96 << 20);
    }
};

const GpuLevel = gpu_record.GpuLevel;
const GpuRunner = gpu_record.GpuRunner;
const GpuLabel = gpu_record.GpuLabel;
const GpuRecord = gpu_record.GpuRecord;

/// One thread track of the trace. `spans` must be sorted by `lessThan`.
pub const ThreadTrack = struct {
    /// The index of the ring in the recorder table.
    ring: u32,
    tid: u64,
    name: []const u8,
    spans: []const Span,
};

/// The loss counts that the recorder hands to the writer.
pub const Meta = struct {
    dropped: u64 = 0,
    threads_dropped: u64 = 0,
    /// The spans of the threads that hold no ring.
    orphan_spans: u64 = 0,
    gpu_dropped: u64 = 0,
    gpu_level: GpuLevel = .off,
};

/// The counts that the writer finds while it writes.
pub const Written = struct {
    misnested: u64 = 0,
    gpu_clamps: u64 = 0,
    gpu_clamp_max_ns: u64 = 0,
    gpu_backsteps: u64 = 0,
};

/// One GPU record on the host clock, in the sort order of its queue.
pub const GpuSlot = struct {
    start_ns: u64,
    end_ns: u64,
    index: u32,
    runner: GpuRunner,
};

/// The inputs of `writeTrace`.
pub const TraceInput = struct {
    /// The two clocks, read back to back after the last kept span.
    monotonic_ns: u64,
    boottime_ns: u64,
    pid: u32,
    threads: []const ThreadTrack,
    gpu_records: []const GpuRecord,
    /// One slot per GPU record. The writer sorts them per queue.
    gpu_slots: []GpuSlot,
    meta: Meta,
};

/// The interned arg names. The writer appends one `<key>_ns` per op key.
const ArgKey = enum(u8) {
    epoch,
    layer,
    tokens,
    rows,
    parts,
    worker,
    gpu_ns,
    host_submit_ns,
    host_wait_ns,
    dropped,
    threads_dropped,
    orphan_spans,
    misnested,
    gpu_records,
    gpu_dropped,
    gpu_clamps,
    gpu_clamp_max_ns,
    gpu_backsteps,
    gpu_level,
};

/// The flow ids of one span: the flows a `pool_run` starts, or the flow a worker share ends.
const Flows = struct {
    flow_ids: []const u64 = &.{},
    terminating: []const u64 = &.{},
};

// -- api --------------------------------------------------------------------

/// The span order of one thread track: by start, and the longer span first on a tie.
pub fn lessThan(_: void, a: Span, b: Span) bool {
    if (a.start_ns != b.start_ns) return a.start_ns < b.start_ns;
    return a.end_ns > b.end_ns;
}

/// Write the trace packets: the clock snapshot, the names, the tracks, the slices of each thread
/// and GPU queue, and the `trace_meta` instant.
pub fn writeTrace(w: *std.Io.Writer, in: TraceInput) std.Io.Writer.Error!Written {
    assert(in.gpu_slots.len == in.gpu_records.len);
    assert(in.pid != 0);
    assert(in.pid < thread_uuid_tag);

    var pw: perfetto.Writer = .init(w, trace_sequence);
    try pw.clockSnapshot(in.monotonic_ns, in.boottime_ns);
    try pw.header(&event_names, &arg_names);
    const process_uuid: u64 = in.pid;
    try pw.process(process_uuid, in.pid, process_name);

    var out: Written = .{};
    for (in.threads) |t| {
        const uuid = threadUuid(t.ring, t.tid);
        try pw.thread(uuid, process_uuid, in.pid, t.tid, t.name);
        out.misnested += try writeThread(&pw, uuid, t.spans);
    }

    try writeGpu(&pw, process_uuid, in, &out);
    try writeMeta(&pw, process_uuid, in, out);
    return out;
}

// -- helpers ----------------------------------------------------------------

fn threadUuid(ring: u32, tid: u64) u64 {
    assert(ring < limits.span_threads_max);
    assert(tid < 1 << ring_shift);

    return thread_uuid_tag | (@as(u64, ring) << ring_shift) | tid;
}

fn gpuUuid(runner: GpuRunner) u64 {
    return gpu_uuid_tag | @intFromEnum(runner);
}

fn nameIid(name: Name) u64 {
    return @as(u64, @intFromEnum(name)) + 1;
}

fn labelIid(label: GpuLabel) u64 {
    return name_count + @as(u64, @intFromEnum(label)) + 1;
}

fn argOf(key: ArgKey, value: u64) perfetto.Arg {
    return .{ .name_iid = @as(u64, @intFromEnum(key)) + 1, .value = value };
}

fn opArg(key: OpKey, ns: u64) perfetto.Arg {
    return .{ .name_iid = key_count + @as(u64, @intFromEnum(key)) + 1, .value = ns };
}

/// The label of the span arg. An op or a `run_share` carries the epoch word only as a worker share.
fn argKey(name: Name, part: u8) ?ArgKey {
    return switch (name) {
        .pool_run => .epoch,
        .layer_forward => .layer,
        .prefill_chunk => .tokens,
        .decode_step, .composed_step, .sched_tick => .rows,
        .run_share => .epoch,
        else => if (name.isOp() and part >= 1) .epoch else null,
    };
}

/// The span arg when it is set, then `parts` for a `pool_run` or `worker` for a share.
fn spanArgs(s: Span, buf: *[2]perfetto.Arg) []const perfetto.Arg {
    var n: usize = 0;
    if (s.flags & Span.flag_arg != 0) {
        if (argKey(s.name, s.part)) |key| {
            buf[n] = argOf(key, s.arg);
            n += 1;
        }
    }

    if (s.part >= 1) {
        buf[n] = argOf(if (s.name == .pool_run) .parts else .worker, s.part);
        n += 1;
    }

    return buf[0..n];
}

/// A `pool_run` of `part` participants starts the flows of workers `1..part`. A share with
/// `part >= 1` ends the flow of its worker.
fn spanFlows(s: Span, buf: *[limits.max_threads]u64) Flows {
    if (s.part == 0) return .{};

    assert(s.name.partFits(s.part));
    if (s.name != .pool_run) {
        buf[0] = flowId(s.arg, s.part);
        return .{ .terminating = buf[0..1] };
    }

    assert(s.part >= 2);
    const n = s.part - 1;
    for (buf[0..n], 1..) |*id, w| {
        id.* = flowId(s.arg, @intCast(w));
    }

    assert(n < limits.max_threads);
    return .{ .flow_ids = buf[0..n] };
}

fn flowId(word: u32, worker: u8) u64 {
    assert(worker >= 1);
    assert(worker < limits.max_threads);

    return (@as(u64, word) << flow_worker_bits) | worker;
}

/// The slices of one thread track. Returns the spans that the walk skipped.
fn writeThread(pw: *perfetto.Writer, uuid: u64, spans: []const Span) !u32 {
    var nest: perfetto.Nest(Span) = .init(spans);
    while (nest.next()) |step| {
        const s = spans[step.index];
        if (step.kind == .end) {
            try pw.event(.{ .ts_ns = s.end_ns, .track_uuid = uuid, .type = .slice_end });
            continue;
        }

        var args_buf: [2]perfetto.Arg = undefined;
        var flow_buf: [limits.max_threads]u64 = undefined;
        const flows = spanFlows(s, &flow_buf);
        try pw.event(.{
            .ts_ns = s.start_ns,
            .track_uuid = uuid,
            .type = .slice_begin,
            .name_iid = nameIid(s.name),
            .args = spanArgs(s, &args_buf),
            .flow_ids = flows.flow_ids,
            .terminating_flow_ids = flows.terminating,
        });
    }

    return nest.misnested;
}

/// The GPU queue tracks and their slices. A queue sorts by host start, so its slices nest.
fn writeGpu(pw: *perfetto.Writer, process_uuid: u64, in: TraceInput, out: *Written) !void {
    // Map each record on the host clock, and count the clamps and the backsteps.
    var used: [std.meta.fields(GpuRunner).len]bool = @splat(false);
    for (in.gpu_records, in.gpu_slots, 0..) |rec, *slot, i| {
        const at = gpu_record.hostInterval(in.gpu_records, i);
        if (at.clamp_ns > 0) {
            out.gpu_clamps += 1;
        }

        out.gpu_clamp_max_ns = @max(out.gpu_clamp_max_ns, at.clamp_ns);
        out.gpu_backsteps += rec.backsteps;
        used[@intFromEnum(rec.runner)] = true;
        slot.* = .{
            .start_ns = @intCast(at.start_ns),
            .end_ns = @intCast(at.end_ns),
            .index = @intCast(i),
            .runner = rec.runner,
        };
    }

    // Name one queue track per runner kind that has a record.
    for (used, 0..) |u, r| {
        if (!u) continue;

        const runner: GpuRunner = @enumFromInt(r);
        var name_buf: [gpu_track_name_bytes]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "GPU queue {t}", .{runner}) catch unreachable;
        try pw.track(gpuUuid(runner), process_uuid, name);
    }

    // Walk each queue as one run of the sorted slots.
    std.mem.sort(GpuSlot, in.gpu_slots, {}, gpuSlotLess);
    var lo: usize = 0;
    while (lo < in.gpu_slots.len) {
        var hi = lo + 1;
        while (hi < in.gpu_slots.len and in.gpu_slots[hi].runner == in.gpu_slots[lo].runner) {
            hi += 1;
        }

        out.misnested += try writeQueue(pw, in.gpu_records, in.gpu_slots[lo..hi]);
        lo = hi;
    }
}

fn writeQueue(pw: *perfetto.Writer, records: []const GpuRecord, slots: []const GpuSlot) !u32 {
    assert(slots.len > 0);

    const uuid = gpuUuid(slots[0].runner);
    var nest: perfetto.Nest(GpuSlot) = .init(slots);
    while (nest.next()) |step| {
        const slot = slots[step.index];
        assert(slot.runner == slots[0].runner);
        if (step.kind == .end) {
            try pw.event(.{ .ts_ns = slot.end_ns, .track_uuid = uuid, .type = .slice_end });
            continue;
        }

        const rec = records[slot.index];
        var args_buf: [gpu_args_max]perfetto.Arg = undefined;
        try pw.event(.{
            .ts_ns = slot.start_ns,
            .track_uuid = uuid,
            .type = .slice_begin,
            .name_iid = labelIid(rec.label),
            .args = gpuArgs(rec, &args_buf),
        });
    }

    return nest.misnested;
}

fn gpuArgs(rec: GpuRecord, buf: *[gpu_args_max]perfetto.Arg) []const perfetto.Arg {
    buf[0] = argOf(.gpu_ns, @intCast(rec.gpuEndNs() - rec.gpuStartNs()));
    buf[1] = argOf(.host_submit_ns, rec.host_submit_ns);
    buf[2] = argOf(.host_wait_ns, rec.host_wait_ns);
    var n: usize = 3;
    if (rec.op_ns) |sums| {
        for (sums, 0..) |ns, k| {
            if (ns == 0) continue;
            buf[n] = opArg(@enumFromInt(k), ns);
            n += 1;
        }
    }

    assert(n <= gpu_args_max);
    return buf[0..n];
}

fn writeMeta(pw: *perfetto.Writer, process_uuid: u64, in: TraceInput, out: Written) !void {
    const m = in.meta;
    const args = [_]perfetto.Arg{
        argOf(.dropped, m.dropped),
        argOf(.threads_dropped, m.threads_dropped),
        argOf(.orphan_spans, m.orphan_spans),
        argOf(.misnested, out.misnested),
        argOf(.gpu_records, in.gpu_records.len),
        argOf(.gpu_dropped, m.gpu_dropped),
        argOf(.gpu_clamps, out.gpu_clamps),
        argOf(.gpu_clamp_max_ns, out.gpu_clamp_max_ns),
        argOf(.gpu_backsteps, out.gpu_backsteps),
        argOf(.gpu_level, @intFromEnum(m.gpu_level)),
    };
    try pw.event(.{
        .ts_ns = in.monotonic_ns,
        .track_uuid = process_uuid,
        .type = .instant,
        .name_iid = meta_iid,
        .args = &args,
    });
}

/// The queue order: by runner, then by start, and the longer slot first on a tie.
fn gpuSlotLess(_: void, a: GpuSlot, b: GpuSlot) bool {
    if (a.runner != b.runner) return @intFromEnum(a.runner) < @intFromEnum(b.runner);
    if (a.start_ns != b.start_ns) return a.start_ns < b.start_ns;
    return a.end_ns > b.end_ns;
}

// -- tests ------------------------------------------------------------------

const prop = @import("prop.zig");
const testing = std.testing;

test "name: partFits accepts a pool run of 2 or more parts and a worker share only" {
    const max: u8 = @intCast(limits.max_threads);
    try testing.expect(Name.pool_run.partFits(2));
    try testing.expect(Name.pool_run.partFits(max));
    try testing.expect(!Name.pool_run.partFits(1));
    try testing.expect(!Name.pool_run.partFits(max + 1));
    try testing.expect(Name.run_share.partFits(0));
    try testing.expect(Name.run_share.partFits(max - 1));
    try testing.expect(!Name.run_share.partFits(max));
    const attention: Name = .ofOps(.initOne(.attention));
    try testing.expect(attention.partFits(1));
    try testing.expect(!attention.partFits(0));
    try testing.expect(!attention.partFits(max));
    try testing.expect(!Name.decode_step.partFits(1));
    try testing.expect(!Name.layer_forward.partFits(0));
}

test "name: an op name follows the tags, maps back to its key, and reads as the key" {
    for (std.enums.values(Name)) |n| {
        try testing.expect(!n.isOp());
    }

    for (0..OpKey.count) |k| {
        const key: OpKey = @enumFromInt(k);
        const n: Name = .ofKey(key);
        try testing.expect(n.isOp());
        try testing.expectEqual(key, n.opKey().?);
        try testing.expectEqualStrings(key.name(), event_names[nameIid(n) - 1]);
    }

    try testing.expectEqual(Name.ofKey(.ofOp(.rope)), Name.ofOps(.initOne(.rope)));
}

/// The bounds of a decoded test trace.
const decoded_tracks_max = 16;
const decoded_begins_max = 160;
const decoded_flows_max = 128;
const row_args_max = 4;

/// One decoded slice begin: its track, its name, its args, and its flows.
const BeginRow = struct {
    track: u64,
    name: []const u8,
    args: [row_args_max]perfetto.Arg,
    n_args: u32,
    n_flows: u32,
    term: ?u64,
};

/// What a decoded trace holds. `decodeTrace` checks the per-track invariants as it reads.
const Decoded = struct {
    interned: perfetto.TestInterned = .{},
    begins: [decoded_begins_max]BeginRow = undefined,
    n_begins: u32 = 0,
    n_ends: u32 = 0,
    flows: [decoded_flows_max]u64 = undefined,
    n_flows: u32 = 0,
    terms: [decoded_flows_max]u64 = undefined,
    n_terms: u32 = 0,
    meta: ?perfetto.TestEvent = null,
    track_names: [decoded_tracks_max][]const u8 = undefined,
    n_track_names: u32 = 0,
    /// The tracks that carry an event.
    n_tracks: u32 = 0,

    fn metaArg(self: *const Decoded, key: []const u8) ?u64 {
        const m = self.meta orelse return null;
        return self.find(m.args[0..m.n_args], key);
    }

    fn argOfRow(self: *const Decoded, row: BeginRow, key: []const u8) ?u64 {
        return self.find(row.args[0..row.n_args], key);
    }

    fn find(self: *const Decoded, args: []const perfetto.Arg, key: []const u8) ?u64 {
        for (args) |a| {
            if (std.mem.eql(u8, self.interned.arg_names[@intCast(a.name_iid)], key)) return a.value;
        }

        return null;
    }
};

/// Decode every packet. Per track, the events balance and their timestamps never fall, and each
/// named event resolves through the interned names.
fn decodeTrace(bytes: []const u8, d: *Decoded) !void {
    const TrackState = struct { uuid: u64, depth: u32, last_ts: u64 };
    var tracks: [decoded_tracks_max]TrackState = undefined;
    var reader: perfetto.TestReader = .init(bytes);
    while (try reader.next(&d.interned)) |p| {
        try testing.expect(p.bytes <= perfetto.packet_bytes_max);
        if (p.track) |t| {
            try testing.expect(d.n_track_names < decoded_tracks_max);
            d.track_names[d.n_track_names] = t.name;
            d.n_track_names += 1;
        }

        const e = p.event orelse continue;

        // Find the state of the event track.
        const ts = p.timestamp.?;
        var st: ?*TrackState = null;
        for (tracks[0..d.n_tracks]) |*t| {
            if (t.uuid == e.track_uuid) {
                st = t;
            }
        }

        if (st == null) {
            try testing.expect(d.n_tracks < decoded_tracks_max);
            tracks[d.n_tracks] = .{ .uuid = e.track_uuid, .depth = 0, .last_ts = 0 };
            st = &tracks[d.n_tracks];
            d.n_tracks += 1;
        }

        const t = st.?;
        try testing.expect(ts >= t.last_ts);
        t.last_ts = ts;

        // Check the balance, resolve the name, and keep the flows.
        switch (e.type) {
            @intFromEnum(perfetto.EventType.slice_begin) => {
                t.depth += 1;
                try keepBegin(d, e);
            },
            @intFromEnum(perfetto.EventType.slice_end) => {
                try testing.expect(t.depth > 0);
                t.depth -= 1;
                d.n_ends += 1;
            },
            @intFromEnum(perfetto.EventType.instant) => {
                const name = d.interned.event_names[@intCast(e.name_iid)];
                try testing.expectEqualStrings("trace_meta", name);
                d.meta = e;
            },
            else => return error.UnknownEventType,
        }

        try testing.expect(d.n_flows + e.n_flows <= decoded_flows_max);
        try testing.expect(d.n_terms + e.n_terms <= decoded_flows_max);
        @memcpy(d.flows[d.n_flows..][0..e.n_flows], e.flows[0..e.n_flows]);
        d.n_flows += e.n_flows;
        @memcpy(d.terms[d.n_terms..][0..e.n_terms], e.terms[0..e.n_terms]);
        d.n_terms += e.n_terms;
    }

    for (tracks[0..d.n_tracks]) |t| {
        try testing.expectEqual(@as(u32, 0), t.depth);
    }

    try testing.expectEqual(d.n_begins, d.n_ends);
}

/// Keep one slice begin. A begin past the arg cap of a row fails the test.
fn keepBegin(d: *Decoded, e: perfetto.TestEvent) !void {
    const name = d.interned.event_names[@intCast(e.name_iid)];
    try testing.expect(name.len > 0);
    try testing.expect(e.n_args <= row_args_max);
    try testing.expect(e.n_terms <= 1);

    var row: BeginRow = .{
        .track = e.track_uuid,
        .name = name,
        .args = undefined,
        .n_args = e.n_args,
        .n_flows = e.n_flows,
        .term = if (e.n_terms == 1) e.terms[0] else null,
    };
    @memcpy(row.args[0..row.n_args], e.args[0..row.n_args]);
    if (d.n_begins < decoded_begins_max) {
        d.begins[d.n_begins] = row;
    }

    d.n_begins += 1;
}

const test_record: GpuRecord = .{
    .runner = .staging,
    .label = .fill,
    .gpu_start = 0,
    .gpu_end = 1,
    .period_ns = 1,
    .host_submit_ns = 1,
    .host_wait_ns = 2,
};

const test_clock: struct { monotonic: u64, boottime: u64 } = .{
    .monotonic = 5_000_000,
    .boottime = 5_000_100,
};

/// Write the tracks with no GPU record, into `out`.
fn writeTracks(out: *std.Io.Writer.Allocating, tracks: []const ThreadTrack, meta: Meta) !Written {
    return writeTrace(&out.writer, .{
        .monotonic_ns = test_clock.monotonic,
        .boottime_ns = test_clock.boottime,
        .pid = 7,
        .threads = tracks,
        .gpu_records = &.{},
        .gpu_slots = &.{},
        .meta = meta,
    });
}

/// A random forest of nested spans on a few threads.
const TraceCase = struct {
    spans: [3][48]Span,
    lens: [3]u32,
};

fn genTrace(_: void, random: std.Random, size: u32) TraceCase {
    var c: TraceCase = .{ .spans = undefined, .lens = @splat(0) };
    for (0..3) |t| {
        var n: u32 = 0;
        var cursor: u64 = 1_000_000 + random.uintLessThan(u64, 1000);
        // Each root holds up to two levels of children inside its bounds.
        while (n + 7 <= 48 and n < size) {
            const root_start = cursor;
            var inner = root_start;
            const kids = random.uintLessThan(u32, 3);
            var k: u32 = 0;
            var buf: [6]Span = undefined;
            var nb: u32 = 0;
            while (k < kids) : (k += 1) {
                const ks = inner + random.uintLessThan(u64, 50);
                var ke = ks + random.uintLessThan(u64, 500);
                if (random.boolean()) {
                    const gs = ks + random.uintLessThan(u64, 10);
                    const ge = gs + random.uintLessThan(u64, 100);
                    buf[nb] = .{ .start_ns = gs, .end_ns = ge, .name = .sample };
                    nb += 1;
                    ke = @max(ke, ge);
                }

                buf[nb] = .{
                    .start_ns = ks,
                    .end_ns = ke,
                    .name = .decode_step,
                    .arg = k,
                    .flags = Span.flag_arg,
                };
                nb += 1;
                inner = ke;
            }

            const root_end = inner + random.uintLessThan(u64, 50);
            // Push order is end order, as the recorder pushes: children first.
            for (buf[0..nb]) |s| {
                c.spans[t][n] = s;
                n += 1;
            }

            c.spans[t][n] = .{ .start_ns = root_start, .end_ns = root_end, .name = .sched_tick };
            n += 1;
            cursor = root_end + random.uintLessThan(u64, 100);
        }

        c.lens[t] = n;
    }

    return c;
}

fn checkTrace(_: void, case: TraceCase) anyerror!void {
    var c = case;
    var tracks: [3]ThreadTrack = undefined;
    var total: u32 = 0;
    for (0..3) |t| {
        const s = c.spans[t][0..c.lens[t]];
        std.mem.sort(Span, s, {}, lessThan);
        tracks[t] = .{ .ring = @intCast(t), .tid = 100 + t, .name = "worker \"q\"", .spans = s };
        total += c.lens[t];
    }

    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();

    const written = try writeTracks(&out, &tracks, .{});
    try testing.expectEqual(@as(u64, 0), written.misnested);

    const d = try testing.allocator.create(Decoded);
    defer testing.allocator.destroy(d);
    d.* = .{};
    try decodeTrace(out.written(), d);
    try testing.expectEqual(total, d.n_begins);
    try testing.expectEqual(@as(?u64, 0), d.metaArg("misnested"));
}

test "trace: the packets decode, and per track the slices balance and never step back" {
    try prop.forAll(
        .{ .site = 0x5350_0002, .cases = 64, .max_size = 40 },
        {},
        TraceCase,
        genTrace,
        checkTrace,
    );
}

test "trace: a pool run starts one flow per worker share, and trace_meta states the counts" {
    // One dispatcher with a three-part `pool_run`, two worker shares, and one partial overlap.
    var main_spans = [_]Span{
        .{
            .start_ns = 1000,
            .end_ns = 2000,
            .name = .decode_step,
            .arg = 1,
            .flags = Span.flag_arg,
        },
        .{
            .start_ns = 1100,
            .end_ns = 1600,
            .name = .pool_run,
            .arg = 7,
            .part = 3,
            .flags = Span.flag_arg,
        },
        .{ .start_ns = 1110, .end_ns = 1500, .name = .run_share, .arg = 7, .flags = Span.flag_arg },
        .{ .start_ns = 1700, .end_ns = 1800, .name = .ofOps(.initOne(.attn_norm)) },
        .{ .start_ns = 1900, .end_ns = 2100, .name = .sample },
    };
    var one = [_]Span{
        .{
            .start_ns = 1120,
            .end_ns = 1450,
            .name = .ofOps(.initOne(.attention)),
            .arg = 7,
            .part = 1,
            .flags = Span.flag_arg,
        },
    };
    var two = [_]Span{
        .{ .start_ns = 1000, .end_ns = 1130, .name = .worker_spin },
        .{
            .start_ns = 1130,
            .end_ns = 1480,
            .name = .ofOps(.initOne(.attention)),
            .arg = 7,
            .part = 2,
            .flags = Span.flag_arg,
        },
    };
    const tracks = [_]ThreadTrack{
        .{ .ring = 0, .tid = 100, .name = "main", .spans = &main_spans },
        .{ .ring = 1, .tid = 101, .name = "zf-pool-1", .spans = &one },
        .{ .ring = 2, .tid = 102, .name = "zf-pool-2", .spans = &two },
    };
    var late = test_record;
    late.gpu_start = 5;
    late.gpu_end = 8;
    late.host_submit_ns = 3;
    late.host_wait_ns = 10;
    const records = [_]GpuRecord{ test_record, late };
    var slots: [records.len]GpuSlot = undefined;
    const meta: Meta = .{
        .dropped = 4,
        .threads_dropped = 1,
        .orphan_spans = 2,
        .gpu_level = .submit,
    };

    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();

    const written = try writeTrace(&out.writer, .{
        .monotonic_ns = test_clock.monotonic,
        .boottime_ns = test_clock.boottime,
        .pid = 7,
        .threads = &tracks,
        .gpu_records = &records,
        .gpu_slots = &slots,
        .meta = meta,
    });
    try testing.expectEqual(@as(u64, 1), written.misnested);

    const d = try testing.allocator.create(Decoded);
    defer testing.allocator.destroy(d);
    d.* = .{};
    try decodeTrace(out.written(), d);

    // Every span but the overlap, and both GPU records.
    try testing.expectEqual(@as(u32, 4 + 1 + 2 + 2), d.n_begins);
    try testing.expectEqualStrings("decode_step", d.begins[0].name);
    try testing.expectEqual(@as(?u64, 1), d.argOfRow(d.begins[0], "rows"));
    try testing.expectEqual(@as(?u64, 3), d.argOfRow(d.begins[1], "parts"));

    // Each terminating flow matches one flow of the pool run.
    try testing.expectEqual(@as(u32, 2), d.n_flows);
    try testing.expectEqual(@as(u32, 2), d.n_terms);
    for (d.terms[0..d.n_terms]) |term| {
        var hits: u32 = 0;
        for (d.flows[0..d.n_flows]) |f| {
            if (f == term) {
                hits += 1;
            }
        }

        try testing.expectEqual(@as(u32, 1), hits);
    }

    // A worker share names its worker and its epoch.
    var shares: u32 = 0;
    for (d.begins[0..d.n_begins]) |row| {
        if (!std.mem.eql(u8, row.name, "attention")) continue;

        shares += 1;
        try testing.expectEqual(@as(?u64, 7), d.argOfRow(row, "epoch"));
        // The flow that the share ends carries its worker index in the low byte.
        const worker = d.argOfRow(row, "worker").?;
        try testing.expect(worker >= 1);
        try testing.expectEqual(worker, row.term.? & 0xff);
    }

    try testing.expectEqual(@as(u32, 2), shares);

    // The tracks carry their names, and trace_meta holds the counts.
    const names = d.track_names[0..d.n_track_names];
    try testing.expectEqual(@as(u32, 5), d.n_track_names);
    try testing.expectEqualStrings("zigferencer", names[0]);
    try testing.expectEqualStrings("main", names[1]);
    try testing.expectEqualStrings("zf-pool-2", names[3]);
    try testing.expectEqualStrings("GPU queue staging", names[4]);
    try testing.expectEqual(@as(?u64, 4), d.metaArg("dropped"));
    try testing.expectEqual(@as(?u64, 1), d.metaArg("threads_dropped"));
    try testing.expectEqual(@as(?u64, 2), d.metaArg("orphan_spans"));
    try testing.expectEqual(@as(?u64, 1), d.metaArg("misnested"));
    try testing.expectEqual(@as(?u64, 2), d.metaArg("gpu_records"));
    try testing.expectEqual(@as(?u64, @intFromEnum(GpuLevel.submit)), d.metaArg("gpu_level"));
}

/// Write `tracks` into `out`, check the result decodes, and return the decoded trace. Its names
/// point into `out`. The caller frees both.
fn writeAndDecode(out: *std.Io.Writer.Allocating, tracks: []const ThreadTrack) !*Decoded {
    const written = try writeTracks(out, tracks, .{});
    try testing.expectEqual(@as(u64, 0), written.misnested);

    const d = try testing.allocator.create(Decoded);
    errdefer testing.allocator.destroy(d);
    d.* = .{};
    try decodeTrace(out.written(), d);
    return d;
}

test "trace: a pool run under 2 parts writes no flow, one of max_threads parts one per worker" {
    var spans = [_]Span{
        .{ .start_ns = 10, .end_ns = 20, .name = .pool_run, .arg = 3, .flags = Span.flag_arg },
        .{
            .start_ns = 30,
            .end_ns = 40,
            .name = .pool_run,
            .arg = 9,
            .part = @intCast(limits.max_threads),
            .flags = 1,
        },
    };
    const tracks = [_]ThreadTrack{.{ .ring = 0, .tid = 5, .name = "main", .spans = &spans }};
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    const d = try writeAndDecode(&out, &tracks);
    defer testing.allocator.destroy(d);

    try testing.expectEqual(@as(u32, 0), d.begins[0].n_flows);
    try testing.expectEqual(limits.max_threads - 1, d.begins[1].n_flows);
    try testing.expectEqual(limits.max_threads - 1, d.n_flows);
    for (d.flows[0..d.n_flows], 1..) |f, w| {
        try testing.expectEqual(@as(u64, w), f & 0xff);
        try testing.expectEqual(@as(u64, 9), f >> 8);
    }
}

test "trace: two rings of one reused tid decode as two balanced tracks" {
    var first = [_]Span{
        .{ .start_ns = 10, .end_ns = 50, .name = .decode_step },
        .{ .start_ns = 20, .end_ns = 30, .name = .sample },
    };
    var second = [_]Span{.{ .start_ns = 15, .end_ns = 60, .name = .sched_tick }};
    const tracks = [_]ThreadTrack{
        .{ .ring = 0, .tid = 55, .name = "zf-pool-1", .spans = &first },
        .{ .ring = 1, .tid = 55, .name = "zf-pool-1", .spans = &second },
    };
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    const d = try writeAndDecode(&out, &tracks);
    defer testing.allocator.destroy(d);

    // The two thread tracks and the process track of `trace_meta`.
    try testing.expectEqual(@as(u32, 3), d.n_tracks);
    try testing.expectEqual(@as(u32, 3), d.n_begins);
    try testing.expect(d.begins[0].track != d.begins[2].track);
}

test "trace: an arg of 0 writes when its flag is set, and a clear flag writes no arg" {
    var spans = [_]Span{
        .{ .start_ns = 10, .end_ns = 20, .name = .decode_step, .flags = Span.flag_arg },
        .{ .start_ns = 30, .end_ns = 40, .name = .decode_step, .arg = 5 },
    };
    const tracks = [_]ThreadTrack{.{ .ring = 0, .tid = 5, .name = "main", .spans = &spans }};
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    const d = try writeAndDecode(&out, &tracks);
    defer testing.allocator.destroy(d);

    try testing.expectEqual(@as(?u64, 0), d.argOfRow(d.begins[0], "rows"));
    try testing.expectEqual(@as(?u64, null), d.argOfRow(d.begins[1], "rows"));
}
