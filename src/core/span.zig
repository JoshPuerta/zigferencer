//! The span recorder: a timeline of the engine boundaries for one process, written as a Perfetto
//! protobuf trace for ui.perfetto.dev and trace_processor.
//!
//! Comptime. `compiled` is `-Dtrace`. Each site entry is `live.x` with it and `off.x` without it.
//! `off` holds empty inline bodies, so a build without trace emits nothing at a site. Asserts pin
//! the size of `Span` and the bound of each event packet.
//!
//! Switches. `$ZIGFERENCER_TRACE_OUT=path` turns the recorder on at startup. A span site then reads
//! the monotonic clock twice and pushes one `Span` into the ring of its thread. With the recorder
//! off, a site costs one atomic load and one branch, and it touches no thread-local state.
//! `$ZIGFERENCER_TRACE_SECONDS=N` bounds the capture window. The recorder writes the trace when the
//! window ends or at exit, whichever comes first. `$ZIGFERENCER_TRACE_SPANS=N` sets the capacity of
//! every span ring. The GPU ring holds `limits.trace_gpu_records`.
//!
//! Gate. One atomic word: bit 0 is set while the recorder records, and the bits above it count the
//! installed `Sums`. A site loads the word once, and a zero word gives an inert scope. The flush
//! clears bit 0 before the writer reads the rings.
//!
//! Scopes. `begin` and `beginArg` save the innermost open name of the thread and set their own.
//! `end` restores it, so `current` names the innermost open `begin` scope. `beginPart` leaves it.
//! Only the end of a `begin` scope with an op name adds to the `Sums` of its thread, so a pool
//! share never counts an op twice. An op name names an `OpKey`: `Name.ofOps` makes it.
//!
//! Storage. It is the dynamic shell of this module. Nothing is allocated when the recorder stays
//! off. Each thread claims one fixed ring from a table bounded by `limits.span_threads_max` and
//! allocates the spans of that ring at its first span. A full ring drops the span and counts it.
//! A kept span never changes after its push, so the writer can read `kept` while the owner still
//! pushes. A ring publishes its `tid` and name with a release store of `published`, and the writer
//! reads them only after an acquire load of that flag.
//!
//! GPU ring. One more fixed ring holds the GPU submit records, which the GPU runners hand over
//! after each wait. Any thread can push: a push claims an index with one atomic add and publishes
//! the record with a release store of its flag. The storage is one block of `per` records, then
//! `per` ready flags. The first push allocates it, so a trace with no GPU record holds none. Two
//! first pushes can race. Each allocates, a compare-exchange installs one block, and the losing
//! thread frees its own block. The winner sets the flags before its release exchange, and every
//! user loads the block with acquire order, so each user sees the cleared flags. A failed
//! allocation turns the ring off for the process. Its push and each later push leave their index
//! unwritten, and `kept` counts each unwritten index as dropped, so the metadata and the write log
//! state the loss. A retry gains nothing: the published prefix cannot pass the unwritten index of
//! the failed push.
//!
//! Output. The flush sorts each ring and hands the rings and the GPU records to
//! `trace_file.writeTrace`, which owns the layout. The GPU folded file goes beside the trace.
//!
//! Flush. Only the first `flush` call writes. A later call waits for that write to end, for at most
//! `flush_wait_max_s`, so an exit after `finish` keeps the whole trace that the window thread
//! writes. Happens-before: the writer closes both files, then stores `done` with release order. A
//! waiter returns only after an acquire load reads `done`, so every file write precedes the return
//! of each caller.
//!
//! Clock. It is `.awake` (`CLOCK_MONOTONIC` on Linux), the clock of `perf record -k mono`, so a
//! span file and a perf file share one time base. A GPU clock maps to that base only through the
//! submit and wait points of each record, with `gpu_record.hostInterval`.

const std = @import("std");
const builtin = @import("builtin");
const limits = @import("limits.zig");
const core_log = @import("utils/log.zig");
const gpu_record = @import("gpu_record.zig");
const layer_op = @import("layer_op.zig");
const trace_file = @import("trace_file.zig");

const OpKey = layer_op.OpKey;
const assert = std.debug.assert;
const log = std.log.scoped(.span);

// -- constants --------------------------------------------------------------

/// True under `-Dtrace`. Without it, every site entry is an empty inline body.
pub const compiled: bool = core_log.hot;

/// The gate bit of a recording recorder, and the gate step of one installed `Sums`.
const gate_record: u32 = 1;
const gate_sink: u32 = 2;

/// The longest wait of a later `flush` caller. It only catches a stalled file system.
const flush_wait_max_s = 60;

const write_buffer_bytes = 1 << 20;

// -- types ------------------------------------------------------------------

pub const Name = trace_file.Name;
pub const Span = trace_file.Span;

/// The nanoseconds per op key of one thread, filled by `installSums`.
pub const Sums = struct {
    ns: [OpKey.count]u64 = @splat(0),

    pub fn get(self: *const Sums, key: OpKey) u64 {
        return self.ns[@intFromEnum(key)];
    }

    fn add(self: *Sums, key: OpKey, ns: u64) void {
        self.ns[@intFromEnum(key)] += ns;
    }
};

pub const GpuLevel = gpu_record.GpuLevel;
pub const GpuRunner = gpu_record.GpuRunner;
pub const GpuLabel = gpu_record.GpuLabel;
pub const GpuRecord = gpu_record.GpuRecord;

const ThreadTrack = trace_file.ThreadTrack;
const Meta = trace_file.Meta;
const Written = trace_file.Written;
const GpuSlot = trace_file.GpuSlot;

/// One fixed ring of spans, written by its owner thread alone.
pub const Ring = struct {
    /// Empty until `Recorder.claim` allocates the ring for its thread.
    spans: []Span = &.{},
    len: std.atomic.Value(u32) = .init(0),
    dropped: std.atomic.Value(u64) = .init(0),
    tid: u64 = 0,
    name_buf: [16]u8 = @splat(0),
    name_len: std.atomic.Value(u8) = .init(0),
    /// Set with release order after the claim writes `tid` and the name.
    published: std.atomic.Value(bool) = .init(false),

    pub fn push(self: *Ring, s: Span) void {
        assert(s.start_ns <= s.end_ns);

        const n = self.len.load(.monotonic);
        assert(n <= self.spans.len);
        if (n == self.spans.len) {
            _ = self.dropped.fetchAdd(1, .monotonic);
            return;
        }

        self.spans[n] = s;
        self.len.store(n + 1, .release);
    }

    pub fn kept(self: *Ring) []Span {
        return self.spans[0..self.len.load(.acquire)];
    }

    fn setName(self: *Ring, name: []const u8) void {
        const n: u8 = @intCast(@min(name.len, self.name_buf.len));
        @memcpy(self.name_buf[0..n], name[0..n]);
        self.name_len.store(n, .release);
    }

    fn label(self: *Ring) []const u8 {
        return self.name_buf[0..self.name_len.load(.acquire)];
    }
};

/// The startup inputs of the recorder, read from the environment by `main`.
pub const Env = struct {
    out: ?[]const u8 = null,
    seconds: ?[]const u8 = null,
    gpu: ?[]const u8 = null,
    spans: ?[]const u8 = null,
};

/// One open span. `end` closes it.
pub const Scope = if (compiled) live.Scope else off.Scope;

/// The recording site entries. Plain functions, so a debugger step filter can skip them by name.
const live = struct {
    /// `start_ns == 0` means the gate was zero at the open, and `end` then does nothing.
    pub const Scope = struct {
        start_ns: u64,
        name: Name,
        prev: ?Name,
        arg: u32,
        part: u8,
        flags: u8,
        kind: Kind,

        pub const Kind = enum(u8) { begin, part };

        pub fn end(self: live.Scope) void {
            if (self.start_ns == 0) return;
            self.endAt(nowNs());
        }

        /// The sink installed at the end counts the span, and the gate at the end decides the push.
        fn endAt(self: live.Scope, end_ns: u64) void {
            assert(self.start_ns != 0);
            assert(self.start_ns <= end_ns);

            if (self.kind == .begin) {
                assert(tl_current == self.name);
                tl_current = self.prev;
                if (self.name.opKey()) |key| {
                    if (tl_sums) |sums| {
                        sums.add(key, end_ns - self.start_ns);
                    }
                }
            }

            if (active.load(.acquire) & gate_record == 0) return;
            pushCurrent(.{
                .start_ns = self.start_ns,
                .end_ns = end_ns,
                .name = self.name,
                .part = self.part,
                .flags = self.flags,
                .arg = self.arg,
            });
        }
    };

    pub fn begin(name: Name) live.Scope {
        return open(name, 0, 0);
    }

    pub fn beginArg(name: Name, arg: u32) live.Scope {
        return open(name, arg, Span.flag_arg);
    }

    /// A `pool_run` or a worker share, in the shape of `Name.partFits`. It leaves `current` and the
    /// `Sums` alone.
    pub fn beginPart(name: Name, arg: u32, part: u8) live.Scope {
        assert(name.partFits(part));

        const scope: live.Scope = .{
            .start_ns = 0,
            .name = name,
            .prev = null,
            .arg = arg,
            .part = part,
            .flags = Span.flag_arg,
            .kind = .part,
        };
        if (active.load(.acquire) == 0) return scope;

        var live_scope = scope;
        live_scope.start_ns = nowNs();
        return live_scope;
    }

    pub fn current() ?Name {
        return tl_current;
    }

    pub fn record(name: Name, start: std.Io.Timestamp, end_ts: std.Io.Timestamp) void {
        if (active.load(.acquire) & gate_record == 0) return;
        pushCurrent(.{ .start_ns = tsNs(start), .end_ns = tsNs(end_ts), .name = name });
    }

    pub fn gpu(rec: GpuRecord) void {
        if (active.load(.acquire) & gate_record == 0) return;

        assert(rec.gpu_start <= rec.gpu_end);
        const r = recorder.?;
        r.gpu.push(r.gpa, rec);
    }

    pub fn enabled() bool {
        return active.load(.acquire) & gate_record != 0;
    }

    pub fn gpuLevel() GpuLevel {
        return gpu_level;
    }

    pub fn setGpuLevel(level: GpuLevel) void {
        gpu_level = level;
    }

    /// Install `sums` as the sink of this thread, or remove the installed one with null.
    pub fn installSums(sums: ?*Sums) void {
        if (sums) |s| {
            assert(tl_sums == null);
            tl_sums = s;
            _ = active.fetchAdd(gate_sink, .acq_rel);
        } else {
            assert(tl_sums != null);
            tl_sums = null;
            const old = active.fetchSub(gate_sink, .acq_rel);
            assert(old >= gate_sink);
        }
    }

    pub fn setFromEnv(io: std.Io, env: Env) error{InvalidTraceSpans}!void {
        return setup(std.heap.page_allocator, io, env);
    }

    pub fn finish() void {
        const r = recorder orelse return;
        _ = r.flush();
    }

    /// A `begin` scope. A zero gate gives an inert scope that leaves `current` alone.
    fn open(name: Name, arg: u32, flags: u8) live.Scope {
        var scope: live.Scope = .{
            .start_ns = 0,
            .name = name,
            .prev = null,
            .arg = arg,
            .part = 0,
            .flags = flags,
            .kind = .begin,
        };
        if (active.load(.acquire) == 0) return scope;

        scope.prev = tl_current;
        scope.start_ns = nowNs();
        tl_current = name;
        return scope;
    }
};

/// The site entries of a build without trace: empty inline bodies.
const off = struct {
    pub const Scope = struct {
        pub inline fn end(_: off.Scope) void {}
    };

    pub inline fn begin(_: Name) off.Scope {
        return .{};
    }

    pub inline fn beginArg(_: Name, _: u32) off.Scope {
        return .{};
    }

    pub inline fn beginPart(_: Name, _: u32, _: u8) off.Scope {
        return .{};
    }

    pub inline fn current() ?Name {
        return null;
    }

    pub inline fn record(_: Name, _: std.Io.Timestamp, _: std.Io.Timestamp) void {}

    pub inline fn gpu(_: GpuRecord) void {}

    pub inline fn enabled() bool {
        return false;
    }

    pub inline fn gpuLevel() GpuLevel {
        return .off;
    }

    pub inline fn setGpuLevel(_: GpuLevel) void {}

    pub inline fn installSums(_: ?*Sums) void {}

    /// Warn once for each trace variable that is set, because this build cannot record it.
    pub fn setFromEnv(_: std.Io, env: Env) error{InvalidTraceSpans}!void {
        const vars = [_]struct { []const u8, bool }{
            .{ "ZIGFERENCER_TRACE_OUT", env.out != null },
            .{ "ZIGFERENCER_TRACE_SECONDS", env.seconds != null },
            .{ "ZIGFERENCER_TRACE_GPU", env.gpu != null },
            .{ limits.trace_spans_env, env.spans != null },
        };
        for (vars) |v| {
            if (!v[1]) continue;

            log.warn(
                "{s} is set, but this build has no trace. Build with -Dtrace=true to record one.",
                .{v[0]},
            );
        }
    }

    pub inline fn finish() void {}
};

/// The trace storage: the ring headers of every thread, and the spans of each claimed ring.
const Recorder = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// The directory that `path` is relative to.
    dir: std.Io.Dir,
    path: []const u8,
    rings: []Ring,
    per: u32,
    claimed: std.atomic.Value(u32) = .init(0),
    threads_dropped: std.atomic.Value(u64) = .init(0),
    orphan_spans: std.atomic.Value(u64) = .init(0),
    gpu: GpuRing,
    flushed: std.atomic.Value(FlushState) = .init(.idle),

    /// The published rings, sorted, and the counts they state.
    const Collected = struct {
        tracks: [limits.span_threads_max]ThreadTrack,
        n_tracks: usize,
        n_spans: u64,
        dropped: u64,
    };

    /// `per` is the capacity of each span ring, and `gpu_per` of the GPU ring. `gpa` must be
    /// thread-safe, because `claim` and the first GPU record allocate on any thread.
    fn create(
        gpa: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
        per: u32,
        gpu_per: u32,
    ) !*Recorder {
        assert(per >= 1);
        assert(per <= limits.spans_per_thread_max);
        assert(gpu_per >= 1);

        const self = try gpa.create(Recorder);
        errdefer gpa.destroy(self);

        const rings = try gpa.alloc(Ring, limits.span_threads_max);
        for (rings) |*r| {
            r.* = .{};
        }

        self.* = .{
            .gpa = gpa,
            .io = io,
            .dir = .cwd(),
            .path = path,
            .rings = rings,
            .per = per,
            .gpu = .{ .per = gpu_per },
        };
        return self;
    }

    /// The owner threads must have stopped, because a ring frees here.
    fn destroy(self: *Recorder) void {
        self.gpu.deinit(self.gpa);
        for (self.rings) |*r| {
            if (r.spans.len > 0) {
                self.gpa.free(r.spans);
            }
        }

        self.gpa.free(self.rings);
        self.gpa.destroy(self);
    }

    /// Claim the next free ring for the calling thread and allocate its spans, or return null.
    /// A full table or a failed allocation counts one dropped thread. A failed ring stays
    /// unpublished, so the writer skips it.
    fn claim(self: *Recorder, tid: u64, name: []const u8) ?*Ring {
        const i = self.claimed.fetchAdd(1, .monotonic);
        if (i >= self.rings.len) {
            _ = self.threads_dropped.fetchAdd(1, .monotonic);
            return null;
        }

        const r = &self.rings[i];
        assert(r.len.load(.monotonic) == 0);
        assert(!r.published.load(.monotonic));
        assert(r.spans.len == 0);
        r.spans = self.gpa.alloc(Span, self.per) catch {
            _ = self.threads_dropped.fetchAdd(1, .monotonic);
            return null;
        };
        assert(r.spans.len == self.per);
        r.tid = tid;
        r.setName(name);
        r.published.store(true, .release);
        return r;
    }

    /// Stop the recorder and write the trace and the GPU folded file.
    fn flush(self: *Recorder) Flushed {
        if (self.flushed.cmpxchgStrong(.idle, .writing, .acq_rel, .acquire)) |state| {
            assert(state != .idle);

            return self.awaitWrite();
        }

        _ = active.fetchAnd(~gate_record, .release);
        self.write() catch |err| log.err("trace: writing {s} failed: {t}", .{ self.path, err });
        self.flushed.store(.done, .release);
        self.io.futexWake(FlushState, &self.flushed.raw, std.math.maxInt(u32));
        return .wrote;
    }

    /// Wait for the write of the first `flush` caller, at most `flush_wait_max_s`. A spurious wake
    /// costs one more pass.
    fn awaitWrite(self: *Recorder) Flushed {
        const deadline: std.Io.Clock.Timestamp = .fromNow(
            self.io,
            .{ .raw = .fromSeconds(flush_wait_max_s), .clock = .awake },
        );
        while (self.flushed.load(.acquire) != .done) {
            self.io.futexWaitTimeout(
                FlushState,
                &self.flushed.raw,
                .writing,
                .{ .deadline = deadline },
            ) catch break;
            const now: std.Io.Clock.Timestamp = .now(self.io, .awake);
            if (now.raw.nanoseconds >= deadline.raw.nanoseconds) break;
        }

        if (self.flushed.load(.acquire) == .done) return .waited;

        log.warn(
            "trace: the write of {s} did not end in {d} s, so this exit can cut the trace",
            .{ self.path, flush_wait_max_s },
        );
        return .timed_out;
    }

    /// Collect the published rings into sorted thread tracks.
    fn collect(self: *Recorder, out: *Collected) void {
        out.n_tracks = 0;
        out.n_spans = 0;
        out.dropped = 0;
        const n_rings = @min(self.claimed.load(.acquire), self.rings.len);
        for (self.rings[0..n_rings], 0..) |*r, i| {
            // A ring claimed but not yet published holds no span.
            if (!r.published.load(.acquire)) continue;

            // The owner writes only past `len`, so the kept prefix sorts in place.
            const spans = r.kept();
            out.dropped += r.dropped.load(.monotonic);
            if (spans.len == 0) continue;

            std.mem.sort(Span, spans, {}, lessThan);
            out.n_spans += spans.len;
            out.tracks[out.n_tracks] = .{
                .ring = @intCast(i),
                .tid = r.tid,
                .name = r.label(),
                .spans = spans,
            };
            out.n_tracks += 1;
        }

        assert(out.n_tracks <= n_rings);
    }

    fn write(self: *Recorder) !void {
        assert(self.flushed.load(.monotonic) == .writing);

        // Collect the rings, then read both clocks back to back.
        var got: Collected = undefined;
        self.collect(&got);
        const gpu_kept = self.gpu.kept();
        const gpu_records = gpu_kept.records;
        const meta: Meta = .{
            .dropped = got.dropped,
            .threads_dropped = self.threads_dropped.load(.monotonic),
            .orphan_spans = self.orphan_spans.load(.monotonic),
            .gpu_dropped = self.gpu.dropped.load(.monotonic) + gpu_kept.unready,
            .gpu_level = gpu_level,
        };
        const monotonic_ns = tsNs(std.Io.Clock.now(.awake, self.io));
        const boottime: u64 = @intCast(std.Io.Clock.now(.boot, self.io).nanoseconds);

        // Write the trace file.
        const gpu_slots = try self.gpa.alloc(GpuSlot, gpu_records.len);
        defer self.gpa.free(gpu_slots);
        const buf = try self.gpa.alloc(u8, write_buffer_bytes);
        defer self.gpa.free(buf);

        var written: Written = undefined;
        {
            const file = try self.dir.createFile(self.io, self.path, .{});
            defer file.close(self.io);

            var fw: std.Io.File.Writer = .init(file, self.io, buf);
            written = try trace_file.writeTrace(&fw.interface, .{
                .monotonic_ns = monotonic_ns,
                .boottime_ns = boottime,
                .pid = processId(),
                .threads = got.tracks[0..got.n_tracks],
                .gpu_records = gpu_records,
                .gpu_slots = gpu_slots,
                .meta = meta,
            });
            try fw.interface.flush();
        }

        // Write the GPU folded file.
        if (gpu_records.len > 0) {
            var name_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const folded = try std.fmt.bufPrint(&name_buf, "{s}.gpu.folded", .{self.path});
            const file = try self.dir.createFile(self.io, folded, .{});
            defer file.close(self.io);

            var fw: std.Io.File.Writer = .init(file, self.io, buf);
            try gpu_record.writeFolded(&fw.interface, gpu_records);
            try fw.interface.flush();
        }

        // Report the counts.
        log.info(
            "trace: wrote {s}: {d} spans on {d} threads, {d} GPU records, dropped {d} " ++
                "spans, {d} threads, {d} spans of threads with no ring, {d} GPU records, " ++
                "{d} misnested",
            .{
                self.path,
                got.n_spans,
                got.n_tracks,
                gpu_records.len,
                meta.dropped,
                meta.threads_dropped,
                meta.orphan_spans,
                meta.gpu_dropped,
                written.misnested,
            },
        );
    }
};

/// The flush progress of the recorder. The futex word of a waiting `flush`.
const FlushState = enum(u32) { idle, writing, done };

/// How one `flush` call ended.
const Flushed = enum { wrote, waited, timed_out };

/// The ring of GPU records.
const GpuRing = struct {
    per: u32,
    block: std.atomic.Value(?[*]align(@alignOf(GpuRecord)) u8) = .init(null),
    failed: std.atomic.Value(bool) = .init(false),
    next: std.atomic.Value(u64) = .init(0),
    /// The pushes past the capacity. `kept` adds the unwritten indexes.
    dropped: std.atomic.Value(u64) = .init(0),

    /// The published records and the claimed indexes that hold none.
    const Kept = struct {
        records: []const GpuRecord,
        unready: u64,
    };

    fn blockBytes(per: u32) usize {
        return @as(usize, per) * (@sizeOf(GpuRecord) + @sizeOf(std.atomic.Value(bool)));
    }

    fn recordsOf(self: *const GpuRing, b: [*]align(@alignOf(GpuRecord)) u8) []GpuRecord {
        return @as([*]GpuRecord, @ptrCast(b))[0..self.per];
    }

    fn readyOf(self: *const GpuRing, b: [*]align(@alignOf(GpuRecord)) u8) []std.atomic.Value(bool) {
        const flags = b + @as(usize, self.per) * @sizeOf(GpuRecord);
        return @as([*]std.atomic.Value(bool), @ptrCast(flags))[0..self.per];
    }

    fn deinit(self: *GpuRing, gpa: std.mem.Allocator) void {
        const b = self.block.load(.acquire) orelse return;
        gpa.free(b[0..blockBytes(self.per)]);
        self.block.store(null, .monotonic);
    }

    /// The block, allocated at the first call. Null after a failed allocation.
    fn storage(self: *GpuRing, gpa: std.mem.Allocator) ?[*]align(@alignOf(GpuRecord)) u8 {
        if (self.block.load(.acquire)) |b| return b;
        if (self.failed.load(.monotonic)) return null;

        const bytes = blockBytes(self.per);
        const fresh = gpa.alignedAlloc(u8, .of(GpuRecord), bytes) catch {
            self.failed.store(true, .monotonic);
            return null;
        };
        for (self.readyOf(fresh.ptr)) |*f| {
            f.* = .init(false);
        }

        if (self.block.cmpxchgStrong(null, fresh.ptr, .acq_rel, .acquire)) |winner| {
            gpa.free(fresh);
            return winner.?;
        }

        return fresh.ptr;
    }

    fn push(self: *GpuRing, gpa: std.mem.Allocator, rec: GpuRecord) void {
        const i = self.next.fetchAdd(1, .monotonic);
        if (i >= self.per) {
            _ = self.dropped.fetchAdd(1, .monotonic);
            return;
        }

        const b = self.storage(gpa) orelse return;
        self.recordsOf(b)[@intCast(i)] = rec;
        self.readyOf(b)[@intCast(i)].store(true, .release);
    }

    /// The published prefix. A record still in flight or never written ends
    /// the prefix, and every claimed index past the prefix counts as unready.
    fn kept(self: *GpuRing) Kept {
        const claimed = @min(self.next.load(.acquire), self.per);
        const b = self.block.load(.acquire) orelse return .{ .records = &.{}, .unready = claimed };
        const ready = self.readyOf(b);
        var n: usize = 0;
        while (n < claimed and ready[n].load(.acquire)) {
            n += 1;
        }

        assert(n <= claimed);
        return .{ .records = self.recordsOf(b)[0..n], .unready = claimed - n };
    }
};

// -- state ------------------------------------------------------------------

/// The gate of every span site: `gate_record` while the recorder records, plus `gate_sink` per
/// installed `Sums`. The recording bit turns on after `recorder` is set.
var active: std.atomic.Value(u32) = .init(0);

/// The GPU timestamp level. `setFromEnv` sets it before any runner exists.
var gpu_level: GpuLevel = .off;

/// The process recorder. It is set before the gate turns on and never freed, so it lives to exit.
var recorder: ?*Recorder = null;

threadlocal var tl_ring: ?*Ring = null;
threadlocal var tl_no_ring: bool = false;
threadlocal var tl_name_buf: [16]u8 = @splat(0);
threadlocal var tl_name_len: u8 = 0;
/// The innermost open `begin` scope of this thread.
threadlocal var tl_current: ?Name = null;
threadlocal var tl_sums: ?*Sums = null;

// -- api --------------------------------------------------------------------

/// Open a span. `Scope.end` closes it, and `current` names it until then.
pub const begin = if (compiled) live.begin else off.begin;

/// Open a span with one arg: a layer index, a chunk of tokens, or a row count.
pub const beginArg = if (compiled) live.beginArg else off.beginArg;

/// Open a pool share or a `pool_run`. `part` is the worker index or the participant count.
pub const beginPart = if (compiled) live.beginPart else off.beginPart;

/// The innermost open `begin` scope of this thread. It is null while the gate is zero, because an
/// inert scope leaves it alone.
pub const current = if (compiled) live.current else off.current;

/// Record a span from two timestamps that the caller already read.
pub const record = if (compiled) live.record else off.record;

/// Hand over one GPU record.
pub const gpu = if (compiled) live.gpu else off.gpu;

/// True while the recorder records.
pub const enabled = if (compiled) live.enabled else off.enabled;

/// The GPU timestamp level of this process.
pub const gpuLevel = if (compiled) live.gpuLevel else off.gpuLevel;

/// Set the GPU timestamp level. Only startup code calls it, before any GPU runner exists, because
/// a runner sizes its query pools from it.
pub const setGpuLevel = if (compiled) live.setGpuLevel else off.setGpuLevel;

/// Install a `Sums` sink on this thread, or remove it with null.
pub const installSums = if (compiled) live.installSums else off.installSums;

/// Read the four trace variables. Call it once in `main`, before any thread or runner exists. A
/// bad ring capacity returns an error, because a silent default drops the spans that the operator
/// asked to keep. A build without trace only warns.
pub const setFromEnv = if (compiled) live.setFromEnv else off.setFromEnv;

/// Write the trace if the window has not written it already. `main` calls it at exit.
pub const finish = if (compiled) live.finish else off.finish;

pub const tickSpan = gpu_record.tickSpan;
pub const lessThan = trace_file.lessThan;

/// Name the calling thread, for the trace track and for the kernel `comm`
/// that perf prints. Call it before the first span of the thread.
pub fn nameThread(name: []const u8) void {
    const n: u8 = @intCast(@min(name.len, tl_name_buf.len - 1));
    @memcpy(tl_name_buf[0..n], name[0..n]);
    tl_name_len = n;
    if (builtin.os.tag == .linux) {
        var z: [16:0]u8 = @splat(0);
        @memcpy(z[0..n], name[0..n]);
        // A name that the kernel refuses changes only the perf label.
        _ = std.posix.prctl(.SET_NAME, .{@intFromPtr(&z)}) catch {};
    }

    if (tl_ring) |r| {
        r.setName(name[0..n]);
    }
}

/// Nanoseconds of a timestamp on the `.awake` clock.
pub fn tsNs(t: std.Io.Timestamp) u64 {
    assert(t.nanoseconds > 0);

    return @intCast(t.nanoseconds);
}

// -- helpers ----------------------------------------------------------------

fn setup(gpa: std.mem.Allocator, io: std.Io, env: Env) error{InvalidTraceSpans}!void {
    assert(active.load(.monotonic) & gate_record == 0);
    assert(recorder == null);

    // Parse the ring capacity and the GPU level.
    const capacity: u32 = if (env.spans) |text| try limits.parseTraceSpans(
        text,
    ) else limits.spans_per_thread;

    if (env.gpu) |g| {
        if (std.meta.stringToEnum(GpuLevel, g)) |level| {
            gpu_level = level;
        } else {
            log.warn(
                "ZIGFERENCER_TRACE_GPU={s} is not off, submit, or phases, " ++
                    "so GPU timestamps stay off",
                .{g},
            );
        }
    }

    // Parse the capture window.
    const path = env.out orelse return;
    if (path.len == 0) return;

    var seconds: ?u32 = null;
    if (env.seconds) |text| {
        seconds = std.fmt.parseInt(u32, text, 10) catch blk: {
            log.warn(
                "ZIGFERENCER_TRACE_SECONDS={s} is not a whole number of seconds, " ++
                    "so the trace runs to exit",
                .{text},
            );
            break :blk null;
        };
    }

    // Start the recorder and the window thread.
    const r = Recorder.create(gpa, io, path, capacity, limits.trace_gpu_records) catch |err| {
        log.warn("trace: no storage for the span rings ({t}), so tracing stays off", .{err});
        return;
    };
    recorder = r;
    nameThread("main");
    _ = active.fetchOr(gate_record, .release);
    if (seconds) |s| {
        const t = std.Thread.spawn(.{}, windowMain, .{ io, s }) catch |err| {
            log.warn("trace: no window thread ({t}), so the trace runs to exit", .{err});
            return;
        };
        t.detach();
    }

    log.info("trace: recording to {s}, {d} spans per thread", .{ path, capacity });
}

fn pushCurrent(s: Span) void {
    const ring = tl_ring orelse blk: {
        const rec = recorder.?;
        if (tl_no_ring) {
            _ = rec.orphan_spans.fetchAdd(1, .monotonic);
            return;
        }

        const r = rec.claim(currentTid(), tl_name_buf[0..tl_name_len]) orelse {
            tl_no_ring = true;
            _ = rec.orphan_spans.fetchAdd(1, .monotonic);
            return;
        };
        tl_ring = r;
        break :blk r;
    };
    ring.push(s);
}

fn nowNs() u64 {
    return tsNs(core_log.now());
}

fn currentTid() u64 {
    return @intCast(std.Thread.getCurrentId());
}

fn processId() u32 {
    if (builtin.os.tag == .linux) return @intCast(std.os.linux.getpid());
    return 1;
}

fn windowMain(io: std.Io, seconds: u32) void {
    io.sleep(.fromSeconds(seconds), .awake) catch {};
    live.finish();
}

// -- tests ------------------------------------------------------------------

const prop = @import("utils/prop.zig");
const testing = std.testing;

/// A recorder for the tests of other modules. The tests start it, run threads, join them, then
/// call `finish` to read the tracks. Only one capture can exist at a time.
pub const TestCapture = struct {
    rec: *Recorder,
    got: Recorder.Collected = undefined,

    const spans_per_thread = 4096;

    pub fn start(gpa: std.mem.Allocator) !*TestCapture {
        assert(recorder == null);
        assert(active.load(.monotonic) & gate_record == 0);

        const self = try gpa.create(TestCapture);
        errdefer gpa.destroy(self);

        self.* = .{ .rec = try Recorder.create(
            gpa,
            std.testing.io,
            "unused.pftrace",
            spans_per_thread,
            1,
        ) };
        recorder = self.rec;
        _ = active.fetchOr(gate_record, .release);
        return self;
    }

    /// Stop recording and return the thread tracks. Every thread that recorded must have ended.
    pub fn finish(self: *TestCapture) []const ThreadTrack {
        _ = active.fetchAnd(~gate_record, .release);
        self.rec.collect(&self.got);
        return self.got.tracks[0..self.got.n_tracks];
    }

    pub fn deinit(self: *TestCapture) void {
        const gpa = self.rec.gpa;
        if (active.load(.monotonic) & gate_record != 0) {
            _ = active.fetchAnd(~gate_record, .release);
        }

        recorder = null;
        tl_ring = null;
        tl_no_ring = false;
        self.rec.destroy();
        gpa.destroy(self);
    }
};

const perfetto = @import("utils/perfetto.zig");

test {
    _ = gpu_record;
    _ = perfetto;
    _ = trace_file;
}

/// A random push sequence against one small ring.
const RingCase = struct {
    capacity: u32,
    pushes: [64]Span,
    n: u32,
};

fn genRing(_: void, random: std.Random, size: u32) RingCase {
    var c: RingCase = .{
        .capacity = 1 + random.uintLessThan(u32, 16),
        .pushes = undefined,
        .n = @min(size, 64),
    };
    for (c.pushes[0..c.n], 0..) |*s, i| {
        const start = random.uintLessThan(u64, 1 << 40) + 1;
        s.* = .{
            .start_ns = start,
            .end_ns = start + random.uintLessThan(u64, 1 << 20),
            .name = @enumFromInt(i % 8),
        };
    }

    return c;
}

fn checkRing(_: void, c: RingCase) anyerror!void {
    var storage: [16]Span = undefined;
    var ring: Ring = .{ .spans = storage[0..c.capacity] };
    for (c.pushes[0..c.n]) |s| {
        ring.push(s);
    }

    const kept_n = @min(c.n, c.capacity);
    try testing.expectEqual(@as(usize, kept_n), ring.kept().len);
    try testing.expectEqual(@as(u64, c.n - kept_n), ring.dropped.load(.monotonic));
    // A kept span is the span pushed at its index, unchanged by later pushes.
    for (ring.kept(), c.pushes[0..kept_n]) |got, want| {
        try testing.expectEqual(want.start_ns, got.start_ns);
        try testing.expectEqual(want.end_ns, got.end_ns);
        try testing.expectEqual(want.name, got.name);
    }
}

test "ring: an overflow counts the drop and keeps every kept span intact" {
    try prop.forAll(.{ .site = 0x5350_0001 }, {}, RingCase, genRing, checkRing);
}

test "scopes: current follows begin and end, and only op begin scopes add to the sums" {
    var sums: Sums = .{};
    const norm: Name = .ofOps(.initOne(.attn_norm));
    const attention: Name = .ofOps(.initOne(.attention));
    live.installSums(&sums);
    {
        defer live.installSums(null);

        try testing.expect(live.current() == null);
        const outer = live.begin(.decode_step);
        try testing.expectEqual(Name.decode_step, live.current().?);
        const op = live.beginArg(norm, 3);
        try testing.expectEqual(norm, live.current().?);
        // A share keeps the open name and never counts its op.
        const share = live.beginPart(attention, 7, 2);
        try testing.expectEqual(norm, live.current().?);
        share.endAt(share.start_ns + 40);
        try testing.expectEqual(norm, live.current().?);
        op.endAt(op.start_ns + 50);
        try testing.expectEqual(Name.decode_step, live.current().?);
        outer.endAt(outer.start_ns + 90);
        try testing.expect(live.current() == null);
    }

    // Only the op scope counts, and only under its own key.
    var total: u64 = 0;
    for (sums.ns) |ns| {
        total += ns;
    }

    try testing.expectEqual(@as(u64, 50), sums.get(.ofOp(.attn_norm)));
    try testing.expectEqual(@as(u64, 50), total);
    // The removal gives the gate back, so a later scope is inert.
    try testing.expectEqual(@as(u32, 0), active.load(.acquire));
    try testing.expectEqual(@as(u64, 0), live.begin(.ofOps(.initOne(.rope))).start_ns);
}

/// Set or clear the recording bit by hand, as `setup` and `flush` do. No recorder exists, so a
/// push under the bit fails the test on the null `recorder`.
fn setRecordBit(on: bool) void {
    if (on) {
        _ = active.fetchOr(gate_record, .acq_rel);
    } else {
        _ = active.fetchAnd(~gate_record, .acq_rel);
    }
}

test "gate: a scope that the flush outlives restores current and pushes nothing" {
    setRecordBit(true);
    const s = live.begin(.decode_step);
    try testing.expect(s.start_ns != 0);
    try testing.expectEqual(Name.decode_step, live.current().?);
    setRecordBit(false);
    s.endAt(s.start_ns + 10);
    try testing.expect(live.current() == null);
    try testing.expectEqual(@as(u32, 0), active.load(.acquire));
}

test "gate: an inert scope ended after the gate rises does nothing" {
    var sums: Sums = .{};
    const s = live.begin(.ofOps(.initOne(.rope)));
    try testing.expectEqual(@as(u64, 0), s.start_ns);
    live.installSums(&sums);
    s.end();
    live.installSums(null);
    try testing.expect(live.current() == null);
    try testing.expectEqual(@as(u64, 0), sums.get(.ofOp(.rope)));
    try testing.expectEqual(@as(u32, 0), active.load(.acquire));
}

test "gate: a sink installed between the open and the end counts the span" {
    // Another sink holds the gate open, so the scope is live without a recorder.
    _ = active.fetchAdd(gate_sink, .acq_rel);
    defer _ = active.fetchSub(gate_sink, .acq_rel);

    var sums: Sums = .{};
    const s = live.begin(.ofOps(.initOne(.swiglu)));
    live.installSums(&sums);
    s.endAt(s.start_ns + 30);
    live.installSums(null);
    try testing.expectEqual(@as(u64, 30), sums.get(.ofOp(.swiglu)));
    try testing.expect(live.current() == null);
}

test "gate: an inner scope opened after the gate falls stays inert inside a live outer one" {
    _ = active.fetchAdd(gate_sink, .acq_rel);
    const outer = live.begin(.decode_step);
    _ = active.fetchSub(gate_sink, .acq_rel);
    const inner = live.begin(.sample);
    try testing.expectEqual(@as(u64, 0), inner.start_ns);
    try testing.expectEqual(Name.decode_step, live.current().?);
    inner.end();
    try testing.expectEqual(Name.decode_step, live.current().?);
    outer.endAt(outer.start_ns + 5);
    try testing.expect(live.current() == null);
    try testing.expectEqual(@as(u32, 0), active.load(.acquire));
}

test "off: every entry compiles with a runtime name and holds no state" {
    try testing.expectEqual(@as(usize, 0), @sizeOf(off.Scope));
    var name: Name = .decode_step;
    _ = &name;
    off.begin(name).end();
    off.beginArg(name, 3).end();
    off.beginPart(name, 1, 2).end();
    try testing.expect(off.current() == null);
    off.record(name, .{ .nanoseconds = 1 }, .{ .nanoseconds = 2 });
    off.gpu(test_record);
    try testing.expect(!off.enabled());
    try testing.expectEqual(GpuLevel.off, off.gpuLevel());
    off.setGpuLevel(.submit);
    try testing.expectEqual(GpuLevel.off, off.gpuLevel());
    var sums: Sums = .{};
    off.installSums(&sums);
    off.installSums(null);
    try off.setFromEnv(testing.io, .{});
    off.finish();
    try testing.expect(tl_ring == null);
}

test "off path: a live site with the gate zero allocates nothing and records nothing" {
    // An unset output path leaves the recorder off and takes no storage.
    var failing: std.testing.FailingAllocator = .init(testing.allocator, .{});
    try setup(failing.allocator(), testing.io, .{});
    try testing.expectEqual(@as(usize, 0), failing.allocations);
    try testing.expect(!live.enabled());
    const s = live.begin(.decode_step);
    try testing.expectEqual(@as(u64, 0), s.start_ns);
    try testing.expect(live.current() == null);
    s.end();
    live.beginArg(.layer_forward, 2).end();
    live.beginPart(.pool_run, 1, 3).end();
    live.record(.gpu_wait, .{ .nanoseconds = 1 }, .{ .nanoseconds = 2 });
    live.gpu(test_record);
    try testing.expect(tl_ring == null);
    try testing.expect(recorder == null);
    try testing.expectEqual(@as(usize, 0), failing.allocations);
}

test "setup: a ring capacity outside the bound refuses before any allocation" {
    var failing: std.testing.FailingAllocator = .init(testing.allocator, .{});
    const bad = [_][]const u8{ "", "0", "4194305", "abc", "-1" };
    for (bad) |text| {
        try testing.expectError(
            error.InvalidTraceSpans,
            setup(failing.allocator(), testing.io, .{ .out = "unused.pftrace", .spans = text }),
        );
        // The capacity also refuses with the recorder off.
        try testing.expectError(
            error.InvalidTraceSpans,
            setup(failing.allocator(), testing.io, .{ .spans = text }),
        );
    }

    try testing.expectEqual(@as(usize, 0), failing.allocations);
    try testing.expect(!enabled());
    try testing.expect(recorder == null);
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

/// The allocations of `Recorder.create`: the recorder and the ring table.
const create_allocations: usize = 2;

test "recorder: create takes no span or GPU storage, and each claim allocates one ring" {
    const per: u32 = 3;
    var failing: std.testing.FailingAllocator = .init(testing.allocator, .{});
    const r = try Recorder.create(failing.allocator(), testing.io, "unused.pftrace", per, per);
    defer r.destroy();

    try testing.expectEqual(create_allocations, failing.allocations);
    for (r.rings) |*ring| {
        try testing.expectEqual(@as(usize, 0), ring.spans.len);
    }

    try testing.expect(r.gpu.block.load(.acquire) == null);
    try testing.expectEqual(@as(usize, 0), r.gpu.kept().records.len);

    var i: u32 = 0;
    while (i < limits.span_threads_max) : (i += 1) {
        const ring = r.claim(i, "t").?;
        try testing.expect(ring.published.load(.acquire));
        try testing.expectEqual(@as(usize, per), ring.spans.len);
        try testing.expectEqual(create_allocations + i + 1, failing.allocations);
    }

    // A claim past the table allocates nothing and counts one dropped thread.
    try testing.expect(r.claim(99, "t") == null);
    try testing.expectEqual(create_allocations + limits.span_threads_max, failing.allocations);
    try testing.expectEqual(@as(u64, 1), r.threads_dropped.load(.monotonic));

    // A full ring, span or GPU, keeps `per` entries and counts each drop.
    // The first GPU record allocates the GPU ring, and no later one does.
    const s: Span = .{ .start_ns = 1, .end_ns = 2, .name = .sample };
    i = 0;
    while (i < per + 2) : (i += 1) {
        r.rings[0].push(s);
        r.gpu.push(r.gpa, test_record);
        try testing.expectEqual(
            create_allocations + limits.span_threads_max + 1,
            failing.allocations,
        );
    }

    try testing.expectEqual(@as(usize, per), r.rings[0].kept().len);
    try testing.expectEqual(@as(u64, 2), r.rings[0].dropped.load(.monotonic));
    const kept = r.gpu.kept();
    try testing.expectEqual(@as(usize, per), kept.records.len);
    try testing.expectEqual(@as(u64, 0), kept.unready);
    try testing.expectEqual(@as(u64, 2), r.gpu.dropped.load(.monotonic));
}

test "recorder: a failed ring allocation drops one thread and the next claim succeeds" {
    var failing: std.testing.FailingAllocator = .init(
        testing.allocator,
        .{ .fail_index = create_allocations + 1 },
    );
    const r = try Recorder.create(failing.allocator(), testing.io, "unused.pftrace", 2, 2);
    defer r.destroy();

    try testing.expect(r.claim(10, "a") == &r.rings[0]);
    try testing.expect(r.claim(11, "b") == null);
    try testing.expectEqual(@as(u64, 1), r.threads_dropped.load(.monotonic));
    try testing.expect(!r.rings[1].published.load(.acquire));
    try testing.expectEqual(@as(usize, 0), r.rings[1].spans.len);
    // The failing allocator fails every later call too, so lift the limit.
    failing.fail_index = std.math.maxInt(usize);
    const next = r.claim(12, "c").?;
    try testing.expect(next == &r.rings[2]);
    try testing.expect(next.published.load(.acquire));
    try testing.expectEqual(@as(u64, 12), next.tid);
    try testing.expectEqual(@as(u64, 1), r.threads_dropped.load(.monotonic));
}

test "gpu ring: a failed allocation counts its record and every later record as dropped" {
    var failing: std.testing.FailingAllocator = .init(testing.allocator, .{ .fail_index = 0 });
    var ring: GpuRing = .{ .per = 4 };
    defer ring.deinit(failing.allocator());

    ring.push(failing.allocator(), test_record);
    // Lift the limit: the ring stays off and allocates nothing more.
    failing.fail_index = std.math.maxInt(usize);
    var i: u32 = 0;
    while (i < 4) : (i += 1) {
        ring.push(failing.allocator(), test_record);
    }

    try testing.expectEqual(@as(usize, 0), failing.allocations);
    const kept = ring.kept();
    try testing.expectEqual(@as(usize, 0), kept.records.len);
    try testing.expectEqual(@as(u64, 4), kept.unready);
    try testing.expectEqual(@as(u64, 1), ring.dropped.load(.monotonic));
}

/// Several threads push their first GPU records at the same moment.
const GpuRaceCase = struct {
    threads: u32,
    each: u32,
};

const race_threads_max = 8;
const race_each_max = 32;

fn genGpuRace(_: void, random: std.Random, size: u32) GpuRaceCase {
    return .{
        .threads = 2 + random.uintLessThan(u32, race_threads_max - 1),
        .each = 1 + random.uintLessThan(u32, @as(u32, @min(size, race_each_max - 1)) + 1),
    };
}

fn pushRace(ring: *GpuRing, gate: *std.atomic.Value(u32), threads: u32, t: u32, each: u32) void {
    _ = gate.fetchAdd(1, .acq_rel);
    // Each thread spins until every thread runs, so the first pushes race.
    while (gate.load(.acquire) < threads) {
        std.atomic.spinLoopHint();
    }

    var k: u32 = 0;
    while (k < each) : (k += 1) {
        var rec = test_record;
        rec.gpu_start = @as(u64, t) * race_each_max + k;
        rec.gpu_end = rec.gpu_start;
        ring.push(testing.allocator, rec);
    }
}

fn checkGpuRace(_: void, c: GpuRaceCase) anyerror!void {
    var ring: GpuRing = .{ .per = c.threads * c.each };
    defer ring.deinit(testing.allocator);

    var gate: std.atomic.Value(u32) = .init(0);
    var handles: [race_threads_max]std.Thread = undefined;
    var spawned: u32 = 0;
    defer for (handles[0..spawned]) |h| h.join();

    while (spawned < c.threads) : (spawned += 1) {
        handles[spawned] = std.Thread.spawn(
            .{},
            pushRace,
            .{ &ring, &gate, c.threads, spawned, c.each },
        ) catch |err| {
            // Release the spinning threads, so the deferred joins return.
            gate.store(std.math.maxInt(u32), .release);
            return err;
        };
    }

    for (handles[0..spawned]) |h| {
        h.join();
    }

    spawned = 0;
    // The joins order every push before these reads.
    const kept = ring.kept();
    try testing.expectEqual(@as(u64, 0), kept.unready);
    try testing.expectEqual(@as(u64, 0), ring.dropped.load(.monotonic));
    try testing.expectEqual(@as(usize, c.threads * c.each), kept.records.len);
    var seen: [race_threads_max * race_each_max]bool = @splat(false);
    for (kept.records) |rec| {
        const at: usize = @intCast(rec.gpu_start);
        try testing.expect(!seen[at]);
        seen[at] = true;
    }
}

test "gpu ring: racing first pushes allocate one block and keep every record" {
    try prop.forAll(
        .{ .site = 0x5350_0004, .cases = 48, .max_size = race_each_max - 1 },
        {},
        GpuRaceCase,
        genGpuRace,
        checkGpuRace,
    );
}

/// One flushing thread: flush, then read the trace back at once.
fn flushAndDecode(r: *Recorder, result: *Flushed, whole: *bool) void {
    result.* = r.flush();
    const bytes = r.dir.readFileAlloc(
        r.io,
        r.path,
        testing.allocator,
        .limited(1 << 22),
    ) catch return;
    defer testing.allocator.free(bytes);

    // The trace is whole when every packet decodes and the last one is the `trace_meta` instant.
    const interned = testing.allocator.create(perfetto.TestInterned) catch return;
    defer testing.allocator.destroy(interned);
    interned.* = .{};
    var reader: perfetto.TestReader = .init(bytes);
    var last_instant = false;
    while (reader.next(interned) catch return) |p| {
        const e = p.event orelse continue;
        last_instant = e.type == @intFromEnum(perfetto.EventType.instant);
    }

    whole.* = last_instant;
}

test "flush: of two racing callers one writes, and both return after the write ends" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const per: u32 = 4096;
    var round: u32 = 0;
    while (round < 8) : (round += 1) {
        const r = try Recorder.create(testing.allocator, testing.io, "trace.pftrace", per, per);
        defer r.destroy();

        r.dir = tmp.dir;
        // A full ring makes the write long enough for the two calls to overlap.
        const ring = r.claim(1, "main").?;
        var i: u64 = 0;
        while (i < per) : (i += 1) {
            ring.push(.{ .start_ns = i + 1, .end_ns = i + 2, .name = .sample });
        }

        r.gpu.push(r.gpa, test_record);

        var results: [2]Flushed = undefined;
        var whole: [2]bool = @splat(false);
        const a = try std.Thread.spawn(.{}, flushAndDecode, .{ r, &results[0], &whole[0] });
        const b = std.Thread.spawn(
            .{},
            flushAndDecode,
            .{ r, &results[1], &whole[1] },
        ) catch |err| {
            a.join();
            return err;
        };
        a.join();
        b.join();
        var wrote: u32 = 0;
        for (results) |res| {
            try testing.expect(res != .timed_out);
            if (res == .wrote) {
                wrote += 1;
            }
        }

        try testing.expectEqual(@as(u32, 1), wrote);
        try testing.expect(whole[0]);
        try testing.expect(whole[1]);
        try testing.expectEqual(FlushState.done, r.flushed.load(.acquire));
    }
}
