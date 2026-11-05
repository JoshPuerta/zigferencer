//! The span recorder: a timeline of the engine boundaries for one process,
//! written as Chrome Trace Event JSON for ui.perfetto.dev.
//!
//! `$ZIGFERENCER_TRACE_OUT=path` turns the recorder on at startup. A span site
//! then reads the monotonic clock twice and pushes one `Span` into the ring of
//! its thread. With the recorder off, a site costs one atomic load and one
//! branch, and it touches no thread-local state. `$ZIGFERENCER_TRACE_SECONDS=N`
//! bounds the capture window. The recorder writes the trace when the window
//! ends or at exit, whichever comes first. `$ZIGFERENCER_TRACE_SPANS=N` sets
//! the capacity of every ring, so a long per-op capture keeps its spans.
//!
//! Two levels, as in `log.zig`. The boundary spans (`Name.isHot` false) compile
//! in every build. The per-op spans compile only under `-Dtrace`, through
//! `beginHot`.
//!
//! Storage is the dynamic shell of this module. Nothing is allocated when the
//! recorder stays off. Each thread claims one fixed ring from a table bounded
//! by `limits.span_threads_max` and allocates the spans of that ring at its
//! first span. A full ring drops the span and counts it. One more fixed ring
//! holds the GPU submit records, which the GPU runners hand over after each
//! wait. That ring allocates its records at the first GPU record, so a trace
//! with no GPU work holds no GPU storage.
//!
//! The clock is `.awake` (`CLOCK_MONOTONIC` on Linux), the clock of
//! `perf record -k mono`, so a span file and a perf file share one time base.
//! A GPU clock maps to that base only through the submit and wait points of
//! each record. The device counter is not a stable clock across submits, so
//! the writer places each record on its own host times (see `place`).

const std = @import("std");
const builtin = @import("builtin");
const limits = @import("limits.zig");
const core_log = @import("log.zig");

const assert = std.debug.assert;
const log = std.log.scoped(.span);

// -- constants --------------------------------------------------------------

pub const gpu_phase_count = std.meta.fields(GpuPhase).len;

/// Records that the local offset of one record reads: the record itself and
/// the records after it.
pub const anchor_window: u32 = 16;

// -- types ------------------------------------------------------------------

/// The closed span vocabulary. The JSON event name is the tag name.
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
    // The per-op spans. Only `-Dtrace` builds record them.
    pool_run,
    run_share,
    worker_wait,
    layer_forward,
    host_prefix,
    // One span per `cpu_profile.ProfileCat`, in its order.
    op_embed,
    op_rmsnorm,
    op_gdn_proj,
    op_gdn_step,
    op_gdn_out,
    op_qkv_proj,
    op_attn_proj,
    op_rope,
    op_qk_norm_rope,
    op_kv_commit,
    op_attention,
    op_attn_gate,
    op_o_proj,
    op_gate_up_proj,
    op_swiglu,
    op_down_proj,
    op_residual,
    op_lm_head,

    /// True for the spans that only `-Dtrace` builds record.
    pub fn isHot(self: Name) bool {
        return @intFromEnum(self) >= @intFromEnum(Name.pool_run);
    }
};

/// One closed span. The times are nanoseconds on the `.awake` clock.
pub const Span = extern struct {
    start_ns: u64,
    end_ns: u64,
    name: Name,
    pad: [6]u8 = @splat(0),

    comptime {
        assert(@sizeOf(Span) == 24);
        // The ring ceiling in `limits` prices one thread ring at 96 MiB.
        assert(@as(u64, limits.spans_per_thread_max) * @sizeOf(Span) == 96 << 20);
    }
};

/// The GPU timestamp level of the process, from `$ZIGFERENCER_TRACE_GPU`.
/// `submit` writes two timestamps per command buffer, start and end. `phases`
/// writes every phase mark of the record functions.
pub const GpuLevel = enum(u8) { off, submit, phases };

/// The runner that recorded a GPU command buffer. `staging` is the one-shot
/// submit of `kernels.submitOnce`, which no runner owns.
pub const GpuRunner = enum(u8) { llama_fused, qwen_fused, llama_hybrid, qwen_hybrid, staging };

/// What a GPU command buffer did. The last four are the one-shot submit
/// labels of `kernels.SubmitLabel`.
pub const GpuLabel = enum(u8) { decode, decode_batch, prefill_chunk, copy, fill, clone, dispatch };

/// The op groups of the GPU phase marks. The matmul phases name the
/// projection group, so a kernel change reads its own time.
pub const GpuPhase = enum(u8) {
    /// GDN input projections: qkv, z, alpha, beta.
    matmul_gdn_in,
    /// Attention input projections: q, k, v.
    matmul_attn_in,
    /// Output projections: ssm_out and o.
    matmul_out,
    /// FFN gate and up.
    matmul_ffn_up,
    /// FFN down.
    matmul_ffn_down,
    gdn_conv,
    gdn_scan,
    qknorm,
    rope,
    attn,
    elementwise,
    norm,
    lmhead,
    copy,
};

/// One GPU command buffer, read back after the wait its path already does.
/// The device ticks stay raw. `period_ns` converts them. The host times are
/// nanoseconds on the `.awake` clock: before the submit and after the wait.
pub const GpuRecord = struct {
    runner: GpuRunner,
    label: GpuLabel,
    gpu_start: u64,
    gpu_end: u64,
    period_ns: f64,
    host_submit_ns: u64,
    host_wait_ns: u64,
    /// The phase sums of a `phases` record. A `submit` record has none.
    phases: ?[gpu_phase_count]u64 = null,
    /// The intervals where the device counter stepped back. Each one counts
    /// as zero ticks in `gpu_end` and in `phases`.
    backsteps: u32 = 0,

    comptime {
        // The ring stores one ready flag beside each record, so one entry
        // costs 169 bytes. `docs/profiling.md` prices the GPU ring with it.
        assert(@sizeOf(GpuRecord) == 168);
    }

    fn gpuStartNs(self: GpuRecord) i64 {
        return ticksToNs(self.gpu_start, self.period_ns);
    }

    fn gpuEndNs(self: GpuRecord) i64 {
        return ticksToNs(self.gpu_end, self.period_ns);
    }
};

/// Where one GPU record sits on the host clock, in nanoseconds.
pub const Placement = struct {
    start_ns: i64,
    end_ns: i64,
    /// How far the clamp moved the record into its host window.
    clamp_ns: u64,
};

/// One fixed ring of spans, written by its owner thread alone. A full ring
/// drops the span and counts it. A kept span never changes after its push, so
/// the writer can read `kept` while the owner still pushes.
pub const Ring = struct {
    /// Empty until `Recorder.claim` allocates the ring for its thread.
    spans: []Span = &.{},
    len: std.atomic.Value(u32) = .init(0),
    dropped: std.atomic.Value(u64) = .init(0),
    tid: u64 = 0,
    name_buf: [16]u8 = @splat(0),
    name_len: std.atomic.Value(u8) = .init(0),
    /// Set with release order after the claim writes `tid` and the name, so
    /// the writer reads them only through an acquire load of this flag.
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

/// One thread track of the JSON output. `spans` must be sorted by `lessThan`.
pub const ThreadTrack = struct {
    tid: u64,
    name: []const u8,
    spans: []const Span,
};

/// The counts the trace metadata states.
pub const Meta = struct {
    dropped: u64 = 0,
    threads_dropped: u64 = 0,
    /// The spans of the threads that hold no ring.
    orphan_spans: u64 = 0,
    gpu_dropped: u64 = 0,
    gpu_level: GpuLevel = .off,
};

/// One open span. `start_ns == 0` means the recorder was off at `begin`, and
/// `end` then does nothing.
pub const Scope = struct {
    start_ns: u64,
    name: Name,

    pub inline fn end(self: Scope) void {
        if (self.start_ns == 0) return;
        endSlow(self);
    }
};

/// The scope of a per-op span: a `Scope` under `-Dtrace`, and an empty value
/// otherwise, so a release build compiles the site out.
pub const HotScope = if (core_log.hot) Scope else struct {
    pub inline fn end(_: @This()) void {}
};

/// The startup inputs of the recorder, read from the environment by `main`.
pub const Env = struct {
    out: ?[]const u8 = null,
    seconds: ?[]const u8 = null,
    gpu: ?[]const u8 = null,
    spans: ?[]const u8 = null,
};

/// The device times of one command buffer, in raw ticks.
pub const TickSpan = struct {
    start: u64,
    end: u64,
    backsteps: u32,
};

/// The trace storage. The table holds the ring headers of every thread, and
/// each claim allocates the spans of one ring.
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

    /// `per` is the capacity of each span ring and of the GPU ring. One
    /// variable sets both, because both defaults match and the metadata counts
    /// the drops of each ring apart. `gpa` must be thread-safe, because
    /// `claim` and the first GPU record allocate on any thread.
    fn create(gpa: std.mem.Allocator, io: std.Io, path: []const u8, per: u32) !*Recorder {
        assert(per >= 1);
        assert(per <= limits.spans_per_thread_max);

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
            .gpu = .{ .per = per },
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

    /// Claim the next free ring for the calling thread and allocate its
    /// spans, or return null. A full table returns null and allocates
    /// nothing. A failed allocation leaves the ring unpublished, so the
    /// writer skips it. Both paths count one dropped thread.
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

    /// Stop the recorder and write the trace and the GPU folded file. Only
    /// the first call writes. A later call waits for that write to end, for
    /// at most `flush_wait_max_s`, so an exit after `finish` keeps the whole
    /// trace that the window thread writes.
    ///
    /// Happens-before: the writer closes both files, then stores `done` with
    /// release order. A waiter returns only after an acquire load reads
    /// `done`, so every file write precedes the return of each caller.
    fn flush(self: *Recorder) Flushed {
        if (self.flushed.cmpxchgStrong(.idle, .writing, .acq_rel, .acquire)) |state| {
            assert(state != .idle);

            return self.awaitWrite();
        }

        on.store(false, .release);
        self.write() catch |err| log.err("trace: writing {s} failed: {t}", .{ self.path, err });
        self.flushed.store(.done, .release);
        self.io.futexWake(FlushState, &self.flushed.raw, std.math.maxInt(u32));
        return .wrote;
    }

    /// Wait for the write of the first `flush` caller, at most
    /// `flush_wait_max_s`. The loop ends at `done` or at the deadline, so a
    /// spurious wake costs one more pass and no more.
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

    fn write(self: *Recorder) !void {
        assert(self.flushed.load(.monotonic) == .writing);

        const n_rings = @min(self.claimed.load(.acquire), self.rings.len);
        var tracks: [limits.span_threads_max]ThreadTrack = undefined;
        var n_tracks: usize = 0;
        const gpu_kept = self.gpu.kept();
        var meta: Meta = .{
            .threads_dropped = self.threads_dropped.load(.monotonic),
            .orphan_spans = self.orphan_spans.load(.monotonic),
            .gpu_dropped = self.gpu.dropped.load(.monotonic) + gpu_kept.unready,
            .gpu_level = gpu_level,
        };
        var n_spans: u64 = 0;
        for (self.rings[0..n_rings]) |*r| {
            // A ring claimed but not yet published holds no span.
            if (!r.published.load(.acquire)) continue;

            // The owner writes only past `len`, so the kept prefix sorts in place.
            const spans = r.kept();
            meta.dropped += r.dropped.load(.monotonic);
            if (spans.len == 0) continue;

            std.mem.sort(Span, spans, {}, lessThan);
            n_spans += spans.len;
            tracks[n_tracks] = .{ .tid = r.tid, .name = r.label(), .spans = spans };
            n_tracks += 1;
        }

        const gpu_records = gpu_kept.records;

        const buf = try self.gpa.alloc(u8, 1 << 20);
        defer self.gpa.free(buf);

        {
            const file = try self.dir.createFile(self.io, self.path, .{});
            defer file.close(self.io);

            var fw: std.Io.File.Writer = .init(file, self.io, buf);
            try writeJson(&fw.interface, processId(), tracks[0..n_tracks], gpu_records, meta);
            try fw.interface.flush();
        }

        if (gpu_records.len > 0) {
            var name_buf: [std.fs.max_path_bytes]u8 = undefined;
            const folded = try std.fmt.bufPrint(&name_buf, "{s}.gpu.folded", .{self.path});
            const file = try self.dir.createFile(self.io, folded, .{});
            defer file.close(self.io);

            var fw: std.Io.File.Writer = .init(file, self.io, buf);
            try writeFolded(&fw.interface, gpu_records);
            try fw.interface.flush();
        }

        log.info(
            "trace: wrote {s}: {d} spans on {d} threads, {d} GPU records, dropped {d} " ++
                "spans, {d} threads, {d} spans of threads with no ring, {d} GPU records",
            .{
                self.path,
                n_spans,
                n_tracks,
                gpu_records.len,
                meta.dropped,
                meta.threads_dropped,
                meta.orphan_spans,
                meta.gpu_dropped,
            },
        );
    }
};

/// The flush progress of the recorder. The futex word of a waiting `flush`.
const FlushState = enum(u32) { idle, writing, done };

/// How one `flush` call ended.
const Flushed = enum { wrote, waited, timed_out };

/// The ring of GPU records. Any thread can push: a push claims an index with
/// one atomic add and publishes the record with a release store of its flag.
///
/// The storage is one block of `per` records, then `per` ready flags. The
/// first push allocates it, so a trace with no GPU record holds none. Two
/// first pushes can race. Each allocates, a compare-exchange installs one
/// block, and the losing thread frees its own block. The winner sets the
/// flags before its release exchange, and every user loads the block with
/// acquire order, so each user sees the cleared flags.
///
/// A failed allocation turns the ring off for the process. Its push and each
/// later push leave their index unwritten, and `kept` counts each unwritten
/// index as dropped, so the metadata and the write log state the loss. A
/// retry gains nothing: the published prefix cannot pass the unwritten index
/// of the failed push.
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

/// The one gate of every span site. It turns on once, after `recorder` is
/// set, and turns off once, before the writer reads the rings.
var on: std.atomic.Value(bool) = .init(false);

/// The GPU timestamp level. `setFromEnv` sets it before any runner exists.
var gpu_level: GpuLevel = .off;

/// The process recorder. It is set before `on` turns on and never freed: the
/// storage lives until the process exits.
var recorder: ?*Recorder = null;

threadlocal var tl_ring: ?*Ring = null;
threadlocal var tl_no_ring: bool = false;
threadlocal var tl_name_buf: [16]u8 = @splat(0);
threadlocal var tl_name_len: u8 = 0;

// -- api --------------------------------------------------------------------

/// True while the recorder records.
pub inline fn enabled() bool {
    return on.load(.acquire);
}

/// The GPU timestamp level of this process.
pub fn gpuLevel() GpuLevel {
    return gpu_level;
}

/// Set the GPU timestamp level. Only startup code calls it, before any GPU
/// runner exists, because a runner sizes its query pools from it.
pub fn setGpuLevel(level: GpuLevel) void {
    gpu_level = level;
}

/// Open a boundary span.
pub inline fn begin(name: Name) Scope {
    if (!on.load(.acquire)) return .{ .start_ns = 0, .name = name };
    return .{ .start_ns = nowNs(), .name = name };
}

/// Open a per-op span. It compiles to nothing without `-Dtrace`.
pub inline fn beginHot(comptime name: Name) HotScope {
    comptime assert(name.isHot());
    if (comptime core_log.hot) return begin(name);
    return .{};
}

/// Record a span from two timestamps that the caller already read, so the
/// site adds no clock read.
pub inline fn record(name: Name, start: std.Io.Timestamp, end_ts: std.Io.Timestamp) void {
    if (!on.load(.acquire)) return;
    pushCurrent(.{ .start_ns = tsNs(start), .end_ns = tsNs(end_ts), .name = name });
}

/// Hand over one GPU record.
pub fn gpu(rec: GpuRecord) void {
    if (!on.load(.acquire)) return;

    assert(rec.gpu_start <= rec.gpu_end);
    const r = recorder.?;
    r.gpu.push(r.gpa, rec);
}

/// Nanoseconds of a timestamp on the `.awake` clock.
pub fn tsNs(t: std.Io.Timestamp) u64 {
    assert(t.nanoseconds > 0);

    return @intCast(t.nanoseconds);
}

/// Name the calling thread, for the trace track and for the kernel `comm`
/// that perf prints. Call it before the first span of the thread.
pub fn nameThread(name: []const u8) void {
    const n: u8 = @intCast(@min(name.len, tl_name_buf.len - 1));
    @memcpy(tl_name_buf[0..n], name[0..n]);
    tl_name_len = n;
    if (builtin.os.tag == .linux) {
        var z: [16:0]u8 = @splat(0);
        @memcpy(z[0..n], name[0..n]);
        // A name the kernel refuses changes only the perf label, so the error
        // has no consumer.
        _ = std.posix.prctl(.SET_NAME, .{@intFromPtr(&z)}) catch {};
    }

    if (tl_ring) |r| {
        r.setName(name[0..n]);
    }
}

/// Read the four trace variables. Call it once in `main`, before any thread
/// or runner exists. A bad level or window logs a warning and leaves that part
/// off. A bad ring capacity returns an error before any other effect, because
/// a silent default drops the spans that the operator asked to keep.
pub fn setFromEnv(io: std.Io, env: Env) error{InvalidTraceSpans}!void {
    return setup(std.heap.page_allocator, io, env);
}

/// Write the trace if the window has not written it already. `main` calls it
/// at exit.
pub fn finish() void {
    const r = recorder orelse return;
    _ = r.flush();
}

/// Place record `i` of `records` on the host clock.
///
/// The true offset of a record, host time minus device time, lies at or below
/// `host_wait - gpu_end` of that record. A device counter that pauses while
/// the GPU idles only lets the host clock run ahead, so the offset never falls
/// from one record to a later one. The bound of a later record therefore
/// bounds this record too, and the minimum over the record and the records
/// after it comes closest to the true offset. Records arrive in host order
/// within a few places, because two threads can push in the reverse order of
/// their waits. The forward window and the clamp absorb that inversion.
/// A clamp moves the record inside `[host_submit, host_wait]` when a counter
/// breaks that rule, and keeps its duration when it fits.
pub fn place(records: []const GpuRecord, i: usize) Placement {
    assert(i < records.len);

    const lo = i;
    const hi = @min(records.len, i + anchor_window);
    var offset: i64 = std.math.maxInt(i64);
    for (records[lo..hi]) |rec| {
        offset = @min(offset, @as(i64, @intCast(rec.host_wait_ns)) - rec.gpuEndNs());
    }

    const rec = records[i];
    const submit: i64 = @intCast(rec.host_submit_ns);
    const wait: i64 = @intCast(rec.host_wait_ns);
    assert(submit <= wait);
    const dur = @min(rec.gpuEndNs() - rec.gpuStartNs(), wait - submit);
    assert(dur >= 0);
    const want = rec.gpuStartNs() + offset;
    const start = std.math.clamp(want, submit, wait - dur);
    const p: Placement = .{
        .start_ns = start,
        .end_ns = start + dur,
        .clamp_ns = @abs(start - want),
    };
    assert(p.start_ns >= submit);
    assert(p.end_ns <= wait);
    return p;
}

/// Read the start, the end, and the interval deltas of `ts`, the raw stamps
/// of one command buffer in write order. A delta wraps at the valid bits of
/// the queue, so each delta is masked. A masked delta above `mask / 2` is a
/// step back of the counter, not a near-full wrap. It counts as zero ticks
/// and as one backstep. The end is the start plus the sum of the deltas,
/// so `end - start` and the phase sums agree. A sum past `u64` saturates.
pub fn tickSpan(ts: []const u64, mask: u64, deltas: ?[]u64) TickSpan {
    assert(ts.len >= 2);
    assert(mask != 0);

    if (deltas) |d| assert(d.len == ts.len - 1);

    var sum: u64 = 0;
    var backsteps: u32 = 0;
    for (ts[0 .. ts.len - 1], ts[1..], 0..) |prev, cur, i| {
        var dt = (cur -% prev) & mask;
        if (dt > mask / 2) {
            dt = 0;
            backsteps += 1;
        }

        if (deltas) |d| {
            d[i] = dt;
        }

        sum +|= dt;
    }

    const start = ts[0] & mask;
    const out: TickSpan = .{ .start = start, .end = start +| sum, .backsteps = backsteps };
    assert(out.start <= out.end);
    assert(out.backsteps < ts.len);
    return out;
}

/// The span order of one thread track: by start, and the longer span first
/// on a tie, so a parent precedes its children.
pub fn lessThan(_: void, a: Span, b: Span) bool {
    if (a.start_ns != b.start_ns) return a.start_ns < b.start_ns;
    return a.end_ns > b.end_ns;
}

/// Write the Chrome Trace Event JSON: one `M` name event and the `X` events
/// of each thread, then one GPU queue track per runner kind, then
/// `otherData`. The `ts` and `dur` fields are microseconds.
pub fn writeJson(
    w: *std.Io.Writer,
    pid: u64,
    threads: []const ThreadTrack,
    gpu_records: []const GpuRecord,
    meta: Meta,
) std.Io.Writer.Error!void {
    var clamps: u32 = 0;
    var clamp_max_ns: u64 = 0;
    var backsteps: u64 = 0;
    try w.writeAll("{\"traceEvents\":[\n");
    var first = true;
    for (threads) |t| {
        try sep(w, &first);
        try w.print(
            "{{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":{d},\"tid\":{d},\"args\":{{\"name\":\"",
            .{ pid, t.tid },
        );
        try writeSafe(w, t.name);
        try w.writeAll("\"}}");
        for (t.spans, 0..) |s, i| {
            if (i > 0) assert(!lessThan({}, s, t.spans[i - 1]));

            try sep(w, &first);
            try w.print(
                "{{\"ph\":\"X\",\"name\":\"{t}\",\"pid\":{d},\"tid\":{d},\"ts\":",
                .{ s.name, pid, t.tid },
            );
            try writeUs(w, @intCast(s.start_ns));
            try w.writeAll(",\"dur\":");
            try writeUs(w, @intCast(s.end_ns - s.start_ns));
            try w.writeAll("}");
        }
    }

    var used: [std.meta.fields(GpuRunner).len]bool = @splat(false);
    for (gpu_records) |rec| {
        used[@intFromEnum(rec.runner)] = true;
    }

    for (used, 0..) |u, i| {
        if (!u) continue;

        const runner: GpuRunner = @enumFromInt(i);
        try sep(w, &first);
        try w.print(
            "{{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":{d},\"tid\":{d},\"args\":{{\"name\":\"GPU queue {t}\"}}}}",
            .{ pid, gpuTid(runner), runner },
        );
    }

    for (gpu_records, 0..) |rec, i| {
        const at = place(gpu_records, i);
        if (at.clamp_ns > 0) {
            clamps += 1;
        }

        clamp_max_ns = @max(clamp_max_ns, at.clamp_ns);
        backsteps += rec.backsteps;
        try sep(w, &first);
        try w.print(
            "{{\"ph\":\"X\",\"name\":\"{t}\",\"pid\":{d},\"tid\":{d},\"ts\":",
            .{ rec.label, pid, gpuTid(rec.runner) },
        );
        try writeUs(w, at.start_ns);
        try w.writeAll(",\"dur\":");
        try writeUs(w, at.end_ns - at.start_ns);
        try w.writeAll(",\"args\":{\"gpu_dur_us\":");
        try writeUs(w, rec.gpuEndNs() - rec.gpuStartNs());
        try w.writeAll(",\"host_submit_us\":");
        try writeUs(w, @intCast(rec.host_submit_ns));
        try w.writeAll(",\"host_wait_us\":");
        try writeUs(w, @intCast(rec.host_wait_ns));
        if (rec.phases) |ph| {
            for (ph, 0..) |ns, p| {
                if (ns == 0) continue;
                try w.print(",\"{t}_us\":", .{@as(GpuPhase, @enumFromInt(p))});
                try writeUs(w, @intCast(ns));
            }
        }

        try w.writeAll("}}");
    }

    try w.writeAll("\n],\"displayTimeUnit\":\"ms\",\"otherData\":{");
    try w.print(
        "\"clock\":\"CLOCK_MONOTONIC\",\"dropped\":{d},\"threads_dropped\":{d},\"orphan_spans\":{d},",
        .{ meta.dropped, meta.threads_dropped, meta.orphan_spans },
    );
    try w.print(
        "\"gpu_level\":\"{t}\",\"gpu_records\":{d},\"gpu_dropped\":{d},\"gpu_backsteps\":{d},",
        .{ meta.gpu_level, gpu_records.len, meta.gpu_dropped, backsteps },
    );
    try w.print(
        "\"gpu_anchor\":\"local\",\"gpu_anchor_window\":{d},\"gpu_clamps\":{d},\"gpu_clamp_max_us\":",
        .{ anchor_window, clamps },
    );
    try writeUs(w, @intCast(clamp_max_ns));
    try w.writeAll("}}\n");
}

/// Write one folded-stack line per runner kind, label, and phase with the
/// summed nanoseconds, for `flamegraph.pl`. A `submit` record adds its whole
/// span to the line of its runner kind and label.
pub fn writeFolded(w: *std.Io.Writer, records: []const GpuRecord) std.Io.Writer.Error!void {
    const n_runner = std.meta.fields(GpuRunner).len;
    const n_label = std.meta.fields(GpuLabel).len;
    // The last phase column holds the records without phases.
    var sum: [n_runner][n_label][gpu_phase_count + 1]u64 = @splat(@splat(@splat(0)));
    for (records) |rec| {
        const cell = &sum[@intFromEnum(rec.runner)][@intFromEnum(rec.label)];
        if (rec.phases) |ph| {
            for (ph, 0..) |ns, p| {
                cell[p] += ns;
            }
        } else {
            cell[gpu_phase_count] += @intCast(rec.gpuEndNs() - rec.gpuStartNs());
        }
    }

    for (sum, 0..) |per_runner, ri| {
        for (per_runner, 0..) |cell, li| {
            for (cell, 0..) |ns, p| {
                if (ns == 0) continue;

                const runner: GpuRunner = @enumFromInt(ri);
                const label: GpuLabel = @enumFromInt(li);
                if (p == gpu_phase_count) {
                    try w.print("{t};{t} {d}\n", .{ runner, label, ns });
                } else {
                    try w.print(
                        "{t};{t};{t} {d}\n",
                        .{ runner, label, @as(GpuPhase, @enumFromInt(p)), ns },
                    );
                }
            }
        }
    }
}

// -- helpers ----------------------------------------------------------------

fn setup(gpa: std.mem.Allocator, io: std.Io, env: Env) error{InvalidTraceSpans}!void {
    assert(!on.load(.monotonic));
    assert(recorder == null);

    const capacity: u32 = if (env.spans) |text| try limits.parseTraceSpans(
        text,
    ) else limits.spans_per_thread;

    if (env.gpu) |g| {
        if (std.meta.stringToEnum(GpuLevel, g)) |level| {
            gpu_level = level;
        } else {
            log.warn(
                "ZIGFERENCER_TRACE_GPU={s} is not off, submit, or phases, so GPU timestamps " ++
                    "stay off",
                .{g},
            );
        }
    }

    const path = env.out orelse return;
    if (path.len == 0) return;

    var seconds: ?u32 = null;
    if (env.seconds) |text| {
        seconds = std.fmt.parseInt(u32, text, 10) catch blk: {
            log.warn(
                "ZIGFERENCER_TRACE_SECONDS={s} is not a whole number of seconds, so the " ++
                    "trace runs to exit",
                .{text},
            );
            break :blk null;
        };
    }

    const r = Recorder.create(gpa, io, path, capacity) catch |err| {
        log.warn("trace: no storage for the span rings ({t}), so tracing stays off", .{err});
        return;
    };
    recorder = r;
    nameThread("main");
    on.store(true, .release);
    if (seconds) |s| {
        const t = std.Thread.spawn(.{}, windowMain, .{ io, s }) catch |err| {
            log.warn("trace: no window thread ({t}), so the trace runs to exit", .{err});
            return;
        };
        t.detach();
    }

    log.info("trace: recording to {s}, {d} spans per thread", .{ path, capacity });
}

/// The longest wait of a later `flush` caller. The write is finite, because
/// every ring is bounded, so the bound only catches a stalled file system.
const flush_wait_max_s = 60;

fn endSlow(s: Scope) void {
    pushCurrent(.{ .start_ns = s.start_ns, .end_ns = nowNs(), .name = s.name });
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

fn processId() u64 {
    if (builtin.os.tag == .linux) return @intCast(std.os.linux.getpid());
    return 1;
}

/// The synthetic thread id of one GPU queue track, past any kernel tid.
fn gpuTid(runner: GpuRunner) u64 {
    return (1 << 32) + @as(u64, @intFromEnum(runner));
}

fn ticksToNs(ticks: u64, period_ns: f64) i64 {
    assert(period_ns > 0);

    return @intFromFloat(@as(f64, @floatFromInt(ticks)) * period_ns);
}

fn windowMain(io: std.Io, seconds: u32) void {
    io.sleep(.fromSeconds(seconds), .awake) catch {};
    finish();
}

fn sep(w: *std.Io.Writer, first: *bool) !void {
    if (!first.*) {
        try w.writeAll(",\n");
    }

    first.* = false;
}

/// Microseconds with three decimals, from nanoseconds, with no float.
fn writeUs(w: *std.Io.Writer, ns: i64) !void {
    const mag: u64 = @abs(ns);
    if (ns < 0) {
        try w.writeAll("-");
    }

    try w.print("{d}.{d:0>3}", .{ mag / 1000, mag % 1000 });
}

/// A thread name with every byte outside [A-Za-z0-9_.-] replaced, so the
/// JSON string needs no escape.
fn writeSafe(w: *std.Io.Writer, text: []const u8) !void {
    for (text) |c| {
        const ok = std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '.' or c == ' ';
        try w.writeByte(if (ok) c else '_');
    }
}

// -- tests ------------------------------------------------------------------

const prop = @import("prop.zig");
const testing = std.testing;

test "every hot span name follows every boundary name" {
    var seen_hot = false;
    for (std.enums.values(Name)) |n| {
        if (n.isHot()) {
            seen_hot = true;
        }

        if (seen_hot) {
            try testing.expect(n.isHot());
        }
    }

    try testing.expect(!Name.decode_step.isHot());
    try testing.expect(Name.op_lm_head.isHot());
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

                buf[nb] = .{ .start_ns = ks, .end_ns = ke, .name = .decode_step };
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
    for (0..3) |t| {
        const s = c.spans[t][0..c.lens[t]];
        std.mem.sort(Span, s, {}, lessThan);
        tracks[t] = .{ .tid = 100 + t, .name = "worker \"q\"", .spans = s };
    }

    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();

    try writeJson(&out.writer, 7, &tracks, &.{}, .{});

    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, out.written(), .{});
    defer parsed.deinit();

    const events = parsed.value.object.get("traceEvents").?.array.items;
    // Per thread: `ts` never falls, and every span nests in the open span
    // that contains its start.
    var last_ts: [3]i64 = @splat(-1);
    var stack: [3][64]i64 = undefined;
    var depth: [3]u32 = @splat(0);
    for (events) |e| {
        if (!std.mem.eql(u8, e.object.get("ph").?.string, "X")) continue;

        const t: usize = @intCast(e.object.get("tid").?.integer - 100);
        // Compare in whole nanoseconds, because float sums round.
        const ts = usToNs(e.object.get("ts").?);
        const end_ts = ts + usToNs(e.object.get("dur").?);
        try testing.expect(ts >= last_ts[t]);
        last_ts[t] = ts;
        while (depth[t] > 0 and stack[t][depth[t] - 1] <= ts) {
            depth[t] -= 1;
        }

        if (depth[t] > 0) {
            try testing.expect(end_ts <= stack[t][depth[t] - 1]);
        }

        stack[t][depth[t]] = end_ts;
        depth[t] += 1;
    }
}

fn usToNs(v: std.json.Value) i64 {
    return @intFromFloat(@round(jsonNumber(v) * 1000));
}

fn jsonNumber(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch unreachable,
        else => unreachable,
    };
}

test "json: the output parses, ts never falls per thread, and spans nest" {
    try prop.forAll(
        .{ .site = 0x5350_0002, .cases = 64, .max_size = 40 },
        {},
        TraceCase,
        genTrace,
        checkTrace,
    );
}

/// GPU records whose device clock drifts against the host clock: a counter
/// that pauses (the offset grows linearly or in steps), or a counter that runs
/// fast (the offset falls). The wait jitter is bounded. `true_start` holds the
/// host time of each GPU start.
const AnchorCase = struct {
    records: [48]GpuRecord,
    true_start: [48]i64,
    n: u32,
    /// True when the offset never falls from one record to the next.
    monotone: bool,
};

const jitter_max_ns: u64 = 2_000;

fn genAnchor(_: void, random: std.Random, size: u32) AnchorCase {
    var c: AnchorCase = .{
        .records = undefined,
        .true_start = undefined,
        .n = @as(u32, @min(size, 47)) + 1,
        .monotone = true,
    };
    const mode = random.uintLessThan(u32, 3);
    c.monotone = mode != 2;
    const rate: i64 = @intCast(random.uintLessThan(u64, 60_000));
    var offset: i64 = @intCast(1_000_000_000 + random.uintLessThan(u64, 1_000_000));
    var gpu_t: u64 = 10_000_000 + random.uintLessThan(u64, 10_000);
    for (c.records[0..c.n], 0..) |*r, i| {
        const change: i64 = switch (mode) {
            0 => rate,
            1 => if (random.uintLessThan(u32, 8) == 0) @intCast(
                1_000 + random.uintLessThan(u64, 5_000_000),
            ) else 0,
            else => -@divTrunc(rate, 8),
        };

        if (i > 0) {
            offset += change;
        }

        const busy = 1_000 + random.uintLessThan(u64, 100_000);
        const lead = 5_000 + random.uintLessThan(u64, 5_000);
        const host_start: i64 = @as(i64, @intCast(gpu_t)) + offset;
        c.true_start[i] = host_start;
        r.* = .{
            .runner = .qwen_fused,
            .label = .decode,
            .gpu_start = gpu_t,
            .gpu_end = gpu_t + busy,
            .period_ns = 1.0,
            .host_submit_ns = @intCast(host_start - @as(i64, @intCast(lead))),
            .host_wait_ns = @intCast(
                host_start + @as(i64, @intCast(busy + random.uintLessThan(u64, jitter_max_ns))),
            ),
        };
        gpu_t += busy + random.uintLessThan(u64, 500);
    }

    return c;
}

fn checkAnchor(_: void, c: AnchorCase) anyerror!void {
    const records = c.records[0..c.n];
    for (records, 0..) |rec, i| {
        const p = place(records, i);
        // Every record sits inside its host window, at its device duration.
        try testing.expect(p.start_ns >= @as(i64, @intCast(rec.host_submit_ns)));
        try testing.expect(p.end_ns <= @as(i64, @intCast(rec.host_wait_ns)));
        try testing.expect(p.end_ns - p.start_ns == rec.gpuEndNs() - rec.gpuStartNs());
        // With an offset that never falls, the placement is late by less
        // than the wait jitter, at any drift rate.
        if (c.monotone) {
            const err = p.start_ns - c.true_start[i];
            try testing.expect(err >= 0);
            try testing.expect(err < @as(i64, @intCast(jitter_max_ns)));
        }
    }
}

test "anchor: each record sits in its host window and follows a drifting device clock" {
    try prop.forAll(
        .{ .site = 0x5350_0003, .max_size = 47 },
        {},
        AnchorCase,
        genAnchor,
        checkAnchor,
    );
}

test "folded: one line per runner, label, and phase with the summed time" {
    var ph: [gpu_phase_count]u64 = @splat(0);
    ph[@intFromEnum(GpuPhase.attn)] = 300;
    ph[@intFromEnum(GpuPhase.lmhead)] = 50;
    const recs = [_]GpuRecord{
        .{ .runner = .llama_fused, .label = .decode, .gpu_start = 0, .gpu_end = 350, .period_ns = 1, .host_submit_ns = 1, .host_wait_ns = 400, .phases = ph },
        .{ .runner = .llama_fused, .label = .decode, .gpu_start = 400, .gpu_end = 750, .period_ns = 1, .host_submit_ns = 1, .host_wait_ns = 800, .phases = ph },
        .{ .runner = .staging, .label = .copy, .gpu_start = 0, .gpu_end = 40, .period_ns = 2, .host_submit_ns = 1, .host_wait_ns = 100 },
    };
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeFolded(&w, &recs);
    try testing.expectEqualStrings(
        "llama_fused;decode;attn 600\n" ++
            "llama_fused;decode;lmhead 100\n" ++
            "staging;copy 80\n",
        w.buffered(),
    );
}

test "off path: a span site allocates nothing and records nothing" {
    // An unset output path leaves the recorder off and takes no storage.
    var failing: std.testing.FailingAllocator = .init(testing.allocator, .{});
    try setup(failing.allocator(), testing.io, .{});
    try testing.expectEqual(@as(usize, 0), failing.allocations);
    try testing.expect(!enabled());
    const s = begin(.decode_step);
    try testing.expectEqual(@as(u64, 0), s.start_ns);
    s.end();
    beginHot(.pool_run).end();
    record(.gpu_wait, .{ .nanoseconds = 1 }, .{ .nanoseconds = 2 });
    gpu(
        .{ .runner = .staging, .label = .fill, .gpu_start = 0, .gpu_end = 1, .period_ns = 1, .host_submit_ns = 1, .host_wait_ns = 2 },
    );
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
            setup(failing.allocator(), testing.io, .{ .out = "unused.json", .spans = text }),
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
    const r = try Recorder.create(failing.allocator(), testing.io, "unused.json", per);
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
    const r = try Recorder.create(failing.allocator(), testing.io, "unused.json", 2);
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
fn flushAndParse(r: *Recorder, result: *Flushed, whole: *bool) void {
    result.* = r.flush();
    const text = r.dir.readFileAlloc(
        r.io,
        r.path,
        testing.allocator,
        .limited(1 << 22),
    ) catch return;
    defer testing.allocator.free(text);

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        testing.allocator,
        text,
        .{},
    ) catch return;
    parsed.deinit();
    whole.* = true;
}

test "flush: of two racing callers one writes, and both return after the write ends" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const per: u32 = 4096;
    var round: u32 = 0;
    while (round < 8) : (round += 1) {
        const r = try Recorder.create(testing.allocator, testing.io, "trace.json", per);
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
        const a = try std.Thread.spawn(.{}, flushAndParse, .{ r, &results[0], &whole[0] });
        const b = std.Thread.spawn(.{}, flushAndParse, .{ r, &results[1], &whole[1] }) catch |err| {
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

test "tickSpan: wrap, step back, equal ticks, and normal order" {
    const narrow: u64 = (1 << 36) - 1;
    const wide: u64 = std.math.maxInt(u64);
    const Row = struct {
        name: []const u8,
        ts: []const u64,
        mask: u64,
        start: u64,
        end: u64,
        backsteps: u32,
        deltas: []const u64,
    };
    const rows = [_]Row{
        .{ .name = "normal order", .ts = &.{ 100, 150, 400 }, .mask = wide, .start = 100, .end = 400, .backsteps = 0, .deltas = &.{ 50, 250 } },
        .{ .name = "equal ticks", .ts = &.{ 7, 7, 7 }, .mask = wide, .start = 7, .end = 7, .backsteps = 0, .deltas = &.{ 0, 0 } },
        // A 36-bit counter wraps between the stamps. The high bits are noise.
        .{ .name = "wrap on a narrow counter", .ts = &.{ (0xab << 36) | (narrow - 9), (0xcd << 36) | 5, 20 }, .mask = narrow, .start = narrow - 9, .end = narrow - 9 + 30, .backsteps = 0, .deltas = &.{ 15, 15 } },
        // The middle stamp reads 10 ticks before the first one.
        .{ .name = "step back on a 64-bit counter", .ts = &.{ 1000, 990, 1100 }, .mask = wide, .start = 1000, .end = 1110, .backsteps = 1, .deltas = &.{ 0, 110 } },
        // The end stamp reads before the start, which overflowed the old sum.
        .{ .name = "end before start on a 64-bit counter", .ts = &.{ 1000, 990 }, .mask = wide, .start = 1000, .end = 1000, .backsteps = 1, .deltas = &.{0} },
        .{ .name = "step back on a narrow counter", .ts = &.{ 50, 40 }, .mask = narrow, .start = 50, .end = 50, .backsteps = 1, .deltas = &.{0} },
    };
    for (rows) |row| {
        errdefer std.debug.print("tickSpan row: {s}\n", .{row.name});

        var deltas: [4]u64 = undefined;
        const got = tickSpan(row.ts, row.mask, deltas[0 .. row.ts.len - 1]);
        try testing.expectEqual(row.start, got.start);
        try testing.expectEqual(row.end, got.end);
        try testing.expectEqual(row.backsteps, got.backsteps);
        try testing.expectEqualSlices(u64, row.deltas, deltas[0 .. row.ts.len - 1]);
        // The staging record passes no delta buffer and reads the same span.
        try testing.expectEqual(got, tickSpan(row.ts, row.mask, null));
    }
}
