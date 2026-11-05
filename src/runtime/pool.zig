//! A persistent fork-join pool for the CPU ops. The pool is the control unit: it owns the threads,
//! the epoch word, and the claim counter. A unit function is the datapath: it computes one unit of
//! an op, and its result does not depend on the thread that runs it.
//!
//! Ownership. The caller owns the pool and keeps its address fixed, because every worker holds a
//! pointer to it. The calling thread is participant 0 of a dispatch, and `threads - 1` workers wait
//! between dispatches.
//!
//! Dispatch. A dispatch publishes one job and a new epoch word. Participant `i` runs unit `i`
//! first, then every participant claims units from a shared counter until the counter passes the
//! unit count. The owner waits until every worker of the epoch reports done. A partial wake can
//! leave a participant asleep, so the wake reaches every waiter. Each worker makes at most
//! `n_units - parts` claims plus one. Each participant makes one failed claim, so the counter ends
//! at `n_units + parts`.
//!
//! Memory order. The protocol, with the order of every access:
//!
//!   owner   write the job, store `claim = parts`, `pending = parts - 1`,
//!           store `ctl` (seq_cst), load `sleepers` (seq_cst), and wake every
//!           waiter on `ctl` if the load is positive. Run unit 0, claim, then
//!           yield until `pending` reads 0 (acquire).
//!   worker  spin on `ctl` for `spin_ns`. Then increment `sleepers` (seq_cst),
//!           load `ctl` again (seq_cst), and futex-wait on the old value. A
//!           worker at or past `parts` skips the epoch and never reads the
//!           job. A participant runs its units and decrements `pending`
//!           (release).
//!
//! No lost wake-up. The two seq_cst pairs fall in one total order. Either the owner load sees the
//! increment of a worker and wakes it, or the worker load sees the new epoch and does not wait. The
//! kernel compares the futex word atomically, so no wake-up is lost between the load and the wait.
//!
//! Sequence wrap. The sequence wraps at 2^24 dispatches. The pool assumes that no worker stays
//! preempted across exactly a multiple of 2^24 dispatches between two of its loads of `ctl`. Such a
//! worker can miss its epoch.

const std = @import("std");
const builtin = @import("builtin");
const limits = @import("../core/limits.zig");
const prop = @import("../core/prop.zig");
const span = @import("../core/span.zig");

const assert = std.debug.assert;
const cache_line = std.atomic.cache_line;

// -- constants --------------------------------------------------------------

/// The epoch word packs a 24-bit sequence over an 8-bit participant count.
const seq_bits: u5 = 24;

const parts_bits: u5 = 8;
const seq_mask: u32 = (1 << seq_bits) - 1;
const parts_mask: u32 = (1 << parts_bits) - 1;

comptime {
    assert(@as(u32, seq_bits) + parts_bits == 32);
}

/// The word before the first dispatch. A dispatch stores at least two participants, never zero.
const ctl_idle: u32 = 0;

/// The word that ends the workers. Its count, 255, is above every legal participant count.
const ctl_stop: u32 = std.math.maxInt(u32);

/// Pause iterations between two clock reads of the spin wait. A clock read costs more than a pause.
const spin_check: u32 = 64;

// -- types ------------------------------------------------------------------

pub const InitError = std.Thread.SpawnError;

/// One dispatch with its types erased. Only the participants of the current
/// epoch read it.
const Job = struct {
    unit_fn: *const fn (ctx: *const anyopaque, unit: u32) void,
    ctx: *const anyopaque,
    n_units: u32,
};

/// One atomic word on a cache line of its own. The alignment rounds the size up to the line.
const Line = struct {
    value: std.atomic.Value(u32) align(cache_line),
};

/// The pool state. Each shared counter has its own cache line, so a claim does not slow a spin.
pub const Pool = struct {
    ctl: Line,
    claim: Line,
    pending: Line,
    sleepers: Line,
    /// The owner writes the job only while the pool is idle.
    job: Job align(cache_line),
    io: std.Io,
    n_threads: u32,
    spin_ns: u64,
    /// The sequence of the last published epoch. Only the owner reads it.
    seq: u32,
    /// Set while a dispatch runs. It checks the one-owner rule in safe builds.
    busy: std.atomic.Value(bool),
    workers: [limits.max_threads - 1]std.Thread,

    // -- api ----------------------------------------------------------------

    /// Start `n_threads - 1` workers. The pool must not move until `deinit`.
    /// `spin_ns` is the time a worker spins on the epoch word before it sleeps. Zero sleeps
    /// at once.
    pub fn init(self: *Pool, io: std.Io, n_threads: u32, spin_ns: u64) InitError!void {
        assert(n_threads >= 1);
        assert(n_threads <= limits.max_threads);

        self.* = .{
            .ctl = .{ .value = .init(ctl_idle) },
            .claim = .{ .value = .init(0) },
            .pending = .{ .value = .init(0) },
            .sleepers = .{ .value = .init(0) },
            .job = undefined,
            .io = io,
            .n_threads = n_threads,
            .spin_ns = spin_ns,
            .seq = 0,
            .busy = .init(false),
            .workers = undefined,
        };
        if (builtin.single_threaded) return;

        var spawned: u32 = 0;
        errdefer self.stopWorkers(spawned);

        while (spawned < n_threads - 1) : (spawned += 1) {
            // Pass the idle word in: a load at thread start can miss a dispatch that ran first.
            self.workers[spawned] = try std.Thread.spawn(
                .{},
                workerMain,
                .{ self, spawned + 1, ctl_idle },
            );
        }

        assert(spawned == n_threads - 1);
    }

    /// Stop and join every worker. No dispatch can run at the same time.
    pub fn deinit(self: *Pool) void {
        assert(!self.busy.load(.monotonic));
        assert(self.pending.value.load(.monotonic) == 0);

        if (!builtin.single_threaded) {
            self.stopWorkers(self.n_threads - 1);
        }

        self.* = undefined;
    }

    /// The participants a dispatch can have, the owner included.
    pub fn threads(self: *const Pool) u32 {
        assert(self.n_threads >= 1);
        assert(self.n_threads <= limits.max_threads);

        return self.n_threads;
    }

    /// Run `unit_fn(ctx, u)` once for every `u < n_units` over `parts` participants, the caller
    /// included, and return when all are done. One thread at a time can dispatch.
    /// `parts = 1` runs every unit on the caller and touches no shared word.
    pub fn run(
        self: *Pool,
        comptime Ctx: type,
        ctx: *const Ctx,
        comptime unit_fn: fn (*const Ctx, u32) void,
        n_units: u32,
        parts: u32,
    ) void {
        assert(parts >= 1);
        assert(parts <= self.n_threads);
        assert(parts <= n_units);
        // The claim counter ends at `n_units + parts`, and it must not wrap.
        assert(n_units <= std.math.maxInt(u32) - limits.max_threads);

        const Erased = struct {
            fn call(erased: *const anyopaque, unit: u32) void {
                unit_fn(@ptrCast(@alignCast(erased)), unit);
            }
        };
        self.enter();
        defer self.leave();

        const sp = span.beginHot(.pool_run);
        defer sp.end();

        // Run inline when one participant owns every unit.
        if (builtin.single_threaded or parts == 1) {
            var unit: u32 = 0;
            while (unit < n_units) : (unit += 1) {
                unit_fn(ctx, unit);
            }

            return;
        }

        // Publish the job, then the new epoch word.
        assert(self.pending.value.load(.monotonic) == 0);
        self.job = .{ .unit_fn = Erased.call, .ctx = ctx, .n_units = n_units };
        self.claim.value.store(parts, .monotonic);
        self.pending.value.store(parts - 1, .monotonic);
        self.seq = (self.seq + 1) & seq_mask;
        self.ctl.value.store(pack(self.seq, parts), .seq_cst);

        // Wake every sleeper. See "Memory order" in the header.
        if (self.sleepers.value.load(.seq_cst) > 0) {
            self.io.futexWake(u32, &self.ctl.value.raw, std.math.maxInt(u32));
        }

        // Run the share of participant 0, then wait for the workers.
        runShare(&self.job, &self.claim.value, 0);
        while (self.pending.value.load(.acquire) != 0) {
            std.Thread.yield() catch {};
        }

        assert(self.claim.value.load(.monotonic) == n_units + parts);
    }

    // -- helpers ------------------------------------------------------------

    fn enter(self: *Pool) void {
        if (!std.debug.runtime_safety) return;
        const was_busy = self.busy.swap(true, .acquire);
        assert(!was_busy);
    }

    fn leave(self: *Pool) void {
        if (!std.debug.runtime_safety) return;
        const was_busy = self.busy.swap(false, .release);
        assert(was_busy);
    }

    /// Store the stop word, wake every waiter, and join the first `count` workers.
    fn stopWorkers(self: *Pool, count: u32) void {
        assert(count < limits.max_threads);

        self.ctl.value.store(ctl_stop, .seq_cst);
        self.io.futexWake(u32, &self.ctl.value.raw, std.math.maxInt(u32));
        for (self.workers[0..count]) |worker| {
            worker.join();
        }

        assert(self.ctl.value.load(.monotonic) == ctl_stop);
    }

    /// The loop of worker `w`. It ends only on the stop word.
    fn workerMain(self: *Pool, w: u32, seen_first: u32) void {
        assert(w >= 1);
        assert(w < self.n_threads);

        var name_buf: [16]u8 = undefined;
        span.nameThread(std.fmt.bufPrint(&name_buf, "zf-pool-{d}", .{w}) catch unreachable);
        var seen = seen_first;
        while (true) {
            const wait_sp = span.beginHot(.worker_wait);
            const word = self.waitChange(seen);
            wait_sp.end();
            assert(word != seen);
            seen = word;
            if (word == ctl_stop) return;
            if (w >= unpackParts(word)) continue;

            runShare(&self.job, &self.claim.value, w);
            _ = self.pending.value.fetchSub(1, .release);
        }
    }

    /// Wait until the epoch word differs from `seen`, and return the new word.
    /// The spin budget restarts on every call, so a worker that skips an epoch spins again.
    fn waitChange(self: *Pool, seen: u32) u32 {
        if (self.spin_ns > 0) {
            const start = nowNs(self.io);
            var spins: u32 = 0;
            while (true) {
                const word = self.ctl.value.load(.acquire);
                if (word != seen) return word;

                std.atomic.spinLoopHint();
                spins +%= 1;
                if (spins % spin_check == 0 and nowNs(self.io) - start >= self.spin_ns) break;
            }
        }

        // Sleep until the word changes. A spurious wake-up goes back to sleep.
        while (true) {
            _ = self.sleepers.value.fetchAdd(1, .seq_cst);
            if (self.ctl.value.load(.seq_cst) == seen) {
                self.io.futexWaitUncancelable(u32, &self.ctl.value.raw, seen);
            }

            _ = self.sleepers.value.fetchSub(1, .monotonic);
            const word = self.ctl.value.load(.acquire);
            if (word != seen) return word;
        }
    }
};

// -- helpers ----------------------------------------------------------------

/// Pack a sequence and a participant count into one epoch word.
fn pack(seq: u32, parts: u32) u32 {
    assert(seq <= seq_mask);
    assert(parts <= parts_mask);

    const word = seq << parts_bits | parts;
    assert(unpackSeq(word) == seq);
    assert(unpackParts(word) == parts);
    return word;
}

/// The sequence of an epoch word, the high `seq_bits` bits.
fn unpackSeq(word: u32) u32 {
    const seq = word >> parts_bits;
    assert(seq <= seq_mask);
    return seq;
}

/// The participant count of an epoch word, the low `parts_bits` bits.
fn unpackParts(word: u32) u32 {
    const parts = word & parts_mask;
    assert(parts <= parts_mask);
    return parts;
}

comptime {
    // The stop word holds a participant count that no dispatch can use.
    assert(unpackParts(ctl_stop) > limits.max_threads);
    assert(@sizeOf(Line) == cache_line);
}

/// Run the first unit of a participant, then claim units until the counter
/// passes the unit count. The loop ends after at most `n_units` claims.
fn runShare(job: *const Job, claim: *std.atomic.Value(u32), first: u32) void {
    const n_units = job.n_units;
    assert(first < n_units);
    const sp = span.beginHot(.run_share);
    defer sp.end();

    job.unit_fn(job.ctx, first);
    while (true) {
        const unit = claim.fetchAdd(1, .monotonic);
        if (unit >= n_units) return;
        job.unit_fn(job.ctx, unit);
    }
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.Timestamp.now(io, .awake).raw.toNanoseconds();
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

/// A unit context that counts the runs of every unit.
const Counts = struct {
    runs: []std.atomic.Value(u32),

    fn unit(self: *const Counts, u: u32) void {
        _ = self.runs[u].fetchAdd(1, .monotonic);
    }

    fn reset(self: Counts) void {
        for (self.runs) |*r| {
            r.store(0, .monotonic);
        }
    }
};

fn expectRuns(counts: Counts, n_units: u32, want: u32) !void {
    for (counts.runs, 0..) |*r, u| {
        const expected: u32 = if (u < n_units) want else 0;
        try testing.expectEqual(expected, r.load(.monotonic));
    }
}

/// Wait until every worker is inside its sleep loop. With `spin_ns = 0`, an idle worker sleeps
/// at once.
fn awaitAllAsleep(pool: *Pool) void {
    while (pool.sleepers.value.load(.seq_cst) != pool.n_threads - 1) {
        std.Thread.yield() catch {};
    }
}

test "pool: pack and unpack, the anchors of the epoch word" {
    const Anchor = struct { seq: u32, parts: u32, word: u32 };
    const anchors = [_]Anchor{
        .{ .seq = 0, .parts = 0, .word = 0 },
        .{ .seq = 1, .parts = 2, .word = 258 },
        .{ .seq = 5, .parts = 8, .word = 1288 },
        .{ .seq = 1000, .parts = 64, .word = 256_064 },
        .{ .seq = seq_mask, .parts = 255, .word = 4_294_967_295 },
    };
    for (anchors) |a| {
        try testing.expectEqual(a.word, pack(a.seq, a.parts));
        try testing.expectEqual(a.seq, unpackSeq(a.word));
        try testing.expectEqual(a.parts, unpackParts(a.word));
    }

    try testing.expectEqual(ctl_stop, pack(seq_mask, 255));
    try testing.expectEqual(ctl_idle, pack(0, 0));
}

fn checkPoolShape(_: void, c: prop.Trial) anyerror!void {
    var runs: [300]std.atomic.Value(u32) = undefined;
    const counts: Counts = .{ .runs = &runs };
    var prng: std.Random.DefaultPrng = .init(c.seed);
    const random = prng.random();
    const n_threads = random.intRangeAtMost(u32, 1, 48);
    const spin_ns: u64 = if (c.index % 2 == 0) 0 else 20_000;
    var pool: Pool = undefined;
    try pool.init(testing.io, n_threads, spin_ns);
    defer pool.deinit();

    for (0..20) |_| {
        const n_units = random.intRangeAtMost(u32, 1, runs.len);
        const parts = random.intRangeAtMost(u32, 1, @min(n_threads, n_units));
        counts.reset();
        pool.run(Counts, &counts, Counts.unit, n_units, parts);
        try expectRuns(counts, n_units, 1);
    }
}

test "pool: every unit runs exactly once over seeded shapes, up to 48 threads" {
    try prop.forAll(
        .{ .site = 0x9001_f0c5, .cases = 24, .max_size = 23 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkPoolShape,
    );
}

test "pool: 100 init and deinit cycles" {
    var runs: [64]std.atomic.Value(u32) = undefined;
    const counts: Counts = .{ .runs = &runs };
    for (0..100) |cycle| {
        const n_threads: u32 = @intCast(1 + cycle % 16);
        const spin_ns: u64 = if (cycle % 3 == 0) 0 else 50_000;
        var pool: Pool = undefined;
        try pool.init(testing.io, n_threads, spin_ns);
        defer pool.deinit();

        try testing.expectEqual(n_threads, pool.threads());
        // Every other cycle skips the dispatch, so deinit also meets workers in their first wait.
        if (cycle % 2 == 1) {
            counts.reset();
            pool.run(Counts, &counts, Counts.unit, runs.len, n_threads);
            try expectRuns(counts, runs.len, 1);
        }
    }
}

test "pool: dispatch and deinit while every worker sleeps" {
    if (builtin.single_threaded) return error.SkipZigTest;

    var runs: [32]std.atomic.Value(u32) = undefined;
    const counts: Counts = .{ .runs = &runs };
    var pool: Pool = undefined;
    try pool.init(testing.io, 8, 0);
    defer pool.deinit();

    awaitAllAsleep(&pool);
    counts.reset();
    pool.run(Counts, &counts, Counts.unit, runs.len, 8);
    try expectRuns(counts, runs.len, 1);
    awaitAllAsleep(&pool);
}

test "pool: one participant runs inline and stores no shared word" {
    var runs: [16]std.atomic.Value(u32) = undefined;
    const counts: Counts = .{ .runs = &runs };
    var pool: Pool = undefined;
    try pool.init(testing.io, 4, 0);
    defer pool.deinit();

    counts.reset();
    pool.run(Counts, &counts, Counts.unit, runs.len, 1);
    try expectRuns(counts, runs.len, 1);
    try testing.expectEqual(ctl_idle, pool.ctl.value.load(.seq_cst));
    try testing.expectEqual(@as(u32, 0), pool.claim.value.load(.seq_cst));
    try testing.expectEqual(@as(u32, 0), pool.pending.value.load(.seq_cst));
    try testing.expectEqual(@as(u32, 0), pool.seq);
}

test "pool: 10000 back-to-back dispatches of tiny units, spinning and sleeping" {
    var runs: [8]std.atomic.Value(u32) = undefined;
    const counts: Counts = .{ .runs = &runs };
    for ([_]u64{ 50_000, 0 }) |spin_ns| {
        var pool: Pool = undefined;
        try pool.init(testing.io, 4, spin_ns);
        defer pool.deinit();

        counts.reset();
        for (0..10_000) |_| {
            pool.run(Counts, &counts, Counts.unit, runs.len, 4);
        }

        try expectRuns(counts, runs.len, 10_000);
        if (!builtin.single_threaded) {
            // The sequence advances once per dispatch.
            try testing.expectEqual(@as(u32, 10_000), pool.seq);
            try testing.expectEqual(pack(10_000, 4), pool.ctl.value.load(.seq_cst));
        }
    }
}
