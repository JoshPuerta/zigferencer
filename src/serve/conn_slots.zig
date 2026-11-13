//! The connection slot table of the serve engine: the state of each slot and the steps that move
//! it. `engine.zig` holds the queue and the socket of slot i in `conns[i]`, and this table decides
//! who can touch them.
//!
//! Comptime. The schedule test takes the slot rules as a type, so one schedule drives the real
//! rules and each broken-rule variant.
//!
//! Role. This file holds the control rules alone, so a test can call the steps of all three
//! threads from one thread, in any order that the rules of each thread allow.
//!
//! States. A slot is `free`, `live`, `closing`, `swept`, or `reclaimable`. Two events end a
//! connection: the last task exits, and the engine handles `conn_closed`. They can come in either
//! order, and the slot is `reclaimable` after both.
//!
//!     free --open--> live --last exitTask--> closing --release--> reclaimable --open--> live
//!       ^              |                                              ^
//!       +--abandon-----+--release--> swept --last exitTask-----------+
//!
//!     teardown: reset (any state but live and swept) --> free
//!
//! Threads.
//!   - The acceptor calls `claim`, `nextId`, `open`, and `abandon`.
//!   - A connection task calls `exitTask`, and `idOf` for its own slot.
//!   - The engine thread calls `find`, `release`, `stateOf`, and `idOf` while it runs. At
//!     teardown, after the acceptor has stopped, it calls `stateOf` and `idOf` for the slots that
//!     still have a task, and `reset` after those tasks exit.
//!
//! Rules. These rules keep the engine away from a slot that the acceptor writes.
//!   1. The acceptor writes a slot (its id, its queue) only in `free` or `reclaimable`, and in
//!      `live` before a task starts (`abandon`).
//!   2. The engine thread reads a slot (find, put, close) only in `live` or `closing`.
//!   3. Only `release` takes a slot out of `live` or `closing` toward `reclaimable` in the engine.
//!      The engine thread calls it after it handles `conn_closed`, so no find of that thread is
//!      open on the slot.
//!   4. The reader task of a connection submits `conn_closed` after its last request. So the
//!      engine handles every request of the reader, and queues the reply that each request gets
//!      while the engine handles it, before it closes the queue. The sweep at `conn_closed` ends
//!      the turns in flight. After `conn_closed`, only a `wakeup` of the writer names the
//!      connection, and a `wakeup` touches no slot.
//!   5. The last exit and `release` can change one slot at the same time. Each one exchanges the
//!      state it read and reads again when the other one won. `abandon` runs before any task can
//!      exit last and before a reader exists, so it races with neither.
//!
//! Cap. `claim` counts only `live` slots against the connection cap. A slot in `closing` or `swept`
//! holds no reader, so a new client can take another slot. The table holds twice the cap, so
//! a slot that waits for the engine does not block the next client. When every spare slot waits,
//! `claim` refuses, and the client gets the refusal of the cap.
//!
//! Ids. `nextId` never returns zero, the id of the requests from inside the process, and never an
//! id that a slot in `live`, `closing`, or `swept` holds. So an id names at most one connection
//! that the engine or a task can still reach.
//!
//! An emit for an id that no `live` or `closing` slot holds is dropped. Such an id belongs to a
//! connection that the engine has swept, so its queue is closed.

const std = @import("std");
const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

/// The slots per allowed connection. One spare slot per connection lets a client connect while
/// the slot of its last connection waits for the engine.
pub const slots_per_connection = 2;

/// The counters that tell a binary reader that its connection is idle, in the order it loads
/// them. The engine stores `sessions` and then lowers `requests_open`. So a reader that loads
/// `requests_open` first and reads zero also reads the new `sessions`, and an opening session
/// never looks idle.
pub const idle_load_order = [2]IdleCounter{ .requests_open, .sessions };

// -- types ------------------------------------------------------------------

pub const State = enum(u8) { free, live, closing, swept, reclaimable };

pub const IdleCounter = enum { requests_open, sessions };

/// One slot. Every field is atomic, because three threads read it.
pub const Slot = struct {
    state: std.atomic.Value(State) = .init(.free),
    /// The connection id. Zero means no connection.
    id: std.atomic.Value(u32) = .init(0),
    /// The tasks of the connection that have not exited.
    tasks: std.atomic.Value(u8) = .init(0),
};

/// A slot that the acceptor can write. `reused` is true for a `reclaimable` slot, whose old
/// tasks the acceptor must await before it writes the slot.
pub const Claim = struct { index: u32, reused: bool };

pub const ConnSlots = struct {
    slots: []Slot,
    /// The most `live` slots at once.
    cap: u32,

    /// The caller owns `slots`, which holds `slots_per_connection` slots per allowed connection.
    /// Every slot must be `free`.
    pub fn init(slots: []Slot, cap: u32) ConnSlots {
        assert(cap >= 1);
        assert(slots.len == @as(usize, cap) * slots_per_connection);
        for (slots) |*slot| {
            assert(slot.state.load(.monotonic) == .free);
        }

        return .{ .slots = slots, .cap = cap };
    }

    // -- the acceptor ---------------------------------------------------

    /// The first slot in `free` or `reclaimable`, or null when the `live` slots reach the cap or no
    /// slot is in either state.
    pub fn claim(self: *const ConnSlots) ?Claim {
        var live: u32 = 0;
        var found: ?Claim = null;
        for (self.slots, 0..) |*slot, i| {
            switch (slot.state.load(.acquire)) {
                .live => live += 1,
                .free => if (found == null) {
                    found = .{ .index = @intCast(i), .reused = false };
                },
                .reclaimable => if (found == null) {
                    assert(slot.tasks.load(.monotonic) == 0);
                    found = .{ .index = @intCast(i), .reused = true };
                },
                .closing, .swept => {},
            }
        }

        assert(live <= self.cap);
        if (live == self.cap) return null;
        return found;
    }

    /// The id after `prev` that is neither zero nor held by a slot that a task or the engine can
    /// still reach.
    pub fn nextId(self: *const ConnSlots, prev: u32) u32 {
        var id = prev +% 1;
        // Each pass skips zero or one held id, and at most `slots.len` slots hold an id.
        var pass: usize = 0;
        while (pass <= self.slots.len + 1) : (pass += 1) {
            if (id != 0 and !self.held(id)) return id;
            id +%= 1;
        }

        unreachable;
    }

    /// Give a claimed slot to connection `id` with `tasks` tasks. The acceptor must write the
    /// queue of the slot before this call, because the engine can find the slot after it.
    pub fn open(self: *ConnSlots, index: u32, id: u32, tasks: u8) void {
        assert(id != 0);
        assert(!self.held(id));
        assert(tasks >= 1);
        const slot = &self.slots[index];
        const from = slot.state.load(.acquire);
        assert(from == .free or from == .reclaimable);
        assert(slot.tasks.load(.monotonic) == 0);

        slot.id.store(id, .monotonic);
        slot.tasks.store(tasks, .monotonic);
        // The release store publishes the id and the queue to a find that sees `live`.
        transition(slot, from, .live);
    }

    /// Free a slot whose reader did not start. No request names the new id, so the engine holds
    /// no find on the slot. The acceptor must await the started tasks first.
    pub fn abandon(self: *ConnSlots, index: u32) void {
        const slot = &self.slots[index];
        slot.tasks.store(0, .monotonic);
        transition(slot, .live, .free);
    }

    // -- the connection tasks -------------------------------------------

    /// One task of the connection exits. True for the last task. The slot is then `closing`, or
    /// `reclaimable` when the engine has released it already.
    pub fn exitTask(self: *ConnSlots, index: u32) bool {
        if (!self.dropTask(index)) return false;

        self.settleExit(index, self.stateOf(index));
        return true;
    }

    /// The connection id of a slot. A task calls it for its own slot (`connId`, `taskExit`). The
    /// engine thread calls it for each `live` slot when it counts sessions, and in `beginClose` at
    /// teardown for a slot that still has a task.
    pub fn idOf(self: *const ConnSlots, index: u32) u32 {
        const id = self.slots[index].id.load(.monotonic);
        assert(id != 0);
        return id;
    }

    // -- the engine thread ----------------------------------------------

    /// The slot of connection `id` in `live` or `closing`, or null. The slot stays readable until
    /// this thread calls `release` (rule 3). Id zero finds no slot, so a request from inside the
    /// process gets no reply.
    pub fn find(self: *const ConnSlots, id: u32) ?u32 {
        if (id == 0) return null;
        for (self.slots, 0..) |*slot, i| {
            switch (slot.state.load(.acquire)) {
                .live, .closing => if (slot.id.load(.monotonic) == id) return @intCast(i),
                .free, .swept, .reclaimable => {},
            }
        }

        return null;
    }

    /// The engine is done with a slot after it handles `conn_closed`. The slot is then `swept`,
    /// or `reclaimable` when every task has exited already.
    pub fn release(self: *ConnSlots, index: u32) void {
        self.settleRelease(index, self.stateOf(index));
    }

    pub fn stateOf(self: *const ConnSlots, index: u32) State {
        return self.slots[index].state.load(.acquire);
    }

    /// Free a slot at teardown. The acceptor has stopped and every task of the slot has exited.
    /// A `closing` slot here lost its `conn_closed`, because teardown frees the queued requests.
    pub fn reset(self: *ConnSlots, index: u32) void {
        const slot = &self.slots[index];
        const from = slot.state.load(.acquire);
        assert(from != .live);
        assert(from != .swept);
        assert(slot.tasks.load(.monotonic) == 0);
        if (from == .free) return;

        transition(slot, from, .free);
    }

    // -- helpers --------------------------------------------------------

    /// Count one task out. True for the last task.
    fn dropTask(self: *ConnSlots, index: u32) bool {
        const prev = self.slots[index].tasks.fetchSub(1, .acq_rel);
        assert(prev >= 1);
        return prev == 1;
    }

    /// The state change of the last exit, from the state that the task `read`. The schedule test
    /// calls it with a read that the engine has made stale.
    fn settleExit(self: *ConnSlots, index: u32, read: State) void {
        step(&self.slots[index], read, .live, .closing, .swept, .reclaimable);
    }

    /// The state change of `release`, from the state that the engine `read`.
    fn settleRelease(self: *ConnSlots, index: u32, read: State) void {
        step(&self.slots[index], read, .live, .swept, .closing, .reclaimable);
    }

    fn held(self: *const ConnSlots, id: u32) bool {
        for (self.slots) |*slot| {
            switch (slot.state.load(.acquire)) {
                .live, .closing, .swept => if (slot.id.load(.monotonic) == id) return true,
                .free, .reclaimable => {},
            }
        }

        return false;
    }
};

// -- helpers ----------------------------------------------------------------

/// One state change whose source can be `first` or `second`, from the state `read`. `first` moves
/// to `first_to`, and `second` moves to `second_to`. The other thread of rule 5 can move `first`
/// to `second` between the read and the exchange, so a failed exchange reads again once.
fn step(
    slot: *Slot,
    read: State,
    first: State,
    first_to: State,
    second: State,
    second_to: State,
) void {
    var from = read;
    var pass: u32 = 0;
    while (pass < 2) : (pass += 1) {
        const to = if (from == first) first_to else blk: {
            assert(from == second);
            break :blk second_to;
        };
        from = slot.state.cmpxchgStrong(from, to, .acq_rel, .acquire) orelse return;
    }

    unreachable;
}

/// One state change with one writer for its source state. A failed exchange is a bug.
fn transition(slot: *Slot, from: State, to: State) void {
    const prev = slot.state.cmpxchgStrong(from, to, .acq_rel, .acquire);
    assert(prev == null);
}

// -- tests ------------------------------------------------------------------

const prop = @import("../core/utils/prop.zig");
const testing = std.testing;

/// The rules of one model table. The real rules hold every default. Each broken-rule variant
/// changes one field, and the schedule must then trip one named check.
const Rules = struct {
    /// The last exit writes `reclaimable` with no release by the engine.
    exit_reclaims: bool = false,
    /// `find` accepts `live` alone, as the rules before this table did.
    find_live_only: bool = false,
    /// The acceptor checks for an open find when it writes a slot.
    check_init: bool = true,
    /// `real` skips zero and held ids. `small` cycles three ids with no check. `wraps` counts
    /// through zero, and `find` then finds id zero.
    ids: enum { real, small, wraps } = .real,
    /// `reader` is the real rule. `first_exit` lets the first task out submit it.
    closer: enum { reader, first_exit } = .reader,
    /// The exit of a reader closes the queue, as the first fix did.
    reader_closes_queue: bool = false,
    /// The exit of a writer closes the queue.
    writer_closes_queue: bool = true,
    /// `claim` honors the cap.
    cap: bool = true,
};

/// A model table with the rules of `rules`. It keeps the states of the real table, but plain
/// fields, so a broken rule needs no assert of the real table to pass.
fn ModelSlots(comptime rules: Rules) type {
    return struct {
        const Self = @This();
        const slot_count = schedule_cap * slots_per_connection;
        const checks_init = rules.check_init;
        const reader_closes = rules.reader_closes_queue;
        const writer_closes = rules.writer_closes_queue;
        const first_exit_closes = rules.closer == .first_exit;
        const id_start: u32 = if (rules.ids == .wraps) std.math.maxInt(u32) - 2 else 0;

        state: [slot_count]State = @splat(.free),
        id: [slot_count]u32 = @splat(0),
        tasks: [slot_count]u8 = @splat(0),

        fn init(self: *Self) void {
            _ = self;
        }

        fn claim(self: *const Self) ?Claim {
            var live: u32 = 0;
            var found: ?Claim = null;
            for (self.state, 0..) |st, i| {
                switch (st) {
                    .live => live += 1,
                    .free, .reclaimable => if (found == null) {
                        found = .{ .index = @intCast(i), .reused = st == .reclaimable };
                    },
                    .closing, .swept => {},
                }
            }

            if (rules.cap and live >= schedule_cap) return null;
            return found;
        }

        fn nextId(self: *const Self, prev: u32) u32 {
            return switch (rules.ids) {
                .real => blk: {
                    var id = prev +% 1;
                    while (id == 0 or self.held(id)) {
                        id +%= 1;
                    }
                    break :blk id;
                },
                .small => prev % 3 + 1,
                .wraps => prev +% 1,
            };
        }

        fn held(self: *const Self, id: u32) bool {
            for (self.state, self.id) |st, slot_id| {
                switch (st) {
                    .live, .closing, .swept => if (slot_id == id) return true,
                    .free, .reclaimable => {},
                }
            }

            return false;
        }

        fn open(self: *Self, index: u32, id: u32, tasks: u8) void {
            self.id[index] = id;
            self.tasks[index] = tasks;
            self.state[index] = .live;
        }

        fn abandon(self: *Self, index: u32) void {
            self.tasks[index] = 0;
            self.state[index] = .free;
        }

        fn dropTask(self: *Self, index: u32) bool {
            self.tasks[index] -= 1;
            return self.tasks[index] == 0;
        }

        fn stateOf(self: *const Self, index: u32) State {
            return self.state[index];
        }

        fn settleExit(self: *Self, index: u32, read: State) void {
            _ = read;
            if (rules.exit_reclaims) {
                self.state[index] = .reclaimable;
                return;
            }

            self.state[index] = switch (self.state[index]) {
                .live => .closing,
                .swept => .reclaimable,
                else => unreachable,
            };
        }

        fn settleRelease(self: *Self, index: u32, read: State) void {
            _ = read;
            self.state[index] = switch (self.state[index]) {
                .live => .swept,
                .closing => .reclaimable,
                // Under a broken rule the slot can already be gone.
                else => self.state[index],
            };
        }

        fn find(self: *const Self, id: u32) ?u32 {
            if (rules.ids != .wraps and id == 0) return null;
            for (self.state, self.id, 0..) |st, slot_id, i| {
                const readable = st == .live or (!rules.find_live_only and st == .closing);
                if (readable and slot_id == id) return @intCast(i);
            }

            return null;
        }

        fn reset(self: *Self, index: u32) void {
            self.state[index] = .free;
        }
    };
}

/// The real table, as the schedule drives it. The exit and the release run as a read and a
/// settle, so the schedule can move the other side between them.
const RealSlots = struct {
    const slot_count = schedule_cap * slots_per_connection;
    const checks_init = true;
    const reader_closes = false;
    const writer_closes = true;
    const first_exit_closes = false;
    const id_start: u32 = 0;

    buf: [slot_count]Slot = @splat(.{}),
    table: ConnSlots = undefined,

    fn init(self: *RealSlots) void {
        self.table = .init(&self.buf, schedule_cap);
    }

    fn claim(self: *const RealSlots) ?Claim {
        return self.table.claim();
    }

    fn nextId(self: *const RealSlots, prev: u32) u32 {
        return self.table.nextId(prev);
    }

    fn open(self: *RealSlots, index: u32, id: u32, tasks: u8) void {
        self.table.open(index, id, tasks);
    }

    fn abandon(self: *RealSlots, index: u32) void {
        self.table.abandon(index);
    }

    fn dropTask(self: *RealSlots, index: u32) bool {
        return self.table.dropTask(index);
    }

    fn stateOf(self: *const RealSlots, index: u32) State {
        return self.table.stateOf(index);
    }

    fn settleExit(self: *RealSlots, index: u32, read: State) void {
        self.table.settleExit(index, read);
    }

    fn settleRelease(self: *RealSlots, index: u32, read: State) void {
        self.table.settleRelease(index, read);
    }

    fn find(self: *const RealSlots, id: u32) ?u32 {
        return self.table.find(id);
    }

    fn reset(self: *RealSlots, index: u32) void {
        self.table.reset(index);
    }
};

const schedule_cap = 2;
/// The connections of one schedule. Each slot serves several of them in turn.
const schedule_conns = 8;
/// Each reader submits at most this many requests before `conn_closed`.
const schedule_sends = 2;
/// Each writer submits at most this many wakeups.
const schedule_wakes = 2;
/// The requests from inside the process in one schedule.
const schedule_owner_requests = 2;
const request_capacity = schedule_conns * (schedule_sends + schedule_wakes + 1) +
    schedule_owner_requests;

const Violation = error{
    /// The acceptor wrote a slot while a find of the engine on that slot had no put yet.
    InitUnderFind,
    /// A put reached a slot that the acceptor wrote after the find.
    PutAfterInit,
    /// An event for one connection reached the queue of another connection.
    WrongQueue,
    /// A reply reached a queue that a reader exit closed, so the client lost it.
    ReplyLost,
    /// The engine handled a request of a connection that it had swept.
    RequestAfterSweep,
    /// A request from inside the process found a connection slot.
    OwnerFound,
    /// The engine found no slot for a `conn_closed`.
    ClosedSlotLost,
    /// More slots were `live` than the cap allows.
    CapExceeded,
    /// A slot held a connection after every client left and the engine went idle.
    SlotStuck,
};

const Action = enum {
    accept_init,
    accept_open,
    accept_fail,
    task_send,
    task_block,
    task_wake,
    task_exit,
    task_settle,
    owner_send,
    engine_find,
    engine_handle,
    engine_put,
    engine_settle,
};

const RequestKind = enum { data, wake, closed, owner };

/// `conn` is the ordinal of the connection in the schedule. An owner request has none.
const Request = struct { kind: RequestKind, conn: u8 };

const FindKind = enum { emit, reply, close };

const Find = struct { index: u32, conn: u8, epoch: u32, kind: FindKind };

/// A read of a two-step state change that waits for its settle.
const Settle = struct { index: u32, read: State };

const CloseKind = enum { open, reader, writer, engine };

const ConnModel = struct {
    id: u32 = 0,
    slot: u32 = 0,
    reader: bool = false,
    writer: bool = false,
    /// The reader waits in a put into its own full queue. Only a queue close ends the wait.
    blocked: bool = false,
    sends: u8 = 0,
    wakes: u8 = 0,
    exits: u8 = 0,
    swept: bool = false,
    /// The last exit read the slot state and waits to settle.
    settle: ?Settle = null,

    fn tasks(c: ConnModel) u8 {
        return @as(u8, @intFromBool(c.reader)) + @intFromBool(c.writer);
    }
};

/// One seeded schedule of the acceptor, the tasks, and the engine over `Table`. Find and put, and
/// each read and settle, are separate steps, so another actor can run between them. An init sets
/// the owner of the queue of a slot, counts its epoch up, and opens it.
fn Schedule(comptime Table: type) type {
    return struct {
        const Self = @This();
        const slot_count = Table.slot_count;

        table: Table = .{},
        queue_owner: [slot_count]u8 = @splat(0),
        epoch: [slot_count]u32 = @splat(0),
        queue_close: [slot_count]CloseKind = @splat(.open),
        conns: [schedule_conns]ConnModel = @splat(.{}),
        opened: u8 = 0,
        last_id: u32 = Table.id_start,
        claimed: ?Claim = null,
        requests: [request_capacity]Request = undefined,
        head: u32 = 0,
        tail: u32 = 0,
        owner_requests: u8 = 0,
        /// The connections that own a session in the scheduler.
        owners: std.StaticBitSet(schedule_conns) = .initEmpty(),
        pending: ?Find = null,
        release: ?Settle = null,

        fn run(seed: u64, steps: u32) Violation!void {
            var prng = std.Random.DefaultPrng.init(seed);
            const random = prng.random();
            var self: Self = .{};
            self.table.init();
            // Half of the schedules start next to the end of the id range, so ids wrap.
            if (Table.id_start == 0 and random.boolean()) {
                self.last_id = std.math.maxInt(u32) - 3;
            }

            var step_n: u32 = 0;
            while (step_n < steps) : (step_n += 1) {
                var enabled: [@typeInfo(Action).@"enum".fields.len]Action = undefined;
                const n = self.enabledActions(&enabled);
                if (n == 0) break;
                try self.apply(enabled[random.uintLessThan(usize, n)], random);
                try self.checkCap();
            }

            if (random.boolean()) {
                try self.clientsLeave(random);
            } else {
                try self.teardown(random);
            }
        }

        fn enabledActions(self: *const Self, out: []Action) usize {
            var n: usize = 0;
            const add = struct {
                fn f(o: []Action, i: *usize, a: Action) void {
                    o[i.*] = a;
                    i.* += 1;
                }
            }.f;
            const can_accept = self.claimed == null and self.opened < schedule_conns and
                self.table.claim() != null;
            if (can_accept) {
                add(out, &n, .accept_init);
            }
            if (self.claimed != null) {
                add(out, &n, .accept_open);
                add(out, &n, .accept_fail);
            }
            if (self.anyConn(.send)) {
                add(out, &n, .task_send);
            }
            if (self.anyConn(.block)) {
                add(out, &n, .task_block);
            }
            if (self.anyConn(.wake)) {
                add(out, &n, .task_wake);
            }
            if (self.anyConn(.exit)) {
                add(out, &n, .task_exit);
            }
            if (self.anyConn(.settle)) {
                add(out, &n, .task_settle);
            }
            if (self.owner_requests < schedule_owner_requests) {
                add(out, &n, .owner_send);
            }
            // The engine thread runs one step at a time, so an open find or release blocks its
            // other steps.
            if (self.release != null) {
                add(out, &n, .engine_settle);
            } else if (self.pending != null) {
                add(out, &n, .engine_put);
            } else {
                if (self.owners.count() > 0) {
                    add(out, &n, .engine_find);
                }
                if (self.head != self.tail) {
                    add(out, &n, .engine_handle);
                }
            }

            return n;
        }

        const Need = enum { send, block, wake, exit, settle };

        fn wants(self: *const Self, c: ConnModel, need: Need) bool {
            return switch (need) {
                .send => c.reader and !c.blocked and c.sends < schedule_sends,
                // Only a binary reader puts into its own queue, a PONG, while its writer runs.
                .block => c.reader and c.writer and !c.blocked and
                    self.queue_close[c.slot] == .open,
                .wake => c.writer and c.wakes < schedule_wakes,
                .exit => (c.reader and !c.blocked) or c.writer,
                .settle => c.settle != null,
            };
        }

        fn anyConn(self: *const Self, need: Need) bool {
            for (self.conns[0..self.opened]) |c| {
                if (self.wants(c, need)) return true;
            }

            return false;
        }

        fn pickConn(self: *const Self, random: std.Random, need: Need) u8 {
            var ordinals: [schedule_conns]u8 = undefined;
            var n: usize = 0;
            for (self.conns[0..self.opened], 0..) |c, i| {
                if (self.wants(c, need)) {
                    ordinals[n] = @intCast(i);
                    n += 1;
                }
            }

            assert(n > 0);
            return ordinals[random.uintLessThan(usize, n)];
        }

        fn apply(self: *Self, action: Action, random: std.Random) Violation!void {
            switch (action) {
                .accept_init => {
                    const c = self.table.claim().?;
                    try self.checkInit(c.index);
                    self.epoch[c.index] += 1;
                    self.queue_owner[c.index] = self.opened;
                    self.queue_close[c.index] = .open;
                    self.claimed = c;
                },
                .accept_open => try self.acceptOpen(random),
                .accept_fail => {
                    // The writer starts, the reader does not, and the acceptor awaits the writer.
                    const c = self.claimed.?;
                    try self.checkInit(c.index);
                    const id = self.table.nextId(self.last_id);
                    self.table.open(c.index, id, 2);
                    const last = self.table.dropTask(c.index);
                    assert(!last);
                    self.table.abandon(c.index);
                    self.claimed = null;
                },
                .task_send => {
                    const ordinal = self.pickConn(random, .send);
                    self.conns[ordinal].sends += 1;
                    self.submit(.{ .kind = .data, .conn = ordinal });
                },
                .task_block => self.conns[self.pickConn(random, .block)].blocked = true,
                .task_wake => {
                    const ordinal = self.pickConn(random, .wake);
                    self.conns[ordinal].wakes += 1;
                    self.submit(.{ .kind = .wake, .conn = ordinal });
                },
                .task_exit => self.taskExit(self.pickConn(random, .exit), random, true),
                .task_settle => {
                    const c = &self.conns[self.pickConn(random, .settle)];
                    const s = c.settle.?;
                    c.settle = null;
                    self.table.settleExit(s.index, s.read);
                },
                .owner_send => {
                    self.owner_requests += 1;
                    self.submit(.{ .kind = .owner, .conn = 0 });
                },
                .engine_find => self.findFor(self.pickOwner(random), .emit),
                .engine_handle => try self.handle(),
                .engine_put => try self.put(),
                .engine_settle => {
                    const s = self.release.?;
                    self.release = null;
                    self.table.settleRelease(s.index, s.read);
                },
            }
        }

        fn checkInit(self: *const Self, index: u32) Violation!void {
            if (!Table.checks_init) return;
            if (self.pending) |p| {
                if (p.index == index) return error.InitUnderFind;
            }
        }

        fn acceptOpen(self: *Self, random: std.Random) Violation!void {
            const c = self.claimed.?;
            try self.checkInit(c.index);
            // A binary connection runs a reader and a writer, and an HTTP one a reader.
            const binary = random.boolean();
            const id = self.table.nextId(self.last_id);
            self.table.open(c.index, id, if (binary) 2 else 1);
            self.conns[self.opened] = .{
                .id = id,
                .slot = c.index,
                .reader = true,
                .writer = binary,
            };
            self.last_id = id;
            self.opened += 1;
            self.claimed = null;
        }

        /// One task of the connection exits: a reader at the end of its stream, a writer at a
        /// write error or a closed queue. `queue_open` is false at teardown, which frees the
        /// queued requests.
        fn taskExit(self: *Self, ordinal: u8, random: std.Random, queue_open: bool) void {
            const c = &self.conns[ordinal];
            const reader_free = c.reader and !c.blocked;
            const reader_exits = reader_free and (!c.writer or random.boolean());
            const first = c.exits == 0;
            c.exits += 1;
            if (reader_exits) {
                c.reader = false;
                if (Table.reader_closes) {
                    self.closeQueue(c.slot, .reader);
                }
                const submits = if (Table.first_exit_closes) first else true;
                if (queue_open and submits) {
                    self.submit(.{ .kind = .closed, .conn = ordinal });
                }
            } else {
                assert(c.writer);
                c.writer = false;
                if (Table.writer_closes) {
                    self.closeQueue(c.slot, .writer);
                }
                if (Table.first_exit_closes and first and queue_open) {
                    self.submit(.{ .kind = .closed, .conn = ordinal });
                }
            }

            if (!self.table.dropTask(c.slot)) return;

            assert(c.tasks() == 0);
            c.settle = .{ .index = c.slot, .read = self.table.stateOf(c.slot) };
        }

        fn closeQueue(self: *Self, index: u32, kind: CloseKind) void {
            if (self.queue_close[index] == .open) {
                self.queue_close[index] = kind;
            }
            // A close ends the wait of a reader that puts into the queue.
            for (self.conns[0..self.opened]) |*c| {
                if (c.slot == index and self.queue_owner[index] == self.ordinalOf(c)) {
                    c.blocked = false;
                }
            }
        }

        fn ordinalOf(self: *const Self, c: *const ConnModel) u8 {
            return @intCast((@intFromPtr(c) - @intFromPtr(&self.conns[0])) / @sizeOf(ConnModel));
        }

        fn submit(self: *Self, req: Request) void {
            assert(self.tail - self.head < request_capacity);
            self.requests[self.tail % request_capacity] = req;
            self.tail += 1;
        }

        fn pickOwner(self: *const Self, random: std.Random) u8 {
            const n = self.owners.count();
            assert(n > 0);
            var skip = random.uintLessThan(usize, n);
            var it = self.owners.iterator(.{});
            while (it.next()) |ordinal| {
                if (skip == 0) return @intCast(ordinal);
                skip -= 1;
            }

            unreachable;
        }

        fn findFor(self: *Self, ordinal: u8, kind: FindKind) void {
            const index = self.table.find(self.conns[ordinal].id) orelse return;
            const epoch = self.epoch[index];
            self.pending = .{ .index = index, .conn = ordinal, .epoch = epoch, .kind = kind };
        }

        fn handle(self: *Self) Violation!void {
            const req = self.requests[self.head % request_capacity];
            self.head += 1;
            switch (req.kind) {
                .owner => if (self.table.find(0) != null) return error.OwnerFound,
                .wake => {},
                .data => {
                    if (self.conns[req.conn].swept) {
                        return error.RequestAfterSweep;
                    }
                    self.owners.set(req.conn);
                    self.findFor(req.conn, .reply);
                },
                .closed => {
                    if (self.conns[req.conn].swept) return error.RequestAfterSweep;
                    self.findFor(req.conn, .close);
                    if (self.pending == null) return error.ClosedSlotLost;
                },
            }
        }

        fn put(self: *Self) Violation!void {
            const p = self.pending.?;
            self.pending = null;
            if (self.epoch[p.index] != p.epoch) return error.PutAfterInit;
            if (self.queue_owner[p.index] != p.conn) return error.WrongQueue;
            switch (p.kind) {
                .emit => {},
                // A queue that a writer closed has no reader of replies, and the engine closes
                // only after the last request. A reader exit must not close it.
                .reply => if (self.queue_close[p.index] == .reader) return error.ReplyLost,
                .close => {
                    self.closeQueue(p.index, .engine);
                    self.owners.unset(p.conn);
                    self.conns[p.conn].swept = true;
                    self.release = .{ .index = p.index, .read = self.table.stateOf(p.index) };
                },
            }
        }

        fn checkCap(self: *const Self) Violation!void {
            var live: u32 = 0;
            for (0..slot_count) |i| {
                if (self.table.stateOf(@intCast(i)) == .live) {
                    live += 1;
                }
            }

            if (live > schedule_cap) return error.CapExceeded;
        }

        /// Every client leaves: a free reader sees the end of its stream, a writer meets a write
        /// error, and a reader that waits on a full queue leaves only at a close. Then every slot
        /// must be `free` or `reclaimable`.
        fn clientsLeave(self: *Self, random: std.Random) Violation!void {
            if (self.claimed != null) {
                try self.acceptOpen(random);
            }

            // Each pass makes progress or ends the loop. The work is bounded by the tasks, the
            // requests, and one settle per connection and per release.
            const passes_max = 4 * (schedule_conns * 4 + request_capacity) + 8;
            var pass: u32 = 0;
            while (pass < passes_max) : (pass += 1) {
                if (self.release != null) {
                    try self.apply(.engine_settle, random);
                } else if (self.pending != null) {
                    try self.put();
                } else if (self.anyConn(.settle)) {
                    try self.apply(.task_settle, random);
                } else if (self.anyConn(.exit)) {
                    self.taskExit(self.pickConn(random, .exit), random, true);
                } else if (self.head != self.tail) {
                    try self.handle();
                } else {
                    break;
                }
            }

            assert(self.pending == null);
            assert(self.release == null);
            for (0..slot_count) |i| {
                switch (self.table.stateOf(@intCast(i))) {
                    .free, .reclaimable => {},
                    .live, .closing, .swept => return error.SlotStuck,
                }
            }
        }

        /// The engine ends its step, the request queue closes with its requests freed, each
        /// queue closes, the tasks exit and settle, and teardown resets every slot.
        fn teardown(self: *Self, random: std.Random) Violation!void {
            if (self.pending != null) {
                try self.put();
            }
            if (self.release != null) {
                try self.apply(.engine_settle, random);
            }
            if (self.claimed != null) {
                try self.acceptOpen(random);
            }
            self.head = self.tail;
            for (0..slot_count) |i| {
                self.closeQueue(@intCast(i), .engine);
            }
            for (self.conns[0..self.opened], 0..) |c, ordinal| {
                var left = c.tasks();
                while (left > 0) : (left -= 1) {
                    self.taskExit(@intCast(ordinal), random, false);
                }
            }
            for (self.conns[0..self.opened]) |*c| {
                const s = c.settle orelse continue;
                c.settle = null;
                self.table.settleExit(s.index, s.read);
            }

            for (0..slot_count) |i| {
                switch (self.table.stateOf(@intCast(i))) {
                    .live, .swept => return error.SlotStuck,
                    .free, .closing, .reclaimable => self.table.reset(@intCast(i)),
                }
            }
        }
    };
}

const ScheduleCase = struct { seed: u64, steps: u32 };

fn genSchedule(_: void, random: std.Random, size: u32) ScheduleCase {
    return .{ .seed = random.int(u64), .steps = 8 + 4 * size };
}

fn checkReal(_: void, case: ScheduleCase) anyerror!void {
    return Schedule(RealSlots).run(case.seed, case.steps);
}

test "conn_slots: no schedule breaks a slot rule, and every slot comes back" {
    try prop.forAll(.{ .site = 0x434f_4e4e_0001 }, {}, ScheduleCase, genSchedule, checkReal);
}

/// The count of fixed-seed schedules under `rules` that end in `want`, or in any violation when
/// `want` is null. The seeds do not depend on the test seed, so the count is fixed.
fn tripCount(comptime rules: Rules, want: ?Violation) u32 {
    const cases = 512;
    var hits: u32 = 0;
    var prng = std.Random.DefaultPrng.init(0x434f_4e4e_0002);
    var i: u32 = 0;
    while (i < cases) : (i += 1) {
        const case = genSchedule({}, prng.random(), @min(i, 64));
        Schedule(ModelSlots(rules)).run(case.seed, case.steps) catch |err| {
            if (want == null or err == want.?) {
                hits += 1;
            }
        };
    }

    return hits;
}

test "conn_slots: each check of the schedule trips under one broken rule" {
    // The model table with the real rules trips no check, so each variant below changes one rule.
    try testing.expectEqual(@as(u32, 0), tripCount(.{}, null));

    const old: Rules = .{ .exit_reclaims = true, .find_live_only = true };
    try testing.expect(tripCount(old, error.InitUnderFind) > 0);
    try testing.expect(tripCount(.{
        .exit_reclaims = true,
        .find_live_only = true,
        .check_init = false,
    }, error.PutAfterInit) > 0);
    try testing.expect(tripCount(.{ .ids = .small }, error.WrongQueue) > 0);
    try testing.expect(tripCount(.{ .reader_closes_queue = true }, error.ReplyLost) > 0);
    try testing.expect(tripCount(.{ .closer = .first_exit }, error.RequestAfterSweep) > 0);
    try testing.expect(tripCount(.{ .ids = .wraps }, error.OwnerFound) > 0);
    try testing.expect(tripCount(.{ .exit_reclaims = true }, error.ClosedSlotLost) > 0);
    try testing.expect(tripCount(.{ .cap = false }, error.CapExceeded) > 0);
    try testing.expect(tripCount(.{ .writer_closes_queue = false }, error.SlotStuck) > 0);
}

/// The count of interleavings, of one idle check that loads in `order` with the two engine writes
/// of a session open, that read zero from both counters. The connection holds a request or a
/// session at every point, so a correct order gives zero.
fn idleMisreads(order: [2]IdleCounter) u32 {
    // Bit i of a schedule gives step i to the reader (1) or to the engine (0). Each side takes
    // two steps, so a schedule has two set bits of four.
    const steps = 4;
    var misreads: u32 = 0;
    for (0..1 << steps) |schedule| {
        if (@popCount(schedule) != 2) continue;

        var sessions: u32 = 0;
        var requests_open: u32 = 1;
        var engine_step: u32 = 0;
        var reader_step: u32 = 0;
        var zeros: u32 = 0;
        for (0..steps) |i| {
            if ((schedule >> @intCast(i)) & 1 == 1) {
                const value = switch (order[reader_step]) {
                    .requests_open => requests_open,
                    .sessions => sessions,
                };
                if (value == 0) {
                    zeros += 1;
                }
                reader_step += 1;
            } else {
                switch (engine_step) {
                    0 => sessions = 1,
                    1 => requests_open = 0,
                    else => unreachable,
                }
                engine_step += 1;
            }
        }
        if (zeros == 2) {
            misreads += 1;
        }
    }

    return misreads;
}

test "conn_slots: an idle check never reads an opening session as idle" {
    try testing.expectEqual(@as(u32, 0), idleMisreads(idle_load_order));
    // The other order misreads, so the count detects a swap.
    try testing.expect(idleMisreads(.{ idle_load_order[1], idle_load_order[0] }) > 0);
}

test "conn_slots: a settle from a read that the other side made stale retries once" {
    var buf: [2]Slot = @splat(.{});
    var t: ConnSlots = .init(&buf, 1);

    // The last exit reads `live`, then the engine releases before the exit settles.
    t.open(t.claim().?.index, 1, 1);
    try testing.expect(t.dropTask(0));
    const exit_read = t.stateOf(0);
    t.release(0);
    try testing.expectEqual(State.swept, t.stateOf(0));
    t.settleExit(0, exit_read);
    try testing.expectEqual(State.reclaimable, t.stateOf(0));

    // The engine reads `live`, then the last exit settles before the release.
    t.open(t.claim().?.index, 2, 1);
    const release_read = t.stateOf(0);
    try testing.expect(t.exitTask(0));
    try testing.expectEqual(State.closing, t.stateOf(0));
    t.settleRelease(0, release_read);
    try testing.expectEqual(State.reclaimable, t.stateOf(0));
}

test "conn_slots: a slot returns to the acceptor after both the engine and the last task" {
    var buf: [2]Slot = @splat(.{});
    var t: ConnSlots = .init(&buf, 1);

    // The last exit first, then the release.
    const c = t.claim().?;
    try testing.expectEqual(Claim{ .index = 0, .reused = false }, c);
    t.open(c.index, 7, 2);
    try testing.expect(!t.exitTask(0));
    try testing.expect(t.exitTask(0));
    try testing.expectEqual(State.closing, t.stateOf(0));
    try testing.expectEqual(@as(?u32, 0), t.find(7));
    try testing.expectEqual(@as(?u32, null), t.find(0));
    t.release(0);
    try testing.expectEqual(State.reclaimable, t.stateOf(0));
    try testing.expectEqual(@as(?u32, null), t.find(7));

    // The release first, then the last exit.
    t.open(t.claim().?.index, 8, 2);
    try testing.expectEqual(State.live, t.stateOf(0));
    t.release(0);
    try testing.expectEqual(State.swept, t.stateOf(0));
    try testing.expectEqual(@as(?u32, null), t.find(8));
    try testing.expect(!t.exitTask(0));
    try testing.expect(t.exitTask(0));
    try testing.expectEqual(State.reclaimable, t.stateOf(0));
}

test "conn_slots: the cap counts live slots, so a closing slot does not refuse a client" {
    const cap = 2;
    var buf: [cap * slots_per_connection]Slot = @splat(.{});
    var t: ConnSlots = .init(&buf, cap);

    var id: u32 = 0;
    for (0..cap) |_| {
        id = t.nextId(id);
        t.open(t.claim().?.index, id, 1);
    }
    try testing.expectEqual(@as(?Claim, null), t.claim());

    // The last task of one connection exits, and the engine has not run yet.
    try testing.expect(t.exitTask(0));
    try testing.expectEqual(State.closing, t.stateOf(0));
    const c = t.claim().?;
    try testing.expect(c.index != 0);
    id = t.nextId(id);
    t.open(c.index, id, 1);
    try testing.expectEqual(@as(?Claim, null), t.claim());
}

test "conn_slots: an id is never zero and never one that a slot holds" {
    var buf: [2 * slots_per_connection]Slot = @splat(.{});
    var t: ConnSlots = .init(&buf, 2);

    const max = std.math.maxInt(u32);
    t.open(t.claim().?.index, max, 1);
    t.open(t.claim().?.index, 1, 1);
    // The successor of the largest id wraps past zero and past the held id 1.
    try testing.expectEqual(@as(u32, 2), t.nextId(max));
    try testing.expectEqual(@as(u32, 2), t.nextId(max - 1));
}

test "conn_slots: teardown frees a slot whose conn_closed was lost" {
    var buf: [2]Slot = @splat(.{});
    var t: ConnSlots = .init(&buf, 1);

    t.open(t.claim().?.index, 1, 1);
    try testing.expect(t.exitTask(0));
    t.reset(0);
    try testing.expectEqual(State.free, t.stateOf(0));

    // An abandoned slot is free again, after its started writer exits.
    t.open(t.claim().?.index, 2, 2);
    try testing.expect(!t.exitTask(0));
    t.abandon(0);
    try testing.expectEqual(State.free, t.stateOf(0));
}
