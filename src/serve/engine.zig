//! The transport-agnostic serve engine: one engine thread around the scheduler, a bounded request
//! queue in, a bounded out-queue per connection, and the connection-slot lifecycle. It owns all
//! serve-layer state and knows no wire format. An `Adapter` supplies the per-connection task
//! model: how many tasks a connection runs, and how they map wire bytes to and from the
//! `Request` and `OutEvent` values. `serve/server.zig` is the binary adapter, and
//! `serve/openai/http_server.zig` is the HTTP one. Both are thin maps onto this engine.
//!
//!     acceptor --accept--> Adapter.onAccept(conn) --decode--> request_queue
//!                                                                  |
//!     conn[i] task(s) <--encode-- out_queue[i] <--emit-- ENGINE LOOP (run)
//!
//! Comptime. Generic over the engine, the codec, and the adapter, so each transport builds its own
//! engine type.
//!
//! Bounds. `Config.max_connections` live connections (at most `limits.max_connections`) in a table
//! of `conn_slots.slots_per_connection` slots per allowed connection, one `request_queue_capacity`
//! request queue, and one `out_queue_capacity` event queue per slot. The engine never blocks on a
//! client. Emission is non-blocking, and a full output
//! queue parks the event in the cell of the session while the draining task signals a `wakeup`.
//!
//! Connection slots. `conn_slots.zig` holds the state of each slot and the rules that keep the
//! engine thread away from a slot that the acceptor rewrites. Slot i of that table owns `conns[i]`.
//!
//! Close order. The reader task submits `conn_closed` when it stops, after its last request. The
//! engine handles the requests in order, so the reply that a request gets while the engine
//! handles it is in the out-queue before `conn_closed`. Then the engine closes the out-queue,
//! sweeps the sessions of the connection, and releases the slot. The sweep ends each turn in
//! flight. So a binary client that closes its send side ends its turns. A writer task drains the
//! queue, flushes, and exits. A writer that stops early closes the queue, so a reader that waits
//! on a full queue wakes. The slot returns to the acceptor after the release and the last task
//! exit, in either order.
//!
//! Idle. A binary reader closes a connection that holds no session and no request in the engine
//! after one `--io-timeout` with no frame. `sessions` and `requests_open` of a `Conn` tell it: the
//! engine thread alone writes `sessions`, the reader adds to `requests_open` before each request,
//! and the engine subtracts after it handles the request. The engine stores `sessions` before it
//! subtracts, so the reader loads `requests_open` first (`conn_slots.idle_load_order`). A zero
//! there comes after the store, and the load of `sessions` then reads the new count.
//!
//! Socket close. Exactly one actor closes the socket of a connection. The first to claim it owns
//! the close. The claim is `live -> task` when the last task exits, or `live -> teardown` when
//! teardown must shut the fd down to wake a blocked task. The other actor must not touch the fd,
//! because a `shutdown` or `close` after the other `close` is an EBADF.
//!
//! Adapter surface. All methods are duck-typed over the engine and a `*Conn`:
//!   - `pub const ConnExt`            per-connection state the adapter owns
//!   - `pub const conn_tasks: u8`     the task count of one connection. One of
//!                                    them is the reader, which submits requests
//!   - `onAccept(engine, conn) bool`  spawn the task(s) into `conn.ext`. False
//!                                    (with the stream closed and the started
//!                                    tasks awaited) if spawning failed
//!   - `beginClose(engine, conn)`     unblock the task(s) at teardown. Close
//!                                    the out-queue, and shut the socket down
//!                                    so a blocked read or write returns
//!   - `awaitConn(engine, conn)`      await the task futures in `conn.ext`
//!   - `refuse(io, *Stream)`          reject a connection at the cap with one
//!                                    send that does not wait

const std = @import("std");
const span = @import("../core/span.zig");
const Io = std.Io;
const net = Io.net;
const Allocator = std.mem.Allocator;
const limits = @import("../core/limits.zig");
const event = @import("event.zig");
const scheduler_mod = @import("scheduler.zig");
const conn_slots = @import("conn_slots.zig");
const timed_stream = @import("timed_stream.zig");
const Request = event.Request;
const OutEvent = event.OutEvent;

const log = std.log.scoped(.serve);

// -- constants --------------------------------------------------------------

pub const request_queue_capacity: usize = 64;
pub const out_queue_capacity: usize = 256;

/// The wait after an `accept` failure that a free resource ends. A pending client keeps
/// `EMFILE` true, so a retry with no wait spins.
const accept_backoff_ms: i64 = 100;

// The actor that closes the socket of a connection.
const fin_live: u8 = 0;
const fin_task: u8 = 1;
const fin_teardown: u8 = 2;

// -- types ------------------------------------------------------------------

/// The part of a connection task in the close order. The reader submits the requests.
pub const TaskRole = enum { reader, writer };

/// Where to bind. Generic across adapters. Both speak TCP or a unix socket.
pub const Endpoint = union(enum) {
    tcp: struct { host: []const u8 = "127.0.0.1", port: u16 = 0 },
    unix: []const u8,
};

/// The answer of the acceptor to one failed `accept`.
pub const AcceptAction = enum {
    /// The task ends, because the listener cannot serve again.
    stop,
    /// The failure belongs to one client, so the next client is unaffected.
    retry_now,
    /// The failure belongs to the host, so an immediate retry fails again.
    back_off,
};

/// `Engine` and `CodecT` are the scheduler boundaries (see scheduler.zig). `Adapter` is the
/// transport. A value must not move once `run` has started, because tasks hold `*Self`.
pub fn ServeEngine(comptime Engine: type, comptime CodecT: type, comptime Adapter: type) type {
    return struct {
        const Self = @This();
        const SchedulerT = scheduler_mod.Scheduler(Engine, CodecT);

        /// A connection slot: the fields the engine and sink touch, plus the adapter state (`ext`).
        /// `slots` holds its state and its connection id.
        pub const Conn = struct {
            index: u32,
            stream: net.Stream = undefined,
            out_q: Io.Queue(OutEvent) = undefined,
            out_buf: [out_queue_capacity]OutEvent = undefined,
            out_closed: std.atomic.Value(bool) = .init(false),
            /// Single-owner guard for closing `stream` (see `fin_*`).
            fin: std.atomic.Value(u8) = .init(fin_live),
            parked_hint: std.atomic.Value(bool) = .init(false),
            /// The sessions attached to the connection, as the engine thread last counted them.
            sessions: std.atomic.Value(u32) = .init(0),
            /// The requests that the reader submitted and the engine has not handled yet.
            requests_open: std.atomic.Value(u32) = .init(0),
            ext: Adapter.ConnExt = .{},
        };

        gpa: Allocator,
        io: Io,
        sched: SchedulerT,
        /// The codec, also held here so a connection task can read model metadata.
        /// The scheduler holds the same pointer for tokenization.
        codec: *const CodecT,
        listener: net.Server,
        conns: []Conn,
        slots: conn_slots.ConnSlots,
        request_q: Io.Queue(Request),
        /// The id of the last accepted connection. Only the acceptor reads and writes it.
        last_conn_id: u32 = 0,
        /// The deadline of one socket read or write. It never changes, so any task reads it.
        io_timeout_ms: u32,
        acceptor_future: ?Io.Future(void) = null,

        /// Binds the endpoint, so `boundAddress` is valid before `run`, and wires the scheduler.
        /// The returned value can move until `run` is called.
        pub fn init(
            arena: Allocator,
            gpa: Allocator,
            io: Io,
            engine: *Engine,
            codec: *const CodecT,
            sched_cfg: scheduler_mod.Config,
            endpoint: Endpoint,
        ) !Self {
            std.debug.assert(sched_cfg.max_connections >= 1);
            std.debug.assert(sched_cfg.max_connections <= limits.max_connections);

            const listener = switch (endpoint) {
                .tcp => |t| blk: {
                    const addr = try net.IpAddress.parse(t.host, t.port);
                    break :blk try addr.listen(io, .{ .reuse_address = true });
                },
                .unix => |path| blk: {
                    const ua = try net.UnixAddress.init(path);
                    break :blk try ua.listen(io, .{});
                },
            };

            log.debug("listen: max {d} connections, max {d} sessions, max_seq {d}", .{
                sched_cfg.max_connections,
                sched_cfg.max_sessions,
                sched_cfg.max_seq,
            });
            const slot_count = sched_cfg.max_connections * conn_slots.slots_per_connection;
            const conns = try arena.alloc(Conn, slot_count);
            for (conns, 0..) |*c, i| {
                c.* = .{ .index = @intCast(i) };
            }
            const slots = try arena.alloc(conn_slots.Slot, slot_count);
            @memset(slots, .{});
            const slot_bytes = @sizeOf(Conn) + @sizeOf(conn_slots.Slot);
            log.debug("listen: {d} connection slots of {d} bytes, {d} bytes in all", .{
                slot_count,
                slot_bytes,
                slot_count * slot_bytes,
            });

            const request_buf = try arena.alloc(Request, request_queue_capacity);
            var self: Self = .{
                .gpa = gpa,
                .io = io,
                .sched = try SchedulerT.init(arena, gpa, io, engine, codec, sched_cfg),
                .codec = codec,
                .listener = listener,
                .conns = conns,
                .slots = .init(slots, sched_cfg.max_connections),
                .io_timeout_ms = sched_cfg.io_timeout_ms,
                .request_q = .init(request_buf),
            };
            // Warm the caches here, single-threaded, before any connection exists.
            // The scheduler holds no interior pointers, so the value still moves.
            self.sched.preloadPrefixes();
            return self;
        }

        /// The bound address, with the kernel-resolved port after a `port = 0` TCP bind.
        /// The READY line prints it.
        pub fn boundAddress(self: *const Self) net.IpAddress {
            return self.listener.socket.address;
        }

        /// The engine loop. Returns after a clean `shutdown` handshake.
        pub fn run(self: *Self) !void {
            span.nameThread("zf-engine");
            self.acceptor_future = try self.io.concurrent(acceptorTask, .{self});
            var sink: Sink = .{ .engine = self };

            main: while (true) {
                var batch: [16]Request = undefined;
                const n = self.request_q.get(self.io, &batch, 0) catch break;
                for (batch[0..n], 0..) |req, i| {
                    if (self.dispatch(req, &sink) == .shutdown) {
                        for (batch[i + 1 .. n]) |rest| {
                            freeRequestText(self.gpa, rest);
                        }

                        break :main;
                    }
                }

                const tick = self.sched.tick(&sink);
                if (n > 0) {
                    log.debug("tick: {d} requests, {s}", .{ n, @tagName(tick) });
                }

                if (tick != .idle) continue;
                if (n > 0) continue;

                log.debug("idle: blocking on request queue", .{});
                const req = self.request_q.getOne(self.io) catch break;
                if (self.dispatch(req, &sink) == .shutdown) break;
            }

            self.teardown();
        }

        fn dispatch(self: *Self, req: Request, sink: *Sink) scheduler_mod.HandleResult {
            if (req.body != .conn_closed) return self.dispatchRequest(req, sink);

            // Only this thread releases a slot, and it releases one only here. So the slot of the
            // reader that submitted this request is still `live` or `closing`.
            const index = self.slots.find(req.conn).?;
            // Every reply of the connection is in the queue, so the close loses none. The writer
            // task drains the queue and exits.
            self.closeOutput(&self.conns[index]);
            const result = self.sched.handle(req, sink);
            std.debug.assert(result == .ok);
            // The sweep left no session of the connection, so no later event names it.
            self.slots.release(index);
            return result;
        }

        fn dispatchRequest(self: *Self, req: Request, sink: *Sink) scheduler_mod.HandleResult {
            // A counted request comes before the conn_closed of its reader, so its slot is still
            // `live` or `closing`.
            const counted = req.conn != 0 and req.body != .wakeup;
            const from: ?u32 = if (counted) self.slots.find(req.conn).? else null;
            const result = self.sched.handle(req, sink);

            // Only these requests attach a session to a connection or take one from it. The
            // count comes before the request leaves `requests_open`, so a reader that loads
            // `requests_open` first never sees zero of both while a session opens.
            switch (req.body) {
                .session_open, .session_clone, .session_resume, .session_close => {
                    self.countSessions();
                },
                else => {},
            }

            if (from) |index| {
                const prev = self.conns[index].requests_open.fetchSub(1, .acq_rel);
                std.debug.assert(prev >= 1);
            }

            return result;
        }

        /// Store the attached sessions of every `live` slot. A resume can take a session from
        /// another connection, so the count covers every slot.
        fn countSessions(self: *Self) void {
            for (self.conns) |*conn| {
                if (self.slots.stateOf(conn.index) != .live) continue;

                const n = self.sched.sessionsOf(self.slots.idOf(conn.index));
                conn.sessions.store(n, .release);
            }
        }

        fn teardown(self: *Self) void {
            // Stop the acceptor. Closing the listening fd does not unblock accept(2), so cancel it.
            log.debug("shutdown: tearing down engine", .{});
            self.request_q.close(self.io);
            if (self.acceptor_future) |*f| {
                f.cancel(self.io);
            }

            // Unblock the tasks. Only a `live` or `swept` slot has a task.
            self.listener.deinit(self.io);
            for (self.conns) |*conn| {
                switch (self.slots.stateOf(conn.index)) {
                    .live, .swept => Adapter.beginClose(self, conn),
                    .free, .closing, .reclaimable => {},
                }
            }

            // Await the tasks, and close the sockets that teardown claimed. The request queue is
            // closed and its requests are freed, so a slot can stay closing with its conn_closed
            // lost.
            for (self.conns) |*conn| {
                if (self.slots.stateOf(conn.index) == .free) continue;

                Adapter.awaitConn(self, conn);
                std.debug.assert(conn.fin.load(.acquire) != fin_live);
                // A task-claimed fd is already closed.
                if (conn.fin.load(.acquire) == fin_teardown) {
                    conn.stream.close(self.io);
                }

                self.slots.reset(conn.index);
            }

            // Free the text of the requests that never reached the scheduler.
            var drain: [16]Request = undefined;
            while (true) {
                const n = self.request_q.get(self.io, &drain, 0) catch break;
                if (n == 0) break;
                for (drain[0..n]) |req| {
                    freeRequestText(self.gpa, req);
                }
            }

            self.sched.deinit();
        }

        // -- the scheduler's sink -------------------------------------------

        const Sink = struct {
            engine: *Self,

            /// This never blocks. A missing or closed connection swallows the event. The engine
            /// closed the queue of a missing connection at its conn_closed. A full queue reports
            /// false so the scheduler parks, and flags the draining task to send a wakeup.
            pub fn emit(s: *Sink, conn_id: u32, ev: OutEvent) bool {
                const conn = s.engine.findConn(conn_id) orelse return true;
                const n = conn.out_q.put(s.engine.io, &.{ev}, 0) catch return true;
                if (n == 1) return true;

                conn.parked_hint.store(true, .release);
                return false;
            }

            pub fn disconnect(s: *Sink, conn_id: u32) void {
                const conn = s.engine.findConn(conn_id) orelse return;
                s.engine.closeOutput(conn);
            }
        };

        /// Only the engine thread calls this. The result stays valid until this thread handles the
        /// conn_closed of the connection.
        pub fn findConn(self: *Self, conn_id: u32) ?*Conn {
            const index = self.slots.find(conn_id) orelse return null;
            return &self.conns[index];
        }

        /// The connection id of a slot. A connection task calls this for its own slot, and
        /// `beginClose` calls it at teardown for a slot that still has a task.
        pub fn connId(self: *const Self, conn: *const Conn) u32 {
            return self.slots.idOf(conn.index);
        }

        /// Idempotent: exactly one close reaches the queue. A binary writer then drains what is
        /// buffered, flushes, shuts the socket down, and exits. The HTTP task ends its turn with
        /// `error.ConnClosed`.
        pub fn closeOutput(self: *Self, conn: *Conn) void {
            if (conn.out_closed.swap(true, .acq_rel)) return;
            conn.out_q.close(self.io);
        }

        /// Teardown claims the socket of a connection so it can shut it down to wake a blocked
        /// task. Returns true only if the last task has not claimed it.
        pub fn claimStreamForTeardown(self: *Self, conn: *Conn) bool {
            _ = self;
            return conn.fin.cmpxchgStrong(fin_live, fin_teardown, .acq_rel, .acquire) == null;
        }

        /// The owner of the process submits a `Request` to the engine. The call blocks.
        /// A closed queue means teardown is underway.
        pub fn submit(self: *Self, req: Request) !void {
            std.debug.assert(req.conn == 0);
            try self.request_q.putOne(self.io, req);
        }

        /// The reader task of `conn` submits a `Request`. The call blocks. A closed queue means
        /// teardown is underway.
        pub fn submitFrom(self: *Self, conn: *Conn, req: Request) !void {
            std.debug.assert(req.conn == self.slots.idOf(conn.index));
            std.debug.assert(req.body != .wakeup and req.body != .conn_closed);
            _ = conn.requests_open.fetchAdd(1, .acq_rel);
            self.request_q.putOne(self.io, req) catch |err| {
                _ = conn.requests_open.fetchSub(1, .acq_rel);
                return err;
            };
        }

        /// True when `conn` holds no session and no request in the engine, so a reader can close
        /// it at its idle deadline.
        pub fn connIdle(self: *const Self, conn: *const Conn) bool {
            _ = self;
            inline for (conn_slots.idle_load_order) |counter| {
                const value = switch (counter) {
                    .requests_open => conn.requests_open.load(.acquire),
                    .sessions => conn.sessions.load(.acquire),
                };
                if (value != 0) return false;
            }

            return true;
        }

        /// Wake an idle engine so its next tick retries a parked event. A full request queue
        /// drops the wake, because a busy engine retries parks anyway.
        pub fn wake(self: *Self, conn_id: u32) void {
            log.debug("wake: conn {d}", .{conn_id});
            _ = self.request_q.put(self.io, &.{.{
                .conn = conn_id,
                .request_id = 0,
                .session_id = 0,
                .body = .wakeup,
            }}, 0) catch {};
        }

        /// The reader submits conn_closed after its last request, and a writer closes the queue.
        /// The last task out closes the socket. The acceptor or teardown awaits the futures.
        pub fn taskExit(self: *Self, conn: *Conn, role: TaskRole) void {
            // A writer that stops can send nothing more, so the close loses no reply. It wakes a
            // reader that waits to put into the full queue.
            if (role == .writer) {
                self.closeOutput(conn);
            }

            if (role == .reader) {
                // The engine must hear about the drop exactly once, even when the queue is
                // briefly full, so a cancel must not end the wait. Only a closed queue loses the
                // request, and then teardown frees the slot.
                const id = self.slots.idOf(conn.index);
                self.request_q.putOneUncancelable(self.io, .{
                    .conn = id,
                    .request_id = 0,
                    .session_id = 0,
                    .body = .conn_closed,
                }) catch {};
            }

            if (!self.slots.exitTask(conn.index)) return;

            // Close the socket unless teardown claimed it. Teardown closes it after the await.
            if (conn.fin.cmpxchgStrong(fin_live, fin_task, .acq_rel, .acquire) == null) {
                conn.stream.close(self.io);
            }
        }

        // -- acceptor: the adapter spawns the tasks -------------------------

        fn acceptorTask(self: *Self) void {
            while (true) {
                var stream = acceptNext(&self.listener, self.io) orelse return;
                const claim = self.slots.claim() orelse {
                    // A cancel signals once. If a refusal write took it, the next accept blocks
                    // for good. So the cancel waits for the accept.
                    const protect = self.io.swapCancelProtection(.blocked);
                    defer _ = self.io.swapCancelProtection(protect);

                    Adapter.refuse(self.io, &stream);
                    continue;
                };
                timed_stream.enableKeepalive(stream);
                const conn = &self.conns[claim.index];
                if (claim.reused) {
                    Adapter.awaitConn(self, conn);
                }

                conn.stream = stream;
                conn.out_q = .init(&conn.out_buf);
                conn.out_closed.store(false, .release);
                conn.fin.store(fin_live, .release);
                conn.parked_hint.store(false, .release);
                conn.sessions.store(0, .release);
                conn.requests_open.store(0, .release);
                self.last_conn_id = self.slots.nextId(self.last_conn_id);
                self.slots.open(claim.index, self.last_conn_id, Adapter.conn_tasks);
                if (!Adapter.onAccept(self, conn)) {
                    // onAccept left the stream closed. Reclaim the slot.
                    self.slots.abandon(claim.index);
                }
            }
        }
    };
}

// -- api --------------------------------------------------------------------

/// Map one `accept` failure to an action. The switch names each error, so a new name in std
/// fails the build. On Linux, `Unexpected` holds the pending network errors of accept(2),
/// which a server must retry.
pub fn acceptAction(err: net.Server.AcceptError) AcceptAction {
    return switch (err) {
        error.Canceled, error.SocketNotListening, error.WouldBlock => .stop,
        error.ConnectionAborted, error.ProtocolFailure, error.BlockedByFirewall => .retry_now,
        error.ProcessFdQuotaExceeded,
        error.SystemFdQuotaExceeded,
        error.SystemResources,
        error.NetworkDown,
        error.Unexpected,
        => .back_off,
    };
}

/// Block until a client connects. Null tells the acceptor task to end. The loop has no bound,
/// because the listener lives as long as the engine. Teardown ends it through `error.Canceled`
/// from `accept` or from the wait.
pub fn acceptNext(listener: *net.Server, io: Io) ?net.Stream {
    while (true) {
        return listener.accept(io) catch |err| {
            switch (acceptAction(err)) {
                .stop => {
                    if (err != error.Canceled) {
                        log.err("accept: {t}, the listener stops", .{err});
                    }

                    return null;
                },
                .retry_now => log.debug("accept: {t}, retry", .{err}),
                .back_off => {
                    log.warn("accept: {t}, retry in {d} ms", .{ err, accept_backoff_ms });
                    io.sleep(.fromMilliseconds(accept_backoff_ms), .awake) catch return null;
                },
            }

            continue;
        };
    }
}

/// Free the gpa-owned text a `Request` carries. The reader duped it across the
/// thread hop, and the scheduler frees it on the happy path. Generic over the
/// wire.
pub fn freeRequestText(gpa: Allocator, req: Request) void {
    switch (req.body) {
        .session_open => |m| gpa.free(m.system_prompt),
        .generate => |m| {
            gpa.free(m.text);
            gpa.free(m.plain);
        },
        .session_append => |m| gpa.free(m.bytes),
        else => {},
    }
}
