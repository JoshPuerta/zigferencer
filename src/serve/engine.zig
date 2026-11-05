//! The transport-agnostic serve engine: one engine thread around the scheduler, a bounded request
//! queue in, a bounded out-queue per connection, and the connection-slot lifecycle. It owns all
//! serve-layer state and knows no wire format. A comptime `Adapter` supplies the per-connection
//! task model: how many tasks a connection runs, and how they map wire bytes to and from the
//! `Request` and `OutEvent` values. `serve/server.zig` is the binary adapter, and
//! `serve/openai/http_server.zig` is the HTTP one. Both are thin maps onto this engine.
//!
//!     acceptor --accept--> Adapter.onAccept(conn) --decode--> request_queue
//!                                                                  |
//!     conn[i] task(s) <--encode-- out_queue[i] <--emit-- ENGINE LOOP (run)
//!
//! Bounds. `Config.max_connections` live connections (at most `limits.max_connections`), one
//! `request_queue_capacity` request queue, and one `out_queue_capacity` event queue per
//! connection. The engine never blocks on a client. Emission is non-blocking, and a full output
//! queue parks the event in the cell of the session while the draining task signals a `wakeup`.
//!
//! Socket close. Exactly one actor closes the socket of a connection. The first to claim it owns
//! the close. The claim is `live -> task` when the last task exits, or `live -> teardown` when
//! teardown must shut the fd down to wake a blocked task. The other actor must not touch the fd,
//! because a `shutdown` or `close` after the other `close` is an EBADF.
//!
//! Adapter surface. All methods are duck-typed over the engine and a `*Conn`:
//!   - `pub const ConnExt`            per-connection state the adapter owns
//!   - `onAccept(engine, conn) bool`  set `conn.tasks_live`, spawn the task(s)
//!                                    into `conn.ext`. False (with the stream
//!                                    closed) if spawning failed
//!   - `beginClose(engine, conn)`     unblock the task(s) at teardown. Close
//!                                    the out-queue, and whatever else the task
//!                                    blocks on (a request reader's socket)
//!   - `awaitConn(engine, conn)`      await the task futures in `conn.ext`
//!   - `refuse(io, *Stream)`          reject a connection when no slot is free

const std = @import("std");
const span = @import("../core/span.zig");
const Io = std.Io;
const net = Io.net;
const Allocator = std.mem.Allocator;
const limits = @import("../core/limits.zig");
const event = @import("event.zig");
const scheduler_mod = @import("scheduler.zig");
const Request = event.Request;
const OutEvent = event.OutEvent;

const log = std.log.scoped(.serve);

// -- constants --------------------------------------------------------------

pub const request_queue_capacity: usize = 64;
pub const out_queue_capacity: usize = 256;

/// The wait after an `accept` failure that a free resource ends. A pending client keeps
/// `EMFILE` true, so a retry with no wait spins.
const accept_backoff_ms: i64 = 100;

const conn_free: u8 = 0;
const conn_live: u8 = 1;
const conn_reclaimable: u8 = 2;

// The actor that closes the socket of a connection. See "Socket close" in the header.
const fin_live: u8 = 0;
const fin_task: u8 = 1;
const fin_teardown: u8 = 2;

// -- types ------------------------------------------------------------------

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
        pub const Conn = struct {
            state: std.atomic.Value(u8) = .init(conn_free),
            id: u32 = 0,
            stream: net.Stream = undefined,
            out_q: Io.Queue(OutEvent) = undefined,
            out_buf: [out_queue_capacity]OutEvent = undefined,
            out_closed: std.atomic.Value(bool) = .init(false),
            /// Single-owner guard for closing `stream` (see `fin_*`).
            fin: std.atomic.Value(u8) = .init(fin_live),
            parked_hint: std.atomic.Value(bool) = .init(false),
            tasks_live: std.atomic.Value(u8) = .init(0),
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
        request_q: Io.Queue(Request),
        next_conn_id: u32 = 1, // acceptor-owned, and ids are never reused
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
            const conns = try arena.alloc(Conn, sched_cfg.max_connections);
            for (conns) |*c| {
                c.* = .{};
            }

            const request_buf = try arena.alloc(Request, request_queue_capacity);
            var self: Self = .{
                .gpa = gpa,
                .io = io,
                .sched = try SchedulerT.init(arena, gpa, io, engine, codec, sched_cfg),
                .codec = codec,
                .listener = listener,
                .conns = conns,
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
            // A dropped connection must end its tasks. Closing the out-queue is the engine half of
            // conn_closed, and the scheduler half sweeps the sessions.
            if (req.body == .conn_closed) {
                if (self.findConn(req.conn)) |conn| {
                    self.closeOutput(conn);
                }
            }

            return self.sched.handle(req, sink);
        }

        fn teardown(self: *Self) void {
            // Stop the acceptor. Closing the listening fd does not unblock accept(2), so cancel it.
            log.debug("shutdown: tearing down engine", .{});
            self.request_q.close(self.io);
            if (self.acceptor_future) |*f| {
                f.cancel(self.io);
            }

            // Unblock the tasks. A reclaimable connection already closed its stream.
            self.listener.deinit(self.io);
            for (self.conns) |*conn| {
                if (conn.state.load(.acquire) == conn_live) {
                    Adapter.beginClose(self, conn);
                }
            }

            // Await the tasks, and close the sockets that teardown claimed.
            for (self.conns) |*conn| {
                if (conn.state.load(.acquire) == conn_free) continue;

                Adapter.awaitConn(self, conn);
                std.debug.assert(conn.fin.load(.acquire) != fin_live);
                // A task-claimed fd is already closed.
                if (conn.fin.load(.acquire) == fin_teardown) {
                    conn.stream.close(self.io);
                }

                conn.state.store(conn_free, .release);
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

            /// This never blocks. A missing or closed connection swallows the event. A full queue
            /// reports false so the scheduler parks, and flags the draining task to send a wakeup.
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

        pub fn findConn(self: *Self, conn_id: u32) ?*Conn {
            for (self.conns) |*conn| {
                if (conn.state.load(.acquire) == conn_live and conn.id == conn_id) return conn;
            }

            return null;
        }

        /// Idempotent: exactly one close reaches the queue. The draining task then drains what is
        /// buffered, flushes, shuts the socket down, and exits.
        pub fn closeOutput(self: *Self, conn: *Conn) void {
            if (conn.out_closed.swap(true, .acq_rel)) return;
            conn.out_q.close(self.io);
        }

        /// Teardown claims the socket of a connection so it can shut it down to wake a blocked
        /// task. Returns true only if the last task has not claimed it. See "Socket close".
        pub fn claimStreamForTeardown(self: *Self, conn: *Conn) bool {
            _ = self;
            return conn.fin.cmpxchgStrong(fin_live, fin_teardown, .acq_rel, .acquire) == null;
        }

        /// A connection task submits a `Request` to the engine. The call blocks.
        /// A closed queue means teardown is underway.
        pub fn submit(self: *Self, req: Request) !void {
            try self.request_q.putOne(self.io, req);
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

        /// The last task out of a connection closes the socket and marks the slot reclaimable.
        /// The acceptor or teardown awaits the futures.
        pub fn taskExit(self: *Self, conn: *Conn) void {
            const prev = conn.tasks_live.fetchSub(1, .acq_rel);
            std.debug.assert(prev >= 1); // never exit more tasks than were spawned
            if (prev == 1) {
                // Close the socket unless teardown claimed it. Teardown closes it after the await.
                if (conn.fin.cmpxchgStrong(fin_live, fin_task, .acq_rel, .acquire) == null) {
                    conn.stream.close(self.io);
                }

                conn.state.store(conn_reclaimable, .release);
            }
        }

        // -- acceptor: the adapter spawns the tasks -------------------------

        fn acceptorTask(self: *Self) void {
            while (true) {
                var stream = acceptNext(&self.listener, self.io) orelse return;
                const conn = self.claimConnSlot() orelse {
                    Adapter.refuse(self.io, &stream);
                    continue;
                };
                conn.id = self.next_conn_id;
                self.next_conn_id += 1;
                conn.stream = stream;
                conn.out_q = .init(&conn.out_buf);
                conn.out_closed.store(false, .release);
                conn.fin.store(fin_live, .release);
                conn.parked_hint.store(false, .release);
                conn.state.store(conn_live, .release);
                if (!Adapter.onAccept(self, conn)) {
                    // onAccept left the stream closed. Reclaim the slot.
                    conn.state.store(conn_free, .release);
                }
            }
        }

        fn claimConnSlot(self: *Self) ?*Conn {
            for (self.conns) |*conn| {
                switch (conn.state.load(.acquire)) {
                    conn_free => return conn,
                    conn_reclaimable => {
                        Adapter.awaitConn(self, conn);
                        conn.state.store(conn_free, .release);
                        return conn;
                    },
                    else => {},
                }
            }

            return null;
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
        .generate => |m| gpa.free(m.text),
        .session_append => |m| gpa.free(m.bytes),
        else => {},
    }
}
