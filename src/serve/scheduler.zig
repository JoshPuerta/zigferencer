//! The control unit of the serve layer: a fixed table of protocol sessions over the session pool
//! of the engine, advanced by one pool call per `tick`. All state lives here and in the pool.
//!
//! Comptime. Generic over the session pool and the codec, so the prongs for snapshots and stats
//! compile away on a pool that lacks them.
//!
//! Tick order. One tick runs four steps in order. It expires leases and retries parked emits. It
//! finalizes stopped turns, which emits `done`. It then makes ONE composed engine call. That call
//! takes every admissible decode row (round-robin up to `max_batch`) plus a prefill chunk of the
//! longest-waiting feeding turn. The chunk fills the rows that decode left free, and it can exceed
//! `max_batch`. When `max_step_rows` is above `max_batch`, which the launcher asserts, its floor of
//! one row keeps saturated decode from starving a waiting turn. A long prefill never preempts a
//! decoder. An empty tick returns `.idle`, and the server then blocks.
//!
//! Session phases. Each entry of the session table is one logical session, not a KV slot. Its
//! `SessionPhase` is free, idle, prefill, or decode. An open or a clone moves an entry from free to
//! idle. A turn, an opening, or an append moves it from idle to prefill. A turn leaves prefill for
//! decode at its first token or at a cancel, so the finalize emits `done`. An opening or an append
//! leaves prefill for idle. A turn leaves decode for idle at its `done`. A close, a drop with no
//! lease, an expiry, or a yield frees an entry from any phase.
//!
//! Backpressure. The scheduler never blocks on a client. Every session event goes through a
//! non-blocking sink. On a full queue the event parks in the single `pending_event` cell of the
//! session, and the session leaves the batch until the writer drains. Control replies have no
//! cell. A client that keeps sending requests with a full queue is broken, and the `disconnect`
//! of the sink is the bounded answer.
//!
//! Parked turns. A tick parks a prefill when no KV slot can take it, or when the RAM tier cannot
//! hold a victim. `fed` stays unchanged, so the next tick picks it again, and the decode rows run
//! alone. A decode row parks too when its restore needs an eviction that the budget refuses. Its
//! state stays intact, so a later blob removal admits it. A finished decoder frees a KV slot, and a
//! closed session removes its blob. Only an external event does the rest. So a parked tick reports
//! no work, because a report of work spins the engine loop. `parks` matches the park log lines.
//!
//! Stop rules. Stop discipline replicates `generate.decode`. The scheduler emits EOS and then ends
//! the turn without stepping it. The token budget ends the turn with the last token sampled but
//! uncommitted, and that token carries into the feed of the next turn. A full context ends the
//! turn before the step that does not fit. The stop reason of a turn is written once, so a natural
//! stop keeps its reason and its carry while it waits for `done`.
//!
//! Think block. A turn can open inside a think block. Its first event is the echo: the `<think>`
//! id alone, which counts as no generated token and never enters the KV. A parked echo holds the
//! one event cell, so the turn skips the prefix restore and prefills cold. A think budget forces
//! ids: at the budget, the close tag and its separator replace the sampled ones. Each forced id
//! goes into the pending token of the engine, and a carry holds it. The stop decision reads a
//! forced id like any other. A new turn resets the think state. `done` counts reasoning events.
//!
//! Append. An append feeds kept (user, assistant) pairs into an idle session. It is all or nothing
//! at admission. It samples nothing, emits no token, inserts no prefix, and leaves the think state
//! alone. It ends with one `appended` reply and moves `turns_done` by the pair count.
//!
//! Id sets. Each entry owns two id sets of `set_words` words at its entry index. The presence set
//! holds the ids accepted since the turn began (the penalty counts this completion only, as the
//! OpenAI API defines it), and it clears at each admission. The context set
//! holds every id fed to the session since it opened. The engine reads both through the sampling
//! params, and the scheduler marks them. They live outside `SessionEntry`, so a reset keeps them.
//!
//! Prefix cache. The cache is best-effort. Any failure counts a miss and the cold path proceeds.
//! A `ready` plan finishes the prefill with zero forward rows, and the engine samples the first
//! token from the stored logits. An append neither restores nor inserts a prefix, because its rows
//! follow a live session.
//!
//! Leases. A lease lets a session outlive its connection. The owner of an entry is free, attached
//! to a connection, or detached with a deadline. `keepLease` gives an attached session a token and
//! a time to live, clamped to `Config.resume_ttl_max_ms`. At a drop, a leased entry detaches and
//! every other entry frees. The drop cuts a turn by the cancel rule. An opening or append prefill
//! runs on, and the events of a detached entry go. A maximum of 0 turns resumption off: no entry
//! holds a lease, every drop frees its sessions, and a keep with a positive time to live fails.
//!
//! Lease expiry. Every transition is a guarded write that keeps `checkLeases`. The expiry scan at
//! the tick head and at each admission frees every detached entry at its deadline. A live open with
//! no free entry takes the detached entry with the earliest deadline. Both stop a prefill in flight
//! before the release. A scan reads the clock once.
//!
//! Resume. `resumeLease` attaches the entry of a token to a new connection and rotates the token.
//! It also takes an attached entry after the drop effect, so the newest holder wins. A resume of a
//! busy entry waits for the idle transition, and every request on the session gets `session_busy`
//! until then. A `resumed` reply goes out when the entry is idle, so the position it names is
//! final.
//!
//! Lease table. Tokens are unique. Live session ids are unique, nonzero, and below the counter. A
//! free entry holds no lease and id 0. A detached entry holds a lease. Every lease has a positive
//! time to live at or below the server maximum.
//!
//! Shares. `Config.sessions_per_connection` below `max_sessions` keeps one client from taking every
//! session. Open, clone, and resume check it before they take an entry.
//!
//! Shutdown. A server that several clients share must not stop on the request of one client. With
//! `allow_shutdown` off, a wire request gets `forbidden`. Connection 0 never comes from the wire,
//! because the acceptor numbers from 1. A request on it is the process owner, and it always stops.
//!
//! Card sampling. `hello_ack` carries the sampling that the model card recommends inside and
//! outside the think block. A client picks the entry of the mode of each turn. The launcher reads
//! the model registry, so the engine holds no per-model table. The default is greedy.
//!
//! Pool boundary. `SessionPool` is the session pool of the runtime (`Sessions(RunnerT)`,
//! `Residency`, or a test fake). The scheduler calls only its composed `advance`. `CodecT` is the
//! text-to-token boundary. `sink` is any value with `emit(conn: u32, ev: OutEvent) bool` and
//! `disconnect(conn: u32) void`.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const limits = @import("../core/limits.zig");
const prop = @import("../core/utils/prop.zig");
const span = @import("../core/span.zig");
const protocol = @import("protocol.zig");
const event = @import("event.zig");
const session_mod = @import("../runtime/session.zig");
const prefix_manager = @import("prefix_manager.zig");
const prefill_chunk = @import("prefill_chunk.zig");
const chat = @import("../core/chat.zig");
const sampler = @import("../runtime/sampler.zig");
const Request = event.Request;
const OutEvent = event.OutEvent;

const core_log = @import("../core/utils/log.zig");
const time = @import("../core/utils/time.zig");

const log = std.log.scoped(.serve);
const log_token = std.log.scoped(.token);

// -- constants --------------------------------------------------------------

/// Draws before a keep or a resume gives up on a token that an entry holds. Two equal draws of 128
/// bits do not occur in practice. The bound only makes the retry loop finite, and its end is an
/// `internal` refusal.
const token_draws_max: u32 = 4;

/// Working ticks between two stats lines. A decode tick costs milliseconds, so the period keeps
/// the line a milestone and not a per-token trace.
const stats_period_ticks: u64 = 512;

/// The refusal of a connection at its share of the session table.
const detail_share_full = "session share of this connection is full";

/// The refusal of a row that an engine call failed.
const detail_advance_failed = "engine advance failed";

/// The pin value of a scheduler that reads the real clock.
const clock_unpinned: i64 = std.math.minInt(i64);

/// The id that names no token. A think config without tags holds it.
pub const no_token: u32 = std.math.maxInt(u32);

// -- types ------------------------------------------------------------------

pub const Config = struct {
    max_sessions: u16,
    /// Sessions of one connection, <= max_sessions. `Scheduler.init` resolves 0 to `max_sessions`.
    sessions_per_connection: u16 = 0,
    max_batch: u16,
    max_seq: u32,
    /// Max rows of one forward pass: decode rows (<= max_batch) plus a prefill chunk.
    /// The row scratch of the runner has exactly this size, so it is the hard cap of a tick.
    max_step_rows: u32,
    /// The prompt rows of one pass, from `--prefill-chunk`. A `derived` width fits `max_step_rows`.
    prefill: prefill_chunk.Rule = .spare_rows,
    /// Live client connections, in 1..limits.max_connections. Only the engine reads it.
    max_connections: u32 = 8,
    /// The deadline of one socket read or write of a connection task. 0 sets no deadline. Only
    /// the engine reads it, and it hands the value to the connection tasks.
    io_timeout_ms: u32 = 0,
    /// A wire client can stop the server with a `shutdown` request.
    allow_shutdown: bool = false,
    /// Lifetime must cover the scheduler's. Must be <= limits.max_model_name_bytes.
    model_name: []const u8,
    vocab_size: u32,
    eos_id: u32,
    /// Every prefix-cache option, in the one group `prefix_manager.zig` reads.
    cache: prefix_manager.CacheConfig = .{},
    /// The think block of the model. The default holds no tags, so a turn never opens a block.
    think: Think = .{},
    /// The sampling for a turn inside the think block.
    recommended_think: protocol.Sampling = protocol.Sampling.greedy,
    /// The sampling for a turn outside the think block.
    recommended_plain: protocol.Sampling = protocol.Sampling.greedy,
    /// The longest lease a client can keep, at most `limits.max_resume_ttl_ms`. 0 turns it off.
    resume_ttl_max_ms: u32 = 0,
};

/// The bearer secret of a session lease. A holder can resume the session on
/// any connection.
pub const Token = protocol.Token;

/// The refusals of a keep. The wire maps the first two to one answer,
/// because routing already hides the sessions of other connections.
pub const KeepError = error{
    /// No live session holds this id.
    NoSuchSession,
    /// The session is attached to another connection, or detached.
    OtherOwner,
    /// The server maximum is 0, so the server keeps no lease.
    ResumptionOff,
    /// An entry already holds the token. Only a caller fault can reach it.
    TokenInUse,
};

/// The refusals of a resume. One error covers an unknown, a rotated, and an
/// expired token, so a probe learns nothing from the difference.
pub const ResumeError = error{
    NoSuchLease,
    /// The new connection holds its share of the session table.
    ShareFull,
    /// An entry already holds the new token. Only a caller fault can reach it.
    TokenInUse,
};

/// The outcome of a resume. `ready` is false while the entry finishes an
/// opening or append prefill, or the stop of a cut turn. The resume then
/// waits in the entry until it is idle, so the position it names is final.
pub const Resumed = struct {
    session_id: u32,
    ready: bool,
};

/// The think tags as ids. The scheduler sees ids only, so the launcher encodes
/// the tags and passes the ids here.
pub const Think = struct {
    /// A turn opens inside the block unless its request says otherwise.
    open_default: bool = false,
    /// `<think>`, echoed as the first event of an open turn.
    open_id: u32 = no_token,
    /// `</think>`, which closes the block.
    close_id: u32 = no_token,
    /// Think tokens before the forced close. Zero is unbounded.
    budget: u32 = 0,
    /// The ids forced at the budget: the close, then its separator.
    force: [limits.think_force_ids_max]u32 = @splat(no_token),
    force_len: u8 = 0,

    fn hasTags(self: *const Think) bool {
        return self.open_id != no_token and self.close_id != no_token;
    }
};

pub const Tick = enum { worked, idle };
pub const HandleResult = enum { ok, shutdown };

const SessionPhase = enum { free, idle, prefill, decode };

/// What a prefill feeds. A `turn` samples its first token at the end. An `opening` system prompt
/// and an `append` of kept pairs sample nothing. They return the entry to idle with one reply.
const PrefillKind = enum { turn, opening, append };

/// The session table over `SessionPool` and `CodecT`.
pub fn Scheduler(comptime SessionPool: type, comptime CodecT: type) type {
    return struct {
        const Self = @This();

        /// The pool can snapshot a KV prefix. Without it the prefix-cache paths compile away.
        const can_snapshot = @hasDecl(SessionPool, "can_snapshot") and SessionPool.can_snapshot;

        /// The owner of an entry that is not free. A detached entry holds the
        /// deadline of its lease, and no connection reaches it.
        const Owner = union(enum) {
            attached: u32,
            detached: Io.Clock.Timestamp,
        };

        const Lease = struct {
            token: Token,
            ttl_ms: u32,
        };

        /// The last finished request of an entry. A resume reports it, so a
        /// client can tell a complete reply from a cut one.
        const Summary = struct {
            request_id: u32,
            finish: protocol.FinishReason,
            gen_tokens: u32,
            reasoning_tokens: u32,
        };

        /// A resume that waits for the entry to become idle.
        const ResumeWait = struct {
            conn: u32,
            request_id: u32,
        };

        const SessionEntry = struct {
            state: SessionPhase = .free,
            proto_id: u32 = 0,
            engine_id: u32 = 0,
            owner: Owner = .{ .attached = 0 },
            lease: ?Lease = null,
            last_done: ?Summary = null,
            resume_wait: ?ResumeWait = null,
            has_system_prompt: bool = false,
            turns_done: u32 = 0,
            kind: PrefillKind = .turn,
            /// The pairs of the append in flight, and the position after each.
            append_n: u16 = 0,
            append_pos: [limits.max_append_pairs]u32 = undefined,
            request_id: u32 = 0,
            turn_tokens: []u32 = &.{},
            /// Prompt size of the turn. It outlives the token buffer for the `done` report.
            turn_prompt_len: u32 = 0,
            fed: usize = 0,
            /// The turn started at position zero, so its prefix is a cache insert candidate.
            caches_prefix: bool = false,
            prefill_seq: u64 = 0,
            max_tokens_turn: u32 = 0,
            generated: u32 = 0,
            last_token: u32 = 0,
            carry: ?u32 = null,
            stop: ?protocol.FinishReason = null,
            /// The turn is inside the think block.
            think_open: bool = false,
            /// Tokens accepted inside the block, the close excluded.
            think_n: u32 = 0,
            /// Think tokens before the forced close, from the request or the server.
            think_budget: u32 = 0,
            /// The forced ids still left are `forced[forced_head..forced_len]`.
            /// A turn loads the queue once, so it needs no wraparound.
            forced: [limits.think_force_ids_max]u32 = @splat(no_token),
            forced_len: u8 = 0,
            forced_head: u8 = 0,
            pending_event: ?OutEvent = null,
            t_turn_start: Io.Clock.Timestamp = undefined,
            t_prefill_end: Io.Clock.Timestamp = undefined,
        };

        /// The decode rows of one composed step, in admission order.
        const DecodeRows = struct {
            ids: [limits.max_batch]u32 = undefined,
            members: [limits.max_batch]*SessionEntry = undefined,
            out: [limits.max_batch]u32 = undefined,
            n: usize = 0,

            fn engineIds(self: *const DecodeRows) []const u32 {
                return self.ids[0..self.n];
            }

            fn entries(self: *const DecodeRows) []const *SessionEntry {
                return self.members[0..self.n];
            }

            fn tokens(self: *const DecodeRows) []const u32 {
                return self.out[0..self.n];
            }

            fn tokensOut(self: *DecodeRows) []u32 {
                return self.out[0..self.n];
            }
        };

        /// One cache entry that a restore plan offers.
        const PrefixHit = @FieldType(prefix_manager.Restore, "ready");

        gpa: Allocator,
        io: Io,
        pool: *SessionPool,
        codec: *const CodecT,
        cfg: Config,
        entries: []SessionEntry,
        /// One id set per entry.
        presence: []u64,
        context: []u64,
        set_words: usize,
        /// Session ids only grow. A resume keeps the id, and a yield or an expiry never returns it.
        next_proto_id: u32 = 1,
        next_prefill_seq: u64 = 1,
        /// Detached entries freed at their deadline.
        lease_expires: u64 = 0,
        /// Detached entries freed for a live open.
        lease_yields: u64 = 0,
        /// Yields before an open or a clone that the engine then refused.
        lease_yields_lost: u64 = 0,
        /// Prefills that a yield or an expiry cut.
        lease_prefills_cut: u64 = 0,
        /// Every lease clock read returns this point in nanoseconds, unless it is `clock_unpinned`.
        /// Tests pin it, and it is atomic because a test thread moves it under the engine thread.
        clock_pin_ns: std.atomic.Value(i64) = .init(clock_unpinned),
        rr_cursor: usize = 0,
        tokens_generated_total: u64 = 0,
        /// Park log lines, one to one. Only the periodic stats line reads it.
        parks: u64 = 0,
        /// Ticks that produced engine work. Only the periodic stats line reads it.
        ticks_worked: u64 = 0,
        /// The prefix cache and its counters. The scheduler owns the tokens that the cache keys on.
        prefix: prefix_manager.PrefixManager,
        t_init: Io.Clock.Timestamp,

        pub fn init(
            arena: Allocator,
            gpa: Allocator,
            io: Io,
            pool: *SessionPool,
            codec: *const CodecT,
            cfg_in: Config,
        ) Allocator.Error!Self {
            // Resolve the share, so the scheduler body never reads the sentinel.
            var cfg = cfg_in;
            if (cfg.sessions_per_connection == 0) {
                cfg.sessions_per_connection = cfg.max_sessions;
            }

            // Check the config.
            std.debug.assert(cfg.max_sessions >= 1);
            std.debug.assert(cfg.sessions_per_connection >= 1);
            std.debug.assert(cfg.sessions_per_connection <= cfg.max_sessions);
            std.debug.assert(cfg.max_batch >= 1);
            std.debug.assert(cfg.max_batch <= limits.max_batch);
            std.debug.assert(cfg.model_name.len <= limits.max_model_name_bytes);
            std.debug.assert(cfg.resume_ttl_max_ms <= limits.max_resume_ttl_ms);

            // The CLI patches the engine-derived fields per backend.
            // These two catch an unpatched base.
            std.debug.assert(cfg.max_seq >= 1);
            std.debug.assert(cfg.max_step_rows >= 1);

            // The decode rows of one pass must fit the row cap.
            std.debug.assert(cfg.max_batch <= cfg.max_step_rows);
            switch (cfg.prefill) {
                .derived => |width| std.debug.assert(width >= 1 and width <= cfg.max_step_rows),
                .fixed => |n| std.debug.assert(n >= 1),
                .spare_rows => {},
            }

            assertThinkConfig(cfg.think, cfg.eos_id, cfg.vocab_size);

            // Allocate the session table and its id sets.
            const entries = try arena.alloc(SessionEntry, cfg.max_sessions);
            for (entries) |*s| {
                s.* = .{};
            }

            const words = sampler.setWords(cfg.vocab_size);
            const presence = try arena.alloc(u64, words * cfg.max_sessions);
            @memset(presence, 0);
            const context = try arena.alloc(u64, words * cfg.max_sessions);
            @memset(context, 0);

            return .{
                .gpa = gpa,
                .io = io,
                .pool = pool,
                .codec = codec,
                .cfg = cfg,
                .entries = entries,
                .presence = presence,
                .context = context,
                .set_words = words,
                .prefix = prefix_manager.PrefixManager.init(gpa, io, cfg.cache, can_snapshot),
                .t_init = Io.Clock.Timestamp.now(io, .awake),
            };
        }

        /// Free the prefix cache. Entries are arena-owned and turn tokens are
        /// freed per turn, so the cache is the only gpa state left.
        pub fn deinit(self: *Self) void {
            self.prefix.deinit();
        }

        /// Warm the prefix cache at boot. The manager drives the warm-up
        /// through this same scheduler, so the entries tokenize exactly as
        /// live traffic does.
        pub fn preloadPrefixes(self: *Self) void {
            self.prefix.preload(self, self.cfg.cache.preload, self.cfg.max_seq);
        }

        /// Consume one request. Frees any request-owned text exactly once.
        pub fn handle(self: *Self, req: Request, sink: anytype) HandleResult {
            defer self.checkLeases();

            switch (req.body) {
                .hello => self.reply(sink, req, self.helloAck(req)),
                .ping => self.reply(
                    sink,
                    req,
                    .{ .request_id = req.request_id, .session_id = 0, .body = .pong },
                ),
                .stats => self.reply(sink, req, self.statsAck(req)),
                .session_open => |m| {
                    defer self.gpa.free(m.system_prompt);

                    if (self.refuseFrequency(req, m.sampling, sink)) return .ok;
                    self.handleOpen(req, m.sampling, m.system_prompt, sink);
                },
                .session_clone => |m| self.handleClone(req, m.sampling, sink),
                .generate => |m| {
                    defer self.gpa.free(m.text);
                    defer self.gpa.free(m.plain);

                    self.handleGenerate(req, m, sink);
                },
                .session_keep => |m| self.handleKeep(req, m.ttl_ms, sink),
                .session_resume => |m| self.handleResume(req, m.token, sink),
                .session_append => |m| {
                    defer self.gpa.free(m.bytes);

                    self.handleAppend(req, m.pairs(), sink);
                },
                .cancel => self.handleCancel(req, sink),
                .session_close => self.handleClose(req, sink),
                .conn_closed => self.sweepConn(req.conn),
                .wakeup => {}, // the tick that follows does the actual retry

                .shutdown => return self.handleShutdown(req, sink),
            }

            return .ok;
        }

        /// One unit of engine work. `.idle` when nothing is runnable. A parked event waits on the
        /// writer, so it is not runnable. A stopped turn that awaits its `done` is runnable.
        pub fn tick(self: *Self, sink: anytype) Tick {
            // An idle tick records no span, so a polling engine fills no ring.
            const sp = span.begin(.sched_tick);
            const t = self.tickWork(sink);
            if (t != .idle) {
                sp.end();
            }

            return t;
        }

        fn tickWork(self: *Self, sink: anytype) Tick {
            if (self.leaseNow()) |now| {
                self.expireDetached(now);
            }

            self.retryParked(sink);
            self.finalizeStops(sink);
            defer self.checkLeases();

            if (self.runComposedStep(sink)) return self.worked();

            for (self.entries) |*entry| {
                if (entry.state == .decode and entry.stop != null and entry.pending_event == null) {
                    return self.worked();
                }
            }

            return .idle;
        }

        /// Count a working tick, and emit the stats line every `stats_period_ticks`.
        fn worked(self: *Self) Tick {
            self.ticks_worked += 1;
            if (self.ticks_worked % stats_period_ticks == 0) {
                self.logStats();
            }

            return .worked;
        }

        /// One line of KV-residency pressure. Only the tiered engine keeps
        /// these counters, so a plain session pool logs nothing here.
        fn logStats(self: *const Self) void {
            if (comptime @hasDecl(SessionPool, "stats")) {
                const s = self.pool.stats();
                const mib = 1024 * 1024;

                // Only a bounded tier has a ceiling to print. A zero budget
                // prints as zero MiB, never as one of the other two tiers.
                var budget_buf: [32]u8 = undefined;
                const budget_txt: []const u8 = switch (s.blob_budget) {
                    .absent => "absent",
                    .unenforced => "unenforced",
                    .bounded => |bytes| std.fmt.bufPrint(
                        &budget_buf,
                        "{d} MiB",
                        .{bytes / mib},
                    ) catch "?",
                };

                // The disk split prints only when the tier exists. A zero on a
                // RAM-only config reads as a cold disk tier.
                var disk_buf: [64]u8 = undefined;
                const disk_txt: []const u8 = if (self.prefix.diskEnabled())
                    std.fmt.bufPrint(&disk_buf, " (disk {d} hit / {d} ins)", .{
                        self.prefix.disk_hits,
                        self.prefix.disk_inserts,
                    }) catch "?"
                else
                    "";

                // Emit the line, then the per-entry and per-blob detail.
                log.info(
                    "stats: resident {d}/{d} slots, {d} logical, ram {d} MiB of {s}, evicts " ++
                        "{d} ({d:.0} ms avg), restores {d} ({d:.0} ms avg), parks {d}, " ++
                        "prefix {d} hit / {d} miss / {d} ins{s}, leases {d} expired / {d} " ++
                        "yielded / {d} lost / {d} prefills cut",
                    .{
                        s.slots_resident,
                        s.n_physical,
                        s.n_logical,
                        s.ram_used / mib,
                        budget_txt,
                        s.evicts,
                        avgMs(s.evict_ms, s.evicts),
                        s.restores,
                        avgMs(s.restore_ms, s.restores),
                        self.parks,
                        self.prefix.hits,
                        self.prefix.misses,
                        self.prefix.inserts,
                        disk_txt,
                        self.lease_expires,
                        self.lease_yields,
                        self.lease_yields_lost,
                        self.lease_prefills_cut,
                    },
                );
                self.prefix.logEntryHits();
                if (comptime @hasDecl(SessionPool, "logBlobs")) {
                    self.pool.logBlobs();
                }
            }
        }

        // -- request handlers -----------------------------------------------

        /// Stop the server: cancel every turn, log the stats, and acknowledge.
        fn handleShutdown(self: *Self, req: Request, sink: anytype) HandleResult {
            // Connection 0 is the process owner.
            if (!self.cfg.allow_shutdown and req.conn != 0) {
                self.refuseNoSession(req, sink, .forbidden, "shutdown not allowed");
                return .ok;
            }

            self.cancelAll(sink);
            self.logStats();
            self.reply(
                sink,
                req,
                .{ .request_id = req.request_id, .session_id = 0, .body = .shutdown_ack },
            );
            return .shutdown;
        }

        /// Refuse `req` with an error event on its session.
        fn refuse(
            self: *Self,
            req: Request,
            sink: anytype,
            code: protocol.ErrorCode,
            detail: []const u8,
        ) void {
            self.reply(sink, req, OutEvent.fail(req.request_id, req.session_id, code, detail));
        }

        /// Refuse `req` with an error event that names no session.
        fn refuseNoSession(
            self: *Self,
            req: Request,
            sink: anytype,
            code: protocol.ErrorCode,
            detail: []const u8,
        ) void {
            self.reply(sink, req, OutEvent.fail(req.request_id, 0, code, detail));
        }

        /// The entry of the session that `req` names. A refusal goes out when none matches.
        fn sessionOf(self: *Self, req: Request, sink: anytype) ?*SessionEntry {
            return self.entryByProto(req.session_id, req.conn) orelse {
                self.refuse(req, sink, .unknown_session, "no such session");
                return null;
            };
        }

        /// The entry of a session between turns, with its parked event sent. A refusal or a
        /// disconnect ends the request when the session is not ready for a new one.
        fn idleSession(self: *Self, req: Request, sink: anytype) ?*SessionEntry {
            const entry = self.sessionOf(req, sink) orelse return null;
            if (entry.resume_wait != null) {
                self.refuseWaiting(req, sink);
                return null;
            }

            if (!self.flushParked(entry, sink)) return null;

            std.debug.assert(entry.pending_event == null);
            if (entry.state != .idle) {
                self.refuse(req, sink, .session_busy, "turn in flight");
                return null;
            }

            return entry;
        }

        /// An entry for a new session, within the share of the connection, or a refusal.
        fn admitEntry(self: *Self, req: Request, sink: anytype) ?Taken {
            if (self.leaseNow()) |now| {
                self.expireDetached(now);
            }

            if (self.shareFull(req.conn)) {
                self.refuseNoSession(req, sink, .sessions_exhausted, detail_share_full);
                return null;
            }

            return self.freeEntry() orelse {
                self.refuseNoSession(req, sink, .sessions_exhausted, "no free session");
                return null;
            };
        }

        /// The server has no frequency penalty yet. A request that asks for
        /// one fails whole and changes nothing, so the value is never dropped.
        /// Returns true when it refused the request.
        fn refuseFrequency(
            self: *Self,
            req: Request,
            sampling: protocol.Sampling,
            sink: anytype,
        ) bool {
            std.debug.assert(@abs(sampling.frequency_penalty) <= protocol.frequency_penalty_max);

            if (sampling.frequency_penalty == 0) return false;

            self.refuse(req, sink, .forbidden, protocol.frequency_unsupported_detail);
            return true;
        }

        fn handleOpen(
            self: *Self,
            req: Request,
            sampling: protocol.Sampling,
            system_prompt: []const u8,
            sink: anytype,
        ) void {
            // Bind an entry to a new engine session.
            const taken = self.admitEntry(req, sink) orelse return;
            const entry = taken.entry;
            const engine_id = self.pool.create(
                sampling.seed,
                self.samplerParams(entry, sampling),
            ) catch {
                self.yieldLost(taken, "open");
                self.refuseNoSession(req, sink, .sessions_exhausted, "no free session");
                return;
            };
            entry.* = .{
                .state = .idle,
                .proto_id = self.next_proto_id,
                .engine_id = engine_id,
                .owner = .{ .attached = req.conn },
                .request_id = req.request_id,
            };
            @memset(self.presenceOf(entry), 0);
            @memset(self.contextOf(entry), 0);
            self.yieldDone(taken, entry.proto_id);
            log.info(
                "admit: session {d} engine {d} conn {d}",
                .{ entry.proto_id, engine_id, req.conn },
            );
            self.next_proto_id += 1;

            // An empty system prompt ends the open here.
            if (system_prompt.len == 0) return self.emitOpened(entry, sink);

            // Tokenize the system prompt, and refuse one past the window.
            // A system block has no think tail, so the switch changes no id.
            const tokens = self.codec.turnTokens(
                self.gpa,
                .system,
                system_prompt,
                &.{},
                self.cfg.think.open_default,
            ) catch {
                self.dropEntry(entry);
                self.refuseNoSession(req, sink, .internal, "tokenize failed");
                return;
            };
            const window = self.pool.capacityRemaining(engine_id);
            if (tokens.len > window) {
                self.gpa.free(tokens);
                self.dropEntry(entry);
                var detail: [detail_max]u8 = undefined;
                const msg = systemTooLongDetail(&detail, tokens.len, window);
                self.refuseNoSession(req, sink, .turn_too_long, msg);
                return;
            }

            entry.has_system_prompt = true;
            self.startPrefill(entry, req.request_id, tokens, .opening, sink);
        }

        /// Fork an open session: shared context, new seed and sampling, new protocol id. The source
        /// must be between turns, because a client cannot reason about a half-written fork.
        fn handleClone(self: *Self, req: Request, sampling: protocol.Sampling, sink: anytype) void {
            if (comptime !SessionPool.can_clone) {
                return self.refuse(req, sink, .internal, "clone unsupported on this backend");
            }

            const source = self.idleSession(req, sink) orelse return;
            if (self.refuseFrequency(req, sampling, sink)) return;

            // Fork the engine session into an entry.
            const taken = self.admitEntry(req, sink) orelse return;
            const entry = taken.entry;
            const engine_id = self.pool.clone(
                source.engine_id,
                sampling.seed,
                self.samplerParams(entry, sampling),
            ) catch |err| {
                // A fork needs both sides VRAM-resident, so tiered KV can refuse. Name the
                // reason: a client that reads "no free session" retries the wrong thing.
                const detail: []const u8 = if (err == error.RamBudgetExceeded)
                    "kv ram budget refused the clone"
                else if (err == error.NoEvictableSlot)
                    "no evictable slot for the clone"
                else
                    "no free session";

                self.yieldLost(taken, "clone");
                self.refuseNoSession(req, sink, .sessions_exhausted, detail);
                return;
            };

            // Fill the entry from the source.
            entry.* = .{
                .state = .idle,
                .proto_id = self.next_proto_id,
                .engine_id = engine_id,
                .owner = .{ .attached = req.conn },
                .has_system_prompt = source.has_system_prompt,
                .turns_done = source.turns_done,
                .carry = source.carry,
                .request_id = req.request_id,
            };
            // A clone shares the context of its source, but its presence set starts empty.
            @memset(self.presenceOf(entry), 0);
            @memcpy(self.contextOf(entry), self.contextOf(source));
            self.yieldDone(taken, entry.proto_id);
            log.info(
                "clone: session {d} from {d} engine {d}",
                .{ entry.proto_id, source.proto_id, engine_id },
            );
            self.next_proto_id += 1;
            self.emitOpened(entry, sink);
        }

        fn handleGenerate(
            self: *Self,
            req: Request,
            m: @FieldType(Request.Body, "generate"),
            sink: anytype,
        ) void {
            const entry = self.idleSession(req, sink) orelse return;

            if (m.sampling) |sampling| {
                if (self.refuseFrequency(req, sampling, sink)) return;
            }

            // The switch resolves once. The render and the entry read this one
            // value, so the tail of the wrap and the think state agree.
            const open = m.think_open orelse self.cfg.think.open_default;

            // Request data must never reach an assert, so a model with no tags refuses a block
            // or a budget. A closed block is its only mode, so that switch passes.
            if ((open or m.think_budget != null) and !self.cfg.think.hasTags()) {
                return self.refuse(req, sink, .forbidden, "this model has no think tags");
            }

            // A budget needs a forced close.
            const budget = m.think_budget orelse self.cfg.think.budget;
            std.debug.assert(budget <= limits.max_model_ctx);
            if (budget > 0 and self.cfg.think.force_len == 0) {
                return self.refuse(req, sink, .forbidden, "this model has no forced close");
            }

            if (m.reseed) std.debug.assert(m.sampling != null);

            // Encode the turn, with any carried token first.
            const tokens = self.encodeTurn(req, entry, m.text, m.plain, open, sink) orelse return;

            // Every refusal is behind, so a refused turn leaves the sampler
            // of the session unchanged.
            if (m.sampling) |sampling| {
                self.pool.setParams(entry.engine_id, self.samplerParams(entry, sampling));
                // Only an explicit reseed restarts the stream, so a client
                // that resends its sampling keeps one stream per session.
                if (m.reseed) {
                    self.pool.reseed(entry.engine_id, sampling.seed);
                }
            }

            // Reset the turn state.
            entry.max_tokens_turn = m.max_tokens;
            entry.generated = 0;
            entry.think_open = open;
            entry.think_budget = budget;
            entry.think_n = 0;
            entry.forced_len = 0;
            entry.forced_head = 0;
            entry.t_turn_start = Io.Clock.Timestamp.now(self.io, .awake);
            @memset(self.presenceOf(entry), 0);

            // The echo goes out before the prefill.
            if (open) {
                self.emitEcho(entry, req.request_id, sink);
            }

            std.debug.assert(entry.think_open == open);
            if (entry.think_budget > 0) std.debug.assert(self.cfg.think.hasTags());

            self.startPrefill(entry, req.request_id, tokens, .turn, sink);
        }

        /// The tokens of a turn, with any carried token first, or a refusal. A turn that does not
        /// fit keeps the carry, because the carried token stays pending for the next turn.
        fn encodeTurn(
            self: *Self,
            req: Request,
            entry: *SessionEntry,
            text: []const u8,
            plain: []const chat.PlainRange,
            open: bool,
            sink: anytype,
        ) ?[]u32 {
            const turn = turnKind(entry.has_system_prompt, entry.turns_done);
            const encoded = self.codec.turnTokens(self.gpa, turn, text, plain, open) catch {
                self.refuse(req, sink, .internal, "tokenize failed");
                return null;
            };
            const carry_before = entry.carry;
            const tokens = self.withCarry(entry, encoded) catch {
                self.gpa.free(encoded);
                self.refuse(req, sink, .internal, "oom");
                return null;
            };

            const room = self.pool.capacityRemaining(entry.engine_id);
            if (tokens.len <= room) return tokens;

            self.gpa.free(tokens);
            entry.carry = carry_before;
            var detail: [detail_max]u8 = undefined;
            const msg = turnTooLongDetail(&detail, tokens.len, room);
            self.refuse(req, sink, .turn_too_long, msg);
            return null;
        }

        /// Feed kept pairs into an idle session, with the guards of `handleGenerate`. A set that
        /// does not fit refuses whole. A carried token of a cut turn joins the first pair.
        fn handleAppend(
            self: *Self,
            req: Request,
            pairs: protocol.AppendPairs,
            sink: anytype,
        ) void {
            std.debug.assert(pairs.n >= 1);
            std.debug.assert(pairs.n <= limits.max_append_pairs);

            const entry = self.idleSession(req, sink) orelse return;

            // Render every pair before any row moves.
            var ends: [limits.max_append_pairs]u32 = undefined;
            var ids = self.renderPairs(req, entry, pairs, &ends, sink) orelse return;

            // Refuse a set past the room.
            const room = self.pool.capacityRemaining(entry.engine_id);
            if (ids.items.len > room) {
                var detail: [detail_max]u8 = undefined;
                const msg = turnTooLongDetail(&detail, ids.items.len, room);
                return self.appendFailed(req, &ids, sink, .turn_too_long, msg);
            }

            // Record the position after each pair.
            const pos0 = self.pool.posOf(entry.engine_id);
            for (ends[0..pairs.n], entry.append_pos[0..pairs.n]) |end, *pos| {
                pos.* = pos0 + end;
            }

            entry.append_n = pairs.n;
            log.info(
                "append: session {d}, {d} pairs, {d} tokens, pos {d} -> {d}",
                .{ entry.proto_id, pairs.n, ids.items.len, pos0, entry.append_pos[pairs.n - 1] },
            );

            // Start the prefill.
            const tokens = ids.toOwnedSlice(self.gpa) catch {
                return self.appendFailed(req, &ids, sink, .internal, "oom");
            };
            if (tokens.len == 0) {
                // Empty pairs of a template-free family render to no token.
                self.gpa.free(tokens);
                entry.request_id = req.request_id;
                entry.kind = .append;
                self.emitAppended(entry, sink);
                self.endPrefill(entry, .idle, sink);
                return;
            }

            self.startPrefill(entry, req.request_id, tokens, .append, sink);
        }

        /// The ids of `pairs` after any carried token, or a refusal. `ends` gets the id count after
        /// each pair.
        fn renderPairs(
            self: *Self,
            req: Request,
            entry: *const SessionEntry,
            pairs: protocol.AppendPairs,
            ends: *[limits.max_append_pairs]u32,
            sink: anytype,
        ) ?std.ArrayList(u32) {
            var ids: std.ArrayList(u32) = .empty;
            if (entry.carry) |c| {
                ids.append(self.gpa, c) catch {
                    self.appendFailed(req, &ids, sink, .internal, "oom");
                    return null;
                };
            }

            var it = pairs.iterator();
            var i: u16 = 0;
            // One pass per pair. The decoder bounds the count.
            while (it.next()) |pair| : (i += 1) {
                const order = turnOrder(entry.has_system_prompt, entry.turns_done + i);
                const toks = self.codec.pairTokens(
                    self.gpa,
                    order,
                    pair.user,
                    pair.assistant,
                ) catch |err| {
                    const detail: []const u8 = if (err == error.AppendUnsupported)
                        "append unsupported"
                    else
                        "tokenize failed";

                    self.appendFailed(req, &ids, sink, .internal, detail);
                    return null;
                };
                defer self.gpa.free(toks);

                ids.appendSlice(self.gpa, toks) catch {
                    self.appendFailed(req, &ids, sink, .internal, "oom");
                    return null;
                };
                ends[i] = @intCast(ids.items.len);
            }

            std.debug.assert(i == pairs.n);
            return ids;
        }

        /// A refused append feeds no row and keeps the carry.
        fn appendFailed(
            self: *Self,
            req: Request,
            ids: *std.ArrayList(u32),
            sink: anytype,
            code: protocol.ErrorCode,
            detail: []const u8,
        ) void {
            ids.deinit(self.gpa);
            self.refuse(req, sink, code, detail);
        }

        /// The reply of a finished open or clone. It names the position of the session.
        fn emitOpened(self: *Self, entry: *SessionEntry, sink: anytype) void {
            self.emitOrPark(entry, sink, .{
                .request_id = entry.request_id,
                .session_id = entry.proto_id,
                .body = .{ .session_opened = .{ .kv_pos = self.pool.posOf(entry.engine_id) } },
            });
        }

        fn emitAppended(self: *Self, entry: *SessionEntry, sink: anytype) void {
            std.debug.assert(entry.kind == .append);

            var body: protocol.Appended = .{ .n = entry.append_n };
            @memcpy(body.kv_pos[0..entry.append_n], entry.append_pos[0..entry.append_n]);
            std.debug.assert(body.kv_pos[entry.append_n - 1] == self.pool.posOf(entry.engine_id));
            self.emitOrPark(entry, sink, .{
                .request_id = entry.request_id,
                .session_id = entry.proto_id,
                .body = .{ .appended = body },
            });
        }

        /// The sampling params of `entry`, which read the two id sets of the entry.
        fn samplerParams(self: *Self, entry: *SessionEntry, s: protocol.Sampling) sampler.Params {
            return .{
                .temperature = s.temperature,
                .top_k = s.top_k,
                .top_p = s.top_p,
                .min_p = s.min_p,
                .presence_penalty = s.presence_penalty,
                .presence = self.presenceOf(entry),
                .repetition_penalty = s.repetition_penalty,
                .context = self.contextOf(entry),
            };
        }

        fn presenceOf(self: *Self, entry: *const SessionEntry) []u64 {
            return self.setOf(self.presence, entry);
        }

        fn contextOf(self: *Self, entry: *const SessionEntry) []u64 {
            return self.setOf(self.context, entry);
        }

        fn setOf(self: *Self, sets: []u64, entry: *const SessionEntry) []u64 {
            const at = (@intFromPtr(entry) - @intFromPtr(self.entries.ptr)) / @sizeOf(SessionEntry);
            std.debug.assert(at < self.entries.len);
            std.debug.assert(sets.len == self.entries.len * self.set_words);
            return sets[at * self.set_words ..][0..self.set_words];
        }

        /// Mark ids that enter the KV of `entry` in its context set.
        fn markFed(self: *Self, entry: *const SessionEntry, ids: []const u32) void {
            sampler.markIds(self.contextOf(entry), self.cfg.vocab_size, ids);
        }

        /// The first event of an open turn: the `<think>` id alone, before the prefill. The
        /// admission check left the cell empty, so this one event can park but never collide.
        fn emitEcho(self: *Self, entry: *SessionEntry, request_id: u32, sink: anytype) void {
            std.debug.assert(entry.state == .idle);
            std.debug.assert(entry.pending_event == null);
            std.debug.assert(entry.think_open);

            const id = self.cfg.think.open_id;
            log.debug("echo: session {d} opens in the think block", .{entry.proto_id});
            self.emitOrPark(
                entry,
                sink,
                OutEvent.token(request_id, entry.proto_id, id, .open_tag, self.codec.piece(id)),
            );
        }

        fn handleCancel(self: *Self, req: Request, sink: anytype) void {
            const entry = self.sessionOf(req, sink) orelse return;
            if (entry.resume_wait != null) return self.refuseWaiting(req, sink);
            switch (entry.state) {
                // An opening has no turn to cut, and its open must end with `session_opened`.
                // So the opening prefill runs on, as it does at a drop.
                .prefill => switch (entry.kind) {
                    .turn => self.cancelTurn(entry),
                    .append => self.cancelAppend(entry, sink),
                    .opening => {},
                },
                .decode => self.cancelTurn(entry),
                else => {}, // an idle cancel has nothing in flight, so it is a no-op
            }
        }

        /// The cancel rule of a turn. A cancel, a drop, and a shutdown share it.
        fn cancelTurn(self: *Self, entry: *SessionEntry) void {
            std.debug.assert(entry.state == .prefill or entry.state == .decode);

            if (entry.state == .prefill) {
                // The partial turn stays in context. A client that cares closes the session.
                std.debug.assert(entry.kind == .turn);

                log.info("stop: session {d} cancelled (prefill state)", .{entry.proto_id});
                self.gpa.free(entry.turn_tokens);
                entry.turn_tokens = &.{};
                entry.t_prefill_end = Io.Clock.Timestamp.now(self.io, .awake);
                std.debug.assert(entry.stop == null);
                entry.stop = .cancelled;
                entry.state = .decode; // finalizeStops emits the done
                return;
            }

            // A natural stop keeps its reason.
            if (entry.stop == null) {
                log.info("stop: session {d} cancelled (decode state)", .{entry.proto_id});
                entry.stop = .cancelled;
            }
        }

        /// A cancelled append frees the unfed tokens and returns the entry to idle. The fed rows
        /// stay, but `turns_done` does not move. The next wrap is then wrong, so the `err` tells
        /// the client to close.
        fn cancelAppend(self: *Self, entry: *SessionEntry, sink: anytype) void {
            std.debug.assert(entry.state == .prefill);
            std.debug.assert(entry.kind == .append);

            log.info(
                "stop: session {d} append cancelled after {d} of {d} tokens",
                .{ entry.proto_id, entry.fed, entry.turn_tokens.len },
            );
            self.gpa.free(entry.turn_tokens);
            entry.turn_tokens = &.{};
            entry.state = .idle;
            entry.kind = .turn;
            self.emitOrPark(
                entry,
                sink,
                OutEvent.fail(entry.request_id, entry.proto_id, .cancelled, "append cancelled"),
            );
            self.deliverResume(entry, sink);
        }

        fn handleClose(self: *Self, req: Request, sink: anytype) void {
            const entry = self.sessionOf(req, sink) orelse return;
            // The client has no `resumed` reply yet, so it cannot know the
            // session it closes. It closes it after the reply.
            if (entry.resume_wait != null) return self.refuseWaiting(req, sink);
            // A failed flush disconnects, and the sweep of that connection
            // releases the entry.
            if (!self.flushParked(entry, sink)) return;

            self.reply(sink, req, .{
                .request_id = req.request_id,
                .session_id = entry.proto_id,
                .body = .session_closed,
            });
            self.releaseEntry(entry);
        }

        fn sweepConn(self: *Self, conn: u32) void {
            log.debug("sweep: conn {d}", .{conn});
            self.detachConn(conn, self.readClock());
        }

        /// The wire form of `keepLease`. A keep on a waiting resume is busy,
        /// because the client has no `resumed` reply yet. The handler draws
        /// a new token for a collision, and it never sends that refusal.
        fn handleKeep(self: *Self, req: Request, ttl_ms: u32, sink: anytype) void {
            const entry = self.sessionOf(req, sink) orelse return;
            if (entry.resume_wait != null) return self.refuseWaiting(req, sink);
            // A failed flush disconnects, and the sweep of that connection
            // handles the entry.
            if (!self.flushParked(entry, sink)) return;

            // Keep with a new token, and retry a draw that an entry already holds.
            var draws: u32 = 0;
            while (draws < token_draws_max) : (draws += 1) {
                // A keep that drops the lease or that resumption refuses
                // reads no token, so it draws none.
                const token: Token = if (ttl_ms == 0 or self.cfg.resume_ttl_max_ms == 0)
                    @splat(0)
                else
                    drawToken(self.io) catch {
                        return self.refuse(req, sink, .internal, "no secure random source");
                    };

                const id = entry.proto_id;
                const granted = self.keepLease(id, req.conn, ttl_ms, token) catch |err| {
                    const code: protocol.ErrorCode, const detail: []const u8 = switch (err) {
                        error.TokenInUse => continue,
                        error.ResumptionOff => .{ .forbidden, "resumption is off" },
                        error.NoSuchSession, error.OtherOwner => .{
                            .unknown_session,
                            "no such session",
                        },
                    };
                    return self.refuse(req, sink, code, detail);
                };
                const sent: Token = if (granted > 0) token else @splat(0);
                if (granted > 0) std.debug.assert(std.mem.eql(u8, &entry.lease.?.token, &token));

                self.reply(
                    sink,
                    req,
                    .{ .request_id = req.request_id, .session_id = entry.proto_id, .body = .{
                        .kept = .{ .granted_ms = granted, .token = sent },
                    } },
                );
                return;
            }

            self.refuse(req, sink, .internal, "token draws collided");
        }

        /// The wire form of `resumeLease`. One answer covers an unknown, a rotated, and an expired
        /// token. The guards run in the order of `resumeLease`. Only a match draws a token, so a
        /// probe costs the source nothing.
        fn handleResume(self: *Self, req: Request, token: Token, sink: anytype) void {
            // With resumption off no entry holds a lease, so no draw is due.
            if (self.cfg.resume_ttl_max_ms == 0) {
                return self.refuse(req, sink, .unknown_session, "no such lease");
            }

            self.expireDetached(self.readClock());
            if (self.shareFull(req.conn)) {
                return self.refuse(req, sink, .sessions_exhausted, detail_share_full);
            }

            if (self.entryByToken(token) == null) {
                return self.refuse(req, sink, .unknown_session, "no such lease");
            }

            // Rotate the token, and retry a draw that an entry already holds.
            var draws: u32 = 0;
            while (draws < token_draws_max) : (draws += 1) {
                const new_token = drawToken(self.io) catch {
                    return self.refuse(req, sink, .internal, "no secure random source");
                };
                const r = self.resumeLease(
                    req.conn,
                    req.request_id,
                    token,
                    self.readClock(),
                    new_token,
                ) catch |err| {
                    const code: protocol.ErrorCode, const detail: []const u8 = switch (err) {
                        error.TokenInUse => continue,
                        error.NoSuchLease => .{ .unknown_session, "no such lease" },
                        error.ShareFull => .{ .sessions_exhausted, detail_share_full },
                    };
                    return self.refuse(req, sink, code, detail);
                };

                // Answer at once, or at the idle transition.
                const entry = self.entryByProto(r.session_id, req.conn).?;
                if (r.ready) {
                    std.debug.assert(entry.state == .idle);

                    self.emitOrPark(entry, sink, self.resumedEvent(entry, req.request_id));
                } else {
                    std.debug.assert(entry.resume_wait != null);
                }

                return;
            }

            self.refuse(req, sink, .internal, "token draws collided");
        }

        /// The `resumed` reply of an idle entry. It names the token, the final
        /// position, and the last finished request.
        fn resumedEvent(self: *Self, entry: *const SessionEntry, request_id: u32) OutEvent {
            std.debug.assert(entry.state == .idle);
            std.debug.assert(entry.resume_wait == null);

            const lease = entry.lease.?;
            const last: ?protocol.LastDone = if (entry.last_done) |d| .{
                .request_id = d.request_id,
                .finish = d.finish,
                .gen_tokens = d.gen_tokens,
                .reasoning_tokens = d.reasoning_tokens,
            } else null;

            return .{
                .request_id = request_id,
                .session_id = entry.proto_id,
                .body = .{ .resumed = .{
                    .token = lease.token,
                    .kv_pos = self.pool.posOf(entry.engine_id),
                    .turns_done = entry.turns_done,
                    .last = last,
                } },
            };
        }

        // -- the session lease ----------------------------------------------

        /// Keep session `proto_id` of connection `conn` with `token` for at most `ttl_ms`. Returns
        /// the granted time to live. A zero drops the lease. A new token replaces the old one. The
        /// wire handler draws `token` with `drawToken` and no other source.
        pub fn keepLease(
            self: *Self,
            proto_id: u32,
            conn: u32,
            ttl_ms: u32,
            token: Token,
        ) KeepError!u32 {
            defer self.checkLeases();

            const entry = self.entryById(proto_id) orelse return error.NoSuchSession;
            switch (entry.owner) {
                .attached => |c| if (c != conn) return error.OtherOwner,
                .detached => return error.OtherOwner,
            }

            if (ttl_ms == 0) {
                entry.lease = null;
                log.info("keep: session {d}, lease dropped", .{proto_id});
                return 0;
            }

            if (self.cfg.resume_ttl_max_ms == 0) return error.ResumptionOff;
            // The check stays in every build mode, because a duplicate token gives one client
            // the session of another.
            if (self.entryByToken(token) != null) return error.TokenInUse;

            // A zero token is the wire form of no lease.
            std.debug.assert(!std.mem.allEqual(u8, &token, 0));
            const granted = @min(ttl_ms, self.cfg.resume_ttl_max_ms);
            entry.lease = .{ .token = token, .ttl_ms = granted };
            log.info("keep: session {d}, lease {d} ms", .{ proto_id, granted });
            std.debug.assert(granted >= 1);
            return granted;
        }

        /// Connection `conn` presents `token` at `now`, and the entry then holds `new_token`. See
        /// "Resume" in the header. The wire handler draws `new_token` with `drawToken` and no other
        /// source.
        pub fn resumeLease(
            self: *Self,
            conn: u32,
            request_id: u32,
            token: Token,
            now: Io.Clock.Timestamp,
            new_token: Token,
        ) ResumeError!Resumed {
            defer self.checkLeases();

            std.debug.assert(!std.mem.allEqual(u8, &new_token, 0));

            // A resume is an admission, so it runs the expiry scan first. A
            // entry at its deadline is then free, and its token is gone.
            self.expireDetached(now);
            if (self.entryByToken(new_token) != null) return error.TokenInUse;
            if (self.shareFull(conn)) return error.ShareFull;

            // Move the entry to the new connection.
            const entry = self.entryByToken(token) orelse return error.NoSuchLease;
            switch (entry.owner) {
                .attached => |c| {
                    log.info("takeover: session {d} conn {d} -> {d}", .{ entry.proto_id, c, conn });
                    self.dropEffect(entry);
                },
                // The scan above freed every entry at its deadline.
                .detached => |d| std.debug.assert(now.raw.nanoseconds < d.raw.nanoseconds),
            }

            entry.owner = .{ .attached = conn };
            entry.lease.?.token = new_token;

            // Answer at once, or wait in the entry until it is idle.
            const ready = entry.state == .idle;
            std.debug.assert(entry.resume_wait == null);
            std.debug.assert(entry.pending_event == null);
            if (ready) {
                log.info("resume: session {d} conn {d}", .{ entry.proto_id, conn });
            } else {
                entry.resume_wait = .{ .conn = conn, .request_id = request_id };
                log.info("resume: session {d} conn {d} waits for idle", .{ entry.proto_id, conn });
            }

            return .{ .session_id = entry.proto_id, .ready = ready };
        }

        /// The end of connection `conn` at `now`. An entry with no lease frees. A leased entry
        /// detaches until `now` plus its time to live, after the drop effect.
        pub fn detachConn(self: *Self, conn: u32, now: Io.Clock.Timestamp) void {
            defer self.checkLeases();

            for (self.entries) |*entry| {
                if (entry.state == .free or !attachedTo(entry, conn)) continue;

                const lease = entry.lease orelse {
                    self.releaseEntry(entry);
                    continue;
                };
                self.dropEffect(entry);
                const deadline = now.addDuration(msDuration(lease.ttl_ms));
                const bound = now.addDuration(msDuration(self.cfg.resume_ttl_max_ms));
                std.debug.assert(deadline.raw.nanoseconds <= bound.raw.nanoseconds);
                entry.owner = .{ .detached = deadline };
                log.info("detach: session {d}, lease {d} ms", .{ entry.proto_id, lease.ttl_ms });
            }

            // Postcondition: no session is attached to the connection. The loop above frees or
            // detaches each one, and an emit names only an attached session or a request sender.
            for (self.entries) |*entry| {
                std.debug.assert(entry.state == .free or !attachedTo(entry, conn));
            }
        }

        /// The lazy expiry scan at `now`. It frees every detached entry at or
        /// past its deadline, after it cancels the prefill in flight.
        pub fn expireDetached(self: *Self, now: Io.Clock.Timestamp) void {
            for (self.entries) |*entry| {
                if (!expired(entry, now)) continue;

                self.lease_expires += 1;
                log.info(
                    "expire: session {d}, {d} expired",
                    .{ entry.proto_id, self.lease_expires },
                );
                self.releaseDetached(entry);
            }
        }

        /// The drop effect on one entry. A turn takes the cancel rule. An opening or append prefill
        /// runs on, because it rebuilds context that the client asked for. A parked event and a
        /// waiting resume belong to the old connection, so they go.
        fn dropEffect(self: *Self, entry: *SessionEntry) void {
            std.debug.assert(entry.state != .free);
            std.debug.assert(entry.owner == .attached);

            switch (entry.state) {
                .decode => self.cancelTurn(entry),
                .prefill => if (entry.kind == .turn) self.cancelTurn(entry),
                .idle, .free => {},
            }

            entry.pending_event = null;
            entry.resume_wait = null;
        }

        /// Release a detached entry. A prefill in flight stops first, so no
        /// row of the released session reaches the engine.
        fn releaseDetached(self: *Self, entry: *SessionEntry) void {
            std.debug.assert(entry.owner == .detached);
            std.debug.assert(entry.pending_event == null);

            if (entry.state == .prefill) {
                // The drop cancelled any turn, so only these two remain.
                std.debug.assert(entry.kind == .opening or entry.kind == .append);

                self.lease_prefills_cut += 1;
                log.info(
                    "stop: session {d} {t} prefill cancelled after {d} of {d} tokens",
                    .{ entry.proto_id, entry.kind, entry.fed, entry.turn_tokens.len },
                );
                self.gpa.free(entry.turn_tokens);
                entry.turn_tokens = &.{};
                entry.state = .idle;
                entry.kind = .turn;
            }

            std.debug.assert(entry.state != .prefill);
            self.releaseEntry(entry);
        }

        /// Send a waiting resume at the moment its entry becomes idle. Every idle transition calls
        /// it, so no request can reach the entry between the two. The events before it went, so the
        /// cell is empty.
        fn deliverResume(self: *Self, entry: *SessionEntry, sink: anytype) void {
            std.debug.assert(entry.state == .idle);

            const wait = entry.resume_wait orelse return;
            std.debug.assert(attachedTo(entry, wait.conn));
            std.debug.assert(entry.pending_event == null);
            log.info(
                "resume ready: session {d} conn {d} at pos {d}",
                .{ entry.proto_id, wait.conn, self.pool.posOf(entry.engine_id) },
            );
            entry.resume_wait = null;
            self.emitOrPark(entry, sink, self.resumedEvent(entry, wait.request_id));
        }

        /// The refusal of a request on an entry whose resume still waits. The
        /// client has no `resumed` answer yet, so the session is busy.
        fn refuseWaiting(self: *Self, req: Request, sink: anytype) void {
            self.refuse(req, sink, .session_busy, "resume in flight");
        }

        /// The lease clock for a scan, or null when resumption is off. No
        /// entry then holds a lease, so a scan has nothing to do.
        fn leaseNow(self: *const Self) ?Io.Clock.Timestamp {
            if (self.cfg.resume_ttl_max_ms == 0) return null;
            return self.readClock();
        }

        /// Pin every lease clock read at `at`. Any thread can move the pin.
        pub fn pinClock(self: *Self, at: Io.Clock.Timestamp) void {
            std.debug.assert(at.clock == .awake);

            const ns: i64 = @intCast(at.raw.nanoseconds);
            std.debug.assert(ns != clock_unpinned);
            self.clock_pin_ns.store(ns, .monotonic);
        }

        fn readClock(self: *const Self) Io.Clock.Timestamp {
            const ns = self.clock_pin_ns.load(.monotonic);
            if (ns == clock_unpinned) return Io.Clock.Timestamp.now(self.io, .awake);
            return .{ .raw = .fromNanoseconds(ns), .clock = .awake };
        }

        /// A detached entry at or past its deadline.
        fn expired(entry: *const SessionEntry, now: Io.Clock.Timestamp) bool {
            if (entry.state == .free) return false;
            return switch (entry.owner) {
                .attached => false,
                .detached => |d| d.raw.nanoseconds <= now.raw.nanoseconds,
            };
        }

        /// The entry that yields to a live open: the detached entry with the
        /// earliest deadline, then the lowest index.
        fn pickYield(entries: []const SessionEntry) ?usize {
            var best: ?usize = null;
            for (entries, 0..) |*entry, i| {
                if (entry.state == .free) continue;

                const d = switch (entry.owner) {
                    .attached => continue,
                    .detached => |d| d,
                };

                if (best) |b| {
                    if (d.raw.nanoseconds >= entries[b].owner.detached.raw.nanoseconds) continue;
                }

                best = i;
            }

            return best;
        }

        /// The sessions attached to connection `conn`. The engine thread calls it after a request
        /// that attaches a session or takes one away.
        pub fn sessionsOf(self: *const Self, conn: u32) u32 {
            var n: u32 = 0;
            for (self.entries) |*entry| {
                if (entry.state != .free and attachedTo(entry, conn)) {
                    n += 1;
                }
            }

            return n;
        }

        fn attachedTo(entry: *const SessionEntry, conn: u32) bool {
            return switch (entry.owner) {
                .attached => |c| c == conn,
                .detached => false,
            };
        }

        /// The connection of an attached entry. Only an attached entry emits.
        fn connOf(entry: *const SessionEntry) u32 {
            return entry.owner.attached;
        }

        /// Check the rules of "Lease table" and "Shares" in the header after each event.
        fn checkLeases(self: *const Self) void {
            if (!std.debug.runtime_safety) return;
            std.debug.assert(self.next_proto_id >= 1);
            for (self.entries, 0..) |*entry, i| {
                // A free entry holds nothing.
                if (entry.state == .free) {
                    std.debug.assert(entry.lease == null);
                    std.debug.assert(entry.proto_id == 0);
                    std.debug.assert(entry.resume_wait == null);

                    continue;
                }

                // A live entry holds a counted id, and a detached entry holds a lease.
                std.debug.assert(entry.proto_id >= 1);
                std.debug.assert(entry.proto_id < self.next_proto_id);
                if (entry.owner == .detached) {
                    std.debug.assert(entry.lease != null);
                    std.debug.assert(entry.pending_event == null);
                    std.debug.assert(entry.resume_wait == null);
                }

                if (entry.resume_wait) |wait| {
                    // A waiting resume ends at the idle transition, and its
                    // entry emits nothing before it.
                    std.debug.assert(entry.state != .idle);
                    std.debug.assert(attachedTo(entry, wait.conn));
                    std.debug.assert(entry.pending_event == null);
                }

                if (entry.lease) |lease| {
                    std.debug.assert(lease.ttl_ms >= 1);
                    std.debug.assert(lease.ttl_ms <= self.cfg.resume_ttl_max_ms);
                }

                // No two live entries share a session id or a token. The first entry of a
                // connection counts every entry that the connection holds.
                var held: u32 = 1;
                for (self.entries[i + 1 ..]) |*other| {
                    if (other.state == .free) continue;
                    std.debug.assert(other.proto_id != entry.proto_id);
                    if (entry.lease != null and other.lease != null) {
                        std.debug.assert(
                            !std.mem.eql(u8, &entry.lease.?.token, &other.lease.?.token),
                        );
                    }

                    if (entry.owner == .attached and attachedTo(other, connOf(entry))) {
                        held += 1;
                    }
                }

                if (entry.owner == .attached) {
                    std.debug.assert(held <= self.cfg.sessions_per_connection);
                }
            }
        }

        /// A shutdown cuts every entry with work in flight by the cancel rule of its kind. A turn
        /// ends with a `done`, an append with an `err`, and an opening with no event.
        fn cancelAll(self: *Self, sink: anytype) void {
            log.debug("cancel-all", .{});
            for (self.entries) |*entry| {
                switch (entry.state) {
                    .free, .idle => {},
                    .decode => self.cancelTurn(entry),
                    .prefill => switch (entry.kind) {
                        .turn => self.cancelTurn(entry),
                        .append => self.cancelAppend(entry, sink),
                        .opening => self.stopOpening(entry, sink),
                    },
                }
            }

            self.finalizeStops(sink);
        }

        /// A shutdown ends an opening prefill with no event. An opening has no turn, so no `done`
        /// can name it, and the close of the connection answers the open.
        fn stopOpening(self: *Self, entry: *SessionEntry, sink: anytype) void {
            std.debug.assert(entry.state == .prefill);
            std.debug.assert(entry.kind == .opening);

            log.info(
                "stop: session {d} opening cancelled after {d} of {d} tokens",
                .{ entry.proto_id, entry.fed, entry.turn_tokens.len },
            );
            self.gpa.free(entry.turn_tokens);
            entry.turn_tokens = &.{};
            entry.state = .idle;
            entry.kind = .turn;
            self.deliverResume(entry, sink);
        }

        // -- the tick, in order ---------------------------------------------

        fn retryParked(self: *Self, sink: anytype) void {
            for (self.entries) |*entry| {
                if (entry.pending_event) |ev| {
                    if (sink.emit(connOf(entry), ev)) {
                        log.debug("unpark: session {d}", .{entry.proto_id});
                        entry.pending_event = null;
                    }
                }
            }
        }

        fn finalizeStops(self: *Self, sink: anytype) void {
            const now = Io.Clock.Timestamp.now(self.io, .awake);
            for (self.entries) |*entry| {
                if (entry.state != .decode or entry.stop == null or entry.pending_event != null) {
                    continue;
                }

                // A cut turn carries its last token.
                const reason = entry.stop.?;
                entry.carry = switch (reason) {
                    .max_tokens, .cancelled => if (entry.generated > 0) entry.last_token else null,
                    .eos, .context_full => null,
                };

                // Report the turn to the client and to a later resume.
                self.emitOrPark(entry, sink, .{
                    .request_id = entry.request_id,
                    .session_id = entry.proto_id,
                    .body = .{ .done = .{
                        .finish_reason = reason,
                        .turn_prompt_tokens = entry.turn_prompt_len,
                        .gen_tokens = entry.generated,
                        .kv_pos = self.pool.posOf(entry.engine_id),
                        .prefill_ms = time.msBetweenF32(entry.t_turn_start, entry.t_prefill_end),
                        .decode_ms = time.msBetweenF32(entry.t_prefill_end, now),
                        .reasoning_tokens = entry.think_n,
                    } },
                });
                entry.last_done = .{
                    .request_id = entry.request_id,
                    .finish = reason,
                    .gen_tokens = entry.generated,
                    .reasoning_tokens = entry.think_n,
                };
                log.info(
                    "complete: session {d} {s}, {d}+{d} tokens, prefill {d:.0}ms decode {d:.0}ms",
                    .{
                        entry.proto_id,
                        @tagName(reason),
                        entry.turn_prompt_len,
                        entry.generated,
                        time.msBetweenF32(entry.t_turn_start, entry.t_prefill_end),
                        time.msBetweenF32(entry.t_prefill_end, now),
                    },
                );

                // End the turn. The idle transition delivers a waiting resume.
                entry.stop = null;
                entry.state = .idle;
                entry.turns_done += 1;
                self.deliverResume(entry, sink);
            }
        }

        /// The one composed engine call of a tick. Returns whether the tick did work.
        fn runComposedStep(self: *Self, sink: anytype) bool {
            const sp = span.begin(.composed_step);
            defer sp.end();

            var rows: DecodeRows = .{};
            self.admitDecodeRows(&rows);
            const feeding = self.pickFeeding();
            if (feeding == null and rows.n == 0) return false;

            const feed_chunk: ?session_mod.FeedChunk =
                if (feeding) |entry| self.nextChunk(entry, rows.n) else null;
            const chunk: usize = if (feed_chunk) |fc| fc.tokens.len else 0;
            logBatch(&rows, feeding, chunk);

            const first = self.pool.advance(
                feed_chunk,
                rows.engineIds(),
                rows.tokensOut(),
            ) catch |err| {
                // The tiers are over-subscribed: park this turn.
                if (isParkError(err)) {
                    // A failed call of decode rows alone needs no retry: it is the same call.
                    const f = feeding orelse return self.parkDecodeRows(rows.n);
                    return self.parkPrefill(f, &rows, err, sink);
                }

                // Any other error fails every row of the call.
                if (feeding) |entry| {
                    self.failTurn(entry, sink, detail_advance_failed);
                }

                self.failDecodeRows(&rows, sink);
                return true;
            };

            if (feeding) |entry| {
                self.finishChunk(entry, chunk, first, sink);
            }

            self.acceptDecodeRows(&rows, sink);
            return true;
        }

        /// Admit decode rows by round-robin up to `max_batch`. A session out of KV stops here and
        /// finalizes on the next tick.
        fn admitDecodeRows(self: *Self, rows: *DecodeRows) void {
            std.debug.assert(rows.n == 0);

            for (0..self.entries.len) |k| {
                if (rows.n == self.cfg.max_batch) break;

                const entry = &self.entries[(self.rr_cursor + k) % self.entries.len];
                // The drop cancelled the turn of a detached entry, so it takes no decode row.
                if (entry.state == .decode and entry.owner == .detached) {
                    std.debug.assert(entry.stop != null);
                }
                if (entry.state != .decode or entry.stop != null or entry.pending_event != null) {
                    continue;
                }

                std.debug.assert(entry.owner == .attached);
                if (self.pool.capacityRemaining(entry.engine_id) == 0) {
                    entry.stop = .context_full;
                    log.info(
                        "stop: session {d} context_full at pos {d}",
                        .{ entry.proto_id, self.pool.posOf(entry.engine_id) },
                    );
                    continue;
                }

                rows.ids[rows.n] = entry.engine_id;
                rows.members[rows.n] = entry;
                rows.n += 1;
            }

            self.rr_cursor = (self.rr_cursor + 1) % self.entries.len;
        }

        /// The longest-waiting feeding turn. A parked event holds its turn back.
        fn pickFeeding(self: *Self) ?*SessionEntry {
            var feeding: ?*SessionEntry = null;
            for (self.entries) |*entry| {
                if (entry.state != .prefill or entry.pending_event != null) continue;
                if (entry.owner == .detached) std.debug.assert(entry.kind != .turn);
                if (feeding == null or entry.prefill_seq < feeding.?.prefill_seq) {
                    feeding = entry;
                }
            }

            return feeding;
        }

        /// The next chunk of `entry` beside `n_decode` rows, by the rule in `prefill_chunk.zig`.
        /// Null when a saturated decode batch leaves no room, so the turn waits for the next tick.
        fn nextChunk(self: *Self, entry: *SessionEntry, n_decode: usize) ?session_mod.FeedChunk {
            const chunk = prefill_chunk.chunk(
                self.cfg.prefill,
                entry.turn_tokens.len - entry.fed,
                n_decode,
                self.cfg.max_batch,
                self.cfg.max_step_rows,
            );
            // Never forward a zero-length chunk: `advance` asserts tokens.len >= 1.
            if (chunk == 0) return null;

            // The engine samples the first token of a turn inside the advance, so the chunk
            // enters the context set first. A second mark of a parked chunk is a no-op.
            const tokens = entry.turn_tokens[entry.fed..][0..chunk];
            self.markFed(entry, tokens);
            return .{
                .id = entry.engine_id,
                .tokens = tokens,
                .final = entry.kind == .turn and entry.fed + chunk == entry.turn_tokens.len,
            };
        }

        fn logBatch(rows: *const DecodeRows, feeding: ?*const SessionEntry, chunk: usize) void {
            if (feeding) |f| {
                log.debug(
                    "batch: decode {d} engines {any}, prefill {d} rows (session {d} fed {d}/{d})",
                    .{
                        rows.n,
                        rows.engineIds(),
                        chunk,
                        f.proto_id,
                        f.fed + chunk,
                        f.turn_tokens.len,
                    },
                );
            } else {
                log.debug("batch: decode {d} engines {any}", .{ rows.n, rows.engineIds() });
            }
        }

        /// The errors that park a turn. The CPU pool and the test fakes cannot return some of
        /// them, so a compare with such an error is statically false and its prong compiles away.
        inline fn isParkError(err: anytype) bool {
            return err == error.NoEvictableSlot or err == error.RamBudgetExceeded;
        }

        /// Park the decode rows of a tick. A parked tick reports no work, so this returns false.
        fn parkDecodeRows(self: *Self, n_decode: usize) bool {
            log.debug("park: {d} decode rows blocked on kv residency", .{n_decode});
            self.parks += 1;
            return false;
        }

        /// Park the prefill of `f`, and retry the decode rows alone. Returns whether the tick
        /// did work.
        fn parkPrefill(
            self: *Self,
            f: *SessionEntry,
            rows: *DecodeRows,
            err: anytype,
            sink: anytype,
        ) bool {
            if (err == error.RamBudgetExceeded) {
                log.debug("park: prefill session {d}, ram budget exhausted", .{f.proto_id});
            } else {
                log.debug(
                    "park: prefill session {d}, no evictable slot ({d} decoding)",
                    .{ f.proto_id, rows.n },
                );
            }

            self.parks += 1;
            if (rows.n == 0) return false;

            const decoded = self.pool.advance(
                null,
                rows.engineIds(),
                rows.tokensOut(),
            ) catch |retry_err| {
                if (isParkError(retry_err)) return self.parkDecodeRows(rows.n);

                self.failDecodeRows(rows, sink);
                return true;
            };
            std.debug.assert(decoded == null);

            self.acceptDecodeRows(rows, sink);
            return true;
        }

        fn failDecodeRows(self: *Self, rows: *const DecodeRows, sink: anytype) void {
            for (rows.entries()) |entry| {
                self.failTurn(entry, sink, detail_advance_failed);
            }
        }

        fn acceptDecodeRows(self: *Self, rows: *const DecodeRows, sink: anytype) void {
            logAdvanceTokens(rows.entries(), rows.tokens());
            for (rows.entries(), rows.tokens()) |entry, token| {
                self.acceptToken(entry, sink, token);
            }
        }

        /// Count the fed chunk of `entry`. The last chunk ends the prefill. A turn then takes
        /// `first`, the token that the engine sampled.
        fn finishChunk(
            self: *Self,
            entry: *SessionEntry,
            chunk: usize,
            first: ?u32,
            sink: anytype,
        ) void {
            entry.fed += chunk;
            if (entry.fed != entry.turn_tokens.len) return;

            self.maybeInsertPrefix(entry);
            switch (entry.kind) {
                .opening => {
                    self.emitOpened(entry, sink);
                    self.endPrefill(entry, .idle, sink);
                },
                .append => {
                    self.emitAppended(entry, sink);
                    self.endPrefill(entry, .idle, sink);
                },
                .turn => {
                    entry.t_prefill_end = Io.Clock.Timestamp.now(self.io, .awake);
                    self.endPrefill(entry, .decode, sink);
                    self.acceptToken(entry, sink, first.?);
                },
            }
        }

        /// One line per tick with the produced tokens, so the serve stream
        /// reports outcomes, not only the composed inputs.
        fn logAdvanceTokens(members: []const *SessionEntry, out: []const u32) void {
            if (members.len == 0) return;

            var protos: [limits.max_batch]u32 = undefined;
            for (members, 0..) |entry, i| {
                protos[i] = entry.proto_id;
            }

            log.debug(
                "tokens: {any} for sessions {any}",
                .{ out[0..members.len], protos[0..members.len] },
            );
        }

        // -- shared turn machinery ------------------------------------------

        /// Record one accepted token. A pending forced id replaces the sampled one. The token is
        /// the pending token of the engine, and it stays uncommitted until the next step.
        fn acceptToken(self: *Self, entry: *SessionEntry, sink: anytype, sampled: u32) void {
            std.debug.assert(entry.state == .decode);
            std.debug.assert(entry.stop == null);
            std.debug.assert(entry.forced_head <= entry.forced_len);

            // Take the token, and mark it in both id sets.
            const forced = entry.forced_head < entry.forced_len;
            const token = if (forced) self.popForced(entry) else sampled;
            entry.generated += 1;
            entry.last_token = token;
            self.tokens_generated_total += 1;
            // Every sampled and forced id counts for both penalties of the
            // next draw. The id enters the KV at the next step.
            sampler.markId(self.presenceOf(entry), self.cfg.vocab_size, token);
            self.markFed(entry, &.{token});
            if (comptime core_log.hot) {
                log_token.debug(
                    "token: session {d} #{d} id {d}",
                    .{ entry.proto_id, entry.generated, token },
                );
            }

            // The event goes out or parks before the stop decision.
            const channel = tokenChannel(entry.think_open, token, self.cfg.think.close_id);
            self.emitOrPark(
                entry,
                sink,
                OutEvent.token(
                    entry.request_id,
                    entry.proto_id,
                    token,
                    channel,
                    self.codec.piece(token),
                ),
            );

            // Decide the stop, then update the think block.
            if (token == self.cfg.eos_id) {
                entry.stop = .eos;
            } else if (entry.generated >= entry.max_tokens_turn) {
                entry.stop = .max_tokens;
            }

            self.acceptThink(entry, token);
            if (entry.stop) |reason| {
                log.info(
                    "stop: session {d} {t} after {d} tokens",
                    .{ entry.proto_id, reason, entry.generated },
                );
            }
        }

        /// Take the next forced id. The engine steps its own pending token, so
        /// the forced id must replace it there too. Otherwise the model steps
        /// the sampled id behind the forced event.
        fn popForced(self: *Self, entry: *SessionEntry) u32 {
            std.debug.assert(entry.forced_head < entry.forced_len);

            const id = entry.forced[entry.forced_head];
            entry.forced_head += 1;
            std.debug.assert(id != self.cfg.eos_id);
            std.debug.assert(id < self.cfg.vocab_size);
            self.pool.overridePending(entry.engine_id, id);
            if (comptime core_log.hot) {
                log_token.debug(
                    "forced: session {d} id {d}, {d} left",
                    .{ entry.proto_id, id, entry.forced_len - entry.forced_head },
                );
            }

            return id;
        }

        /// The think bookkeeping. The close id closes the block and counts nothing. Every other id
        /// counts, and a count at a nonzero budget queues the forced ids. The close leads them, so
        /// the count stays in the budget.
        fn acceptThink(self: *const Self, entry: *SessionEntry, token: u32) void {
            if (!entry.think_open) return;

            const t = &self.cfg.think;
            if (token == t.close_id) {
                entry.think_open = false;
                return;
            }

            entry.think_n += 1;
            const budget = entry.think_budget;
            if (budget > 0 and entry.think_n == budget) {
                std.debug.assert(t.force_len >= 1);

                @memcpy(entry.forced[0..t.force_len], t.force[0..t.force_len]);
                entry.forced_len = t.force_len;
                entry.forced_head = 0;
            }

            if (budget > 0) std.debug.assert(entry.think_n <= budget);
        }

        fn startPrefill(
            self: *Self,
            entry: *SessionEntry,
            request_id: u32,
            tokens: []u32,
            kind: PrefillKind,
            sink: anytype,
        ) void {
            std.debug.assert(tokens.len > 0);
            std.debug.assert(entry.state == .idle);

            // Move the entry to prefill.
            log.debug(
                "prefill: session {d}, {d} prompt tokens ({t}, budget {d})",
                .{ entry.proto_id, tokens.len, kind, entry.max_tokens_turn },
            );
            entry.state = .prefill;
            entry.kind = kind;
            entry.request_id = request_id;
            entry.turn_tokens = tokens;
            entry.turn_prompt_len = @intCast(tokens.len);
            entry.fed = 0;
            entry.caches_prefix = false;
            entry.carry = null;
            entry.prefill_seq = self.next_prefill_seq;
            self.next_prefill_seq += 1;
            if (kind != .turn) {
                entry.t_turn_start = Io.Clock.Timestamp.now(self.io, .awake);
            }

            // An append takes no prefix.
            if (kind == .append) return;

            if (comptime can_snapshot) {
                // A full hit emits the first token at once, and a parked echo
                // holds the one event cell.
                if (entry.pending_event == null) {
                    self.tryRestorePrefix(entry, sink);
                } else {
                    log.debug(
                        "prefix: session {d} skips the restore behind a parked echo",
                        .{entry.proto_id},
                    );
                }
            }
        }

        /// Prefix-cache admission for a turn that starts at position zero: restore the prefix that
        /// the manager offers and prefill only the suffix.
        fn tryRestorePrefix(self: *Self, entry: *SessionEntry, sink: anytype) void {
            if (!self.prefix.enabled()) return;
            if (self.pool.posOf(entry.engine_id) != 0) return;

            entry.caches_prefix = true;
            // The slices of the plan live inside this call only: later inserts reallocate the list.
            switch (self.prefix.planRestore(entry.turn_tokens, entry.kind == .opening)) {
                .miss => self.prefix.recordMiss(),
                .ready => |hit| self.restoreFull(entry, hit, sink),
                .rows => |hit| self.restoreRows(entry, hit, sink),
            }
        }

        /// A full hit: restore the rows, and take the first token from the stored logits.
        fn restoreFull(self: *Self, entry: *SessionEntry, hit: PrefixHit, sink: anytype) void {
            // The engine samples the first token inside the restore, so the prefix enters
            // the context set first. The cold path of a failed restore feeds the same ids.
            self.markFed(entry, entry.turn_tokens[0..hit.n]);
            const first = self.pool.restoreReady(
                entry.engine_id,
                hit.n,
                hit.blob,
                hit.logits,
            ) catch |err| {
                log.debug(
                    "prefix full restore failed: session {d}: {t}",
                    .{ entry.proto_id, err },
                );
                self.prefix.recordMiss();
                return;
            };
            self.prefix.recordHit();
            // The exact entry already serves this prefix, so nothing inserts.
            entry.caches_prefix = false;
            log.info(
                "prefix full hit: session {d} restored {d} prompt tokens, zero forward rows",
                .{ entry.proto_id, hit.n },
            );
            entry.t_prefill_end = Io.Clock.Timestamp.now(self.io, .awake);
            self.endPrefill(entry, .decode, sink);
            self.acceptToken(entry, sink, first);
        }

        /// A partial hit: restore the rows, and leave the suffix to the prefill.
        fn restoreRows(self: *Self, entry: *SessionEntry, hit: PrefixHit, sink: anytype) void {
            self.pool.restorePrefix(entry.engine_id, hit.n, hit.blob) catch |err| {
                log.debug("prefix restore failed: session {d}: {t}", .{ entry.proto_id, err });
                self.prefix.recordMiss();
                return;
            };
            self.markFed(entry, entry.turn_tokens[0..hit.n]);
            self.prefix.recordHit();
            entry.fed = hit.n;
            log.info(
                "prefix hit: session {d} restored {d} of {d} prompt tokens",
                .{ entry.proto_id, hit.n, entry.turn_tokens.len },
            );
            if (entry.fed != entry.turn_tokens.len) return;

            // Only an opening restores its whole prompt here. A generate full hit is
            // `restoreFull`. Nothing inserts.
            std.debug.assert(entry.kind == .opening);

            entry.caches_prefix = false;
            self.emitOpened(entry, sink);
            self.endPrefill(entry, .idle, sink);
        }

        /// Offer a completed position-zero prefix to the cache. The manager
        /// owns the snapshot and the admission gate. This function owns the
        /// eligibility rule: a position-zero completion, offered once.
        fn maybeInsertPrefix(self: *Self, entry: *SessionEntry) void {
            if (comptime !can_snapshot) return;
            if (!entry.caches_prefix) return;

            entry.caches_prefix = false;
            self.prefix.insert(self.pool, .{
                .engine_id = entry.engine_id,
                .proto_id = entry.proto_id,
                .tokens = entry.turn_tokens,
                .opening = entry.kind == .opening,
                .vocab_size = self.cfg.vocab_size,
                .t_turn_start = entry.t_turn_start,
            });
        }

        fn endPrefill(self: *Self, entry: *SessionEntry, state: SessionPhase, sink: anytype) void {
            log.debug("prefill done: session {d} -> {t}", .{ entry.proto_id, state });
            self.gpa.free(entry.turn_tokens);
            entry.turn_tokens = &.{};
            entry.state = state;
            // A turn reaches idle through its `done`, and an opening adds no
            // turn. An append adds one turn per pair.
            if (state == .idle and entry.kind == .append) {
                entry.turns_done += entry.append_n;
            }

            if (state == .decode) std.debug.assert(entry.kind == .turn);

            entry.kind = .turn;
            if (state == .idle) {
                self.deliverResume(entry, sink);
            }
        }

        fn failTurn(self: *Self, entry: *SessionEntry, sink: anytype, detail: []const u8) void {
            log.debug("turn failed: session {d}: {s}", .{ entry.proto_id, detail });
            if (entry.turn_tokens.len > 0) {
                self.gpa.free(entry.turn_tokens);
            }

            entry.turn_tokens = &.{};
            self.emitOrPark(
                entry,
                sink,
                OutEvent.fail(entry.request_id, entry.proto_id, .internal, detail),
            );
            entry.stop = null;
            entry.state = .idle;
            entry.kind = .turn;
            self.deliverResume(entry, sink);
        }

        fn withCarry(self: *Self, entry: *SessionEntry, encoded: []u32) Allocator.Error![]u32 {
            const carry = entry.carry orelse return encoded;
            defer self.gpa.free(encoded);

            const tokens = try self.gpa.alloc(u32, encoded.len + 1);
            tokens[0] = carry;
            @memcpy(tokens[1..], encoded);
            entry.carry = null;
            return tokens;
        }

        /// Session-scoped event. The entry's single pending cell is the
        /// backpressure bound. A parked event excludes the session from
        /// scheduling until the writer drains.
        fn emitOrPark(self: *Self, entry: *SessionEntry, sink: anytype, ev: OutEvent) void {
            _ = self;
            std.debug.assert(entry.pending_event == null);
            // No client reads the events of a detached entry, or the events
            // before a waiting resume, so they go.
            if (entry.owner == .detached or entry.resume_wait != null) {
                log.debug("drop: session {d} event, no attached reader", .{entry.proto_id});
                return;
            }

            if (!sink.emit(connOf(entry), ev)) {
                log.debug("park: session {d}, writer queue full", .{entry.proto_id});
                entry.pending_event = ev;
            }
        }

        /// Send a parked event of `entry` before a reply that names the session, so no reply
        /// overtakes it. Returns false after a disconnect: a failed flush takes the rule of
        /// `reply`.
        fn flushParked(self: *Self, entry: *SessionEntry, sink: anytype) bool {
            _ = self;
            const ev = entry.pending_event orelse return true;
            const conn = connOf(entry);
            if (!sink.emit(conn, ev)) {
                log.debug(
                    "flush: session {d} parked event refused, disconnect conn {d}",
                    .{ entry.proto_id, conn },
                );
                sink.disconnect(conn);
                return false;
            }

            log.debug("flush: session {d} parked event sent ahead of a reply", .{entry.proto_id});
            entry.pending_event = null;
            return true;
        }

        /// Control reply, which has no session cell to park in. A full queue
        /// here means a client sends requests and never reads. Disconnect it.
        fn reply(self: *Self, sink: anytype, req: Request, ev: OutEvent) void {
            _ = self;
            if (ev.body == .err) {
                log.debug(
                    "reject: conn {d} req {d}: {t}",
                    .{ req.conn, req.request_id, ev.body.err.code },
                );
            }

            if (!sink.emit(req.conn, ev)) {
                sink.disconnect(req.conn);
            }
        }

        fn helloAck(self: *const Self, req: Request) OutEvent {
            var body: OutEvent.Body = .{ .hello_ack = .{
                .max_seq = self.cfg.max_seq,
                .max_sessions = self.cfg.max_sessions,
                .max_batch = self.cfg.max_batch,
                .vocab_size = self.cfg.vocab_size,
                .eos_id = self.cfg.eos_id,
                .think_mode = self.thinkMode(),
                .think_budget = self.cfg.think.budget,
                .resume_ttl_max_ms = self.cfg.resume_ttl_max_ms,
                .recommended_think = self.cfg.recommended_think,
                .recommended_plain = self.cfg.recommended_plain,
                .name_len = @intCast(self.cfg.model_name.len),
                .name = undefined,
            } };
            @memcpy(body.hello_ack.name[0..self.cfg.model_name.len], self.cfg.model_name);
            return .{ .request_id = req.request_id, .session_id = 0, .body = body };
        }

        fn thinkMode(self: *const Self) protocol.ThinkMode {
            if (!self.cfg.think.hasTags()) return .none;
            return if (self.cfg.think.open_default) .on else .off;
        }

        fn statsAck(self: *const Self, req: Request) OutEvent {
            var open: u16 = 0;
            var decoding: u16 = 0;
            for (self.entries) |*entry| {
                if (entry.state != .free) {
                    open += 1;
                }

                if (entry.state == .decode) {
                    decoding += 1;
                }
            }

            const now = Io.Clock.Timestamp.now(self.io, .awake);
            return .{ .request_id = req.request_id, .session_id = 0, .body = .{ .stats_ack = .{
                .sessions_open = open,
                .sessions_decoding = decoding,
                .tokens_generated_total = self.tokens_generated_total,
                .uptime_ms = @intFromFloat(@max(0, time.msBetweenF32(self.t_init, now))),
            } } };
        }

        // -- session table --------------------------------------------------

        /// An entry for a live open or clone. `yielded` names the detached
        /// session that gave its entry up, or is 0 for a free entry.
        const Taken = struct { entry: *SessionEntry, yielded: u32 };

        /// The first free entry, else the detached entry that yields. A lease
        /// never blocks a live client. The caller reports the yield with its
        /// outcome, through `yieldDone` or `yieldLost`, as one event.
        fn freeEntry(self: *Self) ?Taken {
            for (self.entries) |*entry| {
                if (entry.state == .free) return .{ .entry = entry, .yielded = 0 };
            }

            const i = pickYield(self.entries) orelse return null;
            const entry = &self.entries[i];
            const yielded = entry.proto_id;
            std.debug.assert(yielded != 0);
            self.releaseDetached(entry);
            std.debug.assert(entry.state == .free);
            return .{ .entry = entry, .yielded = yielded };
        }

        fn yieldDone(self: *Self, taken: Taken, proto_id: u32) void {
            if (taken.yielded == 0) return;

            std.debug.assert(taken.entry.proto_id == proto_id);
            self.lease_yields += 1;
            log.info(
                "yield: session {d} for session {d}, {d} yielded",
                .{ taken.yielded, proto_id, self.lease_yields },
            );
        }

        /// A yield before an engine refusal loses the detached session. The release must come
        /// first, because the engine has no free session before it. An open cannot fail then,
        /// because the engine holds one session per KV slot. A clone on tiered KV can.
        fn yieldLost(self: *Self, taken: Taken, what: []const u8) void {
            if (taken.yielded == 0) return;

            std.debug.assert(taken.entry.state == .free);
            self.lease_yields += 1;
            self.lease_yields_lost += 1;
            log.info(
                "yield: session {d} lost, the engine refused the {s}, {d} lost",
                .{ taken.yielded, what, self.lease_yields_lost },
            );
        }

        /// Whether `conn` already holds its share of the session table. Open
        /// and clone check it before they take an entry, so no connection ever
        /// holds more than its share.
        fn shareFull(self: *const Self, conn: u32) bool {
            const share = self.cfg.sessions_per_connection;
            std.debug.assert(share >= 1);
            std.debug.assert(share <= self.cfg.max_sessions);
            var held: u32 = 0;
            for (self.entries) |*entry| {
                if (entry.state != .free and attachedTo(entry, conn)) {
                    held += 1;
                }
            }

            std.debug.assert(held <= share);
            return held == share;
        }

        /// The entry of session `proto_id` for a request on `conn`. Live ids
        /// are unique, so the first match is the only match. A detached entry
        /// never matches, so a guessed id grants nothing.
        fn entryByProto(self: *Self, proto_id: u32, conn: u32) ?*SessionEntry {
            if (proto_id == 0) return null;
            for (self.entries) |*entry| {
                const live = entry.state != .free and entry.proto_id == proto_id;
                if (live and attachedTo(entry, conn)) {
                    std.debug.assert(entry.owner == .attached);

                    return entry;
                }
            }

            return null;
        }

        /// The live entry of session `proto_id`, whatever its owner.
        fn entryById(self: *Self, proto_id: u32) ?*SessionEntry {
            if (proto_id == 0) return null;
            for (self.entries) |*entry| {
                if (entry.state != .free and entry.proto_id == proto_id) return entry;
            }

            return null;
        }

        /// The entry that holds `token`. Every compare runs in constant
        /// time, so the scan time does not depend on the token bytes.
        fn entryByToken(self: *Self, token: Token) ?*SessionEntry {
            var found: ?*SessionEntry = null;
            for (self.entries) |*entry| {
                const lease = entry.lease orelse continue;
                if (std.crypto.timing_safe.eql(Token, lease.token, token)) {
                    std.debug.assert(found == null);

                    found = entry;
                }
            }

            if (found) |entry| std.debug.assert(entry.state != .free);

            return found;
        }

        fn releaseEntry(self: *Self, entry: *SessionEntry) void {
            log.info("release: session {d} engine {d}", .{ entry.proto_id, entry.engine_id });
            if (entry.turn_tokens.len > 0) {
                self.gpa.free(entry.turn_tokens);
            }

            self.pool.destroy(entry.engine_id);
            entry.* = .{};
        }

        fn dropEntry(self: *Self, entry: *SessionEntry) void {
            self.pool.destroy(entry.engine_id);
            entry.* = .{};
        }
    };
}

// -- api --------------------------------------------------------------------

/// Draw a lease token from the secure random source. A failure refuses the keep or the resume. The
/// server never falls back to a weaker source.
pub fn drawToken(io: Io) error{EntropyUnavailable}!Token {
    var t: Token = undefined;
    io.randomSecure(&t) catch return error.EntropyUnavailable;
    return t;
}

/// The host bytes of the two id sets of one entry over `vocab_size` ids. The
/// boot RAM line sizes them.
pub fn sessionSetBytes(vocab_size: u32) u64 {
    return 2 * @as(u64, sampler.setWords(vocab_size)) * @sizeOf(u64);
}

// -- helpers ----------------------------------------------------------------

/// The order of a turn at index `turns_done`. The first turn of a session
/// takes the first-turn wrap, after a system prompt or with BOS. Later turns
/// continue. A kept pair takes the place of the turn that it replays.
fn turnOrder(has_system_prompt: bool, turns_done: u32) chat.PairOrder {
    if (turns_done > 0) return .continuation;
    return if (has_system_prompt) .first_after_system else .open_first;
}

/// The wrap of a live turn at index `turns_done`.
fn turnKind(has_system_prompt: bool, turns_done: u32) event.TurnKind {
    return kindOfOrder(turnOrder(has_system_prompt, turns_done));
}

fn kindOfOrder(order: chat.PairOrder) event.TurnKind {
    return switch (order) {
        .open_first => .open_first,
        .first_after_system => .first_after_system,
        .continuation => .continuation,
    };
}

/// The channel of a token accepted while the block is `open`. The flag is
/// the one before the think bookkeeping, so a forced id follows the rule of
/// a sampled one.
fn tokenChannel(open: bool, token: u32, close_id: u32) protocol.Channel {
    if (!open) return .content;
    return if (token == close_id) .close_tag else .reasoning;
}

/// Mean milliseconds per event. No event means no mean, which reads as zero.
fn avgMs(total_ms: f64, n: u64) f64 {
    if (n == 0) return 0;
    return total_ms / @as(f64, @floatFromInt(n));
}

/// A span of `ms` milliseconds on the clock that the lease reads.
fn msDuration(ms: u32) Io.Clock.Duration {
    return .{ .raw = .fromMilliseconds(ms), .clock = .awake };
}

/// The think config the launcher builds. The forced ids start with the close for every server
/// budget, because a request can bring its own budget. A nonzero server budget needs forced ids.
/// No forced id and no close is the end-of-sequence id, so a forced id never ends a turn.
fn assertThinkConfig(t: Think, eos_id: u32, vocab_size: u32) void {
    std.debug.assert(t.force_len <= limits.think_force_ids_max);

    for (t.force[0..t.force_len]) |id| {
        std.debug.assert(id != eos_id);
        std.debug.assert(id < vocab_size);
    }

    if (t.close_id != no_token) {
        std.debug.assert(t.close_id != eos_id);
        std.debug.assert(t.close_id < vocab_size);
    }

    if (t.open_id != no_token) std.debug.assert(t.open_id < vocab_size);

    if (t.force_len >= 1) {
        std.debug.assert(t.hasTags());
        std.debug.assert(t.force[0] == t.close_id);
    }

    if (t.budget > 0) std.debug.assert(t.force_len >= 1);

    if (t.open_default) std.debug.assert(t.hasTags());
}

/// The detail bound of an error event. The reference client keeps the same
/// number of bytes, so it never cuts a detail this server sends.
const detail_max = @typeInfo(@FieldType(@FieldType(OutEvent.Body, "err"), "detail")).array.len;

comptime {
    // An err payload is the code, the detail length, and the detail.
    std.debug.assert(protocol.header_bytes + 2 + 2 + detail_max <= protocol.max_server_frame_bytes);
}

/// The refusal of a turn that does not fit. The text fits the bound at the widest usize and u32, so
/// the fallback never runs.
fn turnTooLongDetail(buf: *[detail_max]u8, tokens: usize, room: u32) []const u8 {
    return std.fmt.bufPrint(buf, protocol.turn_too_long_fmt, .{ tokens, room }) catch
        "turn does not fit the window";
}

/// The refusal of a system prompt that does not fit. A request frame holds at most 64 KiB, so the
/// counts stay at six digits. The text then fits the bound.
fn systemTooLongDetail(buf: *[detail_max]u8, tokens: usize, window: u32) []const u8 {
    return std.fmt.bufPrint(buf, protocol.system_too_long_fmt, .{ tokens, window }) catch
        "system prompt does not fit the window";
}

// -- tests ------------------------------------------------------------------

// Test fixtures. The tests drive the control unit against a real `Sessions` pool over a
// canned-logits runner. The fake model chains `next = (token + 1) % 8`. The fake codec maps each
// text byte to `byte % 8` with no template wrap, and eos_id is 7. So byte 6 stops on EOS at once,
// and byte 'x' (120 -> 0) walks 1, 2, 3. A reader derives every expected stream by hand.
//
// `TestScheduler` runs over the plain pool. `ResScheduler` runs over the residency with three
// logical sessions over two physical VRAM slots, so the third open forces an eviction. Three turns
// at once (max_batch 2) make the scheduler swap the odd one out each tick. Two decoders that hold
// both slots also reach the `NoEvictableSlot` park when a third turn wants to prefill. Every
// stream must still walk the chain, because the residency keeps each carried pending token across
// an eviction. `TestRunner.markers` stands in for KV content and the output ignores it, so a
// correct stream proves the bookkeeping survives swaps. The GPU equivalence test proves the KV
// bytes. The presence of `evictSlot` makes `Residency` able to evict.
//
// `NoSnapScheduler` is the other side of the `can_snapshot` gate. `Sessions(RunnerT).can_snapshot`
// reads `@hasDecl(RunnerT, "evictSlot")`, so a runner without the swap surface compiles every
// prefix-cache path away. `NoSnapRunner` is `TestRunner` minus that surface. Its test is the one
// instantiation that analyzes the guarded prongs.

const Sessions = session_mod.Sessions;
const Residency = @import("residency.zig").Residency;
const StepRow = @import("../runtime/step_row.zig").StepRow;

const test_fakes = @import("../runtime/test_fakes.zig");
const test_vocab = test_fakes.vocab;

/// Admission times that `worthwhile` admits, from its own unit test. The fake runner has no real
/// cost, so the measured times let a preempted snapshot copy refuse an entry.
const admit_times_pass: prefix_manager.AdmitTimes = .{
    .prefill_ns = 270_000_000,
    .restore_ns = 4_000_000,
};

/// Admission times that `worthwhile` refuses, from its own unit test.
const admit_times_fail: prefix_manager.AdmitTimes = .{
    .prefill_ns = 3_000_000,
    .restore_ns = 4_000_000,
};

const TestRunner = struct {
    n_slots: u32,
    max_seq: u32,
    logits: []f32,
    rows_log: [256]StepRow = undefined,
    rows_log_len: usize = 0,
    // One marker per physical slot.
    markers: [limits.max_vram_slots]u32 = .{0} ** limits.max_vram_slots,
    evicts: u32 = 0,
    restores: u32 = 0,

    fn rowsSeen(self: *const TestRunner) []const StepRow {
        return self.rows_log[0..self.rows_log_len];
    }

    pub fn maxStepRows(self: *const TestRunner) u32 {
        return self.n_slots;
    }

    pub fn stepBatch(self: *TestRunner, rows: []const StepRow) !void {
        for (rows) |r| {
            std.debug.assert(self.rows_log_len < self.rows_log.len);

            self.rows_log[self.rows_log_len] = r;
            self.rows_log_len += 1;
            self.markers[r.slot] = r.token; // the slot's KV "becomes" its last token
        }

        test_fakes.writeCannedLogits(self.logits, rows);
    }

    // The fake holds no per-slot state, so the copy is a no-op. It exists so that
    // `Sessions(TestRunner).can_clone` holds and a test can reach `handleClone`.
    pub fn cloneSlotState(
        self: *TestRunner,
        source_slot: u32,
        target_slot: u32,
        pos: u32,
    ) error{}!void {
        _ = self;
        _ = source_slot;
        _ = target_slot;
        _ = pos;
    }

    // This fake holds one marker per slot, so a budget stated here counts whole sessions.
    pub fn evictSlotBytesFor(self: *const TestRunner, n_rows: u32) u64 {
        _ = self;
        _ = n_rows;
        return @sizeOf(u32);
    }

    pub fn evictSlot(self: *TestRunner, slot: u32, n_rows: u32, out: []u8) !void {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(out.len == @sizeOf(u32));

        std.mem.writeInt(u32, out[0..4], self.markers[slot], .little);
        self.evicts += 1;
    }

    pub fn restoreSlot(self: *TestRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(blob.len == @sizeOf(u32));

        self.markers[slot] = std.mem.readInt(u32, blob[0..4], .little);
        self.restores += 1;
    }
};

/// `TestRunner` minus the swap surface, so `can_snapshot` is false for it.
const NoSnapRunner = struct {
    n_slots: u32,
    max_seq: u32,
    logits: []f32,

    pub fn maxStepRows(self: *const NoSnapRunner) u32 {
        return self.n_slots;
    }

    pub fn stepBatch(self: *NoSnapRunner, rows: []const StepRow) !void {
        test_fakes.writeCannedLogits(self.logits, rows);
    }
};

const FakeCodec = struct {
    piece_table: [test_vocab][1]u8,
    /// The turn kinds that the scheduler asked for, in order, when a test reads them.
    kinds: ?*KindLog = null,
    /// The think switch of each render, in order, when a test reads it.
    opens: ?*OpenLog = null,

    const OpenLog = struct {
        seen: [16]bool = undefined,
        n: usize = 0,

        fn add(self: *OpenLog, open: bool) void {
            std.debug.assert(self.n < self.seen.len);

            self.seen[self.n] = open;
            self.n += 1;
        }
    };

    const KindLog = struct {
        seen: [16]event.TurnKind = undefined,
        n: usize = 0,

        fn add(self: *KindLog, turn: event.TurnKind) void {
            std.debug.assert(self.n < self.seen.len);

            self.seen[self.n] = turn;
            self.n += 1;
        }
    };

    fn initFake() FakeCodec {
        var t: [test_vocab][1]u8 = undefined;
        for (&t, 0..) |*p, i| {
            p[0] = @intCast('0' + i);
        }

        return .{ .piece_table = t };
    }

    pub fn turnTokens(
        self: *const FakeCodec,
        gpa: Allocator,
        turn: event.TurnKind,
        text: []const u8,
        plain: []const chat.PlainRange,
        open: bool,
    ) ![]u32 {
        _ = plain;
        if (self.kinds) |k| {
            k.add(turn);
        }

        if (self.opens) |o| {
            o.add(open);
        }

        const ids = try gpa.alloc(u32, text.len);
        for (text, 0..) |b, i| {
            ids[i] = @intCast(b % test_vocab);
        }

        return ids;
    }

    /// The pair renders as the user bytes then the reply bytes. A reply
    /// written as the pieces of its ids therefore re-encodes to those ids.
    pub fn pairTokens(
        self: *const FakeCodec,
        gpa: Allocator,
        order: chat.PairOrder,
        user: []const u8,
        assistant: []const u8,
    ) ![]u32 {
        if (self.kinds) |k| {
            k.add(kindOfOrder(order));
        }

        const ids = try gpa.alloc(u32, user.len + assistant.len);
        for (user, 0..) |b, i| {
            ids[i] = @intCast(b % test_vocab);
        }

        for (assistant, 0..) |b, i| {
            ids[user.len + i] = @intCast(b % test_vocab);
        }

        return ids;
    }

    pub fn piece(self: *const FakeCodec, id: u32) []const u8 {
        return &self.piece_table[id];
    }
};

const FakeSink = struct {
    const Seen = struct { conn: u32, ev: OutEvent };
    events: [256]Seen = undefined,
    n: usize = 0,
    fail_conn: ?u32 = null,
    /// Refuse this many emits of any connection, then accept again.
    fail_next: u32 = 0,
    disconnected: ?u32 = null,

    pub fn emit(self: *FakeSink, conn: u32, ev: OutEvent) bool {
        if (self.fail_conn == conn) return false;

        if (self.fail_next > 0) {
            self.fail_next -= 1;
            return false;
        }

        std.debug.assert(self.n < self.events.len);
        self.events[self.n] = .{ .conn = conn, .ev = ev };
        self.n += 1;
        return true;
    }

    pub fn disconnect(self: *FakeSink, conn: u32) void {
        self.disconnected = conn;
    }

    fn seen(self: *const FakeSink) []const Seen {
        return self.events[0..self.n];
    }

    fn last(self: *const FakeSink) Seen {
        std.debug.assert(self.n > 0);

        return self.events[self.n - 1];
    }
};

const TestPool = Sessions(TestRunner);
const TestScheduler = Scheduler(TestPool, FakeCodec);
const ResPool = Residency(Sessions(TestRunner));
const ResScheduler = Scheduler(ResPool, FakeCodec);
const NoSnapScheduler = Scheduler(Sessions(NoSnapRunner), FakeCodec);

/// The answer of a lookup that finds no session entry.
const no_entry: ?*TestScheduler.SessionEntry = null;

const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    runner: *TestRunner,
    pool: *TestPool,
    codec: *FakeCodec,
    sched: *TestScheduler,
    sink: *FakeSink,

    fn setUp(gpa: Allocator, n_slots: u32, max_seq: u32, max_batch: ?u16) !Fixture {
        return setUpPrefix(gpa, n_slots, max_seq, max_batch, 0);
    }

    fn setUpPrefix(
        gpa: Allocator,
        n_slots: u32,
        max_seq: u32,
        max_batch: ?u16,
        prefix_bytes: u64,
    ) !Fixture {
        return setUpDisk(gpa, n_slots, max_seq, max_batch, prefix_bytes, 0, "");
    }

    fn setUpDisk(
        gpa: Allocator,
        n_slots: u32,
        max_seq: u32,
        max_batch: ?u16,
        prefix_bytes: u64,
        prefix_disk_bytes: u64,
        cache_dir: []const u8,
    ) !Fixture {
        return build(gpa, n_slots, max_seq, max_batch, .{
            .ram_bytes = prefix_bytes,
            .disk_bytes = prefix_disk_bytes,
            .dir = cache_dir,
            .model_fp = 0xF00D,
        }, .{});
    }

    /// Three slots with think tags. `Scheduler.init` checks the think config.
    fn setUpThink(gpa: Allocator, max_seq: u32, prefix_bytes: u64, think: Think) !Fixture {
        return build(
            gpa,
            3,
            max_seq,
            null,
            .{ .ram_bytes = prefix_bytes, .model_fp = 0xF00D },
            think,
        );
    }

    fn build(
        gpa: Allocator,
        n_slots: u32,
        max_seq: u32,
        max_batch: ?u16,
        cache: prefix_manager.CacheConfig,
        think: Think,
    ) !Fixture {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        errdefer arena_state.deinit();

        const arena = arena_state.allocator();

        // Build the runner, the pool, and the codec in the arena.
        const runner = try arena.create(TestRunner);
        runner.* = .{
            .n_slots = n_slots,
            .max_seq = max_seq,
            .logits = try arena.alloc(f32, n_slots * test_vocab),
        };
        @memset(runner.logits, 0);
        const pool = try arena.create(TestPool);
        pool.* = try TestPool.init(arena, runner);
        const codec = try arena.create(FakeCodec);
        codec.* = FakeCodec.initFake();

        // Build the scheduler and the sink.
        const sched = try arena.create(TestScheduler);
        sched.* = try TestScheduler.init(arena, gpa, std.testing.io, pool, codec, .{
            .max_sessions = @intCast(n_slots),
            .max_batch = max_batch orelse @intCast(n_slots),
            .max_seq = max_seq,
            .max_step_rows = n_slots,
            // These tests count rows per pass under the spare-rows rule.
            .prefill = .spare_rows,
            .model_name = "fake",
            .vocab_size = test_vocab,
            .eos_id = 7,
            .cache = cache,
            .think = think,
        });
        sched.prefix.pinned_times = admit_times_pass;
        const sink = try arena.create(FakeSink);
        sink.* = .{};

        return .{
            .arena_state = arena_state,
            .runner = runner,
            .pool = pool,
            .codec = codec,
            .sched = sched,
            .sink = sink,
        };
    }

    fn tearDown(self: *Fixture) void {
        self.sched.deinit();
        self.arena_state.deinit();
    }

    fn drain(self: *Fixture) void {
        while (self.sched.tick(self.sink) != .idle) {}
    }

    fn openSession(self: *Fixture, conn: u32, system_prompt: []const u8) !u32 {
        const dup = try std.testing.allocator.dupe(u8, system_prompt);
        _ = self.sched.handle(.{ .conn = conn, .request_id = 1, .session_id = 0, .body = .{
            .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
        } }, self.sink);
        self.drain();
        const e = self.sink.last();
        try std.testing.expectEqual(OutEvent.Body.session_opened, std.meta.activeTag(e.ev.body));
        return e.ev.session_id;
    }

    fn generate(self: *Fixture, conn: u32, sid: u32, text: []const u8, max_tokens: u32) !void {
        const dup = try std.testing.allocator.dupe(u8, text);
        _ = self.sched.handle(.{ .conn = conn, .request_id = 9, .session_id = sid, .body = .{
            .generate = .{ .max_tokens = max_tokens, .sampling = null, .text = dup },
        } }, self.sink);
    }

    /// Send one `session_append` frame under request id 11.
    fn append(self: *Fixture, conn: u32, sid: u32, pairs: []const protocol.Pair) !void {
        var buf: [256]u8 = undefined;
        const wire = protocol.encodeAppendPairs(pairs, &buf);
        const dup = try std.testing.allocator.dupe(u8, wire.bytes);
        _ = self.sched.handle(.{ .conn = conn, .request_id = 11, .session_id = sid, .body = .{
            .session_append = .{ .n = wire.n, .bytes = dup },
        } }, self.sink);
    }

    /// Send one frame and run it to its reply. Returns the positions, or
    /// the error event.
    fn appendDrain(
        self: *Fixture,
        conn: u32,
        sid: u32,
        pairs: []const protocol.Pair,
    ) !OutEvent.Body {
        try self.append(conn, sid, pairs);
        self.drain();
        const e = self.sink.last();
        try std.testing.expectEqual(@as(u32, 11), e.ev.request_id);
        return e.ev.body;
    }
};

fn tokenIds(events: []const FakeSink.Seen, sid: u32) [16]u32 {
    var out: [16]u32 = .{0xFFFF_FFFF} ** 16;
    var n: usize = 0;
    for (events) |e| {
        if (e.ev.session_id != sid) continue;
        if (e.ev.body != .token) continue;

        out[n] = e.ev.body.token.token_id;
        n += 1;
    }

    return out;
}

fn countTokens(events: []const FakeSink.Seen, sid: u32) usize {
    var n: usize = 0;
    for (events) |e| {
        if (e.ev.session_id == sid and e.ev.body == .token) {
            n += 1;
        }
    }

    return n;
}

test "composed tick: a waiting turn's chunk rides the decode forward call" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    const sa = try fx.openSession(1, "a");
    const sb = try fx.openSession(2, "b");
    try fx.generate(1, sa, "x", 6);
    _ = fx.sched.tick(fx.sink); // a: turn prefilled + first token; decoding
    try fx.generate(2, sb, "xxxx", 6); // b's 4-token turn arrives mid-decode

    // One tick advances BOTH: [chunk row (no logits) | decode row (logits)] in one engine call.
    // The chunk width is max_batch - decode rows = 1.
    const rows_before = fx.runner.rows_log_len;
    const a_tokens_before = countTokens(fx.sink.seen(), sa);
    _ = fx.sched.tick(fx.sink);
    const rows = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    try std.testing.expectEqual(@as(u32, 1), rows[0].slot); // b's engine slot
    try std.testing.expectEqual(false, rows[0].want_logits);
    try std.testing.expectEqual(@as(u32, 0), rows[1].slot); // a's decode row
    try std.testing.expectEqual(true, rows[1].want_logits);
    // a's stream never paused for b's prefill.
    try std.testing.expectEqual(a_tokens_before + 1, countTokens(fx.sink.seen(), sa));

    fx.drain();
    // Sharing the call perturbed nobody: both streams walk the fake's chain to
    // their budget (6 stays clear of the fixture's eos_id 7).
    const toks_a = tokenIds(fx.sink.seen(), sa);
    const toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks_a[0..6]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks_b[0..6]);
}

test "decode at max_batch cannot starve a waiting turn: floor of one chunk row" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, 1); // max_batch 1 < 2 sessions
    defer fx.tearDown();

    const sa = try fx.openSession(1, "a");
    const sb = try fx.openSession(2, "b");
    try fx.generate(1, sa, "x", 6);
    _ = fx.sched.tick(fx.sink); // a decoding: fills the whole max_batch
    try fx.generate(2, sb, "xxx", 6);

    // The tick still holds one chunk row for b beside a's decode row.
    const rows_before = fx.runner.rows_log_len;
    _ = fx.sched.tick(fx.sink);
    const rows = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    try std.testing.expectEqual(@as(u32, 1), rows[0].slot);
    try std.testing.expectEqual(@as(u32, 0), rows[1].slot);

    fx.drain();
    const toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks_b[0..6]);
}

test "explicit prefill_chunk widens a chunk past max_batch, apart from the decode width" {
    // Six slots give the runner room. The chunk of 4 rows is TWICE the decode width, so it is the
    // explicit budget and not the spare rows.
    var fx = try Fixture.setUp(std.testing.allocator, 6, 64, 2);
    defer fx.tearDown();

    fx.sched.cfg.prefill = .{ .fixed = 4 }; // the --prefill-chunk flag

    const sid = try fx.openSession(1, "a"); // 1-token system prompt prefilled on open
    try fx.generate(1, sid, "xxxxx", 6); // a 5-token prompt, one feeding turn

    // First tick: a single 4-row chunk (min(remaining 5, budget 4)), wider than
    // max_batch 2. No row wants logits (the chunk is non-final).
    const before0 = fx.runner.rows_log_len;
    _ = fx.sched.tick(fx.sink);
    const c0 = fx.runner.rowsSeen()[before0..];
    try std.testing.expectEqual(@as(usize, 4), c0.len);
    for (c0, 0..) |r, i| {
        try std.testing.expectEqual(c0[0].slot, r.slot); // one chunk, one slot
        try std.testing.expectEqual(c0[0].pos + @as(u32, @intCast(i)), r.pos); // contiguous
        try std.testing.expectEqual(false, r.want_logits);
    }

    // Second tick: the final 1-row chunk (remaining 1). It wants logits, and the
    // first token is sampled.
    const before1 = fx.runner.rows_log_len;
    _ = fx.sched.tick(fx.sink);
    const c1 = fx.runner.rowsSeen()[before1..];
    try std.testing.expectEqual(@as(usize, 1), c1.len);
    try std.testing.expectEqual(c0[0].pos + 4, c1[0].pos);
    try std.testing.expectEqual(true, c1[0].want_logits);

    fx.drain();
    // 'x' % 8 = 0, so the chain samples 1,2,3,4,5,6 (clear of eos 7). The wide chunk
    // leaves the compact-logits readback and the stream of the default.
    const toks = tokenIds(fx.sink.seen(), sid);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks[0..6]);
}

test "lifecycle: open with system prompt, generate to max_tokens, carry feeds the next turn" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "abc"); // system prompt: 3 tokens via chunk size 2
    try std.testing.expectEqual(@as(u32, 3), fx.pool.posOf(0));

    try fx.generate(1, sid, "x", 3); // 'x'%8=0 -> sampled chain 1,2,3
    fx.drain();

    const toks = tokenIds(fx.sink.seen(), sid);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, toks[0..3]);
    const done = fx.sink.last();
    try std.testing.expectEqual(OutEvent.Body.done, std.meta.activeTag(done.ev.body));
    try std.testing.expectEqual(protocol.FinishReason.max_tokens, done.ev.body.done.finish_reason);
    try std.testing.expectEqual(@as(u32, 3), done.ev.body.done.gen_tokens);

    // Budget stop: the engine sampled token 3 and never stepped it. The next
    // turn must lead with it, which is the pending-token rule.
    const rows_before = fx.runner.rows_log_len;
    try fx.generate(1, sid, "z", 1); // 'z'%8=2
    fx.drain();
    const new_rows = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(@as(u32, 3), new_rows[0].token); // the carry
    try std.testing.expectEqual(@as(u32, 2), new_rows[1].token); // then the turn
}

test "EOS stops the turn after emitting the EOS token, with no carry" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    try fx.generate(1, sid, "\x06", 10); // token 6 -> first sampled is 7 == eos
    fx.drain();

    const toks = tokenIds(fx.sink.seen(), sid);
    try std.testing.expectEqual(@as(u32, 7), toks[0]);
    const done = fx.sink.last();
    try std.testing.expectEqual(protocol.FinishReason.eos, done.ev.body.done.finish_reason);
    try std.testing.expectEqual(@as(u32, 1), done.ev.body.done.gen_tokens);

    // The engine never stepped EOS, so the next turn carries nothing.
    const rows_before = fx.runner.rows_log_len;
    try fx.generate(1, sid, "z", 1); // 'z'%8=2
    fx.drain();
    try std.testing.expectEqual(@as(u32, 2), fx.runner.rowsSeen()[rows_before].token);
}

test "backpressure: a full queue parks one event and pauses only that session" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    const sa = try fx.openSession(1, "a");
    const sb = try fx.openSession(2, "a");
    try fx.generate(1, sa, "x", 4);
    try fx.generate(2, sb, "x", 4);
    _ = fx.sched.tick(fx.sink); // prefill a
    _ = fx.sched.tick(fx.sink); // prefill b, so both decode with token 1 emitted

    fx.sink.fail_conn = 2;
    _ = fx.sched.tick(fx.sink); // decode: a gets token 2; b's parks
    var toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqual(@as(u32, 0xFFFF_FFFF), toks_b[1]); // b's second token not delivered

    _ = fx.sched.tick(fx.sink); // b is paused: only a advances
    const toks_a = tokenIds(fx.sink.seen(), sa);
    try std.testing.expectEqual(@as(u32, 3), toks_a[2]);

    fx.sink.fail_conn = null;
    fx.drain(); // parked token flushes first, b rejoins and catches up
    toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, toks_b[0..4]);
    const toks_a2 = tokenIds(fx.sink.seen(), sa);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, toks_a2[0..4]);
}

test "backpressure: control replies on a full queue disconnect that client" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    const seen_before = fx.sink.n;
    try std.testing.expectEqual(@as(?u32, null), fx.sink.disconnected);

    // Connection 3 never drains, so every reply to it fails to emit. A control
    // reply has no session cell to park in, so the disconnect is the bound.
    fx.sink.fail_conn = 3;
    var sent: u32 = 0;
    while (sent < 4) : (sent += 1) {
        _ = fx.sched.handle(.{
            .conn = 3,
            .request_id = 100 + sent,
            .session_id = 0,
            .body = .ping,
        }, fx.sink);
    }

    try std.testing.expectEqual(@as(?u32, 3), fx.sink.disconnected);
    // The scheduler keeps no control-reply backlog, so four refused replies
    // leave the sink exactly where the session left it.
    try std.testing.expectEqual(seen_before, fx.sink.n);

    // The refusal covers one connection. Session 1 keeps streaming.
    fx.sink.fail_conn = null;
    try fx.generate(1, sid, "x", 2);
    fx.drain();
    const toks = tokenIds(fx.sink.seen(), sid);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, toks[0..2]);
    try std.testing.expectEqual(@as(?u32, 3), fx.sink.disconnected);
}

test "request guards: exhausted pool, unknown session, busy session" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");

    // Pool of 1 is full.
    const dup = try std.testing.allocator.dupe(u8, "p");
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 2, .session_id = 0, .body = .{
        .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
    } }, fx.sink);
    try std.testing.expectEqual(
        protocol.ErrorCode.sessions_exhausted,
        fx.sink.last().ev.body.err.code,
    );

    // Unknown session id. A session of another connection is equally unknown.
    try fx.generate(1, 999, "x", 1);
    try std.testing.expectEqual(
        protocol.ErrorCode.unknown_session,
        fx.sink.last().ev.body.err.code,
    );
    try fx.generate(2, sid, "x", 1);
    try std.testing.expectEqual(
        protocol.ErrorCode.unknown_session,
        fx.sink.last().ev.body.err.code,
    );

    // A turn in flight rejects a second generate.
    try fx.generate(1, sid, "x", 4);
    try fx.generate(1, sid, "y", 4);
    try std.testing.expectEqual(protocol.ErrorCode.session_busy, fx.sink.last().ev.body.err.code);
    fx.drain();
}

test "context_full finishes the turn when the KV runs out" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 6, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "ab"); // pos 2
    try fx.generate(1, sid, "x", 100); // turn: 1 token -> pos 3; decode until full
    fx.drain();

    const done = fx.sink.last();
    try std.testing.expectEqual(OutEvent.Body.done, std.meta.activeTag(done.ev.body));
    try std.testing.expectEqual(
        protocol.FinishReason.context_full,
        done.ev.body.done.finish_reason,
    );
    try std.testing.expectEqual(@as(u32, 6), done.ev.body.done.kv_pos);

    // And the next turn cannot fit: turn_too_long, session intact.
    try fx.generate(1, sid, "abc", 4);
    try std.testing.expectEqual(protocol.ErrorCode.turn_too_long, fx.sink.last().ev.body.err.code);
}

test "shutdown cancels in-flight turns and acks" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    fx.sched.cfg.allow_shutdown = true; // the operator opted in

    const sid = try fx.openSession(1, "a");
    try fx.generate(1, sid, "x", 50);
    _ = fx.sched.tick(fx.sink); // mid-turn

    const r = fx.sched.handle(
        .{ .conn = 1, .request_id = 77, .session_id = 0, .body = .shutdown },
        fx.sink,
    );
    try std.testing.expectEqual(HandleResult.shutdown, r);

    const events = fx.sink.seen();
    const done = events[events.len - 2];
    try std.testing.expectEqual(protocol.FinishReason.cancelled, done.ev.body.done.finish_reason);
    try std.testing.expectEqual(
        OutEvent.Body.shutdown_ack,
        std.meta.activeTag(events[events.len - 1].ev.body),
    );
}

test "shutdown from the wire is refused unless allowed" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    try std.testing.expect(!fx.sched.cfg.allow_shutdown);

    const sid = try fx.openSession(1, "a");
    try fx.generate(1, sid, "x", 3);
    _ = fx.sched.tick(fx.sink); // mid-turn

    const refused = fx.sched.handle(
        .{ .conn = 1, .request_id = 77, .session_id = 0, .body = .shutdown },
        fx.sink,
    );
    try std.testing.expectEqual(HandleResult.ok, refused);
    const e = fx.sink.last();
    try std.testing.expectEqual(@as(u32, 77), e.ev.request_id);
    try std.testing.expectEqual(protocol.ErrorCode.forbidden, e.ev.body.err.code);

    // The refusal cancels nothing. The turn runs to its budget, and the
    // scheduler answers the next request.
    fx.drain();
    try std.testing.expectEqual(
        protocol.FinishReason.max_tokens,
        fx.sink.last().ev.body.done.finish_reason,
    );
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 78, .session_id = 0, .body = .ping }, fx.sink);
    try std.testing.expectEqual(OutEvent.Body.pong, std.meta.activeTag(fx.sink.last().ev.body));

    fx.sched.cfg.allow_shutdown = true;
    const r = fx.sched.handle(
        .{ .conn = 1, .request_id = 79, .session_id = 0, .body = .shutdown },
        fx.sink,
    );
    try std.testing.expectEqual(HandleResult.shutdown, r);
    try std.testing.expectEqual(
        OutEvent.Body.shutdown_ack,
        std.meta.activeTag(fx.sink.last().ev.body),
    );
}

test "a shutdown from the process owner is honored without the flag" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    try std.testing.expect(!fx.sched.cfg.allow_shutdown);

    const r = fx.sched.handle(
        .{ .conn = 0, .request_id = 5, .session_id = 0, .body = .shutdown },
        fx.sink,
    );
    try std.testing.expectEqual(HandleResult.shutdown, r);
    try std.testing.expectEqual(
        OutEvent.Body.shutdown_ack,
        std.meta.activeTag(fx.sink.last().ev.body),
    );
}

test "a cancel after a natural stop keeps the natural finish reason" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    try fx.generate(1, sid, "x", 1);
    // One tick feeds the turn in one final chunk and samples token 1, which fills the budget.
    // The tick sets the stop, and the done waits for the finalize of the next tick.
    _ = fx.sched.tick(fx.sink);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(SessionPhase.decode, entry.state);
    try std.testing.expectEqual(@as(?protocol.FinishReason, .max_tokens), entry.stop);
    for (fx.sink.seen()) |e| {
        try std.testing.expect(e.ev.body != .done);
    }

    const seen_before = fx.sink.n;
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 3, .session_id = sid, .body = .cancel },
        fx.sink,
    );
    try std.testing.expectEqual(seen_before, fx.sink.n);
    _ = fx.sched.tick(fx.sink);

    const done = fx.sink.last();
    try std.testing.expectEqual(OutEvent.Body.done, std.meta.activeTag(done.ev.body));
    try std.testing.expectEqual(protocol.FinishReason.max_tokens, done.ev.body.done.finish_reason);
    for (fx.sink.seen()) |e| {
        if (e.ev.body == .done) {
            try std.testing.expect(e.ev.body.done.finish_reason != .cancelled);
        }
    }
}

test "EOS at the token cap finishes the turn with eos, not with max_tokens" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    const from = fx.sink.n;
    // The prompt byte 6 is the token 6, so the first sampled id is 7, the end-of-sequence id.
    try fx.generate(1, sid, "\x06", 1);
    fx.drain();

    const seen = fx.sink.seen()[from..];
    try std.testing.expectEqual(@as(usize, 2), seen.len);
    try std.testing.expectEqual(@as(u32, 7), seen[0].ev.body.token.token_id);
    const done = seen[1].ev.body.done;
    try std.testing.expectEqual(protocol.FinishReason.eos, done.finish_reason);
    try std.testing.expectEqual(@as(u32, 1), done.gen_tokens);
    try std.testing.expectEqual(@as(?u32, null), fx.sched.entries[0].carry);
}

test "a cancel on an idle session emits nothing" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    const seen_before = fx.sink.seen().len;
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 3, .session_id = sid, .body = .cancel },
        fx.sink,
    );
    try std.testing.expectEqual(seen_before, fx.sink.seen().len);
}

test "a connection past its session share gets sessions_exhausted" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    // The fixture passes a zero share, and init resolves it to the table size.
    try std.testing.expectEqual(@as(u16, 2), fx.sched.cfg.sessions_per_connection);
    fx.sched.cfg.sessions_per_connection = 1;

    const sa = try fx.openSession(1, "a");

    // Conn 1 holds its share, so the server refuses its second open while a
    // entry stays free.
    const dup = try std.testing.allocator.dupe(u8, "b");
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 2, .session_id = 0, .body = .{
        .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
    } }, fx.sink);
    const open_refused = fx.sink.last().ev.body.err;
    try std.testing.expectEqual(protocol.ErrorCode.sessions_exhausted, open_refused.code);
    try std.testing.expectEqualStrings(
        "session share of this connection is full",
        open_refused.detail[0..open_refused.detail_len],
    );

    // A clone counts against the same share.
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 3, .session_id = sa, .body = .{
        .session_clone = .{ .sampling = protocol.Sampling.greedy },
    } }, fx.sink);
    const clone_refused = fx.sink.last().ev.body.err;
    try std.testing.expectEqual(protocol.ErrorCode.sessions_exhausted, clone_refused.code);
    try std.testing.expectEqualStrings(
        "session share of this connection is full",
        clone_refused.detail[0..clone_refused.detail_len],
    );

    // Conn 2 still opens into the free entry. Past that, the pool itself is full.
    _ = try fx.openSession(2, "b");
    const dup3 = try std.testing.allocator.dupe(u8, "c");
    _ = fx.sched.handle(.{ .conn = 3, .request_id = 4, .session_id = 0, .body = .{
        .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup3 },
    } }, fx.sink);
    const pool_refused = fx.sink.last().ev.body.err;
    try std.testing.expectEqual(protocol.ErrorCode.sessions_exhausted, pool_refused.code);
    try std.testing.expectEqualStrings(
        "no free session",
        pool_refused.detail[0..pool_refused.detail_len],
    );
}

test "session_clone forks between turns: shared context, fresh id, guards hold" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    const sa = try fx.openSession(1, "ab"); // system prompt: pos 2
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 11, .session_id = sa, .body = .{
        .session_clone = .{ .sampling = protocol.Sampling.greedy },
    } }, fx.sink);
    const opened = fx.sink.last();
    try std.testing.expectEqual(OutEvent.Body.session_opened, std.meta.activeTag(opened.ev.body));
    const sb = opened.ev.session_id;
    try std.testing.expect(sb != sa);
    try std.testing.expectEqual(@as(u32, 2), opened.ev.body.session_opened.kv_pos);

    // Same system prompt, same turn, greedy: the fork's stream equals the
    // source's.
    try fx.generate(1, sa, "x", 3);
    fx.drain();
    try fx.generate(1, sb, "x", 3);
    fx.drain();
    const toks_a = tokenIds(fx.sink.seen(), sa);
    const toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqualSlices(u32, toks_a[0..3], toks_b[0..3]);

    // Guards: a full pool refuses, a mid-turn source is busy, and another
    // connection's session is unknown.
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 12, .session_id = sa, .body = .{
        .session_clone = .{ .sampling = protocol.Sampling.greedy },
    } }, fx.sink);
    try std.testing.expectEqual(
        protocol.ErrorCode.sessions_exhausted,
        fx.sink.last().ev.body.err.code,
    );

    try fx.generate(1, sa, "x", 3); // turn in flight on the source
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 13, .session_id = sa, .body = .{
        .session_clone = .{ .sampling = protocol.Sampling.greedy },
    } }, fx.sink);
    try std.testing.expectEqual(protocol.ErrorCode.session_busy, fx.sink.last().ev.body.err.code);
    fx.drain();

    _ = fx.sched.handle(.{ .conn = 2, .request_id = 14, .session_id = sa, .body = .{
        .session_clone = .{ .sampling = protocol.Sampling.greedy },
    } }, fx.sink);
    try std.testing.expectEqual(
        protocol.ErrorCode.unknown_session,
        fx.sink.last().ev.body.err.code,
    );
}

test "close releases the session entry for reuse, and a conn sweep closes everything it owned" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 5, .session_id = sid, .body = .session_close },
        fx.sink,
    );
    try std.testing.expectEqual(
        OutEvent.Body.session_closed,
        std.meta.activeTag(fx.sink.last().ev.body),
    );

    const sid2 = try fx.openSession(1, "a"); // entry reused, fresh id
    try std.testing.expect(sid2 != sid);
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 0, .session_id = 0, .body = .conn_closed },
        fx.sink,
    );
    try fx.generate(1, sid2, "x", 1);
    try std.testing.expectEqual(
        protocol.ErrorCode.unknown_session,
        fx.sink.last().ev.body.err.code,
    );
    fx.drain();
}

test "over-subscription: three logical sessions over two VRAM slots stream correctly" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    var runner: TestRunner = .{
        .n_slots = 2, // two physical KV slots
        .max_seq = 64,
        .logits = try arena.alloc(f32, 2 * test_vocab),
    };
    @memset(runner.logits, 0);
    var sessions = try Sessions(TestRunner).init(arena, &runner);
    var pool = try ResPool.init(arena, gpa, &sessions, 3, null); // three logical sessions
    defer pool.deinit();

    var codec = FakeCodec.initFake();
    var sched = try ResScheduler.init(arena, gpa, std.testing.io, &pool, &codec, .{
        .max_sessions = 3,
        .max_batch = 2, // <= physical slots
        .max_seq = 64,
        .max_step_rows = 2, // == physical slots == runner.maxStepRows()
        .model_name = "fake",
        .vocab_size = test_vocab,
        .eos_id = 7,
    });
    var sink: FakeSink = .{};

    // Open three system-prompt sessions. The third cannot fit two slots
    // without a swap.
    var sids: [3]u32 = undefined;
    for (0..3) |k| {
        const dup = try gpa.dupe(u8, "a");
        _ = sched.handle(.{ .conn = @intCast(k + 1), .request_id = 1, .session_id = 0, .body = .{
            .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
        } }, &sink);
        var open_iters: u32 = 0;
        while (sched.tick(&sink) != .idle) {
            open_iters += 1;
            try std.testing.expect(open_iters <= 200);
        }

        sids[k] = sink.last().ev.session_id;
    }

    try std.testing.expect(runner.evicts >= 1);

    // Three decoders over two slots: the scheduler round-robins and swaps.
    // Each stream is byte 'x' (=0) -> 1,2,3,4,5,6.
    for (0..3) |k| {
        const dup = try gpa.dupe(u8, "x");
        _ = sched.handle(
            .{ .conn = @intCast(k + 1), .request_id = 9, .session_id = sids[k], .body = .{
                .generate = .{ .max_tokens = 6, .sampling = null, .text = dup },
            } },
            &sink,
        );
    }

    // A swap-scheduling livelock (a prefill no tick can admit) hangs here. The
    // bound turns that regression into a loud failure.
    var iters: u32 = 0;
    while (sched.tick(&sink) != .idle) {
        iters += 1;
        try std.testing.expect(iters <= 200);
    }

    for (sids) |sid| {
        const toks = tokenIds(sink.seen(), sid);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks[0..6]);
    }

    try std.testing.expect(runner.restores >= 1); // an evicted session decoded again

    // Close each session so the residency frees its gpa-owned eviction blob
    // (in production releaseSlot does this on session_close/conn_closed).
    for (sids, 0..) |sid, k| {
        _ = sched.handle(.{
            .conn = @intCast(k + 1),
            .request_id = 0,
            .session_id = sid,
            .body = .session_close,
        }, &sink);
    }
}

test "over-subscription: slots decoupled from the decode width, 12 logical / 6 slots / batch 2" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    // Six physical slots with a decode width of two: breadth past the batch.
    // Twelve logical sessions over-subscribe the six slots, so the run also swaps.
    var runner: TestRunner = .{
        .n_slots = 6,
        .max_seq = 64,
        .logits = try arena.alloc(f32, 6 * test_vocab),
    };
    @memset(runner.logits, 0);
    var sessions = try Sessions(TestRunner).init(arena, &runner);
    var pool = try ResPool.init(arena, gpa, &sessions, 12, null);
    defer pool.deinit();

    var codec = FakeCodec.initFake();
    var sched = try ResScheduler.init(arena, gpa, std.testing.io, &pool, &codec, .{
        .max_sessions = 12,
        .max_batch = 2, // << physical slots: width is stepped, breadth is held
        .max_seq = 64,
        .max_step_rows = 6,
        .model_name = "fake",
        .vocab_size = test_vocab,
        .eos_id = 7,
    });
    var sink: FakeSink = .{};

    var sids: [12]u32 = undefined;
    for (0..12) |k| {
        const dup = try gpa.dupe(u8, "a");
        _ = sched.handle(.{ .conn = @intCast(k + 1), .request_id = 1, .session_id = 0, .body = .{
            .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
        } }, &sink);
        var open_iters: u32 = 0;
        while (sched.tick(&sink) != .idle) {
            open_iters += 1;
            try std.testing.expect(open_iters <= 400);
        }

        sids[k] = sink.last().ev.session_id;
    }

    // Twelve opens over six slots forced at least six evictions.
    try std.testing.expect(runner.evicts >= 6);

    // All twelve generate at once. Two rows decode per tick while six
    // sessions sit resident, and the other six rotate in from RAM.
    for (0..12) |k| {
        const dup = try gpa.dupe(u8, "x");
        _ = sched.handle(
            .{ .conn = @intCast(k + 1), .request_id = 9, .session_id = sids[k], .body = .{
                .generate = .{ .max_tokens = 6, .sampling = null, .text = dup },
            } },
            &sink,
        );
    }

    var iters: u32 = 0;
    while (sched.tick(&sink) != .idle) {
        iters += 1;
        try std.testing.expect(iters <= 400);
    }

    for (sids) |sid| {
        const toks = tokenIds(sink.seen(), sid);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks[0..6]);
    }

    try std.testing.expect(runner.restores >= 6); // the evicted six decoded again

    for (sids, 0..) |sid, k| {
        _ = sched.handle(.{
            .conn = @intCast(k + 1),
            .request_id = 0,
            .session_id = sid,
            .body = .session_close,
        }, &sink);
    }
}

test "over-subscription: a ram budget of one blob parks a prefill, and the loop converges" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    var runner: TestRunner = .{
        .n_slots = 2,
        .max_seq = 64,
        .logits = try arena.alloc(f32, 2 * test_vocab),
    };
    @memset(runner.logits, 0);
    var sessions = try Sessions(TestRunner).init(arena, &runner);
    // The fake adds one u32 per evicted slot, so this budget holds exactly
    // one blob. A rotation transiently needs two: `ensureResident` adds the
    // victim's new blob before the restored session's blob is dropped. So this
    // budget holds no swap at all, and a parked turn waits for a free slot.
    var pool = try ResPool.init(arena, gpa, &sessions, 3, @sizeOf(u32));
    defer pool.deinit();

    var codec = FakeCodec.initFake();
    var sched = try ResScheduler.init(arena, gpa, std.testing.io, &pool, &codec, .{
        .max_sessions = 3,
        .max_batch = 2,
        .max_seq = 64,
        .max_step_rows = 2,
        .model_name = "fake",
        .vocab_size = test_vocab,
        .eos_id = 7,
    });
    var sink: FakeSink = .{};

    var sids: [3]u32 = undefined;
    for (0..3) |k| {
        const dup = try gpa.dupe(u8, "a");
        _ = sched.handle(.{ .conn = @intCast(k + 1), .request_id = 1, .session_id = 0, .body = .{
            .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
        } }, &sink);
        var open_iters: u32 = 0;
        while (sched.tick(&sink) != .idle) {
            open_iters += 1;
            try std.testing.expect(open_iters <= 200);
        }

        sids[k] = sink.last().ev.session_id;
    }

    // The third session filled the whole budget on the first session's blob.
    try std.testing.expectEqual(@as(u32, 1), runner.evicts);

    // The evicted session asks last. No tick can admit its prefill. The swap holds the blob of the
    // victim beside its own, and the budget refuses that.
    for ([_]usize{ 1, 2, 0 }) |k| {
        const dup = try gpa.dupe(u8, "x");
        _ = sched.handle(
            .{ .conn = @intCast(k + 1), .request_id = 9, .session_id = sids[k], .body = .{
                .generate = .{ .max_tokens = 6, .sampling = null, .text = dup },
            } },
            &sink,
        );
    }

    var iters: u32 = 0;
    while (iters < 40) : (iters += 1) {
        _ = sched.tick(&sink);
    }

    // The park is counted, so the operator reads residency pressure as a
    // number and not only as a stalled stream.
    try std.testing.expect(sched.parks >= 1);

    // The refused eviction never reached the device, and it never disturbed the
    // two streams that still run.
    try std.testing.expectEqual(@as(u32, 1), runner.evicts);
    for ([_]usize{ 1, 2 }) |k| {
        const toks = tokenIds(sink.seen(), sids[k]);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks[0..6]);
    }

    try std.testing.expectEqual(@as(usize, 0), countTokens(sink.seen(), sids[0]));

    // Closing the two finished sessions frees their slots, so the parked turn
    // needs no eviction at all. The tick loop then drains.
    for ([_]usize{ 1, 2 }) |k| {
        _ = sched.handle(.{
            .conn = @intCast(k + 1),
            .request_id = 0,
            .session_id = sids[k],
            .body = .session_close,
        }, &sink);
    }

    iters = 0;
    while (sched.tick(&sink) != .idle) {
        iters += 1;
        try std.testing.expect(iters <= 200);
    }

    const toks = tokenIds(sink.seen(), sids[0]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks[0..6]);
    try std.testing.expect(runner.restores >= 1); // the parked session came back from RAM

    _ = sched.handle(
        .{ .conn = 1, .request_id = 0, .session_id = sids[0], .body = .session_close },
        &sink,
    );
}

test "prefix cache: a repeated system prompt restores with zero engine rows" {
    var fx = try Fixture.setUpPrefix(std.testing.allocator, 2, 64, null, 1 << 20);
    defer fx.tearDown();

    // The cold open offers the prefix. The fixture pins admission times that the gate admits.
    const sa = try fx.openSession(1, "abcdefab");
    try std.testing.expectEqual(@as(u64, 0), fx.sched.prefix.hits);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.misses);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.inserts);
    try std.testing.expectEqual(@as(u32, 1), fx.runner.evicts); // the insert's snapshot copy

    // The second open's whole system prompt is cached: the open completes
    // with no engine rows, and the runner restore reinstates the slot's state.
    const rows_before = fx.runner.rows_log_len;
    const sb = try fx.openSession(2, "abcdefab");
    try std.testing.expectEqual(rows_before, fx.runner.rows_log_len);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.hits);
    try std.testing.expectEqual(@as(u64, 0), fx.sched.prefix.disk_hits); // the RAM tier served it
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.cache.?.entries.items[0].hits);
    try std.testing.expectEqual(@as(u32, 1), fx.runner.restores);
    try std.testing.expectEqual(@as(u32, 8), fx.pool.posOf(1)); // second engine session
    try std.testing.expectEqual(fx.runner.markers[0], fx.runner.markers[1]); // restored KV marker

    // Both sessions decode identically from the shared prefix.
    try fx.generate(1, sa, "x", 3);
    try fx.generate(2, sb, "x", 3);
    fx.drain();
    const toks_a = tokenIds(fx.sink.seen(), sa);
    const toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqualSlices(u32, toks_a[0..3], toks_b[0..3]);
}

test "prefix admission: pinned times decide the gate both ways" {
    var fx = try Fixture.setUpPrefix(std.testing.allocator, 4, 64, null, 1 << 20);
    defer fx.tearDown();

    // Refused times: the cold open offers the prefix, the gate refuses it, and a repeat misses.
    fx.sched.prefix.pinned_times = admit_times_fail;
    _ = try fx.openSession(1, "abcdefab");
    const rows_refused = fx.runner.rows_log_len;
    _ = try fx.openSession(2, "abcdefab");
    try std.testing.expectEqual(@as(u64, 0), fx.sched.prefix.inserts);
    try std.testing.expectEqual(@as(u64, 0), fx.sched.prefix.hits);
    try std.testing.expectEqual(@as(u64, 2), fx.sched.prefix.misses);
    try std.testing.expectEqual(rows_refused + 8, fx.runner.rows_log_len);

    // Admitted times: the next cold open inserts, and a repeat restores with no forward row.
    fx.sched.prefix.pinned_times = admit_times_pass;
    _ = try fx.openSession(3, "abcdefab");
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.inserts);
    const rows_admitted = fx.runner.rows_log_len;
    _ = try fx.openSession(4, "abcdefab");
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.hits);
    try std.testing.expectEqual(rows_admitted, fx.runner.rows_log_len);
}

test "prefix cache: a generate turn restores the longest prefix and feeds only the suffix" {
    var fx = try Fixture.setUpPrefix(std.testing.allocator, 2, 64, null, 1 << 20);
    defer fx.tearDown();

    // The ephemeral-session shape: open with no system prompt, one generate
    // turn.
    const sa = try fx.openSession(1, "");
    try fx.generate(1, sa, "xxxxx", 2);
    fx.drain();
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.misses);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.inserts);

    // A continuation turn starts past position zero, so it never asks the cache.
    try fx.generate(1, sa, "z", 1);
    fx.drain();
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.misses);
    try std.testing.expectEqual(@as(u64, 0), fx.sched.prefix.hits);

    // A longer prompt sharing the 5-token prefix restores it and prefills the
    // 2-token suffix only: the fed rows start at position 5.
    const sb = try fx.openSession(2, "");
    const rows_before = fx.runner.rows_log_len;
    try fx.generate(2, sb, "xxxxxxx", 2);
    fx.drain();
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.hits);
    const fed = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(@as(u32, 5), fed[0].pos);
    try std.testing.expectEqual(@as(u32, 6), fed[1].pos);
    const toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, toks_b[0..2]);

    // An exact duplicate is a FULL hit (the regenerate case): the engine restores the rows and
    // samples the first token from the stored logits. No forward row, and the same stream.
    _ = fx.sched.handle(
        .{ .conn = 2, .request_id = 3, .session_id = sb, .body = .session_close },
        fx.sink,
    );
    const sc = try fx.openSession(3, "");
    const rows_full = fx.runner.rows_log_len;
    const restores_before = fx.runner.restores;
    try fx.generate(3, sc, "xxxxx", 1);
    fx.drain();
    try std.testing.expectEqual(@as(u64, 2), fx.sched.prefix.hits);
    try std.testing.expectEqual(rows_full, fx.runner.rows_log_len);
    try std.testing.expectEqual(restores_before + 1, fx.runner.restores);
    const toks_c = tokenIds(fx.sink.seen(), sc);
    try std.testing.expectEqual(@as(u32, 1), toks_c[0]); // the cold first token
}

test "prefix cache: a stateless multi-turn ladder climbs the cached boundaries" {
    // The OpenAI frontend maps each chat request to an ephemeral session: open(""), one generate
    // with the whole rendered history, close. So each request restores the entry of the previous
    // request and prefills only its new turn. The eligibility predicate is position zero, and the
    // cache keys on token ids alone.
    var fx = try Fixture.setUpPrefix(std.testing.allocator, 3, 64, null, 1 << 20);
    defer fx.tearDown();

    // Request 1: [1,2,3]. Cold, completes, inserts the 3-token boundary.
    const s1 = try fx.openSession(1, "");
    try fx.generate(1, s1, "abc", 1);
    fx.drain();
    try std.testing.expectEqual(@as(u64, 0), fx.sched.prefix.hits);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.misses);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.inserts);
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 9, .session_id = s1, .body = .session_close },
        fx.sink,
    );

    // Request 2 extends request 1: [1,2,3,4,5]. The fed rows start at position 3,
    // and its own completion adds the 5-token boundary.
    const s2 = try fx.openSession(2, "");
    var rows_before = fx.runner.rows_log_len;
    try fx.generate(2, s2, "abcde", 1);
    fx.drain();
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.hits);
    try std.testing.expectEqual(@as(u64, 2), fx.sched.prefix.inserts);
    var fed = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(@as(u32, 3), fed[0].pos);
    try std.testing.expectEqual(@as(u32, 4), fed[1].pos);
    _ = fx.sched.handle(
        .{ .conn = 2, .request_id = 9, .session_id = s2, .body = .session_close },
        fx.sink,
    );

    // Request 3 extends request 2: [1,2,3,4,5,6,6]. The LONGEST boundary wins,
    // so the fed rows start at position 5 and not at 3.
    const s3 = try fx.openSession(3, "");
    rows_before = fx.runner.rows_log_len;
    try fx.generate(3, s3, "abcdeff", 1);
    fx.drain();
    try std.testing.expectEqual(@as(u64, 2), fx.sched.prefix.hits);
    try std.testing.expectEqual(@as(u64, 3), fx.sched.prefix.inserts);
    fed = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(@as(u32, 5), fed[0].pos);
    try std.testing.expectEqual(@as(u32, 6), fed[1].pos);
}

test "prefix cache: the disk tier survives a scheduler restart" {
    // Boot one inserts an opening boundary in RAM and on disk. Boot two indexes the same directory
    // at init and serves the same open with zero engine rows.
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dirbuf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dirbuf, ".zig-cache/tmp/{s}/serve-prefix", .{&tmp.sub_path});

    {
        var fx = try Fixture.setUpDisk(gpa, 2, 64, null, 1 << 20, 1 << 20, dir);
        defer fx.tearDown();

        _ = try fx.openSession(1, "abcdefab");
        try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.inserts);
        try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.disk_inserts);
    }

    {
        var fx = try Fixture.setUpDisk(gpa, 2, 64, null, 1 << 20, 1 << 20, dir);
        defer fx.tearDown();

        const rows_before = fx.runner.rows_log_len;
        const sid = try fx.openSession(1, "abcdefab");
        try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.hits);
        try std.testing.expectEqual(
            @as(u64, 1),
            fx.sched.prefix.disk_hits,
        ); // the promoted file served it
        try std.testing.expectEqual(rows_before, fx.runner.rows_log_len);
        try std.testing.expectEqual(@as(u32, 1), fx.runner.restores);

        // The restored session decodes as a session that prefilled cold.
        try fx.generate(1, sid, "x", 3);
        fx.drain();
        const toks = tokenIds(fx.sink.seen(), sid);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, toks[0..3]);
    }
}

test "prefix cache: preload warms the cache at boot" {
    var fx = try Fixture.setUpPrefix(std.testing.allocator, 2, 64, null, 1 << 20);
    defer fx.tearDown();

    // The boot warm-up: one open + close through the normal request path.
    // The prefill lands in the cache and the session leaves no entry behind.
    fx.sched.cfg.cache.preload = &.{"abcdefab"};
    fx.sched.preloadPrefixes();
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.inserts);
    for (fx.sched.entries) |*s| {
        try std.testing.expectEqual(SessionPhase.free, s.state);
    }

    // The first open of a client with the same text restores with zero engine rows.
    const rows_before = fx.runner.rows_log_len;
    _ = try fx.openSession(1, "abcdefab");
    try std.testing.expectEqual(rows_before, fx.runner.rows_log_len);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.hits);
}

test "scheduler: an engine that cannot snapshot serves without a cache" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    var runner: NoSnapRunner = .{
        .n_slots = 1,
        .max_seq = 64,
        .logits = try arena.alloc(f32, test_vocab),
    };
    @memset(runner.logits, 0);
    var sessions = try Sessions(NoSnapRunner).init(arena, &runner);
    var codec = FakeCodec.initFake();
    var sched = try NoSnapScheduler.init(arena, gpa, std.testing.io, &sessions, &codec, .{
        .max_sessions = 1,
        .max_batch = 1,
        .max_seq = 64,
        .max_step_rows = 1,
        .model_name = "fake",
        .vocab_size = test_vocab,
        .eos_id = 7,
        // A RAM budget this engine must still refuse: the gate reads the
        // capability AND the budget.
        .cache = .{ .ram_bytes = 1 << 20 },
    });
    defer sched.deinit();

    try std.testing.expect(!Sessions(NoSnapRunner).can_snapshot);
    try std.testing.expect(!sched.prefix.enabled());

    var sink: FakeSink = .{};
    const dup = try gpa.dupe(u8, "a");
    _ = sched.handle(.{ .conn = 1, .request_id = 1, .session_id = 0, .body = .{
        .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
    } }, &sink);
    var iters: u32 = 0;
    while (sched.tick(&sink) != .idle) {
        iters += 1;
        try std.testing.expect(iters <= 200);
    }

    const sid = sink.last().ev.session_id;

    // The cold path is the only path, and it streams the fake's chain.
    const dup2 = try gpa.dupe(u8, "x");
    _ = sched.handle(.{ .conn = 1, .request_id = 9, .session_id = sid, .body = .{
        .generate = .{ .max_tokens = 3, .sampling = null, .text = dup2 },
    } }, &sink);
    iters = 0;
    while (sched.tick(&sink) != .idle) {
        iters += 1;
        try std.testing.expect(iters <= 200);
    }

    const toks = tokenIds(sink.seen(), sid);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, toks[0..3]);
    try std.testing.expectEqual(@as(u64, 0), sched.prefix.hits);
    try std.testing.expectEqual(@as(u64, 0), sched.prefix.misses);
    try std.testing.expectEqual(@as(u64, 0), sched.prefix.inserts);
}

// The think tests run the fake chain with the tags of a fake vocab. `<think>` is 2, `</think>` is
// 3, and a full budget forces 3 then 0. A reader derives each stream by hand. The prompt "4" feeds
// the one token 4, so the first token is 5. It lands inside the block when the turn opens.

const Client = @import("client.zig").Client;

const think_tags: Think = .{
    .open_default = true,
    .open_id = 2,
    .close_id = 3,
    .budget = 2,
    .force = .{ 3, 0, no_token, no_token },
    .force_len = 2,
};

/// One turn of one session as the sink saw it: the token ids, the echo
/// included, their channels, and the `done`.
const TurnSeen = struct {
    ids: [16]u32 = undefined,
    chans: [16]protocol.Channel = undefined,
    n: usize = 0,
    done: @FieldType(OutEvent.Body, "done") = undefined,

    fn tokens(self: *const TurnSeen) []const u32 {
        return self.ids[0..self.n];
    }

    fn channels(self: *const TurnSeen) []const protocol.Channel {
        return self.chans[0..self.n];
    }
};

/// Run one turn to its `done`, and read it back from the sink.
fn runTurn(fx: *Fixture, conn: u32, sid: u32, prompt: []const u8, max_tokens: u32) !TurnSeen {
    const from = fx.sink.n;
    try fx.generate(conn, sid, prompt, max_tokens);
    fx.drain();
    var seen: TurnSeen = .{};
    var dones: usize = 0;
    for (fx.sink.seen()[from..]) |e| {
        if (e.ev.session_id != sid) continue;
        switch (e.ev.body) {
            .token => |t| {
                try std.testing.expectEqual(@as(usize, 0), dones);
                try std.testing.expect(seen.n < seen.ids.len);
                seen.ids[seen.n] = t.token_id;
                seen.chans[seen.n] = t.channel;
                seen.n += 1;
            },
            .done => |d| {
                seen.done = d;
                dones += 1;
            },
            else => return error.TestUnexpectedResult,
        }
    }

    try std.testing.expectEqual(@as(usize, 1), dones);
    return seen;
}

const ThinkCase = struct {
    /// The turn opens inside the block, so its first event is the echo.
    open: bool,
    prompt: []const u8,
    /// The ids the model steps before its first token: a carry, then the prompt.
    fed: []const u32,
    max_tokens: u32 = 64,
    want: []const u32,
    reason: protocol.FinishReason,
    /// The channel of each event in `want`, when the case checks them.
    chans: ?[]const protocol.Channel = null,
    /// The reasoning count of the `done`.
    reasoning: u32 = 0,
};

/// Run one turn and check the events, the finish reason, the count, and the stepped rows. The rows
/// prove the budget case. A model that steps the sampled 7 in place of the forced 3 samples 0 next.
/// The queue forces the same id, so the events alone cannot show it.
fn expectThinkTurn(fx: *Fixture, sid: u32, c: ThinkCase) !void {
    const rows_before = fx.runner.rows_log_len;
    const turn = try runTurn(fx, 1, sid, c.prompt, c.max_tokens);
    try std.testing.expectEqualSlices(u32, c.want, turn.tokens());
    try std.testing.expectEqual(c.reason, turn.done.finish_reason);
    if (c.chans) |chans| {
        try std.testing.expectEqualSlices(protocol.Channel, chans, turn.channels());
    }

    try std.testing.expectEqual(c.reasoning, turn.done.reasoning_tokens);
    var reasoning: u32 = 0;
    for (turn.channels()) |ch| {
        reasoning += @intFromBool(ch == .reasoning);
    }

    try std.testing.expectEqual(reasoning, turn.done.reasoning_tokens);
    const accepted = c.want[@intFromBool(c.open)..];
    try std.testing.expect(accepted.len >= 1);
    try std.testing.expectEqual(@as(u32, @intCast(accepted.len)), turn.done.gen_tokens);
    if (c.open) {
        try std.testing.expectEqual(think_tags.open_id, c.want[0]);
    }

    const rows = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(c.fed.len + accepted.len - 1, rows.len);
    for (rows[0..c.fed.len], c.fed) |r, id| {
        try std.testing.expectEqual(id, r.token);
    }

    for (rows[c.fed.len..], accepted[0 .. accepted.len - 1]) |r, id| {
        try std.testing.expectEqual(id, r.token);
    }
}

test "think budget: the budget forces the close, and the model steps the forced ids" {
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, think_tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");

    // 5 and 6 use the budget of two, so 3 and 0 replace the sampled 7 and 4.
    // The forced close is a close tag. The later sampled 3 is content, because the block is closed.
    try expectThinkTurn(&fx, sid, .{
        .open = true,
        .prompt = "4",
        .fed = &.{4},
        .want = &.{ 2, 5, 6, 3, 0, 1, 2, 3, 4, 5, 6, 7 },
        .reason = .eos,
        .chans = &.{
            .open_tag,
            .reasoning,
            .reasoning,
            .close_tag,
            .content,
            .content,
            .content,
            .content,
            .content,
            .content,
            .content,
            .content,
        },
        .reasoning = 2,
    });
    try std.testing.expect(!fx.sched.entries[0].think_open);
    try std.testing.expectEqual(@as(u32, 2), fx.sched.entries[0].think_n);
}

test "think budget: a close before the budget leaves the unbounded stream" {
    // The sampled 3 closes the block after one think token, so a budget of
    // five never acts. The stream equals the stream of no budget.
    for ([_]u32{ 5, 0 }) |budget| {
        var tags = think_tags;
        tags.budget = budget;
        var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, tags);
        defer fx.tearDown();

        const sid = try fx.openSession(1, "");
        // A sampled 2 inside the block is reasoning, although it equals the open id.
        try expectThinkTurn(&fx, sid, .{
            .open = true,
            .prompt = "1",
            .fed = &.{1},
            .want = &.{ 2, 2, 3, 4, 5, 6, 7 },
            .reason = .eos,
            .chans = &.{
                .open_tag,
                .reasoning,
                .close_tag,
                .content,
                .content,
                .content,
                .content,
            },
            .reasoning = 1,
        });
        try std.testing.expectEqual(@as(u32, 1), fx.sched.entries[0].think_n);
    }
}

test "think off: the turn has no echo and no forced id" {
    var tags = think_tags;
    tags.open_default = false;
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    // A closed turn: every event is content.
    try expectThinkTurn(&fx, sid, .{
        .open = false,
        .prompt = "4",
        .fed = &.{4},
        .want = &.{ 5, 6, 7 },
        .reason = .eos,
        .chans = &.{ .content, .content, .content },
    });
    try std.testing.expectEqual(@as(u32, 0), fx.sched.entries[0].think_n);
}

test "think budget: the cap lands inside the forced ids, and the next turn has none left" {
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, think_tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");

    // The forced 3 is the third token, so the cap of three leaves the forced 0 pending.
    try expectThinkTurn(&fx, sid, .{
        .open = true,
        .prompt = "4",
        .fed = &.{4},
        .max_tokens = 3,
        .want = &.{ 2, 5, 6, 3 },
        .reason = .max_tokens,
        .chans = &.{ .open_tag, .reasoning, .reasoning, .close_tag },
        .reasoning = 2,
    });
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(@as(u8, 1), entry.forced_len - entry.forced_head);

    // The next turn leads with the carried 3 and drops the pending 0. Its first
    // token is therefore the sampled 5, and its stream repeats the budget case.
    try expectThinkTurn(&fx, sid, .{
        .open = true,
        .prompt = "4",
        .fed = &.{ 3, 4 },
        .want = &.{ 2, 5, 6, 3, 0, 1, 2, 3, 4, 5, 6, 7 },
        .reason = .eos,
        .reasoning = 2,
    });
}

test "think budget: a full window stops the turn with forced ids left" {
    var fx = try Fixture.setUpThink(std.testing.allocator, 2, 0, think_tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");

    // 6 uses the budget, and the window holds no row for the forced 3.
    try expectThinkTurn(&fx, sid, .{
        .open = true,
        .prompt = "4",
        .fed = &.{4},
        .want = &.{ 2, 5, 6 },
        .reason = .context_full,
        .reasoning = 2,
    });
    try std.testing.expectEqual(@as(u32, 2), fx.sink.last().ev.body.done.kv_pos);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(@as(u8, 2), entry.forced_len - entry.forced_head);

    // A full window leaves no room for a next turn. The refusal names the
    // numbers, and it leaves the carry and the session as they were.
    try fx.generate(1, sid, "4", 4);
    const e = fx.sink.last().ev.body.err;
    try std.testing.expectEqual(protocol.ErrorCode.turn_too_long, e.code);
    try std.testing.expectEqualStrings(
        "turn of 1 tokens, 0 positions left",
        e.detail[0..e.detail_len],
    );
    try std.testing.expectEqual(SessionPhase.idle, entry.state);
    try std.testing.expectEqual(@as(?u32, null), entry.carry);
}

test "think budget: end-of-sequence inside the block ends the turn open" {
    var tags = think_tags;
    tags.budget = 10;
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    try expectThinkTurn(&fx, sid, .{
        .open = true,
        .prompt = "3",
        .fed = &.{3},
        .want = &.{ 2, 4, 5, 6, 7 },
        .reason = .eos,
        .reasoning = 4,
    });
    try std.testing.expect(fx.sched.entries[0].think_open);
    try std.testing.expectEqual(@as(u32, 4), fx.sched.entries[0].think_n);
}

test "a cancel in the prefill of an open turn ends it with the echo alone and no carry" {
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, think_tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    const from = fx.sink.n;
    try fx.generate(1, sid, "4", 64);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(SessionPhase.prefill, entry.state);
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 3, .session_id = sid, .body = .cancel },
        fx.sink,
    );
    fx.drain();

    // The echo is no generated token, so the cut turn has no last token to carry.
    const seen = fx.sink.seen()[from..];
    try std.testing.expectEqual(@as(usize, 2), seen.len);
    try std.testing.expectEqual(think_tags.open_id, seen[0].ev.body.token.token_id);
    try std.testing.expectEqual(protocol.Channel.open_tag, seen[0].ev.body.token.channel);
    const done = seen[1].ev.body.done;
    try std.testing.expectEqual(protocol.FinishReason.cancelled, done.finish_reason);
    try std.testing.expectEqual(@as(u32, 0), done.gen_tokens);
    try std.testing.expectEqual(@as(u32, 0), done.reasoning_tokens);
    try std.testing.expectEqual(@as(u32, 0), done.kv_pos);
    try std.testing.expectEqual(@as(?u32, null), entry.carry);
    try std.testing.expectEqual(SessionPhase.idle, entry.state);
}

test "think echo: a parked echo holds the prefill and skips the prefix restore" {
    var tags = think_tags;
    tags.budget = 0;
    var fx = try Fixture.setUpThink(std.testing.allocator, 64, 1 << 20, tags);
    defer fx.tearDown();

    // A cold turn caches its prompt, so the same prompt on a new session is a full hit.
    const sa = try fx.openSession(1, "");
    const a = try runTurn(&fx, 1, sa, "44444", 8);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5, 6, 7 }, a.tokens());
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.inserts);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.misses);

    // The writer of connection 2 is full, so the echo parks. The turn waits in
    // prefill, and it asks the cache nothing.
    const sb = try fx.openSession(2, "");
    fx.sink.fail_conn = 2;
    const seen_before = fx.sink.n;
    const rows_before = fx.runner.rows_log_len;
    try fx.generate(2, sb, "44444", 8);
    fx.drain();
    const entry_b = &fx.sched.entries[1];
    try std.testing.expectEqual(SessionPhase.prefill, entry_b.state);
    try std.testing.expect(entry_b.pending_event != null);
    try std.testing.expectEqual(seen_before, fx.sink.n);
    try std.testing.expectEqual(rows_before, fx.runner.rows_log_len);
    try std.testing.expectEqual(@as(u64, 0), fx.sched.prefix.hits);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.misses);

    // The drained echo leads the stream, and the prefill runs cold from
    // position zero.
    fx.sink.fail_conn = null;
    fx.drain();
    const b = tokenIds(fx.sink.seen()[seen_before..], sb);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5, 6, 7 }, b[0..4]);
    const fed = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(@as(u32, 0), fed[0].pos);
    try std.testing.expectEqual(@as(u32, 4), fed[0].token);
    try std.testing.expectEqual(@as(u64, 0), fx.sched.prefix.hits);

    // With the cell free, the same prompt takes the full hit. The model steps
    // only the decode rows of 5 and 6, and the echo still comes first.
    const sc = try fx.openSession(3, "");
    const rows_c = fx.runner.rows_log_len;
    const c = try runTurn(&fx, 3, sc, "44444", 8);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5, 6, 7 }, c.tokens());
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.hits);
    try std.testing.expectEqual(rows_c + 2, fx.runner.rows_log_len);
}

test "think config: an open turn on a model without think tags fails the request" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");

    const dup = try std.testing.allocator.dupe(u8, "4");
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 9, .session_id = sid, .body = .{
        .generate = .{ .max_tokens = 4, .sampling = null, .text = dup, .think_open = true },
    } }, fx.sink);
    const e = fx.sink.last().ev.body.err;
    try std.testing.expectEqual(protocol.ErrorCode.forbidden, e.code);
    try std.testing.expectEqualStrings("this model has no think tags", e.detail[0..e.detail_len]);

    // The refusal changed nothing, so the session answers a closed turn.
    try std.testing.expectEqual(SessionPhase.idle, fx.sched.entries[0].state);
    const t = try runTurn(&fx, 1, sid, "4", 4);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, t.tokens());
}

test "error details: the refusals name the numbers within the detail bound" {
    var buf: [detail_max]u8 = undefined;
    try std.testing.expectEqualStrings(
        "turn of 999999 tokens, 999999 positions left",
        turnTooLongDetail(&buf, 999_999, 999_999),
    );
    try std.testing.expectEqualStrings(
        "system prompt of 999999 tokens, window of 999999 positions",
        systemTooLongDetail(&buf, 999_999, 999_999),
    );
    // The turn text fits at the widest counts too, so its fallback never runs.
    const widest = turnTooLongDetail(&buf, std.math.maxInt(usize), std.math.maxInt(u32));
    try std.testing.expect(std.mem.startsWith(u8, widest, "turn of "));
    // The reference client keeps as many detail bytes as the server sends.
    try std.testing.expectEqual(@as(usize, Client.LastError.detail_cap), detail_max);

    // The open refuses a system prompt past the window with both numbers.
    var fx = try Fixture.setUp(std.testing.allocator, 1, 4, null);
    defer fx.tearDown();

    const dup = try std.testing.allocator.dupe(u8, "abcdef");
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 1, .session_id = 0, .body = .{
        .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
    } }, fx.sink);
    const e = fx.sink.last().ev.body.err;
    try std.testing.expectEqual(protocol.ErrorCode.turn_too_long, e.code);
    try std.testing.expectEqualStrings(
        "system prompt of 6 tokens, window of 4 positions",
        e.detail[0..e.detail_len],
    );
}

// The append tests. The fake codec renders a pair as its user bytes then its
// reply bytes, one id per byte. A reply written as the pieces of the sampled
// ids therefore re-encodes to those ids. That is the thinking-off case of a
// real model.

/// The token and position of each row the runner stepped for `slot`.
fn slotRows(fx: *const Fixture, slot: u32, out: [][2]u32) [][2]u32 {
    var n: usize = 0;
    for (fx.runner.rowsSeen()) |r| {
        if (r.slot != slot) continue;
        out[n] = .{ r.token, r.pos };
        n += 1;
    }

    return out[0..n];
}

fn expectAppended(body: OutEvent.Body, want: []const u32) !void {
    try std.testing.expectEqual(OutEvent.Body.appended, std.meta.activeTag(body));
    try std.testing.expectEqualSlices(u32, want, body.appended.positions());
}

fn expectRefused(body: OutEvent.Body, code: protocol.ErrorCode, detail: []const u8) !void {
    try std.testing.expectEqual(OutEvent.Body.err, std.meta.activeTag(body));
    try std.testing.expectEqual(code, body.err.code);
    try std.testing.expectEqualStrings(detail, body.err.detail[0..body.err.detail_len]);
}

test "append: kept pairs give the rows of the live turns, and the next turn matches" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    // Live: two turns that end on EOS, so no token carries. "\x04" feeds 4.
    // The chain samples 5, 6, then the EOS 7, which never enters the KV.
    const live = try fx.openSession(1, "a");
    const t1 = try runTurn(&fx, 1, live, "\x04", 16);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, t1.tokens());
    const t2 = try runTurn(&fx, 1, live, "\x02", 16);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4, 5, 6, 7 }, t2.tokens());

    // Kept: the same two turns as pairs, each reply the pieces of its ids
    // without the EOS.
    const kept = try fx.openSession(2, "a");
    const body = try fx.appendDrain(2, kept, &.{
        .{ .user = "\x04", .assistant = "56" },
        .{ .user = "\x02", .assistant = "3456" },
    });
    try expectAppended(body, &.{ t1.done.kv_pos, t2.done.kv_pos });
    try std.testing.expectEqual(@as(u32, 2), fx.sched.entries[1].turns_done);

    // The next turn on each session streams the same ids.
    const n1 = try runTurn(&fx, 1, live, "x", 4);
    const n2 = try runTurn(&fx, 2, kept, "x", 4);
    try std.testing.expectEqualSlices(u32, n1.tokens(), n2.tokens());
    try std.testing.expectEqual(n1.done.kv_pos, n2.done.kv_pos);

    // Both sessions stepped the same rows at the same positions.
    var a_buf: [64][2]u32 = undefined;
    var b_buf: [64][2]u32 = undefined;
    const a_rows = slotRows(&fx, fx.sched.entries[0].engine_id, &a_buf);
    const b_rows = slotRows(&fx, fx.sched.entries[1].engine_id, &b_buf);
    try std.testing.expectEqual(a_rows.len, b_rows.len);
    for (a_rows, b_rows) |a, b| {
        try std.testing.expectEqual(a, b);
    }
}

test "append: a set past the room refuses whole, feeds no row, and keeps the carry" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 16, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    // A cap of two leaves the sampled 2 pending, so it carries.
    _ = try runTurn(&fx, 1, sid, "x", 2);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(@as(?u32, 2), entry.carry);
    const pos = fx.pool.posOf(entry.engine_id);
    const rows_before = fx.runner.rows_log_len;

    // The carry joins the first pair. 1 + 4 + 8 = 13 ids fit the 13
    // positions left, and one byte more does not.
    try std.testing.expectEqual(@as(u32, 13), fx.pool.capacityRemaining(entry.engine_id));
    const refused = try fx.appendDrain(1, sid, &.{
        .{ .user = "abcd", .assistant = "" },
        .{ .user = "efgh", .assistant = "ijklm" },
    });
    try expectRefused(refused, .turn_too_long, "turn of 14 tokens, 13 positions left");
    try std.testing.expectEqual(rows_before, fx.runner.rows_log_len);
    try std.testing.expectEqual(pos, fx.pool.posOf(entry.engine_id));
    try std.testing.expectEqual(@as(u32, 1), entry.turns_done);
    try std.testing.expectEqual(@as(?u32, 2), entry.carry);
    try std.testing.expectEqual(SessionPhase.idle, entry.state);

    // The exact fit is admitted, and the carry leads its rows.
    const fits = try fx.appendDrain(1, sid, &.{
        .{ .user = "abcd", .assistant = "" },
        .{ .user = "efgh", .assistant = "ijkl" },
    });
    try expectAppended(fits, &.{ pos + 5, pos + 13 });
    try std.testing.expectEqual(@as(u32, 2), fx.runner.rowsSeen()[rows_before].token);
    try std.testing.expectEqual(@as(?u32, null), entry.carry);
    try std.testing.expectEqual(@as(u32, 3), entry.turns_done);
}

test "append: the pairs take the live wraps, and the next turn continues" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    var log_with: FakeCodec.KindLog = .{};
    fx.codec.kinds = &log_with;

    const sys = try fx.openSession(1, "a");
    try expectAppended(try fx.appendDrain(1, sys, &.{
        .{ .user = "b", .assistant = "c" },
        .{ .user = "d", .assistant = "e" },
    }), &.{ 3, 5 });
    _ = try runTurn(&fx, 1, sys, "\x06", 4);
    try std.testing.expectEqualSlices(
        event.TurnKind,
        &.{ .system, .first_after_system, .continuation, .continuation },
        log_with.seen[0..log_with.n],
    );

    // With no system prompt the first pair opens the sequence.
    var log_bare: FakeCodec.KindLog = .{};
    fx.codec.kinds = &log_bare;
    const bare = try fx.openSession(2, "");
    try expectAppended(try fx.appendDrain(2, bare, &.{.{ .user = "b", .assistant = "c" }}), &.{2});
    _ = try runTurn(&fx, 2, bare, "\x06", 4);
    try std.testing.expectEqualSlices(
        event.TurnKind,
        &.{ .open_first, .continuation },
        log_bare.seen[0..log_bare.n],
    );
}

test "append: two frames end in the state of one frame, the multi-frame rule" {
    // The admission at window 65 from position 1: two frames admit what one
    // frame admits. One row past the window refuses the whole set.
    const Case = struct { window: u32, frames: []const []const u32, want: ?u32 };
    const cases = [_]Case{
        .{ .window = 65, .frames = &.{&.{ 10, 20 }}, .want = 31 },
        .{ .window = 31, .frames = &.{&.{ 10, 20 }}, .want = 31 },
        .{ .window = 30, .frames = &.{&.{ 10, 20 }}, .want = null },
        .{ .window = 65, .frames = &.{&.{ 10, 20, 33 }}, .want = 64 },
        .{ .window = 65, .frames = &.{ &.{ 10, 20 }, &.{33} }, .want = 64 },
        .{ .window = 65, .frames = &.{&.{ 10, 20, 34 }}, .want = 65 },
        .{ .window = 65, .frames = &.{&.{ 10, 20, 35 }}, .want = null },
    };
    const text = "y" ** 64;
    var rows: [2][128][2]u32 = undefined;
    var row_sets: [2][][2]u32 = undefined;
    for (cases, 0..) |c, ci| {
        var fx = try Fixture.setUp(std.testing.allocator, 1, c.window, null);
        defer fx.tearDown();

        const sid = try fx.openSession(1, "a");
        var last: OutEvent.Body = undefined;
        for (c.frames) |frame| {
            var pairs: [4]protocol.Pair = undefined;
            for (frame, 0..) |cost, i| {
                pairs[i] = .{ .user = text[0..cost], .assistant = "" };
            }

            last = try fx.appendDrain(1, sid, pairs[0..frame.len]);
        }

        if (c.want) |pos| {
            try std.testing.expectEqual(OutEvent.Body.appended, std.meta.activeTag(last));
            const got = last.appended.positions();
            try std.testing.expectEqual(pos, got[got.len - 1]);
            try std.testing.expectEqual(pos, fx.pool.posOf(fx.sched.entries[0].engine_id));
        } else {
            try std.testing.expectEqual(protocol.ErrorCode.turn_too_long, last.err.code);
            try std.testing.expectEqual(@as(u32, 1), fx.pool.posOf(fx.sched.entries[0].engine_id));
        }

        // Cases 3 and 4 hold one frame and two frames of the same pairs. The
        // rows they step must match.
        if (ci == 3 or ci == 4) {
            row_sets[ci - 3] = slotRows(&fx, 0, &rows[ci - 3]);
        }
    }

    try std.testing.expectEqual(row_sets[0].len, row_sets[1].len);
    for (row_sets[0], row_sets[1]) |a, b| {
        try std.testing.expectEqual(a, b);
    }
}

test "append: a cancel during the prefill returns the session idle with an err" {
    // One slot gives one prefill row per tick, so the append spans ticks.
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    try fx.append(1, sid, &.{.{ .user = "bcde", .assistant = "" }});
    _ = fx.sched.tick(fx.sink);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(SessionPhase.prefill, entry.state);
    const fed = fx.pool.posOf(entry.engine_id);
    try std.testing.expect(fed > 1 and fed < 5);

    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 12, .session_id = sid, .body = .cancel },
        fx.sink,
    );
    fx.drain();
    try expectRefused(fx.sink.last().ev.body, .cancelled, "append cancelled");
    try std.testing.expectEqual(@as(u32, 11), fx.sink.last().ev.request_id);
    try std.testing.expectEqual(SessionPhase.idle, entry.state);
    try std.testing.expectEqual(@as(u32, 0), entry.turns_done);
    // The fed rows stay.
    try std.testing.expectEqual(fed, fx.pool.posOf(entry.engine_id));
    for (fx.sink.seen()) |e| {
        try std.testing.expect(e.ev.body != .appended);
    }

    // The session still answers, so a client can close it.
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 13, .session_id = sid, .body = .session_close },
        fx.sink,
    );
    try std.testing.expectEqual(
        OutEvent.Body.session_closed,
        std.meta.activeTag(fx.sink.last().ev.body),
    );
}

test "append: the guards of a turn hold, and an append emits no token" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    try expectRefused(
        try fx.appendDrain(1, 99, &.{.{ .user = "b", .assistant = "" }}),
        .unknown_session,
        "no such session",
    );
    try fx.generate(1, sid, "x", 8);
    try fx.append(1, sid, &.{.{ .user = "b", .assistant = "" }});
    try expectRefused(fx.sink.last().ev.body, .session_busy, "turn in flight");
    fx.drain();
    const before = fx.sink.n;
    try expectAppended(
        try fx.appendDrain(1, sid, &.{.{ .user = "b", .assistant = "c" }}),
        &.{fx.pool.posOf(0)},
    );
    try std.testing.expectEqual(before + 1, fx.sink.n);
}

test "parked frames: a close and a busy refusal flush the parked event first" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    try fx.generate(1, sid, "x", 8);
    // The first token of the turn parks behind a full queue.
    fx.sink.fail_next = 1;
    _ = fx.sched.tick(fx.sink);
    try std.testing.expect(fx.sched.entries[0].pending_event != null);

    // A busy refusal goes out behind the parked token.
    try fx.generate(1, sid, "y", 8);
    const n = fx.sink.n;
    try std.testing.expectEqual(
        OutEvent.Body.token,
        std.meta.activeTag(fx.sink.events[n - 2].ev.body),
    );
    try std.testing.expectEqual(
        protocol.ErrorCode.session_busy,
        fx.sink.events[n - 1].ev.body.err.code,
    );
    try std.testing.expect(fx.sched.entries[0].pending_event == null);

    // The next token parks, and the close sends it before session_closed.
    fx.sink.fail_next = 1;
    _ = fx.sched.tick(fx.sink);
    try std.testing.expect(fx.sched.entries[0].pending_event != null);
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 30, .session_id = sid, .body = .session_close },
        fx.sink,
    );
    const m = fx.sink.n;
    try std.testing.expectEqual(
        OutEvent.Body.token,
        std.meta.activeTag(fx.sink.events[m - 2].ev.body),
    );
    try std.testing.expectEqual(
        OutEvent.Body.session_closed,
        std.meta.activeTag(fx.sink.events[m - 1].ev.body),
    );
    try std.testing.expectEqual(@as(?u32, null), fx.sink.disconnected);
}

test "parked frames: a flush the writer still refuses disconnects, and no reply overtakes" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    try fx.generate(1, sid, "x", 8);
    fx.sink.fail_conn = 1;
    _ = fx.sched.tick(fx.sink);
    try std.testing.expect(fx.sched.entries[0].pending_event != null);
    const n = fx.sink.n;
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 30, .session_id = sid, .body = .session_close },
        fx.sink,
    );
    try std.testing.expectEqual(@as(?u32, 1), fx.sink.disconnected);
    try std.testing.expectEqual(n, fx.sink.n);
    // The sweep of the dropped connection releases the entry.
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 0, .session_id = 0, .body = .conn_closed },
        fx.sink,
    );
    try std.testing.expectEqual(SessionPhase.free, fx.sched.entries[0].state);
}

test "hello: the think mode, the budget, and the card sampling of the model" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    // A model with no tags reports none, the greedy default, and no resumption.
    var ack = fx.sched.helloAck(
        .{ .conn = 1, .request_id = 1, .session_id = 0, .body = .hello },
    ).body.hello_ack;
    try std.testing.expectEqual(protocol.ThinkMode.none, ack.think_mode);
    try std.testing.expectEqual(protocol.Sampling.greedy, ack.recommended_think);
    try std.testing.expectEqual(protocol.Sampling.greedy, ack.recommended_plain);
    try std.testing.expectEqual(@as(u32, 0), ack.resume_ttl_max_ms);

    var think = try Fixture.setUpThink(std.testing.allocator, 64, 0, think_tags);
    defer think.tearDown();

    // A model with tags reports its mode, its budget, and its card.
    const card: protocol.Sampling = .{
        .temperature = 1,
        .top_k = 20,
        .top_p = 0.95,
        .seed = 0,
        .presence_penalty = 1.5,
    };
    const plain: protocol.Sampling = .{
        .temperature = 0.7,
        .top_k = 20,
        .top_p = 0.8,
        .seed = 0,
        .presence_penalty = 1.5,
        .min_p = 0.05,
        .repetition_penalty = 1.1,
    };
    think.sched.cfg.recommended_think = card;
    think.sched.cfg.recommended_plain = plain;
    think.sched.cfg.resume_ttl_max_ms = 60_000;
    ack = think.sched.helloAck(
        .{ .conn = 1, .request_id = 1, .session_id = 0, .body = .hello },
    ).body.hello_ack;
    try std.testing.expectEqual(protocol.ThinkMode.on, ack.think_mode);
    try std.testing.expectEqual(@as(u32, 2), ack.think_budget);
    try std.testing.expectEqual(card, ack.recommended_think);
    try std.testing.expectEqual(plain, ack.recommended_plain);
    try std.testing.expectEqual(@as(u32, 60_000), ack.resume_ttl_max_ms);

    // A closed default reports off.
    think.sched.cfg.think.open_default = false;
    ack = think.sched.helloAck(
        .{ .conn = 1, .request_id = 1, .session_id = 0, .body = .hello },
    ).body.hello_ack;
    try std.testing.expectEqual(protocol.ThinkMode.off, ack.think_mode);
}

test "presence penalty: the set counts this completion only" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    // A negative penalty favors the ids already drawn. The canned logits put 1 on the next id and 0
    // elsewhere. So a drawn id at 0 + 2 wins.
    var s = protocol.Sampling.greedy;
    s.presence_penalty = -2;
    const dup = try std.testing.allocator.dupe(u8, "x");
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 9, .session_id = sid, .body = .{
        .generate = .{ .max_tokens = 3, .sampling = s, .text = dup },
    } }, fx.sink);
    fx.drain();
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 1 }, tokenIds(fx.sink.seen(), sid)[0..3]);

    // The next turn starts with an empty set: its first id follows the
    // chain, not the drawn 1.
    const t = try runTurn(&fx, 1, sid, "\x02", 3);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3, 3 }, t.tokens());
}

test "presence penalty: a budget-forced close counts as a drawn id, the echo does not" {
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, think_tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    // 5 and 6 use the budget, and the forced 3 and 0 follow.
    const t = try runTurn(&fx, 1, sid, "4", 4);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5, 6, 3, 0 }, t.tokens());
    const set = fx.sched.presenceOf(&fx.sched.entries[0]);
    for (0..test_vocab) |id| {
        const want = id == 0 or id == 3 or id == 5 or id == 6;
        const got = set[id / 64] & (@as(u64, 1) << @intCast(id % 64)) != 0;
        try std.testing.expectEqual(want, got);
    }
}

test "parked frames: a clone and an append flush the parked event before their refusal" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    try fx.generate(1, sid, "x", 8);
    fx.sink.fail_next = 1;
    _ = fx.sched.tick(fx.sink);
    try std.testing.expect(fx.sched.entries[0].pending_event != null);
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 40, .session_id = sid, .body = .{
        .session_clone = .{ .sampling = protocol.Sampling.greedy },
    } }, fx.sink);
    var n = fx.sink.n;
    try std.testing.expectEqual(
        OutEvent.Body.token,
        std.meta.activeTag(fx.sink.events[n - 2].ev.body),
    );
    try std.testing.expectEqual(
        protocol.ErrorCode.session_busy,
        fx.sink.events[n - 1].ev.body.err.code,
    );

    fx.sink.fail_next = 1;
    _ = fx.sched.tick(fx.sink);
    try std.testing.expect(fx.sched.entries[0].pending_event != null);
    try fx.append(1, sid, &.{.{ .user = "b", .assistant = "" }});
    n = fx.sink.n;
    try std.testing.expectEqual(
        OutEvent.Body.token,
        std.meta.activeTag(fx.sink.events[n - 2].ev.body),
    );
    try std.testing.expectEqual(
        protocol.ErrorCode.session_busy,
        fx.sink.events[n - 1].ev.body.err.code,
    );
    try std.testing.expectEqual(@as(u32, 11), fx.sink.events[n - 1].ev.request_id);
    fx.drain();
}

/// The one word of an id set over the eight fake ids.
fn setWord(set: []const u64) u64 {
    std.debug.assert(set.len == 1);

    return set[0];
}

/// The id set word of `ids`.
fn idMask(ids: []const u32) u64 {
    var m: u64 = 0;
    for (ids) |id| {
        m |= @as(u64, 1) << @intCast(id);
    }

    return m;
}

test "context set: every feed path marks it, a clone copies it, an open clears it" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    // The opening system prompt is id 3.
    const sa = try fx.openSession(1, "\x03");
    const a = fx.sched.entryByProto(sa, 1).?;
    try std.testing.expectEqual(idMask(&.{3}), setWord(fx.sched.contextOf(a)));
    // A turn feeds 5, and the model draws 6. The cap carries 6.
    var t = try runTurn(&fx, 1, sa, "\x05", 1);
    try std.testing.expectEqualSlices(u32, &.{6}, t.tokens());
    try std.testing.expectEqual(idMask(&.{ 3, 5, 6 }), setWord(fx.sched.contextOf(a)));
    // An append feeds the carried 6 and the pair id 0.
    _ = try fx.appendDrain(1, sa, &.{.{ .user = "\x00", .assistant = "" }});
    try std.testing.expectEqual(idMask(&.{ 0, 3, 5, 6 }), setWord(fx.sched.contextOf(a)));

    // A clone shares the context, so it starts with the set of its source.
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 5, .session_id = sa, .body = .{
        .session_clone = .{ .sampling = protocol.Sampling.greedy },
    } }, fx.sink);
    const sb = fx.sink.last().ev.session_id;
    const b = fx.sched.entryByProto(sb, 1).?;
    try std.testing.expectEqual(setWord(fx.sched.contextOf(a)), setWord(fx.sched.contextOf(b)));
    // A turn of the clone marks the clone alone.
    t = try runTurn(&fx, 1, sb, "\x01", 1);
    try std.testing.expectEqualSlices(u32, &.{2}, t.tokens());
    try std.testing.expectEqual(idMask(&.{ 0, 1, 2, 3, 5, 6 }), setWord(fx.sched.contextOf(b)));
    try std.testing.expectEqual(idMask(&.{ 0, 3, 5, 6 }), setWord(fx.sched.contextOf(a)));

    // A new session in a reused entry starts empty.
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 6, .session_id = sa, .body = .session_close },
        fx.sink,
    );
    const sc = try fx.openSession(1, "");
    try std.testing.expectEqual(
        @as(u64, 0),
        setWord(fx.sched.contextOf(fx.sched.entryByProto(sc, 1).?)),
    );
}

test "context set: a restored prefix marks it with no forward row" {
    var fx = try Fixture.setUpPrefix(std.testing.allocator, 2, 64, null, 1 << 20);
    defer fx.tearDown();

    // The rows path: the second open restores the whole system prompt.
    _ = try fx.openSession(1, "abcdefab");
    const rows_before = fx.runner.rows_log_len;
    const sb = try fx.openSession(2, "abcdefab");
    try std.testing.expectEqual(rows_before, fx.runner.rows_log_len);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.prefix.hits);
    try std.testing.expectEqual(
        idMask(&.{ 1, 2, 3, 4, 5, 6 }),
        setWord(fx.sched.contextOf(fx.sched.entryByProto(sb, 2).?)),
    );
    _ = fx.sched.handle(
        .{ .conn = 2, .request_id = 7, .session_id = sb, .body = .session_close },
        fx.sink,
    );

    // The ready path: an exact repeat of a generate turn restores the rows
    // and draws its first token from the stored logits. The prompt id 0
    // reaches the set through the restore alone.
    const sc = try fx.openSession(3, "");
    _ = try runTurn(&fx, 3, sc, "xxxxx", 2);
    _ = fx.sched.handle(
        .{ .conn = 3, .request_id = 8, .session_id = sc, .body = .session_close },
        fx.sink,
    );
    const sd = try fx.openSession(4, "");
    const rows_full = fx.runner.rows_log_len;
    const t = try runTurn(&fx, 4, sd, "xxxxx", 1);
    try std.testing.expectEqual(rows_full, fx.runner.rows_log_len);
    try std.testing.expectEqual(@as(u64, 2), fx.sched.prefix.hits);
    try std.testing.expectEqualSlices(u32, &.{1}, t.tokens());
    try std.testing.expectEqual(
        idMask(&.{ 0, 1 }),
        setWord(fx.sched.contextOf(fx.sched.entryByProto(sd, 4).?)),
    );
}

test "context set: a budget-forced id counts, the echo does not" {
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, think_tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    // The echo is 2. The prompt feeds 4, the model draws 5 and 6, and the
    // forced 3 and 0 follow.
    const t = try runTurn(&fx, 1, sid, "4", 4);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5, 6, 3, 0 }, t.tokens());
    try std.testing.expectEqual(
        idMask(&.{ 0, 3, 4, 5, 6 }),
        setWord(fx.sched.contextOf(&fx.sched.entries[0])),
    );
}

fn checkContextRecount(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    // Open a session with a random system prompt.
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    var text: [3]u8 = undefined;
    var want: u64 = 0;
    const sys = text[0..r.uintLessThan(usize, text.len + 1)];
    for (sys) |*c| {
        c.* = r.uintLessThan(u8, test_vocab);
    }

    const sid = try fx.openSession(1, sys);
    const entry = fx.sched.entryByProto(sid, 1).?;
    for (sys) |c| {
        want |= @as(u64, 1) << @intCast(c);
    }

    try std.testing.expectEqual(want, setWord(fx.sched.contextOf(entry)));

    // Run random turns and recount the context after each.
    for (0..5) |_| {
        const from = fx.sink.n;
        const msg = text[0 .. 1 + r.uintLessThan(usize, text.len)];
        for (msg) |*c| {
            c.* = r.uintLessThan(u8, test_vocab);
        }

        if (r.boolean()) {
            try fx.generate(1, sid, msg, 1 + r.uintLessThan(u32, 3));
            fx.drain();
        } else {
            _ = try fx.appendDrain(1, sid, &.{.{ .user = msg, .assistant = msg[0..1] }});
        }

        for (msg) |c| {
            want |= @as(u64, 1) << @intCast(c);
        }

        for (fx.sink.seen()[from..]) |e| {
            if (e.ev.session_id == sid and e.ev.body == .token) {
                want |= @as(u64, 1) << @intCast(e.ev.body.token.token_id);
            }
        }

        try std.testing.expectEqual(want, setWord(fx.sched.contextOf(entry)));
    }
}

test "context recount: the set equals the ids fed and drawn over random feed sequences" {
    try prop.forAll(
        .{ .site = 0x7005, .cases = 24 },
        {},
        u64,
        prop.genSeed(void),
        checkContextRecount,
    );
}

// -- lease tests ------------------------------------------------------------
//
// The anchors of the lease model replay through the scheduler with the fake
// sink and a pinned clock. The anchor table has two entries, a share of 2, and
// a maximum of 100 ms. Time counts in milliseconds from zero, and a token
// names its first byte. `viewOf` reads an entry in the shape of the model: the
// owner, the session id, the token, and the time to live.

/// The lease clock at `ms` milliseconds.
fn atMs(ms: i64) Io.Clock.Timestamp {
    return .{ .raw = .fromNanoseconds(@as(i96, ms) * std.time.ns_per_ms), .clock = .awake };
}

fn tok(n: u8) Token {
    var t: Token = @splat(0);
    t[0] = n;
    return t;
}

const EntryView = struct {
    owner: union(enum) { free, attached: u32, detached: i64 },
    pid: u32,
    token: ?u8,
    ttl: u32,
};

fn viewOf(entry: *const TestScheduler.SessionEntry) EntryView {
    const lease = entry.lease;
    return .{
        .owner = if (entry.state == .free) .free else switch (entry.owner) {
            .attached => |c| .{ .attached = c },
            .detached => |d| .{ .detached = d.raw.toMilliseconds() },
        },
        .pid = entry.proto_id,
        .token = if (lease) |l| l.token[0] else null,
        .ttl = if (lease) |l| l.ttl_ms else 0,
    };
}

const free_view: EntryView = .{ .owner = .free, .pid = 0, .token = null, .ttl = 0 };

/// A table of `n_slots` with resumption on at a maximum of 100 ms, and the
/// clock pinned at zero.
fn leaseFixture(n_slots: u32, share: u16) !Fixture {
    var fx = try Fixture.setUp(std.testing.allocator, n_slots, 64, null);
    fx.sched.cfg.resume_ttl_max_ms = 100;
    fx.sched.cfg.sessions_per_connection = share;
    fx.sched.pinClock(atMs(0));
    return fx;
}

const LeaseStep = union(enum) {
    open: struct { conn: u32, sid: u32 },
    /// The open answers `sessions_exhausted`. A nonempty detail must match.
    open_refused: struct { conn: u32, detail: []const u8 = "" },
    keep: struct { sid: u32, conn: u32, ttl: u32, token: u8, granted: u32 },
    detach: struct { conn: u32, at: i64 },
    expire: i64,
    pin: i64,
    resumed: struct { conn: u32, request: u32, token: u8, at: i64, fresh: u8, sid: u32 },
    no_such_lease: struct { conn: u32, request: u32, token: u8, at: i64, fresh: u8 },
    /// A close from `conn`. A refused close answers `unknown_session`.
    close: struct { conn: u32, sid: u32, refused: bool },
    route: struct { sid: u32, conn: u32, entry: ?usize },
    view: struct { entry: usize, want: EntryView },
    counts: struct { expires: u64, yields: u64 },
};

const LeaseCase = struct {
    name: []const u8,
    n_slots: u32,
    share: u16,
    steps: []const LeaseStep,
};

fn checkLease(c: LeaseCase) !void {
    var fx = try leaseFixture(c.n_slots, c.share);
    defer fx.tearDown();

    // Replay the steps and check each effect.
    for (c.steps) |step| {
        switch (step) {
            .open => |s| try std.testing.expectEqual(s.sid, try fx.openSession(s.conn, "")),
            .open_refused => |s| {
                const dup = try std.testing.allocator.dupe(u8, "");
                _ = fx.sched.handle(.{ .conn = s.conn, .request_id = 4, .session_id = 0, .body = .{
                    .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
                } }, fx.sink);
                const refused = fx.sink.last().ev.body.err;
                try std.testing.expectEqual(protocol.ErrorCode.sessions_exhausted, refused.code);
                if (s.detail.len > 0) {
                    try std.testing.expectEqualStrings(
                        s.detail,
                        refused.detail[0..refused.detail_len],
                    );
                }
            },
            .keep => |s| try std.testing.expectEqual(
                s.granted,
                try fx.sched.keepLease(s.sid, s.conn, s.ttl, tok(s.token)),
            ),
            .detach => |s| fx.sched.detachConn(s.conn, atMs(s.at)),
            .expire => |at| fx.sched.expireDetached(atMs(at)),
            .pin => |at| fx.sched.pinClock(atMs(at)),

            // Resume a detached lease.
            .resumed => |s| {
                const r = try fx.sched.resumeLease(
                    s.conn,
                    s.request,
                    tok(s.token),
                    atMs(s.at),
                    tok(s.fresh),
                );
                try std.testing.expectEqual(Resumed{ .session_id = s.sid, .ready = true }, r);
            },
            .no_such_lease => |s| try std.testing.expectError(
                error.NoSuchLease,
                fx.sched.resumeLease(s.conn, s.request, tok(s.token), atMs(s.at), tok(s.fresh)),
            ),
            .close => |s| {
                _ = fx.sched.handle(.{
                    .conn = s.conn,
                    .request_id = 5,
                    .session_id = s.sid,
                    .body = .session_close,
                }, fx.sink);
                if (s.refused) {
                    try std.testing.expectEqual(
                        protocol.ErrorCode.unknown_session,
                        fx.sink.last().ev.body.err.code,
                    );
                }
            },

            // Check the routes, the views, and the counters.
            .route => |s| {
                const Entry = TestScheduler.SessionEntry;
                const want: ?*Entry = if (s.entry) |i| &fx.sched.entries[i] else null;
                try std.testing.expectEqual(want, fx.sched.entryByProto(s.sid, s.conn));
            },
            .view => |s| try std.testing.expectEqual(s.want, viewOf(&fx.sched.entries[s.entry])),
            .counts => |s| {
                try std.testing.expectEqual(s.expires, fx.sched.lease_expires);
                try std.testing.expectEqual(s.yields, fx.sched.lease_yields);
            },
        }
    }
}

const view_detached_60: EntryView = .{
    .owner = .{ .detached = 60 },
    .pid = 1,
    .token = 7,
    .ttl = 50,
};
const view_attached_2_resumed: EntryView = .{
    .owner = .{ .attached = 2 },
    .pid = 1,
    .token = 8,
    .ttl = 50,
};
const view_attached_1_plain: EntryView = .{
    .owner = .{ .attached = 1 },
    .pid = 1,
    .token = null,
    .ttl = 0,
};

/// Entry 1 after a live open took it from a detached session.
const view_yield_taken: EntryView = .{
    .owner = .{ .attached = 2 },
    .pid = 5,
    .token = null,
    .ttl = 0,
};
const view_detached_30: EntryView = .{
    .owner = .{ .detached = 30 },
    .pid = 1,
    .token = 5,
    .ttl = 30,
};
const view_attached_9: EntryView = .{
    .owner = .{ .attached = 9 },
    .pid = 3,
    .token = null,
    .ttl = 0,
};
const view_detached_20: EntryView = .{
    .owner = .{ .detached = 20 },
    .pid = 4,
    .token = 7,
    .ttl = 20,
};
const view_clamped: EntryView = .{
    .owner = .{ .attached = 1 },
    .pid = 1,
    .token = 7,
    .ttl = 100,
};

/// The resume of `detached_anchor` before its deadline.
const resume_before_deadline: LeaseStep = .{ .resumed = .{
    .conn = 2,
    .request = 5,
    .token = 7,
    .at = 59,
    .fresh = 8,
    .sid = 1,
} };

/// The resume of `detached_anchor` at its deadline.
const resume_at_deadline: LeaseStep = .{ .no_such_lease = .{
    .conn = 2,
    .request = 5,
    .token = 7,
    .at = 60,
    .fresh = 8,
} };

/// A replay of token 7 after `resume_before_deadline`.
const resume_replayed: LeaseStep = .{ .no_such_lease = .{
    .conn = 3,
    .request = 6,
    .token = 7,
    .at = 59,
    .fresh = 9,
} };

/// Open on connection 1, keep with 50 and token 7, drop at 10.
const detached_anchor = [_]LeaseStep{
    .{ .open = .{ .conn = 1, .sid = 1 } },
    .{ .keep = .{ .sid = 1, .conn = 1, .ttl = 50, .token = 7, .granted = 50 } },
    .{ .detach = .{ .conn = 1, .at = 10 } },
    .{ .view = .{ .entry = 0, .want = view_detached_60 } },
};

test "lease anchors: the deadline, the token, the takeover order, the share, and the owner rules" {
    const cases = [_]LeaseCase{
        .{
            .name = "a resume before the deadline attaches the session",
            .n_slots = 2,
            .share = 2,
            .steps = &(detached_anchor ++ [_]LeaseStep{
                resume_before_deadline,
                .{ .view = .{ .entry = 0, .want = view_attached_2_resumed } },
            }),
        },
        .{
            .name = "a resume at the deadline fails, and the resume scan frees the session",
            .n_slots = 2,
            .share = 2,
            .steps = &(detached_anchor ++ [_]LeaseStep{
                .{ .expire = 59 },
                .{ .view = .{ .entry = 0, .want = view_detached_60 } },
                resume_at_deadline,
                .{ .view = .{ .entry = 0, .want = free_view } },
                .{ .counts = .{ .expires = 1, .yields = 0 } },
            }),
        },
        .{
            .name = "a replay of the old token after a resume fails",
            .n_slots = 2,
            .share = 2,
            .steps = &(detached_anchor ++ [_]LeaseStep{
                resume_before_deadline,
                resume_replayed,
                .{ .view = .{ .entry = 0, .want = view_attached_2_resumed } },
            }),
        },
        // Entry 2 stays attached to connection 9. The others detach at 0 with
        // the deadlines 30, 20, and 20.
        .{
            .name = "a live open takes the earliest deadline, and the lowest index on a tie",
            .n_slots = 4,
            .share = 4,
            .steps = &.{
                .{ .open = .{ .conn = 1, .sid = 1 } },
                .{ .open = .{ .conn = 1, .sid = 2 } },
                .{ .open = .{ .conn = 9, .sid = 3 } },
                .{ .open = .{ .conn = 1, .sid = 4 } },
                .{ .keep = .{ .sid = 1, .conn = 1, .ttl = 30, .token = 5, .granted = 30 } },
                .{ .keep = .{ .sid = 2, .conn = 1, .ttl = 20, .token = 6, .granted = 20 } },
                .{ .keep = .{ .sid = 4, .conn = 1, .ttl = 20, .token = 7, .granted = 20 } },
                .{ .detach = .{ .conn = 1, .at = 0 } },
                .{ .pin = 5 },
                .{ .open = .{ .conn = 2, .sid = 5 } },
                .{ .view = .{ .entry = 1, .want = view_yield_taken } },
                .{ .view = .{ .entry = 0, .want = view_detached_30 } },
                .{ .view = .{ .entry = 2, .want = view_attached_9 } },
                .{ .view = .{ .entry = 3, .want = view_detached_20 } },
                .{ .counts = .{ .expires = 0, .yields = 1 } },
            },
        },
        .{
            .name = "a table with every session attached refuses the open",
            .n_slots = 2,
            .share = 2,
            .steps = &.{
                .{ .open = .{ .conn = 1, .sid = 1 } },
                .{ .open = .{ .conn = 3, .sid = 2 } },
                .{ .open_refused = .{ .conn = 2, .detail = "no free session" } },
            },
        },
        .{
            .name = "a connection at its share is refused while a session entry is free",
            .n_slots = 2,
            .share = 1,
            .steps = &.{
                .{ .open = .{ .conn = 1, .sid = 1 } },
                .{ .open_refused = .{ .conn = 1 } },
                .{ .view = .{ .entry = 1, .want = free_view } },
            },
        },
        .{
            .name = "route and close refuse another connection",
            .n_slots = 2,
            .share = 2,
            .steps = &.{
                .{ .open = .{ .conn = 1, .sid = 1 } },
                .{ .route = .{ .sid = 1, .conn = 1, .entry = 0 } },
                .{ .route = .{ .sid = 1, .conn = 2, .entry = null } },
                .{ .close = .{ .conn = 2, .sid = 1, .refused = true } },
                .{ .view = .{ .entry = 0, .want = view_attached_1_plain } },
                .{ .close = .{ .conn = 1, .sid = 1, .refused = false } },
                .{ .view = .{ .entry = 0, .want = free_view } },
            },
        },
        .{
            .name = "a zero time to live drops the lease, and a keep clamps to the maximum",
            .n_slots = 2,
            .share = 2,
            .steps = &.{
                .{ .open = .{ .conn = 1, .sid = 1 } },
                .{ .keep = .{ .sid = 1, .conn = 1, .ttl = 50, .token = 7, .granted = 50 } },
                .{ .keep = .{ .sid = 1, .conn = 1, .ttl = 0, .token = 9, .granted = 0 } },
                .{ .view = .{ .entry = 0, .want = view_attached_1_plain } },
                .{ .keep = .{ .sid = 1, .conn = 1, .ttl = 500, .token = 7, .granted = 100 } },
                .{ .view = .{ .entry = 0, .want = view_clamped } },
                // A dropped lease returns the session to the sweep rule.
                .{ .keep = .{ .sid = 1, .conn = 1, .ttl = 0, .token = 9, .granted = 0 } },
                .{ .detach = .{ .conn = 1, .at = 10 } },
                .{ .view = .{ .entry = 0, .want = free_view } },
            },
        },
    };
    for (cases) |c| {
        errdefer std.debug.print("lease case: {s}\n", .{c.name});

        try checkLease(c);
    }
}

test "lease anchor: a takeover with no drop moves the route to the new connection" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    _ = try fx.sched.keepLease(sid, 1, 50, tok(7));
    const r = try fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(8));
    try std.testing.expectEqual(Resumed{ .session_id = sid, .ready = true }, r);
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .attached = 2 }, .pid = 1, .token = 8, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );
    try std.testing.expectEqual(
        @as(?*TestScheduler.SessionEntry, &fx.sched.entries[0]),
        fx.sched.entryByProto(sid, 2),
    );
    try std.testing.expectEqual(no_entry, fx.sched.entryByProto(sid, 1));
    // The old connection gets `unknown_session`, and the new one streams.
    try fx.generate(1, sid, "x", 1);
    try std.testing.expectEqual(
        protocol.ErrorCode.unknown_session,
        fx.sink.last().ev.body.err.code,
    );
    try fx.generate(2, sid, "x", 2);
    fx.drain();
    try std.testing.expectEqual(@as(u32, 2), fx.sink.last().conn);
    try std.testing.expectEqual(
        protocol.FinishReason.max_tokens,
        fx.sink.last().ev.body.done.finish_reason,
    );
}

test "lease: the three keep refusals each take their own path" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");

    // No such session: an id that never existed, and an id of a closed one.
    try std.testing.expectError(error.NoSuchSession, fx.sched.keepLease(999, 1, 50, tok(7)));
    try std.testing.expectError(error.NoSuchSession, fx.sched.keepLease(0, 1, 50, tok(7)));
    const closed = try fx.openSession(1, "");
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 5, .session_id = closed, .body = .session_close },
        fx.sink,
    );
    try std.testing.expectError(error.NoSuchSession, fx.sched.keepLease(closed, 1, 50, tok(7)));

    // Another owner: an attached session of another connection, and a
    // detached session, which no connection owns.
    try std.testing.expectError(error.OtherOwner, fx.sched.keepLease(sid, 2, 50, tok(7)));
    _ = try fx.sched.keepLease(sid, 1, 50, tok(7));
    fx.sched.detachConn(1, atMs(10));
    try std.testing.expectError(error.OtherOwner, fx.sched.keepLease(sid, 1, 50, tok(9)));
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .detached = 60 }, .pid = 1, .token = 7, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );

    // Resumption off: the keep is refused, the lease does not change, and the drop frees the
    // session.
    var off = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer off.tearDown();

    try std.testing.expectEqual(@as(u32, 0), off.sched.cfg.resume_ttl_max_ms);
    const s = try off.openSession(1, "");
    try std.testing.expectError(error.ResumptionOff, off.sched.keepLease(s, 1, 50, tok(7)));
    try std.testing.expectEqual(@as(?TestScheduler.Lease, null), off.sched.entries[0].lease);
    try std.testing.expectEqual(@as(u32, 0), try off.sched.keepLease(s, 1, 0, tok(7)));
    _ = off.sched.handle(
        .{ .conn = 1, .request_id = 0, .session_id = 0, .body = .conn_closed },
        off.sink,
    );
    try std.testing.expectEqual(free_view, viewOf(&off.sched.entries[0]));
}

test "lease: a drop mid-decode cuts the turn, keeps the carry, and the resumed turn feeds it" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    _ = try fx.sched.keepLease(sid, 1, 50, tok(7));
    try fx.generate(1, sid, "x", 10);
    // The prefill samples 1, then two decode steps sample 2 and 3.
    for (0..3) |_| {
        _ = fx.sched.tick(fx.sink);
    }

    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, tokenIds(fx.sink.seen(), sid)[0..3]);

    const seen_before = fx.sink.n;
    fx.sched.detachConn(1, atMs(10));
    fx.drain();
    // The cut turn finalizes, but no event reaches any connection.
    try std.testing.expectEqual(seen_before, fx.sink.n);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(SessionPhase.idle, entry.state);
    try std.testing.expectEqual(@as(?u32, 3), entry.carry);
    try std.testing.expectEqual(protocol.FinishReason.cancelled, entry.last_done.?.finish);
    try std.testing.expectEqual(@as(u32, 3), entry.last_done.?.gen_tokens);
    try std.testing.expectEqual(@as(u32, 9), entry.last_done.?.request_id);

    _ = try fx.sched.resumeLease(2, 5, tok(7), atMs(20), tok(8));
    const rows_before = fx.runner.rows_log_len;
    try fx.generate(2, sid, "z", 1); // 'z' % 8 = 2
    fx.drain();
    const rows = fx.runner.rowsSeen()[rows_before..];
    try std.testing.expectEqual(@as(u32, 3), rows[0].token); // the carry
    try std.testing.expectEqual(@as(u32, 2), rows[1].token);
    try std.testing.expectEqual(@as(u32, 2), fx.sink.last().conn);
}

test "lease: an append finishes while detached, and the resume names the final position" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    var kinds: FakeCodec.KindLog = .{};
    fx.codec.kinds = &kinds;
    const sid = try fx.openSession(1, "a");
    _ = try fx.sched.keepLease(sid, 1, 50, tok(7));
    // Six tokens at two rows a pass: three passes.
    try fx.append(1, sid, &.{.{ .user = "abcd", .assistant = "ef" }});
    _ = fx.sched.tick(fx.sink);
    try std.testing.expectEqual(SessionPhase.prefill, fx.sched.entries[0].state);

    const seen_before = fx.sink.n;
    const rows_before = fx.runner.rows_log_len;
    fx.sched.detachConn(1, atMs(10));
    fx.drain();
    try std.testing.expectEqual(@as(usize, 4), fx.runner.rows_log_len - rows_before);
    try std.testing.expectEqual(seen_before, fx.sink.n);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(SessionPhase.idle, entry.state);
    try std.testing.expectEqual(@as(u32, 1), entry.turns_done);
    try std.testing.expectEqual(@as(u32, 7), fx.pool.posOf(entry.engine_id));

    const r = try fx.sched.resumeLease(2, 5, tok(7), atMs(20), tok(8));
    try std.testing.expectEqual(Resumed{ .session_id = sid, .ready = true }, r);
    try fx.generate(2, sid, "x", 1);
    fx.drain();
    try std.testing.expectEqual(event.TurnKind.continuation, kinds.seen[kinds.n - 1]);
}

test "lease: a resume during an append waits for idle and drops the events before it" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    _ = try fx.sched.keepLease(sid, 1, 50, tok(7));
    try fx.append(1, sid, &.{.{ .user = "abcd", .assistant = "ef" }});
    _ = fx.sched.tick(fx.sink);

    const seen_before = fx.sink.n;
    const r = try fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(8));
    try std.testing.expectEqual(Resumed{ .session_id = sid, .ready = false }, r);
    try std.testing.expect(fx.sched.entries[0].resume_wait != null);
    fx.drain();
    // The append reply belonged to connection 1, so it goes. The one event
    // is the resume, and it names the final position.
    try std.testing.expectEqual(seen_before + 1, fx.sink.n);
    try expectResumed(fx.sink.last(), 2, 5, sid, 7, 8);
    try std.testing.expectEqual(
        @as(?TestScheduler.ResumeWait, null),
        fx.sched.entries[0].resume_wait,
    );
    try std.testing.expectEqual(@as(u32, 7), fx.pool.posOf(fx.sched.entries[0].engine_id));
}

/// A `resumed` reply to `conn` under `request_id` for session `sid` at
/// `kv_pos`, with the token that `token_byte` names.
fn expectResumed(
    e: FakeSink.Seen,
    conn: u32,
    request_id: u32,
    sid: u32,
    kv_pos: u32,
    token_byte: u8,
) !void {
    try std.testing.expectEqual(conn, e.conn);
    try std.testing.expectEqual(request_id, e.ev.request_id);
    try std.testing.expectEqual(sid, e.ev.session_id);
    const r = e.ev.body.resumed;
    try std.testing.expectEqual(kv_pos, r.kv_pos);
    try std.testing.expectEqual(tok(token_byte), r.token);
}

test "lease: the share of the new connection bounds a resume" {
    var fx = try leaseFixture(2, 1);
    defer fx.tearDown();

    const s1 = try fx.openSession(1, "");
    _ = try fx.sched.keepLease(s1, 1, 50, tok(7));
    fx.sched.detachConn(1, atMs(10));
    const s2 = try fx.openSession(2, "");

    try std.testing.expectError(
        error.ShareFull,
        fx.sched.resumeLease(2, 5, tok(7), atMs(20), tok(8)),
    );
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .detached = 60 }, .pid = 1, .token = 7, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );

    _ = fx.sched.handle(
        .{ .conn = 2, .request_id = 6, .session_id = s2, .body = .session_close },
        fx.sink,
    );
    _ = try fx.sched.resumeLease(2, 5, tok(7), atMs(20), tok(8));
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .attached = 2 }, .pid = 1, .token = 8, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );

    // A connection at its share cannot take over its own session.
    try std.testing.expectError(
        error.ShareFull,
        fx.sched.resumeLease(2, 7, tok(8), atMs(30), tok(9)),
    );
    try std.testing.expectEqual(@as(?u8, 8), viewOf(&fx.sched.entries[0]).token);
}

test "lease: a detached session takes no decode row in any step" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sa = try fx.openSession(1, "a");
    const sb = try fx.openSession(2, "a");
    _ = try fx.sched.keepLease(sa, 1, 50, tok(7));
    try fx.generate(1, sa, "x", 6);
    try fx.generate(2, sb, "x", 6);
    for (0..3) |_| {
        _ = fx.sched.tick(fx.sink);
    }

    const engine_a = fx.sched.entries[0].engine_id;
    try std.testing.expectEqual(SessionPhase.decode, fx.sched.entries[0].state);

    fx.sched.detachConn(1, atMs(10));
    const rows_before = fx.runner.rows_log_len;
    var steps: u32 = 0;
    while (fx.sched.tick(fx.sink) != .idle) : (steps += 1) {
        for (fx.runner.rowsSeen()[rows_before..]) |row| {
            try std.testing.expect(row.slot != engine_a);
        }
    }

    try std.testing.expect(steps >= 1);
    try std.testing.expect(fx.runner.rows_log_len > rows_before);
    try std.testing.expectEqualSlices(
        u32,
        &.{ 1, 2, 3, 4, 5, 6 },
        tokenIds(fx.sink.seen(), sb)[0..6],
    );
}

test "lease: a yield cancels an opening prefill in flight before it releases the session" {
    var fx = try leaseFixture(1, 1);
    defer fx.tearDown();

    const dup = try std.testing.allocator.dupe(u8, "aaaaaa");
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 1, .session_id = 0, .body = .{
        .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
    } }, fx.sink);
    // One row a pass: two of the six prompt rows feed.
    _ = fx.sched.tick(fx.sink);
    _ = fx.sched.tick(fx.sink);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(SessionPhase.prefill, entry.state);
    try std.testing.expectEqual(@as(usize, 2), entry.fed);
    _ = try fx.sched.keepLease(entry.proto_id, 1, 50, tok(7));
    fx.sched.detachConn(1, atMs(10));

    const rows_before = fx.runner.rows_log_len;
    fx.sched.pinClock(atMs(20));
    try std.testing.expectEqual(@as(u32, 2), try fx.openSession(2, ""));
    fx.drain();
    try std.testing.expectEqual(rows_before, fx.runner.rows_log_len);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_yields);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_prefills_cut);
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .attached = 2 }, .pid = 2, .token = null, .ttl = 0 },
        viewOf(entry),
    );
    try std.testing.expectEqual(@as(u32, 0), fx.pool.posOf(entry.engine_id));
}

test "lease: an expiry at the tick head cancels an append prefill before the release" {
    var fx = try leaseFixture(1, 1);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    _ = try fx.sched.keepLease(sid, 1, 50, tok(7));
    try fx.append(1, sid, &.{.{ .user = "abcd", .assistant = "ef" }});
    _ = fx.sched.tick(fx.sink);
    _ = fx.sched.tick(fx.sink);
    try std.testing.expectEqual(@as(usize, 2), fx.sched.entries[0].fed);
    fx.sched.detachConn(1, atMs(10));

    // One pass before the deadline feeds one more row of the append.
    fx.sched.pinClock(atMs(59));
    try std.testing.expectEqual(Tick.worked, fx.sched.tick(fx.sink));
    try std.testing.expectEqual(@as(usize, 3), fx.sched.entries[0].fed);

    const rows_before = fx.runner.rows_log_len;
    fx.sched.pinClock(atMs(60));
    try std.testing.expectEqual(Tick.idle, fx.sched.tick(fx.sink));
    fx.drain();
    try std.testing.expectEqual(rows_before, fx.runner.rows_log_len);
    try std.testing.expectEqual(free_view, viewOf(&fx.sched.entries[0]));
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_expires);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_prefills_cut);
}

test "lease: the secure source draws distinct nonzero tokens" {
    const a = try drawToken(std.testing.io);
    const b = try drawToken(std.testing.io);
    try std.testing.expect(!std.mem.eql(u8, &a, &b));
    try std.testing.expect(!std.mem.allEqual(u8, &a, 0));
}

/// Send `body` for session `sid` on `conn` and return the error code of the
/// reply.
fn refusalOf(fx: *Fixture, conn: u32, sid: u32, body: Request.Body) protocol.ErrorCode {
    _ = fx.sched.handle(
        .{ .conn = conn, .request_id = 20, .session_id = sid, .body = body },
        fx.sink,
    );
    const e = fx.sink.last();
    std.debug.assert(e.ev.body == .err);
    return e.ev.body.err.code;
}

/// The refusal of a request on an entry whose resume waits.
fn expectWaitRefusal(e: FakeSink.Seen) !void {
    try std.testing.expectEqual(protocol.ErrorCode.session_busy, e.ev.body.err.code);
    try std.testing.expectEqualStrings(
        "resume in flight",
        e.ev.body.err.detail[0..e.ev.body.err.detail_len],
    );
}

/// An open on connection 1 with a kept lease and an append in flight. The
/// first pass feeds two of its six rows.
fn appendInFlight(fx: *Fixture) !u32 {
    const sid = try fx.openSession(1, "a");
    _ = try fx.sched.keepLease(sid, 1, 50, tok(7));
    try fx.append(1, sid, &.{.{ .user = "abcd", .assistant = "ef" }});
    _ = fx.sched.tick(fx.sink);
    std.debug.assert(fx.sched.entries[0].state == .prefill);
    return sid;
}

test "lease: a waiting resume refuses each request, ends at idle, and the next one streams" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try appendInFlight(&fx);
    const r = try fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(8));
    try std.testing.expect(!r.ready);

    // Every request on the session gets `session_busy` until the resume ends.
    try fx.generate(2, sid, "x", 2);
    try expectWaitRefusal(fx.sink.last());
    try fx.append(2, sid, &.{.{ .user = "a", .assistant = "b" }});
    try expectWaitRefusal(fx.sink.last());
    try std.testing.expectEqual(protocol.ErrorCode.session_busy, refusalOf(&fx, 2, sid, .cancel));
    try expectWaitRefusal(fx.sink.last());
    try std.testing.expectEqual(
        protocol.ErrorCode.session_busy,
        refusalOf(&fx, 2, sid, .{ .session_clone = .{ .sampling = protocol.Sampling.greedy } }),
    );
    try expectWaitRefusal(fx.sink.last());
    try std.testing.expectEqual(SessionPhase.prefill, fx.sched.entries[0].state);

    // The second pass ends the append inside the tick, and the resume ends
    // with it. A generate before the next tick then streams to connection 2.
    _ = fx.sched.tick(fx.sink);
    _ = fx.sched.tick(fx.sink);
    try std.testing.expectEqual(SessionPhase.idle, fx.sched.entries[0].state);
    try std.testing.expectEqual(
        @as(?TestScheduler.ResumeWait, null),
        fx.sched.entries[0].resume_wait,
    );
    const from = fx.sink.n;
    try fx.generate(2, sid, "x", 2);
    fx.drain();
    var tokens: usize = 0;
    for (fx.sink.seen()[from..]) |e| {
        try std.testing.expectEqual(@as(u32, 2), e.conn);
        if (e.ev.body == .token) {
            tokens += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 2), tokens);
    try std.testing.expectEqual(
        protocol.FinishReason.max_tokens,
        fx.sink.last().ev.body.done.finish_reason,
    );
}

test "lease: the client waits for the resume, then generates on the new connection" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try appendInFlight(&fx);
    const seen_before = fx.sink.n;
    _ = try fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(8));
    fx.drain();
    try std.testing.expectEqual(seen_before + 1, fx.sink.n);
    try expectResumed(fx.sink.last(), 2, 5, sid, 7, 8);
    try std.testing.expectEqual(@as(u32, 7), fx.pool.posOf(fx.sched.entries[0].engine_id));
    try fx.generate(2, sid, "x", 3);
    fx.drain();
    try std.testing.expectEqualSlices(
        u32,
        &.{ 1, 2, 3 },
        tokenIds(fx.sink.seen()[seen_before..], sid)[0..3],
    );
    for (fx.sink.seen()[seen_before..]) |e| {
        try std.testing.expectEqual(@as(u32, 2), e.conn);
    }
}

test "lease: a takeover mid-turn cuts the turn, keeps the carry, and the old token opens nothing" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    _ = try fx.sched.keepLease(sid, 1, 50, tok(7));
    try fx.generate(1, sid, "x", 10);
    for (0..3) |_| {
        _ = fx.sched.tick(fx.sink);
    }

    // The resume waits for the stop of the cut turn, and a cancel on it is busy.
    const r = try fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(8));
    try std.testing.expectEqual(Resumed{ .session_id = sid, .ready = false }, r);
    try std.testing.expectEqual(protocol.ErrorCode.session_busy, refusalOf(&fx, 2, sid, .cancel));
    try std.testing.expectEqual(
        protocol.ErrorCode.unknown_session,
        refusalOf(&fx, 1, sid, .cancel),
    );
    const seen_before = fx.sink.n;
    fx.drain();
    // The done of the cut turn belonged to connection 1, so it goes. The
    // resume names the cut turn as the last request.
    try std.testing.expectEqual(seen_before + 1, fx.sink.n);
    try expectResumed(fx.sink.last(), 2, 5, sid, fx.pool.posOf(fx.sched.entries[0].engine_id), 8);
    try std.testing.expectEqual(
        protocol.FinishReason.cancelled,
        fx.sink.last().ev.body.resumed.last.?.finish,
    );
    try std.testing.expectEqual(@as(u32, 9), fx.sink.last().ev.body.resumed.last.?.request_id);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(@as(?TestScheduler.ResumeWait, null), entry.resume_wait);
    try std.testing.expectEqual(@as(?u32, 3), entry.carry);
    try std.testing.expectEqual(protocol.FinishReason.cancelled, entry.last_done.?.finish);
    try std.testing.expectError(
        error.NoSuchLease,
        fx.sched.resumeLease(3, 6, tok(7), atMs(6), tok(9)),
    );

    // The next turn feeds the carry first.
    const rows_before = fx.runner.rows_log_len;
    try fx.generate(2, sid, "z", 1);
    fx.drain();
    try std.testing.expectEqual(@as(u32, 3), fx.runner.rowsSeen()[rows_before].token);
}

test "lease: a second resume takes over a waiting resume" {
    var fx = try leaseFixture(3, 3);
    defer fx.tearDown();

    const sid = try appendInFlight(&fx);
    _ = try fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(8));
    const r = try fx.sched.resumeLease(3, 6, tok(8), atMs(6), tok(9));
    try std.testing.expectEqual(Resumed{ .session_id = sid, .ready = false }, r);
    try std.testing.expectEqual(@as(u32, 3), fx.sched.entries[0].resume_wait.?.conn);
    try std.testing.expectEqual(no_entry, fx.sched.entryByProto(sid, 2));
    try std.testing.expectError(
        error.NoSuchLease,
        fx.sched.resumeLease(2, 7, tok(8), atMs(7), tok(10)),
    );
    fx.drain();
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .attached = 3 }, .pid = sid, .token = 9, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );
    try std.testing.expectEqual(
        @as(?TestScheduler.ResumeWait, null),
        fx.sched.entries[0].resume_wait,
    );
}

test "lease: the new connection drops while its resume waits, and the session detaches again" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try appendInFlight(&fx);
    _ = try fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(8));
    fx.sched.detachConn(2, atMs(20));
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .detached = 70 }, .pid = sid, .token = 8, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );
    try std.testing.expectEqual(
        @as(?TestScheduler.ResumeWait, null),
        fx.sched.entries[0].resume_wait,
    );
    fx.drain();
    try std.testing.expectEqual(SessionPhase.idle, fx.sched.entries[0].state);
    try std.testing.expectEqual(@as(u32, 1), fx.sched.entries[0].turns_done);
    const r = try fx.sched.resumeLease(3, 6, tok(8), atMs(30), tok(9));
    try std.testing.expectEqual(Resumed{ .session_id = sid, .ready = true }, r);
}

test "lease: a shutdown cancels the work of a detached session and keeps its lease" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    _ = try appendInFlight(&fx);
    fx.sched.detachConn(1, atMs(10));
    const sb = try fx.openSession(2, "a");
    try fx.generate(2, sb, "x", 20);
    _ = fx.sched.tick(fx.sink);

    const seen_before = fx.sink.n;
    const res = fx.sched.handle(
        .{ .conn = 0, .request_id = 30, .session_id = 0, .body = .shutdown },
        fx.sink,
    );
    try std.testing.expectEqual(HandleResult.shutdown, res);
    // Connection 2 gets its done, connection 0 its ack, and the detached
    // entry sends nothing.
    for (fx.sink.seen()[seen_before..]) |e| {
        try std.testing.expect(e.conn == 2 or e.conn == 0);
    }

    try std.testing.expectEqual(SessionPhase.idle, fx.sched.entries[0].state);
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .detached = 60 }, .pid = 1, .token = 7, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );
}

test "lease: a session of the process owner keeps a lease like any other" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try fx.openSession(0, "");
    _ = try fx.sched.keepLease(sid, 0, 50, tok(7));
    fx.sched.detachConn(0, atMs(10));
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .detached = 60 }, .pid = sid, .token = 7, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );
    try std.testing.expectEqual(no_entry, fx.sched.entryByProto(sid, 0));
    _ = try fx.sched.resumeLease(5, 5, tok(7), atMs(20), tok(8));
    try std.testing.expectEqual(
        @as(?*TestScheduler.SessionEntry, &fx.sched.entries[0]),
        fx.sched.entryByProto(sid, 5),
    );
    // The owner keeps the right to stop the server.
    try std.testing.expectEqual(
        HandleResult.shutdown,
        fx.sched.handle(
            .{ .conn = 0, .request_id = 30, .session_id = 0, .body = .shutdown },
            fx.sink,
        ),
    );
}

test "lease: a token that a session already holds is refused in every build mode" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const s1 = try fx.openSession(1, "");
    const s2 = try fx.openSession(1, "");
    _ = try fx.sched.keepLease(s1, 1, 50, tok(7));
    try std.testing.expectError(error.TokenInUse, fx.sched.keepLease(s2, 1, 50, tok(7)));
    try std.testing.expectError(error.TokenInUse, fx.sched.keepLease(s1, 1, 50, tok(7)));
    try std.testing.expectEqual(@as(?TestScheduler.Lease, null), fx.sched.entries[1].lease);
    _ = try fx.sched.keepLease(s2, 1, 50, tok(8));
    try std.testing.expectError(
        error.TokenInUse,
        fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(8)),
    );
    try std.testing.expectError(
        error.TokenInUse,
        fx.sched.resumeLease(2, 5, tok(7), atMs(5), tok(7)),
    );
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .attached = 1 }, .pid = s1, .token = 7, .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );
}

/// `TestPool` with a switch that refuses each create and clone, as a full pool or tiered KV can.
/// Two more switches fail `advance`: one for a call that carries a prefill chunk, one for a call
/// of decode rows alone.
const RefusingPool = struct {
    inner: TestPool,
    refuse: bool = false,
    fail_with_feed: ?AdvanceFail = null,
    fail_decode_only: ?AdvanceFail = null,

    const AdvanceFail = enum { park, other };

    pub const can_clone = true;

    pub fn create(
        self: *RefusingPool,
        seed: u64,
        params: sampler.Params,
    ) error{NoFreeSession}!u32 {
        if (self.refuse) return error.NoFreeSession;
        return self.inner.create(seed, params);
    }

    pub fn clone(
        self: *RefusingPool,
        source: u32,
        seed: u64,
        params: sampler.Params,
    ) error{ NoFreeSession, RamBudgetExceeded }!u32 {
        if (self.refuse) return error.RamBudgetExceeded;
        return self.inner.clone(source, seed, params);
    }

    pub fn destroy(self: *RefusingPool, id: u32) void {
        self.inner.destroy(id);
    }

    pub fn setParams(self: *RefusingPool, id: u32, params: sampler.Params) void {
        self.inner.setParams(id, params);
    }

    pub fn reseed(self: *RefusingPool, id: u32, seed: u64) void {
        self.inner.reseed(id, seed);
    }

    pub fn overridePending(self: *RefusingPool, id: u32, token: u32) void {
        self.inner.overridePending(id, token);
    }

    pub fn posOf(self: *RefusingPool, id: u32) u32 {
        return self.inner.posOf(id);
    }

    pub fn capacityRemaining(self: *RefusingPool, id: u32) u32 {
        return self.inner.capacityRemaining(id);
    }

    pub fn advance(
        self: *RefusingPool,
        feed_chunk: ?session_mod.FeedChunk,
        ids: []const u32,
        out: []u32,
    ) !?u32 {
        const scripted = if (feed_chunk != null) self.fail_with_feed else self.fail_decode_only;
        if (scripted) |fail| {
            return switch (fail) {
                .park => error.NoEvictableSlot,
                .other => error.AdvanceBroken,
            };
        }

        return self.inner.advance(feed_chunk, ids, out);
    }
};

const RefusingScheduler = Scheduler(RefusingPool, FakeCodec);

/// A table of `n_slots` over the refusing pool, with resumption on and the clock at zero.
const RefusingFixture = struct {
    arena_state: std.heap.ArenaAllocator,
    pool: *RefusingPool,
    sched: *RefusingScheduler,
    sink: *FakeSink,

    fn setUp(n_slots: u32) !RefusingFixture {
        var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
        errdefer arena_state.deinit();

        const arena = arena_state.allocator();
        const runner = try arena.create(TestRunner);
        runner.* = .{
            .n_slots = n_slots,
            .max_seq = 64,
            .logits = try arena.alloc(f32, n_slots * test_vocab),
        };
        @memset(runner.logits, 0);
        const pool = try arena.create(RefusingPool);
        pool.* = .{ .inner = try TestPool.init(arena, runner) };
        const codec = try arena.create(FakeCodec);
        codec.* = FakeCodec.initFake();
        const sched = try arena.create(RefusingScheduler);
        sched.* = try RefusingScheduler.init(
            arena,
            std.testing.allocator,
            std.testing.io,
            pool,
            codec,
            .{
                .max_sessions = @intCast(n_slots),
                .max_batch = @intCast(n_slots),
                .max_seq = 64,
                .max_step_rows = n_slots,
                .model_name = "fake",
                .vocab_size = test_vocab,
                .eos_id = 7,
                .resume_ttl_max_ms = 100,
            },
        );
        sched.pinClock(atMs(0));
        const sink = try arena.create(FakeSink);
        sink.* = .{};
        return .{ .arena_state = arena_state, .pool = pool, .sched = sched, .sink = sink };
    }

    fn tearDown(self: *RefusingFixture) void {
        self.sched.deinit();
        self.arena_state.deinit();
    }

    fn open(self: *RefusingFixture, conn: u32) !void {
        const dup = try std.testing.allocator.dupe(u8, "");
        _ = self.sched.handle(.{ .conn = conn, .request_id = 1, .session_id = 0, .body = .{
            .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
        } }, self.sink);
    }
};

/// Open one session on `fx` and run the open to its reply.
fn parkOpen(fx: *RefusingFixture, conn: u32) !u32 {
    try fx.open(conn);
    while (fx.sched.tick(fx.sink) != .idle) {}

    const ev = fx.sink.last().ev;
    try std.testing.expectEqual(OutEvent.Body.session_opened, std.meta.activeTag(ev.body));
    return ev.session_id;
}

fn parkGenerate(fx: *RefusingFixture, conn: u32, sid: u32, text: []const u8, max: u32) !void {
    const dup = try std.testing.allocator.dupe(u8, text);
    _ = fx.sched.handle(.{ .conn = conn, .request_id = 9, .session_id = sid, .body = .{
        .generate = .{ .max_tokens = max, .sampling = null, .text = dup },
    } }, fx.sink);
}

test "park: decode rows alone that the engine parks report no work and change nothing" {
    var fx = try RefusingFixture.setUp(2);
    defer fx.tearDown();

    const sid = try parkOpen(&fx, 1);
    try parkGenerate(&fx, 1, sid, "x", 4);
    try std.testing.expectEqual(Tick.worked, fx.sched.tick(fx.sink)); // prefill and first token

    const tokens_before = countTokens(fx.sink.seen(), sid);
    const events_before = fx.sink.n;
    const pos_before = fx.pool.posOf(0);
    try std.testing.expect(tokens_before >= 1 and tokens_before < 4);

    // The tiers refuse the decode call. Each tick parks and reports no work, so the engine loop
    // sleeps and does not spin.
    fx.pool.fail_decode_only = .park;
    const parks_before = fx.sched.parks;
    for (0..3) |_| {
        try std.testing.expectEqual(Tick.idle, fx.sched.tick(fx.sink));
    }

    try std.testing.expectEqual(parks_before + 3, fx.sched.parks);
    try std.testing.expectEqual(events_before, fx.sink.n);
    try std.testing.expectEqual(pos_before, fx.pool.posOf(0));

    // A later blob removal lets the same turn finish with the whole stream.
    fx.pool.fail_decode_only = null;
    var iters: u32 = 0;
    while (fx.sched.tick(fx.sink) != .idle) {
        iters += 1;
        try std.testing.expect(iters <= 50);
    }

    const toks = tokenIds(fx.sink.seen(), sid);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, toks[0..4]);
}

test "park: a parked prefill whose decode retry parks too reports no work" {
    var fx = try RefusingFixture.setUp(2);
    defer fx.tearDown();

    const sa = try parkOpen(&fx, 1);
    const sb = try parkOpen(&fx, 2);
    try parkGenerate(&fx, 1, sa, "x", 6);
    try std.testing.expectEqual(Tick.worked, fx.sched.tick(fx.sink)); // a decodes
    try parkGenerate(&fx, 2, sb, "xxxx", 6); // b prefills beside a

    const events_before = fx.sink.n;
    const parks_before = fx.sched.parks;
    fx.pool.fail_with_feed = .park;
    fx.pool.fail_decode_only = .park;
    try std.testing.expectEqual(Tick.idle, fx.sched.tick(fx.sink));

    // Both calls were refused and both are counted. Nothing reached a stream.
    try std.testing.expectEqual(parks_before + 2, fx.sched.parks);
    try std.testing.expectEqual(events_before, fx.sink.n);

    fx.pool.fail_with_feed = null;
    fx.pool.fail_decode_only = null;
    var iters: u32 = 0;
    while (fx.sched.tick(fx.sink) != .idle) {
        iters += 1;
        try std.testing.expect(iters <= 100);
    }

    const toks_a = tokenIds(fx.sink.seen(), sa);
    const toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks_a[0..6]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks_b[0..6]);
}

test "park: a parked prefill whose decode retry fails otherwise fails only the decode rows" {
    var fx = try RefusingFixture.setUp(2);
    defer fx.tearDown();

    const sa = try parkOpen(&fx, 1);
    const sb = try parkOpen(&fx, 2);
    try parkGenerate(&fx, 1, sa, "x", 6);
    try std.testing.expectEqual(Tick.worked, fx.sched.tick(fx.sink)); // a decodes
    try parkGenerate(&fx, 2, sb, "xxxx", 6);

    fx.pool.fail_with_feed = .park;
    fx.pool.fail_decode_only = .other;
    try std.testing.expectEqual(Tick.worked, fx.sched.tick(fx.sink));

    // The decode row of a fails with an error event. The prefill of b stays parked.
    const last = fx.sink.last();
    try std.testing.expectEqual(sa, last.ev.session_id);
    try std.testing.expectEqual(protocol.ErrorCode.internal, last.ev.body.err.code);

    fx.pool.fail_with_feed = null;
    fx.pool.fail_decode_only = null;
    var iters: u32 = 0;
    while (fx.sched.tick(fx.sink) != .idle) {
        iters += 1;
        try std.testing.expect(iters <= 100);
    }

    // The turn of b streams in full and carries no error.
    const toks_b = tokenIds(fx.sink.seen(), sb);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, 6 }, toks_b[0..6]);
    for (fx.sink.seen()) |e| {
        if (e.ev.session_id == sb) {
            try std.testing.expect(e.ev.body != .err);
        }
    }
}

test "lease: an open that the engine refuses after a yield reports the lost session" {
    var fx = try RefusingFixture.setUp(1);
    defer fx.tearDown();

    try fx.open(1);
    _ = try fx.sched.keepLease(1, 1, 50, tok(7));
    fx.sched.detachConn(1, atMs(10));

    fx.pool.refuse = true;
    try fx.open(2);
    try std.testing.expectEqual(
        protocol.ErrorCode.sessions_exhausted,
        fx.sink.last().ev.body.err.code,
    );
    try std.testing.expectEqual(SessionPhase.free, fx.sched.entries[0].state);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_yields);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_yields_lost);
    try std.testing.expectError(
        error.NoSuchLease,
        fx.sched.resumeLease(3, 5, tok(7), atMs(20), tok(8)),
    );

    // The refused open took no session id.
    fx.pool.refuse = false;
    try fx.open(2);
    try std.testing.expectEqual(@as(u32, 2), fx.sink.last().ev.session_id);
}

test "lease: a clone that the engine refuses after a yield reports the lost session" {
    var fx = try RefusingFixture.setUp(2);
    defer fx.tearDown();

    try fx.open(2);
    try fx.open(1);
    _ = try fx.sched.keepLease(2, 1, 50, tok(7));
    fx.sched.detachConn(1, atMs(10));

    fx.pool.refuse = true;
    _ = fx.sched.handle(.{ .conn = 2, .request_id = 3, .session_id = 1, .body = .{
        .session_clone = .{ .sampling = protocol.Sampling.greedy },
    } }, fx.sink);
    const refused = fx.sink.last().ev.body.err;
    try std.testing.expectEqual(protocol.ErrorCode.sessions_exhausted, refused.code);
    try std.testing.expectEqualStrings(
        "kv ram budget refused the clone",
        refused.detail[0..refused.detail_len],
    );
    try std.testing.expectEqual(SessionPhase.free, fx.sched.entries[1].state);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_yields_lost);
    try std.testing.expectEqual(SessionPhase.idle, fx.sched.entries[0].state);
}

// -- wire handler tests -----------------------------------------------------
//
// These tests drive the lease and the per-turn fields through `handle`, the
// path of a decoded frame. The tokens come from the secure source, so a test
// reads each one back from the reply that carries it.

/// Send one request on `conn` and return the last event of the sink.
fn request(fx: *Fixture, conn: u32, request_id: u32, sid: u32, body: Request.Body) FakeSink.Seen {
    _ = fx.sched.handle(
        .{ .conn = conn, .request_id = request_id, .session_id = sid, .body = body },
        fx.sink,
    );
    return fx.sink.last();
}

fn expectErr(e: FakeSink.Seen, code: protocol.ErrorCode, detail: []const u8) !void {
    try std.testing.expectEqual(OutEvent.Body.err, std.meta.activeTag(e.ev.body));
    try std.testing.expectEqual(code, e.ev.body.err.code);
    try std.testing.expectEqualStrings(detail, e.ev.body.err.detail[0..e.ev.body.err.detail_len]);
}

/// Send a `generate` with every per-turn field of the wire.
fn generateWith(
    fx: *Fixture,
    conn: u32,
    sid: u32,
    text: []const u8,
    max_tokens: u32,
    sampling: ?protocol.Sampling,
    think_open: ?bool,
    think_budget: ?u32,
) !void {
    const dup = try std.testing.allocator.dupe(u8, text);
    _ = fx.sched.handle(
        .{ .conn = conn, .request_id = 9, .session_id = sid, .body = .{ .generate = .{
            .max_tokens = max_tokens,
            .sampling = sampling,
            .text = dup,
            .think_open = think_open,
            .think_budget = think_budget,
        } } },
        fx.sink,
    );
}

test "wire lease: keep, drop, resume, a replayed token, and an expired token" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    _ = try runTurn(&fx, 1, sid, "x", 2);

    // The refusals of a keep: no such session, another connection.
    try expectErr(
        request(&fx, 1, 20, 999, .{ .session_keep = .{ .ttl_ms = 50 } }),
        .unknown_session,
        "no such session",
    );
    try expectErr(
        request(&fx, 2, 21, sid, .{ .session_keep = .{ .ttl_ms = 50 } }),
        .unknown_session,
        "no such session",
    );

    // A keep past the maximum gets the maximum and a token from the source.
    var e = request(&fx, 1, 22, sid, .{ .session_keep = .{ .ttl_ms = 500 } });
    try std.testing.expectEqual(@as(u32, 22), e.ev.request_id);
    try std.testing.expectEqual(sid, e.ev.session_id);
    try std.testing.expectEqual(@as(u32, 100), e.ev.body.kept.granted_ms);
    const first = e.ev.body.kept.token;
    try std.testing.expect(!std.mem.allEqual(u8, &first, 0));
    // A second keep issues a new token, and the first one opens nothing.
    e = request(&fx, 1, 23, sid, .{ .session_keep = .{ .ttl_ms = 50 } });
    try std.testing.expectEqual(@as(u32, 50), e.ev.body.kept.granted_ms);
    const kept = e.ev.body.kept.token;
    try std.testing.expect(!std.mem.eql(u8, &first, &kept));

    // The drop at 10 detaches the session until 60.
    fx.sched.pinClock(atMs(10));
    _ = fx.sched.handle(
        .{ .conn = 1, .request_id = 0, .session_id = 0, .body = .conn_closed },
        fx.sink,
    );
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .detached = 60 }, .pid = sid, .token = kept[0], .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );
    try expectErr(
        request(&fx, 3, 24, 0, .{ .session_resume = .{ .token = first } }),
        .unknown_session,
        "no such lease",
    );

    // A resume on connection 2 at 20 is ready at once, with a new token.
    fx.sched.pinClock(atMs(20));
    e = request(&fx, 2, 25, 0, .{ .session_resume = .{ .token = kept } });
    try std.testing.expectEqual(@as(u32, 2), e.conn);
    try std.testing.expectEqual(@as(u32, 25), e.ev.request_id);
    try std.testing.expectEqual(sid, e.ev.session_id);
    const r = e.ev.body.resumed;
    try std.testing.expect(!std.mem.eql(u8, &kept, &r.token));
    try std.testing.expectEqual(fx.pool.posOf(fx.sched.entries[0].engine_id), r.kv_pos);
    try std.testing.expectEqual(@as(u32, 1), r.turns_done);
    try std.testing.expectEqual(
        protocol.LastDone{
            .request_id = 9,
            .finish = .max_tokens,
            .gen_tokens = 2,
            .reasoning_tokens = 0,
        },
        r.last.?,
    );

    // A replay of the presented token opens nothing.
    try expectErr(
        request(&fx, 3, 26, 0, .{ .session_resume = .{ .token = kept } }),
        .unknown_session,
        "no such lease",
    );

    // A drop at 30 sets the deadline to 80, and a resume at 80 is too late.
    fx.sched.pinClock(atMs(30));
    _ = fx.sched.handle(
        .{ .conn = 2, .request_id = 0, .session_id = 0, .body = .conn_closed },
        fx.sink,
    );
    fx.sched.pinClock(atMs(80));
    try expectErr(
        request(&fx, 3, 27, 0, .{ .session_resume = .{ .token = r.token } }),
        .unknown_session,
        "no such lease",
    );
    try std.testing.expectEqual(SessionPhase.free, fx.sched.entries[0].state);
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_expires);
}

test "wire lease: resumption off, a zero time to live, a waiting resume, and a full share" {
    var off = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer off.tearDown();

    const s = try off.openSession(1, "");
    try expectErr(
        request(&off, 1, 20, s, .{ .session_keep = .{ .ttl_ms = 50 } }),
        .forbidden,
        "resumption is off",
    );
    // A zero time to live always succeeds, and it carries no token.
    var e = request(&off, 1, 21, s, .{ .session_keep = .{ .ttl_ms = 0 } });
    try std.testing.expectEqual(@as(u32, 0), e.ev.body.kept.granted_ms);
    try std.testing.expect(std.mem.allEqual(u8, &e.ev.body.kept.token, 0));
    try expectErr(
        request(&off, 2, 22, 0, .{ .session_resume = .{ .token = tok(7) } }),
        .unknown_session,
        "no such lease",
    );

    var fx = try leaseFixture(2, 1);
    defer fx.tearDown();

    const sid = try appendInFlight(&fx);
    // The resume waits for the append, and a keep on it is busy.
    e = request(&fx, 2, 30, 0, .{ .session_resume = .{ .token = tok(7) } });
    try std.testing.expect(fx.sched.entries[0].resume_wait != null);
    try expectWaitRefusal(request(&fx, 2, 31, sid, .{ .session_keep = .{ .ttl_ms = 50 } }));
    fx.drain();
    try std.testing.expectEqual(OutEvent.Body.resumed, std.meta.activeTag(fx.sink.last().ev.body));
    try std.testing.expectEqual(@as(u32, 30), fx.sink.last().ev.request_id);
    const token = fx.sink.last().ev.body.resumed.token;

    // Connection 3 holds its share of one, so it cannot take the session.
    _ = try fx.openSession(3, "");
    try expectErr(
        request(&fx, 3, 32, 0, .{ .session_resume = .{ .token = token } }),
        .sessions_exhausted,
        "session share of this connection is full",
    );
    // A close releases the kept session and its lease.
    e = request(&fx, 2, 33, sid, .session_close);
    try std.testing.expectEqual(OutEvent.Body.session_closed, std.meta.activeTag(e.ev.body));
    try expectErr(
        request(&fx, 4, 34, 0, .{ .session_resume = .{ .token = token } }),
        .unknown_session,
        "no such lease",
    );
}

test "per-turn budget: a request budget forces the close at its count, not at the server value" {
    // The server value is the budget 2 of `think_tags`. The prompt "3" makes
    // the first token 4. The turn model replays the same two turns.
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, think_tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    try expectThinkTurn(&fx, sid, .{
        .open = true,
        .prompt = "3",
        .fed = &.{3},
        .want = &.{ 2, 4, 5, 3, 0, 1, 2, 3, 4, 5, 6, 7 },
        .reason = .eos,
        .reasoning = 2,
    });

    const want_budget_3 = [_]u32{ 2, 4, 5, 6, 3, 0, 1, 2, 3, 4, 5, 6, 7 };
    const cases = [_]struct { budget: u32, want: []const u32, reasoning: u32, open_after: bool }{
        .{ .budget = 3, .want = &want_budget_3, .reasoning = 3, .open_after = false },
        .{ .budget = 0, .want = &.{ 2, 4, 5, 6, 7 }, .reasoning = 4, .open_after = true },
    };
    for (cases) |c| {
        const from = fx.sink.n;
        try generateWith(&fx, 1, sid, "3", 64, null, null, c.budget);
        fx.drain();
        try std.testing.expectEqual(c.budget, fx.sched.entries[0].think_budget);
        var ids: [16]u32 = undefined;
        var n: usize = 0;
        for (fx.sink.seen()[from..]) |e| {
            if (e.ev.body != .token) continue;
            ids[n] = e.ev.body.token.token_id;
            n += 1;
        }

        try std.testing.expectEqualSlices(u32, c.want, ids[0..n]);
        try std.testing.expectEqual(c.reasoning, fx.sink.last().ev.body.done.reasoning_tokens);
        try std.testing.expectEqual(c.open_after, fx.sched.entries[0].think_open);
    }

    // A turn with no budget takes the server value again.
    try generateWith(&fx, 1, sid, "3", 64, null, null, null);
    fx.drain();
    try std.testing.expectEqual(@as(u32, 2), fx.sched.entries[0].think_budget);
}

test "per-turn budget: a request budget forces the close when the server budget is zero" {
    var tags = think_tags;
    tags.budget = 0;
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, tags);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    const budget: u32 = 3;
    const from = fx.sink.n;
    try generateWith(&fx, 1, sid, "3", 64, null, null, budget);
    fx.drain();
    try std.testing.expectEqual(budget, fx.sched.entries[0].think_budget);
    var ids: [16]u32 = undefined;
    var n: usize = 0;
    for (fx.sink.seen()[from..]) |e| {
        if (e.ev.body != .token) continue;
        ids[n] = e.ev.body.token.token_id;
        n += 1;
    }

    // The echo of the open is event 0, so the close is event budget + 1.
    try std.testing.expect(n > budget + 1);
    try std.testing.expectEqual(tags.close_id, ids[budget + 1]);
    try std.testing.expectEqualSlices(u32, &.{ 2, 4, 5, 6, 3, 0, 1, 2, 3, 4, 5, 6, 7 }, ids[0..n]);
    try std.testing.expectEqual(budget, fx.sink.last().ev.body.done.reasoning_tokens);
}

test "per-turn switch: the render and the session read the one resolved value" {
    var tags = think_tags;
    tags.budget = 0;
    var fx = try Fixture.setUpThink(std.testing.allocator, 65, 0, tags);
    defer fx.tearDown();

    var opens: FakeCodec.OpenLog = .{};
    fx.codec.opens = &opens;
    const sid = try fx.openSession(1, "");
    try std.testing.expectEqual(@as(usize, 0), opens.n);

    // The server mode is open. A closed switch renders closed, and the turn
    // has no echo.
    const cases = [_]struct { open: ?bool, resolved: bool, first: u32 }{
        .{ .open = false, .resolved = false, .first = 5 },
        .{ .open = true, .resolved = true, .first = 2 },
        .{ .open = null, .resolved = true, .first = 2 },
    };
    for (cases, 0..) |c, i| {
        const from = fx.sink.n;
        try generateWith(&fx, 1, sid, "4", 2, null, c.open, null);
        try std.testing.expectEqual(c.resolved, opens.seen[i]);
        try std.testing.expectEqual(c.resolved, fx.sched.entries[0].think_open);
        fx.drain();
        try std.testing.expectEqual(c.first, fx.sink.seen()[from].ev.body.token.token_id);
    }

    try std.testing.expectEqual(cases.len, opens.n);
}

test "per-turn guards: a budget or an open switch needs tags, a closed switch passes" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    for ([_]?u32{ 0, 5 }) |budget| {
        try generateWith(&fx, 1, sid, "4", 4, null, null, budget);
        try expectErr(fx.sink.last(), .forbidden, "this model has no think tags");
    }

    try generateWith(&fx, 1, sid, "4", 4, null, true, null);
    try expectErr(fx.sink.last(), .forbidden, "this model has no think tags");
    try std.testing.expectEqual(SessionPhase.idle, fx.sched.entries[0].state);
    const from = fx.sink.n;
    try generateWith(&fx, 1, sid, "4", 4, null, false, null);
    fx.drain();
    try std.testing.expectEqualSlices(
        u32,
        &.{ 5, 6, 7 },
        tokenIds(fx.sink.seen()[from..], sid)[0..3],
    );
}

/// Send a sampled `generate` with or without the reseed bit.
fn generateSampled(
    fx: *Fixture,
    sid: u32,
    text: []const u8,
    max_tokens: u32,
    sampling: protocol.Sampling,
    reseed: bool,
) !void {
    const dup = try std.testing.allocator.dupe(u8, text);
    _ = fx.sched.handle(.{ .conn = 1, .request_id = 9, .session_id = sid, .body = .{ .generate = .{
        .max_tokens = max_tokens,
        .sampling = sampling,
        .text = dup,
        .reseed = reseed,
    } } }, fx.sink);
}

test "per-turn sampling: only reseed restarts the stream, min_p and repetition land" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 64, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    const eid = fx.sched.entries[0].engine_id;
    // A sampled turn moves the stream of the open seed 0.
    var hot = protocol.Sampling.greedy;
    hot.temperature = 1;
    try generateSampled(&fx, sid, "x", 3, hot, false);
    fx.drain();
    const moved = fx.pool.table[eid].rng.s;
    try std.testing.expect(!std.meta.eql(std.Random.DefaultPrng.init(0).s, moved));

    // A greedy turn with seed 41 and no reseed bit draws nothing and keeps
    // the stream. The penalties still land in the params of the session.
    var s = protocol.Sampling.greedy;
    s.seed = 41;
    s.min_p = 0.25;
    s.repetition_penalty = 1.5;
    try generateSampled(&fx, sid, "x", 2, s, false);
    fx.drain();
    try std.testing.expectEqual(moved, fx.pool.table[eid].rng.s);
    try std.testing.expectEqual(@as(f32, 0.25), fx.pool.table[eid].params.min_p);
    try std.testing.expectEqual(@as(f32, 1.5), fx.pool.table[eid].params.repetition_penalty);

    // The same turn with the reseed bit restarts the stream at 41.
    try generateSampled(&fx, sid, "x", 2, s, true);
    fx.drain();
    try std.testing.expectEqual(std.Random.DefaultPrng.init(41).s, fx.pool.table[eid].rng.s);

    // A turn with no sampling keeps the stream and the params.
    try generateWith(&fx, 1, sid, "x", 2, null, null, null);
    fx.drain();
    try std.testing.expectEqual(std.Random.DefaultPrng.init(41).s, fx.pool.table[eid].rng.s);
    try std.testing.expectEqual(@as(f32, 0.25), fx.pool.table[eid].params.min_p);
}

test "per-turn sampling: a refused turn leaves the params and the stream of the session unchanged" {
    var fx = try Fixture.setUp(std.testing.allocator, 1, 8, null);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "");
    const eid = fx.sched.entries[0].engine_id;
    const params = fx.pool.table[eid].params;
    const rng = fx.pool.table[eid].rng.s;
    var s = protocol.Sampling.greedy;
    s.temperature = 0.5;
    s.seed = 99;
    s.min_p = 0.5;
    // Nine tokens do not fit the window of eight.
    try generateSampled(&fx, sid, "xxxxxxxxx", 2, s, true);
    try std.testing.expectEqual(protocol.ErrorCode.turn_too_long, fx.sink.last().ev.body.err.code);
    try std.testing.expectEqual(rng, fx.pool.table[eid].rng.s);
    try std.testing.expectEqual(params.temperature, fx.pool.table[eid].params.temperature);
    try std.testing.expectEqual(params.min_p, fx.pool.table[eid].params.min_p);
    // A frequency penalty refuses the turn before any change too.
    s.frequency_penalty = 0.5;
    try generateSampled(&fx, sid, "x", 2, s, true);
    try expectErr(fx.sink.last(), .forbidden, "frequency_penalty is not supported");
    try std.testing.expectEqual(rng, fx.pool.table[eid].rng.s);
    try std.testing.expectEqual(params.min_p, fx.pool.table[eid].params.min_p);
}

test "frequency penalty: a nonzero value fails the request, and the session lives" {
    var fx = try Fixture.setUp(std.testing.allocator, 2, 64, null);
    defer fx.tearDown();

    var s = protocol.Sampling.greedy;
    s.frequency_penalty = -1;
    const dup = try std.testing.allocator.dupe(u8, "");
    try expectErr(
        request(&fx, 1, 20, 0, .{ .session_open = .{ .sampling = s, .system_prompt = dup } }),
        .forbidden,
        "frequency_penalty is not supported",
    );
    try std.testing.expectEqual(SessionPhase.free, fx.sched.entries[0].state);
    const sid = try fx.openSession(1, "");
    try expectErr(
        request(&fx, 1, 21, sid, .{ .session_clone = .{ .sampling = s } }),
        .forbidden,
        "frequency_penalty is not supported",
    );
    try std.testing.expectEqual(SessionPhase.free, fx.sched.entries[1].state);
    try std.testing.expectEqual(@as(?u32, null), fx.sink.disconnected);
    // The session still streams on the same connection.
    const t = try runTurn(&fx, 1, sid, "x", 2);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, t.tokens());
}

test "wire lease: a close on a waiting resume is busy, and the close after the reply closes" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try appendInFlight(&fx);
    _ = request(&fx, 2, 30, 0, .{ .session_resume = .{ .token = tok(7) } });
    try std.testing.expect(fx.sched.entries[0].resume_wait != null);
    try expectWaitRefusal(request(&fx, 2, 31, sid, .session_close));
    try std.testing.expect(fx.sched.entries[0].state != .free);
    fx.drain();
    try std.testing.expectEqual(OutEvent.Body.resumed, std.meta.activeTag(fx.sink.last().ev.body));
    const e = request(&fx, 2, 32, sid, .session_close);
    try std.testing.expectEqual(OutEvent.Body.session_closed, std.meta.activeTag(e.ev.body));
    try std.testing.expectEqual(SessionPhase.free, fx.sched.entries[0].state);
}

test "wire lease: a drop before the resume reply strands the lease until its time to live" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try appendInFlight(&fx);
    fx.sched.detachConn(1, atMs(0));
    // The resume rotates the token and waits for the append. Its connection
    // drops at 20, before the `resumed` reply goes out.
    fx.sched.pinClock(atMs(10));
    _ = request(&fx, 2, 30, 0, .{ .session_resume = .{ .token = tok(7) } });
    try std.testing.expect(fx.sched.entries[0].resume_wait != null);
    const rotated = fx.sched.entries[0].lease.?.token;
    try std.testing.expect(!std.mem.eql(u8, &rotated, &tok(7)));
    fx.sched.pinClock(atMs(20));
    _ = fx.sched.handle(
        .{ .conn = 2, .request_id = 0, .session_id = 0, .body = .conn_closed },
        fx.sink,
    );
    const seen = fx.sink.n;
    fx.drain();
    // No client holds a valid token: the old one is no longer valid, and the new one
    // never left the server.
    try std.testing.expectEqual(seen, fx.sink.n);
    try expectErr(
        request(&fx, 3, 31, 0, .{ .session_resume = .{ .token = tok(7) } }),
        .unknown_session,
        "no such lease",
    );
    try std.testing.expectEqual(
        EntryView{ .owner = .{ .detached = 70 }, .pid = sid, .token = rotated[0], .ttl = 50 },
        viewOf(&fx.sched.entries[0]),
    );
    // The entry holds until its time to live ends, then frees.
    fx.sched.expireDetached(atMs(69));
    try std.testing.expect(fx.sched.entries[0].state != .free);
    fx.sched.pinClock(atMs(70));
    _ = fx.sched.tick(fx.sink);
    try std.testing.expectEqual(free_view, viewOf(&fx.sched.entries[0]));
    try std.testing.expectEqual(@as(u64, 1), fx.sched.lease_expires);
}

// -- the lease trace property -----------------------------------------------
//
// One random trace drives the wire surface: open, generate, cancel, keep,
// drop, resume, close, clock steps, and ticks. The checks after each step
// state the lease rules and the stream rules. A dropped connection id never
// returns, as on the server.

const trace_steps = 96;
const trace_conns = 3;
const trace_request_base: u32 = 100;
const trace_ttl_max_ms = 100;

/// What the trace knows of one session from the wire alone: the token the
/// client holds, the lease it asked for, and the deadline that its own clock
/// implies. It reads no entry of the scheduler under test.
const SessionModel = struct {
    /// The connection that owns the session, or null when it is detached.
    conn: ?u32 = null,
    /// The token that a client holds. Null while a resume waits, and after the lease ends.
    token: ?Token = null,
    ttl_ms: u32 = 0,
    deadline: ?i64 = null,
    pending_resume: bool = false,
    gone: bool = false,
    /// The `session_opened` reply came. A guessed id can name a session in its opening prefill.
    opened: bool = false,
    /// Open requests at the detach. A later open can take the entry, so the model cannot tell.
    opens_at_detach: u32 = 0,
    /// The `kv_pos` of the last `resumed` reply.
    last_pos: u64 = 0,
};

const LeaseTrace = struct {
    fx: Fixture,
    now: i64 = 0,
    conns: [trace_conns]u32 = .{ 1, 2, 3 },
    next_conn: u32 = trace_conns + 1,
    next_request: u32 = trace_request_base,
    /// The requests that saw their `done` or their `err`.
    finished: [trace_steps]bool = @splat(false),
    /// Each step opens at most one session, so the distinct ids fit.
    sids: [trace_steps]u32 = undefined,
    models: [trace_steps]SessionModel = @splat(.{}),
    n_sids: usize = 0,
    /// The connection of each request, to name the owner of an opened session.
    req_conn: [trace_steps]u32 = @splat(0),
    /// An open holds an entry until its `session_opened`, so it counts against the share.
    open_pending: [trace_steps]bool = @splat(false),
    /// The id that each admitted open takes, or 0 for an open refused at once. Ids come from a
    /// counter in admission order, so a reply to a guessed id names its open.
    open_sid: [trace_steps]u32 = @splat(0),
    next_sid: u32 = 1,
    n_opens: u32 = 0,
    /// The session that a resume with a live token must reach.
    req_sid: [trace_steps]?u32 = @splat(null),
    tokens: [16]Token = undefined,
    n_tokens: usize = 0,

    fn request(t: *LeaseTrace, conn: u32, sid: u32, body: Request.Body) void {
        t.req_conn[t.next_request - trace_request_base] = conn;
        t.open_pending[t.next_request - trace_request_base] = body == .session_open;
        t.n_opens += @intFromBool(body == .session_open);
        const open = body == .session_open;
        _ = t.fx.sched.handle(
            .{ .conn = conn, .request_id = t.next_request, .session_id = sid, .body = body },
            t.fx.sink,
        );
        if (open) {
            // A refusal of an open comes at once, so an open with no reply took an id.
            const refused = for (t.fx.sink.seen()) |e| {
                if (e.ev.request_id == t.next_request and e.ev.body == .err) break true;
            } else false;

            if (!refused) {
                t.open_sid[t.next_request - trace_request_base] = t.next_sid;
                t.next_sid += 1;
            }
        }

        t.next_request += 1;
    }

    /// The opens of `conn` before request `index` that wait for `session_opened`. A reply to a
    /// guessed id can name the session of an open first, and the model then counts it.
    fn pendingOpens(t: *const LeaseTrace, conn: u32, index: usize) usize {
        var n: usize = 0;
        for (t.open_pending[0..index], t.req_conn[0..index], t.open_sid[0..index]) |p, c, sid| {
            n += @intFromBool(p and c == conn and !t.knows(sid));
        }

        return n;
    }

    fn knows(t: *const LeaseTrace, sid: u32) bool {
        return std.mem.indexOfScalar(u32, t.sids[0..t.n_sids], sid) != null;
    }

    fn modelOf(t: *LeaseTrace, sid: u32) *SessionModel {
        for (t.sids[0..t.n_sids], 0..) |known, i| {
            if (known == sid) return &t.models[i];
        }

        unreachable;
    }

    /// A resume of `token` from `conn`. A full share refuses first, then an unknown, replayed, or
    /// expired token. Any other token reaches its session and rotates. `absorbEvents` checks a
    /// reply that waits.
    fn resumeWith(t: *LeaseTrace, conn: u32, token: Token) !void {
        // Predict the answer from the model alone.
        var attached: usize = 0;
        var found: ?u32 = null;
        for (t.sids[0..t.n_sids], t.models[0..t.n_sids]) |sid, *m| {
            if (m.gone) continue;

            if (m.conn != null and m.conn.? == conn) {
                attached += 1;
            }

            const live = m.deadline == null or m.deadline.? > t.now;
            if (live and m.token != null and std.mem.eql(u8, &m.token.?, &token)) {
                found = sid;
            }
        }

        const rid = t.next_request;
        const index = rid - trace_request_base;
        attached += t.pendingOpens(conn, index);
        const full = attached >= t.fx.sched.cfg.sessions_per_connection;

        // Send the resume, and check any immediate reply.
        t.request(conn, 0, .{ .session_resume = .{ .token = token } });
        var replies: usize = 0;
        for (t.fx.sink.seen()) |e| {
            if (e.ev.request_id != rid) continue;
            replies += 1;
            if (full or found == null) {
                try std.testing.expectEqual(OutEvent.Body.err, std.meta.activeTag(e.ev.body));
                const want: protocol.ErrorCode = if (full)
                    .sessions_exhausted
                else
                    .unknown_session;

                try std.testing.expectEqual(want, e.ev.body.err.code);
            } else if (e.ev.body == .err) {
                // Only a yield to a later open takes a live lease away.
                const m = t.modelOf(found.?);
                try std.testing.expectEqual(protocol.ErrorCode.unknown_session, e.ev.body.err.code);
                try std.testing.expect(m.deadline != null and t.n_opens > m.opens_at_detach);
                m.gone = true;
                return;
            } else {
                try std.testing.expectEqual(OutEvent.Body.resumed, std.meta.activeTag(e.ev.body));
                try std.testing.expectEqual(found.?, e.ev.session_id);
            }
        }

        if (full or found == null) {
            try std.testing.expectEqual(@as(usize, 1), replies);
            return;
        }

        // Record the rotation. The new token reaches the model with the `resumed` reply.
        try std.testing.expect(replies <= 1);
        const m = t.modelOf(found.?);
        m.conn = conn;
        m.deadline = null;
        m.token = null;
        m.pending_resume = true;
        t.req_sid[index] = found;
    }

    /// No token and no second `done` reaches a request after its finish.
    /// The events also name the sessions and the tokens that later steps
    /// pick from.
    fn absorbEvents(t: *LeaseTrace) !void {
        for (t.fx.sink.seen()) |e| {
            const rid = e.ev.request_id;
            const tracked = rid >= trace_request_base and rid < t.next_request;
            switch (e.ev.body) {
                .token => {
                    try std.testing.expect(tracked);
                    try std.testing.expect(!t.finished[rid - trace_request_base]);
                },
                .done => {
                    try std.testing.expect(tracked);
                    try std.testing.expect(!t.finished[rid - trace_request_base]);
                    // An admitted open ends with `session_opened` or `err`, and only a turn
                    // ends with `done`.
                    try std.testing.expectEqual(@as(u32, 0), t.open_sid[rid - trace_request_base]);
                    t.finished[rid - trace_request_base] = true;
                },
                .err => if (tracked) {
                    t.finished[rid - trace_request_base] = true;
                    t.open_pending[rid - trace_request_base] = false;
                },
                .session_opened => {
                    const index = rid - trace_request_base;
                    try std.testing.expectEqual(t.open_sid[index], e.ev.session_id);
                    t.open_pending[index] = false;
                    if (!t.knows(e.ev.session_id)) {
                        t.addSid(e.ev.session_id);
                        t.modelOf(e.ev.session_id).conn = t.req_conn[rid - trace_request_base];
                    }

                    t.modelOf(e.ev.session_id).opened = true;
                },
                .kept => |k| {
                    if (!t.knows(e.ev.session_id)) {
                        t.addSid(e.ev.session_id);
                        t.modelOf(e.ev.session_id).conn = t.req_conn[rid - trace_request_base];
                    }

                    const m = t.modelOf(e.ev.session_id);
                    try std.testing.expect(!m.gone and !m.pending_resume);
                    m.token = if (k.granted_ms > 0) k.token else null;
                    m.ttl_ms = k.granted_ms;
                    if (k.granted_ms > 0) {
                        t.addToken(k.token);
                    }
                },
                .session_closed => {
                    if (!t.knows(e.ev.session_id)) {
                        t.addSid(e.ev.session_id);
                        t.modelOf(e.ev.session_id).conn = t.req_conn[rid - trace_request_base];
                    }

                    const m = t.modelOf(e.ev.session_id);
                    if (!m.opened) {
                        // The close hit an opening prefill, so its `session_opened` never comes.
                        const used = t.next_request - trace_request_base;
                        for (t.open_pending[0..used], t.open_sid[0..used]) |*pending, sid| {
                            if (sid == e.ev.session_id) {
                                pending.* = false;
                            }
                        }
                    }

                    m.* = .{ .gone = true };
                },
                .resumed => |r| {
                    try std.testing.expect(tracked);
                    const m = t.modelOf(e.ev.session_id);
                    try std.testing.expect(m.pending_resume);
                    try std.testing.expectEqual(
                        @as(?u32, e.ev.session_id),
                        t.req_sid[rid - trace_request_base],
                    );
                    // The position of a session never falls across a resume.
                    try std.testing.expect(r.kv_pos >= m.last_pos);
                    m.last_pos = r.kv_pos;
                    m.pending_resume = false;
                    m.token = r.token;
                    t.addToken(r.token);
                },
                else => {},
            }
        }

        t.fx.sink.n = 0;
    }

    fn addSid(t: *LeaseTrace, sid: u32) void {
        for (t.sids[0..t.n_sids]) |known| {
            if (known == sid) return;
        }

        std.debug.assert(t.n_sids < t.sids.len);
        t.sids[t.n_sids] = sid;
        t.models[t.n_sids] = .{};
        t.n_sids += 1;
    }

    fn addToken(t: *LeaseTrace, token: Token) void {
        t.tokens[t.n_tokens % t.tokens.len] = token;
        t.n_tokens += 1;
    }

    fn pickSid(t: *const LeaseTrace, r: std.Random) u32 {
        if (t.n_sids == 0 or r.uintLessThan(u8, 16) == 0) return r.uintAtMost(u32, 2);
        return t.sids[r.uintLessThan(usize, t.n_sids)];
    }

    fn pickConn(t: *const LeaseTrace, r: std.Random) u32 {
        return t.conns[r.uintLessThan(usize, trace_conns)];
    }

    fn pinAt(t: *LeaseTrace, ms: i64) void {
        std.debug.assert(ms >= t.now);

        t.now = ms;
        t.fx.sched.pinClock(atMs(ms));
    }

    /// The earliest deadline of a detached entry.
    fn earliestDeadline(t: *const LeaseTrace) ?i64 {
        var best: ?i64 = null;
        for (t.fx.sched.entries) |*entry| {
            if (entry.state == .free or entry.owner != .detached) continue;
            const d = entry.owner.detached.raw.toMilliseconds();
            if (best == null or d < best.?) {
                best = @intCast(d);
            }
        }

        return best;
    }

    /// The live sessions fit the entries, and no connection holds more than
    /// its share. No token and no session id sits on two entries. A free entry
    /// holds nothing, and a detached entry holds a lease.
    fn expectTable(t: *const LeaseTrace) !void {
        const entries = t.fx.sched.entries;
        var live: usize = 0;
        for (entries, 0..) |*a, i| {
            if (a.state == .free) {
                try std.testing.expect(a.lease == null);
                try std.testing.expectEqual(@as(u32, 0), a.proto_id);
                continue;
            }

            live += 1;
            if (a.owner == .detached) {
                try std.testing.expect(a.lease != null);
            }

            if (a.owner == .attached) {
                var share: usize = 0;
                for (entries) |*b| {
                    share += @intFromBool(
                        b.state != .free and b.owner == .attached and b.owner.attached ==
                            a.owner.attached,
                    );
                }

                try std.testing.expect(share <= t.fx.sched.cfg.sessions_per_connection);
            }

            for (entries[i + 1 ..]) |*b| {
                if (b.state == .free) continue;
                try std.testing.expect(a.proto_id != b.proto_id);
                if (a.lease != null and b.lease != null) {
                    try std.testing.expect(!std.mem.eql(u8, &a.lease.?.token, &b.lease.?.token));
                }
            }
        }

        try std.testing.expect(live <= t.fx.sched.cfg.max_sessions);
    }

    /// One tick, then no detached entry sits at or past its deadline.
    fn tickOnce(t: *LeaseTrace) !void {
        _ = t.fx.sched.tick(t.fx.sink);
        if (t.earliestDeadline()) |d| {
            try std.testing.expect(d > t.now);
        }
    }

    /// A drop detaches each kept session of `conn`, with a deadline of its time to live after the
    /// clock. It frees each other session of `conn`.
    fn drop(t: *LeaseTrace, k: usize) !void {
        // Record what each entry of the connection holds.
        const conn = t.conns[k];
        const entries = t.fx.sched.entries;
        var kept: [limits.max_vram_slots]?u32 = @splat(null);
        for (entries, 0..) |*entry, i| {
            if (entry.state == .free or entry.owner != .attached or entry.owner.attached != conn) {
                continue;
            }

            kept[i] = if (entry.lease) |l| l.ttl_ms else 0;
        }

        // Drop the connection. A session without a lease frees, even in its opening prefill.
        t.request(conn, 0, .conn_closed);
        const requests = t.next_request - trace_request_base;
        for (t.open_pending[0..requests], t.req_conn[0..requests]) |*pending, c| {
            if (c == conn) {
                pending.* = false;
            }
        }

        // Update the model. A waiting resume dies with its connection and its new token. The lease
        // still runs, and a token that the client holds stays valid.
        for (t.models[0..t.n_sids]) |*m| {
            if (m.gone or m.conn == null or m.conn.? != conn) continue;

            const has_lease = m.token != null or m.pending_resume;
            m.conn = null;
            m.pending_resume = false;
            m.opens_at_detach = t.n_opens;
            if (has_lease) {
                m.deadline = t.now + @as(i64, m.ttl_ms);
            } else {
                m.gone = true;
            }
        }

        // Each kept entry detached at its deadline, and every other entry is free.
        for (entries, 0..) |*entry, i| {
            const ttl = kept[i] orelse continue;
            if (ttl == 0) {
                try std.testing.expectEqual(SessionPhase.free, entry.state);
            } else {
                try std.testing.expect(entry.owner == .detached);
                try std.testing.expectEqual(
                    t.now + @as(i64, ttl),
                    entry.owner.detached.raw.toMilliseconds(),
                );
            }
        }

        // A dropped connection id never returns.
        t.conns[k] = t.next_conn;
        t.next_conn += 1;
    }
};

/// A random wire trace keeps the lease table and the streams sound. Live sessions fit the entries
/// and the shares. No token sits on two entries, and no token event follows a finish. No open
/// ends with `done`. Every lease ends within its time to live plus one tick.
fn checkLeaseTrace(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const n_slots = 2 + r.uintLessThan(u32, 2);
    var t: LeaseTrace = .{
        .fx = try leaseFixture(n_slots, @intCast(1 + r.uintLessThan(u32, n_slots))),
    };
    defer t.fx.tearDown();

    const gpa = std.testing.allocator;

    // Run one random step, a few ticks, and the checks.
    for (0..trace_steps) |_| {
        switch (r.uintLessThan(u8, 9)) {
            0 => t.request(t.pickConn(r), 0, .{ .session_open = .{
                .sampling = protocol.Sampling.greedy,
                .system_prompt = try gpa.dupe(u8, "s"),
            } }),
            1 => t.request(t.pickConn(r), t.pickSid(r), .{ .generate = .{
                .max_tokens = 1 + r.uintLessThan(u32, 6),
                .sampling = null,
                .text = try gpa.dupe(u8, "x"),
            } }),
            2 => t.request(t.pickConn(r), t.pickSid(r), .cancel),
            3 => t.request(
                t.pickConn(r),
                t.pickSid(r),
                .{ .session_keep = .{ .ttl_ms = r.uintAtMost(u32, trace_ttl_max_ms + 20) } },
            ),
            4 => try t.drop(r.uintLessThan(usize, trace_conns)),
            5 => {
                var token: Token = undefined;
                if (t.n_tokens == 0 or r.uintLessThan(u8, 8) == 0) {
                    r.bytes(&token);
                    token[0] |= 1;
                } else {
                    token = t.tokens[r.uintLessThan(usize, @min(t.n_tokens, t.tokens.len))];
                }

                try t.resumeWith(t.pickConn(r), token);
            },
            6 => t.request(t.pickConn(r), t.pickSid(r), .session_close),
            7 => {
                // Half the steps land on the earliest deadline, the edge of the expiry rule.
                // A deadline that passed without a tick stays behind the clock.
                const deadline = t.earliestDeadline();
                if (deadline != null and r.boolean()) {
                    t.pinAt(@max(deadline.?, t.now));
                } else {
                    t.pinAt(t.now + r.uintAtMost(u32, 40));
                }
            },
            else => {},
        }

        for (0..r.uintAtMost(usize, 3)) |_| {
            try t.tickOnce();
        }

        try t.absorbEvents();
        try t.expectTable();
    }

    // A clock past every time to live and one tick end every lease.
    t.pinAt(t.now + trace_ttl_max_ms);
    try t.tickOnce();
    try std.testing.expect(t.earliestDeadline() == null);
    // Teardown does not run a turn in flight, so the drain ends it under the stream checks.
    t.fx.drain();
    try t.absorbEvents();
}

test "property: random wire traces keep the table, end leases, and stop tokens at a finish" {
    try prop.forAll(
        .{ .site = 0x1EA5_E001, .cases = 300 },
        {},
        u64,
        prop.genSeed(void),
        checkLeaseTrace,
    );
}

/// The request id of the open in the shutdown cases.
const open_request: u32 = 41;

/// An open of the four-byte system prompt. Two slots feed two tokens a tick, so the prefill of
/// the opening needs two ticks.
fn openOnly(fx: *Fixture, conn: u32) !void {
    const dup = try std.testing.allocator.dupe(u8, "abcd");
    _ = fx.sched.handle(.{ .conn = conn, .request_id = open_request, .session_id = 0, .body = .{
        .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
    } }, fx.sink);
}

fn shutdownNow(fx: *Fixture) !void {
    const res = fx.sched.handle(
        .{ .conn = 0, .request_id = 77, .session_id = 0, .body = .shutdown },
        fx.sink,
    );
    try std.testing.expectEqual(HandleResult.shutdown, res);
    fx.sched.checkLeases();
}

/// No event names the open with a `done`, and the entry counts no turn.
fn expectOpeningStopped(fx: *const Fixture, entry: *const TestScheduler.SessionEntry) !void {
    for (fx.sink.seen()) |e| {
        if (e.ev.body == .done) {
            try std.testing.expect(e.ev.request_id != open_request);
        }
    }

    try std.testing.expectEqual(@as(u32, 0), entry.turns_done);
    try std.testing.expect(entry.last_done == null);
    try std.testing.expect(entry.state != .prefill and entry.state != .decode);
}

test "a shutdown ends an opening with no done and no counted turn: before the first chunk" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    try openOnly(&fx, 1);
    try std.testing.expectEqual(SessionPhase.prefill, fx.sched.entries[0].state);
    try std.testing.expectEqual(@as(usize, 0), fx.sched.entries[0].fed);

    try shutdownNow(&fx);
    try expectOpeningStopped(&fx, &fx.sched.entries[0]);
    try std.testing.expectEqual(
        OutEvent.Body.shutdown_ack,
        std.meta.activeTag(fx.sink.last().ev.body),
    );
}

test "a shutdown ends an opening with no done and no counted turn: between two chunks" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    try openOnly(&fx, 1);
    _ = fx.sched.tick(fx.sink);
    try std.testing.expectEqual(SessionPhase.prefill, fx.sched.entries[0].state);
    try std.testing.expect(fx.sched.entries[0].fed > 0);

    try shutdownNow(&fx);
    try expectOpeningStopped(&fx, &fx.sched.entries[0]);
    try std.testing.expectEqual(
        OutEvent.Body.shutdown_ack,
        std.meta.activeTag(fx.sink.last().ev.body),
    );
}

test "a shutdown ends an opening with no done and no counted turn: after the last chunk" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    try openOnly(&fx, 1);
    fx.drain();
    try std.testing.expectEqual(
        OutEvent.Body.session_opened,
        std.meta.activeTag(fx.sink.last().ev.body),
    );

    try shutdownNow(&fx);
    try expectOpeningStopped(&fx, &fx.sched.entries[0]);
    var opened: usize = 0;
    for (fx.sink.seen()) |e| {
        if (e.ev.body == .session_opened) {
            opened += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 1), opened);
}

test "a shutdown ends an opening with no done and no counted turn: beside a decoding session" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    const sid = try fx.openSession(1, "a");
    try fx.generate(1, sid, "x", 50);
    var ticks: usize = 0;
    while (fx.sched.entries[0].state != .decode and ticks < 8) : (ticks += 1) {
        _ = fx.sched.tick(fx.sink);
    }

    try std.testing.expectEqual(SessionPhase.decode, fx.sched.entries[0].state);
    try openOnly(&fx, 2);
    try std.testing.expectEqual(SessionPhase.prefill, fx.sched.entries[1].state);

    const before = fx.sink.n;
    try shutdownNow(&fx);
    try expectOpeningStopped(&fx, &fx.sched.entries[1]);
    var dones: usize = 0;
    for (fx.sink.seen()[before..]) |e| {
        if (e.ev.body != .done) continue;

        dones += 1;
        try std.testing.expectEqual(@as(u32, 9), e.ev.request_id);
        try std.testing.expectEqual(protocol.FinishReason.cancelled, e.ev.body.done.finish_reason);
    }

    try std.testing.expectEqual(@as(usize, 1), dones);
    try std.testing.expectEqual(@as(u32, 1), fx.sched.entries[0].turns_done);
}

test "a shutdown ends an opening with no done and no counted turn: with a waiting resume" {
    var fx = try leaseFixture(2, 2);
    defer fx.tearDown();

    try openOnly(&fx, 1);
    _ = fx.sched.tick(fx.sink);
    const entry = &fx.sched.entries[0];
    try std.testing.expectEqual(SessionPhase.prefill, entry.state);
    _ = try fx.sched.keepLease(entry.proto_id, 1, 50, tok(7));
    fx.sched.detachConn(1, atMs(10));
    const r = try fx.sched.resumeLease(2, 5, tok(7), atMs(20), tok(8));
    try std.testing.expect(!r.ready);
    try std.testing.expect(entry.resume_wait != null);

    try shutdownNow(&fx);
    try expectOpeningStopped(&fx, entry);
    try std.testing.expect(entry.resume_wait == null);
    try std.testing.expectEqual(SessionPhase.idle, entry.state);
}

test "wire trace: a cancel in an opening prefill leaves the open to end with session_opened" {
    var t: LeaseTrace = .{ .fx = try leaseFixture(2, 1) };
    defer t.fx.tearDown();

    // A guessed id cancels the opening prefill of session 1.
    t.request(1, 0, .{ .session_open = .{
        .sampling = protocol.Sampling.greedy,
        .system_prompt = try std.testing.allocator.dupe(u8, "s"),
    } });
    t.request(1, 1, .cancel);
    t.fx.drain();
    for (t.fx.sink.seen()) |e| {
        try std.testing.expect(e.ev.body != .done);
    }

    try std.testing.expectEqual(OutEvent.Body.session_opened, std.meta.activeTag(
        t.fx.sink.last().ev.body,
    ));
    try t.absorbEvents();

    // Connection 2 takes the session over with its token. Connection 1 then holds no session,
    // so its share admits a resume, and an unknown token answers `unknown_session`.
    t.request(1, 1, .{ .session_keep = .{ .ttl_ms = 50 } });
    try t.absorbEvents();
    try std.testing.expectEqual(@as(usize, 1), t.n_tokens);
    try t.resumeWith(2, t.tokens[0]);
    try t.absorbEvents();
    try t.resumeWith(1, @splat(0xA5));
    try t.absorbEvents();
    try t.expectTable();
}

test "wire trace: a keep in an opening prefill counts the session once against the share" {
    var t: LeaseTrace = .{ .fx = try leaseFixture(2, 2) };
    defer t.fx.tearDown();

    t.request(1, 0, .{ .session_open = .{
        .sampling = protocol.Sampling.greedy,
        .system_prompt = try std.testing.allocator.dupe(u8, "s"),
    } });
    t.request(1, 1, .{ .session_keep = .{ .ttl_ms = 50 } });
    try t.absorbEvents();
    try std.testing.expectEqual(@as(usize, 1), t.n_tokens);

    // Connection 1 holds one session of two, so its own token resumes, and the reply waits.
    try t.resumeWith(1, t.tokens[0]);
    try std.testing.expect(t.modelOf(1).pending_resume);
    t.fx.drain();
    try t.absorbEvents();
    try std.testing.expect(!t.modelOf(1).pending_resume);
    try t.expectTable();
}

test "wire trace: a close in an opening prefill ends the open of that session" {
    var t: LeaseTrace = .{ .fx = try leaseFixture(2, 2) };
    defer t.fx.tearDown();

    for (0..2) |_| {
        t.request(2, 0, .{ .session_open = .{
            .sampling = protocol.Sampling.greedy,
            .system_prompt = try std.testing.allocator.dupe(u8, "s"),
        } });
    }

    // The close names the second open. The first open still ends with `session_opened`.
    t.request(2, 2, .session_close);
    t.fx.drain();
    try t.absorbEvents();
    try std.testing.expect(t.modelOf(1).opened);
    try std.testing.expect(t.modelOf(2).gone);

    // Connection 2 holds one session of two, so an unknown token answers `unknown_session`.
    try t.resumeWith(2, @splat(0xA5));
    try t.absorbEvents();
    try t.expectTable();
}
