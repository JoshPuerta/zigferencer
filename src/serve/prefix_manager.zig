//! The serve prefix cache: its entries, its counters, and its boot warm-up. A turn that starts at
//! position zero restores the longest cached prefix and prefills only the suffix. A completed
//! position-zero prefix is offered back as a new entry. Every path is best-effort: a refusal or a
//! failure degrades to a cold prefill.
//!
//! Boundary with the scheduler. The scheduler keeps every engine call whose result the phase of a
//! session entry consumes (`restorePrefix` and `restoreReady` drive `entry.fed`, `endPrefill`, and
//! `acceptToken`). This module keeps every engine call whose result only the cache consumes
//! (`snapshotBytesFor`, `snapshotPrefix`, and `copyFinalLogits` travel with the insert body).
//!
//! Comptime. `PrefixManager` is a plain struct and not a generic over the session pool. So the
//! scheduler's `can_snapshot` keeps its single definition on the type that owns the pool
//! boundary. `insert` takes `pool: anytype`, and only a `comptime can_snapshot`-guarded call
//! site instantiates it. So this module also compiles for a pool that cannot snapshot.
//!
//! Restore. An opening turn can restore its whole prompt, because it wants no logits. A generate
//! turn whose whole prompt matches an entry with stored logits gets `ready`. A full-length entry
//! without logits cannot serve a generate turn whole, because blobs restore whole or not at all.
//! So the search retries below the last token, and the final chunk samples through the engine.
//! `planRestore` counts nothing. The scheduler counts at its own outcome sites.
//!
//! Insert. The copy of the snapshot is symmetric, so its duration is the restore estimate that the
//! `worthwhile` gate reads. The prefill time is wall time from the turn start, so decode
//! interleaving inflates it, and the gate over-admits at the margin, which the LRU absorbs. A
//! generate turn stores the logits of its final row, because this tick is the only one in which
//! the engine still holds them. An exact-duplicate prompt then needs zero forward rows. An
//! opening turn stores none. A failed logits allocation downgrades the entry to restore-only.
//!
//! Lifetime. A `Hit` points into the cache entry. The caller must use it before the next insert
//! reallocates the RAM list. The strings of `CacheConfig.dir` and `preload` must outlive the
//! manager.
//!
//! Preload. `preload` opens, prefills, and closes one session per text through the normal request
//! path, so the entry tokenizes as live traffic does. The OpenAI launcher pre-wraps each text,
//! because its codec encodes raw text. It runs single-threaded at boot, before the
//! engine loop. `sched` reaches this manager through its own `prefix` field, so `self` and `sched`
//! alias. The body must not hold a manager field across a tick, because a tick can change every
//! counter and the cache list.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const limits = @import("../core/limits.zig");
const time = @import("../core/utils/time.zig");
const protocol = @import("protocol.zig");
const event = @import("event.zig");
const prompt_cache_mod = @import("../runtime/prompt_cache.zig");
const OutEvent = event.OutEvent;

const log = std.log.scoped(.serve);

// -- constants --------------------------------------------------------------

/// A connection id no transport can hold. Preload runs before any live connection exists.
const preload_conn: u32 = std.math.maxInt(u32);

// -- types ------------------------------------------------------------------

/// The prefix cache's share of the serve configuration.
pub const CacheConfig = struct {
    /// RAM budget for the prefix cache. 0 keeps the cache off.
    ram_bytes: u64 = 0,
    /// Disk budget for the prefix cache. 0 keeps the disk tier off.
    disk_bytes: u64 = 0,
    /// The on-disk cache directory. The manager reads it only when `disk_bytes > 0`.
    dir: []const u8 = "",
    /// Weights identity in the cache keys. Another model or runner family must never match it.
    model_fp: u64 = 0,
    /// Boot warm-up texts (`--preload`), at most `limits.serve_preload_max`.
    preload: []const []const u8 = &.{},
};

/// One cache entry offered to a restore.
const Hit = struct {
    n: u32,
    blob: []const u8,
    logits: []const f32 = &.{},
};

/// What the cache offers a turn at position zero. `ready` holds stored logits, so no row is needed.
/// `rows` restores a prefix, and the suffix still needs the engine.
pub const Restore = union(enum) { miss, ready: Hit, rows: Hit };

/// The values of a session entry that an insert reads. The scheduler owns the entry, so they
/// cross as a value.
const Candidate = struct {
    engine_id: u32,
    proto_id: u32,
    tokens: []const u32,
    opening: bool,
    vocab_size: u32,
    t_turn_start: Io.Clock.Timestamp,
};

/// The boot sink: accept every event, and keep the open result. `emit` never refuses.
const PreloadSink = struct {
    opened: ?struct { sid: u32, kv_pos: u32 } = null,
    err_detail: [64]u8 = undefined,
    err_len: u8 = 0,

    pub fn emit(self: *PreloadSink, conn: u32, ev: OutEvent) bool {
        std.debug.assert(conn == preload_conn);

        switch (ev.body) {
            .session_opened => |m| self.opened = .{ .sid = ev.session_id, .kv_pos = m.kv_pos },
            .err => |e| {
                self.err_len = e.detail_len;
                @memcpy(self.err_detail[0..e.detail_len], e.detail[0..e.detail_len]);
            },
            else => {},
        }

        return true;
    }

    pub fn disconnect(self: *PreloadSink, conn: u32) void {
        _ = self;
        _ = conn;
    }
};

pub const PrefixManager = struct {
    gpa: Allocator,
    io: Io,
    model_fp: u64,
    cache: ?prompt_cache_mod.PromptCache,
    hits: u64 = 0,
    misses: u64 = 0,
    inserts: u64 = 0,
    // Disk-tier splits of the aggregates. A disk hit also promotes into RAM.
    disk_hits: u64 = 0,
    disk_inserts: u64 = 0,

    /// Without `can_snapshot` or a RAM budget, the cache stays null and every turn runs cold.
    pub fn init(gpa: Allocator, io: Io, cfg: CacheConfig, can_snapshot: bool) PrefixManager {
        var self: PrefixManager = .{
            .gpa = gpa,
            .io = io,
            .model_fp = cfg.model_fp,
            .cache = if (can_snapshot and cfg.ram_bytes > 0)
                prompt_cache_mod.PromptCache.init(gpa, @intCast(cfg.ram_bytes))
            else
                null,
        };
        if (self.cache) |*cache| {
            if (cfg.disk_bytes > 0) {
                // Best-effort: a failed enable keeps the RAM tier alone.
                if (cache.enableDisk(io, cfg.dir, @intCast(cfg.disk_bytes), cfg.model_fp)) |_| {
                    cache.loadDiskIndex();
                    log.info("prefix disk: {d} MiB at {s}, {d} entries indexed", .{
                        cfg.disk_bytes / (1024 * 1024),
                        cfg.dir,
                        cache.diskIndexLen(),
                    });
                } else |err| {
                    log.warn("prefix disk tier disabled: {t}", .{err});
                }
            }
        }

        return self;
    }

    pub fn deinit(self: *PrefixManager) void {
        if (self.cache) |*cache| {
            cache.deinit();
        }
    }

    pub fn enabled(self: *const PrefixManager) bool {
        return self.cache != null;
    }

    pub fn diskEnabled(self: *const PrefixManager) bool {
        return self.cache != null and self.cache.?.dir != null;
    }

    pub fn logEntryHits(self: *const PrefixManager) void {
        if (self.cache) |*cache| {
            cache.logEntryHits();
        }
    }

    /// The longest cached prefix this turn can restore.
    pub fn planRestore(self: *PrefixManager, tokens: []const u32, opening: bool) Restore {
        std.debug.assert(self.cache != null);

        const cache = &self.cache.?;
        const best = cache.lookupBestPrefix(tokens) orelse return .miss;
        if (!opening and best.n == tokens.len and best.last_logits.len > 0) {
            return .{ .ready = .{ .n = best.n, .blob = best.blob, .logits = best.last_logits } };
        }
        if (opening or best.n < tokens.len) return .{ .rows = .{ .n = best.n, .blob = best.blob } };

        const shorter = cache.lookupBestPrefix(tokens[0 .. tokens.len - 1]) orelse return .miss;
        return .{ .rows = .{ .n = shorter.n, .blob = shorter.blob } };
    }

    /// Count one served restore. The cache records which tier served its last lookup.
    pub fn recordHit(self: *PrefixManager) void {
        std.debug.assert(self.cache != null);

        self.hits += 1;
        if (self.cache.?.last_hit_from_disk) {
            self.disk_hits += 1;
        }
    }

    pub fn recordMiss(self: *PrefixManager) void {
        self.misses += 1;
    }

    /// Offer a completed position-zero prefix to the cache. It runs before the turn frees its
    /// tokens. An allocation or snapshot failure skips the insert.
    pub fn insert(self: *PrefixManager, pool: anytype, c: Candidate) void {
        const cache = if (self.cache) |*p| p else return;
        const tokens = c.tokens;
        const n: u32 = @intCast(tokens.len);
        const key = prompt_cache_mod.prefixKey(self.model_fp, tokens);

        // Skip a prefix that an exact entry already serves. The lookup bumps its LRU rank.
        if (cache.lookup(key, tokens) != null) return;

        // Snapshot the slot.
        const t_now = Io.Clock.Timestamp.now(self.io, .awake);
        const prefill_ns = time.elapsedNsClamped(c.t_turn_start, t_now);
        const bytes: usize = @intCast(pool.snapshotBytesFor(n));
        const buf = self.gpa.alloc(u8, bytes) catch return;
        defer self.gpa.free(buf);

        // Time the copy alone, because a hit never makes this transient allocation.
        const t_copy = Io.Clock.Timestamp.now(self.io, .awake);
        pool.snapshotPrefix(c.engine_id, n, buf) catch |err| {
            log.debug("prefix snapshot failed: session {d}: {t}", .{ c.proto_id, err });
            return;
        };
        const restore_ns = time.elapsedNsClamped(t_copy, Io.Clock.Timestamp.now(self.io, .awake));

        // Copy the final logits of a generate turn.
        var logits: []const f32 = &[_]f32{};
        const logits_buf: []f32 = if (c.opening) &.{} else self.gpa.alloc(
            f32,
            c.vocab_size,
        ) catch &.{};

        defer if (logits_buf.len > 0) self.gpa.free(logits_buf);

        if (logits_buf.len > 0) {
            pool.copyFinalLogits(c.engine_id, logits_buf);
            logits = logits_buf;
        }

        // Admit the entry and count it.
        const admitted = cache.admit(key, tokens, buf, logits, prefill_ns, restore_ns);
        if (admitted) {
            self.inserts += 1;
            if (cache.last_admit_disk) {
                self.disk_inserts += 1;
            }

            log.debug(
                "prefix insert: session {d}, {d} tokens, {d} bytes",
                .{ c.proto_id, n, bytes },
            );
        }
    }

    /// Warm the prefix cache from `texts`. A refusal logs one line and the boot proceeds.
    pub fn preload(
        self: *PrefixManager,
        sched: anytype,
        texts: []const []const u8,
        max_seq: u32,
    ) void {
        if (texts.len == 0) return;

        std.debug.assert(texts.len <= limits.serve_preload_max);
        if (self.cache == null) {
            log.warn("preload skipped: the prefix cache is disabled on this configuration", .{});
            return;
        }

        for (texts) |text| {
            std.debug.assert(text.len > 0);

            // Open one session on the text.
            var sink: PreloadSink = .{};
            const dup = self.gpa.dupe(u8, text) catch {
                log.warn("preload abandoned: out of memory", .{});
                return;
            };
            _ = sched.handle(.{ .conn = preload_conn, .request_id = 0, .session_id = 0, .body = .{
                .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
            } }, &sink);

            // Tick until the prefill ends: one chunk per tick, plus 16 ticks for the open.
            const bound: u32 = max_seq + 16;
            var iters: u32 = 0;
            while (sched.tick(&sink) != .idle) {
                iters += 1;
                if (iters > bound) {
                    log.err("preload stalled after {d} ticks, abandoned", .{bound});
                    return;
                }
            }

            // Close the session and report the warmed tokens.
            const o = sink.opened orelse {
                log.warn("preload refused: {s}", .{sink.err_detail[0..sink.err_len]});
                continue;
            };
            _ = sched.handle(.{
                .conn = preload_conn,
                .request_id = 0,
                .session_id = o.sid,
                .body = .session_close,
            }, &sink);
            log.info("preload: {d} tokens warmed", .{o.kv_pos});
        }
    }
};
