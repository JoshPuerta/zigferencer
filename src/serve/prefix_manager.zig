//! The serve prefix cache: its entries, its counters, and its boot warm-up. A
//! turn that starts at position zero restores the longest cached prefix and
//! prefills only the suffix. A completed position-zero prefix is offered back
//! as a new entry. Every path is best-effort: a refusal or a failure degrades
//! to a cold prefill.
//!
//! The boundary with the scheduler. The scheduler keeps every engine call whose
//! result a slot's state machine consumes (`restorePrefix` and `restoreReady`
//! drive `slot.fed`, `endPrefill`, and `acceptToken`). This module keeps every
//! engine call whose result only the cache consumes (`snapshotBytesFor`,
//! `snapshotPrefix`, and `copyFinalLogits` travel with the insert body).
//!
//! `PrefixManager` is a plain struct and not a generic over the engine. The
//! scheduler's `can_snapshot` then keeps its single definition on the type that
//! owns the Engine boundary. `insert` takes `engine: anytype`, and only a
//! `comptime can_snapshot`-guarded call site instantiates it, so this module
//! also compiles for an engine that cannot snapshot.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const limits = @import("../core/limits.zig");
const protocol = @import("protocol.zig");
const event = @import("event.zig");
const prompt_cache_mod = @import("../runtime/prompt_cache.zig");
const OutEvent = event.OutEvent;

const log = std.log.scoped(.serve);

// -- constants --------------------------------------------------------------

/// A connection id no transport can hold. Preload runs before the engine loop
/// starts, so no live connection exists to collide with it.
const preload_conn: u32 = std.math.maxInt(u32);

// -- types ------------------------------------------------------------------

/// The prefix cache's share of the serve configuration.
pub const CacheConfig = struct {
    /// RAM budget for the prefix cache. 0 keeps the cache off. Only an engine
    /// with a snapshot surface reads it.
    ram_bytes: u64 = 0,
    /// Disk budget for the prefix cache. 0 keeps the disk tier off. The tier
    /// survives restarts, so it reads `dir` beside the budget.
    disk_bytes: u64 = 0,
    /// The on-disk cache directory, as the CLI resolves it from
    /// `$ZIGFERENCER_CACHE_DIR` or the default. The manager reads it only when
    /// `disk_bytes > 0`. Its lifetime must cover the manager's.
    dir: []const u8 = "",
    /// Weights identity for prefix-cache keys, backend-qualified for the disk
    /// tier. A different model or another runner family must never restore
    /// this cache's state.
    model_fp: u64 = 0,
    /// Boot-time cache warming (`--preload`): one session opens and closes per
    /// text before the engine loop starts, so the prefill lands in the prefix
    /// cache. The connector prepares the text. The OpenAI launcher pre-wraps
    /// it, because its codec encodes raw text. At most
    /// `limits.serve_preload_max` entries. Its lifetime must cover `preload`.
    preload: []const []const u8 = &.{},
};

/// One cache entry offered to a restore. The slices point into the entry, so
/// the caller must spend them before the next insert reallocates the RAM list.
const Hit = struct {
    n: u32,
    blob: []const u8,
    logits: []const f32 = &.{},
};

/// What the cache offers a turn that starts at position zero. `ready` carries
/// the stored logits of the last row, so the engine samples the first token
/// with zero forward rows. `rows` restores a prefix, and the suffix still needs
/// the engine.
pub const Restore = union(enum) { miss, ready: Hit, rows: Hit };

/// The slot facts an insert reads. The scheduler owns the slot, so it states
/// them as a value.
const Candidate = struct {
    engine_id: u32,
    proto_id: u32,
    tokens: []const u32,
    opening: bool,
    vocab_size: u32,
    t_turn_start: Io.Clock.Timestamp,
};

/// The boot sink: accept every event, and keep the open result. Nothing parks
/// here, because `emit` never refuses.
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
    // Disk-tier splits of the aggregates. A disk hit is also a promotion
    // into RAM, so no separate promote counter exists.
    disk_hits: u64 = 0,
    disk_inserts: u64 = 0,

    /// `can_snapshot` is the engine's snapshot capability. Without it, or
    /// without a RAM budget, the cache stays null and every turn takes the cold
    /// path.
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

    /// The longest cached prefix this turn can restore. An opening turn can
    /// restore its whole prompt, because it wants no logits. A generate turn
    /// whose whole prompt matches an entry WITH stored logits gets `ready`. A
    /// full-length entry without logits cannot serve a generate turn whole
    /// (blobs restore whole or not at all), so the search retries below the
    /// last token and the final chunk samples through the engine.
    ///
    /// This counts nothing. The scheduler counts at its own outcome sites,
    /// where the engine has already answered.
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

    /// Count one served restore. The cache's own record of its last lookup
    /// names the tier that served it.
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

    /// Offer a completed position-zero prefix to the cache. This snapshots the
    /// slot and times the copy. The copy is symmetric, so its duration is the
    /// restore estimate the measured `worthwhile` gate reads. This runs before
    /// the turn's token buffer is freed. It is best-effort: an allocation or a
    /// snapshot failure skips the insert and costs nothing.
    pub fn insert(self: *PrefixManager, engine: anytype, c: Candidate) void {
        const cache = if (self.cache) |*p| p else return;
        const tokens = c.tokens;
        const n: u32 = @intCast(tokens.len);
        const key = prompt_cache_mod.prefixKey(self.model_fp, tokens);
        // An exact entry already serves this prefix, and the lookup's LRU
        // bump is the only bookkeeping it needs.
        if (cache.lookup(key, tokens) != null) return;

        // Wall time from turn start, so decode interleaving inflates it.
        // The gate then over-admits at the margin, which the LRU absorbs.
        const prefill_ns = nsBetween(c.t_turn_start, Io.Clock.Timestamp.now(self.io, .awake));
        const bytes: usize = @intCast(engine.snapshotBytesFor(n));
        const buf = self.gpa.alloc(u8, bytes) catch return;
        defer self.gpa.free(buf);

        // The restore estimate times the copy alone. A hit pays the symmetric
        // device copy, never this transient allocation.
        const t_copy = Io.Clock.Timestamp.now(self.io, .awake);
        engine.snapshotPrefix(c.engine_id, n, buf) catch |err| {
            log.debug("prefix snapshot failed: session {d}: {t}", .{ c.proto_id, err });
            return;
        };
        const restore_ns = nsBetween(t_copy, Io.Clock.Timestamp.now(self.io, .awake));
        // A generate turn's final row just produced the logits the turn
        // sampled from, and this tick is the one window they survive in
        // the engine. Store them, so an exact-duplicate prompt is served
        // with zero forward rows (the regenerate case). An opening turn
        // computes no logits and stores none. A failed allocation only
        // downgrades the entry to restore-only.
        var logits: []const f32 = &[_]f32{};
        const logits_buf: []f32 = if (c.opening) &.{} else self.gpa.alloc(
            f32,
            c.vocab_size,
        ) catch &.{};

        defer if (logits_buf.len > 0) self.gpa.free(logits_buf);

        if (logits_buf.len > 0) {
            engine.copyFinalLogits(c.engine_id, logits_buf);
            logits = logits_buf;
        }

        const admitted = cache.admit(key, tokens, buf, logits, prefill_ns, restore_ns) catch false;
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

    /// Warm the prefix cache from `texts`: open, prefill, and close one session
    /// per text through the normal request path, so the entry tokenizes exactly
    /// as live traffic does. This runs single-threaded at boot, before the
    /// engine loop. It is best-effort per text. A refusal logs one line and the
    /// boot proceeds, so the operator reads the outcome per text.
    ///
    /// `sched` reaches this same manager through its own `prefix` field, so
    /// `self` and `sched` alias. The body must therefore not hold a manager
    /// field across a tick, because a tick can change every counter and the
    /// cache list.
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

            var sink: PreloadSink = .{};
            const dup = self.gpa.dupe(u8, text) catch {
                log.warn("preload abandoned: out of memory", .{});
                return;
            };
            _ = sched.handle(.{ .conn = preload_conn, .request_id = 0, .session_id = 0, .body = .{
                .session_open = .{ .sampling = protocol.Sampling.greedy, .system_prompt = dup },
            } }, &sink);
            // One chunk advances per tick, so max_seq ticks cover the
            // longest legal prefill. The margin covers the open itself.
            const bound: u32 = max_seq + 16;
            var iters: u32 = 0;
            while (sched.tick(&sink) != .idle) {
                iters += 1;
                if (iters > bound) {
                    log.err("preload stalled after {d} ticks, abandoned", .{bound});
                    return;
                }
            }

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

// -- helpers ----------------------------------------------------------------

/// Whole nanoseconds for the prefix-cache admission gate, clamped at zero so
/// a clock step never wraps the unsigned cost.
fn nsBetween(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) u64 {
    const ns: i64 = @intCast(t0.durationTo(t1).raw.toNanoseconds());
    return if (ns > 0) @intCast(ns) else 0;
}
