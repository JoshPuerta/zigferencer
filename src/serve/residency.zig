//! Tiered KV residency: a swap pool over `Sessions(RunnerT)` that lets the serve layer hold more
//! *logical* sessions than there are *physical* VRAM KV slots. Active sessions keep their KV in
//! VRAM. This layer evicts idle ones to a host (RAM) blob and restores them on their next turn.
//! This is sequence-level swap (the vLLM and LMCache shape), not per-token streaming.
//!
//! Decomposition. `Sessions` stays a 1:1 pool over the physical slots, so its `id == slot`
//! invariant is untouched. This layer owns the LOGICAL session state. A resident session keeps its
//! truth in `sessions.table[phys]`. An evicted session keeps it in `Entry{saved, blob}`. The layer
//! presents the method surface of `Sessions`, keyed by logical id, so the scheduler only swaps the
//! engine type and gains a residency-aware admission step.
//!
//! Control unit and datapath. This file is the control unit: the residency FSM and the LRU. The
//! datapath is the runner's `evictSlot` and `restoreSlot` (pure byte moves) and the step machinery
//! of `Sessions`. A runner without `evictSlot` requires `n_logical <= n_physical`, so nothing is
//! ever evicted and the swap path compiles out. Only a test fake compiles those identity arms.
//!
//! Victim choice. The victim is the least-recently-used resident entry whose physical slot the
//! tick does not need. `exclude` holds those slots: the decode batch and, for a prefill, the feed
//! session's own slot. Eviction is byte-faithful, so any slot outside `exclude` is dormant and a
//! mid-turn or `.feeding` session restores identically. A filter on `state == .ready` is wrong: an
//! opening prefill leaves the session `.feeding`, so that filter pins idle sessions and deadlocks
//! an over-subscribed pool. The scan mirrors the LRU scan in `prompt_cache`.
//!
//! Blob. An evicted session keeps a RAM blob: the trimmed slot bytes and the count of committed
//! rows they cover. Only committed rows travel, because attention reads no position past `pos`.
//! The disk tier adds its header (version, model fingerprint) at serialization time. A session
//! with no blob has no RAM copy, so its next placement resets the slot.
//!
//! Ledger. `ram_used` is the sum of the blob lengths after every public operation. The budget
//! gate runs before any allocation or device copy, so a refusal leaves the pool as it was. A
//! failed restore keeps its blob, because the blob is the only copy until the restore lands. A
//! successful restore also releases it, so VRAM is authoritative. `destroy` is the one legal way a
//! live blob disappears, and it holds the only credit. The swap counters cover device copies only,
//! so the millisecond totals divide by the counts.
//!
//! Prefix seeding. `restorePrefix` and `restoreReady` seed a FRESH logical session with a cached
//! prefix. They place the session first, because the bytes land in a physical slot. The session
//! then reads as one that fed `n_rows` non-final tokens, and `pos` is the one variable that the
//! tiers consult. A placement refusal (`error.NoEvictableSlot`, `error.RamBudgetExceeded`) or a
//! device failure leaves the session at `pos` 0, so the caller degrades to a cold prefill, which
//! rewrites each row before it reads it. The blob stays caller-owned, so the ledger never sees it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const session = @import("../runtime/session.zig");
const sampler = @import("../runtime/sampler.zig");
const limits = @import("../core/limits.zig");
const prop = @import("../core/prop.zig");
const BlobTier = @import("../core/kv_budget.zig").BlobTier;
const core_log = @import("../core/log.zig");
const span = @import("../core/span.zig");
const FeedChunk = session.FeedChunk;

const log = std.log.scoped(.serve);

// -- types ------------------------------------------------------------------

/// One evicted session's RAM copy: the trimmed slot bytes and the committed rows they cover.
const KvBlob = struct {
    n_rows: u32 = 0,
    bytes: []u8 = &.{},
};

/// The residency pool's counters as one immutable value, outside the generic.
const ServeStats = struct {
    slots_resident: u32,
    n_physical: u32,
    n_logical: u32,
    ram_used: u64,
    /// Absent when the slots hold every session, unenforced with no budget, else bounded.
    blob_budget: BlobTier,
    evicts: u64,
    restores: u64,
    evict_ms: f64,
    restore_ms: f64,
};

pub fn Residency(comptime SessionsT: type) type {
    return struct {
        const Self = @This();
        pub const Runner = SessionsT.Runner;
        /// Whether the runner can spill a slot to RAM. Without it the pool is the identity.
        const can_evict = @hasDecl(Runner, "evictSlot");
        pub const can_clone = SessionsT.can_clone;
        /// The prefix surface uses the eviction hook, so a pool that evicts can seed a prefix.
        pub const can_snapshot = can_evict;

        const Entry = struct {
            used: bool = false,
            /// Physical slot when resident. Null when evicted or never placed.
            phys: ?u32 = null,
            /// A KV blob exists to restore (false until first eviction).
            ever_resident: bool = false,
            /// The session bookkeeping while evicted. Resident, `sessions.table[phys]` holds it.
            saved: session.Session = undefined,
            /// The gpa-owned evicted KV and recurrent bytes. Empty until the first eviction.
            blob: KvBlob = .{},
            last_used: u64 = 0,
        };

        gpa: Allocator,
        sessions: *SessionsT,
        entries: []Entry, // [n_logical]
        phys_owner: []?u32, // [n_physical] -> logical id holding the slot
        /// Host bytes the live blobs hold. See "Ledger" in the header.
        ram_used: u64,
        /// Ceiling on `ram_used`. Null accounts the tier without enforcing it.
        blob_budget: ?u64,
        tick: u64,
        /// Swap counters. They cover device copies only. See "Ledger" in the header.
        evicts: u64 = 0,
        restores: u64 = 0,
        evict_ms: f64 = 0,
        restore_ms: f64 = 0,

        /// `n_logical` can exceed the physical slot count only when the runner can evict.
        /// `blob_budget` is the ledger budget of the blob tier: zero refuses every eviction, and
        /// null enforces nothing.
        pub fn init(
            arena: Allocator,
            gpa: Allocator,
            sessions: *SessionsT,
            n_logical: u32,
            blob_budget: ?u64,
        ) Allocator.Error!Self {
            const n_physical = sessions.runner.n_slots;
            std.debug.assert(n_logical >= 1);
            std.debug.assert(n_logical <= limits.max_logical_sessions);
            if (comptime !can_evict) std.debug.assert(n_logical <= n_physical);

            const entries = try arena.alloc(Entry, n_logical);
            for (entries) |*e| {
                e.* = .{};
            }

            const phys_owner = try arena.alloc(?u32, n_physical);
            for (phys_owner) |*o| {
                o.* = null;
            }

            return .{
                .gpa = gpa,
                .sessions = sessions,
                .entries = entries,
                .phys_owner = phys_owner,
                .ram_used = 0,
                .blob_budget = blob_budget,
                .tick = 0,
                .evicts = 0,
                .restores = 0,
                .evict_ms = 0,
                .restore_ms = 0,
            };
        }

        /// Free every RAM blob. The ledger must drain to zero: a residue means
        /// a blob was charged without a matching credit.
        pub fn deinit(self: *Self) void {
            for (self.entries) |*e| {
                self.dropBlob(e);
            }

            std.debug.assert(self.ram_used == 0);
        }

        // -- the RAM byte ledger --------------------------------------------

        /// Book `bytes` into the ledger. The caller checks the budget first, so
        /// an over-budget charge here is a missed check, not backpressure.
        fn charge(self: *Self, bytes: u64) void {
            if (self.blob_budget) |budget| std.debug.assert(self.ram_used + bytes <= budget);
            self.ram_used += bytes;
        }

        /// Book `bytes` out of the ledger. Crediting more than the ledger holds
        /// means the same blob was credited twice.
        fn credit(self: *Self, bytes: u64) void {
            std.debug.assert(bytes <= self.ram_used);

            self.ram_used -= bytes;
        }

        /// Release `e`'s blob and credit its bytes back.
        fn dropBlob(self: *Self, e: *Entry) void {
            if (e.blob.bytes.len == 0) return;

            self.credit(e.blob.bytes.len);
            self.gpa.free(e.blob.bytes);
            e.blob = .{};
            e.ever_resident = false;
        }

        /// The incremental ledger equals the blob sum, and stays under the
        /// budget. The scan is bounded by the logical session count, which
        /// `init` caps at `limits.max_logical_sessions`.
        fn assertLedger(self: *const Self) void {
            if (comptime !std.debug.runtime_safety) return;

            var total: u64 = 0;
            for (self.entries) |*e| {
                total += e.blob.bytes.len;
            }

            std.debug.assert(total == self.ram_used);
            if (self.blob_budget) |budget| std.debug.assert(self.ram_used <= budget);
        }

        fn bump(self: *Self) u64 {
            self.tick += 1;
            return self.tick;
        }

        fn freeLogical(self: *Self) ?u32 {
            for (self.entries, 0..) |*e, i| {
                if (!e.used) return @intCast(i);
            }

            return null;
        }

        fn freePhys(self: *const Self) ?u32 {
            for (self.phys_owner, 0..) |o, i| {
                if (o == null) return @intCast(i);
            }

            return null;
        }

        /// The least-recently-used resident entry whose physical slot is not in `exclude`.
        /// See "Victim choice" in the header.
        fn lruVictim(self: *const Self, exclude: []const u32) ?u32 {
            var best: ?u32 = null;
            for (self.entries, 0..) |*e, i| {
                const phys = e.phys orelse continue;
                if (indexOf(exclude, phys) != null) continue;
                if (best == null or e.last_used < self.entries[best.?].last_used) {
                    best = @intCast(i);
                }
            }

            return best;
        }

        /// Make `id` resident, evicting an idle LRU victim (slot not in `exclude`) if
        /// no slot is free. `error.NoEvictableSlot` and `error.RamBudgetExceeded`
        /// both tell the caller to park the turn and retry next tick.
        pub fn ensureResident(self: *Self, id: u32, exclude: []const u32) !void {
            std.debug.assert(id < self.entries.len);

            const e = &self.entries[id];
            std.debug.assert(e.used);
            if (e.phys != null) {
                e.last_used = self.bump();
                return;
            }

            var slot = self.freePhys();
            if (slot == null) {
                if (comptime !can_evict) {
                    unreachable; // n_logical <= n_physical guarantees a free slot
                }

                const victim = self.lruVictim(exclude) orelse {
                    log.debug("admit: engine {d} blocked, all slots pinned", .{id});
                    return error.NoEvictableSlot;
                };
                slot = try self.evictEntry(victim);
            }

            try self.placeEntry(id, slot.?);
            self.assertLedger();
        }

        /// Spill `victim`'s live state to RAM and return its freed physical slot.
        /// Only the committed rows travel. See "Blob" in the header.
        fn evictEntry(self: *Self, victim: u32) !u32 {
            const e = &self.entries[victim];
            const s = e.phys.?;
            const n_rows = self.sessions.table[s].pos;
            if (comptime can_evict) {
                if (n_rows == 0) {
                    // No committed rows means nothing to restore. A live `pos` never falls.
                    std.debug.assert(e.blob.bytes.len == 0);
                } else {
                    const need: usize = @intCast(self.sessions.runner.evictSlotBytesFor(n_rows));
                    const held: u64 = e.blob.bytes.len;

                    // Gate the blob on the RAM budget before any allocation.
                    if (self.blob_budget) |budget| {
                        if (self.ram_used - held + need > budget) {
                            log.debug(
                                "evict refused: engine {d} wants {d} ram bytes, {d} in use, " ++
                                    "budget {d}",
                                .{ victim, need, self.ram_used, budget },
                            );
                            return error.RamBudgetExceeded;
                        }
                    }

                    // Resize the blob and book the change.
                    if (need != held) {
                        const resized = try self.gpa.realloc(e.blob.bytes, need);
                        self.credit(held);
                        e.blob.bytes = resized;
                        self.charge(need);
                    }

                    // Copy the slot out. A failed copy leaves stale bytes, so drop the blob.
                    errdefer self.dropBlob(e);

                    const t0 = core_log.now();
                    try self.sessions.runner.evictSlot(s, n_rows, e.blob.bytes);
                    const t1 = core_log.now();
                    span.record(.kv_evict, t0, t1);
                    self.evicts += 1;
                    self.evict_ms += msBetween(t0, t1);
                    e.blob.n_rows = n_rows;
                    e.ever_resident = true;
                }
            }

            // Free the physical slot and keep the bookkeeping.
            e.saved = self.sessions.table[s];
            log.debug("swap-out: engine {d} slot {d} pos {d}", .{ victim, s, n_rows });
            self.phys_owner[s] = null;
            e.phys = null;
            std.debug.assert(self.phys_owner[s] == null);
            return s;
        }

        /// Bring `id` into the free physical slot `s`. Restore its blob if it has one. Otherwise
        /// clear the stale recurrent state of a vacated slot.
        fn placeEntry(self: *Self, id: u32, s: u32) !void {
            std.debug.assert(self.phys_owner[s] == null);

            // Install the saved bookkeeping.
            const e = &self.entries[id];
            self.sessions.table[s] = e.saved;
            if (e.ever_resident) {
                log.debug("swap-in: engine {d} slot {d} pos {d}", .{ id, s, e.saved.pos });
            } else {
                log.debug("place: engine {d} slot {d} (fresh)", .{ id, s });
            }

            // Restore the blob, or reset a fresh slot.
            if (comptime can_evict) {
                if (e.ever_resident) {
                    std.debug.assert(
                        e.blob.bytes.len == @as(
                            usize,
                            @intCast(self.sessions.runner.evictSlotBytesFor(e.blob.n_rows)),
                        ),
                    );

                    // The blob stays until the restore succeeds. See "Ledger" in the header.
                    const t0 = core_log.now();
                    try self.sessions.runner.restoreSlot(s, e.blob.n_rows, e.blob.bytes);
                    const t1 = core_log.now();
                    span.record(.kv_restore, t0, t1);
                    self.restores += 1;
                    self.restore_ms += msBetween(t0, t1);
                    self.dropBlob(e);
                } else if (comptime @hasDecl(Runner, "resetSlotState")) {
                    self.sessions.runner.resetSlotState(s);
                }
            } else if (comptime @hasDecl(Runner, "resetSlotState")) {
                if (!e.ever_resident) {
                    self.sessions.runner.resetSlotState(s);
                }
            }

            // Record the new owner.
            self.phys_owner[s] = id;
            e.phys = s;
            e.last_used = self.bump();
        }

        /// One debug line per held blob, so the dump explains the `ram N MiB` of the stats line.
        pub fn logBlobs(self: *const Self) void {
            for (self.entries, 0..) |*e, id| {
                if (!e.used or e.blob.bytes.len == 0) continue;
                log.debug(
                    "ram blob: engine {d}, {d} rows, {d} KiB",
                    .{ id, e.blob.n_rows, e.blob.bytes.len / 1024 },
                );
            }
        }

        // -- the Sessions surface, keyed by logical id ----------------------

        /// Claim a logical session. It holds no physical slot yet. The first
        /// turn (the opening prefill) assigns one.
        pub fn create(self: *Self, seed: u64, params: sampler.Params) error{NoFreeSession}!u32 {
            std.debug.assert(params.temperature >= 0);
            std.debug.assert(params.top_p > 0);

            const id = self.freeLogical() orelse return error.NoFreeSession;
            self.entries[id] = .{
                .used = true,
                .saved = .{
                    .pos = 0,
                    .token_pending = 0,
                    .rng = .init(seed),
                    .params = params,
                    .state = .feeding,
                },
            };
            self.assertLedger();
            return id;
        }

        /// Fork `source` into a new logical session sharing its context but with
        /// its own seed/sampling. Both sides end resident.
        pub fn clone(self: *Self, source: u32, seed: u64, params: sampler.Params) !u32 {
            if (comptime !can_clone) @compileError("runner cannot clone slot state");

            std.debug.assert(source < self.entries.len);
            std.debug.assert(self.entries[source].used);
            try self.ensureResident(source, &.{});
            const src_phys = self.entries[source].phys.?;
            const id = self.freeLogical() orelse return error.NoFreeSession;
            self.entries[id] = .{ .used = true, .saved = self.sessions.table[src_phys] };
            // A residency refusal must not keep the claim. Otherwise a client
            // that retries a refused clone spends a logical session per try.
            errdefer self.entries[id] = .{};

            try self.ensureResident(id, &.{src_phys});
            const dst_phys = self.entries[id].phys.?;
            const src = self.sessions.table[src_phys];
            self.sessions.runner.cloneSlotState(src_phys, dst_phys, src.pos);
            self.sessions.table[dst_phys] = .{
                .pos = src.pos,
                .token_pending = src.token_pending,
                .rng = .init(seed),
                .params = params,
                .state = src.state,
            };
            self.assertLedger();
            return id;
        }

        /// Bytes a prefix snapshot of `n_rows` occupies. A prefix cache sizes
        /// its entries with it, so the figure must not depend on a session.
        pub fn snapshotBytesFor(self: *const Self, n_rows: u32) u64 {
            if (comptime !can_snapshot) @compileError("runner cannot snapshot slot state");
            return self.sessions.snapshotBytesFor(n_rows);
        }

        /// Seed a FRESH logical session with a cached KV prefix. Any failure leaves it at `pos` 0.
        /// See "Prefix seeding" in the header. `blob` stays caller-owned.
        pub fn restorePrefix(self: *Self, id: u32, n_rows: u32, blob: []const u8) !void {
            if (comptime !can_snapshot) @compileError("runner cannot snapshot slot state");

            std.debug.assert(id < self.entries.len);
            std.debug.assert(self.entries[id].used);
            // A restore writes rows 0..n_rows, so it destroys the committed rows of a used session.
            std.debug.assert(self.liveOrSaved(id).pos == 0);
            try self.ensureResident(id, &.{});
            try self.sessions.restorePrefix(self.entries[id].phys.?, n_rows, blob);
            self.assertLedger();
        }

        /// Seed a FRESH logical session with a cached prefix and its stored final-row logits
        /// (`Sessions.restoreReady`). Placement and failure follow `restorePrefix`.
        pub fn restoreReady(
            self: *Self,
            id: u32,
            n_rows: u32,
            blob: []const u8,
            logits: []const f32,
        ) !u32 {
            if (comptime !can_snapshot) @compileError("runner cannot snapshot slot state");

            std.debug.assert(id < self.entries.len);
            std.debug.assert(self.entries[id].used);
            std.debug.assert(self.liveOrSaved(id).pos == 0);
            try self.ensureResident(id, &.{});
            defer self.assertLedger();

            return self.sessions.restoreReady(self.entries[id].phys.?, n_rows, blob, logits);
        }

        /// See `Sessions.copyFinalLogits`, keyed by logical id. The session completed a prefill
        /// this tick, so it is resident.
        pub fn copyFinalLogits(self: *Self, id: u32, out: []f32) void {
            std.debug.assert(id < self.entries.len);

            const e = &self.entries[id];
            std.debug.assert(e.used);
            std.debug.assert(e.phys != null);
            self.sessions.copyFinalLogits(e.phys.?, out);
        }

        /// Copy the first `n_rows` of a session's KV into `out`, for the prefix cache. The caller
        /// runs it in the tick that completed the prefill, so the session is resident and no
        /// placement runs.
        pub fn snapshotPrefix(self: *Self, id: u32, n_rows: u32, out: []u8) !void {
            if (comptime !can_snapshot) @compileError("runner cannot snapshot slot state");

            std.debug.assert(id < self.entries.len);
            const e = &self.entries[id];
            std.debug.assert(e.used);
            std.debug.assert(e.phys != null);
            try self.sessions.snapshotPrefix(e.phys.?, n_rows, out);
        }

        /// Release a logical session. Its credit has no later charge. See "Ledger" in the header.
        pub fn destroy(self: *Self, id: u32) void {
            std.debug.assert(id < self.entries.len);

            const e = &self.entries[id];
            std.debug.assert(e.used);
            if (e.phys) |s| {
                self.sessions.table[s].state = .free;
                self.phys_owner[s] = null;
            }

            self.dropBlob(e);
            std.debug.assert(e.blob.bytes.len == 0);
            e.* = .{};
            self.assertLedger();
        }

        /// The pool's occupancy and swap counters, for the serve layer's
        /// periodic witness. The scan is bounded by the logical session count.
        pub fn stats(self: *const Self) ServeStats {
            var resident: u32 = 0;
            for (self.entries) |*e| {
                if (e.phys != null) {
                    resident += 1;
                }
            }

            return .{
                .slots_resident = resident,
                .n_physical = @intCast(self.phys_owner.len),
                .n_logical = @intCast(self.entries.len),
                .ram_used = self.ram_used,
                .blob_budget = if (self.entries.len <= self.phys_owner.len)
                    .absent
                else if (self.blob_budget) |budget|
                    .{ .bounded = budget }
                else
                    .unenforced,
                .evicts = self.evicts,
                .restores = self.restores,
                .evict_ms = self.evict_ms,
                .restore_ms = self.restore_ms,
            };
        }

        pub fn setParams(self: *Self, id: u32, params: sampler.Params) void {
            self.liveOrSaved(id).params = params;
        }

        /// See `Sessions.reseed`. An evicted session takes the seed in its
        /// saved copy, and the restore carries the stream back to the slot.
        pub fn reseed(self: *Self, id: u32, seed: u64) void {
            self.liveOrSaved(id).rng = .init(seed);
        }

        /// See `Sessions.overridePending`. An evicted session takes the id in
        /// its saved copy, and the restore carries it back to the slot.
        pub fn overridePending(self: *Self, id: u32, token: u32) void {
            const s = self.liveOrSaved(id);
            std.debug.assert(s.state == .ready);
            std.debug.assert(token < self.sessions.vocab);
            s.token_pending = token;
        }

        pub fn posOf(self: *Self, id: u32) u32 {
            return self.liveOrSaved(id).pos;
        }

        pub fn capacityRemaining(self: *Self, id: u32) u32 {
            return self.sessions.runner.max_seq - self.posOf(id);
        }

        /// Pointer to whichever `Session` for `id` is authoritative: the live
        /// slot when the session is resident, and the saved copy when it is
        /// evicted.
        fn liveOrSaved(self: *Self, id: u32) *session.Session {
            std.debug.assert(id < self.entries.len);

            const e = &self.entries[id];
            std.debug.assert(e.used);
            return if (e.phys) |s| &self.sessions.table[s] else &e.saved;
        }

        pub fn feed(self: *Self, id: u32, tokens: []const u32, final: bool) !?u32 {
            try self.ensureResident(id, &.{});
            defer self.assertLedger();

            return self.sessions.feed(self.entries[id].phys.?, tokens, final);
        }

        pub fn decodeBatch(self: *Self, ids: []const u32, tokens_out: []u32) !void {
            var phys_buf: [max_phys]u32 = undefined;
            const phys = try self.residentPhys(ids, &phys_buf, null);
            defer self.assertLedger();

            return self.sessions.decodeBatch(phys, tokens_out);
        }

        pub fn advance(
            self: *Self,
            feed_chunk: ?FeedChunk,
            ids: []const u32,
            tokens_out: []u32,
        ) !?u32 {
            var phys_buf: [max_phys]u32 = undefined;

            // Pin the feed session's slot, so a decode placement cannot evict it in this tick.
            const pin: ?u32 = if (feed_chunk) |fc| self.entries[fc.id].phys else null;
            const decode_phys = try self.residentPhys(ids, &phys_buf, pin);
            var chunk: ?FeedChunk = null;
            if (feed_chunk) |fc| {
                try self.ensureResident(fc.id, decode_phys);
                chunk = .{
                    .id = self.entries[fc.id].phys.?,
                    .tokens = fc.tokens,
                    .final = fc.final,
                };
            }

            defer self.assertLedger();

            return self.sessions.advance(chunk, decode_phys, tokens_out);
        }

        /// Map logical decode ids to physical slots in `out`, in the order of `ids`. No placement
        /// evicts another batch member or `pin`, the slot of the feed session.
        fn residentPhys(self: *Self, ids: []const u32, out: []u32, pin: ?u32) ![]u32 {
            std.debug.assert(ids.len <= out.len);

            var exclude: [max_phys + 1]u32 = undefined;
            for (ids, 0..) |id, i| {
                @memcpy(exclude[0..i], out[0..i]); // already-placed batch members
                var n = i;
                if (pin) |p| {
                    exclude[n] = p;
                    n += 1;
                }

                try self.ensureResident(id, exclude[0..n]);
                out[i] = self.entries[id].phys.?;
            }

            return out[0..ids.len];
        }

        /// Bound on the physical slots a tick touches. The runner caps
        /// `n_slots` at `limits.max_vram_slots`, so the host scratch is small
        /// and fixed.
        const max_phys = limits.max_vram_slots;
    };
}

// -- helpers ----------------------------------------------------------------

fn msBetween(t0: std.Io.Timestamp, t1: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
}

fn indexOf(haystack: []const u32, needle: u32) ?usize {
    for (haystack, 0..) |h, i| {
        if (h == needle) return i;
    }

    return null;
}

// -- tests ------------------------------------------------------------------

const test_fakes = @import("../runtime/test_fakes.zig");
const test_vocab = test_fakes.vocab;
const fake_row_bytes: usize = @sizeOf(u32);

/// The recording fake drives three surfaces: the FSM with its LRU, the RAM ledger, and the prefix
/// calls. A per-slot marker makes a round trip observable, and a blob costs one u32 per committed
/// row. `stepBatch` sets the marker to the token. The ledger pool has 2 physical slots, 3 logical.
const FakeRunner = struct {
    n_slots: u32,
    max_seq: u32 = 64,
    logits: []f32,
    markers: []u32, // per physical slot, a stand-in for KV content
    evicts: u32 = 0,
    restores: u32 = 0,
    resets: u32 = 0,
    last_evict_rows: u32 = 0,
    last_restore_rows: u32 = 0,

    fn initFake(gpa: Allocator, n_slots: u32) !FakeRunner {
        const logits = try gpa.alloc(f32, n_slots * test_vocab);
        @memset(logits, 0);
        const markers = try gpa.alloc(u32, n_slots);
        @memset(markers, 0);
        return .{ .n_slots = n_slots, .logits = logits, .markers = markers };
    }

    fn deinitFake(self: *FakeRunner, gpa: Allocator) void {
        gpa.free(self.logits);
        gpa.free(self.markers);
    }

    pub fn evictSlotBytesFor(self: *const FakeRunner, n_rows: u32) u64 {
        _ = self;
        return @as(u64, n_rows) * fake_row_bytes;
    }

    pub fn evictSlot(self: *FakeRunner, slot: u32, n_rows: u32, out: []u8) !void {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(out.len == n_rows * fake_row_bytes);

        std.mem.writeInt(u32, out[0..4], self.markers[slot], .little);
        self.evicts += 1;
        self.last_evict_rows = n_rows;
    }

    pub fn restoreSlot(self: *FakeRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(blob.len == n_rows * fake_row_bytes);

        self.markers[slot] = std.mem.readInt(u32, blob[0..4], .little);
        self.restores += 1;
        self.last_restore_rows = n_rows;
    }

    pub fn resetSlotState(self: *FakeRunner, slot: u32) void {
        self.markers[slot] = 0;
        self.resets += 1;
    }

    pub fn maxStepRows(self: *const FakeRunner) u32 {
        return self.n_slots;
    }

    pub fn stepBatch(
        self: *FakeRunner,
        rows: []const @import("../runtime/step_row.zig").StepRow,
    ) !void {
        for (rows) |r| {
            self.markers[r.slot] = r.token; // KV "becomes" the last token
        }

        test_fakes.writeCannedLogits(self.logits, rows);
    }
};

const Sessions = session.Sessions;
const TestRes = Residency(Sessions(FakeRunner));

/// The truth the incremental ledger must equal.
fn blobSum(res: *const TestRes) u64 {
    var total: u64 = 0;
    for (res.entries) |*e| {
        total += e.blob.bytes.len;
    }

    return total;
}

fn expectLedger(res: *const TestRes, expected: u64) !void {
    try std.testing.expectEqual(expected, res.ram_used);
    try std.testing.expectEqual(expected, blobSum(res));
    if (res.blob_budget) |budget| {
        try std.testing.expect(res.ram_used <= budget);
    }
}

/// A two-row prefix blob carrying `marker`, the value the fake's KV holds after
/// a session steps that token.
fn prefixBlob(marker: u32) [2 * fake_row_bytes]u8 {
    var blob = [_]u8{0} ** (2 * fake_row_bytes);
    std.mem.writeInt(u32, blob[0..4], marker, .little);
    return blob;
}

/// The pool holds pointers into itself, so it is built in place at a caller
/// address that outlives the test.
const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    fake: FakeRunner,
    pool: Sessions(FakeRunner),
    res: TestRes,

    fn setUp(self: *Fixture, gpa: Allocator, n_slots: u32, n_logical: u32, budget: ?u64) !void {
        self.arena_state = .init(gpa);
        errdefer self.arena_state.deinit();

        const arena = self.arena_state.allocator();
        self.fake = try FakeRunner.initFake(gpa, n_slots);
        self.pool = try Sessions(FakeRunner).init(arena, &self.fake);
        self.res = try TestRes.init(arena, gpa, &self.pool, n_logical, budget);
    }

    fn tearDown(self: *Fixture, gpa: Allocator) void {
        self.res.deinit();
        self.fake.deinitFake(gpa);
        self.arena_state.deinit();
    }
};

test "residency: more logical sessions than physical slots, streams survive eviction" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    var fake = try FakeRunner.initFake(gpa, 2); // 2 physical slots
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena, &fake);
    var res = try TestRes.init(arena, gpa, &pool, 4, null); // 4 logical sessions
    defer res.deinit();

    var ids: [4]u32 = undefined;
    for (&ids, 0..) |*id, k| {
        id.* = try res.create(@intCast(k), .{});
        const first = (try res.feed(id.*, &.{@intCast(k)}, true)).?; // argmax(k+1)
        try std.testing.expectEqual(@as(u32, @intCast((k + 1) % test_vocab)), first);
    }

    // Only 2 of the 4 can be resident. At least the other 2 were evicted.
    try std.testing.expect(fake.evicts >= 2);

    // Decode each session one at a time. A non-resident one must be restored
    // (its marker reinstated) so its chain continues correctly.
    for (ids, 0..) |id, k| {
        var out: [1]u32 = undefined;
        try res.decodeBatch(&.{id}, &out);
        // pending was (k+1). Step it -> (k+2)
        try std.testing.expectEqual(@as(u32, @intCast((k + 2) % test_vocab)), out[0]);
    }

    // Destroy frees the slot/blob. A fresh session reuses capacity.
    res.destroy(ids[0]);
    const reused = try res.create(99, .{});
    _ = try res.feed(reused, &.{1}, true);
    res.destroy(reused);
    for (ids[1..]) |id| {
        res.destroy(id);
    }
}

test "residency: identity when physical >= logical (no eviction)" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    var fake = try FakeRunner.initFake(gpa, 4);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena, &fake);
    var res = try TestRes.init(arena, gpa, &pool, 4, null); // logical == physical
    defer res.deinit();

    const a = try res.create(1, .{});
    const b = try res.create(2, .{});
    _ = try res.feed(a, &.{2}, true);
    _ = try res.feed(b, &.{6}, true);
    var out: [2]u32 = undefined;
    try res.decodeBatch(&.{ a, b }, &out);
    try std.testing.expectEqual(@as(u32, 4), out[0]); // (2+1)->3 pending, step ->4
    try std.testing.expectEqual(@as(u32, 0), out[1]); // (6+1)->7 pending, step ->0
    try std.testing.expectEqual(@as(u32, 0), fake.evicts); // nothing evicted
}

test "residency: an overridden pending token survives an eviction and steps after the restore" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    // `a` samples 1 and is then evicted by the third session, so its pending
    // token lives in the saved copy.
    const a = try res.create(1, .{});
    try std.testing.expectEqual(@as(u32, 1), (try res.feed(a, &.{0}, true)).?);
    const b = try res.create(2, .{});
    _ = try res.feed(b, &.{0}, true);
    const c = try res.create(3, .{});
    _ = try res.feed(c, &.{0}, true);
    try std.testing.expectEqual(@as(?u32, null), res.entries[a].phys);

    res.overridePending(a, 5);
    try std.testing.expectEqual(@as(u32, 5), res.entries[a].saved.token_pending);

    // The restore brings the override back, so the step commits 5 and the
    // chain samples 6. A lost override steps the sampled 1 and samples 2.
    var out: [1]u32 = undefined;
    try res.decodeBatch(&.{a}, &out);
    try std.testing.expectEqual(@as(u32, 6), out[0]);
    try std.testing.expectEqual(@as(u32, 5), fx.fake.markers[res.entries[a].phys.?]);

    // A resident session takes the override in its live slot.
    res.overridePending(a, 2);
    try res.decodeBatch(&.{a}, &out);
    try std.testing.expectEqual(@as(u32, 3), out[0]);

    res.destroy(a);
    res.destroy(b);
    res.destroy(c);
}

test "residency ledger: the incremental count equals the blob sum across a trace" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, 64);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    // Two sessions fill the two slots, and the third evicts the older of them.
    // The ledger then reads exactly one one-row blob.
    const a = try res.create(1, .{});
    _ = try res.feed(a, &.{0}, true);
    const b = try res.create(2, .{});
    _ = try res.feed(b, &.{0}, true);
    try expectLedger(res, 0);

    const c = try res.create(3, .{});
    _ = try res.feed(c, &.{0}, true);
    try expectLedger(res, 1 * fake_row_bytes);
    try std.testing.expectEqual(@as(?u32, null), res.entries[a].phys);

    // Every swap charges the victim's rows and credits the restored session's rows.
    var out: [1]u32 = undefined;
    try res.decodeBatch(&.{a}, &out);
    try expectLedger(res, 1 * fake_row_bytes); // b's row, a's credited back
    try res.decodeBatch(&.{b}, &out);
    try expectLedger(res, 1 * fake_row_bytes); // c's row, b's credited back

    res.destroy(c); // the one session still out of VRAM
    try expectLedger(res, 0);
    res.destroy(a);
    res.destroy(b);
    try expectLedger(res, 0);
}

test "residency ledger: an over-budget eviction refuses and changes nothing" {
    const gpa = std.testing.allocator;
    // A budget of exactly one one-row blob. The second eviction cannot fit.
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, 1 * fake_row_bytes);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    const a = try res.create(1, .{});
    _ = try res.feed(a, &.{0}, true);
    const b = try res.create(2, .{});
    _ = try res.feed(b, &.{0}, true);
    const c = try res.create(3, .{});
    _ = try res.feed(c, &.{0}, true); // evicts a, and the budget is spent
    try expectLedger(res, 1 * fake_row_bytes);

    const b_phys = res.entries[b].phys;
    const c_phys = res.entries[c].phys;
    const a_blob_len = res.entries[a].blob.bytes.len;
    const a_blob_rows = res.entries[a].blob.n_rows;
    const evicts_before = fx.fake.evicts;

    // Restoring `a` needs a victim, and every victim needs a second blob.
    var out: [1]u32 = undefined;
    try std.testing.expectError(error.RamBudgetExceeded, res.decodeBatch(&.{a}, &out));

    try expectLedger(res, 1 * fake_row_bytes);
    try std.testing.expectEqual(b_phys, res.entries[b].phys);
    try std.testing.expectEqual(c_phys, res.entries[c].phys);
    try std.testing.expectEqual(a_blob_len, res.entries[a].blob.bytes.len);
    try std.testing.expectEqual(a_blob_rows, res.entries[a].blob.n_rows);
    try std.testing.expectEqual(evicts_before, fx.fake.evicts); // no device copy ran

    res.destroy(a);
    res.destroy(b);
    res.destroy(c);
}

test "residency ledger: a session with no committed rows evicts without a blob" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    // Place three sessions without feeding any of them, so every `pos` is 0.
    const a = try res.create(1, .{});
    try res.ensureResident(a, &.{});
    const b = try res.create(2, .{});
    try res.ensureResident(b, &.{});
    const resets_before = fx.fake.resets;

    const c = try res.create(3, .{});
    try res.ensureResident(c, &.{}); // evicts a, which has nothing to save
    try expectLedger(res, 0);
    try std.testing.expectEqual(@as(u32, 0), fx.fake.evicts);
    try std.testing.expectEqual(false, res.entries[a].ever_resident);
    try std.testing.expectEqual(@as(usize, 0), res.entries[a].blob.bytes.len);

    // The next placement therefore takes the fresh path, not a restore.
    try res.ensureResident(a, &.{});
    try std.testing.expectEqual(@as(u32, 0), fx.fake.restores);
    try std.testing.expectEqual(resets_before + 2, fx.fake.resets);

    res.destroy(a);
    res.destroy(b);
    res.destroy(c);
}

test "residency ledger: eviction moves the committed rows only" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    const a = try res.create(1, .{});
    _ = try res.feed(a, &.{ 0, 1, 2 }, true); // pos 3
    const b = try res.create(2, .{});
    _ = try res.feed(b, &.{0}, true); // pos 1
    const c = try res.create(3, .{});
    _ = try res.feed(c, &.{0}, true); // evicts a

    try std.testing.expectEqual(@as(u32, 3), fx.fake.last_evict_rows);
    try std.testing.expectEqual(@as(u32, 3), res.entries[a].blob.n_rows);
    try expectLedger(res, 3 * fake_row_bytes);

    // The restore states the same row count the eviction wrote, and it credits
    // those three rows back. Only b, the victim of this swap, is still charged.
    var out: [1]u32 = undefined;
    try res.decodeBatch(&.{a}, &out);
    try std.testing.expectEqual(@as(u32, 3), fx.fake.last_restore_rows);
    try expectLedger(res, 1 * fake_row_bytes); // b's one row

    // The reported counters are the fake's own device-copy counts, and the
    // occupancy is the two slots a and c hold after the swap back.
    const s = res.stats();
    try std.testing.expectEqual(@as(u64, fx.fake.evicts), s.evicts);
    try std.testing.expectEqual(@as(u64, fx.fake.restores), s.restores);
    try std.testing.expectEqual(@as(u64, 2), s.evicts); // a, then b
    try std.testing.expectEqual(@as(u64, 1), s.restores); // a came back
    try std.testing.expectEqual(res.ram_used, s.ram_used);
    try std.testing.expectEqual(@as(u32, 2), s.slots_resident);
    try std.testing.expectEqual(@as(u32, 2), s.n_physical);
    try std.testing.expectEqual(@as(u32, 3), s.n_logical);
    try std.testing.expectEqual(BlobTier.unenforced, s.blob_budget);

    res.destroy(a);
    res.destroy(b);
    res.destroy(c);
}

test "residency ledger: a restore releases the blob, and the next eviction re-creates it" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    const a = try res.create(1, .{});
    _ = try res.feed(a, &.{ 0, 1 }, true); // pos 2
    const b = try res.create(2, .{});
    _ = try res.feed(b, &.{0}, true);
    const c = try res.create(3, .{});
    _ = try res.feed(c, &.{0}, true); // evicts a at pos 2
    try expectLedger(res, 2 * fake_row_bytes);
    try std.testing.expectEqual(true, res.entries[a].ever_resident);

    // Closing `c` frees a slot, so restoring `a` needs no victim. The release is the only change.
    res.destroy(c);
    var out: [1]u32 = undefined;
    try res.decodeBatch(&.{a}, &out); // pos 2 -> 3
    try std.testing.expectEqual(@as(u32, 1), fx.fake.restores);
    try std.testing.expectEqual(@as(usize, 0), res.entries[a].blob.bytes.len);
    try std.testing.expectEqual(false, res.entries[a].ever_resident);
    try expectLedger(res, 0);

    // A later eviction re-creates the blob at the current rows of `a`.
    try res.ensureResident(b, &.{}); // make `a` the LRU entry again
    const d = try res.create(4, .{});
    const evicts_before = fx.fake.evicts;
    try res.ensureResident(d, &.{}); // evicts a at pos 3
    try std.testing.expectEqual(evicts_before + 1, fx.fake.evicts);
    try std.testing.expectEqual(@as(?u32, null), res.entries[a].phys);
    try std.testing.expectEqual(true, res.entries[a].ever_resident);
    try std.testing.expectEqual(@as(u32, 3), res.entries[a].blob.n_rows);
    try expectLedger(res, 3 * fake_row_bytes);

    res.destroy(a);
    res.destroy(b);
    res.destroy(d);
}

test "residency prefix: a restore places the session and charges nothing" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    const a = try res.create(1, .{});
    const blob = prefixBlob(2);
    try std.testing.expectEqual(@as(u64, blob.len), res.snapshotBytesFor(2));
    try res.restorePrefix(a, 2, &blob);

    try std.testing.expect(res.entries[a].phys != null);
    try std.testing.expectEqual(@as(u32, 2), res.posOf(a));
    try std.testing.expectEqual(@as(u32, 2), fx.fake.markers[res.entries[a].phys.?]);
    try std.testing.expectEqual(@as(u32, 1), fx.fake.restores);
    try expectLedger(res, 0); // the blob is the caller's, not the tier's

    res.destroy(a);
}

test "residency prefix: a restored session evicts and returns before its first feed" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    // `a` holds a two-row prefix and nothing else. Two more sessions then take
    // both slots, so `a` spills with only restored rows to its name.
    const a = try res.create(1, .{});
    const blob = prefixBlob(2);
    try res.restorePrefix(a, 2, &blob);
    const b = try res.create(2, .{});
    _ = try res.feed(b, &.{0}, true);
    const c = try res.create(3, .{});
    _ = try res.feed(c, &.{0}, true); // evicts a

    try std.testing.expectEqual(@as(?u32, null), res.entries[a].phys);
    try std.testing.expectEqual(@as(u32, 2), fx.fake.last_evict_rows);
    try expectLedger(res, 2 * fake_row_bytes); // the tier owns the copy

    // The suffix feed brings `a` back and continues the same chain a cold
    // session that fed 1, 2, 3 produces.
    const first = (try res.feed(a, &.{3}, true)).?;
    try std.testing.expectEqual(@as(u32, 4), first); // argmax(3+1)
    try std.testing.expectEqual(@as(u32, 3), res.posOf(a));
    try std.testing.expectEqual(@as(u32, 2), fx.fake.restores); // the prefix, then the swap-in
    try expectLedger(res, 1 * fake_row_bytes); // b's row, a's credited back

    var out: [1]u32 = undefined;
    try res.decodeBatch(&.{a}, &out);
    try std.testing.expectEqual(@as(u32, 5), out[0]); // pending 4 -> 5
    try std.testing.expectEqual(@as(u32, 4), res.posOf(a));

    res.destroy(a);
    res.destroy(b);
    res.destroy(c);
}

test "residency prefix: restoreReady places the session decoding-ready and charges nothing" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    const a = try res.create(1, .{});
    var logits = [_]f32{0} ** test_vocab;
    logits[5] = 1;
    const blob = prefixBlob(2);
    const first = try res.restoreReady(a, 2, &blob, &logits);
    try std.testing.expectEqual(@as(u32, 5), first);
    try std.testing.expect(res.entries[a].phys != null);
    try std.testing.expectEqual(@as(u32, 2), res.posOf(a));
    try expectLedger(res, 0); // blob and logits stay the caller's

    // The session decodes on from the restored rows: pending 5 steps to
    // marker 5, and the chain samples 6.
    var out: [1]u32 = undefined;
    try res.decodeBatch(&.{a}, &out);
    try std.testing.expectEqual(@as(u32, 6), out[0]);
    try std.testing.expectEqual(@as(u32, 3), res.posOf(a));

    res.destroy(a);
}

test "residency prefix: a snapshot reads a resident slot and moves nothing" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    const a = try res.create(1, .{});
    _ = try res.feed(a, &.{ 0, 1 }, true); // pos 2, marker 1
    const phys = res.entries[a].phys;

    var out = [_]u8{0} ** (2 * fake_row_bytes);
    try res.snapshotPrefix(a, 2, &out);
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, out[0..4], .little));
    try std.testing.expectEqual(phys, res.entries[a].phys);
    try std.testing.expectEqual(@as(u32, 2), res.posOf(a));
    try expectLedger(res, 0);

    // The device copy ran, but it is not an eviction, so the tier's counters
    // stand still.
    try std.testing.expectEqual(@as(u32, 1), fx.fake.evicts);
    try std.testing.expectEqual(@as(u64, 0), res.stats().evicts);

    res.destroy(a);
}

test "residency prefix: an unfundable placement refuses and leaves a cold session" {
    const gpa = std.testing.allocator;
    // A budget of exactly one one-row blob, spent by the third session.
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 4, 1 * fake_row_bytes);
    defer fx.tearDown(gpa);

    const res = &fx.res;

    const a = try res.create(1, .{});
    _ = try res.feed(a, &.{0}, true);
    const b = try res.create(2, .{});
    _ = try res.feed(b, &.{0}, true);
    const c = try res.create(3, .{});
    _ = try res.feed(c, &.{0}, true); // evicts a, and the budget is spent
    try expectLedger(res, 1 * fake_row_bytes);

    const b_phys = res.entries[b].phys;
    const restores_before = fx.fake.restores;

    // Seeding `d` needs a slot, and freeing one needs a second blob.
    const d = try res.create(4, .{});
    const blob = prefixBlob(2);
    try std.testing.expectError(error.RamBudgetExceeded, res.restorePrefix(d, 2, &blob));

    // The caller can still prefill `d` cold: it is unplaced at position zero,
    // exactly as `create` left it.
    try std.testing.expectEqual(@as(?u32, null), res.entries[d].phys);
    try std.testing.expectEqual(@as(u32, 0), res.posOf(d));
    try std.testing.expectEqual(b_phys, res.entries[b].phys);
    try std.testing.expectEqual(restores_before, fx.fake.restores);
    try expectLedger(res, 1 * fake_row_bytes);

    res.destroy(a);
    res.destroy(b);
    res.destroy(c);
    res.destroy(d);
}

test "residency ledger: deinit drains every blob" {
    const gpa = std.testing.allocator;
    var fx: Fixture = undefined;
    try fx.setUp(gpa, 2, 3, null);
    const res = &fx.res;

    const a = try res.create(1, .{});
    _ = try res.feed(a, &.{0}, true);
    const b = try res.create(2, .{});
    _ = try res.feed(b, &.{0}, true);
    const c = try res.create(3, .{});
    _ = try res.feed(c, &.{0}, true); // evicts a
    try expectLedger(res, 1 * fake_row_bytes);

    // No session is destroyed: deinit alone must drain the ledger to zero.
    fx.tearDown(gpa);
}

/// A runner with no swap surface, so `Residency` compiles its identity arms.
const NoSwapRunner = struct {
    n_slots: u32,
    max_seq: u32 = 64,
    logits: []f32,

    pub fn maxStepRows(self: *const NoSwapRunner) u32 {
        return self.n_slots;
    }

    pub fn stepBatch(
        self: *NoSwapRunner,
        rows: []const @import("../runtime/step_row.zig").StepRow,
    ) !void {
        test_fakes.writeCannedLogits(self.logits, rows);
    }
};

test "residency: a runner without evictSlot is the identity pool" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const logits = try gpa.alloc(f32, 2 * test_vocab);
    defer gpa.free(logits);

    @memset(logits, 0);
    var fake: NoSwapRunner = .{ .n_slots = 2, .logits = logits };
    var pool = try Sessions(NoSwapRunner).init(arena, &fake);
    const Res = Residency(Sessions(NoSwapRunner));
    try std.testing.expect(!Res.can_evict);
    try std.testing.expect(!Res.can_snapshot);
    var res = try Res.init(arena, gpa, &pool, 2, null); // n_logical == n_physical
    defer res.deinit();

    var ids: [2]u32 = undefined;
    for (&ids, 0..) |*id, k| {
        id.* = try res.create(@intCast(k), .{});
        const first = (try res.feed(id.*, &.{@intCast(k)}, true)).?; // argmax(k+1)
        try std.testing.expectEqual(@as(u32, @intCast((k + 1) % test_vocab)), first);
    }

    // Both stay resident, and one batched decode steps both chains.
    var out: [2]u32 = undefined;
    try res.decodeBatch(&ids, &out);
    try std.testing.expectEqual(@as(u32, 2 % test_vocab), out[0]);
    try std.testing.expectEqual(@as(u32, 3 % test_vocab), out[1]);
}

// -- the residency trace property -------------------------------------------
//
// The oracle is a plain pool with one physical slot per logical session, so it never evicts. The
// pool under test runs the same trace on fewer slots. Every session must read the same in both
// after each step, so an evict and the later restore stay invisible.

const trace_logical_max = 5;
const trace_steps = 48;

const ResidencyTrace = struct {
    fx: Fixture,
    oracle_fake: FakeRunner,
    oracle: Sessions(FakeRunner),
    /// The oracle id of each logical id, null for a free one.
    oracle_id: [trace_logical_max]?u32 = @splat(null),

    fn setUp(
        self: *ResidencyTrace,
        gpa: Allocator,
        n_slots: u32,
        n_logical: u32,
        budget: ?u64,
    ) !void {
        try self.fx.setUp(gpa, n_slots, n_logical, budget);
        errdefer self.fx.tearDown(gpa);

        self.oracle_fake = try FakeRunner.initFake(gpa, n_logical);
        errdefer self.oracle_fake.deinitFake(gpa);

        self.oracle = try Sessions(FakeRunner).init(
            self.fx.arena_state.allocator(),
            &self.oracle_fake,
        );
        self.oracle_id = @splat(null);
    }

    fn tearDown(self: *ResidencyTrace, gpa: Allocator) void {
        self.oracle_fake.deinitFake(gpa);
        self.fx.tearDown(gpa);
    }

    /// The marker that the KV of logical session `id` holds: the live slot,
    /// or the first row of its blob.
    fn marker(self: *const ResidencyTrace, id: u32) u32 {
        const e = &self.fx.res.entries[id];
        if (e.phys) |s| return self.fx.fake.markers[s];
        return std.mem.readInt(u32, e.blob.bytes[0..4], .little);
    }

    /// The ledger equals the blob sum under the budget, a resident session
    /// holds no blob, and an evicted one holds exactly its committed rows.
    /// Each session matches its oracle twin field by field.
    fn expectAgree(self: *const ResidencyTrace) !void {
        const res = &self.fx.res;
        try std.testing.expectEqual(blobSum(res), res.ram_used);
        if (res.blob_budget) |budget| {
            try std.testing.expect(res.ram_used <= budget);
        }

        for (res.entries, 0..) |*e, i| {
            const id: u32 = @intCast(i);
            const twin = self.oracle_id[i] orelse {
                try std.testing.expect(!e.used);
                continue;
            };
            try std.testing.expect(e.used);
            const s = if (e.phys) |p| res.sessions.table[p] else e.saved;
            const o = self.oracle.table[twin];
            if (e.phys) |p| {
                try std.testing.expectEqual(@as(?u32, id), res.phys_owner[p]);
                try std.testing.expectEqual(@as(usize, 0), e.blob.bytes.len);
            } else {
                try std.testing.expectEqual(@as(usize, s.pos) * fake_row_bytes, e.blob.bytes.len);
            }

            try std.testing.expectEqual(o.state, s.state);
            try std.testing.expectEqual(o.pos, s.pos);
            try std.testing.expectEqual(o.token_pending, s.token_pending);
            try std.testing.expectEqual(o.rng.s, s.rng.s);
            if (s.pos > 0) {
                try std.testing.expectEqual(self.oracle_fake.markers[twin], self.marker(id));
            }
        }
    }
};

fn liveIds(t: *const ResidencyTrace, want_ready: bool, out: *[trace_logical_max]u32) []u32 {
    var n: usize = 0;
    for (t.oracle_id, 0..) |twin, i| {
        const o = twin orelse continue;
        if (want_ready and t.oracle.table[o].state != .ready) continue;

        out[n] = @intCast(i);
        n += 1;
    }

    return out[0..n];
}

/// A refusal of the pool under test leaves every session where it was, so
/// the oracle skips the step. Only a pool with a budget can refuse for RAM.
fn refused(err: anyerror, budget: ?u64) bool {
    return (err == error.RamBudgetExceeded and budget != null) or err == error.NoEvictableSlot;
}

/// Random create, feed, decode, place, and destroy steps on fewer physical
/// slots than sessions: the ledger is the blob sum, a restore releases its
/// blob, and every session reads as on a pool that never evicts.
fn checkResidencyTrace(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const gpa = std.testing.allocator;

    // Pick a random pool shape.
    const n_slots = 2 + r.uintLessThan(u32, 2);
    const n_logical = n_slots + 1 + r.uintLessThan(u32, trace_logical_max - n_slots);
    const budget: ?u64 = if (r.boolean()) null else r.uintAtMost(u64, 24) * @as(
        u64,
        fake_row_bytes,
    );

    var t: ResidencyTrace = undefined;
    try t.setUp(gpa, n_slots, n_logical, budget);
    defer t.tearDown(gpa);

    const res = &t.fx.res;

    // Run random steps, and compare the two pools after each one.
    for (0..trace_steps) |_| {
        var ids_buf: [trace_logical_max]u32 = undefined;
        switch (r.uintLessThan(u8, 5)) {
            0 => {
                // Create a session on both pools.
                const create_seed = r.int(u64);
                const params: sampler.Params = .{ .temperature = if (r.boolean()) 0 else 0.8 };
                const id = res.create(create_seed, params) catch |err| {
                    try std.testing.expectError(err, t.oracle.create(create_seed, params));
                    continue;
                };
                t.oracle_id[id] = try t.oracle.create(create_seed, params);
            },
            1 => {
                // Feed random tokens to one session.
                const live = liveIds(&t, false, &ids_buf);
                if (live.len == 0) continue;

                const id = live[r.uintLessThan(usize, live.len)];
                var tokens: [3]u32 = undefined;
                for (&tokens) |*tk| {
                    tk.* = r.uintLessThan(u32, test_vocab);
                }

                const fed = tokens[0 .. 1 + r.uintLessThan(usize, tokens.len)];
                const final = r.boolean();
                const got = res.feed(id, fed, final) catch |err| {
                    if (refused(err, budget)) continue;
                    try std.testing.expectError(err, t.oracle.feed(t.oracle_id[id].?, fed, final));
                    continue;
                };
                try std.testing.expectEqual(try t.oracle.feed(t.oracle_id[id].?, fed, final), got);
            },
            2 => {
                // Decode a random batch of ready sessions.
                const ready = liveIds(&t, true, &ids_buf);
                if (ready.len == 0) continue;

                r.shuffle(u32, ready);
                const ids = ready[0..@min(ready.len, 1 + r.uintLessThan(usize, n_slots))];
                var twins: [trace_logical_max]u32 = undefined;
                for (ids, twins[0..ids.len]) |id, *o| {
                    o.* = t.oracle_id[id].?;
                }

                var got: [trace_logical_max]u32 = undefined;
                res.decodeBatch(ids, got[0..ids.len]) catch |err| {
                    if (refused(err, budget)) continue;
                    try std.testing.expectError(
                        err,
                        t.oracle.decodeBatch(twins[0..ids.len], got[0..ids.len]),
                    );
                    continue;
                };
                var want: [trace_logical_max]u32 = undefined;
                try t.oracle.decodeBatch(twins[0..ids.len], want[0..ids.len]);
                try std.testing.expectEqualSlices(u32, want[0..ids.len], got[0..ids.len]);
            },
            3 => {
                // Place one session.
                const live = liveIds(&t, false, &ids_buf);
                if (live.len == 0) continue;

                res.ensureResident(live[r.uintLessThan(usize, live.len)], &.{}) catch |err| {
                    if (!refused(err, budget)) return err;
                };
            },
            else => {
                // Destroy one session on both pools.
                const live = liveIds(&t, false, &ids_buf);
                if (live.len == 0) continue;

                const id = live[r.uintLessThan(usize, live.len)];
                res.destroy(id);
                t.oracle.destroy(t.oracle_id[id].?);
                t.oracle_id[id] = null;
            },
        }

        try t.expectAgree();
    }

    // Destroy the rest, and check that the ledger drains.
    for (t.oracle_id, 0..) |twin, i| {
        if (twin != null) {
            res.destroy(@intCast(i));
        }
    }

    try std.testing.expectEqual(@as(u64, 0), res.ram_used);
}

test "property: random traces keep the ledger and match a pool that never evicts" {
    try prop.forAll(
        .{ .site = 0x2E51_DE4C, .cases = 400 },
        {},
        u64,
        prop.genSeed(void),
        checkResidencyTrace,
    );
}
