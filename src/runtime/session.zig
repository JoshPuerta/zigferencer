//! The session pool: per-conversation decode state over the KV slots of a
//! runner. A `Session` is one conversation: a KV slot, a position that persists
//! across turns, a seeded RNG, and sampling params.
//!
//! Comptime. `Sessions` is generic over the runner. `@hasDecl` keeps each optional hook free for a
//! runner without it, and a missing required hook stops the build.
//!
//! `Sessions` advances any subset of them one token per `decodeBatch` tick,
//! which is the multi-session liveliness primitive. It feeds prompt turns
//! through the same batched forward (`feed`), so chunked prefill and batched
//! decode run one code path. Both are shells over `advance`, the composed tick
//! that can fold one prefill chunk and a decode step into ONE forward call.
//! That is the continuous-batching boundary of the serve scheduler, where the
//! prefill rows use the weight pass of the decode rows.
//!
//! The engine boundary is token ids in, token ids out. Chat templating and
//! tokenization stay caller-side, and the caller never sees logits. Sampling
//! happens here, against the RNG of the session.
//!
//! The pending-token rule makes the turn boundary explicit. Every `decodeBatch`
//! tick steps the token sampled before it, which commits the KV of that token,
//! and it samples a new one. The new token stays *uncommitted* until the next
//! tick or feed. When the caller ends a turn, the pending token was therefore
//! never written into context. On a budget stop the caller saw the token, so it
//! prepends the token to the `feed` of the next turn. On an EOS stop nothing is
//! needed. The chat continuation templates already open with the turn-end
//! token, precisely because the terminal token never entered the KV. This is
//! the deliberate stop asymmetry of `generate.decode`, restated as a value at
//! the boundary instead of a loop quirk.
//!
//! State machine per session. A violation is a programmer error (assert).
//! Running out of KV positions is runtime data (`error.ContextFull`):
//!
//!     free --create--> feeding --feed(final)--> ready --decodeBatch--> ready
//!       ^                 ^                       |
//!       |                 +------- feed ----------+   (next turn)
//!       +---- destroy ----+-----------------------+
//!
//! Atomic capacity. `feed`, `decodeBatch`, and `advance` check the KV capacity of the whole call
//! before they step any row. On `error.ContextFull` no session changes.
//!
//! Compact logits. The runner writes logits only for the want-logits rows: the final row of the
//! chunk when `final`, then every decode row. They land at `logits[0..]` in that order, not at
//! their batch-row offset. So the buffer needs `logitRows` rows even for a wider batch.
//!
//! Prefix seeding. `restorePrefix` makes a fresh session read as one that fed `n_rows` non-final
//! tokens: the rows are committed, the turn is open, and no token is pending. `pos` is the one
//! variable the rest of the engine reads, so the suffix feed and the capacity checks work
//! unchanged. A snapshot is a read, because a committed row is never rewritten and `pos` only
//! ascends.
//!
//! KV reuse. A slot needs no KV scrub on destroy or create, because a new session rewrites each
//! position before it reads it. A runner with non-positional recurrent state, such as the GDN
//! scan, must zero that state through `resetSlotState`.
//!
//! Runner surface. `Sessions` reaches a runner through a duck-typed surface. A renamed member
//! breaks only the call sites that instantiate it, and a GPU-only runner can break where no CPU
//! test looks. So `Sessions(RunnerT)` checks the rows it reads when the pool type is built, and
//! each family row checks the whole surface of its runners in `resolve/family.zig`. A runner
//! that holds recurrent state must declare `has_recurrent_state` and both slot hooks, because
//! `@hasDecl` alone lets a runner omit `resetSlotState` and inherit the state of the last
//! conversation.

const std = @import("std");
const Allocator = std.mem.Allocator;
const sampler = @import("sampler.zig");
const prompt_cache = @import("prompt_cache.zig");
const StepRow = @import("step_row.zig").StepRow;
const runner_surface = @import("runner_surface.zig");
const core_log = @import("../core/utils/log.zig");

const log = std.log.scoped(.serve);
const log_token = std.log.scoped(.token);

// -- types ------------------------------------------------------------------

pub const State = enum { free, feeding, ready };

/// One same-slot chunk of a turn's prompt tokens, for `advance`. It is the
/// value a caller (the serve scheduler) passes to the engine to fold prefill
/// rows into a decode tick. `final` marks the turn's last chunk: logits for
/// its last row, sample the first response token.
pub const FeedChunk = struct {
    id: u32,
    tokens: []const u32,
    final: bool,
};

/// One conversation's decode state. Plain value. The KV lives in the runner's
/// slot of the same index.
pub const Session = struct {
    /// Next KV write position. It persists across turns.
    pos: u32,
    /// Sampled, and not stepped so far (valid in `ready`).
    token_pending: u32,
    rng: std.Random.DefaultPrng,
    params: sampler.Params,
    state: State,
};

/// Fixed pool of sessions over the KV slots of `RunnerT`. Session id == slot index.
/// `RunnerT` is duck-typed: `stepBatch(rows)`, `logits` (rows at vocab stride), `n_slots`, and
/// `max_seq`.
pub fn Sessions(comptime RunnerT: type) type {
    return struct {
        const Self = @This();

        /// The runner type, so the layers above can gate on its hooks.
        pub const Runner = RunnerT;

        // Check the runner surface.
        comptime {
            runner_surface.requireSessionsSurface(RunnerT);
        }

        /// Whether `clone` is available: the runner must copy the state of a slot.
        pub const can_clone = @hasDecl(RunnerT, "cloneSlotState");

        /// Whether prefix snapshot and restore are available. It uses the hooks of the swap tier.
        pub const can_snapshot = @hasDecl(RunnerT, "evictSlot");

        runner: *RunnerT,
        table: []Session,
        /// Scratch rows for one `stepBatch` call.
        rows: []StepRow,
        /// The vocab stride of the compact logits rows, derived once at init.
        vocab: usize,
        /// The sampler scratch. The pool samples one row at a time, so one scratch serves all.
        sample_scratch: []f32,
        /// The session whose final chunk the latest advance completed. Its logits sit in row zero.
        last_final: ?u32 = null,

        /// Rows of the compact logits buffer. A fused GPU runner can declare fewer than its slots.
        fn logitRows(runner: *const RunnerT) u32 {
            if (comptime @hasDecl(RunnerT, "logitRowsMax")) return runner.logitRowsMax();
            return runner.n_slots;
        }

        pub fn init(arena: Allocator, runner: *RunnerT) Allocator.Error!Self {
            std.debug.assert(runner.n_slots >= 1);
            std.debug.assert(logitRows(runner) >= 1);
            std.debug.assert(runner.logits.len % logitRows(runner) == 0);

            const table = try arena.alloc(Session, runner.n_slots);
            for (table) |*s| {
                s.* = .{
                    .pos = 0,
                    .token_pending = 0,
                    .rng = .init(0),
                    .params = .{},
                    .state = .free,
                };
            }

            const vocab = runner.logits.len / logitRows(runner);
            return .{
                .runner = runner,
                .table = table,
                .rows = try arena.alloc(StepRow, runner.maxStepRows()),
                .vocab = vocab,
                .sample_scratch = try arena.alloc(f32, sampler.scratchLen(vocab)),
            };
        }

        /// Claim a free slot. The seed is explicit, so generation replays.
        pub fn create(self: *Self, seed: u64, params: sampler.Params) error{NoFreeSession}!u32 {
            std.debug.assert(params.temperature >= 0);
            std.debug.assert(params.top_p > 0);

            for (self.table, 0..) |*s, id| {
                if (s.state != .free) continue;

                if (comptime @hasDecl(RunnerT, "resetSlotState")) {
                    self.runner.resetSlotState(@intCast(id));
                }

                s.* = .{
                    .pos = 0,
                    .token_pending = 0,
                    .rng = .init(seed),
                    .params = params,
                    .state = .feeding,
                };
                log.debug(
                    "engine create: id {d}, slot {d}, max_seq {d}",
                    .{ id, id, self.runner.max_seq },
                );
                return @intCast(id);
            }

            return error.NoFreeSession;
        }

        /// Fork a session into a free slot: KV rows `0..pos`, `pos`, and the pending token.
        /// The clone gets its own seed and params. The source stays untouched.
        pub fn clone(
            self: *Self,
            source_id: u32,
            seed: u64,
            params: sampler.Params,
        ) error{NoFreeSession}!u32 {
            std.debug.assert(source_id < self.table.len);

            const source = &self.table[source_id];
            std.debug.assert(source.state != .free);
            std.debug.assert(params.temperature >= 0);
            std.debug.assert(params.top_p > 0);

            for (self.table, 0..) |*s, id| {
                if (s.state != .free) continue;

                std.debug.assert(id != source_id);
                self.runner.cloneSlotState(source_id, @intCast(id), source.pos);
                s.* = .{
                    .pos = source.pos,
                    .token_pending = source.token_pending,
                    .rng = .init(seed),
                    .params = params,
                    .state = source.state,
                };
                log.debug("engine clone: {d} -> {d}, pos {d}", .{ source_id, id, source.pos });
                return @intCast(id);
            }

            return error.NoFreeSession;
        }

        /// Bytes of a prefix snapshot of `n_rows`. A caller sizes a cache entry with it.
        pub fn snapshotBytesFor(self: *const Self, n_rows: u32) u64 {
            if (comptime !can_snapshot) @compileError("runner cannot snapshot slot state");

            std.debug.assert(n_rows >= 1);
            std.debug.assert(n_rows <= self.runner.max_seq);
            return self.runner.evictSlotBytesFor(n_rows);
        }

        /// Seed a fresh session with a cached KV prefix.
        /// `blob` must come from `snapshotPrefix` at the same `n_rows`. The caller keeps it.
        pub fn restorePrefix(self: *Self, id: u32, n_rows: u32, blob: []const u8) !void {
            if (comptime !can_snapshot) @compileError("runner cannot snapshot slot state");

            std.debug.assert(id < self.table.len);
            const s = &self.table[id];
            std.debug.assert(s.state == .feeding);
            std.debug.assert(s.pos == 0);
            // A cache file supplies the blob, so a mismatch is a miss, not a broken invariant.
            try prompt_cache.checkRows(n_rows, self.runner.max_seq);
            try prompt_cache.checkSizes(blob.len, self.snapshotBytesFor(n_rows), 0, 0);
            try self.runner.restoreSlot(id, n_rows, blob);
            s.pos = n_rows;
            log.debug("engine restore prefix: id {d}, slot {d}, {d} rows", .{ id, id, n_rows });
        }

        /// Seed a fresh session with a prefix and its final-row logits, then sample the first
        /// response token like a final feed. The session becomes `ready`, so a full hit costs
        /// no forward row. Returns the sampled token.
        pub fn restoreReady(
            self: *Self,
            id: u32,
            n_rows: u32,
            blob: []const u8,
            logits: []const f32,
        ) !u32 {
            if (comptime !can_snapshot) @compileError("runner cannot snapshot slot state");

            try prompt_cache.checkSizes(0, 0, logits.len, self.vocab);
            try self.restorePrefix(id, n_rows, blob);
            const s = &self.table[id];
            std.debug.assert(s.state == .feeding);
            std.debug.assert(s.pos == n_rows);
            s.token_pending = try sampler.sample(
                logits,
                s.params,
                s.rng.random(),
                self.sample_scratch,
            );
            s.state = .ready;
            log.debug(
                "engine restore ready: id {d}, {d} rows, first {d}",
                .{ id, n_rows, s.token_pending },
            );
            return s.token_pending;
        }

        /// Copy the final-row logits of the latest advance into `out`. Legal only for the
        /// session that the latest advance completed, until the next forward overwrites them.
        pub fn copyFinalLogits(self: *const Self, id: u32, out: []f32) void {
            std.debug.assert(self.last_final != null);
            std.debug.assert(self.last_final.? == id);
            std.debug.assert(out.len == self.vocab);

            @memcpy(out, self.runner.logits[0..self.vocab]);
        }

        /// Copy the first `n_rows` of a live session's slot into `out`. The session goes on
        /// decoding unchanged.
        pub fn snapshotPrefix(self: *Self, id: u32, n_rows: u32, out: []u8) !void {
            if (comptime !can_snapshot) @compileError("runner cannot snapshot slot state");

            std.debug.assert(id < self.table.len);
            const s = &self.table[id];
            std.debug.assert(s.state != .free);
            std.debug.assert(n_rows <= s.pos); // uncommitted rows hold no state to cache
            std.debug.assert(out.len == @as(usize, @intCast(self.snapshotBytesFor(n_rows))));
            try self.runner.evictSlot(id, n_rows, out);
            log.debug(
                "engine snapshot prefix: id {d}, slot {d}, {d} of {d} rows",
                .{ id, id, n_rows, s.pos },
            );
        }

        /// Release a slot.
        pub fn destroy(self: *Self, id: u32) void {
            std.debug.assert(id < self.table.len);
            std.debug.assert(self.table[id].state != .free);

            log.debug("engine destroy: id {d}, pos {d}", .{ id, self.table[id].pos });
            self.table[id].state = .free;
            self.table[id].pos = 0;
        }

        /// KV positions still unwritten. The pending token of a `ready` session takes one.
        pub fn capacityRemaining(self: *const Self, id: u32) u32 {
            std.debug.assert(id < self.table.len);
            std.debug.assert(self.table[id].state != .free);

            return self.runner.max_seq - self.table[id].pos;
        }

        /// Committed KV length of the conversation.
        pub fn posOf(self: *const Self, id: u32) u32 {
            std.debug.assert(id < self.table.len);
            std.debug.assert(self.table[id].state != .free);

            return self.table[id].pos;
        }

        /// Replace the pending token of a decoding session. The scheduler closes a think block so.
        pub fn overridePending(self: *Self, id: u32, token: u32) void {
            std.debug.assert(id < self.table.len);

            const s = &self.table[id];
            std.debug.assert(s.state == .ready);
            std.debug.assert(token < self.vocab);
            s.token_pending = token;
        }

        /// Restart the sampling stream of session `id` at `seed`, so a client can replay a turn.
        /// A greedy draw reads no stream, so greedy output never moves.
        pub fn reseed(self: *Self, id: u32, seed: u64) void {
            std.debug.assert(id < self.table.len);
            std.debug.assert(self.table[id].state != .free);

            self.table[id].rng = .init(seed);
        }

        /// Replace the sampler params for later turns. The RNG stream continues.
        pub fn setParams(self: *Self, id: u32, params: sampler.Params) void {
            std.debug.assert(id < self.table.len);
            std.debug.assert(self.table[id].state != .free);
            std.debug.assert(params.temperature >= 0);
            std.debug.assert(params.top_p > 0);

            self.table[id].params = params;
            log.debug("params: engine {d} temp {d:.2} top_k {d} top_p {d:.2}", .{
                id, params.temperature, params.top_k, params.top_p,
            });
        }

        /// Append a turn's prompt tokens in same-slot chunks. With `final`, sample and return the
        /// first response token and make the session `ready`. Otherwise return null.
        /// Fails with `error.ContextFull` before any step, as the header states.
        pub fn feed(self: *Self, id: u32, tokens: []const u32, final: bool) !?u32 {
            std.debug.assert(id < self.table.len);
            std.debug.assert(tokens.len > 0);

            const s = &self.table[id];
            std.debug.assert(s.state != .free);
            if (@as(usize, s.pos) + tokens.len > self.runner.max_seq) return error.ContextFull;

            const chunk_max = self.rows.len;
            var done: usize = 0;
            var first: ?u32 = null;
            while (done < tokens.len) {
                const chunk = @min(chunk_max, tokens.len - done);
                const last_call = done + chunk == tokens.len;
                first = try self.advance(.{
                    .id = id,
                    .tokens = tokens[done..][0..chunk],
                    .final = final and last_call,
                }, &.{}, &.{});
                done += chunk;
            }

            std.debug.assert(done == tokens.len);
            if (final) std.debug.assert(first != null);
            if (!final) std.debug.assert(first == null);

            log.debug("engine feed: id {d}, +{d} tokens, pos {d}", .{ id, tokens.len, s.pos });
            return first;
        }

        /// Advance each listed session one token: commit its pending token, sample the next,
        /// and write it to `tokens_out` in `ids` order.
        pub fn decodeBatch(self: *Self, ids: []const u32, tokens_out: []u32) !void {
            std.debug.assert(ids.len >= 1);

            _ = try self.advance(null, ids, tokens_out);
        }

        /// The composed tick under `feed` and `decodeBatch`: one prefill chunk plus a decode
        /// step for `ids` in one forward, so the chunk shares the weight pass. Rows are
        /// [chunk | decode]. Returns the first response token when the chunk is final.
        pub fn advance(
            self: *Self,
            feed_chunk: ?FeedChunk,
            ids: []const u32,
            tokens_out: []u32,
        ) !?u32 {
            // Check the capacity of the whole call.
            const n_feed: usize = if (feed_chunk) |fc| fc.tokens.len else 0;
            std.debug.assert(n_feed + ids.len >= 1);
            std.debug.assert(n_feed + ids.len <= self.rows.len);
            std.debug.assert(tokens_out.len >= ids.len);
            if (feed_chunk) |fc| {
                std.debug.assert(fc.id < self.table.len);
                std.debug.assert(fc.tokens.len >= 1);
                std.debug.assert(self.table[fc.id].state != .free);

                if (@as(usize, self.table[fc.id].pos) + fc.tokens.len > self.runner.max_seq) {
                    return error.ContextFull;
                }
            }

            for (ids, 0..) |id, i| {
                std.debug.assert(id < self.table.len);
                std.debug.assert(self.table[id].state == .ready);

                if (feed_chunk) |fc| std.debug.assert(fc.id != id);

                for (ids[0..i]) |prev| {
                    std.debug.assert(prev != id);
                }

                if (self.table[id].pos >= self.runner.max_seq) return error.ContextFull;
            }

            // Assemble the rows: the chunk first, then one decode row per session.
            if (feed_chunk) |fc| {
                const s = &self.table[fc.id];
                s.state = .feeding;
                for (fc.tokens, 0..) |t, i| {
                    self.rows[i] = .{
                        .slot = fc.id,
                        .token = t,
                        .pos = s.pos + @as(u32, @intCast(i)),
                        .want_logits = fc.final and i == fc.tokens.len - 1,
                    };
                }
            }

            for (ids, 0..) |id, i| {
                self.rows[n_feed + i] = .{
                    .slot = id,
                    .token = self.table[id].token_pending,
                    .pos = self.table[id].pos,
                    .want_logits = true,
                };
            }

            // Step the runner once.
            try self.runner.stepBatch(self.rows[0 .. n_feed + ids.len]);
            self.last_final = if (feed_chunk) |fc| (if (fc.final) fc.id else null) else null;

            // Sample from the compact logits. `base` skips the final chunk row.
            const vocab = self.vocab;
            const base: usize = if (feed_chunk) |fc| (if (fc.final) @as(usize, 1) else 0) else 0;
            var first: ?u32 = null;

            // Refuse a non-finite row before any session moves, so a failed call leaves no half
            // step. The scheduler then fails the turns of the call.
            for (0..base + ids.len) |row| {
                sampler.checkLogits(self.runner.logits[row * vocab ..][0..vocab]) catch |err| {
                    log.err("the logits of a step hold a NaN, a +inf, or no finite value. " ++
                        "A model weight can hold a non-finite value", .{});
                    return err;
                };
            }

            // Sample the first token of a final chunk.
            if (feed_chunk) |fc| {
                const s = &self.table[fc.id];
                s.pos += @intCast(fc.tokens.len);
                if (fc.final) {
                    const logits = self.runner.logits[0..vocab];
                    s.token_pending = try sampler.sample(
                        logits,
                        s.params,
                        s.rng.random(),
                        self.sample_scratch,
                    );
                    s.state = .ready;
                    first = s.token_pending;
                    if (comptime core_log.hot) {
                        log_token.debug(
                            "sample: slot {d} pos {d} -> {d}",
                            .{ fc.id, s.pos, s.token_pending },
                        );
                    }
                }
            }

            // Sample the next token of each decoded session.
            for (ids, 0..) |id, i| {
                const s = &self.table[id];
                s.pos += 1;
                const logits = self.runner.logits[(base + i) * vocab ..][0..vocab];
                s.token_pending = try sampler.sample(
                    logits,
                    s.params,
                    s.rng.random(),
                    self.sample_scratch,
                );
                tokens_out[i] = s.token_pending;
                if (comptime core_log.hot) {
                    log_token.debug(
                        "sample: slot {d} pos {d} -> {d}",
                        .{ id, s.pos, s.token_pending },
                    );
                }
            }

            return first;
        }
    };
}

// -- tests ------------------------------------------------------------------

// Canned logits make the model a pure function of the row token: argmax of row i is
// (rows[i].token + 1) % vocab. Greedy params then give predictable streams.
const test_fakes = @import("test_fakes.zig");
const test_vocab = test_fakes.vocab;

const FakeRunner = struct {
    n_slots: u32,
    max_seq: u32,
    logits: []f32,
    calls: usize = 0,
    rows_log: [64]StepRow = undefined,
    rows_log_len: usize = 0,
    clones_log: [8][3]u32 = undefined,
    clones_log_len: usize = 0,

    fn initFake(gpa: Allocator, n_slots: u32, max_seq: u32) !FakeRunner {
        const logits = try gpa.alloc(f32, n_slots * test_vocab);
        @memset(logits, 0);
        return .{
            .n_slots = n_slots,
            .max_seq = max_seq,
            .logits = logits,
        };
    }

    fn deinitFake(self: *FakeRunner, gpa: Allocator) void {
        gpa.free(self.logits);
    }

    fn rowsSeen(self: *const FakeRunner) []const StepRow {
        return self.rows_log[0..self.rows_log_len];
    }

    pub fn maxStepRows(self: *const FakeRunner) u32 {
        return self.n_slots;
    }

    pub fn stepBatch(self: *FakeRunner, rows: []const StepRow) !void {
        self.calls += 1;
        for (rows) |r| {
            std.debug.assert(self.rows_log_len < self.rows_log.len);

            self.rows_log[self.rows_log_len] = r;
            self.rows_log_len += 1;
        }

        test_fakes.writeCannedLogits(self.logits, rows);
    }

    pub fn cloneSlotState(self: *FakeRunner, source_slot: u32, target_slot: u32, pos: u32) void {
        std.debug.assert(self.clones_log_len < self.clones_log.len);

        self.clones_log[self.clones_log_len] = .{ source_slot, target_slot, pos };
        self.clones_log_len += 1;
    }
};

/// A fake with the swap hooks. A u32 marker per slot stands in for KV content.
/// A blob costs one u32 per row and carries the marker in its first four bytes.
const SnapFakeRunner = struct {
    n_slots: u32,
    max_seq: u32,
    logits: []f32,
    markers: []u32,
    evicts: u32 = 0,
    restores: u32 = 0,

    fn initFake(gpa: Allocator, n_slots: u32, max_seq: u32) !SnapFakeRunner {
        const logits = try gpa.alloc(f32, n_slots * test_vocab);
        @memset(logits, 0);
        const markers = try gpa.alloc(u32, n_slots);
        @memset(markers, 0);
        return .{ .n_slots = n_slots, .max_seq = max_seq, .logits = logits, .markers = markers };
    }

    fn deinitFake(self: *SnapFakeRunner, gpa: Allocator) void {
        gpa.free(self.logits);
        gpa.free(self.markers);
    }

    pub fn maxStepRows(self: *const SnapFakeRunner) u32 {
        return self.n_slots;
    }

    pub fn stepBatch(self: *SnapFakeRunner, rows: []const StepRow) !void {
        for (rows) |r| {
            self.markers[r.slot] = r.token;
        }

        test_fakes.writeCannedLogits(self.logits, rows);
    }

    pub fn evictSlotBytesFor(self: *const SnapFakeRunner, n_rows: u32) u64 {
        _ = self;
        return @as(u64, n_rows) * @sizeOf(u32);
    }

    pub fn evictSlot(self: *SnapFakeRunner, slot: u32, n_rows: u32, out: []u8) !void {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(out.len == n_rows * @sizeOf(u32));

        std.mem.writeInt(u32, out[0..4], self.markers[slot], .little);
        self.evicts += 1;
    }

    pub fn restoreSlot(self: *SnapFakeRunner, slot: u32, n_rows: u32, blob: []const u8) !void {
        std.debug.assert(n_rows >= 1);
        std.debug.assert(blob.len == n_rows * @sizeOf(u32));

        self.markers[slot] = std.mem.readInt(u32, blob[0..4], .little);
        self.restores += 1;
    }
};

test "create/destroy lifecycle and pool exhaustion" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try FakeRunner.initFake(gpa, 2, 16);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena_state.allocator(), &fake);
    const a = try pool.create(1, .{});
    const b = try pool.create(2, .{});
    try std.testing.expect(a != b);
    try std.testing.expectError(error.NoFreeSession, pool.create(3, .{}));
    pool.destroy(a);
    const c = try pool.create(4, .{});
    try std.testing.expectEqual(a, c); // freed slot is reusable
}

test "feed chunks at n_slots rows, positions ascend, logits only on the last row" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try FakeRunner.initFake(gpa, 2, 16);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena_state.allocator(), &fake);
    const id = try pool.create(7, .{});
    const first = try pool.feed(id, &.{ 4, 5, 6, 7, 0 }, true);

    // 5 tokens at chunk size 2: calls of 2, 2, 1 rows.
    try std.testing.expectEqual(@as(usize, 3), fake.calls);
    try std.testing.expectEqual(@as(usize, 5), fake.rowsSeen().len);
    for (fake.rowsSeen(), 0..) |r, i| {
        try std.testing.expectEqual(id, r.slot);
        try std.testing.expectEqual(@as(u32, @intCast(i)), r.pos);
        try std.testing.expectEqual(i == 4, r.want_logits);
    }

    // Last fed token 0 -> fake argmax 1: the sampled first response token.
    try std.testing.expectEqual(@as(u32, 1), first.?);
    try std.testing.expectEqual(@as(u32, 5), pool.table[id].pos);
    try std.testing.expectEqual(State.ready, pool.table[id].state);
}

test "decodeBatch steps pending tokens and walks the fake's token chain" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try FakeRunner.initFake(gpa, 3, 16);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena_state.allocator(), &fake);
    const a = try pool.create(1, .{});
    const b = try pool.create(2, .{});
    const ta = (try pool.feed(a, &.{2}, true)).?; // argmax(2+1) = 3
    const tb = (try pool.feed(b, &.{6}, true)).?; // argmax(6+1) = 7
    try std.testing.expectEqual(@as(u32, 3), ta);
    try std.testing.expectEqual(@as(u32, 7), tb);

    var out: [2]u32 = undefined;
    try pool.decodeBatch(&.{ a, b }, &out);
    // Pending 3 -> 4. Pending 7 -> 0 (mod 8). Each session's chain advances
    // independently inside one batched call.
    try std.testing.expectEqual(@as(u32, 4), out[0]);
    try std.testing.expectEqual(@as(u32, 0), out[1]);
    try std.testing.expectEqual(@as(u32, 2), pool.table[a].pos);
    try std.testing.expectEqual(@as(u32, 2), pool.table[b].pos);

    // The stepped rows carried the pending tokens at the pre-step positions.
    const last_rows = fake.rowsSeen()[fake.rowsSeen().len - 2 ..];
    try std.testing.expectEqual(@as(u32, 3), last_rows[0].token);
    try std.testing.expectEqual(@as(u32, 1), last_rows[0].pos);
    try std.testing.expectEqual(@as(u32, 7), last_rows[1].token);
    try std.testing.expectEqual(@as(u32, 1), last_rows[1].pos);
}

test "ContextFull is atomic: no step happens, state is unchanged" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try FakeRunner.initFake(gpa, 2, 4);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena_state.allocator(), &fake);
    const id = try pool.create(1, .{});
    try std.testing.expectError(error.ContextFull, pool.feed(id, &.{ 1, 2, 3, 4, 5 }, true));
    try std.testing.expectEqual(@as(usize, 0), fake.calls);
    try std.testing.expectEqual(@as(u32, 0), pool.table[id].pos);

    // Fill to the cap, then a decode tick must refuse before stepping.
    _ = try pool.feed(id, &.{ 1, 2, 3, 4 }, true);
    const calls_before = fake.calls;
    var out: [1]u32 = undefined;
    try std.testing.expectError(error.ContextFull, pool.decodeBatch(&.{id}, &out));
    try std.testing.expectEqual(calls_before, fake.calls);
    try std.testing.expectEqual(@as(u32, 4), pool.table[id].pos);
}

test "clone forks a session: state copied, continuation identical, source untouched" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try FakeRunner.initFake(gpa, 3, 16);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena_state.allocator(), &fake);
    try std.testing.expect(Sessions(FakeRunner).can_clone);
    const a = try pool.create(1, .{});
    _ = (try pool.feed(a, &.{ 2, 4 }, true)).?; // pos 2, pending 5

    const b = try pool.clone(a, 9, .{});
    try std.testing.expect(b != a);
    try std.testing.expectEqual(pool.table[a].pos, pool.table[b].pos);
    try std.testing.expectEqual(pool.table[a].token_pending, pool.table[b].token_pending);
    try std.testing.expectEqual(State.ready, pool.table[b].state);
    // The runner hook saw the fork: (source, target, pos).
    try std.testing.expectEqual([3]u32{ a, b, 2 }, fake.clones_log[0]);

    // Same pending token + greedy: both continue with the same next token,
    // each against its own slot.
    var out: [2]u32 = undefined;
    try pool.decodeBatch(&.{ a, b }, &out);
    try std.testing.expectEqual(@as(u32, 6), out[0]); // pending 5 -> 6
    try std.testing.expectEqual(out[0], out[1]);
    try std.testing.expectEqual(@as(u32, 3), pool.table[a].pos);
    try std.testing.expectEqual(@as(u32, 3), pool.table[b].pos);

    // Pool exhaustion refuses the fork. The source is unaffected.
    _ = try pool.create(3, .{});
    try std.testing.expectError(error.NoFreeSession, pool.clone(a, 1, .{}));
    try std.testing.expectEqual(State.ready, pool.table[a].state);
}

test "advance: a prefill chunk and decode rows share one forward call" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try FakeRunner.initFake(gpa, 4, 16);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena_state.allocator(), &fake);
    const a = try pool.create(1, .{});
    const b = try pool.create(2, .{});
    const c = try pool.create(3, .{});
    _ = (try pool.feed(a, &.{2}, true)).?; // pending 3, pos 1
    _ = (try pool.feed(b, &.{4}, true)).?; // pending 5, pos 1

    // One call: c's two-token chunk (non-final) shares a's and b's decode step.
    const calls_before = fake.calls;
    var out: [2]u32 = undefined;
    const none = try pool.advance(
        .{ .id = c, .tokens = &.{ 1, 2 }, .final = false },
        &.{ a, b },
        &out,
    );
    try std.testing.expectEqual(@as(usize, 1), fake.calls - calls_before);
    try std.testing.expectEqual(@as(?u32, null), none);
    try std.testing.expectEqual(@as(u32, 4), out[0]); // a: pending 3 -> 4
    try std.testing.expectEqual(@as(u32, 6), out[1]); // b: pending 5 -> 6

    // Row layout of that call: [c's chunk | a | b], logits only on the suffix.
    const rows = fake.rowsSeen()[fake.rowsSeen().len - 4 ..];
    try std.testing.expectEqual(c, rows[0].slot);
    try std.testing.expectEqual(@as(u32, 0), rows[0].pos);
    try std.testing.expectEqual(false, rows[0].want_logits);
    try std.testing.expectEqual(c, rows[1].slot);
    try std.testing.expectEqual(@as(u32, 1), rows[1].pos);
    try std.testing.expectEqual(false, rows[1].want_logits);
    try std.testing.expectEqual(a, rows[2].slot);
    try std.testing.expectEqual(true, rows[2].want_logits);
    try std.testing.expectEqual(b, rows[3].slot);
    try std.testing.expectEqual(true, rows[3].want_logits);

    // The final chunk samples c's first token from the right logits row.
    const first = try pool.advance(.{ .id = c, .tokens = &.{6}, .final = true }, &.{ a, b }, &out);
    try std.testing.expectEqual(@as(u32, 7), first.?); // argmax(6+1)
    try std.testing.expectEqual(State.ready, pool.table[c].state);
    try std.testing.expectEqual(@as(u32, 3), pool.table[c].pos);
    try std.testing.expectEqual(@as(u32, 3), pool.table[a].pos);

    // ContextFull on the chunk is atomic for the WHOLE call: neither the
    // chunk's session nor the decode rows move. Drive c to pos 14 of 16
    // first, so a 3-token chunk no longer fits.
    var one: [1]u32 = undefined;
    for (0..11) |_| {
        try pool.decodeBatch(&.{c}, &one);
    }

    try std.testing.expectEqual(@as(u32, 14), pool.table[c].pos);
    const calls_full = fake.calls;
    const a_pos = pool.table[a].pos;
    try std.testing.expectError(
        error.ContextFull,
        pool.advance(.{ .id = c, .tokens = &.{ 6, 6, 6 }, .final = false }, &.{a}, &out),
    );
    try std.testing.expectEqual(calls_full, fake.calls);
    try std.testing.expectEqual(@as(u32, 14), pool.table[c].pos);
    try std.testing.expectEqual(a_pos, pool.table[a].pos);
}

test "pending-token turn boundary: budget stop prepends, EOS stop drops" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try FakeRunner.initFake(gpa, 2, 32);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(FakeRunner).init(arena_state.allocator(), &fake);
    const id = try pool.create(1, .{});
    const t0 = (try pool.feed(id, &.{0}, true)).?; // pending 1, pos 1
    try std.testing.expectEqual(@as(u32, 1), t0);

    // Budget stop after the first sampled token: the caller saw `1` but its
    // KV was never written. Next turn prepends it before the new user tokens.
    const pos_before = pool.table[id].pos;
    const next = (try pool.feed(id, &.{ t0, 5 }, true)).?;
    try std.testing.expectEqual(pos_before + 2, pool.table[id].pos);
    try std.testing.expectEqual(@as(u32, 6), next); // argmax(5+1)

    // EOS-style stop: just feed the new turn. The pending token is forgotten.
    const next2 = (try pool.feed(id, &.{2}, true)).?;
    try std.testing.expectEqual(@as(u32, 3), next2);
}

test "can_snapshot follows the runner's swap hooks" {
    try std.testing.expect(!Sessions(FakeRunner).can_snapshot);
    try std.testing.expect(Sessions(SnapFakeRunner).can_snapshot);
}

test "restorePrefix seeds a fresh session, and its suffix walks the cold chain" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try SnapFakeRunner.initFake(gpa, 2, 16);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(SnapFakeRunner).init(arena_state.allocator(), &fake);

    // The reference: one cold session feeds the whole prompt.
    const cold = try pool.create(1, .{});
    const cold_first = (try pool.feed(cold, &.{ 1, 2, 3 }, true)).?;
    try std.testing.expectEqual(@as(u32, 4), cold_first); // argmax(3+1)

    // The warm path: a two-row prefix blob holding the marker tokens 1 and 2
    // leave behind, then the one-token suffix.
    var blob = [_]u8{0} ** (2 * @sizeOf(u32));
    std.mem.writeInt(u32, blob[0..4], 2, .little);
    try std.testing.expectEqual(@as(u64, blob.len), pool.snapshotBytesFor(2));

    const warm = try pool.create(1, .{});
    try pool.restorePrefix(warm, 2, &blob);
    try std.testing.expectEqual(@as(u32, 2), pool.table[warm].pos);
    try std.testing.expectEqual(State.feeding, pool.table[warm].state);
    try std.testing.expectEqual(@as(u32, 0), pool.table[warm].token_pending);
    try std.testing.expectEqual(@as(u32, 2), fake.markers[warm]);
    try std.testing.expectEqual(@as(u32, 1), fake.restores);

    // The suffix feeds at pos 2, so both sessions end on the same token and the
    // same length.
    const warm_first = (try pool.feed(warm, &.{3}, true)).?;
    try std.testing.expectEqual(cold_first, warm_first);
    try std.testing.expectEqual(pool.table[cold].pos, pool.table[warm].pos);
    try std.testing.expectEqual(State.ready, pool.table[warm].state);
}

test "restoreReady seeds a ready session that continues the cold chain with zero steps" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try SnapFakeRunner.initFake(gpa, 2, 16);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(SnapFakeRunner).init(arena_state.allocator(), &fake);

    // Cold: feed [1,2,3] final. The pending token is argmax(3+1) = 4, and
    // the completing feed leaves the final row's logits at compact row zero.
    const cold = try pool.create(1, .{});
    const cold_first = (try pool.feed(cold, &.{ 1, 2, 3 }, true)).?;
    try std.testing.expectEqual(@as(u32, 4), cold_first);
    var logits = [_]f32{0} ** test_vocab;
    pool.copyFinalLogits(cold, &logits);
    try std.testing.expectEqual(@as(f32, 1), logits[4]);

    // Warm: a 3-row blob carrying the marker token 3 leaves behind, plus the
    // stored logits. No stepBatch runs, because the pending token comes from the
    // store, sampled with the session's own RNG inside the engine.
    var blob = [_]u8{0} ** (3 * @sizeOf(u32));
    std.mem.writeInt(u32, blob[0..4], 3, .little);
    const warm = try pool.create(1, .{});
    const first = try pool.restoreReady(warm, 3, &blob, &logits);
    try std.testing.expectEqual(cold_first, first);
    try std.testing.expectEqual(State.ready, pool.table[warm].state);
    try std.testing.expectEqual(@as(u32, 3), pool.table[warm].pos);

    // Both sessions continue on the same chain from the same state.
    var out: [2]u32 = undefined;
    try pool.decodeBatch(&.{ cold, warm }, &out);
    try std.testing.expectEqual(@as(u32, 5), out[0]);
    try std.testing.expectEqual(out[0], out[1]);
}

test "snapshotPrefix copies rows out and leaves the session running" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    var fake = try SnapFakeRunner.initFake(gpa, 2, 16);
    defer fake.deinitFake(gpa);

    var pool = try Sessions(SnapFakeRunner).init(arena_state.allocator(), &fake);
    const id = try pool.create(1, .{});
    _ = (try pool.feed(id, &.{ 1, 2 }, true)).?; // pos 2, marker 2, pending 3
    const before = pool.table[id];

    var out = [_]u8{0} ** (2 * @sizeOf(u32));
    try pool.snapshotPrefix(id, 2, &out);
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, out[0..4], .little));
    try std.testing.expectEqual(@as(u32, 1), fake.evicts);

    // A snapshot is a read: position, state, and the pending token all stand.
    try std.testing.expectEqual(before.pos, pool.table[id].pos);
    try std.testing.expectEqual(before.state, pool.table[id].state);
    try std.testing.expectEqual(before.token_pending, pool.table[id].token_pending);

    // And the session decodes on from where it was.
    var next: [1]u32 = undefined;
    try pool.decodeBatch(&.{id}, &next);
    try std.testing.expectEqual(@as(u32, 4), next[0]); // pending 3 -> 4
}
