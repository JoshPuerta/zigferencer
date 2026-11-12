//! Prefix-keyed, byte-budgeted LRU of model-state snapshots, in two tiers (RAM
//! and disk). Each entry holds the serialized forward state after a token
//! prefix (the opaque snapshot `blob` the runner produces), the prefix tokens
//! (collision guard), the last-token logits (so an exact hit samples the first
//! output with zero forwards), and the measured prefill cost. A lookup that
//! matches both the key and the stored tokens lets the caller restore instead
//! of re-prefilling.
//!
//! Prefill is redundant recomputation across re-runs and shared prefixes. The
//! measured 9B break-even is restore ~4-9 ms against prefill 0.27-11 s. That
//! ratio is architecture-dependent, so the admission limit is not hardcoded.
//! `worthwhile` compares the measured prefill saved against the measured
//! restore cost, and the 2x margin comes from that spread.
//!
//! Two tiers, checked in order. A hit restores forward state instead of
//! re-running prefill:
//!
//!   lookup ->  [ RAM tier ]  --miss-->  [ disk tier ]  --miss-->  re-prefill
//!              LRU, byte-budgeted        <dir>/<key>.zkv
//!              linear scan               (mmap + verify, then
//!                  ^                       PROMOTE into RAM)
//!                  |                              |
//!                  +------------ promote ---------+
//!
//!   admit gates each tier by `worthwhile` (disk uses a higher bar that adds
//!   the read-back time of the snapshot). A future hit on a kept entry must
//!   cost less than the prefill it replaces.
//!
//! Disk file. `<dir>/<hexkey>.zkv` holds a fixed `DiskHeader`, then the tokens, the last logits,
//! and the blob. `payload_fnv` guards a torn file, and `model_fp` plus the stored tokens guard
//! hash collisions across models. A bad file is deleted and reads as a miss.
//!
//! Disk index. `lookupBestPrefix` needs the disk tier scannable, because its callers track no
//! boundary lists. A row holds the key and the tokens of one file, never the blob. A row goes
//! stale when its file is evicted or torn, and the verified load at promotion drops it, so the
//! index heals itself. `loadDiskIndex` is bounded by `limits.serve_prefix_disk_index_max` and
//! reports the files past the bound, because a capped index must never read as a complete one.
//!
//! Namespaces. `model_fp` names the namespace of every file a cache writes, and `disk_budget`
//! bounds that namespace alone. Many models and processes can share one directory. A scan skips a
//! foreign file silently, spends none of the bound on it, and never counts or deletes it. A file
//! that fails to open, to read, or to stat skips both roles, so one transient IO error never
//! makes a namespace look small enough to keep growing.
//!
//! Admission. `admit` copies the transient buffers of the caller. Each tier is optional, so a
//! failure in one logs a line and the other still runs. The disk bar adds the read-back time at
//! `disk_bw_bps`, which `persistToDisk` refines from the observed write throughput, so the gate
//! adapts to the hardware. The GDN floor of about 50 MB makes the disk gate decline short prompts.
//! The RAM tier can exceed its budget by one entry, when a single snapshot is larger than it.
//!
//! Prefix hits. `lookupLongestPrefix` and `lookupBestPrefix` return a pointer that stays valid
//! only until the next mutating call, because an insert reallocates the RAM list. `hits` counts
//! longest-prefix queries that an entry served, so it reads as restore demand. Exact lookups stay
//! out of it. Every scan is linear over at most budget/snapshot_size entries, which is tens of
//! rows.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const mmap = @import("../io/mmap.zig");
const limits = @import("../core/limits.zig");
const prop = @import("../core/prop.zig");

const log = std.log.scoped(.cache);

/// The test runner prints a warn line, and a mismatch test expects these lines.
const mismatch_level: std.log.Level = if (@import("builtin").is_test) .debug else .warn;

// -- constants --------------------------------------------------------------

/// Environment variable that relocates the cache directory. A relative path resolves from the
/// cwd. Unset or empty means `.prompt_cache` under the cwd.
pub const cache_dir_env = "ZIGFERENCER_CACHE_DIR";

const default_cache_dir = ".prompt_cache";

/// Magic and version of the snapshot file.
const magic: u32 = 0x315A4B5A; // "ZKV1"

const version: u32 = 1;

// -- the linear-scan model --------------------------------------------------
//
// The test model holds the RAM tier as a plain list. Each sequence is distinct.
const model_cap = 32;
const model_tokens_max = 6;

// -- types ------------------------------------------------------------------

/// Why a snapshot image does not parse.
pub const ImageError = error{ ImageTruncated, ImageForeign, ImageSizeOverflow, ImageTorn };

/// The views of one snapshot image into its bytes.
const Image = struct {
    n_tokens: u32,
    tokens: []const u8,
    logits: []const u8,
    blob: []const u8,
};

const DiskHeader = extern struct {
    magic: u32,
    version: u32,
    model_fp: u64,
    n_tokens: u32,
    n_logits: u32,
    blob_len: u64,
    payload_fnv: u64,
};

const Entry = struct {
    key: u64,
    tokens: []u32, // owned, and verifies the key match (collision guard)
    n: u32, // prefix length (== tokens.len), the restore position
    blob: []u8, // owned snapshot bytes (opaque to the cache)
    last_logits: []f32, // owned, logits of token n-1 (sample first output, 0 forwards)
    last_used: u64, // LRU tick
    hits: u64 = 0, // longest-prefix queries served
};

/// One row of the disk index: the key and the prefix tokens of a file, never the blob.
const DiskIndexEntry = struct {
    key: u64,
    tokens: []u32, // owned
};

pub const PromptCache = struct {
    gpa: Allocator,
    budget_bytes: usize, // RAM tier byte budget
    used_bytes: usize,
    tick: u64,
    entries: std.ArrayList(Entry),
    // Disk tier. A null `dir` means RAM-only.
    io: ?Io = null,
    dir: ?[]const u8 = null,
    disk_budget: usize = 0,
    disk_bw_bps: f64 = 200.0e6,
    model_fp: u64 = 0,
    last_hit_from_disk: bool = false,
    last_admit_ram: bool = false, // which tiers the most recent admit() reached
    last_admit_disk: bool = false,
    /// The disk index. Empty until `loadDiskIndex`, then kept in step by persist and drop.
    disk_index: std.ArrayList(DiskIndexEntry) = .empty,

    // -- init ---------------------------------------------------------------

    pub fn init(gpa: Allocator, budget_bytes: usize) PromptCache {
        return .{
            .gpa = gpa,
            .budget_bytes = budget_bytes,
            .used_bytes = 0,
            .tick = 0,
            .entries = .empty,
        };
    }

    /// Enable the disk tier: create `dir` (cwd-relative) and record the budget and fingerprint.
    /// `dir` must outlive the cache.
    pub fn enableDisk(
        self: *PromptCache,
        io: Io,
        dir: []const u8,
        disk_budget: usize,
        model_fp: u64,
    ) !void {
        try Io.Dir.cwd().createDirPath(io, dir);
        self.io = io;
        self.dir = dir;
        self.disk_budget = disk_budget;
        self.model_fp = model_fp;
    }

    pub fn deinit(self: *PromptCache) void {
        for (self.entries.items) |*e| {
            self.freeEntry(e);
        }

        self.entries.deinit(self.gpa);
        for (self.disk_index.items) |*ie| {
            self.gpa.free(ie.tokens);
        }

        self.disk_index.deinit(self.gpa);
    }

    fn entryBytes(e: *const Entry) usize {
        return e.tokens.len * @sizeOf(u32) + e.blob.len + e.last_logits.len * @sizeOf(f32);
    }

    fn freeEntry(self: *PromptCache, e: *Entry) void {
        self.gpa.free(e.tokens);
        self.gpa.free(e.blob);
        self.gpa.free(e.last_logits);
    }

    // -- lookup -------------------------------------------------------------

    /// Find the entry for `key` whose stored tokens equal `tokens`, and bump its LRU tick.
    /// A RAM miss tries the disk tier and promotes a hit. `last_hit_from_disk` records the tier.
    /// Null means a miss.
    pub fn lookup(self: *PromptCache, key: u64, tokens: []const u32) ?*const Entry {
        self.last_hit_from_disk = false;
        for (self.entries.items) |*e| {
            if (e.key == key and std.mem.eql(u32, e.tokens, tokens)) {
                self.tick += 1;
                e.last_used = self.tick;
                log.debug("ram hit: key {x:0>16}, {d} tokens restored", .{ key, e.n });
                return e;
            }
        }

        if (self.dir != null) {
            if (self.loadFromDisk(key, tokens) catch null) |loaded| {
                self.insertEntry(loaded) catch return null;
                self.last_hit_from_disk = true;
                log.debug("disk hit: key {x:0>16}, {d} tokens restored", .{ key, loaded.n });
                return &self.entries.items[self.entries.items.len - 1];
            }
        }

        log.debug("miss: key {x:0>16}, {d} tokens", .{ key, tokens.len });
        return null;
    }

    /// Return the entry of the longest boundary that is a prefix of `tokens`. The boundaries are
    /// the prefix lengths saved at turn ends. Two passes keep the pointer valid, because a
    /// lookup can promote a disk entry and reallocate the RAM list.
    pub fn lookupLongestPrefix(
        self: *PromptCache,
        model_fp: u64,
        tokens: []const u32,
        boundaries: []const u32,
    ) ?*const Entry {
        var best_n: u32 = 0;
        for (boundaries) |b| {
            if (b == 0 or b > tokens.len or b <= best_n) continue;
            if (self.lookup(prefixKey(model_fp, tokens[0..b]), tokens[0..b]) != null) {
                best_n = b;
            }
        }

        if (best_n == 0) return null;

        return self.lookup(prefixKey(model_fp, tokens[0..best_n]), tokens[0..best_n]);
    }

    /// Return the entry with the largest `n` whose tokens equal `tokens[0..n]`, and bump its
    /// LRU tick. The scan covers the RAM tier and the disk index, and promotes a longer disk
    /// prefix. Null means no candidate.
    pub fn lookupBestPrefix(self: *PromptCache, tokens: []const u32) ?*const Entry {
        self.last_hit_from_disk = false;

        // Find the longest RAM prefix, as the bar for the disk tier.
        var floor_n: u32 = 0;
        for (self.entries.items) |*e| {
            if (e.n == 0 or e.n > tokens.len) continue;
            if (!std.mem.eql(u32, e.tokens, tokens[0..e.n])) continue;
            if (e.n > floor_n) {
                floor_n = e.n;
            }
        }

        // Promote a longer disk prefix when one loads.
        if (self.promoteBestFromDisk(tokens, floor_n)) |hit| {
            hit.hits += 1;
            self.last_hit_from_disk = true;
            log.debug("disk hit: key {x:0>16}, {d} tokens promoted", .{ hit.key, hit.n });
            return hit;
        }

        // Rescan RAM for a stable pointer, because a failed promotion can evict entries.
        var best: ?*Entry = null;
        for (self.entries.items) |*e| {
            if (e.n == 0 or e.n > tokens.len) continue;
            if (!std.mem.eql(u32, e.tokens, tokens[0..e.n])) continue;
            if (best == null or e.n > best.?.n) {
                best = e;
            }
        }

        const hit = best orelse {
            log.debug("miss: best prefix of {d} tokens", .{tokens.len});
            return null;
        };
        self.tick += 1;
        hit.last_used = self.tick;
        hit.hits += 1;
        log.debug("ram hit: key {x:0>16}, {d} tokens restored", .{ hit.key, hit.n });
        return hit;
    }

    /// Log one debug line per RAM entry: prefix length and hits. Eviction erases the count.
    pub fn logEntryHits(self: *const PromptCache) void {
        for (self.entries.items) |*e| {
            log.debug("prefix entry: {d} tokens, {d} hits", .{ e.n, e.hits });
        }
    }

    /// Try the index rows longer than `floor_n`, longest first, and promote the first file that
    /// loads. A failed row drops, so the loop ends within the index length.
    fn promoteBestFromDisk(self: *PromptCache, tokens: []const u32, floor_n: u32) ?*Entry {
        while (true) {
            var best_i: ?usize = null;
            var best_n: usize = floor_n;
            for (self.disk_index.items, 0..) |*ie, i| {
                if (ie.tokens.len <= best_n or ie.tokens.len > tokens.len) continue;
                if (!std.mem.eql(u32, ie.tokens, tokens[0..ie.tokens.len])) continue;

                best_i = i;
                best_n = ie.tokens.len;
            }

            const i = best_i orelse return null;
            const key = self.disk_index.items[i].key;
            const loaded = (self.loadFromDisk(
                key,
                self.disk_index.items[i].tokens,
            ) catch null) orelse {
                self.removeIndexRow(key);
                continue;
            };
            self.insertEntry(loaded) catch return null;
            return &self.entries.items[self.entries.items.len - 1];
        }
    }

    // -- the disk index -----------------------------------------------------

    /// Build the disk index: one row per `.zkv` file of this namespace. Best-effort, because a
    /// missing row is a cache miss.
    pub fn loadDiskIndex(self: *PromptCache) void {
        const io = self.io orelse return;
        var dir = Io.Dir.cwd().openDir(io, self.dir.?, .{ .iterate = true }) catch return;
        defer dir.close(io);

        var skipped: usize = 0;
        var it = dir.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zkv")) continue;

            // Check the fingerprint before the cap, so a foreign file spends no capacity.
            const tokens = (self.readOwnTokens(io, entry.name) catch continue) orelse continue;
            if (self.disk_index.items.len >= limits.serve_prefix_disk_index_max) {
                self.gpa.free(tokens);
                skipped += 1;
                continue;
            }

            self.addIndexRow(prefixKey(self.model_fp, tokens), tokens) catch self.gpa.free(tokens);
        }

        if (skipped > 0) {
            log.warn("disk index full at {d} rows: {d} files not indexed in {s}", .{
                limits.serve_prefix_disk_index_max, skipped, self.dir.?,
            });
        }
    }

    /// The count of indexed disk entries, for the startup witness.
    pub fn diskIndexLen(self: *const PromptCache) usize {
        return self.disk_index.items.len;
    }

    /// Return the owned prefix tokens of one file, or null for a foreign, bad, or short file.
    /// The caller derives the key from the tokens, so a renamed file cannot alias a prefix.
    /// The promotion load checks the payload FNV before any byte reaches a KV slot.
    fn readOwnTokens(self: *PromptCache, io: Io, name: []const u8) !?[]u32 {
        var pathbuf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&pathbuf, "{s}/{s}", .{ self.dir.?, name });
        var mapped = try mmap.openReadOnly(io, path);
        defer mapped.close();

        const bytes = mapped.bytes();
        if (bytes.len < @sizeOf(DiskHeader)) return null;

        var hdr: DiskHeader = undefined;
        @memcpy(std.mem.asBytes(&hdr), bytes[0..@sizeOf(DiskHeader)]);
        if (hdr.magic != magic or hdr.version != version or hdr.model_fp != self.model_fp) {
            return null;
        }

        const tok_bytes: usize = @as(usize, hdr.n_tokens) * @sizeOf(u32);
        if (hdr.n_tokens == 0 or bytes.len < @sizeOf(DiskHeader) + tok_bytes) return null;

        const tokens = try self.gpa.alloc(u32, hdr.n_tokens);
        @memcpy(std.mem.sliceAsBytes(tokens), bytes[@sizeOf(DiskHeader)..][0..tok_bytes]);
        return tokens;
    }

    /// Take ownership of `tokens` and record the row. A row for the same key is replaced.
    fn addIndexRow(self: *PromptCache, key: u64, tokens: []u32) !void {
        if (self.indexRowOf(key)) |i| {
            self.gpa.free(self.disk_index.items[i].tokens);
            self.disk_index.items[i].tokens = tokens;
            return;
        }

        try self.disk_index.append(self.gpa, .{ .key = key, .tokens = tokens });
    }

    fn indexRowOf(self: *const PromptCache, key: u64) ?usize {
        for (self.disk_index.items, 0..) |*ie, i| {
            if (ie.key == key) return i;
        }

        return null;
    }

    fn removeIndexRow(self: *PromptCache, key: u64) void {
        const i = self.indexRowOf(key) orelse return;
        self.gpa.free(self.disk_index.items[i].tokens);
        _ = self.disk_index.swapRemove(i);
    }

    // -- admission ----------------------------------------------------------

    /// Admit a snapshot for `tokens` to each tier that `worthwhile` accepts.
    /// Returns true when the snapshot reached at least one tier.
    pub fn admit(
        self: *PromptCache,
        key: u64,
        tokens: []const u32,
        blob: []const u8,
        last_logits: []const f32,
        prefill_ns: u64,
        restore_ns: u64,
    ) bool {
        self.last_admit_ram = false;
        self.last_admit_disk = false;

        // RAM bar: the measured restore cost.
        if (worthwhile(prefill_ns, restore_ns)) {
            self.last_admit_ram = self.admitRam(key, tokens, blob, last_logits);
        }

        if (self.dir != null) {
            // Disk bar: restore plus the read-back of this blob.
            const disk_read_ns: u64 = @intFromFloat(
                @as(f64, @floatFromInt(blob.len)) / self.disk_bw_bps * 1e9,
            );
            if (worthwhile(prefill_ns, restore_ns + disk_read_ns)) {
                // The index row lands only with the file, so an evicted RAM entry stays findable.
                if (self.persistToDisk(key, tokens, blob, last_logits)) |_| {
                    self.last_admit_disk = true;
                    if (self.gpa.dupe(u32, tokens)) |dup| {
                        self.addIndexRow(key, dup) catch self.gpa.free(dup);
                    } else |_| {}
                } else |_| {}
            }
        }

        return self.last_admit_ram or self.last_admit_disk;
    }

    /// Copy a snapshot into the RAM tier. It returns true when the tier holds the entry.
    fn admitRam(
        self: *PromptCache,
        key: u64,
        tokens: []const u32,
        blob: []const u8,
        last_logits: []const f32,
    ) bool {
        const entry = self.copyEntry(key, tokens, blob, last_logits) catch |err| {
            log.debug("ram admit: {x:0>16}, copy failed: {t}", .{ key, err });
            return false;
        };
        self.insertEntry(entry) catch |err| {
            log.debug("ram admit: {x:0>16}, insert failed: {t}", .{ key, err });
            return false;
        };
        return true;
    }

    fn copyEntry(
        self: *PromptCache,
        key: u64,
        tokens: []const u32,
        blob: []const u8,
        last_logits: []const f32,
    ) Allocator.Error!Entry {
        const e_tokens = try self.gpa.dupe(u32, tokens);
        errdefer self.gpa.free(e_tokens);

        const e_blob = try self.gpa.dupe(u8, blob);
        errdefer self.gpa.free(e_blob);

        const e_logits = try self.gpa.dupe(f32, last_logits);
        return .{
            .key = key,
            .tokens = e_tokens,
            .n = @intCast(tokens.len),
            .blob = e_blob,
            .last_logits = e_logits,
            .last_used = 0, // set by insertEntry
        };
    }

    /// Insert an owned entry into the RAM tier, and evict LRU entries to stay under budget.
    fn insertEntry(self: *PromptCache, entry: Entry) !void {
        var e = entry;
        const need = entryBytes(&e);
        while (self.used_bytes + need > self.budget_bytes) {
            if (!self.evictLru()) break;
        }

        self.tick += 1;
        e.last_used = self.tick;
        self.entries.append(self.gpa, e) catch |err| {
            self.freeEntry(&e);
            return err;
        };
        self.used_bytes += need;
    }

    /// Evict the least-recently-used entry. Returns false when the cache is empty.
    fn evictLru(self: *PromptCache) bool {
        if (self.entries.items.len == 0) return false;

        var lru_i: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used < self.entries.items[lru_i].last_used) {
                lru_i = i;
            }
        }

        var victim = self.entries.items[lru_i];
        log.debug("evict: {d}-token entry after {d} hits", .{ victim.n, victim.hits });
        self.used_bytes -= entryBytes(&victim);
        self.freeEntry(&victim);
        _ = self.entries.swapRemove(lru_i);
        return true;
    }

    // -- disk tier ----------------------------------------------------------

    /// `<dir>/<hexkey>.zkv` into `buf`.
    fn diskPath(self: *const PromptCache, key: u64, buf: []u8) ![]u8 {
        return std.fmt.bufPrint(buf, "{s}/{x:0>16}.zkv", .{ self.dir.?, key });
    }

    /// Load and verify the snapshot for `key` and `tokens`. Returns an owned entry, or null
    /// for an absent, corrupt, or mismatched file. A bad file is deleted.
    fn loadFromDisk(self: *PromptCache, key: u64, tokens: []const u32) !?Entry {
        // Map the file.
        const io = self.io orelse return null;
        var pathbuf: [512]u8 = undefined;
        const path = self.diskPath(key, &pathbuf) catch return null;
        var mapped = mmap.openReadOnly(io, path) catch return null; // not found => miss
        defer mapped.close();

        // Validate the image, then the collision guard.
        const bytes = mapped.bytes();
        const image = parseImage(bytes, self.model_fp) catch |err| {
            log.warn("disk cache: {x:0>16}.zkv is not a valid snapshot ({t}), so it is a miss", .{
                key, err,
            });
            return self.dropDisk(key);
        };
        if (!std.mem.eql(u8, image.tokens, std.mem.sliceAsBytes(tokens))) {
            return null; // collision: different prompt
        }

        // Copy the payload into an owned entry.
        const e_tokens = try self.gpa.dupe(u32, tokens);
        errdefer self.gpa.free(e_tokens);

        const e_logits = try self.gpa.alloc(f32, image.logits.len / @sizeOf(f32));
        errdefer self.gpa.free(e_logits);

        @memcpy(std.mem.sliceAsBytes(e_logits), image.logits);
        const e_blob = try self.gpa.dupe(u8, image.blob);
        log.debug("disk restore: {x:0>16}.zkv, {d} bytes read", .{ key, bytes.len });
        return Entry{
            .key = key,
            .tokens = e_tokens,
            .n = image.n_tokens,
            .blob = e_blob,
            .last_logits = e_logits,
            .last_used = 0,
        };
    }

    /// Delete a bad or stale disk file, and report a miss.
    fn dropDisk(self: *PromptCache, key: u64) ?Entry {
        const io = self.io orelse return null;
        self.removeIndexRow(key);
        var pathbuf: [512]u8 = undefined;
        const path = self.diskPath(key, &pathbuf) catch return null;
        Io.Dir.cwd().deleteFile(io, path) catch {};
        return null;
    }

    /// Write a snapshot file, after it evicts the oldest own files to fit the budget.
    /// Refines `disk_bw_bps` from the write throughput, which stands in for the read speed.
    fn persistToDisk(
        self: *PromptCache,
        key: u64,
        tokens: []const u32,
        blob: []const u8,
        last_logits: []const f32,
    ) !void {
        // Make room, and build the header.
        const io = self.io orelse return;
        const tok_bytes = std.mem.sliceAsBytes(tokens);
        const log_bytes = std.mem.sliceAsBytes(last_logits);
        const total = @sizeOf(DiskHeader) + tok_bytes.len + log_bytes.len + blob.len;
        self.diskEvictToFit(total);

        var h = std.hash.Fnv1a_64.init();
        h.update(tok_bytes);
        h.update(log_bytes);
        h.update(blob);
        const hdr: DiskHeader = .{
            .magic = magic,
            .version = version,
            .model_fp = self.model_fp,
            .n_tokens = @intCast(tokens.len),
            .n_logits = @intCast(last_logits.len),
            .blob_len = blob.len,
            .payload_fnv = h.final(),
        };

        // Write the file, and time it.
        var pathbuf: [512]u8 = undefined;
        const path = try self.diskPath(key, &pathbuf);
        const wbuf = try self.gpa.alloc(u8, 1 << 20);
        defer self.gpa.free(wbuf);

        const t0 = Io.Clock.Timestamp.now(io, .awake);
        const file = try Io.Dir.cwd().createFile(io, path, .{});
        var fw: Io.File.Writer = .init(file, io, wbuf);
        try fw.interface.writeAll(std.mem.asBytes(&hdr));
        try fw.interface.writeAll(tok_bytes);
        try fw.interface.writeAll(log_bytes);
        try fw.interface.writeAll(blob);
        try fw.interface.flush();
        file.close(io);
        log.debug("disk persist: {x:0>16}.zkv, {d} bytes written", .{ key, total });
        const dt_ns: f64 = @floatFromInt(
            @as(
                i64,
                @intCast(t0.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.toNanoseconds()),
            ),
        );
        if (dt_ns > 0) {
            self.disk_bw_bps = @as(f64, @floatFromInt(total)) / (dt_ns / 1e9);
        }
    }

    /// The own-namespace share of the directory: bytes, file count, and the oldest file by mtime.
    const OwnScan = struct {
        total: u64,
        count: usize,
        oldest: [256]u8,
        oldest_len: usize,
    };

    /// Measure the own-namespace share of the directory. Only the `model_fp` in the header tells
    /// an own file from a foreign one. Null means the directory did not open.
    fn scanOwnNamespace(self: *const PromptCache, io: Io) ?OwnScan {
        var dir = Io.Dir.cwd().openDir(io, self.dir.?, .{ .iterate = true }) catch return null;
        defer dir.close(io);

        var scan: OwnScan = .{ .total = 0, .count = 0, .oldest = undefined, .oldest_len = 0 };
        var oldest_ns: i96 = std.math.maxInt(i96);
        var it = dir.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".zkv")) continue;
            if (!self.ownsFile(io, &dir, entry.name)) continue;

            const st = dir.statFile(io, entry.name, .{}) catch continue;
            scan.total += st.size;
            scan.count += 1;
            if (st.mtime.nanoseconds < oldest_ns and entry.name.len <= scan.oldest.len) {
                oldest_ns = st.mtime.nanoseconds;
                @memcpy(scan.oldest[0..entry.name.len], entry.name);
                scan.oldest_len = entry.name.len;
            }
        }

        return scan;
    }

    /// Report whether `name` holds a snapshot of this namespace. It reads the header alone.
    fn ownsFile(self: *const PromptCache, io: Io, dir: *Io.Dir, name: []const u8) bool {
        const file = dir.openFile(io, name, .{}) catch return false;
        defer file.close(io);

        var hdr: DiskHeader = undefined;
        const n = file.readPositionalAll(io, std.mem.asBytes(&hdr), 0) catch return false;
        if (n < @sizeOf(DiskHeader)) return false;

        return hdr.magic == magic and hdr.version == version and hdr.model_fp == self.model_fp;
    }

    /// Evict the oldest own file by mtime until this namespace plus `need` fits `disk_budget`.
    /// Best-effort: a directory error stops the eviction.
    fn diskEvictToFit(self: *PromptCache, need: usize) void {
        const io = self.io orelse return;
        var scan = self.scanOwnNamespace(io) orelse return;
        // Each pass deletes one own file, so the first scan count bounds the passes.
        const max_passes = scan.count;
        for (0..max_passes) |_| {
            if (scan.total + need <= self.disk_budget or scan.oldest_len == 0) return;

            var pathbuf: [512]u8 = undefined;
            const path = std.fmt.bufPrint(&pathbuf, "{s}/{s}", .{
                self.dir.?, scan.oldest[0..scan.oldest_len],
            }) catch return;
            Io.Dir.cwd().deleteFile(io, path) catch return;
            log.debug(
                "disk evict: {s}, own namespace over budget",
                .{scan.oldest[0..scan.oldest_len]},
            );
            scan = self.scanOwnNamespace(io) orelse return;
        }
    }
};

const ModelEntry = struct {
    tokens: [model_tokens_max]u32,
    n: u32,
    bytes: usize,
    last_used: u64,
};

const CacheModel = struct {
    entries: [model_cap]ModelEntry = undefined,
    len: usize = 0,
    used: usize = 0,
    tick: u64 = 0,
    budget: usize,

    fn find(m: *const CacheModel, tokens: []const u32) ?usize {
        for (m.entries[0..m.len], 0..) |*e, i| {
            if (std.mem.eql(u32, e.tokens[0..e.n], tokens)) return i;
        }

        return null;
    }

    fn insert(m: *CacheModel, tokens: []const u32, bytes: usize) void {
        while (m.used + bytes > m.budget and m.len > 0) {
            var lru: usize = 0;
            for (m.entries[0..m.len], 0..) |*e, i| {
                if (e.last_used < m.entries[lru].last_used) {
                    lru = i;
                }
            }

            m.used -= m.entries[lru].bytes;
            m.entries[lru] = m.entries[m.len - 1];
            m.len -= 1;
        }

        std.debug.assert(m.len < model_cap);
        m.tick += 1;
        var e: ModelEntry = .{
            .tokens = undefined,
            .n = @intCast(tokens.len),
            .bytes = bytes,
            .last_used = m.tick,
        };
        @memcpy(e.tokens[0..tokens.len], tokens);
        m.entries[m.len] = e;
        m.len += 1;
        m.used += bytes;
    }

    /// The longest held sequence that is a prefix of `query`, and a hit
    /// bumps it. The empty sequence never counts.
    fn bestPrefix(m: *CacheModel, query: []const u32) ?usize {
        var best: ?usize = null;
        for (m.entries[0..m.len], 0..) |*e, i| {
            if (e.n == 0 or e.n > query.len) continue;
            if (!std.mem.eql(u32, e.tokens[0..e.n], query[0..e.n])) continue;
            if (best == null or e.n > m.entries[best.?].n) {
                best = i;
            }
        }

        if (best) |i| {
            m.tick += 1;
            m.entries[i].last_used = m.tick;
        }

        return best;
    }
};

// -- api --------------------------------------------------------------------

/// Parse the bytes of one `.zkv` file of the namespace `model_fp`. The file is input, so every size
/// sum is checked and every view lies inside `bytes`.
pub fn parseImage(bytes: []const u8, model_fp: u64) ImageError!Image {
    if (bytes.len < @sizeOf(DiskHeader)) return error.ImageTruncated;

    var hdr: DiskHeader = undefined;
    @memcpy(std.mem.asBytes(&hdr), bytes[0..@sizeOf(DiskHeader)]);
    if (hdr.magic != magic or hdr.version != version or hdr.model_fp != model_fp) {
        return error.ImageForeign;
    }

    // Size the three parts in u64, then check each sum.
    const tok_bytes: u64 = @as(u64, hdr.n_tokens) * @sizeOf(u32);
    const log_bytes: u64 = @as(u64, hdr.n_logits) * @sizeOf(f32);
    const after_tokens = std.math.add(u64, @sizeOf(DiskHeader), tok_bytes) catch unreachable;
    const after_logits = std.math.add(u64, after_tokens, log_bytes) catch unreachable;
    const need = std.math.add(u64, after_logits, hdr.blob_len) catch {
        return error.ImageSizeOverflow;
    };
    if (bytes.len < need) return error.ImageTruncated;

    // Check the payload FNV.
    const image: Image = .{
        .n_tokens = hdr.n_tokens,
        .tokens = bytes[@sizeOf(DiskHeader)..@intCast(after_tokens)],
        .logits = bytes[@intCast(after_tokens)..@intCast(after_logits)],
        .blob = bytes[@intCast(after_logits)..@intCast(need)],
    };
    var h = std.hash.Fnv1a_64.init();
    h.update(image.tokens);
    h.update(image.logits);
    h.update(image.blob);
    if (h.final() != hdr.payload_fnv) return error.ImageTorn;

    return image;
}

/// Refuse a snapshot of no rows, or of more rows than a runner of `max_rows` positions holds. A
/// cache file can hold either, so the caller treats the refusal as a miss. Call it before the
/// caller prices the blob of `n_rows` rows.
pub fn checkRows(n_rows: u32, max_rows: u32) error{SnapshotMismatch}!void {
    if (n_rows >= 1 and n_rows <= max_rows) return;

    const format = "prompt cache: a snapshot of {d} rows does not fit a runner of {d} rows, " ++
        "so it is a miss";
    if (mismatch_level == .debug) {
        log.debug(format, .{ n_rows, max_rows });
    } else {
        log.warn(format, .{ n_rows, max_rows });
    }
    return error.SnapshotMismatch;
}

/// Refuse a snapshot whose blob or logits differ in length from what the runner restores.
pub fn checkSizes(
    blob_len: usize,
    want_blob_len: u64,
    logits_len: usize,
    want_logits_len: usize,
) error{SnapshotMismatch}!void {
    if (blob_len == want_blob_len and logits_len == want_logits_len) return;

    const format = "prompt cache: a snapshot of {d} blob bytes and {d} logits differs from " ++
        "the {d} bytes and {d} logits of this runner, so it is a miss";
    const args = .{ blob_len, logits_len, want_blob_len, want_logits_len };
    if (mismatch_level == .debug) {
        log.debug(format, args);
    } else {
        log.warn(format, args);
    }
    return error.SnapshotMismatch;
}

/// Apply the default to a raw env value. The CLI reads the variable once at startup.
pub fn resolveCacheDir(env_value: ?[]const u8) []const u8 {
    const v = env_value orelse return default_cache_dir;
    return if (v.len == 0) default_cache_dir else v;
}

/// Weights identity: a model must never restore the state of another. It pairs the registry
/// name with the GGUF byte size. A file swap that keeps both stays undetected.
pub fn modelFingerprint(pub_name: []const u8, file_size: u64) u64 {
    var h = std.hash.Fnv1a_64.init();
    h.update(pub_name);
    h.update(std.mem.asBytes(&file_size));
    return h.final();
}

/// Key for the prefix `tokens` under `model_fp`: FNV-1a over the fingerprint, then the token
/// bytes. The lookup compares the stored tokens, so the hash only needs a good spread.
pub fn prefixKey(model_fp: u64, tokens: []const u32) u64 {
    var h = std.hash.Fnv1a_64.init();
    h.update(std.mem.asBytes(&model_fp));
    h.update(std.mem.sliceAsBytes(tokens));
    return h.final();
}

/// Fold a backend tag into a model fingerprint. The gate pins greedy tokens per backend and
/// not KV bits across backends, so an entry must never restore into another family.
pub fn backendFingerprint(model_fp: u64, backend_tag: []const u8) u64 {
    var h = std.hash.Fnv1a_64.init();
    h.update(std.mem.asBytes(&model_fp));
    h.update(backend_tag);
    return h.final();
}

/// The admission limit: a restore must cost under half of the prefill it replaces.
/// The caller measures both, and the restore estimate is the timed D2H save. The 2x margin
/// drops a marginal prefix whose win does not survive noise.
pub fn worthwhile(prefill_ns: u64, restore_ns: u64) bool {
    return prefill_ns > restore_ns *| 2;
}

// -- helpers ----------------------------------------------------------------

// -- tests (pure logic, no GPU) ---------------------------------------------
/// Test helper: the on-disk path of one cached prefix.
fn testDiskPath(buf: []u8, dir: []const u8, key: u64) ![]u8 {
    return std.fmt.bufPrint(buf, "{s}/{x:0>16}.zkv", .{ dir, key });
}

/// Test helper: pin one snapshot file's mtime. Eviction follows mtime, and back-to-back files tie.
fn testSetMtime(io: Io, dir: []const u8, key: u64, ns: i96) !void {
    var buf: [256]u8 = undefined;
    const path = try testDiskPath(&buf, dir, key);
    try Io.Dir.cwd().setTimestamps(io, path, .{
        .modify_timestamp = .{ .new = .{ .nanoseconds = ns } },
    });
}

/// Test helper: whether a snapshot file is present.
fn testDiskFileExists(io: Io, dir: []const u8, key: u64) bool {
    var buf: [256]u8 = undefined;
    const path = testDiskPath(&buf, dir, key) catch return false;
    const f = Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

test "PromptCache: disk pressure evicts the oldest file of the same namespace" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dirbuf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dirbuf, ".zig-cache/tmp/{s}/pc", .{&tmp.sub_path});

    const fp = modelFingerprint("m", 1);
    const blob = [_]u8{7} ** 256;
    const logits = [_]f32{ 0, 1 };
    const t1 = [_]u32{ 1, 1, 1, 1 };
    const t2 = [_]u32{ 2, 2, 2, 2 };
    const t3 = [_]u32{ 3, 3, 3, 3 };
    // The persisted file is header + tokens + logits + blob. A budget of two
    // files plus a margin admits the third only after one eviction.
    const file_bytes: u64 = @sizeOf(
        DiskHeader,
    ) + t1.len * @sizeOf(u32) + logits.len * @sizeOf(f32) + blob.len;
    const budget: usize = @intCast(2 * file_bytes + 8);

    var cache = PromptCache.init(gpa, 1 << 20);
    defer cache.deinit();

    try cache.enableDisk(io, dir, budget, fp);

    // Prefill seconds against restore milliseconds clears both admission bars
    // at any plausible disk bandwidth.
    try std.testing.expect(
        cache.admit(prefixKey(fp, &t1), &t1, &blob, &logits, 10_000_000_000, 1_000_000),
    );
    try std.testing.expect(cache.last_admit_disk);
    try testSetMtime(io, dir, prefixKey(fp, &t1), 1_000_000_000);

    try std.testing.expect(
        cache.admit(prefixKey(fp, &t2), &t2, &blob, &logits, 10_000_000_000, 1_000_000),
    );
    try std.testing.expect(cache.last_admit_disk);
    try testSetMtime(io, dir, prefixKey(fp, &t2), 2_000_000_000);

    // Two files already fill the budget, so the third forces one eviction.
    try std.testing.expect(
        cache.admit(prefixKey(fp, &t3), &t3, &blob, &logits, 10_000_000_000, 1_000_000),
    );
    try std.testing.expect(cache.last_admit_disk);
    try testSetMtime(io, dir, prefixKey(fp, &t3), 3_000_000_000);

    try std.testing.expect(!testDiskFileExists(io, dir, prefixKey(fp, &t1))); // oldest
    try std.testing.expect(testDiskFileExists(io, dir, prefixKey(fp, &t2)));
    try std.testing.expect(testDiskFileExists(io, dir, prefixKey(fp, &t3)));

    const scan = cache.scanOwnNamespace(io) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(usize, 2), scan.count);
    try std.testing.expect(scan.total <= budget);
}

test "PromptCache: disk eviction never counts or deletes a foreign namespace" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dirbuf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dirbuf, ".zig-cache/tmp/{s}/pc", .{&tmp.sub_path});

    const fp_a = modelFingerprint("a", 1);
    const fp_b = modelFingerprint("b", 2);
    const blob = [_]u8{7} ** 256;
    const logits = [_]f32{ 0, 1 };
    const t1 = [_]u32{ 1, 1, 1, 1 };
    const t2 = [_]u32{ 2, 2, 2, 2 };
    const t3 = [_]u32{ 3, 3, 3, 3 };
    const t4 = [_]u32{ 4, 4, 4, 4 };
    const file_bytes: u64 = @sizeOf(
        DiskHeader,
    ) + t1.len * @sizeOf(u32) + logits.len * @sizeOf(f32) + blob.len;
    // One file per namespace. Every further admission must evict.
    const budget: usize = @intCast(file_bytes);

    // Both caches point at one directory under their own fingerprints. The
    // prefix tokens repeat across the two, and the key folds the fingerprint,
    // so the two namespaces hold different files for the same prefix.
    var a = PromptCache.init(gpa, 1 << 20);
    defer a.deinit();

    try a.enableDisk(io, dir, budget, fp_a);
    var b = PromptCache.init(gpa, 1 << 20);
    defer b.deinit();

    try b.enableDisk(io, dir, budget, fp_b);

    // B lays down two files, both older than anything A writes. A blind policy
    // that summed the directory evicts these first.
    try std.testing.expect(
        b.admit(prefixKey(fp_b, &t1), &t1, &blob, &logits, 10_000_000_000, 1_000_000),
    );
    try std.testing.expect(b.last_admit_disk);
    try testSetMtime(io, dir, prefixKey(fp_b, &t1), 1_000_000_000);
    b.disk_budget = 1 << 20; // hold both B files while A takes the pressure
    try std.testing.expect(
        b.admit(prefixKey(fp_b, &t2), &t2, &blob, &logits, 10_000_000_000, 1_000_000),
    );
    try std.testing.expect(b.last_admit_disk);
    try testSetMtime(io, dir, prefixKey(fp_b, &t2), 2_000_000_000);

    // A now writes four files under a one-file budget.
    const a_keys = [_]u64{
        prefixKey(fp_a, &t1),
        prefixKey(fp_a, &t2),
        prefixKey(fp_a, &t3),
        prefixKey(fp_a, &t4),
    };
    const a_tokens = [_][]const u32{ &t1, &t2, &t3, &t4 };
    for (a_keys, a_tokens, 0..) |k, toks, i| {
        try std.testing.expect(a.admit(k, toks, &blob, &logits, 10_000_000_000, 1_000_000));
        try std.testing.expect(a.last_admit_disk);
        try testSetMtime(io, dir, k, @as(i96, @intCast(3 + i)) * 1_000_000_000);
    }

    // Every B file survives, and A holds one file inside its own budget.
    try std.testing.expect(testDiskFileExists(io, dir, prefixKey(fp_b, &t1)));
    try std.testing.expect(testDiskFileExists(io, dir, prefixKey(fp_b, &t2)));
    const scan_a = a.scanOwnNamespace(io) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(usize, 1), scan_a.count);
    try std.testing.expect(scan_a.total <= budget);
    try std.testing.expect(testDiskFileExists(io, dir, a_keys[3])); // newest A file

    // The other direction. A's one file becomes the oldest in the directory,
    // and B goes over its budget. A blind policy takes A's file.
    try testSetMtime(io, dir, a_keys[3], 500_000_000);
    b.disk_budget = budget;
    try std.testing.expect(
        b.admit(prefixKey(fp_b, &t3), &t3, &blob, &logits, 10_000_000_000, 1_000_000),
    );
    try std.testing.expect(b.last_admit_disk);
    try testSetMtime(io, dir, prefixKey(fp_b, &t3), 9_000_000_000);

    try std.testing.expect(testDiskFileExists(io, dir, a_keys[3]));
    const scan_a2 = a.scanOwnNamespace(io) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(usize, 1), scan_a2.count);
    const scan_b = b.scanOwnNamespace(io) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(usize, 1), scan_b.count);
    try std.testing.expect(scan_b.total <= budget);
    try std.testing.expect(!testDiskFileExists(io, dir, prefixKey(fp_b, &t1))); // oldest B file
    try std.testing.expect(!testDiskFileExists(io, dir, prefixKey(fp_b, &t2)));
    try std.testing.expect(testDiskFileExists(io, dir, prefixKey(fp_b, &t3)));
}

test "worthwhile: model-adaptive admission limit" {
    // Low-cost prefill (state costs more to move than recompute) -> declined.
    try std.testing.expect(!worthwhile(3_000_000, 4_000_000));
    // Marginal (within 2x margin) -> declined.
    try std.testing.expect(!worthwhile(7_000_000, 4_000_000));
    // Qwen-9B-like: prefill seconds, restore ms -> admitted.
    try std.testing.expect(worthwhile(270_000_000, 4_000_000));
}

test "prefixKey: distinct prefixes and models are collision-free, and the same input is stable" {
    const fp_a = modelFingerprint("model-a", 100);
    const fp_b = modelFingerprint("model-b", 100);
    const t1 = [_]u32{ 1, 2, 3 };
    const t2 = [_]u32{ 1, 2, 4 };
    try std.testing.expect(prefixKey(fp_a, &t1) == prefixKey(fp_a, &t1)); // stable
    try std.testing.expect(prefixKey(fp_a, &t1) != prefixKey(fp_a, &t2)); // diff tokens
    try std.testing.expect(prefixKey(fp_a, &t1) != prefixKey(fp_b, &t1)); // diff model
}

test "PromptCache: admit, hit, collision-miss, LRU eviction" {
    const gpa = std.testing.allocator;
    const blob = [_]u8{0} ** 1000;
    const logits = [_]f32{0} ** 10;
    // Budget fits ~2 entries: each entry = 3*4 + 1000 + 10*4 = 1052 B.
    var cache = PromptCache.init(gpa, 2 * 1052 + 10);
    defer cache.deinit();

    const fp = modelFingerprint("m", 1);
    const ta = [_]u32{ 1, 2, 3 };
    const tb = [_]u32{ 4, 5, 6 };
    const tc = [_]u32{ 7, 8, 9 };
    const ka = prefixKey(fp, &ta);
    const kb = prefixKey(fp, &tb);
    const kc = prefixKey(fp, &tc);

    // Admit a and b (cost: prefill >> restore).
    try std.testing.expect(cache.admit(ka, &ta, &blob, &logits, 100_000_000, 1_000_000));
    try std.testing.expect(cache.admit(kb, &tb, &blob, &logits, 100_000_000, 1_000_000));

    // Hit a (also bumps it as most-recently-used).
    try std.testing.expect(cache.lookup(ka, &ta) != null);
    // Collision guard: same key bytes do not matter. Different tokens under a
    // *wrong* key still miss, and a real key with mismatched tokens misses.
    try std.testing.expect(cache.lookup(ka, &tb) == null);

    // Admitting c evicts the LRU (b, since a was just hit).
    try std.testing.expect(cache.admit(kc, &tc, &blob, &logits, 100_000_000, 1_000_000));
    try std.testing.expect(cache.lookup(kb, &tb) == null); // evicted
    try std.testing.expect(cache.lookup(ka, &ta) != null); // kept
    try std.testing.expect(cache.lookup(kc, &tc) != null); // newest

    // Not worthwhile -> not admitted, cache unchanged.
    const td = [_]u32{ 10, 11, 12 };
    try std.testing.expect(
        !cache.admit(prefixKey(fp, &td), &td, &blob, &logits, 1_000_000, 1_000_000),
    );
}

/// An allocator that fails one allocation, and then succeeds. The std `FailingAllocator`
/// fails every allocation after its index.
const FailOnce = struct {
    left: usize,

    const vtable: Allocator.VTable = .{
        .alloc = alloc,
        .resize = Allocator.noResize,
        .remap = Allocator.noRemap,
        .free = free,
    };

    fn allocator(self: *FailOnce) Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FailOnce = @ptrCast(@alignCast(ctx));
        const fail = self.left == 0;
        self.left -%= 1;
        if (fail) return null;

        return std.testing.allocator.rawAlloc(len, a, ra);
    }

    fn free(ctx: *anyopaque, memory: []u8, a: std.mem.Alignment, ra: usize) void {
        _ = ctx;
        std.testing.allocator.rawFree(memory, a, ra);
    }
};

test "PromptCache: admit under OOM leaks nothing and reports what the RAM tier holds" {
    const fp = modelFingerprint("m", 1);
    const blob = [_]u8{7} ** 256;
    const logits = [_]f32{ 0, 1 };
    const tokens = [_]u32{ 1, 2, 3 };
    const key = prefixKey(fp, &tokens);
    // An admission makes four allocations: tokens, blob, logits, and the entry list.
    const allocations = 4;
    for (0..allocations + 1) |fail_index| {
        var failing: std.testing.FailingAllocator = .init(
            std.testing.allocator,
            .{ .fail_index = fail_index },
        );
        var cache = PromptCache.init(failing.allocator(), 1 << 20);
        defer cache.deinit();

        const admitted = cache.admit(key, &tokens, &blob, &logits, 100_000_000, 1_000_000);
        try std.testing.expectEqual(fail_index == allocations, admitted);
        try std.testing.expectEqual(admitted, cache.last_admit_ram);
        try std.testing.expectEqual(@intFromBool(admitted), cache.entries.items.len);
        try std.testing.expectEqual(admitted, cache.lookup(key, &tokens) != null);
    }
}

test "PromptCache: a RAM tier out of memory leaves the disk tier its turn" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dirbuf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dirbuf, ".zig-cache/tmp/{s}/pc", .{&tmp.sub_path});

    const fp = modelFingerprint("m", 1);
    const blob = [_]u8{7} ** 256;
    const logits = [_]f32{ 0, 1 };
    const tokens = [_]u32{ 1, 2, 3 };
    const key = prefixKey(fp, &tokens);
    // The RAM tier takes the first four allocations of an admission.
    for (0..4) |fail_at| {
        var once: FailOnce = .{ .left = std.math.maxInt(usize) };
        var cache = PromptCache.init(once.allocator(), 1 << 20);
        defer cache.deinit();

        try cache.enableDisk(io, dir, 1 << 20, fp);
        once.left = fail_at;
        const admitted = cache.admit(key, &tokens, &blob, &logits, 10_000_000_000, 1_000_000);
        try std.testing.expect(admitted);
        try std.testing.expect(!cache.last_admit_ram);
        try std.testing.expect(cache.last_admit_disk);
    }
}

test "PromptCache: longest-prefix match" {
    const gpa = std.testing.allocator;
    const blob = [_]u8{0} ** 100;
    const logits = [_]f32{0} ** 4;
    var cache = PromptCache.init(gpa, 1 << 20);
    defer cache.deinit();

    const fp = modelFingerprint("m", 1);

    // A conversation that strictly extends: [1,2] then [1,2,3,4] are saved boundaries.
    const seq = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const b2 = 2; // prefix [1,2]
    const b4 = 4; // prefix [1,2,3,4]
    try std.testing.expect(
        cache.admit(
            prefixKey(fp, seq[0..b2]),
            seq[0..b2],
            &blob,
            &logits,
            100_000_000,
            1_000_000,
        ),
    );
    try std.testing.expect(
        cache.admit(
            prefixKey(fp, seq[0..b4]),
            seq[0..b4],
            &blob,
            &logits,
            100_000_000,
            1_000_000,
        ),
    );

    // New request = full seq. The longest cached prefix is b4 (not b2).
    const hit = cache.lookupLongestPrefix(
        fp,
        &seq,
        &[_]u32{ b2, b4 },
    ) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, b4), hit.n);

    // A request that only shares the first 3 tokens [1,2,9,...]: b4 prefix [1,2,3,4]
    // does NOT match (token 3 differs), but b2 [1,2] does.
    const other = [_]u32{ 1, 2, 9, 9 };
    const hit2 = cache.lookupLongestPrefix(
        fp,
        &other,
        &[_]u32{ b2, b4 },
    ) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, b2), hit2.n);

    // No shared prefix -> miss.
    const none = [_]u32{ 7, 8 };
    try std.testing.expect(cache.lookupLongestPrefix(fp, &none, &[_]u32{ b2, b4 }) == null);
}

test "PromptCache: best-prefix match without caller boundaries" {
    const gpa = std.testing.allocator;
    const blob = [_]u8{0} ** 100;
    const logits = [_]f32{0} ** 4;
    var cache = PromptCache.init(gpa, 1 << 20);
    defer cache.deinit();

    const fp = modelFingerprint("m", 1);

    // The same two saved prefixes, but the caller passes no boundary list.
    const seq = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try std.testing.expect(
        cache.admit(
            prefixKey(fp, seq[0..2]),
            seq[0..2],
            &blob,
            &logits,
            100_000_000,
            1_000_000,
        ),
    );
    try std.testing.expect(
        cache.admit(
            prefixKey(fp, seq[0..4]),
            seq[0..4],
            &blob,
            &logits,
            100_000_000,
            1_000_000,
        ),
    );

    // Both entries are prefixes of the request, so the longer one wins.
    const hit = cache.lookupBestPrefix(&seq) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 4), hit.n);

    // [1,2,9,9] diverges at token 3, so only the [1,2] entry matches.
    const other = [_]u32{ 1, 2, 9, 9 };
    const hit2 = cache.lookupBestPrefix(&other) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 2), hit2.n);

    // An entry as long as the whole request still counts as a prefix.
    const exact = cache.lookupBestPrefix(seq[0..4]) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 4), exact.n);

    // No shared prefix -> miss. No tokens -> miss.
    try std.testing.expect(cache.lookupBestPrefix(&[_]u32{ 7, 8 }) == null);
    try std.testing.expect(cache.lookupBestPrefix(&[_]u32{}) == null);
}

test "PromptCache: best-prefix hit counts as a use" {
    const gpa = std.testing.allocator;
    const blob = [_]u8{0} ** 1000;
    const logits = [_]f32{0} ** 10;
    // Budget fits ~2 entries: each entry = 3*4 + 1000 + 10*4 = 1052 B.
    var cache = PromptCache.init(gpa, 2 * 1052 + 10);
    defer cache.deinit();

    const fp = modelFingerprint("m", 1);
    const ta = [_]u32{ 1, 2, 3 };
    const tb = [_]u32{ 4, 5, 6 };
    const tc = [_]u32{ 7, 8, 9 };
    try std.testing.expect(
        cache.admit(prefixKey(fp, &ta), &ta, &blob, &logits, 100_000_000, 1_000_000),
    );
    try std.testing.expect(
        cache.admit(prefixKey(fp, &tb), &tb, &blob, &logits, 100_000_000, 1_000_000),
    );

    // The prefix hit bumps a, so admitting c evicts b instead.
    try std.testing.expect(cache.lookupBestPrefix(&ta) != null);
    try std.testing.expect(
        cache.admit(prefixKey(fp, &tc), &tc, &blob, &logits, 100_000_000, 1_000_000),
    );
    try std.testing.expect(cache.lookup(prefixKey(fp, &tb), &tb) == null); // evicted
    try std.testing.expect(cache.lookup(prefixKey(fp, &ta), &ta) != null); // kept
}

test "PromptCache: the disk index survives a new instance and promotes on best-prefix" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dirbuf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dirbuf, ".zig-cache/tmp/{s}/pc", .{&tmp.sub_path});

    const fp = modelFingerprint("m", 1);
    const seq = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const blob = [_]u8{7} ** 64;
    const logits = [_]f32{ 0, 1 };

    // Instance 1 admits past both bars and persists the 4-token boundary.
    {
        var cache = PromptCache.init(gpa, 1 << 20);
        defer cache.deinit();

        try cache.enableDisk(io, dir, 1 << 20, fp);
        try std.testing.expect(
            cache.admit(
                prefixKey(fp, seq[0..4]),
                seq[0..4],
                &blob,
                &logits,
                100_000_000,
                1_000_000,
            ),
        );
        try std.testing.expect(cache.last_admit_disk);
    }

    // Instance 2 (a restart): the boot scan indexes the file, and the
    // best-prefix lookup promotes it with tokens, logits, and blob intact.
    var cache = PromptCache.init(gpa, 1 << 20);
    defer cache.deinit();

    try cache.enableDisk(io, dir, 1 << 20, fp);
    try std.testing.expectEqual(@as(usize, 0), cache.diskIndexLen());
    cache.loadDiskIndex();
    try std.testing.expectEqual(@as(usize, 1), cache.diskIndexLen());

    const hit = cache.lookupBestPrefix(&seq) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 4), hit.n);
    try std.testing.expect(cache.last_hit_from_disk);
    try std.testing.expectEqualSlices(u8, &blob, hit.blob);
    try std.testing.expectEqualSlices(f32, &logits, hit.last_logits);

    // The promotion inserted it, so a second lookup serves from RAM. Both
    // serves count on the entry: the promote and the RAM hit.
    const again = cache.lookupBestPrefix(&seq) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 4), again.n);
    try std.testing.expect(!cache.last_hit_from_disk);
    try std.testing.expectEqual(@as(u64, 2), again.hits);

    // A foreign fingerprint is another namespace: its scan indexes nothing.
    var other = PromptCache.init(gpa, 1 << 20);
    defer other.deinit();

    try other.enableDisk(io, dir, 1 << 20, modelFingerprint("other", 2));
    other.loadDiskIndex();
    try std.testing.expectEqual(@as(usize, 0), other.diskIndexLen());
}

test "PromptCache: admit and find an entry that carries no logits" {
    const gpa = std.testing.allocator;
    const blob = [_]u8{0} ** 100;
    var cache = PromptCache.init(gpa, 1 << 20);
    defer cache.deinit();

    const fp = modelFingerprint("m", 1);

    // The serve path samples from a fresh forward, so it stores no logits. The
    // empty slice must round-trip.
    const t = [_]u32{ 1, 2, 3 };
    try std.testing.expect(
        cache.admit(prefixKey(fp, &t), &t, &blob, &[_]f32{}, 100_000_000, 1_000_000),
    );
    const hit = cache.lookupBestPrefix(&t) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 3), hit.n);
    try std.testing.expectEqual(@as(usize, 0), hit.last_logits.len);
}

fn genTokens(r: std.Random, buf: *[model_tokens_max]u32) []u32 {
    const n = r.uintAtMost(usize, model_tokens_max);
    // Three token values make shared prefixes common.
    for (buf[0..n]) |*t| {
        t.* = r.uintAtMost(u32, 2);
    }

    return buf[0..n];
}

/// Over random admits and queries, the cache holds what the model holds, and
/// `lookupBestPrefix` returns the longest held prefix of the query.
fn checkPromptCache(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const fp = modelFingerprint("m", 1);
    const budget = 64 + r.uintAtMost(usize, 512);
    var cache = PromptCache.init(std.testing.allocator, budget);
    defer cache.deinit();

    var model: CacheModel = .{ .budget = budget };
    const blob = [_]u8{0x33} ** 160;
    const logits = [_]f32{0} ** 8;

    for (0..64) |_| {
        var buf: [model_tokens_max]u32 = undefined;
        const tokens = genTokens(r, &buf);
        switch (r.uintLessThan(u8, 3)) {
            0 => {
                // A held sequence is never admitted twice, so each query has one answer.
                if (model.find(tokens) != null) continue;

                const b = blob[0..r.uintAtMost(usize, blob.len)];
                const l = logits[0..r.uintAtMost(usize, logits.len)];
                try std.testing.expect(
                    cache.admit(prefixKey(fp, tokens), tokens, b, l, 100_000_000, 1_000_000),
                );
                model.insert(tokens, tokens.len * @sizeOf(u32) + b.len + l.len * @sizeOf(f32));
            },
            1 => {
                const want = model.bestPrefix(tokens);
                const got = cache.lookupBestPrefix(tokens);
                try std.testing.expectEqual(want == null, got == null);
                if (got) |hit| {
                    try std.testing.expectEqualSlices(
                        u32,
                        model.entries[want.?].tokens[0..model.entries[want.?].n],
                        hit.tokens,
                    );
                    try std.testing.expectEqualSlices(u32, tokens[0..hit.n], hit.tokens);
                }
            },
            else => {
                const want = model.find(tokens);
                const got = cache.lookup(prefixKey(fp, tokens), tokens);
                try std.testing.expectEqual(want == null, got == null);
                if (want) |i| {
                    model.tick += 1;
                    model.entries[i].last_used = model.tick;
                }
            },
        }

        try std.testing.expectEqual(model.len, cache.entries.items.len);
        try std.testing.expectEqual(model.used, cache.used_bytes);
        for (cache.entries.items) |*e| {
            try std.testing.expect(model.find(e.tokens) != null);
        }
    }
}

test "property: the RAM tier matches a linear-scan model, and best-prefix is the longest" {
    try prop.forAll(
        .{ .site = 0xCAC4_E001, .cases = 300 },
        {},
        u64,
        prop.genSeed(void),
        checkPromptCache,
    );
}

/// A valid image of a 3-token prefix, 8 logits, and a blob of `zkv_blob_len(3)` bytes, with the
/// offset of each header field.
const ZkvImage = struct {
    bytes: [zkv_image_len]u8 align(8),
    const fields = [_]struct { offset: usize, width: u8 }{
        .{ .offset = @offsetOf(DiskHeader, "magic"), .width = 4 },
        .{ .offset = @offsetOf(DiskHeader, "version"), .width = 4 },
        .{ .offset = @offsetOf(DiskHeader, "model_fp"), .width = 8 },
        .{ .offset = @offsetOf(DiskHeader, "n_tokens"), .width = 4 },
        .{ .offset = @offsetOf(DiskHeader, "n_logits"), .width = 4 },
        .{ .offset = @offsetOf(DiskHeader, "blob_len"), .width = 8 },
        .{ .offset = @offsetOf(DiskHeader, "payload_fnv"), .width = 8 },
    };
};

const zkv_fp: u64 = 0xF00D;
const zkv_rows_max: u32 = 16;
const zkv_vocab: u32 = 8;

/// The blob bytes of the test runner at `n` rows: 24 bytes a row and a fixed 40.
fn zkvBlobLen(n: u32) u64 {
    return @as(u64, n) * 24 + 40;
}

const zkv_image_len = @sizeOf(DiskHeader) + 3 * 4 + zkv_vocab * 4 + 3 * 24 + 40;

fn validZkv() ZkvImage {
    var img: ZkvImage = .{ .bytes = undefined };
    var payload: [zkv_image_len - @sizeOf(DiskHeader)]u8 = undefined;
    for (&payload, 0..) |*b, i| {
        b.* = @truncate(i * 7);
    }

    const hdr: DiskHeader = .{
        .magic = magic,
        .version = version,
        .model_fp = zkv_fp,
        .n_tokens = 3,
        .n_logits = zkv_vocab,
        .blob_len = zkvBlobLen(3),
        .payload_fnv = std.hash.Fnv1a_64.hash(&payload),
    };
    @memcpy(img.bytes[0..@sizeOf(DiskHeader)], std.mem.asBytes(&hdr));
    @memcpy(img.bytes[@sizeOf(DiskHeader)..], &payload);
    return img;
}

/// Parse `bytes` and check the snapshot against the test runner. A typed error at any stage is
/// a miss. A parsed image lies inside the bytes.
fn loadZkv(bytes: []const u8) !void {
    const image = parseImage(bytes, zkv_fp) catch return;
    const lo = @intFromPtr(bytes.ptr);
    for ([_][]const u8{ image.tokens, image.logits, image.blob }) |view| {
        try std.testing.expect(@intFromPtr(view.ptr) >= lo);
        try std.testing.expect(@intFromPtr(view.ptr) + view.len <= lo + bytes.len);
    }

    checkRows(image.n_tokens, zkv_rows_max) catch return;
    checkSizes(image.blob.len, zkvBlobLen(image.n_tokens), image.logits.len / 4, zkv_vocab) catch
        return;
}

test "parseImage: the valid image parses and fits the test runner" {
    const img = validZkv();
    const image = try parseImage(&img.bytes, zkv_fp);
    try std.testing.expectEqual(@as(u32, 3), image.n_tokens);
    try checkRows(image.n_tokens, zkv_rows_max);
    try checkSizes(image.blob.len, zkvBlobLen(3), image.logits.len / 4, zkv_vocab);
    try std.testing.expectError(error.ImageForeign, parseImage(&img.bytes, zkv_fp + 1));
    try std.testing.expectError(error.ImageTruncated, parseImage(img.bytes[0..20], zkv_fp));
}

/// One header change. `rehash` writes the FNV of the payload the new header names, as a writer
/// that emits a consistent file does, so the case reaches the shape checks.
const ZkvMutation = struct { field: usize, op: u3, cut: ?usize, rehash: bool };

fn genZkv(_: void, r: std.Random, size: u32) ZkvMutation {
    _ = size;
    return .{
        .field = r.uintLessThan(usize, ZkvImage.fields.len),
        .op = r.uintLessThan(u3, 6),
        .cut = if (r.uintLessThan(u8, 4) == 0) r.uintAtMost(usize, zkv_image_len) else null,
        .rehash = r.boolean(),
    };
}

fn checkZkv(_: void, m: ZkvMutation) anyerror!void {
    var img = validZkv();
    const f = ZkvImage.fields[m.field];
    const at = img.bytes[f.offset..];
    const old: u64 = switch (f.width) {
        4 => std.mem.readInt(u32, at[0..4], .little),
        else => std.mem.readInt(u64, at[0..8], .little),
    };
    const max: u64 = if (f.width == 4) std.math.maxInt(u32) else std.math.maxInt(u64);
    const value: u64 = switch (m.op) {
        0 => 0,
        1 => 1,
        2 => if (old == 0) max else old - 1,
        3 => @min(old +| 1, max),
        4 => @min(old *| 2, max),
        else => max,
    };
    if (f.width == 4) {
        std.mem.writeInt(u32, at[0..4], @intCast(value), .little);
    } else {
        std.mem.writeInt(u64, at[0..8], value, .little);
    }

    if (m.rehash) {
        rehashZkv(&img);
    }
    try loadZkv(img.bytes[0 .. m.cut orelse zkv_image_len]);
}

/// Write the FNV of the payload that the header of `img` names, when the payload fits the image.
fn rehashZkv(img: *ZkvImage) void {
    var hdr: DiskHeader = undefined;
    @memcpy(std.mem.asBytes(&hdr), img.bytes[0..@sizeOf(DiskHeader)]);
    const len = @as(u128, hdr.n_tokens) * 4 + @as(u128, hdr.n_logits) * 4 + hdr.blob_len;
    if (len > zkv_image_len - @sizeOf(DiskHeader)) return;

    const payload = img.bytes[@sizeOf(DiskHeader)..][0..@intCast(len)];
    hdr.payload_fnv = std.hash.Fnv1a_64.hash(payload);
    @memcpy(img.bytes[0..@sizeOf(DiskHeader)], std.mem.asBytes(&hdr));
}

test "property: a .zkv image with one changed header field or a cut loads as a miss or fits" {
    try prop.forAll(.{ .site = 0x2C0B_EC01, .cases = 600 }, {}, ZkvMutation, genZkv, checkZkv);
}
