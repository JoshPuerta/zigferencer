//! The KV budget planner: byte budgets in, `n_slots x window` out. It is pure: no allocation, no
//! device query, no I/O. One arithmetic serves both the plan and the preflight, and any caller can
//! replay it in a test.
//!
//! Comptime. `clipCopies` is generic over the copy record type.
//!
//! Modes. Mode A takes a VRAM byte budget X. Alone, X buys slots at the resolved window. With a
//! slot count beside it (`--vram-slots`), the count is pinned and the window follows the money.
//! Mode B has no budget: the caller's structure passes through and the window is the serve
//! default. Degenerate input therefore yields the shipped shapes, which keeps the decoded token
//! hashes stable.
//!
//! Window policy. The window has two ceilings. The model ceiling is `Caps.model_ctx` under the
//! engine bound, and a named window past it is a refusal, never a clamp. The default ceiling also
//! caps at the `run` default of the model, and the money window stops there when the request names
//! no window. The caller fills both default windows from `defaultWindow`, a byte budget per
//! session.
//!
//! Layer split planner. A second planner shares the file and the same usable-VRAM rule. It prices
//! one layer split: the host layers take the prefix `[0..host_layers)` and the GPU layers take the
//! suffix `[host_layers..n_layers)`. It reads the KV structure the first planner resolved, so it
//! never derives a window or a slot count of its own.
//!
//! Layer prices. The planner takes a per-layer price vector, so a family whose layers differ in
//! price is priced by span instead of by one product. It also prices the host RAM of the split, and
//! it refuses a split whose anonymous host bytes pass the usable share of the free RAM.
//!
//! Overflow ledger. The window is at most 2^18 (`limits.max_model_ctx`). The per-token column sums
//! to at most 2^30 and the fixed column to at most 2^32. So one slot's span stays under 2^49. The
//! slot count is at most 2^7 (`limits.max_vram_slots`), so `kv_vram` stays under 2^56.
//!
//! RAM fit formula. The host spend is linear in the window w: spend(w) = P * w + F. P sums the
//! per-token bytes the vector holds per position: `n_slots` copies of the host prefix, plus
//! `blobs` copies of the whole model when the blobs grow with the window. F sums the fixed slabs
//! of the same tiers plus a constant that the caller states: the stage pair, the prefix, and a blob
//! share that the window does not move. The stage pair is constant, because a stage holds at most
//! one chunk, and the stage at the refused window bounds the stage at any narrower one. So the
//! widest w with P * w + F within the usable RAM is floor((usable - F) / P), and no search is
//! needed.

const std = @import("std");
const dtype = @import("dtype.zig");
const limits = @import("limits.zig");
const prop = @import("prop.zig");

// -- constants --------------------------------------------------------------

/// Ceiling on a declared per-token KV cost. It bounds the whole-model sum and
/// any one layer's entry alike. Llama-3.1-8B declares 262144 for the whole
/// model, so this sits far above any real model and keeps every product inside
/// a `u64`.
const max_kv_bytes_per_token: u64 = 1 << 30;

/// Ceiling on one layer's declared fixed per-slot cost, one recurrent state
/// slab of a hybrid model. The Qwen3.5-2B GDN layer declares 1122304 bytes and
/// the 9B layer 2195456, so this sits far above either.
const max_fixed_state_bytes_per_layer: u64 = 1 << 30;

/// Ceiling on the whole model's fixed per-slot cost, every layer's slab
/// summed. Bounded for the same reason as `max_kv_bytes_per_token`.
const max_fixed_state_bytes: u64 = 1 << 32;

/// Ceiling on a declared per-layer weight byte count. A model at this ceiling on
/// every one of `limits.max_layers` layers still sums far inside a `u64`. The
/// head weights take the same ceiling, for the same reason.
const max_layer_weight_bytes: u64 = 1 << 40;

/// Bytes the widest blob refusal occupies. The test `formatBlobRefusal: the
/// sessions fit form, and the widest figures` proves that the widest figures fit.
pub const blob_refusal_text_max: usize = 512;

/// Bytes the widest RAM refusal occupies. `formatRamRefusal: the widest text
/// and both advice forms` proves that the widest figures fit.
pub const ram_refusal_text_max: usize = 512;

/// Bytes the longest span form occupies. A caller sizes its stack buffer with
/// it, and `formatLayerSplit: the three span forms` proves the ceiling holds.
pub const layer_split_text_max: usize = 64;

/// The frames the ledger line writes. They are named, so the ceiling below
/// reads the same literals the formatter writes. The RAM part has two forms:
/// the spend against the usable RAM, or the spend alone when the check is off.
const ledger_frame = "ledger: {d} slots x {d} window | gpu {d} MiB weights + {d} MiB kv" ++
    " | host {d} MiB kv + {d} MiB blobs + {d} MiB prefix + {d} MiB stages";

const ledger_ram_checked = " | ram {d} of {d} MiB usable";
const ledger_ram_unchecked = " | ram {d} MiB, not checked";

/// Bytes the widest ledger line occupies. The frame lengths count the `{d}`
/// placeholders too, and the byte fields print in MiB rather than whole, so
/// this figure over-estimates. An over-estimate is sound for a ceiling, and
/// `formatLedger: the widest ledger` proves the ceiling holds for both forms.
pub const layer_split_ledger_text_max: usize = ledger_frame.len +
    @max(ledger_ram_checked.len, ledger_ram_unchecked.len) +
    2 * std.fmt.comptimePrint("{d}", .{std.math.maxInt(u32)}).len +
    8 * std.fmt.comptimePrint("{d}", .{std.math.maxInt(u64)}).len;

/// Bytes the widest VRAM refusal occupies. `formatVramRefusal: the three
/// families` proves that the widest figures fit.
pub const vram_refusal_text_max: usize = 512;

// -- types ------------------------------------------------------------------

/// Device state and the static ceilings the planner reads. The static ceilings
/// default to the engine limits. The two default windows have no default
/// value, because each is a per-model value that the caller derives.
pub const Caps = struct {
    vram_free: u64,
    weights: u64,
    logits: u64,
    /// Device bytes of the activation scratch. Named, so the budget and the preflight count it.
    scratch: u64 = 0,
    /// The model ceiling the caller resolved through `limits.contextCeiling`.
    model_ctx: u32,
    /// The window of a request that names none, outside the pinned money mode.
    /// The caller fills it from `defaultWindow` at the serve budget.
    serve_default: u32,
    /// The money window stops here when the request names no window. The
    /// caller fills it from `defaultWindow` at the `run` budget.
    default_ceil: u32,
    min_window: u32 = limits.min_kv_window,
    max_model_ctx: u32 = limits.max_model_ctx,
    max_batch: u32 = limits.max_batch,
    max_vram_slots: u32 = limits.max_vram_slots,
};

/// What one KV slot costs. `costModel` derives it from the model dimensions, so
/// no call site rebuilds the arithmetic.
pub const KvCostModel = struct {
    kv_bytes_per_token: u64,
    fixed_state_bytes_per_slot: u64 = 0,
};

/// The model dimensions the KV price reads, copied out of the model card. The
/// planner takes this value and not the card, so the core never imports the
/// loader.
pub const KvShape = struct {
    n_layers: u32,
    /// Elements one K row or one V row holds for one position: the KV heads
    /// times the head width.
    kv_row_elems: u32,
    /// The layer mix of a hybrid family. Null for a family whose every layer
    /// holds KV.
    recurrent: ?Recurrent = null,

    /// Layer `i` holds KV when `(i + 1) % full_attention_interval == 0`. Every
    /// other layer holds one recurrent state and one conv ring per slot.
    pub const Recurrent = struct {
        full_attention_interval: u32,
        conv_kernel: u32,
        state_size: u32,
        group_count: u32,
        time_step_rank: u32,
    };
};

/// The request. A null field is an absent flag: `window` is `--max-seq`, and
/// `slots` is `--vram-slots`. `batch` is the decode width. `n_logical` is the
/// conversation count a serving caller tracks. A bench and `run` state none,
/// and the RAM budget then funds the blob ledger alone.
pub const Request = struct {
    x_bytes: ?u64 = null,
    y_bytes: ?u64 = null,
    window: ?u32 = null,
    slots: ?u32 = null,
    batch: u32 = 1,
    n_logical: ?u32 = null,
};

/// Named byte reservations plus the structure they buy. The reservations stay
/// separate so a later split of one of them cannot disturb the KV arithmetic.
pub const KvPlan = struct {
    pub const Vram = struct {
        weights: u64,
        kv: u64,
        logits: u64,
        scratch: u64,
    };

    pub const Ram = struct {
        blobs: u64,
        prefix: u64,
    };

    vram: Vram,
    ram: Ram,
    n_slots: u32,
    /// Decode rows per tick, at most `n_slots` (a row holds a slot) and `limits.max_batch`.
    n_streams: u32,
    window: u32,
    /// True when the slot ceiling cut the count the money bought. The caller prints the clamp.
    slots_clamped: bool = false,
};

/// The slot count a request resolves to, plus whether the ceiling cut it. The
/// witness comes out of the one division that derived the count, so no caller
/// repeats that arithmetic to learn the ceiling bound.
const ResolvedSlots = struct {
    n_slots: u32,
    clamped: bool = false,
};

/// The two default windows of one model: `run` at `limits.run_session_kv_bytes`
/// and `serve` at `limits.serve_session_kv_bytes`.
pub const DefaultWindows = struct {
    run: u32,
    serve: u32,
};

/// What one slot costs under a request: the window the planner resolves, and
/// the bytes a slot occupies at that window.
pub const SlotQuote = struct {
    window: u32,
    per_slot_bytes: u64,
};

/// What one layer of one slot costs. Two kinds of layer exist: an attention
/// layer charges per token, and a recurrent layer holds a fixed slab whatever
/// the position. A layer of either kind states the other field as zero.
pub const LayerKvCost = struct {
    /// KV bytes one token adds to this layer. Zero on a stateless or recurrent
    /// layer.
    kv_bytes_per_token: u32 = 0,
    /// Fixed per-slot bytes this layer holds at any position, the recurrent
    /// state slab. Zero on an attention layer.
    fixed_bytes_per_slot: u32 = 0,
};

/// One entry per layer, held as a bounded array rather than a slice, so no
/// ownership crosses the planner boundary and the value stays copyable. Every
/// narrowing to `u32` happens inside `set`, behind the per-layer ceilings.
pub const KvCostVector = struct {
    n_layers: u32,
    entries: [limits.max_layers]LayerKvCost = @splat(.{}),

    /// The vector of a family that charges every layer the same. A uniform
    /// vector states one price, and the planner still walks it by span.
    pub fn uniform(n_layers: u32, kv_bytes_per_token: u64) KvCostVector {
        std.debug.assert(n_layers >= 1);
        std.debug.assert(n_layers <= limits.max_layers);

        var self: KvCostVector = .{ .n_layers = n_layers };
        var i: u32 = 0;
        while (i < n_layers) : (i += 1) {
            self.set(i, kv_bytes_per_token, 0);
        }

        return self;
    }

    /// State one layer's price. This is the only narrowing site in the file, so
    /// a declared cost past a `u32` fails here rather than downstream.
    pub fn set(self: *KvCostVector, layer: u32, per_token: u64, fixed: u64) void {
        std.debug.assert(self.n_layers <= limits.max_layers);
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(per_token <= max_kv_bytes_per_token);
        std.debug.assert(fixed <= max_fixed_state_bytes_per_layer);

        self.entries[layer] = .{
            .kv_bytes_per_token = @intCast(per_token),
            .fixed_bytes_per_slot = @intCast(fixed),
        };
    }

    /// The live prefix. The tail past `n_layers` stays zero, and no caller
    /// reads it.
    pub fn layers(self: *const KvCostVector) []const LayerKvCost {
        std.debug.assert(self.n_layers <= limits.max_layers);

        return self.entries[0..self.n_layers];
    }

    /// The whole-model cost model: the two columns summed. It is the assert
    /// gate as well as the sum, because the ceilings that bound every product
    /// a caller forms from a vector fire here.
    pub fn total(self: *const KvCostVector) KvCostModel {
        std.debug.assert(self.n_layers >= 1);
        std.debug.assert(self.n_layers <= limits.max_layers);

        var sum_kv: u64 = 0;
        var sum_fixed: u64 = 0;
        for (self.layers()) |entry| {
            // Split, so a failing entry names the column it broke.
            std.debug.assert(entry.kv_bytes_per_token <= max_kv_bytes_per_token);
            std.debug.assert(entry.fixed_bytes_per_slot <= max_fixed_state_bytes_per_layer);

            sum_kv += @as(u64, entry.kv_bytes_per_token);
            sum_fixed += @as(u64, entry.fixed_bytes_per_slot);
        }

        std.debug.assert(sum_kv <= max_kv_bytes_per_token);
        std.debug.assert(sum_fixed <= max_fixed_state_bytes);
        return .{ .kv_bytes_per_token = sum_kv, .fixed_state_bytes_per_slot = sum_fixed };
    }
};

/// The resolved KV structure a layer split prices, plus what each layer charges.
/// A family states one entry per layer, and a family that charges every layer
/// the same states a uniform vector, so a split by layer count is exact by
/// summation and needs no rounding.
pub const KvStructure = struct {
    n_slots: u32,
    window: u32,
    costs: KvCostVector,
};

/// One layer split. The host layers form the prefix `[0..host_layers)` and the
/// GPU layers form the suffix `[host_layers..n_layers)`, so the residual stream
/// crosses the host-device boundary once per token. `host_layers == n_layers` is the
/// pure CPU runner and `host_layers == 0` is the whole-model GPU runner.
pub const LayerSplit = struct {
    /// The host bytes one split implies, all of it anonymous memory. The
    /// vector funds nothing from Y. `planLayerSplit` refuses a split whose
    /// vector passes the usable RAM, and the stages price zero for a caller
    /// without residency.
    pub const Ram = struct {
        /// The KV plan splits Y before a layer split exists, so this planner carries both shares.
        blobs: u64,
        prefix: u64,
        /// `kv_host` under a second name, so the host ledger reads whole from
        /// one vector.
        host_kv: u64,
        /// Both swap stages at high water: one chunk of `limits.swap_stage_chunk_bytes` each, or
        /// one slot span when smaller. Zero without a residency pool.
        swap_stages: u64,

        /// The anonymous host bytes the split spends: the figure the RAM
        /// check compares with the usable RAM. The sum saturates, because Y is
        /// an operator figure, and a saturated spend still refuses.
        pub fn spend(self: Ram) u64 {
            return self.blobs +| self.prefix +| self.host_kv +| self.swap_stages;
        }
    };

    host_layers: u32,
    n_layers: u32,
    /// Weight bytes of the GPU suffix, resident. Any GPU layer adds the final norm and lm_head.
    weight_resident: u64,
    kv_vram: u64,
    kv_host: u64,
    ram: Ram,
    /// The usable share of free RAM that funded `ram`. Null from `planLayerSplit` or unknown RAM.
    ram_usable: ?u64,
    logits: u64,
    scratch: u64,
    n_slots: u32,
    window: u32,

    /// The device bytes this split reserves. A pure CPU split holds no
    /// device context, so it reserves none, exactly as `pinned` does.
    pub fn spend(self: LayerSplit) u64 {
        if (self.n_layers <= self.host_layers) return 0;
        return self.weight_resident + self.kv_vram + self.logits + self.scratch;
    }

    /// The split with the host vector of its RAM quote. The planner checks
    /// no RAM, and every split quotes its RAM through `quoteRam`, so the quote
    /// carries the funded blobs.
    pub fn priced(self: LayerSplit, q: RamQuote) LayerSplit {
        // The VRAM plan checks no RAM, and the quote prices the
        // same split, so every term but the blobs agrees.
        std.debug.assert(self.ram_usable == null);
        std.debug.assert(q.ram.host_kv == self.kv_host);
        std.debug.assert(q.ram.prefix == self.ram.prefix);
        std.debug.assert(q.ram.swap_stages == self.ram.swap_stages);

        // Only a derived blob share differs, and an unset flag carries none.
        if (q.ram.blobs != self.ram.blobs) std.debug.assert(self.ram.blobs == 0);

        var plan = self;
        plan.ram = q.ram;
        plan.ram_usable = q.usable;
        return plan;
    }
};

/// What a pinned split costs, against what the device budget allows.
pub const LayerSplitQuote = struct {
    spend: u64,
    usable: u64,
};

/// The value of `--gpu-layers` on every command that takes it. A count pins the
/// GPU layers, counted from the top. `auto` gives the layer split planner a null
/// pin, so the widest GPU suffix the live budget funds wins.
pub const GpuLayers = union(enum) {
    auto,
    count: u32,

    /// The exact lower-case word `auto`, or a decimal `u32` count. Any other
    /// text is null, and the parse site names the flag in its refusal.
    pub fn parse(raw: []const u8) ?GpuLayers {
        if (std.mem.eql(u8, raw, "auto")) return .auto;
        const n = std.fmt.parseInt(u32, raw, 10) catch return null;
        return .{ .count = n };
    }

    /// The flag parser of `cli/args.zig` reads these three declarations: the
    /// parse, the value placeholder of the help, and the words of the refusal.
    pub const parseFlag = parse;
    pub const flag_value = "N|auto";
    pub const flag_choices = "a count or auto";

    /// The pin the planner takes. Null asks the planner for the auto split.
    pub fn pin(self: GpuLayers) ?u32 {
        return switch (self) {
            .auto => null,
            .count => |n| n,
        };
    }

    /// True only for `--gpu-layers 0`, the all-host pin that takes the CPU
    /// placement with no device context.
    pub fn isZeroCount(self: GpuLayers) bool {
        return switch (self) {
            .auto => false,
            .count => |n| n == 0,
        };
    }
};

/// The host side of one layer split request: the vector terms the split does
/// not set. The free RAM is not part of it, because only `quoteRam` checks it.
pub const HostRequest = struct {
    /// The Y split the KV plan already ran. It goes into the host vector, so
    /// one value states the whole host ledger.
    ram_split: KvPlan.Ram,
    /// True for a caller with a residency pool, the only one that grows the two swap stages.
    residency: bool,
};

/// The figures a RAM refusal names, at the split the planner took. The byte
/// fields hold whole bytes, and the text prints them in MiB.
pub const RamRefusal = struct {
    /// The anonymous host bytes the placement needs: the sum of the four tiers.
    spend: u64,
    /// `MemAvailable` at boot, and its usable share.
    free: u64,
    usable: u64,
    host_kv: u64,
    swap_stages: u64,
    blobs: u64,
    prefix: u64,
    n_slots: u32,
    window: u32,
    gpu_layers: u32,
    n_layers: u32,
    residency: bool,
    /// The widest window the usable RAM funds at the same split and the same
    /// Y split. Null when no window at or over the window floor fits.
    fit: ?u32,
};

/// The blob tier of one boot. The three cases are different states, so one
/// integer must not carry them.
pub const BlobTier = union(enum) {
    /// No session can be evicted, so no blob exists.
    absent,
    /// `--kv-ram` is unset and the free RAM is unknown, so nothing prices the
    /// need and the ledger enforces nothing.
    unenforced,
    /// The ledger enforces this many bytes. Zero bytes is a real budget that
    /// refuses every eviction.
    bounded: u64,

    /// The ledger budget of the residency pool. An absent tier enforces zero,
    /// because no eviction is reachable. Null enforces nothing.
    pub fn budget(self: BlobTier) ?u64 {
        return switch (self) {
            .absent => 0,
            .unenforced => null,
            .bounded => |bytes| bytes,
        };
    }
};

/// The host RAM quote of one serve placement at one split: the vector, its tier, and
/// the blob figures the boot lines print. The byte fields hold whole bytes.
pub const RamQuote = struct {
    ram: LayerSplit.Ram,
    /// The usable share of the free RAM. Null when the free RAM is unknown.
    usable: ?u64,
    tier: BlobTier,
    /// One slot of the whole model at the window: the largest blob one
    /// evicted session holds.
    blob: u64,
    /// The provable blob need, `(evictable + 1) * blob`, or zero when no session can be evicted.
    /// The extra blob exists because a swap holds the evicted and the restored blob at once.
    need: u64,
    /// The sessions the slots cannot hold.
    evictable: u32,
};

/// The figures a blob refusal names. The byte fields hold whole bytes, and the
/// text prints them in MiB.
pub const BlobRefusal = struct {
    need: u64,
    /// The usable RAM that the rest of the vector leaves for blobs.
    left: u64,
    blob: u64,
    n_logical: u32,
    n_slots: u32,
    window: u32,
    /// The widest window whose whole vector, blobs included, fits the usable
    /// RAM. Null when no window at or over the window floor fits.
    fit_window: ?u32,
    /// The most logical sessions whose blobs fit beside the rest. Null when none fit.
    fit_sessions: ?u32,
};

/// One piece of a chunk walk: `len` bytes at byte `offset` of the walked blob.
pub const Chunk = struct {
    offset: u64,
    len: u64,
};

/// The walk over `total` bytes in chunks of at most `chunk` bytes, in order.
/// The chunks tile the blob once, so a copy through one stage of `chunk` bytes
/// moves every byte exactly once. The walk takes `count(total, chunk)` steps.
pub const ChunkWalk = struct {
    total: u64,
    chunk: u64,
    offset: u64 = 0,

    pub fn init(total: u64, chunk: u64) ChunkWalk {
        std.debug.assert(chunk >= 1);

        return .{ .total = total, .chunk = chunk };
    }

    /// The steps a walk takes: `total / chunk`, rounded up.
    pub fn count(total: u64, chunk: u64) u64 {
        std.debug.assert(chunk >= 1);

        return std.math.divCeil(u64, total, chunk) catch unreachable;
    }

    pub fn next(self: *ChunkWalk) ?Chunk {
        std.debug.assert(self.offset <= self.total);

        if (self.offset == self.total) return null;

        const len = @min(self.chunk, self.total - self.offset);
        const piece: Chunk = .{ .offset = self.offset, .len = len };
        self.offset += len;
        std.debug.assert(piece.len >= 1);
        std.debug.assert(piece.len <= self.chunk);
        return piece;
    }
};

/// The figures a VRAM refusal names. The byte fields hold whole bytes, and the
/// text prints them in MiB.
pub const VramRefusal = struct {
    /// Device bytes the plan needs in all.
    planned: u64,
    /// The live device-local budget. The text also names its usable share.
    available: u64,
    weights: u64,
    kv: u64,
    /// The recurrent state of a hybrid family. Null for a family whose every
    /// layer holds KV.
    recurrent_state: ?u64 = null,
    n_slots: u32,
    window: u32,
    /// The widest window the budget funds. Null when no window fits.
    fit: ?u32,
    /// True for a runner that holds the suffix of a split model. The weights
    /// then read as suffix weights, and the advice names `--gpu-layers`.
    suffix: bool = false,
};

// -- api --------------------------------------------------------------------

/// Plan the KV economy. A named window past the model ceiling is a refusal, never a clamp.
/// A budget that cannot host one decode row per tick is a refusal, never a narrower batch.
pub fn kvPlan(
    caps: Caps,
    cost: KvCostModel,
    req: Request,
) error{ KvVramBudgetTooSmall, WindowExceedsCeiling }!KvPlan {
    // Check the caps and the request.
    std.debug.assert(req.batch >= 1);
    std.debug.assert(req.batch <= limits.max_batch);
    std.debug.assert(req.batch <= caps.max_batch);
    // A decode row needs a slot to sit in. Both bounds also keep the byte products inside a `u64`.
    std.debug.assert(caps.max_batch <= caps.max_vram_slots);
    std.debug.assert(caps.max_vram_slots <= limits.max_vram_slots);
    std.debug.assert(caps.max_model_ctx <= limits.max_model_ctx);
    std.debug.assert(caps.model_ctx >= 1);
    // The serve budget sits under the run budget, so the serve default stays under the ceiling.
    std.debug.assert(caps.serve_default >= 1);
    std.debug.assert(caps.serve_default <= caps.default_ceil);
    std.debug.assert(cost.kv_bytes_per_token <= max_kv_bytes_per_token);
    std.debug.assert(cost.fixed_state_bytes_per_slot <= max_fixed_state_bytes);

    // A zero window zeroes a slot price and turns a policy question into a
    // division refusal, so the planner refuses that corner outright.
    if (req.window) |w| std.debug.assert(w >= 1);

    if (req.slots) |s| {
        std.debug.assert(s >= 1);
        std.debug.assert(s >= req.batch);
        std.debug.assert(s <= caps.max_vram_slots);
    }

    if (req.n_logical) |n| {
        std.debug.assert(n >= 1);
        std.debug.assert(n <= limits.max_logical_sessions);
    }

    // Resolve the window, then the slots that fund it.
    const window = try resolveWindow(caps, cost, req);
    const slots = resolveSlots(caps, cost, req, window) orelse
        return error.KvVramBudgetTooSmall;
    const n_slots = slots.n_slots;

    const plan: KvPlan = .{
        .vram = .{
            .weights = caps.weights,
            .kv = @as(u64, n_slots) * perSlotBytes(cost, window),
            .logits = caps.logits,
            .scratch = caps.scratch,
        },
        .ram = splitRam(cost, req, n_slots, window),
        .n_slots = n_slots,
        .n_streams = req.batch,
        .window = window,
        .slots_clamped = slots.clamped,
    };

    // Check the plan against the request.
    std.debug.assert(plan.window <= windowCeil(caps));
    std.debug.assert(plan.n_streams <= plan.n_slots);
    if (req.window) |w| {
        // Outside the pinned money mode, a named window is the plan's window,
        // exactly.
        if (req.x_bytes == null or req.slots == null) std.debug.assert(plan.window == w);
        std.debug.assert(plan.window <= w);
    } else {
        // An unnamed window never passes the default ceiling, in any mode.
        std.debug.assert(plan.window <= caps.default_ceil);
    }

    std.debug.assert(plan.n_slots <= caps.max_vram_slots);
    if (plan.slots_clamped) {
        // Only the mode A money path can clamp, and it lands exactly on the
        // ceiling that cut it.
        std.debug.assert(req.x_bytes != null);
        std.debug.assert(req.slots == null);
        std.debug.assert(plan.n_slots == caps.max_vram_slots);
    }

    // Check the budget.
    if (req.x_bytes) |x| {
        // Mode A never spends more KV bytes than the budget.
        std.debug.assert(plan.vram.kv <= x);
        // Mode A slots reach the decode width.
        std.debug.assert(plan.n_slots >= req.batch);

        // The derived budget is what remains, so its plan always passes the
        // preflight.
        if (x == derivedX(caps)) std.debug.assert(planFits(caps, plan));
    } else {
        // Mode B reads no budget, so the structure passes through unchanged.
        std.debug.assert(plan.n_slots == (req.slots orelse req.batch));
    }

    // Check the RAM split.
    if (req.y_bytes != null and req.n_logical != null) {
        // The split spends Y whole. The prefix cache takes only what the blobs cannot reach.
        std.debug.assert(plan.ram.blobs + plan.ram.prefix == req.y_bytes.?);

        if (plan.ram.prefix > 0) {
            const need = blobNeed(req.n_logical.?, plan.n_slots, perSlotBytes(cost, plan.window));
            std.debug.assert(plan.ram.blobs == need);
        }
    } else {
        // Without both fields the split does not run: Y funds the ledger alone.
        std.debug.assert(plan.ram.blobs == (req.y_bytes orelse 0));
        std.debug.assert(plan.ram.prefix == 0);
    }

    return plan;
}

/// A plan that states a shape and reserves no device bytes, for a CPU runner or a bench that fixes
/// a shape. A device runner must come through `kvPlan`, whose reservations its preflight checks.
pub fn pinned(n_slots: u32, window: u32) KvPlan {
    std.debug.assert(n_slots >= 1);
    std.debug.assert(n_slots <= limits.max_vram_slots);
    std.debug.assert(window >= 1);
    std.debug.assert(window <= limits.max_model_ctx);

    return .{
        .vram = .{ .weights = 0, .kv = 0, .logits = 0, .scratch = 0 },
        .ram = .{ .blobs = 0, .prefix = 0 },
        .n_slots = n_slots,
        .n_streams = @min(n_slots, limits.max_batch),
        .window = window,
        // A pinned shape buys nothing, so no ceiling can cut it.
        .slots_clamped = false,
    };
}

/// Whether a plan's VRAM reservations fit the usable share of the device
/// budget. This is the preflight, and it reuses the one arithmetic in `limits`.
pub fn planFits(caps: Caps, plan: KvPlan) bool {
    // Saturation refuses an absurd reservation rather than wrapping past it.
    const planned = plan.vram.weights +| plan.vram.kv +| plan.vram.logits +| plan.vram.scratch;
    return limits.vramPlanFits(planned, caps.vram_free);
}

/// The price the planner charges for one slot under `req`. A refusal message
/// reads it, so no call site rebuilds the arithmetic that refused it. It
/// refuses the window exactly where `kvPlan` does.
pub fn quote(caps: Caps, cost: KvCostModel, req: Request) error{WindowExceedsCeiling}!SlotQuote {
    const window = try resolveWindow(caps, cost, req);
    return .{ .window = window, .per_slot_bytes = perSlotBytes(cost, window) };
}

/// What one KV slot costs, from the model dimensions, before any runner exists. A uniform family
/// charges one K row and one V row per layer per token. A hybrid family charges them on its
/// full-attention layers alone. Its recurrent state stays f32 under any KV dtype.
pub fn costModel(shape: KvShape, kv_dtype: dtype.KvDtype) KvCostModel {
    std.debug.assert(shape.n_layers >= 1);
    std.debug.assert(shape.n_layers <= limits.max_layers);
    std.debug.assert(shape.kv_row_elems >= 1);

    const row_pair: u64 = 2 * kv_dtype.rowBytes(shape.kv_row_elems);
    const cost: KvCostModel = if (shape.recurrent) |r| blk: {
        std.debug.assert(r.full_attention_interval >= 1);
        std.debug.assert(r.conv_kernel >= 1);

        // Layer `i` holds KV when `i + 1` is a multiple of the interval, so the
        // stack holds the floor quotient of such layers.
        const n_full: u64 = shape.n_layers / r.full_attention_interval;
        const n_recurrent: u64 = shape.n_layers - n_full;
        const key_dim: u64 = @as(u64, r.group_count) * r.state_size;
        const value_dim: u64 = @as(u64, r.time_step_rank) * r.state_size;
        const state_elems: u64 = @as(u64, r.time_step_rank) * r.state_size * r.state_size;
        const conv_elems: u64 = @as(u64, r.conv_kernel - 1) * (2 * key_dim + value_dim);
        break :blk .{
            .kv_bytes_per_token = n_full * row_pair,
            .fixed_state_bytes_per_slot = n_recurrent * (state_elems + conv_elems) * @sizeOf(f32),
        };
    } else .{ .kv_bytes_per_token = @as(u64, shape.n_layers) * row_pair };

    std.debug.assert(cost.kv_bytes_per_token <= max_kv_bytes_per_token);
    std.debug.assert(cost.fixed_state_bytes_per_slot <= max_fixed_state_bytes);
    if (shape.recurrent == null) std.debug.assert(cost.fixed_state_bytes_per_slot == 0);

    return cost;
}

/// The window of a session that no flag names: the session byte budget in whole granules,
/// floored at `limits.min_kv_window`, and capped by the ceiling. Fixed state stays outside.
/// A family with no per-token cost gets the whole ceiling.
pub fn defaultWindow(cost: KvCostModel, ceiling: u32, session_bytes: u64) u32 {
    std.debug.assert(ceiling >= 1);
    std.debug.assert(ceiling <= limits.max_model_ctx);
    std.debug.assert(cost.kv_bytes_per_token <= max_kv_bytes_per_token);

    if (cost.kv_bytes_per_token == 0) return ceiling;

    const whole = limits.floorToGranule(session_bytes / cost.kv_bytes_per_token);
    const window: u32 = @intCast(
        @min(@as(u64, ceiling), @max(@as(u64, limits.min_kv_window), whole)),
    );
    std.debug.assert(window >= 1);
    std.debug.assert(window <= ceiling);
    // A budget that funds the floor also funds the window, so a default never
    // reserves more than its byte budget.
    if (@as(u64, limits.min_kv_window) * cost.kv_bytes_per_token <= session_bytes) {
        std.debug.assert(@as(u64, window) * cost.kv_bytes_per_token <= session_bytes);
    }

    return window;
}

/// Both default windows of one model under its ceiling.
pub fn defaultWindows(cost: KvCostModel, ceiling: u32) DefaultWindows {
    const windows: DefaultWindows = .{
        .run = defaultWindow(cost, ceiling, limits.run_session_kv_bytes),
        .serve = defaultWindow(cost, ceiling, limits.serve_session_kv_bytes),
    };
    // The serve budget sits under the run budget, and `defaultWindow` never
    // shrinks as its budget grows.
    std.debug.assert(windows.serve <= windows.run);
    std.debug.assert(windows.run <= ceiling);
    return windows;
}

/// The widest window, at most `window`, that the derived budget funds for `n_slots` slots. The
/// result floors at `min_window`, so a budget below the floor still gets it, and the preflight
/// then refuses it.
pub fn fitWindow(caps: Caps, cost: KvCostModel, n_slots: u32, window: u32) u32 {
    std.debug.assert(n_slots >= 1);
    std.debug.assert(n_slots <= caps.max_vram_slots);
    std.debug.assert(window >= 1);
    std.debug.assert(window <= windowCeil(caps));

    // A model with no per-token cost has no window to buy, so the window stands.
    if (cost.kv_bytes_per_token == 0) return window;

    const x = derivedX(caps);
    const fit = moneyWindow(caps, cost, x, n_slots, window);
    std.debug.assert(fit <= window);
    // A budget that funds the floor funds the fit, so its plan passes the
    // preflight.
    if (@as(u64, n_slots) * perSlotBytes(cost, @min(window, caps.min_window)) <= x) {
        std.debug.assert(@as(u64, n_slots) * perSlotBytes(cost, fit) <= x);
    }

    return fit;
}

/// The widest window under a refused `window` whose plan for `n_slots` slots fits the usable
/// share of `available` beside `reserved`. Null means that no window in
/// `min_kv_window..window-1` fits, or that the model has no per-token cost.
pub fn largestFittingWindow(
    available: u64,
    reserved: u64,
    cost: KvCostModel,
    n_slots: u32,
    window: u32,
) ?u32 {
    std.debug.assert(n_slots >= 1);
    std.debug.assert(n_slots <= limits.max_vram_slots);
    std.debug.assert(window >= 1);
    std.debug.assert(window <= limits.max_model_ctx);
    std.debug.assert(cost.kv_bytes_per_token <= max_kv_bytes_per_token);
    std.debug.assert(cost.fixed_state_bytes_per_slot <= max_fixed_state_bytes);

    if (cost.kv_bytes_per_token == 0) return null;
    if (window <= limits.min_kv_window) return null;

    // `w` fits when `n_slots * (w * kv + fixed) <= usable - reserved`, and the
    // floor divisions below solve that for the largest `w`.
    const share = (limits.vramUsable(available) -| reserved) / n_slots;
    const affordable = (share -| cost.fixed_state_bytes_per_slot) / cost.kv_bytes_per_token;
    const widest = @min(affordable, @as(u64, window - 1));
    if (widest < limits.min_kv_window) return null;

    const fit: u32 = @intCast(widest);
    std.debug.assert(fit >= limits.min_kv_window);
    std.debug.assert(fit < window);
    std.debug.assert(windowFits(available, reserved, cost, n_slots, fit));
    if (fit + 1 < window) {
        std.debug.assert(!windowFits(available, reserved, cost, n_slots, fit + 1));
    }

    return fit;
}

/// The model ceiling: `caps.model_ctx` under the engine bound. A named window
/// past it is a refusal.
pub fn windowCeil(caps: Caps) u32 {
    return @min(caps.model_ctx, caps.max_model_ctx);
}

/// The default KV VRAM budget: usable VRAM less the weights, the logits, and the scratch. It
/// saturates, so an over-subscribed device yields zero and the planner refuses.
pub fn derivedX(caps: Caps) u64 {
    return usable(caps) -| caps.weights -| caps.logits -| caps.scratch;
}

/// Bytes one slot holds over the layer span `[from..to)`: the window price of every token
/// position, plus the fixed state. The device suffix form is the blob half that one swap moves
/// (`llama_hybrid.deviceSwapBytes` at `n_rows == window`).
pub fn slotSpanBytes(kv: KvStructure, from: u32, to: u32) u64 {
    std.debug.assert(from <= to);
    std.debug.assert(to <= kv.costs.n_layers);

    var span: u64 = 0;
    for (kv.costs.entries[from..to]) |entry| {
        span += @as(u64, kv.window) * @as(u64, entry.kv_bytes_per_token) +
            @as(u64, entry.fixed_bytes_per_slot);
    }

    return span;
}

/// Bytes one swap stage holds for a device span of `span` bytes: one chunk, or
/// the whole span when the span is smaller. A swap walks the span in chunks,
/// so the stage never grows with the window.
pub fn stageBytes(span: u64) u64 {
    return stageBytesAt(limits.swap_stage_chunk_bytes, span);
}

/// The cap every swap stage holds to after each grow: one chunk, or the whole
/// span when the span is smaller, so its RAM never grows with the window. The
/// three runners that grow a stage share this one check.
pub fn assertStageCap(capacity: u64, chunk: u64, span: u64) void {
    std.debug.assert(chunk >= 1);
    std.debug.assert(capacity <= @min(chunk, span));
}

/// Clip `full`, the copies of one whole blob in blob order, to the blob byte range
/// `[offset, offset + len)`. The pieces tile the range once. `to_host` is true for a save.
/// `CopyT` is any record with `src_offset`, `dst_offset`, and `size`.
pub fn clipCopies(
    comptime CopyT: type,
    full: []const CopyT,
    to_host: bool,
    offset: u64,
    len: u64,
    out: []CopyT,
) []CopyT {
    std.debug.assert(len >= 1);
    std.debug.assert(out.len >= full.len);

    // Clip each copy to the range.
    const end = offset + len;
    var at: u64 = 0;
    var c: usize = 0;
    var clipped: u64 = 0;
    for (full) |copy| {
        const host_offset = if (to_host) copy.dst_offset else copy.src_offset;
        // Each copy starts where the last one ended.
        std.debug.assert(host_offset == at);
        const lo = @max(at, offset);
        const hi = @min(at + copy.size, end);
        if (lo < hi) {
            var piece = copy;
            piece.size = hi - lo;
            if (to_host) {
                piece.src_offset = copy.src_offset + (lo - at);
                piece.dst_offset = lo - offset;
            } else {
                piece.src_offset = lo - offset;
                piece.dst_offset = copy.dst_offset + (lo - at);
            }

            out[c] = piece;
            c += 1;
            clipped += piece.size;
        }

        at += copy.size;
    }

    // Check that the range lies inside the blob and the pieces cover it.
    std.debug.assert(end <= at);
    std.debug.assert(clipped == len);
    return out[0..c];
}

/// Plan the layer split against a resolved KV structure. A null `gpu_layers` follows the
/// money, and a pinned count that the budget cannot fund is a refusal, never a shrink. It checks
/// no RAM and never reads `caps.weights`. `head_bytes` is the final norm plus the lm_head.
pub fn planLayerSplit(
    caps: Caps,
    weights: []const u64,
    head_bytes: u64,
    kv: KvStructure,
    host: HostRequest,
    gpu_layers: ?u32,
) error{WeightsVramBudgetTooSmall}!LayerSplit {
    // Check the weights and the KV structure.
    std.debug.assert(weights.len >= 1);
    std.debug.assert(weights.len <= limits.max_layers);

    // These ceilings keep every byte product below inside a `u64`.
    for (weights) |w| {
        std.debug.assert(w <= max_layer_weight_bytes);
    }

    std.debug.assert(head_bytes <= max_layer_weight_bytes);
    std.debug.assert(kv.n_slots >= 1);
    std.debug.assert(kv.n_slots <= limits.max_vram_slots);
    std.debug.assert(kv.window >= 1);
    std.debug.assert(kv.window <= limits.max_model_ctx);
    // Two per-layer vectors describe one stack, and neither side trusts the other.
    std.debug.assert(kv.costs.n_layers == weights.len);
    // The whole-model sums fire the cost ceilings that bound every byte product below.
    const whole = kv.costs.total();

    // Choose the split, and refuse a pin that the budget cannot fund.
    const n_layers: u32 = @intCast(weights.len);
    if (gpu_layers) |g| std.debug.assert(g <= n_layers);

    const split = if (gpu_layers) |g| n_layers - g else autoSplit(caps, weights, head_bytes, kv);
    if (!layerSplitFits(caps, weights, head_bytes, kv, split)) {
        // Only a pinned count reaches here. The auto scan stops on a feasible
        // split, and the pure CPU split is feasible under any budget.
        std.debug.assert(gpu_layers != null);

        return error.WeightsVramBudgetTooSmall;
    }

    // Price both sides of the split. The pure CPU split carries no head bytes.
    const head_resident: u64 = if (split < n_layers) head_bytes else 0;
    const plan: LayerSplit = .{
        .host_layers = split,
        .n_layers = n_layers,
        .weight_resident = suffixWeights(weights, split) + head_resident,
        .kv_vram = kvVramBytes(kv, split),
        .kv_host = kvHostBytes(kv, split),
        .ram = ramAt(kv, split, n_layers, host),
        .ram_usable = null,
        .logits = caps.logits,
        .scratch = caps.scratch,
        .n_slots = kv.n_slots,
        .window = kv.window,
    };

    // Check the plan against the budget and the KV structure.
    std.debug.assert(plan.spend() <= usable(caps));
    // One contiguous split covers the model exactly once.
    std.debug.assert(plan.host_layers <= plan.n_layers);
    // The split moves KV cost between the two ledgers and creates none.
    std.debug.assert(plan.kv_vram + plan.kv_host ==
        @as(u64, kv.n_slots) * perSlotBytes(whole, kv.window));
    // Each degenerate split empties the ledger it leaves behind.
    if (plan.host_layers == 0) std.debug.assert(plan.kv_host == 0);
    if (plan.host_layers == plan.n_layers) std.debug.assert(plan.kv_vram == 0);

    // The KV structure and the caps reservations pass through unchanged.
    std.debug.assert(plan.n_slots == kv.n_slots);
    std.debug.assert(plan.window == kv.window);
    std.debug.assert(plan.logits == caps.logits);
    std.debug.assert(plan.scratch == caps.scratch);

    // Check the host vector. The KV plan ran the Y split, so the vector passes it through.
    std.debug.assert(plan.ram.blobs == host.ram_split.blobs);
    std.debug.assert(plan.ram.prefix == host.ram_split.prefix);
    // One host ledger under two names.
    std.debug.assert(plan.ram.host_kv == plan.kv_host);
    // With residency, the staging pair costs at most two chunks.
    if (host.residency) {
        std.debug.assert(
            plan.ram.swap_stages == 2 * stageBytes(slotSpanBytes(kv, split, n_layers)),
        );
        std.debug.assert(plan.ram.swap_stages <= 2 * limits.swap_stage_chunk_bytes);
        std.debug.assert(plan.ram.swap_stages * kv.n_slots <= 2 * plan.kv_vram);
    }

    if (!host.residency) std.debug.assert(plan.ram.swap_stages == 0);
    // The pure CPU placement holds no device suffix, so it stages nothing.
    if (plan.host_layers == plan.n_layers) std.debug.assert(plan.ram.swap_stages == 0);

    // Check the split against the pin or the auto rule.
    if (gpu_layers) |g| {
        // A pinned count is honored exactly, or refused above.
        std.debug.assert(plan.host_layers == n_layers - g);
    } else {
        // The auto split is feasible and no smaller split is, so the GPU
        // suffix is the widest the budget funds.
        std.debug.assert(layerSplitFits(caps, weights, head_bytes, kv, plan.host_layers));

        if (plan.host_layers > 0) {
            std.debug.assert(!layerSplitFits(caps, weights, head_bytes, kv, plan.host_layers - 1));
        }
    }

    return plan;
}

/// The price the planner charges for a pinned `gpu_layers` count. A refusal
/// message reads it, so no call site rebuilds the arithmetic that refused it.
pub fn quoteLayerSplit(
    caps: Caps,
    weights: []const u64,
    head_bytes: u64,
    kv: KvStructure,
    gpu_layers: u32,
) LayerSplitQuote {
    std.debug.assert(weights.len >= 1);
    std.debug.assert(weights.len <= limits.max_layers);
    // The same pair validation the plan runs, because a quote prices the same
    // stack the plan refused.
    std.debug.assert(kv.costs.n_layers == weights.len);

    const whole = kv.costs.total();
    const n_layers: u32 = @intCast(weights.len);
    std.debug.assert(gpu_layers <= n_layers);
    // A suffix never prices more than the whole model.
    std.debug.assert(slotSpanBytes(kv, n_layers - gpu_layers, n_layers) <=
        perSlotBytes(whole, kv.window));
    return .{
        .spend = deviceSpend(caps, weights, head_bytes, kv, n_layers - gpu_layers),
        .usable = usable(caps),
    };
}

/// Quote the host RAM of one serve placement at split `k`. An unset blob share funds
/// `(evictable + 1)` blobs or refuses `error.BlobRamBudgetTooSmall`. A vector past the usable RAM
/// refuses `error.HostRamBudgetTooSmall`. Unknown `ram_free` skips both.
pub fn quoteRam(
    kv: KvStructure,
    k: u32,
    host: HostRequest,
    n_logical: u32,
    kv_ram_named: bool,
    ram_free: ?u64,
) error{ HostRamBudgetTooSmall, BlobRamBudgetTooSmall }!RamQuote {
    return quoteRamAt(
        kv,
        k,
        host,
        n_logical,
        kv_ram_named,
        ram_free,
        limits.swap_stage_chunk_bytes,
    );
}

/// The RAM refusal of a vector at split `k`, so no call site rebuilds the refusing arithmetic.
/// The caller states `ram_free`, and the blob share is the one `host.ram_split` carries.
pub fn ramRefusal(kv: KvStructure, k: u32, host: HostRequest, ram_free: u64) RamRefusal {
    // Price the vector at the split.
    const n_layers = kv.costs.n_layers;
    std.debug.assert(k <= n_layers);
    std.debug.assert(kv.n_slots >= 1);
    std.debug.assert(kv.window >= 1);
    const whole = kv.costs.total();
    const free = ram_free;
    const usable_ram = limits.ramUsable(free);
    const ram = ramAt(kv, k, n_layers, host);
    // The host KV never prices more than every slot of the whole model.
    std.debug.assert(ram.host_kv <= @as(u64, kv.n_slots) * perSlotBytes(whole, kv.window));
    // Advise a window that fits. The advice keeps headroom, and the check stays exact.
    const fit = adviceFit(
        ramFitWindow(
            kv,
            k,
            0,
            ram.swap_stages +| ram.blobs +| ram.prefix,
            adviceUsable(usable_ram),
        ),
    );
    // A refused window passes the usable RAM, so the fit sits under it.
    if (ram.spend() > usable_ram) {
        if (fit) |f| std.debug.assert(f < kv.window);
    }

    return .{
        .spend = ram.spend(),
        .free = free,
        .usable = usable_ram,
        .host_kv = ram.host_kv,
        .swap_stages = ram.swap_stages,
        .blobs = ram.blobs,
        .prefix = ram.prefix,
        .n_slots = kv.n_slots,
        .window = kv.window,
        .gpu_layers = n_layers - k,
        .n_layers = n_layers,
        .residency = host.residency,
        .fit = fit,
    };
}

/// The blob refusal of an unset `--kv-ram` at split `k`: the need, what the
/// rest of the vector leaves, and both fits. The caller states the free RAM.
pub fn blobRefusal(
    kv: KvStructure,
    k: u32,
    host: HostRequest,
    n_logical: u32,
    ram_free: u64,
) BlobRefusal {
    return blobRefusalAt(kv, k, host, n_logical, ram_free, limits.swap_stage_chunk_bytes);
}

/// Write the refusal of an unset `--kv-ram` whose blob need the usable RAM
/// cannot fund. This function writes into the caller's buffer, so the planner
/// stays free of allocation and of I/O.
pub fn formatBlobRefusal(buf: []u8, r: BlobRefusal) []const u8 {
    std.debug.assert(buf.len >= blob_refusal_text_max);
    std.debug.assert(r.n_logical > r.n_slots);

    if (r.fit_window) |fit| std.debug.assert(fit < r.window);
    if (r.fit_sessions) |m| std.debug.assert(m < r.n_logical);

    const mib = 1024 * 1024;
    var w: std.Io.Writer = .fixed(buf);
    // The buffer holds the widest refusal, so no write can fail.
    w.print("ram budget: the blob tier needs {d} MiB of RAM, but the plan leaves only {d} MiB. ", .{
        r.need / mib, r.left / mib,
    }) catch unreachable;
    w.print(
        "One evicted session holds up to {d} MiB. " ++
            "The plan serves {d} sessions on {d} {s} at {d} positions. ",
        .{
            r.blob / mib, r.n_logical, r.n_slots, slotWord(r.n_slots), r.window,
        },
    ) catch unreachable;
    if (r.fit_window) |fit| {
        w.print("Use --max-seq {d} or less. Or use ", .{fit}) catch unreachable;
    } else {
        w.writeAll("No window fits these sessions. Use ") catch unreachable;
    }

    if (r.fit_sessions) |m| {
        w.print("--max-sessions {d} or less.", .{m}) catch unreachable;
    } else {
        w.print("--max-sessions {d}.", .{r.n_slots}) catch unreachable;
    }

    const text = w.buffered();
    std.debug.assert(text.len <= blob_refusal_text_max);
    return text;
}

/// Write the refusal of a placement that the usable RAM cannot hold. This
/// function writes into the caller's buffer, so the planner stays free of
/// allocation and of I/O.
pub fn formatRamRefusal(buf: []u8, r: RamRefusal) []const u8 {
    std.debug.assert(buf.len >= ram_refusal_text_max);
    std.debug.assert(r.n_slots >= 1);
    std.debug.assert(r.gpu_layers <= r.n_layers);

    if (r.fit) |fit| std.debug.assert(fit < r.window);

    const mib = 1024 * 1024;
    var w: std.Io.Writer = .fixed(buf);
    // The buffer holds the widest refusal, so no write can fail. Name the need and the plan.
    w.print(
        "ram budget: the placement needs {d} MiB of RAM, " ++
            "but only {d} MiB of the {d} MiB free are usable. ",
        .{
            r.spend / mib, r.usable / mib, r.free / mib,
        },
    ) catch unreachable;
    w.print(
        "It holds {d} MiB of host KV, {d} MiB of swap stages, {d} MiB of blobs, " ++
            "and {d} MiB of prefix cache. ",
        .{
            r.host_kv / mib, r.swap_stages / mib, r.blobs / mib, r.prefix / mib,
        },
    ) catch unreachable;
    w.print("The plan serves {d} {s} x {d} positions at gpu {d}/{d} layers. ", .{
        r.n_slots, slotWord(r.n_slots), r.window, r.gpu_layers, r.n_layers,
    }) catch unreachable;
    // Name the advice.
    if (r.fit) |fit| {
        // Only a hybrid placement names its layer count. The other placements take no
        // `--gpu-layers` value.
        if (r.gpu_layers > 0 and r.gpu_layers < r.n_layers) {
            w.print(
                "Use --gpu-layers {d} --max-seq {d} or less.",
                .{ r.gpu_layers, fit },
            ) catch unreachable;
        } else {
            w.print("Use --max-seq {d} or less.", .{fit}) catch unreachable;
        }
    } else {
        // Name only the levers that lower this spend.
        const fewer_slots = r.n_slots > 1;
        const lower_y = r.blobs > 0 or r.prefix > 0;
        w.writeAll("No window fits this split. Use ") catch unreachable;
        if (fewer_slots) {
            w.writeAll("fewer slots, ") catch unreachable;
        }

        if (lower_y) {
            w.writeAll("a lower --kv-ram, ") catch unreachable;
        }

        w.writeAll(
            if (fewer_slots or lower_y) "or a smaller model." else "a smaller model.",
        ) catch unreachable;
    }

    const text = w.buffered();
    std.debug.assert(text.len <= ram_refusal_text_max);
    return text;
}

/// Write the split in span form, such as `placement: 0..24 host, 24..32 resident (gpu 8/32
/// layers)`, into `buf`. A degenerate split drops the empty span.
pub fn formatLayerSplit(plan: LayerSplit, buf: []u8) []const u8 {
    std.debug.assert(buf.len >= layer_split_text_max);
    std.debug.assert(plan.host_layers <= plan.n_layers);

    const gpu = plan.n_layers - plan.host_layers;
    const text = if (plan.host_layers == 0)
        std.fmt.bufPrint(
            buf,
            "placement: 0..{d} resident (gpu {d}/{d} layers)",
            .{ plan.n_layers, gpu, plan.n_layers },
        )
    else if (plan.host_layers == plan.n_layers)
        std.fmt.bufPrint(
            buf,
            "placement: 0..{d} host (gpu {d}/{d} layers)",
            .{ plan.n_layers, gpu, plan.n_layers },
        )
    else
        std.fmt.bufPrint(
            buf,
            "placement: 0..{d} host, {d}..{d} resident (gpu {d}/{d} layers)",
            .{ plan.host_layers, plan.host_layers, plan.n_layers, gpu, plan.n_layers },
        );

    // The buffer holds the widest span form, so the write cannot fail.
    return text catch unreachable;
}

/// Write the split in ledger form, which names what the spans cost, into `buf`: `ledger: 2
/// slots x 512 window | gpu 3 MiB weights + 5 MiB kv | host 7 MiB kv + 11 MiB blobs + ... | ram
/// 48 of 1792 MiB usable`.
pub fn formatLedger(plan: LayerSplit, buf: []u8) []const u8 {
    std.debug.assert(buf.len >= layer_split_ledger_text_max);

    const mib = 1024 * 1024;
    var w: std.Io.Writer = .fixed(buf);
    // The buffer holds the widest ledger, so no write can fail.
    w.print(ledger_frame, .{
        plan.n_slots,
        plan.window,
        plan.weight_resident / mib,
        plan.kv_vram / mib,
        plan.ram.host_kv / mib,
        plan.ram.blobs / mib,
        plan.ram.prefix / mib,
        plan.ram.swap_stages / mib,
    }) catch unreachable;
    if (plan.ram_usable) |ram_usable| {
        const args = .{ plan.ram.spend() / mib, ram_usable / mib };
        w.print(ledger_ram_checked, args) catch unreachable;
    } else {
        w.print(ledger_ram_unchecked, .{plan.ram.spend() / mib}) catch unreachable;
    }

    return w.buffered();
}

/// Write the refusal of a KV plan that the free VRAM cannot hold. One text serves every runner
/// family, and the optional parts carry the differences.
pub fn formatVramRefusal(buf: []u8, r: VramRefusal) []const u8 {
    std.debug.assert(buf.len >= vram_refusal_text_max);
    std.debug.assert(r.n_slots >= 1);

    if (r.fit) |fit| std.debug.assert(fit < r.window);

    const mib = 1024 * 1024;
    var w: std.Io.Writer = .fixed(buf);
    // The buffer holds the widest refusal, so no write can fail. Name the need and the parts.
    w.print(
        "vram budget: the KV plan needs {d} MiB of VRAM, " ++
            "but only {d} MiB of the {d} MiB free are usable. It holds {d} MiB of {s}weights",
        .{
            r.planned / mib,
            limits.vramUsable(
                r.available,
            ) / mib,
            r.available / mib,
            r.weights / mib,
            if (r.suffix) "suffix " else "",
        },
    ) catch unreachable;
    if (r.recurrent_state) |state| {
        w.print(
            ", {d} MiB of KV, and {d} MiB of recurrent state",
            .{ r.kv / mib, state / mib },
        ) catch unreachable;
    } else {
        w.print(" and {d} MiB of KV", .{r.kv / mib}) catch unreachable;
    }

    w.print(" for {d} slots x {d} positions. ", .{ r.n_slots, r.window }) catch unreachable;

    // Name the advice.
    if (r.fit) |fit| {
        w.print(
            "Use --max-seq {d} or less, or {s}",
            .{ fit, if (r.suffix) "lower --gpu-layers" else "fewer sessions" },
        ) catch unreachable;
    } else if (r.suffix) {
        w.writeAll(
            "No window fits the suffix. Lower --gpu-layers, or use a smaller or Q4 model",
        ) catch unreachable;
    } else {
        w.print(
            "No window fits {d} slots. Use fewer sessions, or a smaller or Q4 model",
            .{r.n_slots},
        ) catch unreachable;
    }

    const text = w.buffered();
    std.debug.assert(text.len <= vram_refusal_text_max);
    return text;
}

// -- helpers ----------------------------------------------------------------

/// `quoteRam` at an explicit stage chunk. The shipped quote takes
/// `limits.swap_stage_chunk_bytes`, and a test replays the split proofs
/// at the chunk they state.
fn quoteRamAt(
    kv: KvStructure,
    k: u32,
    host: HostRequest,
    n_logical: u32,
    kv_ram_named: bool,
    ram_free: ?u64,
    chunk: u64,
) error{ HostRamBudgetTooSmall, BlobRamBudgetTooSmall }!RamQuote {
    // Check the split and the sessions.
    const n_layers = kv.costs.n_layers;
    std.debug.assert(k <= n_layers);
    std.debug.assert(kv.n_slots >= 1);
    std.debug.assert(kv.n_slots <= limits.max_vram_slots);
    std.debug.assert(kv.window >= 1);
    std.debug.assert(n_logical >= 1);
    std.debug.assert(n_logical <= limits.max_logical_sessions);
    // Only a residency pool can evict a session.
    if (n_logical > kv.n_slots) std.debug.assert(host.residency);
    // An unset flag names no Y, so the KV plan split none into blobs.
    if (!kv_ram_named) std.debug.assert(host.ram_split.blobs == 0);

    // Price the blob tier: named, derived from the RAM left, or unenforced.
    const evictable = n_logical -| kv.n_slots;
    const blob = slotSpanBytes(kv, 0, n_layers);
    const need = blobNeed(n_logical, kv.n_slots, blob);
    const ram_cap: ?u64 = if (ram_free) |free| limits.ramUsable(free) else null;
    var ram = ramAtChunk(kv, k, n_layers, host, chunk);
    const tier: BlobTier = tier: {
        if (evictable == 0) break :tier .absent;
        if (kv_ram_named) break :tier .{ .bounded = ram.blobs };

        const ram_usable = ram_cap orelse break :tier .unenforced;
        if (fixedRam(ram) +| need > ram_usable) return error.BlobRamBudgetTooSmall;

        ram.blobs = need;
        break :tier .{ .bounded = need };
    };

    // Refuse a vector past the usable RAM.
    if (ram_cap) |ram_usable| {
        if (ram.spend() > ram_usable) return error.HostRamBudgetTooSmall;
    }

    // Check the accepted quote. It spends inside the usable RAM.
    if (ram_cap) |ram_usable| std.debug.assert(ram.spend() <= ram_usable);
    // The stages need residency and a device suffix, and never pass two chunks.
    if (!host.residency) std.debug.assert(ram.swap_stages == 0);
    if (k == n_layers) std.debug.assert(ram.swap_stages == 0);

    std.debug.assert(ram.swap_stages <= 2 * chunk);
    switch (tier) {
        .absent => {
            std.debug.assert(evictable == 0);
            std.debug.assert(ram.blobs == 0);
        },
        .unenforced => {
            std.debug.assert(ram_cap == null);
            std.debug.assert(!kv_ram_named);
        },
        .bounded => |bytes| {
            std.debug.assert(evictable >= 1);
            std.debug.assert(bytes == ram.blobs);

            // A derived budget funds the need exactly. A named one funds at
            // most the need, because the Y split caps it there.
            if (!kv_ram_named) std.debug.assert(bytes == need);

            std.debug.assert(bytes <= need);
        },
    }

    return .{
        .ram = ram,
        .usable = ram_cap,
        .tier = tier,
        .blob = blob,
        .need = need,
        .evictable = evictable,
    };
}

/// `blobRefusal` at an explicit stage chunk, the twin of `quoteRamAt`.
fn blobRefusalAt(
    kv: KvStructure,
    k: u32,
    host: HostRequest,
    n_logical: u32,
    ram_free: u64,
    chunk: u64,
) BlobRefusal {
    // Check the split, then price the need against the RAM left.
    const n_layers = kv.costs.n_layers;
    std.debug.assert(host.ram_split.blobs == 0);
    std.debug.assert(k <= n_layers);
    std.debug.assert(n_logical > kv.n_slots);
    std.debug.assert(host.residency);
    const evictable = n_logical - kv.n_slots;
    const blob = slotSpanBytes(kv, 0, n_layers);
    // A refusal needs a blob to price, so the need is never zero here.
    std.debug.assert(blob >= 1);
    const need = blobNeed(n_logical, kv.n_slots, blob);
    const usable_ram = limits.ramUsable(ram_free);
    const ram = ramAtChunk(kv, k, n_layers, host, chunk);
    const fixed = fixedRam(ram);
    // The quote refused exactly here.
    std.debug.assert(fixed +| need > usable_ram);
    const left = usable_ram -| fixed;

    // The largest evictable count `E` with `(E + 1)` blobs in the RAM left is `left / blob - 1`,
    // and it serves `n_slots + E` sessions. Both fits keep the headroom.
    const advice_usable = adviceUsable(usable_ram);
    const funded = (advice_usable -| fixed) / blob;
    const fit_sessions: ?u32 = if (funded >= 2) kv.n_slots + @as(
        u32,
        @intCast(funded - 1),
    ) else null;

    if (fit_sessions) |m| std.debug.assert(m < n_logical);

    // Advise the window that fits. The blobs grow with the window, and the stage pair does not.
    const fit_window = adviceFit(
        ramFitWindow(kv, k, evictable + 1, ram.swap_stages +| ram.prefix, advice_usable),
    );
    if (fit_window) |f| std.debug.assert(f < kv.window);

    return .{
        .need = need,
        .left = left,
        .blob = blob,
        .n_logical = n_logical,
        .n_slots = kv.n_slots,
        .window = kv.window,
        .fit_window = fit_window,
        .fit_sessions = fit_sessions,
    };
}

/// The noun of a slot count in a refusal text.
fn slotWord(n: u32) []const u8 {
    return if (n == 1) "slot" else "slots";
}

/// The share of the device budget a plan can occupy. It is the share
/// `limits.vramPlanFits` reads, which the preflight calls.
fn usable(caps: Caps) u64 {
    return limits.vramUsable(caps.vram_free);
}

/// The default ceiling: the model ceiling under the `run` default. The money
/// window stops here when the request names no window.
fn implicitCeil(caps: Caps) u32 {
    return @min(windowCeil(caps), caps.default_ceil);
}

fn perSlotBytes(cost: KvCostModel, window: u32) u64 {
    return @as(u64, window) * cost.kv_bytes_per_token + cost.fixed_state_bytes_per_slot;
}

/// Whether `n_slots` slots at `window` fit beside `reserved`. Saturation
/// refuses an absurd reservation rather than wrapping past it.
fn windowFits(available: u64, reserved: u64, cost: KvCostModel, n_slots: u32, window: u32) bool {
    const kv = @as(u64, n_slots) *| perSlotBytes(cost, window);
    return limits.vramPlanFits(reserved +| kv, available);
}

/// The window a pinned slot count affords under budget X: each slot's share of X, minus the fixed
/// state, in whole tokens. Clamped to `[min_window, cap]`. `resolveSlots` refuses a floor that X
/// cannot fund.
fn moneyWindow(caps: Caps, cost: KvCostModel, x: u64, n_slots: u32, cap: u32) u32 {
    std.debug.assert(n_slots >= 1);
    std.debug.assert(cost.kv_bytes_per_token >= 1);
    std.debug.assert(cap >= 1);

    const share = x / n_slots;
    const derived = (share -| cost.fixed_state_bytes_per_slot) / cost.kv_bytes_per_token;
    const capped: u32 = @intCast(@min(derived, @as(u64, cap)));
    // The cap wins over the floor: a model whose whole context sits under
    // `min_window` still gets a window it can address.
    const window = @min(cap, @max(caps.min_window, capped));
    std.debug.assert(window <= cap);
    return window;
}

/// The window resolves before the slots, because a slot costs one window of bytes. A budget with
/// a pinned slot count takes the money window. Everything else takes the named window or the
/// serve default.
fn resolveWindow(caps: Caps, cost: KvCostModel, req: Request) error{WindowExceedsCeiling}!u32 {
    const ceil = windowCeil(caps);
    if (req.window) |w| {
        if (w > ceil) return error.WindowExceedsCeiling;
    }

    const flag_or_default = @min(req.window orelse caps.serve_default, ceil);
    const x = req.x_bytes orelse return flag_or_default;
    const s = req.slots orelse return flag_or_default;
    // A model with no per-token cost (pure recurrent state) has no window to
    // buy, so the flag or the default stands.
    if (cost.kv_bytes_per_token == 0) return flag_or_default;
    if (req.window) |w| return @min(moneyWindow(caps, cost, x, s, ceil), @min(w, ceil));

    return moneyWindow(caps, cost, x, s, implicitCeil(caps));
}

/// Slots follow the money in mode A, or stand pinned when named. Null is a refusal. Mode B passes
/// the request through. Only the money path can clamp: a pinned count is honored whole or refused.
fn resolveSlots(caps: Caps, cost: KvCostModel, req: Request, window: u32) ?ResolvedSlots {
    const x = req.x_bytes orelse return .{ .n_slots = req.slots orelse req.batch };
    const per_slot = perSlotBytes(cost, window);
    if (per_slot == 0) return null;

    if (req.slots) |s| {
        // Refuse when X cannot fund the pinned slots. `moneyWindow` floors at `min_window`, so an
        // under-funded floor surfaces here and never shrinks silently.
        if (@as(u64, s) * per_slot > x) return null;
        return .{ .n_slots = s };
    }

    const raw = x / per_slot;
    if (raw < req.batch) return null;

    const ceiling: u64 = caps.max_vram_slots;
    return .{ .n_slots = @intCast(@min(raw, ceiling)), .clamped = raw > ceiling };
}

/// What the residency blob ledger can reach: one blob per session the slots cannot hold, plus one
/// transient blob mid-swap. `blob` is one slot of the whole model at the window.
fn blobNeed(n_logical: u32, n_slots: u32, blob: u64) u64 {
    const evictable = n_logical -| n_slots;
    if (evictable == 0) return 0;
    return @as(u64, evictable + 1) * blob;
}

/// The host vector terms that an unset `--kv-ram` funds its blobs beside: the
/// prefix cache, the host KV, and the stage pair.
fn fixedRam(ram: LayerSplit.Ram) u64 {
    return ram.prefix +| ram.host_kv +| ram.swap_stages;
}

/// Split the RAM budget Y between the blob ledger and the prefix cache. Bytes past what the ledger
/// can reach fund the prefix cache. A caller with no logical count leaves the split unrun.
fn splitRam(cost: KvCostModel, req: Request, n_slots: u32, window: u32) KvPlan.Ram {
    const y = req.y_bytes orelse return .{ .blobs = 0, .prefix = 0 };
    const n_logical = req.n_logical orelse return .{ .blobs = y, .prefix = 0 };
    const blobs = @min(y, blobNeed(n_logical, n_slots, perSlotBytes(cost, window)));
    return .{ .blobs = blobs, .prefix = y - blobs };
}

/// The widest window the usable RAM funds at split `k`. `constant` is the fixed term that the
/// window does not move.
fn ramFitWindow(kv: KvStructure, k: u32, blobs: u64, constant: u64, ram_usable: u64) ?u32 {
    std.debug.assert(k <= kv.costs.n_layers);
    std.debug.assert(blobs <= limits.max_logical_sessions + 1);

    var host_token: u64 = 0;
    var host_fixed: u64 = 0;
    for (kv.costs.entries[0..k]) |entry| {
        host_token += entry.kv_bytes_per_token;
        host_fixed += entry.fixed_bytes_per_slot;
    }

    const whole = kv.costs.total();
    const per_position = @as(u64, kv.n_slots) * host_token + blobs * whole.kv_bytes_per_token;
    const fixed = (@as(
        u64,
        kv.n_slots,
    ) * host_fixed + blobs * whole.fixed_state_bytes_per_slot) +| constant;
    if (per_position == 0) return null;
    if (ram_usable < fixed) return null;

    const widest = (ram_usable - fixed) / per_position;
    if (widest < limits.min_kv_window) return null;

    // The engine bound caps a window that no request can name anyway.
    const fit: u32 = @intCast(@min(widest, limits.max_model_ctx));
    // The fit window spends inside the usable RAM.
    std.debug.assert(per_position * fit + fixed <= ram_usable);
    return fit;
}

/// The host vector of split `k`, the one home of the stage formula. Only a residency pool grows
/// the two swap stages. A stage holds one chunk at most, so the pair stops growing with the window.
fn ramAt(kv: KvStructure, k: u32, n_layers: u32, host: HostRequest) LayerSplit.Ram {
    return ramAtChunk(kv, k, n_layers, host, limits.swap_stage_chunk_bytes);
}

/// `ramAt` at an explicit stage chunk.
fn ramAtChunk(
    kv: KvStructure,
    k: u32,
    n_layers: u32,
    host: HostRequest,
    chunk: u64,
) LayerSplit.Ram {
    std.debug.assert(k <= n_layers);
    std.debug.assert(n_layers == kv.costs.n_layers);

    return .{
        .blobs = host.ram_split.blobs,
        .prefix = host.ram_split.prefix,
        .host_kv = kvHostBytes(kv, k),
        .swap_stages = if (host.residency) 2 * stageBytesAt(
            chunk,
            slotSpanBytes(kv, k, n_layers),
        ) else 0,
    };
}

/// One stage at an explicit chunk: the chunk, or the whole span when the span
/// is smaller.
fn stageBytesAt(chunk: u64, span: u64) u64 {
    std.debug.assert(chunk >= 1);

    const stage = @min(chunk, span);
    std.debug.assert(stage <= span);
    std.debug.assert(stage <= chunk);
    return stage;
}

/// The usable RAM a printed fit is computed against. Free RAM moves between two
/// boots, so the advice keeps `limits.ram_fit_headroom_bytes` back.
fn adviceUsable(ram_usable: u64) u64 {
    return ram_usable -| limits.ram_fit_headroom_bytes;
}

/// A printed window fit in whole granules, the unit a default window takes. A
/// fit under the window floor prints no window.
fn adviceFit(exact: ?u32) ?u32 {
    const fit = exact orelse return null;
    const floored: u32 = @intCast(limits.floorToGranule(fit));
    if (floored < limits.min_kv_window) return null;

    std.debug.assert(floored <= fit);
    return floored;
}

/// The weight bytes of the GPU suffix `[k..n_layers)`.
fn suffixWeights(weights: []const u64, k: u32) u64 {
    var total: u64 = 0;
    for (weights[@min(k, weights.len)..]) |w| {
        total += w;
    }

    return total;
}

fn kvVramBytes(kv: KvStructure, k: u32) u64 {
    return @as(u64, kv.n_slots) * slotSpanBytes(kv, k, kv.costs.n_layers);
}

fn kvHostBytes(kv: KvStructure, k: u32) u64 {
    return @as(u64, kv.n_slots) * slotSpanBytes(kv, 0, k);
}

/// The device bytes one split spends. The pure CPU runner holds no device context and charges
/// nothing. Every other split runs the head on the device, so it pays `head_bytes` whole.
fn deviceSpend(caps: Caps, weights: []const u64, head_bytes: u64, kv: KvStructure, k: u32) u64 {
    if (weights.len <= k) return 0;
    return suffixWeights(weights, k) + head_bytes + kvVramBytes(kv, k) +
        caps.logits + caps.scratch;
}

/// The preflight at one split. It reuses `usable`, so the 7/8 rule keeps one
/// home for both planners.
fn layerSplitFits(caps: Caps, weights: []const u64, head_bytes: u64, kv: KvStructure, k: u32) bool {
    return deviceSpend(caps, weights, head_bytes, kv, k) <= usable(caps);
}

/// The least split the budget funds, so the GPU suffix is the widest. Weights and VRAM KV shrink
/// as the split rises, so the first feasible split on an upward walk is best. The pure CPU split
/// spends nothing, so the walk always ends on a feasible split.
fn autoSplit(caps: Caps, weights: []const u64, head_bytes: u64, kv: KvStructure) u32 {
    const n_layers: u32 = @intCast(weights.len);
    var k: u32 = 0;
    while (k < n_layers) : (k += 1) {
        if (layerSplitFits(caps, weights, head_bytes, kv, k)) return k;
    }

    return n_layers;
}

// -- tests ------------------------------------------------------------------

test "GpuLayers: parse takes auto or a decimal u32 and nothing else, pin and isZeroCount follow" {
    const Case = struct { raw: []const u8, want: ?GpuLayers, pin: ?u32 = null, zero: bool = false };
    const u32_max = std.math.maxInt(u32);
    const cases = [_]Case{
        .{ .raw = "auto", .want = .auto },
        .{ .raw = "0", .want = .{ .count = 0 }, .pin = 0, .zero = true },
        .{ .raw = "1", .want = .{ .count = 1 }, .pin = 1 },
        .{ .raw = "12", .want = .{ .count = 12 }, .pin = 12 },
        .{ .raw = "17", .want = .{ .count = 17 }, .pin = 17 },
        .{ .raw = "4294967295", .want = .{ .count = u32_max }, .pin = u32_max },
        // One past the `u32` ceiling is a refusal, not a wrap.
        .{ .raw = "4294967296", .want = null },
        .{ .raw = "", .want = null },
        .{ .raw = "x", .want = null },
        .{ .raw = "-1", .want = null },
        .{ .raw = "auto2", .want = null },
        .{ .raw = "AUTO", .want = null },
    };
    for (cases) |c| {
        errdefer std.debug.print("GpuLayers case: {s}\n", .{c.raw});

        try std.testing.expectEqual(c.want, GpuLayers.parse(c.raw));
        const parsed = c.want orelse continue;
        try std.testing.expectEqual(c.pin, GpuLayers.pin(parsed));
        try std.testing.expectEqual(c.zero, GpuLayers.isZeroCount(parsed));
    }
}

// A shared fixture: Llama-3.1-8B holds 2 * 32 layers * 1024 bytes * 4 = 262144
// KV bytes per token, and declares no fixed per-slot state.
const llama_8b_cost: KvCostModel = .{ .kv_bytes_per_token = 262_144 };
// The same model at the f16 KV dtype: half the element size halves the
// per-token cost, and nothing else in the economy moves.
const llama_8b_f16_cost: KvCostModel = .{ .kv_bytes_per_token = 131_072 };
// The default windows of Llama-3.1-8B at f32 KV: `134217728 / 262144 = 512` at the serve budget
// and `2147483648 / 262144 = 8192` at the run budget. The fixtures carry them at every dtype.
const llama_8b_serve_default: u32 = 512;
const llama_8b_default_ceil: u32 = 8192;
const no_vram_caps: Caps = llama8bCaps(0);

fn llama8bCaps(vram_free: u64) Caps {
    return .{
        .vram_free = vram_free,
        .weights = 0,
        .logits = 0,
        .model_ctx = 131_072,
        .serve_default = llama_8b_serve_default,
        .default_ceil = llama_8b_default_ceil,
    };
}

// A shared split fixture: 32 layers of 1000 weight bytes, one slot at a 100-token window, and
// 10 KV bytes per layer per token. The device spend at split `k` is `2000 * (32 - k)`.
const anchor_weights = [_]u64{1000} ** 32;
const anchor_kv: KvStructure = .{
    .n_slots = 1,
    .window = 100,
    .costs = KvCostVector.uniform(32, 10),
};

const no_ram_split: KvPlan.Ram = .{ .blobs = 0, .prefix = 0 };

/// The host request the split tests ran before the RAM check existed: a
/// serve caller that prices both stages, with the check off. Each such test
/// keeps its meaning under it.
const unchecked_serve: HostRequest = .{ .ram_split = no_ram_split, .residency = true };

/// The layer split planner reads neither default window, so the fixture pins both
/// at the window floor.
fn anchorCaps(vram_free: u64) Caps {
    return .{
        .vram_free = vram_free,
        .weights = 0,
        .logits = 0,
        .model_ctx = 4096,
        .serve_default = limits.min_kv_window,
        .default_ceil = limits.min_kv_window,
    };
}

/// A non-uniform fixture: layer `i` charges `10 + 2 * i` bytes a token, and
/// every third layer holds a 500-byte slab instead. Both columns therefore vary
/// by layer, which a uniform price cannot reproduce at any split.
fn rampCosts(n_layers: u32) KvCostVector {
    var costs = KvCostVector.uniform(n_layers, 0);
    var i: u32 = 0;
    while (i < n_layers) : (i += 1) {
        if (i % 3 == 2) {
            costs.set(i, 0, 500);
        } else {
            costs.set(i, 10 + 2 * @as(u64, i), 0);
        }
    }

    return costs;
}

/// The Qwen3.5-2B shape as a synthetic vector: 24 layers at a full-attention
/// interval of 4, so layers 3, 7, 11, 15, 19, and 23 charge 4096 KV bytes a
/// token and the other 18 hold a 1122304-byte recurrent slab.
fn qwen2bCosts() KvCostVector {
    var costs = KvCostVector.uniform(24, 0);
    var i: u32 = 0;
    while (i < 24) : (i += 1) {
        if ((i + 1) % 4 == 0) {
            costs.set(i, 4_096, 0);
        } else {
            costs.set(i, 0, 1_122_304);
        }
    }

    return costs;
}

/// Every entry of `costs` at half price, in both columns. The ramp fixture
/// halves exactly, so this states a pointwise-cheaper vector with no rounding.
fn halfCosts(costs: KvCostVector) KvCostVector {
    var out = KvCostVector.uniform(costs.n_layers, 0);
    var i: u32 = 0;
    while (i < costs.n_layers) : (i += 1) {
        const entry = costs.entries[i];
        std.debug.assert(entry.kv_bytes_per_token % 2 == 0);
        std.debug.assert(entry.fixed_bytes_per_slot % 2 == 0);
        out.set(
            i,
            @as(u64, entry.kv_bytes_per_token) / 2,
            @as(u64, entry.fixed_bytes_per_slot) / 2,
        );
    }

    return out;
}

const PlanWant = struct {
    err: ?anyerror = null,
    n_slots: ?u32 = null,
    n_streams: ?u32 = null,
    window: ?u32 = null,
    vram_kv: ?u64 = null,
    slots_clamped: ?bool = null,
    /// The planned window against the default ceiling of the caps.
    versus_default_ceil: ?std.math.Order = null,
    /// The planned window equals the run default of this cost.
    run_default_of: ?KvCostModel = null,
};

const PlanCase = struct {
    name: []const u8,
    cost: KvCostModel,
    req: Request,
    /// A model context other than the one of `no_vram_caps`.
    model_ctx: ?u32 = null,
    want: PlanWant,
};

/// A refusal of the plan is the refusal of the quote, so no message prices a
/// window that the plan refuses.
fn checkPlan(c: PlanCase) !void {
    // Apply the model context of the case.
    var caps = no_vram_caps;
    if (c.model_ctx) |ctx| {
        caps.model_ctx = ctx;
    }

    const w = c.want;

    // Check that the plan and the quote refuse.
    if (w.err) |err| {
        try std.testing.expectEqual(
            err,
            if (kvPlan(caps, c.cost, c.req)) |_| error.PlanDidNotRefuse else |e| e,
        );
        try std.testing.expectEqual(
            err,
            if (quote(caps, c.cost, c.req)) |_| error.QuoteDidNotRefuse else |e| e,
        );
        return;
    }

    // Check each field that the case names.
    const plan = try kvPlan(caps, c.cost, c.req);
    if (w.n_slots) |v| {
        try std.testing.expectEqual(v, plan.n_slots);
    }

    if (w.n_streams) |v| {
        try std.testing.expectEqual(v, plan.n_streams);
    }

    if (w.window) |v| {
        try std.testing.expectEqual(v, plan.window);
    }

    if (w.vram_kv) |v| {
        try std.testing.expectEqual(v, plan.vram.kv);
    }

    if (w.slots_clamped) |v| {
        try std.testing.expectEqual(v, plan.slots_clamped);
    }

    if (w.versus_default_ceil) |v| {
        try std.testing.expectEqual(v, std.math.order(plan.window, caps.default_ceil));
    }

    if (w.run_default_of) |cost| {
        try std.testing.expectEqual(
            defaultWindow(cost, caps.model_ctx, limits.run_session_kv_bytes),
            plan.window,
        );
    }
}

test "kvPlan: the planner anchors, the f16 dtype anchors, and the named-window anchors" {
    const gib: u64 = 1 << 30;
    const qwen_f16: KvCostModel = .{ .kv_bytes_per_token = 12_288 };
    const cases = [_]PlanCase{
        // A 5 GiB KV budget at window 512 buys 40 slots, of which 8 decode per tick.
        .{
            .name = "budget, window, and batch",
            .cost = llama_8b_cost,
            .req = .{ .x_bytes = 5 * gib, .window = 512, .batch = 8 },
            .want = .{ .n_slots = 40, .n_streams = 8, .window = 512, .vram_kv = 5 * gib },
        },
        // No flags: slots equal the decode width and the window is the serve default.
        .{
            .name = "no flags",
            .cost = llama_8b_cost,
            .req = .{ .batch = 4 },
            .want = .{ .n_slots = 4, .n_streams = 4, .window = 512 },
        },
        // A pinned slot count with a budget takes the window the money affords:
        // 1 GiB over 4 slots at 256 KiB/token is a 1024-token window.
        .{
            .name = "pinned slots take the money window",
            .cost = llama_8b_cost,
            .req = .{ .x_bytes = gib, .slots = 4, .batch = 4 },
            .want = .{ .n_slots = 4, .window = 1024, .vram_kv = gib },
        },
        // A slot at window 512 costs 128 MiB at 262144 bytes per token and 64 MiB
        // at 131072, so 1 GiB buys 8 slots at f32 and 16 at f16.
        .{
            .name = "f32 buys 8 slots",
            .cost = llama_8b_cost,
            .req = .{ .x_bytes = gib, .window = 512, .batch = 8 },
            .want = .{ .n_slots = 8, .slots_clamped = false },
        },
        .{
            .name = "f16 buys 16 slots",
            .cost = llama_8b_f16_cost,
            .req = .{ .x_bytes = gib, .window = 512, .batch = 8 },
            .want = .{ .n_slots = 16, .slots_clamped = false },
        },
        // 9 GiB over a 64 MiB slot derives 144 slots, which the 128-slot ceiling
        // cuts. The 1 GiB left over stays unspendable.
        .{
            .name = "f16 saturates the slot ceiling",
            .cost = llama_8b_f16_cost,
            .req = .{ .x_bytes = 9 * gib, .window = 512, .batch = 8 },
            .want = .{
                .n_slots = limits.max_vram_slots,
                .slots_clamped = true,
                .vram_kv = 8 * gib,
            },
        },
        .{
            .name = "f32 reaches no ceiling at 9 GiB",
            .cost = llama_8b_cost,
            .req = .{ .x_bytes = 9 * gib, .window = 512, .batch = 8 },
            .want = .{ .n_slots = 72, .slots_clamped = false },
        },
        // Half the per-token price buys twice the window, and the default ceiling
        // of 8192 grants that doubling free.
        .{
            .name = "f16 money window doubles below the default ceiling",
            .cost = llama_8b_f16_cost,
            .req = .{ .x_bytes = gib, .slots = 4, .batch = 4 },
            .want = .{
                .n_slots = 4,
                .window = 2048,
                .versus_default_ceil = .lt,
                .slots_clamped = false,
            },
        },
        // 5 GiB over 4 pinned slots buys 10240 tokens at f16. The request names no
        // window, so the default ceiling cuts it to the run default of Llama-8B at
        // f32. A pinned count never raises the clamp witness.
        .{
            .name = "f16 money window lands on the default ceiling",
            .cost = llama_8b_f16_cost,
            .req = .{ .x_bytes = 5 * gib, .slots = 4, .batch = 4 },
            .want = .{
                .n_slots = 4,
                .window = 8192,
                .run_default_of = llama_8b_cost,
                .slots_clamped = false,
                .vram_kv = 4 * gib,
            },
        },
        // The named window is the last context past the model ceiling of 262144.
        .{
            .name = "a named window one past the model ceiling",
            .cost = qwen_f16,
            .model_ctx = 262_144,
            .req = .{ .window = 262_145 },
            .want = .{ .err = error.WindowExceedsCeiling },
        },
        // The named 32768 sits under the model ceiling, so the money fixes the window.
        .{
            .name = "a named window past the default ceiling takes the money window",
            .cost = llama_8b_f16_cost,
            .req = .{ .x_bytes = 5 * gib, .slots = 4, .batch = 4, .window = 32_768 },
            .want = .{ .window = 10_240, .versus_default_ceil = .gt },
        },
        // One slot at 12288 bytes a token over 262144 positions reserves 3 GiB.
        .{
            .name = "a named window at the model ceiling",
            .cost = qwen_f16,
            .model_ctx = 262_144,
            .req = .{ .window = 262_144, .batch = 1 },
            .want = .{ .window = 262_144, .vram_kv = 3_221_225_472 },
        },
    };
    for (cases) |c| {
        errdefer std.debug.print("plan case: {s}\n", .{c.name});

        try checkPlan(c);
    }
}

test "kvPlan mode A: refusal, spend bound, slot band, and monotonicity in X" {
    const caps = llama8bCaps(8 << 30);
    const windows = [_]u32{ 128, 512, 2048 };
    const batches = [_]u32{ 1, 4, 8 };
    // 64 budgets from zero to 8 GiB. The grid is fixed, so a failure replays.
    const step: u64 = 128 << 20;
    for (windows) |w| {
        for (batches) |batch| {
            var previous_slots: u32 = 0;
            for (0..64) |i| {
                const x: u64 = @as(u64, @intCast(i)) * step;
                const raw = x / (@as(u64, w) * llama_8b_cost.kv_bytes_per_token);
                const plan = kvPlan(caps, llama_8b_cost, .{
                    .x_bytes = x,
                    .window = w,
                    .batch = batch,
                }) catch {
                    try std.testing.expect(raw < batch);
                    continue;
                };
                try std.testing.expect(raw >= batch);
                try std.testing.expect(plan.vram.kv <= x);
                try std.testing.expect(plan.n_slots >= batch);
                try std.testing.expect(plan.n_slots <= caps.max_vram_slots);
                try std.testing.expectEqual(batch, plan.n_streams);
                try std.testing.expect(plan.window <= windowCeil(caps));
                try std.testing.expect(plan.n_slots >= previous_slots);
                previous_slots = plan.n_slots;
            }

            // The widest budget on the grid saturates the physical ceiling once
            // the window is small enough to make a slot cheap.
            if (w == 128) {
                try std.testing.expectEqual(caps.max_vram_slots, previous_slots);
            }
        }
    }
}

test "kvPlan mode A: a cheaper KV dtype never buys less" {
    // The two costs differ only in the per-token price, which is the one term
    // the dtype moves.
    const caps = llama8bCaps(8 << 30);
    const windows = [_]u32{ 128, 512, 2048 };
    const batches = [_]u32{ 1, 4, 8 };
    // 64 budgets from zero to 8 GiB. The grid is fixed, so a failure replays.
    const step: u64 = 128 << 20;
    for (windows) |w| {
        for (batches) |batch| {
            for (0..64) |i| {
                const req: Request = .{
                    .x_bytes = @as(u64, @intCast(i)) * step,
                    .window = w,
                    .batch = batch,
                };
                // A budget the dear cost refuses says nothing about the cheap
                // one, so the pair is only comparable where f32 succeeds.
                const wide = kvPlan(caps, llama_8b_cost, req) catch continue;
                // A budget that funds f32 funds f16, so `try` is the implication.
                const half = try kvPlan(caps, llama_8b_f16_cost, req);
                try std.testing.expect(half.n_slots >= wide.n_slots);
                try std.testing.expect(half.window >= wide.window);
            }
        }
    }
}

test "kvPlan mode A with pinned slots: the money window, its clamps, and its refusal" {
    const caps = llama8bCaps(8 << 30);
    // The spend never exceeds X, and the window grows monotonically with X.
    var previous_window: u32 = 0;
    for (1..33) |i| {
        const x: u64 = @as(u64, @intCast(i)) * (256 << 20);
        const plan = kvPlan(caps, llama_8b_cost, .{ .x_bytes = x, .slots = 4, .batch = 4 }) catch {
            // A refusal means 4 slots at the minimum window cost more than X.
            try std.testing.expect(4 * perSlotBytes(llama_8b_cost, limits.min_kv_window) > x);
            try std.testing.expectEqual(@as(u32, 0), previous_window);
            continue;
        };
        try std.testing.expectEqual(@as(u32, 4), plan.n_slots);
        try std.testing.expect(plan.vram.kv <= x);
        try std.testing.expect(plan.window >= limits.min_kv_window);
        try std.testing.expect(plan.window >= previous_window);
        previous_window = plan.window;
    }

    // A roomy budget saturates the default ceiling. 4 GiB a slot buys 16384
    // tokens, and the byte rule stops the unnamed window at
    // `2147483648 / 262144 = 8192`.
    const roomy = try kvPlan(caps, llama_8b_cost, .{ .x_bytes = 8 << 30, .slots = 2, .batch = 2 });
    try std.testing.expectEqual(@as(u32, 8192), roomy.window);
    try std.testing.expectEqual(
        defaultWindow(llama_8b_cost, caps.model_ctx, limits.run_session_kv_bytes),
        roomy.window,
    );

    // An explicit `--max-seq` caps the money window, never raises it.
    const capped = try kvPlan(
        caps,
        llama_8b_cost,
        .{ .x_bytes = 1 << 30, .slots = 4, .batch = 4, .window = 300 },
    );
    try std.testing.expectEqual(@as(u32, 300), capped.window);

    // A budget under the minimum window refuses loudly instead of flooring.
    try std.testing.expectError(error.KvVramBudgetTooSmall, kvPlan(caps, llama_8b_cost, .{
        .x_bytes = 4 * perSlotBytes(llama_8b_cost, limits.min_kv_window) - 1,
        .slots = 4,
        .batch = 4,
    }));

    // Fixed per-slot state is funded before window bytes, and a pure
    // recurrent-state model keeps the default window.
    const gdn_cost: KvCostModel = .{
        .kv_bytes_per_token = 0,
        .fixed_state_bytes_per_slot = 1 << 20,
    };
    const gdn = try kvPlan(caps, gdn_cost, .{ .x_bytes = 8 << 20, .slots = 4, .batch = 4 });
    try std.testing.expectEqual(caps.serve_default, gdn.window);
    try std.testing.expectEqual(@as(u64, 4 << 20), gdn.vram.kv);
    try std.testing.expectError(error.KvVramBudgetTooSmall, kvPlan(caps, gdn_cost, .{
        .x_bytes = 3 << 20,
        .slots = 4,
        .batch = 4,
    }));
}

test "derivedX: its plan fits the preflight, and an over-subscribed device refuses" {
    const combos = [_]struct { weights: u64, logits: u64, scratch: u64 }{
        .{ .weights = 0, .logits = 0, .scratch = 0 },
        .{ .weights = 5 << 30, .logits = 512 << 20, .scratch = 128 << 20 },
        .{ .weights = 6 << 30, .logits = 1 << 30, .scratch = 0 }, // exactly the usable share
        .{ .weights = 7 << 30, .logits = 1 << 30, .scratch = 1 << 20 }, // past the usable share
    };
    for (combos) |combo| {
        var caps = llama8bCaps(8 << 30);
        caps.weights = combo.weights;
        caps.logits = combo.logits;
        caps.scratch = combo.scratch;
        const x = derivedX(caps);
        const plan = kvPlan(caps, llama_8b_cost, .{
            .x_bytes = x,
            .window = 512,
            .batch = 1,
        }) catch {
            // A refusal means the remainder cannot buy one slot.
            try std.testing.expect(x < 512 * llama_8b_cost.kv_bytes_per_token);
            continue;
        };
        try std.testing.expect(planFits(caps, plan));
    }

    // The last two combos leave nothing, so the refusal above is reached.
    var full = llama8bCaps(8 << 30);
    full.weights = 7 << 30;
    full.logits = 1 << 30;
    try std.testing.expectEqual(@as(u64, 0), derivedX(full));
}

test "quote: the price it names is the price the plan pays" {
    const caps = llama8bCaps(8 << 30);
    const req: Request = .{ .x_bytes = 1 << 30, .window = 512, .batch = 2 };
    const q = try quote(caps, llama_8b_cost, req);
    const plan = try kvPlan(caps, llama_8b_cost, req);
    try std.testing.expectEqual(plan.window, q.window);
    try std.testing.expectEqual(plan.vram.kv, @as(u64, plan.n_slots) * q.per_slot_bytes);

    // One slot short of the decode width refuses, and the quote explains why.
    var tight = req;
    tight.x_bytes = q.per_slot_bytes;
    try std.testing.expectError(error.KvVramBudgetTooSmall, kvPlan(caps, llama_8b_cost, tight));
    try std.testing.expect(tight.x_bytes.? / q.per_slot_bytes < tight.batch);

    // With no window flag the quote reads the window the planner defaults to.
    const bare = try quote(caps, llama_8b_cost, .{ .x_bytes = 1 << 30, .batch = 1 });
    try std.testing.expectEqual(caps.serve_default, bare.window);

    // With pinned slots the quote reads the money window the planner derives.
    const pinned_req: Request = .{ .x_bytes = 1 << 30, .slots = 4, .batch = 4 };
    const pq = try quote(caps, llama_8b_cost, pinned_req);
    const pp = try kvPlan(caps, llama_8b_cost, pinned_req);
    try std.testing.expectEqual(pp.window, pq.window);
}

test "pinned: passes the shape through and reserves nothing" {
    const plan = pinned(3, 777);
    try std.testing.expectEqual(@as(u32, 3), plan.n_slots);
    try std.testing.expectEqual(@as(u32, 3), plan.n_streams);
    try std.testing.expectEqual(@as(u32, 777), plan.window);
    try std.testing.expectEqual(@as(u64, 0), plan.vram.weights);
    try std.testing.expectEqual(@as(u64, 0), plan.vram.kv);
    try std.testing.expectEqual(@as(u64, 0), plan.vram.logits);
    try std.testing.expectEqual(@as(u64, 0), plan.vram.scratch);
    // A pinned plan reserves nothing, so no device budget can refuse it.
    try std.testing.expect(planFits(no_vram_caps, plan));
    // A pinned slot count past the decode ceiling keeps the width at the
    // ceiling: breadth is bought, width is stepped.
    try std.testing.expectEqual(limits.max_batch, pinned(40, 512).n_streams);
}

test "kvPlan mode B: degenerate input reproduces the shipped shapes" {
    for (1..limits.max_batch + 1) |b| {
        const batch: u32 = @intCast(b);
        const plan = try kvPlan(no_vram_caps, llama_8b_cost, .{ .batch = batch });
        try std.testing.expectEqual(batch, plan.n_slots);
        try std.testing.expectEqual(batch, plan.n_streams);
        try std.testing.expectEqual(no_vram_caps.serve_default, plan.window);
        try std.testing.expectEqual(@as(u64, 0), plan.ram.blobs);
        try std.testing.expectEqual(@as(u64, 0), plan.ram.prefix);
    }

    // A model whose context sits below the serve default caps the window.
    var narrow_caps = no_vram_caps;
    narrow_caps.model_ctx = 256;
    const narrow = try kvPlan(narrow_caps, llama_8b_cost, .{ .batch = 1 });
    try std.testing.expectEqual(@as(u32, 256), narrow.window);

    // A slot count without a budget passes through at the serve default, and
    // slots past the decode ceiling keep the requested width.
    const caps = llama8bCaps(8 << 30);
    const cost: KvCostModel = .{ .kv_bytes_per_token = 4 << 20 };
    const wide = try kvPlan(caps, cost, .{ .slots = 40, .batch = 2 });
    try std.testing.expectEqual(@as(u32, 40), wide.n_slots);
    try std.testing.expectEqual(@as(u32, 2), wide.n_streams);
    try std.testing.expectEqual(caps.serve_default, wide.window);

    // An explicit window wins, and the RAM budget passes through.
    const explicit = try kvPlan(
        caps,
        cost,
        .{ .slots = 2, .batch = 2, .window = 300, .y_bytes = 1 << 30 },
    );
    try std.testing.expectEqual(@as(u32, 300), explicit.window);
    try std.testing.expectEqual(@as(u64, 1 << 30), explicit.ram.blobs);
}

test "kvPlan: a named window is refused past the model ceiling, an unnamed one stays under" {
    // The last context sits past the engine bound, so the ceiling there is the
    // bound itself.
    const contexts = [_]u32{ 256, 8192, 131_072, limits.max_model_ctx, limits.max_model_ctx + 1 };
    const costs = [_]KvCostModel{
        llama_8b_cost,
        .{ .kv_bytes_per_token = 12_288, .fixed_state_bytes_per_slot = 20_201_472 },
        .{ .kv_bytes_per_token = 0, .fixed_state_bytes_per_slot = 1 << 20 },
    };
    const requests = [_]Request{
        .{ .batch = 1 },
        .{ .slots = 4, .batch = 4 },
        .{ .x_bytes = 8 << 30, .batch = 1 },
        .{ .x_bytes = 8 << 30, .slots = 4, .batch = 4 },
    };
    for (contexts) |model_ctx| {
        for (costs) |cost| {
            var caps = no_vram_caps;
            caps.model_ctx = model_ctx;
            const ceil = windowCeil(caps);
            try std.testing.expectEqual(@min(model_ctx, limits.max_model_ctx), ceil);
            // The caller derives both defaults at the model ceiling.
            caps.serve_default = defaultWindow(cost, ceil, limits.serve_session_kv_bytes);
            caps.default_ceil = defaultWindow(cost, ceil, limits.run_session_kv_bytes);
            const windows = [_]?u32{
                null,
                1,
                limits.min_kv_window,
                ceil - 1,
                ceil,
                ceil + 1,
                std.math.maxInt(u32),
            };
            for (requests) |base| {
                for (windows) |window| {
                    var req = base;
                    req.window = window;
                    const past = if (window) |w| w > ceil else false;
                    if (kvPlan(caps, cost, req)) |plan| {
                        try std.testing.expect(!past);
                        try std.testing.expect(plan.window <= ceil);
                        if (window == null) {
                            try std.testing.expect(plan.window <= caps.default_ceil);
                        }
                    } else |err| {
                        // A window past the ceiling is the one window refusal,
                        // and every other refusal is the budget's.
                        const want: anyerror = if (past)
                            error.WindowExceedsCeiling
                        else
                            error.KvVramBudgetTooSmall;

                        try std.testing.expectEqual(want, err);
                    }
                }
            }
        }
    }
}

test "defaultWindow: the registry models get their byte-budget windows" {
    // One row per registry shape at its registry KV dtype. The run budget is
    // 2147483648 bytes and the serve budget 134217728.
    const Row = struct {
        shape: KvShape,
        kv_dtype: dtype.KvDtype,
        declared: u32,
        override: ?u32 = null,
        cost: u64,
        fixed: u64 = 0,
        ceiling: u32,
        run: u32,
        serve: u32,
    };
    const qwen_2b: KvShape.Recurrent = .{
        .full_attention_interval = 4,
        .conv_kernel = 4,
        .state_size = 128,
        .group_count = 16,
        .time_step_rank = 16,
    };
    var qwen_9b = qwen_2b;
    qwen_9b.time_step_rank = 32;
    const rows = [_]Row{
        // SmolLM2-135M at f32: `2 * 30 * 192 * 4 = 46080` bytes a token.
        // `2147483648 / 46080 = 46603` rounds down to 46592, which the 8192
        // ceiling cuts, and `134217728 / 46080 = 2912` rounds down to 2816.
        .{
            .shape = .{ .n_layers = 30, .kv_row_elems = 3 * 64 },
            .kv_dtype = .f32,
            .declared = 8192,
            .cost = 46_080,
            .ceiling = 8192,
            .run = 8192,
            .serve = 2816,
        },
        // Llama-3.1-8B Q8_0 at f16: `2 * 32 * 1024 * 2 = 131072` bytes a token.
        // `2147483648 / 131072 = 16384` and `134217728 / 131072 = 1024`.
        .{
            .shape = .{ .n_layers = 32, .kv_row_elems = 8 * 128 },
            .kv_dtype = .f16,
            .declared = 131_072,
            .cost = 131_072,
            .ceiling = 131_072,
            .run = 16_384,
            .serve = 1024,
        },
        // Llama-3.1-8B Q4_0 at f32 under its 8192 override: 262144 bytes a
        // token. `2147483648 / 262144 = 8192` meets the override ceiling, and
        // `134217728 / 262144 = 512`.
        .{
            .shape = .{ .n_layers = 32, .kv_row_elems = 8 * 128 },
            .kv_dtype = .f32,
            .declared = 131_072,
            .override = 8192,
            .cost = 262_144,
            .ceiling = 8192,
            .run = 8192,
            .serve = 512,
        },
        // The Llama-3.1-8B shape at f32 with no context override: 262144
        // bytes a token under the full 131072 ceiling, so
        // `2147483648 / 262144 = 8192` and `134217728 / 262144 = 512`.
        .{
            .shape = .{ .n_layers = 32, .kv_row_elems = 8 * 128 },
            .kv_dtype = .f32,
            .declared = 131_072,
            .cost = 262_144,
            .ceiling = 131_072,
            .run = 8192,
            .serve = 512,
        },
        // Qwen3.5-2B at f16: 6 attention layers at `2 * 512 * 2 = 2048` bytes
        // make 12288 bytes a token, and 18 recurrent layers hold 1122304 bytes
        // each. `2147483648 / 12288 = 174762` rounds down to 174592, and
        // `134217728 / 12288 = 10922` rounds down to 10752.
        .{
            .shape = .{ .n_layers = 24, .kv_row_elems = 2 * 256, .recurrent = qwen_2b },
            .kv_dtype = .f16,
            .declared = 262_144,
            .cost = 12_288,
            .fixed = 18 * 1_122_304,
            .ceiling = 262_144,
            .run = 174_592,
            .serve = 10_752,
        },
        // Qwen3.5-9B at f16: 8 attention layers at `2 * 1024 * 2 = 4096`
        // bytes make 32768 bytes a token, and 24 recurrent layers hold 2195456
        // bytes each. `2147483648 / 32768 = 65536` and
        // `134217728 / 32768 = 4096`.
        .{
            .shape = .{ .n_layers = 32, .kv_row_elems = 4 * 256, .recurrent = qwen_9b },
            .kv_dtype = .f16,
            .declared = 262_144,
            .cost = 32_768,
            .fixed = 24 * 2_195_456,
            .ceiling = 262_144,
            .run = 65_536,
            .serve = 4096,
        },
    };
    for (rows) |row| {
        const cost = costModel(row.shape, row.kv_dtype);
        try std.testing.expectEqual(row.cost, cost.kv_bytes_per_token);
        try std.testing.expectEqual(row.fixed, cost.fixed_state_bytes_per_slot);
        const ceiling = limits.contextCeiling(row.declared, row.override);
        try std.testing.expectEqual(row.ceiling, ceiling);
        try std.testing.expectEqual(
            row.run,
            defaultWindow(cost, ceiling, limits.run_session_kv_bytes),
        );
        try std.testing.expectEqual(
            row.serve,
            defaultWindow(cost, ceiling, limits.serve_session_kv_bytes),
        );
        // No registry model gets a window under 8192 for `run` or under 512
        // per serve session.
        try std.testing.expect(row.run >= 8192);
        try std.testing.expect(row.serve >= 512);
    }
}

test "defaultWindow: never past the ceiling or the budget, and never shallower for a cheaper cost" {
    const costs = [_]u64{ 0, 1, 12_288, 32_768, 46_080, 131_072, 262_144, 1 << 30 };
    const ceilings = [_]u32{
        1,
        limits.min_kv_window,
        limits.window_granule,
        8192,
        131_072,
        limits.max_model_ctx,
    };
    const budgets = [_]u64{
        0,
        limits.serve_session_kv_bytes,
        limits.run_session_kv_bytes,
        1 << 40,
    };
    for (ceilings) |ceiling| {
        for (budgets) |bytes| {
            for (costs, 0..) |per_token, i| {
                const cost: KvCostModel = .{ .kv_bytes_per_token = per_token };
                const window = defaultWindow(cost, ceiling, bytes);
                try std.testing.expect(window >= 1);
                try std.testing.expect(window <= ceiling);
                if (per_token == 0) {
                    // No per-token cost buys the whole ceiling.
                    try std.testing.expectEqual(ceiling, window);
                    continue;
                }

                // A budget that funds the floor bounds the reservation.
                if (@as(u64, limits.min_kv_window) * per_token <= bytes) {
                    try std.testing.expect(@as(u64, window) * per_token <= bytes);
                }

                // The list is sorted, so every later cost is dearer.
                for (costs[i + 1 ..]) |dearer| {
                    const deeper = defaultWindow(.{ .kv_bytes_per_token = dearer }, ceiling, bytes);
                    try std.testing.expect(deeper <= window);
                }
            }
        }
    }
}

test "defaultWindows: each field is its budget window, and serve never passes run" {
    const costs = [_]u64{ 0, 1, 12_288, 131_072, 262_144, 1 << 30 };
    const ceilings = [_]u32{ 1, limits.min_kv_window, 8192, 131_072, limits.max_model_ctx };
    for (ceilings) |ceiling| {
        for (costs) |per_token| {
            const cost: KvCostModel = .{ .kv_bytes_per_token = per_token };
            const windows = defaultWindows(cost, ceiling);
            try std.testing.expectEqual(
                defaultWindow(cost, ceiling, limits.run_session_kv_bytes),
                windows.run,
            );
            try std.testing.expectEqual(
                defaultWindow(cost, ceiling, limits.serve_session_kv_bytes),
                windows.serve,
            );
            try std.testing.expect(windows.serve <= windows.run);
        }
    }

    // The costliest registry model: an 8B at f32 KV keeps 8192 and 512.
    const llama = defaultWindows(llama_8b_cost, 131_072);
    try std.testing.expectEqual(@as(u32, 8192), llama.run);
    try std.testing.expectEqual(@as(u32, 512), llama.serve);
}

test "fitWindow: it keeps a window that fits, cuts one that binds, and floors at the minimum" {
    // 16 GiB free leaves 14 GiB usable, which funds the 2 GiB of one slot at
    // 8192 tokens, so the window stands.
    try std.testing.expectEqual(
        @as(u32, 8192),
        fitWindow(llama8bCaps(16 << 30), llama_8b_cost, 1, 8192),
    );

    // 1 GiB free leaves `1073741824 / 8 * 7 = 939524096` usable bytes, which
    // buy `939524096 / 262144 = 3584` tokens for one slot.
    const tight = llama8bCaps(1 << 30);
    const fit = fitWindow(tight, llama_8b_cost, 1, 8192);
    try std.testing.expectEqual(@as(u32, 3584), fit);
    // The fit is the widest window whose plan passes the preflight.
    try std.testing.expect(planFits(tight, try kvPlan(tight, llama_8b_cost, .{ .window = fit })));
    try std.testing.expect(
        !planFits(tight, try kvPlan(tight, llama_8b_cost, .{ .window = fit + 1 })),
    );
    // Two slots split the same budget, so each gets `3584 / 2 = 1792` tokens.
    try std.testing.expectEqual(@as(u32, 1792), fitWindow(tight, llama_8b_cost, 2, 8192));

    // 16 MiB free leaves 14680064 usable bytes, which buy only 56 tokens. The
    // fit floors at the minimum window, and the preflight refuses that plan.
    const starved = llama8bCaps(16 << 20);
    const floor = fitWindow(starved, llama_8b_cost, 1, 8192);
    try std.testing.expectEqual(limits.min_kv_window, floor);
    try std.testing.expect(
        !planFits(starved, try kvPlan(starved, llama_8b_cost, .{ .window = floor })),
    );
    // A window under the floor keeps its own size, because the cap wins over
    // the floor.
    try std.testing.expectEqual(@as(u32, 100), fitWindow(starved, llama_8b_cost, 1, 100));
    // A model with no per-token cost has no window to fit.
    try std.testing.expectEqual(
        @as(u32, 8192),
        fitWindow(starved, .{ .kv_bytes_per_token = 0 }, 1, 8192),
    );
}

test "largestFittingWindow: the widest fit sits between the floor and the refused window" {
    // 8000000 available bytes leave 7000000 usable. A 6500000 reservation
    // leaves 500000, which buys 500 tokens at 1000 bytes each.
    const cost: KvCostModel = .{ .kv_bytes_per_token = 1000 };
    try std.testing.expectEqual(
        @as(?u32, 500),
        largestFittingWindow(8_000_000, 6_500_000, cost, 1, 1000),
    );
    // The fixed state comes off each slot's share first: 620000 over two
    // slots is 310000, and 300000 of it buys 300 tokens.
    const hybrid: KvCostModel = .{
        .kv_bytes_per_token = 1000,
        .fixed_state_bytes_per_slot = 10_000,
    };
    try std.testing.expectEqual(
        @as(?u32, 300),
        largestFittingWindow(8_000_000, 6_380_000, hybrid, 2, 1000),
    );
    // A budget past the refused window stops one under it.
    try std.testing.expectEqual(@as(?u32, 999), largestFittingWindow(8_000_000, 0, cost, 1, 1000));
    // The 8B at 1 GiB free agrees with `fitWindow`: 3584 tokens for one slot
    // and 1792 for two.
    try std.testing.expectEqual(
        @as(?u32, 3584),
        largestFittingWindow(1 << 30, 0, llama_8b_cost, 1, 8192),
    );
    try std.testing.expectEqual(
        @as(?u32, 1792),
        largestFittingWindow(1 << 30, 0, llama_8b_cost, 2, 8192),
    );
}

test "largestFittingWindow: a widest fit under the floor gives null" {
    // 100000 bytes buy 100 tokens, which fit but sit under the floor.
    const cost: KvCostModel = .{ .kv_bytes_per_token = 1000 };
    try std.testing.expect(windowFits(8_000_000, 6_900_000, cost, 1, 100));
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(8_000_000, 6_900_000, cost, 1, 1000),
    );
    // The 8B at 16 MiB free buys 56 tokens.
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(16 << 20, 0, llama_8b_cost, 1, 8192),
    );
}

test "largestFittingWindow: a refused window at or under the floor gives null" {
    const cost: KvCostModel = .{ .kv_bytes_per_token = 1000 };
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(8_000_000, 6_900_000, cost, 1, limits.min_kv_window),
    );
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(8_000_000, 6_900_000, cost, 1, 100),
    );
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(8_000_000, 0, cost, 1, limits.min_kv_window),
    );
    // One past the floor leaves the floor itself to suggest.
    try std.testing.expectEqual(
        @as(?u32, limits.min_kv_window),
        largestFittingWindow(8_000_000, 0, cost, 1, limits.min_kv_window + 1),
    );
}

test "largestFittingWindow: null when no window fits" {
    const cost: KvCostModel = .{ .kv_bytes_per_token = 1000 };
    // The reservation alone takes the usable share, or passes it.
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(8_000_000, 7_000_000, cost, 1, 1000),
    );
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(8_000_000, 7_000_001, cost, 1, 1000),
    );
    try std.testing.expectEqual(@as(?u32, null), largestFittingWindow(0, 0, cost, 1, 1000));
    // The fixed state alone passes the share of one slot.
    const hybrid: KvCostModel = .{
        .kv_bytes_per_token = 1000,
        .fixed_state_bytes_per_slot = 1_000_000,
    };
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(8_000_000, 6_500_000, hybrid, 1, 1000),
    );
}

test "largestFittingWindow: a model with no per-token cost gives null" {
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(1 << 40, 0, .{ .kv_bytes_per_token = 0 }, 1, 8192),
    );
    const state_only: KvCostModel = .{
        .kv_bytes_per_token = 0,
        .fixed_state_bytes_per_slot = 1 << 20,
    };
    try std.testing.expectEqual(
        @as(?u32, null),
        largestFittingWindow(1 << 40, 0, state_only, 4, 8192),
    );
}

fn checkLargestFit(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const cost: KvCostModel = .{
        .kv_bytes_per_token = r.intRangeAtMost(u64, 0, 64),
        .fixed_state_bytes_per_slot = r.intRangeAtMost(u64, 0, 4096),
    };
    const n_slots = r.intRangeAtMost(u32, 1, 4);
    const window = r.intRangeAtMost(u32, 1, 400);
    const reserved = r.intRangeAtMost(u64, 0, 1 << 20);
    // The spread covers budgets under the floor, inside the range, and
    // past the window.
    const top = reserved + @as(
        u64,
        n_slots,
    ) * (420 * cost.kv_bytes_per_token + cost.fixed_state_bytes_per_slot);
    const available = r.intRangeAtMost(u64, 0, 2 * top);
    const got = largestFittingWindow(available, reserved, cost, n_slots, window);
    if (cost.kv_bytes_per_token == 0) {
        try std.testing.expectEqual(@as(?u32, null), got);
        return;
    }

    var want: ?u32 = null;
    var w: u32 = window;
    while (w > limits.min_kv_window) {
        w -= 1;
        const kv = @as(
            u64,
            n_slots,
        ) * (@as(u64, w) * cost.kv_bytes_per_token + cost.fixed_state_bytes_per_slot);
        if (limits.vramPlanFits(reserved + kv, available)) {
            want = w;
            break;
        }
    }

    try std.testing.expectEqual(want, got);
}

test "largestFittingWindow: the result matches a scan down from the refused window" {
    try prop.forAll(
        .{ .site = 0x5EED_0032, .cases = 2000 },
        {},
        u64,
        prop.genSeed(void),
        checkLargestFit,
    );
}

test "kvPlan: the RAM split funds the blob ledger first and the prefix cache with the rest" {
    // Llama-8B at window 512 prices a slot at 128 MiB.
    try std.testing.expectEqual(@as(u64, 128 << 20), perSlotBytes(llama_8b_cost, 512));

    // 6 sessions over 4 slots evict 2, so the ledger reaches 3 blobs (384 MiB).
    const over = try kvPlan(no_vram_caps, llama_8b_cost, .{
        .slots = 4,
        .batch = 4,
        .window = 512,
        .y_bytes = 512 << 20,
        .n_logical = 6,
    });
    try std.testing.expectEqual(@as(u64, 384 << 20), over.ram.blobs);
    try std.testing.expectEqual(@as(u64, 128 << 20), over.ram.prefix);

    // A budget under that reach funds the ledger alone.
    const tight = try kvPlan(no_vram_caps, llama_8b_cost, .{
        .slots = 4,
        .batch = 4,
        .window = 512,
        .y_bytes = 256 << 20,
        .n_logical = 6,
    });
    try std.testing.expectEqual(@as(u64, 256 << 20), tight.ram.blobs);
    try std.testing.expectEqual(@as(u64, 0), tight.ram.prefix);

    // Every session sits in a slot, so no eviction can occur and the whole
    // budget funds the prefix cache.
    const resident = try kvPlan(no_vram_caps, llama_8b_cost, .{
        .slots = 4,
        .batch = 4,
        .window = 512,
        .y_bytes = 512 << 20,
        .n_logical = 4,
    });
    try std.testing.expectEqual(@as(u64, 0), resident.ram.blobs);
    try std.testing.expectEqual(@as(u64, 512 << 20), resident.ram.prefix);
}

test "kvPlan: the RAM split conserves Y, and both shares grow with it" {
    const logicals = [_]u32{ 4, 5, 6, 8 };
    // 17 budgets from zero to 1 GiB. The grid is fixed, so a failure replays.
    const step: u64 = 64 << 20;
    for (logicals) |n_logical| {
        const evictable = n_logical -| 4;
        const need: u64 = if (evictable == 0) 0 else @as(
            u64,
            evictable + 1,
        ) * perSlotBytes(llama_8b_cost, 512);

        var previous_blobs: u64 = 0;
        var previous_prefix: u64 = 0;
        for (0..17) |i| {
            const y: u64 = @as(u64, @intCast(i)) * step;
            const plan = try kvPlan(no_vram_caps, llama_8b_cost, .{
                .slots = 4,
                .batch = 4,
                .window = 512,
                .y_bytes = y,
                .n_logical = n_logical,
            });
            try std.testing.expectEqual(y, plan.ram.blobs + plan.ram.prefix);
            try std.testing.expect(plan.ram.blobs >= previous_blobs);
            try std.testing.expect(plan.ram.prefix >= previous_prefix);
            // The prefix cache takes nothing until the ledger holds its whole
            // reach.
            if (plan.ram.prefix > 0) {
                try std.testing.expectEqual(need, plan.ram.blobs);
            }

            previous_blobs = plan.ram.blobs;
            previous_prefix = plan.ram.prefix;
        }

        // The widest budget on the grid outgrows every reach on the list.
        try std.testing.expectEqual(need, previous_blobs);
    }
}

test "kvPlan: a half-stated split keeps the shipped reservation" {
    // No logical count: the whole budget funds the ledger, bit for bit as before.
    const no_count = try kvPlan(no_vram_caps, llama_8b_cost, .{
        .slots = 4,
        .batch = 4,
        .window = 512,
        .y_bytes = 512 << 20,
    });
    try std.testing.expectEqual(@as(u64, 512 << 20), no_count.ram.blobs);
    try std.testing.expectEqual(@as(u64, 0), no_count.ram.prefix);

    // A logical count with no budget has nothing to split.
    const no_budget = try kvPlan(no_vram_caps, llama_8b_cost, .{
        .slots = 4,
        .batch = 4,
        .window = 512,
        .n_logical = 6,
    });
    try std.testing.expectEqual(@as(u64, 0), no_budget.ram.blobs);
    try std.testing.expectEqual(@as(u64, 0), no_budget.ram.prefix);
}

test "kvPlan mode A: the RAM split reads the budget-derived slot count" {
    // The 5 GiB anchor buys 40 slots at window 512, so 48 sessions evict 8 and
    // the ledger reaches 9 blobs (1152 MiB of the 2 GiB budget).
    const plan = try kvPlan(no_vram_caps, llama_8b_cost, .{
        .x_bytes = 5 * 1024 * 1024 * 1024,
        .window = 512,
        .batch = 8,
        .y_bytes = 2 << 30,
        .n_logical = 48,
    });
    try std.testing.expectEqual(@as(u32, 40), plan.n_slots);
    try std.testing.expectEqual(@as(u64, 9 * (128 << 20)), plan.ram.blobs);
    try std.testing.expectEqual(@as(u64, (2 << 30) - 9 * (128 << 20)), plan.ram.prefix);
    try std.testing.expectEqual(@as(u64, 2 << 30), plan.ram.blobs + plan.ram.prefix);
}

test "planLayerSplit: the three split anchors" {
    // 56000 free bytes give 49000 usable. The suffix `[8..32)` spends
    // `2000 * 24 = 48000` and fits. The suffix `[7..32)` spends
    // `2000 * 25 = 50000` and does not.
    const tight = try planLayerSplit(
        anchorCaps(56_000),
        &anchor_weights,
        0,
        anchor_kv,
        unchecked_serve,
        null,
    );
    try std.testing.expectEqual(@as(u32, 8), tight.host_layers);
    try std.testing.expectEqual(@as(u64, 24_000), tight.weight_resident);
    try std.testing.expectEqual(@as(u64, 24_000), tight.kv_vram);
    try std.testing.expectEqual(@as(u64, 8_000), tight.kv_host);

    // A pinned 32 GPU layers asks for `k = 0`, which spends `2000 * 32 = 64000`
    // against the same 49000 usable, so the planner refuses.
    try std.testing.expectError(error.WeightsVramBudgetTooSmall, planLayerSplit(
        anchorCaps(56_000),
        &anchor_weights,
        0,
        anchor_kv,
        unchecked_serve,
        32,
    ));

    // 80000 free bytes give 70000 usable, which funds `2000 * 32 = 64000` whole.
    const roomy = try planLayerSplit(
        anchorCaps(80_000),
        &anchor_weights,
        0,
        anchor_kv,
        unchecked_serve,
        null,
    );
    try std.testing.expectEqual(@as(u32, 0), roomy.host_layers);
    try std.testing.expectEqual(@as(u64, 32_000), roomy.weight_resident);
    try std.testing.expectEqual(@as(u64, 0), roomy.kv_host);
}

// The RAM anchors: Llama-3.1-8B Q8_0 at a 131072-token window and one slot.
// Every one of the 32 layers holds 231768064 weight bytes and charges 4096 f16
// KV bytes a token, and the head holds 558186496 bytes. The device has 15 GiB
// free and the host has 22 GiB free, which leaves 20669530112 usable bytes.
const ram_anchor_weights = [_]u64{231_768_064} ** 32;
const ram_anchor_head: u64 = 558_186_496;
const ram_anchor_free: u64 = 23_622_320_128;

fn ramAnchorCaps() Caps {
    return .{
        .vram_free = 16_106_127_360,
        .weights = 0,
        .logits = 513_024,
        .scratch = 0,
        .model_ctx = 131_072,
        .serve_default = limits.min_kv_window,
        .default_ceil = limits.min_kv_window,
    };
}

fn ramAnchorKv(window: u32) KvStructure {
    return .{ .n_slots = 1, .window = window, .costs = KvCostVector.uniform(32, 4_096) };
}

test "quoteRam: the RAM anchors, a run and a serve that fit, and a small host that refuses" {
    const caps = ramAnchorCaps();
    try std.testing.expectEqual(@as(u64, 20_669_530_112), limits.ramUsable(ram_anchor_free));
    // The VRAM planner picks the split and checks no RAM. The quote then
    // funds or refuses that split.
    const vram_host: HostRequest = .{ .ram_split = no_ram_split, .residency = false };
    const vram = try planLayerSplit(
        caps,
        &ram_anchor_weights,
        ram_anchor_head,
        ramAnchorKv(131_072),
        vram_host,
        null,
    );
    try std.testing.expectEqual(@as(u32, 15), vram.host_layers);
    try std.testing.expectEqual(@as(?u64, null), vram.ram_usable);

    // The run placement: the stages price zero, and the host KV alone is the
    // spend.
    const run_host: HostRequest = .{ .ram_split = no_ram_split, .residency = false };
    const run = try quoteRam(
        ramAnchorKv(131_072),
        vram.host_layers,
        run_host,
        1,
        false,
        ram_anchor_free,
    );
    try std.testing.expectEqual(@as(u64, 8_053_063_680), run.ram.host_kv);
    try std.testing.expectEqual(@as(u64, 0), run.ram.swap_stages);
    try std.testing.expectEqual(@as(u64, 8_053_063_680), run.ram.spend());
    try std.testing.expectEqual(@as(?u64, 20_669_530_112), run.usable);
    const run_plan = vram.priced(run);
    try std.testing.expectEqual(@as(?u64, 20_669_530_112), run_plan.ram_usable);

    // The serve placement: two 256 MiB chunks replace two 8704 MiB device
    // halves, so the 1 GiB prefix and the stage pair fit beside the host KV.
    const serve_host: HostRequest = .{
        .ram_split = .{ .blobs = 0, .prefix = 1_073_741_824 },
        .residency = true,
    };
    const serve = try quoteRam(ramAnchorKv(131_072), 15, serve_host, 1, false, ram_anchor_free);
    try std.testing.expectEqual(@as(u64, 2 * limits.swap_stage_chunk_bytes), serve.ram.swap_stages);
    try std.testing.expectEqual(
        @as(u64, 8_053_063_680 + 536_870_912 + 1_073_741_824),
        serve.ram.spend(),
    );

    // A host with 8 GiB free: 7168 MiB usable. The constant stage pair
    // and the prefix leave 5632 MiB for 61440 host bytes a position, which
    // funds 96119 positions, and one more refuses. The printed fit keeps
    // 128 MiB of headroom and whole granules: 5504 MiB funds 93934, so
    // 93696.
    const small = serve_host;
    const small_free: u64 = 8 << 30;
    try std.testing.expectError(
        error.HostRamBudgetTooSmall,
        quoteRam(ramAnchorKv(131_072), 15, small, 1, false, small_free),
    );
    const refusal = ramRefusal(ramAnchorKv(131_072), 15, small, small_free);
    try std.testing.expectEqual(@as(u64, 8_053_063_680), refusal.host_kv);
    try std.testing.expectEqual(@as(u64, 536_870_912), refusal.swap_stages);
    try std.testing.expectEqual(@as(u64, 9_663_676_416), refusal.spend);
    try std.testing.expectEqual(@as(u64, 7_516_192_768), refusal.usable);
    try std.testing.expectEqual(@as(u32, 17), refusal.gpu_layers);
    try std.testing.expectEqual(@as(u32, 32), refusal.n_layers);
    try std.testing.expectEqual(@as(?u32, 93_696), refusal.fit);
    const advised = try quoteRam(ramAnchorKv(93_696), 15, small, 1, false, small_free);
    try std.testing.expectEqual(@as(u64, 7_367_294_976), advised.ram.spend());
    // The check stays exact at the edge.
    const edge = try quoteRam(ramAnchorKv(96_119), 15, small, 1, false, small_free);
    try std.testing.expectEqual(@as(u64, 7_516_164_096), edge.ram.spend());
    try std.testing.expectError(
        error.HostRamBudgetTooSmall,
        quoteRam(ramAnchorKv(96_120), 15, small, 1, false, small_free),
    );
    try std.testing.expectEqual(
        @as(u64, 7_516_225_536),
        ramRefusal(ramAnchorKv(96_120), 15, small, small_free).spend,
    );

    // An unknown free RAM quotes the refused vector unchanged.
    const open = try quoteRam(ramAnchorKv(131_072), 15, small, 1, false, null);
    try std.testing.expectEqual(@as(u64, 9_663_676_416), open.ram.spend());
    try std.testing.expectEqual(@as(?u64, null), open.usable);
}

// The blob anchors use these figures, in MiB: 32 layers of 512 MiB a slot at
// 131072 positions, 15 host layers, a 1024 MiB prefix, a 64 MiB chunk, and
// 19712 MiB usable. One blob is 16384 MiB.
const blob_anchor_prefix: KvPlan.Ram = .{ .blobs = 0, .prefix = 1 << 30 };

/// This host holds a pool exactly when a session is evictable. A serve pool
/// also prices the stages with no evictable session, because it snapshots
/// prefixes through them.
fn blobAnchorHost(n_logical: u32) HostRequest {
    return .{ .ram_split = blob_anchor_prefix, .residency = n_logical > 1 };
}

test "quoteRam: the blob anchors, E = 0 spends 8704 MiB and E = 1 refuses with both fits" {
    const mib: u64 = 1 << 20;
    // The anchors state a 64 MiB chunk, so this test replays them at it.
    const model_chunk: u64 = 64 << 20;
    const kv = ramAnchorKv(131_072);
    const lone = try quoteRamAt(kv, 15, blobAnchorHost(1), 1, false, ram_anchor_free, model_chunk);
    try std.testing.expectEqual(BlobTier.absent, lone.tier);
    try std.testing.expectEqual(@as(u64, 0), lone.ram.blobs);
    try std.testing.expectEqual(@as(u64, 0), lone.ram.swap_stages);
    try std.testing.expectEqual(8704 * mib, lone.ram.spend());
    try std.testing.expectEqual(16_384 * mib, lone.blob);
    try std.testing.expectEqual(@as(u64, 0), lone.need);
    // A serve pool snapshots prefixes through the stages at any session count,
    // so the same boot with a pool spends the stage pair as well.
    var pooled = blobAnchorHost(1);
    pooled.residency = true;
    const served = try quoteRamAt(kv, 15, pooled, 1, false, ram_anchor_free, model_chunk);
    try std.testing.expectEqual(BlobTier.absent, served.tier);
    try std.testing.expectEqual(8832 * mib, served.ram.spend());

    try std.testing.expectError(
        error.BlobRamBudgetTooSmall,
        quoteRamAt(kv, 15, blobAnchorHost(2), 2, false, ram_anchor_free, model_chunk),
    );
    const r = blobRefusalAt(kv, 15, blobAnchorHost(2), 2, ram_anchor_free, model_chunk);
    try std.testing.expectEqual(32_768 * mib, r.need);
    try std.testing.expectEqual(10_880 * mib, r.left);
    try std.testing.expectEqual(16_384 * mib, r.blob);
    // spend(w) = (15 + 2 * 32) * 4096 * w + 1152 MiB, so the widest w is
    // (19712 - 1152) MiB / 323584 bytes = 60143. The printed fit keeps 128 MiB
    // of headroom, 59729, in whole granules: 59648.
    try std.testing.expectEqual(
        @as(?u32, 60_143),
        ramFitWindow(kv, 15, 2, 1152 * mib, limits.ramUsable(ram_anchor_free)),
    );
    try std.testing.expectEqual(@as(?u32, 59_648), r.fit_window);
    // 10880 MiB funds no blob of 16384 MiB, so the text names the slot count.
    try std.testing.expectEqual(@as(?u32, null), r.fit_sessions);
    var buf: [blob_refusal_text_max]u8 = undefined;
    try std.testing.expectEqualStrings(
        "ram budget: the blob tier needs 32768 MiB of RAM, but the plan leaves only 10880 MiB." ++
            " One evicted session holds up to 16384 MiB. The plan serves 2 sessions on 1 " ++
            "slot at 131072 positions." ++
            " Use --max-seq 59648 or less. Or use --max-sessions 1.",
        formatBlobRefusal(&buf, r),
    );

    // The exact fit boots, and one more position refuses.
    const at_fit = try quoteRamAt(
        ramAnchorKv(60_143),
        15,
        blobAnchorHost(2),
        2,
        false,
        ram_anchor_free,
        model_chunk,
    );
    try std.testing.expectEqual(BlobTier{ .bounded = at_fit.need }, at_fit.tier);
    try std.testing.expect(at_fit.ram.spend() <= limits.ramUsable(ram_anchor_free));
    try std.testing.expectError(
        error.BlobRamBudgetTooSmall,
        quoteRamAt(
            ramAnchorKv(60_144),
            15,
            blobAnchorHost(2),
            2,
            false,
            ram_anchor_free,
            model_chunk,
        ),
    );
}

test "quoteRam: the three tiers stay distinct, and a named budget stays as named" {
    const kv = ramAnchorKv(32_768);
    // One blob at 32768 positions is 4 GiB, and one evictable session needs two.
    const derived = try quoteRam(kv, 15, blobAnchorHost(2), 2, false, ram_anchor_free);
    try std.testing.expectEqual(BlobTier{ .bounded = 8 << 30 }, derived.tier);
    try std.testing.expectEqual(@as(u64, 8 << 30), derived.ram.blobs);
    try std.testing.expectEqual(@as(?u64, 8 << 30), derived.tier.budget());

    // An unknown free RAM leaves an unset flag unenforced, with no blob share.
    const open = try quoteRam(kv, 15, blobAnchorHost(2), 2, false, null);
    try std.testing.expectEqual(BlobTier.unenforced, open.tier);
    try std.testing.expectEqual(@as(?u64, null), open.tier.budget());
    try std.testing.expectEqual(@as(u64, 0), open.ram.blobs);
    try std.testing.expectEqual(@as(?u64, null), open.usable);

    // A named share below the need stays as named. The caller warns.
    var named = blobAnchorHost(2);
    named.ram_split.blobs = 1 << 30;
    const below = try quoteRam(kv, 15, named, 2, true, ram_anchor_free);
    try std.testing.expectEqual(BlobTier{ .bounded = 1 << 30 }, below.tier);
    try std.testing.expect(below.ram.blobs < below.need);

    // No evictable session is absent at any budget, and an absent tier
    // enforces zero bytes.
    const lone = try quoteRam(kv, 15, blobAnchorHost(1), 1, false, null);
    try std.testing.expectEqual(BlobTier.absent, lone.tier);
    try std.testing.expectEqual(@as(?u64, 0), lone.tier.budget());
    // A bounded tier of zero bytes is a budget, not the unenforced tier.
    try std.testing.expectEqual(@as(?u64, 0), (BlobTier{ .bounded = 0 }).budget());
}

test "quoteRam: the CPU end and the fused end are the two ends of one vector" {
    const kv = ramAnchorKv(4_096);
    const n_layers = kv.costs.n_layers;
    const whole_host = kvHostBytes(kv, n_layers);
    // The CPU placement holds every layer on the host and stages nothing.
    const cpu = try quoteRam(kv, n_layers, blobAnchorHost(2), 2, false, null);
    try std.testing.expectEqual(@as(u64, 0), cpu.ram.swap_stages);
    try std.testing.expectEqual(whole_host, cpu.ram.host_kv);
    // The fused placement holds no host KV and stages one chunk or one slot each way.
    const fused = try quoteRam(kv, 0, blobAnchorHost(2), 2, false, null);
    try std.testing.expectEqual(@as(u64, 0), fused.ram.host_kv);
    try std.testing.expectEqual(
        2 * stageBytes(slotSpanBytes(kv, 0, n_layers)),
        fused.ram.swap_stages,
    );
}

const QuoteSweep = struct {
    bounded: u32 = 0,
    blob_refused: u32 = 0,
    window_fits: u32 = 0,
    session_fits: u32 = 0,
};

fn checkQuoteSweep(sweep: *QuoteSweep, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const stack = sweepStack(random, random.boolean());
    const n = stack.kv.costs.n_layers;
    const k = random.intRangeAtMost(u32, 0, n);
    const n_logical = random.intRangeAtMost(u32, 1, 3 * stack.kv.n_slots);
    const residency = n_logical > stack.kv.n_slots;
    const ram_free = random.intRangeAtMost(u64, 0, 1 << 36);
    const host: HostRequest = .{
        .ram_split = .{ .blobs = 0, .prefix = random.intRangeAtMost(u64, 0, 1 << 30) },
        .residency = residency,
    };
    const usable_ram = limits.ramUsable(ram_free);

    // The stage pair is the same at any window past the chunk, and at
    // most two chunks at every window.
    const stages = ramAt(stack.kv, k, n, host).swap_stages;
    try std.testing.expect(stages <= 2 * limits.swap_stage_chunk_bytes);
    var wide = stack.kv;
    wide.window = random.intRangeAtMost(u32, stack.kv.window, limits.max_model_ctx);
    const wide_stages = ramAt(wide, k, n, host).swap_stages;
    try std.testing.expect(wide_stages >= stages);
    if (stages == 2 * limits.swap_stage_chunk_bytes) {
        try std.testing.expectEqual(stages, wide_stages);
    }

    if (quoteRam(stack.kv, k, host, n_logical, false, ram_free)) |q| {
        try std.testing.expect(q.ram.spend() <= usable_ram);
        switch (q.tier) {
            .absent => try std.testing.expect(!residency),
            .unenforced => unreachable,
            .bounded => |bytes| {
                sweep.bounded += 1;
                try std.testing.expectEqual(q.need, bytes);
                try std.testing.expectEqual(
                    @as(u64, n_logical - stack.kv.n_slots + 1) * q.blob,
                    bytes,
                );
            },
        }
    } else |err| {
        switch (err) {
            error.BlobRamBudgetTooSmall => {
                sweep.blob_refused += 1;
                try std.testing.expect(residency);
                const r = blobRefusal(stack.kv, k, host, n_logical, ram_free);
                try std.testing.expect(r.left < r.need);
                // Each printed fit boots, and it still boots after the free
                // RAM falls far enough to cut 64 MiB of usable RAM.
                const lower = ram_free -| lowered_free_bytes;
                try std.testing.expect(
                    usable_ram - limits.ramUsable(lower) >= 64 << 20 or lower == 0,
                );
                if (r.fit_window) |fit| {
                    sweep.window_fits += 1;
                    var narrow = stack.kv;
                    narrow.window = fit;
                    _ = try quoteRam(narrow, k, host, n_logical, false, ram_free);
                    _ = try quoteRam(narrow, k, host, n_logical, false, lower);
                }

                if (r.fit_sessions) |m| {
                    sweep.session_fits += 1;
                    var fewer = host;
                    fewer.residency = m > stack.kv.n_slots;
                    _ = try quoteRam(stack.kv, k, fewer, m, false, ram_free);
                    _ = try quoteRam(stack.kv, k, fewer, m, false, lower);
                }
            },
            // No blob is funded, so only the rest of the vector refuses.
            error.HostRamBudgetTooSmall => try std.testing.expect(!residency),
        }
    }
}

test "quoteRam: over 2048 seeded stacks a derived spend fits and a refusal has a reason" {
    var sweep: QuoteSweep = .{};
    try prop.forAll(
        .{ .site = 0x5eed_0b10, .cases = 2048 },
        &sweep,
        u64,
        prop.genSeed(*QuoteSweep),
        checkQuoteSweep,
    );
    try std.testing.expect(sweep.bounded > 0);
    try std.testing.expect(sweep.blob_refused > 0);
    try std.testing.expect(sweep.window_fits > 0);
    try std.testing.expect(sweep.session_fits > 0);
}

/// A fall in free RAM that cuts at least 64 MiB of usable RAM: 74 MiB of free
/// RAM times 7/8 is 64.75 MiB. The printed fits keep 128 MiB of headroom, so
/// they still boot after it.
const lowered_free_bytes: u64 = 74 << 20;

test "ramFitWindow: the stage pair is a constant term, so the fit never moves with it" {
    const kv = ramAnchorKv(131_072);
    // The split proofs state a 64 MiB chunk, so the stage pair here is
    // their 128 MiB.
    const stages: u64 = 2 * (64 << 20);
    // On a host with 8 GiB free, only the host prefix grows with the window:
    // 15 layers of 4096 bytes.
    const usable_ram = limits.ramUsable(8 << 30);
    try std.testing.expectEqual(@as(?u32, 120_149), ramFitWindow(kv, 15, 0, stages, usable_ram));
    // A larger constant lowers the fit by its bytes over the per-position
    // price, and adds no per-position term.
    try std.testing.expectEqual(
        @as(?u32, 102_673),
        ramFitWindow(kv, 15, 0, stages + (1 << 30), usable_ram),
    );
    // Two growing blobs add the whole model per position.
    try std.testing.expectEqual(
        @as(?u32, 60_143),
        ramFitWindow(kv, 15, 2, stages + (1 << 30), limits.ramUsable(ram_anchor_free)),
    );
    // A constant past the usable RAM funds no window.
    try std.testing.expectEqual(
        @as(?u32, null),
        ramFitWindow(kv, 15, 0, usable_ram + 1, usable_ram),
    );
}

test "ChunkWalk: the worked example, three chunks and a short tail" {
    const expect = [_]Chunk{
        .{ .offset = 0, .len = 4 },
        .{ .offset = 4, .len = 4 },
        .{ .offset = 8, .len = 2 },
    };
    var walk = ChunkWalk.init(10, 4);
    for (expect) |want| {
        try std.testing.expectEqual(want, walk.next().?);
    }

    try std.testing.expectEqual(@as(?Chunk, null), walk.next());
    try std.testing.expectEqual(@as(u64, 3), ChunkWalk.count(10, 4));

    var even = ChunkWalk.init(192, 64);
    var i: u64 = 0;
    while (even.next()) |c| : (i += 1) {
        try std.testing.expectEqual(Chunk{ .offset = 64 * i, .len = 64 }, c);
    }

    try std.testing.expectEqual(@as(u64, 3), i);
    // An empty blob walks nothing.
    var empty = ChunkWalk.init(0, 64);
    try std.testing.expectEqual(@as(?Chunk, null), empty.next());
    try std.testing.expectEqual(@as(u64, 0), ChunkWalk.count(0, 64));
}

/// A copy record with the fields `clipCopies` reads. The two handles name a
/// device region and the stage.
const TestCopy = struct {
    src: u32,
    dst: u32,
    size: u64,
    src_offset: u64 = 0,
    dst_offset: u64 = 0,
};

const test_stage_handle: u32 = 99;
const test_region_bytes: usize = 1024;

/// The whole-blob copy list over `sizes`, one region per size, in blob order.
/// Each region starts at a distinct device offset, so a wrong device offset
/// moves the wrong bytes.
fn testFullCopies(sizes: []const u64, to_host: bool, out: []TestCopy) []TestCopy {
    var at: u64 = 0;
    for (sizes, 0..) |size, i| {
        const region: u32 = @intCast(i);
        const dev: u64 = 3 + i;
        out[i] = if (to_host)
            .{
                .src = region,
                .src_offset = dev,
                .dst = test_stage_handle,
                .dst_offset = at,
                .size = size,
            }
        else
            .{
                .src = test_stage_handle,
                .src_offset = at,
                .dst = region,
                .dst_offset = dev,
                .size = size,
            };
        at += size;
    }

    return out[0..sizes.len];
}

/// Save `device` into a blob through a stage of `chunk` bytes, then restore the
/// blob into a fresh device, and check both against the whole-blob copy.
fn testChunkedSwap(sizes: []const u64, chunk: u64) !void {
    var total: u64 = 0;
    for (sizes) |size| {
        total += size;
    }

    var device: [16][test_region_bytes]u8 = undefined;
    for (&device, 0..) |*region, r| {
        for (region, 0..) |*b, j| {
            b.* = @truncate(r * 131 + j * 7 + 1);
        }
    }

    // The whole-blob save is the oracle.
    var full_buf: [16]TestCopy = undefined;
    var whole: [4096]u8 = undefined;
    for (testFullCopies(sizes, true, &full_buf)) |c| {
        @memcpy(whole[c.dst_offset..][0..c.size], device[c.src][c.src_offset..][0..c.size]);
    }

    // The chunked save: each chunk lands in the stage at offset zero, then
    // moves to its offset of the blob.
    var blob: [4096]u8 = @splat(0);
    var stage: [4096]u8 = undefined;
    var clipped_buf: [16]TestCopy = undefined;
    var covered: u64 = 0;
    var walk = ChunkWalk.init(total, chunk);
    while (walk.next()) |piece| {
        try std.testing.expectEqual(covered, piece.offset);
        const pieces = clipCopies(
            TestCopy,
            testFullCopies(sizes, true, &full_buf),
            true,
            piece.offset,
            piece.len,
            &clipped_buf,
        );
        var stage_at: u64 = 0;
        for (pieces) |c| {
            try std.testing.expectEqual(test_stage_handle, c.dst);
            try std.testing.expectEqual(stage_at, c.dst_offset);
            try std.testing.expect(c.dst_offset + c.size <= chunk);
            @memcpy(stage[c.dst_offset..][0..c.size], device[c.src][c.src_offset..][0..c.size]);
            stage_at += c.size;
        }

        try std.testing.expectEqual(piece.len, stage_at);
        @memcpy(blob[piece.offset..][0..piece.len], stage[0..piece.len]);
        covered += piece.len;
    }

    try std.testing.expectEqual(total, covered);
    try std.testing.expectEqualSlices(u8, whole[0..total], blob[0..total]);

    // The chunked restore writes every region byte back where it came from.
    var restored: [16][test_region_bytes]u8 = @splat(@splat(0));
    walk = ChunkWalk.init(total, chunk);
    while (walk.next()) |piece| {
        @memcpy(stage[0..piece.len], blob[piece.offset..][0..piece.len]);
        const pieces = clipCopies(
            TestCopy,
            testFullCopies(sizes, false, &full_buf),
            false,
            piece.offset,
            piece.len,
            &clipped_buf,
        );
        for (pieces) |c| {
            try std.testing.expectEqual(test_stage_handle, c.src);
            @memcpy(restored[c.dst][c.dst_offset..][0..c.size], stage[c.src_offset..][0..c.size]);
        }
    }

    for (sizes, 0..) |size, r| {
        const dev = 3 + r;
        try std.testing.expectEqualSlices(
            u8,
            device[r][dev..][0..size],
            restored[r][dev..][0..size],
        );
    }
}

test "clipCopies: chunks tile the blob once, in order, on Llama rows and on the Qwen GDN regions" {
    // A Llama slot: four layers of K and V rows, 72 bytes a region.
    const llama = [_]u64{72} ** 8;
    // A Qwen slot: three GDN layers of a state and a conv region, then one
    // full-attention layer of K and V rows, twice over.
    const qwen = [_]u64{ 300, 40, 300, 40, 300, 40, 72, 72, 300, 40, 300, 40, 300, 40, 72, 72 };
    const chunks = [_]u64{ 1, 7, 64, 72, 100, 1000, 4096 };
    for (chunks) |chunk| {
        try testChunkedSwap(&llama, chunk);
        try testChunkedSwap(&qwen, chunk);
    }
}

test "planLayerSplit: the auto split is maximal and monotone, and every plan conserves KV" {
    const layer_counts = [_]u32{ 1, 8, 32, limits.max_layers };
    // 33 budgets from zero to 160000 bytes. The grid is fixed, so a failure
    // replays.
    const step: u64 = 5_000;
    for (layer_counts) |n| {
        var table: [limits.max_layers]u64 = undefined;
        // A ramp, so a wrong suffix sum cannot pass by symmetry.
        for (table[0..n], 0..) |*w, i| {
            w.* = 1_000 + 100 * @as(u64, i);
        }

        const weights = table[0..n];
        const kv: KvStructure = .{
            .n_slots = 2,
            .window = 100,
            .costs = KvCostVector.uniform(n, 10),
        };
        const total_kv: u64 = @as(u64, kv.n_slots) * perSlotBytes(kv.costs.total(), kv.window);

        var previous_split: u32 = n;
        for (0..33) |i| {
            const caps = anchorCaps(@as(u64, @intCast(i)) * step);
            const plan = try planLayerSplit(caps, weights, 0, kv, unchecked_serve, null);
            try std.testing.expectEqual(n, plan.n_layers);
            try std.testing.expectEqual(total_kv, plan.kv_vram + plan.kv_host);
            try std.testing.expect(plan.spend() <= usable(caps));
            try std.testing.expect(layerSplitFits(caps, weights, 0, kv, plan.host_layers));
            if (plan.host_layers > 0) {
                try std.testing.expect(!layerSplitFits(caps, weights, 0, kv, plan.host_layers - 1));
            }

            // A wider budget never moves a layer back to the host.
            try std.testing.expect(plan.host_layers <= previous_split);
            previous_split = plan.host_layers;

            // A pinned count lands on exactly its suffix, or refuses exactly
            // when that suffix does not fit.
            var g: u32 = 0;
            while (g <= n) : (g += 1) {
                const k = n - g;
                if (planLayerSplit(caps, weights, 0, kv, unchecked_serve, g)) |pin| {
                    try std.testing.expectEqual(k, pin.host_layers);
                    try std.testing.expect(layerSplitFits(caps, weights, 0, kv, k));
                    try std.testing.expectEqual(total_kv, pin.kv_vram + pin.kv_host);
                    try std.testing.expect(pin.spend() <= usable(caps));
                } else |err| {
                    try std.testing.expectEqual(error.WeightsVramBudgetTooSmall, err);
                    try std.testing.expect(!layerSplitFits(caps, weights, 0, kv, k));
                }
            }
        }

        // A zero budget puts every layer on the host, and a roomy one puts none.
        const starved = try planLayerSplit(anchorCaps(0), weights, 0, kv, unchecked_serve, null);
        try std.testing.expectEqual(n, starved.host_layers);
        try std.testing.expectEqual(@as(u64, 0), starved.spend());
        const roomy = try planLayerSplit(
            anchorCaps(1 << 30),
            weights,
            0,
            kv,
            unchecked_serve,
            null,
        );
        try std.testing.expectEqual(@as(u32, 0), roomy.host_layers);

        // The same grid on a ramp vector. A planner that multiplies one price by a layer count
        // moves the split here.
        const ramp: KvStructure = .{ .n_slots = 2, .window = 100, .costs = rampCosts(n) };
        const ramp_total: u64 = @as(u64, ramp.n_slots) * perSlotBytes(
            ramp.costs.total(),
            ramp.window,
        );
        var ramp_previous: u32 = n;
        for (0..33) |i| {
            const caps = anchorCaps(@as(u64, @intCast(i)) * step);
            const plan = try planLayerSplit(caps, weights, 0, ramp, unchecked_serve, null);
            try std.testing.expectEqual(ramp_total, plan.kv_vram + plan.kv_host);
            try std.testing.expect(plan.spend() <= usable(caps));
            try std.testing.expect(layerSplitFits(caps, weights, 0, ramp, plan.host_layers));
            if (plan.host_layers > 0) {
                try std.testing.expect(
                    !layerSplitFits(caps, weights, 0, ramp, plan.host_layers - 1),
                );
            }

            try std.testing.expect(plan.host_layers <= ramp_previous);
            ramp_previous = plan.host_layers;
        }
    }
}

test "planLayerSplit: the KV structure and the caps reservations pass through" {
    var caps = anchorCaps(1 << 20);
    caps.weights = 9_999;
    caps.logits = 4_096;
    caps.scratch = 2_048;
    const kv: KvStructure = .{ .n_slots = 3, .window = 128, .costs = KvCostVector.uniform(32, 16) };
    const plan = try planLayerSplit(caps, &anchor_weights, 0, kv, unchecked_serve, null);
    try std.testing.expectEqual(@as(u64, 4_096), plan.logits);
    try std.testing.expectEqual(@as(u64, 2_048), plan.scratch);
    try std.testing.expectEqual(@as(u32, 3), plan.n_slots);
    try std.testing.expectEqual(@as(u32, 128), plan.window);
    try std.testing.expectEqual(@as(u32, 32), plan.n_layers);
    // The per-layer vector replaces the scalar weight total, so `caps.weights`
    // never reaches the plan.
    try std.testing.expectEqual(@as(u64, 32_000), plan.weight_resident);

    // A pure CPU placement reserves no device bytes, the way `pinned` does, and
    // the whole KV lands on the host ledger.
    const cpu = try planLayerSplit(
        anchorCaps(0),
        &anchor_weights,
        0,
        anchor_kv,
        unchecked_serve,
        0,
    );
    try std.testing.expectEqual(@as(u32, 32), cpu.host_layers);
    try std.testing.expectEqual(@as(u64, 0), cpu.weight_resident);
    try std.testing.expectEqual(@as(u64, 0), cpu.kv_vram);
    try std.testing.expectEqual(@as(u64, 32_000), cpu.kv_host);
    try std.testing.expectEqual(@as(u64, 0), cpu.spend());
}

test "planLayerSplit: the head anchors, with the head bytes on the device" {
    // The same fixture under a 1500-byte head. The spend at split `k` is
    // `2000 * (32 - k) + 1500`, so 49000 usable funds `k = 9` (47500) and
    // refuses `k = 8` (49500).
    const head: u64 = 1_500;
    const auto = try planLayerSplit(
        anchorCaps(56_000),
        &anchor_weights,
        head,
        anchor_kv,
        unchecked_serve,
        null,
    );
    try std.testing.expectEqual(@as(u32, 9), auto.host_layers);
    try std.testing.expectEqual(@as(u64, 24_500), auto.weight_resident);
    try std.testing.expectEqual(@as(u64, 23_000), auto.kv_vram);
    try std.testing.expectEqual(@as(u64, 47_500), auto.spend());

    // A pinned 24 GPU layers asks for `k = 8`, which the same budget refuses.
    try std.testing.expectError(error.WeightsVramBudgetTooSmall, planLayerSplit(
        anchorCaps(56_000),
        &anchor_weights,
        head,
        anchor_kv,
        unchecked_serve,
        24,
    ));

    // A pinned 23 GPU layers lands on `k = 9`: the 23000-byte suffix plus the
    // head.
    const narrow = try planLayerSplit(
        anchorCaps(56_000),
        &anchor_weights,
        head,
        anchor_kv,
        unchecked_serve,
        23,
    );
    try std.testing.expectEqual(@as(u32, 9), narrow.host_layers);
    try std.testing.expectEqual(@as(u64, 24_500), narrow.weight_resident);
}

test "planLayerSplit: the head bytes are charged exactly when a layer is resident" {
    const head: u64 = 4_000;
    // 33 budgets from zero to 160000 bytes. The grid is fixed, so a failure
    // replays.
    const step: u64 = 5_000;
    for (0..33) |i| {
        const caps = anchorCaps(@as(u64, @intCast(i)) * step);
        const plan = try planLayerSplit(
            caps,
            &anchor_weights,
            head,
            anchor_kv,
            unchecked_serve,
            null,
        );
        const suffix = suffixWeights(&anchor_weights, plan.host_layers);
        if (plan.host_layers < plan.n_layers) {
            try std.testing.expectEqual(suffix + head, plan.weight_resident);
        } else {
            try std.testing.expectEqual(suffix, plan.weight_resident);
            try std.testing.expectEqual(@as(u64, 0), plan.spend());
        }

        try std.testing.expect(plan.spend() <= usable(caps));
    }

    // Both sides of the boundary, pinned under a roomy budget: the whole-host
    // split charges no head, and one GPU layer charges it whole.
    const host = try planLayerSplit(
        anchorCaps(1 << 30),
        &anchor_weights,
        head,
        anchor_kv,
        unchecked_serve,
        0,
    );
    try std.testing.expectEqual(@as(u64, 0), host.weight_resident);
    const one = try planLayerSplit(
        anchorCaps(1 << 30),
        &anchor_weights,
        head,
        anchor_kv,
        unchecked_serve,
        1,
    );
    try std.testing.expectEqual(@as(u64, 1_000) + head, one.weight_resident);
}

test "quoteLayerSplit: the price it names is the price the plan pays" {
    const head: u64 = 1_500;
    const caps = anchorCaps(56_000);
    const funded = quoteLayerSplit(caps, &anchor_weights, head, anchor_kv, 23);
    const plan = try planLayerSplit(caps, &anchor_weights, head, anchor_kv, unchecked_serve, 23);
    try std.testing.expectEqual(plan.spend(), funded.spend);
    try std.testing.expectEqual(@as(u64, 49_000), funded.usable);
    try std.testing.expect(funded.spend <= funded.usable);

    // One layer wider refuses, and the quote names both sides of the refusal.
    const over = quoteLayerSplit(caps, &anchor_weights, head, anchor_kv, 24);
    try std.testing.expectEqual(@as(u64, 49_500), over.spend);
    try std.testing.expect(over.spend > over.usable);
    try std.testing.expectError(error.WeightsVramBudgetTooSmall, planLayerSplit(
        caps,
        &anchor_weights,
        head,
        anchor_kv,
        unchecked_serve,
        24,
    ));

    // The whole-host pin holds no device context, whatever the head costs.
    const none = quoteLayerSplit(caps, &anchor_weights, head, anchor_kv, 0);
    try std.testing.expectEqual(@as(u64, 0), none.spend);
}

test "formatLayerSplit: the three span forms" {
    var buf: [layer_split_text_max]u8 = undefined;
    const mixed: LayerSplit = .{
        .host_layers = 24,
        .n_layers = 32,
        .weight_resident = 8_000,
        .kv_vram = 8_000,
        .kv_host = 24_000,
        .ram = .{
            .blobs = 0,
            .prefix = 0,
            .host_kv = 24_000,
            .swap_stages = 16_000,
        },
        .ram_usable = null,
        .logits = 0,
        .scratch = 0,
        .n_slots = 1,
        .window = 100,
    };
    try std.testing.expectEqualStrings(
        "placement: 0..24 host, 24..32 resident (gpu 8/32 layers)",
        formatLayerSplit(mixed, &buf),
    );

    var whole = mixed;
    whole.host_layers = 0;
    try std.testing.expectEqualStrings(
        "placement: 0..32 resident (gpu 32/32 layers)",
        formatLayerSplit(whole, &buf),
    );

    var host = mixed;
    host.host_layers = 32;
    try std.testing.expectEqualStrings(
        "placement: 0..32 host (gpu 0/32 layers)",
        formatLayerSplit(host, &buf),
    );

    // The widest span the layer limit allows still fits the stated ceiling.
    var widest = mixed;
    widest.n_layers = limits.max_layers;
    widest.host_layers = limits.max_layers / 2;
    try std.testing.expectEqualStrings(
        "placement: 0..32 host, 32..64 resident (gpu 32/64 layers)",
        formatLayerSplit(widest, &buf),
    );
}

test "planLayerSplit: the RAM vector, and both staging buffers at high water" {
    // At `k = 8` the host holds `1 * 100 * 8 * 10 = 8000` bytes, one stage holds
    // `100 * 24 * 10 = 24000`, and the pair holds 48000.
    const tight = try planLayerSplit(
        anchorCaps(56_000),
        &anchor_weights,
        0,
        anchor_kv,
        unchecked_serve,
        null,
    );
    try std.testing.expectEqual(@as(u32, 8), tight.host_layers);
    try std.testing.expectEqual(@as(u64, 8_000), tight.ram.host_kv);
    try std.testing.expectEqual(@as(u64, 48_000), tight.ram.swap_stages);
    try std.testing.expectEqual(tight.kv_host, tight.ram.host_kv);

    // The Model 2 split anchor passes through byte for byte.
    const carried = try planLayerSplit(anchorCaps(56_000), &anchor_weights, 0, anchor_kv, .{
        .ram_split = .{ .blobs = 384 << 20, .prefix = 128 << 20 },
        .residency = true,
    }, null);
    try std.testing.expectEqual(@as(u64, 384 << 20), carried.ram.blobs);
    try std.testing.expectEqual(@as(u64, 128 << 20), carried.ram.prefix);

    // The pure CPU placement holds no device suffix, so it stages nothing.
    const cpu = try planLayerSplit(
        anchorCaps(56_000),
        &anchor_weights,
        0,
        anchor_kv,
        unchecked_serve,
        0,
    );
    try std.testing.expectEqual(@as(u32, 32), cpu.host_layers);
    try std.testing.expectEqual(@as(u64, 0), cpu.ram.swap_stages);

    // 70000 usable bytes fund `k = 0`, where one stage holds the whole model's
    // 32000 KV bytes.
    const roomy = try planLayerSplit(
        anchorCaps(80_000),
        &anchor_weights,
        0,
        anchor_kv,
        unchecked_serve,
        null,
    );
    try std.testing.expectEqual(@as(u32, 0), roomy.host_layers);
    try std.testing.expectEqual(@as(u64, 2 * 32_000), roomy.ram.swap_stages);
    try std.testing.expectEqual(2 * roomy.kv_vram, roomy.ram.swap_stages * roomy.n_slots);

    // The swap-unit bridge figures: 4 layers, 3 rows, and 2 f32 per row give a
    // 24-byte region, so one layer's per-slot KV is 48 bytes, two device layers
    // hold 96, and the staging pair holds 192.
    const bridge_weights = [_]u64{100} ** 4;
    const bridge_kv: KvStructure = .{
        .n_slots = 1,
        .window = 3,
        .costs = KvCostVector.uniform(4, 16),
    };
    const bridge = try planLayerSplit(
        anchorCaps(1 << 20),
        &bridge_weights,
        0,
        bridge_kv,
        unchecked_serve,
        2,
    );
    try std.testing.expectEqual(@as(u64, 96), bridge.kv_vram);
    try std.testing.expectEqual(@as(u64, 96), bridge.kv_host);
    try std.testing.expectEqual(@as(u64, 192), bridge.ram.swap_stages);

    // The f16 companion row of the same bridge: 2 elements per row at 2 bytes
    // halve the per-layer figure to 8, and every framing figure halves with it.
    const bridge_kv_f16: KvStructure = .{
        .n_slots = 1,
        .window = 3,
        .costs = KvCostVector.uniform(4, 8),
    };
    const bridge_half = try planLayerSplit(
        anchorCaps(1 << 20),
        &bridge_weights,
        0,
        bridge_kv_f16,
        unchecked_serve,
        2,
    );
    try std.testing.expectEqual(@as(u64, 48), bridge_half.kv_vram);
    try std.testing.expectEqual(@as(u64, 48), bridge_half.kv_host);
    try std.testing.expectEqual(@as(u64, 96), bridge_half.ram.swap_stages);
}

test "planLayerSplit: the RAM vector conserves the split and prices the stages at every split" {
    const layer_counts = [_]u32{ 1, 8, 32, limits.max_layers };
    const split_ram: KvPlan.Ram = .{ .blobs = 7 << 20, .prefix = 3 << 20 };
    // 33 budgets from zero to 160000 bytes. The grid is fixed, so a failure
    // replays.
    const step: u64 = 5_000;
    for (layer_counts) |n| {
        var table: [limits.max_layers]u64 = undefined;
        // A ramp, so a wrong suffix sum cannot pass by symmetry.
        for (table[0..n], 0..) |*w, i| {
            w.* = 1_000 + 100 * @as(u64, i);
        }

        const weights = table[0..n];
        const kv: KvStructure = .{
            .n_slots = 2,
            .window = 100,
            .costs = KvCostVector.uniform(n, 10),
        };
        const total_kv: u64 = @as(u64, kv.n_slots) * perSlotBytes(kv.costs.total(), kv.window);

        for (0..33) |i| {
            const caps = anchorCaps(@as(u64, @intCast(i)) * step);
            var g: u32 = 0;
            while (g <= n) : (g += 1) {
                // A budget that refuses this pin prices nothing to check.
                const plan = planLayerSplit(
                    caps,
                    weights,
                    0,
                    kv,
                    .{ .ram_split = split_ram, .residency = true },
                    g,
                ) catch continue;
                try std.testing.expectEqual(split_ram.blobs, plan.ram.blobs);
                try std.testing.expectEqual(split_ram.prefix, plan.ram.prefix);
                try std.testing.expectEqual(plan.kv_host, plan.ram.host_kv);
                try std.testing.expectEqual(2 * plan.kv_vram, plan.ram.swap_stages * kv.n_slots);
                try std.testing.expect(plan.ram.swap_stages * kv.n_slots <= 2 * total_kv);
                if (plan.host_layers == 0) {
                    try std.testing.expectEqual(2 * total_kv, plan.ram.swap_stages * kv.n_slots);
                }

                if (plan.host_layers == n) {
                    try std.testing.expectEqual(@as(u64, 0), plan.ram.swap_stages);
                }
            }
        }
    }
}

test "KvCostVector: the column sums are the whole-model cost model" {
    // Llama-3.1-8B: 32 layers at 8192 bytes a token each, and no fixed state.
    const llama = KvCostVector.uniform(32, 8_192);
    try std.testing.expectEqual(llama_8b_cost, llama.total());
    try std.testing.expectEqual(@as(usize, 32), llama.layers().len);
    // The tail past the live prefix stays zero, so a stale entry cannot enter a
    // sum through a later `n_layers`.
    for (llama.entries[llama.n_layers..]) |entry| {
        try std.testing.expectEqual(@as(u32, 0), entry.kv_bytes_per_token);
        try std.testing.expectEqual(@as(u32, 0), entry.fixed_bytes_per_slot);
    }

    // The two-kind stack sums each column over its own layers: 6 attention
    // layers price the tokens, and 18 recurrent layers price the slabs.
    const qwen = qwen2bCosts();
    const whole = qwen.total();
    try std.testing.expectEqual(@as(u64, 6 * 4_096), whole.kv_bytes_per_token);
    try std.testing.expectEqual(@as(u64, 18 * 1_122_304), whole.fixed_state_bytes_per_slot);
    try std.testing.expectEqual(@as(usize, 24), qwen.layers().len);

    // A uniform vector of zero price is a legal statement, and it sums to zero.
    const empty = KvCostVector.uniform(limits.max_layers, 0);
    try std.testing.expectEqual(@as(u64, 0), empty.total().kv_bytes_per_token);
    try std.testing.expectEqual(@as(u64, 0), empty.total().fixed_state_bytes_per_slot);
}

test "planLayerSplit: the uniform vector reproduces the whole-model product at every split" {
    // The pre-vector planner charged `n_slots * window * layers * kv_per_layer`
    // at every split. This grid replays that closed form field for field, so a
    // span sum that drifted from it fails here.
    const layer_counts = [_]u32{ 1, 8, 32, limits.max_layers };
    const windows = [_]u32{ 100, 512, 8_192 };
    const slot_counts = [_]u32{ 1, 2, 128 };
    const per_layer_costs = [_]u64{ 8, 10, 8_192 };
    // The widest corner spends 2^39 KV bytes, so this budget funds every k on
    // the grid and no plan on it can refuse.
    const caps = llama8bCaps(1 << 43);
    for (layer_counts) |n| {
        var table: [limits.max_layers]u64 = undefined;
        for (table[0..n]) |*w| {
            w.* = 1_000;
        }

        const weights = table[0..n];
        for (windows) |window| {
            for (slot_counts) |n_slots| {
                for (per_layer_costs) |per_layer| {
                    const kv: KvStructure = .{
                        .n_slots = n_slots,
                        .window = window,
                        .costs = KvCostVector.uniform(n, per_layer),
                    };
                    var k: u32 = 0;
                    while (k <= n) : (k += 1) {
                        const plan = try planLayerSplit(
                            caps,
                            weights,
                            0,
                            kv,
                            unchecked_serve,
                            n - k,
                        );
                        try std.testing.expectEqual(k, plan.host_layers);
                        try std.testing.expectEqual(
                            @as(u64, n_slots) * window * (n - k) * per_layer,
                            plan.kv_vram,
                        );
                        try std.testing.expectEqual(
                            @as(u64, n_slots) * window * k * per_layer,
                            plan.kv_host,
                        );
                        // One stage holds the device span of one slot, or one
                        // chunk when the span is larger.
                        try std.testing.expectEqual(
                            @as(u64, 2) * @min(
                                @as(u64, window) * (n - k) * per_layer,
                                limits.swap_stage_chunk_bytes,
                            ),
                            plan.ram.swap_stages,
                        );
                        try std.testing.expectEqual(plan.kv_host, plan.ram.host_kv);
                    }
                }
            }
        }
    }
}

test "planLayerSplit: a two-kind stack prices each tier by its own layers" {
    const weights = [_]u64{1_000} ** 24;
    const kv: KvStructure = .{ .n_slots = 2, .window = 512, .costs = qwen2bCosts() };
    // A mid split: layers `[12..24)` hold 3 attention layers and 9 recurrent
    // ones, so the device pays 3 window prices and 9 slabs.
    const plan = try planLayerSplit(anchorCaps(1 << 30), &weights, 0, kv, unchecked_serve, 12);
    try std.testing.expectEqual(@as(u32, 12), plan.host_layers);
    try std.testing.expectEqual(
        @as(u64, kv.n_slots) * (@as(u64, kv.window) * 3 * 4_096 + 9 * 1_122_304),
        plan.kv_vram,
    );
    try std.testing.expectEqual(
        @as(u64, kv.n_slots) * (@as(u64, kv.window) * 3 * 4_096 + 9 * 1_122_304),
        plan.kv_host,
    );
    // A uniform price divides the whole-model token cost by 24 and charges
    // every layer 1024 bytes, which prices this suffix at half its real KV and
    // none of its recurrent state.
    try std.testing.expect(plan.kv_vram > @as(u64, kv.n_slots) * kv.window * 12 * 1_024);
    // The ledger still partitions, and the staging pair still prices one slot's
    // device suffix twice.
    try std.testing.expectEqual(
        @as(u64, kv.n_slots) * perSlotBytes(kv.costs.total(), kv.window),
        plan.kv_vram + plan.kv_host,
    );
    try std.testing.expectEqual(2 * plan.kv_vram, plan.ram.swap_stages * kv.n_slots);
}

test "planLayerSplit: a prefix with no attention layer still charges its recurrent state" {
    const weights = [_]u64{1_000} ** 24;
    const kv: KvStructure = .{ .n_slots = 2, .window = 512, .costs = qwen2bCosts() };
    const whole = @as(u64, kv.n_slots) * perSlotBytes(kv.costs.total(), kv.window);
    // Layers 0, 1, and 2 are all recurrent, so a prefix that stops inside them
    // holds slabs and not one token of window.
    for ([_]u32{ 1, 2, 3 }) |k| {
        const plan = try planLayerSplit(
            anchorCaps(1 << 30),
            &weights,
            0,
            kv,
            unchecked_serve,
            24 - k,
        );
        try std.testing.expect(plan.kv_host > 0);
        try std.testing.expectEqual(@as(u64, kv.n_slots) * k * 1_122_304, plan.kv_host);
        try std.testing.expectEqual(whole, plan.kv_vram + plan.kv_host);
        try std.testing.expectEqual(2 * plan.kv_vram, plan.ram.swap_stages * kv.n_slots);
    }

    // The mirror: an all-recurrent SUFFIX stages fixed bytes and no window
    // price at all.
    var tail = KvCostVector.uniform(4, 0);
    tail.set(0, 4_096, 0);
    for ([_]u32{ 1, 2, 3 }) |i| {
        tail.set(i, 0, 1_122_304);
    }

    const tail_kv: KvStructure = .{ .n_slots = 2, .window = 512, .costs = tail };
    const tail_weights = [_]u64{1_000} ** 4;
    const tail_plan = try planLayerSplit(
        anchorCaps(1 << 30),
        &tail_weights,
        0,
        tail_kv,
        unchecked_serve,
        3,
    );
    try std.testing.expectEqual(@as(u32, 1), tail_plan.host_layers);
    try std.testing.expectEqual(@as(u64, tail_kv.n_slots) * 3 * 1_122_304, tail_plan.kv_vram);
    try std.testing.expectEqual(@as(u64, 2) * 3 * 1_122_304, tail_plan.ram.swap_stages);
    try std.testing.expectEqual(
        @as(u64, tail_kv.n_slots) * tail_kv.window * 4_096,
        tail_plan.kv_host,
    );
}

test "planLayerSplit: the window never multiplies the fixed state" {
    const weights = [_]u64{1_000} ** 24;
    const costs = qwen2bCosts();
    const narrow: KvStructure = .{ .n_slots = 2, .window = 512, .costs = costs };
    const wide: KvStructure = .{ .n_slots = 2, .window = 1_024, .costs = costs };
    const narrow_plan = try planLayerSplit(
        anchorCaps(1 << 30),
        &weights,
        0,
        narrow,
        unchecked_serve,
        12,
    );
    const wide_plan = try planLayerSplit(
        anchorCaps(1 << 30),
        &weights,
        0,
        wide,
        unchecked_serve,
        12,
    );

    // The suffix `[12..24)` holds 3 attention layers and 9 recurrent ones. The
    // token column doubles with the window, and the slab column does not move.
    const token_narrow = @as(u64, narrow.n_slots) * narrow.window * 3 * 4_096;
    const token_wide = @as(u64, wide.n_slots) * wide.window * 3 * 4_096;
    const fixed: u64 = @as(u64, narrow.n_slots) * 9 * 1_122_304;
    try std.testing.expectEqual(2 * token_narrow, token_wide);
    try std.testing.expectEqual(fixed, narrow_plan.kv_vram - token_narrow);
    try std.testing.expectEqual(fixed, wide_plan.kv_vram - token_wide);
}

test "planLayerSplit: a pointwise-cheaper vector never yields a larger split" {
    // Every entry halves in both columns, so the cheaper vector is pointwise no
    // dearer and the auto scan can only reach the same split or a lower one.
    const n: u32 = 32;
    var table: [limits.max_layers]u64 = undefined;
    for (table[0..n], 0..) |*w, i| {
        w.* = 1_000 + 100 * @as(u64, i);
    }

    const weights = table[0..n];
    const dear: KvStructure = .{ .n_slots = 2, .window = 100, .costs = rampCosts(n) };
    const cheap: KvStructure = .{ .n_slots = 2, .window = 100, .costs = halfCosts(rampCosts(n)) };
    // 33 budgets from zero to 160000 bytes. The grid is fixed, so a failure
    // replays.
    const step: u64 = 5_000;
    for (0..33) |i| {
        const caps = anchorCaps(@as(u64, @intCast(i)) * step);
        const dear_plan = try planLayerSplit(caps, weights, 0, dear, unchecked_serve, null);
        const cheap_plan = try planLayerSplit(caps, weights, 0, cheap, unchecked_serve, null);
        try std.testing.expect(cheap_plan.host_layers <= dear_plan.host_layers);
    }
}

test "formatLedger: the widest ledger" {
    var buf: [layer_split_ledger_text_max]u8 = undefined;
    const mib = 1024 * 1024;
    const round: LayerSplit = .{
        .host_layers = 24,
        .n_layers = 32,
        .weight_resident = 3 * mib,
        .kv_vram = 5 * mib,
        .kv_host = 7 * mib,
        .ram = .{
            .blobs = 11 * mib,
            .prefix = 13 * mib,
            .host_kv = 7 * mib,
            .swap_stages = 17 * mib,
        },
        .ram_usable = null,
        .logits = 0,
        .scratch = 0,
        .n_slots = 2,
        .window = 512,
    };
    try std.testing.expectEqualStrings(
        "ledger: 2 slots x 512 window | gpu 3 MiB weights + 5 MiB kv" ++
            " | host 7 MiB kv + 11 MiB blobs + 13 MiB prefix + 17 MiB stages | ram 48 MiB, " ++
            "not checked",
        formatLedger(round, &buf),
    );
    var checked = round;
    checked.ram_usable = 1792 * mib;
    try std.testing.expectEqualStrings(
        "ledger: 2 slots x 512 window | gpu 3 MiB weights + 5 MiB kv" ++
            " | host 7 MiB kv + 11 MiB blobs + 13 MiB prefix + 17 MiB stages | ram 48 of " ++
            "1792 MiB usable",
        formatLedger(checked, &buf),
    );

    // The widest figure every field can carry still fits the stated ceiling,
    // in both RAM forms.
    const wide: u64 = std.math.maxInt(u64);
    const widest: LayerSplit = .{
        .host_layers = 0,
        .n_layers = limits.max_layers,
        .weight_resident = wide,
        .kv_vram = wide,
        .kv_host = wide,
        .ram = .{ .blobs = wide, .prefix = wide, .host_kv = wide, .swap_stages = wide },
        .ram_usable = wide,
        .logits = 0,
        .scratch = 0,
        .n_slots = std.math.maxInt(u32),
        .window = std.math.maxInt(u32),
    };
    _ = formatLedger(widest, &buf);
    var widest_unchecked = widest;
    widest_unchecked.ram_usable = null;
    _ = formatLedger(widest_unchecked, &buf);
}

test "formatVramRefusal: the three families" {
    var buf: [vram_refusal_text_max]u8 = undefined;
    const mib = 1024 * 1024;
    // The whole-model llama runner. 8 GiB free leaves 7168 MiB usable.
    const whole: VramRefusal = .{
        .planned = 9000 * mib,
        .available = 8192 * mib,
        .weights = 8000 * mib,
        .kv = 1000 * mib,
        .n_slots = 4,
        .window = 4096,
        .fit = 2048,
    };
    try std.testing.expectEqualStrings(
        "vram budget: the KV plan needs 9000 MiB of VRAM, " ++
            "but only 7168 MiB of the 8192 MiB free are usable. " ++
            "It holds 8000 MiB of weights and 1000 MiB of KV for 4 slots x 4096 positions. " ++
            "Use --max-seq 2048 or less, or fewer sessions",
        formatVramRefusal(&buf, whole),
    );
    var none = whole;
    none.fit = null;
    try std.testing.expectEqualStrings(
        "vram budget: the KV plan needs 9000 MiB of VRAM, " ++
            "but only 7168 MiB of the 8192 MiB free are usable. " ++
            "It holds 8000 MiB of weights and 1000 MiB of KV for 4 slots x 4096 positions. " ++
            "No window fits 4 slots. Use fewer sessions, or a smaller or Q4 model",
        formatVramRefusal(&buf, none),
    );

    // The llama hybrid runner, which holds the suffix of a split model.
    var suffix = whole;
    suffix.suffix = true;
    try std.testing.expectEqualStrings(
        "vram budget: the KV plan needs 9000 MiB of VRAM, " ++
            "but only 7168 MiB of the 8192 MiB free are usable. " ++
            "It holds 8000 MiB of suffix weights " ++
            "and 1000 MiB of KV for 4 slots x 4096 positions. " ++
            "Use --max-seq 2048 or less, or lower --gpu-layers",
        formatVramRefusal(&buf, suffix),
    );
    suffix.fit = null;
    try std.testing.expectEqualStrings(
        "vram budget: the KV plan needs 9000 MiB of VRAM, " ++
            "but only 7168 MiB of the 8192 MiB free are usable. " ++
            "It holds 8000 MiB of suffix weights " ++
            "and 1000 MiB of KV for 4 slots x 4096 positions. No window fits the suffix. " ++
            "Lower --gpu-layers, or use a smaller or Q4 model",
        formatVramRefusal(&buf, suffix),
    );

    // The qwen runner, which adds the recurrent state of its GDN layers.
    var recurrent = whole;
    recurrent.recurrent_state = 20 * mib;
    try std.testing.expectEqualStrings(
        "vram budget: the KV plan needs 9000 MiB of VRAM, " ++
            "but only 7168 MiB of the 8192 MiB free are usable. " ++
            "It holds 8000 MiB of weights, 1000 MiB of KV, " ++
            "and 20 MiB of recurrent state for 4 slots x 4096 positions. " ++
            "Use --max-seq 2048 or less, or fewer sessions",
        formatVramRefusal(&buf, recurrent),
    );
    recurrent.fit = null;
    try std.testing.expectEqualStrings(
        "vram budget: the KV plan needs 9000 MiB of VRAM, " ++
            "but only 7168 MiB of the 8192 MiB free are usable. " ++
            "It holds 8000 MiB of weights, 1000 MiB of KV, " ++
            "and 20 MiB of recurrent state for 4 slots x 4096 positions. " ++
            "No window fits 4 slots. Use fewer sessions, or a smaller or Q4 model",
        formatVramRefusal(&buf, recurrent),
    );

    // The widest figure every field can carry, with every optional part set,
    // still fits the stated ceiling.
    const wide: u64 = std.math.maxInt(u64);
    const widest: VramRefusal = .{
        .planned = wide,
        .available = wide,
        .weights = wide,
        .kv = wide,
        .recurrent_state = wide,
        .n_slots = std.math.maxInt(u32),
        .window = std.math.maxInt(u32),
        .fit = std.math.maxInt(u32) - 1,
        .suffix = true,
    };
    try std.testing.expect(formatVramRefusal(&buf, widest).len <= vram_refusal_text_max);
    var widest_none = widest;
    widest_none.fit = null;
    widest_none.suffix = false;
    try std.testing.expect(formatVramRefusal(&buf, widest_none).len <= vram_refusal_text_max);
}

/// One random stack for the RAM sweeps: a uniform stack, or a Qwen-shaped one
/// whose every fourth layer charges per token and whose other layers hold a
/// fixed slab.
const SweepStack = struct {
    weights: [limits.max_layers]u64,
    head: u64,
    kv: KvStructure,
};

fn sweepStack(random: std.Random, qwen_shaped: bool) SweepStack {
    const n = random.intRangeAtMost(u32, 1, limits.max_layers);
    var stack: SweepStack = .{
        .weights = @splat(0),
        .head = random.intRangeAtMost(u64, 0, 1 << 26),
        .kv = .{
            .n_slots = random.intRangeAtMost(u32, 1, 8),
            .window = random.intRangeAtMost(u32, limits.min_kv_window, 1 << 16),
            .costs = KvCostVector.uniform(n, 0),
        },
    };
    const per_token = random.intRangeAtMost(u64, 1, 8_192);
    const slab = random.intRangeAtMost(u64, 1, 1 << 21);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        stack.weights[i] = random.intRangeAtMost(u64, 1, 1 << 24);
        if (!qwen_shaped) {
            stack.kv.costs.set(i, per_token, 0);
        } else if ((i + 1) % 4 == 0) {
            stack.kv.costs.set(i, per_token, 0);
        } else {
            stack.kv.costs.set(i, 0, slab);
        }
    }

    return stack;
}

/// The RAM check over `cases` seeded random splits. The VRAM planner picks
/// the split, and `quoteRam` funds or refuses it. The sweep checks the four
/// statements the check owes against the VRAM plan of the same request, and
/// the fit window against two re-quotes.
const RamSweep = struct {
    qwen_shaped: bool,
    accepted: u32 = 0,
    refused: u32 = 0,
    fits: u32 = 0,
};

fn checkRamSweep(sweep: *RamSweep, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const stack = sweepStack(random, sweep.qwen_shaped);
    const n = stack.kv.costs.n_layers;
    const weights = stack.weights[0..n];
    const whole_device = deviceSpend(anchorCaps(0), weights, stack.head, stack.kv, 0);
    var caps = anchorCaps(random.intRangeAtMost(u64, 0, 2 * whole_device));
    caps.model_ctx = limits.max_model_ctx;
    const residency = random.boolean();
    const gpu_layers: ?u32 = if (random.boolean()) null else random.intRangeAtMost(u32, 0, n);
    // Every slot holds its session, so no blob is reachable and the Y
    // split funds the prefix alone.
    const ram_split: KvPlan.Ram = if (random.boolean())
        no_ram_split
    else
        .{ .blobs = 0, .prefix = random.intRangeAtMost(u64, 0, 1 << 30) };

    const n_logical = stack.kv.n_slots;
    const open: HostRequest = .{ .ram_split = ram_split, .residency = residency };

    // A VRAM refusal comes first and reads no RAM figure.
    const vram_plan = planLayerSplit(
        caps,
        weights,
        stack.head,
        stack.kv,
        open,
        gpu_layers,
    ) catch |err| {
        try std.testing.expectEqual(error.WeightsVramBudgetTooSmall, err);
        return;
    };
    // No residency gives zero stages, and residency prices two stages of
    // one chunk or one device suffix of one slot, whichever is smaller.
    if (residency) {
        try std.testing.expectEqual(
            2 * stageBytes(slotSpanBytes(stack.kv, vram_plan.host_layers, n)),
            vram_plan.ram.swap_stages,
        );
    } else {
        try std.testing.expectEqual(@as(u64, 0), vram_plan.ram.swap_stages);
    }

    const need = vram_plan.ram.spend();
    const ram_free = random.intRangeAtMost(u64, 0, 2 * need + 16);
    const host = open;
    const ram_usable = limits.ramUsable(ram_free);
    if (quoteRam(stack.kv, vram_plan.host_layers, host, n_logical, true, ram_free)) |q| {
        sweep.accepted += 1;
        // An accepted quote spends at most the usable RAM, and it kept the
        // vector of the split the VRAM planner chose.
        try std.testing.expect(q.ram.spend() <= ram_usable);
        try std.testing.expectEqual(vram_plan.ram, q.ram);
        const plan = vram_plan.priced(q);
        try std.testing.expectEqual(vram_plan.host_layers, plan.host_layers);
        try std.testing.expectEqual(@as(?u64, ram_usable), plan.ram_usable);
    } else |err| {
        sweep.refused += 1;
        try std.testing.expectEqual(error.HostRamBudgetTooSmall, err);
        // A refusal happens exactly when the vector at the VRAM split
        // passes the usable RAM.
        try std.testing.expect(ram_usable < need);

        const refusal = ramRefusal(stack.kv, vram_plan.host_layers, host, ram_free);
        try std.testing.expectEqual(need, refusal.spend);
        try std.testing.expectEqual(ram_usable, refusal.usable);
        try std.testing.expectEqual(n - vram_plan.host_layers, refusal.gpu_layers);

        // Without residency, an auto refusal is final: every split the
        // VRAM funds spends at least as much RAM.
        if (gpu_layers == null and !residency) {
            var k: u32 = 0;
            while (k <= n) : (k += 1) {
                if (!layerSplitFits(caps, weights, stack.head, stack.kv, k)) continue;
                try std.testing.expectError(
                    error.HostRamBudgetTooSmall,
                    quoteRam(stack.kv, k, host, n_logical, true, ram_free),
                );
            }
        }

        // The printed fit boots at the quoted split, and it still boots
        // after the free RAM falls far enough to cut 64 MiB of usable RAM.
        var narrow = stack.kv;
        if (refusal.fit) |fit| {
            sweep.fits += 1;
            try std.testing.expect(fit < stack.kv.window);
            try std.testing.expectEqual(@as(u32, 0), fit % limits.window_granule);
            narrow.window = fit;
            const fitted = try quoteRam(
                narrow,
                vram_plan.host_layers,
                host,
                n_logical,
                true,
                ram_free,
            );
            try std.testing.expect(fitted.ram.spend() <= ram_usable);
            _ = try quoteRam(
                narrow,
                vram_plan.host_layers,
                host,
                n_logical,
                true,
                ram_free -| lowered_free_bytes,
            );
        }

        // The exact fit holds the stage pair of the refused window, which bounds narrower pairs.
        // So one more position, or the floor with no exact fit, refuses on that same pair.
        const exact = ramFitWindow(
            stack.kv,
            vram_plan.host_layers,
            0,
            refusal.swap_stages +| refusal.blobs +| refusal.prefix,
            ram_usable,
        );
        if (refusal.fit) |fit| {
            try std.testing.expect(fit <= exact.?);
        }

        if (exact) |e| {
            narrow.window = e;
            _ = try quoteRam(narrow, vram_plan.host_layers, host, n_logical, true, ram_free);
            narrow.window = e + 1;
        } else {
            narrow.window = caps.min_window;
        }

        if (ramAt(narrow, vram_plan.host_layers, n, host).swap_stages == refusal.swap_stages) {
            try std.testing.expectError(
                error.HostRamBudgetTooSmall,
                quoteRam(narrow, vram_plan.host_layers, host, n_logical, true, ram_free),
            );
        }
    }
}

fn ramSweep(site: u64, qwen_shaped: bool, cases: u32) !void {
    var sweep: RamSweep = .{ .qwen_shaped = qwen_shaped };
    try prop.forAll(
        .{ .site = site, .cases = cases },
        &sweep,
        u64,
        prop.genSeed(*RamSweep),
        checkRamSweep,
    );
    // The grid reaches both verdicts and the fit form, so no branch above
    // passes empty.
    try std.testing.expect(sweep.accepted > 0);
    try std.testing.expect(sweep.refused > 0);
    try std.testing.expect(sweep.fits > 0);
}

test "quoteRam: the RAM check over 2048 seeded uniform stacks" {
    try ramSweep(0x5eed_0001, false, 2048);
}

test "quoteRam: the RAM check over 2048 seeded Qwen-shaped stacks" {
    try ramSweep(0x5eed_0002, true, 2048);
}

test "formatRamRefusal: the widest text and both advice forms" {
    var buf: [ram_refusal_text_max]u8 = undefined;
    // The fit form: the serve anchor at the auto split, on a host with 8 GiB
    // free.
    const refusal = ramRefusal(ramAnchorKv(131_072), 15, .{
        .ram_split = .{ .blobs = 0, .prefix = 1_073_741_824 },
        .residency = true,
    }, 8 << 30);
    try std.testing.expectEqualStrings(
        "ram budget: the placement needs 9216 MiB of RAM, " ++
            "but only 7168 MiB of the 8192 MiB free are usable. " ++
            "It holds 7680 MiB of host KV, 512 MiB of swap stages, 0 MiB of blobs, " ++
            "and 1024 MiB of prefix cache. " ++
            "The plan serves 1 slot x 131072 positions at gpu 17/32 layers. " ++
            "Use --gpu-layers 17 --max-seq 93696 or less.",
        formatRamRefusal(&buf, refusal),
    );

    // The no-fit forms name only the levers that lower the spend.
    const mib = 1024 * 1024;
    const crowded: RamRefusal = .{
        .spend = 9000 * mib,
        .free = 4096 * mib,
        .usable = 3584 * mib,
        .host_kv = 4000 * mib,
        .swap_stages = 4488 * mib,
        .blobs = 0,
        .prefix = 512 * mib,
        .n_slots = 4,
        .window = 4096,
        .gpu_layers = 8,
        .n_layers = 32,
        .residency = true,
        .fit = null,
    };
    try std.testing.expectEqualStrings(
        "ram budget: the placement needs 9000 MiB of RAM, " ++
            "but only 3584 MiB of the 4096 MiB free are usable. " ++
            "It holds 4000 MiB of host KV, 4488 MiB of swap stages, 0 MiB of blobs, " ++
            "and 512 MiB of prefix cache. " ++
            "The plan serves 4 slots x 4096 positions at gpu 8/32 layers. " ++
            "No window fits this split. Use fewer slots, a lower --kv-ram, or a smaller model.",
        formatRamRefusal(&buf, crowded),
    );
    var lone = crowded;
    lone.n_slots = 1;
    lone.prefix = 0;
    lone.spend = 8488 * mib;
    try std.testing.expectEqualStrings(
        "ram budget: the placement needs 8488 MiB of RAM, " ++
            "but only 3584 MiB of the 4096 MiB free are usable. " ++
            "It holds 4000 MiB of host KV, 4488 MiB of swap stages, 0 MiB of blobs, " ++
            "and 0 MiB of prefix cache. " ++
            "The plan serves 1 slot x 4096 positions at gpu 8/32 layers. " ++
            "No window fits this split. Use a smaller model.",
        formatRamRefusal(&buf, lone),
    );

    // The widest figure every field can carry, in both advice forms, still
    // fits the stated ceiling.
    const wide: u64 = std.math.maxInt(u64);
    const widest: RamRefusal = .{
        .spend = wide,
        .free = wide,
        .usable = wide,
        .host_kv = wide,
        .swap_stages = wide,
        .blobs = wide,
        .prefix = wide,
        .n_slots = std.math.maxInt(u32),
        .window = std.math.maxInt(u32),
        .gpu_layers = std.math.maxInt(u32),
        .n_layers = std.math.maxInt(u32),
        .residency = true,
        .fit = std.math.maxInt(u32) - 1,
    };
    try std.testing.expect(formatRamRefusal(&buf, widest).len <= ram_refusal_text_max);
    var widest_none = widest;
    widest_none.fit = null;
    try std.testing.expect(formatRamRefusal(&buf, widest_none).len <= ram_refusal_text_max);
}

test "formatBlobRefusal: the sessions fit form, and the widest figures" {
    var buf: [blob_refusal_text_max]u8 = undefined;
    const mib: u64 = 1 << 20;
    const three: BlobRefusal = .{
        .need = 24_726 * mib,
        .left = 18_000 * mib,
        .blob = 8_242 * mib,
        .n_logical = 4,
        .n_slots = 1,
        .window = 262_144,
        .fit_window = 131_072,
        .fit_sessions = 2,
    };
    try std.testing.expectEqualStrings(
        "ram budget: the blob tier needs 24726 MiB of RAM, but the plan leaves only 18000 MiB." ++
            " One evicted session holds up to 8242 MiB. The plan serves 4 sessions on 1 " ++
            "slot at 262144 positions." ++
            " Use --max-seq 131072 or less. Or use --max-sessions 2 or less.",
        formatBlobRefusal(&buf, three),
    );
    var no_window = three;
    no_window.fit_window = null;
    try std.testing.expectEqualStrings(
        "ram budget: the blob tier needs 24726 MiB of RAM, but the plan leaves only 18000 MiB." ++
            " One evicted session holds up to 8242 MiB. The plan serves 4 sessions on 1 " ++
            "slot at 262144 positions." ++
            " No window fits these sessions. Use --max-sessions 2 or less.",
        formatBlobRefusal(&buf, no_window),
    );

    const wide: u64 = std.math.maxInt(u64);
    const widest: BlobRefusal = .{
        .need = wide,
        .left = wide,
        .blob = wide,
        .n_logical = std.math.maxInt(u32),
        .n_slots = std.math.maxInt(u32) - 2,
        .window = std.math.maxInt(u32),
        .fit_window = std.math.maxInt(u32) - 1,
        .fit_sessions = std.math.maxInt(u32) - 1,
    };
    try std.testing.expect(formatBlobRefusal(&buf, widest).len <= blob_refusal_text_max);
    var widest_none = widest;
    widest_none.fit_window = null;
    widest_none.fit_sessions = null;
    try std.testing.expect(formatBlobRefusal(&buf, widest_none).len <= blob_refusal_text_max);
}
