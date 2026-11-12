//! Role. The device placements of a boot as values: the fused fit, the VRAM budgets, and the
//! hybrid placement. The CLI measures the device once (`ModelNeeds`, `DeviceCapacity`), `resolve`
//! calls `choosePlacement`, and the CLI logs the value and constructs the runner it names. A table
//! test runs every device row with no GPU.
//!
//! Log order. The value keeps the order of the boot lines. A `LayerChoice` value holds the lines
//! before the split and the split itself. The placement holds the lines of the runner the split
//! selects. A refusal follows every line before it, so the CLI logs the `LayerChoice` lines first
//! and then formats the refusal.
//!
//! VRAM budgets. One KV region holds every slot at the window, so the device buffer limits bound
//! the product of the two. With pinned slots the money buys the window, so the clamp runs on the
//! plan and a cut window replans at the same slot count. The money can still lower a named window,
//! which is a cap there (`kv_budget.resolveWindow`).
//!
//! Unpinned budgets. The window is known first: the flag, or the serve default the limits leave for
//! one slot. The money buys the slots, so the limits cap the slot count. The limits never shrink a
//! named window, and a shape they refuse at any slot count reaches the runner, which refuses it
//! with the numbers.
//!
//! Window fits. A named window passes through every fit, so the planner or the runner refuses it
//! with the numbers. A default window shrinks, first to the device buffer limits and then to the
//! VRAM figure. When no granule fits, the default stands and the runner refuses it.
//!
//! Slots. A fused serve plan holds at least the `fused_min_slots` of the family row: the llama
//! runner keeps a spare stream. The hybrid runner indexes its device suffix by `limits.max_batch`,
//! so a hybrid slot count past that ceiling refuses when a flag pinned it and clamps when the
//! serve options derived it.
//!
//! Family. A placement reads the values of the family row of the card, never the layer shape.
//!
//! Per-kind rules.
//!
//!   rule                          run               serve                     bench
//!   slots of the fused placement  1                 the structural slots      1
//!                                                   (at least fused_min_slots)
//!   default window fit            clamp, fit        clamp, fit                clamp, fit
//!   VRAM budget                   none              --kv-vram, --vram-slots   none
//!   residency priced in RAM       no                yes                       no
//!   whole-model split             falls through     the placement it selects  refuses
//!                                 to the placement

const std = @import("std");

const limits = @import("../core/limits.zig");
const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_index = @import("../core/kv_index.zig");
const loader = @import("../io/gguf/loader.zig");
const kv_cost = @import("../runtime/kv_cost.zig");
const family = @import("family.zig");
const resolve = @import("resolve.zig");

const Kind = resolve.Kind;
const Input = resolve.Input;
const Window = resolve.Window;
const Refusal = resolve.Refusal;
const Result = resolve.Result;

// -- types ------------------------------------------------------------------

/// The values of a family row that a placement reads.
const RowValues = struct {
    /// The fewest slots of a fused serve plan.
    fused_min_slots: u32,
    /// The fused path of `run` honors `--prompt-cache`.
    prompt_cache: bool,
    /// `run` logs a placement line where each placement starts.
    run_logs_every_placement: bool,
};

/// The reservations the fused runners make for one model. The CLI measures each field with the
/// function the runner calls. A build without the fused ops states zero past the weight census.
pub const ModelNeeds = struct {
    n_layers: u32,
    /// `tensor_lookup.layerWeightBytes`. The first `n_layers` entries hold.
    layer_weights: [limits.max_layers]u64,
    /// `tensor_lookup.headWeightBytes`.
    head_weights: u64,
    /// The whole-model upload: `weightsVramBytes` of the fused runner.
    weights: u64 = 0,
    /// `scratchVramBytes` of the fused runner, per KV dtype.
    scratch: std.EnumArray(dtype.KvDtype, u64) = .initFill(0),
    /// One logits row: `logitsVramBytes(vocab_size, 1)` of the fused runner.
    logits_row: u64 = 0,

    pub fn layers(self: *const ModelNeeds) []const u64 {
        std.debug.assert(self.n_layers >= 1);
        std.debug.assert(self.n_layers <= limits.max_layers);

        return self.layer_weights[0..self.n_layers];
    }
};

/// What the device context states.
pub const DeviceCapacity = struct {
    /// The live VRAM budget, or the heap size on a driver without the budget extension.
    vram: u64,
    /// The driver, which picks the usable share of `vram`.
    driver: limits.Driver,
    buffer_limits: limits.BufferLimits,
};

/// A default window that the device buffer limits cut, with the figures of its log line.
pub const Clamp = struct {
    n_slots: u32,
    base: u32,
    /// The KV region the base window needs, in the unit of `limit`.
    need: u64,
    limit: kv_index.Limit,
    window: u32,
};

/// A default window that the VRAM figure cuts.
pub const Cut = struct {
    n_slots: u32,
    base: u32,
    window: u32,
};

/// A budget whose slot count the slot ceiling cut: the bytes it strands.
pub const Stranded = struct {
    max_slots: u32,
    stranded: u64,
    x_bytes: u64,
};

/// A budget whose slot count the device buffer limits cut.
pub const DeviceSlots = struct {
    limit: kv_index.Limit,
    slots: u32,
    window: u32,
};

pub const PlacementKind = enum { fused, hybrid, cpu };

/// The whole-model fused placement.
pub const Fused = struct {
    caps: kv_budget.Caps,
    /// The cut of a default window to the buffer limits.
    clamp: ?Clamp = null,
    /// The window after the clamp. Run and bench log it when no cut follows.
    window: Window,
    cut: ?Cut = null,
    stranded: ?Stranded = null,
    device_slots: ?DeviceSlots = null,
    /// True when no `--kv-ram` named a Y, so the prefix took the serve default.
    prefix_default: bool = false,
    plan: kv_budget.KvPlan,
    quote: Result(kv_budget.RamQuote),
    /// The all-device split that `run` states for a llama model.
    report: ?kv_budget.LayerSplit = null,
};

/// The hybrid placement: a split with layers on both tiers.
pub const Hybrid = struct {
    /// The window the split prices.
    window: Window,
    quote: Result(kv_budget.RamQuote),
    /// The split with its RAM quote, when the quote holds.
    plan: ?kv_budget.LayerSplit,
};

pub const Placement = union(PlacementKind) {
    fused: Fused,
    hybrid: Hybrid,
    /// An auto split that funds no GPU layer, or a serve split that keeps no
    /// layer on the device.
    cpu: Result(resolve.HostPlacement),
};

/// The lines a `--gpu-layers` boot logs before its placement, and the split.
pub const LayerChoice = struct {
    /// Null is the auto pin.
    pin: ?u32,
    /// The structural slot count that the hybrid slot ceiling cut.
    slots_from: ?u32 = null,
    clamp: ?Clamp,
    window: Window,
    prefix_default: bool = false,
    /// The layer split, or null when the planner refused the pin.
    layer_split: ?kv_budget.LayerSplit = null,
    /// The placement the split selects. Null when the split refused.
    selects: ?PlacementKind = null,
};

pub const DevicePlacement = struct {
    /// Null for a boot that names no split.
    layer_choice: ?LayerChoice = null,
    placement: Result(Placement),
};

const Budgeted = struct {
    plan: kv_budget.KvPlan,
    clamp: ?Clamp = null,
    stranded: ?Stranded = null,
    device_slots: ?DeviceSlots = null,
};

// -- api --------------------------------------------------------------------

/// The device placement of one boot. `window` is the window before any device fit.
pub fn choosePlacement(
    kind: Kind,
    input: Input,
    card: loader.ModelCard,
    kv_dtype: dtype.KvDtype,
    window: Window,
    resources: resolve.Resources,
) DevicePlacement {
    std.debug.assert(resources.model != null);
    std.debug.assert(resources.device != null);
    std.debug.assert(resources.model.?.n_layers == card.n_layers);

    return switch (kind) {
        .serve => servePlacement(input, card, kv_dtype, window, resources),
        .run, .bench => sessionPlacement(kind, input, card, kv_dtype, window, resources),
    };
}

/// The planner inputs of one fused runner: the figures `kvCaps` of the runner
/// returns for the same arguments.
pub fn caps(
    card: loader.ModelCard,
    m: ModelNeeds,
    vram: u64,
    driver: limits.Driver,
    ceiling: u32,
    logit_rows: u32,
    kv_dtype: dtype.KvDtype,
) kv_budget.Caps {
    std.debug.assert(ceiling >= 1);
    std.debug.assert(ceiling <= card.context_length);

    const windows = kv_budget.defaultWindows(
        kv_budget.costModel(card.kvShape(), kv_dtype),
        ceiling,
    );
    return .{
        .vram_free = vram,
        .driver = driver,
        .weights = m.weights,
        .logits = @as(u64, logit_rows) * m.logits_row,
        .scratch = m.scratch.get(kv_dtype),
        .model_ctx = ceiling,
        .serve_default = windows.serve,
        .default_ceil = windows.run,
    };
}

/// The placement line of `plan`: the split the auto planner takes at `c`. No runner reads it.
pub fn report(
    c: kv_budget.Caps,
    card: loader.ModelCard,
    m: ModelNeeds,
    kv_dtype: dtype.KvDtype,
    plan: kv_budget.KvPlan,
) kv_budget.LayerSplit {
    const kv: kv_budget.KvStructure = .{
        .n_slots = plan.n_slots,
        .window = plan.window,
        .costs = kv_cost.costVector(card, kv_dtype),
    };
    // The all-host split reserves no device bytes, so the auto planner cannot refuse.
    return kv_budget.planLayerSplit(
        c,
        m.layers(),
        m.head_weights,
        kv,
        .{ .ram_split = plan.ram, .residency = false },
        null,
    ) catch unreachable;
}

/// The window a fused or hybrid placement plans at, before any VRAM fit. A default window whose
/// region breaks a device limit shrinks to `kv_index.maxGranuleWindow`.
pub fn clampDefault(
    n_slots: u32,
    kv_row_elems: u32,
    kv_dtype: dtype.KvDtype,
    lim: limits.BufferLimits,
    base: u32,
    flag: ?u32,
) struct { window: u32, clamp: ?Clamp } {
    std.debug.assert(base >= 1);
    std.debug.assert(base <= limits.max_model_ctx);

    if (flag) |w| {
        std.debug.assert(base == w);

        return .{ .window = base, .clamp = null };
    }

    const deepest = kv_index.maxWindowForLimits(n_slots, kv_row_elems, kv_dtype, lim);
    if (base <= deepest) return .{ .window = base, .clamp = null };

    // This is `kv_index.maxGranuleWindow` without a second solve.
    const window: u32 = @intCast(limits.floorToGranule(deepest));
    if (window == 0) return .{ .window = base, .clamp = null };

    std.debug.assert(window < base);

    // Name the limit that binds the window. One layer stands for all, because every region has
    // one size.
    const at_base = kv_index.Layout.init(n_slots, base, kv_row_elems, 1, kv_dtype);
    const verdict = kv_index.fits(
        kv_index.Layout.init(n_slots, deepest + 1, kv_row_elems, 1, kv_dtype),
        lim,
    );
    std.debug.assert(verdict != .fits);
    std.debug.assert(verdict != .misaligned);
    const limit = kv_index.limitOf(verdict, lim);
    const need = switch (limit.unit) {
        .elements => at_base.regionElems(),
        .bytes => at_base.regionBytes(),
    };

    return .{
        .window = window,
        .clamp = .{
            .n_slots = n_slots,
            .base = base,
            .need = need,
            .limit = limit,
            .window = window,
        },
    };
}

/// The window a whole-model fused placement plans at: the widest window the VRAM figure funds for
/// `n_slots` slots.
pub fn fitDefault(
    c: kv_budget.Caps,
    cost: kv_budget.KvCostModel,
    n_slots: u32,
    base: u32,
    flag: ?u32,
) struct { window: u32, cut: ?Cut } {
    if (flag) |w| {
        std.debug.assert(base == w);

        return .{ .window = base, .cut = null };
    }

    const fit = kv_budget.fitWindow(c, cost, n_slots, base);
    std.debug.assert(fit <= base);
    if (fit == base) return .{ .window = base, .cut = null };

    return .{ .window = fit, .cut = .{ .n_slots = n_slots, .base = base, .window = fit } };
}

/// The slots a hybrid placement can hold.
pub fn hybridSlots(structural: u32, pinned: bool) error{HybridSlotsPinned}!u32 {
    std.debug.assert(structural >= 1);

    if (structural <= limits.max_batch) return structural;
    if (pinned) return error.HybridSlotsPinned;

    return limits.max_batch;
}

/// The placement a serve split selects. A split with no host layer is the whole-model placement
/// again. A split with no GPU layer is the all-host placement. A zero count reaches here from the
/// bench only, and the bench refuses the result.
pub fn placementKindFor(gpu_layers: ?kv_budget.GpuLayers, split: u32, n_layers: u32) PlacementKind {
    std.debug.assert(n_layers >= 1);
    std.debug.assert(split <= n_layers);

    const flag = gpu_layers orelse return .fused;
    switch (flag) {
        .count => |n| {
            std.debug.assert(n <= n_layers);
            // The planner honors a pinned count exactly, or refuses it.
            std.debug.assert(split == n_layers - n);
        },
        .auto => {},
    }

    if (split == n_layers) return .cpu;
    if (split == 0) return .fused;

    return .hybrid;
}

// -- helpers ----------------------------------------------------------------

/// The serve placements: a split when `--gpu-layers` names one, else the fused placement.
fn servePlacement(
    input: Input,
    card: loader.ModelCard,
    kv_dtype: dtype.KvDtype,
    window: Window,
    resources: resolve.Resources,
) DevicePlacement {
    const shape = input.serve;
    const flag = input.engine.gpu_layers orelse return .{
        .placement = fusedOk(serveFused(input, card, kv_dtype, window, resources)),
    };
    // A zero count takes the host placement before the boot opens a device.
    std.debug.assert(!flag.isZeroCount());

    // Refuse a pin past the model.
    const m = resources.model.?;
    const d = resources.device.?;
    const n_layers: u32 = card.n_layers;
    const pin = flag.pin();
    if (pin) |n| {
        if (n > n_layers) {
            return .{
                .placement = .{
                    .refused = .{ .gpu_layers_past_model = .{ .n = n, .n_layers = n_layers } },
                },
            };
        }
    }

    // Plan the slots and the window the split prices.
    const slots = hybridSlots(shape.structural_slots, shape.slots_req != null) catch
        return .{ .placement = .{ .refused = .{ .hybrid_slots = shape.structural_slots } } };
    const c = caps(card, m, d.vram, d.driver, window.ceiling, slots, kv_dtype);
    const cost = kv_budget.costModel(card.kvShape(), kv_dtype);
    const clamped = clampDefault(
        slots,
        card.kvShape().kv_row_elems,
        kv_dtype,
        d.buffer_limits,
        window.base,
        window.flag,
    );
    // One KV plan runs the Y split, and the placement carries its host vector.
    var plan = planShape(c, cost, .{
        .slots = slots,
        .batch = shape.batch,
        .window = clamped.window,
        .y_bytes = shape.kv_ram_bytes,
        .n_logical = shape.max_sessions,
    });
    const prefix_default = withPrefixDefault(&plan, shape.kv_ram_bytes);
    std.debug.assert(plan.n_slots == slots);
    std.debug.assert(plan.window == clamped.window);

    // Record the lines before the split.
    var layer_choice: LayerChoice = .{
        .pin = pin,
        .slots_from = if (slots < shape.structural_slots) shape.structural_slots else null,
        .clamp = clamped.clamp,
        .window = window,
        .prefix_default = prefix_default,
    };
    layer_choice.window.base = clamped.window;
    layer_choice.window.device_clamped = clamped.clamp != null;

    // Price the split. A residency pool moves blobs through both swap stages, so it prices both.
    const kv: kv_budget.KvStructure = .{
        .n_slots = slots,
        .window = clamped.window,
        .costs = kv_cost.costVector(card, kv_dtype),
    };
    const host: kv_budget.HostRequest = .{ .ram_split = plan.ram, .residency = true };
    // Only a pinned count is refused: the auto rule leaves every layer on the host instead.
    const split = kv_budget.planLayerSplit(c, m.layers(), m.head_weights, kv, host, pin) catch |err|
        return .{
            .layer_choice = layer_choice,
            .placement = .{ .refused = switch (err) {
                error.WeightsVramBudgetTooSmall => layerSplitRefusal(.serve, c, m, kv, pin.?),
                error.NoKvLayerInRange => .{ .layer_split_no_kv = pin.? },
            } },
        };
    layer_choice.layer_split = split;

    // Build the placement that the split selects.
    const selects = placementKindFor(flag, split.host_layers, n_layers);
    layer_choice.selects = selects;
    return .{ .layer_choice = layer_choice, .placement = switch (selects) {
        .fused => fusedOk(serveFused(input, card, kv_dtype, layer_choice.window, resources)),
        .cpu => .{
            .ok = .{
                .cpu = resolve.hostPlacement(.serve, input, card, kv_dtype, window, resources),
            },
        },
        .hybrid => blk: {
            std.debug.assert(split.host_layers > 0);
            std.debug.assert(split.host_layers < n_layers);

            const q = quoteOf(
                kv,
                split.host_layers,
                host,
                shape.max_sessions,
                shape.kv_ram_bytes != null,
                resources.ram_free,
            );
            break :blk .{
                .ok = .{
                    .hybrid = .{
                        .window = layer_choice.window,
                        .quote = q,
                        .plan = pricedOf(split, q),
                    },
                },
            };
        },
    } };
}

/// The whole-model serve placement. A budget flag spends a device budget. Otherwise every session
/// holds a slot at the base window, fitted as the header says.
fn serveFused(
    input: Input,
    card: loader.ModelCard,
    kv_dtype: dtype.KvDtype,
    window: Window,
    resources: resolve.Resources,
) Result(Fused) {
    const shape = input.serve;
    const m = resources.model.?;
    const d = resources.device.?;
    const min_slots = rowValues(&card).fused_min_slots;
    const cost = kv_budget.costModel(card.kvShape(), kv_dtype);
    // The flag checks fixed every count at one or more, so a floor of one keeps each count.
    std.debug.assert(shape.batch >= 1);
    std.debug.assert(shape.structural_slots >= 1);
    if (shape.slots_req) |s| std.debug.assert(s >= 1);
    const kv_row_elems = card.kvShape().kv_row_elems;
    var out: Fused = undefined;
    if (shape.slots_req != null or shape.kv_vram_bytes != null or shape.kv_vram_rest) {
        // Plan a VRAM budget. `--kv-vram` states X and buys the slots. `--vram-slots` pins the
        // count, and the VRAM left after the other reservations buys the window.
        const req_batch: u32 = @max(min_slots, @as(u32, shape.batch));
        const c = caps(card, m, d.vram, d.driver, window.ceiling, req_batch + 1, kv_dtype);
        const slots: ?u32 = if (shape.slots_req) |s| @max(min_slots, @as(u32, s)) else null;

        const budgeted = switch (planBudgetOnDevice(c, cost, .{
            .x_bytes = shape.kv_vram_bytes orelse kv_budget.derivedX(c),
            .y_bytes = shape.kv_ram_bytes,
            .window = input.engine.max_seq,
            .slots = slots,
            .batch = req_batch,
        }, shape.max_sessions, kv_row_elems, kv_dtype, d.buffer_limits)) {
            .ok => |b| b,
            .refused => |r| return .{ .refused = r },
        };

        out = .{
            .caps = c,
            .clamp = budgeted.clamp,
            .window = window,
            .stranded = budgeted.stranded,
            .device_slots = budgeted.device_slots,
            .plan = budgeted.plan,
            .quote = undefined,
        };
    } else {
        // Plan one slot per session at the base window.
        const n_slots: u32 = @max(min_slots, @as(u32, shape.structural_slots));

        const c = caps(card, m, d.vram, d.driver, window.ceiling, n_slots, kv_dtype);
        const clamped = clampDefault(
            n_slots,
            kv_row_elems,
            kv_dtype,
            d.buffer_limits,
            window.base,
            window.flag,
        );
        const fitted = fitDefault(c, cost, n_slots, clamped.window, window.flag);
        out = .{
            .caps = c,
            .clamp = clamped.clamp,
            .window = window,
            .cut = fitted.cut,
            .plan = planShape(c, cost, .{
                .slots = n_slots,
                .batch = shape.batch,
                .window = fitted.window,
                .y_bytes = shape.kv_ram_bytes,
                .n_logical = shape.max_sessions,
            }),
            .quote = undefined,
        };
    }

    // Quote the RAM that the plan needs.
    out.prefix_default = withPrefixDefault(&out.plan, shape.kv_ram_bytes);
    const kv: kv_budget.KvStructure = .{
        .n_slots = out.plan.n_slots,
        .window = out.plan.window,
        .costs = kv_cost.costVector(card, kv_dtype),
    };
    out.quote = quoteOf(
        kv,
        0,
        .{ .ram_split = out.plan.ram, .residency = true },
        shape.max_sessions,
        shape.kv_ram_bytes != null,
        resources.ram_free,
    );
    return .{ .ok = out };
}

/// The `run` and `bench-model` placements: one stream. A split prices one slot at the window the
/// device limits leave.
fn sessionPlacement(
    kind: Kind,
    input: Input,
    card: loader.ModelCard,
    kv_dtype: dtype.KvDtype,
    window: Window,
    resources: resolve.Resources,
) DevicePlacement {
    std.debug.assert(kind != .serve);

    const flag = input.engine.gpu_layers orelse return .{
        .placement = .{
            .ok = .{ .fused = sessionFused(kind, input, card, kv_dtype, window, resources) },
        },
    };
    // A zero count of `run` takes the host placement. The bench prices it, and the whole-model
    // refusal stops it.
    if (kind == .run) std.debug.assert(!flag.isZeroCount());

    // Refuse a pin past the model.
    const m = resources.model.?;
    const d = resources.device.?;
    const n_layers: u32 = card.n_layers;
    const pin = flag.pin();
    if (pin) |n| {
        if (n > n_layers) {
            return .{
                .placement = .{
                    .refused = .{ .gpu_layers_past_model = .{ .n = n, .n_layers = n_layers } },
                },
            };
        }
    }

    // Price the split at the window the device limits leave.
    const c = caps(card, m, d.vram, d.driver, window.ceiling, 1, kv_dtype);
    const clamped = clampSession(window, card, kv_dtype, d.buffer_limits);
    const kv: kv_budget.KvStructure = .{
        .n_slots = 1,
        .window = clamped.window.base,
        .costs = kv_cost.costVector(card, kv_dtype),
    };
    // One stream holds no residency pool, so the stages price zero.
    const host: kv_budget.HostRequest = .{
        .ram_split = .{ .blobs = 0, .prefix = 0 },
        .residency = false,
    };
    var layer_choice: LayerChoice = .{
        .pin = pin,
        .clamp = clamped.clamp,
        .window = clamped.window,
    };
    // Only a pinned count is refused: the auto rule leaves every layer on the host instead.
    const split = kv_budget.planLayerSplit(c, m.layers(), m.head_weights, kv, host, pin) catch |err|
        return .{
            .layer_choice = layer_choice,
            .placement = .{ .refused = switch (err) {
                error.WeightsVramBudgetTooSmall => layerSplitRefusal(kind, c, m, kv, pin.?),
                error.NoKvLayerInRange => .{ .layer_split_no_kv = pin.? },
            } },
        };
    layer_choice.layer_split = split;

    // Select the placement.
    if (kind == .bench) {
        // A whole-model split is a runner that the bench measures without the flag.
        const selected = placementKindFor(flag, split.host_layers, n_layers);
        if (selected != .hybrid) {
            return .{
                .layer_choice = layer_choice,
                .placement = .{ .refused = .{ .whole_model_split = pin } },
            };
        }

        layer_choice.selects = selected;
    } else {
        layer_choice.selects = placementKindFor(flag, split.host_layers, n_layers);
        switch (layer_choice.selects.?) {
            // The host placement runs at the base window. Only the device clamps it.
            .cpu => return .{
                .layer_choice = layer_choice,
                .placement = .{
                    .ok = .{
                        .cpu = resolve.hostPlacement(
                            .run,
                            input,
                            card,
                            kv_dtype,
                            window,
                            resources,
                        ),
                    },
                },
            },
            // The clamped window clamps again to itself.
            .fused => return .{
                .layer_choice = layer_choice,
                .placement = .{
                    .ok = .{
                        .fused = sessionFused(
                            kind,
                            input,
                            card,
                            kv_dtype,
                            clamped.window,
                            resources,
                        ),
                    },
                },
            },
            .hybrid => {},
        }
    }

    // Quote the hybrid.
    const q = quoteOf(kv, split.host_layers, host, 1, false, resources.ram_free);
    const plan = pricedOf(split, q);
    if (plan) |p| {
        std.debug.assert(p.host_layers >= 1);
        std.debug.assert(p.host_layers < n_layers);
    }

    return .{
        .layer_choice = layer_choice,
        .placement = .{
            .ok = .{ .hybrid = .{ .window = clamped.window, .quote = q, .plan = plan } },
        },
    };
}

/// The whole-model placement of one stream, fitted as the header says.
fn sessionFused(
    kind: Kind,
    input: Input,
    card: loader.ModelCard,
    kv_dtype: dtype.KvDtype,
    window: Window,
    resources: resolve.Resources,
) Fused {
    const m = resources.model.?;
    const d = resources.device.?;
    const c = caps(card, m, d.vram, d.driver, window.ceiling, 1, kv_dtype);
    const cost = kv_budget.costModel(card.kvShape(), kv_dtype);
    const clamped = clampSession(window, card, kv_dtype, d.buffer_limits);
    std.debug.assert(kv_budget.windowCeil(c) == window.ceiling);
    const fitted = fitDefault(c, cost, 1, clamped.window.base, clamped.window.flag);
    std.debug.assert(fitted.window <= clamped.window.base);
    // The batch of one states the one slot, and the plan funds no RAM.
    const row = rowValues(&card);
    const plan = planShape(c, cost, .{ .window = fitted.window, .batch = 1 });
    std.debug.assert(plan.n_slots == 1);
    std.debug.assert(plan.ram.blobs == 0);
    std.debug.assert(plan.ram.prefix == 0);
    // The run quote of a family that honors the prompt cache prices its snapshots and stages.
    const cache = kind == .run and row.prompt_cache and input.prompt_cache;
    const ram_split: kv_budget.KvPlan.Ram = .{
        .blobs = 0,
        .prefix = if (cache) limits.prompt_cache_ram_default else 0,
    };

    const kv: kv_budget.KvStructure = .{
        .n_slots = plan.n_slots,
        .window = plan.window,
        .costs = kv_cost.costVector(card, kv_dtype),
    };

    // Assemble the placement.
    return .{
        .caps = c,
        .clamp = clamped.clamp,
        .window = clamped.window,
        .cut = fitted.cut,
        .plan = plan,
        .quote = quoteOf(
            kv,
            0,
            .{ .ram_split = ram_split, .residency = cache },
            1,
            false,
            resources.ram_free,
        ),
        .report = if (kind == .run and row.run_logs_every_placement) report(
            c,
            card,
            m,
            kv_dtype,
            plan,
        ) else null,
    };
}

/// `clampDefault` for one stream. The result records the cut, and a second call cuts nothing.
fn clampSession(
    w: Window,
    card: loader.ModelCard,
    kv_dtype: dtype.KvDtype,
    lim: limits.BufferLimits,
) struct { window: Window, clamp: ?Clamp } {
    const clamped = clampDefault(1, card.kvShape().kv_row_elems, kv_dtype, lim, w.base, w.flag);
    std.debug.assert(clamped.window <= w.base);
    if (clamped.window == w.base) return .{ .window = w, .clamp = null };

    std.debug.assert(w.flag == null);
    var out = w;
    out.base = clamped.window;
    out.device_clamped = true;
    return .{ .window = out, .clamp = clamped.clamp };
}

/// A plan that buys no slot with money. The window lies under the ceiling, so the planner never
/// refuses.
fn planShape(
    c: kv_budget.Caps,
    cost: kv_budget.KvCostModel,
    req: kv_budget.Request,
) kv_budget.KvPlan {
    std.debug.assert(req.x_bytes == null);
    std.debug.assert(req.window.? <= kv_budget.windowCeil(c));

    return kv_budget.kvPlan(c, cost, req) catch unreachable;
}

/// Zero-config serve keeps the prefix cache. With no `--kv-ram`, the prefix budget takes the serve
/// default, and the RAM quote funds the full blob need beside it or refuses.
fn withPrefixDefault(plan: *kv_budget.KvPlan, kv_ram_bytes: ?u64) bool {
    if (kv_ram_bytes != null) return false;
    plan.ram.prefix = limits.serve_prefix_ram_default;
    return true;
}

/// Turn a KV VRAM byte budget into `n_slots x window`, or a refusal with the numbers.
fn planBudget(
    caps_in: kv_budget.Caps,
    cost: kv_budget.KvCostModel,
    req: kv_budget.Request,
    n_logical: u32,
    clamp: ?Clamp,
) Result(Budgeted) {
    std.debug.assert(req.x_bytes != null);
    std.debug.assert(n_logical >= 1);
    std.debug.assert(req.n_logical == null);

    var r = req;
    r.n_logical = n_logical;
    // Cap the slots at the session count, because a slot no session can occupy is dead VRAM.
    // `req.batch` can carry the llama spare-stream floor, so the max keeps that floor legal.
    var c = caps_in;
    c.max_vram_slots = @min(caps_in.max_vram_slots, @max(req.batch, n_logical));
    // A decode row owns a slot, so the width ceiling never exceeds the slot ceiling.
    c.max_batch = @min(c.max_batch, c.max_vram_slots);

    // Plan, or refuse with the numbers.
    const plan = kv_budget.kvPlan(c, cost, r) catch |err| switch (err) {
        error.WindowExceedsCeiling => return .{
            .refused = .{
                .kv_vram_ceiling = .{ .window = r.window.?, .ceiling = kv_budget.windowCeil(c) },
            },
        },
        error.KvVramBudgetTooSmall => {
            // The window resolved before the refusal, so the quote resolves the same one.
            const q = kv_budget.quote(c, cost, r) catch unreachable;
            return .{ .refused = .{ .kv_vram_small = .{
                .x_bytes = r.x_bytes.?,
                .rows = @max(r.batch, r.slots orelse 0),
                .per_slot = q.per_slot_bytes,
                .window = q.window,
                .clamp = clamp,
            } } };
        },
    };

    // Cap and signal: the ceiling cut the slot count, so the remainder buys nothing.
    var out: Budgeted = .{ .plan = plan, .clamp = clamp };
    if (plan.slots_clamped) {
        out.stranded = .{
            .max_slots = c.max_vram_slots,
            .stranded = r.x_bytes.? - plan.vram.kv,
            .x_bytes = r.x_bytes.?,
        };
    }

    return .{ .ok = out };
}

/// Plan a VRAM budget inside the device buffer limits.
fn planBudgetOnDevice(
    caps_in: kv_budget.Caps,
    cost: kv_budget.KvCostModel,
    req: kv_budget.Request,
    n_logical: u32,
    kv_row_elems: u32,
    kv_dtype: dtype.KvDtype,
    lim: limits.BufferLimits,
) Result(Budgeted) {
    // Pinned slots: the money buys the window, so the clamp runs on the plan.
    if (req.slots != null) {
        const first = switch (planBudget(caps_in, cost, req, n_logical, null)) {
            .ok => |b| b,
            .refused => |r| return .{ .refused = r },
        };

        std.debug.assert(first.stranded == null);
        if (req.window != null) return .{ .ok = first };

        const clamped = clampDefault(
            first.plan.n_slots,
            kv_row_elems,
            kv_dtype,
            lim,
            first.plan.window,
            null,
        );
        if (clamped.window == first.plan.window) return .{ .ok = first };

        std.debug.assert(clamped.window < first.plan.window);
        var pinned = req;
        pinned.window = clamped.window;
        const second = switch (planBudget(caps_in, cost, pinned, n_logical, clamped.clamp)) {
            .ok => |b| b,
            .refused => |r| return .{ .refused = r },
        };

        std.debug.assert(second.plan.window == clamped.window);
        std.debug.assert(second.plan.n_slots == first.plan.n_slots);
        return .{ .ok = second };
    }

    // Unpinned slots: fix the window first, then let the limits cap the slot count.
    var c = caps_in;
    var clamp: ?Clamp = null;
    if (req.window == null) {
        const clamped = clampDefault(1, kv_row_elems, kv_dtype, lim, c.serve_default, null);
        c.serve_default = clamped.window;
        clamp = clamped.clamp;
    }

    std.debug.assert(c.serve_default <= c.default_ceil);
    const window = @min(req.window orelse c.serve_default, kv_budget.windowCeil(c));
    const device_slots = kv_index.maxSlotsForLimits(window, kv_row_elems, kv_dtype, lim);
    // Fewer slots than decode rows leaves a row with no slot, so the runner refuses the shape.
    if (device_slots < req.batch) return planBudget(c, cost, req, n_logical, clamp);

    c.max_vram_slots = @min(c.max_vram_slots, device_slots);
    var out = switch (planBudget(c, cost, req, n_logical, clamp)) {
        .ok => |b| b,
        .refused => |r| return .{ .refused = r },
    };

    std.debug.assert(out.plan.window == window);
    std.debug.assert(out.plan.n_slots <= device_slots);
    if (out.plan.slots_clamped and out.plan.n_slots == device_slots) {
        out.device_slots = deviceSlotsOf(window, device_slots, kv_row_elems, kv_dtype, lim);
    }

    return .{ .ok = out };
}

/// The device limit that caps a budget at `slots` slots of `window` positions.
fn deviceSlotsOf(
    window: u32,
    slots: u32,
    kv_row_elems: u32,
    kv_dtype: dtype.KvDtype,
    lim: limits.BufferLimits,
) DeviceSlots {
    std.debug.assert(slots >= 1);

    // Scale the region of one slot, because a layout of `slots + 1` slots can pass the ceiling.
    const one_slot = kv_index.Layout.init(1, window, kv_row_elems, 1, kv_dtype);
    const next: u64 = @as(u64, slots) + 1;
    const verdict = limits.bufferFits(
        next * one_slot.regionElems(),
        next * one_slot.regionBytes(),
        lim,
    );
    std.debug.assert(verdict != .fits);
    std.debug.assert(verdict != .misaligned);
    return .{ .limit = kv_index.limitOf(verdict, lim), .slots = slots, .window = window };
}

/// The RAM quote of one placement at split `k`, or the refusal that names it.
fn quoteOf(
    kv: kv_budget.KvStructure,
    k: u32,
    host: kv_budget.HostRequest,
    n_logical: u32,
    named: bool,
    ram_free: ?u64,
) Result(kv_budget.RamQuote) {
    const q = kv_budget.quoteRam(kv, k, host, n_logical, named, ram_free) catch |err| switch (err) {
        error.HostRamBudgetTooSmall => return .{
            .refused = .{ .host_ram = kv_budget.ramRefusal(kv, k, host, ram_free.?) },
        },
        error.BlobRamBudgetTooSmall => return .{
            .refused = .{ .blob_ram = kv_budget.blobRefusal(kv, k, host, n_logical, ram_free.?) },
        },
    };
    return .{ .ok = q };
}

fn pricedOf(
    split: kv_budget.LayerSplit,
    q: Result(kv_budget.RamQuote),
) ?kv_budget.LayerSplit {
    return switch (q) {
        .ok => |v| split.priced(v),
        .refused => null,
    };
}

/// A pinned split the VRAM figure cannot fund, with its cost. Only a count reaches it.
fn layerSplitRefusal(
    kind: Kind,
    c: kv_budget.Caps,
    m: ModelNeeds,
    kv: kv_budget.KvStructure,
    n_gpu: u32,
) Refusal {
    const q = kv_budget.quoteLayerSplit(c, m.layers(), m.head_weights, kv, n_gpu);
    return .{
        .layer_split_vram = .{ .kind = kind, .n_gpu = n_gpu, .spend = q.spend, .usable = q.usable },
    };
}

/// The values of the family row of `card`.
fn rowValues(card: *const loader.ModelCard) RowValues {
    return switch (card.arch) {
        inline else => |arch| .{
            .fused_min_slots = family.Of(arch).fused_min_slots,
            .prompt_cache = family.Of(arch).prompt_cache,
            .run_logs_every_placement = family.Of(arch).run_logs_every_placement,
        },
    };
}

fn fusedOk(r: Result(Fused)) Result(Placement) {
    return switch (r) {
        .ok => |f| .{ .ok = .{ .fused = f } },
        .refused => |x| .{ .refused = x },
    };
}

// -- tests ------------------------------------------------------------------

/// A translation-layer driver: the spec-minimum storage range, a 2 GiB
/// allocation, and 16-byte binding offsets.
const test_spec_min_limits: limits.BufferLimits = .{
    .storage_range = 128 << 20,
    .allocation_size = 2 << 30,
    .offset_alignment = 16,
};

test "hybridSlots: a pin over the ceiling refuses, a derivation clamps" {
    const cap = limits.max_batch;
    try std.testing.expectError(error.HybridSlotsPinned, hybridSlots(2 * cap, true));
    try std.testing.expectEqual(cap, try hybridSlots(cap, true));
    try std.testing.expectEqual(cap, try hybridSlots(cap + 1, false));
    try std.testing.expectEqual(@as(u32, 4), try hybridSlots(4, false));
    try std.testing.expectEqual(@as(u32, 1), try hybridSlots(1, true));
}

test "fitDefault: a named window never shrinks, and a default window shrinks to the budget" {
    // Llama-3.1-8B at f32 KV, four slots at the serve default of 512.
    const cost: kv_budget.KvCostModel = .{ .kv_bytes_per_token = 262_144 };
    var c: kv_budget.Caps = .{
        .vram_free = 0,
        .weights = 0,
        .logits = 0,
        .model_ctx = 131_072,
        .serve_default = 512,
        .default_ceil = 8192,
    };
    // A zero budget funds nothing, so a named window stands and the preflight
    // refuses it, while a default window drops to the floor.
    try std.testing.expectEqual(@as(u32, 512), fitDefault(c, cost, 4, 512, 512).window);
    try std.testing.expectEqual(limits.min_kv_window, fitDefault(c, cost, 4, 512, null).window);
    // A budget that funds the default leaves it whole.
    c.vram_free = 16 << 30;
    try std.testing.expectEqual(@as(u32, 512), fitDefault(c, cost, 4, 512, null).window);
}

test "clampDefault: a default shrinks to the device limits, and a named window never does" {
    const lim = test_spec_min_limits;
    // The 2B run default of 174592 needs a 170 MiB region at f16, so it
    // shrinks to 131072. The same window named on the flag stands, and the
    // runner refuses it.
    try std.testing.expectEqual(
        @as(u32, 131_072),
        clampDefault(1, 512, .f16, lim, 174_592, null).window,
    );
    try std.testing.expectEqual(
        @as(u32, 174_592),
        clampDefault(1, 512, .f16, lim, 174_592, 174_592).window,
    );
    // Defaults inside the limits stand: the 2B serve default at four slots,
    // the 8B run default at f16, and the SmolLM2 ceiling.
    try std.testing.expectEqual(
        @as(u32, 10_752),
        clampDefault(4, 512, .f16, lim, 10_752, null).window,
    );
    try std.testing.expectEqual(
        @as(u32, 16_384),
        clampDefault(1, 1024, .f16, lim, 16_384, null).window,
    );
    try std.testing.expectEqual(@as(u32, 8192), clampDefault(1, 192, .f32, lim, 8192, null).window);
    // A default at the deepest window stands, and one position more shrinks to
    // the granule floor.
    try std.testing.expectEqual(
        @as(u32, 65_536),
        clampDefault(1, 1024, .f16, lim, 65_536, null).window,
    );
    try std.testing.expectEqual(
        @as(u32, 65_536),
        clampDefault(1, 1024, .f16, lim, 65_537, null).window,
    );
    // A window off the granule that still fits stands whole. The range here
    // holds 4010 rows of 768 bytes.
    var odd = lim;
    odd.storage_range = 4010 * 768 + 100;
    try std.testing.expectEqual(@as(u32, 4000), clampDefault(1, 192, .f32, odd, 4000, null).window);
    try std.testing.expectEqual(@as(u32, 3840), clampDefault(1, 192, .f32, odd, 4096, null).window);
    // When no granule fits, the default stands and the runner refuses it.
    var narrow = lim;
    narrow.storage_range = 64 << 20;
    try std.testing.expectEqual(
        @as(u32, 512),
        clampDefault(limits.max_vram_slots, 1024, .f32, narrow, 512, null).window,
    );
}

test "clampSession: the cut is recorded once, and a named window passes" {
    const lim = test_spec_min_limits;
    // The 2B run default at f16: one KV row of 512 elements per position.
    var card: loader.ModelCard = undefined;
    card.n_layers = 24;
    card.n_kv_heads = 2;
    card.head_dim = 256;
    card.gdn = null;
    const run: Window = .{
        .base = 174_592,
        .ceiling = limits.max_model_ctx,
        .flag = null,
        .session_bytes = limits.run_session_kv_bytes,
    };
    const clamped = clampSession(run, card, .f16, lim);
    try std.testing.expectEqual(@as(u32, 131_072), clamped.window.base);
    try std.testing.expect(clamped.window.device_clamped);
    try std.testing.expect(clamped.clamp != null);
    try std.testing.expectEqual(run.ceiling, clamped.window.ceiling);
    try std.testing.expectEqual(@as(?u32, null), clamped.window.flag);
    // A second clamp finds the window inside the limits and changes nothing.
    const again = clampSession(clamped.window, card, .f16, lim);
    try std.testing.expectEqual(clamped.window, again.window);
    try std.testing.expect(again.clamp == null);
    const named: Window = .{
        .base = 262_144,
        .ceiling = limits.max_model_ctx,
        .flag = 262_144,
        .session_bytes = limits.run_session_kv_bytes,
    };
    try std.testing.expectEqual(named, clampSession(named, card, .f16, lim).window);
}

test "planBudgetOnDevice: pinned slots cut the window, and bought slots stop at the device limits" {
    const lim = test_spec_min_limits;
    // Qwen3.5-2B at f16, with the run and serve defaults of its byte rule.
    const cost: kv_budget.KvCostModel = .{
        .kv_bytes_per_token = 12_288,
        .fixed_state_bytes_per_slot = 20_201_472,
    };
    const c: kv_budget.Caps = .{
        .vram_free = 0,
        .weights = 0,
        .logits = 0,
        .model_ctx = limits.max_model_ctx,
        .serve_default = 10_752,
        .default_ceil = 174_592,
    };
    // Pinned slots: one slot of 4 GiB buys the default ceiling, which the
    // storage range cuts to 131072.
    const pinned = planBudgetOnDevice(
        c,
        cost,
        .{ .x_bytes = 4 << 30, .slots = 1, .batch = 1 },
        1,
        512,
        .f16,
        lim,
    ).ok.plan;
    try std.testing.expectEqual(@as(u32, 1), pinned.n_slots);
    try std.testing.expectEqual(@as(u32, 131_072), pinned.window);
    try std.testing.expectEqual(@as(u64, 131_072) * 12_288 + 20_201_472, pinned.vram.kv);
    // Bought slots: 16 GiB funds 112 slots at the serve default, but one
    // region holds 12 of them under the range. The planner clamps the count
    // and keeps the window.
    const bought = planBudgetOnDevice(
        c,
        cost,
        .{ .x_bytes = 16 << 30, .batch = 1 },
        limits.max_logical_sessions,
        512,
        .f16,
        lim,
    ).ok.plan;
    try std.testing.expectEqual(@as(u32, 12), bought.n_slots);
    try std.testing.expectEqual(@as(u32, 10_752), bought.window);
    try std.testing.expect(bought.slots_clamped);
    try std.testing.expectEqual(
        limits.BufferVerdict.fits,
        kv_index.fits(kv_index.Layout.init(bought.n_slots, bought.window, 512, 6, .f16), lim),
    );
    // A budget that funds fewer slots than the region holds is not clamped.
    const small = planBudgetOnDevice(
        c,
        cost,
        .{ .x_bytes = 1 << 30, .batch = 1 },
        limits.max_logical_sessions,
        512,
        .f16,
        lim,
    ).ok.plan;
    try std.testing.expectEqual(@as(u32, 7), small.n_slots);
    try std.testing.expect(!small.slots_clamped);
    // A named window never shrinks, pinned or bought, and it caps the slots.
    const named = planBudgetOnDevice(
        c,
        cost,
        .{ .x_bytes = 4 << 30, .slots = 1, .batch = 1, .window = 174_592 },
        1,
        512,
        .f16,
        lim,
    ).ok.plan;
    try std.testing.expectEqual(@as(u32, 174_592), named.window);
    const named_bought = planBudgetOnDevice(
        c,
        cost,
        .{ .x_bytes = 16 << 30, .batch = 1, .window = 32_768 },
        limits.max_logical_sessions,
        512,
        .f16,
        lim,
    ).ok.plan;
    try std.testing.expectEqual(@as(u32, 32_768), named_bought.window);
    try std.testing.expectEqual(@as(u32, 4), named_bought.n_slots);
    // When one slot at the window already breaks a limit, the cap stays off
    // and the runner refuses the shape.
    const refused = planBudgetOnDevice(
        c,
        cost,
        .{ .x_bytes = 16 << 30, .batch = 1, .window = 262_144 },
        1,
        512,
        .f16,
        lim,
    ).ok.plan;
    try std.testing.expectEqual(@as(u32, 262_144), refused.window);
    try std.testing.expectEqual(
        limits.BufferVerdict.past_storage_range,
        kv_index.fits(kv_index.Layout.init(refused.n_slots, refused.window, 512, 6, .f16), lim),
    );
}

test "placementKindFor: the flag and the resolved split select the runner" {
    const Case = struct { flag: ?kv_budget.GpuLayers, host_layers: u32, want: PlacementKind };
    const n: u32 = 32;
    const cases = [_]Case{
        // No flag keeps the whole-model path.
        .{ .flag = null, .host_layers = 0, .want = .fused },
        // A funded count of every layer is the whole-model path again.
        .{ .flag = .{ .count = n }, .host_layers = 0, .want = .fused },
        .{ .flag = .{ .count = 16 }, .host_layers = 16, .want = .hybrid },
        // The boundary pins: a suffix of one layer, and a prefix of one.
        .{ .flag = .{ .count = 1 }, .host_layers = n - 1, .want = .hybrid },
        .{ .flag = .{ .count = n - 1 }, .host_layers = 1, .want = .hybrid },
        // Auto selects by the split alone. The two degenerate splits are the
        // whole-model runners, one per tier.
        .{ .flag = .auto, .host_layers = 0, .want = .fused },
        .{ .flag = .auto, .host_layers = 1, .want = .hybrid },
        .{ .flag = .auto, .host_layers = 15, .want = .hybrid },
        .{ .flag = .auto, .host_layers = n - 1, .want = .hybrid },
        .{ .flag = .auto, .host_layers = n, .want = .cpu },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c.want, placementKindFor(c.flag, c.host_layers, n));
    }
}

test "placementKindFor: the qwen 24-layer stack selects through the same placements" {
    // The qwen placement reuses `placementKindFor` unchanged, so the 2B's 24 layers select the
    // same way the Llama 32 do. The boundary values are the gate points: a
    // suffix of one layer, and a prefix of one.
    const n: u32 = 24;
    try std.testing.expectEqual(PlacementKind.fused, placementKindFor(null, 0, n));
    try std.testing.expectEqual(PlacementKind.fused, placementKindFor(.{ .count = n }, 0, n));
    for ([_]u32{ 1, 2, 6, 12, 18, 22, 23 }) |g| {
        try std.testing.expectEqual(
            PlacementKind.hybrid,
            placementKindFor(.{ .count = g }, n - g, n),
        );
        try std.testing.expectEqual(PlacementKind.hybrid, placementKindFor(.auto, n - g, n));
    }

    try std.testing.expectEqual(PlacementKind.cpu, placementKindFor(.auto, n, n));
}

test "placementKindFor: the four-layer route of the placement model" {
    // The anchor of the route in the placement model: split 0 is the fused runner, a split at the
    // layer count is the CPU runner, and the splits between are the hybrid.
    const want = [_]PlacementKind{ .fused, .hybrid, .hybrid, .hybrid, .cpu };
    for (want, 0..) |kind, host_layers| {
        try std.testing.expectEqual(kind, placementKindFor(.auto, @intCast(host_layers), 4));
    }
}

/// The three rules of the route, stated once for the grid below.
fn expectedKind(host_layers: u32, n_layers: u32) PlacementKind {
    if (host_layers == 0) return .fused;
    if (host_layers == n_layers) return .cpu;
    return .hybrid;
}

test "placementKindFor: the flag follows the route, so the hybrid sees interior counts only" {
    for ([_]u32{ 1, 4, 24, 30 }) |n_layers| {
        // Auto selects by the host layer count alone, at every count the planner can return.
        var host_layers: u32 = 0;
        while (host_layers <= n_layers) : (host_layers += 1) {
            const kind = placementKindFor(.auto, host_layers, n_layers);
            try std.testing.expectEqual(expectedKind(host_layers, n_layers), kind);
            if (kind == .hybrid) {
                try std.testing.expect(host_layers >= 1);
                try std.testing.expect(host_layers < n_layers);
            }
        }

        // A pinned count of GPU layers leaves the rest on the host.
        var n_gpu: u32 = 1;
        while (n_gpu <= n_layers) : (n_gpu += 1) {
            try std.testing.expectEqual(
                expectedKind(n_layers - n_gpu, n_layers),
                placementKindFor(.{ .count = n_gpu }, n_layers - n_gpu, n_layers),
            );
        }
    }
}

test "hybridSlots: the qwen placement inherits the eight-slot ceiling" {
    // The qwen placement inherits the cap rather than lifting it, so a qwen serve
    // derives the same clamp and refuses the same pin the Llama placement does.
    try std.testing.expectEqual(@as(u32, 8), limits.max_batch);
    try std.testing.expectError(error.HybridSlotsPinned, hybridSlots(16, true));
    try std.testing.expectEqual(@as(u32, 8), try hybridSlots(16, false));
    try std.testing.expectEqual(@as(u32, 8), try hybridSlots(8, true));
}
