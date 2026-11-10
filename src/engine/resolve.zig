//! Role. The boot decisions of `run`, `serve`, and `bench-model` as one pure step. The CLI
//! measures the host once (`Resources`), calls `early` before the model loads and `resolve` after,
//! logs the `Resolved` value, and formats a `Refusal`. Nothing here reads a clock, a file, or a
//! device, and nothing logs, so a table test runs every boot shape with no GPU.
//!
//! Comptime. Build options pick the Vulkan placements, so a CPU build drops them. `Result` is
//! generic over its value type.
//!
//! Per-kind rules. Each difference between the kinds is on purpose.
//!
//!   rule                        run               serve                     bench
//!   default window budget       run budget        serve budget              run budget
//!   slots of the CPU placement  1                 the structural slots      1
//!   residency priced in RAM     no                yes                       no
//!   prefix default (no Y)       none              the serve default         none
//!   whole-model split           falls through     the placement it selects  refuses
//!                               to the placement
//!   thinking refusals           `on`              `on` and a budget         none (auto)
//!   per-op vulkan refusals      f16 KV            every boot                f16 KV
//!
//! Refusal order. Every refusal that needs no model file is in `early`, so it lands before the
//! load.
//!
//! Device placements. The fused and hybrid placements live in `device_placement.zig`. They read one
//! VRAM figure, the live budget, which the CLI measures from one device context and passes in
//! `Resources.device`.
//!
//! RAM refusals. A RAM refusal of the host placement sits in `Resolved.cpu`, so the CLI reports it
//! where the placement starts, after the boot lines that come before it. A device placement that
//! falls back to the host calls `hostPlacement` itself.
//!
//! Window. A named window must lie in `1..ceiling`. Only a default window shrinks to fit the VRAM
//! or the device buffer limits.

const std = @import("std");
const build_options = @import("build_options");

const limits = @import("../core/limits.zig");
const dtype = @import("../core/dtype.zig");
const chat = @import("../core/chat.zig");
const kv_budget = @import("../core/kv_budget.zig");
const kv_index = @import("../core/kv_index.zig");
const loader = @import("../io/gguf/loader.zig");
const kv_cost = @import("../runtime/kv_cost.zig");
const parallel = @import("../runtime/parallel.zig");
const flags_mod = @import("flags.zig");
pub const device_placement = @import("device_placement.zig");

pub const EngineFlags = flags_mod.EngineFlags;
pub const ModelNeeds = device_placement.ModelNeeds;
pub const DeviceCapacity = device_placement.DeviceCapacity;
pub const DevicePlacement = device_placement.DevicePlacement;
pub const caps = device_placement.caps;

// -- types ------------------------------------------------------------------

pub const Kind = enum { run, serve, bench };

/// The registry values of the model: what the boot knows before the load.
pub const Entry = struct {
    pub_name: []const u8,
    chat_template_kind: chat.Kind,
    kv_dtype_default: dtype.KvDtype,
    context_limit: ?u32,
};

/// The serve shape the flags checks already fixed. Run and bench hold one
/// slot and leave the defaults.
pub const ServeShape = struct {
    max_sessions: u16 = 1,
    /// The slots every session holds when no device buys a count.
    structural_slots: u16 = 1,
    /// `--kv-ram` in bytes. Null names no RAM budget.
    kv_ram_bytes: ?u64 = null,
    /// The decode width.
    batch: u16 = 1,
    /// `--vram-slots`, capped at the session count. Null names no count.
    slots_req: ?u16 = null,
    /// `--kv-vram` in bytes. Null names no device budget.
    kv_vram_bytes: ?u64 = null,
    /// Buy slots with all the VRAM that the other reservations leave. `kv_vram_bytes` wins.
    kv_vram_rest: bool = false,
};

pub const Input = struct {
    engine: EngineFlags = .{},
    serve: ServeShape = .{},
    /// `run --prompt-cache`. The qwen fused placement prices the cache in its RAM
    /// quote.
    prompt_cache: bool = false,
};

/// What the CLI measures once per boot.
pub const Resources = struct {
    /// `MemAvailable` at boot. Null turns the RAM check off.
    ram_free: ?u64,
    /// The weight census and the fused runner reservations. Null skips the
    /// placement line of the host placement.
    model: ?ModelNeeds = null,
    /// What the device context states. Null resolves no device placement.
    device: ?DeviceCapacity = null,
};

/// The window of a boot before any device fit, and the figures its log line
/// names.
pub const Window = struct {
    /// The `--max-seq` value, or the default window when the flag is absent.
    base: u32,
    /// The model ceiling from `limits.contextCeiling`.
    ceiling: u32,
    /// Null when the default applies.
    flag: ?u32,
    /// The byte budget the default window reads.
    session_bytes: u64,
    /// True when a device clamp cut the default to the buffer limits and logged the window line.
    device_clamped: bool = false,
};

/// The decisions that need no model file.
pub const Early = struct {
    /// The chat kind with the thinking mode applied.
    chat_kind: chat.Kind,
    kv_dtype: dtype.KvDtype,
};

/// The all-host placement: the plan, and the RAM quote of the split that keeps
/// every layer on the host.
pub const HostPlacement = struct {
    plan: kv_budget.KvPlan,
    /// True when no `--kv-ram` named a Y, so the prefix took the serve default.
    prefix_default: bool,
    quote: kv_budget.RamQuote,
};

/// The host placement of a `run` of a llama model: the plan and the placement line.
pub const RunHost = struct {
    plan: kv_budget.KvPlan,
    /// Null when `Resources` holds no weight census.
    report: ?kv_budget.LayerSplit,
};

pub const Resolved = struct {
    early: Early,
    window: Window,
    /// The all-host placement, or null for a device placement.
    cpu: ?Result(HostPlacement),
    /// The device placement, when the boot opens a device and `Resources` states it.
    device: ?DevicePlacement = null,
};

/// Why a boot stops. The CLI formats each tag with its message.
pub const Refusal = union(enum) {
    thinking_unsupported,
    thinking_budget_unsupported,
    serve_per_op_vulkan,
    kv_dtype_per_op_vulkan: dtype.KvDtype,
    window_out_of_range: u32,
    cpu_slots: struct { slots: u16, sessions: u16 },
    host_ram: kv_budget.RamRefusal,
    blob_ram: kv_budget.BlobRefusal,
    gpu_layers_past_model: struct { n: u32, n_layers: u32 },
    /// The structural slot count a pinned `--vram-slots` asks of a split.
    hybrid_slots: u32,
    layer_split_vram: struct { kind: Kind, n_gpu: u32, spend: u64, usable: u64 },
    /// The pinned GPU layer count of a split whose GPU layers hold no KV layer.
    layer_split_no_kv: u32,
    /// The pin of a bench split that keeps the whole model on one tier. Null
    /// is the auto pin.
    whole_model_split: ?u32,
    kv_vram_ceiling: struct { window: u32, ceiling: u32 },
    kv_vram_small: struct {
        x_bytes: u64,
        rows: u64,
        per_slot: u64,
        window: u32,
        clamp: ?device_placement.Clamp,
    },
};

pub fn Result(comptime T: type) type {
    return union(enum) { ok: T, refused: Refusal };
}

// -- api --------------------------------------------------------------------

/// True when the boot opens a device context. The bench of a fused build always opens one,
/// because it prices a zero count to refuse it as a whole model.
pub fn needsDevice(kind: Kind, flags: EngineFlags, card: loader.ModelCard) bool {
    if (comptime build_options.use_vulkan_fused_ops) {
        if (kind == .bench) return true;
        const zero = if (flags.gpu_layers) |g| g.isZeroCount() else false;
        return !zero;
    }

    if (comptime build_options.use_vulkan_ops) {
        return kind != .serve and card.arch != .qwen35;
    }

    return false;
}

/// The decisions that need only the flags and the registry entry. Every
/// refusal here lands before the load.
pub fn early(kind: Kind, input: Input, entry: Entry) Result(Early) {
    const flags = input.engine;
    const thinking: chat.ThinkingFlag = if (kind == .bench) .auto else flags.thinking;
    const chat_kind = chat.resolveThinking(entry.chat_template_kind, thinking) catch
        return .{ .refused = .thinking_unsupported };
    if (kind == .serve and flags.thinking_budget > 0 and chat_kind != .chatml_qwen) {
        return .{ .refused = .thinking_budget_unsupported };
    }

    const kv_dtype = flags.kv_dtype orelse entry.kv_dtype_default;
    if (comptime build_options.use_vulkan_ops) {
        // The serve paths never start the per-op kernel context, and the per-op backend uploads f32
        // KV.
        if (kind == .serve) return .{ .refused = .serve_per_op_vulkan };
        if (kv_dtype != .f32) return .{ .refused = .{ .kv_dtype_per_op_vulkan = kv_dtype } };
    }

    return .{ .ok = .{ .chat_kind = chat_kind, .kv_dtype = kv_dtype } };
}

/// The whole boot after the load. `card` is the parsed model card.
pub fn resolve(
    kind: Kind,
    input: Input,
    entry: Entry,
    card: loader.ModelCard,
    resources: Resources,
) Result(Resolved) {
    const e = switch (early(kind, input, entry)) {
        .ok => |v| v,
        .refused => |r| return .{ .refused = r },
    };

    // Size the base window.
    const ceiling = limits.contextCeiling(card.context_length, entry.context_limit);
    const cost = kv_budget.costModel(card.kvShape(), e.kv_dtype);
    const session_bytes = if (kind == .serve)
        limits.serve_session_kv_bytes
    else
        limits.run_session_kv_bytes;

    const default_window = kv_budget.defaultWindow(cost, ceiling, session_bytes);
    const base = baseWindow(input.engine.max_seq, ceiling, default_window) catch
        return .{ .refused = .{ .window_out_of_range = ceiling } };
    const window: Window = .{
        .base = base,
        .ceiling = ceiling,
        .flag = input.engine.max_seq,
        .session_bytes = session_bytes,
    };
    std.debug.assert(window.base <= window.ceiling);

    // Pick the host placement or the device placement.
    var out: Resolved = .{ .early = e, .window = window, .cpu = null };
    if (!needsDevice(kind, input.engine, card)) {
        out.cpu = hostPlacement(kind, input, card, e.kv_dtype, window, resources);
    } else if (resources.device != null) {
        out.device = device_placement.choosePlacement(
            kind,
            input,
            card,
            e.kv_dtype,
            window,
            resources,
        );
    }

    return .{ .ok = out };
}

/// The input of a `serve` boot of `sessions` sessions that buys slots with all the VRAM the other
/// reservations leave, as `--max-sessions` with a `--kv-vram` of that size does.
pub fn spendVram(input: Input, sessions: u16) Input {
    std.debug.assert(sessions >= 1);

    const width: u16 = @intCast(@min(@as(u32, sessions), limits.max_batch));
    var out = input;
    out.serve = .{
        .max_sessions = sessions,
        .structural_slots = width,
        .batch = width,
        .kv_vram_rest = true,
    };
    return out;
}

/// The most sessions, at most `cap`, that `spendVram` gives a slot each. Null when no count fits.
/// The walk starts at `cap`, because a plan refuses a decode width it cannot fund, so a refused
/// count states nothing about the count that fits.
pub fn mostSessions(
    input: Input,
    entry: Entry,
    card: loader.ModelCard,
    resources: Resources,
    cap: u16,
) ?u16 {
    std.debug.assert(cap >= 1);
    std.debug.assert(cap <= limits.max_logical_sessions);

    var sessions = cap;
    while (sessions >= 1) : (sessions -= 1) {
        const shaped = spendVram(input, sessions);
        if (eachSessionHasSlot(shaped, entry, card, resources)) return sessions;
    }

    return null;
}

/// The all-host placement at `w`: the plan and its RAM quote. Serve holds every session resident
/// and prices the residency stages. Run and bench hold one slot and price no stage.
pub fn hostPlacement(
    kind: Kind,
    input: Input,
    card: loader.ModelCard,
    kv_dtype: dtype.KvDtype,
    w: Window,
    resources: Resources,
) Result(HostPlacement) {
    const window = w.base;
    std.debug.assert(window >= 1);
    std.debug.assert(window <= limits.max_model_ctx);

    // Plan the slots of the kind.
    const shape = input.serve;
    var plan: kv_budget.KvPlan = undefined;
    var prefix_default = false;
    var n_logical: u32 = 1;
    const residency = kind == .serve;
    if (kind == .serve) {
        if (shape.structural_slots < shape.max_sessions) {
            return .{
                .refused = .{
                    .cpu_slots = .{
                        .slots = shape.structural_slots,
                        .sessions = shape.max_sessions,
                    },
                },
            };
        }

        plan = kv_budget.pinned(shape.structural_slots, window);
        plan.ram.prefix = shape.kv_ram_bytes orelse limits.serve_prefix_ram_default;
        prefix_default = shape.kv_ram_bytes == null;
        n_logical = shape.max_sessions;
    } else if (kind == .run and card.gdn == null) {
        plan = runHost(card, resources.model, kv_dtype, w).plan;
    } else {
        plan = kv_budget.pinned(1, window);
    }

    // Quote the RAM, or refuse.
    const kv: kv_budget.KvStructure = .{
        .n_slots = plan.n_slots,
        .window = plan.window,
        .costs = kv_cost.costVector(card, kv_dtype),
    };
    const host: kv_budget.HostRequest = .{ .ram_split = plan.ram, .residency = residency };
    const named = kind == .serve and shape.kv_ram_bytes != null;
    const quote = kv_budget.quoteRam(
        kv,
        card.n_layers,
        host,
        n_logical,
        named,
        resources.ram_free,
    ) catch |err| switch (err) {
        error.HostRamBudgetTooSmall => return .{
            .refused = .{
                .host_ram = kv_budget.ramRefusal(kv, card.n_layers, host, resources.ram_free.?),
            },
        },
        error.BlobRamBudgetTooSmall => return .{
            .refused = .{
                .blob_ram = kv_budget.blobRefusal(
                    kv,
                    card.n_layers,
                    host,
                    n_logical,
                    resources.ram_free.?,
                ),
            },
        },
    };
    std.debug.assert(plan.window == window);
    return .{ .ok = .{ .plan = plan, .prefix_default = prefix_default, .quote = quote } };
}

/// The host placement of a llama `run`. A fused build plans against the fused runner reservations
/// at no VRAM, so the placement line prices what the device placement prices.
pub fn runHost(
    card: loader.ModelCard,
    model: ?ModelNeeds,
    kv_dtype: dtype.KvDtype,
    w: Window,
) RunHost {
    std.debug.assert(card.gdn == null);
    std.debug.assert(w.base <= w.ceiling);

    const m = model orelse return .{ .plan = kv_budget.pinned(1, w.base), .report = null };
    const c = device_placement.caps(card, m, 0, w.ceiling, 1, kv_dtype);
    const plan = if (comptime build_options.use_vulkan_fused_ops)
        kv_budget.kvPlan(
            c,
            kv_budget.costModel(card.kvShape(), kv_dtype),
            .{ .window = w.base, .batch = 1 },
        ) catch unreachable
    else
        kv_budget.pinned(1, w.base);

    std.debug.assert(plan.window == w.base);
    return .{ .plan = plan, .report = device_placement.report(c, card, m, kv_dtype, plan) };
}

/// The window before any device fit: the `--max-seq` value, or else the default window.
pub fn baseWindow(flag: ?u32, ceiling: u32, default_window: u32) error{WindowOutOfRange}!u32 {
    std.debug.assert(ceiling >= 1);
    std.debug.assert(ceiling <= limits.max_model_ctx);
    std.debug.assert(default_window >= 1);
    std.debug.assert(default_window <= ceiling);

    const w = flag orelse return default_window;
    if (w == 0 or w > ceiling) return error.WindowOutOfRange;

    std.debug.assert(w >= 1);
    std.debug.assert(w <= ceiling);
    return w;
}

// -- helpers ----------------------------------------------------------------

/// Whether the `serve` boot of `input` holds every session in a slot, inside the device limits. A
/// host figure without a device reads the host placement.
fn eachSessionHasSlot(
    input: Input,
    entry: Entry,
    card: loader.ModelCard,
    resources: Resources,
) bool {
    const host_only: Resources = .{ .ram_free = resources.ram_free, .model = resources.model };
    const boot = switch (resolve(.serve, input, entry, card, host_only)) {
        .ok => |r| r,
        .refused => return false,
    };

    const device = resources.device orelse {
        const host = boot.cpu orelse return false;
        return host == .ok;
    };

    const dev = device_placement.choosePlacement(
        .serve,
        input,
        card,
        boot.early.kv_dtype,
        boot.window,
        resources,
    );
    const placement = switch (dev.placement) {
        .ok => |p| p,
        .refused => return false,
    };

    const f = switch (placement) {
        .fused => |fused_placement| fused_placement,
        .hybrid, .cpu => return false,
    };

    if (f.quote == .refused) return false;
    if (f.plan.n_slots < input.serve.max_sessions) return false;

    // A plan wider than the device limits leaves the refusal to the runner, so check here.
    const row = card.kvShape().kv_row_elems;
    const slots_max = kv_index.maxSlotsForLimits(
        f.plan.window,
        row,
        boot.early.kv_dtype,
        device.buffer_limits,
    );
    return f.plan.n_slots <= slots_max;
}

// -- tests ------------------------------------------------------------------

const prop = @import("../core/prop.zig");

const mib: u64 = 1024 * 1024;

fn smollm2Card() loader.ModelCard {
    return .{
        .arch = .llama,
        .architecture = "llama",
        .n_layers = 30,
        .n_heads = 9,
        .n_kv_heads = 3,
        .head_dim = 64,
        .embedding_length = 576,
        .feed_forward_length = 1536,
        .context_length = 8192,
        .vocab_size = 49152,
        .rope_freq_base = 100000.0,
        .rope_dim_count = 64,
        .rms_norm_eps = 1e-5,
        .bos_token_id = 1,
        .eos_token_id = 2,
        .chat_template = "",
    };
}

fn qwenCard() loader.ModelCard {
    return .{
        .arch = .qwen35,
        .architecture = "qwen3.5",
        .n_layers = 24,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
        .embedding_length = 2048,
        .feed_forward_length = 8192,
        .context_length = 262144,
        .vocab_size = 248320,
        .rope_freq_base = 5000000.0,
        .rope_dim_count = 64,
        .rms_norm_eps = 1e-6,
        .bos_token_id = 0,
        .eos_token_id = 0,
        .chat_template = "",
        .gdn = .{
            .conv_kernel = 4,
            .state_size = 128,
            .group_count = 16,
            .inner_size = 2048,
            .time_step_rank = 16,
            .full_attention_interval = 4,
        },
    };
}

const smollm2_entry: Entry = .{
    .pub_name = "SmolLM2-135M-Instruct-Q8_0",
    .chat_template_kind = .chatml_smollm2,
    .kv_dtype_default = .f32,
    .context_limit = null,
};
const qwen_entry: Entry = .{
    .pub_name = "Qwen3.5-2B-Q8_0",
    .chat_template_kind = .{ .chatml_qwen = .off },
    .kv_dtype_default = .f16,
    .context_limit = null,
};

/// One row of the CPU boot-log oracle: the command and its flags, and the
/// figures its log lines state.
const Row = struct {
    name: []const u8,
    kind: Kind,
    qwen: bool,
    input: Input,
    /// The free RAM of the host, in MiB.
    ram_mib: u64 = 64 * 1024,
    want: CpuWant,
};

/// The models of the tables. The name of a row joins the kind, the model, and the label.
const Model = enum {
    smollm,
    qwen,

    fn label(m: Model) []const u8 {
        return switch (m) {
            .smollm => "SmolLM2",
            .qwen => "Qwen3.5",
        };
    }
};

const smollm: Model = .smollm;
const qwen: Model = .qwen;

/// The flags that rows share.
const seq512: EngineFlags = .{ .max_seq = 512 };
const seq1024: EngineFlags = .{ .max_seq = 1024 };
const seq_huge: EngineFlags = .{ .max_seq = 99999999 };
const f16kv: EngineFlags = .{ .kv_dtype = .f16 };
const think_on: EngineFlags = .{ .thinking = .on };
const budget: EngineFlags = .{ .thinking_budget = 100 };
const gl0: EngineFlags = .{ .gpu_layers = .{ .count = 0 } };
const gl6: EngineFlags = .{ .gpu_layers = .{ .count = 6 } };
const gl31: EngineFlags = .{ .gpu_layers = .{ .count = 31 } };
const auto: EngineFlags = .{ .gpu_layers = .auto };

/// Returns `base` with each field of `over` set. A field that `base` lacks is a compile error.
fn overlay(base: anytype, over: anytype) @TypeOf(base) {
    var out = base;
    inline for (std.meta.fields(@TypeOf(over))) |f| {
        @field(out, f.name) = @field(over, f.name);
    }

    return out;
}

/// An input from a literal such as `.{ .engine = gl6 }`, which a tuple keeps as an anonymous type.
fn asInput(comptime v: anytype) Input {
    if (@TypeOf(v) == Input) return v;
    return overlay(@as(Input, .{}), v);
}

/// A serve input from a base shape. The `engine` key of `over` sets the flags, and every other
/// key sets a field of the shape.
fn serveIn(base: ServeShape, over: anytype) Input {
    var input: Input = .{ .serve = base };
    inline for (std.meta.fields(@TypeOf(over))) |f| {
        if (comptime std.mem.eql(u8, f.name, "engine")) {
            input.engine = over.engine;
        } else {
            @field(input.serve, f.name) = @field(over, f.name);
        }
    }

    return input;
}

/// The figures of a CPU row that resolves.
const CpuWant = union(enum) {
    ok: struct { window: u32, slots: u32, host_kv_mib: u64, prefix_mib: u64 },
    refused: std.meta.Tag(Refusal),
};

fn ok(window: u32, slots: u32, host_kv_mib: u64, prefix_mib: u64) CpuWant {
    return .{ .ok = .{
        .window = window,
        .slots = slots,
        .host_kv_mib = host_kv_mib,
        .prefix_mib = prefix_mib,
    } };
}

fn rejects(tag: std.meta.Tag(Refusal)) CpuWant {
    return .{ .refused = tag };
}

/// The serve input of a default boot: four sessions, every one resident.
fn serve4(over: anytype) Input {
    return serveIn(.{ .max_sessions = 4, .structural_slots = 4 }, over);
}

const serve2: Input = serve4(.{ .max_sessions = 2, .structural_slots = 2 });

/// Rows of one kind and model. A case is a label, an input, a want, and optional row fields.
fn cpuRows(comptime kind: Kind, comptime model: Model, comptime cases: anytype) [cases.len]Row {
    var out: [cases.len]Row = undefined;
    inline for (cases, 0..) |c, i| {
        out[i] = .{
            .name = @tagName(kind) ++ "-" ++ model.label() ++ "-" ++ c[0],
            .kind = kind,
            .qwen = model == .qwen,
            .input = asInput(c[1]),
            .want = c[2],
        };
        if (c.len == 4) {
            out[i] = overlay(out[i], c[3]);
        }
    }

    return out;
}

const rows =
    cpuRows(.run, smollm, .{
        .{ "default", .{}, ok(8192, 1, 360, 0) },
        .{ "maxseq512", .{ .engine = seq512 }, ok(512, 1, 22, 0) },
        .{ "maxseq-huge", .{ .engine = seq_huge }, rejects(.window_out_of_range) },
        .{ "f16", .{ .engine = f16kv }, ok(8192, 1, 180, 0) },
        .{ "chat-thinking-on", .{ .engine = think_on }, rejects(.thinking_unsupported) },
    }) ++
    cpuRows(.run, qwen, .{
        .{ "default", .{}, ok(174592, 1, 2065, 0) },
        .{ "maxseq-huge", .{ .engine = seq_huge }, rejects(.window_out_of_range) },
        .{ "maxseq512", .{ .engine = seq512 }, ok(512, 1, 25, 0) },
        .{ "f16", .{ .engine = f16kv }, ok(174592, 1, 2065, 0) },
        .{ "chat-thinking-on", .{ .engine = think_on }, ok(174592, 1, 2065, 0) },
        .{ "ram-small", .{}, rejects(.host_ram), .{ .ram_mib = 64 } },
    }) ++
    cpuRows(.serve, smollm, .{
        .{ "default", serve4(.{}), ok(2816, 4, 495, 1024) },
        .{ "kvram1", serve4(.{ .kv_ram_bytes = 1 * mib }), ok(2816, 4, 495, 1) },
        .{ "maxseq1024", serve4(.{ .engine = seq1024 }), ok(1024, 4, 180, 1024) },
        .{ "maxseq-huge", serve4(.{ .engine = seq_huge }), rejects(.window_out_of_range) },
        .{ "thinkbudget100", serve4(.{ .engine = budget }), rejects(.thinking_budget_unsupported) },
        .{ "f16", serve4(.{ .engine = f16kv }), ok(5632, 4, 495, 1024) },
        .{ "sessions2", serve2, ok(2816, 2, 247, 1024) },
        .{ "kvram64", serve4(.{ .kv_ram_bytes = 64 * mib }), ok(2816, 4, 495, 64) },
        .{ "ram-small", serve4(.{}), rejects(.host_ram), .{ .ram_mib = 600 } },
        .{ "cpu-slots-below-sessions", serve4(.{ .structural_slots = 2 }), rejects(.cpu_slots) },
    }) ++
    cpuRows(.serve, qwen, .{
        .{ "default", serve4(.{}), ok(10752, 4, 581, 1024) },
        .{ "sessions2", serve2, ok(10752, 2, 290, 1024) },
        .{ "maxseq1024", serve4(.{ .engine = seq1024 }), ok(1024, 4, 125, 1024) },
        .{ "thinkbudget100", serve4(.{ .engine = budget }), ok(10752, 4, 581, 1024) },
        .{ "kvram1", serve4(.{ .kv_ram_bytes = 1 * mib }), ok(10752, 4, 581, 1) },
        .{ "kvram64", serve4(.{ .kv_ram_bytes = 64 * mib }), ok(10752, 4, 581, 64) },
        .{ "f16", serve4(.{ .engine = f16kv }), ok(10752, 4, 581, 1024) },
        .{ "maxseq-huge", serve4(.{ .engine = seq_huge }), rejects(.window_out_of_range) },
    }) ++
    cpuRows(.bench, smollm, .{
        .{ "default", .{}, ok(8192, 1, 360, 0) },
        .{ "maxseq512", .{ .engine = seq512 }, ok(512, 1, 22, 0) },
        .{ "f16", .{ .engine = f16kv }, ok(8192, 1, 180, 0) },
    }) ++
    cpuRows(.bench, qwen, .{
        .{ "default", .{}, ok(174592, 1, 2065, 0) },
        .{ "maxseq512", .{ .engine = seq512 }, ok(512, 1, 25, 0) },
        .{ "f16", .{ .engine = f16kv }, ok(174592, 1, 2065, 0) },
    });

test "resolve: the CPU oracle rows resolve to their logged window, slots, and RAM" {
    if (comptime build_options.use_vulkan_ops or build_options.use_vulkan_fused_ops) {
        return error.SkipZigTest;
    }

    for (rows) |row| {
        const resources: Resources = .{ .ram_free = row.ram_mib * mib };
        const card = if (row.qwen) qwenCard() else smollm2Card();
        const entry = if (row.qwen) qwen_entry else smollm2_entry;
        const got = resolve(row.kind, row.input, entry, card, resources);
        errdefer std.debug.print("row {s}\n", .{row.name});

        switch (row.want) {
            .refused => |tag| {
                // A window or flag refusal stops the boot. A RAM or slot refusal goes with the host
                // placement.
                const refusal = switch (got) {
                    .refused => |r| r,
                    .ok => |r| switch (r.cpu.?) {
                        .refused => |x| x,
                        .ok => return error.TestUnexpectedResult,
                    },
                };

                try std.testing.expectEqual(tag, std.meta.activeTag(refusal));
            },
            .ok => |w| {
                const r = switch (got) {
                    .ok => |v| v,
                    .refused => return error.TestUnexpectedResult,
                };

                const placement = r.cpu.?.ok;
                try std.testing.expectEqual(w.window, placement.plan.window);
                try std.testing.expectEqual(w.slots, placement.plan.n_slots);
                try std.testing.expectEqual(w.host_kv_mib, placement.quote.ram.host_kv / mib);
                try std.testing.expectEqual(w.prefix_mib, placement.quote.ram.prefix / mib);
            },
        }
    }
}

test "resolve: a RAM figure below the host KV refuses with the host tag" {
    if (comptime build_options.use_vulkan_ops or build_options.use_vulkan_fused_ops) {
        return error.SkipZigTest;
    }

    const got = resolve(.run, .{}, qwen_entry, qwenCard(), .{ .ram_free = 64 * mib });
    try std.testing.expectEqual(
        std.meta.Tag(Refusal).host_ram,
        std.meta.activeTag(got.ok.cpu.?.refused),
    );
    // An absent figure checks nothing, so the same boot resolves.
    try std.testing.expect(
        resolve(.run, .{}, qwen_entry, qwenCard(), .{ .ram_free = null }) == .ok,
    );
}

const Case = struct {
    kind: Kind,
    qwen: bool,
    input: Input,
    ram_free: ?u64,
    context_limit: ?u32,
};

fn genCase(_: void, r: std.Random, size: u32) Case {
    const kind = r.enumValue(Kind);
    const is_qwen = r.boolean();
    const ceiling_hint: u32 = if (is_qwen) 262144 else 8192;
    var input: Input = .{};
    if (r.boolean()) {
        input.engine.max_seq = r.uintAtMost(u32, ceiling_hint + 2 + size);
    }

    if (r.boolean()) {
        input.engine.kv_dtype = r.enumValue(dtype.KvDtype);
    }

    if (kind == .serve) {
        const sessions = r.intRangeAtMost(u16, 1, 8);
        input.serve = .{
            .max_sessions = sessions,
            .structural_slots = r.intRangeAtMost(u16, 1, sessions),
            .kv_ram_bytes = if (r.boolean()) (@as(
                u64,
                r.uintAtMost(u32, 4096),
            ) + 1) * mib else null,
        };
    }

    return .{
        .kind = kind,
        .qwen = is_qwen,
        .input = input,
        .ram_free = if (r.boolean()) @as(u64, r.uintAtMost(u32, 1 << 16)) * mib else null,
        .context_limit = if (r.boolean()) r.intRangeAtMost(u32, 1, ceiling_hint) else null,
    };
}

fn checkCase(_: void, c: Case) anyerror!void {
    var entry = if (c.qwen) qwen_entry else smollm2_entry;
    entry.context_limit = c.context_limit;
    const card = if (c.qwen) qwenCard() else smollm2Card();
    const got = resolve(c.kind, c.input, entry, card, .{ .ram_free = c.ram_free });
    const r = switch (got) {
        .ok => |v| v,
        .refused => |refusal| {
            // The generated flags break no other rule, so the only legal refusal is a
            // named window of zero or past the ceiling.
            const ceiling = limits.contextCeiling(card.context_length, c.context_limit);
            try std.testing.expectEqual(
                std.meta.Tag(Refusal).window_out_of_range,
                std.meta.activeTag(refusal),
            );
            try std.testing.expectEqual(ceiling, refusal.window_out_of_range);
            const w = c.input.engine.max_seq orelse return error.TestUnexpectedResult;
            try std.testing.expect(w == 0 or w > ceiling);
            return;
        },
    };

    // A window never passes the ceiling, and a named window is the window.
    try std.testing.expect(r.window.base >= 1);
    try std.testing.expect(r.window.base <= r.window.ceiling);
    try std.testing.expect(r.window.ceiling <= card.context_length);
    if (c.context_limit) |l| {
        try std.testing.expect(r.window.ceiling <= l);
    }

    if (c.input.engine.max_seq) |w| {
        try std.testing.expectEqual(w, r.window.base);
    }

    const placement = switch (r.cpu orelse return) {
        .ok => |v| v,
        .refused => |refusal| {
            switch (refusal) {
                // Only a known RAM figure refuses on RAM.
                .host_ram, .blob_ram => try std.testing.expect(c.ram_free != null),
                .cpu_slots => |s| try std.testing.expect(s.slots < s.sessions),
                else => return error.TestUnexpectedResult,
            }

            return;
        },
    };

    // A plan never spends more host RAM than the resources it was given.
    if (c.ram_free) |free| {
        try std.testing.expect(placement.quote.ram.spend() <= free);
    }

    try std.testing.expect(placement.plan.n_slots >= 1);
    try std.testing.expectEqual(r.window.base, placement.plan.window);
}

test "resolve: no window past the ceiling, no plan past the RAM it was given" {
    if (comptime build_options.use_vulkan_ops or build_options.use_vulkan_fused_ops) {
        return error.SkipZigTest;
    }

    try prop.forAll(.{ .site = 0x7265736f6c766521 }, {}, Case, genCase, checkCase);
}

// -- tests: the device rows -------------------------------------------------

/// The weight census of SmolLM2-135M-Instruct-Q8_0: 30 equal layers and a tied
/// head. The ledger lines of the device rows state 50 MiB for six layers.
fn smollm2Model() ModelNeeds {
    var m: ModelNeeds = .{ .n_layers = 30, .layer_weights = @splat(0), .head_weights = 30_083_328 };
    for (m.layer_weights[0..30]) |*w| {
        w.* = 3_764_736;
    }

    m.weights = 143_025_408;
    m.scratch = .initFill(64 * mib);
    m.logits_row = 49_152 * 4;
    return m;
}

/// The weight census of Qwen3.5-2B-Q8_0: every fourth layer is full attention
/// and lighter, and the head is tied. The ledger lines state 860 MiB for six
/// layers.
fn qwenModel() ModelNeeds {
    var m: ModelNeeds = .{
        .n_layers = 24,
        .layer_weights = @splat(0),
        .head_weights = 540_352_512,
    };
    for (m.layer_weights[0..24], 0..) |*w, i| {
        w.* = if (i % 4 == 3) 55_724_032 else 62_575_232;
    }

    m.weights = 2_001_050_880;
    m.scratch = .initFill(256 * mib);
    m.logits_row = 248_320 * 4;
    return m;
}

/// The device of the capture: the storage range it reports, which no row of
/// the default storage range passes. The rows log no VRAM cut, so any figure
/// that funds every row stands for the live budget.
const capture_device: DeviceCapacity = .{
    .vram = 12 << 30,
    .buffer_limits = .{
        .storage_range = 128 << 20,
        .allocation_size = 2 << 30,
        .offset_alignment = 16,
    },
};

/// One row of the device boot-log oracle: the command, its flags, the storage
/// range, and the figures its log lines state. The RAM figures are MiB.
const DeviceRow = struct {
    name: []const u8,
    kind: Kind,
    qwen: bool,
    input: Input,
    storage_range: u64 = 128 << 20,
    /// The live VRAM budget and the free host RAM, in MiB. A small figure
    /// makes the VRAM budgets refuse.
    vram_mib: u64 = 12 * 1024,
    ram_mib: u64 = 64 * 1024,
    /// The window a clamped row cuts to.
    clamp_window: u32 = 131_072,
    /// The split of the placement line, for a `--gpu-layers` row.
    host_layers: ?u32 = null,
    /// True when the window line names the storage-buffer range.
    clamped: bool = false,
    want: DeviceWant,
};

/// The figures of a placement row, in MiB.
const PlacementWant = struct {
    tag: device_placement.PlacementKind,
    slots: u32,
    window: u32,
    vram_kv_mib: u64 = 0,
    host_kv_mib: u64 = 0,
    blobs_mib: u64 = 0,
    prefix_mib: u64 = 0,
    stages_mib: u64 = 0,
};

const DeviceWant = union(enum) { placement: PlacementWant, refused: std.meta.Tag(Refusal) };

fn refused(tag: std.meta.Tag(Refusal)) DeviceWant {
    return .{ .refused = tag };
}

/// A fused placement of a run or bench boot: the KV lives on the device.
fn fused(slots: u32, window: u32, vram_kv_mib: u64) DeviceWant {
    return .{ .placement = .{
        .tag = .fused,
        .slots = slots,
        .window = window,
        .vram_kv_mib = vram_kv_mib,
    } };
}

/// A hybrid placement of a run or bench boot: the KV lives on the host.
fn hybrid(slots: u32, window: u32, host_kv_mib: u64) DeviceWant {
    return .{ .placement = .{
        .tag = .hybrid,
        .slots = slots,
        .window = window,
        .host_kv_mib = host_kv_mib,
    } };
}

/// The same placements of a serve boot, which add the prefix and the swap stages.
fn fusedS(slots: u32, window: u32, vram_kv_mib: u64, prefix_mib: u64, stages_mib: u64) DeviceWant {
    var w = fused(slots, window, vram_kv_mib);
    w.placement.prefix_mib = prefix_mib;
    w.placement.stages_mib = stages_mib;
    return w;
}

fn hybridS(slots: u32, window: u32, host_kv_mib: u64, prefix_mib: u64, stages_mib: u64) DeviceWant {
    var w = hybrid(slots, window, host_kv_mib);
    w.placement.prefix_mib = prefix_mib;
    w.placement.stages_mib = stages_mib;
    return w;
}

/// A placement that also holds weight blobs in RAM.
fn blobs(blobs_mib: u64, want: DeviceWant) DeviceWant {
    var w = want;
    w.placement.blobs_mib = blobs_mib;
    return w;
}

/// The serve input of a device boot: four sessions, a decode width of four.
fn gpu4(over: anytype) Input {
    return serveIn(.{ .max_sessions = 4, .structural_slots = 4, .batch = 4 }, over);
}

const gpu2: Input = gpu4(.{ .max_sessions = 2, .structural_slots = 2, .batch = 2 });
const pinned16: Input = gpu4(.{
    .engine = gl6,
    .max_sessions = 16,
    .structural_slots = 16,
    .slots_req = 16,
});
const vram_slots2: Input = gpu4(.{ .batch = 2, .slots_req = 2 });
const kv_vram512: Input = gpu4(.{ .batch = 2, .kv_vram_bytes = 512 * mib });
const kv_vram1: Input = gpu4(.{ .batch = 2, .kv_vram_bytes = 1 * mib });
const auto_in: Input = .{ .engine = auto };
const gl0_in: Input = .{ .engine = gl0 };
const glall30_in: Input = .{ .engine = .{ .gpu_layers = .{ .count = 30 } } };
const glall24_in: Input = .{ .engine = .{ .gpu_layers = .{ .count = 24 } } };

/// Row fields that name the storage range or the VRAM budget of a capture.
const range_1gib = .{ .storage_range = 1 << 30 };
const range_64mib = .{ .storage_range = 64 << 20, .clamp_window = 65_536, .clamped = true };
const tight_vram = .{ .vram_mib = 512, .clamped = true };

/// Rows of one kind and model. A case is a label, an input, a want, and optional row fields.
fn deviceRows(
    comptime kind: Kind,
    comptime model: Model,
    comptime cases: anytype,
) [cases.len]DeviceRow {
    var out: [cases.len]DeviceRow = undefined;
    inline for (cases, 0..) |c, i| {
        out[i] = .{
            .name = @tagName(kind) ++ "-" ++ model.label() ++ "-" ++ c[0],
            .kind = kind,
            .qwen = model == .qwen,
            .input = asInput(c[1]),
            .want = c[2],
        };
        if (c.len == 4) {
            out[i] = overlay(out[i], c[3]);
        }
    }

    return out;
}

const device_rows =
    deviceRows(.run, smollm, .{
        .{ "default", .{}, fused(1, 8192, 360) },
        .{ "maxseq512", .{ .engine = seq512 }, fused(1, 512, 22) },
        .{ "f16", .{ .engine = f16kv }, fused(1, 8192, 180) },
        .{ "gl6", .{ .engine = gl6 }, hybrid(1, 8192, 288), .{ .host_layers = 24 } },
        .{ "glauto", auto_in, fused(1, 8192, 360), .{ .host_layers = 0 } },
        .{ "range1073741824", .{}, fused(1, 8192, 360), range_1gib },
        .{ "gl31", .{ .engine = gl31 }, refused(.gpu_layers_past_model) },
    }) ++
    deviceRows(.serve, smollm, .{
        .{ "default", gpu4(.{}), fusedS(4, 2816, 495, 1024, 247) },
        .{ "sessions2", gpu2, fusedS(2, 2816, 247, 1024, 247) },
        .{ "maxseq1024", gpu4(.{ .engine = seq1024 }), fusedS(4, 1024, 180, 1024, 90) },
        .{ "kvram64", gpu4(.{ .kv_ram_bytes = 64 * mib }), fusedS(4, 2816, 495, 64, 247) },
        .{ "kvram1", gpu4(.{ .kv_ram_bytes = 1 * mib }), fusedS(4, 2816, 495, 1, 247) },
        .{ "f16", gpu4(.{ .engine = f16kv }), fusedS(4, 5632, 495, 1024, 247) },
        .{
            "gl6",
            gpu4(.{ .engine = gl6 }),
            hybridS(4, 2816, 396, 1024, 49),
            .{ .host_layers = 24 },
        },
        .{
            "glauto",
            gpu4(.{ .engine = auto }),
            fusedS(4, 2816, 495, 1024, 247),
            .{ .host_layers = 0 },
        },
        .{ "range1073741824", gpu4(.{}), fusedS(4, 2816, 495, 1024, 247), range_1gib },
        .{ "gl31", gpu4(.{ .engine = gl31 }), refused(.gpu_layers_past_model) },
        .{ "gl6-slots-pinned", pinned16, refused(.hybrid_slots) },
        .{ "vramslots2", vram_slots2, blobs(1080, fusedS(2, 8192, 720, 1024, 512)) },
        .{ "vramslots2-ram-small", vram_slots2, refused(.blob_ram), .{ .ram_mib = 1300 } },
        .{ "kvvram512", kv_vram512, fusedS(4, 2816, 495, 1024, 247) },
        .{ "kvvram-tiny", kv_vram1, refused(.kv_vram_small) },
    }) ++
    deviceRows(.bench, smollm, .{
        .{ "default", .{}, fused(1, 8192, 360) },
        .{ "maxseq512", .{ .engine = seq512 }, fused(1, 512, 22) },
        .{ "f16", .{ .engine = f16kv }, fused(1, 8192, 180) },
        .{ "gl0", gl0_in, refused(.whole_model_split), .{ .host_layers = 30 } },
        .{ "gl6", .{ .engine = gl6 }, hybrid(1, 8192, 288), .{ .host_layers = 24 } },
        .{ "glall", glall30_in, refused(.whole_model_split), .{ .host_layers = 0 } },
        .{ "glauto", auto_in, refused(.whole_model_split), .{ .host_layers = 0 } },
    }) ++
    deviceRows(.run, qwen, .{
        .{ "default", .{}, fused(1, 131072, 1555), .{ .clamped = true } },
        .{ "maxseq512", .{ .engine = seq512 }, fused(1, 512, 25) },
        .{
            "gl6",
            .{ .engine = gl6 },
            hybrid(1, 131072, 1038),
            .{ .host_layers = 18, .clamped = true },
        },
        .{ "glauto", auto_in, fused(1, 131072, 1555), .{ .host_layers = 0, .clamped = true } },
        .{ "range1073741824", .{}, fused(1, 174592, 2065), range_1gib },
        .{ "gl6-vram-small", .{ .engine = gl6 }, refused(.layer_split_vram), tight_vram },
        .{ "range64MiB", .{}, fused(1, 65_536, 787), range_64mib },
    }) ++
    deviceRows(.serve, qwen, .{
        .{ "default", gpu4(.{}), fusedS(4, 10752, 581, 1024, 290) },
        .{ "sessions2", gpu2, fusedS(2, 10752, 290, 1024, 290) },
        .{ "maxseq1024", gpu4(.{ .engine = seq1024 }), fusedS(4, 1024, 125, 1024, 62) },
        .{ "kvram64", gpu4(.{ .kv_ram_bytes = 64 * mib }), fusedS(4, 10752, 581, 64, 290) },
        .{ "kvram1", gpu4(.{ .kv_ram_bytes = 1 * mib }), fusedS(4, 10752, 581, 1, 290) },
        .{
            "gl6",
            gpu4(.{ .engine = gl6 }),
            hybridS(4, 10752, 395, 1024, 92),
            .{ .host_layers = 18 },
        },
        .{
            "glauto",
            gpu4(.{ .engine = auto }),
            fusedS(4, 10752, 581, 1024, 290),
            .{ .host_layers = 0 },
        },
        .{ "range1073741824", gpu4(.{}), fusedS(4, 10752, 581, 1024, 290), range_1gib },
    }) ++
    deviceRows(.bench, qwen, .{
        .{ "default", .{}, fused(1, 131072, 1555), .{ .clamped = true } },
        .{ "maxseq512", .{ .engine = seq512 }, fused(1, 512, 25) },
        .{ "range1073741824", .{}, fused(1, 174592, 2065), range_1gib },
        .{ "gl0", gl0_in, refused(.whole_model_split), .{ .host_layers = 24, .clamped = true } },
        .{
            "gl6",
            .{ .engine = gl6 },
            hybrid(1, 131072, 1038),
            .{ .host_layers = 18, .clamped = true },
        },
        .{
            "glall",
            glall24_in,
            refused(.whole_model_split),
            .{ .host_layers = 0, .clamped = true },
        },
        .{ "glauto", auto_in, refused(.whole_model_split), .{ .host_layers = 0, .clamped = true } },
    });

/// The refusal of a device placement: the placement itself, or the RAM quote of the placement it
/// took.
fn deviceRefusal(placement: Result(device_placement.Placement)) ?Refusal {
    const taken = switch (placement) {
        .refused => |r| return r,
        .ok => |v| v,
    };

    return switch (taken) {
        .fused => |f| if (f.quote == .refused) f.quote.refused else null,
        .hybrid => |h| if (h.quote == .refused) h.quote.refused else null,
        .cpu => |c| if (c == .refused) c.refused else null,
    };
}

/// The window, slots, VRAM KV, and RAM quote of the placement a device row takes.
fn deviceShape(placement: device_placement.Placement) struct {
    slots: u32,
    window: u32,
    vram_kv: u64,
    ram: kv_budget.LayerSplit.Ram,
} {
    return switch (placement) {
        .fused => |f| .{
            .slots = f.plan.n_slots,
            .window = f.plan.window,
            .vram_kv = f.plan.vram.kv,
            .ram = f.quote.ok.ram,
        },
        .hybrid => |h| .{
            .slots = h.plan.?.n_slots,
            .window = h.plan.?.window,
            .vram_kv = 0,
            .ram = h.quote.ok.ram,
        },
        .cpu => |c| .{
            .slots = c.ok.plan.n_slots,
            .window = c.ok.plan.window,
            .vram_kv = 0,
            .ram = c.ok.quote.ram,
        },
    };
}

test "mostSessions: VRAM and the storage range bound the count" {
    const case = struct {
        fn most(range: u64, vram_mib: u64, cap: u16) ?u16 {
            var device = capture_device;
            device.buffer_limits.storage_range = range;
            device.vram = vram_mib * mib;
            const resources: Resources = .{
                .ram_free = 32 << 30,
                .model = qwenModel(),
                .device = device,
            };
            return mostSessions(.{}, qwen_entry, qwenCard(), resources, cap);
        }
    };

    // The 128 MiB range holds 12 slots at the serve window. With a 1 GiB range the VRAM decides:
    // 7/8 of 16 GiB less the weights, the scratch, and the logits of this fixture leaves 12,163
    // MiB, and a slot costs 145.3 MiB.
    try std.testing.expectEqual(@as(?u16, 12), case.most(128 << 20, 16 << 10, 128));
    try std.testing.expectEqual(@as(?u16, 83), case.most(1 << 30, 16 << 10, 128));
    // The cap binds first, and a budget below the weights fits no session.
    try std.testing.expectEqual(@as(?u16, 4), case.most(1 << 30, 16 << 10, 4));
    try std.testing.expectEqual(@as(?u16, null), case.most(1 << 30, 512, 128));
}

test "mostSessions: every count it returns plans a slot per session, and more VRAM never cuts it" {
    const models = [_]struct { card: loader.ModelCard, need: ModelNeeds, entry: Entry }{
        .{ .card = qwenCard(), .need = qwenModel(), .entry = qwen_entry },
        .{ .card = smollm2Card(), .need = smollm2Model(), .entry = smollm2_entry },
    };

    for (models) |m| {
        var last: u16 = 0;
        var vram_mib: u64 = 1 << 9;
        while (vram_mib <= 32 << 10) : (vram_mib *= 2) {
            var device = capture_device;
            device.vram = vram_mib * mib;
            const resources: Resources = .{
                .ram_free = 64 << 30,
                .model = m.need,
                .device = device,
            };
            const most = mostSessions(.{}, m.entry, m.card, resources, 128) orelse continue;

            try std.testing.expect(most >= last);
            last = most;

            const shaped = spendVram(.{}, most);
            const boot = resolve(.serve, shaped, m.entry, m.card, .{ .ram_free = 64 << 30 }).ok;
            const placed = device_placement.choosePlacement(
                .serve,
                shaped,
                m.card,
                boot.early.kv_dtype,
                boot.window,
                resources,
            ).placement.ok.fused;
            try std.testing.expect(placed.plan.n_slots >= most);
            try std.testing.expect(kv_budget.planFits(placed.caps, placed.plan));
        }

        try std.testing.expect(last >= 1);
    }
}

test "mostSessions: a host with no device stops at the decode width it holds" {
    const resources: Resources = .{ .ram_free = 64 << 30 };
    const cap: u16 = @intCast(limits.max_batch);
    const most = mostSessions(.{}, qwen_entry, qwenCard(), resources, cap);
    if (comptime build_options.use_vulkan_fused_ops) return error.SkipZigTest;

    try std.testing.expectEqual(@as(?u16, cap), most);
    // A RAM figure below one session fits no count.
    const small: Resources = .{ .ram_free = 1 << 20 };
    const none = mostSessions(.{}, qwen_entry, qwenCard(), small, cap);
    try std.testing.expectEqual(@as(?u16, null), none);
}

test "resolve: the device oracle rows resolve to their logged split, window, slots, and RAM" {
    if (comptime build_options.use_vulkan_ops) return error.SkipZigTest;
    for (device_rows) |row| {
        const card = if (row.qwen) qwenCard() else smollm2Card();
        const entry = if (row.qwen) qwen_entry else smollm2_entry;
        var device = capture_device;
        device.buffer_limits.storage_range = row.storage_range;
        device.vram = row.vram_mib * mib;
        const resources: Resources = .{
            .ram_free = row.ram_mib * mib,
            .model = if (row.qwen) qwenModel() else smollm2Model(),
            .device = device,
        };
        // A host build resolves no device placement, so the row takes the window of
        // the boot and calls the device placement itself.
        const pre = early(row.kind, row.input, entry).ok;
        const base = switch (resolve(
            row.kind,
            row.input,
            entry,
            card,
            .{ .ram_free = resources.ram_free },
        )) {
            .ok => |r| r.window,
            .refused => return error.TestUnexpectedResult,
        };

        const dev = device_placement.choosePlacement(
            row.kind,
            row.input,
            card,
            pre.kv_dtype,
            base,
            resources,
        );
        errdefer std.debug.print("row {s}\n", .{row.name});

        if (row.host_layers) |want| {
            try std.testing.expectEqual(want, dev.layer_choice.?.layer_split.?.host_layers);
        }

        if (row.host_layers == null and row.want != .refused) {
            try std.testing.expect(dev.layer_choice == null);
        }

        switch (row.want) {
            .refused => |tag| try std.testing.expectEqual(
                tag,
                std.meta.activeTag(
                    deviceRefusal(dev.placement) orelse return error.TestUnexpectedResult,
                ),
            ),
            .placement => |w| {
                const placement = dev.placement.ok;
                try std.testing.expectEqual(w.tag, std.meta.activeTag(placement));
                const got = deviceShape(placement);
                try std.testing.expectEqual(w.slots, got.slots);
                try std.testing.expectEqual(w.window, got.window);
                try std.testing.expectEqual(w.vram_kv_mib, got.vram_kv / mib);
                try std.testing.expectEqual(w.host_kv_mib, got.ram.host_kv / mib);
                try std.testing.expectEqual(w.blobs_mib, got.ram.blobs / mib);
                try std.testing.expectEqual(w.prefix_mib, got.ram.prefix / mib);
                try std.testing.expectEqual(w.stages_mib, got.ram.swap_stages / mib);
                // The run of a llama model states the all-device placement.
                if (placement == .fused and row.kind == .run and !row.qwen) {
                    const report = placement.fused.report.?;
                    try std.testing.expectEqual(@as(u32, 0), report.host_layers);
                }
            },
        }

        const clamp = if (dev.layer_choice) |p|
            p.clamp
        else if (dev.placement == .ok and dev.placement.ok == .fused)
            dev.placement.ok.fused.clamp
        else
            null;

        try std.testing.expectEqual(row.clamped, clamp != null);
        if (clamp) |c| {
            // The logged line: 178782208 bytes at 174592 positions, past the
            // storage-buffer range of 134217728 bytes.
            try std.testing.expectEqual(@as(u64, 178_782_208), c.need);
            try std.testing.expectEqual(row.storage_range, c.limit.value);
            try std.testing.expectEqual(row.clamp_window, c.window);
        }
    }
}

test {
    _ = device_placement;
}
