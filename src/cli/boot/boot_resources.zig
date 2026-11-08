//! What a boot measures. `run`, `serve`, and the benches read the host and the device once
//! through these functions, and hand the resources to `engine/resolve`.
//!
//! Comptime. A build option picks the fused placement, and `measureFused` is generic over the
//! family.
//!
//! Logged gaps. A missing figure logs one line, so a skipped guard stays visible. The RAM line
//! prints once per boot through `DeviceBoot.ram`. The heap-size line prints once per process.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const context = @import("../context.zig");
const args = @import("../args.zig");
const family = @import("../../engine/family.zig");
const resolve_mod = @import("../../engine/resolve.zig");
const dtype = @import("../../core/dtype.zig");
const hwinfo = @import("../../core/hwinfo.zig");
const kv_budget = @import("../../core/kv_budget.zig");
const limits = @import("../../core/limits.zig");
const registry = @import("../../core/registry.zig");
const loader = @import("../../io/gguf/loader.zig");
const parallel = @import("../../runtime/parallel.zig");
const tensor_lookup = @import("../../runtime/tensor_lookup.zig");
// The device context resolves to `void` on a CPU build, which never analyzes it.
const vulkan_ctx_mod = if (fused_build) @import("../../runtime/vulkan_ctx.zig") else void;

const log = std.log.scoped(.serve);
const fused_build = build_options.use_vulkan_fused_ops;
const device_placement = resolve_mod.device_placement;

const boot_lines = @import("boot_lines.zig");

// -- types ------------------------------------------------------------------

/// The outcome of a device attempt. `.all_host` hands the host placement back, because an auto
/// split can fund no GPU layer.
pub const DeviceAttempt = union(enum) {
    done,
    all_host: resolve_mod.Result(resolve_mod.HostPlacement),
};

/// What a device boot resolves on: the flags, the registry entry, the card, and the weight census.
pub const DeviceBoot = struct {
    kind: resolve_mod.Kind,
    input: resolve_mod.Input,
    entry: resolve_mod.Entry,
    card: loader.ModelCard,
    model: resolve_mod.ModelNeeds,
    /// The RAM figure of an earlier read.
    ram: RamFigure = .unread,

    pub const RamFigure = union(enum) { unread, read: ?u64 };

    pub fn ramFree(self: *const DeviceBoot, io: Io) ?u64 {
        return switch (self.ram) {
            .unread => hostRamFree(io),
            .read => |free| free,
        };
    }

    /// Measure the device of `ctxp` once and resolve the device placement. Only a device refusal
    /// can follow.
    pub fn resolveOn(self: *const DeviceBoot, io: Io, ctxp: anytype) !resolve_mod.DevicePlacement {
        const resources: resolve_mod.Resources = .{
            .ram_free = self.ramFree(io),
            .model = self.model,
            .device = .{ .vram = vramFigure(ctxp), .buffer_limits = ctxp.buffer_limits },
        };
        const result = resolve_mod.resolve(self.kind, self.input, self.entry, self.card, resources);
        return switch (result) {
            .ok => |r| r.device.?,
            .refused => |r| boot_lines.refuse(io, self.entry.pub_name, r),
        };
    }
};

/// A boot after the load: what `resolve` states, and the value that a device context resolves on.
pub const LoadedBoot = struct {
    values: resolve_mod.Resolved,
    device: DeviceBoot,
};

// -- state ------------------------------------------------------------------

/// The heap-size line of `vramFigure` prints once per process.
var heap_fallback_logged = false;

// -- api --------------------------------------------------------------------

pub const baseWindow = resolve_mod.baseWindow;

/// Measure the loaded model and resolve the boot on the resources that need no device.
pub fn resolveLoaded(
    ctx: context.Context,
    kind: resolve_mod.Kind,
    input: resolve_mod.Input,
    entry: resolve_mod.Entry,
    loaded: *const loader.Loaded,
) !LoadedBoot {
    const model_needs = try modelNeeds(ctx.gpa, loaded);
    const ram_free = hostRamFree(ctx.io);
    const resources: resolve_mod.Resources = .{ .ram_free = ram_free, .model = model_needs };
    const values = switch (resolve_mod.resolve(kind, input, entry, loaded.card, resources)) {
        .ok => |r| r,
        .refused => |r| return boot_lines.refuse(ctx.io, entry.pub_name, r),
    };

    return .{
        .values = values,
        .device = .{
            .kind = kind,
            .input = input,
            .entry = entry,
            .card = loaded.card,
            .model = model_needs,
            .ram = .{ .read = ram_free },
        },
    };
}

/// The registry values `resolve` reads.
pub fn entryOf(model: *const registry.Model) resolve_mod.Entry {
    return .{
        .pub_name = model.pub_name,
        .chat_template_kind = model.chat_template_kind,
        .kv_dtype_default = model.kv_dtype_default,
        .context_limit = model.context_limit,
    };
}

/// A new device context at a stable arena address. The kernels capture it, so a runner goes down
/// before the context.
pub fn openDevice(ctx: context.Context) !*vulkan_ctx_mod.Context {
    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, ctx.vk_options);
    return ctxp;
}

/// The fused placement of a one-stream boot that names no split, resolved on the context.
pub fn fusedOn(io: Io, boot: *const DeviceBoot, ctxp: anytype) !device_placement.Fused {
    const dev = try boot.resolveOn(io, ctxp);
    std.debug.assert(dev.layer_choice == null);
    return switch (dev.placement) {
        .ok => |a| a.fused,
        .refused => |r| boot_lines.refuse(io, boot.entry.pub_name, r),
    };
}

/// The weight census of one model and the reservations of its fused runner. A build without the
/// fused ops measures the census only.
pub fn modelNeeds(gpa: std.mem.Allocator, loaded: *const loader.Loaded) !resolve_mod.ModelNeeds {
    const card = loaded.card;
    var m: resolve_mod.ModelNeeds = .{
        .n_layers = card.n_layers,
        .layer_weights = @splat(0),
        .head_weights = tensor_lookup.headWeightBytes(loaded.tensors),
    };
    _ = tensor_lookup.layerWeightBytes(loaded.tensors, card.n_layers, &m.layer_weights);

    if (comptime fused_build) {
        switch (card.arch) {
            inline else => |arch| try measureFused(family.Of(arch), gpa, loaded, &m),
        }
    }

    return m;
}

/// `MemAvailable` at boot: the free RAM that every RAM quote checks against.
pub fn hostRamFree(io: Io) ?u64 {
    const free = hwinfo.readMemAvailable(io);
    if (free == null) {
        log.info("the RAM check is off: /proc/meminfo states no MemAvailable figure " ++
            "on this platform", .{});
    }

    return free;
}

/// The KV dtype that one process frames every row at. Call it before any plan, price, or device,
/// so no preflight assert meets a half-applied dtype.
pub fn kvDtypeFor(flag: ?dtype.KvDtype, model: *const registry.Model) dtype.KvDtype {
    return flag orelse model.kv_dtype_default;
}

/// The threading policy of one command: the host defaults, then the flags. A placement with no CPU
/// op gets a pool of one thread.
pub fn resolvePolicy(
    io: Io,
    flags: parallel.Flags,
    runs_cpu_ops: bool,
) error{UsageError}!parallel.Policy {
    const host_policy = parallel.Policy.forHost(io);
    const resolved = parallel.resolve(host_policy, flags, runs_cpu_ops) catch |err| switch (err) {
        error.TooManyThreads => return args.fail(
            io,
            "--threads is {d}. The limit is {d}.",
            .{ flags.threads.?, limits.max_threads },
        ),
        error.SpinTooLong => return args.fail(
            io,
            "--spin-us is {d}. The limit is {d}.",
            .{ flags.spin_us.?, parallel.max_spin_us },
        ),
    };
    parallel.logResolved(resolved);
    return resolved.policy;
}

/// The one VRAM figure that every device placement plans against: the live budget of
/// `VK_EXT_memory_budget`. The heap size overstates what a shared device has free.
pub fn vramFigure(ctxp: anytype) u64 {
    if (ctxp.deviceLocalBudget() == null and !heap_fallback_logged) {
        heap_fallback_logged = true;
        log.info("vram: the driver states no live budget (VK_EXT_memory_budget), " ++
            "so the plan reads the heap size", .{});
    }

    return ctxp.deviceLocalAvailableBytes();
}

// -- helpers ----------------------------------------------------------------

/// Measure the fused runner of `Family`. The runner caps and the resolve caps must agree.
fn measureFused(
    comptime Family: type,
    gpa: std.mem.Allocator,
    loaded: *const loader.Loaded,
    m: *resolve_mod.ModelNeeds,
) !void {
    const card = loaded.card;
    m.weights = try Family.fusedWeightBytes(gpa, loaded);
    m.logits_row = Family.Fused.logitsVramBytes(card.vocab_size, 1);

    const ceiling = @min(card.context_length, limits.max_model_ctx);
    for (std.enums.values(dtype.KvDtype)) |d| {
        m.scratch.set(d, Family.Fused.scratchVramBytes(card, d));
        const runner_caps = try Family.kvCaps(gpa, loaded, ceiling, 0, 1, d);
        const resolve_caps = resolve_mod.caps(card, m.*, 0, ceiling, 1, d);
        std.debug.assert(std.meta.eql(resolve_caps, runner_caps));
    }
}

// -- tests ------------------------------------------------------------------

test "baseWindow: a named window must lie in 1..ceiling, and no flag takes the serve default" {
    const ceiling: u32 = 8192;
    const serve_default: u32 = 2816;
    try std.testing.expectError(error.WindowOutOfRange, baseWindow(0, ceiling, serve_default));
    try std.testing.expectEqual(@as(u32, 1), try baseWindow(1, ceiling, serve_default));
    try std.testing.expectEqual(ceiling, try baseWindow(ceiling, ceiling, serve_default));
    try std.testing.expectError(
        error.WindowOutOfRange,
        baseWindow(ceiling + 1, ceiling, serve_default),
    );
    try std.testing.expectEqual(serve_default, try baseWindow(null, ceiling, serve_default));
}

test "baseWindow: a ceiling below the session budget window is the default window" {
    // Qwen3.5-2B at f16 costs 12288 bytes per position, so the budget buys 10752 positions.
    // A ceiling of 4096 binds first.
    const cost: kv_budget.KvCostModel = .{ .kv_bytes_per_token = 12_288 };
    try std.testing.expectEqual(
        @as(u32, 10_752),
        kv_budget.defaultWindow(cost, limits.max_model_ctx, limits.serve_session_kv_bytes),
    );
    const ceiling: u32 = 4096;
    const serve_default = kv_budget.defaultWindow(cost, ceiling, limits.serve_session_kv_bytes);
    try std.testing.expectEqual(ceiling, try baseWindow(null, ceiling, serve_default));
    try std.testing.expectEqual(ceiling, try baseWindow(ceiling, ceiling, serve_default));
    try std.testing.expectError(
        error.WindowOutOfRange,
        baseWindow(ceiling + 1, ceiling, serve_default),
    );
}
