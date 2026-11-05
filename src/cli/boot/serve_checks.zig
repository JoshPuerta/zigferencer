//! The checks of the serve flags that need no model file, in the order a user reads them.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const context = @import("../context.zig");
const args = @import("../args.zig");
const flags_mod = @import("../flag_groups.zig");
const resolve_mod = @import("../../engine/resolve.zig");
const limits = @import("../../core/limits.zig");
const prefill_chunk = @import("../../serve/prefill_chunk.zig");

const log = std.log.scoped(.serve);
const fused_build = build_options.use_vulkan_fused_ops;

const boot_lines = @import("boot_lines.zig");

// -- constants --------------------------------------------------------------

const kib = 1024;

// -- types ------------------------------------------------------------------

/// The options of the binary wire that `serve-openai` does not take. Every other
/// serve option is a field of `flag_groups.ServeFlags`.
pub const Wire = struct {
    allow_shutdown: bool = false,
    /// In 1..max_sessions. Absent takes `max_sessions`.
    sessions_per_connection: ?u16 = null,
    /// In 0..limits.max_resume_ttl_ms. 0 turns resumption off.
    resume_ttl_ms: u32 = 0,
};

/// The serve flags after the checks that need no model file.
pub const CheckedFlags = struct {
    shape: resolve_mod.ServeShape,
    sessions_per_connection: u16,
    prefix_disk_bytes: u64,
    preload: []const []const u8,
    /// The prefill rule of a CPU runner.
    cpu_prefill: prefill_chunk.Rule,
    /// The row cap of one step: at most `batch` decode rows and a prefill chunk.
    max_step_rows: u32,
};

const ByteBudgets = struct {
    kv_vram: ?u64,
    kv_ram: ?u64,
    prefix_disk: u64,
};

const SlotCounts = struct {
    requested: ?u16,
    structural: u16,
};

// -- api --------------------------------------------------------------------

/// Every serve refusal that needs no model file, in the order a user reads them.
pub fn checkFlags(
    ctx: context.Context,
    serve: *const flags_mod.ServeFlags,
    wire: Wire,
) !CheckedFlags {
    const io = ctx.io;

    const sessions_per_connection = try checkSessions(io, serve, wire);
    const budgets = try checkBudgets(io, serve);
    const preload = try readPreload(ctx, serve);
    const slots = try checkSlots(io, serve, budgets);
    const batch = try checkBatch(io, serve, slots.structural);

    if ((serve.prefill_chunk orelse 0) > limits.max_prefill_chunk) {
        return args.fail(io, "--prefill-chunk must be in 0..{d}", .{limits.max_prefill_chunk});
    }
    if (serve.engine.thinking_budget > limits.max_model_ctx) {
        return args.fail(io, "--thinking-budget must be in 0..{d}", .{limits.max_model_ctx});
    }

    // Never below the slot count: the scheduler's floor-1 prefill can put up to
    // n_slots rows in a step even at prefill_chunk == 0. The `batch + 1` floor keeps
    // the one-row prefill legal when a budget buys more slots than the decode width.
    const cpu_prefill = prefill_chunk.fromFlag(serve.prefill_chunk, limits.cpu_prefill_chunk);
    return .{
        .shape = .{
            .max_sessions = serve.max_sessions,
            .structural_slots = slots.structural,
            .kv_ram_bytes = budgets.kv_ram,
            .batch = batch,
            .slots_req = slots.requested,
            .kv_vram_bytes = budgets.kv_vram,
        },
        .sessions_per_connection = sessions_per_connection,
        .prefix_disk_bytes = budgets.prefix_disk,
        .preload = preload,
        .cpu_prefill = cpu_prefill,
        .max_step_rows = prefill_chunk.stepRows(cpu_prefill, slots.structural, batch),
    };
}

// -- helpers ----------------------------------------------------------------

/// `--max-sessions` is the logical conversation count. Only the fused-GPU build can
/// evict idle sessions to RAM, so only it can over-subscribe the physical slots.
fn checkSessions(io: Io, serve: *const flags_mod.ServeFlags, wire: Wire) !u16 {
    const session_cap: u16 = if (fused_build)
        @intCast(limits.max_logical_sessions)
    else
        @intCast(limits.max_batch);

    if (serve.max_sessions == 0 or serve.max_sessions > session_cap) {
        return args.fail(io, "--max-sessions must be in 1..{d}", .{session_cap});
    }
    if (serve.max_connections == 0 or serve.max_connections > limits.max_connections) {
        return args.fail(io, "--max-connections must be in 1..{d}", .{limits.max_connections});
    }
    if (wire.resume_ttl_ms > limits.max_resume_ttl_ms) {
        return args.fail(
            io,
            "--resume-ttl must be in 0..{d} ms, the lease ceiling",
            .{limits.max_resume_ttl_ms},
        );
    }

    const per_connection: u16 = wire.sessions_per_connection orelse serve.max_sessions;
    if (per_connection == 0 or per_connection > serve.max_sessions) {
        return args.fail(
            io,
            "--sessions-per-connection must be in 1..{d}, the --max-sessions value",
            .{serve.max_sessions},
        );
    }

    std.debug.assert(per_connection >= 1);
    return per_connection;
}

fn checkBudgets(io: Io, serve: *const flags_mod.ServeFlags) !ByteBudgets {
    // --kv-vram states a budget, and --vram-slots states the count that budget
    // buys. Two flags for one decision can disagree, so the pair is a mistake.
    if (serve.kv_vram != null and serve.vram_slots != null) {
        return args.fail(
            io,
            "--kv-vram and --vram-slots are mutually exclusive (the budget buys the slots)",
            .{},
        );
    }
    if (serve.engine.gpu_layers != null and !fused_build) {
        return args.fail(io, "--gpu-layers requires -Dops-backend=vulkan_fused", .{});
    }

    const kv_vram: ?u64 = blk: {
        const value = serve.kv_vram orelse break :blk null;
        if (!fused_build) {
            return args.fail(
                io,
                "--kv-vram applies only to the fused-GPU build (it spends a device KV budget)",
                .{},
            );
        }
        if (value == 0) return args.fail(io, "--kv-vram must be >= 1 (MiB)", .{});

        break :blk mibBytes(value);
    };

    // The placement planner already prices the pinned suffix and its KV against
    // the device budget, so a second statement of it can only disagree.
    if (serve.engine.gpu_layers != null and kv_vram != null) {
        return args.fail(
            io,
            "--gpu-layers and --kv-vram are mutually exclusive " ++
                "(a pinned placement prices its own KV, it does not buy slots)",
            .{},
        );
    }

    const kv_ram: ?u64 = blk: {
        const value = serve.kv_ram orelse break :blk null;
        if (value == 0) return args.fail(io, "--kv-ram must be >= 1 (MiB)", .{});
        break :blk mibBytes(value);
    };

    const prefix_disk: u64 = blk: {
        const value = serve.prefix_disk orelse break :blk 0;
        if (value == 0) return args.fail(io, "--prefix-disk must be >= 1 (MiB)", .{});
        break :blk mibBytes(value);
    };

    return .{ .kv_vram = kv_vram, .kv_ram = kv_ram, .prefix_disk = prefix_disk };
}

fn mibBytes(value: u32) u64 {
    return @as(u64, value) << 20;
}

/// Read the `--preload` files before the model loads, so a bad path fails in
/// milliseconds. An oversize or empty file fails too, because the operator asked
/// for warmth and a silent skip is a correctness bug.
fn readPreload(ctx: context.Context, serve: *const flags_mod.ServeFlags) ![]const []const u8 {
    const paths = serve.preload.slice();
    std.debug.assert(paths.len <= limits.serve_preload_max);

    const limit: Io.Limit = .limited(limits.serve_preload_text_max);
    const texts = try ctx.arena.alloc([]const u8, paths.len);
    for (paths, texts) |path, *out| {
        const text = Io.Dir.cwd().readFileAlloc(ctx.io, path, ctx.arena, limit) catch |err| {
            const cap_kib = limits.serve_preload_text_max / kib;
            return args.fail(ctx.io, "--preload {s}: {t} (cap {d} KiB)", .{ path, err, cap_kib });
        };
        if (text.len == 0) return args.fail(ctx.io, "--preload {s}: the file is empty", .{path});
        out.* = text;
    }

    return texts;
}

/// The requested slot count and the structural one. With no flag every session
/// is resident, unless a `--kv-vram` budget lets the planner buy the count.
fn checkSlots(
    io: Io,
    serve: *const flags_mod.ServeFlags,
    budgets: ByteBudgets,
) !SlotCounts {
    const requested: ?u16 = blk: {
        const value = serve.vram_slots orelse break :blk null;
        if (!fused_build) {
            return args.fail(
                io,
                "--vram-slots applies only to the fused-GPU build (VRAM<->RAM KV swapping)",
                .{},
            );
        }
        if (value == 0 or value > limits.max_vram_slots) {
            return args.fail(io, "--vram-slots must be in 1..{d}", .{limits.max_vram_slots});
        }

        // A slot no session can occupy is VRAM bought and left dead.
        break :blk @min(value, serve.max_sessions);
    };

    const structural: u16 = requested orelse blk: {
        if (budgets.kv_vram != null) {
            break :blk @intCast(@min(@as(u32, serve.max_sessions), limits.max_batch));
        }
        if (serve.max_sessions > limits.max_batch) {
            return args.fail(
                io,
                "--max-sessions > {d} needs --vram-slots or --kv-vram to swap KV " ++
                    "between VRAM and RAM",
                .{limits.max_batch},
            );
        }

        break :blk serve.max_sessions;
    };

    // The CPU runner holds one slot per session, so a zero pin with fewer slots
    // than sessions refuses before the model loads.
    if (serve.engine.gpu_layers) |g| {
        if (g.isZeroCount()) {
            try refuseCpuSlots(io, structural, serve.max_sessions);
        }
    }

    return .{ .requested = requested, .structural = structural };
}

/// The decode width: slots bound breadth, and `limits.max_batch` bounds the rows
/// one tick advances.
fn checkBatch(io: Io, serve: *const flags_mod.ServeFlags, structural_slots: u16) !u16 {
    const width_cap: u16 = @intCast(@min(@as(u32, structural_slots), limits.max_batch));
    if (serve.max_batch) |req| {
        if (req == 0 or req > limits.max_batch) {
            return args.fail(io, "--max-batch must be in 1..{d}", .{limits.max_batch});
        }
    }

    const batch: u16 = @min(serve.max_batch orelse width_cap, width_cap);
    if (serve.max_batch) |req| {
        if (req > batch) {
            log.info("--max-batch {d} is above the {d} slots, so the decode width is {d}", .{
                req, structural_slots, batch,
            });
        }
    }

    if (batch == 0) return args.fail(io, "--max-batch must be >= 1", .{});

    return batch;
}

/// The CPU arm holds every session resident, and its plan funds no blob. Fewer
/// slots than sessions then park a turn with no eviction to free a slot.
fn refuseCpuSlots(io: Io, n_slots: u16, max_sessions: u16) error{UsageError}!void {
    if (n_slots >= max_sessions) return;
    return args.fail(io, boot_lines.cpu_slots_fmt, .{ n_slots, max_sessions });
}
