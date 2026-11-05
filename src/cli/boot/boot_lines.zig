//! The lines a boot prints: the boot log lines of `run`, `serve`, and the benches, and the
//! message of each `resolve` refusal. Every line keeps the `serve` log scope.

const std = @import("std");
const Io = std.Io;

const args = @import("../args.zig");
const resolve_mod = @import("../../engine/resolve.zig");
const dtype = @import("../../core/dtype.zig");
const kv_budget = @import("../../core/kv_budget.zig");
const limits = @import("../../core/limits.zig");
const loader = @import("../../io/gguf/loader.zig");
const tensor_lookup = @import("../../runtime/tensor_lookup.zig");

const log = std.log.scoped(.serve);
const device_arm = resolve_mod.device_arm;

// -- constants --------------------------------------------------------------

const mib = 1024 * 1024;

/// The refusal of a slot count below the session count on the CPU runner. It takes
/// the slots and the sessions.
pub const cpu_slots_fmt = "the CPU runner holds one slot per session, so --vram-slots {d} " ++
    "cannot serve --max-sessions {d}. Drop --vram-slots, or set it to the --max-sessions value";

/// The refusal of a `--max-seq` value that `baseWindow` puts out of range. It
/// takes the model ceiling.
pub const window_out_of_range_fmt = "--max-seq must be in 1..{d} for this model";

// -- types ------------------------------------------------------------------

/// The window of a boot before any device fit. `resolve` states it.
pub const SessionWindow = resolve_mod.Window;

/// Where a window that no device fit cut came from. A named window is the
/// flag. A default at the model ceiling is the ceiling, and a default below it
/// is the session byte budget.
const WindowSource = enum { flag, ceiling, budget };

// -- api --------------------------------------------------------------------

/// Print one refusal of `resolve` with its message, and return the error the
/// exit path knows. `name` is the public name of the model.
pub fn refuse(io: Io, name: []const u8, r: resolve_mod.Refusal) anyerror {
    const no_mode = " does not apply to {s}. Its chat template has no thinking mode";
    switch (r) {
        .thinking_unsupported => return args.fail(
            io,
            "--thinking on" ++ no_mode,
            .{name},
        ),
        .thinking_budget_unsupported => return args.fail(
            io,
            "--thinking-budget" ++ no_mode,
            .{name},
        ),
        .serve_per_op_vulkan => {
            log.err("serve does not run on the per-op vulkan backend. " ++
                "Build -Dops-backend=vulkan_fused or a CPU backend", .{});
            return error.UnsupportedServeBackend;
        },
        .kv_dtype_per_op_vulkan => |d| return args.fail(
            io,
            "--kv-dtype {t} does not run on -Dops-backend=vulkan (that backend uploads the " ++
                "KV slab to the device as f32 on every attention call). Build a CPU backend",
            .{d},
        ),
        .window_out_of_range => |ceiling| return args.fail(
            io,
            window_out_of_range_fmt,
            .{ceiling},
        ),
        .cpu_slots => |c| return args.fail(io, cpu_slots_fmt, .{ c.slots, c.sessions }),
        .host_ram => |q| {
            var buf: [kv_budget.ram_refusal_text_max]u8 = undefined;
            log.err("{s}", .{kv_budget.formatRamRefusal(&buf, q)});
            return error.HostRamBudgetTooSmall;
        },
        .blob_ram => |q| {
            var buf: [kv_budget.blob_refusal_text_max]u8 = undefined;
            log.err("{s}", .{kv_budget.formatBlobRefusal(&buf, q)});
            return error.BlobRamBudgetTooSmall;
        },
        .gpu_layers_past_model => |g| return args.fail(
            io,
            "--gpu-layers {d} exceeds the model's {d} layers",
            .{ g.n, g.n_layers },
        ),
        .hybrid_slots => |n| return args.fail(
            io,
            "--gpu-layers holds at most {d} vram kv slots, and this run asks for {d}. " ++
                "Lower --vram-slots, or drop --gpu-layers",
            .{ limits.max_batch, n },
        ),
        .placement_vram => |p| return refusePlacementVram(io, p),
        .whole_model_split => |pin| return refuseWholeModelSplit(io, pin),
        .kv_vram_ceiling => |k| {
            log.err("kv budget: --max-seq {d} is past the model ceiling of {d} positions", .{
                k.window, k.ceiling,
            });
            return error.WindowExceedsCeiling;
        },
        .kv_vram_small => |k| {
            logKvVramSmall(k);
            return error.KvVramBudgetTooSmall;
        },
    }
}

/// The boot lines of a whole-model fused arm, in boot order, and its RAM
/// quote. `session` names `run` and `bench-model`, which log the window line
/// when no cut replaces it.
pub fn logFused(
    io: Io,
    name: []const u8,
    f: device_arm.Fused,
    session: bool,
    kv_ram_named: bool,
    kv_dtype: dtype.KvDtype,
) !kv_budget.RamQuote {
    if (f.clamp) |c| {
        logClamp(c);
    }

    if (f.cut) |c| {
        log.info("kv window: {d} slots at the default window of {d} positions do not fit " ++
            "the free VRAM, so the window is {d}", .{ c.n_slots, c.base, c.window });
    } else if (session) {
        logSessionWindow(f.window);
    }

    if (f.window.flag) |named| {
        if (f.plan.window < named) {
            log.warn("kv window: --vram-slots {d} at --max-seq {d} does not fit the VRAM " ++
                "budget, so the window is {d}", .{ f.plan.n_slots, named, f.plan.window });
        }
    }

    if (f.stranded) |st| {
        log.warn("kv budget: the ceiling of {d} slots cut the count this budget bought. " ++
            "{d} MiB of the {d} MiB stays unspendable. Raise --max-seq to give each slot " ++
            "a wider window", .{ st.max_slots, st.stranded / mib, st.x_bytes / mib });
    }

    if (f.device_slots) |ds| {
        log.info("kv budget: the {s} of {d} {t} holds one KV region of {d} slots at {d} " ++
            "positions, so the budget buys at most {d} slots", .{
            ds.limit.name, ds.limit.value, ds.limit.unit, ds.slots, ds.window, ds.slots,
        });
    }

    if (f.prefix_default) {
        logPrefixDefault();
    }

    logPlan(f.plan, kv_dtype);

    return switch (f.quote) {
        .ok => |q| blk: {
            logRamQuote(q, kv_ram_named);
            break :blk q;
        },
        .refused => |r| refuse(io, name, r),
    };
}

/// The line of a default window that the device buffer limits cut.
pub fn logClamp(c: device_arm.Clamp) void {
    log.info("kv window: {d} slots at the default window of {d} positions need a KV region " ++
        "of {d} {t}. The {s} is {d} {t}, so the window is {d}", .{
        c.n_slots,
        c.base,
        c.need,
        c.limit.unit,
        c.limit.name,
        c.limit.value,
        c.limit.unit,
        c.window,
    });
}

/// The resolved count of `--gpu-layers auto`, so a log names the split the live
/// budget chose.
pub fn logAutoSplit(split: u32, n_layers: u32) void {
    std.debug.assert(n_layers >= 1);
    std.debug.assert(split <= n_layers);

    log.info("--gpu-layers auto: gpu {d}/{d} layers", .{ n_layers - split, n_layers });
}

/// The startup witness of the KV economy. `kv_dtype` is the dtype the prices
/// came from.
pub fn logPlan(plan: kv_budget.KvPlan, kv_dtype: dtype.KvDtype) void {
    log.info("kv plan: {d} slots x {d} window{s} = {d} MiB vram kv, {d} MiB ram, {d} MiB prefix", .{
        plan.n_slots,
        plan.window,
        kv_dtype.planToken(),
        plan.vram.kv / mib,
        plan.ram.blobs / mib,
        plan.ram.prefix / mib,
    });
}

/// The whole precision map of a boot. The engine holds no graph IR a reader can
/// inspect, so this line is where the per-site precisions become visible.
pub fn logPrecision(loaded: *const loader.Loaded, kv_dtype: dtype.KvDtype) void {
    log.info("precision: weights {s}, activations f32, accum f32, kv {t}, logits f32", .{
        weightQuantName(loaded),
        kv_dtype,
    });
}

/// Log the base window of a one-session command and its source. A window that the
/// device limits cut logs nothing here, because the clamp logged it with both
/// numbers.
pub fn logSessionWindow(w: SessionWindow) void {
    if (w.device_clamped) return;

    switch (windowSource(w.base, w.ceiling, w.flag)) {
        .flag => log.info("kv window: {d} positions, set by --max-seq (model ceiling {d})", .{
            w.base, w.ceiling,
        }),
        .ceiling => log.info("kv window: {d} positions, the model ceiling", .{w.base}),
        .budget => log.info("kv window: {d} positions, the {d} MiB session budget " ++
            "(model ceiling {d})", .{ w.base, w.session_bytes / mib, w.ceiling }),
    }
}

/// The boot lines of one RAM quote: the spend, then the blob tier. An absent
/// tier prints no tier line, because no blob can exist.
pub fn logRamQuote(q: kv_budget.RamQuote, kv_ram_named: bool) void {
    const r = q.ram;
    const parts_fmt = "{d} MiB host kv + {d} MiB blobs + {d} MiB prefix + {d} MiB stages";
    if (q.usable) |usable| {
        log.info("ram: {d} of {d} MiB usable = " ++ parts_fmt, .{
            r.spend() / mib,
            usable / mib,
            r.host_kv / mib,
            r.blobs / mib,
            r.prefix / mib,
            r.swap_stages / mib,
        });
    } else {
        log.info("ram: {d} MiB, not checked = " ++ parts_fmt, .{
            r.spend() / mib, r.host_kv / mib, r.blobs / mib, r.prefix / mib, r.swap_stages / mib,
        });
    }

    switch (q.tier) {
        .absent => {},
        .unenforced => log.info("kv ram tier: unenforced, {d} MiB need ({d} evictable sessions " ++
            "+ 1 in a swap, {d} MiB each). Set --kv-ram to bound it", .{
            q.need / mib, q.evictable, q.blob / mib,
        }),
        .bounded => |bytes| {
            log.info("kv ram tier: bounded, {d} MiB for {d} evictable sessions " ++
                "(need {d} MiB, {d} MiB a blob)", .{
                bytes / mib, q.evictable, q.need / mib, q.blob / mib,
            });
            // A named budget below the need is a choice of backpressure, so
            // the boot accepts it and says what it costs.
            if (kv_ram_named and bytes < q.need) {
                log.warn("kv ram tier: --kv-ram funds {d} MiB of blobs, but the need is " ++
                    "{d} MiB. A turn can wait until another session closes", .{
                    bytes / mib, q.need / mib,
                });
            }
        },
    }
}

/// The lines of a `--gpu-layers` boot before its arm.
pub fn logPlaced(p: device_arm.Placed, n_layers: u32) void {
    if (p.slots_from) |from| {
        log.info("hybrid: {d} vram kv slots, clamped from {d}", .{ limits.max_batch, from });
    }

    if (p.clamp) |c| {
        logClamp(c);
    }

    if (p.prefix_default) {
        logPrefixDefault();
    }

    if (p.split) |s| {
        var span_buffer: [kv_budget.placement_text_max]u8 = undefined;
        log.info("{s}", .{kv_budget.formatPlacement(s, &span_buffer)});
        if (p.pin == null) {
            logAutoSplit(s.split, n_layers);
        }
    }

    if (p.selects == .fused) {
        logFusedArm(p.pin);
    }
}

/// Zero-config serve keeps the prefix cache at a named default when no
/// `--kv-ram` names a budget.
pub fn logPrefixDefault() void {
    log.info("prefix cache: {d} MiB default budget (set --kv-ram to size both ram tiers)", .{
        limits.serve_prefix_ram_default / mib,
    });
}

// -- helpers ----------------------------------------------------------------

fn refusePlacementVram(io: Io, p: @FieldType(resolve_mod.Refusal, "placement_vram")) anyerror {
    const advice = switch (p.kind) {
        .run => "Lower --gpu-layers, or use a smaller / Q4 model",
        .serve => "Lower --gpu-layers or --max-seq, or use a smaller / Q4 model",
        .bench => "Lower --gpu-layers or use a smaller / Q4 model",
    };

    return args.fail(
        io,
        "--gpu-layers {d} needs ~{d} MiB of device budget, usable ~{d} MiB. {s}",
        .{ p.n_gpu, p.spend / mib, p.usable / mib, advice },
    );
}

fn refuseWholeModelSplit(io: Io, pin: ?u32) anyerror {
    const served = " resolves to a whole-model placement, which the fused (or CPU) " ++
        "runner already serves. Drop the flag, ";
    if (pin) |n_gpu| {
        return args.fail(
            io,
            "--gpu-layers {d}" ++ served ++
                "or pass --force-hybrid to construct the hybrid at this split for the " ++
                "cross-runner gate",
            .{n_gpu},
        );
    }

    return args.fail(
        io,
        "--gpu-layers auto" ++ served ++
            "or pass a count with --force-hybrid to construct the hybrid at that " ++
            "split for the cross-runner gate",
        .{},
    );
}

fn logKvVramSmall(k: @FieldType(resolve_mod.Refusal, "kv_vram_small")) void {
    if (k.clamp) |c| {
        logClamp(c);
    }

    log.err("kv budget: {d} MiB of KV cannot fund {d} slot(s). One slot costs {d:.1} " ++
        "MiB at window {d}, so raise --kv-vram to {d} MiB or more, or lower " ++
        "--max-seq / --vram-slots", .{
        k.x_bytes / mib,
        k.rows,
        @as(f64, @floatFromInt(k.per_slot)) / mib,
        k.window,
        (k.per_slot * k.rows + mib - 1) / mib,
    });
}

/// The line of a placement that funds the whole model on the device. Null is
/// the auto pin.
fn logFusedArm(pin: ?u32) void {
    const funded = "the budget funds the whole model, so the fused runner serves it";
    if (pin) |n| {
        log.info("--gpu-layers {d}: " ++ funded, .{n});
    } else {
        log.info("--gpu-layers auto: " ++ funded, .{});
    }
}

fn windowSource(window: u32, ceiling: u32, flag: ?u32) WindowSource {
    std.debug.assert(window >= 1);
    std.debug.assert(window <= ceiling);

    if (flag) |w| {
        std.debug.assert(w == window);

        return .flag;
    }

    return if (window == ceiling) .ceiling else .budget;
}

/// The weight quant the precision line names: the type of the first down projection,
/// which every block of every family holds.
fn weightQuantName(loaded: *const loader.Loaded) []const u8 {
    const t = tensor_lookup.find(loaded.tensors, "blk.0.ffn_down.weight") orelse return "unknown";
    return std.enums.tagName(dtype.GgmlType, t.dtype) orelse "unknown";
}

// -- tests ------------------------------------------------------------------

test "windowSource: a flag names its window, and a default names the ceiling or the budget that bound it" {
    const ceiling: u32 = 8192;
    try std.testing.expectEqual(WindowSource.ceiling, windowSource(ceiling, ceiling, null));
    try std.testing.expectEqual(WindowSource.budget, windowSource(2816, ceiling, null));
    // A named window is the flag, even at the ceiling or at the default value.
    for ([_]u32{ 1, 2816, ceiling }) |w| {
        try std.testing.expectEqual(WindowSource.flag, windowSource(w, ceiling, w));
    }

    // The smallest ceiling holds one position, and a default there is the ceiling.
    try std.testing.expectEqual(WindowSource.ceiling, windowSource(1, 1, null));
}
