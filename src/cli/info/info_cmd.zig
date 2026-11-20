//! `zigferencer info`: print the host hardware card, the build options, the threading defaults, and
//! the architecture support matrix. A Vulkan build also prints the storage-buffer limits of the
//! device, because they bound the KV window.
//!
//! Comptime. A build option decides whether the storage-buffer limits print.

const std = @import("std");
const Io = std.Io;
const builtin = @import("builtin");
const build_options = @import("build_options");

const context = @import("../context.zig");
const args = @import("../args.zig");
const style = @import("../style.zig");
const arch_matrix = @import("arch_matrix.zig");
const log = @import("../../core/utils/log.zig");
const hwinfo = @import("../../core/utils/hwinfo.zig");
const op_requires = @import("../../core/op_requires.zig");
const naive_ops = @import("../../ops/naive.zig");
const simd_ops = @import("../../ops/simd.zig");
const simd_pf_ops = @import("../../ops/simd_pf.zig");
const parallel = @import("../../runtime/parallel.zig");
const limits = @import("../../core/limits.zig");
const vulkan_ctx = if (build_options.use_vulkan) @import("../../runtime/vulkan_ctx.zig") else void;

// -- constants --------------------------------------------------------------

const all_backends = [_]BackendEntry{
    .{ .name = "naive", .req = naive_ops.requires },
    .{ .name = "simd", .req = simd_ops.requires },
    .{ .name = "simd_pf", .req = simd_pf_ops.requires },
    .{ .name = "vulkan", .note = "GPU per-op    (needs -Druntime=vulkan)" },
    .{ .name = "vulkan_fused", .note = "GPU fused     (needs -Druntime=vulkan)" },
};

// -- types ------------------------------------------------------------------

const BackendEntry = struct {
    name: []const u8,
    /// Null marks a GPU backend. `note` then replaces the arch and SIMD fields.
    req: ?op_requires.OpRequires = null,
    note: []const u8 = "",
};

/// The flags of `info`. It takes none.
const Flags = struct {
    pub const help: args.Help = .{
        .usage = "zigferencer info",
        .about =
        \\Print the host CPU / RAM / SIMD card, this binary's build config
        \\(optimize, asserts, runtime, ops backend + per-host eligibility), the
        \\active $ZIGFERENCER_LOG trace level, the threading defaults, and the
        \\architecture / op-backend support matrix ([+] marks this backend).
        \\A Vulkan build also prints the device storage-buffer limits and the
        \\range that $ZIGFERENCER_VK_STORAGE_RANGE sets. Takes no flags.
        \\
        ,
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    _ = try args.parseInto(Flags, ctx) orelse return;

    // Print each section of the card.
    const card = hwinfo.detect(ctx.io, ctx.arena);
    const p = style.Palette.init(ctx.color);

    const w = &ctx.stdout.interface;

    try printHost(w, p, card);
    try printSimd(w, p, card.simd);
    try printBuild(w, p);
    try printBackends(w, p, card);
    try printThreading(ctx, w, p);
    try arch_matrix.print(w, p);

    if (comptime build_options.use_vulkan) {
        try printBufferLimits(ctx, w, p);
    }

    try w.flush();
}

// -- helpers ----------------------------------------------------------------

fn printHost(w: *Io.Writer, p: style.Palette, card: hwinfo.HwCard) !void {
    try w.print("{s}Host:{s}\n", .{ p.bold, p.reset });
    try w.print("  arch:      {s}\n", .{card.arch});
    try w.print("  os:        {s}\n", .{card.os});
    if (card.cpu_model) |m| {
        try w.print("  cpu:       {s}\n", .{m});
    }

    try w.print("  cores:     {d}\n", .{card.cores_total});
    if (card.ram_total_bytes) |r| {
        try w.print("  ram:       {d:.1} GiB\n", .{
            @as(f64, @floatFromInt(r)) / (1024.0 * 1024.0 * 1024.0),
        });
    }

    const gpu_c = if (card.gpu.present) p.green else p.dim;
    try w.print("  gpu:       {s}{s}{s}\n", .{ gpu_c, card.gpu.label, p.reset });
}

/// Print the build identity: the options `-Druntime` and `-Dops-backend`, and the trace level.
fn printBuild(w: *Io.Writer, p: style.Palette) !void {
    try w.print("\n{s}Build (this binary):{s}\n", .{ p.bold, p.reset });
    try w.print("  optimize:        {s}\n", .{@tagName(builtin.mode)});

    // Asserts run in Debug and ReleaseSafe. ReleaseFast strips them.
    const asserts_on = builtin.mode == .Debug or builtin.mode == .ReleaseSafe;
    const ac = if (asserts_on) p.green else p.dim;
    const asserts_text = if (asserts_on)
        "on"
    else
        "off (ReleaseFast, build with -Doptimize=ReleaseSafe to enable)";
    try w.print("  asserts:         {s}{s}{s}\n", .{ ac, asserts_text, p.reset });
    try w.print(
        "  runtime:         {s}{s:<8}{s}  (cpu | vulkan)\n",
        .{ p.green, build_options.runtime, p.reset },
    );
    try w.print("  ops backend:     {s}{s}{s}\n", .{ p.green, build_options.ops_backend, p.reset });

    // The value echoes byte for byte, so a misspelling stays visible.
    if (log.rawEnv()) |v| {
        try w.print("  trace level:     {s}{s}{s}\n", .{ p.green, v, p.reset });
    } else {
        try w.print("  trace level:     {s}info (unset){s}\n", .{ p.dim, p.reset });
    }

    try w.print(
        "  {s}($ZIGFERENCER_LOG=debug. Scopes: " ++
            "serve serve_openai serve_demo run runner kv gpu gguf{s}\n",
        .{ p.dim, p.reset },
    );
    try w.print(
        "  {s} fetch parallel cache bench_swap bench_serve + hot: " ++
            "layer dispatch token, see core/utils/log.zig){s}\n",
        .{ p.dim, p.reset },
    );

    if (log.hot) {
        try w.print("  trace:           {s}compiled (-Dtrace){s}\n", .{ p.green, p.reset });
    } else {
        try w.print(
            "  trace:           {s}off (build with -Dtrace=true to compile){s}\n",
            .{ p.dim, p.reset },
        );
    }
}

fn printBackends(w: *Io.Writer, p: style.Palette, card: hwinfo.HwCard) !void {
    try w.print("\n{s}Op backends{s} ({s}[+]{s} runnable):\n", .{
        p.bold, p.reset, p.green, p.reset,
    });
    for (all_backends) |b| {
        // CPU backends gate on host SIMD. GPU backends gate on a GPU being present.
        const eligible = if (b.req) |r| op_requires.isMetByHost(r, card) else card.gpu.present;
        const active = std.mem.eql(u8, b.name, build_options.ops_backend);
        try printBackend(w, p, b, eligible, active);
    }

    try w.print(
        "  {s}(rebuild: -Dops-backend=<name>  [+ -Druntime=vulkan for GPU]){s}\n",
        .{ p.dim, p.reset },
    );
}

fn printThreading(ctx: context.Context, w: *Io.Writer, p: style.Palette) !void {
    const cores = parallel.hostCores(ctx.io);
    const host_policy = parallel.Policy.forHost(ctx.io);

    try w.print("\n{s}Threading defaults:{s}\n", .{ p.bold, p.reset });
    const source = if (cores.from_os_topology) "OS topology" else "half the logical CPUs";
    try w.print("  physical cores:        {d}  {s}({s}){s}\n", .{
        cores.physical, p.dim, source, p.reset,
    });
    try w.print(
        "  threads:               {d}  {s}(min(physical cores, {d})){s}\n",
        .{ host_policy.n_threads, p.dim, parallel.max_default_threads, p.reset },
    );
    try w.print("  min_parallel_flops:    {d}\n", .{host_policy.min_parallel_flops});
    try w.print("  spin_ns:               {d}\n", .{host_policy.spin_ns});
    try w.print(
        "  {s}(override: --threads N, --min-parallel-flops F, --spin-us N){s}\n",
        .{ p.dim, p.reset },
    );
}

/// Print the storage-buffer limits of the device and the range that the buffer checks read. A host
/// with no usable device prints the open error and keeps the rest of the card.
fn printBufferLimits(ctx: context.Context, w: *std.Io.Writer, p: style.Palette) !void {
    try w.print("\n{s}GPU buffer limits:{s}\n", .{ p.bold, p.reset });
    var vc = vulkan_ctx.Context.init(ctx.arena, ctx.vk_options) catch |err| {
        try w.print("  {s}no device context ({t}){s}\n", .{ p.dim, err, p.reset });
        return;
    };
    defer vc.deinit();

    const lim = vc.buffer_limits;
    try w.print(
        "  device:                 {s}\n",
        .{std.mem.sliceTo(&vc.physical_device_props.device_name, 0)},
    );
    try w.print("  reported storage range: {d} bytes\n", .{vc.reported_storage_range});
    try w.print("  allocation size:        {d} bytes\n", .{lim.allocation_size});
    try w.print("  offset alignment:       {d} bytes\n", .{lim.offset_alignment});
    if (ctx.vk_options.storage_range_override != null) {
        try w.print(
            "  storage-buffer range {s}{d}{s} bytes (reported {d}, set by " ++
                "" ++ limits.storage_range_env ++ ")\n",
            .{ p.green, lim.storage_range, p.reset, vc.reported_storage_range },
        );
    } else {
        try w.print("  storage-buffer range {d} bytes (reported)\n", .{lim.storage_range});
    }

    try w.print(
        "  {s}(override: " ++ limits.storage_range_env ++ "=<bytes>, 1..{d}){s}\n",
        .{ p.dim, limits.storage_range_max, p.reset },
    );
}

/// Print only the ISA flags of the build target, because the other ISA is always off.
fn printSimd(w: *std.Io.Writer, p: style.Palette, s: hwinfo.SimdFeatures) !void {
    try w.print("\n{s}SIMD (build target):{s}\n", .{ p.bold, p.reset });
    switch (builtin.cpu.arch) {
        .x86, .x86_64 => {
            try printFlag(w, p, "sse2", s.sse2);
            try printFlag(w, p, "sse4_1", s.sse4_1);
            try printFlag(w, p, "sse4_2", s.sse4_2);
            try printFlag(w, p, "avx", s.avx);
            try printFlag(w, p, "avx2", s.avx2);
            try printFlag(w, p, "fma", s.fma);
            try printFlag(w, p, "f16c", s.f16c);
            try printFlag(w, p, "avx512f", s.avx512f);
            try printFlag(w, p, "avx512vnni", s.avx512vnni);
            try printFlag(w, p, "avx_vnni", s.avx_vnni);
        },
        .aarch64, .aarch64_be => {
            try printFlag(w, p, "neon", s.neon);
            try printFlag(w, p, "dotprod", s.dotprod);
            try printFlag(w, p, "fp16", s.fp16);
            try printFlag(w, p, "sve", s.sve);
        },
        else => {},
    }
}

fn printBackend(
    w: *std.Io.Writer,
    p: style.Palette,
    b: BackendEntry,
    eligible: bool,
    active: bool,
) !void {
    const mark_c = if (eligible) p.green else p.dim;
    const mark: []const u8 = if (eligible) "+" else "-";
    try w.print("  {s}[{s}]{s} {s:<14} {s}", .{ mark_c, mark, p.reset, b.name, p.dim });
    if (b.req) |req| {
        if (req.arch == .any and req.min_simd == .none and !req.needs_fma) {
            try w.writeAll("(no requirements)");
        } else {
            try w.print("arch={s}  min_simd={s}", .{ @tagName(req.arch), @tagName(req.min_simd) });
            if (req.needs_fma) {
                try w.writeAll("  +fma");
            }
        }
    } else {
        try w.writeAll(b.note);
    }

    try w.writeAll(p.reset);
    if (active) {
        try w.print("  {s}{s}<-- active{s}", .{ p.bold, p.green, p.reset });
    }

    try w.writeAll("\n");
}

fn printFlag(w: *std.Io.Writer, p: style.Palette, name: []const u8, on: bool) !void {
    const c = if (on) p.green else p.dim;
    const mark: []const u8 = if (on) "+" else "-";
    try w.print("  {s}[{s}] {s}{s}\n", .{ c, mark, name, p.reset });
}
