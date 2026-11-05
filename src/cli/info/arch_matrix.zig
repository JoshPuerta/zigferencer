//! The support matrix of which op backends can execute each model architecture. `zigferencer info`
//! prints it and marks the backend linked into this binary. The notes carry the limits that a
//! reader
//! otherwise finds in the runners: the weight quants a backend accepts, and what runs where.
//!
//! A comptime check pins every row to `op_requires.ops_backend_names`, in order, so a new backend
//! forces every architecture to state its support.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const style = @import("../style.zig");
const op_requires = @import("../../core/op_requires.zig");

// -- constants --------------------------------------------------------------

const arch_rows = [_]ArchRow{
    .{
        .title = "llama: SmolLM2, Llama 3.x (full softmax attention)",
        .entries = .{
            .{ .backend = "naive", .runtime = "cpu", .note = "scalar f32 reference. q8_0 linears" },
            .{ .backend = "simd", .runtime = "cpu", .note = "AVX2 @Vector(8) linear kernels, rest shared with naive. q8_0 linears" },
            .{ .backend = "simd_pf", .runtime = "cpu", .note = "AVX2+FMA register-blocked + prefetch (the auto pick). q8_0 linears" },
            .{ .backend = "vulkan", .runtime = "vulkan", .note = "per-op GPU submits (correctness/bench reference). q8_0 linears" },
            .{ .backend = "vulkan_fused", .runtime = "vulkan", .note = "one command buffer per token, weights in VRAM. q8_0, or q4_0 with a q6_k head" },
        },
    },
    .{
        .title = "qwen35: Qwen3.5 hybrid (Gated-DeltaNet + full attention)",
        .entries = .{
            .{ .backend = "naive", .runtime = "cpu", .note = "GDN recurrence + attention on CPU. q8_0" },
            .{ .backend = "simd", .runtime = "cpu", .note = "SIMD linear kernels, GDN shared with naive. q8_0" },
            .{ .backend = "simd_pf", .runtime = "cpu", .note = "register-blocked linears, GDN shared with naive. q8_0" },
            .{ .backend = "vulkan", .runtime = "vulkan", .note = "run only: matmuls per-op on GPU, GDN stays on CPU. bench-model unsupported" },
            .{ .backend = "vulkan_fused", .runtime = "vulkan", .note = "one command buffer per token, GDN + attention on GPU. q8_0 only" },
        },
    },
};

comptime {
    for (arch_rows) |row| {
        for (row.entries, op_requires.ops_backend_names) |entry, canonical| {
            if (!std.mem.eql(u8, entry.backend, canonical)) {
                @compileError("arch matrix row '" ++ row.title ++ "' diverges from " ++
                    "op_requires.ops_backend_names at '" ++ canonical ++ "'");
            }
        }
    }
}

// -- types ------------------------------------------------------------------

/// One backend's support statement for one architecture.
const Entry = struct {
    backend: []const u8,
    runtime: []const u8,
    note: []const u8,
};

const ArchRow = struct {
    title: []const u8,
    // One entry per backend, in the order of `ops_backend_names`.
    entries: [op_requires.ops_backend_names.len]Entry,
};

// -- api --------------------------------------------------------------------

/// Print the matrix to `w`. The caller flushes.
pub fn print(w: *Io.Writer, p: style.Palette) Io.Writer.Error!void {
    try w.print(
        "\n{s}Supported architectures{s}   {s}[+]{s} current backend  {s}[-] available via " ++
            "rebuild{s}\n",
        .{
            p.bold, p.reset, p.green, p.reset, p.dim, p.reset,
        },
    );
    try w.print("{s}[*] ops backend: {s}  (rebuild: -Dops-backend=<name>){s}\n", .{
        p.dim, build_options.ops_backend, p.reset,
    });

    for (&arch_rows) |*row| {
        try w.print("\n{s}{s}{s}\n", .{ p.bold, row.title, p.reset });

        for (&row.entries) |*entry| {
            const linked = std.mem.eql(u8, entry.backend, build_options.ops_backend);
            const mark_c = if (linked) p.green else p.dim;
            const mark: []const u8 = if (linked) "+" else "-";
            try w.print("  {s}[{s}]{s} {s}{s:<13}{s} {s}{s:<7}{s} {s}\n", .{
                mark_c,     mark,          p.reset,
                p.bold,     entry.backend, p.reset,
                p.dim,      entry.runtime, p.reset,
                entry.note,
            });
        }
    }

    try w.print("\n{s}[*] vulkan backends need -Druntime=vulkan (glslc + libvulkan-dev){s}\n", .{
        p.dim, p.reset,
    });
}

// -- tests ------------------------------------------------------------------

const test_buffer_bytes = 4096;

test "the linked backend is always one of the canonical names" {
    var found = false;
    for (op_requires.ops_backend_names) |name| {
        if (std.mem.eql(u8, name, build_options.ops_backend)) {
            found = true;
        }
    }

    try std.testing.expect(found);
}

test "print names every architecture and marks exactly the linked backend" {
    var buf: [test_buffer_bytes]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try print(&w, style.Palette.init(false));

    const text = w.buffered();
    for (arch_rows) |row| {
        try std.testing.expect(std.mem.find(u8, text, row.title) != null);
    }

    // One linked mark per architecture row, plus the legend.
    try std.testing.expectEqual(arch_rows.len + 1, std.mem.count(u8, text, "[+]"));
}
