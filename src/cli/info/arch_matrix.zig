//! The support matrix of which op backends can execute each model architecture. `zigferencer info`
//! prints it and marks the backend linked into this binary. The notes carry the limits that a
//! reader otherwise finds in the runners: the weight quants a backend accepts, and what runs where.
//!
//! Comptime. A check pins every architecture entry to `op_requires.ops_backend_names`, in order,
//! so a new backend forces every architecture to state its support.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const style = @import("../style.zig");
const op_requires = @import("../../core/op_requires.zig");

// -- constants --------------------------------------------------------------

const arch_entries = [_]ArchEntry{
    .{
        .title = "llama: SmolLM2, Llama 3.x (full softmax attention)",
        .entries = .{
            cpu("naive", "scalar f32 reference. q8_0 linears"),
            cpu("simd", "@Vector(8) linears, attention, norm, and activations. q8_0 linears"),
            cpu("simd_pf", "simd with prefetch in linears and attention (the auto pick). q8_0"),
            vulkan("vulkan", "per-op GPU submits (correctness/bench reference). q8_0 linears"),
            vulkan(
                "vulkan_fused",
                "one command buffer per token, weights in VRAM. q8_0, or q4_0 with a q6_k head",
            ),
        },
    },
    .{
        .title = "qwen35: Qwen3.5 hybrid (Gated-DeltaNet + full attention)",
        .entries = .{
            cpu("naive", "GDN recurrence + attention on CPU. q8_0"),
            cpu("simd", "the simd kernels of llama, and the GDN key-head unit. q8_0"),
            cpu("simd_pf", "simd with prefetch (the auto pick), and the GDN key-head unit. q8_0"),
            vulkan(
                "vulkan",
                "run only: matmuls per-op on GPU, GDN stays on CPU. bench-model unsupported",
            ),
            vulkan(
                "vulkan_fused",
                "one command buffer per token, GDN + attention on GPU. q8_0 only",
            ),
        },
    },
};

comptime {
    for (arch_entries) |arch| {
        for (arch.entries, op_requires.ops_backend_names) |entry, canonical| {
            if (!std.mem.eql(u8, entry.backend, canonical)) {
                @compileError("arch matrix entry '" ++ arch.title ++ "' diverges from " ++
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

const ArchEntry = struct {
    title: []const u8,
    /// One entry per backend, in the order of `ops_backend_names`.
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

    for (&arch_entries) |*arch| {
        try w.print("\n{s}{s}{s}\n", .{ p.bold, arch.title, p.reset });

        for (&arch.entries) |*entry| {
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

    const hint = "\n{s}[*] vulkan backends need -Druntime=vulkan " ++
        "(glslc + vk.xml to build, the Vulkan loader to run){s}\n";
    try w.print(hint, .{ p.dim, p.reset });
}

// -- helpers ----------------------------------------------------------------

fn cpu(backend: []const u8, note: []const u8) Entry {
    return .{ .backend = backend, .runtime = "cpu", .note = note };
}

fn vulkan(backend: []const u8, note: []const u8) Entry {
    return .{ .backend = backend, .runtime = "vulkan", .note = note };
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
    for (arch_entries) |arch| {
        try std.testing.expect(std.mem.find(u8, text, arch.title) != null);
    }

    // One linked mark per architecture entry, plus the legend.
    try std.testing.expectEqual(arch_entries.len + 1, std.mem.count(u8, text, "[+]"));
}
