//! The host hardware capabilities relevant to inference: arch and OS strings, CPU model, core
//! count, total RAM, GPU presence, and a bool-per-feature SIMD card.
//!
//! Comptime. `detectSimd` reads only `builtin.cpu`, so it can evaluate at comptime.
//!
//! Role. The card is a boundary value, a plain struct with no hidden state. It feeds
//! `op_requires.isMetByHost`, which `info` uses to label eligible backends, and `info` prints it.
//!
//! Sources:
//!   - `builtin.cpu.features` for the SIMD flags. That is the BUILD target's
//!     set. On `-Dtarget=native`, the default, it matches the host. A binary
//!     copied elsewhere reports the build host's features, because no runtime
//!     cpuid runs.
//!   - `os/host_cpu_name.zig` for the CPU model name.
//!   - `os/host_memory.zig` for the total RAM, and `os/host_gpu.zig` for the GPU.
//!   - `std.Thread.getCpuCount` for online cores.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const host_cpu_name = @import("../../os/host_cpu_name.zig");
const host_gpu = @import("../../os/host_gpu.zig");
const host_memory = @import("../../os/host_memory.zig");
const os_process = @import("../../os/process.zig");

// -- types ------------------------------------------------------------------

/// Per-ISA SIMD feature flags (x86 + ARM), one bool each.
pub const SimdFeatures = struct {
    sse2: bool = false,
    sse4_1: bool = false,
    sse4_2: bool = false,
    avx: bool = false,
    avx2: bool = false,
    fma: bool = false,
    f16c: bool = false,
    avx512f: bool = false,
    avx512vnni: bool = false,
    avx_vnni: bool = false,
    neon: bool = false,
    dotprod: bool = false,
    fp16: bool = false,
    sve: bool = false,
};

pub const Gpu = host_gpu.Gpu;

/// Host hardware snapshot.
pub const HwCard = struct {
    arch: []const u8,
    os: []const u8,
    cpu_model: ?[]const u8,
    cores_total: usize,
    ram_total_bytes: ?u64,
    simd: SimdFeatures,
    gpu: Gpu,
};

// -- api --------------------------------------------------------------------

/// Build a HwCard. Allocations land on `arena` (cpu_model + an unknown gpu label).
pub fn detect(io: Io, arena: Allocator) HwCard {
    return .{
        .arch = @tagName(builtin.cpu.arch),
        .os = os_process.os_name,
        .cpu_model = host_cpu_name.read(io, arena),
        .cores_total = std.Thread.getCpuCount() catch 1,
        .ram_total_bytes = host_memory.totalBytes(io),
        .simd = detectSimd(),
        .gpu = host_gpu.detect(io, arena),
    };
}

/// Read the build target's SIMD feature flags. It reads only `builtin.cpu`,
/// so it evaluates at comptime and the simd-probe bakes the result into a
/// static string.
pub fn detectSimd() SimdFeatures {
    var s: SimdFeatures = .{};
    switch (builtin.cpu.arch) {
        .x86, .x86_64 => {
            const F = std.Target.x86.Feature;
            const set = builtin.cpu.features;
            s.sse2 = std.Target.x86.featureSetHas(set, F.sse2);
            s.sse4_1 = std.Target.x86.featureSetHas(set, F.sse4_1);
            s.sse4_2 = std.Target.x86.featureSetHas(set, F.sse4_2);
            s.avx = std.Target.x86.featureSetHas(set, F.avx);
            s.avx2 = std.Target.x86.featureSetHas(set, F.avx2);
            s.fma = std.Target.x86.featureSetHas(set, F.fma);
            s.f16c = std.Target.x86.featureSetHas(set, F.f16c);
            s.avx512f = std.Target.x86.featureSetHas(set, F.avx512f);
            s.avx512vnni = std.Target.x86.featureSetHas(set, F.avx512vnni);
            s.avx_vnni = std.Target.x86.featureSetHas(set, F.avxvnni);
        },
        .aarch64, .aarch64_be => {
            const F = std.Target.aarch64.Feature;
            const set = builtin.cpu.features;
            s.neon = std.Target.aarch64.featureSetHas(set, F.neon);
            s.dotprod = std.Target.aarch64.featureSetHas(set, F.dotprod);
            s.fp16 = std.Target.aarch64.featureSetHas(set, F.fullfp16);
            s.sve = std.Target.aarch64.featureSetHas(set, F.sve);
        },
        else => {},
    }

    return s;
}

// -- tests ------------------------------------------------------------------

test "detectSimd does not panic for the current target" {
    _ = detectSimd();
}
