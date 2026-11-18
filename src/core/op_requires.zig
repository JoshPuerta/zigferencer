//! What a single op-backend variant needs from the target. Each `ops/<variant>.zig` exposes
//! `pub const requires: OpRequires = ...;`.
//!
//! Host check. `isMetByHost` scores a variant against the host card, so `info` reports what the
//! running host supports even for a `-Dcpu=baseline` binary. No SIMD probing runs in the hot path.
//!
//! Build mirror. The build does not read this value. `simdBackendSupportedBy` in `build.zig`
//! mirrors what the two SIMD variants declare, and it rejects a bad target and backend pair at
//! build time rather than as a runtime SIGILL. The mirror and the declaration must agree, and a
//! change to one needs the same change to the other.

const std = @import("std");
const builtin = @import("builtin");
const hwinfo = @import("utils/hwinfo.zig");

// -- constants --------------------------------------------------------------

/// Every selectable `-Dops-backend` value, in display order. The run command legend and the
/// support matrix of `info` read this list. `build.zig` resolves the same names.
pub const ops_backend_names = [_][]const u8{ "naive", "simd", "simd_pf", "vulkan", "vulkan_fused" };

// -- types ------------------------------------------------------------------

/// Target CPU architecture family (`.any` = runs anywhere). Only the
/// families a variant declares exist. Add a value when a variant needs it.
pub const Arch = enum {
    any,
    x86_64,
};

/// Dispatch path the variant runs on. Orthogonal to `Arch`: a Vulkan variant does not care which
/// CPU arch the host has, but the binary must build with `-Druntime=vulkan`.
pub const Runtime = enum {
    cpu,
    vulkan,
};

/// Strongest SIMD instruction set the variant needs. Only the levels a
/// variant declares exist. Add a level when a variant needs it.
pub const SimdLevel = enum {
    none,
    avx2,
};

/// The numeric bound a variant promises to honor against the `naive` scalar reference.
/// The bench and test gate consults it. A variant that claims `.bit_exact` and then drifts
/// fails the gate, so a loose claim is fine and a false claim is not.
pub const Tolerance = union(enum) {
    /// The f32 output bytes match `naive` exactly. Needs scalar reduction order and no FMA.
    bit_exact,
    /// Maximum per-element relative error. Fits variants that reorder reductions or use FMA.
    relative: f32,
};

pub const OpRequires = struct {
    /// Target CPU architecture. `.any` means "works on every arch".
    arch: Arch = .x86_64,
    /// Dispatch path. `.vulkan` is eligible only when the binary builds with `-Druntime=vulkan`.
    runtime: Runtime = .cpu,
    /// Minimum SIMD level. `.none` means scalar-OK.
    min_simd: SimdLevel = .none,
    /// True when the variant uses FMA (x86) or fused multiply-add (ARM).
    needs_fma: bool = false,
    /// Tolerance against `naive`. The default of 1e-3 relative error admits SIMD reordering and
    /// FMA, and it rejects sign swaps, output mis-indexing, and scale decode bugs.
    tolerance: Tolerance = .{ .relative = 1e-3 },
};

// -- api --------------------------------------------------------------------

/// Check an observed max relative error and a hash-match bit against the declared tolerance.
/// Returns true only when the variant honors it. The bench prints both numbers either way,
/// so a hash-equal variant under a `.relative` claim reads as "tighter than promised."
pub fn toleranceSatisfied(tol: Tolerance, max_rel_err: f32, hash_match: bool) bool {
    return switch (tol) {
        .bit_exact => hash_match,
        .relative => |eps| max_rel_err <= eps,
    };
}

/// True when this variant's `runtime` matches the binary's selected host runtime. `info` combines
/// it with `isMetBy` and `isMetByHost`, which score only arch, SIMD, and FMA.
pub fn runtimeMatches(req: OpRequires, host: Runtime) bool {
    return req.runtime == host;
}

/// Build-time eligibility check using a resolved `std.Target.Cpu`.
/// `tools/bench_matmul.zig` calls this to filter its lines, and `build.zig` keeps its own mirror.
pub fn isMetBy(req: OpRequires, cpu: std.Target.Cpu) bool {
    const arch_ok = switch (req.arch) {
        .any => true,
        .x86_64 => cpu.arch == .x86_64,
    };

    if (!arch_ok) return false;
    if (!simdLevelMetBy(req.min_simd, cpu)) return false;
    if (req.needs_fma and !fmaAvailableBy(cpu)) return false;

    return true;
}

/// Runtime eligibility check using a detected host card.
/// `info` calls this to show which variants the running host satisfies.
pub fn isMetByHost(req: OpRequires, card: hwinfo.HwCard) bool {
    const arch_ok = switch (req.arch) {
        .any => true,
        .x86_64 => std.mem.eql(u8, card.arch, "x86_64"),
    };

    if (!arch_ok) return false;
    if (!simdLevelMetByHost(req.min_simd, card.simd)) return false;
    if (req.needs_fma and !card.simd.fma) return false;

    return true;
}

// -- helpers ----------------------------------------------------------------

fn simdLevelMetBy(min: SimdLevel, cpu: std.Target.Cpu) bool {
    return switch (min) {
        .none => true,
        .avx2 => (cpu.arch == .x86_64 or cpu.arch == .x86) and
            std.Target.x86.featureSetHas(cpu.features, .avx2),
    };
}

fn fmaAvailableBy(cpu: std.Target.Cpu) bool {
    return switch (cpu.arch) {
        .x86_64, .x86 => std.Target.x86.featureSetHas(cpu.features, .fma),
        else => false,
    };
}

fn simdLevelMetByHost(min: SimdLevel, s: hwinfo.SimdFeatures) bool {
    return switch (min) {
        .none => true,
        .avx2 => s.avx2,
    };
}

// -- tests ------------------------------------------------------------------

test "default OpRequires (any arch, no SIMD) is met by anything" {
    const req: OpRequires = .{ .arch = .any };
    try std.testing.expect(isMetBy(req, builtin.cpu));
}

test "avx2 requirement is filtered by arch" {
    const req: OpRequires = .{ .arch = .x86_64, .min_simd = .avx2, .needs_fma = true };
    // The test cannot assume the host has AVX2, so it checks the arch axis alone.
    const host_arch_ok = builtin.cpu.arch == .x86_64;
    if (!host_arch_ok) {
        try std.testing.expect(!isMetBy(req, builtin.cpu));
    }
}

test "runtime default is cpu and matches host=cpu" {
    const req: OpRequires = .{ .arch = .any };
    try std.testing.expectEqual(Runtime.cpu, req.runtime);
    try std.testing.expect(runtimeMatches(req, .cpu));
    try std.testing.expect(!runtimeMatches(req, .vulkan));
}

test "vulkan runtime is orthogonal to arch checks" {
    const req: OpRequires = .{ .arch = .any, .runtime = .vulkan };
    // arch=.any => isMetBy is trivially true on every host
    try std.testing.expect(isMetBy(req, builtin.cpu));
    // but the runtime axis only matches the vulkan host
    try std.testing.expect(!runtimeMatches(req, .cpu));
    try std.testing.expect(runtimeMatches(req, .vulkan));
}

test "tolerance: bit_exact requires hash match" {
    try std.testing.expect(toleranceSatisfied(.bit_exact, 0.0, true));
    try std.testing.expect(!toleranceSatisfied(.bit_exact, 0.0, false));
    // A hash mismatch is a drift even at zero relative error, such as a NaN sign bit.
    try std.testing.expect(!toleranceSatisfied(.bit_exact, 0.0, false));
}

test "tolerance: relative gate ignores hash" {
    const tol: Tolerance = .{ .relative = 1e-3 };
    try std.testing.expect(toleranceSatisfied(tol, 5e-4, false));
    try std.testing.expect(toleranceSatisfied(tol, 5e-4, true));
    try std.testing.expect(toleranceSatisfied(tol, 1e-3, false));
    try std.testing.expect(!toleranceSatisfied(tol, 2e-3, true));
}
