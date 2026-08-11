//! What a single op-backend variant needs from the target. Each
//! `ops/<variant>.zig` exposes `pub const requires: OpRequires = ...;`.
//! `isMetByHost` scores it against the host card, so `info` reports what the
//! running host supports even for a `-Dcpu=baseline` binary. The build does not
//! read this value. `simdBackendSupportedBy` in `build.zig` mirrors what the two
//! SIMD variants declare, and it rejects a bad target and backend pair at build
//! time rather than as a runtime SIGILL. The mirror and the declaration must
//! agree, and a change to one needs the same change to the other. Deterministic
//! over dynamic: no SIMD probing runs in the hot path.

const std = @import("std");
const builtin = @import("builtin");
const hwinfo = @import("hwinfo.zig");

// -- types ------------------------------------------------------------------

/// Target CPU architecture family (`.any` = runs anywhere). Only the
/// families a variant declares exist. Add a value when a variant needs it.
pub const Arch = enum {
    any,
    x86_64,
};

/// Dispatch path the variant runs on. Orthogonal to `Arch`: a Vulkan compute
/// variant does not care which CPU arch the host has. The binary still must be
/// built to drive the Vulkan loader (`-Druntime=vulkan`). The CPU variants
/// leave this at its default.
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

/// The numeric bound a variant promises to honor against the `naive` scalar
/// reference. The bench and test gate consults it, so a variant stays green as
/// long as it satisfies what it claims.
///
/// - `.bit_exact`: the f32 output bytes match `naive`'s bytes exactly.
///   Use it for variants that preserve scalar reduction order and disable
///   FMA contraction (GLSL `precise`, no `@fma`-style intrinsics).
///   Hash equality is the gate.
/// - `.relative = eps`: max per-element relative error <= eps. Use it for
///   variants that reorder reductions (CPU SIMD, GPU tiled) or accept
///   FMA contraction. The bench computes max_abs / max_rel and gates
///   on this bound.
///
/// A variant does not have to claim more than it can prove. Loose is fine.
/// LYING is not: a variant that declares `.bit_exact` and then drifts fails
/// the gate.
pub const Tolerance = union(enum) {
    bit_exact,
    relative: f32,
};

pub const OpRequires = struct {
    /// Target CPU architecture. `.any` means "works on every arch".
    arch: Arch = .x86_64,
    /// Dispatch path. `.cpu` is the default and matches every variant under
    /// `src/ops/`. `.vulkan` marks the variant as a Vulkan compute kernel,
    /// only eligible when the binary was built with `-Druntime=vulkan`.
    runtime: Runtime = .cpu,
    /// Minimum SIMD level. `.none` means scalar-OK.
    min_simd: SimdLevel = .none,
    /// True when the variant uses FMA (x86) or fused multiply-add (arm).
    needs_fma: bool = false,
    /// Numeric tolerance against the naive variant. The default is
    /// deliberately permissive (1e-3 relative error): SIMD reordering on CPU
    /// and FMA on GPU both pass it. Bugs that swap signs, mis-index a row, or
    /// break a scale decode do not. A tight variant opts into `.bit_exact`.
    tolerance: Tolerance = .{ .relative = 1e-3 },
};

// -- api --------------------------------------------------------------------

/// Every selectable `-Dops-backend` value, in display order. The one list that
/// the UI surfaces enumerating backends read from (the run command's legend,
/// the support matrix of `info`). `build.zig` resolves the same names at build time.
pub const ops_backend_names = [_][]const u8{ "naive", "simd", "simd_pf", "vulkan", "vulkan_fused" };

/// Combine an observed max relative error and a hash-match bit against the
/// variant's declared tolerance. Returns true only when the variant honors
/// that tolerance. The bench prints both numbers either way, so a hash-equal
/// variant under a `.relative` claim is visible as "tighter than promised."
pub fn toleranceSatisfied(tol: Tolerance, max_rel_err: f32, hash_match: bool) bool {
    return switch (tol) {
        .bit_exact => hash_match,
        .relative => |eps| max_rel_err <= eps,
    };
}

/// True when this variant's `runtime` dimension matches the binary's selected
/// host runtime. Orthogonal to `isMetBy` / `isMetByHost`, which only score the
/// arch/SIMD/FMA axis. `info` combines the two.
pub fn runtimeMatches(req: OpRequires, host: Runtime) bool {
    return req.runtime == host;
}

/// Build-time eligibility check using a resolved `std.Target.Cpu`.
/// `tools/bench_matmul.zig` calls this to filter its rows, and `build.zig` keeps its own mirror.
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
    // On an aarch64 host this is false. On x86_64 with AVX2 it is true.
    // The test cannot assume the host has AVX2, only that the check agrees
    // with the same check inlined.
    const host_arch_ok = builtin.cpu.arch == .x86_64;
    if (!host_arch_ok) try std.testing.expect(!isMetBy(req, builtin.cpu));
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
    // Even at zero relative error, a hash mismatch means a drift somewhere
    // (a sign bit on a NaN, for example). That still fails under bit_exact.
    try std.testing.expect(!toleranceSatisfied(.bit_exact, 0.0, false));
}

test "tolerance: relative gate ignores hash" {
    const tol: Tolerance = .{ .relative = 1e-3 };
    try std.testing.expect(toleranceSatisfied(tol, 5e-4, false));
    try std.testing.expect(toleranceSatisfied(tol, 5e-4, true));
    try std.testing.expect(toleranceSatisfied(tol, 1e-3, false));
    try std.testing.expect(!toleranceSatisfied(tol, 2e-3, true));
}
