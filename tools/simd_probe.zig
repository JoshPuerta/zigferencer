//! Symbol-export probe for `tools/simd-report.sh`. In a release build the op
//! kernels inline straight into the forward pass, so neither the binary nor its
//! assembly keeps a per-op symbol to inspect. This artifact exists to give each
//! kernel a standalone, named symbol. It takes the address of every op across
//! all three CPU backends, and parks each address behind `doNotOptimizeAway`.
//! The compiler then emits the function body without a call site, and so without
//! an inline.
//!
//! The report reads the assembly of the compiler, not a disassembly of the
//! binary. `zig build simd-probe` emits the `.s` of this module for the build
//! target, so the tool works under cross-compilation (`-Dtarget=...`,
//! `-Dcpu=...`) for any architecture Zig can target, and needs no multi-arch
//! objdump. A foreign target never runs the probe, so the probe cannot print the
//! SIMD ceiling of that target at run time. It bakes the ceiling into a static
//! `##SIMD-PROBE##` marker string instead, which the report greps straight out
//! of the assembly.
//!
//! The op list below is the source of truth for which kernels the report
//! verifies. To add a kernel, take its address here. The simd and simd_pf
//! backends hand-write the matmul kernels and the attention part kernel.
//! linearF32Q80, the hot quantized matmul, and the attention part kernel exist
//! across all three backends, and linearF32, the dense f32 matmul, in simd
//! alone. The list names those once per specializing backend. Every other op is
//! shared naive code that each backend re-exports unchanged. Its width is
//! whatever the compiler auto-vectorizes for the target, which is the same
//! width whichever backend ships.

const std = @import("std");
const builtin = @import("builtin");
const z = @import("zigferencer");

// -- constants --------------------------------------------------------------

const probed_ops = .{
    // Backend-specialized: the kernels that simd and simd_pf hand-write. The
    // report keeps the variants of one kernel next to each other, so the widths
    // compare at a glance.
    &z.ops.linearF32Q80, // naive: scalar reference
    &z.ops_simd.linearF32Q80, // simd: @Vector(8)
    &z.ops_simd_pf.linearF32Q80, // simd_pf: register-blocked + prefetch
    &z.ops.linearF32, // naive: scalar reference
    &z.ops_simd.linearF32, // simd: @Vector(8)
    // The part kernel is generic over the KV element type. A wrapper per type
    // and backend pins one concrete instance each.
    &attentionPartF32, // naive: scalar reference
    &attentionPartF32Simd, // simd: @Vector(8)
    &attentionPartF32SimdPf, // simd_pf: @Vector(8) + prefetch
    &attentionPartF16, // naive: scalar reference
    &attentionPartF16Simd, // simd: @Vector(8) + F16C
    &attentionPartF16SimdPf, // simd_pf: @Vector(8) + F16C + prefetch
    // Shared: defined once in naive, re-exported by every backend.
    &z.ops.linearF32Q40,
    &z.ops.rmsnorm,
    &z.ops.swiglu,
    &z.ops.residualAdd,
    &z.ops.ropeInplace,
    &z.ops.attentionMerge,
    &z.ops.dequantQ80,
    &z.ops.dequantQ40,
    &z.ops.dequantQ6K,
    &z.ops.embedLookup,
    &z.gdn.step,
    &z.gdn.chunkScan,
};

/// The SIMD ceiling of the build target, baked into the assembly so the report
/// can show it without running the probe, which can be cross-compiled. One line,
/// with comma-separated feature lists, so the script parses it with a field
/// split:
///   ##SIMD-PROBE## arch=x86_64 cpu=znver3 available=avx2,fma absent=avx512f
const caps_marker = capsLine();

// -- types ------------------------------------------------------------------

/// The arguments of one part kernel call. A wrapper takes them as one value,
/// so each concrete instance stays one line.
fn PartArgs(comptime E: type) type {
    return struct {
        q: []const f32,
        h_lo: usize,
        h_hi: usize,
        n_heads: usize,
        head_dim: usize,
        k_cache: []const E,
        v_cache: []const E,
        n_kv_heads: usize,
        t_lo: usize,
        t_hi: usize,
        recs: []f32,
        rec_stride: usize,
    };
}

// -- api --------------------------------------------------------------------

pub fn main() void {
    // Every probed op reaches the assembly as a standalone symbol, and the
    // capability marker stays in it. The body runs none of them.
    inline for (probed_ops) |op_ptr| {
        std.mem.doNotOptimizeAway(op_ptr);
    }

    std.mem.doNotOptimizeAway(caps_marker.ptr);
}

// -- helpers ----------------------------------------------------------------

/// Inline the part kernel of `Kernels` into the calling wrapper, so the body
/// lands under the name of the wrapper.
inline fn partCall(comptime Kernels: type, comptime E: type, a: PartArgs(E)) void {
    @call(
        .always_inline,
        Kernels.attentionPartTyped,
        .{ E, a.q, a.h_lo, a.h_hi, a.n_heads, a.head_dim, a.k_cache, a.v_cache, a.n_kv_heads, a.t_lo, a.t_hi, a.recs, a.rec_stride },
    );
}

fn attentionPartF32(a: PartArgs(f32)) void {
    partCall(z.ops, f32, a);
}

fn attentionPartF32Simd(a: PartArgs(f32)) void {
    partCall(z.ops_simd, f32, a);
}

fn attentionPartF32SimdPf(a: PartArgs(f32)) void {
    partCall(z.ops_simd_pf, f32, a);
}

fn attentionPartF16(a: PartArgs(f16)) void {
    partCall(z.ops, f16, a);
}

fn attentionPartF16Simd(a: PartArgs(f16)) void {
    partCall(z.ops_simd, f16, a);
}

fn attentionPartF16SimdPf(a: PartArgs(f16)) void {
    partCall(z.ops_simd_pf, f16, a);
}

fn capsLine() []const u8 {
    @setEvalBranchQuota(10_000);
    const simd = z.hwinfo.detectSimd();
    // The list holds the flags of the build architecture alone, so an x86 report
    // never lists NEON as absent. `absent` is the headroom the kernels leave.
    const names = switch (builtin.cpu.arch) {
        .x86, .x86_64 => .{
            "sse2",
            "sse4_2",
            "avx",
            "avx2",
            "fma",
            "f16c",
            "avx512f",
            "avx512vnni",
            "avx_vnni",
        },
        .aarch64, .aarch64_be => .{ "neon", "dotprod", "fp16", "sve" },
        else => .{},
    };

    var available: []const u8 = "";
    var absent: []const u8 = "";
    inline for (names) |n| {
        if (@field(simd, n)) {
            available = available ++ (if (available.len == 0) "" else ",") ++ n;
        } else {
            absent = absent ++ (if (absent.len == 0) "" else ",") ++ n;
        }
    }

    return "##SIMD-PROBE## arch=" ++ @tagName(builtin.cpu.arch) ++
        " cpu=" ++ builtin.cpu.model.name ++
        " available=" ++ available ++ " absent=" ++ absent;
}
