//! Symbol-export probe for `tools/simd-report.sh`. This artifact gives each op kernel a
//! standalone, named symbol.
//!
//! Comptime. `PartArgs` and `partCall` take the element type and the kernel set, so each probe
//! symbol is one short wrapper.
//!
//! Why. In a release build the op kernels inline straight into the forward pass, so neither the
//! binary nor its assembly keeps a per-op symbol to inspect. The probe takes the address of
//! every op across all three CPU backends, and parks each address behind `doNotOptimizeAway`.
//! The compiler then emits the function body without a call site, and so without an inline.
//!
//! Assembly. The report reads the assembly of the compiler, not a disassembly of the binary.
//! `zig build simd-probe` emits the `.s` of this module for the build target, so the tool works
//! under cross-compilation (`-Dtarget=...`, `-Dcpu=...`) for any architecture Zig can target,
//! and needs no multi-arch objdump. A foreign target never runs the probe, so the probe cannot
//! print the SIMD ceiling of that target at run time. It bakes the ceiling into a static
//! `##SIMD-PROBE##` marker string instead, which the report greps straight out of the assembly.
//!
//! Op list. The list below is the source of truth for which kernels the report verifies. To add
//! a kernel, take its address here. The simd and simd_pf backends hand-write the matmul kernels
//! and the attention part kernel. linearF32Q80, the hot quantized matmul, and the attention part
//! kernel exist across all three backends, and linearF32, the dense f32 matmul, in simd alone.
//! The list names those once per specializing backend. The Q8_0 linears of simd and simd_pf share
//! the source of `linear_simd`, so a wrapper per backend gives each its own symbol.
//!
//! Shared ops. Every other op is shared naive code that each backend re-exports unchanged. Its
//! width is whatever the compiler auto-vectorizes for the target, which is the same width
//! whichever backend ships.

const std = @import("std");
const builtin = @import("builtin");
const zigferencer = @import("zigferencer");
const Tensor = zigferencer.tensor.Tensor;

// -- constants --------------------------------------------------------------

const probed_ops = .{
    // Backend-specialized: the kernels that simd and simd_pf hand-write. The report keeps the
    // variants of one kernel together.
    &zigferencer.ops.linearF32Q80, // naive: scalar reference
    &linearF32Q80Simd, // simd: @Vector(8)
    &linearF32Q80SimdPf, // simd_pf: register-blocked + prefetch
    &linearF32Q80BatchSimd, // simd: batch panel path
    &linearF32Q80BatchSimdPf, // simd_pf: batch panel path
    &zigferencer.ops.linearF32, // naive: scalar reference
    &zigferencer.ops_simd.linearF32, // simd: @Vector(8)
    // The part kernel is generic over the KV element type. A wrapper per type
    // and backend pins one concrete instance each.
    &attentionPartF32, // naive: scalar reference
    &attentionPartF32Simd, // simd: @Vector(8)
    &attentionPartF32SimdPf, // simd_pf: @Vector(8) + prefetch
    &attentionPartF16, // naive: scalar reference
    &attentionPartF16Simd, // simd: @Vector(8) + F16C
    &attentionPartF16SimdPf, // simd_pf: @Vector(8) + F16C + prefetch
    // Shared: defined once in naive, re-exported by every backend.
    &zigferencer.ops.linearF32Q40,
    &zigferencer.ops.rmsnorm,
    &zigferencer.ops.swiglu,
    &zigferencer.ops.residualAdd,
    &zigferencer.ops_simd_pf.residualAdd,
    &zigferencer.ops.ropeInplace,
    &zigferencer.ops.ropeHeads,
    &zigferencer.ops.attentionMerge,
    &zigferencer.ops.dequantQ80,
    &zigferencer.ops.dequantQ40,
    &zigferencer.ops.dequantQ6K,
    &zigferencer.ops.embedLookup,
    &zigferencer.gdn.step,
    &zigferencer.gdn.chunkScan,
};

/// The SIMD ceiling of the build target, baked into the assembly so a cross-compiled report can
/// show it. One line with comma-separated lists, which the script splits by field:
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
        .{
            E,
            a.q,
            a.h_lo,
            a.h_hi,
            a.n_heads,
            a.head_dim,
            a.k_cache,
            a.v_cache,
            a.n_kv_heads,
            a.t_lo,
            a.t_hi,
            a.recs,
            a.rec_stride,
        },
    );
}

fn attentionPartF32(a: PartArgs(f32)) void {
    partCall(zigferencer.ops, f32, a);
}

fn attentionPartF32Simd(a: PartArgs(f32)) void {
    partCall(zigferencer.ops_simd, f32, a);
}

fn attentionPartF32SimdPf(a: PartArgs(f32)) void {
    partCall(zigferencer.ops_simd_pf, f32, a);
}

fn attentionPartF16(a: PartArgs(f16)) void {
    partCall(zigferencer.ops, f16, a);
}

fn attentionPartF16Simd(a: PartArgs(f16)) void {
    partCall(zigferencer.ops_simd, f16, a);
}

fn attentionPartF16SimdPf(a: PartArgs(f16)) void {
    partCall(zigferencer.ops_simd_pf, f16, a);
}

fn linearF32Q80Simd(input: []const f32, weight: Tensor, out: []f32) void {
    @call(.always_inline, zigferencer.ops_simd.linearF32Q80, .{ input, weight, out });
}

fn linearF32Q80SimdPf(input: []const f32, weight: Tensor, out: []f32) void {
    @call(.always_inline, zigferencer.ops_simd_pf.linearF32Q80, .{ input, weight, out });
}

fn linearF32Q80BatchSimd(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    @call(
        .always_inline,
        zigferencer.ops_simd.linearF32Q80Batch,
        .{ inputs, weight, outs, batch, out_stride },
    );
}

fn linearF32Q80BatchSimdPf(
    inputs: []const f32,
    weight: Tensor,
    outs: []f32,
    batch: usize,
    out_stride: usize,
) void {
    @call(
        .always_inline,
        zigferencer.ops_simd_pf.linearF32Q80Batch,
        .{ inputs, weight, outs, batch, out_stride },
    );
}

fn capsLine() []const u8 {
    @setEvalBranchQuota(10_000);
    const simd = zigferencer.hwinfo.detectSimd();
    // List the flags of the build architecture alone, so an x86 report never lists NEON as absent.
    // `absent` is the headroom the kernels leave.
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
