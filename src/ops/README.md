# Ops

This directory holds the CPU kernels of the model: pure f32 ops that allocate nothing and keep no
state. The caller supplies the output buffer. A `Tensor` crosses the edge inward, and a
caller-owned `[]f32` crosses outward. `naive.zig` is the reference, and the SIMD backends are
shims that override only the hot kernels. The build fixes the backend, so the decode loop branches
on no backend.

## Files

| File | Holds |
| --- | --- |
| `naive.zig` | The scalar reference and the whole op surface. |
| `simd.zig` | The AVX2 and FMA backend, which overrides the f32 and Q8_0 linears. |
| `simd_pf.zig` | The register-blocked backend with prefetch, the default on AVX2 and FMA. |
| `attention_simd.zig` | The vector attention kernels that both SIMD backends share. |
| `exp_simd.zig` | The vector `expNeg`. |
| `attention_ref.zig` | The f64 reference and the checks for the CPU attention tests. |
| `gdn.zig` | The Gated DeltaNet reference of the Qwen layers. |

## How the pieces fit

`naive.zig` defines the op surface, and the two SIMD files repeat it. Each shim re-exports what it
does not override, so every backend offers the same names and a missing override is a fallback,
never a gap.

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| The op surface | `naive.zig` | The linears with `Batch` forms, the dequants, `embedLookup`, `rmsnorm`, `swiglu`, `residualAdd`, `ropeInplace`, and the attention part, merge, and row | the file itself | `simd`, `simd_pf`, `ops_vulkan/wrap`, `linear_dispatch`, the runners |
| `requires` | each backend | An `OpRequires`: the target and the tolerance against `naive` | each backend | `cli/info`, `tools/bench_util` |
| The attention kernels | `attention_simd.zig` | `plain` and `prefetched` sets of part, merge, and row | the file itself | `simd`, `simd_pf` |
| `expNeg` | `exp_simd.zig` | The vector exponential for inputs at or below 0 | the file itself | `attention_simd`, `runtime/sampler` |
| `RopeStyle` | `naive.zig` | `norm` or `neox` | `llama_runner` and `qwen_runner`, per family | `ropeInplace` |
| `gdn.step` | `gdn.zig` | The per-token recurrence over a state the runner owns | the file itself | `runtime/qwen_runner`, `runtime/parallel`, `dev/gpu_selftest` |
| `gdn.chunkScan` | `gdn.zig` | The chunked matrix form of the same recurrence | the file itself | `tools/simd_probe`, the tests |

A runner picks the backend once, at compile time. The `ops` constant of `runtime/llama_runner`,
`runtime/qwen_runner`, and `runtime/linear_dispatch` reads `build_options`, and `build.zig` sets
them from `-Dops-backend`:

```
  -Dops-backend --> build.resolveOpsBackend --> build_options
                                                     |
       use_vulkan_ops  -->  ops_vulkan/wrap.zig  (GPU kernels behind the same names)
       cpu_ops_simd_pf -->  ops/simd_pf.zig  --+
       cpu_ops_simd    -->  ops/simd.zig     --+--> ops/naive.zig  (re-exports the rest)
       otherwise       -->  ops/naive.zig
```

The shims share the attention kernels:

```
  simd.zig     --> attention_simd.plain      ---+--> exp_simd.expNeg
  simd_pf.zig  --> attention_simd.prefetched ---+
  both         --> naive.attentionMerge for the merge of the parts
```

`gdn.zig` stands outside the chain. It is not part of the swappable op set, and the runner passes it
the recurrent state and the convolution ring buffer. `gdn.step` is the form that prefill and decode
both run. `ops_vulkan/shaders/gdn_chunk_scan.comp` follows the `chunkScan` form, and its test pins
`chunkScan` against the sequential recurrence.

## Rules

- Every op asserts its shape preconditions. Under runtime safety every op also asserts that its
  output is finite. Two asserts per kernel is the floor.
- `naive` declares `.{ .arch = .any }` and is the reference. Every other backend must match its
  output. A backend that keeps the reduction order matches bit for bit. A backend that reorders
  declares a measured `.relative` tolerance in its `OpRequires`. `tools/bench_util.zig` checks the
  relative error per backend and fails the row on drift.
- `resolveOpsBackend` in `build.zig` does not read `requires`. `simdBackendSupportedBy` mirrors what
  the two SIMD backends declare, and the mirror and the declaration must agree.
- `simd_pf.linearF32Q80Batch` accumulates each (row, output) pair in an order that does not depend
  on the batch size. The batched output is therefore bit-identical to the single-row call. The tests
  assert it. The single-row entry point delegates to the batch body, so exactly one kernel body
  exists.
- `attention_simd` gives one head the same bits at any head block, thread count, or prefetch. A head
  width off the lane grid runs the naive kernel.
- No op allocates and no op holds state. The one allocator parameter in the directory belongs to
  `gdn.chunkScan`, which frees its scratch before it returns. No production path calls it.
- `gdn.chunkScan` keeps every decay ratio at or below 1, because each factor is a product of
  per-token decays in the half-open interval (0, 1]. The scan is therefore stable across a full
  chunk with no reciprocal blow-up.
- `naive.RopeStyle` must match the layout of the GGUF Q and K weights. The Llama family uses
  `norm`, and the Qwen family uses `neox`. The wrong style scrambles attention at every
  position and greedy decode degenerates into repetition.
