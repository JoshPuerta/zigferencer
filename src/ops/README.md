# src/ops

Pure, allocation-free f32 kernels. The caller supplies the output buffer, so no
op allocates and no op holds state. This is the CPU datapath of the split that
[docs/architecture.md](../../docs/architecture.md) describes. A `Tensor` crosses
the edge inward, a caller-owned `[]f32` crosses outward, and the inner slices are
raw `[]f32` the caller keeps. The build fixes the variant, so the decode loop
branches on no backend.

## Files

| File | Role |
| --- | --- |
| `naive.zig` | The scalar f32 reference that every other backend must match. It holds the full op set a forward pass needs. The set has embed lookup, RMSNorm, SwiGLU, residual add, and RoPE in both styles. It has decode attention over either KV element type: the part kernel, the ordered merge of the parts, and the serial row. It also has the Q8_0, Q4_0, Q4_1, and Q6_K dequant and linear kernels. |
| `simd.zig` | The `@Vector(8)` AVX2 and FMA variant. It overrides `linearF32` and `linearF32Q80` and their `Batch` forms, takes attention from `attention_simd.zig`, and re-exports every other op from `naive`. Width 8 matches the 256-bit register, and Zig lowers it to a NEON 4 plus 4 pair. |
| `simd_pf.zig` | A register-blocked f32 against Q8_0 microkernel with software prefetch, four output rows per inner block. It overrides `linearF32Q80` and `linearF32Q80Batch`, takes the prefetching attention kernels from `attention_simd.zig`, and re-exports the rest. It is the automatic default on a target with AVX2 and FMA. |
| `attention_simd.zig` | The vector part kernel of decode attention for both SIMD variants: `@Vector(8)` dot products with 8 register chains, F16C widening of f16 KV rows, a fixed lane tree, and the value sums in registers. It keeps the tiles, the fold, and the record layout of the naive part kernel. One head gets the same bits at any head block, thread count, or prefetch. A head width off the lane grid runs the naive kernel. Its tests and the naive tests share the f64 reference and the checks of `attention_ref.zig`, a test-support file. |
| `exp_simd.zig` | The `@Vector(8)` exponential `expNeg` for inputs at or below 0, at most 1.02 ulps from the f64 `exp`. Each lane depends on its own input alone. The vector attention fold and the full-vocab draw of `runtime/sampler.zig` share it. |
| `attention_ref.zig` | Test support for the CPU attention kernels: the f64 reference row, the scalar full-row oracle with its own softmax, the input fillers, and the checks that every CPU backend runs. Only test code imports it. |
| `gdn.zig` | The Gated-DeltaNet linear-attention reference for the Qwen3.5 layers: `step`, the allocation-free per-token recurrent form that the forward pass runs, and `chunkScan`, the chunked matrix form that the GPU prefill kernel follows. It sits outside the shim chain and is not part of the swappable op set. |

## How the pieces fit

`naive.zig` defines the whole op surface. `simd.zig` and `simd_pf.zig` are shims
with the same public signatures. Each re-exports the non-linear ops straight back
from `naive` and overrides only the linear kernels that dominate the bandwidth
budget. A runner therefore swaps one import and gets a complete set.

```
              re-export: the non-linear ops and the q4 and q6_k linears
              <---------------------------------------------------+
  runner ---> simd_pf.zig                                          |
   import      |  override: linearF32Q80, linearF32Q80Batch        |
               |  attention: attention_simd.zig, with prefetch     |
               +-----------------------------------------> naive.zig (the reference)
              simd.zig                                             ^
                  override: linearF32, linearF32Q80, both Batch ---+
                  attention: attention_simd.zig
```

Both shims take the attention part kernel and the serial row from
`attention_simd.zig`, and the merge from `naive`. What a shim leaves alone, the
Q4_0, Q4_1, and Q6_K linears, the embed lookup, the norms, and RoPE,
routes to `naive`. A missing override is therefore a correctness fallback, never
a gap. `resolveOpsBackend` in `build.zig` picks the import once. It does not read the `pub const requires: OpRequires` of a
variant. `simdBackendSupportedBy` in `build.zig` mirrors what the two SIMD
variants declare, and the mirror and the declaration must agree.

`gdn.zig` stands outside that chain. `runtime/qwen_runner.zig` is its only caller
in the forward pass, and `cli/gpu_selftest_cmd.zig` drives it as the reference of
the GPU check. The runner owns the recurrent `d x d` state and the convolution
ring buffer and passes both in. `step` is the per-token form that prefill and
decode both run. `chunkScan` is the chunked matrix form, and it is the one op
here that takes an allocator, for its per-chunk scratch. No production path calls
it. Its own test pins it against the sequential recurrence, and
`ops_vulkan/shaders/gdn_chunk_scan.comp` follows the same form.

## Rules

- Every op asserts its shape preconditions. Under runtime safety every op also
  asserts that its output is finite. Two asserts per kernel is the floor.
- `naive` declares `.{ .arch = .any }` and is the reference. Every other variant
  must match its output. A variant that keeps the reduction order matches bit for
  bit. A variant that reorders declares a measured `.relative` tolerance in its
  `OpRequires`, and `tools/bench.zig` prints the relative error per variant and
  fails the row on drift.
- `simd_pf.linearF32Q80Batch` accumulates each (row, output) pair in an order
  that does not depend on the batch size, so the batched output is bit-identical
  to the single-row call. The tests assert it. The single-row entry point
  delegates to the batch body, so exactly one kernel body exists.
- No op allocates and no op holds state. The one allocator parameter in the
  directory belongs to `gdn.chunkScan`, which frees its scratch before it
  returns.
- `gdn.chunkScan` keeps every decay ratio at or below 1, because each factor is a
  product of per-token decays in the half-open interval (0, 1]. The scan is
  therefore stable across a full chunk with no reciprocal blow-up.
- `naive.RopeStyle` must match how the conversion laid out the GGUF Q and K
  weights: `norm` for the Llama family and `neox` for the Qwen family. The wrong
  style scrambles attention at every position and greedy decode degenerates into
  repetition.

## Where to start

1. `naive.zig`: every op in scalar form, and the output every variant must reproduce.
2. `simd.zig`: the shim shape and a direct vectorization.
3. `simd_pf.zig`: the register-blocked, prefetching, batch-resident kernel that is the CPU default.
4. `gdn.zig`: read it beside `runtime/qwen_runner.zig`, which owns the state this op transforms.
