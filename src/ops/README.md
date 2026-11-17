# Ops

This directory holds the CPU kernels of the model: pure f32 ops that allocate nothing and keep no
state. The caller supplies the output buffer. A `Tensor` crosses the edge inward, and a
caller-owned `[]f32` crosses outward. `naive.zig` is the reference, and the SIMD backends are
files that override only the hot kernels. The build fixes the backend, so the decode loop branches
on no backend.

## Files

| File | Holds |
| --- | --- |
| `naive.zig` | The scalar reference and the whole op surface. |
| `simd.zig` | The AVX2 and FMA backend without prefetch, the control for `simd_pf`. |
| `simd_pf.zig` | The AVX2 and FMA backend with prefetch, the auto default. |
| `linear_simd.zig` | The Q8_0 linear kernels that both SIMD backends share. |
| `attention_simd.zig` | The vector attention kernels that both SIMD backends share. |
| `elementwise_simd.zig` | The vector `residualAdd`, `rmsnorm`, `swiglu`, and `sigmoidGate` of both SIMD backends. |
| `exp_simd.zig` | The vector `expNeg` and `sigmoid`. |
| `lane_tree_simd.zig` | The sum of the 8 lanes of a vector in one fixed order (`laneSum`, `transposeAdd`), and the `dot` and `sumSquares` that use it. |
| `attention_ref.zig` | The f64 reference and the checks for the CPU attention tests. |
| `gdn.zig` | The control flow of the Gated DeltaNet step of the Qwen layers, and its scalar reference. |
| `gdn_simd.zig` | The lane arithmetic of the GDN step that both SIMD backends share. |

## How the pieces fit

`naive.zig` defines the op surface, and the two SIMD files repeat it. Each SIMD file re-exports
what it does not override. So every backend offers the same names, and a missing override is a
fallback, never a gap.

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| The op surface | `naive.zig` | The linears with `Batch` forms, the dequants, `embedLookup`, `rmsnorm`, `swiglu`, `sigmoidGate`, `residualAdd`, `ropeInplace`, `ropeHeads`, the attention part, merge, and row, `swiglu_flops_per_element`, and `gdnStepKeyHead` | the file itself | `simd`, `simd_pf`, `ops_vulkan/wrap`, `linear_dispatch`, the runners |
| `requires` | each backend | An `OpRequires`: the target and the tolerance against `naive` | each backend | `cli/info`, `tools/bench_util` |
| The Q8_0 linears | `linear_simd.zig` | `plain` and `prefetched` sets of `linearF32Q80` and `linearF32Q80Batch` | the file itself | `simd`, `simd_pf` |
| The attention kernels | `attention_simd.zig` | `plain` and `prefetched` sets of part, merge, and row | the file itself | `simd`, `simd_pf` |
| `expNeg` | `exp_simd.zig` | The vector exponential for inputs at or below 0 | the file itself | `attention_simd`, `runtime/sampler` |
| `sigmoid` | `exp_simd.zig` | The lane-wise sigmoid on `expNeg` | the file itself | `elementwise_simd`, `gdn_simd`, `tools/simd_probe`, the tests |
| `transposeAdd`, `laneSum` | `lane_tree_simd.zig` | The sum of the 8 lanes of a vector. Every SIMD kernel of this directory adds the lanes in this one order, the lane tree, so the sum has the same bits in each of them | the file itself | `attention_simd`, `linear_simd`, `dot` |
| `sumSquares` | `lane_tree_simd.zig` | The sum of squares in one fixed order: 4 accumulators, then the lane tree | the file itself | `elementwise_simd.rmsnorm`, `gdn_simd`, `tools/simd_probe`, the tests |
| `dot` | `lane_tree_simd.zig` | The dot product in the order of `sumSquares`, which calls it | the file itself | `sumSquares`, `gdn_simd` |
| `RopeStyle` | `naive.zig` | `norm` or `neox` | `llama_layer` and `qwen_layer`, per family | `ropeInplace`, `ropeHeads` |
| `RopeTable` | `naive.zig` | The base angle of each rotated pair and the optional frequency divisors | `runtime/cpu_runner`, with `fillRopeThetas` | `ropeHeads` |
| `residualAdd` | `elementwise_simd.zig` | The vector add, bit-identical to the naive loop | the file itself | `simd`, `simd_pf` |
| `rmsnorm`, `swiglu`, `sigmoidGate` | `elementwise_simd.zig` | The vector forms of the naive ops, with `sumSquares` and the lane-wise sigmoid, and a padded tail | the file itself | `simd`, `simd_pf` |
| `swiglu_flops_per_element` | each CPU backend | The FLOP weight of one `swiglu` element of that backend | `naive`, `elementwise_simd` | `runtime/parallel` |
| `gdn.Kernels` | `gdn.zig` | The control flow of one step over an arithmetic: the head mapping, the q scale, and the order of the parts | the file itself | `gdn.reference`, `gdn_simd` |
| `gdn.reference`, `gdn.step` | `gdn.zig` | `gdn.Kernels` over the scalar helpers, and its serial per-token recurrence over a state the runner owns | the file itself | `dev/gpu_selftest`, `tools/bench_gdn`, `tools/simd_probe`, the tests |
| `gdn.stepKeyHead` | `gdn.zig` | The same recurrence for one key head, so disjoint key heads can run at once | the file itself | `naive` |
| `gdn_simd.kernels` | `gdn_simd.zig` | `gdn.Kernels` over the lane arithmetic | the file itself | `simd`, `simd_pf`, `tools/bench_gdn`, `tools/simd_probe` |
| `gdnStepKeyHead` | each backend | The key-head unit of the GDN step | `gdn.zig` or `gdn_simd.zig` | `runtime/parallel` |
| `gdn.chunkScan` | `gdn.zig` | The chunked matrix form of the same recurrence | the file itself | `tools/simd_probe`, the tests |

A runner picks the backend once, at compile time. The `ops` constant of `runtime/cpu_runner`,
`runtime/cpu_layer_kernels`, and `runtime/linear_dispatch` reads `build_options`, and `build.zig`
sets them from `-Dops-backend`:

```
  -Dops-backend --> build.resolveOpsBackend --> build_options
                                                     |
       use_vulkan_ops  -->  ops_vulkan/wrap.zig  (GPU kernels behind the same names)
       cpu_ops_simd_pf -->  ops/simd_pf.zig  --+
       cpu_ops_simd    -->  ops/simd.zig     --+--> ops/naive.zig  (re-exports the rest)
       otherwise       -->  ops/naive.zig
```

The SIMD files share the Q8_0 linears, the attention kernels, the elementwise ops, and the GDN
step:

```
  simd.zig     --> linear_simd.plain         ---+--> lane_tree_simd.transposeAdd, laneSum
  simd_pf.zig  --> linear_simd.prefetched    ---+
  simd.zig     --> attention_simd.plain      ---+--> exp_simd.expNeg,
  simd_pf.zig  --> attention_simd.prefetched ---+    lane_tree_simd.transposeAdd, laneSum
  both         --> naive.attentionMerge for the merge of the parts
  both         --> elementwise_simd: residualAdd, rmsnorm, swiglu, sigmoidGate
                       --> lane_tree_simd.sumSquares, exp_simd.sigmoid
  both         --> gdn_simd.kernels.stepKeyHead
                       --> gdn.Kernels, gdn.convStep, lane_tree_simd.dot, sumSquares,
                           exp_simd.sigmoid
```

The GDN step joins the op set through one op, `gdnStepKeyHead`. `naive` exports
`gdn.stepKeyHead`, and the SIMD backends export `gdn_simd.kernels.stepKeyHead`. The runner passes
it the recurrent state and the convolution ring buffer, and runs it per key head. `gdn.step` is the
serial reference. `ops_vulkan/shaders/gdn_chunk_scan.comp` follows the
`chunkScan` form, and its test pins `chunkScan` against the sequential recurrence.

## Rules

1. **Asserts.** Every op asserts its shape preconditions. Under runtime safety it also asserts
   that its output is finite. Two asserts per kernel is the floor.
2. **No state.** No op allocates and no op holds state. `gdn.chunkScan` is the one op that
   allocates. It takes an allocator for its scratch and frees the scratch before it returns. No
   production path calls it.
3. **`naive` is the reference.** It declares `.{ .arch = .any }`, and every other backend must
   match its output. A backend that keeps the order of the adds matches bit for bit. A backend
   that changes the order declares a measured `.relative` tolerance in its `OpRequires`.
   `tools/bench_util.zig` checks the relative error per backend and prints `FAIL` on drift.
4. **One input, one result.** In the ops of `simd` and `simd_pf`, the bits of a value depend on
   its inputs alone. The sampler and the runners are outside this rule. The batch size, the
   thread count, the head block, the key-head split, the lane position, the slice alignment,
   and prefetch change no bit. So a batched call gives the bits of per-row calls, and `simd`
   gives the bits of `simd_pf`. The unit tests of each file assert the cases of that file, and
   `tools/gate.py threads` checks the thread counts on a whole model. The kernels keep the rule
   in four ways:
   - Every multiply-accumulate is one `@mulAdd`.
   - Every sum of 8 lanes takes the lane tree order of `lane_tree_simd`.
   - Every sigmoid is `exp_simd.sigmoid`.
   - A kernel pads a tail under 8 lanes with zeros into one vector.
5. **Scalar parts.** These parts run the same scalar code in every backend, so rule 4 holds for
   them with no vector arithmetic:
   - `ropeInplace` and `ropeHeads`: scalar `@cos`, `@sin`, and `std.math.pow`.
   - `attentionMerge`: scalar `std.math.exp`, in a fixed order.
   - An attention head width off the lane grid or over 256: the naive kernel.
   - The GDN conv, `gdn.convStep`: 4 scalar taps per channel, shared with the reference.
   - The GDN per-head scalars (`sigmoid(b)`, `softplus`, the decay): scalar `@exp` and `@log`.
   - The Q4_0, Q4_1, and Q6_K linears, `dequantQ80`, and `embedLookup`: naive.
6. **Build mirror.** `resolveOpsBackend` in `build.zig` does not read `requires`.
   `simdBackendSupportedBy` mirrors what the two SIMD backends declare, and the two must agree.
7. **RoPE style.** `naive.RopeStyle` must match the layout of the GGUF Q and K weights: `norm`
   for the Llama family and `neox` for the Qwen family. The wrong style scrambles attention at
   every position, and greedy decode then repeats itself.
