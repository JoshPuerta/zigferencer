# src/ops_vulkan/shaders

The GPU datapath: 27 GLSL compute sources that transform activation and weight
buffers and hold no host state. Each one mirrors a scalar op in `ops/naive.zig`
or `ops/gdn.zig`, and none of them calls another. They live apart from the Zig
layer because the recording side owns every handle and every submit, while a
shader owns only its arithmetic. The build compiles the sources to 31 SPIR-V
blobs and embeds each blob with `@embedFile`, and a `Pipeline` struct in
[`../kernels.zig`](../kernels.zig) wraps each blob one to one. The GPU section of
[the architecture map](../../../docs/architecture.md) places this datapath under
the per-op backend and the fused runners.

One source can compile more than once. The `defines` column of the
`vulkan_shaders` table in `build.zig` reaches `glslc` as `-D<name>`. Four
attention shaders use it to produce a KV-f32 and a KV-f16 blob each:
`attention`, `attention_batch`, `attention_multi_stream`, and
`attention_split`. `attention_merge` reads only f32 records, so it compiles
once. That is where 27 sources become 31 blobs.

## Files

| File | Role |
| --- | --- |
| `linear_q8_0.comp` | Q8_0 matvec, one invocation per output `j`, running the exact scalar loop of the CPU kernel. It is bit-exact, and it validates the whole GPU stack against the token hash before any faster matmul exists. |
| `linear_q8_0_reduce.comp` | Q8_0 matvec with one workgroup per output `j`: 32 lanes stride the K blocks so the loads coalesce, and lane 0 sums the partials in shared memory. It wins when `n` is too small to keep the compute units fed. Tolerance `2e-4`. |
| `linear_q8_0_subgroup.comp` | `_reduce` with the input staged in shared memory and one `subgroupAdd` in place of the shared-memory round trip and barrier. Tolerance `2e-4`. |
| `linear_q8_0_batch.comp` | Batched matmul `(B,K) x (N,K) -> (B,N)`. A single-wave workgroup caches one dequantized weight column in registers across `B_TILE = 16` rows, so a prefill streams the weights `ceil(B/16)` times. Tolerance `2e-4`. |
| `linear_q8_0_gemm.comp` | Register-blocked, shared-memory-staged GEMM: a 256-thread workgroup, `BM = BN = 64`, `BK = 32`, and a 4x4 micro-tile per thread. It decouples tile reuse from register pressure and beats the single-wave batch kernel on prefill. Tolerance `2e-4`. |
| `linear_q4_0_reduce.comp` | The Q4_0 sibling of `_reduce`. The structure is the same and only the dequant changes, from bytes to nibbles, so a bandwidth-bound decode matvec reads about half the weight bytes. Tolerance `2e-4`. |
| `linear_q4_0_batch.comp` | The Q4_0 sibling of `_batch`, with its own tile geometry: `B_TILE = 8`, `N_TILE = 32`, `K_TILE = 32`. Tolerance `2e-4`. |
| `attention.comp` | Fused scaled-dot-product attention for decode, one query head per workgroup at 64 lanes. Grouped-query attention maps many query heads onto one KV head. The host runs it for a row of at most `limits.attn_split_len` positions. Tolerance `1e-3`. |
| `attention_batch.comp` | Causal prefill attention for the resident prefix of a chunk, one workgroup per `(batch row, head)`. The mask is implicit, because row `bi` reads only through position `start_pos + bi`. All rows share one K and V cache, so the batch is free parallelism. A row past `limits.attn_split_len` positions takes `attention_split.comp`. Tolerance `1e-3`. |
| `attention_multi_stream.comp` | Lockstep or arbitrary-`(slot, position)` decode: the rows share `t_len` and read independent K and V slabs. `q_row_base` and `kv_stream_base` are decoupled, so a caller with per-row positions records one single-row dispatch per row. The row limit of `attention` applies. Tolerance `1e-3`. |
| `attention_split.comp` | The first pass of decode and prefill attention for a row past `limits.attn_split_len` positions. One workgroup per split and per run of up to `limits.attn_split_queries` query heads of one GQA group walks at most `limits.attn_split_len` positions. It reads each K and V row once for all heads of the run and writes one partial record per head: the unnormalized V sums, the maximum score, and the exponential sum. Tolerance `1e-3`. |
| `attention_merge.comp` | The second pass: one workgroup per query head and 64 elements combines the partial records of a row in ascending split order, so one depth gives one result bit for bit. The workgroup computes each split weight once into LDS, and each lane sums one element. It uses no atomics and no tree over the splits. Tolerance `1e-3`. |
| `kv_commit_f16.comp` | Narrows one layer's K and V commit scratch from f32 to f16 in a single dispatch, between the scratch write and the scatter. It never sees the cache layout, and it declares the SPIR-V round-to-nearest-even mode at width 16. Bit-exact. |
| `gdn_decode.comp` | The per-token Gated-DeltaNet step of a Qwen3.5 GDN layer: causal depthwise conv1d with SiLU, L2-norm of q and k, the q scale, the gated delta recurrence, and a gated RMSNorm. One workgroup per key head, decoupled-head aware. It mirrors `step` in `ops/gdn.zig`. Tolerance `1e-4`. |
| `gdn_chunk_conv.comp` | The conv1d, SiLU, and L2-norm frontend of the chunked GDN prefill. It writes the `q_n`, `k_n`, and `v_conv` that the scan kernel consumes. Tolerance `1e-4`. |
| `gdn_chunk_scan.comp` | The chunked-parallel gated delta-rule scan, one workgroup per block of `limits.gdn_scan_cols` state columns of a value head. Each lane holds its state column in registers for the whole chunk. A sub-chunk triangular solve in shared memory breaks the per-token dependency chain. It mirrors `chunkScan` in `ops/gdn.zig` and writes the raw readout. Tolerance `1e-4`. |
| `gdn_chunk_norm.comp` | The gated RMSNorm of the chunked GDN prefill, one workgroup per value head and token. It couples the columns of a head, so it runs after the scan, in place on the readout. |
| `rmsnorm.comp` | RMSNorm of one row. A single 256-lane workgroup runs two cooperative passes with a shared-memory tree-reduce between them. Tolerance `5e-5`. |
| `rmsnorm_batch.comp` | `rmsnorm` applied per row of a `(B,d)` input, with `gl_WorkGroupID.x` as the row. Per-row independence needs no cross-row synchronization. Tolerance `5e-5`. |
| `qwen_qk_norm.comp` | Per-head RMSNorm of the Qwen3.5 full-attention Q and K, fused with the de-interleave of the `query|gate` projection. Every lane re-sums the squared head in sequence to hold the CPU summation order. It mirrors `fullMixer` in `runtime/qwen_runner.zig`, and its wrapper declares no tolerance. |
| `rope.comp` | In-place RoPE on one `head_dim` vector at one position. `style_norm` picks adjacent pairs for the Llama family or half-split pairs for the neox and Qwen family. Tolerance `5e-5`. |
| `rope_multi.comp` | `rope` over every head of one token in one dispatch, indexed by `(head, pair)`. It collapses tens of per-head dispatches per layer into one. Tolerance `1e-5`. |
| `rope_multi_batch.comp` | `rope_multi` extended over `B` token rows. `pos_stride` switches between prefill, where row `bi` is token `start_pos + bi`, and lockstep decode, where every row shares `start_pos`. `row_base` offsets the buffer row without moving the position math. Tolerance `1e-5`. |
| `swiglu.comp` | `dst[i] = silu(gate[i]) * up[i]`, the activation between the two FFN matmuls. Tolerance `1e-5`. |
| `sigmoid_gate.comp` | `dst[i] *= sigmoid(gate[i])` in place, the Qwen3.5 full-attention output gate. It gates the attention readout, which is a different job from the FFN gating of `swiglu`. Tolerance `1e-5`. |
| `residual_add.comp` | `dst[i] += x[i]`, the residual connection around the attention block and the FFN block. Binding 1 is read-write, so one buffer is both accumulator and destination. Bit-exact. |
| `vec_add.comp` | `out[i] = a[i] + b[i]`, the proof-of-life kernel. One dispatch exercises the instance, device, queue, descriptor set, pipeline, and command buffer that every other shader reuses. Bit-exact. |

## How the pieces fit

Each shader is self-contained: set-0 storage buffers in, shape in push constants,
one result buffer out. The recording side binds the buffers and sequences the
dispatches. The families compose into a transformer layer the way the runners
drive them:

```
  per layer (decode):
    rmsnorm --> linear_q8_0* (Q,K,V proj) --> rope[_multi] --> attention
                                                                  |
                       (Qwen3.5 GDN layer: gdn_decode replaces    | the attention
                        block. qwen_qk_norm and sigmoid_gate      | wrap a full-
                        attention layer.)                         |
                                                                  v
    residual_add <-- linear_q8_0* (out proj) <-------------- attention output
        |
        v
    rmsnorm --> linear_q8_0* (gate, up) --> swiglu --> linear_q8_0* (down) --> residual_add
```

The variants inside a family are interchangeable behind one shape. Every
`linear_q8_0_*` kernel, and the Q4_0 pair, computes the same matmul against the
same storage-buffer surface: four bindings under the same names. The push
constants differ by one field, because the batch and GEMM forms carry the row
count `b` in front of `k` and `n`. A runner therefore picks
`_reduce` for a `B = 1` decode and `_batch` or `_gemm` for prefill, and no caller
changes. `attention_batch` and `attention_multi_stream` are the two mirror-image batch layouts. In `attention_batch` the rows share one cache and differ by causal `t_len`. In `attention_multi_stream` the rows share `t_len` and differ by K and V slab. For a decode or prefill row
past `limits.attn_split_len`, `attention_split` then `attention_merge` replace
the resident attention dispatch. The four GDN shaders pipeline as
`gdn_chunk_conv`, `gdn_chunk_scan`, then `gdn_chunk_norm` for prefill, and
`gdn_decode` is the single-token path.

## Rules

- **Bit-exact or a measured tolerance, and nothing between.** A bit-exact kernel keeps the CPU reduction order and marks every accumulator `precise`. Then `glslc -O` cannot fold `a*b + c` into an `fma`. The four bit-exact kernels are
  `vec_add`, `residual_add`, `linear_q8_0`, and `kv_commit_f16`.
  A kernel that reorders a reduction, or that reads a GPU transcendental,
  declares a measured `.relative` value on its `Pipeline` wrapper in
  `../kernels.zig`. Each header ends with the value its wrapper declares, and
  `tools/bench.zig` prints the relative error per variant and fails the row on
  drift. `qwen_qk_norm.comp` is the one kernel whose wrapper declares no
  tolerance, and the token-hash gates cover it.
- **The split length bounds the resident attention shaders.**
  `attention.comp`, `attention_multi_stream.comp`, and `attention_batch.comp`
  hold every score of a row in a shared-memory row of `limits.attn_split_len` positions. That limit is 256 and reaches `glslc` as `ATTN_SPLIT`. The host
  dispatches them only for a row that fits. A deeper decode or prefill row
  takes `attention_split.comp`, one workgroup per split and run of query heads,
  then `attention_merge.comp`. The merge combines the splits in ascending split
  order, so one depth gives one result bit for bit.
  The resident decode shaders hold no per-lane register array, so they carry
  no `head_dim` bound. The split pass stages the q row of a head in shared
  memory of `limits.attn_max_head_dim` floats, which reaches `glslc` as
  `ATTN_D_MAX`. Each GPU runner refuses at boot a model whose `head_dim` is
  past that width or not a multiple of 8, and each dispatch asserts both rules.
- **Storage packing follows the device, not taste.** The target driver exposes
  16-bit storage and fp16 arithmetic, so scales are `float16_t`. It exposes
  neither 8-bit storage nor `shaderInt8`, so the host packs four i8 quants per
  `uint`, little-endian. The kernels decode with `bitfieldExtract(int, off, 8)`,
  which sign-extends the way `@as(i8, @bitCast(byte))` does on the CPU. The Q4_0
  kernels unpack a nibble as `(nibble - 8) * scale`.
- **The f16 narrowing pins its rounding rule.** `kv_commit_f16.comp` declares the SPIR-V `RoundingModeRTE` execution mode at width 16 through `GL_EXT_spirv_intrinsics`. The device then rounds to nearest even by declaration and not by default. `tests/vulkan_kv_commit_f16_test.zig` compares raw f16 bit
  patterns, never widened values.
- **Every workgroup size is a multiple of RDNA2 wave32.** The matvec and batch
  matmuls run 32 lanes, one wave. The attention, narrowing, and RoPE kernels run
  64 lanes, two waves. The GDN decode, conv, and norm kernels run 128
  lanes, one per head-dim index. The GDN scan runs 32 lanes, one per state
  column of its block. The elementwise kernels, the norms, and the GEMM run 256
  threads, eight waves. A tail lane masks its write rather than returning early,
  because the whole wave must reach every cooperative load and every
  `barrier()`.
- **The f32 attention blobs are unchanged by the KV variants.** One source
  compiles twice, and only the storage type of the K and V bindings moves. The
  loads widen to `float` at the read, the score buffer stays `shared float`, the
  accumulators stay `precise float`, and the element indexing is unchanged.
- **Every kernel has a named CPU reference.** `gpu-selftest` bit-checks the
  matmul, the attention, and the GDN kernels against `ops/naive.zig` and
  `ops/gdn.zig` on random inputs. On the Zig side an assert holds the Q8_0 and
  Q4_0 block geometry equal to `core/dtype.zig`.

## Where to start

1. `vec_add.comp`: the minimal kernel, and the binding, push-constant, and dispatch shape every other shader reuses.
2. `linear_q8_0.comp`: the bit-exact matvec, the reduction-order argument, and the four-quants-per-uint packing.
3. `linear_q8_0_reduce.comp`, then `linear_q8_0_subgroup.comp`: the decode matvec variants, and what reordering costs.
4. `linear_q8_0_batch.comp`, then `linear_q8_0_gemm.comp`: the prefill matmuls and the occupancy argument for the GEMM.
5. `rmsnorm.comp`, `rope.comp`: the cooperative-reduction and elementwise patterns a layer is built from.
6. `attention.comp`, then `attention_batch.comp` and `attention_multi_stream.comp`: decode attention and its two batch layouts. Then `attention_split.comp` with `attention_merge.comp`: decode and prefill attention past the split length.
7. `gdn_decode.comp`, then `gdn_chunk_conv.comp` with `gdn_chunk_scan.comp` and `gdn_chunk_norm.comp`: the Qwen3.5 GDN path, read beside `ops/gdn.zig`.
