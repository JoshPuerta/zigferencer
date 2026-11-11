# Shaders

This directory holds the GLSL compute sources. Each one transforms activation and weight buffers and
holds no host state. Each mirrors a scalar op in `ops/naive.zig` or `ops/gdn.zig`, and none calls
another. The build compiles each source to a SPIR-V blob with `glslc` and embeds it with
`@embedFile`. A `Pipeline` in [`../kernels/pipeline.zig`](../kernels/pipeline.zig) wraps each blob.

## Files

| File | Holds |
| --- | --- |
| `vec_add.comp` | The proof-of-life kernel, `out = a + b`. |
| `residual_add.comp` | The residual add, in place. |
| `rmsnorm.comp` | RMSNorm of one row. |
| `rmsnorm_batch.comp` | RMSNorm of each row of a batch. |
| `qwen_qk_norm.comp` | The per-head Q and K norm of Qwen3.5 full attention. |
| `rope.comp` | RoPE of one head at one position. |
| `rope_multi.comp` | RoPE of every head of one token. |
| `rope_multi_batch.comp` | RoPE of every head of a batch of tokens. |
| `swiglu.comp` | The FFN activation, `silu(gate) * up`. |
| `sigmoid_gate.comp` | The output gate of Qwen3.5 full attention. |
| `linear_q8_0.comp` | The bit-exact Q8_0 matvec. |
| `linear_q8_0_reduce.comp` | The Q8_0 matvec with one workgroup per output. |
| `linear_q8_0_subgroup.comp` | The reduce matvec with a subgroup sum. |
| `linear_q8_0_batch.comp` | The Q8_0 batched matmul for prefill. |
| `linear_q8_0_gemm.comp` | The tiled Q8_0 GEMM for prefill. |
| `linear_q4_0_reduce.comp` | The Q4_0 matvec with one workgroup per output. |
| `linear_q4_0_batch.comp` | The Q4_0 batched matmul. |
| `attention.comp` | Decode attention for a row that fits the split length. |
| `attention_batch.comp` | Causal prefill attention over the resident prefix of a chunk. |
| `attention_multi_stream.comp` | Decode attention over independent K and V slabs. |
| `attention_split.comp` | The first pass of attention for a row past the split length. |
| `attention_merge.comp` | The second pass: the ordered merge of the split records. |
| `kv_commit_f16.comp` | The f32 to f16 narrowing of the KV commit scratch. |
| `gdn_decode.comp` | The Gated-DeltaNet step of one token. |
| `gdn_chunk_conv.comp` | The conv and norm frontend of chunked GDN prefill. |
| `gdn_chunk_scan.comp` | The chunked delta-rule scan. |
| `gdn_chunk_norm.comp` | The gated RMSNorm of chunked GDN prefill. |

## How the pieces fit

Each shader is self-contained: set-0 storage buffers in, shape in push constants, one result buffer
out. The recording side binds the buffers and sequences the dispatches. The shaders form families
that a runner swaps by shape.

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| Matvec family | `linear_q8_0.comp`, `_reduce`, `_subgroup`, `linear_q4_0_reduce.comp` | One matmul over one storage-buffer surface | `build.zig` | `LinearQ80`, `LinearQ80Reduce`, `LinearQ80Subgroup`, `LinearQ40Reduce` |
| Batch matmul family | `linear_q8_0_batch.comp`, `linear_q8_0_gemm.comp`, `linear_q4_0_batch.comp` | The same matmul with a row count `b` | `build.zig` | `LinearQ80Batch`, `LinearQ80Gemm`, `LinearQ40Batch` |
| Attention family | `attention*.comp` | A resident pass, or a split pass and a merge | `build.zig` | `Attention`, `AttentionBatch`, `AttentionMultiStream`, `AttentionSplit` |
| GDN family | `gdn_*.comp` | The decode step, or the three chunked prefill passes | `build.zig` | `GdnDecode`, `GdnChunkParallel` |

A decode layer composes the families in this order.

```
  rmsnorm --> linear_q8_0* (Q, K, V) --> rope[_multi] --> attention
                                                              |
        (a Qwen3.5 GDN layer swaps gdn_decode in for this     |
         block, and a full-attention layer adds               |
         qwen_qk_norm and sigmoid_gate)                       v
  residual_add <-- linear_q8_0* (out proj) <------------ attention output
      |
      v
  rmsnorm --> linear_q8_0* (gate, up) --> swiglu --> linear_q8_0* (down) --> residual_add
```

A deeper row takes the split path, and a GDN prefill takes three passes.

```
  attention.comp                     a row of at most limits.attn_split_len positions
  attention_split --> attention_merge    a deeper decode or prefill row
  gdn_decode                         one token
  gdn_chunk_conv --> gdn_chunk_scan --> gdn_chunk_norm    a prefill chunk
```

The variants of a family hide behind one shape. Every `linear_q8_0_*` kernel and the Q4_0 pair use
the same four bindings. The batch and GEMM forms add the row count `b` to the push constants. A
runner picks `_reduce` for a `B = 1` decode and `_batch` or `_gemm` for prefill, and no caller
changes. In `attention_batch` the rows share one cache and differ by causal `t_len`. In
`attention_multi_stream` the rows share `t_len` and differ by K and V slab.

## Rules

- **One source can compile more than once.** The `defines` column of the `vulkan_shaders` table in
  `build.zig` reaches `glslc` as `-D<name>`. The attention shaders `attention`, `attention_batch`,
  `attention_multi_stream`, and `attention_split` use it to produce a KV-f32 blob and a KV-f16 blob
  each. `attention_merge` reads only f32 records, so it compiles once.
- **Bit-exact or a measured tolerance, and nothing between.** A bit-exact kernel keeps the CPU
  reduction order and marks every accumulator `precise`. Then the device compiler cannot fuse `a*b +
  c` into an `fma`. The bit-exact kernels are `vec_add`, `residual_add`, `linear_q8_0`, and
  `kv_commit_f16`. A kernel that reorders a reduction, or that reads a GPU transcendental, declares
  a measured `.relative` value on its `Pipeline` wrapper in `../kernels/`. The `Numerics` section
  of each header names the value its wrapper declares. `tools/bench.zig` prints the relative error
  per variant and fails the row on drift. `qwen_qk_norm.comp` is the one kernel whose wrapper
  declares no tolerance, and the token-hash gates cover it.
- **The split length bounds the resident attention shaders.** `attention.comp`,
  `attention_multi_stream.comp`, and `attention_batch.comp` hold every score of a row in shared
  memory. The row has `limits.attn_split_len` positions. That limit reaches `glslc` as `ATTN_SPLIT`.
  The host dispatches them only for a row that fits. A deeper decode or prefill row takes
  `attention_split.comp`, one workgroup per split and run of query heads, then
  `attention_merge.comp`. The merge combines the splits in ascending split order, so one depth gives
  one result bit for bit. The resident decode shaders hold no per-lane register array, so they carry
  no `head_dim` bound. The split pass stages the q row of a head in shared memory of
  `limits.attn_max_head_dim` floats. That limit reaches `glslc` as `ATTN_D_MAX`. Each GPU runner
  refuses at boot a model whose `head_dim` is past that width or not a multiple of 8. Each dispatch
  asserts both rules.
- **Storage packing follows the device, not taste.** The target driver exposes 16-bit storage and
  fp16 arithmetic, so scales are `float16_t`. It exposes neither 8-bit storage nor `shaderInt8`, so
  the host packs four i8 quants per `uint`, little-endian. The kernels decode with
  `bitfieldExtract(int, off, 8)`, which sign-extends the way `@as(i8, @bitCast(byte))` does on the
  CPU. The Q4_0 kernels unpack a nibble as `(nibble - 8) * scale`.
- **The f16 narrowing pins its rounding rule.** `kv_commit_f16.comp` declares the SPIR-V
  `RoundingModeRTE` execution mode at width 16 through `GL_EXT_spirv_intrinsics`. The device then
  rounds to nearest even by declaration and not by default. `tests/vulkan_kv_commit_f16_test.zig`
  compares raw f16 bit patterns, never widened values.
- **Every workgroup size is a multiple of RDNA2 wave32.** The matvec and batch matmuls run 32 lanes,
  one wave. The attention, narrowing, and RoPE kernels run 64 lanes, two waves. The GDN decode,
  conv, and norm kernels run 128 lanes, one per head-dim index. The GDN scan runs 32 lanes, one per
  state column of its block. The elementwise kernels, the norms, and the GEMM run 256 threads, eight
  waves. In a kernel with a `barrier()`, a tail lane masks its write and never returns early. The
  whole wave must reach every cooperative load and every `barrier()`.
- **The f32 attention blobs are unchanged by the KV variants.** One source compiles twice. Only the
  storage type of the K and V bindings moves. The loads widen to `float` at the read, the score
  buffer stays `shared float`, the accumulators stay `precise float`, and the element indexing is
  unchanged.
- **Every kernel has a named CPU reference.** `gpu-selftest` checks the matmul, the attention, and
  the GDN kernels against `ops/naive.zig` and `ops/gdn.zig` on random inputs. Each kernel must stay
  within its declared tolerance. On the Zig side an assert holds the Q8_0 and Q4_0 block geometry
  equal to `core/dtype.zig`.

## The form of a shader

The header comment of each shader is its specification. It holds labeled sections in this order:
`What`, `Math`, `Schedule`, `Memory`, `Hardware`, `Numerics`, `Bindings`, and `Defines`. A tiny
kernel keeps `What`, `Hardware`, `Numerics`, and `Bindings`. Inside `main`, `// Pass N:` labels
match the pass rows of `Schedule`.
