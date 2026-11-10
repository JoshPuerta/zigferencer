# Vulkan ops

This directory holds the GPU op backend. It has one Zig wrapper per compute shader and the buffer
helpers that move bytes to the device. It also has the facade that backs the CPU op API.
The whole directory compiles only under `-Druntime=vulkan`. Values cross its edges: a `Tensor` or a
plain slice goes in, a caller-owned `[]f32` comes out, and no device handle escapes.

## Files

| File | Holds |
| --- | --- |
| `kernels.zig` | One struct per compute shader, each with a `Pipeline`, and the buffer helpers. |
| `dispatch_list.zig` | `DispatchList`, the command lists and the device loader of a trace build. |
| `stream_ring.zig` | `StreamRing`, the double-buffered host-to-device upload ring. |
| `wrap.zig` | The op-API facade that maps the CPU op signatures onto kernels. |
| [`shaders/`](./shaders/README.md) | The GLSL compute sources. |

## How the pieces fit

The fused runners of `runtime/` drive the kernel structs directly, and `wrap.zig` drives them for
the per-op path. Both sit on a `Context`.

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| `Context` | `runtime/vulkan_ctx` | The device, the queue, the command pool, and the staging buffer | `runtime/vulkan_runner`, `cli/boot/boot_resources` | Every kernel |
| `Runner` | `runtime/vulkan_runner` | A `Context` and the per-op kernel structs | `wrap.initBackend` | `wrap.zig` |
| `Pipeline` | `kernels.zig` | The shader module, the set layout, the pipeline layout, and the handle | Each kernel `init` | The same kernel |
| Kernel struct | `kernels.zig` | A `Pipeline` and a `record` or `run` method, such as `LinearQ80` or `Attention` | `Runner`, the fused runners | `wrap.zig`, `runtime/attn_record`, the fused runners |
| `KvStore` | `kernels.zig` | The device KV regions, one per layer and tensor | `FusedRunner` | The same runner |
| `VramArena` | `kernels.zig` | Aligned views of a few device allocations for the weights | `FusedRunner` | The same runner |
| `StreamRing` | `stream_ring.zig` | Two upload segments, their fences, and a command pool | `tests/vulkan_stream_ring_test` | The same test |

The per-op path runs from a CPU op signature to a submit.

```
  a CPU runner calls the CPU op signatures
      |  values in, values out
      v
  wrap.zig      heavy ops --> GPU        small ops --> ops/naive.zig
      |         linearF32Q80, rmsnorm,   embedLookup, ropeInplace, linearF32
      |         swiglu, residualAdd, attentionSingle, gdnStep
      v
  runtime/vulkan_runner.Runner    Context + the per-op kernel structs
      v
  kernels.zig   Pipeline + OneShot --> one dispatch, submitOnce, wait idle
```

The fused path records a whole token into one command buffer instead.

```
  FusedRunner(Family)                           (runtime/)
      |  KvStore + VramArena bind the weights and the KV
      |  attn_record walks an attn_split.AttnPlan through kernels.AttnKernels
      v
  kernel structs record cmdBindPipeline + cmdDispatch into the DecodeRing buffer
      v
  one submit per token, with the fence of the DecodeRing
```

`KvStore` puts the KV regions of `core/kv_index.zig` in device allocations. `VramArena` hands out
aligned views and never frees one. A plan from a dry run of the upload sizes each allocation to its
views. A one-shot submit, such as an upload or a slot fill, goes through `submitOnce` on the staging
command buffer of the `Context`. `StreamRing` stays outside both paths. It owns fences and a command
pool, and `core/ring_fsm.zig` holds its slot table. No runner drives it.

## Rules

- **Only `StreamRing` tracks a submission.** Every other path here submits with a null fence and
  blocks on `queueWaitIdle`. The ring leaves one upload outstanding on purpose. No runner drives it.
  `src/main.zig` takes the address of every method so the build analyzes the bodies, and
  `tests/vulkan_stream_ring_test.zig` gates it against a live device.
- **A shader header owns its kernel.** The header of each shader holds the math, the schedule, the
  memory table, and the bindings. A wrapper states what the host owes: a precondition or a
  tolerance.
- **Build guard.** All three Zig files raise `@compileError` when `build_options.use_vulkan` is
  false. The backend is unreachable outside `-Druntime=vulkan`.
- **The split length bounds the resident attention kernels.** `Attention`, `AttentionMultiStream`,
  and `AttentionBatch` hold every score of a row in shared memory of `limits.attn_split_len`
  positions. A decode row or a prefill row past that length runs the split pass and the merge pass
  of `AttentionSplit`. A prefill chunk runs its resident prefix as one `AttentionBatch` dispatch,
  and `AttentionBatch.recordInto` asserts that the prefix ends at or before the split length.
- **One attention kernel set per build.** `AttnKernels` is the set that `-Dattn-kernel` selects.
  `SplitAttnKernels` is the only one. A kernel set has the method set `recordResident`,
  `recordResidentRun`, `recordSplit`, `recordMerge`, `recordBarrier`, and `recordMark`, and it
  declares its summation order class and its tolerance. The fused and hybrid runners record each attention layer
  through `runtime/attn_record.zig`, which walks an `attn_split.AttnPlan`. A new kernel is one
  kernel set struct and one entry in the build enum.
- **A kernel needs a stable `Context` address.** Every kernel captures `*const Context` at `init`.
  `runtime/vulkan_runner.Runner.init` heap-allocates the bundle, so `&self.ctx` holds for the life
  of the kernels. Never hand a kernel a `Context` that lives on a stack or that a function returns
  by value.
- **Kernels own pipelines, not data.** A kernel struct owns its `Pipeline`, its `OneShot`, and its
  grow-on-demand staging buffers. Activations and weights cross the edge as slices and `Tensor`
  values the caller keeps.
- **The weight cache never evicts.** `wrap.linearF32Q80` keys the cache on the pointer of the weight
  slice. A weight that many layers reference stages to VRAM exactly once. Weights live in VRAM for
  the life of the program.
- **Declared tolerance, never a guessed one.** `wrap.requires.tolerance` is `.relative = 2e-3`, the
  loosest per-op value with a cushion. No op in the chain widens the drift past its own declared
  value. `VecAdd`, `ResidualAdd`, `LinearQ80`, and `KvCommitF16` declare `.bit_exact`. The rest
  declare a measured `.relative` value that `tools/bench.zig` checks per row. `QwenQkNorm` is the
  one kernel struct with no `requires`, and the token-hash gates cover it.
- **`Pipeline.init` asserts its inputs.** The SPIR-V length must be a multiple of four. The binding
  count must be at least one. The push-constant size must be a multiple of four and at least four. A
  misaligned blob returns `error.SpirVMisaligned`.
- **The per-op path panics on a GPU failure.** The op signatures return no error, so `wrap.zig`
  calls `std.debug.panic` with the failing kernel named. This path is a correctness reference and a
  benchmark target. The fused runners in `runtime/` carry the fast path and their own error sets.
