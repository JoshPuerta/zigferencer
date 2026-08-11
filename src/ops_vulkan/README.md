# src/ops_vulkan

The GPU op backend. It holds one Zig wrapper per compute shader and the buffer helpers that move bytes to the device. It also holds the facade that backs the CPU op API with those wrappers. It is a separate layer because it owns Vulkan handles,
records dispatches, and submits, while the shaders it drives stay pure. The
whole directory compiles only under `-Druntime=vulkan`, and a `cpu` build links
none of it. Values cross its edges. A `Tensor` or a plain slice goes in, a caller-owned `[]f32` comes out, and no device handle escapes.
[docs/architecture.md](../../docs/architecture.md) places this layer in the
pipeline.

## Files

| File | Role |
| --- | --- |
| `kernels.zig` | One struct per compute shader (`VecAdd`, `LinearQ80` and its six siblings, `Rmsnorm`, `Swiglu`, `Attention`, `GdnDecode`, and the rest), each owning a `Pipeline`: the shader module, the descriptor-set layout, the pipeline layout, and the pipeline handle. It also holds the buffer helpers (`HostBuffer`, `DeviceBuffer`, `KvStore`, `VramArena`, `OneShot`, `Copy`, `submitOnce`, `runStagingCopies`) and re-exports `StreamRing`. `KvStore` places the KV regions of `core/kv_index.zig` in device allocations. Twenty-three of the kernel structs declare a `requires` of their own, because the variants span more than one tolerance class. |
| `stream_ring.zig` | `StreamRing`, a double-buffered host-to-device upload ring over plain byte segments. The producer fills one mapped segment while the device uploads the other, and `view` hands a finished segment back as a descriptor binding. It is the one place in the engine that owns fences. It carries its own command pool, so a pending upload never shares `ctx.command_pool` with the wait-idle paths. |
| `wrap.zig` | The op-API facade. It maps `linearF32Q80`, `rmsnorm`, `swiglu`, `residualAdd`, `attentionSingle`, and `gdnStep` onto kernels, one submit and one `waitIdle` per op. It re-exports the small or fragmented ops (`embedLookup`, `ropeInplace`, `linearF32`) straight from `ops/naive.zig`, because a dispatch costs more than the CPU microseconds. |
| `shaders/` | The 27 GLSL compute sources. The build compiles them to 31 SPIR-V blobs with `glslc` and embeds each blob with `@embedFile`. See [shaders/README.md](./shaders/README.md). |

## How the pieces fit

`wrap.zig` never touches a shader. It holds three process-wide values. The first is the `vulkan_runner.Runner` that owns the `Context` and the per-op kernel bundle. The second is a VRAM weight cache keyed by the mmap slice pointer. The third is a `GdnDecode` built on the first `gdnStep` call. `initBackend` creates them and `deinitBackend` destroys
them. Each op call reaches into the Runner for its kernel struct and calls `run`
or `dispatch`.

```
  a runner calls the CPU op signatures
      |  values in, values out
      v
  wrap.zig      heavy ops --> GPU        small or fragmented ops --> ops/naive.zig
      |         linearF32Q80, rmsnorm,   embedLookup, ropeInplace,
      |         swiglu, residualAdd,     linearF32
      |         attentionSingle, gdnStep
      v
  runtime/vulkan_runner.Runner    Context + 10 per-op kernel structs, fixed heap address
      |
      v
  kernels.zig   Pipeline {module, set layout, pipeline layout, handle}
      |         + OneShot {descriptor pool, set, command buffer}
      |         + HostBuffer / DeviceBuffer / VramArena staging, KvStore regions
      v
  shaders/*.spv (@embedFile)   record one dispatch, submit, wait idle
```

`kernels.zig` splits by role. `Pipeline` is the four-object compute chain every
wrapper embeds. `OneShot` is the descriptor pool, set, and command buffer that a
kernel with a self-contained `run` needs, and a record-only kernel leaves those
to its caller. The buffer structs stage host bytes into VRAM. `VramArena` hands
out aligned views and never frees one, so a weight lives as long as the runner.
`KvStore` holds the device KV of a fused runner or a hybrid suffix: one region
per layer and tensor, packed into whole allocations. The device buffer limits
therefore bind one region. The kernel struct binds what the caller supplies and records
`cmdBindPipeline` plus `cmdDispatch`.

A one-shot submit, such as an upload, a swap chunk, a slot fill or clone, or a
test dispatch, goes through `submitOnce`. It resets the one staging command
buffer of the `Context`, records, submits, and blocks on `queueWaitIdle`. The
translation layer makes a native command allocator for each command buffer, and
the paravirtual driver leaks part of it when the buffer is freed. So no one-shot
submit allocates a command buffer. A buffer that `allocateOnePrimaryCmdBuffer`
makes lives as long as its owner.

`StreamRing` sits beside that path rather than inside it. Every other Vulkan
path here submits with a null fence and blocks on `queueWaitIdle`, so it tracks
no submission. The ring leaves one upload outstanding on purpose, so it owns
fences and a command pool of its own, and `core/ring_fsm.zig` holds its slot
state table. No runner drives the ring. `src/main.zig` takes the address of
every method so the build analyzes the bodies, and
`tests/vulkan_stream_ring_test.zig` gates it against a live device.

## Rules

- **Build guard.** All three Zig files raise `@compileError` when
  `build_options.use_vulkan` is false. The backend is unreachable outside
  `-Druntime=vulkan`.
- **The split length bounds the resident attention kernels.** `Attention`,
  `AttentionMultiStream`, and `AttentionBatch` hold every score of a row in
  shared memory of `limits.attn_split_len` positions. A decode row or a prefill
  row past that length runs the split pass and the merge pass of
  `AttentionSplit`. A prefill chunk runs its resident prefix as one
  `AttentionBatch` dispatch, and `AttentionBatch.recordInto` asserts that the
  prefix ends at or before the split length.
- **One attention kit per build.** `AttnKit` is the kit that
  `-Dattn-kernel` selects, `SplitAttnKit` today. A kit has the method set
  `recordResident`, `recordResidentRun`, `recordSplit`, `recordMerge`, and
  `recordBarrier`, and it
  declares its summation order class and its tolerance. The fused and hybrid
  runners record each attention layer through `runtime/attn_record.zig`, which
  walks an `attn_split.AttnPlan`. A new kernel is one kit struct and one entry
  in the build enum.
- **A kernel needs a stable `Context` address.** Every kernel captures
  `*const Context` at `init`. `runtime/vulkan_runner.Runner.init` heap-allocates
  the bundle, so `&self.ctx` holds for the life of the kernels. Never hand a
  kernel a `Context` that lives on a stack or that a function returns by value.
- **Kernels own pipelines, not data.** A kernel struct owns its `Pipeline`, its
  `OneShot`, and its grow-on-demand staging buffers. Activations and weights
  cross the edge as slices and `Tensor` values the caller keeps.
- **The weight cache never evicts.** `wrap.linearF32Q80` keys the cache on the pointer of the weight slice. A weight that many layers reference stages to VRAM exactly once. Weights live in VRAM for the life of the program.
- **Declared tolerance, never a guessed one.** `wrap.requires.tolerance` is `.relative = 2e-3`, the loosest per-op value with a cushion. No op in the chain widens the drift past its own declared value. `VecAdd`, `ResidualAdd`,
  `LinearQ80`, and `KvCommitF16` declare `.bit_exact`. The rest
  declare a measured `.relative` value that `tools/bench.zig` checks per row.
  `QwenQkNorm` is the one kernel struct with no `requires`, and the token-hash
  gates cover it.
- **`Pipeline.init` asserts its inputs.** The SPIR-V length must be a multiple of four. The binding count must be at least one. The push-constant size must be a multiple of four and at least four. A misaligned blob returns `error.SpirVMisaligned`.
- **The per-op path panics on a GPU failure.** The op signatures return no
  error, so `wrap.zig` calls `std.debug.panic` with the failing kernel named.
  This path is a correctness reference and a benchmark target. The fused runners
  in `runtime/` carry the fast path and their own error sets.

## Where to start

1. `kernels.zig`: read `Pipeline` and `OneShot` first, then `VecAdd` end to end, then the buffer helpers.
2. `wrap.zig`: how an op signature picks a kernel, and where the CPU and GPU line falls.
3. `runtime/vulkan_runner.zig`: the factory that pins the `Context` address the kernels capture.
4. `stream_ring.zig` with `core/ring_fsm.zig`: the asynchronous upload path and the state table it routes through.
5. `shaders/README.md`, then `runtime/llama_fused.zig`: the GLSL itself, and the fused path this simple backend measures against.
