# src/runtime

The control units of the engine. Every module here owns mutable state: the KV
slabs, the Gated-DeltaNet recurrent slabs, the scratch buffers, the decode
position, and the GPU command-buffer recording. Every module here also sequences
the pure ops, which stay out in `ops/` and `core/`. It is a separate layer
because state and sequencing belong on one side of the control and datapath
split that [docs/architecture.md](../../docs/architecture.md) describes. The
values that cross its edges carry no storage: `Tensor`, `ModelCard`, and
`TensorRef` come in, and logits and token ids go out.

## Files

| File | Role |
| --- | --- |
| `step_row.zig` | `StepRow`, the batch row every runner steps, with the row rule (`validateRows`) and the start of the logits suffix (`wantLogitsFirst`). It owns no state, so a runner of either family depends on it and not on another runner. |
| `linear_dispatch.zig` | `linearBatchDispatch`, the batched-linear interface. It picks the kernel by weight dtype and fans the quantized paths across output rows. Both CPU runners call it, so the dtype ladder cannot drift. |
| `llama_runner.zig` | The Llama-family forward pass over a bounded set of KV slots. `stepBatch` advances a batch of `(slot, token, pos)` rows and `step` is the one-row wrapper. |
| `qwen_runner.zig` | The Qwen3.5 hybrid forward pass on the CPU: three of every four layers are Gated-DeltaNet, and every fourth is full attention with QK-norm and an output gate. It holds the per-slot recurrent state and the full-attention KV, and it is the coherence reference every GPU kernel is validated against. |
| `kv_cache.zig` | `KvCache`, the static KV slab. One flat byte slab holds the K half then the V half. Each half has the shape `[n_layers][max_seq][n_kv_heads * head_dim]` and carries the element type it was framed at. Ops never see the struct. They get plain typed rows. |
| `session.zig` | `Sessions(RunnerT)`: per-conversation slots over the KV slots of a runner. `feed` and `decodeBatch` are two shells over `advance`, the composed tick that folds one prefill chunk and a decode step into one forward call. It also holds the compile-time check of the runner surface. |
| `sampler.zig` | `greedy` argmax, which has no RNG and anchors every cross-backend token hash. `sample` adds seeded temperature, top-k, min-p, and top-p, and takes one float from the RNG per sampled token. A positive top-k runs the cuts inside its pool. A top-k of 0 runs them exactly over the full vocab with a radix select over the weight bits, and never sorts the vocab. A penalty copies the row into the scratch of `Sessions`. It then changes the logit of each marked id: the repetition rule for an id in the context set, then the presence penalty for an id in the presence set. It never writes the logits. A neutral request reads the logits as they are. The full-vocab draw writes its weights with the vector exponential of `ops/exp_simd.zig`. |
| `parallel.zig` | The threading `Policy` and the row-parallel matmul dispatcher over the fork-join pool of `pool.zig`. `Policy.forHost` gives the defaults: min(physical cores, 8) threads, a fan-out floor of 250,000 FLOPs, and a 50 us worker spin. Below the floor an op runs inline at zero overhead. Above it, it fans out across output-row chunks. The partition is fixed, so the result is bit-identical across thread counts. It also holds the control unit of CPU attention, `attentionTyped`. It runs the units of one row over the same pool and floor, then merges the parts in order. A unit is one KV group over one part. `gdnStep` runs the Gated-DeltaNet prologue on the caller and fans the value heads out over the pool. |
| `pool.zig` | `Pool`, the persistent fork-join pool of the CPU ops. The caller owns it and keeps its address fixed, and the calling thread is participant 0 of every dispatch. `run` publishes a job and a new epoch word. Each participant runs its own first unit, then claims units from a shared counter. The caller returns when every participant reports done. Idle workers spin on the epoch word for `spin_ns`, then sleep on a futex. |
| `cpu_profile.zig` | The per-op CPU decode timer of both runner families. One `ProfileCat` enum names the categories, a family charges a fixed subset, and the subset partitions one decode step. Without `-Dtrace` every type is `void` and no clock is read. Under `-Dtrace` each bracket also feeds the span recorder. |
| `generate.zig` | The greedy decode loop that `run` and `bench-model` share. It owns the sequencing and the timing, and the caller passes the per-token effect in as a `sink` value with an `onToken` method. A sink that also declares `onStep` gets the position and the wall time of each step. |
| `tensor_lookup.zig` | Tensor-by-name lookup over a `loader.TensorRef` list, shared by every runner. `find` returns an optional and `findRequired` returns `error.MissingTensor`, so one rule covers every weight resolution. |
| `test_fakes.zig` | Test support. `test_window` is the fixed KV window of every test that sizes a window from a real model. `test_serve_window` is the smaller fixed window of a test that plans several slots. It also holds the canned-logits body and the vocabulary constant that the runner fakes of `session.zig`, `serve/residency.zig`, `serve/scheduler.zig`, and `serve/server.zig` share. Test code imports it. The root module re-exports it only for the tests under `tests/`. |
| `kv_cost.zig` | The per-layer KV price of a model card, `costVector`: the uniform Llama vector, or the Qwen3.5 vector that prices attention layers by the token and GDN layers by the slot. It touches no device, so a CPU build prices the same vector as a fused-GPU build, and the fused runners forward to it. |
| `attn_record.zig` | The attention recorder of the fused and hybrid runners. `planStep` turns the rows of a step into an `attn_split.AttnPlan`. `record` and `recordPrefill` walk the plan of one layer through the attention kernel set that `-Dattn-kernel` selects. The walk records the resident rows, then per record group the split passes, a barrier, and the merges. It knows no backend, so a new kernel set or backend reuses the record order. |
| `prompt_cache.zig` | A prefix-keyed, byte-budgeted LRU of model-state snapshots in two tiers, RAM and disk. A hit restores the forward state instead of re-running prefill, and `worthwhile` gates admission on the measured prefill against the measured restore. |
| `vulkan_ctx.zig` | GPU builds only. `Context`, the single owning handle for the Vulkan instance, physical device, logical device, compute queue, and command pool. It also owns the one staging command buffer that every one-shot submit resets. It also holds the device-selection ranking and the wave32 subgroup pin. It reads the device buffer limits once at open, and `Options` can replace the reported storage range with the value of `$ZIGFERENCER_VK_STORAGE_RANGE`. `guardKvLayout` refuses a KV layout whose region breaks one of those limits before any allocation, and its message names the widest window that fits. |
| `vulkan_runner.zig` | GPU builds only. `Runner`, the heap-allocated bundle of a `Context` and the ten per-op kernels. Heap allocation fixes the `Context` address that every kernel captures at `init`. |
| `llama_fused.zig` | GPU builds only. `LlamaFusedRunner` records the whole Llama forward pass for one token into one command buffer and submits once. Weights, KV, and scratch stay resident in VRAM, and the descriptor sets bind at `init`. The KV lives in a `kernels.KvStore`, one region per layer and tensor. |
| `qwen_fused.zig` | GPU builds only. `QwenFusedRunner`, the Qwen3.5 counterpart. It adds `prefillBatch` over the chunk-parallel GDN kernels and a multi-slot `stepBatch`, plus hidden-row entry points that read activation rows a caller already computed. |
| `llama_hybrid.zig` | GPU builds only. `LlamaHybridRunner` splits one Llama forward pass. The CPU runs layers `[0..split)` over the mmap weights. The GPU runs the suffix plus the final norm and the lm_head. It records its own GPU layer bodies, dispatch-identical to the fused runner at `split == 0`. |
| `qwen_hybrid.zig` | GPU builds only. `QwenHybridRunner` splits the Qwen3.5 forward pass the same way, but it composes the two shipped runners instead of copying a layer body. `QwenRunner.initPrefix` builds the host prefix and the hidden-row entry points of `QwenFusedRunner` take the crossing rows. |
| `decode_ring.zig` | GPU builds only. `DecodeRing`, the decode command buffers of one fused or hybrid runner. It holds two buffers, one persistent fence, and the `core/record_ring.zig` state. A step submits the buffer that holds its key, or records it first. While the GPU runs, the step records the next position into the other buffer, then waits on the fence. Every other submit of the runner invalidates the ring first. |
| `gpu_profile.zig` | GPU builds only. The timestamps of one command buffer. Every GPU command buffer of both families holds one `GpuProfile`: each decode ring buffer, the batch and prefill buffer of each runner. The one-shot buffer holds its two slots in the `Context`. The level comes from `$ZIGFERENCER_TRACE_GPU`. At `off` the record path emits no command, `submit` writes a start and an end, and `phases` writes every `mark`. `collect` reads a buffer back after the wait its path already does, and `Readout.report` prints the per-`Phase` table of `--profile-decode` and `--profile-prefill`. |

## How the pieces fit

A runner is the core. It owns its `KvCache` (Llama) or its per-slot recurrent slabs plus full-attention KV (Qwen). It resolves every weight once at `init` through `tensor_lookup`. It exposes one `stepBatch` over `StepRow` rows.
`step_row.zig` defines `StepRow` and every runner takes it from there, so
the six forward passes share one batched-step value. `parallel.zig` sits under the
runner's matmuls: the runner hands it the build-selected linear kernel and a
FLOPs estimate, and it either calls the kernel inline or fans out. It also sits under the CPU attention of both runners. The runner hands it the part kernels. It runs the units of one row inline or fans them out, then merges.

`session.zig` wraps a runner. It groups physical KV slots into conversations and
turns `feed` and `decodeBatch` into batched `stepBatch` calls. Sampling happens
there against each session's own RNG, so logits never leave this directory.
`generate.zig` is the other driver, for the single-sequence CLI path.

```
  generate.decode        Sessions.decodeBatch / feed
  (CLI: one row, sink)   (serve: many rows, per-session RNG)
        |                          |
        +------------+-------------+
                     v
              runner.stepBatch  (StepRow rows)
                     |  batched q8_0 projections
                     v
          parallel.linearBatch  --(under the threshold)--> kernel inline
                     |
            +--------+---------+
            v                  v
        KvCache slab      GDN recurrent state   (the state the runner owns)
            ^
            +-- prompt_cache restores it, sampler.greedy feeds the next id back
```

On a GPU build `vulkan_ctx.Context` is the owning handle and
`vulkan_runner.Runner` heap-allocates it beside the kernel bundle, so a captured
`*const Context` never dangles. `llama_fused` and `qwen_fused` record a whole
token into one command buffer on top of that. The two hybrids reach that path differently. `llama_hybrid` records its own GPU layer bodies, dispatch-identical to the fused runner's at `split == 0`. `qwen_hybrid` calls the fused runner's hidden-row entry points instead. In both, the residual stream crosses the host
and device boundary once per call.

`--gpu-layers` picks the placement on a fused-GPU build. `0` selects the CPU runner, the full layer count selects the fused runner, and a count in between selects the hybrid. Without the flag the build's whole-model fused runner runs. A
hybrid pins its GPU layers from the top of the stack, so the final norm and the
lm_head always sit on the device.

## Rules

- **Static allocation after init.** The runtime sizes the KV slab, the recurrent state, the scratch, and the device buffers once. It reads the `ModelCard` dims and the plan value. Nothing grows per step. The ceilings live in
  `core/limits.zig`, and the loader rejects a model that exceeds one. The
  context is the exception: the loader caps it at the engine bound and logs it.
- **A runner takes its plan as a value.** `KvPlan` carries the slot count, the window, and the named byte reservations. `init` reads them and derives no shape of its own. The two hybrids take `PlacementPlan` beside it,
  which names the layer split and gives each tier its own KV reservation. A
  runner cannot outgrow the plan it was built with.
- **Batched output equals solo output, bit for bit on the CPU.** A row in a
  `stepBatch` call sees exactly the inputs of its single-row call. The runner
  writes every row's K and V before any row attends, and same-slot rows arrive
  adjacent and ascending, so batch order is time order. The serve equivalence
  tests pin this.
- **Causality holds by construction.** Row `i` attends over `0..=pos_i`. No mask
  is patched in afterward.
- **Greedy tokens are equal across backends.** `sampler.greedy` is pure argmax,
  so the `bench-model` token hash gates every backend. The sampled path makes no
  bit-identity promise, because it goes through f32 `exp`. The gate stays greedy.
- **The decode stop is asymmetric, and the asymmetry is a value.** EOS and a full
  cache stop before the final step. A token-budget stop commits that token and
  surfaces it as pending, so the next turn prepends it. The rule lives once in
  `generate.decode` and once as the pending-token rule of `session.zig`.
- **A GPU kernel needs a stable `Context` address.** `vulkan_runner.Runner.init`
  heap-allocates the bundle, which makes the guarantee structural. The fused
  runners rely on it.
- **The state crosses as bytes, never as a handle.** The snapshot `blob` a runner
  writes is opaque to `prompt_cache` and to the serve residency. A hybrid blob is
  one walk in whole-model order, the host prefix first and the device suffix
  after, so the byte order matches what a single-tier runner writes.

## The runner surface

`Sessions(RunnerT)` names no interface type. It reaches a runner through a
duck-typed surface, and `serve/residency.zig` reaches the swap hooks through
`Sessions`. The table is that surface. Every row comes from a call site. A runner that satisfies the table composes with the session pool, the serve residency, and the CLI decode loop with no further work.

| Declaration | Shape | The consumer that reads it | Carried by |
| --- | --- | --- | --- |
| `stepBatch` | `fn (*Self, []const StepRow) !void` | `session.advance`, the one row-assembly path under `feed` and `decodeBatch` | every runner |
| `step` | `fn (*Self, u32, u32) ![]f32` | `generate.decode`, and the `run` prefill loop of a runner without `chunked_prefill` | every runner |
| `chunked_prefill` | `bool` | `step_row.chunkedPrefill`, which picks the chunked prefill loop in `run_cmd.prefillAndStream` and `bench_model_exec.doOneTurn` | the Llama fused runner, both hybrid runners, and the two CPU runners in a build without GPU ops. Absent or false, the prefill loop calls `step` per token. The Qwen fused runner has its own `run` loop |
| `prefillBatch` | `fn (*Self, []const u32, u32) ![]f32` | `run_cmd.prefillAndStream`, `run_cmd.prefillCachedAndStream`, and `bench_model_exec.prefillChunks`, which pass chunks of slot 0 at consecutive positions | every runner: the two CPU runners, both fused runners, and both hybrid runners |
| `prefill_batch_max` | `u32` | the same two loops, which cut the prompt into chunks of at most this many rows | every runner |
| `maxStepRows` | `fn (*const Self) u32` | `Sessions.init` sizes the row scratch with it. `cli/boot/serve_boot` clamps the prefill chunk to it | every runner |
| `logits` | `[]f32` | `Sessions.init` derives the vocab stride from its length. `advance` and `copyFinalLogits` read the compact rows | every runner |
| `n_slots` | `u32` | `Sessions.init` sizes the session table. `Residency.init` reads it as the physical slot count | every runner |
| `max_seq` | `u32` | `feed`, `advance`, and `Residency.capacityRemaining` gate the window on it | every runner |
| `logitRowsMax` | `fn (*const Self) u32` | `Sessions.logitRows` | a runner that holds fewer compact logits rows than KV slots. Absent, the pool reads one row per slot |
| `has_recurrent_state` | `bool` | the compile-time check in `session.zig` | a family whose slot holds non-positional state. The three Qwen runners declare it, and the declaration makes the two hooks below required by name |
| `resetSlotState` | `fn (*Self, u32) void` | `Sessions.create`, `Residency.placeEntry` | a runner with non-positional state. KV needs no scrub on reuse, a recurrent slab does |
| `cloneSlotState` | `fn (*Self, u32, u32, u32) void` | `Sessions.clone`, `Residency.clone` | a runner that forks a conversation. Its presence lights `can_clone` |
| `evictSlotBytesFor` | `fn (*const Self, u32) u64` | `Sessions.snapshotBytesFor`, `Residency.evictEntry`, the RAM ledger | a runner that serves |
| `evictSlot` | `fn (*Self, u32, u32, []u8) !void` | `Residency.evictEntry`, `Sessions.snapshotPrefix` | a runner that serves. Its presence alone lights `can_snapshot` |
| `restoreSlot` | `fn (*Self, u32, u32, []const u8) !void` | `Residency.placeEntry`, `Sessions.restorePrefix` | a runner that serves |

`Self` is the runner's own type, and every method takes a pointer to it. Each
runner family declares its own error set, so the table states the payload a
method yields and never the set. The last three rows are one unit: the pools test
for `evictSlot` alone, so the sizing hook and the restore hook must arrive with
it.

`Sessions(RunnerT)` checks the rows it reads itself when a pool type is built. Every build that names a pool, test fakes included, fails to compile on a drifted runner. The test `runner surface: the session seam holds for every
runner in this build` in `session.zig` pins the wider table, `step` included. It
instantiates every row against each runner type the build knows: the two CPU runners always, and the four GPU runners on a
`vulkan_fused` build. It also checks that the two fused runners and the two CPU runners read a `KvPlan`. It checks that the two hybrids read a `PlacementPlan`. It checks that the marker and the Llama family's lack of one both hold. A drifting runner
therefore fails to compile, not to serve.

## Where to start

1. `kv_cache.zig`: the state shape the forward pass addresses.
2. `step_row.zig`, then `llama_runner.zig`: the batch row every runner steps, then how the ops compose into a batched forward pass.
3. `tensor_lookup.zig` and `sampler.zig`: the small shared datapaths the runner uses.
4. `parallel.zig`, then `generate.zig`: the matmul fan-out, then the single-sequence decode loop and its `sink` boundary.
5. `session.zig`: slots, turns, the batched tick, and the runner surface it pins.
6. `qwen_runner.zig` beside `ops/gdn.zig`, then `vulkan_ctx.zig` to `vulkan_runner.zig` to `llama_fused.zig` to `qwen_fused.zig`: the second architecture, then the GPU path from the context outward.
