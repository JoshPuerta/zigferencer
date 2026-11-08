# Runtime

This directory holds the control units of the engine. Each module owns mutable state or sequences
the pure ops of `ops/` and `core/`. That state is the KV slabs, the recurrent slabs, the scratch
buffers, the decode position, and the GPU command-buffer recording. The values that cross its edges
carry no storage. A `Loaded` model comes in, and token ids go out.

## Files

| File | Holds |
| --- | --- |
| `llama_runner.zig` | `LlamaRunner`, the CPU forward pass of the Llama family. |
| `qwen_runner.zig` | `QwenRunner`, the CPU forward pass of the Qwen3.5 hybrid, and the coherence reference. |
| `llama_fused.zig` | `LlamaFusedRunner`, the whole-model GPU runner of the Llama family. |
| `qwen_fused.zig` | `QwenFusedRunner`, the whole-model GPU runner of Qwen3.5. |
| `llama_hybrid.zig` | `LlamaHybridRunner`, a CPU prefix of layers and a GPU suffix. |
| `qwen_hybrid.zig` | `QwenHybridRunner`, the same split, composed from the two Qwen runners. |
| `step_row.zig` | `StepRow`, the batch row every runner steps. |
| `session.zig` | `Sessions(RunnerT)`, the conversation slots over the KV slots of a runner. |
| `kv_cache.zig` | `KvCache`, the static KV slab of a CPU runner. |
| `kv_cost.zig` | `costVector`, the per-layer KV price of a model card. |
| `prompt_cache.zig` | `PromptCache`, the two-tier store of state snapshots. |
| `sampler.zig` | The greedy and the seeded sampling of one logits row. |
| `generate.zig` | The single-sequence decode loop and its `sink` boundary. |
| `parallel.zig` | The threading `Policy`, the row-parallel matmul, and CPU attention dispatch. |
| `pool.zig` | `Pool`, the fork-join pool of the CPU ops. |
| `linear_dispatch.zig` | `linearBatchDispatch`, the batched linear by weight dtype. |
| `tensor_lookup.zig` | The lookup of a weight by name over a `TensorRef` list. |
| `cpu_profile.zig` | The per-op CPU decode timer. |
| `attn_record.zig` | The attention recorder of the fused and hybrid runners. |
| `decode_ring.zig` | `DecodeRing`, the two decode command buffers of a GPU runner. |
| `gpu_profile.zig` | `GpuProfile`, the timestamps of one command buffer. |
| `vulkan_ctx.zig` | `Context`, the owning handle of the Vulkan device. |
| `vulkan_runner.zig` | `Runner`, the `Context` and the per-op kernel bundle. |
| `qwen_gpu_guard.zig` | The load gate of the Qwen3.5 GPU kernels. |
| `test_fakes.zig` | The shared test windows and the canned-logits fakes. |

## How the pieces fit

A loaded model becomes a runner, a runner becomes a session pool, and the serving layer sits on top.
The runtime owns the middle of that chain.

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| `Loaded` | `io/gguf/loader` | A `ModelCard`, a `TensorRef` list, and the mapped file | `loader.load` | Every runner, `engine/family` |
| `KvPlan`, `LayerSplit` | `core/kv_budget` | The slots, the window, and the layer split | `engine/resolve` | Every runner `init` |
| `StepRow` | `step_row.zig` | One `(slot, token, pos)` row of a batch | `Sessions`, `generate` | Every runner `stepBatch` |
| `KvCache` | `kv_cache.zig` | One flat K and V slab per slot | `LlamaRunner`, `QwenRunner` | The CPU runners |
| `Sessions` | `session.zig` | The slots, the positions, the RNG, and the sampling params | `cli/boot/serve_boot`, `dev/` tools | `serve/residency`, `serve/scheduler` |
| `PromptCache` | `prompt_cache.zig` | State snapshots keyed by token prefix | `cli/run_cmd`, `serve/prefix_manager` | The same two |
| `Pool` | `pool.zig` | The workers of the CPU ops | `cli/run_cmd`, `cli/boot/serve_boot` | `parallel`, the CPU runners |
| `Context` | `vulkan_ctx.zig` | The Vulkan device, queue, and command pool | `vulkan_runner.Runner`, `cli/boot/boot_resources` | Every GPU kernel |

The runners form a grid of three placements for two families. `engine/family` names the six.

| Placement | Llama | Qwen3.5 |
| --- | --- | --- |
| CPU | `LlamaRunner` | `QwenRunner` |
| GPU, whole model | `LlamaFusedRunner` | `QwenFusedRunner` |
| CPU prefix, GPU suffix | `LlamaHybridRunner` | `QwenHybridRunner` |

A model file reaches the serving layer through this chain.

```
  io/gguf/loader.Loaded  +  core/kv_budget.KvPlan (LayerSplit for a hybrid)
          |
          v
  runner.init            LlamaRunner | QwenRunner | *FusedRunner | *HybridRunner
          |   owns the KV and the recurrent state, takes StepRow rows in stepBatch
          v
  Sessions(Runner)       the conversations, the RNG, the sampling
          |
          +--> serve/residency.Residency   logical sessions over the physical slots
          |            |
          |            v
          |    serve/scheduler             one composed tick
          +--> generate.decode             the single-sequence CLI path
```

A snapshot leaves the runner as bytes and returns the same way.

```
  runner.evictSlot / snapshotPrefix --blob--> Residency (RAM ledger)
                                        \---> PromptCache (RAM tier, disk tier)
  runner.restoreSlot / restorePrefix <--blob-- the same two
```

On a GPU build, `Context` heads the stack. `Runner` heap-allocates it beside the kernels of
`ops_vulkan/kernels.zig`. A fused runner records a whole token through `attn_record` into the
buffers of its `DecodeRing`, and each buffer holds one `GpuProfile`. A hybrid composes a CPU prefix
with that path. `llama_hybrid` records its own GPU layer bodies. `qwen_hybrid` calls the hidden-row
entry points of `QwenFusedRunner`.

`parallel` and `pool` serve the CPU runners and the CPU half of a hybrid. `step_row` gives the six
runners one batch value. The edge to `serve/` is one-way: the scheduler and the residency reach a
runner only through the `Sessions` surface in "The runner surface".

## Rules

- **Static allocation after init.** The runtime sizes the KV slab, the recurrent state, the scratch,
  and the device buffers once. It reads the `ModelCard` dims and the plan value. Nothing grows per
  step. The ceilings live in `core/limits.zig`, and the loader rejects a model that exceeds one. The
  context is the exception: the loader caps it at the engine bound and logs it.
- **A runner takes its plan as a value.** `KvPlan` carries the slot count, the window, and the named
  byte reservations. `init` reads them and derives no shape of its own. The two hybrids take
  `LayerSplit` beside it, which names the layer split and gives each tier its own KV reservation.
  A runner cannot outgrow the plan it was built with.
- **Batched output equals solo output, bit for bit on the CPU.** A `stepBatch` row sees exactly the
  inputs of its single-row call. The runner writes every row's K and V before any row
  attends. Same-slot rows arrive adjacent and ascending, so batch order is time order. The serve
  equivalence tests pin this.
- **Causality holds by construction.** Row `i` attends over `0..=pos_i`. No mask is patched in
  afterward.
- **Greedy tokens are equal across backends.** `sampler.greedy` is pure argmax, so the `bench-model`
  token hash gates every backend. The sampled path makes no bit-identity promise, because it goes
  through f32 `exp`. The gate stays greedy.
- **The decode stop is asymmetric, and the asymmetry is a value.** EOS and a full cache stop before
  the final step. A token-budget stop commits that token and surfaces it as pending, so the next
  turn prepends it. The rule lives once in `generate.decode` and once as the pending-token rule of
  `session.zig`.
- **A GPU kernel needs a stable `Context` address.** `vulkan_runner.Runner.init` heap-allocates the
  bundle, which makes the guarantee structural. The fused runners rely on it.
- **The state crosses as bytes, never as a handle.** The snapshot `blob` a runner writes is opaque
  to `prompt_cache` and to the serve residency. A hybrid blob is one walk in whole-model order. The
  host prefix comes first and the device suffix after, so the byte order matches what a single-tier
  runner writes.

## The runner surface

`Sessions(RunnerT)` names no interface type. It reaches a runner through a duck-typed surface, and
`serve/residency.zig` reaches the swap hooks through `Sessions`. The table is that surface. Every
row comes from a call site. A runner that satisfies the table composes with the session pool, the
serve residency, and the CLI decode loop with no further work.

| Declaration | Shape | The consumer that reads it | Carried by |
| --- | --- | --- | --- |
| `stepBatch` | `fn (*Self, []const StepRow) !void` | `session.advance`, the one row-assembly path under `feed` and `decodeBatch` | every runner |
| `step` | `fn (*Self, u32, u32) ![]f32` | `generate.decode`, and the `run` prefill loop of a runner without `chunked_prefill` | every runner |
| `chunked_prefill` | `bool` | `step_row.chunkedPrefill`, which picks the chunked prefill loop in `run_cmd.prefillAndStream` and `bench_model_exec.prefill` | the Llama fused runner, both hybrid runners, and the two CPU runners in a build without GPU ops. Absent or false, the prefill loop calls `step` per token. The Qwen fused runner has its own `run` loop |
| `prefillBatch` | `fn (*Self, []const u32, u32) ![]f32` | `run_cmd.prefillAndStream`, `run_cmd.prefillCachedAndStream`, and `bench_model_exec.prefillChunks`, which pass chunks of slot 0 at consecutive positions | every runner: the two CPU runners, both fused runners, and both hybrid runners |
| `prefill_batch_max` | `u32` | the same two loops, which cut the prompt into chunks of at most this many rows | every runner |
| `maxStepRows` | `fn (*const Self) u32` | `Sessions.init` sizes the row scratch with it. `cli/boot/serve_boot` clamps the prefill chunk to it | every runner |
| `logits` | `[]f32` | `Sessions.init` derives the vocab stride from its length. `advance` and `copyFinalLogits` read the compact rows | every runner |
| `n_slots` | `u32` | `Sessions.init` sizes the session table. `Residency.init` reads it as the physical slot count | every runner |
| `max_seq` | `u32` | `feed`, `advance`, and `Residency.capacityRemaining` gate the window on it | every runner |
| `logitRowsMax` | `fn (*const Self) u32` | `Sessions.logitRows` | a runner that holds fewer compact logits rows than KV slots. Absent, the pool reads one row per slot |
| `has_recurrent_state` | `bool` | the compile-time check in `session.zig` | a family whose slot holds non-positional state. The three Qwen runners declare it, and the declaration makes the two hooks below required by name |
| `resetSlotState` | `fn (*Self, u32) void` | `Sessions.create`, `Residency.bringResident` | a runner with non-positional state. KV needs no scrub on reuse, a recurrent slab does |
| `cloneSlotState` | `fn (*Self, u32, u32, u32) void` | `Sessions.clone`, `Residency.clone` | a runner that forks a conversation. Its presence lights `can_clone` |
| `evictSlotBytesFor` | `fn (*const Self, u32) u64` | `Sessions.snapshotBytesFor`, `Residency.evictEntry`, the RAM ledger | a runner that serves |
| `evictSlot` | `fn (*Self, u32, u32, []u8) !void` | `Residency.evictEntry`, `Sessions.snapshotPrefix` | a runner that serves. Its presence alone lights `can_snapshot` |
| `restoreSlot` | `fn (*Self, u32, u32, []const u8) !void` | `Residency.bringResident`, `Sessions.restorePrefix` | a runner that serves |

`Self` is the runner's own type, and every method takes a pointer to it. Each runner family declares
its own error set, so the table states the payload a method yields and never the set. The last three
rows are one unit: the pools test for `evictSlot` alone, so the sizing hook and the restore hook
must arrive with it.

`Sessions(RunnerT)` checks the rows it reads when a pool type is built. Every build that names a
pool, test fakes included, fails to compile on a drifted runner. The runner surface test in
`session.zig` pins the wider table, `step` included. It instantiates every row against each runner
type of the build: the two CPU runners always, and the four GPU runners on a `vulkan_fused` build.
It checks that the fused and CPU runners read a `KvPlan` and that the hybrids read a
`LayerSplit`. It checks the recurrent-state marker on the Qwen runners and its absence on the
Llama family. A drifting runner therefore fails to compile, not to serve.
