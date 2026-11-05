# src/core

This directory holds the values the rest of the engine is written in, and the
pure helpers that read or build them. Those values are the tensor view, the quant
geometry, and the static engine ceilings. They also include the two memory planners, the
device KV layout, the split geometry of decode attention, the host hardware card, what each
op variant needs from the target, the model table, and the chat wraps. The layer is the
datapath end of the split that [docs/architecture.md](../../docs/architecture.md) describes. No
module here owns engine state, and what crosses its edges is a plain immutable
struct. Four files touch the outside and take an `Io` for it. `log.zig` writes
the diagnostics to stderr, `span.zig` writes the trace file, `hwinfo.zig` reads
the host details from `/proc`, and `registry.locate` checks a path on disk.

## Files

| File | Role |
| --- | --- |
| `tensor.zig` | `Tensor`, the immutable rank-4 byte view that crosses every op boundary. `data` is a borrow, the rank is fixed at `limits.max_n_dims`, and `assertValid` checks the rank, the positive dims, and the byte length. |
| `dtype.zig` | `GgmlType` and the per-type block geometry (`hasBlockGeometry`, `blockSize`, `bytesPerBlock`, `bytesForNumel`). It also holds `KvDtype`, whose `rowBytes` is the single product every KV size site derives from. |
| `limits.zig` | The static engine ceilings, one home per bound. They cover the rank, the layer, vocabulary, batch, slot, and session caps, the request caps of the HTTP surface, and the usable VRAM and RAM fractions. The window rules live here: the engine context bound (`max_model_ctx`), the model ceiling (`contextCeiling`), the window granule of 256 positions, and the KV byte budget of one session (2 GiB for `run`, 128 MiB for serve). The swap stage chunk (`swap_stage_chunk_bytes`) and the headroom of a printed RAM fit (`ram_fit_headroom_bytes`) live here too. So do the device buffer limits (`BufferLimits`, `bufferFits`, the `ZIGFERENCER_VK_STORAGE_RANGE` override) and the bounds on one KV row and one KV region. The shader geometry that the host and the GLSL read as one number lives here, and the build passes each value to `glslc` as a define: the GPU attention split length (`ATTN_SPLIT`), its head width and query heads (`ATTN_D_MAX`, `ATTN_QUERIES`), the Q8_0 and Q4_0 batch tiles, the GEMM tile, the GDN scan geometry (`CSUB`, `W`, `DH`), and the subgroup shared-memory capacity. `gdnScanGroups` and `gdnScanCell` give the grid and the column owner of the GDN scan. The CPU attention geometry (`attn_cpu_tile`, `attn_cpu_parts`) and the split scratch rows (`attn_split_rows`) live here as well. The doc of each constant holds its facts. |
| `attn_split.zig` | The geometry of split-depth decode attention. `splitCount` and `usesSplit` follow a row length alone, so a row gets the same splits at every window, batch, and slot. `Scratch` lays out the partial records, one per row, query head, and split, and `fits` checks the scratch against the device buffer limits through `limits.bufferFits`. It is pure: no allocation, no device, and no I/O. |
| `attn_cpu.zig` | The geometry of CPU attention. `partCount` and `partRange` follow a row length alone, so a row gets the same parts and the same bits at every thread count, window, batch, and slot. `recordOffset`, `recordStride`, and `partialLen` lay out the partial buffer. It holds one record per query head and part, and the records of one head sit next to each other in part order. It is pure: no allocation, no threads, and no I/O. |
| `kv_budget.zig` | Two planners over one usable-VRAM rule. `kvPlan` turns a byte budget into `n_slots x window` plus named reservations, and it refuses a named window past the model ceiling. `costModel` prices one slot from the model dimensions. `defaultWindow` turns a session byte budget into the default window, and `fitWindow` shrinks a default window to the free VRAM. `placementPlan` prices a weight placement, the host layer prefix against the GPU layer suffix, from a per-layer cost vector. `quoteRam` prices the host RAM of every serve placement at one split and funds the blob need of an unset `--kv-ram`, or refuses. `ChunkWalk` and `clipCopies` cut a swap blob into the chunks one stage moves. Both planners are pure arithmetic. |
| `kv_index.zig` | The index algebra of the device KV cache. The cache holds one region per layer and tensor, and each region holds the rows of every slot in slot, position, element order. `Layout` computes every offset, stride, and size. `fits` checks one region against the device buffer limits. `maxWindowForLimits` and `maxSlotsForLimits` solve for the widest window and the most slots that pass, and `pack` places the regions into allocations. It is pure: no allocation, no device, and no I/O. |
| `op_requires.zig` | `OpRequires`: the target an op variant needs (arch, SIMD level, FMA, runtime) and its numeric `Tolerance` against `naive`. `info` scores it against the host card through `isMetByHost`. The build does not read it: `simdBackendSupportedBy` in `build.zig` mirrors what the two SIMD variants declare, and the mirror and the declaration must agree. |
| `hwinfo.zig` | `HwCard`: the arch and OS strings, the CPU model, the core count, the total RAM, a bool-per-feature SIMD card, and a best-effort GPU presence flag that links no GPU library. |
| `registry.zig` | The static `Model` table, the `$ZIGFERENCER_MODELS_DIR` resolution, and the path helpers. Each entry carries the on-disk path, the download URL, the chat family, and the KV dtype the model is verified at. An entry can carry a `context_limit`, which lowers the model ceiling for a file that cannot support its declared context. A Qwen entry also carries the thinking mode of its model file. It also carries the card sampling of each thinking mode, which the serve launcher sends in `hello_ack`. A comptime check holds every card inside the wire guards. |
| `chat.zig` | The per-family chat wraps. `Kind` is a tagged union: none, SmolLM2 ChatML, Llama 3, and Qwen ChatML, which carries its thinking mode. A raw message goes in and tokenizer-ready text comes out, for one turn, a continuation, a system prompt, or a whole message list. `resolveThinking` applies a `--thinking` flag to a kind. `wrapPairUserAlloc` renders the user part of a kept (user, assistant) pair at its place in the conversation, with the closed think tail. |
| `log.zig` | Process logging. It routes `std.log` to stderr with a `+ms` prefix, gated at run time by `$ZIGFERENCER_LOG` and its per-scope grammar. The three hot scopes compile in only under `-Dtrace`. |
| `span.zig` | The span recorder. `$ZIGFERENCER_TRACE_OUT` turns it on, and with it off a span site costs one atomic load and one branch. Each thread pushes closed spans into one fixed ring, and a full ring drops and counts. The GPU runners hand over one `GpuRecord` per submit. The writer emits Chrome Trace Event JSON with one GPU queue track per runner kind. `place` puts each record on its own host times. The writer also emits a folded file of the GPU phases. It also holds the GPU timestamp level of the process (`$ZIGFERENCER_TRACE_GPU`). See [docs/profiling.md](../../docs/profiling.md). |
| `time.zig` | Elapsed-time arithmetic over `std.Io.Clock.Timestamp` pairs: `elapsedNs` and `msBetween`. One definition for the whole tree, so the math cannot drift between callers. |
| `step_stats.zig` | Fixed-space timers for the decode step. `Stat` keeps a sum, a max, and a count. `DecodeHost` names the six host steps: record, submit, speculate, wait, logits copy, and argmax, plus the hit, miss, and waste counts of the record ring. The fused runners and the decode loop fill them, and `bench-model` prints them. |
| `record_ring.zig` | The state of the decode record ring: two buffers, each `free`, `recorded k`, or `pending k`. `prepare` picks a hit or a miss. Every transition asserts the ring invariants: no record into a pending buffer, at most one pending buffer, and a hit key equal to the wanted key. It also holds `DecodeKey`, the key of a decode recording, whose GPU timestamp level keeps a recording at one level from a replay at another. No Vulkan and no allocation. |
| `ring_fsm.zig` | The slot state machine of the upload ring in `ops_vulkan/stream_ring.zig`, checked at compile time. It holds the four states, the four operations, the `next` table, and a replayable anchor trace. No Vulkan and no allocation. |
| `prop.zig` | `forAll`, the property driver of the tests. Case i has size min(i, `max_size`), so the first failing case is a small one. Each case seeds its own generator from the call site, `std.testing.random_seed`, and the case index, so one case replays alone. A failure prints the case, the size, the seed, the site, and the `zig build test --seed` command that replays it. |
| `test_fixtures.zig` | `path`, the model file of a test by registry name, through `registry.locate`. `decide` holds the rule: a found file runs, and a missing file skips the test only when `$ZIGFERENCER_MODELS_DIR` is unset. With the variable set, a missing file fails the test with `error.FixtureMissing`. The root module re-exports it only for the tests under `tests/`. |
| [`tokenizer/`](./tokenizer/README.md) | The byte-level BPE encoder and decoder over the `Vocab` value. |

## How the pieces fit

These files call each other rarely. They define the vocabulary the other layers
speak, and most of the traffic flows outward. Six edges run inside the
directory:

```
  dtype.GgmlType  --field of-->    tensor.Tensor
  limits.max_n_dims --read by-->   tensor.Tensor (rank 4) and kv_budget (caps)
  hwinfo.HwCard   --scored by-->   op_requires.isMetByHost
  chat.Kind       --field of-->    registry.Model
  dtype.KvDtype   --priced by-->   kv_budget.costModel and kv_index.Layout
  limits.BufferLimits --read by--> kv_index.fits, its window and slot bounds, and attn_split.fits
```

Outward, the loader fills one `Tensor` per `TensorRef` with the `dtype` geometry
and checks the `limits` ceilings. `info` builds a `HwCard` and labels the
`OpRequires` of each variant against it. The CLI resolves a `Model` from
`registry`, applies `chat.wrapAlloc` to the prompt, resolves the window, and
hands a `KvPlan` or a `PlacementPlan` to a runner constructor. The fused runners
and `runtime/vulkan_ctx.zig` read the device KV layout from `kv_index`. The
fused runners and `ops_vulkan/kernels.zig` read the split geometry of decode
attention from `attn_split`.
`ring_fsm` faces one consumer, the upload ring, whose GPU type mirrors the table
here.

## Rules

- `Tensor` owns nothing and frees nothing. The storage, the map, the arena, the
  scratch and KV slabs, lives in the control layer.
- `dtype.hasBlockGeometry` names the wire types the two geometry functions can
  size. The loader asks it first and refuses the rest with a typed error. A file the engine
  cannot read fails at the trust boundary and never returns a wrong size in silence.
  `blockSize` and `bytesPerBlock` panic past that gate, which is a caller mistake, not
  untrusted input.
- The loader raises a typed error on a `limits` breach and never truncates in
  silence. The context is the one bound it caps: a file that declares more than
  `max_model_ctx` loads at that bound, and the log names both numbers. A
  relaxation is one edit here, and every site reads the same constant.
- A named window never shrinks. `kvPlan` refuses a named window past the model
  ceiling with `error.WindowExceedsCeiling`. Only a default window shrinks to fit
  the device, and the caller logs each cut with both numbers.
- Both planners are pure: no allocation, no device query, no I/O. One arithmetic
  therefore serves the plan and the preflight, and a test can replay it.
- `kvPlan` refuses a budget that cannot host one decode row with
  `error.KvVramBudgetTooSmall`. Where the slot ceiling cuts the count the budget
  bought, `KvPlan.slots_clamped` carries the witness, so the caller names the
  clamp instead of hiding it.
- `OpRequires.tolerance` is the numeric agreement against `naive`. `.bit_exact`
  means the output bytes match. `.relative = eps` means the largest per-element
  relative error stays at or under `eps`. `toleranceSatisfied` is the single
  decision function, and a claim tighter than the variant delivers fails the gate.
- A non-null `registry.locate` result is a guarantee that the file exists at that
  path, not a guess. The repository-local `models/` directory wins over the
  `$ZIGFERENCER_MODELS_DIR` store.
- The chat wraps are hand-coded per family, with no template interpreter and no
  dependency. They pair-validate against the special tokens the BPE encoder
  matches greedily.
- `ring_fsm.next` returns null for an operation the table does not name, and the
  caller asserts. The table and the anchor trace run at comptime, so drift fails
  the build instead of a test run.

## Where to start

1. `tensor.zig`, then `dtype.zig`: the value and the quant geometry the rest of the engine is written in.
2. `limits.zig`: the ceilings the loader enforces.
3. `op_requires.zig` with `hwinfo.zig`: what a variant needs, and the card it is scored against.
4. `kv_budget.zig`, then `kv_index.zig`: how a byte budget becomes slots, a window, and a layer split, then how those slots lay out in device regions.
5. `registry.zig`, then `chat.zig`: how a named model resolves to a path and a prompt wrap.
6. `log.zig`, `time.zig`, and `ring_fsm.zig`: the three small shared files. `log.zig` and `time.zig` hold one definition for the whole tree. `ring_fsm.zig` faces one consumer, the upload ring.
