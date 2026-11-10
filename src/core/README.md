# Core

This directory holds the tensor view, the quant geometry, and the static engine limits. 
They also include the two memory planners, the device KV layout, the split geometry of decode attention, 
the host hardware card, what each backend needs from the target, the model table, and the chat wraps. No
module here owns engine state, and what crosses its edges is a plain immutable
struct. Four files touch the outside and take an `Io` for it. `log.zig` writes
the diagnostics to stderr, `span.zig` writes the trace file, `hwinfo.zig` reads
the host details from `/proc`, and `registry.locate` checks a path on disk.

## Files

| File | Holds |
| --- | --- |
| `tensor.zig` | `Tensor`, the byte view that crosses every op boundary. |
| `dtype.zig` | The GGUF types, their block geometry, and the KV element types. |
| `limits.zig` | The static engine limits. |
| `kv_budget.zig` | The two memory planners. |
| `kv_index.zig` | The index algebra of the device KV cache. |
| `attn_split.zig` | The split geometry of GPU attention. |
| `attn_cpu.zig` | The part geometry of CPU attention. |
| `op_requires.zig` | What a backend needs from the target, and its tolerance. |
| `hwinfo.zig` | The host hardware card. |
| `registry.zig` | The table of known models and the paths of the model store. |
| `chat.zig` | The chat wrap of each model family. |
| `log.zig` | The process log, gated at run time. |
| `span.zig` | The trace sites, the per-thread span logs, and the flush. |
| `trace_file.zig` | The layout of the trace: names, tracks, slices, flows, and args. |
| `perfetto.zig` | The protobuf encoder and the nesting walk. |
| `gpu_record.zig` | The GPU record and the host-clock mapping of its ticks. |
| `layer_op.zig` | The one list of op names, and the keys that a span and a GPU mark time. |
| `time.zig` | Elapsed-time arithmetic. |
| `step_stats.zig` | The timers of a decode step. |
| `record_ring.zig` | The state of the decode record ring. |
| `ring_fsm.zig` | The slot state machine of the upload ring. |
| `prop.zig` | The property test driver. |
| `test_fixtures.zig` | The path of a test model. |
| [`tokenizer/`](./tokenizer/README.md) | The BPE encoder and decoder. |

## How the pieces fit

Most files here define a value that the other layers read, so the traffic flows outward. These are
the major values:

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| `Tensor` | `tensor.zig` | A `GgmlType`, a shape, and a borrowed byte view | `io/gguf/loader` | `ops/`, `ops_vulkan/`, `runtime/` |
| `Model` | `registry.zig` | The paths, a `chat.Kind`, and the default `KvDtype` | The static table | `cli/`, `io/` |
| `KvPlan` | `kv_budget.zig` | The slots, the window, and the VRAM and RAM spend | `kvPlan` | `engine/`, `runtime/`, `cli/` |
| `LayerSplit` | `kv_budget.zig` | A layer split and its spend | `planLayerSplit` | `engine/`, the hybrid runners |
| `Layout` | `kv_index.zig` | The shape of the KV regions | The GPU runners | `runtime/`, `ops_vulkan/` |
| `AttnPlan` | `attn_split.zig` | The split rows of one step | `attn_split` | `runtime/attn_record` |
| `OpRequires` | `op_requires.zig` | A target and a `Tolerance` | Each backend | `cli/info` |
| `HwCard` | `hwinfo.zig` | The CPU, RAM, SIMD, and GPU of the host | `hwinfo` | `op_requires`, `cli/info` |

A model file becomes tensors:

```
  registry.Model --path--> io/gguf/loader --> Tensor per weight --> ops/, ops_vulkan/, runtime/
        |
        +--chat.Kind--> chat.zig wraps the prompt
```

A byte budget becomes the shape of the KV cache:

```
  engine/resolve       the byte budgets
  runtime/kv_cost      KvCostVector         --> kv_budget --> KvPlan, LayerSplit
  limits, dtype        the ceilings, KvDtype                       |
                                                                   v
                       limits.BufferLimits  --> kv_index.Layout, attn_split.AttnPlan
                                                                   |
                                                                   v
                                                 the GPU runners, ops_vulkan/kernels
```

Inside the directory, `Tensor` reads `dtype` and `limits`, `Model` reads `chat` and `dtype`, and
`OpRequires` reads `hwinfo`. `ring_fsm` faces one consumer, `ops_vulkan/stream_ring`, and
`record_ring` faces `runtime/decode_ring`. `span` reads `trace_file` and `gpu_record`, and
`trace_file` reads `perfetto` and `gpu_record`. `span`, `trace_file`, and `gpu_record` read the op names of `layer_op`.

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
