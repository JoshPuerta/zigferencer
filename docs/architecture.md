# Architecture

zigferencer turns a prompt into generated text in five stages. It downloads a
model, loads it, tokenizes the prompt, runs one forward pass per token, and
samples the next token. The serve layer runs the same forward pass for many
conversations at once. This document describes how the code organizes those
stages, where the boundaries between them lie, and which module owns what. Every
source file also carries a `//!` header with its own summary, and every top-level
source directory carries a `README.md` with one row per file. This document sits
one level above both.

## 1. The organizing idea: control and datapath

The code separates into two kinds of module:

- **Datapath** modules hold pure functions that transform values: bytes into typed
  metadata, activations into activations. They hold no state and do no I/O, so a
  test can call them in isolation. The datapath lives in `core/`, `ops/`, the
  shaders, and `io/gguf/parser.zig`.
- **Control** modules hold state (the KV cache, the recurrent state of a
  Gated-DeltaNet layer, scratch buffers, the decode position, GPU command buffers)
  and sequence the pure functions. Control lives in `runtime/` and `serve/`.

The same split appears elsewhere as control unit and datapath in hardware design, and
as functional core and imperative shell in software design. Each module is one or the
other, and that classification predicts whether it carries state.

## 2. The pipeline

```
  fetch                  load                                 run
  -----                  ----                                 ---
  Hugging Face URL       GGUF on disk                         prompt text
      |                      |                                    |
      v                      v  mmap, read-only                   v  chat.wrapAlloc (--chat)
  models/<repo>/x.gguf   io/gguf: parser -> loader           bpe.encodeAlloc -> [token ids]
                             |  ModelCard + []TensorRef           |
                             +----------------+-------------------+
                                              v
                    runtime: one runner (CPU, fused GPU, or a CPU-prefix + GPU-suffix hybrid)
                       prefill the ids, then loop:
                         stepBatch(rows) --ops.*--> logits --sampler.greedy--> id
                                              |
                                              v  bpe.decodeAlloc
                                          text -> stdout

  serve / serve-openai
  --------------------
  client frames or HTTP requests -> adapter -> Request queue -> scheduler tick
     -> Sessions.feed / decodeBatch (one stepBatch for every active conversation)
     -> OutEvent per connection -> adapter -> client
```

Prefill and decode share one code path. `stepBatch` takes a batch of `(slot, token,
pos)` rows. Prefill runs the prompt tokens through it and keeps only the final
logits. Decode feeds each sampled token back as the next row. `run` samples greedily,
so its token sequence is reproducible, and a hash of that sequence is the correctness
signal across backends (section 9).

## 3. Module map

```
src/
  main.zig                entry point of zigferencer, module re-exports for tests/, the test discovery block
  dev_main.zig            entry point of zigferencer-dev: the release commands plus the developer commands
  cli/                    argv -> one handler per subcommand (the imperative edge)
    command.zig             static release and dev {name, summary, handler} tables and the Context value
    args.zig                the struct flag parser and help generator, one usage-error exit path
    flags.zig               EngineFlags and ServeFlags, the knobs several commands share
    select.zig              optional <pub_name> -> registry model, TTY picker
    style.zig               ANSI palette, empty strings off a TTY
    prompt.zig              interactive stdin reads, degrade to "no" off a TTY
    fetch_cmd.zig           download a GGUF (registry name or org/repo/file.gguf)
    list_cmd.zig            list registered models and their presence on disk
    inspect_cmd.zig         print the ModelCard of a downloaded GGUF
    info_cmd.zig            host CPU/RAM/SIMD card, the backends this binary links, the support matrix
    arch_matrix.zig         model architecture x op-backend support matrix, a section of info
    tokens_cmd.zig          print a prompt's token ids in llama.cpp format
    run_cmd.zig             load -> tokenize -> prefill -> decode loop -> stream to stdout
    serve_cmd.zig           binary-protocol serving: flags and the launcher over serve_common
    serve_openai_cmd.zig    OpenAI HTTP serving: flags and the launcher over serve_common
    serve_common.zig        the shared engine builder: plans, runner selection, both launchers
    serve_demo_cmd.zig      drive a serve process end to end (spawn or connect), also a smoke test
    client_cmd.zig          chat with a running serve over the binary wire, streamed or whole turns
    client_transcript.zig   the chat transcript and the pure compaction decisions of the client
    gpu_selftest_cmd.zig    bit-check one GPU kernel against its CPU reference
    bench_model_cmd.zig     end-to-end TTFT, ITL, token hash. Flag parsing and dispatch
    bench_model_common.zig  shared Options and result values (imports no sibling)
    bench_model_exec.zig    drive a loaded model through one prompt and its decode, generic runner
    bench_model_llama.zig   the llama placement head and the CPU decode timer of both families
    bench_model_qwen.zig    the qwen modes: placement head, fused-GPU bench and its profilers
    bench_model_placement.zig  the family-independent --gpu-layers placement body
    bench_model_report.zig  human table, single-line JSON
    bench_serve_cmd.zig     aggregate tok/s and ITL tails against batch size over loopback
    bench_swap_cmd.zig      tiered-KV swap cost: evict and restore one slot against a tick
    bench_decode_host_cmd.zig  host legs of one decode step: record, resubmit, sync floor, idle gap
  engine/                 the boot decisions as pure values
    flags.zig               EngineFlags, the knobs of one engine boot
    resolve.zig             flags + registry entry + card + host facts -> window, dtype, plan, RAM quote, or a refusal
    device_arm.zig          the device arms: fused fit, serve budget, hybrid split, from the facts of one context
  core/                   boundary values and pure helpers (datapath)
    tensor.zig              Tensor: the immutable byte view that crosses every op seam
    dtype.zig               GgmlType, quant block geometry, KvDtype
    limits.zig              the static engine ceilings, one home per bound
    kv_budget.zig           the KV budget planner (KvPlan) and the weight placement planner (PlacementPlan)
    kv_index.zig            the device KV layout: one region per layer and tensor, packed into allocations
    attn_split.zig          the split geometry of GPU attention: split count, scratch record offsets, and the AttnPlan planner with its record budget groups
    attn_cpu.zig            the part geometry of CPU attention: tile count, part ranges, and partial record offsets
    op_requires.zig         per-variant target requirements and numeric tolerance
    hwinfo.zig              host hardware card (arch, cores, RAM, SIMD flags, GPU presence)
    registry.zig            the static model table, $ZIGFERENCER_MODELS_DIR resolution
    chat.zig                per-family chat templates (SmolLM2, Llama 3, Qwen)
    log.zig                 stderr logging with a +ms prefix, gated at runtime by $ZIGFERENCER_LOG
    span.zig                the span recorder: per-thread rings, Chrome trace JSON, the GPU queue track
    time.zig                elapsed-time helpers over std.Io.Clock.Timestamp
    prop.zig                test support: the seeded property driver
    test_fixtures.zig       test support: model file lookup for the tests
    step_stats.zig          sum, max, count timers for the host legs of a decode step
    record_ring.zig         the two-buffer state of the decode record ring
    ring_fsm.zig            the slot state machine of the upload ring, checked at compile time
    tokenizer/
      vocab.zig             Vocab built from the GGUF tokenizer tables
      bpe.zig               byte-level BPE encode and decode over a Vocab
  io/                     the I/O shell
    download.zig            std.http.Client -> disk over HTTPS, zero dependencies
    mmap.zig                read-only whole-file memory map, prefault or lazy
    gguf/
      parser.zig            bytes -> Parsed (header, KV pairs, tensor info), pure
      loader.zig            mmap + parser -> ModelCard + []TensorRef, bounds checked
  ops/                    CPU op variants, each one matches naive
    naive.zig               scalar f32 reference and correctness oracle, the full op set
    simd.zig                @Vector(8) AVX2+FMA linear kernels and vector attention, the rest re-exported
    simd_pf.zig             register-blocked Q8_0 microkernel with software prefetch (the AVX2 default)
    attention_simd.zig      the vector part kernel of CPU attention, shared by both SIMD variants
    attention_ref.zig       test support: the f64 attention reference and the shared checks
    exp_simd.zig            the vector exponential of CPU attention and the sampler
    gdn.zig                 Gated-DeltaNet step and chunked scan (Qwen3.5 layers)
  runtime/                the control units, the only layer that owns state
    step_row.zig            StepRow, the batch row every runner steps, and its row rule
    attn_record.zig         the attention recorder: walks an AttnPlan through the attention kit of the build
    linear_dispatch.zig     batched-linear seam, kernel by weight dtype, shared by both CPU runners
    kv_cost.zig             the KV price of a model card, one entry per layer
    test_fakes.zig          test support: the canned-logits runner fakes
    llama_runner.zig        Llama-family forward pass over KV slots
    qwen_runner.zig         Qwen3.5 hybrid forward pass (GDN and full-attention layers)
    kv_cache.zig            static KV slab tagged with its element type
    session.zig             Sessions(RunnerT): per-conversation slots, feed and decodeBatch
    sampler.zig             greedy argmax, seeded sampling with penalties, top-k, temperature, min-p, and top-p
    generate.zig            the greedy decode loop that run and bench-model share
    tensor_lookup.zig       tensor-by-name lookup shared by every runner
    parallel.zig            the thread policy and the matmul, attention, and GDN fan-out
    pool.zig                the persistent fork-join pool of the CPU ops
    cpu_profile.zig         the per-op CPU decode timer of both runner families
    prompt_cache.zig        two-tier (RAM and disk) cache of end-of-prompt model state
    vulkan_ctx.zig          Vulkan device, queue, command pool, buffer limits    (GPU builds only)
    vulkan_runner.zig       the per-op kernel bundle at a fixed address          (GPU builds only)
    llama_fused.zig         one command buffer per token, Llama                  (GPU builds only)
    qwen_fused.zig          one command buffer per token, Qwen3.5, multi-slot    (GPU builds only)
    llama_hybrid.zig        CPU layer prefix + GPU layer suffix, Llama           (GPU builds only)
    qwen_hybrid.zig         CPU layer prefix + GPU layer suffix, Qwen3.5         (GPU builds only)
    gpu_profile.zig         the timestamps of one GPU command buffer, per level  (GPU builds only)
    decode_ring.zig         two decode command buffers and one fence per runner  (GPU builds only)
  ops_vulkan/             the GPU op backend                                    (GPU builds only)
    wrap.zig                backs the CPU op API with kernels, one submit per op
    kernels.zig             one pipeline struct per compute shader, plus the buffer helpers
    stream_ring.zig         double-buffered host-to-device upload ring with its own fences
    shaders/*.comp          27 GLSL sources, compiled to 31 SPIR-V blobs at build time
  serve/                  the serving layer (section 10)
    protocol.zig            the binary wire format: framed messages, pure encode and decode
    event.zig               Request, OutEvent, Codec: the values between the serve threads
    scheduler.zig           the control unit: per-session state machine, prefill and decode batching
    prefill_chunk.zig       the prefill chunk rule: how many prompt rows one pass takes
    zmtp.zig                the ZMTP 3.1 codec for a ZeroMQ DEALER peer
    engine.zig              ServeEngine(Engine, Codec, Adapter): engine thread, queues, sink, lifecycle
    residency.zig           tiered KV: logical sessions over VRAM slots and RAM blobs, LRU
    prefix_manager.zig      the serve prefix cache: entries, counters, boot warm-up
    server.zig              the binary transport adapter, two tasks per connection
    client.zig              the reference binary client (tests, bench-serve, the port a client copies)
    openai/
      http_server.zig       the OpenAI HTTP adapter over std.http.Server, one task per connection
      map.zig               JSON request -> Request values, pure, the trust boundary
      types.zig             the OpenAI request and response structs
      raw_codec.zig         tokenize without a chat wrap, plus model metadata

tests/                    26 integration tests against real GGUF files, plus 2 helpers (a test skips on an absent file only without $ZIGFERENCER_MODELS_DIR)
tools/                    benches and scripts
  bench.zig                 op micro-benchmarks: per-kernel GFLOP/s and relative error against naive
  bench_util.zig            timing, error, and table helpers
  bench_matmul.zig          matmul sections
  bench_layer.zig           per-layer elementwise and attention sections
  bench_gdn.zig             Gated-DeltaNet sections
  bench_parallel.zig        pool dispatch cost against std.Io.Group, DRAM bandwidth ceiling
  bench_transfer.zig        host-device crossing latency and staged-copy bandwidth
  simd_probe.zig            symbol-export probe for assembly inspection
  bench_model_all.sh        cross-backend token-hash gate over the CPU variants, plus per-op vulkan under GPU=1
  bench_llama_hybrid_gate.sh  five-way llama token-hash gate: CPU, fused, and three hybrid splits
  bench_deep_gate.sh       deep token-hash gate: 8064+64 on CPU and fused against the hashes in the script
  bench_ttft_sweep.sh       GPU TTFT and ITL against prefill length
  bench_threads_sweep.sh    CPU thread-scaling sweep -> JSONL
  bench_sweep_report.py     renders the sweep JSONL (python3 standard library)
  bench-openai.sh           curl bench of the OpenAI endpoint
  demo-openai.sh            curl demo of the OpenAI endpoint
  openai-common.sh          the shared spawn and READY block of the two scripts above
  menu.sh                   the Makefile prompts, status lines, and shared recipe bodies
  simd-report.sh            per-op SIMD width and assembly inspector
  profile.sh                the perf profiler driver behind `make profile`
  profile_convert.py        converts perf script text for profile.sh (python3 standard library)
  profile-calls/            the function lists that `profile.sh calls` probes
```

## 4. Values at the boundaries

The boundaries between layers carry plain immutable structs, not stateful objects. A
test can build a value without setup. A log can print the value in full. Threads can
share the value without locks.

| Value | Defined in | Crosses the boundary between |
| --- | --- | --- |
| `Tensor` | `core/tensor.zig` | loader and runtime -> ops (a borrowed view, and ops never allocate) |
| `ModelCard` | `io/gguf/loader.zig` | loader -> runtime (dims, RoPE, EOS, GDN parameters) |
| `TensorRef` | `io/gguf/loader.zig` | loader -> runtime (a name and a `Tensor` into the map) |
| `Vocab` | `core/tokenizer/vocab.zig` | loader -> bpe (token, merge, and byte tables) |
| `HwCard` | `core/hwinfo.zig` | hwinfo -> info and op_requires |
| `OpRequires` | `core/op_requires.zig` | each op variant -> build.zig and info |
| `StepRow` | `runtime/step_row.zig` | sessions -> any runner (one `(slot, token, pos)` of a batched step) |
| `KvPlan` | `core/kv_budget.zig` | the CLI planner -> any runner constructor (slots, window, byte reservations) |
| `PlacementPlan` | `core/kv_budget.zig` | the CLI planner -> a hybrid runner (the layer split and the KV of each tier) |
| wire frames | `serve/protocol.zig` | client <-> server (a 20-byte header and a bounded payload) |
| `Request`, `OutEvent` | `serve/event.zig` | adapter tasks -> engine loop -> adapter tasks |

The control layer owns the storage (the mmap, the arena, the scratch and KV slabs,
the device buffers), and that storage lives outside these values. A `Tensor` is a
borrow: `{dtype, shape, stride, data: []const u8}`. It owns nothing and frees nothing.

## 5. Configuration axes

A small set of axes fixes what the engine does. The axes bind at different times.
The build compiles two of them into the binary. The model fixes some at load. A flag
sets the rest per run.

| Axis | Set by | Binds | Values |
| --- | --- | --- | --- |
| Runtime | `-Druntime` | build | `cpu` (default), `vulkan` |
| Op backend | `-Dops-backend` | build | `auto` (default), `naive`, `simd`, `simd_pf`, `vulkan`, `vulkan_fused` |
| Optimize | `-Doptimize` | build | `Debug`, `ReleaseSafe`, `ReleaseFast` |
| Hot tracing | `-Dtrace` | build | compiles the `layer`, `dispatch`, `token` log scopes. On for Debug, off for release |
| Vulkan registry | `-Dvk-registry` | build | the path of the Vulkan registry XML (`vk.xml`). The default is the Debian and Ubuntu `libvulkan-dev` location. The build reads it under `-Druntime=vulkan` alone, so another distribution needs this flag |
| GPU attention kit | `-Dattn-kernel` | build | `split` (the only kit). A CPU build ignores it |
| Test threads | `-Dtest-threads` | build | the pool size of the runner tests whose thread count is incidental. The default is 4 |
| Test filter | `-Dtest-filter` | build | run only the tests whose name holds this string. Repeatable. A filtered-out GPU test touches no device |
| Model architecture | the GGUF | load | Llama family, Qwen3.5 hybrid |
| Weight dtype | the GGUF | load | Q8_0, Q4_0, Q4_1, Q6_K. Activations are f32 throughout |
| Model ceiling | the GGUF, the registry | load | the declared context, capped at 262144 positions and by the `context_limit` of the registry entry |
| KV dtype | `--kv-dtype` | run | `f32`, `f16`. The registry sets the default per model |
| KV window | `--max-seq` (`run`, `bench-model`, `serve`, `serve-openai`, `bench-serve`) | run | KV positions per session, from 1 to the model ceiling. The default comes from a byte budget per session (section 8) |
| Threads | `--threads`, `--min-parallel-flops`, `--spin-us` | run | `0` auto (the physical cores, at most 8), `1` serial, N threads up to 64, the FLOPs floor of a fan-out (default 250,000), and the worker spin before sleep (default 50 microseconds) |
| GPU layers | `--gpu-layers` | run | on the fused-GPU build: `0` the CPU runner, all the fused runner, a mid split the hybrid, `auto` the most layers the free VRAM funds |
| Sampling | per-session serve parameters | run | `run` is greedy only. Serve clients set temperature, top-k, top-p, min-p, a presence penalty in [-2, 2], a repetition penalty in (0, 2], and a seed per session. The chain runs the penalties, top-k, temperature, min-p, top-p, then the draw. The presence penalty counts the ids that the current turn accepted. The repetition penalty counts every id fed to the session. Both follow the vLLM rules. With top-k 0, the min-p and top-p cuts are exact over the full vocab. The frequency penalty is a wire field, and the server refuses a non-zero value. `hello_ack` carries the model card sampling of the think mode, and `client --sampling model\|greedy` picks the card or greedy (default greedy). `serve-openai --sampling model\|greedy` sets the fields that a request omits (default model) |
| Thinking | `--thinking` (`run --chat`, `serve`, `serve-openai`, `client`), `--thinking-budget` (`serve`, `serve-openai`, `client`) | run | `auto` (the default of the model file, and on `client` no switch), `on`, `off`. The budget is the think tokens before the server forces the close, and `0` is unbounded. A budget at or past the KV window fails at boot. A `generate` frame can set the think switch and the think budget for one turn. On the OpenAI front the same two controls are `chat_template_kwargs.enable_thinking` and `thinking_token_budget`. |
| Serve shape | `--max-sessions`, `--max-batch`, `--prefill-chunk` | run | logical sessions, decode width, prefill chunk (unset = full runner row width when no row decodes) |
| Serve policy | `--max-connections`, `--sessions-per-connection`, `--allow-shutdown`, `--resume-ttl` | run | live connections, one connection's session share, whether a wire client can stop the server, and the longest lease that a client can keep past a drop (0 is off, at most one hour) |
| Memory budgets | `--kv-vram`, `--vram-slots`, `--kv-ram`, `--prefix-disk` | run | VRAM KV budget or pinned slots, RAM ledger for evicted KV, the prefix disk tier |
| Tracing | `$ZIGFERENCER_LOG` (Makefile `LOG=`) | run | `off`, `err`, `warn`, `info`, `debug`, or per scope: `gguf,runner`, `parallel=debug` |
| Span trace | `$ZIGFERENCER_TRACE_OUT`, `$ZIGFERENCER_TRACE_SECONDS`, `$ZIGFERENCER_TRACE_GPU`, `$ZIGFERENCER_TRACE_SPANS` | run | the trace file, the capture window, the GPU timestamp level, and the capacity of every trace ring (1 to 4194304, default 16384). `docs/profiling.md` prices the rings |
| Storage-buffer range | `$ZIGFERENCER_VK_STORAGE_RANGE` | run | GPU builds only. A byte count from 1 to 4294967295 that replaces the storage-buffer range the device reports. `info` prints both values |

The build resolves the build-time axes once and burns them into the binary. The
decode loop holds no SIMD probe and no backend branch. The engine reads the
load-time axes from the GGUF, so one binary runs either architecture and every dtype
its path supports.

### The two flags that pick the code path

`-Dops-backend` selects the op kernels. `auto` picks the strongest variant the
target CPU supports: `simd_pf` on AVX2 with FMA, else `naive`
(`resolveOpsBackend` in `build.zig`). The build does not read the `OpRequires` of
a variant. `simdBackendSupportedBy` in `build.zig` mirrors what the two SIMD
variants declare, and `op_requires.isMetByHost` scores the same axes against the
running host for `info`. The mirror and the declaration must agree, and a change
to one needs the same change to the other. A name for an unmet variant fails the
build.
`-Dops-backend=simd` on a target without AVX2 panics with a message at build time, so
the build never produces a binary that cannot run. The choice becomes a set of
`build_options` booleans that the runner imports directly. There is no runtime
dispatch:

```
build.zig  --resolveOpsBackend-->  build_options.cpu_ops_simd_pf = true
                                          |
llama_runner.zig:  const ops = if (use_vulkan_ops) @import("../ops_vulkan/wrap.zig")
                               else if (cpu_ops_simd_pf) @import("../ops/simd_pf.zig")
                               else if (cpu_ops_simd)    @import("../ops/simd.zig")
                               else                      @import("../ops/naive.zig");
```

`-Druntime` selects `cpu` (the default) or `vulkan`. A `cpu` build depends on `std`
alone. A `vulkan` build adds the `vulkan-zig` binding generator as a dependency and
links `libvulkan` and `libc`. `libc` is needed because the Vulkan loader uses
thread-local storage. The build compiles the 27 `ops_vulkan/shaders/*.comp` sources to
31 SPIR-V blobs with `glslc` and embeds each blob with `@embedFile`. Four
attention shaders compile twice, once per KV element type, through a `-DKV_F16`
define. A `cpu` build touches none of that toolchain.

`-Druntime=vulkan` is the only setting that allows the two GPU op backends (`vulkan`,
`vulkan_fused`). `resolveOpsBackend` panics otherwise. That is the one cross-axis
rule.

When the build gets a GPU op backend, it also resolves a CPU variant beside it
(`simd_pf` or `naive`). That variant covers the ops the GPU path leaves on the CPU,
the CPU-side runners, and the CPU prefix of a hybrid runner.

A new CPU variant takes three steps:

1. Add `ops/<name>.zig`.
2. Give it a `pub const requires: OpRequires`.
3. Add a case to `resolveOpsBackend`.

The decode loop stays untouched.

### Makefile entry points

The `Makefile` wraps the common invocations, and `make help` lists them grouped.
`ZIG` is `zig` on `PATH`. `make <target> ZIG=/path/to/zig` overrides it. A `BACKEND=`
knob maps to build flags through `backend_flags` in `tools/menu.sh`. `vulkan` and
`vulkan_fused` expand to `-Dops-backend=<that> -Druntime=vulkan`. `auto` or an empty
value expands to `-Dops-backend=auto`, and any other value expands to
`-Dops-backend=<that>`.

The groups are build and quality (`build`, `release`, `test`, `fmt`, `check`, `check-targets`,
`clean`) and benchmarks (`bench`, `bench-gpu`, `bench-attn`, `bench-model`, `bench-serve`, `bench-swap`, `bench-decode-host`,
`bench-openai`, `bench-model-all`, `gate-llama-hybrid`, `gate-deep`, `bench-ttft-sweep`,
`bench-threads-sweep`). Then SIMD inspection (`simd-report`, `simd-list`,
`simd-asm`), profiling (`profile`), and inference (`run`, `run-all`, `chat`, `serve`, `demo`,
`demo-openai`). Then models and host (`info`, `list`, `fetch`, `inspect`), GPU
(`gpu`, `gpu-selftest`, `gpu-flush`), and `zig-version`.

Every build target installs both executables in one `zig build install dev`
call. The release binary is `$(BIN)`, `zig-out/bin/zigferencer`. The developer
binary is `$(DEV_BIN)`, `zig-out/bin/zigferencer-dev`. Every developer target
(the model benches, the gates, `demo`, `gpu-selftest`, and `profile`) runs
`$(DEV_BIN)`. The op micro-benchmark targets (`bench`, `bench-gpu`, and `bench-attn`)
build and run a third executable, `zigferencer-bench`. The release binary refuses a developer command.
The message names the developer binary.

When `MODEL=` is unset and stdin is a TTY, a model-consuming target prompts for a
model. A piped or CI invocation takes the default silently.

## 6. The forward passes

Every runner has one shape: a control unit that owns state and calls pure ops. Six
runners exist, because two model architectures cross three placements.

**The two model architectures** (the load-time axis):

- **Llama family** (`llama_runner.zig`) runs two blocks per layer. The first is a pre-norm
  attention block: RMSNorm, the Q/K/V projection, RoPE, the KV write, grouped-query
  attention, and the output projection. The runner adds that output back to the
  residual stream. The second is a pre-norm SwiGLU feed-forward block, added back
  the same way. The same code drives SmolLM2-135M and Llama-3.1-8B.
- **Qwen3.5 hybrid** (`qwen_runner.zig`) replaces three of every four attention
  blocks with Gated-DeltaNet (`ops/gdn.zig`). Gated-DeltaNet is linear attention. It
  keeps a fixed-size `d x d` recurrent state per head instead of a growing KV cache,
  so its cost per token does not grow with the sequence. Every fourth layer is full
  softmax attention with QK-norm and an output gate.

**The three placements** (the run-time axis on a fused-GPU build):

| Runner | Placement | Weights | KV |
| --- | --- | --- | --- |
| `LlamaRunner`, `QwenRunner` | CPU | the mmap | host slab |
| `LlamaFusedRunner`, `QwenFusedRunner` | GPU | resident in VRAM | VRAM slots |
| `LlamaHybridRunner`, `QwenHybridRunner` | CPU prefix `[0..split)`, GPU suffix `[split..n_layers)` | mmap for the prefix, VRAM for the suffix | each tier holds the KV of its own layers |

`--gpu-layers` picks the placement on a fused-GPU build. `0` selects the CPU runner,
the full layer count selects the fused runner, and a count in between selects the
hybrid. `auto` puts the most layers on the GPU that the free VRAM funds, and the
split it resolves selects the runner by the same rule. Without the flag, the build's
whole-model runner runs. Every arm checks the anonymous host RAM of its placement
against 7/8 of the available RAM, as section 8 states. A
hybrid pins its GPU layers from the top of the stack, so the final norm and the
lm_head always sit on the GPU. The residual stream crosses the host-device boundary
once per token. The Llama hybrid records its own GPU layer bodies. The Qwen hybrid
composes the two shipped runners through a prefix constructor and a suffix entry
point, so it records nothing of its own.

**Prefill and decode are one code path.** Every runner exposes `stepBatch` over
`StepRow` rows, and `step` is the one-row wrapper. Prefill runs the prompt tokens
through it and keeps only the final logits. Decode feeds each sampled token back as
the next row. A batch row gets exactly the inputs of its single-row call, so the
batched output is bit-identical to that row alone. The serve layer relies on this
property.

**The op API boundary.** A runner names a fixed op set and imports the module the
build resolved. The set is `linearF32Q80`, `rmsnorm`, `swiglu`, `residualAdd`,
`ropeInplace`, `embedLookup`, and the rest. Attention is the one
op whose names differ by path. A CPU module exports `attentionPartTyped` and
`attentionMerge` for the control unit in `parallel.zig`, and the per-op GPU module
exports `attentionSingle`. The three execution paths differ only in the module
behind those names:

```
        run / serve  -->  generate.decode | session.decodeBatch
                                   |  runner.stepBatch(StepRow[])
                                   v
   CPU build                 per-op GPU (=vulkan)            fused GPU (=vulkan_fused)
   ops = ops/simd_pf.zig      ops = ops_vulkan/wrap.zig       llama_fused.LlamaFusedRunner  (Llama)
   (or simd / naive)          one submit + waitIdle PER op    qwen_fused.QwenFusedRunner (Qwen)
        |                     heavy ops GPU, small ops CPU     llama_hybrid / qwen_hybrid (split)
        v                            |                               |
   parallel.linearBatch       submission overhead dominates    weights and KV resident in VRAM,
   (row-parallel on the pool,                                  descriptor sets pre-bound,
    under a FLOPs floor)                                       one submit per token
```

- **CPU** (`-Druntime=cpu`, any CPU op backend): the runner calls the resolved
  `ops.*` directly. Matmuls go through `parallel.zig` (section 7). Attention goes
  through `parallel.zig` too. A row splits into at most `limits.attn_cpu_parts`
  parts of whole `limits.attn_cpu_tile` tiles. The unit of thread parallelism is
  one KV group over one part, and a merge adds the parts in part order. The parts
  follow the row length alone, so one depth gives one result bit for bit at any
  thread count. The Gated-DeltaNet heads of the Qwen layers fan out over the same
  pool. The only attention buffer is one record per head and part, and the window
  does not change its size.
- **Per-op GPU** (`-Dops-backend=vulkan`): `ops_vulkan/wrap.zig` backs the same op
  API with kernels, one submit and `waitIdle` per op. Small or fragmented ops
  (embedding lookup, per-head RoPE, f32 norms) stay on the CPU. Submission overhead
  dominates. This path is a correctness reference and a benchmark target, not the
  fast path. `run` and `gpu-selftest` use it. `serve` refuses it.
- **Fused GPU** (`-Dops-backend=vulkan_fused`): the fused runners record the whole
  forward pass of one token into one command buffer and submit once. Weights, KV,
  and scratch stay resident in VRAM, and the descriptor sets are bound once. This is
  the fast GPU path. The hybrids run their suffix through the same recorded bodies.
  Attention runs resident kernels up to `limits.attn_split_len` positions. A decode row uses one workgroup per query head. A prefill chunk uses one batch run for its resident prefix. A deeper decode or prefill row splits its walk across workgroups. A merge pass combines the splits in a fixed order, so one depth gives one result bit for bit. Each runner plans the attention rows of a step or a
  chunk once (`attn_split.AttnPlan`). The plan groups the split rows by the record
  budget of the scratch. `runtime/attn_record.zig` records the plan of each
  attention layer through the attention kit that `-Dattn-kernel` selects.

**Coverage.** One binary runs either architecture. The supported weight dtypes
differ by path, and the activations are f32 everywhere:

| Path | Llama | Qwen3.5 | Weight dtypes | KV dtypes |
| --- | --- | --- | --- | --- |
| CPU (`naive`, `simd`, `simd_pf`) | yes | yes | Q8_0, Q4_0, Q4_1, Q6_K | f32, f16 |
| Per-op GPU (`vulkan`) | yes | yes | Q8_0 only (others return `error.UnsupportedWeightDtype`) | f32 |
| Fused GPU (`vulkan_fused`) and the hybrids | yes | yes | Q8_0, Q4_0 | f32, f16 |

The KV dtype frames every K and V row of the Llama family and the full-attention
rows of Qwen3.5. The Gated-DeltaNet state stays f32. Within one dtype, the batched
and the solo run are bit-equal. Across dtypes and across backends the greedy tokens
are compared, never the bytes.

## 7. The CPU threading model

Decode on the CPU is a matrix-vector product per token: about two floating-point
operations per weight byte, so it is memory-bound. Prefill in `run` and
`bench-model` feeds the prompt through `stepBatch` in chunks of
`limits.cpu_prefill_chunk` rows. One weight read then serves every row of a chunk,
and only the last row runs the lm_head.

**The pool.** `pool.zig` holds a persistent fork-join pool. A command creates one
pool and passes it to the runner, and the calling thread is participant 0 of every
dispatch. A dispatch publishes a job and a new epoch word. Each participant runs its
own first unit, then claims units from a shared counter, and the caller returns when
every participant reports done. Between dispatches a worker spins on the epoch word
for a short budget, then sleeps on a futex. A dispatch takes no lock and allocates
nothing. It costs about half a microsecond at 8 participants, against about 50
microseconds for the same fan-out through `std.Io.Group`.

**The policy.** `parallel.Policy.forHost` sets three values. `--threads`,
`--min-parallel-flops`, and `--spin-us` override them, and `info` prints all
three. `Policy.parts` is the one place that turns them into the participant
count of an op.

- Threads: the physical cores from the sysfs topology, at most 8. Decode saturates
  the memory bandwidth of a dual-channel desktop between 4 and 8 threads, and
  prefill keeps gaining to 8. `--threads` raises the cap up to `limits.max_threads`.
  A fused GPU placement runs no CPU op, so its pool has one thread.
- The fan-out floor: an op under 250,000 FLOPs runs inline on the caller, with no
  pool access. One floor serves every op class. A sweep measured it best on a small
  model and on a 2 GB model.
- The spin budget of a worker: 50 microseconds.

**The units.** A matmul unit is a chunk of output rows, a multiple of 4, about four
chunks per participant. An attention unit is one KV group over one part of the row.
A Gated-DeltaNet unit is one value head, after a serial prologue
that runs the convolution and the key norms.

**Determinism.** The result of a unit does not depend on the thread that runs it. A
matmul row is one kernel call in the order of the kernel. An attention head writes
its own record for each part, and a merge adds the parts in part order on the
calling thread. A value head writes only its own state and output. So the token hash
holds across thread counts, the floor, the spin budget, and the claim order. Every
thread sweep doubles as a determinism gate (`tools/bench_threads_sweep.sh`). The test
suite runs every runner test whose thread count is incidental at `-Dtest-threads`
(default 4) with the floor at 0, and `make test TEST_THREADS=N` changes the count.

On the GPU paths the CPU is not the bottleneck. For an 8B model the host spends
under a tenth of a decode step on record, submit, the logits copy, and sampling.

## 8. Memory budgets

The runtime sizes every slab once, from a plan value, and never grows it per step.

**`KvPlan`** (`core/kv_budget.zig`) turns a byte budget into `n_slots x window` plus
named byte reservations. It is pure arithmetic, so the preflight and the plan agree,
and a test can replay it. With a VRAM budget (`--kv-vram`) the budget buys slots at
the resolved window. With a pinned slot count beside it (`--vram-slots`) the count
holds and the window follows the budget. Without a budget the requested structure
passes through at the named window or at the default window of the model.

**Windows.** The model ceiling is the largest window of a model: the context its
file declares, capped at 262144 positions (`limits.max_model_ctx`) and by its
registry `context_limit`. A file that declares more loads at 262144, and the loader
logs both numbers. A registry entry lowers the ceiling for a file that cannot
support its declared context. The loader refuses a file that declares a RoPE
scaling the engine does not apply.

Without `--max-seq`, a byte budget per session sets the default window. `run` and a
one-stream `bench-model` get 2 GiB. Each serve session and each stream of a
multi-stream bench gets 128 MiB. The default is the budget over the KV bytes of one
position. It rounds down to a multiple of 256 positions, never below 128, and the
ceiling caps it. An 8B model at f32 KV gets 8192 and 512, and a cheaper KV family
gets a deeper window.

On the fused-GPU build a default window can shrink, and each cut logs one line with
both windows. First it shrinks to the widest multiple of 256 positions whose KV
region passes the device buffer limits. On the whole-model fused path it then
shrinks to the widest window that the free VRAM funds. A named window never
shrinks. A `--max-seq` past the ceiling is a usage error. A named window that the
device cannot hold fails at the runner, and the message names the largest window
that fits.

**The device KV layout.** The fused runners and the device suffix of a hybrid hold
their KV in a `kernels.KvStore`, laid out by `core/kv_index.zig`. It holds one region
per layer and tensor, and each region holds the rows of every slot. The regions pack
into as few allocations as the device allows. One descriptor binds at most one
region, so the device limits bind one region and not the whole cache.

`vulkan_ctx.Context` reads three limits when it opens: the storage-buffer range, the
allocation size, and the binding offset alignment. A shader indexes KV with a signed
32-bit value, so one region also holds at most 2^31 elements. A driver can report the
128 MiB spec minimum range while it reads past it correctly.
`$ZIGFERENCER_VK_STORAGE_RANGE` then states the range an operator proved.
`Context.guardKvLayout` refuses a layout that breaks a limit before the first
allocation. The slots that a budget buys stop at the count that one region holds,
and a log line names the limit.

**`PlacementPlan`** prices a weight placement for the hybrids. The host layers form
the prefix and the GPU layers form the suffix. A per-layer cost vector prices each
span, so a family whose layers differ in price (Qwen3.5) is priced by span, not by
one product. The KV of a layer follows its executor. `split == n_layers` is the CPU
runner and `split == 0` is the fused runner, so the planner and the runners agree
on the degenerate cases.

**The VRAM preflight.** Before the first device allocation, a fused or hybrid runner
sums its planned footprint (weights, KV, recurrent state, logits, scratch). It refuses
a plan above 7/8 of the live device-local budget with `error.VramBudgetExceeded`. The
live budget is what `VK_EXT_memory_budget` reports free, not the heap size, because a
desktop shares the card. The message names the needed, free, and usable bytes, the
largest window that fits, and the flags that lower the plan. The eighth in reserve absorbs the upload staging buffer,
allocator fragmentation, and the driver's own overhead.

**The host RAM check.** Every arm of `serve`, `serve-openai`, `run`, and
`bench-model` quotes its anonymous host RAM through one function,
`kv_budget.quoteRam`, before it allocates a runner. The quote holds the host KV,
the two swap stages, the evicted KV blobs, and the prefix cache. The CPU arm is the
split at the layer count and the fused arm is the split at zero, so every arm prices
one vector. Over 7/8 of `MemAvailable` the boot refuses and names the numbers and
a window that fits. The check is exact, but the printed fit keeps 128 MiB of
headroom and whole granules of 256 positions, because free RAM moves between two
boots. A swap walks a blob in chunks of 256 MiB
(`limits.swap_stage_chunk_bytes`) through one stage per direction, so the stages
cost the same at any window.

**The RAM ledger.** With `--vram-slots` below `--max-sessions`, `serve/residency.zig`
holds the extra conversations as byte blobs in RAM and charges each blob to a
ledger. A swap holds the evicted blob and the restored blob at once, so the need is
one blob more than the sessions the slots cannot hold. An unset `--kv-ram` funds
that whole need, or the boot refuses with the largest `--max-seq` and the largest
`--max-sessions` that fit. A named `--kv-ram` stays as named, and a value below the
need logs one warning. An eviction that exceeds the ledger is refused, and the
scheduler parks that turn until a slot frees. A blob is the only copy of its
conversation, so backpressure is the only legal response. The prefix cache takes the
surplus of the same RAM budget after the blobs have their provable need. `--prefix-disk` adds a disk
tier under `.prompt_cache` (or `$ZIGFERENCER_CACHE_DIR`) whose entries survive a
restart.

## 9. The correctness model

The numeric guarantee has layers. Each layer localizes a divergence, so it does not
appear as garbage tokens many layers downstream.

- **`naive` is the in-repo oracle.** Every other op variant, SIMD or GPU, must match
  its output. The match is bit-for-bit where the variant keeps the reduction order.
  Otherwise the match is within the relative tolerance the variant declares in its
  `OpRequires`. `tools/bench.zig` prints the relative error per variant and fails the
  row on drift.
- **`gpu-selftest`** bit-checks one GPU kernel against its CPU reference on random
  inputs. It localizes a failure that the end-to-end hash cannot.
- **The `bench-model` token hash** is an FNV-1a over the generated token ids. Equality
  across thread counts is the determinism check. Equality across backends for one
  model is the cross-backend check. `tools/bench_model_all.sh` gates the CPU variants,
  and `tools/bench_llama_hybrid_gate.sh` gates the CPU runner, the fused runner, and three
  hybrid splits against one hash.
- **`tests/snapshot_smollm2_test.zig`** pins the first greedy token ids of a fixed
  prompt against the real GGUF. A regression anywhere in the stack surfaces here,
  which makes it the highest-leverage test in the suite.
- **The equivalence tests** (`llama_session_equivalence`, `qwen_session_equivalence`,
  `llama_vulkan_serve_equivalence`, `qwen_vulkan_serve_equivalence`,
  `llama_hybrid_residency`, `llama_hybrid_serve`) pin the batched-equals-solo
  invariant on every runner family.
- The external reference is the f32 path of llama.cpp on the same GGUF and prompt.
  The engine does not pursue bit-parity with PyTorch, because f32 reductions are not
  associative.

## 10. The serve layer

`zigferencer serve` turns the engine into a component that a host process spawns.
Conversations are sessions, and each session owns a KV slot whose position persists
across turns. Every tick advances up to `--max-batch` of them through one batched
forward pass. The weights then stream from memory once per tick instead of once per
session. The same pass carries the prefill chunk of a waiting turn in the spare rows
(continuous batching). A long prompt therefore never preempts the sessions that
decode. When no session decodes, a pass takes a full runner row width of prompt, so
a lone prompt prefills in ceil(N / width) passes. `--prefill-chunk 0` keeps the spare
rows always, and `--prefill-chunk N` takes N rows.

The wire is a framed binary protocol that a client implements in under a hundred
lines. `serve/protocol.zig` is the format and its documentation, at
`protocol_version` 3, and the server refuses every other version. Version 3
carries session leases (`session_keep`, `session_resume`, `serve --resume-ttl`)
and the per-turn sampling and think fields. `serve/client.zig` is the reference
implementation. The frames ride as ZMTP messages, so a ZeroMQ
DEALER socket in any language is a client. `src/serve/README.md` holds the procedure.

One engine, two connectors:

| Connector | Transport | Traffic it serves |
| --- | --- | --- |
| `serve` | framed binary protocol over ZMTP (ZeroMQ) | embedded interactive traffic: many concurrent, short, low-latency sessions that often share a prefilled prefix, under a tail-latency budget |
| `serve-openai` | OpenAI HTTP API (`/v1/chat/completions` with SSE, `/v1/models`, `/healthz`) | general interop, so existing clients, SDKs, and evaluation tools work unchanged |

Both connectors are thin maps onto one engine. `serve/engine.zig` owns the engine
thread, the bounded queues, the non-blocking sink, and the connection lifecycle. A
comptime `Adapter` supplies the per-connection task model and the mapping between a
wire format and the `Request` and `OutEvent` values. `server.zig` (`BinaryAdapter`)
runs two tasks per connection over `protocol.zig` frames. `openai/http_server.zig`
(`HttpAdapter`) runs one task per connection over `std.http.Server`. A third
transport is a third adapter, with no engine change.

When a server is ready, it prints one machine-parseable line to stdout: `READY tcp
<addr>` or `READY unix <path>`. Every diagnostic goes to stderr, so a spawning process
parses stdout for the port. Six commands exercise this layer:

| Command | What it does |
| --- | --- |
| `serve` | Load one model, bind a socket, print `READY`, run the engine loop over the binary protocol. |
| `serve-openai` | The same over the OpenAI HTTP API. `make demo-openai` curls it and `make bench-openai` measures it. |
| `serve-demo` | Spawn (or `--connect` to) a server and drive it: a streamed turn, follow-up turns that reuse the KV cache, one session cloned three ways, then three sessions batched and demultiplexed on one connection. |
| `client` | Connect to a running server over TCP or a unix socket and chat, on protocol version 3. Each typed line is one turn, tokens stream as they arrive, and the slash commands drive cancel, stats, ping, close, clone, compact, history, and quit. The token channel marks reasoning, which prints dim, and the think tags, which print nothing. `--sampling model\|greedy` picks the model card row or greedy that `hello_ack` sends for the think mode of the turns. `--thinking` and `--thinking-budget` set the think switch and the budget of every turn, and `--min-p` and `--repetition-penalty` set those sampling fields. `--reserve` sets the positions a compaction leaves free. The client compacts after a full window and after a refused turn. It closes the session, opens one with the original system prompt, and sends the kept turns with `session_append`. `--no-stream` prints each reply whole. |
| `bench-serve` | Start a fresh in-process server per batch size in `{1, 2, 4, 8}`, drive that many concurrent sessions, and report aggregate tok/s and ITL tails. |
| `bench-swap` | Time the eviction and the restore of one KV slot against a decode tick (fused-GPU build only). |
| `bench-decode-host` | Time the host legs of one decode step: record, replayed submit, sync floor, and GPU idle gap (fused-GPU build only). |

Three knobs bound the batch, and each one is distinct. `--max-sessions` is the
logical conversation count. Its cap is 8 on a CPU build and 128 on the fused-GPU
build when `--vram-slots` or `--kv-vram` enables the KV tier. `--vram-slots` is the
physical KV slot count in VRAM (fused-GPU only, up to 128). `--max-batch` is the number
of sessions one decode tick advances, at most 8.

The runner behind the pool follows section 6: the CPU runners, the fused runners, or
a hybrid when `--gpu-layers` names or resolves a mid split. The per-op GPU backend returns
`error.UnsupportedServeBackend`.

The decomposition repeats the shape of the whole project one level up. `protocol.zig`
and the runner's `stepBatch` are datapaths. `scheduler.zig` is the control unit that
owns every session's state machine. `engine.zig` is the imperative shell. The engine
thread never blocks on a client. A full output queue parks one event in the pending
cell of that session. The session leaves the batch until the draining task signals a
wakeup. `--max-connections` sets the live connections, 8 by default and 64 at most
(`core/limits.zig`). Each binary connection costs two worker threads and one payload
buffer of 20 bytes plus 64 KiB. The other bounds are one request queue of 64 entries
and one event queue of 256 entries per connection (`serve/engine.zig`).

When VRAM is the binding constraint, `residency.zig` adds a tier without a change to
that shape. `Sessions` stays a one-to-one pool over the physical KV slots of the
runner. `Residency(Sessions(RunnerT))` sits above it and owns the logical sessions. It
maps each one to a resident slot or to a RAM blob with an LRU clock. It presents the
same method surface as `Sessions`, so the scheduler only swaps its engine type.
Eviction is a byte-faithful whole-slot copy through the runner's `evictSlot` and
`restoreSlot`, so a restored conversation continues identically. Without
`--vram-slots` the residency is the identity.

The scheduler also owns a prefix cache (`prefix_manager.zig`). A turn that starts at
position zero restores the longest cached prefix of its tokens and prefills only the
suffix. A repeated system prompt therefore opens with zero forward passes. A
completed position-zero prefix is offered back as a new entry when the measured
restore beats the prefill it replaces. A repeatable `--preload FILE` warms the cache
at boot. Every cache path is best-effort: a refusal degrades to a cold prefill, and a
warm restore equals a cold prefill token for token.

The invariant under everything: a batched session produces exactly the tokens of that
session alone. On the CPU the tokens and the logits are bit-equal. Across backends the
greedy tokens are equal. The equivalence tests of section 9 pin it on every runner,
`tests/serve_integration_test.zig` observes it from the wire, and
`tests/openai_integration_test.zig` observes it over HTTP.

## 11. Strict core, relaxed edges

The numeric core holds to a strict bar: bounded loops, asserted preconditions, and
zero dependencies. That core is the forward pass, the ops, the KV cache, the GGUF
parser, and the bounds. The features around it (the CLI subcommands, the bench
harness, the caches) are looser. The boundary between the two is a boundary value, so
a relaxed feature cannot reach into the core and corrupt it.

## 12. Operating notes

- **Measure ReleaseFast binaries only.** A bare `zig build` selects Debug. Every make
  target that runs a model passes `-Doptimize=ReleaseFast`. A Debug binary runs the
  host layers about ten times slower while a GPU suffix stays fast, which reads like
  device trouble. Check the optimize mode before you diagnose a slowdown.
- **Run GPU tests one at a time.** `-Dtest-filter="<substring>"` runs one named test,
  and a filtered-out test touches no device. Read the `Build Summary` line of `zig build test -Druntime=vulkan --summary all`. On some drivers a test process exits abnormally after its tests have reported. The summary line carries the result.
- **Compile-only gates.** `zig build test-build`, `zig build bench-build`, and
  `zig build check`, each with `-Druntime=vulkan`, compile the GPU tests, the bench, and the binaries. None of them runs a kernel. A Vulkan build below
  `-Dops-backend=vulkan_fused` compiles the kernel gates alone, and the fused
  build compiles the runner gates as well. Use them on a host without a device or
  before a long GPU session.
- **Stay under the preflight.** On a 16 GB card plan about 12.5 GB of device memory at
  most. A plan that passes the 7/8 preflight can still lose the device near the
  ceiling under a translation-layer driver. After a device removal, end the process.
  `make gpu-flush` ends a lingering one.
- **Step through the code.** `make debug` and `make debug-test` build for lldb or
  gdb. [debugging.md](debugging.md) has the commands.
- **Repeated large runs.** Space large-model GPU runs apart and prefer `--runs 1` for
  them. Small models survive any workload.

## 13. Suggested reading order

1. `core/tensor.zig` and `core/dtype.zig`: the value and the quant geometry. The rest
   of the engine is written in terms of them.
2. `io/gguf/parser.zig` then `io/gguf/loader.zig`: bytes to a `ModelCard`.
3. `ops/naive.zig`: every op the forward pass needs, in scalar form.
4. `runtime/llama_runner.zig`: how those ops compose into a forward pass.
5. `cli/run_cmd.zig`: how the pieces wire into a command.

From there: `ops/simd_pf.zig` for the CPU performance work, and `ops/gdn.zig` with
`runtime/qwen_runner.zig` for the Qwen hybrid. Then `core/kv_budget.zig` for the plans,
`ops_vulkan/` with `runtime/llama_fused.zig` for the GPU path, and `serve/scheduler.zig`
for the serving layer. Each top-level source directory's `README.md` is the layer
between this map and the file headers.