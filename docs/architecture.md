# Architecture

This document explains how zigferencer works, from the build to a served token.

**Contents**

- [1. Glossary](#1-glossary)
- [2. Build approach](#2-build-approach)
- [3. From a prompt to text](#3-from-a-prompt-to-text)
- [4. Control and datapath](#4-control-and-datapath)
- [5. Code placement](#5-code-placement)
- [6. The core data types](#6-the-core-data-types)
- [7. The forward pass](#7-the-forward-pass)
- [8. CPU threads](#8-cpu-threads)
- [9. Memory](#9-memory)
- [10. Serving](#10-serving)
- [11. Correctness](#11-correctness)
- [12. What calls what](#12-what-calls-what)
- [13. Operating notes](#13-operating-notes)

## 1. Glossary

**The model**

| Term | Meaning |
| --- | --- |
| token | An integer id for a piece of text. The tokenizer turns text into tokens and tokens back into text. |
| GGUF | The model file format. One file holds the metadata and every weight tensor. |
| weights | The tensors of the model. A quantized weight stores blocks of 8-bit or 4-bit integers with one scale per block. |
| layer | One decoder layer of the model. A model is a stack of layers, and each layer reads the output of the layer before it. |
| forward pass | The computation of the model over a set of tokens. It ends in logits. |
| logits | One score for each entry of the vocabulary. The sampler picks the next token from them. |
| attention | The op that lets a token read all earlier positions. |
| KV cache | The keys and values that attention keeps for each earlier position, so that a new token does not compute them again. |
| position | The index of a token in the sequence of one session, counted from 0. Each attention layer keeps one K row and one V row for each position, and RoPE encodes the position into Q and K. A session keeps counting across its turns, so the second turn starts where the first turn ended. |
| Gated-DeltaNet (GDN) | A linear-attention layer of Qwen3.5. It keeps a state of fixed size instead of a KV cache that grows with each token. |
| convolution ring | The last inputs of the convolution of one GDN layer, `conv_kernel - 1` for each channel. It is part of the GDN state of a slot. |

**The engine**

| Term | Meaning |
| --- | --- |
| engine | The library: every directory under `src/` but `cli/` and `dev/`. The directory `src/resolve/` holds its boot decisions: it resolves the flags and the host resources into a setup or a refusal. In the serve layer, `ServeEngine` is the thread that owns the scheduler, and the type parameter `SessionPool` is the session pool under it. |
| op | One operation of the model: a matrix product, a norm, attention. `LayerOp` in `core/layer_op.zig` lists them. Every op span of a CPU trace and every GPU mark names an op of that list, or one pass of an op. The other spans name a stage, for example `sched_tick`. |
| kernel | One implementation of an op: a Zig function on the CPU, a compute shader on the GPU. |
| backend | The set of kernels that the build selects (`-Dops-backend`). |
| layer kernel set | The ten methods that a decoder layer calls: `norm`, `project`, `qkNorm`, `rope`, `kvCommit`, `attention`, `gate`, `residual`, `swiglu`, and `gdn`. `CpuLayerKernels` computes each one, and `FusedLayerKernels` records each one into a command buffer. A decoder layer file (`llama_layer`, `qwen_layer`) calls them in the order of its `forward`. |
| runtime | What the binary links and can drive: the CPU only, or the CPU and a Vulkan GPU (`-Druntime`). |
| runner | The code that runs the forward pass of one model family on one placement. It owns all the state of that pass. |
| placement | Where the layers of a model run: `fused` (all on the GPU), `hybrid` (a CPU prefix and a GPU suffix), or `cpu`. It says nothing about sessions. A session enters a VRAM slot and is then resident. |
| layer split | The numbers of a hybrid placement (`LayerSplit`): the layer where the GPU part starts, and the cost of each side. An attention split is another idea: it divides one attention row across GPU workgroups. |
| step | One call of a runner, `stepBatch(rows)`: one forward pass over a batch of rows. It runs every layer once for all its rows together, and it writes the KV of each row into the slot of that row. The serve layer puts rows of several sessions into one step: the decode rows of some, and a prompt chunk of another. |
| row | One `(slot, token, position)` entry of a step, plus a flag that asks for logits. A row carries one token of one session through every layer, at the position that the row names. The rows of one slot are adjacent and ascend by one position, and only the rows that ask for logits reach the lm_head. |
| prefill | The steps that feed the prompt into the model. Only the last prompt row needs logits. |
| decode | The steps that make one new token for each session. |
| host, device | The CPU side and the GPU side. |
| decode ring | The two decode command buffers of a fused runner (`DecodeRing`). While the GPU runs one decode step, the host records the next step into the other buffer. `RecordRing` in `core/record_ring.zig` holds its state. |
| upload ring | Two host-to-device upload segments (`StreamRing`). No runner uses it. |
| span ring | The trace buffer of one thread in `core/span.zig`. [profiling.md](profiling.md) calls it the span log. |

**Memory**

| Term | Meaning |
| --- | --- |
| slot | The KV storage of one session inside a runner. It has room for a full window of positions in every attention layer, plus the GDN state on Qwen3.5. A runner allocates all its slots at start, and a session keeps its slot from turn to turn. With the KV tiers, a server keeps more sessions than slots, and a session with no slot waits in RAM as a blob. |
| window | How many positions one session can hold (`--max-seq`). The window sets the size of every slot, so the KV memory is fixed before the first step. When a session reaches the end of its window, decode stops, and `serve` refuses a turn whose prompt does not fit what is left. The default comes from a byte budget for each session (section 9). |
| model ceiling | The largest window of a model. |
| plan | A value that the engine computes before it allocates: the slots, the window, the bytes of each part. |
| refusal | A stop at boot, before any large allocations. |
| tier | One level of KV storage: VRAM slots, RAM blobs, the disk. |

**Serving**

| Term | Meaning |
| --- | --- |
| session | One conversation that the server keeps. |
| turn | One prompt of a session and its reply. |
| scheduler | The code that decides, on each tick, which rows go into the next step. |
| tick | One iteration of the scheduler loop. |
| frontend | The protocol side of the server: the binary frontend (`serve`) or the OpenAI frontend (`serve-openai`). The code calls a frontend an `Adapter`. |
| prefix cache | Stored model state after a prompt prefix, so that a repeated system prompt needs no prefill. |

**Checks**

| Term | Meaning |
| --- | --- |
| reference | The `naive` backend. Every other backend must match its output. |
| token hash | The FNV-1a hash of the generated token ids. Two runs with equal hashes generated equal text. |
| gate | A check that must pass before a merge or a release. |

## 2. Build approach

`zig build` reads `build.zig` and makes these artifacts:

| Artifact | Root file | Where it goes | What it is |
| --- | --- | --- | --- |
| `zigferencer` | `src/main.zig` | `zig-out/bin/` | The release binary: `list`, `fetch`, `inspect`, `info`, `run`, `serve`, `serve-openai`, and `client`. |
| `zigferencer-dev` | `src/dev/dev_main.zig` | `zig-out/bin/` | The release commands, plus the developer commands: `bench-model`, `bench-serve`, `bench-swap`, `bench-decode-host`, `gpu-selftest`, `tokens`, `serve-demo`, `dispatch-list`, and `compare-logits`. |
| `zigferencer-bench` | `tools/bench.zig` | not installed | The op micro-benchmarks. `zig build bench` builds and runs it. |
| `zigferencer-simd-probe` | `tools/simd_probe.zig` | `zig-out/` | The assembly of each op, for `make simd-report`. `zig build simd-probe` installs only the `.s` file. |
| test binaries | `src/main.zig`, `src/dev/dev_main.zig`, `tools/shader_lint.zig`, `tests/*_test.zig` | `zig-out/test/` | The unit tests of each module, the shader lint tests, and one binary for each integration test. Only `test-install` installs one. |
| SPIR-V blobs | `src/ops_vulkan/shaders/*.comp` | inside the binary | The GPU kernels. Only a GPU build compiles them. |

A plain `zig build` installs only the release binary. `zig build install dev`
installs both. The release binary refuses a developer command, and its message
names `zigferencer-dev`.

### Binaries

`build.zig` makes two Zig modules, one for each binary, with the same target and
the same optimize mode. The dev binary therefore computes the same numbers as the
release binary.

- The **release module** roots at `src/main.zig`. It holds the engine and the release
  commands, and `src/main.zig` re-exports the engine modules under the package name.
- The **dev module** roots at `src/dev/dev_main.zig`. It holds the developer commands.
  The build gives it the release module as the import `zigferencer`.

The layers inside the tree point one way:

```
  src/dev/  -->  zigferencer.cli  (src/cli.zig)  -->  src/cli/
     |                                                   |
     +---------------->  resolve/, runtime/, serve/, io/, ops*/, core/  <--+
```

1. The engine (`resolve/`, `runtime/`, `serve/`, `io/`, `ops/`, `ops_vulkan/`, `core/`)
   reads no file of `src/cli/` or `src/dev/`.
2. A building block of `src/cli/` (the flag parser, the boot steps, the styles) reads
   no command file.
3. A command reads no other command.
4. No file outside `src/dev/` reads a file of `src/dev/`.

**Extending commands**

1. A release command takes `src/cli/NAME_cmd.zig`, or a directory with its name when it
   needs more than one file. Add one entry to `command.release` in `src/cli/command.zig`.
2. A developer command takes `src/dev/NAME_cmd.zig`, or a directory. Add one entry to the
   table in `src/dev/dev_command.zig` and the name to `command.dev_names`. Add a file
   that holds tests to the `test {}` block of `src/dev/dev_main.zig`.
3. When a dev file needs an engine module that `src/main.zig` does not re-export, add one
   `pub const` line there. When it needs a CLI building block, add one line to `src/cli.zig`.

The main build steps:

| Step | What it does |
| --- | --- |
| `zig build` | Builds and installs the release binary. |
| `zig build dev` | Builds and installs the dev binary. |
| `zig build check` | Type-checks both executables. It installs nothing. |
| `zig build test` | Runs the unit tests of both modules (`unit` and `dev_unit`), every integration test of this build, the format check, and the shader style rules. |
| `zig build test-unit` | Runs the unit tests of both modules and the tests of `tools/shader_lint.zig`. It runs no integration test. |
| `zig build test-build` | Compiles every test and runs none. |
| `zig build test-install -Dtest-bin=NAME` | Installs one test binary, so that a debugger can run it. |
| `zig build bench` | Builds and runs the op micro-benchmarks. |
| `zig build bench-build` | Compiles the benchmarks and runs none. |
| `zig build lint-shaders` | Checks the style rules of the shaders (`tools/shader_lint.zig`). It needs no Vulkan SDK. With `-Druntime=vulkan` it also runs `glslc -Werror` and `spirv-val`. |

### Code paths

Two build options decide which code computes the model:

- `-Druntime` decides what the binary links. `cpu` links only the Zig standard
  library. `vulkan` adds the `vulkan-zig` bindings, `libvulkan`, and `libc`, and it
  compiles the shaders. The Vulkan loader uses thread-local storage and the libc
  allocator, so a Vulkan binary must link `libc`.
- `-Dops-backend` decides which kernels compute the ops: `naive`, `simd`, or
  `simd_pf` on the CPU, `vulkan` or `vulkan_fused` on the GPU. `auto`, the
  default, selects `simd_pf` on an x86_64 target with AVX2 and FMA, and `naive` on
  any other target.

They are two options because a GPU binary also runs CPU kernels. A hybrid runs its
first layers on the CPU, and a GPU build can also run the CPU placement. A GPU build
resolves a CPU backend next to its GPU backend: `simd_pf` where the target
supports it, otherwise `naive`. A GPU backend selects `-Druntime=vulkan` when the flag is
absent, and it refuses `-Druntime=cpu`. Any other pair is legal.

| Build command | Runtime | GPU kernels | CPU kernels |
| --- | --- | --- | --- |
| `zig build` (x86_64 with AVX2) | `cpu` | none | `simd_pf` |
| `zig build -Dops-backend=naive` | `cpu` | none | `naive` |
| `zig build -Druntime=vulkan -Dops-backend=vulkan_fused` | `vulkan` | the fused kernels | `simd_pf` |

The build turns the choice into compile-time constants in `build_options`, and each
runner imports exactly one op module:

```
-Dops-backend=auto                                         cpu_runner.zig
-Druntime=cpu       --> resolveOpsBackend -->  build_options  -->  const ops = if (use_vulkan_ops) wrap.zig
                        (build.zig)           cpu_ops_simd_pf         else if (cpu_ops_simd_pf) simd_pf.zig
                                              = true                  else if (cpu_ops_simd) simd.zig
                                                                      else naive.zig;
```

The compiler sees one import, so the binary holds no backend branch and no CPU
feature check at run time. An explicit `simd` or `simd_pf` on a target without AVX2
and FMA stops the build with a message.

All SIMD work lives inside the op modules. A runner calls `ops.rmsnorm` and the
other op names, and it does not know which module answers. A new CPU backend is
therefore a new module. It holds the op functions and a
`pub const requires: OpRequires` that declares its target requirements. In
`build.zig`, `OpsBackendOption`, `OpsBackendResolved`, `resolveOpsBackend`, and
`cpuOpsBackend` get a case for it, and `addBuildOptions` gets a flag. The `ops`
alias in `cpu_runner.zig`, `cpu_layer_kernels.zig`, and `linear_dispatch.zig`
gets one branch each. `op_requires.ops_backend_names` and the backend tables of
`cli/info/` get an entry, so that `info` lists it. No layer file changes.

### The other build options

| Option | Effect |
| --- | --- |
| `-Doptimize` | `Debug` (the default of `zig build`), `ReleaseSafe` (keeps every assertion), or `ReleaseFast` (every measurement). |
| `-Dtrace` | Compiles the trace: the span sites, the per-op CPU timer, and the hot log scopes (`layer`, `dispatch`, `token`). It is off by default in every mode. |
| `-Dllvm` | Forces the LLVM code generator. A debugger needs it to show local variables. |
| `-Dvk-registry` | The path of the Vulkan registry, `vk.xml`. The default is the Debian and Ubuntu location. |
| `-Dattn-kernel` | The set of GPU attention kernels. `split` is the only set. |
| `-Dtest-filter`, `-Dtest-threads`, `-Dtest-bin` | Run only the tests whose name holds a string, the thread count of the runner tests, and the binary that `test-install` installs. |

A GPU build compiles each shader with `glslc` and embeds the SPIR-V in the binary
with `@embedFile`. The attention shaders that read KV compile twice, once for f32
KV and once for f16 KV (`-DKV_F16`), so the two variants share one source.
`attention_merge` reads no KV and compiles once. The tile sizes come from
`core/limits.zig` as `-D` defines, so a shader and the host code that sizes its
buffers read the same value.

### The Makefile

The `Makefile` is a small layer over `zig build`, and `make help` lists the targets.
[../tools/README.md](../tools/README.md) holds the rule for a new target.

- `BACKEND=` sets `-Dops-backend`. A GPU backend selects `-Druntime=vulkan` by itself.
- `OPT`, `ReleaseFast` by default, sets the optimize mode of `build`, `run`, `bench`, and
  `simd-report`. `make test` builds ReleaseSafe, and `make debug` builds Debug with LLVM.
- `ARGS=` holds the words after the binary or the script name. `FLAGS=` holds more `zig build`
  flags.
- `make build` and `make run` install both executables in one `zig build install dev` call.
  `make run` runs `zigferencer-dev`, which holds every command.
- `make ci` runs the CPU part of CI. `make ci-gpu` compiles every GPU path and runs none.
- `make gate`, `make profile`, and `make simd-report` call the scripts of `tools/`, and
  `ARGS=` carries their arguments. `make debug-build` calls `tools/debug-build.sh` with no
  arguments.

## 3. From a prompt to text

The three commands that generate text drive one pipeline:

```
     model file (GGUF)                          prompt text
            |                                        |
            v                                        v
     +-------------+                         +---------------+
     |    load     |                         |   tokenize    |
     +-------------+                         +---------------+
            | weights, metadata                      | token ids
            +-------------------+--------------------+
                                v
            +-------------------------------------------+
            | runner                                    |
            |   prefill: the prompt rows, chunk by chunk|
            |   decode:  step -> logits -> sample       |
            |            -> the next token, fed back    |
            +-------------------------------------------+
                                | token ids
                                v
                           detokenize --> text
```

The three commands differ in what sits around the runner.

- **`run`** generates text for one prompt, then exits. It prefills the prompt and
  decodes with greedy sampling, so the same prompt always gives the same tokens.
  `run` and the benchmark `bench-model` share this loop (`runtime/generate.zig`).
- **`serve`** stays up and keeps many sessions. It loads the model once. A
  scheduler advances all active sessions together, so one step carries one decode
  row for each active session, plus prompt rows of a waiting turn. The weights then
  are read from memory once for all sessions, not once for each session. A session
  keeps its KV cache between turns, so a follow-up turn prefills only its new
  tokens. Clients speak a small binary protocol over ZMQ (the goal for this small
  protocol is to make it easily embeddable in other systems).
- **`serve-openai`** runs the same engine and scheduler behind the OpenAI HTTP API.

`run` and `serve` are separate commands because they need different parts. `run`
is the shortest path from a prompt to text: no queue, no scheduler, no session
state. That makes it the reference for checks and measurements. `serve` adds the
parts that only matter when many conversations share one model:

| | `run` | `serve`, `serve-openai` |
| --- | --- | --- |
| Lifetime | One prompt, then exit | Until stopped |
| Sessions | One | Many (`--max-sessions`) |
| Decode | One row for each step | One row for each active session in one step |
| KV cache after a turn | Freed at exit | Kept for the session. The prefix cache and the RAM tier extend it |
| Sampling | Greedy | Set for each session: temperature, top-k, top-p, min-p, penalties, seed |
| Output | stdout | Frames of the binary protocol, or HTTP: one JSON body, or server-sent events when the request sets `stream` |

### The options

`zigferencer <command> --help` prints the flags of a command with their defaults.

| Options | Kind | Section |
| --- | --- | --- |
| `-Druntime`, `-Dops-backend`, and the other build options | build option | 2 |
| the model family and the weight type | read from the GGUF at load | 7 |
| `--gpu-layers` | flag | 7 |
| `--threads`, `--min-parallel-flops`, `--spin-us` | flag | 8 |
| `--max-seq`, `--kv-dtype` | flag | 9 |
| `--kv-vram`, `--vram-slots`, `--kv-ram`, `--prefix-disk`, `--preload` | flag of `serve` and `serve-openai` | 9, 10 |
| `--max-sessions`, `--max-batch`, `--prefill-chunk`, `--io-timeout` | flag of `serve` and `serve-openai` | 10 |
| `--max-connections`, `--sessions-per-connection`, `--allow-shutdown`, `--resume-ttl` | flag of `serve` (`--max-connections` also of `serve-openai`) | 10 |
| `--thinking`, `--thinking-budget`, and the sampling fields | flag, and a field of each turn | 10 |
| `$ZIGFERENCER_VK_STORAGE_RANGE` | environment variable | 9 |
| `$ZIGFERENCER_VK_SOFTWARE` | environment variable | [debugging.md](debugging.md) |
| `$ZIGFERENCER_LOG`, `$ZIGFERENCER_MODELS_DIR`, `$ZIGFERENCER_CACHE_DIR` | environment variable | 13 |
| `$ZIGFERENCER_TRACE_OUT` and the other trace variables | environment variable | [profiling.md](profiling.md) |

## 4. Control and datapath

The code has two kinds of modules:

- A **datapath** module holds as-pure-as-possible functions: bytes into metadata, 
  activations into activations. It holds no state and does no I/O, so a test can call 
  it with no setup. The datapath is the kernels in `ops/` and in the shaders, the GGUF 
  parser (`io/gguf/parser.zig`), the boot decisions in `resolve/`, and most of `core/`.
- A **control** module owns state and calls the datapath in order. The state is the
  KV cache, the GDN state, the scratch buffers, the decode position, the GPU command
  buffers, and the sessions. Control lives in `runtime/` and `serve/`.

`core/` also holds three small services with state: the log, the span recorder,
and the host probe. The `cli/` layer is the entrypoint. It reads the command line and the
host, calls the boot decisions, and builds the runners.

Note: This is just an organizational metaphor I tried to respect in this repo. I tried to carry
the lessons from digital design in VHDL I had many years ago.

## 5. Code placement

| Directory | What it holds | Files to start with |
| --- | --- | --- |
| [`src/`](../src/README.md) | The two root files. | `main.zig` (the entry of `zigferencer`, the re-exports, and the test list), `cli.zig` (the CLI building blocks that `src/dev/` can read) |
| [`src/cli/`](../src/cli/README.md) | The commands: the command line in, one handler for each command. | `command.zig` (the command tables), `args.zig` (the flag parser), `run_cmd.zig`, `boot/serve_boot.zig` (builds the serve engine) |
| [`src/dev/`](../src/dev/README.md) | The developer commands of `zigferencer-dev`. | `dev_command.zig` (the developer table), `bench_model/bench_model_cmd.zig`, `gpu_selftest/gpu_selftest_cmd.zig` |
| [`src/resolve/`](../src/resolve/) | The boot decisions. | `resolve.zig`, `device_placement.zig` (the GPU fit), `family.zig` (the runners of each family) |
| [`src/core/`](../src/core/README.md) | The shared values and helpers. | `tensor.zig`, `dtype.zig`, `limits.zig` (every bound), `kv_budget.zig` (the plans), `registry.zig` (the model table), `tokenizer/` |
| [`src/io/`](../src/io/README.md) | Files and the network. | `gguf/parser.zig`, `gguf/loader.zig`, `mmap.zig`, `download.zig` |
| [`src/os/`](../src/os/README.md) | Every OS condition: the calls that differ between systems. | `host_memory.zig`, `host_cores.zig`, `socket_wait.zig` (the selector of the serve socket calls) |
| [`src/ops/`](../src/ops/README.md) | The CPU kernels. | `naive.zig` (the reference), `simd_pf.zig` (the default), `gdn.zig` |
| [`src/ops_vulkan/`](../src/ops_vulkan/README.md) | The GPU kernels. | `shaders/`, `kernels.zig` and `kernels/` (one pipeline for each shader), `wrap.zig` (the per-op GPU backend) |
| [`src/runtime/`](../src/runtime/README.md) | The runners and their state. | `cpu_runner.zig`, `fused_runner.zig`, `hybrid_runner.zig`, `session.zig`, `sampler.zig`, `parallel.zig` |
| [`src/serve/`](../src/serve/README.md) | The server. | `scheduler.zig`, `engine.zig`, `protocol.zig`, `server.zig`, `openai/`, `residency.zig`, `prefix_manager.zig` |
| [`tests/`](../tests/README.md) | The integration tests against real model files. | `snapshot_smollm2_test.zig`, the `*_equivalence_test.zig` files |
| [`tools/`](../tools/README.md) | The op micro-benchmarks, the gates, the profiler driver, and the shader lint. | `bench.zig`, `gate.py`, `profile.sh`, `shader_lint.zig` |

### The surface between the CLI and the engine

The CLI is one client of the engine. It owns what belongs to a process: the command line, the
environment, stdout, the terminal, and the exit code. No engine file reads the command line or
the environment, sets the exit code, or imports `src/cli/`. Three engine files write to stdout
or stderr. `core/utils/log.zig` writes the log, and `io/download.zig` writes the download progress,
both on stderr. `runtime/gpu_profile.zig` writes the GPU profile table on stdout.

| Step | CLI -> Engine | Engine -> CLI |
| --- | --- | --- |
| Parse | `EngineFlags` | |
| Measure | | `Resources`, `ModelNeeds`, `DeviceCapacity`, filled by `cli/boot/boot_resources.zig` |
| Decide | `Input`, `Entry`, `Resources` | `Resolved` or `Refusal` |
| Build | `KvPlan`, `Policy`, `KvDtype`, `*Pool`, `*Loaded` | a runner of the family and the placement |
| `run` | token ids and a sink | `Generated` |
| `serve` | `scheduler.Config`, `Codec`, `*Residency` | `OutEvent` values, through the frontend |

## 6. The core data types

| Type | File | What it holds | Made by, read by |
| --- | --- | --- | --- |
| `Tensor` | `core/tensor.zig` | A view of one tensor: the dtype, the rank, the shape, the byte stride, and a borrowed slice of bytes. | The loader makes it over the memory map. Every op reads it. |
| `ModelCard` | `io/gguf/loader.zig` | The dimensions of the model, the end token, and the chat template. | The loader makes it. The runners and the boot decisions read it. |
| `Vocab` | `core/tokenizer/vocab.zig` | The token, merge, and byte tables of the tokenizer. | Built once at load. The BPE tokenizer reads it, and nothing changes it after the load. |
| `OpRequires` | `core/op_requires.zig` | What a backend needs from the target (architecture, SIMD level, FMA) and its tolerance against the reference. | Each op module declares one. `info` and the benchmarks read it. |
| `EngineFlags` | `resolve/flags.zig` | The flags that several commands share: the KV dtype, the window, the GPU layers, the threads, and the thinking mode. | The flag parser fills it. `resolve` reads it. |
| `Resources` | `resolve/resolve.zig` | What the host measured once: free RAM and, on a GPU build, the free VRAM and the device buffer limits. | The CLI measures it. `resolve` reads it. |
| `Resolved`, `Refusal` | `resolve/resolve.zig` | The result of the boot decision: the window, the KV dtype, the placement, the plan, or the reason the boot stops. | `resolve` makes it as a pure function. The CLI builds the runner from it or prints the refusal. |
| `KvPlan` | `core/kv_budget.zig` | The slot count, the window, and the bytes that each part reserves in VRAM and RAM. | The planner makes it. The CPU and fused runner constructors read it. The hybrid constructor reads a `LayerSplit`. |
| `LayerSplit` | `core/kv_budget.zig` | The layer split of a hybrid and the cost of each side. | The device decision makes it. A hybrid runner reads it, and `run` and `serve` print it. |
| `Policy` | `runtime/parallel.zig` | The thread count, the fan-out floor, and the spin budget of the CPU pool. | Set from the host and the thread flags. The pool and every parallel op read it. |
| `StepRow` | `runtime/step_row.zig` | One row of a step: the slot, the token, the position, and whether the row needs logits. | Sessions and the decode loop make it. Every runner step reads it. |
| `Request`, `OutEvent` | `serve/event.zig` | A request from a client, and an event back to a client. | The front-end tasks make the requests, and the scheduler makes the events. |

## 7. The forward pass

### Two model families

The model architecture comes from the GGUF at load, so one binary runs both
families.

- **Llama family** (`runtime/llama_layer.zig`). Each layer has two blocks. The
  first block is attention: RMSNorm, the Q, K, and V projections, RoPE, the KV
  write, grouped-query attention, and the output projection. The second block is a
  SwiGLU feed-forward network. Each block adds its output back to the residual
  stream. The same code runs SmolLM2-135M and Llama-3.1-8B.
- **Qwen3.5 hybrid family** (`runtime/qwen_layer.zig`). Most layers replace
  attention with Gated-DeltaNet (`ops/gdn.zig`). A GDN layer keeps a recurrent state
  of fixed size for each head. A KV cache grows with the sequence, and this state
  does not. Every Nth layer is full attention with a norm on Q and K and an output
  gate. The model file sets N in the GGUF key `qwen35.full_attention_interval`.
  Qwen3.5-2B sets 4.

### Runners and placements

With `-Dops-backend=vulkan`, the CPU runner drives the model, and each heavy op is
one GPU submit with its own wait and read-back. This per-op backend is a
correctness reference, and `serve` refuses it. With `-Dops-backend=vulkan_fused`,
the fused runner records the whole step as one command buffer and submits it once.
Its weights, KV, and activations stay in VRAM.

A runner runs the forward pass of one family on one placement. Two families on
three placements give six runners:

| Placement | Llama | Qwen3.5 | Weights | KV |
| --- | --- | --- | --- | --- |
| CPU | `CpuRunner(llama_layer)` | `CpuRunner(qwen_layer)` | the memory map | host RAM |
| GPU (fused) | `FusedRunner(llama_layer)` | `FusedRunner(qwen_layer)` | VRAM | VRAM slots |
| Hybrid | `HybridRunner(Llama)` | `HybridRunner(Qwen)` | the memory map for the CPU layers, VRAM for the GPU layers | each side holds the KV of its own layers |

```
  hybrid, split = 10 of 24 layers
  token --> [ layers 0..9 on the CPU ] --residual--> [ layers 10..23 on the GPU ] --> final norm, lm_head --> logits
                                        one copy
                                        per token
```

On a fused GPU build, `--gpu-layers` selects the placement:

- no flag: the fused runner.
- `0`: the CPU runner. The process opens no GPU context.
- `N`: the top `N` layers run on the GPU, and the rest on the CPU. `N` equal to the
  layer count is the fused runner.
- `auto`: the most layers that the free VRAM can hold.

A hybrid always keeps the final norm and the lm_head on the GPU, so the residual
stream crosses from the host to the device once for each token.

Every runner checks the weight type of each tensor when it starts. A tensor with a type
that the kernels of the runner do not read stops the start with
`error.UnsupportedWeightDtype`, before any forward pass runs.

### Steps and rows

Every runner has one entry point for work, `stepBatch(rows)`. Each `StepRow` names
a slot, a token, a position, and whether the row needs logits. `step` is the
single row form. The rows of one slot are adjacent and ascend by one position, and the
rows that need logits come last.

- **Prefill** feeds the prompt as rows of one slot, a chunk at a time. Only the last
  row of the prompt needs logits. The CPU runner runs the lm_head on that row only. The
  fused and the hybrid runners run it on the last row of each chunk.
- **Decode** feeds the last token of each session as one row and asks for logits.

A row in a batch gets the same inputs as the same row alone. A batched step
therefore gives each session the result it gets alone. The serve layer depends on
this rule (section 10).

### The fused step and the rings

A fused step is one command buffer that holds every dispatch of the step. The
buffer copies the input rows from the host, runs every layer, the final norm, and
the lm_head, and copies the logits rows back. The host submits it once and waits
once.

The host writes every dispatch and barrier of the step into the buffer, and the
GPU idles while a record runs before its submit. A decode record reads no token,
because the token is data in the input stage and the descriptor sets are fixed at
start. So while the GPU runs the step at position `p`, the decode ring records
the step at `p + 1` into its other buffer. The key of a record is its position
and its GPU timestamp level.

A step whose key equals the key of a recorded buffer is a hit, and it submits that
buffer with no record. Any other step is a miss, and it records first. A wasted
record is a recording that no step submitted. A new record, or a submit that is
not a decode step, discards it. Only `step`, the one-row decode of `run` and
`bench-model`, uses the ring. `serve` calls `stepBatch`, which records its buffer
on every step.

| Ring | File | What it holds | Slots | When it is full |
| --- | --- | --- | --- | --- |
| decode ring | `runtime/decode_ring.zig`, the state in `core/record_ring.zig` | The decode command buffers of a fused runner. | 2, one for each value of `record_ring.Buf` | It never blocks. A new record replaces a recording, which counts as wasted. |
| upload ring | `ops_vulkan/stream_ring.zig`, the slot table in `core/ring_fsm.zig` | Host-to-device upload segments (`StreamRing`). | `ring_fsm.depth` | `acquire` returns null, and the caller decides. |
| span ring | `core/span.zig` | The trace spans of one thread. One more ring holds the GPU submit records. | `limits.spans_per_thread` by default, `limits.trace_gpu_records` | It drops the new entry and counts it. |
| convolution ring | `ops/gdn.zig` | The last inputs of the convolution of one GDN layer and slot. | `conv_kernel - 1` for each channel | It never fills. Each token drops the oldest input. |

In this document, "the ring" is the decode ring. The `host_ring` line of
`bench-model` counts its hits, misses, speculative records, and wasted records.
`StreamRing` keeps one upload in flight while the host fills the next segment. No
runner uses it, and only `tests/vulkan_stream_ring_test.zig` drives it.

### The op API

A family writes the forward pass of a decoder layer once, as calls of the layer
kernel set that each runner supplies. `CpuLayerKernels`, the set of the CPU runner,
turns each call into op calls (**The calls of one layer**, below). The name `ops`
is a compile-time alias for the op module of the build, and the CPU modules export
the same names and signatures:

```
                llama_layer.forward, qwen_layer.forward
                                 |
                                 |  calls the kernel set of the runner
                                 v
      CpuLayerKernels                         FusedLayerKernels
      the CPU runner, and the CPU layers      the fused runner, and the GPU layers
      of a hybrid                             of a hybrid. It records each call
                                              into one command buffer.
              |
              |  calls  ops.<name>
              v
      +----------------+------------------+---------------------+
      | ops/naive.zig  | ops/simd_pf.zig  | ops_vulkan/wrap.zig |
      | scalar Zig     | AVX2, prefetch   | one GPU submit      |
      | the reference  | the CPU default  | for each op         |
      +----------------+------------------+---------------------+
             the ops alias names exactly one op module.
         simd_pf.zig and wrap.zig re-export some naive kernels.
```

Attention is the one op whose names differ. A CPU module exports a part kernel and a
merge, and `runtime/parallel.zig` runs them on the pool (section 8). The per-op GPU
module exports one attention call.

The fused runners do not call the `ops` module. They record every kernel of a step into one
command buffer and submit it once. Their layer weights, the lm_head, the KV, and the
scratch stay in VRAM, and their descriptor sets are bound once at start. The
embedding lookup reads the memory map on the host. The three paths compare like
this:

| Path | Build | How a step runs | Use |
| --- | --- | --- | --- |
| CPU | any CPU backend | The runner calls the kernels directly. The large ops fan out on the thread pool. | The default without a GPU. |
| Per-op GPU | `-Dops-backend=vulkan` | Each op is one GPU submit and one wait. The small ops (embedding lookup, RoPE) stay on the CPU. | A correctness reference and a benchmark. `serve` refuses it. |
| Fused GPU | `-Dops-backend=vulkan_fused` | The whole step is one command buffer and one submit. | The fast GPU path, also for the GPU layers of a hybrid. |

**Attention on the CPU.** The runner splits a row into parts of whole tiles
(`limits.attn_cpu_tile`, at most `limits.attn_cpu_parts` parts). One unit of work is
one KV head group over one part. A merge adds the parts in part order. The parts
depend only on the row length, so a row gives the same bits at any thread count.

**Attention on the GPU.** A row up to `limits.attn_split_len` positions runs in one
workgroup for each query head. A longer row splits its positions across workgroups,
and a merge pass adds the splits in a fixed order. A row length therefore gives the
same bits in any batch. Each runner plans the attention rows of a step once
(`AttnPlan` in `core/attn_split.zig`).

**The calls of one layer.** A decoder layer calls the layer kernel set in the order
of its `forward`. The first table numbers the calls of each kind of layer. The
second table names the CPU function and the fused shader behind each call. `ops`
is the op module of the build, and a shader name is its file in
`ops_vulkan/shaders/` without `.comp`.

| Op | Call | Llama | Qwen3.5 attention | Qwen3.5 GDN |
| --- | --- | --- | --- | --- |
| `attn_norm` | `norm` | 1 | 1 | 1 |
| `attn_proj` | `project` | 2: Q, K, V | 2: Q with its gate, K, V | |
| `qk_norm` | `qkNorm` | | 3 | |
| `rope` | `rope` | 3: adjacent pairs | 4: half-split pairs | |
| `kv_commit` | `kvCommit` | 4 | 5 | |
| `attention` | `attention` | 5 | 6 | |
| `attn_gate` | `gate` | | 7 | |
| `o_proj` | `project` | 6 | 8 | |
| `gdn_proj` | `project` | | | 2: qkv, z, alpha, beta |
| `gdn` | `gdn` | | | 3 |
| `gdn_out` | `project` | | | 4 |
| `mixer_residual` | `residual` | 7 | 9 | 5 |
| `ffn_norm` | `norm` | 8 | 10 | 6 |
| `gate_up_proj` | `project` | 9 | 11 | 7 |
| `swiglu` | `swiglu` | 10 | 12 | 8 |
| `down_proj` | `project` | 11 | 13 | 9 |
| `ffn_residual` | `residual` | 12 | 14 | 10 |

| Call | CPU function (`CpuLayerKernels`) | Fused shader (`FusedLayerKernels`) |
| --- | --- | --- |
| `norm` | `ops.rmsnorm` | `rmsnorm` for `step`, `rmsnorm_batch` for the other steps |
| `project` | `ops.linearF32Q80Batch`, or the batch op of the weight type, on the pool | `linear_q8_0_reduce` for one row, `linear_q8_0_batch` for more. Qwen3.5 records `linear_q8_0_gemm` from `gemm_min_batch` rows. A Llama Q4_0 model records the `linear_q4_0_` forms. |
| `qkNorm` | `ops.rmsnorm` on each head | `qwen_qk_norm` |
| `rope` | `ops.ropeHeads` once per row, with the base angles that the runner fills at init | `rope_multi` for `step`, `rope_multi_batch` for the other steps |
| `kvCommit` | a copy into the slot, narrowed to the KV type | a buffer copy, after `kv_commit_f16` at f16 KV |
| `attention` | `ops.attentionPartTyped` for each part on the pool, then `ops.attentionMerge` | `attention` for `step`, `attention_multi_stream` for `stepBatch`, `attention_batch` for a prompt chunk. A row past `limits.attn_split_len` adds `attention_split` and `attention_merge`. |
| `gate` | `ops.sigmoidGate` once per row | `sigmoid_gate` |
| `residual` | `ops.residualAdd` | `residual_add` |
| `swiglu` | `parallel.swiglu` on the pool, which runs `ops.swiglu` over element ranges | `swiglu` |
| `gdn` | `ops.gdnStepKeyHead` for each key head on the pool: `gdn.stepKeyHead` under naive, the lane step of `ops/gdn_simd.zig` under the SIMD backends | `gdn_decode` for `step` and `stepBatch`. A prompt chunk records `gdn_chunk_conv`, `gdn_chunk_scan`, and `gdn_chunk_norm`. |

All three layers share the norms, the projections, the residual adds, and the
feed-forward, on the same CPU functions. The Qwen3.5 attention layer adds
`qkNorm` and `gate`, and it rotates half-split pairs where Llama rotates adjacent
pairs. A GDN layer keeps a fixed-size recurrent state and no KV, so `gdn` replaces
RoPE, the KV commit, and attention.

### Support by path

| Path | Llama | Qwen3.5 | Weight types | KV types |
| --- | --- | --- | --- | --- |
| CPU (`naive`, `simd`, `simd_pf`) | yes | yes | Q8_0, Q4_0, Q4_1, Q6_K | f32, f16 |
| Per-op GPU (`vulkan`) | yes | yes | Q8_0 | f32 |
| Fused GPU and hybrid (`vulkan_fused`) | yes | yes | Llama: Q8_0, Q4_0. Qwen3.5: Q8_0 | f32, f16 |

Activations are f32 on every path. The KV type sets the element of every K and V row:
f16 halves the bytes of the cache. The Gated-DeltaNet state stays f32. A run at one
KV type gives the same bits batched and alone. Across KV types and across backends
the checks compare the greedy tokens, not the bytes.

## 8. CPU threads

### Why decode and prefill differ

Decode multiplies each weight matrix by one vector. Each weight byte is read once
and used for a few floating-point operations, so the memory bandwidth limits the
speed, not the arithmetic. More threads help only until they fill the memory
bandwidth. Prefill feeds a chunk of prompt rows through one step. One read of a
weight then serves every row of the chunk, so prefill gains from more threads for
longer. `run` prefills in chunks of `limits.cpu_prefill_chunk` rows. `bench-model`
takes the chunk from `--prefill-chunk`, and the default is the same limit. Only
the last chunk of a prompt runs the final norm and the `lm_head`.

### The pool

`runtime/pool.zig` holds a persistent fork-join pool. A command makes one pool and
passes it to the runner. The calling thread is participant 0 of every dispatch:

```
  caller (participant 0)               workers 1 .. N-1
  ----------------------               ----------------
  publish the job and a new epoch -->  see the new epoch (spinning) or wake (futex)
  run its first unit                   run its first unit
  claim more units from one counter    claim more units from the same counter
  wait until every participant is done <--  report done, spin a short time, then sleep
```

A dispatch takes no lock and allocates nothing.

### The policy controls

- **`--threads N`** sets the participants. `0` or no flag takes the physical cores
  from the OS topology (sysfs on Linux), at most 8.
  Without a topology it takes half the logical CPUs.
  On a desktop with two memory channels, decode fills the bandwidth between 4 and 8 threads, 
  and prefill gains up to 8. A named value can go up to `limits.max_threads`.
  A placement that runs no CPU op, such as the fused GPU placement, gets one thread.
- **`--min-parallel-flops F`** is the fan-out floor. An op below `F`
  operations runs on the caller and does not touch the pool. A dispatch has a fixed
  cost, and a small op ends before the workers wake. For example, a 576 x 576
  matrix-vector product is 2 x 576 x 576 = 663,552 operations, above the default
  floor, so it fans out. A 64 x 64 product is 8,192 operations, so it runs inline.
  One floor serves every op class.
- **`--spin-us N`** is the time that a worker spins after its last unit before it
  sleeps on a futex. The ops of one decode step follow each other quickly, so a
  spinning worker sees the next dispatch at once. A sleeping worker uses no CPU while
  the engine waits. `0` sleeps at once.

### Units and determinism

| Op | One unit |
| --- | --- |
| Matrix product | A chunk of outputs, a multiple of 4, about four chunks for each participant. |
| Attention | One KV head group over one part of one row of the step. |
| Gated-DeltaNet | One key head with its value heads, over every row of the step, in row order. No serial part runs on the caller. |
| `swiglu` | A range of elements of the flat gate buffer, about four ranges for each participant. |

The result of a unit does not depend on the thread that runs it. A chunk of outputs is one
kernel call. An attention part writes its own record, and the caller adds the parts
in order. A key head and its value heads write only their own channels, conv ring
entries, and states. The token hash therefore holds across thread counts, floors,
spin budgets, and claim orders. `tools/gate.py threads` checks the thread counts.

## 9. Memory

### The terms

| Term | Meaning |
| --- | --- |
| window | The positions that one session can hold. |
| named window | A window that you set with `--max-seq`. A default window comes from a byte budget instead. |
| model ceiling | The largest window of a model: the context that the file declares, capped by `limits.max_model_ctx` and by the `context_limit` of the registry entry. |
| slot | The KV of one session for a full window, plus the GDN state on Qwen3.5. |
| region | One K or one V buffer of one layer, holding the KV rows of every slot. The device buffer limits apply to one region. |
| KV row | The K or the V of one layer at one position, for every KV head. It is a different thing from a row of a step. |
| budget | The bytes that a plan can use: a VRAM budget, a RAM budget, or the default budget of one session. |
| preflight | The check, before the first device allocation, that a plan fits the usable share of the free VRAM (`limits.vram_shares`). |
| RAM plan | The host RAM that a placement needs, checked before the runner allocates. |
| blob | The RAM copy of the KV of a session that left its VRAM slot. It is the only copy of that session. |
| ledger | The running total of blob bytes against the blob budget. |

```
  model card: KV layers, KV heads, head size --+
  --kv-dtype: bytes of one element ------------+--> KV bytes of one position
                                                              |
  budget of one session --> budget / KV bytes, rounded down to 256, capped at the ceiling
  (run or serve)                                              |
                                          --max-seq ------> window  (a named window skips the budget)
                                                              |
                                 slot = window x KV bytes of one position + GDN state
                                                              |
                        --max-sessions, --vram-slots --> KV plan = slots x slot
                                                              |
                    +-----------------------------------------+---------------------------+
                    v                                                                     v
    region check: one region (one layer, K or V,             VRAM preflight: weights + KV plan
    every slot) fits the device buffer limits                + scratch + logits fit the usable free VRAM
```

### A worked example: Qwen3.5-2B

The numbers below are for Qwen3.5-2B at Q8_0 with f16 KV.
The host has an AMD Ryzen 9 5950X with 32 GB of DDR4 and an AMD Radeon RX 6800 XT.
The GPU runs through Vulkan on D3D12 under WSL2.
Only the GPU `run` window depends on that driver. The model has 24 layers. 6 layers are full attention, with 2
KV heads of 256 dimensions. The other 18 layers are Gated-DeltaNet, with 16 heads and
a state of 128 x 128. The lm_head shares its weights with the embedding table. MiB
values round down, as the boot log prints them.

| Quantity | Value | Rationale |
| --- | --- | --- |
| Weights | 1908 MiB | The tensors of the file: 1393 MiB of layers and 515 MiB of embedding table. The file is 1919 MiB, and the rest is metadata, mostly the tokenizer. |
| KV bytes of one position | 12,288 B | 6 layers x 2 heads x 256 dimensions x 2 (K and V) x 2 bytes. The GDN layers store no KV. |
| GDN state of one slot | 19.27 MiB | 18 layers x (16 heads x 128 x 128 + a convolution ring of 3 x 6144) x 4 bytes. It is f32 at every KV type, and it does not grow with the window. |
| Model ceiling | 262,144 | The declared context, which is also `limits.max_model_ctx`. |
| `run` window from the budget | 174,592 | The 2 GiB budget / 12,288 B = 174,762, rounded down to a multiple of 256. The budget holds KV rows only. |
| `run` slot on the CPU | 2065 MiB of host RAM | 174,592 x 12,288 B = 2046 MiB of KV, plus the GDN state. The weights stay in the memory map, which the RAM plan does not count. |
| `run` window on the GPU | 131,072 | The region check cuts the window. One region holds the K or the V of one layer: 174,592 x 2 heads x 256 x 2 bytes = 178,782,208 B. The driver reports a storage-buffer range of 128 MiB, so the window shrinks to 134,217,728 B / 1024 B = 131,072. With a larger range, set by `$ZIGFERENCER_VK_STORAGE_RANGE`, the GPU window stays at 174,592. |
| `run` VRAM plan | 3519 MiB | The weights (1908 MiB), the KV (131,072 x 12,288 B = 1536 MiB), the GDN state (19 MiB), the scratch (55 MiB), and the logits (1 MiB). The embedding table goes to VRAM once, as the lm_head. The embedding lookup reads the same tensor from the memory map. |
| `serve` window | 10,752 | The 128 MiB budget / 12,288 B = 10,922, rounded down to a multiple of 256. One region is 4 x 10,752 x 1024 B = 42 MiB, inside the range. |
| `serve` KV and GDN state, 4 sessions | 581 MiB of VRAM | 4 slots x (10,752 x 12,288 B + 19.27 MiB). The boot log calls it `vram kv`. |
| `serve` host RAM plan | 1314 MiB | Two copy stages of one slot each (2 x 145 MiB) and the default prefix budget (1024 MiB). The prefix cache allocates each entry when it stores one. An entry holds its KV rows and the whole GDN state. |

**Why the GPU `run` window is smaller.** VRAM is not the limit here, because the
plan fits. The limit is the size of one binding. A shader reads a KV region through
one storage-buffer binding, and the driver reports the largest binding it accepts,
`maxStorageBufferRange`. In this example it is 128 MiB, or 134,217,728 B. Each
position adds one row to a region: 2 KV heads x 256 dimensions x 2 bytes = 1024 B.
One region can therefore hold 134,217,728 B / 1024 B = 131,072 positions. At the
budget window of 174,592 positions, one region needs 178,782,208 B, so the window
shrinks to 131,072. The 12 regions together hold 1536 MiB, because each region has
its own binding.

**How the scratch is sized.** The scratch holds what one step computes and then
drops. `scratchVramBytes` adds two parts, and neither depends on the window or on
the session count:

- **Work buffers, 23 MiB.** Each op writes a result that the next op reads: the
  hidden state, the K and V projections, the GDN key and value, the FFN output. The
  sum of these widths is 46,112 values for one row, and they come from the model shape.
  The runner multiplies that sum by 4 bytes and by 128 rows, `prefill_rows_max`, the
  widest step it accepts. At f16 it adds one K row and one V row for each of the 128 rows.
- **Split attention records, 32 MiB.** Each part of a split attention row leaves
  one summary record for each query head. The store holds 4 rows
  (`limits.attn_split_rows`) x 8 heads x 1024 parts x 258 values x 4 bytes. The 1024
  parts cover the declared context, not the window, because the planner subtracts
  the scratch before it derives the window.

For comparison, Llama-3.1-8B at f32 stores 2 x 32 layers x 8 heads x 128 x 4 bytes
for each position. That is 262,144 bytes, 21 times more. The same budgets give it a `run`
window of 8192 positions and a `serve` window of 512.

### How `run` and `serve` reserve memory

`run` reserves one slot. That slot gets one budget, `limits.run_session_kv_bytes`,
on every placement. The budget is a fixed number. It does not measure the free RAM
or the free VRAM. The CPU and the GPU therefore start from the same window, 174,592
positions. The GPU window differs only when a device check cuts it, as the region
check does in the example.

`serve` reserves one slot for each session. With no budget flag, `--max-sessions`
(default 4) is also the slot count. Each slot gets the smaller budget
`limits.serve_session_kv_bytes`. The runner allocates every slot at boot, so a slot
with no client costs the same VRAM as a busy slot. An open session takes a free
slot and keeps it between its turns.

### The KV cache as a tensor

The KV cache is one array with six axes. These are the axes for `serve` with 4
sessions:

```
KV [K or V] [layer] [slot] [position] [KV head] [dimension]
      2         6       4     10,752       2         256     = 264,241,152 f16 elements = 504 MiB
                the full-attention layers
```

From the reservation down to one element:

```
KV and GDN state: 581 MiB of VRAM
|
+-- KV cache: 504 MiB = 12 regions, the K and the V of 6 layers
|   |
|   +-- one region, for example the K of layer 3: 42 MiB = 4 slots
|       |
|       +-- one slot in that region: 10.5 MiB = 10,752 positions
|           |
|           +-- one position: one KV row, 1024 B = 2 KV heads
|               |
|               +-- one KV head: 512 B = 256 elements of 2 bytes
|
+-- GDN state: 77 MiB = 4 slots
    |
    +-- one slot: 19.27 MiB = 18 GDN layers of 1.07 MiB
```

A region cuts the array by the first two axes. A shader binds one K region and one
V region for each attention layer. A session cuts the array the other way. Its
slot is one stripe in each of the 12 regions:

```
            K of layer 3    K of layer 7           V of layer 23
           +------------+  +------------+         +------------+
  slot 0   |            |  |            |         |            |
  slot 1   |            |  |            |   ...   |            |
  slot 2   |############|  |############|         |############|  one session: 12 stripes
  slot 3   |            |  |            |         |            |  of 10.5 MiB = 126 MiB
           +------------+  +------------+         +------------+
            position 0 .. 10,751, left to right
```

A token at one position of one slot adds one KV row to each of the 12 regions. That
is 12 x 1024 B = 12,288 B, the KV bytes of one position. Attention then reads the
rows from position 0 to the current position. The 8 query heads share the 2 KV heads.
Query heads 0 to 3 read KV head 0, and query heads 4 to 7 read KV head 1.

The index of one element inside its region (`core/kv_index.zig`) is:

```
(slot x window + position) x (KV heads x dimensions) + KV head x dimensions + dimension
```

For example, take dimension 7 of KV head 1, at position 100 of slot 2. Its element
index is (2 x 10,752 + 100) x 512 + 1 x 256 + 7 = 11,061,511. At 2 bytes for each
element, it sits at byte 22,123,022 of its region.

### Why the axes have this order

The order follows what the engine does with the cache, from the outer axis to the
inner axis:

- **K or V, then layer: the regions.** One long session does not fit one Vulkan
  buffer. Llama-3.1-8B at 131,072 positions holds 16 GiB of KV at f16. Attention needs only
  the K and the V of one layer at a time (the device limits then apply to one region, not to the whole cache).
- **Slot before position: one block for each session.** The rows of one session stay
  together. A shader reaches a session with one stride, `window x KV heads x
  dimensions`. A move to RAM or a prefix snapshot copies one range from each region,
  plus the GDN state on Qwen3.5.
- **Position before KV head: one write for each token.** The K and V projections of
  a token give one whole row. A write then copies one row to each region of
  the session, after its last row. Attention then reads the positions in ascending order.
- **KV head before dimension: a head is contiguous.** Grouped-query attention reads
  one KV head at a time. Each lane walks one row of that head in order, so it uses
  every cache line that it reads.

> Note: Better orders can exist. This topic is ongoing work.

The CPU keeps the same inner axes in one `KvCache` for each slot, so the slot is its
outer axis: `[slot][K or V][layer][position][KV head][dimension]`. It has no regions,
because the device limits do not apply to it.

### Why slots are fixed (no paged attention)

A slot reserves a full window at boot, also for a session that stays short. Engines
such as vLLM use paged attention instead. The KV cache is a pool of small blocks, a
session gets blocks as it grows, and a block table maps its positions to blocks.
zigferencer does not do this yet, for three reasons:

- **Every attention kernel depends on contiguous rows.** A block table adds one
  indirection to each attention kernel, on the CPU and in the shaders.
- **The bit-reproducibility check depends on the row layout.** Today the row length
  alone sets the parts, the splits, and the merge order. A block layout reopens that
  rule for every kernel.
- **Whole-slot moves give most of the value.** The KV tiers move a whole session to
  RAM. The prefix cache restores a stored prefix into a slot. Neither needs a change
  to a kernel.

The cost is real. A short session holds a full window of memory, and two sessions
with one prefix hold two copies of its rows. A session cannot outgrow its window, so
`client` compacts its transcript instead. Paged attention stays future work, for the
case where whole-slot moves measure as not enough.

### Where the bytes live

```
  disk                 host RAM                                    VRAM (GPU build)
  ----                 --------                                    ----------------
  GGUF file  --mmap--> weights of the CPU layers                   weights of the GPU layers
                       KV slots of the CPU layers                  KV slots of the GPU layers
                       blobs of sessions out of VRAM  <--evict--   (one region for each
                                                      --restore->   layer and K or V)
                       swap stages                                 scratch, logits
  prefix files <-----> prefix cache (RAM tier)
```

The runtime sizes the weights, the KV slots, and the scratch once, from a plan,
before the first step. The swap stages grow on the first swap, up to one chunk. The
blobs and the prefix entries allocate when the server stores one. The plan sizes
these too. No buffer grows during a step.

### The window

`--max-seq` names the window. Without it, the default comes from a byte budget for
one session: `limits.run_session_kv_bytes` for `run` and `bench-model`, and
`limits.serve_session_kv_bytes` for each serve session. The default window is that
budget over the KV bytes of one position. It rounds down to a multiple of 256
positions, stays at least 128, and stops at the model ceiling. Qwen3.5 keeps KV only
in its full-attention layers. The same budget therefore gives it a much deeper
window than a Llama model of the same size.

A `--max-seq` past the model ceiling is a usage error. The loader refuses a file
that declares a RoPE scaling that the engine does not apply.

A window that you name with `--max-seq` is a request that the engine meets or
refuses with a message. A silent smaller window breaks a client that sized its
prompts for the named window. A default window is the choice of the engine, so
the GPU checks below shrink it to fit. A refusal of a default punishes a user who
asked for nothing. With `--vram-slots`, the slot count is the request, and the
VRAM budget holds the window. A named window then caps that window, and the boot
log warns when the budget holds less.

### The GPU checks

A GPU build checks its plan before the first device allocation. Two checks can
shrink a default window. A GDN slab or an activation buffer past a device buffer
limit stops the runner with `error.StepBufferPastDeviceLimit`. A named window
that fails a check stops the runner, and the message names the largest window
that fits. With `--vram-slots`, a named window is a cap instead: the plan takes
the smaller of `--max-seq` and the window that the VRAM budget can hold.

1. **The region check.** `vulkan_ctx.Context` reads three device limits at start:
   the storage-buffer range, the largest allocation, and the binding offset
   alignment. A shader also indexes KV with a signed 32-bit value, so one region
   holds at most 2^31 elements. When one region breaks a limit, the default window
   shrinks to the widest multiple of 256 positions that passes. The `run` window of
   the example shrinks this way, from 174,592 to 131,072 positions.
2. **The VRAM preflight.** A fused or hybrid runner adds up its plan: the weights,
   the KV, the recurrent state, the logits, and the scratch. The plan must fit the
   usable share of the free VRAM. `limits.vram_shares` holds one entry for each
   driver: 3/4 on the translation-layer driver of the reference setup, the one
   driver id that `vulkan_ctx.driverEntry` names, and 7/8 on every other driver. The
   3/4 entry is a heuristic of that setup. On the whole-model fused path, a default
   window first shrinks to the widest window that fits. A plan that still does not
   fit stops with `error.VramBudgetExceeded`. The message names the needed, free,
   and usable bytes, the largest window that fits, and the flags that lower the
   plan.

The free VRAM comes from `VK_EXT_memory_budget`, because a desktop shares the card
with other programs. Without that extension it is the heap size. The reserve holds
the upload stage, the fragmentation of the allocator, and the overhead of the
driver. The preflight refuses a plan that cannot fit. A plan under the line can
still fail on a driver whose budget does not show its host-visible staging or its
residency.

Some drivers report the minimum storage-buffer range of the Vulkan specification,
128 MiB, and still read past it correctly. `$ZIGFERENCER_VK_STORAGE_RANGE` then
states the range that you checked, and `info` prints both values.

**The reference setup.** A little history on this topic.
The GPU path grew on one desktop card, through WSL2 (experimental reasons). 
Its Vulkan driver translates every call to the graphics API of Windows. 
That driver reports the 128 MiB range, which shrinks the
window of the example. It also allows only 32 KiB of shared memory for each
workgroup, and its allocation overhead runs out of memory before the card is full.
Under memory pressure it can lose the device. More info on [gpu-rdna2.md](gpu-rdna2.md).

**Budgets for serve.** `--kv-vram MiB` gives the KV a VRAM budget, and the budget
holds as many slots as it can at the window. `--vram-slots N` pins the slot count
instead, and the window then follows from the budget. A `LayerSplit` sizes a
hybrid: the CPU layers on one side and the GPU layers on the other. Each layer has
its own size.

### Host RAM

Every placement of `run`, `bench-model`, `serve`, and `serve-openai` plans its host
RAM through one function, `kv_budget.planRam`, before it allocates a runner. The
RAM plan holds the host KV, the swap stages, the blobs, and the prefix cache. A
RAM plan above 7/8 of the available RAM stops the boot, and the message names the
numbers and a window that fits. The printed fit keeps a margin, because free RAM
moves between two boots. A host that reports no available RAM gets no check, and
the log says so.

A session that moves between VRAM and RAM passes through a swap stage, in chunks of
`limits.swap_stage_chunk_bytes`. When `--vram-slots` is below `--max-sessions`, the
blobs need room for one blob more than the sessions that the slots cannot hold. A
swap holds the leaving blob and the arriving blob at the same time. With
no `--kv-ram`, the plan holds that whole need or refuses the boot. A named
`--kv-ram` stays as named, and a value below the need logs a warning. The prefix
cache gets the rest of the RAM budget. Section 10 tells what the scheduler does when
the ledger is full.

### Example: use 14 GiB of VRAM and 28 GiB of RAM on `serve`

This example asks how many Qwen3.5-2B sessions fit when 16 GiB of VRAM and 32 GiB
of RAM are free, on a driver of the default entry. The 7/8 rules leave 14 GiB of VRAM
and 28 GiB of RAM. The planner
computed every number below.

Three limits decide the answer:

1. **VRAM holds slots.** The weights, the scratch, and the logits take about 1970 MiB.
   That leaves about 12,364 MiB for slots. One slot at the `serve` window costs 145.3
   MiB, so the VRAM holds 85 slots.
2. **One region limits slots x window.** One region holds every slot, at 1024 B for
   each position. A range of 128 MiB therefore holds slots x window up to 131,072,
   which is 12 slots at 10,752 positions. `$ZIGFERENCER_VK_STORAGE_RANGE` lifts this
   limit. The entries with a 1 GiB range set it to 1073741824.
3. **RAM holds the sessions that have no slot.** Each one needs a blob of up to
   145.3 MiB. `--max-sessions` stops at `limits.max_logical_sessions`, which is 128.

| Goal | Range | Flags | Slots | Window | Sessions | VRAM plan | RAM plan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Many short sessions | 128 MiB | `--max-sessions 128 --vram-slots 12 --max-seq 10752` | 12 | 10,752 | 128 | 3715 MiB | 18,310 MiB |
| Many short sessions | 1 GiB | `--max-sessions 128 --kv-vram 12492` | 85 | 10,752 | 128 | 14,319 MiB | 7706 MiB |
| Longer sessions | 1 GiB | `--max-sessions 15 --vram-slots 15 --max-seq 65536` | 15 | 65,536 | 15 | 13,780 MiB | 1536 MiB |
| Long sessions | 1 GiB | `--max-sessions 7 --vram-slots 7 --max-seq 131072` | 7 | 131,072 | 7 | 12,857 MiB | 1536 MiB |
| The longest sessions | 1 GiB | `--max-sessions 4 --vram-slots 4 --max-seq 262144` | 4 | 262,144 | 4 | 14,333 MiB | 1536 MiB |

With the 128 MiB range, the long entries hold 2 slots at 65,536 positions and 1 slot at
131,072. No slot fits at 262,144. RAM also limits the blobs of long sessions: 28 GiB
holds 35 sessions at 65,536 positions and 17 at 131,072.

The flags to set:

- `--max-sessions` sets the sessions. Above `limits.max_batch`, it needs `--vram-slots`
  or `--kv-vram`.
- `--vram-slots` pins the slots. Name `--max-seq` next to it. Without `--max-seq`, the
  slots share all the usable VRAM, and the window grows: 12 slots get 86,278
  positions at a 1 GiB range. The blobs grow with the window, so `--max-sessions 128`
  then stops at the RAM check.
- `--kv-vram` sets a VRAM budget instead, and the budget holds slots at the window.
- `--kv-ram` sizes the RAM tier. With no flag, the plan holds every blob and a prefix
  cache of 1024 MiB. A larger value gives the rest to the prefix cache. On the first
  entry, `--kv-ram 21093` leaves 4 GiB for it.
- `--max-connections` limits the clients (default 8). One connection can hold many
  sessions.

More slots do not make more sessions decode in one step. At most `limits.max_batch`
rows decode together, and the scheduler rotates the others. A slot keeps a session
ready, because a session with no slot must move back to VRAM before its next row.

## 10. Serving

`serve` and `serve-openai` turn the engine into a server that a host process
starts and keeps. This section describes the threads and queues first, then the
scheduler, which decides what each step computes.

### Threads and queues

```
  client                frontend              engine thread
  ------                ---------              -------------
  frames or HTTP  --->  reader task  --Request-->  request queue  --->  scheduler.tick
                                                    (bounded)              |
                                                                           v  one runner step
  frames or SSE   <---  writer task  <--OutEvent--  event queue of   <-----+
                                                    the connection
                                                    (bounded)
```

One engine thread owns the scheduler and the runner. A frontend task never
touches the runner: it turns wire bytes into `Request` values and `OutEvent` values
back into wire bytes. The engine loop reads a few requests without a wait, handles
each one, then runs one tick. It sleeps on the request queue only when the tick did
no work and no request came in.

A failed `accept` does not end the server. A failure that comes from a client, such as an
aborted connection, retries at once. A failure from a lack of resources, such as file
descriptors or memory, retries after a wait (`accept_backoff_ms`). Each retry logs one line.
Only a cancel at shutdown, a closed listener, or `error.WouldBlock` stops the acceptor task
(`acceptAction` in `serve/engine.zig`).

The engine thread never waits for a client. When the event queue of a connection is
full, the scheduler puts the event in the pending cell of that session. That session
gets no row until the event goes out, and every other session keeps its rows.

### The two frontends

| | Binary frontend (`serve`) | OpenAI frontend (`serve-openai`) |
| --- | --- | --- |
| Wire | Framed binary messages over ZMTP. `serve/protocol.zig` defines the frames and refuses every other protocol version. | HTTP: `POST /v1/chat/completions`, with one JSON reply or with server-sent events when the request sets `stream`, `GET /v1/models`, `GET /healthz`. |
| Client | Any ZMQ DEALER socket. `serve/client.zig` is the reference client. | Any OpenAI client or SDK. |
| Tasks | Two for each connection: a reader and a writer. | One for each connection. |
| Sessions | A client opens, uses, and closes sessions, many on one connection. | Each request opens a session, runs one turn, and closes it. |

Both frontends implement one comptime `Adapter` interface over the same engine
(`serve/engine.zig`).

When the server is ready, it prints one line to stdout: `READY tcp <address>` or
`READY unix <path>`. A process that starts the server reads that line for the port.

### Session states

```
                  open                     turn, opening, append
      free  --------------------> idle  --------------------------> prefill
        ^                          ^  ^                                |
        |                          |  +---- opening or append done ----+
        |                          |        (session_opened, appended) |  last chunk
        |                          |                                   v  of a turn
        |                          +-------- stop, send done --------- decode
        |                                    (end token, max tokens,   (one token
        |                                     window full, cancel)      each tick)
        +---- close, a drop with no lease, a lease that expires, or a lease that
              yields to a new open (from any state)
```

- **open** gives a session a free entry of the session table. **clone** copies an
  idle session into a new one.
- A **turn** (`generate`) tokenizes the prompt and moves to prefill. An **opening**
  prefills a system prompt. An **append** prefills earlier turns that a client sends
  back. An opening and an append produce no tokens. They end with a
  `session_opened` or an `appended` reply and return to idle.
- The last chunk of a turn produces logits. The scheduler samples the first token
  from them, and the session moves to decode.
- Decode adds one token on each tick until a stop. A stop is the end token of the
  model, the `max_tokens` of the turn, a full window, or a cancel. The scheduler then sends
  `done`, and the session is idle again.
- A turn whose prompt does not fit the window is refused before it starts.

### One tick

1. Release the sessions whose lease expired.
2. Send the events that wait in pending cells.
3. Send `done` for each turn that stopped.
4. Pick the decode rows. The scheduler walks the sessions from a start point, takes
   each session in decode that has no pending event, and stops at `--max-batch`
   rows. The start point moves one session on each tick, so every session gets rows
   when more sessions decode than `--max-batch` allows.
5. Pick one turn to prefill: the turn that waits longest.
6. Size its prompt chunk by the prefill chunk rule (below).
7. Run one step with the rows `[prompt chunk | decode rows]`. Only the last row of
   a finished prompt and the decode rows produce logits.
8. Sample each session with its own sampler state and seed.
9. Accept each token: send the token event, then check the stops and the think
   budget.

If no row is ready, the tick reports no work, and the engine thread sleeps until a
request comes in.

### The prefill chunk rule

Decode is memory-bound: a step reads every weight once, for one row or for many
rows. A prompt chunk in the spare rows of a decode step therefore costs little, and
the sessions that decode never wait for a long prompt. This is continuous batching.
`serve/prefill_chunk.zig` sizes the chunk:

| `--prefill-chunk` | Prompt rows in one step |
| --- | --- |
| unset (the default) | With no session in decode: the full row width of the runner. With sessions in decode: the spare rows, `--max-batch` minus the decode rows. |
| `0` | Always the spare rows. |
| `N` | `N` rows, also next to decode rows. The row capacity of the runner caps it. |

A waiting prompt always gets at least one row, so a step can hold one row more than
`--max-batch`. Example: with `--max-batch 8`, three sessions in decode, and a
prompt of 1000 tokens in wait, each step holds three decode rows and five prompt
rows. When the three sessions stop, the prompt takes the full row width.

### Sampling and thinking

Each session has its own sampler: a seed, a temperature, top-k, top-p, min-p, a
presence penalty, and a repetition penalty. A turn can change them. The chain
applies the penalties, then top-k, the temperature, min-p, top-p, and the draw. With
top-k 0, every cut is exact over the full vocabulary. The wire also has a frequency
penalty field, and the server refuses a value other than zero. `run` and
`bench-model` always sample greedily.

A model with a think block (Qwen3.5) can think before it answers. `--thinking
auto|on|off` sets the default for each turn, and `auto` takes the thinking mode of
the registry entry of the model. `--thinking-budget N` limits the think tokens of a
turn, and `0` means no limit. When a turn reaches the budget, the scheduler forces
the close tag, and the model continues with the answer. A turn can set both values
for itself. The OpenAI frontend takes them as `chat_template_kwargs.enable_thinking`
and `thinking_token_budget`. There, the vLLM rule applies: a budget of 0 turns
thinking off, and -1 or no field keeps the value of the server.

### State that outlives a turn

- **The prefix cache** (`serve/prefix_manager.zig` over `runtime/prompt_cache.zig`).
  A turn at position 0 looks up the longest stored prefix of its tokens. On a hit,
  the runner restores that prefix and prefills only the rest. When the whole prompt
  is stored with its logits, the turn needs no prompt row at all. After a prefill at
  position 0, the scheduler stores the state when the prefill took more than twice
  the time of a restore. The RAM tier is an LRU list sized by the boot plan.
  `--prefix-disk` adds a disk tier, and `--preload FILE` fills the cache at boot. A
  restore gives the same tokens as a full prefill.
- **The KV tiers** (`serve/residency.zig`). With `--vram-slots` or `--kv-vram`, the
  server can keep more sessions than it has VRAM slots. When a session needs a slot,
  the least recently used resident session moves its written KV rows to a RAM blob.
  When the RAM budget cannot hold one more blob, the move is refused and the turn
  waits until a slot or RAM frees. A blob is the only copy of its session, so the
  server never drops one. With neither flag, every session has its own slot and
  nothing moves.
- **Leases.** A client can ask the server to keep a session after its connection
  drops (`session_keep`), for at most `--resume-ttl` milliseconds. A new connection
  takes the session back with `session_resume`. The default is `0`, no leases.

### Bounds

| Bound | Where it is set |
| --- | --- |
| Requests in the queue of the engine | `serve/engine.zig` |
| Events in the queue of one connection | `serve/engine.zig` |
| Live connections | `--max-connections`, at most `limits.max_connections` |
| Wait of one socket read or write | `--io-timeout`, at most `limits.max_io_timeout_s` (Linux only) |
| Size of one binary frame | a 20-byte header and at most 64 KiB of payload (`serve/protocol.zig`) |
| Sessions | `--max-sessions`, at most `limits.max_batch` on a CPU build. A fused GPU build allows up to `limits.max_logical_sessions` when `--vram-slots` or `--kv-vram` enables the KV tiers |
| Decode rows in one step | `--max-batch`, at most `limits.max_batch` |
| VRAM slots | `--vram-slots`, at most `limits.max_vram_slots` |
| Lease time | `--resume-ttl`, at most `limits.max_resume_ttl_ms` |

### The commands of this layer

| Command | Binary | What it does |
| --- | --- | --- |
| `serve` | release | The binary frontend. |
| `serve-openai` | release | The OpenAI frontend. |
| `client` | release | A terminal chat over the binary frontend, with slash commands for cancel, clone, compact, and the others. |
| `serve-demo` | dev | Starts a server and runs four acts: a streamed turn, follow-up turns, one session cloned and three sessions batched, and a warm open from the prefix cache. |
| `bench-serve` | dev | Aggregate tokens per second with prefill, and latency tails, for each session count up to the most the model holds. |
| `bench-swap` | dev | The time to move one session between VRAM and RAM (fused GPU build). |
| `bench-decode-host` | dev | The host steps of one GPU decode step: record, submit, wait, and the GPU idle gap (fused GPU build). |

One invariant holds under all of this: a session in a batch generates exactly the
tokens that it generates alone. On the CPU the logits are equal bit for bit. Across
backends the greedy tokens are equal. Section 11 names the tests that check it.

## 11. Correctness

- **The reference.** `naive` is scalar code that every other backend must match.
  Each backend declares its tolerance in its `OpRequires`: bit-exact where the
  kernel keeps the reduction order of `naive`, otherwise a measured relative error.
  `tools/bench.zig` prints the relative error of each kernel and marks a line `FAIL`
  when the error passes the tolerance.
- **`gpu-selftest`** checks one GPU kernel against its CPU reference on random
  inputs.
- **The token hash.** `bench-model` prints an FNV-1a hash of the generated token
  ids. Equal hashes across thread counts check determinism. Equal hashes across
  backends and placements check that the paths agree. `tools/gate.py` holds the gates as
  tables. `backends` covers the CPU backends and the per-op Vulkan backend. `threads`
  covers the thread counts, `hybrid` the CPU runner, the fused runner, and three hybrid
  splits, and `deep` a deep prompt.
- **The snapshot test** (`tests/snapshot_smollm2_test.zig`) pins the first greedy
  tokens of a fixed prompt on a real model file. A fault anywhere in the stack fails
  it.
- **The equivalence tests** check that a session in a batch generates the tokens
  of the same session alone. They cover every runner (`tests/*_equivalence_test.zig`
  and the hybrid tests), the wire, and HTTP (the serve and OpenAI integration
  tests).
- **Assertions.** The core asserts its preconditions. `ReleaseSafe` keeps every
  assertion, and `make test` builds `ReleaseSafe`. `ReleaseFast` removes them, so a
  measured binary runs without them. Input from outside the process (a model file,
  a wire frame, a flag) returns a typed error in every build mode, with one stated
  limit: weight values. For a model file, `checkWeights`
  (`runtime/weight_check.zig`) checks the shape of every weight that a runner reads
  and the f32 dtype of each vector, and `linear_dispatch.requireWeights` refuses a
  matrix dtype that no kernel reads. Both run before a runner reads a weight. No
  check scans weight values. A NaN or an infinite weight that a CPU kernel reads
  stops a `ReleaseSafe` build at an assertion. In every other case the value
  reaches the logits, and the sampler returns the error `NonFiniteLogits`.
- **The external reference** is the f32 path of llama.cpp on the same model file and
  prompt. Bit parity with other engines is not a goal, but helps.

## 12. What calls what

These call trees follow a command from `main` to the kernels. The arguments are
simplified.

A release build inlines some functions, for example `generate.decode` and
`layerForward`. perf then shows their work under their callers. A Debug build keeps
every frame ([profiling.md](profiling.md)).

### `run` on the CPU

`zigferencer run SmolLM2-135M-Instruct-Q8_0 --prompt "Hi"` on a CPU build:

```
main -> entry(init, dispatchRelease) -> dispatchOver(argv)   main.zig, cli/command.zig
  run_cmd.run(ctx)                                         cli/run_cmd.zig
    args.parseInto(Flags), select.model(name)              cli/args.zig, cli/select.zig
    resolve.early(.run, flags, entry)                      resolve/resolve.zig: refusals before the load
    loader.load(path)                                      io/gguf/loader.zig: mmap, parse, ModelCard, tensors
    resolve.resolve(.run, flags, card, resources)          resolve/resolve.zig: window, KvPlan, RAM plan
    parallel.initPool(policy)                              runtime/parallel.zig: the worker threads
    vocab.buildFromParsed, bpe.encodeAlloc(prompt)         core/tokenizer/: the prompt ids
      --chat: chat.promptAlloc, bpe.encodePromptAlloc      the prompt text stays plain
    runHost
      CpuRunner(llama_layer).init(loaded, plan, pool)      runtime/cpu_runner.zig: weight views, KV slab
      prefillAndStream(runner, ids)                        cli/run_cmd.zig
        runner.prefillBatch(chunk, pos) -> stepBatch(rows) one chunk of prompt rows
        generate.decode(runner, logits, max_tokens)        runtime/generate.zig: the decode loop
          sampler.greedy(logits) -> id                     runtime/sampler.zig
          sink.onToken(id) -> bpe.decodeAlloc -> stdout
          runner.step(id, pos) -> stepBatch(one row)
```

One step of the Llama runner:

```
CpuRunner(llama_layer).stepBatch(rows)                     runtime/cpu_runner.zig
  ops.embedLookup(token), for each row                     ops = the backend module of the build
  llama_layer.forward(CpuLayerKernels, layer), for each layer
    CpuLayerKernels.norm                                   -> ops.rmsnorm
    CpuLayerKernels.project (q, k, v)                      -> linear_dispatch.linearBatchDispatch -> pool.run -> ops.linearF32Q80Batch
    CpuLayerKernels.rope                                   -> ops.ropeHeads, once per row
    CpuLayerKernels.kvCommit                               writes K and V into the slot, narrowed to the KV dtype
    CpuLayerKernels.attention                              -> parallel.attentionRowsTyped -> ops.attentionPartTyped, one fork for the step
                                                           -> ops.attentionMerge, once per row
    CpuLayerKernels.project (out), .residual               -> linearBatchDispatch, ops.residualAdd
    CpuLayerKernels.norm, .project (gate, up)              -> ops.rmsnorm, linearBatchDispatch
    CpuLayerKernels.swiglu, .project (down), .residual     -> parallel.swiglu -> ops.swiglu, linearBatchDispatch, ops.residualAdd
  ops.rmsnorm, linearBatch(lm_head), on the rows that want logits
```

### `run` on the fused GPU path

`zigferencer run Qwen3.5-2B-Q8_0` on a `vulkan_fused` build. Only the parts that
differ from the CPU tree:

```
run_cmd.run(ctx)
  resolve.resolve(.run, flags, card, resources)            no device capacity yet, so no placement yet
  runFused                                                 cli/run_cmd.zig
    openDevice -> vulkan_ctx.Context.init                  runtime/vulkan_ctx.zig: instance, device, queue, limits
    fusedOn(ctx) -> resolve.resolve with device capacity   cli/boot/boot_resources.zig, resolve/device_placement.zig: region check, VRAM fit, plan
    runFusedOn(ctx, fused, ...)                            cli/run_cmd.zig
      FusedRunner(qwen_layer).init(ctx, loaded, plan)      runtime/fused_runner.zig
        guardKvLayout, the step buffer guards, the VRAM preflight
        the pipelines, the scratch buffers, the KvStore, the weight upload
        the descriptor sets (bound once), DecodeRing.init
      runner.prefillBatch(chunk, pos)                      record the step, submit, wait, copy the logits
      generate.decode -> runner.step(id, pos)
        DecodeRing.step(key, next)                         runtime/decode_ring.zig
          the recorded buffer of this key, or record it now
          queueSubmit, record the predicted next step, wait on the fence
        copy the logits to the host, then sampler.greedy as on the CPU
```

### `run` on the hybrid placement

`zigferencer run Qwen3.5-2B-Q8_0 --gpu-layers 12`. Only the parts that differ from
the fused tree:

```
run_cmd.run(ctx)                                           the pool gets workers again, for the CPU layers
  attemptDevice(pin = 12)                                  cli/run_cmd.zig
    openDevice, DeviceBoot.resolveOn -> DevicePlacement    cli/boot/boot_resources.zig, resolve/device_placement.zig: kv_budget.planLayerSplit picks the split
    HybridRunner(Qwen).init(ctx, loaded, plan)             runtime/hybrid_runner.zig
      FusedRunner(qwen_layer).initSuffix(host_layers)      the GPU holds layers host_layers .. n
      CpuRunner(qwen_layer).initPrefix(host_layers)        the CPU holds layers 0 .. host_layers
    hybrid.step(id, pos)
      hostPrefix(rows): prefix.layerForward for each CPU layer
      crossing(rows): the residual rows, one copy to the GPU
      suffix.stepHidden(x, pos) -> DecodeRing.step as on the fused path
```

### `serve` from boot to the engine loop

`zigferencer serve Qwen3.5-2B-Q8_0 --max-sessions 4` on a `vulkan_fused` build:

```
serve_cmd.run(ctx)                                         cli/serve_cmd.zig
  serve_boot.buildAndServe(flags, launcher)                cli/boot/serve_boot.zig: the engine builder
    resolve.early, loader.load, resolve.resolve(.serve)    the window, the KV type, the chat kind
    parallel.initPool, the vocab, scheduler.Config
    the placement: fused | hybrid | CPU                    the build options and --gpu-layers
      fused: Context.init, resolveOn -> DevicePlacement      the serve rules of the plan
             FusedRunner(qwen_layer).init
    launch(runner, sessions = 4)
      Sessions.init(runner)                                runtime/session.zig: the slot table, the row buffer
      Residency.init(sessions, blob budget)                serve/residency.zig: logical sessions over slots
      BinaryLauncher.serve
        ServeEngine.init: listen, the request queue        serve/engine.zig
          Scheduler.init -> PrefixManager.init             serve/scheduler.zig, serve/prefix_manager.zig
          preloadPrefixes                                  --preload
        printReadyTcp -> "READY tcp host:port"             cli/boot/serve_boot.zig
        ServeEngine.run                                    the engine loop, on the main thread
          acceptorTask                                     a concurrent task
          loop: take requests -> scheduler.handle(req), then scheduler.tick
```

### One turn on the binary frontend

A client sends a `generate` frame:

```
reader task: readerLoop(conn)                              serve/server.zig
  read one frame: zmtp frame head, protocol.decodeHeader, protocol.decodeClientPayload
  toRequest -> Request{ .generate }                        serve/event.zig
  engine.submit(req) -> the request queue                  serve/engine.zig
engine thread: scheduler.handle(req) -> handleGenerate     serve/scheduler.zig
  codec.turnTokens: chat.prompt*Alloc, bpe.encodePromptAlloc  request text stays plain
  startPrefill(entry, tokens) -> tryRestorePrefix          the prefix cache
engine thread, each tick: scheduler.tick -> runComposedStep
  pick the decode rows and one prompt chunk
  Residency.advance -> Sessions.advance                    a session out of VRAM moves back first
    runner.stepBatch(prompt rows + decode rows)
    sampler.sample(logits, params, rng), for each row that wants logits
  acceptToken(entry, token) -> emitOrPark -> sink.emit     the event queue of the connection
writer task: take an event -> writeEvent -> protocol.encodeServerMsg -> a zmtp frame -> the socket
```

### One request on the OpenAI frontend

`POST /v1/chat/completions` with `"stream": true`, on `serve-openai`:

```
acceptorTask -> HttpAdapter.onAccept                       serve/openai/http_server.zig: one task for each connection
  serveConn: receiveHead -> handleRequest -> handleChat
    map.parseChatRequest(body) -> messages, stream, sampling    serve/openai/map.zig
    chat.promptMessagesAlloc(messages) -> text, plain ranges  core/chat.zig
    openSession -> submit(.session_open), wait for session_opened
    streamChat -> runTurn -> submit(.generate)
      the engine runs the turn as on the binary frontend
      each token event -> SseSink.onPiece -> one server-sent event
      done -> the last chunk, then "data: [DONE]"
    closeSession -> submit(.session_close)
```

## 13. Operating notes

- **Measure ReleaseFast binaries only.** A plain `zig build` makes a Debug binary,
  which runs the host code much slower. With the default variables, every `make`
  target that measures a model builds `ReleaseFast`. `make test` builds `ReleaseSafe`,
  and `FLAVOR=structure` of `make profile` builds Debug. Check the optimize mode before
  you look for the cause of a slowdown.
- **Run GPU tests one at a time.** `-Dtest-filter="<text>"` runs only the tests
  whose name holds the text. A test that the filter skips touches no device.
  Read the `Build Summary` line of `zig build test -Druntime=vulkan --summary all`.
  On some drivers a test process exits abnormally after its tests report, and the
  summary line still carries the result.
- **Use the compile-only steps on a host with no GPU.** `zig build check`,
  `test-build`, and `bench-build`, each with `-Druntime=vulkan -Dops-backend=vulkan_fused`,
  compile the fused GPU code and run no kernel. `make ci-gpu` compiles every GPU backend.
- **Stay under the preflight.** The preflight refuses a plan that cannot fit. A plan
  under the line can still fail on a driver whose budget does not show its
  host-visible staging or its residency. After a device loss,
  end the process. `pkill -x 'zigferencer|zigferencer-dev'` ends a process that stays.
- **Output.** stdout carries the program output. A reader that closes the pipe, as `head`
  does, ends the command with exit 1 and no message. A write that fails for another cause,
  such as `/dev/full`, ends the command with exit 1 and one line that names the cause.
- **Logs.** `$ZIGFERENCER_LOG` sets the log level with no rebuild: `off`, `err`,
  `warn`, `info` (the default), `debug`, or a level for each scope, such as
  `info,runner=debug`. `src/core/utils/log.zig` holds the grammar.
- **Files.** `$ZIGFERENCER_MODELS_DIR` moves the model store, and `fetch`
  downloads into it. The default is `models/`. `$ZIGFERENCER_CACHE_DIR` moves the
  disk tier of the prefix cache. The default is `.prompt_cache`.
- **Debugging.** `make debug` and `make debug-test` build for lldb and gdb.
  [debugging.md](debugging.md) has the steps.
