# tools

The development edge: per-kernel benchmarks, a symbol-export probe, and the
shell and Python scripts that drive the built binary. It is a separate layer
because it is the relaxed side of the strict-core and relaxed-edge split. It
reaches the engine only across the same value boundaries every other caller
uses, and it never reaches into the numeric core.
[../docs/architecture.md](../docs/architecture.md) holds the layer map and that
split.

The bench and simd-probe roots build at ReleaseFast, but the kernels follow
`-Doptimize`. The Makefile targets pass `-Doptimize=$(OPT)`, and the default is
ReleaseFast. A bare `zig build bench` or `zig build simd-probe` uses Debug
kernels, and each step prints a note then.

Two Zig entry points live here. `bench.zig` is `zig build bench`, and
`simd_probe.zig` is `zig build simd-probe`. Five bench section files and one helper file sit under
`bench.zig`, and thirteen scripts sit beside them. Nine of the thirteen run
from a Makefile target and drive the built binary. The other four never run
from a target. The bench and gate scripts drive the developer binary,
`zig-out/bin/zigferencer-dev`, and a script that builds runs
`zig build install dev`, so both binaries come from one build. The two OpenAI scripts drive the release binary, `zig-out/bin/zigferencer`. The
Makefile sources `menu.sh`, and the two OpenAI scripts source
`openai-common.sh`. `menu.sh` reaches `simd-report.sh`, which reads assembly
rather than the binary. `profile.sh` calls `profile_convert.py`, and
`profile-calls/` holds the function lists that `profile.sh calls` probes. A
script takes its compiler from `ZIG` in the environment, with `zig` on `PATH` as
the default.

## Files

| File | Role |
| --- | --- |
| `bench.zig` | The bench entry point (`zig build bench`, plus `-Druntime=vulkan` for the GPU rows). `main` is the control unit: it owns the GPU runner's lifetime, sets color, and drives the section files. |
| `bench_util.zig` | The shared datapath every section builds rows on: `timed` and `rate`, `computeErrors` for the relative error against a reference, `hashOut` for the output snapshot, and the buffered colored-table reporter. |
| `bench_matmul.zig` | The matmul section: the Q8_0 decode kernel and its variant table, the Q4_0 decode kernel, and the batched prefill Q8_0 kernels. |
| `bench_layer.zig` | The per-layer section: the elementwise ops (residual add, SwiGLU, RMSNorm, RoPE), fused single-head attention, the dispatch-amortization demo, and the Qwen output gate. The attention part also walks a context-depth grid and prices the f16 KV cache. It times the naive and vector kernels on one thread and through the threaded control unit, with a hash check against the serial row. |
| `bench_gdn.zig` | The Gated-DeltaNet section: the `gdn.step` CPU reference on every build, plus the GPU `GdnDecode` (per token) and `GdnChunkParallel` (whole chunk) rows under a Vulkan build. |
| `bench_parallel.zig` | The threading section. It measures the cost of one pool dispatch inside `parallel.linearBatch`, the dispatch cost of empty units on the pool against `std.Io.Group`, and the DRAM read bandwidth of the host against thread count. Each prints a plain table and takes no GPU runner. |
| `bench_transfer.zig` | The host and device transfer section, GPU only. `benchCrossingLatency` prices one crossing at the residual-row payloads, and `benchStagedCopy` measures the bandwidth of one `runStagingCopies` submission in both directions. |
| `simd_probe.zig` | The symbol-export probe. It takes the address of each kernel once per backend that hand-writes it. It parks each address behind `doNotOptimizeAway`, which forces the compiler to emit a standalone named symbol. `linearF32Q80` therefore appears once per CPU backend and `linearF32` once per backend that overrides it. The attention part kernel appears once per CPU backend and KV element type. Every other op is shared `naive` code that each backend re-exports, so the list names it once. The `probed_ops` list is what `simd-report.sh` reads. |
| `bench_model_all.sh` | The cross-backend determinism gate (`make bench-model-all`). It builds each CPU backend in turn, and the per-op vulkan backend under `GPU=1`. It benches the same prompt through each and checks that every eligible backend emits one `tokens_hash`. |
| `bench_llama_hybrid_gate.sh` | The five-way llama placement gate (`make gate-llama-hybrid`). One token hash must cover the CPU runner, the fused runner, the hybrid forced all-host, the hybrid forced all-device, and the hybrid at a mid split. |
| `bench_deep_gate.sh` | The deep token-hash gate (`make gate-deep`). Qwen3.5-2B and SmolLM2 fill 8064 prompt positions and decode 64 tokens in a window of 8192, on the CPU build and on the fused build. Each row must equal the hash on record in the script. `cpu` runs the CPU rows alone and needs no GPU. |
| `bench_ttft_sweep.sh` | The GPU fused TTFT and ITL sweep over prefill length (`make bench-ttft-sweep`). `--prompt-tokens N` synthesizes an exact prefill, so TTFT exposes the prefill work while ITL stays weight-bandwidth-bound and flat. |
| `bench_threads_sweep.sh` | The CPU thread-scaling sweep (`make bench-threads-sweep`). It wraps `bench-model --json` in `/usr/bin/time` and appends one JSONL row per thread count. One invocation takes one fan-out floor (`FLOPS`, default 250000), so a floor sweep needs one invocation per value. |
| `bench_sweep_report.py` | Renders the sweep JSONL into a scaling table: mean and standard deviation per cell, tok/s, efficiency against the one-thread cell, whole-process CPU percentage, and the token hash. It uses the python3 standard library alone. |
| `bench-openai.sh` | Benchmarks the OpenAI HTTP endpoint with curl (`make bench-openai`): sequential and concurrent throughput, mean per-request latency, and streaming time to first byte. |
| `demo-openai.sh` | Drives the OpenAI HTTP endpoint with curl (`make demo-openai`): `/v1/models`, then one buffered and one streamed completion. |
| `openai-common.sh` | The shared spawn, READY wait, and reap block of the two OpenAI scripts. It is sourced, never executed, and it installs the EXIT trap that kills the server. |
| `menu.sh` | The interactive UX and the shared recipe bodies of the Makefile. It is sourced, never executed. It covers variable resolution, the status markers, `backend_flags`, and the bodies `make run`, `make chat`, and their siblings share. |
| `profile.sh` | The profiler driver (`make profile`). It records one command with `perf record -g` into a cell directory. It renders the cell as a flame graph, folded stacks, the call tree down and up, the hot source lines, and the sampled timeline. With root it also traces every call of a function list through uprobes, in order and with stacks, and records where the threads wait off the CPU. It finds a perf built with libelf and DWARF support. It keeps the profiled binary inside the cell, so an old cell still resolves its symbols after a rebuild. With `TRACE` set, the cell also holds the span trace of the engine (`trace.json`), and `report` renders the GPU phases into `flame-gpu.svg`. [../docs/profiling.md](../docs/profiling.md) holds the flavors, the variables, and the span names. |
| `profile_convert.py` | Converts `perf script` text for `profile.sh`. `fold` expands each frame into its inline chain through one batched `addr2line` call, because a release build inlines many boundaries and perf adds no inlined frame. `tree` writes the call trees. `calls` pairs uprobe entries and returns into a call tree per thread and a Chrome trace. `offcpu` weights each wait stack by its time off the CPU. It uses the python3 standard library alone. |
| `profile-calls/` | One function list per flow for `profile.sh calls`: `run.txt` and `serve.txt` name the functions a release build keeps as symbols, at one hit per step or request. `structure-run.txt` names inlined boundaries too, for a Debug build. |
| `simd-report.sh` | The per-op SIMD width and assembly inspector. It reads the assembly the probe emits, in three modes: a width table for every kernel, `list` for the menu, and `asm <substring>` for the drill-down. |

## How the pieces fit

Two toolchains share this directory and never call each other: the bench binary
in Zig, and the probe and report path.

```
  zig build bench [-Druntime=vulkan]
        |
        v
  bench.zig (main)  owns *vulkan_runner.Runner, sets color, calls the sections
        |  passes (gpa, io, runner) into each
        +--> bench_matmul.runQ80 / benchQ40 / benchBatchedMatmul
        +--> bench_layer.benchElementwise / benchSigmoidGate
        +--> bench_gdn.benchGdn / benchGdnChunked
        +--> bench_parallel.benchFanoutCost / benchDispatchCost / benchBandwidth
        +--> bench_transfer.benchCrossingLatency / benchStagedCopy
                 |  every section builds its rows on
                 v
            bench_util  (timed, rate, computeErrors, hashOut, the reporter)

  zig build simd-probe [-Dtarget= -Dcpu=]
        |  emits zig-out/zigferencer-simd-probe.s
        v
  simd_probe.zig (the probed_ops list, the source of truth for which kernels)
        |
        v
  simd-report.sh   reads that .s:  width table | list | asm <name>

  the scripts over the built binary:
    bench_model_all.sh     rebuild per backend (CPU, plus per-op vulkan under GPU=1), run bench-model, compare tokens_hash
    bench_llama_hybrid_gate.sh  build cpu then fused, run five placements, compare tokens_hash
    bench_deep_gate.sh     build cpu then fused, run two models at depth, compare tokens_hash to the record
    bench_ttft_sweep.sh    build fused, sweep bench-model --prompt-tokens N
    bench_threads_sweep.sh bench-model --json per thread count at one floor -> JSONL
                             -> bench_sweep_report.py renders the scaling table
    bench-openai.sh        spawn serve-openai through openai-common.sh, drive it
    demo-openai.sh           with curl
    profile.sh             perf record | perf script -> profile_convert.py fold -> flame.svg,
                             callees.txt, callers.txt; perf report -> lines.txt;
                             with root: uprobes -> calls.txt, calls.json; sched_switch -> flame-offcpu.svg
                             with TRACE: the engine writes trace.json; its .gpu.folded -> flame-gpu.svg
    menu.sh                sourced by the Makefile: variable resolution and the
                           shared recipe bodies
```

Each section file imports `bench_util.zig` as `h` and uses only its row helpers.
A section never imports a sibling section. `bench.zig` is the one file that knows every section exists. It also holds the GPU runner, so every GPU variant borrows one stable `Context` for the whole run.

The `probed_ops` tuple in `simd_probe.zig` is what `simd-report.sh` reads. To add
a kernel to the report, take its address there and leave the script alone. The
shell benches parse the `--json` line of the bench binary with grep and awk
alone. `bench_sweep_report.py` is the one Python file, and it renders the sweep
JSONL from the standard library.

## Rules

- **Every bench row carries an output hash and a relative error against the
  `naive` reference.** A variant that declares bit-exact must match the CPU
  reference exactly. A variant that reorders a reduction must stay inside the
  tolerance its `OpRequires` declares. `computeErrors` measures it and the status
  column fails the row on drift. The bench reports speed beside correctness, and
  never in place of it.
- **`bench.zig` owns the GPU runner's lifetime.** A kernel captures its `Context` at init and dangles if that address moves. So `main` holds the one `*Runner` and every section borrows it.
- **A missing GPU degrades, and never aborts.** A Vulkan init failure prints a
  skip banner and every CPU row still runs.
- **Color is a display concern, local to `bench_util` and `menu.sh`.** Piped
  output stays plain, so it diffs and snapshots cleanly.
- **`bench_model_all.sh` is the cross-backend determinism gate.** Greedy sampling
  is deterministic, so every eligible CPU backend must emit the identical
  `tokens_hash`, and a divergence exits non-zero. The SIMD attention differs
  from naive in bits, inside a declared tolerance. The gate relies on the
  tokens to tolerate that difference, so a mismatch alone does not prove a
  kernel bug. The script skips a backend
  whose instruction set the target lacks, and it does not fail it.
- **`bench_llama_hybrid_gate.sh` is the llama placement gate.** The same hash must cover all
  five placements, because a split forward pass that disagrees with its parents is a
  correctness bug.
- **`bench_deep_gate.sh` is the deep token gate.** Each row must equal the
  hash on record, not only the other executor. A change that moves both
  executors the same way therefore also fails the gate. Investigate a DIFF
  before you update a record.
- **`simd-report.sh` reads the compiler's assembly, never a disassembly.** It
  therefore reports the target's vector width under cross-compilation. A scalar f32 kernel below the SIMD ceiling is deliberate. The f32 paths emit a separate multiply and add, not a fused one. The result stays bit-identical with `naive`, and the token-hash gate depends on that.
- **The tools read engine values and mutate no engine state.** What crosses in
  is `Tensor`, `OpRequires`, and the kernel structs on the `Runner`. The tools
  allocate their own fixtures.

## Where to start

1. `bench_util.zig`: the row helpers, the timing, the error measurement, and the reporter every section uses.
2. `bench.zig`: how `main` owns the runner and sequences the sections.
3. One section, such as `bench_matmul.zig`: the variant table and eligibility shape the others repeat.
4. `simd_probe.zig`, then `simd-report.sh`: the export probe and the inspector that reads its assembly.
5. `bench_model_all.sh`, `bench_llama_hybrid_gate.sh`, and `bench_deep_gate.sh`: the three token-hash gates over the built binary.
6. `menu.sh` last. It is Makefile UX, and part of neither toolchain.
