# Profiling

This page tells you how to see where the time goes in a zigferencer command,
and which function calls which. Linux perf samples the host. The engine adds
what perf cannot see: a span timeline of its own boundaries and the GPU time
of each command buffer.

## The three flavors

`make profile` builds the binary in one of three flavors, records one command,
and renders the result into one cell directory.

| Flavor | Build | Use it for |
| --- | --- | --- |
| `release` (the default) | ReleaseFast | Times and stacks of the shipped binary. |
| `trace` | ReleaseFast with `-Dtrace` | The release flavor plus the span trace of the engine, with the per-op spans. |
| `structure` | Debug | Every function is a frame. Use it to read the call order. Its times are not release times. |

A release build inlines many small functions. The report expands each inlined
frame from the DWARF records, so a flame graph still shows it.

## Commands

1. Profile one command:

   ```
   make profile ARGS="bench-model SmolLM2-135M-Instruct-Q8_0 --ignore-eos" BACKEND=simd_pf
   ```

2. Add the span trace and the GPU queue:

   ```
   ZIGFERENCER_TRACE_GPU=submit make profile FLAVOR=trace ARGS="run Qwen3.5-2B-Q8_0" BACKEND=vulkan_fused
   ```

3. Open `zig-out/profiles/<commit>/index.html`. It links every file of every
   cell.

`tools/profile.sh` is the driver behind the target. Call it directly for a
running server:

```
ZIGFERENCER_TRACE_OUT=/tmp/serve.json ZIGFERENCER_TRACE_SECONDS=20 zig-out/bin/zigferencer serve Qwen3.5-2B-Q8_0 &
TRACE=1 tools/profile.sh attach CELL "$!" 20
tools/profile.sh report CELL
```

| Subcommand | What it writes | Needs root |
| --- | --- | --- |
| `record DIR -- CMD` | `perf.data`, and `trace.json` when `TRACE` is set | no |
| `attach DIR PID SECONDS` | `perf.data`, and a copy of the trace file of the process when `TRACE` is set | no |
| `report DIR` | `flame.svg`, `stacks.folded`, `callees.txt`, `callers.txt`, `lines.txt`, `samples.perf`, and `flame-gpu.svg` when the cell holds `trace.json.gpu.folded` | no |
| `calls DIR LIST -- CMD` | `calls.txt` (the call tree per thread, in order), `calls.json`, `counts.txt` | yes |
| `offcpu DIR PID SECONDS` | `flame-offcpu.svg` and `sched.txt`: where the threads wait, and for how long | yes |
| `diff A B OUT.svg`, `index ROOT` | a differential flame graph, and the index page of a profile root | no |

Open a `.svg` file in a browser. Open `trace.json` and `calls.json` in
ui.perfetto.dev. Open `samples.perf` in profiler.firefox.com or speedscope.app.

## Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `ZIGFERENCER_TRACE_OUT=PATH` | the engine | Turns the span recorder on and names the trace file. With the variable unset, a span site costs one atomic load and one branch. |
| `ZIGFERENCER_TRACE_SECONDS=N` | the engine | Writes the trace N seconds after startup, or at exit if that comes first. A server writes its trace with no shutdown. |
| `ZIGFERENCER_TRACE_GPU=off\|submit\|phases` | the engine | The GPU timestamp level. `submit` writes a start and an end stamp on every GPU command buffer. `phases` also writes every phase mark of the record functions. The default `off` records no timestamp command, so the command stream does not change. |
| `ZIGFERENCER_TRACE_SPANS=N` | the engine | The capacity of every trace ring, from 1 to 4194304 (4 Mi). The default is 16384 (16 Ki). Each thread ring holds N spans, and the GPU ring holds N records. The engine refuses a value outside the range at startup. |
| `TRACE` | `tools/profile.sh` | Any value sets `ZIGFERENCER_TRACE_OUT=DIR/trace.json` for `record`, and copies the trace file for `attach`. |
| `PERF`, `FLAMEGRAPH`, `FREQ`, `PYTHON` | `tools/profile.sh` | The perf binary, the FlameGraph scripts, the sample rate, and the python3 of the converter. |

Timestamps change the command buffer, so a GPU time taken at `submit` or
`phases` is close to, not equal to, the time without them. The token output
stays the same at every level.

`phases` costs much more than `submit`. A bottom-of-pipe timestamp after every
phase stops the next phase until the GPU drains, so the pipeline runs serial.
On Qwen3.5-2B on an RDNA2 card, the fused decode ITL is 8.33 ms at `submit` and
28.75 ms at `phases`. Use `submit` for time and for the timeline. Use `phases` for the
proportions between phases only.

`bench-decode-host --mode gap` reads the GPU idle gap from the end stamp of one
decode step to the start stamp of the next. On a driver whose timestamp
counter pauses while the GPU idles, that gap reads low.

## The trace file

`trace.json` is Chrome Trace Event JSON. Each thread is one track. Pool workers
show as `zf-pool-N` and the serve threads as `zf-engine`, `zf-read`,
`zf-write`, and `zf-http`. The same names show in the `comm` column of perf.
The clock is `CLOCK_MONOTONIC`, the clock of `perf record -k mono`, so a span
and a perf sample share one time base.

Each GPU runner kind gets one track, `GPU queue <kind>`. A GPU clock maps to
the host clock only through the submit and the wait of each command buffer. On
some drivers the timestamp counter is not a stable clock across submits. It can
slow or pause while the GPU idles, so one offset for the whole run misplaces
the later records.

The writer therefore places each record on its own host times. Call the gap
between a host wait and its GPU end the wait gap. A pausing counter only lets
the host clock run ahead, so a later wait gap never falls below the true
offset of an earlier record. The local offset is the smallest wait gap over the
record and the 15 records after it. A clamp moves a record inside its host
window, from the submit to the wait, when a counter breaks that rule.

The duration inside one command buffer stays valid. A tick difference above
half the counter range reads as a step back of the device clock, not as a
wrap. It counts as zero ticks, and `gpu_backsteps` in `otherData` counts it.
`otherData` also holds the window, the clamp count, the largest clamp in
microseconds, and the drop counts.

A full span ring drops the new span and counts it. A thread past the thread
table records nothing and counts as a dropped thread. Its spans count as
`orphan_spans`. The writer reads the GPU ring up to the first record that is
not written yet. That record and each record after it count as dropped. `otherData` and one log line state every
drop.

The per-op spans of `-Dtrace` fill a ring of the default size in seconds. Set
`ZIGFERENCER_TRACE_SPANS` to keep a longer steady state. Price the memory
before you raise N:

| Ring | Count | Bytes per entry | Bytes |
| --- | --- | --- | --- |
| thread spans | 1 per thread that records | 24 | N x 24 per thread, which is 384 KiB at 16 Ki and 96 MiB at 4 Mi |
| GPU records | 1 | 169 | N x 169, which is 2.6 MiB at 16 Ki and 676 MiB at 4 Mi |

A thread allocates its span ring at its first span. A thread that records no
span costs nothing. The first GPU record allocates the GPU ring, so a CPU-only
trace holds no GPU ring. If that allocation fails, every GPU record
counts as dropped.

`trace.json.gpu.folded` holds one line per runner kind, label, and phase with
the GPU nanoseconds. A `submit` record adds its whole time under its runner
kind and label.

## Span names

The boundary spans compile into every build:

| Span | Where |
| --- | --- |
| `model_load` | `loader.loadWith` |
| `tokenize` | `bpe.encodeAlloc` |
| `prefill_chunk` | one prefill chunk of `run` and `bench-model` |
| `decode_step` | one runner step of the decode loop |
| `sample` | one sampler pick |
| `gpu_record`, `gpu_submit`, `gpu_speculate`, `gpu_wait` | the host legs of one GPU submit |
| `sched_tick` | one serve scheduler tick that did work |
| `composed_step` | one composed serve step: decode rows and a prefill chunk |
| `kv_evict`, `kv_restore` | one KV swap of the serve residency |
| `socket_read`, `socket_write` | one binary frame in, one event out |
| `http_chat` | one chat completion request |
| `client_stream` | one frame that the client receives |

The per-op spans compile only under `-Dtrace`:

| Span | Where |
| --- | --- |
| `pool_run` | one fork-join dispatch of the CPU pool |
| `run_share` | the units one participant runs in one dispatch |
| `worker_wait` | a pool worker between two dispatches |
| `layer_forward` | one layer of a CPU runner |
| `host_prefix` | the CPU layer prefix of a hybrid runner |
| `op_*` | one CPU op group of a decode step, one name per category of `--profile-cpu-decode` |

The GPU labels are `decode`, `decode_batch`, `prefill_chunk`, and the one-shot
labels `copy`, `fill`, `clone`, and `dispatch`. The GPU phases are the rows of
`--profile-decode`: the five matmul groups, `gdn_conv`, `gdn_scan`, `qknorm`,
`rope`, `attn`, `elementwise`, `norm`, `lmhead`, and `copy`.
