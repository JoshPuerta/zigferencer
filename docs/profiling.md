# Profiling

This page tells you how to find where a zigferencer command spends its time, and
which function calls which.

## What you need

The host profiler works on Linux only. The trace of the engine works wherever the
binary runs.

| Tool | Needed for | Notes |
| --- | --- | --- |
| Linux, x86_64 | `make profile`, `tools/profile.sh` | A Linux virtual machine works. Some virtual-machine kernels refuse uprobes, so `calls` fails there. |
| `perf` with libelf and DWARF support | every perf step | The `linux-tools` package of the distribution. A perf without libelf cannot name a Zig function. `PERF=` names another binary. |
| The FlameGraph scripts and `perl` | the `.svg` files and `diff` | `git clone https://github.com/brendangregg/FlameGraph`, then put `flamegraph.pl` on `PATH` or set `FLAMEGRAPH=` to the clone. |
| `python3`, and `addr2line`, `readelf`, and `nm` (binutils) | the report step and `calls` | `tools/profile_convert.py` uses only the Python standard library. |
| `gawk` | `calls` | |
| A browser | the results | Open `.svg` files directly, `trace.json` in ui.perfetto.dev, and `samples.perf` in profiler.firefox.com or speedscope.app. |

A command writes the trace of the engine to a JSON file when `$ZIGFERENCER_TRACE_OUT`
names one.

> Note: The `calls` and `offcpu` steps need root (`sudo`). Work to replace them is in
> progress, so you can skip them for now.

## Two kinds of evidence

**Samples (perf).** perf stops each thread about 1000 times a second and records
its call stack. A function that shows in many samples takes much time. perf sees
every function of the process in user space. It does not see the GPU,
and it sees a waiting thread only with `offcpu`. Zig keeps frame pointers on x86_64
in every optimize mode, so perf walks the stacks with no rebuild. A release build
inlines many small functions. The report step reads the DWARF inline records with
`addr2line -i`, so an inlined function still shows as a frame with its file and
line.

**Instrumentation (the engine).** The engine records spans at its own boundaries.
Examples are the model load, a prefill chunk, a decode step, a GPU submit and its
wait, a scheduler tick, and a socket read. The result is
a timeline for each thread. On the GPU, the engine records a timestamp at the start
and at the end of each command buffer. At the `phases` level it also records one at
each phase. perf cannot see that time.

| Question | Where to look |
| --- | --- |
| Which host function takes the time? | `flame.svg`, `callees.txt`, `lines.txt` |
| Which functions call this function? | `callers.txt` |
| In what order do things happen on each thread? | `trace.json` (spans) |
| How long does the GPU take for each step and each phase? | the GPU tracks of `trace.json`, `flame-gpu.svg` |
| Where do the threads wait, and for how long? | `flame-offcpu.svg`, `sched.txt` |

## The three build flavors

`make profile` builds the binary in one of three flavors, records one command, and
writes the result into one cell directory.

| Flavor | Build | Use it for |
| --- | --- | --- |
| `release` (the default) | ReleaseFast | Times and stacks of the shipped binary. |
| `trace` | ReleaseFast with `-Dtrace` | The release flavor plus the per-op spans of the engine. |
| `structure` | Debug | Every function is a frame, so the stacks show every call. Its times are not release times. |

## Profile a command

1. Profile one command on the CPU:

   ```
   make profile ARGS="bench-model SmolLM2-135M-Instruct-Q8_0 --ignore-eos" BACKEND=simd_pf
   ```

2. To add the span trace and the GPU timestamps, use the trace flavor and set the
   GPU level:

   ```
   ZIGFERENCER_TRACE_GPU=submit make profile FLAVOR=trace ARGS="run Qwen3.5-2B-Q8_0" BACKEND=vulkan_fused
   ```

3. Open `zig-out/profiles/<commit>/index.html`. It links the reports of every cell.

`ARGS` can name any command of `zigferencer-dev`. `CELL=` names the cell directory,
and `PROFILE_DIR=` moves the profile root.

## Profile a running server

A server does not exit, so it writes its trace after a time window. Start it with
the window, then attach perf for the same time:

```
ZIGFERENCER_TRACE_OUT=/tmp/serve.json ZIGFERENCER_TRACE_SECONDS=20 zig-out/bin/zigferencer serve Qwen3.5-2B-Q8_0 &
TRACE=1 tools/profile.sh attach CELL "$!" 20
tools/profile.sh report CELL
```

Send the server some work during the 20 seconds, for example with
`zig-out/bin/zigferencer client`.

## The steps of `tools/profile.sh`

`make profile` calls this script. Call it directly for a server or for one step.

| Step | What it writes |
| --- | --- |
| `record DIR -- CMD` | `perf.data`, and `trace.json` when `TRACE` is set |
| `attach DIR PID SECONDS` | `perf.data`, and a copy of the trace file of the process when `TRACE` is set |
| `report DIR` | `flame.svg`, `stacks.folded`, `callees.txt`, `callers.txt`, `lines.txt`, `samples.perf`, and `flame-gpu.svg` when the cell holds GPU records |
| `calls DIR LIST -- CMD` | `calls.txt` (the call tree of each thread, in order), `calls.json`, `counts.txt`. `LIST` names the functions to trace, as in `tools/profile-calls/` |
| `offcpu DIR PID SECONDS` | `flame-offcpu.svg` and `sched.txt`: where the threads wait, and for how long |
| `diff A B OUT.svg` | a differential flame graph of two cells |
| `index ROOT` | the index page of a profile root |

## Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `ZIGFERENCER_TRACE_OUT=PATH` | the engine | Turns the span recorder on and names the trace file. While it is unset, a span costs one atomic load and one branch. |
| `ZIGFERENCER_TRACE_SECONDS=N` | the engine | Writes the trace N seconds after the start, or at exit if the exit comes first. |
| `ZIGFERENCER_TRACE_GPU=off\|submit\|phases` | the engine | The GPU timestamp level. `submit` stamps the start and the end of each command buffer. `phases` also stamps each phase of a step. `off`, the default, records no timestamp, so the commands to the GPU do not change. |
| `ZIGFERENCER_TRACE_SPANS=N` | the engine | The capacity of each trace ring. `limits.spans_per_thread_max` is the largest value. The engine refuses a value out of range at the start. |
| `TRACE` | `tools/profile.sh` | Any value sets `ZIGFERENCER_TRACE_OUT=DIR/trace.json` for `record`, and copies the trace file for `attach`. |
| `PERF`, `FLAMEGRAPH`, `FREQ`, `PYTHON` | `tools/profile.sh` | The perf binary, the FlameGraph directory, the sample rate, and the python3 of the converter. |

## Read the results

**A flame graph** shows one box for each function on a stack. The width of a box
is its share of the samples. A wide box near the top is a function that does the
work itself. A wide box near the bottom only calls the work.

**The trace** opens in ui.perfetto.dev. Each thread is one track. The pool workers
show as `zf-pool-N`, and the server threads as `zf-engine`, `zf-read`, `zf-write`,
and `zf-http`. perf shows the same names in its `comm` column. The spans use
`CLOCK_MONOTONIC`, the clock of `perf record -k mono`, so a span and a perf sample
share one time base.

**The GPU tracks.** Each kind of GPU runner gets one track, `GPU queue <kind>`.
The GPU has its own clock, and the engine knows the relation between the two
clocks only at each submit and each wait. Some drivers slow or pause the GPU
counter while the GPU is idle. One offset for the whole run then puts the later
records in the wrong place. The trace writer therefore places each record by its own
offset. That offset is the smallest gap between a host wait and its GPU end, over
the record and the 15 records after it. When a record still falls outside its host window, from the
submit to the wait, the writer moves it inside and counts a clamp. Read the GPU
durations as exact and the GPU positions as close. A tick difference above half the
counter range is a step back of the GPU clock, not a wrap. It counts as zero time,
and `gpu_backsteps` counts it.

**The cost of timestamps.** A timestamp changes the commands of a command buffer.
A time taken at `submit` or `phases` is therefore close to the time without
timestamps, not equal to it. The tokens stay the same at every level. `phases` costs
much more than `submit`. A timestamp after each phase makes the next phase wait
until the GPU drains, so the phases run one after another. For example, a Qwen3.5-2B decode step
on an RDNA2 card took 8.33 ms at `submit` and 28.75 ms at `phases`. Use `submit`
for time and for the timeline. Use `phases` only for the proportions between
phases.

**Drops.** A full span ring drops the new span and counts it. A thread past the
thread table records nothing, and its spans count as `orphan_spans`. The writer
reads the GPU ring up to the first record that is not complete, and counts each
later record as dropped. The `otherData` block of the trace and one log line state
every drop, the clamps, and the largest clamp.

**Ring sizes.** The per-op spans of `-Dtrace` fill a ring of the default size in
seconds. To keep a longer steady state, raise `ZIGFERENCER_TRACE_SPANS`. Each ring
allocates at its first record, so a thread that records nothing costs nothing:

| Ring | Count | Bytes for each entry |
| --- | --- | --- |
| Span ring | one for each thread that records | 24 |
| GPU ring | one | 169 |

For example, at 16384 entries a span ring takes 384 KiB and the GPU ring 2.6 MiB.
At 4194304 entries they take 96 MiB and 676 MiB.

`bench-decode-host --mode gap` reads the GPU idle time from the end stamp of one
decode step to the start stamp of the next. On a driver whose counter pauses while
the GPU is idle, that gap reads low.

## Span names

The boundary spans compile into every build:

| Span | Where |
| --- | --- |
| `model_load` | `loader.loadWith` |
| `tokenize` | `bpe.encodeAlloc` |
| `prefill_chunk` | one prefill chunk of `run` and `bench-model` |
| `decode_step` | one runner step of the decode loop |
| `sample` | one sampler pick |
| `gpu_record`, `gpu_submit`, `gpu_speculate`, `gpu_wait` | the host steps of one GPU submit |
| `sched_tick` | one serve scheduler tick that did work |
| `composed_step` | one serve step with decode rows and a prompt chunk |
| `kv_evict`, `kv_restore` | one KV move of the serve KV tiers |
| `socket_read`, `socket_write` | one binary frame in, one event out |
| `http_chat` | one chat completion request |
| `client_stream` | one frame that the client receives |

The per-op spans compile only under `-Dtrace`:

| Span | Where |
| --- | --- |
| `pool_run` | one fork-join dispatch of the CPU pool |
| `run_share` | the units that one participant runs in one dispatch |
| `worker_wait` | a pool worker between two dispatches |
| `layer_forward` | one layer of a CPU runner |
| `host_prefix` | the CPU layers of a hybrid runner |
| `op_*` | one CPU op group of a decode step, one name for each category of `--profile-cpu-decode` |

The GPU labels are `decode`, `decode_batch`, `prefill_chunk`, and the one-shot
labels `copy`, `fill`, `clone`, and `dispatch`. The GPU phases are the rows of
`--profile-decode`: the five matmul groups, `gdn_conv`, `gdn_scan`, `qknorm`,
`rope`, `attn`, `elementwise`, `norm`, `lmhead`, and `copy`.
