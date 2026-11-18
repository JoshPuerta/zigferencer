# Profiling

This page tells you how to find where the time of a zigferencer command goes, and
which function calls which.

## What you need

The host profiler works on Linux only. The trace of the engine works wherever the
binary runs, if the build has `-Dtrace=true`.

| Tool | Needed for | Notes |
| --- | --- | --- |
| Linux, x86_64 | `make profile`, `tools/profile.sh` | A Linux virtual machine works. |
| `perf` with libelf and DWARF support | every perf step | The `linux-tools` package of the distribution. A perf without libelf cannot name a Zig function. `PERF=` names another binary. |
| The FlameGraph scripts and `perl` | the `.svg` files and `diff` | `git clone https://github.com/brendangregg/FlameGraph`, then put `flamegraph.pl` on `PATH` or set `FLAMEGRAPH=` to the clone. |
| `uv` and `addr2line` (binutils) | the report step | `uv` runs `tools/profile_convert.py` and supplies its Python. The script uses only the Python standard library. |
| A browser | the results | Open `.svg` files directly, `.pftrace` files in ui.perfetto.dev, and `samples.perf` in profiler.firefox.com or speedscope.app. |

A command writes the trace of the engine to a Perfetto protobuf file when
`$ZIGFERENCER_TRACE_OUT` names one. The binary must come from a `-Dtrace=true` build.

> Note: The `offcpu` step needs root (`sudo`).

## Two kinds of evidence

**Samples (perf).** perf stops each thread about 1000 times a second and records
its call stack. A function that shows in many samples takes much time. perf sees
every function of the process in user space. It does not see the GPU,
and it sees a waiting thread only with `offcpu`. Zig keeps frame pointers on x86_64
in every optimize mode, so perf walks the stacks with no rebuild. A release build
inlines many small functions. The report step reads the DWARF inline records with
`addr2line -i`, so an inlined function still shows as a frame with its file and
line.

**Instrumentation (the engine).** The engine records spans at its own boundaries and
around each op. Examples are the model load, a prefill chunk, a decode step, a GPU
submit and its wait, a scheduler tick, and a socket read. The pool records the work
of every worker too. The result is a timeline for each thread. On the GPU, the engine
records a timestamp at the start and at the end of each command buffer. At the `phases`
level it also records one after each op. perf cannot see that time.

The trace exists only in a build with `-Dtrace=true`. A build without it holds no span
site and no clock read for the trace. If such a build sees a `ZIGFERENCER_TRACE_*`
variable, it warns once for each variable and records nothing.

| Question | Where to look |
| --- | --- |
| Which host function takes the time? | `flame.svg`, `callees.txt`, `lines.txt` |
| Which functions call this function? | `callers.txt` |
| In what order do things happen on each thread? | `trace.pftrace` (spans) |
| How long does the GPU take for each step and each op? | the GPU tracks of `trace.pftrace`, `flame-gpu.svg` |
| What did each thread do while the samples fell? | `timeline.pftrace` |
| Where do the threads wait, and for how long? | `flame-offcpu.svg`, `sched.txt` |

## The three build flavors

`make profile` builds the binary in one of three flavors, records one command, and
writes the result into one cell directory.

| Flavor | Build | Use it for |
| --- | --- | --- |
| `release` (the default) | ReleaseFast | Perf only: the flame graph, the trees, and the hot lines of the shipped binary. |
| `trace` | ReleaseFast with `-Dtrace=true` | The release outputs, plus `trace.pftrace`, `samples.pftrace`, and `timeline.pftrace`. |
| `structure` | Debug | Every function is a frame, so the stacks show every call. Its times are not release times. |

A trace build is slower than a release build. Read a trace for order and proportions,
and a release flame graph for times.

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

A server does not exit, so it writes its trace after a time window. The server needs a
`-Dtrace=true` build for the span trace. Start it with the window, then attach perf for
the same time:

```
ZIGFERENCER_TRACE_OUT=/tmp/serve.pftrace ZIGFERENCER_TRACE_SECONDS=20 zig-out/bin/zigferencer serve Qwen3.5-2B-Q8_0 &
TRACE=1 tools/profile.sh attach CELL "$!" 20
tools/profile.sh report CELL
```

Send the server some work during the 20 seconds, for example with
`zig-out/bin/zigferencer client`.

## The steps of `tools/profile.sh`

`make profile` calls `tools/profile.sh all`. Call the script directly for a server or for one
step.

| Step | What it writes |
| --- | --- |
| `all CMD...` | Builds the flavor, then runs `record`, `report`, and `index` on one new cell |
| `record DIR -- CMD` | `perf.data`, and `trace.pftrace` with `trace.pftrace.gpu.folded` when `TRACE` is set |
| `attach DIR PID SECONDS` | `perf.data`, and a copy of the trace file and its `.gpu.folded` file when `TRACE` is set |
| `report DIR` | `flame.svg`, `stacks.folded`, `callees.txt`, `callers.txt`, `lines.txt`, `samples.perf`, `samples.pftrace`, `flame-gpu.svg` when the cell holds GPU records, and `timeline.pftrace` when the cell holds `trace.pftrace` |
| `offcpu DIR PID SECONDS` | `flame-offcpu.svg` and `sched.txt`: where the threads wait, and for how long |
| `diff A B OUT.svg` | a differential flame graph of two cells |
| `index ROOT` | the index page of a profile root |

## Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `ZIGFERENCER_TRACE_OUT=PATH` | the engine | Turns the span recorder on and names the trace file. It needs `-Dtrace=true`. While it is unset, a span costs one atomic load and one branch. |
| `ZIGFERENCER_TRACE_SECONDS=N` | the engine | Needs `-Dtrace=true`. Writes the trace N seconds after the start, or at exit if the exit comes first. |
| `ZIGFERENCER_TRACE_GPU=off\|submit\|phases` | the engine | Needs `-Dtrace=true`. The GPU timestamp level. `submit` stamps the start and the end of each command buffer. `phases` also stamps each op of a step. `off`, the default, records no timestamp, so the commands to the GPU do not change. |
| `ZIGFERENCER_TRACE_SPANS=N` | the engine | Needs `-Dtrace=true`. The capacity of each span log. `limits.spans_per_thread_max` is the largest value. The engine refuses a value out of range at the start. |
| `TRACE` | `tools/profile.sh` | Any value sets `ZIGFERENCER_TRACE_OUT=DIR/trace.pftrace` for `record`, and copies the trace file for `attach`. |
| `PERF`, `FLAMEGRAPH`, `FREQ` | `tools/profile.sh` | The perf binary, the FlameGraph directory, and the sample rate. |
| `BACKEND`, `FLAGS`, `FLAVOR`, `CELL`, `PROFILE_DIR`, `ZIG` | `tools/profile.sh all` | The ops backend, more `zig build` flags, the flavor, the cell name, the profile root, and the zig binary. |

## Read the results

**A flame graph** shows one box for each function on a stack. The width of a box
is its share of the samples. A wide box near the top is a function that does the
work itself. A wide box near the bottom only calls the work.

**The trace** opens in ui.perfetto.dev. The files are Perfetto protobuf. Each thread is
one track: `main`, the pool workers `zf-pool-N`, and the server and client threads
`zf-engine`, `zf-read`, `zf-write`, `zf-http`, and `zf-client`. perf shows the same
names in its `comm` column. The spans use `CLOCK_MONOTONIC`, the clock of
`perf record -k mono`, so a span and a perf sample share one time base.

**The pool work.** A worker shows `worker_spin` while it spins, which costs CPU, and
`worker_sleep` while it waits on the futex. Between them it shows the share that it
runs, named by the op of the dispatch, for example `gate_up_proj`. The dispatching
thread shows the op bracket, then `pool_run`, then its own `run_share`. Each `pool_run`
has a flow arrow to each worker share of that dispatch. The slices carry these args:
`layer`, `tokens`, `rows`, `epoch`, `parts`, and `worker`. A GPU slice adds `gpu_ns`,
`host_submit_ns`, `host_wait_ns`, and at `phases` one `<op>_ns` for each op with time.

**The timeline.** `timeline.pftrace` holds the engine trace and the perf samples in one
file, both on `CLOCK_MONOTONIC`. Each thread shows its slices and its CPU samples on one
track. In ui.perfetto.dev, select a time range on a thread to get the flame graph of the
samples in that range. `samples.pftrace` opens alone too.

**Queries.** `trace_processor_shell`, from perfetto.dev, runs SQL on these files. For
example, put this query in `slices.sql` to count the slices of each thread:

```
select thread.name, count(*) as slices
from slice
join thread_track on slice.track_id = thread_track.id
join thread using (utid)
group by thread.name
order by slices desc;
```

Then run `trace_processor_shell -q slices.sql trace.pftrace`.

**The GPU tracks.** Each kind of GPU runner gets one track, `GPU queue <kind>`.
The GPU has its own clock, and the engine knows the relation between the two
clocks only at each submit and each wait. Some drivers slow or pause the GPU
counter while the GPU is idle. One offset for the whole run then puts the later
records in the wrong place. The trace writer therefore maps each record by its own
offset. That offset is the smallest gap between a host wait and its GPU end, over
the record and the 15 records after it. When a record still falls outside its host window, from the
submit to the wait, the writer moves it inside and counts a clamp. Read the GPU
durations as exact and the GPU positions as close. A tick difference above half the
counter range is a step back of the GPU clock, not a wrap. It counts as zero time,
and `gpu_backsteps` counts it.

**The cost of timestamps.** A timestamp changes the commands of a command buffer.
A time taken at `submit` or `phases` is therefore close to the time without
timestamps, not equal to it. The tokens stay equal at every level. The engine reads
the stamps without the wait flag, after the wait that it already does, so the readback adds
no blocking call. Take a Qwen3.5-2B decode on an RDNA2 card. One run each took 8.83 ms for
each token at `off`, 8.96 ms at `submit`, and 9.05 ms at `phases`. So `phases` is fit for
the timeline and for the proportions between ops.

**Counts.** The trace holds one instant, `trace_meta`, on the process track. It carries
every count: `dropped`, `threads_dropped`, `orphan_spans`, `misnested`, `gpu_records`,
`gpu_dropped`, `gpu_clamps`, `gpu_clamp_max_ns`, `gpu_backsteps`, and `gpu_level`. One log
line at the write states the same counts. A full span log drops the new span and counts
it. A thread past the thread table records nothing, and its spans count as
`orphan_spans`. `misnested` counts a span that overlaps a sibling without nesting.

**Capacity.** Each thread that records gets a span log at its first span. The default
holds 1,048,576 spans (`limits.spans_per_thread`) of 24 bytes each, so it reserves 24 MiB
of address space. Memory pages fill only as spans arrive. A thread that records nothing
costs nothing. `ZIGFERENCER_TRACE_SPANS` sets the capacity, up to 4,194,304. The GPU
records have their own fixed capacity, 65,536 records of 281 bytes
(`limits.trace_gpu_records`). A full log drops the new entry and counts it.

`bench-decode-host --mode gap` reads the GPU idle time from the end stamp of one
decode step to the start stamp of the next. On a driver whose counter pauses while
the GPU is idle, that gap reads low.

## Span names

Every span compiles only under `-Dtrace=true`.

| Span | Where |
| --- | --- |
| `model_load` | `loader.loadWith` |
| `tokenize` | `bpe.encodePromptAlloc`, which `bpe.encodeAlloc` calls |
| `prefill_chunk` | one prefill chunk of `run` and `bench-model` |
| `decode_step` | one runner step of the decode loop |
| `sample` | one sampler pick |
| `gpu_record`, `gpu_submit`, `gpu_speculate`, `gpu_wait`, `gpu_collect` | the host steps of one GPU submit |
| `sched_tick` | one serve scheduler tick that did work |
| `composed_step` | one serve step with decode rows and a prompt chunk |
| `kv_evict`, `kv_restore` | one KV move of the serve KV tiers |
| `socket_read`, `socket_write` | one binary frame in, one event out |
| `http_chat` | one chat completion request |
| `client_stream` | one frame that the client receives |
| `pool_run` | one fork-join dispatch of the CPU pool |
| `run_share` | the units that one participant runs in one dispatch |
| `worker_spin` | a pool worker that spins between two dispatches |
| `worker_sleep` | a pool worker that waits on the futex between two dispatches |
| `layer_forward` | one layer of a CPU runner |
| `host_prefix` | the CPU layers of a hybrid runner |
| an op, such as `attn_proj` | one op of a CPU step, one line of `--profile-cpu-decode` |

The GPU labels are `decode`, `decode_batch`, `prefill_chunk`, and the one-shot
labels `copy`, `fill`, `clone`, and `dispatch`.

## Op names

One list names the ops of the CPU spans and of the GPU marks: `LayerOp` in
`src/core/layer_op.zig`. It holds the ops of a decoder layer, then the ops of the
runner: `embed`, `final_norm`, `lm_head`, and the two transfers of a GPU step,
`stage_in` (the input copy) and `readback` (the logits copy).

Each op has a class: `matmul`, `attention`, `recurrent`, `elementwise`, `norm`, or
`transfer`. The tables of `--profile-cpu-decode`, `--profile-decode`, and
`--profile-prefill` show each class, then the ops of the class.

A GPU op of two passes marks each pass. The GPU table shows the passes under their op,
and the op line holds their sum:

| Op | Passes |
| --- | --- |
| `kv_commit` | `narrow` and `copy`, at a narrow KV dtype |
| `gdn` | `conv` and `scan`, in a prefill chunk |
| `attention` | `split` and `merge`, for each row past one split |

A trace arg names a pass as `op.part`, for example `attention.split_ns`. A folded stack
of `flame-gpu.svg` reads the runner, the label, the class, the op, and the pass. A
kernel that fuses two ops marks both, and its line reads `a+b`. Each such set must stand
in `fused_sets` of `layer_op.zig`.
