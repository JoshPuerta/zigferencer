# src/cli

The CLI turns one process invocation into one of fifteen subcommands, in two
executables. `zigferencer` holds the eight release commands. `zigferencer-dev`
holds those eight plus the seven developer commands. The CLI is the
imperative shell, and the relaxed edge of the strict-core split that
[docs/architecture.md](../../docs/architecture.md) describes. Every command works
through the values the core exposes (`registry.Model`, `ModelCard`, `Vocab`, a
runner, an engine). A loose command file therefore cannot reach into the numeric
core. The state here is transient, the parsed flags and one `Context`, and the
lasting state lives below in `runtime/` and `serve/`.

## Files

| File | Role |
| --- | --- |
| `command.zig` | Two static `{ name, summary, handler }` tables, plus the `Context` value every handler receives. `release` holds `list`, `fetch`, `inspect`, `run`, `serve`, `serve-openai`, `client`, and `info`. `dev` holds `bench-model`, `bench-serve`, `bench-swap`, `bench-decode-host`, `gpu-selftest`, `tokens`, and `serve-demo`. `dispatchRelease` dispatches over `release`, and `dispatchDev` over `release ++ dev`. The release binary knows the dev commands only by name, so it prints a hint for them and never compiles their handlers. The same summary string feeds the usage screen, so help and dispatch cannot drift. The words `--help`, `-h`, and `help` print the usage screen to stdout before any environment check. No argument prints it to stderr. |
| `args.zig` | The flag parser. `parseInto(T, ctx)` fills a value of the flag struct `T` from argv. A field `snake_case` is the flag `--snake-case`, the field default is the flag default, and a nested struct flattens its flags in. The same struct generates the flag lines of `--help`, with each type and default. The errors read the same on every command. A value is missing, a value is not a number, a value is not one of the choices, a flag is unknown, or a flag appears twice. A known flag written as `--name=value` gets a hint to write `--name value`. It also holds the one `writeStderr` and the one `fail` the whole CLI shares. |
| `flags.zig` | The flag values that several commands share. `EngineFlags` (KV dtype, window, GPU layers, threads, thinking mode, thinking budget) nests in `run`, `serve`, `serve-openai`, and the benches. `ServeFlags` holds the knobs of both serve commands. A command hides a knob it does not take through its `omit` list. |
| `select.zig` | Turn an optional `<pub_name>` into a `registry.Model`. The load picker offers only the models on disk. Each row shows the model ceiling and the default windows of `run` and of one serve session. `fetchPick` offers every registry entry. Both name the models directory. |
| `style.zig` | An ANSI `Palette` whose codes become empty strings when color is off, so a call site formats a color span with no branch. The stdout color and the stderr color resolve apart. |
| `prompt.zig` | The interactive stdin reads the subcommands share: `confirm` and `chooseIndex`, each split into an imperative read and a pure parse. Every read resolves to "no" or "no selection" off a terminal. |
| `list_cmd.zig` | `list`: print every registry model, whether its GGUF is on disk, the exact path it loads from, and the resolve URL. |
| `fetch_cmd.zig` | `fetch`: download a GGUF by registry name, or by the `<org>/<repo>/<file>.gguf` path form. It confirms before it replaces a copy already on disk, and a non-interactive stdin refuses instead. |
| `inspect_cmd.zig` | `inspect`: map and parse a fetched GGUF, then print its `ModelCard`, the weight inventory, and a chat-template preview. It is the fastest check after a download. `inspect --raw <path>` skips the loader and dumps every key-value pair and every tensor from the parser. This is the one way to read an architecture the loader refuses. |
| `run_cmd.zig` | `run`: the full pipeline streamed to stdout. Load, build the `Vocab`, tokenize, prefill, then decode greedily. `--chat` applies the model's chat wrap first, and `--thinking` sets the thinking mode of that wrap. `--max-seq` names the KV window, and the default comes from the 2 GiB session budget. |
| `serve_cmd.zig` | `serve`: the binary-protocol connector. It parses its flags and hands a `BinaryLauncher` to the shared builder in `serve_common.zig`. `--resume-ttl MS` sets the longest lease a client can keep past a drop, 0 by default, which turns resumption off. The builder refuses a value past `limits.max_resume_ttl_ms`, one hour, and the error names that limit. |
| `serve_openai_cmd.zig` | `serve-openai`: the OpenAI HTTP connector over `std.http.Server`, serving `/v1/chat/completions` buffered and as SSE, `/v1/models`, and `/healthz`. It hands an `OpenAiLauncher` to the same builder. `--sampling model\|greedy` picks the source of the sampling fields that a request omits: the card row of the resolved thinking mode, or the greedy identity. |
| `serve_common.zig` | The shared engine builder both connectors call: one model load, the runner choice, the tiered-KV wrap, and the two `READY` writers. It also parses `--gpu-layers`, a count or `auto`, and `--thinking` for every command that takes them. `modelFacts` and `DeviceBoot` measure the facts `resolve` reads, once per boot and once per device context. `logFused`, `logClamp`, and `refuse` print the lines of the arm `resolve` returns, the RAM and blob tier lines, and every refusal. It encodes the think tags for the scheduler. `run` and `bench-model` log their arms through the same functions. It refuses a `--thinking-budget` at or past the window. `resolvePolicy` applies the thread flags of every command to the host defaults and logs the result. |
| `serve_demo_cmd.zig` | `serve-demo`: spawn `serve` as a child, or `--connect` to a running one, then drive four acts over the wire. It doubles as a smoke test of the whole deployment path. |
| `client_cmd.zig` | `client`: an interactive chat over the binary wire against a running `serve`. Each typed line is one turn, and the slash commands drive cancel, stats, ping, close, clone, compact, history, and quit. The default mode streams tokens through two tasks and one bounded queue, and `--no-stream` prints each reply whole. A compaction leaves free the positions of the longest turn so far, between one eighth and one half of the window. `--reserve` sets a fixed value. The client compacts at once after a turn that ends `context_full`, and after a refused turn it compacts once and sends that turn again. `--thinking on\|off\|auto` sends the think switch with every turn, and `auto`, the default, sends none. `--thinking-budget N` sends the think budget, 0 unbounded. Both refuse to start against a model with no think tags, except `--thinking off`. `--min-p X` and `--repetition-penalty X` set those sampling fields of the session. `--sampling model` takes the card row of the think mode of the turns from `hello_ack`. `--seed N` sets the seed of the session, and every turn then asks for a reseed, so each turn restarts the sampling stream at N. Without `--seed` the stream continues across turns. The client has no frequency penalty flag, because the server refuses that penalty until it supports it. `--resume-ttl MS` asks the server to keep each session for MS after a drop, 0 by default. The server clamps MS to its own maximum, and the client prints the granted value once. With resumption off on the server, the client prints one note, and a drop ends the chat. On a drop the client tries at most 8 reconnects, with one note per attempt. The first wait is 100 ms, each wait is twice the one before, and the longest is 2 s. An attempt waits at most 10 s for the greeting and 120 s for the resume. The server answers a resume only after the prefill in flight. A `/quit` or the end of stdin ends the wait at once. A new server with another model, window, or end token ends the chat. Otherwise the client resumes the session with the stored token and stores the new token. The verdict of `reconcile` decides the turn in flight. A complete reply needs nothing, a cut reply gets a note, and a turn that the server never ran goes out again. If the server holds no lease, the client opens a new session with the original system prompt and appends the kept turns. Then it sends the turn in flight again. After 8 failed attempts, the client exits with an error. A clone source drops its lease, and `/quit` closes the session and every clone source, so no session waits out a lease. The flag needs the streaming mode. |
| `client_transcript.zig` | The bounded transcript of the chat client and the pure compaction decisions. These are the plan over live costs, the cut of the kept pairs into `session_append` frames, and the refit after a refused frame. They also cover the exact costs that each `appended` reply teaches, the refusal parse, and the retry. `reconcile` classifies the turn in flight at a drop from the `resumed` summary: complete, cut by k tokens, or never run. The held pair bytes equal one append frame, `protocol.append_pairs_bytes_max`, so one held turn always fits one frame. |
| `info_cmd.zig` | `info`: the host hardware card, the ops backend and optimize level of this binary, and the support matrix of `arch_matrix.zig`. It also prints the threading defaults: the physical core count, the default threads, the fan-out floor, and the worker spin. A Vulkan build also opens the device context once and prints its buffer limits and the storage-buffer range that `$ZIGFERENCER_VK_STORAGE_RANGE` sets. The kernel checks belong to `gpu-selftest`. |
| `arch_matrix.zig` | The model architecture against op-backend support matrix, with the backend of this binary marked. `info` prints it as one section. A comptime check pins every row to the canonical backend list, in order. |
| `tokens_cmd.zig` | `tokens`: print a prompt's token ids in the `llama-tokenize --ids` format, so a comparison against llama.cpp costs one command per side. |
| `gpu_selftest_cmd.zig` | `gpu-selftest`: bit-check one GPU kernel (`gdn`, `attn`, or `matmul`) against its CPU reference on random inputs. Each check is one short dispatch, so it localizes what a token hash cannot. `attn` also records decode and prefill plans through the attention kit of the build. |
| `bench_serve_cmd.zig` | `bench-serve`: start a fresh in-process server per batch size, drive that many concurrent sessions over loopback, and report the aggregate rate and the inter-token tails. |
| `bench_swap_cmd.zig` | `bench-swap`: time one KV slot's eviction and its restore against a decode tick. It needs a fused-GPU build, because the swap is a device-to-host transfer. |
| `bench_decode_host_cmd.zig` | `bench-decode-host`: split one decode step into its host legs. Four modes time the record pass alone, a replayed recording, the submit-and-wait floor of an empty buffer, and the GPU idle gap between steps. It needs a fused-GPU build. |
| `bench_model_cmd.zig` | `bench-model`: parse the flags into `common.Options`, load the model, and dispatch to one executor. It reports time to first token, inter-token latency, and an FNV-1a hash of the generated ids. `--runs` must be in 1..`limits.bench_runs_max`. `--json` adds `itl_by_depth`, the mean inter-token latency per 512 KV positions. `--max-seq` names the KV window of the run. Each run benches one prompt and its decode. |
| `bench_model_common.zig` | The shared `Options`, `PlanInfo`, result values, and `DepthCurve` of per-step decode times, plus the leaf helpers more than one sibling needs. It imports no sibling, so the group cannot form an import cycle. |
| `bench_model_exec.zig` | The run body: drive a loaded model through one prompt and its decode, and return the timing and hash values. It is generic over the runner type, so both families and every placement reach it. |
| `bench_model_llama.zig` | The llama `--gpu-layers` placement head and the `--profile-cpu-decode` per-op CPU timer. The timer serves the llama and the qwen35 CPU runners, needs a hot-trace build (`OPT=Debug` or `TRACE=1`), and prints its diagnostic to stderr. |
| `bench_model_qwen.zig` | The Qwen3.5 bench modes: the `--gpu-layers` placement head and the fused-GPU whole-model bench with its two profilers. Only a fused-GPU build imports it, so its device imports are never analyzed on a CPU build. |
| `bench_model_placement.zig` | The family-independent placement body both heads call. It prices a pinned or `auto` split through the real planner and constructs the real hybrid runner. `--force-hybrid` also builds at the degenerate splits, which is the cross-runner token gate. |
| `bench_model_report.zig` | Pure formatting. It renders the result values as the human table and the single-line JSON, and it keeps no state. |

## How the pieces fit

`main.zig` (the release root) and `dev_main.zig` (the dev root) both call
`main.entry`, which hands argv to `command.dispatchRelease` or
`command.dispatchDev`. The dispatcher builds a `Context`. Each handler
is self-contained below that point. `bench-model` is the one command that spans
several files, and they all meet on `bench_model_common`.

```
       main.zig            dev_main.zig
           \                    /
            v                  v
       main.entry(init, .release | .dev)
                 |  Context { io, arena, gpa, args, models_root,
                 |            cache_dir, vk_storage_range, color, color_err }
                 v
   command.dispatchRelease | dispatchDev --match name--> <subcommand>_cmd.run(ctx)
                                                 |
          +---------------+---------------+------+--------+
          v               v               v               v
      args.parseInto select.model     style.Palette   prompt.confirm
      flags and      -> registry      ANSI codes or   TTY-gated stdin
      errors            .Model        empty strings
                                                 |
                                                 v
              core/registry, io/gguf/loader, runtime/*, serve/*

  bench-model spans seven files:

      bench_model_cmd   flags -> Options, dispatch on arch and backend
        |        |          |            |             |
        v        v          v            v             v
      _exec    _llama     _qwen     _placement     _report
      run      llama-     qwen       the placed    values ->
               only       modes      split body    text
               modes
        |        |          |            |             |
        +--------+----------+------------+-------------+
                            v
                   bench_model_common
          Options, PlanInfo, RunResult, DepthCurve.
          It imports no sibling, so no cycle can form.
```

`bench_model_cmd` reads `loaded.card.arch`. A `--gpu-layers` split, a count or
`auto`, routes to the family head, `bench_model_llama` or `bench_model_qwen`,
and both hand the split to `bench_model_placement`. A Qwen3.5 model on a
fused-GPU build otherwise routes to `bench_model_qwen`. Everything else runs
through `bench_model_exec`. The family heads and `bench_model_placement` each
import `exec`, `report`, and `common`, and none imports `cmd`.

`serve` and `serve-openai` split the same way. Each file parses its own flags and
supplies a launcher. `serve_common.buildAndServe` does the rest. It builds one of
six engine flavors, the two model architectures across the CPU, fused-GPU, and
hybrid placements, and wraps it in the tiered-KV residency.

## Rules

- stdout carries program output and stderr carries diagnostics. `run` streams
  text to stdout, and the print commands and the benches write their results
  there. Pickers, confirms, progress bars, and flag errors go to stderr. The two
  serve commands print exactly one machine-readable `READY` line to stdout, and
  route everything else through the scoped stderr logger.
- The command space is closed. `command.release` and `command.dev` are
  compile-time tables, so a reader can enumerate every command. Each command
  accepts the fixed flags of its flag struct, and an unmatched `-` or `--`
  token raises `error.UsageError` instead of passing as a positional argument.
- Every interactive read degrades off a terminal. `confirm` returns false and `chooseIndex` returns null on EOF, on a read error, and on a non-interactive stdin. A piped run or an automated run never blocks. A command that needs a
  distinct message checks `prompt.stdinIsTty` first.
- Color is presentation and resolves once. `style.Palette.init(false)` returns
  empty codes, so a piped run or a `$NO_COLOR` run prints plain text with no
  branch at any call site.
- A handler exchanges values, never handles. It takes the immutable `Context`,
  resolves a `*const registry.Model`, and passes `ModelCard`, `Vocab`,
  `KvPlan`, and result structs to the layers below.
- A usage mistake exits cleanly. `args.fail` prints one `error: ...` line and
  returns `error.UsageError`, and `main` reports that error with no developer
  stack trace.

## Where to start

1. `command.zig`: the two dispatch tables and the `Context` every handler receives.
2. `args.zig` and `flags.zig`, then `style.zig` and `prompt.zig`: the shared parts a handler uses before any work.
3. `select.zig`: how an optional `<pub_name>` becomes a model.
4. `run_cmd.zig`: the smallest handler that walks the whole pipeline.
5. `bench_model_common.zig`, then `bench_model_cmd.zig`: the values first, then the dispatch across the other six files.
6. `serve_common.zig`: the six engine flavors and the launcher boundary.
