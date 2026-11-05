# CLI

The CLI turns one process invocation into one command, in two executables.
`zigferencer` holds the release commands. `zigferencer-dev` holds those commands
plus the developer ones: the benches, the kernel checks, the demo, and the tokenizer
dump. All lasting state lives below in `runtime/` and `serve/`.

## Files

| File | What it holds |
| --- | --- |
| `command.zig` | The command tables and the `Context` every handler receives. `main.zig` enters here. |
| `args.zig` | The flag parser. One struct states the flags of a command and generates its `--help`. |
| `flags.zig` | The flag groups that several commands share: `EngineFlags` and `ServeFlags`. |
| `select.zig` | Turns an optional model name into a registry entry, with a picker on a terminal. |
| `style.zig` | ANSI color codes that become empty strings when color is off. |
| `prompt.zig` | Interactive questions on stdin. Off a terminal, each one answers no. |
| `list_cmd.zig` | `list`: print the registered models and where each one sits on disk. |
| `fetch_cmd.zig` | `fetch`: download a GGUF by registry name or by path. |
| `inspect_cmd.zig` | `inspect`: print the metadata and the tensors of a GGUF. |
| `info_cmd.zig` | `info`: print the host, the build, and the threading defaults. |
| `arch_matrix.zig` | The table of model architectures against backends that `info` prints. |
| `run_cmd.zig` | `run`: one prompt, from the load to the last token. |
| `serve_common.zig` | The engine builder of both servers. It also holds the boot log lines that `run` and the benches share. |
| `serve_cmd.zig` | `serve`: the binary protocol server. |
| `serve_openai_cmd.zig` | `serve-openai`: the OpenAI HTTP server. |
| `client_cmd.zig` | `client`: an interactive chat against `serve`, with reconnect and session resume. |
| `client_transcript.zig` | The bounded transcript of the chat and the pure compaction decisions. |
| `serve_demo_cmd.zig` | `serve-demo`: start or join a server, then drive a scripted conversation. |
| `tokens_cmd.zig` | `tokens`: print the token ids of a prompt, to compare with llama.cpp. |
| `gpu_selftest_cmd.zig` | `gpu-selftest`: check one GPU kernel against its CPU reference. |
| `bench_model_cmd.zig` | `bench-model`: check the flags, load the model, and pick a path. |
| `bench_model_common.zig` | The flags and the values that the `bench_model_*` files share. |
| `bench_model_host.zig` | The `bench-model` path of the CPU backends and the per-op Vulkan backend. |
| `bench_model_fused.zig` | The `bench-model` path of the whole model on the GPU. |
| `bench_model_placement.zig` | The `bench-model` path of a CPU and GPU split. |
| `bench_model_exec.zig` | The run body that all three paths share. |
| `bench_model_profile.zig` | `--profile-cpu-decode`: the per-op timer of the CPU runners. |
| `bench_model_report.zig` | Print the results as a table or as JSON. |
| `bench_serve_cmd.zig` | `bench-serve`: drive concurrent sessions over loopback and report the rate. |
| `bench_swap_cmd.zig` | `bench-swap`: time a KV eviction and restore against a decode step. |
| `bench_decode_host_cmd.zig` | `bench-decode-host`: split one decode step into its host steps. |

## How the pieces fit

An arrow means "imports". A name with a slash is the first module outside `src/cli`.

Every command starts the same way:

```
main.zig, dev_main.zig
    |
    v
command.zig --- Context ---> <command>_cmd.run(ctx)
                                 |
        +---------+----------+---+------+-----------+
        v         v          v          v           v
    args.zig  flags.zig  select.zig  style.zig  prompt.zig
                  |          |
                  v          v
         engine/flags   core/registry
                        io/gguf/loader
```

The commands that read files or print facts stop early:

```
list_cmd ----> core/registry
fetch_cmd ---> core/registry, io/download
inspect_cmd -> io/gguf/loader, io/gguf/parser
tokens_cmd --> io/gguf/loader, core/tokenizer/bpe
info_cmd ----> arch_matrix --> core/op_requires
           +-> core/hwinfo, runtime/vulkan_ctx
```

The commands that run a model meet in `serve_common` and `engine/`. `engine/resolve`
makes the boot decisions, and `engine/family` holds the runner of each placement,
for each model family. A command turns the architecture into a family once:

```
run_cmd ----------+
serve_cmd --------+--> serve_common --> engine/resolve --> engine/device_arm
serve_openai_cmd -+          |
bench_serve_cmd --+          +--------> engine/family --> runtime/llama_runner
bench_swap_cmd ---+          |                            runtime/llama_fused
bench_decode_host_cmd -------+                            runtime/llama_hybrid
                             |                            runtime/qwen_runner
                             v                            runtime/qwen_fused
                       serve/scheduler                    runtime/qwen_hybrid
                       serve/residency

run_cmd ---------> runtime/generate
serve_cmd -------> serve/server
serve_openai_cmd -> serve/openai/http_server
```

`bench-model` has one file for each placement, and they share one run body:

```
bench_model_cmd --> engine/family, engine/resolve
    |
    +--> bench_model_host ------> bench_model_profile --> runtime/cpu_profile
    +--> bench_model_fused -----+
    +--> bench_model_placement -+--> bench_model_exec --> bench_model_report
                                         |
                                         +--> runtime/generate

bench_model_common: Options, Bench, RunResult. Every file above imports it.
```

The client talks to a server over the wire, so it needs no runner:

```
client_cmd ------> client_transcript --> serve/protocol
           +-----> serve/client
serve_demo_cmd --> serve/client
gpu_selftest_cmd -> ops_vulkan/kernels, ops/naive
```

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
- A command branches on the model family once, with an `inline else` switch over
  `family.Of`. A new family adds one row in `engine/family.zig`, and the build fails
  until the row is complete.
- A handler exchanges values, never handles. It takes the immutable `Context`,
  resolves a `*const registry.Model`, and passes `ModelCard`, `Vocab`,
  `KvPlan`, and result structs to the layers below.
- A usage mistake exits cleanly. `args.fail` prints one `error: ...` line and
  returns `error.UsageError`, and `main` reports that error with no developer
  stack trace.
