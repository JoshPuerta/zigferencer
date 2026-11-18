# CLI

The CLI turns one process invocation into one command. `zigferencer` holds the commands of
this directory. `zigferencer-dev` holds those commands plus the developer commands of
[`src/dev/`](../dev/README.md). All lasting state lives below in `runtime/` and `serve/`.

A command with one file sits in this directory. A command with more than one file gets a
directory with its name. The files that every command reads sit here too, and `boot/` holds
the steps from the flags to a running engine.

## Files

The building blocks:

| File | What it holds |
| --- | --- |
| `command.zig` | The release table and the dispatcher. `main.zig` enters here. |
| `context.zig` | The `Context` every handler receives. |
| `args.zig` | The flag parser. One struct states the flags of a command and generates its `--help`. |
| `flag_groups.zig` | The flag groups that several commands share: `EngineFlags` and `ServeFlags`. |
| `select.zig` | Turns an optional model name into a registry entry, with a picker on a terminal. |
| `style.zig` | ANSI color codes that become empty strings when color is off. |
| `prompt.zig` | Interactive questions on stdin. Off a terminal, each one answers no. |

`src/cli.zig`, one level up, lists the files of this directory that `src/dev/` can read.

`boot/`, the steps from the flags to a running engine:

| File | What it holds |
| --- | --- |
| `boot_resources.zig` | What a boot measures: the model needs, the free RAM and VRAM, the device context, and the thread policy. `run`, the servers, and the benches share it. |
| `boot_lines.zig` | What a boot prints: the boot log lines and the message of each `resolve` refusal. `run`, the servers, and the benches share it. |
| `serve_checks.zig` | The checks of the serve flags that need no model file. |
| `codec_parts.zig` | The token pieces of a frontend codec and the think tags of the scheduler. |
| `serve_boot.zig` | The engine builder of both servers. `buildAndServe` is the entry. |

The commands:

| File | What it holds |
| --- | --- |
| `list_cmd.zig` | `list`: print the registered models and where each one sits on disk. |
| `fetch_cmd.zig` | `fetch`: download a GGUF by registry name or by path. |
| `inspect_cmd.zig` | `inspect`: print the metadata and the tensors of a GGUF. |
| `info/info_cmd.zig` | `info`: print the host, the build, and the threading defaults. |
| `info/arch_matrix.zig` | The table of model architectures against backends that `info` prints. |
| `run_cmd.zig` | `run`: one prompt, from the load to the last token. |
| `serve_cmd.zig` | `serve`: the binary protocol server. |
| `serve_openai_cmd.zig` | `serve-openai`: the OpenAI HTTP server. |

`client/`, an interactive chat against `serve`, with reconnect and session resume:

| File | What it holds |
| --- | --- |
| `client_cmd.zig` | `client`: the start of the chat, then one of the two loops. |
| `client_startup.zig` | The flags, the server check, and the first session. |
| `client_input.zig` | The stdin lines, the slash commands, and the queue of the streaming mode. |
| `client_output.zig` | The turn summary, the compaction notes, and the shades of a reply. |
| `client_history.zig` | The transcript with the numbers that weigh it, and one compaction cycle. |
| `client_transcript.zig` | The bounded transcript of the chat and the pure compaction decisions. |
| `client_chat.zig` | The state of the streaming control unit. |
| `client_streaming.zig` | The streaming mode: the line, frame, and compaction transitions. |
| `client_reconnect.zig` | The `reconnecting` state: the backoff, the attempt tasks, and the resume. |
| `client_blocking.zig` | The `--no-stream` mode: one blocking loop with no tasks. |

## How the pieces fit

The layers point one way. The engine reads no CLI file. A building block reads no command.
`boot/` reads no command. A command reads no other command. `tests/tidy_test.zig` checks the
four rules on every build of the test suite.

```
main.zig --> cli.zig
                |
                v
command.zig --- release table ---> <command>_cmd.run(ctx)
                                       |
       +---------+-------------+-------+------+-----------+
       v         v             v              v           v
   args.zig  flag_groups.zig  select.zig  style.zig  prompt.zig
                  |              |
                  v              v
          resolve/flags    core/registry
                           io/gguf/loader
```

A command takes one of three paths:

| Path | Commands | What it reads below the CLI |
| --- | --- | --- |
| Look something up | `list`, `fetch`, `inspect`, `info` | `core/`, `io/` |
| Run a model | `run`, `serve`, `serve-openai` | `boot/`, `resolve/`, then `runtime/` or `serve/` |
| Talk to a server | `client` | `serve/client`, `serve/protocol` |

A command that runs a model takes four steps:

1. `boot/boot_resources` measures the host and the device.
2. `resolve/resolve` picks the placement, or refuses.
3. `resolve/family` names the runner of that placement.
4. `run_cmd` runs the turn through `runtime/generate`. `boot/serve_boot` builds the engine, and
   `serve_cmd` or `serve_openai_cmd` hands it to its transport.

`client` needs no runner, because it talks to a server over the wire. `client_cmd` picks a loop:
`client_streaming`, or `client_blocking` under `--no-stream`.

## Rules

- stdout carries program output and stderr carries diagnostics. `run` streams
  text to stdout, and the print commands and the benches write their results
  there. Pickers, confirms, progress bars, and flag errors go to stderr. The two
  serve commands print exactly one machine-readable `READY` line to stdout, and
  route everything else through the scoped stderr logger.
- A terminal shows stdout and stderr together. `run` writes each log line of a
  turn before the first byte of the text or after the last byte. Thus the
  prompt echo and the reply stay one block. Under `--chat` the reply starts on
  a new line.
- The command space is closed. `command.release` and the table of
  `src/dev/dev_command.zig` are compile-time tables, so a reader can enumerate every
  command. Each command accepts the fixed flags of its flag struct, and an unmatched `-`
  or `--` token raises `error.UsageError` instead of passing as a positional argument.
- Every interactive read degrades off a terminal. `confirm` returns false and `chooseIndex` returns null on EOF, on a read error, and on a non-interactive stdin. A piped run or an automated run never blocks. A command that needs a
  distinct message checks `prompt.stdinIsTty` first.
- Color is presentation and resolves once. `style.Palette.init(false)` returns
  empty codes, so a piped run or a `$NO_COLOR` run prints plain text with no
  branch at any call site.
- A command branches on the model family once, with an `inline else` switch over
  `family.Of`. A new family adds one entry in `resolve/family.zig`, and the build fails
  until the entry is complete.
- A handler exchanges values, never handles. It takes the immutable `Context`,
  resolves a `*const registry.Model`, and passes `ModelCard`, `Vocab`,
  `KvPlan`, and result structs to the layers below.
- A usage mistake exits cleanly. `args.fail` prints one `error: ...` line and
  returns `error.UsageError`, and `main` reports that error with no developer
  stack trace.
