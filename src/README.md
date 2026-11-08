# src

The engine root holds `main.zig`, `cli.zig`, and one directory for each subsystem.
`main.zig` also re-exports every module under the package name. Zig forbids a
relative import across module roots, and the tests under `tests/` and the commands
under `dev/` need a way in. It carries the `test {}` block that decides which modules
compile their tests into `zig build test`. `cli.zig` lists the CLI files that `dev/` can
read.

## Files

| File | Role |
| --- | --- |
| `main.zig` | The entry point of `zigferencer`. `main` calls `entry`, which applies `$ZIGFERENCER_LOG`, then passes control to the `dispatch` function that its caller names. The file also holds the `pub const` re-exports for `tests/`, the `isUserError` exit list, and the `test {}` discovery block. |
| `cli.zig` | The CLI files that a developer command can read. `dev/` reads this layer through this list alone. |
| [`cli/`](./cli/README.md) | argv to one handler per command. |
| [`dev/`](./dev/README.md) | The developer commands and `dev_main.zig`, the root of `zigferencer-dev`. This directory is a module of its own, and it imports the engine as the `zigferencer` module. |
| [`core/`](./core/README.md) | The boundary values and the pure helpers: `Tensor`, quant geometry, the static bounds, the two planners, the device KV layout, the model table, the tokenizer, and the chat wraps. |
| [`engine/`](./engine/README.md) | The boot decisions. `resolve.zig` takes the flags, the registry entry, the model card, and the host resources. It returns the thinking mode, the KV dtype, the window, and the all-host plan with its RAM quote, or a `Refusal`. `device_placement.zig` adds the device placements from the resources one device context states: the fused fit, the VRAM budgets, and the hybrid split. `flags.zig` holds `EngineFlags`, the options it reads. `family.zig` holds one row for each model family: the runner of each placement. |
| [`io/`](./io/README.md) | The I/O shell: an HTTPS download, a read-only whole-file map, and the GGUF parser and loader. |
| [`ops/`](./ops/README.md) | The CPU op variants. |
| [`ops_vulkan/`](./ops_vulkan/README.md) | The GPU op backend: one pipeline struct per shader, the upload ring, and the GLSL sources. GPU builds only. |
| [`runtime/`](./runtime/README.md) | The control units that own state: the six runners, the KV cache, the session pool, the sampler, and the prompt cache. |
| [`serve/`](./serve/README.md) | The serving layer: the scheduler, the transport-independent engine, the residency tier, and two transport adapters. |

## How the pieces fit

`main.zig` does three jobs and delegates everything else.

```
  process argv                                  the test binary
       |                                              |
       v                                              v
  main(init) or dev/dev_main.main(init)         test { } block
       |                                              |  one reference per
       v                                              |  test-bearing module
  entry(init, dispatch)                               |
       |  log.setFromEnv($ZIGFERENCER_LOG)            |
       v                                              v
  dispatchRelease | dev_command.dispatch           every `test "..."` in those
       |  ---> cli/command.zig and               modules runs under
       |  dev/dev_command.zig, the               `zig build test`
       v  { name, summary, handler } tables
  isUserError(err)
       yes -> std.process.exit(1), no return trace
       no  -> return the error, trace intact
```

## Rules

- Add a tests to the `test {}` block, or its tests never run. The
  compiler stays silent, because an unreferenced module is dead code. A file under
  `dev/` goes in the block of `dev/dev_main.zig`, because the test binary of each
  module runs the tests of that module alone.
- Reference a function by its address (`_ = &f`) to get its body analyzed. A
  discarded value (`_ = f`) resolves the signature and leaves the body alone.
- The eight GPU aliases evaluate to an empty `struct {}` when their build flag is
  off, so the compiler never reads a file it cannot use. `vulkan_ctx`, `ops_vulkan`, `ops_vulkan_wrap`, and `vulkan_runner` check on `build_options.use_vulkan`. The four runners `llama_fused`, `qwen_fused`, `llama_hybrid`, and `qwen_hybrid` check one level stricter on `build_options.use_vulkan_fused_ops`. Every construction site tests that flag. The `test {}` block gates the same two ways.
- `isUserError` is the single list of errors whose handler already printed a
  readable line to stderr. `main` exits 1 on one of them with no return trace,
  and it keeps the trace for every other error. A new usage error belongs in
  that switch. An internal failure does not.
- `entry` maps a failed write to stdout to an exit. A reader that closes the pipe, as `head`
  does, ends the command with exit 1 and no message. Another cause, such as `/dev/full`, ends it
  with exit 1 and one line that names the cause.
- `std_options` routes `std.log` through `core/log.zig`. The comptime ceiling is
  `debug` in every build, and `$ZIGFERENCER_LOG` is the runtime check with a
  default of `info`. No log line reaches stdout, because stdout carries the
  generated text.
