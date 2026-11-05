# src

The engine root holds two files and nine subsystem directories. `main.zig` is the
imperative edge: it routes the process arguments into the command layer.
`main.zig` also re-exports every module under the package name. Zig forbids a
relative import across module roots, and the tests under `tests/` and the commands
under `dev/` need a way in. It carries the `test {}` block that decides which modules
compile their tests into `zig build test`. `cli.zig` lists the CLI files that `dev/` can
read. Everything else lives one level down, and the map above this index is
[docs/architecture.md](../docs/architecture.md).

## Files

| File | Role |
| --- | --- |
| `main.zig` | The entry point of `zigferencer`. `main` calls `entry`, which applies `$ZIGFERENCER_LOG`, then passes control to the `dispatch` function that its caller names. The file also holds the `pub const` re-exports for `tests/`, the `isUserError` exit list, and the `test {}` discovery block. |
| `cli.zig` | The CLI files that a developer command can read. `dev/` reads this layer through this list alone. |
| [`cli/`](./cli/README.md) | argv to one handler per command. The imperative edge. |
| [`dev/`](./dev/README.md) | The developer commands and `dev_main.zig`, the root of `zigferencer-dev`. This directory is a module of its own, and it imports the engine as the `zigferencer` module. |
| [`core/`](./core/README.md) | The boundary values and the pure helpers: `Tensor`, quant geometry, the static bounds, the two planners, the device KV layout, the model table, the tokenizer, and the chat wraps. |
| `engine/` | The boot decisions as pure values. `resolve.zig` takes the flags, the registry entry, the model card, and the host facts. It returns the thinking mode, the KV dtype, the window, and the all-host plan with its RAM quote, or a `Refusal`. `device_arm.zig` adds the device placements from the facts one device context states: the fused fit, the serve budget placements, and the hybrid split. `flags.zig` holds `EngineFlags`, the options it reads. `family.zig` holds one row for each model family: the runner of each placement. |
| [`io/`](./io/README.md) | The I/O shell: an HTTPS download, a read-only whole-file map, and the GGUF parser and loader. |
| [`ops/`](./ops/README.md) | The CPU op variants. `naive` is the reference that every other variant matches. |
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

The command set stops at the dispatcher. `cli/command.zig` owns the release table,
`dev/dev_command.zig` owns the developer table, and `main` names no command. The re-exports and the `test {}` list point at the
same modules for two different reasons. A re-export lets a file under `tests/`
write `@import("zigferencer").loader`. A `test {}` reference pulls that module's
own tests into the test binary.

## Rules

- Add a test-bearing module to the `test {}` block, or its tests never run. The
  compiler stays silent, because an unreferenced module is dead code. A file under
  `dev/` goes in the block of `dev/dev_main.zig`, because the test binary of each
  module runs the tests of that module alone.
- Reference a function by its address (`_ = &f`) to get its body analyzed. A
  discarded value (`_ = f`) resolves the signature and leaves the body alone.
- The eight GPU aliases evaluate to an empty `struct {}` when their build flag is
  off, so the compiler never reads a file it cannot use. `vulkan_ctx`, `ops_vulkan`, `ops_vulkan_wrap`, and `vulkan_runner` gate on `build_options.use_vulkan`. The four runners `llama_fused`, `qwen_fused`, `llama_hybrid`, and `qwen_hybrid` gate one level stricter on `build_options.use_vulkan_fused_ops`. Every construction site tests that flag. The `test {}` block gates the same two ways.
- `isUserError` is the single list of errors whose handler already printed a
  readable line to stderr. `main` exits 1 on one of them with no return trace,
  and it keeps the trace for every other error. A new usage error belongs in
  that switch. An internal failure does not.
- `std_options` routes `std.log` through `core/log.zig`. The comptime ceiling is
  `debug` in every build, and `$ZIGFERENCER_LOG` is the runtime gate with a
  default of `info`. No log line reaches stdout, because stdout carries the
  generated text.

## Where to start

1. `main.zig`: the two edges it owns, the dispatch call and the `test {}` list.
2. [`core/`](./core/README.md): the values every other layer is written in.
3. [`io/`](./io/README.md): how a GGUF file becomes a `ModelCard` and a tensor list.
4. [`ops/`](./ops/README.md): every op of the forward pass in scalar form.
5. [`runtime/`](./runtime/README.md): how those ops compose into a forward pass.
6. [`cli/`](./cli/README.md): how one command wires the layers together.
