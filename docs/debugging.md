# Debugging

This page tells you how to step through a zigferencer command or test in a
native debugger. You need the Zig toolchain and lldb or gdb. A Debug build keeps
the safety checks, the stack traces, and the error return traces. The build
installs a normal executable, and the debugger runs it. A test is a normal
executable too.

## Build for a debugger

Run `make debug`. It runs `zig build install dev -Doptimize=Debug -Dllvm=true`
and installs `zig-out/bin/zigferencer` and `zig-out/bin/zigferencer-dev`. The
default x86_64 Debug code generator lets a debugger stop at a line, but it emits
no local variables. The LLVM code generator emits them, and gdb also needs it to
find a breakpoint by file and line. A Debug binary is much slower than a release
binary, so its times are not engine times. The next `make` target that builds
replaces it.

## Step through a command

1. If you have no model yet, run `zig-out/bin/zigferencer fetch SmolLM2-135M-Instruct-Q8_0`.
2. Start `run` in lldb, with a stop at the greedy sample:

   ```
   lldb zig-out/bin/zigferencer -o "breakpoint set --file generate.zig --line 154" \
     -o "run run SmolLM2-135M-Instruct-Q8_0 --max-tokens 1 --threads 1"
   ```

3. At the stop, type `frame variable` for the locals and `bt` for the call stack.

A line number moves when the code changes. Before you use a line from this page,
read that line in the file. Use `--threads 1`. Then the main thread runs the
whole forward pass, and a breakpoint in a kernel stops on that thread. The same
lines work with `gdb --args` and `break generate.zig:154`, and `info locals`
lists the locals. A local below the current line holds the `0xaa` fill of
`undefined` until the code writes it.

To step through the server, use two terminals:

1. In the first terminal, start the server in the debugger:

   ```
   lldb zig-out/bin/zigferencer -o "breakpoint set --file session.zig --line 452" \
     -o "run serve SmolLM2-135M-Instruct-Q8_0 --port 8000 --threads 1"
   ```

2. In the second terminal, connect and type one line: `zig-out/bin/zigferencer client --connect 127.0.0.1:8000`.

The breakpoint stops the engine thread when a prompt finishes its prefill. For
the GPU build, run `make debug BACKEND=vulkan_fused`. The host code steps the
same way, but a CPU debugger cannot step through a shader.

## Step through a test

1. Build one test binary with only the tests whose name contains the filter: `make debug-test TEST=unit FILTER="decode: greedy walk"`.
2. From the repository root, run `lldb zig-out/test/unit -o "breakpoint set --file generate.zig --line 154" -o run`.

`TEST=unit` selects the tests of `src/`. The stem of a `tests/*_test.zig` file
selects that integration test, for example `TEST=snapshot_smollm2_test`. An
unknown name fails the build with the list of names. An integration test finds
its model files through `$ZIGFERENCER_MODELS_DIR` (default `models/`). Run the
test binary with no arguments. The test runner then prints one line per test.

## The tools of Zig itself

- A panic and a failed `try` print a stack trace and an error return trace in Debug and ReleaseSafe builds.
- `@breakpoint()` stops an attached debugger at that line.
- `std.debug.dumpCurrentStackTrace(.{})` prints the call stack and continues.
- `$ZIGFERENCER_LOG` sets the log level per scope with no rebuild. For example, `ZIGFERENCER_LOG=info,runner=debug` shows all milestones and every `runner` message. The grammar is in `src/core/log.zig`.
- `-Dtrace` compiles the hot log scopes `layer`, `dispatch`, and `token`, and the per-op span sites. A Debug build has them by default. [profiling.md](profiling.md) tells how to record the spans.
- `zig build -freference-trace` shows the chain of references to a compile error deep in generic code.
- The tidy test fails on a `std.debug.print` call outside a test in `src/`. Thus a temporary print cannot ship.

## An editor

The CodeLLDB extension of VS Code runs the same binaries. Add a configuration
like this one to the `launch.json` file of your workspace:

```json
{
  "type": "lldb",
  "request": "launch",
  "name": "run SmolLM2",
  "program": "${workspaceFolder}/zig-out/bin/zigferencer",
  "args": ["run", "SmolLM2-135M-Instruct-Q8_0", "--max-tokens", "1", "--threads", "1"],
  "cwd": "${workspaceFolder}"
}
```

Run `make debug` before you start the configuration.
