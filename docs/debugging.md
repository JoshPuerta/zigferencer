# Debugging

This page tells you how to step through a zigferencer command or test in a
native debugger. You need the Zig toolchain and lldb or gdb. A Debug build keeps
the safety checks, the stack traces, and the error return traces. The build
installs a normal executable, and the debugger runs it. A test is a normal
executable too.

## Build for a debugger

Run `make debug`. It runs `zig build install dev -Doptimize=Debug -Dllvm=true`
and installs `zig-out/bin/zigferencer` and `zig-out/bin/zigferencer-dev`. The
default x86_64 Debug code generator gives each debugger only a part. lldb stops
at a line but shows no local variables. gdb shows local variables but cannot find
a breakpoint by file and line. The LLVM code generator gives both debuggers both. A Debug binary is much slower than a release
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
whole forward pass, and a breakpoint in a kernel stops on that thread.

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

`TEST=unit` selects the tests of `src/` outside `src/dev/`, and `TEST=dev_unit` selects the tests of `src/dev/`. The stem of a `tests/*_test.zig` file
selects that integration test, for example `TEST=snapshot_smollm2_test`. An
unknown name fails the build with the list of names. An integration test finds
its model files through `$ZIGFERENCER_MODELS_DIR` (default `models/`). Run the
test binary with no arguments. The test runner then prints one line per test.

## Debug build.zig

`zig build` compiles `build.zig` and the Zig build runner into one executable, and
then runs it. The `build` function runs at each `zig build`, also with a cached
runner. Most questions need no debugger:

- `std.debug.print` in `build` writes to stderr at each run. In the `make` function
  of a step, it prints only when the step runs.
- `zig build -l` runs `build`, lists the steps, and stops. It takes about 0.2 s.
- `zig build check -Druntime=vulkan --summary all` prints the tree of steps, with
  the status and the time of each step.
- `--verbose` prints each command, for example each `glslc` line. A cached step runs
  no command. To see every command, add `--cache-dir` with an empty directory.
- `-freference-trace` shows the chain of calls to a compile error in `build.zig`.

To step through `build.zig`:

1. Run `make debug-build`. It compiles the runner with LLVM to
   `zig-out/debug/build-runner`. `zig build` cannot do this, because the runner
   refuses `-fllvm`.
2. From the repository root, run the `lldb` command that the target prints. It stops
   in `resolveOpsBackend`. Change the breakpoint to the line that you need.
3. At the stop, type `frame variable` for the locals and `bt` for the call stack.


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

## Shader rules in an editor

Run `zig build lint-shaders` to check the shader rules of `tools/shader_lint.zig`. Add
`-Druntime=vulkan` to also run `glslc -Werror` and `spirv-val` on each shader. The first step needs
no Vulkan SDK. A finding has the form `path:line:column: error: rule: message`, which is the form of
a `glslc` diagnostic. A run with no finding prints nothing.

VS Code shows these findings with no extension. Add a task like this one to the `tasks.json` file
of your workspace. Run it from the Terminal menu, and read the Problems panel:

```json
{
  "label": "shaders: lint",
  "type": "shell",
  "command": "zig",
  "args": ["build", "lint-shaders", "-Druntime=vulkan"],
  "group": "build",
  "presentation": { "reveal": "silent", "clear": true },
  "problemMatcher": {
    "owner": "shaders",
    "source": "shaders",
    "fileLocation": ["autoDetect", "${workspaceFolder}"],
    "pattern": {
      "regexp": "^(.*\\.comp):(\\d+):(\\d+):\\s+(error|warning):\\s+(.*)$",
      "file": 1, "line": 2, "column": 3, "severity": 4, "message": 5
    }
  }
}
```

To check on each save, run `zig build lint-shaders --watch --summary all --color off` as a
background task. Set `isBackground` to true, and add a `background` block to the matcher:

```json
"background": {
  "activeOnStart": true,
  "beginsPattern": "^(lint-shaders$|Build Summary)",
  "endsPattern": "^lint-shaders (success|cached|failure|transitive failure)"
}
```

The flag `--color off` removes the progress escape codes from the output. The flag
`--summary all` prints two lines for a clean cycle, so the Problems panel clears a fixed finding at
once. A watch holds the build lock while it runs, so call plain `zig` and not a wrapper that shares
the lock with other builds.
