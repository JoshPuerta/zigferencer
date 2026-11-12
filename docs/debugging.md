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

`BACKEND` picks the ops of the build. Each line below builds both binaries, and each one replaces
the build before it.

| Backend | Build | Where the ops run |
| --- | --- | --- |
| `naive` | `make debug BACKEND=naive` | The CPU, in scalar code. It is the reference. |
| `simd_pf` | `make debug BACKEND=simd_pf` | The CPU, in AVX2 code with prefetch. |
| `vulkan` | `make debug BACKEND=vulkan` | The GPU, one dispatch per operation. |
| `vulkan_fused` | `make debug BACKEND=vulkan_fused` | The GPU, the whole step in one submission. |

## Step through a command

1. If you have no model yet, run `zig-out/bin/zigferencer fetch SmolLM2-135M-Instruct-Q8_0`.
2. Start `run` in lldb, with a stop at the greedy sample:

   ```
   lldb zig-out/bin/zigferencer -o "breakpoint set --name timedGreedy" \
     -o "run run SmolLM2-135M-Instruct-Q8_0 --max-tokens 1 --threads 1"
   ```

3. At the stop, type `frame variable` for the locals and `bt` for the call stack.

A breakpoint by function name stays valid when the code moves. Use `--threads 1`. Then
the main thread runs the whole forward pass, and a breakpoint in a kernel stops on that thread.

`--name` stops in every function of that name. To name one function, use `--func-regex`. Do not
join `--file` and `--name`: the build holds one compile unit, so that pair finds nothing. `--file`
works with `--line`. If `lldb` is not on your path, the CodeLLDB extension holds one in its
`lldb/bin` directory.

To step through the server, use two terminals:

1. In the first terminal, start the server in the debugger:

   ```
   lldb zig-out/bin/zigferencer -o "breakpoint set --func-regex 'Sessions.*advance'" \
     -o "run serve SmolLM2-135M-Instruct-Q8_0 --port 8000 --threads 1"
   ```

2. In the second terminal, connect and type one line: `zig-out/bin/zigferencer client --connect 127.0.0.1:8000`.

The breakpoint stops the engine thread at each step of the session pool, one prefill chunk plus
the decode rows.

## Step through the GPU paths

A GPU build is a normal executable too. The debugger steps the host code: the plan, the record of
each dispatch, the submit, the wait, and the readback. It cannot step through a shader.

Build the backend, then stop at the part that you want:

| Backend | Stop at | Breakpoint |
| --- | --- | --- |
| `vulkan_fused` | One forward step | `--func-regex 'FusedRunner\(.*llama_layer\)\.step$'` |
| `vulkan_fused` | The record of one attention layer | `--func-regex 'attn_record\.record'` |
| `vulkan` | One attention dispatch | `--func-regex 'Attention\.runTyped'` |

For example:

```
make debug BACKEND=vulkan_fused
lldb zig-out/bin/zigferencer -o "breakpoint set --func-regex 'FusedRunner\(.*llama_layer\)\.step$'" \
  -o "run run SmolLM2-135M-Instruct-Q8_0 --prompt Hello --max-tokens 1"
```

To check one kernel family against its CPU reference, run
`zig-out/bin/zigferencer-dev gpu-selftest attn`. The other families are `gdn` and `matmul`.

### Run with no GPU

Set `ZIGFERENCER_VK_SOFTWARE=1`. The engine then takes a CPU Vulkan device, such as Mesa lavapipe,
and no other, and the shaders run on the CPU:

```
ZIGFERENCER_VK_SOFTWARE=1 zig-out/bin/zigferencer run SmolLM2-135M-Instruct-Q8_0 --prompt Hello
```

A warning names the device. With the value `0`, or with no variable, the engine refuses a CPU
device. This mode is for logic and token streams:

- Times mean nothing.
- The subgroup-size pin stays off, because such a device holds fewer lanes.
- A numeric check can differ. `gpu-selftest matmul` can report a row past its tolerance that a GPU
  passes.
- The faults of a GPU driver do not show, such as a lost device.

### Check the Vulkan calls

The Khronos validation layer needs no code change:

```
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation zig-out/bin/zigferencer-dev gpu-selftest attn
```

A clean run prints nothing from the layer, so silence alone does not prove that it loaded. To
prove it, add `VK_LAYER_REPORT_FLAGS=error,warn,info`. The layer then prints
`Khronos Validation Layer Active`.

## Step through a test

1. Build one test binary with only the tests whose name contains the filter: `make debug-test TEST=unit FILTER="decode: greedy walk"`.
2. From the repository root, run `lldb zig-out/test/unit -o "breakpoint set --name timedGreedy" -o run`.

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
- `$ZIGFERENCER_LOG` sets the log level per scope with no rebuild. For example, `ZIGFERENCER_LOG=info,runner=debug` shows all milestones and every `runner` message. The grammar is in `src/core/utils/log.zig`.
- `-Dtrace` compiles the span sites, the per-op CPU timer, and the hot log scopes `layer`, `dispatch`, and `token`. It is off by default in every mode, Debug included, so a Debug build runs no measurement code. For the hot scopes, build with `make debug FLAGS=-Dtrace=true`. [profiling.md](profiling.md) tells how to record the spans.
- A site stays in the source as a call to `core.span`. In a Debug build without trace, the call is an empty inline function, and lldb can still stop on it once. In a `-Dtrace=true` Debug build, the call is a real function. To step over both, run `settings set target.process.thread.step-avoid-regexp '^core\.span\.'`. This replaces any earlier value of that setting. If you keep a filter, add this pattern to it as an alternative.
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
a `glslc` diagnostic. A run with no finding prints nothing. `zig build test` runs the first step too.

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
