# Developer commands

`zigferencer-dev` is the release binary plus the commands of this directory: the benches,
the kernel checks, the demo, and the tokenizer dump. `dev_main.zig` roots a module of its own,
and `build.zig` gives that module the `zigferencer` module as an import. The `zigferencer`
module holds no path back, so the release binary cannot hold a file of this directory.

A file here reads the engine with `const zigferencer = @import("zigferencer");`. It reads the
CLI building blocks that `src/cli.zig` lists, through `zigferencer.cli`. A relative import out of this
directory puts one file into two modules, and Zig rejects that.

A command with one file sits in this directory. A command with more than one file gets a
directory with its name.

## Files

| File | What it holds |
| --- | --- |
| `dev_main.zig` | The root of the module. It calls `entry` with the developer dispatch, and it holds the `test {}` block of the `dev_unit` test binary. |
| `dev_command.zig` | The table of developer commands and the dispatch over both tables. |
| `tokens_cmd.zig` | `tokens`: print the token ids of a prompt, to compare with llama.cpp. |
| `serve_demo_cmd.zig` | `serve-demo`: start or join a server, then drive a scripted conversation. |
| `bench_serve_cmd.zig` | `bench-serve`: drive concurrent sessions over loopback and report the rate. |
| `bench_swap_cmd.zig` | `bench-swap`: time a KV eviction and restore against a decode step. |
| `bench_decode_host_cmd.zig` | `bench-decode-host`: split one decode step into its host steps. |
| `bench_model/bench_model_cmd.zig` | `bench-model`: check the flags, load the model, and pick a path. |
| `bench_model/bench_model_common.zig` | The flags and the values that the `bench_model_*` files share. |
| `bench_model/bench_model_host.zig` | The `bench-model` path of the CPU backends and the per-op Vulkan backend. |
| `bench_model/bench_model_fused.zig` | The `bench-model` path of the whole model on the GPU. |
| `bench_model/bench_model_placement.zig` | The `bench-model` path of a CPU and GPU split. |
| `bench_model/bench_model_exec.zig` | The run body that all three paths share. |
| `bench_model/bench_model_profile.zig` | `--profile-cpu-decode`: the per-op timer of the CPU runners. |
| `bench_model/bench_model_report.zig` | Print the results as a table or as JSON. |
| `gpu_selftest/gpu_selftest_cmd.zig` | `gpu-selftest`: pick the check of one kernel family. |
| `gpu_selftest/gpu_selftest_attn.zig` | The attention checks against the CPU attention reference. |
| `gpu_selftest/gpu_selftest_matmul.zig` | The matmul check against the naive Q8_0 reference. |
| `gpu_selftest/gpu_selftest_gdn.zig` | The Gated DeltaNet checks against `ops/gdn.zig`. |
| `gpu_selftest/gpu_selftest_common.zig` | The fill and compare helpers of the three checks. |

## How the pieces fit

An arrow means "imports". A name with a slash is the first module outside `src/dev`.

```
dev_main.zig --> dev_command.zig --- dev table ---> <command>_cmd.run(ctx)
                       |                                  |
                       v                                  v
                 zigferencer.cli.command            zigferencer.cli.args, .select, .boot_facts,
                 release table, dispatcher          .boot_lines, zigferencer.family, .resolve,
                                                    .serve_client
```

`bench-model` has one file for each placement, and they share one run body:

```
bench_model_cmd --> zigferencer.family, .resolve
    |
    +--> bench_model_host ------> bench_model_profile --> zigferencer.cpu_profile
    +--> bench_model_fused -----+
    +--> bench_model_placement -+--> bench_model_exec --> bench_model_report
                                         |
                                         +--> zigferencer.generate

bench_model_common: Options, Bench, RunResult. Every file above imports it.
```

## Rules

- `command.dev_names` repeats the names of the dev table, in order, because the release
  binary prints a hint for a developer command and cannot read this directory. A comptime
  check in `dev_command.zig` fails the build when the two lists differ.
- A test here runs in the `dev_unit` binary, not in `unit`. Add a test-bearing file to the
  `test {}` block of `dev_main.zig`.
- When a file here needs another engine module, add one `pub const` line to `main.zig`.
- When a file here needs another CLI building block, add one line to `src/cli.zig`.
