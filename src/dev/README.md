# Developer commands

`zigferencer-dev` is the release binary plus the commands of this directory: the benches, the
kernel checks, the demo, and the tokenizer dump. `dev_main.zig` roots a module of its own, and
`build.zig` gives that module the `zigferencer` module as an import. The `zigferencer` module
holds no path back, so the release binary cannot hold a file of this directory.

## Files

| File | Holds |
| --- | --- |
| `dev_main.zig` | The root of the module and the `test {}` block of the `dev_unit` binary. |
| `dev_command.zig` | The table of developer commands and the dispatch over both tables. |
| `tokens_cmd.zig` | `tokens`: the token ids of a prompt. |
| `serve_demo_cmd.zig` | `serve-demo`: a scripted conversation against a server. |
| `bench_serve_cmd.zig` | `bench-serve`: concurrent sessions over loopback. |
| `bench_swap_cmd.zig` | `bench-swap`: the cost of a KV eviction and restore. |
| `bench_decode_host_cmd.zig` | `bench-decode-host`: the host steps of one decode step. |
| `bench_model/bench_model_cmd.zig` | `bench-model`: the flags, the load, and the choice of path. |
| `bench_model/bench_model_common.zig` | `Options`, `Bench`, `RunResult`, and the other shared values. |
| `bench_model/bench_model_host.zig` | The path of the CPU backends and the per-op Vulkan backend. |
| `bench_model/bench_model_fused.zig` | The path of the whole model on the GPU. |
| `bench_model/bench_model_hybrid.zig` | The path of a CPU and GPU split. |
| `bench_model/bench_model_exec.zig` | The run body of the three paths. |
| `bench_model/bench_model_profile.zig` | `--profile-cpu-decode`: the per-op timer. |
| `bench_model/bench_model_report.zig` | The result table and the JSON. |
| `gpu_selftest/gpu_selftest_cmd.zig` | `gpu-selftest`: the choice of kernel family. |
| `gpu_selftest/gpu_selftest_attn.zig` | The attention checks. |
| `gpu_selftest/gpu_selftest_matmul.zig` | The matmul check. |
| `gpu_selftest/gpu_selftest_gdn.zig` | The Gated DeltaNet checks. |
| `gpu_selftest/gpu_selftest_common.zig` | The fill and compare helpers of the checks. |

## How the pieces fit

Every file here is a command or a part of one. A command takes the process `Context`, parses its
flags with `args`, picks a model with `select`, and reads the engine through the `zigferencer`
module. The CLI blocks come through `src/cli.zig`, which lists the whole boundary between the two
binaries.

| Block | Module | Holds | Read by |
| --- | --- | --- | --- |
| `command` | `cli/command.zig` | The release table and `dev_names` | `dev_command` |
| `Context` | `cli/context.zig` | The `Io`, the allocators, and the stdout writer | every command |
| `args` | `cli/args.zig` | The flag parser, one flag struct per command | every command |
| `select` | `cli/select.zig` | The model picker over the registry | the model commands |
| `boot_resources` | `cli/boot/boot_resources.zig` | The host and device measure, then `resolve` | the benches |
| `boot_lines` | `cli/boot/boot_lines.zig` | The boot lines of a resolved plan | the benches |
| `flag_groups` | `cli/flag_groups.zig` | The shared flag groups | `bench-model`, `bench-serve` |
| `style` | `cli/style.zig` | The terminal palette | `bench-decode-host`, `serve-demo` |

The dispatch runs from the root to one command:

```
  dev_main --> zigferencer.entry(dev_command.dispatch)
                       |
                       v
  dev_command --> command rows --> <name>_cmd.run(Context)
   (dev table + release table)
```

A model command boots like this:

```
  args, select --> registry.Model --> loader.load --> Loaded
                                                                     |
  boot_resources --> engine/resolve --> Resolved --> boot_lines      |
                                            |                        |
                                            v                        v
                                      family.Of(arch) --> runner --> measured rows
```

`bench-model` has one file for each placement, and they share one run body:

```
  bench_model_cmd --> bench_model_common (Options, Bench, RunResult)
      |
      +--> bench_model_host ------> bench_model_profile --> cpu_profile
      +--> bench_model_fused -----+
      +--> bench_model_hybrid ----+--> bench_model_exec --> bench_model_report
                                           |
                                           +--> generate
```

`gpu-selftest` follows the same shape. `gpu_selftest_cmd` picks one of the three check files, and
they share `gpu_selftest_common`. Each check runs one `ops_vulkan` kernel against its CPU
reference. `tokens` and `serve-demo` skip the boot: `tokens` calls `bpe` on a `Vocab`, and
`serve-demo` drives `serve_client` over the binary wire.

## Rules

- A file here reads the engine with `const zigferencer = @import("zigferencer");` and the CLI
  through `zigferencer.cli`. A relative import out of this directory puts one file into two
  modules, and Zig rejects that.
- A command with one file sits in this directory. A command with more than one file gets a
  directory with its name.
- `command.dev_names` repeats the names of the dev table, in order. The release binary prints a
  hint for a developer command and cannot read this directory. A comptime check in
  `dev_command.zig` fails the build when the two lists differ.
- A test here runs in the `dev_unit` binary, not in `unit`. Add a test-bearing file to the
  `test {}` block of `dev_main.zig`.
- When a file here needs another engine module, add one `pub const` line to `main.zig`.
- When a file here needs another CLI building block, add one line to `src/cli.zig`.
