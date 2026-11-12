# Resolve

This directory resolves the flags and the measured host resources into a setup or a refusal. It
holds the boot decisions as values and the family table. It does not hold the forward pass
(`runtime/`) or the serve loop (`serve/`). A command measures the host once, passes the resources
here, and gets back a plan or a refusal. Nothing here reads a clock, a file, or a
device, and nothing logs, so a table test runs every boot shape with no GPU.

## Files

| File | Holds |
| --- | --- |
| `flags.zig` | `EngineFlags`, the options of one boot. |
| `resolve.zig` | `early` and `resolve`: flags and resources in, `Resolved` or `Refusal` out. |
| `device_placement.zig` | The device placement: the fused fit, the VRAM budgets, and the hybrid split. |
| `family.zig` | One row per model family: the runner of each placement. |

## How the pieces fit

`resolve.zig` is the center. `flags.zig` feeds it, `device_placement.zig` extends it, and
`family.zig` stands apart.

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| `EngineFlags` | `flags.zig` | The KV dtype, the window, the GPU layers, the thread flags, the thinking mode | `cli/args`, from the flags of a command | `resolve`, `device_placement` |
| `Input` | `resolve.zig` | `EngineFlags`, a `ServeShape`, and the bench and prompt-cache switches | `cli/run_cmd`, `cli/boot/serve_boot`, `dev/bench_model` | `early`, `resolve` |
| `Entry` | `resolve.zig` | The registry values of the model | `boot_resources.entryOf` | `early`, `resolve` |
| `Resources` | `resolve.zig` | The free RAM, the `ModelNeeds`, and the `DeviceCapacity` | `cli/boot/boot_resources` | `resolve`, `device_placement` |
| `ModelNeeds` | `device_placement.zig` | The weight census and the fused runner reservations | `boot_resources.modelNeeds` | `device_placement` |
| `DeviceCapacity` | `device_placement.zig` | The live VRAM budget and the buffer limits | `boot_resources` | `device_placement` |
| `Resolved` | `resolve.zig` | The `Early` decisions, the `Window`, the `HostPlacement`, the `DevicePlacement` | `resolve` | `cli/boot/boot_resources`, `dev/bench_model` |
| `Placement` | `device_placement.zig` | The choice of one boot: fused, hybrid, or cpu, with its plan | `choosePlacement` | `cli/run_cmd`, `cli/boot/serve_boot`, `dev/bench_model` |
| `PlacementKind` | `device_placement.zig` | The tag of `Placement` | `placementKindFor` | `device_placement`, `LayerChoice` |
| `DevicePlacement` | `device_placement.zig` | A `LayerChoice` and a `Result(Placement)` | `choosePlacement` | `cli/boot/boot_resources`, `cli/run_cmd`, `cli/boot/serve_boot` |
| `HostPlacement` | `resolve.zig` | The all-host choice: the plan and its RAM quote | `hostPlacement` | `cli/run_cmd`, `cli/boot/serve_boot`, `dev/bench_model` |
| `LayerChoice` | `device_placement.zig` | What a `--gpu-layers` boot chose, for its log lines | `choosePlacement` | `boot_lines.logLayerChoice` |
| `Refusal` | `resolve.zig` | The reason a boot stops, one tag each | `early`, `resolve`, `device_placement` | `cli/boot/boot_lines.refuse` |
| `family.Of(arch)` | `family.zig` | The `Cpu`, `Fused`, and `Hybrid` runner types and the few values that differ | A comptime switch on `loader.Arch` | every command that builds a runner |

The boot is one pass:

```
  cli/args, flag_groups --> EngineFlags --> Input
  registry.Model --------> Entry
                                  |
  boot_resources --> Resources (ModelNeeds, DeviceCapacity)
                                  |
                                  v
  resolve.early(kind, Input, Entry) --> Early | Refusal       before the load
  loader.load --> ModelCard
  resolve.resolve(kind, Input, Entry, card, Resources)        after the load
        |
        +--> core/kv_budget    KvPlan, LayerSplit
        +--> device_placement.choosePlacement    the fused or hybrid placement
        v
  Resolved | Refusal --> the command logs it and builds the runner it names
```

The runner comes from a family row:

```
  loader.card.arch --> switch (arch) { inline else => |a| family.Of(a) }
                                   |
                                   +--> Cpu     runtime/cpu_runner
                                   +--> Fused   runtime/fused_runner
                                   +--> Hybrid  runtime/hybrid_runner
```

Inside the directory, `resolve` imports `device_placement`, and `device_placement` imports
`resolve` back for `Input`, `Window`, `Refusal`, and `Result`. `family.zig` imports neither.

## Rules

- **Pure values.** A function here takes values and returns a value. A new boot decision goes here,
  not in a command, so a table test can reach it.
- **One row per family.** `family.Of` names every architecture, so a new one fails the build until
  its row exists. No command holds an `arch ==` branch.
- **Refusals before the load.** `early` refuses what needs no model file, so a bad flag costs no
  load.
- **The header of `resolve.zig` holds the per-kind rules.** The table tests show every boot shape.
