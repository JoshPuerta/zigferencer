# tools

The development edge: benchmarks, a symbol-export probe, the token-hash gates, and the profiler
driver. It is the relaxed side of the strict-core split and never reaches into the numeric core.
See [../docs/architecture.md](../docs/architecture.md). The bench kernels follow `-Doptimize`, so a
bare `zig build bench` uses Debug kernels and says so.

## Files

| File | Role |
| --- | --- |
| `bench.zig` | The bench entry point (`zig build bench`). It owns the GPU runner and drives the sections. |
| `bench_{matmul,layer,gdn,parallel,transfer}.zig` | The sections. Each imports `bench_util.zig`. |
| `bench_util.zig` | The row helpers: timing, relative error, output hash, and the reporter. |
| `simd_probe.zig` | The export probe (`zig build simd-probe`). `probed_ops` feeds `simd-report.sh`. |
| `shader_lint.zig` | The shader rules (`zig build lint-shaders`). It needs no Vulkan SDK. |
| `gate.sh`, `simd-report.sh` | The token-hash gates (`make gate`), and the SIMD width of each op (`make simd-report`). |
| `profile.sh`, `profile_convert.py` | The perf driver (`make profile`) and its converter, see [profiling](../docs/profiling.md). |
| `debug-build.sh`, `common.sh`, `help.awk` | `make debug-build`, the shared status lines and `zig_build`, and `make help`. |

## The gates

| Gate | Rows | A row must equal | Needs |
| --- | --- | --- | --- |
| `backends` | naive, simd, simd_pf, vulkan | the first row | `--cpu` needs no GPU |
| `threads` | 1, 2, 4, 8 threads | the first row | no GPU |
| `hybrid` | cpu, fused, k=L, k=0, mid | the first row | a GPU |
| `deep` | 2B and SmolLM2, CPU and fused | the hash on record | `--cpu` needs no GPU |

A DIFF in `backends` can come from the SIMD attention, which differs from naive in bits inside a
tolerance, so check the kernel tolerance tests first. A DIFF in `hybrid` is a correctness bug. In
`deep`, investigate before you update a record, and update only after the CPU and fused rows agree.

## Recipes with no script

```bash
for n in 32 64 128 256; do   # the prefill sweep
  zig-out/bin/zigferencer-dev bench-model MODEL --prompt-tokens "$n" --max-tokens 128 --ignore-eos
done
/usr/bin/time -v zig-out/bin/zigferencer-dev bench-model MODEL --threads 8 --runs 3
zig-out/bin/zigferencer list --present | while read -r m; do   # run every model
  zig-out/bin/zigferencer run "$m" --prompt Hello --max-tokens 32
done
```

## Where a command lives

Ask in order. A behavior of the product is a command or flag of the binary. A compile, a test run,
or a format check of one configuration is a `zig build` step. One line with no trap is nothing, so
document it. A join of commands or a fixed default in 3 lines is a `make` target. Else write a script.

## Script rules

The tidy test checks rules 1, 2, and 17 in part. A reader checks the rest.

1. Stay within 100 columns. Use 4 spaces and one statement per line.
2. Start with `#!/usr/bin/env bash` and a header that says why, then `set -euo pipefail`.
   `common.sh` has neither, because it is sourced.
3. Change to the repository root with `cd "$(dirname "${BASH_SOURCE[0]}")/.."`. A script that takes
   paths from the caller stays in place and keeps a `here` variable.
4. Keep this order: constants, state, tables, helpers, `main`. The last line is `main "$@"`.
5. Constants are `readonly` at the top. `parse_args` sets options once, and `main` makes them
   `readonly`. Each function variable is `local`.
6. Use one temp directory from `mktemp -d` in `main`, and remove it with an `EXIT` trap.
7. Offer `usage()` on `-h` and `--help`, to stdout. List options, environment, and exits.
8. Exit 0 for success, 1 for a failed run, and 2 for a usage error, through `die` and `die_usage`.
   Send program output to stdout and status to stderr. Only `common.sh` and `help.awk` emit color.
9. Name each positional word in `usage()`. Read the environment only for a value that arrives
   through make (`ZIG`, `FLAGS`, `BACKEND`, `FLAVOR`) or names a tool (`PERF`, `PYTHON`).
10. Quote every expansion. Pass an argument list as an array, `"${a[@]}"`.
11. Build an array with `read -ra` or `mapfile`. Use `read -r` every time.
12. Write `local x` and `x="$(cmd)"` on two lines when the status of `cmd` counts.
13. Use `[ ]`, and `[[ ]]` only for a glob match. Use `$(...)`, never backticks.
14. Do not loop over `$(ls ...)` or `$(cmd)`. Do not use `eval`. Do not synchronize with `sleep`.
15. Depend on bash 5, coreutils, grep, sed, and POSIX awk. Any other tool appears in `usage()` and
   in the README table of the root, and the script checks for it at the start.
16. Python: start with `#!/usr/bin/env python3` and a docstring that lists the commands. Use the
   standard library alone, type hints on every function, and `NamedTuple` for a record.
17. Use `argparse`, `def main() -> int`, and `sys.exit(main())` under the `__name__` guard.
   Keep no mutable state at module level. Open files with `with` and `encoding="utf-8"`. Send
   errors to stderr, with exit 1. Check with `/usr/bin/python3 -I -m py_compile FILE`.
18. Makefile: a target has at most 3 lines and one `##` line. Only `help` uses `@`. A longer body
   goes to a script. Use no `.ONESHELL` and no `export` line.

Run `shellcheck -x tools/*.sh` when the host has it. `make ci` skips it. Else rules 10 to 13 apply.

## Rules
- **Every bench row carries an output hash and a relative error against `naive`.** A missing GPU
  degrades with a skip banner and never aborts.
- **`simd-report.sh` reads assembly, never a disassembly.** The tools read engine values and
  mutate no engine state. Start with `bench_util.zig`.
