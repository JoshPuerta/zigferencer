# Zigferencer

zigferencer is a small inference engine for LLMs (to be extended to other models), written
in Zig 0.16. It reads a model file in the GGUF format and runs the model, with some extra conveniences.

One of the main goals of this project is to provide a sandbox to tinker with small models and kernels implemented on
non-typical hardware, using Zig. Don't expect a production-grade inference engine (not because of Zig, but due to the scope).
Regarding development, AI assistance was and is heavily used. Most decisions came from experiments and iterations,
but each relevant aspect was agreed, so in case of complain, blame the author and not the tool.

- **Models:** the Llama 3 family and the Qwen3.5 hybrid family (not all quantizations, check registered models with `list`)
- **Hardware:** x86_64 CPUs, with scalar and AVX2 paths, and GPUs through Vulkan.
  The GPU kernels are tuned for one AMD RDNA2 card, but the design aims to support
  extensions (ideally, with limited pain).
- **Use:** one prompt from the command line, or a server that holds many
  conversations at once.

## Commands

| Command | What it does |
| --- | --- |
| `run` | Generates text for one prompt, then exits. It always picks the most likely token, so the same prompt gives the same text. |
| `serve` | Starts a server that keeps many conversations open. It advances all active conversations together, one batched step per token, and it keeps the state of each conversation between turns (clients connect through ZMQ). |
| `serve-openai` | Starts the same server behind the OpenAI HTTP API, so OpenAI clients and tools can use it. |
| `client` | Chats with a running `serve` from the terminal. |
| `list`, `fetch`, `inspect`, `info` | List the registered models, download one, print the metadata of a model file, and print what the host and the build support. |

A **backend** is the set of kernels that computes the operations of the model:
the matrix products, the normalizations, and attention. The build selects one
backend:

| Backend | Where | What it is |
| --- | --- | --- |
| `naive` | CPU | Scalar code. It is the slowest backend, and the reference that every other backend must match. |
| `simd`, `simd_pf` | CPU | AVX2 vector code. `simd_pf` adds software prefetch. It is the default on a CPU with AVX2 and FMA. |
| `vulkan` | GPU | One GPU dispatch per operation. It is a correctness reference, not focusing in performance (dispatch per op is costly). |
| `vulkan_fused` | GPU | The whole model step in one GPU submission. It is the fast GPU path. `--gpu-layers` can also split the layers between the CPU and the GPU. |

The engine is also a Zig module. A dependent package reads it with `dep.module("zigferencer")`,
and its API can change in any release.

## Requirements

The project is tested on Linux x86_64 with Ubuntu 24.04. The CPU build also
compiles for Windows and macOS, but these platforms are not tested yet.

The CPU build needs only Zig 0.16.0 on `PATH`. It uses only
the Zig standard library and links no libc, so the binary is static and runs on any
x86_64 Linux. By default the build targets every feature of the CPU that builds it.

**`fetch`** downloads over HTTPS from huggingface, with the TLS client of the
Zig standard library. It needs:

- The `ca-certificates` package.
- A direct connection (does not read `HTTPS_PROXY`).
- Disk space. For convenience, `$ZIGFERENCER_MODELS_DIR` moves the model store.

Every registered model is in a public repository, so `fetch` needs no account and
no token.

The GPU build needs `glslc` and the Vulkan loader with its headers.
```
sudo apt install glslc libvulkan-dev
```

`libvulkan-dev` also installs the Vulkan registry,
`/usr/share/vulkan/registry/vk.xml`, which the build reads by default. Another
distribution passes its path with `-Dvk-registry`.

A GPU run needs a Vulkan 1.2 driver with `VK_EXT_subgroup_size_control`, 16-bit
storage, and f16 arithmetic. The kernels require a subgroup size of 32.

- On Linux with an AMD card, RADV in the stock `mesa-vulkan-drivers` package
  provides it (not fully tested yet).
- Under WSL2, the driver is Mesa Dozen, which the stock Mesa of Ubuntu 24.04 does
  not include. The tested setup uses Mesa 26.1 from the kisak PPA
  (`ppa:kisak/kisak-mesa`), the LunarG Vulkan SDK 1.4.313, and the AMD driver of
  Windows. [`docs/gpu-rdna2.md`](docs/gpu-rdna2.md) lists the limits of that setup.

**Optional tools:**

| Tool | Needed for |
| --- | --- |
| GNU make, bash | the `Makefile` and `tools/*.sh` |
| `python3` | the profiler |
| `libzmq5` | the ZMQ interop test, which skips without the library |
| `perf`, the FlameGraph scripts | `make profile`, see [`docs/profiling.md`](docs/profiling.md) |
| `lldb` or `gdb` | `make debug`, see [`docs/debugging.md`](docs/debugging.md) |
| the model files | the integration tests. A test skips a missing model file, unless `$ZIGFERENCER_MODELS_DIR` is set |

## Quick start

Install the requirements above first. A `Makefile` wraps the common `zig build` commands. To use a Zig binary that is not
on `PATH`, run `make <target> ZIG=/path/to/zig`.

```bash
make build   # ReleaseFast, the fastest CPU backend the host supports
make test    # unit tests, integration tests, and the format check
make help    # every target and variable

zig-out/bin/zigferencer fetch SmolLM2-135M-Instruct-Q8_0  # download the small test model
zig-out/bin/zigferencer run SmolLM2-135M-Instruct-Q8_0 --prompt "Tell me a joke" --max-tokens 64
zig-out/bin/zigferencer serve SmolLM2-135M-Instruct-Q8_0 --port 8000 --max-sessions 4
zig-out/bin/zigferencer client --connect 127.0.0.1:8000   # or any ZMQ DEALER socket
zig-out/bin/zigferencer info  # host CPU, RAM, SIMD, and the backends this build can run
```

`make build BACKEND=vulkan_fused` builds the GPU binary, and `make debug` builds for lldb or gdb.
The Makefile is a convenience only: it prints each `zig build` command before it runs it. A bare
`zig build` makes a Debug binary, so do not measure it.

`zig build --help` lists every build option. `zigferencer --help` lists the commands. The
benchmarks, the kernel checks, and the demo are in a second executable, `zigferencer-dev`.
`make build` installs both.

## Performance

One desktop made these numbers, the host of the worked example in
[`docs/architecture.md`](docs/architecture.md), section 9.

| Model | Weights | Backend | Prompt + generated tokens | TTFT (mean) | ITL (mean) | Tokens/s, 1 prompt | Tokens/s, 4 sessions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SmolLM2-135M-Instruct | Q8_0 | `simd_pf`, 8 threads | 512 + 128 | 0.97 s | 5.1 ms | 197 | 380 |
| Qwen3.5-2B | Q8_0 | `simd_pf`, 8 threads | 512 + 128 | 8.0 s | 53 ms | 19 | 46 |
| Qwen3.5-2B | Q8_0 | `vulkan_fused` | 512 + 128 | 0.52 s | 8.3 ms | 121 | 66 |
| Llama-3.1-8B-Instruct | Q8_0 | `vulkan_fused` | 512 + 128 | 2.1 s | 24 ms | 41 | 22 |
| Llama-3.1-8B-Instruct | Q4_0 | `vulkan_fused` | 512 + 128 | 2.4 s | 18 ms | 54 | 33 |

 To measure your own host, run:

```bash
zigferencer-dev bench-model MODEL --prompt-tokens 512 --max-tokens 128 --ignore-eos --runs 3
zigferencer-dev bench-serve MODEL --max-tokens 128
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md): how the engine works. The build,
  the forward pass, CPU threads, memory, serving, and correctness. Start here.
- [`docs/gpu-rdna2.md`](docs/gpu-rdna2.md): the target GPU. The RDNA2 hardware details
  and how the Vulkan kernels use them.
- [`docs/profiling.md`](docs/profiling.md): how to find where the time is spent. Linux
  perf on the host, the span trace of the engine (a `-Dtrace=true` build), and GPU timestamps.
- [`docs/debugging.md`](docs/debugging.md): step through a command, a test, or `build.zig` in lldb
  or gdb.
- Each top-level source directory (`src/*`, `tests/`, `tools/`) has a `README.md`
  with one row per file and a short guide to the directory.

## License

MIT. See [`LICENSE`](LICENSE).
