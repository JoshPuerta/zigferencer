<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.svg">
    <img src="docs/assets/logo-light.svg" alt="Zigferencer" width="480">
  </picture>
</h1>

zigferencer is a small inference engine for LLMs (to be extended), written
in Zig 0.16. It reads a model file in the GGUF format and runs the model, with some extra conveniences.

```console
$ zigferencer run Qwen3.5-2B-Q8_0 --chat --prompt "Write a haiku about a GPU." --max-tokens 48
+    0.276ms info(parallel): threads 8 (auto), floor 250000 FLOPs (auto), spin 50000 ns (auto)
+    0.664ms info(gguf): loading Qwen3.5-2B-Q8_0.gguf...
+  109.418ms info(gguf): loaded qwen35: 320 tensors, ctx 262144, 1919 MiB in 109 ms
+  109.535ms info(run): backend: simd_pf
+  171.469ms info(serve): precision: weights q8_0, activations f32, accum f32, kv f16, logits f32
+  171.488ms info(serve): kv window: 174592 positions, the 2048 MiB session budget (model ceiling 262144)
+  171.490ms info(serve): ram: 2065 of 14899 MiB usable = 2065 MiB host kv + 0 MiB blobs + 0 MiB prefix + 0 MiB stages
+ 1090.939ms info(run): qwen cpu runner ready in 919 ms
+ 1090.958ms info(run): generating up to 48 tokens (prompt 20 tokens)
Write a haiku about a GPU.
Silent chips,
Huge heat,
Powerful dreams.
+ 2124.741ms info(run): 15 tokens | TTFT 309.2 ms | ITL 51.8 ms = 19.3 tok/s
```

The same engine serves many conversations at once, through ZMQ or through the OpenAI HTTP API:

```console
$ zigferencer serve-openai Qwen3.5-2B-Q8_0 --port 8080 &
$ curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.5-2B-Q8_0","messages":[{"role":"user","content":"Name three prime numbers."}],"max_tokens":48}'
{"id":"chatcmpl-1","object":"chat.completion","created":0,"model":"Qwen3.5-2B-Q8_0","choices":[{"index":0,"message":{"role":"assistant","content":"Here are three prime numbers:\n\n1.  **2** (the only even prime number)\n2.  **3**\n3.  **7**"},"finish_reason":"stop"}],"usage":{"prompt_tokens":17,"completion_tokens":35,"total_tokens":52}}
```

One of the main goals of this project is to provide a sandbox to tinker with small models and kernels implemented on
non-typical hardware, using Zig. Don't expect a production-grade inference engine (not because of Zig, but due to the scope).
Another side-quest was to minimize dependencies (that's one of the reasons for the size of the repo).

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
| `vulkan` | GPU | One GPU dispatch per operation. It is a correctness reference and does not target performance: one dispatch per operation is costly. |
| `vulkan_fused` | GPU | The whole model step in one GPU submission. It is the fast GPU path. `--gpu-layers` can also split the layers between the CPU and the GPU. |

The servers have no TLS and no authentication, and they bind to 127.0.0.1 by default. To reach
one from another network, put a reverse proxy or a tunnel that adds both in front of it.

The engine is also a Zig module. A dependent package reads it with `dep.module("zigferencer")`,
and its API can change in any release.

## Requirements

Ubuntu 24.04 on x86_64 is the tested platform. These commands install what the quick start needs:

```bash
sudo apt install make curl xz-utils
curl -fsSLO https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz
tar -xf zig-x86_64-linux-0.16.0.tar.xz
export PATH="$PWD/zig-x86_64-linux-0.16.0:$PATH"

# Only for the GPU build:
sudo apt install glslc libvulkan-dev
```

- **CPU build.** It needs only Zig 0.16.0. The binary links no libc and is static. By default the
  build targets the CPU that builds it, and `-Dcpu=x86_64_v3` gives a binary for any x86_64 CPU of
  that level. The CPU build also passes the compile check for Windows and macOS, which are not tested yet.
- **`fetch`.** It downloads over HTTPS from huggingface and needs no account and no token. It does
  not read `HTTPS_PROXY`. It writes to `models/` under the current directory, or to
  `$ZIGFERENCER_MODELS_DIR` when that is set. A command looks in `models/` first, then in that
  store, and `list` prints where each model is.
- **GPU build.** It reads the Vulkan registry that `libvulkan-dev` installs. Another distribution
  passes its path with `-Dvk-registry`.
- **GPU run.** It needs a Vulkan 1.2 driver with `VK_EXT_subgroup_size_control`, 16-bit storage,
  f16 arithmetic, and a subgroup size of 32. On Linux with an AMD card, RADV in the stock
  `mesa-vulkan-drivers` package provides it (not fully tested yet). The tested setup is WSL2 with
  the Mesa Dozen driver, which needs more than the stock packages:
  [`docs/gpu-rdna2.md`](docs/gpu-rdna2.md) lists its versions and its limits.

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
zig-out/bin/zigferencer client --connect 127.0.0.1:8000  # in a second terminal
zig-out/bin/zigferencer info  # host CPU, RAM, SIMD, and the backends this build can run

zig-out/bin/zigferencer fetch Qwen3.5-2B-Q8_0  # the model of the examples above, about 2 GB
```

`make build BACKEND=vulkan_fused` builds the GPU binary, and `make debug` builds for lldb or gdb.

`zig build --help` lists every build option. `zigferencer --help` lists the commands. The
benchmarks and the kernel checks are in a second executable, `zigferencer-dev`.
`make build` installs both.

## Performance

To measure your own host, run:

```bash
zigferencer-dev bench-model MODEL --prompt-tokens 512 --max-tokens 128 --ignore-eos --runs 3
zigferencer-dev bench-serve MODEL --max-tokens 128
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md): how the engine works. The build,
  the forward pass, CPU threads, memory, serving, and correctness.
- [`docs/gpu-rdna2.md`](docs/gpu-rdna2.md): the target GPU. The RDNA2 hardware details
  and how the Vulkan kernels use them.
- [`docs/profiling.md`](docs/profiling.md): how to find where the time is spent.
- [`docs/debugging.md`](docs/debugging.md): step through a command, a test, or `build.zig` in lldb
  or gdb.
- Each top-level source directory (`src/*`, `tests/`, `tools/`) has a `README.md`
  with one row per file and a short guide to the directory.

## Roadmap

Planned after this release, in no fixed order:

- A Windows port, with performance numbers on the native AMD driver.
- MacOS and metal.
- Better kernels.
- Numbers from a native Linux run on RADV.
- More model families, and models with other KV cache designs (a sliding window, sparse attention).
- An evented serve layer.
- More of the OpenAI API: `tools`, `logprobs`, and `stream_options`.

## How this was built

AI assistance was and is heavily used in the development. Most decisions came from experiments and iterations,
but each relevant aspect was agreed, so in case of a complaint, blame the author and not the tool.

## License

MIT. See [`LICENSE`](LICENSE).
