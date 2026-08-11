# zigferencer

zigferencer is a small LLM inference engine in Zig 0.16. It loads a GGUF model,
tokenizes a prompt, runs the forward pass, and samples tokens. It runs Llama-family
and Qwen3.5 hybrid models on the CPU (scalar or AVX2) or on the GPU (Vulkan compute).
It exists to be embedded: a host process starts it, and the engine holds many
conversations at once and advances them with batched decode.

Two front ends share one engine. `serve` speaks a framed binary protocol for
embedded interactive traffic: many concurrent, short, low-latency sessions that
often share a prefilled prefix. Its frames ride on ZMTP, so a ZeroMQ DEALER
socket in any language is a client. `serve-openai` speaks the OpenAI HTTP API for
general interop. `run` is the single-prompt command line.

The CPU build has zero dependencies. The GPU build adds one binding-generation
dependency, `vulkan-zig`, and needs `glslc` and `libvulkan` at build time. The
numeric core keeps a strict standard: bounded loops, asserted preconditions, and a
scalar reference implementation that every faster kernel must match.

## Platforms

The project is tested on Linux x86_64. Windows and macOS are compile-checked, not tested. The GPU build needs a Vulkan driver.

## Quick start

You need Zig 0.16.0 on `PATH`. The GPU build also needs `glslc` and `libvulkan-dev`.
A `Makefile` wraps the common `zig build` commands. To use a Zig binary that is not
on `PATH`, run `make <target> ZIG=/path/to/zig`.

```bash
make build                                   # debug build, strongest CPU backend
make test                                    # unit + integration tests + fmt check
make fetch MODEL=SmolLM2-135M-Instruct-Q8_0  # download a registered GGUF (~145 MB)
make run   MODEL=SmolLM2-135M-Instruct-Q8_0 PROMPT="Tell me a joke" N=64
make serve MODEL=SmolLM2-135M-Instruct-Q8_0 SESSIONS=4   # prints one READY line with the port
make info                                    # host CPU/RAM/SIMD + eligible backends
make debug                                   # Debug build for lldb or gdb, see docs/debugging.md
make help                                    # every target and knob
```

The Makefile is optional. The raw commands do the same work:

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zigferencer run SmolLM2-135M-Instruct-Q8_0 --prompt "Hello" --max-tokens 32
./zig-out/bin/zigferencer serve SmolLM2-135M-Instruct-Q8_0 --port 8000
./zig-out/bin/zigferencer client --connect 127.0.0.1:8000   # or any ZeroMQ DEALER socket
zig build -Druntime=vulkan -Dops-backend=vulkan_fused        # GPU build
```

`zig build --help` lists every build flag. `zigferencer --help` lists the release
subcommands. The benches, the kernel checks, and the demo live in a second
executable, `zigferencer-dev`, which `zig build dev` installs beside the first.
Every `make` target that builds installs both.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): the whole-repository map. The
  layers, the build flags, the forward passes, the threading and memory budgets, the
  correctness model, and the serving layer. Start here.
- [`docs/gpu-rdna2.md`](docs/gpu-rdna2.md): the GPU substrate. The CUDA-to-Vulkan
  term map, tiling and coalescing, the occupancy budget, and the bandwidth budget of
  the reference card.
- [`docs/profiling.md`](docs/profiling.md): `make profile`, the perf flavors, the span trace,
  and the GPU timestamp levels.
- [`docs/debugging.md`](docs/debugging.md): step through a command or a test in lldb or gdb.
- Each top-level source directory (`src/*`, `tests/`, `tools/`) carries a `README.md`. It has one row per file, a diagram of how the files fit, and the rules a reader relies on.

## License

MIT. See [`LICENSE`](LICENSE).
