# GPU substrate: RDNA2 and Vulkan compute

This document gives the hardware and API facts under the GPU backend. The general
data-parallel rules say that the work hierarchy and the memory traffic decide
performance, not the kernel body. This document maps those rules onto the substrate the kernels run on. It gives the term map from the textbook to the Vulkan API and the two techniques that pay for inference kernels. It also gives the occupancy budget, the numbers of the reference card, and the bandwidth budget that sets the performance target.

Read this before you write or tune a shader.

## 1. Term map

The reference textbook for GPU compute is *Programming Massively Parallel Processors*.
Read it for the mental model, not for the syntax. The CUDA examples translate to AMD
and Vulkan with a name swap. The hierarchy underneath is the same, and that hierarchy
decides whether a kernel runs at peak.

| CUDA | Vulkan / AMD |
|---|---|
| thread / warp / block / grid | invocation / subgroup / workgroup / dispatch |
| shared memory | GLSL `shared`, which is the AMD LDS (Local Data Share) |
| SM (streaming multiprocessor) | AMD CU (Compute Unit). RDNA pairs two CUs into a WGP (Workgroup Processor). |
| warp size 32 | RDNA wave32, the compute default. Wave64 is the older GCN width, and the compiler can still select it. |
| constant memory | Vulkan push constants and uniform buffers |

## 2. The two techniques that pay for inference kernels

See PMPP chapters 4, 5, and 6.

1. **Tiled matmul with LDS staging.** One workgroup computes one output tile. The
   workgroup loads `tile_M x tile_K` of A and `tile_K x tile_N` of B from global
   memory into LDS one time. Then every invocation in the workgroup reads from LDS.
   This cuts global-memory traffic by a factor of `tile_K`.
2. **Coalesced global loads.** Adjacent invocations in a wave must touch adjacent
   global addresses. A pattern that strides across a wave wastes the cache line.

## 3. Occupancy budget

See PMPP chapter 5. Each CU runs N waves in flight. Two pressures cap N: the VGPRs
of each wave, and the LDS bytes of each workgroup. Stay under both budgets to keep
two or more waves resident. Resident waves hide memory latency.

## 4. The reference card: RX 6800 XT, gfx1030, RDNA2

The budgets in this document and the tile constants in the shaders are computed for
this card. Another RDNA2 or RDNA3 card changes the numbers, not the method.

- 72 CUs and 36 WGPs. Each WGP is two CUs that share a scheduler and 128 KB L1.
- 16 GB GDDR6 at about 512 GB/s peak.
- 128 MB Infinity Cache (L3) as a bandwidth amplifier. It holds SmolLM2 entirely. It
  holds about 12 percent of an 8B Q8_0 model.
- 256 VGPRs for each wave. 64 KB LDS for each CU. 16 KB scalar L1 and 16 KB vector
  L1 for each WGP. 4 MB L2.
- Wave32 is the compute default, because RDNA breaks from the wave64 default of GCN.
  The compiler can select wave64 for a high-occupancy or branchy kernel. The engine
  pins wave32 on every compute pipeline through the subgroup-size-control extension.
- A translation-layer Vulkan driver (Vulkan on D3D12) exposes Vulkan 1.2, 16-bit
  storage, and fp16 arithmetic, but no 8-bit storage and no `shaderInt8`. The
  kernels therefore pack four i8 quants into one `uint` and decode them with
  `bitfieldExtract`. The same driver caps workgroup shared memory at 32 KiB, which
  sizes the shared-memory rows of the attention kernels.

## 5. Bandwidth budget

A Q8_0 Llama-8B decode step streams about 8.5 GB of weights. At the GDDR6 peak of
512 GB/s the ceiling is about 60 tokens per second. A sustained 60 to 70 percent of
peak gives 35 to 40 tokens per second. The CPU `simd_pf` variant with four threads
reaches about 5 tokens per second. The target is therefore a speedup of 7x to 8x
before any kernel-level tuning, and the fused Llama-8B decode measures about 40
tokens per second.

This is the performance budget of the GPU path. Optimize against the slowest
resource, which here is memory bandwidth. Halving the weight bytes (Q4_0) is the
one lever that moves the ceiling.

## 6. Discipline on this substrate

The general schemas hold. Only the substrate changes.

- **Control unit and datapath.** Vulkan command buffers and queues are the control
  unit. Compute shaders are the datapath.
- **Boundaries carry values.** Buffers, descriptor sets, and push constants are plain
  values.
- **A limit on everything.** The workgroup count and the LDS size are the loop
  bounds. The host asserts every shape before it records a dispatch.
- **Bit-exact where the order is preserved.** A kernel that keeps the CPU reduction order marks its accumulators `precise`. Then `glslc -O` cannot fuse a multiply and an add and change the token hash. A kernel that reorders a reduction declares a
  measured relative tolerance on its Zig wrapper, never a guessed one.

## 7. On portability

Vendor lock-in is a dependency, so the general advice is to prefer a portable layer.
Vulkan compute is the portable layer for this project. SYCL, HIP, Kokkos, RAJA, and
OpenMP target each need a C or C++ toolchain, and the zero-dependency rule rejects
that toolchain. Do not read the general advice as a reason to migrate off Vulkan.

## 8. References

- *Programming Massively Parallel Processors*, 4th edition, Kirk, Hwu, and Hajj. The
  textbook.
- **AMD RDNA2 Shader Instruction Set Architecture** (AMD, 2020). Ground truth for
  what the compiler can emit. Read the chapters on memory operations and scheduling.
- **AMD RDNA2 Performance Guide.** About 30 pages on the hardware-level knobs: wave
  occupancy, LDS bank conflicts, and branch coherence.
- **Mesa RADV**, `mesa/src/amd/`. Production Vulkan on AMD, with the kernels that
  real engines hit. The RADV compiler-tuning posts from the Mesa team are short and
  useful.
- **LLVM AMDGPU backend**, `llvm/lib/Target/AMDGPU/`. Codegen intuition for what
  `glslc -> SPIR-V -> LLVM-AMDGPU` produces.
- **zinc**, <https://github.com/zolotukhin/zinc>. A Zig, Vulkan, and GLSL inference
  engine, and the closest prior art for this path. Read its kernels before you start
  a new shader.
- **The matmul tutorial in the Vulkan-Samples repository.** A clean compute-only
  Vulkan setup, and a good first reference.
- **llama.cpp.** The f32 numerics oracle for this project.
