# The target GPU: AMD RDNA2

The GPU kernels of zigferencer target the AMD Radeon RX 6800 XT
(Navi 21, `gfx1030`, RDNA2), through Vulkan compute. This page introduces the parts
of that GPU that decide how fast a kernel runs. The header of each shader in `src/ops_vulkan/shaders/` and
of each file in `src/ops_vulkan/kernels/` holds the details. Another RDNA2 or RDNA3 card
changes the numbers, but ideally not the method.

Take this as condensed material from investigations while writing the kernels.

## 1. Words: CUDA, Vulkan, and AMD

| CUDA | Vulkan (GLSL) | AMD hardware |
| --- | --- | --- |
| thread | invocation | lane |
| warp (32 threads) | subgroup | wave: 32 lanes (wave32) or 64 lanes (wave64) |
| block | workgroup | runs on one compute unit (CU) or one workgroup processor (WGP) |
| grid | dispatch | |
| shared memory | `shared` variables | LDS (local data share) |
| SM (streaming multiprocessor) | | CU. RDNA pairs two CUs into one WGP |
| constant memory | push constants, uniform buffers | |

## 2. The reference setup: WSL2 and Vulkan on D3D12

The project set out to use one desktop card, an RX 6800 XT, from Linux under WSL2
on Windows. On that setup the only Vulkan driver is Mesa Dozen. Dozen translates
Vulkan to Direct3D 12, and the Windows AMD driver then runs the work. RADV needs a
DRM device that WSL2 does not expose for AMD cards, and ROCm does not support RDNA2
under WSL2.

The tested software is Ubuntu 24.04 with Mesa 26.1 from the kisak PPA, which builds
Dozen. `glslc` and the Vulkan loader come from the LunarG Vulkan SDK 1.4.313. The stock
Mesa of Ubuntu 24.04 has no Dozen. The stock `glslc` 2023.8 and `libvulkan-dev`
1.3.275 also build the GPU path, but their SPIR-V differs byte for byte from the SDK
output.

This driver path sets most of the limits that the kernels and the planners respect:

| Limit of the setup | How the engine handles it |
| --- | --- |
| Vulkan 1.2, no 8-bit storage, no 8-bit integers in shaders | The engine splits Q8_0 into an f16 scale array and packed 8-bit words (section 9). |
| 32 KiB of shared memory for each workgroup | The attention kernels fit their LDS into 32 KiB, and a compile-time assert checks it (section 10). |
| A storage-buffer range of 128 MiB, the Vulkan minimum | The KV cache uses one region for each layer, a default window shrinks to fit, and `$ZIGFERENCER_VK_STORAGE_RANGE` states a checked range ([architecture.md](architecture.md), section 9). |
| An allocation overhead that runs out of memory before the 16 GB card is full | On this driver the VRAM preflight keeps 1/4 of the free VRAM in reserve, against 1/8 on other drivers (`limits.vram_shares`). The 1/4 is a heuristic from this setup. The preflight refuses a plan that cannot fit. A plan under the line can still fail on a driver whose budget does not show its host-visible staging or its residency. |
| A device loss under memory pressure, which stays until the virtual machine restarts | Run one large GPU job at a time, with time between runs. `pkill -x 'zigferencer\|zigferencer-dev'` ends a process that stays. |
| Windows stops a GPU submit that runs longer than its timeout (2 s by default) | A large-model prefill needs a longer `TdrDelay` in the Windows registry. |
| A GPU timestamp counter that is not a stable clock across submits | The trace maps each GPU record by its own host window ([profiling.md](profiling.md)). |
| Host-visible memory at about 30 GB/s over the bus of the virtual machine | The weights, the KV, and the scratch live in device-local memory. |
| No view of the shader ISA, the registers, or the occupancy, and no hardware counters | Tuning uses measured time only. |

### Device loss during the weight upload

Two failures set the line and the messages of this driver.

- A Llama-8B plan of 13524 MiB (9086 MiB of weights and 4438 MiB of KV) passed the 7/8
  line and lost the device during the weight upload. A plan of about 11.1 GB then ran in
  the same virtual machine. The usual budget of the card is 15568 MiB, and 3/4 of it is
  11676 MiB, between the two plans. So the Vulkan-on-D3D12 entry of `limits.vram_shares` is
  3/4. This entry is a heuristic from this one setup. Another setup, for example a native
  Linux driver, uses the default entry of 7/8 and can plan more of the card.
- A Llama-8B Q8_0 fused run at a plan of 9737 MiB loses the device during the weight
  upload in some runs. The same command passes in other runs of the same build, and the
  cause is not proven. This plan is under both lines, so the 3/4 entry does not explain it.

What changed after these failures:

- The weight upload creates one host buffer pair for all tensors, not a pair for each
  tensor. The pair holds one swap chunk at most, and a larger tensor goes in passes.
- `ZIGFERENCER_LOG=runner` prints the budget and the usage of each memory heap before
  and after the weight upload.
- A device loss or a refused allocation during the init of a fused runner stops with a
  message that asks for enough free device memory and a retry. Under WSL2, restart the
  virtual machine first (`wsl --shutdown`) when the log shows a device removal.

The second failure occurred again after these changes, so they do not remove it.

A native Linux driver such as RADV removes most of these limits. It reports 64 KiB
of shared memory and the real storage range, and the vendor profilers work with it.
The engine reads each limit from the driver at start, except the 32 KiB of shared
memory. That bound is a constant in `core/limits.zig`, and every AMD driver allows
at least 32 KiB.

## 3. The card in numbers

| Part | RX 6800 XT |
| --- | --- |
| Compute units | 72 CUs in 36 WGPs. Each CU has two SIMD32 units. |
| Layout | 4 shader engines with 2 shader arrays each. Each array has its own graphics L1. |
| Clock | Boost clock up to 2250 MHz. |
| Arithmetic peak | 20.7 TFLOPS at f32 (4608 lanes x 2 operations x 2.25 GHz), 41.5 TFLOPS at packed f16. |
| VRAM | 16 GB of GDDR6 on a 256-bit bus at 16 Gbps: 512 GB/s. |
| Infinity Cache | 128 MB between the L2 and the VRAM, about 2 TB/s at peak. |
| L2 cache | 4 MB. |
| Graphics L1 cache | 128 KB for each shader array. |
| In each WGP | 128 KB of LDS, a 32 KB instruction cache, and a 16 KB scalar cache. |
| In each CU | A 16 KB L0 vector cache. |
| In each SIMD32 | A register file of 128 KB: 1024 vector registers (VGPRs) of 32 lanes. One wave can address up to 256 VGPRs. Up to 16 waves are resident. |
| Integer dot products | 8-bit and 4-bit dot-product instructions, for example `V_DOT4_I32_I8`. |

## 4. Decode is a bandwidth problem

The weights go to VRAM once, when the runner starts, and they stay there. The cost
in this section is not that upload. A decode step reads every weight matrix once,
from VRAM into the registers of the kernels. It uses each weight for one multiply and
one add. The matrices are far larger than the caches, so the memory bus moves
those bytes again for each token.

A Q8_0 block stores 32 weights in 34 bytes, which is 1.0625 bytes for each weight.
The bytes that one step reads, over the memory bandwidth, give the shortest possible
step. The byte counts come from the tensors of
the Q8_0 files:

| Model | Tensor bytes in the file | Bytes one decode step reads | Shortest step at 512 GB/s | Most tokens per second |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B | 2.00 GB | 2.00 GB | 3.9 ms | 256 |
| Llama-3.1-8B | 8.53 GB | 7.97 GB | 15.6 ms | 64 |

A step reads only the embedding of one token from the embedding table. Llama-3.1-8B has a separate
lm_head, so a step skips the rest of its 0.56 GB table. Qwen3.5-2B ties its lm_head
to the embedding table, so its lm_head reads the whole table on each step. These
limits leave out the KV reads, which grow with the context.

The roofline shows why. The card does 20.7 TFLOPS against 512 GB/s, so a kernel
needs about 40 operations for each byte to use all its arithmetic. A decode
matrix-vector product does 2 operations for each 1.0625 bytes, about 1.9 operations
for each byte, more than 20 times too few. Its speed therefore depends only on how
fast it reads memory. The Infinity Cache does not change this. It holds 1.7 % of
the bytes of an 8B step and 6.7 % of the bytes of a 2B step. A stream that reads
each byte once for each token gets almost no reuse from a cache.

How the kernels use this:

- The matrix-vector kernels read the weights in coalesced loads and do little
  else. On a Qwen3.5-9B decode the matrix time measured 18.7 ms against a floor of
  18.6 ms (9.5 GB at 512 GB/s).
- Fewer bytes for each weight raise the limit. A Q4_0 block stores 32 weights in 18
  bytes, about half of Q8_0.
- One submission for each step removes the host gaps between kernels (section 11).
- The prefill kernels feed many rows through one read of the weights, so they move
  toward the arithmetic limit (section 8).

## 5. Waves: wave32 and the pin

An RDNA SIMD unit runs a wave of 32 lanes natively. A wave64 instruction issues
twice, once for each half of the wave. A driver can still compile a compute shader
for wave64, and RADV, the Mesa driver, does so by default. The engine therefore pins every compute pipeline to a subgroup
size of 32 through `VK_EXT_subgroup_size_control` (`kernels/pipeline.zig`,
`runtime/vulkan_ctx.zig`). On a SmolLM2 decode the pin cut the inter-token latency by
31 % and its standard deviation by a factor of 4.

Every workgroup size is a multiple of 32, and each size has a reason:

| Workgroup | Waves | Kernels | Why |
| --- | --- | --- | --- |
| 32 lanes | 1 | the decode matrix-vector kernels, the prefill batch kernels, the Gated-DeltaNet scan | All lanes are in one wave, so a barrier costs nothing and no wave waits for another. |
| 64 lanes | 2 | attention, RoPE, the KV commit | Two resident waves hide the memory latency of the KV reads. |
| 128 lanes | 4 | the Gated-DeltaNet decode step, convolution, and norm | One lane for each of the 128 dimensions of a head. |
| 256 lanes | 8 | elementwise ops, norms, the prefill GEMM | For the elementwise ops, eight waves keep a WGP busy without so many registers that fewer waves fit. The GEMM is a 16 x 16 grid of lanes, one 4 x 4 output block for each lane. |

## 6. Occupancy: registers and LDS

A CU keeps several waves resident and switches to another wave while one waits for
memory. This switch hides the memory latency. Two budgets limit how many waves fit:
the vector registers (VGPRs) of each wave, and the LDS of each workgroup. A kernel
that uses more of either keeps fewer waves resident.

A SIMD32 has 1024 VGPRs, allocated in steps of 16, and holds at most 16 waves:

| VGPRs for each wave (wave32) | 64 or fewer | 80 | 96 | 128 | 160 | 192 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Waves on one SIMD32 | 16 | 12 | 10 | 8 | 6 | 5 | 4 |

The LDS decides next. A WGP has 128 KB of LDS. In CU mode each CU uses one half of
64 KB, and one workgroup can take at most 64 KB.

How the kernels use this:

- The subgroup matrix-vector kernel stages its input vector in 16 KB of LDS, so four
  workgroups still fit on one CU.
- The prefill kernel `linear_q8_0_batch` works on a tile of 16 rows. A larger tile
  needs more registers for each lane and fits fewer waves, and the achieved
  bandwidth falls. A sweep on a 9B prefill measured 16 as the knee
  (`limits.zig`).
- Split attention runs up to 4 query heads in one workgroup. A sweep of 8, 4, and 2
  picked 4: at 8, the LDS of one workgroup halves the workgroups that fit
  (`limits.attn_split_queries`).
- A per-lane array with an index that changes at run time goes to scratch memory,
  which is slow. The Gated-DeltaNet scan therefore fixes its head size at 128 and
  unrolls the loop, so every index is a constant.

## 7. Coalescing: adjacent lanes, adjacent bytes

A wave sends one memory request for each cache line that its lanes touch. When the
32 lanes read 32 adjacent words, one request serves the whole wave. When each lane
reads a different line, the wave sends 32 requests for the same work.

How the kernels use this:

- The elementwise kernels give lane `i` element `i`.
- The decode matrix-vector kernel `linear_q8_0_reduce` gives one workgroup to each
  output. Lane `L` reads the quant blocks `L`, `L + 32`, `L + 64`, and so on, so
  the lanes of the wave read adjacent blocks. Lane 0 then adds the 32 partial sums
  in LDS. Vulkan guarantees only 65535 workgroups on each grid axis, and an lm_head
  of a registered model has up to 248320 outputs. The loader takes up to
  `limits.max_vocab` (262144). So the outputs past `limits.reduce_grid_x` (32768) fold
  onto the y axis of the grid.
- Split attention loads K in 16-byte vectors, eight loads for each 128-byte line.
- One counter-example stays on purpose. The resident decode attention gives each
  lane its own K row, so the K reads do not coalesce across the wave. Its V pass
  gives each lane one dimension, and those reads do coalesce.

## 8. Reuse and tiling

A tile stages data in LDS, so that many lanes read it from LDS and not from VRAM.
A tile helps only when the same bytes feed more than one output.

- **Decode.** A matrix-vector product uses each weight byte for one row only. There
  is no reuse, so the decode kernels do not tile. They read the weights once.
- **Prefill.** A chunk of rows multiplies the same weights. `linear_q8_0_batch`
  holds an input tile of 16 rows in LDS. Each lane keeps the quant block of one
  weight column in its registers and applies it to all 16 rows. Each weight byte
  therefore serves 16 rows. `linear_q8_0_gemm` computes a 64 x 64 output tile. Both
  inputs sit in 16 KB of LDS, and each lane keeps a 4 x 4 block of outputs in its
  registers.
- The Qwen3.5 fused runner selects the kernel by the row count of a step. One row
  takes the matrix-vector kernel, fewer than 64 rows take the batch kernel, and 64
  rows or more take the GEMM. On a 9B prefill the GEMM cut the matrix time from 564 ms to 379 ms.

## 9. Quantized weights without 8-bit storage

A Q8_0 block holds 32 signed 8-bit values and one f16 scale, 34 bytes in all. The
driver class of the reference setup is Vulkan on D3D12. It exposes 16-bit storage
and f16 arithmetic, but no 8-bit storage and no 8-bit integer type in shaders. At
load the engine therefore splits the blocks of each weight into two arrays. One
array holds the f16 scales, and one holds the 8-bit values (`kernels/weights.zig`). A shader reads the 8-bit array
as 32-bit words and takes each value out with `bitfieldExtract(int(word), (i & 3) *
8, 8)`. A Q4_0 shader takes its two 4-bit values out of each byte with a shift and a
mask.

The split also helps the reads: each array is contiguous, so a wave reads the scales
and the values with coalesced loads.

RDNA2 also has 8-bit dot-product instructions. A native driver exposes them to
SPIR-V through `VK_KHR_shader_integer_dot_product`. The kernels do not use them yet:
they convert each value to f32 and multiply in f32.

## 10. Driver limits

The kernels respect these limits of the driver:

- **Shared memory of one workgroup.** Vulkan on D3D12 and the AMD drivers allow
  32 KiB. RADV allows 64 KiB. The kernels assume 32 KiB, and a compile-time assert
  checks each layout against it (`core/limits.zig`).
- **Storage-buffer range.** Some drivers report the minimum range of the Vulkan
  specification, 128 MiB. The KV cache uses one region for each layer and
  tensor, so the limit applies to one region and not to the whole cache.
- **Two kinds of memory.** A shader reads device-local memory at about 500 GB/s. It
  reads host-visible memory over the bus to the host at about 30 GB/s. The
  weights, the KV, and the scratch therefore live in device-local memory, and uploads
  pass through a host-visible staging buffer.

## 11. One command buffer for each token

A decode step has hundreds of ops. On Llama-3.1-8B it has about 450, and one host
round trip costs about 0.27 ms, so a submit per op takes about 120 ms a token on
synchronization alone. The per-op backend submits each op and waits for it. The
fused runners record the whole step instead:

```
  per-op:  [record op 1][submit][wait] [record op 2][submit][wait] ... hundreds of times
  fused:   [record every dispatch, a barrier between dependent ones][submit][wait]
```

- The weights, the KV, and the scratch stay in VRAM, and each descriptor set is
  bound once, when the runner starts.
- A barrier sits between two dispatches only when the second reads what the first
  wrote.
- The decode ring (`runtime/decode_ring.zig`) holds two command buffers. While the
  GPU runs step N, the host records the predicted step N + 1 into the other buffer.
  When the prediction misses, the next step records its own buffer before it submits.

## 12. Why Vulkan?

Vulkan compute runs on AMD, NVIDIA, and Intel GPUs, and on other platforms through a
translation layer, with no vendor toolkit. SYCL, HIP, Kokkos, RAJA, and OpenMP target
each need a C or C++ toolchain. The CPU build has no dependency outside the Zig
standard library, and the GPU build adds only the Vulkan bindings.

## 13. References

- Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. *Programming Massively Parallel
  Processors: A Hands-on Approach*, 4th edition. Elsevier, 2022. The textbook for
  the hierarchy, tiling, coalescing, and reductions. Read it for the model, not for
  the CUDA syntax.
- AMD. *"RDNA 2" Instruction Set Architecture: Reference Guide*. 2020.
  <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>. What
  the hardware runs: waves, registers, LDS, and the dot-product instructions.
- AMD. *RDNA Architecture* (white paper). 2019. <https://gpuopen.com/rdna/>. The
  WGP, the cache hierarchy, and wave32.
- Andrew Pomianowski. *AMD RDNA 2 Gaming Architecture*. Hot Chips 33, 2021.
  <https://hc33.hotchips.org/assets/program/conference/day2/2021%20Hot%20Chips%20RDNA2%20Pomianowski%20Final%2020210820.pdf>.
  The layout of Navi 21, the cache sizes and bandwidths, and the Infinity Cache.
- AMD GPUOpen. *RDNA Performance Guide*. <https://gpuopen.com/learn/rdna-performance-guide/>.
- AMD GPUOpen. *Occupancy explained*. <https://gpuopen.com/learn/occupancy-explained/>.
  How registers and LDS limit the resident waves.
- LLVM. *User Guide for AMDGPU Backend*. <https://llvm.org/docs/AMDGPUUsage.html>,
  with the source in `llvm/lib/Target/AMDGPU`. The occupancy formulas and the
  target features of `gfx1030`.
- Mesa: RADV in `src/amd/vulkan` and its compiler ACO in `src/amd/compiler`.
  <https://gitlab.freedesktop.org/mesa/mesa>. Production Vulkan on AMD, and the
  source of each limit that RADV reports.
- Khronos. The reference pages of `VK_EXT_subgroup_size_control` and
  `VK_KHR_shader_integer_dot_product`. <https://docs.vulkan.org/refpages/latest/>.
- vulkan.gpuinfo.org. The features and limits that each driver reports for a card.
- llama.cpp, the Vulkan backend in `ggml/src/ggml-vulkan`.
  <https://github.com/ggml-org/llama.cpp>. Prior art for the kernels, and the f32
  reference for the numbers.
- zinc. <https://github.com/zolotukhin/zinc>. A Zig inference engine on Vulkan and
  Metal. Its README lists RDNA3 and RDNA4 for AMD.
- Khronos Vulkan-Samples, `samples/api/compute_nbody`.
  <https://github.com/KhronosGroup/Vulkan-Samples>. A small compute sample with
  shared memory.
