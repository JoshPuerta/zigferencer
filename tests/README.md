# Tests

This directory holds the cross-module integration tests. A host without the model or without a
GPU skips the test, and the suite stays green.

## Files

| File | Holds |
| --- | --- |
| `snapshot_smollm2_test.zig` | The first greedy token ids of a fixed prompt on SmolLM2, Q8_0 and Q4_0. |
| `tokenizer_llama3_test.zig` | The `llama-bpe` ids against `llama-tokenize`. |
| `tokenizer_qwen_test.zig` | The `qwen35` ids and the added tokens against `llama-tokenize`. |
| `gguf_smollm2_test.zig` | The `ModelCard` that `mmap`, the parser, and the loader build from SmolLM2. |
| `tokenizer_smollm2_test.zig` | The encode and decode round trip on the SmolLM2 vocab. |
| `chat_prompt_test.zig` | Request text that holds control strings, on the vocab of each chat family. |
| `llama_prefix_runner_test.zig` | A Llama runner over a layer span against the whole-model runner. |
| `qwen_prefix_runner_test.zig` | The same for the Qwen runner, plus the blob of a prefix. |
| `llama_hybrid_residency_test.zig` | The swap blob of the Llama hybrid across two tiers. |
| `llama_session_equivalence_test.zig` | Batched decode against solo decode on the CPU Llama runner. |
| `qwen_session_equivalence_test.zig` | The same on the CPU Qwen runner, with the GDN state. |
| `cpu_prefill_chunk_test.zig` | Chunked prefill against token by token, on both CPU runners. |
| `llama_vulkan_serve_equivalence_test.zig` | Batched fused Llama streams against solo CPU streams. |
| `qwen_vulkan_serve_equivalence_test.zig` | The same for the fused Qwen runner and its slot copy. |
| `llama_hybrid_serve_test.zig` | The serve stack over a mid split of the Llama hybrid. |
| `serve_integration_test.zig` | The binary wire against a live server: every client method. |
| `serve_resume_test.zig` | Session resumption over real sockets. |
| `client_compaction_test.zig` | A conversation past a small window, compacted and continued. |
| `zmq_interop_test.zig` | The serve wire against the system libzmq. |
| `openai_integration_test.zig` | The HTTP frontend over real sockets. |
| `serve_harness.zig` | The shared boot of the serve tests. Not a test. |
| `test_policy.zig` | The thread policy of the runner tests. Not a test. |
| `tidy_test.zig` | The source and script rules of the whole repository. |
| `vulkan_vec_add_test.zig` | `vec_add` on the GPU against the CPU. |
| `vulkan_linear_q8_0_test.zig` | `linearF32Q80` on the GPU against `naive`. |
| `vulkan_kv_commit_f16_test.zig` | The f16 rounding of the KV commit. |
| `vulkan_attention_test.zig` | GPU attention against the CPU reference. |
| `vulkan_prefill_kernels_test.zig` | The batched-prefill kernels at the Qwen shapes. |
| `vulkan_stream_ring_test.zig` | The fence and slot reuse of the upload ring. |

## How the pieces fit

The files group by the invariant each one pins.

| Group | Files | Exercises | Needs |
| --- | --- | --- | --- |
| Load | `gguf_smollm2`, `tokenizer_smollm2` | `mmap`, the parser, the loader, `Vocab`, `bpe` | SmolLM2 |
| Golden ids | `tokenizer_llama3`, `tokenizer_qwen` | `bpe` against llama.cpp | Llama-3.1-8B, Qwen3.5-2B |
| Snapshot | `snapshot_smollm2` | The whole stack, as a short list of token ids | SmolLM2 |
| Runner boundary | `llama_prefix_runner`, `qwen_prefix_runner` | The runner over a layer span | the model of the runner |
| Batched is solo, CPU | `llama_session_equivalence`, `qwen_session_equivalence`, `cpu_prefill_chunk` | `Sessions` over a CPU runner | the model of the runner |
| Batched is solo, GPU | `llama_vulkan_serve_equivalence`, `qwen_vulkan_serve_equivalence`, `llama_hybrid_serve`, `llama_hybrid_residency` | The fused and hybrid runners | a GPU, the model |
| Serve | `serve_integration`, `serve_resume`, `client_compaction`, `zmq_interop`, `openai_integration` | `Server`, `Client`, and the HTTP adapter | SmolLM2 or Qwen3.5-2B |
| Kernel gates | the six `vulkan_*` files | One `ops_vulkan` kernel against its CPU op | a GPU, no GGUF |
| Repo rules | `tidy_test` | The source tree, the scripts, and the `Makefile` | none |

Three build arrays register the files, and the fixture is the shared dependency. The CPU files
reach the model through one load chain. The GPU files add a device probe in front. The kernel
gates skip the model and build synthetic inputs.

```
  build.zig: cpu_, vulkan_, fused_integration_tests[]   one addTest per file
                              |
                              v
                     zigferencer module
                              ^
  core/test_fixtures.path --> loader.load --> Loaded         miss: skip, or fail when the
                              |                              model store is set
      load, tokenizer:        +--> ModelCard, Vocab --> bpe
      snapshot, runners:      +--> family.Of(arch).Cpu | Sessions --> sampler
      serve:                  +--> Server(Sessions(Cpu)) <-- Client | libzmq | http client
      GPU runners:            device probe --> fused or hybrid runner against the CPU runner
      kernel gates:           device probe --> one kernel against its CPU op, with no GGUF
```

## Rules

- **A missing model skips the test.** Every model test takes its file from
  `test_fixtures.path` with the registry name of the model. That function finds the file
  under `models/` first, then under `$ZIGFERENCER_MODELS_DIR`. When neither holds the file
  and the variable is unset, the test returns `error.SkipZigTest`. When the variable is set, the
  test fails with `error.FixtureMissing` and names the two paths it searched.
  A machine that says where its models are must run every model test.
  A GPU test also skips when the device probe returns `NoVulkanDevice`,
  `InitializationFailed`, `IncompatibleDriver`, or `FeatureNotPresent`, and a
  fused or hybrid test also skips on `error.VramBudgetExceeded`.
- **Run the Qwen tests in ReleaseSafe.** The Qwen3.5-2B fixture is about 2 GB.
  When the file is present, a Debug `zig build test` runs the Qwen runner tests
  for minutes. The gate runs the suite with `-Doptimize=ReleaseSafe`.
- **Tests run at fixed windows.** A test that sizes a KV window from a real model
  takes `test_fakes.test_window`, 8192 positions. A test that plans several slots
  takes `test_fakes.test_serve_window`, 512 positions. A test never takes the
  default rule, which reads the model cost and the free VRAM. A test therefore
  runs the same shape on any machine.
- **Each gate declares its tolerance.** Some gates are bit-exact. The code keeps the
  reduction order there: `snapshot`, `llama_session_equivalence`,
  `qwen_session_equivalence`, both prefix boundaries, `vulkan_linear_q8_0`,
  `vec_add`, and `kv_commit_f16`. Some gates are greedy-tokens-exact across backends.
  The per-op numerics there live under the declared tolerance of each kernel:
  `llama_vulkan_serve`, `qwen_vulkan_serve`, and `llama_hybrid_serve`. `vulkan_attention` is
  the one gate that states a relative tolerance of its own, `2e-3`, the same
  value the bench uses.
- **Bytes are compared within one backend, tokens across two.** The equivalence
  files hash the raw logits rows with FNV-1a. They compare the hashes where both sides
  ran on the same backend. Across backends the KV bits differ, because a GPU
  reduce kernel sums in a different order. Those gates compare greedy tokens
  instead.
- **The values that cross in carry no hidden state.** A `ModelCard` and a `[]TensorRef` come from
  the loader. A `Vocab` goes into `bpe`, and token ids come back out. The serve tests exchange
  framed wire messages between `Client` and `Server`.
- **Registration is a build-graph rule.** A new `*_test.zig` file under `tests/` must join
  the `cpu_integration_tests`, `vulkan_integration_tests`, or
  `fused_integration_tests` array in `build.zig`. The build fails with a message that names a
  file in no array. Each entry becomes its own
  `addTest` target that imports the `zigferencer` module. The Vulkan entries
  compile only under `-Druntime=vulkan`, because they reference the Vulkan
  modules and the embedded SPIR-V. The fused entries name a fused or a hybrid
  runner. `src/main.zig` re-exports those runners under `-Dops-backend=vulkan_fused`
  alone, so the build appends that list for that backend alone. A helper such
  as `serve_harness.zig` joins no array, because the test file that imports it
  compiles it into its own binary. The `zmq_interop_test.zig` module is the one
  test module that links libc, and `build.zig` says why.
- **Traces inside a test.** The test runner owns `std_options`, so
  `$ZIGFERENCER_LOG` applies to the binaries alone. To see engine traces while a
  test runs, set `std.testing.log_level = .debug;` in the test body.
- **Investigate before you regenerate.** On a snapshot or golden divergence, walk
  the stack first. The expected values are the record of a verified run.
- **Step through one test in a debugger.** `make debug-test TEST=<file stem>
  FILTER="<test name>"` installs the test binary at `zig-out/test/<file stem>`.
  [../docs/debugging.md](../docs/debugging.md) has the commands.
