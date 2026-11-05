# tests

Cross-module integration tests. Each file drives the engine end to end through
the `zigferencer` module that `../src/main.zig` re-exports, against real
collaborators and a real GGUF on disk. This directory is the outer edge of the
test pyramid, above the `test {}` blocks that live beside each `src/` module. It
is separate because a file here owns a fixture and a device probe rather than a
unit. A missing fixture fails a test only on a host that sets
`$ZIGFERENCER_MODELS_DIR`. Elsewhere the test returns `error.SkipZigTest`, and
the suite stays green on a host without the model or without a GPU. The
whole-repository map is [../docs/architecture.md](../docs/architecture.md), which
lists this directory in one line.

## Files

Twenty-eight files: twenty-six test files, grouped by the invariant each one pins, and
two shared helpers that are not tests. The order inside a group runs from the
simplest to the widest.

### Snapshot, the highest-leverage gate

| File | Invariant it pins |
| --- | --- |
| `snapshot_smollm2_test.zig` | The first N greedy token ids for a fixed prompt against the real SmolLM2-135M GGUFs, Q8_0 and the mixed-quant Q4_0. A regression in the parser, the tokenizer, any op, the KV cache, the scheduler, or the sampler surfaces here. The Q4_0 case also drives the q4_0 and q4_1 dequant kernels. |

### Golden ids against the llama.cpp reference

| File | Invariant it pins |
| --- | --- |
| `tokenizer_llama3_test.zig` | The `llama-bpe` pre-tokenizer matches `llama-tokenize --ids` on the real Llama-3.1-8B vocab, digit-run grouping included (`2147` splits as `214`, `7`). |
| `tokenizer_qwen_test.zig` | The `qwen35` pre-tokenizer matches the same reference on the real Qwen3.5-2B vocab: single-digit runs, no `ignore_merges`, and whitespace-run grouping. The added tokens `<think>`, `</think>`, and `<tool_call>` each encode to one id. The text on each side of a tag encodes on its own. |

### Round trip and consistency

| File | Invariant it pins |
| --- | --- |
| `gguf_smollm2_test.zig` | `mmap -> parser -> loader` over about 145 MB of real data produces the expected `ModelCard`: dims, head counts, BOS and EOS, and the tensor-table size. It catches schema drift that a synthetic fixture cannot. |
| `tokenizer_smollm2_test.zig` | `decode(encode(s)) == s` for short ASCII inputs, and `<|im_start|>` round-trips as one id. These are oracle-free invariants over the real vocab. |

### Runner boundaries, a layer span against the whole model

| File | Invariant it pins |
| --- | --- |
| `llama_prefix_runner_test.zig` | A Llama runner built with `initPrefix` runs the same layers, over the same weights and the same KV, as the runner `init` builds. Both are driven the way a composing runner drives them, and the residual streams must match byte for byte. |
| `qwen_prefix_runner_test.zig` | The same for `QwenRunner.initPrefix` at an arbitrary layer count `k`, plus the slab shapes and the blob claim the hybrid needs: a prefix blob equals the leading span of the whole model's blob. |
| `llama_hybrid_residency_test.zig` | The hybrid swap unit. One blob spans two tiers in whole-model order, its size does not move with the split, and the host half is byte-equal to the CPU runner's. The provenance tag keeps one placement's snapshots out of another placement's namespace. One case builds at split 0 and proves that the hybrid's own GPU layer bodies decode the stream of the fused runner. |

### Batched is solo, the serving invariant

| File | Invariant it pins |
| --- | --- |
| `llama_session_equivalence_test.zig` | On the CPU `LlamaRunner`, batched decode is bit-identical in tokens and in logits to each session run alone. The cases cover a whole-prompt feed, chunked prefill, continuous-batching ticks, a mid-conversation clone, and forced parallelism of the matmuls and the attention parts. |
| `qwen_session_equivalence_test.zig` | On the CPU `QwenRunner`, time-sliced multi-session decode is bit-identical to each session alone. It pins the GDN and KV state routing, slot reuse through `resetSlotState`, and continuous ticks. It also pins a clone that copies both slabs and forced parallelism of the matmuls and the attention parts. |
| `cpu_prefill_chunk_test.zig` | On both CPU runners, a prompt fed through `prefillBatch` in chunks of 1, 17, and 64 rows gives the same last-row logits, bit for bit, as the same prompt fed one token at a time through `step`. The greedy continuation is the same too. Each chunked path runs at one thread and at four threads with the fan-out forced on. Both runners refuse rows past the row scratch. |
| `llama_vulkan_serve_equivalence_test.zig` | On the fused `LlamaFusedRunner`, greedy token streams from batched multi-stream decode equal the solo CPU `LlamaRunner` streams. The comparison is cross-backend: tokens exact, per-op numerics under each kernel's declared tolerance. |
| `qwen_vulkan_serve_equivalence_test.zig` | The same for `QwenFusedRunner` and its multi-slot `stepBatch`. A clone must also continue the stream of its source, which pins the device-side slot copy. |
| `llama_hybrid_serve_test.zig` | The invariant under the real serve stack at a true mid split, over `Residency(Sessions(LlamaHybridRunner))`. One case runs at full residency, and one over-subscribes the pool so at least one turn decodes over a restored blob. It runs once per KV element type. |
| `test_policy.zig` | Not a test. It is the thread policy of the runner tests, at `-Dtest-threads` with the fan-out floor at zero, so the suite runs the pool path at short rows. It also starts the pool that each test owns. |
| `serve_harness.zig` | Not a test. It is the shared boot for the serve tests. It loads the fixture and starts an in-process server on TCP or on a unix socket. It stops the server over the wire or through a request of the process owner. `Harness` runs the SmolLM2 `LlamaRunner`, and `QwenHarness` the CPU `QwenRunner` over Qwen3.5-2B. Options set the window, the chat template, and the prefix cache. Only a missing fixture makes it skip, so a bind failure fails the test. |
| `serve_integration_test.zig` | The same invariant observed from the binary wire, on one connection and across four connections at once. It drives every client method against a live server, with a cancel in prefill and a cancel in decode. It also covers the unix transport against TCP, the connection cap, and the per-connection session share. It checks the sweep after a hang-up and a wire shutdown that the server refuses. The first test covers `context_full` surfacing as `done`, a mid-generate close that leaves the slot reusable, and a clean shutdown handshake. Two tests read a `turn_too_long` refusal, which names the turn size and the positions left. One refuses a prompt past a fresh window, and the same session then answers a fitting turn. The other caps a turn at the whole window, so the turn ends `context_full` at the window. The next turn then gets the refusal with zero positions left. Two tests run a conversation live and then send its turns to a fresh session with `session_append`. One uses SmolLM2, and one uses the 2B with thinking off. Each `appended` position equals the live one, and the next turn streams the same ids. |
| `serve_resume_test.zig` | Session resumption over a real model and real sockets. A session that a client keeps, drops, and resumes on a new connection continues with the ids and the position of an undropped twin. This holds for greedy and for seeded sampling. An expired lease answers `unknown_session`, and with resumption off a drop frees the session. |
| `client_compaction_test.zig` | A conversation past a window of 64 positions on SmolLM2 continues after compaction. It runs through both triggers: a turn that ends `context_full` and a turn the server refuses. Each cycle re-opens with the original system prompt, which the prefix cache serves, and appends the kept pairs. The position after the appends stays at or under the estimate, and each resent turn is admitted. A second test plants a cost that is too low, so the append refuses and the cycle refits with the learned cost. |
| `zmq_interop_test.zig` | The interop oracle for the serve wire. It opens the system libzmq at run time through `std.DynLib`. It drives a real DEALER against the in-process server over `tcp` and `ipc`, and also two DEALERs at once. Every stream must equal the stream of the reference client. A DEALER also appends kept pairs and reads the same position list as the reference client, and a version 1 frame gets `unsupported_version`. It skips where the library is absent. Its module links libc, because the pure-Zig ELF loader does not resolve the dependencies of that library. |
| `openai_integration_test.zig` | The HTTP connector end to end over real sockets: a 200 carrying a `chat.completion` object with non-empty content and a finish reason. A body that names another model must get a 404 of the `model_not_found` type, and the served id must pass the same check. It catches a regression in the framing, the JSON, or the render, tokenize, and decode path. A streamed reply carries the text of the buffered reply. A budget past the room fills the window and ends with `length`, and a request with no budget gets the whole window. A prompt past the window gets a 400 that names both numbers, buffered or streamed, and the streamed request gets no event stream. `enable_thinking: true` on the fixture, which has no thinking mode, gets a 400. `false` and an absent switch give the plain reply byte for byte. |

### Repo-wide rules

| File | Invariant it pins |
| --- | --- |
| `tidy_test.zig` | No test waits on a clock, and non-test code does not print to stderr. Every source file keeps the shape rules: braces, 100 columns, and the section order. The test reads the source tree from the build root, and a short allow list names the files that break a rule on purpose. Every import points down the layers. |

### GPU kernel gates, a kernel against its CPU reference

| File | Invariant it pins |
| --- | --- |
| `vulkan_vec_add_test.zig` | `vec_add` on the GPU is bit-equal to `a[i] + b[i]` on the CPU. `n = 1024` covers the exact-multiple workgroup case and `n = 1000` covers the tail early-return. It is the proof-of-life the other gates build on. |
| `vulkan_linear_q8_0_test.zig` | `linearF32Q80` on the GPU is bit-exact against `naive`. The reduction order, the f16 scale decode, and the i8 byte decode must all match, and no tolerance absorbs any of them. |
| `vulkan_kv_commit_f16_test.zig` | The rounding gate for the f16 KV cache. It compares raw f16 bit patterns, never widened values, so it witnesses that the device and `@floatCast` round to nearest even the same way. It covers ties in both directions, the subnormal range, and the workgroup boundary. One dispatch narrows K and V together, and V carries the fixtures in reverse order, so a kernel that feeds one source into both outputs fails the gate. |
| `vulkan_attention_test.zig` | Attention fidelity against the CPU reference at the same `2e-3` relative tolerance the bench uses. The SmolLM2 shape (`n_heads = 9`, `n_kv_heads = 3`, `head_dim = 64`, `t_len = 128`) stays pinned on the resident kernel, so a real regression cannot hide behind an old report. Eight head geometries run at both KV element types. The decode kernel runs at the resident depths up to `limits.attn_split_len`, and the split and merge passes run at five depths past it. The depth cases put a needle on the last position of each row, so a row that reads one position too few or too many fails. The needle rows of four model families must keep the output digests that a device run minted. Prefill chunks record through `attn_record.recordPrefill`: the resident batch body for rows inside `limits.attn_split_len` positions, and the split passes for rows past it. Three tests pin the bytes of that path. A prefill row must equal the decode row of its depth byte for byte. Every chunk width must give the same bytes, and so must the minimum record budget. A resident batch row must equal the row of the `attention.comp` kernel. The multi-row decode rows read four KV streams of different content from one region. They run through the multi-stream kernel and through the split passes, with the q row base and the stream base set apart. Two tests run a split row and a deep chunk twice, and each pair of outputs must match byte for byte. |
| `vulkan_prefill_kernels_test.zig` | Digest gate for the batched-prefill kernels at the Qwen3.5-2B shapes. The chunked-parallel GDN conv and scan run at the widths 64, 128, and 37, with a non-zero state and conv ring. The tiled Q8_0 GEMM runs at the widths 64 and 128 on the three projection shapes, and the batch kernel runs at width 7. Each output must keep the digest that a device run minted. A null entry fails the test and prints the table to paste. |
| `vulkan_stream_ring_test.zig` | The synchronization of the upload ring, not its arithmetic. Each test uploads a known byte pattern and reads the device segment back. A missing barrier, a mis-sized copy, or an overwritten slot shows up as a byte mismatch. `core/ring_fsm.zig` proves the state table at compile time, and this file proves the fence on hardware. |

## How the pieces fit

The fixture is the shared dependency. The CPU files reach the model through one
load chain. The GPU files add a device probe in front of that chain. The kernel
gates skip the model entirely and build synthetic inputs.

```
  build.zig                              ../src/main.zig
  cpu_integration_tests[]    --import-->   zigferencer module
  vulkan_integration_tests[]
  fused_integration_tests[]    (re-exports loader, bpe, the runners, serve, the
        |                        vulkan modules, and the embedded SPIR-V)
        v  one addTest per path                    v
   test_fixtures.path(name)               loader.load(io, arena, path)
        |  miss -> skip, or fail with store set    |
        +---------------------------+--------------+
                                    v
   model on disk?  no -> skip or fail.  yes -> the chain the file under test needs:

     gguf / tokenizer:        loader -> ModelCard, Vocab -> bpe
     snapshot / llama / qwen: loader -> LlamaRunner | QwenRunner | Sessions -> sampler
     prefix boundaries:       loader -> two runners at different layer counts
     serve_integration:       loader -> Server(Sessions(LlamaRunner)) <-- Client (TCP, unix)
     client_compaction:       loader -> Server(Sessions(LlamaRunner)) <-- Client plus the transcript compaction
     zmq_interop:             loader -> Server(Sessions(LlamaRunner)) <-- libzmq DEALER (tcp, ipc)
     openai_integration:      loader -> the HTTP adapter <-- std.http.Client
     llama_vk / qwen_vk:      device? no -> skip.  yes -> a fused runner against the CPU runner
     llama_hybrid_residency:  device? no -> skip.  yes -> a mid split, plus split 0
     llama_hybrid_serve:      device? no -> skip.  yes -> a true mid split
     vec_add / linear / attn /
       kv_commit_f16 / stream_ring: device? no -> skip. yes -> kernel against CPU op (no GGUF)
```

The equivalence files share one shape. They build a one-slot reference path, then
a multi-slot batched path, then they compare the two. The reference is always the
single-sequence run, so each file is self-checking and stores no external golden.
The snapshot and tokenizer files are the exception: their expected values come
from the llama.cpp reference and live in the file.

## Rules

- **A missing fixture skips only where no model store is set.** Every model
  test takes its file from `test_fixtures.path` with the registry name of the
  model. That function finds the file under `models/` first, then under
  `$ZIGFERENCER_MODELS_DIR`. When neither holds the file and the variable is
  unset, the test returns `error.SkipZigTest`. When the variable is set, the
  test fails with `error.FixtureMissing` and names the two paths it searched. A
  machine that says where its models are must run every model test. A GPU test
  also skips when the device probe returns `NoVulkanDevice`,
  `InitializationFailed`, `IncompatibleDriver`, or `FeatureNotPresent`, and a
  placement test also skips on `error.VramBudgetExceeded`.
- **Run the Qwen tests in ReleaseSafe.** The Qwen3.5-2B fixture is about 2 GB.
  When the file is present, a Debug `zig build test` runs the Qwen runner tests
  for minutes. The gate runs the suite with `-Doptimize=ReleaseSafe`.
- **Tests run at fixed windows.** A test that sizes a KV window from a real model
  takes `test_fakes.test_window`, 8192 positions. A test that plans several slots
  takes `test_fakes.test_serve_window`, 512 positions. A test never takes the
  default rule, which reads the model cost and the free VRAM. A test therefore
  runs the same shape on any machine.
- **Each gate declares its own tolerance tier.** A gate is bit-exact where the
  code keeps the reduction order (`snapshot`, `llama_session_equivalence`,
  `qwen_session_equivalence`, both prefix boundaries, `vulkan_linear_q8_0`,
  `vec_add`, `kv_commit_f16`). A gate is greedy-tokens-exact across backends where
  the per-op numerics live under each kernel's declared tolerance
  (`llama_vulkan_serve`, `qwen_vulkan_serve`, `llama_hybrid_serve`). `vulkan_attention` is
  the one gate that states a relative tolerance of its own, `2e-3`, the same
  value the bench uses.
- **Bytes are compared within one executor, tokens across two.** The equivalence
  files hash the raw logits rows with FNV-1a and compare them where both sides
  ran on the same executor. Across executors the KV bits differ, because a GPU
  reduce kernel sums in a different order, so those gates compare greedy tokens
  instead.
- **The values that cross in carry no hidden state.** A `ModelCard` and a `[]TensorRef` come from the loader. A `Vocab` goes into `bpe`, and token ids come back out. The serve tests exchange framed wire messages between `Client` and `Server`.
- **Registration is a build-graph fact.** A new file under `tests/` does not run
  until it joins the `cpu_integration_tests`, `vulkan_integration_tests`, or
  `fused_integration_tests` array in `build.zig`. Each entry becomes its own
  `addTest` target that imports the `zigferencer` module. The Vulkan entries
  compile only under `-Druntime=vulkan`, because they reference the Vulkan
  modules and the embedded SPIR-V. The fused entries name a fused or a hybrid
  runner, which `src/main.zig` re-exports under `-Dops-backend=vulkan_fused`
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

## Where to start

1. `gguf_smollm2_test.zig`: the load chain every other file uses, in its simplest form.
2. `tokenizer_smollm2_test.zig`, then `tokenizer_llama3_test.zig`: the oracle-free round trip, then the golden-id shape against llama.cpp.
3. `snapshot_smollm2_test.zig`: the highest-leverage gate, and why one short token list catches a regression anywhere in the stack.
4. `llama_session_equivalence_test.zig`, then `serve_integration_test.zig`: the batched-is-solo invariant on the CPU runner, then the same invariant seen from the wire.
5. `vulkan_vec_add_test.zig` to `vulkan_linear_q8_0_test.zig` to `vulkan_attention_test.zig`: the kernel gates, simplest first.
6. `llama_prefix_runner_test.zig`, then `llama_hybrid_residency_test.zig` and `llama_hybrid_serve_test.zig`: the layer boundary, then the two-tier runner built on it.
