# zigferencer: convenience wrappers around `zig build`.
#
# The build system of record is `zig build` (see build.zig). Every target here is a
# thin alias for a command you can run by hand, and `make help` lists them grouped.
#
# Every target uses one symbolic status line (shared with `zigferencer info`):
#
#   * Every external command is echoed before it runs, shell-quoted behind a [$]
#     marker, so you can copy the exact `zig build` / `zigferencer` invocation.
#   * Every selectable variable resolves by one rule. Set it to skip the prompt, or leave
#     it unset to be asked:
#       1. `make build BACKEND=naive`, `make run MODEL=..`  -> used as-is, no prompt
#       2. unset, on a terminal                             -> prompted, Enter = default
#       3. unset, piped or CI (no TTY)                      -> the default, silently
#
# Zig 0.16.0 is required. `ZIG` is `zig` on PATH. Override it with
# `make <target> ZIG=/path/to/zig`.
#
# Variables. Each resolves by the rule above.
#   MODEL       registry name
#   PROMPT      prompt text
#   N           tokens
#   RUNS        bench repetitions
#   BACKEND     op kernel
#   THREADS     pool threads
#   SESSIONS    serve sessions
#   TURNS       demo follow-ups
#   PORT        serve tcp port
#
# Pass-through variables (never prompted, appended only when set):
#   PREFILL     serve prefill chunk
#   VRAM_SLOTS  serve VRAM KV slots
#   KV_VRAM     serve VRAM KV budget MiB, fused-GPU only.
#   KV_RAM      serve RAM KV budget  MiB for evicted KV blobs. Any backend
#   KV_DTYPE    KV element type      f32 or f16
#   PREFIX_DISK serve prefix-cache disk budget MiB. Entries survive restarts
#   RESUME_TTL  serve lease ceiling  ms a kept session outlives a drop, cap 3600000.
#   PROMPT_TOKENS bench prompt length  bench-model only.
#   MAX_SEQ     bench window
#   POS         swap KV depth        bench-swap only
#   STREAMS     swap KV slots        bench-swap only
#   REPS, MODE  decode-host probe    bench-decode-host only.
#   GPU_LAYERS  pinned GPU layers    fused-GPU build only
#   LAYERS      model layer count    gate-llama-hybrid only (default 30 = SmolLM2)
#   TEST        test binary          debug-test only (default unit)
#   FILTER      test name part       debug-test only
#   MODELS      sweep model list     bench-ttft-sweep only (space-separated names)
#   OUT         sweep output path    bench-threads-sweep only (default bench_threads.jsonl)
#   ONLY        one model            gate-deep only
#   GPU         add per-op vulkan    bench-model-all only (set to 1)
#   SETTLE, RETRIES  run spacing     bench-ttft-sweep only
#   ZF_READY_TIMEOUT_S  ready wait   the OpenAI scripts only
#   FLOPS       fan-out floor        bench-model, bench-serve, bench-threads-sweep.
#
# For more information, run `zigferencer --help`, then `zigferencer run --help`.
#
# Environment (read by the binary. make passes the environment through, so
# `ZIGFERENCER_MODELS_DIR=/data/models make run ..` works with no variable):
#   ZIGFERENCER_MODELS_DIR  model storage root      default: models/
#   ZIGFERENCER_CACHE_DIR   prompt-cache directory  default: .prompt_cache
#   ZIGFERENCER_LOG         runtime trace grammar   the LOG variable writes this one

ZIG     ?= zig
OPT     ?= ReleaseFast
MODEL   ?=
PROMPT  ?=
N       ?=
RUNS    ?=
BACKEND ?=
THREADS ?=
FLOPS   ?=
SESSIONS ?=
PREFILL ?=
TURNS   ?=
PORT    ?=
VRAM_SLOTS ?=
KV_VRAM ?=
KV_RAM  ?=
KV_DTYPE ?=
RESUME_TTL ?=
PREFIX_DISK ?=
GPU_LAYERS ?=
FORCE_HYBRID ?=
PROMPT_TOKENS ?=
IGNORE_EOS ?=
MAX_SEQ ?=
POS     ?=
STREAMS ?=
BUDGET  ?=
LAYERS  ?=
REPS    ?=
MODE    ?=
OUT     ?=
MODELS  ?=
TEST    ?= unit
FILTER  ?=
CACHE   ?=
DRY     ?=
FORCE   ?=
LOG     ?=
TRACE   ?=
ATTN    ?=
ARGS    ?=
FLAVOR  ?=
CALLS   ?=
CELL    ?=
PROFILE_DIR ?=
KERNEL  ?= simd.linearF32Q80
BIN     := zig-out/bin/zigferencer
# The developer binary: the release commands plus the benches, the kernel checks,
# and the demo. Every build target installs both binaries in one build.
DEV_BIN := zig-out/bin/zigferencer-dev
PROBE   := zig-out/zigferencer-simd-probe.s
PROBE_FLAGS  := -Doptimize=$(OPT) $(if $(TARGET),-Dtarget=$(TARGET)) $(if $(CPU),-Dcpu=$(CPU))

# Variables cross into the recipes as environment variables (not make-expanded text), so
# a prompt value with spaces or an apostrophe survives intact. The recipes read them
# as $$MODEL, $$PROMPT, ... and tools/menu.sh resolves each per the rule at the top.
# OPT / TEST_THREADS / TARGET / CPU / KERNEL are build-time make variables instead. They expand as
# $(OPT) and they shape `zig build`. They are not prompted and not exported.
export MODEL PROMPT N RUNS REPS MODE BACKEND THREADS FLOPS SESSIONS PREFILL TURNS PORT VRAM_SLOTS KV_VRAM KV_RAM KV_DTYPE PREFIX_DISK RESUME_TTL GPU_LAYERS FORCE_HYBRID PROMPT_TOKENS IGNORE_EOS MAX_SEQ POS STREAMS BUDGET LAYERS TEST FILTER CACHE DRY FORCE LOG TRACE ATTN ZIG OUT MODELS ARGS FLAVOR CALLS CELL PROFILE_DIR

# Recipes are bash, one shell per target (so a recipe can source the helper, prompt,
# then build + run as one flow), and fail on the first error.
SHELL := bash
.SHELLFLAGS := -e -o pipefail -c
.ONESHELL:

.DEFAULT_GOAL := help
.PHONY: help build release test fmt check check-targets clean debug debug-test debug-build bench bench-gpu bench-attn bench-model \
        bench-serve bench-swap bench-decode-host bench-model-all bench-ttft-sweep bench-threads-sweep \
        gate-llama-hybrid gate-deep simd-report simd-list simd-asm profile run run-all chat serve demo \
        demo-openai bench-openai info list fetch inspect gpu gpu-selftest \
        gpu-flush zig-version

##@ Build & quality
help: ## Show this help (targets grouped by section)
	@echo "zigferencer make targets (ZIG=$(ZIG)):"
	if [ -t 1 ] && [ -z "$${NO_COLOR+x}" ]; then b=$$'\033[1m'; cy=$$'\033[36m'; r=$$'\033[0m'; else b=''; cy=''; r=''; fi
	awk -v b="$$b" -v cy="$$cy" -v r="$$r" 'BEGIN {FS = ":.*## "} \
		/^##@/ {printf "\n%s%s%s\n", b, substr($$0, 5), r; next} \
		/^[a-zA-Z][a-zA-Z_-]*:.*## / {printf "  %s%-19s%s %s\n", cy, $$1, r, $$2}' \
		$(MAKEFILE_LIST)
	printf '\nVariables: MODEL PROMPT N RUNS BACKEND THREADS FLOPS SESSIONS PREFILL TURNS PORT VRAM_SLOTS KV_VRAM KV_RAM KV_DTYPE PREFIX_DISK RESUME_TTL GPU_LAYERS PROMPT_TOKENS MAX_SEQ POS STREAMS REPS MODE LAYERS MODELS OUT TEST FILTER  Profile: ARGS FLAVOR CALLS CELL PROFILE_DIR  Set to 1: CACHE DRY FORCE FORCE_HYBRID IGNORE_EOS BUDGET  Trace: LOG TRACE  Build: OPT TEST_THREADS ATTN (simd-*: TARGET CPU KERNEL)\n'
	printf 'Script variables: ONLY (gate-deep) GPU (bench-model-all) SETTLE RETRIES (bench-ttft-sweep) ZF_READY_TIMEOUT_S (OpenAI scripts)\n'
	printf 'Env (pass through make): ZIGFERENCER_MODELS_DIR (models/) ZIGFERENCER_CACHE_DIR (.prompt_cache) ZIGFERENCER_LOG (via LOG)\n'
	printf 'Set a variable to skip its prompt. Unset -> prompted (Enter = default).\n'
	printf 'Symbols: %s[+]%s ok  %s[-]%s no  %s[*]%s note  %s[>]%s prompt  %s[?]%s confirm  %s[$$]%s command run  %s[!]%s error\n' \
		"$$cy" "$$r" "$$cy" "$$r" "$$cy" "$$r" "$$cy" "$$r" "$$cy" "$$r" "$$cy" "$$r" "$$cy" "$$r"

# `make build` is Debug. release / run / chat / serve / bench-* build ReleaseFast.
# All write the same $(BIN) and $(DEV_BIN), so the last build wins. `make info`
# reports whatever is there now (a Debug `make build` followed by `make run` leaves
# a ReleaseFast binary).
build: ## Debug build. Prompts for the op backend (BACKEND=.. to skip)
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	run_cmd $(ZIG) build install dev $$(backend_flags "$$backend")
	ok "debug build ready ($$backend): $(BIN) $(DEV_BIN)"

release: ## Release build -> zig-out/bin/zigferencer and zigferencer-dev. Prompts for the op backend (BACKEND=.. to skip)
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	zig_build $(ZIG) $(OPT) "$$backend"
	ok "$(OPT) build ready ($$backend): $(BIN) $(DEV_BIN)"

# ReleaseSafe keeps every assertion and runs the model tests in minutes. A Debug
# suite runs the same checks, much slower.
test: ## Unit + integration tests (+ zig fmt --check), ReleaseSafe: make test [TEST_THREADS=N]
	@source tools/menu.sh
	run_cmd $(ZIG) build test -Doptimize=ReleaseSafe $(if $(TEST_THREADS),-Dtest-threads=$(TEST_THREADS)) --summary all
	ok "tests passed"

fmt: ## Format src/ tests/ tools/ build.zig. Lists files changed (silent tree = clean)
	@source tools/menu.sh
	out=$$(run_cmd $(ZIG) fmt src tests build.zig build.zig.zon tools)
	if [ -n "$$out" ]; then say "reformatted:"; printf '%s\n' "$$out" | sed 's/^/  /'; else ok "already clean"; fi

check: ## Type-check the two executables, no artifact (what ZLS runs). Tests: zig build test-build. Tools: zig build bench-build
	@source tools/menu.sh
	run_cmd $(ZIG) build check --summary all
	ok "type-check passed"

check-targets: ## Type-check for Windows and macOS (CPU runtime), no artifact
	@source tools/menu.sh
	for t in x86_64-windows-gnu x86_64-macos aarch64-macos; do
	  run_cmd $(ZIG) build check -Dtarget=$$t --summary all
	done
	ok "cross-target type-check passed"

clean: ## Remove build artifacts (.zig-cache, zig-out)
	@source tools/menu.sh
	run_cmd rm -rf .zig-cache zig-out
	ok "removed .zig-cache and zig-out"

##@ Debugging
# A Debug build keeps the safety checks and the stack traces. The LLVM code
# generator adds the local variables that a debugger shows. The next build
# target replaces these binaries, and their times are not engine times.
debug: ## Debug build with LLVM for lldb or gdb (locals visible): make debug [BACKEND=..]
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	run_cmd $(ZIG) build install dev -Doptimize=Debug -Dllvm=true $$(backend_flags "$$backend")
	ok "debug build ready ($$backend): $(BIN) $(DEV_BIN)"
	say "example: lldb $(BIN) -o \"run run SmolLM2-135M-Instruct-Q8_0 --max-tokens 1 --threads 1\""

# A test binary runs as a normal executable when no build runner drives it.
debug-test: ## One Debug test binary with LLVM -> zig-out/test/<TEST>: make debug-test [TEST=unit] [FILTER=".."] [BACKEND=..]
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	run_cmd $(ZIG) build test-install -Doptimize=Debug -Dllvm=true -Dtest-bin="$$TEST" $${FILTER:+"-Dtest-filter=$$FILTER"} $$(backend_flags "$$backend")
	ok "test binary ready ($$backend): zig-out/test/$$TEST"
	say "example, from the repository root: lldb zig-out/test/$$TEST -o run"

# `zig build` compiles build.zig with the default code generator. Only a direct
# compile selects LLVM, and gdb and lldb need LLVM for lines and locals.
debug-build: ## LLVM build runner to debug build.zig -> zig-out/debug/build-runner
	@source tools/menu.sh
	run_cmd $(ZIG) build --fetch=all
	run_cmd $(ZIG) build -l >/dev/null
	env=$$($(ZIG) env)
	field() { sed -n "s/^ *\.$$1 = \"\(.*\)\",$$/\1/p" <<<"$$env"; }
	lib=$$(field lib_dir)
	deps=$$(ls -t .zig-cache/o/*/dependencies.zig | sed -n 1p)
	vulkan=$$(sed -n 's/^ *\.{ "vulkan", "\(.*\)" },$$/\1/p' "$$deps")
	mkdir -p zig-out/debug
	run_cmd $(ZIG) build-exe -fllvm -ODebug -femit-bin=zig-out/debug/build-runner \
	  --dep @build --dep @dependencies -Mroot=$$lib/compiler/build_runner.zig -M@build=build.zig \
	  --dep $$vulkan -M@dependencies=$$deps -M$$vulkan=zig-pkg/$$vulkan/build.zig
	ok "build runner ready: zig-out/debug/build-runner"
	say "example: lldb zig-out/debug/build-runner -o \"breakpoint set --name build.resolveOpsBackend\" -o \"run $$(field zig_exe) $$lib $$PWD .zig-cache $$(field global_cache_dir) -l\""

##@ Benchmarks
bench: ## Op micro-benchmarks, CPU kernels (per-kernel GFLOP/s)
	@source tools/menu.sh
	run_cmd $(ZIG) build bench -Doptimize=$(OPT)

bench-gpu: ## Op micro-benchmarks incl. GPU kernels (needs glslc + libvulkan-dev)
	@source tools/menu.sh
	run_cmd $(ZIG) build bench -Doptimize=$(OPT) -Druntime=vulkan $${ATTN:+-Dattn-kernel=$$ATTN}

bench-attn: ## GPU attention kernel set grid only (shape x depth x rows, ms and GB/s): make bench-attn [ATTN=name]
	@source tools/menu.sh
	run_cmd $(ZIG) build bench -Doptimize=$(OPT) -Druntime=vulkan $${ATTN:+-Dattn-kernel=$$ATTN} -- attn-kit

bench-serve: ## Serve throughput vs batch size: make bench-serve [MODEL=..] [N=..] [MAX_SEQ=N] [BACKEND=..] [THREADS=N] [FLOPS=F]
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	n=$$(prompt_value "$$N" "decode tokens per turn" 48)
	threads=$$(prompt_value "$$THREADS" "pool threads, 0 = auto" 0)
	zig_build $(ZIG) $(OPT) "$$backend"
	cmd=($(DEV_BIN) bench-serve)
	[ -n "$$MODEL" ] && cmd+=("$$MODEL")
	cmd+=(--max-tokens "$$n")
	[ -n "$$MAX_SEQ" ] && cmd+=(--max-seq "$$MAX_SEQ")
	[ -n "$$threads" ] && cmd+=(--threads "$$threads")
	[ -n "$$FLOPS" ] && cmd+=(--min-parallel-flops "$$FLOPS")
	run_cmd "$${cmd[@]}"

# Times one KV slot's evict and restore against a decode tick. The swap moves KV
# between VRAM and RAM, so the target always builds the fused-GPU binary.
bench-swap: ## Tiered-KV swap cost vs a decode tick (fused-GPU build): make bench-swap [MODEL=..] [POS=..] [STREAMS=..] [BUDGET=1]
	@source tools/menu.sh
	zig_build $(ZIG) $(OPT) vulkan_fused
	cmd=($(DEV_BIN) bench-swap)
	[ -n "$$MODEL" ] && cmd+=("$$MODEL")
	[ -n "$$POS" ] && cmd+=(--pos "$$POS")
	[ -n "$$STREAMS" ] && cmd+=(--streams "$$STREAMS")
	[ -n "$$BUDGET" ] && cmd+=(--budget)
	run_cmd "$${cmd[@]}"

# Splits one decode step into its host steps. The probe records, submits, and
# waits on the fused-GPU command buffer, so the target builds that binary.
bench-decode-host: ## Host steps of one decode step (fused-GPU build): make bench-decode-host [MODEL=..] [POS=..] [REPS=..] [MODE=m3|m4|m5|gap|all]
	@source tools/menu.sh
	zig_build $(ZIG) $(OPT) vulkan_fused
	cmd=($(DEV_BIN) bench-decode-host)
	[ -n "$$MODEL" ] && cmd+=("$$MODEL")
	[ -n "$$POS" ] && cmd+=(--pos "$$POS")
	[ -n "$$REPS" ] && cmd+=(--reps "$$REPS")
	[ -n "$$MODE" ] && cmd+=(--mode "$$MODE")
	run_cmd "$${cmd[@]}"

bench-model: ## End-to-end TTFT/ITL/hash: make bench-model [MODEL=..] [N=..] [RUNS=..] [PROMPT_TOKENS=N] [MAX_SEQ=N] [IGNORE_EOS=1] [BACKEND=..] [THREADS=N] [FLOPS=F] [GPU_LAYERS=N] [KV_DTYPE=f16] [FORCE_HYBRID=1]
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	n=$$(prompt_value "$$N" "decode tokens" 32)
	runs=$$(prompt_value "$$RUNS" "runs" 3)
	threads=$$(prompt_value "$$THREADS" "pool threads, 0 = auto" 0)
	zig_build $(ZIG) $(OPT) "$$backend"
	cmd=($(DEV_BIN) bench-model)
	[ -n "$$MODEL" ] && cmd+=("$$MODEL")
	cmd+=(--max-tokens "$$n" --runs "$$runs")
	[ -n "$$threads" ] && cmd+=(--threads "$$threads")
	[ -n "$$FLOPS" ] && cmd+=(--min-parallel-flops "$$FLOPS")
	[ -n "$$GPU_LAYERS" ] && cmd+=(--gpu-layers "$$GPU_LAYERS")
	[ -n "$$KV_DTYPE" ] && cmd+=(--kv-dtype "$$KV_DTYPE")
	[ -n "$$FORCE_HYBRID" ] && cmd+=(--force-hybrid)
	[ -n "$$PROMPT_TOKENS" ] && cmd+=(--prompt-tokens "$$PROMPT_TOKENS")
	[ -n "$$MAX_SEQ" ] && cmd+=(--max-seq "$$MAX_SEQ")
	[ -n "$$IGNORE_EOS" ] && cmd+=(--ignore-eos)
	run_cmd "$${cmd[@]}"

# Targets the OpenAI HTTP endpoint with curl (spawns + reaps its own server):
# aggregate tok/s, mean latency, streaming time-to-first-byte. The numbers include
# the HTTP and client overhead, which is the cost of the interop front. The script
# builds the CPU default.
bench-openai: ## Benchmark the OpenAI endpoint (tok/s, latency, TTFT): make bench-openai [MODEL=..] [N=..] [RUNS=..]
	@source tools/menu.sh
	run_cmd $(ZIG) build install dev -Doptimize=$(OPT)
	model=$$(prompt_model "$$MODEL" $(BIN))
	n=$$(prompt_value "$$N" "max tokens per request" 64)
	runs=$$(prompt_value "$$RUNS" "request count" 20)
	port=$$(prompt_value "$$PORT" "tcp port (0 = ephemeral)" 0)
	run_cmd tools/bench-openai.sh "$${model}" "$${port}" "$${n}" "$${runs}"

# The cross-backend determinism gate: greedy decoding is deterministic, so the
# three CPU backends must emit the identical token stream. Exits non-zero on a
# hash divergence (a kernel disagrees with the naive reference).
bench-model-all: ## Cross-backend token-hash gate (naive/simd/simd_pf, GPU=1 adds per-op vulkan): make bench-model-all [MODEL=..] [PROMPT=".."] [N=..] [RUNS=..] [GPU=1]
	@source tools/menu.sh
	run_cmd tools/bench_model_all.sh "$${MODEL:-SmolLM2-135M-Instruct-Q8_0}" "$${PROMPT:-Hello}" "$${N:-32}" "$${RUNS:-5}"
	ok "cross-backend token streams match"

# The five-way llama hybrid equality gate: the CPU runner, the fused runner, and
# the hybrid at k=L / k=0 / a mid split must emit one token hash. The script
# builds the CPU and the fused binaries itself. SmolLM2 is the default, because
# its repeated GPU runs stay far inside the device budget.
gate-llama-hybrid: ## Five-way llama hybrid token-hash gate (cpu/fused/k=L/k=0/mid): make gate-llama-hybrid [MODEL=..] [LAYERS=..] [N=..] [KV_DTYPE=f16]
	@source tools/menu.sh
	run_cmd tools/bench_llama_hybrid_gate.sh "$${MODEL:-SmolLM2-135M-Instruct-Q8_0}" "$${LAYERS:-30}" "$${N:-32}"
	ok "five-way llama hybrid token streams match"

# The deep token-hash gate: Qwen3.5-2B and SmolLM2 at 8064 prompt positions
# plus 64 decode steps must emit the hash on record, on the CPU build and on the
# fused build. The script builds both binaries itself. ONLY=cpu needs no GPU.
gate-deep: ## Deep token-hash gate against the recorded hashes (cpu, fused): make gate-deep [ONLY=cpu|fused]
	@source tools/menu.sh
	run_cmd tools/bench_deep_gate.sh "$${ONLY:-all}"
	ok "deep token streams match the record"

# Long back-to-back GPU prefills. A translation-layer driver can drop the device
# under repeated large submits, so the script spaces and retries the runs.
# The script builds the fused GPU binaries itself.
bench-ttft-sweep: ## GPU TTFT/ITL vs prefill length (fused-GPU build): make bench-ttft-sweep [N=decode] [MODELS=".."]
	@source tools/menu.sh
	run_cmd tools/bench_ttft_sweep.sh $${N:+"$$N"}

bench-threads-sweep: ## CPU thread-scaling sweep + report: make bench-threads-sweep [MODEL=..] [N=..] [RUNS=..] [OUT=..] [THREADS="1 2 4 8"] [FLOPS=F|auto] [BACKEND=..]
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	zig_build $(ZIG) $(OPT) "$$backend"
	out="$${OUT:-bench_threads.jsonl}"
	run_cmd tools/bench_threads_sweep.sh $(DEV_BIN) "$${MODEL:-SmolLM2-135M-Instruct-Q8_0}" 128 "$${N:-32}" "$${RUNS:-3}" "$${FLOPS}" "$$out" $${THREADS:-1 2 4 8}
	run_cmd python3 tools/bench_sweep_report.py "$$out"

##@ SIMD inspection
simd-report: ## Per-op SIMD width for the build target: make simd-report [TARGET=..] [CPU=..]
	@source tools/menu.sh
	simd_report $(ZIG) "$(PROBE)" "$(PROBE_FLAGS)"

simd-list: ## List inspectable kernels + their backends: make simd-list [TARGET=..] [CPU=..]
	@source tools/menu.sh
	simd_report $(ZIG) "$(PROBE)" "$(PROBE_FLAGS)" list

simd-asm: ## Dump matching kernels' asm + mnemonics: make simd-asm KERNEL=linearF32Q80 [TARGET=..]
	@source tools/menu.sh
	simd_report $(ZIG) "$(PROBE)" "$(PROBE_FLAGS)" asm $(KERNEL)

##@ Profiling
# The arguments after the binary name go in ARGS. The target runs the developer
# binary, so ARGS can name any command. FLAVOR=structure builds Debug,
# so every function is a frame, and its times are not release times.
# FLAVOR=trace builds ReleaseFast with -Dtrace and writes the span trace of the
# engine into the cell. ZIGFERENCER_TRACE_GPU=submit|phases adds the GPU queue.
#   make profile ARGS="bench-model SmolLM2-135M-Instruct-Q8_0 --ignore-eos" BACKEND=simd_pf
profile: ## Profile one command with perf: make profile ARGS=".." [FLAVOR=release|trace|structure] [BACKEND=..] [CALLS=run|serve] [PROFILE_DIR=..] [CELL=name]
	@source tools/menu.sh
	[ -n "$$ARGS" ] || { err "set ARGS to the zigferencer arguments, for example ARGS=\"bench-model SmolLM2-135M-Instruct-Q8_0\""; exit 2; }
	flavor="$${FLAVOR:-release}"
	case "$$flavor" in
	    release) opt=ReleaseFast ;;
	    trace) opt=ReleaseFast; export TRACE=1 ;;
	    structure) opt=Debug ;;
	    *) err "FLAVOR must be release, trace, or structure"; exit 2 ;;
	esac
	backend=$$(prompt_backend "$$BACKEND")
	zig_build $(ZIG) "$$opt" "$$backend"
	read -ra argv <<<"$$ARGS"
	name="$${CELL:-$$(printf '%s' "$$ARGS" | tr -cs 'A-Za-z0-9.' '-' | cut -c1-60)}"
	cell="$${PROFILE_DIR:-zig-out/profiles}/$$(git rev-parse --short HEAD)/$$name-$$flavor"
	run_cmd tools/profile.sh record "$$cell" -- $(DEV_BIN) "$${argv[@]}"
	run_cmd tools/profile.sh report "$$cell"
	[ -z "$$CALLS" ] || run_cmd tools/profile.sh calls "$$cell" "tools/profile-calls/$$CALLS.txt" -- $(DEV_BIN) "$${argv[@]}"
	run_cmd tools/profile.sh index "$$(dirname "$$cell")"
	ok "open $$cell/flame.svg, or $$(dirname "$$cell")/index.html"

##@ Run inference
# CPU by default. Pick the GPU at the backend prompt (vulkan_fused), or skip it with
#   make run BACKEND=vulkan_fused MODEL=Qwen3.5-2B-Q8_0
run: ## Generate: make run [MODEL=..] [PROMPT=".."] [N=..] [BACKEND=..] [THREADS=N] [CACHE=1] [GPU_LAYERS=N] [KV_DTYPE=f16]
	@source tools/menu.sh
	run_infer $(ZIG) $(OPT) $(BIN)

# One backend, one build, every on-disk model in sequence: the same prompt under
# equal conditions, so the outputs are comparable side by side. A model a backend
# cannot run (an unsupported quant, see `make info`) is reported and skipped, not
# fatal.
run-all: ## Same prompt across every on-disk model: make run-all [PROMPT=".."] [N=..] [BACKEND=..] [THREADS=N]
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	prompt=$$(prompt_value "$$PROMPT" "prompt" "Hello")
	n=$$(prompt_value "$$N" "max tokens" 32)
	threads=$$(prompt_value "$$THREADS" "pool threads, 0 = auto" 0)
	zig_build $(ZIG) $(OPT) "$$backend"
	mapfile -t models < <($(BIN) list --present)
	if [ "$${#models[@]}" -eq 0 ]; then err "no models on disk. Run 'make fetch' first"; exit 1; fi
	say "$${#models[@]} model(s), backend $$backend, $$n tokens each"
	fails=0
	for m in "$${models[@]}"; do
		printf '\n' >&2
		say "==> $$m"
		cmd=($(BIN) run "$$m" --prompt "$$prompt" --max-tokens "$$n")
		[ -n "$$threads" ] && cmd+=(--threads "$$threads")
		run_cmd "$${cmd[@]}" || { err "$$m failed"; fails=$$((fails + 1)); }
	done
	printf '\n' >&2
	if [ "$$fails" -eq 0 ]; then ok "all $${#models[@]} model(s) completed"; \
	else say "$$fails of $${#models[@]} failed (usually an unsupported quant for this backend, see 'make info')"; fi

chat: ## Same as run but with the model's chat template applied
	@source tools/menu.sh
	run_infer $(ZIG) $(OPT) $(BIN) --chat

# Runs until Ctrl-C. A client can stop it with `shutdown` only under
# `--allow-shutdown`, and this target does not pass that flag. The READY line
# on stdout carries the bound address. Everything else goes to stderr.
serve: ## Serve sessions over TCP: make serve [MODEL=..] [SESSIONS=..] [PREFILL=N] [PORT=..] [VRAM_SLOTS=N] [KV_VRAM=MiB] [KV_RAM=MiB] [KV_DTYPE=f16] [PREFIX_DISK=MiB] [RESUME_TTL=ms] [BACKEND=..] [THREADS=N]
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	sessions=$$(prompt_value "$$SESSIONS" "concurrent sessions" 4)
	port=$$(prompt_value "$$PORT" "tcp port (0 = ephemeral)" 0)
	threads=$$(prompt_value "$$THREADS" "pool threads, 0 = auto" 0)
	zig_build $(ZIG) $(OPT) "$$backend"
	cmd=($(BIN) serve)
	[ -n "$$MODEL" ] && cmd+=("$$MODEL")
	cmd+=(--max-sessions "$$sessions" --port "$$port")
	[ -n "$$threads" ] && cmd+=(--threads "$$threads")
	[ -n "$$PREFILL" ] && cmd+=(--prefill-chunk "$$PREFILL")
	[ -n "$$VRAM_SLOTS" ] && cmd+=(--vram-slots "$$VRAM_SLOTS")
	[ -n "$$KV_VRAM" ] && cmd+=(--kv-vram "$$KV_VRAM")
	[ -n "$$KV_RAM" ] && cmd+=(--kv-ram "$$KV_RAM")
	[ -n "$$KV_DTYPE" ] && cmd+=(--kv-dtype "$$KV_DTYPE")
	[ -n "$$PREFIX_DISK" ] && cmd+=(--prefix-disk "$$PREFIX_DISK")
	[ -n "$$RESUME_TTL" ] && cmd+=(--resume-ttl "$$RESUME_TTL")
	run_cmd "$${cmd[@]}"

# Spawns its own serve, drives it over the wire, reaps it. Shows the connector's
# capabilities: live streaming, KV reuse across follow-up turns, one session
# cloned three ways, and batched multi-session decode demuxed on one connection.
demo: ## Showcase the connector (streaming, KV reuse, clone, batched): make demo [MODEL=..] [N=..] [TURNS=..] [SESSIONS=..] [KV_DTYPE=f16] [BACKEND=..]
	@source tools/menu.sh
	backend=$$(prompt_backend "$$BACKEND")
	n=$$(prompt_value "$$N" "decode tokens per turn" 48)
	turns=$$(prompt_value "$$TURNS" "follow-up rounds" 2)
	zig_build $(ZIG) $(OPT) "$$backend"
	cmd=($(DEV_BIN) serve-demo)
	[ -n "$$MODEL" ] && cmd+=("$$MODEL")
	cmd+=(--max-tokens "$$n" --turns "$$turns")
	[ -n "$$SESSIONS" ] && cmd+=(--max-sessions "$$SESSIONS")
	[ -n "$$KV_DTYPE" ] && cmd+=(--kv-dtype "$$KV_DTYPE")
	run_cmd "$${cmd[@]}"

# Drives the OpenAI-compatible HTTP endpoint with curl: GET /v1/models, one
# buffered completion, and one streamed (curl -N) completion. The script builds
# the CPU default.
demo-openai: ## curl the OpenAI endpoint (/v1/chat/completions, streaming + not): make demo-openai [MODEL=..] [PORT=..] [SESSIONS=..]
	@source tools/menu.sh
	run_cmd $(ZIG) build install dev -Doptimize=$(OPT)
	model=$$(prompt_model "$$MODEL" $(BIN))
	port=$$(prompt_value "$$PORT" "tcp port (0 = ephemeral)" 8080)
	run_cmd tools/demo-openai.sh "$$model" "$$port" "$$SESSIONS"

##@ Models & host
# list / fetch / inspect read model data and never run a kernel. Any existing
# binary serves them (the CPU default is built only when none exists), and the
# binary's own picker chooses the model when MODEL is unset.
info: ## Inspect the CURRENT binary (build config, host card, support matrix). Does NOT rebuild. Run make build or make gpu first
	@source tools/menu.sh
	bin_subcommand $(BIN) info

list: ## List registered models + whether each GGUF is present on disk
	@source tools/menu.sh
	ensure_bin "$(BIN)" "$(ZIG)"
	run_cmd $(BIN) list

fetch: ## Download a model: make fetch [MODEL=<name|org/repo/file.gguf>] [DRY=1] [FORCE=1]
	@source tools/menu.sh
	ensure_bin "$(BIN)" "$(ZIG)"
	cmd=($(BIN) fetch)
	[ -n "$$MODEL" ] && cmd+=("$$MODEL")
	[ -n "$$DRY" ] && cmd+=(--dry-run)
	[ -n "$$FORCE" ] && cmd+=(--force)
	run_cmd "$${cmd[@]}"

inspect: ## Print a downloaded GGUF's ModelCard: make inspect [MODEL=..] (prompts if unset)
	@source tools/menu.sh
	ensure_bin "$(BIN)" "$(ZIG)"
	cmd=($(BIN) inspect)
	[ -n "$$MODEL" ] && cmd+=("$$MODEL")
	run_cmd "$${cmd[@]}"

##@ GPU (Vulkan)
# Needs glslc (shaderc) + libvulkan-dev. `make gpu` builds the fused-GPU binary.
# The vulkan_fused choice at the `make run` (or `make build`) backend prompt does the same.
gpu: ## Build the fused-GPU binary (= release BACKEND=vulkan_fused)
	@source tools/menu.sh
	zig_build $(ZIG) $(OPT) vulkan_fused
	ok "$(OPT) GPU build ready (vulkan_fused): $(BIN) $(DEV_BIN)"

gpu-selftest: gpu ## Bit-check a GPU kernel vs its CPU reference (default: gdn)
	@source tools/menu.sh
	run_cmd $(DEV_BIN) gpu-selftest gdn

# zigferencer owns its VRAM for the life of one process and frees it on exit, so
# there is no cross-process GPU cache to drop. "flush" therefore means: end a
# lingering run that still holds the device (a hung, backgrounded, or crashed one),
# which releases its allocations back to the host.
gpu-flush: ## Reclaim GPU memory by ending a lingering zigferencer process (FORCE=1 skips the prompt)
	@source tools/menu.sh
	mapfile -t pids < <(pgrep -x 'zigferencer|zigferencer-dev' || true)
	if [ "$${#pids[@]}" -eq 0 ]; then ok "no zigferencer or zigferencer-dev process is holding the GPU (its VRAM frees on exit)"; exit 0; fi
	say "$${#pids[@]} zigferencer or zigferencer-dev process(es) hold GPU memory:"
	ps -o pid,etime,rss,args -p "$$(IFS=,; echo "$${pids[*]}")" >&2 || true
	if [ -z "$$FORCE" ] && ! confirm "end them and reclaim their VRAM?"; then say "left untouched (FORCE=1 skips this prompt)"; exit 0; fi
	run_cmd kill "$${pids[@]}" || true
	for pid in "$${pids[@]}"; do timeout 3 tail --pid="$$pid" -f /dev/null 2>/dev/null || true; done
	mapfile -t alive < <(pgrep -x 'zigferencer|zigferencer-dev' || true)
	if [ "$${#alive[@]}" -ne 0 ]; then run_cmd kill -9 "$${alive[@]}" || true; fi
	ok "ended $${#pids[@]} process(es). VRAM released to the host"

##@ Misc
zig-version: ## Print the Zig binary the Makefile uses (must be 0.16.0)
	@source tools/menu.sh
	run_cmd $(ZIG) version
