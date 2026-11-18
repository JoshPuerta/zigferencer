# zigferencer: short names for the common flows, and the one entry for CI.
#
# `zig build` is the build system of record. A target exists for one of three reasons: it joins
# commands, it fixes a default that a bare `zig build` gets wrong (Debug in a measurement), or it is
# the documented entry of a flow. make echoes each command before it runs, so you can copy
# the exact `zig build` or `zigferencer` line. `make help` lists the targets and the variables.
#
# ARGS is the one way to pass arguments. It holds the words after the binary or the script name.
# The shell parses it, so a prompt with spaces takes inner quotes:
#   make run ARGS='run SmolLM2-135M-Instruct-Q8_0 --prompt "Tell me a joke"'

## ZIG      the zig 0.16.0 binary (default zig)
## OPT      the optimize mode of build, run, bench, and simd-report (default ReleaseFast)
## BACKEND  auto, naive, simd, simd_pf, vulkan, or vulkan_fused (default auto)
## FLAGS    more zig build flags, for example FLAGS="-Dtrace=true -Dtarget=x86_64-windows-gnu"
## ARGS     the words after the binary or the script name
## TEST     the test binary of debug-test: unit, dev_unit, or a tests/ file stem (default unit)
## FILTER   a part of the test names of debug-test

ZIG     ?= zig
OPT     ?= ReleaseFast
BACKEND ?= auto
FLAGS   ?=
ARGS    ?=
TEST    ?= unit
FILTER  ?=

DEV      := zig-out/bin/zigferencer-dev
CROSS    := x86_64-windows-gnu x86_64-macos aarch64-macos
GPU_BACKENDS := vulkan_fused vulkan simd
TEST_RUN := --test-timeout 120s --summary all
CONFIG    = $(strip -Dops-backend=$(BACKEND) $(FLAGS))

SHELL         := bash
.SHELLFLAGS   := -euo pipefail -c
.DEFAULT_GOAL := help
.NOTPARALLEL:
.PHONY: help build test fmt check-targets ci ci-gpu clean run bench simd-report profile gate
.PHONY: debug debug-test debug-build

##@ Build and check
help: ## Show this help
	@awk -f tools/help.awk $(MAKEFILE_LIST)

build: ## Build both binaries with OPT and BACKEND
	$(ZIG) build install dev -Doptimize=$(OPT) $(CONFIG)

test: ## Run the unit and integration tests and the format check in ReleaseSafe
	$(ZIG) build test -Doptimize=ReleaseSafe $(CONFIG) $(TEST_RUN)

fmt: ## Format the sources. zig fmt prints each file it changes
	$(ZIG) fmt src tests tools build.zig build.zig.zon

check-targets: ## Type-check the CPU: default, naive, Windows, and macOS
	$(ZIG) build check -Dops-backend=naive $(FLAGS)
	for t in native $(CROSS); do $(ZIG) build check -Dtarget=$$t $(FLAGS); done

ci: FLAGS += -Dtrace=true --seed 0x2a
ci: check-targets test ## Run the CPU part of CI. It needs no GPU and no model file
	$(ZIG) build test-unit bench-build simd-probe $(FLAGS) $(TEST_RUN)

ci-gpu: ## Compile every GPU path and run none. It needs glslc and libvulkan-dev, not a GPU
	for b in $(GPU_BACKENDS); do $(ZIG) build test-build bench-build install dev -Druntime=vulkan \
	  -Dops-backend=$$b --prefix zig-out/ci-gpu/$$b; done

clean: ## Remove .zig-cache and zig-out
	rm -rf .zig-cache zig-out

##@ Run and measure
run: build ## Build, then run the developer binary: make run ARGS="bench-model MODEL"
	$(DEV) $(ARGS)

bench: ## Run the op micro-benchmarks. A GPU BACKEND adds the GPU lines
	$(ZIG) build bench -Doptimize=$(OPT) $(CONFIG) -- $(ARGS)

simd-report: ## Print the SIMD width of each op, or the assembly of the kernels in ARGS
	$(ZIG) build simd-probe -Doptimize=$(OPT) $(FLAGS)
	tools/simd-report.sh $(ARGS)

profile: ## Profile one command with perf: make profile ARGS="bench-model MODEL"
	tools/profile.sh all $(ARGS)

gate: ## Run a token-hash gate: make gate ARGS="backends|threads|hybrid|deep"
	tools/gate.py $(ARGS)

##@ Debug
debug: ## Build both binaries in Debug with LLVM, for lldb or gdb
	$(ZIG) build install dev -Doptimize=Debug -Dllvm=true $(CONFIG)

debug-test: ## Build one test binary in Debug with LLVM at zig-out/test/TEST
	$(ZIG) build test-install -Doptimize=Debug -Dllvm=true -Dtest-bin=$(TEST) $(CONFIG) \
	  $(if $(FILTER),"-Dtest-filter=$(FILTER)")

debug-build: ## Build the zig build runner with LLVM, to step through build.zig
	tools/debug-build.sh
