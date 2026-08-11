#!/usr/bin/env bash
# The five-way llama hybrid equality gate. Greedy decoding is deterministic and
# the GPU kernels are bit-exact against the CPU oracle, so five paths through one
# llama model must emit one token stream: the CPU runner, the fused runner, the
# hybrid forced to all-host (k = L), the hybrid forced to all-GPU (k = 0), and
# the hybrid at a mid split. A mismatch means the split forward pass disagrees with
# its parents, which is a correctness bug.
#
# The default model is SmolLM2, because a small model keeps repeated GPU runs
# far inside the device budget. Pass a large model as a single spaced run alone.
#
# Usage: tools/bench_llama_hybrid_gate.sh [model] [layers] [decode]
# Env:   ZIG       the zig 0.16 binary (default: zig on PATH)
#        KV_DTYPE  KV element type, f32 or f16. Empty (the default) passes no
#                  flag, which keeps every command line here byte for byte

set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

ZIG="${ZIG:-zig}"
model="${1:-SmolLM2-135M-Instruct-Q8_0}"
layers="${2:-30}"
decode="${3:-32}"
mid=$((layers / 2))
readonly bin=./zig-out/bin/zigferencer-dev
bench_args=(--runs 1 --max-tokens "$decode" --ignore-eos)
if [ -n "${KV_DTYPE:-}" ]; then
    bench_args+=(--kv-dtype "$KV_DTYPE")
fi
readonly bench_args

# The last 0x... token on stdout is the tokens_hash column of the result row.
run_hash() {
    "$bin" bench-model "$model" "${bench_args[@]}" "$@" 2>/dev/null |
        grep -oE '0x[0-9a-f]+' | tail -1
}

printf 'model:  %s (%s layers, mid split %s)\n' "$model" "$layers" "$mid"
printf 'decode: %s tokens, greedy, --ignore-eos, kv %s\n\n' "$decode" "${KV_DTYPE:-f32 (default)}"

printf '==> build cpu (default backend)\n'
"$ZIG" build install dev -Doptimize=ReleaseFast >/dev/null 2>&1 || die "cpu build failed"
h_cpu=$(run_hash) || die "cpu bench failed"
printf '    cpu             %s\n' "$h_cpu"

printf '==> build fused gpu (-Dops-backend=vulkan_fused)\n'
"$ZIG" build install dev -Doptimize=ReleaseFast -Druntime=vulkan -Dops-backend=vulkan_fused ${ATTN:+-Dattn-kernel=$ATTN} \
    >/dev/null 2>&1 || die "fused build failed"
h_gpu=$(run_hash) || die "fused bench failed"
printf '    fused           %s\n' "$h_gpu"
h_kl=$(run_hash --gpu-layers 0 --force-hybrid) || die "hybrid k=L bench failed"
printf '    hybrid k=L      %s\n' "$h_kl"
h_k0=$(run_hash --gpu-layers "$layers" --force-hybrid) || die "hybrid k=0 bench failed"
printf '    hybrid k=0      %s\n' "$h_k0"
h_mid=$(run_hash --gpu-layers "$mid") || die "hybrid mid bench failed"
printf '    hybrid mid      %s\n' "$h_mid"

printf '\n'
for h in "$h_gpu" "$h_kl" "$h_k0" "$h_mid"; do
    [ "$h" = "$h_cpu" ] || die "token-hash divergence: cpu $h_cpu vs $h"
done
printf 'five-way match: %s\n' "$h_cpu"
