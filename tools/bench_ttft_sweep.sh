#!/usr/bin/env bash
# TTFT / ITL sweep over prefill length for the GPU fused path.
#
# For each model and each prefill token count, it measures TTFT (prefill plus
# the first sample) and ITL (steady-state decode). `--prompt-tokens N`
# synthesizes an exact, tokenizer-agnostic prefill, because the timing of these
# kernels does not depend on the data. TTFT exposes the prefill-throughput work
# (GEMM plus chunked GDN). ITL is weight-bandwidth-bound and stays about flat in
# the prefill length.
#
# Usage: tools/bench_ttft_sweep.sh [decode] [prefill_len...]
#   decode       decode tokens per run (default 128)
#   prefill_len  prefill token counts to sweep (default: 32 64 128 256)
# Env:
#   ZIGFERENCER_MODELS_DIR  GGUF directory (default: models/. Set an external store)
#   ZIG                     the zig 0.16 binary (default: zig on PATH)
#   MODELS                  space-separated model names (default: Llama-3.1-8B
#                           and Qwen3.5-9B, both Q8_0)
#   SETTLE / RETRIES        settle seconds and retries, to ride out a driver that
#                           drops the device under repeated large submits
#
# Each (model, length) pair reloads the model, which takes 15 to 40 s from disk.
# The reported timings exclude that load. A full 2x4 sweep takes a few minutes.
#
# Dependencies: coreutils and awk alone. grep and awk parse the --json line.

set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

ZIG="${ZIG:-zig}"
DECODE="${1:-128}"
if [ "$#" -gt 1 ]; then shift; LENS=("$@"); else LENS=(32 64 128 256); fi
read -r -a MODELS <<< "${MODELS:-Meta-Llama-3.1-8B-Instruct-Q8_0 Qwen3.5-9B-Q8_0}"
SETTLE="${SETTLE:-4}"
RETRIES="${RETRIES:-2}"

need_cmd "$ZIG"
need_cmd awk
readonly bin=./zig-out/bin/zigferencer-dev

# Echo the single-line --json for one (model, prefill) run. It retries when a
# driver that drops the device under repeated large submits fails a GPU init. It
# returns non-zero only after every attempt fails.
run_bench() {
    local model="$1" n="$2" attempt json
    for ((attempt = 0; attempt <= RETRIES; attempt++)); do
        if json="$("$bin" bench-model "$model" --prompt-tokens "$n" --max-tokens "$DECODE" \
            --ignore-eos --runs 1 --json 2>/dev/null)" && grep -q '"ttft_ms"' <<< "$json"; then
            printf '%s\n' "$json"
            return 0
        fi
        sleep "$SETTLE"
    done
    return 1
}

# First "<field>":<number> on stdin.
json_number() { grep -oE "\"$1\":[-0-9.eE]+" | head -1 | cut -d: -f2; }

[ -n "${ZIGFERENCER_MODELS_DIR:-}" ] ||
    echo "warning: ZIGFERENCER_MODELS_DIR is unset. The run falls back to ./models" >&2

echo "==> build install dev -Druntime=vulkan -Dops-backend=vulkan_fused${ATTN:+ -Dattn-kernel=$ATTN} -Doptimize=ReleaseFast"
"$ZIG" build install dev -Druntime=vulkan -Dops-backend=vulkan_fused ${ATTN:+-Dattn-kernel=$ATTN} -Doptimize=ReleaseFast

printf '\ndecode: %s tokens/run   prefill lengths: %s\n' "$DECODE" "${LENS[*]}"

for model in "${MODELS[@]}"; do
    printf '\n########## %s ##########\n' "$model"
    printf '%8s  %12s  %10s  %10s  %14s\n' prefill ttft_ms itl_ms tok/s prefill_ms/tok
    echo '  ------------------------------------------------------------------'
    for n in "${LENS[@]}"; do
        if ! json="$(run_bench "$model" "$n")"; then
            printf '%8s  (run failed after %d attempts)\n' "$n" "$((RETRIES + 1))"
            continue
        fi
        ttft="$(json_number ttft_ms <<< "$json" || true)"
        itl="$(json_number itl_ms <<< "$json" || true)"
        if [ -z "$ttft" ] || [ -z "$itl" ]; then
            printf '%8s  (parse failed)\n' "$n"
            continue
        fi
        # tok/s = 1000/itl. prefill_ms/tok = (ttft - one decode step) / N.
        awk -v t="$ttft" -v i="$itl" -v n="$n" 'BEGIN {
            toks = (i > 0) ? 1000.0 / i : 0
            per  = (n > 0) ? (t - i) / n : 0
            printf "%8d  %12.2f  %10.2f  %10.1f  %14.2f\n", n, t, i, toks, per }'
        sleep "$SETTLE"
    done
done
