#!/usr/bin/env bash
# Build each op backend in turn (naive -> simd -> simd_pf, plus the per-op
# vulkan backend under GPU=1), bench the same
# prompt through each, and check they all emit the identical token stream.
#
# The cross-backend tokens_hash is the determinism gate. Greedy sampling is
# deterministic, so every eligible backend must produce the same tokens. The
# SIMD attention differs from the naive oracle in bits, inside a declared
# tolerance, and the gate relies on the tokens to tolerate that difference. So
# a mismatch alone does not prove a kernel bug. Check the kernel tolerance
# tests first. The TTFT and ITL per backend ride along as a coarse speed view.
#
# Usage: tools/bench_model_all.sh [model] [prompt] [decode] [runs] [--chat]
# Env:   ZIG   the zig 0.16 binary (default: zig on PATH)
#        GPU=1 adds the per-op vulkan backend to the ladder. It needs glslc and
#              libvulkan-dev, and it runs every backend at f32 KV, because the
#              per-op arm frames its KV at f32 alone.
#
# Dependencies: coreutils and awk alone. The bench binary emits --json as one
# line of a known shape, which grep and awk parse here. A JSON library, and
# python with it, stays out by choice: a parser dependency for five scalars does
# not pay for itself.

set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

ZIG="${ZIG:-zig}"
model="${1:-SmolLM2-135M-Instruct-Q8_0}"
prompt="${2:-Hello}"
decode="${3:-32}"
runs="${4:-5}"
chat_args=()
if [ "${5:-}" = "--chat" ]; then chat_args+=(--chat); fi

need_cmd "$ZIG"
need_cmd awk
readonly bin=./zig-out/bin/zigferencer-dev
backends=(naive simd simd_pf)
kv_args=()
if [ "${GPU:-}" = "1" ]; then
    backends+=(vulkan)
    kv_args+=(--kv-dtype f32)
fi
readonly backends kv_args

# Every "<field>":<number> on stdin, one numeric value per line.
json_numbers() { grep -oE "\"$1\":[-0-9.eE]+" | cut -d: -f2; }
# "<mean> <sample-stddev>" of the numbers on stdin (stddev 0 for a single value).
mean_stddev() {
    awk '{ a[NR] = $1; sum += $1 }
         END { if (!NR) { print "0 0"; exit }
               m = sum / NR
               for (i = 1; i <= NR; i++) var += (a[i] - m) ^ 2
               printf "%.3f %.3f\n", m, (NR > 1) ? sqrt(var / (NR - 1)) : 0 }'
}

printf 'model:   %s\n' "$model"
printf 'prompt:  %s  %s\n' "$prompt" "${chat_args[*]}"
printf 'decode:  %s tokens x %s runs\n\n' "$decode" "$runs"

rows=()
hashes=()
for backend in "${backends[@]}"; do
    build_args=(-Doptimize=ReleaseFast -Dops-backend="$backend")
    if [ "$backend" = vulkan ]; then build_args+=(-Druntime=vulkan ${ATTN:+-Dattn-kernel=$ATTN}); fi
    printf '==> build %s\n' "${build_args[*]}"
    if ! "$ZIG" build install dev "${build_args[@]}" >/dev/null 2>&1; then
        printf '    skipped (build failed: the target lacks the needed ISA)\n'
        continue
    fi
    json="$("$bin" bench-model "$model" --prompt "$prompt" --max-tokens "$decode" \
        --runs "$runs" "${chat_args[@]}" "${kv_args[@]}" --json)" || die "bench-model failed for backend '$backend'"

    read -r ttft_mean ttft_stddev < <(json_numbers ttft_ms <<< "$json" | mean_stddev)
    read -r itl_mean itl_stddev < <(json_numbers itl_ms <<< "$json" | mean_stddev)
    hash="$(grep -oE '"tokens_hash":"[^"]*"' <<< "$json" | head -1 | sed -E 's/.*:"(.*)"/\1/' || true)"

    rows+=("$(printf '%-10s  %10s  %10s  %10s  %10s   %s' \
        "$backend" "$ttft_mean" "$ttft_stddev" "$itl_mean" "$itl_stddev" "$hash")")
    hashes+=("$hash")
done

echo
echo '=== summary ==='
printf '%-10s  %10s  %10s  %10s  %10s   %s\n' backend ttft_mean ttft_std itl_mean itl_std tokens_hash
echo '---'
[ "${#rows[@]}" -gt 0 ] || die "no backend was successfully benched"
printf '%s\n' "${rows[@]}"

echo
n_distinct="$(printf '%s\n' "${hashes[@]}" | sort -u | grep -c . || true)"
if [ "$n_distinct" -eq 1 ]; then
    printf 'cross-backend determinism: OK  (%d backend(s) emitted the same token stream)\n' "${#hashes[@]}"
else
    printf 'cross-backend determinism: DIVERGED  (%d distinct hashes across %d backends)\n' \
        "$n_distinct" "${#hashes[@]}"
    echo '  -> an ops/* kernel disagrees with the naive oracle. Investigate it.'
    exit 1
fi
