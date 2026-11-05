#!/usr/bin/env bash
# The deep token-hash gate. Greedy decoding is deterministic, so one model at
# one depth must emit one recorded token stream on the CPU build and on the
# fused GPU build. Each row fills 8064 prompt positions and decodes 64 tokens
# in a window of 8192. The prompt passes `limits.attn_split_len`, so the
# prefill rows reach the split passes of the GPU. It also passes the tiles of
# the CPU attention. So the rows reach the deep attention paths that short
# gates never run. Every row must equal the hash on record for its model. A
# change that moves both executors the same way therefore also fails the
# gate. The hashes on record belong to the default `split` attention kernel set
# (`ATTN`).
#
# The rows run Qwen3.5-2B first, then SmolLM2. The CPU row of the 2B is the
# slowest row, because the whole prefill runs on the host. The table prints
# TTFT and ITL as information only, and the gate never judges them.
#
# Usage: tools/bench_deep_gate.sh [all|cpu|fused]
#        all    (the default) builds and runs the CPU rows, then the fused rows.
#        cpu    builds and runs the CPU rows alone. It needs no GPU, and it
#               leaves the CPU binaries in zig-out.
#        fused  builds and runs the fused GPU rows alone.
# Env:   ZIG    the zig 0.16 binary (default: zig on PATH)

set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# The expected hash of each model. A DIFF is a regression until a person
# proves otherwise. Investigate the divergence first. Update a record only
# after the CPU and the fused rows agree on the new value on one tree.
readonly expect_qwen_2b=0xd4f86c9b4daf7731
readonly expect_smollm2=0xed2a39102b50bc4e

readonly models=(Qwen3.5-2B-Q8_0 SmolLM2-135M-Instruct-Q8_0)
readonly bin=./zig-out/bin/zigferencer-dev
readonly bench_args=(--prompt-tokens 8064 --max-tokens 64 --ignore-eos --runs 1 --max-seq 8192 --json)

ZIG="${ZIG:-zig}"
mode="${1:-all}"
case "$mode" in
    all) executors=(cpu fused) ;;
    cpu) executors=(cpu) ;;
    fused) executors=(fused) ;;
    *) die "unknown mode '$mode'. Use all, cpu, or fused." ;;
esac

expected_for() {
    case "$1" in
        Qwen3.5-2B-Q8_0) printf '%s' "$expect_qwen_2b" ;;
        SmolLM2-135M-Instruct-Q8_0) printf '%s' "$expect_smollm2" ;;
        *) die "no recorded hash for $1" ;;
    esac
}

build() {
    case "$1" in
        cpu)
            printf '==> build cpu (ReleaseFast, default backend)\n'
            "$ZIG" build install dev -Doptimize=ReleaseFast >/dev/null 2>&1 || die "cpu build failed"
            ;;
        fused)
            printf '==> build fused gpu (ReleaseFast, -Dops-backend=vulkan_fused)\n'
            "$ZIG" build install dev -Doptimize=ReleaseFast -Druntime=vulkan -Dops-backend=vulkan_fused ${ATTN:+-Dattn-kernel=$ATTN} \
                >/dev/null 2>&1 || die "fused build failed"
            ;;
    esac
}

# One row per run, kept for the verdict: model, executor, hash.
results=()

for executor in "${executors[@]}"; do
    build "$executor"
    for model in "${models[@]}"; do
        printf '    run %s on %s\n' "$model" "$executor"
        json=$("$bin" bench-model "$model" "${bench_args[@]}" 2>/dev/null) ||
            die "bench failed: $model on $executor"
        hash=$(grep -oE '"tokens_hash":"0x[0-9a-f]{16}"' <<<"$json" | head -1 |
            grep -oE '0x[0-9a-f]{16}') || die "no tokens_hash: $model on $executor"
        ttft=$(grep -oE '"ttft_ms":[0-9.]+' <<<"$json" | head -1 | cut -d: -f2)
        itl=$(grep -oE '"itl_ms":[0-9.]+' <<<"$json" | head -1 | cut -d: -f2)
        results+=("$model $executor $hash ${ttft:-?} ${itl:-?}")
    done
done

printf '\n%-28s %-6s %-18s %-18s %-5s %12s %9s\n' model exec hash expected check ttft_ms itl_ms
failed=0
for row in "${results[@]}"; do
    read -r model executor hash ttft itl <<<"$row"
    expected=$(expected_for "$model")
    check=ok
    if [ "$hash" != "$expected" ]; then
        check=DIFF
        failed=1
    fi
    printf '%-28s %-6s %-18s %-18s %-5s %12s %9s\n' \
        "$model" "$executor" "$hash" "$expected" "$check" "$ttft" "$itl"
done

# Two executors that agree on a new value point at a change in the shared
# model path, not at one backend.
if [ "${#executors[@]}" -eq 2 ]; then
    for model in "${models[@]}"; do
        cpu_hash=""
        fused_hash=""
        for row in "${results[@]}"; do
            read -r m executor hash _ _ <<<"$row"
            [ "$m" = "$model" ] || continue
            if [ "$executor" = cpu ]; then cpu_hash=$hash; else fused_hash=$hash; fi
        done
        if [ "$cpu_hash" = "$fused_hash" ] && [ "$cpu_hash" != "$(expected_for "$model")" ]; then
            printf '%s: the CPU and the fused rows agree on %s, which differs from the record.\n' \
                "$model" "$cpu_hash"
        fi
    done
fi

[ "$failed" -eq 0 ] || die "a deep row differs from its record"
printf '\ndeep rows match the record\n'
