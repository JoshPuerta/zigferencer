#!/usr/bin/env bash
# The token-hash gates. Greedy decoding is deterministic, so every path through one model must emit
# one token stream. A gate is a table of rows, and a row names a build, a label, the expected hash,
# and the arguments of `bench-model`. A row expects `same`, the hash of the first row that runs, or
# a hash on record. The table prints TTFT and ITL as information, and a gate never judges them.
#
# To add a case, add one row. Rows that share a build stay adjacent, so the gate builds once.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=tools/common.sh
source tools/common.sh

# -- constants ---------------------------------------------------------------

readonly bin=zig-out/bin/zigferencer-dev
readonly smollm2=SmolLM2-135M-Instruct-Q8_0
readonly qwen_2b=Qwen3.5-2B-Q8_0

# A DIFF is a regression until a person proves otherwise. Update a record only after the CPU and the
# fused rows agree on the new value on one tree.
readonly record_smollm2=0xed2a39102b50bc4e
readonly record_qwen_2b=0xd4f86c9b4daf7731

# -- state -------------------------------------------------------------------

# parse_args sets these once, and main makes them readonly.
gate=""
scope=all
model="$smollm2"
list=0
extra=()

# -- tables ------------------------------------------------------------------

# Every CPU kernel set against the naive reference. The SIMD attention differs from naive in bits,
# inside a declared tolerance, so a DIFF here sends you to the kernel tolerance tests first. The
# per-op backend frames its KV at f32 alone, so its row sets f32 on every row.
rows_backends() {
    local a="$model --prompt Hello --max-tokens 32 --runs 1"
    [ "$scope" = cpu ] || a="$a --kv-dtype f32"
    cat <<EOF
naive    naive    same  $a
simd     simd     same  $a
simd_pf  simd_pf  same  $a
vulkan   vulkan   same  $a
EOF
}

# The pool split keeps the reduction order of every output, so a thread count cannot move a token.
rows_threads() {
    local a="$model --prompt-tokens 128 --max-tokens 32 --ignore-eos --runs 1"
    cat <<EOF
auto  t=1  same  $a --threads 1
auto  t=2  same  $a --threads 2
auto  t=4  same  $a --threads 4
auto  t=8  same  $a --threads 8
EOF
}

# The CPU runner, the fused runner, and the hybrid forced all-host (k=L), forced all-GPU (k=0), and
# at a mid split. A split forward pass that disagrees with its parents is a correctness bug. @all
# and @half take their values from the model file. Run a large model alone, with time between runs.
rows_hybrid() {
    local a="$model --runs 1 --max-tokens 32 --ignore-eos"
    cat <<EOF
auto          cpu    same  $a
vulkan_fused  fused  same  $a
vulkan_fused  k=L    same  $a --gpu-layers 0 --force-hybrid
vulkan_fused  k=0    same  $a --gpu-layers @all --force-hybrid
vulkan_fused  mid    same  $a --gpu-layers @half
EOF
}

# The prompt passes `limits.attn_split_len`, so the prefill reaches the split passes of the GPU and
# the tiles of the CPU attention. Short gates never run these paths. Each row must equal the record,
# so a change that moves both executors the same way also fails. The 2B CPU row is the slowest.
rows_deep() {
    local a="--prompt-tokens 8064 --max-tokens 64 --ignore-eos --runs 1 --max-seq 8192"
    cat <<EOF
auto          2B-cpu      $record_qwen_2b  $qwen_2b $a
auto          smol-cpu    $record_smollm2  $smollm2 $a
vulkan_fused  2B-fused    $record_qwen_2b  $qwen_2b $a
vulkan_fused  smol-fused  $record_smollm2  $smollm2 $a
EOF
}

# -- helpers -----------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: tools/gate.sh GATE [--cpu | --gpu] [--model NAME] [--list] [-- BENCH_FLAG...]

Gates:
  backends  naive, simd, simd_pf, and per-op vulkan emit one token stream
  threads   1, 2, 4, and 8 pool threads emit one token stream
  hybrid    the CPU runner, the fused runner, and three hybrid splits emit one token stream
  deep      Qwen3.5-2B and SmolLM2 at 8064 positions match the hashes on record

Options:
  --cpu         run only the rows of a CPU build
  --gpu         run only the rows of a GPU build
  --model NAME  replace SmolLM2 in backends, threads, or hybrid
  --list        print the rows and run nothing. Without a GATE, print every gate
  -- FLAG...    add bench-model flags to every row, for example -- --kv-dtype f16

Environment:
  ZIG                     the zig 0.16.0 binary (default zig)
  FLAGS                   more zig build flags
  ZIGFERENCER_MODELS_DIR  the model store (default models/)

Exit codes: 0 every row matches, 1 a row differs or fails, 2 a usage error.
EOF
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            -h | --help)
                usage
                exit 0
                ;;
            --cpu | --gpu) scope="${1#--}" ;;
            --list) list=1 ;;
            --model)
                [ $# -ge 2 ] || die_usage "--model needs a name"
                model="$2"
                shift
                ;;
            --)
                shift
                extra=("$@")
                break
                ;;
            backends | threads | hybrid | deep) gate="$1" ;;
            *) die_usage "unknown argument '$1'" ;;
        esac
        shift
    done
    if [ -z "$gate" ]; then
        [ "$list" -eq 1 ] || die_usage "name a gate"
        gate=all
    fi
    if [ "$gate" = deep ] && [ "$model" != "$smollm2" ]; then
        die_usage "deep holds a record per model, so it takes no --model"
    fi
}

host_has_simd() { grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; }

# The rows of the gate that this run can reach. The build refuses a vector backend on a host without
# AVX2 and FMA, so such a row is a skip. Every other build failure stops the gate.
selected_rows() {
    local build rest
    "rows_$1" | while read -r build rest; do
        case "$build" in
            vulkan*) [ "$scope" != cpu ] || continue ;;
            *) [ "$scope" != gpu ] || continue ;;
        esac
        case "$build" in
            simd*) host_has_simd || { note "skip $build: the host lacks AVX2 or FMA"; continue; } ;;
        esac
        printf '%-13s %s\n' "$build" "$rest"
    done
}

# Replace @all and @half with the layer count of the model. The file is the one source of that
# count, so a row cannot name a split that the model lacks.
resolve_layers() {
    local args="$1" layers
    if [[ "$args" == *@* ]]; then
        layers="$("$bin" inspect "$model" | awk '$1 == "layers:" { print $2 }')"
        [ -n "$layers" ] || die "inspect printed no layer count for $model"
        args="${args//@all/$layers}"
        args="${args//@half/$((layers / 2))}"
    fi
    printf '%s' "$args"
}

# The value of one field of the bench-model --json line. A row runs once, so a field appears once.
json_field() { sed -nE "s/.*\"$1\":\"?([^\",}]+).*/\1/p" "$tmp/out"; }

print_row() { printf '%-11s %-13s %-18s %-18s %-5s %11s %9s\n' "$@"; }

run_rows() {
    local last="" first="" failed=0
    local row build label expect args argv want hash check
    print_row label build hash expected check ttft_ms itl_ms
    for row in "$@"; do
        read -r build label expect args <<<"$row"
        if [ "$build" != "$last" ]; then
            zig_build ReleaseFast "$build"
            last="$build"
        fi
        read -ra argv <<<"$(resolve_layers "$args")"
        if ! "$bin" bench-model "${argv[@]}" "${extra[@]}" --json >"$tmp/out" 2>"$tmp/err"; then
            cat "$tmp/err" >&2
            die "row $label failed"
        fi
        hash="$(json_field tokens_hash)"
        [ -n "$hash" ] || die "row $label printed no tokens_hash"

        want="$expect"
        if [ "$expect" = same ]; then
            first="${first:-$hash}"
            want="$first"
        fi
        check=ok
        if [ "$hash" != "$want" ]; then
            check=DIFF
            failed=$((failed + 1))
        fi
        print_row "$label" "$build" "$hash" "$want" "$check" \
            "$(json_field ttft_ms)" "$(json_field itl_ms)"
    done

    if [ "$failed" -gt 0 ]; then
        die "gate $gate: $failed row(s) differ. Investigate before you change a record"
    fi
    if [ -n "$first" ]; then
        ok "gate $gate: $# rows emit $first"
    else
        ok "gate $gate: $# rows match the record"
    fi
}

main() {
    parse_args "$@"
    readonly gate scope model list extra

    local table=() g
    if [ "$gate" = all ]; then
        for g in backends threads hybrid deep; do
            printf '# %s\n' "$g"
            selected_rows "$g"
        done
        return
    fi
    mapfile -t table < <(selected_rows "$gate")
    if [ "$list" -eq 1 ]; then
        printf '%s\n' "${table[@]}"
        return
    fi
    if [ "${#table[@]}" -lt 2 ]; then
        die_usage "gate $gate selected ${#table[@]} row(s), too few to compare"
    fi

    # The EXIT trap runs after main returns, so `tmp` stays a global.
    tmp="$(mktemp -d)"
    readonly tmp
    trap 'rm -rf "$tmp"' EXIT
    run_rows "${table[@]}"
}

main "$@"
