#!/usr/bin/env bash
# Thread and fan-out floor scaling sweep for the CPU op backends. It emits JSON
# lines, one object per cell, that merge `bench-model --json` with the
# whole-process CPU%, wall time, and maxrss from /usr/bin/time. The CPU% is the
# direct measure of how many cores the run uses: at threads=T it tops out near
# T*100% only when the work parallelizes. A much lower value means the op is
# bandwidth-bound, or that it falls below the fan-out floor FLOPS. The token
# hash of every cell must be the same, because the pool split keeps the
# reduction order of every output.
#
# Usage:
#   bench_threads_sweep.sh BIN MODEL PROMPT_TOK MAX_TOK RUNS FLOPS OUT THREADS...
# An empty FLOPS or `auto` passes no --min-parallel-flops flag, so the default
# floor of the binary applies. The JSON line then records "auto".
# Example:
#   bench_threads_sweep.sh zig-out/bin/zigferencer-dev SmolLM2-135M-Instruct-Q8_0 \
#       128 32 5 auto out.jsonl 1 2 4 8
set -euo pipefail
BIN=$1; MODEL=$2; PROMPT_TOK=$3; MAX_TOK=$4; RUNS=$5; FLOPS=$6; OUT=$7; shift 7
THREADS="$*"

flops_args=()
flops_json='"auto"'
if [ -n "$FLOPS" ] && [ "$FLOPS" != auto ]; then
  flops_args=(--min-parallel-flops "$FLOPS")
  flops_json=$FLOPS
else
  FLOPS=auto
fi

for T in $THREADS; do
  err=$(mktemp); out=$(mktemp)
  /usr/bin/time -f '{"wall_s":%e,"maxrss_kb":%M,"cpu_pct":"%P"}' \
    "$BIN" bench-model "$MODEL" \
      --prompt-tokens "$PROMPT_TOK" --max-tokens "$MAX_TOK" --ignore-eos \
      --runs "$RUNS" --threads "$T" "${flops_args[@]}" --json \
      >"$out" 2>"$err" || { echo "FAILED model=$MODEL T=$T flops=$FLOPS" >&2; sed 's/^/  /' "$err" >&2; rm -f "$err" "$out"; continue; }
  bench_json=$(grep '^{' "$out" | tail -1)
  time_json=$(grep '"wall_s"' "$err" | tail -1)
  echo "{\"min_parallel_flops\":$flops_json,\"time\":$time_json,\"bench\":$bench_json}" >>"$OUT"
  echo "  done model=$MODEL T=$T flops=$FLOPS" >&2
  rm -f "$err" "$out"
done
