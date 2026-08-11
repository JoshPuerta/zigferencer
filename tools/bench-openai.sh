#!/usr/bin/env bash
# Minimal benchmark of the OpenAI HTTP endpoint, the way a real client hits it:
# spawn `serve-openai`, fire one fixed prompt repeatedly with curl, then report
# aggregate throughput, mean per-request latency, and the streaming
# time-to-first-byte.
#
# `make bench-openai` calls it. Curl handles HTTP and chunked transfer for the
# script. The numbers hold the client and HTTP overhead on purpose, because that
# overhead is the cost of the interop front, which is what these rows price. The
# binary `bench-serve` prices the batched path instead. Args:
#   $1  model registry name (make resolves it first, because the spawned server
#       cannot prompt)
#   $2  tcp port (0 = ephemeral. The bound port is read off READY)
#   $3  max_tokens per request (default 64)
#   $4  request count          (default 20)
set -euo pipefail

bin="zig-out/bin/zigferencer"
model="${1:-}"
port="${2:-0}"
n="${3:-64}"
runs="${4:-20}"
prompt="Explain what a hash table is, in two sentences."

# shellcheck source=tools/openai-common.sh
source "$(dirname "$0")/openai-common.sh"
# Greedy defaults keep the reply lengths, and so the timings, equal across runs.
spawn_openai_server "$bin" "$model" "$port" --max-tokens "$n" --max-sessions 8 --sampling greedy
discover_model_id "$base" "${model:-unknown}"
body() { printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%d,"stream":%s}' "$model_id" "$prompt" "$n" "$1"; }

echo "bench-openai: model=$model_id max_tokens=$n runs=$runs at $base"

# Warm the path (page-in weights, fill caches) so the timed runs are steady-state.
curl -s "$base/v1/chat/completions" -H 'Content-Type: application/json' -d "$(body false)" >/dev/null

# Throughput: sequential buffered completions. Aggregate tok/s = total completion
# tokens / total wall time. The response's own usage.completion_tokens is the count.
total_tok=0
total_time=0
for _ in $(seq 1 "$runs"); do
    out="$(curl -s -w '\n%{time_total}' "$base/v1/chat/completions" -H 'Content-Type: application/json' -d "$(body false)")"
    t="$(tail -n1 <<<"$out")"
    tok="$(grep -o '"completion_tokens":[0-9]*' <<<"$out" | grep -o '[0-9]*' | head -1 || echo 0)"
    total_tok=$((total_tok + tok))
    total_time="$(awk -v a="$total_time" -v b="$t" 'BEGIN{print a+b}')"
done
awk -v tok="$total_tok" -v t="$total_time" -v r="$runs" 'BEGIN{
    printf "  sequential: %d req, %d tokens, %.2fs -> %.1f tok/s, %.0f ms/req mean\n", r, tok, t, tok/t, t/r*1000
}'

# Concurrent: fire up to max-sessions (8) requests at once. The scheduler batches
# them into one forward pass per tick, so the aggregate tok/s must beat the
# sequential rows on a bandwidth-bound decode. That gap is the payoff of the
# batched serve engine.
burst=8
[ "$runs" -lt "$burst" ] && burst="$runs"
tmpdir="$(mktemp -d)"
trap 'kill "$server_pid" 2>/dev/null || true; rm -rf "$tmp" "$tmpdir"' EXIT
pids=()
start="$(date +%s.%N)"
for i in $(seq 1 "$burst"); do
    curl -s "$base/v1/chat/completions" -H 'Content-Type: application/json' -d "$(body false)" >"$tmpdir/$i" &
    pids+=("$!")
done
wait "${pids[@]}" # the curls alone. A bare `wait` also waits on the server.
end="$(date +%s.%N)"
burst_tok=0
for i in $(seq 1 "$burst"); do
    tok="$(grep -o '"completion_tokens":[0-9]*' "$tmpdir/$i" | grep -o '[0-9]*' | head -1 || echo 0)"
    burst_tok=$((burst_tok + tok))
done
awk -v tok="$burst_tok" -v s="$start" -v e="$end" -v b="$burst" 'BEGIN{
    t=e-s; printf "  concurrent: %d req in parallel, %d tokens, %.2fs -> %.1f tok/s aggregate\n", b, tok, t, tok/t
}'

# Streaming: the time to the first byte, which is when the first SSE chunk lands.
ttft="$(curl -sN -o /dev/null -w '%{time_starttransfer}' "$base/v1/chat/completions" -H 'Content-Type: application/json' -d "$(body true)")"
awk -v x="$ttft" 'BEGIN{ printf "  streaming: time-to-first-byte %.0f ms\n", x*1000 }'

echo "done, stopping serve-openai"
