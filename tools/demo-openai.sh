#!/usr/bin/env bash
# Drive the OpenAI-compatible HTTP endpoint with curl, end to end: spawn
# `serve-openai`, discover its model, run a buffered and a streamed chat
# completion, then reap the server.
#
# `make demo-openai` calls it, and builds a CPU binary first, because
# serve-openai runs on the CPU alone. Args:
#   $1  model registry name (make resolves it first, because the spawned server
#       cannot prompt)
#   $2  tcp port            (0 = ephemeral. The bound port is read off READY)
#   $3  max sessions        (empty -> the server default of 4. This is the batch
#                            width, so it also caps the prefill chunk, 1 to 8)
set -euo pipefail

bin="zig-out/bin/zigferencer"
model="${1:-}"
port="${2:-0}"
sessions="${3:-}"

# Start the server. Its READY line ("READY tcp 127.0.0.1:PORT") lands on stdout.
# shellcheck source=tools/openai-common.sh
source "$(dirname "$0")/openai-common.sh"
extra=()
[ -n "$sessions" ] && extra+=(--max-sessions "$sessions")
spawn_openai_server "$bin" "$model" "$port" ${extra[@]+"${extra[@]}"}
echo "serve-openai ready at $base"

# Discover the served model id (one model per process).
echo
echo "== GET /v1/models =="
discover_model_id "$base" "${model:-unknown}"
echo "$models_json"

# Buffered completion (stream:false) -> one chat.completion JSON object.
echo
echo "== POST /v1/chat/completions (buffered) =="
curl -s "$base/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is a KV cache?\"}],\"stream\":false}"
echo

# Streamed completion (stream:true) -> SSE. curl -N disables buffering so the
# `data:` chunks print as they arrive, terminated by `data: [DONE]`.
echo
echo "== POST /v1/chat/completions (streaming, curl -N) =="
curl -sN "$base/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain prefill versus decode in two short sentences.\"}],\"stream\":true}"
echo

echo
echo "done, stopping serve-openai"
