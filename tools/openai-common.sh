#!/usr/bin/env bash
# Shared spawn/READY/reap block for the two OpenAI endpoint scripts
# (bench-openai.sh, demo-openai.sh). This file is sourced, never executed.

# spawn_openai_server <bin> <model> <port> [extra argv...]
# Start `serve-openai` in the background with its output captured, then wait
# for the READY line. Sets three variables for the caller:
#   server_pid  the child's pid
#   tmp         the capture file
#   base        http://host:port parsed off the READY line
# The function installs an EXIT trap that kills the server and removes the
# capture file. A caller that re-traps must keep both actions.
# The READY deadline is ZF_READY_TIMEOUT_S (default 180 s): a multi-GB GGUF is
# slow to mmap+prefault (a cold 9B is ~40 s). The wait stops the instant the
# server dies.
spawn_openai_server() {
    local bin="$1" model="$2" port="$3"
    shift 3
    local argv=("$bin" serve-openai)
    [ -n "$model" ] && argv+=("$model")
    argv+=(--port "$port" "$@")

    tmp="$(mktemp)"
    "${argv[@]}" >"$tmp" 2>&1 &
    server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null || true; rm -f "$tmp"' EXIT

    local addr="" line deadline
    deadline=$(($(date +%s) + ${ZF_READY_TIMEOUT_S:-180}))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        line="$(grep -m1 '^READY ' "$tmp" || true)"
        if [ -n "$line" ]; then
            addr="$(awk '{print $3}' <<<"$line")"
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "server exited before READY:" >&2
            cat "$tmp" >&2
            exit 1
        fi
        sleep 0.1
    done
    [ -n "$addr" ] || {
        echo "timed out waiting for READY after ${ZF_READY_TIMEOUT_S:-180}s (raise ZF_READY_TIMEOUT_S for a larger model)" >&2
        cat "$tmp" >&2
        exit 1
    }
    base="http://$addr"
}

# discover_model_id <base> <fallback>
# GET /v1/models and take the first model id (one model per process). Sets
# `models_json` and `model_id` for the caller.
discover_model_id() {
    models_json="$(curl -s "$1/v1/models")"
    model_id="$(grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' <<<"$models_json" | head -1 | sed 's/.*"\([^"]*\)"$/\1/' || true)"
    [ -n "$model_id" ] || model_id="$2"
}
