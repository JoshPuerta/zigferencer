#!/usr/bin/env bash
# Shared interactive UX for the Makefile's convenience targets. This file is
# sourced, never executed:
#
#   source tools/menu.sh
#
# It then provides the prompt and menu functions that the run-style targets use
# to resolve a variable, the status-line functions that every target uses for output,
# and backend_flags, which maps a backend name to its `zig build` flags. One
# place owns the look and the resolution rules, so `make run`, `make chat`, and
# `make bench-model` behave the same and read the same.
#
# The core markers ([+] [-] [*] [>] [?]) match the output of zigferencer itself.
# `info` and the interactive pickers emit the same set. [!] (error) and
# [$] (the echoed command, see run_cmd below) belong to the Makefile alone,
# because the Zig side reports an error as plain `error:` text and never echoes a
# command. The palette collapses to plain text when stderr is not a terminal or
# $NO_COLOR is set. That is the no-color.org convention: the variable disables
# color at any value.
#
#   [+] green   present / on / the default option   (shared with zigferencer)
#   [-] dim     absent / off                        (shared)
#   [*] dim     neutral note                        (shared)
#   [>] bold    a prompt or an action about to run  (shared)
#   [?] bold    a yes/no confirmation               (shared)
#   [!] red     an error                            (Makefile-only)
#
# The resolution rule fixes the order for every variable:
#
#   1. value already set (env var / `make VAR=...`)  -> used as-is, no prompt
#   2. stdin is not a TTY (a pipe, CI)               -> the default, silently
#   3. interactive                                    -> prompt. Enter takes default
#
# Prompts and menus go to stderr. The resolved value is the stdout of the
# function, so a caller captures it with command substitution:
#
#   backend=$(prompt_choice "$BACKEND" "op backend" auto auto naive simd_pf)
#   prompt=$(prompt_value "$PROMPT" "prompt" "Hello")

# -- palette -----------------------------------------------------------------
if [ -t 2 ] && [ -z "${NO_COLOR+x}" ]; then
    c_reset=$'\033[0m'; c_bold=$'\033[1m'; c_dim=$'\033[2m'
    c_green=$'\033[32m'; c_red=$'\033[31m'
else
    c_reset=''; c_bold=''; c_dim=''; c_green=''; c_red=''
fi

# -- trace level -------------------------------------------------------------
# LOG, a make variable, maps to $ZIGFERENCER_LOG, which every zigferencer process
# reads once at startup. `make run LOG=runner,kv` or `make inspect LOG=gguf`
# therefore switches traces at run time, with no rebuild. A given LOG overrides a
# $ZIGFERENCER_LOG from the environment. With no LOG, that value stays.
[ -n "${LOG:-}" ] && export ZIGFERENCER_LOG="$LOG"

# -- status lines (stderr. stdout stays reserved for program output) ---------
say() { printf '%s[*]%s %s\n' "$c_dim"   "$c_reset" "$*" >&2; }
ok()  { printf '%s[+]%s %s\n' "$c_green" "$c_reset" "$*" >&2; }
err() { printf '%s[!]%s %s\n' "$c_red"   "$c_reset" "$*" >&2; }

# run_cmd <argv...>
# Echo the command behind a dim [$], shell-quoted so it is ready to paste, then
# run it. The echo and the run share one argv, so the printed line is exactly
# what ran, and `make run` doubles as a lesson in the zigferencer CLI it drives.
# The echo goes to stderr, so a captured stdout (`make run > out`) still holds
# program output alone.
run_cmd() {
    local q=() a IFS=' '
    for a in "$@"; do q+=("$(printf '%q' "$a")"); done
    printf '%s[$]%s %s\n' "$c_dim" "$c_reset" "${q[*]}" >&2
    "$@"
}

# -- prompts -----------------------------------------------------------------
# prompt_value <current> <label> <default>
# Resolve a free-text variable by the resolution rule above. Echo the result on
# stdout.
prompt_value() {
    local current="$1" label="$2" default="$3" reply
    if [ -n "$current" ]; then printf '%s' "$current"; return; fi
    if [ ! -t 0 ]; then printf '%s' "$default"; return; fi
    if [ -n "$default" ]; then
        printf '%s[>]%s %s %s[%s]%s: ' "$c_bold" "$c_reset" "$label" "$c_dim" "$default" "$c_reset" >&2
    else
        printf '%s[>]%s %s %s(blank = none)%s: ' "$c_bold" "$c_reset" "$label" "$c_dim" "$c_reset" >&2
    fi
    IFS= read -r reply || reply=''
    [ -n "$reply" ] || reply="$default"
    printf '%s' "$reply"
}

# prompt_choice <current> <label> <default> <option>...
# Resolve an enumerated variable. The numbered menu marks the default and accepts a
# 1-based number or an exact option name. Enter takes the default.
prompt_choice() {
    local current="$1" label="$2" default="$3"; shift 3
    local opts=("$@") reply i n=$#
    if [ -n "$current" ]; then printf '%s' "$current"; return; fi
    if [ ! -t 0 ]; then printf '%s' "$default"; return; fi
    printf '%s%s%s %s(Enter = %s)%s\n' "$c_bold" "$label" "$c_reset" "$c_dim" "$default" "$c_reset" >&2
    for i in "${!opts[@]}"; do
        local o="${opts[$i]}" mark mc
        if [ "$o" = "$default" ]; then mark='[+]'; mc="$c_green"; else mark='[-]'; mc="$c_dim"; fi
        printf '  %s%s%s %s%2d%s  %s\n' "$mc" "$mark" "$c_reset" "$c_bold" "$((i + 1))" "$c_reset" "$o" >&2
    done
    printf '%s[>]%s choose [1-%d or name]: ' "$c_bold" "$c_reset" "$n" >&2
    IFS= read -r reply || reply=''
    [ -n "$reply" ] || { printf '%s' "$default"; return; }
    if [[ "$reply" =~ ^[0-9]+$ ]] && [ "$reply" -ge 1 ] && [ "$reply" -le "$n" ]; then
        printf '%s' "${opts[$((reply - 1))]}"
    else
        printf '%s' "$reply"
    fi
}

# prompt_backend <current>
# The op-backend menu in one place, so every build-capable target offers the same
# set with the same default. 'auto' lets build.zig pick the strongest eligible
# CPU variant.
prompt_backend() {
    prompt_choice "$1" "op backend" auto auto naive simd simd_pf vulkan vulkan_fused
}

# prompt_model <current> <bin>
# Resolve the model variable against the models present on disk (<bin> list
# --present). A run-style target does not need this menu, because its binary
# prompts by itself. A script-driven target such as bench-openai or demo-openai
# spawns the server with its output captured, so the picker of the binary cannot
# appear, and the server refuses to start without a model name. This menu is the
# replacement, and it always resolves a concrete name. Enter, or a stdin that is
# a pipe, takes the first present model.
prompt_model() {
    local current="$1" bin="$2" models=()
    if [ -n "$current" ]; then printf '%s' "$current"; return; fi
    mapfile -t models < <("$bin" list --present)
    if [ "${#models[@]}" -eq 0 ]; then err "no models on disk. 'make fetch' one first"; return 1; fi
    prompt_choice '' 'model' "${models[0]}" "${models[@]}"
}

# confirm <question>
# A y/N gate, with no as the default. A non-interactive stdin answers no. A
# caller pairs the gate with a FORCE override, so a pipe opts in on purpose
# rather than blocking. It returns 0 for yes and 1 for no, so it reads as
# `if confirm "..."; then` or `! confirm`.
confirm() {
    local reply
    [ -t 0 ] || return 1
    printf '%s[?]%s %s %s[y/N]%s ' "$c_bold" "$c_reset" "$*" "$c_dim" "$c_reset" >&2
    IFS= read -r reply || return 1
    [[ "$reply" == [yY]* ]]
}

# -- build flags -------------------------------------------------------------
# backend_flags <backend>
# The `zig build` flags that a backend name implies. 'auto', and a blank value,
# let build.zig pick the strongest eligible CPU variant. The two vulkan backends
# also switch the runtime to the GPU, and they pass ATTN, the attention kernel set, when
# it is set. The echo is space-separated, for unquoted word-splitting.
backend_flags() {
    case "$1" in
        vulkan | vulkan_fused) printf -- '-Dops-backend=%s -Druntime=vulkan%s' "$1" "${ATTN:+ -Dattn-kernel=$ATTN}" ;;
        '' | auto)             printf -- '-Dops-backend=auto' ;;
        *)                     printf -- '-Dops-backend=%s' "$1" ;;
    esac
}

# ensure_bin <bin> <zig>
# Reuse <bin> when it exists, because the binary of any backend serves the
# read-only commands. Build the CPU default only when no binary is there. That is
# how a GPU build survives `make list`. A set BACKEND gets a note rather than a
# silent drop, because these targets never run a kernel and BACKEND cannot
# matter here.
ensure_bin() {
    [ -n "${BACKEND:-}" ] && say "BACKEND is ignored here (any binary serves this command)"
    [ -x "$1" ] && return
    say "no binary present, building the CPU default"
    run_cmd "$2" build install dev
}

# -- shared target bodies ----------------------------------------------------
# zig_build <zig> <opt> <backend>
# The one release-build invocation the run-style targets share. It installs the
# release and the developer binary in one build, so zig-out/bin never holds a
# pair from two builds.
zig_build() {
    run_cmd "$1" build install dev -Doptimize="$2" ${TRACE:+-Dtrace=true} $(backend_flags "$3")
}

# run_infer <zig> <opt> <bin> [extra argv...]
# The shared body of `make run` and `make chat`. It prompts for the backend, the
# prompt text, the token count, and the threads, then builds, then runs. The
# extra argv appends to the generated command, such as `--chat` for chat. MODEL
# and CACHE pass through when they are set.
run_infer() {
    local zig="$1" opt="$2" bin="$3"
    shift 3
    local backend prompt n threads
    backend=$(prompt_backend "${BACKEND:-}")
    prompt=$(prompt_value "${PROMPT:-}" "prompt" "Hello")
    n=$(prompt_value "${N:-}" "max tokens" 32)
    threads=$(prompt_value "${THREADS:-}" "pool threads, 0 = auto" 0)
    zig_build "$zig" "$opt" "$backend"
    local cmd=("$bin" run)
    [ -n "${MODEL:-}" ] && cmd+=("$MODEL")
    cmd+=(--prompt "$prompt" --max-tokens "$n")
    [ -n "$threads" ] && cmd+=(--threads "$threads")
    [ -n "${CACHE:-}" ] && cmd+=(--prompt-cache)
    [ -n "${GPU_LAYERS:-}" ] && cmd+=(--gpu-layers "$GPU_LAYERS")
    [ -n "${KV_DTYPE:-}" ] && cmd+=(--kv-dtype "$KV_DTYPE")
    cmd+=("$@")
    run_cmd "${cmd[@]}"
}

# simd_report <zig> <probe> <probe_flags> [mode argv...]
# The shared body of the simd-* targets. It builds the probe assembly, then runs
# the report script in the given mode.
simd_report() {
    local zig="$1" probe="$2" flags="$3"
    shift 3
    run_cmd "$zig" build simd-probe $flags
    run_cmd tools/simd-report.sh "$@" "$probe"
}

# bin_subcommand <bin> <subcommand...>
# Run a command on the CURRENT binary, and refuse when no binary exists. It is
# the shared body of the target that reports on the built binary (info).
bin_subcommand() {
    local bin="$1"
    shift
    if [ ! -x "$bin" ]; then
        err "no binary at $bin. Build one first: 'make build' (CPU) or 'make gpu' (vulkan)"
        exit 1
    fi
    run_cmd "$bin" "$@"
}
