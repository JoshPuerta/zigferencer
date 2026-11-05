#!/usr/bin/env bash
# Profile one zigferencer command with Linux perf, and render what perf saw. One cell is one
# directory that holds every file of one profiled run. A cell answers two questions: where the
# time goes, and which function calls which, in what order.
#
# The samples come from the `cpu-clock` software event, because it works on every host, also on a
# virtual machine with no hardware counters. The stacks come from frame pointers, which Zig keeps
# in every optimize mode on x86_64. A release build inlines many small functions. The fold step
# reads the DWARF inline records, so an inlined function still shows as a frame, with its file
# and line. A Debug build inlines nothing, so its stacks show every call, but its times are not
# the times of a release build.
#
# This script takes paths from the caller, so it does not change directory and it keeps `here`.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly here
# shellcheck source=tools/common.sh
source "$here/common.sh"

# -- constants ---------------------------------------------------------------

readonly freq="${FREQ:-999}"
readonly python="${PYTHON:-python3}"

# -- state -------------------------------------------------------------------

# main sets this once, after it finds a perf with libelf and DWARF support.
PERF_BIN=""

# -- helpers -----------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  tools/profile.sh all CMD [ARG...]                 build, record, report, and index one cell
  tools/profile.sh record DIR -- CMD [ARG...]       sample CMD from start to exit
  tools/profile.sh attach DIR PID SECONDS           sample a running process
  tools/profile.sh report DIR                       render the samples of DIR
  tools/profile.sh offcpu DIR PID SECONDS           record where the threads of PID wait
  tools/profile.sh diff DIR_A DIR_B OUT.svg         write a differential flame graph
  tools/profile.sh index ROOT                       write ROOT/index.html
  tools/profile.sh -h | --help                      print this text

`offcpu` needs root for the kernel trace interface. It calls sudo for each step that needs it,
and it runs nothing else as root.

Environment:
  PERF        A perf binary built with libelf and DWARF support. The default is the first such
              perf on PATH or under /usr/lib/linux-tools.
  FLAMEGRAPH  The directory of the FlameGraph scripts
              (git clone https://github.com/brendangregg/FlameGraph). The default is the
              directory of flamegraph.pl on PATH.
  FREQ        The sample rate in Hz (default 999).
  PYTHON      The python3 that runs tools/profile_convert.py (default python3).
  TRACE       Any value, such as 1, turns on the span recorder of the engine. `record` sets
              ZIGFERENCER_TRACE_OUT=DIR/trace.json for the command. `attach` copies the trace
              file that the process names in its own ZIGFERENCER_TRACE_OUT into DIR/trace.json
              after the capture. ZIGFERENCER_TRACE_GPU, ZIGFERENCER_TRACE_SECONDS, and
              ZIGFERENCER_TRACE_SPANS pass through to the command. `report` renders
              DIR/trace.json.gpu.folded into flame-gpu.svg when the file exists.
  The `all` step also reads these:
  BACKEND     The ops backend of the build (default auto).
  FLAGS       More zig build flags.
  FLAVOR      release (default), trace, or structure. The trace flavor builds with
              -Dtrace=true and sets TRACE. The structure flavor builds Debug.
  CELL        The name of the cell (default: the command, as a slug).
  PROFILE_DIR The root of the cells (default zig-out/profiles).
  ZIG         The zig binary (default zig).

Exit codes: 0 success, 1 a failed run, 2 a usage error.
EOF
}

# A perf without libelf cannot read the symbols of a Zig binary, and one without DWARF support
# cannot expand inlined frames. Both features show in `perf version --build-options`.
perf_ok() {
    local opts
    opts="$(env -u PERF_EXEC_PATH "$1" version --build-options 2>/dev/null)" || return 1
    grep -Eq 'libelf: +\[ on' <<<"$opts" && grep -Eq '^ +dwarf: +\[ on' <<<"$opts"
}

find_perf() {
    if [ -n "${PERF:-}" ]; then
        perf_ok "$PERF" || die "PERF=$PERF lacks libelf or DWARF support"
        printf '%s' "$PERF"
        return
    fi
    local candidates=() candidate
    candidates+=("$(command -v perf || true)")
    mapfile -t -O "${#candidates[@]}" candidates < <(
        printf '%s\n' /usr/lib/linux-tools/*/perf | sort -rV
    )
    for candidate in "${candidates[@]}"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ] && perf_ok "$candidate"; then
            printf '%s' "$candidate"
            return
        fi
    done
    die "no perf with libelf and DWARF support. Set PERF."
}

find_flamegraph() {
    if [ -n "${FLAMEGRAPH:-}" ]; then
        [ -x "$FLAMEGRAPH/flamegraph.pl" ] || die "FLAMEGRAPH=$FLAMEGRAPH holds no flamegraph.pl"
        printf '%s' "$FLAMEGRAPH"
        return
    fi
    local fg
    fg="$(command -v flamegraph.pl || true)"
    [ -n "$fg" ] || die "set FLAMEGRAPH to a clone of https://github.com/brendangregg/FlameGraph"
    dirname "$fg"
}

# perf reads PERF_EXEC_PATH for its helper scripts. A value left by another perf install points it
# at the wrong tree, so every call clears it.
run_perf() { env -u PERF_EXEC_PATH "$PERF_BIN" "$@"; }
sudo_perf() { sudo env -u PERF_EXEC_PATH "$PERF_BIN" "$@"; }

# Zig names a generic instance after its type arguments, with full module paths, and names an
# anonymous instance `__anon_N`. Inside the parentheses only the type names stay, so
# `Sessions(runtime.llama_fused.LlamaFusedRunner)` becomes `Sessions(LlamaFusedRunner)`. The module
# path of the frame itself stays, because it names the layer: `ops.`, `runtime.`, `serve.`,
# `core.`.
shorten_names() {
    perl -pe '
        s/__anon_\d+//g;
        1 while s{\(([^()]*)\)}{
            my $inner = $1;
            $inner =~ s/\b(?:[a-z_][a-z0-9_]*\.)+(?=[A-Za-z_])//g;
            "\x01" . $inner . "\x02"
        }e;
        tr/\x01\x02/()/;
    '
}

# A zigferencer binary carries no build-id note, so perf cannot cache a copy of it. The cell keeps
# the exact binary under symfs/ at its original path, and every report reads it through --symfs,
# so a rebuild cannot change the symbols of an old cell.
keep_binary() {
    local dir="$1" bin
    bin="$(readlink -f "$2")"
    [ -f "$bin" ] || return 0
    mkdir -p "$dir/symfs$(dirname "$bin")"
    cp "$bin" "$dir/symfs$bin"
    printf '%s\n' "$bin" >"$dir/binary.txt"
}

# -- commands ----------------------------------------------------------------

# Build the flavor, record one command, and render one cell. `make profile` lands here.
cmd_all() {
    [ $# -gt 0 ] || die_usage "all needs a command, for example: all bench-model MODEL"
    find_flamegraph >/dev/null
    local flavor="${FLAVOR:-release}" opt=ReleaseFast trace=()
    case "$flavor" in
        release) ;;
        trace)
            trace=(-Dtrace=true)
            export TRACE=1
            ;;
        structure) opt=Debug ;;
        *) die_usage "FLAVOR must be release, trace, or structure" ;;
    esac
    zig_build "$opt" "${BACKEND:-auto}" "${trace[@]}"

    local slug commit root cell
    slug="$(printf '%s' "$*" | tr -cs 'A-Za-z0-9.' '-' | cut -c1-60)"
    commit="$(git rev-parse --short HEAD 2>/dev/null || printf 'tree')"
    root="${PROFILE_DIR:-zig-out/profiles}/$commit"
    cell="$root/${CELL:-$slug}-$flavor"
    cmd_record "$cell" -- zig-out/bin/zigferencer-dev "$@"
    cmd_report "$cell"
    cmd_index "$root"
    ok "open $cell/flame.svg or $root/index.html"
}

cmd_record() {
    [ $# -ge 2 ] || die_usage "record needs DIR -- CMD [ARG...]"
    local dir="$1"
    shift
    [ "$1" = "--" ] || die_usage "record needs DIR -- CMD [ARG...]"
    shift
    [ $# -gt 0 ] || die_usage "record needs a command"
    mkdir -p "$dir"
    {
        printf '%q ' "$@"
        printf '\n'
    } >"$dir/command.txt"
    note "record $(cat "$dir/command.txt")"
    local trace_env=()
    if [ -n "${TRACE:-}" ]; then
        rm -f "$dir/trace.json" "$dir/trace.json.gpu.folded"
        trace_env=(ZIGFERENCER_TRACE_OUT="$dir/trace.json")
        # The engine log line "trace: recording to" states the ring capacity it uses.
        note "span trace to $dir/trace.json (GPU level ${ZIGFERENCER_TRACE_GPU:-off}," \
            "ring ${ZIGFERENCER_TRACE_SPANS:-at the engine default})"
    fi
    run_perf record -e cpu-clock:u -F "$freq" -g -k mono -o "$dir/perf.data" \
        -- env "${trace_env[@]}" "$@"
    keep_binary "$dir" "$(command -v "$1" || printf '%s' "$1")"
}

cmd_attach() {
    [ $# -eq 3 ] || die_usage "attach needs DIR PID SECONDS"
    local dir="$1" pid="$2" seconds="$3"
    # A dead pid or a process of another user has no readable environ.
    [ -r "/proc/$pid/environ" ] ||
        die "cannot read /proc/$pid/environ: pid $pid is not a running process of this user"
    mkdir -p "$dir"
    printf 'attach pid %s for %s s: %s\n' "$pid" "$seconds" \
        "$(tr '\0' ' ' </proc/"$pid"/cmdline)" >"$dir/command.txt"
    note "attach to $pid for $seconds s"
    # The environment of a running process is fixed, so the process must start with
    # ZIGFERENCER_TRACE_OUT, best with a capture window of at most SECONDS. The cell takes a copy
    # of the file the process writes.
    local trace_src=""
    if [ -n "${TRACE:-}" ]; then
        trace_src="$(tr '\0' '\n' </proc/"$pid"/environ |
            sed -n 's/^ZIGFERENCER_TRACE_OUT=//p')"
        [ -n "$trace_src" ] ||
            note "pid $pid has no ZIGFERENCER_TRACE_OUT, so the cell gets no trace.json"
    fi
    run_perf record -e cpu-clock:u -F "$freq" -g -k mono -o "$dir/perf.data" \
        -p "$pid" -- sleep "$seconds"
    keep_binary "$dir" "$(readlink -f /proc/"$pid"/exe)"
    if [ -n "$trace_src" ]; then
        case "$trace_src" in
            /*) ;;
            *) trace_src="$(readlink -f /proc/"$pid"/cwd)/$trace_src" ;;
        esac
        if [ -f "$trace_src" ]; then
            cp "$trace_src" "$dir/trace.json"
            if [ -f "$trace_src.gpu.folded" ]; then
                cp "$trace_src.gpu.folded" "$dir/trace.json.gpu.folded"
            fi
            note "copied $trace_src into the cell"
        else
            note "$trace_src does not exist yet. Set ZIGFERENCER_TRACE_SECONDS to at most $seconds"
        fi
    fi
}

cmd_report() {
    [ $# -eq 1 ] || die_usage "report needs DIR"
    local dir="$1"
    [ -f "$dir/perf.data" ] || die "$dir holds no perf.data"
    [ -f "$dir/binary.txt" ] || die "$dir holds no binary.txt. Record the cell with this script."
    local fg title bin sym=()
    fg="$(find_flamegraph)"
    title="$(head -c 120 "$dir/command.txt" 2>/dev/null || basename "$dir")"
    bin="$(cat "$dir/binary.txt")"
    [ ! -d "$dir/symfs" ] || sym=(--symfs="$dir/symfs")

    # perf 6.8 `script --inline` adds no inlined frame, so the fold step reads the DWARF inline
    # records itself.
    note "folded stacks with inlined frames"
    run_perf script -i "$dir/perf.data" "${sym[@]}" \
        -F comm,pid,tid,time,period,event,ip,sym,dso --ns 2>/dev/null >"$dir/samples.raw"
    "$python" "$here/profile_convert.py" fold \
        "$dir/samples.raw" "$bin" "$dir/symfs$bin" "$dir/stacks.raw.folded"
    shorten_names <"$dir/stacks.raw.folded" >"$dir/stacks.folded"
    rm -f "$dir/samples.raw" "$dir/stacks.raw.folded"
    "$fg/flamegraph.pl" --title "$title" --subtitle "CPU samples, inlined frames shown" \
        --countname ns "$dir/stacks.folded" >"$dir/flame.svg"

    note "caller and callee trees"
    "$python" "$here/profile_convert.py" tree \
        "$dir/stacks.folded" "$dir/callees.txt" "$dir/callers.txt"

    note "hot source lines"
    # grep exits 1 on a report with no line left, and pipefail then stops the script. An empty
    # lines.txt is a valid result.
    run_perf report -i "$dir/perf.data" "${sym[@]}" --stdio --no-children \
        --sort srcline,sym -g none --percent-limit 0.3 2>/dev/null |
        { grep -v -e '^#' -e '^$' || true; } |
        shorten_names >"$dir/lines.txt"

    note "sampled timeline"
    run_perf script -i "$dir/perf.data" "${sym[@]}" -F +pid >"$dir/samples.perf" 2>/dev/null

    if [ -s "$dir/trace.json.gpu.folded" ]; then
        note "GPU phases"
        "$fg/flamegraph.pl" --title "$title" --subtitle "GPU time by runner, label, and phase" \
            --countname ns --colors aqua "$dir/trace.json.gpu.folded" >"$dir/flame-gpu.svg"
    fi
    note "done: $dir/flame.svg"
}

cmd_offcpu() {
    [ $# -eq 3 ] || die_usage "offcpu needs DIR PID SECONDS"
    local dir="$1" pid="$2" seconds="$3" fg
    mkdir -p "$dir"
    fg="$(find_flamegraph)"
    note "scheduler switches of every CPU for $seconds s"
    sudo_perf record -e sched:sched_switch -a -g -k mono -o "$dir/offcpu.data" -- sleep "$seconds"
    sudo chown "$(id -u):$(id -g)" "$dir/offcpu.data"
    # The decode runs as root, so the kernel frames of each wait resolve to names. The `-f` flag
    # is needed because the data file belongs to the user by now, and perf refuses a file that
    # the reading user does not own.
    sudo_perf script -f -i "$dir/offcpu.data" \
        -F comm,pid,tid,time,event,trace,ip,sym --ns >"$dir/offcpu.raw"
    "$python" "$here/profile_convert.py" offcpu "$dir/offcpu.raw" "$pid" "$dir/offcpu.folded"
    shorten_names <"$dir/offcpu.folded" >"$dir/offcpu.short.folded"
    if [ -s "$dir/offcpu.short.folded" ]; then
        "$fg/flamegraph.pl" --title "off-CPU time, pid $pid" --countname us --colors io \
            "$dir/offcpu.short.folded" >"$dir/flame-offcpu.svg"
    else
        note "no switch of pid $pid in the capture, so no off-CPU flame graph"
    fi
    note "wait and syscall summaries for $seconds s"
    {
        printf '# perf sched timehist summary, pid %s\n' "$pid"
        sudo_perf sched record -k mono -o "$dir/sched.data" -p "$pid" -- sleep "$seconds" \
            >/dev/null 2>&1 || true
        sudo_perf sched timehist -f -i "$dir/sched.data" -s -p "$pid" 2>/dev/null || true
        printf '\n# perf trace syscall summary, pid %s\n' "$pid"
        sudo_perf trace -s -p "$pid" -- sleep "$seconds" 2>&1 || true
    } >"$dir/sched.txt"
    sudo rm -f "$dir/sched.data"
    note "done: $dir/flame-offcpu.svg, $dir/sched.txt"
}

cmd_diff() {
    [ $# -eq 3 ] || die_usage "diff needs DIR_A DIR_B OUT.svg"
    local a="$1" b="$2" out="$3" fg
    fg="$(find_flamegraph)"
    "$fg/difffolded.pl" -n "$a/stacks.folded" "$b/stacks.folded" |
        "$fg/flamegraph.pl" --title "$(basename "$b") against $(basename "$a")" >"$out"
}

# One static page per profile root: a row per cell, a link per file, and how to open each kind of
# file. It holds no script.
cmd_index() {
    [ $# -eq 1 ] || die_usage "index needs ROOT"
    local root="$1" cell f
    [ -d "$root" ] || die "$root is not a directory"
    {
        cat <<EOF
<!doctype html><meta charset="utf-8"><title>profiles $(basename "$root")</title>
<style>body{font:14px sans-serif;margin:16px}td,th{padding:4px 8px;text-align:left}</style>
<h1>profiles $(basename "$root")</h1>
<p>SVG: open in a browser, click a frame to zoom, Ctrl-F to search.
trace.json: ui.perfetto.dev, Open trace file.
samples.perf: profiler.firefox.com or speedscope.app. Text files: any editor.</p>
<table><tr><th>cell</th><th>files</th></tr>
EOF
        for cell in "$root"/*/; do
            [ -d "$cell" ] || continue
            printf '<tr><td>%s</td><td>' "$(basename "$cell")"
            for f in "$cell"*.svg "$cell"*.json "$cell"*.txt "$cell"*.md \
                "$cell"samples.perf "$cell"structure/*.svg; do
                [ ! -f "$f" ] || printf '<a href="%s">%s</a> ' "${f#"$root"/}" "${f#"$cell"}"
            done
            printf '</td></tr>\n'
        done
        printf '</table>\n'
    } >"$root/index.html"
    note "wrote $root/index.html"
}

# -- main --------------------------------------------------------------------

main() {
    local sub="${1:-}"
    case "$sub" in
        -h | --help)
            usage
            return 0
            ;;
        "") die_usage "expected a command: all, record, attach, report, offcpu, diff, or index" ;;
    esac
    shift
    case "$sub" in
        diff)
            cmd_diff "$@"
            return
            ;;
        index)
            cmd_index "$@"
            return
            ;;
    esac
    PERF_BIN="$(find_perf)"
    readonly PERF_BIN
    case "$sub" in
        all) cmd_all "$@" ;;
        record) cmd_record "$@" ;;
        attach) cmd_attach "$@" ;;
        report) cmd_report "$@" ;;
        offcpu) cmd_offcpu "$@" ;;
        *) die_usage "unknown command '$sub'" ;;
    esac
}

main "$@"
