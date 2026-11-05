#!/usr/bin/env bash
# Profile one zigferencer command with Linux perf, and render what perf saw.
# One cell is one directory that holds every file of one profiled run. A cell
# answers two questions: where the time goes, and which function calls which,
# in what order.
#
# The samples come from the `cpu-clock` software event, because it works on
# every host, also on a virtual machine with no hardware counters. The stacks
# come from frame pointers, which Zig keeps in every optimize mode on x86_64.
# A release build inlines many small functions. The `--inline` reports read the
# DWARF inline records, so an inlined function still shows as a frame, with its
# file and line. A Debug build inlines nothing, so its stacks show every call,
# but its times are not the times of a release build.
#
# Usage:
#   tools/profile.sh record DIR -- CMD [ARG...]       sample CMD from start to exit
#   tools/profile.sh attach DIR PID SECONDS           sample a running process
#   tools/profile.sh report DIR                       render the samples of DIR
#   tools/profile.sh calls DIR LIST -- CMD [ARG...]   trace every call of the
#                                                     functions that LIST names
#   tools/profile.sh offcpu DIR PID SECONDS           record where the threads of
#                                                     PID wait, and for how long
#   tools/profile.sh diff DIR_A DIR_B OUT.svg         a differential flame graph
#   tools/profile.sh index ROOT                       write ROOT/index.html
#
# `calls` and `offcpu` need root for the kernel trace interface. They call
# sudo for each step that needs it, and they run CMD as the calling user.
# LIST is a text file with one extended regular expression per line, matched
# against the full symbol name without its `__anon_N` suffix. A line that
# starts with `#` is a comment.
#
# Env:
#   PERF        a perf binary built with libelf and DWARF support. The default
#               is the first such perf on PATH or under /usr/lib/linux-tools.
#   FLAMEGRAPH  the directory of the FlameGraph scripts
#               (git clone https://github.com/brendangregg/FlameGraph).
#               The default is the directory of flamegraph.pl on PATH.
#   FREQ        the sample rate in Hz (default 999).
#   PYTHON      the python3 that runs tools/profile_convert.py (default python3).
#   TRACE       any value, such as 1, turns on the span recorder of the
#               engine. `record` sets ZIGFERENCER_TRACE_OUT=DIR/trace.json for
#               the command. `attach` copies the trace file that the process
#               names in its own ZIGFERENCER_TRACE_OUT into DIR/trace.json
#               after the capture. ZIGFERENCER_TRACE_GPU,
#               ZIGFERENCER_TRACE_SECONDS, and ZIGFERENCER_TRACE_SPANS pass
#               through to the command. `report` renders
#               DIR/trace.json.gpu.folded into
#               flame-gpu.svg when the file exists.

set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*" >&2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly here
readonly freq="${FREQ:-999}"
readonly probe_group=zf
readonly python="${PYTHON:-python3}"

# A perf without libelf cannot read the symbols of a Zig binary, and one
# without DWARF support cannot expand inlined frames. Both features show in
# `perf version --build-options`.
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
    local candidate
    for candidate in "$(command -v perf || true)" $(ls -d /usr/lib/linux-tools/*/perf 2>/dev/null | sort -rV); do
        [ -n "$candidate" ] && [ -x "$candidate" ] && perf_ok "$candidate" && { printf '%s' "$candidate"; return; }
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

# perf reads PERF_EXEC_PATH for its helper scripts. A value left by another
# perf install points it at the wrong tree, so every call clears it.
run_perf() { env -u PERF_EXEC_PATH "$PERF_BIN" "$@"; }
sudo_perf() { sudo env -u PERF_EXEC_PATH "$PERF_BIN" "$@"; }

# Zig names a generic instance after its type arguments, with full module
# paths, and names an anonymous instance `__anon_N`. Inside the parentheses
# only the type names stay, so `Sessions(runtime.llama_fused.LlamaFusedRunner)`
# becomes `Sessions(LlamaFusedRunner)`. The module path of the frame itself
# stays, because it names the layer: `ops.`, `runtime.`, `serve.`, `core.`.
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

# A zigferencer binary carries no build-id note, so perf cannot cache a copy
# of it. The cell keeps the exact binary under symfs/ at its original path,
# and every report reads it through --symfs, so a rebuild cannot change the
# symbols of an old cell.
keep_binary() {
    local dir="$1" bin
    bin="$(readlink -f "$2")"
    [ -f "$bin" ] || return 0
    mkdir -p "$dir/symfs$(dirname "$bin")"
    cp "$bin" "$dir/symfs$bin"
    printf '%s\n' "$bin" >"$dir/binary.txt"
}

symfs_args() {
    local dir="$1"
    [ -d "$dir/symfs" ] && printf -- '--symfs=%s' "$dir/symfs"
    return 0
}

cmd_record() {
    local dir="$1"; shift
    [ "${1:-}" = "--" ] || die "usage: record DIR -- CMD [ARG...]"
    shift
    [ $# -gt 0 ] || die "record needs a command"
    mkdir -p "$dir"
    printf '%q ' "$@" >"$dir/command.txt"; printf '\n' >>"$dir/command.txt"
    note "record $(cat "$dir/command.txt")"
    local trace_env=()
    if [ -n "${TRACE:-}" ]; then
        rm -f "$dir/trace.json" "$dir/trace.json.gpu.folded"
        trace_env=(ZIGFERENCER_TRACE_OUT="$dir/trace.json")
        # The engine log line "trace: recording to" states the ring capacity it uses.
        note "span trace to $dir/trace.json (GPU level ${ZIGFERENCER_TRACE_GPU:-off}, ring ${ZIGFERENCER_TRACE_SPANS:-at the engine default})"
    fi
    run_perf record -e cpu-clock:u -F "$freq" -g -k mono -o "$dir/perf.data" -- env "${trace_env[@]}" "$@"
    keep_binary "$dir" "$(command -v "$1" || printf '%s' "$1")"
}

cmd_attach() {
    [ $# -eq 3 ] || die "usage: attach DIR PID SECONDS"
    local dir="$1" pid="$2" seconds="$3"
    # A dead pid or a process of another user has no readable environ.
    [ -r "/proc/$pid/environ" ] || die "cannot read /proc/$pid/environ: pid $pid is not a running process of this user"
    mkdir -p "$dir"
    printf 'attach pid %s for %s s: %s\n' "$pid" "$seconds" "$(tr '\0' ' ' </proc/"$pid"/cmdline)" >"$dir/command.txt"
    note "attach to $pid for $seconds s"
    # The environment of a running process is fixed, so the process must
    # start with ZIGFERENCER_TRACE_OUT, best with a capture window of at most
    # SECONDS. The cell takes a copy of the file the process writes.
    local trace_src=""
    if [ -n "${TRACE:-}" ]; then
        trace_src="$(tr '\0' '\n' </proc/"$pid"/environ | sed -n 's/^ZIGFERENCER_TRACE_OUT=//p')"
        [ -n "$trace_src" ] || note "pid $pid has no ZIGFERENCER_TRACE_OUT, so the cell gets no trace.json"
    fi
    run_perf record -e cpu-clock:u -F "$freq" -g -k mono -o "$dir/perf.data" -p "$pid" -- sleep "$seconds"
    keep_binary "$dir" "$(readlink -f /proc/"$pid"/exe)"
    if [ -n "$trace_src" ]; then
        case "$trace_src" in /*) ;; *) trace_src="$(readlink -f /proc/"$pid"/cwd)/$trace_src" ;; esac
        if [ -f "$trace_src" ]; then
            cp "$trace_src" "$dir/trace.json"
            [ -f "$trace_src.gpu.folded" ] && cp "$trace_src.gpu.folded" "$dir/trace.json.gpu.folded"
            note "copied $trace_src into the cell"
        else
            note "$trace_src does not exist yet. Set ZIGFERENCER_TRACE_SECONDS to at most $seconds"
        fi
    fi
}

cmd_report() {
    local dir="$1"
    [ -f "$dir/perf.data" ] || die "$dir holds no perf.data"
    local fg
    fg="$(find_flamegraph)"
    local title
    title="$(head -c 120 "$dir/command.txt" 2>/dev/null || basename "$dir")"
    # perf 6.8 `script --inline` adds no inlined frame, so the fold step reads
    # the DWARF inline records itself.
    note "folded stacks with inlined frames"
    [ -f "$dir/binary.txt" ] || die "$dir holds no binary.txt. Record the cell with this script."
    local bin sym
    bin="$(cat "$dir/binary.txt")"
    sym="$(symfs_args "$dir")"
    run_perf script -i "$dir/perf.data" $sym -F comm,pid,tid,time,period,event,ip,sym,dso --ns 2>/dev/null >"$dir/samples.raw"
    "$python" "$here/profile_convert.py" fold "$dir/samples.raw" "$bin" "$dir/symfs$bin" "$dir/stacks.raw.folded"
    shorten_names <"$dir/stacks.raw.folded" >"$dir/stacks.folded"
    rm -f "$dir/samples.raw" "$dir/stacks.raw.folded"
    "$fg/flamegraph.pl" --title "$title" --subtitle "CPU samples, inlined frames shown" --countname ns \
        "$dir/stacks.folded" >"$dir/flame.svg"
    note "caller and callee trees"
    "$python" "$here/profile_convert.py" tree "$dir/stacks.folded" "$dir/callees.txt" "$dir/callers.txt"
    note "hot source lines"
    # grep exits 1 on a report with no line left, and pipefail then stops the
    # script. An empty lines.txt is a valid result.
    run_perf report -i "$dir/perf.data" $sym --stdio --no-children --sort srcline,sym -g none --percent-limit 0.3 \
        2>/dev/null | { grep -v -e '^#' -e '^$' || true; } | shorten_names >"$dir/lines.txt"
    note "sampled timeline"
    run_perf script -i "$dir/perf.data" $sym -F +pid >"$dir/samples.perf" 2>/dev/null
    if [ -s "$dir/trace.json.gpu.folded" ]; then
        note "GPU phases"
        "$fg/flamegraph.pl" --title "$title" --subtitle "GPU time by runner, label, and phase" --countname ns \
            --colors aqua "$dir/trace.json.gpu.folded" >"$dir/flame-gpu.svg"
    fi
    note "done: $dir/flame.svg"
}

# Resolve each LIST pattern to text symbols of BIN, and print one line per
# symbol: the file offset, a short event name, and the full name. A uprobe
# takes a file offset, which is the address minus the start of its load
# segment plus the file offset of that segment.
resolve_symbols() {
    local bin="$1" list="$2"
    # strtonum is a gawk extension. mawk, the default awk of some
    # distributions, lacks it.
    command -v gawk >/dev/null || die "calls needs gawk for strtonum. Install gawk"
    local segments
    segments="$(readelf -lW "$bin" | awk '$1 == "LOAD" && ($7 ~ /E/ || $8 == "E") { print $2, $3 }')"
    [ -n "$segments" ] || die "$bin has no executable load segment"
    local patterns
    patterns="$(grep -Ev '^[[:space:]]*(#|$)' "$list")" || die "$list names no function"
    nm --defined-only "$bin" | sed -E 's/__anon_[0-9]+//g' \
        | grep -E -f <(printf '%s\n' "$patterns" | sed -E 's/^\^?/^[0-9a-f]+ [tT] /') \
        | gawk -v segs="$segments" '
            BEGIN {
                n = split(segs, f, /[ \n]/)
                for (i = 1; i + 1 <= n; i += 2) { off[++m] = strtonum(f[i]); vaddr[m] = strtonum(f[i + 1]) }
            }
            {
                addr = strtonum("0x" $1)
                name = $0; sub(/^[^ ]+ [tT] /, "", name)
                for (j = m; j >= 1; j--) if (addr >= vaddr[j]) break
                printf "0x%x f%d %s\n", addr - vaddr[j] + off[j], NR - 1, name
            }'
}

cmd_calls() {
    local dir="$1" list="$2"; shift 2
    [ "${1:-}" = "--" ] || die "usage: calls DIR LIST -- CMD [ARG...]"
    shift
    [ $# -gt 0 ] || die "calls needs a command"
    local bin
    bin="$(readlink -f "$(command -v "$1" || printf '%s' "$1")")"
    [ -f "$bin" ] || die "cannot find the binary of $1"
    mkdir -p "$dir"
    printf '%q ' "$@" >"$dir/command.txt"; printf '\n' >>"$dir/command.txt"
    resolve_symbols "$bin" "$list" >"$dir/probes.txt"
    local count
    count="$(wc -l <"$dir/probes.txt")"
    [ "$count" -gt 0 ] || die "no symbol of $bin matches $list"
    note "$count functions, $((count * 2)) probes"

    local tracefs=/sys/kernel/tracing
    sudo test -d "$tracefs" || tracefs=/sys/kernel/debug/tracing
    # Global, because the EXIT trap runs after this function returns.
    UPROBE_EVENTS="$tracefs/uprobe_events"
    local off ev _name
    : >"$dir/probes.defs"
    while read -r off ev _name; do
        printf 'p:%s/%s %s:%s\nr:%s/%s_ret %s:%s\n' "$probe_group" "$ev" "$bin" "$off" "$probe_group" "$ev" "$bin" "$off" \
            >>"$dir/probes.defs"
    done <"$dir/probes.txt"
    # The kernel runs a multi-line write line by line and stops at the first
    # refused line, so each definition and each removal is its own write. An
    # earlier run that stopped half way can leave events of the group behind.
    clear_probes() {
        local u
        for u in $(sudo grep -oE "^[pr]:$probe_group/[A-Za-z0-9_]+" "$UPROBE_EVENTS" | sed 's/^[pr]:/-:/'); do
            printf '%s\n' "$u" | sudo tee -a "$UPROBE_EVENTS" >/dev/null 2>&1 || true
        done
    }
    clear_probes
    trap clear_probes EXIT
    local line
    while IFS= read -r line; do
        if ! printf '%s\n' "$line" | sudo tee -a "$UPROBE_EVENTS" >/dev/null; then
            sudo tail -n 3 "$tracefs/error_log" >&2 || true
            die "the kernel refused the uprobe definition: $line"
        fi
    done <"$dir/probes.defs"

    note "record $(cat "$dir/command.txt")"
    sudo -E env -u PERF_EXEC_PATH "$PERF_BIN" record -e "$probe_group:*" -g -k mono -o "$dir/calls.data" \
        -- sudo -E -u "$(id -un)" "$@"
    sudo chown "$(id -u):$(id -g)" "$dir/calls.data"
    run_perf script -i "$dir/calls.data" -F comm,tid,time,event,ip,sym --ns 2>/dev/null >"$dir/calls.raw"
    "$python" "$here/profile_convert.py" calls "$dir/calls.raw" "$dir/probes.txt" "$dir"
    note "done: $dir/calls.txt, $dir/calls.json, $dir/counts.txt"
}

cmd_offcpu() {
    local dir="$1" pid="$2" seconds="$3"
    mkdir -p "$dir"
    local fg
    fg="$(find_flamegraph)"
    note "scheduler switches of every CPU for $seconds s"
    sudo_perf record -e sched:sched_switch -a -g -k mono -o "$dir/offcpu.data" -- sleep "$seconds"
    sudo chown "$(id -u):$(id -g)" "$dir/offcpu.data"
    # The decode runs as root, so the kernel frames of each wait resolve to names.
    # `-f`: the data file belongs to the user by now, and perf refuses a file
    # that the reading user does not own.
    sudo_perf script -f -i "$dir/offcpu.data" -F comm,pid,tid,time,event,trace,ip,sym --ns >"$dir/offcpu.raw"
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
        sudo_perf sched record -k mono -o "$dir/sched.data" -p "$pid" -- sleep "$seconds" >/dev/null 2>&1 || true
        sudo_perf sched timehist -f -i "$dir/sched.data" -s -p "$pid" 2>/dev/null || true
        printf '\n# perf trace syscall summary, pid %s\n' "$pid"
        sudo_perf trace -s -p "$pid" -- sleep "$seconds" 2>&1 || true
    } >"$dir/sched.txt"
    sudo rm -f "$dir/sched.data"
    note "done: $dir/flame-offcpu.svg, $dir/sched.txt"
}

cmd_diff() {
    local a="$1" b="$2" out="$3"
    local fg
    fg="$(find_flamegraph)"
    "$fg/difffolded.pl" -n "$a/stacks.folded" "$b/stacks.folded" \
        | "$fg/flamegraph.pl" --title "$(basename "$b") against $(basename "$a")" >"$out"
}

# One static page per profile root: a row per cell, a link per file, and how
# to open each kind of file. It holds no script.
cmd_index() {
    local root="$1"
    [ -d "$root" ] || die "$root is not a directory"
    {
        printf '<!doctype html><meta charset="utf-8"><title>profiles %s</title>\n' "$(basename "$root")"
        printf '<style>body{font:14px sans-serif;margin:16px}td,th{padding:4px 8px;text-align:left;vertical-align:top}</style>\n'
        printf '<h1>profiles %s</h1>\n' "$(basename "$root")"
        printf '<p>SVG: open in a browser, click a frame to zoom, Ctrl-F to search. '
        printf 'trace.json and calls.json: ui.perfetto.dev, Open trace file. '
        printf 'samples.perf: profiler.firefox.com or speedscope.app. Text files: any editor.</p>\n'
        printf '<table><tr><th>cell</th><th>files</th></tr>\n'
        local cell f
        for cell in "$root"/*/; do
            [ -d "$cell" ] || continue
            printf '<tr><td>%s</td><td>' "$(basename "$cell")"
            for f in "$cell"*.svg "$cell"*.json "$cell"*.txt "$cell"*.md "$cell"samples.perf "$cell"structure/*.svg; do
                [ -f "$f" ] && printf '<a href="%s">%s</a> ' "${f#"$root"/}" "${f#"$cell"}"
            done
            printf '</td></tr>\n'
        done
        printf '</table>\n'
    } >"$root/index.html"
    note "wrote $root/index.html"
}

main() {
    local sub="${1:-}"
    [ -n "$sub" ] || die "usage: tools/profile.sh record|attach|report|calls|offcpu|diff|index ..."
    shift
    case "$sub" in
        diff) cmd_diff "$@"; return ;;
        index) cmd_index "$@"; return ;;
    esac
    PERF_BIN="$(find_perf)"
    readonly PERF_BIN
    case "$sub" in
        record) cmd_record "$@" ;;
        attach) cmd_attach "$@" ;;
        report) cmd_report "$@" ;;
        calls) cmd_calls "$@" ;;
        offcpu) cmd_offcpu "$@" ;;
        *) die "unknown command '$sub'" ;;
    esac
}

main "$@"
