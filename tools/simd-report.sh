#!/usr/bin/env bash
# Per-op SIMD inspector for CPU-kernel development and porting. It reads the assembly of the
# compiler, which `zig build simd-probe` emits for the build target. It therefore works for any
# architecture Zig can compile, cross-compilation included, and needs no multi-arch objdump.
#
# The width table is the overview across every kernel, one row each. A kernel name is the
# drill-down. One kernel already runs to dozens or hundreds of instructions, so the script takes
# one name and dumps only the kernels that match it.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source tools/common.sh

# -- constants ---------------------------------------------------------------

readonly asm_file=zig-out/zigferencer-simd-probe.s

# -- state -------------------------------------------------------------------

# load_probe sets these once.
caps=""
arch=""

# -- helpers -----------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: tools/simd-report.sh [KERNEL]

Print the SIMD width of every probed kernel, or the assembly of the kernels that match KERNEL.

  (no argument)  Print the build target, then one row per kernel with its vector width.
  KERNEL         Print the assembly and the mnemonic histogram of each kernel whose name
                 holds the substring. For example, linearF32Q80 matches the naive, simd, and
                 simd_pf variants together.
  -h, --help     Print this text.

Build the probe first with `zig build simd-probe`, or run `make simd-report`. To inspect a port,
add the target: `zig build simd-probe -Dtarget=aarch64-linux-gnu -Dcpu=cortex_a72`.

The width column answers one question: did this kernel vectorize, and how wide? A backend that
must reach 256-bit or NEON and reports `scalar` did not vectorize. A kernel below the target
ceiling is headroom, not a bug. The `naive` kernels are the scalar reference by design. The simd
and simd_pf f32 linears are one kernel. The simd and simd_pf Q8_0 linears share one source and
give the same bits. The naive kernels sum in a
different order, so the SIMD kernels differ from them inside the declared tolerance. The
attention part kernels and the GDN step of simd and simd_pf fuse, and tests hold them to a
measured tolerance. An op under `naive` alone is one that every backend shares.

Exit codes: 0 success, 1 a missing probe file or no match, 2 a usage error.
EOF
}

# The build target comes from the marker that the probe bakes into the assembly. Nothing runs the
# probe, because it can be cross-compiled for a foreign architecture.
load_probe() {
    [ -f "$asm_file" ] || die "'$asm_file' not found. Build it first: 'zig build simd-probe'."
    caps="$(grep -oE '##SIMD-PROBE##[^"]*' "$asm_file" | head -1 || true)"
    [ -n "$caps" ] ||
        die "no '##SIMD-PROBE##' marker in '$asm_file'. Rebuild with 'zig build simd-probe'."
    arch="$(printf '%s' "$caps" | sed -E 's/.*arch=([^ ]+).*/\1/')"
}

# The compiler names a generic instantiation with an `__anon_<n>` suffix. Such a symbol is a
# helper inside a probed op and not a probed op, so the report drops it and lists exactly what
# `probed_ops` in `tools/simd_probe.zig` names. A probe wrapper, such as
# `simd_probe.attentionPartF32`, pins one concrete instance of a generic kernel, so the report
# reads it too. The probe `main` is not a kernel.
all_syms() {
    grep -oE '^(ops|simd_probe)\.[A-Za-z0-9_.]+:' "$asm_file" |
        sed 's/:$//' |
        grep -vE '(__anon_[0-9]+|^simd_probe\.main)$' |
        sort -u
}

# Dump the assembly of one kernel, then a mnemonic histogram. The dump drops the debug and unwind
# noise, and keeps the basic-block labels for the loop structure.
dump_kernel() {
    local sym="$1" body
    echo "== $sym  (target: $arch) =="
    body="$(awk -v s="$sym:" '
        $0 == s { f = 1 }
        f && /^[[:space:]]*\.size[[:space:]]/ { exit }
        f { print }
    ' "$asm_file" |
        grep -vE '^[[:space:]]*\.(cfi|loc|file|type|p2align|size)' |
        grep -vE '^\.L(tmp|func)' || true)"
    printf '%s\n' "$body"
    echo
    echo "-- mnemonic histogram --"
    printf '%s\n' "$body" |
        grep -E '^[[:space:]]+[a-z]' |
        awk '{ print $1 }' |
        sort |
        uniq -c |
        sort -rn || true
    echo
}

# The awk pass attributes each instruction line to the current kernel symbol and tallies the
# vector activity of the architecture. The width verdict separates real packed arithmetic from
# scalar FP that only lives in a vector register. It drops an `__anon_<n>` symbol for the reason
# that `all_syms` states.
readonly width_awk='
function declkey(s,   d) { d = s; sub(/.*\./, "", d); return d }
/^(ops|simd_probe)\.[A-Za-z0-9_.]+:$/ {
    cur = substr($0, 1, length($0) - 1)
    if (cur ~ /__anon_[0-9]+/ || cur == "simd_probe.main") { cur = ""; next }
    if (!(cur in seen)) { seen[cur] = 1; order[++n] = cur }
    next
}
/^[[:space:]]*\.size[[:space:]]/ { cur = ""; next }
cur == "" { next }
$0 !~ /^[[:space:]]+[a-z]/ { next }
{
    insns[cur]++
    mn = $1
    if (arch == "x86_64" || arch == "x86") {
        if ($0 ~ /[^a-z]zmm[0-9]/) w512[cur]++
        if ($0 ~ /[^a-z]ymm[0-9]/) w256[cur]++
        if (mn ~ /^vf(n)?m(add|sub)/) fma[cur]++
        if (mn ~ /^prefetch/) pf[cur]++
        if (mn ~ packed_x86 && $0 ~ /[^a-z]xmm[0-9]/ && $0 !~ /[^a-z]ymm[0-9]/) w128[cur]++
    } else if (arch == "aarch64" || arch == "aarch64_be") {
        if ($0 ~ /[^a-z]z[0-9]+\.[bhsd]/) sve[cur]++
        if ($0 ~ /[^a-z]v[0-9]+\.(16b|8h|4s|2d)/) n128[cur]++
        if ($0 ~ /[^a-z]v[0-9]+\.(8b|4h|2s)/) n64[cur]++
        if (mn ~ /^(fmla|fmls)$/) fma[cur]++
        if (mn ~ /^[su]dot$/) dot[cur]++
        if (mn ~ /^prfm/) pf[cur]++
    }
}
END {
    for (i = 1; i <= n; i++) {
        s = order[i]
        if (arch == "x86_64" || arch == "x86") {
            if (w512[s]) vec = "512-bit"
            else if (w256[s]) vec = "256-bit"
            else if (w128[s]) vec = "128-bit"
            else vec = "scalar"
            if (fma[s]) vec = vec "+fma"
            if (pf[s]) vec = vec "+pf"
            printf "%s\t%-34s %7d %6d %6d %6d %6d  %s\n", declkey(s), s, insns[s],
                w256[s] + 0, w512[s] + 0, fma[s] + 0, pf[s] + 0, vec
        } else {
            if (sve[s]) vec = "sve"
            else if (n128[s]) vec = "neon-128"
            else if (n64[s]) vec = "neon-64"
            else vec = "scalar"
            if (dot[s]) vec = vec "+dot"
            if (fma[s]) vec = vec "+fma"
            if (pf[s]) vec = vec "+pf"
            printf "%s\t%-34s %7d %6d %6d %6d %6d  %s\n", declkey(s), s, insns[s],
                (n128[s] + n64[s]) + 0, sve[s] + 0, fma[s] + 0, pf[s] + 0, vec
        }
    }
}'

# A packed x86 arithmetic mnemonic that runs on an xmm register is a 128-bit kernel.
readonly packed_x86_re='^v(mul|add|sub|div|max|min|sqrt|rsqrt|rcp|fmadd[0-9]*|fmsub[0-9]*|'\
'fnmadd[0-9]*|fnmsub[0-9]*|hadd|cvtdq2)p[sd]$'

width_table() {
    local legend c3 c4
    echo "== build target =="
    printf '%s\n' "$caps" | sed -E \
        -e 's/##SIMD-PROBE## //' \
        -e 's/ available=/\n  available: /' \
        -e 's/ absent=/\n  absent:    /' \
        -e 's/,/ /g'
    echo

    case "$arch" in
        x86_64 | x86)
            legend="ymm=256b  zmm=512b  fma=vfmadd  pf=prefetch"
            c3=ymm
            c4=zmm
            ;;
        aarch64 | aarch64_be)
            legend="neon=v.<N><T>  sve=z.<T>  dot=sdot/udot  fma=fmla  pf=prfm"
            c3=neon
            c4=sve
            ;;
        *)
            legend="(unrecognized arch '$arch'. The counts can be empty)"
            c3=v256
            c4=v512
            ;;
    esac
    echo "== per-kernel SIMD width ($legend) =="
    printf '%-34s %7s %6s %6s %6s %6s  %s\n' kernel insns "$c3" "$c4" fma pf vectorization

    awk -v arch="$arch" -v packed_x86="$packed_x86_re" "$width_awk" "$asm_file" |
        sort -t$'\t' -k1,1 -k2,2 |
        cut -f2-
}

# -- main --------------------------------------------------------------------

main() {
    case "${1:-}" in
        -h | --help)
            usage
            return 0
            ;;
        -*) die_usage "unknown option '$1'" ;;
    esac
    [ $# -le 1 ] || die_usage "expected at most one KERNEL"

    load_probe
    if [ $# -eq 0 ]; then
        width_table
        return 0
    fi

    local kernel="$1" syms count sym
    syms="$(all_syms | grep -F -- "$kernel" || true)"
    if [ -z "$syms" ]; then
        echo "error: no kernel matches '$kernel'. The kernels are:" >&2
        all_syms >&2
        exit 1
    fi
    count="$(printf '%s\n' "$syms" | grep -c .)"
    if [ "$count" -gt 1 ]; then
        echo "matched $count kernels for '$kernel':"
        echo
    fi
    while IFS= read -r sym; do
        dump_kernel "$sym"
    done <<<"$syms"
}

main "$@"
