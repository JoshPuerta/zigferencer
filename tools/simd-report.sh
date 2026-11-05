#!/usr/bin/env bash
# Per-op SIMD inspector for CPU-kernel development and porting. Three uses:
#
#   tools/simd-report.sh [asm-file]                 # width table for every kernel
#   tools/simd-report.sh list [asm-file]            # which kernels + backends exist
#   tools/simd-report.sh asm <substring> [asm-file] # assembly + mnemonics for every
#                                                   #   kernel whose name matches
#
# It reads the assembly of the compiler, which `zig build simd-probe` emits for
# the build target. It therefore works for any architecture Zig can compile,
# cross-compilation included, and needs no multi-arch objdump. To inspect a port,
# build the probe for the target and run the report:
#
#   zig build simd-probe -Dtarget=aarch64-linux-gnu -Dcpu=cortex_a72
#   tools/simd-report.sh
#
# The width table is the overview across every kernel, one row each. `asm` is the
# drill-down. It takes a name, because one kernel already runs to dozens or
# hundreds of instructions, and twenty at once make an unreadable wall. The name
# is a substring, so `asm linearF32Q80` dumps the naive, simd, and simd_pf
# variants together. `list` prints the menu, and names the backends that carry a
# distinct symbol for each op. An op under `naive` alone is one that every
# backend shares.
#
# The width column answers one question: did this kernel vectorize, and how wide?
# A regression is therefore visible during kernel work. A backend that must reach
# 256-bit or NEON and reports `scalar` did not vectorize. A kernel below the
# target ceiling in the header is headroom, not a bug. `naive` is the scalar
# reference by design, and the f32 kernels emit a separate multiply and add rather
# than a fused fma or fmla, which keeps them bit-identical with that reference. The
# cross-backend token-hash gate rests on that equality. The attention part
# kernels of simd and simd_pf fuse, and a test holds them to a measured
# tolerance against naive instead.

set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

readonly default_asm="zig-out/zigferencer-simd-probe.s"

# -- argument parsing --------------------------------------------------------
mode="table"
kernel=""
case "${1:-}" in
    list) mode="list"; asm="${2:-$default_asm}" ;;
    asm)  mode="asm"; kernel="${2:-}"; asm="${3:-$default_asm}" ;;
    *)    asm="${1:-$default_asm}" ;;
esac

[ -f "$asm" ] || die "'$asm' not found. Build it first: 'zig build simd-probe'."

# The build target comes from the marker that the probe bakes into the assembly.
# Nothing runs the probe, because it can be cross-compiled for a foreign arch.
caps="$(grep -oE '##SIMD-PROBE##[^"]*' "$asm" | head -1 || true)"
[ -n "$caps" ] || die "no '##SIMD-PROBE##' marker in '$asm' (stale or wrong file?). Rebuild with 'zig build simd-probe'."
arch="$(printf '%s' "$caps" | sed -E 's/.*arch=([^ ]+).*/\1/')"

# The compiler names a generic instantiation with an `__anon_<n>` suffix. Such a
# symbol is a helper inside a probed op and not a probed op, so the report drops
# it and lists exactly what `probed_ops` in `tools/simd_probe.zig` names. A probe
# wrapper, such as `simd_probe.attentionPartF32`, pins one concrete instance of a
# generic kernel, so the report reads it too. The probe `main` is not a kernel.
all_syms() { grep -oE '^(ops|simd_probe)\.[A-Za-z0-9_.]+:' "$asm" | sed 's/:$//' | grep -vE '(__anon_[0-9]+|^simd_probe\.main)$' | sort -u; }

# List every inspectable kernel, grouped by decl name, with the backends that
# carry a distinct symbol for it. The input arrives sorted, so the backends list
# as naive, then simd, then simd_pf.
list_kernels() {
    echo "== inspectable kernels (KERNEL=<substring>. A substring dumps all it matches) =="
    all_syms | awk '
        { be=$0; decl=$0; sub(/^ops\./,"",be); sub(/\..*/,"",be); sub(/^ops\.[^.]+\./,"",decl); sub(/^simd_probe\./,"",decl)
          if (!((decl SUBSEP be) in seen)) { seen[decl SUBSEP be]=1; bes[decl]=bes[decl]" "be } }
        END { for (d in bes) printf "  %-24s%s\n", d, bes[d] }
    ' | sort
    echo "  note: a kernel under 'naive' alone is shared. simd and simd_pf re-export"
    echo "        it unchanged, so there is no distinct simd symbol to inspect."
}

# Dump the assembly of one kernel, then a mnemonic histogram. The dump drops the
# debug and unwind noise, and keeps the basic-block labels for the loop structure.
dump_kernel() {
    local sym="$1" body
    echo "== $sym  (target: $arch) =="
    body="$(awk -v s="$sym:" '
        $0==s {f=1}
        f && /^[[:space:]]*\.size[[:space:]]/ {exit}
        f {print}
    ' "$asm" | grep -vE '^[[:space:]]*\.(cfi|loc|file|type|p2align|size)' | grep -vE '^\.L(tmp|func)' || true)"
    printf '%s\n' "$body"
    echo
    echo "-- mnemonic histogram --"
    printf '%s\n' "$body" | grep -E '^[[:space:]]+[a-z]' | awk '{print $1}' | sort | uniq -c | sort -rn || true
    echo
}

# -- list mode: the menu of inspectable kernels ------------------------------
if [ "$mode" = "list" ]; then
    echo "target: $arch"
    list_kernels
    exit 0
fi

# -- asm mode: assembly + histogram for every kernel the substring matches ---
if [ "$mode" = "asm" ]; then
    if [ -z "$kernel" ]; then
        echo "error: 'asm' needs a kernel substring. choose from:" >&2
        list_kernels >&2
        exit 1
    fi
    syms="$(all_syms | grep -F "$kernel" || true)"
    if [ -z "$syms" ]; then
        echo "error: no kernel matches '$kernel'. available:" >&2
        list_kernels >&2
        exit 1
    fi
    count="$(printf '%s\n' "$syms" | grep -c .)"
    if [ "$count" -gt 1 ]; then
        echo "matched $count kernels for '$kernel':"
        echo
    fi
    while IFS= read -r sym; do dump_kernel "$sym"; done <<< "$syms"
    exit 0
fi

# -- table mode: per-kernel width across every probed kernel -----------------
echo "== build target =="
printf '%s\n' "$caps" | sed -E 's/##SIMD-PROBE## //; s/ available=/\n  available: /; s/ absent=/\n  absent:    /; s/,/ /g'
echo

case "$arch" in
    x86_64|x86)         legend="ymm=256b  zmm=512b  fma=vfmadd  pf=prefetch"; c3=ymm; c4=zmm ;;
    aarch64|aarch64_be) legend="neon=v.<N><T>  sve=z.<T>  dot=sdot/udot  fma=fmla  pf=prfm"; c3=neon; c4=sve ;;
    *)                  legend="(unrecognized arch '$arch'. The counts can be empty)"; c3=v256; c4=v512 ;;
esac
echo "== per-kernel SIMD width ($legend) =="
printf '%-34s %7s %6s %6s %6s %6s  %s\n' kernel insns "$c3" "$c4" fma pf vectorization

# One awk pass attributes each instruction line to the current kernel symbol and
# tallies the vector activity of the architecture. The width verdict separates
# real packed arithmetic from scalar FP that only lives in a vector register.
# The pass drops an `__anon_<n>` symbol for the reason `all_syms` states above.
awk -v arch="$arch" '
function declkey(s,   d) { d=s; sub(/.*\./,"",d); return d }
/^(ops|simd_probe)\.[A-Za-z0-9_.]+:$/ { cur=substr($0,1,length($0)-1); if(cur ~ /__anon_[0-9]+/ || cur == "simd_probe.main"){cur=""; next}; if(!(cur in seen)){seen[cur]=1; order[++n]=cur}; next }
/^[[:space:]]*\.size[[:space:]]/ { cur=""; next }
cur=="" { next }
$0 !~ /^[[:space:]]+[a-z]/ { next }
{
    insns[cur]++
    mn=$1
    if (arch=="x86_64" || arch=="x86") {
        if ($0 ~ /[^a-z]zmm[0-9]/) w512[cur]++
        if ($0 ~ /[^a-z]ymm[0-9]/) w256[cur]++
        if (mn ~ /^vf(n)?m(add|sub)/) fma[cur]++
        if (mn ~ /^prefetch/) pf[cur]++
        if (mn ~ /^v(mul|add|sub|div|max|min|sqrt|rsqrt|rcp|fmadd[0-9]*|fmsub[0-9]*|fnmadd[0-9]*|fnmsub[0-9]*|hadd|cvtdq2)p[sd]$/ && $0 ~ /[^a-z]xmm[0-9]/ && $0 !~ /[^a-z]ymm[0-9]/) w128[cur]++
    } else if (arch=="aarch64" || arch=="aarch64_be") {
        if ($0 ~ /[^a-z]z[0-9]+\.[bhsd]/) sve[cur]++
        if ($0 ~ /[^a-z]v[0-9]+\.(16b|8h|4s|2d)/) n128[cur]++
        if ($0 ~ /[^a-z]v[0-9]+\.(8b|4h|2s)/) n64[cur]++
        if (mn ~ /^(fmla|fmls)$/) fma[cur]++
        if (mn ~ /^[su]dot$/) dot[cur]++
        if (mn ~ /^prfm/) pf[cur]++
    }
}
END {
    for (i=1;i<=n;i++) {
        s=order[i]
        if (arch=="x86_64" || arch=="x86") {
            if (w512[s]) vec="512-bit"; else if (w256[s]) vec="256-bit"; else if (w128[s]) vec="128-bit"; else vec="scalar"
            if (fma[s]) vec=vec"+fma"; if (pf[s]) vec=vec"+pf"
            printf "%s\t%-34s %7d %6d %6d %6d %6d  %s\n", declkey(s), s, insns[s], w256[s]+0, w512[s]+0, fma[s]+0, pf[s]+0, vec
        } else {
            if (sve[s]) vec="sve"; else if (n128[s]) vec="neon-128"; else if (n64[s]) vec="neon-64"; else vec="scalar"
            if (dot[s]) vec=vec"+dot"; if (fma[s]) vec=vec"+fma"; if (pf[s]) vec=vec"+pf"
            printf "%s\t%-34s %7d %6d %6d %6d %6d  %s\n", declkey(s), s, insns[s], (n128[s]+n64[s])+0, sve[s]+0, fma[s]+0, pf[s]+0, vec
        }
    }
}
' "$asm" | sort -t$'\t' -k1,1 -k2,2 | cut -f2-
