#!/usr/bin/env bash
# Compile the zig build runner with LLVM, so lldb and gdb can step through build.zig. `zig build`
# compiles the runner with the default code generator and refuses -fllvm. A direct compile of the
# same modules gets past that. The module list follows lib/compiler/build_runner.zig of Zig 0.16.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=tools/common.sh
source tools/common.sh

# -- constants ---------------------------------------------------------------

readonly zig="${ZIG:-zig}"
readonly out=zig-out/debug/build-runner

# -- helpers -----------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: tools/debug-build.sh

Compile the zig build runner with LLVM to zig-out/debug/build-runner, then print the lldb command.

Environment:
  ZIG  the zig 0.16.0 binary (default zig)

Exit codes: 0 the runner is ready, 1 a build failed, 2 a usage error.
EOF
}

# The value of one field of `zig env`.
env_field() { sed -n "s/^ *\.$2 = \"\(.*\)\",$/\1/p" <<<"$1"; }

main() {
    case "${1:-}" in
        "") ;;
        -h | --help)
            usage
            return
            ;;
        *) die_usage "unknown argument '$1'" ;;
    esac

    # The first command fetches the lazy dependency. The second writes dependencies.zig.
    "$zig" build --fetch=all
    "$zig" build -l >/dev/null

    local zig_env lib deps vulkan
    zig_env="$("$zig" env)"
    lib="$(env_field "$zig_env" lib_dir)"
    deps="$(ls -t .zig-cache/o/*/dependencies.zig | sed -n 1p)"
    vulkan="$(sed -n 's/^ *\.{ "vulkan", "\(.*\)" },$/\1/p' "$deps")"
    [ -n "$vulkan" ] || die "no vulkan package in $deps"

    mkdir -p "$(dirname "$out")"
    "$zig" build-exe -fllvm -ODebug -femit-bin="$out" \
        --dep @build --dep @dependencies -Mroot="$lib/compiler/build_runner.zig" \
        -M@build=build.zig --dep "$vulkan" -M@dependencies="$deps" \
        -M"$vulkan=zig-pkg/$vulkan/build.zig"

    ok "build runner ready: $out"
    note "from the repository root, run:"
    printf 'lldb %s -o "breakpoint set --name build.resolveOpsBackend" \\\n' "$out"
    printf '  -o "run %s %s %s .zig-cache %s -l"\n' \
        "$(env_field "$zig_env" zig_exe)" "$lib" "$PWD" \
        "$(env_field "$zig_env" global_cache_dir)"
}

main "$@"
