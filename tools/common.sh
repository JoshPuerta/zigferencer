# The status lines and the build call that the scripts of tools/ share. This file is sourced, never
# executed. The markers match the output of zigferencer: [+] done, [*] note, [!] error. Status goes
# to stderr, so stdout stays program output. Color needs a terminal, and $NO_COLOR at any value
# turns it off.

if [ -t 2 ] && [ -z "${NO_COLOR+x}" ]; then
    readonly c_reset=$'\033[0m' c_dim=$'\033[2m' c_green=$'\033[32m' c_red=$'\033[31m'
else
    readonly c_reset='' c_dim='' c_green='' c_red=''
fi

note() { printf '%s[*]%s %s\n' "$c_dim" "$c_reset" "$*" >&2; }
ok() { printf '%s[+]%s %s\n' "$c_green" "$c_reset" "$*" >&2; }

# Exit code 1 is a failed run, and 2 is a usage error.
die() {
    printf '%s[!]%s %s\n' "$c_red" "$c_reset" "$*" >&2
    exit 1
}

die_usage() {
    printf '%s[!]%s %s. Run %s --help.\n' "$c_red" "$c_reset" "$*" "$0" >&2
    exit 2
}

# zig_build OPT BACKEND [FLAG...]
# Build both binaries in one call, so zig-out/bin never holds a pair from two builds. ZIG and FLAGS
# mean what they mean in the Makefile.
zig_build() {
    local opt="$1" backend="$2" flags=()
    shift 2
    read -ra flags <<<"${FLAGS:-}"
    note "zig build install dev -Doptimize=$opt -Dops-backend=$backend${FLAGS:+ $FLAGS}${*:+ $*}"
    "${ZIG:-zig}" build install dev -Doptimize="$opt" -Dops-backend="$backend" "${flags[@]}" "$@"
}
