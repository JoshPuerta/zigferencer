#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Check that every path through one model emits one token stream.

Greedy decoding is deterministic, so the cases of a gate must emit one token hash or the hash on
record. A pair of cases also compares their logits within a bound. TTFT and ITL print as
information, and a gate never judges them. To add a gate, a case, or a pair, edit `GATES`.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

BIN = Path("zig-out/bin/zigferencer-dev" + (".exe" if sys.platform == "win32" else ""))
SMOLLM2 = "SmolLM2-135M-Instruct-Q8_0"
QWEN_2B = "Qwen3.5-2B-Q8_0"
SAME = "same"


class Scope(StrEnum):
    ALL = "all"
    CPU = "cpu"
    GPU = "gpu"


class Check(StrEnum):
    OK = "ok"
    NEAR = "near"
    DIFF = "DIFF"


class PairCheck(StrEnum):
    OK = "ok"
    NEAR = "near"
    DIFF = "DIFF"
    ERROR = "error"
    MEASURED = "measured"


@dataclass(frozen=True, slots=True)
class Case:
    """One `bench-model` call. `expect` is SAME, the hash of the first SAME case, or a record."""
    build: str
    label: str
    args: str
    expect: str = SAME


@dataclass(frozen=True, slots=True)
class Pair:
    """Two cases whose logits differ by at most `bound`. A bound of None is not measured yet."""
    ref: str
    cand: str
    bound: float | None


@dataclass(frozen=True, slots=True)
class Gate:
    summary: str
    cases: tuple[Case, ...]
    pairs: tuple[Pair, ...] = ()
    takes_model: bool = True


@dataclass(frozen=True, slots=True)
class Options:
    gate: str | None
    scope: Scope
    model: str
    listing: bool
    measure: bool
    extra: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    hash: str
    want: str
    check: Check
    ttft_ms: str
    itl_ms: str


@dataclass(frozen=True, slots=True)
class PairResult:
    max_dlogit: str
    top1_agree: str
    entries: str
    kl_max: str
    check: PairCheck


@dataclass(frozen=True, slots=True)
class Verdict:
    checks: Mapping[str, Check]
    stale: tuple[str, ...]
    failed: int
    near: int


class GateError(Exception):
    """A failed run: exit 1."""


# {kv} pins the KV to f32 outside --cpu, because the per-op vulkan backend has no other KV dtype.
BACKENDS = "{model} --prompt Hello --max-tokens 32 --runs 1{kv}"
THREADS = "{model} --prompt-tokens 128 --max-tokens 32 --ignore-eos --runs 1"
HYBRID = "{model} --runs 1 --max-tokens 32 --ignore-eos"
DEEP = "--prompt-tokens 8064 --max-tokens 64 --ignore-eos --runs 1 --max-seq 8192"

# Keep the cases of one build adjacent: the gate builds when the build changes. A case of a pair
# needs `--runs 1 --ignore-eos`. A new pair starts with the bound None. tools/README.md says what
# a DIFF of each gate means and how to measure a bound.
GATES: Mapping[str, Gate] = MappingProxyType({
    "backends": Gate("naive, simd, simd_pf, and per-op vulkan emit one token stream", (
        Case("naive", "naive", BACKENDS),
        Case("simd", "simd", BACKENDS),
        Case("simd_pf", "simd_pf", BACKENDS),
        Case("vulkan", "vulkan", BACKENDS),
    )),
    # No pairs here: a thread count must not move a bit.
    "threads": Gate("1, 2, 4, and 8 pool threads emit one token stream", (
        Case("auto", "t=1", THREADS + " --threads 1"),
        Case("auto", "t=2", THREADS + " --threads 2"),
        Case("auto", "t=4", THREADS + " --threads 4"),
        Case("auto", "t=8", THREADS + " --threads 8"),
    )),
    # @last and @half take their values from the layer count of the model file.
    "hybrid": Gate(
        "the CPU runner, the fused runner, and three hybrid splits emit one token stream", (
            Case("auto", "cpu", HYBRID),
            Case("vulkan_fused", "fused", HYBRID),
            Case("vulkan_fused", "k=1", HYBRID + " --gpu-layers @last"),
            Case("vulkan_fused", "mid", HYBRID + " --gpu-layers @half"),
            Case("vulkan_fused", "k=L-1", HYBRID + " --gpu-layers 1"),
        )),
    # The last field of a case is its hash on record. To update a record, replace that hash.
    "deep": Gate(
        "Qwen3.5-2B and SmolLM2 at 8064 positions match the hashes on record", (
            Case("auto", "2B-cpu", f"{QWEN_2B} {DEEP}", "0xd4f86c9b4daf7731"),
            Case("auto", "smol-cpu", f"{SMOLLM2} {DEEP}", "0xed2a39102b50bc4e"),
            Case("vulkan_fused", "2B-fused", f"{QWEN_2B} {DEEP}", "0xd4f86c9b4daf7731"),
            Case("vulkan_fused", "smol-fused", f"{SMOLLM2} {DEEP}", "0xed2a39102b50bc4e"),
        ), takes_model=False),
})

EPILOG = """\
pairs:
  A gate can hold pairs: a reference case, a candidate case, and a bound T on the largest logit
  distance. A pair prints ok (equal hashes, within T), near (different hashes, within T), or DIFF.
  When the hashes differ, the candidate runs again with the ids of the reference. A candidate case
  that expects `same` and differs passes as near when its pair passes and its reference case
  matches. A case with a hash on record never passes as near. The pairs resolve in table order.

  Arguments after -- go to every bench-model call, for example -- --kv-dtype f16.

environment:
  ZIG                     the zig 0.16.0 binary (default zig)
  FLAGS                   more zig build flags
  ZIGFERENCER_MODELS_DIR  the model store (default models/)

exit codes:
  0  every case and pair passes
  1  a case or a pair differs or fails
  2  a usage error
"""


def select_cases(cases: Iterable[Case], scope: Scope, model: str,
                 has_simd: bool) -> tuple[tuple[Case, ...], tuple[str, ...]]:
    """The cases in the scope, with their arguments filled, and a note per skip."""
    kv = "" if scope is Scope.CPU else " --kv-dtype f32"
    chosen: list[Case] = []
    notes: list[str] = []
    for case in cases:
        on_gpu = case.build.startswith("vulkan")
        if (scope is Scope.CPU and on_gpu) or (scope is Scope.GPU and not on_gpu):
            continue
        # The build refuses a vector backend on such a host. Other build failures stop the gate.
        if case.build.startswith("simd") and not has_simd:
            notes.append(f"skip {case.build}: the host lacks AVX2 or FMA")
            continue
        chosen.append(replace(case, args=case.args.format(model=model, kv=kv)))
    return tuple(chosen), tuple(notes)


def select_pairs(pairs: Iterable[Pair], cases: Iterable[Case], measure: bool,
                 listing: bool) -> tuple[tuple[Pair, ...], tuple[str, ...]]:
    """The pairs whose two cases are selected, and a note per skip. A refused pair raises."""
    labels = {case.label for case in cases}
    chosen: list[Pair] = []
    notes: list[str] = []
    for pair in pairs:
        name = f"pair {pair.ref}/{pair.cand}"
        if pair.bound is not None and not pair.bound >= 0:
            raise ValueError(f"{name}: the bound {pair.bound} is not a number of 0 or more")
        if pair.ref not in labels or pair.cand not in labels:
            notes.append(f"skip {name}: this run selected only one of its cases")
            continue
        if pair.bound is None and not measure and not listing:
            raise ValueError(f"{name} has no bound. Run --measure and record one")
        chosen.append(pair)
    return tuple(chosen), tuple(notes)


def resolve_layers(args: str, layers: int) -> str:
    assert layers > 0
    return args.replace("@last", str(layers - 1)).replace("@half", str(layers // 2))


def pair_check(same: bool, bound: float | None, measure: bool, compared: bool) -> PairCheck:
    """The check of one pair. Without a bound, a refusal of compare-logits is an `error`."""
    if not compared:
        return PairCheck.ERROR if measure else PairCheck.DIFF
    if measure:
        return PairCheck.MEASURED
    if same:
        return PairCheck.OK
    # A bound of 0 admits no `near`: equal logits must give equal hashes.
    return PairCheck.DIFF if bound == 0 else PairCheck.NEAR


def judge(cases: Iterable[Case], results: Mapping[str, CaseResult], pairs: tuple[Pair, ...],
          pair_results: tuple[PairResult, ...]) -> Verdict:
    """The final check of every case, after the pairs in table order."""
    assert len(pairs) == len(pair_results)
    expect = {case.label: case.expect for case in cases}
    checks = {label: results[label].check for label in expect}
    stale: list[str] = []
    pair_failed = 0
    for pair, res in zip(pairs, pair_results):
        if res.check in (PairCheck.DIFF, PairCheck.ERROR):
            pair_failed += 1
        elif (res.check in (PairCheck.OK, PairCheck.NEAR) and checks[pair.cand] is Check.DIFF
              and checks[pair.ref] is not Check.DIFF):
            if expect[pair.cand] == SAME:
                checks[pair.cand] = Check.NEAR
            else:
                stale.append(pair.cand)
    failed = sum(check is Check.DIFF for check in checks.values()) + pair_failed
    near = sum(check is Check.NEAR for check in checks.values())
    return Verdict(checks, tuple(stale), failed, near)


def bound_text(bound: float | None) -> str:
    return "-" if bound is None else str(bound)


def json_line(text: str) -> dict:
    """The fields of the first JSON line, with each number kept as its printed text."""
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                return json.loads(line, parse_float=str, parse_int=str)
            except ValueError:
                return {}
    return {}


def case_line(label: str, build: str, got: str, want: str, check: str, ttft: str,
              itl: str) -> str:
    return f"{label:<11} {build:<13} {got:<18} {want:<18} {check:<5} {ttft:>11} {itl:>9}"


def pair_line(name: str, ref: str, cand: str, dlogit: str, top1: str, kl: str, check: str) -> str:
    return f"{name:<22} {ref:<18} {cand:<18} {dlogit:<15} {top1:<10} {kl:<22} {check}"


CASE_HEADER = case_line("label", "build", "hash", "expected", "check", "ttft_ms", "itl_ms")


def note(text: str) -> None:
    print(f"[*] {text}", file=sys.stderr, flush=True)


def host_has_simd() -> bool:
    """Ask Zig for the CPU of a native build. A simd backend needs x86_64 with AVX2 and FMA."""
    argv = [os.environ.get("ZIG") or "zig", "build-exe", "--show-builtin"]
    try:
        text = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    except OSError as err:
        raise GateError(f"cannot ask zig for the host CPU: {err}") from err
    except subprocess.CalledProcessError as err:
        raise GateError(f"cannot ask zig for the host CPU: {err}\n{err.stderr}") from err
    if "pub const cpu: std.Target.Cpu" not in text:
        raise GateError("zig printed no CPU for the host")
    return ".avx2," in text and ".fma," in text


def build(backend: str) -> None:
    """Build both binaries in one call, so zig-out/bin never holds a pair from two builds."""
    flags = os.environ.get("FLAGS", "")
    note(f"zig build install dev -Doptimize=ReleaseFast -Dops-backend={backend}"
         + (f" {flags}" if flags else ""))
    argv = [os.environ.get("ZIG") or "zig", "build", "install", "dev",
            "-Doptimize=ReleaseFast", f"-Dops-backend={backend}", *flags.split()]
    try:
        code = subprocess.run(argv).returncode
    except OSError as err:
        raise GateError(f"the build of {backend} failed: {err}") from err
    if code != 0:
        raise GateError(f"the build of {backend} failed")


def call(argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run([BIN, *argv], capture_output=True, text=True)
    except OSError as err:
        raise GateError(f"cannot run {BIN}: {err}") from err


def bench_args(case: Case, model: str) -> list[str]:
    """The arguments of a case. The model file is the one source of its layer count."""
    if "@" not in case.args:
        return case.args.split()
    for words in (line.split() for line in call(["inspect", model]).stdout.splitlines()):
        if len(words) >= 2 and words[0] == "layers:" and words[1].isdigit():
            return resolve_layers(case.args, int(words[1])).split()
    raise GateError(f"inspect printed no layer count for {model}")


def bench(argv: list[str], failure: str) -> dict:
    """Run `bench-model` and return the fields of its one run."""
    proc = call(["bench-model", *argv])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise GateError(failure)
    runs = json_line(proc.stdout).get("runs")
    return runs[0] if isinstance(runs, list) and runs else {}


def run_case(case: Case, opts: Options, logits: Path | None, first: str) -> CaseResult:
    dump = ["--logits-out", str(logits)] if logits else []
    fields = bench([*bench_args(case, opts.model), *dump, *opts.extra, "--json"],
                   f"case {case.label} failed")
    got = str(fields.get("tokens_hash", ""))
    if not got:
        raise GateError(f"case {case.label} printed no tokens_hash")
    want = (first or got) if case.expect == SAME else case.expect
    check = Check.OK if got == want else Check.DIFF
    return CaseResult(got, want, check, str(fields.get("ttft_ms", "")),
                      str(fields.get("itl_ms", "")))


def logits_path(tmp: Path, label: str) -> Path:
    return tmp / f"{label}.bin"


def forced_path(tmp: Path, pair: Pair) -> Path:
    """The logits file of the forced run of a pair. Two pairs can share a candidate."""
    return tmp / f"{pair.ref}--{pair.cand}.forced.bin"


def compare_pair(tmp: Path, pair: Pair, same: bool, measure: bool) -> PairResult:
    cand = logits_path(tmp, pair.cand) if same else forced_path(tmp, pair)
    limit = [] if measure else ["--max-dlogit", bound_text(pair.bound)]
    proc = call(["compare-logits", str(logits_path(tmp, pair.ref)), str(cand), *limit, "--json"])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    fields = {key: str(value) for key, value in json_line(proc.stdout).items()}
    return PairResult(fields.get("max_dlogit", ""), fields.get("top1_agree", ""),
                      fields.get("entries", ""), fields.get("kl_max", ""),
                      pair_check(same, pair.bound, measure, proc.returncode == 0))


def print_list(cases: Iterable[Case], pairs: Iterable[Pair], width: int) -> None:
    for case in cases:
        print(f"{case.build:<13} {case.label:<{width}}  {case.expect}  {case.args}")
    for pair in pairs:
        print(f"pair  {pair.ref} {pair.cand} {bound_text(pair.bound)}")


def print_cases(cases: Iterable[Case], results: Mapping[str, CaseResult],
                checks: Mapping[str, Check]) -> None:
    for case in cases:
        res = results[case.label]
        print(case_line(case.label, case.build, res.hash, res.want, checks[case.label],
                        res.ttft_ms, res.itl_ms))


def print_pairs(pairs: Iterable[Pair], pair_results: Iterable[PairResult],
                results: Mapping[str, CaseResult]) -> None:
    print(pair_line("pair", "ref_hash", "cand_hash", "max_dlogit", "top1_agree", "kl_max",
                    "result"))
    for pair, res in zip(pairs, pair_results):
        print(pair_line(f"{pair.ref}/{pair.cand}", results[pair.ref].hash,
                        results[pair.cand].hash, res.max_dlogit,
                        f"{res.top1_agree}/{res.entries}", res.kl_max, res.check))


def run_cases(opts: Options, cases: tuple[Case, ...], pairs: tuple[Pair, ...],
              tmp: Path) -> tuple[dict[str, CaseResult], str, str]:
    """Run every case. Return the results by label, the first SAME hash, and the last build."""
    named = {pair.ref for pair in pairs} | {pair.cand for pair in pairs}
    results: dict[str, CaseResult] = {}
    last_build = first = ""
    # With pairs, the cases print after the pairs, because a pair can turn a DIFF case into near.
    if not pairs:
        print(CASE_HEADER)
    for case in cases:
        if case.build != last_build:
            build(case.build)
            last_build = case.build
        logits = logits_path(tmp, case.label) if case.label in named else None
        result = results[case.label] = run_case(case, opts, logits, first)
        if case.expect == SAME:
            first = first or result.hash
        if not pairs:
            print_cases((case,), results, {case.label: result.check})
    return results, first, last_build


def force_candidates(opts: Options, cases: Iterable[Case], differ: tuple[Pair, ...],
                     last_build: str, tmp: Path) -> None:
    """Run the candidate of each pair whose hashes differ again, with the ids of its reference."""
    by_label = {case.label: case for case in cases}
    # The runs go by build, the build in zig-out first, so each build compiles at most once more.
    builds = sorted(dict.fromkeys(by_label[p.cand].build for p in differ),
                    key=lambda b: b != last_build)
    for backend in builds:
        if backend != last_build:
            build(backend)
            last_build = backend
        for pair in (p for p in differ if by_label[p.cand].build == backend):
            cand = by_label[pair.cand]
            bench([*bench_args(cand, opts.model), *opts.extra, "--json", "--force-tokens",
                   str(logits_path(tmp, pair.ref)), "--logits-out", str(forced_path(tmp, pair))],
                  f"the forced run of {cand.label} failed")


def run_gate(opts: Options, cases: tuple[Case, ...], pairs: tuple[Pair, ...]) -> None:
    """Run the cases, then the pairs, and print the tables. A failed gate raises GateError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        results, first, last_build = run_cases(opts, cases, pairs, tmp)
        differ = tuple(p for p in pairs if results[p.ref].hash != results[p.cand].hash)
        force_candidates(opts, cases, differ, last_build, tmp)
        pair_results = tuple(compare_pair(tmp, p, p not in differ, opts.measure) for p in pairs)

    verdict = judge(cases, results, pairs, pair_results)
    if pairs:
        print_pairs(pairs, pair_results, results)
        print()
        print(CASE_HEADER)
        print_cases(cases, results, verdict.checks)
        for label in verdict.stale:
            note(f"case {label} differs from its record and its pair passes: update the record")
    if verdict.failed > 0:
        raise GateError(f"gate {opts.gate}: {verdict.failed} case(s) or pair(s) differ."
                        " Investigate before you change a record")
    if verdict.near > 0:
        text = f"{len(cases)} cases pass, {verdict.near} of them near through a pair"
    elif first:
        text = f"{len(cases)} cases emit {first}"
    else:
        text = f"{len(cases)} cases match the record"
    print(f"[+] gate {opts.gate}: {text}", file=sys.stderr)


def make_parser(gates: Mapping[str, Gate]) -> argparse.ArgumentParser:
    width = max(map(len, gates))
    names = "\n".join(f"  {name:<{width}}  {gate.summary}" for name, gate in gates.items())
    with_model = ", ".join(name for name, gate in gates.items() if gate.takes_model)
    parser = argparse.ArgumentParser(
        prog="tools/gate.py",
        usage="%(prog)s GATE [--cpu | --gpu] [--model NAME] [--list] [--measure]"
              " [-- BENCH_FLAG...]",
        description="Check that every path through one model emits one token stream."
                    f"\n\ngates:\n{names}",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("gate", nargs="?", choices=gates, metavar="GATE",
                        help="the gate to run or to list")
    parser.add_argument("--cpu", dest="scope", action="store_const", const=Scope.CPU,
                        default=Scope.ALL, help="run only the cases of a CPU build")
    parser.add_argument("--gpu", dest="scope", action="store_const", const=Scope.GPU,
                        help="run only the cases of a GPU build")
    parser.add_argument("--model", metavar="NAME", default=SMOLLM2,
                        help=f"replace %(default)s in {with_model}")
    parser.add_argument("--list", dest="listing", action="store_true",
                        help="print the cases and the pairs and run nothing."
                             " Without a GATE, print every gate")
    parser.add_argument("--measure", action="store_true",
                        help="run every pair of the gate and print its metrics, but judge no pair")
    return parser


def parse(parser: argparse.ArgumentParser, gates: Mapping[str, Gate], argv: list[str]) -> Options:
    cut = argv.index("--") if "--" in argv else len(argv)
    ns = parser.parse_args(argv[:cut])
    if ns.gate is None and not ns.listing:
        parser.error("name a gate")
    if ns.gate and not gates[ns.gate].takes_model and ns.model != SMOLLM2:
        parser.error(f"{ns.gate} holds a record per model, so it takes no --model")
    if ns.measure and ns.gate is None:
        parser.error("--measure needs a gate")
    return Options(ns.gate, ns.scope, ns.model, ns.listing, ns.measure, tuple(argv[cut + 1:]))


def select(parser: argparse.ArgumentParser, gate: Gate, opts: Options,
           has_simd: bool) -> tuple[tuple[Case, ...], tuple[Pair, ...]]:
    cases, notes = select_cases(gate.cases, opts.scope, opts.model, has_simd)
    try:
        pairs, more = select_pairs(gate.pairs, cases, opts.measure, opts.listing)
    except ValueError as err:
        parser.error(str(err))
    for text in notes + more:
        note(text)
    return cases, pairs


def run(argv: list[str], gates: Mapping[str, Gate] = GATES) -> int:
    """The exit code of one run. A test can pass other tables."""
    parser = make_parser(gates)
    opts = parse(parser, gates, argv)
    try:
        has_simd = host_has_simd()
        if opts.listing:
            for name in (opts.gate,) if opts.gate else tuple(gates):
                cases, pairs = select(parser, gates[name], opts, has_simd)
                if not opts.gate:
                    print(f"# {name}")
                print_list(cases, pairs, max(len(case.label) for case in gates[name].cases))
            return 0
        assert opts.gate is not None
        cases, pairs = select(parser, gates[opts.gate], opts, has_simd)
        if len(cases) < 2:
            parser.error(f"gate {opts.gate} selected {len(cases)} case(s), too few to compare")
        if opts.measure and not pairs:
            parser.error(f"gate {opts.gate} selected no pair to measure")
        run_gate(opts, cases, pairs)
    except GateError as err:
        print(f"[!] {err}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    os.chdir(Path(__file__).absolute().parent.parent)
    # A pipe must show each table line before the status lines that follow it on stderr.
    sys.stdout.reconfigure(line_buffering=True)
    return run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
