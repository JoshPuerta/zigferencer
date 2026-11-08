#!/usr/bin/env python3
"""Convert `perf script` text from tools/profile.sh into views of a run.

Commands:

offcpu RAW PID OUT
    RAW holds `sched:sched_switch` events of every CPU, each with the stack of
    the thread that leaves the CPU. A thread of PID is off the CPU from its
    switch out to its next switch in. Writes folded stacks weighted by that
    time in microseconds, for flamegraph.pl.

fold RAW DSO FILE OUT
    RAW holds CPU samples with call chains. Each frame of DSO, read from FILE,
    expands into its inline chain, read from the DWARF records through one
    batched addr2line call. A caller frame holds a return address, which can
    lie in the next inlined scope, so the lookup uses the return address minus
    one. Writes folded stacks, root first, weighted by the sample period.

samples RAW DSO FILE OUT
    RAW, DSO, and FILE are as for `fold`, and the frames expand the same way,
    with the same names as the flame graph. Writes a Perfetto proto trace on
    the CLOCK_MONOTONIC clock that `perf record -k mono` uses: one packet per
    sample with its thread and its call stack, and the thread names. Perfetto
    shows the samples beside the engine spans, where the thread ids match. The
    trace uses packet sequence 2. The engine spans use sequence 1, so
    `cat spans samples` makes one valid trace.

tree FOLDED CALLEES CALLERS
    FOLDED holds one stack per line, root first, then a weight. The first
    frame names the thread, and the tree keeps only the thread name, so equal
    threads merge. CALLEES gets the tree from the roots down: each frame with
    its inclusive and self share, children by weight. CALLERS gets one tree
    per hot frame, upward: who called it, by weight. Both drop branches under
    0.5 percent of the total.

It uses the Python standard library alone.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Iterator, NamedTuple, Optional

HEADER = re.compile(
    r"^\s*(?P<comm>.*?)\s+(?:(?P<pid>\d+)/)?(?P<tid>\d+)\s+"
    r"(?P<sec>\d+)\.(?P<frac>\d+):\s+(?:(?P<period>\d+)\s+)?"
    r"(?P<event>\S+):(?:\s+(?P<rest>.*))?$"
)
FRAME = re.compile(r"^\s+(?P<ip>[0-9a-f]+)\s+(?P<sym>.+?)(?:\s+\((?P<dso>[^()]*)\))?$")
SYMOFF = re.compile(r"\+0x[0-9a-f]+$")
THREAD_ID = re.compile(r"-[?\d]+/\d+$")
TREE_CUT = 0.005
ANON = re.compile(r"__anon_\d+")
PAREN = re.compile(r"\(([^()]*)\)")
MODULE_PREFIX = re.compile(r"\b(?:[a-z_][a-z0-9_]*\.)+(?=[A-Za-z_])")

# Perfetto constants. The field numbers come from the protos of Perfetto v59.1.
CLOCK_MONOTONIC = 3
CLOCK_BOOTTIME = 6
SAMPLE_SEQUENCE = 2
SEQ_CLEARED = 1
SEQ_NEEDS_STATE = 2

# perf prints a switch in one of two forms: the raw tracepoint fields, or the
# libtraceevent form `prev_comm:prev_pid [prio] STATE ==> next_comm:next_pid [prio]`.
SWITCH_FIELDS = re.compile(
    r"prev_comm=(?P<prev_comm>.*?) prev_pid=(?P<prev>\d+) .*?prev_state=(?P<state>\S+) "
    r"==> next_comm=.*? next_pid=(?P<next>\d+)"
)
SWITCH_PRETTY = re.compile(
    r"^(?P<prev_comm>.+):(?P<prev>\d+) \[-?\d+\] (?P<state>\S+) ==> .+:(?P<next>\d+) \[-?\d+\]$"
)


class Frame(NamedTuple):
    ip: int
    sym: str
    dso: str


class Sample(NamedTuple):
    comm: str
    pid: Optional[int]
    tid: int
    ns: int
    event: str
    rest: str
    frames: list[Frame]  # leaf first
    period: int


def parse(path: str) -> Iterator[Sample]:
    """Yield one Sample per sample of the `perf script` text in PATH."""
    sample: Optional[Sample] = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if sample:
                    yield sample
                sample = None
                continue
            if sample is not None and line[:1] in (" ", "\t"):
                m = FRAME.match(line)
                if m:
                    ip = int(m.group("ip"), 16)
                    sample.frames.append(Frame(ip, m.group("sym"), m.group("dso") or ""))
                    continue
            m = HEADER.match(line)
            if not m:
                continue
            if sample:
                yield sample
            frac = m.group("frac").ljust(9, "0")[:9]
            sample = Sample(
                comm=m.group("comm"),
                pid=int(m.group("pid")) if m.group("pid") else None,
                tid=int(m.group("tid")),
                ns=int(m.group("sec")) * 1_000_000_000 + int(frac),
                event=m.group("event"),
                rest=m.group("rest") or "",
                frames=[],
                period=int(m.group("period")) if m.group("period") else 1,
            )
    if sample:
        yield sample


def inline_chains(binary: str, addresses: set[int]) -> dict[int, list[str]]:
    """Map each address to its inline chain, innermost function first."""
    chains: dict[int, list[str]] = {}
    if not addresses:
        return chains
    text = "\n".join(f"0x{a:x}" for a in sorted(addresses)) + "\n"
    try:
        out = subprocess.run(
            ["addr2line", "-e", binary, "-f", "-i", "-a"],
            input=text,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except FileNotFoundError:
        sys.exit("error: addr2line is not on PATH. Install binutils.")
    except subprocess.CalledProcessError as err:
        sys.stderr.write(err.stderr or "")
        sys.exit(
            f"error: addr2line failed on {binary} with exit code {err.returncode}. "
            "Check that the cell keeps the binary under symfs/ "
            "and that the binary has DWARF records."
        )
    current: Optional[int] = None
    expect_name = True
    for line in out.splitlines():
        if line.startswith("0x") and expect_name:
            current = int(line, 16)
            chains[current] = []
            continue
        if current is None:
            continue
        if expect_name:
            chains[current].append(line)
            expect_name = False
        else:
            expect_name = True
    return chains


def frame_names(
    frame: Frame, depth: int, dso_name: str, chains: dict[int, list[str]]
) -> list[str]:
    """Return the names of one frame, outermost first.

    A frame of DSO expands into its inline chain.
    """
    names: list[str] = []
    if frame.dso == dso_name:
        names = chains.get(frame.ip if depth == 0 else frame.ip - 1, [])
    names = [n for n in names if n != "??"]
    if names:
        return list(reversed(names))
    if frame.sym == "[unknown]" and frame.dso:
        return [f"[{os.path.basename(frame.dso)}]"]
    return [SYMOFF.sub("", frame.sym)]


def wanted_addresses(stacks, dso_name: str) -> set[int]:
    """Return the addresses of the frames of DSO to look up.

    A caller frame holds a return address, so the lookup uses the address minus one.
    """
    wanted: set[int] = set()
    for frames in stacks:
        for depth, frame in enumerate(frames):
            if frame.dso == dso_name:
                wanted.add(frame.ip if depth == 0 else frame.ip - 1)
    return wanted


def shorten_name(name: str) -> str:
    """Apply `shorten_names` of profile.sh.

    It drops `__anon_N`, and the module paths inside parentheses.
    """
    name = ANON.sub("", name)
    prev = None
    while prev != name:
        prev = name
        name = PAREN.sub(lambda m: "\x01" + MODULE_PREFIX.sub("", m.group(1)) + "\x02", name)
    return name.replace("\x01", "(").replace("\x02", ")")


def fold(raw: str, dso_name: str, binary: str, out_path: str) -> None:
    # One pass sums the period of each distinct stack, so memory grows with
    # the distinct stacks, not with the samples.
    stacks: Counter[tuple[str, Optional[int], int, tuple[Frame, ...]]] = Counter()
    for s in parse(raw):
        stacks[(s.comm, s.pid, s.tid, tuple(s.frames))] += s.period
    wanted = wanted_addresses((key[3] for key in stacks), dso_name)
    chains = inline_chains(binary, wanted)
    folded: dict[str, int] = defaultdict(int)
    for (comm, pid, tid, frames), period in stacks.items():
        path = [f"{comm}-{pid if pid is not None else '?'}/{tid}"]
        for depth in range(len(frames) - 1, -1, -1):
            path.extend(frame_names(frames[depth], depth, dso_name, chains))
        folded[";".join(path)] += period
    with open(out_path, "w", encoding="utf-8") as out:
        for key, weight in sorted(folded.items()):
            out.write(f"{key} {weight}\n")


def varint(v: int) -> bytes:
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if not v:
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)


def field_varint(field: int, v: int) -> bytes:
    return varint(field << 3) + varint(v)


def field_bytes(field: int, data: bytes | str) -> bytes:
    if isinstance(data, str):
        data = data.encode()
    return varint(field << 3 | 2) + varint(len(data)) + data


def packet(body: bytes) -> bytes:
    return field_bytes(1, body)  # Trace.packet


def samples(raw: str, dso_name: str, binary: str, out_path: str) -> None:
    """Write the samples of RAW as a Perfetto trace. The module docstring has the layout."""
    # Pass one finds the distinct stacks, so memory grows with them, not with the samples.
    distinct: set[tuple[Frame, ...]] = set()
    threads: dict[int, tuple[int, str]] = {}  # tid -> (pid, comm), the last sample wins
    for s in parse(raw):
        distinct.add(tuple(s.frames))
        threads[s.tid] = (s.pid if s.pid is not None else s.tid, s.comm)
    chains = inline_chains(binary, wanted_addresses(distinct, dso_name))

    name_ids: dict[str, int] = {}  # one id per name serves as function id and as frame id
    callstack_ids: dict[tuple[Frame, ...], int] = {}
    callstacks: list[bytes] = []
    for frames in sorted(distinct, key=lambda fs: [(f.ip, f.sym, f.dso) for f in fs]):
        ids: list[int] = []
        for depth in range(len(frames) - 1, -1, -1):
            for name in frame_names(frames[depth], depth, dso_name, chains):
                ids.append(name_ids.setdefault(shorten_name(name), len(name_ids) + 1))
        callstack_ids[frames] = len(callstack_ids) + 1
        callstacks.append(
            field_bytes(
                7,
                field_varint(1, callstack_ids[frames])
                + b"".join(field_varint(2, i) for i in ids),
            )
        )
    interned = (
        b"".join(
            field_bytes(5, field_varint(1, i) + field_bytes(2, n)) for n, i in name_ids.items()
        )
        + b"".join(
            field_bytes(6, field_varint(1, i) + field_varint(2, i)) for i in name_ids.values()
        )
        + b"".join(callstacks)
    )

    mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    clocks = (
        field_bytes(1, field_varint(1, CLOCK_MONOTONIC) + field_varint(2, mono))
        + field_bytes(1, field_varint(1, CLOCK_BOOTTIME) + field_varint(2, boot))
        + field_varint(2, CLOCK_MONOTONIC)
    )
    tree = b"".join(
        field_bytes(2, field_varint(1, tid) + field_bytes(2, comm) + field_varint(3, pid))
        for tid, (pid, comm) in sorted(threads.items())
    )
    seq = field_varint(10, SAMPLE_SEQUENCE)
    with open(out_path, "wb") as out:
        out.write(packet(seq + field_bytes(6, clocks)))
        out.write(
            packet(
                seq
                + field_varint(13, SEQ_CLEARED)
                + field_bytes(59, field_varint(58, CLOCK_MONOTONIC))
                + field_bytes(12, interned)
            )
        )
        out.write(packet(seq + field_bytes(2, tree)))
        for s in parse(raw):
            perf_sample = (
                field_varint(1, 0)  # the perf script text has no CPU column
                + field_varint(2, s.pid if s.pid is not None else s.tid)
                + field_varint(3, s.tid)
                + field_varint(4, callstack_ids[tuple(s.frames)])
            )
            out.write(
                packet(
                    field_varint(8, s.ns)
                    + seq
                    + field_varint(13, SEQ_NEEDS_STATE)
                    + field_bytes(66, perf_sample)
                )
            )


def new_node() -> dict:
    return {"total": 0, "self": 0, "kids": {}}


def add_path(root: dict, frames: list[str], weight: int) -> None:
    node = root
    node["total"] += weight
    for frame in frames:
        node = node["kids"].setdefault(frame, new_node())
        node["total"] += weight
    node["self"] += weight


def write_tree(out, node: dict, grand: int, depth: int, show_self: bool) -> None:
    for name, kid in sorted(node["kids"].items(), key=lambda kv: -kv[1]["total"]):
        if kid["total"] < grand * TREE_CUT:
            continue
        share = f"{100.0 * kid['total'] / grand:6.2f}%"
        if show_self:
            share += f" {100.0 * kid['self'] / grand:6.2f}%"
        out.write(f"{share}  {'  ' * depth}{name}\n")
        write_tree(out, kid, grand, depth + 1, show_self)


def tree(folded: str, callees_path: str, callers_path: str) -> None:
    down = new_node()
    # frame -> reversed caller path -> weight
    leaf_paths: dict[str, dict[tuple[str, ...], int]] = defaultdict(lambda: defaultdict(int))
    with open(folded, encoding="utf-8", errors="replace") as f:
        for line in f:
            stack, _, weight = line.rstrip("\n").rpartition(" ")
            if not stack or not weight.isdigit():
                continue
            frames = stack.split(";")
            frames[0] = THREAD_ID.sub("", frames[0])
            add_path(down, frames, int(weight))
            leaf_paths[frames[-1]][tuple(reversed(frames[:-1]))] += int(weight)
    grand = down["total"] or 1
    with open(callees_path, "w", encoding="utf-8") as out:
        out.write(
            "# The call tree from the thread roots down. "
            "Columns: inclusive share, self share, frame.\n"
        )
        out.write(
            f"# Total weight {grand} (ns of CPU samples). "
            f"Branches under {100 * TREE_CUT}% are dropped.\n"
        )
        write_tree(out, down, grand, 0, True)
    with open(callers_path, "w", encoding="utf-8") as out:
        out.write(
            "# One tree per hot frame, by self share, upward through its callers. "
            "Column: share of the total.\n"
        )
        for leaf, paths in sorted(leaf_paths.items(), key=lambda kv: -sum(kv[1].values())):
            own = sum(paths.values())
            if own < grand * TREE_CUT:
                continue
            up = new_node()
            for path, weight in paths.items():
                add_path(up, list(path), weight)
            out.write(f"\n{100.0 * own / grand:6.2f}%  {leaf}\n")
            write_tree(out, up, grand, 1, False)


def offcpu(raw: str, pid: str, out_path: str) -> None:
    target = int(pid)
    pending: dict[int, tuple[int, str]] = {}  # tid -> (ns, folded stack)
    folded: dict[str, int] = defaultdict(int)
    for s in parse(raw):
        syms = [f.sym for f in s.frames]
        m = SWITCH_FIELDS.search(s.rest) or SWITCH_PRETTY.match(s.rest.strip())
        if not m:
            continue
        nxt = int(m.group("next"))
        if nxt in pending:
            start, key = pending.pop(nxt)
            folded[key] += (s.ns - start) // 1000
        if s.pid == target:
            state = f"[{m.group('state')}]"
            stack = ";".join([m.group("prev_comm"), *reversed(syms), state])
            pending[int(m.group("prev"))] = (s.ns, stack.replace(" ", "_"))
    with open(out_path, "w", encoding="utf-8") as out:
        for key, us in sorted(folded.items()):
            if us > 0:
                out.write(f"{key} {us}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fold", help="fold CPU samples into stacks with inlined frames")
    p.add_argument("raw")
    p.add_argument("dso")
    p.add_argument("file")
    p.add_argument("out")
    p = sub.add_parser("samples", help="write the CPU samples as a Perfetto trace")
    p.add_argument("raw")
    p.add_argument("dso")
    p.add_argument("file")
    p.add_argument("out")
    p = sub.add_parser("tree", help="write the callee and the caller trees")
    p.add_argument("folded")
    p.add_argument("callees")
    p.add_argument("callers")
    p = sub.add_parser("offcpu", help="fold the wait time of one process")
    p.add_argument("raw")
    p.add_argument("pid")
    p.add_argument("out")
    args = ap.parse_args()
    if args.command == "fold":
        fold(args.raw, args.dso, args.file, args.out)
    elif args.command == "samples":
        samples(args.raw, args.dso, args.file, args.out)
    elif args.command == "tree":
        tree(args.folded, args.callees, args.callers)
    else:
        offcpu(args.raw, args.pid, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
