#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Convert `perf script` text from tools/profile.sh into views of a run.

The views are folded stacks for flamegraph.pl, a Perfetto trace of the samples, and the caller
and callee trees. Run a command with --help for its inputs and outputs. To add a command, write
its function, then add its parser in `parser()` and its case in `main()`.
"""

import argparse
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TextIO

# A sample is a header line, one frame line per call with the leaf first, and a blank line.
HEADER = re.compile(
    r"^\s*(?P<comm>.*?)\s+(?:(?P<pid>\d+)/)?(?P<tid>\d+)\s+"
    r"(?P<sec>\d+)\.(?P<frac>\d+):\s+(?:(?P<period>\d+)\s+)?"
    r"(?P<event>\S+):(?:\s+(?P<rest>.*))?$"
)
FRAME = re.compile(r"^\s+(?P<ip>[0-9a-f]+)\s+(?P<sym>.+?)(?:\s+\((?P<dso>[^()]*)\))?$")
# perf prints a switch as the raw tracepoint fields, or in the libtraceevent form
# `prev_comm:prev_pid [prio] STATE ==> next_comm:next_pid [prio]`.
SWITCH_FIELDS = re.compile(
    r"prev_comm=(?P<prev_comm>.*?) prev_pid=(?P<prev>\d+) .*?prev_state=(?P<state>\S+) "
    r"==> next_comm=.*? next_pid=(?P<next>\d+)"
)
SWITCH_PRETTY = re.compile(
    r"^(?P<prev_comm>.+):(?P<prev>\d+) \[-?\d+\] (?P<state>\S+) ==> .+:(?P<next>\d+) \[-?\d+\]$"
)

SYMOFF = re.compile(r"\+0x[0-9a-f]+$")
THREAD_ID = re.compile(r"-[?\d]+/\d+$")
ANON = re.compile(r"__anon_\d+")
PAREN = re.compile(r"\(([^()]*)\)")
MODULE_PREFIX = re.compile(r"\b(?:[a-z_][a-z0-9_]*\.)+(?=[A-Za-z_])")

TREE_CUT = 0.005
SAMPLE_SEQUENCE = 2


# The Perfetto field numbers and values come from the protos of Perfetto v59.1.
class Trace(IntEnum):
    PACKET = 1


class Packet(IntEnum):
    PROCESS_TREE = 2
    CLOCK_SNAPSHOT = 6
    TIMESTAMP = 8
    SEQUENCE_ID = 10
    INTERNED_DATA = 12
    SEQUENCE_FLAGS = 13
    DEFAULTS = 59
    PERF_SAMPLE = 66


class Defaults(IntEnum):
    TIMESTAMP_CLOCK_ID = 58


class ClockSnapshot(IntEnum):
    CLOCKS = 1
    PRIMARY_TRACE_CLOCK = 2


class Clock(IntEnum):
    ID = 1
    TIMESTAMP = 2


class ClockId(IntEnum):
    MONOTONIC = 3
    BOOTTIME = 6


class SequenceFlag(IntEnum):
    INCREMENTAL_STATE_CLEARED = 1
    NEEDS_INCREMENTAL_STATE = 2


class ProcessTree(IntEnum):
    THREADS = 2


class Thread(IntEnum):
    TID = 1
    NAME = 2
    TGID = 3


class InternedData(IntEnum):
    FUNCTION_NAMES = 5
    FRAMES = 6
    CALLSTACKS = 7


class Interned(IntEnum):
    """The fields of an interned string, frame, or callstack: its id, then its value."""

    IID = 1
    VALUE = 2


class PerfSample(IntEnum):
    CPU = 1
    PID = 2
    TID = 3
    CALLSTACK_IID = 4


@dataclass(frozen=True, slots=True)
class Frame:
    ip: int
    sym: str
    dso: str


@dataclass(frozen=True, slots=True)
class Sample:
    comm: str
    pid: int | None
    tid: int
    ns: int
    event: str
    rest: str
    frames: list[Frame]  # leaf first
    period: int


@dataclass(slots=True)
class Node:
    total: int = 0
    own: int = 0
    kids: dict[str, "Node"] = field(default_factory=dict)


def parse(path: str) -> Iterator[Sample]:
    sample: Sample | None = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if sample is not None:
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
            if sample is not None:
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
    if sample is not None:
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
    # addr2line prints the address, then a function line and a location line per inline scope.
    current: int | None = None
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


def lookup_address(frame: Frame, depth: int) -> int:
    # A caller frame holds a return address, which can lie in the next inline scope.
    return frame.ip if depth == 0 else frame.ip - 1


def frame_names(
    frame: Frame, depth: int, dso_name: str, chains: dict[int, list[str]]
) -> list[str]:
    """Return the names of one frame, outermost first. A frame of the DSO expands inline."""
    names: list[str] = []
    if frame.dso == dso_name:
        names = chains.get(lookup_address(frame, depth), [])
    names = [n for n in names if n != "??"]
    if names:
        return list(reversed(names))
    if frame.sym == "[unknown]" and frame.dso:
        return [f"[{frame.dso.rpartition('/')[2]}]"]
    return [SYMOFF.sub("", frame.sym)]


def wanted_addresses(stacks: Iterable[Sequence[Frame]], dso_name: str) -> set[int]:
    return {
        lookup_address(frame, depth)
        for frames in stacks
        for depth, frame in enumerate(frames)
        if frame.dso == dso_name
    }


def shorten_name(name: str) -> str:
    """Apply `shorten_names` of profile.sh: drop `__anon_N` and the module paths in parentheses."""
    name = ANON.sub("", name)
    # The markers hide each finished pair from PAREN, so the next pass reaches the outer pair.
    prev = None
    while prev != name:
        prev = name
        name = PAREN.sub(lambda m: "\x01" + MODULE_PREFIX.sub("", m.group(1)) + "\x02", name)
    return name.replace("\x01", "(").replace("\x02", ")")


def fold(raw: str, dso_name: str, binary: str, out_path: str) -> None:
    # The sum per distinct stack keeps the memory in proportion to the stacks, not the samples.
    stacks: Counter[tuple[str, int | None, int, tuple[Frame, ...]]] = Counter()
    for s in parse(raw):
        stacks[(s.comm, s.pid, s.tid, tuple(s.frames))] += s.period
    chains = inline_chains(binary, wanted_addresses((key[3] for key in stacks), dso_name))
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
    """Encode V as a protobuf varint. A negative V takes its 64-bit two's complement form."""
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if not v:
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)


def field_varint(number: int, v: int) -> bytes:
    return varint(number << 3) + varint(v)


def field_bytes(number: int, data: bytes | str) -> bytes:
    if isinstance(data, str):
        data = data.encode()
    return varint(number << 3 | 2) + varint(len(data)) + data


def packet(body: bytes) -> bytes:
    return field_bytes(Trace.PACKET, body)


def samples(raw: str, dso_name: str, binary: str, out_path: str) -> None:
    # Pass one keeps the distinct stacks alone, so the memory does not grow with the samples.
    distinct: set[tuple[Frame, ...]] = set()
    threads: dict[int, tuple[int, str]] = {}  # tid to (pid, comm), the last sample wins
    for s in parse(raw):
        distinct.add(tuple(s.frames))
        threads[s.tid] = (s.pid if s.pid is not None else s.tid, s.comm)
    chains = inline_chains(binary, wanted_addresses(distinct, dso_name))

    name_ids: dict[str, int] = {}  # one id per name is the function id and the frame id
    callstack_ids: dict[tuple[Frame, ...], int] = {}
    callstacks: list[bytes] = []
    for frames in sorted(distinct, key=lambda fs: [(f.ip, f.sym, f.dso) for f in fs]):
        ids: list[int] = []
        for depth in range(len(frames) - 1, -1, -1):
            for name in frame_names(frames[depth], depth, dso_name, chains):
                ids.append(name_ids.setdefault(shorten_name(name), len(name_ids) + 1))
        callstack_ids[frames] = len(callstack_ids) + 1
        callstack = field_varint(Interned.IID, callstack_ids[frames]) + b"".join(
            field_varint(Interned.VALUE, i) for i in ids
        )
        callstacks.append(field_bytes(InternedData.CALLSTACKS, callstack))
    interned = (
        b"".join(
            field_bytes(
                InternedData.FUNCTION_NAMES,
                field_varint(Interned.IID, i) + field_bytes(Interned.VALUE, n),
            )
            for n, i in name_ids.items()
        )
        + b"".join(
            field_bytes(
                InternedData.FRAMES,
                field_varint(Interned.IID, i) + field_varint(Interned.VALUE, i),
            )
            for i in name_ids.values()
        )
        + b"".join(callstacks)
    )

    mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    clocks = (
        field_bytes(
            ClockSnapshot.CLOCKS,
            field_varint(Clock.ID, ClockId.MONOTONIC) + field_varint(Clock.TIMESTAMP, mono),
        )
        + field_bytes(
            ClockSnapshot.CLOCKS,
            field_varint(Clock.ID, ClockId.BOOTTIME) + field_varint(Clock.TIMESTAMP, boot),
        )
        + field_varint(ClockSnapshot.PRIMARY_TRACE_CLOCK, ClockId.MONOTONIC)
    )
    tree = b"".join(
        field_bytes(
            ProcessTree.THREADS,
            field_varint(Thread.TID, tid)
            + field_bytes(Thread.NAME, comm)
            + field_varint(Thread.TGID, pid),
        )
        for tid, (pid, comm) in sorted(threads.items())
    )
    seq = field_varint(Packet.SEQUENCE_ID, SAMPLE_SEQUENCE)
    with open(out_path, "wb") as out:
        out.write(packet(seq + field_bytes(Packet.CLOCK_SNAPSHOT, clocks)))
        out.write(
            packet(
                seq
                + field_varint(Packet.SEQUENCE_FLAGS, SequenceFlag.INCREMENTAL_STATE_CLEARED)
                + field_bytes(
                    Packet.DEFAULTS, field_varint(Defaults.TIMESTAMP_CLOCK_ID, ClockId.MONOTONIC)
                )
                + field_bytes(Packet.INTERNED_DATA, interned)
            )
        )
        out.write(packet(seq + field_bytes(Packet.PROCESS_TREE, tree)))
        for s in parse(raw):
            perf_sample = (
                field_varint(PerfSample.CPU, 0)  # the perf script text has no CPU column
                + field_varint(PerfSample.PID, s.pid if s.pid is not None else s.tid)
                + field_varint(PerfSample.TID, s.tid)
                + field_varint(PerfSample.CALLSTACK_IID, callstack_ids[tuple(s.frames)])
            )
            out.write(
                packet(
                    field_varint(Packet.TIMESTAMP, s.ns)
                    + seq
                    + field_varint(Packet.SEQUENCE_FLAGS, SequenceFlag.NEEDS_INCREMENTAL_STATE)
                    + field_bytes(Packet.PERF_SAMPLE, perf_sample)
                )
            )


def add_path(root: Node, frames: Iterable[str], weight: int) -> None:
    node = root
    node.total += weight
    for frame in frames:
        node = node.kids.setdefault(frame, Node())
        node.total += weight
    node.own += weight


def write_tree(out: TextIO, node: Node, grand: int, depth: int, show_self: bool) -> None:
    for name, kid in sorted(node.kids.items(), key=lambda kv: -kv[1].total):
        if kid.total < grand * TREE_CUT:
            continue
        share = f"{100.0 * kid.total / grand:6.2f}%"
        if show_self:
            share += f" {100.0 * kid.own / grand:6.2f}%"
        out.write(f"{share}  {'  ' * depth}{name}\n")
        write_tree(out, kid, grand, depth + 1, show_self)


def tree(folded: str, callees_path: str, callers_path: str) -> None:
    down = Node()
    # Each leaf frame maps its caller paths, nearest caller first, to a weight.
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
    grand = down.total or 1
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
            leaf_total = sum(paths.values())
            if leaf_total < grand * TREE_CUT:
                continue
            up = Node()
            for path, weight in paths.items():
                add_path(up, path, weight)
            out.write(f"\n{100.0 * leaf_total / grand:6.2f}%  {leaf}\n")
            write_tree(out, up, grand, 1, False)


def offcpu(raw: str, pid: int, out_path: str) -> None:
    switched_out: dict[int, tuple[int, str]] = {}  # tid to (ns, folded stack)
    folded: dict[str, int] = defaultdict(int)
    for s in parse(raw):
        m = SWITCH_FIELDS.search(s.rest) or SWITCH_PRETTY.match(s.rest.strip())
        if not m:
            continue
        nxt = int(m.group("next"))
        if nxt in switched_out:
            start, key = switched_out.pop(nxt)
            folded[key] += (s.ns - start) // 1000
        if s.pid == pid:
            syms = [f.sym for f in reversed(s.frames)]
            stack = ";".join([m.group("prev_comm"), *syms, f"[{m.group('state')}]"])
            switched_out[int(m.group("prev"))] = (s.ns, stack.replace(" ", "_"))
    with open(out_path, "w", encoding="utf-8") as out:
        for key, us in sorted(folded.items()):
            if us > 0:
                out.write(f"{key} {us}\n")


def add_positionals(p: argparse.ArgumentParser, **helps: str) -> None:
    for name, text in helps.items():
        p.add_argument(name, metavar=name.upper(), help=text)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    expansion = (
        "Each frame of DSO expands into its inline chain from the DWARF records of FILE, "
        "through one addr2line call. A caller frame holds a return address, so the lookup "
        "uses that address minus one."
    )
    for name, short, long in (
        (
            "fold",
            "fold CPU samples into stacks with inlined frames",
            "Write folded stacks of the CPU samples in RAW for flamegraph.pl: root first, then "
            "the sample period as the weight.",
        ),
        (
            "samples",
            "write the CPU samples as a Perfetto trace",
            "Write the CPU samples in RAW as a Perfetto proto trace: one packet per sample with "
            "its thread and call stack, plus the thread names. The times stay on "
            "CLOCK_MONOTONIC, the clock of `perf record -k mono`. The trace uses packet "
            f"sequence {SAMPLE_SEQUENCE} and the engine spans use sequence 1, so "
            "`cat spans samples` makes one valid trace.",
        ),
    ):
        p = sub.add_parser(name, help=short, description=f"{long} {expansion}")
        add_positionals(
            p,
            raw="`perf script` text of CPU samples with call chains",
            dso="the path of the binary as perf names it",
            file="a copy of that binary with DWARF records",
            out="the output file",
        )

    p = sub.add_parser(
        "tree",
        help="write the callee and the caller trees",
        description="Read folded stacks and write two trees. The first frame names the thread, "
        "and the trees keep only the thread name, so equal threads merge. CALLEES holds the "
        "tree from the roots down, with the inclusive and self share of each frame. CALLERS "
        "holds one tree per hot frame, upward through its callers. Both drop the branches under "
        f"{100 * TREE_CUT} percent of the total weight.",
    )
    add_positionals(
        p,
        folded="folded stacks: one stack per line, then a weight",
        callees="the output file of the callee tree",
        callers="the output file of the caller trees",
    )

    p = sub.add_parser(
        "offcpu",
        help="fold the wait time of one process",
        description="Write folded stacks of the time that the threads of PID wait off the CPU, "
        "in microseconds, for flamegraph.pl. A wait lasts from the switch out of a thread to "
        "its next switch in.",
    )
    add_positionals(
        p,
        raw="`perf script` text of sched:sched_switch events with call chains",
        pid="the process to fold",
        out="the output file",
    )
    return ap


def main() -> int:
    args = parser().parse_args()
    match args.command:
        case "fold":
            fold(args.raw, args.dso, args.file, args.out)
        case "samples":
            samples(args.raw, args.dso, args.file, args.out)
        case "tree":
            tree(args.folded, args.callees, args.callers)
        case "offcpu":
            offcpu(args.raw, int(args.pid), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
