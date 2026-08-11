#!/usr/bin/env python3
"""Convert `perf script` text from tools/profile.sh into views of a run.

calls RAW PROBES OUTDIR
    RAW holds uprobe hits: an entry event `zf:fN` and a return event
    `zf:fN_ret` per call. PROBES maps each `fN` to its full symbol name.
    The entry and the return of one call happen on one thread, and calls on
    one thread nest, so a stack per thread pairs them. Writes calls.txt (the
    call tree of each thread in time order), calls.json (Chrome trace events
    for ui.perfetto.dev), and counts.txt (calls and time per function).

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
    one.
    Writes folded stacks, root first, weighted by the sample period.

tree FOLDED CALLEES CALLERS
    FOLDED holds one stack per line, root first, then a weight. The first
    frame names the thread, and the tree keeps only the thread name, so equal
    threads merge. CALLEES gets the tree from the roots down: each frame with
    its inclusive and self share, children by weight. CALLERS gets one tree
    per hot frame, upward: who called it, by weight. Both drop branches under
    0.5 percent of the total.

It uses the Python standard library alone.
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

HEADER = re.compile(r"^\s*(?P<comm>.*?)\s+(?:(?P<pid>\d+)/)?(?P<tid>\d+)\s+(?P<sec>\d+)\.(?P<frac>\d+):\s+(?:(?P<period>\d+)\s+)?(?P<event>\S+):(?:\s+(?P<rest>.*))?$")
FRAME = re.compile(r"^\s+(?P<ip>[0-9a-f]+)\s+(?P<sym>.+?)(?:\s+\((?P<dso>[^()]*)\))?$")


def parse(path):
    """Yield (comm, pid, tid, ns, event, rest, frames, period) per sample.

    Frames are (ip, symbol, dso) and run leaf first.
    """
    sample = None
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
                    sample[6].append((int(m.group("ip"), 16), m.group("sym"), m.group("dso") or ""))
                    continue
            m = HEADER.match(line)
            if not m:
                continue
            if sample:
                yield sample
            ns = int(m.group("sec")) * 1_000_000_000 + int(m.group("frac").ljust(9, "0")[:9])
            pid = int(m.group("pid")) if m.group("pid") else None
            period = int(m.group("period")) if m.group("period") else 1
            sample = (m.group("comm"), pid, int(m.group("tid")), ns, m.group("event"), m.group("rest") or "", [], period)
    if sample:
        yield sample


def calls(raw, probes, outdir):
    names = {}
    with open(probes, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 2)
            if len(parts) == 3:
                names[parts[1]] = parts[2]

    open_calls = defaultdict(list)  # tid -> [(event, start_ns, depth, stack)]
    finished = []  # (start_ns, tid, depth, name, dur_ns, caller_stack)
    events = []
    thread_names = {}
    unmatched = 0
    t0 = None
    for comm, _pid, tid, ns, event, _rest, frames, _period in parse(raw):
        frames = [sym for _ip, sym, _dso in frames]
        t0 = ns if t0 is None else min(t0, ns)
        thread_names.setdefault(tid, comm)
        probe = event.split(":")[-1]
        if probe.endswith("_ret"):
            base = probe[: -len("_ret")]
            stack = open_calls[tid]
            while stack and stack[-1][0] != base:
                stack.pop()
                unmatched += 1
            if not stack:
                unmatched += 1
                continue
            _, start, depth, callers = stack.pop()
            name = names.get(base, base)
            finished.append((start, tid, depth, name, ns - start, callers))
            events.append({"name": name, "ph": "X", "ts": start / 1000.0, "dur": (ns - start) / 1000.0, "pid": 1, "tid": tid})
        else:
            stack = open_calls[tid]
            stack.append((probe, ns, len(stack), frames[1:6]))
    for stack in open_calls.values():
        unmatched += len(stack)

    finished.sort(key=lambda c: (c[1], c[0], c[2]))
    with open(f"{outdir}/calls.txt", "w", encoding="utf-8") as out:
        out.write(f"# one line per call, per thread, in time order. Indentation is the nesting of probed calls.\n")
        out.write(f"# columns: start (ms from the first event), duration (us), call. Unmatched entries or returns: {unmatched}.\n")
        current = None
        for start, tid, depth, name, dur, callers in finished:
            if tid != current:
                current = tid
                out.write(f"\n## thread {tid} {thread_names.get(tid, '')}\n")
            line = f"{(start - t0) / 1e6:12.3f} {dur / 1e3:12.1f}  {'  ' * depth}{name}"
            if depth == 0 and callers:
                line += "   <- " + " <- ".join(callers[:3])
            out.write(line + "\n")

    for tid, comm in thread_names.items():
        events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": f"{comm} {tid}"}})
    with open(f"{outdir}/calls.json", "w", encoding="utf-8") as out:
        json.dump({"traceEvents": events, "displayTimeUnit": "ns", "otherData": {"unmatched": unmatched}}, out)

    totals = defaultdict(lambda: [0, 0, 0])  # name -> [count, total_ns, max_ns]
    for _start, _tid, _depth, name, dur, _callers in finished:
        t = totals[name]
        t[0] += 1
        t[1] += dur
        t[2] = max(t[2], dur)
    with open(f"{outdir}/counts.txt", "w", encoding="utf-8") as out:
        out.write(f"{'calls':>10} {'total ms':>12} {'mean us':>10} {'max us':>10}  function (inclusive time)\n")
        for name, (count, total, longest) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
            out.write(f"{count:>10} {total / 1e6:>12.3f} {total / count / 1e3:>10.1f} {longest / 1e3:>10.1f}  {name}\n")


SYMOFF = re.compile(r"\+0x[0-9a-f]+$")


def inline_chains(binary, addresses):
    """Map each address to its inline chain, innermost function first."""
    chains = {}
    if not addresses:
        return chains
    text = "\n".join(f"0x{a:x}" for a in sorted(addresses)) + "\n"
    try:
        out = subprocess.run(["addr2line", "-e", binary, "-f", "-i", "-a"], input=text, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("error: addr2line is not on PATH. Install binutils.")
    except subprocess.CalledProcessError as err:
        sys.stderr.write(err.stderr or "")
        sys.exit(f"error: addr2line failed on {binary} with exit code {err.returncode}. "
                 "Check that the cell keeps the binary under symfs/ and that the binary has DWARF records.")
    current = None
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


def fold(raw, dso_name, binary, out_path):
    # One pass sums the period of each distinct stack, so memory grows with
    # the distinct stacks, not with the samples.
    stacks = Counter()
    for comm, pid, tid, _ns, _event, _rest, frames, period in parse(raw):
        stacks[(comm, pid, tid, tuple(frames))] += period
    wanted = set()
    for (_comm, _pid, _tid, frames) in stacks:
        for depth, (ip, _sym, dso) in enumerate(frames):
            if dso == dso_name:
                wanted.add(ip if depth == 0 else ip - 1)
    chains = inline_chains(binary, wanted)
    folded = defaultdict(int)
    for (comm, pid, tid, frames), period in stacks.items():
        path = [f"{comm}-{pid if pid is not None else '?'}/{tid}"]
        for depth in range(len(frames) - 1, -1, -1):
            ip, sym, dso = frames[depth]
            names = chains.get(ip if depth == 0 else ip - 1, []) if dso == dso_name else []
            names = [n for n in names if n != "??"]
            if names:
                path.extend(reversed(names))
            elif sym == "[unknown]" and dso:
                path.append(f"[{os.path.basename(dso)}]")
            else:
                path.append(SYMOFF.sub("", sym))
        folded[";".join(path)] += period
    with open(out_path, "w", encoding="utf-8") as out:
        for key, weight in sorted(folded.items()):
            out.write(f"{key} {weight}\n")


THREAD_ID = re.compile(r"-[?\d]+/\d+$")
TREE_CUT = 0.005


def new_node():
    return {"total": 0, "self": 0, "kids": {}}


def add_path(root, frames, weight):
    node = root
    node["total"] += weight
    for frame in frames:
        node = node["kids"].setdefault(frame, new_node())
        node["total"] += weight
    node["self"] += weight


def write_tree(out, node, grand, depth, show_self):
    for name, kid in sorted(node["kids"].items(), key=lambda kv: -kv[1]["total"]):
        if kid["total"] < grand * TREE_CUT:
            continue
        share = f"{100.0 * kid['total'] / grand:6.2f}%"
        if show_self:
            share += f" {100.0 * kid['self'] / grand:6.2f}%"
        out.write(f"{share}  {'  ' * depth}{name}\n")
        write_tree(out, kid, grand, depth + 1, show_self)


def tree(folded, callees_path, callers_path):
    down = new_node()
    leaf_paths = defaultdict(lambda: defaultdict(int))  # frame -> reversed caller path -> weight
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
        out.write("# The call tree from the thread roots down. Columns: inclusive share, self share, frame.\n")
        out.write(f"# Total weight {grand} (ns of CPU samples). Branches under {100 * TREE_CUT}% are dropped.\n")
        write_tree(out, down, grand, 0, True)
    with open(callers_path, "w", encoding="utf-8") as out:
        out.write("# One tree per hot frame, by self share, upward through its callers. Column: share of the total.\n")
        for leaf, paths in sorted(leaf_paths.items(), key=lambda kv: -sum(kv[1].values())):
            own = sum(paths.values())
            if own < grand * TREE_CUT:
                continue
            up = new_node()
            for path, weight in paths.items():
                add_path(up, path, weight)
            out.write(f"\n{100.0 * own / grand:6.2f}%  {leaf}\n")
            write_tree(out, up, grand, 1, False)


# perf prints a switch in one of two forms: the raw tracepoint fields, or the
# libtraceevent form `prev_comm:prev_pid [prio] STATE ==> next_comm:next_pid [prio]`.
SWITCH_FIELDS = re.compile(r"prev_comm=(?P<prev_comm>.*?) prev_pid=(?P<prev>\d+) .*?prev_state=(?P<state>\S+) ==> next_comm=.*? next_pid=(?P<next>\d+)")
SWITCH_PRETTY = re.compile(r"^(?P<prev_comm>.+):(?P<prev>\d+) \[-?\d+\] (?P<state>\S+) ==> .+:(?P<next>\d+) \[-?\d+\]$")


def offcpu(raw, pid, out_path):
    target = int(pid)
    pending = {}  # tid -> (ns, folded stack)
    folded = defaultdict(int)
    for _comm, sample_pid, _tid, ns, _event, rest, frames, _period in parse(raw):
        frames = [sym for _ip, sym, _dso in frames]
        m = SWITCH_FIELDS.search(rest) or SWITCH_PRETTY.match(rest.strip())
        if not m:
            continue
        nxt = int(m.group("next"))
        if nxt in pending:
            start, key = pending.pop(nxt)
            folded[key] += (ns - start) // 1000
        if sample_pid == target:
            stack = ";".join([m.group("prev_comm")] + list(reversed(frames)) + [f"[{m.group('state')}]"])
            pending[int(m.group("prev"))] = (ns, stack.replace(" ", "_"))
    with open(out_path, "w", encoding="utf-8") as out:
        for key, us in sorted(folded.items()):
            if us > 0:
                out.write(f"{key} {us}\n")


def main(argv):
    if len(argv) == 5 and argv[1] == "calls":
        calls(argv[2], argv[3], argv[4])
    elif len(argv) == 6 and argv[1] == "fold":
        fold(argv[2], argv[3], argv[4], argv[5])
    elif len(argv) == 5 and argv[1] == "tree":
        tree(argv[2], argv[3], argv[4])
    elif len(argv) == 5 and argv[1] == "offcpu":
        offcpu(argv[2], argv[3], argv[4])
    else:
        sys.exit("usage: profile_convert.py calls RAW PROBES OUTDIR | fold RAW DSO FILE OUT | tree FOLDED CALLEES CALLERS | offcpu RAW PID OUT")


if __name__ == "__main__":
    main(sys.argv)
