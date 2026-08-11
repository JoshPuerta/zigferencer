#!/usr/bin/env python3
"""Render bench_threads_sweep.sh JSONL into a scaling table.

For each cell: mean/stddev ITL+TTFT over runs, tok/s (1000/itl_mean), scaling
efficiency vs the same-backend 1-thread cell, whole-process CPU% (cores used),
and the token hash (must be invariant within a backend/model = determinism gate).

Usage: bench_sweep_report.py FILE.jsonl [--bytes-per-token GB]
       --bytes-per-token adds a derived GB/s column = GB * tok/s.
"""
import json, sys, statistics, argparse

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--bytes-per-token", type=float, default=0.0,
                help="weight GB streamed per decode token; adds derived GB/s")
ap.add_argument("--group", default="backend", help="primary group key (backend|model)")
args = ap.parse_args()

cells = []
with open(args.file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        b = o["bench"]
        runs = b["runs"]
        itls = [r["itl_ms"] for r in runs]
        ttfts = [r["ttft_ms"] for r in runs]
        hashes = {r["tokens_hash"] for r in runs}
        cells.append({
            "model": b["model"], "backend": b["backend"], "threads": b["threads"],
            "flops": o["min_parallel_flops"],
            "itl_mean": statistics.mean(itls),
            "itl_min": min(itls),
            "itl_std": statistics.pstdev(itls) if len(itls) > 1 else 0.0,
            "ttft_mean": statistics.mean(ttfts),
            "cpu_pct": o["time"].get("cpu_pct", "?"),
            "wall_s": o["time"].get("wall_s", 0),
            "hash": ("MIXED:" + ",".join(sorted(hashes))) if len(hashes) > 1 else next(iter(hashes)),
        })

# baseline ITL per (group, model) at min thread count for scaling efficiency.
gkey = args.group
base = {}
for c in cells:
    k = (c[gkey], c["model"])
    if k not in base or c["threads"] < base[k][0]:
        base[k] = (c["threads"], c["itl_mean"])

hdr = f"{'model':<28} {'backend':<8} {'flops':>10} {'thr':>3} {'itl_ms':>9} {'±std':>6} {'tok/s':>7} {'scal':>6} {'eff%':>5} {'CPU%':>6}"
if args.bytes_per_token:
    hdr += f" {'GB/s':>6}"
hdr += "  hash"
print(hdr)
print("-" * len(hdr))
last = None
# A sweep without a floor flag records "auto". It sorts before every number.
for c in sorted(cells, key=lambda c: (c["model"], c[gkey], (c["flops"] != "auto", c["flops"] if c["flops"] != "auto" else 0), c["threads"])):
    tok = 1000.0 / c["itl_mean"]
    k = (c[gkey], c["model"])
    bt, bi = base[k]
    btok = 1000.0 / bi
    scal = tok / btok
    eff = 100.0 * scal / (c["threads"] / bt) if bt else 0.0
    grp = (c["model"], c[gkey], c["flops"])
    if last is not None and grp != last:
        print()
    last = grp
    row = (f"{c['model']:<28} {c['backend']:<8} {c['flops']:>10} {c['threads']:>3} "
           f"{c['itl_mean']:>9.2f} {c['itl_std']:>6.2f} {tok:>7.2f} {scal:>5.2f}x {eff:>5.0f} {c['cpu_pct']:>6}")
    if args.bytes_per_token:
        row += f" {args.bytes_per_token*tok:>6.1f}"
    row += f"  {c['hash']}"
    print(row)
