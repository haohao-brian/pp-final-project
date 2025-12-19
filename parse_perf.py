#!/usr/bin/env python3
"""
parse_perf.py

Merge perf stat -x, outputs (stderr captured per run) into a single CSV.

Usage:
  python3 parse_perf.py perf_logs merged_perf.csv

It expects files named:
  perf_<impl>_N<n>_T<t>.csv

Each file contains lines like:
  value,unit,event,run,...
We aggregate by event (mean over any repeated lines).
"""
import csv, glob, os, re, sys
from collections import defaultdict

def parse_value(s: str):
    s = s.strip()
    if s in ("", "<not supported>", "<not counted>"):
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

pat = re.compile(r"perf_(?P<impl>[^_]+)_N(?P<n>\d+)_T(?P<t>\d+)\.csv$")

def main():
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    logdir, outcsv = sys.argv[1], sys.argv[2]

    rows = []
    for path in sorted(glob.glob(os.path.join(logdir, "perf_*.csv"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        impl = m.group("impl")
        n = int(m.group("n"))
        t = int(m.group("t"))

        vals = defaultdict(list)
        with open(path, newline="") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                v = parse_value(parts[0])
                event = parts[2]
                if v is None:
                    continue
                vals[event].append(v)

        out = {"impl": impl, "n": n, "threads": t}
        for event, arr in vals.items():
            out[event] = sum(arr) / len(arr) if arr else ""
        rows.append(out)

    # header
    base = ["impl", "n", "threads"]
    all_keys = set(base)
    for r in rows:
        all_keys |= set(r.keys())
    others = sorted(k for k in all_keys if k not in base)
    header = base + others

    with open(outcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {outcsv} with {len(rows)} rows")

if __name__ == "__main__":
    main()
