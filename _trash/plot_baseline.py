#!/usr/bin/env python3
import csv, sys
from collections import defaultdict

def f(x, default=None):
    try: return float(x)
    except: return default

def main():
    if len(sys.argv) != 2:
        print("usage: python3 summarize_results.py results.csv", file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    rows = []
    with open(path, newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            row["n"] = int(row["n"])
            row["threads"] = int(row["threads"])
            row["gflops"] = f(row["gflops"], 0.0)
            row["sec"] = f(row["sec"], 0.0)
            row["sec_per_iter"] = f(row["sec_per_iter"], 0.0)
            rows.append(row)

    # baseline GFLOPs at T=1 for each (impl,n)
    base = {}
    for row in rows:
        if row["threads"] == 1:
            base[(row["impl"], row["n"])] = row["gflops"]

    # write long summary
    long_out = "summary_long.csv"
    with open(long_out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["impl","n","threads","gflops","sec","sec_per_iter","speedup_vs_t1"])
        for row in sorted(rows, key=lambda x:(x["impl"], x["n"], x["threads"])):
            b = base.get((row["impl"], row["n"]), None)
            sp = (row["gflops"]/b) if (b and b > 0) else ""
            w.writerow([row["impl"], row["n"], row["threads"], row["gflops"], row["sec"], row["sec_per_iter"], sp])

    # pivot speedup
    piv = defaultdict(dict)
    threads_set = set()
    for row in rows:
        b = base.get((row["impl"], row["n"]), None)
        if not (b and b > 0): 
            continue
        piv[(row["impl"], row["n"])][row["threads"]] = row["gflops"]/b
        threads_set.add(row["threads"])

    ths = sorted(threads_set)
    piv_out = "summary_pivot_speedup.csv"
    with open(piv_out, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["impl","n"] + [f"T{t}" for t in ths])
        for (impl,n) in sorted(piv.keys()):
            w.writerow([impl,n] + [piv[(impl,n)].get(t,"") for t in ths])

    print("Wrote:", long_out, "and", piv_out)

if __name__ == "__main__":
    main()
