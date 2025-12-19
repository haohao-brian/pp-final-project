#!/usr/bin/env python3
"""
join_derive.py

Merge results.csv (from bench_csv) + perf_merged.csv (from parse_perf.py)
and derive "bytes & misses profile" columns per FFT.

Usage:
  python3 join_derive.py results.csv perf_merged.csv baseline_profile.csv

Outputs baseline_profile.csv with columns including:
  impl,n,threads,sec_per_iter,gflops,
  cycles_per_fft,instr_per_fft,
  cache_misses_per_fft,LLC_load_misses_per_fft,dTLB_load_misses_per_fft,
  cache_miss_rate,ipc

Notes:
- Pure stdlib (no pandas/matplotlib).
- perf_merged.csv must have impl,n,threads columns.
- Event column names differ across perf versions; we try a few aliases.
"""
import csv, sys, math

ALIASES = {
  "cycles": ["cycles"],
  "instructions": ["instructions"],
  "cache_references": ["cache-references", "cache_references", "cache references"],
  "cache_misses": ["cache-misses", "cache_misses", "cache misses"],
  "dtlb_load_misses": ["dTLB-load-misses", "dtlb-load-misses", "dTLB_load_misses", "dTLB-load-miss"],
  "llc_load_misses": ["LLC-load-misses", "LLC_load_misses", "LLC-load-miss", "LLC-misses", "LLC_misses"],
  "llc_loads": ["LLC-loads", "LLC_loads"],
  "dtlb_loads": ["dTLB-loads", "dtlb-loads", "dTLB_loads"],
}

def f(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def pick(d, keys):
    for k in keys:
        if k in d and d[k] not in ("", None):
            v = f(d[k], None)
            if v is not None:
                return v
    return None

def read_csv(path):
    with open(path, newline="") as fp:
        r = csv.DictReader(fp)
        return list(r), r.fieldnames

def main():
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    results_path, perf_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    results, _ = read_csv(results_path)
    perf, _ = read_csv(perf_path)

    # index perf by (impl,n,threads)
    perf_idx = {}
    for row in perf:
        key = (row.get("impl"), int(row.get("n")), int(row.get("threads")))
        perf_idx[key] = row

    out_rows = []
    for r in results:
        impl = r["impl"]
        n = int(r["n"])
        t = int(r["threads"])
        key = (impl, n, t)
        p = perf_idx.get(key, {})

        sec_per_iter = f(r.get("sec_per_iter"), None)
        gflops = f(r.get("gflops"), None)

        cycles = pick(p, ALIASES["cycles"])
        instr = pick(p, ALIASES["instructions"])
        cache_ref = pick(p, ALIASES["cache_references"])
        cache_miss = pick(p, ALIASES["cache_misses"])
        dtlb_miss = pick(p, ALIASES["dtlb_load_misses"])
        llc_miss = pick(p, ALIASES["llc_load_misses"])

        # Perf stat values here are totals over the measured command (across all threads).
        # Convert to "per FFT" by dividing by iters (which is in results.csv).
        iters = int(r.get("iters", "0") or "0")
        denom = iters if iters > 0 else None

        def per_fft(x):
            if x is None or denom is None: return ""
            return x / denom

        cycles_per_fft = per_fft(cycles)
        instr_per_fft = per_fft(instr)
        cache_misses_per_fft = per_fft(cache_miss)
        dtlb_misses_per_fft = per_fft(dtlb_miss)
        llc_misses_per_fft = per_fft(llc_miss)

        ipc = ""
        if cycles is not None and instr is not None and cycles > 0:
            ipc = instr / cycles

        cache_miss_rate = ""
        if cache_ref is not None and cache_miss is not None and cache_ref > 0:
            cache_miss_rate = cache_miss / cache_ref

        out = {
            "impl": impl,
            "n": n,
            "threads": t,
            "sec_per_iter": sec_per_iter if sec_per_iter is not None else "",
            "gflops": gflops if gflops is not None else "",
            "cycles_per_fft": cycles_per_fft,
            "instr_per_fft": instr_per_fft,
            "cache_misses_per_fft": cache_misses_per_fft,
            "LLC_load_misses_per_fft": llc_misses_per_fft,
            "dTLB_load_misses_per_fft": dtlb_misses_per_fft,
            "cache_miss_rate": cache_miss_rate,
            "ipc": ipc,
        }
        out_rows.append(out)

    header = [
        "impl","n","threads","sec_per_iter","gflops",
        "cycles_per_fft","instr_per_fft","ipc",
        "cache_misses_per_fft","LLC_load_misses_per_fft","dTLB_load_misses_per_fft","cache_miss_rate",
    ]

    with open(out_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=header)
        w.writeheader()
        for row in sorted(out_rows, key=lambda x:(x["impl"], x["n"], x["threads"])):
            w.writerow(row)

    print(f"Wrote {out_path} with {len(out_rows)} rows")

if __name__ == "__main__":
    main()
