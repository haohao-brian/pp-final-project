#!/usr/bin/env python3
"""
join_derive_v2.py

Robust merge for perf stat outputs that use hybrid PMU event names like:
  cpu_core/cycles/, cpu_atom/cycles/, cpu_core/instructions/, cpu_atom/instructions/, ...

It merges:
  - results.csv (from bench_csv)
  - perf_merged.csv (from parse_perf.py)

and produces:
  baseline_profile.csv

Usage:
  python3 join_derive_v2.py results.csv perf_merged.csv baseline_profile.csv

Key behavior:
- For cycles/instructions/cache-* and tlb/llc events, we SUM across all matching columns
  (e.g., cpu_core + cpu_atom) to get a total.
- Pure stdlib.
"""
import csv, sys

def to_f(x):
    try:
        return float(x)
    except Exception:
        return None

def sum_matching(row, must_substrings):
    total = 0.0
    found = False
    for k, v in row.items():
        if v in ("", None):
            continue
        kk = k.lower()
        ok = True
        for s in must_substrings:
            if s not in kk:
                ok = False
                break
        if not ok:
            continue
        fv = to_f(v)
        if fv is None:
            continue
        total += fv
        found = True
    return total if found else None

def read_csv(path):
    with open(path, newline="") as fp:
        r = csv.DictReader(fp)
        return list(r)

def main():
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    results_path, perf_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    results = read_csv(results_path)
    perf = read_csv(perf_path)

    perf_idx = {}
    for row in perf:
        try:
            key = (row.get("impl"), int(row.get("n")), int(row.get("threads")))
        except Exception:
            continue
        perf_idx[key] = row

    out_rows = []
    for r in results:
        impl = r["impl"]
        n = int(r["n"])
        t = int(r["threads"])
        key = (impl, n, t)
        p = perf_idx.get(key, {})

        iters = int(r.get("iters", "0") or "0")
        denom = iters if iters > 0 else None

        sec_per_iter = to_f(r.get("sec_per_iter"))
        gflops = to_f(r.get("gflops"))

        cycles = sum_matching(p, ["cycles"])
        instr  = sum_matching(p, ["instructions"])

        cache_ref  = sum_matching(p, ["cache-references"])
        cache_miss = sum_matching(p, ["cache-misses"])

        llc_miss = (sum_matching(p, ["llc-load-misses"]) or
                    sum_matching(p, ["llc-misses"]))
        dtlb_miss = sum_matching(p, ["dtlb-load-misses"])

        def per_fft(x):
            if x is None or denom is None:
                return ""
            return x / denom

        cycles_per_fft = per_fft(cycles)
        instr_per_fft  = per_fft(instr)
        cache_misses_per_fft = per_fft(cache_miss)
        llc_misses_per_fft   = per_fft(llc_miss)
        dtlb_misses_per_fft  = per_fft(dtlb_miss)

        ipc = ""
        if cycles is not None and instr is not None and cycles > 0:
            ipc = instr / cycles

        cache_miss_rate = ""
        if cache_ref is not None and cache_miss is not None and cache_ref > 0:
            cache_miss_rate = cache_miss / cache_ref

        out_rows.append({
            "impl": impl,
            "n": n,
            "threads": t,
            "sec_per_iter": sec_per_iter if sec_per_iter is not None else "",
            "gflops": gflops if gflops is not None else "",
            "cycles_per_fft": cycles_per_fft,
            "instr_per_fft": instr_per_fft,
            "ipc": ipc,
            "cache_misses_per_fft": cache_misses_per_fft,
            "LLC_load_misses_per_fft": llc_misses_per_fft,
            "dTLB_load_misses_per_fft": dtlb_misses_per_fft,
            "cache_miss_rate": cache_miss_rate,
        })

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
