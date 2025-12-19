#!/usr/bin/env python3
import csv, os, math, sys

def f(row, k):
    try:
        return float(row[k])
    except Exception:
        return float("nan")

def load(path):
    rows = {}
    with open(path) as fp:
        r = csv.DictReader(fp)
        for row in r:
            key = (int(row["n"]), int(row["threads"]), row["impl"])
            rows[key] = row
    return rows

def wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(rows)

def main():
    if len(sys.argv) != 3:
        print("usage: make_a1_artifacts.py baseline_cmp.csv report_artifacts_dir", file=sys.stderr)
        return 2
    inp, outdir = sys.argv[1], sys.argv[2]
    data = load(inp)

    # infer Ns present for threads=8, impl in {fftw,a1}
    Ns = sorted({n for (n,t,impl) in data.keys() if t == 8 and impl in ("fftw","a1")})
    if not Ns:
        print("No rows for threads=8 and impl in {fftw,a1}", file=sys.stderr)
        return 2

    # R1: gflops vs n + speedup
    r1 = []
    for n in Ns:
        a = data[(n,8,"fftw")]
        b = data[(n,8,"a1")]
        gf_a = f(a,"gflops"); gf_b = f(b,"gflops")
        sp = gf_b / gf_a if gf_a > 0 else float("nan")
        r1.append([n, gf_a, gf_b, sp])
    wcsv(os.path.join(outdir, "R1_gflops_vs_n.csv"),
         ["n","gflops_fftw","gflops_a1","speedup_a1_over_fftw"], r1)

    # R2: dTLB misses per FFT + reduction frac
    r2 = []
    for n in Ns:
        a = data[(n,8,"fftw")]
        b = data[(n,8,"a1")]
        dt_a = f(a,"dTLB_load_misses_per_fft")
        dt_b = f(b,"dTLB_load_misses_per_fft")
        red = 1.0 - (dt_b/dt_a) if dt_a > 0 else float("nan")
        r2.append([n, dt_a, dt_b, red])
    wcsv(os.path.join(outdir, "R2_dtlb_misses_vs_n.csv"),
         ["n","dtlb_fftw","dtlb_a1","reduction_frac_(1-a1/fftw)"], r2)

    # R3: LLC misses per FFT + reduction frac
    r3 = []
    for n in Ns:
        a = data[(n,8,"fftw")]
        b = data[(n,8,"a1")]
        llc_a = f(a,"LLC_load_misses_per_fft")
        llc_b = f(b,"LLC_load_misses_per_fft")
        red = 1.0 - (llc_b/llc_a) if llc_a > 0 else float("nan")
        r3.append([n, llc_a, llc_b, red])
    wcsv(os.path.join(outdir, "R3_llc_misses_vs_n.csv"),
         ["n","llc_fftw","llc_a1","reduction_frac_(1-a1/fftw)"], r3)

    # R4: IPC vs n + delta
    r4 = []
    for n in Ns:
        a = data[(n,8,"fftw")]
        b = data[(n,8,"a1")]
        ipc_a = f(a,"ipc")
        ipc_b = f(b,"ipc")
        r4.append([n, ipc_a, ipc_b, ipc_b - ipc_a])
    wcsv(os.path.join(outdir, "R4_ipc_vs_n.csv"),
         ["n","ipc_fftw","ipc_a1","delta_ipc_(a1-fftw)"], r4)

    # Table summary (single table, easy to paste into Word)
    tsum = []
    for n in Ns:
        a = data[(n,8,"fftw")]
        b = data[(n,8,"a1")]
        gf_a, gf_b = f(a,"gflops"), f(b,"gflops")
        llc_a, llc_b = f(a,"LLC_load_misses_per_fft"), f(b,"LLC_load_misses_per_fft")
        dt_a, dt_b = f(a,"dTLB_load_misses_per_fft"), f(b,"dTLB_load_misses_per_fft")
        ipc_a, ipc_b = f(a,"ipc"), f(b,"ipc")
        tsum.append([
            n, 8,
            gf_a, gf_b, (gf_b/gf_a if gf_a>0 else float("nan")),
            llc_a, llc_b, (1-llc_b/llc_a if llc_a>0 else float("nan")),
            dt_a, dt_b, (1-dt_b/dt_a if dt_a>0 else float("nan")),
            ipc_a, ipc_b, (ipc_b-ipc_a),
        ])
    wcsv(os.path.join(outdir, "Table_A1_summary.csv"),
         ["n","threads","gflops_fftw","gflops_a1","speedup",
          "llc_fftw","llc_a1","llc_reduction_frac",
          "dtlb_fftw","dtlb_a1","dtlb_reduction_frac",
          "ipc_fftw","ipc_a1","delta_ipc"], tsum)

    # Correctness proxy: checksum difference (quick sanity check, not a formal error bound)
    corr = []
    for n in Ns:
        a = data[(n,8,"fftw")]
        b = data[(n,8,"a1")]
        ca, cb = f(a,"checksum"), f(b,"checksum")
        rel = abs(cb-ca) / max(abs(ca), 1e-12)
        corr.append([n, 8, ca, cb, rel])
    wcsv(os.path.join(outdir, "Table_correctness_checksum.csv"),
         ["n","threads","checksum_fftw","checksum_a1","relative_error"], corr)

    print(f"Wrote artifacts into: {outdir}")

if __name__ == "__main__":
    raise SystemExit(main())
