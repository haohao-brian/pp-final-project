#!/usr/bin/env bash
set -euo pipefail

BENCH_FFTW=${1:-./bench_csv}
BENCH_A1=${2:-./bench_csv_a1}

# N / threads（跟你報告那組一致）
THREADS=8
NS=(16777216 67108864)   # 2^24, 2^26

EVENTS="cycles,instructions,cache-references,cache-misses,dTLB-loads,dTLB-load-misses,LLC-loads,LLC-load-misses"

OUT=results_cmp.csv
LOGDIR=perf_logs_cmp
PERF_MERGED=perf_merged_cmp.csv
BASELINE=baseline_cmp.csv

rm -f "$OUT" "$PERF_MERGED" "$BASELINE"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

# --- detect P-core cpuset (Intel hybrid: core_type=1 means P-core) ---
detect_pcore_cpuset() {
  if [[ -r /sys/devices/system/cpu/cpu0/topology/core_type ]]; then
    python3 - <<'PY'
import glob
pc=[]
for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_type"):
    cpu=int(path.split("/cpu")[1].split("/")[0])
    try:
        t=open(path).read().strip()
    except:
        continue
    if t=="1":
        pc.append(cpu)
pc=sorted(pc)
if not pc:
    import os; print(f"0-{os.cpu_count()-1}")
    raise SystemExit
# compress to ranges
ranges=[]
s=pc[0]; p=pc[0]
for x in pc[1:]:
    if x==p+1: p=x
    else:
        ranges.append((s,p))
        s=p=x
ranges.append((s,p))
print(",".join([f"{a}-{b}" if a!=b else f"{a}" for a,b in ranges]))
PY
  else
    echo "0-$(($(nproc)-1))"
  fi
}

PCORE_CPUSET="$(detect_pcore_cpuset)"
echo "[info] PCORE_CPUSET=$PCORE_CPUSET" >&2

# Optional: reduce migration noise
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Header
"$BENCH_FFTW" --impl fftw --n 1024 --threads 1 --iters 1 --plan estimate --inplace 0 --seed 1 --header 1 \
  | head -n1 > "$OUT"

run_one () {
  local impl="$1" n="$2" bin="$3"
  local log="$LOGDIR/perf_${impl}_N${n}_T${THREADS}.csv"
  echo "== CMP impl=$impl N=$n T=$THREADS ==" >&2

  # 固定 iters：避免 auto-iter 讓兩者跑不同次而難比
  # 如果你想更穩，可以把 --iters 1 改成 --iters 2 或 4
  taskset -c "$PCORE_CPUSET" sudo -E perf stat -x, -r 1 -e "$EVENTS" -- \
    "$bin" --impl "$impl" --n "$n" --threads "$THREADS" --iters 1 --plan estimate --inplace 0 --seed 1 --header 0 \
    1>>"$OUT" 2>"$log" || true
}

for n in "${NS[@]}"; do
  run_one fftw "$n" "$BENCH_FFTW"
  run_one a1   "$n" "$BENCH_A1"
done

python3 parse_perf.py "$LOGDIR" "$PERF_MERGED"
python3 join_derive_v2.py "$OUT" "$PERF_MERGED" "$BASELINE"

echo "[done] results: $OUT $BASELINE (and $LOGDIR/)" >&2
