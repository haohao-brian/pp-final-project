#!/usr/bin/env bash
set -euo pipefail

# run_sweep_perf.sh
# Sweeps N and threads, runs perf stat, writes:
#   1) results.csv        (bench_csv stdout appended; exactly 1 row per (impl,n,threads))
#   2) perf_logs/*.csv    (perf stat -x, stderr per run)
#
# Usage:
#   ./run_sweep_perf.sh ./bench_csv results.csv perf_logs
#
# Notes (paper-grade hygiene):
# - We DO NOT use perf -r > 1, because that would re-run the benchmark and duplicate CSV rows.
# - Timing stability is controlled by bench_csv --sec.
# - Script clears output files at start to avoid accidental appends across runs.

BENCH=${1:-./bench_csv}
OUTCSV=${2:-results.csv}
LOGDIR=${3:-perf_logs}

# Perf event set (generic names; on hybrid CPUs perf may emit cpu_core/... and cpu_atom/...).
EVENTS="cycles,instructions,cache-references,cache-misses,dTLB-loads,dTLB-load-misses,LLC-loads,LLC-load-misses"

POWERS=(16 18 20 22 24 26)
THREADS=(1 2 4 8 16 24)
IMPLS=("fftw" "ours")

# Clean outputs (avoid duplicate rows across runs)
rm -f "$OUTCSV"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

# Write header
"$BENCH" --impl fftw --n 1024 --threads 1 --sec 0.01 --plan estimate --inplace 0 --header 1 | head -n 1 > "$OUTCSV"

# One perf measurement per point
REPS=1

for impl in "${IMPLS[@]}"; do
  for p in "${POWERS[@]}"; do
    n=$((1<<p))

    # big-N policy
    SEC=0.3
    INPLACE=0
    if [[ $p -ge 22 ]]; then
      SEC=0.1
      INPLACE=1
    fi

    # optional: skip ours for big N first
    if [[ "$impl" == "ours" && $p -ge 22 ]]; then
      echo "skip impl=ours for p>=22 (for now)" >&2
      continue
    fi

    for t in "${THREADS[@]}"; do
      echo "== impl=$impl N=2^$p ($n) threads=$t ==" >&2
      log="$LOGDIR/perf_${impl}_N${n}_T${t}.csv"

      # Keep environment (OMP_*), require sudo due to perf_event_paranoid=4
      if ! taskset -c 0-23 sudo -E perf stat -x, -r $REPS -e "$EVENTS" -- \
          "$BENCH" --impl "$impl" --n "$n" --threads "$t" --sec $SEC --plan estimate \
          --inplace $INPLACE --seed 1 \
          1>>"$OUTCSV" 2>"$log"
      then
        echo "FAIL impl=$impl n=$n threads=$t (see $log)" >&2
      fi
    done
  done
done

echo "Done. CSV: $OUTCSV ; perf logs: $LOGDIR/"
