#!/usr/bin/env bash
set -euo pipefail

# Compare FFTW vs A1 (4-step/blocked transpose version) at fixed (N, threads)
# Outputs:
#   results_cmp.csv
#   perf_logs_cmp/*.csv
#   perf_merged_cmp.csv
#   baseline_cmp.csv
#   report_artifacts/*.csv

BENCH_FFTW=${1:-./bench_csv}       # outputs impl=fftw
BENCH_A1=${2:-./bench_csv_a1}      # outputs impl=a1
OUTCSV=${3:-results_cmp.csv}
LOGDIR=${4:-perf_logs_cmp}
EVENTS=${EVENTS:-"cycles,instructions,cache-references,cache-misses,dTLB-loads,dTLB-load-misses,LLC-loads,LLC-load-misses"}

# default test points (same as你現在交差用的)
THREADS=${THREADS:-8}
NS=(${NS:-$((1<<24)) $((1<<26))})

mkdir -p "$LOGDIR" report_artifacts
rm -f "$OUTCSV" perf_merged_cmp.csv baseline_cmp.csv

# ---------- Detect P-core cpuset (best effort) ----------
detect_pcore_cpuset() {
  local sys=/sys/devices/system/cpu
  local have=0
  for f in $sys/cpu*/topology/core_type; do
    [[ -f "$f" ]] && have=1 && break
  done

  # If no core_type, fallback to "all CPUs"
  if [[ "$have" -eq 0 ]]; then
    local last=$(( $(nproc --all) - 1 ))
    echo "0-$last"
    return 0
  fi

  # If core_type exists, decide P/E by avg max freq
  # Pick the core_type with higher avg cpuinfo_max_freq as P-core.
  declare -A sum cnt
  for cpu_dir in $sys/cpu[0-9]*; do
    local cpu=${cpu_dir##*/cpu}
    [[ ! -f "$cpu_dir/topology/core_type" ]] && continue
    local ct
    ct=$(cat "$cpu_dir/topology/core_type" 2>/dev/null || echo "")
    [[ -z "$ct" ]] && continue

    local fpath="$cpu_dir/cpufreq/cpuinfo_max_freq"
    local mhz=0
    if [[ -f "$fpath" ]]; then
      mhz=$(cat "$fpath" 2>/dev/null || echo 0)
    fi
    sum[$ct]=$(( ${sum[$ct]:-0} + mhz ))
    cnt[$ct]=$(( ${cnt[$ct]:-0} + 1 ))
  done

  local best_ct="" best_avg=-1
  for ct in "${!cnt[@]}"; do
    local c=${cnt[$ct]}
    local s=${sum[$ct]:-0}
    local avg=0
    if [[ "$c" -gt 0 ]]; then avg=$(( s / c )); fi
    if [[ "$avg" -gt "$best_avg" ]]; then
      best_avg=$avg
      best_ct=$ct
    fi
  done

  # Build cpuset list for chosen type
  local list=""
  for cpu_dir in $sys/cpu[0-9]*; do
    local cpu=${cpu_dir##*/cpu}
    [[ ! -f "$cpu_dir/topology/core_type" ]] && continue
    local ct
    ct=$(cat "$cpu_dir/topology/core_type" 2>/dev/null || echo "")
    [[ "$ct" == "$best_ct" ]] || continue
    list+="${cpu},"
  done
  list="${list%,}"
  [[ -z "$list" ]] && list="0-$(( $(nproc --all)-1 ))"
  echo "$list"
}

PCORE_CPUSET=${PCORE_CPUSET:-$(detect_pcore_cpuset)}
echo "[info] PCORE_CPUSET=$PCORE_CPUSET" >&2

# ---------- Stable OpenMP placement (for FFTW threads too, often helps reproducibility) ----------
export OMP_PROC_BIND=${OMP_PROC_BIND:-true}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_SCHEDULE=${OMP_SCHEDULE:-static}

# ---------- Write header ----------
"$BENCH_FFTW" --impl fftw --n 1024 --threads 1 --sec 0.01 --plan estimate --inplace 0 --header 1 | head -n1 > "$OUTCSV"

run_one() {
  local impl=$1
  local n=$2
  local t=$3
  local bench=$4
  local log="$LOGDIR/perf_${impl}_N${n}_T${t}.csv"

  echo "== CMP impl=$impl N=$n T=$t ==" >&2
  taskset -c "$PCORE_CPUSET" sudo -E perf stat -x, -r 1 -e "$EVENTS" -- \
    "$bench" --impl "$impl" --n "$n" --threads "$t" --sec 0.3 --plan estimate --inplace 0 --seed 1 \
    1>>"$OUTCSV" 2>"$log" || true
}

for n in "${NS[@]}"; do
  run_one fftw "$n" "$THREADS" "$BENCH_FFTW"
  run_one a1   "$n" "$THREADS" "$BENCH_A1"
done

python3 parse_perf.py "$LOGDIR" perf_merged_cmp.csv
python3 join_derive_v2.py "$OUTCSV" perf_merged_cmp.csv baseline_cmp.csv
python3 make_a1_artifacts.py baseline_cmp.csv report_artifacts

echo "[done] results: $OUTCSV baseline_cmp.csv report_artifacts/ (tables)" >&2
