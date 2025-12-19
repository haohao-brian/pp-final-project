#!/usr/bin/env bash
set -euo pipefail

# run_scaling_pcore_vs_all.sh
# Run FFTW threads scaling at a single N for two CPU policies:
#   - all:   use all allowed CPUs
#   - pcore: auto-detect "faster" core group on hybrid CPUs (P-cores)
#
# Outputs:
#   results_all.csv, perf_logs_all/, baseline_all.csv
#   results_pcore.csv, perf_logs_pcore/, baseline_pcore.csv
#   baseline_policy.csv  (combined with a "policy" column)
#
# Usage:
#   chmod +x run_scaling_pcore_vs_all.sh
#   ./run_scaling_pcore_vs_all.sh ./bench_csv

BENCH=${1:-./bench_csv}

# pick your target N here (2^24)
N=$((1<<24))

# threads to test (will be capped by available CPUs in each policy)
THREADS=(1 2 4 8 16 24)

EVENTS="cycles,instructions,cache-references,cache-misses,dTLB-loads,dTLB-load-misses,LLC-loads,LLC-load-misses"

# FFTW parameters
SEC=0.3
PLAN=estimate
INPLACE=0
IMPL=fftw

# ---------- helpers ----------
read_allowed_cpus() {
  # Prefer cgroup cpuset if present (containers/Slurm often)
  if [[ -r /sys/fs/cgroup/cpuset.cpus.effective ]]; then
    cat /sys/fs/cgroup/cpuset.cpus.effective
  else
    # fallback: all CPUs
    local max=$(( $(nproc) - 1 ))
    echo "0-${max}"
  fi
}

expand_cpuset_to_list() {
  # input like "0-3,8,10-11" -> "0 1 2 3 8 10 11"
  local s="$1"
  local out=()
  IFS=',' read -ra parts <<< "$s"
  for p in "${parts[@]}"; do
    if [[ "$p" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local a="${BASH_REMATCH[1]}" b="${BASH_REMATCH[2]}"
      for ((i=a;i<=b;i++)); do out+=("$i"); done
    elif [[ "$p" =~ ^[0-9]+$ ]]; then
      out+=("$p")
    fi
  done
  printf "%s " "${out[@]}"
}

list_to_compact_cpuset() {
  # input: "0 1 2 3 8 10 11" -> "0-3,8,10-11"
  local nums=("$@")
  if [[ ${#nums[@]} -eq 0 ]]; then echo ""; return; fi
  IFS=$'\n' nums=($(printf "%s\n" "${nums[@]}" | sort -n))
  local res=()
  local start="${nums[0]}" prev="${nums[0]}"
  for ((i=1;i<${#nums[@]};i++)); do
    local x="${nums[i]}"
    if (( x == prev + 1 )); then
      prev="$x"
    else
      if (( start == prev )); then res+=("$start"); else res+=("${start}-${prev}"); fi
      start="$x"; prev="$x"
    fi
  done
  if (( start == prev )); then res+=("$start"); else res+=("${start}-${prev}"); fi
  (IFS=','; echo "${res[*]}")
}

cpu_max_freq_khz() {
  local cpu="$1"
  local p="/sys/devices/system/cpu/cpu${cpu}/cpufreq/cpuinfo_max_freq"
  [[ -r "$p" ]] && cat "$p" || echo "0"
}

cpu_core_type() {
  # exists on many hybrid kernels: /sys/.../topology/core_type
  local cpu="$1"
  local p="/sys/devices/system/cpu/cpu${cpu}/topology/core_type"
  [[ -r "$p" ]] && cat "$p" || echo ""
}

detect_pcore_cpuset() {
  local allowed
  allowed="$(read_allowed_cpus)"
  local cpus
  cpus=($(expand_cpuset_to_list "$allowed"))

  # If core_type exists and shows >1 distinct type, choose the type with higher avg max_freq.
  local has_type=0
  local types=()
  for c in "${cpus[@]}"; do
    local t
    t="$(cpu_core_type "$c")"
    if [[ -n "$t" ]]; then
      has_type=1
      types+=("$t")
    fi
  done

  if (( has_type == 1 )); then
    # count distinct types
    IFS=$'\n' types=($(printf "%s\n" "${types[@]}" | sort -n | uniq))
    if (( ${#types[@]} >= 2 )); then
      local best_t="" best_avg=-1
      for t in "${types[@]}"; do
        local sum=0 cnt=0
        for c in "${cpus[@]}"; do
          [[ "$(cpu_core_type "$c")" == "$t" ]] || continue
          local f
          f="$(cpu_max_freq_khz "$c")"
          sum=$((sum + f)); cnt=$((cnt + 1))
        done
        if (( cnt > 0 )); then
          local avg=$((sum / cnt))
          if (( avg > best_avg )); then
            best_avg="$avg"
            best_t="$t"
          fi
        fi
      done

      local pcores=()
      for c in "${cpus[@]}"; do
        [[ "$(cpu_core_type "$c")" == "$best_t" ]] && pcores+=("$c")
      done
      list_to_compact_cpuset "${pcores[@]}"
      return 0
    fi
  fi

  # Fallback: choose CPUs whose cpuinfo_max_freq is within 95% of the max (often P-cores).
  local maxf=0
  for c in "${cpus[@]}"; do
    local f; f="$(cpu_max_freq_khz "$c")"
    (( f > maxf )) && maxf="$f"
  done
  if (( maxf == 0 )); then
    # no cpufreq info; fallback to "all"
    echo "$allowed"
    return 0
  fi

  local thr=$(( maxf * 95 / 100 ))
  local pcores=()
  for c in "${cpus[@]}"; do
    local f; f="$(cpu_max_freq_khz "$c")"
    (( f >= thr )) && pcores+=("$c")
  done

  # If everything passes threshold, then it's probably non-hybrid → pcore == all
  if (( ${#pcores[@]} == ${#cpus[@]} )); then
    echo "$allowed"
  else
    list_to_compact_cpuset "${pcores[@]}"
  fi
}

cap_threads_by_cpuset() {
  local cpuset="$1"
  local cpus
  cpus=($(expand_cpuset_to_list "$cpuset"))
  echo "${#cpus[@]}"
}

run_policy() {
  local policy="$1"
  local cpuset="$2"

  local OUT="results_${policy}.csv"
  local LOGDIR="perf_logs_${policy}"
  rm -f "$OUT"
  rm -rf "$LOGDIR"
  mkdir -p "$LOGDIR"

  # write header
  "$BENCH" --impl "$IMPL" --n 1024 --threads 1 --sec 0.01 --plan "$PLAN" --inplace "$INPLACE" --header 1 | head -n 1 > "$OUT"

  local maxT
  maxT="$(cap_threads_by_cpuset "$cpuset")"
  echo "[policy=$policy] cpuset=$cpuset (cpus=$maxT)" >&2

  for t in "${THREADS[@]}"; do
    if (( t > maxT )); then
      echo "  skip threads=$t (exceeds cpus=$maxT)" >&2
      continue
    fi

    local log="${LOGDIR}/perf_${IMPL}_N${N}_T${t}.csv"
    echo "== policy=$policy impl=$IMPL N=$N threads=$t ==" >&2

    taskset -c "$cpuset" sudo -E perf stat -x, -r 1 -e "$EVENTS" -- \
      "$BENCH" --impl "$IMPL" --n "$N" --threads "$t" --sec "$SEC" --plan "$PLAN" \
      --inplace "$INPLACE" --seed 1 \
      1>>"$OUT" 2>"$log" || echo "FAIL policy=$policy threads=$t (see $log)" >&2
  done

  python3 parse_perf.py "$LOGDIR" "perf_merged_${policy}.csv"
  python3 join_derive_v2.py "$OUT" "perf_merged_${policy}.csv" "baseline_${policy}.csv"
}

combine_baselines() {
  local out="baseline_policy.csv"
  rm -f "$out"
  # add a 'policy' column in front
  echo "policy,$(head -n1 baseline_all.csv)" > "$out"
  tail -n +2 baseline_all.csv   | awk 'BEGIN{OFS=","}{print "all",$0}'   >> "$out"
  tail -n +2 baseline_pcore.csv | awk 'BEGIN{OFS=","}{print "pcore",$0}' >> "$out"
  echo "Wrote $out" >&2
}

# ---------- main ----------
ALL_CPUSET="$(read_allowed_cpus)"
PCORE_CPUSET="$(detect_pcore_cpuset)"

# If pcore == all, still run both but warn
if [[ "$PCORE_CPUSET" == "$ALL_CPUSET" ]]; then
  echo "[warn] pcore cpuset == all cpuset ($ALL_CPUSET). This may be a non-hybrid CPU or missing sysfs info." >&2
fi

run_policy "all"   "$ALL_CPUSET"
run_policy "pcore" "$PCORE_CPUSET"
combine_baselines

echo "Done. See baseline_policy.csv"
