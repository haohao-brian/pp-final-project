#!/usr/bin/env bash
set -euo pipefail

# --------- config ----------
# Expect files:
#   unpinned_t1.csv unpinned_t2.csv unpinned_t4.csv unpinned_t8.csv
#   pinned_t1.csv   pinned_t2.csv   pinned_t4.csv   pinned_t8.csv
# Columns (header):
# N,threads,fftw_flag,include_plan,mkl_best_ms,mkl_avg_ms,fftw_best_ms,fftw_avg_ms,max_err

threads_list=(1 2 4 8)
mkl_col="mkl_best_ms"
fftw_col="fftw_best_ms"
# --------------------------

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# Normalize csvs into: scenario,threads,N,fftw_flag,mkl_ms,fftw_ms
normalize_one() {
  local scenario="$1" t="$2" file="$3"
  if [[ ! -f "$file" ]]; then
    echo "[ERR] missing $file" >&2
    exit 1
  fi

  awk -F',' -v sc="$scenario" -v t="$t" -v mcol="$mkl_col" -v fcol="$fftw_col" '
    NR==1{
      for(i=1;i<=NF;i++){
        if($i==mcol) mi=i
        if($i==fcol) fi=i
        if($i=="N") ni=i
        if($i=="threads") ti=i
        if($i=="fftw_flag") gi=i
      }
      if(!mi||!fi||!ni||!ti||!gi){ print "[ERR] header mismatch in",FILENAME > "/dev/stderr"; exit 2 }
      next
    }
    NR>1{
      N=$ni; thr=$ti; flag=$gi; m=$mi; f=$fi;
      # emit: scenario,threads,N,flag,m,f
      printf "%s,%d,%s,%s,%.10f,%.10f\n", sc, thr, N, flag, m, f
    }
  ' "$file"
}

> "$tmpdir/all.norm"
for t in "${threads_list[@]}"; do
  normalize_one "unpinned" "$t" "unpinned_t${t}.csv" >> "$tmpdir/all.norm"
  normalize_one "pinned"   "$t" "pinned_t${t}.csv"   >> "$tmpdir/all.norm"
done

# Build lookup tables for later joins (by scenario,threads,N,flag)
# Also compute winner + pct diff (positive => winner faster by that %)
awk -F',' '
  BEGIN{OFS=","}
  {
    sc=$1; thr=$2; N=$3; flag=$4; m=$5; f=$6;
    # pct = (slower/faster - 1)*100
    if (m<=0 || f<=0) next
    if (m < f) { winner="MKL"; pct=(f/m - 1.0)*100.0 }
    else if (f < m) { winner="FFTW"; pct=(m/f - 1.0)*100.0 }
    else { winner="TIE"; pct=0.0 }
    key=sc FS thr FS N FS flag
    M[key]=m; F[key]=f; W[key]=winner; P[key]=pct
  }
  END{
    # dump tables as csv
    for (k in W){
      split(k, a, FS)
      sc=a[1]; thr=a[2]; N=a[3]; flag=a[4]
      printf "%s,%s,%s,%s,%s,%.4f,%.10f,%.10f\n",
        sc,thr,N,flag,W[k],P[k],M[k],F[k]
    }
  }
' "$tmpdir/all.norm" \
| sort -t, -k1,1 -k2,2n -k3,3n -k4,4 \
> summary_wins.csv

echo "[OK] wrote summary_wins.csv"

# Now compute:
# - speedup for each lib at threads T relative to T=1 (same scenario,N,flag)
# - pin benefit at same T: unpinned vs pinned (same N,flag,T)
#
# Output columns:
# N,fftw_flag,threads,
# unp_mkl_ms,unp_fftw_ms,unp_mkl_spdup,unp_fftw_spdup,
# pin_mkl_ms,pin_fftw_ms,pin_mkl_spdup,pin_fftw_spdup,
# pin_benefit_mkl_pct,pin_benefit_fftw_pct
awk -F',' '
  BEGIN{OFS=","}
  NR==FNR{
    sc=$1; thr=$2; N=$3; flag=$4; winner=$5; pct=$6; m=$7; f=$8;
    key=sc FS thr FS N FS flag
    M[key]=m; F[key]=f
    next
  }
  END{
    # Build list of (N,flag,thr) present in both scenarios
    for (k in M){
      split(k,a,FS)
      sc=a[1]; thr=a[2]; N=a[3]; flag=a[4]
      if(sc!="unpinned") continue
      kup=k
      kpin="pinned" FS thr FS N FS flag
      if(!(kpin in M)) continue

      # Need baseline T=1 for speedup inside each scenario
      k_u1="unpinned" FS 1 FS N FS flag
      k_p1="pinned"   FS 1 FS N FS flag
      if(!(k_u1 in M) || !(k_p1 in M)) continue

      um=M[kup]; uf=F[kup]
      pm=M[kpin]; pf=F[kpin]

      # speedup = time(T=1)/time(T)
      um1=M[k_u1]; uf1=F[k_u1]
      pm1=M[k_p1]; pf1=F[k_p1]

      um_sp=um1/um; uf_sp=uf1/uf
      pm_sp=pm1/pm; pf_sp=pf1/pf

      # pin benefit = (unpinned/pinned - 1)*100 (positive => pinned faster)
      pin_m=(um/pm - 1.0)*100.0
      pin_f=(uf/pf - 1.0)*100.0

      printf "%s,%s,%d,%.10f,%.10f,%.4f,%.4f,%.10f,%.10f,%.4f,%.4f,%.2f,%.2f\n",
        N,flag,thr,
        um,uf,um_sp,uf_sp,
        pm,pf,pm_sp,pf_sp,
        pin_m,pin_f
    }
  }
' summary_wins.csv summary_wins.csv \
| sort -t, -k1,1n -k2,2 -k3,3n \
> summary_speedup_pinbenefit.csv

echo "[OK] wrote summary_speedup_pinbenefit.csv"
echo "Next: open the CSVs with: column -s, -t < file | less -S"
