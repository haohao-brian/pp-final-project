## step 1

```
g++ -O3 -march=native -ffast-math -fopenmp bench_csv.cpp \
  -lfftw3 -lfftw3_threads -lpthread \
  -o bench_csv
```

## step 2
```
g++ -O3 -march=native -ffast-math -fopenmp -c fft_a1.cpp -o fft_a1.o
```
## step 3
```
g++ -O3 -march=native -ffast-math -fopenmp bench_csv_a1.cpp fft_a1.o \
  -lfftw3 -lfftw3_threads -lpthread \
  -o bench_csv_a1
```
## step 4
```
./run_cmp_fftw_a1.sh ./bench_csv ./bench_csv_a1
```
## result 
```
    •    results_cmp.csv（bench 本身的 sec/gflops/checksum）
    •    perf_logs_cmp/（每次 perf stat 的 raw log）
    •    perf_merged_cmp.csv（parse_perf.py 合併後）
    •    baseline_cmp.csv（join_derive_v2.py 合併後：cycles/IPC/LLC/dTLB 等 per-FFT 指標）
```
## show result
```
cat baseline_cmp.csv
可以看到各項數據
```
## example result
```
impl,n,threads,sec_per_iter,gflops,cycles_per_fft,instr_per_fft,ipc,cache_misses_per_fft,LLC_load_misses_per_fft,dTLB_load_misses_per_fft,cache_miss_rate
a1,16777216,8,0.104112408,19.337425,16253908606.0,21872935658.0,1.345703128287911,79264091.0,6007918.0,598004.0,0.8430416312377367
a1,67108864,8,0.435016736,20.054751,60053211483.0,92635747265.0,1.5425610883645005,325309703.0,28153193.0,2130046.0,0.7863417783381503
fftw,16777216,8,0.248484402,8.102182,6503918200.0,6384126014.0,0.9815815355734333,86621888.0,21773282.0,42198319.0,0.4514306914106025
fftw,67108864,8,1.483275748,5.881679,38638075228.0,30125917271.0,0.7796950829778534,515566905.0,152587724.0,172609353.0,0.4636450198173262
```
