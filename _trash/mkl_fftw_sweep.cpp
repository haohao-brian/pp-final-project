// mkl_fftw_sweep.cpp
// Compare Intel MKL DFTI vs FFTW3 with repeats + statistics + CSV.
// - Sweeps N = 2^min_pow .. 2^max_pow
// - Sweeps threads list (e.g., 1,2,4,8)
// - FFTW flag: ESTIMATE or MEASURE
// - Reports plan time + exec time separately
// - Computes max |MKL-FFTW| once per config
//
// Build example (Ubuntu):
//   c++ -O3 -std=c++17 mkl_fftw_sweep.cpp -o mkl_fftw_sweep \
//     -lfftw3 -lfftw3_threads -lm -lpthread -ldl -fopenmp \
//     -I"$MKLROOT/include" -L"$MKLROOT/lib/intel64" -lmkl_rt
//
// Run example:
//   export OMP_PROC_BIND=true
//   export OMP_PLACES=cores
//   ./mkl_fftw_sweep -min_pow 10 -max_pow 22 -threads 1,2,4,8 -fftw ESTIMATE -repeats 20 -iters 50 -warmup 5 -pin 0-7 -csv out.csv

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include <chrono>

#include <fftw3.h>
#include "mkl_dfti.h"
#include <mkl.h>              // mkl_set_num_threads, mkl_set_dynamic
#include <sched.h>
#include <unistd.h>

using cplx = std::complex<double>;

static double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Stats {
    double min = 0, median = 0, mean = 0, std = 0, p95 = 0;
};

static Stats calc_stats(std::vector<double> v_ms) {
    Stats s;
    if (v_ms.empty()) return s;
    std::sort(v_ms.begin(), v_ms.end());
    s.min = v_ms.front();
    s.median = v_ms[v_ms.size()/2];
    s.mean = std::accumulate(v_ms.begin(), v_ms.end(), 0.0) / (double)v_ms.size();

    double var = 0.0;
    for (double x : v_ms) var += (x - s.mean) * (x - s.mean);
    var /= (double)v_ms.size();
    s.std = std::sqrt(var);

    size_t idx95 = (size_t)std::floor(0.95 * (v_ms.size() - 1));
    s.p95 = v_ms[idx95];
    return s;
}

static double max_abs_err(const std::vector<cplx>& a, const std::vector<cplx>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double e = std::abs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

static bool set_affinity_from_str(const std::string& pin) {
    // pin formats:
    //   "0-7" or "0,2,4,6" or "0-3,8-11"
    cpu_set_t set;
    CPU_ZERO(&set);

    auto add_range = [&](int a, int b) {
        if (a > b) std::swap(a, b);
        for (int c = a; c <= b; c++) CPU_SET(c, &set);
    };

    size_t i = 0;
    while (i < pin.size()) {
        while (i < pin.size() && (pin[i] == ' ' || pin[i] == ',')) i++;
        if (i >= pin.size()) break;

        // parse int
        char* endp = nullptr;
        int a = (int)std::strtol(pin.c_str() + i, &endp, 10);
        if (endp == pin.c_str() + i) return false;
        i = (size_t)(endp - pin.c_str());

        while (i < pin.size() && pin[i] == ' ') i++;
        if (i < pin.size() && pin[i] == '-') {
            i++;
            while (i < pin.size() && pin[i] == ' ') i++;
            char* endp2 = nullptr;
            int b = (int)std::strtol(pin.c_str() + i, &endp2, 10);
            if (endp2 == pin.c_str() + i) return false;
            i = (size_t)(endp2 - pin.c_str());
            add_range(a, b);
        } else {
            add_range(a, a);
        }
    }

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        return false;
    }
    return true;
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == ',')) i++;
        if (i >= s.size()) break;
        char* endp = nullptr;
        int v = (int)std::strtol(s.c_str() + i, &endp, 10);
        if (endp == s.c_str() + i) break;
        out.push_back(v);
        i = (size_t)(endp - s.c_str());
    }
    if (out.empty()) out.push_back(1);
    return out;
}

static unsigned fftw_flag_from_str(const std::string& f) {
    if (f == "ESTIMATE") return FFTW_ESTIMATE;
    if (f == "MEASURE")  return FFTW_MEASURE;
    if (f == "PATIENT")  return FFTW_PATIENT;
    if (f == "EXHAUSTIVE") return FFTW_EXHAUSTIVE;
    return FFTW_ESTIMATE;
}

struct Opt {
    int min_pow = 10;
    int max_pow = 22;
    std::vector<int> threads = {1,2,4,8};
    std::string fftw_flag = "ESTIMATE";
    int repeats = 20;
    int warmup = 5;
    int iters = 50;
    std::string pin = "";       // e.g., "0-7"
    std::string csv = "out.csv";
};

static Opt parse_args(int argc, char** argv) {
    Opt o;
    for (int i = 1; i < argc; i++) {
        auto need = [&](const char* k) {
            if (i + 1 >= argc) { std::fprintf(stderr, "Missing value for %s\n", k); std::exit(1); }
        };
        if (!std::strcmp(argv[i], "-min_pow")) { need("-min_pow"); o.min_pow = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-max_pow")) { need("-max_pow"); o.max_pow = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-threads")) { need("-threads"); o.threads = parse_int_list(argv[++i]); }
        else if (!std::strcmp(argv[i], "-fftw")) { need("-fftw"); o.fftw_flag = argv[++i]; }
        else if (!std::strcmp(argv[i], "-repeats")) { need("-repeats"); o.repeats = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-warmup")) { need("-warmup"); o.warmup = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-iters")) { need("-iters"); o.iters = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "-pin")) { need("-pin"); o.pin = argv[++i]; }
        else if (!std::strcmp(argv[i], "-csv")) { need("-csv"); o.csv = argv[++i]; }
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::printf(
                "Usage: %s -min_pow 10 -max_pow 22 -threads 1,2,4,8 -fftw ESTIMATE|MEASURE "
                "-repeats 20 -warmup 5 -iters 50 -pin 0-7 -csv out.csv\n", argv[0]);
            std::exit(0);
        }
    }
    if (o.max_pow < o.min_pow) std::swap(o.max_pow, o.min_pow);
    return o;
}

int main(int argc, char** argv) {
    Opt opt = parse_args(argc, argv);

    if (!opt.pin.empty()) {
        if (!set_affinity_from_str(opt.pin)) {
            std::fprintf(stderr, "Failed to set affinity: %s\n", opt.pin.c_str());
            return 1;
        }
    }

    // FFTW threads init once
    if (fftw_init_threads() == 0) {
        std::fprintf(stderr, "fftw_init_threads failed\n");
        return 1;
    }

    // CSV
    std::FILE* fp = std::fopen(opt.csv.c_str(), "w");
    if (!fp) { perror("fopen"); return 1; }

    std::fprintf(fp,
        "N,threads,fftw_flag,"
        "mkl_plan_min_ms,mkl_plan_med_ms,mkl_plan_mean_ms,mkl_plan_std_ms,mkl_plan_p95_ms,"
        "mkl_exec_min_ms,mkl_exec_med_ms,mkl_exec_mean_ms,mkl_exec_std_ms,mkl_exec_p95_ms,"
        "fftw_plan_min_ms,fftw_plan_med_ms,fftw_plan_mean_ms,fftw_plan_std_ms,fftw_plan_p95_ms,"
        "fftw_exec_min_ms,fftw_exec_med_ms,fftw_exec_mean_ms,fftw_exec_std_ms,fftw_exec_p95_ms,"
        "max_err\n"
    );

    unsigned fflag = fftw_flag_from_str(opt.fftw_flag);

    for (int p = opt.min_pow; p <= opt.max_pow; p++) {
        const int N = 1 << p;

        // deterministic input
        std::vector<cplx> x(N);
        for (int i = 0; i < N; i++) {
            double t = 2.0 * M_PI * i / (double)N;
            x[i] = { std::sin(10*t) + 0.1*std::cos(3*t), 0.2*std::sin(2*t) };
        }

        for (int T : opt.threads) {
            if (T <= 0) continue;

            // -------- MKL plan (descriptor) --------
            mkl_set_dynamic(0);
            mkl_set_num_threads(T);

            std::vector<double> mkl_plan_ms, mkl_exec_ms;
            std::vector<double> fftw_plan_ms, fftw_exec_ms;

            // build FFTW plan (depends on threads!)
            fftw_plan plan_fftw = nullptr;

            // We'll rebuild plans once per config, but measure plan time across repeats too
            // to see stability.
            // For execution: we reuse the plan within each repeat.

            // output buffers
            std::vector<cplx> buf_mkl(N), buf_fftw(N);
            std::vector<cplx> out_mkl(N), out_fftw(N);

            double maxerr = 0.0;
            bool did_err = false;

            for (int r = 0; r < opt.repeats; r++) {
                // ---- MKL plan time ----
                DFTI_DESCRIPTOR_HANDLE desc = nullptr;
                double t0 = now_s();
                MKL_LONG status = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
                if (status != 0) { std::fprintf(stderr, "DftiCreateDescriptor failed: %ld\n", status); return 2; }
                status = DftiCommitDescriptor(desc);
                if (status != 0) { std::fprintf(stderr, "DftiCommitDescriptor failed: %ld\n", status); return 2; }
                double t1 = now_s();
                mkl_plan_ms.push_back((t1 - t0) * 1e3);

                // ---- FFTW plan time ----
                // Note: FFTW plan uses global thread count
                fftw_plan_with_nthreads(T);
                std::memcpy(buf_fftw.data(), x.data(), sizeof(cplx) * N);
                double f0 = now_s();
                plan_fftw = fftw_plan_dft_1d(
                    N,
                    reinterpret_cast<fftw_complex*>(buf_fftw.data()),
                    reinterpret_cast<fftw_complex*>(buf_fftw.data()),
                    FFTW_FORWARD,
                    fflag
                );
                double f1 = now_s();
                if (!plan_fftw) { std::fprintf(stderr, "fftw_plan_dft_1d failed\n"); return 3; }
                fftw_plan_ms.push_back((f1 - f0) * 1e3);

                // ---- warmup ----
                for (int w = 0; w < opt.warmup; w++) {
                    std::memcpy(buf_mkl.data(), x.data(), sizeof(cplx) * N);
                    DftiComputeForward(desc, buf_mkl.data());
                    std::memcpy(buf_fftw.data(), x.data(), sizeof(cplx) * N);
                    fftw_execute(plan_fftw);
                }

                // ---- MKL exec timing (iters) ----
                double e0 = now_s();
                for (int it = 0; it < opt.iters; it++) {
                    std::memcpy(buf_mkl.data(), x.data(), sizeof(cplx) * N);
                    DftiComputeForward(desc, buf_mkl.data());
                }
                double e1 = now_s();
                mkl_exec_ms.push_back(((e1 - e0) * 1e3) / (double)opt.iters);

                // ---- FFTW exec timing (iters) ----
                double g0 = now_s();
                for (int it = 0; it < opt.iters; it++) {
                    std::memcpy(buf_fftw.data(), x.data(), sizeof(cplx) * N);
                    fftw_execute(plan_fftw);
                }
                double g1 = now_s();
                fftw_exec_ms.push_back(((g1 - g0) * 1e3) / (double)opt.iters);

                // ---- error check once per config ----
                if (!did_err) {
                    std::memcpy(out_mkl.data(), x.data(), sizeof(cplx) * N);
                    DftiComputeForward(desc, out_mkl.data());

                    std::memcpy(out_fftw.data(), x.data(), sizeof(cplx) * N);
                    fftw_execute_dft(plan_fftw,
                        reinterpret_cast<fftw_complex*>(out_fftw.data()),
                        reinterpret_cast<fftw_complex*>(out_fftw.data())
                    );

                    maxerr = max_abs_err(out_mkl, out_fftw);
                    did_err = true;
                }

                // cleanup plans created in this repeat
                fftw_destroy_plan(plan_fftw);
                plan_fftw = nullptr;
                DftiFreeDescriptor(&desc);
            }

            Stats mp = calc_stats(mkl_plan_ms);
            Stats me = calc_stats(mkl_exec_ms);
            Stats fplan = calc_stats(fftw_plan_ms);
            Stats fexec = calc_stats(fftw_exec_ms);

            std::printf("N=%d T=%d FFTW_%s\n", N, T, opt.fftw_flag.c_str());
            std::printf("  MKL  plan(ms): med=%.4f p95=%.4f | exec(ms): med=%.4f p95=%.4f\n",
                        mp.median, mp.p95, me.median, me.p95);
            std::printf("  FFTW plan(ms): med=%.4f p95=%.4f | exec(ms): med=%.4f p95=%.4f\n",
                        fplan.median, fplan.p95, fexec.median, fexec.p95);
            std::printf("  max |MKL-FFTW| = %.3e\n", maxerr);

            std::fprintf(fp,
                "%d,%d,%s,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,"
                "%.6e\n",
                N, T, opt.fftw_flag.c_str(),
                mp.min, mp.median, mp.mean, mp.std, mp.p95,
                me.min, me.median, me.mean, me.std, me.p95,
                fplan.min, fplan.median, fplan.mean, fplan.std, fplan.p95,
                fexec.min, fexec.median, fexec.mean, fexec.std, fexec.p95,
                maxerr
            );

            std::fflush(fp);
        }
    }

    std::fclose(fp);
    fftw_cleanup_threads();
    return 0;
}
