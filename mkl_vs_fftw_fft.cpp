#include <fftw3.h>
#include "mkl_dfti.h"
#include "mkl_service.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
static double now_sec() { return omp_get_wtime(); }
#else
#include <chrono>
static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
#endif

static void die(const char* msg) {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(1);
}

static long long parse_ll(const char* s) {
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (!s || *s == '\0' || (end && *end != '\0')) die("bad integer arg");
    return v;
}

static double max_abs_err(const std::vector<std::complex<double>>& a,
                          const std::vector<std::complex<double>>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double e = std::abs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

struct Options {
    int n = -1;
    int min_pow = 20;
    int max_pow = 20;
    int iters = 10;
    int warmup = 3;
    int threads = 1;
    bool fftw_measure = false;
    bool include_plan = false;
    std::string csv_path;
};

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::string m = "missing value after ";
                m += flag;
                die(m.c_str());
            }
            return argv[++i];
        };

        if (a == "-n") o.n = (int)parse_ll(need("-n"));
        else if (a == "-min_pow") o.min_pow = (int)parse_ll(need("-min_pow"));
        else if (a == "-max_pow") o.max_pow = (int)parse_ll(need("-max_pow"));
        else if (a == "-iters") o.iters = (int)parse_ll(need("-iters"));
        else if (a == "-warmup") o.warmup = (int)parse_ll(need("-warmup"));
        else if (a == "-threads") o.threads = (int)parse_ll(need("-threads"));
        else if (a == "-fftw_measure") o.fftw_measure = true;
        else if (a == "-include_plan") o.include_plan = true;
        else if (a == "-csv") o.csv_path = need("-csv");
        else if (a == "-h" || a == "--help") {
            std::printf(
                "Usage:\n"
                "  ./mkl_vs_fftw_fft [-n N] [-min_pow P] [-max_pow P]\n"
                "                    [-threads T] [-iters I] [-warmup W]\n"
                "                    [-fftw_measure] [-include_plan] [-csv out.csv]\n"
                "\n"
                "Examples:\n"
                "  ./mkl_vs_fftw_fft -n 1048576 -threads 1\n"
                "  ./mkl_vs_fftw_fft -min_pow 10 -max_pow 22 -threads 8 -csv out.csv\n"
                "  ./mkl_vs_fftw_fft -min_pow 10 -max_pow 22 -fftw_measure\n"
            );
            std::exit(0);
        } else {
            std::string m = "unknown arg: " + a;
            die(m.c_str());
        }
    }
    if (o.threads <= 0) die("threads must be >= 1");
    if (o.iters <= 0) die("iters must be >= 1");
    if (o.warmup < 0) die("warmup must be >= 0");
    if (o.n != -1 && o.n <= 0) die("N must be > 0");
    if (o.n == -1 && (o.min_pow < 1 || o.max_pow < o.min_pow)) die("bad pow range");
    return o;
}

static void fill_input(std::vector<std::complex<double>>& x) {
    const int N = (int)x.size();
    for (int i = 0; i < N; i++) {
        double t = 2.0 * M_PI * i / (double)N;
        x[i] = {std::sin(10 * t) + 0.1 * std::cos(3 * t), 0.0};
    }
}

struct Stat { double best_ms, avg_ms; };

template <class Fn>
static Stat bench_ms(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn();

    double best = 1e100, sum = 0.0;
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        fn();
        double t1 = now_sec();
        double ms = (t1 - t0) * 1e3;
        best = std::min(best, ms);
        sum += ms;
    }
    return {best, sum / iters};
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    // Set MKL threading (also respects env, but we set explicitly here)
    mkl_set_dynamic(0);
    mkl_set_num_threads(opt.threads);

    // FFTW threads (only used if we call fftw_init_threads + plan_with_nthreads)
    fftw_init_threads();
    fftw_plan_with_nthreads(opt.threads);

    FILE* csv = nullptr;
    if (!opt.csv_path.empty()) {
        csv = std::fopen(opt.csv_path.c_str(), "w");
        if (!csv) die("failed to open csv");
        std::fprintf(csv, "N,threads,fftw_flag,include_plan,mkl_best_ms,mkl_avg_ms,fftw_best_ms,fftw_avg_ms,max_err\n");
    }

    auto run_one = [&](int N) {
        std::vector<std::complex<double>> x(N), y_mkl(N), y_fftw(N);
        fill_input(x);

        // -------- MKL setup (descriptor) --------
        DFTI_DESCRIPTOR_HANDLE desc = nullptr;
        auto mkl_make_desc = [&]() {
            if (desc) DftiFreeDescriptor(&desc);
            MKL_LONG st = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
            if (st != 0) die("DftiCreateDescriptor failed");
            st = DftiCommitDescriptor(desc);
            if (st != 0) die("DftiCommitDescriptor failed");
        };

        // -------- FFTW setup (plan) --------
        fftw_plan plan = nullptr;
        const unsigned fftw_flag = opt.fftw_measure ? FFTW_MEASURE : FFTW_ESTIMATE;

        auto fftw_make_plan = [&]() {
            if (plan) fftw_destroy_plan(plan);
            plan = fftw_plan_dft_1d(
                N,
                reinterpret_cast<fftw_complex*>(x.data()),
                reinterpret_cast<fftw_complex*>(y_fftw.data()),
                FFTW_FORWARD,
                fftw_flag
            );
            if (!plan) die("fftw_plan_dft_1d failed");
        };

        if (!opt.include_plan) {
            mkl_make_desc();
            fftw_make_plan();
        }

        // ---- benchmark MKL ----
        Stat mkl_stat = bench_ms([&]() {
            if (opt.include_plan) mkl_make_desc();
            y_mkl = x;
            MKL_LONG st = DftiComputeForward(desc, y_mkl.data());
            if (st != 0) die("DftiComputeForward failed");
            if (opt.include_plan) DftiFreeDescriptor(&desc);
        }, opt.warmup, opt.iters);

        // ---- benchmark FFTW ----
        Stat fftw_stat = bench_ms([&]() {
            if (opt.include_plan) fftw_make_plan();
            // FFTW reads x, writes y_fftw (out-of-place)
            fftw_execute(plan);
            if (opt.include_plan) fftw_destroy_plan(plan), plan = nullptr;
        }, opt.warmup, opt.iters);

        if (!opt.include_plan) {
            DftiFreeDescriptor(&desc);
            fftw_destroy_plan(plan);
            plan = nullptr;
        }

        // correctness
        double err = max_abs_err(y_mkl, y_fftw);

        const char* flag_name = opt.fftw_measure ? "MEASURE" : "ESTIMATE";
        std::printf(
            "N=%d  T=%d  FFTW_%s  include_plan=%d\n"
            "  MKL : best %.3f ms, avg %.3f ms\n"
            "  FFTW: best %.3f ms, avg %.3f ms\n"
            "  max |MKL-FFTW| = %.17e\n",
            N, opt.threads, flag_name, opt.include_plan ? 1 : 0,
            mkl_stat.best_ms, mkl_stat.avg_ms,
            fftw_stat.best_ms, fftw_stat.avg_ms,
            err
        );

        if (csv) {
            std::fprintf(
                csv, "%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6e\n",
                N, opt.threads, flag_name, opt.include_plan ? 1 : 0,
                mkl_stat.best_ms, mkl_stat.avg_ms,
                fftw_stat.best_ms, fftw_stat.avg_ms,
                err
            );
            std::fflush(csv);
        }
    };

    if (opt.n != -1) {
        run_one(opt.n);
    } else {
        for (int p = opt.min_pow; p <= opt.max_pow; p++) {
            run_one(1 << p);
        }
    }

    if (csv) std::fclose(csv);
    fftw_cleanup_threads();
    return 0;
}
