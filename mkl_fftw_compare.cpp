// mkl_fftw_compare.cpp
// Compare FFTW3 vs Intel MKL (DFTI) for 1D complex-to-complex FFT.
// Build (example):
//   c++ -O3 -std=c++17 mkl_fftw_compare.cpp -o mkl_fftw_compare \
//     -I"$MKLROOT/include" -L"$MKLROOT/lib/intel64" -lmkl_rt \
//     -lfftw3 -lfftw3_threads -lm -lpthread -ldl -fopenmp
//
// Run (example):
//   source /opt/intel/oneapi/setvars.sh
//   ./mkl_fftw_compare -min_pow 10 -max_pow 22 -threads 8 -fftw MEASURE -iters 50 -warmup 5 -csv out.csv
//
// Notes:
// - FFTW uses its own threads interface (fftw_init_threads + fftw_plan_with_nthreads).
// - MKL uses mkl_set_dynamic / mkl_set_num_threads (requires <mkl.h>).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <fftw3.h>

#include "mkl_dfti.h"
#include <mkl.h>        // for mkl_set_num_threads, mkl_set_dynamic

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

using cd = std::complex<double>;

static void die(const std::string& msg) {
    std::cerr << "Error: " << msg << "\n";
    std::exit(1);
}

static double now_ms() {
    using clk = std::chrono::steady_clock;
    static auto t0 = clk::now();
    auto t = clk::now();
    return std::chrono::duration<double, std::milli>(t - t0).count();
}

static double max_abs_err(const std::vector<cd>& a, const std::vector<cd>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double e = std::abs(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

static void make_input(std::vector<cd>& x) {
    // deterministic, non-trivial signal
    const int N = (int)x.size();
    for (int i = 0; i < N; i++) {
        double t = 2.0 * M_PI * i / (double)N;
        double re = std::sin(10*t) + 0.1*std::cos(3*t) + 0.01*std::sin(123*t);
        double im = 0.2*std::cos(7*t) - 0.05*std::sin(5*t);
        x[i] = cd(re, im);
    }
}

enum class FFTWFlag { ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE };

static FFTWFlag parse_fftw_flag(const std::string& s) {
    if (s == "ESTIMATE") return FFTWFlag::ESTIMATE;
    if (s == "MEASURE") return FFTWFlag::MEASURE;
    if (s == "PATIENT") return FFTWFlag::PATIENT;
    if (s == "EXHAUSTIVE") return FFTWFlag::EXHAUSTIVE;
    die("Unknown FFTW flag: " + s + " (use ESTIMATE/MEASURE/PATIENT/EXHAUSTIVE)");
    return FFTWFlag::ESTIMATE;
}

static unsigned to_fftw_flag(FFTWFlag f) {
    switch (f) {
        case FFTWFlag::ESTIMATE:   return FFTW_ESTIMATE;
        case FFTWFlag::MEASURE:    return FFTW_MEASURE;
        case FFTWFlag::PATIENT:    return FFTW_PATIENT;
        case FFTWFlag::EXHAUSTIVE: return FFTW_EXHAUSTIVE;
    }
    return FFTW_ESTIMATE;
}

#ifdef __linux__
static std::vector<int> parse_cpu_list(const std::string& s) {
    // Accept formats: "0-7", "0,2,4,6", "0-3,8-11"
    std::vector<int> cpus;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == ',')) i++;
        if (i >= s.size()) break;

        // parse number
        auto parse_int = [&](size_t& j) -> int {
            if (j >= s.size() || s[j] < '0' || s[j] > '9') die("Bad -pin cpu list: " + s);
            int v = 0;
            while (j < s.size() && s[j] >= '0' && s[j] <= '9') {
                v = v * 10 + (s[j] - '0');
                j++;
            }
            return v;
        };

        size_t j = i;
        int a = parse_int(j);

        if (j < s.size() && s[j] == '-') {
            j++;
            int b = parse_int(j);
            if (b < a) die("Bad -pin range: " + s);
            for (int v = a; v <= b; v++) cpus.push_back(v);
        } else {
            cpus.push_back(a);
        }
        i = j;
        while (i < s.size() && s[i] != ',' ) i++;
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

static void pin_to_cpus(const std::string& cpu_list) {
    auto cpus = parse_cpu_list(cpu_list);
    if (cpus.empty()) return;

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        die("Failed to set CPU affinity");
    }
}
#endif

struct Options {
    int min_pow = -1;
    int max_pow = -1;
    int n = -1;
    int threads = 1;
    int warmup = 5;
    int iters  = 50;
    bool include_plan = false;
    bool verify = true;
    FFTWFlag fftw_flag = FFTWFlag::ESTIMATE;
    std::string csv_path;
    std::string pin_cpus; // linux only
};

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const std::string& k) -> std::string {
            if (i + 1 >= argc) die("Missing value for " + k);
            return std::string(argv[++i]);
        };

        if (a == "-min_pow") opt.min_pow = std::stoi(need(a));
        else if (a == "-max_pow") opt.max_pow = std::stoi(need(a));
        else if (a == "-n") opt.n = std::stoi(need(a));
        else if (a == "-threads") opt.threads = std::stoi(need(a));
        else if (a == "-warmup") opt.warmup = std::stoi(need(a));
        else if (a == "-iters") opt.iters = std::stoi(need(a));
        else if (a == "-include_plan") opt.include_plan = true;
        else if (a == "-no_verify") opt.verify = false;
        else if (a == "-fftw") opt.fftw_flag = parse_fftw_flag(need(a));
        else if (a == "-csv") opt.csv_path = need(a);
#ifdef __linux__
        else if (a == "-pin") opt.pin_cpus = need(a);
#endif
        else {
            die("Unknown arg: " + a);
        }
    }

    if (opt.n < 0) {
        if (opt.min_pow < 0 || opt.max_pow < 0 || opt.max_pow < opt.min_pow) {
            die("Provide either -n N, or -min_pow A -max_pow B");
        }
    }
    if (opt.threads <= 0) die("-threads must be >= 1");
    if (opt.iters <= 0) die("-iters must be >= 1");
    if (opt.warmup < 0) die("-warmup must be >= 0");
    return opt;
}

struct Stats {
    double best_ms = 0;
    double avg_ms = 0;
};

static Stats summarize(const std::vector<double>& ms) {
    if (ms.empty()) return Stats{0,0};
    double best = ms[0], sum = 0;
    for (double x : ms) {
        best = std::min(best, x);
        sum += x;
    }
    return Stats{best, sum / (double)ms.size()};
}

// ---- FFTW runner ----
struct FFTWRunner {
    int N;
    int threads;
    unsigned flag;
    bool in_place;
    fftw_plan plan = nullptr;
    std::vector<cd> in, out;

    FFTWRunner(int n, int t, unsigned f, bool inplace)
        : N(n), threads(t), flag(f), in_place(inplace), in(n), out(n) {
        // init threads each time is ok for simple tool; you can also init once globally
        if (!fftw_init_threads()) die("fftw_init_threads failed");
    }

    ~FFTWRunner() {
        if (plan) fftw_destroy_plan(plan);
        fftw_cleanup_threads();
    }

    double make_plan(const std::vector<cd>& x) {
        in = x;
        if (in_place) out = in;

        fftw_plan_with_nthreads(threads);

        double t0 = now_ms();
        if (in_place) {
            plan = fftw_plan_dft_1d(
                N,
                reinterpret_cast<fftw_complex*>(out.data()),
                reinterpret_cast<fftw_complex*>(out.data()),
                FFTW_FORWARD,
                flag
            );
        } else {
            plan = fftw_plan_dft_1d(
                N,
                reinterpret_cast<fftw_complex*>(in.data()),
                reinterpret_cast<fftw_complex*>(out.data()),
                FFTW_FORWARD,
                flag
            );
        }
        if (!plan) die("fftw_plan_dft_1d failed");
        double t1 = now_ms();
        return t1 - t0;
    }

    void exec() {
        fftw_execute(plan);
    }

    const std::vector<cd>& result() const { return out; }
};

// ---- MKL runner ----
struct MKLRunner {
    int N;
    int threads;
    bool in_place;
    DFTI_DESCRIPTOR_HANDLE desc = nullptr;
    std::vector<cd> buf;        // for in-place
    std::vector<cd> out;        // for out-of-place (we still keep result here)
    std::vector<cd> in;         // for out-of-place

    MKLRunner(int n, int t, bool inplace)
        : N(n), threads(t), in_place(inplace), buf(n), out(n), in(n) {}

    ~MKLRunner() {
        if (desc) DftiFreeDescriptor(&desc);
    }

    double make_plan(const std::vector<cd>& x) {
        // set MKL threading policy
        mkl_set_dynamic(0);
        mkl_set_num_threads(threads);

        double t0 = now_ms();

        MKL_LONG status = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        if (status != 0) die("DftiCreateDescriptor failed: " + std::to_string(status));

        if (!in_place) {
            // MKL can do out-of-place with placement option
            status = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            if (status != 0) die("DftiSetValue(PLACEMENT) failed: " + std::to_string(status));
        }

        status = DftiCommitDescriptor(desc);
        if (status != 0) die("DftiCommitDescriptor failed: " + std::to_string(status));

        // set input buffers
        if (in_place) {
            buf = x;
        } else {
            in = x;
            std::fill(out.begin(), out.end(), cd(0,0));
        }

        double t1 = now_ms();
        return t1 - t0;
    }

    void exec() {
        MKL_LONG status;
        if (in_place) {
            status = DftiComputeForward(desc, buf.data());
            if (status != 0) die("DftiComputeForward failed: " + std::to_string(status));
        } else {
            status = DftiComputeForward(desc, in.data(), out.data());
            if (status != 0) die("DftiComputeForward(out-of-place) failed: " + std::to_string(status));
        }
    }

    std::vector<cd> result() const {
        return in_place ? buf : out;
    }
};

static void run_one_size(int N, const Options& opt, std::ofstream* csv) {
    std::vector<cd> x(N);
    make_input(x);

    const unsigned fftw_flag = to_fftw_flag(opt.fftw_flag);

    // In practice: many libs are fastest in-place; keep it consistent
    const bool inplace = true;

    // --- FFTW ---
    FFTWRunner fftw(N, opt.threads, fftw_flag, inplace);
    double fftw_plan_ms = fftw.make_plan(x);

    // Warmup
    for (int i = 0; i < opt.warmup; i++) fftw.exec();

    std::vector<double> fftw_times;
    fftw_times.reserve(opt.iters);
    for (int i = 0; i < opt.iters; i++) {
        // reset input each iter to be fair (avoid "already transformed" effects)
        if (inplace) fftw.out = x;
        else fftw.in = x;

        double t0 = now_ms();
        fftw.exec();
        double t1 = now_ms();
        double dt = t1 - t0;
        if (opt.include_plan) dt += fftw_plan_ms;
        fftw_times.push_back(dt);
    }
    Stats fftw_s = summarize(fftw_times);

    // --- MKL ---
    MKLRunner mkl(N, opt.threads, inplace);
    double mkl_plan_ms = mkl.make_plan(x);

    for (int i = 0; i < opt.warmup; i++) mkl.exec();

    std::vector<double> mkl_times;
    mkl_times.reserve(opt.iters);
    for (int i = 0; i < opt.iters; i++) {
        if (inplace) mkl.buf = x;
        else mkl.in = x;

        double t0 = now_ms();
        mkl.exec();
        double t1 = now_ms();
        double dt = t1 - t0;
        if (opt.include_plan) dt += mkl_plan_ms;
        mkl_times.push_back(dt);
    }
    Stats mkl_s = summarize(mkl_times);

    double err = 0.0;
    if (opt.verify) {
        auto y_mkl = mkl.result();
        const auto& y_fftw = fftw.result();
        err = max_abs_err(y_mkl, y_fftw);
    }

    // Print
    std::cout << "N=" << N
              << "  T=" << opt.threads
              << "  FFTW_" << (opt.fftw_flag==FFTWFlag::ESTIMATE? "ESTIMATE":
                             opt.fftw_flag==FFTWFlag::MEASURE ? "MEASURE":
                             opt.fftw_flag==FFTWFlag::PATIENT ? "PATIENT":"EXHAUSTIVE")
              << "  include_plan=" << (opt.include_plan?1:0)
              << "\n";
    std::cout << "  MKL : best " << std::fixed << std::setprecision(3) << mkl_s.best_ms
              << " ms, avg " << mkl_s.avg_ms << " ms\n";
    std::cout << "  FFTW: best " << fftw_s.best_ms
              << " ms, avg " << fftw_s.avg_ms << " ms\n";
    if (opt.verify) {
        std::cout << "  max |MKL-FFTW| = " << std::scientific << std::setprecision(3) << err << "\n";
        std::cout << std::fixed;
    }

    if (csv && csv->good()) {
        (*csv) << N << ","
               << opt.threads << ","
               << (opt.fftw_flag==FFTWFlag::ESTIMATE? "ESTIMATE":
                   opt.fftw_flag==FFTWFlag::MEASURE ? "MEASURE":
                   opt.fftw_flag==FFTWFlag::PATIENT ? "PATIENT":"EXHAUSTIVE") << ","
               << (opt.include_plan?1:0) << ","
               << std::setprecision(6) << mkl_s.best_ms << ","
               << mkl_s.avg_ms << ","
               << fftw_s.best_ms << ","
               << fftw_s.avg_ms << ","
               << std::scientific << err
               << "\n";
        (*csv) << std::fixed;
    }
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

#ifdef __linux__
    if (!opt.pin_cpus.empty()) {
        pin_to_cpus(opt.pin_cpus);
    }
#endif

    std::ofstream csv;
    std::ofstream* csvp = nullptr;
    if (!opt.csv_path.empty()) {
        csv.open(opt.csv_path);
        if (!csv) die("Failed to open csv: " + opt.csv_path);
        csv << "N,threads,fftw_flag,include_plan,mkl_best_ms,mkl_avg_ms,fftw_best_ms,fftw_avg_ms,max_err\n";
        csvp = &csv;
    }

    if (opt.n > 0) {
        run_one_size(opt.n, opt, csvp);
    } else {
        for (int p = opt.min_pow; p <= opt.max_pow; p++) {
            int N = 1 << p;
            run_one_size(N, opt, csvp);
        }
    }
    return 0;
}
