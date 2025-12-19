// bench_csv_a1.cpp
// Reproducible FFT benchmark producing ONE CSV row per run.
// Supports:
//   - FFTW3 (complex-to-complex, FP64, 1D)
//   - "ours" FFT (if compiled with -DBENCH_HAVE_OURS and linked with your fft.o / fft.cpp)
//   - A1 FFT (your 4-step / blocked-transpose variant) via fft_a1.h / fft_a1.cpp
//
// Build (FFTW only):
//   g++ -O3 -march=native -ffast-math -fopenmp bench_csv_a1.cpp \
//       -lfftw3 -lfftw3_threads -lpthread -o bench_csv_a1
//
// Build (FFTW + your FFT in this repo):
//   g++ -O3 -march=native -ffast-math -fopenmp bench_csv_a1.cpp -DBENCH_HAVE_OURS \
//       fft.o -lfftw3 -lfftw3_threads -lpthread -o bench_csv_a1
//
// Build (FFTW + A1):
//   g++ -O3 -march=native -ffast-math -fopenmp bench_csv_a1.cpp fft_a1.o \
//       -lfftw3 -lfftw3_threads -lpthread -o bench_csv_a1
//
// Build (FFTW + ours + A1):
//   g++ -O3 -march=native -ffast-math -fopenmp bench_csv_a1.cpp -DBENCH_HAVE_OURS \
//       fft.o fft_a1.o -lfftw3 -lfftw3_threads -lpthread -o bench_csv_a1
//
// Example:
//   ./bench_csv_a1 --impl fftw --n 1048576 --threads 24 --sec 0.3 --plan estimate --inplace 0 --header 1
//   ./bench_csv_a1 --impl a1   --n 1048576 --threads 8  --sec 0.3 --plan estimate --inplace 0 --header 1

#include <fftw3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <complex>
#include <string>

#include "fft_a1.h"   // must provide: fft_a1_plan, fft_a1_plan_dft_1d, fft_a1_execute, fft_a1_destroy_plan

#ifdef BENCH_HAVE_OURS
  #include "fft.h"
#endif

static double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static void die(const char* msg) {
  std::fprintf(stderr, "FATAL: %s\n", msg);
  std::exit(2);
}

static bool streq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

static int parse_int(const char* s, const char* what) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0') { std::fprintf(stderr, "Bad %s: %s\n", what, s); std::exit(2); }
  return (int)v;
}
static long long parse_ll(const char* s, const char* what) {
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (!end || *end != '\0') { std::fprintf(stderr, "Bad %s: %s\n", what, s); std::exit(2); }
  return v;
}
static double parse_double(const char* s, const char* what) {
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (!end || *end != '\0') { std::fprintf(stderr, "Bad %s: %s\n", what, s); std::exit(2); }
  return v;
}

// Deterministic RNG (xorshift64*)
static inline unsigned long long xorshift64s(unsigned long long& x) {
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  return x * 2685821657736338717ULL;
}

static void fill_complex(fftw_complex* buf, int n, unsigned long long seed) {
  unsigned long long s = seed ? seed : 0xdeadbeefcafebabeULL;
  for (int i = 0; i < n; i++) {
    auto r1 = xorshift64s(s);
    auto r2 = xorshift64s(s);
    double a = (r1 >> 11) * (1.0 / 9007199254740992.0);
    double b = (r2 >> 11) * (1.0 / 9007199254740992.0);
    buf[i][0] = a;
    buf[i][1] = b;
  }
}

static void fill_complex_std(std::complex<double>* buf, int n, unsigned long long seed) {
  unsigned long long s = seed ? seed : 0x12345678abcdef00ULL;
  for (int i = 0; i < n; i++) {
    auto r1 = xorshift64s(s);
    auto r2 = xorshift64s(s);
    double a = (r1 >> 11) * (1.0 / 9007199254740992.0);
    double b = (r2 >> 11) * (1.0 / 9007199254740992.0);
    buf[i] = std::complex<double>(a, b);
  }
}

static double checksum_some_fftw(const fftw_complex* out, int n) {
  int idxs[8] = {0, n/7, 2*n/7, 3*n/7, 4*n/7, 5*n/7, 6*n/7, n-1};
  double s = 0.0;
  for (int k = 0; k < 8; k++) {
    int i = idxs[k];
    if (i < 0) i = 0;
    if (i >= n) i = n-1;
    s += out[i][0] * 0.3 + out[i][1] * 0.7;
  }
  return s;
}

static void print_header() {
  std::printf("impl,n,threads,inplace,plan,seed,iters,sec,sec_per_iter,gflops,plan_sec,checksum\n");
}

static void* malloc_aligned(size_t alignment, size_t bytes) {
  void* p = nullptr;
  // posix_memalign requires alignment to be power-of-2 and multiple of sizeof(void*)
  if (posix_memalign(&p, alignment, bytes) != 0) return nullptr;
  return p;
}

int main(int argc, char** argv) {
  std::string impl = "fftw";           // fftw | ours | a1
  int n = 1 << 20;
  int threads = 1;
  bool inplace = false;
  std::string plan_mode = "estimate";  // estimate | measure
  double target_sec = 0.3;
  long long iters_override = -1;
  unsigned long long seed = 1;
  bool header = false;
  const char* wisdom_in = nullptr;
  const char* wisdom_out = nullptr;

  for (int i = 1; i < argc; i++) {
    if (streq(argv[i], "--impl") && i + 1 < argc) impl = argv[++i];
    else if (streq(argv[i], "--n") && i + 1 < argc) n = parse_int(argv[++i], "n");
    else if (streq(argv[i], "--threads") && i + 1 < argc) threads = parse_int(argv[++i], "threads");
    else if (streq(argv[i], "--inplace") && i + 1 < argc) inplace = parse_int(argv[++i], "inplace") != 0;
    else if (streq(argv[i], "--plan") && i + 1 < argc) plan_mode = argv[++i];
    else if (streq(argv[i], "--sec") && i + 1 < argc) target_sec = parse_double(argv[++i], "sec");
    else if (streq(argv[i], "--iters") && i + 1 < argc) iters_override = parse_ll(argv[++i], "iters");
    else if (streq(argv[i], "--seed") && i + 1 < argc) seed = (unsigned long long)parse_ll(argv[++i], "seed");
    else if (streq(argv[i], "--header") && i + 1 < argc) header = parse_int(argv[++i], "header") != 0;
    else if (streq(argv[i], "--wisdom-in") && i + 1 < argc) wisdom_in = argv[++i];
    else if (streq(argv[i], "--wisdom-out") && i + 1 < argc) wisdom_out = argv[++i];
    else if (streq(argv[i], "--help")) {
      std::fprintf(stderr,
        "bench_csv_a1 options:\n"
        "  --impl fftw|ours|a1\n"
        "  --n N\n"
        "  --threads T\n"
        "  --inplace 0|1\n"
        "  --plan estimate|measure\n"
        "  --sec S            target exec time for auto-iter selection (default 0.3)\n"
        "  --iters K          override iterations\n"
        "  --seed SEED\n"
        "  --header 0|1\n"
        "  --wisdom-in  file  (FFTW) import wisdom before planning\n"
        "  --wisdom-out file  (FFTW) export wisdom after planning\n");
      return 0;
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      return 2;
    }
  }

  if (n <= 0) die("n must be > 0");
  if (threads <= 0) die("threads must be > 0");
  if (!(plan_mode == "estimate" || plan_mode == "measure")) die("--plan must be estimate|measure");

  if (header) print_header();

  // Convention FLOP model for complex FFT:
  // FLOPs ~= 5 N log2 N
  double log2n = std::log2((double)n);
  double flops_per_fft = 5.0 * (double)n * log2n;

  double plan_sec = 0.0;
  long long iters = (iters_override > 0) ? iters_override : 1;
  double exec_sec = 0.0;
  double checksum = 0.0;

  if (impl == "fftw") {
    fftw_complex* in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (size_t)n);
    fftw_complex* out = inplace ? in : (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (size_t)n);
    if (!in) die("fftw_malloc(in) failed");
    if (!out) die("fftw_malloc(out) failed");

    fill_complex(in, n, seed);

    double t0 = now_sec();
    if (fftw_init_threads() == 0) die("fftw_init_threads failed");
    fftw_plan_with_nthreads(threads);

    if (wisdom_in) { (void)fftw_import_wisdom_from_filename(wisdom_in); }

    unsigned flags = (plan_mode == "measure") ? FFTW_MEASURE : FFTW_ESTIMATE;
    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, flags);
    if (!plan) die("fftw_plan_dft_1d failed");
    plan_sec = now_sec() - t0;

    if (wisdom_out) { (void)fftw_export_wisdom_to_filename(wisdom_out); }

    // Warmup (not timed)
    fftw_execute(plan);

    if (iters_override <= 0) {
      long long k = 1;
      while (true) {
        double s0 = now_sec();
        for (long long i = 0; i < k; i++) fftw_execute(plan);
        double s1 = now_sec();
        double dt = s1 - s0;
        if (dt >= target_sec) { iters = k; exec_sec = dt; break; }
        if (k > (1LL<<30)) { iters = k; exec_sec = dt; break; }
        k *= 2;
      }
    } else {
      double s0 = now_sec();
      for (long long i = 0; i < iters; i++) fftw_execute(plan);
      double s1 = now_sec();
      exec_sec = s1 - s0;
    }

    checksum = checksum_some_fftw(out, n);

    fftw_destroy_plan(plan);
    fftw_cleanup_threads();
    if (!inplace) fftw_free(out);
    fftw_free(in);

  } else if (impl == "a1") {
    // A1: 4-step / blocked transpose FFT
    auto* in  = (std::complex<double>*)malloc_aligned(64, sizeof(std::complex<double>) * (size_t)n);
    auto* out = inplace ? in : (std::complex<double>*)malloc_aligned(64, sizeof(std::complex<double>) * (size_t)n);
    if (!in) die("posix_memalign(in) failed");
    if (!out) die("posix_memalign(out) failed");

    fill_complex_std(in, n, seed);

    double t0 = now_sec();
    fft_a1_plan plan = fft_a1_plan_dft_1d(n, in, out, /*inplace=*/inplace, threads);
    plan_sec = now_sec() - t0;

    // Warmup
    fft_a1_execute(plan);

    if (iters_override <= 0) {
      long long k = 1;
      while (true) {
        double s0 = now_sec();
        for (long long i = 0; i < k; i++) fft_a1_execute(plan);
        double s1 = now_sec();
        double dt = s1 - s0;
        if (dt >= target_sec) { iters = k; exec_sec = dt; break; }
        if (k > (1LL<<30)) { iters = k; exec_sec = dt; break; }
        k *= 2;
      }
    } else {
      double s0 = now_sec();
      for (long long i = 0; i < iters; i++) fft_a1_execute(plan);
      double s1 = now_sec();
      exec_sec = s1 - s0;
    }

    // checksum a few points
    int idxs[8] = {0, n/7, 2*n/7, 3*n/7, 4*n/7, 5*n/7, 6*n/7, n-1};
    double s = 0.0;
    for (int k = 0; k < 8; k++) {
      int i = idxs[k];
      if (i < 0) i = 0;
      if (i >= n) i = n-1;
      s += out[i].real() * 0.3 + out[i].imag() * 0.7;
    }
    checksum = s;

    fft_a1_destroy_plan(plan);
    if (!inplace) std::free(out);
    std::free(in);

  } else if (impl == "ours") {
  #ifdef BENCH_HAVE_OURS
    auto* in  = (std::complex<double>*)malloc_aligned(64, sizeof(std::complex<double>) * (size_t)n);
    auto* out = inplace ? in : (std::complex<double>*)malloc_aligned(64, sizeof(std::complex<double>) * (size_t)n);
    if (!in) die("posix_memalign(in) failed");
    if (!out) die("posix_memalign(out) failed");

    fill_complex_std(in, n, seed);

    double t0 = now_sec();
    // Signature from your repo:
    // fft_plan fft_plan_dft_1d(N, input, out, inverse, num_threads, simd, pthread)
    fft_plan plan = fft_plan_dft_1d(n, in, out, /*inverse=*/false, threads, /*simd=*/false, /*pthread=*/false);
    plan_sec = now_sec() - t0;

    // Warmup
    fft_execute(plan);

    if (iters_override <= 0) {
      long long k = 1;
      while (true) {
        double s0 = now_sec();
        for (long long i = 0; i < k; i++) fft_execute(plan);
        double s1 = now_sec();
        double dt = s1 - s0;
        if (dt >= target_sec) { iters = k; exec_sec = dt; break; }
        if (k > (1LL<<30)) { iters = k; exec_sec = dt; break; }
        k *= 2;
      }
    } else {
      double s0 = now_sec();
      for (long long i = 0; i < iters; i++) fft_execute(plan);
      double s1 = now_sec();
      exec_sec = s1 - s0;
    }

    int idxs[8] = {0, n/7, 2*n/7, 3*n/7, 4*n/7, 5*n/7, 6*n/7, n-1};
    double s = 0.0;
    for (int k = 0; k < 8; k++) {
      int i = idxs[k];
      if (i < 0) i = 0;
      if (i >= n) i = n-1;
      s += out[i].real() * 0.3 + out[i].imag() * 0.7;
    }
    checksum = s;

    fft_destroy_plan(plan);
    if (!inplace) std::free(out);
    std::free(in);
  #else
    die("impl=ours requested but compiled without -DBENCH_HAVE_OURS");
  #endif

  } else {
    die("--impl must be fftw or ours or a1");
  }

  double sec_per_iter = exec_sec / (double)iters;
  double gflops = (flops_per_fft * (double)iters) / (exec_sec * 1e9);

  std::printf("%s,%d,%d,%d,%s,%llu,%lld,%.9f,%.9e,%.6f,%.9f,%.6f\n",
              impl.c_str(), n, threads, inplace ? 1 : 0,
              plan_mode.c_str(), seed, iters,
              exec_sec, sec_per_iter, gflops, plan_sec, checksum);
  return 0;
}
