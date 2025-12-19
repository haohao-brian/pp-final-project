#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>

static double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static void usage(const char* prog) {
  std::fprintf(stderr,
    "Usage: %s [--stream-mb M] [--trials T] [--compute-sec S]\n"
    "  --stream-mb   per-array size in MiB (3 arrays allocated)\n"
    "  --trials      number of timed trials (warmup included)\n"
    "  --compute-sec compute benchmark duration (seconds)\n",
    prog);
}

int main(int argc, char** argv) {
  int stream_mb = 32;
  int trials = 5;
  double compute_sec = 0.2;

  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--stream-mb") && i + 1 < argc) stream_mb = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) trials = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--compute-sec") && i + 1 < argc) compute_sec = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
    else { std::fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
  }

  int maxT = omp_get_max_threads();
  std::cout << "Threads(max): " << maxT << "\n" << std::flush;

  // -------------------------
  // STREAM-like triad (double)
  // -------------------------
  const size_t bytes_per_array = (size_t)stream_mb * 1024ull * 1024ull;
  const size_t n = bytes_per_array / sizeof(double);
  if (n < 1024) { std::fprintf(stderr, "stream_mb too small.\n"); return 2; }

  std::cout << "[1/2] Allocating 3 arrays of " << stream_mb << " MiB each (total ~"
            << (3 * stream_mb) << " MiB) ...\n" << std::flush;

  std::vector<double> a(n), b(n), c(n);

  // First-touch init in parallel (also warms pages)
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
    c[i] = 0.0;
  }

  const double scalar = 3.0;
  auto triad_once = [&]() {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      c[i] = a[i] + scalar * b[i];
    }
  };

  // Warmup
  triad_once();

  std::cout << "[1/2] Measuring bandwidth (" << trials << " trials) ...\n" << std::flush;
  double best_bw_gbs = 0.0;

  // Triad traffic model: read a, read b, write c  => 3 * 8 bytes per element? (a+b + store)
  // Many roofline/STREAM conventions count 24B/elem for triad in double (2 loads + 1 store).
  // We'll use 24B/elem to match common STREAM triad counting.
  const double bytes_per_elem = 24.0;
  const double total_bytes = bytes_per_elem * (double)n;

  for (int t = 0; t < trials; t++) {
    double t0 = now_sec();
    triad_once();
    double t1 = now_sec();
    double dt = t1 - t0;
    double gbs = (total_bytes / dt) / 1e9;
    if (gbs > best_bw_gbs) best_bw_gbs = gbs;
  }

  // Prevent the compiler from getting too clever: checksum
  double checksum = 0.0;
  #pragma omp parallel for reduction(+:checksum)
  for (size_t i = 0; i < n; i += 4096) checksum += c[i];
  std::cout << "[1/2] checksum: " << checksum << "\n";

  // -------------------------
  // Compute peak (all-core)
  // -------------------------
  std::cout << "[2/2] Measuring compute peak (all-core) for ~" << compute_sec
            << " sec ...\n" << std::flush;

  // A small working set to live in L1/L2
  const int m = 1 << 15; // 32768 doubles (~256 KiB)
  std::vector<double> x(m, 1.0), y(m, 2.0);

  double t_start = now_sec();
  uint64_t total_iters = 0;

  #pragma omp parallel
  {
    uint64_t iters = 0;
    double t0 = now_sec();

    // Each inner loop does: y[i] = y[i] * alpha + x[i]
    // That's 2 FLOPs (mul+add) per element, and compilers typically generate FMA.
    const double alpha = 1.0000001;

    while (true) {
      #pragma omp for schedule(static)
      for (int i = 0; i < m; i++) {
        y[i] = y[i] * alpha + x[i];
      }
      iters++;
      double t1 = now_sec();
      if (t1 - t0 >= compute_sec) break;
    }

    #pragma omp atomic
    total_iters += iters;
  }

  double t_end = now_sec();
  double dt_compute = t_end - t_start;

  // FLOPs: per iter, each element does 2 FLOPs
  // total flops = total_iters * m * 2
  double flops = (double)total_iters * (double)m * 2.0;
  double gflops = (flops / dt_compute) / 1e9;

  double pi = gflops;
  double beta = best_bw_gbs;
  double ridge = (beta > 0.0) ? (pi / beta) : 0.0;

  std::cout << "\n=== Roofline args (measured) ===\n";
  std::cout << "beta (GB/s)   = " << beta << "\n";
  std::cout << "pi (GFLOP/s)  = " << pi << "\n";
  std::cout << "ridge (F/B)   = " << ridge << "\n";

  return 0;
}
