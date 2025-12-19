// fft_a1.cpp
// Minimal 4-step FFT implementation with blocked transpose + simple radix-2 FFT kernel.
// Goal: compile & run immediately for ablation A1 (cache-friendly decomposition).
//
// Notes:
// - This kernel uses bit-reversal + iterative Cooley-Tukey (correct normal-order output).
// - Replace fft_inplace_radix2() with your faster micro-kernel later.
// - Requires N is power-of-two (your benchmarks already do).
//
// Build example:
//   g++ -O3 -march=native -ffast-math -fopenmp -c fft_a1.cpp -o fft_a1.o

#include "fft_a1.h"

#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <algorithm>
#include <cstdint>

static inline bool is_pow2_u32(uint32_t x) {
  return x && ((x & (x - 1)) == 0);
}

static inline int ilog2_u32(uint32_t x) {
  // x must be power of two
#if defined(__GNUG__) || defined(__clang__)
  return 31 - __builtin_clz(x);
#else
  int r = 0;
  while ((x >>= 1) != 0) r++;
  return r;
#endif
}

static inline uint32_t bit_reverse_u32(uint32_t x, int bits) {
  // Reverse lowest 'bits' bits of x.
  // Simple loop; good enough for minimal version.
  uint32_t r = 0;
  for (int i = 0; i < bits; i++) {
    r = (r << 1) | (x & 1u);
    x >>= 1;
  }
  return r;
}

// In-place radix-2 FFT (forward), normal-order output.
// Complexity is OK for "minimal usable" A1.
static void fft_inplace_radix2(std::complex<double>* a, int n) {
  if (n <= 1) return;
  if (!is_pow2_u32((uint32_t)n)) return;

  const int lg = ilog2_u32((uint32_t)n);

  // Bit reversal permutation
  for (uint32_t i = 0; i < (uint32_t)n; i++) {
    uint32_t j = bit_reverse_u32(i, lg);
    if (j > i) std::swap(a[i], a[j]);
  }

  // Iterative FFT
  for (int len = 2; len <= n; len <<= 1) {
    const int half = len >> 1;
    const double ang = -2.0 * M_PI / (double)len;
    const double wlen_re = std::cos(ang);
    const double wlen_im = std::sin(ang);

    for (int i = 0; i < n; i += len) {
      double w_re = 1.0;
      double w_im = 0.0;
      for (int j = 0; j < half; j++) {
        const auto u = a[i + j];
        const auto v0 = a[i + j + half];

        // v = v0 * w
        const double vr = v0.real() * w_re - v0.imag() * w_im;
        const double vi = v0.real() * w_im + v0.imag() * w_re;
        const std::complex<double> v(vr, vi);

        a[i + j] = u + v;
        a[i + j + half] = u - v;

        // w *= wlen
        const double nw_re = w_re * wlen_re - w_im * wlen_im;
        const double nw_im = w_re * wlen_im + w_im * wlen_re;
        w_re = nw_re;
        w_im = nw_im;
      }
    }
  }
}

// Blocked transpose: src is [rows x cols] row-major, dst is [cols x rows] row-major.
static void transpose_blocked(std::complex<double>* dst,
                              const std::complex<double>* src,
                              int rows, int cols,
                              int B = 32) {
  // dst[j*rows + i] = src[i*cols + j]
#pragma omp parallel for collapse(2) schedule(static)
  for (int i0 = 0; i0 < rows; i0 += B) {
    for (int j0 = 0; j0 < cols; j0 += B) {
      const int i_max = std::min(i0 + B, rows);
      const int j_max = std::min(j0 + B, cols);
      for (int i = i0; i < i_max; i++) {
        const int src_base = i * cols;
        for (int j = j0; j < j_max; j++) {
          dst[j * rows + i] = src[src_base + j];
        }
      }
    }
  }
}

// Choose n1, n2 for 4-step.
// Balanced split: n1 = 2^(floor(log2N/2)), n2 = N/n1.
static void choose_factors_pow2(int n, int& n1, int& n2) {
  const int lgN = ilog2_u32((uint32_t)n);
  const int lg1 = lgN / 2;
  n1 = 1 << lg1;
  n2 = n / n1;
}

static inline std::complex<double> cis(double theta) {
  return std::complex<double>(std::cos(theta), std::sin(theta));
}

fft_a1_plan fft_a1_plan_dft_1d(int n,
                              std::complex<double>* in,
                              std::complex<double>* out,
                              bool inplace,
                              int threads) {
  fft_a1_plan p{};
  p.n = n;
  p.threads = threads;
  p.inplace = inplace;
  p.in = in;
  p.out = out;
  p.scratch = nullptr;
  p.n1 = 0;
  p.n2 = 0;

  if (n <= 0 || !is_pow2_u32((uint32_t)n)) {
    // Minimal: just set factors; execute will early-out or no-op.
    p.n1 = n;
    p.n2 = 1;
    return p;
  }

  choose_factors_pow2(n, p.n1, p.n2);

  // scratch size = n complex doubles (16 bytes each)
  // aligned_alloc requires size multiple of alignment.
  const size_t bytes = (size_t)n * sizeof(std::complex<double>);
  const size_t aligned_bytes = (bytes + 63ull) & ~63ull;
  p.scratch = (std::complex<double>*)std::aligned_alloc(64, aligned_bytes);
  if (!p.scratch) {
    // fallback: no scratch => will degrade (execute will handle)
    p.n1 = n;
    p.n2 = 1;
    return p;
  }

  // First-touch scratch pages in parallel (helps NUMA / placement stability)
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int i = 0; i < n; i++) {
    p.scratch[i] = std::complex<double>(0.0, 0.0);
  }

  return p;
}

void fft_a1_execute(fft_a1_plan& p) {
  const int n = p.n;
  if (n <= 1) return;
  if (!is_pow2_u32((uint32_t)n)) return;

  // Work buffer buf0 holds data we operate on.
  // If out-of-place, write final result directly into out by copying input once.
  std::complex<double>* buf0 = p.inplace ? p.in : p.out;
  if (!p.inplace) {
    std::memcpy(buf0, p.in, (size_t)n * sizeof(std::complex<double>));
  }

  // If no scratch, fall back to single FFT on whole buffer.
  if (!p.scratch || p.n2 == 1 || p.n1 == n) {
#pragma omp parallel num_threads(p.threads)
    {
#pragma omp single
      {
        // single-thread for fallback kernel; minimal correctness
        fft_inplace_radix2(buf0, n);
      }
    }
    return;
  }

  std::complex<double>* buf1 = p.scratch;
  const int n1 = p.n1; // row length
  const int n2 = p.n2; // number of rows
  const double invN = 1.0 / (double)n;

  // Stage 1: for each row k (0..n2-1), FFT length n1 on contiguous segment buf0[k*n1 ..]
#pragma omp parallel for num_threads(p.threads) schedule(static)
  for (int k = 0; k < n2; k++) {
    std::complex<double>* row = buf0 + (size_t)k * n1;
    fft_inplace_radix2(row, n1);

    // Twiddle multiply: row[j] *= exp(-2πi * j*k / N)
    // For fixed k, w_j is a geometric progression in j:
    // step = exp(-2πi * k / N), w starts at 1.
    const double ang_step = -2.0 * M_PI * (double)k * invN;
    const std::complex<double> step = cis(ang_step);
    std::complex<double> w(1.0, 0.0);
    for (int j = 0; j < n1; j++) {
      row[j] *= w;
      w *= step;
    }
  }

  // Transpose X[n2 x n1] -> Y[n1 x n2]
  // X is buf0 (row-major with row length n1), Y is buf1 (row-major with row length n2)
  // Y[j*n2 + k] = X[k*n1 + j]
  transpose_blocked(buf1, buf0, /*rows=*/n2, /*cols=*/n1, /*B=*/32);

  // Stage 2: FFT length n2 on each row of Y (there are n1 rows)
#pragma omp parallel for num_threads(p.threads) schedule(static)
  for (int j = 0; j < n1; j++) {
    std::complex<double>* row = buf1 + (size_t)j * n2;
    fft_inplace_radix2(row, n2);
  }

  // Transpose back Y[n1 x n2] -> Z[n2 x n1] into buf0
  transpose_blocked(buf0, buf1, /*rows=*/n1, /*cols=*/n2, /*B=*/32);
}

void fft_a1_destroy_plan(fft_a1_plan& p) {
  if (p.scratch) std::free(p.scratch);
  p.scratch = nullptr;
}
