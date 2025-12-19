// fft_a1.h
#pragma once
#include <complex>

struct fft_a1_plan {
  int n;
  int threads;
  bool inplace;
  std::complex<double>* in;
  std::complex<double>* out;

  std::complex<double>* scratch; // size n
  int n1, n2; // n = n1 * n2
};

fft_a1_plan fft_a1_plan_dft_1d(int n,
                              std::complex<double>* in,
                              std::complex<double>* out,
                              bool inplace,
                              int threads);

void fft_a1_execute(fft_a1_plan& p);
void fft_a1_destroy_plan(fft_a1_plan& p);
