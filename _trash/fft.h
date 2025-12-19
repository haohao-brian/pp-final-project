#ifndef FFT_H
#define FFT_H
#endif

#include <complex>
#include "immintrin.h"
#include <pthread.h>

typedef long long int len_t;

struct complex_simd {
    __m256 re, im;
};

struct fft_plan {
    len_t n, upper_n;
    std::complex<double> *in, *out, *out_pad, *ws;
    bool reverse;
    int num_threads;
    bool SIMD, pth;
    pthread_barrier_t *barr;
};

struct pthread_args {
    fft_plan *plan;
    int thread_id;
};

void sequential_fft_itr(std::complex<double>* output, std::complex<double>* ws, len_t n, bool reverse, int num_threads);

fft_plan fft_plan_dft_1d(len_t n, std::complex<double> *in, std::complex<double> *out, bool reverse, int num_threads, bool SIMD, bool pth);

void fft_execute(fft_plan &plan);

void fft_destroy_plan(fft_plan &plan);

// TODO: define complex number as structure of two int