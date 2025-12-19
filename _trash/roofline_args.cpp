#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
static double omp_get_wtime(){
    using clk = std::chrono::steady_clock;
    static auto t0 = clk::now();
    auto t = clk::now();
    return std::chrono::duration<double>(t - t0).count();
}
static int omp_get_max_threads(){ return 1; }
#endif

// Simple aligned allocation
static void* xaligned_alloc(size_t alignment, size_t size){
#if defined(_ISOC11_SOURCE)
    return aligned_alloc(alignment, ((size + alignment - 1)/alignment)*alignment);
#elif defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* p=nullptr;
    if(posix_memalign(&p, alignment, size)!=0) return nullptr;
    return p;
#endif
}

static void xaligned_free(void* p){
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

// STREAM-like triad: A[i] = B[i] + scalar*C[i]
// Reports best GB/s among trials.
static double measure_bandwidth_gbs(size_t n_elems, int trials){
    const double scalar = 3.0;
    const size_t bytes = n_elems * sizeof(double);

    double* A = (double*)xaligned_alloc(64, bytes);
    double* B = (double*)xaligned_alloc(64, bytes);
    double* C = (double*)xaligned_alloc(64, bytes);
    if(!A || !B || !C){
        std::cerr << "Allocation failed. Try smaller n_elems.\n";
        std::exit(1);
    }

    #pragma omp parallel for
    for(size_t i=0;i<n_elems;i++){
        A[i] = 0.0;
        B[i] = 1.0 + (double)(i & 7);
        C[i] = 2.0 + (double)(i & 15);
    }

    // Warmup
    for(int w=0; w<2; w++){
        #pragma omp parallel for
        for(size_t i=0;i<n_elems;i++){
            A[i] = B[i] + scalar*C[i];
        }
    }

    double best_gbs = 0.0;
    for(int t=0;t<trials;t++){
        double t0 = omp_get_wtime();
        #pragma omp parallel for
        for(size_t i=0;i<n_elems;i++){
            A[i] = B[i] + scalar*C[i];
        }
        double t1 = omp_get_wtime();
        double sec = t1 - t0;

        // STREAM triad nominally counts: read B + read C + write A = 3 words
        // (write allocate / RFO effects are platform-dependent; measure is still useful as "sustained".)
        double bytes_moved = 3.0 * (double)bytes;
        double gbs = (bytes_moved / sec) / 1e9;
        best_gbs = std::max(best_gbs, gbs);
    }

    // Prevent dead-code elimination
    volatile double sink = A[n_elems/2];
    (void)sink;

    xaligned_free(A); xaligned_free(B); xaligned_free(C);
    return best_gbs;
}

// Compute microbenchmark: many independent FMAs in registers.
// Compile with -O3 -ffast-math -march=native for best results.
static double measure_peak_gflops(double target_seconds){
    const double alpha = 1.0000001;
    const double beta  = 0.9999997;

    // 32 independent accumulators
    double a[32];
    for(int i=0;i<32;i++) a[i] = 1.0 + 0.001*i;

    // Calibrate iterations so the timed region lasts ~target_seconds.
    long long iters = 1;
    auto run = [&](long long k){
        double t0 = omp_get_wtime();
        for(long long i=0;i<k;i++){
            #pragma unroll
            for(int j=0;j<32;j++){
                a[j] = a[j]*alpha + beta; // expect FMA with fast-math
            }
        }
        double t1 = omp_get_wtime();
        return t1 - t0;
    };

    // Increase iters until we exceed target_seconds/4
    while(true){
        double sec = run(iters);
        if(sec > target_seconds/4.0) {
            // Scale to target_seconds
            long long scaled = (long long)(iters * (target_seconds / sec));
            iters = std::max(1LL, scaled);
            break;
        }
        if(iters > (1LL<<62)) break;
        iters *= 2;
    }

    // Timed runs: take best
    double best_gflops = 0.0;
    for(int t=0;t<5;t++){
        // reset a a bit
        for(int i=0;i<32;i++) a[i] = 1.0 + 0.001*i;
        double t0 = omp_get_wtime();
        for(long long i=0;i<iters;i++){
            #pragma unroll
            for(int j=0;j<32;j++){
                a[j] = a[j]*alpha + beta;
            }
        }
        double t1 = omp_get_wtime();
        double sec = t1 - t0;

        // Each a[j]*alpha+beta is 2 FLOPs if fused (mul+add). Even if not fused, still 2 FLOPs.
        double flops = (double)iters * 32.0 * 2.0;
        double gflops = (flops / sec) / 1e9;
        best_gflops = std::max(best_gflops, gflops);
    }

    // Prevent DCE
    volatile double sink = a[0] + a[31];
    (void)sink;

    return best_gflops;
}

static double fft_flops_complex(size_t N){
    // Common rough model: ~ 5 N log2(N) real FLOPs for a complex FFT.
    return 5.0 * (double)N * std::log2((double)N);
}

static double fft_bytes_model(size_t N, bool fp64, bool out_of_place){
    // Very rough model for bytes moved to main memory.
    // Complex element size = 2 * sizeof(float/double)
    double elem = fp64 ? 16.0 : 8.0;
    double array_bytes = elem * (double)N;
    // If out-of-place: read input + write output per stage (rough) => 2*array_bytes per stage.
    // If in-place: still typically reads+writes, but could be less; keep same order.
    double per_stage = out_of_place ? 2.0 * array_bytes : 2.0 * array_bytes;
    return per_stage * std::log2((double)N);
}

int main(int argc, char** argv){
    // Defaults chosen to be safe on typical machines; override via args.
    size_t stream_mb = 64; // per array
    int trials = 5;
    double compute_target_sec = 0.3;

    size_t fftN = 0;
    double fft_sec = 0.0;
    bool fp64 = true;
    bool out_of_place = true;

    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        auto need = [&](const char* name){
            if(i+1>=argc){
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if(a=="--stream-mb") stream_mb = (size_t)std::stoull(need("--stream-mb"));
        else if(a=="--trials") trials = std::stoi(need("--trials"));
        else if(a=="--compute-sec") compute_target_sec = std::stod(need("--compute-sec"));
        else if(a=="--fft-n") fftN = (size_t)std::stoull(need("--fft-n"));
        else if(a=="--fft-sec") fft_sec = std::stod(need("--fft-sec"));
        else if(a=="--fp32") fp64 = false;
        else if(a=="--in-place") out_of_place = false;
        else if(a=="--help"){
            std::cout << "Usage: roofline_args [--stream-mb MB] [--trials T] [--compute-sec S]\n"
                      << "                     [--fft-n N --fft-sec SEC] [--fp32] [--in-place]\n";
            return 0;
        }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Threads(max): " << omp_get_max_threads() << "\n";

    // Bandwidth measurement
    size_t bytes_per_array = stream_mb * 1024ULL * 1024ULL;
    size_t n_elems = bytes_per_array / sizeof(double);
    double beta_gbs = measure_bandwidth_gbs(n_elems, trials);

    // Peak compute measurement
    double pi_gflops = measure_peak_gflops(compute_target_sec);

    std::cout << "\nMeasured roofline params (rough):\n";
    std::cout << "  beta (peak bandwidth)   = " << beta_gbs << " GB/s\n";
    std::cout << "  pi   (peak compute)     = " << pi_gflops << " GFLOP/s\n";
    if(beta_gbs > 0) {
        std::cout << "  ridge point (pi/beta)   = " << (pi_gflops / beta_gbs) << " FLOP/byte\n";
    }

    if(fftN > 0 && fft_sec > 0.0){
        double flops = fft_flops_complex(fftN);
        double gflops = (flops / fft_sec) / 1e9;
        double bytes = fft_bytes_model(fftN, fp64, out_of_place);
        double I = flops / bytes;
        std::cout << "\nFFT point (model-based):\n";
        std::cout << "  N                      = " << fftN << "\n";
        std::cout << "  time                   = " << fft_sec << " s\n";
        std::cout << "  perf (5Nlog2N / time)   = " << gflops << " GFLOP/s\n";
        std::cout << "  intensity (model)       = " << I << " FLOP/byte\n";
        std::cout << "  note: intensity uses a rough bytes model; for real roofline, prefer measured DRAM bytes.\n";
    }

    return 0;
}
