#ifndef FFT_CUDA_H
#define FFT_CUDA_H
#endif

struct cpxcuda {
    double re, im;
};

struct fft_plan_cuda {
    int n, batch;
    cpxcuda *out, *out_device, *ws_device;
    bool reverse;
};

fft_plan_cuda fft_plan_cuda_1d(int n, int batch, cpxcuda *in, cpxcuda *out, bool reverse);

void fft_execute_plan_cuda(fft_plan_cuda &plan);

void fft_destroy_plan_cuda(fft_plan_cuda &plan);