#include "fft_cuda.h"
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cstdio>

__device__ __host__ cpxcuda operator+ (const cpxcuda &a, const cpxcuda &b) {
    cpxcuda result;
    result.re = a.re + b.re;
    result.im = a.im + b.im;
    return result;
}

__device__ __host__ cpxcuda operator- (const cpxcuda &a, const cpxcuda &b) {
    cpxcuda result;
    result.re = a.re - b.re;
    result.im = a.im - b.im;
    return result;
}

__device__ __host__ cpxcuda operator* (const cpxcuda &a, const cpxcuda &b) {
    cpxcuda result;
    result.re = a.re * b.re - a.im * b.im;
    result.im = a.re * b.im + a.im * b.re;
    return result;
}

cpxcuda e_imaginary_cuda(double x) {
    cpxcuda w;
    w.re = cos(x);
    w.im = sin(x);
    return w;
}

__global__ void kernel(cpxcuda *out_device, cpxcuda *ws, int n, int batch, bool reverse) {
    int idX = blockIdx.x;
    int idY = threadIdx.y;
    for (int block_size = 2, p = 0; block_size <= n; block_size *= 2, p++) {
        int half_block_size = block_size >> 1;
        int step = n / block_size;
        for (int i = idY; i < n/2; i += blockDim.y) {
            int j = i & (half_block_size - 1);
            int index = ((i >> p) << (p + 1)) + j;
            cpxcuda x = out_device[idX * n + index];
            cpxcuda y = out_device[idX * n + (index + half_block_size)] * ws[j * step];
            out_device[idX * n + index] = x + y;
            out_device[idX * n + (index + half_block_size)] = x - y;
        }
        __syncthreads();
    }
    if (reverse) {
        cpxcuda mult;
        mult.re = 1/(double)n;
        mult.im = 0;
        for (int i = idY; i < n; i += blockDim.y) {
            out_device[idX * n + i] = out_device[idX * n + i] * mult;
        }
    }
}

fft_plan_cuda fft_plan_cuda_1d(int n, int batch, cpxcuda *in, cpxcuda *out, bool reverse) {
    int upper_n = 1;
    while (upper_n < n) upper_n <<= 1;
    
    // swap
    int *index_mapping = (int*) calloc(upper_n, sizeof(int));
    int bit=0;
    while ((1<<bit) < upper_n) bit++;
    for (int i = 0; i < upper_n - 1; i++) {
        index_mapping[i] = (index_mapping[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    }
    index_mapping[upper_n - 1] = upper_n - 1;
    for (int i = 0; i < batch; i++)
        for (int j = 0; j < upper_n; j++)
            out[i * upper_n + index_mapping[j]] = in[i * upper_n + j];

    cpxcuda *out_device = NULL;
    cudaMalloc(&out_device, sizeof(cpxcuda) * batch * upper_n);
    cudaMemcpy(out_device, out, sizeof(cpxcuda) * upper_n * batch, cudaMemcpyHostToDevice);

    cpxcuda *ws = (cpxcuda *) calloc(upper_n / 2, sizeof(cpxcuda));
    ws[0].re = 1; ws[0].im = 0;
    ws[1] = e_imaginary_cuda(double(2 * M_PI) / double(upper_n) * (reverse ? -1 : 1)); 
    for (int i = 2; i < upper_n / 2; i++)
        ws[i] = ws[i - 1] * ws[1];

    cpxcuda *ws_device = NULL;
    cudaMalloc(&ws_device, sizeof(cpxcuda) * upper_n / 2);
    cudaMemcpy(ws_device, ws, sizeof(cpxcuda) * upper_n / 2, cudaMemcpyHostToDevice);

    free(ws);
    free(index_mapping);
    
    fft_plan_cuda plan;
    plan.n = upper_n;
    plan.batch = batch;
    plan.out = out;
    plan.out_device = out_device;
    plan.ws_device = ws_device;
    plan.reverse = reverse;

    return plan;
}

void fft_execute_plan_cuda(fft_plan_cuda &plan) {
    dim3 blockDim(1, 256);
    dim3 gridDim(plan.batch, 1);
    kernel<<<gridDim, blockDim>>>(plan.out_device, plan.ws_device, plan.n, plan.batch, plan.reverse);
    cudaDeviceSynchronize();
    cudaMemcpy(plan.out, plan.out_device, sizeof(cpxcuda) * plan.n * plan.batch, cudaMemcpyDeviceToHost);
}

void fft_destroy_plan_cuda(fft_plan_cuda &plan) {
    cudaFree(plan.ws_device);
    cudaFree(plan.out_device);
}

//reference: CUDA example slides.