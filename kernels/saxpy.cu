#include <cuda_runtime.h>

__global__ void saxpy_kernel(float alpha, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    y[i] = alpha * x[i] + y[i];
  }
}

void launch_saxpy(float alpha, const float* x, float* y, int n, int blocks, int threads_per_block) {
  saxpy_kernel<<<blocks, threads_per_block>>>(alpha, x, y, n);
}