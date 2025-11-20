// nvcc scale2.cu -o scale2

#include <cstdio>
#include <cuda_runtime.h>

// kernel
__global__ void scale2(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 2.0f;
}

int main() {
  const int n = 16;                                                      // array of floats                  
  float h_x[n];                                                          // host array (lives in cpu)
  for (int i = 0; i < n; ++i)                                            // array fill loop
    h_x[i] = float(i);

  float* d_x = nullptr;                                                  // pointer that points to device mem
  cudaMalloc(&d_x, n * sizeof(float));                                   // allocate n floats to GPU, d_x points to GPU mem space
  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);       // copies n floats from host to device

  const int B = 256;                                                     // block size (threads per block)
  const int G = (n + B - 1) / B;                                         // grid size
  scale2<<<G, B>>>(d_x, n);                                              // launch kernel (G * B threads, each running scale2)
  cudaDeviceSynchronize();                                               // wait until all GPU work is completed

  cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);       // copy back floats from device to host
  cudaFree(d_x);                                                         // free device memory

  for (int i = 0; i < n; ++i) printf("%d: %f\n", i, h_x[i]);             // print results
  return 0;
}