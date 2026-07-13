#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

void launch_reduce_sum(const float* x, float* partial_sums, int n, int blocks, int threads_per_block);

int main() {
  int n = 1 << 20;
  int threads_per_block = 256;
  int blocks = (n + threads_per_block - 1) / threads_per_block;

  size_t input_bytes = n * sizeof(float);
  size_t partial_bytes = blocks * sizeof(float);

  std::vector<float> h_x(n);
  std::vector<float> h_partial_sums(blocks);

  for (int i = 0; i < n; i++) { h_x[i] = 1.0f; }

  float* d_x = nullptr;
  float* d_partial_sums = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_partial_sums, partial_bytes));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), input_bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  launch_reduce_sum(d_x, d_partial_sums, n, blocks, threads_per_block);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));

  launch_reduce_sum(d_x, d_partial_sums, n, blocks, threads_per_block);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums, partial_bytes, cudaMemcpyDeviceToHost));

  float gpu_sum = 0.0f;
  for (int i = 0; i < blocks; i++) { gpu_sum += h_partial_sums[i]; }

  float expected = static_cast<float>(n);
  bool ok = std::fabs(gpu_sum - expected) < 1e-3f;

  std::cout << "reduce_sum: " << (ok ? "PASS" : "FAIL") << "\n";
  std::cout << "n: " << n << "\n";
  std::cout << "blocks: " << blocks << "\n";
  std::cout << "threads per block: " << threads_per_block << "\n";
  std::cout << "sum: " << gpu_sum << "\n";
  std::cout << "expected: " << expected << "\n";
  std::cout << "kernel time: " << ms << " ms\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_partial_sums));
  
  return ok ? 0 : 1;
}