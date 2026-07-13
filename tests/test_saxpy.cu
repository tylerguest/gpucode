#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
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

void launch_saxpy(float alpha, const float* x, float* y, int n, int blocks,
                  int threads_per_block);

int main() {
  int n = 1 << 26;
  float alpha = 2.0f;
  size_t bytes = n * sizeof(float);

  std::vector<float> h_x(n);
  std::vector<float> h_y(n);
  std::vector<float> h_expected(n);

  for (int i = 0; i < n; i++) {
    h_x[i] = static_cast<float>(i);
    h_y[i] = static_cast<float>(i * 3);
    h_expected[i] = alpha * h_x[i] + h_y[i];
  }

  float* d_x = nullptr;
  float* d_y = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  int threads_per_block = 256;
  int blocks = (n + threads_per_block - 1) / threads_per_block;

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  /* warmup */
  launch_saxpy(alpha, d_x, d_y, n, blocks, threads_per_block);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  /* SAXPY modifies y in-place, so reset it before the timed run. */
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaEventRecord(start));

  launch_saxpy(alpha, d_x, d_y, n, blocks, threads_per_block);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (std::fabs(h_y[i] - h_expected[i]) > 1e-5f) {
      ok = false;
      std::cerr << "Mismatch at index " << i << ": got " << h_y[i]
                << ", expected " << h_expected[i] << "\n";
      break;
    }
  }

  double bytes_moved = 3.0 * n * sizeof(float);
  double seconds = ms / 1000.0;
  double bandwidth_gb_s = bytes_moved / seconds / 1e9;

  std::cout << "saxpy: " << (ok ? "PASS" : "FAIL") << "\n";
  std::cout << "n: " << n << "\n";
  std::cout << "alpha: " << alpha << "\n";
  std::cout << "blocks: " << blocks << "\n";
  std::cout << "threads per block: " << threads_per_block << "\n";
  std::cout << "kernel time: " << ms << " ms\n";
  std::cout << "est. bandwidth: " << bandwidth_gb_s << " GB/s\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  
  return ok ? 0 : 1;
}
