#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

/* CUDA error checking */
/*
   Most CUDA runtime calls return cudaError_t.
   Wrapping them with CUDA_CHECK makes failures visible immediately.
   This is useful because CUDA errors can otherwise be easy to miss.
*/
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

/* CUDA kernel */
/*
   vecadd | c[i] = a[i] + b[i] | 3 = 1 + 2
   threadIdx.x gives the thread's index inside its block.
   blockIdx.x gives the block's index inside the grid.
   blockDim.x gives the number of threads per block.
*/
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
  /* maps a block/thread pair to one linear array index */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  /* bounds check wrapping computation */
  for (; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }

}

int main() {
  /* number of elements in each vector: 1 << 26 == 2^26 */
  int n = 1 << 26;

  /* number of bytes needed for CUDA memory allocation */
  size_t bytes = n * sizeof(float);

  /* host memory allocation */
  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n);

  /* host input initialization */
  for (int i = 0; i < n; i++) {
    /* index based values */
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i*2);
  }

  /* device pointer declaration */
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  /* device memory allocation */
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  /* copy input data from host to device */
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
  
  /* kernel launch config */
  int threads_per_block = 256;
  int blocks = (n + threads_per_block - 1) / threads_per_block;
  
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  /* warm up launch */
  vecadd<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  
  /* kernel launch */
  vecadd<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
  
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double bytes_moved = 3.0 * n * sizeof(float);
  double seconds = ms / 1000.0;
  double bandwidth_gb_s = bytes_moved / seconds / 1e9;

  /* copy output data from device to host */
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  /* correctness check 1.0 + 2.0 = 3.0 */
  bool ok = true;
  for (int i = 0; i < n; i++) {
    float expected = h_a[i] + h_b[i];
    if (std::fabs(h_c[i] - expected) > 1e-5f) {
      ok = false;
      std::cerr << "Mismatch at index " << i << ": got " << h_c[i] << ", expected" << expected << "\n";
      break;
    }
  }
  
  /* print summary */
  std::cout << "vecadd: " << (ok ? "PASS" : "FAIL") << "\n";
  std::cout << "n: " << n << "\n";
  std::cout << "blocks: " << blocks << "\n";
  std::cout << "threads per block: " << threads_per_block << "\n";
  std::cout << "kernel time: " << ms << "ms\n";
  std::cout << "est. bandwidth: " << bandwidth_gb_s << " GB/s\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  /* clean up device memory */
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  /* return 0 for success and 1 for failure */
  return ok ? 0 : 1;
}