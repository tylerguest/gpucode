#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// forward-declare your 2D kernel (adjust signature if different)
extern __global__ void twodimkernel(float* x, int width, int height);

static inline void checkCuda(cudaError_t e, const char* msg = "") {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error%s: %s\n", msg, cudaGetErrorString(e));
    exit(1);
  }
}

void bench_2d(int width, int height, int reps) {
  int n = width * height;
  size_t bytes = size_t(n) * sizeof(float);
  float* h = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) h[i] = float(i);

  float* d = nullptr;
  checkCuda(cudaMalloc(&d, bytes));

  cudaEvent_t e_start, e_stop;
  checkCuda(cudaEventCreate(&e_start));
  checkCuda(cudaEventCreate(&e_stop));

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // warmup
  checkCuda(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  twodimkernel<<<grid, block>>>(d, width, height);
  checkCuda(cudaDeviceSynchronize());

  // H2D (Host To Device)
  checkCuda(cudaEventRecord(e_start));
  for (int i = 0; i < reps; ++i)
    checkCuda(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaEventRecord(e_stop));
  checkCuda(cudaEventSynchronize(e_stop));
  float ms_h2d = 0.f; checkCuda(cudaEventElapsedTime(&ms_h2d, e_start, e_stop));

  // kernel-only
  checkCuda(cudaEventRecord(e_start));
  for (int i = 0; i < reps; ++i)
    twodimkernel<<<grid, block>>>(d, width, height);
  checkCuda(cudaEventRecord(e_stop));
  checkCuda(cudaEventSynchronize(e_stop));
  float ms_kernel = 0.f; checkCuda(cudaEventElapsedTime(&ms_kernel, e_start, e_stop));

  // D2H (Device To Host)
  checkCuda(cudaEventRecord(e_start));
  for (int i = 0; i < reps; ++i)
    checkCuda(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaEventRecord(e_stop));
  checkCuda(cudaEventSynchronize(e_stop));
  float ms_d2h = 0.f; checkCuda(cudaEventElapsedTime(&ms_d2h, e_start, e_stop));

  printf("2D %dx%d,\nreps=%d,\nH2D avg=%.3f ms,\nkernel avg=%.3f ms,\nD2H avg=%.3f ms\n",
         width, height, reps, ms_h2d / reps, ms_kernel / reps, ms_d2h / reps);

  cudaEventDestroy(e_start);
  cudaEventDestroy(e_stop);
  cudaFree(d);
  free(h);
}

int main(int argc, char** argv) {
  int width = (argc >= 2) ? atoi(argv[1]) : 2048;
  int height = (argc >= 3) ? atoi(argv[2]) : width;
  int reps = (argc >= 4) ? atoi(argv[3]) : 500;
  bench_2d(width, height, reps);
  return 0;
}