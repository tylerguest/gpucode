#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

void matMulTiled(const float* d_A, const float* d_B, float* d_C, int M, int K, int N);

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

static void cpuMatMul(const float* A, const float* B, float* C, int M, int K, int N) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += A[i*K + k] * B[k*N + j];
      C[i*N + j] = s;
    }
}

int main(int argc, char** argv) {
  int M = 256, K = 256, N = 256;          
  if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }

  size_t Asz = size_t(M)*K*sizeof(float), Bsz = size_t(K)*N*sizeof(float), Csz = size_t(M)*N*sizeof(float);

  float* hA = new float[size_t(M) * K];
  float* hB = new float[size_t(K) * N];
  float* hC = new float[size_t(M) * N];
  float* hC_ref = new float[size_t(M) * N];
  for (size_t i = 0; i < size_t(M) * K; ++i) hA[i] = float((int)(i % 13) - 6);
  for (size_t i = 0; i < size_t(K) * N; ++i) hB[i] = float((int)(i % 7) - 3);

  float *dA=nullptr,*dB=nullptr,*dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, Asz));
  CUDA_CHECK(cudaMalloc(&dB, Bsz));
  CUDA_CHECK(cudaMalloc(&dC, Csz));
  CUDA_CHECK(cudaMemcpy(dA, hA, Asz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, Bsz, cudaMemcpyHostToDevice));

  // warmup
  matMulTiled(dA,dB,dC,M,K,N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // timed single-shot kernel
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  matMulTiled(dA,dB,dC,M,K,N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // copy back and verify
  CUDA_CHECK(cudaMemcpy(hC, dC, Csz, cudaMemcpyDeviceToHost));
  cpuMatMul(hA, hB, hC_ref, M, K, N);

  const float EPS = 1e-3f;
  bool ok = true;
  for (size_t i = 0; i < size_t(M)*N; ++i) {
    if (fabs(hC[i] - hC_ref[i]) > EPS * fmax(1.0f, fabs(hC_ref[i]))) { ok = false; break; }
  }

  double sec = ms / 1000.0;
  double flops = 2.0 * double(M) * double(N) * double(K);
  double gflops = flops / (sec * 1e9);

  printf("Sizes: M=%d K=%d N=%d\n", M, K, N);
  printf("Time: %.3f ms, %.3f GFLOPS\n", ms, gflops);
  printf("Result: %s\n", ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  
  delete[] hA;
  delete[] hB;
  delete[] hC;
  delete[] hC_ref;
  
  return ok ? 0 : 1;
}