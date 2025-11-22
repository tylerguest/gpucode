#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void naive_gemm(int M, int N, int K, const float* A, const float* B, float* C, float alpha, float beta) {
  // A is M x K, B is K x N, C is M x N (row-major)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) { sum += A[i*K + k] * B[k*N + j]; }
      C[i*N + j] = alpha * sum + beta * C[i*N + j];
    }
  }
}

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
  int M = 512, K = 512, N = 512;
  float alpha = 1.0f, beta = 0.0f;

  size_t Asz = (size_t)M * K;
  size_t Bsz = (size_t)K * N;
  size_t Csz = (size_t)M * N;

  float *A = (float*)malloc(Asz * sizeof(float));
  float *B = (float*)malloc(Bsz * sizeof(float));
  float *C = (float*)malloc(Csz * sizeof(float));
  if (!A || !B || !C) { perror("malloc"); return 1; }

  // initialize with simple values
  for (size_t i = 0; i < Asz; ++i) A[i] = (float)(i % 17) / 17.0f;
  for (size_t i = 0; i < Bsz; ++i) B[i] = (float)(i % 13) / 13.0f;
  for (size_t i = 0; i < Csz; ++i) C[i] = 0.0f;

  double t0 = now_sec();
  naive_gemm(M, N, K, A, B, C, alpha, beta);
  double t1 = now_sec();

  // quick checksum for basic correctness
  double sum = 0.0;
  for (size_t i = 0; i < Csz; ++i) sum += C[i];
  printf("M=%d K=%d N=%d time=%.6f s checksum=%.6f\n", M, K, N, t1 - t0, sum);

  free(A); free(B); free(C);
  return 0;
}

// TODO: setup nvbench and test cublass kernel against my own
//       start with this naive kernel