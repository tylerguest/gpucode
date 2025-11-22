#include <cstdio>

#define MAX_FILTER_RADIUS 16
__constant__ float d_filter[2 * MAX_FILTER_RADIUS + 1];

__global__ void conv1d_const_filter(const float* __restrict__ x, const float* __restrict__ f, float* __restrict__ y, int n, int radius) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float acc = 0.0f;

  // sum over neighbors in [-r, r]
  for (int j = -radius; j <= radius; ++j) {
    int x_idx = i + j;  // neighbor index in x
    float x_val = 0.0f;

    // zero-padding for ghost cells
    if (x_idx >= 0 && x_idx < n) x_val = x[x_idx];

    int f_idx = j + radius;  // shift [-r, r] -> [0, 2r]
    acc += d_filter[f_idx] * x_val;
  }

  y[i] = acc;
}

// simple host driver to test the kernel
int main() {
  const int n = 7; 
  const int radius = 2;

  float h_x[n] = {8, 2, 5, 4, 1, 7, 3};
  float h_f[2*radius + 1] = {1, 3, 5, 3, 1};
  float h_y[n] = {0};

  // device pointers
  float *d_x, *d_f, *d_y;
  size_t bytes_x = n * sizeof(float);
  size_t bytes_f = (2*radius + 1) * sizeof(float);

  cudaMalloc(&d_x, bytes_x);
  cudaMalloc(&d_f, bytes_f);
  cudaMalloc(&d_y, bytes_x);

  cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, h_f, bytes_f, cudaMemcpyHostToDevice);

  // launch configuration
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  
  cudaMemcpyToSymbol(d_filter, h_f, bytes_f);

  conv1d_const_filter<<<grid_size, block_size>>>(d_x, d_f, d_y, n, radius);
  cudaDeviceSynchronize();

  cudaMemcpy(h_y, d_y, bytes_x, cudaMemcpyDeviceToHost);

  printf("Output y:\n");
  for (int i = 0; i < n; ++i) printf("y[%d] = %f\n", i, h_y[i]);

  cudaFree(d_x);
  cudaFree(d_f);
  cudaFree(d_y);
  return 0;
}