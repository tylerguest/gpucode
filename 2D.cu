#include <cstdio>
#include <cuda_runtime.h>

// 2D kernel
__global__ void twodimkernel(float* x, int width, int height) {
  // 2D thread coordinates in the global grid
  int col = blockIdx.x * blockDim.x + threadIdx.x;                              // x = column
  int row = blockIdx.y * blockDim.y + threadIdx.y;                              // y = row

  // bounds check
  if (row < height && col < width) {
    int idx = row * width + col;                                                // 2D -> 1D index
    x[idx] *= 2.0f;
  }
}

__global__ void debug_kernel(int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < height && col < width) {
    int idx = row * width + col;
    printf("block=(%d,%d) thread=(%d,%d) -> (row,col)=(%d,%d) idx=%d\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y,
           row, col, idx);
  }
}

int main() {
  const int width = 4;
  const int height = 4;
  const int n = width * height;

  float h_x[n];
  for (int i = 0; i < n; ++i) h_x[i] = float(i);
  
  float* d_x = nullptr;
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

  // 2D block and grid
  dim3 block(2, 2);                                                              // 2x2 = 4 threads per block
  dim3 grid(
    (width + block.x - 1) / block.x,                                             // grid.x
    (height + block.y - 1) / block.y                                             // grid.y
  );

  twodimkernel<<<grid, block>>>(d_x, width, height);
  debug_kernel<<<grid, block>>>(width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);

  // print as a 2D matrix to see rows/cols
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int idx = row * width + col;
      printf("%2d:%6.1f ", idx, h_x[idx]);
    }
    printf("\n");
  }

  return 0;
}