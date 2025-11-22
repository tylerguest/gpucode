#include <cstdio>
#include <cuda_runtime.h>

// 2D kernel
__global__ void twodimkernel(float* x, int width, int height) {
  // 2D thread coordinates in the global grid
  int col = blockIdx.x * blockDim.x + threadIdx.x;                              // x = column
  int row = blockIdx.y * blockDim.y + threadIdx.y;                              // y = row

  if (row < height && col < width) {                                            // bounds check
    int idx = row * width + col;                                                // 2D -> 1D index
    x[idx] *= 2.0f;                                                             // double the value at i 
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
  const int width = 4;                                                           // number of columns
  const int height = 4;                                                          // number of rows
  const int n = width * height;                                                  // total number of elements (4*4 = 16)

  float h_x[n];                                                                  // host array of n floats
  for (int i = 0; i < n; ++i) h_x[i] = float(i);                                 // initialize host array
  
  float* d_x = nullptr;                                                          // device gpu pointer
  cudaMalloc(&d_x, n * sizeof(float));                                           // allocate n floats on the GPU
  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);               // copy data from host array h_x -> device array d_x

  // 2D block and grid
  dim3 block(2, 2);                                                              // 2x2 = 4 threads per block
  dim3 grid(
    (width + block.x - 1) / block.x,                                             // grid.x
    (height + block.y - 1) / block.y                                             // grid.y
  );

  twodimkernel<<<grid, block>>>(d_x, width, height);                             // launch 2D kernel                             
  debug_kernel<<<grid, block>>>(width, height);                                  // debug 2D kernel
  cudaDeviceSynchronize();

  cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);               // copy results from device array d_x -> host array h_x
  cudaFree(d_x);                                                                 // free the device mem

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int idx = row * width + col;
      printf("%2d:%6.1f ", idx, h_x[idx]);
    }
    printf("\n");
  }

  return 0;
}