#include <cuda_runtime.h>

#define TILE_SIZE 16

// C = A (M x K) * B (K x N)
// A is row major: A[row * K + col]
// B is row major: B[row * N + col]
// C is row mahor: C[row * N + col]
__global__ void matMulTiledKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
  // 2D indices within the block
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 2D indicies within the output matrix C
  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  // shared mem tiles
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  float acc = 0.0f;

  // number of tiles we need along the K dimension
  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int  t = 0; t < numTiles; ++t) {
    // global column for A and global row for B that this tile covers
    int A_col = t * TILE_SIZE + tx;
    int B_row = t * TILE_SIZE + ty;

    // load A tile element if in bounds, else 0
    if (row < M && A_col < K) As[ty][tx] = A[row * K + A_col];
    else As[ty][tx] = 0.0f;

    // load B tile element if in bounds, else 0
    if (B_row < K && col < N) Bs[ty][tx] = B[B_row * N + col];
    else Bs[ty][tx] = 0.0f;
    
    // wait for all threads to finish loading this tile
    __syncthreads();

    // multiply the two tiles (small inner matmul)
    #pragma unroll
    for (int k_inner = 0; k_inner < TILE_SIZE; ++k_inner) acc += As[ty][k_inner] * Bs[k_inner][tx];

    // wait before loading next tile
    __syncthreads();
  }

  // write result if in bounds
  if (row < M && col < N) C[row * N + col] = acc;
}

// simple launcher function: A (M x K), B (K x N), C (M x N)
void matMulTiled(const float* d_A, const float* d_B, float* d_C, int M, int K, int N) {
  dim3 blockDim(TILE_SIZE, TILE_SIZE); 
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
               (M + TILE_SIZE - 1) / TILE_SIZE);
  
  matMulTiledKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
}