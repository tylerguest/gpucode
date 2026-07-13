__global__ void reduce_sum_kernel(const float* x, float* partial_sums, int n) {
  /* creates shared memory for each block*/
  extern __shared__ float shared[];

  /* the thread index inside its block */
  int tid = threadIdx.x;
  /* maps the block/thread pair to a global input index */
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /* threads outside the valid input range contribute 0 */
  float value = 0.0f;
  if (i < n) { value = x[i]; }

  /* each thread writes its value into shared memory, then all threads wait until block finishes writing */
  shared[tid] = value;
  __syncthreads();

  /* half the number of active threads*/
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    /* lower-index threads add values from the upper half */
    if (tid < stride) { shared[tid] += shared[tid + stride]; }
    __syncthreads();
  }
  
  /* shared[0] containa that block's sum, thread 0 writes it to global memory */
  if (tid == 0) { partial_sums[blockIdx.x] = shared[0]; }
}

void launch_reduce_sum(const float* x, float* partial_sums, int n, int blocks, int threads_per_block) {
  size_t shared_bytes = threads_per_block * sizeof(float);
  reduce_sum_kernel<<<blocks, threads_per_block, shared_bytes>>>(x, partial_sums, n);
}