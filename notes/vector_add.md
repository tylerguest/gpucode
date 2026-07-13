# Vector Add

`vector_add.cu` is the first CUDA kernel in this repo.

The kernel computes:

```text
c[i] = a[i] + b[i]
```

## Key CUDA Idea

Each CUDA thread handles one element of the array.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

That line converts CUDA's block/thread coordinates into a single global array index.

## Launch Configuration

The program uses:

```cpp
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
```

The `blocks` formula rounds up so every element has a thread assigned to it.

## Bounds Check

The kernel uses:

```cpp
if (i < n) {
  c[i] = a[i] + b[i];
}
```

This matters because the number of launched threads is usually rounded up to a multiple of the block size. Some extra threads may exist past the end of the array.

## Memory Flow

The program does this:

```text
allocate CPU vectors
allocate GPU memory with cudaMalloc
copy CPU input to GPU with cudaMemcpy
launch CUDA kernel
copy GPU output back to CPU with cudaMemcpy
check result on CPU
free GPU memory
```

## What To Understand Before Moving On

- What `threadIdx.x` means.
- What `blockIdx.x` means.
- What `blockDim.x` means.
- Why the global index formula works.
- Why the bounds check is necessary.
- What memory lives on the CPU versus GPU.
- Why `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost` go in different directions.
