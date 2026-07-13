# GPU Code

A small CUDA learning repo focused on kernel concepts first.

The current goal is simple: write one standalone `.cu` file per concept, compile it with `nvcc`, run it, and understand what the kernel is doing.

No Python bindings, no PyTorch extensions, no Triton, no CMake, and no benchmark framework yet. Those can come later after the CUDA basics feel natural.

## Layout

```text
gpucode/
  Makefile
  kernels/
    vector_add.cu
  notes/
    vector_add.md
```

## Build And Run

Build the first kernel:

```bash
make vector_add
```

Run it:

```bash
./bin/vector_add
```

Clean build outputs:

```bash
make clean
```

## Learning Order

Start with one concept at a time:

1. `vector_add.cu`: indexing, launch config, memory allocation, copies.
2. `saxpy.cu`: memory bandwidth style vector operation.
3. `reduce_sum.cu`: shared memory and synchronization.
4. `transpose.cu`: coalescing and shared-memory tiling.
5. `matmul.cu`: 2D indexing and tiled compute.

## Current Kernel

`kernels/vector_add.cu` teaches:

- `threadIdx.x`
- `blockIdx.x`
- `blockDim.x`
- global thread indexing
- bounds checks
- `cudaMalloc`
- `cudaMemcpy`
- kernel launch syntax
- `cudaFree`

The point is not performance yet. The point is understanding the CUDA execution model.
