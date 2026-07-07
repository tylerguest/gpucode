# PMPP Mapping

| PMPP Concept | Repo Kernel |
|---|---|
| Thread hierarchy | vector_add, saxpy |
| Memory coalescing | transpose_naive, transpose_tiled |
| Shared memory | reduce_sum, transpose_tiled, matmul_tiled |
| Synchronization | reduce_sum, scan, matmul_tiled |
| Atomics | histogram |
| Prefix sum | scan |
| Stencils | stencil_1d, stencil_2d, convolution_2d |
| Tiling | transpose_tiled, matmul_tiled, convolution_2d |
| Occupancy | profiling notes |
| ML workloads | softmax, layernorm, rmsnorm, attention_scores |
