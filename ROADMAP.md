# Roadmap

This roadmap works backward from the final target repo. Each milestone should leave the project in a working state with tests and documentation, not just new files.

## Milestone 1: Project Foundation

- Configure CMake with CUDA support.
- Add Python dependency setup.
- Add build, test, benchmark, profile, and cleanup scripts.
- Add CUDA error-checking helpers.
- Add initial Python binding structure.
- Add `docs/pmpp_mapping.md`.

## Milestone 2: Basic Parallel Kernels

- Implement `vector_add.cu`.
- Implement `saxpy.cu`.
- Add correctness tests for vector add and SAXPY.
- Add memory-bandwidth benchmarks.
- Document basic thread indexing and memory bandwidth.

## Milestone 3: Reductions, Synchronization, and Atomics

- Implement `reduce_sum.cu`.
- Implement `histogram.cu`.
- Implement `scan.cu`.
- Add tests for reductions, histograms, and scans.
- Document shared memory, synchronization, atomics, and prefix scan.

## Milestone 4: Memory Access and Tiling

- Implement `transpose_naive.cu`.
- Implement `transpose_tiled.cu`.
- Implement `stencil_1d.cu`.
- Implement `stencil_2d.cu`.
- Add tests and benchmarks for transpose and stencil kernels.
- Document coalescing, shared-memory tiling, halos, and bank conflicts if observed.

## Milestone 5: Dense Compute

- Implement `matmul_naive.cu`.
- Implement `matmul_tiled.cu`.
- Implement `convolution_2d.cu`.
- Add tests against PyTorch references.
- Add benchmark comparisons against PyTorch.
- Document tiling, arithmetic intensity, and memory reuse.

## Milestone 6: ML-Relevant CUDA Kernels

- Implement `softmax.cu` with numerical stability.
- Implement `layernorm.cu`.
- Implement `rmsnorm.cu`.
- Implement `attention_scores.cu` as a tiled `Q @ K^T / sqrt(d)` kernel.
- Add tests against PyTorch references.
- Add benchmark comparisons and profiling notes.

## Milestone 7: Triton Kernels

- Implement Triton matmul.
- Implement Triton softmax.
- Implement Triton LayerNorm.
- Implement Triton RMSNorm.
- Implement a simple fused MLP or fused bias plus activation kernel.
- Compare CUDA, Triton, and PyTorch where appropriate.

## Milestone 8: Profiling and Final Report

- Add Nsight Compute walkthrough.
- Add real benchmark tables under `benchmarks/results/`.
- Add roofline notes for selected kernels.
- Compare naive and optimized transpose.
- Compare naive and tiled matmul.
- Summarize lessons learned and remaining limitations.

## Final Done Definition

- All implemented CUDA tests pass.
- All implemented Triton tests pass on supported hardware.
- Benchmarks run from one script.
- The README status table reflects reality.
- Docs explain this repo's actual kernels, not generic CUDA examples.
- Benchmark results include real hardware information.
- At least one Nsight profiling walkthrough exists.
- Limitations are documented honestly.
