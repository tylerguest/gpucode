# GPU Kernel Lab

A CUDA and Triton kernel lab for implementing, testing, benchmarking, and profiling GPU kernels from first principles.

This project is designed as a portfolio-quality learning repo based on core ideas from *Programming Massively Parallel Processors*: thread hierarchy, memory coalescing, shared memory, reductions, scans, atomics, tiling, occupancy, and profiling.

The repo starts with foundational CUDA kernels and builds toward ML-relevant kernels such as matrix multiplication, softmax, LayerNorm, RMSNorm, and attention score computation.

## Goals

- Implement core GPU programming patterns from scratch.
- Build correctness tests against NumPy or PyTorch references.
- Benchmark custom CUDA and Triton kernels against PyTorch where appropriate.
- Use Nsight tools to study memory access, occupancy, and bottlenecks.
- Document the performance tradeoffs behind each optimization.
- Build a practical GPU programming portfolio project.

## Kernel Status

| Kernel | CUDA | Triton | Tests | Benchmarks | Main Concept |
|---|---:|---:|---:|---:|---|
| Vector Add | planned | no | planned | planned | thread indexing |
| SAXPY | planned | no | planned | planned | memory bandwidth |
| Reduce Sum | planned | no | planned | planned | parallel reduction |
| Histogram | planned | no | planned | planned | atomics |
| Prefix Scan | planned | no | planned | planned | parallel scan |
| Transpose | planned | no | planned | planned | coalescing/shared memory |
| Stencil 1D/2D | planned | no | planned | planned | neighbor access |
| Matmul | planned | planned | planned | planned | tiling |
| Convolution 2D | planned | no | planned | planned | stencil/tiling |
| Softmax | planned | planned | planned | planned | row-wise reduction |
| LayerNorm | planned | planned | planned | planned | normalization |
| RMSNorm | planned | planned | planned | planned | transformer norm |
| Attention Scores | planned | planned | planned | planned | tiled ML workload |
| Fused MLP | no | planned | planned | planned | kernel fusion |

## Project Layout

```text
gpu-kernel-lab/
  cuda/          CUDA kernels, C++ wrappers, Python bindings, and CUDA tests
  triton/        Triton kernels and tests
  benchmarks/    CUDA, Triton, and PyTorch benchmark scripts
  docs/          Notes connecting PMPP concepts to this repo's kernels
  scripts/       Build, test, benchmark, profile, and cleanup helpers
```

## Build

The build system will use CMake with CUDA enabled. Python tests and benchmarks will call kernels through bindings once implemented.

```bash
./scripts/build.sh
```

## Test

```bash
./scripts/run_tests.sh
```

## Benchmark

```bash
./scripts/run_benchmarks.sh
```

## Profiling

Nsight Compute profiling notes and commands will live in `docs/profiling_walkthrough.md` and `scripts/profile_nsight.sh`.

## Development Approach

Each kernel should be completed with the same standard:

- CUDA or Triton implementation.
- Correctness test against a trusted reference.
- Benchmark entry.
- Documentation of the main GPU concept.
- Known limitations listed honestly.

The goal is not to beat cuBLAS or PyTorch immediately. The goal is to understand why optimized libraries are fast and to build the skills needed to reason about GPU performance.
