#!/usr/bin/env bash
set -euo pipefail

python benchmarks/benchmark_cuda.py
python benchmarks/benchmark_triton.py
python benchmarks/compare_pytorch.py
