#!/usr/bin/env bash
# The short benchmark/validation run (scripts/bench_compare.sh) of the SYCL
# build on one Intel GPU of the cluster (one stack of a Data Center GPU
# Max 1550). Builds the binary on the node first: ahead of time for PVC if
# the node has ocloc (the login node doesn't), else JIT.
#
# Usage (from the repository root):
#   sbatch -A <account> scripts/slurm_bench_intel.sh
# then compare with the laptop's CUDA run:
#   scripts/compare_runs.py bench_cuda bench_sycl --names CUDA SYCL
#SBATCH --job-name=phi4vmc_bench
#SBATCH --partition=testing
#SBATCH --gres=gpu:intel_xt1550:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

module swap gnu13 intel/2025.3.0
module load dpl

hostname
sycl-ls
# one stack: the GPUs are listed one per stack (FLAT hierarchy)
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}

if command -v ocloc > /dev/null; then
	make -B sycl
else
	make -B sycl SYCL_AOT=
fi

LS=${LS:-"256 512 1024"} SEEDS=${SEEDS:-"1 2 3"} scripts/bench_compare.sh ./phi4vmc_sycl "${OUTDIR:-bench_sycl}"
