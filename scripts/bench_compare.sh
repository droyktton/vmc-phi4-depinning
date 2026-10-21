#!/usr/bin/env bash
# Short benchmark + validation run, the same on every machine: solves
# seeds SEEDS at sizes LS with the given binary (CUDA ./phi4vmc or SYCL
# ./phi4vmc_sycl), saving each run's output, critical wall and field
# (--dump-field) under OUTDIR/L<L>_seed<seed>/, and one line per run in
# OUTDIR/bench.csv (L, seed, h_d, wall time, sweeps, tile work). Compare
# two machines' OUTDIRs with scripts/compare_runs.py.
#
# Usage (from the repository root):
#   scripts/bench_compare.sh ./phi4vmc bench_cuda
#   scripts/bench_compare.sh ./phi4vmc_sycl bench_sycl
# On the Intel cluster, as a job (one PVC stack):
#   sbatch -p testing -A <account> --gres=gpu:intel_xt1550:1 -t 20 \
#       --wrap "module swap gnu13 intel/2025.3.0; module load dpl; scripts/bench_compare.sh ./phi4vmc_sycl bench_sycl"
set -euo pipefail

BINARY=$(realpath "$1")
OUTDIR=$2
LS=${LS:-"256 512"}
SEEDS=${SEEDS:-"1 2 3 4 5"}
H_LOW=${H_LOW:-0.0}
H_HIGH=${H_HIGH:-0.1}
TOL=${TOL:-1e-4}

mkdir -p "$OUTDIR"
OUTDIR=$(realpath "$OUTDIR")
echo "L,seed,h_d,seconds,sweeps,tile_work" > "$OUTDIR/bench.csv"
for L in $LS; do
	for seed in $SEEDS; do
		dir="$OUTDIR/L${L}_seed${seed}"
		mkdir -p "$dir"
		cd "$dir"
		start=$(date +%s.%N)
		"$BINARY" "$L" "$H_LOW" "$H_HIGH" "$TOL" "$seed" --dump-field > out.txt 2> err.txt
		end=$(date +%s.%N)
		cd - > /dev/null
		hd=$(sed -n 's/^depinning field h_d = //p' "$dir/out.txt")
		sweeps=$(sed -n 's/.* sweeps=\([0-9]*\) .*/\1/p' "$dir/out.txt" | paste -sd+ | bc)
		work=$(sed -n 's/.* tile_work=\([0-9.e+-]*\) sweeps/\1/p' "$dir/out.txt" | paste -sd+ | bc)
		seconds=$(echo "$end - $start" | bc)
		echo "$L,$seed,$hd,$seconds,$sweeps,$work" >> "$OUTDIR/bench.csv"
		echo "L=$L seed=$seed h_d=$hd ${seconds}s"
	done
done
