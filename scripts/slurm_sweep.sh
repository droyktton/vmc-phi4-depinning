#!/usr/bin/env bash
# SLURM job array version of sweep_disorder.sh: one array task (one GPU)
# per (L, Delta) pair, each running run_batch.py --dump-field into
# $OUTROOT/L<L>_D<Delta>/. Like sweep_disorder.sh, seeds already listed
# in hd_summary.csv are skipped, so resubmitting resumes a cut run.
#
# Binaries phi4vmc_D<Delta> must be built beforehand on the frontend
# (no GPU there, so the Makefile builds for all the gpu partition's archs):
#   for D in $DELTAS; do make -B AMPDIS=$D TARGET=phi4vmc_D$D; done
#
# Usage (from the repository root; number of tasks = #DELTAS * #LS):
#   DELTAS="0.4 0.5 0.6 0.7 0.85 1.0" LS="1024 2048" NSAMPLES=30 \
#       sbatch --array=0-11 scripts/slurm_sweep.sh
#SBATCH --job-name=phi4vmc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

DELTAS=${DELTAS:-"0.4 0.5 0.6 0.7 0.85 1.0"}
LS=${LS:-"1024 2048"}
NSAMPLES=${NSAMPLES:-30}
TOL=${TOL:-1e-4}
OUTROOT=${OUTROOT:-data/sweep}

read -ra DARR <<< "$DELTAS"
read -ra LARR <<< "$LS"
i=${SLURM_ARRAY_TASK_ID:?submit with sbatch --array}
D=${DARR[$((i % ${#DARR[@]}))]}
L=${LARR[$((i / ${#DARR[@]}))]}

# Same bracket as sweep_disorder.sh: [0, 3*h_est], h_est = 0.008*(Delta/0.2)^(4/3)
HHIGH=$(python3 -c "print(3*0.008*($D/0.2)**(4/3))")
echo "=== $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-?} Delta=$D L=$L h in [0, $HHIGH] ==="
python3 -u scripts/run_batch.py --binary "./phi4vmc_D$D" --L "$L" \
    --h-low 0 --h-high "$HHIGH" --tol "$TOL" --nsamples "$NSAMPLES" \
    --dump-field --outdir "$OUTROOT/L${L}_D${D}"
