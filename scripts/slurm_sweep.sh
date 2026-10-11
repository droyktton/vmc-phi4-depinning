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
#
# To split one (L, Delta) over several GPUs, submit one job per seed range
# with SEED_START and NSAMPLES (seeds SEED_START..SEED_START+NSAMPLES-1);
# they append to the same hd_summary.csv.
#
# Memory: the solver keeps ~4 copies of the L x L field on the GPU
# (~4.5 GB at L=16384, plus ~3 GB transiently when saving), so for
# L >= 8192 keep away from GPUs shared with other processes (e.g.
# --exclude=<node>).
#SBATCH --job-name=phi4vmc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

DELTAS=${DELTAS:-"0.4 0.5 0.6 0.7 0.85 1.0"}
LS=${LS:-"1024 2048"}
NSAMPLES=${NSAMPLES:-30}
SEED_START=${SEED_START:-1}
LYK=${LYK:-}   # if set, Ly = LYK*L^1.25 (else a square L x L box)
PATHLEN=${PATHLEN:-}    # if set (with HEADROOM): whole-row escape after PATHLEN rows,
HEADROOM=${HEADROOM:-}  # Ly = 32+PATHLEN+HEADROOM (see ./phi4vmc --help)
ADAPT=${ADAPT:-}        # if set (with PATHLEN): double a sample's headroom while its wall nears Ly
TOL=${TOL:-1e-4}
OUTROOT=${OUTROOT:-data/sweep}

read -ra DARR <<< "$DELTAS"
read -ra LARR <<< "$LS"
i=${SLURM_ARRAY_TASK_ID:?submit with sbatch --array}
D=${DARR[$((i % ${#DARR[@]}))]}
L=${LARR[$((i / ${#DARR[@]}))]}

# Bracket [0.85, 1.15]*h_est around the measured h_d ~ 0.0195*(Delta/0.4)^1.34
# (L=1024-2048 sweep; sample-to-sample spread ~2%). The solver widens it
# by itself if it doesn't contain a sample's h_d.
HEST=$(python3 -c "print(0.0195*($D/0.4)**1.34)")
HLOW=${HLOW:-$(python3 -c "print(0.85*$HEST)")}
HHIGH=${HHIGH:-$(python3 -c "print(1.15*$HEST)")}
echo "=== $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-?} Delta=$D L=$L LYK=${LYK:-square} PATHLEN=${PATHLEN:--} HEADROOM=${HEADROOM:--} h in [$HLOW, $HHIGH] ==="
python3 -u scripts/run_batch.py --binary "./phi4vmc_D$D" --L "$L" \
    --h-low "$HLOW" --h-high "$HHIGH" --tol "$TOL" --nsamples "$NSAMPLES" --seed-start "$SEED_START" \
    ${LYK:+--ly-k "$LYK"} ${PATHLEN:+--path "$PATHLEN" --headroom "$HEADROOM"} ${ADAPT:+--adapt-headroom} --dump-field --outdir "$OUTROOT/L${L}_D${D}"
