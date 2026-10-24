#!/usr/bin/env bash
# SLURM job array over a job list from scripts/make_vdyn_jobs.py: each
# line is "<outdir> <binary and arguments>", one main_vdyn.cu run. Array
# task i runs lines i*PER_TASK .. i*PER_TASK+PER_TASK-1 one after another
# on its GPU, each in its outdir (log in log_seed<s>.txt). A run that
# finished leaves a .done_seed<s> marker and is skipped on resubmission,
# so resubmitting the same list resumes a cut job.
#
# The binary must be built beforehand on the frontend (no GPU there, so
# the Makefile builds for all the gpu partition's archs):
#   make vdyn AMPDIS=0.3 VDYN_TARGET=phi4vmc_vdyn_D0.3
#
# Usage (from the repository root):
#   python3 scripts/make_vdyn_jobs.py ff > jobs_ff.txt
#   N=$(wc -l < jobs_ff.txt); PER_TASK=4
#   JOBS=jobs_ff.txt PER_TASK=$PER_TASK sbatch --array=0-$(( (N+PER_TASK-1)/PER_TASK - 1 )) scripts/slurm_vdyn.sh
#SBATCH --job-name=phi4vdyn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

JOBS=${JOBS:?set JOBS to a job list from make_vdyn_jobs.py}
PER_TASK=${PER_TASK:-1}
i=${SLURM_ARRAY_TASK_ID:?submit with sbatch --array}
ROOT=$(pwd)

first=$(( i * PER_TASK + 1 ))
last=$(( first + PER_TASK - 1 ))
sed -n "${first},${last}p" "$JOBS" | while read -r outdir cmd; do
    seed=$(awk '{print $9}' <<< "$cmd")   # binary L Ly H0 h_u v dt tmax seed ...
    mkdir -p "$outdir"
    if [[ -e "$outdir/.done_seed$seed" ]]; then
        echo "skip $outdir seed $seed (done)"
        continue
    fi
    bin=${cmd%% *}; rest=${cmd#* }
    [[ $bin = /* ]] || bin="$ROOT/$bin"
    echo "=== $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-?} $outdir seed $seed: $(date)"
    if (cd "$outdir" && "$bin" $rest > "log_seed$seed.txt" 2>&1 < /dev/null); then
        touch "$outdir/.done_seed$seed"
        echo "    done $(date)"
    else
        echo "    FAILED (see $outdir/log_seed$seed.txt)"
    fi
done
