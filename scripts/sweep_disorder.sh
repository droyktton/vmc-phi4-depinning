#!/usr/bin/env bash
# Sweeps the disorder strength Delta (a compile-time constant, AMPDIS):
# for each Delta builds its own binary phi4vmc_D<Delta>, then for each L
# runs run_batch.py with --dump-field into $OUTROOT/L<L>_D<Delta>/ (the raw
# critical fields are what local_roughness.py analyses).
#
# The bisection bracket is [0, 3*h_est] with h_est = 0.008*(Delta/0.2)^(4/3)
# (measured h_d ~ 0.008 at Delta=0.2, L=256; paper: h_d ~ Delta^(4/3)).
# run_batch.py skips seeds already in hd_summary.csv, so re-running this
# script with a larger NSAMPLES just extends the existing samples.
#
# ARCH (optional) forces the GPU architecture; by default the Makefile
# picks it (see there).
#
# Usage (run from the repository root; all variables optional):
#   DELTAS="0.4 0.6 1.0" LS="256 512" NSAMPLES=30 OUTROOT=data/sweep \
#       scripts/sweep_disorder.sh
set -euo pipefail

DELTAS=${DELTAS:-"0.4 0.5 0.6 0.7 0.85 1.0"}
LS=${LS:-"256 512"}
NSAMPLES=${NSAMPLES:-30}
TOL=${TOL:-1e-4}
OUTROOT=${OUTROOT:-data/sweep}

for D in $DELTAS; do
    make -B AMPDIS="$D" TARGET="phi4vmc_D$D" ${ARCH:+ARCH="$ARCH"} > /dev/null
    HHIGH=$(python3 -c "print(3*0.008*($D/0.2)**(4/3))")
    for L in $LS; do
        echo "=== Delta=$D L=$L h in [0, $HHIGH] ==="
        python3 scripts/run_batch.py --binary "./phi4vmc_D$D" --L "$L" \
            --h-low 0 --h-high "$HHIGH" --tol "$TOL" --nsamples "$NSAMPLES" \
            --dump-field --outdir "$OUTROOT/L${L}_D${D}"
    done
done
