# Does the crossover length set the threshold exponent? 4×1 and 16×1 blocks

## Context
§5.9 of `docs/analysis_math.md`, weak 8×1 blocks at Δ = 0.2:
- **The geometry is qKPZ.** ζ_W = 0.62–0.67 for l ≈ 300–5000, S(q) ~ q^−2.27, and λ > 0 independent of L.
- **The threshold is not.** With 200 seeds per L (256–8192), shift and spread of h_d give 1/ν = 0.685 ± 0.02,
  6σ from qKPZ (0.577) and 3.5σ from qEW (0.75). The box, the extreme-value bias and the KPZ slope term do not
  explain this.

**Hypothesis.** 0.685 is pre-asymptotic. ζ_W crosses 1.125 at ℓ× ≈ 35, but the plateau only starts at
ℓ_p ≈ 300, so at L = 8192 we have L/ℓ_p ≈ 16–27. A global quantity like σ(h_d) would need a larger ratio.
Against it: the local slopes of σ between successive L (0.85, 0.63, 0.54, 0.74, 0.78, ±0.10–0.14) show no
downward drift.

**Test.** Going to larger L is expensive, so we change ℓ_p instead. If 1/ν_eff depends only on L/ℓ_p, then
different block shapes must give the same function of L/ℓ_p.

**How the block shape moves ℓ_p.** The KPZ term matches the elastic one when w(l) ~ c/λ. With w ~ l^{3/2}
in the Larkin regime, that gives ℓ× ~ (c/λ)^{2/3}, times a disorder-dependent factor.
- Δ is a weak knob: ℓ× ≈ 57 → 38 for Δ = 0.1 → 0.2 in the 8×1 pilot (`results/aniso_weak/pilot_ax8/lo_vs_delta.png`),
  roughly ℓ× ~ Δ^−0.6.
- The block aspect ratio is a strong knob: λ/h_d ≈ −3 to −4 for 4×1 and −8 to −9 for 8×1. At L = 1024 and
  Δ = 0.2, ζ_W crosses 1 at l ≈ 150 for 4×1 (no plateau) and at l ≈ 70 for 8×1 (`results/aniso_weak/pilot_ax*/zeta_local_L1024.png`).

So we compare three shapes at fixed Δ = 0.2: 4×1 (ℓ_p larger), 8×1 (already done) and 16×1 (ℓ_p
expected about half of 8×1).

**Predictions under the hypothesis**, at the same L:

| blocks | ℓ_p | 1/ν_eff at L ≤ 2048 |
|---|---|---|
| 4×1 | larger than 8×1 | closer to 0.75 (or to the isotropic 0.88) |
| 8×1 | ≈ 300 | ≈ 0.69 (measured) |
| 16×1 | ≈ 150? | closer to 0.58 |

## Stage 0 — Build (frontend; compiling only)
`slurm_sweep.sh` runs `${BIN}<Delta>`, so the block shape goes into the prefix:

```bash
for A in 4 16; do make -B AMPDIS=0.2 AX=$A AY=1 TARGET=phi4vmc_ax${A}ay1_D0.2; done
```

**Regression check.** One seed of 8×1 at L = 1024 with the rebuilt binary gives an h_d within the
spread of §5.9 (⟨h_d⟩ = 0.01866, σ/⟨h_d⟩ = 2.6 %).

**Change to `scripts/slurm_sweep.sh`.** Add `DUMP=${DUMP:-1}` and pass `--dump-field` only if it is set.
This way seeds 31–200 can run without field dumps, as in `data/aniso_weak/seeds/`.

## Stage 1 — Short pilot (L = 1024, s = 0, plus a short tilt)
- **16×1 at Δ = 0.2, 10 seeds, `--dump-field`.** Use a generous box (P ≈ 20 × the 8×1 width, w ≈ 10 → P = 200)
  with `ADAPT=1`. The solver widens the h bracket if needed.
  ```bash
  DELTAS=0.2 LS=1024 NSAMPLES=10 PATHLEN=200 HEADROOM=200 ADAPT=1 TOL=1e-5 \
      BIN=./phi4vmc_ax16ay1_D HLOW=0.014 HHIGH=0.030 OUTROOT=data/aniso_nu/pilot_ax16 \
      sbatch --array=0 scripts/slurm_sweep.sh
  ```
- **Tilt for 16×1:** L = 512, s ∈ {0, ±0.05, ±0.1}, 10 seeds (`TILT=m` with m = s·L), giving h_d''/h_d near s = 0.
- **What to look at:**
  - overhangs (expected none);
  - ζ_W(l) and ℓ× with `scripts/local_roughness.py`;
  - whether λ/h_d roughly doubles again (≈ −25 or more near s = 0).
- **Go/no-go.** Production goes ahead if ℓ× and the plateau onset for 16×1 are clearly below 8×1's. If they
  don't move, 32×1 is no better and the test changes to the elastic constant (`make CEL=...`), which lowers
  c/λ directly.

## Stage 2 — Calibrate the box (κ = 20) for 4×1 and 16×1
- **Measure w(L)** at L = 256, 512, 1024 and 2048, 10 seeds, with the definition of `data/kappa/analyze.py`
  (as in `data/aniso_weak/calib*`). For reference, 8×1 gave w = 3.7, 5.7, 10.0, 13.5.
- **Set P(L) = 20·w(L).** Fit w(L) locally; don't assume an exponent.
- **Check.** ⟨h_d⟩(L) should fall like the ordinary finite-size shift. With 8×1 it fell at every κ with the
  same h_∞; use the same criterion here.

## Stage 3 — Production (s = 0)
- **Sizes and seeds:** L = 256, 512, 1024 and 2048 for 4×1 and 16×1, with 200 seeds per L and `--tol 1e-5`.
  The bisection step must stay below σ/5: at L = 2048, σ/⟨h_d⟩ ≈ 1.8 % for 8×1.
- **Field dumps:** seeds 1–30 with `--dump-field`, for ℓ_p and κ_eff; seeds 31–200 with `DUMP=`.
- **Jobs:** one array task per L (P depends on L). Split the seeds over GPUs with SEED_START/NSAMPLES. For example,
  4×1 at L = 2048:
  ```bash
  DELTAS=0.2 LS=2048 PATHLEN=<20 w(2048)> HEADROOM=<P> ADAPT=1 TOL=1e-5 BIN=./phi4vmc_ax4ay1_D \
      OUTROOT=data/aniso_nu/ax4 SEED_START=1 NSAMPLES=30 sbatch --array=0 scripts/slurm_sweep.sh
  DUMP= SEED_START=31 NSAMPLES=170 ... sbatch --array=0 scripts/slurm_sweep.sh
  ```
- **Statistics:** the error on σ with N = 200 is ≈ 5 %. A free slope over L = 256–2048 then has an error of
  ≈ 0.03, enough to separate 0.58 from 0.69 from 0.75. If the slopes between successive L are needed at
  ±0.05, use 800 seeds at L = 256 and 512 (they are cheap).
- **Control:** isotropic disorder (1×1) at the same Δ = 0.2, L and seeds, with P = 0.8 × the κ = 25 boxes of
  `data/kappa`. The existing isotropic Δ = 0.2 runs have only 30 seeds and `--tol 1e-4`.
- **Optional:** 16×1 at L = 4096 (30 + 170 seeds), if stage 4 points to a trend.

## Stage 4 — Analysis (cpu queue, never on the frontend)
1. **ℓ× and ℓ_p per shape.** Run `scripts/local_roughness.py` on the dumped L = 2048 fields: ℓ× is where
   ζ_W first drops below 1.125, and ℓ_p is where it drops below 0.75 (`--zeta-th`). For 8×1, reuse the
   §5.9 fields.
2. **σ(h_d) and the shift per shape.** Generalize the glob in `scripts/hd_variance.py` (and
   `data/aniso_weak/kappa_sens/joint_fit.py`) from `ax8ay1` to `ax*ay1`. For each shape, report the free
   1/ν and the χ² at 0.577 and 0.75, over the same L range (256–2048) for all three, including 8×1.
3. **Collapse.** Plot the local slopes of σ and of ⟨h_d⟩ − h_∞ against L/ℓ_p for the three shapes on
   one figure, with the qKPZ and qEW lines.
4. **κ_eff** from the dumped fields, as in §5.9, to rule out the box shape by shape.
5. **Document** as a new §5.10 in `docs/analysis_math.md`, with figures in `results/aniso_nu/`.

## Reading the outcome
- **The slopes collapse on L/ℓ_p and 16×1 gives 1/ν_eff ≲ 0.62:** the 0.685 is a crossover effect. Then
  extrapolate the collapse to estimate the L where 8×1 would reach 0.58, and quote 1/ν_qKPZ from 16×1.
- **16×1 gives ≈ 0.69 even though its ℓ_p is clearly smaller:** the threshold exponent is not set by the
  geometric crossover. Then the next suspect is the finite-size scaling itself: what the threshold of a
  directed-percolation blocking path in an L × P box measures. κ = P/w fixed already gives P ~ L^{ν⊥/ν∥},
  so the question is the corrections in P and the escape criterion (first blocking path vs. any). That is a
  separate plan.
- **4×1 and 16×1 differ but don't collapse on L/ℓ_p:** ℓ_p is not the right scale. Try ℓ× or c/λ from the
  tilt.

## Files
- **New, not versioned:** `data/aniso_nu/{pilot_ax16,calib_ax4,calib_ax16,ax4,ax16}/` and their `run.sh`.
- **Changed:**
  - `scripts/slurm_sweep.sh`: the `DUMP` switch;
  - `scripts/hd_variance.py` and `joint_fit.py`: the `ax*ay1` glob and the plot against L/ℓ_p.

## Outcome (2026-10-21)
Done; documented in §5.10 of `docs/analysis_math.md` (figures in `results/aniso_nu/`). Joint 1/ν over
L = 256–2048, 200 seeds per L: 1×1 0.80 (qEW-compatible control), 4×1 0.66, 8×1 0.67, 16×1 0.73, with
ℓ_p = —, 270 ± 170, 130 ± 20, 93 ± 10. The second case of "Reading the outcome" applies: 16×1 has the
shortest ℓ_p and is the farthest from 0.577, and the local slopes do not collapse on L/ℓ_p. The threshold
exponent is not set by the geometric crossover.
ζ_W(l) (`results/aniso_nu/zeta_W_shapes_L2048.png`) shows the knob is weaker than the pilot suggested:
8×1 and 16×1 have nearly the same crossover (ℓ_p ratio 1.5, they collapse under l/ℓ_p), and 4×1 crosses
over later but with a different curve that levels at 0.68–0.75.
