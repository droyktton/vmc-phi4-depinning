# Quasi-static sweep in Δ: h_d(Δ), avalanches and the pinning-force correlator

## Goal
Run the quasi-static velocity-driven ensemble (`main_vd.cu`, v → 0⁺) over a grid of disorder strengths Δ, at
fixed L, for a few spring constants H0. From each run we get:

- **h_d(Δ):** the stationary ⟨F⟩, extrapolated as H0 → 0. The deficit h_d − ⟨F⟩ ∝ L_m^{-1/ν}.
- **Avalanches:** P(S), τ, and the cutoff S_m = ⟨S²⟩/2⟨S⟩ ~ L_m^{d+ζ}.
- **The pinning-force correlator:** C(w) = ⟨F(W) F(W+w)⟩_c, with L·C(w) = Δ(w) the renormalized disorder
  correlator of the FRG. Its width scales as L_m^ζ.

The physics question is whether these change character at some Δ_c, from elastic (qEW: ζ ≈ 1.25,
τ ≈ 1.11) to plastic depinning.

### Key scale
Provided L ≫ L_m, the cutoff length is the spring's L_m = sqrt(0.47/H0), not L. The elastic/plastic
criterion is therefore L_m vs l_o(Δ): l_o ≈ 1500 at Δ = 0.3 and ≈ 100 at Δ = 1.

- If Δ_c moves with H0 as l_o(Δ_c) ≈ L_m, the change is a crossover.
- If Δ_c stays put, it is a transition.

## Pilot (2026-10-29): what it fixed
The data are on the cluster in `data/pilot_qs/`, and in `data/pilot_qs` (vader, L = 512) and
`data/pilot_qs_L1024` (cluster, L = 1024) in this worktree. All runs are one seed, with H0 = 3e-5
(L_m = 125) unless stated.

| check | result |
|---|---|
| dW convergence | dW = 0.5 and 0.1 give the same ⟨F⟩, sd(F) and number of S ≥ 10 avalanches (to 3–5 digits). dW = 2 already merges 5–8 % of avalanches. **Use dW = 0.5.** |
| L at fixed L_m | L = 512 and 1024 give the same ⟨F⟩ and S_m. sd(F) ∝ L^{-1/2}. The avalanche rate per row of W is ∝ L, so large L/L_m is cheaper statistics. **Use L = 1024.** |
| transient | With h_u ≈ h_d, F is stationary from the start. Discard the first 10 % anyway. |
| no-passing | No S < 0 in any run. |
| bubbles | No detached φ > 0 or φ < 0 domains in any dumped configuration (19 per run at Δ = 0.3, 1.0, 1.5; L = 512). Isolated sites can't be bistable with c = 1 (p = 3 − r > 0), so nucleation needs a whole weak region. **Check it anyway in production** (see below). |
| non-convergence | At Δ = 1.5, L = 1024, some steps hit the sweep cap (2000·Lx) with `converged = 0` and S = 0. There are 3, 11 and 41 per ~1000 rows for dW = 0.5, 0.1 and 0.02, i.e. ∝ 1/dW. They are most likely the saddle-node bottleneck when W lands just below an avalanche trigger. The avalanche is counted on the next step, so S is unaffected, but each one costs minutes of GPU. A small dW makes it worse. |
| cost | There is a floor of about 3000 sweeps per step, even with no avalanche. One step takes ~0.07 s at L = 512² on the GTX 1080, and 0.06–0.1 s at 1024² on the RTX 5070 Ti (node211). |

Avalanche rates (S ≥ 10, L = 1024) are about 0.26 per row of W at Δ = 0.3 and about 0.12 at Δ = 1.5.
With dW = 0.5, 10⁴ avalanches therefore take ~10⁵ steps, which is a few GPU-hours per seed.

A first hint, from one seed and 100–500 avalanches: between L_m = 125 and 217, S_m grows as L_m^{2.1} at
Δ = 0.3 (ζ ≈ 1.1, elastic) but as L_m^{1.05} at Δ = 1.5 (ζ ≈ 0). At Δ = 1.5 the drive shows clean
stick-slip with system-spanning avalanches.

## Production protocol

| parameter | value |
|---|---|
| Δ | 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2. Capped at 1.2 until h_nuc(Δ) is known (bubbles, below). |
| H0 | 1e-4, 3e-5, 1e-5 (L_m ≈ 70, 125, 217); add 3e-6 (L_m ≈ 400) at a few Δ if time allows |
| L × Ly | 1024 × 1024, `--reflect-y` (L ≥ 4 L_m except at H0 = 3e-6) |
| dW | 0.5 |
| h_u | ≈ h_d(Δ, L) from bisection; it only shortens the transient. Use h_est = 0.008 (Δ/0.2)^{4/3} if unknown (0.0663 measured at Δ = 1). |
| seeds | 5 |
| length | ≥ 10⁴ avalanches with S ≥ 10 per (Δ, H0) in total. For C(w), a W travel of ≥ 50 correlation lengths of C(w) (60–100 rows at L_m = 125). Take the larger of the two. |
| dumps | `--dump-every k` with ~100 configurations per seed (full box: 4 MB each at 1024²) |

Build one binary per Δ on the frontend (it builds for every architecture of the gpu partition):

```bash
make vd AMPDIS=0.85 VD_TARGET=phi4vmc_vd_D0.85
```

On vader, force the GTX 1080 architecture: `make vd ARCH=sm_61 ...`. Otherwise the Makefile detects
the GTX 980 and builds sm_52.

A run, from its output directory:

```bash
$ROOT/phi4vmc_vd_D0.85 1024 1024 3e-5 <h_u> 0.5 <nsteps> <seed> --reflect-y --dump-every <k>
```

It writes `vd_seed<s>.dat` (`step W u F S sweeps converged`) and `vdfield_step<k>_seed<s>.bin`.

For job arrays, `scripts/slurm_vdyn.sh` reads the seed from field 9 of each command, which is the
`main_vdyn.cu` argument order. For `main_vd.cu` the seed is field 8: either make the field
configurable, or use the small runner of the pilot (`data/pilot_qs/slurm_pilot.sh` on the cluster).

## Analysis
- `scripts/vdrive_stats.py vd_seed*.dat --skip <n>` computes ⟨F⟩, Var F, C(w) and P(S) per (Δ, H0), and
  warns about non-converged steps and S < 0.
- τ comes from a fit of P(S) between S ≈ 10 (below that, single-site flips dominate) and S_m, checked
  with a collapse P(S) S^τ vs S/S_m.
- S_m(L_m) gives d + ζ.
- ζ from geometry: `scripts/vdrive_sq.py` reads the `vdfield_step*` dumps directly.
- Δ(w): rescale L·C(w) by Δ(0) and by its width, then compare the shapes across Δ.

## Checks for every run
1. `converged == 1` for all steps. Count, and keep a list of, the steps that hit the cap.
2. No S < 0.
3. **Bubbles:** in every dump, count the φ > 0 domains not connected to the bulk below the wall, and the
   φ < 0 domains not connected to the bulk above it, with periodic x:
   `python3 scripts/bubbles.py <rundirs> --L 1024 --csv bubbles.csv`.
   - A bubble *ahead* of the wall is nucleation: flag that (Δ, H0) and analyse it separately.
   - A bubble *behind* is a pocket pinched off and left behind by the wall. That is plasticity, and
     worth tracking against Δ and L_m.

   In the Δ = 1 fixed-force runs with full-box dumps (`data/vdrive_D1.0_full`, L = 1024, ε = 0.0125
   and 0.025), 5 of 200 snapshots have bubbles. All of them are behind the wall: 1.7k–24.5k sites,
   130–490 rows back. None is ahead.

## Open before or alongside production
- **h_nuc(Δ):** the field at which a uniform φ = −1 state with no wall nucleates domains, compared with
  h_d(Δ). It sets the largest safe Δ. It needs a uniform initial condition in the driver.
- **Diagnose the non-converged steps** at Δ = 1.5: dump the field at one such W and see what keeps moving
  (bubble, marginal site, saddle-node bottleneck).
- **Optional speed-up:** an adaptive W advance that skips the linear stretches of F(W) where nothing
  flips. They are most of the steps at large Δ.
