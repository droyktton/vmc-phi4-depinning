# β, equivalence of ensembles and S(q) of moving walls (Δ = 0.3, L = 1024 and 2048, T = 0)

## Context
`main_vdyn.cu` (Euler, T = 0) runs either at fixed force (H0 = 0) or at imposed velocity with a
spring (H0 > 0); see `docs/vdrive.md`. At L = 256 the two give the same v(h) within 0.2 % for
v ≥ 0.027. Near threshold the spring gives a lower drive, by 1–2 %, and the gap shrinks with H0.

Here, at the sizes where the wall is still elastic:
- **Velocity exponent β:** v = A (h − h_c)^β, with h_c from the bisection.
- **Ensemble equivalence:** does ⟨F⟩ at imposed v converge to the fixed-force h as L_m grows?
- **Roughness:** S(q) of the moving walls.

Δ = 0.3 has the bisection production up to L = 8192 (κ = 20): ⟨h_d⟩ = 0.01339 at L = 8192, and
l_o ≈ 1500. With L = 1024 and 2048 the wall is mostly elastic, so the reference is qEW, β ≈ 0.25.
No thermal noise for now.

## Runs
Job lists come from `scripts/make_vdyn_jobs.py`, one line per run. They are run by
`scripts/slurm_vdyn.sh` (a job array that resumes on resubmission, with a `.done_seed<s>` marker per
run), under `data/vdrive_beta/`.

Box and dumps:

| L | Ly | dumped band (`--dump-band`) |
|---|---|---|
| 1024 | 512 | ±192 rows |
| 2048 | 1024 | ±384 rows |

The box is several wall widths plus the recentering margin. Seeds 1–5 dump 20 band snapshots from
the second half of each run.

1. **Stage ff, fixed force.**
   - h = h_c (1 + ε), with ε = 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, and 10 seeds.
   - The wall travels at least 800 rows after an equal transient.
   - Cost: 120 runs, about 24 GPU-h on a GTX 1080 (5.4 h for L = 1024, 18.5 h for L = 2048). The
     longest run is about 0.8 h. Snapshots take about 5 GB.
2. **Stage vd, velocity-driven, after ff.**
   - The v measured at each h, started at F0 = h.
   - Two v below the ff range (0.5 and 0.25 times the smallest), with F0 from a power-law fit.
   - Springs H0 = 3e-5, 1e-5 and 3e-6, so L_m ≈ 125, 217 and 396; 5 seeds.
   - Each run lasts at least 8 relaxation times of the spring, 1/(μ H0), with μ = dv/dh from the
     fit. This dominates the cost: at H0 = 3e-6 a run at L = 2048 takes about 2 h on a GTX 1080.
     The whole stage is roughly 70–140 GPU-h. To trim it, drop H0 = 3e-6 at L = 2048 (`--H0s` per
     call) or use fewer `--vd-seeds`.

## Commands (repository root, on the frontend)
```bash
make vdyn AMPDIS=0.3 VDYN_TARGET=phi4vmc_vdyn_D0.3
mkdir -p logs
python3 scripts/make_vdyn_jobs.py ff > jobs_ff.txt
N=$(wc -l < jobs_ff.txt); P=4
JOBS=jobs_ff.txt PER_TASK=$P sbatch --array=0-$(( (N+P-1)/P - 1 )) scripts/slurm_vdyn.sh
# when ff is done (check: ls data/vdrive_beta/ff_L*/h*/.done_seed* | wc -l == 120)
python3 scripts/make_vdyn_jobs.py vd > jobs_vd.txt
N=$(wc -l < jobs_vd.txt); P=2
JOBS=jobs_vd.txt PER_TASK=$P sbatch --array=0-$(( (N+P-1)/P - 1 )) scripts/slurm_vdyn.sh
```

## Analysis (cpu queue)
- **v(h), β (h_c fixed and free) and equivalence:**
  `python3 scripts/vdrive_beta.py data/vdrive_beta --outdir results/vdrive_beta`. It writes
  `beta.txt` and `v_of_h.png` (left: v against h − h_c for both ensembles; right: the relative
  difference between ⟨F⟩ and the fixed-force h at the same v, against L_m).
- **S(q):** one call per condition and L, for example
  `python3 scripts/vdrive_sq.py data/vdrive_beta/ff_L2048/h0.0161 --L 2048 --outdir results/vdrive_beta/Sq_ff_h0.0161`.
  Then `scripts/sq_regimes.py` on those directories gives the slopes against the qEW (3.5),
  qKPZ (2.26) and Larkin (4) references.

## Reading the outcome
- **Equivalence holds** if the relative difference goes to 0 as L_m grows, at every v. The v where
  it fails at a given H0 estimates ξ_v ≈ L_m.
- **β.** Compare it with the qEW value, 0.25. If h_c fixed and h_c free give different β, the
  points closest to the threshold carry finite-size effects. Then compare L = 1024 against 2048 at
  the same ε, and rely more on the velocity-driven points below the ff range.
- **S(q) of the moving wall.** It should follow the critical one up to ξ_v and cross over to the
  fast-flow regime (EW, ζ = 1/2) above it.

## Status (2026-10-25)
- **Stage ff done** (array 973476, 120 runs, 2026-10-24): results in `docs/vdrive.md` §5. Not
  yet critical: the local β goes from 0.71 to 0.35 without a plateau, and S(q) shows that the
  walls are critical over most of L only at ε ≲ 0.1. The ε range must extend below 0.05.
- **Stage vd not launched.**

## Checked on vader (L = 256, miniature of the whole chain)
- Both stages ran through `slurm_vdyn.sh`, and resubmission skips finished runs.
- `vdrive_beta.py` and `vdrive_sq.py` read the outputs, including band snapshots.
- Equivalence within 0.06–0.8 % on these short runs.
