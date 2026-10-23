# Velocity-driven ensemble: quasi-static (VMC) and finite velocity (Euler)

## Context
So far the ensemble is force-driven: bisection on a uniform field for h_d. Here the mean velocity is
imposed instead. Only macroscopically flat walls are covered.
- **Spring.** A Zeeman term with a gradient, h(y) = h_u − H0 (y − W), ties the wall to the well
  center W. A wall at mean height u feels F = h_u + H0 (W − u), a spring 2H0 (the field pushes the
  wall with 2h per unit length), with mass length L_m = √(σ/2H0), σ = 2√2/3 for c = ε0 = 1.
- **Uniform part.** h_u keeps the wall near W instead of h_d/H0 rows behind it.
- **Travelling disorder.** Moving the disorder back, r(x, y + vt), is the same as advancing W.
- **Measured.** F, its mean, variance and correlator in W (Δ(w) of the FRG), and the avalanches.

## Code (branch `vdrive`)
- **`misistema.h`.** All changes are opt-in; with H0 = 0 and offset 0 the force-driven solver is
  bit-for-bit unchanged.
  - `Drive {h, grad, center, row_offset}` replaces the uniform field in the kernels.
  - `site_disorder` takes the lab row with a 64-bit index.
  - `shift_rows(n)` is the conveyor belt: it moves φ and the disorder by whole rows, an exact symmetry.
  - `positive_area_lab()` gives the φ > 0 area in the lab frame, continuous across shifts.
  - `euler_step(dt)` integrates the real dynamics with the force of `velocityop`.
- **`main_vd.cu`** (`make vd`), quasi-static, v → 0⁺. W advances in steps dW and the wall relaxes
  to the next metastable state with the VMC (no-passing holds). Output `vd_seed*.dat`:
  `step W u F S sweeps converged`.
- **`main_vdyn.cu`** (`make vdyn`), finite v. W = W₀ + vt with explicit Euler at T = 0. Output
  `vdyn_v*_seed*.dat`: `t W u F`.
- **Analysis.** `scripts/vdrive_stats.py` gives ⟨F⟩, Var F, C(w) and P(S);
  `scripts/vdyn_stats.py` gives F(v), τ_F and the quasi-static point.

## Validation (vader, GTX 1080, 2026-10-22/23)
- **Regression.** The force-driven solver of the branch gives the same h_d and critical field,
  bit for bit, as `main`: L = 256, Δ = 0.3, 3 seeds, 1×1 and 8×1 blocks. Adding the Euler left
  the quasi-static output bit-identical.
- **Clean lattice (Δ = 0).**
  - *Quasi-static:* the wall advances one row at a time, at F = 0.0029. That is the Peierls–Nabarro
    threshold; bisection gives 0.00288.
  - *Euler:* F ≈ v/2.12 (the continuum φ⁴ mobility) plus a lattice correction of 0.001–0.003, the
    same for dt = 0.05 and 0.025 to six digits.
- **Conveyor belt.** Ly = 512 and 2048, aligned with `--row-offset`, give bit-identical F and u at
  every output, both quasi-static (1001 steps) and with Euler (2001 outputs).
- **No-passing.** S ≥ 0 at every quasi-static step, and every step converged.
- **Euler towards the quasi-static limit** (Δ = 0.3, L = 256, H0 = 1e-4, seed 1). ⟨F⟩ = 0.0369,
  0.0205, 0.0133 and 0.0114 for v = 0.1, 0.03, 0.01 and 0.003, against 0.0105 for the VMC. Only
  3–11 τ_F per run, and the fast runs had not relaxed (next item), so the v = 0.1 value is a
  transient.
- **Relaxation of the spring.** The mean drive relaxes towards its stationary value in
  τ ≈ 1/(μ H0), with μ ≈ 2.12 the clean-wall mobility (longer with disorder): τ ≈ 4700 at
  H0 = 1e-4. Runs must last several τ, or start with F0 near the expected value.
- **Ensemble equivalence, clean lattice.** `main_vdyn.cu` with H0 = 0 runs at fixed force. With
  runs longer than 8τ, imposing v = 0.05 gives ⟨F⟩ = 0.02535 (H0 = 1e-3) and 0.02526 (H0 = 1e-4),
  and the fixed force h = 0.02535 gives v = 0.0502: agreement within 0.4 %.
- **Ensemble equivalence with disorder** (Δ = 0.3, L = 256, 3 seeds). Fixed force first, then
  the same v imposed with the spring, starting at F0 = h and running at least 8/(2.5 H0):

  | v | h (fixed force) | ⟨F⟩, H0 = 1e-4 | ⟨F⟩, H0 = 3e-5 |
  |---|---|---|---|
  | 0.0923 | 0.05 | 0.04993 | 0.04994 |
  | 0.0501 | 0.03 | 0.02995 | 0.02994 |
  | 0.0274 | 0.02 | 0.01999 | 0.01994 |
  | 0.0132 | 0.015 | 0.01468 ± 0.00015 | 0.01485 ± 0.00009 |

  For v ≥ 0.027 the two agree within 0.2 %. At v = 0.013 the spring gives a lower drive, by 2 %
  at H0 = 1e-4 and 1 % at H0 = 3e-5. That is what is expected once L_m is no longer much larger
  than the correlation length ξ_v, which grows towards the threshold. At h = 0.013 the fixed-force
  wall stops in all 3 seeds: an unbounded path finds stronger pinning than the 192-row path of
  the bisection (h_d ≈ 0.0128).

## First comparison with the force-driven ensemble
Δ = 0.3, L = 256, 3 seeds, quasi-static, steps 400–2000 of dW = 0.5:

| H0 | L_m | ⟨F⟩ |
|---|---|---|
| 1e-4 | 69 | 0.01081 ± 0.00014 |
| 3e-5 | 125 | 0.01189 ± 0.00018 |
| 1e-5 (Ly = 2048) | 217 | 0.0126 ± 0.0001, not stationary (halves differ by up to 0.0017) |
| bisection, square box | — | 0.01275–0.01280 |

⟨F⟩ rises towards h_d as L_m approaches L. The two need not agree at finite L: the bisection
takes the maximum over the path (it depends on κ), while the spring averages over the stationary
sequence of metastable states. A real comparison needs extrapolations: H0 → 0 and L growing,
against ⟨h_d⟩(L) at fixed κ. At small H0 each avalanche moves the wall by hundreds of rows, so the
runs must be much longer, and Ly must exceed the largest avalanche (Ly = 512 was too small at
H0 = 1e-5).

## Next
- **v(h) with disorder** by both routes (fixed force, H0 = 0; velocity-driven ⟨F⟩), checking the
  equivalence at finite v (L_m ≫ ξ_v), and β.
- **S(q) of snapshot series** (`--dump-every`).
- **Thermal noise.**
- **Production on the cluster.**
