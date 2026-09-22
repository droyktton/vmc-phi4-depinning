# phi4_vmc_min

Minimal GPU implementation of the **variant Monte Carlo (VMC)** solver
for depinning in the disordered phi^4 model, and of the driver that
finds, for each disorder realization, the depinning field `h_d` and the
critical interface configuration at depinning.

Reference: Ferrero, Kolton, et al., *"Depinning without the elastic
approximation: pinch-off, overhangs and the structure factor"*,
[arXiv:2306.13415](https://arxiv.org/abs/2306.13415).

This is a trimmed-down version of a larger, multi-experiment codebase:
everything not needed to reproduce that paper's algorithm and analysis
(alternative Euler/Langevin dynamics, a different disorder model, an
unrelated AC-pulse experiment, an OpenGL live viewer, OpenCV contour
tooling, etc.) has been removed.

## Model

```
d phi/dt = c * Laplacian(phi) + eps0 * [ (1 + r(x,y)) * phi - phi^3 ] + h
```

on an `L x L` grid, periodic in `x`, anti-periodic in `y` (so that the
dynamics always hosts a single domain wall spanning the sample). The
quenched disorder `r(x,y)` is an uncorrelated random-bond field, uniform
in `[-Delta, Delta]` per site (`Delta` == `AMPDIS`). The paper sets
`c = eps0 = 1` without loss of generality; these are the `CEL` and
`EPSILON0` build constants here (see `Makefile`).

## The VMC algorithm

An elementary move replaces `phi_ij` by the root of the cubic
steady-state equation (`d phi_ij/dt = 0`) closest to its current value
(`vieta_solver` in `misistema.h`, using Vieta's trigonometric formula).
Sites are swept in a checkerboard (2-coloring) decomposition so that a
whole color updates in parallel on the GPU (`vmcop`), since each site's
four neighbors always belong to the other color. Sweeps repeat until the
mean residual velocity `d phi/dt` of the steady-state equation drops
below a cutoff `epsilon = TOLVEL` (`find_next_metastable`), signalling
that a metastable state has been reached.

`h_d` is located per sample by **bisection on h** (`find_depinning_field`
in `main.cu`): for each trial field the interface is relaxed from a flat
initial condition, and whether it escapes the sample (top lane flips
sign) or stays pinned determines the next bisection interval. Restarting
from the same flat initial condition at every trial `h` is valid because
Middleton's no-passing theorems guarantee the pinned/depinned outcome at
a given `h` doesn't depend on the relaxation path.

Around `h_d`, `scan_magnetization_jumps()` additionally records the
sequence of metastable states on a finer field grid — the
magnetization-jump ("avalanche") statistics discussed in the paper.

## Build

Requires the NVIDIA CUDA toolkit (`nvcc`) and a GPU. The counter-based
RNG (`Random123/`, vendored, header-only) is the only third-party
dependency — no `cufft`, `glut`/`GL`, or OpenCV are needed.

```
make                       # builds ./phi4vmc with the default parameters
make AMPDIS=0.4 CEL=1.0    # override disorder strength / elastic constant
```

## Run

```
./phi4vmc L h_low h_high tol seed
```

- `L`: lattice size (must be even, for the checkerboard decomposition)
- `h_low`, `h_high`: bisection bracket (must be pinned at `h_low`, depinned at `h_high`)
- `tol`: bisection tolerance on `h_d` (the paper uses `1e-4`)
- `seed`: disorder-realization seed

Example:

```
./phi4vmc 256 0.0 0.1 0.0001 1234
```

Per run, this writes:

- `critica_h<h_d>_seed<seed>.dat` — the critical (depinning) interface
  configuration, as `(x, y)` domain-wall coordinates.
- `criticamag_h<h_d>_seed<seed>.dat` — the row/column magnetization
  profile of the critical configuration.
- `jumps_seed<seed>.dat` — magnetization jump vs. field near `h_d`.
- `metas_seed<seed>.dat`, `metasmag_seed<seed>.dat` — the corresponding
  sequence of metastable interface configurations / profiles.
- `logfile.dat` — the run's parameters.

To collect statistics over many samples (the paper uses from tens to a
couple hundred, depending on `L` and `Delta`), just loop over the seed in
a shell script, e.g.:

```
for seed in $(seq 1 100); do
  ./phi4vmc 256 0.0 0.1 0.0001 $seed
done
```

## Analysis

`scripts/` has the gnuplot/awk/octave pipeline used for the paper's
structure-factor and overhang analysis, run on the `critica_*.dat` files:

- `tiene_overhang.awk`: classifies a critical configuration as having
  overhangs or not, and reports the roughness of its (single-valued)
  projection.
- `extrae_Sq.gnu` (uses `fft1block.m`): computes the structure factor
  `S(q) = |FFT(u(x))|^2` of a critical configuration's interface height
  `u(x)`.
- `overhangprob.sh` / `todoslosoverhangs.sh` / `proboverhangs.gnu`:
  aggregate the overhang probability over many samples, sizes and
  disorder strengths.
- `procesa_wall.gnu`, `hacer_todas_fft.gnu`, `hacer_todas_w2.gnu`,
  `extrae_Sq_from_wall.gnu`, `extrae_Sq_and_w2_from_wall.gnu`: batch
  variants of the above over a directory of samples.

These scripts predate this cleanup and use ad hoc file-naming
conventions (e.g. `L<size>_<Delta>/critica_*.dat` directories) — check
each script's header comment for its expected usage before running it.

## Notes / things worth double-checking against the paper

- `velocity_phi()` normalizes the summed residual by `L` (the linear
  size), not `L2` (the site count), so `TOLVEL` is not literally a
  per-site mean velocity but a size-scaled one. This matches the
  original code that produced published results, so it was kept as-is
  rather than "fixed" — but it's worth deciding deliberately if you
  change `L` a lot and want `TOLVEL` to mean the same thing across sizes.

## Citing

If you use this code, please cite the paper it implements the method for:

> A. B. Kolton, E. E. Ferrero, and A. Rosso, "Depinning free of the
> elastic approximation," Phys. Rev. B **108**, 174201 (2023).
> https://doi.org/10.1103/PhysRevB.108.174201

```bibtex
@article{PhysRevB.108.174201,
  title     = {Depinning free of the elastic approximation},
  author    = {Kolton, Alejandro B. and Ferrero, Ezequiel E. and Rosso, Alberto},
  journal   = {Phys. Rev. B},
  volume    = {108},
  issue     = {17},
  pages     = {174201},
  year      = {2023},
  month     = {Nov},
  publisher = {American Physical Society},
  doi       = {10.1103/PhysRevB.108.174201},
  url       = {https://doi.org/10.1103/PhysRevB.108.174201}
}
```

## License

MIT — see [LICENSE](LICENSE) — for the code in this repository.

`Random123/` is a vendored, header-only copy of the
[Random123](https://www.deshawresearch.com/resources_random123.html)
counter-based RNG library by D. E. Shaw Research, BSD-3-Clause licensed
(license text in each file's header).
