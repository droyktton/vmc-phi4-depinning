# vmc-phi4-depinning

GPU implementation of the **variant Monte Carlo (VMC)** solver for
depinning in the disordered phi^4 model, and of the driver that finds,
for each disorder realization, the depinning field `h_d` and the
critical interface configuration at depinning.

Reference: A. B. Kolton, E. E. Ferrero, and A. Rosso, "Depinning free of
the elastic approximation," Phys. Rev. B **108**, 174201 (2023)
([arXiv:2306.13415](https://arxiv.org/abs/2306.13415)).

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
that a metastable state has been reached. (`velocity_phi()` normalizes
the summed residual by `L`, not `L2`, so `TOLVEL` is a size-scaled
cutoff rather than a literal per-site mean velocity.)

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
RNG (`Random123/`, vendored, header-only) is the only dependency.

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

### Depinning field statistics and structure factor (Python)

`scripts/run_batch.py` runs `./phi4vmc` over many disorder seeds for a
fixed `(L, h_low, h_high, tol)` and collects `h_d` per seed into a CSV;
`scripts/analyze_hd.py` then reports the mean/variance/std of `h_d` and
plots its histogram, and `scripts/structure_factor.py` computes the
sample-averaged structure factor `S(q)` of the critical domain-wall
configurations. Requires `numpy` and `matplotlib`.

```
make
python3 scripts/run_batch.py --binary ./phi4vmc --L 128 \
    --h-low 0 --h-high 0.05 --tol 1e-4 --nsamples 40 --outdir data/L128_D0.2

python3 scripts/analyze_hd.py data/L128_D0.2/hd_summary.csv
python3 scripts/structure_factor.py data/L128_D0.2 --L 128
```

`analyze_hd.py` writes `<summary>_stats_histogram.dat`/`.png`;
`structure_factor.py` writes `structure_factor.dat`/`.png` (columns
`q`, mean `S(q)`, standard error) in `--outdir`. To sweep the disorder
strength `Delta`, rebuild with `make AMPDIS=<Delta>` before each batch
(`Delta` is a compile-time constant, see [Model](#model)) and use a
separate `--outdir` per `(L, Delta)`.

### Overhang / structure-factor pipeline from the paper (gnuplot/awk/octave)

The rest of `scripts/` has the original gnuplot/awk/octave pipeline used
for the paper's structure-factor and overhang analysis, run on the
`critica_*.dat` files:

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

Check each script's header comment for its expected file-naming
convention before running it.

## Citing

If you use this code, please cite:

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
