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

on an `Lx x Ly` grid, periodic in `x`, anti-periodic in `y` (so that the
dynamics always hosts a single domain wall spanning the sample). `Lx`
is the wall's length (the `L` argument); `Ly`, the room it has to move
in, defaults to `Lx` but should scale as the wall's vertical extent,
`Ly = k Lx^1.25` at fixed `k` (`--ly-k k`): a square box squeezes the
wall more and more as `L` grows (it is `k = L^-0.25`), and since a
longer path meets stronger pinning, `h_d` itself depends on `k`. The
quenched disorder `r(x,y)` is an uncorrelated random-bond field, uniform
in `[-Delta, Delta]` per site (`Delta` == `AMPDIS`). The paper sets
`c = eps0 = 1` without loss of generality; these are the `CEL` and
`EPSILON0` build constants here (see `Makefile`).

## The VMC algorithm

An elementary move replaces `phi_ij` by the root of the cubic
steady-state equation (`d phi_ij/dt = 0`) closest to its current value
(`vieta_solver` in `misistema.h`, using Vieta's trigonometric formula).
Sites are swept in a checkerboard (2-coloring) decomposition so that a
whole color updates in parallel on the GPU, since each site's four
neighbors always belong to the other color. Only the 32x32 tiles where
something changed nearby are updated (`vmc_tiles_kernel`); near `h_d`
that is a few % of the lattice. Sweeps repeat until a metastable state
is reached (`find_next_metastable`): the largest residual
`max |d phi/dt|` must drop below `TOLRES = 1e-5`, and then no site may
move more than `1e-4` over the next 1000 sweeps (this tells a fixed
point from the saddle-node bottleneck just above `h_d`). The legacy
criterion, a cutoff `TOLVEL` on the *mean* residual (`--tolres 0`), is
blind to local avalanches at large `L` and left 13-43 % of the saved
critical configurations of the L=1024/2048 sweep still moving.

`h_d` is located per sample by **bisection on h** (`find_depinning_field`
in `main.cu`): whether the interface escapes the sample (the lane at
0.95 Ly flips sign) or stays pinned determines the next bisection
interval. Each trial starts from the last pinned state (the first from
a flat interface), and after an escape that state is restored. By
Middleton's no-passing theorems that gives the same outcome as
restarting from the flat interface, and checked directly
(`--verify-cold`) it does, in every one of 60 samples (L=512-2048).
See [docs/analysis_math.md](docs/analysis_math.md) §2.

Around `h_d`, `scan_magnetization_jumps()` additionally records the
sequence of metastable states on a finer field grid — the
magnetization-jump ("avalanche") statistics discussed in the paper.

## Build

Requires the NVIDIA CUDA toolkit (`nvcc`) and a GPU. The counter-based
RNG (`Random123/`, vendored, header-only) is the only dependency.

```
make                       # builds ./phi4vmc with the default parameters
make AMPDIS=0.4 CEL=1.0    # override disorder strength / elastic constant
make AMPDIS=0.6 TARGET=phi4vmc_D0.6               # named binary (one per Delta)
make ARCH=sm_86                                    # force a GPU architecture
```

The GPU architecture is chosen automatically, so the same `make` works on
a workstation/laptop and on the cluster: with a visible GPU it builds for
that GPU's compute capability (from `nvidia-smi`); with none (e.g. a
cluster frontend) it builds one binary for all of `CLUSTER_ARCHS`
(the cluster's `gpu` partition: `sm_86`, `sm_120`). `ARCH=sm_XY` forces
one. It compiles as C++17 (required by the Thrust in CUDA >= 13), or
C++14 with CUDA < 11. `TARGET` lets one binary per `Delta` coexist;
since `make` doesn't track the `-D` values, use `make -B` when
rebuilding with a different `AMPDIS`.

## Run

```
./phi4vmc L h_low h_high tol seed [--with-magjumps] [--dump-field] [options]
```

- `L`: lattice size (must be even, for the checkerboard decomposition;
  a multiple of 32 for the active tiles)
- `h_low`, `h_high`: bisection bracket (widened automatically if
  `h_low` depins or `h_high` stays pinned)
- `tol`: bisection tolerance on `h_d` (the paper uses `1e-4`)
- `seed`: disorder-realization seed
- `--with-magjumps`: also scan 100 fields within 5% of `h_d` for
  magnetization-jump ("avalanche") statistics. This probes deliberately
  in the critical-slowing-down region, so it's comparatively expensive;
  it's off by default since `h_d` and the critical configuration are
  already fully determined without it.
- `--dump-field`: also save the whole critical field `phi` (see below),
  needed for the arc-length analysis of `scripts/local_roughness.py`.
- `--ly N` or `--ly-k k`: the box height (above). With `k = 1` the
  critical wall spans at most ~12 % of `Ly` (L=512-1024, Delta=0.4-1.0),
  against up to ~45 % in a square box at L=1024. Both keep the legacy
  escape test (some site of row 0.95 Ly flips), which censors wide
  critical walls whose strongest pinning lies just below that row.
- `--path P --headroom H [--y0 Y]`: start at row `Y` (default 32), and
  count the trial as depinned once *all* of row `Y+P` has flipped;
  `Ly = Y+P+H`. `P` sets the finite-size regime (path over wall width,
  arXiv:1308.4329), and `H` must exceed the widest critical wall. The
  solver prints the critical wall's rows and a `WARNING` when it comes
  within 32 rows of `Ly`. `run_batch.py --adapt-headroom` then solves that
  seed again with twice the headroom: the disorder doesn't depend on
  `Ly`, so it is the same sample. See `docs/analysis_math.md` §1.
- Checks and comparisons: `--verify-cold` (relax again from a flat
  interface at `h_d` and compare), `--no-warm` (every trial from a flat
  interface), `--no-tiles` (sweep the whole lattice), `--tolres x`
  (`0`: legacy criterion), `--tile-eps x`, and `--relax-from file`
  (relax a saved field at `h_low` and report how far it moves).

Each trial prints a progress line (field, outcome, sweeps, time, and the
work actually done in full-sweep units).

Example:

```
./phi4vmc 256 0.0 0.1 0.0001 1234
```

Per run, this writes:

- `critica_h<h_d>_seed<seed>.dat` — the critical (depinning) interface
  configuration, as `(x, y)` domain-wall coordinates.
- `criticamag_h<h_d>_seed<seed>.dat` — the row/column magnetization
  profile of the critical configuration.
- `logfile.dat` — the run's parameters.

With `--dump-field`, it additionally writes:

- `criticafield_h<h_d>_seed<seed>.bin` — the critical field `phi` as raw
  float32, `L*L` values, row-major (index `x + y*L`); 4 MB at `L=1024`.

With `--with-magjumps`, it additionally writes:

- `jumps_seed<seed>.dat` — magnetization jump vs. field near `h_d`.
- `metas_seed<seed>.dat`, `metasmag_seed<seed>.dat` — the corresponding
  sequence of metastable interface configurations / profiles.

To collect statistics over many samples (the paper uses from tens to a
couple hundred, depending on `L` and `Delta`), just loop over the seed in
a shell script, e.g.:

```
for seed in $(seq 1 100); do
  ./phi4vmc 256 0.0 0.1 0.0001 $seed
done
```

## Performance

Cost grows steeply with `L`, dominated by trials near `h_d` where the
physics itself relaxes slowly (critical slowing down) — not by the
per-sweep GPU cost, which is cheap even at large `L`. Rough data points
on a GTX 1080, one sample each:

| `L`  | default run (no `--with-magjumps`) |
|------|-------------------------------------|
| 32   | < 1 s                                |
| 128  | ~4 s                                 |
| 1024 | ~4 min                               |

Two things keep this bounded rather than unpredictable:

- **`--with-magjumps` is off by default** (see [Run](#run)) — the
  avalanche scan probes 100 fields all within 5% of `h_d`, i.e.
  deliberately in the slow region, and was the dominant cost at large
  `L` even after the fix below.
- **`MAXSWEEPS`** (`misistema.h`) caps how long `find_next_metastable`
  waits for a trial to converge or escape: max(2×10⁶, 500 L) sweeps (at
  `L=8192`, one pinned trial took 8.3×10⁵ to converge). A trial still
  moving at the cap counts as depinned, so the saved critical
  configuration is always a confirmed fixed point (a capped trial
  counted as pinned was seen to escape later, at `L=8192`). The
  original cap (1e8 sweeps) was, in practice, an unbounded wait.

An earlier attempt at warm starts was dropped because, on an `L=32`
sample, it found a different `h_d` than the flat restart. With the
legacy (mean-residual) stopping criterion that does happen, and was
reproduced here: a warm start stops even more easily on a slow local
avalanche. The warm start used now only ever starts from a *pinned*
state below the trial field, and with the local stopping criterion
it agrees with the flat restart in every sample tested.

Cost per sample, legacy vs. current, same GPU: `L=2048`: ×8
(`Delta=0.4`) to ×19 (`Delta=1.0`); `L=4096`, `Delta=1.0`: 1361 s → 62 s.

## Analysis

The mathematics of the whole pipeline (model, VMC move and bisection,
wall extraction, every roughness estimator, the l_o definitions and fits,
the Larkin / two-crossover model) is written out in
[docs/analysis_math.md](docs/analysis_math.md).

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

`scripts/fit_roughness.py` fits the roughness exponent `zeta` from
`S(q) ~ q^-(1+2*zeta)` over a chosen low-`q` range of
`structure_factor.dat`:

```
python3 scripts/fit_roughness.py data/L128_D0.2/structure_factor.dat
python3 scripts/fit_roughness.py data/L128_D0.2/structure_factor.dat --qmin 0.05 --qmax 0.2
```

With no `--qmin`/`--qmax`, it auto-picks the best-R² contiguous window
in the low-`q` half of the data, but also prints the local
(secant-slope) effective exponent between every pair of points so you
can judge the fit range yourself — the paper's two roughness regimes
(quenched-EW `zeta~1.2` below the crossover length `l_o`, versus
invasion-percolation `zeta_eff~0.5` above it) mean the "best" window
isn't always the physically relevant one, especially with modest `L`
and sample counts. Use `--qmin`/`--qmax` to pin the fit to a specific
range once you've picked one from the diagnostics.

`scripts/plot_critical_configs.py` plots each sample's critical
configuration: the raw (possibly multivalued, where there are
overhangs) wall points, overlaid with the single-valued height profile
`u(x)` used by `structure_factor.py`. One PNG per sample is written to
`<datadir>/configs/`; pass `--grid` for a single overview figure with
every sample as a small subplot.

```
python3 scripts/plot_critical_configs.py data/L128_D0.2 --L 128 --grid
```

`scripts/detect_overhangs.py` detects multivalued (overhang) regions in
each sample and reports their size relative to `L`: the fraction of
columns that are multivalued (a per-sample estimate of the paper's
overhang probability `rho`), the paper's overhang-size measure `u_o`
(and `u_o/L`), and each individual overhang region's extent as a
fraction of `L`. Run it on one file for a detailed report, or on a
directory for a per-sample summary CSV plus sample-averaged statistics;
`plot_critical_configs.py --highlight-overhangs` shades the same
regions on the configuration plots.

```
python3 scripts/detect_overhangs.py data/L128_D0.2/critica_h0.008787_seed13.dat --L 128
python3 scripts/detect_overhangs.py data/L128_D0.2 --L 128
python3 scripts/plot_critical_configs.py data/L128_D0.2 --L 128 --grid --highlight-overhangs
```

Note: `is_wall_smooth`'s 5-point smoothing always spreads one genuine
crossing over 2 (occasionally more) adjacent rows, so raw wall-point
counts alone would flag *every* column as "multivalued". `detect_overhangs.py`
first collapses same-column detections within `--merge-gap` rows
(default 1) of each other into a single branch, so `--threshold`
(default 2 branches) reflects genuinely separate crossings; the
original `tiene_overhang.awk` used a threshold of 6 *raw* points, which
is roughly 3 branches here.

(`scripts/wall_io.py` holds the wall-file parsing, height-profile and
overhang-detection code shared by the scripts above; it's a library
module, not meant to be run directly.)

### Local roughness exponent and crossover length (Python)

These scripts compare ways of measuring the scale-dependent roughness
exponent `zeta(l)` and the crossover length `l_o` between the
self-affine qEW regime (`zeta ~ 1.25`) and the large-scale self-similar
one. They work on the raw fields (`--dump-field`), not on the
`critica_*.dat` points: those come from a detector that only finds sign
changes along `y`, so it misses the vertical stretches of the wall
(the overhangs) and gives unordered points. Instead
`scripts/wall_contour.py` traces the `phi=0` line (sub-pixel marching
squares, periodic in `x`) into an ordered curve. Requires `scipy` and
`scikit-image`.

`scripts/local_roughness.py` measures, per sample, and averages
(bootstrap errors over samples):

- global, on the univalued `u(x)` (paper Eq. 9): `S(q)`, `B(r)`, `W2(L)`;
- local, on segments of arc length `l` rigidly rotated to their tilt,
  with tilt = end-to-end chord and = principal axis (pca): `W2(l)`,
  `B(r,l)`, `S(q,l)` (chord segments: FFT; pca segments: cosine
  transform, `--pca-transform`); plus the fraction of
  segments still multivalued in their own frame, the aspect ratio
  `W(l)/l` and the rms tilt angle.

From these it gets local exponents (log-derivatives), several
definitions of `l_o` (local exponent dropping below `--zeta-th`,
multivalued fraction reaching `--fmv-th`, and data collapse across
`Delta` at fixed `L`), and fits `l_o ~ Delta^-alpha` (paper: ~2.2).
Datasets with `l_o > --lo-max-frac * L` (default `L/4`) are left out of
the fits and collapses, and the collapses only use scales `l <= L/4`:
comparing `L=1024` with `L=2048`, `l_o` agrees between the two up to
about `L/4` but is overestimated beyond (open markers in
`lo_vs_delta.png`).

```
DELTAS="0.4 0.6 0.85 1.0" LS="256 512" NSAMPLES=30 scripts/sweep_disorder.sh
python3 scripts/local_roughness.py data/sweep/L*_D* --outdir data/sweep/analysis
```

`scripts/sweep_disorder.sh` builds one binary per `Delta` and runs
`run_batch.py --dump-field` into `data/sweep/L<L>_D<Delta>/`. On a SLURM
cluster, `scripts/slurm_sweep.sh` does the same as a job array with one
GPU per `(L, Delta)` (build the binaries first; see its header):

```
DELTAS="0.4 0.5 0.6 0.7 0.85 1.0" LS="1024 2048" NSAMPLES=30 \
    sbatch --array=0-11 scripts/slurm_sweep.sh
```
 Outputs of
`local_roughness.py`: `lo_table.dat`, `alpha.dat`, per-dataset
`local_*.dat`/`global_*.dat`/`Sql_*.dat`/`Brl_*.dat`, and plots
(`zeta_local_L*.png`, `crossover_indicators_L*.png`, `collapse_*.png`,
`lo_vs_delta.png`).

`scripts/fit_crossovers.py` (run after `local_roughness.py`, it reads
its per-sample cache) addresses why the local exponent of `W2(l)` shows
no qEW plateau: the qEW window sits between the Larkin length
`L_c ~ Delta^-2/3` (below it `zeta_L = 1.5`) and `l_o`. It collapses the
small-scale part of `zeta_W(l)` across `Delta` (shifts `~ Delta^-x_c`),
and fits all `(L, Delta)` curves to a two-crossover model (logistic steps
in `ln l` at `L_c` and `l_o`) to get the qEW exponent `zeta_q` without a
plateau. Outputs `crossover_fit.dat`, `collapse_larkin.png`,
`crossover_fit.png`.

```
python3 scripts/fit_crossovers.py data/sweep/L*_D* --outdir data/sweep/analysis --alpha-fixed 2.1
```

`scripts/synthetic_profiles.py` runs the same estimators on Gaussian
profiles with a known `zeta1 -> zeta2` crossover at a known `l_o`, to
calibrate them. What it shows:

- for `zeta > 1`, first-difference measures (global `B(r)`, and local
  `B(r,l)` too, even after tilt removal) can't report more than
  `zeta = 1`. Only `W2(l)` of rotated segments and the spectra see
  `1.25`;
- pca segments' ends don't match, so a plain FFT leaks (spectrum
  `~q^-2`, apparent `zeta` capped near 0.5); the cosine transform (even
  extension, no jump) recovers `1.25` and `0.5` as well as chord+FFT
  does, whereas a Hann window biases `zeta` up by 0.1-0.4. Every
  transform reads ~0.1 high at wavelengths below ~16 (arc-length
  resampling), and pca+DCT rises to ~2 below wavelength ~5: an
  artifact, not wall structure;
- even without any crossover, the local exponent from `W2(l)` drops
  near `l ~ L/2` (finite size), so only `l_o` well below that is
  meaningful.

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
