# Mathematics of the solver and of the analysis

This document writes out, in the code's notation, everything that is
computed: the model and the algorithm that finds the depinning field and
the critical configuration (`main.cu`, `misistema.h`), the extraction of
the wall (`scripts/wall_contour.py`), the roughness estimators
(`scripts/roughness_estimators.py`, `scripts/local_roughness.py`), and
the two-crossover model (`scripts/fit_crossovers.py`). The results of
the `L = 1024, 2048` sweep are at the end (figures in
`results/sweep_L1024_L2048/`).

Reference: A. B. Kolton, E. E. Ferrero, A. Rosso, Phys. Rev. B **108**,
174201 (2023).

---

## 1. Model

Scalar field $\phi(x,y)$ on an $L_x\times L_y$ lattice ($L \equiv L_x$):

```math
\partial_t \phi = c\,\nabla^2\phi + \epsilon_0\left[(1+r(x,y))\,\phi - \phi^3\right] + h
```

- $\nabla^2$ is the 5-point discrete Laplacian.
- The disorder is *random bond*: $r(x,y)$ is independent from site to
  site and uniform in $[-\Delta,\Delta]$, so $\langle r\rangle = 0$ and
  $\langle r^2\rangle = \Delta^2/3$ (`site_disorder`, a Random123
  generator keyed by site and seed).
- $c$ = `CEL`, $\epsilon_0$ = `EPSILON0` and $\Delta$ = `AMPDIS`; the
  runs use $c = \epsilon_0 = 1$.

**Boundaries.** Periodic in $x$, $\phi(x+L_x,y) = \phi(x,y)$;
anti-periodic in $y$, $\phi(x,y+L_y) = -\phi(x,y)$. With an anti-periodic
boundary any configuration has an odd number of walls, so there is
always a wall spanning the sample, and no fixed boundary that holds or
pulls it. For $h>0$ the $\phi>0$ region lies below the wall and grows
as $h$ increases. (Up to commit `9548076` the sign flip was applied to
the wrong neighbour, the interior one, of the rows $y=0$ and $L_y-1$.
That left a fixed defect at the seam, rows at $\phi\approx\mp0.24$, far
from the wall: in a square box the $h_d$ of 20 seeds came out identical
with either boundary.)

**Box height.** The wall starts at $0.2L_y$ and escapes at $0.95L_y$, so
$h_d$ is the strongest pinning met over a path $0.75L_y$ long, and it
grows with $L_y$. The wall's vertical extent grows as $L_x^{\zeta}$, so
for finite-size scaling the box keeps its shape in those units,
$L_y = k\,L_x^{\zeta}$ with $\zeta = 1.25$ and fixed $k$ (`--ly-k`,
rounded to a multiple of 32). A square box has $k = L^{-1/4}$, which
shrinks with $L$. At $L=1024$ the extent of the critical wall is
0.02–0.08 $L_x^{1.25}$ on average (Δ = 0.4–1.0), up to 0.12; going from
the square box to $k=1$ raises $h_d$ by 2.9 % (Δ = 0.4) and 1 % (Δ = 1).
With $k = 1$ the wall spans at most ~12 % of $L_y$.

**Equilibrium states.** Without disorder or field the local minima are
$\phi = \pm 1$. The flat wall is $\phi_0(y-u) = \tanh[(y-u)/\xi\sqrt2]$
(with the sign set by the boundary), of width $\xi = \sqrt{c/\epsilon_0}$
in the continuum.

## 2. Variant Monte Carlo (VMC)

### 2.1 Elementary move

At each site, the steady-state condition $\partial_t\phi_i = 0$ with the
neighbours held fixed is a cubic in $\phi_i$:

```math
c\left(\textstyle\sum_{\rm nn}\phi - 4\phi_i\right) + \epsilon_0\left[(1+r_i)\phi_i - \phi_i^3\right] + h = 0
\;\Longleftrightarrow\;
\phi_i^3 + p\,\phi_i - q = 0,
```

```math
p = \frac{4c}{\epsilon_0} - (1+r_i), \qquad q = \frac{c\sum_{\rm nn}\phi + h}{\epsilon_0}.
```

`vieta_solver` solves the cubic. With $Q = p/3$, $R = q/2$ and
$D = Q^3 + R^2$:

- **If $D \ge 0$** there is a single real root (Cardano):
  $\phi = \sqrt[3]{R+\sqrt D} + \sqrt[3]{R-\sqrt D}$.
- **If $D < 0$** there are three real roots (Vieta's trigonometric
  form):

```math
\phi_k = 2\sqrt{-Q}\,\cos\frac{\theta + 2\pi k}{3},\quad k=0,1,2,\qquad \theta = \arccos\frac{R}{\sqrt{-Q^3}},
```

with $\phi_0 > \phi_2 > \phi_1$: the largest, the middle and the
smallest.

The middle root $\phi_2$ is unstable. The move sends $\phi_i$ to the
stable root in the same basin as its current value: to $\phi_0$ if
$\phi_i > \phi_2$, and to $\phi_1$ otherwise. The site thus jumps
straight to the nearest local minimum instead of integrating the
dynamics.

### 2.2 Sweeps and convergence

- **Checkerboard.** All sites with $x+y$ even are updated in parallel,
  then the odd ones (two colours per sweep). Since a site's four
  neighbours are all of the other colour, each half-update is exact.
- **Active tiles.** The move is idempotent: with the same neighbours it
  returns the same value. A site therefore only needs updating if a
  neighbour changed in the previous (other-colour) half-sweep. The
  lattice is split into $32\times32$ tiles, and a half-sweep only
  updates the tiles where, in the previous half-sweep, some site of the
  tile or of its 4 neighbouring tiles changed (`vmc_tiles_kernel`,
  `spread_activity`). Skipping the rest is exact: with this rule alone
  the result is bit-for-bit that of updating every site (checked). In
  float arithmetic, though, converged sites keep flickering in the last
  bit, so no tile would ever go quiet. A change $|\delta\phi| \le$
  `TILE_EPS` $=10^{-6}$ is therefore applied but does not keep the tile
  active. It leaves residuals $\sim c\,$`TILE_EPS` on the neighbours,
  well below the stopping cutoff, and each block of `chop` sweeps
  starts with one full sweep. Near $h_d$ the wall moves by local
  avalanches, and the work drops to a few % of the full sweeps.
- **Residual velocity.** The legacy criterion (`--tolres 0`) computes,
  every `chop` = 100 sweeps,

```math
v = \frac{1}{L}\sum_{i=1}^{L^2}\partial_t\phi_i
```

  (`velocity_phi`). It is normalised by $L$, not $L^2$: since $\phi$
  jumps by 2 across the wall, a wall of length $L$ moving at mean
  velocity $\bar v_{\rm wall}$ gives
  $\sum_i\partial_t\phi_i\approx 2L\,\bar v_{\rm wall}$, so
  $v\approx 2\bar v_{\rm wall}$ whatever the size.
  and stops when $|v| <$ `TOLVEL` $= 10^{-4}$.
- **Why that is not enough.** $v$ is a mean over the wall. A local
  avalanche of size $l$ moving at speed $v_a$ only contributes
  $\sim l\,v_a/L$, so at large $L$ the relaxation can stop on a wall
  that is still moving. Relaxing the saved legacy critical
  configurations further, at the same $h$, showed that 13–43 % of them
  (by set, $L=1024$–$2048$, $\Delta=0.4$, $1.0$) were not metastable:
  the wall moved on by up to hundreds of rows, and 3 of 120 depinned.
- **Stopping (default).** A local criterion: the largest residual

```math
\max_i |\partial_t\phi_i| < \texttt{TOLRES} = 10^{-5}
```

  (`max_residual`), which vanishes only at a fixed point. Just above
  $h_d$, however, the dynamics has the bottleneck of a saddle-node
  bifurcation: the wall lingers almost still for many sweeps before
  escaping, and an instantaneous residual can't tell that from a fixed
  point. The criterion is therefore confirmed: once it holds, no site may
  move more than `CONFIRM_DPHI` $=10^{-4}$ over the next
  `CONFIRM_SWEEPS` $=1000$ sweeps (a mean speed 100 times below
  `TOLRES`); otherwise the wait starts again. Checks are made every
  `chop` = 1000 sweeps.
- **Cap.** After $\max(2\times10^6,\,500L)$ sweeps (`MAXSWEEPS`,
  `MAXSWEEPS_PER_L`) the relaxation gives up, and a trial still moving
  then counts as *depinned*. Near $h_d$ relaxations get very long at
  large $L$: at $L=8192$ a pinned trial took $8.3\times10^5$ sweeps to
  converge, and a trial that hit the former cap ($5\times10^5$ sweeps,
  then counted as pinned) escaped $2.3\times10^5$ sweeps later, which
  had both raised $h_d$ and saved a moving wall as the critical
  configuration. Counting a capped trial as depinned keeps every
  critical configuration a confirmed fixed point, at the cost of an
  $h_d$ that can only come out low, and only if a pinned relaxation
  outlasts the cap. The solver prints "still moving at the cap" when
  that happens.

### 2.3 Depinning field by bisection

For each trial $h$ (`find_depinning_field`):

1. **Initial condition:** the last pinned state found (warm start,
   below). The first trial starts from the flat wall, $\phi = +1$ for
   $y < 0.2L_y$ and $\phi=-1$ for $y \ge 0.2L_y$.
2. **Relaxation** with the rule above, with an early exit: if the wall
   crosses the row $y_e = 0.95L_y$, i.e. if $\max_x \phi(x, y_e) > 0$
   (`magnetization_lane`, `ESCAPE_LANE`), it stops there.
3. **Classification:** the trial is *depinned* if
   $\max_x\phi(x,y_e) > 0$, and *pinned* otherwise.

First the ends are checked: $h_{\rm low}$ must be pinned (if not, it is
divided by 1.5) and $h_{\rm high}$ depinned (if not, it becomes the new
$h_{\rm low}$ and $h_{\rm high}$ is multiplied by 1.5), so a bracket
that misses a sample's $h_d$ costs a few trials, not a wrong result.
Bisection on $[a,b]$ then replaces $a$ (pinned) or $b$ (depinned) by the
midpoint until $b-a \le$ `tol` $= 10^{-4}$. This defines

```math
h_d \equiv a, \qquad \text{critical configuration} = \text{metastable state at } h_d.
```

(The legacy code then stepped back, $a \leftarrow a-\text{tol}$, until
the trial was pinned again, and reported that. Its $h_d$ are thus one
`tol` lower.)

**Warm starts.** Let $M(h)$ be the state reached from the flat wall at
field $h$. For $a < h$, the pinned state $M(a)$ lies between the flat
wall and $M(h)$, so by no-passing (below) the dynamics started from
$M(a)$ also ends at $M(h)$, or escapes if and only if the flat start
escapes. Each trial therefore starts from the last pinned state, and
after an escape that state is restored (`save_state`,
`restore_state`). The wall only has to move from $M(a)$ to $M(h)$
instead of all the way from $y=0.2L_y$, and the final step-back is no
longer needed: the saved state at $a$ *is* the critical configuration.
The ordering argument is not strict in the bulk, where the flat
start's $\pm1$ differs from the relaxed $\pm\sqrt{1+r}$ either way;
those sites relax in a few sweeps, so it only matters near the wall.
The check is therefore empirical: `--verify-cold` relaxes once more
from the flat wall at $h_d$ and compares. With the default stopping criterion the two agree in every
sample tested (60, $L=512$–$2048$): both pinned, $\max|\delta\phi|
\le 5\times10^{-4}$. With the legacy criterion they did not: the warm
start then stops on slow local avalanches even more easily, and
reported $h_d$ up to 5 `tol` too high.

**Why restarting from the flat wall is valid.** By Middleton's
no-passing theorems the dynamics preserves order: if $\phi \le \phi'$
at every site, it stays so. Hence, at fixed $h$, whether the wall gets
pinned or escapes does not depend on the path. Concretely, the VMC rule
is monotonic: the roots of the cubic increase with $q$, i.e. with the
neighbours and with $h$, and the threshold $\phi_2$ decreases.

**Why the early exit doesn't change $h_d$.** With the anti-periodic
boundary, the magnetisation $M = \sum\phi$ grows as the wall moves up
and is largest with the wall at $y = L_y$. A depinned wall advances to
there and converges, instead of circulating. Stopping once it crosses
$y_e$ doesn't change the classification, since the decision is made on
that row anyway. Pinned trials never cross $y_e$, so they relax exactly
as before and the saved critical configuration is unchanged. Checked on
13 seeds: same $h_d$, and at $L=1024$ the total time dropped by 32 %.

**Initial bracket.** `slurm_sweep.sh` uses $[0.85,\,1.15]\,h_{\rm est}$
with $h_{\rm est} = 0.0195\,(\Delta/0.4)^{1.34}$, the fit of the
$L=1024$–$2048$ sweep (§8); `sweep_disorder.sh` keeps $[0,\,3h_{\rm est}]$
with $h_{\rm est} = 0.008\,(\Delta/0.2)^{4/3}$ (Larkin scaling, §7).

**Cost.** Per sample, on the same GPU, legacy vs. default: $L=2048$,
$\Delta=0.4$: 254 s → 33 s (×8); $\Delta=1.0$: 250 s → 13 s (×19);
$L=4096$, $\Delta=1.0$: 1361 s → 62 s (×22). The gain grows with $L$:
the warm start saves the wall's travel, $\propto L$, and the tiles
skip the quiet part of the lattice, a growing fraction.

## 3. The wall as a curve

The critical configuration is saved in full (`--dump-field`). The points
in `critica_*.dat` can't be used for this analysis: they only detect
sign changes along $y$, so they miss the vertical stretches (the
overhangs) and come unordered.

1. **Contour.** The wall is the $\phi = 0$ level line, traced with
   sub-pixel *marching squares* (`skimage.measure.find_contours`) on the
   field tiled 3 times along $x$. The code takes the longest open
   contour that spans the three copies without touching the boundary
   rows in $y$, and cuts exactly one period, from a point $P$ to
   $P + (L,0)$. Closed contours are bubbles (isolated domains): they are
   counted and discarded.
2. **Arc-length resampling.** The curve is resampled with $n$ points
   equally spaced in arc length $s$, with
   $ds = s_{\rm tot}/n \approx 1$ (`periodic_arclength`), so that point
   $n$ is the image of point 0. The total arc length per period,
   $s_{\rm tot}/L \ge 1$, measures how far the wall is from a straight
   line.
3. **Univalued profile** (Eq. 9 of the paper, `univalued_profile`). At
   each column $x+\tfrac12$ the curve crosses $m(x)$ times (an odd
   number) at heights $u_n(x)$, and

```math
u(x) = \frac{1}{m(x)}\sum_{n=1}^{m(x)} u_n(x).
```

   A column has an overhang if $m(x)\ge 3$. From this,
   $\rho_{\rm cols} = \langle\, \mathbb 1[m(x)\ge3]\,\rangle_x$ is the
   fraction of multivalued columns, and a sample counts as "having
   overhangs" if it has at least one.

## 4. Roughness observables

### 4.1 Global, on $u(x)$

With $\delta u = u - \bar u$ and $q_k = 2\pi k/L$ ($k = 1..L/2$):

```math
S(q_k) = \frac{1}{L}\left|\sum_x \delta u(x)\,e^{-iq_kx}\right|^2,\qquad
B(r) = \langle[u(x+r)-u(x)]^2\rangle_x = 2[C(0)-C(r)],\qquad
W^2 = \langle\delta u^2\rangle_x,
```

where $C(r)$ is the circular autocorrelation, computed by FFT. For a
self-affine interface with exponent $\zeta$:

```math
S(q)\sim q^{-(1+2\zeta)},\qquad B(r)\sim r^{2\zeta},\qquad W^2\sim L^{2\zeta}.
```

**Limitation for $\zeta>1$** (super-roughness, like qEW with
$\zeta\approx1.25$). Since $|u(x+r)-u(x)| \le r\max|u'|$, $B(r)$ can't
grow faster than $r^2$: for $\zeta > 1$ the increments are dominated by
the local slope and $B \sim r^{2\min(\zeta,1)}$. The same holds for the
whole-sample $W^2$. So the global $B$ and $W^2$ report $\zeta = 1$, and
among the global observables only $S(q)$ sees $\zeta>1$.

### 4.2 Local, on rotated segments

The code takes every segment of arc length $l$ (with $l+1$ points,
starting every $l/2$ points along the period, `_segments`). Each segment
is rigidly rotated so that its tilt $\theta$ becomes horizontal. With
$(dx, dy)$ measured from the first point:

- **Chord** (`chord`): $\theta = \operatorname{atan2}(dy_{\rm end}, dx_{\rm end})$,
  the straight line between the ends.
- **Principal axis** (`pca`): $\theta = \tfrac12\operatorname{atan2}(2s_{xy},\, s_{xx}-s_{yy})$,
  with $s$ the covariances of the segment's points. It is the axis of
  least perpendicular spread.

```math
X_r(s) = \cos\theta\,dx + \sin\theta\,dy,\qquad h(s) = -\sin\theta\,dx + \cos\theta\,dy.
```

$h(s)$, the coordinate perpendicular to the axis as a function of arc
length $s$, is well defined even when the wall has overhangs. Averaging
over segments:

```math
W^2(l) = \left\langle \operatorname{var}_s h \right\rangle,\qquad
B(r,l) = \left\langle[h(s+r)-h(s)]^2\right\rangle_{s},\qquad
\theta^2(l) = \langle\theta^2\rangle .
```

In the self-affine regime $W^2(l)\sim l^{2\zeta}$ even for $\zeta>1$:
subtracting each segment's tilt removes the $\sim l^2$ component that
saturates $B$. Synthetic profiles with pure $\zeta = 1.25$ confirm it:
$\zeta_W$ reads between 1.20 and 1.27 for $7 \le l \lesssim L/6$, with
no upward bias at small scales. $B(r,l)$, instead, is still capped at
$\zeta\le1$, because the increments within a segment are also dominated
by the local slope; on the data it reads ≈ 0.8.

**Local structure factor.** With $\delta h = h - \bar h$:

```math
\text{chord (FFT):}\quad S(q,l) = \frac{ds}{l}\left|\sum_s \delta h(s)e^{-iqs}\right|^2,\quad q = \frac{2\pi k}{l};
```

```math
\text{principal axis (DCT-II):}\quad S(q,l) = \frac{ds}{2l}\,\big[\mathrm{DCT}(\delta h)_k\big]^2,\quad q = \frac{\pi k}{l}.
```

The DCT is the FFT of the even extension of $h$, of period $2l$. **Why
each transform:** the FFT assumes the segment is periodic.

- If $h(0)\ne h(l)$, as happens with the principal axis, the jump at the
  boundary gives coefficients $\sim 1/q$ and $S\sim q^{-2}$, i.e. an
  apparent $\zeta = 0.5$.
- With the chord, $h(0)=h(l)=0$ by construction, so there is no jump.
- The even extension has no jump either, only a kink in the slope,
  which gives $\sim 1/q^2$ and $S\sim q^{-4}$, enough to measure
  $\zeta < 1.5$.

### 4.3 Multivalued fraction

A segment is still multivalued in its own frame if its along-axis
coordinate backtracks by more than one lattice spacing:

```math
f_{\rm mv}(l) = \Big\langle \mathbb 1\Big[\max_s\big(\max_{s'\le s}X_r(s') - X_r(s)\big) > 1\Big]\Big\rangle_{\rm segments}.
```

It is $0$ for a self-affine wall (at scale $l$ it is a graph in its own
frame) and tends to $1$ once overhangs dominate that scale.

## 5. Local exponents and crossover length

### 5.1 Scale-dependent exponent

For a curve $y(x)$ on log-log scales, the local slope is that of the
least-squares line of $\log y$ against $\log x$, through the points
within $\pm0.15$ decades of each $x$ (`local_slope`):

```math
\zeta_W(l) = \tfrac12\,\frac{d\ln W^2}{d\ln l},\qquad
\zeta_S^{\rm glob}(\ell = 2\pi/q) = -\tfrac12\left(\frac{d\ln S}{d\ln q}+1\right).
```

The global $S$ is first averaged in logarithmic bins (12 per decade).
From the local spectrum, $S(q,l)\sim q^{-(1+2\zeta)}$ is fitted over
$k=2..8$ (FFT) or $k=4..16$ (DCT), i.e. over wavelengths $l/8..l/2$ in
both cases, and $B(r,l)\sim r^{2\zeta}$ over $r\in[l/16,\, l/4]$.

### 5.2 Definitions of $l_o$

$l_o(L,\Delta)$ is the first scale, interpolated in $\ln l$, where:

| definition | criterion |
|---|---|
| `zW_chord`, `zW_pca` | $\zeta_W(l)$ drops below $\zeta_{\rm th} = 1.125$ (for $l\ge 8$) |
| `zS_glob` | $\zeta_S^{\rm glob}$ drops below $\zeta_{\rm th}$ |
| `fmv_chord`, `fmv_pca` | $f_{\rm mv}(l)$ rises above $0.05$ |

The threshold on $f_{\rm mv}$ is low because $f_{\rm mv}$ only
approaches 1 for $L \gg l_o$.

### 5.3 Data collapse

At fixed $L$, if $g(l) = G(l/l_o(\Delta))$ for a common function $G$,
shifting the curves in $\ln l$ by $a_\Delta = \ln[l_o(\Delta)/l_o(\Delta_0)]$
makes them all fall on top of each other. `collapse_shifts` minimises

```math
\mathcal C(\{a\}) = \frac{1}{N_{\rm pairs}}\sum_{i<j}\big\langle [g_i(\ln l + a_i) - g_j(\ln l + a_j)]^2\big\rangle_{\rm overlap},
```

with $a_{\Delta_0} = 0$ (Nelder–Mead). A pair of curves overlapping by
less than 0.3 in $\ln l$ is penalised. For $f_{\rm mv}$ only the part
of each curve with $f_{\rm mv}\ge0.01$ is used, since a curve that is
identically zero collapses onto anything.

### 5.4 Exponent $\alpha$ and the finite-size cut

```math
l_o \sim \Delta^{-\alpha}:\qquad \ln l_o = c - \alpha\ln\Delta \quad\text{(least squares)}.
```

The paper gives $\alpha \approx 1/0.45 \approx 2.2$.

**Finite size.** Even without a crossover, $\zeta_W(l)$ drops near
$l\sim L/2$, because a segment comparable to the sample is constrained
by periodicity. Comparing the two sizes, $l_o$ agrees up to
$\approx L/4$ ($\Delta=0.7$: 240 vs 237) and is overestimated beyond
($\Delta=0.6$: 365 vs 274, at $0.36L$). Hence (`--lo-max-frac 0.25`):

- the $\alpha$ fits leave out the $\Delta$ values with $l_o > L/4$,
  decided on the full sample;
- the collapses leave out the same $\Delta$ values (by the $f_{\rm mv}$
  $l_o$) and only use scales $l\le L/4$.

### 5.5 Errors

Bootstrap over samples: the $n$ samples are resampled with replacement
and every average, curve, $l_o$, collapse and fit is recomputed. The
error of a quantity is the standard deviation over resamples. This
method is used because neighbouring points in $l$ are correlated, as
they come from the same overlapping segments, so a fit's $\chi^2$
doesn't give reliable errors.

## 6. Two-crossover model for $\zeta_W(l)$

`scripts/fit_crossovers.py`.

### 6.1 Local exponent

```math
\zeta(l) = \zeta_L + (\zeta_q-\zeta_L)\,\sigma\!\left(\frac{\ln(l/L_c)}{w_1}\right) + (\zeta_\infty-\zeta_q)\,\sigma\!\left(\frac{\ln(l/l_o)}{w_2}\right),
\qquad \sigma(x) = \frac{1}{1+e^{-x}},
```

```math
L_c = a\,\Delta^{-x_c},\qquad l_o = b\,\Delta^{-\alpha}.
```

Limits:

- $\zeta\to\zeta_L$ (Larkin) for $l\ll L_c$;
- $\zeta\approx\zeta_q$ (qEW) for $L_c\ll l\ll l_o$;
- $\zeta\to\zeta_\infty$ for $l\gg l_o$.

$L_c$ and $l_o$ are the centres of the crossovers; $w_1$ and $w_2$ their
widths in $\ln l$.

### 6.2 Equivalent form for $W^2(l)$

Since $\zeta = \tfrac12\,d\ln W^2/d\ln l$, integrating gives

```math
W^2(l) = A\,l^{2\zeta_L}\,\Big[1+(l/L_c)^{n_1}\Big]^{2(\zeta_q-\zeta_L)/n_1}\,\Big[1+(l/l_o)^{n_2}\Big]^{2(\zeta_\infty-\zeta_q)/n_2},\qquad n_i = 1/w_i,
```

because

```math
\frac{d}{d\ln l}\ln\!\Big[1+(l/\ell)^n\Big] = \frac{(l/\ell)^n}{1+(l/\ell)^n} = \sigma\big(n\ln(l/\ell)\big).
```

Each factor $[1+(l/\ell)^n]^{2\delta\zeta/n}$ changes the exponent of
$W^2$ by $2\delta\zeta$ around $l=\ell$, and $n$ measures how sharp the
change is. This is the standard form of a crossover between two power
laws.

### 6.3 Why no plateau is seen

A logistic step goes from 10 % to 90 % over
$\Delta\ln l = 2\ln 9\cdot w \approx 4.4\,w$, i.e. $\approx 1.9\,w$
decades. For $\zeta(l)$ to get close to $\zeta_q$ both tails have to be
saturated, and to see a plateau $D$ decades wide,

```math
\log_{10}\frac{l_o}{L_c} \gtrsim 0.95\,(w_1 + w_2) + D .
```

From the fit, $l_o/L_c = (b/a)\,\Delta^{-(\alpha - x_c)} \approx 5\,\Delta^{-1.43}$.
That is ≈ 5 at $\Delta=1$ and ≈ 19 at $\Delta=0.4$, whereas merely
reaching $\zeta_q$ needs $\gtrsim 30$ ($\Delta\approx0.28$), and a
one-decade plateau $\gtrsim 300$ ($\Delta\approx0.057$).

### 6.4 Fit

- **Data:** every $(L,\Delta)$ curve at once, with $5\le l\le L/4$.
- **Weights:** $\sigma_i$ is each point's bootstrap error, floored at
  0.01.
- **Method:** least squares on $(\zeta_{\rm model}-\zeta_W)/\sigma$
  (`scipy.optimize.least_squares`), with $w\in[0.05,5]$,
  $x_c\in[0,5]$ and $\alpha\in[0,6]$.
- **Variants:** all free; $x_c = 2/3$ fixed; $x_c$ and $\alpha=2.1$
  fixed. $\zeta_L = 1.5$ is fixed unless `--free-zeta-L`.
- **Errors:** by bootstrap, refitting the resampled curves.

**Larkin collapse.** The part with $\zeta_W\ge1.3$ is collapsed by
shifts in $\ln l$ (§5.3), and the shifts are fitted to $\Delta^{-x_c}$.

### 6.5 Dependence on the step shape

$\zeta_q$ is not observed but extrapolated from the two crossovers, so
it depends on the assumed shape of the step $s(x)$, $x = \ln(l/\ell)$.
`--forms` fits four shapes:

| form | $s(x)$ | approach to the limits |
|---|---|---|
| `logistic` | $1/(1+e^{-x/w})$ | power law $(l/\ell)^{\pm 1/w}$, same on both sides |
| `erf` | $\tfrac12[1+\operatorname{erf}(x/w)]$ | Gaussian in $\ln l$ |
| `asym` | $(1+\nu e^{-x/w})^{-1/\nu}$ (Richards) | power law, rate $1/(w\nu)$ below and $1/w$ above |
| `sum` | no free width: $1/W^2 = \sum_i (l/\ell_i)^{-2\zeta_i}$ | $\zeta(l) = \sum_i\zeta_i t_i/\sum_i t_i$ ($t_i$ the terms), a logistic of fixed width $w = 1/(2|\delta\zeta|)$ |

In `sum`, each regime is an independent contribution to $W^2$ and the
one with the smallest $W^2$ dominates. Its widths are fixed by the jumps
in $\zeta$: $w_1 = 1/(2\cdot0.3)\approx1.7$ and
$w_2 = 1/(2\cdot0.15)\approx3.3$, much broader than what the free-width
shapes find.

Results ($\zeta_L = 1.5$; bootstrap errors; $\chi^2/\text{dof}$ only
relative):

| form | all free | $x_c = 2/3$ | $x_c = 2/3$, $\alpha = 2.1$ | $\chi^2/\text{dof}$ |
|---|---|---|---|---|
| logistic | 1.30 ± 0.06 | 1.20 ± 0.03 | 1.231 ± 0.011 | 0.73–0.79 |
| erf | 1.27 ± 0.05 | 1.21 ± 0.03 | 1.235 ± 0.011 | 0.74–0.78 |
| asym | 1.29 ± 0.06 | 1.25 ± 0.04 | 1.243 ± 0.012 | 0.72 |
| sum | 1.11 ± 0.02 | 1.08 ± 0.01 | 1.170 ± 0.013 | 0.97–1.25 |

With $\zeta_L$ free as well it fits to 1.45–1.47, and $\zeta_q$ becomes
1.24–1.25 (± 0.01) with $x_c$ (and $\alpha$) fixed, and 1.30–1.32
(± 0.06) with everything free.

- **The free-width shapes agree.** At fixed variant, logistic, erf and
  asym differ by $\lesssim0.05$ in $\zeta_q$, with the same $\chi^2$.
  The step shape is not the main systematic.
- **What is held fixed is.** With $x_c$ and $\alpha$ fixed $\zeta_q$ =
  1.23–1.24. Freeing $x_c$ moves it to 1.27–1.30, with $x_c\to0$–0.3,
  far from the Larkin 2/3; freeing only $\alpha$, to 1.20–1.25, with
  $\alpha$ = 2.0–2.7 ± 0.7–0.9, compatible with 2.1.
- **`sum` is disfavoured.** It has a higher $\chi^2$, and to follow the
  data it pushes parameters to unphysical values: $\alpha$ at its bound
  (6) and $x_c\to0$. The data need crossovers sharper than independent
  additive contributions give.
- **$\zeta_q = 1.25$ is within reach.** With free width, $\zeta_q$ ranges
  from 1.20 to 1.30 over shapes and variants. The best-constrained fits
  (fixed $x_c$, $\alpha$) sit 1–2σ below 1.25, and on it with free
  $\zeta_L$.

These numbers are from the current solver (`data/sweep_v2`). With the
legacy critical configurations (§8) the same fits gave $\zeta_q$ ≈ 0.03
lower (1.203–1.214 with $x_c$, $\alpha$ fixed; 1.14–1.21 overall): the
unrelaxed walls biased it down.

## 7. Larkin scaling

**Effective elasticity.** For slow displacements the wall is an elastic
line $u(x)$ with surface tension $\gamma\sim\sqrt{c\,\epsilon_0}$. The
random-bond disorder couples to it through the energy

```math
V(x,u) = -\frac{\epsilon_0}{2}\int dy\; r(x,y)\,\phi_0^2(y-u).
```

Its correlator $R(u)$ has range $\xi$ in $u$ and amplitude
$\propto \epsilon_0^2\langle r^2\rangle\propto\Delta^2$.

**Perturbation theory (Larkin).** A line of length $l$ is deformed by
the accumulated random force:

```math
\overline{u^2}(l) \sim \frac{|R''(0)|}{\gamma^2}\,l^{\,4-d}\quad (d=1:\; l^3)
\;\Rightarrow\; \zeta_L = \frac{4-d}{2} = \frac32 .
```

**Larkin length.** The scale at which the deformation reaches the wall
width, $u(L_c)\sim\xi$:

```math
L_c \sim \left(\frac{\gamma^2\,\xi^2}{|R''(0)|}\right)^{1/3}\propto\Delta^{-2/3},
\qquad |R''(0)|\sim R(0)/\xi^2\propto\Delta^2.
```

**Critical field.** The field's force per unit length on the wall is
$2h$ (the jump of $\phi$ is 2). Balancing it against the elastic force
at $L_c$:

```math
2h_d \sim \frac{\gamma\,\xi}{L_c^2}\;\Rightarrow\; h_d\propto\Delta^{4/3}.
```

Above $L_c$ the qEW regime starts ($\zeta\approx1.25$ in $d=1$), up to
$l_o$.

## 8. Results ($L=1024$ and $2048$; $\Delta=0.4$–$1.0$; 30 samples)

From `data/sweep_v2` (current solver, §2.2–2.3). The legacy sweep
(`data/sweep`) left 13–43 % of the critical configurations still moving:
relaxing them at the same $h$ shifted the wall by up to ≈ 360 rows. Its
$h_d$, $l_o$, $\alpha$ and $x_c$ agree with these within errors; its
$\zeta_q$ was ≈ 0.03 lower.

| quantity | result |
|---|---|
| $h_d$ | $\propto\Delta^{1.340\pm0.007}$ ($L=1024$), $\Delta^{1.339\pm0.005}$ ($L=2048$); values agree between $L$ to ≲ 0.8 % |
| $l_o$ from $f_{\rm mv}$, chord ($L=2048$) | 504, 280, 233, 140, 102 for $\Delta$ = 0.5, 0.6, 0.7, 0.85, 1.0; independent of $L$ for $\Delta\ge0.7$ |
| $\alpha$ ($L=2048$, $l_o\le L/4$) | $f_{\rm mv}$: 2.25 ± 0.17 (chord), 2.04 ± 0.17 (principal axis); collapses: 2.20 ± 0.43, 2.11 ± 0.11 |
| Larkin collapse | $x_c = 0.62\pm0.05$ ($L=2048$), $0.88\pm0.11$ ($L=1024$); compatible with 2/3 |
| two-crossover fit | free $\zeta_L$: 1.45–1.47; $\zeta_q = 1.231\pm0.011$ with $x_c$, $\alpha$ fixed; 1.20–1.30 over step shapes and variants; 1.24–1.25 with $\zeta_L$ free (§6.5) |
| $l_o$ from $\zeta_W$ or global $S$ | unreliable ($\alpha$ between −3.3 and 1.7): the 1.125 threshold falls where the two crossovers overlap |

**Reading.**

- **Larkin, confirmed.** The ~1.45 at small scales is physical: it
  doesn't appear on synthetic $\zeta = 1.25$ profiles, it scales with
  $\Delta^{-2/3}$, and the fit with free $\zeta_L$ gives 1.45–1.47.
- **$\zeta_q$, compatible with 1.25.** The best-constrained fits give
  1.23–1.24 ± 0.01, and 1.24–1.25 when $\zeta_L$ is also free. The step
  shape changes it by $\lesssim0.05$. What remains open is how much it
  depends on which of $x_c$ and $\alpha$ are fixed. All of it also rests
  on extrapolating through a window where the plateau is never reached.
- **The qEW window is too narrow.** With $l_o/L_c\approx5$–19 it can't be
  resolved. Per §6.3, merely reaching $\zeta_q$ needs $\Delta\lesssim0.3$,
  and a plateau needs a much smaller $\Delta$.
