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

Scalar field $\phi(x,y)$ on an $L\times L$ lattice:

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

**Boundaries.** Periodic in $x$, $\phi(x+L,y) = \phi(x,y)$;
anti-periodic in $y$, $\phi(x,y+L) = -\phi(x,y)$. With an anti-periodic
boundary any configuration has an odd number of walls, so there is
always a wall spanning the sample. For $h>0$ the $\phi>0$ region lies
below the wall and grows as $h$ increases.

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
  then the odd ones (`vmcop`, two colours per sweep). Since a site's
  four neighbours are all of the other colour, each half-update is
  exact.
- **Residual velocity.** Every `chop` = 100 sweeps the code computes

```math
v = \frac{1}{L}\sum_{i=1}^{L^2}\partial_t\phi_i
```

  (`velocity_phi`). It is normalised by $L$, not $L^2$: since $\phi$
  jumps by 2 across the wall, a wall of length $L$ moving at mean
  velocity $\bar v_{\rm wall}$ gives
  $\sum_i\partial_t\phi_i\approx 2L\,\bar v_{\rm wall}$, so
  $v\approx 2\bar v_{\rm wall}$ whatever the size.
- **Stopping.** Relaxation stops when $|v| <$ `TOLVEL` $= 10^{-4}$ (a
  metastable state), or after `MAXITERATIONS` $= 5000$ blocks
  ($5\times10^5$ sweeps).

### 2.3 Depinning field by bisection

For each trial $h$ (`find_depinning_field`):

1. **Flat initial condition:** $\phi = +1$ for $y < 0.2L$ and $\phi=-1$
   for $y \ge 0.2L$.
2. **Relaxation** with the rule above, with an early exit: if the wall
   crosses the row $y_e = 0.95L$, i.e. if $\max_x \phi(x, y_e) > 0$
   (`magnetization_lane`, `ESCAPE_LANE`), it stops there.
3. **Classification:** the trial is *depinned* if
   $\max_x\phi(x,y_e) > 0$, and *pinned* otherwise.

Bisection on $[a,b]$ replaces $a$ (pinned) or $b$ (depinned) by the
midpoint until $b-a \le$ `tol` $= 10^{-4}$. It then steps back,
$a \leftarrow a - \text{tol}$, until the trial is pinned. This defines

```math
h_d \equiv a, \qquad \text{critical configuration} = \text{metastable state at } h_d.
```

**Why restarting from the flat wall is valid.** By Middleton's
no-passing theorems the dynamics preserves order: if $\phi \le \phi'$
at every site, it stays so. Hence, at fixed $h$, whether the wall gets
pinned or escapes does not depend on the path. Concretely, the VMC rule
is monotonic: the roots of the cubic increase with $q$, i.e. with the
neighbours and with $h$, and the threshold $\phi_2$ decreases.

**Why the early exit doesn't change $h_d$.** With the anti-periodic
boundary, the magnetisation $M = \sum\phi$ grows as the wall moves up
and is largest with the wall at $y = L$. A depinned wall advances to
there and converges, instead of circulating. Stopping once it crosses
$y_e$ doesn't change the classification, since the decision is made on
that row anyway. Pinned trials never cross $y_e$, so they relax exactly
as before and the saved critical configuration is unchanged. Checked on
13 seeds: same $h_d$, and at $L=1024$ the total time dropped by 32 %.

**Initial bracket.** $[0,\, 3h_{\rm est}]$, with
$h_{\rm est} = 0.008\,(\Delta/0.2)^{4/3}$ (Larkin scaling, §7).

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

From the fit, $l_o/L_c = (b/a)\,\Delta^{-(\alpha - x_c)} \approx 4\,\Delta^{-1.43}$.
That is ≈ 4 at $\Delta=1$ and ≈ 15 at $\Delta=0.4$, whereas merely
reaching $\zeta_q$ needs $\gtrsim 25$, and a one-decade plateau
$\gtrsim 250$ ($\Delta\approx0.055$).

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

| quantity | result |
|---|---|
| $h_d$ | $\propto\Delta^{1.343\pm0.007}$ ($L=1024$), $\Delta^{1.342\pm0.005}$ ($L=2048$); values agree between $L$ to ≲ 0.8 % |
| $l_o$ from $f_{\rm mv}$, chord ($L=2048$) | 513, 274, 237, 136, 106 for $\Delta$ = 0.5, 0.6, 0.7, 0.85, 1.0; independent of $L$ for $\Delta\ge0.7$ |
| $\alpha$ ($L=2048$, $l_o\le L/4$) | $f_{\rm mv}$: 1.98 ± 0.15 (chord), 2.20 ± 0.23 (principal axis); collapses: 2.06 ± 0.12, 2.23 ± 0.14 |
| Larkin collapse | $x_c = 0.65\pm0.06$ ($L=2048$), $0.83\pm0.09$ ($L=1024$); compatible with 2/3 |
| two-crossover fit | free $\zeta_L$: 1.47–1.51; $\zeta_q = 1.203\pm0.014$ with $x_c$, $\alpha$ fixed; between 1.11 and 1.22 depending on the variant |
| $l_o$ from $\zeta_W$ or global $S$ | unreliable ($\alpha$ between −2.7 and 2.0): the 1.125 threshold falls where the two crossovers overlap |

**Reading.**

- **Larkin, confirmed.** The ~1.45 at small scales is physical: it
  doesn't appear on synthetic $\zeta = 1.25$ profiles, it scales with
  $\Delta^{-2/3}$, and the fit with free $\zeta_L$ gives 1.5.
- **$\zeta_q$, not resolved.** The Larkin collapse keeps working below
  1.25, down to ≈ 1.2, and the fit gives $\zeta_q\approx1.20$. The
  systematic error from the model's shape is ~0.05–0.1, so there is no
  significant difference from 1.25.
- **The qEW window is too narrow.** With $l_o/L_c\approx4$–15 it can't be
  resolved; per §6.3, it would take a much smaller $\Delta$.
