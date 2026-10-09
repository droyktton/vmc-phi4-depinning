"""Roughness estimators for one domain-wall configuration, global and
local-detrended, plus the scale-dependent helpers (local exponents,
threshold crossings, data collapse) used by local_roughness.py and
synthetic_profiles.py. Library module.

Global observables, on the univalued profile u(x), x=0..L-1, periodic
(paper Eq. 9 for walls with overhangs):
    S(q)  = |sum_x u(x) e^{-iqx}|^2 / L          ~ q^-(1+2 zeta)
    B(r)  = <[u(x+r) - u(x)]^2>_x                ~ r^(2 zeta)
    W2    = <u^2> - <u>^2                         ~ L^(2 zeta)
For a super-rough interface (zeta > 1, as the qEW zeta~1.25) B(r) and
W2 can't grow faster than r^2 and L^2: they report zeta=1 whatever
the true zeta is, only S(q) sees zeta > 1.

Local observables, on segments of arc length l of the wall, taken as an
ordered curve sampled at uniform arc length (wall_contour.py). Each
segment is rigidly rotated so that its tilt is horizontal, with the
tilt either the end-to-end chord ("chord") or the principal axis of
its points ("pca"); h(s) is then the coordinate perpendicular to that
axis, as a function of arc length s along the segment (single-valued
whatever the shape), and
    W2(l)   = <h^2> - <h>^2 over the segment
    B(r,l)  = <[h(s+r) - h(s)]^2>_s, r < l
    S(q,l)  = |sum_s h(s) e^{-iqs} ds|^2 / l, q = 2 pi k / l   (transform "fft")
            or the same for the mirror (even) extension of h over 2l,
            i.e. a cosine transform, q = pi k / l              (transform "dct")
all averaged over segments. Which transform: the FFT treats the segment
as periodic, so for pca (whose ends don't match, h(0) != h(l)) the jump
leaks as q^-2 and caps the measurable zeta at 0.5; the mirror extension
has no jump, only a slope kink (q^-4, zeta < 1.5), like chord + FFT
(h(0) = h(l) = 0). So: chord + "fft", pca + "dct" (synthetic_profiles.py
tests show both unbiased at zeta = 1.25 up to the short-wavelength
resampling bias common to every transform; a Hann window instead
overestimates zeta by 0.1-0.4). Also recorded per l: the tilt-angle
variance theta2(l) of the chord/principal axis, and the fraction
fmv(l) of segments that are still multivalued in their own rotated
frame (the along-axis coordinate backtracks by more than one lattice
spacing) -- 0 for a self-affine wall, -> 1 once overhangs dominate.
"""
import numpy as np
from scipy.fft import dct
from scipy.optimize import minimize


# ---------------------------------------------------------------- global

def global_observables(u):
    """S(q), B(r), W2 of a periodic univalued profile u (length L).
    Returns dict with q, S (k=1..L/2), r, B (r=1..L/2), W2."""
    L = len(u)
    du = u - u.mean()
    U = np.fft.fft(du)
    P = np.abs(U) ** 2
    k = np.arange(1, L // 2 + 1)
    C = np.fft.ifft(P).real / L
    r = np.arange(1, L // 2 + 1)
    return {
        "q": 2 * np.pi * k / L,
        "S": P[k] / L,
        "r": r.astype(float),
        "B": 2 * (C[0] - C[r]),
        "W2": float(np.mean(du ** 2)),
    }


# ----------------------------------------------------------------- local

def log_grid(lo, hi, n):
    """Unique integers log-spaced in [lo, hi]."""
    return np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), n)).astype(int))


def _segments(points, L, npts, stride):
    """All windows of npts consecutive points of the periodic curve
    (period: n points, translation (L, 0)), starting every `stride`
    points over one period. Returns X, Y of shape (nseg, npts)."""
    n = len(points)
    reps = int(np.ceil((n + npts) / n)) + 1
    ext = np.concatenate([points + np.array([i * L, 0.0]) for i in range(reps)])
    starts = np.arange(0, n, stride)
    idx = starts[:, None] + np.arange(npts)[None, :]
    return ext[idx, 0], ext[idx, 1]


def _rotate(X, Y, tilt):
    """Rotates each segment (rows of X, Y) so its tilt axis is
    horizontal. Returns along-axis Xr, perpendicular h, and the tilt
    angle theta (in (-pi/2, pi/2] w.r.t. the x axis for pca, (-pi, pi]
    for chord)."""
    dx = X - X[:, :1]
    dy = Y - Y[:, :1]
    if tilt == "chord":
        theta = np.arctan2(dy[:, -1], dx[:, -1])
    elif tilt == "pca":
        cx = dx - dx.mean(axis=1, keepdims=True)
        cy = dy - dy.mean(axis=1, keepdims=True)
        sxx, syy, sxy = (cx * cx).mean(1), (cy * cy).mean(1), (cx * cy).mean(1)
        theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    else:
        raise ValueError(f"unknown tilt {tilt!r}")
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    Xr = c * dx + s * dy
    h = -s * dx + c * dy
    if tilt == "pca":
        flip = Xr[:, -1] < Xr[:, 0]  # orient along the direction of travel
        Xr[flip] *= -1
        h[flip] *= -1
    return Xr, h, theta


def local_observables(points, ds, L, ls, rs, tilt="chord", transform="fft",
                      stride_frac=0.5, kmax=None, backtrack=1.0):
    """Local detrended observables of one wall (see module docstring).

    points: (n, 2) periodic arc-length sampling of the wall (spacing ds,
        period translation (L, 0)), from wall_contour.periodic_arclength.
    ls: segment arc lengths (in units of ds); rs: lags for B(r, l).
    transform: "fft" or "dct" for S(q, l) (see module docstring).
    Returns dict of arrays indexed by l: W2, theta2, fmv, and
    B (len(ls), len(rs)) [nan where r >= l], S (len(ls), kmax)
    [S at k = 1..kmax, q = 2 pi k / l (fft) or pi k / l (dct); nan
    beyond the segment's last mode].
    """
    n = len(points)
    nl = len(ls)
    kmax = kmax or int(max(ls)) // 2
    out = {
        "W2": np.full(nl, np.nan), "theta2": np.full(nl, np.nan), "fmv": np.full(nl, np.nan),
        "B": np.full((nl, len(rs)), np.nan), "S": np.full((nl, kmax), np.nan),
    }
    for i, l in enumerate(ls):
        npts = int(l) + 1
        if npts > n + 1:
            continue
        stride = max(1, int(l * stride_frac))
        X, Y = _segments(points, L, npts, stride)
        Xr, h, theta = _rotate(X, Y, tilt)

        out["W2"][i] = np.mean(h.var(axis=1))
        out["theta2"][i] = np.mean(theta ** 2)
        back = np.max(np.maximum.accumulate(Xr, axis=1) - Xr, axis=1)
        out["fmv"][i] = np.mean(back > backtrack)

        for j, r in enumerate(rs):
            if r >= npts - 1:
                break
            out["B"][i, j] = np.mean((h[:, r:] - h[:, :-r]) ** 2)

        hh = h - h.mean(axis=1, keepdims=True)
        if transform == "fft":
            P = np.abs(np.fft.rfft(hh, axis=1)) ** 2 * ds / l  # |sum h ds|^2 / (l ds)
        elif transform == "dct":
            # DCT-II = FFT coefficient of the even extension (period 2l)
            P = dct(hh, type=2, axis=1) ** 2 * ds / (2 * l)
        else:
            raise ValueError(f"unknown transform {transform!r}")
        Pm = P.mean(axis=0)[1:]
        m = min(len(Pm), kmax)
        out["S"][i, :m] = Pm[:m]
    return out


# ------------------------------------------------------------- analysis

def local_slope(x, y, half_width=0.15, min_points=3):
    """d ln y / d ln x at each x, by a least-squares line through the
    points within half_width decades of it. nan where too few points
    or non-positive y."""
    lx = np.log10(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        ly = np.where(y > 0, np.log10(y), np.nan)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        sel = (np.abs(lx - lx[i]) <= half_width) & np.isfinite(ly)
        if sel.sum() >= min_points:
            out[i] = np.polyfit(lx[sel], ly[sel], 1)[0]
    return out


def log_bin(x, y, bins_per_decade=10):
    """Averages y in logarithmic bins of x. Returns bin-mean x and y."""
    lx = np.log10(x)
    edges = np.arange(np.floor(lx.min() * bins_per_decade),
                      np.ceil(lx.max() * bins_per_decade) + 1) / bins_per_decade
    which = np.digitize(lx, edges)
    xs, ys = [], []
    for b in np.unique(which):
        sel = which == b
        xs.append(10 ** lx[sel].mean())
        ys.append(y[sel].mean())
    return np.array(xs), np.array(ys)


def fit_exponent(x, y, xmin, xmax):
    """Slope of ln y vs ln x over xmin <= x <= xmax (nan if < 3 points)."""
    sel = (x >= xmin) & (x <= xmax) & np.isfinite(y) & (y > 0)
    if sel.sum() < 3:
        return np.nan
    return np.polyfit(np.log(x[sel]), np.log(y[sel]), 1)[0]


def zeta_from_Sql(ls, S, kfit=(2, 8)):
    """zeta(l) from the local structure factor of segments of size l:
    S(q, l) ~ q^-(1+2 zeta) fitted over k = kfit[0]..kfit[1], i.e. over
    the scales l/kfit[1] .. l/kfit[0] of the segment."""
    k = np.arange(1, S.shape[1] + 1)
    out = np.full(len(ls), np.nan)
    for i, l in enumerate(ls):
        m = fit_exponent(k.astype(float), S[i], kfit[0], kfit[1])
        out[i] = -(m + 1) / 2
    return out


def zeta_from_Brl(ls, rs, B, rfit=(1 / 16, 1 / 4)):
    """zeta(l) from B(r, l) ~ r^(2 zeta) fitted over r in
    [rfit[0]*l, rfit[1]*l] within segments of size l."""
    out = np.full(len(ls), np.nan)
    for i, l in enumerate(ls):
        out[i] = fit_exponent(rs.astype(float), B[i], rfit[0] * l, rfit[1] * l) / 2
    return out


def crossing(x, y, level, direction="down"):
    """First x (log-interpolated) where y crosses `level`, going
    'down' (from above to below) or 'up'. nan if it never does."""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    above = y > level if direction == "down" else y < level
    for i in range(1, len(y)):
        if above[i - 1] and not above[i]:
            t = (level - y[i - 1]) / (y[i] - y[i - 1])
            return float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1]))))
    return np.nan


def collapse_shifts(curves, anchor=0):
    """Data collapse by horizontal shifts in ln x: finds a_i such that
    all y_i(x * exp(-a_i)) (curves: list of (x, y)) fall onto a common
    curve, with a_anchor = 0. Cost: mean squared difference between
    every pair of curves over their overlap (on a common grid in
    ln x). Returns (shifts a_i, i.e. l_o,i / l_o,anchor = exp(a_i), cost).
    """
    curves = [(np.log(np.asarray(x, float)), np.asarray(y, float)) for x, y in curves]
    curves = [(lx[np.isfinite(y)], y[np.isfinite(y)]) for lx, y in curves]
    free = [i for i in range(len(curves)) if i != anchor]

    def cost(a_free):
        a = np.zeros(len(curves))
        a[free] = a_free
        total, npair = 0.0, 0
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                xi, yi = curves[i][0] - a[i], curves[i][1]
                xj, yj = curves[j][0] - a[j], curves[j][1]
                lo, hi = max(xi.min(), xj.min()), min(xi.max(), xj.max())
                if hi - lo < 0.3:
                    total += 10.0  # penalise (almost) no overlap
                    npair += 1
                    continue
                g = np.linspace(lo, hi, 30)
                total += np.mean((np.interp(g, xi, yi) - np.interp(g, xj, yj)) ** 2)
                npair += 1
        return total / max(npair, 1)

    # initial guess: align the mid-range crossings of each curve
    a0 = []
    ya = curves[anchor][1]
    level = 0.5 * (np.nanmin(ya) + np.nanmax(ya))
    xa = crossing(np.exp(curves[anchor][0]), ya, level, "up" if ya[-1] > ya[0] else "down")
    for i in free:
        lx, y = curves[i]
        xc = crossing(np.exp(lx), y, level, "up" if y[-1] > y[0] else "down")
        a0.append(np.log(xc / xa) if np.isfinite(xc) and np.isfinite(xa) else 0.0)
    res = minimize(cost, np.array(a0), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 4000})
    a = np.zeros(len(curves))
    a[free] = res.x
    return a, float(res.fun)


def power_law_fit(x, y):
    """ln y = c - alpha ln x. Returns (alpha, c)."""
    sel = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if sel.sum() < 2:
        return np.nan, np.nan
    m, c = np.polyfit(np.log(x[sel]), np.log(y[sel]), 1)
    return -m, c
