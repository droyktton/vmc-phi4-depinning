"""Domain wall of a critical configuration as an ordered curve, from the
raw field dumped by `./phi4vmc ... --dump-field`
(criticafield_h<h_d>_seed<seed>.bin: L*L float32, row-major, index x+y*L).

The wall-point files (critica_*.dat) can't be used for this: their
detector (is_wall_smooth) only looks for sign changes along y, so
vertical stretches of the wall -- i.e. the overhangs -- are missing,
and the points are unordered. Here the wall is instead the phi=0 level
line, traced with sub-pixel marching squares, which gives an ordered
curve whatever its shape. Library module, used by local_roughness.py.
"""
import re

import numpy as np
from skimage import measure

FIELD_RE = re.compile(r"criticafield_h([-\d.eE+]+)_seed(\d+)\.bin$")


def parse_field_filename(path):
    """Returns (h_d, seed) parsed from a criticafield_h<h_d>_seed<seed>.bin path."""
    m = FIELD_RE.search(str(path))
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))


def read_field(path):
    """Returns phi as an (L, L) array indexed [y, x]."""
    phi = np.fromfile(path, dtype=np.float32)
    L = int(round(np.sqrt(phi.size)))
    if L * L != phi.size:
        raise ValueError(f"{path}: {phi.size} values is not a square field")
    return phi.reshape(L, L)


def wall_curve(phi, edge_margin=3):
    """The domain wall as an ordered open curve over exactly one period
    in x: an (N, 2) array of (x, y) with the last point equal to the
    first shifted by (L, 0), so the curve tiles periodically.

    phi is tiled 3x along x (periodic) and its phi=0 level lines are
    traced; the wall is the longest one spanning the whole tile that
    stays clear of the y edges (the seam rows at y~0 and y~L-1 carry
    their own flat sign change, which is not the wall). Detached
    bubbles (closed contours) are ignored; their number is returned.
    Returns (curve, n_bubbles).
    """
    L = phi.shape[1]
    tiled = np.concatenate([phi, phi, phi], axis=1)
    contours = measure.find_contours(tiled, 0.0)

    wall, n_bubbles = None, 0
    for c in contours:
        y, x = c[:, 0], c[:, 1]
        if np.allclose(c[0], c[-1]):
            n_bubbles += 1
            continue
        if y.min() < edge_margin or y.max() > L - 1 - edge_margin:
            continue
        if x.min() > 0.5 or x.max() < 3 * L - 1.5:
            continue  # doesn't span the tile
        if wall is None or len(c) > len(wall):
            wall = c
    if wall is None:
        raise ValueError("no spanning wall contour found")

    xy = wall[:, ::-1]
    if xy[0, 0] > xy[-1, 0]:
        xy = xy[::-1]
    # bubbles come in triplicate from the tiling
    n_bubbles //= 3

    # cut one period: from the first point at x >= L to its image at +L
    i0 = int(np.argmax(xy[:, 0] >= L))
    target = xy[i0] + np.array([L, 0.0])
    i1 = i0 + 1 + int(np.argmin(np.sum((xy[i0 + 1:] - target) ** 2, axis=1)))
    if np.sum((xy[i1] - target) ** 2) > 1e-6:
        raise ValueError("wall contour isn't periodic in x")
    curve = xy[i0:i1 + 1] - np.array([L, 0.0])
    return curve, n_bubbles


def resample_arclength(curve, ds=1.0):
    """Resamples the curve at uniform arc-length spacing ds. Returns
    (points (n, 2), total arc length); the last point sits at the
    largest multiple of ds not exceeding the total length."""
    seg = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 0])
    s, pts = s[keep], curve[keep]
    s_new = np.arange(0.0, s[-1] + 1e-9, ds)
    out = np.column_stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])])
    return out, s[-1]


def periodic_arclength(curve, ds=1.0):
    """Arc-length resampling of one period, as n points with spacing ds
    such that point n would coincide with point 0 shifted by (L, 0):
    i.e. the period's arc length is rounded to n*ds, by rescaling ds
    by a factor of order 1/n. Returns (points, ds_used, L)."""
    L = curve[-1, 0] - curve[0, 0]
    seg = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 0])
    s, pts = s[keep], curve[keep]
    n = max(int(round(s[-1] / ds)), 2)
    ds_used = s[-1] / n
    s_new = np.arange(n) * ds_used
    out = np.column_stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])])
    return out, ds_used, L


def univalued_profile(curve, L):
    """The paper's univalued field (Eq. 9), from the curve: at each
    column x+0.5 (x=0..L-1), the mean y of all its crossings with the
    wall, u(x) = (1/m(x)) sum_n u_n(x). Returns (u, m) with m(x) the
    (odd, for a spanning wall) number of crossings.
    """
    x0 = curve[:, 0] % L
    shift = curve[0, 0] - x0[0]
    xa, xb = curve[:-1, 0] - shift, curve[1:, 0] - shift
    ya, yb = curve[:-1, 1], curve[1:, 1]
    lo, hi = np.minimum(xa, xb), np.maximum(xa, xb)

    # columns c+0.5 in [lo, hi) for each segment, unwrapped
    c_first = np.ceil(lo - 0.5)
    c_last = np.ceil(hi - 0.5) - 1
    counts = np.maximum(c_last - c_first + 1, 0).astype(int)
    idx = np.repeat(np.arange(len(xa)), counts)
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    c = c_first[idx] + offs
    xc = c + 0.5
    t = (xc - xa[idx]) / (xb[idx] - xa[idx])
    yc = ya[idx] + t * (yb[idx] - ya[idx])
    col = (c.astype(int)) % L

    m = np.bincount(col, minlength=L)
    sums = np.bincount(col, weights=yc, minlength=L)
    if (m == 0).any():
        raise ValueError("some column has no wall crossing")
    return sums / m, m
