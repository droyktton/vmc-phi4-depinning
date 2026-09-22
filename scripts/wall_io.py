"""Shared helpers for reading critica_h*_seed*.dat wall-configuration
files (written by System::print_wall), turning them into a
single-valued height profile, and detecting multivalued (overhang)
regions. Used by structure_factor.py, plot_critical_configs.py and
detect_overhangs.py.
"""
import re

import numpy as np

FILENAME_RE = re.compile(r"critica_h([-\d.eE+]+)_seed(\d+)\.dat$")


def parse_filename(path):
    """Returns (h_d, seed) parsed from a critica_h<h_d>_seed<seed>.dat path."""
    m = FILENAME_RE.search(str(path))
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))


def read_wall_points(path):
    """Returns arrays (x, y) of the wall coordinates in a critica_*.dat file."""
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            x, y = line.split()
            xs.append(int(x))
            ys.append(int(y))
    return np.array(xs), np.array(ys)


def height_profile(xs, ys, L):
    """Single-valued, evenly-sampled height profile u(x), x=0..L-1: y is
    averaged over repeated x (equivalent to gnuplot's "smooth unique"),
    and any x with no wall crossing is filled by periodic linear
    interpolation from its neighbors.
    """
    sums = np.zeros(L)
    counts = np.zeros(L)
    np.add.at(sums, xs, ys)
    np.add.at(counts, xs, 1)

    present = counts > 0
    if not present.any():
        raise ValueError("no wall points found")

    u_present = sums[present] / counts[present]
    x_present = np.nonzero(present)[0]

    if present.all():
        return u_present

    x_ext = np.concatenate([x_present - L, x_present, x_present + L])
    u_ext = np.concatenate([u_present, u_present, u_present])
    return np.interp(np.arange(L), x_ext, u_ext)


def _periodic_runs(mask):
    """Maximal contiguous True-runs of a circular boolean array (x is a
    periodic direction), as a list of (x_start, length) tuples.
    """
    L = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, L)]

    idx = np.nonzero(mask)[0]
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])  # inclusive
    runs = [(int(s), int(e - s + 1)) for s, e in zip(starts, ends)]

    # merge the run touching x=0 with the run touching x=L-1, if distinct
    if mask[0] and mask[-1] and len(runs) > 1:
        s_last, len_last = runs[-1]
        _, len_first = runs[0]
        runs = runs[1:-1] + [(s_last, len_last + len_first)]

    return runs


def _cluster_branches(col_y, merge_gap):
    """Collapses a column's sorted y-values into "branches": groups of
    values within merge_gap of their neighbor are one branch (its
    representative value is their mean). This is needed because
    is_wall_smooth's 5-point smoothing always spreads a single, genuine
    crossing over 2 (occasionally more) adjacent rows -- without this
    collapse, every ordinary column looks "multivalued".
    """
    col_y = np.sort(col_y)
    branches = [[col_y[0]]]
    for y in col_y[1:]:
        if y - branches[-1][-1] <= merge_gap:
            branches[-1].append(y)
        else:
            branches.append([y])
    return np.array([np.mean(b) for b in branches])


def overhang_stats(xs, ys, L, threshold=2, merge_gap=1):
    """Overhang diagnostics for one critical configuration.

    Raw wall detections at the same x that are within `merge_gap` rows
    of each other are first collapsed into a single "branch" (see
    _cluster_branches) -- is_wall_smooth's 5-point smoothing otherwise
    always reports 2 adjacent-row detections for an ordinary, single
    interface crossing, which would look like an overhang at every
    column. A column x is then "multivalued" (an overhang) if it has at
    least `threshold` branches (default 2; the paper's original
    tiene_overhang.awk used 6 *raw* detections, which is roughly
    equivalent to 3 branches here -- pass --threshold 3 to match it
    more closely).

    Returns a dict:
      multiplicity: array of length L, m(x) = number of branches at
        column x (0 if none).
      overhang_mask: boolean array, True where m(x) >= threshold.
      overhang_fraction: overhang_mask.mean(), the fraction of the L
        columns that are multivalued -- a per-sample estimate of the
        paper's overhang probability rho.
      u_o, u_o2: the paper's overhang-size measure,
        u_o^2 = (1/L) * sum_x [ <u_n(x)^2> - <u_n(x)>^2 ], i.e. the
        mean over x of the within-column variance of the branches
        u_n(x) (zero where m(x) <= 1). u_o/L gives its size relative
        to the system.
      regions: list of (x_start, length) for each maximal contiguous
        (periodic) run of columns with m(x) >= threshold -- the
        individual overhang/pinch-off stretches.
    """
    m = np.zeros(L, dtype=int)
    sum1 = np.zeros(L)
    sum2 = np.zeros(L)

    order = np.argsort(xs, kind="stable")
    xs_sorted, ys_sorted = xs[order], ys[order]
    unique_x, start_idx = np.unique(xs_sorted, return_index=True)
    start_idx = np.append(start_idx, len(xs_sorted))

    for i, x in enumerate(unique_x):
        branches = _cluster_branches(ys_sorted[start_idx[i]:start_idx[i + 1]], merge_gap)
        m[x] = len(branches)
        sum1[x] = branches.sum()
        sum2[x] = (branches ** 2).sum()

    present = m >= 1
    mean = np.divide(sum1, m, out=np.zeros(L), where=present)
    mean_sq = np.divide(sum2, m, out=np.zeros(L), where=present)
    variance = np.zeros(L)
    variance[present] = mean_sq[present] - mean[present] ** 2
    u_o2 = variance.sum() / L
    u_o = np.sqrt(max(u_o2, 0.0))

    overhang_mask = m >= threshold

    return {
        "multiplicity": m,
        "overhang_mask": overhang_mask,
        "overhang_fraction": float(overhang_mask.mean()),
        "u_o2": float(u_o2),
        "u_o": float(u_o),
        "regions": _periodic_runs(overhang_mask),
    }
