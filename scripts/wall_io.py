"""Shared helpers for reading critica_h*_seed*.dat wall-configuration
files (written by System::print_wall) and turning them into a
single-valued height profile. Used by structure_factor.py and
plot_critical_configs.py.
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
