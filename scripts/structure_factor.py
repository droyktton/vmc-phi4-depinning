#!/usr/bin/env python3
"""Sample-averaged structure factor S(q) of the critical (depinning)
domain-wall configurations produced by ./phi4vmc, i.e. the
critica_h*_seed*.dat files written into a run_batch.py output directory.

For each sample: the (possibly multivalued, due to overhangs) wall
points (x, y) are collapsed to a single-valued height profile u(x) by
averaging y over repeated x (equivalent to gnuplot's "smooth unique",
used by the original analysis scripts); any x with no wall crossing is
filled in by periodic linear interpolation from its neighbors. Then
S(q) = |FFT(u)(q)|^2 for q = 2*pi*k/L, k = 0..L/2-1 (unnormalized, same
convention as scripts/fft1block.m). S(q) is then averaged over samples.

Example:
    ./structure_factor.py data/L128_D0.2 --L 128
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wall_io import read_wall_points, height_profile


def structure_factor(u):
    """S(q) = |FFT(u)|^2 for q=2*pi*k/L, k=0..L/2-1 (unnormalized)."""
    L = len(u)
    u = u - u.mean()  # only removes the q=0 term; all q>0 values are unaffected
    spectrum = np.fft.fft(u)
    k = np.arange(L // 2)
    q = 2 * np.pi * k / L
    return q, np.abs(spectrum[:L // 2]) ** 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datadir", help="directory with critica_h*_seed*.dat files (a run_batch.py --outdir)")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--out-prefix", default=None,
                    help="output files are <prefix>.dat and .png (default: <datadir>/structure_factor)")
    args = p.parse_args()

    datadir = Path(args.datadir)
    files = sorted(glob.glob(str(datadir / "critica_h*_seed*.dat")))
    if not files:
        raise SystemExit(f"no critica_h*_seed*.dat files found in {datadir}")

    all_Sq = []
    q = None
    for path in files:
        xs, ys = read_wall_points(path)
        u = height_profile(xs, ys, args.L)
        q, Sq = structure_factor(u)
        all_Sq.append(Sq)

    all_Sq = np.array(all_Sq)
    mean_Sq = all_Sq.mean(axis=0)
    sem_Sq = all_Sq.std(axis=0, ddof=1) / np.sqrt(len(files))

    print(f"{len(files)} samples, L={args.L}")

    prefix = Path(args.out_prefix) if args.out_prefix else datadir / "structure_factor"

    dat_path = prefix.with_suffix(".dat")
    with open(dat_path, "w") as f:
        f.write("# q S_mean S_sem\n")
        for qi, si, ei in zip(q, mean_Sq, sem_Sq):
            f.write(f"{qi:.6g} {si:.6g} {ei:.6g}\n")
    print(f"data written to {dat_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(q[1:], mean_Sq[1:], yerr=sem_Sq[1:], fmt="o", ms=3, color="#4c72b0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$S(q)$")
    ax.set_title(f"sample-averaged structure factor (n={len(files)}, L={args.L})")
    fig.tight_layout()
    png_path = prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    print(f"plot written to {png_path}")


if __name__ == "__main__":
    main()
