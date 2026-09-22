#!/usr/bin/env python3
"""Statistics (mean, variance, histogram) of the depinning field h_d over
the disorder samples collected by run_batch.py.

Example:
    ./analyze_hd.py data/L128_D0.2/hd_summary.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_hd(summary_csv):
    with open(summary_csv) as f:
        return np.array([float(row["h_d"]) for row in csv.DictReader(f)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summary_csv", help="hd_summary.csv written by run_batch.py")
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--out-prefix", default=None,
                    help="output files are <prefix>_histogram.dat and .png (default: alongside the input CSV)")
    args = p.parse_args()

    summary_csv = Path(args.summary_csv)
    hd = load_hd(summary_csv)
    n = len(hd)
    mean = hd.mean()
    var = hd.var(ddof=1)
    std = hd.std(ddof=1)
    sem = std / np.sqrt(n)

    print(f"n samples = {n}")
    print(f"mean h_d  = {mean:.6g}")
    print(f"variance  = {var:.6g}")
    print(f"std dev   = {std:.6g}")
    print(f"sem       = {sem:.6g}")

    prefix = Path(args.out_prefix) if args.out_prefix else summary_csv.with_name(summary_csv.stem + "_stats")

    counts, edges = np.histogram(hd, bins=args.bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_path = prefix.with_name(prefix.name + "_histogram.dat")
    with open(hist_path, "w") as f:
        f.write("# h_d_bin_center count\n")
        for c, n_c in zip(centers, counts):
            f.write(f"{c:.6g} {n_c}\n")
    print(f"histogram data written to {hist_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(hd, bins=args.bins, color="#4c72b0", edgecolor="white")
    ax.axvline(mean, color="#c44e52", linestyle="--", label=f"mean = {mean:.4g}")
    ax.set_xlabel(r"$h_d$")
    ax.set_ylabel("count")
    ax.set_title(f"depinning field distribution (n={n})")
    ax.legend()
    fig.tight_layout()
    png_path = prefix.with_name(prefix.name + "_histogram.png")
    fig.savefig(png_path, dpi=150)
    print(f"histogram plot written to {png_path}")


if __name__ == "__main__":
    main()
