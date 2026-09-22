#!/usr/bin/env python3
"""Detects multivalued (overhang) regions in the critical domain-wall
configuration(s) written by ./phi4vmc (critica_h*_seed*.dat), and
reports their importance relative to the system size L.

is_wall_smooth's 5-point smoothing always spreads one genuine crossing
over 2 (occasionally more) adjacent rows, so raw wall-point counts
alone would flag *every* column as "multivalued". Same-column
detections within --merge-gap rows (default 1) of each other are first
collapsed into a single branch; a column is then "multivalued" if it
has at least --threshold branches (default 2). The paper's original
tiene_overhang.awk used a threshold of 6 *raw* points, roughly
equivalent to --threshold 3 here.

For each sample this reports:
  - overhang_fraction: fraction of the L columns that are multivalued
    -- a per-sample estimate of the paper's overhang probability rho.
  - u_o: the paper's overhang-size measure -- sqrt of the mean, over x,
    of the within-column variance of the branches u_n(x). u_o/L is its
    size relative to the system.
  - regions: the individual contiguous multivalued stretches (as
    (x_start, length)), so you can see how many separate overhangs
    there are and how big each one is relative to L.

Run on a single file for a detailed per-region report, or on a
directory of samples (a run_batch.py --outdir) for a per-sample summary
CSV plus the sample-averaged overhang probability and size.

Examples:
    ./detect_overhangs.py data/L128_D0.2/critica_h0.008787_seed13.dat --L 128
    ./detect_overhangs.py data/L128_D0.2 --L 128
    ./detect_overhangs.py data/L128_D0.2 --L 128 --threshold 3
"""
import argparse
import glob
from pathlib import Path

import numpy as np

from wall_io import read_wall_points, overhang_stats, parse_filename


def print_detail(path, L, threshold, merge_gap):
    xs, ys = read_wall_points(path)
    stats = overhang_stats(xs, ys, L, threshold=threshold, merge_gap=merge_gap)

    print(Path(path).name)
    print(f"  overhang_fraction (m>={threshold}) = {stats['overhang_fraction']:.4f}"
          f"  ({int(stats['overhang_mask'].sum())}/{L} columns)")
    print(f"  u_o = {stats['u_o']:.4f}   u_o/L = {stats['u_o'] / L:.5f}")
    if stats["regions"]:
        print(f"  {len(stats['regions'])} overhang region(s), longest first:")
        for x_start, length in sorted(stats["regions"], key=lambda r: -r[1]):
            x_end = (x_start + length) % L
            print(f"    x in [{x_start}, {x_end}) (periodic), "
                  f"length = {length}, length/L = {length / L:.4f}")
    else:
        print("  no overhang regions found")
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="a single critica_*.dat file, or a directory of them")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--threshold", type=int, default=2,
                    help="minimum branches per column to count as an overhang (default: 2)")
    p.add_argument("--merge-gap", type=int, default=1,
                    help="same-column detections at most this many rows apart are merged into one branch (default: 1)")
    p.add_argument("--out-csv", default=None,
                    help="for a directory input: summary CSV path (default: <path>/overhang_summary.csv)")
    args = p.parse_args()

    path = Path(args.path)
    if path.is_file():
        print_detail(path, args.L, args.threshold, args.merge_gap)
        return

    files = sorted(glob.glob(str(path / "critica_h*_seed*.dat")))
    if not files:
        raise SystemExit(f"no critica_h*_seed*.dat files found in {path}")

    rows = []
    for f in files:
        hd, seed = parse_filename(f)
        xs, ys = read_wall_points(f)
        stats = overhang_stats(xs, ys, args.L, threshold=args.threshold, merge_gap=args.merge_gap)
        rows.append((seed, hd, stats["overhang_fraction"], stats["u_o"], len(stats["regions"])))
        print(f"seed {seed}: overhang_fraction={stats['overhang_fraction']:.4f}  "
              f"u_o={stats['u_o']:.4f}  n_regions={len(stats['regions'])}")

    out_csv = Path(args.out_csv) if args.out_csv else path / "overhang_summary.csv"
    with open(out_csv, "w") as fout:
        fout.write("seed,h_d,overhang_fraction,u_o,n_regions\n")
        for seed, hd, frac, u_o, n_regions in rows:
            fout.write(f"{seed},{hd if hd is not None else ''},{frac:.6g},{u_o:.6g},{n_regions}\n")

    fracs = np.array([r[2] for r in rows])
    u_os = np.array([r[3] for r in rows])
    n = len(rows)
    print()
    print(f"{n} samples, L={args.L}, threshold={args.threshold}")
    print(f"mean overhang_fraction (~ paper's rho) = {fracs.mean():.4f} +/- {fracs.std(ddof=1) / np.sqrt(n):.4f}")
    print(f"mean u_o = {u_os.mean():.4f} +/- {u_os.std(ddof=1) / np.sqrt(n):.4f}   (u_o/L = {u_os.mean() / args.L:.5f})")
    print(f"summary written to {out_csv}")


if __name__ == "__main__":
    main()
