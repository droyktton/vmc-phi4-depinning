#!/usr/bin/env python3
"""Plots the critical (depinning) domain-wall configuration of every
sample in a run_batch.py output directory, i.e. every critica_h*_seed*.dat
file.

Each plot shows the raw wall points (x, y) as dots -- a single x can
have more than one y where the interface has overhangs/pinch-off loops
-- overlaid with the single-valued height profile u(x) used for the
structure factor (see structure_factor.py), obtained by averaging y
over repeated x and periodically interpolating any x with no crossing.

By default, one PNG per sample is written to <datadir>/configs/. Pass
--grid to also save a single overview figure with all samples as small
subplots.

Example:
    ./plot_critical_configs.py data/L128_D0.2 --L 128 --grid
"""
import argparse
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wall_io import read_wall_points, height_profile, overhang_stats, parse_filename


def plot_one(ax, xs, ys, u, L, title=None, full_box=False, overhang_regions=None):
    ax.plot(xs, ys, ".", ms=3, color="#999999", label="wall points")
    ax.plot(range(L), u, "-", lw=1, color="#c44e52", label="u(x)")
    for x_start, length in overhang_regions or []:
        x_end = x_start + length
        if x_end <= L:
            ax.axvspan(x_start, x_end, color="#dd8452", alpha=0.25, lw=0)
        else:
            # region wraps around the periodic x boundary: draw both halves
            ax.axvspan(x_start, L, color="#dd8452", alpha=0.25, lw=0)
            ax.axvspan(0, x_end - L, color="#dd8452", alpha=0.25, lw=0)
    ax.set_xlim(0, L)
    if full_box:
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
    else:
        # zoom to the actual fluctuation range, so roughness is visible
        y_min, y_max = ys.min(), ys.max()
        pad = max(1.0, 0.1 * (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)
    if title:
        ax.set_title(title, fontsize=8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datadir", help="directory with critica_h*_seed*.dat files (a run_batch.py --outdir)")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--outdir", default=None, help="default: <datadir>/configs")
    p.add_argument("--grid", action="store_true", help="also save one combined overview figure")
    p.add_argument("--full-box", action="store_true",
                    help="show the full [0,L]x[0,L] box instead of zooming to the fluctuation range")
    p.add_argument("--highlight-overhangs", action="store_true",
                    help="shade multivalued (overhang) x-regions, see detect_overhangs.py")
    p.add_argument("--overhang-threshold", type=int, default=2,
                    help="minimum branches per column to count as an overhang (default: 2), see detect_overhangs.py")
    args = p.parse_args()

    datadir = Path(args.datadir)
    files = sorted(glob.glob(str(datadir / "critica_h*_seed*.dat")))
    if not files:
        raise SystemExit(f"no critica_h*_seed*.dat files found in {datadir}")

    outdir = Path(args.outdir) if args.outdir else datadir / "configs"
    outdir.mkdir(parents=True, exist_ok=True)

    samples = []
    for path in files:
        hd, seed = parse_filename(path)
        xs, ys = read_wall_points(path)
        u = height_profile(xs, ys, args.L)
        regions = overhang_stats(xs, ys, args.L, threshold=args.overhang_threshold)["regions"] \
            if args.highlight_overhangs else None
        samples.append((seed, hd, xs, ys, u, regions))

        fig, ax = plt.subplots(figsize=(4, 4))
        title = f"seed {seed}, h_d = {hd:.6g}" if hd is not None else Path(path).name
        plot_one(ax, xs, ys, u, args.L, title=title, full_box=args.full_box, overhang_regions=regions)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        out_path = outdir / f"critical_config_seed{seed}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"seed {seed}: wrote {out_path}")

    if args.grid:
        n = len(samples)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
        axes = axes.flatten() if n > 1 else [axes]
        for ax, (seed, hd, xs, ys, u, regions) in zip(axes, samples):
            plot_one(ax, xs, ys, u, args.L, title=f"seed {seed}", full_box=args.full_box, overhang_regions=regions)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout()
        grid_path = outdir / "all_critical_configs.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"overview grid written to {grid_path}")


if __name__ == "__main__":
    main()
