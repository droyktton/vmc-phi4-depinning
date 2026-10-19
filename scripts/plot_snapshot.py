#!/usr/bin/env python3
"""Snapshot of one critical configuration: the field phi around the wall
(full width, cropped in y) with the phi=0 wall traced on top, and a zoom
on a window of the wall at equal aspect ratio. Reads a dumped
criticafield_h*_seed*.bin (run_batch.py --dump-field) and the run's
logfile.dat for L and the disorder parameters.

Example:
    python3 scripts/plot_snapshot.py data/aniso_neg/ax1ay8_L1024_D0.2_m0 \
        --out results/aniso_neg/snapshot_ax1ay8_L1024_D0.2.png
"""
import argparse
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wall_contour import read_field, wall_curve, univalued_profile, parse_field_filename


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datadir")
    p.add_argument("--seed", type=int, default=None, help="default: the first dumped field")
    p.add_argument("--zoom", type=int, default=256, help="width of the zoomed window")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    log = {}
    for line in open(f"{a.datadir}/logfile.dat"):
        if "=" in line:
            k, v = line.split("=", 1)
            log[k.strip()] = v.strip()
    L = int(log["L"])
    fs = sorted(glob.glob(f"{a.datadir}/criticafield_h*_seed{a.seed if a.seed else '*'}.bin"))
    f = fs[0]
    hd, seed = parse_field_filename(f)
    phi = read_field(f, L)
    curve, nb = wall_curve(phi)
    u, m = univalued_profile(curve, L)
    w = u.std()
    arc = np.sum(np.hypot(*np.diff(curve, axis=0).T))
    frac_mv = np.mean(m > 1)

    pad = max(30, int(3 * w))
    y0, y1 = int(max(0, u.min() - pad)), int(min(phi.shape[0], u.max() + pad))
    title = (f"critical configuration, L = {L}, Δ = {log.get('AMPDIS (Delta)')}, "
             f"blocks {log.get('AX')}×{log.get('AY')}, seed {seed}, h_d = {hd:.6f}\n"
             f"w = {w:.1f}, arc length / L = {arc / L:.3f}, multivalued columns = {100 * frac_mv:.1f} %, bubbles = {nb}")

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0])
    ax.imshow(phi[y0:y1], origin="lower", extent=(0, L, y0, y1), cmap="RdBu_r", vmin=-1.2, vmax=1.2,
              aspect="auto", interpolation="nearest")
    ax.plot(curve[:, 0], curve[:, 1], "k-", lw=0.6)
    ax.set_xlim(0, L); ax.set_xlabel("x"); ax.set_ylabel("y (direction of motion)")
    ax.set_title(title, fontsize=10)

    # zoom on the window with the largest local excursion, at equal aspect
    z = min(a.zoom, L)
    ex = [np.ptp(u[np.arange(i, i + z) % L]) for i in range(0, L, z // 4)]
    i0 = int(np.argmax(ex)) * (z // 4)
    sel = (curve[:, 0] >= i0) & (curve[:, 0] <= i0 + z)
    yc = u[np.arange(i0, i0 + z) % L].mean()
    zy0, zy1 = int(max(0, yc - z / 4)), int(min(phi.shape[0], yc + z / 4))
    az = fig.add_subplot(gs[1])
    xs = np.arange(i0, i0 + z) % L
    az.imshow(phi[zy0:zy1][:, xs], origin="lower", extent=(i0, i0 + z, zy0, zy1), cmap="RdBu_r",
              vmin=-1.2, vmax=1.2, interpolation="nearest")
    az.plot(curve[sel, 0], curve[sel, 1], "k-", lw=0.8)
    az.set_xlim(i0, i0 + z); az.set_ylim(zy0, zy1)
    az.set_xlabel("x"); az.set_ylabel("y")
    az.set_title(f"zoom, x = {i0}–{i0 + z}, equal aspect", fontsize=10)
    ax.axvspan(i0, i0 + z, color="k", alpha=0.08)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(title.replace("\n", "; "))
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
