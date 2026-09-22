#!/usr/bin/env python3
"""Fits the roughness exponent zeta of the critical domain-wall from its
structure factor, S(q) ~ q^-(1+2*zeta), i.e.
    log S(q) = -(1+2*zeta) * log(q) + const
over a chosen low-q range.

Reads the structure_factor.dat file written by structure_factor.py
(columns: q, S_mean, S_sem). By default (no --qmin/--qmax given), the
fit range is chosen automatically: among windows of at least
--min-points contiguous points in the low-q half of the data, the one
with the best weighted-least-squares R^2 (and a decaying slope) is
used. This is a starting point, not a substitute for looking at the
data -- the script also prints the local (secant-slope) effective
exponent between every pair of consecutive points, and the diagnostic
plot marks which points were actually used; override with --qmin/--qmax
if the auto-selected window doesn't look right.

Per the paper, S(q) generally shows two power-law regimes separated by
a disorder-dependent crossover length l_o ~ Delta^(-1/0.45): quenched-EW
roughness zeta~1.2 for length scales below l_o (q > 2*pi/l_o), crossing
over to invasion-percolation zeta_eff~0.5 above l_o (q < 2*pi/l_o). Low
q (large length scales) only probes the second regime if L is several
times l_o; with modest sample counts and L, expect noisy low-q points
and don't over-interpret a fit from just a handful of them.

Examples:
    ./fit_roughness.py data/L128_D0.2/structure_factor.dat
    ./fit_roughness.py data/L128_D0.2/structure_factor.dat --qmin 0.05 --qmax 0.2
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    data = np.loadtxt(path, comments="#")
    q, S, sem = data[:, 0], data[:, 1], data[:, 2]
    mask = q > 0  # drop the q=0 term (S(0) is identically 0 after demeaning)
    return q[mask], S[mask], sem[mask]


def weighted_fit(log_q, log_S, weights):
    """Weighted least squares fit of log_S = m*log_q + c. Returns m, c, cov, r2."""
    A = np.vstack([log_q, np.ones_like(log_q)]).T
    W = np.diag(weights)
    ATA = A.T @ W @ A
    ATy = A.T @ W @ log_S
    m, c = np.linalg.solve(ATA, ATy)
    cov = np.linalg.inv(ATA)
    pred = A @ np.array([m, c])
    residual = log_S - pred
    ss_res = np.sum(weights * residual ** 2)
    ss_tot = np.sum(weights * (log_S - np.average(log_S, weights=weights)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m, c, cov, r2


def best_window(log_q, log_S, weights, min_points):
    """Best-R^2 contiguous window (of >= min_points) among the low-q half."""
    n = len(log_q)
    upper = max(min_points, n // 2 + 1)
    best = None
    for i in range(0, upper - min_points + 1):
        for j in range(i + min_points, upper + 1):
            m, c, cov, r2 = weighted_fit(log_q[i:j], log_S[i:j], weights[i:j])
            if m >= 0:
                continue  # unphysical: S(q) must decay with q
            if best is None or r2 > best[0]:
                best = (r2, i, j, m, c, cov)
    if best is None:
        raise RuntimeError("no decaying window found in the low-q half of the data")
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("structure_factor_dat")
    p.add_argument("--qmin", type=float, default=None, help="lower edge of the fit range (default: auto)")
    p.add_argument("--qmax", type=float, default=None, help="upper edge of the fit range (default: auto)")
    p.add_argument("--min-points", type=int, default=4, help="minimum points in the auto-selected window")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()

    path = Path(args.structure_factor_dat)
    q, S, sem = load(path)
    log_q, log_S = np.log(q), np.log(S)

    # error propagation sigma_logS ~= sigma_S / S (delta method); fall
    # back to unweighted (sigma=1) where sem is zero/unavailable
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_logS = np.where(sem > 0, sem / S, 1.0)
    weights = 1.0 / sigma_logS ** 2

    print("local effective exponent zeta(q) from consecutive-point secant slopes:")
    print(" q_mid            zeta_local")
    for k in range(len(q) - 1):
        slope = (log_S[k + 1] - log_S[k]) / (log_q[k + 1] - log_q[k])
        zeta_local = -(slope + 1) / 2
        q_mid = np.sqrt(q[k] * q[k + 1])
        print(f" {q_mid:<16.5g} {zeta_local:.3f}")

    if args.qmin is not None or args.qmax is not None:
        qmin = args.qmin if args.qmin is not None else q.min()
        qmax = args.qmax if args.qmax is not None else q.max()
        idx = np.nonzero((q >= qmin) & (q <= qmax))[0]
        if len(idx) < 2:
            raise SystemExit("fewer than 2 points in the requested [qmin, qmax] range")
        i, j = idx[0], idx[-1] + 1
        m, c, cov, r2 = weighted_fit(log_q[i:j], log_S[i:j], weights[i:j])
        auto = False
    else:
        r2, i, j, m, c, cov = best_window(log_q, log_S, weights, args.min_points)
        auto = True
    qmin, qmax = q[i], q[j - 1]

    zeta = -(m + 1) / 2
    zeta_err = np.sqrt(cov[0, 0]) / 2

    print()
    print(f"fit range: q in [{qmin:.4g}, {qmax:.4g}] ({j - i} points{', auto-selected' if auto else ''})")
    print(f"S(q) ~ q^{m:.3f}  ->  zeta = {zeta:.3f} +/- {zeta_err:.3f}   (weighted R^2 = {r2:.4f})")

    prefix = Path(args.out_prefix) if args.out_prefix else path.with_name(path.stem + "_zeta_fit")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(q, S, yerr=sem, fmt="o", ms=3, color="#999999", label="data", zorder=2)
    ax.errorbar(q[i:j], S[i:j], yerr=sem[i:j], fmt="o", ms=4, color="#4c72b0", label="fit range", zorder=3)
    q_line = np.array([qmin, qmax])
    ax.plot(q_line, np.exp(c) * q_line ** m, "--", color="#c44e52",
            label=rf"fit: $\zeta$={zeta:.2f}$\pm${zeta_err:.2f}", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$S(q)$")
    ax.legend()
    fig.tight_layout()
    png_path = prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    print(f"plot written to {png_path}")


if __name__ == "__main__":
    main()
