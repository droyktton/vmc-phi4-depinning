#!/usr/bin/env python3
"""Benchmark of the roughness estimators of local_roughness.py on
synthetic single-valued Gaussian profiles with a known structure factor

    S(q) ~ q^-(1+2 zeta1)   for q > 2 pi / l_o    (small scales)
    S(q) ~ q^-(1+2 zeta2)   for q < 2 pi / l_o    (large scales)

(continuous at q = 2 pi / l_o; pass --lo 0 for a pure zeta1 profile).
The amplitude is set by the rms nearest-neighbour step, --step.
Each profile is turned into a curve (x, u(x)) and goes through exactly
the same measurement code as the real walls (measure_curve), so this
checks which estimators recover zeta1 / zeta2 and the input l_o, and
shows the expected artifact of the global B(r) and W2: for zeta1 > 1
they can't report more than zeta = 1.

Example:
    ./synthetic_profiles.py --L 512 --nsamples 30 --zeta1 1.25 --zeta2 0.5 --lo 64 \\
        --outdir data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np

import local_roughness as lr


def synthetic_profile(L, zeta1, zeta2, lo, step, rng):
    k = np.arange(1, L // 2 + 1)
    q = 2 * np.pi * k / L
    S = q ** -(1 + 2 * zeta1)
    if lo > 0:
        qo = 2 * np.pi / lo
        low = q < qo
        S[low] = qo ** -(1 + 2 * zeta1) * (q[low] / qo) ** -(1 + 2 * zeta2)
    U = np.zeros(L // 2 + 1, complex)
    U[1:] = np.sqrt(S / 2) * (rng.standard_normal(len(k)) + 1j * rng.standard_normal(len(k)))
    u = np.fft.irfft(U, n=L)
    rms_step = np.sqrt(np.mean((np.roll(u, -1) - u) ** 2))
    return u * step / rms_step


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--nsamples", type=int, default=30)
    p.add_argument("--zeta1", type=float, default=1.25)
    p.add_argument("--zeta2", type=float, default=0.5)
    p.add_argument("--lo", type=float, default=64, help="crossover length (0: pure zeta1)")
    p.add_argument("--step", type=float, default=0.5, help="rms nearest-neighbour height difference")
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--nboot", type=int, default=50)
    p.add_argument("--half-width", type=float, default=0.15)
    p.add_argument("--zeta-th", type=float, default=None,
                   help="l_o threshold on the local exponent (default: midway between zeta1 and zeta2)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    L = args.L
    ls, rs = lr.grids(L)
    rows = []
    for _ in range(args.nsamples):
        u = synthetic_profile(L, args.zeta1, args.zeta2, args.lo, args.step, rng)
        # points at x+0.5, where univalued_profile samples the columns,
        # so that its linear interpolation doesn't smooth the profile
        x = np.arange(L + 1, dtype=float) + 0.5
        curve = np.column_stack([x, np.append(u, u[0])])
        rows.append(lr.measure_curve(curve, L, ls, rs))
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    d.update(ls=ls, rs=rs, n_bubbles=np.array(0))

    th = args.zeta_th if args.zeta_th is not None else 0.5 * (args.zeta1 + args.zeta2)
    a = argparse.Namespace(half_width=args.half_width, zeta_th=th, fmv_th=0.05,
                           lmin=8, nboot=args.nboot, pca_transform="dct")
    full, boots = lr.bootstrap(d, L, a, rng)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"synthetic_L{L}_z{args.zeta1:g}_{args.zeta2:g}_lo{args.lo:g}"

    print(f"{name}: n={args.nsamples}, zeta1={args.zeta1}, zeta2={args.zeta2}, l_o={args.lo}")
    print("\nlocal exponent, median over scale ranges (bootstrap error):")
    keys = ["zeta_S_glob", "zeta_B_glob", "zeta_W_chord", "zeta_W_pca",
            "zeta_B_chord", "zeta_B_pca", "zeta_S_chord", "zeta_S_pca"]
    lo = args.lo if args.lo > 0 else L
    ranges = [("8 <= l <= l_o/2", 8, lo / 2), ("l >= 2 l_o", 2 * lo, L)]
    print(f"  {'estimator':14s}" + "".join(f"{r[0]:>22s}" for r in ranges))
    for k in keys:
        x, y = full["curves"][k]
        cells = []
        for _, a0, a1 in ranges:
            sel = (x >= a0) & (x <= a1) & np.isfinite(y)
            if sel.sum() == 0:
                cells.append(f"{'--':>22s}")
                continue
            v = np.median(y[sel])
            e = np.nanstd([np.nanmedian(b["curves"][k][1][sel]) for b in boots])
            cells.append(f"{v:>15.3f} +/- {e:.3f}")
        print(f"  {k:14s}" + "".join(cells))

    print(f"\nl_o estimates (input l_o = {args.lo:g}; threshold zeta_th = {th:.3f}):")
    for k, v in full["lo"].items():
        e = np.nanstd([b["lo"][k] for b in boots])
        print(f"  {k:14s} {v:9.4g} +/- {e:.2g}")

    sets = [dict(L=L, delta=0.0, full=full, boots=boots)]
    lr.plot_zeta_panels({L: sets}, outdir, a)
    src = outdir / f"zeta_local_L{L}.png"
    dst = outdir / f"zeta_local_{name}.png"
    src.replace(dst)
    print(f"\nplot written to {dst}")


if __name__ == "__main__":
    main()
