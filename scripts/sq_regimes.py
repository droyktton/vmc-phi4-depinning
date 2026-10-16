#!/usr/bin/env python3
"""Regimes of the global structure factor S(q) ~ q^-(1+2 zeta).

Reads the Sq_L*.dat files written by structure_factor.py (q, S_mean, S_sem),
bins them logarithmically in q, and plots
  - S(q) for every L (rescaled by q^2.26 to make the qKPZ regime flat),
  - the local slope -d ln S / d ln q between successive bins, against
    l = 2 pi / q, with the reference values 1+2 zeta: 2.26 (qKPZ),
    3.5 (qEW), 4 (Larkin, zeta = 3/2);
and fits a power law in each window given with --window lmin:lmax
(in l = 2 pi / q), weighted by the standard errors of the binned S.

Example:
    ./sq_regimes.py results/aniso_weak/Sq --window 2:6 --window 15:40 --window 300:2000
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REFS = {'qKPZ': 2.26, 'qEW': 3.5, 'Larkin': 4.0}


def logbin(q, S, e, per_decade):
    edges = 10 ** np.arange(np.floor(np.log10(q.min())), np.log10(q.max()) + 1 / per_decade, 1 / per_decade)
    idx = np.digitize(q, edges)
    qb, Sb, eb = [], [], []
    for i in np.unique(idx):
        m = idx == i
        qb.append(np.exp(np.log(q[m]).mean()))
        Sb.append(S[m].mean())
        eb.append(np.sqrt(np.sum(e[m] ** 2)) / m.sum())
    return map(np.array, (qb, Sb, eb))


def fit(q, S, e, lmin, lmax):  # weighted fit ln S = a - x ln q in the window
    m = (q >= 2 * np.pi / lmax) & (q <= 2 * np.pi / lmin)
    if m.sum() < 3:
        return None
    X = np.vstack([np.ones(m.sum()), -np.log(q[m])]).T
    w = (S[m] / e[m]) ** 2
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    b = cov @ (X.T @ (w * np.log(S[m])))
    chi2 = np.sum(w * (np.log(S[m]) - X @ b) ** 2)
    return b, np.sqrt(cov[1, 1]), chi2, m.sum() - 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sqdir', help='directory with Sq_L*.dat')
    p.add_argument('--window', action='append', default=[], help='lmin:lmax in l = 2 pi/q (repeatable)')
    p.add_argument('--per-decade', type=int, default=10, help='log bins per decade of q')
    p.add_argument('--fit-per-decade', type=int, default=40, help='finer bins used for the fits')
    args = p.parse_args()

    sqdir = Path(args.sqdir)
    files = sorted(glob.glob(str(sqdir / 'Sq_L*.dat')), key=lambda f: int(re.search(r'L(\d+)', f).group(1)))
    windows = [tuple(map(float, w.split(':'))) for w in args.window]
    lines = ['# fits S ~ q^-x in windows of l = 2 pi/q; x = 1 + 2 zeta  (qKPZ 2.26, qEW 3.5, Larkin 4)',
             '# L  lmin  lmax  x  err  zeta  chi2/dof  nbins']
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.get_cmap('viridis')
    for i, f in enumerate(files):
        L = int(re.search(r'L(\d+)', f).group(1))
        q, S, e = np.loadtxt(f, unpack=True)
        k = q > 0
        q, S, e = q[k], S[k] / L, e[k] / L  # S/L: same amplitude for every L
        c = cmap(i / max(len(files) - 1, 1))
        qb, Sb, eb = logbin(q, S, e, args.per_decade)
        ax[0].errorbar(qb, Sb * qb ** 2.26, eb * qb ** 2.26, color=c, marker='o', ms=3, lw=1, label=f'L={L}')
        sl = -np.diff(np.log(Sb)) / np.diff(np.log(qb))
        lm = 2 * np.pi / np.sqrt(qb[1:] * qb[:-1])
        esl = np.hypot(eb[1:] / Sb[1:], eb[:-1] / Sb[:-1]) / np.diff(np.log(qb))
        ok = esl < 0.6
        ax[1].errorbar(lm[ok], sl[ok], esl[ok], color=c, marker='o', ms=3, lw=1, label=f'L={L}')
        qf, Sf, ef = logbin(q, S, e, args.fit_per_decade)
        for lmin, lmax in windows:
            if lmax > L / 2:
                continue
            r = fit(qf, Sf, ef, lmin, lmax)
            if r is None:
                continue
            b, ex, chi2, dof = r
            lines.append(f'{L:5d} {lmin:5g} {lmax:5g}  {b[1]:.3f} {ex:.3f}  {(b[1] - 1) / 2:.3f}  {chi2:.1f}/{dof}  {dof + 2}')
            if L == int(re.search(r'L(\d+)', files[-1]).group(1)):
                qq = np.array([2 * np.pi / lmax, 2 * np.pi / lmin])
                ax[0].plot(qq, np.exp(b[0]) * qq ** (2.26 - b[1]), 'r-', lw=2.5, alpha=0.6)
                ax[1].plot([lmin, lmax], [b[1]] * 2, 'r-', lw=2.5, alpha=0.6)
                ax[1].annotate(f'{b[1]:.2f}', (np.sqrt(lmin * lmax), b[1]), textcoords='offset points',
                               xytext=(0, 8), ha='center', color='r')
    for name, x in REFS.items():
        ax[1].axhline(x, color='k', ls='--', lw=0.8)
        ax[1].text(1.5, x + 0.05, f'{name} {x}', fontsize=8)
    ax[0].set(xscale='log', yscale='log', xlabel='q', ylabel=r'$q^{2.26}\,S(q)/L$',
              title='global S(q), flat = qKPZ (red: fits at the largest L)')
    ax[1].set(xscale='log', xlabel=r'$l = 2\pi/q$', ylabel=r'$-d\ln S/d\ln q = 1+2\zeta$',
              title='local slope of S(q)', ylim=(-0.5, 6))
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(sqdir / 'Sq_regimes.png', dpi=130)
    print('\n'.join(lines))
    (sqdir / 'Sq_regimes.dat').write_text('\n'.join(lines) + '\n')
    print(f'plot: {sqdir / "Sq_regimes.png"}')


if __name__ == '__main__':
    main()
