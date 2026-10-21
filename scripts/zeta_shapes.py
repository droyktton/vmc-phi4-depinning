"""Local roughness exponent zeta_W(l) (chord) at Delta = 0.2 for 1x1, 4x1, 8x1 and 16x1 blocks (§5.10):
does the block shape move the crossover? Left: zeta_W vs l at L = 2048 (solid) and 1024 (dashed), with
l_x (zeta_W < 1.125) and l_p (zeta_W < 0.8) marked. Middle and right: the same against l/l_x and l/l_p; if
one length sets the crossover, the anisotropic curves collapse there. The lengths are the first crossing
of the L = 2048 curve (lo_table.dat in the same directories gives them with errors). Reads
<root>/roughness_ax{1,4,8,16}_th0.8/local_L{1024,2048}_D0.2.dat, written by scripts/local_roughness.py
from data/aniso_nu/analyze.sh.

  srun -p cpu --mem=2G python3 scripts/zeta_shapes.py"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--root', default='results/aniso_nu')
p.add_argument('--out', default='results/aniso_nu/zeta_W_shapes_L2048.png')
args = p.parse_args()


def load(A, L):
    d = np.loadtxt(f'{args.root}/roughness_ax{A}_th0.8/local_L{L}_D0.2.dat')
    ok = np.isfinite(d[:, 4])
    return d[ok, 0], d[ok, 4]


def first_below(l, z, th):
    i = np.flatnonzero(z < th)
    return l[i[0]] if len(i) else np.nan


fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for A, c in ((1, 'k'), (4, 'C0'), (8, 'C3'), (16, 'C2')):
    l, z = load(A, 2048)
    lx, lp = first_below(l, z, 1.125), first_below(l, z, 0.8)
    print(f"{A:2d}x1: l_x = {lx:.0f}, l_p = {lp:.0f}")
    l1, z1 = load(A, 1024)
    ax[0].plot(l, z, c=c, label=f'{A}×1' + ('' if np.isnan(lp) else f'  (ℓ× = {lx:.0f}, ℓ_p = {lp:.0f})'))
    ax[0].plot(l1, z1, c=c, ls='--', lw=0.8)
    for x in (lx, lp):
        ax[0].plot([x], [1.125 if x == lx else 0.8], 'o', c=c, mfc='none', ms=9)
    if A > 1:
        ax[1].plot(l / lx, z, c=c, label=f'{A}×1')
        ax[2].plot(l / lp, z, c=c, label=f'{A}×1')
for a in ax:
    for y, ls in ((1.25, '--'), (0.63, ':')):
        a.axhline(y, c='k', ls=ls, lw=1)
    for y in (1.125, 0.8):
        a.axhline(y, c='0.6', lw=0.8)
    a.set(xscale='log', ylim=(0.3, 1.65))
ax[0].text(5, 1.27, 'qEW 1.25', fontsize=9)
ax[0].text(5, 0.65, 'qKPZ 0.63', fontsize=9)
ax[0].set(xlabel='l', ylabel='ζ_W(l), chord', title='L = 2048 (solid), 1024 (dashed); ○ = ℓ×, ℓ_p')
ax[1].set(xlabel='l / ℓ×', title='rescaled by ℓ× (ζ_W = 1.125)')
ax[2].set(xlabel='l / ℓ_p', title='rescaled by ℓ_p (ζ_W = 0.8)')
ax[0].legend(fontsize=9, loc='lower left')
ax[2].legend()
fig.suptitle('Local roughness exponent, Δ = 0.2, 30 seeds per L')
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=120)
print(f"plot: {args.out}")
