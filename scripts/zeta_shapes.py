"""Local roughness exponent zeta_W(l) (chord) at L = 2048, Delta = 0.2, for 1x1, 4x1, 8x1 and 16x1 blocks
(§5.10), with the thresholds that define l_x (1.125) and l_p (0.8). Reads
<root>/roughness_ax{1,4,8,16}_th0.8/local_L2048_D0.2.dat, written by scripts/local_roughness.py
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

fig, ax = plt.subplots(figsize=(7.5, 5))
for A, c in ((1, 'k'), (4, 'C0'), (8, 'C3'), (16, 'C2')):
    d = np.loadtxt(f'{args.root}/roughness_ax{A}_th0.8/local_L2048_D0.2.dat')
    ax.plot(d[:, 0], d[:, 4], c=c, label=f'{A}×1')
for y, ls, lab in ((1.25, '--', 'qEW, ζ = 1.25'), (0.63, ':', 'qKPZ, ζ = 0.63')):
    ax.axhline(y, c='k', ls=ls, lw=1, label=lab)
for y in (1.125, 0.8):
    ax.axhline(y, c='0.6', lw=0.8)
ax.set(xscale='log', ylim=(0.3, 1.6), xlabel='l', ylabel='ζ_W(l), chord',
       title='Local roughness exponent, L = 2048, Δ = 0.2 (30 seeds)')
ax.legend()
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=130)
print(f"plot: {args.out}")
