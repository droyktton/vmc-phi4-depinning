"""sigma(h_d) L^0.577 vs L for the weak 8x1 walls (Delta = 0.2, 200 seeds per L; §5.9): flat if the
threshold spread scales with the qKPZ 1/nu = 0.577. Reference lines for 1/nu = 0.685 (the fit) and 0.75
(qEW), normalized at L = 1024. Errors by bootstrap over samples.

  srun -p cpu --mem=2G python3 scripts/sigma_scaled.py --outdir results/aniso_weak/hd_variance"""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--root', default='data/aniso_weak/merged_hd')
p.add_argument('--outdir', default='results/aniso_weak/hd_variance')
p.add_argument('--nboot', type=int, default=4000)
args = p.parse_args()
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(3)
Ls = np.array([256, 512, 1024, 2048, 4096, 8192])

s, e = [], []
for L in Ls:
    x = np.array([float(r['h_d'])
                  for r in csv.DictReader(open(f'{args.root}/ax8ay1_L{L}_D0.2_m0/hd_summary.csv'))])
    s.append(x.std(ddof=1))
    e.append(np.std(x[rng.integers(0, len(x), (args.nboot, len(x)))].std(1, ddof=1)))
y, ey = np.array(s) * Ls ** 0.577, np.array(e) * Ls ** 0.577
for L, a, b in zip(Ls, y, ey):
    print(f"L = {L:5d}: sigma L^0.577 = {a:.4f} +- {b:.4f}")

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(Ls, y, ey, fmt='*', ms=12, c='#c0394b', label='8×1, Δ = 0.2, 200 seeds per L')
Lf = np.geomspace(200, 10000, 50)
for x, ls, lab in ((0.577, ':', 'qKPZ, 1/ν = 0.577 (flat)'), (0.685, '-', 'fit, 1/ν = 0.685'),
                   (0.75, '--', 'qEW, 1/ν = 0.75')):
    ax.plot(Lf, y[2] * (Lf / Ls[2]) ** (0.577 - x), 'k', ls=ls, lw=1, label=lab)
ax.set(xscale='log', xlabel='L', ylabel='σ(h_d) · L^0.577', title='Spread of the threshold, rescaled by the qKPZ exponent')
ax.legend()
fig.tight_layout()
fig.savefig(out / 'sigma_L0577.png', dpi=130)
print(f"plot: {out / 'sigma_L0577.png'}")
