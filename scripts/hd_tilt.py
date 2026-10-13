"""Depinning field versus the wall's mean slope s = m/L (helical BC, --tilt m).

For qKPZ depinning the threshold depends on the tilt, with the coupling
lambda_c = F_c''(0) (Tang, Kardar, Dhar 1995); in an isotropic medium F_c(s)
is flat and the class is qEW. Reads <root>/[ax<AX>ay<AY>_]L*_D*_m*/hd_summary.csv
(the prefix: block disorder, see ./phi4vmc AX/AY), fits
<h_d>(s) = a + b s + (lambda_c/2) s^2 per (L, Delta) weighted by the
standard errors, and plots <h_d>(s)/<h_d>(0).

  srun -p cpu --mem=2G python3 scripts/hd_tilt.py --root data/tilt --outdir results/hd_tilt"""
import argparse, csv, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--root', default='data/tilt')
p.add_argument('--outdir', default='results/hd_tilt')
args = p.parse_args()
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)

data = {}  # (blocks, L, Delta) -> list of (s, mean, sem, n)
for d in glob.glob(f'{args.root}/*L*_D*_m*'):
    bl, L, D, m = re.search(r'(?:ax(\d+ay\d+)_)?L(\d+)_D([\d.]+)_m(-?\d+)$', d).groups()
    bl = (bl or '1ay1').replace('ay', 'x')
    f = Path(d) / 'hd_summary.csv'
    if not f.exists():
        continue
    h = np.array([float(r['h_d']) for r in csv.DictReader(open(f)) if r['seed'] != 'seed'])
    if len(h) < 2:
        continue
    data.setdefault((bl, int(L), float(D)), []).append((int(m) / int(L), h.mean(), h.std(ddof=1) / np.sqrt(len(h)), len(h)))

lines = [f"{'AXxAY':>5} {'L':>5} {'Delta':>5} {'s':>7} {'n':>3} {'<h_d>':>9} {'sem':>8} {'<h_d>/<h_d>(0)-1':>17}"]
fits = {}
for key in sorted(data):
    pts = sorted(data[key])
    s, h, e, n = map(np.array, zip(*pts))
    h0 = h[np.argmin(np.abs(s))]
    for si, hi, ei, ni in pts:
        lines.append(f"{key[0]:>5} {key[1]:>5} {key[2]:>5} {si:>7.3f} {ni:>3} {hi:>9.6f} {ei:>8.1e} {100 * (hi / h0 - 1):>+16.2f}%")
    if len(s) >= 4:
        A = np.vstack([np.ones_like(s), s, s ** 2 / 2]).T
        w = 1 / e ** 2
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        c = cov @ (A.T @ (w * h))
        chi2 = np.sum(w * (h - A @ c) ** 2)
        fits[key] = c
        lines.append(f"   fit: <h_d>(0) = {c[0]:.6f} +- {np.sqrt(cov[0, 0]):.1e}   b = {c[1]:+.2e} +- {np.sqrt(cov[1, 1]):.1e}"
                     f"   lambda_c = {c[2]:+.2e} +- {np.sqrt(cov[2, 2]):.1e}"
                     f"   lambda_c/<h_d>(0) = {c[2] / c[0]:+.3f} +- {np.sqrt(cov[2, 2]) / c[0]:.3f}"
                     f"   chi2 = {chi2:.1f}/{len(s) - 3}")
print('\n'.join(lines))
(out / 'hd_tilt.dat').write_text('#' + '\n'.join(lines) + '\n')

fig, ax = plt.subplots(figsize=(6, 4.2))
for i, key in enumerate(sorted(data)):
    s, h, e, n = map(np.array, zip(*sorted(data[key])))
    ref = fits[key][0] if key in fits else h[np.argmin(np.abs(s))]
    ax.errorbar(s, h / ref, e / ref, marker='os^vDP<>'[i % 8], ls='none', color=f'C{i % 10}', label=(f'{key[0]}, ' if key[0] != '1x1' else '') + f'L={key[1]}, Δ={key[2]}')
    if key in fits:
        ss = np.linspace(s.min(), s.max(), 100)
        c = fits[key]
        ax.plot(ss, (c[0] + c[1] * ss + c[2] * ss ** 2 / 2) / ref, color=f'C{i % 10}', lw=0.8)
ax.axhline(1, color='k', lw=0.5, ls=':')
ax.set(xlabel='mean wall slope s = m/L', ylabel=r'$\langle h_d\rangle(s)/\langle h_d\rangle(0)$',
       title='threshold vs tilt (helical BC)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / 'hd_tilt.png', dpi=130)
print(f'plot: {out}/hd_tilt.png')
