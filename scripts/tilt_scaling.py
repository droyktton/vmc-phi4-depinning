"""Finite-size scaling of the threshold-vs-tilt curvature (weak 8x1 blocks, Delta = 0.2; §5.9).

C(L,s) = -2 [<f_c(L,s)>/<f_c(L,0)> - 1] / s^2. With f_c(L,s) - f_c(L,0) = f_c L^(-1/nu) g(s L^(1-zeta)) and
g(x) ~ x^2 at small x, C ~ L^(2(1-zeta) - 1/nu): +0.16 for qKPZ (zeta 0.63, 1/nu 0.577), -1.25 for qEW
(zeta 1.25, 1/nu 0.75, and lambda = 0). Only s L^(1-zeta) << 1 tests it.

The tilt runs use seeds 1-30 at every s, the same as at s = 0, so the ratio is taken over the common seeds
(paired), with bootstrap errors over seeds; +s and -s are averaged per seed. Reads
<root>/ax8ay1_L{512,1024,2048}_D0.2_m*/hd_summary.csv (data/aniso_weak/merged).

  srun -p cpu --mem=2G python3 scripts/tilt_scaling.py --outdir results/aniso_weak/tilt"""
import argparse, csv, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--root', default='data/aniso_weak/merged')
p.add_argument('--outdir', default='results/aniso_weak/tilt')
p.add_argument('--nboot', type=int, default=4000)
args = p.parse_args()
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
Ls = (512, 1024, 2048)


def load(L, m):
    return {int(r['seed']): float(r['h_d'])
            for r in csv.DictReader(open(f'{args.root}/ax8ay1_L{L}_D0.2_m{m}/hd_summary.csv'))}


ms = {}
for d in glob.glob(f'{args.root}/ax8ay1_L*_D0.2_m*'):
    L, m = map(int, re.search(r'L(\d+)_D0.2_m(-?\d+)$', d).groups())
    if m > 0 and L in Ls:
        ms.setdefault(L, []).append(m)

res = {}  # (L, s) -> (r, err), r = <f_c(s)>/<f_c(0)> - 1
for L in Ls:
    f0 = load(L, 0)
    for m in sorted(ms[L]):
        fp, fm = load(L, m), load(L, -m)
        S = sorted(set(f0) & set(fp) & set(fm))
        a0 = np.array([f0[k] for k in S])
        a = np.array([(fp[k] + fm[k]) / 2 for k in S])
        idx = rng.integers(0, len(S), (args.nboot, len(S)))
        res[(L, m / L)] = (a.mean() / a0.mean() - 1, np.std(a[idx].mean(1) / a0[idx].mean(1) - 1))

print("    L      s   <f_c(s)>/<f_c(0)>-1     C(L,s)")
for (L, s), (r, e) in sorted(res.items(), key=lambda t: (round(t[0][1], 2), t[0][0])):
    print(f"{L:5d} {s:6.3f}   {r:+.4f} +- {e:.4f}   {-2*r/s**2:6.2f} +- {2*e/s**2:5.2f}")

pred = {'qKPZ': 2 * (1 - 0.63) - 0.577, 'zeta 0.63, 1/nu 0.685': 2 * (1 - 0.63) - 0.685,
        'qEW': 2 * (1 - 1.25) - 0.75}
print("\nslope of ln C vs ln L:")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cols = {0.05: '#9fb0c6', 0.1: '#2f5f98', 0.2: '#c0394b', 0.3: '#7a1f2b'}
for sr, c in cols.items():
    L, C, eC = map(np.array, zip(*sorted((L, -2*r/s**2, 2*e/s**2) for (L, s), (r, e) in res.items()
                                        if abs(s - sr) < 0.01)))
    w = (C / eC) ** 2
    X = np.c_[np.ones(len(L)), np.log(L)]
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    b = cov @ X.T @ (w * np.log(C))
    print(f"  |s| = {sr:4.2f}: {b[1]:+.3f} +- {np.sqrt(cov[1, 1]):.3f}  (chi2 {np.sum(w * (np.log(C) - X @ b) ** 2):.1f}/1)")
    ax[0].errorbar(L * (1 + 0.04 * (sr - 0.15) / 0.1), C, eC, fmt='o-', c=c, label=f'|s| = {sr}')
for k, v in pred.items():
    print(f"  predicted, {k}: {v:+.3f}")
Lf = np.geomspace(450, 2300, 20)
for x, ls, lab in ((pred['qKPZ'], ':', 'qKPZ: 2(1−ζ)−1/ν = +0.16'), (pred['qEW'], '--', 'qEW: −1.25')):
    ax[0].plot(Lf, 14.5 * (Lf / 1024) ** x, 'k', ls=ls, lw=1, label=lab)
ax[0].set(xscale='log', yscale='log', xlabel='L', ylabel='C(L,s) = −2[⟨f_c(s)⟩/⟨f_c(0)⟩ − 1]/s²', ylim=(3, 40),
          title='C(L) does not depend on L')
ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_xticks(Ls)
ax[0].set_xticklabels([str(L) for L in Ls])
ax[0].legend(fontsize=9)
for (L, s), (r, e) in res.items():
    c = dict(zip(Ls, ('#9fb0c6', '#2f5f98', '#c0394b')))[L]
    ax[1].errorbar(s * L ** 0.37, -r * L ** 0.577, e * L ** 0.577, fmt='o', c=c, label=f'L = {L}')
h, l = ax[1].get_legend_handles_labels()
u = dict(zip(l, h))
ax[1].legend(u.values(), u.keys())
ax[1].set(xscale='log', yscale='log', xlabel='|s| L^(1−ζ),  ζ = 0.63',
          ylabel='−[⟨f_c(s)⟩/⟨f_c(0)⟩ − 1] L^(1/ν),  1/ν = 0.577', title='Collapse with qKPZ exponents')
fig.tight_layout()
fig.savefig(out / 'C_tilt.png', dpi=130)
print(f"\nplot: {out / 'C_tilt.png'}")
