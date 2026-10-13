"""Sample-to-sample spread of h_d vs L at fixed kappa = P/w.

For qEW depinning in 1+1, sigma(h_d) ~ L^(-1/nu) with 1/nu = 2 - zeta = 0.75
(zeta_q = 1.25), if the box keeps the finite-size regime fixed (kappa fixed).
Reads data/kappa/L*_D*_kap* (kappa 15/25/40, tol 1e-4) and the kappa=20
production data/prod_kappa20/L*_D* (tol 1e-5).

The bisection reports h_d on a grid of step (h_high-h_low)/2^n, the first
halving below tol; sigma^2 is corrected for it with Sheppard's -step^2/12.
Errors on sigma are bootstrap over samples.

  srun -p cpu --mem=4G python3 scripts/hd_variance.py --outdir results/hd_variance"""
import argparse, csv, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--outdir', default='results/hd_variance')
p.add_argument('--nboot', type=int, default=1000)
p.add_argument('--min-samples', type=int, default=5)
args = p.parse_args()
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)


def step(hl, hh, tol):
    w = hh - hl
    n = int(np.ceil(np.log2(w / tol)))
    return w / 2 ** n


def kappa_bracket(D):  # data/kappa/run.sh
    hest = 0.0195 * (D / 0.4) ** 1.34
    return 0.85 * hest, 1.15 * hest


sets = []  # (D, kappa, L, h_d array, bisection step)
for d in glob.glob('data/kappa/L*_D*_kap*'):
    L, D, K = re.search(r'L(\d+)_D([\d.]+)_kap(\d+)$', d).groups()
    sets.append((float(D), int(K), int(L), d, step(*kappa_bracket(float(D)), 1e-4)))
prod_bracket = {0.3: (0.01256, 0.01416), 0.2: (0.0074, 0.0084)}  # data/prod_kappa20 runs, tol 1e-5
for d in glob.glob('data/prod_kappa20/L*_D*'):
    m = re.search(r'L(\d+)_D([\d.]+)$', d)
    if m:
        L, D = int(m.group(1)), float(m.group(2))
        sets.append((D, 20, L, d, step(*prod_bracket[D], 1e-5)))

rows = []
for D, K, L, d, st in sorted(sets):
    h = np.array([float(r['h_d']) for r in csv.DictReader(open(d + '/hd_summary.csv')) if r['seed'] != 'seed'])
    if len(h) < args.min_samples:
        print(f'skip {d}: {len(h)} samples')
        continue
    var_obs = h.var(ddof=1)
    sig = np.sqrt(max(var_obs - st ** 2 / 12, 0))
    boot = [np.sqrt(max(b.var(ddof=1) - st ** 2 / 12, 0)) for b in (rng.choice(h, len(h)) for _ in range(args.nboot))]
    rows.append(dict(D=D, K=K, L=L, n=len(h), mean=h.mean(), sig=sig, err=np.std(boot),
                     step=st, sheppard=st ** 2 / 12 / var_obs, distinct=len(np.unique(h))))

hdr = f"{'Delta':>5} {'kappa':>5} {'L':>6} {'n':>3} {'<h_d>':>9} {'sigma':>9} {'err':>8} {'sigma/<h>':>9} {'step':>8} {'Shep %':>6} {'distinct':>8}"
lines = [hdr]
for r in rows:
    lines.append(f"{r['D']:>5} {r['K']:>5} {r['L']:>6} {r['n']:>3} {r['mean']:>9.6f} {r['sig']:>9.3e} {r['err']:>8.1e}"
                 f" {100 * r['sig'] / r['mean']:>8.2f}% {r['step']:>8.1e} {100 * r['sheppard']:>6.1f} {r['distinct']:>8}")


def fit(sel):  # weighted fit ln sigma = a - x ln L
    L = np.array([r['L'] for r in sel], float); s = np.array([r['sig'] for r in sel]); e = np.array([r['err'] for r in sel])
    w = (s / e) ** 2
    A = np.vstack([np.ones_like(L), -np.log(L)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, x = cov @ (A.T @ (w * np.log(s)))
    chi2 = np.sum(w * (np.log(s) - A @ [a, x]) ** 2)
    return a, x, np.sqrt(cov[1, 1]), chi2, len(sel) - 2


lines.append('')
lines.append('# fits ln sigma = a - (1/nu) ln L, per (Delta, kappa); expected 1/nu = 2 - zeta_q = 0.75')
groups = sorted({(r['D'], r['K']) for r in rows})
fits = {}
for g in groups:
    sel = [r for r in rows if (r['D'], r['K']) == g]
    if len(sel) < 3:
        continue
    a, x, ex, chi2, dof = fit(sel)
    fits[g] = (a, x)
    lines.append(f"Delta={g[0]:<4} kappa={g[1]:<3} L={min(r['L'] for r in sel)}-{max(r['L'] for r in sel)}"
                 f"  1/nu = {x:.3f} +- {ex:.3f}   chi2/dof = {chi2:.1f}/{dof}")
    sel = [r for r in sel if r['L'] >= 512]
    if len(sel) >= 3:
        a, x, ex, chi2, dof = fit(sel)
        lines.append(f"{'':21} L>=512      1/nu = {x:.3f} +- {ex:.3f}   chi2/dof = {chi2:.1f}/{dof}")
print('\n'.join(lines))
(out / 'hd_variance.dat').write_text('#' + '\n'.join(lines) + '\n')

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
colors = {0.2: 'C0', 0.3: 'C1', 1.0: 'C2'}
marks = {15: 'o', 20: '*', 25: 's', 40: '^'}
for g in groups:
    sel = [r for r in rows if (r['D'], r['K']) == g]
    L = np.array([r['L'] for r in sel]); s = np.array([r['sig'] for r in sel]); e = np.array([r['err'] for r in sel])
    m = np.array([r['mean'] for r in sel])
    kw = dict(color=colors.get(g[0], 'k'), marker=marks.get(g[1], 'x'), ms=9 if g[1] == 20 else 5,
              ls='-' if len(sel) > 1 else 'none', lw=0.8, label=f'Δ={g[0]}, κ={g[1]}')
    ax[0].errorbar(L, s / m, e / m, **kw)
    ax[1].errorbar(L, s / m * L ** 0.75, e / m * L ** 0.75, **kw)
Lr = np.array([200, 20000])
ax[0].plot(Lr, 0.3 * (Lr / 200.) ** -0.75, 'k--', lw=1, label='$L^{-0.75}$')
ax[0].set(xscale='log', yscale='log', xlabel='L', ylabel=r'$\sigma(h_d)/\langle h_d\rangle$',
          title='sample-to-sample spread of $h_d$')
ax[1].set(xscale='log', yscale='log', xlabel='L', ylabel=r'$L^{0.75}\,\sigma(h_d)/\langle h_d\rangle$',
          title='rescaled by $L^{2-\\zeta_q}$')
ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(out / 'hd_variance.png', dpi=130)
print(f'plot: {out}/hd_variance.png')
