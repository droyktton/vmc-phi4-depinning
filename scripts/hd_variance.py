"""Sample-to-sample spread of h_d vs L at fixed kappa = P/w.

For qEW depinning in 1+1, sigma(h_d) ~ L^(-1/nu) with 1/nu = 2 - zeta = 0.75
(zeta_q = 1.25), if the box keeps the finite-size regime fixed (kappa fixed).
Reads data/kappa/L*_D*_kap* (kappa 15/25/40, tol 1e-4) and the kappa=20
production data/prod_kappa20/L*_D* (tol 1e-5), both isotropic disorder, and
the weak 8x1 block disorder run data/aniso_weak/merged/ax8ay1_L*_D0.2_m0
(kappa=20, tol 1e-5; docs/plans/aniso_weak_qkpz.md), for which the qKPZ
class predicts 1/nu = 0.577. Also: local slopes of sigma between successive
L, and the threshold shift <h_d>(L) = h_inf + A L^(-1/nu).

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


sets = []  # (disorder, D, kappa, L, h_d array, bisection step)
for d in glob.glob('data/kappa/L*_D*_kap*'):
    L, D, K = re.search(r'L(\d+)_D([\d.]+)_kap(\d+)$', d).groups()
    sets.append(('iso', float(D), int(K), int(L), d, step(*kappa_bracket(float(D)), 1e-4)))
prod_bracket = {0.3: (0.01256, 0.01416), 0.2: (0.0074, 0.0084)}  # data/prod_kappa20 runs, tol 1e-5
for d in glob.glob('data/prod_kappa20/L*_D*'):
    m = re.search(r'L(\d+)_D([\d.]+)$', d)
    if m:
        L, D = int(m.group(1)), float(m.group(2))
        sets.append(('iso', D, 20, L, d, step(*prod_bracket[D], 1e-5)))
for d in glob.glob('data/aniso_weak/merged/ax8ay1_L*_D0.2_m0'):  # data/aniso_weak/prod/run.sh
    sets.append(('8x1', 0.2, 20, int(re.search(r'_L(\d+)_', d).group(1)), d, step(0.012, 0.026, 1e-5)))

rows = []
for B, D, K, L, d, st in sorted(sets):
    h = np.array([float(r['h_d']) for r in csv.DictReader(open(d + '/hd_summary.csv')) if r['seed'] != 'seed'])
    if len(h) < args.min_samples:
        print(f'skip {d}: {len(h)} samples')
        continue
    var_obs = h.var(ddof=1)
    sig = np.sqrt(max(var_obs - st ** 2 / 12, 0))
    boot = [np.sqrt(max(b.var(ddof=1) - st ** 2 / 12, 0)) for b in (rng.choice(h, len(h)) for _ in range(args.nboot))]
    rows.append(dict(B=B, D=D, K=K, L=L, n=len(h), mean=h.mean(), sig=sig, err=np.std(boot),
                     step=st, sheppard=st ** 2 / 12 / var_obs, distinct=len(np.unique(h))))

hdr = f"{'dis':>4} {'Delta':>5} {'kappa':>5} {'L':>6} {'n':>3} {'<h_d>':>9} {'sigma':>9} {'err':>8} {'sigma/<h>':>9} {'step':>8} {'Shep %':>6} {'distinct':>8}"
lines = [hdr]
for r in rows:
    lines.append(f"{r['B']:>4} {r['D']:>5} {r['K']:>5} {r['L']:>6} {r['n']:>3} {r['mean']:>9.6f} {r['sig']:>9.3e} {r['err']:>8.1e}"
                 f" {100 * r['sig'] / r['mean']:>8.2f}% {r['step']:>8.1e} {100 * r['sheppard']:>6.1f} {r['distinct']:>8}")


def fit(sel):  # weighted fit ln sigma = a - x ln L
    L = np.array([r['L'] for r in sel], float); s = np.array([r['sig'] for r in sel]); e = np.array([r['err'] for r in sel])
    w = (s / e) ** 2
    A = np.vstack([np.ones_like(L), -np.log(L)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, x = cov @ (A.T @ (w * np.log(s)))
    chi2 = np.sum(w * (np.log(s) - A @ [a, x]) ** 2)
    return a, x, np.sqrt(cov[1, 1]), chi2, len(sel) - 2


def label(g):
    return f"{g[0]} D={g[1]} k={g[2]}"


lines.append('')
lines.append('# fits ln sigma = a - (1/nu) ln L, per (Delta, kappa); expected 1/nu = 2 - zeta_q = 0.75')
groups = sorted({(r['B'], r['D'], r['K']) for r in rows})
fits = {}
for g in groups:
    sel = [r for r in rows if (r['B'], r['D'], r['K']) == g]
    if len(sel) < 3:
        continue
    a, x, ex, chi2, dof = fit(sel)
    fits[g] = (a, x)
    lines.append(f"{label(g):<18} L={min(r['L'] for r in sel)}-{max(r['L'] for r in sel)}"
                 f"  1/nu = {x:.3f} +- {ex:.3f}   chi2/dof = {chi2:.1f}/{dof}")
    sel = [r for r in sel if r['L'] >= 512]
    if len(sel) >= 3:
        a, x, ex, chi2, dof = fit(sel)
        lines.append(f"{'':18} L>=512      1/nu = {x:.3f} +- {ex:.3f}   chi2/dof = {chi2:.1f}/{dof}")

# qEW (1/nu = 2 - zeta = 0.75) vs qKPZ (directed-percolation depinning: no tilt
# symmetry, so nu = nu_par = 1.734, 1/nu = 0.577), slope fixed, one amplitude per
# (Delta, kappa); and a common free slope
lines.append('')
lines.append('# slope fixed at qEW 0.75 / qKPZ 0.577, free amplitude per (Delta, kappa)')
data = {g: [(r['L'], r['sig'], r['err']) for r in rows if (r['B'], r['D'], r['K']) == g] for g in groups}
for B in sorted({g[0] for g in groups}):  # totals and common slope per disorder type
    gs = [g for g in groups if g[0] == B]
    tot, dof = {0.75: 0., 0.577: 0.}, 0
    for g in gs:
        L, s, e = map(np.array, zip(*data[g]))
        y, w = np.log(s), (s / e) ** 2
        c2 = {}
        for x in tot:
            a = np.sum(w * (y + x * np.log(L))) / w.sum()
            c2[x] = np.sum(w * (y + x * np.log(L) - a) ** 2)
            tot[x] += c2[x]
        dof += len(L) - 1
        lines.append(f"{label(g):<18} n={len(L)}  chi2(0.75) = {c2[0.75]:5.1f}  chi2(0.577) = {c2[0.577]:5.1f}")
    lines.append(f"{B} total ({dof} dof): chi2(0.75) = {tot[0.75]:.1f}  chi2(0.577) = {tot[0.577]:.1f}")
    X, Y, W = [], [], []
    for i, g in enumerate(gs):
        for L, s, e in data[g]:
            row = np.zeros(len(gs) + 1); row[i] = 1; row[-1] = -np.log(L)
            X.append(row); Y.append(np.log(s)); W.append((s / e) ** 2)
    X, Y, W = map(np.array, (X, Y, W))
    cov = np.linalg.inv(X.T @ (X * W[:, None])); b = cov @ (X.T @ (W * Y))
    lines.append(f"{B} common free slope: 1/nu = {b[-1]:.3f} +- {np.sqrt(cov[-1, -1]):.3f}"
                 f"  chi2 = {np.sum(W * (Y - X @ b) ** 2):.1f}/{len(Y) - len(b)}")
lines.append('')
lines.append('# local slopes -d ln sigma / d ln L between successive L')
for g in groups:
    sel = sorted((r for r in rows if (r['B'], r['D'], r['K']) == g), key=lambda r: r['L'])
    for a, b in zip(sel, sel[1:]):
        x = -np.log(b['sig'] / a['sig']) / np.log(b['L'] / a['L'])
        ex = np.hypot(a['err'] / a['sig'], b['err'] / b['sig']) / np.log(b['L'] / a['L'])
        lines.append(f"{label(g):<18} L={a['L']}-{b['L']}: {x:.3f} +- {ex:.3f}")

# threshold shift <h_d>(L) = h_inf + A L^(-x), x fixed (qEW, qKPZ) or free
lines.append('')
lines.append('# <h_d>(L) = h_inf + A L^(-x), weighted by the standard errors')
for g in groups:
    sel = sorted((r for r in rows if (r['B'], r['D'], r['K']) == g), key=lambda r: r['L'])
    if len(sel) < 4:
        continue
    L = np.array([r['L'] for r in sel], float); m = np.array([r['mean'] for r in sel])
    e = np.array([r['sig'] / np.sqrt(r['n']) for r in sel])
    for x in (0.75, 0.577):
        A = np.vstack([np.ones_like(L), L ** -x]).T / e[:, None]
        c, *_ = np.linalg.lstsq(A, m / e, rcond=None); cov = np.linalg.inv(A.T @ A)
        lines.append(f"{label(g):<18} x={x}: h_inf = {c[0]:.6f} +- {np.sqrt(cov[0, 0]):.6f}  A = {c[1]:+.4f}"
                     f"  chi2 = {np.sum((m / e - A @ c) ** 2):.1f}/{len(L) - 2}")
    xs = np.linspace(0.1, 2, 381); c2 = []
    for x in xs:
        A = np.vstack([np.ones_like(L), L ** -x]).T / e[:, None]
        c, *_ = np.linalg.lstsq(A, m / e, rcond=None); c2.append(np.sum((m / e - A @ c) ** 2))
    c2 = np.array(c2); ok = xs[c2 <= c2.min() + 1]
    lines.append(f"{label(g):<18} free x = {xs[c2.argmin()]:.2f} (chi2+1: {ok.min():.2f}-{ok.max():.2f})"
                 f"  chi2 = {c2.min():.1f}/{len(L) - 3}")
print('\n'.join(lines))
(out / 'hd_variance.dat').write_text('#' + '\n'.join(lines) + '\n')

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
colors = {0.2: 'C0', 0.3: 'C1', 1.0: 'C2'}  # isotropic; 8x1 blocks in C3
marks = {15: 'o', 20: '*', 25: 's', 40: '^'}
for g in groups:
    sel = [r for r in rows if (r['B'], r['D'], r['K']) == g]
    L = np.array([r['L'] for r in sel]); s = np.array([r['sig'] for r in sel]); e = np.array([r['err'] for r in sel])
    m = np.array([r['mean'] for r in sel])
    kw = dict(color='C3' if g[0] == '8x1' else colors.get(g[1], 'k'), marker=marks.get(g[2], 'x'), ms=9 if g[2] == 20 else 5,
              ls='-' if len(sel) > 1 else 'none', lw=0.8,
              label=f"{'8x1 blocks, ' if g[0] == '8x1' else ''}Δ={g[1]}, κ={g[2]}")
    ax[0].errorbar(L, s / m, e / m, **kw)
    ax[1].errorbar(L, s / m * L ** 0.75, e / m * L ** 0.75, **kw)
Lr = np.array([200, 20000])
ax[0].plot(Lr, 0.3 * (Lr / 200.) ** -0.75, 'k--', lw=1, label='$L^{-0.75}$')
ax[0].plot(Lr, 0.05 * (Lr / 200.) ** -0.577, 'k:', lw=1, label='$L^{-0.577}$')
ax[0].set(xscale='log', yscale='log', xlabel='L', ylabel=r'$\sigma(h_d)/\langle h_d\rangle$',
          title='sample-to-sample spread of $h_d$')
ax[1].set(xscale='log', yscale='log', xlabel='L', ylabel=r'$L^{0.75}\,\sigma(h_d)/\langle h_d\rangle$',
          title='rescaled by $L^{2-\\zeta_q}$')
ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(out / 'hd_variance.png', dpi=130)
print(f'plot: {out}/hd_variance.png')
