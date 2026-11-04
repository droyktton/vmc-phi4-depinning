"""Threshold finite-size scaling for 4x1, 8x1 and 16x1 blocks, and isotropic disorder (1x1) as control, at
Delta = 0.2, L = 256-2048, 200 seeds per L (§5.10; docs/plans/aniso_crossover_nu.md, stage 4).
Per shape: joint fit of <h_d> = h_inf + A L^-x and sigma = B L^-x (as data/aniso_weak/kappa_sens/joint_fit.py), local slopes between successive L, and kappa_eff = P/<w>.
The local slopes are plotted against L/l_p, with l_p where zeta_W (chord) first drops below 0.8 at L=2048,
and l_x where it drops below 1.125 (results/aniso_nu/roughness_ax*_th*/lo_table.dat).
Reads data/aniso_nu/merged/ax{1,4,16}ay1_L*_D0.2_m0 (h_d of all seeds, fields of seeds 1-30, merged by
data/aniso_nu/analyze.sh) and the 8x1 runs of §5.9 in data/aniso_weak/.

  srun -p cpu --mem=8G python3 scripts/shape_nu.py"""
import csv, glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts')
from wall_contour import read_field, wall_curve, univalued_profile
R = 'results/aniso_nu'
rng = np.random.default_rng(2)
Ls = [256, 512, 1024, 2048]
src = {1: 'data/aniso_nu/merged/ax1ay1', 4: 'data/aniso_nu/merged/ax4ay1', 8: 'data/aniso_weak/merged_hd/ax8ay1', 16: 'data/aniso_nu/merged/ax16ay1'}
fld = {1: 'data/aniso_nu/merged/ax1ay1', 4: 'data/aniso_nu/merged/ax4ay1', 8: 'data/aniso_weak/merged/ax8ay1', 16: 'data/aniso_nu/merged/ax16ay1'}
Pbox = {1: dict(zip(Ls, (88, 210, 498, 1185))), 4: dict(zip(Ls, (85, 147, 261, 547))), 8: dict(zip(Ls, (74, 113, 200, 271))), 16: dict(zip(Ls, (55, 86, 130, 219)))}

def lo(ax, th, L=2048):
    for r in open(f'{R}/roughness_ax{ax}_th{th}/lo_table.dat'):
        if not r.startswith('#') and int(r.split()[0]) == L:
            return float(r.split()[8]), float(r.split()[9])
    return np.nan, np.nan

def wls(X, y, e):
    W = 1 / e
    c, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    return c, np.sum(((X @ c - y) * W) ** 2)

def chi2(x, d):
    L, m, em, s, es = d[:, 0], d[:, 2], d[:, 3], d[:, 4], d[:, 5]
    cm, c1 = wls(np.c_[np.ones_like(L), L ** -x], m, em)
    cs, c2 = wls((L ** -x)[:, None], s, es)
    return c1, c2, cm, cs

xs = np.linspace(0.3, 1.2, 9001)
def best(d, k=None):
    c = np.array([sum(chi2(x, d)[:2]) if k is None else chi2(x, d)[k] for x in xs])
    i = c.argmin(); ok = xs[c <= c[i] + 1]
    return xs[i], ok.min(), ok.max(), c[i]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
cols = {1: 'k', 4: 'C0', 8: 'C3', 16: 'C2'}
for A in (1, 4, 8, 16):
    D = []
    for L in Ls:
        x = np.array([float(r['h_d']) for r in csv.DictReader(open(f'{src[A]}_L{L}_D0.2_m0/hd_summary.csv'))])
        b = np.array([rng.choice(x, len(x)).std(ddof=1) for _ in range(4000)])
        w = [univalued_profile(wall_curve(read_field(f, L))[0], L)[0].std()
             for f in sorted(glob.glob(f'{fld[A]}_L{L}_D0.2_m0/criticafield_*.bin'))[:30]]
        D.append((L, len(x), x.mean(), x.std(ddof=1) / np.sqrt(len(x)), x.std(ddof=1), b.std(), Pbox[A][L] / np.mean(w)))
    D = np.array(D)
    lx, lp = lo(A, 1.125), lo(A, 0.8)
    print(f"\n=== {A}x1   l_x = {lx[0]:.0f} +- {lx[1]:.0f}   l_p = {lp[0]:.0f} +- {lp[1]:.0f}   (L=2048, chord)")
    print("    L    n     <h_d>       err      sigma      err  kappa_eff")
    for L, n, m, em, s, es, k in D:
        print(f"{L:5.0f} {n:4.0f}  {m:.6f} {em:.1e}  {s:.3e} {es:.1e}  {k:5.1f}")
    x0, lo_, hi_, c = best(D)
    _, _, cm, _ = chi2(x0, D)
    print(f"  joint: 1/nu = {x0:.3f} ({lo_:.3f}-{hi_:.3f})  chi2 = {c:.1f}/{2*len(D)-4}  h_inf = {cm[0]:.6f}")
    for xx, nm in [(0.577, 'qKPZ'), (0.75, 'qEW')]:
        a, b, _, _ = chi2(xx, D); print(f"    x = {xx} ({nm}): chi2 = {a+b:.1f}/{2*len(D)-3}")
    for nm, k in [('shift', 0), ('sigma', 1)]:
        xb, l1, h1, cb = best(D, k); print(f"  {nm} only: x = {xb:.3f} ({l1:.3f}-{h1:.3f})  chi2 = {cb:.1f}")
    L, s, es, m, em = D[:, 0], D[:, 4], D[:, 5], D[:, 2] - cm[0], D[:, 3]
    Lm = np.sqrt(L[1:] * L[:-1])
    ss = -np.log(s[1:] / s[:-1]) / np.log(2); ess = np.hypot(es[1:] / s[1:], es[:-1] / s[:-1]) / np.log(2)
    sm = -np.log(m[1:] / m[:-1]) / np.log(2); esm = np.hypot(em[1:] / m[1:], em[:-1] / m[:-1]) / np.log(2)
    print("  local slopes (sigma): " + ", ".join(f"{a:.2f}+-{b:.2f}" for a, b in zip(ss, ess)))
    print("  local slopes (shift): " + ", ".join(f"{a:.2f}+-{b:.2f}" for a, b in zip(sm, esm)))
    ax[0].errorbar(L, s, es, fmt='o-', c=cols[A], label=f'{A}x1, 1/nu = {x0:.2f}')
    for k, (y, e, mk) in enumerate([(ss, ess, 'o'), (sm, esm, 's')]):
        ax[1].errorbar(Lm * (1 + 0.03 * k), y, e, fmt=mk, c=cols[A], mfc='none' if k else cols[A],
                       label=f'{A}x1' + (' shift' if k else ' sigma'))
        ax[2].errorbar(Lm / lp[0] * (1 + 0.03 * k), y, e, fmt=mk, c=cols[A], mfc='none' if k else cols[A])
ax[0].set(xscale='log', yscale='log', xlabel='L', ylabel='sigma(h_d)', title='spread, 200 seeds per L'); ax[0].legend(fontsize=8)
ax[1].set(xscale='log', xlabel='L (geometric mean of pair)', ylabel='local exponent', title='local slopes vs L', ylim=(0, 1.4))
ax[2].set(xscale='log', xlabel='L / l_p', title='local slopes vs L / l_p  (l_p: zeta_W = 0.8)', ylim=(0, 1.4))
ax[1].legend(fontsize=7, ncol=2)
for a in ax[1:]:
    a.axhline(0.577, c='k', ls=':', lw=.8); a.axhline(0.75, c='k', ls='--', lw=.8)
fig.tight_layout(); fig.savefig(f'{R}/shape_nu.png', dpi=120)
print(f"\nplot: {R}/shape_nu.png")
