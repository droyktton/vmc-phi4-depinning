#!/usr/bin/env python3
"""beta and the equivalence of ensembles from the runs of make_vdyn_jobs.py.

Reads <root>/ff_L<L>/h<h>/vdyn_h*_seed*.dat (fixed force: v = slope of
u(t) over the second half of each run) and <root>/vd_L<L>_H<H0>/v<v>/
vdyn_v*_seed*.dat (velocity-driven: <F> over the second half), averaged
over seeds with errors from their spread. Then:
- beta: v = A (h - h_c)^beta fitted to the fixed-force points of each L,
  with h_c fixed (--hc) and with h_c free;
- local beta: d ln v / d ln(h - h_c) between successive fields;
- equivalence: for each velocity-driven point, the fixed-force field at
  the same v, h_ff(v), interpolated in the ff curve (log-log), against
  <F>; the relative difference should vanish as L_m = sqrt(0.47/H0) grows.
  Velocity-driven points below the ff range (the extra small v) only
  enter the v(h) plot.
Writes beta.txt and v_of_h.png into --outdir.

  python3 scripts/vdrive_beta.py data/vdrive_beta --outdir results/vdrive_beta
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('root')
p.add_argument('--hc', type=float, default=0.01339)
p.add_argument('--outdir', default='results/vdrive_beta')
args = p.parse_args()
root = Path(args.root)
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)


def mean_err(x):
    x = np.asarray(x)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan), len(x)


def second_half(f):
    a = np.loadtxt(f)
    return a[len(a) // 2:] if a.ndim == 2 else None


ff = {}   # L -> array of (h, v, err, n)
for d in sorted(glob.glob(str(root / 'ff_L*'))):
    L = int(re.search(r'ff_L(\d+)', d).group(1))
    rows = []
    for hd in glob.glob(f'{d}/h*'):
        vs = [np.polyfit(a[:, 0], a[:, 1], 1)[0] for a in map(second_half, glob.glob(f'{hd}/vdyn_h*_seed*.dat'))
              if a is not None and len(a) > 10]
        if vs:
            rows.append((float(Path(hd).name[1:]), *mean_err(vs)))
    ff[L] = np.array(sorted(rows))

vd = {}   # (L, H0) -> array of (v, F, err, n)
for d in sorted(glob.glob(str(root / 'vd_L*_H*'))):
    m = re.search(r'vd_L(\d+)_H([-\d.eE+]+)$', d)
    L, H0 = int(m.group(1)), float(m.group(2))
    rows = []
    for vdir in glob.glob(f'{d}/v*'):
        Fs = [a[:, 3].mean() for a in map(second_half, glob.glob(f'{vdir}/vdyn_v*_seed*.dat'))
              if a is not None and len(a) > 10]
        if Fs:
            rows.append((float(Path(vdir).name[1:]), *mean_err(Fs)))
    vd[(L, H0)] = np.array(sorted(rows))

lines = []
npan = 3 if any(len(b_) for b_ in vd.values()) else 2   # the equivalence panel needs stage vd
fig, ax = plt.subplots(1, npan, figsize=(5 * npan, 4.2))
for i, L in enumerate(sorted(ff)):
    a = ff[L]
    moving = a[a[:, 1] > 0]
    lines.append(f'L={L}  fixed force: ' + ', '.join(f'h={h:.5g} v={v:.4g}+-{e:.1g}' for h, v, e, n in a))
    x = moving[:, 0] - args.hc
    if len(moving) >= 3 and np.all(x > 0):
        b, lnA = np.polyfit(np.log(x), np.log(moving[:, 1]), 1)
        lines.append(f'  beta (h_c = {args.hc:g} fixed) = {b:.3f}')
        hcs = np.linspace(args.hc * 0.5, moving[:, 0].min() * 0.9999, 2000)
        res = [np.sum(np.polyfit(np.log(moving[:, 0] - hc), np.log(moving[:, 1]), 1, full=True)[1]) for hc in hcs]
        hc_free = hcs[int(np.argmin(res))]
        b_free = np.polyfit(np.log(moving[:, 0] - hc_free), np.log(moving[:, 1]), 1)[0]
        lines.append(f'  beta (h_c free) = {b_free:.3f} at h_c = {hc_free:.5g}')
        # local exponent between successive fields: a single power law only fits if it is flat
        bl = np.diff(np.log(moving[:, 1])) / np.diff(np.log(x))
        lines.append('  local beta (h_c fixed): ' + ', '.join(f'{b:.3f}' for b in bl))
        ax[1].plot(np.sqrt(x[1:] * x[:-1]) / args.hc, bl, 's-', mfc='none', color=f'C{i}', label=f'L={L}')
    ax[0].errorbar(x, moving[:, 1], yerr=moving[:, 2], fmt='s', mfc='none', color=f'C{i}', label=f'fixed force, L={L}')
    for j, (key, b_) in enumerate(sorted((k, v) for k, v in vd.items() if k[0] == L)):
        if len(b_):
            ax[0].errorbar(b_[:, 1] - args.hc, b_[:, 0], xerr=b_[:, 2], fmt='o', ms=3, color=f'C{i}',
                           alpha=0.4 + 0.2 * j, label=f'v-driven, L={L}, H0={key[1]:g}')
    # equivalence: <F> against h_ff(v), within the ff range
    for key in sorted(k for k in vd if k[0] == L):
        b_ = vd[key]
        Lm = np.sqrt(2 * np.sqrt(2) / 3 / (2 * key[1]))
        # the v of the velocity-driven runs are the ff ones rounded to 6 digits
        lo, hi = moving[:, 1].min() * (1 - 1e-5), moving[:, 1].max() * (1 + 1e-5)
        inside = b_[(b_[:, 0] >= lo) & (b_[:, 0] <= hi)] if len(moving) >= 2 else []
        for v, F, e, n in inside:
            h_ff = np.exp(np.interp(np.log(v), np.log(moving[:, 1]), np.log(moving[:, 0])))
            lines.append(f'  H0={key[1]:g} (L_m={Lm:.0f}) v={v:.4g}: <F>={F:.5g}+-{e:.1g}  h_ff={h_ff:.5g}  '
                         f'diff={100 * (F - h_ff) / h_ff:+.2f}%')
            ax[2].errorbar(Lm, 100 * (F - h_ff) / h_ff, yerr=100 * e / h_ff, fmt='o', color=f'C{i}', alpha=0.6)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel(f'h - h_c  (h_c = {args.hc:g})'); ax[0].set_ylabel('v')
ax[0].legend(fontsize=7)
ax[1].axhline(0.25, color='0.5', lw=0.8, ls='--'); ax[1].text(1.0, 0.25, ' qEW', transform=ax[1].get_yaxis_transform(), va='center', fontsize=8)
ax[1].set_xscale('log'); ax[1].set_xlabel('ε = (h - h_c)/h_c'); ax[1].set_ylabel('local β = d ln v / d ln(h - h_c)')
ax[1].set_ylim(0, 1); ax[1].legend(fontsize=7)
if npan == 3:
    ax[2].axhline(0, color='0.5', lw=0.8)
    ax[2].set_xscale('log'); ax[2].set_xlabel('L_m'); ax[2].set_ylabel('(<F> - h_ff(v)) / h_ff  [%]')
    ax[2].set_title('equivalence of ensembles')
fig.tight_layout()
fig.savefig(out / 'v_of_h.png', dpi=130)
(out / 'beta.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'wrote {out}/beta.txt and v_of_h.png')
