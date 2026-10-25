#!/usr/bin/env python3
"""S(q) of moving walls at several fields, one L: the crossover from the
critical regimes (Larkin, qEW) at small scales to fast flow (EW) at
large scales, which moves to larger scales as h -> h_c.

Reads the Sq_L<L>.dat that vdrive_sq.py writes, one directory per field,
named ..._h<h> (as in docs/vdrive.md), bins them logarithmically in q
(sq_regimes.logbin) and plots
  - S(q) q^3.5, flat in the qEW regime,
  - the local slope -d ln S / d ln q against l = 2 pi / q, with the
    references 1 + 2 zeta: 2 (EW), 3.5 (qEW), 4 (Larkin).
Fields are coloured from light (far above h_c) to dark (closest).

  python3 scripts/vdrive_sq_plot.py results/vdrive_beta/Sq_ff_L2048_h* --L 2048 --hc 0.01339 \
      --out results/vdrive_beta/Sq_L2048.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sq_regimes import logbin

REFS = {'EW': 2.0, 'qEW': 3.5, 'Larkin': 4.0}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('sqdirs', nargs='+')
p.add_argument('--L', type=int, required=True)
p.add_argument('--hc', type=float, required=True)
p.add_argument('--per-decade', type=int, default=8)
p.add_argument('--out', required=True)
args = p.parse_args()

runs = []
for d in args.sqdirs:
    h = float(re.search(r'_h([\d.eE+-]+)$', d.rstrip('/')).group(1))
    a = np.loadtxt(Path(d) / f'Sq_L{args.L}.dat')
    a = a[a[:, 0] > 0]
    runs.append((h, *logbin(a[:, 0], a[:, 1], a[:, 2], args.per_decade)))
runs.sort(key=lambda r: -r[0])

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
cmap = plt.get_cmap('Blues')
for i, (h, q, S, e) in enumerate(runs):
    c = cmap(0.35 + 0.65 * i / max(1, len(runs) - 1))
    lab = f'ε = {h / args.hc - 1:.3g}'
    ax[0].plot(q, S * q ** 3.5, '-', color=c, lw=2, label=lab)
    lq, lS = np.log(q), np.log(S)
    slope = -np.diff(lS) / np.diff(lq)
    ax[1].plot(2 * np.pi / np.exp(0.5 * (lq[1:] + lq[:-1])), slope, '-', color=c, lw=2, label=lab)
for name, x in REFS.items():
    ax[1].axhline(x, color='0.6', lw=0.8, ls='--')
    ax[1].text(1.02, x, name, transform=ax[1].get_yaxis_transform(), va='center', fontsize=8, color='0.35')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('q'); ax[0].set_ylabel(r'$S(q)\,q^{3.5}$')
ax[0].set_title(f'L = {args.L}, h_c = {args.hc:g}', fontsize=10)
ax[1].set_xscale('log'); ax[1].set_ylim(1, 6)
ax[1].set_xlabel(r'$l = 2\pi/q$'); ax[1].set_ylabel(r'$-d\ln S/d\ln q$')
ax[1].legend(fontsize=8, frameon=False)
for a in ax:
    a.grid(True, which='major', color='0.9', lw=0.6)
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(args.out, dpi=130)
print(f'wrote {args.out}')
