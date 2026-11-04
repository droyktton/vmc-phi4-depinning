"""Statistics of the quasi-static velocity-driven runs (main_vd.cu).

Each vd_seed*.dat has one line per step of the well center W:
  step W u F S sweeps converged
with F = h_u + H0 (W - u) the drive and S the avalanche (change of the
phi>0 area, in sites). After discarding the first --skip steps (the
transient until F reaches its stationary value), per file and pooled:
- <F> and Var F, with errors from the spread between seeds;
- the connected correlator C(w) = <F(W) F(W+w)>_c; in the parabolic-well
  ensemble Lx C(w) estimates the renormalized disorder correlator Delta(w)
  of the FRG (Le Doussal, Wiese), up to normalization conventions;
- P(S) of the avalanches S > 0, log-binned.

  python3 scripts/vdrive_stats.py runs/vd_seed*.dat --skip 200 --outdir results/vdrive"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('files', nargs='+')
p.add_argument('--skip', type=int, default=200, help='transient steps to discard')
p.add_argument('--maxlag', type=float, default=None, help='largest w for C(w), in rows (default: a quarter of the run)')
p.add_argument('--outdir', default='results/vdrive')
args = p.parse_args()
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)


def header(path):
    with open(path) as f:
        line = f.readline()
    return dict(kv.split('=') for kv in line[1:].split())


runs = []
for path in args.files:
    d = np.loadtxt(path)
    if d.ndim < 2 or len(d) <= args.skip + 2:
        print(f'{path}: too short, skipped')
        continue
    d = d[args.skip:]
    if not np.all(d[:, 6] == 1):
        print(f'{path}: {np.sum(d[:, 6] != 1)} steps not converged')
    if np.any(d[1:, 4] < 0):
        print(f'{path}: {np.sum(d[1:, 4] < 0)} steps with S < 0 (no-passing violated?)')
    runs.append((path, header(path), d))
if not runs:
    raise SystemExit('no usable runs')

hdr = runs[0][1]
dW = float(hdr['dW'])
Lx = int(hdr['Lx'])
H0 = float(hdr['H0'])

means = np.array([d[:, 3].mean() for _, _, d in runs])
vars_ = np.array([d[:, 3].var() for _, _, d in runs])
n = len(runs)
for (path, _, d), m, v in zip(runs, means, vars_):
    print(f'{Path(path).name}: steps={len(d)} <F>={m:.6g} sd(F)={np.sqrt(v):.4g} sum S/Lx={d[1:, 4].sum() / Lx:.1f} rows')
err = lambda a: a.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
print(f'pooled over {n} seeds: <F> = {means.mean():.6g} +- {err(means):.2g}, '
      f'Var F = {vars_.mean():.4g} +- {err(vars_):.2g}  (Lx={Lx}, H0={H0:g}, L_m={np.sqrt(2 * np.sqrt(2) / 3 / (2 * H0)):.0f})')

# connected correlator in w = lag*dW, averaged over seeds
nmin = min(len(d) for _, _, d in runs)
maxlag = int((args.maxlag / dW) if args.maxlag else nmin // 4)
maxlag = min(maxlag, nmin - 1)
C = np.zeros(maxlag + 1)
for _, _, d in runs:
    f = d[:, 3] - d[:, 3].mean()
    C += np.array([np.mean(f[:len(f) - k] * f[k:]) for k in range(maxlag + 1)])
C /= n
w = np.arange(maxlag + 1) * dW

# avalanches
S = np.concatenate([d[1:, 4] for _, _, d in runs])
S = S[S > 0]

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for path, _, d in runs[:5]:
    ax[0].plot(d[:, 1], d[:, 3], lw=0.6)
ax[0].set_xlabel('W (rows)')
ax[0].set_ylabel('F')
ax[0].set_title('drive (first seeds)')
ax[1].plot(w, C / C[0])
ax[1].axhline(0, color='0.6', lw=0.5)
ax[1].set_xlabel('w (rows)')
ax[1].set_ylabel('C(w)/C(0)')
ax[1].set_title('connected correlator of F')
if len(S):
    bins = np.logspace(0, np.log10(S.max() + 1), 30)
    h, e = np.histogram(S, bins=bins, density=True)
    c = np.sqrt(e[1:] * e[:-1])
    ax[2].loglog(c[h > 0], h[h > 0], 'o', ms=3)
    ax[2].set_xlabel('S (sites)')
    ax[2].set_ylabel('P(S)')
    ax[2].set_title(f'avalanches ({len(S)})')
fig.tight_layout()
fig.savefig(out / 'vdrive_stats.png', dpi=130)
np.savetxt(out / 'correlator.dat', np.column_stack([w, C]), header='w C(w)')
print(f'wrote {out}/vdrive_stats.png and correlator.dat')
