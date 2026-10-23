"""Force-velocity curve of the finite-velocity velocity-driven runs (main_vdyn.cu).

Each vdyn_v*_seed*.dat has one line per output time:
  t W u F
with F = h_u + H0 (W - u) the drive. After discarding the first --skip
fraction of each run (the transient), per file: <F>, Var F and the
autocorrelation time of F; then, grouped by v, the pooled <F> with errors
from the spread between seeds. With --qs, the quasi-static runs of
main_vd.cu (vd_seed*.dat, same columns as vdrive_stats.py reads) give the
v -> 0 point. The clean-wall mobility line v/2.1213 (continuum phi^4,
c = eps0 = 1) is drawn for reference.

  python3 scripts/vdyn_stats.py runs/vdyn_v*_seed*.dat --qs runs/vd_seed*.dat --outdir results/vdyn"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('files', nargs='+')
p.add_argument('--skip', type=float, default=0.5, help='fraction of each run discarded as transient')
p.add_argument('--qs', nargs='*', default=[], help='quasi-static vd_seed*.dat files (v -> 0 point)')
p.add_argument('--qs-skip', type=int, default=400, help='transient steps of the quasi-static runs')
p.add_argument('--outdir', default='results/vdyn')
args = p.parse_args()
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)


def header(path):
    with open(path) as f:
        line = f.readline()
    return dict(kv.split('=') for kv in line[1:].split())


def autocorr_time(f):
    """integrated autocorrelation time, in samples, summed up to the first zero"""
    f = f - f.mean()
    if f.var() == 0:
        return 1.0
    n = len(f)
    c = np.correlate(f, f, mode='full')[n - 1:] / (f.var() * np.arange(n, 0, -1))
    k = np.argmax(c <= 0) if np.any(c <= 0) else n
    return max(1.0, 1 + 2 * c[1:k].sum())


by_v = defaultdict(list)
for path in args.files:
    d = np.loadtxt(path)
    if d.ndim < 2 or len(d) < 10:
        print(f'{path}: too short, skipped')
        continue
    h = header(path)
    d = d[int(args.skip * len(d)):]
    F = d[:, 3]
    tau = autocorr_time(F) * float(h['tout'])
    by_v[float(h['v'])].append(F.mean())
    print(f'{Path(path).name}: v={h["v"]} <F>={F.mean():.6g} sd(F)={F.std():.3g} '
          f'tau_F={tau:.3g} (t from {d[0, 0]:g} to {d[-1, 0]:g}, {(d[-1, 0] - d[0, 0]) / tau:.0f} tau)')

rows = []
for v in sorted(by_v):
    m = np.array(by_v[v])
    rows.append((v, m.mean(), m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.nan, len(m)))

qs = None
if args.qs:
    qm = np.array([np.loadtxt(f)[args.qs_skip:, 3].mean() for f in args.qs])
    qs = (qm.mean(), qm.std(ddof=1) / np.sqrt(len(qm)) if len(qm) > 1 else np.nan, len(qm))

print('\n    v        <F>        err   seeds')
if qs:
    print(f'  0+ (qs) {qs[0]:.6f}  {qs[1]:.2g}  {qs[2]}')
for v, m, e, n in rows:
    print(f'  {v:<8g} {m:.6f}  {e:.2g}  {n}')
np.savetxt(out / 'F_of_v.dat', np.array(rows), header='v <F> err nseeds')

fig, ax = plt.subplots(figsize=(5.5, 4))
vs = np.array([r[0] for r in rows])
ax.errorbar(vs, [r[1] for r in rows], yerr=[0 if np.isnan(r[2]) else r[2] for r in rows], fmt='o', label='Euler')
if qs:
    ax.axhline(qs[0], color='C1', lw=1, label='quasi-static (VMC)')
if len(vs):
    vv = np.logspace(np.log10(vs.min()), np.log10(vs.max()), 50)
    ax.plot(vv, vv / 2.1213, 'k:', lw=1, label='clean wall v/2.12')
ax.set_xscale('log')
ax.set_xlabel('v')
ax.set_ylabel('<F>')
ax.legend()
fig.tight_layout()
fig.savefig(out / 'F_of_v.png', dpi=130)
print(f'wrote {out}/F_of_v.png and F_of_v.dat')
