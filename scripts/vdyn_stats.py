"""Force-velocity curve of the finite-velocity velocity-driven runs (main_vdyn.cu).

Each vdyn_v*_seed*.dat has one line per output time:
  t W u F
with F = h_u + H0 (W - u) the drive; the fixed-force runs (H0 = 0),
vdyn_h*_seed*.dat, have t u, and give v as the slope of u(t). Both are
plotted as v against the drive, which tests the equivalence of the two
ensembles. The mean drive of a velocity-driven run relaxes in about
1/(mu H0), mu ~ 2.12; a run whose discarded transient is shorter than
three times that is flagged. After discarding the first --skip
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


by_v = defaultdict(list)   # velocity-driven: v -> [<F> per seed]
by_h = defaultdict(list)   # fixed force (H0 = 0): h -> [v per seed]
for path in args.files:
    d = np.loadtxt(path)
    if d.ndim < 2 or len(d) < 10:
        print(f'{path}: too short, skipped')
        continue
    h = header(path)
    d = d[int(args.skip * len(d)):]
    if float(h['H0']) == 0:
        # fixed force: v is the slope of u(t); a wall that stopped gives v = 0
        v = np.polyfit(d[:, 0], d[:, 1], 1)[0]
        stuck = d[-1, 1] == d[len(d) // 2, 1]
        by_h[float(h['h_u'])].append(0.0 if stuck else v)
        print(f'{Path(path).name}: h={h["h_u"]} v={v:.5g}{" (stuck)" if stuck else ""} '
              f'(t from {d[0, 0]:g} to {d[-1, 0]:g}, u advanced {d[-1, 1] - d[0, 1]:.1f} rows)')
        continue
    F = d[:, 3]
    tau = autocorr_time(F) * float(h['tout'])
    # the mean drive relaxes towards its stationary value in ~1/(mu*H0),
    # mu ~ 2.12 (clean-wall mobility; slower with disorder)
    tau_k = 1 / (2.1213 * float(h['H0']))
    by_v[float(h['v'])].append(F.mean())
    print(f'{Path(path).name}: v={h["v"]} <F>={F.mean():.6g} sd(F)={F.std():.3g} '
          f'tau_F={tau:.3g} (t from {d[0, 0]:g} to {d[-1, 0]:g}, {(d[-1, 0] - d[0, 0]) / tau:.0f} tau)'
          + ('' if d[0, 0] > 3 * tau_k else f'  WARNING: transient discarded up to t={d[0, 0]:g} < 3/(mu H0) = {3 * tau_k:.3g}'))


def pooled(groups):
    rows = []
    for k in sorted(groups):
        m = np.array(groups[k])
        rows.append((k, m.mean(), m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.nan, len(m)))
    return rows


rows = pooled(by_v)
rows_h = pooled(by_h)

qs = None
if args.qs:
    qm = np.array([np.loadtxt(f)[args.qs_skip:, 3].mean() for f in args.qs])
    qs = (qm.mean(), qm.std(ddof=1) / np.sqrt(len(qm)) if len(qm) > 1 else np.nan, len(qm))

if rows or qs:
    print('\nvelocity-driven:   v        <F>        err   seeds')
    if qs:
        print(f'                  0+ (qs) {qs[0]:.6f}  {qs[1]:.2g}  {qs[2]}')
    for v, m, e, n in rows:
        print(f'                  {v:<8g} {m:.6f}  {e:.2g}  {n}')
    np.savetxt(out / 'F_of_v.dat', np.array(rows).reshape(-1, 4), header='v <F> err nseeds')
if rows_h:
    print('\nfixed force:       h        <v>        err   seeds')
    for hh, m, e, n in rows_h:
        print(f'                  {hh:<8g} {m:.6f}  {e:.2g}  {n}')
    np.savetxt(out / 'v_of_h.dat', np.array(rows_h), header='h <v> err nseeds')

# v against the drive, both ensembles on the same axes
fig, ax = plt.subplots(figsize=(5.5, 4))
if rows:
    ax.errorbar([r[1] for r in rows], [r[0] for r in rows], xerr=[0 if np.isnan(r[2]) else r[2] for r in rows],
                fmt='o', label='velocity-driven (<F>, v)')
if rows_h:
    ax.errorbar([r[0] for r in rows_h], [r[1] for r in rows_h], yerr=[0 if np.isnan(r[2]) else r[2] for r in rows_h],
                fmt='s', mfc='none', label='fixed force (h, <v>)')
if qs:
    ax.axvline(qs[0], color='C2', lw=1, label='quasi-static <F> (VMC)')
fmax = max([r[1] for r in rows] + [r[0] for r in rows_h] + [0.01])
ff = np.linspace(0, fmax, 50)
ax.plot(ff, 2.1213 * ff, 'k:', lw=1, label='clean wall v = 2.12 h')
ax.set_xlabel('drive (h or <F>)')
ax.set_ylabel('v')
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / 'v_of_h.png', dpi=130)
print(f'wrote {out}/v_of_h.png')
