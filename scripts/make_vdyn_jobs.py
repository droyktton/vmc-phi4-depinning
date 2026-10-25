#!/usr/bin/env python3
"""Job lists for scripts/slurm_vdyn.sh: the velocity exponent beta, the
equivalence of the force- and velocity-driven ensembles at finite v, and
S(q) of the moving walls, with main_vdyn.cu (Euler, T = 0).

Two stages, one job list each (one line per run: "<outdir> <arguments of
the binary>"):

  ff  fixed force (H0 = 0) at h = h_c (1 + eps): v(h) for beta, and the
      reference curve for the equivalence test.
  vd  velocity-driven, after ff has run: the v measured at each h,
      started at F0 = h, for several H0 (does <F> converge to h as
      L_m = sqrt(0.47/H0) grows?), plus smaller v below the ff range,
      started at F0 from a power-law fit of the ff points.

Run lengths: the wall travels at least --rows rows after a transient of
the same length, and a velocity-driven run lasts at least 8 relaxation
times of the spring, 1/(mu H0), with mu = dv/dh from the ff fit. Seeds
1..--dump-seeds dump --nsnap band snapshots (--dump-band) of the second
half, for S(q) (scripts/vdrive_sq.py).

  python3 scripts/make_vdyn_jobs.py ff --outroot data/vdrive_beta > jobs_ff.txt
  python3 scripts/make_vdyn_jobs.py vd --outroot data/vdrive_beta > jobs_vd.txt   # after ff
"""
import argparse
import glob
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('stage', choices=['ff', 'vd'])
p.add_argument('--outroot', default='data/vdrive_beta')
p.add_argument('--Ls', default='1024 2048')
p.add_argument('--Ly', default='512 1024', help='box height for each L (several wall widths plus the recentering margin)')
p.add_argument('--band', default='192 384', help='--dump-band for each L (half-height of the dumped band)')
p.add_argument('--hc', type=float, default=0.01339, help='threshold estimate (bisection, kappa = 20, L = 8192)')
p.add_argument('--eps', default='0.05 0.1 0.2 0.4 0.8 1.6', help='fixed-force fields h = hc (1 + eps)')
p.add_argument('--H0s', default='3e-5 1e-5 3e-6', help='springs of the velocity-driven runs')
p.add_argument('--extra-v', default='0.5 0.25', help='extra v below the ff range, as fractions of the smallest ff v')
p.add_argument('--seeds', type=int, default=10, help='seeds of stage ff')
p.add_argument('--vd-seeds', type=int, default=5, help='seeds of stage vd (<F> self-averages at fixed v)')
p.add_argument('--dump-seeds', type=int, default=5)
p.add_argument('--nsnap', type=int, default=20)
p.add_argument('--rows', type=float, default=800, help='rows travelled after the transient')
p.add_argument('--dt', type=float, default=0.05)
p.add_argument('--binary', default='./phi4vmc_vdyn_D0.3')
args = p.parse_args()

Ls = [int(x) for x in args.Ls.split()]
Lys = dict(zip(Ls, [int(x) for x in args.Ly.split()]))
bands = dict(zip(Ls, [int(x) for x in args.band.split()]))
root = Path(args.outroot)
NOUT = 4000  # outputs per run


def line(outdir, L, H0, h_u, v, T, seed, extra=''):
    tout = T / NOUT
    dump = ''
    if seed <= args.dump_seeds:
        every = max(1, NOUT // 2 // args.nsnap)
        dump = f' --dump-every {every} --dump-after {T / 2:g} --dump-band {bands[L]}'
    return (f'{outdir} {args.binary} {L} {Lys[L]} {H0:g} {h_u:.6g} {v:.6g} {args.dt:g} {T:.6g} {seed}'
            f' --tout {tout:.6g}{extra}{dump}')


def v_guess(h):
    # rough v(h) for the run lengths only: v/h_c against eps, from L = 256 at Delta = 0.3 (it
    # nearly collapses with Delta = 1.0); on the low side, so runs come out long rather than short
    return 0.035 / 0.01339 * args.hc * max(h / args.hc - 1, 1e-3) ** 0.75


if args.stage == 'ff':
    for L in Ls:
        for eps in [float(x) for x in args.eps.split()]:
            h = args.hc * (1 + eps)
            T = 2 * args.rows / v_guess(h)
            for s in range(1, args.seeds + 1):
                print(line(root / f'ff_L{L}' / f'h{h:.6g}', L, 0, h, 0, T, s))
else:
    for L in Ls:
        # measured v(h) of stage ff: slope of u(t) over the second half, averaged over seeds
        pts = []
        for d in sorted(glob.glob(str(root / f'ff_L{L}' / 'h*'))):
            h = float(Path(d).name[1:])
            vs = []
            for f in glob.glob(f'{d}/vdyn_h*_seed*.dat'):
                a = np.loadtxt(f)
                a = a[len(a) // 2:]
                if len(a) > 10:
                    vs.append(np.polyfit(a[:, 0], a[:, 1], 1)[0])
            if vs and np.mean(vs) > 0:
                pts.append((h, np.mean(vs)))
        if len(pts) < 3:
            raise SystemExit(f'L={L}: stage ff incomplete ({len(pts)} fields with v > 0)')
        pts = np.array(pts)
        # v = A (h - hc)^b over the ff points, for F0 and mu = dv/dh of the extra v
        b, lnA = np.polyfit(np.log(pts[:, 0] - args.hc), np.log(pts[:, 1]), 1)
        h_of_v = lambda v: args.hc + (v / np.exp(lnA)) ** (1 / b)
        vmin = pts[:, 1].min()
        targets = [(v, h) for h, v in pts] + [(f * vmin, h_of_v(f * vmin)) for f in map(float, args.extra_v.split())]
        for H0 in [float(x) for x in args.H0s.split()]:
            for v, h in targets:
                mu = b * v / (h - args.hc)  # dv/dh of the fit
                T = max(2 * args.rows / v, 8 / (mu * H0))
                for s in range(1, args.vd_seeds + 1):
                    print(line(root / f'vd_L{L}_H{H0:g}' / f'v{v:.6g}', L, H0, h, v, T, s))
