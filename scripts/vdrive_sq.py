#!/usr/bin/env python3
"""Structure factor S(q) of the moving walls of the velocity-driven runs.

Reads the snapshots that main_vdyn.cu (vdynfield_t<t>[_y<y0>]_seed<s>.bin,
whole field or a band of rows, --dump-band) or main_vd.cu
(vdfield_step<k>_seed<s>.bin) dump into one or more run directories,
all for the same L and the same condition (v or h, H0). Each snapshot's
wall is traced as the phi = 0 level line (wall_contour.wall_curve, so
overhangs are kept) and reduced to the paper's univalued profile
(univalued_profile); S(q) = |FFT(u)|^2 as in structure_factor.py. S(q)
is averaged first over the snapshots of each seed, then over seeds,
with the error from the spread between seeds (snapshots of one run are
correlated). Snapshots before --tmin (time, or step for main_vd.cu) are
skipped. Writes <outdir>/Sq_L<L>.dat (q S_mean S_sem), the format
sq_regimes.py reads, so the outdirs of several L can be compared there.

  python3 scripts/vdrive_sq.py runs/ff_L1024_h0.0161/ --L 1024 --tmin 20000 --outdir results/vdrive_sq/ff_h0.0161
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from structure_factor import structure_factor
from wall_contour import read_field, univalued_profile, wall_curve

SNAP_RE = re.compile(r"vd(?:yn)?field_(?:t|step)([-\d.eE+]+)(?:_y(-?\d+))?_seed(\d+)\.bin$")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rundirs', nargs='+', help='directories with the snapshots of one condition')
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--tmin', type=float, default=0.0, help='skip snapshots before this time (or step)')
    p.add_argument('--outdir', required=True)
    args = p.parse_args()

    by_seed = defaultdict(list)
    nskip = nfail = 0
    for d in args.rundirs:
        for path in sorted(Path(d).glob('vd*field_*_seed*.bin')):
            m = SNAP_RE.search(path.name)
            if not m:
                continue
            if float(m.group(1)) < args.tmin:
                nskip += 1
                continue
            try:
                curve, _ = wall_curve(read_field(path, args.L))
                u, _ = univalued_profile(curve, args.L)
            except ValueError as e:
                nfail += 1
                print(f'{path.name}: {e}')
                continue
            q, S = structure_factor(u)
            by_seed[int(m.group(3))].append(S)
    if not by_seed:
        raise SystemExit('no usable snapshots')

    per_seed = np.array([np.mean(v, axis=0) for v in by_seed.values()])
    n = len(per_seed)
    S_mean = per_seed.mean(axis=0)
    S_sem = per_seed.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full_like(S_mean, np.nan)
    nsnap = sum(len(v) for v in by_seed.values())
    print(f'L={args.L}: {nsnap} snapshots from {n} seeds ({nskip} before tmin, {nfail} without a traceable wall)')

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'Sq_L{args.L}.dat'
    with open(path, 'w') as f:
        f.write(f'# q S_mean S_sem  ({nsnap} snapshots, {n} seeds, t >= {args.tmin:g})\n')
        for qi, si, ei in zip(q, S_mean, S_sem):
            f.write(f'{qi:.6g} {si:.6g} {ei:.6g}\n')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
