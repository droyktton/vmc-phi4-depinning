#!/usr/bin/env python3
"""Detached domains (bubbles) in dumped fields: a check that the runs
describe a single domain wall and no domains nucleated away from it.

In every snapshot (main_vd.cu vdfield_step<k>_seed<s>.bin, main_vdyn.cu
vdynfield_t<t>[_y<y0>]_seed<s>.bin, or main.cu criticafield_*.bin) the
phi > 0 and phi < 0 sites are split into connected components
(4-neighbour, periodic in x). The largest component of each sign is its
bulk; every other component is a bubble:
  - ahead:  a phi > 0 domain detached from the phi > 0 bulk below the
            wall (nucleated in front of it, or pinched off an overhang);
  - behind: a phi < 0 domain detached from the phi < 0 bulk above it
            (left behind by the wall).
A bubble touching the first or last row is reported as "edge": in a
band dump (--dump-band) it may be part of a bulk outside the band.

For each bubble the size (sites) and the distance of its centroid from
the mean wall row (phi > 0 area / Lx) are kept. Prints one line per
snapshot with bubbles, and a summary per run directory; with --csv, all
bubbles are written there.

  python3 scripts/bubbles.py data/qs/D0.85_H3e-5/ --L 1024 --min-size 4
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy import ndimage

from wall_contour import read_field

SNAP_RE = re.compile(r"(?:vd(?:yn)?field_(?:t|step)([-\d.eE+]+)(?:_y(-?\d+))?|criticafield_h([-\d.eE+]+))_seed(\d+)\.bin$")


def components(mask):
    """Connected components of mask (4-neighbour, periodic in x) as a
    label array and their sizes; labels run from 1."""
    lab, n = ndimage.label(mask)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(lab[:, 0], lab[:, -1]):
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(n + 1)])
    _, merged = np.unique(roots, return_inverse=True)   # root 0 (background) stays 0
    lab = merged[lab]
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab, sizes


def bubbles(phi, min_size=1):
    """List of (kind, size, dy, edge) for the detached domains of phi;
    dy is the centroid row minus the mean wall row."""
    Ly, L = phi.shape
    wall_row = np.count_nonzero(phi > 0) / L
    out = []
    for kind, mask in (('ahead', phi > 0), ('behind', phi < 0)):
        lab, sizes = components(mask)
        if sizes.max() == 0:
            continue
        bulk = sizes.argmax()
        for i in np.nonzero(sizes)[0]:
            if i == bulk or sizes[i] < min_size:
                continue
            ys = np.nonzero(lab == i)[0]
            edge = ys.min() == 0 or ys.max() == Ly - 1
            out.append((kind, int(sizes[i]), ys.mean() - wall_row, edge))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rundirs', nargs='+', help='directories with dumped fields')
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--min-size', type=int, default=1, help='ignore bubbles smaller than this (sites)')
    p.add_argument('--csv', help='write every bubble here')
    args = p.parse_args()

    rows = []
    for d in args.rundirs:
        nsnap = nwith = 0
        sizes = []
        for path in sorted(Path(d).glob('*field_*_seed*.bin')):
            m = SNAP_RE.search(path.name)
            if not m:
                continue
            nsnap += 1
            found = bubbles(read_field(path, args.L), args.min_size)
            if not found:
                continue
            nwith += 1
            for kind, size, dy, edge in found:
                rows.append((str(path), kind, size, dy, edge))
                sizes.append(size)
            desc = ', '.join(f'{k} {s} sites dy={dy:+.0f}{" (edge)" if e else ""}' for k, s, dy, e in found)
            print(f'{path.name}: {desc}')
        if nsnap:
            big = f', largest {max(sizes)} sites' if sizes else ''
            print(f'{d}: {nwith} of {nsnap} snapshots with bubbles (>= {args.min_size} sites), {len(sizes)} bubbles{big}')
        else:
            print(f'{d}: no snapshots')

    if args.csv:
        with open(args.csv, 'w') as f:
            f.write('file,kind,size,dy,edge\n')
            for path, kind, size, dy, edge in rows:
                f.write(f'{path},{kind},{size},{dy:.1f},{int(edge)}\n')
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
