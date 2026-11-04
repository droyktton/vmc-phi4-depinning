#!/usr/bin/env python3
"""Compares two runs of scripts/bench_compare.sh (e.g. CUDA on one machine,
SYCL on another): for every (L, seed) in both, whether h_d agrees, whether
the critical walls (critica_*.dat) are identical, and how far apart the
critical fields (criticafield_*.bin) are -- largest |phi_a - phi_b| and the
number of sites whose sign differs -- plus each side's wall time. Prints a
Markdown table.

Usage: compare_runs.py DIR_A DIR_B [--names A B]
"""
import argparse
import csv
import glob
import os

import numpy as np


def read_bench(d):
    with open(os.path.join(d, "bench.csv")) as f:
        return {(int(r["L"]), int(r["seed"])): r for r in csv.DictReader(f)}


def one(pattern):
    files = glob.glob(pattern)
    return files[0] if len(files) == 1 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--names", nargs=2, default=["A", "B"])
    args = p.parse_args()
    na, nb = args.names
    ba, bb = read_bench(args.a), read_bench(args.b)

    print(f"| L | seed | h_d {na} | h_d {nb} | wall | max abs dphi | sign flips | t {na} (s) | t {nb} (s) | {na}/{nb} |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(set(ba) & set(bb)):
        L, seed = key
        ra, rb = ba[key], bb[key]
        da = os.path.join(args.a, f"L{L}_seed{seed}")
        db = os.path.join(args.b, f"L{L}_seed{seed}")
        wa, wb = one(f"{da}/critica_h*.dat"), one(f"{db}/critica_h*.dat")
        if wa and wb:
            same_wall = open(wa).read() == open(wb).read()
            wall = "identical" if same_wall else "differs"
        else:
            wall = "missing"
        fa, fb = one(f"{da}/criticafield_*.bin"), one(f"{db}/criticafield_*.bin")
        if fa and fb:
            pa = np.fromfile(fa, dtype=np.float32)
            pb = np.fromfile(fb, dtype=np.float32)
            dphi = f"{np.abs(pa - pb).max():.2e}"
            flips = str(int(np.count_nonzero(np.sign(pa) != np.sign(pb))))
        else:
            dphi = flips = "-"
        ta, tb = float(ra["seconds"]), float(rb["seconds"])
        print(f"| {L} | {seed} | {ra['h_d']} | {rb['h_d']} | {wall} | {dphi} | {flips} "
              f"| {ta:.1f} | {tb:.1f} | {ta / tb:.2f} |")


if __name__ == "__main__":
    main()
