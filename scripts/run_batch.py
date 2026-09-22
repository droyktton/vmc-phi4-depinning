#!/usr/bin/env python3
"""Runs ./phi4vmc over many disorder-realization seeds for a fixed
(L, h_low, h_high, tol), collecting the per-sample depinning field h_d
into a summary CSV. Each run's output files (critical configuration,
avalanche statistics, ...) are kept in `outdir`, one run's files per
seed (named after the seed, so nothing gets overwritten).

Note: the disorder strength Delta is a compile-time constant (AMPDIS in
the Makefile), not a runtime argument -- rebuild with `make AMPDIS=...`
before running this script if you want a different Delta.

Example:
    ./run_batch.py --binary ../phi4vmc --L 128 --h-low 0 --h-high 0.05 \\
        --tol 1e-4 --nsamples 40 --outdir data/L128_D0.2
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

HD_LINE = re.compile(r"depinning field h_d = ([-\d.eE+]+)")


def run_one(binary, L, h_low, h_high, tol, seed, cwd):
    result = subprocess.run(
        [str(binary), str(L), str(h_low), str(h_high), str(tol), str(seed)],
        cwd=cwd, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"seed {seed} failed (exit {result.returncode}):\n{result.stderr}"
        )
    match = HD_LINE.search(result.stdout)
    if not match:
        raise RuntimeError(f"seed {seed}: could not find h_d in output")
    return float(match.group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", default="./phi4vmc", help="path to the compiled solver")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--h-low", type=float, required=True)
    p.add_argument("--h-high", type=float, required=True)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--nsamples", type=int, required=True)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        sys.exit(f"binary not found: {binary} (build it first with `make`)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "hd_summary.csv"
    done_seeds = set()
    if summary_path.exists():
        with open(summary_path) as f:
            done_seeds = {int(row["seed"]) for row in csv.DictReader(f)}

    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["seed", "h_d"])
        for i in range(args.nsamples):
            seed = args.seed_start + i
            if seed in done_seeds:
                print(f"seed {seed}: already done, skipping")
                continue
            hd = run_one(binary, args.L, args.h_low, args.h_high, args.tol, seed, cwd=outdir)
            print(f"seed {seed}: h_d = {hd:.6f}")
            writer.writerow([seed, hd])
            f.flush()

    print(f"\nsummary written to {summary_path}")


if __name__ == "__main__":
    main()
