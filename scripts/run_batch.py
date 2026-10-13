#!/usr/bin/env python3
"""Runs ./phi4vmc over many disorder-realization seeds for a fixed
(L, h_low, h_high, tol), collecting the per-sample depinning field h_d
into a summary CSV. Each run's output files (critical configuration,
and avalanche statistics if --with-magjumps is passed) are kept in
`outdir`, one run's files per seed (named after the seed, so nothing
gets overwritten).

Note: the disorder strength Delta is a compile-time constant (AMPDIS in
the Makefile), not a runtime argument -- rebuild with `make AMPDIS=...`
before running this script if you want a different Delta.

Example:
    ./run_batch.py --binary ../phi4vmc --L 128 --h-low 0 --h-high 0.05 \\
        --tol 1e-4 --nsamples 40 --outdir data/L128_D0.2
"""
import argparse
import csv
import fcntl
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HD_LINE = re.compile(r"depinning field h_d = ([-\d.eE+]+)")
LY_LINE = re.compile(r"Lx=\d+ Ly=(\d+)")


def run_one(binary, L, h_low, h_high, tol, seed, cwd, with_magjumps=False, dump_field=False, verbose=True,
            box_args=()):
    """Returns (h_d, the solver's WARNING lines, Ly)."""
    cmd = [str(binary), str(L), str(h_low), str(h_high), str(tol), str(seed), *box_args]
    if with_magjumps:
        cmd.append("--with-magjumps")
    if dump_field:
        cmd.append("--dump-field")
    # the solver's per-trial progress lines are passed through as they come
    hd, warnings, ly = None, [], None
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            if verbose:
                print(f"  [seed {seed}] {line}", end="", flush=True)
            match = HD_LINE.search(line)
            if match:
                hd = float(match.group(1))
            if line.startswith("WARNING"):
                warnings.append(line.strip())
            match = LY_LINE.search(line)
            if match:
                ly = int(match.group(1))
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed} failed (exit {proc.returncode})")
    if hd is None:
        raise RuntimeError(f"seed {seed}: could not find h_d in output")
    return hd, warnings, ly


def solve_adaptive(binary, args, seed, outdir, max_headroom):
    """Solves seed with --path/--headroom, doubling the headroom while the
    critical wall comes close to Ly. Each attempt runs in its own
    directory; the accepted one's files are moved to outdir.
    Returns (h_d, warnings, headroom, attempts, Ly)."""
    headroom, attempts = args.headroom, 0
    tmp = outdir / f"tmp_seed{seed}"
    while True:
        attempts += 1
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        box_args = ["--path", str(args.path), "--headroom", str(headroom)]
        if args.y0 is not None:
            box_args += ["--y0", str(args.y0)]
        if args.tilt:
            box_args += ["--tilt", str(args.tilt)]
        hd, warnings, ly = run_one(binary, args.L, args.h_low, args.h_high, args.tol, seed, cwd=tmp,
                                   with_magjumps=args.with_magjumps, dump_field=args.dump_field, box_args=box_args)
        if not warnings or 2 * headroom > max_headroom:
            break
        print(f"seed {seed}: critical wall near Ly with headroom {headroom}, solving again with {2 * headroom}")
        headroom *= 2
    for f in tmp.iterdir():
        shutil.move(str(f), str(outdir / f.name))
    tmp.rmdir()
    return hd, warnings, headroom, attempts, ly


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", default="./phi4vmc", help="path to the compiled solver")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--h-low", type=float, required=True)
    p.add_argument("--h-high", type=float, required=True)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--nsamples", type=int, required=True)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--ly-k", type=float, default=None,
                   help="Ly = k*L^1.25 (see ./phi4vmc --help); default: square L x L")
    p.add_argument("--path", type=int, default=None,
                   help="with --headroom: whole-row escape after P rows, Ly = y0+P+H (see ./phi4vmc --help)")
    p.add_argument("--headroom", type=int, default=None)
    p.add_argument("--y0", type=int, default=None, help="starting row with --path (default: the solver's, 32)")
    p.add_argument("--tilt", type=int, default=0,
                   help="with --path: helical BC, mean wall slope tilt/L (even; see ./phi4vmc --help)")
    p.add_argument("--adapt-headroom", action="store_true",
                   help="if a sample's critical wall comes close to Ly (WARNING), solve that seed again with twice "
                        "the headroom, up to --max-headroom. The disorder of a site doesn't depend on Ly, so it is "
                        "the same sample, and since the path (escape row) is kept, the same finite-size regime. "
                        "Each attempt runs in outdir/tmp_seed<N>/; headroom.csv records the headroom used.")
    p.add_argument("--max-headroom", type=int, default=None, help="cap for --adapt-headroom (default: 16 x --headroom)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--with-magjumps", action="store_true",
                    help="also run the magnetization-jump scan near h_d for each sample (see ./phi4vmc --help)")
    p.add_argument("--dump-field", action="store_true",
                    help="also save each critical field phi as raw float32 (criticafield_h*_seed*.bin, L*L*4 bytes), "
                         "needed by local_roughness.py")
    args = p.parse_args()
    if (args.path is None) != (args.headroom is None):
        p.error("--path and --headroom go together")
    if args.adapt_headroom and args.path is None:
        p.error("--adapt-headroom needs --path and --headroom")
    max_headroom = args.max_headroom or 16 * (args.headroom or 0)
    box_args = []
    if args.ly_k:
        box_args += ["--ly-k", str(args.ly_k)]
    if args.path is not None:
        box_args += ["--path", str(args.path), "--headroom", str(args.headroom)]
        if args.y0 is not None:
            box_args += ["--y0", str(args.y0)]
        if args.tilt:
            box_args += ["--tilt", str(args.tilt)]

    binary = Path(args.binary).resolve()
    if not binary.exists():
        sys.exit(f"binary not found: {binary} (build it first with `make`)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "hd_summary.csv"
    done_seeds = set()
    if summary_path.exists():
        with open(summary_path) as f:
            done_seeds = {int(row["seed"]) for row in csv.DictReader(f) if row["seed"] != "seed"}

    n_warned = 0
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        # several jobs (seed ranges) may share outdir and start together:
        # only the first to lock an empty file writes the header
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.seek(0, os.SEEK_END) == 0:
            writer.writerow(["seed", "h_d"])
            f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
        for i in range(args.nsamples):
            seed = args.seed_start + i
            if seed in done_seeds:
                print(f"seed {seed}: already done, skipping")
                continue
            if args.adapt_headroom:
                hd, warnings, headroom, attempts, ly = solve_adaptive(binary, args, seed, outdir, max_headroom)
                with open(outdir / "headroom.csv", "a") as hf:
                    fcntl.flock(hf, fcntl.LOCK_EX)
                    if hf.seek(0, os.SEEK_END) == 0:
                        hf.write("seed,headroom,attempts,Ly,warning\n")
                    hf.write(f"{seed},{headroom},{attempts},{ly},{int(bool(warnings))}\n")
            else:
                hd, warnings, _ = run_one(binary, args.L, args.h_low, args.h_high, args.tol, seed, cwd=outdir,
                                          with_magjumps=args.with_magjumps, dump_field=args.dump_field,
                                          box_args=box_args)
            if warnings:
                # the critical wall came close to where it would be cut off
                with open(outdir / "warnings.txt", "a") as wf:
                    for w in warnings:
                        wf.write(f"seed {seed}: {w}\n")
                n_warned += 1
            print(f"seed {seed}: h_d = {hd:.6f}")
            writer.writerow([seed, hd])
            f.flush()

    print(f"\nsummary written to {summary_path}")
    if n_warned:
        print(f"{n_warned} sample(s) with a WARNING this run, see {outdir / 'warnings.txt'}")


if __name__ == "__main__":
    main()
