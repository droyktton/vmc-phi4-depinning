#!/usr/bin/env python3
"""Local roughness exponent zeta(l) and crossover length l_o of the
critical domain walls, comparing global and local-detrended estimators
(see roughness_estimators.py for the definitions):

  global, on the univalued u(x) (paper Eq. 9): S(q), B(r), W2(L)
  local, on segments of arc length l rotated to their tilt, with
  tilt = end-to-end chord and = principal axis (pca): W2(l), B(r,l), S(q,l)

Input: one or more run_batch.py output directories holding the raw
critical fields (criticafield_h*_seed*.bin, i.e. runs made with
--dump-field), e.g. those of scripts/sweep_disorder.sh. L and Delta are
read from each directory's logfile.dat. Per-sample measurements are
cached in <datadir>/local_roughness_cache.npz.

Scale-dependent exponents (log-derivatives over a sliding window of
--half-width decades):
  zeta_S_glob(l=2pi/q), zeta_B_glob(r)        global
  zeta_W_<tilt>(l)  = (1/2) dlnW2/dlnl         local
  zeta_B_<tilt>(l)  from B(r,l) over r in [l/16, l/4]
  zeta_S_<tilt>(l)  from S(q,l): chord by FFT over k = 2..8, pca by cosine
                    transform (its ends don't match) over k = 4..16 -- the
                    same wavelengths l/8..l/2 either way. Plotted vs the
                    probed wavelength, l/4

Crossover length l_o per dataset, several definitions:
  zW_<tilt>, zS_glob : where the local exponent first drops below --zeta-th
  fmv_<tilt>         : where the fraction of segments still multivalued in
                       their own rotated frame reaches --fmv-th
and, across the Delta values of each L, by data collapse (horizontal
shifts in ln l) of fmv_chord(l) and zeta_W_chord(l). Each l_o(Delta) is
fitted to l_o ~ Delta^-alpha (paper: alpha ~ 1/0.45 ~ 2.2). Errors:
bootstrap over samples. Datasets whose l_o exceeds --lo-max-frac * L are
left out of the fits and collapses (finite size: the local exponent drops
near l ~ L/2 even without a crossover, and comparing L=1024 with L=2048,
l_o agrees up to ~L/4 but is overestimated at ~0.36 L); the collapses use
the multivalued-fraction l_o for that cut, the most robust definition, and
only the scales l <= --lo-max-frac * L of each curve.

Example:
    ./local_roughness.py data/sweep/L*_D* --outdir data/sweep/analysis
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wall_contour import read_field, wall_curve, periodic_arclength, univalued_profile
import roughness_estimators as re_

TILTS = ("chord", "pca")
CACHE_VERSION = 2

# reference palette (dataviz skill): fixed categorical order for
# estimators, ordinal one-hue (blue) ramp for Delta
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, MUTED = "#0b0b0b", "#52514e"


# ------------------------------------------------------------ measuring

def grids(L):
    ls = re_.log_grid(4, L, 32)
    rs = re_.log_grid(1, L // 2, 32)
    return ls, rs


def measure_curve(curve, L, ls, rs, pca_transform="dct"):
    """All per-sample measurements of one wall given as a periodic
    curve (wall_contour.wall_curve). Returns a flat dict of arrays."""
    u, m = univalued_profile(curve, L)
    g = re_.global_observables(u)
    pts, ds, _ = periodic_arclength(curve, 1.0)
    out = {
        "S_glob": g["S"], "B_glob": g["B"], "W2_glob": g["W2"],
        "rho_cols": float(np.mean(m >= 3)), "has_overhang": float(np.any(m >= 3)),
        "arclen": len(pts) * ds,
    }
    kmax = L // 2
    for tilt in TILTS:
        loc = re_.local_observables(pts, ds, L, ls, rs, tilt=tilt,
                                    transform=(pca_transform if tilt == "pca" else "fft"), kmax=kmax)
        for key, val in loc.items():
            out[f"{key}_{tilt}"] = val
    return out


def logfile_params(datadir):
    text = (Path(datadir) / "logfile.dat").read_text()
    L = int(re.search(r"L=\s*(\d+)", text).group(1))
    delta = float(re.search(r"AMPDIS \(Delta\)=\s*([-\d.eE+]+)", text).group(1))
    return L, delta


def measure_dir(datadir, pca_transform="dct", refresh=False):
    """Per-sample measurements of every critical field in datadir,
    stacked over samples (cached). Returns (L, Delta, dict of arrays)."""
    datadir = Path(datadir)
    L, delta = logfile_params(datadir)
    files = sorted(glob.glob(str(datadir / "criticafield_h*_seed*.bin")))
    if not files:
        raise SystemExit(f"no criticafield_h*_seed*.bin files in {datadir} (run with --dump-field)")
    cache = datadir / "local_roughness_cache.npz"
    key = f"v{CACHE_VERSION}|{pca_transform}|" + "|".join(Path(f).name for f in files)
    if cache.exists() and not refresh:
        c = np.load(cache, allow_pickle=False)
        if str(c["key"]) == key:
            return L, delta, {k: c[k] for k in c.files if k != "key"}

    ls, rs = grids(L)
    rows, nbub = [], 0
    for f in files:
        curve, nb = wall_curve(read_field(f, L))
        nbub += nb
        rows.append(measure_curve(curve, L, ls, rs, pca_transform))
    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data.update(ls=ls, rs=rs, n_bubbles=np.array(nbub))
    np.savez_compressed(cache, key=key, **data)
    return L, delta, data


# ------------------------------------------------------------- averaging

def mean_curves(d, idx):
    """Sample-averaged observables over sample indices idx."""
    sel = lambda k: d[k][idx]
    out = {k: np.nanmean(sel(k), axis=0) for k in d
           if k not in ("ls", "rs", "n_bubbles") and d[k].ndim >= 1 and d[k].shape[0] == len(d["arclen"])}
    out["ls"], out["rs"] = d["ls"], d["rs"]
    return out


def derived(m, L, args):
    """Scale-dependent exponents and l_o definitions from averaged curves."""
    ls, rs = m["ls"].astype(float), m["rs"].astype(float)
    hw = args.half_width
    res = {"curves": {}, "lo": {}}

    # global S: log-binned, as a function of l = 2 pi / q (ascending)
    q = 2 * np.pi * np.arange(1, L // 2 + 1) / L
    qb, Sb = re_.log_bin(q, m["S_glob"], 12)
    lq = (2 * np.pi / qb)[::-1]
    zS = (-(re_.local_slope(qb, Sb, hw) + 1) / 2)[::-1]
    res["curves"]["zeta_S_glob"] = (lq, zS)
    res["curves"]["zeta_B_glob"] = (np.arange(1, L // 2 + 1.0), re_.local_slope(np.arange(1, L // 2 + 1.0), m["B_glob"], hw) / 2)
    res["lo"]["zS_glob"] = re_.crossing(lq[lq >= args.lmin], zS[lq >= args.lmin], args.zeta_th, "down")

    for t in TILTS:
        W2 = m[f"W2_{t}"]
        zW = re_.local_slope(ls, W2, hw) / 2
        res["curves"][f"zeta_W_{t}"] = (ls, zW)
        res["curves"][f"zeta_B_{t}"] = (ls, re_.zeta_from_Brl(ls, rs, m[f"B_{t}"]))
        # a cosine transform's k is in units of pi/l, an FFT's of 2pi/l:
        # both fits cover wavelengths l/8..l/2, plotted at their
        # geometric mean l/4
        dct_ = t == "pca" and args.pca_transform == "dct"
        kfit = (4, 16) if dct_ else (2, 8)
        res["curves"][f"zeta_S_{t}"] = (ls / 4,
                                         re_.zeta_from_Sql(ls, m[f"S_{t}"], kfit))
        A = np.sqrt(W2) / ls
        th = np.sqrt(m[f"theta2_{t}"])
        res["curves"][f"A_{t}"] = (ls, A)
        res["curves"][f"theta_{t}"] = (ls, th)
        res["curves"][f"fmv_{t}"] = (ls, m[f"fmv_{t}"])
        # skip the smallest scales, where the lattice/contour
        # discretization dominates, before looking for crossings
        big = ls >= args.lmin
        res["lo"][f"zW_{t}"] = re_.crossing(ls[big], zW[big], args.zeta_th, "down")
        res["lo"][f"fmv_{t}"] = re_.crossing(ls, m[f"fmv_{t}"], args.fmv_th, "up")
    return res


def bootstrap(d, L, args, rng):
    """Derived quantities for the full sample set and for --nboot
    bootstrap resamples. Returns (full, list of resampled)."""
    n = len(d["arclen"])
    full = derived(mean_curves(d, np.arange(n)), L, args)
    boots = [derived(mean_curves(d, rng.integers(0, n, n)), L, args) for _ in range(args.nboot)]
    return full, boots


# ---------------------------------------------------------------- output

def write_table(path, header, cols):
    with open(path, "w") as f:
        f.write("# " + " ".join(header) + "\n")
        for row in zip(*cols):
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def delta_colors(deltas):
    deltas = sorted(deltas)
    n = len(deltas)
    pick = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int) if n > 1 else [len(BLUE_RAMP) // 2]
    return {D: BLUE_RAMP[i] for D, i in zip(deltas, pick)}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)


def plot_zeta_panels(by_L, outdir, args):
    """Per L: one panel per estimator, zeta_loc(l) for each Delta."""
    panels = ["zeta_S_glob", "zeta_B_glob", "zeta_W_chord", "zeta_W_pca",
              "zeta_B_chord", "zeta_B_pca", "zeta_S_chord", "zeta_S_pca"]
    titles = ["global S(q)", "global B(r)", "local W²(l), chord", "local W²(l), pca",
              "local B(r,l), chord", "local B(r,l), pca", "local S(q,l), chord", "local S(q,l), pca"]
    for L, sets in by_L.items():
        cols = delta_colors([s["delta"] for s in sets])
        fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
        for ax, key, title in zip(axes.flat, panels, titles):
            style(ax)
            for s in sorted(sets, key=lambda s: s["delta"]):
                x, y = s["full"]["curves"][key]
                err = np.nanstd([b["curves"][key][1] for b in s["boots"]], axis=0)
                ok = np.isfinite(y)
                ax.plot(x[ok], y[ok], "-", lw=2, color=cols[s["delta"]], label=f"Δ={s['delta']:g}")
                ax.fill_between(x[ok], (y - err)[ok], (y + err)[ok], color=cols[s["delta"]], alpha=0.15, lw=0)
            for z, lab in ((1.25, "qEW 1.25"), (1.0, "1")):
                ax.axhline(z, color=MUTED, lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_ylim(0, 1.8)
            ax.set_title(title, fontsize=9, color=INK)
        for ax in axes[1]:
            ax.set_xlabel("scale l (lattice units)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("local ζ", fontsize=8)
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"local roughness exponent, L={L} (dashed: 1.25 and 1)", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"zeta_local_L{L}.png", dpi=150)
        plt.close(fig)


def plot_crossover_panels(by_L, outdir):
    """Per L: fmv(l), W/l and theta_rms(l) per Delta, chord tilt."""
    for L, sets in by_L.items():
        cols = delta_colors([s["delta"] for s in sets])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, key, lab in zip(axes, ["fmv_chord", "A_chord", "theta_chord"],
                                ["fraction multivalued after rotation", "aspect ratio W(l)/l", "rms tilt angle (rad)"]):
            style(ax)
            for s in sorted(sets, key=lambda s: s["delta"]):
                x, y = s["full"]["curves"][key]
                ax.plot(x, y, "-", lw=2, color=cols[s["delta"]], label=f"Δ={s['delta']:g}")
            ax.set_xscale("log")
            if key != "fmv_chord":
                ax.set_yscale("log")
            ax.set_xlabel("arc length l", fontsize=8)
            ax.set_title(lab, fontsize=9, color=INK)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"crossover indicators, L={L} (chord tilt)", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"crossover_indicators_L{L}.png", dpi=150)
        plt.close(fig)


def collapse(sets, key, floor=None, min_points=4, which="full", lmax=np.inf):
    """Shift-collapse of curve `key` across the Delta values of one L,
    using only scales l <= lmax (finite size).
    With `floor`, only the part of each curve with y >= floor is used,
    and only curves with >= min_points such points take part (a curve
    that is identically ~0, like the multivalued fraction below l_o,
    collapses onto anything). `which` = "full" or a bootstrap index.
    Returns (used sets, l_o relative to the first used set, cost)."""
    used, curves = [], []
    for s in sorted(sets, key=lambda s: s["delta"]):
        src = s["full"] if which == "full" else s["boots"][which]
        x, y = src["curves"][key]
        ok = np.isfinite(y) & (x <= lmax) & (y >= floor if floor is not None else True)
        if ok.sum() >= min_points:
            used.append(s)
            curves.append((x[ok], y[ok]))
    if len(used) < 2:
        return used, np.array([]), np.nan
    a, cost = re_.collapse_shifts(curves)
    return used, np.exp(a), cost


def plot_collapse(sets, key, rel, L, outdir, ylabel):
    sets = sorted(sets, key=lambda s: s["delta"])
    cols = delta_colors([s["delta"] for s in sets])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    style(ax)
    for s, r in zip(sets, rel):
        x, y = s["full"]["curves"][key]
        ax.plot(x / r, y, "-", lw=2, color=cols[s["delta"]], label=f"Δ={s['delta']:g}")
    ax.set_xscale("log")
    ax.set_xlabel(f"l / l_o (l_o relative to Δ={sets[0]['delta']:g})", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"collapse of {key}, L={L}", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"collapse_{key}_L{L}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datadirs", nargs="+")
    p.add_argument("--outdir", required=True)
    p.add_argument("--zeta-th", type=float, default=1.125,
                   help="l_o where the local exponent drops below this (default: midway between 1.25 and 1)")
    p.add_argument("--fmv-th", type=float, default=0.05,
                   help="l_o where the fraction of segments multivalued in their own frame reaches this "
                        "(it only approaches 1 for L >> l_o, so the default is low)")
    p.add_argument("--fmv-floor", type=float, default=0.01,
                   help="collapse of the multivalued fraction uses only its part above this")
    p.add_argument("--lo-max-frac", type=float, default=0.25,
                   help="leave datasets with l_o above this fraction of L out of the alpha fits and collapses, "
                        "and collapse only scales below it (finite size; open markers in lo_vs_delta.png)")
    p.add_argument("--lmin", type=float, default=8, help="ignore scales below this when looking for crossings")
    p.add_argument("--half-width", type=float, default=0.15, help="decades, for the local log-derivatives")
    p.add_argument("--nboot", type=int, default=100)
    p.add_argument("--pca-transform", choices=("dct", "fft"), default="dct",
                   help="transform for S(q,l) of pca segments (default dct: their ends don't match, "
                        "so a plain FFT leaks and caps zeta at 0.5)")
    p.add_argument("--refresh", action="store_true", help="ignore cached per-sample measurements")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sets = []
    for dd in args.datadirs:
        L, delta, d = measure_dir(dd, args.pca_transform, args.refresh)
        full, boots = bootstrap(d, L, args, rng)
        n = len(d["arclen"])
        sets.append(dict(dir=dd, L=L, delta=delta, n=n, data=d, full=full, boots=boots))
        print(f"{dd}: L={L} Delta={delta:g} n={n} arclen/L={d['arclen'].mean() / L:.3f} "
              f"bubbles={int(d['n_bubbles'])} rho(sample has overhang)={d['has_overhang'].mean():.2f}")

        # per-dataset curve tables
        name = f"L{L}_D{delta:g}"
        m = mean_curves(d, np.arange(n))
        ls = d["ls"].astype(float)
        cols, head = [ls], ["l"]
        for t in TILTS:
            for k in ("W2", "theta2", "fmv"):
                cols.append(m[f"{k}_{t}"]); head.append(f"{k}_{t}")
            for k in ("zeta_W", "zeta_B", "zeta_S", "A"):
                # zeta_S is listed against the segment length l it came
                # from (it probes the shorter wavelength l/sqrt(k1 k2))
                cols.append(full["curves"][f"{k}_{t}"][1])
                head.append(f"{k}_{t}" + ("(segment_l)" if k == "zeta_S" else ""))
        write_table(outdir / f"local_{name}.dat", head, cols)

        # full local S(q,l) and B(r,l), long format
        rs = d["rs"].astype(float)
        rows_S, rows_B = [], []
        for i, l in enumerate(ls):
            for kk in range(1, int(l) // 2 + 1):
                q_pca = (np.pi if args.pca_transform == "dct" else 2 * np.pi) * kk / l
                rows_S.append((l, kk, 2 * np.pi * kk / l, m["S_chord"][i, kk - 1], q_pca, m["S_pca"][i, kk - 1]))
            for j, r in enumerate(rs):
                if np.isfinite(m["B_chord"][i, j]):
                    rows_B.append((l, r, m["B_chord"][i, j], m["B_pca"][i, j]))
        write_table(outdir / f"Sql_{name}.dat", ["l", "k", "q_chord", "S_chord", "q_pca", f"S_pca({args.pca_transform})"], list(zip(*rows_S)))
        write_table(outdir / f"Brl_{name}.dat", ["l", "r", "B_chord", "B_pca"], list(zip(*rows_B)))
        q = 2 * np.pi * np.arange(1, L // 2 + 1) / L
        sem = lambda k: np.nanstd(d[k], axis=0, ddof=1) / np.sqrt(n)
        write_table(outdir / f"global_{name}.dat", ["q", "S", "S_sem", "r", "B", "B_sem"],
                    [q, m["S_glob"], sem("S_glob"), np.arange(1, L // 2 + 1.0), m["B_glob"], sem("B_glob")])

    # l_o table
    lo_keys = list(sets[0]["full"]["lo"].keys())
    rows = []
    for s in sorted(sets, key=lambda s: (s["L"], s["delta"])):
        vals = [s["full"]["lo"][k] for k in lo_keys]
        errs = [np.nanstd([b["lo"][k] for b in s["boots"]]) for k in lo_keys]
        w2 = s["data"]["W2_glob"]
        rows.append((s["L"], s["delta"], s["n"], w2.mean(), s["data"]["has_overhang"].mean(),
                     s["data"]["rho_cols"].mean(), vals, errs))
    with open(outdir / "lo_table.dat", "w") as f:
        f.write("# L Delta n W2_glob rho_sample rho_cols " + " ".join(f"lo_{k} err" for k in lo_keys) + "\n")
        for L, D, n, w2, rs_, rc, vals, errs in rows:
            f.write(f"{L} {D:g} {n} {w2:.6g} {rs_:.4g} {rc:.4g} "
                    + " ".join(f"{v:.4g} {e:.2g}" for v, e in zip(vals, errs)) + "\n")
    print(f"\nl_o table written to {outdir / 'lo_table.dat'}")

    by_L = {}
    for s in sets:
        by_L.setdefault(s["L"], []).append(s)

    # alpha fits: threshold definitions, and collapse
    with open(outdir / "alpha.dat", "w") as f:
        f.write("# L definition alpha alpha_err(bootstrap) n_Delta  (l_o ~ Delta^-alpha; paper ~2.2)\n")
        print("\nl_o ~ Delta^-alpha:")
        for L, ss in sorted(by_L.items()):
            if len(ss) < 2:
                continue
            ss = sorted(ss, key=lambda s: s["delta"])
            D = np.array([s["delta"] for s in ss])
            lo_max = args.lo_max_frac * L
            for k in lo_keys:
                lo = np.array([s["full"]["lo"][k] for s in ss])
                # the cut is decided on the full-sample l_o, so every
                # bootstrap resample fits the same Delta values
                keep = lo <= lo_max
                alpha, _ = re_.power_law_fit(D[keep], lo[keep])
                ab = [re_.power_law_fit(D[keep], np.array([s["boots"][b]["lo"][k] for s in ss])[keep])[0]
                      for b in range(args.nboot)]
                nD = int(keep.sum())
                cut = D[np.isfinite(lo) & ~keep]
                f.write(f"{L} {k} {alpha:.4g} {np.nanstd(ab):.2g} {nD}\n")
                print(f"  L={L:5d} {k:14s} alpha = {alpha:6.3f} +/- {np.nanstd(ab):.3f}  ({nD} Delta values"
                      + (f"; l_o > {lo_max:g} at Delta = {', '.join(f'{d:g}' for d in cut)}" if len(cut) else "") + ")")
            for key, ylabel, floor in (("fmv_chord", "fraction multivalued", args.fmv_floor),
                                       ("fmv_pca", "fraction multivalued", args.fmv_floor),
                                       ("zeta_W_chord", "local ζ from W²(l)", None)):
                ref = key if key.startswith("fmv") else "fmv_chord"
                ok = [s for s in ss if s["full"]["lo"][ref] <= lo_max]
                used, rel, cost = collapse(ok, key, floor, lmax=lo_max)
                if len(used) < 2:
                    print(f"  L={L:5d} collapse {key:14s} needs >= 2 Delta with signal, skipped")
                    continue
                Du = np.array([s["delta"] for s in used])
                alpha, _ = re_.power_law_fit(Du, rel)
                ab = []
                for b in range(min(args.nboot, 50)):
                    ub, rb, _ = collapse(used, key, floor, which=b, lmax=lo_max)
                    if len(ub) == len(used):
                        ab.append(re_.power_law_fit(Du, rb)[0])
                err = np.nanstd(ab) if ab else np.nan
                f.write(f"{L} collapse_{key} {alpha:.4g} {err:.2g} {len(used)}\n")
                print(f"  L={L:5d} collapse {key:14s} alpha = {alpha:6.3f} +/- {err:.3f}  "
                      f"(Delta = {', '.join(f'{d:g}' for d in Du)}; cost {cost:.3g})")
                plot_collapse(used, key, rel, L, outdir, ylabel)
        print()

    plot_zeta_panels(by_L, outdir, args)
    plot_crossover_panels(by_L, outdir)
    plot_lo_vs_delta(rows, lo_keys, outdir, args.lo_max_frac)
    print(f"plots written to {outdir}")


def plot_lo_vs_delta(rows, lo_keys, outdir, lo_max_frac):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    style(ax)
    Ls = sorted({r[0] for r in rows})
    mk = ["o", "s", "^", "D"]
    for i, k in enumerate(lo_keys):
        color = SERIES[i]  # <= 8 definitions: fixed categorical order
        for j, L in enumerate(Ls):
            rr = [r for r in rows if r[0] == L]
            D = np.array([r[1] for r in rr])
            v = np.array([r[6][i] for r in rr])
            e = np.array([r[7][i] for r in rr])
            ok = np.isfinite(v)
            if not ok.any():
                continue
            ax.errorbar(D[ok], v[ok], yerr=e[ok], fmt=mk[j % len(mk)] + "-", ms=5, lw=1.2,
                        color=color, label=f"{k} (L={L})")
            # open markers: left out of the fits (l_o > lo_max_frac * L)
            out = ok & (v > lo_max_frac * L)
            ax.plot(D[out], v[out], mk[j % len(mk)], ms=5, mfc="white", mec=color, mew=1.2, zorder=3)
    # reference slope, through the median l_o at the largest Delta
    Dg = np.array(sorted({r[1] for r in rows}))
    at_max = [v for r in rows if r[1] == Dg.max() for v in r[6] if np.isfinite(v)]
    if len(Dg) > 1 and at_max:
        ref = np.median(at_max) * (Dg / Dg.max()) ** (-1 / 0.45)
        ax.plot(Dg, ref, "--", color=MUTED, lw=1, label="Δ^-2.2 (paper)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Δ", fontsize=9)
    ax.set_ylabel("l_o", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.set_title(f"crossover length vs disorder, all definitions (open: l_o > {lo_max_frac:g} L, not fitted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "lo_vs_delta.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
