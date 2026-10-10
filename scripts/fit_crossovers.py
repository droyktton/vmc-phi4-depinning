#!/usr/bin/env python3
"""Two-crossover analysis of the local roughness exponent zeta_W(l)
(from W2(l) of chord-rotated segments, see local_roughness.py), to get
the qEW exponent even when it shows no plateau.

zeta_W(l) drifts from ~1.45 at small l to ~1 at large l with no plateau
at the qEW 1.25: the qEW window is squeezed between the Larkin length
L_c (below it, Larkin roughness zeta_L = (4-d)/2 = 1.5 in d=1; h_d ~
Delta^(4/3) suggests L_c ~ Delta^(-2/3)) and the crossover length l_o ~
Delta^(-alpha) to the large-scale regime. Two analyses:

1. Collapse of the small-scale part (zeta_W >= --zc) across Delta, by
   horizontal shifts in ln l, at each L: shifts ~ Delta^(-x_c), with
   x_c = 2/3 if that part is Larkin -> qEW.

2. Global fit of every (L, Delta) curve, at scales --lmin <= l <= L/4
   (finite size), to

     zeta(l) = zeta_L + (zeta_q - zeta_L) s(ln(l/L_c)/w1)
                      + (zeta_inf - zeta_q) s(ln(l/l_o)/w2)

   s the logistic function, L_c = a Delta^(-x_c), l_o = b Delta^(-alpha),
   with zeta_q, zeta_inf, w1, w2, a, b global. Fitted with x_c and alpha
   free, and with them fixed (--xc-fixed, --alpha-fixed, e.g. alpha from
   the multivalued-fraction l_o). zeta_L is fixed to 1.5 unless
   --free-zeta-L.

   zeta_q is not seen directly (no plateau) but extrapolated from both
   crossovers, so it depends on the assumed step shape. --forms fits
   each of (x = ln(l/l_*), l_* = L_c or l_o):
     logistic  s(x) = 1/(1+e^{-x/w}): W2 ~ l^{2 z1} [1+(l/l_*)^{1/w}]^{2w(z2-z1)},
               approaches both limits as a power of l/l_*
     erf       s(x) = (1+erf(x/w))/2: approaches them as a Gaussian in ln l
     asym      s(x) = (1+nu e^{-x/w})^{-1/nu} (Richards): different
               power-law approach on each side (rates 1/(w nu) and 1/w)
     sum       no step width: 1/W2 = sum_i (l/l_i)^{-2 zeta_i}, the
               smallest W2 term dominating, so zeta(l) = sum_i zeta_i t_i /
               sum_i t_i with t_i the terms -- a logistic of fixed width
               w = 1/(2 |delta zeta|)
   and the spread of zeta_q over them is the model systematic.

Errors: bootstrap over samples (each refit on resampled curves);
neighbouring l points are correlated, so the fit's own chi^2 errors
aren't used.

Input: run_batch.py output directories already processed by
local_roughness.py (it reads their local_roughness_cache.npz).

Example:
    ./fit_crossovers.py data/sweep/L*_D* --outdir data/sweep/analysis --alpha-fixed 2.1
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

import roughness_estimators as re_
from local_roughness import measure_dir, delta_colors, style, INK, MUTED

HW = 0.15  # decades, as local_roughness.py's default --half-width


def zeta_W(W2, ls):
    return re_.local_slope(ls, W2, HW) / 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


PARAMS = ["zeta_L", "zeta_q", "zeta_inf", "ln_a", "x_c", "ln_b", "alpha", "w1", "w2", "nu1", "nu2"]
FORMS = ("logistic", "erf", "asym", "sum")
# parameters each form doesn't use, held fixed
UNUSED = {"logistic": {"nu1": 1.0, "nu2": 1.0}, "erf": {"nu1": 1.0, "nu2": 1.0},
          "asym": {}, "sum": {"w1": 1.0, "w2": 1.0, "nu1": 1.0, "nu2": 1.0}}


def step(x, w, nu, form):
    if form == "logistic":
        return logistic(x / w)
    if form == "erf":
        return 0.5 * (1 + erf(x / w))
    if form == "asym":
        return (1 + nu * np.exp(np.clip(-x / w, -50, 50))) ** (-1 / nu)
    raise ValueError(form)


def model(p, l, D, form="logistic"):
    zL, zq, zi, lna, xc, lnb, al, w1, w2, nu1, nu2 = p
    lnLc = lna - xc * np.log(D)
    lnlo = lnb - al * np.log(D)
    ll = np.log(l)
    if form == "sum":
        # ln of the three terms of 1/W2, equal pairwise at L_c and at l_o
        lt = np.stack([-2 * zL * (ll - lnLc), -2 * zq * (ll - lnLc),
                       -2 * zq * (lnlo - lnLc) - 2 * zi * (ll - lnlo)])
        wgt = np.exp(lt - lt.max(axis=0))
        return (np.array([zL, zq, zi])[:, None] * wgt).sum(0) / wgt.sum(0)
    return zL + (zq - zL) * step(ll - lnLc, w1, nu1, form) + (zi - zq) * step(ll - lnlo, w2, nu2, form)


def fit(curves, fixed, p0, form="logistic"):
    """curves: list of (l, zeta, sigma, Delta). fixed: {name: value}."""
    fixed = dict(UNUSED[form], **fixed)
    free = [i for i, n in enumerate(PARAMS) if n not in fixed]
    base = np.array([fixed.get(n, v) for n, v in zip(PARAMS, p0)], float)
    l = np.concatenate([c[0] for c in curves])
    z = np.concatenate([c[1] for c in curves])
    s = np.concatenate([c[2] for c in curves])
    D = np.concatenate([np.full(len(c[0]), c[3]) for c in curves])

    def resid(q):
        p = base.copy()
        p[free] = q
        return (model(p, l, D, form) - z) / s

    lo = np.array([-np.inf] * len(PARAMS)); hi = -lo
    for n, (a, b) in {"w1": (0.05, 5), "w2": (0.05, 5), "x_c": (0, 5), "alpha": (0, 6),
                      "nu1": (0.1, 10), "nu2": (0.1, 10)}.items():
        i = PARAMS.index(n); lo[i], hi[i] = a, b
    x0 = np.clip(base[free], lo[free] + 1e-6, hi[free] - 1e-6)
    r = least_squares(resid, x0, bounds=(lo[free], hi[free]), x_scale="jac")
    p = base.copy()
    p[free] = r.x
    return p, float(np.sum(r.fun ** 2)), len(z) - len(free)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datadirs", nargs="+")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--zc", type=float, default=1.30, help="collapse uses the part of zeta_W(l) above this")
    ap.add_argument("--lmin", type=float, default=5)
    ap.add_argument("--lmax-frac", type=float, default=0.25, help="fit scales l <= this * L")
    ap.add_argument("--xc-fixed", type=float, default=2 / 3)
    ap.add_argument("--alpha-fixed", type=float, default=2.1)
    ap.add_argument("--free-zeta-L", action="store_true")
    ap.add_argument("--forms", nargs="+", choices=FORMS, default=list(FORMS),
                    help="crossover step shapes to fit (the plot uses the first)")
    ap.add_argument("--nboot", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sets = []
    for dd in args.datadirs:
        L, D, d = measure_dir(dd)
        ls = d["ls"].astype(float)
        W2 = d["W2_chord"]
        n = len(W2)
        idx = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(args.nboot)]
        Z = np.array([zeta_W(np.nanmean(W2[i], axis=0), ls) for i in idx])  # [0] = full
        sets.append(dict(L=L, D=D, ls=ls, Z=Z))
    sets.sort(key=lambda s: (s["L"], s["D"]))
    Ls = sorted({s["L"] for s in sets})
    out = open(outdir / "crossover_fit.dat", "w")

    # ---- 1. collapse of the small-scale part
    print(f"1. collapse of zeta_W(l) >= {args.zc} (shifts ~ Delta^-x_c; Larkin: 2/3)")
    out.write(f"# collapse of zeta_W >= {args.zc}: L x_c x_c_err\n")
    xcs = {}
    for L in Ls:
        ss = [s for s in sets if s["L"] == L]
        Ds = np.array([s["D"] for s in ss])
        res = []
        for b in range(args.nboot + 1):
            curves = []
            for s in ss:
                z = s["Z"][b]
                ok = np.isfinite(z) & (z >= args.zc) & (s["ls"] >= args.lmin)
                curves.append((s["ls"][ok], z[ok]))
            a, _ = re_.collapse_shifts(curves)
            res.append(re_.power_law_fit(Ds, np.exp(a))[0])
        xc, err = res[0], np.nanstd(res[1:])
        xcs[L] = xc
        out.write(f"{L} {xc:.4g} {err:.2g}\n")
        print(f"  L={L}: x_c = {xc:.3f} +/- {err:.3f}")

    fig, axes = plt.subplots(1, len(Ls), figsize=(5 * len(Ls), 3.8), squeeze=False)
    for ax, L in zip(axes[0], Ls):
        style(ax)
        ss = [s for s in sets if s["L"] == L]
        cols = delta_colors([s["D"] for s in ss])
        for s in ss:
            ok = (s["ls"] >= args.lmin) & (s["ls"] <= args.lmax_frac * L)
            ax.plot(s["ls"][ok] * s["D"] ** (2 / 3), s["Z"][0][ok], "-", lw=2, color=cols[s["D"]], label=f"Δ={s['D']:g}")
        ax.axhline(1.25, color=MUTED, ls="--", lw=0.8)
        ax.axhline(1.5, color=MUTED, ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("l · Δ^(2/3)", fontsize=8)
        ax.set_ylabel("ζ_W(l), chord", fontsize=8)
        ax.set_title(f"L={L}: Larkin scaling of the small-scale part (dashed 1.25, dotted 1.5)", fontsize=8, color=INK)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(outdir / "collapse_larkin.png", dpi=150); plt.close(fig)

    # ---- 2. global two-crossover fit
    def curves_for(b):
        cs = []
        for s in sets:
            ok = (s["ls"] >= args.lmin) & (s["ls"] <= args.lmax_frac * s["L"]) & np.isfinite(s["Z"][0])
            sig = np.nanstd(s["Z"][1:], axis=0)[ok]
            sig = np.maximum(sig, 0.01)
            cs.append((s["ls"][ok], s["Z"][b][ok], sig, s["D"]))
        return cs

    p0 = [1.5, 1.25, 1.0, np.log(5.0), 2 / 3, np.log(100.0), 2.1, 0.7, 0.7, 1.0, 1.0]
    variants = [("all free", {}),
                ("x_c fixed", {"x_c": args.xc_fixed}),
                ("x_c, alpha fixed", {"x_c": args.xc_fixed, "alpha": args.alpha_fixed})]
    print(f"\n2. global fit, {args.lmin:g} <= l <= {args.lmax_frac:g} L, all (L, Delta) "
          f"(zeta_L {'free' if args.free_zeta_L else 'fixed 1.5'})")
    out.write("# global fit: form variant param value err\n")
    fits, summary = {}, []
    for form in args.forms:
        for name, fixed in variants:
            if not args.free_zeta_L:
                fixed = dict(fixed, zeta_L=1.5)
            p, chi2, dof = fit(curves_for(0), fixed, p0, form)
            boots = np.array([fit(curves_for(b), fixed, p, form)[0] for b in range(1, args.nboot + 1)])
            err = np.nanstd(boots, axis=0)
            fits[form, name] = p
            used = dict(UNUSED[form], **fixed)
            summary.append((form, name, p[1], err[1], chi2 / dof))
            print(f"  [{form}, {name}] chi2/dof = {chi2 / dof:.2f} (correlated points: only relative)")
            for i, n in enumerate(PARAMS):
                if n in used:
                    continue
                v, e = p[i], err[i]
                if n.startswith("ln_"):
                    print(f"    {n[3:]:8s} = {np.exp(v):8.3f}  (x/÷ {np.exp(e):.2f})")
                else:
                    print(f"    {n:8s} = {v:8.3f} +/- {e:.3f}")
                out.write(f"{form} {name.replace(' ', '_').replace(',', '')} {n} {v:.5g} {e:.2g}\n")
    out.close()

    print("\nzeta_q by step shape (spread = model systematic; in parentheses chi2/dof):")
    print(f"  {'form':9s}" + "".join(f"{v[0]:>25s}" for v in variants))
    for form in args.forms:
        row = [r for r in summary if r[0] == form]
        print(f"  {form:9s}" + "".join(f"{r[2]:10.3f} +/- {r[3]:.3f} ({r[4]:.2f})" for r in row))
    for vname, _ in variants:
        zq = np.array([r[2] for r in summary if r[1] == vname])
        print(f"  {vname}: zeta_q from {zq.min():.3f} to {zq.max():.3f} over {len(zq)} shapes")

    # data vs model, per L, for the most constrained variant
    form = args.forms[0]
    p = fits[form, "x_c, alpha fixed"]
    fig, axes = plt.subplots(1, len(Ls), figsize=(5 * len(Ls), 3.8), squeeze=False, sharey=True)
    for ax, L in zip(axes[0], Ls):
        style(ax)
        ss = [s for s in sets if s["L"] == L]
        cols = delta_colors([s["D"] for s in ss])
        for s in ss:
            ok = (s["ls"] >= args.lmin) & np.isfinite(s["Z"][0])
            fitted = ok & (s["ls"] <= args.lmax_frac * L)
            err = np.nanstd(s["Z"][1:], axis=0)
            ax.errorbar(s["ls"][fitted], s["Z"][0][fitted], yerr=err[fitted], fmt="o", ms=3, color=cols[s["D"]], label=f"Δ={s['D']:g}")
            ax.plot(s["ls"][ok & ~fitted], s["Z"][0][ok & ~fitted], "o", ms=3, mfc="white", mec=cols[s["D"]])
            lg = np.logspace(np.log10(args.lmin), np.log10(L), 200)
            ax.plot(lg, model(p, lg, s["D"], form), "-", lw=1.2, color=cols[s["D"]])
        ax.axhline(p[1], color=MUTED, ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("arc length l", fontsize=8)
        ax.set_title(f"L={L}: ζ_W(l) and two-crossover fit, {form} steps (dashed: ζ_q={p[1]:.3f}; open: not fitted)", fontsize=8, color=INK)
        ax.legend(fontsize=7, frameon=False)
    axes[0, 0].set_ylabel("ζ_W(l), chord", fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "crossover_fit.png", dpi=150); plt.close(fig)
    print(f"\nwritten {outdir / 'crossover_fit.dat'}, collapse_larkin.png, crossover_fit.png")


if __name__ == "__main__":
    main()
