# Blocks elongated along the motion: no λ < 0 at Δ = 0.2

Weak disorder, Δ = 0.2, constant over blocks of 1×AY sites, elongated along the direction of motion y.
In §5.8 of `docs/analysis_math.md` (Δ = 0.3, L = 512), 1×8 blocks gave a threshold that rose slightly with
the tilt, h_d″/h_d = +0.38 ± 0.29, i.e. λ < 0 but only at 1.3σ. At Δ = 0.2 this does not hold.

**Threshold vs tilt** (`tilt/hd_tilt.png`; L = 512, 30 seeds per point, `scripts/hd_tilt.py`):

| blocks | ⟨h_d⟩(0) | s = −0.2 | s = −0.1 | s = +0.1 | s = +0.2 | h_d″/h_d (parabola) |
|---|---|---|---|---|---|---|
| 1×8 | 0.00552 | −3.1 % | −1.6 % | +0.2 % | −0.7 % | −0.9 ± 0.6 (χ² 0.1/2) |
| 1×32 | 0.00375 | −35.6 % | −24.7 % | −23.8 % | −38.5 % | −14.7 ± 0.9 (χ² 71/2) |

- **1×8:** the curvature is compatible with zero, i.e. λ ≈ 0.
- **1×32:** the threshold drops with the tilt (λ > 0 in sign), but as a cusp at s = 0, not a parabola.
  This is not a KPZ term: the flat wall locks onto a horizontal edge of the block grid (rows at multiples
  of 32), and any tilt forces it across those edges.

**Snapshots** of one critical configuration each (seed 1, L = 1024; `scripts/plot_snapshot.py`). Top panel:
φ over the whole width, cropped in y, with the φ = 0 line. Bottom panel: zoom at equal aspect ratio.
- `snapshot_ax1ay8_L1024_D0.2.png`, 1×8:
  - h_d = 0.005420, w = 28.7, arc length/L = 1.047, no overhangs or bubbles;
  - nearly flat segments joined by steep ramps.
- `snapshot_ax1ay32_L1024_D0.2.png`, 1×32:
  - h_d = 0.003336, w = 2.9, arc length/L = 1.008;
  - the wall sits just above the block edge at y = 384, in terraces of a few heights.

Reproduce (P = 600, H = 300 at L = 1024; P = 300, H = 150 at L = 512; tilts m = ±52, ±102):
```bash
make AMPDIS=0.2 AX=1 AY=8 TARGET=phi4vmc_ax1ay8_D0.2
python3 scripts/run_batch.py --binary ./phi4vmc_ax1ay8_D0.2 --L 1024 --h-low 0.002 --h-high 0.010 --tol 1e-5 \
    --nsamples 1 --seed-start 1 --path 600 --headroom 300 --y0 32 --adapt-headroom --dump-field \
    --outdir data/aniso_neg/ax1ay8_L1024_D0.2_m0
python3 scripts/plot_snapshot.py data/aniso_neg/ax1ay8_L1024_D0.2_m0 --out results/aniso_neg/snapshot_ax1ay8_L1024_D0.2.png
```
The tilt runs and the 1×32 snapshot: `data/aniso_neg/run_local.sh` (not versioned).
