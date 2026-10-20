# One critical wall with 1×8 block disorder (λ < 0 expected)

`snapshot_ax1ay8_L1024_D0.2.png`: the critical configuration of one sample (seed 1).
- **Disorder:** r(x, y) constant over blocks of 1×8 sites, elongated along the direction of motion y, with Δ = 0.2.
  In §5.8 of `docs/analysis_math.md` (Δ = 0.3, L = 512) these blocks gave a threshold that rises slightly with the
  tilt, h_d″/h_d = +0.38 ± 0.29, i.e. λ < 0 but only at 1.3σ.
- **Run:** L = 1024, path P = 600, headroom 300 (adaptive), `--tol 1e-5`. h_d = 0.005420.
- **Wall:** w = 28.7, arc length/L = 1.047, no overhangs, no bubbles.
- **Top panel:** φ over the whole width, cropped in y, with the φ = 0 line. **Bottom panel:** zoom at equal aspect ratio.

The wall is made of nearly flat segments joined by steep ramps.

Reproduce:
```bash
make AMPDIS=0.2 AX=1 AY=8 TARGET=phi4vmc_ax1ay8_D0.2
python3 scripts/run_batch.py --binary ./phi4vmc_ax1ay8_D0.2 --L 1024 --h-low 0.003 --h-high 0.010 --tol 1e-5 \
    --nsamples 1 --seed-start 1 --path 600 --headroom 300 --y0 32 --adapt-headroom --dump-field \
    --outdir data/aniso_neg/ax1ay8_L1024_D0.2_m0
python3 scripts/plot_snapshot.py data/aniso_neg/ax1ay8_L1024_D0.2_m0 --out results/aniso_neg/snapshot_ax1ay8_L1024_D0.2.png
```
