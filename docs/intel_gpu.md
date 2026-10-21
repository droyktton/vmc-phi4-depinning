# Intel GPUs: the SYCL port

`misistema_sycl.h` is a port of `misistema.h` to SYCL 2020 (Intel oneAPI
`icpx`, with oneDPL in place of Thrust), meant for the Intel Data Center
GPU Max 1550 (Ponte Vecchio) nodes of ClementinaXXI. The driver is the
same `main.cu`: with `-DVMC_SYCL` it includes the SYCL header instead of
the CUDA one. So the command line, the output files and the disorder
samples are the same. `site_disorder` is unchanged (Philox + the same
double arithmetic), so a seed gives the same r(x,y) on both versions.

## What changed

| CUDA (`misistema.h`)                         | SYCL (`misistema_sycl.h`)                                   |
|----------------------------------------------|-------------------------------------------------------------|
| `thrust::device_ptr` / `device_vector`       | USM `sycl::malloc_device` on an in-order `sycl::queue`       |
| Thrust algorithms                            | oneDPL algorithms on a device policy over that queue         |
| `vmc_tiles_kernel<<<ntiles,512>>>`           | `parallel_for(nd_range)`, one 512-item work-group per tile   |
| `atomicAdd(work,1ULL)`                       | `sycl::atomic_ref<unsigned long long>`                       |
| `cudaMemset`                                 | `queue::memset`                                              |
| element access `d_phi[i]` from the host      | bulk `queue::memcpy` to a host vector                       |

The physics (`vieta_solver`, `vmc_site_update`, the boundary conditions,
the active tiles, the convergence criteria) is the same code. Only the
`__device__` annotations are gone.

## Build

```
module swap gnu13 intel/2025.3.0     # gnu13 is loaded by default and conflicts
module load dpl                      # oneDPL
make sycl                            # ./phi4vmc_sycl, compiled ahead of time for PVC
make sycl SYCL_AOT=                  # JIT (spir64): runs on any SYCL device
make sycl AMPDIS=0.4 AX=4 AY=1 TARGET_SYCL=phi4vmc_sycl_D0.4_ax4ay1
```

- The ahead-of-time build (`-fsycl-targets=spir64_gen -device pvc`) needs
  `ocloc`, which is part of the GPU driver and is **not** on the login
  node. Either build on a GPU node (`scripts/slurm_bench_intel.sh` does
  that) or build JIT on the login node with `SYCL_AOT=`. The JIT build
  compiles the kernels for the GPU the first time they run, which takes
  a few seconds.
- `-fp-model=precise` is in the Makefile on purpose: `icpx` uses fast
  math by default, which would change the roots of the cubic and the
  convergence tests.
- The `-D` model parameters (`AMPDIS`, `CEL`, `AX`, `AY`, ...) are the
  same as for CUDA.

## Run

Same command line as `./phi4vmc` (see the README). The device is SYCL's
default one, and its name goes to stderr and to `logfile.dat`:

- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (or `opencl:cpu` to run on the
  CPU, e.g. to test the JIT build; not on the login node, which is not
  for compute);
- each Max 1550 shows up as two GPUs, one per stack, and
  `ZE_AFFINITY_MASK=0` (or `1`, ...) picks one stack. One sample runs on
  one stack, so a 4-card node can run 8 samples at once.

Jobs: partitions `testing` (20 min) and `gpunode` (2 days), gres
`gpu:intel_xt1550:N`. `scripts/slurm_bench_intel.sh` is a complete
example. For production sweeps, `scripts/run_batch.py --binary
./phi4vmc_sycl` works unchanged, with one process per stack, each with
its own `ZE_AFFINITY_MASK` and seed range (`--seed-start`, `--nsamples`).

## Validation and benchmark

`scripts/bench_compare.sh BINARY OUTDIR` solves the same short set of
samples on any machine: by default L = 256, 512, 1024 with seeds 1–3, at
Delta = 0.2, `h` in [0, 0.1] and tol = 1e-4, with `--dump-field`. It
writes `OUTDIR/bench.csv`. Then

```
scripts/compare_runs.py bench_cuda bench_sycl --names CUDA SYCL
```

tabulates, per sample: h_d on each side, whether the critical walls are
identical, max |Δphi| between the critical fields, the number of sites
whose sign differs, and the time ratio.

### Results so far (2026-10-21)

**Correctness, SYCL on CPU vs CUDA.** The SYCL build ran on the cluster's
login node CPU (Xeon Gold 6326, OpenCL CPU device, JIT). For L=64 with
seed 1 and tol=1e-3, it gives the **same h_d (0.0101563) and a
byte-identical critical wall** (`critica_*.dat`) as CUDA on the RTX A4000.
The magnetization profiles differ only in the last digits (float
rounding). That was only a tiny smoke test: compute must not run on the
login node, so the comparison at L ≥ 128 is waiting for a GPU job (the
CUDA side is already in `results/intel_port/bench_cuda_a4000.csv`).

Bit-for-bit equality isn't guaranteed in general: `cbrtf`/`acosf`/`cosf`
may differ by an ulp between the CUDA and Intel math libraries, and the
compilers may contract to FMA differently. A difference of that size can
move h_d by one bisection step at most, in rare samples.

**Performance, CUDA on the laptop** (NVIDIA RTX A4000 Laptop GPU; wall
time per sample, including I/O, from `results/intel_port/bench_cuda_a4000.csv`):

| L    | seeds | time per sample (s) | VMC sweeps per sample |
|------|-------|---------------------|-----------------------|
| 128  | 1–2   | 1.5–1.8             | ~1.1e5                |
| 256  | 1–3   | 2.5–3.0             | ~1.8e5                |
| 512  | 1–3   | 6.7–7.7             | ~3.8e5                |
| 1024 | 1–3   | 18–25               | ~9.2e5                |

**Performance on the Max 1550: pending.** No job could be submitted. The
only QOS of account `pisca_73`, `qos_pisca_73`, has
`GrpTRESMins=cpu=0,gres/gpu=8322`: GPU minutes but zero CPU minutes, so
SLURM rejects every job (`QOSGrpCPUMinutesLimit`), since a job always
holds at least one CPU. The cluster admins need to fix that. Then:

```
cd ~/vmc-phi4-depinning
sbatch -A pisca_73 scripts/slurm_bench_intel.sh          # -> bench_sycl/
# back on the laptop:
rsync -a akolton@ssh.clementinaxxi.org.ar:vmc-phi4-depinning/bench_sycl/ bench_sycl/
scripts/compare_runs.py <laptop bench_cuda dir> bench_sycl --names CUDA SYCL
```

### Expectations for performance

At these sizes a sample is dominated by many short sweeps of a few
active tiles. Each sweep submits 6 small operations (2 × memset, tile
kernel, activity spread). So the per-launch latency matters more than
raw throughput. Level Zero launches are typically slower than CUDA's, so
small L may be no faster on the Max 1550 than on the laptop, even
though the card is much bigger. If that shows up, the first
optimization is to fuse the memsets into `spread_activity` and to batch
a CHOP's worth of sweeps with fewer host round trips.

## Running the SYCL version on NVIDIA

Possible, but not set up or tested. It needs oneAPI with Codeplay's
"oneAPI for NVIDIA GPUs" plugin (`-fsycl-targets=nvptx64-nvidia-cuda`),
or AdaptiveCpp. Nothing in the code is Intel-specific, but on NVIDIA the
CUDA version (`make`) remains the one to use.
