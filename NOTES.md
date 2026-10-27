# DFT speed-up workspace: notes and experiment log

**Policy: exact speedups only.** Results must stay within 1e-9 Eh of unmodified PySCF; that means no new
approximations by default. **Target workloads:**
- medium molecules (DZ/TZ, GGA/hybrid/meta-GGA, RKS/UKS)
- direct `eval_ao`/`eval_rho`/`eval_xc` use in scripts
- gradients, TDDFT and Hessians

LibXC gets attention only if a profile shows it hot. Plan: `~/.claude/plans/i-would-like-you-polished-matsumoto.md`.

## Layout

Every upstream clone has `origin` = upstream and a local branch `perf` (one commit per optimization, never pushed).

| dir | upstream | pinned at | role |
|---|---|---|---|
| `pyscf/` | github.com/pyscf/pyscf | master `44313dfb45` (2026-10-26) | working tree, all PySCF changes |
| `pyscf-base/` | worktree of `pyscf/` | `44313dfb45`, detached | unmodified reference build "B" |
| `libcint/` | github.com/sunqm/libcint | `v6.2.0` (master's pin) | integrals |
| `qcint/` | github.com/sunqm/qcint | `v6.2.0` | SIMD integrals (It2) |
| `libxc/` | gitlab.com/libxc/libxc | `7.1.2` (master's pin) | XC functionals |
| `OpenBLAS/` | github.com/OpenMathLib/OpenBLAS | `v0.3.34` | BLAS (build-only so far) |

- **Python:** two uv venvs on a uv-managed CPython 3.12.8, with identical pins (numpy 2.5.4, scipy 1.18.1, h5py 3.16.0, pytest 9.2.0).
  - `.venv` runs the source trees; `PYTHONPATH` picks the tree.
  - `.venv-wheel` has the PyPI `pyscf==2.12.1` wheel: baseline "A", what the user runs today.
  - The anaconda base env is never touched.

## How to

```bash
# dependencies (install/<name>), then a PySCF tree wired to a chosen cint/libxc build
scripts/build_openblas.sh
scripts/build_cint.sh libcint stock            # or: qcint stock | libcint native -DBUILD_MARCH_NATIVE=ON
scripts/build_libxc.sh stock                   # or: build_libxc.sh znver5 -DCMAKE_C_FLAGS=-march=znver5
scripts/build_pyscf.sh pyscf libcint-stock libxc-stock [extra cmake args]

scripts/check_power.sh                          # benchmark-ready? (AC, performance profile, no freq cap)
cd bench
../.venv/bin/python kernels.py new=pyscf        # fixed-input kernel tests vs pyscf-base (rtol 1e-11)
../.venv/bin/python validate.py new=pyscf [--suite full]   # tight-convergence results vs pyscf-base
../.venv/bin/python run_bench.py --tag NAME A=wheel B=pyscf-base:OPENBLAS_CORETYPE=Barcelona new=pyscf
../.venv/bin/python compare.py results/<file>.json [--base B]
```

- **Config syntax:** `LABEL=TREE[:VAR=VAL,...]`. TREE is a PySCF tree directory, or `wheel`. VARs override the pinned worker environment, e.g. `OPENBLAS_CORETYPE=SkylakeX` or `OMP_NUM_THREADS=1`.
- **Workers:**
  - Each worker is a fresh process: warm-up first (imports, library loads), then the timed `run()` under cProfile.
  - It reports wall time and a per-category cost breakdown.
  - It records the `.so` files actually loaded, and fails if a source tree loads BLAS, cint or libxc from outside `install/`.

## Gotchas found so far
- **The wheel's BLAS is the biggest single problem.** It bundles OpenBLAS 0.3.3, which reports `Barcelona SINGLE_THREADED` (SSE2 kernels), and every `pyscf.lib.dot` goes through it.
  - Measured: 114 GFLOP/s, vs 231–261 GFLOP/s for 0.3.34's Zen/SkylakeX/Cooperlake kernels (n=3000, 2 GHz cap).
- **OpenBLAS ≤0.3.34 `DYNAMIC_ARCH` doesn't know Zen 5** (AMD exfamily 11). It falls back to Cooperlake, whose DGEMM is the SkylakeX one.
  - Our build lists BARCELONA HASWELL ZEN SKYLAKEX COOPERLAKE, so `OPENBLAS_CORETYPE` switches kernels at runtime.
  - The OpenMP build **ignores `OPENBLAS_NUM_THREADS`**, which therefore only sizes numpy's own pool.
- **libcint ↔ qcint is a build-time choice per PySCF tree.** The two have different `CINTEnvVars` layouts (qcint adds SIMD-width arrays and function pointers). About ten PySCF C files read those structs (`ft_ao.c`, `cint3c2e.c`, `nr_sr_vhf.c`, …). An `LD_LIBRARY_PATH` swap would corrupt memory; only BLAS may be swapped at runtime.
- **anaconda's python has `RPATH $ORIGIN/../lib`**, which would leak anaconda's libs (MKL, conda libgomp) into library resolution. The venvs therefore use uv-managed CPython (no rpath).
- **PySCF fixes its RUNPATH to `$ORIGIN/deps/lib`**, so `scripts/link_deps.sh` symlinks our installs there. A system `/usr/lib/x86_64-linux-gnu/libopenblas.so.0` exists and would load silently if a link were missing; `build_pyscf.sh` and the harness both check for this.
- **Don't launch python from the workspace root.** `pkgutil.extend_path` then adds the clone root `my-pyscf/pyscf/` to `pyscf.__path__`. It is harmless, but noisy. The harness runs workers from `scratch/`.
- **Thread binding is not set by default.** `OMP_PROC_BIND` pins Python's main thread to one core, and pthread pools created afterwards (numpy/scipy OpenBLAS) inherit that mask. Binding is part of the It3 sweep.
- **Machine:** 4× Zen 5 (CPUs 0–7, up to 5.09 GHz) + 6× Zen 5c (CPUs 8–19, up to 3.32 GHz). At setup, `scaling_max_freq` was pinned to 2.0 GHz on all cores (balanced profile, on AC).
  - Lift it for benchmarking with `powerprofilesctl set performance; sudo cpupower frequency-set -u 5.1GHz`.
  - `perf` needs `kernel.perf_event_paranoid` ≤ 2 (currently 4); use `py-spy record --native` instead.
- **Build flags as found:**
  - PySCF stock: `-msse3 -O2 -g`.
  - libcint stock: `-O3` with no `-march`.
  - qcint: `-march=native -ffast-math -O3` (SIMDD=8).
  - libxc: `-O3 -std=c99` (ISO C, so FP contraction is off).

## Noise floor (It0)

Kernel tests on caffeine/def2-SVP, max |Δ| / max(1, |ref|):

| comparison | Vxc / ρ / J / K | DF-J |
|---|---|---|
| identical builds, repeated runs (OpenMP reduction order) | ≤ 2e-16 | 3e-14 |
| BLAS kernels Barcelona vs Cooperlake | 2–9e-14 | 8e-14 |
| 1 vs 10 threads | 1–7e-14 | 3e-13 |

Rounding-level noise is therefore about 1e-13 relative. The kernel test tolerance (1e-11) leaves two orders of magnitude of margin.

## Baseline profile before any changes (pip wheel 2.12.1, 10 threads, 2 GHz cap; from planning)

| workload | 4c J/K | lib.dot GEMM | scale/contract | AO eval | LibXC |
|---|---|---|---|---|---|
| caffeine def2-SVP PBE / B3LYP / r2SCAN SCF | 61 / 63 / 45% | 25 / 24 / 42% | 3–5% | 5–7% | ≤0.5% |
| caffeine def2-TZVP DF-PBE SCF | 23% | 58% | 4% | 10% | 0.3% |
| benzene def2-SVP B3LYP gradient / TDA / Hessian | 82 / 14 / 46% | 5 / 55 / 32% | 2 / 24 / 14% | ≤3% | ~0 |
| direct eval_ao + eval_rho + eval_xc (295k points) | – | 50–82% (`eval_rho`) | incl. | 17–50% | ≤1.4% |

## Experiment log

### It0: setup and baselines (2026-10-27)
- Clones, venvs, OpenBLAS, libcint, qcint, libxc, and the `pyscf` and `pyscf-base` trees are built.
- The harness is written: `bench/{molecules,cases,worker,common,run_bench,compare,validate,kernels}.py`.
- Build times: OpenBLAS 7.5 min, libxc 2.3 min, PySCF 11 s, libcint 38 s, qcint 55 s.
