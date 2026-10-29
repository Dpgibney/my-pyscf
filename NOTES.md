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
| `pyscf-g/` | worktree of `pyscf/` | `3ecf507dc`, detached | `perf` before It7 (G in the It7 benchmarks) |
| `pyscf-f/`, `pyscf-prev/` | worktrees of `pyscf/` | `8fc5eca4e`, `ca881ddc3`, detached | `perf` before It6 and before It5b, for before/after runs |
| `pyscf-native/` | worktree of `pyscf/` | `8fc5eca4e`, detached | `BUILD_MARCH_NATIVE=ON` build (It3) |
| `pyscf-qcint/`, `pyscf-qcint-avx2/` | worktrees of `pyscf/` | `44313dfb45`, detached | unmodified master on qcint AVX-512 / AVX2 (It2, It6 SGX check) |
| `libcint/` | github.com/sunqm/libcint | `v6.2.0` (master's pin) | integrals |
| `qcint/` | github.com/sunqm/qcint | `v6.2.0` | SIMD integrals (It2); `perf` has the It8 changes, and the `pyscf` tree loads its profile-guided build `install/qcint-pgo` |
| `libxc/` | gitlab.com/libxc/libxc | `7.1.2` (master's pin) | XC functionals |
| `OpenBLAS/` | github.com/OpenMathLib/OpenBLAS | `v0.3.34` | BLAS (build-only so far) |

- **`bench/diag/`** holds the one-off scripts behind the numbers quoted here, indexed in its README.
- **Python:** two uv venvs on a uv-managed CPython 3.12.8, with identical pins (numpy 2.5.4, scipy 1.18.1, h5py 3.16.0, pytest 9.2.0).
  - `.venv` runs the source trees; `PYTHONPATH` picks the tree.
  - `.venv-wheel` has the PyPI `pyscf==2.12.1` wheel: baseline "A", what the user runs today.
  - The anaconda base env is never touched.

## How to

```bash
# dependencies (install/<name>), then a PySCF tree wired to a chosen cint/libxc build
scripts/build_openblas.sh
scripts/build_cint.sh libcint stock            # or: qcint stock | libcint native -DBUILD_MARCH_NATIVE=ON
scripts/build_qcint_pgo.sh                     # qcint perf branch, profile-guided -> install/qcint-pgo (It8)
scripts/build_libxc.sh stock                   # or: build_libxc.sh znver5 -DCMAKE_C_FLAGS=-march=znver5
scripts/build_pyscf.sh pyscf libcint-stock libxc-stock [extra cmake args]   # the perf tree: pyscf qcint-pgo libxc-stock

scripts/check_power.sh                          # benchmark-ready? (AC, performance profile, no freq cap)
cd bench
../.venv/bin/python kernels.py new=pyscf        # fixed-input kernel tests vs pyscf-base (rtol 1e-11)
../.venv/bin/python validate.py new=pyscf [--suite full]   # tight-convergence results vs pyscf-base
../.venv/bin/python run_bench.py --tag NAME A=wheel B=pyscf-base:OPENBLAS_CORETYPE=Barcelona new=pyscf
../.venv/bin/python compare.py results/<file>.json [--base B]
bench/qcint/qbench install/qcint-pgo/lib/libcint.so.6 scratch/qcint/caffeine.bin int2e [--compare LIB2]   # It8
```

- **Config syntax:** `LABEL=TREE[:VAR=VAL,...]`. TREE is a PySCF tree directory, or `wheel`. VARs override the pinned worker environment, e.g. `OPENBLAS_CORETYPE=SkylakeX` or `OMP_NUM_THREADS=1`. `LD_LIBRARY_PATH=$WS/install/qcint-<variant>/lib` swaps in another qcint build of the same ABI (It8); the label still shows the tree's linked variant, while each result records the libraries actually loaded.
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
- **Machine:** 4× Zen 5 (CPUs 0–7, up to 5.09 GHz) + 6× Zen 5c (CPUs 8–19, up to 3.32 GHz).
- **The firmware clamps the clocks, and not consistently.** `scaling_max_freq` reads 2.0 GHz on every core, even on AC.
  - `sudo cpupower -c all frequency-set -u 5.1GHz` ran (confirmed in the sudo log), but `scaling_max_freq` stayed at 2.0 GHz.
  - Not thermal: every Processor cooling device is at state 0, and the hottest zone was 55 °C.
  - Likely the HP firmware (`hp-wmi`) applying a platform limit: an underpowered or unrecognized adapter, or a BIOS power/turbo setting.
  - It1 ran entirely at the 2.0 GHz clamp. From It2 on, the machine toggled between the clamp and boost (see "Power state is not constant" below), so every run records its measured clocks.
  - This slows all of the user's computations, not just this project.
- `perf` needs `kernel.perf_event_paranoid` ≤ 2 (currently 4); use `py-spy record --native` instead.
- **TD `conv_tol` is a residual-norm tolerance** (`tdscf/_lr_eig.py`), and the solver drops corrections shorter than sqrt(`lindep`) = 1e-6. With the default `lindep`, residuals below ~1e-7 are out of reach, so a smaller `conv_tol` silently runs to `max_cycle` (It6).
- **Direct-SCF energies drift once the density is stationary**, by ~1e-10 Eh per cycle on n-C20H42/def2-TZVP. Each cycle adds the J/K of the density change to the previous one, and the screening error accumulates. Tight validation of that case rebuilds J/K every cycle (`bench/cases.py:_full_jk_rebuild`) (It6).
- **A tight direct SCF stops at a run-dependent point, in every build** (It7).
  - Direct J/K are summed in dynamic thread order, so at `conv_tol=1e-12` (a few ulp of a 680 Eh energy) the stopping cycle varies. On caffeine/B3LYP it took 24–27 cycles in base, G and the It7 build alike, and the converged densities differed by up to 6e-9.
  - The energy cannot show this, but TD excitation energies move with it, by ~2e-9 Eh: over the 1e-9 gate. Earlier passes were lucky.
  - A tighter `conv_tol_grad` doesn't help. With incremental J/K, the orbital gradient floors at ~2e-8; 1e-10 needs full J/K rebuilds and 70–79 cycles.
  - Post-SCF validation cases (gradients, TD, Hessians) therefore start from the reference's SCF solution: `cases.converged_ks`, `BENCH_SCF_OUT`/`BENCH_SCF_IN`, cached next to the reference. The `scf_` cases still validate the SCF itself.
- **qcint fails two SGX gradient tests.** Translational invariance is 7.1e-12 against a 5e-12 tolerance (libcint: 1e-14) (It6).
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

`validate.py` noise floor: the `pyscf` tree, built identically, against `pyscf-base` on the quick suite, tight convergence.

| quantity | max \|Δ\| | tolerance |
|---|---|---|
| SCF energy | ≤ 9e-13 Eh | 1e-9 |
| benzene gradient | 1.0e-10 | 1e-8 |
| benzene TDA excitations | 5.3e-11 Eh | 1e-9 |
| grid checksums | identical | — |

- At `conv_tol=1e-12` the cycle count varies even between identical builds (24 vs 27), because the last iterations jitter at the quadrature noise floor. Cycle differences are therefore warnings, not failures.
- Integral variants (It2 correctness, kernel tests): libcint `-march=native`, qcint AVX-512 and qcint AVX2 all match the stock libcint J/K/DF-J to 2e-14 to 1e-13 relative, the same as the BLAS noise.

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

#### Two upstream changes between the 2.12.1 wheel and master that affect comparisons with A
1. **Master no longer prunes the grid by density by default.** `small_rho_cutoff` went from 1e-7 to 0 in `1b0caaee5` (#3192, 2026-04-20).
   - Caffeine/SVP SCF therefore integrates 294,800 grid points on master versus 247,368 in the wheel: 19% more XC work.
   - It is also why master's last grid block takes the sparse path (`_dot_ao_ao_sparse` self-time).
   - Exact-only policy keeps master's default. `mf.small_rho_cutoff = 1e-7` restores the old behaviour if the user wants it.
2. **LibXC 7.1.2 (master's pin) dropped r2SCAN 3rd/4th derivatives.**
   - The regenerated `maple2c/mgga_exc/mgga_{x,c}_r2scan.c` carry only EXC/VXC/FXC flags, so `max_deriv_order` is 2 (7.0.0 had 3). SCAN still has KXC/LXC.
   - This breaks r2SCAN TDDFT excited-state gradients (they need kxc) on upstream master too.
   - It is an upstream regression; flagged to the user, not patched.

#### It0/It1 timings (fixed 2 GHz, 10 threads, median of 3, interleaved; `results/20261027-185848_it0-it1-blas.json`)

Configs: A = wheel; B = master stock with `OPENBLAS_CORETYPE=Barcelona` (wheel-like kernels); Bzen / Bskx = master stock with modern kernels.

| case | A (s) | B (s) | Bzen (s) | Bskx (s) | `_dgemm` B → Bzen | Bzen vs B | Bzen vs A |
|---|---|---|---|---|---|---|---|
| caffeine SVP PBE SCF | 49.1 | 52.1 | 46.2 | 46.2 | 14.0 → 8.1 s | 1.13× | 1.06× |
| caffeine SVP B3LYP SCF | 49.1 | 51.4 | 45.5 | 45.8 | 13.3 → 7.1 s | 1.13× | 1.08× |
| caffeine SVP r2SCAN SCF | 42.5 | 46.3 | 37.2 | 37.2 | 19.2 → 10.2 s | 1.25× | 1.14× |
| benzene B3LYP gradient | 2.63 | 2.77 | 2.59 | 2.71 | 0.15 → 0.10 s | 1.07× | 1.02× |
| benzene B3LYP TDA (5 states) | 10.2 | 11.6 | 9.26 | 9.17 | 6.0 → 3.9 s | 1.25× | 1.10× |
| direct grid pipeline | 5.53 | 5.46 | 4.11 | 4.06 | 3.0 → 1.7 s | 1.33× | 1.35× |

- **Micro GEMM** (`lib.dot`, GFLOP/s):
  - Wheel 119–174.
  - B 155–175.
  - Bzen 253–374; Zen is best on the numint shapes (289 for 246×67200×246).
  - Bskx 228–367; SkylakeX is best on square 2000³ (355).
- **It0 prediction** (B within ±15% of A): ✔. B's 4–12% deficit is the 19% larger grid, not BLAS.
- **It1 predictions:**
  - ✔ `_dgemm` −42…47% (predicted ≥35%).
  - ✔ `lib.dot` at 4000³ reaches 374 GFLOP/s (predicted ≥180).
  - ✔ TDA −21% (predicted about −25%).
  - ✔ Zen ≈ SkylakeX within ±10%.
- **Wall-time gains are modest** because the integrals (`jk`, about 29 s of 46 s for caffeine) are untouched by BLAS. That is It2.
- **Decision:** from It2 on, use the build's default dispatch, which picks Cooperlake (the SkylakeX DGEMM). It is equal to Zen at the application level, and it is what the user gets with no environment variables set.

#### Power state is not constant (found during It2)
- The firmware **toggles between a 2.0 GHz clamp and boost** while `scaling_max_freq` keeps reading 2.0 GHz.
  - Boosted: Zen 5 at 3.8–4.4 GHz, falling as it heats; Zen 5c at 3.31 GHz.
  - It looks tied to power delivery: the battery drained 97% → 94% under load and then recharged.
- **It1 ran entirely clamped.** In It2 the caffeine cases ran boosted and the benzene cases clamped. The proof: `nr_direct_drv` was 29 s in It1 and 18–20 s in It2 for the same code and the same RSS, while benzene TDA and gradient times matched It1 exactly.
- **Comparisons within a case stay valid:** configs are interleaved per repetition and the spreads are tight. Absolute times across runs do not compare.
- **Fix:** `worker.py` now samples the real per-core-type clock (APERF/MPERF via `cpuinfo_avg_freq`) and battery state during every timed run, and `compare.py` shows the Zen 5 GHz range per config.

### It2: integral libraries (2026-10-27, `results/20261027-193653_it2-integrals.json`)

C = stock libcint (SSE2); Ccn = libcint `-march=native` (swapped in at runtime; same ABI); Cq = qcint AVX-512; Cq2 = qcint AVX2.

| case | C (s) | Ccn | Cq | Cq2 | jk C → Cq |
|---|---|---|---|---|---|
| caffeine SVP PBE SCF | 32.7 | 1.03× | 1.07× | 1.04× | 19.7 → 17.3 s |
| caffeine SVP B3LYP SCF | 32.6 | 1.05× | 1.06× | 1.04× | 20.4 → 17.9 s |
| benzene gradient | 2.75 | 0.98× | **1.36×** | 1.37× | 2.30 → 1.56 s |
| caffeine gradient | 14.9 | 0.98× | **1.32×** | 1.33× | 12.6 → 9.2 s |
| benzene Hessian | 50.9 | 0.98× | **1.13×** | 1.13× | 22.5 → 16.9 s |
| benzene TDA (ERIs in memory) | 9.18 | 1.00× | 1.00× | 1.00× | — |
| caffeine TZVP DF-PBE | 32.9 | 1.02× | 1.00× | 1.00× | — |

- **Predictions:** `nr_direct_drv` −30…50% held for derivative integrals (−27…−37%) but not for plain SCF ERIs (−10…12%). Gradient wall time −24% (predicted −30% or better).
- **Correctness:** all three variants passed the kernel tests at the noise floor.
- **Decision:** adopt **qcint (AVX-512 build)** for the `perf` tree. AVX2 and AVX-512 are equal on this chip's 256-bit datapath; AVX-512 should win on desktop Zen 4/5. libcint `-march=native` is not worth it.

### It4: low-risk exact code edits (`pyscf` perf branch, built against qcint)
Four commits on `perf`, one per change:
1. `12bc0f988` **NPdgemm**
   - The k-split partial products used to be summed serially in arbitrary order inside `omp critical`. They are now reduced in parallel, in fixed thread order, with `NPomp_dsum_reduce_inplace`.
   - `lib.dot` is now **bitwise reproducible** for a given thread count: 20 repeated calls are identical at 1, 3, 10 and 20 threads.
2. `f707acade` **Hessians honour `with_k`.** `_partial_hess_ejk` (RHF and UHF) ignored `with_k=False`, so pure-functional Hessians contracted the K patterns and then multiplied them by `hyb=0`.
3. `f9fd538db` **`eval_rho` symmetrizes real DMs.**
   - ρ, ∇ρ, τ and ∇²ρ depend only on (D+Dᵀ)/2. With the default `hermi=0` and a GGA, `eval_rho` did a second `ao·dmᵀ` GEMM and three extra contractions.
   - This matters for direct-grid scripts.
4. `4292af173` **dsyrk for the symmetric dense XC terms.**
   - Covers the LDA `aoᵀ diag(w) ao` term and the meta-GGA τ term, Σₓ ∇ₓaoᵀ diag(w) ∇ₓao.
   - Grid points are split by the sign of w and scaled by √|w|, then `P ᵀP − NᵀN` is accumulated with dsyrk on one triangle: half the flops of the dgemms it replaces.
   - Static chunk schedule plus fixed-order reduction keep the dense-path XC matrices bitwise reproducible (checked for LDA, PBE and r2SCAN).

**Bug found and fixed during It4.** `NPomp_dsum_reduce_inplace` reads `vec[0]` *before* its internal barrier. Callers must publish their buffer pointers earlier and pass a barrier first; existing PySCF callers do this implicitly through an `omp for`. My first draft stored the pointer immediately before the call: a race that corrupted the heap (`munmap_chunk(): invalid pointer`) or gave 2e-7 errors. AddressSanitizer on a standalone C test pinpointed it.

**Correctness:**
- Kernel tests, including the new UKS, f_xc and non-symmetric-DM outputs, match the reference to about 1e-14.
- `validate.py` passes on the quick suite plus PBE/B3LYP Hessians and UKS r2SCAN. Hessian |Δ| ≤ 2e-11.
- **r2SCAN tight-SCF noise floor:** below |g| ≈ 1e-8 the r2SCAN SCF stalls, so meeting `conv_tol=1e-12` takes a variable number of cycles (24–70 seen, for both the base and It4 builds). Final energies scatter by about 1e-11, with one 3.9e-10 outlier after a 70-cycle wander. This is a property of the functional and grid at that tolerance; it is not visible at the default 1e-9.

### It5a: Hessian XC term without the per-atom matrices
- **What `_get_vxc_deriv2` did:** built `vxc[natm,3,3,nao,nao]` with 18 Ng·nao² GEMMs per atom per grid block. `partial_hess_elec` (RKS, UKS, DF-RKS, DF-UKS) only ever contracts it with dm0, over AO rows of each atom and all columns.
- **New `_get_vxc_deriv2_dm`:** evaluates that contraction directly from atom-resolved pair densities Σ_{p∈B} (ao_k·ao_dm0_c)[g,p]. These are computed once per grid block for all atoms (`_atom_pair_sum`). The atom-independent `ipip` terms are kept as matrices and contracted at the end.
- **Speed on benzene/SVP** (prototype vs original, same tree): PBE 17.6 → 2.2 s, B3LYP 17.5 → 1.8 s, r2SCAN 28.0 → 3.3 s, LDA 3.9 → 0.7 s.
- **Agreement** with the original contraction is 1–4e-15 relative for RKS and UKS, across LDA/GGA/MGGA/hybrid.
- **Range-separated hybrids:** `partial_hess_elec` now starts the long-range K terms from zero and adds the precontracted XC part.

### It4 + It5a timings (`results/20261027-205222_it4-it5a.json`, boosted clocks)

Q = qcint, unmodified code; P4 = + It4; P5 = + It5a. All three use the same qcint/BLAS build.

| case | Q (s) | P4 | P5 | notes |
|---|---|---|---|---|
| r2SCAN SCF | 35.2 @3.6 GHz | 32.7 @3.35 GHz (1.08×) | 32.3 | τ dsyrk; P4 ran at about 7% lower clock |
| direct-grid pipeline | 3.98 | 3.63 (1.10×) | 3.57 | `eval_rho` symmetrization |
| UKS r2SCAN SCF | 6.14 | 5.69 (1.08×) | 5.74 | τ dsyrk via `nr_uks` |
| PBE Hessian | 41.8 | 40.7 (1.03×, `with_k`) | **32.1 (1.30×)** | |
| B3LYP Hessian | 45.2 | 45.0 | **36.8 (1.23×)** | |
| PBE SCF, TDA | — | unchanged | unchanged | as expected |

- **dsyrk vs dgemm:** the τ dsyrk kernel is 1.6× faster than 3× (scale + dgemm) on the realistic shape (3 components, 246 AOs, 67,200 points): 57 vs 94 ms. That is below the ideal 2× because of the gather pass and dsyrk efficiency. Chunk size (256 / 1024 / 4096) makes no difference.
- **Profiler pitfall:** a `FreqSampler` *thread* corrupted cProfile attribution in Python 3.12, showing main-thread C calls as `_thread.lock.acquire`. Sampling now runs in a separate process.

### It5b: batched f_xc contraction (`8fc5eca4e`)
- **The problem:** `nr_rks_fxc` (used by TDA/TDDFT and by the CPHF of Hessians/response) handled each trial DM separately on every grid block. That is two GEMMs plus contraction and scaling passes, each re-streaming the AO block. After It1, these bandwidth-bound passes cost as much as the GEMMs.
- **`_rks_fxc_batch`** (LDA/GGA, real DMs, dense blocks) instead does:
  - one GEMM for `dm_k·ao₀ᵀ` over all k;
  - `VXC_dcontract_rho_batch`, which computes ρ₁[k,c,g] reading each AO value once;
  - `VXC_dscale_ao_batch`, which forms `aow[k]` reading each AO value once;
  - one accumulating GEMM for all V_k.
- **Unchanged paths:** sparse blocks keep the per-DM kernels; MGGA and complex DMs use the old path.
- **Correctness:**
  - The batch kernels match `einsum` exactly.
  - f_xc kernel tests (LDA/PBE/B3LYP/r2SCAN, 3 non-symmetric trial DMs) match the reference to ≤ 7e-16.
  - TDA, TDDFT, both benzene Hessians and all nine Hessian-path cases pass `validate.py`.
- **Validation rule fix:** tight RPA-TDDFT (`conv_tol=1e-8`) does not converge on unmodified master either. `validate.py` now requires the candidate to converge wherever the *reference* did, instead of everywhere.

**It5b timings** (`results/20261027-213023_it5b-fxc-batch.json`, P5 = It5a vs P6 = It5b).
P6 ran 0.2–0.4 GHz slower than P5 in every pair (see below), so these gains are understated by about 5–10%.

| case | P5 (s) | P6 (s) | speedup | `xc_kernels` |
|---|---|---|---|---|
| benzene TDA | 9.12 | 8.74 | 1.04× | 3.09 → 2.79 s |
| benzene TDDFT (RPA) | 24.98 | 22.82 | 1.09× | 8.33 → 6.58 s |
| caffeine TDA | 133.1 | 127.2 | 1.05× | 27.6 → 18.6 s |
| PBE Hessian | 31.8 | 29.7 | 1.07× | 6.68 → 5.01 s |
| B3LYP Hessian | 36.5 | 34.1 | 1.07× | 7.37 → 5.53 s |

- **Caffeine TDA is now J/K-bound:** 64 of 127 s goes to recomputing the 4-center integrals in every Davidson iteration, because nao = 246 does not fit in memory at the default `max_memory=4000`.
- **Methodology fix: ABAB → ABBA.** With boost clocks, the config that always runs second sees a hotter chip and lower clocks. `run_bench.py` now alternates the order every repetition.

### It3: compile flags and memory configuration (`results/20261027-215849_it3-flags-mem.json`)

P6 = `perf` with stock flags; N = the same code with `-DBUILD_MARCH_NATIVE=ON`; M = P6 with `PYSCF_MAX_MEMORY=20000`.

| case | P6 (s) | N | M | `jk` P6 → M |
|---|---|---|---|---|
| caffeine PBE SCF | 29.9 | 0.98× | **1.96×** | 16.8 → 2.1 s |
| caffeine B3LYP SCF | 30.9 | 1.01× | **1.79×** | 18.2 → 4.6 s |
| caffeine r2SCAN SCF | 32.7 | 1.02× | **1.67×** | 15.2 → 1.9 s |
| caffeine TDA | 127.2 | 0.98× | 1.14× | 63.9 → 46.3 s |
| benzene gradient, TDA, Hessian; direct grid | — | 0.93–1.01× | 0.97–1.03× | already in memory or unaffected |

- **`-march=native` for PySCF's own C code: no net gain, so the stock flags stay.**
  - The XC kernel passes get 20–35% faster in SCF.
  - But J/K (libcvhf contraction code) is 4–6% slower and benzene TDA 7% slower.
- **In-memory ERIs are the biggest exact win available by configuration.**
  - PySCF keeps the s8 ERIs in memory when nao⁴/1e6 MB + RSS < 0.95·max_memory. The default 4000 MB misses caffeine/SVP (nao = 246, 3.7 GB) by a hair.
  - With 20000 MB on this 30 GB machine, every nao ≲ 370 qualifies.
  - Energies agree to 2e-12.
  - `scripts/pyscf-env.sh` sets this.
  - The TDA gain is smaller because each trial vector makes a full bandwidth-bound pass over the stored integrals.

**Thread sweep** (`results/20261027-222823_it3-threads.json`; `perf` tree; times relative to 10 threads)

| threads | caffeine PBE | caffeine r2SCAN | benzene TDA | direct grid | benzene Hessian |
|---|---|---|---|---|---|
| 10 (one per physical core) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8 | 0.87× | 0.88× | 0.92× | 1.09× | 0.97× |
| 20 (SMT; the OpenMP default when `OMP_NUM_THREADS` is unset) | **0.67×** | **0.72×** | **0.70×** | **0.73×** | **0.80×** |
| 10 + `PYSCF_MAX_MEMORY=20000` | **1.94×** | **1.70×** | 1.01× | 0.99× | 1.00× |

- **SMT hurts everything**, which falsifies the plan's prediction that it would help integrals:
  - J/K gains nothing (18.8 vs 16.6 s);
  - BLAS and XC kernels slow down;
  - scipy `eigh` goes from 0.4 to 3.0 s, from thread-pool contention.
- **The user's shell does not set `OMP_NUM_THREADS`**, so their PySCF runs use 20 threads. `scripts/pyscf-env.sh` now defaults it to 10.

### Upstream tests on the final `perf` tree (`scripts/run_tests.sh pyscf`)
- dft, scf, df, grad, hessian and tdscf, with upstream's `-k "not _high_cost and not _skip"`: **839 passed, 4 failed, 57 skipped** in 5 min.
- All four failures are `AttributeError: module 'pyscf.dft' has no attribute 'xcfun'`, because the workspace builds with `ENABLE_XCFUN=OFF`:
  - `test_tdrks.py::{test_custom_rsh, test_tda_b3lyp_xcfun, test_tda_lda_xcfun, test_tddft_b3lyp_xcfun}`.
  - They fail identically on unmodified `pyscf-base`.
- **No regressions.**

## Final comparison against the wheel as used today (`results/20261027-225626_final.json`)

This is the state before It6. The It6 section below compares against it and against A20.

- **A20** is the user's current setup: the pip wheel 2.12.1 with `OMP_NUM_THREADS` unset, so 20 threads.
- **A** is the same wheel at 10 threads.
- **B** is unmodified master with our OpenBLAS and stock libcint.
- **F** is `perf` (`8fc5eca4e`, qcint).
- **FM** is F with `PYSCF_MAX_MEMORY=20000`, which is what `scripts/pyscf-env.sh` sets.
- Runs used boost clocks (Zen 5 at 2.7–4.5 GHz depending on the case), 2 reps in ABBA order; spreads are ≤ 2%.
- Speedup = A20 time / config time.

| workload | A20 (s) | A | B | F | FM |
|---|---|---|---|---|---|
| caffeine/SVP PBE SCF | 46.0 | 1.31× | 1.39× | 1.51× | **3.06×** |
| caffeine/SVP B3LYP SCF | 45.3 | 1.29× | 1.37× | 1.46× | **2.62×** |
| caffeine/SVP r2SCAN SCF | 52.2 | 1.22× | 1.38× | 1.60× | **2.69×** |
| caffeine/TZVP DF-PBE SCF | 48.3 | 1.20× | 1.45× | 1.47× | 1.47× |
| caffeine⁺/SVP UKS B3LYP SCF | 87.6 | 1.30× | 1.42× | 1.50× | **2.24×** |
| benzene/SVP B3LYP gradient | 2.84 | 1.11× | 1.05× | 1.39× | 1.40× |
| caffeine/SVP B3LYP gradient | 13.5 | 0.91× | 0.90× | 1.18× | 1.18× |
| benzene/SVP B3LYP TDA, 5 states | 13.0 | 1.29× | 1.40× | 1.51× | 1.52× |
| benzene/SVP B3LYP TDDFT, 5 states | 36.8 | 1.31× | 1.47× | 1.59× | 1.61× |
| benzene/SVP B3LYP Hessian | 61.5 | 1.14× | 1.20× | **1.79×** | 1.79× |
| direct grid pipeline, caffeine/SVP | 7.04 | 1.31× | 1.72× | **2.00×** | 1.97× |

**Where the gain comes from** (each step relative to the previous column):
- **20 → 10 threads:** 1.20–1.31×. Gradients are the exception (0.91×), see below.
- **Master with a current BLAS:** 1.00–1.31×. This holds even though master integrates 19% more grid points.
- **`perf` code and qcint:** 1.01–1.49×.
  - Per workload, F vs B: SCF 1.07× (PBE, B3LYP), 1.16× (r2SCAN), 1.05× (UKS), 1.01× (DF); gradients 1.31–1.33×; TDA/TDDFT 1.08×; Hessian 1.49×; direct grid 1.16×.
- **In-memory ERIs:** up to 2.0× more, for conventional (non-DF) SCF with nao ≲ 370.

**Accuracy** (default convergence, every recorded value):
- F and FM vs B: ≤ 8e-12 (energies, gradients, excitations, Hessian elements, grid checksums).
- **Wheel vs master: 1–4e-10 Eh in energies and excitations.** This is the grid-pruning default (#3192). With `small_rho_cutoff=1e-7`, master reproduces the wheel's benzene energy to 1e-13.
- **Wheel vs master: 5.2e-7 in Hessian elements.** This is master's CPHF fix "Prevent premature Krylov termination" (#3334, `ba2474364`).
  - With 2.12.1's `_qr`/`krylov` calls patched back in, master reproduces the wheel's Hessian to 3.7e-12 (benzene/SVP B3LYP, tight SCF).
  - So 2.12.1's CPHF stops early, and master's Hessians are the better-converged ones.

**Large cases** (1 rep each; `results/20261028-000210_final-large-tda.json`, `results/20261028-001636_final-large-alkane.json`):

| workload | A20 (s) | A | B | F | FM |
|---|---|---|---|---|---|
| caffeine/SVP B3LYP TDA, 5 states | 177.1 | 1.17× | 1.33× | 1.39× | 1.59× |
| n-C20H42/SVP B3LYP SCF (62 atoms, nao 490) | 86.1 | 1.06× | 1.05× | 1.04× | (= F; ERIs would need 58 GB) |

- **Before It6, the 62-atom alkane gains almost nothing.** Its cost is direct-SCF J/K (46 s) and the XC work (27 s), and that XC work runs in PySCF's own sparse kernels (`nr_numint_sparse.c`), not in BLAS.
  - None of the changes so far touch those kernels, and qcint does not speed up its J/K (47 vs 46 s).
  - SMT does help this J/K (A20 40 s vs A 46 s), but the XC kernels lose more than that at 20 threads.
  - **The dense/sparse switch is not the problem.** `NumInt.nr_rks` with B3LYP, `perf` tree, best of 3:

    | system | default | forced dense | forced sparse |
    |---|---|---|---|
    | C20H42/SVP (nao 490, 694k points, 34% of shell×block screen entries nonzero) | 3.04 s | 3.01 s | 3.10 s |
    | caffeine/SVP (63% nonzero) | 0.55 s | 0.50 s | 1.40 s |

  - Upstream's switch is right for both. To make the alkane's XC cheaper, the sparse path needs BLAS-efficient sparsity, not a different threshold.
- **Gradients and threads** (`results/20261027-235822_grad-threads.json`):
  - The wheel's gradient is 1.10× faster at 20 threads (SSE2 libcint derivative integrals are latency-bound).
  - The `perf` build gains nothing from 20 threads (caffeine 11.16 vs 11.17 s) and loses 8% on benzene.
  - So 10 threads stays the default for every workload.

**Notes:**
- **DF-PBE/TZVP gets nothing from the code changes (F = B).**
  - Its cost is the GGA XC GEMM (13.4 of 33 s), DF-J (8.5 s) and AO evaluation (5.1 s).
  - GGA's `ao₀ᵀ·aow` product is not symmetric, so the dsyrk path does not apply.
  - PySCF already does one GEMM plus `hermi_sum`, which is the minimum flop count for that term.
- **With in-memory ERIs, the XC part is now 75% of caffeine PBE SCF:** XC GEMMs 5.8 s, AO evaluation 2.9 s and XC kernels 2.3 s, out of 15 s.

## It6: sparse XC with BLAS (gather → `dgemm`/`dsyrk` → scatter), 2026-10-28

`pyscf` `perf` commits `21d1e9b48` (A), `7e6ca19c5` (B), `c2c1caa76` (C), `3ecf507dc` (D); harness `67c0978`.

**Why.** For molecules large enough that `block_loop` takes the screened path (n-C20H42/def2-SVP: nao 490, 34% of the shell × 56-point-block screen index nonzero; always when nao > 800), the XC work ran in `nr_numint_sparse.c`, which computes every AO pair as a dot product over grid points (8-wide accumulators, `-msse3`).
- `NumInt.nr_rks` (B3LYP) on the alkane took 3.07 s per call: `_dot_ao_ao_sparse` 1.64 s, `_dot_ao_dm_sparse` 0.55 s, AO evaluation 0.60 s, contractions and scaling 0.24 s, LibXC 0.02 s.
- **Sparsity is spatially compact.** Each 56-point block has on average 154 of the 490 AOs unscreened (max 196). Groups of 2/4/8/16 consecutive blocks have 156/159/161/164, because the grids are sorted into boxes. Gathered dense products therefore need only 0.10–0.12 of the dense flops.

**Change** (four commits):
- **A** `VXCdot_aow_ao_sparse_gemm`, `VXCdot_ao_dm_sparse_gemm`:
  - Per group of 4 blocks (224 points), gather the AOs of the shells unscreened on any of the 4 blocks into contiguous buffers. Values on screened blocks are gathered as zeros, so only unscreened entries are read, as before.
  - Multiply with `dgemm`, or with `dsyrk` for symmetric products (points split by the sign of the weight), then scatter-add.
  - Thread-private accumulation with a static round-robin schedule, reproducible for a given thread count.
  - `_dot_ao_ao_sparse`/`_dot_ao_dm_sparse` call them.
- **B** fused kernels:
  - `VXCeval_rho_sparse_gemm`: ρ, ∇ρ, τ of a symmetric dm (`eval_rho1` with hermi=1).
  - `VXCdot_ao_aow_sparse_gemm`: `_dot_ao_aow_sparse` replaces `_scale_ao_sparse` + `_dot_ao_ao_sparse` at 14 call sites.
  - `VXCdot_ao_wv_ao_sparse_gemm`: the meta-GGA τ term as one `dsyrk` over the three components.
- **C** `_dot_ao_ao`/`_dot_ao_dm` take the gather kernels when given a screen index for the whole molecule (gradients, Hessians, `eval_rho2`), with the new rectangular `VXCdot_ao_mo_sparse_gemm`. Before, they multiplied the full dense matrices for nao < 800, and above that used 56 × 56 × 56 GEMMs per AO box (`VXCdot_ao_ao`, `omp critical`).
- **D** `block_loop` takes the screened path while less than 90% of the screen index is nonzero, instead of 50% (`dft_numint_sparse_threshold`). This moves medium molecules onto the new kernels.

**Exactness.** The new kernels drop only exact zeros. Upstream's sparse kernels also drop products estimated below 1e-13 (`nbins`, `pair_mask`), so the new sparse path is *closer* to the dense result.
- PBE, MO-tagged density, alkane, level-2 grid: upstream sparse vs its own dense path 7.8e-13 (Vxc) and 1.1e-12 (gradient XC term); new sparse vs dense 7.4e-16 and 1.3e-15.
- `bench/kernels.py` now has 22 sparse-path entries on the alkane (LDA/PBE/r2SCAN RKS and UKS Vxc, f_xc, MO-tagged densities, gradient XC term). All are within 1.1e-12 of `pyscf-base`; the largest are the PBE MO-density cases above.
- Kernel micro-checks with NaN written into every screened block: never read.
- **Tight validation** (FULL + the new SPARSE suite, 19 cases, vs `pyscf-base`): energies ≤ 5.5e-12 Eh, gradients ≤ 1.0e-10, Hessians ≤ 2e-11, excitation energies ≤ 1.8e-13 Eh.

**Kernel timings** (one 67 200-point block of the alkane, 10 threads):
- `ao^T·aow` 107 → 19 ms (dense `dgemm` 78 ms); `ao·dm` 63 → 28 ms (dense 108 ms).
- Group size 2/4/8 blocks: no difference.
- Schedule for the XC matrix: static contiguous 19.4–20.3 ms, static round-robin 18.5–18.8 ms, dynamic 17.3–17.6 ms (Zen 5 / Zen 5c speed difference). Round robin kept for reproducibility.

**Per call on n-C20H42/def2-SVP** (10 threads, MO-tagged density as in SCF; F = before It6, then after each commit):

| | F | A | B | C |
|---|---|---|---|---|
| `nr_rks` B3LYP | 3.08 s | 1.36 s | 1.11 s | 1.16 s |
| `nr_rks` r2SCAN | 6.35 s | 2.43 s | 2.40 s | 1.97 s |
| gradient XC term (`grad.rks.get_vxc`), B3LYP | 10.9 s | 10.8 s | 10.5 s | 6.13 s |
| same, r2SCAN | 19.5 s | 20.1 s | 20.2 s | 8.67 s |

**The 90% switch (D).** B3LYP, 10 threads at the 2.0 GHz cap, every block forced dense → screened, interleaved, best of 3:

| system | nao | nonzero | `nr_rks` | `nr_rks_fxc` (10 dms) | gradient XC |
|---|---|---|---|---|---|
| water/SVP | 24 | 0.95 | 1.04× | 1.46× | 0.96× |
| benzene/SVP | 114 | 0.82 | 1.20× | 1.55× | 0.98× |
| benzene/TZVP | 222 | 0.77 | 1.14× | 1.48× | 0.96× |
| n-C8H18/SVP | 202 | 0.66 | 1.41× | 2.16× | 1.13× |
| caffeine/SVP | 246 | 0.63 | 1.16× | 2.05× | 1.15× |
| caffeine/TZVP | 494 | 0.58 | 1.36× | 1.94× | 1.25× |
| n-C12H26/SVP | 298 | 0.51 | 1.77× | 3.09× | 1.42× |

- Water stays on the dense path.
- Between 0.75 and 0.9 nonzero, the gradient XC term is 2–4% slower on the screened path. It is a small part of a gradient, where the derivative integrals dominate.
- Caffeine/SVP is the median of three runs (`nr_rks` 1.13–1.21×). An earlier, non-interleaved session gave 1.50×.

**Benchmarks** (`results/20261028-112117_it6-medium.json`, 2 reps ABBA; `results/20261028-114255_it6-alkane.json`, 1 rep):
- **F** is `perf` before It6 (`8fc5eca4e`, worktree `pyscf-f`); **G** is `3ecf507dc`. Both use qcint and `PYSCF_MAX_MEMORY=20000`, so ERIs are in memory where they fit.
- Boost clocks (Zen 5 at 2.8–4.6 GHz, recorded per run). The **A20** column is the user's original setup (2.12.1 wheel, 20 threads), taken from the final comparison above. F reproduces that comparison's FM timings to within 2%, so the cross-session ratio holds.

| workload | A20 (s) | F (s) | G (s) | G vs F | G vs A20 |
|---|---|---|---|---|---|
| caffeine/SVP PBE SCF | 46.0 | 15.16 | 11.83 | 1.28× | **3.9×** |
| caffeine/SVP B3LYP SCF | 45.3 | 16.98 | 13.98 | 1.21× | **3.2×** |
| caffeine/SVP r2SCAN SCF | 52.2 | 19.36 | 14.02 | 1.38× | **3.7×** |
| caffeine/TZVP DF-PBE SCF | 48.3 | 32.95 | 26.77 | 1.23× | 1.8× |
| caffeine⁺/SVP UKS B3LYP SCF | 87.6 | 38.70 | 31.48 | 1.23× | 2.8× |
| benzene/SVP B3LYP gradient | 2.84 | 2.06 | 2.12 | 0.97× | 1.3× |
| caffeine/SVP B3LYP gradient | 13.5 | 11.23 | 11.03 | 1.02× | 1.2× |
| benzene/SVP B3LYP TDA, 5 states | 13.0 | 8.44 | 5.75 | 1.47× | 2.3× |
| benzene/SVP B3LYP TDDFT, 5 states | 36.8 | 22.79 | 15.73 | 1.45× | 2.3× |
| caffeine/SVP B3LYP TDA, 5 states | 177.1 | 111.61 | 74.98 | 1.49× | 2.4× |
| benzene/SVP B3LYP Hessian | 61.5 | 34.38 | 29.05 | 1.18× | 2.1× |
| direct grid pipeline, caffeine/SVP | 7.04 | 3.63 | 3.55 | 1.02× | 2.0× |
| n-C20H42/SVP B3LYP SCF | 86.1 | 82.66 | 62.19 | 1.33× | 1.4× |
| n-C20H42/TZVP PBE SCF | – | 346.2 | 314.2 | 1.10× | |
| n-C20H42/SVP r2SCAN SCF | – | 121.7 | 69.8 | **1.74×** | |
| n-C20H42/SVP B3LYP gradient | – | 46.65 | 41.63 | 1.12× | |
| n-C20H42/SVP PBE TDA, 5 states | – | 228.9 | 96.6 | **2.37×** | |

- Every computed value agrees with F to ≤ 4.7e-12 (default convergence).
- G's runs often had lower clocks than F's (alkane TDA 3.1 vs 3.8 GHz, 1 rep), so the ratios understate the gain rather than overstate it.
- **Where the XC time went:**
  - Caffeine TDA: the XC response work fell from 62 to 26 s, and in-memory J/K (46 s) is now 61% of the run.
  - Alkane PBE TDA: XC fell from 164 to 33 s.
  - DF-PBE/TZVP had gained nothing from the earlier iterations. It now takes the screened path (58% nonzero), and its GGA XC GEMM (13.5 s) is gone.
- **Unchanged, as expected:**
  - Gradients: derivative integrals are 80–85% of them.
  - The direct grid pipeline: it calls `eval_ao`/`eval_rho` without a screen index, so it stays on the dense path.
- The worker's cost categories missed the new wrappers in these runs: `_dot_ao_aow_sparse` and `_eval_rho_sparse` appear under "other". They are now counted as XC kernels.

**Upstream tests:** dft scf df grad hessian tdscf sgx on `3ecf507dc`: **853 passed, 6 failed**, the same failures as before D.
- The 4 xcfun tests, as before.
- `sgx/test/test_dm_screening.py::{test_rhf,test_uhf}_dm_screening`: the SGX gradient with grid response is translationally invariant to 7.08e-12, against a 5e-12 tolerance (upstream with libcint: 1e-14).
  - **Caused by qcint, not by this iteration.** Unmodified master built with qcint (`pyscf-qcint`) and the previous `perf` state fail identically.
  - sgx was not in the earlier test selection.

**Three harness problems found while validating** (fixed in `67c0978`):
- **TD `conv_tol` is a residual-norm tolerance.** In this PySCF the eigensolver (`tdscf/_lr_eig.py`) also drops corrections shorter than sqrt(lindep) = 1e-6.
  - On caffeine the residual stalls near 1.5e-7 in both builds, so the tight setting of 1e-8 never converged. The unmodified build ran 85 min, adding 20 trial vectors per iteration.
  - The response operator itself is linear and symmetric to 1e-13 in both builds. With `lindep=1e-16` the same run reaches 1e-8 in 12 iterations.
  - Tight TD runs now use 1e-6, which leaves the excitation energies converged to ~1e-11 Eh.
- **Direct-SCF energies drift once the density is stationary.** `dft/rks.py:get_veff` adds the J/K of each density change to the previous cycle's, and the screening error of these updates accumulates.
  - On n-C20H42/def2-TZVP PBE the energy falls by ~1e-10 Eh per cycle from about cycle 13, so conv_tol=1e-11 is never met. After 100 cycles both builds had drifted by 1.4e-8 Eh.
  - With J/K rebuilt from the full density every cycle, unmodified PySCF converges in 18 cycles, within 2e-11 Eh of the energy before the drift.
  - This is likely also why the SVP alkane converges erratically at conv_tol=1e-12 (14–100 cycles, energies spread over 1e-10).
- **A reference that never converged passed the gate.** `validate.py` only required the candidate to converge where the reference did. It now fails the case.
- Outside the harness: PySCF's default TD `conv_tol=1e-5` is the same residual tolerance, which converges excitation energies to about 1e-9 Eh. Long direct-SCF runs on large molecules carry the drift in their reported energies.

## It7: J/K, 2026-10-28

`pyscf` `perf` commits `971adcac5` (A), `d4c2a25b0` (B), `f80cf77b2` (C), `8abbb1834` (D), `e904d7d84` (E); harness `98bdc34`, `f7d7d19`.

**Where J/K time went** (2 GHz cap, 10 threads; `bench/diag/jk_profile.py`, `jk_native_profile.py`, `grad_jk_native.py`, `jk_scaling.py`, `td_jk_calls.py`):

| build | 1 dm (SCF) | 20 dms (TDA step) |
|---|---|---|
| caffeine/SVP, ERIs in memory, J+K | 0.16 s | 5.9 s (J 1.8, K 4.5) |
| caffeine/SVP, direct, J+K | 1.26 s | 7.6 s |
| n-C20H42/SVP, direct, J+K | 5.2 s | 18.2 s (10 dms) |

- **One dm, direct:** integral evaluation (qcint) is 79% of a build at 1 thread, the contraction 17% and screening 1%.
- **Many dms, both paths:** the contraction dominates, because each dm is contracted separately by scalar kernels. In memory, J for 20 dms cost 40× J for one dm.
- **Gradient J/K** (caffeine, 1 thread): the derivative integrals are 84%, the contraction 14%.
- **Caffeine TDA, 5 states:** 10 in-memory J/K calls of up to 20 dms (hermi=0) took 47 of 76 s.
- **Thread scaling** of direct J+K (caffeine): 1/2/4/10 threads take 7.16/3.72/2.05/1.08 s.
  - Pinned to 4 Zen 5c cores it is 21% slower than on 4 Zen 5 cores at the same 2 GHz cap (2.58 vs 2.13 s), so 10 threads reach ~88% of the machine's capacity.
  - Blocks of 32 or 16 AOs instead of 64 gain 3–5%; thread binding gains nothing.

**A: in-memory J/K of several dms** (`nr_incore_batch.c`, from 2 dms).
- **K:** each packed ERI row (ij|kl), kl ≤ ij, read as a symmetric matrix M[k,l], yields all 8 index permutations of its integrals with two `dsymm` calls, M·Y[j] → Z[i] and M·Y[i] → Z[j].
  - Y holds rows i and j of all dms and of their transposes; Z accumulates K and Kᵀ.
  - The (ij|ij) element appears in both products and is halved. Symmetric dms need half the work, with K = Z + Zᵀ.
- **J:** packed rows in 64 × 1024 tiles, vj[rows] += P·dm[cols] and vj[cols] += Pᵀ·dm[rows] with `dgemm`.
- **Reproducibility:** static schedule and fixed-order reduction, so the result is bitwise reproducible for a given thread count.
- **Timings** (caffeine/SVP, 20 dms): 5.7 → 0.71 s (hermi=0) and 4.4 → 0.53 s (hermi=1). K runs at ~270 GFLOP/s, near this machine's peak at 2 GHz.
- **Gotcha:** with hermi=1 the per-dm kernel returns the lower triangle of K for *any* dm, and an upstream test (`test_uhf.py::test_energy_tot`) passes random non-symmetric dms with hermi=1. The symmetric formulations are therefore used only for dms symmetric to 1e-13; others take the general one and mirror its lower triangle.

**B: direct J/K of several dms** (`nr_direct_batch.c`).
- Each shell quartet is contracted with all dms in one pass. The dms and the thread-private sums are stored as [p,q,n], taken 4 dms at a time as vectors held in registers.
- The kernels are compiled for AVX2/FMA and generically, chosen at run time; PySCF itself builds with `-msse3`.
- **Screening:** the same per-quartet prescreen as `CVHFnr_direct_drv`, after a bound with max(dm_cond) that skips quartets the prescreen would reject.
- The thread-private sums use huge pages (`MADV_HUGEPAGE`): 5% on the alkane, nothing on caffeine.
- **Timings:** caffeine/SVP J+K for 20 dms 8.2 → 2.8 s (hermi=0) and 5.8 → 2.3 s (hermi=1); alkane J only for 20 dms 12.1 → 6.4 s.
- A first version with the dm loops in memory under the stock `-msse3` flags gave only 1.8×. The register-held 4-wide chunks gave the rest.
- **Upstream bug seen on the way:** `CVHFnrs8_vj_prescreen_block` tests `q_kl > 4*cutoff/rho` where the per-quartet J test implies `cutoff/(4*rho)`. Blocks are therefore skipped 16× too eagerly for J-only quartets. The effect is below 1e-12 here, and the batched driver does not use block screening.

**C: a single dm through the batched direct driver** (a scalar variant of the kernels).
- On a single SCF density: caffeine J+K 1.01 → 0.90 s and J only 0.92 → 0.85 s; alkane J+K 4.92 → 4.49 s and J only 4.31 → 3.93 s, i.e. 1.09–1.12×.
- Every direct SCF cycle now takes this path.

**D: DF J-only builds from an in-memory 3-center tensor.**
- For pure functionals, `df_jk.get_jk` called `get_j`, which evaluates the 3-center integrals twice in every call. It now builds `_cderi` once when `DF.build` would keep it in memory.
- It does so only for a Cholesky-decomposable metric. Both paths then fit with the same factorization; otherwise `cholesky_eri` would fall back to an eigendecomposition with a linear-dependency cutoff.
- **Timings** (caffeine/TZVP, 0.78 GB tensor built in 0.58 s): J 0.35 s → 0.035 s per call; the DF-PBE SCF 25.7 → 19.0 s in the same 16 cycles, energies 3e-12 Eh apart.

**E: in-memory J/K of a single symmetric dm in one pass** (`CVHFnrs8_incore_jk_1dm`).
- **What was slow:** the per-dm K kernel summed its dot products into elements of vk. The compiler must reload and store those at every step because they may alias the other updates, and the kernel also read and wrote along columns.
- **Change:** the formulation of A for one symmetric dm, reading the packed row in place, with J in the same pass. The inner loops are contiguous with 4-wide vector sums (AVX2/FMA and generic).
- **Timings** (caffeine/SVP): J+K 0.159 → 0.056 s, about the time it takes to stream the 3.7 GB of ERIs once. J alone stays at 0.043 s; K alone goes from 0.13 to 0.051 s.

**Exactness.**
- **Kernel tests** (`bench/kernels.py` v5): 13 J/K entries against `pyscf-base`, including in-memory and direct with 5 non-symmetric, 2 symmetric and 1 symmetric dm. All are within 6e-14; the difference comes from the integral library (qcint vs libcint).
- **Against the per-dm kernels of the same tree:** rounding level.
  - In memory: ≤ 6e-15.
  - Direct: ≤ 1e-12 relative for 20 dms. On the same input, upstream's own direct and in-memory J/K differ by 2.5e-12.
- **Tight validation:** SCF energies ≤ 7.7e-12 Eh (direct, and in memory at 20 GB); TDA excitations ≤ 2e-15 Eh with the shared SCF (below); Hessians ≤ 1.5e-10; DF-PBE 2.3e-12 Eh.
- **Upstream tests:** dft scf df grad hessian tdscf sgx give 856 passed and the known 6 failed (4 xcfun, 2 qcint SGX). There are new tests in `scf/test/test_vhf.py` (batched in-memory and direct, single dm, all hermi cases) and `df/test/test_df_jk.py`.

**Harness problem found while validating B: the tight SCF endpoint scatters between runs.**
- **Symptom:** the caffeine TDA case failed, with excitations 2.2e-9 Eh from the reference.
- **What it wasn't:** the batched J/K matched the old path to 1e-17 Eh on every Davidson call. From one fixed SCF solution, the TDA results of the old and new paths agreed to 3e-15.
- **Cause:** a tight direct SCF stops after 24–27 cycles depending on the summation order of its J/K, in base, G and It7 alike. The converged densities differ by up to 6e-9, which the energy does not show but excitation energies do.
- **Fix** (`98bdc34`): post-SCF cases (gradients, TD, Hessians) start from the reference's SCF solution, cached next to the reference. The caffeine TDA then agrees to 1e-15.

**Benchmarks** (`results/20261028-165620_it7-medium.json`, 2 reps ABBA; `…171459_it7-alkane.json` and `…173757_it7-direct-tda.json`, 1 rep).
- **G** is `perf` before It7 (`3ecf507dc`, worktree `pyscf-g`); **H** is `e904d7d84`. Both use qcint and `PYSCF_MAX_MEMORY=20000` unless noted.
- **Clocks:** boost. H's runs averaged 0–15% higher clocks than G's (shorter runs heat less), so the wall-time ratios overstate the gains a little. The J/K column (cProfile, J/K functions) shows the part It7 changed.
- **Cycles:** SCF cycle counts are identical in every pair.
- The A20 column is the user's original setup, from the final comparison above.

| workload | G (s) | H (s) | H vs G | J/K G → H (s) | H vs A20 |
|---|---|---|---|---|---|
| caffeine/SVP PBE SCF | 12.40 | 12.24 | 1.01× | 2.0 → 2.1 | 3.8× |
| caffeine/SVP B3LYP SCF | 14.01 | 11.71 | **1.20×** | 4.7 → 2.5 | 3.9× |
| caffeine/SVP r2SCAN SCF | 13.96 | 13.84 | 1.01× | 2.0 → 2.0 | 3.8× |
| caffeine/TZVP DF-PBE SCF | 26.73 | 18.92 | **1.41×** (1.35× at a fixed 2 GHz) | 8.7 → 0.3, plus the tensor build | 2.6× |
| caffeine⁺/SVP UKS B3LYP SCF | 31.24 | 26.95 | **1.16×** | 10.9 → 6.7 | 3.3× |
| benzene/SVP B3LYP gradient | 1.95 | 2.05 | 0.95× (noise) | 1.5 → 1.6 | 1.4× |
| caffeine/SVP B3LYP gradient | 11.22 | 11.09 | 1.01× | 9.4 → 9.3 | 1.2× |
| benzene/SVP B3LYP TDA, 5 states | 5.75 | 4.61 | **1.25×** | 1.34 → 0.18 | 2.8× |
| benzene/SVP B3LYP TDDFT, 5 states | 15.91 | 12.74 | **1.25×** | 3.6 → 0.5 | 2.9× |
| caffeine/SVP B3LYP TDA, 5 states | 75.95 | 35.24 | **2.16×** | 47.0 → 6.1 | **5.0×** |
| same with direct ERIs (`PYSCF_MAX_MEMORY=4000`) | 96.98 | 52.91 | **1.83×** | 67.8 → 23.5 | |
| benzene/SVP B3LYP Hessian | 29.59 | 27.70 | 1.07× | 16.9 → 15.0 | 2.2× |
| benzene/SVP PBE Hessian | 25.10 | 24.61 | 1.02× | 13.4 → 12.9 | |
| direct grid pipeline | 3.68 | 3.50 | 1.05× (noise) | – | 2.0× |
| n-C20H42/SVP B3LYP SCF | 62.83 | 59.07 | 1.06× | 47.7 → 44.3 | 1.5× |
| n-C20H42/TZVP PBE SCF | 319.0 | 297.7 | 1.07× | 284 → 263 | |
| n-C20H42/SVP r2SCAN SCF | 70.82 | 66.25 | 1.07× | 45.3 → 40.9 | |
| n-C20H42/SVP B3LYP gradient | 42.52 | 42.33 | 1.00× | 35.8 → 35.6 | |
| n-C20H42/SVP PBE TDA, 5 states | 98.65 | 72.05 | **1.37×** | 60.0 → 33.4 | |

- **Unchanged, as expected:**
  - SCFs that need only J with the ERIs in memory (PBE, r2SCAN): J was already bound by memory bandwidth.
  - Gradients: derivative integrals.
  - The integral part of direct SCF: the alkane SCFs gain only C's contraction savings.
- **Where the time is now:**
  - Caffeine TDA: J/K is 6 of 35 s, and the XC response (26 s) dominates.
  - Alkane SCFs: J/K is still 75–88%, nearly all of it integral evaluation.

## It8: qcint, 2026-10-28

`qcint` `perf` commits `49205fb` (A), `a95651f` (B); profile-guided build (C, `scripts/build_qcint_pgo.sh`); harness `bench/qcint/` (`qbench.c`, `dump_basis.py`, `eri_native_profile.py`), workspace `efc1186`, `f565c24`. The `pyscf` tree now links `install/qcint-pgo`.

**Harness.** `bench/qcint/qbench` loads a libcint/qcint build with `dlopen` and times its integral function over the
unique Schwarz-screened shell quartets of a molecule (dumped from PySCF by `dump_basis.py`), grouped by class
(angular momenta x primitive combinations), single thread pinned to one Zen 5 core. `--compare LIB2` loads a second
build with `dlmopen` and compares every integral. Same-ABI qcint builds swap into PySCF with `LD_LIBRARY_PATH`
(config env), so no PySCF rebuild is needed.

**Where the time goes** (stock qcint 6.2.0, AVX-512, caffeine/def2-SVP, 14.6M quartets, 5.6 s per pass):
- by primitive combinations per quartet: 1: 12% (3.6M quartets, scalar path), 2-7: 34% (6.2M, partly filled SIMD
  batches), >= 8: 54% (4.7M).
- the cost is overhead, not arithmetic: (ss|ss) with 29 primitive combinations 470 ns (~14 ns per combination beyond
  a fixed ~80 ns per quartet); (pp|pp) with 17 combinations 1.4 us.
- leaf functions (py-spy, PySCF's in-memory ERI build): CINT2e_loop 21.5%, CINTgout2e 14.1%, scalar Rys roots 12%,
  CINTg0_2e 8.9%, per-primitive contraction calls 7%, cart2sph and copies 7%, env setup 3%.

**A `49205fb`: segmented contractions summed directly; no hardware scatter.**
- For quartets whose shells each have one contracted function (all def2 sets), the coefficient of i is folded
  into the primitive prefactor (as j, k, l already were) and each SIMD batch is added straight into the output,
  instead of recording a contraction target per primitive quartet and calling CINTiprim_to_ctr for each one.
- GOUT_SCATTER (SIMDD = 8): a store and 8 scalar copies instead of `_mm512_i32scatter_pd` (microcoded on Zen 4/5): 2%.
- int2e 5.65 -> 4.99 s (1.13x), int2e_ip1 40.8 -> 35.9 s (1.14x), one core.
- Tried and dropped: a branch-free vectorized one-root Rys routine (all six intervals for every lane, vector exp):
  no gain; the scalar root code is cheaper than its profile share suggested, and AVX-512 runs at half rate on
  this core.
- AVX2 build of the same code: 7% slower than AVX-512 here (5.37 vs 5.00 s).

**Exactness.** qbench --compare against stock qcint, all quartets of water, benzene (def2-SVP, cc-pVDZ) and caffeine,
for int2e, ip1, ipip1, ip1ip2, ipvip1, cart, range-separated int2e: max difference 2.8e-14 on values up to 131 (as
large as qcint vs libcint); non-zero flags identical. libcint's test_int2e.py (adapted to the current API):
same pass/fail pattern as stock qcint.
- PySCF with each build swapped in (A, A+B, the PGO build): kernel tests pass. Tight validation of 22 cases passes,
  covering SCF, UKS, DF, gradients, TDA/TDDFT and all 11 Hessians (range-separated included).
  - Against unmodified PySCF: energies <= 2.7e-12 Eh, gradients <= 1.6e-12, excitations <= 4.6e-14 Eh,
    Hessians <= 6.9e-11.
- Upstream tests with the PGO build linked: 856 passed, the known 6 failed.

**B `a95651f`: the lanes of segmented batches summed once per quartet.** Each batch is added lane by lane to SIMD
accumulators (no transposition to per-lane blocks); the lanes are summed at the end of the quartet. int2e through
`_gout2e_acc`; the generated gout functions of the gradient and Hessian integrals (int2e_ip1, ip2, ipip1, ipvip1,
ip1ip2) through an accumulating mode of GOUT_SCATTER, switched on by a hidden thread-local only around them.
int2e 5.01 -> 4.78 s (5%), ip1 36.4 -> 35.4 s (3%).

**C: profile-guided build** (`scripts/build_qcint_pgo.sh`: instrumented build, ~4 min of qbench training on the
SCF, gradient, Hessian, Cartesian and range-separated integrals, rebuild with `-fprofile-use
-fprofile-partial-training`). int2e 4.73 -> 4.47 s (5.5%), ip1 35.3 -> 33.3 s (6%). LTO on top: nothing. Results
differ from the non-PGO build in rounding only (compiler choices under -ffast-math); short-range ERIs move by up to
1.7e-13, within the 3-6e-13 by which stock qcint and libcint differ on them.

**Cumulative, one core:** int2e 5.67 -> 4.47 s (1.27x), int2e_ip1 40.8 -> 33.3 s (1.23x).

**Harness: tight UKS SCF fragile.** scf_caffeine+_svp_b3lyp_uks with incremental direct J/K drifts by up to
5e-12 Eh per cycle near convergence, so conv_tol=1e-12 was met after 47-51 cycles or not within 100, by chance,
in every build (dev failed once, dev2 passed once). The case now rebuilds J/K every cycle (`f565c24`): 47-50 cycles
every time.

**Benchmarks** (`results/20261028-193453_it8-medium.json` 2 reps ABBA, `results/20261028-194620_it8-alkane.json` 1 rep).
- **Setup:** the same `pyscf` tree with stock qcint (H, via `LD_LIBRARY_PATH`) and with `qcint-pgo` (I).
  `PYSCF_MAX_MEMORY=20000`, boost clocks within a few % of each other in each pair. SCF cycle counts are identical.

| workload | H (s) | I (s) | I vs H | J/K H -> I (s) |
|---|---|---|---|---|
| n-C20H42/SVP B3LYP SCF | 58.6 | 48.8 | **1.20x** | 43.7 -> 33.7 (1.30x) |
| n-C20H42/TZVP PBE SCF | 291.4 | 252.9 | **1.15x** | 256.6 -> 218.2 (1.18x) |
| n-C20H42/SVP r2SCAN SCF | 65.8 | 56.5 | **1.17x** | 40.5 -> 31.2 (1.30x) |
| n-C20H42/SVP B3LYP gradient | 42.0 | 36.0 | **1.17x** | 35.3 -> 29.3 (1.21x) |
| n-C20H42/SVP PBE TDA | 71.2 | 65.1 | 1.09x | 33.1 -> 26.8 (1.23x) |
| benzene/SVP B3LYP gradient | 2.07 | 1.79 | **1.16x** | 1.62 -> 1.34 |
| caffeine/SVP B3LYP gradient | 10.95 | 9.58 | **1.14x** | 9.15 -> 7.74 |
| benzene/SVP B3LYP Hessian | 27.2 | 24.6 | 1.11x | 14.7 -> 12.2 |
| benzene/SVP PBE Hessian | 24.2 | 21.7 | 1.12x | 12.7 -> 10.2 |
| caffeine/SVP PBE, B3LYP SCF (ERIs in memory) | 12.0, 11.9 | 11.7, 11.8 | 1.01-1.03x | the ERIs are computed once |
| caffeine/SVP TDA, caffeine/TZVP DF-PBE SCF | 34.5, 19.3 | 34.3, 19.1 | 1.00x | no 4-center integrals per call |

- The alkane's direct J/K gains 1.30x, slightly more than the single-core int2e figure (1.27x). Its quartets are
  mostly contracted s/p shells, where the segmented path saves the most.
- **What is left in qcint:** the per-quartet fixed cost (~80 ns: environment setup, driver, cart2sph and copies),
  which dominates the quartets with 1-7 primitive combinations (47% of the time); and the scalar Rys roots for 2-3
  roots (~10%). Batching across quartets of one class would fill the SIMD lanes, but the 2D-integral kernels read
  each quartet's geometry as scalars, so that is a deep change.

## Next iterations (by measured remaining cost, after It8)
1. **Integral evaluation is still most of J/K**, even after It8's 1.2-1.3x.
   - Where it is: direct SCF on large molecules, gradients and Hessians.
   - In qcint (see It8):
     - the per-quartet fixed cost (~80 ns), which dominates quartets with few primitive combinations;
     - the scalar Rys roots for 2-3 roots (~10%; evaluating only the x intervals present in a batch might halve them);
     - batching across quartets of one class, a deep change.
   - Gradients: the s2kl loop evaluates 4 `int2e_ip1` quartets per s8 orbit, where 3 would do with translational
     invariance (-25%). That needs a driver that contracts the gradient inside the integral loop.
   - SMT for the direct J/K only (13% in It3) is a configuration choice.
2. **Remaining XC costs.**
   - The XC response is now the largest part of the caffeine TDA (26 of 35 s).
   - Screened-path `nr_rks`: AO evaluation is ~0.6 of 1.1 s on the alkane. Measure the zero-fill share first (`bench/diag/eval_ao_cost.py`).
   - The gradient XC temporaries in `_make_dR_dao_w`.
3. **Extend It5b batching** to `nr_uks_fxc` (UKS TDDFT/CPHF) and the meta-GGA τ term.
4. **Hessian `_get_vxc_deriv1`** (h1ao for the CPHF right-hand side): batch it like It5b.
5. **AO evaluation on the dense path** (`eval_gto`, 5–10%).
6. **Upstream:**
   - The 15 `perf` commits are PR candidates: six from It4/It5, four from It6, five from It7. All of It7's have unit tests.
   - The two qcint `perf` commits (It8) are candidates for sunqm/qcint.
   - Report the LibXC 7.1.2 r2SCAN kxc regression to LibXC.
   - Possible PySCF issues:
     - Direct SCF's incremental J/K drift on large molecules.
     - A TD `conv_tol` below the `lindep` floor silently runs to `max_cycle`.
     - `CVHFnrs8_vj_prescreen_block` tests `q_kl > 4*cutoff/rho`, 16× stricter than its per-quartet J test.
7. **Hardware:**
   - The firmware 2 GHz clamp and boost toggling decide benchmark reliability.
   - Faster code runs at higher average boost clocks, which skews wall-time ratios (It7).
