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
| `pyscf-it11/` | worktree of `pyscf/` | `474f6366c`, detached | `perf` before It12, for before/after runs |
| `pyscf-it12/` | worktree of `pyscf/` | `1ccef1836`, detached | `perf` before It13, for before/after runs |
| `pyscf-it13/` | worktree of `pyscf/` | `d88076cb8`, detached | `perf` before It14, for before/after runs |
| `pyscf-it15/` | worktree of `pyscf/` | `84f2271ca`, detached | `perf` before It16 (stock flags) |
| `pyscf-it16a/`, `pyscf-it16c/` | worktrees of `pyscf/` | `930f39c41`, `a8a68d87f` | It16 steps A and A+C, built with `-DNUMINT_X86_64_V3=ON` |
| `pyscf-it16/` | worktree of `pyscf/` | `0835ee427`, detached | `perf` before It17, built with `-DNUMINT_X86_64_V3=ON` |
| `pyscf-it17a/`, `pyscf-it17b/` | worktrees of `pyscf/` | `d3545eefb`, `3ad92c150` | It17 steps A and A+B, built with `-DNUMINT_X86_64_V3=ON` |
| `pyscf-it17/` | worktree of `pyscf/` | `1e31fb4d1`, detached | `perf` before It18, built with `-DNUMINT_X86_64_V3=ON` |
| `pyscf-it18/` | worktree of `pyscf/` | `219115c29`, detached | `perf` before It19, built with `-DNUMINT_X86_64_V3=ON` |
| `pyscf-it19/` | worktree of `pyscf/` | `06b7253bc`, detached | It19, for its gate, built with `-DNUMINT_X86_64_V3=ON` |
| `pyscf-it20/` | worktree of `pyscf/` | `9ab2db1e0`, detached | It20, for its gate, built with `-DNUMINT_X86_64_V3=ON` |
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
                                               #   -DNUMINT_X86_64_V3=ON (It16: AVX2 numint/GTO grid kernels, bitwise)

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

## It9: semi-direct SCF, 2026-10-28

`pyscf` `perf` commit `0a6af55a2`; harness: `bench/diag/semidirect_stats.py` (+ `semidirect_count.c`),
`semidirect_profile.py`, `bench/qcint/qcost.c`, `bench/pyscf_conf.py` (`BENCH_SEMI_DIRECT=0|1`), kernels v6,
cases `scf_caffeine_tzvp_b3lyp` and `scf_caffeine_svp_camb3lyp`.

**Why.** Direct SCF evaluates every shell quartet it needs again in every cycle. For molecules whose s8 ERIs do
not fit in memory, the quartets that pass the Schwarz test (q_ij q_kl > `direct_scf_tol` = 1e-13) are a fraction
of the tensor:

| system | full s8 (GB) | Schwarz-significant (GB) | quartets |
|---|---|---|---|
| caffeine/def2-SVP | 4 | 2.5 | 1.5e7 |
| n-C20H42/def2-SVP | 58 | 9.7 | 8.6e7 |
| n-C12H26/def2-TZVP | 78 | 29 | 1.5e8 |
| caffeine/def2-TZVP | 60 | 41 | 1.1e8 |
| n-C20H42/def2-TZVP | 580 | 98 | 5.0e8 |

- n-C20H42/def2-SVP B3LYP, direct: 31.7 of 46.6 s in J/K, 11 calls.
- Every call evaluated 4.5-8.6 GB of these integrals (39-78M quartets). The incremental Fock build screens out
  at most half of them by the last cycle; the union over all calls is 8.6 GB.

**Change** (`_vhf.SemiDirectERI`, `nr_direct_batch.c`).
- `CVHFnr_direct_jk_batch` stores the integrals of a set of quartets when a call first evaluates them, and
  later calls read them. Quartets outside the set are evaluated as before.
- Each call screens exactly as without the cache (same `fprescreen`, same density bounds), so it contracts the
  same integrals: the results are direct SCF's, to rounding.
- **Layout:**
  - per pair row ij, the stored kl in loop order (int32 `ksh<<16|lsh`);
  - a state byte per quartet: not evaluated yet, stored, or zero;
  - the integral blocks back to back.
  - The index costs 5 bytes per quartet. Memory is committed as blocks are first stored, so quartets no call
    needs cost no RAM.
  - The loop over the quartets not in the set skips stored ones by merging with the row's sorted list.
- **Safety:** a cache is used only by calls with the atm/bas/env it was made for (geometry, basis, omega,
  integral cutoffs) and the same q_cond. So `with mol.with_range_coulomb(...)`, another geometry or a changed
  cutoff falls back to direct evaluation.
- **`SCF.semi_direct`** (default True; config `scf_hf_SCF_semi_direct`):
  - `SCF.get_jk` attaches a cache to each direct-SCF `vhfopt`, the range-separated ones included, within 3/4 of
    the memory left under `max_memory`.
  - TDDFT/CPHF response J/K through `mf.get_jk` read the same cache.
  - Setting it to False releases the integrals at the next J/K call.

**Which quartets, when not all fit.** Keeping the quartets with the largest q_ij q_kl helped little: on the TZVP
alkane, 14.7 GB of them saved ~15% per cycle (224.2 s vs 256.7 s direct).
- **Time per integral varies ~20-fold between classes** (`bench/qcint/qcost.c`, one core, qcint-pgo): from 65 ns
  for uncontracted (ss|ss) down to 3.4 ns for (pd|dd)-type blocks.
- **The cache keeps the quartets with the largest value:** estimated evaluation time per integral, times u, the
  share of the cycles expected to need the quartet. u is 1 from q_ij q_kl = 1e-7 up and falls linearly in
  log q to 0.05 at the cutoff, calibrated on the SVP alkane's per-call screening.
- **The time model** is a 5-parameter fit over ~8000 class timings (def2-SVP/TZVP, cc-pVDZ):
  80 + nprim (9 + .5 nroots g + .15 nroots ncart) + 1.3 ncart ns. Here nprim is the number of primitive
  combinations, g = (li+lj+1)(lk+ll+1) and ncart the number of Cartesian functions. Its ranking holds nearly
  what the measured times would.
- **Selection** takes a value histogram over all Schwarz-significant quartets (`CVHFsd_histogram`, 4096 bins):
  1.8 s for the TZVP alkane's 5e8 quartets, 0.1 s when everything fits.

| share of one cycle's integral time held by a 14 GB cache (qcost) | by q_ij q_kl | by modelled time/byte | by measured time/byte |
|---|---|---|---|
| caffeine/def2-TZVP (41 GB significant) | 0.30 | 0.60 | 0.62 |
| n-C20H42/def2-TZVP (98 GB) | 0.12 | 0.44 | 0.44 |

**Exactness.**
- **Unit test:** `test_vhf.py::test_semi_direct` covers full and partial caches; J+K, J-only and K-only; hermi
  0/1; one or several dms; bypass under another omega; SCF energies; switching off.
- **Kernel tests v6:** 12 new entries for J/K from a full and a 40% cache (5 non-symmetric, 2 symmetric and 1
  symmetric dms), all within 6.3e-14 of unmodified PySCF's direct J/K, the same as the direct path.
- **Tight validation against `pyscf-base`:**
  - At the default 4000 MB, the caches are partial for the alkanes and complete for caffeine. 10 cases: SCF
    energies ≤ 3.0e-12 Eh; the caffeine TDA, whose Davidson J/K read the cache, ≤ 2.1e-15 Eh.
  - At 20000 MB: the alkane SVP B3LYP and r2SCAN (complete caches) and TZVP PBE (14.7 GB partial) ≤ 4.6e-13 Eh;
    caffeine/SVP CAM-B3LYP (regular ERIs in memory, long-range K semi-direct) 7.3e-12 Eh; caffeine/TZVP B3LYP
    (partial) 2.7e-12 Eh.
  - Tight cycle counts differ by one in some cases, the known endpoint scatter of tight direct SCF (It7).
- **Range-separated hybrids:** CAM-B3LYP, wB97X and HSE06 (benzene, direct) agree with semi-direct off to
  ≤ 3e-12 Eh. Each uses two caches, full-range and long/short-range.
- **Harness fix:** the new caffeine/TZVP B3LYP case did not converge in 100 cycles in unmodified PySCF at tight
  settings (incremental J/K drift, It6), so it rebuilds J/K every cycle like the TZVP alkane.

**Benchmarks**
- Runs: `results/20261028-221349_it9-alkane.json`, 1 rep, same tree with `BENCH_SEMI_DIRECT=0` (off) and on,
  `PYSCF_MAX_MEMORY=20000`, boost clocks.
- The "on" runs averaged 0.1-0.5 GHz higher Zen 5 clocks, as faster code does. The J/K column is the part that
  changed.

| workload | off (s) | on (s) | speedup | J/K off -> on (s) | peak RSS off -> on |
|---|---|---|---|---|---|
| n-C20H42/SVP B3LYP SCF | 47.5 | 25.3 | **1.88x** | 32.5 -> 10.5 | 1.3 -> 10.9 GB |
| n-C20H42/SVP r2SCAN SCF | 57.1 | 36.5 | **1.57x** | 31.2 -> 7.8 | 1.5 -> 11.0 GB |
| n-C20H42/SVP PBE TDA, 5 states | 67.3 | 52.2 | **1.29x** | 27.9 -> 11.0 | 2.2 -> 11.7 GB |
| n-C20H42/TZVP PBE SCF | 259.1 | 192.3 | **1.35x** | 224.2 -> 154.5 | 2.4 -> 16.4 GB |

Medium molecules (`results/20261028-223438_it9-medium.json` at 20000 MB, `…224403_it9-4000.json` at the
default 4000 MB; 2 reps ABBA, within 3%):

| workload | off (s) | on (s) | speedup | J/K off -> on (s) |
|---|---|---|---|---|
| caffeine/TZVP B3LYP SCF (14.7 of 41 GB cached) | 151.9 | 90.3 | **1.68x** | 135.6 -> 74.1 |
| caffeine/SVP CAM-B3LYP SCF (regular ERIs in memory, long-range K cached) | 24.3 | 13.2 | **1.84x** | 15.8 -> 4.8 |
| caffeine/SVP B3LYP SCF, 4000 MB (all 2.5 GB cached) | 23.2 | 13.5 | **1.72x** | 13.7 -> 3.8 |
| caffeine+/SVP UKS B3LYP SCF, 4000 MB | 41.8 | 26.7 | **1.57x** | 21.2 -> 6.1 |
| caffeine/SVP B3LYP TDA, 5 states, 4000 MB | 50.9 | 43.4 | **1.17x** | 20.9 -> 12.5 |

- **CAM-B3LYP** now costs about what B3LYP does with the ERIs in memory (11.8 s, It8). Its long-range K was
  always direct, even when the regular ERIs were in memory.
- **At the 4000 MB default**, caffeine/SVP SCF gets most of the 20 GB in-memory speed (13.5 vs 11.8 s).
- **The TDA gains least:** with 20 trial dms per J/K call, the vector kernels' contraction is most of a call
  even when it reads stored integrals.

- **Per call** (SVP B3LYP): the first call 3.98 s vs 3.73 s direct (storing 8 GB), later calls 0.47-1.0 s vs
  2.1-4.0 s.
- **TDA:** reads the cache its SCF filled, in every Davidson J build.
- **TZVP:** the first call also builds the index, 2.6 s above direct in total; later calls save a third.

**What limits the calls that read the cache.**
- A call on the SVP alkane takes 0.57 s at 10 threads (3.97 s at 1): J alone 0.25 s, K alone 0.41 s. That is
  ~54 ns per quartet at 1 thread for blocks of 14 integrals on average.
- It is not bandwidth: the stored integrals stream at 2 GB/s per thread.
- Two A/B tests changed nothing beyond the ~10% clock drift between runs minutes apart, and were reverted:
  - prescreen and kernels inlined into a per-row loop instead of three indirect calls per quartet;
  - J and symmetric-dm K accumulated transposed, which makes the innermost loops contiguous.
- So the cost is the kernels' per-quartet overhead on the mostly s/p blocks: loop setup, and loop exits whose
  trip counts change from quartet to quartet. Kernels specialized by block shape would be the next step.

**Memory.** A direct SCF now uses up to 3/4 of the memory left under `max_memory`, as in-memory ERIs already
could (`scripts/pyscf-env.sh` says so). Run several jobs at once with lower `PYSCF_MAX_MEMORY` each, or set
`mf.semi_direct = False`.

**Clock note.** `scaling_max_freq` reads 2.0 GHz on every core, but the cores boost to 4.7-4.9 GHz
(amd-pstate-epp, active; `/proc/cpuinfo`). The value no longer shows whether the firmware clamp is on.

## AO-value cache test: radial factors vs AO values, 2026-10-29
Question: how much memory does caching only each shell's radial part save, compared with caching the AO values,
and what does each cost per pass over the grid (an SCF cycle or response iteration)?

Run: `bench/diag/ao_radial_cache.py` with `ao_radial.c`, 10 threads, default level-3 grids.
- **Radial cache:** for each 56-point block and shell kept by the screen index, store the contracted radial factor
  e(r) = Σ_p c_p exp(-a_p r²). Deriv 1 also stores the -2a-weighted sum.
  - The AO values are rebuilt from it with PySCF's own polynomial and c2s code.
  - The rebuilt values equal `eval_ao` bit for bit, as do the copied-back full-cache values; every block was checked.
- **Full cache:** the finished AO values of the same blocks and shells.
- **Each pass was timed two ways:**
  - in `block_loop` blocks: dense 67200-point blocks in RAM. "zeros" is `eval_ao` with every shell screened, i.e.
    only writing the block.
  - fused: each 56-point block goes into a per-thread buffer that stays in cache, then is discarded.

Median of 5 interleaved passes; times in s per pass.

| case | deriv | AO values, screened (GB) | radial (GB) | ratio | block_loop: zeros / eval / radial / full | fused: eval / radial / full |
|---|---|---|---|---|---|---|
| caffeine SVP | 0 | 0.35 | 0.17 | 2.1× | .023 / .032 / .026 / .028 | .021 / .004 / .005 |
| caffeine SVP | 1 | 1.41 | 0.34 | 4.2× | .098 / .100 / .106 / .121 | .029 / .012 / .019 |
| caffeine TZVP | 0 | 0.68 | 0.27 | 2.6× | .047 / .055 / .052 / .058 | .029 / .007 / .010 |
| caffeine TZVP | 1 | 2.73 | 0.53 | 5.1× | .195 / .202 / .213 / .240 | .048 / .026 / .037 |
| C20H42 SVP | 0 | 0.85 | 0.47 | 1.8× | .106 / .138 / .116 / .120 | .068 / .018 / .017 |
| C20H42 SVP | 1 | 3.42 | 0.94 | 3.6× | .451 / .462 / .492 / .520 | .112 / .061 / .064 |
| C20H42 TZVP | 0 | 1.50 | 0.70 | 2.1× | .200 / .228 / .214 / .221 | .102 / .033 / .031 |
| C20H42 TZVP | 1 | 6.01 | 1.40 | 4.3× | .825 / .831 / .872 / .931 | .176 / .113 / .116 |

- **Memory:** the radial cache is 1.8-2.6× smaller for LDA and 3.6-5.1× for GGA/meta-GGA.
  - Per shell the ratio is 1/3/5/7 (deriv 0) and 2/6/10/14 (deriv 1) for s/p/d/f.
  - The H-rich alkane has the lowest ratios.
  - Building it once takes 0.04-0.17 s.
- **In `block_loop` blocks, `eval_ao` costs what writing its dense block costs.**
  - For GGA, writing zeros alone comes within 1-3% of a full evaluation, at ~23.5 GB/s.
  - Neither cache saves time. Both are a few % slower, because they read the cache on top of the same writes.
  - On the alkane, 2/3 of the bytes written are zeros for screened shells (34% of block-shell pairs are kept).
- **Fused, AO evaluation is 3.4-4.7× cheaper for GGA, with no cache.**
  - The radial cache then cuts it a further 1.6-2.4× (GGA) or 3-6× (LDA).
  - The full cache is no faster than the radial one: it is bound by RAM reads (52-74 GB/s) of 2-5× the bytes.
- **Conclusion:** a cache pays only once AO evaluation and its consumers work on cache-sized blocks, and then the
  radial form is the one to keep. The first lever is the memory traffic of the dense blocks, which the rho and
  Vxc kernels also pay when they read them back.
- **Caveat:** the fused numbers cover AO evaluation only. Small blocks change the consumers' GEMM shapes and the
  Vxc accumulation, which this test does not measure.

## It10: AO evaluation on the screened path, 2026-10-29

`pyscf` `perf` commit `0b787b90e` (A). B is a prototype in `bench/diag/fused_xc.{py,c}`, not in the tree.

**A. `eval_ao` leaves screened AO values unwritten** (`0b787b90e`)
- **Why:** the AO-cache test showed that `eval_ao` is bound by writing its AO block.
  - On the screened path, 46% (caffeine/SVP) to 66–79% (n-C20H42/SVP) of that block are zeros for screened
    shells.
  - The sparse kernels never read them.
- **Change:**
  - `GTOeval_skip_screened` in `libcgto` turns off the zero fill for one call, using a per-thread flag.
  - `eval_gto`/`eval_ao(zero_screened=False)`.
  - `block_loop(zero_screened=False)` applies it to blocks yielded with a mask whose length is a multiple of
    `ALIGNMENT_UNIT`.
  - `nr_rks`, `nr_uks`, `nr_rks_fxc`/`_rks_fxc_batch` and `nr_uks_fxc` opt in through `_sparse_reader_kw`.
    They do so only when NumInt's own `block_loop`/`eval_ao`/`eval_rho1`/`eval_rho2`/`_gen_rho_evaluator` are
    used and the dms and orbitals are real.
- **Exactness: bit for bit.**
  - With NaN written into every value left unwritten, every opted-in path equals the zero-filled run bit for bit
    on n-C20H42 and caffeine/SVP (`bench/diag/screened_unwritten_check.py`; 9.3e8 NaNs per GGA call on the
    alkane). Covered: LDA/PBE/B3LYP/r2SCAN, RKS/UKS, MO-tagged and plain dms, f_xc.
  - Kernel tests pass.
  - Tight validation, FULL + SPARSE (21 cases) vs `pyscf-base`:

    | quantity | largest difference |
    |---|---|
    | energy | 1.2e-11 Eh |
    | gradient | 3.8e-12 |
    | Hessian | 4.5e-12 |
    | excitation energy | 4.6e-14 Eh |

  - Upstream dft gto tdscf grad hessian tests: 655 passed, 4 failed (the xcfun tests; xcfun is not built).
- **Timings** (10 threads, zeros vs unwritten, interleaved):

  | | zeros | unwritten | |
  |---|---|---|---|
  | `nr_rks`, n-C20H42/SVP, PBE | 1.207 s | 0.871 s | 1.39× |
  | same, B3LYP | 1.186 s | 0.843 s | 1.41× |
  | same, r2SCAN | 1.950 s | 1.616 s | 1.21× |
  | `nr_rks`, caffeine/SVP, PBE/B3LYP/r2SCAN | | | 1.07–1.10× |
  | SCF, n-C20H42/SVP, B3LYP (`PYSCF_MAX_MEMORY=20000`) | 24.7 s | 21.6 s | 1.14× |
  | same, r2SCAN | 31.9 s | 28.8 s | 1.11× |
  | SCF, caffeine/SVP, B3LYP and r2SCAN | | | 1.03× |

- **Harness pitfalls:**
  - `bench/diag/run_in.py` replaces `PYSCF_MAX_MEMORY` with 4000 unless the config sets it
    (`G=pyscf:PYSCF_MAX_MEMORY=20000`). With 4000 the semi-direct cache is smaller, and the same SCF A/B gave
    1.03–1.08×.
  - `lib.temporary_env(numint.NumInt, eval_ao=...)` restores a staticmethod as the bare function, because
    `getattr` unwraps it. That broke `NumInt.eval_ao` for every later test in the pytest worker (17 failures).
    The test now restores `NumInt.__dict__['eval_ao']`.

**B. Prototype: AO values computed inside the It6 kernels** (`bench/diag/fused_xc.{py,c}`)
- **What:** `FUSEDeval_rho`, `FUSEDdot_ao_aow` and `FUSEDdot_ao_wv_ao` are `VXCeval_rho_sparse_gemm`,
  `VXCdot_ao_aow_sparse_gemm` and `VXCdot_ao_wv_ao_sparse_gemm` with one change: the gather from `eval_ao`'s
  block is replaced by evaluating each 224-point group's AOs straight into the gather buffers, with PySCF's GTO
  code.
  - Optionally the AOs come from a radial cache: contracted radial factors per 56-point block and unscreened
    shell, built once.
- **Exactness:** `nr_rks_fused` follows `nr_rks` with a plain dm step by step. It equals `nr_rks` bit for bit on
  caffeine SVP/TZVP and n-C20H42 SVP/TZVP, for PBE and r2SCAN, with and without the cache.
- **Timings:** seconds per call, 10 threads, median of 5 interleaved. "MO dm" is `nr_rks` with the MO-tagged dm
  that SCF passes.

  | system | xc | `nr_rks` | `nr_rks` MO dm | fused | fused + radial cache | cache |
  |---|---|---|---|---|---|---|
  | caffeine SVP | PBE | 0.341 | 0.402 | 0.272 | 0.230 | 0.34 GB |
  | caffeine SVP | r2SCAN | 0.674 | 0.551 | 0.671 | 0.572 | |
  | caffeine TZVP | PBE | 0.842 | 0.817 | 0.715 | 0.673 | 0.53 GB |
  | caffeine TZVP | r2SCAN | 1.886 | 1.273 | 1.777 | 1.697 | |
  | C20H42 SVP | PBE | 0.919 | 0.892 | 0.808 | 0.625 | 0.94 GB |
  | C20H42 SVP | r2SCAN | 1.679 | 1.624 | 1.592 | 1.409 | |
  | C20H42 TZVP | PBE | 1.834 | 1.967 | 1.582 | 1.447 | 1.40 GB |
  | C20H42 TZVP | r2SCAN | 3.978 | 3.031 | 3.805 | 3.617 | |

- **GGA:**
  - 1.14–1.25× without the cache, because the AOs are evaluated twice per call (for ρ and for Vxc).
  - 1.25–1.48× with the cache, and 1.21–1.75× against the SCF (MO dm) path. The cache builds in 0.05–0.14 s.
- **Meta-GGA:** 1.0–1.19×. The prototype is slower than the SCF path on caffeine and the TZVP alkane: it
  computes τ from the dm (three more ng·ns² GEMMs per group), while `eval_rho2` multiplies the AO gradients by
  the occupied orbitals.
- **Where the time is:** after A, the AO block traffic left is smaller than the AO-cache test suggested. Most of
  each call is the GEMMs, the gathers of dm blocks and the scatter-adds.
- **Not integrated.** Integration would need:
  - the kernels in `libdft`, with the group code shared;
  - dispatch in `nr_rks`/`nr_uks`/f_xc;
  - several dms batched per AO evaluation (UKS α/β, TDDFT trial densities);
  - an MO-path ρ kernel for meta-GGA;
  - a lifetime and memory budget for the radial cache.

## It11: sparse XC kernels that compute the AO values in place, 2026-10-29

`pyscf` `perf` commit `474f6366c`. This is It10 B built into NumInt.

**What**
- **C, `lib/dft/nr_numint_fused.c`:**
  - `VXCfused_eval_rho` is `_eval_rho_sparse` for a batch of dms.
  - `VXCfused_eval_rho_mo` is `eval_rho2`'s orbital path (`VXCdot_ao_mo_sparse_gemm` + `_contract_rho`) for a
    batch of orbital sets.
  - `VXCfused_dot_wv` is `_dot_ao_ao_sparse` (LDA, with hermi), `_dot_ao_aow_sparse` and `_tau_dot_sparse` for a
    batch of potentials.
  - `VXCfused_radial_fill` fills the radial cache.
  - Each kernel evaluates a 224-point group's AO values directly into the gather buffers, once for the whole
    batch, using the GTO functions `eval_ao` calls. The values then go through the same dgemm/dsyrk calls and
    sums as before.
  - The It6 group code moved to `nr_numint_gemm.h`.
- **Python, `numint.py`:**
  - `_fused_xc` returns a `_FusedXC` for `nr_rks`, `nr_uks`, `nr_rks_fxc`, `nr_uks_fxc`, `cache_xc_kernel` and
    `cache_xc_kernel1` when all of these hold:
    - `FUSED_XC` (config `dft_numint_fused_xc`);
    - the conditions of `_sparse_reader_kw`;
    - spherical GTOs;
    - every `block_loop` block takes the sparse path.
  - Otherwise those functions run as before.
  - `_FusedXC` runs their loops over the same blocks, with the same LibXC calls and sums.
  - Densities take the path `_gen_rho_evaluator` would take. Its setup moved into `_rho_setup`/`_rho1_preferred`,
    which both share.
  - UKS evaluates α and β in one kernel call.
  - The XC matrices of many sets are split over calls, to bound the threads' private matrices.
- **The radial cache** lives in `grids._fused_ao_data`, checked against mol and the grids by content on every
  use.
  - It is built when it takes less than `RADIAL_CACHE_FRACTION` (default 1/4) of the memory left.
  - Sizes: 0.34 GB caffeine/SVP, 0.53 GB caffeine/TZVP, 0.94 GB alkane/SVP, 1.40 GB alkane/TZVP.
  - SCF builds it and TDDFT reuses it.

**Exactness: bit for bit** (`bench/diag/fused_numint_check.py`).
- On caffeine and n-C20H42/SVP, 36 results equal the `block_loop` path bit for bit:
  - `nr_rks` and `nr_uks` with MO-tagged dms, plain dms, and 3 non-symmetric dms at hermi=0;
  - `nr_rks_fxc` at hermi 0 and 1, and `nr_uks_fxc`;
  - `cache_xc_kernel` and `cache_xc_kernel1`, RKS and UKS;
  - each for LDA, PBE and r2SCAN.
- They also match with the radial cache, without it, and with one set per `VXCfused_dot_wv` call. Every call took
  the fused path (42/42).
- Both density paths for MO-tagged dms (each forced) are bit-identical as well.
- Gate:
  - kernel tests pass;
  - tight validation FULL + SPARSE (21 cases) gives the same largest differences as It10 (energies ≤ 1.2e-11 Eh,
    gradients ≤ 3.8e-12, Hessians ≤ 4.3e-12, excitations ≤ 4.6e-14 Eh);
  - upstream dft gto tdscf grad hessian: 656 passed, 4 failed (the xcfun tests);
  - `dft/test/test_numint.py::test_fused_xc` added.

**Timings.** Per call, 10 threads, block_loop path → fused with the radial cache:

| system | `nr_rks` PBE / r2SCAN | `nr_uks` PBE / r2SCAN | `nr_rks_fxc` (5 dms) PBE / r2SCAN |
|---|---|---|---|
| caffeine/SVP | 1.67× / 1.46× | 1.58× / 1.49× | 1.46× / 1.27× |
| caffeine/TZVP | 1.45× / 1.27× | 1.37× / 1.25× | 1.33× / 1.23× |
| n-C20H42/SVP | 1.58× / 1.56× | 1.62× / 1.53× | 1.47× / 1.23× |
| n-C20H42/TZVP | 1.47× / 1.35× | 1.45× / 1.28× | 1.28× / 1.16× |

`nr_uks` was measured before α and β shared one ρ kernel call.

End to end (`results/20261029-091520_it11-medium.json`: medians of 2 ABBA reps; `20261029-092429_it11-alkane.json`:
1 rep; `PYSCF_MAX_MEMORY=20000`; configs `U` = `BENCH_FUSED_XC=0` and `F` = `BENCH_FUSED_XC=1`):

| case | U (s) | F (s) | |
|---|---|---|---|
| caffeine/SVP PBE SCF | 11.05 | 9.04 | 1.22× |
| caffeine/SVP B3LYP SCF | 11.44 | 9.29 | 1.23× |
| caffeine/SVP r2SCAN SCF | 13.44 | 11.45 | 1.17× |
| caffeine⁺/SVP UKS B3LYP SCF | 26.15 | 21.71 | 1.20× |
| caffeine/SVP B3LYP TDA, 5 states | 33.64 | 25.24 | 1.33× |
| benzene/SVP B3LYP TDDFT, 5 states | 12.43, 14.94 | 8.48, 8.67 | 1.45–1.74× |
| benzene/SVP B3LYP Hessian | 24.07 | 21.46 | 1.12× |
| n-C20H42/SVP B3LYP SCF | 21.84 | 19.18 | 1.14× |
| n-C20H42/SVP r2SCAN SCF | 28.67 | 23.15 | 1.24× |
| n-C20H42/SVP PBE TDA | 47.96 | 41.93 | 1.14× |
| n-C20H42/TZVP PBE SCF | 166.7 | 162.7 | 1.02× (J/K 139 s) |

- In these runs the worker filed the fused kernels under "other". `worker.py` now counts them as `xc_kernels`,
  including their AO evaluation.
- On caffeine TDA, the XC part fell from 26.4 s (`xc_kernels` + `ao_eval`) to 18.8 s.
- Still on the old path:
  - gradients and the Hessian's own XC terms (`deriv=2` AOs);
  - NLC (VV10);
  - dense molecules (water, small benzene blocks);
  - cartesian bases.

## It12: XC terms of gradients and Hessians, 2026-10-29

`pyscf` `perf` commit `1ccef1836`. Worktree `pyscf-it11` (`474f6366c`) is the "before" build.

**Why** (`bench/diag/deriv_xc_time.py`, profiles; caffeine/def2-SVP B3LYP, per call):

| term | time | where it went |
|---|---|---|
| grad `get_vxc` | 1.77 s | dense `_scale_ao` passes 0.48, `eval_gto` (deriv 2) 0.42, gather GEMMs 0.39, `_make_dR_dao_w` 0.29 |
| hess `_get_vxc_diag` | 3.0 s | GEMMs 0.94, `eval_gto` (deriv 3) 0.86, `_scale_ao` 0.72 |
| hess `_get_vxc_deriv2_dm` | 6.3 s | numpy `einsum` per atom 3.5 |
| hess `_get_vxc_deriv1` | 12.3 s (r2SCAN 35.3 s) | per-atom gather GEMMs 5.1, per-atom dense `_scale_ao` over the whole AO block 4.9 |

- AO evaluation is only 3–7% of the two large Hessian terms, so fusing it alone would not have helped them.
- The weights of each atom's terms are not local: caffeine's atoms are active on 98% of the 224-point groups, and
  the alkane's on 55–60%. Skipping inactive groups would save little.

**What**
- **`VXCfused_dot_terms`** (`nr_numint_fused.c`): sums of products of weighted AO values up to third
  derivatives, computed in place per group. `_eval_group` now covers deriv 2 and 3 through `GTOprim_exp` and
  `GTOshell_eval_grid_cart_deriv2/3`; no radial cache for those.
  - The terms are recipes (`numint._xc_grad_terms`, `_xc_hess_diag_terms`) stacked along the grid points: one
    GEMM per output and group.
  - Used by `grad.rks/uks.get_vxc` and `hessian.rks/uks._get_vxc_diag` through `_FusedXC`.
- **`VXCdot_sets_aow_ao_sparse`** (`nr_numint_sparse.c`) computes the per-atom matrices of
  `_get_vxc_deriv1` for all atoms per block.
  - Each group's AO values are gathered once into buffers the threads share, and the threads take the
    (atom, direction) sets, with a barrier per group.
  - There are no weighted AO arrays and no private matrices. Each matrix is summed in group order, independent
    of the thread count.
  - The meta-GGA τ part is one `dsyrk`.
  - UKS LDA now adds the vxc part to the atoms' rows as RKS does.
- **`_get_vxc_deriv2_dm`:** the per-atom `einsum('scg,btcg->bst')` is one GEMM over all atoms
  (`_atom_pair_dot`).

**Exactness** (`bench/diag/deriv_xc_check.py`, references saved from `474f6366c`):
- Every term agrees to ≤ 2.9e-14 relative, for RKS caffeine and UKS benzene⁺ with SVWN, B3LYP and r2SCAN.
- Gate:
  - kernel tests pass;
  - tight validation FULL + SPARSE (21 cases) passes, with Hessians within 4.3e-12;
  - the `hessian` suite (9 water Hessians: LDA/r2SCAN/CAM-B3LYP RKS, LDA/PBE/r2SCAN/wB97X UKS, DF RKS/UKS)
    agrees with unmodified PySCF to ≤ 6.7e-11;
  - upstream tests: 658 passed, 4 failed (the xcfun tests);
  - new tests `test_dot_sets_aow_ao` and `test_fused_xc_deriv`.

**Per call** (caffeine/SVP; benzene⁺/SVP UKS):

| term | SVWN | B3LYP | r2SCAN | benzene⁺ UKS (SVWN / B3LYP / r2SCAN) |
|---|---|---|---|---|
| grad `get_vxc` | 0.43 → 0.26 s | 1.77 → 0.58 s | 2.36 → 1.05 s | 1.9× / 2.9× / 2.1× |
| hess `_get_vxc_diag` | 0.87 → 0.42 s | 3.01 → 0.92 s | 4.37 → 1.82 s | 1.8× / 2.7× / 2.0× |
| hess `_get_vxc_deriv2_dm` | 1.77 → 1.24 s | 6.33 → 4.08 s | 9.67 → 6.80 s | 1.2× / 1.3× / 1.2× |
| hess `_get_vxc_deriv1` | 7.94 → 3.71 s | 12.26 → 6.90 s | 35.3 → 16.3 s | 2.9× / 1.5× / 1.8× |

**End to end** (`results/20261029-111718_it12-deriv.json` medians of 2, and `20261029-112227_it12-hess-caffeine.json`
1 rep; P = `pyscf-it11`, N = `1ccef1836`; `PYSCF_MAX_MEMORY=20000`):

| case | P (s) | N (s) | |
|---|---|---|---|
| benzene/SVP B3LYP gradient | 1.61 | 1.27 | 1.27× |
| caffeine/SVP B3LYP gradient | 9.69 | 8.60 | 1.13× (J/K 7.6–8 s) |
| benzene/SVP B3LYP Hessian | 21.68 | 20.45 | 1.06× |
| benzene/SVP PBE Hessian | 19.06 | 18.07 | 1.05× |
| caffeine/SVP B3LYP Hessian | 211.6 | 198.3 | 1.07× (XC 69.2 → 56.8 s; J/K 117 s) |

- `hess_caffeine_svp_b3lyp` is a timing-only case in `bench/cases.py` (no suite).
- What is left of the Hessian's XC time is mostly the CPHF response: `nr_rks_fxc` for 3·natm dms per iteration.
  This is the low-rank (MO-basis) f_xc candidate in "Next iterations".

## It13: CPHF XC response from the factors of the response dms, 2026-10-29

`pyscf` `perf` commits A `53bd01c13` and B `d88076cb8`. Worktree `pyscf-it12` (`1ccef1836`) is the "before" build.

**Why.** After It12, most of the Hessian's XC time is the CPHF response: `nr_rks_fxc` of 3·natm dms per Krylov
iteration. Those dms are dm1 = C x Cocc^T + h.c., of rank 2·nocc, and the solver needs only C^T v1 Cocc.

### A: fused fxc blocks (`_fused_fxc_blksize`)
- **Problem:** `nr_rks_fxc` gave the fused kernels `_rks_fxc_batch`'s grid blocks, so that both paths agreed
  bit for bit. Those blocks shrink as 1/nset, because `_rks_fxc_batch` holds 2·nset AO arrays per block.
  - The fused kernels hold no AO arrays. But `VXCfused_dot_wv` allocates and sums nthreads·nset private
    [nao,nao] matrices per block: 3.6 GB per block for 186 dms of n-C20H42/SVP.
  - `bench/diag/fxc_blocks.py` (It12 tree, LDA, 186 dms): 64 blocks take 76.8 s, 11 blocks 45.7 s, so about
    0.59 s per block.
  - At 11 blocks, 18, 62 and 186 dms all cost about 243 ms per dm.
- **Fix:** block_loop's blocks, with fewer points only where rho1 and wv of all dms would exceed a quarter of
  `max_memory`. `nr_uks_fxc` takes the same rule; before, it used block_loop's blocks with no bound on rho1/wv.
- Where the blocks differ, the fused fxc results now agree with `_rks_fxc_batch` to rounding instead of bit for
  bit. The two still agree bit for bit where everything fits in one block, as in `test_fused_xc`.
- **Upstream** (`pyscf-base`, per-dm sparse kernels, `fxc_tree.py`) takes 292.9 s for the same 186-dm LDA call.
  It12 was already 3.8× faster; with A it is 6.6× faster.

### B: factored path (`_FusedXC.rks_fxc_mo`, `VXCfused_eval_rho_lowrank`, `VXCfused_dot_ao_mo`)
- **Densities from the factors:**
  - Notation: A = C x is per dm, B = Cocc is shared. On each 224-point group, TA[c] = ao[c]·A and TB[c] = ao[c]·B.
  - ρ = 2 Σ TA0·TB0, ∇ρ = 2 Σ (TAx·TB0 + TA0·TBx), τ = Σ_x Σ TAx·TBx.
  - GGA takes Σ TAx·TB0 = Σ_i ∂_x φ_i Z_i, with Z = A·TB0^T. That is two GEMMs of the size of TA per dm
    instead of four; a first version with Y = D φ0 took three.
- **Potential:** v1·Q (Q = Cocc) is added on the group's AO rows with one stacked GEMM per dm and group,
  ns × nq × K, where K = ng (LDA), 2ng (GGA) or 5ng (meta-GGA).
  - Each thread keeps private [nset,nao,nq] matrices, summed by a deterministic reduction.
  - C^T (v1 Q) is applied once at the end.
- **Flops per grid point and dm**, with r = nq = nocc:

  | | dm path | factored, per dm |
  |---|---|---|
  | LDA | 3·ns² | (2r + 2nq)·ns |
  | GGA | 4·ns² | (4r + 4nq)·ns |
  | meta-GGA | 13·ns² | (8r + 10nq)·ns |

  Once per pass, the factored path also pays 2·ns·(r+nq) per AO component for TB and ao·Q.
- **Path choice:** `_gen_rhf_response` (singlet None, hermi 1, MOs given) attaches `vind.vind_mo(dm1, x)`,
  which returns C^T vind(dm1) Cocc. `hessian.rhf.gen_vind` uses it.
  - Each call takes the factored XC path where `lowrank_cheaper(xctype, nocc, nocc, nset)` puts its flops
    below 0.8 of the dm path's; otherwise it calls vind(dm1).
  - The choice is per call because the Krylov batches shrink as vectors converge, and the per-pass work does
    not amortize over a few dms.
- **Memory:** the dms go in passes whose private v1·Q copies fit in half of `max_memory`. Blocks are sized so
  that rho1 and wv fit in a quarter.
- **Applicability:** the factored path is offered for any molecule (`_fused_xc(match_block_loop=False)`). It
  does not reproduce a block_loop path, so it does not need the sparse switch.

**Exactness**
- `lowrank_fxc_check.py`: the XC response agrees with C^T `nr_rks_fxc`(dm1) Cocc to ≤ 8.1e-15 relative
  (12 cases).
- `vind_mo` against the projected `vind`:
  - ≤ 1.2e-14 where the ERIs are in memory (benzene, caffeine/SVP);
  - 3e-14–1.1e-12 with semi-direct J (caffeine/TZVP, the alkane).
  - There, two identical J builds already differ by 7–8e-14 in the AO basis, and by 1.3e-12 in the MO basis,
    where the coefficients (up to 49.5) amplify it (`j_repro.py`).
- **Unit tests:**
  - `test_fxc_lowrank`: SVWN, PBE and r2SCAN, in one pass, and in one-dm passes of 224-point blocks;
  - `test_cphf_xc_lowrank`: both paths of `vind_mo` against `FUSED_XC=False`, to 1e-13.
- **Gate:**
  - kernel tests pass;
  - tight validation FULL and SPARSE pass (the alkane SCF energies within 2.7e-12 Eh);
  - the `hessian` suite passes: water Hessians within 1.0–2.1e-11, and 6.7e-11 with DF;
  - the benzene Hessians agree within 4.3e-12;
  - upstream tests (`scf df grad hessian tdscf dft`): 850 passed, 4 failed, the same xcfun tests as on
    unmodified master.
  - The gate ran on A and B together. A alone (a temporary worktree with the It12 libraries, whose C code is
    identical) passes `test_numint`, `test_tdrks` and the RKS Hessian tests, except the known xcfun failure.

**Per call** (`lowrank_fxc_check.py`, 3·natm dms; medians of 3 interleaved runs; 10 threads;
`max_memory` 16000). Each cell gives It12 dm path → dm path with A → factored (B), and It12 / B.

| system (nocc, dms) | SVWN | B3LYP | r2SCAN |
|---|---|---|---|
| benzene/SVP (21, 36) | 0.56 → 0.63 → 0.16 s (3.5×) | 0.83 → 0.88 → 0.49 s (1.7×) | 2.47 → 2.56 → 0.88 s (2.8×) |
| caffeine/SVP (51, 72) | 6.77 → 6.22 → 2.21 s (3.1×) | 8.77 → 8.36 → 5.93 s (1.5×) | 26.1 → 25.3 → 11.6 s (2.3×) |
| caffeine/TZVP (51, 72) | 25.6 → 22.4 → 4.18 s (6.1×) | 34.0 → 29.9 → 11.6 s (2.9×) | 91.4 → 89.9 → 22.2 s (4.1×) |
| n-C20H42/SVP (81, 186) | 76.1 → 44.4 → 21.8 s (3.5×) | 92.4 → 59.4 s, dm path kept (1.6×; factored 53.4 s) | 170.3 → 172.3 → 105.6 s (1.6×) |

- **Model check:** the factored/dm flop ratio was 0.22–0.68 in the 11 cases where the model chose the factored
  path. There, the measured time ratio was 0.19–0.71.
  - Relative to the dm path, a factored flop costs about 0.85× (LDA), 1.1–1.24× (GGA) and 1.04–1.15×
    (meta-GGA) on the dense molecules, and 0.74–0.91× on the alkane.
  - At the 0.8 threshold, the worst case is therefore break-even (GGA, few occupied orbitals). The one miss is
    the alkane with B3LYP (flop ratio 0.99): it keeps the dm path, which is 1.11× slower than factored would be.
- Benzene's dm path has the same blocks before and after A. The 4–12% differences there are run-to-run noise:
  the two runs were hours apart, and the calls take under 3 s.

**End to end** (Hessians; P = `pyscf-it12`, N = It13; `PYSCF_MAX_MEMORY=20000`). Benzene is medians of 2
(`results/20261029-172327_it13-hess.json`), caffeine 1 rep (`20261029-172759_it13-hess-caffeine.json`).
J/K is the harness's `jk` category. CPHF XC is the tottime of the kernel wrappers in the runs' profiles:
`dot` + `_rho_kernel` before, `rho_lowrank` + `dot_ao_mo` after.

| case | P (s) | N (s) | | CPHF XC (s) | J/K (s) |
|---|---|---|---|---|---|
| benzene/SVP B3LYP | 20.11 | 18.32 | 1.10× | 5.0 → 2.9 | 11.6 → 12.0 |
| benzene/SVP PBE | 17.71 | 15.84 | 1.12× | 4.5 → 2.6 | 9.9 → 9.9 |
| benzene/SVP r2SCAN | 30.20 | 20.24 | 1.49× | 15.0 → 4.9 | 9.8 → 10.0 |
| caffeine/SVP B3LYP | 198.0 | 178.9 | 1.11× | 53.2 → 35.9 | 116.2 → 114.2 |
| caffeine/SVP r2SCAN | 280.0 | 199.4 | 1.40× | 147.7 → 66.3 | 91.6 → 92.0 |

- The before columns include the SCF's XC kernels (about 1–3 s); after, those fall below the profile's top 12.
- J/K is now about two thirds of the B3LYP Hessians and nearly half of the r2SCAN ones.
- The harness's `worker.py` now counts the It12 and It13 kernel wrappers (`dot_terms`, `_dot_sets_aow_ao`,
  `rho_lowrank`, `dot_ao_mo`) under `xc_kernels`; these two result files predate that and show them under
  `other`.
- New timing-only cases: `hess_benzene_svp_r2scan`, `hess_caffeine_svp_r2scan`.

## Swarm review after It13, 2026-10-29

**What ran.** A workflow of 17 agents on the It13 tree (`perf` `d88076cb8`). It was read-only apart from short
1-thread experiments.
- Eight area analysts: J/K, XC, TD, gradients and Hessians, UKS and DF, Python overhead, SCF driver, system and
  build.
- One skeptical verifier per area, which reran the analyst's key measurements and tried to refute each finding.
- A synthesis that merged duplicates across areas and ranked what survived.

56 findings: 48 confirmed or adjusted, 8 refuted.
- Full report: `results/profile-it13/swarm/report.md`. §2 ranks 25 items; §3 gives designs for the top 8.
- Analyst and verifier data per finding: `swarm/findings.json`.
- The agents' prototype scripts are in `swarm/prototypes/<area>/`. They are unreviewed and keep the paths they
  ran with.

**Profile used.**
- `results/20261029-180928_profile-it13.json`: 10 threads, full suite, `BENCH_TOP=30`.
- Its summary: `results/profile-it13/summary.txt`.
- 1-thread native profiles: `profile-it13/native_*.txt`.

The profile runs overlapped. This was my error.
- The five caffeine/benzene native profiles ran at the same time.
- The 10-thread suite started at 18:09:28, while the caffeine SCF native profile ran until 18:09:59.
- As a result, the XC kernels of `scf_caffeine_svp_{pbe,b3lyp}` read about 10–15% high.
- The native profiles' absolute times are about 2× inflated. Their percentages are usable, but 1→10-thread
  ratios taken from them are not.
- Run profiles one at a time.

**Ranked opportunities.** Savings are the verifiers' 10-thread estimates. None has been measured end to end.
"caf" is caffeine/SVP and "bz" is benzene/SVP.

| # | opportunity | expected saving | exactness | effort |
|---|---|---|---|---|
| 1 | Hessian 2e J/K: `int2e_ip1ip2`/`int2e_ipvip1` only for the rows the pair loop reads (atoms ≤ A) | caf B3LYP Hessian −28 to −34 s of 181 | bitwise | S |
| 2 | Hessian diagonal 2e blocks from translational invariance: no `int2e_ipip1`, no A=A blocks | caf a further −25 to −27 s; bz −3 to −4 s | ≤ 7e-12 on diagonal blocks (qcint's TI error) | M |
| 3 | `hess_elec`: GEMMs instead of 3-operand `numpy.einsum` and dm1 rebuilt for every pair | caf −8.5 to −11 s | 6.5e-15 | S |
| 4 | `-O3 -march=x86-64-v3 -ffp-contract=off` for libdft and libcgto's grid files | XC 3–5% (SCF); caf Hessian −4 to −6 s | bitwise | S |
| 5 | Adaptive AO block size (64 → 32/16) in `CVHFnr_direct_drv` when a call has few tasks | bz Hessians −4.5 to −6 s (25–35%); bz gradient −0.6 to −0.8 s | rounding | S–M |
| 6 | TDA/RPA/Casida XC response from the trial-vector factors (`rks_fxc_mo`) | caf TDA −3.5 to −4.1 s; bz TDDFT ≈ −2.6 s | ≤ 2e-14 | M |
| 7 | numpy's and scipy's own OpenBLAS pools spin about 134 ms after each call: A/B `OPENBLAS_THREAD_TIMEOUT`, `OPENBLAS_NUM_THREADS`, OMP and malloc settings | SCFs −0.25 to −1.8 s (unmeasured) | bitwise (timeout) | config |
| 8 | Grid grouping: an int64 box key and a stable argsort instead of `numpy.unique(axis=0)` | caf SCF −0.16 s; alkane −0.4 s | bitwise | S |
| 9 | GGA density on the MO path via Z = C·v0 (2 GEMMs instead of 4) | caf PBE/B3LYP −0.6 to −0.75 s | rounding | M |
| 10 | UKS in-core J/K: both spin dms in one ERI pass | caf⁺ UKS −2.3 to −3.0 s | rounding | S–M |
| 11 | Gradient J/K: an s8 driver with 3 of 4 centre derivatives, contracted straight into dE/dR | alkane gradient −5.3 to −6.5 s (≈ 20%) | rounding | L |

Three more items are large but have no bench case yet:
- the VV10 pair kernel: ≈ −150 to −250 s per caffeine wB97M-V SCF;
- the UKS factored CPHF/TD response: ≈ −35 to −42 s per caffeine⁺ B3LYP Hessian;
- DF-K of response dms from their factors: DF hybrid TD 1.45–1.75×.

**Opt-in only (approximations):**
- Hessian derivative-integral screening. Upstream's `_make_vhfopt` is dead code, because `rhf.py:192` tests the
  intor names without their suffixes.
- Davidson restricted to the target roots.
- A looser CPHF.
- Coarser grids.
- Frozen core.
- float32 storage in the semi-direct cache.
- Becke pair screening.

**Ruled out**, among others:
- cartesian contraction of the derivative integrals;
- `-O3`/PGO for libcvhf (42% slower);
- libxc built with `-march` (bitwise neutral);
- SMT;
- BLIS/AOCL;
- `OPENBLAS_CORETYPE` changes;
- a different eigensolver (it changes cycle counts);
- MO-basis J for TD below ≈ 280 dms.

**Proposed order** (report §6). Steps 1 and 2 are done; see "Thread pools" and "It14" below.
1. The thread-pool A/B (#7), with no code. Adopt the winner in `scripts/pyscf-env.sh` and `bench/common.py`,
   then re-profile.
2. It14, Hessian derivative J/K and `hess_elec`: #1, #2, #5, #3. The summed estimate takes the caf B3LYP Hessian
   from 181 s to about 105–119 s.
3. It15, XC setup, grid build and the libxc interface: #8, #14, #18, #22.
4. It16, fused XC kernels: #4, then #9, #16 and the NPdgemm chunks.
5. It17, TD response: #6, #12, #17, then #20.
6. It18, UKS: #10, then bench cases and the UKS factored response.
7. It19, derivative J/K drivers: #11, #15.

## Thread pools: `OPENBLAS_THREAD_TIMEOUT=20` (swarm #7), 2026-10-29

**Problem.** numpy and scipy each bundle their own pthread OpenBLAS (numpy's `libscipy_openblas64_`, scipy's
`libscipy_openblas`), each with OMP_NUM_THREADS = 10 threads.
- After a threaded call, the idle workers spin for 2^28 TSC ticks (about 134 ms) before they sleep.
- PySCF's OpenMP regions that start in that window share the cores with 9, or after scipy and then numpy, 18
  spinning threads.
- `run_tests.sh` already set `OPENBLAS_THREAD_TIMEOUT=4`, copied from upstream CI. The benchmarks and
  `pyscf-env.sh` never did.

**Measurements: `bench/diag/blas_pools.py`** (caffeine/SVP sizes, 10 threads).
- **CPU burned while the main thread sleeps 0.5 s after a call:**
  - 1.2 s after a numpy GEMM, a numpy ddot of 128k elements, or a scipy eigh;
  - 2.4 s after an eigh followed by a GEMM;
  - 0.05 s after a PySCF `lib.dot`, which is libgomp's own short spin.
- **One in-core J+K build (180 ms of OpenMP), right after the call vs after a pause:**

  | right after | default | timeout 4 | `OPENBLAS_NUM_THREADS=1` |
  |---|---|---|---|
  | a numpy GEMM | 210 vs 188 ms | 180–184 ms | 179–183 ms |
  | a scipy eigh | 207 vs 184 ms | 180–184 ms | 179–183 ms |
  | eigh, then GEMM | 258 vs 185 ms | 180–184 ms | 179–183 ms |

- **Short OpenMP calls** (`get_overlap_cond`, `condense`) were not slowed with one pool spinning: the 20 hardware
  threads absorb 9 spinners. The swarm's 5 ms vs 0.13 ms came from the overlapping profile runs.
- **eigh(F,S) and GEMM back to back**, median ms (several runs; scipy's spinning pool also slows numpy's GEMM):

  | n | default eigh / GEMM | timeout 4 | timeout 16 | timeout 20 | `NUM_THREADS=1` |
  |---|---|---|---|---|---|
  | 246 | 4.6–5.2 / 0.15 | 4.5–4.6 | 3.9–4.3 | 4.0 | 4.7 / 0.49 |
  | 520 | 19–21 / 1.2–1.3 | 20–26 | 18 | 16 | 29 / 4.5 |
  | 872 | 46–50 / 6.2 | 60–63 / 3.6–5.4 | 47–55 / 3.5 | 47 / 3.4 | 110 / 21 |

**End to end** (`results/20261029-201727_pools-ab.json`): medians of 3 ABBA runs, 10 threads,
`PYSCF_MAX_MEMORY=20000`, `pyscf` at `d88076cb8`.

| case | default (s) | timeout 4 (s) | timeout 20 (s) | default / timeout 20 |
|---|---|---|---|---|
| scf_caffeine_svp_b3lyp | 9.38 | 5.32 | 5.45 | 1.72× |
| scf_caffeine+_svp_b3lyp_uks | 21.60 | 14.49 | 14.84 | 1.46× |
| scf_benzene+_svp_r2scan_uks | 4.18 | 2.13 | 2.18 | 1.92× |
| scf_c20h42_svp_b3lyp | 19.49 | 15.94 | 15.97 | 1.22× |
| tda_caffeine_svp_b3lyp | 25.48 | 25.14 | 24.97 | 1.02× |
| hess_benzene_svp_b3lyp | 17.56 | 17.42 | 17.35 | 1.01× |
| grad_benzene_svp_b3lyp | 1.41 | 1.33 | 1.35 | 1.04× |
| grid_caffeine_svp | 3.82 | 3.41 | 3.43 | 1.11× |

- **Where it shows:** the SCF's XC kernels, about 1.85× faster (caffeine B3LYP 4.8–5.0 → 2.6–2.7 s; the alkane
  6.4 → 4.1 s). Each SCF cycle calls eigh and numpy just before the XC kernels. TD and Hessians call numpy
  rarely between their long kernels.
- **Results:**
  - identical bit for bit wherever a case is deterministic (in-core SCFs, TDA, gradient, grid);
  - the semi-direct alkane SCF and the direct-J/K Hessian keep their usual run-to-run spread (3.6e-12 Eh,
    8.8e-14) under every setting, with the same cycle counts.
- **Decision:** `OPENBLAS_THREAD_TIMEOUT=20` (about 0.5 ms) goes in `scripts/pyscf-env.sh` and in
  `bench/common.py`'s pinned environment, for every tree including `pyscf-base` and the wheel.
  - Timeout 4 is as fast end to end, but eigh at n = 872 is 30% slower: every LAPACK-internal BLAS call wakes
    sleeping threads.
  - `OPENBLAS_NUM_THREADS=1` makes eigh 2.3× and GEMM 6× slower at n = 872.
- **Comparability:** benchmark times before this change include the penalty on both sides of every
  comparison, so they are not comparable with times after it.

## It14: Hessian two-electron terms and the direct J/K driver's blocks, 2026-10-29

`pyscf` `perf` commits:
- A `4034c2ae6`: J/K driver blocks;
- B `fb7275305`: rows that are read, and translational invariance;
- C `5c07d15d0`: `hess_elec`.

These are swarm items #5, #1+#2 and #3. Worktree `pyscf-it13` (`d88076cb8`) is the "before" build. All times
use the pinned `OPENBLAS_THREAD_TIMEOUT=20` (previous section).

**Why.** After It13, derivative J/K was 60% of the caffeine B3LYP Hessian: 107.8 of 181 s, in 73 calls (ipip1 ×1,
ip1ip2 ×24, ipvip1 ×24, make_h1 ×24). `hess_elec`'s three-operand `numpy.einsum`s took another 8.8–11.3 s.

### A: adaptive AO blocks in `CVHFnr_direct_drv` (#5)
- **Problem:** the driver's tasks are the (j,k,l) blocks of 64 AOs, and each task loops over the i blocks. Benzene
  (114 AOs) has 8 tasks per gradient call, and the Hessian's per-atom calls have similarly few. At 10 threads,
  derivative J/K therefore scaled only 2.6× (gradient) and 3.2× (Hessian).
- **Fix:** halve the blocks, down to 16 AOs, until there are at least 16 tasks per thread. The same applies to
  `CVHFnr_direct_ex_drv`.
- **Required alongside:** `keys_cache`, which tracks a thread's output blocks, was sized for 64-AO blocks. Smaller
  blocks would overflow it, so the new `CVHFallocate_JKArray_bs` sizes it from the actual block size.
  `CVHFallocate_JKArray` (64 AOs, used by `nr_sr_vhf.c`) is unchanged.
- **Effect on results:** rounding-level only. The order in which threads write back their blocks was already
  dynamic.

### B: only the rows the pair loop reads, and diagonal blocks from translational invariance (#1, #2)
- **Rows:** the pair loop of `_partial_hess_ejk` reads, for atom A, only the rows of the ip1ip2/ipvip1 contractions
  that belong to atoms B ≤ A. All other rows were computed and thrown away, so every atom pair was computed twice.
  - The ip1ip2 calls now take k shells, and the ipvip1 calls j shells, only up to the last of those atoms.
- **Translational invariance:** when atmlst holds every atom, Σ_B ∂²E/∂R_A∂R_B = 0 at fixed density gives
  ej[A,A] = −Σ_{B≠A} ej[A,B], and likewise ek.
  - Then int2e_ipip1 is not computed, the rows of A itself are dropped, and the first atom needs no call at all.
  - With a subset of atoms, ipip1 still gives the diagonal blocks.
- **One function for all callers:** `hessian.rhf._partial_hess_2e(mol, dm_j, dms_k, k_fac, atmlst, with_j,
  with_k, omega)` serves:
  - RHF `_partial_hess_ejk`;
  - UHF `_partial_hess_ejk`, with dms_k = [dm0a, dm0b] and k_fac = 2;
  - the long-range K of range-separated hybrids in `rks`/`uks.partial_hess_elec`, which was mixed into veff with
    the XC terms and is now `de2 -= (alpha-hyb) * ek_lr`.
  - XC has no grid response here, so it is not translation-invariant and keeps its own diagonal terms.

### C: `hess_elec` with GEMMs (#3)
- **Before:** `hess_elec` rebuilt dm1[B] = mo1[B] Cᵀ and its energy-weighted version for every pair (A,B), with
  `numpy.einsum` (three operands, no BLAS).
- **Now:** `_hess_elec_mo1` forms once per atom h1ao[A] C − s1ao[A] C e and Cᵀ s1ao[A] C, then two GEMMs against
  all stacked mo1[B] and mo_e1[B]. It is used by RHF and per spin by UHF.
- The lower-triangle-plus-mirror convention is kept.

**Exactness** (`bench/diag/hess_ejk_check.py`, 1 thread, against copies of the replaced code):
- Off-diagonal blocks are identical bit for bit, for RHF and UHF, J only, range-separated K, and all-atom,
  reversed and subset atom lists. The subset path is bit-for-bit identical everywhere.
- Diagonal blocks from translational invariance:

  | system | ej | ek | long-range K |
  |---|---|---|---|
  | water | 8.1e-13 (max \|ej\| 7.3) | 6.7e-13 (max \|ek\| 1.1) | ≤ 5.2e-14 (water and ethylene) |
  | ethylene | 1.1e-12 (max \|ej\| 13.2) | 6.4e-13 | |
  | ethylene⁺ UHF | 2.8e-12 | 2.2e-13 | |
  | benzene | 7.3e-12 (max \|ej\| 22.5) | 1.2e-12 | |

  - This is the integrals' own translational-invariance error, not summation order.
  - It stays below the Hessian spread between builds that It7 and It8 recorded (up to 1.5e-10).
- `hess_elec` terms: ≤ 4.3e-16.
- 1 thread, benzene:
  - two-electron blocks 31.45 → 10.92 s (2.9×); J only 27.52 → 9.40 s;
  - `hess_elec` terms 0.23 → 0.003 s.

**Gate**
- Kernel tests pass.
- Tight validation passes on the hessian, full and sparse suites:
  - water Hessians (all functional types, RKS/UKS, DF) within 8.6e-12–1.8e-11, DF 6.7e-11 (as in It13);
  - benzene Hessians 6.9–7.3e-12 (4.3e-12 in It13; the new part is the translational-invariance diagonal);
  - caffeine gradient 1.6e-12.
- Upstream tests: 850 passed, 4 failed (the known xcfun failures).

**Timings** (P = `pyscf-it13`, N = It14; 10 threads; `PYSCF_MAX_MEMORY=20000`; medians of 3 ABBA runs,
`results/20261029-211033_it14-hess-grad.json`; caffeine: means of 2,
`results/20261029-212113_it14-hess-caffeine.json`). J/K is the harness's `jk` category.

| case | P (s) | N (s) | | J/K P → N (s) |
|---|---|---|---|---|
| hess_benzene_svp_b3lyp | 17.98 | 8.28 | 2.17× | 11.93 → 2.63 |
| hess_benzene_svp_pbe | 16.12 | 7.61 | 2.12× | 10.26 → 2.14 |
| hess_benzene_svp_r2scan | 20.12 | 12.05 | 1.67× | 10.10 → 2.16 |
| grad_benzene_svp_b3lyp | 1.45 | 0.71 | 2.04× | 1.30 → 0.57 |
| grad_caffeine_svp_b3lyp | 8.11 | 7.67 | 1.06× | 7.48 → 7.00 |
| grad_c20h42_svp_b3lyp | 29.11 | 29.21 | 1.00× | 27.44 → 27.44 (64-AO blocks kept: 288 tasks) |
| hess_caffeine_svp_b3lyp | 180.8 | 105.0 | 1.72× | 115.8 → 52.0 |
| hess_caffeine_svp_r2scan | 201.0 | 135.0 | 1.49× | 93.0 → 38.6 |

- **`hess_elec`:** the harness's `numpy` category (mostly the einsums) drops from 15.2 to 3.7 s (caffeine
  B3LYP) and from 16.7 to 5.2 s (r2SCAN).
- **Against the swarm's estimate:** it predicted 181 → 105–119 s for the caffeine B3LYP Hessian; the result is
  105 s.
- **What is left in the Hessians:** the CPHF XC response is unchanged (caffeine: 44.6 s B3LYP, 86.0 s r2SCAN).
  - For B3LYP it now nearly equals J/K (52 s), and for r2SCAN it is the largest part.
  - On benzene, XC (3.9 s B3LYP, 7.5 s r2SCAN) exceeds J/K (2.1–2.6 s).
- **Unit test:** `test_partial_hess_2e_translational_invariance` (`hessian/test/test_rhf.py`) checks that the
  diagonal blocks from translational invariance equal those from int2e_ipip1 to 1e-9, for RHF, UHF-type spin
  densities and range-separated K.

## It15: grid grouping key and threaded xc_deriv transforms, 2026-10-29

Swarm items #8 and #22, in `pyscf` `perf` commits A `a737220e0` (grid key) and B `84f2271ca` (xc_deriv).
Worktree `pyscf-it13` is the "before" build: It14 changed nothing these cases run.

**Dropped from the plan.** Two items lost their target once the thread-pool fix was in. Evidence:
`results/20261029-215557_profile-it14.json`, run with `OPENBLAS_THREAD_TIMEOUT=20`.
- **#14, the duplicated XC setup:** `get_overlap_cond` and `condense` now cost 0.01–0.03 s per SCF, down from
  0.3–0.5 s.
- **#18, libxc in one parallel region:** libxc now costs 0.14–0.18 s per caffeine SCF (was 0.67–0.88 s) and
  scales about 5.6× on 10 threads.

The "fused-context" slowdowns the swarm saw were the spinning pools.

### A: grid points grouped by an integer box key (#8)
- **Before:** `arg_group_grids` sorted the (x,y,z) box ids as structured rows (`numpy.unique(axis=0)`, single
  thread): 0.16 s per caffeine grid and 0.41 s for the alkane. After the thread-pool fix that is 3% of a caffeine
  SCF.
- **Now:** a mixed-radix int64 key and one stable argsort. The ids are clipped to [-1, boxes], so the key increases
  with the rows' lexicographic order, and its stable argsort is the same permutation.
- **Check (`bench/diag/grid_group_check.py`):** identical permutations in all 10 tests, covering water to the
  alkane, caffeine at level 5, and points displaced by N(0, 3 bohr). Times:

  | grid | before | after |
  |---|---|---|
  | caffeine | 165 ms | 13 ms |
  | alkane | 406 ms | 31 ms |
  | caffeine, level 5 | 515 ms | 39 ms |

- **Test:** `test_arg_group_coords` now also covers displaced points. It passes with both versions.

### B: xc_deriv transforms on all threads (#22)
- **Before:** the kernels of `xc_deriv.c` ran on one thread, and so did `transform_xc`'s fancy-index gathers and
  additions in numpy.
- **Now:**
  - `VXCunfold_sigma_spin0/1`, `VXCud2ts`/`VXCts2ud` and `VXCfg_to_direct_deriv` run over 1024-point blocks with
    OpenMP.
  - The row gather in `_unfold_gga` uses the new `VXCgather_rows`.
  - `transform_xc` records each `xc_tensor.transpose(...)[diag_idx] += xc_sub` as pairs of row numbers. To get
    them it applies the same transpose and indexing to arrays of row numbers. `VXCadd_rows_indexed` then adds the
    pairs in the same order at every point.
  - Within one update the target rows are distinct; the check asserts this. So the sums equal numpy's.
- **Bit for bit (`bench/diag/xc_deriv_check.py`):** `transform_xc` matches the old Python code driving the old
  library for GGA and MGGA, spin 0 and 1, orders 1–4. So do `ud2ts`, `ts2ud` and the fg kernel.
- **Times at 1M points, transform only:** 1.3–1.8× faster.
  - Spin-1 order 3: GGA 1.39 → 0.77 s, MGGA 2.56 → 1.53 s.
  - The rest is memory traffic of the derivative tensors: 4 GB at 1M points for spin-1 GGA order 3.
- **`micro_libxc`** (`results/20261029-220536_it15-xcderiv.json`, medians of 3): 1.69 → 1.43 s, with `xc_post`
  0.76 → 0.47 s. Third-derivative calls are 1.3–1.4× faster, first-derivative calls unchanged.

**Gate**
- Kernel tests pass.
- Tight validation passes on the full and sparse suites, except one run of `scf_caffeine_svp_camb3lyp`:
  - It did not converge to conv_tol=1e-12 within 100 cycles (1.9e-9 Eh off).
  - Reruns (`results/camb3lyp-flaky.log`) failed 1 time in 3 on both the It15 tree and `pyscf-it13`.
  - The SCF's long-range K uses the batched J/K kernels, which neither It14 nor It15 touched.
  - So this is the incremental-J/K stall that NOTES records for the TZVP and UKS cases.
  - The case now rebuilds J/K in tight mode (`full_jk=True`), like those. Its reference was regenerated, and 3 of
    3 runs pass within 2.3e-13–1.8e-12 (`results/camb3lyp-fulljk.log`).
- Upstream tests: 851 passed, 4 failed (the known xcfun failures).

**Timings** (P = `pyscf-it13`, N = It15; 10 threads; medians of 5 ABBA runs;
`results/20261029-224239_it15-scf.json`)

| case | P (s) | N (s) | | grid sort P → N (s) |
|---|---|---|---|---|
| scf_caffeine_svp_b3lyp | 5.36 | 5.17 | −0.19 s (3.5%) | 0.168 → below the top 40 |
| scf_c20h42_svp_b3lyp | 15.64 | 15.21 | −0.43 s (2.7%) | 0.417 → below the top 40 |
| scf_benzene+_svp_r2scan_uks | 2.12 | 2.12 | within the ±7% spread | 0.080 → below the top 40 |

- On benzene⁺ the parts do show the gain: the grid sort −0.08 s and `_unfold_gga` 0.052 → 0.021 s.
- Results are identical where the case is deterministic.

## It16: the fused XC kernels: AVX2 flags, GEMM groups of 2 blocks, GGA densities from Z, 2026-10-29

`pyscf` `perf` commits:
- A `930f39c41`: build option `NUMINT_X86_64_V3`.
- C `a8a68d87f`: `GEMM_NBLK` 2.
- B `0835ee427`: GGA densities from Z, and the density path by flop count.

These are swarm items #4, #16 and #9. Worktrees: `pyscf-it15` (before, stock flags), `pyscf-it16a` (A) and
`pyscf-it16c` (A+C), the last two built with `-DNUMINT_X86_64_V3=ON`. The `pyscf` tree is now configured with
that option; it stays in the CMake cache, and a fresh build must pass it (How to).

### A: AVX2 for the numint and GTO grid kernels (#4)
- **Problem:** the stock `-msse3 -O2` leaves the elementwise loops of the fused kernels scalar (`_contract_mo`,
  `_contract_group`, the weighted gathers in `VXCfused_dot_wv`).
- **Fix:** `-DNUMINT_X86_64_V3=ON` compiles `dft/nr_numint.c`, `nr_numint_sparse.c`, `nr_numint_fused.c` and
  `gto/grid_ao_drv.c`, `deriv1.c`, `deriv2.c` with `-O3 -march=x86-64-v3 -ffp-contract=off`.
  - Every other library keeps the stock flags (It3: `-march=native` everywhere made J/K 4–6% slower).
  - The option is off by default, because the libraries then need an AVX2 CPU.
- **Bit for bit, because FMA contraction is off and nothing is reassociated:**
  - all 72 entries of `kernels.py` against `pyscf-it15` at 1 thread with `--rtol 0`;
  - every gradient and Hessian XC term of `deriv_xc_check.py` (caffeine RKS, benzene⁺ UKS; SVWN, B3LYP,
    r2SCAN);
  - the CPHF factored response `rks_fxc_mo` (benzene, caffeine; 3 functionals).
- **Timings** (10 threads, medians of 3 ABBA runs, `results/20261029-231043_it16a-flags.json`):

  | case | before (s) | after (s) | |
  |---|---|---|---|
  | caffeine B3LYP SCF | 5.26 | 4.97 | 1.06× |
  | caffeine r2SCAN SCF | 7.14 | 6.84 | 1.05× |
  | DF-TZVP SCF | 10.89 | 10.56 | 1.03× |
  | caffeine TDA | 25.55 | 23.90 | 1.07× (XC 18.5 → 16.9 s) |
  | benzene Hessian | 8.51 | 7.91 | 1.08× |
  | grid pipeline | 3.18 | 3.20 | unchanged; dense `eval_ao` is bound by writing its arrays |

### C: GEMM groups of 2 grid blocks (#16)
- **Problem:** the sparse and fused kernels gather the AOs unscreened on a group of `GEMM_NBLK` blocks and run one
  GEMM over the group. With 4 blocks (224 points) a thread's GGA AO buffer is 1.2 MB at ns = 170, above the
  1 MB L2.
- **Fix:** 2 blocks halve the buffer, at nearly the same mean ns (caffeine/SVP 152 vs 154).
  - `numint.py` reads the size from the library (`VXCfused_gemm_nblk`) for `group_naos` and `rks_fxc_mo`'s
    alignment.
- **Results** change by rounding: Vxc ≤ 1.5e-15 relative, f_xc 2e-17, exc identical. The fused and block_loop paths
  still agree bit for bit.
- **Timings on top of A** (`results/20261029-232522_it16c-nblk.json`):

  | case | A (s) | A+C (s) | |
  |---|---|---|---|
  | caffeine B3LYP SCF | 4.93 | 4.49 | 1.10× (XC kernels 2.36 → 1.96 s) |
  | caffeine r2SCAN SCF | 6.97 | 6.60 | 1.06× |
  | DF-TZVP SCF | 10.62 | 10.21 | 1.04× |
  | alkane SCF | 14.92 | 14.64 | 1.02× |
  | caffeine TDA | 23.89 | 23.54 | 1.015× |
  | benzene Hessian | 7.86 | 7.82 | within noise |

  - The 1-thread gain the swarm measured was 5%. The larger 10-thread gain fits the cache argument.

### B: GGA densities from the orbitals with one GEMM for the gradient; density path by flop count (#9)
- **The Z trick:**
  - `VXCfused_eval_rho_mo` formed the GGA gradient from v0 = ao[0]C and three more GEMMs v_c = ao[c]C, with
    rho_c = 2 Σ_o v0 v_c.
  - With Z = C v0ᵀ (one GEMM of the same size), rho_c = 2 Σ_n ao[c,n] Z[n]. That is two GEMMs per group instead
    of four, as It13's factored CPHF kernel already does. Meta-GGA keeps the v_c for tau.
- **The path choice:**
  - The fused path now picks between the orbital and density-matrix kernels by their flops summed over the AO
    groups (`_FusedXC.mo_path_cheaper`).
  - Before, it copied `_rho1_preferred`'s pair-mask ratio, which was tuned for the unfused kernels.
  - Flops per point and set (2-block groups):

    | system | xctype | orbitals then | orbitals with Z | dm |
    |---|---|---|---|---|
    | caffeine/SVP | GGA | 61.8k | 31.4k | 49.4k |
    | caffeine/TZVP | GGA | 120k | 61k | 188k |
    | alkane | GGA | 101k | 51.1k | 51.7k |
    | alkane | LDA/meta-GGA | — | 25.3k (orbitals) | 51.0k |

- **Results** change by rounding (Vxc 2.0e-16 relative in `test_fused_xc`).
  - `test_fused_xc` now compares dms that carry orbitals to 1e-13, and all others bit for bit.
- **Timings on top of A+C** (`results/20261029-233957_it16b-gga-z.json`). Cycle counts and energies to 9 decimals
  are identical.

  | case | A+C (s) | A+C+B (s) | |
  |---|---|---|---|
  | caffeine PBE SCF | 4.78 | 4.39 | 1.09× |
  | caffeine B3LYP SCF | 4.95 | 4.56 | 1.09× |
  | DF-TZVP SCF | 10.32 | 9.24 | 1.12× |
  | caffeine⁺ UKS SCF | 13.35 | 12.42 | 1.08× |
  | alkane B3LYP SCF | 14.60 | 14.48 | 1.01× (the two paths cost about the same) |
  | alkane r2SCAN, benzene⁺ UKS r2SCAN | | | unchanged; meta-GGA was already on the orbital path |

**Gate**
- Kernel tests pass against `pyscf-base`.
- Tight validation passes on the hessian, full and sparse suites.
- Upstream tests: 851 passed, 4 failed (the known xcfun failures).

## Session summary: It13 → It16 with the thread-pool fix, 2026-10-30

P = `pyscf-it13` with OpenBLAS's default spin (`OPENBLAS_THREAD_TIMEOUT=28`); N = `perf` `0835ee427` with the
pinned timeout of 20. 10 threads, `PYSCF_MAX_MEMORY=20000`, means of 2 ABBA runs
(`results/20261030-001558_session-it13-it16.json`).

| case | P (s) | N (s) | |
|---|---|---|---|
| scf_caffeine_svp_b3lyp | 8.83 | 4.20 | 2.10× |
| scf_caffeine_svp_r2scan | 11.22 | 6.56 | 1.71× |
| scf_caffeine+_svp_b3lyp_uks | 21.83 | 12.81 | 1.70× |
| scf_benzene+_svp_r2scan_uks | 4.01 | 2.06 | 1.95× |
| scf_caffeine_tzvp_dfpbe | 15.06 | 9.21 | 1.64× |
| scf_c20h42_svp_b3lyp | 19.02 | 14.46 | 1.31× |
| tda_caffeine_svp_b3lyp | 25.47 | 23.30 | 1.09× |
| grad_benzene_svp_b3lyp | 1.48 | 0.69 | 2.15× |
| hess_benzene_svp_b3lyp | 18.07 | 7.84 | 2.30× |
| grid_caffeine_svp | 3.49 | 3.17 | 1.10× |
| micro_libxc | 1.72 | 1.42 | 1.21× |

- The caffeine/SVP B3LYP Hessian (It14, both sides with the timeout of 20) went from 180.8 to 105.0 s.
- What remains largest:
  - the XC response of TD (16.5 of 23.3 s on caffeine TDA): It17, the factored TD response (#6);
  - the CPHF XC response and derivative J/K of Hessians;
  - the semi-direct J/K of large SCFs.

## It17: TD XC response from the factors, and the fxc kernels, 2026-10-30

`pyscf` `perf` commits (swarm items #6, #12, #17, #20):
- A `d3545eefb`: TDA, TDDFT and Casida XC response from the factors of the trial dms (#6).
- B1 `7a57c167e`: the fxc potentials of all sets in one threaded kernel (#12).
- B2 `3ad92c150`: the fused fxc contractions in one grid block per call (#17).
- C `1e31fb4d1`: the factored response in one kernel per group of points, and the flop model recalibrated
  (#20).

Worktrees: `pyscf-it16` (before), `pyscf-it17a` (A), `pyscf-it17b` (A+B).

**Why.** After It16 the XC response was 16.9 of 23.1 s of caffeine/SVP B3LYP TDA: 10 response calls, 160 trial
vectors, about 100 ms per vector on the dm path (`td_fxc_calls.py`). On benzene TDDFT it was 5.6 of 6.5 s
(17 calls, 280 vectors). It13's factored path served only CPHF.

### A: the TD response from the factors (#6)
- **The factors.** The XC response depends only on the symmetric part of a response dm. For the TD trial dms
  that part is a·orboᵀ + orbo·aᵀ, and the operators need only orbvᵀ v_xc orbo:

  | method | a |
  |---|---|
  | TDA | orbv zᵀ |
  | TDDFT (RPA) | orbv (x+y)ᵀ; the symmetric XC block adds the same to both halves |
  | Casida | 2 orbv (z·√e_ia)ᵀ |

- **Response function.** `_gen_rhf_response` now has, in every branch, one `vind_jk` per spin case (singlet and
  CPHF: J and K; triplet: K only), `vind_nlc` and `vind_xc` (`nr_rks_fxc`, or `nr_rks_fxc_st` for TD).
  - It attaches `vind.vind_lowrank(dm1, a, P, Q) -> (v1, w)`: vind(dm1) = v1 + v_xc, with w = Pᵀ v_xc Q from the
    factors where `lowrank_cheaper` says so; otherwise (vind(dm1), 0).
  - The singlet or triplet kernel fxc[0,:,0] ± fxc[0,:,1] is formed at the first factored call.
  - CPHF's `vind_mo` is now built on `vind_lowrank`.
  - `_gen_tda_operation`, `_gen_tdhf_operation` and `CasidaTDDFT.gen_vind` use it where it is offered. J, K and
    NLC stay in the AO basis.
- **Conditions:** NumInt's own `nr_rks_fxc` and `nr_rks_fxc_st` (`numint._fused_xc_mo`), and real factors.
  Wrapped response functions (the solvent models) carry no `vind_lowrank` and keep the AO path.
- **Exactness:**
  - On the dm path the TD operators (TDA, TDDFT, Casida; singlet and triplet; SVWN, B3LYP, r2SCAN, CAM-B3LYP)
    and the CPHF operator are bit for bit those of It16 (benzene, 1 thread; `td_cphf_bitwise.py`).
  - On the factored path the operators agree with `FUSED_XC=False` to ≤ 1.8e-14 relative (water and benzene;
    SVWN, PBE, B3LYP, r2SCAN, CAM-B3LYP; `td_lowrank_check.py`).
  - New test: `test_tdrks.test_td_xc_lowrank`.
- **Timings** (P = `pyscf-it16`; medians of 3 interleaved runs; `results/20261030-052350_it17a-td.json`, and
  `results/20261030-052834_it17a-td-mem20g.json` for caffeine and the alkane):

  | case | P (s) | A (s) | |
  |---|---|---|---|
  | TDA benzene/SVP B3LYP | 2.42 | 1.65 | 1.47× |
  | TDDFT benzene/SVP B3LYP | 7.25 | 4.52 | 1.60× |
  | TDA caffeine/SVP B3LYP | 23.32 | 18.36 | 1.27× (XC 16.4 → 11.4 s) |
  | TDA n-C20H42/SVP PBE | 36.76 | 36.69 | the dm path is kept (flop ratio 1.13) |

- **Harness note.** `run_bench`'s default `PYSCF_MAX_MEMORY` (4000) keeps caffeine's ERIs out of memory: the
  caffeine TDA then spends 12 s in semi-direct J/K instead of 5.6 s in memory. TD and SCF comparisons use
  `:PYSCF_MAX_MEMORY=20000`, as the user's environment does.

### B: the fxc potentials in one kernel (#12), and one block per call (#17)
- **B1.** `_FusedXC.rks_fxc` and `rks_fxc_mo` formed wv = Σ_y rho1[y] fxc[x,y] w per set with `numpy.einsum`, on
  one thread, reading fxc once per set. `numint._fxc_wv` (`VXCfused_fxc_wv`) does all sets on all threads in
  blocks of 512 points, reading each block of fxc once.
  - The terms are (rho1·fxc)·w added in the order of y, as numpy's einsum does. The results are bit for bit the
    einsum's (benzene, caffeine; SVWN, PBE, r2SCAN; CPHF and TD singlet kernels; `fxc_wv_check.py`).
  - Seven sets on caffeine's whole grid: PBE 0.028 → 0.008 s, r2SCAN 0.043 → 0.011 s.
- **B2.** `_fused_fxc_blksize` gave the fused fxc contractions `block_loop`'s blocks (at most 1200·56 points).
  The fused kernels hold no AO arrays, but each block allocates and sums nthreads·nset private nao² matrices.
  - The blocks are now bounded only by rho1 and wv in a quarter of `max_memory`. On the bench molecules that
    is one block per call; for TD with 20 dms it was 3 (benzene), 5 (caffeine) and 11 (alkane).
  - Results change by rounding: the sums over blocks go in another order.
- **Timings, A → A+B** (`results/20261030-064646_it17-abc.json`, medians of 3 interleaved runs):

  | case | A (s) | A+B (s) | |
  |---|---|---|---|
  | TDA benzene | 1.57 | 1.48 | 1.06× |
  | TDDFT benzene | 4.47 | 4.24 | 1.05× |
  | TDA caffeine | 18.27 | 17.83 | 1.02× |
  | TDA alkane (dm path) | 36.83 | 33.74 | 1.09× |
  | Hessian benzene B3LYP | 7.85 | 7.58 | 1.04× |
  | Hessian benzene r2SCAN | 11.02 | 10.55 | 1.04× |

### C: the factored response in one kernel (#20), and the flop model
- **One kernel.** `VXCfused_fxc_lowrank` does all of `rks_fxc_mo`'s work on each group of points:
  - the AO values and TB = ao·B once;
  - then per set the densities of A Bᵀ + B Aᵀ, the potentials (halved as `_dot_wv` does), and v1 B, whose
    products ao·Q are TB when Q = B.
  - Before, two kernels each evaluated the AO values and their products with the orbitals, and rho1 and wv of
    every block went through Python.
  - The three kernels now share their per-group code. `VXCfused_eval_rho_lowrank` and `VXCfused_dot_ao_mo` give
    the same results as before, bit for bit (`lowrank_kernels_bitwise.py`).
  - `rks_fxc_mo` takes the one kernel where Q is B, as in all its callers; `one_pass=False` keeps the two.
  - One kernel agrees with two to 7.7e-15 relative, and with `nr_rks_fxc` to 1.3e-14.
- **Batching the sets into wider GEMMs** (the other half of #20) is ruled out (`gemm_shapes.c`). Per thread, dgemm
  at one set's shape (M = 112 points, K = 150 AOs, N = 51 orbitals) runs at 510 GFLOP/s, and wider N at
  333–419. The per-set operands stay in the 1 MB L2; batched ones do not.
- **Per call, two kernels → one kernel** (`lowrank_paths_time.py`, TDA-type dms, 10 threads):

  | system, xc | 1 dm | 5 dms | 20 dms |
  |---|---|---|---|
  | caffeine SVWN | 0.068 → 0.041 s | 0.165 → 0.144 s | 0.608 → 0.550 s |
  | caffeine B3LYP | 0.177 → 0.115 s | 0.411 → 0.355 s | 1.540 → 1.436 s |
  | caffeine r2SCAN | 0.262 → 0.205 s | 0.974 → 0.923 s | 2.713 → 2.764 s |
  | alkane PBE | 0.706 → 0.503 s | 1.685 → 1.465 s | 5.359 → 5.146 s |

  - Meta-GGA at 20 or more dms is 2–3% slower in one kernel; its per-thread buffers (5 GEMM segments, about
    2 MB) overflow L2.
- **Flop model.** `lowrank_cheaper` now models the one-kernel pass (the products with Q once per pass), and
  weighs a factored flop at 0.8 (LDA), 0.9 (GGA) and 1.0 (meta-GGA) of a dm flop.
  - Measured time ratio over modelled flop ratio: LDA 0.6–0.8, GGA 0.83–1.0, meta-GGA 0.9–1.0 (benzene,
    caffeine, alkane; 1–40 dms).
  - It13's threshold of 0.8 dated from when a factored flop cost 1.1–1.24 dm flops. It kept calls on the dm
    path that run 1.2–1.45× faster factored (caffeine B3LYP with 2–5 dms).
  - The alkane's PBE calls with 10 or more dms now go factored: modelled 1.02–1.09, measured 0.86–0.96.
- **Timings, A+B → A+B+C** (same file):

  | case | A+B (s) | A+B+C (s) | |
  |---|---|---|---|
  | TDA benzene | 1.48 | 1.40 | 1.06× |
  | TDDFT benzene | 4.24 | 3.86 | 1.10× |
  | TDA caffeine | 17.83 | 17.35 | 1.03× |
  | TDA alkane | 33.74 | 31.92 | 1.06× |
  | Hessian benzene B3LYP | 7.58 | 7.32 | 1.04× |
  | Hessian benzene r2SCAN | 10.55 | 10.35 | 1.02× |

  - `worker.py` now counts `_fxc_wv` and `rks_fxc_mo` (which calls the one kernel through ctypes) as
    `xc_kernels`. The file above predates that, so the C runs show their XC time under `other`. From the saved
    top functions, XC went from 11.5 to 10.8 s (caffeine TDA) and from 25.0 to 20.8 s (alkane TDA) between A
    and A+B+C.

### The accumulating kernels' static schedule (measured, not changed)
- `VXCfused_dot_wv` (the XC matrices of SCF and the dm-path fxc), `VXCfused_dot_ao_mo` and
  `VXCfused_fxc_lowrank` use `schedule(static, 1)`, so that each thread's private sums, and with them the results,
  do not depend on timing. The density kernels use `dynamic, 4`.
- Experiment (`fused_sched_toggle.py`: `dynamic, 4` in those three kernels, which makes them
  non-deterministic):
  - per call on caffeine (20–40 dms): one-kernel factored path 1.235× (B3LYP) and 1.16× (r2SCAN); dm path
    1.06–1.19×;
  - SCFs (`results/20261030-062252_it17-sched-probe.json`): caffeine B3LYP 4.27 → 3.89 s (XC kernels 1.66 → 1.46
    s), r2SCAN 6.39 → 6.06 s; alkane XC −4%.
- The implied speed ratio of a Zen 5 to a Zen 5c core in these kernels is about 1.6: static(1) gives every thread
  the same work, so the four fast cores idle about a third of the time.
- A deterministic dynamic schedule would need per-slot sums (a fixed partition of the groups into slots, taken
  dynamically, each summed into its own buffer, and the buffers added in slot order). That is swarm #19, larger
  than its estimate of 1–2.5%.
- The user first chose to keep the static schedule, then asked for it the same day: It18 below.

### It16 → It17
P = `pyscf-it16` (`0835ee427`), N = `perf` `1e31fb4d1`; 10 threads, `PYSCF_MAX_MEMORY=20000`, means of 2 interleaved
runs (`results/20261030-073148_it16-it17.json`; the caffeine Hessian 1 rep, `20261030-073915_it16-it17-hess.json`).

| case | P (s) | N (s) | |
|---|---|---|---|
| TDA benzene/SVP B3LYP | 2.40 | 1.33 | 1.81× |
| TDDFT benzene/SVP B3LYP | 6.72 | 3.69 | 1.82× |
| TDA caffeine/SVP B3LYP | 24.65 | 17.88 | 1.38× (XC 17.4 → 11.2 s; J/K 5.9 → 6.1 s) |
| TDA n-C20H42/SVP PBE | 36.65 | 32.03 | 1.14× (XC 24.9 → 20.9 s) |
| Hessian benzene/SVP B3LYP | 7.74 | 7.28 | 1.06× |
| Hessian benzene/SVP r2SCAN | 10.87 | 10.34 | 1.05× |
| Hessian caffeine/SVP B3LYP | 94.88 | 94.18 | unchanged (its CPHF calls are large, where one kernel gains 4–7%) |
| SCF caffeine/SVP B3LYP (control) | 4.32 | 4.33 | unchanged; It17 does not touch the SCF |

- What remains in TD:
  - caffeine TDA: XC 11.2 s, in-core J/K 6.1 s;
  - alkane TDA: XC 20.9 s (dm path for small calls), semi-direct J 10.5 s.

### Gate
- A (`pyscf-it17a`):
  - kernel tests pass;
  - tight validation passes on the hessian, full and sparse suites. TD excitations agree with unmodified PySCF
    to 2.2e-15 Eh (caffeine TDA), 4.8e-15 (benzene TDA), 4.4e-14 (benzene TDDFT) and 2.0e-15 (alkane TDA);
  - upstream tests: 852 passed (851 + the new test), 4 failed (the known xcfun failures).
- B+C (`pyscf` at `1e31fb4d1`):
  - kernel tests pass;
  - tight validation passes on the hessian, full and sparse suites. TD excitations agree to 1.6e-15 Eh (caffeine
    TDA), 4.1e-15 (benzene TDA), 7.5e-14 (benzene TDDFT) and 2.1e-15 (alkane TDA); Hessians to 7.3e-12
    (benzene) and 6.6e-11 (water, DF);
  - upstream tests: 852 passed, 4 failed (the known xcfun failures).

## It18: deterministic dynamic schedules (swarm #19), 2026-10-30

`pyscf` `perf` commits:
- A `c93703cb0`: the XC kernels that sum partial matrices take their groups by a deterministic dynamic
  schedule.
- B `219115c29`: `NPdgemm`'s m- and n-splits take fixed chunks by a dynamic schedule.

Worktree `pyscf-it17` (`1e31fb4d1`) is the before-build. UKS, planned as It18, moves to It19.

**Why.** It17 measured that the static schedule of these kernels costs 6–24% on the mixed cores: the Zen 5 cores
run them about 1.6× faster than the Zen 5c cores, and `schedule(static, 1)` gives every thread the same work.

### A: streams of slots, each summed in order (`nr_numint_gemm.h`)
- **Kernels:** `VXCfused_dot_wv` (SCF XC matrices and dm-path fxc), `VXCfused_dot_terms` (gradient and Hessian
  XC terms), `VXCfused_dot_ao_mo` and `VXCfused_fxc_lowrank` (factored response), and `_dot_ao_ao_gemm` (behind
  the `*_sparse_gemm` kernels).
- **Design:**
  - The groups go in slots of 4, and the slots in 2·nthreads contiguous streams, each with a partial matrix
    of its own.
  - A thread takes the next slot of an unlocked stream (`omp_test_lock`) and adds the slot's products to that
    stream's matrix while it holds the lock. It prefers the stream with the most slots left, but stays on its
    previous stream while that has at most one slot fewer.
  - Each stream is thus summed in slot order, whichever threads take its slots, and the streams' matrices are
    added in stream order.
  - Results are reproducible for a given thread count, as before. The fused and sparse kernels still agree bit
    for bit (`test_fused_xc`).
- **Why streams and not fixed slots:** the swarm proposed 4·nthreads fixed slots, each with its own matrix.
  A simulation with the measured core speeds (`sched_sim.py`) gives these makespans relative to perfect
  balance:

  | schedule | makespan |
  |---|---|
  | static(1) | 1.18–1.28 |
  | 40 fixed slots | 1.15–1.23 |
  | 64 fixed slots | 1.09–1.14 |
  | 2·nthreads streams | 1.01–1.05 (with the stickiness margin of 1) |
  | non-deterministic dynamic,4 | 1.003–1.017 |

- **Allocation matters.** A first version callocked all the streams' matrices in one block per call, and lost
  more than balance saved:
  - per call, 9.7 MB for one set and 194 MB for 20 on caffeine/SVP;
  - page faults when the block was freshly mapped, or one thread zeroing it when it came from the heap;
  - `dot_wv` on SCF blocks went 58.0 → 67.9 ms with one set.
  - Now each stream's matrix is a separate `malloc` (reused across calls) and is zeroed when its first slot is
    taken, by the thread that adds to it.
- **The stickiness margin** halves the stream switches; in `dot_wv` it is within noise of none.
- **Memory:** twice the per-thread matrices of before. `numint.STREAMS_PER_THREAD` (from the library) sizes the
  Python side's batches.
- **Exactness** (`sched_determinism.py`):
  - repeated calls at 10 threads give the same bits: `nr_rks` fused and unfused, `nr_rks_fxc`, `rks_fxc_mo` in
    one and two kernels, gradient `get_vxc`; caffeine/SVP, PBE and r2SCAN;
  - against the static schedule on the same orbitals: ≤ 5.3e-15 relative.
- **`VXCfused_dot_wv` per call** (`dot_wv_time.py`, 10 threads; static → streams; a non-deterministic
  `dynamic, 4` in brackets):

  | case | 1 set | 2 sets | 20 sets |
  |---|---|---|---|
  | caffeine/SVP B3LYP, SCF blocks | 58.0 → 54.3 ms (51.9) | 115.2 → 99.2 ms (97.0) | 1067 → 1068 ms (1024) |
  | caffeine/SVP B3LYP, whole grid | 55.6 → 50.8 ms (49.1) | 101.5 → 92.5 ms (90.7) | 1019 → 954 ms (997) |
  | caffeine/TZVP PBE, SCF blocks | 274.6 → 243.8 ms (242.9) | 497.3 → 472.2 ms (453.1) | 4910 → 4472 ms (4599) |
  | caffeine/TZVP PBE, whole grid | 274.5 → 231.0 ms (239.0) | 476.3 → 429.6 ms (441.1) | 4764 → 4019 ms (3968) |

  - With 20 sets on SCF-sized blocks the doubled partial memory per call costs about what balance saves. The
    fxc contractions now run in one block per call (It17 B2).

### B: `NPdgemm`'s m- and n-splits in fixed chunks
- Chunks of about 4 per thread, multiples of 64 rows or columns, by `schedule(dynamic, 1)`.
- Each element is still computed by one dgemm over all of k, so the result does not depend on the timing.
  Moving the chunk boundaries changes m-split products by rounding in OpenBLAS (≤ 1.3e-15 relative). The n-split
  shapes tried, and the k-split (unchanged), are bit for bit (`npdgemm_check.py`).
- `lib.dot`: 50000×246×246 20.7–27.1 → 18.4–18.8 ms, 60001×114×114 7.1–7.5 → 6.5–6.9 ms. The n- and k-split
  shapes are within noise.

### End to end
P = `pyscf-it17`, N = `perf` `219115c29`; 10 threads, `PYSCF_MAX_MEMORY=20000`, means of 2 interleaved runs
(`results/20261030-085612_it18-sched2.json`).

| case | P (s) | N (s) | |
|---|---|---|---|
| SCF caffeine/SVP B3LYP | 4.45 | 4.16 | 1.07× (XC kernels 1.83 → 1.54 s) |
| SCF caffeine/SVP r2SCAN | 6.58 | 6.34 | 1.04× |
| SCF caffeine⁺/SVP B3LYP UKS | 12.48 | 12.17 | 1.025× |
| SCF caffeine/TZVP DF-PBE | 9.15 | 8.84 | 1.035× |
| SCF n-C20H42/SVP B3LYP | 14.57 | 14.32 | 1.02× |
| SCF benzene⁺/SVP r2SCAN UKS | 2.11 | 1.86 | 1.14× |
| TDA caffeine/SVP B3LYP | 17.07 | 15.95 | 1.07× |
| TDDFT benzene/SVP B3LYP | 3.78 | 3.40 | 1.11× |
| TDA n-C20H42/SVP PBE | 31.66 | 29.23 | 1.08× |
| gradient caffeine/SVP B3LYP | 7.12 | 7.04 | 1.01× |
| Hessian benzene/SVP B3LYP | 7.65 | 7.05 | 1.085× |
| Hessian benzene/SVP r2SCAN | 10.35 | 9.69 | 1.07× |
| grid pipeline caffeine/SVP | 3.21 | 3.09 | 1.04× |

- The first version (one zeroed block per call, `results/20261030-083244_it18-sched.json`) made the UKS and
  DF-TZVP SCFs 1–2% slower.

### Gate
- Kernel tests pass against `pyscf-base`.
- Tight validation passes on the hessian, full and sparse suites. Largest differences from unmodified PySCF:
  energies 4.6e-12 Eh, gradients 3.8e-12, excitations 2.6e-14, Hessians 6.6e-11 (water, DF).
- Upstream tests: 852 passed, 4 failed (the known xcfun failures).
- **Cycle counts.** The tight SCFs' cycle-count warnings move between gates even when the SCF code does not
  change (It17a and It17 B+C flagged different cases). This is the run-dependent endpoint of tight direct SCFs
  noted since It7: the semi-direct and direct J/K sum in thread order.
  - The tight CAM-B3LYP SCF took 30 cycles against the reference's 24 in this gate.
  - Six reruns with It18 took 24, 24, 24, 27, 29 and 30 cycles, and three with It17 took 24, 24 and 27.
  - Its energy was always within 2.5e-12 Eh. Its long-range K is semi-direct with a dynamic schedule; the XC
    kernels and `NPdgemm` are deterministic.

## It19: UKS: J/K of both spins in one pass, the factored XC response, J of the total density, 2026-10-30

`pyscf` `perf` commits:
- A `22c74b9e4` (swarm #10): in-memory J/K of up to 6 symmetric dms in one pass over the ERIs.
- C `ab7906363`: the UKS XC response from the factors of the spin dms (TDA, TDDFT, Casida, CPHF).
- D `06b7253bc`: J of the total density in the pure-functional UKS response and in the UHF/UKS Hessians'
  `make_h1`.
- B (workspace): bench cases for the UKS response.

Worktree `pyscf-it18` (`219115c29`) is the before-build; `pyscf-it19` is the gated build.

### A: J/K of both spins in one pass over the in-memory ERIs (#10)
- **Why.** UHF, UKS, ROHF and ROKS pass both spin densities to `get_jk`. With the ERIs in memory, two dms took
  It7's BLAS-3 batch kernels: J by tiled GEMMs with 2 columns, and K by copying each packed row into a square
  matrix for two `dsymm` with 2 columns. That was 181 ms per call against 54 ms for one dm (caffeine+/SVP,
  3.69 GB of ERIs, 10 threads), and 5.6 of the 12 s of the caffeine+ UKS SCF.
- **Kernel.** `CVHFnrs8_incore_jk_ndm` replaces `CVHFnrs8_incore_jk_1dm`. Each ERI row is read from memory once
  for all dms: J of each dm in one loop over the row, then K one segment of the row (k fixed) at a time.
  - A first version summed the dms one after another: 82 ms for 2 dms. Its dot products were bound by the
    latency of one chain of dependent additions: 4 cycles per 4 elements, 3 such loops per element and dm,
    about 38 ms per dm at 10 threads, as measured.
  - The sums of all dms, and both K updates of a row, now go side by side in one loop, in chunks of up to 4
    dms (a compile-time constant). Without `#pragma GCC unroll 4` the per-dm arrays stayed in memory and 2 dms
    took 99 ms.
  - Each sum keeps its order, so each dm's J and K are bit for bit those of a call with that dm alone. With one
    dm the kernel equals the old one bit for bit (51–53 against 54 ms), so the RKS SCF is unchanged.
- **Per call** (`incore_ndm_time.py`, caffeine+, 10 threads, ms):

  | dms | batch kernels | one pass |
  |---|---|---|
  | 1 | 54 (old 1-dm kernel) | 52 |
  | 2 | 181 | 66 |
  | 3 | 203 | 91 |
  | 4 | 202 | 136 |
  | 6 | 231 | 210 |
  | 8 | 248 | 355 |

  - J alone: the one-pass kernel is faster up to 3 dms (2 dms: 46.6 against 56.3 ms), equal at 4, slower at 6
    (120 against 82 ms).
  - Cutoffs: `INCORE_FUSED_MAX = 6` and `INCORE_FUSED_J_MAX = 3`. On benzene+ (0.17 GB) the one-pass kernel wins
    at every n up to 8.
- caffeine+ UKS B3LYP SCF: 12.05 → 9.22 s (J/K 5.68 → 2.68 s), 23 cycles in both, energies 2.7e-12 Eh apart.

### B: bench cases for the UKS response
- `td_case` takes charge and spin, and `converged_ks`, `td_case` and `hess_case` take `full_jk`: the caffeine+
  cases rebuild J/K in the tight SCF, as `scf_caffeine+_svp_b3lyp_uks` does.
- FULL: `tda_caffeine+_svp_b3lyp_uks` and `hess_benzene+_svp_b3lyp_uks`. Timings only:
  `hess_caffeine+_svp_b3lyp_uks`.
- The 'hessian' suite, which collects the response code paths, adds water+ UKS TDA for SVWN, PBE, r2SCAN, B3LYP
  and wB97X, and UKS TDDFT for PBE and r2SCAN (Casida) and B3LYP (full RPA).
- `worker.py` counts `uks_fxc_mo` as XC.

### C: the UKS XC response from the factors of the spin dms
- **Why.** It13 and It17 gave the RKS CPHF and TD operators the XC response from the factors of the response
  dms. UKS kept the dm path: `nr_uks_fxc` on two dms per set. It was 27 of the 39 s of the caffeine+ UKS TDA
  and 115 of the 204 s of the caffeine+ UKS Hessian.
- **Kernel.** `VXCfused_fxc_lowrank` becomes `_fxc_lowrank` for one or two spins, with
  `VXCfused_fxc_lowrank_uks` as the UKS entry. Each group of points:
  - evaluates its AO values once for both spins;
  - takes the densities of both spins from their factors;
  - forms the spin-coupled potentials of `nr_uks_fxc`, with the kernel's first index pair contracted with
    the densities, as `nr_uks_fxc` does (the UKS kernel is symmetric only to rounding, about 1e-17 relative);
  - adds v1[s] Q[s] for each spin.

  The RKS kernel performs the same operations as before: `sched_determinism.py` against It18 gives 0.0 for every
  kernel.
- **Python.**
  - `_FusedXC.uks_fxc_mo` drives the kernel.
  - `lowrank_cheaper(nspin=2)` weighs it against the 2·nset dms of the dm path, with the factored terms summed
    over the spins.
  - `_gen_uhf_response` is split as `_gen_rhf_response` is (J/K, NLC, XC). It attaches
    `vind_lowrank(dm1, a, P, Q)` for factors given per spin, and `vind_mo(dm1, x)` for CPHF when the orbitals
    come per spin. ROHF's own `mo_coeff` and `mo_occ`, which callers without orbitals fall back to, get no
    `vind_mo`.
  - The UHF/UKS TDA (a = orbv zᵀ/2), TDHF (a = orbv (X+Y)ᵀ/2, the XC block added to both halves) and Casida
    (a = orbv (z·d)ᵀ) operators and the UHF Hessian's CPHF operator use them where offered.
- **A bug that the edge cases caught:** with no beta electrons (H2+), `vind_mo` inferred the number of sets
  from each spin's x, and the empty beta block cannot tell. It now takes the count from a spin with orbitals.
  The H2+ case is in `test_cphf_xc_lowrank`.
- **Per call** (`lowrank_uks_time.py`, caffeine+, 10 threads, s; dm path → factored):

  | xc | 1 set | 10 sets | 40 sets |
  |---|---|---|---|
  | SVWN | 0.163 → 0.068 | 1.884 → 0.548 | 7.667 → 2.161 |
  | PBE | 0.210 (dm path kept; 0.217 factored) | 2.068 → 1.287 | 8.255 → 4.821 |
  | B3LYP | 0.215 (dm path kept; 0.212 factored) | 2.063 → 1.272 | 8.257 → 4.834 |
  | r2SCAN | 0.610 → 0.352 | 6.091 → 2.501 | 24.286 → 9.757 |

  - The flop model picks the faster path at every point.
  - The paths agree to 2.6e-14 relative.
  - The UKS gain (GGA 1.6–1.7×) is larger than the RKS one (1.3–1.4×): the dm path pays for both spins'
    dms, where the factored path shares the AO values and their products between the spins.
- **Tests:**
  - `test_tduks.test_td_xc_lowrank` (TDA, TDDFT, Casida; SVWN, BP86, B3LYP, M06-L) and hessian
    `test_uks.test_cphf_xc_lowrank`, both against `FUSED_XC=False` to 1e-13 with either path forced.
  - Edge cases (`uks_lowrank_edge.py`): H2+ and the H3 quartet, which have no beta electrons, agree to
    ≤ 1.5e-15 for TDA, TDDFT and CPHF.

### D: J of the total density
- J depends on the spin densities only through their sum.
  - The pure-functional branch of `_gen_uhf_response` now takes J of the summed spin dms: half the J work on
    any J backend.
  - The UHF and UKS `make_h1` contract each atom's `int2e_ip1` integrals with one Coulomb script on the total
    density instead of one per spin: 6 contractions per quartet instead of 8 (hybrids), 2 instead of 4 (pure).
- `make_h1`'s J/K call for one atom (`h1_jk_scripts.py`, ms): caffeine+ B3LYP 805 → 741, PBE 581 → 518;
  benzene+ 83 → 77 and 61 → 55. That is about 1.5 s per caffeine+ Hessian. On benzene+ it is lost in the
  noise of `make_h1` (2.7–3.0 s, mostly the XC term; `make_h1_time.py`). h1ao changes by 1.8e-14 relative.
- **Not done:** J of the spin sum in hybrid responses. `get_jk` computes J and K in one pass on the direct and
  semi-direct backends, so splitting it would evaluate the integrals twice. In memory J and K are separate
  passes, but the saving would be about 1% of a UKS TDA.

### End to end
It18 against It19: 10 threads, `PYSCF_MAX_MEMORY=20000`, means of 2 interleaved runs
(`results/20261030-132547_it19-uks.json`).

| case | It18 (s) | It19 (s) | |
|---|---|---|---|
| SCF caffeine+/SVP B3LYP UKS | 11.90 | 8.61 | 1.38× (J/K 5.61 → 2.60 s) |
| TDA caffeine+/SVP B3LYP UKS | 38.91 | 27.48 | 1.42× (XC 26.9 → 17.4 s) |
| Hessian benzene+/SVP B3LYP UKS | 19.19 | 12.41 | 1.55× |
| Hessian caffeine+/SVP B3LYP UKS | 203.9 | 156.8 | 1.30× (CPHF XC about 100 → 61 s) |
| SCF benzene+/SVP r2SCAN UKS | 1.83 | 1.82 | unchanged (pure: J of one summed dm already) |
| SCF, TDA caffeine/SVP B3LYP; Hessian benzene/SVP B3LYP (RKS) | 4.25, 16.42, 7.03 | 4.30, 15.99, 7.07 | bit-for-bit results; times within noise |

- **Exactness between the trees:**
  - UKS SCF energies differ by 2.7e-12 Eh with the same cycles, and excitations by 1.3e-14.
  - The Hessians differ by 1.6e-9 (benzene+) and 1.6e-11 (caffeine+).
- **The benzene+ Hessian:**
  - Its CPHF amplifies rounding: run to run, each tree's Hessian varies by 5e-11, against 5e-13 for neutral
    benzene. Benzene+ is a Jahn–Teller cation with near-degenerate orbitals.
  - The 1.6e-9 comes from each run's own default SCF, whose endpoint (conv_tol 1e-9) differs between the trees.
  - On the same orbitals (`hess_cpscf_tol.py`) It19 and It18 differ by 5.2e-11 at the default
    `conv_tol_cpscf` and 1.4e-10 at 1e-12. It18 against itself gives 1.2e-11 and 4.5e-11.
- **Where UKS time goes now:**
  - caffeine+ TDA (27.4 s): XC response 17.4 s, J/K 8.8 s.
  - caffeine+ Hessian (157 s):
    - CPHF XC 61.3 s;
    - derivative J/K (`nr_direct_drv`) 48.4 s, It20's target;
    - CPHF J/K 18.6 s;
    - the Hessian XC terms about 13 s (`_dot_sets_aow_ao` 7.3 s, the per-atom wv einsums 5.2 s).

### Gate
- Kernel tests pass against `pyscf-base`.
- Tight validation passes on the hessian suite (now with the 8 water+ UKS TD cases), the full suite and the
  sparse suite. Largest differences from unmodified PySCF:

  | quantity | max difference | case |
  |---|---|---|
  | energies | 5.5e-12 Eh | |
  | gradients | 3.8e-12 | |
  | excitations | 1.3e-12 | water+ UKS r2SCAN Casida; caffeine+ UKS TDA 7.9e-16 |
  | Hessians | 6.6e-11 | water, DF; benzene+ UKS 4.7e-11 |

- Upstream tests: 855 passed, 4 failed (the known xcfun failures).
- **Cycle-count warnings** on five tight SCFs, among them the RKS caffeine B3LYP (27 against 24).
  - None of It19's code is on its path: at the validation's 4000 MB it runs semi-direct.
  - Three reruns each: It19 took 24, 27 and 26 cycles, It18 24, 29 and 24, with energies within 3.9e-12 Eh.
  - This is the run-dependent endpoint of tight direct SCFs, noted since It7.

## It20: derivative J/K: the gradient's two-electron term inside the integral loop, and the contraction kernels, 2026-10-30

`pyscf` `perf` commits:
- A `89ee07e8d` (swarm #15, first step): derivative J/K kernels sum four outputs at a time in registers.
- B `9ab2db1e0` (swarm #11): the two-electron gradient contracted inside the integral loop.
- Bench cases (workspace): `grad_caffeine+_svp_b3lyp_uks` (FULL), and water gradients with CAM-B3LYP (RKS) and
  PBE, B3LYP, wB97X (UKS) in the 'hessian' suite.

Worktree `pyscf-it19` (`06b7253bc`) is the before-build; `pyscf-it20` is the gated build.

**Where the time went.** The gradient's J/K step (`rhf_grad.get_jk` through `grad_elec`) was 94–95% of every
gradient case (benzene 0.47 of 0.58 s, caffeine 5.99 of 6.42 s, n-C20H42 28.6 of 30.1 s at 10 threads). One
thread, caffeine: `int2e_ip1` evaluation about 80% of it, the two contraction scripts 17%, the prescreen 1%.

### A: derivative J/K kernels (#15, first step)
- **Change.** Six kernels of `nr_direct_dot.c` used by the gradient and Hessian scripts (`ji->s1kl`, `li->s1kj`,
  `lj->s1ki`, `jk->s1il`, and the s2kl branches of `li->s1kj` and `jk->s1il`) summed one output at a time: a
  chain of dependent additions, or a load and store of the output for every term. They now hold four outputs (or
  two pairs) in registers and sum them side by side. Each output keeps the order of the one-output loop.
- **Exactness.** Bit for bit: the 34 matrices of `deriv_jk_kernels.py` (gradient RHF and UHF, `make_h1` RKS and
  UKS scripts, `int2e_ip1ip2` and `int2e_ipvip1` of `_partial_hess_2e`, two atoms of benzene, 1 thread) are
  equal to It19's.
- **Per script** (`deriv_jk_scripts.py`, benzene atom 6, 1 thread; marginal cost of adding each script, s):

  | call | script | It19 | It20 |
  |---|---|---|---|
  | `int2e_ip1ip2` s1 | `li->s1kj` | 0.047 | 0.031 |
  | | `lj->s1ki` | 0.050 | 0.038 |
  | `int2e_ipvip1` s2kl | `li->s1kj` | 0.035 | 0.028 |
  | `int2e_ip1` s2kl (`make_h1`) | `lk->s1ij` | 0.013 | 0.013 |
  | | `li->s1kj` | 0.019 | 0.016 |
  | | `jk->s1il` | 0.020 | 0.017 |

  Whole calls: `int2e_ip1ip2` 0.552 → 0.531 s, `int2e_ipvip1` 0.284 → 0.279 s, `make_h1` 0.190 → 0.189 s.
- **Why so little.** Each script is 3–9% of its call; the integrals are about 85%. The second step of #15,
  fusing the scripts of a call, can at best bring them to the cost of one contraction per integral. For
  `_partial_hess_2e`, about 80% of the Hessians' derivative J/K, an energy-style driver like B's would do that
  (the ej/ek blocks are contracted with the density right away): about 12% of it for RKS (3 + 2 scripts become
  one contraction each) and 25% for UKS (5 + 3). Its integral count is already minimal: 6 second-derivative
  evaluations per s8 orbit (4 `int2e_ip1ip2`, 2 `int2e_ipvip1`), with the diagonal blocks from translational
  invariance since It14. Not done in It20.

### B: the gradient's two-electron term contracted inside the integral loop (#11)
- **Before.** `grad_elec` contracted the J/K matrices of `get_jk` with the density. The s2kl driver evaluates
  `int2e_ip1` (nabla i j|kl) for all i, all j, k >= l: one integral per center of every s8 orbit, 4 per orbit,
  each contracted by two scripts into matrices that `grad_elec` then contracts again.
- **Driver.** `CVHFgrad_jk_energy` (`lib/vhf/grad_jk_energy.c`) visits each s8 orbit once and contracts its
  derivative integrals with Gamma_pqrs = cj Dj_pq Dj_rs - ck/2 sum_n (Dk_pr Dk_qs + Dk_ps Dk_qr) straight into
  dE/dR (dE2/dR_A = -deg/2 sum over the centers on A of G_c). By translational invariance the centers on one
  atom need no integrals once the others are known: 4 atoms take 3 evaluations, 2 + 2 take 2, 3 + 1 take 1, one
  atom none. The atom left out is the one with the most centers, then the lowest angular momentum (3% faster than
  the highest on caffeine, 1 thread; a VRR-size model was in between).
- **Screening.** Each of the 4 orderings gets the s2kl driver's own test (`CVHFgrad_{jk,j,k}_prescreen`). An
  orbit is skipped only if none passes; otherwise it takes the cheaper of the passing orderings (exactly the
  integrals the s2kl driver evaluates) or all 4 centers by translational invariance. Both include everything the
  s2kl driver includes. A pair bound (max of the two derivative estimates and the plain one) and a pair list
  sorted by it stop each pair's inner loop at the first partner that cannot pass.
- **Determinism.** The orbits are summed in a fixed partition of up to 4096 chunks of the pair list, each into
  its own buffer, then the buffers in order: the result does not depend on the thread count or the schedule
  (`repeat 0` in every check).
- **Python.** `grad.rhf.jk_energy_per_atom(mol, dm, j_factor, k_factor, omega, screen)` gives dE2/dR for a total
  (RHF) or alpha/beta (UHF) density. `grad_elec` of RHF/RKS and of UHF/UKS/ROHF/ROKS (which share `uhf.grad_elec`)
  calls `_get_veff_jk_energy`. It uses the energy path when the object's `get_veff` is a stock one
  (`rhf_grad._VEFF_SPLIT`) and its `get_jk`, `get_j`, `get_k` are `GradientsBase`'s. Density fitting, SGX,
  DFT+U and any override keep the matrices. `get_veff` itself still returns the full matrices; the XC part of
  RKS/UKS moved to `_get_vxc_veff`. A hybrid's J/K screens as `get_jk` did (`screen='jk'`), its long-range K as
  `get_k`. Switch: config `grad_rhf_jk_energy` (module attribute `rhf_grad.JK_ENERGY`).
- **Integral evaluations** (`grad_ejk_stats.py`, minao density, def2-SVP):

  | molecule | orbits that pass | s2kl evaluations | new | ratio | per orbit |
  |---|---|---|---|---|---|
  | benzene | 1.06e6 | 4.11e6 | 2.59e6 | 0.63 | 2.45 |
  | caffeine | 1.43e7 | 5.50e7 | 3.80e7 | 0.69 | 2.65 |
  | n-C20H42 | 8.21e7 | 3.09e8 | 2.25e8 | 0.73 | 2.74 |

- **Per call** (`grad_ejk_check.py`, 10 threads, minao density): benzene RHF 0.47 → 0.25 s, UHF 0.55 → 0.26 s;
  caffeine RHF 6.03 → 3.94 s, UHF 8.43 → 4.43 s.
- **After** (1 thread, caffeine): integral evaluation about 91%; building Gamma 2.9%, contracting 3.5%, the orbit
  logic and screening 2.2%.
- **Exactness.** Against the matrix path: 1e-14 relative per call (water to caffeine; RHF, UHF, J or K alone,
  range-separated and short-range K), and 1.2e-14 at most on whole gradients of RHF, UHF, ROHF, RKS (PBE, B3LYP
  with and without grid response, CAM-B3LYP, wB97X, r2SCAN), UKS (PBE, B3LYP, wB97X), ROKS (B3LYP, CAM-B3LYP)
  and atom subsets. Cartesian, generally contracted (cc-pVTZ) and ghost-atom bases agree to 3e-15; a lone atom
  now gives exactly 0. The DF gradient keeps the matrices and is unchanged bit for bit.
- **Tests.** `test_jk_energy_per_atom` and `test_grad_elec_jk_energy` in `grad/test/test_rhf.py`; the upstream
  run of the gate covers them with the grad, df, solvent and qmmm tests.

### End to end
It19 against It20: 10 threads, `PYSCF_MAX_MEMORY=20000`, means of 2 interleaved runs
(`results/20261030-165121_it20.json`; J/K of the It20 gradients is `jk_energy_per_atom`, which that run still
counted as "other").

| case | It19 (s) | It20 (s) | |
|---|---|---|---|
| gradient benzene/SVP B3LYP | 0.58 | 0.36 | 1.60× (J/K 0.47 → 0.29 s) |
| gradient caffeine/SVP B3LYP | 7.33 | 4.66 | 1.57× (J/K 6.85 → 4.21 s); It19's two runs were 6.89 and 7.76 s |
| gradient caffeine+/SVP B3LYP UKS (new) | 9.42 | 5.25 | 1.80× (J/K 8.51 → 4.38 s) |
| gradient n-C20H42/SVP B3LYP | 29.19 | 17.65 | 1.65× (J/K 27.68 → 16.33 s) |
| Hessian benzene/SVP B3LYP, PBE | 7.15, 6.40 | 7.14, 6.40 | unchanged |
| Hessian benzene+/SVP B3LYP UKS | 12.68 | 12.58 | −0.8% |
| Hessian caffeine/SVP B3LYP | 92.4 | 90.7 | −1.8% (derivative J/K 43.5 → 41.4 s) |
| Hessian caffeine+/SVP B3LYP UKS | 157.6 | 155.1 | −1.6% (derivative J/K 48.4 → 46.8 s) |

- **Where the time goes now:**
  - Gradients: J/K is still 90% (caffeine 4.2 of 4.7 s, n-C20H42 16.2 of 17.7 s), and integral evaluation is
    about 91% of it. Fewer evaluations per orbit are not possible this way; cheaper ones would need a qcint
    kernel that derives all centers from one set of Rys roots (estimated 15–20% of the integral time: the roots
    and the primitive loop are shared, the g arrays and contractions are not).
  - caffeine Hessian (90.7 s): derivative J/K 41.4 s, CPHF XC 26.6 s, CPHF J/K 8.5 s.
  - caffeine+ UKS Hessian (155.1 s): CPHF XC 61.5 s, derivative J/K 46.8 s, CPHF J/K 18.7 s.

### Gate
- Kernel tests pass against `pyscf-base`.
- Tight validation passes on the hessian suite (with the 4 new water gradients), the full suite (with
  `grad_caffeine+_svp_b3lyp_uks`) and the sparse suite. Largest differences from unmodified PySCF:

  | quantity | max difference | case |
  |---|---|---|
  | energies | 4.6e-12 Eh | |
  | gradients | 3.8e-12 | n-C20H42 (as in It19); caffeine 1.7e-12, caffeine+ UKS 1.6e-12, water 2.4e-14 |
  | excitations | 1.3e-12 | water+ UKS r2SCAN Casida |
  | Hessians | 6.6e-11 | water, DF; benzene+ UKS 2.8e-11 |

- Upstream tests (dft, scf, df, grad, hessian, tdscf, solvent, qmmm): 997 passed, 4 failed (the known xcfun
  failures).
- **Cycle-count warnings** on four tight SCFs, which It20 does not touch (the kernels it changed that the SCF
  could reach are bit for bit). The new one, caffeine PBE at 41 cycles against 39, was rerun twice in each tree:
  It20 40 and 39, It19 41 and 44, energies within 5e-12 Eh. This is the run-dependent endpoint of tight direct
  SCFs, noted since It7.

## It21: SCF initial guess and ADIIS+CDIIS, 2026-10-30

The user's direction after It20: fewer SCF cycles, from a better initial guess and a hybrid ADIIS/CDIIS. These
change the cycle counts by design, so the "same cycle counts" check does not apply here. Both are opt-in; the
defaults (`minao`, CDIIS with 8 vectors) are unchanged, so every bench and validation case is too.

`pyscf` `perf` commits:
- `0ddec1161`: `scf.diis.ADIIS_CDIIS`.
- `802f7e744`: `init_guess='pre_scf'`.
- Workspace: a test set of 28 molecules (`bench/geom/*.xyz`, RDKit ETKDG + MMFF94 from `bench/geom/make_geoms.py`,
  run in the tools venv `.venv-tools`), harder cases in `bench/geom/hard/` (pentacene, coronene, beta-carotene,
  high-spin [Fe(H2O)6]2+), bench knobs `BENCH_INIT_GUESS`, `BENCH_DIIS`, `BENCH_DIIS_SPACE`, `BENCH_PRE_SCF_TOL`,
  `BENCH_PRE_SCF_LEVEL`, and the study data in `results/it21-scf/`.

### The guesses PySCF has (`scf_guess_study.py`, B3LYP/def2-SVP, CDIIS, conv_tol 1e-9)
16 molecules with all six guesses (12–74 atoms, RKS, UKS radicals, a triplet, ions):

| guess | cycles against minao |
|---|---|
| `minao` (default) | best or tied on every molecule (9–16 cycles) |
| `atom` | +0 to +2 |
| `huckel`, `mod_huckel` | +1 to +8 |
| `sap`, `vsap` | +3 to +15 |

No built-in guess beats the default. The traces also show the step that every variant wastes: cycle 2 is the
plain Roothaan step from the first density (DIIS holds one Fock matrix then) and often raises the energy (aspirin
0.36 → 1.41 Eh above the final). Putting the guess's Fock matrix into DIIS (`diis_start_cycle=0`) did not help
CDIIS (122 against 121 cycles on 10 molecules) and hurt the hybrid (158–187).

### ADIIS+CDIIS (`scf.diis.ADIIS_CDIIS`)
- **Method.** ADIIS (Hu and Yang 2010) far from convergence, CDIIS close to it, switched by the largest element of
  the CDIIS error vector as in EDIIS+DIIS (Kudin, Scuseria and Cancès 2002): ADIIS above 1e-1, CDIIS below 1e-4,
  the blend w F_ADIIS + (1-w) F_CDIIS in between. Upstream `ADIIS` minimizes over all `space` slots of its buffer,
  so the empty ones (zero density and Fock matrices) take part in the first iterations; the hybrid keeps only the
  iterations seen. ROHF (total density, effective Fock matrix) gets CDIIS alone.
- **Organic molecules** (`scf_diis_variants.py`, 32 molecules, minao): total cycles

  | variant | cycles | |
  |---|---|---|
  | CDIIS, 8 vectors (default) | 384 | |
  | CDIIS, 12 vectors | 374 | −2.6% |
  | ADIIS+CDIIS, linear blend | 376 | −2.1%; caffeine+ UKS 27 against 23 |
  | ADIIS+CDIIS, 12 vectors | 373 | −2.9% |
  | ADIIS+CDIIS, log blend | 398 | +3.6% |
  | ADIIS+CDIIS, RMS switching (Psi4's measure) | as CDIIS | the ADIIS phase hardly starts |

  No variant failed. Where CDIIS behaves, the ADIIS phase is the slow part (caffeine+: CDIIS takes |g| from 3.5
  to 0.13 in 4 cycles, ADIIS needs 8 to get below 0.1).
- **Hard cases** (`results/it21-scf/hard_*.jsonl`; cycles, minao):

  | case | CDIIS | 12 vectors | ADIIS+CDIIS |
  |---|---|---|---|
  | [Fe(H2O)6]2+ quintet UKS PBE | 54, on a state 2.7e-4 Eh higher | 48 | 20 (lower state) |
  | same, UKS B3LYP | 64 | not converged in 100 | 45 |
  | same, UHF | 46, on a state 0.035 Eh higher | 66 | 88 (lowest) |
  | pentacene, coronene, beta-carotene (PBE), stretched water (RHF, PBE, B3LYP) | 10–14 | same | same or +1–2 |
  | caffeine+ UKS PBE | not converged in 100 | | not converged in 100 (see It22) |

  On the Fe complex the variants end on different SCF solutions: there, any change of guess or DIIS changes the
  result, which is why none of this is a default.

### The guess from a cheaper SCF (`init_guess='pre_scf'`)
- **Method.** A copy of the SCF object with density fitting for J and K (unless it fits already) and, for DFT,
  grids of level 1, converged from minao to 1e-6; its density starts the SCF, which runs unchanged (Psi4's
  DF_SCF_GUESS). The copy gets its own grids, DIIS and integral caches and no chkfile. A smaller basis
  projected (minao or STO-3G into def2-SVP: 1–3 cycles saved) and a PBE pre-SCF for hybrids (9–10 cycles left
  instead of 5) were tried and dropped.
- **Cycles of the SCF itself:** 4–8 instead of 9–23 everywhere (RKS, UKS, ROKS, RHF, DF), 11–12 instead of 54 on
  the Fe complex with PBE. Energies agree with the minao start to 2e-9 Eh at the default thresholds.
- **Wall time** (10 threads, 20 GB, `results/20261030-185741_it21-guess.json` and `-192547_it21-guess-tzvp.json`;
  the pre-SCF included):

  | case | minao | pre_scf | |
  |---|---|---|---|
  | caffeine def2-TZVP B3LYP | 73.0 s (14) | 37.7 s (5) | 1.94× |
  | n-C20H42 def2-TZVP PBE | 148.6 s (10) | 92.6 s (5) | 1.60× |
  | caffeine def2-TZVP DF-PBE | 8.61 s (16) | 6.53 s (5) | 1.32× |
  | benzene+ SVP r2SCAN UKS | 1.81 s (11) | 1.43 s (5) | 1.27× |
  | caffeine SVP PBE | 4.10 s (16) | 3.45 s (6) | 1.19× |
  | caffeine SVP r2SCAN | 6.13 s (14) | 5.44 s (7) | 1.13× |
  | caffeine+ SVP B3LYP UKS | 8.95 s (23) | 8.32 s (8) | 1.08× |
  | caffeine SVP CAM-B3LYP | 6.23 s (13) | 5.85 s (4) | 1.06× |
  | caffeine SVP B3LYP | 4.21 s (15) | 4.08 s (5) | 1.03× |
  | n-C20H42 SVP B3LYP | 13.91 s (9) | 17.98 s (4) | 0.77× |

  beta-carotene/SVP PBE (`hard_pbe.jsonl`): about 110 → 76 s. A pre-SCF tolerance of 1e-4 gives the same within
  noise. It pays when the SCF's own Fock builds are expensive (TZVP, direct J/K, large grids, meta-GGA, UKS) and
  loses when they are cheap and density-fitted K is not: n-C20H42/SVP B3LYP is semi-direct (the integrals are
  read back after the first cycle) and needs only 9 cycles, while RI-K over 490 AOs costs more per cycle.

## It22: why caffeine+ UKS PBE does not converge, 2026-10-31

The user's direction after It21: keep the new guess and DIIS opt-in, and find out why caffeine+ UKS PBE/def2-SVP
converges with nothing. This is a diagnosis; PySCF is unchanged. Scripts: `bench/diag/scf_hard_*.py`,
`scf_flipmix.py`, `scf_flip_scan.py`; data in `results/it22-scf/`.

- **Not a regression.** Unmodified master (`pyscf-base`) and the 2.12.1 wheel fail the same way as `pyscf`: 88
  cycles, the same final state, and the same failed extra cycle (`trace_PBE_cdiis_{perf,base,wheel}.log`).
- **Three SCF solutions** (`scf_hard_states.py`; the symmetric ones with Cs symmetry and fixed irrep occupations,
  in the same frame and on the same grids; stability in the non-symmetric UKS, so symmetry-breaking rotations
  count; `states_PBE.log`):

  | solution | E − E_BS | β orbitals around the hole | stable |
  |---|---|---|---|
  | hole spread over a π and a σ orbital, mirror symmetry broken | 0 (−678.7490982 Eh) | aufbau, gap 0.0025 Eh (0.07 eV) | yes |
  | hole in π (²A″) | +1.16 kcal/mol | the empty π orbital 0.0092 Eh *below* the highest occupied σ | yes |
  | hole in the σ lone pair (²A′) | +4.66 kcal/mol | the empty σ orbital 0.0175 Eh below the highest occupied π | yes |

- **What the SCF does** (`scf_hard_trace.py`: per cycle the frontier orbitals' mirror parity and the overlap of
  the occupied space with the previous cycle's):
  - The minao start puts the β hole in π. Both symmetric states violate aufbau, so each diagonalization moves the
    hole to the other orbital. The β occupied space swaps (overlap 0 with the previous cycle's) at cycles 7, 8, 9,
    11, 16, 20, ..., and the energy wanders between −678.74 and −678.66 Eh.
  - Fock matrices of mirror-symmetric densities, and their DIIS combinations, keep the orbitals symmetric. This
    goes on until numerical noise, amplified at the near-degeneracies (`HOMO >= LUMO` warnings), breaks the
    symmetry, around cycle 60–65. CDIIS then converges to the broken-symmetry minimum in about 20 cycles.
  - There it meets the loop's criteria (|g| 1–3e-5), but the extra cycle (a diagonalization without DIIS)
    multiplies |g| by 20–36 and raises E by 4e-9 to 5e-7 Eh. So the run is flagged not converged, also at
    conv_tol 1e-10 (|g| 2.4e-6 → 7.9e-5), although the last DIIS iterate is within 1e-9 Eh of the minimum.
  - The swapping phase is chaotic: 81–97 cycles between builds and runs. With PySCF's default `max_cycle=50` the
    run stops inside it, 0.059 Eh above the minimum.
- **B3LYP has no such problem** (`states_B3LYP.log`). Its π-hole state obeys aufbau (β gap 1.57 eV), is stable,
  and is where CDIIS goes in 23 cycles; the σ-hole state is 11.1 kcal/mol higher and unstable; there is no
  broken-symmetry minimum. The PBE picture is the GGA's delocalization error: an emptied orbital drops below the
  occupied ones, and spreading the hole over two orbitals lowers the energy.
- **Remedies, from minao** (`scf_hard_remedies.py`, `remedies_PBE.jsonl`; E above E_BS, max_cycle 100):

  | remedy | cycles | converged | ends on |
  |---|---|---|---|
  | CDIIS, `max_cycle=50` (PySCF's default) | 50 | no | mid-swap, +0.059 Eh |
  | CDIIS | 92 | no (extra cycle) | broken symmetry, +5e-7 Eh |
  | CDIIS, conv_tol 1e-10 | 91 | no (extra cycle) | broken symmetry, +4e-9 Eh |
  | ADIIS+CDIIS | 97 | yes, by chance: the extra cycle's ΔE 9.8e-9 is just under 1e-8 | broken symmetry |
  | damping 0.5 | 86 | no | broken symmetry |
  | level shift 0.1 / 0.25 Eh | 21 / 42 | no: the loop converges on the σ-hole state, the extra cycle (no shift, aufbau) moves the hole | +0.052 Eh |
  | level shift 0.5 Eh | 100 | no | π hole |
  | SOSCF (`mf.newton()`) | 5 macro, 67 J/K | yes | π hole, +1.16 kcal/mol |
  | 50 CDIIS cycles, then SOSCF | 7 macro, 157 J/K | yes | σ hole, +4.66 kcal/mol |
  | Fermi smearing 0.005 / 0.01 Eh, then integer occupations | 32+100 / 28+97 | no | swapping / broken symmetry |
  | the B3LYP density as guess | 23+100 | no | broken symmetry |
  | `init_guess='pre_scf'` | 28 | no | broken symmetry |

- **Breaking the symmetry on purpose** (`scf_flipmix.py`, `flipmix_PBE.jsonl`):
  - From the converged π-hole state, rotating the π hole into the highest occupied σ orbital by 20° or 45°:
    CDIIS reaches the broken-symmetry minimum in 19 or 27 cycles (the extra cycle still fails).
  - Generic version, "flip-pair mixing". Per spin, watch the occupied space. When an orbital swaps out (smallest
    singular value of the overlap with the previous cycle's below 0.1, from cycle 4) and the same orbital later
    swaps back in (overlap above 0.9), replace the orbital that came in by the normalized sum of it and the one
    that left, then restart CDIIS from that density.
    - It fires at cycle 8 here, on the β π/σ pair (mirror parity of the mixed orbital −0.04).
    - CDIIS then needs 26 cycles: 34 in all and 11.8 s, against 92 cycles and 28.8 s. ΔE is 1e-8 and the extra
      cycle fails. With SOSCF after the mixing it takes 6 macro iterations and ΔE is 1.6e-10.
    - A looser trigger (from cycle 3, overlap below 0.5) fired at cycle 5 on two σ orbitals during the initial
      reordering. The mix then kept the symmetry, and the run failed.
  - Trigger scan (`scf_flip_scan.py`, `flip_scan.log`; 8 open-shell molecules, PBE and B3LYP, plain CDIIS):
    - It fires on four runs:
      - caffeine+ PBE (cycle 8);
      - benzene+ PBE (cycle 5; 21 cycles against B3LYP's 9; the hole sits in a degenerate pair);
      - [Fe(H2O)6]2+ PBE (cycle 5; 54 cycles);
      - [Fe(H2O)6]2+ B3LYP (cycle 7; 64 cycles).
    - In the other 12 runs no orbital swaps at all.
    - Whether mixing helps benzene+ and the Fe complex, and which Fe state it ends on, is untested.
- **What works now, without code changes:**
  - **The lowest solution:** `max_cycle=100` with `conv_check=False`, which accepts the DIIS iterate. Or polish
    the end of a 100-cycle run with SOSCF (12.8 s).
  - **The symmetric π-hole state:** `mol.symmetry=True` with `mf.irrep_nelec = {"A'": (41, 41), 'A"': (10, 9)}`
    (25 cycles), or `mf.newton()` from minao.

## node1: GPU4PySCF on the GTX 1650 Super (setup and smoke test), 2026-10-31
- **Machine.** `ssh node1` reaches `danny-desktop`:
  - GPU: GeForce GTX 1650 SUPER (TU116, compute capability 7.5, 4 GB, 20 SMs), driver 595.99.02 (CUDA 13.2).
  - CPU: i7-11700K (8 cores, AVX-512), 31 GB, Ubuntu 24.04.5. No CUDA toolkit.
- **What the card is for.**
  - GPU4PySCF needs compute capability 7.0 or newer.
  - The card's FP64 peak (~138 GFLOPS, 1/32 of FP32) is below the laptop CPU's DGEMM. It is a development and
    verification card, not a fast one.
  - FP64 kernel tuning needs a final run on an A100 or H100, because `gpu4pyscf/__config__.py` picks other
    settings on small cards: blocks of 64 AOs instead of 256 under 16 GB, and shared memory sized from the device
    (48 KB per block here).
- **Setup.**
  - Scripts:
    - `scripts/node1_gpu_venv.sh` creates the venv; pins are in `scripts/requirements-gpu.txt`.
    - `scripts/node1-gpu-env.sh` is sourced before runs.
  - node1 has the same workspace path, `~/Chemistry/my-pyscf`, holding only the files the GPU runs need. The venv
    is `.venv-gpu` (system Python 3.12.3, venv + pip, 4.3 GB).
  - Pins:
    - upstream's `requirements.txt`, the configuration of its nightly benchmarks: CuPy 13.4.1, cuTENSOR 2.2.0,
      PySCF 2.14.0, gpu4pyscf-libxc 0.8.1;
    - gpu4pyscf-cuda12x 1.9.0;
    - NVIDIA's CUDA 12.9 pip libraries.

    The full list is in `bench/results/node1/pip-freeze.txt`.
  - **Two gotchas without a CUDA toolkit:**
    - CuPy 13.4.1 does not look inside NVIDIA's pip packages, and fails with `libnvrtc.so.12: cannot open shared
      object file`. `node1-gpu-env.sh` puts their `lib` directories, and cuTENSOR's, on `LD_LIBRARY_PATH`.
    - GPU4PySCF loads cuSOLVER with `ctypes.util.find_library('cusolver')`, falling back to
      `numpy.ctypeslib.load_library('libcusolver', .../nvidia/cusolver/lib)`. Both need an unversioned
      `libcusolver.so`, which the pip package does not ship (only `libcusolver.so.11`). The setup script adds
      the symlink inside the venv.
  - **The desktop session runs on the card.** Xorg and GNOME use ~65 MiB, and CUDA reports the kernel watchdog
    as enabled (`kernelExecTimeoutEnabled`). A kernel running longer than a few seconds would be killed; the smoke
    test was not affected.
- **Smoke test** (`bench/gpu/smoke.py`; `bench/results/node1/smoke.jsonl` and the three logs).
  Caffeine B3LYP/def2-SVP (246 AOs), conv_tol 1e-10, PySCF on 8 CPU threads against GPU4PySCF:

  | run | E (Eh) | cycles | grid points | wall |
  |---|---|---|---|---|
  | PySCF 2.14.0 wheel, CPU, 8 threads | −679.8127486398 | 15 | 294 800 | 45.3 s |
  | GPU4PySCF 1.9.0, first run | −679.8127486398 | 15 | 294 912 | 58.0 s |
  | GPU4PySCF 1.9.0, second run | −679.8127486398 | 15 | 294 912 | 57.6 s |

  - **Agreement.** GPU − CPU is −1.6e-11 and −1.7e-11 Eh, with the same cycle count. Water B3LYP/def2-SVP
    agrees to 7.3e-12 Eh.
  - **Grid points.** The 112 extra points are GPU4PySCF's padding of the grid to a multiple of 256
    (`grid_aligned`).
  - **Start-up cost is small.** The two GPU runs differ by 0.4 s.
  - **The GPU is 1.27× slower than the CPU here,** as the FP64 rates predict. Both run direct SCF: in-memory
    ERIs for 246 AOs need 3.7 GB, more than PySCF's 4000 MB default leaves (corrected: first written up as an
    in-memory CPU run; the profile below shows forced-direct and default CPU runs timing the same).
- **SCF profile per part.**
  - Scripts: `bench/gpu/scf_parts.py`, `parse_gpu_log.py`, `jk_variants.py`.
  - Results in `bench/results/node1/`: `scf_parts.jsonl`, `scf_parts_v6.txt`, `scf_parts_v4_and_jk.txt`,
    `gpu_timers_v6.txt`, `parts_logs/`.
  - **Method.**
    - Each SCF method is wrapped and timed exclusively, between device synchronizations: guess, grids, XC
      (`nr_rks`), J, K, Fock+DIIS, eig, density, energy, gradient. Untimed time is ≤ 0.004 s per cycle.
    - GPU runs are profiled on their second SCF. The headline GPU numbers come from verbose 4.
    - CPU numbers are PySCF 2.14.0's wheel on 8 threads, not our perf tree.
    - GPU4PySCF's own timers (verbose 6) split XC and give J/K per angular-momentum class.
      - Use their GPU-event times. Each timer reads the host clock before it synchronizes, so a host wall time
        also contains the previous label's GPU work.
      - They synchronize after every J/K kernel launch, which slows B3LYP's K by 10% (2.54 against 2.31 s per
        cycle).
    - The forced-direct CPU runs (`cpu-direct`) time the same as the default ones. At 246 AOs PySCF is already
      direct.
  - **Per SCF cycle, caffeine/def2-SVP** (s, steady state; first calls excluded). GPU and CPU energies agree to
    ≤ 1e-10 Eh in every case, with the same cycle counts.

    | case | device | XC | J | K | eig, DIIS, rest | SCF wall (cycles) |
    |---|---|---|---|---|---|---|
    | PBE | GPU | 0.63 | 0.68 | – | 0.02 | 25.1 s (18) |
    | PBE | CPU | 1.34 | 1.24 | – | 0.01 | 50.1 s (18) |
    | B3LYP | GPU | 0.63 | 0.69 | 2.31 | 0.02 | 57.6 s (15) |
    | B3LYP | CPU | 1.36 | 1.41 (J+K) | | 0.01 | 45.3 s (15) |
    | r2SCAN | GPU | 1.47 | 0.69 | – | 0.02 | 37.0 s (16) |
    | r2SCAN | CPU | 2.69 | 1.25 | – | 0.01 | 67.9 s (16) |
    | DF-B3LYP | GPU | 0.63 | 0.17 (DF J+K) | | 0.02 | 14.5 s (15) |
    | DF-B3LYP | CPU | 1.36 | 0.19 (DF J+K) | | 0.01 | 26.0 s (15) |

    - The GPU is 2.0× faster than the CPU for PBE, 1.8× for r2SCAN and DF-B3LYP, and 0.79× for B3LYP.
    - K is 60% of a B3LYP cycle on the GPU. On the CPU, K adds only 0.17 s to J, because PySCF builds both in
      one integral pass.
  - **Inside XC** (GPU-event time per call):

    | functional | density on the grid | functional (libxc) | Vxc integration |
    |---|---|---|---|
    | PBE, B3LYP | 0.31 s | 0.006–0.008 s | 0.31 s |
    | r2SCAN | 0.32 s | 0.012 s | 1.15 s |

    - The density and the integration are each one nao×nao contraction over the grid in FP64. If dense, that is
      2·246²·294 912 = 3.6e10 FLOP, i.e. ~115 GFLOP/s, 70–80% of the card's FP64 peak (138–169 GFLOPS, base boost
      to max clock).
    - r2SCAN's integration adds the three τ contractions: 4 × 3.6e10 FLOP at ~125 GFLOP/s.
    - On this card XC only gets faster by doing fewer FLOPs.
  - **Inside J** (MD J-engine): 21 classes, none dominant. The largest are (ps|ss) 13%, (pp|ps) 10%, (dp|ss) 10%
    and (ds|ps) 10%.
  - **Inside K** (Rys, B3LYP): 25 classes, 4256 kernel launches over the SCF.
    - Classes with only s and p shells take 72% of K: (ps|ss) 23%, (ss|ss) 17%, (ps|ps) 14%, (pp|ss) 8%.
      Classes with one d shell take 20%, with two or more 8%.
    - These quartets carry the least quadrature arithmetic. Their cost is probably per-quartet work (Boys function
      in FP64, screening, accumulation). Only a kernel-level profile can say which.
  - **J and K in one pass** (`jk_variants.py`; converged B3LYP density, median of 3):

    | build | time |
    |---|---|
    | J-engine (`get_j`) | 0.78 s |
    | Rys K (`get_k`) | 2.62 s |
    | both, as RKS does now | 3.39 s |
    | Rys J+K in one pass (`get_jk`) | 2.63 s |

    - The one-pass result agrees to 7.7e-12 (J) and 4.1e-13 (K).
    - J inside the K pass costs ~0.02 s, so dropping the separate J-engine pass for hybrids would save ~0.7 s per
      cycle, ~18% of the B3LYP SCF on this card (estimate).
    - GPU4PySCF presumably chose the J-engine on faster cards; an A100/H100 has to confirm the trade.
  - **One-time costs** (first calls, GPU): XC 0.80 s, K 1.67 s, J 0.31 s (screening setup included), grids
    0.07 s, guess 0.03 s.
- **Next candidates (not started):**
  - J from the K pass for hybrids: done as an opt-in in It23 (below). A default needs A100/H100 numbers first.
  - A kernel-level profile of K's s/p classes (Nsight Compute; needs a user-local CUDA toolkit).
  - Our perf PySCF built on node1, for a CPU baseline that isn't the stock wheel.

## Why GPU4PySCF builds J and K in separate passes (node1), 2026-10-31
Question: HF and hybrid DFT on the GPU build J with the MD J-engine and K with a K-only Rys pass. A single Rys pass
produces both, and on the 1650 Super it is ~20% cheaper (profile above). Why the split?

- **History.** Sources: a `gpu4pyscf` clone at master c4ce229 (gitignored, read-only so far) and the PRs via the
  GitHub API.
  - **v1.0 (2024, arXiv 2407.09700):** one Rys kernel builds J and K together.
  - **#225 (Nov 2024):** the MD J-engine is added. The PR has no description.
  - **v1.4.1 (May 2025):**
    - The low-memory drivers for large systems (`hf_lowmem.py`, `rks_lowmem.py`) build J with the J-engine and K
      with a K-only Rys call. The one-pass call is left as a comment: `#:vj, vk = vhfopt.get_jk(dm, hermi, True,
      True, log)`.
    - CHANGELOG: "The MD J-engine is utilized by default for large system HF and DFT computation."
    - The regular RKS driver still calls `get_jk` for hybrids.
  - **#442 (Jul 2025, v1.4.2):** "In the direct SCF driver, the MD-J is set as the default function for Coulomb
    matrices."
    - `get_veff` of RKS, UKS and HF now always builds J with `get_j` and adds K from `get_k`.
    - The PR lists the multi-density MD-J kernels and the split into separate direct and DF drivers. It gives no
      benchmark for hybrids.
  - **#773 (May 2026):** K kernel optimizations (density contraction, general contractions, diffuse orbitals).
    Master on 2026-10-31 is unchanged in this respect.
  - **No measurement on record.** The repository's benchmark tables are from v1.0 (A100), before the switch.
    Nothing measures the split for small or medium hybrids.
- **What the split buys, in the code:**
  - **Screening** (`gpu4pyscf/lib/gvhf-rys/create_tasks.cu`).
    - The K-only filler (`_fill_vk_tasks`) keeps a quartet (ij|kl) when q_ij + q_kl plus the largest
      log|D| over the exchange-type blocks (ik, jk, il, jl) passes the cutoff.
    - The J+K filler (`_fill_vjk_tasks`) also accepts the Coulomb-type blocks (ij, kl). Those stay large for any
      significant pair, however far apart bra and ket are.
    - So one pass evaluates every quartet J needs (O(N²)), while K alone tends to O(N) in large insulators.
    - The `s_cond` and diffuse-exponent screening seen in `get_k` belongs to the short-range (erfc) fillers only.
  - **A faster J.** On this card the MD J-engine is 3.8–4.9× faster than the Rys J-only kernel (below). This
    matters for pure functionals, where J is the only two-electron term.
  - **One code path.** J is built the same way for HF, pure and hybrid DFT, range-separated Coulomb and several
    densities (UKS builds J of the total density once). Range-separated K needs its own SR/LR passes anyway.
  - So #442 generalized the large-system choice of the low-memory drivers.
- **Measured on the 1650 Super.**
  - Script: `bench/gpu/jk_scaling.py`. Results: `bench/results/node1/jk_scaling.{jsonl,txt}`.
  - Converged PBE densities, all-trans chains; median of 3 calls after one untimed call; times in s.

  | system | basis | nao | J-engine | Rys J | Rys K | Rys J+K | J inside K | one pass saves |
  |---|---|---|---|---|---|---|---|---|
  | caffeine | def2-SVP | 246 | 0.78 | 2.96 | 2.61 | 2.63 | 0.02 s (0.8% of K) | 22% |
  | C10H22 | def2-SVP | 250 | 0.61 | 2.42 | 2.46 | 2.48 | 0.02 s (0.6%) | 19% |
  | C20H42 | def2-SVP | 490 | 1.71 | 8.14 | 6.35 | 6.52 | 0.17 s (2.6%) | 19% |
  | C30H62 | def2-SVP | 730 | 3.46 | 17.27 | 10.85 | 11.71 | 0.87 s (8.0%) | 18% |
  | C40H82 | def2-SVP | 970 | 5.91 | 28.97 | 16.22 | 18.21 | 1.98 s (12.2%) | 18% |
  | caffeine | def2-TZVP | 494 | 5.41 | 23.13 | 17.93 | 18.55 | 0.62 s (3.5%) | 21% |

  - **The screening effect is real but small here.** J inside the K pass grows from 1% of K (caffeine, C10) to 12%
    (C40H82, a 50 Å chain). That is 2.5% to 34% of the J-engine's own cost, so one pass wins at every size tested.
  - **Incremental densities agree.** On the B3LYP SCF's density differences (caffeine, cycles 2, 6, 10), J inside
    K costs 0.024–0.029 s against 0.65–0.77 s for the J-engine; one pass saves 22–23%.
  - **Accuracy.** All J matrices agree to ≤ 2e-11 and all K matrices to ≤ 1.3e-12.
  - **Where the crossover might be.** A chain is the worst case for one pass at a given size, since it has the
    most far-apart bra–ket pairs; globular molecules should cross over later. From C30 to C40 the ratio rises
    9 points; at that rate the crossover is near ~2700 basis functions (rough extrapolation).
- **Not knowable on this card.**
  - The developers' hardware (A100/H100) has ~60× this card's FP64 rate but only ~8–10× its memory bandwidth.
  - The FP64-heavy MD J-engine may gain more there than the J accumulation added to the K kernel (atomics, memory
    traffic). The split could then be faster on those cards even for small molecules.
  - This is an untested hypothesis; `jk_scaling.py` measures it in ~30 minutes of A100/H100 time.
- **Conclusion.**
  - The split is deliberate, not an oversight: #442 made the large-system driver's choice the default, together
    with a uniform J path.
  - Its benefit, exchange screening, only shows in large, extended systems. On this card one pass is faster
    everywhere up to 970 AOs, and nothing records the trade being measured for small and medium hybrids.
  - A size- or hardware-aware choice between the two is an exact candidate, but only after A100/H100 numbers.

## It23: opt-in one-pass J+K in GPU4PySCF (`SCF.jk_one_pass`), 2026-10-31
- **Change.** gpu4pyscf `perf` c085eaf, on tag v1.9.0 (2fdc195); Python only.
  - `SCF.jk_one_pass` defaults to False. It can also be set with `scf_hf_SCF_jk_one_pass` in a PySCF config
    file, e.g. `bench/gpu/conf_jk_one_pass.py`.
  - When on, `get_veff` of RHF, UHF (and ROHF), RKS and UKS (and ROKS) builds J and K with one Rys `get_jk`
    pass, instead of the MD J-engine plus the K-only Rys pass.
  - **DFT:** global hybrids only (omega == 0).
    - The one-pass K is scaled by hyb. `get_k(omega=0, hyb, hyb)` applies the same factor in its kernel; they
      agree to 1e-15.
    - Range-separated hybrids keep their builders. With omega ≠ 0, `jk.get_jk` runs the K-only pass plus the
      slow Rys J-only pass, so it offers no single pass.
  - **UKS/UHF:** one pass over both spin densities, with J = J(α) + J(β).
  - **Unaffected:** density fitting and the low-memory drivers, which have their own `get_veff`.
  - **Flag off:** the same calls run in the same order as before.
  - **Test:** `gpu4pyscf/scf/tests/test_jk_one_pass.py` runs 8 methods with the flag off and on (|ΔE| < 1e-9),
    plus the K scaling against `get_k`.
- **Running the perf tree on node1** (no CUDA toolkit there).
  - The wheel's 304 `.py` files are byte-identical to tag v1.9.0, so the unmodified wheel is the reference build.
  - `scripts/node1_gpu4pyscf_tree.sh` copies the tree and links the wheel's 11 compiled libraries and libxc into
    it. It refuses if anything but `.py` files differs from v1.9.0, since the binaries would then not match.
  - `source scripts/node1-gpu-env.sh perf` puts the tree first on PYTHONPATH.
- **Gate.** Script: `bench/gpu/one_pass_gate.py`; results: `bench/results/node1/one_pass_gate.{jsonl,txt}`.
  - Caffeine/def2-SVP, conv_tol 1e-10, flag off and on interleaved; median of 2 for the first three cases, 1 run
    for the others.

  | case | ΔE (on − off) | cycles | off | on | speed-up |
  |---|---|---|---|---|---|
  | RKS B3LYP | −5.2e-12 | 15 / 15 | 57.66 s | 47.36 s | 1.22× |
  | UKS B3LYP (caffeine+) | −7.3e-12 | 26 / 26 | 108.68 s | 91.75 s | 1.18× |
  | RHF | −1.3e-10 | 14 / 14 | 43.41 s | 34.33 s | 1.27× |
  | RKS PBE0 | −3.1e-11 | 15 / 15 | 57.56 s | 47.21 s | 1.22× |
  | UHF (caffeine+) | −6.4e-11 | 31 / 31 | 93.52 s | 74.36 s | 1.26× |
  | RKS wB97X (flag does not apply) | −4.5e-13 | 14 / 14 | 78.82 s | 78.80 s | 1.00× |
  | RKS PBE (flag does not apply) | −4.5e-13 | 18 / 18 | 25.12 s | 25.13 s | 1.00× |

  - **Default path.** The perf tree with the flag off differs from the wheel by 0 to 6.8e-13 Eh. That is the
    run-to-run spread of the GPU kernels (≤ 7e-13 between repeats of one build), so the default is unchanged.
    The wheel takes the same times (57.78, 108.68, 43.39, 93.51 s).
  - **One pass.** Energies shift by 5e-12 to 1.3e-10 Eh, reproducibly, with the same cycle counts. The speed-up
    is larger than the estimate from the J-engine's time alone (~17%).
- **Upstream tests** (`bench/results/node1/upstream_tests_jk_one_pass.txt`).
  - Files: `scf/tests/test_rhf.py`, `test_uhf.py`, `dft/tests/test_rks.py`, `test_uks.py`, plus the new file.
  - With the flag forced on through the config file: 66 passed, 4 skipped (376 s).
  - With it off: the same 66 passed, 4 skipped (442 s).
- **Not covered:**
  - TDDFT/CPHF response, gradients and Hessians, which build their J/K themselves;
  - GHF and range-separated hybrids;
  - data-center GPUs. On those the J-engine may be the faster choice (see the J/K investigation), which is why
    this stays opt-in.
- **Usage:** `mf = gpu4pyscf.dft.RKS(mol, xc='b3lyp'); mf.jk_one_pass = True`.

## GPU4PySCF investigation, part 1: toolchain, kernel counters, and the K kernels (node1), 2026-10-31
Goal set by the user: a deep investigation of GPU4PySCF for improvements that carry over to all GPUs, not only the
1650 Super. The method is to count work that does not depend on the card (FP64 operations, instructions, atomics,
integrals) next to the times.

### Setup
- **CUDA 12.9.1 toolchain** in `~/opt/cuda-12.9.1` on node1 (`scripts/node1_cuda_toolkit.sh`).
  - It comes from NVIDIA's redistributable archives, checked against their sha256 sums: nvcc, cudart, cccl,
    cuobjdump, nvdisasm, compute-sanitizer, CUPTI and Nsight Compute 2025.2.1, 1.7 GB in all.
  - It needs a `lib64 → lib` link for nvcc.
- **GPU performance counters.** The user set `NVreg_RestrictProfilingToAdminUsers=0` in
  `/etc/modprobe.d/nvidia-profiling.conf` and rebooted node1. Without it Nsight Compute stops with
  ERR_NVGPUCTRPERM.
- **Source builds.** `scripts/node1_gpu4pyscf_tree.sh [gpu4pyscf|gpu4pyscf-base]`:
  - rsyncs a tree and builds all 12 CUDA libraries for sm_75 in 3 minutes (`BUILD_LIBXC=OFF`; libxc is linked
    from the wheel);
  - removes wheel links before a build, so nothing is written through them;
  - `gpu4pyscf-base` is a laptop worktree at v1.9.0; `source scripts/node1-gpu-env.sh perf|base` selects a tree.
- **The wheel** ships SASS for sm_70/80/90 only, without PTX for the integral libraries, so the 1650 Super runs the
  Volta code.
  - The base source build gives the same energies (≤ 2.3e-13) and the same times as the wheel at equal clocks
    (25.31 against 25.32 s, PBE; `bench/results/node1/ab_clock.jsonl`).
- **Clocks vary between sessions.** Right after the reboot the GPU boosted, and the wheel ran 20% faster than in all
  other runs, which held 1530 MHz in P2.
  - `bench/gpu/gpu_monitor.py` now records the SM and memory clocks, the P-state and throttle reasons with every
    timing in `one_pass_gate.py`. Only runs at equal clocks are compared.
  - Nsight Compute locks the base clock itself.

### K-build kernel counters (`bench/gpu/ncu_parts.py`, `ncu_summary.py`; `bench/results/node1/ncu/`)
- **How the profile is taken.** One K build at the converged caffeine B3LYP density.
  - Every `RYS_build_k` call is labelled with its shell-group class, in launch order. Each call launches one
    kernel; there are 266 per build.
- **Totals.** 2.635 s GPU, 182 GFLOP FP64 (69 GFLOP/s, 57% of the 122 GFLOPS peak at 1530 MHz), 11.6 G warp
  instructions, 12.3 M atomic instructions and 0.13 GB DRAM traffic.
- **The low-angular-momentum classes dominate.**
  - (ps|ss) 24% of K time (44 GFLOP), (ss|ss) 18% (34 GFLOP), (ps|ps) 14%.
  - The FP64 pipe is 80–89% busy in all classes, so on this card K is FP64-bound.
  - SIMT efficiency is 0.86–0.96, occupancy 25–50%, registers 88–255.
  - FP64 is only a third to half of the instructions. On FP64-strong cards the other two thirds and the occupancy
    limit would set the time instead.
- **The six shell groups for K** (`sorted_mol`): s with 5 primitives (14 shells, the C/N/O cores), s3 (10), s1
  (38), p3 (14), p1 (24), d1 (14). Contractions are not split; Schwarz and overlap screening act on whole
  contracted pairs only.

### What the K kernels do (`gpu4pyscf/lib/gvhf-rys/unrolled_rys_k.cu`; generator not in the repository)
- **No primitive screening.** Every primitive quartet gets a Boys-function evaluation. Between neighbouring atoms,
  tight core primitive pairs have exp(−θR²) ≈ exp(−4000) = 0.0. CPU PySCF (libcint) skips primitive pairs below
  e^−60.
  - Census (`bench/gpu/prim_census.py`): libcint's cutoff keeps 71% of the primitive pairs in (s5,s5) shell pairs,
    56% in (s3,s5) and 67% in (p3,s5). It keeps all of them in pairs involving an uncontracted shell.
- **A barrier per primitive pair that protects nothing.** In 19 of the 25 unrolled kernels one thread computes one
  quartet (`rys_roots_for_k` with stride 1), and there each thread uses only its own shared-memory column. Yet the
  bra-primitive loop starts with `__syncthreads()`.
- **Padding threads do full work.** Threads with `task_id ≥ ntasks` run the complete primitive loops with a zero
  factor, so that the barriers stay aligned. A bra pair often has far fewer ket pairs than the block's 256 threads.

### Prototype (gpu4pyscf `exp/prim-screen` e39fc09, on perf; generated by `bench/gpu/exp_prim_screen.py`)
- **Variants**, one K build at the same density, median of 5 (`prim_screen_variants.txt`):

  | variant | K build | vs base | max \|ΔK\| |
  |---|---|---|---|
  | base (v1.9.0) | 2.616 s | | |
  | barrier removed | 1.838 s | −30% | 2.2e-15 |
  | + skip exactly-zero ket/bra primitive pairs (padding, underflow): bit-exact | 1.278 s | −51% | 1.8e-15 |
  | + ket pairs below e^−60 | 1.247 s | −52% | 1.8e-15 |
  | + bra pairs below e^−60 (committed) | 1.191 s | −54% | 2.3e-15 |

- **Counters** (`summary_k_exp.txt`): FP64 182 → 62 GFLOP (−66%), warp instructions 11.6 → 4.4 G (−62%).
  - (ss|ss) 34 → 3.5 GFLOP, (ps|ss) 44 → 7.7 GFLOP.
  - Registers rose (88 → 136 in (ss|ss)), so occupancy fell to 25%.
  - These instruction and flop reductions do not depend on the card.
- **SCF**, caffeine/def2-SVP, interleaved, all at 1530 MHz (`prim_screen_scf.jsonl`):

  | case | base | prototype | speed-up | ΔE |
  |---|---|---|---|---|
  | RKS B3LYP | 57.90 s | 37.64 s | 1.54× | ≤ 4.5e-13 |
  | RHF | 43.45 s | 24.88 s | 1.75× | ≤ 4.5e-13 |

  - Cycle counts are unchanged.
  - `jk_one_pass` (unpatched J+K kernels) is now slower than the default: 47.5 and 34.4 s.

### Next
- **J+K kernels.** All 25 bra-primitive loops in `unrolled_rys_jk.cu` start with the same barrier. Patching them
  should make the one-pass J+K the fastest path for hybrids.
- **The remaining K kernels.** The 6 multi-thread unrolled kernels and the generic `rys_k_kernel` also do padding
  work.
- **Occupancy** lost to the higher register counts.
- **Other kernel families.** The gradient kernels (`unrolled_rys_jk_ip1.cu`, `unrolled_ejk_ip*.cu`) have no
  inner-loop barrier, but primitive screening and padding work there are unchecked. The MD J-engine and the XC
  kernels are still to analyze.
- **Master.** It changed 27 files in gvhf-rys/gvhf-md since v1.9.0. The patch is script-generated, so it can be
  reapplied there.

## GPU4PySCF investigation, part 2: the one-pass J+K kernels and the remaining K kernels (node1), 2026-10-31
Asked by the user: apply part 1's fix to the one-pass J+K kernels and to the K kernels it left out.

### What each kernel family allows (`gpu4pyscf/lib/gvhf-rys`)
- **One-pass J+K, 19 one-thread kernels** (`unrolled_rys_jk.cu`). They have the same structure as the 19 K kernels
  of part 1: one thread per quartet, `rys_roots_for_k` with stride 1, and shared memory used only for the
  read-only bra cache and each thread's own root column (checked for every kernel). So all three changes apply:
  no barrier, and zero or below-e^−60 ket and bra primitive pairs skipped.
- **Multi-thread unrolled kernels, 6 for K and 6 for J+K** (2021, 2111, 2120, 2210, 3011, 3110). Four threads work
  on one quartet. With `t_id = 64·gout_id + sq_id`, those four threads sit in four different warps and exchange
  the g arrays, Rpq and the roots through shared memory, so the barriers are needed.
  - Per-quartet skips would break the barrier count.
  - A bra skip is safe: all threads of a block hold the same bra pair. It goes before the barrier, so the whole
    block skips together.
- **The generic kernels** (`rys_k_kernel`, `rys_jk_kernel`) use the same cooperative pattern with any block
  shape. They get the same uniform bra skip.
- **`PRIM_CUTOFF`** moves to `vhf.cuh` (e^−60 = 8.8e−27, libcint's expcutoff).
  - A skipped primitive pair enters its integrals with a factor below that, many orders of magnitude under the
    SCF's 1e−13 shell-quartet cutoff.
  - libcint's own test also weighs an estimate of the integral's size (`CINTset_pairdata`). It drops less than
    this test for diffuse pairs, and more for tight ones.

### How much is left in the kernels the patch cannot change much (caffeine/def2-SVP, converged B3LYP density)
- **Primitive pairs** (`prim_census.py`): pairs with the single-primitive d shells keep 89–100% of their
  primitive pairs (d1,s5 100%; d1,s3 89%; d1,p3 98%; d1,d1 100%). For def2-SVP the multi-thread and generic kernels
  all have a d shell, so a primitive skip saves little there. It matters for bases with contracted d or f shells.
- **Padding** (`bench/gpu/task_census.py`, a host replay of `_fill_vk_tasks` for every shell-group quartet).
  - A block takes one bra pair and runs its ket pairs nsq_per_block at a time.

  | kernels | quartets per block | ket pairs per bra pair | padded slots | warps without a real quartet |
  |---|---|---|---|---|
  | one-thread unrolled | 256 | 86–296 | 44% | 40% |
  | multi-thread unrolled | 64 × 4 threads | 95–287 | 18.5% | 10% |
  | generic | 16–48 × 4–16 threads | 26–186 | 20% | 3% |

  - In the one-thread kernels the padding threads now skip all their work (part 1). In the others only whole idle
    warps could be saved, about 10% of the multi-thread kernels' 0.14 s, or 1% of the K build.
  - That would need named barriers over the active warps. It is not worth the change for now.

### Results (gpu4pyscf `exp/prim-screen` e52d6ef; all runs at 1530 MHz SM / 5750 MHz memory, P2)
- **One build at the converged B3LYP density** (`jk_at_dm.py`, median of 5; `prim_screen2_fixed_dm.txt`):

  | build | v1.9.0 | patched | max \|Δ\| vs v1.9.0 | v1.9.0 against itself |
  |---|---|---|---|---|
  | one-pass J+K (`get_jk`) | 2.637 s | 1.234 s (2.14×) | 3.2e−14 | 2.1e−14 |
  | K only (`get_k`) | 2.620 s | 1.189 s (2.20×) | 1.8e−15 | 1.4e−15 |

  One-pass J+K now costs only 4% more than K alone.
- **SCF**, caffeine/def2-SVP, conv_tol 1e−10 (`one_pass_gate.py`; `prim_screen2_scf.jsonl`):

  | case | v1.9.0 | patched, one-pass off | patched, one-pass on | \|ΔE\| off / on | cycles |
  |---|---|---|---|---|---|
  | RHF | 43.49 s | 24.89 s (1.75×) | 15.83 s (2.75×) | 6.8e−13 / 1.3e−10 | 14 |
  | RKS B3LYP | 57.76 s | 37.50 s (1.54×) | 27.45 s (2.10×) | 1.4e−12 / 5.5e−12 | 15 |
  | RKS PBE0 | 57.53 s | 37.40 s (1.54×) | 27.38 s (2.10×) | 4.5e−13 / 3.1e−11 | 15 |
  | UKS B3LYP (caffeine⁺) | 109.14 s | 75.42 s (1.45×) | 58.92 s (1.85×) | 2.3e−13 / 7.0e−12 | 26 |
  | RKS ωB97X (range-separated K) | 78.96 s | 44.02 s (1.79×) | (no one-pass) | 2.3e−13 | 14 |
  | RKS HSE06 (short-range K only) | 79.80 s | 43.63 s (1.83×) | (no one-pass) | 1.1e−12 | 15 |

  - With one-pass off, energies agree with v1.9.0 to 1.4e−12 Eh, the run-to-run spread of the atomics.
  - One-pass on and off differ by up to 1.3e−10 (RHF). It23 measured the same 1.3e−10 with the unpatched
    kernels: it comes from where the SCF stops at conv_tol 1e−10.
  - Cycle counts are the same in all cases.
  - The PBE0 one-pass-off run overlapped a short census run on the same GPU. Its time matches B3LYP's.
- **Counters** (Nsight Compute, one build at the converged density; `ncu/summary_jk*.txt`, `summary_k_new.txt`):

  | build | GPU time | FP64 | warp instructions | atomics |
  |---|---|---|---|---|
  | one-pass J+K, v1.9.0 | 2.656 s | 183.5 GFLOP | 11.69 G | 18.8 M |
  | one-pass J+K, patched | 1.245 s | 65.3 GFLOP (−64%) | 4.59 G (−61%) | 18.8 M |
  | K only, v1.9.0 | 2.635 s | 182.3 GFLOP | 11.55 G | 12.3 M |
  | K only, part 1 prototype | 1.200 s | 62.2 GFLOP | 4.41 G | 12.3 M |
  | K only, patched | 1.199 s | 62.2 GFLOP | 4.38 G | 12.3 M |

  - In v1.9.0, one-pass J+K already cost only 1% more than K alone. Each quartet's integrals feed both J and K,
    and the extra 6.4 M atomics are the J updates.
  - Per class, the J+K kernels drop like the K ones: (ps|ss) 0.630 → 0.195 s, (ss|ss) 0.461 → 0.127 s,
    (ps|ps) 0.378 → 0.129 s.
  - The multi-thread and generic classes stay within ±0.002 s, as the census predicted. The largest change is
    the generic (dp|dp) K, 0.038 → 0.036 s with 4% fewer instructions.
- **def2-TZVP** (caffeine, 494 AOs, f functions on C/N/O; converged B3LYP density, median of 3;
  `prim_screen2_fixed_dm_tzvp.txt`):

  | build | v1.9.0 | patched | max \|Δ\| |
  |---|---|---|---|
  | K | 17.97 s | 12.83 s (1.40×) | 1.8e−15 |
  | one-pass J+K | 18.58 s | 13.50 s (1.38×) | 5.7e−14 |

  - The gain is smaller than with def2-SVP (2.2×). TZVP puts more of the time into the d and f classes, which the
    patch barely changes.
  - Patched K by kernel family (Nsight Compute; `ncu/summary_k_tzvp_new.txt`, `summary_k_new.txt`):

    | kernels | def2-SVP: time | def2-TZVP: time | def2-TZVP: warp instructions |
    |---|---|---|---|
    | one-thread unrolled | 0.968 s (81%) | 4.81 s (37%) | 15.8 G |
    | multi-thread unrolled | 0.140 s (12%) | 1.52 s (12%) | 5.9 G |
    | generic `rys_k_kernel` | 0.091 s (8%) | 6.57 s (51%) | 52.1 G |

  - The time spreads over about 60 classes, none above 5.3%. Every class outside the 25 unrolled ones goes to the
    generic kernel: most classes with an f shell, and (dp|dp), (dd|pp) and higher.
  - FP64 is 16–19% of that kernel's instructions, against 32–55% in the unrolled kernels. On this card the FP64
    pipe still sets its time (79% busy). On FP64-rich cards, its 52 G instructions (70% of the build's) would
    set it instead.
- **Upstream tests** (`upstream_tests_prim_screen2.txt`):
  - `scf/tests/test_scf_jk.py` passes, 11 of 11 (v1.9.0 too). It compares J and K with CPU PySCF to 1e−9 for
    def2-TZVP with spread-out atoms and dense random densities, at ω = 0 and ±0.2, for hermi 0/1/2, general
    contractions and l = 5.
  - `test_rhf.py`, `test_uhf.py`, `dft/tests/test_rks.py`, `test_uks.py` and `test_jk_one_pass.py`: 66 passed,
    4 skipped, with one-pass off (337 s) and on (269 s), as in It23 (442 and 377 s then).
  - `test_diffuse_orbital.py` (diffuse bases against Q-Chem, including Hessians and frequencies): 11 passed.

### The launch shape: one block per SM
- Every Rys J/K launch has a grid of `multiProcessorCount` blocks (20 here): persistent blocks that pull bra
  pairs from an atomic counter.
- Each block has 256 threads, or 512 when the kernel needs at most 128 registers (`adjust_threads`). So occupancy
  is 8 warps per SM, or 16 with 512 threads. That is 25% (50%) on Turing and 12.5% (25%) on A100/H100, which hold
  64 warps.
  - Part 1's register rise mattered only through that rule: the (ss|ss) kernel went from 88 to 136 registers,
    so its block went from 512 threads to 256.
- In the one-thread kernels 40% of the warp passes have no real quartet (census above). Without barriers those
  warps finish early and wait for the block's next bra pair, so on average only about 5 of the 8 warps per SM
  are busy.
  - The FP64 pipe is no longer saturated in the cheap classes: 53% busy in (ss|ss), 63% in (ps|ss).
- Kernels with 255 registers cannot hold more than 8 warps per SM on any current GPU (64 K registers per SM).
  Those with 128–200 could run 2–3 blocks of 128 threads.
  - Smaller blocks also shorten the idle tail (census model under Next).

### Next
- **Launch shape** (all GPUs). The cheap test first: 128-thread blocks and 2–3 blocks per SM for the one-thread
  kernels.
  - Census model (`task_census_caffeine_svp_model.txt`): the work-weighted idle share of the warp time falls
    from 45% with 256 quartets per block to 26% with 128, 15% with 64 and 7% with one warp per bra pair.
  - Their block size comes from `blockDim.x`, so only the dispatchers and the task pool (QUEUE_DEPTH entries per
    block, sized in `jk.py`) change.
  - If the idle tail remains, warp-level bra-pair scheduling: each warp fills and runs its own task list with
    ballots, with no block barriers.
- **The generic kernel** for larger bases: half of K with def2-TZVP and 70% of its instructions. Either unrolled
  kernels for more classes (the generator is not in the repository) or less index and barrier overhead in it.
- **One-pass J+K as the default for HF and global hybrids** (the user's decision; It23 made it opt-in). It is now
  1.28–1.57× faster than the J-engine + K default on this card. The trade on FP64-rich cards is unmeasured.
- **The MD J-engine** (0.69 s per caffeine cycle, half of a PBE cycle), **XC** (0.63 s) and the SCF host
  overheads, as planned.
- **The gradient kernels** (`unrolled_rys_jk_ip1.cu`, `unrolled_ejk_ip*.cu`) for primitive screening and padding.
- **Master.** The generator applies to upstream master c4ce229 unchanged, with the same 19 + 19, 6 + 6 and 2
  kernels. Since v1.9.0, master has only moved the CUDA includes of these files into `vhf.cuh`.

## GPU4PySCF investigation, part 3: the derivative kernels (gradients, Hessians) (node1), 2026-10-31
The user: "continue with whatever you think will provide the largest transferable speed up". A short scoping chose
the derivative kernels: the fix of parts 1–2 removes work on every card, and these kernels had the same waste.

### Scoping
- **Gradients.** Caffeine/def2-SVP, patched tree (`bench/gpu/grad_parts.py`). The two-electron part
  (`jk_energy_per_atom`) is 75% of the gradient: B3LYP 5.31 of 7.05 s, PBE 5.10 of 6.83 s.
  - Every DFT gradient sends J and K through the Rys energy-derivative kernels (`RYS_per_atom_jk_ip1`). Pure
    functionals pass k_factor 0 and pay the same, because there is no J-engine gradient.
  - The task filler already drops exchange-only quartets when k_factor is 0.
- **The kernels** (`gvhf-rys`):

  | file | used by | one-thread | multi-thread | generic |
  |---|---|---|---|---|
  | `unrolled_ejk_ip1.cu` | SCF and TDDFT gradients, NACs | 11 | 7 | 3 (`rys_contract_jk_ip1.cu`) |
  | `unrolled_ejk_ip2_type12.cu`, `_type3.cu` | Hessian energy second derivatives | 4 + 4 | 3 + 3 | 2 (`rys_contract_jk_ip2.cu`) |
  | `unrolled_rys_jk_ip1.cu` | J/K derivative matrices (Hessian, response) | 25 | 8 | 1 (unchanged) |

  - None skips primitive pairs.
  - Padding threads run the full primitive loops, including all Rys roots; they skip only the accumulation.
  - The energy kernels have no barrier at the top of the bra loop.
  - The 25 one-thread J/K-matrix kernels have one per primitive pair. Thread 0 publishes aij, aj/aij and 2ai
    through shared memory (`aij_cache`), though every thread has just computed the same values.

### Change (gpu4pyscf `exp/prim-screen` 258a730, from `bench/gpu/exp_prim_screen.py`)
- **One-thread energy kernels:** padding threads leave the task loop at once (no barrier follows inside it).
  Zero or below-e^−60 ket and bra primitive pairs are skipped: ckcl and cicj scale every term.
- **One-thread J/K-matrix kernels:** each thread uses its own aij, aj/aij and 2ai. The barrier and the shared
  copies go, and then the same skips apply (ckcl carries fac_sym, which is 0 on padding threads). The values are the
  same expressions on the same inputs.
- **Multi-thread and generic kernels:** the block-uniform bra skip, as in part 2.
- The K and J+K files come out of the generator byte for byte as in part 2.

### Results
- **Fixed density** (`bench/gpu/deriv_at_dm.py`, converged caffeine/def2-SVP B3LYP density, median of 3;
  `deriv_fixed_dm.txt`):

  | build | v1.9.0 | patched | max \|Δ\| | v1.9.0 against itself |
  |---|---|---|---|---|
  | gradient J + K (also B3LYP's factors) | 5.31 s | 4.58 s (1.16×) | 6e−14 | 3–8e−14 |
  | gradient J alone (pure functionals) | 5.09 s | 4.41 s (1.15×) | 6e−14 | 5e−14 |
  | gradient short-range K (ωB97X, ω −0.3) | 9.94 s | 8.41 s (1.18×) | 6e−16 | 6e−16 |
  | Hessian energy second derivatives (`_partial_ejk_ip2`) | 53.2 s | 51.3 s (1.04×) | 3e−13 | 1.3e−13 |
  | Hessian J/K derivative matrices (`_get_jk_ip1`) | 14.6 s | 11.25 s (1.30×) | 1.1e−13 | 6e−14 |

  - My mistake: in the last patched repeat, an Nsight Compute run of the gradient overlapped the ip2 and jk_ip1
    builds on the GPU (76.9 and 12.7 s). Those two times are not used; the first patched run is clean.
- **End to end** (`grad_parts.py`, `hess_parts.py`; `deriv_grad_e2e.jsonl`, `deriv_hess_e2e.txt`; the trees
  converge their own SCF, conv_tol 1e−10):

  | calculation | v1.9.0 | patched | max difference |
  |---|---|---|---|
  | B3LYP gradient | 7.05 s | 6.32 s (1.12×) | 5.0e−13 Eh/bohr |
  | PBE gradient | 6.84 s | 6.15 s (1.11×) | 5.4e−13 Eh/bohr |
  | ωB97X gradient | 16.93 s | 14.67 s (1.15×) | 4.6e−13 Eh/bohr |
  | B3LYP Hessian | 305.7 s | 291.6 s (1.05×) | 9.4e−12 |

  - At this density, one call each of the two derivative builds takes 51 + 11 s of the Hessian's 292 s. The rest,
    not profiled yet, includes the CPHF response and the XC second derivatives.

- **Where the gradient's time goes now** (Nsight Compute, `ncu/summary_grad*.txt`):

  | kernels | v1.9.0 | patched |
  |---|---|---|
  | one-thread | 2.55 s, 175 GFLOP | 1.80 s (−29%), 103 GFLOP |
  | multi-thread | 1.93 s | 1.92 s |
  | generic | 0.87 s | 0.88 s |

  - The one-thread kernels lose 29% here, against about 60% for K's one-thread kernels, because K's barrier came off
    too.
  - Multi-thread and generic kernels are 61% of the patched gradient, against 19% of SVP K. The gradient's extra
    angular momentum puts (pp|pp), (dp|ps), (ds|pp) and (dp|pp) in multi-thread kernels.
  - Those classes keep the FP64 pipe 91% busy, and FP64 is 63–70% of their instructions, so their time is real
    arithmetic.
  - What remains avoidable there is per-quartet: ket primitive pairs below the cutoff, and padding lanes. Their
    block-wide barriers prevent skipping either.
- **Unit tests** (`upstream_tests_deriv.txt`), all passing:
  - `grad/tests/test_rhf_grad.py`, `test_rks_grad.py`, `test_uhf_grad.py`, `test_uks_grad.py`: 40.
  - `hessian/tests/test_large_exponent.py`, `test_rhf_hessian.py`, `test_rks_hessian.py`: 27, 1 skipped.
  - `grad/tests/test_tdrks_grad.py`: 7.

### What is left in the multi-thread and generic kernels (`task_census.py --skip`; `task_census_caffeine_svp_skip.txt`)
- Their threads share a quartet across warps, so they run every padding slot and every ket primitive pair. Only
  bra pairs below e^−60 are skipped, block-wide.
- Counted on the K tasks of caffeine/def2-SVP, per-quartet skips would remove 28% of the multi-thread kernels'
  primitive quartets ((dp|pp) 28%, (dp|ds) 34%) and 22% of the generic kernel's.
- **A way to get there:** put a quartet's threads in one warp (`t = gout_stride·sq_id + gout_id`) and replace the
  block barriers inside the task loop with `__syncwarp` on the quartet's lanes. Quartets then skip independently,
  padding quartets leave at once, and no block-wide barrier remains in the integral loops.
  - Shared memory is already one column per quartet.
  - `rys_roots_for_k` needs a variant whose barriers sync the quartet's lanes.
  - For gout_stride ≤ 32, which covers every multi-thread kernel and the generic classes up to about (ff|pd).

### Next
- **The multi-thread and generic kernels, warp-local** (above): they are 61% of the gradient's two-electron time,
  63% of K with def2-TZVP and most of the Hessian's energy second derivatives. The 22–28% of their work removed
  carries over to every card, and the block barriers removed matter more on cards with more warps per SM.
- **The launch shape** of the one-thread kernels (part 2): 45% of their warp time is idle tail.
- **The rest of the Hessian** (CPHF response, XC second derivatives): 230 of its 292 s is outside these kernels.
- **A J-engine for gradients**: pure-functional gradients compute J with the Rys kernels; GPU4PySCF has no MD
  gradient.

## GPU4PySCF investigation, part 4 (scoping only): per-quartet skips in the multi-thread and generic kernels, 2026-10-31
The user approved restructuring these kernels so that a quartet's threads share one warp and skip padding and ket
primitive pairs on their own (part 3, Next). The design check stopped it before any kernel change.

### Why the per-quartet design fails
- **Divergence.** In the unrolled multi-thread kernels, each of a quartet's 4–16 threads accumulates its share of
  the outputs in a `switch (gout_id)` with hard-coded indices. Today a warp holds 32 quartets of one `gout_id`, so
  the switch is uniform. With a quartet's threads in one warp, that phase would serialize 4–16×.
- **Bank conflicts.** Shared memory has one column per quartet, with a row stride of nsq doubles (48 or 64, a
  multiple of 32 banks). A quartet's threads read different rows of their column, which all fall in one bank.
  That gives 4–16-way conflicts, worst on FP64-rich cards, where shared memory would then set the pace.

### What the warp-uniform alternatives would capture (`task_census.py --skip`; `task_census_caffeine_svp_skip.txt`)

| | multi-thread | generic |
|---|---|---|
| per-quartet skip (the infeasible design) | 28.2% | 22.4% |
| ket compaction over the block: it runs as many ket iterations as its neediest quartet | 4.5% | 0.6% |
| the same over groups of 32 quartets, with named barriers so an all-padding half block exits | 17.3% | 5.3% |

- These are shares of the primitive-quartet work on the def2-SVP K tasks.
- Even the 32-quartet version would be about 2% of K and 3–4% of the gradient's two-electron time, so it was not
  built.

### Warp stalls (Nsight Compute, caffeine/def2-SVP, patched tree; `ncu/ncu_stall_{k,grad}.csv`)

| kernels | stalled at a barrier | waiting on memory | FP64 pipe busy |
|---|---|---|---|
| one-thread, K | 44% | 43% | 72% ((ss\|ss) 53%) |
| one-thread, gradient | 50% | 24% | 80% ((ss\|ss) 47%) |
| multi-thread, K / gradient | 14% / 8% | 50% / 43% | 85% / 91% |
| generic, K / gradient | 12% / 3% | 67% / 49% | 80% / 87% |

- Stalls are percentages of active warps.
- On this card the multi-thread and generic kernels are limited by double-precision throughput, so only less
  arithmetic would speed them up.
- The one-thread kernels have no barrier left inside their task loop. Their barrier stall is the idle tail: warps
  done with their share of a bra pair wait at the block barrier for the slowest warp. The census model predicted
  45%, and 44–50% is measured.
- **Next, offered to the user:** the launch shape of the one-thread kernels (part 2): smaller blocks, more of them per
  SM, then warp-level bra-pair scheduling. These kernels are 81% of SVP K, 37% of TZVP K and 39% of the gradient's
  two-electron time.

## GPU4PySCF investigation, part 5: the launch shape of the one-thread kernels (node1), 2026-10-31
The user approved the change part 4 recommended.

### The problem
- Every Rys J/K kernel ran one persistent block per SM (grid = multiProcessorCount), with 256 threads, or 512
  when the kernel needs at most 128 registers (`adjust_threads`).
- The one-thread kernels have had no barrier inside their task loop since parts 1–3. A warp done with its share of
  a bra pair waits at the next bra pair's barrier for the slowest warp: 44–50% of their warp time (part 4).
- The task pools are larger than the kernels use. jk.py, grad/rhf.py and hessian/rhf.py allocate
  multiProcessorCount × 262144 entries (+1 or 3 heads), while the kernels use QUEUE_DEPTH = 65536 per block. That
  leaves room for 4 queues per SM.

### Change (gpu4pyscf `exp/prim-screen` 5128d98; `bench/gpu/exp_prim_screen.py`, `--launch 128x4` now the default)
- **The one-thread kernels** (K 19, J+K 19, gradient energy 11, Hessian energy 4 + 4, J/K derivative matrices 25)
  run blocks of 128 threads. As many blocks per SM as the occupancy calculator allows are launched, up to
  `POOL_BLOCKS_PER_SM` = 4 (`LAUNCH_JKMATRIX_KERNEL_PERSISTENT`, `LAUNCH_JKENERGY_KERNEL_PERSISTENT`).
- **The heads** of the five entry points that launch them (`RYS_build_k`, `RYS_build_jk`, `RYS_build_jk_ip1`,
  `RYS_per_atom_jk_ip1`, `RYS_per_atom_jk_ip2_type12/type3`) move behind the fourth queue.
- **The energy-derivative kernels** fixed their block size at compile time (`constexpr int nsq_per_block = 256`, and
  two root offsets written as `*256`). These are set to 128 to match.
  - Their dispatchers no longer double the block below 129 registers. With the constant, a 512-thread launch would
    have run each task twice and aliased the shared-memory columns.
  - The wheel's sm_70/80/90 builds use 168–255 registers in these kernels, so it never happened, but nothing
    guarded it.
- Multi-thread and generic kernels keep one block per SM: 160–255 registers leave no room for a second block of 256
  threads.

### Block-shape sweep (K and J+K builds at the converged caffeine/def2-SVP density, median of 5, 1530 MHz)

| threads per block × max blocks per SM | K | J+K |
|---|---|---|
| v1 shape: 256 × 1 (512 for ≤ 128 registers) | 1.189 s | 1.234 s |
| 128 × 1 | 1.173 s | 1.204 s |
| 128 × 2 | 1.028 s | 1.053 s |
| 128 × 3 | 1.015 s | 1.045 s |
| 128 × 4 (occupancy gives 2–4) | 1.010 s | 1.040 s |
| 64 × 4 | 1.044 s | 1.067 s |
| 32 × 4 | 1.038 s | 1.067 s |

- The sweep used a run-time variant of the dispatcher (`--launch env`, GPU4PYSCF_RYS_NSQ / _BLOCKS).
- All differences from v1.9.0 sit at the run-to-run noise.
- Counters at 128 × (2–3) (`ncu/ncu_stall_k128.csv`): the one-thread K kernels take 0.787 s instead of 0.969 s.
  - The FP64 pipe is 87% busy instead of 72% ((ss|ss) 53 → 81%, (ps|ss) 63 → 85%).
  - The barrier stall falls from 44% to 28%; occupancy rises from 25% to 33%.
- On this card these kernels are now limited by double-precision throughput, so warp-level bra-pair scheduling would
  add little here. Whether it pays on FP64-rich cards is for the A100/H100 suite.

### Results
- **Fixed density** (caffeine/def2-SVP, converged B3LYP density, against v1.9.0; `launch_fixed_dm.txt`):

  | build | v1.9.0 | before (parts 1–3) | launch shape | max \|Δ\| vs v1.9.0 |
  |---|---|---|---|---|
  | K | 2.620 s | 1.189 s | 1.011 s (2.59×) | 3.6e−15 |
  | one-pass J+K | 2.637 s | 1.234 s | 1.037 s (2.54×) | 4.2e−14 |
  | gradient J + K | 5.31 s | 4.58 s | 4.43 s (1.20×) | 8.3e−13 |
  | gradient J alone | 5.09 s | 4.41 s | 4.26 s (1.20×) | 7.1e−13 |
  | gradient short-range K | 9.94 s | 8.41 s | 8.18 s (1.22×) | 1.8e−14 |
  | Hessian energy second derivatives | 53.2 s | 51.3 s | 50.9 s (1.04×) | 1.3e−12 |
  | Hessian J/K derivative matrices | 14.6 s | 11.25 s | 10.88 s (1.34×) | 1.5e−13 |

- The new shape against itself differs by 4e−14 to 1.3e−13 (`launch_fixed_dm.txt`), the same spread as v1.9.0
    against itself.
  - The larger differences from v1.9.0 (8e−13 for the gradient, 1.3e−12 for the Hessian term; about 3e−14
    relative) come from the different summation order: with 128-thread blocks each thread accumulates other
    tasks.
  - Every unit test agrees with its reference.
- **SCF** (caffeine/def2-SVP, conv_tol 1e−10; `launch_scf.jsonl`; v1.9.0 times from part 2, same clocks):

  | case | v1.9.0 | one-pass off | one-pass on |
  |---|---|---|---|
  | RHF | 43.49 s | 22.42 s (1.94×) | 12.98 s (3.35×) |
  | RKS B3LYP | 57.76 s | 34.78 s (1.66×) | 24.44 s (2.36×) |
  | UKS B3LYP | 109.14 s | 70.72 s (1.54×) | 53.59 s (2.04×) |
  | RKS ωB97X | 78.96 s | 40.42 s (1.95×) | (no one-pass) |

  - Energies are those of v1.9.0 to the printed 10 decimals, with the same cycle counts. One-pass on and off
    differ as before.
- **Gradients** (`launch_grad.jsonl`): B3LYP 7.05 → 6.16 s (1.14×), PBE 6.84 → 6.00 s (1.14×); the two-electron
  part is 4.43 and 4.26 s.
- **def2-TZVP** (caffeine, fixed density): K 17.97 → 12.02 s (1.49×; 12.83 s before), J+K 18.58 → 12.60 s (1.47×;
  13.50 s before), at the noise against v1.9.0. The generic kernel, unchanged here, is half of it.
- **Unit tests** (`upstream_tests_launch.txt`), all passing:
  - `test_scf_jk.py` with `test_diffuse_orbital.py`: 22.
  - The SCF/DFT files and `test_jk_one_pass.py`: 66 with one-pass off and 66 with it on, 4 skipped each.
  - Gradient and TDDFT-gradient tests: 47.
  - Hessian tests, `test_large_exponent.py` included: 27, 1 skipped.
- **End-to-end gradients** against v1.9.0: max |Δg| 9.6e−13 (B3LYP) and 1.2e−12 Eh/bohr (PBE), same SCF cycles.

### Where things stand (caffeine/def2-SVP on the 1650 Super, against v1.9.0, parts 1–5)
- SCF with one-pass off: RHF 1.94×, B3LYP 1.66×, UKS B3LYP 1.54×, ωB97X 1.95×. With one-pass on: RHF 3.35×,
  B3LYP 2.36×, UKS B3LYP 2.04×.
- K at a fixed density 2.59×; def2-TZVP K 1.49×.
- Gradients 1.14×, Hessian 1.05× (part 3).

### Next
- **The generic kernel:** half of def2-TZVP K, 70% of its instructions, most of the Hessian's energy second
  derivatives. On FP64-rich cards its instruction overhead would set its time.
- **Warp-level bra-pair scheduling:** the one-thread kernels still wait at the bra-pair barrier 28% of the time.
  They are FP64-bound on this card, so this is for FP64-rich cards: measure there.
- **The rest of the Hessian** (230 of 292 s) and **a J-engine gradient**, as before.
- **jk_one_pass as the default** for HF and global hybrids (the user's decision). It is now 1.32–1.73× faster than
  the default path on this card.
- **A test suite for the A100/H100 run:** the user wants to run one once more options are exhausted, so the card is
  needed only briefly. `one_pass_gate.py`, `jk_at_dm.py`, `deriv_at_dm.py`, `grad_parts.py`, `hess_parts.py`,
  `ncu_parts.py` and the upstream test lists already run unchanged.

## GPU4PySCF investigation, part 6: the generic kernel (node1), 2026-10-31
The user approved working on it.
- **Why it matters:** with def2-TZVP the generic kernel is 51% of K and 70% of its instructions (part 2), and it runs
  most of the Hessian's energy second derivatives.
- **Lesson from part 4:** the code and the counters were checked before anything was built.

### Where the generic kernel spends its work (`bench/gpu/ncu_class.py`, `generic_kernel_profile.txt`)
- **Method:** Nsight Compute source counters, from a build with `-lineinfo` (line tables only; the generated code is
  the same). One launch each of (dp|dp), (fs|dp), (fp|pp) and (fd|ps), at the converged def2-TZVP density.
- **Phases:** the output products (`inner_dot`) carry 38–52% of the FP64 flops. The TRR along i and k carry 30–35%,
  the HRR 5–10%, and the roots and primitive setup the rest.
- **Instruction mix:** FP64 is 16–17% of the warp instructions. For (fp|pp): IMAD 21%, LDS 19%, IADD3 14%,
  DFMA + DMUL 15%.
  - Each product takes three shared-memory loads, whose addresses come from runtime index arrays, plus their
    address arithmetic.
  - The loop bounds are runtime class values, so the compiler can neither fold the addresses nor reuse repeated loads.
- **Stalls and spills:** stalls are mostly long scoreboard. Spills are a few bytes (ptxas), so these stalls are not
  local memory.
- **Redundant divisions:** two per-root lines recompute double divisions in every thread of a quartet.
  - `b10 = .5/aij·(1−rt_aij)` is 4.9% of the FP64 work. It re-reads `aij` from shared memory after a barrier, so
    `.5/aij` cannot leave the loop. The equivalent `b01 = .5/akl` uses a register and was hoisted (0.6%).
  - `rt_aa = rt/(aij+akl)` is 4.6%.

### Change (gpu4pyscf `exp/prim-screen` 411a428; `exp_prim_screen.py --generic on`, the default)
- **Registers instead of shared copies:** inside the root loop, `rys_k_kernel` and the `BUILD_4C_GXYZ` macro use the
  `aij` and `aj/aij` every thread already holds, instead of the shared copies.
  - The macro builds the g arrays of `rys_jk_kernel`, the four generic gradient kernels, the two generic Hessian
    kernels and int4c2e.
- **No TRR on padding quartets:** their arithmetic is skipped, and the barriers stay. Every user drops padding
  quartets right after the TRR, so their g arrays are never read.
- Both are bit-for-bit for real quartets. Spills stay at 12–28 bytes.
- **Effect:** FP64 work falls 7–10% in the four classes, but warp instructions only 0.3–1.7%, and the FP64 pipe stays
  about 79% busy. Padding quartets share warps with real ones, so skipping their TRR idles lanes but removes no warp
  instructions.
- **Times** (fixed density; `generic_tweak_fixed_dm.txt`):
  - def2-TZVP K 12.02 → 11.75 s, J+K 12.60 → 12.35 s (−2%).
  - def2-SVP and the derivative builds move by 0–0.6%.
  - All at the noise against v1.9.0.

### What specialized code would buy (`generic_vs_unrolled.txt`)
- **Method:** a temporary dispatcher switch (not committed) sent every class of the def2-SVP K build through the
  generic kernel, on the same build. The 25 classes that have unrolled kernels can then be compared directly.
- **Results:**

  | unrolled kernels | speed against the generic kernel | where the difference comes from |
  |---|---|---|
  | one-thread ((ss\|ss) 3.7×, (ps\|ss) 2.7×, (ps\|ps) 2.7×, (dp\|ps) 2.1×) | 2.1–3.7× | 2–6× less FP64 work: per-quartet primitive skips and no padding work |
  | multi-thread ((dp\|pp), (dp\|ds), (dd\|ps), (ds\|dp)) | 1.2–1.3× | the same or more FP64 work, but 2.7× fewer instructions (FP64 39–45% of them against 16–18%) and an FP64 pipe 82–87% busy instead of 67–78% |

  - The whole def2-SVP K build takes 1.016 s with its unrolled kernels and 2.277 s with the generic kernel alone.
- **The f classes need the multi-thread design.** All of def2-TZVP's generic classes (f with p, d or f, and d with d)
  have more outputs than one thread's registers can hold: (fp|pp) has 270, against 54 in the largest one-thread kernel.
- **Estimate:** generated multi-thread kernels for those classes would be about 1.2–1.3× faster on this card, about
  10% of def2-TZVP K. On FP64-rich cards the 2.7× fewer instructions would matter more.
- **Cost:** a code generator, or C++ templates on (li, lj, lk, ll) with compile-time output tiles, plus per-class
  checks. That covers about 40 K classes, then the J+K, gradient and Hessian families.
- **Not done: skipping wasted f-tile slots.** f has 10 Cartesian components in 4 tiles of 3, so 2 of 12 slots are
  wasted.
  - A per-thread tile pattern would skip them, but it diverges in warps that mix tiles: classes with 48 quartets per
    block or 16 threads per quartet.
  - In those warps it would cost more than it saves.

### Gate (`upstream_tests_generic.txt`)
- **Unit tests**, all passing:
  - `test_scf_jk.py` and `test_int4c2e.py`: 13.
  - Gradients and TDDFT gradients: 47.
  - Hessians, `test_large_exponent.py` included: 27, 1 skipped.
- **Caffeine/def2-TZVP B3LYP SCF** (one-pass off, conv_tol 1e−10), against v1.9.0 with all of parts 1–6:
  321.6 → 235.5 s (1.37×), E within 7e−13 Eh, 15 cycles each.
- The node1 build now keeps `-DCMAKE_CUDA_FLAGS=-lineinfo` (line tables for source profiles). It does not change the
  generated code.

### Next
- **Generated multi-thread kernels for the generic classes** (above): about 10% of def2-TZVP K on this card, and more
  where FP64 is cheap. Several iterations: the generator or templates, then about 40 K classes, then the J+K,
  gradient and Hessian families.
- **Otherwise, as before:** the rest of the Hessian (230 of 292 s), a J-engine gradient, jk_one_pass as the default
  (the user's call), and the A100/H100 test suite.

## GPU4PySCF investigation, part 7: generated multi-thread K kernels (node1), 2026-10-31
The user chose the kernel generator. This first iteration covers K only.
- **Commit:** gpu4pyscf `exp/prim-screen` 7eb2117. `bench/gpu/exp_prim_screen.py gpu4pyscf` regenerates it from
  `perf` with every default: `--gen-k` now defaults to the 18 classes below (`GEN_K`), and `--gen-k off` leaves
  them out.

### The generator (`bench/gpu/gen_rys_k.py`)
- **Model:** GPU4PySCF's six unrolled multi-thread kernels (rys_k_2021 … rys_k_3110). Their generator is not in the
  repository, so `gen_rys_k.py` reproduces their style for any (li, lj, lk, ll).
- **Block shape:** nsq quartets per block (32 or 64, so that a warp holds one `gout_id`) and gs threads per quartet,
  256 threads in all.
- **Structure:**
  - The prologue, primitive loops and bra-pair skip are those of the unrolled kernels.
  - Threads 0–2 each run the TRR and HRR of one Cartesian direction, fully unrolled with constant offsets. The g
    arrays use the generic kernel's in-place layout: g(i, k) at i + k·stride_k, then the HRR to (i, j) and (k, l).
  - The outputs, in lexical Cartesian order with i fastest, are dealt round robin to the gs threads. Each thread
    accumulates its own `gout[]` with constant shared-memory addresses, in a `switch (gout_id)` that never diverges.
  - The K contraction is the generic kernel's (vk_il from dm_jk, vk_ik from dm_jl; when ish ≠ jsh, vk_jl from dm_ik
    and vk_jk from dm_il), unrolled per thread with one atomicAdd per element.
- **Integration:** `exp_prim_screen.py --gen-k CLASS:NSQxGS,...` writes `generated_rys_k.cu`, adds it to the
  gvhf-rys library and makes `RYS_build_k` try `rys_k_generated()` after the unrolled kernels and before the generic
  one.
- **Experiment builds:** `--gen-first` adds run-time switches. `GPU4PYSCF_RYS_GEN_FIRST=1` runs the generated kernels
  before the unrolled ones; `GPU4PYSCF_RYS_GEN_OFF=1` skips them.

### Validation against GPU4PySCF's own kernels
- Generated versions of the six upstream multi-thread classes, run instead of them on the def2-SVP K build
  (`GEN_FIRST=1`), take 1.015 s against 1.014 s. Registers match: 2111 200 against 198–200, 2021 160 against 162.
- K agrees with v1.9.0 to 1.1e−14.

### Which classes fit (ptxas, sm_75)

| outputs per thread | registers, spills | classes |
|---|---|---|
| ≤ 45 | 168–255, none to 112 bytes | (fp\|pp), (fd\|ps), (fp\|ds), (fs\|dp), (dp\|dp), (fd\|ds), (ff\|ps), (fs\|fs), (fp\|fs), (fs\|fp), (dd\|pp), (fs\|dd), (dd\|ds), (ff\|ss), (ds\|dd) |
| 68–75 | 255, 76–164 bytes | (fp\|dp), (fd\|pp), (fs\|fd): still faster than the generic kernel |
| 75 | 255, 416–704 bytes | (fd\|fs), (ff\|ds): left to the generic kernel |
| ≥ 81 | heavy spills | (dp\|dd), (dd\|dp), (fp\|dd) and larger: need another design |

### Results (caffeine, GTX 1650 Super at 1530 MHz)
- **Per class** (Nsight Compute, the def2-TZVP K build at the converged B3LYP density; `gen_kernel_profile.txt`):
  - **Totals:** the 18 classes go from 3.93 to 2.76 s (1.43×), each 1.10–2.00× faster. All K integral kernels go
    from 11.82 to 10.65 s.
  - **Instructions:** warp instructions fall 2.3–4.4×, and FP64 is now 32–44% of them, against 16–18%.
  - **What sets the time on this card:** both kernels are FP64-bound, so each class's time follows its FP64 warp
    instructions divided by the FP64 pipe's utilization.
    - FP64 warp instructions fall 4–43%. They fall 4–5% for (dp|dp) and (dd|pp), which gain 1.10× and 1.14×, and
      35–43% for the classes with two f shells, which gain 1.55–2.00×.
    - Part of the difference is the generic kernel's 3-wide tiles: an f shell's 10 components fill 12 slots.
    - The FP64 pipe is 74–87% busy, against 65–83%.
  - **Other cards:** on cards with fast FP64 the generic kernel's other instructions would weigh more, so the 2.3–4.4×
    fewer instructions should count for more there. That needs the A100/H100 suite.

  | class | generic → generated | warp instructions (G) | FP64 warp instructions (G) |
  |---|---|---|---|
  | (fp\|pp) | 0.426 → 0.340 s (1.25×) | 3.81 → 1.34 | 0.629 → 0.542 |
  | (fp\|dp) | 0.401 → 0.323 s (1.24×) | 3.52 → 1.18 | 0.616 → 0.506 |
  | (fd\|ps) | 0.374 → 0.248 s (1.51×) | 3.15 → 0.96 | 0.550 → 0.400 |
  | (fp\|ds) | 0.374 → 0.246 s (1.52×) | 3.02 → 0.90 | 0.543 → 0.381 |
  | (fs\|dp) | 0.353 → 0.219 s (1.61×) | 2.98 → 0.80 | 0.515 → 0.353 |
  | (dp\|dp) | 0.262 → 0.239 s (1.10×) | 2.35 → 0.96 | 0.390 → 0.375 |
  | (fd\|pp) | 0.225 → 0.173 s (1.30×) | 2.03 → 0.65 | 0.341 → 0.279 |
  | (fd\|ds) | 0.223 → 0.150 s (1.49×) | 1.83 → 0.61 | 0.322 → 0.207 |
  | (ff\|ps) | 0.183 → 0.102 s (1.79×) | 1.41 → 0.41 | 0.245 → 0.151 |
  | (fp\|fs) | 0.177 → 0.114 s (1.55×) | 1.39 → 0.43 | 0.251 → 0.162 |
  | (fs\|fs) | 0.175 → 0.093 s (1.88×) | 1.33 → 0.36 | 0.231 → 0.138 |
  | (dd\|pp) | 0.150 → 0.131 s (1.14×) | 1.39 → 0.53 | 0.220 → 0.210 |
  | (fs\|fp) | 0.149 → 0.091 s (1.64×) | 1.16 → 0.35 | 0.205 → 0.133 |
  | (fs\|fd) | 0.110 → 0.055 s (1.99×) | 0.84 → 0.19 | 0.143 → 0.081 |
  | (fs\|dd) | 0.108 → 0.068 s (1.58×) | 0.90 → 0.28 | 0.156 → 0.105 |
  | (dd\|ds) | 0.095 → 0.078 s (1.22×) | 0.80 → 0.34 | 0.139 → 0.111 |
  | (ff\|ss) | 0.091 → 0.045 s (2.00×) | 0.68 → 0.19 | 0.107 → 0.069 |
  | (ds\|dd) | 0.055 → 0.040 s (1.38×) | 0.45 → 0.18 | 0.075 → 0.059 |
- **K at a fixed density** (`gen_fixed_dm.txt`):
  - def2-TZVP 11.75 → 10.59 s (1.11×). v1.9.0 takes 17.97 s, so parts 1–7 give 1.70×.
  - def2-SVP 1.014 → 0.999 s; its generated classes are (dp|dp), (dd|pp), (dd|ds) and (ds|dd).
  - Max |ΔK| against v1.9.0: 4.0e−14 (TZVP) and 9.9e−15 (SVP).
- **Non-symmetric densities** (hermi=0, as the TDDFT response uses; `gen_fixed_dm.txt`).
  - The unit tests call the K-only path with symmetric densities only.
  - So `jk_at_dm.py --antisym 0.5` adds an antisymmetric matrix of the density's size and sparsity, and K is compared
    with v1.9.0 at def2-TZVP:

  | Coulomb | v1.9.0 | patched | max \|ΔK\| |
  |---|---|---|---|
  | full | 19.17 s | 11.59 s | 3.6e−14 |
  | long range (ω = 0.3) | 21.13 s | 11.36 s | 1.2e−14 |
  | short range (ω = −0.3) | 34.05 s | 19.65 s | 1.8e−14 |
- **Caffeine/def2-TZVP B3LYP SCF** (one-pass off, conv_tol 1e−10; `upstream_tests_gen.txt`): 235.5 → 221.0 s
  (−6%). Against v1.9.0's 321.6 s that is 1.46× for parts 1–7. The energy is the same as in part 6, within 7e−13 Eh of
  v1.9.0, with 15 cycles each.
- **Unit tests**, all passing: `test_scf_jk.py`, `test_int4c2e.py` and `test_diffuse_orbital.py` (24), and the
  RHF, UHF, RKS and UKS files (57, 4 skipped). The derivative, J and J+K kernels are unchanged.
- **Final dispatcher.** After the gate, the hook became `if (!rys_k_unrolled(...) && !rys_k_generated(...))`, the
  same calls in the same order. node1 was rebuilt with it and gave def2-SVP K within 1.2e−14 of v1.9.0 again; the
  non-symmetric checks above ran on that build.

### What stays generic
- **Share:** 22 classes, 2.39 of the 10.65 s of def2-TZVP K integral kernels (22%; `gen_kernel_profile.txt`).
  - (fd|fs) and (ff|ds) take 0.26 s. At 75 outputs per thread they spill 416–704 bytes.
  - The rest have 81 to 1250 outputs per thread at 8 threads per quartet. The largest are (fd|dp) 0.29 s, (fp|fp)
    0.23 s, and (ff|dp), (fp|dd), (fd|fp), (ff|pp) and (fp|fd) at 0.13–0.15 s each.
- **Why the design stops there:** 256 threads per block at 255 registers already fill the register file. With 32
  quartets per block a warp holds one `gout_id`, so that leaves 8 threads per quartet. More threads per quartet would
  mean fewer quartets per block, and then two `gout_id`s would share a warp and the switch would diverge.
  - The generic kernel avoids the divergence because all its threads run the same code, with addresses from index
    arrays. That indexing is exactly what costs it its instructions.
- **Two passes over the outputs** (an estimate, not tried): each pass would recompute the roots, TRR and HRR for
  half of the outputs.
  - For these classes the products dominate the FP64 work. (fd|dp), for example, has 1080 outputs against a g array
    of 72 per direction.
  - Repeating the TRR and HRR would cost about what the generic kernel loses to its tile waste. So two passes would
    do about the same FP64 work as the generic kernel, with 2–4× fewer instructions.
  - On this FP64-bound card that is about even. On cards with fast FP64 it should be a gain, but only the A100/H100
    suite can show that.

### Next
- **Generated derivative kernels:** the energy-derivative kernels (the gradients' `ejk_ip1`, the Hessian's `ip2`)
  multiply each integral at once by a precomputed density product (`dd_cache`) and add it to 12 gradient components
  per thread.
  - They hold no output arrays, so registers do not limit them by class size.
  - On def2-SVP, 61% of the gradient's two-electron time is in multi-thread and generic kernels (part 3).
  - def2-TZVP should weigh more, but it has not been profiled yet; that would be the first step.
- **The J+K one-pass kernels** (`rys_jk_kernel`): the same generator plus the J contraction. They matter only with
  `jk_one_pass`, which is opt-in.
- **Big K classes in two passes** (above): about even here; for cards with fast FP64.
- **Otherwise, as before:** the rest of the Hessian, a J-engine gradient, `jk_one_pass` as the default (the user's
  call), and the A100/H100 test suite.

## GPU4PySCF investigation, part 8: generated gradient kernels (node1), 2026-10-31
The user approved generated derivative kernels for gradients and Hessians, starting with a profile of a def2-TZVP
gradient. This part covers the gradient (`RYS_per_atom_jk_ip1`); the Hessian kernels are not done yet.
- **Commit:** gpu4pyscf `exp/prim-screen` e1626eb. `bench/gpu/exp_prim_screen.py gpu4pyscf` regenerates it from `perf`
  with every default; `--gen-ejk` now defaults to the 64 classes below (`GEN_EJK`), and `--gen-ejk off` leaves them
  out.

### Where the def2-TZVP gradient spends its time (`grad_tzvp_baseline.txt`, `gen_ejk_profile.txt`)
- **Gradient:** the caffeine/def2-TZVP B3LYP gradient takes 86.7 s on gpu4pyscf 7eb2117, 37% of an SCF (221 s).
  - The two-electron part is 81.5 s of it (94%).
  - v1.9.0 takes 83.9 s for that part, so parts 1–7 barely touched it.
- **Kernel families** (Nsight Compute, the two-electron part at the converged density):

  | kernels | time | share |
  |---|---|---|
  | generic `rys_ejk_ip1_kernel` | 64.7 s | 79% |
  | the 7 unrolled multi-thread kernels | 10.3 s | 13% |
  | the 11 unrolled one-thread kernels | 6.6 s | 8% |

- **The generic kernel is FP64-bound:** its FP64 pipe is 86% busy, although FP64 is only 27% of its instructions.

### The change: sums per Cartesian tuple
- **What GPU4PySCF does:** its kernels compute, for every output (ij|kl) of a quartet and every root, twelve
  derivative terms `v_ix += (2 ai g_x[t+1] − ix g_x[t−1]) · Iy Iz dd`, and so on. That is about 42 FP64 operations
  per output.
- **The observation:** each derivative factor depends on one direction's index tuple only, t = (ix, jx, kx, lx) for x.
  Grouping the outputs by tuple, one direction at a time, gives the same sum:

  Σ_t f_x(t) S_x(t), where S_x(t) sums dd · Iy · Iz over the outputs with x tuple t.

  - Each output then costs 2 FP64 operations per direction.
  - Each tuple of a direction costs its 4 derivative factors and 4 FMAs.
- **Three passes per root, one per direction:** in each pass the outputs, ordered by tuple, are cut into runs of equal
  cost, one per thread. A tuple is thus split between at most two threads.
- **Exactness:** the terms are the same, summed in another order.
  - The results differ from v1.9.0 by 2–8e−12 on values up to 31 (1e−13 to 3e−13 relative).
  - That is v1.9.0's own rounding. Its two-electron gradient summed over the atoms, zero in exact arithmetic by
    translational invariance, is 3e−12 to 1.1e−11 on def2-SVP; with the generated kernels it is 5e−12 to 7e−12.

### The generator (`bench/gpu/gen_rys_ejk.py`)
- **Multi-thread kernels** (nsq quartets per block, gs threads per quartet):
  - The prologue, primitive loops and bra-pair skip of GPU4PySCF's multi-thread gradient kernels.
  - The density products of each task in `dd_pool`, as there. A barrier at the top of each task is new: the threads
    now read products they did not fill themselves.
  - Threads 0–2 run the TRR and HRR of one direction each, as `gen_rys_k.recursions` for (li+1, lj, lk+1, ll).
  - The output phase reads shared memory through a `volatile` pointer. Otherwise the compiler keeps every value it
    loaded for a later run in registers: 255 registers and spills, or 0.3–3.7 KB of spills at a 128-register cap.
    With `volatile` there are none at 255 registers.
- **Block shapes:**
  - 32×8 or 64×4 where 32 quartets fit in 64 KB of shared memory, short-range roots included.
  - 16×16 and 8×32 for the larger classes. A warp then holds two or four `gout_id`s.
    - To keep the warp from diverging, its lanes run one instruction stream: each address is selected per lane
      (h = `gout_id` % k), all lanes apply their sums wherever one lane's tuple changes, and padding reads a zero
      density slot.
    - h is read through `asm volatile` once per pass. Otherwise the compiler hoists every selected address out of
      all the loops and spills 9–15 KB.
  - One thread per quartet (128×1) for the smallest classes. The g arrays and density products live in registers
    (local arrays with constant indices), and blocks launch as for the one-thread kernels of part 5.
- **Classes:** 64 in all, with each shape picked from the per-class measurements below. (ff|ff) does not fit at
  8 quartets per block and stays generic, as do classes with g functions.
- **Build time:** the kernels go into four translation units, so the library compiles in parallel (3.5 min on node1
  instead of 8).
- **Integration:** `exp_prim_screen.py --gen-ejk CLASS:NSQxGS,...` writes `generated_ejk_ip1*.cu` and adds them to
  the gvhf-rys library. `RYS_per_atom_jk_ip1` tries `rys_ejk_ip1_generated()` first, then the unrolled kernels, then
  the generic one. `--gen-first` builds read `GPU4PYSCF_RYS_GEN_OFF=1` to skip them.

### Results (caffeine, GTX 1650 Super at 1530 MHz)
- **Per kernel family** (`gen_ejk_profile.txt`; the classes grouped by the kernel they ran in 7eb2117):

  | classes that ran | time | warp instructions | FP64 warp instructions |
  |---|---|---|---|
  | the generic kernel | 64.7 → 17.0 s (3.81×) | 4.16× fewer | 3.97× fewer |
  | the unrolled multi-thread kernels | 10.3 → 5.6 s (1.84×) | 1.20× fewer | 1.90× fewer |
  | the unrolled one-thread kernels | 6.6 → 5.3 s (1.24×) | 1.17× fewer | 1.25× fewer |
  | all | 81.6 → 27.9 s (2.92×) | 3.32× fewer | 2.99× fewer |

  - The FP64 pipe is 80–91% busy in all but the 8×32 classes, so on this card the time follows the FP64
    instructions.
  - The 8×32 classes run at 43–76% FP64 utilization (spills, code size) and gain 2.2–3.1×. The 16×16 ones gain
    3.6–4.3×.
  - On cards with fast FP64, the 3.3× fewer instructions and the lighter memory traffic should matter as well. By
    the code, an output takes about 9 memory accesses per root against about 20. The A100/H100 suite has to show it.
- **Shape experiments** (same density):
  - 64×4 against 32×8: 3–12% faster for six small classes, 5–15% slower for (pp|pp), (fp|ss) and (fd|ss); 0.7% in all.
  - One thread per quartet: 1.05–1.42× over GPU4PySCF's one-thread kernels for its 11 classes. For (fs|ss), (fs|ps)
    and (fp|ss) it is 1.6–2.0× over the multi-thread generated kernels.
- **The two-electron part at a fixed density** (`gen_ejk_fixed_dm.txt`; the 7eb2117 column for def2-SVP is the
  experiment build with the generated kernels switched off):

  | caffeine, factors | v1.9.0 | 7eb2117 | generated | max \|Δ\| against v1.9.0 |
  |---|---|---|---|---|
  | def2-TZVP, B3LYP's J + 0.2 K | 83.9 s | 81.5 s | 27.7 s (2.94×) | 6.2e−12 (values up to 28) |
  | def2-TZVP, short-range K (ω = −0.3) | 157.2 s | 152.4 s | 50.2 s (3.04×) | 1.2e−13 |
  | def2-SVP, B3LYP's factors | – | 4.42 s | 2.59 s (1.70×) | 2.5e−12 |
  | def2-SVP, short-range K | – | 8.17 s | 4.68 s (1.75×) | 5.1e−14 |
- **Whole gradients** (`upstream_tests_ejk.txt`, `gen_ejk_grad_e2e.jsonl`), the same SCF cycles each:

  | caffeine | gradient | two-electron part | max \|Δg\| |
  |---|---|---|---|
  | def2-SVP B3LYP | 7.05 → 4.34 s (1.62× against v1.9.0) | 5.32 → 2.60 s | 2.6e−12 Eh/bohr |
  | def2-SVP PBE | 6.84 → 4.24 s (1.61×) | 5.10 → 2.50 s | 3.8e−12 |
  | def2-SVP ωB97X | 16.93 → 8.96 s (1.89×) | 15.21 → 7.23 s | 2.4e−12 |
  | def2-TZVP B3LYP | 86.7 → 33.0 s (2.63× against 7eb2117) | 81.5 → 27.7 s | 8.7e−12 against 7eb2117 |

  - The def2-TZVP gradient now costs 15% of its SCF (221 s), against 39% before.
- **Unit tests**, all passing: the RHF, RKS, UHF and UKS gradient tests (40), the TDDFT gradient tests for all four
  (25), and the TDDFT ground-to-excited NAC tests (9).

### Next
- **The Hessian's energy second derivatives** (`ejk_ip2` type12 and type3; 50.7 s of a caffeine/def2-SVP Hessian,
  part 6).
  - Their diagonal terms (xx, yy, zz) sum per tuple as in the gradient.
  - The mixed terms (xy, xz, ...) multiply derivative factors of two directions, so grouping by one tuple saves less.
  - GPU4PySCF unrolls them only up to (pp|pp); everything with d functions runs the generic kernels.
- **TDDFT gradients:** their multi-density kernels (`RYS_per_atom_jk_ip1_multidm`, `_sum`) are generic only.
- **Otherwise, as before:** the rest of the Hessian, a J-engine gradient, `jk_one_pass` as the default (the user's
  call), and the A100/H100 test suite.

## GPU4PySCF investigation, part 9: where a Hessian and a TDDFT gradient spend their time; one-pass J+K in the Hessian CPHF (node1), 2026-11-01
After part 8 the user chose to re-profile before writing more kernels. They then approved using the one-pass J+K
kernels inside the Hessian's CPHF, the path they had kept as their call for the SCF (`jk_one_pass`).

### Stage profile (`bench/gpu/stage_times.py`, `hess_tdgrad_stages.txt`; caffeine def2-SVP B3LYP, e1626eb)
- **Method:** synchronizing timers around the stages. Times are inclusive (the CPHF contains its J/K and XC
  response); within a group only the outermost call counts.
- **Hessian, 290 s:**

  | stage | time |
  |---|---|
  | CPHF (`solve_mo1`), 7 iterations | 170 s |
  | of it: J-engine J for 72 densities | 75 s |
  | of it: XC response (`nr_rks_fxc_mo`) | 71 s |
  | of it: K for 72 densities | 23 s |
  | two-electron second derivatives (`_partial_ejk_ip2`) | 51 s |
  | XC first derivatives of V and XC second derivatives | 29 + 27 s |
  | J/K derivative matrices (`_get_jk_ip1`) | 11 s |
  | ground-state XC kernel, rebuilt per CPHF iteration | 0.9 s |

- **TDA gradient of state 1 (of 5), 58 s:**
  - The Z-vector CPHF takes 35 s: J/K 24 s and XC response 14 s.
  - Then the XC kernel contractions (10 s) and the two-electron derivatives (the summed multi-density kernels, 9 s).
  - The full-TDDFT gradient stopped in the excited-state solver: out of memory on the 4 GB card.

### Why J cost three times K in the Hessian (`multi_dm_jk.py`)
- **J-engine:** its loop over blocks of 8 densities (`dm_offset` in `unrolled_md_j_4dm.cu`, `DM_BLOCK` in
  `md_contract_j.cu`) encloses the whole integral evaluation, Boys functions and Hermite recursions included. For
  the CPHF's 3 × natm = 72 densities it evaluates every integral 9 times.
- **Rys J+K kernels:** they evaluate each integral once and contract it with all densities.
- **Timings at the converged density** (times (1 + X) per density):

  | densities | J-engine J | K | one-pass J+K | J + K apart |
  |---|---|---|---|---|
  | 1 | 0.78 s | 1.00 s | 1.05 s | 1.78 s |
  | 8 | 1.73 s | – | – | – |
  | 72 | 15.6 s | 4.5 s | 7.0 s | 20.1 s |

  - The one-pass J agrees with the J-engine to 4.7e−12 (values up to 44).
  - The Rys J-only path (`jk._VHFOpt.get_j`) is slower for one density (2.9 s) and failed with an illegal memory
    access for 8 and 72, so pure functionals keep the J-engine.

### Change (gpu4pyscf `perf` 3dfd27d, merged into `exp/prim-screen` 7089dfb)
- **`hessian.rhf._get_jk_mo` and `hessian.uhf._get_jk_mo`:** for full-range Coulomb with both J and K, J and K now
  come from `_VHFOpt.get_jk` in one pass. UHF takes J of each spin density and sums the two.
- **Switch:** `hessian_rhf_jk_mo_one_pass = False` in the config restores the J-engine.
- **Unchanged:** range-separated K and J alone keep their paths.
- **Memory:** the CPHF's block size already counts the AO matrices of the densities, J and K together, for RHF and
  for UHF.

### Results (`hess_onepass.txt`)
- **Time:** the caffeine def2-SVP B3LYP Hessian goes from 290 to 227 s (1.28×; 306 s in v1.9.0, so 1.35× there). Its
  CPHF J/K goes from 98 to 35 s.
- **Accuracy:** the Hessian agrees with v1.9.0 to 1.9e−11 (values up to 1.0).
- **Tests:** the RHF, RKS, UHF and UKS Hessian tests (37) and `test_large_exponent.py` with
  `test_level_shift_hessian.py` (11, 1 skipped) pass.
- **Where the Hessian stands now (227 s):** CPHF XC response 71 s, XC derivative terms 56 s, two-electron second
  derivatives 51 s, CPHF J/K 35 s, J/K derivative matrices 11 s.

### The CPHF's XC response (`fxc_mo_parts.py`, 72 densities on the 102,400-point CPHF grid, 14.2 s per call)
- **Split:** the AO integration takes 55%: one nao × nao GEMM per density and grid block. The response densities
  (`eval_rho4`) take 45%: one GEMM per density with nocc = 51 columns. AO values and the fxc contraction are below
  1%.
- **Possible exact restructurings, not done:**
  - Integrate in the MO basis: V_mo = Φᵀ W Φ_occ plus the GGA transpose term. That needs nmo × nocc × 2 instead of
    nao² per density and point, about 2× fewer FP64 operations (caffeine: 25k against 61k).
  - Do all densities in a few wide GEMMs per grid block, for both parts.

### Next
- The CPHF's XC response (above): about 30 s of the Hessian by estimate, and it applies to every hybrid or GGA Hessian.
- Otherwise: the XC derivative terms (56 s), the Hessian's second-derivative kernels (51 s), the TDDFT-gradient
  kernels (9 s of 58 s), and the A100/H100 suite.

## GPU4PySCF investigation, part 10: the Hessian CPHF's XC response in the occupied-MO basis (node1), 2026-11-01
The user approved reworking the CPHF's XC response (part 9: 71 of the Hessian's 227 s).

### Where the time went (`fxc_mo_parts.py`, `hess_fxc_mo.txt`)
- **Inputs:** one call of `hessian.rks.nr_rks_fxc_mo` for 72 densities on the 102,400-point CPHF grid, caffeine def2-SVP.
- **Split:** in the AO integration, each density builds the full AO matrix V of every grid block. That is a GEMM of
  nao_sub × nao_sub, transformed to C^T V C_occ only at the end. The AO integration takes 55% (B3LYP), 80% (SVWN)
  and 82% (r2SCAN). The response densities (`eval_rho4`, one GEMM of nao_sub × nocc per density) take the rest.
- **The response densities are already at their minimum FP64 work:** χ_i = aoᵀ mo1_i is needed for every component.
  On this card they also run at about 65% of the FP64 peak, so batching them would gain little here.

### Change (gpu4pyscf `perf` 1213746, merged into `exp/prim-screen` 5108750)
- **The idea:** each grid block adds its part of V C_occ directly, with the occupied orbitals on the grid
  (ψ = aoᵀ C_occ, with gradients). The terms are:

  | functional | V C_occ per grid block |
  |---|---|
  | LDA | ao (w ψ)ᵀ |
  | GGA (V = A + Aᵀ) | ao₀ Xᵀ + aow ψ₀ᵀ, with X = Σ_k w_k ψ_k |
  | meta-GGA | the GGA terms plus Σ_k ao_k (w_τ ψ_k)ᵀ |

  - That is a GEMM of nao_sub × nocc per density instead of nao_sub × nao_sub, two for GGA and five for meta-GGA.
  - The halving of w₀ (and w_τ) is unchanged.
- **Batching:** the densities go through cuBLAS in batches (about 256 MB of intermediates), in the layouts it takes
  directly. Through cuTENSOR, the (density, AO, occupied) output layout made one GEMM 1.5× slower.
- **Fallback:** blocks where GEMMs per density × nocc ≥ GEMMs × nao_sub keep the old integration. That happens in
  small bases and in sparse blocks of large molecules, where the old path is cheaper. The result is added through
  V C_occ at the end.
- **Exactness:** the same sums, in another order. The results agree with the old algorithm to 1e−16 in every case,
  and bit for bit where every block falls back (caffeine STO-3G, GGA).
- **Not covered:** the UKS Hessian uses the general `numint.nr_uks_fxc`, which TDDFT shares, and it is unchanged.

### Results (caffeine, GTX 1650 Super)
- **One call for 72 densities:**

  | basis | B3LYP | SVWN | r2SCAN |
  |---|---|---|---|
  | def2-SVP (nao 246) | 14.2 → 10.3 s (1.37×) | 9.4 → 3.4 s (2.76×) | 36.7 → 14.8 s (2.49×) |
  | 6-31G (nao 146) | 1.12× | 2.08× | 1.84× |
  | STO-3G (nao 80) | 1.00× (all blocks fall back) | 1.69× | 1.48× |

- **Hessians:**

  | caffeine def2-SVP Hessian | v1.9.0 | before | now | max \|ΔH\| against v1.9.0 |
  |---|---|---|---|---|
  | B3LYP | 306 s | 227 s (part 9) | 208 s | 2.0e−11 |
  | r2SCAN | 450 s | – | 339 s (1.33×) | 1.2e−11 |

  - The B3LYP CPHF's XC response went from 71 to 52 s.
  - Both Hessians kept their SCF cycles (15 and 16).
  - r2SCAN has no exact exchange, so its CPHF J still goes through the J-engine's 9 passes per 72 densities.
- **Tests:** the RHF, RKS, UHF and UKS Hessian tests pass (37; the RKS file covers LDA, PBE, B3LYP, M06, ωB97 and HF),
  as do `test_large_exponent.py` and `test_level_shift_hessian.py` (11, 1 skipped).
- **After the gate:** the NLC check moved before the loop and the GEMM counts were renamed. The A/Bs and the RKS
  Hessian tests were run again on that code.
- **Where the B3LYP Hessian stands (208 s):**

  | stage | time |
  |---|---|
  | CPHF XC response | 52 s |
  | two-electron second derivatives | 51 s |
  | CPHF J/K | 35 s |
  | XC derivative terms | 29 + 27 s |
  | J/K derivative matrices | 11 s |

### Next
- **J for many densities without K** (pure functionals such as r2SCAN or PBE): the J-engine re-evaluates its integrals
  per block of 8 densities. The fix would be a J-engine kernel change (integrals once, contraction per density), or a
  working Rys J-only path.
- **The XC derivative terms** (56 s, not profiled inside), **the second-derivative kernels** (51 s), **the
  TDDFT-gradient kernels**, and **the A100/H100 suite**.

## GPU4PySCF investigation, part 11 (profile only): the Hessian's XC derivative terms (node1), 2026-11-01
The user approved profiling the XC derivative terms (56 of the B3LYP Hessian's 208 s). Nothing was changed.

### Method (`hess_xc_parts.py`, `hess_xc_profile.txt`)
- **Calls:** `_get_vxc_diag` and `_get_vxc_deriv2` (the two halves of `_get_exc_deriv2`) and `_get_vxc_deriv1`
  (`make_h1`), each run once at the converged caffeine def2-SVP B3LYP SCF, on 294,912 grid points in 72 blocks.
- **Timers:** synchronizing, around every helper they call; `contract` is grouped by its subscripts.

### Where the time goes
| function | time | largest parts |
|---|---|---|
| `_get_vxc_deriv1` | 30.2 s | per-atom matrix elements `_d1_dot_` 18.5 s; AO transform to the original basis 2.1 s; `_gga_grad_sum_` 2.0 s; fxc · ∂ρ/∂R per atom 1.8 s; `_scale_ao` 1.7 s |
| `_get_vxc_deriv2` | 25.5 s | `contract('yjg,xjg->xyj')` 11.8 s; `_d1d2_dot_` 3.3 s; `_make_dR_dao_w` 2.2 s; AO transform 2.1 s; dm0 · AO 1.3 s |
| `_get_vxc_diag` | 3.6 s | two GEMM patterns, 1.7 and 1.1 s |

### Findings
- **`contract('yjg,xjg->xyj')` is pathologically slow** (in the GGA, LDA and meta-GGA paths of `_get_vxc_deriv2`, and in
  the UKS Hessian for both spins, with `'xjg,jg->xj'` next to it).
  - It computes nine dot products over the grid for each AO, inside the loop over atoms. That is 31 GFLOP in 1,728
    calls, at 2.6 GFLOP/s: through cuTENSOR, the AO index sits in both inputs and the output.
  - Timings for (3, 246, 4096) inputs (`xyj_bench.py`), with the 48 MB they move taking 0.25 ms at 192 GB/s:

    | method | time per call |
    |---|---|
    | cuTENSOR (now) | 9.6 ms |
    | nine products with `.sum(axis=-1)` | 1.8 ms |
    | batched `cupy.matmul` | 72 ms |
    | fused RawKernel, one thread block per AO | 0.28 ms |

  - All agree to 2e−15 relative. `cupy_helper.py` already compiles RawKernels.
- **`_get_vxc_deriv1`'s per-atom GEMMs** run at a normal 70 GFLOP/s, but on all nao AOs for every atom. Each block's AOs
  are transformed to the original basis, so its screening mask is lost and the atom slices stay contiguous.
  - Coverage per block (`atom_cover.py`):

    | molecule | atoms with AOs in a block | screened AOs |
    |---|---|---|
    | caffeine def2-SVP | 99% | 68% |
    | n-C20H42 def2-SVP | 53% | 35% |

  - Working on the screened AOs in the sorted basis would cut those GEMMs 1.5× for caffeine. Skipping atoms without AOs
    in the block (their ∂ρ/∂R is exactly zero) adds little for caffeine, but for the alkane the two together cut the
    FP64 work about 5×.
- **Redundancy:** `_get_vxc_deriv1` and `_get_vxc_deriv2` each recompute ρ, fxc, the AO transform and dm0 · AO on the
  same grid, about 4 s each.

### Options (exact; not started)
1. A fused kernel for the per-AO dot products: about 12 s of the B3LYP Hessian (6%), RKS and UKS. A small change.
2. `_get_vxc_deriv1` on the screened AOs, skipping atoms without AOs in the block: 1.5× on 18.5 s for caffeine, about 5×
   for larger molecules. A larger refactor (per-atom AO lists in the sorted basis).
3. One grid pass for both derivative terms: about 4 s. A larger refactor across `partial_hess_elec` and `make_h1`.

## GPU4PySCF investigation, part 12: the per-AO dot products of the Hessian's XC second derivatives in a fused kernel (node1), 2026-11-01
The user approved options 1 and then 2 of part 11; this is option 1.

### Change (gpu4pyscf `perf` 1bb59c8, merged into `exp/prim-screen` ffc0018)
- **`cupy_helper.dot_rows(a, b, out=None, beta=0.)`** computes out[x,y,j] = beta·out[x,y,j] + Σ_g a[x,j,g] b[y,j,g],
  i.e. `contract('xjg,yjg->xyj')`. With a 2-D b[j,g] it gives out[x,j].
  - **Kernel:** one 256-thread block per row j. Each thread sums the na·nb products over a stride of g, then a warp
    shuffle and shared memory reduce the 8 warps. One thread writes each output; beta = 0 overwrites.
  - **Caching:** a RawKernel is compiled per (na, nb) and cached in `_kernel_registery`, like the other helpers there.
  - **Requirements:** FP64, a contiguous last axis in a and b, and a contiguous j in out (all asserted).
- **Where it replaces cuTENSOR:** `_get_vxc_deriv2`, every per-atom `contract('yjg,xjg->xyj')`, `'xjg,jg->xj'` and
  `'jg,xjg->xj'`. That is 14 call sites in `hessian/rks.py` and 28 in `hessian/uks.py` (LDA, GGA, meta-GGA, both spins).
- **Exactness:** the same sums in another order.

### Results (caffeine def2-SVP, GTX 1650 Super; `hess_dot_rows.txt`)
- **`_get_vxc_deriv2` at identical orbitals against v1.9.0** (`vxc_deriv_check.py`, which was `vxc_deriv2_check.py`
  then; the orbitals are saved once and loaded by both trees):

  | | SVWN | B3LYP | r2SCAN |
  |---|---|---|---|
  | RKS | 18.4 → 7.0 s (2.61×) | 23.9 → 12.6 s (1.90×) | 37.3 → 25.9 s (1.44×) |
  | UKS (cation) | 36.6 → 13.9 s (2.63×) | 49.8 → 27.1 s (1.84×) | 77.9 → 54.8 s (1.42×) |

  - All agree to 3.4e−13 or better, with values up to 2e2.
- **Hessians** (with the `exp/prim-screen` kernels):

  | caffeine def2-SVP Hessian | v1.9.0 | before (part 10) | now | max \|ΔH\| against v1.9.0 |
  |---|---|---|---|---|
  | B3LYP | 306 s | 208 s | 197 s | 1.9e−11 |
  | r2SCAN | 450 s | 339 s | 327 s | 8.7e−12 |

  - The SCF cycle counts are unchanged (15 and 16).
  - `_get_exc_deriv2` went from 27.4 to 16.1 s.
- **Tests:** the RHF, RKS, UHF and UKS Hessian tests pass (37), as do `test_large_exponent.py` and
  `test_level_shift_hessian.py` (11, 1 skipped).
- **Where the B3LYP Hessian stands (197 s; `hess_dot_rows_stages.jsonl`):**

  | stage | time |
  |---|---|
  | CPHF (`solve_mo1`) | 89 s, of which XC response 52 s and J/K 35 s |
  | two-electron second derivatives | 51 s |
  | XC first derivatives of V (`_get_vxc_deriv1`) | 29 s |
  | XC second derivatives (`_get_exc_deriv2`) | 16 s |
  | J/K derivative matrices | 11 s |

## GPU4PySCF investigation, part 13: the Hessian's XC first derivatives on each grid block's own AOs (node1), 2026-11-01
Option 2 of part 11, approved together with option 1 (part 12).

### Change (gpu4pyscf `perf` 3c9ad0f, merged into `exp/prim-screen` 1b941ed)
- **The waste:** `_get_vxc_deriv1` (RKS and UKS) moved each grid block's AO values to all nao original AOs with
  `contract('nip,ij->njp', ao, coeff[mask])`. That is a GEMM with a permutation matrix, and it fills the AOs that are not
  on the block with zeros. The per-atom GEMMs (nao × nocc, three directions, two terms) then ran on all nao AOs, for every
  atom.
- **The block's own AOs:** the sorted AOs are a permutation of the original ones (`opt.coeff` is `eye(nao)[_ao_idx]`).
  - `_block_ao_subset` gathers the block's AO values in ascending original order with `cupy.take`. That is bit for bit what
    the GEMM gives.
  - It maps each atom's AO range into the subset with `searchsorted`, so the ranges stay contiguous and `_make_dR_rho1`'s
    slices still work.
  - The orbitals, `dm0` and the AO-pair term `v_ip` are taken on the subset (`mocc[idx]`, `take_last2d`, `add_sparse`).
    Each atom's (3, nao_sub, nocc) block is added at the subset's rows.
- **Skipping:** an atom without AOs on the block has dR_rho1 = 0 there, so all its terms are exactly zero, and it is
  skipped. Blocks with no AOs at all are skipped too.
- **Smaller changes in the same loop:**
  - fxc · dR_rho1 goes through `_contract_rho1_fxc`, as `_get_vxc_deriv2` already does, instead of cuTENSOR (1.05 →
    0.08 ms per atom and block). UKS does both spins in one call; this relies on fxc being symmetric.
  - Each atom's two GEMMs take the three directions in one call each (`_d1_dot_pair`).
  - The meta-GGA τ term has two equal halves, (ao_j w) mo_jᵀ and ao_j (mo_j w)ᵀ. It is now computed once with twice the
    weight, which halves the τ GEMMs.
- **Exactness:** the same sums, apart from terms that are exactly zero, in another order.
- **New tools:**
  - `vxc_deriv_check.py --func deriv1 [--mol c20h42]`.
  - `hess_parts.py --uks [--charge --spin]`.
  - `d1_gemm_shapes.py`, which times the per-atom GEMMs one atom at a time and batched over atoms.

### Results (GTX 1650 Super; `hess_deriv1.txt`)
- **`_get_vxc_deriv1` at identical orbitals against v1.9.0:**

  | | SVWN | B3LYP | r2SCAN |
  |---|---|---|---|
  | caffeine RKS | 22.6 → 14.2 s (1.59×) | 28.8 → 16.1 s (1.79×) | 89.0 → 32.8 s (2.71×) |
  | caffeine⁺ UKS | 35.3 → 28.3 s (1.24×) | 46.8 → 32.2 s (1.45×) | 139 → 66 s (2.11×) |
  | n-C20H42 RKS | – | 423 → 71 s (5.98×) | – |
  | n-C20H42⁺ UKS | – | 655 → 141 s (4.66×) | – |

  - All agree to 2.4e−14 or better, with values up to 2.6.
  - The alkane gains the most: its blocks hold 35% of the AOs and 53% of the atoms (part 11).
  - The RKS gains are larger because v1.9.0's RKS path was the slower one. Its UKS SVWN run takes only 1.56× its RKS
    run, with twice the orbitals; here the ratio is 2.0×. The likely cause, not profiled, is the RKS path's cuTENSOR
    call with a transposed output and beta = 1.
- **Hessians** (caffeine def2-SVP, with the `exp/prim-screen` kernels):

  | Hessian | v1.9.0 | before (part 12) | now | max \|ΔH\| against v1.9.0 |
  |---|---|---|---|---|
  | B3LYP | 306 s | 197 s | 184 s | 1.6e−11 |
  | r2SCAN | 450 s | 327 s | 272 s | 1.1e−11 |

  - The SCF cycle counts are unchanged (15 and 16).
- **UKS Hessians end to end** (benzene def2-SVP, both trees):
  - **Neutral benzene with UKS (spin 0)** agrees with v1.9.0 to 3.7e−12 (SVWN), 1.3e−12 (B3LYP) and 3.5e−12 (r2SCAN),
    with the same cycle counts. The times go 40.3 → 33.9, 62.5 → 44.4 and 85.4 → 75.6 s.
  - **The benzene cation is no test:**
    - Its SVWN SCF does not converge in either tree (50 cycles).
    - Its B3LYP and r2SCAN Hessians differ from v1.9.0 by 7.4e−10 and 3.5e−7. v1.9.0 differs from itself on a second
      run by 1.3e−10 and 1.3e−7.
    - This Jahn–Teller system amplifies the GPU atomics' noise. The differences between trees are of that size.
- **Tests:** the RHF, RKS, UHF and UKS Hessian tests pass (37), as do `test_large_exponent.py` and
  `test_level_shift_hessian.py` (11, 1 skipped).
- **Where `_get_vxc_deriv1` stands** (caffeine B3LYP, 17.2 s with synchronizing timers):

  | part | time |
  |---|---|
  | the per-atom GEMM pairs | 11.3 s (65%) |
  | `_scale_ao` | 1.3 s |
  | `v_ip` (`_gga_grad_sum_`) | 1.2 s |
  | `_make_dR_rho1` | 1.1 s |
  | dm0 · AO | 0.7 s |

  - For r2SCAN (35 s), the τ GEMMs add about 13 s.
- **Batching the per-atom GEMMs over atoms** (`d1_gemm_shapes.py`, one mean-sized block):

  | molecule | one atom at a time | batched over atoms |
  |---|---|---|
  | caffeine | 62 GFLOP/s | 94 GFLOP/s (1.51×) |
  | n-C20H42 | 71 GFLOP/s | 87 GFLOP/s (1.23×) |

  - The card's FP64 peak at 1530 MHz is 122 GFLOP/s.
  - Each atom's GEMMs are skinny (3·168 × 51 and 168 × 153 outputs, K = 4096), so they cover few output tiles. On GPUs with
    many more SMs than this card's 20, one GEMM per atom should leave more of the GPU idle; that is unmeasured.
- **Where the B3LYP Hessian stands (184 s; `hess_deriv1_stages.jsonl`):**

  | stage | time |
  |---|---|
  | CPHF (`solve_mo1`) | 89 s, of which XC response 52 s and J/K 35 s |
  | two-electron second derivatives | 51 s |
  | XC first derivatives of V | 16 s |
  | XC second derivatives | 16 s |
  | J/K derivative matrices | 11 s |

### Options (exact; not started)
1. **`_get_vxc_deriv2` on each block's own AOs, skipping atoms without AOs** (part 13's change in the second-derivative
   term, RKS and UKS).
   - What shrinks: its AO transform (2.1 s), the AO-pair GEMMs `_d1d2_dot_` over all nao² (3.3 s for B3LYP, 13.1 s for
     r2SCAN) and dm0 · AO (1.3 s).
   - Gain: about 4.6 of 14.0 s for caffeine B3LYP and 10 of 28.5 s for r2SCAN.
   - For larger molecules the per-atom loop also shrinks by the share of atoms absent from each block (47% for n-C20H42).
2. **Batch `_get_vxc_deriv1`'s per-atom GEMMs over atoms,** in chunks sized from free memory.
   - On this card: 1.5× (caffeine) and 1.2× (n-C20H42) on those GEMMs, about 4 s of B3LYP's 17 s, plus the meta-GGA τ
     GEMMs.
   - Probably more on GPUs with many SMs; that needs the A100/H100 suite.
3. **One grid pass for both derivative terms:** about 4 s (part 11).

## GPU4PySCF investigation, part 14: the Hessian's XC second derivatives on each grid block's own AOs, the atoms' terms in one GEMM (node1), 2026-11-01
Option 1 of part 13, approved by the user. Change A is that option. Change B, a separate commit, goes further in the
same function: it replaces the per-atom loop that remained after A. Change C fixes a memory side effect of part 13 that
B's first version brought to light.

### Change A (gpu4pyscf `perf` 21bc83a, merged into `exp/prim-screen` 4469f87)
- **Part 13's change in `_get_vxc_deriv2`** (RKS and UKS). Each grid block gathers its own AOs in the original order
  (`_block_ao_subset`) instead of a GEMM with the permutation matrix `opt.coeff`.
- **The AO-pair term `ipip`** was nine GEMMs of nao × nao per block (36 for meta-GGAs). It is now one GEMM of
  3·nao_sub × 3·nao_sub (`_d1d2_dot_gemm`), plus three for the meta-GGA τ part, added at the block's AOs
  (`_add_d1d2_`, through `add_sparse`).
- **dm0 · AO** was computed twice per block: for all nao rows in the original basis (for dR_rho1) and on the mask in
  the sorted basis (for the matrix elements). It is now computed once, on the block's AOs.
- **Skipping:** atoms without AOs on the block are skipped (their dR_rho1 is zero there), and the results are
  accumulated in the original basis, so the final unsort goes away.
- **UKS:** fxc · dR_rho1 goes through `_contract_rho1_fxc` instead of cuTENSOR's `'xbyg,sxg->bsyg'` (as part 13 did in
  `_get_vxc_deriv1`).

### Change B (gpu4pyscf `perf` d318e8a, merged into `exp/prim-screen` 581c671)
- **What remained after A:** a loop over the atoms of each block. Per atom it built dR_rho1 (21 row contractions plus
  their adds for GGA), contracted it with fxc, and formed the atom's terms in vmat_dm with three `_make_dR_dao_w`,
  three `_scale_ao` and four `dot_rows` passes over the block's AO values (13 `dot_rows` for meta-GGAs). All of these
  are bound by memory bandwidth, and their number grows with the atoms per block.
- **The identity:** both halves are made of the same per-AO products. Let t[x,j,c,g] be AO j's terms of dR_rho1
  (direction x, density component c, grid point g):

  | component | t[x,j,c] |
  |---|---|
  | ρ | 2 ∂_xφ_j D0_j |
  | ∇_c ρ | 2 (∂_x∂_cφ_j D0_j + ∂_xφ_j D_c,j) |
  | τ | Σ_c ∂_x∂_cφ_j D_c,j |

  with D_c,j = Σ_i ∂̃_cφ_i dm0[i,j]. Then dR_rho1 of atom A is Σ_{j on A} t[:,j], and the atom's terms in vmat_dm are
  v[A,i,x,j] = ½ Σ_{c,g} (fxc · dR_rho1_A)[i,c,g] t[x,j,c,g], for LDA, GGA and meta-GGA alike. The one factor ½
  combines the old code's ½ on the ρ and τ components with the 2 of t's ∇ρ terms.
- **One GEMM per block:** for all atoms with AOs on a block, the terms are one GEMM of (3·natm_blk) × (3·nao_sub)
  with K = ncomp × ngrids. t takes one kernel (`_dR_rho1_terms`), the atoms' sums another (`_atom_sums`), and the fxc
  contraction a third (`_contract_rho1_fxc_rows`).
- **Memory:** t is built for grid sub-blocks of 4096/ncomp points, so it never exceeds three (nao_sub, 4096) arrays
  and fits in the block's `aow` buffer. The fxc kernel has no temporaries (`_contract_rho1_fxc` makes
  (rows, nv, nv, ngrids) ones). The UKS path, which allocated its arrays per block, uses fixed buffers like the RKS
  path.
  - A first version built t for the whole block and allocated it per block. It was as fast, but CuPy's memory pool
    peaked at 2.7–3.2× v1.9.0's (859 against 319 MB for caffeine B3LYP; 2056 against 646 MB for the UKS alkane):
    t's size changes with each block's AO count, so the freed blocks did not fit the next one.
- **Exactness:** the same sums, in another order.
- **Prototype** (`d2_atoms_gemm.py`, random inputs shaped like a caffeine block): the GEMM form agrees with the
  per-atom form to 2e−15 relative. Per block it takes 22 ms (GGA) and 28 ms (meta-GGA), the GEMM 90% of that, at
  60 GFLOP/s: the M dimension is only 3 × 24 atoms. Other GEMM orientations or K splits change it by less than 10%
  (`d2_gemm_orient.py`).

### Results (GTX 1650 Super; `hess_deriv2.txt`)
- **`_get_vxc_deriv2` at identical orbitals** (`vxc_deriv_check.py`). v1.9.0 ran interleaved with the first version of
  B; the B column is the final version, run right after it (0–5% faster than the first). Part 12 and A are from
  their own runs:

  | | v1.9.0 | part 12 | A | B |
  |---|---|---|---|---|
  | caffeine RKS SVWN | 18.4 s | 7.0 s | 4.0 s | 2.45 s (7.5×) |
  | caffeine RKS B3LYP | 23.9 s | 12.6 s | 7.4 s | 4.56 s (5.2×) |
  | caffeine RKS r2SCAN | 37.3 s | 25.9 s | 15.9 s | 10.1 s (3.7×) |
  | caffeine⁺ UKS SVWN | 36.6 s | 13.9 s | 8.5 s | 4.84 s (7.6×) |
  | caffeine⁺ UKS B3LYP | 49.9 s | 27.1 s | 15.6 s | 9.03 s (5.5×) |
  | caffeine⁺ UKS r2SCAN | 77.9 s | 54.8 s | 34.8 s | 20.1 s (3.9×) |
  | n-C20H42 RKS B3LYP | 160 s | – | – | 11.6 s (13.8×) |
  | n-C20H42⁺ UKS B3LYP | 331 s | – | – | 22.7 s (14.6×) |

  - The final outputs agree with v1.9.0 to 5.4e−13 or better, compared directly (values up to 2.2e2; the UKS alkane
    through the first version of B, 1.1e−13 + 3.6e−14). A agrees with part 12 to 5.4e−13, B with A to 2.6e−13.
  - The alkane gains the most: its blocks hold 35% of its AOs and 53% of its atoms (part 11).
- **Memory** (`deriv_mem.py`: the peak in use is tracked at every allocation of CuPy's pool; the pool also holds the
  freed blocks it caches):

  | `_get_vxc_deriv2` | peak in use, v1.9.0 → B | pool, v1.9.0 → B |
  |---|---|---|
  | caffeine RKS SVWN / B3LYP / r2SCAN | 150 / 269 / 294 → 117 / 234 / 262 MB | 185 / 301 / 325 → 148 / 264 / 288 MB |
  | caffeine⁺ UKS SVWN / B3LYP / r2SCAN | 241 / 364 / 418 → 155 / 302 / 307 MB | 421 / 667 / 611 → 188 / 329 / 346 MB |
  | n-C20H42 RKS / UKS B3LYP | 455 / 561 → 385 / 539 MB | 507 / 852 → 442 / 573 MB |

  - The first version of B had the same timings but pool peaks of 2.7–3.2× v1.9.0's (see Change B).
- **Hessians** (caffeine def2-SVP, with the `exp/prim-screen` kernels):

  | Hessian | v1.9.0 | before (part 13) | now | max \|ΔH\| against v1.9.0 |
  |---|---|---|---|---|
  | B3LYP | 306 s | 184 s | 177 s | 2.1e−11 |
  | r2SCAN | 450 s | 272 s | 256 s | 7.4e−12 |

  - The SCF cycle counts are unchanged (15 and 16). `_get_exc_deriv2` went from 16.1 to 8.1 s (B3LYP).
  - Neutral benzene with UKS (spin 0) agrees with v1.9.0 to 2.9e−12 (SVWN), 2.4e−12 (B3LYP) and 5.4e−12 (r2SCAN),
    with the same cycle counts. The times go 40.3 → 32.6, 62.5 → 41.7 and 85.4 → 71.3 s.
- **Tests:** the RHF, RKS, UHF and UKS Hessian tests pass (37), as do `test_large_exponent.py` and
  `test_level_shift_hessian.py` (11, 1 skipped), after each of A, B and C.
- **Where `_get_vxc_deriv2` stands** (`hess_xc_parts.py`, synchronizing timers):

  | part | B3LYP (4.7 s) | r2SCAN (10.3 s) |
  |---|---|---|
  | AO-pair GEMMs `_d1d2_dot_gemm` (3·nao_sub square, K = 4096) | 1.66 s | 6.63 s (with the three τ GEMMs) |
  | the atoms' terms `_dR_rho1_vmat` | 1.70 s (GEMM 1.56 s) | 2.13 s (GEMM 1.95 s) |
  | dm0 · AO | 0.70 s | 0.70 s |
  | ρ, fxc, AO values | 0.48 s | 0.55 s |

  - Taking every block at the mean of 167 AOs, the AO-pair GEMMs run at about 90 GFLOP/s (the card's FP64 peak at
    1530 MHz is 122) and the atoms' GEMM at about 55: its M dimension is 3 × the atoms of a block.
  - The meta-GGA τ part of the AO-pair term is symmetric, so cuBLAS's `dsyrkx` (one triangle) would halve its three
    GEMMs, about 2.5 s for r2SCAN. CuPy does not wrap it.
- **Where the B3LYP Hessian stands (177 s; `hess_deriv2_stages.jsonl`):**

  | stage | time |
  |---|---|
  | CPHF (`solve_mo1`) | 89 s, of which XC response 52 s and J/K 35 s |
  | two-electron second derivatives | 51 s |
  | XC first derivatives of V | 16 s |
  | J/K derivative matrices | 11 s |
  | XC second derivatives | 8 s |
- **Tools, new or changed:**
  - `deriv_mem.py`: the peak memory in use and the pool's size for `_get_vxc_deriv1|2`.
  - `vxc_deriv_check.py` also prints the pool's high-water mark and compares flattened outputs. The UKS references
    of part 12 were saved unflattened.
  - `hess_xc_parts.py` lists a timed call made inside another under it, with the outer call's own rest.
  - `d2_atoms_gemm.py` and `d2_gemm_orient.py`: B's prototype and its GEMM shapes.

### Change C (gpu4pyscf `perf` 87d85e8, merged into `exp/prim-screen` 5ef6c7d)
- **Fixed buffers in the UKS `_get_vxc_deriv1`.** B's first version showed what per-block arrays sized by each
  block's AO count do to CuPy's pool. Part 13's UKS `_get_vxc_deriv1` has the same pattern (`ao[:,order]`, dm0 · AO,
  `aow`): its pool peaked at 789 MB for caffeine⁺ B3LYP and 782 MB for r2SCAN, against v1.9.0's 403 and 395 MB, while
  its peak in use was lower (302 against 347 MB). v1.9.0 allocated these arrays at the full nao, the same size every
  block.
- **The change:** those arrays go in buffers allocated once, as in the RKS path. The results are bit for bit the same.
- **Results** (`hess_deriv2.txt`, section 8):

  | UKS `_get_vxc_deriv1` | time | pool, v1.9.0 / part 13 / now | peak in use, v1.9.0 / part 13 / now |
  |---|---|---|---|
  | caffeine⁺ SVWN | 28.4 s | – / – / 204 MB | – / – / 168 MB |
  | caffeine⁺ B3LYP | 31.9 s | 403 / 789 / 377 MB | 347 / 302 / 344 MB |
  | caffeine⁺ r2SCAN | 66.4 s | 395 / 782 / 477 MB | 350 / 310 / 353 MB |
  | n-C20H42⁺ B3LYP | 140.6 s | 778 / – / 708 MB | 676 / – / 660 MB |

  - The peak in use goes back to about v1.9.0's: the buffers are sized for all AOs.
  - The meta-GGA pool stays above v1.9.0's, most likely from `rks_grad._tau_grad_dot_` (shared with the gradient
    code), which gathers (3, nao_sub, ngrids) copies with fancy indices on every call. The RKS meta-GGA path has the
    same excess (397 against 301 MB) at the same peak in use.
  - The outputs are identical to part 13's (all differences exactly zero), at the same times.
  - Neutral benzene UKS Hessians agree with v1.9.0 to 4.6e−12 (SVWN, r2SCAN) and 2.1e−12 (B3LYP); the Hessian tests
    pass (37; 11 and 1 skipped).

### Options (exact; not started)
1. **Batch `_get_vxc_deriv1`'s per-atom GEMMs over atoms** (part 13's option 2). They take 11.4 of its 16 s for
   caffeine B3LYP, at 64 GFLOP/s one atom at a time; batched they ran at 94 GFLOP/s (`d1_gemm_shapes.py`), about 4 s
   less. Probably more on GPUs with many SMs, to measure in the A100/H100 suite.
2. **Profile the two-electron second derivatives** (`_partial_ejk_ip2`, 51 s, the largest stage outside the CPHF):
   which integral classes and kernels take the time, and whether the generated kernels of parts 7 and 8 carry over.
   A profile first, as in part 11.
3. **The CPHF** (89 s): its J/K (35 s in 7 calls) has not been profiled since part 9; its XC response (52 s) was
   near the FP64 floor in part 10.
4. **Small items:**
   - `dsyrkx` for the symmetric meta-GGA τ part of `_get_vxc_deriv2`'s AO-pair term: about 2.5 s of r2SCAN's 10.3 s.
     It needs a ctypes call into cuBLAS.
   - `_tau_grad_dot_` without the fancy-index copies (the meta-GGA pool excess above).

## GPU4PySCF investigation, part 15 (profile only): the Hessian's two-electron second derivatives (node1), 2026-11-01
Option 2 of part 14, approved by the user: where `_partial_ejk_ip2` spends its time, and whether part 8's generated
kernels carry over. Nothing was changed.

### What was known (`hess_ejk2_profile.txt`, taken 2026-10-31 and not written up until now)
- **The stage:** 50.8 s at the converged caffeine def2-SVP B3LYP density, all in `RYS_per_atom_jk_ip2_type12`
  (28.0 s; ∂²i, ∂i∂j, ∂²j on the bra, the same on the ket) and `_type3` (22.8 s; ∂i∂k, ∂i∂l, ∂j∂k, ∂j∂l).
- **Kernels:** every class with d functions runs GPU4PySCF's generic kernels (38.0 s, 75%); the s and p classes run
  its unrolled ones (12.7 s).
- **The largest classes**, type12 + type3: (dp|ps) 7.5 s, (dp|pp) 6.8 s, (ds|pp) 3.8 s, (pp|pp) 3.6 s, (ds|ps)
  3.4 s, (dp|dp) 3.2 s.
- **Bound:** FP64. The FP64 pipe is 80–91% busy in all of them, although FP64 is only 24–39% of the generic kernels'
  instructions (57–83% of the unrolled ones').

### Method (`ncu_ejk2_class.py`, `ncu_source_regions.py`; `hess_ejk2_source.txt`)
- **Launches:** one per class and kernel (the first of each in the stage's order), 16 classes, under Nsight
  Compute's SourceCounters. The library is built with `-lineinfo`.
- **Attribution:** the source page (`--print-source cuda,sass`) gives each SASS instruction's executions and
  warp-stall samples. An instruction is listed under every line of its inlining stack; it is counted once, at its line
  in `rys_contract_jk_ip2.cu`.
- **Regions:** setup and task screening, density products, primitive pairs, Rys roots, the g recursions
  (`BUILD_4C_GXYZ`), the output phase (the loop over outputs inside the loop over roots), atomics.
- **Iterations:** the output phase's are counted by a line that runs once per output and root (the density
  product's load).

### What the generic kernels do per output and root
- **Per direction:** type12 builds the first-derivative factors of i, j, k and l and the second-derivative factors
  ii, ij, jj, kk, kl and ll, then adds 42 sums. type3 builds the first-derivative factors and the mixed factors ik,
  il, jk and jl, then adds 36 sums.
- **Measured,** nearly the same in all 11 classes:

  | kernel | warp instructions per output and root | of them FP64 |
  |---|---|---|
  | type12 | 545–571 | 178–190 |
  | type3 | 454–502 | 152–173 |

  - The gradient's generic kernel spends about 42 FP64 per output (part 8).
- **The other two thirds** are the outputs' Cartesian powers read from tables, address arithmetic, the conditional
  terms of zero powers, and about 30 shared-memory loads of g values.

### Where the time goes (shares of the warp-stall samples; `hess_ejk2_source.txt`, section 2)
| class | kernel (block) | output phase | g recursions | roots, primitive pairs |
|---|---|---|---|---|
| (dp\|ps) | type12 (16×16) / type3 (64×4) | 82% / 93% | 14% / 5% | 4% / 2% |
| (dp\|pp) | type12 (16×16) / type3 (32×8) | 94% / 96% | 5% / 3% | 1% / 1% |
| (dp\|dp) | type12 (8×32) / type3 (16×16) | 95% / 96% | 4% / 3% | 1% / 1% |
| (dd\|pp) | type12 (8×32) / type3 (16×16) | 94% / 96% | 4% / 3% | 1% / 1% |
| (ds\|ps) | type12 (32×8) / type3 (64×4) | 74% / 80% | 20% / 14% | 6% / 6% |
| (dp\|ss) | type12 (32×8) / type3 (64×4) | 73% / 38% | 21% / 41% | 6% / 20% |
| (ds\|ss) | type12 (32×8) / type3 (128×2) | 41% / 16% | 43% / 55% | 16% / 28% |

- **The output phase** takes 82–97% of the time in the classes that carry most of it.
- **In the smallest generic classes the g recursions take 41–55%.** Only one thread per quartet runs
  `BUILD_4C_GXYZ` (`if (gout_id == 0)`), and the others wait at the barrier. The blocks give a quartet 2 to 64
  threads.

### Do part 8's sums per tuple carry over? (`ip2_tuple_model.py`)
- **Same-direction terms** (xx, yy, zz: 18 of type12's 42 sums, 12 of type3's 36) work exactly as in the gradient.
  With S_x(t) = Σ dd · Iy · Iz over the outputs with x tuple t, each term is Σ_t F(t) S_x(t). F is the tuple's
  second-derivative factor: ii, ij, ... for type12, ik, il, ... for type3.
- **Mixed terms** (xy, xz, yz: 24 sums in each kernel) save less, such as Σ f2x · f1y · Iz · dd.
  - An output's x and y tuples fix its z tuple, so no sum is shared over pairs of tuples.
  - Grouped by the x tuple, T(t_x) = Σ f1y(t_y) · Iz · dd costs one FMA per output.
  - The four factors of the other direction (type12: f1 and f2 of the bra, f1′ and f2′ of the ket; type3: fi, fj,
    fk, fl) give four such sums per pair of directions. Each tuple multiplies them by its own factors, 8 FMAs per
    pair.
- **Count:**
  - Per output and root: 20 FP64 in three passes (by x with S_x, T_xy and T_xz; by y with S_y and T_yz; by z with
    S_z), if the other directions' factors come from per-tuple tables. 44 if each output recomputes them from its
    g values.
  - Per tuple and root: about 150 FP64 (type12) or 132 (type3) for its factors and contributions.
  - Against about 180 (type12) and 160 (type3) FP64 per output now:

  | class | outputs | tuples per direction | type12 FP64 per root, now → with tables |
  |---|---|---|---|
  | (ds\|ps) | 18 | 6 | 3,258 → 1,260 (2.6×) |
  | (dp\|ps) | 54 | 12 | 9,774 → 2,880 (3.4×) |
  | (dp\|pp) | 162 | 24 | 29,322 → 6,840 (4.3×) |
  | (dp\|dp) | 324 | 36 | 58,644 → 11,880 (4.9×) |
  | (dd\|dd) | 1,296 | 81 | 234,576 → 38,070 (6.2×) |

- **Predicted time:** the output phase's time scaled by its FP64 ratio, the rest unchanged.
  - The 11 profiled generic classes (33.0 of the 38.0 s) go to 12.8 s (2.6×) with tables, or 16.7 s (2.0×)
    recomputing the factors. For all generic classes that means 38.0 → about 15–19 s.
  - The unrolled s and p classes (12.7 s) would gain 1.5–3.6× by the same count ((ps|ss) to (pp|pp)), if their cost
    per output is like the generic kernels'. That was not measured: they have no iteration marker.
  - So the stage would go from 51 s to roughly 20–28 s, and the caffeine B3LYP Hessian from 177 s to about 150 s,
    on this card.
- **Caveat:** the estimate counts FP64 only. Generated code would also drop most of the index arithmetic, the other
  two thirds of the instructions. Part 8's generated kernels executed 4.2× fewer warp instructions on the gradient's
  generic classes. That matters more on GPUs where FP64 is not the bottleneck.

### What the work would be
- **A generator for the ip2 kernels** on the pattern of `gen_rys_ejk.py` (part 8). The multi-thread template, the
  block shapes, the runs of outputs per thread and the warp-uniform lane scheme carry over. The recursions come from
  `gen_rys_k.recursions`: type12 needs i+1, j+1, k+1 and l+1, with the HRR for the second derivatives, as now. Threads
  0–2 run them one direction each, which also helps the smallest classes.
- **New:**
  - the second-derivative and mixed factors per tuple;
  - the four partial sums per pair of directions;
  - 42 (type12) or 36 (type3) sums per thread instead of the gradient's 12, which raises register pressure;
  - the tables of the other directions' factors, 3 × tuples × 4 values per quartet in shared memory: 2.3 KB for
    (dp|pp), 7.8 KB for (dd|dd).
- **Validation as in part 8:** the stage at a fixed density against v1.9.0 (whose own rounding is about 1e−13
  relative), Hessians end to end, the Hessian tests.
- **Branch:** like parts 7 and 8, CUDA sources on `exp/prim-screen`, rebuilt on node1.

### Options (exact; not started)
1. **Generated ip2 kernels with sums per tuple:** about 20–30 s of the caffeine B3LYP Hessian's 177 s on this card,
   probably more on GPUs with fast FP64. A job the size of part 8. Start with the generic classes of both kernels,
   then measure whether the unrolled s and p classes are worth replacing.
2. **Batch `_get_vxc_deriv1`'s per-atom GEMMs over atoms** (part 13): about 4 s.
3. **Profile the CPHF's J/K** (35 s in 7 calls), not looked at since part 9.

## GPU4PySCF investigation, part 16: generated kernels for the Hessian's two-electron second derivatives (node1), 2026-11-01
Option 1 of part 15, approved by the user: part 8's sums per Cartesian tuple in generated kernels for
`RYS_per_atom_jk_ip2_type12` and `_type3`.

### The generator (`bench/gpu/gen_rys_ejk_ip2.py`)
- **Sums per tuple:**
  - Same-direction terms (xx, yy, zz) are formed as in the gradient: the tuple's second-derivative factor times
    S_d(t), the sum of dd · I_e · I_f over the tuple's outputs.
  - Mixed terms are grouped by one direction's tuple. Four partial sums T_b(t_d) = Σ B_b(t_e) · I_f · dd, one per
    first-derivative factor B_b (fi, fj, fk, fl) of the other direction, give all eight terms of a pair of
    directions: type12 ij, ji, ii, jj, kl, lk, kk, ll; type3 ik, ki, jk, kj, il, li, jl, lj.
- **Per root:**
  - Threads 0–2 run the TRR and HRR of one direction each, through `gen_rys_k.recursions`. type12 uses
    (li+1, lj+1, lk+1, ll+1) and type3 (li+1, lj, lk+1, ll), the generic kernels' layouts. The ∂² factors use the
    HRR for the +2 shifts, as there.
  - All threads of a quartet then build a shared table of the four first-derivative factors of every tuple and
    direction.
  - Three passes follow: by x tuples (with the pair x, y), by y (y, z) and by z (z, x). Each output costs one MUL
    (dd · I_f), an FMA for S, and four table loads and FMAs for the T's: 6 FP64 and 7 loads per pass. At each tuple's
    end, the second-derivative factors are computed from g (the formulas of the generic kernels) and applied with the
    eight mixed terms.
- **Sums per thread:** type12 has 42 (its bra-pair sums ij, ii and jj are added once per bra pair), type3 has 36.
  They take 254–255 registers, with 0–260 bytes of spills.
- **Shapes:** 256 threads per block, with the most quartets per block (8 to 64) whose g arrays, factor table and
  roots fit in 64 KB. Warps that hold two or four gout_ids use part 8's per-lane addresses.
- **Integration** (`exp_prim_screen.py`):
  - `--gen-ip2` defaults to the 38 kernels below; `--gen-ip2-files 4` sets the translation units.
  - `RYS_per_atom_jk_ip2_type12/type3` try `rys_ejk_ip2_type12/3_generated()` first, then the unrolled and generic
    kernels.
  - `--gen-ip2-env` builds read `GPU4PYSCF_RYS_GEN_IP2_OFF=1`, so both kernels can be compared in one build.
- **Not generated:** (dd|dd), and type12 (dd|dp) and (dp|dd), do not fit even 8 quartets in 64 KB (0.95 s of the 51).
- **Exactness:** the terms are the same, summed in another order.

### Per class (`ejk2_class_check.py`)
Only that class's launches execute. The generated kernels run against the other kernels in the same
`--gen-ip2-env` build.
- **Classes with d functions** (`hess_ejk2_gen.txt`, section 1; type12 + type3; before, GPU4PySCF's generic kernels):

  | class | before | generated | | class | before | generated |
  |---|---|---|---|---|---|---|
  | (dp\|ps) | 7.52 s | 2.46 s (3.05×) | | (dd\|ps) | 1.60 s | 0.59 s (2.72×) |
  | (dp\|pp) | 6.80 s | 2.45 s (2.78×) | | (dp\|ds) | 1.62 s | 0.61 s (2.64×) |
  | (ds\|pp) | 3.83 s | 1.24 s (3.09×) | | (ds\|ss) | 1.33 s | 0.77 s (1.72×) |
  | (ds\|ps) | 3.41 s | 1.29 s (2.66×) | | (ds\|dp) | 1.41 s | 0.48 s (2.91×) |
  | (dp\|dp) | 3.19 s | 0.98 s (3.27×) | | (ds\|ds) | 0.89 s | 0.34 s (2.62×) |
  | (dp\|ss) | 1.88 s | 0.71 s (2.64×) | | (dd\|dp), (dp\|dd) | 1.45 s | 0.96 s (type3 only) |
  | (dd\|pp) | 1.77 s | 0.57 s (3.10×) | | (dd\|ss), (dd\|ds), (ds\|dd), (dd\|dd) | 1.56 s | 0.65 s |

  - In all: 38.3 → 14.1 s (2.71×). (dd|dd) is not generated; (dd|dp) and (dp|dd) only for type3. Every class agrees
    with the generic kernels to 1e−13 relative or better.
  - The 11 classes profiled in part 15 went 33.0 → 11.9 s. Its FP64-only model predicted 12.8 s; the index
    arithmetic the generated code drops also counts.
  - With both spins' densities (n_dm = 2, the UKS path) (dp|ps) goes 7.15 → 2.33 s, within 2.5e−14.
- **s and p classes** (section 2; before, GPU4PySCF's unrolled kernels):

  | class | type12 | type3 |
  |---|---|---|
  | (pp\|pp) | 2.51 → 0.76 s (3.29×) | 1.12 → 0.70 s (1.58×) |
  | (pp\|ps) | 1.60 → 1.39 s (1.15×) | 1.34 → 0.83 s (1.62×) |
  | (ps\|pp) | 0.98 → 0.78 s (1.25×) | 0.94 → 0.51 s (1.85×) |
  | (ps\|ps) | 1.05 → 1.38 s (0.76×) | 0.91 → 1.18 s (0.77×) |
  | (pp\|ss) | 0.56 → 0.70 s (0.80×) | 0.50 → 0.62 s (0.80×) |
  | (ps\|ss) | 0.59 → 1.09 s (0.54×) | 0.51 → 0.88 s (0.58×) |
  | (ss\|ss) | 0.20 → 0.47 s (0.42×) | 0.18 → 0.37 s (0.47×) |

  - The first three are generated in the final build. The others keep the unrolled one-thread kernels: with 1–9
    outputs per quartet, a multi-thread kernel's barriers and table cost more than the passes save.

### Results (GTX 1650 Super at 1530 MHz; `hess_ejk2_gen.txt`)
- **The stage at the converged caffeine def2-SVP B3LYP density** (`deriv_at_dm.py`; this tree median of 3, v1.9.0 one
  run):

  | build | v1.9.0 | generated | max \|Δ\| against v1.9.0 |
  |---|---|---|---|
  | B3LYP's J + 0.2 K | 53.2 s | 22.8 s (2.34×) | 1.3e−11 (values up to 53) |
  | short-range K (ω = −0.3) | 76.8 s | 31.6 s (2.43×) | 1.0e−13 (up to 0.71) |
  | long-range K (ω = 0.3) | 52.5 s | 22.3 s (2.36×) | 5.5e−14 (up to 0.27) |

  - Before this part the tree took 50.8 s for the first build, and differed from v1.9.0 by 1.3e−12 there (part 6's
    launch of the one-thread kernels).
  - **The 1.3e−11 is v1.9.0's own rounding** (section 3). The large elements are sums of class contributions up to
    about 290 that cancel to 53, accumulated by atomic adds whose order changed.
    - Summed over the atoms, which is zero by translational invariance, v1.9.0's result reaches 1.2e−11 (the first
      build), 3.0e−13 (short range) and 5.6e−14 (long range). The generated kernels reach 6.5e−12, 3.4e−13 and
      4.0e−14.
    - v1.9.0 against itself differs by 2.0e−13, 8.2e−15 and 1.2e−15.
- **Hessians** (caffeine def2-SVP, with the `exp/prim-screen` kernels):

  | Hessian | v1.9.0 | before (part 14) | now | max \|ΔH\| against v1.9.0 |
  |---|---|---|---|---|
  | B3LYP | 306 s | 177 s | 149 s | 1.8e−11 |
  | r2SCAN | 450 s | 256 s | 229 s | 1.9e−11 |

  - The SCF cycle counts are unchanged (15 and 16).
  - Neutral benzene with UKS agrees with v1.9.0 to 3.0e−12 (SVWN), 2.5e−12 (B3LYP) and 3.6e−12 (r2SCAN), with the
    same cycle counts. The times go 32.6 → 28.7, 41.7 → 37.5 and 71.3 → 67.2 s.
- **Tests:** the RHF, RKS, UHF and UKS Hessian tests pass (37), as do `test_large_exponent.py` and
  `test_level_shift_hessian.py` (11, 1 skipped). They include range-separated functionals.
- **Where the B3LYP Hessian stands (149 s; `hess_ejk2_gen_stages.jsonl`):**

  | stage | time |
  |---|---|
  | CPHF (`solve_mo1`) | 89 s, of which XC response 52 s and J/K 35 s |
  | two-electron second derivatives | 23 s (51 s before) |
  | XC first derivatives of V | 16 s |
  | J/K derivative matrices (`_get_jk_ip1`) | 11 s |
  | XC second derivatives | 8 s |

- **New tools:**
  - `gen_rys_ejk_ip2.py`, the generator.
  - `ejk2_class_check.py`, which runs the stage one class at a time against a reference or the run-time switch.
  - `deriv_at_dm.py --parts ejk2` now also builds the short- and long-range K.
  - `exp_prim_screen.py --gen-ip2`.

### Options (exact; not started)
1. **Profile the CPHF's J/K** (35 s in 7 calls of 72 densities), not looked at since part 9's one-pass J+K. It is
   now the largest stage outside the XC response, which part 10 brought near its FP64 floor.
2. **The J/K derivative matrices** (`_get_jk_ip1`, 11 s): `RYS_build_jk_ip1` has 25 one-thread, 8 multi-thread and
   a generic kernel (part 3). A profile first; the per-tuple sums do not apply directly, since these kernels produce
   matrices, not energies.
3. **Batch `_get_vxc_deriv1`'s per-atom GEMMs over atoms** (part 13): about 4 s.
4. **The rest of the second derivatives,** 2–4 s at most:
   - the four kernels that do not fit (about 1 s), with per-direction factor tables or 4 quartets per block;
   - one-thread generated kernels for the small s and p classes (4.5 s now on GPU4PySCF's);
   - shape tuning.

## GPU4PySCF investigation, part 17 (profile only): the Hessian CPHF's J/K (node1), 2026-11-01
The user approved profiling the CPHF's J/K (part 16's option 1). Nothing in GPU4PySCF changed.
- **Setup:** GTX 1650 Super at 1530 MHz; the caffeine def2-SVP B3LYP Hessian (149 s); `exp/prim-screen` ab75a1b.
- **Data:** `hess_cphf_jk_profile.txt`. The raw Nsight CSVs are in `ncu/ncu_cphfjk*.csv`.

### Where the 35 s go (`cphf_jk_dms.py`, `cphf_jk_parts.py`)
- **Calls:** `_get_jk_mo` runs 7 times, with 72, 69, 69, 59, 18, 1 and 72 densities (35.0 s). The last is
  `solve_withs1`'s final `hs += fvind(mo1)`.
- **What a call does:** one `RYS_build_jk` pass (part 9). Each kernel evaluates a shell quartet's integrals once, then
  loops over the densities (`for (int i_dm ...)`). In each pass of the loop it loads the density elements it needs
  and adds to J and K with one FP64 atomic per element.
- **Cost against the number of densities** (the first N of the first call's densities):

  | densities | 1 | 8 | 24 | 72 |
  |---|---|---|---|---|
  | time | 1.29 s | 1.74 s | 2.98 s | 7.29 s |

  - About 1.2 s does not depend on the densities (integrals, screening); each density adds about 85 ms.
  - At 72 densities, the density loop is 84% of the time.
- **Classes:** none dominates. At 72 densities: (dp|pp) 0.64 s, (dp|ps) 0.61, (ds|ps) 0.54, (ps|ps) 0.52,
  (ps|ss) 0.48. The top 12 classes make 71%.

### What bounds the density loop
- **Source counters, the largest launch of each of the top 12 classes** (1.42 s of the 7.0 s): the density loop has
  84–96% of the warp instructions and 89–96% of the stall samples.
  - **Its stall samples:** waiting for loads (long scoreboard) 42–83% of all samples; a full load/store queue
    (lg throttle) up to 41%.
  - **Its instructions,** as shares of the launch's: FP64 15–45%; integer address arithmetic and other 29–55%;
    density loads 4–12%; atomics 4–13%.
  - **Coalescing:** the loads and atomics are mostly coalesced, at 1.1–2.4× the ideal sector count, since
    consecutive tasks are neighbouring shells.
  - **Representative launches:** the first launch of each class in the loop order is not representative. Those are
    contracted core shells, taking 4.5–55 ms against 53–179 ms for the largest, and task-screening barriers dominate
    them. `ncu_class.py` now picks a launch by its shell groups, e.g. `(ps|ss)@4.2.2.2`.
- **Why the loads wait: the order the compiler keeps** (`ldg_sass_check.py`, compile only).
  - Each output group's density loads come after the previous group's atomics, since the compiler cannot rule out
    that `dm` aliases `vj` or `vk`.
  - A (ps|ss) iteration therefore waits for memory 6 times in sequence: its loads come in 6 groups, 5 of them behind
    atomics. The (dp|pp) kernel (four thread roles) has 35 such points.
  - `__restrict__` on the loop's pointers changes nothing (NVCC ignores it on local pointers).
  - **`__ldg`:** loading the densities through `__ldg` (they are read-only in these kernels) frees the compiler.

    | kernel | loads behind atomics | registers | spills |
    |---|---|---|---|
    | (ps\|ss) | 5 → 0 | 140 → 161 | – |
    | (ps\|ps) | 5 → 0 | 168 → 168 | – |
    | (ds\|ps) | 5 → 0 | 174 → 178 | – |
    | (dp\|ps) | 5 → 4 | 255 → 255 | 32 → 48 bytes |
    | (dp\|pp) | 35 → 21 | 208 → 254 | – |

    The FP64 instruction counts are unchanged in all five kernels.
- **Whole build** (raw metrics, all 277 launches at 72 densities):
  - **FP64:** 199.7 G FP64 thread instructions. That is 3.3 s at the card's FP64 peak (2 per clock per SM), or 3.8 s
    counting 16 clocks per FP64 warp instruction. The FP64 pipe is 57% busy.
  - **Atomics:** 1354 M warp instructions, about 41 G adds. They are `ATOMG` with a null destination, since sm_75
    has no FP64 `RED`.
    - 94% of them hit in L2.
    - The L2 atomic units are 49% busy (40–65% in the largest launches).
  - **Loads:** 55% L2 hits, 64% L1 hits. DRAM is at 20% of peak (259 GB per build).
  - **Stalls:** long scoreboard 60%, lg throttle 20%, barriers 11%.
- **What FP64 atomic adds cost on this card** (`atomic_patterns.py`, G adds per second at 32 warps per SM):

  | array | scattered | one address per warp | coalesced | per-thread runs of 72 |
  |---|---|---|---|---|
  | 512 KB (in L2) | 8.4 | 8.4 | 11.4 | 8.4 |
  | 128 MB (DRAM) | 0.40 | 6.2 | 10.0 | 0.42 (1.13 at 8 warps per SM) |

  - The L2 atomic units take about one FP64 add per clock each, wherever the addresses fall.
  - At those rates, the build's 41 G adds need 3.6–4.9 s. The 49% busy share gives 3.4 s.
- **So:**
  - The loop is latency-bound (7.0 s per 72-density call). Two floors sit at about 3.5–4 s each: FP64, and the L2
    atomics.
  - Hiding the latency can bring the call toward 4–5 s. Going below that needs fewer atomics.
  - Cards with fast FP64 (A100, H100) have a much lower FP64 floor, which leaves latency and atomics as the bound
    there.

### Options (exact; not started)
1. **Density loads through `__ldg`** in the density loops of the J+K and K kernels: `unrolled_rys_jk.cu`,
   `unrolled_rys_k.cu`, the generic kernels' `load_dm` and `gen_rys_k.py`.
   - A mechanical change. The FP64 instructions stay the same, so each thread's sums are bitwise unchanged; only the
     order of the atomics (already free) moves.
   - **Prediction:** 1.2–1.4× on this card's CPHF J/K (35 s → about 25–29 s), less where registers rise. A
     class-by-class check of registers and spills goes with it.
   - It helps wherever several densities share a pass (CPHF, TDDFT, UKS), on every card.
2. **Fewer atomics: accumulate outputs that consecutive tasks share.**
   - **Layout:** lanes run over densities in a density-fastest layout ([nao, nao, n_dm], transposed in the driver).
     The integrals pass between lanes by shuffles or through shared memory.
   - **Accumulation in registers before one atomic:** J_ij for all tasks of a block, and K_ik and K_jk while k
     repeats (tasks sorted by k within a batch).
   - **Effect:** this removes up to 56–79% of the atomics in the top classes (J_ij alone 14–32%), and the remaining
     loads and atomics become coalesced.
   - **Floor:** on this card it becomes FP64 (about 4 s per 72-density call); cards with fast FP64 would gain more.
   - **Cost:** a new generator (extending `gen_rys_k.py`) over several iterations. Shared memory or shuffles limit
     the larger classes.
3. Not exact, so not proposed: reusing the Krylov products in place of the final response (call 7: 7 s of J/K plus
   its XC response) changes results at the CPHF tolerance, since that last `fvind` also refines mo1.
4. Still open from part 16:
   - `_get_jk_ip1` (11 s);
   - `_get_vxc_deriv1`'s per-atom GEMMs (about 4 s);
   - the rest of the second derivatives (2–4 s).

## GPU4PySCF investigation, part 18 (profile only): the SCF and the gradient (node1), 2026-11-01
The user asked for a profile of the SCF iterations and the gradients after part 17. Nothing in GPU4PySCF changed.
- **Setup:** GTX 1650 Super at 1530 MHz; `exp/prim-screen` ab75a1b. The SCF and gradient code have not changed
  since part 8.
- **Cases:** caffeine at def2-SVP (246 AOs) and def2-TZVP (494 AOs), with PBE, B3LYP and r2SCAN; conv_tol 1e−10.
- **Data:** `scf_grad_profile.txt`. The raw Nsight CSVs are in `ncu/ncu_scfgrad_*.csv` (the XC ones gzipped;
  `ncu_summary.py` reads both).
- **New in the tools:**
  - `stage_times.py grad` times the gradient's stages, and `--first-stages` times its first call.
  - `ncu_parts.py --part gradxc` profiles the gradient's XC term. `--part xc` and `gradxc` now pass the density with
    its orbitals attached, as the SCF and the gradient do, so the density pass works from the orbitals there too.
  - `ncu_summary.py --launches` prints one row per integral launch with its grid size; `--gemms` groups the GEMM
    launches by grid size.
  - `grad_xc_first_call.py` (cProfile of the XC term's first two calls, and the grid blocks' AO counts) and
    `gemm_tile_padding.py` (the Vxc GEMM's time against a block's AO count).

### Where an SCF cycle spends its time (`scf_parts.py`; seconds per cycle, the second SCF of each run)

| caffeine | cycles | XC | J | K | rest | cycle |
|---|---|---|---|---|---|---|
| def2-SVP PBE | 18 | 0.628 (47%) | 0.685 (51%) | – | 0.018 | 1.33 |
| def2-SVP B3LYP | 15 | 0.629 (25%) | 0.693 (28%) | 1.132 (46%) | 0.018 | 2.47 |
| def2-SVP r2SCAN | 16 | 1.476 (68%) | 0.690 (32%) | – | 0.018 | 2.18 |
| def2-TZVP PBE | 18 | 1.353 (24%) | 4.213 (75%) | – | 0.052 | 5.62 |
| def2-TZVP B3LYP | 15 | 1.346 (9%) | 4.257 (28%) | 9.521 (63%) | 0.051 | 15.18 |
| def2-TZVP r2SCAN | 16 | 3.666 (46%) | 4.269 (53%) | – | 0.052 | 7.99 |

- **K** is the largest part of a hybrid's cycle. Parts 1–7 worked on its kernels, which are FP64-bound at 74–87%
  busy (part 7), so it was not profiled again.
- **J** is half of a pure functional's cycle at def2-SVP, and 75% of PBE's at def2-TZVP.
- **XC:** GPU4PySCF's sub-timers split it into the density pass, libxc and the Vxc pass: 0.33, 0.006–0.013 and
  0.33 s at def2-SVP (r2SCAN's Vxc pass 1.22 s); 0.58–0.59, 0.006–0.013 and 0.88 s at def2-TZVP (r2SCAN 3.33 s).
  libxc is 1–2% of the XC part.
- **Everything else** (Fock build, DIIS, eigensolver, density) takes 18 ms per cycle at def2-SVP and 52 ms at
  def2-TZVP.

### The XC part (Nsight Compute, one `nr_rks` call at the converged density; def2-SVP)

| kernels | B3LYP | r2SCAN |
|---|---|---|
| all | 0.628 s | 1.467 s |
| density pass GEMMs, C^T φ and C^T ∇_iφ (4 per block, 51 orbital columns) | 0.272 s | 0.272 s |
| Vxc pass GEMMs | 0.271 s (1 per block) | 1.085 s (4 per block, 3 of them for τ: 0.81 s) |
| AO evaluation (in each pass) | 0.056 s | 0.056 s |
| scaling and density contractions | 0.019 s | 0.040 s |
| libxc | 0.006 s | 0.010 s |

- **The grid blocks:** 72 blocks of 4096 points, each with its own subset of 128–222 of the 246 AOs (median 165;
  `grad_xc_first_call.py`). 58 of them have 129–192 AOs.
- **The FP64 pipe is 99% busy in the GEMMs (96–116 GFLOP/s counted, against 122 at peak), but much of it is tile
  padding.**
  - cuBLAS runs the Vxc GEMM (n × 4096)·(4096 × n) in 128 × 64 tiles. The flops Nsight Compute counts per launch are
    exactly 2·256·192·4096 for every block with 129–192 AOs, and 2·256·256·4096 above that (`--gemms`).
  - The time follows the tiles (`gemm_tile_padding.py`): 1.43 ms for 128 AOs, 3.48–3.59 ms for 129–192, 5.06 ms
    for 193–256. A block with 139 AOs does useful work at 45 GFLOP/s.
  - **Useful share** of the FP64 work, from the blocks' AO counts: 55% in the Vxc GEMMs (also r2SCAN's τ GEMMs,
    which have the same shape) and 78% in the density pass (51 orbitals in 64-row tiles).
  - Padding-free GEMMs at the same pipe rate would save about 0.12 s per cycle for a GGA and 0.49 s for r2SCAN at
    def2-SVP.
  - At def2-TZVP the blocks have 210–442 of the 494 AOs (median 319). With the same tiles about a quarter of the
    Vxc GEMMs' work would be padding.
  - Tiles are chosen per architecture and shape, so other cards differ.
- **Fewer flops, separately from the padding:**
  - **GGA density pass:** its 4 GEMMs cost 8 n·nocc·g flops. ψ = C(C^T φ) = Dφ gives ∇ρ = 2 Σ_μ ∇φ_μ ψ_μ with 2 GEMMs.
    Meta-GGAs need C^T ∇_iφ for τ anyway.
  - **Meta-GGA τ term:** three full GEMMs ∇_iφ^T (w_τ ∇_iφ) build a symmetric result. One product with K = 3g, of
    which only the upper triangle is computed, would halve them.
- **Other cards:** on cards with fast FP64 these GEMMs shrink 10–100×. The AO evaluation, which runs in both passes,
  and the elementwise kernels then weigh more.

### The J engine (Nsight Compute, one `get_j` at the converged B3LYP density)
- **Totals:** 0.776 s at def2-SVP and 5.40 s at def2-TZVP. The SCF's per-cycle J is lower (0.69 and 4.2 s), since
  most cycles build from the density change.
- **The unrolled kernels are FP64-bound where they run:** the FP64 pipe is 88% busy (time-weighted) on the SMs they
  occupy.
  - SIMT efficiency is 0.54–0.74; it is 1.00 only for (ss|ss).
  - The likely cause: lanes of a warp take different branches of the Boys function (`gamma_inc.cu`, three branches
    and a loop). No source-level profile has been done.
- **Fixed tiles leave SMs idle at def2-SVP.** Each kernel's block takes a fixed tile of pair batches; `md_j_3_0`, for
  example, takes 48 × 46 batches of 16 pairs.
  - At def2-SVP, 7 of the 21 launches have 7–19 blocks for the 20 SMs, at one block (8 warps) per SM.
  - Counting each launch's time by its blocks per SM (rounded up) against an even spread gives 0.20 s (26%) of idle
    SMs at def2-SVP and 0.40 s (7%) at def2-TZVP.
  - Screening is per 16-pair batch on global batch indices, so smaller tiles compute the same integrals; only the
    order of the atomic adds changes.
  - On cards with 80–132 SMs the same grids would fill only 5–60% of the SMs.
- **At def2-TZVP the generic kernel `md_j_1dm_kernel` runs 32 classes** (those with f functions, and (dp|dp) and
  (dd|·)): 2.05 s, 38% of the build.
  - Its FP64 pipe is 62% busy, and FP64 is only 11–12% of its instructions; barrier stalls are 32–49%.
  - At def2-SVP it runs 5 classes in 34 ms (4%).

### The gradient (`stage_times.py grad`)

| caffeine | gradient | 2e derivatives | XC term | rest | first call |
|---|---|---|---|---|---|
| def2-SVP PBE | 4.26 s | 2.51 (59%) | 1.67 (39%) | 0.08 | 5.84 |
| def2-SVP B3LYP | 4.33 s | 2.59 (60%) | 1.67 (38%) | 0.07 | 5.88 |
| def2-SVP r2SCAN | 6.12 s | 2.50 (41%) | 3.55 (58%) | 0.07 | 6.15 |
| def2-TZVP PBE | 31.52 s | 26.29 (83%) | 5.09 (16%) | 0.14 | 33.21 |
| def2-TZVP B3LYP | 32.99 s | 27.78 (84%) | 5.07 (15%) | 0.14 | 34.70 |
| def2-TZVP r2SCAN | 38.29 s | 26.41 (69%) | 11.74 (31%) | 0.14 | 38.38 |

- **The two-electron part** runs part 8's generated kernels. PBE's (J only) costs as much as B3LYP's (J + 0.2 K): the
  kernels evaluate each integral once for both, and only the density products differ.
- **The XC term** (Nsight Compute, `get_exc` at def2-SVP; the kernels add up to the measured time):

  | kernels | B3LYP | r2SCAN |
  |---|---|---|
  | all | 1.640 s | 3.507 s |
  | GEMMs building AO matrices: `_gga_grad_sum_` (2 per block), `_tau_grad_dot_` (3 more) | 1.203 s (73%) | 3.002 s (86%) |
  | the density pass's orbital GEMMs | 0.272 s | 0.272 s |
  | AO evaluation with second derivatives | 0.099 s | 0.099 s |
  | `dR_dao_w`, scaling, gathers, contractions | 0.047 s | 0.111 s |
  | libxc (per block) | 0.009 s | 0.013 s |

  - **What it does:** each block builds three n × n matrices, ∇_xφ^T aow + (∂_x∇φ·w)^T φ and so on (12 n²g flops;
    meta-GGA τ adds 18 n²g). It then contracts them with D at once (`einsum('nij,ij->ni')`), into one number per AO
    and direction. Their (3n × 4096)·(4096 × n) GEMMs pad less than the SCF's: 73–86% of the work is useful.
  - **Contracting D first gives the same sums:** exc1[μ,x] = Σ_g ∇_xφ_μ (D aow)_μ + Σ_g (∂_x∇φ·w)_μ (Dφ)_μ. That is
    2 GEMMs (4 n²g) plus dot products per AO over the grid. Meta-GGA τ adds Σ_j Σ_g ∂_x∂_jφ_μ w_τ (D∇_jφ)_μ
    (6 n²g).
  - **With the orbitals,** which `get_exc` already takes for the density: D aow = C(C^T aow), Dφ = C(C^T φ) and
    D∇_jφ = C(C^T ∇_jφ), where C^T φ and C^T ∇_jφ are the density pass's own products. That is 6 n·nocc·g (GGA) or
    12 n·nocc·g (meta-GGA) flops, against 12 n²g and 30 n²g now.
- **The first call** (`--first-stages`, `grad_xc_first_call.py`): PBE's XC term takes 3.22 s on its first call and
  1.67 s after; r2SCAN's takes 3.55 s both times.
  - The extra 1.55 s is in `cupyx.cutensor.create_tensor_descriptor`: 1.54 s over its 432 calls in the first call,
    13 ms for all of `_d1_dot_` in the second. Its first call starts up cuTENSOR, once per process.
    `_gga_grad_sum_`'s `_d1_dot_` goes through cuTENSOR (`gpu4pyscf.lib.cutensor.contract`).
  - A GGA SCF uses cuBLAS only (`ao.dot`), so in a GGA job the gradient pays the start-up. A meta-GGA SCF's
    `_tau_dot` already uses cuTENSOR, so there it falls in the SCF's first cycle.
  - For caffeine nothing else in an SCF and gradient uses cuTENSOR (the gradient's other uses are for ECPs).
  - The start-up is the same 1.55 s at def2-TZVP (6.67 → 5.12 s), so it does not depend on the system.

### Options (exact; not started)
Savings are for this card. Every option keeps the same terms and changes only the order of the sums.
1. **The gradient's XC term with the density contracted first** (Python, `perf`).
   - The orbital form makes the GEMMs 6 n·nocc·g (GGA) or 12 n·nocc·g (meta-GGA) flops; the dense form with D makes
     them 4 n²g and 10 n²g.
   - **Prediction:**
     - def2-SVP: XC term 1.67 → 0.7–0.9 s (GGA) and 3.55 → 1.0–1.6 s (r2SCAN). Gradients 4.2–4.3 → 3.3–3.5 s
       (−18 to −23%) and 6.12 → 3.6–4.2 s (−32 to −42%).
     - def2-TZVP, from the blocks' tile counts (they reproduce the measured 5.1 s XC term within 4%): −8 to −11%
       (GGA) and −17 to −24% (r2SCAN).
     - It also removes the 1.55 s cuTENSOR start-up from GGA jobs.
2. **The GGA density pass from ψ = C(C^T φ)** (Python, `perf`): 2 GEMMs instead of 4.
   - **Prediction:** −0.12 s per def2-SVP cycle (PBE −9%, B3LYP −5%) and about −0.2 s per def2-TZVP cycle (PBE −4%).
   - The gradient's density pass gains the same, and option 1's Dφ comes with it.
3. **The XC GEMMs without tile padding** (microbenchmark first).
   - Smaller tiles for the blocks' 128–222 AOs, through cuBLASLt's algorithm selection per shape. cupy does not wrap
     cuBLASLt, but `libcublasLt.so.12` ships with the wheel.
   - **Prediction:** up to −0.12 s per def2-SVP cycle for GGAs and −0.49 s for r2SCAN, if smaller tiles keep the
     pipe as busy.
4. **The meta-GGA τ term as one triangular product** (microbenchmark first; `cublasDsyrkx`, or a batched GEMM over
   the triangle's tiles).
   - **Prediction:** −0.3 to −0.4 s per def2-SVP r2SCAN cycle (−14 to −19%) and about −1.1 s per def2-TZVP cycle
     (−14%). It compounds with option 3.
5. **J-engine launch tiles from the problem size** (CUDA, `exp/prim-screen`).
   - **Prediction:** J −0.15 to −0.2 s per def2-SVP cycle (PBE −12%, B3LYP −7%, r2SCAN −8%) and about −0.3 s at
     def2-TZVP (PBE −5%).
   - It should gain far more on cards with many SMs.
6. **J engine: generated kernels for the f classes** (CUDA).
   - **Prediction:** at the unrolled kernels' 88% FP64 busy, −0.6 s per full def2-TZVP build (−11%); per cycle
     about −0.45 s (PBE −8%).
   - Fewer instructions would count for more on cards with fast FP64.
7. **A J-engine gradient for pure functionals** (large: new derivative kernels).
   - The two-electron part is 69–83% of a def2-TZVP PBE or r2SCAN gradient.
   - **Rough estimate:** the derivative raises the bra's Hermite order by one, so 1.5–2 J builds, 6–9 s instead of
     26 s. Hybrids keep the Rys kernels for K.
8. **A source-level profile of the J engine's divergence** (SIMT 0.54–0.74).
9. **Still open:** part 17's `__ldg` for the CPHF J/K.

## GPU4PySCF investigation, part 19: the gradient's XC term through the orbitals, and the GGA density in two GEMMs (node1), 2026-11-01
The user approved part 18's options 1 and 2 ("Do both").
- **Commits:** gpu4pyscf `perf` a730c90 (the density) and 2258df3 (the gradient), merged into `exp/prim-screen`
  1fc4576. The check is `bench/gpu/xc_grad_check.py` (new).
- **Data:** `p19_xc_grad.txt`.

### Change A: the GGA density from dm·ao (`dft/numint.py`)
- **Before:** `_eval_rho2` (the density from the occupied orbitals, cpos = C sqrt(occ)) computed C^T ao and
  C^T ∇_i ao per grid block. That is four GEMMs of nocc × n_blk × 4096, 8 n·nocc·g flops.
- **Now:** ρ comes from c0 = C^T ao as before, unchanged to the bit. One more GEMM gives dm·ao = C c0, and
  ∇ρ = 2 Σ_μ ∇ao_μ (dm·ao)_μ is libgdft's `GDFTcontract_rho4`, already in the library. That makes 2 GEMMs per block.
  - The meta-GGA branch needs C^T ∇ao for τ anyway. It moves unchanged into `_eval_rho2_mgga`, which can keep all
    three products. Both helpers hand their products to the caller (change B).
  - dm·ao sits after c0 in the caller's buffer, and the SCF's density passes allocate nocc × 4096 more for it.
    Allocated per block instead, it grew CuPy's pool by 7 MB for caffeine/def2-SVP; this way by 3 MB.
- **Reach:** every GGA user of `eval_rho2`/`_eval_rho2`: the SCF (RKS and UKS), gradients, TDDFT, Hessians, NLC and
  shielding.

### Change B: the gradient's XC term without the AO matrices (`grad/rks.py`, `grad/uks.py`)
- **Before:** `_get_exc_task` built per grid block the three n × n matrices of `_gga_grad_sum_` (12 n²g flops;
  `_d1_dot_`, 6 n²g, for LDA; `_tau_grad_dot_`, 18 n²g more, for meta-GGAs), then contracted them with D at once.
- **Now:** with D = C C^T, the same sums are dot products over the grid, per AO, of the AO derivatives with three
  products:
  - D·aow = C (C^T aow);
  - D·ao = C (C^T ao), from change A;
  - D·(w_τ ∇_j ao) = C (w_τ C^T ∇_j ao), also from change A's products.
  - The dot products run in `cupy_helper.dot_rows` (part 12).
- **Flops:** GEMMs per block, the density included, are 8 n·nocc·g (GGA), 20 n·nocc·g (meta-GGA) and
  6 n·nocc·g (LDA).
- **Other effects:**
  - The density matrix now enters through the orbitals, as it already did in the density.
  - Nothing in the path uses cuTENSOR any more, so a GGA job's first gradient no longer pays the 1.55 s start-up.
- **Unchanged:** `_gga_grad_sum_`, `_tau_grad_dot_` and `_d1_dot_` themselves, which the Hessians, PBC, TDDFT and
  the grid-response gradient still use. The NLC gradient (`get_nlc_exc`) also still builds the matrices.

### Exactness (`xc_grad_check.py`; ab75a1b stores, the new tree compares; conv_tol 1e−10)
At fixed orbitals: the gradient's XC term, and the XC matrix of `nr_rks`/`nr_uks`. Then each tree's own SCF and
gradient.

| caffeine | XC term | XC matrix | own SCF: ΔE, cycles | own gradient |
|---|---|---|---|---|
| def2-SVP SVWN | 1.4e−15 | 0 | −2.3e−13 Eh, 18 / 18 | 6.5e−13 |
| def2-SVP PBE | 8.4e−16 | 4.4e−16 | 2.3e−13 Eh, 18 / 18 | 5.6e−13 |
| def2-SVP B3LYP | 1.1e−15 | 4.4e−16 | 4.6e−13 Eh, 15 / 15 | 8.1e−13 |
| def2-SVP r2SCAN | 1.1e−15 | 0 | 2.3e−13 Eh, 16 / 16 | 4.6e−13 |
| cation, UKS SVWN | 1.6e−15 | 0 | −9.4e−11 Eh, 50 / 50 | 2.5e−10 |
| cation, UKS B3LYP | 7.0e−16 | 4.4e−16 | −2.3e−13 Eh, 26 / 26 | 6.5e−13 |
| cation, UKS r2SCAN | 1.8e−15 | 0 | −1.8e−12 Eh, 27 / 27 | 2.2e−12 |
| def2-TZVP PBE | 9.8e−16 | 8.9e−16 | 0, 18 / 18 | 1.5e−12 |
| def2-TZVP r2SCAN | 9.7e−16 | 0 | −6.8e−13 Eh, 16 / 16 | 1.5e−12 |

- **Scales:** the XC term is up to 0.32, the XC matrix up to 4.4, the gradients up to 0.15 Eh/bohr.
- **To the bit:** the XC energy and the electron count in every case. The XC matrix too, wherever the density
  code did not change (LDA, meta-GGA).
- **UKS SVWN** stops at max_cycle (50) in both trees.
- **The final `numint.py`** (dm·ao in the SCF's buffer) gives the same fixed-orbital numbers for PBE, UKS B3LYP and
  def2-TZVP PBE.
- **Tolerance:** the first run used conv_tol 1e−12 and conv_tol_grad 1e−8. There two runs of the unmodified tree
  took 43 and 50 (max_cycle) cycles and ended 9e−10 Eh apart. That is below the GPU J/K's rounding noise, so the
  checks use conv_tol 1e−10 as the earlier parts' did.

### Timings (GTX 1650 Super at 1530 MHz; ab75a1b, then the new tree, per case)
- **Gradients** (`stage_times.py grad`):

  | caffeine | gradient | first call | XC term |
  |---|---|---|---|
  | def2-SVP PBE | 4.24 → 3.05 s (−28%) | 5.82 → 3.08 s (−47%) | 1.67 → 0.47 s |
  | def2-SVP B3LYP | 4.34 → 3.13 s (−28%) | 5.92 → 3.17 s (−46%) | 1.67 → 0.46 s |
  | def2-SVP r2SCAN | 6.11 → 3.47 s (−43%) | 6.14 → 3.50 s (−43%) | 3.54 → 0.90 s |
  | def2-TZVP PBE | 31.50 → 27.22 s (−14%) | 33.27 → 27.41 s (−18%) | 5.08 → 0.79 s |
  | def2-TZVP r2SCAN | 38.06 → 27.90 s (−27%) | 38.27 → 28.01 s (−27%) | 11.66 → 1.51 s |

  - Part 18 predicted −18 to −23% (GGA) and −32 to −42% (r2SCAN) at def2-SVP. The gradient's own density pass also
    gains from change A.
  - The two-electron part is now 82–97% of the gradient.
- **SCF** (`scf_parts.py`; per cycle):

  | caffeine | XC | of it the density pass | SCF |
  |---|---|---|---|
  | def2-SVP PBE | 0.624 → 0.494 s | 0.330 → 0.191 s | 25.05 → 22.66 s (−9.5%) |
  | def2-SVP B3LYP | 0.625 → 0.497 s | 0.335 → 0.195 s | 38.70 → 36.72 s (−5.1%) |
  | def2-SVP r2SCAN | 1.464 → 1.474 s | 0.339 → 0.342 s | 36.75 → 37.02 s (code unchanged) |
  | def2-TZVP PBE | 1.352 → 1.136 s | 0.579 → 0.332 s | 104.53 → 100.10 s (−4.2%) |

  - Same energies to the printed digits and the same cycle counts.
  - Part 18 predicted −0.12 s per def2-SVP cycle and −0.2 s per def2-TZVP cycle; measured −0.13 and −0.22 s.
- **An SCF plus its first gradient:**
  - def2-SVP: PBE 30.9 → 25.7 s (−17%), B3LYP 44.6 → 39.9 s (−11%), r2SCAN 42.9 → 40.5 s (−6%).
  - def2-TZVP: PBE 137.8 → 127.5 s (−7%), r2SCAN 171.7 → 161.4 s (−6%; part 18's SCF time, whose code is
    unchanged).

### Unit tests (the new tree)
- All pass. The gradient files ran again after the HF fix below, the others before it (the fix changes only
  `get_exc`'s HF path):
  - gradients: `test_rks_grad.py` (15), `test_uks_grad.py` (12), `test_tdrks_grad.py` (7);
  - DFT: `test_numint.py` (24), `test_rks.py` (17, 1 skipped), `test_uks.py` (10);
  - Hessians: `test_rks_hessian.py` (12), `test_uks_hessian.py` (12);
  - TDDFT: `test_tdrks.py` (28), `test_tduks.py` (20).
- **A bug the first run found:** with `xc='hf'` (xctype HF) the new task looked up a buffer size for HF and failed.
  The old code fell through its branches and returned zeros, and the new one now does the same.
  `test_grad_hf` failed in both gradient files before the fix and passes after it.

### Next
- Still open from part 18:
  - (3) the XC GEMMs without tile padding;
  - (4) the meta-GGA τ term as a triangle;
  - (5) J-engine launch tiles;
  - (6) generated J kernels for the f classes;
  - (7) a J-engine gradient;
  - (8) the J engine's divergence;
  - (9) part 17's `__ldg`.
- **Weighs more now:** (7). The two-electron part is 82–97% of every gradient above, and for pure functionals it is
  J only.
- **The same change elsewhere:** the NLC gradient (`get_nlc_exc`) and the grid-response gradient
  (`get_exc_full_response`) still build the AO matrices; the TDDFT gradients' XC terms too.

## GPU4PySCF investigation, part 20: the gradient's grid-response and NLC terms without the AO matrices (node1), 2026-11-01
The user approved all of part 19's next steps ("All of those sound good"). They run one at a time, and this is the
first: part 19's change in the gradient's remaining XC terms.
- **Commits:** gpu4pyscf `perf` ab8e66a, merged into `exp/prim-screen` 0f32fb7 (node1's tree updated).
- **Tools:** `stage_times.py grad --grid-response`; `xc_grad_check.py` compares the grid-response and NLC terms too;
  `nlc_grad_time.py`; `nlc_profile.py`.
- **Data:** `p20_grad_response_nlc.txt`.

### Change (`grad/rks.py`, `grad/uks.py`)
- **What changes:** `get_exc_full_response` (RKS, UKS), `get_nlc_exc` and `get_nlc_exc_full_response` now form
  part 19's per-AO sums: `dot_rows` of the AO derivatives with dm·aow, dm·ao and dm·(w_τ ∇_j ao).
- **What that replaces:**
  - the accumulated (3, nao, nao) matrix (UKS: (2, 3, nao, nao)) and its `add_sparse` or fancy-index scatter;
  - the per-block einsum for the grid's own atom;
  - `contract_h1e_dm`.
  The per-AO sums give both the atoms' terms and the grid atom's trace (2 Σ_μ).
- **Two paths:**
  - **With orbitals:** when the density matrix carries `mo_coeff` and `mo_occ`, as the gradients pass it, the
    products go through the orbitals (`_exc1_rho`, `_exc1_ao_dots`), and so does the density on the grid.
  - **Without them:** `cuest_wrapper` may pass a bare density matrix. Then the products are GEMMs with the block's
    density matrix (`_exc1_ao_dots_dm`), still 3× fewer flops than before.
  - For UKS NLC the total density matrix is the two spins' occupied orbitals side by side.

### Exactness (`xc_grad_check.py`; ab75a1b stores, the new tree compares; conv_tol 1e−10)
- **Grid response, caffeine/def2-SVP:**

  | case | orbital path | density-matrix path | own SCF and gradient: ΔE, cycles, max \|Δg\| |
  |---|---|---|---|
  | RKS SVWN | 5.0e−16 | 5.3e−16 | −2.3e−13 Eh, 18 / 18, 3.8e−13 |
  | RKS B3LYP | 6.7e−16 | 5.6e−16 | −1.1e−12 Eh, 15 / 15, 3.5e−13 |
  | RKS r2SCAN | 1.1e−15 | 7.9e−16 | 9.1e−13 Eh, 16 / 16, 4.9e−13 |
  | cation, UKS B3LYP | 4.7e−16 | 3.6e−16 | −2.3e−13 Eh, 26 / 26, 3.5e−13 |
  | cation, UKS r2SCAN | 5.6e−16 | 5.6e−16 | 6.8e−13 Eh, 27 / 27, 2.5e−12 |

  The term reaches 0.65.
- **NLC, benzene/def2-SVP B97M-V** (VV10 on caffeine would mean 512 s per SCF):
  - At fixed orbitals: `get_nlc_exc` agrees within 1.2e−18 (RKS) and 6.7e−19 (UKS), on values up to 7e−4. The
    grid-response forms agree within 1.7e−18 on both paths, on values up to 1.5e−3.
  - Each tree's own SCF: the same cycle counts (9 RKS, 11 UKS) and ΔE ≤ 2.3e−13 Eh. Gradients agree within 2.6e−13
    (RKS) and 1.1e−9 Eh/bohr (UKS). The UKS difference is SCF convergence noise: at fixed orbitals UKS agrees to
    1e−18.

### Timings (GTX 1650 Super; ab75a1b, then the new tree)

| caffeine, gradient with the grids' response | gradient | first call | XC term |
|---|---|---|---|
| def2-SVP B3LYP | 5.32 → 4.12 s (−23%) | 6.89 → 4.16 s | 2.63 → 1.46 s |
| def2-SVP r2SCAN | 8.17 → 4.63 s (−43%) | 8.21 → 4.68 s | 5.61 → 2.07 s |
| def2-TZVP PBE | 34.03 → 28.77 s (−15%) | 35.77 → 28.91 s | 7.60 → 2.31 s |

- **What is left in the XC term:** the grid-weight derivatives (0.32 s) and the extra density pass on the
  atom-sorted grid.
- **NLC terms** (caffeine B97M-V, at PBE orbitals): `get_nlc_exc` 30.8 → 29.8 s, `get_nlc_exc_full_response`
  64.9 → 63.8 s. The VV10 pair kernels take the rest (next section).

### Where VV10 spends its time (`nlc_profile.py`, caffeine/def2-SVP B97M-V)
- **SCF:** 512 s for 15 cycles. `nr_nlc_vxc` runs 16 times for 475 s, 29.7 s each, which is 93% of the SCF. The
  NLC grid has as many points as the main grid (294,912).
- **Gradient:** `get_nlc_exc` takes 30.7 s, of which the AO part is about 1 s and VV10's pair kernel the rest.
- **The kernel** (`vv10.cu`, `vv10_fock_eval_UWE_kernel`) loops over all pairs of grid points, both orders.
  - Per pair: r_ij², g_ij = ω_i r² + κ_i, g_ji, one FP64 reciprocal of g_ij g_ji (g_ij + g_ji), then E, U and W.
  - On this card that is FP64-bound.
  - The pair kernel Φ_ij is symmetric in i and j, and so are g_ij g_ji (g_ij + g_ji) and its reciprocal, so each
    unordered pair is evaluated twice.
- **Option (exact; not started, not yet approved):** evaluate each pair once and add its terms to both points.
  - Tiles of pairs with J ≥ I; the j-sums by warp shuffles.
  - About 1.6× fewer FP64 instructions, so a caffeine/def2-SVP B97M-V SCF of 512 s would take about 330 s, and its
    gradient about 22 s instead of 34 s. Estimates.

### Unit tests (the final tree)
- `test_rks_grad.py` (15), `test_uks_grad.py` (12), `test_grid_response.py` (7) and `test_vv10_grid.py` (20) pass.

### Next
Part 21: the XC GEMMs, padding-free tiles and the meta-GGA τ triangle, starting with a benchmark
(`xc_gemm_bench.py`).

## GPU4PySCF investigation, part 21: the XC part's per-block A·Bᵀ products through cuBLAS's algorithm 1 (node1), 2026-11-01
Second of the approved items: the XC GEMMs without tile padding, and the meta-GGA τ term as a triangle, starting
with a benchmark.
- **Commits:** gpu4pyscf `perf` 79b2c43, merged into `exp/prim-screen` c6c7ce0.
- **Tools:** `xc_gemm_bench.py` (cuBLAS algorithms for the Vxc product; the τ term three ways) and
  `xc_gemm_algos.py` (every per-block GEMM shape of the XC code against cuBLAS's algorithms).
- **Data:** `p21_xc_gemm.txt`.

### What the benchmark showed (GTX 1650 Super; K = 4096 grid points, n a block's AO count)
- **The Vxc product ao·aowᵀ** (n × K)·(K × n): `cublasGemmEx` with `CUBLAS_GEMM_ALGO1` against cuBLAS's default
  heuristic, which is what `cupy.dot` and cuTENSOR use.

  | n | 48 | 96 | 128 | 139 | 165–192 | 222 | 260 | 319 | 400 | 442 | 600 | 800 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | default (ms) | 0.225 | 1.389 | 1.404 | 3.479 | 3.58 | 5.054 | 7.951 | 7.958 | 18.27 | 18.27 | 32.87 | 54.78 |
  | algorithm 1 (ms) | 0.196 | 0.730 | 1.401 | 1.921 | 2.62 | 3.486 | 5.737 | 6.95 | 11.81 | 13.72 | 25.18 | 43.44 |

  - The default's times step at multiples of 128 (part 18's padding); algorithm 1's grow with n.
  - Algorithm 1 is the fastest, or within 3% of it, at every n from 40 to 800. Below that it is at most 5 µs
    slower (0.112 against 0.107 ms at n = 24 and 32).
- **The thin products** (C^T ao, nocc × K; C c0, n × K) are already fastest with the default, within 1–7%.
- **The τ term** (three products with the derivatives, symmetric in sum):
  - `cublasDsyrkx` on the stacked derivatives (K = 3·4096, lower triangle only) is slower than three GEMMs at
    these sizes: 16.5 against 10.8 ms at n = 165. No triangle, then.
  - One stacked GEMM beats the three default ones (7.9 ms), but with algorithm 1 the three take 7.85 ms against the
    stack's 7.63 ms. That is not worth the stacking copy.

### Change (`lib/cupy_helper.py`, `dft/numint.py`)
- **`cupy_helper.dot_abt(a, b, out, beta)`:** out = a·bᵀ + β·out for row-major a and b, through `cublasGemmEx`.
  - On compute capability 7.5 it uses algorithm 1, elsewhere the default; newer devices ignore the algorithm
    numbers anyway.
  - Where algorithm 1 returns `CUBLAS_STATUS_NOT_SUPPORTED` (odd block lengths, for one), it runs the default.
- **Callers:** the Vxc passes of `nr_rks`, `nr_uks` and `nr_nlc_vxc`; the response in `nr_rks_fxc` and
  `nr_uks_fxc`; and `_tau_dot`. That is every per-block ao·aowᵀ of the XC code.
- **cuTENSOR:** the meta-GGA paths no longer go through it.

### Exactness (`xc_grad_check.py` against part 19's stored unmodified-code results; conv_tol 1e−10)

| caffeine | XC matrix at fixed orbitals | own SCF: ΔE, cycles | own gradient |
|---|---|---|---|
| def2-SVP SVWN | 3.6e−15 | −2.3e−13 Eh, 18 / 18 | 4.1e−13 |
| def2-SVP PBE | 2.2e−15 | −2.3e−13 Eh, 18 / 18 | 5.4e−13 |
| def2-SVP B3LYP | 1.8e−15 | 4.6e−13 Eh, 15 / 15 | 9.4e−13 |
| def2-SVP r2SCAN | 2.7e−15 | 0, 16 / 16 | 3.4e−13 |
| cation, UKS SVWN | 3.6e−15 | −2.6e−11 Eh, 50 / 50 (max_cycle) | 6.7e−11 |
| cation, UKS B3LYP | 1.8e−15 | 2.3e−13 Eh, 26 / 26 | 4.2e−13 |
| cation, UKS r2SCAN | 4.0e−15 | −1.4e−12 Eh, 27 / 27 | 2.0e−12 |
| def2-TZVP PBE | 7.6e−15 | 4.6e−13 Eh, 18 / 18 | 9.1e−13 |
| def2-TZVP r2SCAN | 9.8e−15 | −4.6e−13 Eh, 16 / 16 | 1.7e−12 |

The XC matrix reaches 4.4. The XC energy and the electron count stay the same to the bit, since the density code
did not change.

### Timings (GTX 1650 Super; part 20, then part 21)

| caffeine | XC per cycle | of it the Vxc pass | SCF |
|---|---|---|---|
| def2-SVP PBE | 0.494 → 0.396 s | 0.332 → 0.229 s | 22.6 → 20.8 s (−8%) |
| def2-SVP B3LYP | 0.497 → 0.400 s | 0.336 → 0.231 s | 36.7 → 35.2 s (−4%) |
| def2-SVP r2SCAN | 1.474 → 1.081 s | 1.222 → 0.802 s | 37.0 → 30.3 s (−18%) |
| def2-TZVP PBE | 1.136 → 0.975 s | 0.879 → 0.712 s | 100.1 → 97.0 s (−3%) |
| def2-TZVP r2SCAN | 3.665 → 3.024 s | 3.328 → 2.648 s | 133.4 → 122.5 s (−8%) |

- Same energies to the printed digits, and the same cycle counts.
- **TDA gradient** (def2-SVP B3LYP, state 1): 57.4 → 54.7 s; its XC response (`nr_rks_fxc`, 27 calls)
  13.9 → 11.3 s.

### Unit tests
- **All pass on the final tree:**
  - DFT: `test_numint.py` (24), `test_rks.py` (17, 1 skipped), `test_uks.py` (10);
  - TDDFT: `test_tdrks.py` (28), `test_tduks.py` (20);
  - Hessians: `test_rks_hessian.py` (12), `test_uks_hessian.py` (12);
  - gradients: `test_rks_grad.py` (15), `test_uks_grad.py` (12), `test_grid_response.py` (7),
    `test_tdrks_grad.py` (7), `test_vv10_grid.py` (20).
- **What the first run caught:** UKS, TDDFT, the UKS Hessian, grid response and VV10 failed with
  `CUBLAS_STATUS_NOT_SUPPORTED`. Their small molecules and the grid response's masked grids give blocks of
  arbitrary length, and for some of them algorithm 1 is unsupported. Caffeine's blocks are all 4096 points, so the
  checks and timings above never met such a block.
- **The fix:** `dot_abt` now runs the default there. Nothing has been computed when the call fails, so the
  fallback is exact. The timings and checks are unaffected.

### Other cards
- **Ampere and newer:** cuBLAS ignores the algorithm numbers there, so those cards keep their default. Their padding
  costs would have to be measured on the A100/H100 suite; cuBLASLt's tile selection would be the tool there.
- **Volta (sm_70):** it honours the numbers but was not measured, so it keeps the default too.

## GPU4PySCF investigation, part 22: the J engine's launch tiles sized to fill the SMs (node1), 2026-11-02
Third of the approved items (part 18's option 5).
- **Commits:** gpu4pyscf `exp/prim-screen` ef2327e (the unrolled kernels' run-time bra tiles) and fb338f2 (no
  empty bra tiles), both generated by `exp_prim_screen.py --j-tiles auto`; `perf` 452c283 (the generic kernel's
  tile cut), merged into `exp/prim-screen` e417101.
- **Tools:**
  - `exp_prim_screen.py --j-tiles auto|off --j-blocks-per-sm N`;
  - `jk_at_dm.py --ndm N` (several densities at once, which run the multi-density kernels);
  - `j_memcheck.py` (the J build under compute-sanitizer, with CuPy's memory pool off).
- **Data:** `p22_j_tiles.txt`.

### Change
- **The unrolled kernels** (`unrolled_md_j.cu` and `unrolled_md_j_4dm.cu`, 12 + 11 kernels):
  - Each block took a fixed tile of TILEX × TILEY batches of 16 bra and ket pairs. Part 18 counted 7–24 blocks in
    some def2-SVP launches, for 20 SMs.
  - The bra tile only sets three things: the batch loop, the batch index of the batch-level screening, and the
    block's triangle skip. `qd_ij_max` holds one maximum per 16 pairs, whatever the tile.
  - The ket tile also lays out shared memory, so it stays.
  - Each kernel now derives its bra tile from `gridDim.x`. The launcher's `md_j_grid_x` picks `gridDim.x` so that a
    launch has at least `MD_J_BLOCKS_PER_SM` = 64 blocks per SM, with the compiled TILEX as the largest tile.
    Diagonal classes get twice that, since their blocks above the diagonal return at once.
- **The generic kernel** (`j_engine.py`): `_fill_sms_tilex` cuts the scheme's bra tile per launch in the same way.
  `MD_build_j` derives its grid from the tile.
- **Exactness:** the integrals and the screening are the same; only the order of the atomic adds changes.

### Blocks per SM (J at fixed densities, unrolled kernels only)

| blocks per SM | part 21 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|
| def2-SVP (s) | 0.778 | 0.632 | 0.610 | 0.563 | 0.537 | 0.516 | 0.508 | 0.503 | 0.503 | 0.503 |
| def2-TZVP (s) | 5.396 | 5.382 | 5.352 | 5.256 | 5.140 | 5.052 | 4.985 | 4.964 | 4.957 | 4.959 |

- One block per SM already takes def2-SVP from 0.778 to 0.632 s. That is close to part 18's estimate of 0.20 s of
  idle SMs.
- The rest comes from finer tiles:
  - Pairs are sorted by their bound, so the first tiles hold most of the work. Many small tiles are handed out to
    the SMs as they finish, which spreads that work evenly.
  - The small kernels also fit two blocks on an SM (49% occupancy in part 18's profile); the larger ones fit one.
- The time is flat from 64 to 256 blocks per SM, so 64 it is.

### Timings (GTX 1650 Super; part 21, then part 22)

| caffeine, J at a fixed B3LYP density | part 21 | part 22 |
|---|---|---|
| def2-SVP | 0.778 s | 0.500 s (−36%) |
| def2-SVP, 2 densities | 0.746 s | 0.657 s |
| def2-SVP, 4 densities | 1.071 s | 0.950 s |
| def2-TZVP | 5.396 s | 4.868 s (−10%) |
| def2-TZVP, 2 densities | 7.513 s | 7.294 s |

| caffeine SCF | J per cycle | SCF |
|---|---|---|
| def2-SVP PBE | 0.685 → 0.378 s | 20.81 → 15.13 s (−27%) |
| def2-SVP B3LYP | 0.693 → 0.388 s | 35.21 → 30.51 s (−13%) |
| def2-TZVP PBE | 4.215 → 3.667 s | 97.00 → 86.75 s (−11%) |

- **Which trees:** the def2-SVP SCFs ran with the unrolled kernels' tiles alone. The generic kernel's cut gains
  another 4 ms per def2-SVP build, and 0.09 s per def2-TZVP build.
- **Exactness:** same energies to the printed digits, and the same cycle counts. J is within 9.9e−13 of part 21,
  for values up to 35.
- **Against the prediction:** part 18 predicted −0.15 to −0.2 s per def2-SVP cycle and −0.3 s at def2-TZVP. The
  measured gains are −0.31 s and −0.55 s. Counting idle SMs missed the imbalance within launches.

### An out-of-bounds read, found and fixed before the write-up
- **The bug:** after a cut, `md_j_grid_x` chose `gridDim.x` = min(nbatch, ⌈want/grid_y⌉). The kernels then take
  tiles of ⌈nbatch/gridDim.x⌉ batches, which can leave blocks past the last tile.
  - Example: 100 batches in 43 blocks gives tiles of 3, so blocks 34–42 get none.
  - Such a block read `q_cond_ij[task_ij0]` past the array's end before returning. If the value it read did not
    screen it out, it added zeros to J.
  - The results were right, but the read is undefined.
- **How it was found:** reading the generated launcher for this write-up.
  - `compute-sanitizer --tool memcheck` then caught it in the first launch, (ps|ss): a read 93 bytes past the (ps)
    pairs' 19,620-byte `q_cond`.
  - CuPy's memory pool had to be off, so that every array is its own allocation.
- **The fix:** `md_j_grid_x` launches ⌈nbatch/t⌉ blocks, with t = ⌈nbatch/g⌉.
  - The kernels compute the same t, and every block has a tile: ⌈n/⌈n/t⌉⌉ = t when t = ⌈n/g⌉.
  - memcheck now reports 0 errors at def2-SVP and def2-TZVP, for 1 and 2 densities.
  - The J times and values are as before.
- **The generic kernel was not affected:** `MD_build_j` derives its grid from the tile.

### Unit tests
- **All pass on each tree** (the unrolled tiles; with the generic cut; and the fixed final tree):
  - `test_scf_j_engine.py` (8) and `test_scf_jk.py` (11);
  - `test_rks.py` (17, 1 skipped) and `test_uks.py` (10);
  - `test_tdrks.py` (28).

### Other cards
- `md_j_grid_x` and `_fill_sms_tilex` read the SM count from the device.
- With fixed tiles, cards with 40–132 SMs would have had most of their SMs idle in a caffeine-sized J build: part 18
  found 5–60% filled. They should gain more than this card. The A100/H100 suite would measure it.

## GPU4PySCF investigation, part 23: generated J kernels for the f-function classes (node1), 2026-11-02
Fourth of the approved items (part 18's option 6).
- **Commit:** gpu4pyscf `exp/prim-screen` 4d3d6fe (`generated_md_j.cu`, from `exp_prim_screen.py --gen-j`).
- **Tools:**
  - `gen_md_j.py`, the generator;
  - `ncu_j_class.py`, which puts the J launches of chosen classes in the profiler range;
  - `ncu_source_regions.py --j`, which splits J kernels into the Boys function, the R tensor, the contraction and the
    rest;
  - `j_exact_check.py`, which compares J matrices with PySCF's unscreened J on the CPU.
- **Data:** `p23_j_generated.txt`.

### Where the J kernels spend their instructions (source-level profile, def2-TZVP, part 22)
- **The generic kernel** (`md_j_1dm_kernel`, here (fs|dp), (fp|dp), (fd|dp)):
  - FP64 is 10% of its instructions; its FP64 pipe is 61–66% busy.
  - The R tensor takes 32–37% of its instructions and the contraction 51–54%, most of it integer address arithmetic
    from index tables (IMAD, ISETP, MOV, BRA).
  - The R tensor also has half of its stall samples, at the barriers between recursion levels.
- **The Boys function** (`gamma_inc.cu`), in the unrolled kernels:

  | class | (ss\|ss) | (ps\|ss) | (ps\|ps) | (ds\|ps) | (dp\|ps) | (dp\|ds) |
  |---|---|---|---|---|---|---|
  | Boys share of FP64 instructions | 60% | 81% | 72% | 61% | 51% | 36% |
  | active lanes in it | 1.00 | 0.40 | 0.36 | 0.33 | 0.32 | 0.30 |

  - The series branch (t < m/2 + 1/2) is taken by 6–12% of lanes. Its loop divides once per pass (`x *= t / bi`,
    about 20 passes), while the rest of the warp waits. That one line is 18–30% of all instructions in md_j_1_0 to
    md_j_3_2.
  - The erf branch (`SQRTPIE4 / tt * erf(tt)`) is 49% of the FP64 work in (ss|ss) and 19% in (ps|ss).
  - Not changed here: the low-order classes are not on the approved list, and the change would alter rounding. It is
    proposed below.

### The generator (`gen_md_j.py`)
- **Frame:** that of the unrolled kernels.
  - 16 × 16 threads, one (bra pair, ket pair) quartet per thread.
  - Run-time bra tiles (part 22) and TILEY batches of 16 ket pairs.
  - The same screening, reductions and atomics.
- **Boys function:** into a local array, which stays in registers; the unrolled kernels use shared memory.
- **R tensor:** the McMurchie–Davidson recursions, z first, then y, then x, column (u, v) by column.
  - Each R⁽⁰⁾_tuv is contracted as soon as it exists, into both sides' accumulators:
    - J_kl[k] += R[i+k]·D_ij[i], with the bra density from shared memory, reduced over the 16 bra lanes per quartet;
    - J_ij[i] += R[i+k]·D_kl[k], with the ket density in registers and the phase (−1)^|k| folded in.
  - The recursion path is the same as the generic and unrolled kernels': x if t > 0, else y, else z.
  - Only 15 (order 6) to 32 (order 9) intermediates are live at once. The unrolled kernels instead keep the whole
    R tensor live until the end of the quartet: md_j_3_2 holds 56 R values and needs 216 registers.
- **Registers:** the accumulators dominate, nf3(lij) + 2·nf3(lkl) doubles. In ptxas -v at sm_75:
  - (3,3) and (4,2) use 254 registers without spills.
  - (5,1) and (3,4) spill 80 B, (4,3) 172 B, (5,2) 440 B and (4,4) 1.1 KB.
  - (5,3) and (6,0)–(6,2) spill 0.9–2.4 KB, (5,4) and (6,3) 4–7 KB, (5,5) and (6,4) 13 KB.
  - (6,5) spills 38 KB and (6,6) 104 KB.
  - Reading the ket density from memory at each use instead of registers changed nothing: ptxas already reloads it.
- **Dispatch:** `md_j_generated()` is tried by `MD_build_j` after `md_j_unrolled()`, for one density.

### Per class (def2-TZVP, ncu GPU time; generic → generated)

| class (lij, lkl) | time | FP64 pipe | warp instructions |
|---|---|---|---|
| (fs\|dp) (3,3) | 166 → 78 ms | 61 → 91% | 1.71 → 0.21 G |
| (fp\|ds) (4,2) | 161 → 71 ms | 57 → 91% | 1.52 → 0.19 G |
| (fd\|ps) (5,1) | 140 → 68 ms | 56 → 90% | 1.22 → 0.18 G |
| (fp\|dp) (4,3) | 129 → 69 ms | 66 → 92% | 1.44 → 0.17 G |
| (fd\|dp) (5,3), 1.3 KB spills | 88 → 38 ms | | 0.88 → 0.09 G |
| (fp\|fp) (4,4), 1.1 KB spills | 59 → 26 ms | | 0.66 → 0.06 G |
| (ff\|fs) (6,3), 7 KB spills | 22 → 13 ms | | |
| (ff\|fd) (6,5), 38 KB spills | 11 → 15 ms | | |
| (ff\|ff) (6,6), 104 KB spills | 2 → 7 ms | | |

- **All 32 classes of order ≥ 6:** 1.96 s → 0.99 s. The classes of order 6–8 run 1.6–2.4× faster, and those of
  order 9–10 1.1–2.0×.
- **The two exceptions:** (6,5) and (6,6) ran slower than the generic kernel, so they stay generic (13 ms).
- **FP64 work is 12–17% lower:** there are no int8 phases or recursion factors to convert and multiply.

### Exactness
- **Against part 21,** J moves by up to 7.3e−12 (values up to 35.5), where part 22's reordering moved it by 9.6e−14.
  The arithmetic of each quartet now differs (the order of its sums and the pairing of FMAs), not only the order of
  the atomic adds.
- **Against the exact J,** the generated kernels are as close or closer:

  | caffeine | max error | RMS error | E_J error | elements with error > 1e−12 |
  |---|---|---|---|---|
  | def2-TZVP, part 21 | 1.55e−11 | 7.65e−13 | −9.55e−12 Eh | 10,518 |
  | def2-TZVP, part 23 | 1.55e−11 | 7.30e−13 | −9.35e−12 Eh | 9,524 |
  | def2-SVP, part 21 | 6.16e−12 | 4.17e−13 | −5.85e−12 Eh | 1,678 |
  | def2-SVP, part 23 | 6.16e−12 | 4.20e−13 | −5.73e−12 Eh | 1,682 |

  - The exact J is PySCF's on the CPU without screening, which takes 11 s at def2-TZVP. The largest error, common
    to both, is screening.
  - The elements that moved most are those where the generic kernel was furthest off. For example, a d–d element of
    atoms 11 and 13, 11.4 bohr apart: exact 1.3050e−10, generic −7.9e−12, generated −1.2e−12. The likely cause is
    the generic kernel's screening by batches of its own size, but that was not checked.
- **SCFs:** same energies to the printed digits and the same cycle counts.
- **compute-sanitizer memcheck:** 0 errors at def2-TZVP.

### Timings (GTX 1650 Super; part 22, then part 23)

| caffeine | J at a fixed density | J per SCF cycle | SCF |
|---|---|---|---|
| def2-TZVP PBE | 4.870 → 3.892 s (−20%) | 3.665 → 2.945 s | 86.71 → 73.86 s (−15%) |
| def2-SVP PBE | 0.500 → 0.490 s | 0.376 → 0.367 s | 15.10 → 14.90 s |
| def2-SVP B3LYP | | 0.384 → 0.376 s | 30.40 → 30.27 s |

- **def2-SVP gains little:** its only classes of order ≥ 6 are (dp|dp) and (dd|·), about 30 ms per build.
- **Several densities are unchanged:** UKS and response J (def2-TZVP with 2 densities: 7.295 s, as before) keep
  `md_j_4dm_kernel` for these classes. A multi-density variant would multiply the accumulators, which already set
  the register limit.

### Unit tests
- **17-class build:** `test_scf_j_engine.py` (8), `test_scf_jk.py` (11), `test_rks.py` (17, 1 skipped),
  `test_uks.py` (10) and `test_tdrks.py` (28) pass.
- **Final 15-class build:** the J-engine tests pass. It differs only in sending (6,5) and (6,6) back to the generic
  kernel.

### Other cards
- **Fast FP64:** the generic kernel executed 8× the instructions for the same FP64 work. Where FP64 is fast (A100,
  H100), those instructions bound the time, so the generated kernels should gain more there than this card's 2×.
- **Spills:** ptxas allocates registers per architecture, so the spills, and the choice to keep (6,5) and (6,6)
  generic, would need checking on sm_80/sm_90.

### Proposed next (not approved): the Boys function
- **Division-free series:** the series loop and the downward recursion can multiply by tabulated reciprocals
  1/(m + k + 1/2) instead of dividing. That is the same arithmetic up to rounding, about 1e−16 relative in F_m.
- **erf skip:** erf(√t) is 1.0 in double precision once t is larger than about 36–49. Skipping it there is
  bit-exact if CUDA's erf returns exactly 1.0 there, which needs checking.
- **What it would gain:** the series line alone is 18–30% of the instructions of the low-order unrolled kernels. Those
  kernels make up about 80% of a def2-TZVP J build and nearly all of a def2-SVP one.
- **How to check it:** `j_exact_check.py` measures exactly this kind of rounding-level change.

## GPU4PySCF investigation, part 24: the J engine's nuclear gradient for pure functionals (node1), 2026-11-02
Fifth and last of the approved items (part 18's option 7).
- **Commits:** gpu4pyscf `exp/prim-screen` 2536863 (`MD_build_j_ip1`, `md_j_grad.cu` and `generated_md_j_ip1.cu`,
  from `exp_prim_screen.py --j-grad on --gen-j-ip1`); `perf` 6011265 (`j_engine._VHFOpt.get_ej_ip1` and the
  dispatch in `grad.rhf.Gradients.jk_energy_per_atom`), merged into `exp/prim-screen` 0f3c06d.
- **Tools:**
  - `j_grad_proto.py`, the prototype, which compares with the Rys gradient at a fixed density;
  - `grad_jengine_check.py`, which compares full gradients by both paths at one SCF;
  - `j_grad_exact.py`, which compares with PySCF's unscreened `int2e_ip1` on the CPU;
  - `ncu_parts.py --part jgrad` and `j_memcheck.py --grad`.
- **Data:** `p24_j_grad.txt`.

### Where the gradient spent its time (part 23)
- **The two-electron term dominated:** 2.50 of 3.04 s (82%) of a PBE/def2-SVP gradient, and 26.28 of 27.21 s
  (97%) at def2-TZVP.
- **It ran the Rys kernel even without exchange:** for pure functionals `jk_energy_per_atom` called
  `RYS_per_atom_jk_ip1` with k_factor 0, through a Rys optimizer built only for it. The SCF's J-engine optimizer was
  already there.

### Method
- **The pair derivatives:** a pair's charge distribution G_i G_j moves with its centers as
  ∂/∂A_x (G_i G_j) = 2a G_{i+1_x} G_j − i_x G_{i−1_x} G_j, and likewise for B. Its Hermite expansion reaches one
  order more than the pair's.
- **The gradient:** E_J = 2 Σ_P D^P·V^P over the J engine's unique pairs, so dE_J/dA = 4 Σ_{P∋A} D_x^{P,A}·V^P.
  - V^P is the pair's Hermite potential to order lij+1.
  - D_x^{P,A} is its derivative Hermite density, built from the E coefficients of the shifted pairs.
- **`MD_build_j_ip1`:** `MD_build_j` with lshift = 1.
  - It first tries the gradient's own kernels (`md_j_generated_ip1`), in which R runs to order L+1.
    - The bra potential to lij+1 comes from the ket density to lkl.
    - The ket potential to lkl+1 comes from the bra density to lij.
    - They come from `gen_md_j.py`'s new `g` specs: 26 classes, (0,0) to (6,3).
  - Otherwise it runs the J kernels with both orders raised and the densities zero-padded. Those take R one order
    higher and spend half of their contraction on zeros.
  - The pair data, the screening and the tasks are the J build's.
- **`Rt_ip1_dot_dm`** (`md_j_grad.cu`): one thread per pair, 12 ms at def2-TZVP.
  - It builds the E coefficients up to (li+1, lj+1, lij+1).
  - It contracts the six derivative densities with V^P and the density, and adds the results to the two atoms.
- **Dispatch:** `get_ej_ip1` runs when k_factor is 0, omega is 0, the library has the functions, and no shells lie
  beyond LMAX.
  - It takes the total density (the sum over spins for UKS), symmetrized; E_J depends only on the symmetric part.
  - It reuses the SCF's `_opt_jengine`.
- **The constant 4:** the prototype measured it as exactly 4.000000 in every component.

### Exactness

| caffeine, Coulomb gradient | max error | RMS error | sum over atoms |
|---|---|---|---|
| def2-SVP, Rys | 7.08e−10 | 2.16e−10 | 5.3e−12 |
| def2-SVP, J engine | 1.62e−10 | 4.44e−11 | 4.8e−14 |
| def2-TZVP, Rys | 1.09e−09 | 3.59e−10 | 1.1e−11 |
| def2-TZVP, J engine | 5.68e−10 | 1.52e−10 | 8.3e−14 |

- **Reference:** the exact gradient from PySCF's `int2e_ip1` on the CPU without screening (8 s at def2-SVP, 121 s at
  def2-TZVP).
- **Result:** the J engine is 2–5× closer to it than the Rys path. It is also translationally invariant to 1e−13,
  where the Rys path is at 1e−11.
- **The two paths:** they differ by up to 6.6e−10 (def2-SVP) and 1.3e−9 (def2-TZVP), on components up to 31.
- **UKS (the caffeine cation, PBE/def2-SVP):** the paths differ by 2.4e−9, all of it the Rys path's error. The
  Rys gradient is 2.4e−9 from the exact one, the J engine 2.4e−11.
- **Several densities:** the J engine needs the total density only, so UKS uses one density as well.

### Timings (GTX 1650 Super)

| caffeine, Coulomb gradient at a fixed density | Rys | raised J kernels | own kernels |
|---|---|---|---|
| def2-SVP | 2.489 s | 1.004 s | 0.718 s (3.5×) |
| def2-TZVP | 26.17 s | 8.749 s | 5.879 s (4.45×) |

| caffeine, full gradient at one SCF | Rys (part 23) | part 24 |
|---|---|---|
| PBE/def2-SVP | 3.04 s | 1.26 s (2.4×) |
| r2SCAN/def2-SVP | 3.47 s | 1.69 s |
| cation UKS PBE/def2-SVP | 3.30 s | 1.61 s |
| PBE/def2-TZVP | 27.23 s | 6.77 s (4.0×) |

- **Where the time goes now:** in the def2-SVP PBE gradient the two-electron term takes 0.72 s of 1.26 s, and the
  XC term 0.47 s.
- **Per class** (def2-TZVP, ncu): every class with its own kernel beats the raised path, even with spills of up to
  30 KB.
  - FP64 work falls from 653 to 391 GFLOP and GPU time from 8.75 to 5.88 s.
  - (6,4) to (6,6) keep the raised path: 77 ms.

### Unit tests and memcheck
- **All pass on the final tree:**
  - gradients: `test_rks_grad.py` (15), `test_uks_grad.py` (12), `test_grid_response.py` (7),
    `test_tdrks_grad.py` (7), `test_vv10_grid.py` (20);
  - J engine: `test_scf_j_engine.py` (8), `test_scf_jk.py` (11);
  - DFT: `test_rks.py` (17, 1 skipped), `test_uks.py` (10).
- **J builds:** unchanged, within 7.1e−15 of part 23's J at the same speed.
- **A use-after-free, found by memcheck and fixed before the commit:**
  - The first version passed `cp.asarray(ao_loc).data.ptr` to `Rt_ip1_dot_dm` inline, so the temporary array was
    freed before the asynchronous kernel read it.
  - CuPy's pool kept the memory, so the results were right, but memcheck (pool off) reported invalid 4-byte reads
    at `md_j_grad.cu`:67.
  - Holding the array until the kernel has run gives 0 errors at def2-SVP and def2-TZVP, with the same gradient
    (3.9e−14).

### Other cards
- **Same structure as the J build:** the gradient runs the J build's tasks, tiles and kernels. Parts 22 and 23's
  notes on SM count and FP64 rate apply to it too.

## GPU4PySCF investigation, part 25: the MD J engine's Boys function without divisions (node1), 2026-11-02
Approved by the user after part 24 ("look into the boys function and the VV10 symmetric pair"); proposed in part 23.
- **Commit:** gpu4pyscf `exp/prim-screen` 0faf2ef (`gvhf-rys/gamma_inc.cu`, from `exp_prim_screen.py --boys new`).
- **Tool:** `boys_check.cu`. It checks CUDA's erf at large arguments, compares the old and new functions in ulps,
  and times both.
- **Data:** `p25_boys.txt`.

### Change (`eval_gamma_inc_fn`, used only by gvhf-md: the J build, its gradient, DF int3c2e, PBC J)
- **Series branch** (t < m/2 + ½): the loop divided once per term (`x *= t / bi`, about 20 terms), and the downward
  recursion divided once per step.
  - Both now multiply by 1/(n + ½), from a 96-entry table in constant memory.
  - All lanes in the loop read the same index, so the reads are broadcast.
  - The terms shrink at least twofold per step (t/(k + ½) < ½), so the index stays below m + 55.
- **erf branch:** skips `erf(√t)` for t ≥ 49, where CUDA's erf returns exactly 1.0. `boys_check.cu` found no
  exception among 2³⁰ points of [7, 40]; erfc(7) = 4e−23.

### Exactness (`boys_check.cu`, m = 0–18, 2²² values of t in [0, 80))
- **Unchanged to the bit:** the tiny-t, erf and erf-skipped branches.
- **Series branch:** F_n differs by at most 3 (m = 1) to 8 (m = 18) ulps, about 1e−15 relative. That is the rounding
  of t·(1/b) against t/b, summed over the series and carried down the recursion.
- **J:** within 7.1e−15 of part 24's J, the same as the noise between two runs of part 24 itself.
- **Against the exact J:** errors identical to part 24's to every printed digit (RMS 7.30e−13 at def2-TZVP,
  4.20e−13 at def2-SVP). The J-engine gradient's errors are unchanged too.
- **SCFs:** same energies to the printed digits, and the same cycle counts.

### Timings (GTX 1650 Super; part 24, then part 25)
| caffeine | part 24 | part 25 |
|---|---|---|
| J at a fixed density, def2-SVP | 0.488 s | 0.381 s (−22%) |
| J at a fixed density, def2-TZVP | 3.890 s | 3.163 s (−19%) |
| J-engine Coulomb gradient, def2-SVP | 0.718 s | 0.578 s |
| J-engine Coulomb gradient, def2-TZVP | 5.84 s | 4.99 s |
| SCF PBE/def2-SVP (J per cycle) | 14.90 s (0.367 s) | 13.41 s (0.286 s) |
| SCF B3LYP/def2-SVP | 30.16 s | 29.00 s |
| SCF PBE/def2-TZVP (J per cycle) | 73.88 s (2.946 s) | 63.85 s (2.399 s) |
| PBE gradient, def2-SVP / def2-TZVP | 1.26 s / 6.82 s | 1.12 s / 5.93 s |

- **Per class (def2-TZVP, ncu):** the low-order classes gain most, (ps|ss) 363 → 270 ms (−26%) and (ds|ss),
  (pp|ss), (dp|ss) −27 to −28%. The higher classes gain 8–15%, since the Boys function is a smaller share there.
  The whole build drops from 3.91 to 3.18 s of GPU time, with warp instructions down from 12.3 to 9.95 G.
- **The microbenchmark** (`boys_check.cu`) puts one evaluation at 1.3–2.2× faster. Examples: m = 6 with t in
  [0, 4] (all series) 3.75 → 1.87 ns; t in [0, 200] 2.19 → 1.46 ns.

### Unit tests and memcheck
- **All pass:** `test_scf_j_engine.py` (8), `test_scf_jk.py` (11), `test_rks.py` (17, 1 skipped), `test_uks.py` (10),
  `test_rks_grad.py` (15), `test_uks_grad.py` (12), `test_df_int3c2e.py` (15) and `test_pbc_scf_j_engine.py` (12).
- **memcheck:** 0 errors for the J build and the J gradient at def2-SVP.

### Not done
- **The quartet prefactor:** `fac / (aij*akl*sqrt(aij+akl))` and `aij*akl/(aij+akl)` cost two divisions and a
  square root per quartet: 33% of (ss|ss)'s FP64 work and 10–13% of the other low-order classes'.
  - With 1/aij and 1/akl kept beside the pair data, one reciprocal and one square root would do.
  - That means changing every unrolled kernel's shared-memory layout, so it is left as an option.

## GPU4PySCF investigation, part 26: VV10's Fock kernel over each pair of grid points once (node1), 2026-11-02
Approved with part 25; proposed in part 20, where VV10 was 93% of a B97M-V SCF.
- **Commit:** gpu4pyscf `exp/prim-screen` a837bd7 (`gdft/vv10.cu`, from `exp_prim_screen.py --vv10 sym`).
- **Tools:**
  - `vv10_check.cu`: the original, a reduction variant and the rotation variant on synthetic data, and a long-double
    reference;
  - `vv10_at_dm.py`: `nr_nlc_vxc` at a fixed density, with memcheck mode.
- **Data:** `p26_vv10.txt`.

### Change
- **The original:** `VXC_vv10nlc_fock_eval_UWE` summed over all ordered pairs of grid points, about 25 FP64
  instructions per pair including a reciprocal.
- **What is symmetric:** Φ_ij = −1/(g_ij g_ji (g_ij + g_ji)), r_ij² and the reciprocal. U and W differ between the
  two sides only through g_ij against g_ji.
- **The new kernel evaluates each unordered pair once and adds to both points,** about 33 instructions per pair.
  - Tiles are 128 points. A block takes one I tile against 4 tiles J ≥ I.
  - In a warp each lane keeps its i and meets the 32 j's of a chunk in turn: at step s, j = (lane + s) mod 32, read
    from shared memory.
  - The j side's three sums travel with their j to the next lane, one shuffle per sum per step. After 32 steps each
    sum is back on its lane, complete for the warp's 32 i's.
  - U, W and E are summed with atomics, then scaled.
  - Diagonal tiles take j > i, and the i = j term on the i side only.
- **A first attempt failed:** reducing the j side over the warp per j made every lane execute 15 FP64 adds and 30
  shuffles per pair. It was 1.05× faster, against 1.43× for the rotation.

### Exactness
- **The terms are the same;** only the sums are shorter.
- **On 60,000 synthetic points** against a long-double reference, the new kernel's errors are 10–40× smaller than
  the original's sequential sums: U 3.5e−15, W 1.8e−15, E 1.8e−15, against 1.4e−13, 1.8e−14 and 2.1e−14.
- **Caffeine:** the NLC energy agrees with the original to 4.1e−15 (0.385229398810) and the potential matrix to
  2.2e−16 (values up to 4.7e−3).
- **SCF B97M-V/def2-SVP:** the same energy, −679.7292741775 Eh, and the same 15 cycles.

### Timings (GTX 1650 Super; caffeine def2-SVP, B97M-V, 294,912 NLC grid points; part 25, then part 26)

| | part 25 | part 26 |
|---|---|---|
| `nr_nlc_vxc` at a fixed density | 29.58 s | 21.16 s (1.40×) |
| gradient: `get_nlc_exc` | 29.76 s | 21.33 s |
| gradient: `get_nlc_exc_full_response` | 66.77 s | 55.45 s |
| SCF (15 cycles) | 496.0 s | 360.6 s (−27%) |

- **The rest of the full-response gradient's NLC time** is the grid-response kernel, unchanged here (see Not done).
- **Tests:** `test_vv10_grid.py` (20) runs in 443 s instead of 598 s. It passes, as do `test_numint.py` (24,
  including `_vv10nlc` against PySCF), `test_rks.py` (17, 1 skipped) and `test_uks.py` (10).
- **memcheck:** 0 errors (benzene).

### Not done
- **The grid response's own all-pair kernel** (`vv10_grad_eval_E_grid_response_offdiagonal`): about 34 s, the rest
  of the full-response gradient's NLC time. Done in part 28.
  - Its pair factor is symmetric too.
  - But it writes per-(atom, point) sums for batches of points, so using the symmetry means restructuring that
    driver.
- **The Hessian's VV10 kernels** (`vv10_hess_eval_UWABCE` and the f_t kernels) are all-pair loops of the same kind. Done in part 29.

## GPU4PySCF investigation, part 27: the J kernels' quartet prefactor without divisions (node1), 2026-11-02
Approved with parts 28–29 ("Let's work on all of those"); found in part 25.
- **Commit:** gpu4pyscf `exp/prim-screen` 9835bad (`exp_prim_screen.py --j-prefactor rsqrt`: the unrolled J kernels
  patched, the generated ones written by `gen_md_j.py --prefactor`).
- **Data:** `p27_prefactor.txt`.

### Change
- **What each quartet did:** it took `fac / (aij*akl*sqrt(aij+akl))` and `aij*akl/(aij+akl)`, two divisions and a
  square root.
- **What it does now:**
  - The pair caches keep 1/aij (a fifth row of Rp, 16 doubles) and 1/akl (a fifth row of Rq, the ket tile's size),
    computed once per pair.
  - The quartet takes one `rsqrt(aij + akl)` (within 1 ulp): theta = aij·akl·rs², and fac ×= (1/aij)(1/akl)·rs.
- **The patch, per kernel** (12 + 11 unrolled kernels, matched and asserted):
  - Everything after the caches moves up: the density cache, the Boys buffer and the zeroing loop's bound.
  - Each launch's shared memory grows by 16·TILEY + 16 doubles; the largest is 57 KB.
  - The generated kernels (J and gradient) get the same layout from the generator. The generic kernels keep the old
    code.
- **The 48 KB limit:** a launch above 48 KB of dynamic shared memory needs `cudaFuncSetAttribute`. The unpatched
  launchers call it only where they were already above that.
  - With the range-separated buffer (omega < 0), `md_j_4dm_3_1` grew from 48,640 to 49,536 bytes and failed with
    "invalid argument" in `test_j_engine_multiple_dms`.
  - Every launch now sets the attribute.

### Exactness
- **Against part 25's J:** within 1.8e−14 (values up to 35).
- **Against the exact J:** the same errors as before (RMS 7.30e−13 at def2-TZVP, 4.20e−13 at def2-SVP), as are the
  J-engine gradient's.
- **SCFs:** same energies to the printed digits, and the same cycle counts.

### Timings (GTX 1650 Super; part 26, then part 27)

| caffeine | part 26 | part 27 |
|---|---|---|
| J at a fixed density, def2-SVP | 0.382 s | 0.364 s (−5%) |
| J at a fixed density, def2-TZVP | 3.165 s | 3.066 s (−3%) |
| J with 2 / 4 densities, def2-SVP | 0.534 / 0.828 s | 0.517 / 0.811 s |
| J-engine gradient, def2-SVP / def2-TZVP | 0.578 / 4.958 s | 0.561 / 4.896 s |
| SCF PBE/def2-SVP, B3LYP/def2-SVP | 13.42 s, 28.96 s | 13.21 s, 28.69 s |
| SCF PBE/def2-TZVP (J per cycle) | 63.81 s (2.396 s) | 62.49 s (2.325 s) |

- **Per class (def2-TZVP, ncu):** (ps|ss) 270 → 251 ms, with its FP64 work 16% lower. (ds|ss) and (pp|ss) are −5%,
  the others −1 to −3%. The build drops from 3.18 to 3.08 s of GPU time, with FP64 work 223 → 211 GFLOP.
- **Smaller than the FP64 count suggested:** after part 25 the low-order kernels are still the Boys function's.
  Their instruction count fell 4–7%.

### Unit tests and memcheck
- **All pass:** `test_scf_j_engine.py` (8), `test_scf_jk.py` (11), `test_rks.py` (17, 1 skipped), `test_uks.py` (10),
  `test_rks_grad.py` (15), `test_uks_grad.py` (12), `test_tdrks.py` (28) and `test_tduks.py` (20). The J-engine test
  passed only after the attribute fix.
- **memcheck:** 0 errors for J, J with 2 densities, and the J gradient.

## GPU4PySCF investigation, part 28: VV10's grid-response gradient term over each pair of points once (node1), 2026-11-02
Approved with parts 27 and 29 ("Let's work on all of those"); left over from part 26.
- **Commits:** gpu4pyscf `exp/prim-screen` 397ebae (`gdft/vv10.cu`, from `exp_prim_screen.py --vv10 sym`, which now
  also writes the new kernel and its entry) and `perf` f63f97d (`grad/rks.py`), merged as 1a034c1.
- **Tools:**
  - `vv10_grid_check.cu`: the original kernel with the driver's post-processing, the new kernel, and a long-double
    reference, on synthetic points grouped by atom;
  - `vv10_at_dm.py --grad`: `get_nlc_exc_full_response` at a fixed density, with memcheck mode.
- **Data:** `p28_vv10_grid.txt`.

### Change
- **What the full-response gradient did** (`get_nlc_exc_full_response`, the term from the grid points' own motion):
  - `vv10_grad_eval_E_grid_response_offdiagonal` filled E_Bgr[B][i] = 1.5 Σ_{j on B} ρw_j Φ_ij f_ij (r_i − r_j) for
    every atom B ≠ atom(i) and every point i, in batches of points (an natm × 3 × batch array).
  - Python set each point's own-atom entry to minus the sum of the others, one atom at a time, and took
    de[B] = Σ_i ρw_i E_Bgr[B][i].
  - Every ordered pair of points was evaluated once.
- **The identity:** K_ij = ρw_i ρw_j Φ_ij f_ij is symmetric. With that diagonal, de[B] = 3 Σ_{j on B} T_j, where
  T_j = Σ_{i on another atom} K_ij (r_i − r_j).
- **What it does now:** `VXC_vv10nlc_grad_grid_response_T` takes each unordered pair once with part 26's rotation
  scheme.
  - Each pair adds K_ij (r_i − r_j) to T_j and subtracts it from T_i.
  - Pairs on the same atom are skipped through each point's atom index (padding gets atom −2).
  - Python sums 3·T per atom with `bincount`.
  - There is no natm × 3 × N array, no batching and no diagonal fix-up.
- **Fallback:** the old path stays when the library lacks the entry (`perf` with the wheel's libraries).

### Exactness
- **Prototype** (60,000 points on 24 atoms): max |Δde| 2.2e−13, with |de| up to 15.9. Against long double, the
  original's error is 2.2e−14 and the new kernel's 3.1e−15 (relative).
- **Caffeine def2-SVP, B97M-V, at a B3LYP density:** max |Δde| 1.4e−17, with |de| up to 2.1e−3.

### Timings (GTX 1650 Super; caffeine def2-SVP, B97M-V, 294,912 NLC grid points; part 27, then part 28)

| | part 27 | part 28 |
|---|---|---|
| `get_nlc_exc_full_response` at a fixed density | 55.43 s | 41.54 s (−25%) |
| the same in `nlc_grad_time.py` (PBE orbitals) | 55.03 s | 41.44 s |
| the kernel alone (prototype, 60,000 points; the original with its fix-up) | 1807.1 ms | 1050.8 ms (1.72×) |

- **What is left of the 41.5 s:** the U, W, E sums by part 26's Fock kernel (about 21 s, as in `get_nlc_exc`), the
  grid response's pair term (about 20 s, down from 34 s), and the AO-side and weight-derivative terms.

### Unit tests and memcheck
- **All pass:** `test_vv10_grid.py` (20, 439 s against 443 s; its molecules are small),
  `test_grid_response.py` (7), `test_rks_grad.py` (15) and `test_uks_grad.py` (12).
- **memcheck:** 0 errors (`get_nlc_exc_full_response`, benzene, core guess).

## GPU4PySCF investigation, part 29: the Hessian's VV10 kernels with more trials per pair evaluation (node1), 2026-11-02
Approved with parts 27 and 28 ("Let's work on all of those").
- **Commit:** gpu4pyscf `exp/prim-screen` b03af79 (`gdft/vv10.cu`, from `exp_prim_screen.py --vv10-hess wide`).
- **Tools:**
  - `hess_nlc_parts.py`: a VV10 Hessian with synchronizing timers around its steps, the VV10 parts and every libgdft
    VV10 entry (with its grid and trial counts), and a memcheck mode (water, small grids);
  - `vv10_ft_check.cu`, `vv10_uwabce_check.cu`, `vv10_euw_w1_check.cu`: each kernel's original against the new
    version on synthetic data, with long-double references where the arithmetic changes.
- **Data:** `p29_vv10_hess.txt`.

### Where the time went
Benzene def2-SVP, B97M-V, RKS without grid response (the default), on part 28's build (GTX 1650 Super). The NLC grid
has 143,616 points, 125,886 after screening.
- **The Hessian took 1222.7 s** (the SCF before it 54.7 s), and VV10 took 1138 s of it (93%):
  - f_t, 980 s (80%): 136.5 s in the energy term (`_get_enlc_deriv2`, 36 trials), 136.4 s in H1
    (`_get_vnlc_deriv1`, 36) and 707.4 s in the CPHF (6 Krylov cycles, 36 down to 7 trials);
  - UWABCE, 80.3 s: 8 calls, 6 of them in the CPHF, which recomputes these ground-state sums every cycle;
  - EUW_with_weight1, 66.6 s (H1, 36 derivatives), and the EUW grid response, 11.2 s (H1).
- **`_get_vnlc_deriv1` sets `grid_response = True`** whatever the Hessian's setting. So its grid-response kernels
  always run, and its f_t inputs differ from the energy term's: there is no duplicate call to share.

### Change
- **f_t** (`vv10_hess_eval_f_t_offdiagonal_kernel`, the response to trial densities):
  - For each point i and trial t it sums w_j M_ij (ρ_t, γ_t)[t][j] over all j. M_ij is a 2×2 pair matrix of one
    division and about 50 other operations; each trial then costs 6 (3 per output).
  - A thread owned one i and 6 trials, so M_ij was evaluated once per 6 trials.
  - Now a thread owns up to 36 trials: the fewest chunks of at most 36, each rounded up to a multiple of 6. w_j is
    folded into M_ij, so each output takes two FMAs per trial.
  - **Unused trial slots are not free:** the predicated-off FP64 instructions still take their issue slots. 36 per
    thread at 9 trials is 0.82× the original; hence the chunks sized to the trial count.
- **UWABCE** (`vv10_hess_eval_UWABCE_kernel`, the ground-state sums): each pair of points once, part 26's rotation with
  six sums instead of three.
- **EUW_with_weight1** (the grid weights' derivatives, 3·natm of them, in `_get_vnlc_deriv1`): the same kernel with
  up to 24 derivatives per thread instead of 6 (three sums each, so 36 would not fit in the registers). Only the
  launcher changes; the results are bitwise equal.

### Exactness
- **f_t:** folding w_j in puts two roundings per term into each running sum instead of one.
  - On 20,000 synthetic points the outputs move by at most 2.3e−14 of the largest.
  - Against long double, the largest relative error at 64 sampled (point, trial) pairs grows 2–6×: 7.3e−14 against
    3.3e−14 at 36 trials, 1.1e−13 against 6.1e−14 at 72, and 1.4e−12 against 2.2e−13 at 9 (probably a sample whose
    terms cancel).
  - Wider chunks alone keep the original arithmetic and are bitwise equal, but give 1.86× instead of 2.66×.
- **UWABCE:** 10–100× closer to long double than the original's sequential sums (U 3.5e−15 against 1.4e−13, A 3.4e−15
  against 2.8e−13).
- **EUW_with_weight1:** bitwise equal.
- **The benzene Hessian:** within 5.3e−12 of part 28's (values up to 0.76). The SCF before it does not use these
  kernels: the same energy, −232.0857485935 Eh, and the same 9 cycles.

### Timings
| benzene def2-SVP, B97M-V (GTX 1650 Super) | part 28 | part 29 |
|---|---|---|
| the Hessian | 1222.7 s | 582.9 s (2.10×) |
| energy term (`_get_enlc_deriv2`) | 151.7 s | 64.4 s |
| H1 term (`_get_vnlc_deriv1`) | 236.4 s | 124.0 s |
| CPHF's NLC response (`nr_rks_fnlc_mo`, 6 calls) | 806.8 s | 367.6 s |
| f_t, 8 calls | 980.3 s | 383.7 s (2.55×) |
| UWABCE, 8 calls | 80.3 s | 63.5 s (1.26×) |
| EUW_with_weight1 | 66.6 s | 41.4 s (1.61×) |
| EUW grid response | 11.2 s | 11.2 s |

- **The kernels on synthetic data** (prototypes):
  - f_t: 2.66× at 36 trials, 2.63× at 72, 1.74× at 9 (20,000 points);
  - UWABCE: 1.28× (60,000 points);
  - EUW_with_weight1: 1.63× at 36 derivatives, 1.73× at 72.
- **What is left of the 583 s:** f_t 384 s (66%), UWABCE 64 s, EUW_with_weight1 41 s, the EUW grid response 11 s,
  and 83 s outside VV10.
- **f_t is near its floor:** each pair still costs 4 FMAs per trial, plus its matrix (about 59 operations) once per
  chunk of up to 36 trials. At 36 trials that is about 1.4× the trial work alone.

### Unit tests and memcheck
- **All pass:** `test_vv10_hessian.py` (14, 605 s), `test_tdrks_vv10.py` (6) and the two VV10 tests of
  `test_polarizability.py`.
- **memcheck:** 0 errors.
  - The three prototypes ran on 1000 points with 40 trials or derivatives, so the last chunk is partial.
  - A water Hessian ran on small grids, including the CPHF's response.

### Not done
- **The CPHF recomputes the ground-state sums every cycle:** `nr_rks_fnlc_mo` evaluates ρ0 and runs UWABCE on each
  call, 6 times in the benzene Hessian (47.7 s of 583 s after this part; once would save about 40 s). Done in part 30. Computing them once per response function
  is exact, but it changes `nr_rks_fnlc_mo`'s interface and its callers (the Hessian's CPHF, `gen_response` for TDDFT
  and polarizabilities).
- **EUW_grid_response_offdiagonal** (11 s, 1%): a per-(atom, point) layout like the gradient's before part 28.

## GPU4PySCF investigation, part 30: VV10's response, its ground-state part once per solve (node1), 2026-11-02
Proposed in part 29 (Not done); the user approved it after asking how its memory scales.
- **Commits:** gpu4pyscf `perf` eb34eb5 (`hessian/rks.py`, `hessian/uks.py`, `scf/_response_functions.py`), merged
  into `exp/prim-screen` as 04c5428. Python only, so it works with the wheel's libraries too.
- **Tools:** `hess_nlc_parts.py --task tddft|polar` (TDDFT with `exclude_nlc = False`, the static polarizability).
- **Data:** `p30_vv10_resp.txt`.

### Change
- **What was repeated:** `nr_rks_fnlc_mo` (VV10's term of the CPHF and TDDFT response) runs once per Krylov or
  Davidson cycle. On each call it evaluated ρ0 and ∇ρ0 on the NLC grid (a pass over the AOs), ω, κ and their
  derivatives, and the all-pair U, W, A, B, C kernel, all of which depend only on the ground state.
- **The split:** `get_nlc_response_ground_state` builds that part, an `NLCResponseGroundState` (a named tuple of
  the masked grid, ρ, ∇ρ, ω, κ and their derivatives, U..C, and one combined factor).
  - `nr_rks_fnlc_mo` and `nr_uks_fnlc_mo` take it as `ground_state`. They build it themselves when it isn't given,
    so other callers see no change.
  - A `kept` list argument builds it on the first call and returns it after.
- **Who keeps it:**
  - `gen_response`'s closures (the orbital Hessian, singlet TDDFT, UKS) keep it from their first call. The triplet
    response has no VV10 term.
  - The RKS and UKS Hessians' `gen_vind` wraps its fx. While fx runs, the kept list sits on the Hessian object,
    where `get_veff_resp_mo` finds it.
  - It can't be an argument: the PCM and SMD Hessians wrap `get_veff_resp_mo` with a fixed signature.
  - IR and Raman call `gen_vind`, so they get it too. Outside fx, `get_veff_resp_mo` builds a fresh one, as before.
- **Memory:** 23 doubles per NLC grid point after screening, plus a 1-byte mask over the full grid.
  - Linear in the number of atoms, about 2 MB per atom at the default NLC grid (about 12,000 points per atom, 88%
    kept). Benzene: 23 MB.
  - Independent of the basis set and of the number of trial vectors.
  - About the size of one trial density's buffers in the same function. The batches are sized from free memory.
  - It is held for one solve.
- **gpu4pyscf's TDDFT leaves VV10's response out by default** (`exclude_nlc = True`, "the nlc contribution is small
  in TDDFT"). For TDDFT this part applies only with `exclude_nlc = False`. The CPHF-based paths (Hessians,
  polarizabilities, IR, Raman) always include VV10's response.

### Exactness
- **The same arrays, computed once instead of every cycle.** The U..C kernel uses atomics, so each evaluation already
  differed from the last at the rounding level.
- **Benzene def2-SVP, B97M-V:**
  - the Hessian is within 3.5e−12 of part 29's (values up to 0.76);
  - the 5 excitation energies are within 1.2e−13;
  - the polarizability is within 1.2e−11 (values up to 74).

### Timings (GTX 1650 Super; benzene def2-SVP, B97M-V, 125,886 NLC points after screening)

| | part 29 | part 30 | U..C kernel calls |
|---|---|---|---|
| the Hessian | 582.9 s | 541.9 s (−7%) | 8 → 3 |
| TDDFT, 5 states, `exclude_nlc = False` | 339.6 s | 259.6 s (1.31×) | 11 → 1 |
| the static polarizability | 300.4 s | 228.3 s (1.32×) | 10 → 1 |

- **The saving is the kernel:** 7.9 s per call, 27% of each TDDFT cycle's VV10 response (5–6 trial vectors per
  cycle) and 27% of the polarizability's (2–3).
- **In the Hessian the trial counts are larger** (up to 36), so f_t dominates each cycle and the saving is 7%.
- **The ground-state AO pass is small:** `get_nlc_response_ground_state` took 8.0 s in all, 7.94 s of it the
  kernel.

### Unit tests
- **All pass:**
  - `test_vv10_hessian.py` (14, 547 s against 605 s);
  - `test_tdrks_vv10.py` (6, 138 s against 158 s);
  - the VV10 tests of `test_polarizability.py` (2), `test_rks_hessian_grid_response.py` (1) and
    `test_uks_hessian_grid_response.py` (2);
  - `test_raman_wb97mv_densityfitting`. The direct-SCF `test_raman_wb97mv` is skipped upstream ("Too slow").

## GPU4PySCF overview: unmodified v1.9.0 against parts 1–30 (node1), 2026-11-02
Asked for by the user ("how much faster different things have become … compared to the unmodified version").
- **Trees:** `gpu4pyscf-base` (tag v1.9.0, built from source with the same toolchain) against the main tree,
  `exp/prim-screen` 04c5428.
- **How:**
  - The same workloads ran back to back, v1.9.0 first, every timed section at 1530 MHz.
  - Each run warms up on water with the same settings, then times the SCF, gradient, Hessian and TDDFT from one
    converged SCF.
  - RKS, `conv_tol` 1e-10, default settings.
- **Tools:** `overview.py`, `overview_table.py`, `grad_at_mo.py`.
- **Data:** `p31_overview.txt`.

| workload | task | v1.9.0 | current | speedup | largest difference |
|---|---|---|---|---|---|
| caffeine def2-svp PBE, 246 AOs | SCF (18 cycles) | 25.2 s | 13.2 s | 1.91× | 9.1e-13 Eh |
|  | gradient | 6.9 s | 1.1 s | 6.25× | 6.8e-10 Eh/bohr (values up to 0.15) |
|  | Hessian | 260.2 s | 159.7 s | 1.63× | 2.5e-11 Eh/bohr² (values up to 1) |
| caffeine def2-tzvp PBE, 494 AOs | SCF (18 cycles) | 104.8 s | 62.6 s | 1.67× | 9.1e-13 Eh |
|  | gradient | 85.7 s | 5.9 s | 14.63× | 1.0e-09 Eh/bohr (values up to 0.14) |
| caffeine def2-svp R2SCAN, 246 AOs | SCF (16 cycles) | 37.1 s | 23.6 s | 1.57× | 1.1e-12 Eh |
|  | gradient | 8.8 s | 1.5 s | 5.73× | 7.8e-10 Eh/bohr (values up to 0.15) |
|  | Hessian | 450.8 s | 211.5 s | 2.13× | 2.1e-11 Eh/bohr² (values up to 1) |
| caffeine def2-tzvp B3LYP, 494 AOs | SCF (15 cycles) | 321.6 s | 185.5 s | 1.73× | 6.8e-13 Eh |
|  | gradient | 89.2 s | 28.9 s | 3.09× | 6.4e-12 Eh/bohr (values up to 0.15) |
| caffeine def2-svp B3LYP (DF), 246 AOs | SCF (15 cycles) | 14.6 s | 10.9 s | 1.34× | 2.3e-13 Eh |
|  | gradient | 3.0 s | 1.8 s | 1.67× | 2.4e-08 Eh/bohr (values up to 0.16) |
|  | Hessian | 199.7 s | 151.5 s | 1.32× | 1.2e-10 Eh/bohr² (values up to 1) |
| caffeine def2-svp B97M-V, 246 AOs | SCF (15 cycles) | 514.4 s | 360.5 s | 1.43× | 1.4e-12 Eh |
|  | gradient | 39.7 s | 22.8 s | 1.74× | 6.9e-10 Eh/bohr (values up to 0.15) |
| benzene def2-svp B97M-V, 114 AOs | SCF (9 cycles) | 78.3 s | 54.4 s | 1.44× | 1.1e-13 Eh |
|  | gradient | 9.4 s | 5.5 s | 1.70× | 1.1e-10 Eh/bohr (values up to 0.0029) |
|  | Hessian | 1244.4 s | 541.6 s | 2.30× | 5.2e-12 Eh/bohr² (values up to 0.76) |
| caffeine def2-svp B3LYP, 246 AOs | SCF (15 cycles) | 57.9 s | 24.4 s | 2.37× | 2.3e-13 Eh |
|  | gradient | 7.1 s | 3.2 s | 2.24× | 2.5e-12 Eh/bohr (values up to 0.16) |
|  | Hessian | 304.2 s | 146.3 s | 2.08× | 1.9e-11 Eh/bohr² (values up to 1) |
|  | TDDFT, 5 states | 70.0 s | 47.8 s | 1.47× | 2.3e-14 Eh |

- **Per SCF cycle:** the cycle counts are identical, so the per-cycle speedups are the same as the totals. For
  example, B3LYP/def2-SVP goes from 3.86 s to 1.63 s per cycle, and PBE/def2-TZVP from 5.82 s to 3.48 s.
- **Opt-in one-pass J+K** (`jk_one_pass = True`, still off by default): the B3LYP SCF is 2.78× (def2-SVP, 20.9 s)
  and 1.87× (def2-TZVP, 172.0 s) faster than v1.9.0. The energies agree to 5.0e−12 and 1.6e−11.
- **Exactness:**
  - SCF energies agree within 1.4e−12 Eh, Hessians within 2.5e−11 (DF 1.2e−10), and excitation energies within
    2.3e−14.
  - Hybrid gradients agree within 6.4e−12.
  - **Pure-functional gradients** (PBE, r2SCAN, B97M-V) differ by 7e−10 to 1e−9. At fixed orbitals the difference
    remains (6.8e−10 for PBE), so it is the code: the J-engine Coulomb gradient (part 24) replaces v1.9.0's screened
    Rys one, and part 24 measured it 2–5× closer to the exact integrals.
  - **The DF gradient** differs by 2.4e−8. At fixed orbitals the code agrees to 7.8e−14. The rest is the DF SCF's
    run-to-run variation: v1.9.0 against itself differs by 1.1e−8 with identical energies.
- **Where the gains come from:**
  - Hybrid SCFs: the K kernels (parts 1–2, 5–7), the J engine (parts 22–27) and the XC part (parts 19–21).
  - Pure-functional SCFs: the J engine and the XC part. VV10: part 26.
  - Pure-functional gradients: the J engine's nuclear gradient (part 24), plus parts 19–20.
  - Hybrid gradients: the derivative kernels (parts 3 and 8).
  - Hessians:
    - the CPHF: one-pass J+K (part 9) and its XC response (part 10);
    - the XC derivative terms (parts 12–14);
    - the generated second-derivative kernels (part 16);
    - VV10 (parts 29–30).
  - DF runs get only the XC changes, so 1.3–1.7×.
- **Found on the way:** GPU4PySCF's TDDFT sizes its trial space from `max_memory`, a CPU-memory setting that
  defaults to 4000 MB.
  - For caffeine/def2-SVP that asks for about 7 GB of GPU memory, so both trees fail on the 4 GB card.
  - The TDDFT rows use `td.max_memory = 1000`.
  - SCF + gradient + Hessian + TDDFT in one process also ran out of memory in both trees, so that workload ran as
    two processes.
- **Not covered here:** UKS, TDDFT gradients, larger systems, and data-center GPUs.
  - The 1650 Super runs FP64 at 1/32 of FP32, so the changes that cut FP64 work (VV10, the Boys function, the J
    prefactor) may give other ratios on an A100 or H100.

## GPU4PySCF check: broken-symmetry singlets (node1), 2026-11-02
The user asked whether symmetry-broken singlets had been validated. They had not.
- **What the earlier UKS checks covered:**
  - doublet cations (α ≠ β, but different electron counts);
  - spin-0 closed shells (α = β);
  - the unit tests.
- **What only a broken-symmetry singlet tests:**
  - shortcuts that take spin 0 to mean α = β;
  - density screening of the two spins' K with one spin's bound. In a broken-symmetry singlet α and β sit on
    different centres.
- **Code reading:** the Python diff since v1.9.0 has no spin-0 shortcut. K's screening bound is
  `condense('absmax', dms)`, the maximum over both spin densities (upstream code, used unchanged).
- **Tools:**
  - `bs_singlet.py`: UKS from a saved guess, the same in every tree; ⟨S²⟩, spin populations of the halves,
    gradient, Hessian, TDA; fixed-orbital comparisons; CPU PySCF mode;
  - `bs_table.py`; `hess_cpu_gpu.py`.
- **Data:** `p32_bs_singlets.txt`.

### Cases
- **Molecules:** twisted ethylene, and two benzyl radicals stacked 3.8 Å apart (28 atoms, 266 AOs: α on one molecule,
  β on the other). Both start from a HOMO/LUMO 45° mix of an RKS B3LYP guess.
- **Methods:** PBE, B3LYP, r2SCAN, B97M-V and DF-B3LYP; SCF, gradient, Hessian, and TDA (5 states) from the reference.
- **Every run reached a genuine broken-symmetry state:** ⟨S²⟩ 1.010–1.100, and spin populations of about ±1.0 on the
  two halves.
  - Both trees, and CPU PySCF, reached the same state in the same number of cycles.

### Current against v1.9.0
- **Energies** within 1.6e−12, **⟨S²⟩** within 4e−13 and **Hessians** within 1.8e−11.
- **TDA excitation energies** within 4e−14.
- **Gradients:**
  - hybrids within 1.6e−12;
  - pure functionals within 1.3e−9: the J-engine Coulomb gradient, which differs from v1.9.0's Rys one at fixed
    orbitals too (part 24);
  - DF 2.5e−9, but 3.2e−14 at fixed orbitals: the DF SCF's run-to-run variation.

### Against CPU PySCF 2.14 (independent)
- **Energies** within 1.6e−11 (B97M-V 1.6e−8). **Gradients at the same orbitals** within 6.6e−10 (B97M-V 6.3e−9, DF
  4.1e−8). **TDA** within 4.5e−7.
- **With each code's own orbitals the gradients differ by up to 1.4e−6.** A broken-symmetry state's soft
  spin-polarization mode lets the two codes' DIIS stop at orbitals that differ at that level within
  `conv_tol_grad`.

### Upstream findings (the same in v1.9.0)
- **GPU4PySCF does not break spin symmetry by default.**
  - `init_guess_breaksym` is 0, where PySCF's is 1.
  - For twisted ethylene, GPU4PySCF's default spin-0 UKS guess did not converge in 200 cycles.
  - With `mf.init_guess_breaksym = True` it converges in 25 cycles to the same broken-symmetry state that PySCF's
    default reaches.
  - **So spin-0 UKS of a diradical in GPU4PySCF needs `init_guess_breaksym = True` (or a broken-symmetry guess).**
- **The Hessian's CPHF tolerance:** GPU4PySCF's `conv_tol_cpscf` defaults to 1e−6, against PySCF's 1e−8.
  - For the broken-symmetry singlet that leaves the Hessian 6.9e−6 from its 1e−10 value; for the triplet, 2.4e−6.
- **PySCF's and GPU4PySCF's Hessians differ at the 1e−5 level for any state.** For example, closed-shell RKS water
  differs by 2.2e−5 with matched CPHF tolerances and grids.
  - Part of it is GPU4PySCF's coarser default `cphf_grids`, (50, 194) SG-1.
  - The rest is not pinned down; it is about 3e−5 relative, or ~0.05 cm⁻¹ in a frequency.
- **The DF UKS Hessian of the 28-atom dimer runs out of the 4 GB card in both trees.**

## GPU4PySCF investigation, part 33: the ωB97M-V SCF, and VV10's Fock kernel with a lean inner loop (node1), 2026-11-02
Asked for by the user ("a deep dive into the SCF procedure of wB97M-V … specifically for the GPU variant").
- **Commit:** gpu4pyscf `exp/prim-screen` 29bfd96 (`gdft/vv10.cu`, from `exp_prim_screen.py --vv10-fock lean`).
- **Tools:**
  - `scf_parts.py` now times VV10 (`nr_nlc_vxc`) as its own part;
  - `vv10_fock2_check.cu`: part 26's kernel against the lean variants, and the library version between markers;
  - `rcp_newton_check.cu`, `rsh_k_split.py`, `nlc_grid_levels.py`.
- **Data:** `p33_vv10_lean.txt`.

### Where the ωB97M-V SCF goes (caffeine def2-SVP, part 30)
- **360 s for 14 cycles, about 25.7 s per cycle:**
  - VV10 20.98 s (87%);
  - the range-separated K 1.68 s (7%);
  - the meta-GGA XC 1.07 s (4%: density 0.34, integration 0.80);
  - J 0.28 s (1%).
- **The NLC grid is the main grid's, level 3** (294,912 points), the same as PySCF's default.
- **The K is already one pass:** `RYS_build_k` evaluates 0.15/r + 0.85·erf(ωr)/r with two sets of Rys roots per
  quartet. That is 1.72× a full-range K (1.713 against 0.998 s).

### Change: VV10's Fock kernel
- **Part 26's rotation kernel ran 78 instructions per pair of grid points, 34 of them FP64.**
  - The diagonal tile's tests (j ≥ i, j > i) branched on every pair; about 17 instructions went to branch bookkeeping.
  - The lane-rotation index took a signed `% 32`, about 7 instructions.
  - g_ij was recomputed in each branch.
- **The lean version:**
  - Off-diagonal tiles (all but 1 in ntiles) run a loop with no tests; the diagonal tile keeps them.
  - The index is `(lane + s) & 31`, and the tile's six values are read as three `double2`.
  - The reciprocal is `rcp.approx.ftz.f64` and two Newton steps, without the IEEE division's slow-path test. It is
    bitwise the correctly rounded 1/x on 2²⁸ values each of [1e−300, 1e300], [1e−3, 1e18] and [1e300, 1e307].
  - The sums are S_E = Σ ρw_j inv and S_U = Σ ρw_j inv² (g_ji g_sum + g_ij g_ji), and S_W likewise with r²; they are
    scaled by 1.5, 1.5 and −1.5.
- **The result:** 46.5 instructions per pair, 28 of them FP64.
  - It is 1.23× faster than part 26 on synthetic data (920 against 1133 ms, 60,000 points), with the same errors
    against long double.
  - Packed loads alone change nothing on this FP64-bound card (949 against 948 ms). They reduce the instruction count,
    which matters where issue rather than FP64 is the limit.

### Exactness
- **The NLC energy** at a fixed density is identical to 12 digits; the potential agrees to 8.7e−19.
- **The SCFs** reach the same energies (within 1e−12 Eh) in the same cycles. The gradients agree within 3e−13.

### Timings (GTX 1650 Super, caffeine def2-SVP; part 30, then part 33)

| | part 30 | part 33 |
|---|---|---|
| `nr_nlc_vxc` at a fixed density (ωB97M-V) | 21.15 s | 17.27 s (1.22×) |
| ωB97M-V SCF (14 cycles) | 358.3 s | 299.9 s (1.19×) |
| ωB97M-V gradient | 29.6 s | 25.7 s |
| B97M-V SCF (15 cycles) | 360.4 s | 298.3 s (1.21×) |
| B97M-V gradient | 22.8 s | 18.9 s |

- **After this part:** VV10 takes 17.25 of about 21.7 s per cycle (85%); K 1.69 s, XC 1.08 s, J 0.28 s.
- **Tests:** `test_numint.py` (24), `test_rks.py` (17, 1 skipped), `test_uks.py` (10), `test_vv10_grid.py` (20,
  368 s against 439 s) and `test_tdrks_vv10.py` (6) pass. memcheck: 0 errors.

### Looked at, not done
- **K as two passes, K(erf) + K(erfc):**
  - The erfc pass has its own distance screening, but at caffeine's size it removes little.
  - K(erf) costs 0.997 s and K(erfc) 1.726 s, so the two together are 0.63× the one pass.
  - It could pay off only for much larger molecules.
  - The two passes agree with the one pass to 2.2e−13. So the one pass's screening, with the erf part's Schwarz
    bounds for an operator that is 15% full-range, costs at most ~2e−13 in K.
- **The meta-GGA XC (4%):**
  - τ takes three products per block next to the GGA-like one, and the AOs are evaluated twice per cycle (density,
    then integration).
  - Merging the products, or keeping the AOs (2.3 GB for caffeine), would gain at most ~1% here.
- **VV10's pair arithmetic is near its floor:** 28 FP64 per pair. r² from precomputed |r|² would save 2 more but
  loses precision for close pairs.

### The NLC grid (an approximation; for information)
ωB97M-V on caffeine/def2-SVP with a coarser NLC grid:

| NLC grid | points | SCF | ΔE | max Δgradient |
|---|---|---|---|---|
| level 3 (default) | 294,912 | 297.6 s | – | – |
| level 2 | 209,920 | 172.4 s | +2.4e−7 Eh | 1.3e−6 |
| SG-1, (50, 194) (Q-Chem's default for VV10) | 102,400 | 76.0 s | +1.6e−6 Eh | 5.3e−6 |
| level 1 | 97,024 | 71.4 s | −2.8e−6 Eh | 1.2e−5 |
| level 0 | 21,504 | 42.7 s | −2.3e−4 Eh | 4.1e−5 |

- **Settings:** `mf.nlcgrids.atom_grid = (50, 194)` and `mf.nlcgrids.prune = gpu4pyscf.dft.gen_grid.sg1_prune` give
  SG-1; `mf.nlcgrids.level = 2` gives level 2.
- **The user's decision (2026-11-02):** not a default, since the work here is exact changes, but it stays in view for
  future approximate-mode options.
- **Hessians too:** the Hessian's VV10 kernels (f_t, UWABCE) also scale with the square of the NLC grid's points. Not
  measured.

## GPU4PySCF investigation, part 34: the exchange kernels' atomics - measured, __ldg tried, density lanes modelled (node1), 2026-11-02
Asked for by the user ("Could you try the exchange kernels idea?"): part 17's option 2, outputs that consecutive tasks
share kept in registers instead of one FP64 atomic per element. Nothing in GPU4PySCF changed.
- **Tools:**
  - `dlane_check.cu`: the density phase of a multi-density J+K build, current against density lanes;
  - `exp_prim_screen.py --dm-ldg` (off by default);
  - the profiles reuse part 17's tools (`ncu_parts.py`, `ncu_summary.py --l2`, `cphf_jk_parts.py`).
- **Data:** `p34_exchange_atomics.txt`.

### Where atomics matter (caffeine def2-SVP, B3LYP, current build)

| | K at one density (the SCF's) | CPHF J+K at 72 densities |
|---|---|---|
| GPU time | 1.01 s | 6.99 s |
| FP64 work | 59.3 GFLOP | 358 GFLOP |
| atomic warp instructions | 14 M | 1354 M (about 41 G adds) |
| L2 atomic units busy | 3.5% | 49% (47–57% in the top classes) |
| FP64 pipe busy | 82–90% | 38–88% by class |
| main stalls | | long scoreboard 59%, lg throttle 20% |

- **The SCF's K is bound by FP64.** Its atomics hardly load the memory system, so keeping outputs in registers cannot
  speed up SCF exchange.
  - Part 17's "atomics are half the floor" holds for the multi-density builds only: the Hessian's CPHF, TDDFT and
    polarizabilities.
- **The CPHF call is unchanged since part 17:** 1.30 s at 1 density, 7.26 s at 72 (85 ms per density).

### __ldg on the density loads (part 17's option 1)
- **Applied** to the unrolled K and J+K kernels (1372 and 2080 loads), the generated K kernels (21,560) and the
  generic kernels' `load_dm`.
- **Exact:**
  - J/K within 5e−14 at 1, 8 and 72 densities, and short-range K within 2e−15;
  - the Hessian within 1.0e−11 and TDDFT within 4.5e−15;
  - `test_scf_jk`, `test_jk_one_pass`, `test_rks`, `test_uks`, `test_rks_hessian` and `test_tdrks` pass.
- **No faster:**
  - the CPHF call 7.26 → 7.34 s at 72 densities, 1.30 → 1.29 s at 1;
  - K 1.000 → 1.004 s;
  - the Hessian 146.0 → 146.2 s, TDDFT 47.8 → 47.6 s.
- **So the loads did not wait on the order the compiler kept,** contrary to part 17's reading of the SASS; the memory
  system itself is the bound. The option stays in the generator, off.

### Density lanes, modelled (`dlane_check.cu`: the density phase only, integrals from a hash)
- **The design:**
  - a block takes one bra pair, with ket pairs sorted by k;
  - a batch of integrals is staged in shared memory;
  - each lane takes one density, with densities laid out fastest;
  - J_ij stays in registers for the block, and K_ik and K_jk while k repeats;
  - the remaining atomics are coalesced.
- **s functions (nf 1, 120 shells):** 3.12× at 32 densities and 2.97× at 64. The current pattern runs at 10× the
  FP64 floor; lanes run at 3.3×.
- **p functions (nf 3, 40 shells):** 1.06× and 1.10×. Both versions are near the FP64 floor (172 ms for 486
  multiply-adds per quartet and density); lanes spill registers in this first version.
- **The ceiling for the real build:**
  - Every CPHF launch at 90% FP64-busy with the same FP64 work would give 6.99 → 4.38 s (1.60×).
  - That is about 13 s of the 146 s B3LYP Hessian at best. Realistically about 6%; TDDFT more; the SCF and gradients
    nothing.

### Decision
- **Not built:** density lanes would mean a new density phase for every K and J+K kernel family (unrolled, generated,
  generic, multi-thread), the transposed density layout and k-sorted task lists, for at most 1.6× on the multi-density
  builds on this card.
- **Where it should pay more:** on cards with fast FP64 (A100, H100) the FP64 floor drops about 16× against memory, so
  most classes would sit where the s-function model gains 3×. `dlane_check.cu` runs in seconds there.

## GPU4PySCF investigation, part 35: translational invariance in the derivative kernels (node1), 2026-11-03
Asked for by the user (option 1 of the symmetries question; "Yes" to measuring it first): a shell quartet's
derivatives with respect to its four centers sum to zero, so one of them follows from the other three.
- **Tools:**
  - `gen_rys_ejk.py --drop-j`, as `exp_prim_screen.py --ejk-drop j` (now the default);
  - `gen_rys_ejk_ip2.py --drop-l`, as `exp_prim_screen.py --ip2-drop l` (off);
  - the A/B is `scratch/p35_ab.sh` (`deriv_at_dm.py`, `overview.py`, unit tests); the profiles use `ncu_parts.py`.
- **Data:** `p35_translational.txt`.

### Why it is exact per root and per tuple
- At a fixed Rys root, the 2D integral I_x(t) of a tuple t = (ix, jx, kx, lx) depends on the centers' relative
  positions only. The root itself depends on ρ|P−Q|², which a common shift leaves unchanged.
- Its derivatives with respect to the four centers' x coordinates are the factors the kernels already form,
  f_i(t) = 2a_i I_x(ix+1) − ix I_x(ix−1), and so on. So f_i + f_j + f_k + f_l = 0 for every tuple and root.
- The same holds for each thread's sums per quartet, since each is Σ_t f_c(t) S(t) over that thread's tuples.
- A derivative formed as minus the other three carries their rounding. It is not an approximation.

### Where the gradient's time goes (ncu, the build before this part)
- **caffeine def2-TZVP, B3LYP's J + 0.2 K:** 27.9 s of GPU time under ncu.
  - 99.3% is in the generated kernels, with the FP64 pipe 80–91% busy in the large classes.
  - The generic kernel, for (ff|ff) only, takes 0.7%. No upstream unrolled kernel runs.
- **def2-SVP:** 2.6 s, all of it in generated kernels.
- **What this means for the idea:** part 8's kernels already sum per tuple. Upstream's kernels add 12 derivative
  terms per output, where dropping a center would cut a quarter of that work; those kernels no longer run.
  - What is left to drop is center j's factor and FMA per tuple and direction: 3–4 of the ~14 FP64 operations.
  - The per-output work (2 FP64 per output and direction) stays.

### The gradient kernels (`--drop-j`)
- **The change:**
  - j's factor and FMA are left out of every tuple.
  - v_j now accumulates the sum over tasks of v_k + v_l, and j's atom gets −(v_i + v_j). This needs no extra
    registers, and j's atomics stay once per bra pair.
  - The run model's cost per tuple drops from 14 to 10 FP64, so the runs are cut at slightly different places.
- **Model:** the output phase's FP64 per thread (the largest thread of a quartet) drops 10–31% by class.
- **Registers:** 255 or below, as before; 28 of the 63 kernels changed.
  - The stack (spills) shrank in 21, e.g. (fp|ff) 888 → 656 bytes and (fd|ff) 2576 → 2248.
  - It grew in 5, e.g. (ff|dd) 904 → 1072 and (ff|fd) 2784 → 2856.
  - (ps|ss) and (ds|ss) need fewer registers (174 → 168 and 230 → 168).
- **The stage at fixed densities** (caffeine, B3LYP density, `deriv_at_dm.py`; median of 3, two interleaved
  rounds, 1530 MHz):

  | build | before | after | max \|Δ\| (values up to) |
  |---|---|---|---|
  | def2-SVP, J + K | 2.596 s | 2.317 s (−10.7%) | 2.1e−12 (30) |
  | def2-SVP, B3LYP's J + 0.2 K | 2.595 s | 2.317 s (−10.7%) | 4.0e−12 (31) |
  | def2-SVP, short-range K (ω = −0.3) | 4.680 s | 4.157 s (−11.2%) | 3.1e−14 (0.35) |
  | def2-TZVP, J + K | 27.755 s | 25.081 s (−9.6%) | 8.6e−12 (27) |
  | def2-TZVP, B3LYP's J + 0.2 K | 27.760 s | 25.086 s (−9.6%) | 7.0e−12 (28) |
  | def2-TZVP, short-range K | 50.17 s | 45.09 s (−10.1%) | 1.9e−13 (0.30) |
  | J alone (the J engine, unchanged) | 0.560 / 4.861 s | 0.561 / 4.864 s | 4e−14 |

  - The differences are part 8's rounding level: the generated kernels differ from v1.9.0 by 2–8e−12.
  - The two-electron gradient summed over the atoms, zero in exact arithmetic, is 3.1e−12 with this change
    (9.4e−12 before) for B3LYP's factors on def2-SVP.
- **End to end** (B3LYP caffeine def2-SVP, `overview.py`):
  - the gradient takes 3.19 → 2.89 s (−9.4%), within 3.0e−12 Eh/bohr;
  - the SCF is unchanged (24.45 → 24.48 s, 15 cycles both, ΔE 4.5e−13 Eh);
  - so is the Hessian (146.05 → 145.94 s, within 1.1e−11), which does not use these kernels.
- **Per class under ncu** (def2-TZVP; section 5 of the data):
  - 27.92 → 25.24 s of GPU time, and FP64 work 2063 → 1804 GFLOP (−12.5%).
  - Every generated class is faster, by 2.5–14%, most by 7–13%.
  - The smallest gains are in three of the classes that spill most, (ff|fd) 2.5%, (fd|ff) 3.8% and (ff|fp) 4.4%,
    and in (ss|ss), 3.5%, whose time is mostly roots and primitive loops.
- **Unit tests:** test_rhf_grad, test_rks_grad, test_uhf_grad, test_uks_grad, test_tdrks_grad, test_rks_hessian and
  test_uks_hessian pass.
- **Commit:** gpu4pyscf `exp/prim-screen` e5e3551 (`generated_ejk_ip1*.cu`, from `exp_prim_screen.py --ejk-drop j`,
  now the default). node1's tree is that build; `scratch/tree_p35` keeps a copy.

### The Hessian kernels (`--drop-l`): neutral, so off
- **The change:**
  - Per output and pass the kernels form w = dd · I_f, S += w · I_e and four sums T_b += B_b(t_e) · w, one per
    first-derivative factor: 6 FP64 and 7 loads.
  - With T3 = −(T0 + T1 + T2) at each tuple end, that is 5 FP64 and 6 loads per output, and 2 adds per tuple end.
- **Measured** (def2-SVP, two interleaved rounds):
  - B3LYP's factors 22.95 → 22.98 s;
  - short-range K 31.79 → 31.81 s;
  - long-range K 22.44 → 22.48 s;
  - values within 1.0e−11.
- **Why nothing:** the tuple ends dominate.
  - Each costs about 50 FP64: six second-derivative factors and eight mixed-term FMAs.
  - A lane's run averages about 6 outputs per tuple end. In warps with 2 or 4 lanes, every lane also applies its
    sums wherever another lane's tuple changes.
  - So the dropped FMA was about 7% of a pass, and the two adds per tuple end took most of that back.

### What is left from this symmetry
- **Fusing type12 and type3.** Both kernels form the same S and T0–T3 per output; only their tuple-end factors
  differ (type12: ii, jj, ij, kk, ll, kl; type3: ik, jk, il, jl).
  - One kernel would do the per-output passes once instead of twice.
  - Translational invariance would cut its ten second-derivative blocks to six: ii, ik, il, kk, kl, ll, with
    ij = −(ii + ik + il), kj = −(ki + kk + kl), lj = −(li + lk + ll), and jj = −(ji + jk + jl).
  - The cost: type12's g layout (all four centers one higher), and about 63 sums per thread against 42 and 36 now,
    so more spills.
  - The stage is about 23 s of the 146 s B3LYP Hessian, so even 1.7× on it would save about 8 s (5%).
- **The generic (ff|ff) gradient kernel** is 0.7% of the def2-TZVP gradient: not worth it.

## node2: the original against the modified PySCF on a second desktop, 2026-11-03
Asked for by the user ("How does the modified PySCF compare to the original version and the new GPU4PySCF?").
The last like-for-like run of `pyscf-base` against `pyscf` was before It6 (2026-10-27); every later timing compared
one iteration with the previous one. The laptop was on battery and node1 busy, so the user offered node2.
- **Machine:** `ssh node1` then `ssh node2` (`danny-node2`, 10.42.0.222 behind node1).
  - Ryzen 5 4500: Zen 2, 6 cores / 12 threads, AVX2 without AVX-512, up to 3.6 GHz under `schedutil`.
  - 15 GB RAM; Ubuntu 26.04.1, Python 3.14.4.
  - Quadro M2000 (compute capability 5.2). GPU4PySCF needs 7.0 or newer, so it cannot run there.
  - The user installed `build-essential gfortran cmake git python3-venv python3-dev`.
- **Setup** (`~/Chemistry/my-pyscf` on node2; `scripts/node2_run.sh`, `scripts/node2_tzvp12.sh`):
  - Sources are `git archive`s of the laptop's commits: `pyscf` perf 802f7e744, `pyscf-base` 44313dfb4, libcint
    42ec76d, qcint a95651f, libxc f1eb21e8d, OpenBLAS e0166008b. They streamed through node1's SSH in 6 s.
  - The venv has numpy 2.5.4, scipy 1.18.1, h5py 3.16.0, ninja and pytest; libxc's configure step needs pytest.
  - The laptop's build scripts then build OpenBLAS (DYNAMIC_ARCH), libcint-stock, libxc-stock, and qcint with PGO
    trained on node2 (its AVX2 kernels), and the two trees:
    - `pyscf-base` with libcint-stock;
    - `pyscf` with qcint-pgo and `NUMINT_X86_64_V3=ON`, as on the laptop.
  - `bench/kernels.py`: all 72 fixed-input kernel tests pass against `pyscf-base` at rtol 1e−11.
- **Benchmark:** `run_bench.py`, ABBA, 2 reps, 6 threads, `PYSCF_MAX_MEMORY=9000` in both trees.
  - With that setting the original also keeps caffeine/def2-SVP's integrals in memory, so the def2-SVP rows
    compare like with like.
  - The harness flags the runs as not benchmark-ready (a desktop has no AC or platform-profile entries). The two
    runs of each case agree within 1%.
- **Data:** `bench/results/node2/`.

| workload | original | modified | speedup |
|---|---|---|---|
| caffeine/SVP PBE SCF (16 cycles) | 26.3 s | 8.76 s | 3.0× |
| caffeine/SVP B3LYP SCF (15) | 27.6 s | 8.58 s | 3.2× |
| caffeine/SVP r2SCAN SCF (14) | 37.0 s | 13.1 s | 2.8× |
| caffeine⁺/SVP UKS B3LYP SCF | 66.4 s | 17.1 s | 3.9× |
| caffeine/TZVP DF-PBE SCF | 57.6 s | 26.2 s | 2.2× |
| caffeine/TZVP B3LYP SCF (14) | 337.5 s | 190.4 s | 1.77× |
| benzene/SVP B3LYP gradient | 3.84 s | 0.91 s | 4.2× |
| caffeine/SVP B3LYP gradient | 25.9 s | 8.58 s | 3.0× |
| benzene/SVP B3LYP TDA, 5 states | 22.0 s | 2.49 s | 8.8× |
| benzene/SVP B3LYP TDDFT, 5 states | 58.2 s | 6.54 s | 8.9× |
| caffeine/SVP B3LYP TDA, 5 states | 253.2 s | 36.1 s | 7.0× |
| benzene/SVP B3LYP Hessian | 100.3 s | 14.0 s | 7.2× |
| caffeine/SVP B3LYP Hessian | 1254.7 s (1 run) | 189.4 s | 6.6× |

- **The same results:** energies within 3.0e−12 Eh, gradients 1.7e−12, excitation energies 5.6e−13 and Hessians
  1.6e−11, with the same cycle counts.
- The user stopped the second run of the original's caffeine Hessian; the first one is in the table.
- **caffeine/TZVP B3LYP and the semi-direct cache (It9):** the 41 GB of Schwarz-significant integrals do not fit,
  so the modified tree caches what fits and the original computes everything in every cycle.

  | setting | peak RSS | modified SCF | against the original |
  |---|---|---|---|
  | 9000 MB | 6.5 GB (the original 1.6 GB) | 190.4 s (J/K 168.7 s) | 1.77× |
  | 12000 MB | 8.4 GB | 177.9 s (J/K 156.3 s) | 1.90× |

  - The 12000 MB rerun is its own 2 runs. The energy is −680.5697781455 Eh, as on the laptop.
  - The extra ~2 GB of cache saved 6.5%. The cache keeps the quartets that are dearest to recompute first, so the
    next gigabytes hold cheaper ones. On the laptop at 20000 MB, 14.7 GB of cache held about 60% of a cycle's
    integral time.
- **Against GPU4PySCF** (node1's GTX 1650 Super, part 31 overview and part 35; different machines):

  | workload | original PySCF, node2 | modified PySCF, node2 | GPU4PySCF 1.9.0 | modified GPU4PySCF |
  |---|---|---|---|---|
  | caffeine/SVP PBE SCF | 26.3 s | 8.76 s | 25.2 s | 13.2 s |
  | caffeine/SVP B3LYP SCF | 27.6 s | 8.58 s | 57.9 s | 24.4 s |
  | caffeine/SVP r2SCAN SCF | 37.0 s | 13.1 s | 37.1 s | 23.6 s |
  | caffeine/TZVP B3LYP SCF | 337.5 s | 177.9 s (12 GB) | 321.6 s | 185.5 s |
  | caffeine/SVP B3LYP gradient | 25.9 s | 8.58 s | 7.1 s | 2.89 s |
  | caffeine/SVP B3LYP Hessian | 1254.7 s | 189.4 s | 304.2 s | 146 s |

  - The GPU SCFs ran to `conv_tol` 1e−10 (PBE 18 and r2SCAN 16 cycles against 16 and 14 here); B3LYP has 15
    cycles in both.
  - So the modified CPU tree on a 6-core desktop is 1.5–2.8× faster than the modified GPU4PySCF on the 1650 Super
    for def2-SVP SCFs. The two are even at def2-TZVP, and the GPU leads for the gradient (3.0×) and the Hessian
    (1.3×).
  - The 1650 Super runs FP64 at 1/32 rate; an A100 or H100 would lead every row.

## node2: wB97M-V, the reduced VV10 grid, and It24 (the VV10 kernels on the CPU), 2026-11-03
Asked for by the user: a wB97M-V comparison with the reduced, pruned VV10 grid ("Could you add a wB97M-V
comparison? and also include a comparison with the reduced grid size and pruning for the VV10 term"). The user then
noticed I had said neither PySCF tree had VV10 changes ("I believe we worked on it specifically no?"). That was right
for PySCF: all of the VV10 work (parts 26, 28–30, 33) was in GPU4PySCF, and the swarm review's CPU item ("the VV10
pair kernel: ≈ −150 to −250 s per caffeine wB97M-V SCF") was never scheduled. The user asked to port it ("yes, port
the VV10 kernel improvements to the CPU").
- **Harness:** `bench/cases.py` gains `BENCH_NLC_GRID` (a level, or `sg1` for (50, 194) with SG-1 pruning, Q-Chem's
  default NLC grid), `nlc_points` in the SCF records of VV10 functionals, and the cases `scf_caffeine_svp_wb97mv`
  and `grad_caffeine_svp_wb97mv` (in no suite). Scripts: `scripts/node2_wb97mv.sh`, `node2_wb97mv_grad.sh`,
  `node2_vv10.sh`. Data: `bench/results/node2/`.

### It24: `pyscf` `perf` 8845fbde8, the VV10 kernels
- **Before:** `VXC_vv10nlc` evaluated every ordered pair (i, j) with three divisions (`VXC_vv10nlc_grad`: four),
  read the inner points with a stride of 3, and summed each outer point's terms into one variable. GCC does not
  reorder a floating-point sum, so the loop stayed scalar even with the perf tree's `-march=x86-64-v3` for
  `nr_numint.c`.
- **One division per pair:** with D = 1/(g gp gt), T = RpW D, and 1/g + 1/gt = (g + gt) gp D. For the gradient
  kernel, Q = RpW D² (W0_i gp gt + W0p_j g gt + (W0_i + W0p_j) g gp).
- **Vector lanes:** the inner points are copied into padded arrays per coordinate. Padding points have RpW = 0,
  W0 = 0 and K = 1, so they add exact zeros. Each sum runs in 8 partial sums, added in a fixed order. The compiled
  loops use `vdivpd` on ymm.
- **Each pair once** (`VXC_vv10nlc_sym`) where the outer grid is the inner one, as in the SCF's `nr_nlc_vxc` and the
  gradient's `get_nlc_vxc`:
  - T_ij and T_ji share D;
  - blocks of 128 points, with the pairs of blocks visited in the rounds of a round-robin tournament (circle
    method). The threads of a round update disjoint blocks, with a barrier between rounds;
  - no locks or per-thread arrays, and the results are bitwise the same for 1, 3 and 10 threads.
  - `numint._vv10nlc` takes this path when `vvrho is rho and vvcoords is coords`.
- **Exactness:**
  - Fixed inputs (`bench/diag/vv10_check.py`: caffeine/def2-SVP, minao density, level-1 NLC grid): within 6e−14
    relative of the previous kernels, NLC energy within 8e−16.
  - `bench/kernels.py` VERSION 7 adds 5 VV10 entries (NLC energy, potential, gradient term, grid-response energy
    and potential). All 77 pass against `pyscf-base` at rtol 1e−11 on the laptop and on node2, the new ones within
    1.7e−15.
  - PySCF's VV10 unit tests pass: `test_numint` and `test_h2o` VV10/NLC, RKS/UKS finite-difference NLC gradients,
    the response-function test, `test_tdrks_vv10`, `test_vv10_hessian`.
  - The Hessian's VV10 kernels (`VXC_vv10nlc_hessian_eval_*`) are unchanged.
- **Kernel times** (`vv10_check.py`, 97,000 points; laptop 10 threads at the 2 GHz battery cap, node2 6 threads):

  | call | laptop: before → after | node2: before → after |
  |---|---|---|
  | outer grid = inner grid (the SCF's) | 2.04 → 0.63 s (3.3×) | 3.70 → 1.14 s (3.2×) |
  | two grids | 2.03 → 0.95 s (2.1×) | 3.71 → 1.93 s (1.9×) |
  | gradient kernel (20,000 outer points) | 0.49 → 0.24 s (2.1×) | 0.84 → 0.42 s (2.0×) |

  - On node2 the old loop took about 8.5 cycles per ordered pair. The new pair-once loop takes about 2.6, near the
    FP-pipe limit of its ~31 operations and one division per unordered pair.
  - FMA contraction stays off: the whole file is compiled with `-ffp-contract=off`, so that the other kernels stay
    bitwise equal to the default build.

### wB97M-V caffeine/def2-SVP on node2 (6 threads, 9 GB), against GPU4PySCF on node1
| run | VV10 grid points | original PySCF | modified, before It24 | modified, It24 | modified GPU4PySCF (part 33) |
|---|---|---|---|---|---|
| SCF, default NLC grid (level 3) | 294,800 | 816.7 s | 740.2 s | 192.1 s | 297.6 s |
| SCF, SG-1 (50, 194), pruned | 102,384 | 146.3 s | 92.7 s | 46.5 s | 76.0 s |
| gradient, default grid | | (skipped) | 68.0 s | 32.6 s | 25.5 s |
| gradient, SG-1 | | 54.8 s | 23.8 s | 20.7 s | 10.6 s |

- **Same results:**
  - all SCFs converge in 13 cycles (GPU4PySCF 14, at `conv_tol` 1e−10);
  - original and modified agree within 1.1e−12 Eh (gradients 1.6e−12);
  - It24 against the previous modified build agrees within 3.9e−12 Eh (gradients 6e−13).
  - A control rerun of the previous build (`pyscf-it23`) at SG-1 matched its earlier run (92.7 s, energy within
    2.3e−13).
- **The SG-1 grid:**
  - +1.62e−6 Eh in every code, CPU and GPU, and the gradient moves by at most 5.3e−6 (largest component 0.167),
    as on the GPU;
  - CPU and GPU energies differ by 3e−8 Eh at either grid.
  - An approximation, so the user's option (part 33), never a default.
- **Before It24** the default-grid SCF spent about 705–723 s in VV10 in both PySCF trees; It24 takes that to about
  157 s. Within the modified tree, It24 makes the SCF 3.85× faster at the default grid and 2.0× at SG-1. The
  gradients gain less, as J/K derivatives are most of them.
- **Against the 1650 Super:** the modified CPU tree is now faster for the wB97M-V SCF (192 against 298 s;
  SG-1 46.5 against 76.0 s). The GPU still leads the gradients.
- **FMA contraction in the VV10 functions only: tried, then reverted** (`perf` ecc08b80b, reverted by c7ed53964).
  The user asked for it ("Sure give it a try"), then asked for the rollback: "it adds a notable difference from the
  rest of the code for minimal improvement". The branch's VV10 kernels are 8845fbde8's again, with no FMA anywhere
  in libdft, and node2's rebuild is byte-identical to the It24 build. The measurements:
  - `#pragma GCC optimize ("fp-contract=fast")` around them. libdft now has 144 FMA instructions, all in the VV10
    functions; the rest of `nr_numint.c` keeps `-ffp-contract=off`.
  - Results move by 4e−16 relative against It24. The unit tests and `kernels.py` pass.
  - node2: the wB97M-V SCF at the default grid takes 192.1 → 188.9 s (ABBA, 2 reps, energies within 3e−12 Eh), and
    the kernels on the level-1 grid are 1–4% faster. The laptop's pair-once call did not change; its two-grid
    call is 1.16× faster.
  - I had estimated 1.2–1.4×, but FMA removes ~18% of the loop's FP operations for 2% of time. So operations are
    not what bounds the loop; the likely bound is its one division per pair (`vdivpd` on ymm, slow and only
    partly pipelined on Zen 2).
- **One division per two pairs: tried, slower, reverted** (asked for by the user: "Sure try it"):
  - Montgomery's trick: q = 1/(p_a p_b) gives 1/p_a = p_b q and 1/p_b = p_a q, for lanes j..j+3 against
    j+4..j+7.
  - GCC's auto-vectorizer kept two of the four loops scalar ("complicated access pattern", "not profitable"),
    so the version tested wrote the loops with explicit 4-wide vectors (`vector_size(32)`). All four then did one
    `vdivpd` per 8 pairs.
  - Results moved by 3e−16.
  - node2 (`node2_recip2.sh`), against the FMA build:

    | test | FMA build | shared division |
    |---|---|---|
    | the SCF's (pair-once) call | 1.12 s | 1.31 s, 16% slower |
    | two-grid call | 1.95–2.03 s | 1.90 s |
    | gradient kernel | 0.41–0.42 s | 0.40–0.41 s |
    | wB97M-V SCF, default grid (ABBA, 2 reps) | 189.3 s | 212.2 s, 12% slower |

    The laptop was slightly slower in every call.
  - So the division was not the bound either; or, if it was, the pair-once tile's register spills cost more than
    the saved division. That tile spilled to 120 stack accesses: AVX2 has 16 ymm registers for its ~26 live
    vectors.
  - The file went back to ecc08b80b, which was itself reverted afterwards (above).
- **Not done:**
  - the Hessian's VV10 kernels;
  - the NLC term's own pass over its grid (AO values, density, potential matrix). At the default level its points
    are the main grid's, which `nr_rks` has already evaluated. By difference it is about 1.5 s of each SCF cycle
    (157 s of "other" against about 10.5 s per kernel call).

## Next iterations (by measured remaining cost, after It9)
The swarm review above re-ranks these after It13. Its proposed order is the current plan, and the items below are
background.

1. **Integral evaluation is still most of J/K**, even after It8's 1.2-1.3x.
   - Where it is: direct SCF on large molecules, gradients and Hessians.
   - In qcint (see It8):
     - the per-quartet fixed cost (~80 ns), which dominates quartets with few primitive combinations;
     - the scalar Rys roots for 2-3 roots (~10%; evaluating only the x intervals present in a batch might halve them);
     - batching across quartets of one class, a deep change.
   - Gradients: the s2kl loop evaluates 4 `int2e_ip1` quartets per s8 orbit, where 3 would do with translational
     invariance (-25%). That needs a driver that contracts the gradient inside the integral loop.
   - SMT for the direct J/K only (13% in It3) is a configuration choice.
   - Direct SCF (It9): semi-direct removes most of it where the Schwarz-significant quartets fit. What remains:
     - the batched kernels' per-quartet overhead, now the cost of each cycle that reads the cache (kernels
       specialized by block shape);
     - for TZVP-sized molecules, the quartets that do not fit (98 GB for the TZVP alkane).
2. **Remaining XC costs.**
   - The XC response is now the largest part of the caffeine TDA (26 of 35 s).
   - Screened-path `nr_rks`: the zero fill is gone (It10 A), and AO evaluation is fused into the gather kernels
     (It11).
     - What is left of the XC time is the GEMMs and the gather/scatter of dm and XC-matrix blocks.
   - Gradient and Hessian XC terms: done in It12. The RKS CPHF f_xc response from the factors: done in It13.
   - The same factoring for TDA/TDDFT: P = C_vir and Q = C_occ in `rks_fxc_mo`, with nset the number of
     states. The XC response is 26 of the 35 s of the caffeine TDA.
     - For UKS CPHF (`_gen_uhf_response`, `hessian.uhf.gen_vind`), use one A, B, Q per spin.
     - Other `gen_response(hermi=1)` users with orbital-shaped dms (polarizabilities, the TDDFT gradient's
       Z-vector) could call `vind_mo` too.
   - The fused dm path still allocates and sums nthreads·nset private [nao,nao] matrices per grid block (about
     6 s of the alkane's 44 s for 186 dms after It13 A). Accumulating across blocks, or splitting the work by
     AO rows, would remove that.
   - The gradient XC temporaries in `_make_dR_dao_w`.
3. **Extend It5b batching** to `nr_uks_fxc` (UKS TDDFT/CPHF) and the meta-GGA τ term.
4. **Hessian `_get_vxc_deriv1`** (h1ao for the CPHF right-hand side): batch it like It5b.
5. **AO evaluation on the dense path** (`eval_gto`, 5–10%). It is bound by writing the block, not by computing it
   (AO-cache test). Caching AO values helps only after fusion; then cache the radial factors, not the AO values.
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
