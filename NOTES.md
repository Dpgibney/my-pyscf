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

## node2: PySCF's seminumerical exchange (SGX, COSX-like) against exact J/K, 2026-11-04
Asked for by the user: "Could you do some CPU timings with it on node2 to see how it stacks up compared to our
modifications?". This followed the discussion of RI-COSX, where I noted that PySCF already has `pyscf.sgx` and
GPU4PySCF has nothing like it.
- **The two modes of `sgx_fit`:**
  - the default (`pjs=False`): SGX for J and K, no screening;
  - `pjs=True`: density-fitted J and SGX K with P-junction screening, the RIJCOSX-like setup.
  - Both use SGX's defaults: grid level 2, ORCA-like optimized grids, overlap fitting, and a full rebuild every 5
    steps.
- **Harness:** `bench/cases.py` gains `BENCH_SGX` (`sgx` or `pjs`), `sgx_points` in the SCF records, and the case
  `scf_c20h42_tzvp_b3lyp`. Script `scripts/node2_sgx.sh`; data `bench/results/node2/*node2-sgx*`.
- **Setup:** node2, 6 threads, `PYSCF_MAX_MEMORY=12000`, one run each. SGX was run in both trees; its code is the
  same in both (the perf branch does not touch `pyscf/sgx`).

| B3LYP SCF | exact, modified | exact, original | SGX pjs (modified / original) | SGX default (modified / original) |
|---|---|---|---|---|
| caffeine def2-SVP (246 AOs) | 8.7 s | 27.5 s | 66.4 / 92.7 s | 67.8 / 94.2 s |
| caffeine def2-TZVP (494 AOs) | 176.2 s | 337.6 s | 194.0 / 253.1 s | 200.6 / 260.8 s |
| n-C20H42 def2-SVP (490 AOs) | 31.1 s | 131.2 s | 202.7 / 270.5 s | 246.3 / 324.2 s |
| n-C20H42 def2-TZVP (872 AOs) | 422.0 s | 629.2 s | 550.7 / 699.7 s | 630.0 / 792.4 s |

- **SGX energy errors** against exact, identical in both trees:

  | B3LYP | pjs | default |
  |---|---|---|
  | caffeine def2-SVP | −2.1e−4 Eh | −2.8e−5 Eh |
  | caffeine def2-TZVP | −2.7e−4 Eh | −4.7e−5 Eh |
  | n-C20H42 def2-SVP | −4.6e−4 Eh | −1.0e−3 Eh (10 cycles against 9) |
  | n-C20H42 def2-TZVP | −6.8e−4 Eh | +2.6e−4 Eh |

  The SGX grids have 71,000 points for caffeine and 177,000 for the alkane.
- **The modified exact tree is the fastest everywhere,** from 7.6× faster than the best SGX at caffeine/SVP to
  1.3× at C20H42/TZVP.
  - The original exact code beats SGX in three of the four cases; at caffeine/TZVP, SGX pjs is 1.3× faster than
    it.
  - The gap shrinks with basis and size, so SGX would win only beyond these systems.
- **Where SGX's time goes:** 75–85% is in `sgx_jk.py:k_part`, which recomputes the integrals of every AO pair at
  every grid point each cycle (C20H42/TZVP: 419 of 551 s, about 35 s per K build). The exact tree screens its
  integrals, and the modified tree caches them across cycles (It9).
- **The modified tree makes SGX itself 1.26–1.4× faster,** through its XC, AO-evaluation and DF-J changes. `k_part`
  takes about the same time in both trees.
- **What follows for the RI-COSX discussion:** PySCF's SGX is no shortcut at these sizes on the CPU. A faster
  seminumerical K would have to be new code. The better place for one is GPU4PySCF, which already has the grid,
  AO and `int3c1e` pieces.
  - It25 (next section) made SGX 1.6–2.0× faster without approximations. It then beats this exact tree at
    def2-TZVP, but not at def2-SVP.

## It25: PySCF's SGX on the CPU, a deep investigation (node2 and the laptop), 2026-11-04
Asked for by the user after the SGX timings above: "Could you do a deep investigation into performance improvements
possible for the CPU SGX?" Exact only, as always: the SGX results must stay within ~1e−9 Eh of the unmodified SGX,
with the same cycles.
- **Tools** (workspace):
  - `bench/diag/sgx_profile.py`: an SGX SCF, then timed K builds, split into the Python steps and, with an
    `SGX_PROFILE` build of libcvhf, the C driver's phases;
  - `bench/qcint/gbench.c`: int1e_grids of a library on all shell pairs and the SGX batches of a grid
    (`dump_sgx_grid.py`), by class: time, the scalar root step alone, the cost per call, the distribution of x, and
    `--compare` (bitwise fraction and max |diff|, by class);
  - `bench/qcint/kbench.c`: the SGX K contraction alone;
  - `bench/qcint/rchunk_check.c`: `CINTrys_roots_chunk` against `CINTrys_roots`, by interval of x;
  - `bench/qcint/callcost.c`: the per-call setup of `CINT1e_grids_loop`, piece by piece;
  - `bench/qcint/root1_check.c`: an erf-form one-root prototype (not used: rys_root1's own fits vectorize as well,
    with the same values).
- **Scripts:** `scripts/node2_sgxprof.sh`, `node2_gbench.sh`, `node2_sgx2.sh` (step 1), `node2_sgx3.sh` (step 2),
  `node2_sgx4.sh` (the shared chunk function, the block size alone, the ERIs, C20H42/TZVP), `node2_sgx5.sh` (the
  committed qcint). Data: `bench/results/node2/`.

### Where the K build's time goes (node2, 6 threads, pjs, converged density, before)
| case | K build | int1e_grids | contraction | rest of the C driver | Python steps |
|---|---|---|---|---|---|
| caffeine def2-SVP (1.87e9 integrals) | 3.41 s | 88.8% | 9.1% | 2.0% | 7% of the wall |
| caffeine def2-TZVP (7.56e9) | 10.0 s | 83.5% | 14.8% | 1.7% | 6% |
| n-C20H42 def2-SVP (8.86e9) | 16.9 s | 88.4% | 9.2% | 2.4% | 8% |

- The C driver keeps 99% of 6 threads busy. Shell-pair and grid screening are under 1.2% together.
- **Inside int1e_grids** (gbench, caffeine def2-SVP, single thread, qcint-pgo): 8.5 ns per integral.
  - The Rys roots took 53%. `CINTg0_1e_grids` called `CINTrys_roots` once per primitive pair and grid point.
  - 68–85% of those calls (by class) are at x ≥ 35 + 5 nroots, where the roots are a closed form:
    u = rt/(x − rt), w = ww sqrt(π/4x).
  - (ss) and (ps) are 60% of the time; one root each.
- **The contraction** (`nrs2_ijg_gj_gi`, 2 FMAs per integral) ran at 0.9–1.1 ns per integral. libcvhf is built
  with PySCF's default `-O2 -msse3`, at which GCC leaves these loops scalar.

### Changes
1. **qcint `sgx-exp` 40cbc68: the Rys roots of a SIMD chunk at once** (`CINTrys_roots_chunk` in `rys_roots.c`;
   AVX and AVX-512 builds, SSE builds unchanged).
   - Large x (all lanes at x ≥ 35 + 5 nroots): the closed form, for any number of roots.
   - One or two roots at any x > 3e−7: rys_root1's and rys_root2's own fits. The fit of each interval of x is
     evaluated on the whole chunk when a lane is in it, and selected per lane by blends.
   - exp(−x) of the lanes is one loop, which GCC vectorizes with glibc's libmvec under qcint's `-ffast-math`. The
     stock build already does this for 9–12 other exp loops.
   - Other chunks go point by point as before: three or more roots below the large-x limit, x ≤ 3e−7, and
     partial chunks.
   - The one-root case is a `static inline` function in the new `rys_roots_chunk.h`, so that int1e_grids inlines
     it.
   - `_CINTrys_roots_batch` calls it. That is the root step of qcint's ERIs, nuclear attraction, 3c1e and F12
     integrals.
2. **qcint 3659e67: `CINTg0_1e_grids` takes the roots a chunk of grid points at a time:**
   - one root: `CINTrys_root1_chunk` inlined, forming t² = u/(u+1) directly (F1/w, 0.5/x);
   - all lanes large: t² = rt/x = (4rt/π)·(π/4x) from the square of the weight's sqrt(π/4x), instead of u/(u+1)
     of u = rt/(x − rt), two divisions fewer per root;
   - two roots: `CINTrys_roots_chunk`.
   - The first version called `CINTrys_roots_chunk` for one root too (`sx-uni`, the u form plus a division). It was
     3–7% slower on node2; the inlined t² form is bitwise equal to the earlier `sx-vexp`.
3. **qcint 99158a3: `GRID_BLKSIZE` 104 → 112**, so that a 224-point SGX batch is two blocks instead of three
   (bitwise: the chunks are the same). On node2 it is 3–4% faster than 104 with the same code (`sx-uni` against
   `sx-uni104`): 3.61 vs 3.76, 3.06 vs 3.14 and 2.10 vs 2.19 ns per integral.
4. **PySCF `sgx-exp`: `nr_sgx_direct.c` with the `NUMINT_X86_64_V3` kernel flags** (`-O3 -march=x86-64-v3
   -ffp-contract=off`, as `dft/nr_numint*.c` and the GTO grid files). The J/K contractions vectorize over the grid
   points; each element's sums keep their order and nothing is contracted into FMA, so the results are bitwise
   the same. kbench, single thread: 0.75–0.98 → 0.29 ns per integral (laptop).
   - This is a per-file flag difference, which the user preferred to avoid for small gains (the FMA rollback).
     It adds a file to the existing `NUMINT_X86_64_V3` set rather than a new kind of difference, and its results
     are bitwise those of the default flags. For the user to keep or drop.
5. **PySCF: the default SGX mode computes the overlap-fitting projection once per grid.** `get_jk_favorj`
   evaluated the AOs on the whole grid a second time per J/K build and solved for a projection that depends on
   the grids and the basis only. It is now kept in `sgx._overlap_correction_matrix`, which `build()` and `reset()`
   already cleared but nothing used, for the grids it was computed on. Bitwise the same.

### int1e_grids on node2, step by step (gbench, single thread, AVX2, PGO builds; ns per integral)
| qcint | caffeine def2-SVP | caffeine def2-TZVP | n-C20H42 def2-SVP |
|---|---|---|---|
| qcint-pgo (before) | 8.48 | 5.94 | 5.16 |
| + large-x chunks (`sgxpgo`) | 5.65 | 4.22 | 3.08 |
| + one-root chunks, `GRID_BLKSIZE` 112 (`sx-sameops`) | 4.64 | 3.74 | 2.55 |
| + t² without u (`sx-fewdiv`) | 4.29 | 3.55 | 2.32 |
| + two-root chunks (`sx-root2`) | 3.84 | 3.21 | 2.23 |
| + vector exp (`sx-vexp`) | 3.37 | 2.96 | 1.97 |
| **the committed qcint 3659e67** (`sx-fin`) | **3.37** | **2.96** | **1.97** |
| total | 2.5× | 2.0× | 2.6× |

- Two runs each, interleaved; they agree to 0.5%.
- `sx-fin` (`node2_sgx5.sh`) gives the integrals of `sx-vexp` bitwise on all three, and its ERIs are bitwise those
  of `sx-uni` (the ERI rows below).
- Against qcint-pgo, the largest difference grows from 1.1e−16 (step 1) to 5.6e−16 (two-root chunks) and
  1.8–3.6e−15 with the vector exp: at most two ulps of each case's largest integral (7.7, 12.3, 5.7), in (ss),
  where 99.5–99.7% stay bitwise equal.
- After the changes (caffeine def2-SVP, `sx-vexp`): (ss) 9.7 ns per integral, (ps) 4.0, (pp) 3.2. The cost per call
  is now 8–15% of the time (C20H42: 15%), the setup an optimizer would hold.
- **The contraction alone** (kbench, node2, single thread): 0.91 → 0.38 ns per integral (2.4×) with the kernel
  flags, bitwise equal. Tiling it over grid points is slower with both flag sets.

### The K build on node2 after the changes (6 threads, pjs, converged density; `pyscf-sgxprof` on `sx-vexp`)
| case | K build | int1e_grids (thread-s) | contraction (thread-s) | Python steps | integrals per thread-s |
|---|---|---|---|---|---|
| caffeine def2-SVP | 3.41 → 1.59 s (2.15×) | 16.7 → 6.7 | 1.71 → 0.86 | 0.24 s, 15% | 112 → 280 M |
| caffeine def2-TZVP | 10.0 → 5.51 s (1.81×) | 46.6 → 23.5 | 8.46 → 4.68 | 0.57 s, 10% | 162 → 322 M |
| n-C20H42 def2-SVP | 16.9 → 7.61 s (2.22×) | 81.1 → 30.0 | 8.39 → 4.00 | 1.44 s, 19% | 109 → 296 M |

- In the C driver, int1e_grids is now 80–84% and the contraction 11–16%.
- **The contraction gains 1.8–2.1× in the driver,** against 2.4× in kbench (one thread, caffeine-SVP shells).
  The cause of the difference was not measured.
- **The Python steps did not change:** `_sgxdot_ao_dm_sparse`, `_sgxdot_ao_gv_sparse` (in `nr_sgx_direct.c`, but
  BLAS-bound), `eval_gto` and `get_g_threshold`. They are now a tenth to a fifth of the K build.
- The profile runs' SCFs (one each): 67.3 → 37.2 s, 195.4 → 125.9 s, 204.9 → 107.4 s, with the same energies to
  the 10 decimals printed and the same cycles. The A/B runs below are the reference.

### SGX SCFs on node2: the modified tree (qcint-pgo) against `pyscf-sgx` (`sx-vexp`, changes 1–5)
6 threads, `PYSCF_MAX_MEMORY=12000`, two runs each, interleaved (ABBA); `bench/results/node2/*node2-sgx3*`.
`sx-vexp`'s int1e_grids are bitwise those of the final qcint. The final one also vectorizes the batch roots of
the 2e code, which pjs's DF-J (`int3c2e`) uses; that part is not in these timings.

| B3LYP SCF | before | after | speedup | energies (after − before) | cycles |
|---|---|---|---|---|---|
| pjs, caffeine def2-SVP | 66.5 s | 36.4 s | 1.83× | ≤ 1e−12 Eh | 15 |
| pjs, caffeine def2-TZVP | 193.6 s | 124.1 s | 1.56× | ≤ 2e−12 | 14 |
| pjs, n-C20H42 def2-SVP | 202.6 s | 105.9 s | 1.91× | ≤ 4e−12 | 9 |
| default mode, caffeine def2-SVP | 67.8 s | 34.2 s | 1.98× | 0 | 15 |

- Step 1 alone (large-x chunks, `sgxpgo`; `*node2-sgx2*`): 66.4 → 51.7 s, 193.8 → 160.0 s, 203.5 → 145.0 s.
- The runs' cost categories put the saving in "other", where the C K driver is counted. jk (the DF-J) and XC
  are unchanged.

### Against exact J/K and the original PySCF (node2, 6 threads, 12 GB; exact and original from the SGX section above)
| B3LYP SCF | exact, modified tree | SGX pjs, original PySCF | SGX pjs, modified tree | SGX pjs, now |
|---|---|---|---|---|
| caffeine def2-SVP | 8.7 s | 92.7 s | 66.5 s | 36.4 s |
| caffeine def2-TZVP | 176.2 s | 253.1 s | 193.6 s | **124.1 s** |
| n-C20H42 def2-SVP | 31.1 s | 270.5 s | 202.6 s | 105.9 s |
| n-C20H42 def2-TZVP | 422.0 s | 699.7 s | 552.8 s | **321.5 s** |

- SGX is now 2.0–2.6× faster than the original PySCF's.
- **It now beats the modified tree's exact SCF at def2-TZVP:** 1.42× at caffeine, 1.31× at C20H42. Before it was
  10% and 30% slower.
- At def2-SVP the exact SCF stays 3.4–4.2× faster.
- The C20H42/TZVP pair is one run each (`*node2-sgx4-c20tz*`): 1.72× faster, energies within 4.1e−12, 10 cycles
  both.
- The SGX energy errors against exact are unchanged (see the SGX section above): −2.1e−4 to −6.8e−4 Eh in pjs mode.

### qcint's ERIs, a side effect (node2, PGO builds, single thread; qbench)
`_CINTrys_roots_batch` now takes `CINTrys_roots_chunk` (change 1), so the 2e integrals' roots are vectorized too.

| integrals | before | after | saving | max \|diff\| (largest value) |
|---|---|---|---|---|
| int2e, caffeine def2-SVP (14.6 M quartets) | 9.03 s | 8.52 s | 5.6% | 1.1e−16 (4.8) |
| int2e, caffeine def2-TZVP (88.6 M) | 70.6 s | 68.8 s | 2.5% | 1.7e−16 (8.0) |
| int2e_ip1, benzene def2-SVP | 4.67 s | 4.45 s | 4.8% | 1.1e−16 (3.3) |
| int2e_ip1, caffeine def2-SVP | 57.7 s | 55.6 s | 3.7% | 2.2e−16 (6.5) |

- The same nonzero flags everywhere.
- The committed qcint (`sx-fin`) repeats the first row: int2e on caffeine 9.02 → 8.51 s, bitwise `sx-uni`'s.
- Laptop (AVX2): int2e 5.45 → 5.02 s, benzene ip1 2.94 → 2.77 s.
- End-to-end exact SCFs and gradients with this qcint were not timed.

### Exactness
- **int1e_grids** (gbench `--compare`, all shell pairs on the SGX batches):
  - against the previous build, at most two ulps of the largest integral of a case;
  - node2 (PGO): 1.8e−15 at 7.7 (caffeine def2-SVP), 3.6e−15 at 12.3 (def2-TZVP), 1.8e−15 at 5.7 (C20H42), in (ss),
    where 99.5–99.7% stay bitwise equal;
  - the laptop's AVX2 build: 8.9e−16 at 6.7, 68% bitwise equal.
  - **For scale,** the previous code differs from itself by 8.9e−16 between its AVX2 and AVX-512 builds (82%
    bitwise equal), and by 1.8e−15 between its `-msse3` and native builds.
  - Under qcint's `-ffast-math`, any edit of `CINTg0_1e_grids` changes how GCC reassociates the recursion that
    follows the roots. Moving the unchanged root loop into a function of its own changed the bits of every class
    (max 8.9e−16), so bitwise equality for two or more roots is not available to any change of that function.
  - The roots themselves (`bench/qcint/rchunk_check.c`: `CINTrys_roots_chunk` against `CINTrys_roots`, 400,000
    chunks over x in [0, 60), AVX2 and AVX-512): all roots and weights within 8.2e−16 relative, about 4 ulps.
    One root: x ≥ 15 bitwise, below that up to 6.7e−16 from the vector exp. Two roots: the roots bitwise except
    (10, 15], the weights up to 8.2e−16 from the order in which their formula is evaluated; x > 33 bitwise.
- **ERIs** with the vectorized batch roots (qbench `--compare`, against the previous build), with the same
  nonzero flags:
  - caffeine def2-SVP, 14.6 M quartets: max |diff| 1.1e−16 at |value| 4.8;
  - def2-TZVP, 88.6 M quartets: 1.7e−16 at 8.0.
- **PySCF's checks** on the laptop with the new qcint swapped in (`LD_LIBRARY_PATH`; the `uni` build, whose ERIs
  are bitwise those of the final one):
  - `kernels.py`: ALL PASS (rtol 1e−11);
  - `validate.py`: ALL PASS against pyscf-base. SCF energies within 2.5e−12, the benzene gradient within 2.8e−14,
    TDA excitations within 4.3e−15. Its one warning, r2SCAN in 24 cycles against pyscf-base's 25, is the
  same with the tree's own qcint.
- **PySCF's unit tests** on the laptop's `pyscf-sgx` (all of these changes, linked to the `uni` qcint):
  `sgx/test`, `sgx/grad/test`, `qmmm/test` (int1e_grids on MM charges) and `lib/gto/test/test_gridao.py`: 42
  passed, 2 failed. The two are `test_dm_screening`'s translational-invariance check, 7.08e−12 against 5e−12: qcint's
  known one (It6's upstream tests), the same 7.08e−12 with the `perf` tree on qcint-stock or qcint-pgo, and
  passing on pyscf-base.
- **Changes 4 and 5 are bitwise:** the contraction gives the same bits with both flag sets (kbench), and the
  default-mode SCFs with the cached projection give the same energies to the last digit printed
  (benzene −232.084241761618, caffeine −679.812776162158, same cycles).
- **The SGX SCFs** (node2, the A/B above): pjs energies within 3e−12 Eh of the old tree's, inside its own
  rep-to-rep spread (up to 2e−12 from the order of OpenMP's accumulation); default mode identical in every digit
  printed (−679.8127761621591 in all four runs); the same cycles everywhere.

### Looked at, not done
- **An optimizer for plain int1e_grids** (`int1e_grids_optimizer` returns NULL; the derivative kernels have one).
  - Each call redoes the setup an optimizer could hold, about 200 ns on caffeine (timed with dlsym'd pieces): the
    coefficients' logs 60 ns, the pair data (an exp per primitive pair) 105 ns, the nonzero-coefficient tables
    45 ns. That is 3–5% of an improved K build.
  - Not done: the pair data and the g-index tables depend on `env[PTR_RANGE_OMEGA]` (the screening estimate; ω < 0
    doubles the roots). An optimizer built under one ω and used under another would give wrong integrals, and
    PySCF's range-separated SGX path creates objects inside `with_range_coulomb`. The NULL optimizer cannot fail
    that way.
- **Three- and four-root fits on chunks:** at most 2% of int1e_grids on caffeine def2-TZVP. The bound: forcing
  those chunks through the large-x form (wrong values) saved 0.033 of 1.75 s.
- **Mixed chunks** (large-x lanes as a vector, only the others point by point): no measurable change.
- **(ss) without the gx = gy = 1 arrays:** bitwise the same, 4% of (ss), about 1% overall.
- **An erf form for one root** (`root1_check.c`): no better than vectorizing rys_root1's fits, which keep its
  values.
- **Tiling the contraction** over grid points (16–56 at a time): slower once the loop is vectorized.
- **Caching the integrals across cycles,** as It9 does for ERIs. By an estimate (not measured), reading an
  integral back costs about half of computing it now: about 1.6 ns per 8 bytes per thread at ~30 GB/s shared by 6
  threads, against 3.6 ns to compute it (280 M per thread-second). A cache would gain at most about 2× on what it
  holds. Caffeine/def2-SVP needs 15 GB
  per K build (1.9e9 integrals), C20H42 71 GB, and pjs's density screening changes the set from cycle to cycle.
- **PGO:** the qcint PGO training (`train_qcint_pgo.sh`) has no int1e_grids, so these paths are built without a
  profile, as before.

## It26: SGX re-explored for symmetries and repeated work (node2 and the laptop), 2026-11-04
Asked for by the user after It25: "Sure let's keep that flag change for now, could you reexplore the sgx code to see
if there are optimizations or symmetries that could be exploited for further speed improvements?" Exact only, as in
It25.

### Symmetries
SGX's K is K_μν = Σ_g X_μg Σ_λ A_νλ(g) F_λg, with A_νλ(g) the potential of the AO pair at grid point g.
- **Already used by PySCF:**
  - A_νλ(g) = A_λν(g): s2 shell pairs, one integral feeding both contractions (`nrs2_ijg_gj_gi`);
  - in default mode, J and K from the same integrals;
  - in the SGX gradient, translational invariance: only the grid derivative of A is computed, and the AO-center
    part is −∂A/∂g;
  - density and AO screening per batch.
- **One-center pairs** (both shells on one atom, so every primitive pair has the center P = A): far from the
  atom, the pair's potential is a finite multipole expansion, and the sum over primitive pairs could collapse
  into one evaluation per point.
  - Measured (gbench `--pairs same|diff`, caffeine def2-SVP, unscreened): one-center pairs are 14% of int1e_grids
    time. Their (ss) costs 21 ns per integral (5.8 primitive pairs, 91% at large x), two-center (ss) 6.8 ns.
  - Not done: all primitive pairs are at large x only beyond ~2.4 bohr for carbon's 1s core and ~6.4 bohr for
    2p×2p, and valence shells have one primitive. The upper bound is ~3–5% of int1e_grids.
- **Point-group symmetry:** would need symmetry-adapted grids, and most molecules have none.
- **Two-center primitive pairs** each have their own center P_p, so nothing collapses. One division and one
  square root per primitive pair and SIMD chunk (for 1/|g − P_p|) set a floor of about 1.5 ns per primitive pair
  and point on Zen 2. Two-center (ss) runs at 2.3.

### Repeated work found and removed
1. **qcint `sgx-exp` 1f1b739: the grid coordinates transposed by shuffles instead of gathers.**
   - `CINT1e_grids_loop` copies the coordinates of each block of points into x, y, z rows on every call. With
     three AVX2 gathers per SIMD chunk, that is about 170 gathers per 224-point call, and SGX makes one call per
     shell pair and batch, all with the same points.
   - Gathers are microcoded and slow on some CPUs, Zen 2 among them. Four points now take three loads, three
     `vperm2f128` and three `vshufpd`.
   - Bitwise the same integrals.
2. **qcint 7bb9b92: `int1e_grids_optimizer`, the per-call setup precomputed.** It returned NULL, so every call
   recomputed for its two shells:
   - the coefficients' logs;
   - the pair data (an exp per primitive pair);
   - the nonzero-coefficient tables and the g indices.
   That is about 200 ns per call (`bench/qcint/callcost.c`).
   - The optimizer now holds them: `CINTall_1e_grids_optimizer`, plus pair data for every ordered pair of shells,
     made by the same `CINTset_pairdata` call as the loop's.
   - **Bitwise:** the integrals equal the per-call path's in AVX2 and SSE3 builds (`bench/qcint/optcheck.c`,
     caffeine and C20H42, all ordered pairs). The AVX-512 and node2's PGO builds are within 1.4e−16: GCC compiles
     the two copies of the squared distance differently under `-ffast-math`.
   - **Layout:** the pair data are laid out as `CINTOpt_setij`'s, (i, j) next to (j, i). Row by row
     (`sx-opt2`) was 1.5–3% slower on node2 with the same values.
   - A first version took the pair data from `CINTOpt_setij`. Its (i < j) pairs are transposed copies whose
     centers are formed from the other end, which differed by up to 2.0e−15 on C20H42.
   - **The ω guard:** the pair data and g indices depend on `env[PTR_RANGE_OMEGA]` (the screening estimate; ω < 0
     doubles the roots). The optimizer is made only for ω = 0 and used only when the call's ω is 0 and it has
     pair data. With ω = ±0.3 set after it was made, the integrals are bitwise the per-call ones.
   - PySCF's range-separated SGX builds its objects inside `with_range_coulomb`, so it gets none. The derivative
     kernels' optimizers have no pair data, so they are never mistaken for this one.
   - SGX gets it with no PySCF change, through `make_cintopt(..., 'int1e_grids')`.
3. **PySCF `sgx-exp` 64b526734: the AO values on the grid kept across K builds.**
   - With enough memory, `get_loop_data` makes the whole grid one block, and every K build evaluated the same AO
     values on it.
   - They are now kept in `_pjs_data` (pjs), which is remade when the grids or molecule change, and on the SGX
     object for its grids (default mode).
   - They are dropped as soon as the grid takes more than one block.
   - Only read, so bitwise the same.
   - Memory: nao × ngrids doubles, 0.14 GB (caffeine/SVP) to 1.2 GB (C20H42/TZVP).

### node2 (6 threads; PGO builds of qcint)
**int1e_grids** (gbench, single thread, ns per integral, two runs each):

| qcint | caffeine def2-SVP | caffeine def2-TZVP | n-C20H42 def2-SVP |
|---|---|---|---|
| qcint-pgo (before It25) | 8.47 | 5.94 | 5.15 |
| `sx-fin` (It25, 3659e67) | 3.39 | 2.96 | 1.98 |
| + transpose (`sx-tr`, 1f1b739) | 3.10 | 2.77 | 1.76 |
| + optimizer, pair data from `CINTOpt_setij` (`sx-opt`) | 2.96 | 2.67 | 1.63 |
| + optimizer, pair data row by row (`sx-opt2`) | 3.04 | 2.71 | 1.68 |
| **+ optimizer, final** (`sx-opt3`, 7bb9b92) | **2.96** | **2.67** | **1.65** |
| total, against qcint-pgo | 2.86× | 2.22× | 3.12× |

- `sx-tr` is bitwise `sx-fin`. `sx-opt3` is bitwise `sx-opt` on SGX's pairs (i ≥ j); the SCF A/B below ran
  `sx-opt`.
- The cost per call drops from 8–15% of the time to 6–11%.

**The K build** (pjs, converged density; `pyscf-sgxprof26` on `sx-opt` with the AO cache, against It25's):

| case | K build | C driver | int1e_grids (thread-s) | eval_gto |
|---|---|---|---|---|
| caffeine def2-SVP | 1.59 → 1.39 s | 1.35 → 1.22 s | 6.67 → 5.92 | 0.072 s → kept |
| caffeine def2-TZVP | 5.51 → 5.06 s | 4.94 → 4.61 s | 23.5 → 21.4 | 0.125 s → kept |
| n-C20H42 def2-SVP | 7.61 → 6.53 s | 6.18 → 5.44 s | 30.0 → 25.6 | 0.334 s → kept |

- Default mode, caffeine/SVP: 1.31 s per J/K build (1.15 s C driver).

**SGX SCFs** (12 GB, ABBA, two runs each; `pyscf-sgx` = It25 against `pyscf-sgx26` = changes 1–3 on `sx-opt`;
`bench/results/node2/*node2-sgx6*`):

| B3LYP SCF | It25 | It26 | | before It25 | original PySCF | exact, modified tree |
|---|---|---|---|---|---|---|
| pjs, caffeine def2-SVP | 36.4 s | 32.8 s | 1.11× | 66.5 s | 92.7 s | 8.7 s |
| pjs, caffeine def2-TZVP | 124.3 s | 115.9 s | 1.07× | 193.6 s | 253.1 s | 176.2 s |
| pjs, n-C20H42 def2-SVP | 105.9 s | 94.1 s | 1.13× | 202.6 s | 270.5 s | 31.1 s |
| default mode, caffeine def2-SVP | 34.3 s | 30.8 s | 1.11× | 67.8 s | 94.2 s | 8.7 s |

- **Energies:** pjs within 2.7e−12 Eh of It25's, inside its rep-to-rep spread; the same cycles.
- **Default mode:** within 2.3e−12. It is not bitwise reproducible either: It25's tree gave −679.8127761621591
  in the It25 runs and −…1589 now.
- **Since before It25:** 2.0–2.2× (pjs 1.67× at caffeine/TZVP). Against the original PySCF, 2.2–3.1×.
- **Against exact:** the exact SCF (It9's cached ERIs) stays 3.0–3.8× faster at def2-SVP. SGX is 1.52× faster at
  caffeine/def2-TZVP.

### Tried, no gain
- **`_dot_ao_dm_l1` (fg = D χ) with the AOs j of a shell blocked, 8 at a time,** so that each AO row is read once
  per block instead of once per j. Bitwise the same (the sums keep their order), but no faster on the laptop
  (0.057–0.067 s against 0.060 s for caffeine def2-SVP's whole grid, `bench/diag/sgxdot_check.py`); reverted. The
  56-point accumulator already amortizes the strided density loads, and the screening keeps the loops short.

### Left
- **A semi-direct SGX: reuse the integrals across cycles.** They do not depend on the density.
  - Per integral, (ss) and (ps) blocks cost 4–10 ns to compute against about 2 ns to read back. That read cost is
    an estimate: 8 bytes at ~25 GB/s shared by 6 threads.
  - The other classes cost 1.6–3.3 ns, so a cache should hold the expensive blocks first.
  - Caffeine/SVP's (ss)+(ps) blocks are 4.3 GB per K build and 45% of int1e_grids time, about −20% of a K build if
    cached.
  - TZVP and C20H42 need 10–27 GB for the same set; with 4 GB, about −10%.
  - pjs's density screening changes the needed set from cycle to cycle, so the cache must allow misses.
  - Memory against time, for the user to decide.
- **The grid transpose once per batch instead of once per call** (an estimated ~120 ns per call now) would need
  an int1e_grids interface that takes the transposed grids.
- **The one-center collapse** above: ≤ 3–5% of int1e_grids.
- **SMT:** 12 threads on node2's 6 cores do not help. The caffeine/SVP C driver takes the same 1.22 s with 2×
  the thread-seconds, and the SCF is slower (41.5 s against 33.5 s).

## node2: B3LYP/PCM memory at ~1140 basis functions, 2026-11-04
Asked for by the user: "On Node2 could you create a molecule with ~1140 basis functions and run B3LYP with PCM using
the new and old code? I believe it may run out of memory and want to diagnose where that may be coming from."
- **Setup:**
  - n-C26H54/def2-TZVP: 80 atoms, 1130 AOs, and 9,998 PCM surface points (C-PCM, water, Lebedev 302 per atom),
    so one surface matrix is 0.80 GB.
  - node2: 15 GB RAM, 6 threads, PySCF's default `max_memory` (4000 MB).
  - `bench/diag/pcm_memory.py` samples the RSS every 0.2 s by phase, and its watchdog stops a run at 13 GB.
  - Scripts `scripts/node2_pcm.sh` and `node2_pcmfix.sh`; logs `bench/results/node2/node2_pcm_*.log`.

| tree | energy (Eh) | cycles | peak RSS (kernel) | where | SCF |
|---|---|---|---|---|---|
| original (`pyscf-base`) | −1023.7319952041 | 9 | 11.2 GB | PCM `get_D_S` | 2494 s |
| modified (`pyscf`) | −1023.7319952041 | 9 | 13.4 GB | PCM `get_D_S`, on top of the 2.7 GB ERI cache | 2704 s |
| modified + `pcm-mem` dfb50b018 | −1023.7319952040 | 9 | 6.5 GB | PCM `_get_vind` | 2688 s |

- **The peak is PCM's `get_D_S` (`PCM.build`, the first SCF cycle).** It formed S and D from about 13 surface ×
  surface temporaries at once:
  - two `meshgrid` copies of the charge exponents;
  - the distances and their products;
  - two (n, n, 3) displacement arrays.
  That is 10.4 GB above its start for 9,998 points (the laptop measures the same), and nothing
  in it looks at `max_memory`. Every other phase stays under 4 GB in the original tree (XC 3.9, PCM's integral
  steps 3.6, J/K 2.9).
- **It scales with the square of the surface points, which grow with atoms, not basis functions.** n-C47H96/def2-SVP
  (143 atoms, 1138 AOs) has 17,747 points: 2.5 GB a matrix, ~33 GB for stock `get_D_S`.
- **The modified tree adds its ERI cache (It9):**
  - It is sized from `max_memory` minus the RSS at the first J/K build, here "up to 2691 MB". That happens before
    PCM exists, so PCM's memory comes on top.
  - The RSS then stays at 5.4 GB, over the 4 GB `max_memory`. PCM's integral blocks, which are sized from what is
    left, drop to their floor of 8 points.
- **The fix (`pyscf` branch `pcm-mem`, dfb50b018):** `get_D_S` a block of rows (~128 MB of temporaries) at a time,
  with the same expression for every element.
  - Bitwise the same S and D (C26H54 surface); C6H14 B3LYP/C-PCM gives the same energy in every digit.
  - `get_D_S` peaks 1.8 GB above its start instead of 10.4. The SCF's peak drops from 13.4 to 6.5 GB.
- **PCM's integral steps trust `max_memory`:** `_get_v`/`_get_vmat` take (`max_memory` − RSS)·0.9/(8·nao²) points per
  block.
  - With `max_memory` near or above the RAM, one block can outgrow it. With 20000 MB on node2, a test of mine asked
    for ~15.7 GB and the kernel OOM-killed it, and only it.
  - `PYSCF_MAX_MEMORY=12000` (the SGX benchmarks' setting) with this SCF in the modified tree would likely do the
    same.
- **The time is mostly PCM's, and PCM's integrals are where the modified tree loses its 8%.** `_get_v` + `_get_vmat`
  per SCF cycle, by block size (`bench/diag/pcm_blk_time.py`, `scripts/node2_pcmblk.sh`;
  `bench/results/node2/node2_pcm_blk_time.log`):

| block | original | modified |
|---|---|---|
| 8 points | 139 s | 155 s |
| ~125 points | 128 s | 144 s |
| ~475 points | 128 s | 143 s |

  - The SCF makes 11 such calls.
    - The original SCF runs them at ~100–120 points (from its RSS of ~2.8 GB between phases): ~1410 s, 57% of its
      2494 s.
    - The modified one runs them at the 8-point floor: ~1710 s, 63% of its 2704 s.
  - That is ~300 s more, more than the whole 210 s difference. The rest of the SCF (J/K, XC, PCM build and solves) is
    only ~90 s faster in the modified tree (~995 against ~1085 s), so its faster J/K and XC hardly show here. That
    rest was not split further.
  - The split of `_get_v` (`bench/diag/pcm_int3c_time.py`, `scripts/node2_pcmint3c.sh`;
    `bench/results/node2/node2_pcm_int3c.log`; 20 blocks of 123 points, run twice, equal to 3 digits):
    - the int3c2e integrals take 0.615 s a block with stock libcint and 0.712 s with qcint-pgo (+16%);
    - the einsum takes 0.177 s in both.
  - PCM's Python code and libcgto's 3-center driver are the same in both trees, so the integral library makes the
    difference: qcint's int3c2e on (ij|g) triples, with g one primitive s function per surface point.
    - qcint's PGO training has no 3-center integrals, and `-fprofile-partial-training` compiles them as without a
      profile.
    - Stock qcint was not measured.
  - Both steps use `aosym='s1'` (all ordered AO pairs) and `numpy.einsum`.
- **Not done:**
  - C-PCM keeps D (not needed for its energy) and R = −f·I as dense matrices: 1.6 GB of the 2.4 GB resident.
  - The PCM gradient's `get_dD_dS` has the same full-matrix pattern with (3, n, n) tensors and would not fit here.
  - The ERI cache could be sized after PCM is built. That would also keep PCM's blocks off their 8-point floor.
  - PCM's integrals, in both trees: `aosym='s2ij'` with BLAS contractions would halve them, with differences at the
    rounding level.
  - Or GPU4PySCF's route: its PCM calls `int1e_grids(mol, grid_coords, dm=..., charge_exponents=charge_exp**2)`,
    contracted directly.
    - On the CPU, qcint's int1e_grids takes one omega per call, so the points would be grouped by exponent, each
      group with omega = its exponent.
    - PCM's exponents, XI/(R_vdw·√w), take one value per Lebedev weight class and atomic radius, a few dozen here.
    - That is the SGX integral, made 2.0–2.6x faster in It25/It26 (at omega = 0).

## node2: ORCA 6.1.1 on the B3LYP/PCM system, 2026-11-04
Asked for by the user: "Could you run an orca timing comparison on this system?" The system is the one of the
section above: n-C26H54/def2-TZVP, B3LYP, C-PCM (water).
- **Setup:**
  - ORCA 6.1.1, the user's install on node2. It runs 6 MPI ranks (`OMP_NUM_THREADS=1`, `%maxcore 1800`); PySCF ran 6
    OpenMP threads.
  - OpenMPI 4.1.8 for it, built in user space by `scripts/node2_openmpi.sh` (`~/Chemistry/openmpi-4.1.8`). It needs
    the Fortran bindings: with a C-only build, `orca_startup_mpi` fails in `MPI_Type_match_size` (`MPI_ERR_ARG`).
  - Inputs in `bench/orca/c26/`, run by `scripts/node2_orca.sh`. `bench/diag/rss_watch.py` sums the PSS of all ORCA
    and OpenMPI processes, so memory shared between the ranks counts once.
  - Raw outputs in `bench/results/node2/orca-raw/`, which is not in git: ORCA's EULA (3c, 3d) limits sharing data
    generated with ORCA to academic purposes. Summary: `bench/results/node2/node2_orca.log`.
- **Matched to PySCF:**
  - B3LYP/G, whose VWN-3 equals for a closed shell the VWN-RPA of libxc's B3LYP (PySCF's);
  - def2-TZVP;
  - C-PCM with ε 78.3553 and 302 Lebedev points on every sphere. ORCA's default radii are PySCF's (1.2 × Bondi:
    C 2.04, H 1.32 Å).
- **Not matched:**
  - the grid: ORCA's DefGrid2 has 326,210 points, PySCF's level 3 896,704;
  - convergence and guess: TightSCF against `conv_tol` 1e-9, PModel against minao;
  - the pruning of the surface: 10,546 points against 9,998.

| ORCA run | time | cycles | Coulomb + exchange | XC | PCM | peak memory | energy (Eh) |
|---|---|---|---|---|---|---|---|
| exact J/K | 1023 s | 12 | 915 s | 19 s | 61 s | 2.5 GB | −1023.7320596187 |
| RIJCOSX | 236 s | 12 | 122 s (RI-J 16, COSX 106) | 20 s | 61 s | 3.0 GB | −1023.7333863173 |
| out of the box (RIJCOSX, ORCA's CPCM(Water)) | 199 s | 12 | 122 s | 20 s | 33 s | 1.9 GB | −1023.7333747396 |

| PySCF run | time | cycles | PCM integral steps | the rest | peak memory | energy (Eh) |
|---|---|---|---|---|---|---|
| original | 2494 s | 9 | ~1410 s | ~1085 s | 11.2 GB | −1023.7319952041 |
| modified | 2704 s | 9 | ~1710 s | ~995 s | 13.4 GB | −1023.7319952041 |
| modified + `pcm-mem` | 2688 s | 9 | ~1710 s | ~980 s | 6.5 GB | −1023.7319952040 |

- ORCA's time is its whole run; PySCF's is the SCF (its whole run is about 1 s longer).
- ORCA's out-of-the-box CPCM(Water): ε 80.151, "constant charge density", 6,610 points.
- PySCF's PCM integral steps are 11 × the per-cycle times of the section above. "The rest" is J/K, XC, the PCM build
  and solves, and DIIS, by difference.
- **ORCA's exact run is 2.4x faster than the original PySCF and 2.6x faster than the modified one. PCM makes the
  difference.**
  - ORCA's CPCM terms take 61 s for the whole SCF, ~5 s a cycle, plus 8.7 s of setup with a Cholesky factorization of
    the PCM matrix. PySCF's two PCM integral steps take 128–155 s a cycle.
  - The rest: PySCF ~1085 s (original) and ~995 s (modified) against ORCA's ~960 s without its CPCM terms. PySCF's grid
    has 2.75x the points; ORCA makes 12 cycles, PySCF 9 (11 Fock builds).
  - So the modified PySCF with PCM integrals as fast as ORCA's would be about as fast as ORCA's exact run. PySCF's J/K
    and XC were not timed separately.
- **Energies:** ORCA's exact energy is 64 µEh below PySCF's.
  - 20 µEh of it is the dielectric energy (−4.5028 against −4.4827 mEh), from the different surfaces. The rest is
    mostly the grids.
  - RIJCOSX is 4.3x faster than ORCA's exact run and 1.33 mEh below it.
- **Memory:** ORCA peaks at 2.5 GB in all (3.0 GB with RIJCOSX), against PySCF's 6.5–13.4 GB.

## node2: PCM's surface integrals through int1e_grids, a prototype, 2026-11-04
Asked for by the user: "Do you have ideas about how the PCM integrals could be done differently to speed them up?
Does the GPU4PySCF show a more efficient alternative?"
- **What GPU4PySCF does** (`gpu4pyscf/solvent/pcm.py`, `gto/int3c1e.py`, `lib/gint/g3c1e.cu`):
  - Its `int1e_grids(mol, grid_coords, dm=... | charges=..., charge_exponents=...)` computes the two contractions
    directly. The (ij|g) tensor is never formed.
    - The density side applies the HRR to the density: D becomes (L+1)(L+2)(L+3)/6 components per shell pair, at
      the combined angular momentum L = l_i + l_j. Each surface point then takes only the vertical recursion and one
      dot product.
    - On the charge side, one thread per shell pair loops over all points and accumulates in registers.
    - Each point has its own exponent: a factor q/(p+q) on the roots and its square root on the prefactor.
  - `build()` makes one `lu_factor` of K; every cycle uses `lu_solve`.
  - For C-PCM and COSMO it makes S only (`with_D=False`) and applies R = −f as a scalar.
- **The same on the CPU, with what PySCF has:**
  - int1e_grids takes Gaussian charges through `env[PTR_RINV_ZETA]`, one exponent per call. PCM's exponents
    XI/(R_vdw·√w) take 24 values here, one per Lebedev weight class (12) and element, so the points go in 24 groups.
  - SGX's C driver has both contractions, with s2 symmetry: `SGXnrs2_ijg_ji_g` (Σ_ij D_ij (ij|g), the potential)
    and `SGXnrs2_ijg_g_ij` (Σ_g q_g (ij|g), the Fock term, lower triangle then `lib.hermi_triu`).
    `sgx_jk._gen_jk_direct(mol, 's2', True, False, tol)` gives them, with an overlap prescreen (here at 1e-15).
  - `bench/diag/pcm_int1e_proto.py` (`scripts/node2_pcmproto.sh`; `bench/results/node2/node2_pcm_int1e_proto.log`)
    compares the two paths at the initial guess density. It also times one Cholesky factorization of K against
    `_get_vind`'s two `numpy.linalg.solve`.

| n-C26H54/def2-TZVP, 6 threads | original tree (libcint) | modified tree (qcint) |
|---|---|---|
| potential: int3c2e → int1e_grids | 64.6 → 10.9 s (5.9x) | 72.7 → 8.8 s (8.2x) |
| Fock term | 63.8 → 11.1 s (5.7x) | 72.0 → 9.1 s (8.0x) |
| solves: two `solve` → one Cholesky, then a solve | 5.6 → 1.9 s, then 0.43 s | 5.4 → 1.9 s, then 0.43 s |

- **Exact to rounding:** max |dv| 1.7e-12 (|v| up to 19.9), max |dV| 2.1e-17, E_diel within 1.9e-14 Eh,
  tr(D dV) 6e-16. q from the Cholesky factorization is within 5.5e-18. For C-PCM the two solves give the same q, since
  K = S is symmetric and R = −f·I.
- **qcint's int1e_grids is faster than libcint's (8.8 against 10.9 s), while its int3c2e is slower.**
- **Projection for the SCF of the sections above, not measured:**
  - modified tree: 2704 s → ~1140 s. The rest stays ~936 s; PCM integrals 11 × 17.9 s; one Cholesky plus 11 solves.
  - original tree: 2494 s → ~1270 s.
  - ORCA's exact run took 1023 s; its CPCM terms are still ~3.5x faster per cycle (~5 s against 17.9 s).
- **The new path needs no (nao, nao, block) buffer.** The integral steps no longer depend on `max_memory`, so the
  8-point floor and the out-of-memory risk go away.
- **Further steps (not done):**
  - qcint's Gaussian-charge branch of int1e_grids (`zeta > 0`) still takes the Rys roots point by point. It25's
    chunked roots cover only the point-charge branch, but this branch has the same x, scaled by ζ/(ζ + a_ij).
  - A per-point exponent in qcint would make the grouping unnecessary.
  - A GPU4PySCF-like kernel: the HRR on the density for the potential, and for the Fock term the transpose
    (accumulate the combined-L components over the points, then one HRR per pair). That saves the per-point HRR,
    the c2s and the written (ij|g) block.
  - PCM gradients use the same int3c2e/fakemol pattern; int1e_grids_ip, as in the SGX gradients, would apply.
  - Not exact, so opt-in only: fewer surface points (ORCA's default discretization gave 6,610 here) and a multipole
    far field.

## The new PCM code (pyscf `pcm-mem` bd44b24c9) on the B3LYP/PCM system, 2026-11-04
The user said "Yes please do that" to: int1e_grids for the integrals, K factorized once, no D or dense R for C-PCM, on
the `pcm-mem` branch, then a full SCF on node2.
- **The change** (`pcm.py`, `grad/pcm.py`, `hessian/pcm.py`, `test/test_pcm.py`):
  - `_int1e_grids_dot` is the prototype's path: int1e_grids under `mol.with_rinv_zeta(ξ²)` for each group of equal
    exponents, through `sgx_jk._gen_jk_direct(mol, 's2', ...)`, with an overlap cutoff of 1e-15. `_get_v` and
    `_get_vmat` call it.
    - On n-C12H26/def2-TZVP the cutoff left the results as with none (max |dv| 6.3e-13 against int3c2e either way),
      and was 15% faster. At 1e-13 a difference showed (|dV| 9.7e-16).
  - `_K_solve` factorizes K at its first solve and keeps the factor in `_intermediates`: `K_cho` (Cholesky) for C-PCM
    and COSMO, where K = S; `K_lu` otherwise.
    - It serves `_get_q` (in `_get_vind` and `_B_dot_x`) and the gradient's and Hessian's solves of v.
    - SMD's own `build()` needs no change.
  - C-PCM and COSMO make no D and keep `R = None`; `_R_dot` applies R = −f, or the dense R of the other models, to
    the last axis.
    - Their two solves give the same charges, so `_get_q` makes one.
    - The Hessian's C-PCM `@ R` and `get_dqsym_dx_fix_K_R` use `_R_dot`.
  - `test_memory_blocking` (gh-3266) instrumented `df.incore.aux_e2`, which is no longer called. It now checks that the
    results do not depend on `max_memory`, and checks them against the int3c2e integrals of fakemol's charges.
- **Laptop checks against the code before:**
  - `bench/diag/pcm_check.py` (formaldehyde/def2-SVP): SCF energies within 1.7e-13 for C-PCM, COSMO, IEF-PCM,
    SS(V)PE, UKS and Cartesian, with the same cycles.
  - Also: gradients 2.7e-14, RHF Hessians (C-PCM, IEF-PCM) 2.3e-13, TDA with equilibrium solvation 2.8e-13, SMD's
    electrostatics 6e-17.
  - `pyscf/solvent/test`: PCM, PCM gradient, Hessian and TDSCF, SMD and COSMO-RS: 74 passed, 47 skipped. The skips
    are SMD's Fortran library, not built here, and PCM-TDDFT gradients, not available.
- **node2, the full SCF** (`scripts/node2_pcmint.sh`; `bench/results/node2/node2_pcm_pyscf-pcmint.log`,
  `node2_pcm_pyscf-base-pcmint.log`):
  - The branch's solvent files went into copies of both trees (`pyscf/solvent` is the same in both before the branch).
  - `bench/diag/pcm_memory.py` now also gives each phase's wall time.

| run | SCF | J/K | XC | PCM integrals + solves | peak RSS | energy (Eh) |
|---|---|---|---|---|---|---|
| original tree, old PCM code | 2494 s | | | ~1410 + ~62 s | 11.2 GB | −1023.7319952041 |
| original tree, new PCM code | 1272 s | 868 s | 146 s | 242 + 9 s | 4.2 GB | −1023.7319952041 |
| modified tree, old PCM code | 2704 s | | | ~1710 + ~59 s | 13.4 GB | −1023.7319952041 |
| modified tree, new PCM code | 1122 s | 719 s | 191 s | 197 + 8 s | 5.7 GB | −1023.7319952041 |
| ORCA, exact J/K | 1023 s | 915 s | 19 s | 61 s (+ 8.7 s setup) | 2.5 GB | −1023.7320596187 |

- **The SCF is 2.4x faster in the modified tree and 2.0x faster in the original one**, with the same 9 cycles. The
  cycle energies agree within 1e-11 Eh (the printed digits). The PCM build takes 2.2 s; the projection was ~1140 s.
- **The modified tree's XC is slower than the original's: 191 against 146 s.**
  - Its RSS stays above `max_memory` (4.7 GB against 4000 MB): the ERI cache (2.7 GB, sized at the first J/K build,
    before PCM) plus PCM's S and its Cholesky factor (1.6 GB).
  - So the XC blocks run at their floor and the fused XC has no room for its radial cache. Not measured further.
  - Sizing the cache after PCM is built, or leaving room for the later phases, is the open option.
- **The modified tree's J/K is 1.21x faster than the original's (719 against 868 s);** the 2.7 GB cache holds only a
  part of the ERIs at this size.
- **Against ORCA's exact run** (1023 s): PySCF's J/K is faster (719 against 915 s, with 11 Fock builds against
  ORCA's ~12).
  - The XC is 10x slower: the grid has 2.75x the points, at ~4x the cost per point.
  - PCM still takes 205 against ~70 s.

## PCM merged, the solvent first in get_veff, SIMD roots for Gaussian charges, 2026-11-04
The user said "All those sound good" to three steps: merge `pcm-mem` into `perf`; size the ERI cache after the PCM
build; SIMD Rys roots for qcint's Gaussian-charge branch of int1e_grids.
- **`pcm-mem` merged:** `perf` fast-forwarded to bd44b24c9. The PCM tests pass, and node2's `pyscf` tree has the
  branch's solvent files.
- **The solvent part first** (pyscf `perf` f1a9356c8, branch `pcm-cache`):
  - Before, `SCFWithSolvent.get_veff` made the J/K and XC first. Their caches were sized at their first calls, before
    the solvent model existed.
  - Now the solvent part comes first; it depends on dm only. `dm=None` becomes the density matrix, as in the base
    get_veff.
  - Laptop: `pcm_check.py` as before; `pyscf/solvent/test` 129 passed (ddCOSMO and ddPCM share the mixin).
- **qcint SIMD roots for Gaussian charges** (qcint `sgx-exp` 54bef5c):
  - The last branch of `CINTg0_1e_grids` covers Gaussian charges, the long-range operator and both. It now takes the
    roots a SIMD chunk at a time, as the bare Coulomb branch does: one root inlined, all-large-x chunks, two roots by
    `CINTrys_roots_chunk`.
  - It uses x = a0·|r_ij − r_g|² and t² times θ; the bare Coulomb branch is not touched.
  - Laptop, int1e_grids of n-C6H14/def2-TZVP on 600 points: long range, Gaussian charge and both agree with stock
    libcint within 2.7e-16, as `sgx-exp` did before; the bare operator is bitwise the same.
  - node2 (`scripts/node2_zeta.sh`, PGO build `qcint-sx-zeta`; `bench/results/node2/node2_zeta.log`): SGX's
    integrals of caffeine and C20 (1.36e9) are bitwise those of `qcint-sx-opt3`, in the same time.
  - PCM's integrals of n-C26H54/def2-TZVP (`bench/diag/pcm_int_time.py`, 6 threads):

| qcint | `_get_v` | `_get_vmat` |
|---|---|---|
| `perf` (qcint-pgo) | 8.76 s | 9.03 s |
| `sgx-exp` before (qcint-sx-opt3) | 8.37 s | 8.67 s |
| `sgx-exp` 54bef5c (qcint-sx-zeta) | 3.93 s | 4.18 s |

  - Against int3c2e: max |dv| 1.6e-12 and max |dV| 1.9e-17 with each build.
- **The full SCF on node2** (modified tree, B3LYP/C-PCM, max_memory 4000 MB; `node2_pcm_pyscf-pcmcache.log`,
  `node2_pcm_pyscf-sxzeta.log`; `scripts/node2_pcmcache.sh`, `node2_pcmzeta_scf.sh`):

| modified tree | SCF | J/K | XC | PCM | peak RSS |
|---|---|---|---|---|---|
| old PCM code | 2704 s | | | ~1770 s | 13.4 GB |
| new PCM code (bd44b24c9) | 1122 s | 719 s | 191 s | 205 s | 5.7 GB |
| + the solvent first (f1a9356c8) | 1045 s | 783 s | 51 s | 205 s | 3.1 GB |
| + qcint `sgx-exp` 54bef5c (`LD_LIBRARY_PATH`) | 920 s | 763 s | 51 s | 99 s | 3.1 GB |
| ORCA, exact J/K | 1023 s | 915 s | 19 s | ~70 s | 2.5 GB |

- **All runs give −1023.7319952041 in 9 cycles;** the cycle energies agree within 1e-11.
- **With the solvent first, 1508 MB are left for the ERI cache, under 1% of the ERIs, so `semi_direct_eri` stores
  none.** It9's rule is None below 1%. The 2.7 GB cache before held about 1%: it saved 64 s of J/K, but it pushed the
  RSS over `max_memory`, which cost the XC 140 s.
- **The XC at its floor (224-point blocks) took 3.7x the time it takes with room (191 against 51 s);** a small
  `max_memory` margin is costly for the fused XC.
- **The last line needs qcint `sgx-exp` (6 commits on `perf` a95651f) in the modified tree's build.** It is not merged.
  Its J/K is 2.5% faster from It25's roots in the ERIs.

## qcint `sgx-exp` merged into `perf`; the semi-direct 1% rule, 2026-11-04
The user said "Let's merge" and asked: "For lowering the cache threshold is there any downside?"
- **The merge:** qcint `perf` fast-forwarded to `sgx-exp` 54bef5c (It25, It26, the Gaussian-charge roots).
  - qcint-pgo rebuilt in place on the laptop and node2; the trees' `deps` link it, and the installed `cint.h` and
    `cint_funcs.h` are unchanged. The old builds are kept as `install/qcint-pgo-a95651f`.
  - node2's `qcint/src` has the merged sources (`scripts/node2_qcintmerge.sh`; `bench/results/node2/node2_qcintmerge.log`).
- **ERIs, new against old** (qbench; laptop water and caffeine, node2 caffeine SVP and TZVP): within 1.1-1.7e-16, with
  the same nonzero quartets; water's int2e_ip1 too.
  - node2, one thread: caffeine/SVP int2e 9.03 -> 8.53 s, caffeine/TZVP 70.5 -> 68.7 s.
- **PySCF tests on the laptop** (gto/test_moleintor, scf/test_vhf, sgx, solvent, dft/test_numint): 197 passed. The 2
  failures are `test_dm_screening`'s 7.08e-12 against 5e-12, qcint's known one (It6), the same with the old build.
- **The PCM SCF with the merged build, no `LD_LIBRARY_PATH`:** 918.6 s, −1023.7319952041 in 9 cycles, 3.1 GB, as the
  qcint-sx-zeta run (`node2_pcm_pyscf-merged.log`).
- **The 1% rule, measured** (`bench/diag/pcm_memory.py --sd-min-fraction 0.001`, a copy of `_vhf.semi_direct_eri` with
  that share; `node2_pcm_pyscf-merged-sd0.001.log`):

| semi-direct rule | ERIs stored | SCF | J/K | XC | peak RSS |
|---|---|---|---|---|---|
| none below 1% (It9) | none (1505 MB available) | 918.6 s | 762.7 s | 50.9 s | 3.1 GB |
| none below 0.1% | 0.60%, 1480 MB | 919.1 s | 729.0 s | 85.2 s | 4.6 GB |

  - The cache saved 34 s of J/K, and the XC lost 34 s: its blocks are sized from the memory the cache took.
  - The energies agree within 1e-11 cycle by cycle.
  - The threshold binds only when memory is short, which is when the cache and the XC compete for it. So lowering it
    trades XC time for J/K time there, and with room it does not apply.
  - A better rule would size the cache after a reserve for the XC's blocks. Not done.

## PCM's surface integrals in one call over the threads (pyscf `perf` 6acd867c1), 2026-11-04
The user said "This is quite competitive with orca although the PCM is still notably slower", then "Sure and then we
can do the GPU style kernel" to doing this first.
- **Where the time went:**
  - py-spy (`--native`), laptop, n-C12H26/def2-TZVP, one thread:
    - int1e_grids 76%: `CINTg0_1e_grids` 44% (the roots in it 5%), `CINTgout1e_grids` 19%, c2s 7%;
    - the contractions 21% (potential 13%, Fock term 8%);
    - the per-call setup (`_gen_jk_direct`) 0.013 s.
  - 1, 2, 4 and 8 threads took 1.00, 0.83, 0.79 and 0.84 s. SGX's driver splits its work by 224-point batch, and each
    group of equal exponents (one driver call) held one or two.
- **The change:**
  - `SGXnr_direct_drv_batches` takes a table of batches with an exponent for each, set in each thread's copy of env.
    `SGXnr_direct_drv` is the same loop with fixed batches.
  - `_int1e_grids_dot` sorts the points by exponent (stable), cuts each group into batches of up to 224 points and
    makes one call.
- **Exactness:**
  - Each point's sum is as before, and the potential is bitwise the same from run to run. The Fock term's per-thread
    sums are added in the threads' order, as in SGX's J (1.7e-17 from run to run).
  - `pcm_check.py` against the code before the PCM work: energies 1.7e-13, gradients 2.0e-14, TDA 1.6e-13, Hessians
    4.7e-13.
  - `pyscf/solvent/test` and `pyscf/sgx/test`: 137 passed, and the 2 known qcint `test_dm_screening` failures.
- **Speed:**
  - Laptop (C12H26): 1.01, 0.55, 0.33 and 0.25 s at 1, 2, 4 and 8 threads.
  - node2 (`scripts/node2_pcmpar.sh`; `bench/results/node2/node2_pcmpar.log`, `node2_pcm_pyscf-pcmpar.log`):
    `_get_v` 3.93 -> 1.37 s, `_get_vmat` 4.18 -> 1.42 s.
  - The SCF: 918.6 -> 857.6 s, −1023.7319952041 in 9 cycles, 3.1 GB.
  - PCM 99 -> 39 s in all. ORCA's CPCM terms take ~70 s; its exact run 1023 s.

## The GPU-style PCM kernel, an experiment that loses on node2, 2026-11-04
The user had said "...then we can do the GPU style kernel".
- **What it is** (qcint branch `pcm-kernel` 08cc5d9, pyscf branch `pcm-kernel` bbf8aecad; neither merged):
  - `int1e_grids_dm_contract_cart` and `int1e_grids_charge_contract_cart` (`src/cint1e_grids_contract.c`) take one
    shell pair over all the batches. The pair is ordered so that A is the larger shell.
  - `CINTg0_1e_grids` runs with lj set to 0, so it does only the vertical recursion, to li+lj on A.
  - The density goes once to A's monomials t, li <= |t| <= li+lj, through (x−B)^j = Σ_k C(j,k) (A−B)^(j−k) (x−A)^k.
    Each point then takes Σ_t D_t gx gy gz, factored over z, y and x. General contractions are folded in per
    primitive pair.
  - For charges, the sums over the points are taken on the monomials, and the HRR is applied to them at the end.
  - `SGXnr_grids_contract_drv` runs the shell pairs, a fixed cyclic set per thread; the potential is reduced in thread
    order.
  - `pcm._int1e_grids_contract` works on Cartesian functions, with c2s on the matrices, and is taken when the library
    has the kernels.
- **Exact:**
  - Against int1e_grids_cart for every ordered shell pair of formaldehyde: def2-TZVP within 4.7e-15 relative a pair,
    cc-pVTZ (general contractions) 1.7e-14.
  - Against the SGX path: C6H14 and C12H26, spherical and Cartesian, |dv| <= 4.7e-13, |dV| <= 4e-16
    (`bench/diag/pcm_kernel_check.py`, `pcm_contract_vs_sgx.py`).
- **Speed against the SGX path:**

| | `_get_v` | `_get_vmat` |
|---|---|---|
| laptop (Zen 5, AVX-512, non-PGO), C12H26, 1 thread | 0.99 -> 0.81 s | 1.01 -> 0.86 s |
| node2 (Zen 2, AVX2, PGO), C26H54, 1 thread | 7.75 -> 9.45 s | 8.04 -> 9.44 s |
| node2, C26H54, 6 threads | 1.36 -> 1.65 s | 1.42 -> 1.65 s |

  (`scripts/node2_pcmkernel.sh`; `bench/results/node2/node2_pcmkernel.log`)
- **Why:**
  - py-spy (laptop, one thread): `CINTg0_1e_grids` 35% + roots 7%, and the dot products 53%.
  - The dot products over the monomials cost about as many FMAs as libcint's per-point products they replace: 31
    against 36 for (dd), 74 against 100 for (ff). So the HRR, the c2s and the contraction pass are all that is saved.
  - Each FMA needs two loads (the density's broadcast and a strided gz). With AVX2's four lanes that halves the FMA
    rate.
  - Keeping a y/z row per root or the tz run's sums in local arrays was slower: GCC keeps them in memory, since their
    sizes are run time.
  - GPU4PySCF gets its gain from kernels templated on L, which keep everything in registers. On the CPU that would
    need generated code for each (li, lj).
- **The laptop and node2 are back on `perf`:** builds, sources and node2's pyscf tree; node2 keeps
  `install/qcint-sx-contract`.
- **What is left of PCM:** ~3.6% of the SCF (31 s of 857 s). A templated or generated kernel could at most take a
  fraction of that.

## ORCA's RIJCOSX against PySCF's SGX, with PCM, on the large system (node2), 2026-11-05
Asked for by the user: "How does the large system orca RIJCOSX PCM compare to PySCF SGX PCM?" The system is
n-C26H54/def2-TZVP, B3LYP, C-PCM (water), as in the ORCA comparison above.
- **PySCF:** the `pyscf` tree (`perf` with the new PCM code, qcint `perf` with It25 and It26) and `sgx.sgx_fit(mf,
  pjs=True)`, the RIJCOSX-like setup:
  - density-fitted J with 2,922 auxiliary functions (on disk: `PYSCF_TMPDIR`, since node2's /tmp is tmpfs);
  - SGX K with P-junction screening, on SGX's default grid (227,904 points in the last cycle);
  - C-PCM as in the exact runs; 6 threads; `max_memory` 4000 MB.
  - `bench/diag/pcm_memory.py --sgx pjs`, `scripts/node2_pcmsgx.sh`; `bench/results/node2/node2_pcm_pyscf-sgx-pjs.log`.

| run | time | cycles | J and K | XC | PCM | peak memory | energy (Eh) | error against own exact |
|---|---|---|---|---|---|---|---|---|
| ORCA RIJCOSX, the matched CPCM | 236 s | 12 | 122 s (RI-J 16, COSX 106) | 20 s | 61 s (+ 8.7 setup) | 3.0 GB | −1023.7333863 | −1.33 mEh |
| ORCA RIJCOSX, its CPCM(Water) | 199 s | 12 | 122 s | 20 s | 33 s | 1.9 GB | −1023.7333747 | |
| PySCF SGX pjs | 560 s | 9 | 461 s | 51 s | 41 s | 3.5 GB | −1023.7328749 | −0.88 mEh |
| PySCF exact (`perf`) | 858 s | 9 | 761 s | 51 s | 41 s | 3.1 GB | −1023.7319952 | |
| ORCA exact | 1023 s | 12 | 915 s | 19 s | 61 s | 2.5 GB | −1023.7320596 | |

- **ORCA's RIJCOSX is 2.4x faster than PySCF's SGX, and all of that is the approximate J/K: 122 against 461 s.**
  - PySCF's SGX takes ~42 s a build on one grid of 228k points.
  - ORCA's COSX takes ~9 s a cycle on three grids (41,658, 85,740 and 185,684 points), the small ones in the early
    cycles. RI-J adds ~1.4 s a cycle.
  - XC (51 against 20 s; PySCF's grid has 2.75x the points) and PCM (41 against ~70 s) are as in the exact runs.
- **The errors are alike:** −0.88 mEh for PySCF's SGX against its exact energy, −1.33 mEh for ORCA's RIJCOSX against
  its own.
- **Approximate against exact in each code:** PySCF's SGX is 1.5x faster than PySCF's exact run, ORCA's RIJCOSX 4.3x
  faster than ORCA's exact run. The exact codes are within 1.2x of each other (858 against 1023 s); their
  approximations are not.
- **What is not in it:** pyscf `sgx-exp`'s SGX changes (It25 and It26, PySCF side; their K builds were 7–13% faster).
- **The default SGX mode (`pjs=False`: SGX J and K) crashes:** `assert i0 % BLKSIZE == 0` in `sgx_jk.get_jk_favorj`.
  - Outside debug mode, `get_jk_favorj` and `get_jk_favork` take a block of the grid from the memory left
    (`max(SGX_BLKSIZE, max_memory*1e6/8/((2+2*nset)*nao))`), not rounded to a multiple of `BLKSIZE` (56).
  - A grid that takes more than one block then starts its second block off the AO blocks. Here that is ~30,400 of the
    228k points at 4000 MB.
  - It is upstream's code (pyscf-base has it too). The earlier SGX runs had `max_memory` 12000, where one block took
    the whole grid.
  - The fix: round the block down to a multiple of `SGX_BLKSIZE`, as the debug branch does. Not done.

### PySCF's SGX set up as ORCA's RIJCOSX (node2), 2026-11-05
The user asked whether the default SGX mode is closer to what ORCA does: no. RIJCOSX takes RI-J (def2/J) and COSX
for the exchange only, which is `pjs`. The default mode takes J on the grid too. Then: "Yes please do that" to a
`pjs` run set up as ORCA's.
- **What differed from ORCA:**
  - **The J fitting basis.** For a hybrid, PySCF's `predefined_auxbasis` takes the JK-fitting `def2-tzvp-jkfit`
    (2,922 functions), though `pjs` fits only J. `def2-universal-jfit` has 1,868 functions, as many as ORCA's def2/J.
  - **The grids.** SGX's levels 0, 1 and 2 take Lebedev 50, 110 and 194 at most (`SGX_ANG_MAPPING`), like ORCA's
    GRIDX1, 2 and 3.
    - SGX's default is level 2 throughout (`grids_level_i = grids_level_f = 2`).
    - ORCA iterates on GRIDX1 and GRIDX2, then recomputes the exchange energy once on GRIDX3 (26.5 s).
- **`bench/diag/pcm_memory.py`** gains `--auxbasis`, `--sgx-levels I F` and `--sgx-final-level L`. The last
  evaluates the converged density's energy once more on level L, as ORCA does on GRIDX3.
  - Runs: `scripts/node2_pcmsgx2.sh`; `bench/results/node2/node2_pcm_pyscf-sgx-{orca,jfit,stage}.log`.

| PySCF SGX pjs | time | cycles | SGX J/K | XC | PCM | energy (Eh) | against PySCF's exact |
|---|---|---|---|---|---|---|---|
| defaults (jkfit, level 2) | 560 s | 9 | 461 s | 51 s | 41 s | −1023.7328749 | −0.88 mEh |
| jfit, level 2 | 544 s | 9 | 445 s | 51 s | 41 s | −1023.7333211 | −1.33 mEh |
| jfit, levels 0 -> 2 | 455 s | 10 | 350 s | 56 s | 45 s | −1023.7333211 | −1.33 mEh |
| jfit, levels 0 -> 1, then level 2 once (ORCA's way) | 374 s (318 + 56) | 10 | 258 s | 60 s | 48 s | −1023.7333210 | −1.33 mEh |
| ORCA RIJCOSX, the matched CPCM | 236 s | 12 | 122 s | 20 s | 61 s | −1023.7333863 | −1.33 mEh against ORCA's exact |

- **Set up as ORCA's, PySCF's SGX takes 374 against 236 s: 1.6x, from 2.4x.**
  - The grids gave most of it. The J basis saved only 16 s, since DF-J is small next to the exchange.
  - The last step's energy differs from the self-consistent level 2 by 4e-8 Eh. On level 1 alone the energy is
    −1.45 mEh off.
- **The errors are now the same:** −1.326 mEh against PySCF's exact, −1.327 mEh for ORCA against its own. That error
  is the J fitting basis's: with the JK-fitting basis it is −0.88 mEh.
- **What is left of the 1.6x:**
  - the exchange: 258 against 122 s, about 2x per grid point. ORCA's exchange takes ~9 s a cycle on 42k–86k points;
    PySCF's levels 0/1 take ~18 s a call on about as many;
  - the XC: 60 against 20 s, on a grid of 2.75x the points;
  - PCM is ahead (48 against ~70 s).
- **pyscf `sgx-exp`'s SGX changes** (7–13% on K builds) are still not in `perf`.

## The CPU setup in one command (`scripts/cpu_setup.sh`), 2026-11-05
The user asked for the CPU setup as a one-line command, like the GPU's:
`git clone https://github.com/Dpgibney/my-pyscf && cd my-pyscf && bash scripts/cpu_setup.sh`.
- **What it does:**
  - It clones the trees of `scripts/pins.txt` with depth 1 (tags and commits, from GitHub and GitLab) and makes
    the venv with the pinned numpy/scipy/h5py.
  - It builds in this order: OpenBLAS, libxc, a plain qcint, PySCF against it, qbench, then qcint with PGO in
    place. The PGO training needs a built PySCF tree for its molecule dumps, so the plain qcint comes first.
  - It ends with a water/def2-SVP B3LYP energy and the libraries loaded. `--with-base` adds pyscf-base with the
    stock libcint, and `bench/kernels.py` against it.
  - Each step is stamped with its trees' commits (`build/cpu_setup/`), so a rerun after new pins rebuilds only
    what changed; a new qcint rebuilds the plain qcint, PySCF and the PGO.
- **Fixes it needed:**
  - `dump_basis.py` had the laptop's path, and `train_qcint_pgo.sh` did not create `scratch/qcint`.
  - The dumps are now written before the instrumented build (`train_qcint_pgo.sh --dumps`), so the profile is
    the training alone, as on the laptop.
  - `pyscf-env.sh` defaults to the physical cores and 2/3 of the RAM, at most 20000 MB: 10 and 20000 on the
    laptop, as before, and 6 and 10738 on node2.
- **Tested on node2 from a fresh clone at another path (`~/cpu_setup_test`, `--with-base`): 20 min.**
  - With 12 jobs: OpenBLAS 299 s, libxc 131 s, qcint 24 s, PySCF 14 s, PGO 382 s, libcint 17 s, pyscf-base 10 s.
  - Water B3LYP: −76.3581418310 Eh in 7 cycles, −5.7e-14 from the laptop. `kernels.py`: all 77 arrays agree with
    pyscf-base.
  - A rerun: every step up to date, 1.8 s.
  - `fetch()` alone moved a clone to a new tag, an older tag, a bare SHA and back, each a depth-1 fetch. A tree
    with uncommitted changes stops it.

## Thread and memory defaults that follow a Slurm job's allocation, 2026-11-05
The user asked whether the setup works on a Slurm node where a job uses less than the node, then whether the
values could be dynamic or user-settable, since each job's request will vary. They chose defaults that follow each
job and stay overridable, with no 20000 MB cap inside jobs.
- **What was wrong:** `pyscf-env.sh` read the whole node (lscpu's cores, MemTotal). Under `taskset -c 0-3` (2 of the
  laptop's cores) it still set 10 threads; in a 6 GiB memory cgroup, still 20000 MB. A job given part of a node
  would oversubscribe its CPUs, or be OOM-killed once the semi-direct cache filled.
- **Now (`e082fa1`):** `scripts/resources.py` measures what the process may use.
  - Cores: the physical cores (`thread_siblings_list`) of its CPU affinity mask, which Slurm's binding or cpuset
    sets. At most `$SLURM_CPUS_PER_TASK` (else `$SLURM_CPUS_ON_NODE`), for clusters that don't bind jobs.
  - Memory: the least of MemTotal, the cgroup v1/v2 limits from the process's cgroup up, and `$SLURM_MEM_PER_NODE`
    (or `$SLURM_MEM_PER_CPU` x `$SLURM_CPUS_ON_NODE`; `--mem=0` counts as the whole node).
  - `pyscf-env.sh` sets `OMP_NUM_THREADS` to those cores and `PYSCF_MAX_MEMORY` to 2/3 of that memory, with the
    20000 MB cap only outside Slurm jobs. Exported values still win. `cpu_setup.sh` builds with the mask's CPUs
    and runs its checks on its cores.
- **Laptop** (CPUs 2k and 2k+1 share a core; scopes from `systemd-run --user --scope -p MemoryMax=...`):

| case | OMP_NUM_THREADS | PYSCF_MAX_MEMORY |
|---|---|---|
| nothing set (10 cores, 32 GB) | 10 | 20000 |
| `taskset -c 0-3` (2 cores x 2 threads) | 2 | 20000 |
| `taskset -c 0,2,4,6` (4 cores) | 4 | 20000 |
| 6 GiB memory cgroup | 10 | 4294 |
| Slurm `-c 3 --mem=8000` (variables only) | 3 | 5592 |
| Slurm `-n 4 --mem-per-cpu=2000` (variables only) | 4 | 5592 |
| Slurm `--mem=0` (whole node): no cap | 10 | 21586 |
| Slurm `-c 8 --mem=16G`, CPUs 0-7, 16 GiB cgroup | 4 | 11452 |
| Slurm `-c 8 --mem=64G` (more than the RAM) | 4 | 21586 |
| exported 7 and 1234 | 7 | 1234 |

  PySCF picks them up: `lib.num_threads()` 4 and `mol.max_memory` 11452 in the `-c 8 --mem=16G` case.
- **node2: `cpu_setup.sh --with-base` from a fresh clone as a `-c 6 --mem=8G` job** (`taskset -c 0-2,6-8`, an 8 GiB
  scope and Slurm's variables): 3 cores, 6 CPUs and 8 GB detected, 6 build jobs, all builds within the 8 GiB, 26
  min. Water B3LYP −1.4e-14 Eh from the laptop; `kernels.py` on 3 threads: all 77 arrays agree; then
  `OMP_NUM_THREADS=3 PYSCF_MAX_MEMORY=5726`.
- **A build on the user's cluster failed in PySCF's step**, and the script showed only the log's last 25 lines,
  other files' warnings from the parallel build. A failed step now prints the log's errors first (ninja's
  `FAILED:` lines, compiler and assembler errors, OOM kills, full disks), and the stamps include the GCC version,
  so a new compiler module rebuilds everything (`c463b11`). The cause is not known yet. Our changed C files compile
  cleanly with GCC 13 in gnu99 mode with `-Wpedantic`; their collapsed loops are rectangular.

## A segfault in qcint's int1e_grids under SMD, and SMD's missing module, 2026-11-05
The user's lenalidomide + HPO3(2-) / def2-TZVPD wB97M-V/SMD(water) job (grids (99,590), nlcgrids (50,194), 4
cores and 12 GB on their cluster) died in the first SCF cycle. Their diagnosis: only that geometry, in
`_int1e_grids_dot` (pcm.py:333), inside qcint's int1e_grids.
- **SMD's CDS module was never built.** Upstream compiles libsolvent (the Minnesota Fortran code of SMD's
  cavity-dispersion-solvent structure term) only with `-DENABLE_SMD=ON`, which is off by default. Every SMD
  calculation with our build stopped at its first energy with `RuntimeError: SMD module is not available`; a
  stand-in of the same composition (1001 functions) did on the laptop, confined to 4 cores and 12 GiB.
  - `build_pyscf.sh` now passes the option (`72f8bea`).
  - caffeine/def2-SVP SMD(water), modified against unmodified: B3LYP 2e-12 Eh apart, wB97M-V identical, 13
    cycles each.
- **The segfault: qcint read past the grid coordinates.**
  - CINT1e_grids_loop rounds the point count up to SIMDD and copied the padding lanes' coordinates from past
    the caller's last point: up to 7 points, 168 bytes, with AVX-512. Upstream qcint's gathers read the same.
  - Callers append the grids at the end of env, and the PCM driver gives each thread a copy of env that ends
    exactly there. So the read ran past the copy, and faulted when that crossed into an unmapped page, which
    depends on the number of surface points: only some geometries crashed.
  - AddressSanitizer showed it: a 32-byte read 16 bytes past a thread's 15,792-byte env copy, for water's SMD
    surface.
  - Fixed in qcint (`d7be30d`, tag `perf-2026-11-06`): the padding lanes repeat the block's last point.
    AddressSanitizer is clean for SMD and SGX SCFs and `mol.intor('int1e_grids')`, in AVX-512 and SSE builds.
    SGX and `mol.intor('int1e_grids')` had the same over-read, where the grids end a numpy array.
  - caffeine/def2-SVP B3LYP with C-PCM, IEF-PCM and SMD(water): within 2e-12 Eh of the previous build, same
    cycles. `kernels.py`: all 77 arrays agree with pyscf-base.
  - PySCF's solvent and SGX tests: 188 passed, 3 failed. `test_dm_screening` (RHF, UHF) fails with any qcint
    build, as before. `test_smd_grad::test_grad_solvent` misses Q-Chem's reference by 1.17e-4 against a 1e-4
    tolerance, and unmodified PySCF fails it too.
- **SMD's cost on this system** (8833 surface points, 2 threads): setup 34 s (D and S 7 s, the rest dense
  algebra, mostly D·A·S), CDS 0 s, K's factorization 9.5 s once, the potential and Fock term 18.4 s per cycle.
  A cycle takes 8-9 min on 4 cores (two exchange builds, XC on 1.18M points, VV10 on 242k), so SMD is about 2%
  of it. Its setup is upstream code that holds about six 8833 x 8833 matrices (0.62 GB each) at once: a 7.7 GB
  peak in the 12 GB job.
- `cpu_setup.sh`'s stamps include each step's build scripts (`72f8bea`), so these changes rebuild what they
  touch.

## J/K for range-separated hybrids on def2-TZVPD: quartets in the SIMD lanes (prototype), and the short-range split, 2026-11-06
The user asked whether the J/K integral algorithm could be made significantly faster; of the options offered
they chose both: prototype quartets in the SIMD lanes, and measure a short-range + long-range split.
- **Where a cycle goes** (lenalidomide + HPO3(2-) stand-in, def2-TZVPD, 1001 AOs, wB97M-V, 6 threads, minao
  density): J + full-range K 173 s, long-range K (omega 0.3) 143 s, XC on 1.18M points 44 s, VV10 14 s (7.5 s
  with SG-1 pruning). J/K is 84%.
- **Where the integrals go** (`qbench`, caffeine/def2-TZVPD, 234 shells, 283M quartets, one core): 178 s per
  full-range pass, 218 s long range. Quartets with d or f shells take 87% of it; the fixed per-quartet cost
  (~80 ns) about 13%. The d and f shells (and the diffuse ones) have one primitive: qcint puts the primitive
  combinations of a quartet in its lanes, so 39% of the time runs on its scalar path (one combination) and 47%
  with the 8 AVX-512 lanes about half full (about 4 combinations).
- **The short-range split: no.** wB97M-V's J and full K could come from a long-range pass plus a short-range
  one (J = J_SR + J_LR, K = K_SR + K_LR exactly), if the short-range integrals screen by distance.
  - On the stand-in (6 threads): full-range J+K 180.4 s, long range 152.9 s, short range through PySCF's SR
    driver (`CVHFnr_sr_direct_drv`) 156.6 s: 310 against 333 s, 7%.
  - The SR driver's screening is heuristic (each shell's most diffuse exponent, (ii|kk)-type overlaps,
    float32 logs): K_SR + K_LR differs from K by up to 2e-3, the K energy term by 3.5e-4 Eh (caffeine/SVP:
    1.2e-5).
  - With the regular driver's Schwarz screening the split is exact (6e-12), but the short-range pass is
    slower than the full one (caffeine/SVP 2.2 against 1.4 s): it needs more Rys roots and gains nothing
    from Schwarz bounds.
  - Upstream's `test_direct_sr_vhf` builds its optimizer but never passes it, so it does not exercise the SR
    driver.
- **Quartets in the lanes: about 1.75x on the integrals in a prototype.**
  - `bench/qcint/xq_proto.c`: single-primitive (dd|dd), 8 quartets per vector (GCC vector types, qcint's own
    roots per lane, libcint's recurrences, HRR, cartesian products, cart -> sph). 364,471 quartets: qcint 5524
    ns per quartet, the prototype 1627 (3.4x); integrals within 2.6e-14 of qcint's.
  - `bench/qcint/xq_generic.c`: any l <= 3, segmented contractions summed per lane, full or long range;
    projects a pass from up to 4000 sampled quartets per type, both codes timed on the same samples.
    - Generic (runtime l): 0.97x. Specialized per class (compile-time l through an always-inlined template
      and a 256-way switch): 1.65x full range, 1.62x long range. Single-primitive quartets 2.1-2.7x, 2-7
      combinations 1.6-1.9x, >= 8 combinations 0.6-0.9x (qcint's lanes are already full there).
    - Keeping qcint for >= 8 combinations: 173.5 -> 97.9 s full range (1.77x), 178.2 -> 101.6 s long range
      (1.75x). Sampled integrals within 1.0e-12 (full range, values up to 8; the prototype goes u -> t^2 -> u)
      and 1.0e-13 (long range).
    - Untuned: roots point by point, the cart -> sph coefficients filled at run time, HRR per primitive
      combination.
  - For the stand-in's cycle (J/K 84%, most of it integrals) that would be about 1.4-1.6x.
  - A real implementation: a batched int2e in qcint (up to 8 quartets of one class; kernels generated per
    class; full, long and short range; vectorized roots), the batched direct J/K driver queueing screened
    quartets by class, qbench --compare over all classes, kernels.py and the tight validation, PGO training.
    Several days; derivative integrals (gradients, Hessians) would need their own kernels.

## Quartets in the SIMD lanes: qcint's int2e_sph_batch in the direct J/K, 2026-11-06
After the prototype (above), the user said "Yes go ahead".
- **qcint `int2e_sph_batch`** (`perf` 11295ba, 34ec26d, 9781f05).
  - It computes up to SIMDD quartets of one class together, one quartet per lane, each summing its own
    primitive combinations in turn. A class has the same l and number of primitives on each shell, with
    l <= 3, one contracted function and at most 7 primitives per shell, and the full or long-range
    Coulomb operator.
  - The kernels are an always-inlined template with compile-time l, instantiated for every
    (li,lj,lk,ll) in `src/cint2e_batch_l{0..3}.c` (compiled in parallel, about 45 s). They use
    int2e_sph's scalar-path roots, coefficients and recurrences on the same base centres (i if
    li > lj, else j; k if lk > ll, else l). Starting from the other centre had cost accuracy: up to
    1.4e-12 against 2.5e-14. The roots of all lanes come from `_CINTrys_roots_batch`, as in int2e_sph's
    SIMD path (node2, 4 lanes: 1.63x -> 1.67x full range, 1.58x -> 1.66x long range).
  - API: `int2e_sph_batch`, `int2e_sph_batch_ok`, `int2e_sph_batch_lanes`,
    `int2e_sph_batch_cache_size`.
  - Quartets of 8 or more primitive combinations are batched too. They fill int2e_sph's lanes, but
    are still 1.1-1.2x faster with 8 lanes and 1.3-1.5x with 4, except s and p only with 8 lanes (0.97x).
- **PySCF** (`perf` 9924bd26f).
  - `CVHFnr_direct_jk_batch` queues the screened quartets that `int2e_sph_batch_ok` takes, per row, by
    the (l, nprim) types of their ket shells. It evaluates a queue when it holds a SIMD width of
    quartets, and at the row's end; fewer than half the lanes go quartet by quartet. The contraction
    kernels are unchanged.
  - The qcint functions are weak symbols, so with libcint nothing changes. `_vhf.INT2E_BATCH`
    (config `scf_vhf_int2e_batch`, bench knob `BENCH_INT2E_BATCH`) switches the batching off.
  - The semi-direct cache's stored quartets are still evaluated one at a time on their first call.
  - New test: `test_vhf.py test_direct_int2e_batch`.
- **Harness:** `bench/qcint/bbench.c` compares and times int2e_sph_batch against int2e_sph (every quartet,
  or sampled per type with a projected pass). `train_qcint_pgo.sh` runs it on caffeine when the library
  has the batch functions; `cpu_setup.sh` builds it.
- **Exactness.**
  - Every quartet against int2e_sph, PGO build:
    - laptop (AVX-512, 8 lanes): caffeine/def2-TZVP within 3.6e-14 (all 113M quartets);
    - node2 (AVX2, 4 lanes): within 2.5e-14 (all 113M quartets); caffeine/def2-SVP within 1.8e-15;
    - long range: within 6e-15.
  - qcint's own int2e_sph is bitwise identical to the previous PGO build on all 113M caffeine/def2-TZVP
    quartets, although the training now includes bbench.
  - `kernels.py`: 77/77. `validate.py` full suite: all pass, energies within 3.4e-12 Eh, gradients
    1.7e-12, Hessians 2.0e-11, excitations 8e-14. The tight SCFs' cycle warnings (e.g. 20 vs 19) are
    the scatter earlier gates show too.
  - PySCF's tests (scf dft df grad hessian tdscf solvent sgx): 1049 passed. 7 failed, all known: 4 xcfun
    (not built), 2 SGX `test_dm_screening` (any qcint), `test_smd_grad` (unmodified PySCF too).
- **Speed.**
  - Projected one pass, one core, against int2e_sph, PGO builds:
    - caffeine/def2-TZVPD: full range 2.09x (186 -> 89 s), long range 1.90x;
    - the lenalidomide + HPO3(2-) / def2-TZVPD stand-in: 2.24x;
    - node2 (4 lanes): caffeine/def2-TZVPD 1.86x, the stand-in 1.93x.
  - The stand-in, one call, 6 threads, max_memory 8589: J+K 147.6 -> 90.2 s (1.64x), long-range K
    139.6 -> 67.1 s (2.08x), J/K within 9e-13. Its wB97M-V cycle would go from about 374 to about 215 s.
  - `bench/results/20261106-112331_int2e-batch.json`, laptop, 10 threads, ABBA x 2, the same tree with
    the batching off and on. SCF cycles identical in every pair, energies within 3.6e-12 Eh.

| case | off (s) | on (s) | total | J/K |
|---|---|---|---|---|
| scf_caffeine_tzvp_b3lyp | 139.3 | 89.8 | 1.55x | 131.0 -> 81.5 (1.61x) |
| scf_c20h42_tzvp_pbe | 245.4 | 142.8 | 1.72x | 221.6 -> 118.6 (1.87x) |
| scf_c20h42_tzvp_b3lyp | 300.0 | 186.5 | 1.61x | 275.3 -> 161.9 (1.70x) |
| scf_c20h42_svp_b3lyp | 29.0 | 21.6 | 1.34x | 23.2 -> 15.7 (1.48x) |
| scf_caffeine_svp_camb3lyp | 12.7 | 10.8 | 1.17x | 9.8 -> 7.8 (1.26x) |
| tda_c20h42_svp_pbe | 47.6 | 43.5 | 1.09x | 22.7 -> 18.2 (1.24x) |
| scf_caffeine_svp_wb97mv | 142.8 | 145.1 | 0.98x (VV10: 125 s of it) | 9.5 -> 7.4 (1.30x) |

- **Not batched yet:** derivative integrals (gradients, Hessians), g functions, shells of more than 7
  primitives, cartesian bases, short range, and the semi-direct cache's first call. SSE builds (SIMDD 2)
  batch 2 quartets.

## Gradients: qcint's int2e_ip1_sph_batch in the gradient J/K, 2026-11-06
The first of the items above, after the user's "Yes and then we can work on what wasn't covered yet".
- **qcint `int2e_ip1_sph_batch`** (`perf` a55f70d).
  - The batched kernels' template takes a derivative order. With 1 it computes int2e_ip1_sph the way
    libcint's nabla1i does, d/dx f_i = i f_{i-1} - 2 a_i f_{i+1}. The recurrences go one higher on i,
    the base centre is chosen from li+1 as int2e_ip1 does, and the 3 components come out as
    int2e_ip1_sph's (component slowest).
  - Instantiated for every (li,lj,lk,ll) with l <= 3 in `src/cint2e_batch_ip1_l{0..3}.c`.
  - `int2e_ip1_sph_batch` and `int2e_ip1_sph_batch_ok` take the same quartets as int2e_sph_batch;
    `int2e_sph_batch_cache_size` covers both.
- **PySCF** (`perf` 328ed43b7), `grad_jk_energy.c`.
  - CVHFgrad_jk_energy evaluated each orbit's int2e_ip1 quartets one at a time. An orbit is now planned
    first: which centres it evaluates, and how their sums go to the atoms by translational invariance.
  - Orbits that int2e_ip1_sph_batch_ok takes are queued by the types (l, nprim) of their 4 shells and
    the set of centres they evaluate, so a queue's orbits need the same integral classes. A full queue
    is evaluated centre by centre with int2e_ip1_sph_batch for all its orbits, each orbit's Gamma
    computed once. Fewer than half the lanes go orbit by orbit.
  - The queues sit in a small hash table per thread, flushed in table order at each chunk's end (and
    when the table fills), so the sums still do not depend on the thread count.
  - `CVHFgrad_jk_energy_mode` takes the switch (`_vhf.INT2E_BATCH`, from grad.rhf);
    `CVHFgrad_jk_energy` keeps its arguments and batches. New test:
    `test_rhf.py test_jk_energy_per_atom_int2e_batch`.
- **Harness** (workspace 1a90c27, 3b233ae): `bbench --ip1` compares and times int2e_ip1_sph_batch
  against int2e_ip1_sph (each ordered bra pair, ket k >= l). `train_qcint_pgo.sh` adds it to the
  training when the library has it.
- **Exactness.**
  - Against int2e_ip1_sph, PGO build, laptop: water/def2-SVP within 2.7e-15 and benzene/def2-SVP
    4.7e-15 (every quartet), long range 5.7e-16; caffeine/def2-TZVP sampled (7.3M quartets) within
    1.5e-13, values up to 9.6.
  - int2e_sph_batch on caffeine/def2-TZVP: within 1.9e-14 of int2e_sph (all 113M quartets).
  - qcint's own int2e_sph (113M quartets) and int2e_ip1_sph (57.6M quartets) are bitwise identical to
    the previous PGO build.
  - `kernels.py`: 77/77. `validate.py` full suite: all pass, gradients within 1.6e-12, Hessians
    2.4e-11. Its tight SCFs' cycle warnings (r2SCAN 33 vs 25, CAM-B3LYP 30 vs 24) are tight-convergence
    scatter; at normal convergence (below) the cycles are identical.
  - PySCF's tests: 1050 passed, the same 7 known failures.
- **Speed.**
  - Projected one pass, one core, caffeine/def2-TZVP: 661.7 -> 326.9 s (2.02x). s and p with one
    primitive combination 4.7x, d 2.6-2.8x, f 2.1-2.3x, 8 or more combinations 1.07-1.21x.
  - `bench/results/20261106-145704_int2e-batch-grad.json`, laptop, 10 threads, ABBA x 2, the same tree
    with the batching off and on. The grad cases time the gradient only. Gradients within 7e-13 of the
    batching off (off against off: up to 3.4e-13).

| case | off (s) | on (s) | total | J/K |
|---|---|---|---|---|
| grad_c20h42_svp_b3lyp | 20.7 | 12.3 | 1.68x | 18.9 -> 10.6 (1.78x) |
| grad_caffeine_svp_b3lyp | 4.6 | 3.0 | 1.54x | 4.1 -> 2.5 (1.66x) |
| grad_caffeine+_svp_b3lyp_uks | 5.9 | 4.1 | 1.44x | 4.9 -> 3.1 (1.60x) |
| grad_benzene_svp_b3lyp | 0.5 | 0.4 | 1.27x | 0.4 -> 0.3 (1.38x) |
| grad_caffeine_svp_wb97mv | 22.5 | 18.7 | 1.21x (VV10's gradient) | 9.8 -> 6.4 (1.52x) |

  - The same file has caffeine/def2-SVP SCFs at normal convergence: cycles identical (r2SCAN 14, B3LYP
    15, PBE 16, CAM-B3LYP 13), energies within 3.0e-12 Eh. B3LYP, PBE and r2SCAN 0.95-1.06x, within the
    noise: under the bench's max_memory of 4000 MB the semi-direct cache holds most of their quartets
    (all of them are 2.6 GB), and its first call is not batched yet. CAM-B3LYP 1.25x: its long-range K
    is evaluated directly.

## The semi-direct cache's first call through int2e_sph_batch, 2026-11-06
- **PySCF** (`perf` a485a3ce5).
  - The cache's stored quartets were still evaluated one at a time on their first call. They are now
    queued by the types of their ket shells, as the direct quartets are. A full queue is evaluated with
    int2e_sph_batch, copied to the quartets' slots, marked stored and contracted from the slots.
  - A row's queues are flushed after its stored quartets, before its direct ones. Fewer than half the
    lanes go one at a time.
  - New test: `test_vhf.py test_semi_direct_int2e_batch`. It compares the stored integrals with and
    without the batching, for all and part of the quartets, full and long range.
- **Exactness.**
  - Semi-direct J/K against direct J/K, benzene/def2-SVP: within 8e-14, with a full and a half-size
    cache, one and two dms, hermi 0 and 1, the batching off and on.
  - caffeine/def2-TZVP, the batching off against on: within 3.2e-13.
  - `validate.py` full: all pass (gradients 1.55e-12, Hessians 7.0e-11). The tight SCFs' cycle warnings
    are the usual scatter.
  - Tests: 1051 passed, the same 7 known failures.
- **Speed.**
  - caffeine/def2-TZVP, 10 threads, an 8.8 GB partial cache (max_memory 12000):
    - the first call 9.1 -> 6.0 s, of which the stored quartets 3.7 -> 2.6 s;
    - the second call 5.4 -> 3.4 s (its direct part, batched before).
  - caffeine/def2-SVP, all 2.6 GB stored: the first call 0.80-1.01 -> 0.58 s; later calls 0.14 s
    (contraction only).
  - `bench/results/20261106-154547_int2e-batch-semidirect.json`: laptop, 10 threads, ABBA x 2, the bench's
    max_memory of 4000 MB. Cycles identical, energies within 4.3e-12 Eh. The caffeine cases' J/K was
    0.96-1.04x in the gradient benchmark above, before this change.

| case | off (s) | on (s) | total | J/K |
|---|---|---|---|---|
| scf_caffeine_svp_b3lyp | 5.8 | 5.4 | 1.07x | 3.3 -> 3.0 (1.09x) |
| scf_caffeine_svp_pbe | 4.6 | 4.2 | 1.09x | 1.8 -> 1.6 (1.14x) |
| scf_caffeine_svp_r2scan | 6.5 | 6.2 | 1.05x | 1.8 -> 1.6 (1.14x) |
| scf_caffeine_svp_camb3lyp | 10.7 | 8.4 | 1.27x | 8.2 -> 6.0 (1.37x) |
| scf_c20h42_svp_b3lyp | 27.1 | 19.6 | 1.38x | 21.6 -> 14.0 (1.54x) |

## The Hessians' integrals batched (qcint int2e_ipip1/ipvip1/ip1ip2_sph_batch), and direct_mapdm/direct_bindm, 2026-11-06
- **qcint** (`perf` 4c59d1e): `int2e_ipip1_sph_batch`, `int2e_ipvip1_sph_batch`, `int2e_ip1ip2_sph_batch`.
  - They have a template of their own (`cint2e_batch_hess.h`), so int2e's and int2e_ip1's validated
    kernels are untouched.
  - They follow int2e_sph's scalar path (autocode/hess.c, g2e_simd1.c):
    - the 2D integrals go one higher on each differentiated centre (two on i for ipip1), from the base
      centres of the raised l;
    - then g1 = nabla_J g0 (nabla_K for ip1ip2, nabla_I for ipip1), g2 = nabla_I g0 and g3 = nabla_I g1,
      at the output's powers;
    - the 9 products come in hess.c's order. Each primitive combination's root sum is added to the
      cartesian integrals, which go to spherical one component at a time.
  - Instantiated for every (li,lj,lk,ll) with l <= 3 in 12 files. The whole library builds in 2m17s on
    16 jobs (16 CPU-minutes).
  - `_ok` is int2e_sph_batch's. `int2e_hess_sph_batch_cache_size` gives their cache: 8.7 MB with 8 lanes,
    sized for (ff|ff).
- **PySCF** (`perf` 46ebf7c53): batching in `CVHFnr_direct_drv` and `CVHFnr_direct_ex_drv`.
  - These are the drivers of `direct_mapdm` and `direct_bindm`. Their users include the Hessian's
    `_partial_hess_2e` (ip1ip2, ipvip1; ipip1 when atmlst is not every atom), `make_h1` (int2e_ip1), and
    `grad.rhf.get_jk` (int2e_ip1: TDDFT and other gradients that build the J/K matrices).
  - Quartets that the batched intor takes are queued per thread, by the types of their 4 shells, in a hash
    table. A full queue is evaluated at once and contracted by the scripts quartet by quartet.
  - The contractions write to the JKArray of the quartet's block, so a queued quartet keeps its block, and
    the JKArrays are pointed at it for the contraction.
  - The queues outlive the blocks, and are flushed at each thread's end, when every output fits in its
    JKArray's stack and keys cache. Each block pair is then stored once and never written back early.
    Otherwise they are flushed at the end of each block.
  - The first version flushed at every block. With the 16-AO blocks that 10 threads bring for benzene,
    most queues stayed short: py-spy showed 48% of the time in the scalar `int2e_ip1ip2_sph` and
    `int2e_ipvip1_sph`. On benzene/def2-SVP, `_partial_hess_2e` at 10 threads was 1.20x that way and is
    1.83x with the queues kept (1 thread: 1.67x -> 1.77x).
  - The kernels of `nr_direct_dot.c` write up to 2*di*dj doubles of scratch after a quartet's
    integrals, so a batch is contracted last quartet first, with room after the last.
  - `CVHFnr_direct_drv_mode` and `CVHFnr_direct_ex_drv_mode` take the switch (`_vhf.INT2E_BATCH`); the old
    entry points batch.
  - New tests: `test_vhf.py test_direct_bindm_int2e_batch` (each batched intor, slices and an excluded
    block, full and long range) and `hessian/test/test_rhf.py test_hess_2e_int2e_batch`.
  - Memory: per thread and call, up to 8.7 MB of the Hessian kernels' cache, 1.4 MB of output and 1 MB of
    queues, touched as used. On 64 threads that is about 0.7 GB during the Hessian's 2e part.
- **Harness** (workspace b1aba50): `bbench --int NAME` (ip1, ipip1, ipvip1, ip1ip2).
  `train_qcint_pgo.sh` trains the three Hessian kernels on benzene and caffeine/def2-TZVP.
- **Exactness.**
  - Against int2e_ipip1_sph, int2e_ipvip1_sph and int2e_ip1ip2_sph, each ordered bra pair with k >= l:
    - laptop (AVX-512, 8 lanes), PGO build, every benzene/def2-SVP quartet: ipip1 within 3.9e-14
      (values up to 53), ipvip1 1.8e-14, ip1ip2 6.0e-15;
    - laptop, every water quartet: within 4.3e-14 (values up to 131), long range 1.1e-14;
    - laptop, caffeine/def2-TZVP sampled (755K quartets): within 8.5e-13 on values up to 540;
    - node2 (AVX2, 4 lanes), plain build, every water and benzene quartet: within 1.1e-13 (water ipip1,
      values up to 131), int2e and int2e_ip1 unchanged (6.7e-16, 4.7e-15).
  - qcint's own functions are bitwise identical to the previous PGO build (a55f70d): int2e_sph on all
    113M caffeine/def2-TZVP quartets, int2e_ip1_sph on all 450M, and the three Hessian integrals on every
    benzene quartet.
  - `kernels.py`: 77/77. `validate.py` full: all pass, Hessians within 6.8e-11, gradients 1.55e-12.
  - The tight caffeine/def2-SVP r2SCAN SCF took 65 cycles once (reference 25). Reruns with the batching
    took 34, 33 and 24; without it, 25, 24 and 33. It is the run-to-run scatter of tight convergence. At
    normal convergence the cycles are identical in every A/B so far.
  - PySCF's tests: 1053 passed, the same 7 known failures.
- **Speed.**
  - Projected one pass, one core, caffeine/def2-TZVP: 2.11-2.13x (PGO, 8 lanes), 1.78-1.80x (node2, 4
    lanes, plain build). s and p with one primitive combination 5.5x, d 3x, f 1.9-2.4x, 8 or more
    combinations 1.15-1.34x.
  - benzene/def2-TZVP B3LYP, 10 threads, plain build: `_partial_hess_2e` 21.4 -> 12.9 s, `make_h1` 7.7 ->
    5.0 s, the Hessian 43.7 -> 34.4 s, within 1.6e-13.
  - caffeine/def2-SVP B3LYP, 10 threads, PGO: `_partial_hess_2e` 35.3 -> 24.6 s, `make_h1` 11.8 -> 9.8 s.
    These are about half of the Hessian's J/K. The other half is the CPHF response J/K: one integral
    pass per iteration for all 72 perturbations, bound by the contractions.
  - `bench/results/20261106-163624_int2e-batch-hess.json`: laptop, PGO, 10 threads, ABBA x 2. Hessians
    within 1.7e-10 of the batching off (off against off: up to 2.6e-11).

| case | off (s) | on (s) | total | J/K |
|---|---|---|---|---|
| hess_benzene_svp_b3lyp | 8.2 | 7.5 | 1.10x | 3.1 -> 2.3 (1.38x) |
| hess_benzene_svp_pbe | 7.5 | 6.8 | 1.11x | 2.6 -> 1.8 (1.45x) |
| hess_benzene+_svp_b3lyp_uks | 15.7 | 14.7 | 1.07x | 4.1 -> 3.2 (1.30x) |
| hess_caffeine_svp_b3lyp | 141.7 | 134.8 | 1.05x | 91.2 -> 80.8 (1.13x) |
| hess_caffeine+_svp_b3lyp_uks | 242.2 | 228.2 | 1.06x | 135.2 -> 120.9 (1.12x) |

- **Not batched yet:**
  - g functions; shells of more than 7 primitives or more than one contracted function; cartesian bases;
    short range.
  - None of these occur in the def2 bases of the workloads: caffeine/def2-TZVP(D) and the
    lenalidomide + HPO3(2-)/def2-TZVPD stand-in are fully batched.
  - cc-pVXZ would be the gap: PySCF keeps their s shells generally contracted (carbon cc-pVDZ: 8
    primitives, 2 functions), so no quartet with them is batched.
- **Next for Hessians:** the CPHF response J/K, whose contractions over many dms dominate it.

## The CPHF step: J/K of many dms through dense matrix products, 2026-11-06
After the push, the user said "let's do the CPHF step". After the first attempts gained little, they asked:
"Would it be worth it to just refactor the entire CPHF equation code? Seemed like it was a struggle to get
better performance from it although there should have been headroom". My answer: not the solver, about 1%
of the time; the J/K of many dms is where the headroom is. The user: "Go for it".
- **Where the time went** (caffeine/def2-SVP B3LYP Hessian, PGO, 10 threads, max_memory 4000, py-spy):
  - the CPHF solve, 57%:
    - its response J/K 31%, of which the contraction kernels 25%;
    - the XC response 26% (It13's factored path);
    - the Krylov solver itself about 1%;
  - `_partial_hess_2e` 18%, `make_h1` 15%.
  - The CPHF made 8 J/K calls of up to 72 dms: 34 s of 111 s.
  - caffeine/def2-TZVP (494 AOs, an 8.2 GB partial cache), one J/K call: 1 dm 4.0 s, 8 dms 7.8 s,
    72 dms 79 s, mostly contraction.
- **Why.**
  - The one-pass kernels add a few products per dm and quartet to rows of D and VK spread over the
    arrays: each thread's private sums are 2 nao^2 npad doubles, 281 MB per thread at TZVP with 72 dms.
  - That runs at about 8 GFLOP/s per core.
  - J and K each cost about 0.27 s per dm up to 8 dms and 0.6 s from 24 (TZVP, 10 threads).
- **Tried, dropped:**
  - Each thread its own share of the dms, the integrals shared through a buffer:
    - TZVP 82.7 -> 74.3 s, SVP 4.56 -> 4.05 s;
    - a thread's private copy of its dms loses the L3 sharing of the dms that one pass has.
  - Output-stationary kernels, each output element's sum in registers and added once per quartet:
    - exact, but 0.75-1.14x with compile-time shell sizes, 0.4-0.9x without;
    - the kernels are bound by their accesses to D and VK, not by chains of dependent additions.
  - Groups of 8 dms over integrals computed once:
    - 1.35x in isolation (TZVP 82 -> 61 s), 1.00-1.10x on the Hessians' J/K
      (`results/20261106-193122_dm-partition.json`);
    - partly because its larger memory estimate split the 72 dms into two batches;
    - superseded by the products below and removed.
- **Kept: `_direct_jk_gemm`** (PySCF `perf` 0bc4500ff; `_vhf.JK_GEMM`, bench knob `BENCH_JK_GEMM`), for
  13 dms or more.
  - Blocks: the shells are cut into blocks of contiguous shells of about 16 AOs.
  - Each s8-unique quartet of blocks (IJ|KL) is evaluated whole into a dense E[(i,j),(k,l)]:
    - its shell quartets are those of the one-pass loop, each once, with the same screening, batched
      through int2e_sph_batch;
    - the copies that I == J, K == L or (IJ) == (KL) bring are filled in, the rest stays 0.
  - J and K are dgemm products with all dms at once:
    - VJ[I,J] += cJ E Ds[K,L], VJ[K,L] += cJ E^T Ds[I,J];
    - the four exchange terms from E in two orders, X1[(i,l),(j,k)] and X2[(i,k),(j,l)];
    - with non-symmetric dms (TDA, UHF TDDFT), the other four with D^T into sums of VK^T.
  - The weights are the quartet kernels', at the block level.
  - The dms and the sums are kept by block pairs, shared by all threads, one lock per block:
    - no thread holds sums of its own, which matters for many-core nodes;
    - `_direct_s8_batch` sizes its batches by the engine's footprint: 4 nao^2 npad doubles, 6 for
      non-symmetric K.
  - Blocks of 8, 12, 24 and 32 AOs were slower than 16 on SVP and TZVP.
  - The products are about 58% of its time, the integrals 17%.
  - About 42-47% of the products' work multiplies integrals; the rest is the zeros of screened quartets
    inside blocks.
- **Exactness.**
  - Against one pass, caffeine with 72 dms: J within 1e-14 and K within 4e-15, relative to the largest
    element.
  - New test: `test_vhf.py test_direct_jk_gemm`. It compares against one pass with the semi-direct cache
    holding all, part and none of the quartets, hermi 0 and 1, J and K alone.
  - `validate.py` full: all pass, Hessians within 7.7e-11, excitations 8.1e-14.
  - PySCF's tests: 1054 passed, the same 7 known failures.
- **Speed.**
  - One call, 72 dms, 10 threads, no cache: caffeine/def2-SVP 5.8 -> 2.2 s, def2-TZVP 85.5 -> 28.6 s.
  - caffeine/def2-TZVP B3LYP Hessian, 10 threads, max_memory 20000: 1178 -> 816 s (1.44x); its CPHF J/K
    577.5 -> 218.4 s (2.64x); Hessian within 2.1e-13. benzene/def2-TZVP keeps its ERIs in memory, so it
    does not use the direct J/K (32 s either way).
  - `bench/results/20261106-203002_jk-gemm.json`: laptop, 10 threads, ABBA x 2, max_memory 4000. Hessians
    within 1.0e-10 of one pass (one pass against itself: up to 3.0e-11), excitations within 7.1e-14.

| case | one pass (s) | engine (s) | total | J/K |
|---|---|---|---|---|
| hess_caffeine_svp_b3lyp | 118.5 | 99.7 | 1.19x | 69.9 -> 52.1 (1.34x) |
| hess_caffeine+_svp_b3lyp_uks | 225.3 | 176.1 | 1.28x | 119.2 -> 70.5 (1.69x) |
| tda_caffeine+_svp_b3lyp_uks | 46.9 | 39.2 | 1.20x | 25.1 -> 17.2 (1.46x) |
| tda_caffeine_svp_b3lyp | 26.0 | 25.8 | 1.01x | under 13 dms a call: one pass |
| hess_benzene_svp_b3lyp | 6.3 | 6.2 | 1.02x | 1.9 -> 1.8 |
| hess_benzene+_svp_b3lyp_uks | 12.3 | 12.7 | 0.97x | 2.5 -> 2.5 |

- **node2** (AVX2, 12 threads), a fresh clone of workspace 0988591 (pins PySCF perf-2026-11-06c, qcint
  perf-2026-11-06c):
  - `cpu_setup.sh --with-base` took 32 min, of which qcint 19 min; water B3LYP -8.5e-14 Eh from the
    laptop; `kernels.py` 77/77.
  - `test_vhf.py`: 17 passed.
  - caffeine/def2-SVP, 72 dms: 14.0 -> 4.25 s (3.3x), J within 5.4e-15, K 2.4e-15.
- **Left in the engine:**
  - It does not read the semi-direct cache: the CPHF calls compute again what the SCF stored (about
    0.6 s a call on caffeine/def2-SVP).
  - Products on the zeros of screened quartets inside blocks: up to about 2x on its products.
- **The rest of the caffeine/def2-TZVP Hessian** (598 of 816 s): `_partial_hess_2e` and `make_h1`
  (batched above), the XC Hessian parts and the CPHF XC response.

## The Hessian's two-electron blocks as energies (lib/vhf/hess_jk_energy.c), 2026-11-06
The user: "let's focus on the other areas for speeding up the hessian calculation". `_partial_hess_2e` was
the largest remaining part of the caffeine/def2-TZVP B3LYP Hessian (355 of 831 s under py-spy).
- **What it computed.** For each atom A, int2e_ip1ip2 over (A, all, atoms before A, all) in s1 and
  int2e_ipvip1 over (A, atoms before A, all, all) in s2kl, contracted into J/K-like matrices that are then
  dotted with the densities. Each cross-atom centre pair of each s8 orbit is evaluated once; the diagonal
  blocks come from translational invariance.
- **No screening upstream.** `_make_vhfopt` checks `hasattr(libcvhf, 'int2e_ip1ip2')`, but libcint's
  functions are `int2e_ip1ip2_sph` etc., so it always returns None and `direct_bindm` runs
  `CVHFnoscreen`: every quartet of the Hessian's 2e part is computed, in upstream PySCF too. The screening
  it sets up could not work anyway:
  - `direct_bindm`'s `set_dm` would get the scripts' sliced dms (`dm[:,p0:p1]` next to `dm`), which
    numpy cannot stack;
  - `CVHFnr_int2e_pppp_q_cond` read the (ab|ab) components of `int2e_ipip1ipip2`/`int2e_ipvip1ipvip2` at
    17*b of a 16x16 layout. Those integrals have 81 components, (ab|ab) at 10*b: it read past the
    integrals into uninitialised memory, and gave run-to-run different, sometimes NaN, bounds. Fixed
    (81 components, offsets 10*b); checked on benzene/def2-TZVP: deterministic, finite, and
    |integral| / bound <= 0.98-0.996 over 600 random quartets of each kind.
- **The driver** (PySCF `perf` 388dfe257): `hessian.rhf.hess_jk_energy`, taken by `_partial_hess_2e` when
  atmlst holds every atom (`hessian_rhf_jk_energy`, default on).
  - E = 1/2 (pq|rs) Gamma_pqrs with Gamma_J = Dj Dj and Gamma_K = kfac/4 sum_n (Dk_pr Dk_qs + Dk_ps Dk_qr):
    d2E/dR_A dR_B = sum over s8 orbits of deg/2 sum_{c on A, d on B} M_cd, M_cd the 3x3 contraction of the
    second-derivative integral of centres c, d with Gamma. Both Gammas share the integrals; no J/K
    matrices.
  - M_cd is int2e_ipvip1 (c, d of one pair) or int2e_ip1ip2 (of different pairs), M_cc int2e_ipip1, of
    the shells permuted to put c first.
  - Translational invariance per orbit (sum_d M_cd = 0): the atom with the most centres needs no
    integrals of its own. 3+1 centres on two atoms take 1 evaluation (int2e_ipip1) instead of 3, 2+2 3
    instead of 4, 2+1+1 3 instead of 5, four atoms 6 either way, one atom none. The three integrals cost
    the same per quartet (bbench, caffeine/def2-TZVP: 638, 646, 647 s projected one core).
  - Screening: an orbit is computed when max(qd qd, qv q) > tol and that times the largest density
    factor > tol, which every test of `CVHFip1ip2_prescreen` and `CVHFipvip1_prescreen` of its integrals
    implies; a computed orbit takes all its integrals. tol = 1e-14 (`hessian_rhf_jk_energy_tol`, the
    `_VHFOpt` default of `_make_vhfopt`); 0 computes every orbit whose Gammas are not zero.
  - qcint's batched Hessian integrals, orbits queued by their 4 shell types and the pattern of their
    atoms: 97% of the orbits on caffeine/def2-SVP.
  - Chunks of the sorted pair list, each summed into its own buffer (the A > B blocks), the buffers in
    order: independent of the thread count and the schedule. The buffers take at most 128 MB.
  - Per orbit, the accumulation goes through a list of centre pairs per pattern (at most 16 block
    additions and the invariance's row sums): the first version's generic loops were 20% of the time.
  - Chunks: 256 (at most); 1024 left more part-filled batches at the chunk ends (caffeine/def2-TZVP,
    10 threads: 157.5 -> 149.8 s).
- **Exactness.**
  - Against the loops of `_partial_hess_2e` (JK_ENERGY off), caffeine/def2-TZVP B3LYP density: ej within
    1.7e-13 (|ej| up to 57), ek within 6.5e-14 (|ek| 2.1); the screening (1e-14 against 0) moves them by
    4.3e-14 and 6.4e-14.
  - benzene/def2-SVP: RHF, UHF-like spin densities, J alone, K alone (long and short range), with and
    without the screening: within 1.3e-13 (|ej| 22.5); the screening moves ek by up to 1.5e-13 (spin
    densities with a large random part).
  - New test `hessian/test/test_rhf.py test_hess_jk_energy`: water + He (cc-pVDZ, general contraction:
    the unbatched path), spherical and cartesian, all the cases above, atoms out of order, screening on
    and off, batching on and off, against the loops to 1e-11.
- **Speed** (caffeine/def2-TZVP, B3LYP density, 10 threads, PGO):
  - the loops: 351 s, 1019M integral evaluations;
  - the energy driver without screening: 328 s, 922M (translational invariance alone: 1.10x fewer);
  - with the screening: 150 s (2.34x), 588M evaluations (65% of the orbits computed); py-spy: the
    integrals 81%, the 9-component dot products with the Gammas 7%, Gamma and its permutations 4%, the
    unbatched quartets (2% of the orbits) 5%.
  - benzene/def2-SVP: 1.33 -> 0.76 s.
  - `bench/results/20261106-235743_hess-jk-energy.json`: laptop, PGO, 10 threads, ABBA x 2. Hessians within
    7.7e-12 of the loops (the loops against themselves: up to 3.0e-12). `validate.py` full and hessian
    suites: all pass, Hessians within 1.0e-11 (6.4e-11 for density fitting, which does not use this code).
    PySCF's tests: 1055 passed, the same 7 known failures.

| case | loops (s) | energy driver (s) | total | J/K |
|---|---|---|---|---|
| hess_benzene_svp_b3lyp | 6.38 | 5.77 | 1.11x | 1.94 -> 0.78 |
| hess_benzene_svp_pbe | 5.84 | 5.65 | 1.03x | 1.50 -> 0.58 |
| hess_benzene+_svp_b3lyp_uks | 12.30 | 11.82 | 1.04x | 2.56 -> 1.27 |
| hess_caffeine_svp_b3lyp | 97.58 | 84.90 | 1.15x | 50.2 -> 25.7 |
| hess_caffeine+_svp_b3lyp_uks | 174.93 | 157.12 | 1.11x | 69.8 -> 39.4 |
| hess_caffeine_svp_r2scan | 129.92 | 119.18 | 1.09x | 39.5 -> 17.3 |

  The J/K column is the bench's category, which counts qcint's kernels; the driver's own loops land in
  "other" (about 11 s of the caffeine cases' growth there).
- **Left in this part:** the integrals are 81% of it. Contracting with Gamma inside qcint's kernels would
  skip their cart -> sph step, the output transpose and the dot products here (about 17%). `make_h1`'s
  int2e_ip1 J/K are not screened either (no vhfopt).

## make_h1's J/K screened; the Hessian integrals contracted inside qcint's kernels, 2026-11-07
- **make_h1** (PySCF `perf` b648b82bf). The derivative Fock matrices' int2e_ip1 J/K went through
  `_get_jk` without a vhfopt in all four make_h1 (rhf, rks, uhf, uks): every quartet was computed, upstream
  too.
  - `_make_h1_vhfopt`: `CVHFip1_h1_{jk,j,k}_prescreen` at 1e-14 (`hessian_rhf_make_h1_tol`, 0 for none;
    the `_VHFOpt` default), with the gradients' pair bounds (`grad.rhf._calc_q_cond`: int2e_ip1ip2, int2e).
  - dm_cond from the full densities, set on the vhfopt directly (no dmcondname, so `direct_bindm`'s
    `set_dm` leaves it): the scripts take slices (`dm0[:,p0:p1]`), which `set_dm` cannot stack.
  - The prescreens test every density element the four scripts read through s2kl: dm[j,i] and 2 dm[l,k]
    (J), dm[l,i], dm[k,i], dm[j,k], dm[j,l] (K). Range-separated K: bounds of its own, made within
    `with_range_coulomb`.
  - caffeine/def2-TZVP, RHF make_h1, 10 threads: 105.5 -> 67.9 s (1.55x); h1 within 6.6e-13 of the
    unscreened (|h1| 3.2; benzene/def2-SVP: 1.65e-13 against a run-to-run 2.0e-14).
  - New test `test_rhf.py test_make_h1_screening` (RHF, B3LYP, PBE, wB97X, UHF, UKS B3LYP and CAM-B3LYP
    against the unscreened, 1e-11).
- **qcint** (`perf` cd07a5d): `int2e_ipip1/ipvip1/ip1ip2_sph_batch_dot`, the batched Hessian integrals
  contracted with 1 or 2 arrays of each quartet: out[q][n][9] = sum_f int[q][c][f] gam[q][n][f].
  - After the primitive loop, gam goes to the cartesian functions by the transpose of the cart -> sph
    step (once per quartet and array, in the then free space of the 2D integrals: at most 0.6 of it, for
    ipip1 (ff|ff)), and is contracted with the cartesian integrals in one pass with the 9 x ng sums in
    registers. No component goes to spherical and no block is written.
  - Tried first: adding each root sum to the 9 x ng sums inside the primitive loop. Slower for s and p:
    the sums left the registers (py-spy on caffeine/def2-TZVP: the l0 and l1 kernels 13-60% slower, the l2
    and l3 ones 13-26% faster; 158.7 -> 143.0 s overall). Then the contraction after the loop over each
    array in turn re-read the (ff|ff) block (5.8 MB with 8 lanes) per array: 128.8 s; one pass: 124.7 s.
  - bbench `--dot` (new): against int2e_*_sph contracted in bbench, with random arrays; also in the PGO
    training.
  - Exactness: within 1.1e-13 on every benzene quartet (values up to 48), 1.8e-12 on sampled
    caffeine/def2-TZVP (values up to 455), long range 1.7e-15 (water). qcint's own functions bitwise
    identical to the previous PGO build (4c59d1e): int2e_sph on all 113M caffeine/def2-TZVP quartets,
    int2e_ip1 and the three Hessian integrals on every benzene quartet. `kernels.py` 77/77.
  - One core, PGO, caffeine/def2-TZVP sampled: the batched kernels alone 553-563 s projected, the _dot
    kernels 478-483 s (1.16x), and they spare the contraction in PySCF as well.
- **PySCF** (`perf` a3ef1e042): `hess_jk_energy` contracts through the _dot kernels when qcint has them
  (`hessian_rhf_jk_energy_dot`, default on); off, the batched integrals are contracted in PySCF as before.
  - caffeine/def2-TZVP, `hess_jk_energy` at 10 threads, plain qcint build, py-spy at 25 Hz: 158.7 s with
    the contraction in PySCF, 124.7 s through the _dot kernels; unprofiled in one process 167.8 -> 134.1 s
    (the laptop's timings drift by 5-15% between runs). Within 2.1e-14 of each other, 1.85e-13 of the
    loops.
- **Exactness.** `validate.py` full and hessian suites with the new PGO build: all pass, Hessians within
  4.3e-12 to 6.4e-11. PySCF's tests: 1056 passed, the same 7 known failures.
- **Bench** (`bench/results/20261107-020326_hess-h1-dot.json`: laptop, PGO, 10 threads, ABC CBA): base =
  make_h1 unscreened and the contraction in PySCF, h1 = make_h1 screened, dot = both. Hessians within
  8.1e-11 of base (base against itself: up to 6.4e-11). At def2-SVP the XC kernels are about half of these
  Hessians, and the two-electron parts left are small:

| case | base (s) | h1 (s) | dot (s) | total |
|---|---|---|---|---|
| hess_benzene_svp_b3lyp | 6.09 | 5.90 | 5.81 | 1.05x |
| hess_benzene_svp_pbe | 5.66 | 5.63 | 5.58 | 1.02x |
| hess_benzene+_svp_b3lyp_uks | 12.40 | 12.42 | 12.31 | 1.01x |
| hess_caffeine_svp_b3lyp | 86.82 | 82.40 | 80.99 | 1.07x |
| hess_caffeine+_svp_b3lyp_uks | 159.04 | 154.59 | 153.41 | 1.04x |
| hess_caffeine_svp_r2scan | 120.01 | 116.30 | 115.17 | 1.04x |

- **caffeine/def2-TZVP B3LYP Hessian** (10 threads, max_memory 20000, one process): 817.5 s with
  hess_jk_energy, the make_h1 screening and the _dot kernels off, 538.4 s with them (1.52x; 1178 s before
  the CPHF engine). Hessian within 5.3e-13 (max|h| 0.93).
  - RKS partial hessian 374.9 -> 145.6 s (its two-electron part about 115 s);
  - CPHF 250.4 -> 252.4 s, of which J/K 216.8 -> 218.4 s;
  - make_h1 and the rest 192.2 -> 140.4 s.
- **Left:** the CPHF J/K (40% now; the engine's products on the zeros of screened quartets inside blocks),
  the Hessian integrals (about 21%), make_h1's J/K (about 13%; no translational invariance or s8 there).

## Pushed perf-2026-11-07; the fresh clone on node2 found an overflow, fixed in perf-2026-11-07b, 2026-11-07
The user: "We can leave the screening in as PySCF was supposed to have that working previously. Go ahead and push
this and run the fresh clone check."
- **Pushed:** PySCF `perf` a3ef1e042 + tag perf-2026-11-07, qcint `perf` cd07a5d + tag perf-2026-11-07,
  workspace dbd9ad5 (pins, README). The screening of the Hessian's two-electron parts stays on by default.
- **node2** (AVX2, 4 lanes, 6 threads), a fresh clone of dbd9ad5:
  - `cpu_setup.sh --with-base`: 42 min (32 before; qcint PGO 1328 s against 896 s, the _dot kernels doubling
    the Hessian instances and their training); water B3LYP -8.5e-14 Eh from the laptop; `kernels.py` 77/77.
  - bbench `--dot` with 4 lanes, every benzene quartet: ipip1 within 3.9e-14, ipvip1 3.9e-14, ip1ip2 5.9e-15,
    long range (water) 1.1e-15.
  - caffeine/def2-SVP, 6 threads: `_partial_hess_2e` 44.33 s (loops) -> 20.49 s (energy driver, contracted in
    PySCF) -> 19.07 s (_dot kernels), 2.32x, within 4.6e-13; RKS B3LYP `make_h1` 32.59 -> 24.81 s (1.31x),
    within 2.6e-13.
  - `test_rhf.py test_finite_diff_rhf_hess1` aborted every time (3/3; passes with one thread): "double free or
    corruption" in a free() of `CVHFnrs8_incore_vk_batch` (the CPHF's in-memory K). Off with
    `hessian_rhf_jk_energy` or `_dot` or `scf_vhf_int2e_batch`, not with `make_h1_tol = 0`.
- **The bug** (PySCF `perf` 43838f635, tag perf-2026-11-07b): hess_jk_energy's batch output buffer held
  lanes x 9 x di^4 doubles, the integral blocks; the _dot kernels write 9 x nh sums per quartet, more than
  that when every shell is an s shell (H4/6-31G, di = 1) and both Gammas are contracted. The laptop's heap
  took the overflow silently. Found by AddressSanitizer on an AVX2 (4-lane) qcint build
  (`build_cint.sh qcint asan-avx2 "-DCMAKE_C_FLAGS=-mno-avx512f -fsanitize=address -fno-omit-frame-pointer -g"`,
  then `LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS=detect_leaks=0:log_path=...` and pytest
  `-s`, as pytest's capture swallows ASan's report).
  - After the fix, under ASan: hessian test_rhf/uhf/rks/uks, grad test_rhf/uhf, scf test_vhf: 78 passed, no
    report; test_hess_jk_energy's spherical half within 1.3e-14 of the loops, no report.
  - Its cartesian half hits an over-read in qcint's own `d_ket_cart2spheric` (cart2sph.c:3754, upstream code,
    last changed upstream in e231f46): the SIMD remainder loads 4 doubles where fewer are left, up to 24 bytes
    past the array (through `mole.py` `cart2sph`). The extra values are discarded; a fault would need the
    array to end at a page boundary. Not fixed here.
- **node2 with perf-2026-11-07b** (the clone updated, `cpu_setup.sh` rebuilt PySCF alone in 1 min): water and
  `kernels.py` as above; hessian test_rhf/uhf/rks/uks 29 passed; caffeine/def2-SVP timings as above; PySCF's
  tests (`run_tests.sh`, after pytest-xdist and pytest-timeout in that venv: cpu_setup.sh installs pytest
  alone) 1056 passed, the same 7 known failures as the laptop. The clone was removed afterwards.

## The refactor: one template for qcint's derivative integrals, one loop for the energy drivers, 2026-11-07
The user asked whether to refactor the J/K integral code given the bespoke drivers; my answer: not the
matrix-building J/K (SCF, CPHF, in-core; upstream interfaces, no speed to gain), but the two energy drivers
and qcint's two templates, together with their next uses. The user: "Please start it".
- **qcint** (`perf` e25a81f): `int2e_ip1_sph_batch` now comes from the Hessians' template
  (`cint2e_batch_hess.h`, a fourth kind HESS_IP1: g0 and g2 = nabla_I g0, 3 components), so one template
  gives every batched derivative integral, as blocks or `_dot`. New `int2e_ip1_sph_batch_dot`
  (out[(q*ng+n)*3+c]); its cache is `int2e_sph_batch_cache_size()`, which the template's ip1 layout fits
  (70124 of 72420 bvec at (ff|ff)).
  - `cint2e_batch.h`'s template keeps its (no longer instantiated) ip1 branch: removing it made GCC contract
    other multiply-adds into FMAs, and 501 of benzene/def2-SVP's 969 int2e classes changed in their last
    bits (found with a new ctypes check that runs two builds' batched kernels side by side). With the branch
    in place `int2e_sph_batch` is bitwise identical on every class.
  - Exactness: `int2e_ip1_sph_batch` against `int2e_ip1_sph`, every quartet: water 3.6e-15, benzene 4.7e-15
    (as before); its `_dot` 3.6e-15 and 4.3e-15, long range 1.6e-15; the Hessian kinds unchanged.
  - Speed (one core, caffeine/def2-TZVP sampled, plain builds): `int2e_ip1_sph_batch` 224.9 s (the old
    kernel 223.9 s); `_dot` 216.1 s.
  - bbench `--dot` takes ip1 too (its result arrays sized by quartets x components: nf < 3 for (ss|ss));
    the PGO training runs the ip1 `_dot` kernels.
- **PySCF** (`perf` c9bfd2ea5): `lib/vhf/orbit_engine.c` (`CVHFoe_run`), the loop both energy drivers had
  copied: the pair list sorted by a bound with the break at the first failing partner, the fixed chunk
  partition summed in order, the shell types and the per-thread hash tables of queues for the batched
  integrals. A client gives its pair bounds, thresholds and callbacks (plan an orbit: screening, its
  integrals, its queue key; evaluate one orbit or a full queue) and keeps its own arithmetic. Shared:
  `CVHFoe_permute` (the shell orders of an orbit's integrals). grad_jk_energy.c 717 -> 616 lines,
  hess_jk_energy.c 1069 -> 937, the engine 373 with its header.
  - Both drivers give bitwise the same results as before (jk_energy_per_atom: caffeine/def2-SVP RHF, spin
    densities, long and short range; hess_jk_energy: benzene/def2-SVP RHF, spin densities, J alone, long-
    and short-range K; water + He cartesian with a general contraction).
- **PySCF** (`perf` 28d60a20e): `jk_energy_per_atom` through `int2e_ip1_sph_batch_dot`
  (`grad_rhf_jk_energy_dot`, default on): each centre's Gamma permuted into its integral's order and
  contracted in qcint. caffeine/def2-TZVP, 10 threads: 27.9 -> 25.8 s (1.08x); def2-SVP unchanged (2.1 s);
  within 1.8e-15. `test_jk_energy_per_atom_int2e_batch` covers both paths.
- **Stage 3 skipped** (the user: "Let's skip it"): make_h1 with translational invariance, re-estimated at
  about 11 s of the 538 s TZVP Hessian (its J/K 73 s, integrals about half, cut about 1.4x), needing a new
  accumulation for natm x 3 x nao^2 outputs shared by the threads.
- **The PGO build** (qcint e25a81f, the training with the ip1 _dot kernels too) is not bitwise identical to
  the previous PGO build (cd07a5d), although the plain builds are for int2e: every function moved at the
  rounding level, the unbatched ones included, whose source did not change (int2e_sph within 1.2e-15 on all
  113M caffeine/def2-TZVP quartets, values up to 8; ipip1 1.1e-14 on benzene, values up to 53; the batched
  int2e 3.3e-16). Probably GCC's profile-based hotness thresholds, which follow the whole training: the
  previous two PGO builds (cd07a5d, 4c59d1e) were bitwise identical on int2e.
- **Exactness.** `kernels.py` 77/77. `validate.py` full and hessian suites: all pass, gradients within
  3.6e-14 (benzene) and 1.5e-12 (caffeine), Hessians 7.1e-12 to 6.4e-11; the tight SCFs' cycle counts
  scatter as before. PySCF's tests: 1056 passed, the same 7 known failures. AddressSanitizer on a 4-lane
  qcint (grad, hessian and scf J/K tests): 104 passed, no report.
- **Bench** (`bench/results/20261107-100854_grad-jk-dot.json`: laptop, PGO, 10 threads, ABBA x 2; new
  timing-only case `grad_caffeine_tzvp_b3lyp`), the gradient step with the _dot kernels off and on; gradients
  within 2.3e-12 (off against off: up to 1.8e-12):

| case | off (s) | on (s) | speedup |
|---|---|---|---|
| grad_benzene_svp_b3lyp | 0.44 | 0.46 | 0.95x (noise; ran at lower clocks) |
| grad_caffeine_svp_b3lyp | 3.18 | 3.06 | 1.04x |
| grad_caffeine+_svp_b3lyp_uks | 3.61 | 3.48 | 1.04x |
| grad_caffeine_tzvp_b3lyp | 37.56 | 33.47 | 1.12x |

## RIJCOSX in PySCF: screening K by shell pair, a J-engine for RI-J, ORCA's grid scheme, 2026-11-07
The user relaxed the exact-only rule: "Let's relax the exact requirement now ... I've always liked the RIJCOSX
approach in orca. Could we explore that here?", then "Could you monitor memory and disk usage as well? Something
I've noticed about orca is it doesn't need much memory whereas PySCFs RIJ seems very memory hungry".
- **Where:** pyscf branch `rijcosx` (worktree `pyscf-rx`): `sgx-exp` merged (It25/It26's AO and projection
  caches, the kernel flags), then fca1515c6 (J-engine), c07235166 (SGX) and 0e7843bfd (the final step keeps the SCF's
  SGX data for the gradients). Not in `perf` yet.
- **Tools** (workspace): `bench/diag/rijcosx.py` (one SCF, exact / RIJK / RIJCOSX-like, J/K/XC split, a line per K
  build, peak RSS and scratch-disk bytes sampled every 0.1 s in a per-run `PYSCF_TMPDIR` on the disk, since `/tmp`
  is a tmpfs here); `cosx_oracle.py` (the exact energy share of every shell pair and 224-point batch, against what a
  screening keeps); `sgx_pairtol.py`; `rij_engine_check.py`; `sgx_profile.py` gains `--level --etol --vtol --incr
  --auxbasis`. ORCA references: `bench/orca/rx/`, `scripts/node2_orca_rx.sh`, summary
  `bench/results/node2/node2_orca_rx.log`. Laptop logs: `bench/results/rijcosx/`.
- **Laptop runs:** 10 threads, the power-saver profile with its 2.0 GHz clamp, PySCF's default `max_memory` (4000
  MB), B3LYP/def2-TZVP. "RIJCOSX" is SGX with dfj (`pjs`), def2-universal-jfit (ORCA's def2/J), grid levels 0 -> 1
  and the energy once more on level 2, as ORCA does.

### What the first measurements showed
- **SGX's default screening does nothing.** "auto" sets sgx_tol_energy to direct_scf_tol (1e-13) and splits it over
  the batches and the shells of each batch (~1e-18 a shell). Caffeine, level 1: SGX computes 98.5% of the integrals of
  no screening at all, 92% in an incremental build (|dD| 4e-4); sgx_tol_energy from 1e-13 to 1e-7 changes neither the
  energy (10 digits) nor the time. So incremental builds cost as much as full ones (3.2 s each at level 2).
- **What a given accuracy needs** (`cosx_oracle.py`, caffeine, level 1, 2.7 M pair-batches): to keep the dropped
  share of the exchange energy under 1e-6 Eh, 56% of the pair-batches in a full build and 19% in an incremental one.
- **SGX's own bound is a usable estimate:** rbar_IJ m_bI m_bJ times the batch norms of F exceeds the true share by a
  median 26x and never falls under it in a full build (0.53 at worst, incremental). A fixed threshold on it at 1e-10
  keeps 62% / 20% with errors of 8e-9 / 1e-8 Eh.
- **RI-J was the next cost:** SGX's dfj J (`df_jk.get_j`, integral direct, two passes over int3c2e) took 0.4 s a build
  on caffeine, ~19 ns per (ab|P), 90% of it in qcint's scalar int3c2e. ORCA's Split-RI-J is ~3x faster per build
  (scaled from node2).
- **Memory:** RIJK (`mf.density_fit()`) stores the 3-index tensor, on disk past `max_memory`: 6.9 GB for C20H42 and
  15.0 GB for C26H54 at the default 4000 MB. SGX's RIJCOSX path stores nothing: peak RSS 2-3.2 GB, no disk. ORCA
  peaks at 0.6-1.0 GB here.

### Changes
1. **`sgx_tol_pair`** (`SGXscreen_grid_pair`, `SGXscreen_shells_pair` in `nr_sgx_direct.c`; SGXData `pair_tol`):
   each shell pair on each batch is evaluated if rbar_IJ m_bI m_bJ (f_I F_J + F_I f_J) > sgx_tol_pair, f the weighted
   norms of the F contracted, F those of the full density (equal in a full build); with sgx_tol_potential, also if
   its potential estimate exceeds that. sgx_tol_energy/potential still screen the dense D X and X G steps.
   - Caffeine level 1, 1e-9: a full build 1.14 -> 0.86 s (69% of the integrals), the exchange energy within 1.2e-8;
     an incremental build 1.17 -> 0.41 s (26%), within 3e-10.
   - **A fixed threshold broke the SCF:** caffeine 50 cycles (4e-9) against 15, n-C26H54 37 (1e-9) against 10. The
     dropped parts of an incremental build move E at first order (tr D dK) while dE falls at second order, so dE
     never got under conv_tol: every full rebuild landed on the same energy (to 1e-12), the incremental ones scattered
     by ~5e-9.
   - **Fix:** an incremental build's threshold is min(sgx_tol_pair, 0.01 |dE of the last cycle|) (`_SGXHF` keeps
     the last two energies). Caffeine 15 cycles at 4e-9, C20H42 and C26H54 11 at 1e-9 (10 unscreened).
2. **The J-engine** (`lib/vhf/rij_engine.c`, `df_jk.get_j`, `df_jk.RIJ_ENGINE`): McMurchie-Davidson RI-J. Each AO
   pair's density goes to Hermite form once (E coefficients) and meets every auxiliary primitive through the R
   integrals only; pass 2 builds the pair's J block from a Hermite potential. Spherical auxiliary functions take the
   Hermite components of their own order only (Hobson). Auxiliary primitives of one l go 8 at a time through a
   tabulated Boys function (Taylor from a 1/16 grid, downward recursion), the R recursion and the contractions.
   - Same J as int3c2e to 1e-13 relative (cc-pVDZ's general contractions, def2-QZVP's g functions, CH3Br, caffeine).
   - 2.1x a build on caffeine/def2-TZVP; in the SCF 3.3x (J 7.9 -> 2.4 s), its density screening taking the small
     incremental dD better. Stores nothing.
   - mol.cart (Cartesian auxiliary functions) and l > 6 (AO) or > 8 (auxiliary) keep the int3c2e path. Its data follow
     auxmol, the molecule and its geometry (a first version keyed on the Mole's identity reused the old geometry in
     `as_scanner`; the SGX gradient tests caught it).
3. **`grids_level_energy`** (the converged density's energy once more on a larger SGX grid, orbitals unchanged:
   ORCA's GRIDX3 step) and **`rijcosx(mf)` / `mf.RIJCOSX()`** (def2-universal-jfit, levels 0 -> 1, the energy on level
   2, sgx_tol_pair 1e-9).
4. **`get_jk_favorj`** rounds its grid blocks to SGX_BLKSIZE (it failed `assert i0 % BLKSIZE == 0` when the grid took
   more than one block: the C26H54/PCM crash of 2026-11-05); `get_jk_favork`'s memory estimate counts nao.
- **Tests:** `test_get_j_engine`, `test_sgx_tol_pair`, `test_rijcosx`, `test_rijcosx_grad`, `test_sgx_jk_blocks`;
  PySCF's suites (dft scf df grad hessian tdscf solvent sgx): 1060 passed, the 7 known failures (before
  test_rijcosx_grad; sgx and sgx/grad again after it: 13 passed).

### Laptop results (B3LYP/def2-TZVP, 10 threads, max_memory 4000 MB; total = SCF + the final level-2 step)
| system | exact | RIJK (`density_fit`) | RIJCOSX before | RIJCOSX now | now vs exact |
|---|---|---|---|---|---|
| caffeine (494 AOs) | 55.5 s | 16.0 s | 41.3 s | 28.6 s | 1.9x |
| n-C20H42 (872) | 157.6 s | 80.1 s, 6.9 GB on disk | 96.4 s | 67.7 s | 2.3x |
| n-C26H54 (1130) | 331.0 s | 183.8 s, 15.0 GB on disk | 164.2 s | 107.0 s | 3.1x |

- Errors against exact: RIJCOSX -0.56, -1.03, -1.33 mEh (ORCA's RIJCOSX against its exact: -0.56, -1.03, -1.33 with
  PCM); RIJK -0.14, -0.53, -0.68. The screened energies are within 1e-9 Eh of the unscreened ones.
- Peak RSS: exact 3.5-3.7 GB (its ERI cache), RIJK 2.3 GB plus the disk, RIJCOSX 2.0-3.2 GB and nothing on disk.
- "Before" = SGX's default screening and the int3c2e RI-J; "now" = J-engine and sgx_tol_pair 1e-9.
- **C26H54 now:** K in the SCF ~49 s, the final level-2 step 15 s, XC ~30 s (2.5 s a call: level 3 has ~900k points,
  and the fused XC's radial cache needs more than 25% of max_memory), J 8.5 s.
- The harness's first "XC and rest" column subtracted the final step's K too (fixed: it now times nr_rks and keeps
  the final K apart); the totals above are wall times.

### ORCA 6.1.1 on node2 (6 ranks, B3LYP/G def2-TZVP, gas phase, TightSCF, DefGrid2)
| system | RIJCOSX | of which COSX / Split-RI-J / XC / GRIDX3 step | exact | RIJCOSX vs exact | peak PSS |
|---|---|---|---|---|---|
| caffeine | 57 s, 14 cycles | 37.7 / 4.2 / 9.7 / 7.5 s | 211 s | 3.7x | 0.60 GB |
| n-C20H42 | 110 s, 12 cycles | 72.5 / 9.7 / 14.8 / 17.4 s | 526 s | 4.8x | 0.95 GB |

- GRIDX1/2/3: caffeine 14,827 / 32,471 / 71,266 points (PySCF's levels 0/1/2: 17,464 / 32,256 / 71,208);
  C20H42 32,286 / 66,432 / 143,960 (PySCF level 1: 79,312; level 2: 176,984).

### PySCF on node2 beside ORCA (6 threads, max_memory 4000 MB; `scripts/node2_rijcosx.sh`)
The `rijcosx` branch as a `git archive` built there as `pyscf-rx`; `bench/results/node2/node2_rijcosx.{log,jsonl}`.

| run | caffeine | of which J / K / XC / final step | n-C20H42 | of which J / K / XC / final step | peak memory |
|---|---|---|---|---|---|
| PySCF exact | 215.5 s | | 453.1 s | | 3.2-3.3 GB |
| PySCF RIJK | 35.0 s | | 122.1 s (6.9 GB on disk) | | 2.2-2.7 GB |
| PySCF RIJCOSX before | 71.9 s | 12.1 / 30.5 / 21.9 / 6.0 | 170.6 s | 27.3 / 81.6 / 35.2 / 22.2 | 2.0-3.1 GB |
| **PySCF RIJCOSX now** | **52.4 s** | 2.9 / 21.1 / 21.9 / 5.0 | **116.4 s** | 7.2 / 50.3 / 37.9 / 16.3 | 1.9-3.1 GB |
| ORCA RIJCOSX | 57 s | 4.2 / 37.7 / 9.7 / 7.5 | 110 s | 9.7 / 72.5 / 14.8 / 17.4 | 0.6-1.0 GB |
| ORCA exact | 211 s | | 526 s | | 0.5-0.7 GB |

- **PySCF's RIJCOSX is level with ORCA's:** 0.92x and 1.06x its time. Its J and K are faster (J 1.3-1.4x, K
  1.4-1.8x); its XC is 2.3-2.6x slower, on PySCF's level-3 grid of 2.5x DefGrid2's points (caffeine 294,800 against
  116,136; PySCF's level 1 has 97,000).
- Against each code's exact run: PySCF 4.1x and 3.9x, ORCA 3.7x and 4.8x.
- Same errors as ORCA's (-0.56, -1.03 mEh). PySCF's memory is 2-3x ORCA's (the SGX AO cache, the XC blocks).
- **RIJCOSX gradients** (laptop, caffeine/def2-TZVP; needed 0e7843bfd, the final step had dropped the SGX gradient's
  data): 10.9 s against the exact gradient's 29.9 s; max |dg| 1.1e-4, rms 3.2e-5 Eh/bohr.

### Left
- **XC:** the level-3 grid (a setting, not code) and the fused XC's radial cache, which needs more than 25% of
  max_memory at C26H54 (2.5 s a call without it on the laptop).
- **K:** the early incremental builds (large dD) save little; the per-integral cost (~2.4 ns, It25/It26); full
  rebuilds every 5 cycles (`rebuild_nsteps`).
- **Memory against ORCA's:** the SGX AO cache (nao x grid) and the final level-2 step.
- The final step recomputes J and XC too (generic get_veff); K alone would do.
- The J-engine for Cartesian auxiliary functions (mol.cart) and for the DF-J gradient (int3c2e_ip1/ip2 now).

### What is left on the speed side, measured (2026-11-07)
The user asked for the remaining speed items in detail. `bench/results/rijcosx/laptop_left_xc_grad.log`.
- **The split now** (node2, n-C20H42/def2-TZVP, 116 s; ORCA 110 s): K in the SCF 50.3 s (ORCA's COSX 72.5), XC 37.9
  (14.8), the final level-2 step 16.3 (17.4), J 7.2 (9.7), the rest ~4.7.
- **XC:** per point PySCF is ORCA's speed (node2, caffeine: 4.6 against 5.6 us per point and cycle); the gap is the
  grid. E_xc of the converged RIJCOSX density by PySCF grid level against level 8 (laptop, 10 threads):

| level | caffeine points | E_xc error | s per call | C20H42 points | E_xc error | s per call |
|---|---|---|---|---|---|---|
| 0 | 21,312 | -1.5e-2 | 0.06 | 47,760 | -2.8e-2 | 0.11 |
| 1 | 97,000 | +1.5e-5 | 0.21 | 207,424 | +5.3e-4 | 0.40 |
| 2 | 209,768 | +1.2e-5 | 0.41 | 449,784 | -1.3e-4 | 0.94 |
| 3 (default) | 294,800 | +1.2e-5 | 0.56 | 694,296 | -1.2e-4 | 1.50 (1.29 with the radial cache) |
| 4 | 553,968 | -1.3e-7 | 1.17 | 1,231,896 | +6.2e-6 | 2.77 (2.48) |

  - Levels 2 and 3 make the same error here; level 2 takes 30-35% fewer points and 37% less XC time. ORCA's DefGrid2
    has 116,136 and 252,602 points (between PySCF's levels 1 and 2).
  - The fused XC's radial cache (built when it fits 25% of the memory left) is worth 10-14% of the XC, not more.
- **K per build** (node2, C20H42): the level-0 phase 16.2 s (its incremental builds 64-81% of a full one: dD is still
  large), level 1 34.2 s; full builds (the first on each grid, the rebuild every 5 cycles) 34% of K; late
  incremental builds 33-46% of a full one, as the |dE|-scaled threshold tightens. The dense steps of every build
  (F = D X, K += X G) took 23% of K in the laptop profile, whatever dD.
- **The RIJCOSX gradient** (caffeine, laptop, 11.7 s under cProfile): K 7.5 s (64%), J 2.2 s (19%, int3c2e_ip1/ip2
  in df/grad), XC 1.8 s (15%). The gradient's K takes the pair screening with whatever threshold the SCF's last K
  build left in the C struct (a tightened one): accurate, but not set on purpose.

### def2-QZVP: n-C20H42 (2400 basis functions) against ORCA on node2, 2026-11-07
The user asked for the SCF cycle of one of the alkanes in def2-QZVP with RIJCOSX, then for ORCA on it. PySCF's
`perf` (laptop) and node2's `pyscf-rx` (the same SCF path), `bench/diag/rijcosx.py --pair-tol 1e-9` at the default
max_memory (`bench/results/rijcosx/laptop_qzvp.{log,jsonl}`, `bench/results/node2/node2_rijcosx_qzvp.{log,jsonl}`);
ORCA 6.1.1 `bench/orca/rx/c20h42_qzvp_rijcosx.inp` after PySCF had finished (`node2_orca_qzvp.log`).

| node2, 6 cores | PySCF RIJCOSX | ORCA RIJCOSX |
|---|---|---|
| total | 651 s (SCF 567 + final step 84) | 620 s |
| cycles, typical cycle | 11, ~40 s (35-61) | 12, ~33 s (24-63) |
| K / COSX | 291.5 s | 418.9 s |
| XC | 200.2 s (694,296 points, level 3) | 57.2 s (257,324, DefGrid2) |
| J / Split-RI-J | 33.2 s (1442 auxiliary functions) | 42.1 s (1442) |
| final step on the larger grid | 83.8 s (177,536 points) | 97.9 s (GRIDX3 154,128) |
| diagonalization, DIIS, the rest | ~42 s | ~21 s (SOSCF steps) |
| peak memory | 3.5 GB RSS, no disk | 3.3 GB PSS |
| energy | -787.8304080992 | -787.8304390662 |

- PySCF 5% slower: faster K (1.4x), J (1.3x) and final step, the XC 3.5x slower on 2.7x the points (per point
  within ~20%). With XC level 2 (as accurate as level 3 here) PySCF's XC would be ~125 s and the total ~575 s.
- The laptop (10 threads, 2 GHz clamp): 379 s (SCF 328 + 51), cycles 20-34 s; K 172, XC 108, J 22.5, the rest 24 s;
  the same energy to the last digit. def2-TZVP took 68 s there (5.6x for 2.75x the basis functions).
- The two codes' energies differ by 3.1e-5 Eh (def2-TZVP 3.5e-5), from their different XC grids.

### wB97M-V against B3LYP: n-C20H42/def2-TZVP, PySCF and ORCA on node2, 2026-11-07
The user: "How does the def2-TZVP basis with wB97M-V compare to B3LYP?". Three things showed up on the way.
- **VV10 dominates wB97M-V in PySCF.** PySCF (upstream too) evaluates it self-consistently on `nlcgrids`, whose
  default level is the XC grid's (3), and the double sum grows with the square of the points: caffeine/def2-SVP 82%
  of the SCF (7.4 s a cycle, against 7.3 s for both K parts).
- **Upstream bug, SGX's range-separated path** (pyscf `perf` baba529cb): `SGX.get_jk`'s omega branch called the RSH
  copy as `rsh_df.get_jk(dm, hermi, with_j, with_k, direct_scf_tol)`, putting with_j into vhfopt, with_k into with_j
  and direct_scf_tol into with_k, so every long-range K also made a long-range RI-J (int3c2e, discarded). The copy
  also never got the build's `_full_dm` and `_pair_tol_now`. Now by name, and the state is handed over for the call.
  `test_rijcosx_rsh`.
- **Incremental builds kept wB97M-V from converging** (24 cycles against exact's 9 on the laptop; 10 with full builds
  only, 10 without pair screening): the screened parts of incremental builds moved the energy by ~3e-8 Eh from cycle
  8 on, 100x caffeine's. pyscf `perf` 9ac0bc855: with sgx_tol_pair, every build is a full one once |dE| < 1000
  conv_tol. wB97M-V 10 cycles; B3LYP 10 (11 before).
- The harness's final step had a B3LYP-only formula (21 Eh off for range-separated hybrids); it now calls the
  library's `_sgx_energy_on_grid`, as mf.RIJCOSX() does.

| n-C20H42/def2-TZVP, node2, 6 cores | B3LYP | wB97M-V |
|---|---|---|
| PySCF RIJCOSX, VV10 on nlcgrids level 1 | 123 s (10 cycles) | 427 s (10 cycles) |
| PySCF RIJCOSX, PySCF's default VV10 grid | | 1204 s |
| ORCA RIJCOSX, its default (VV10 once after the SCF, 73,272 points) | 110 s | 242 s (11 cycles) |
| ORCA RIJCOSX, SCNL (VV10 in every cycle) | | 1206 s |
| ORCA exact | 526 s | 822 s |

- PySCF wB97M-V (VV10 level 1): K 122.7 s (the long-range K doubles B3LYP's 56.1), XC 62.3 (meta-GGA; B3LYP 35.4),
  VV10 171.4, J 7.0, the final step 58.1 (VV10 again in it). With the default VV10 grid VV10 takes 896.9 s.
- With VV10 in every cycle the two codes take the same time (1204 against 1206 s). ORCA's default adds VV10 once
  after the SCF, 1.6e-4 Eh from its SCNL energy; PySCF has no such option (it would bring PySCF to ~270 s).
- Energies (Eh): ORCA RIJCOSX -787.262218122162, SCNL -787.262379119423, exact -787.261186165096; PySCF RIJCOSX
  -787.2623445252 (VV10 level 1), -787.2623269883 (default VV10 grid), PySCF exact (VV10 level 1) -787.2612969440.
  RIJCOSX against each code's exact: PySCF -1.05, ORCA -1.03 mEh (B3LYP -1.03 both). PySCF against ORCA SCNL 3.5e-5.
- Peak memory: PySCF 5.4-5.5 GB (B3LYP 3.3; the RSH copy of SGX keeps its own grid, AO cache and projection on the same
  grid as the main object's), ORCA 0.95 GB.
- B3LYP with 9ac0bc855 on node2: n-C20H42 123 s against 116 s before, caffeine 57.5 s against 52.4 s (ORCA 57 s): the
  full builds at the end and the final step's J and XC.
- **Pushed** (the user: "Push them"): PySCF `perf` 0e7843bfd..9ac0bc855 + annotated tag perf-2026-11-07d; pins and
  README -> perf-2026-11-07d (qcint stays perf-2026-11-07c). Before it: sgx, df and solvent tests 260 passed, the 3
  known failures there; node2's runs above are of 9ac0bc855 itself.

### Sharing the grid between SGX's full-range object and its long-range copy: what is duplicated (2026-11-07)
For the user's question on sharing the grid. wB97X, n-C20H42/def2-TZVP RIJCOSX, laptop, cProfile:
- The RSH copy (`SGX._rsh_df`) builds its own grid, identical to the main object's (79,312 points at level 1), keeps
  its own AO values on it (0.55 GB each; 1.24 GB each at level 2 in the final step) and its own overlap-fitting
  projection (0.2 s, 6 MB each), and computes F = D X again (F is the same for both: 7.3 s for the two over the SCF).
  Per-operator: the integral bounds (1.6 s per grid level), the integrals and their contraction (C driver 59.6 s for
  the two), G and K += X G (5.9 s). The grids took 2.1 s, five builds.
- The long-range K as K_full - K_SR (erf + erfc = 1): the identity holds (1.6e-9 Eh), but a short-range (erfc) build
  takes 12.7 s against 3.2 s full range and 3.7 s long range (qcint's erfc integrals double the roots and have no
  optimizer), so no gain.

### Option A done: the long-range copy shares the grid data (pyscf `perf` 01f1d9b62, local), 2026-11-07
The user: "Let's do A first, then measure". `SGX.build` gives the RSH copies its grid and optimizer
(`_share_grid_of`); their SGXData take the overlap projection, the screening's AO norms and the AO values on the
grid from the full-range object's (`share`), and the full-range build keeps F = D X for the copy's build of the
same density (while a copy exists). The integral bounds stay each operator's. `test_rijcosx_rsh` checks it.

| n-C20H42/def2-TZVP RIJCOSX, laptop, alternating | before (9ac0bc855) | shared |
|---|---|---|
| wB97X total (2 runs each) | 108.5 / 109.8 s | 101.2 / 107.1 s |
| wB97X K | 62.0 / 63.2 s | 56.8 / 60.9 s |
| wB97X RSS in the SCF | 2.77 GB | 2.20 GB |
| wB97M-V (VV10 level 1) total | 193.0 s | 185.9 s |
| wB97M-V K | 65.5 s | 60.8 s |
| wB97M-V peak RSS | 5.42 GB | 4.89 GB |

- The energies are the same in every digit printed (wB97X -787.5433505627, wB97M-V -787.2623445252).
- The final level-2 step gains little memory: at the default max_memory that grid takes several blocks, where
  neither version keeps AO values.
- wB97M-V's RSS stays 2.6 GB above wB97X's (4.8 against 2.2 GB in the SCF): that is the XC/VV10 side, not SGX.
  (`bench/results/rijcosx/laptop_share_ab.log`)

### wB97M-V's extra memory was VV10's AO blocks: VV10 on the fused XC kernels (pyscf `perf` 4e37cc8fe, local), 2026-11-07
The user: "Investigate that extra memory overhead" (wB97M-V's SCF held 2.6 GB more than wB97X's).
- **Where it went** (`bench/diag/mem_parts.py`: RSS sampled by phase, n-C20H42/def2-TZVP RIJCOSX, 3 cycles): all
  inside `numint.nr_nlc_vxc`, and none of it kept (the RSS after each VV10 call within 0.03 GB of before). Its
  density and potential passes ran through `block_loop` with `max_memory`, the free budget, so each pass made AO
  blocks with first derivatives as large as the budget allowed (4 x 872 AOs x up to 67,200 points = 1.9 GB) plus
  `_dot_ao_aow_sparse`'s buffer. The meta-GGA XC was not the cause (with VV10 off the peak was wB97X's). At
  max_memory 20000 MB the VV10 phase peaked at 5.74 GB against 2.2 GB around it.
- **Change:** both passes take the fused kernels that `nr_rks` uses (`_fused_xc`, then `rho_evaluator` and
  `_dot_wv`), which compute the AO values in place; the gradient's VV10 term (`grad.rks.get_nlc_vxc`, AO blocks
  with second derivatives before, 10 components) takes `dot_terms` with `_xc_grad_terms('GGA')`. New keyword
  `_fused_xc(..., radial_cache=False)`: the NLC grid keeps no radial parts (0.43 GB at NLC level 1; at the default
  NLC level, the XC grid's, a second copy of the XC grid's 1.4 GB); two passes a cycle gain little from them. Where
  `_fused_xc` declines (Cartesian basis, small dense systems, a NumInt with its own functions) the `block_loop` path
  stays. UKS (the summed dm), the DF and SGX gradients and the Hessian's `nr_nlc_vxc` call go through the same two
  functions; the Hessian's own VV10 terms keep their `block_loop` passes (not measured).
- **Checks:** from dms bit for bit, from orbitals within 1.3e-18 (|v| 5e-3), the gradient term within 1.6e-17
  (`bench/diag/vv10_fused_check.py`, `vv10_fused_grad.py`); `test_fused_nlc`; the VV10 tests of dft, grad, hessian
  and df: 137 passed.

| n-C20H42/def2-TZVP wB97M-V RIJCOSX (VV10 level 1), laptop, 10 threads, alternating | VV10 on `block_loop` | fused (4e37cc8fe) |
|---|---|---|
| SCF at max_memory 4000 MB (2 runs each) | 157.1 / 158.4 s | 143.0 / 148.3 s |
| VV10 | 63.2 / 59.5 s | 46.3 / 48.7 s |
| RSS in the SCF | 4.84 / 4.85 GB | 2.24 / 2.24 GB (wB97X 2.20) |
| peak RSS (the final level-2 step) | 4.86 GB | 3.27 GB (wB97X 3.27) |
| SCF and VV10 at 20000 MB | 159.8 s, 61.5 s | 143.9 s, 47.1 s |
| RSS in the SCF and peak RSS at 20000 MB | 5.91 GB, 6.30 GB | 3.69 GB, 6.25 GB (wB97X 3.64, 6.23) |

- The gradient's VV10 term alone (`vv10_fused_grad.py`, max_memory ~17.7 GB as `pyscf-env.sh` sets it here):
  caffeine/def2-TZVP +5.3 -> +0.1 GB and 2.7 -> 1.3 s per call; n-C20H42/def2-TZVP +9.3 -> +0.4 GB and
  8.9 -> 4.4 s.
- The energies are the same in every digit printed (-787.2624487547 on the SCF grids, -787.2623445252 after the
  final step). The laptop ran in the balanced power profile (on AC); old and new alternated within the run.
- What is left: wB97M-V's memory is now wB97X's at both budgets. With a large max_memory the final level-2 step
  sets the peak for both (6.2 GB at 20000 MB): SGX keeps its AO values on the level-2 grid when they fit (1.24 GB),
  and the XC grid keeps its radial cache (1.40 GB at 20000 MB, the level-3 grid; it saves ~14% of the XC time,
  `dft_numint_radial_cache_fraction = 0` turns it off). ORCA's peak was 0.95 GB.
  (`bench/results/rijcosx/laptop_vv10_fused.log`, `laptop_vv10_mem_parts.log`, `laptop_vv10_grad.log`)
- **Pushed** (the user: "Push and update the pins"): PySCF `perf` 9ac0bc855..4e37cc8fe (option A 01f1d9b62, this
  4e37cc8fe) + annotated tag perf-2026-11-07e; pins and README -> perf-2026-11-07e (qcint stays perf-2026-11-07c).
  Before it, on the laptop: `kernels.py` ALL PASS (the five VV10 arrays within 1.7e-15 of pyscf-base); PySCF's
  tests (dft scf df grad hessian tdscf solvent sgx) 1063 passed, the same 7 known failures. No fresh-clone rerun
  (Python-only changes since perf-2026-11-07d).

### The wB97M-V Hessian: VV10's terms rewritten (pyscf `perf` e526b08ca, local), 2026-11-07
The user: "Could you see the memory usage for the wB97M-V functional for calculating the hessian on the hydrocarbon
system", then, at an estimate of 15 h for n-C20H42/def2-TZVP: "are there issues with the hessian in this code
path?" and "Let's stop it and work on fixing it". The exact Hessian (SGX has gradients, no Hessian), VV10 grid
level 1, max_memory 4000 MB, monitored by phase with `bench/diag/hess_mem.py`.
- **Where the time went.** propane/def2-SVP: 213 s, of it about 210 s in the three VV10 terms that PySCF took from
  GPU4PySCF (Liang et al., JCP 158, 204109 (2023)): `_get_enlc_deriv2` (energy second derivative), `_get_vnlc_deriv1`
  (Fock derivative, with grid response), `get_vnlc_resp1` (the response, every CPHF iteration, and TDDFT's).
  `bench/diag/vv10_hess_parts.py` timed them:
  - the pair kernels: `VXC_vv10nlc_hessian_eval_f_t` looped over all pairs of points once per group of 6 trial
    densities (for n-C20H42's 186 perturbations every pair 31 times, scalar), `_EUW_with_weight1` once per weight
    derivative, `_EUW_grid_response` visited every pair once per atom;
  - the AO work: `contract = numpy.einsum` (no BLAS, one core) per atom and per atom pair, with AO values to second
    and third derivatives evaluated again per atom: the Fock derivative alone ~5e14 flops for n-C20H42;
  - the batches assumed max(16 GB, max_memory/2) free (AO batches of 7.3 GB for nao 872 in the second derivative).
  - the stopped run: SCF 332 s (peak 3.6 GB), the two-electron part 254 s; the rest estimated from propane at ~15 h.
- **Change** (e526b08ca):
  - C: the pair kernels evaluate a block of 128 x 256 pairs once and apply it to all trials (or weight derivatives)
    in dgemm (each thread owns 128 output rows); the grid response sums each pair once into its atoms.
  - The ground-state quantities, with the O(N^2) U, W, A, B, C, E (`_VV10HessData`), are kept on the NLC grid for
    the calls with the same density (the CPHF iterations, TDDFT's Davidson iterations).
  - The densities and potential matrices of the perturbations through the fused kernels, all sets per block of
    points; the second derivative from the XC Hessian's pieces (the AO-diagonal `dot_terms`, the ipip pairs of
    `_get_vxc_deriv2_dm`, atom-resolved density derivatives by `_atom_pair_sum` once per block); the Fock
    derivative's AO-derivative terms from the gradient's `dot_terms` (for the AOs of each atom, and per atom for its
    points: the grids are sorted spatially, `sort_grids=True`, so an atom's points are not contiguous).
  - Batches follow max_memory.
- **Against the code before** (`vv10_hess_parts.py --compare`, `vv10_hess_ref.py`):

| | before | after | agreement |
|---|---|---|---|
| propane/def2-SVP Hessian | 220.4 s | 19.6 s | 4.0e-13 (max abs H 0.65) |
| propane TDA, VV10 in the response (`exclude_nlc = False`) | 594.9 s | 46.1 s | 7.9e-14 Eh |
| propane TDDFT, same | 417.4 s | 36.9 s | 7.3e-14 Eh |
| propane: energy 2nd derivative / Fock derivative / response of 33 densities | 15.5 / 50.3 / 20.4 s | 1.8 / 3.5 / 1.5 s | 5e-13 / 3e-14 / 3e-13 relative |
| water/def2-SVP Hessian, VV10 level 1 / level 3 | 4.6 / 37.2 s | 0.9 / 7.4 s | 6.2e-13 / 3.1e-12 |
| water TDA and TDDFT with VV10 | | | 2.4e-13 Eh |

  - n-C8H18/def2-TZVP (356 AOs, 85,936 VV10 points): second derivative 17.6 s, Fock derivative 37.9 s (f_t 12.8 s,
    the 78 potential matrices 9.7 s, EUW_w 9.0 s, EUW_gr 5.6 s, the AO-derivative terms 5.0 s), a response of 78
    densities 19.0 s. f_t runs at the dgemm rate (about 4.6e12 flops in 12.8 s); the CPHF passes almost all the
    perturbations in every iteration (propane: 33, 33, 33, 30, 30, 30, 5, 33).
  - Tests: test_vv10_hessian with its high-cost tests (9 passed), test_tdrks_vv10, test_response_function, hessian
    test_rks, dft test_numint, grad test_rks: pass.
- **n-C20H42/def2-TZVP wB97M-V Hessian, e526b08ca**: stopped by the user after the SCF (330 s, peak 3.60 GB) and the
  full-range two-electron terms (291.5 s, `_partial_hess_2e` 254.8 s, peak 3.84 GB), to work on the low-hanging
  fruit first.
- **Low-hanging fruit** (the user, of eight options: "Sure let's stop the current run and work on 1,2,5"):
  1. **No extra fvind after the Krylov solver** (pyscf `perf` 571affb7d). `cphf.solve_withs1` called fvind once
     more on the converged solution, for e1 and the last update of its virtual block. The solver now starts from x0
     (by default the uncoupled solution) with the occupied block set to -1/2 S1, so every Krylov vector has no
     occupied rows and the solution keeps that block exactly; fvind of the solution is the same combination of the
     responses already computed (`_krylov_coefficients`, `_krylov_combination`; checked against a call: 5e-16, 4e-13
     with VV10). One call fewer per CPHF (water RHF 9 -> 8, B3LYP 7 -> 6, wB97M-V 8 -> 7; propane wB97M-V 8 -> 7,
     vectors 227 -> 196). x0 also takes a guess from outside (`cphf.solve(..., x0=)`). hessian, scf, df and solvent
     tests pass (the known test_smd_grad).
  2. **One pair-kernel pass per group of dm1s in the VV10 response** (21e80c7e6): each call of `eval_f_t` costs
     about 2 s on n-C8H18/def2-TZVP's 82,054 points whatever the number of trials (1 trial 1.95 s, 31 trials 6.9 s,
     186 trials 31.0 s; dgemm at 345 GFLOP/s), ~12 s on n-C20H42's. The dm1s go to the kernel in groups as large as
     their masked data allow, their full-grid densities and potentials in batches of 32: n-C20H42 at the CPHF's
     2000 MB, 3 kernel calls per response of 186 dm1s instead of 5. One division per pair instead of three changed
     nothing (the loop is AVX2-vectorized; the per-call cost is the kernel blocks' traffic) and is not in.
- **Found on the way: PySCF's CPHF stops short for DFT** (upstream too, `bench/diag/cphf_residual.py`, one more fvind
  on the returned solution). `lib.krylov` drops and stops on the absolute norms of Krylov vectors that are orthogonal
  but not normalized and shrink by orders of magnitude, and its lindep floor (eps) caps the norm at ~1.5e-8 whatever
  tol; a small but significant direction can carry a large coefficient. True residuals of the returned mo1 against
  tol = 1e-8 * natm:

| | stock solver, before 571affb7d | stock, after | converged to 1e-11 (normalized Krylov, 2 Gram-Schmidt passes) |
|---|---|---|---|
| water/def2-SVP RHF | 3.7e-8 | 3.7e-8 | |
| water/def2-SVP B3LYP | 9.7e-5 | 2.2e-5 | 2.5e-10 / 1.2e-12 in 12-13 cycles, ~100 vectors (stock 6-7 cycles) |
| propane/def2-SVP wB97M-V | 3.2e-6 | 2.0e-6 | 4e-12 / 5e-11 in 28-30 cycles, ~900 vectors (stock 196-227) |

  The Hessians move with this: water B3LYP old against new 2.1e-5 (max abs H 0.71), propane wB97M-V 6.1e-7. Converged,
  the old and the new start give the same Hessian (0.70676300, 0.65438408). Converging to 1e-11 costs ~4x the
  response work for propane; a residual test with a looser tolerance would sit in between.
  **Future work** (the user: "Let's leave it as is for now but flag it as potential future work"): a stopping test
  on the true residual (the response of the solution is free since 571affb7d), a normalized Krylov with
  reorthogonalization, or preconditioned CG for the symmetric orbital Hessian of real perturbations.
  5. **Warm start from a cheap response** (`bench/diag/cphf_warm.py`: the CPHF of the cheap response, DF-J by the
     J-engine + SGX-K on a level-0 grid, the XC kernel on a level-1 grid, no VV10, then the exact solve from it with
     `x0`): two exact calls fewer, and a solution 40-60x closer, but slower in all:

| wB97M-V | cold: exact calls, time, residual | warm: exact calls + cheap calls, total, residual |
|---|---|---|
| propane/def2-SVP (33 perturbations), cheap tol x1000 | 7, 10.5 s, 2.0e-6 | 5 (7.5 s) + 5 (4.9 s), 12.5 s, 3.3e-8 |
| n-C8H18/def2-TZVP (78), cheap tol x1000 | 7, 282.5 s, 3.1e-6 | 5 (182.0 s) + 4 (328.5 s), 511.7 s, 7.0e-8 |

     The cheap response is no cheaper: on n-C8H18 82 s a call for 78 densities against 40 s for the exact one, whose
     J/K the many-density engine takes (SGX handles many densities poorly). A cheap response of the exact J/K and XC
     without VV10 would cost about half an exact call and break even at best. Not taken into the library (x0 is in
     `cphf.solve` from item 1).
  Also considered (the user asked about symmetries before the large run): point-group symmetry (n-C20H42 here is
  C2h, 21 symmetry-unique atoms of 62: the CPHF and make_h1 for 63 coordinates instead of 186; the grids are
  symmetric for this molecule; PySCF has no point-group Hessian), translational invariance (mostly used already),
  the locality of the atomic perturbations in the VV10 terms, the symmetry of the VV10 pair kernel (halves only the
  kernel evaluation and block traffic, ~1-2% of the Hessian). None done; the user chose to measure first.
- **n-C20H42/def2-TZVP wB97M-V Hessian, 21e80c7e6** (`bench/diag/hess_mem.py`, laptop, 10 threads, max_memory
  4000 MB, VV10 grid level 1, balanced power profile; `bench/results/hessian/laptop_c20h42_tzvp_wb97mv_hessian.log`):
  8538 s in all, the Hessian 8208 s (2 h 17 min), peak RSS 13.3 GB.

| phase | time | peak RSS |
|---|---|---|
| SCF (10 cycles) | 329.5 s | 3.61 GB |
| partial Hessian: two-electron terms, full-range J/K | 290.2 s | 3.83 GB |
| partial Hessian: long-range K | 208.0 s | 3.74 GB |
| partial Hessian: XC (meta-GGA) diagonal + second derivative | 13.3 + 63.1 s | 7.57 GB |
| partial Hessian: VV10 second derivative | 209.2 s | 5.60 GB |
| make_h1: XC Fock derivative | 214.5 s | 7.27 GB |
| make_h1: VV10 Fock derivative | 525.7 s | 8.89 GB |
| make_h1: derivative J/K | 316.6 s | |
| CPHF, 7 responses: J + full-range K / long-range K | 2323.8 / 1200.1 s | 13.32 GB |
| CPHF: VV10 response | 1703.2 s | 11.76 GB |
| CPHF: XC response | 1060.5 s | 8.73 GB |
| CPHF: the rest (Krylov, transformations) | 79.8 s | |

  - The CPHF is 78% of the Hessian (6367 s); exact J/K 55% of it. A response of 186 densities: J + K 473 s, long-range
    K 245 s, VV10 294 s, XC 184 s; the last two responses had fewer vectors (575 s and 187 s in all).
  - VV10 in all: 2438 s (30% of the Hessian), estimated at ~15 h before e526b08ca.
  - Memory above max_memory, three sizings: (1) the CPHF takes the 62 atoms in one block (solve_mo1 sizes the block
    by the MO-basis arrays only) and its response holds several stacks of 186 AO matrices, 1.13 GB each (the
    densities, the XC and VV10 potentials, J, K and long-range K), on top of h1ao (1.13 GB) and the SCF's 3.3 GB;
    (2) `_get_vxc_deriv2_dm` and `_get_vxc_deriv1` size their AO blocks (to second derivatives) by the whole budget
    and add the products, AO-dm products and per-atom arrays per block (~4 GB over); (3) the VV10 Fock derivative's
    per-atom grid arrays (~3 GB) beside its 1.13 GB result.
- **Pushed** (the user: "Push it and then work on the memory usage"): PySCF `perf` 4e37cc8fe..21e80c7e6 (e526b08ca,
  571affb7d, 21e80c7e6) + annotated tag perf-2026-11-08; pins and README -> perf-2026-11-08 (qcint stays
  perf-2026-11-07c; the README says which change is not to rounding, the CPHF's). Before it, on the laptop:
  `kernels.py` ALL PASS; PySCF's tests (dft scf df grad hessian tdscf solvent sgx) 1063 passed, the same 7 known
  failures. No fresh-clone rerun (Python and libdft only; the build is the same).

### The Hessian's memory (pyscf `perf` f5ca066cb, c6570dbac, 8e9a5f85a, local), 2026-11-08
The user: "Push it and then work on the memory usage". The n-C20H42/def2-TZVP wB97M-V Hessian above peaked at
13.3 GB under max_memory 4000 MB (nao 872, 186 perturbations: a stack of 186 AO matrices is 1.13 GB).
- **Where it went:**
  - **The SCF's semi-direct ERIs, 2722 MB**, resident through the Hessian (`_init_semi_direct` gives them 3/4 of what
    is left under max_memory). No integral of the Hessian reads them: its derivative integrals are other ones, and
    the CPHF's many-density J/K (`_direct_jk_gemm`, 13 dms or more) evaluates all its quartets. With them the
    Hessian started at 3.3 GB, and every phase's `max(2000, max_memory*.9 - mem_now)` was the 2000 MB floor.
  - **The CPHF J/K (the 13.32 GB peak):** the engine's symmetry test `abs(dms - dms.transpose(0,2,1))` made two
    temporaries of all 186 dms (2.26 GB); J and K were made while the dms, the XC + VV10 potentials and the VV10
    response's own stack were alive; the long-range K was made while J and K were; `vj - .5*vk` temporaries.
  - **The XC Fock derivative and second derivative:** AO blocks (to second derivatives) sized by the whole budget, with
    the per-block products and per-atom arrays on top (~4 GB over).
  - **The VV10 Fock derivative:** drho/dA, dgamma/dA and w f_gamma d(nabla rho)/dA of all atoms on all points
    (~3 GB) beside its 1.13 GB result.
- **Changes** (results to rounding):
  - f5ca066cb:
    - the RKS response adds each J or K stack, scaled in place, before the next is made;
    - the VV10 response adds its matrices to v1 a batch at a time (`get_vnlc_resp*(out=)`);
    - the VV10 Fock derivative keeps drho/dA and dgamma/dA on the masked points only and evaluates d(nabla rho)/dA
      again per block, its blocks sized from max_memory (`_vv10_blksize`);
    - the XC Fock derivative and second derivative size their AO blocks from the memory left, the other arrays of a
      block counted (`_xc_block_memory`, a 500 MB floor).
  - c6570dbac:
    - `HessianBase.kernel` releases the semi-direct ERIs, and keeps get_jk from storing them again, when H1 and the
      CPHF's arrays of all perturbations (about 5 AO matrices each) would not fit beside them under max_memory. The
      next SCF stores them again. Small molecules keep them: their CPHF calls have fewer than 13 dms and read them.
    - The RKS response makes J and K first and adds the XC and VV10 responses to the memory of J; the HF response
      adds -K/2 to J in place (the same numbers bitwise).
    - The direct and incore J/K test the dms' symmetry a few at a time and free each batch's buffers before the next.
  - 8e9a5f85a: the VV10 Fock derivative adds to h1ao (`_get_vnlc_deriv1(..., out=)`: h1ao halved, the half matrices
    accumulated into it, one symmetrization) instead of returning a stack of its own; also in the DF Hessian.
- **Checks:**
  - Old (21e80c7e6) against new, `bench/diag/hess_resp_ref.py`: Hessians (water HF, UHF H2O+, B3LYP with its ERIs
    in memory, direct, and with the release forced, UKS H2O+, HSE06, LRC-wPBE, CAM-B3LYP, wB97M-V; propane wB97M-V,
    the GEMM J/K): within 8.5e-11 relative (CAM-B3LYP's 6e-11 is the CPHF tolerance's noise; HF and UHF 1.5e-15).
    TDA and TDDFT singlets and triplets (HF, B3LYP, HSE06, LRC-wPBE, CAM-B3LYP, wB97M-V with VV10): within 6e-13.
    With the release forced, no semi-direct ERIs after the Hessian, and the next get_jk stores them again.
  - The VV10 Fock derivative of propane against 21e80c7e6: 2.7e-14 relative.
  - PySCF's tests (dft scf df grad hessian tdscf solvent sgx): 1063 passed, the 7 known failures, after f5ca066cb and
    after c6570dbac; after 8e9a5f85a hessian, df and dft (264 passed) and test_vv10_hessian with its high-cost tests
    (9 passed). Two runs of the same tree differ by up to 1.4e-11 in these Hessians (the threads' order of sums), the
    size of the old-against-new differences.
- **n-C8H18/def2-TZVP wB97M-V Hessian, max_memory 4000 MB** (laptop, 10 threads, balanced profile; snapshots, old
  21e80c7e6, new c6570dbac then 8e9a5f85a, run old, new, old, new; `scratch/hess-mem-ab/run.sh`, `run2.sh`;
  `bench/results/hessian/laptop_c8h18_tzvp_wb97mv_hessian_mem_ab.log`, `bench/diag/hess_mem_compare.py`). The SCF's
  semi-direct ERIs: 2461 + 622 MB; after the SCF 3.82 GB resident, so the new Hessian releases them.

| phase | old: time (two runs) | old: peak RSS | new: time (two runs) | new: peak RSS |
|---|---|---|---|---|
| SCF (the same code in both) | 36.8, 41.9 s | 3.86 GB | 44.8, 44.5 s | 3.86 GB |
| partial Hessian | 78.6, 87.6 s | 8.08 GB | 92.3, 93.1 s | 3.24 GB |
| of it `_partial_hess_2e`, full-range J/K and long-range K (the same code) | 45.8, 52.9 s | 4.12 GB | 56.6, 57.1 s | 1.10 GB |
| of it the XC second derivative | 10.1, 10.3 s | 8.08 GB | 10.7, 10.7 s | 3.24 GB |
| of it the VV10 second derivative (the same code) | 18.3, 19.7 s | 4.65 GB | 20.3, 20.5 s | 1.75 GB |
| make_h1 | 105.6, 110.9 s | 6.90 GB | 115.1, 115.9 s | 2.88 GB |
| of it the XC Fock derivative | 29.3, 31.1 s | 6.90 GB | 31.7, 32.2 s | 2.88 GB |
| of it the VV10 Fock derivative | 40.9, 42.9 s | 5.89 GB | 46.6, 47.0 s | 2.80, 2.72 GB |
| CPHF | 304.8, 306.8 s | 5.53 GB | 304.9, 304.2 s | 2.45 GB |
| of it the J/K | 100.0, 100.9 s | 4.98 GB | 100.0, 99.5 s | 1.88, 1.83 GB |
| of it the VV10 response | 134.5, 134.8 s | 5.53 GB | 134.2, 134.0 s | 2.45 GB |
| of it the XC response | 67.8, 68.5 s | 4.52 GB | 68.2, 68.1 s | 1.45 GB |
| Hessian | 489.1, 505.2 s | 8.08 GB | 512.4, 513.3 s | 3.24 GB |
| whole run (ru_maxrss) | 526, 547 s | 7.89, 7.90 GB | 557, 558 s | 3.77 GB |

  - The whole run's peak is now the SCF's (its semi-direct ERIs); the Hessian stays under max_memory.
  - Time: the phases whose code is the same in both trees (the SCF, the two-electron terms, the VV10 second
    derivative) moved by up to 22% between runs, the first run after an idle laptop the fastest; the Hessians'
    489-513 s is mostly that. The new code's one change of time is the VV10 Fock derivative, +4-6 s: d(nabla rho)/dA
    evaluated again per block instead of 9 natm ngrids doubles kept (0.93 GB on n-C20H42). The CPHF, where the
    response changes are, took the same time in all four runs (304-307 s). max|H| the same (0.6161).
- **Projected for n-C20H42/def2-TZVP at 4000 MB** (from the array sizes, not measured): the Hessian starts at about
  0.6 GB instead of 3.3; the XC terms and make_h1 stay near max_memory (3-4 GB); the CPHF peaks at about 7-8.6 GB
  (before 13.3), from what it still holds at once: h1ao, the 186 densities, J and K (1.13 GB each), the J/K engine's
  batch buffers (about 1.25 GB: it batches 41 dms at the 1000 MB its formula gives), and the MO-basis arrays with the
  Krylov space (0.7-2.2 GB over the iterations).
- **Left** (time against memory; for the user to decide):
  - Chunking the CPHF response over the perturbations: the densities, J/K and VV10 by chunks, the XC response from
    the factors for all at once. About 4-5 GB on n-C20H42 at 4000 MB. The J/K engine's integral passes would stay
    (5 per response there either way); the VV10 pair kernel would run once per chunk (about 5 calls of ~12 s per
    response instead of 2, ~+250 s, ~3%). With room under max_memory it is one chunk, as now.
  - h1ao on disk between make_h1, the CPHF and the last contraction: 1.13 GB on n-C20H42, a few seconds of I/O.
  - The UHF/UKS response (`_gen_uhf_response`) has not had the J/K-first change.

### The RIJCOSX-style CPHF response: SGX-K of many densities (pyscf `perf` 0ad1f8318, local), 2026-11-08
The user: "Let's start on the RIJCOSX next and then come back to the direct paths higher memory usage".
- **Approach.** There is no RIJCOSX Hessian (`_SGXHF.Hessian = method_not_implemented`). Here the SCF and every
  derivative integral stay exact and only the CPHF response's J and K are approximated: DF-J by the J-engine and
  SGX-K of an RIJCOSX copy of the SCF object, as an opt-in (not in the library's API yet).
- **Accuracy** (`bench/diag/rijcosx_resp_accuracy.py`, wB97M-V/def2-SVP, the XC and VV10 responses exact, harmonic
  frequencies by `thermo.harmonic_analysis`; `--converge`: every CPHF converged, normalized Krylov and
  conv_tol_cpscf 1e-12):

| response J and K | water: max\|dH\|, max\|df\| | propane: max\|dH\|, max\|df\| (mean) |
|---|---|---|
| DF-J (def2-universal-jfit) + exact K | 1.3e-4, 0.41 cm-1 | 4.3e-5, 0.45 cm-1 (0.08) |
| DF-J + SGX-K, level 0 | 1.3e-4, 0.38 | 4.0e-5, 0.53 (0.09) |
| exact J + SGX-K, level 0 | 1.3e-5, 0.03 | 2.7e-5, 0.11 (0.04) |
| DF-J + SGX-K, level 1 | 1.3e-4, 0.41 | 5.4e-5, 0.46 (0.09) |
| exact J + SGX-K, level 1 | 5.7e-7, 0.00 | 1.1e-5, 0.03 (0.01) |

  - Exact J is direct SCF's (`sgx.dfj = False`, `sgx.direct_j = True`;
    `bench/results/rijcosx/laptop_resp_accuracy_direct_j.log`). The first version of these two rows set only
    `dfj = False`, which leaves J to SGX's grid as well (those runs: level 0 0.20 and 0.33 cm-1, level 1 0.01 and 0.04).
  - DF-J's fitting basis carries most of the error: the response densities of nuclear perturbations have
    core-region character that a J-fitting basis does not describe. DF-J + exact K with def2-universal-jkfit 0.11
    and 0.24 cm-1, PySCF's even-tempered `aug_etb` basis (2.4x jfit's size) 0.02 and 0.05, `autoaux` 0.02 and 0.24
    (stock CPHF).
  - With the stock CPHF, one run of DF-J + SGX-K level 0 on propane gave 3.74 cm-1 (max|dH| 2.0e-3): `lib.krylov`
    stopping short (above, flagged as future work) can move the Hessian more than the approximation, so an
    approximate response needs the converged CPHF.
- **SGX-K of many densities, before** (`bench/diag/sgx_many_dm.py`, n-C8H18/def2-TZVP B3LYP, 78 CPHF-like
  densities, 10 threads): exact J and K 7.7 s, DF-J 3.4 s, SGX-K level 0 21.7 s, level 1 38.8 s. One density took
  0.40 / 0.60 s and every further one 0.26 / 0.51 s: `SGXnr_direct_k_drv` evaluates a batch's shell-pair integrals
  once but contracts them with one density after another, each with accumulators of its own (78 of nao x 224
  doubles per thread), out of cache.
- **Change** (0ad1f8318, `lib/vhf/nr_sgx_many.c`, `sgx_jk._get_k_only_many`): `get_k_only` takes 8 densities or
  more at once. Per block of points, F = X D and K = X^T w G are matrix products over all densities (the density
  index innermost, the AOs with values on the block), and G = A F is `SGXnr_direct_k_many`: per batch the screening
  of `SGXscreen_grid_pair` (shell norms over all densities, so the shell pairs of the old driver on the same
  densities), the integrals of every pair once, stored point-major; per point A gathered by row shell (both
  triangles, flat column lists), then 8 densities at a time with the row's sums in registers (AVX2/FMA picked at
  run time, the file built at -O3). Other screenings keep the old driver. Against the old driver on the same
  densities: within 1.4e-14 relative; `test_sgx_jk.py ManyDensities`. PySCF's tests: 1064 passed, the 7 known
  failures.
  - The contraction on C8H18, level 1: per-density loop 33 s; point-major integrals with an AXPY over the densities
    9.8 s; the sums by row shell 7.4 s; flat columns 5.2 s; -O3 2.7 s (GCC at -O2 kept the accumulators on the
    stack, every FMA through memory; the row kernels then run at 265 GFLOP/s).
  - Tried and dropped: the F and K products on runs of consecutive AOs instead of a gather (2-4x slower, every
    run's product reads all of F or G again).

| SGX-K and DF-J of the CPHF-like densities | exact J + K | DF-J | SGX-K level 0 | SGX-K level 1 |
|---|---|---|---|---|
| n-C8H18/def2-TZVP, 78 densities (from make_h1) | 8.4 s | 3.6 s | 4.9 s | 8.7 s |
| n-C12H26/def2-TZVP, 114 | 48.6 s | 13.7 s | 18.1 s | 34.4 s |
| n-C20H42/def2-TZVP, 186 (`bench/results/rijcosx/laptop_c20h42_sgx_many_dm.log`) | 473 s (the Hessian's) | 62.6 s | 101.1 s | 193.0 s |

  - n-C20H42, level 1: F = X D 35.4 s, G = A F 86.0 s, K = X^T w G 68.3 s. On C8H18 (the make_h1 densities)
    against exact: DF-J 1.7e-3 of max|J|, SGX-K level 0 1.5e-3, level 1 1.6e-4 of max|K|.
  - Exact J and K grow about as N^4.4 here, DF-J and SGX-K as N^3.4: no gain at C8H18, even at C12H26 (level 1).
- **Exact J against DF-J** (the user: "What is the timing difference between exact and DF-J for the hessians?";
  `sgx_many_dm.py --h1 --j-only`, the CPHF's first densities; `bench/results/rijcosx/laptop_exact_j_vs_dfj.log`):

| | exact J alone | DF-J (jfit) | DF-J (`aug_etb`) | exact J + K | exact K alone |
|---|---|---|---|---|---|
| n-C8H18/def2-TZVP, 78 densities | 4.5 s | 3.9 s | 7.9 s | 9.0 s | 6.6 s |
| n-C12H26/def2-TZVP, 114 | 19.4 s | 13.6 s | 27.7 s | 50.4 s | 31.7 s |
| n-C20H42/def2-TZVP, 186, max_memory 4000 MB | 172.6 s | 65.3 s | | 473 s (the Hessian's) | |
| the same, 12000 MB | 136.5 s | 70.8 s | | | |

  - The J engine holds 54 dms at 4000 MB, so n-C20H42's 186 take 4 passes over the integrals; one at 12000 MB, and at
    the two smaller sizes. A power law through C8 and C12 gave ~120 s for n-C20H42. At 12000 MB the ratio to DF-J is
    the expected one (1.93x; that run's SCF and make_h1 were 10-12% slower than the 4000 MB run's), at 4000 MB 2.64x.
  - On these densities DF-J (jfit) is within 1.7e-3 of max|J|, `aug_etb` 1.6e-4.
  - DF-J with exact K saves little (C12H26: 13.6 + 31.7 s against 50.4 s; the whole propane Hessian 19.2 s against
    19.5 s): K is where the time goes.
  - Over the Hessian (its J + K is 4.9 full responses), exact J costs about 530 s more than DF-J at 4000 MB (6% of
    the 8208 s) and 320 s at 12000 MB (4%).
- **Where that leaves the response** (n-C20H42/def2-TZVP wB97M-V, a response of 186 densities): exact J + K 473 s
  and long-range K 245 s; DF-J + SGX-K level 1 63 + 193 s and a long-range SGX-K of about as much (less with F
  shared), so the J/K 1.6-1.9x faster, level 0 2.5-2.9x. With direct J at 4000 MB: 173 + 193 + 193 s at level 1
  (1.3x against 718 s), 173 + 101 + 101 s at level 0 (1.9x); the long-range SGX-K is assumed equal to the full-range
  one, not measured. The XC (151 s) and VV10 (243 s) responses stay: a CPHF iteration from about 1110 s to 650-950 s.
- **Left:** the opt-in response in the Hessian, with the converged CPHF; the F and K products (half of SGX-K at
  C20H42: dense over the block's AOs times all AOs; F shared between the full- and long-range K); AVX-512 row
  kernels. For the J of the response the user, before the measurements above: "Let's do a measurement first, but
  assume it is in line with your estimate I think the direct J is a good call". `aug_etb` is no faster than direct J
  at these sizes.

### The response in the Hessian, and a CPHF solver that stops on the residual (pyscf `perf` cd7fe5bec, local), 2026-11-08
The user: "Yes wire it in" (direct J + SGX-K, level 0 by default, level 1 selectable, with a converged CPHF).
- **What is there now.** `hobj = mf.Hessian(); hobj.sgx_response = True` (`hobj.sgx_response_level = 1` for level 1):
  the J and K of the CPHF response only are direct J (`SGX.get_jk` with `direct_j`, now with the J-only prescreen of
  `SCF.get_jk`) and SGX-K, from an RIJCOSX copy of the SCF object (`rhf._sgx_response_scf`). The SCF, the derivative
  integrals and the XC and VV10 responses stay exact. `hobj.cphf_residual_tol = 1e-4` takes the new solver with any
  response; `sgx_response` uses it with tol 1e-4 unless told otherwise. RHF and RKS only: UHF/UKS and density-fitted
  SCFs raise NotImplementedError. Nothing changes without these switches. Tests:
  `pyscf/hessian/test/test_sgx_response.py` (the solver against dense solves on a toy problem, the options, the
  NotImplementedError cases). `scripts/run_tests.sh`: 1072 passed (the 8 new ones among them), the 7 known failures.
- **Why a new solver.** The "converged CPHF" behind the accuracy tables above (normalized Krylov, tol 1e-12) runs until
  the space is nearly exhausted, ~4x the response work (propane ~900 vectors), more than the approximate response saves.
  With SGX-K the stock solver also misbehaves: its Krylov vectors shrink under SGX's absolute thresholds (propane, level
  0: 42 calls and 250 vectors; level 1: 23 calls, 232 vectors).
- **`cphf.solve_residual`** (`bench/diag/cphf_solver_variants.py`, the prototype with the choices not taken;
  `bench/results/rijcosx/laptop_cphf_solver_variants.log`). One trial space for all perturbations; each cycle adds the
  preconditioned residuals of the perturbations above tol, made orthonormal (two Gram-Schmidt passes, then pivoted QR);
  stops on the true residuals. Three choices the scan settled:
  - the projection in the plain dot on M = (e_a - e_i) + fvind (symmetric), not on the preconditioned operator as
    `lib.krylov`: smaller Hessian errors at the same cost, 3x at tol 3e-3 and 200x at 1e-5 (propane, tol 1e-4: 2.3e-8
    against 6.3e-7);
  - the last update of the virtual block from the response of the solution (free): water at tol 1e-4, 7.7e-9 with it,
    7.1e-6 without;
  - starting from the uncoupled solution in one call (starting from the occupied block alone, which costs one call
    more, is no better).
  The Hessian's error goes about as the square of the residual.

| propane/def2-SVP wB97M-V, 33 perturbations; errors against tol 1e-10 with the same response | response calls | vectors | max\|dH\| | max\|df\| cm-1 |
|---|---|---|---|---|
| stock, exact response | 7 | 196 | 6.9e-7 | 0.0022 |
| stock, direct J + SGX-K level 0 / level 1 | 42 / 23 | 250 / 232 | 1.5e-5 / 2.0e-5 | 0.014 / 0.031 |
| `solve_residual`, tol 3e-3 | 3 | 87 | 1.6e-5 | 0.38 |
| tol 1e-3 | 4 | 97 | 4.5e-6 | 0.017 |
| tol 3e-4 | 4 | 120 | 9.5e-8 | 0.0020 |
| tol 1e-4 | 5 | 130 | 2.3e-8 | 0.0001 |
| tol 1e-5 | 6 | 160 | 6.6e-10 | 0.0000 |

  - The `solve_residual` rows are the same to the digits shown with the exact response and SGX-K on levels 0 and 1, so the
    solver's error does not depend on the approximation. Water/def2-SVP wB97M-V: stock 7 calls, 47 vectors, 2.8e-6;
    tol 1e-4 5 calls, 34 vectors, 7.7e-9.
  - Against an independent solve (normalized Krylov, tol 1e-12; water/def2-SVP): `solve_residual` tol 1e-9 is within
    8.4e-13 (B3LYP) and 4.4e-16 (RHF); the stock solver 7.5e-6 and 3.2e-9; tol 1e-4 5.6e-8 and 2.5e-8.
- **Through the API, n-C5H12/def2-SVP wB97M-V** (nao 130, 51 perturbations; `bench/diag/hess_sgx_response.py`,
  `bench/results/rijcosx/laptop_c5h12_sgx_response.log`; against `cphf_residual_tol` 1e-8 with the exact response):

| | CPHF | response calls | vectors | max\|dH\| | max\|df\| (mean) cm-1 |
|---|---|---|---|---|---|
| stock, exact response | 39.0 s | 7 | 302 | 3.0e-7 | 0.003 (0.000) |
| `solve_residual` 1e-4, exact response | 27.0 s | 5 | 207 | 2.7e-8 | 0.000 (0.000) |
| `solve_residual` 1e-4, direct J + SGX-K level 0 | 30.4 s | 5 | 207 | 2.7e-5 | 0.155 (0.038) |
| `solve_residual` 1e-4, direct J + SGX-K level 1 | 33.6 s | 5 | 207 | 1.3e-5 | 0.031 (0.007) |

  - At this size SGX-K gains nothing (as expected from the N^4.4 against N^3.4 above); the new solver alone is 31% off
    the CPHF.
  - The largest frequency error of level 0 grows with size: water 0.03, propane 0.11, n-C5H12 0.155 cm-1. Level 1: 0.00,
    0.03, 0.03.
- **Projected for n-C20H42/def2-TZVP wB97M-V (before the measurement below, which came out higher)**, from the stock Hessian's CPHF of 6367 s (J/K 3524, VV10
  1703, XC 1060, rest 80), n-C5H12's 207/302 of the vectors, and J/K per response of 718 s exact, ~375 s (level 0),
  ~560 s (level 1), the long-range SGX-K assumed equal to the full-range one: exact response with the new solver ~4390 s
  (24% off the 8208 s Hessian), level 1 ~3860 s (31%), level 0 ~3230 s (38%).
- **Measured, n-C20H42/def2-TZVP wB97M-V** (nao 872, 186 perturbations, VV10 grid level 1, 10 threads, max_memory
  4000 MB, laptop on balanced; `bench/diag/hess_mem.py --residual-tol 1e-4 [--sgx-level 0] --save`,
  `bench/results/hessian/laptop_c20h42_tzvp_wb97mv_hessian_{residual,sgx0}.log`; the two new runs one after the other on
  2026-11-08 from a snapshot of cd7fe5bec. The stock column is the earlier run on 21e80c7e6, before the Hessian's memory
  commits):

| | stock solver, exact response (21e80c7e6) | `solve_residual` 1e-4, exact response | `solve_residual` 1e-4, direct J + SGX-K level 0 |
|---|---|---|---|
| response calls | 7 | 5 | 5 |
| J/K in the response | 3524 s | 3539 s | 1961 s |
| XC response | 1061 s | 912 s | 905 s |
| VV10 response | 1703 s | 1428 s | 1399 s |
| CPHF | 6367 s | 5896 s | 4283 s |
| Hessian | 8208 s | 8043 s | 6413 s |
| whole run | 8538 s | 8431 s | 6763 s |
| peak RSS (sampled) | 13.32 GB | 8.12 GB | 11.55 GB |

  - Level 0 against the exact response, same solver, same tree, same evening: J/K 45% less, CPHF 27% less, Hessian 20%
    less. XC and VV10 are now 54% of level 0's CPHF.
  - Level 0's error at this size, the two saved Hessians compared: max|dH| 8.1e-5; max|df| 0.518 cm-1, at a -18 cm-1
    mode (the geometry is not optimized: 9 imaginary modes in both); 0.175 cm-1 at most above 200 cm-1; mean|df| 0.067 and
    mean signed df -0.067, so the shifts are of one sign; ZPE -0.017 kcal/mol out of 368. The largest error over all
    modes went water 0.03, propane 0.11, n-C5H12 0.155, n-C20H42 0.518, the last from a soft mode (df ~ dlambda/2f).
  - **The new solver saves little at this size with the exact response**: CPHF -7%, Hessian -2%. The Hessian's -2% is
    inside the noise (the SCF, identical work, took 351 s and 388 s in the two runs). All of it is in XC + VV10 (2340 s
    against 2764 s, fewer vectors); J/K shows no saving. The stock solver's J/K per response falls, 717, 670, 634, 590, 533,
    293, 88 s, while over the first five responses its XC (182-185 s) and VV10 (291-295 s) stay flat: the same number of
    vectors, of smaller and smaller norm, most likely screened harder (not isolated). `solve_residual` orthonormalizes its
    trial vectors and its J/K does not fall: 736, 788, 793, 792, 431 s. The tol-1e-4 solver's accuracy is not measured at
    this size (no tight reference; n-C5H12: 2.7e-8 against the stock solver's 3.0e-7).
  - Against the projection: CPHF 5896 s against ~4390 s (exact), 4283 s against ~3230 s (level 0). The projection took
    n-C5H12's 207/302 of the vectors, which implied ~3.4 full-size responses; going by the timings the run took 4 at
    full size and one at about half. And J/K per full response came out at 420-438 s for level 0 (direct J + full-range
    SGX-K 307-322 s, long-range SGX-K 109-118 s; projected ~375 s) and 736-793 s for the exact response (projected 718 s).
  - The phases the solver does not touch were slower than in the stock run (SCF 351-388 s against 330, partial_hess_elec
    885-908 s against 784, make_h1 1239-1244 s against 1057). The memory commits and the machine's state cannot be
    told apart here, so the comparison with the stock column is not like-for-like; the other two columns are.
  - Memory: level 0's peak is 3.4 GB above the exact response's, so that much is SGX-K's own, on top of the direct path's
    8.12 GB (which is itself twice max_memory). The peak comes in the J/K of the response in both.
- **Left:** whether `solve_residual` should be the Hessian's default solver (it changes results by what the stock solver
  leaves unconverged, and at this size it saves only ~7% of the CPHF); scaling each trial vector by its residual norm
  before the exact response and back after (J/K is linear), to recover the screening the stock solver gets, not tried and
  worth about 560 s of the 3539 s if the J/K fell as the stock solver's does (not for SGX-K, whose absolute thresholds are
  why the vectors are normalized); SGX-K's and the direct path's peak memory; UHF/UKS.

### Merged into `perf`, tagged perf-2026-11-07c, 2026-11-07
The user: "Yes let's merge it, this sounds very much like what I have been looking for within PySCF".
- **PySCF `perf` fast-forwarded to `rijcosx` 0e7843bfd** (with it the unpushed refactor commits c9bfd2ea5 and
  28d60a20e); qcint `perf` stays at e25a81f (the refactor's template). Both tagged perf-2026-11-07c; pins and README
  updated.
- **Checks on the merged tree (laptop):**
  - `kernels.py`: ALL PASS.
  - AddressSanitizer (`pyscf-rx` with `-fsanitize=address -fno-omit-frame-pointer -g`, `LD_PRELOAD` libasan,
    pytest `-s`): df_jk, sgx and sgx/grad tests 31 passed, `rij_engine_check.py` within 1.5e-13, no report.
  - `validate.py` full and hessian suites: all pass (21 + 21 cases), energies within 5.5e-12, gradients 1.5e-12,
    Hessians 6.4e-11; the tight SCFs' cycle counts scatter by 1-2 as before.
  - PySCF's tests (dft scf df grad hessian tdscf solvent sgx): 1061 passed, the same 7 known failures.
- **Pushed:** PySCF `perf` 43838f635..0e7843bfd and qcint `perf` cd07a5d..e25a81f with their tags; workspace main
  eba7254 (pins and README).
- **node2, a fresh clone of eba7254** (`bench/results/node2/node2_fresh_clone_2026-11-07c.log`):
  `cpu_setup.sh --with-base` 41 min; water B3LYP -2.8e-14 Eh from the laptop; `kernels.py` 77 arrays agree; PySCF's
  tests 1061 passed, the same 7 known failures; `rij_engine_check.py` within 1.6e-13 (AVX2); caffeine/def2-TZVP
  RIJCOSX 47.3 + 5.0 s, the energy of the earlier node2 run to the last digit, peak RSS 2.0 GB, no disk. The clone
  was removed afterwards.

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
