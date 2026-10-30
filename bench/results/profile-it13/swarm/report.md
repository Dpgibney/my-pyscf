# Exact speedups after It13: final report

Code paths are relative to `/home/danny/Chemistry/my-pyscf/pyscf/pyscf/`. `scripts/`, `bench/` and `NOTES.md` are under `/home/danny/Chemistry/my-pyscf/`. The analysts' prototype scripts are under `bench/results/profile-it13/swarm/prototypes/<area>/`.

**Scope.** Eight areas produced 56 findings. Verification confirmed or adjusted 48 and refuted 8. One of the 48 is a screening change, so it goes to §4 as opt-in. The other 47 merge into 25 ranked opportunities on bench workloads, plus 4 that need a bench case before they can be measured.

**Headline.**
- **Hessians: the largest exact savings.** Four S/M changes to the derivative J/K and `hess_elec` (#1, #2, #3, #5) should take `hess_caffeine_svp_b3lyp` from 181 s to about 105–119 s. That range is the sum of the verified estimates; it has not been measured end to end.
- **Next, in order:**
  - per-file vector flags for the fused XC kernels (#4), which give bitwise-identical results;
  - the factored TD response (#6);
  - a thread-pool configuration fix (#7), whose size is unknown until one A/B run is done.

## 1. Where the time goes now

Source: `/home/danny/Chemistry/my-pyscf/bench/results/profile-it13/summary.txt`, 10 threads, perf `d88076cb8`. All times are in seconds. "XC" means the fused kernels: `dot` + `_rho_kernel`, or `rho_lowrank` + `dot_ao_mo` for CPHF.

**Caveat: some of these numbers were taken on a loaded machine.**
- `scf_caffeine_svp_pbe` and `scf_caffeine_svp_b3lyp` ran while a 1-thread py-spy profile was still running. The 10-thread run started at 18:09:28; the native scf_caffeine profile ended at 18:09:59. Their XC kernels read about 10–15% above clean It11 runs.
- The five caffeine/benzene native profiles also overlapped each other and are about 2× inflated. Any 1→10-thread scaling ratio taken from them is unreliable.

**SCF**

| case | wall | largest parts |
|---|---|---|
| caffeine/SVP PBE | 10.22 | XC 5.71 (dot 2.88, rho 2.83); J/K 1.91 (in-core 1.04, ERI build 0.87); libxc 0.88 |
| caffeine/SVP B3LYP | 10.03 | XC 5.16 (dot 2.66, rho 2.50); J/K 2.34 (in-core 1.39, ERI build 0.95); libxc 0.67 |
| caffeine/SVP r2SCAN | 11.67 | XC 6.99 (dot 4.78, rho 2.22); J/K 1.85; libxc 1.05 |
| caffeine/SVP CAM-B3LYP | 10.69 | J/K 4.66 (LR-K semi-direct 2.59, in-core 1.12, ERI build 0.94); XC 3.95; libxc 0.60 |
| caffeine/TZVP DF-PBE | 14.96 | XC 10.89 (dot 6.92, rho 3.97); eigh 0.82; get_overlap_cond + condense 0.55 |
| caffeine/TZVP B3LYP | 81.18 | J/K 69.28 (semi-direct 68.77); XC 9.62; eigh 0.65 |
| caffeine+ UKS B3LYP | 21.16 | XC 9.87 (dot 5.30, rho 4.58); J/K 6.43 (in-core 5.50); libxc 1.52 |
| benzene+ UKS r2SCAN | 4.09 | XC 2.05 (dot 1.57, rho 0.48); libxc 1.08; get_overlap_cond + condense 0.30 |
| n-C20H42/SVP B3LYP | 18.77 | J/K 9.84 (semi-direct 9.75); XC 6.14; grid build 0.77 (point sort 0.40, Becke partition 0.37) |
| n-C20H42/SVP r2SCAN | 22.74 | XC 12.75 (dot 8.45, rho 4.30); J/K 6.42; libxc 1.10 |
| n-C20H42/TZVP PBE | 159.26 | J/K 138.02 (semi-direct 136.02); XC 16.78; eigh 1.42 |

- Smaller recurring parts, targeted in §2:
  - get_overlap_cond + condense: 0.37–0.55 s per caffeine RKS SCF, 1.02 s for UKS;
  - scipy eigh: 0.34–0.45 s (caffeine/SVP) to 1.42 s (alkane/TZVP);
  - grid build: 0.27–0.30 s (caffeine) and 0.75–0.80 s (alkane).
- The two largest walls (caffeine/TZVP B3LYP and alkane/TZVP PBE) are 85–87% semi-direct J/K on quartets that are not in the cache. No exact lever was found there beyond faster qcint and a larger memory budget.

**Gradients**

| case | wall | largest parts |
|---|---|---|
| benzene/SVP B3LYP | 1.46 | J/K 1.30 (one int2e_ip1 call); XC 0.12 |
| caffeine/SVP B3LYP | 8.14 | J/K 7.48; XC 0.57 (dot_terms 0.50) |
| n-C20H42/SVP B3LYP | 28.26 | J/K 26.55; XC 1.39 (dot_terms 1.23); ints1e 0.15 |

**TDA/TDDFT** (5 states)

| case | wall | largest parts |
|---|---|---|
| TDA benzene/SVP B3LYP | 3.03 | XC response 2.35 (dot 1.22, rho 1.12); J/K 0.22; einsum 0.15 |
| TDA caffeine/SVP B3LYP | 25.25 | XC response 18.05 (dot 9.35, rho 8.70); J/K 5.56 (in-core, 160 trial dms); einsum 0.45 |
| TDDFT benzene/SVP B3LYP | 8.52 | XC response 6.57 (dot 3.50, rho 3.08); J/K 0.56; einsum 0.44 |
| TDA n-C20H42/SVP PBE | 36.79 | XC response 25.07 (dot 14.05, rho 11.02); J/K 10.13 (semi-direct J); einsum 0.56 |

**Hessians**

| case | wall | largest parts |
|---|---|---|
| benzene/SVP B3LYP | 17.87 | derivative J/K 11.34 (37 calls); CPHF XC 2.87 (dot_ao_mo 1.62, rho_lowrank 1.25); einsum 1.05 |
| benzene/SVP PBE | 15.64 | derivative J/K 9.71; CPHF XC 2.56; einsum 0.99 |
| caffeine/SVP B3LYP | 181.18 | derivative J/K 107.76 (73 calls: ipip1 ×1, ip1ip2 ×24, ipvip1 ×24, make_h1 ×24); CPHF XC 36.24 (dot_ao_mo 18.63, rho_lowrank 17.61); einsum 14.87 (8.8–11.3 of it in hess_elec); CPHF J/K 8.57 |
| benzene/SVP r2SCAN (NOTES It13) | 20.24 | J/K 10.0; CPHF XC 4.9 |
| caffeine/SVP r2SCAN (NOTES It13) | 199.4 | J/K 92.0; CPHF XC 66.3 |

**Grid pipeline and direct API**

| case | wall | largest parts |
|---|---|---|
| grid_caffeine_svp | 3.64 | eval_gto 1.70 (24 calls); _dgemm 1.32 (42); _contract_rho 0.41 (96) |
| micro_libxc | 1.74 | libxc 0.93; xc_deriv post-processing 0.76 (_unfold_gga 0.58, transform_xc 0.18) |

## 2. Ranked opportunities

**What is included.** Only findings that survived verification, are exact under the policy, and are not already done or rejected. Duplicates across areas are merged; their finding ids are in brackets.

**How to read the numbers.** Savings are the verifiers' 10-thread estimates, in seconds. Where several verifiers covered the same item, the range spans all of them.

**How it is ranked.** By expected saving across the bench workloads, weighted by confidence and divided by effort. This is a judgment, not a formula. #14 and #18 overlap #7, so their savings do not add to it.

Abbreviations:

| short | case |
|---|---|
| caf | caffeine/SVP |
| bz | benzene/SVP |
| alk | n-C20H42/SVP |
| caf+ UKS | scf_caffeine+_svp_b3lyp_uks |
| bz+ UKS | scf_benzene+_svp_r2scan_uks |
| DF-TZVP | scf_caffeine_tzvp_dfpbe |

| # | Opportunity [ids] | Workloads | Now (s) | Expected saving (s) | Exactness | Effort | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Hessian 2e J/K: compute int2e_ip1ip2 / int2e_ipvip1 only for the rows the pair loop reads (atoms ≤ A) [jk-hess-tri, deriv-hess-pairs] | caf and bz Hessians (all functionals); UKS, RSH | ip1ip2+ipvip1 ≈70–73 of 107.8 derivative J/K (caf); ≈6.5–6.7 of 11.3 (bz) | caf B3LYP −28 to −34 (16–18%); caf r2SCAN ≈−25; bz −1.3 to −2 each (more with #5) | bitwise | S | confirmed, adjusted |
| 2 | Hessian diagonal blocks from translational invariance (TI): drop int2e_ipip1 and the A=A blocks [deriv-hess-ti] | caf and bz Hessians; UKS, RSH | ipip1 ≈26 (caf); ≈2.5–3 (bz) | caf B3LYP −25 to −27 on top of #1 (#1+#2: 181 → ≈120–125); bz −3 to −4 (≈−5 with #1) | rounding level (diagonal residual ≤7e-12 abs, grows with natm) | M | confirmed |
| 3 | hess_elec: GEMMs instead of optimize=False 3-operand einsums and per-pair dm1 rebuilds [deriv-hesselec-blas, py-hess-elec-gemm, uks_df-hess-elec-einsum] | caf and bz Hessians; UKS, DF Hessians | 8.8–11.3 single-threaded (caf) | caf B3LYP and r2SCAN −8.5 to −11 each (5–6%); bz −0.2; UKS ×2 | rounding (6.5e-15) | S | confirmed ×3 |
| 4 | Per-file `-O3 -march=x86-64-v3 -ffp-contract=off` for libdft and libcgto's grid files [sys-cflags-libdft, td-fused-simd, flag part of xc-ao-simd-exp] | every XC workload | XC 5.2–7.0 (caf SCFs), 18.05 (caf TDA), 25.07 (alk TDA), 6.57 (bz TDDFT), 44.7 (caf Hessian) | caf SCFs −0.3 to −0.5 (3–5%); caf TDA −1.0 to −1.4; alk TDA −1.5 to −2; bz TDDFT −0.4 to −0.5; caf Hessian −4 to −6; r2SCAN 1–3% | bitwise | S | adjusted, confirmed |
| 5 | Adaptive AO block size (64 → 32/16) in CVHFnr_direct_drv when a call has few tasks [jk-par-blocks, deriv-jk-tasks] | bz Hessians, bz and caf gradients, caf Hessians | bz Hessian J/K 11.34 (B3LYP), 9.71 (PBE), scaling 3.2×; bz grad J/K 1.30 of 1.46; caf grad 7.48 | bz Hessians −4.5 to −6 each (25–35%); bz grad −0.6 to −0.8; caf grad ≈−0.5; caf Hessian −1 to −4 (after #1–#2) | rounding | S–M | confirmed ×2 |
| 6 | TDA/RPA/Casida XC response from the trial-vector factors via rks_fxc_mo [td-factored-response; NOTES Next #2] | caf TDA, bz TDA, bz TDDFT (alk: none) | XC response 18.05, 2.35, 6.57 | caf TDA −3.5 to −4.1 (14–16%); bz TDA ≈−0.9 (30%); bz TDDFT ≈−2.6 (30%); meta-GGA TD ≈2× on XC | rounding (≤2e-14 on XC) | M | confirmed |
| 7 | numpy's and scipy's bundled OpenBLAS pools spin 134 ms after each call. A/B, then set OPENBLAS_THREAD_TIMEOUT or OPENBLAS_NUM_THREADS (plus GOMP and malloc arms) in pyscf-env.sh and bench [xc-blas-spin, py-blas-spin, scf-blas-spin, sys-blas-pools, sys-xc-scaling] | all 10-thread SCFs, grid; TD and Hessian less | eigh 0.34–0.45 per caf SCF (≈0.26 overhead); get_overlap_cond + condense 0.37–0.55 (caf RKS), 1.02 (caf+ UKS); libxc 0.67–1.05 (2–3× the same calls after the unfused kernels) | caf SCFs −0.25 to −1.0 each; caf+ UKS −0.5 to −1.8; bz+ UKS −0.3 to −1.0; alk −0.3 to −0.5; grid −0.2 to −0.6. Unmeasured until the A/B. | bitwise (timeout); rounding (NUM_THREADS=1) | S (config) | confirmed, adjusted ×4 |
| 8 | Grid grouping: scalar box key + stable argsort instead of numpy.unique(axis=0) [py-grid-group-key, scf-grid-sort] | every scf_* (each Grids.build) | 0.17–0.19 (caf SCFs), 0.40–0.41 (alk SCFs) | caf −0.15 to −0.17 (1.5–1.7%); alk −0.37 to −0.43 (≈2%); bz+ UKS −0.06 to −0.1 | bitwise | S | confirmed ×2 |
| 9 | GGA density on the MO path via Z = C·v0 (2 GEMMs per group instead of 4); MO/dm path chosen by flop count [xc-rho-mo-gga-z] | caf PBE/B3LYP/CAM-B3LYP, DF-TZVP, caf+ UKS | rho kernel 2.83, 2.50, 1.81, 3.97, 4.58 | PBE, B3LYP −0.6 to −0.75 (5–7%); CAM −0.45; DF-TZVP −0.8 to −1.4; caf+ UKS −1.0 to −1.2 | rounding | M | confirmed |
| 10 | UKS in-core J/K: both spin dms through the one-pass fused kernel [jk-uks-incore, uks_df-incore-2dm-jk] | caf+ UKS; in-core UHF/ROKS hybrids | 5.50 of 21.16 (0.220 s/call vs 0.082 for RKS) | −1.4 (Python, 1-dm kernel per spin) or −2.3 to −3.0 (C, 2 dms per ERI pass; 11–14%) | rounding | S / M | confirmed, adjusted |
| 11 | Gradient J/K: s8 driver using 3 of 4 center derivatives, contracting straight into dE/dR [jk-grad-3of4; NOTES Next #1] | caf, alk, bz gradients | J/K 7.48, 26.55, 1.30 | caf −1.5 to −1.8; alk −5.3 to −6.5 (≈20%); bz ≈−0.25 | rounding | L | confirmed |
| 12 | One OpenMP kernel for wv = fxc·rho1·w over all sets [td-fxc-wv-kernel, py-fxc-wv-kernel, uks_df-fxc-wv-kernel] | TD cases, Hessians | einsum 0.44 (bz TDDFT), 0.15 (bz TDA), 0.45 (caf TDA), 0.56 (alk TDA); ≈1–1.5 (caf Hessian) | bz TDDFT −0.33 to −0.37 (≈4%); bz TDA −0.13 (≈4%); caf TDA −0.4; alk −0.45 to −0.5; bz Hessian ≈−0.27; caf Hessian −1 to −1.5 | bitwise if einsum's order is kept | S | confirmed ×2, adjusted |
| 13 | Hessian XC: build dR_rho1 from t; one grid pass for _get_vxc_deriv1 and _get_vxc_deriv2_dm; threaded atom-pair sums [deriv-xc-onepass, py-hess-xc-onepass] | caf and bz Hessians | ≈3–4.4 einsums + ≈0.9 duplicated pass (caf B3LYP); ≈0.6 (bz) | caf −3 to −4.5 (≈2%; more for r2SCAN); bz −0.4 to −0.5 (≈3%); step (a) alone −0.7 to −1.1 | rounding; step (a) bitwise | M (a: S) | adjusted ×2 |
| 14 | XC setup: call _fused_xc first, cache get_overlap_cond, skip unused condense [xc-setup-dup, py-xc-setup-dedup, scf-xc-setup] | all SCFs, TD | 0.37–0.55 (caf RKS), 1.02 (caf+ UKS), 0.30 (bz+ UKS), 0.25 (caf TDA) | caf RKS −0.28 to −0.45; caf+ UKS −0.67 to −0.9; bz+ UKS −0.2 to −0.25; caf TDA ≈−0.25. Only ≈−0.01 to −0.03 per SCF if #7 removes the contention. | bitwise | S | confirmed, adjusted ×2 |
| 15 | Derivative J/K contraction kernels: restrict pointers and register accumulators first, then fused per-call script sets [jk-hess-fused-scripts] | caf and bz Hessians, caf gradient | contraction ≈19–25 (caf Hessian), 2.1–2.6 (bz), ≈1.2 (caf grad) | caf Hessian −6 to −10 now, −3 to −5 after #1–#2; bz −0.7 to −1.1; caf grad −0.3 to −0.5 | bitwise (restrict) / rounding (fusion) | M (restrict: S) | adjusted, low confidence |
| 16 | GEMM_NBLK 4 → 2 for the fused kernels, with numint.py group_naos [sys-gemm-nblk] | SCF nr_rks/nr_uks | fused SCF kernels 5.2–7.0 (caf), 10.9 (DF-TZVP), 6.1–12.8 (alk) | caf −0.15 to −0.3 (2–3%); alk −0.3 to −0.5; DF-TZVP ≈−0.3; TD and Hessian <1% | rounding (≈6e-16) | S | adjusted |
| 17 | Fused fxc: keep the private nao² matrices across grid blocks, or use one block per call [td-fxc-one-block; NOTES Next #2] | alk TDA, caf TDA, dm-path CPHF | ≈0.3–2.6 (alk TDA, disputed); ≈0.1–0.3 (caf TDA) | alk TDA −0.3 to −2 (disputed, see §5); caf TDA −0.1 to −0.4 | rounding | S | adjusted; py duplicate refuted |
| 18 | LIBXC_eval_xc as one parallel region with dynamic chunks and no per-call malloc [sys-libxc-dynamic] | SCFs, especially bz+ UKS | libxc 0.67–1.05 (caf SCFs), 1.52 (caf+ UKS), 1.08 of 4.09 (bz+ UKS) | about half of: caf −0.45 to −0.6, caf+ UKS −1.0, bz+ UKS −0.6 to −0.8. Cause of the slowdown not yet known. | bitwise | S | adjusted |
| 19 | Deterministic dynamic schedule: slot buffers for the static(1) XC-matrix kernels; fixed chunks for NPdgemm [xc-dot-detdyn, sys-sched-hetero] | SCF, TD, grid | dot 2.66–4.78 (caf SCFs), 6.92 (DF-TZVP), 9.35 (caf TDA), 14.05 (alk TDA); NPdgemm 1.32 (grid) | 1–2.5% each: caf SCF −0.1 to −0.25; DF-TZVP −0.2 to −0.4; TDA −0.3 to −0.8; grid −0.05 to −0.15 | rounding, deterministic | M (NPdgemm: S) | adjusted ×2 |
| 20 | One-pass factored fxc kernel: ao·C_o shared or cached, sets batched per GEMM, wv built in C [td-lowrank-single-pass] | caf TDA, bz TDDFT; alk only with a recalibrated lowrank_cheaper | factored XC after #6: ≈14 (caf TDA), ≈3.9 (bz TDDFT) | caf TDA −1.1 to −2; bz TDDFT −0.5 to −0.8; alk 0 to −3 | rounding | L | adjusted |
| 21 | TD exchange from vvoo integrals built once when the ERIs are in memory [td-incore-mo-exchange] | caf TDA (B3LYP), bz TD | K ≈4.2 (caf TDA); J/K 0.56 and 0.22 (bz) | caf −1.3 to −2 (5–8%); bz TDDFT ≈−0.25; bz TDA ≈−0.1 | rounding | M | adjusted, low confidence |
| 22 | OpenMP in xc_deriv.c; a C kernel for transform_xc's additions [xc-deriv-omp] | micro_libxc; direct eval_xc_eff at deriv 2–3 | 0.76 of 1.74 | −0.45 to −0.55 (25–32%); direct deriv-2/3 calls 1.5–3×; SCF/TD <0.1 | bitwise | S | confirmed |
| 23 | Dense eval_rho: point-blocked GEMM with the contractions done in cache [xc-dense-rho-fuse] | grid (direct API) | contractions 0.41 + temporaries inside 1.32 of GEMM | −0.25 to −0.4 (7–11%) | rounding | M | confirmed |
| 24 | SIMD exp (libmvec or SLEEF) in AO evaluation [xc-ao-simd-exp] | grid; deriv-2/3 AO terms | eval_gto 1.70 (partly memory and page-fault bound) | grid −0.15 to −0.4 (with #4); gradients <2% | rounding (1–4 ulp) | M | adjusted |
| 25 | One-pass SCF XC group kernel, with libxc called from C [xc-onepass-scf] | caf SCFs, DF-TZVP | one AO pass ≈0.2–0.45 per caf SCF | −0.2 to −0.45 (+ ≤0.2); 3–5% | rounding | L | adjusted |

**Exact, but no bench workload yet.** Add a case before claiming a number.

| Opportunity [ids] | Workloads | Now | Expected saving | Exactness | Effort | Verdict |
|---|---|---|---|---|---|---|
| VV10 (NLC) kernel: one division per pair, SIMD, symmetric pair loop; same for the gradient and Hessian kernels [xc-vv10-pairs] | wB97M-V, wB97X-V, B97M-V, SCAN-rVV10 | ≈18–20 s per SCF cycle on caf (≈250–300 s per SCF) | −11 to −16 s per cycle (≈−150 to −250 s per SCF); grows as N² | rounding | M | confirmed |
| UKS per-spin factored XC response for CPHF and TD [uks_df-ucphf-lowrank-xc, td-uks-factored; NOTES Next #2] | UKS Hessians and TD | est. 105–114 s of CPHF XC (caf+ UKS B3LYP Hessian) | ≈−35 to −42 (B3LYP), −150 to −185 (r2SCAN) per caf+ UKS Hessian; UKS TD XC 1.3–2× faster | rounding | M | confirmed, adjusted |
| DF exchange of response dms from their factors; dsyrk in the DF-SCF K branch [uks_df-df-k-response-lowrank] | DF hybrid TDA/TDDFT/Hessians | 29–30% of a 1-thread bz DF-B3LYP TDA; est. 17–31 s of DF-K in a caf DF-B3LYP TDA | K per dm 4–5× (SVP), 8–10× (TZVP) faster; bz DF TDA −22% (measured); caf DF TDA ≈1.45–1.75× faster | rounding | M | confirmed |
| UKS response: J of the spin-summed dms (a two-line change in the non-hybrid branch) [uks_df-j-spin-sum] | pure-functional UKS TD; UKS Hessians and gradients | half of the response J (pure); ≈1/6 of the CPHF J/K (hybrid) | pure UKS TDA 8–15% (est.); hybrids 1–3% | rounding | S | adjusted |

## 3. Top 8 in detail

**#1 Hessian 2e J/K: compute only the rows that are read**
- **Change:** `hessian/rhf.py` `_partial_hess_ejk` (shls_slice at :142; int2e_ip1ip2 at :147, int2e_ipvip1 at :157), `hessian/uhf.py:158-203`, and the RSH long-range loops in `hessian/rks.py:89-116` and `uks.py:90-126`.
- **Evidence:**
  - The pair loop (rhf.py:171) reads the vj1/vk1 rows only for atoms B ≤ A. These rows are the k index of ip1ip2 and the j index of ipvip1, after the transposes.
  - The rows of later atoms are computed and then discarded, so each atom pair is computed twice.
  - Verifier rerun at 1 thread: benzene ip1ip2+ipvip1 23.6 → 13.2 s. ej/ek are bitwise identical with the default, permuted and subset atmlst, and with with_k=False.
- **Design:**
  - Set `bsh1 = max(aoslices[ja][1] for ja in atmlst[:i0+1])`.
  - Pass shls_slice `(shl0,shl1,0,nbas,0,bsh1,0,nbas)` to ip1ip2 and `(shl0,shl1,0,bsh1,0,nbas,0,nbas)` to ipvip1. vj1/vk1/veff shrink to (3,3,nb,nao).
  - About 15 lines per file. Prototype: `ejk_tri()` in `bench/results/profile-it13/swarm/prototypes/jk/hess_tri.py`.
  - The max over atmlst handles any atom list.
- **Risk:** the shortened calls for the first atoms have only a few 64-AO tasks. On benzene at 10 threads, only −1.3 to −2 s survives without #5: modelled ip1ip2+ipvip1 goes 6.74 → 5.26 s with 64-AO blocks, against 3.09 → 1.81 s with 32-AO blocks.
- **Validate:**
  - bitwise ej/ek at 1 thread;
  - `validate.py` hessian suite (RKS/UKS/RSH/DF) and caffeine, against pyscf-base;
  - time `hess_caffeine_*` and `hess_benzene_*`.

**#2 Hessian diagonal blocks from translational invariance (TI)**
- **Change:** when atmlst covers all atoms:
  - skip int2e_ipip1 (rhf.py:121-126);
  - restrict the per-atom ip1ip2/ipvip1 calls to atoms before A;
  - set `ej[A,A] = −Σ_{B≠A} ej[A,B]`, the same for ek, and per spin for UHF.
  - Keep e1 and all XC terms as they are: with no grid response, XC is not translation-invariant.
  - For RSH, split the long-range K diagonal out of veff_diag/vxc_dm.
- **Evidence:**
  - ipip1 costs ≈157 s at 1 thread (≈26 s at 10) on caffeine, and it only feeds ej[A,A] and ek[A,A].
  - Prototype `ejk_pairs.py` (together with #1): benzene `_partial_hess_ejk` 33.35 → 11.67 s.
  - Off-diagonal blocks and e1 are bitwise identical. Diagonal blocks match within 7.1e-12 for ej (|ej| up to 22.5) and 1.2e-12 for ek.
- **Risk:**
  - The residual is qcint's own TI error in the derivative integrals, and it grows with natm.
  - It is below the Hessian spread between builds that NOTES already records (6.9e-11 in It8, 1.5e-10 in It7), but larger than pure summation-order noise.
  - Keep the direct path for atmlst subsets.
- **Validate:** caffeine and the hessian suite against pyscf-base (policy 1e-9; validate.py's Hessian tolerance is 1e-7). Check UHF and RSH explicitly.

**#3 hess_elec with GEMMs**
- **Change:** `hessian/rhf.py:63-90` (the s1oo einsum at :73; the per-pair dm1/dme1 at :79 and :81) and `hessian/uhf.py:75-96`. Optionally switch solve_mo1's einsum (rhf.py:345) to lib.einsum.
- **Evidence:**
  - numpy.einsum without `optimize` runs `'xpq,pi,qj->xij'` as a naive 3·nao²·nocc² loop.
  - dm1 and dme1 are rebuilt for all 300 pairs although they depend only on ja.
  - Three verifier reruns measured 8.8–11.3 s, single-threaded, out of c_einsum's 14.87 s. The GEMM form takes 0.06 s, with a relative difference of 6.5e-15.
- **Design:**
  - Per atom: h1mo = h1ao·mocc; s1mo from atom A's rows only; s1oo = moccᵀ·s1mo plus its transpose.
  - Stack mo1 for all atoms as (3natm, nao·nocc). Form the h1, s1 and e1 terms as three GEMMs over all pairs.
  - Keep the lower-triangle-plus-mirror convention. Block over atoms if memory requires it.
- **Risk:** rounding only. Left as is, the cost grows as natm·nao²·nocc²: about 240–250 s extrapolated for an n-C20H42/SVP Hessian.
- **Validate:** Hessians against the current tree to about 1e-12 on the hessian suite (including DF and UKS) and on caffeine.

**#4 Per-file vector flags for libdft and the GTO grid files**
- **Change:**
  - In `lib/dft/CMakeLists.txt` (nr_numint_fused.c, nr_numint_sparse.c, nr_numint.c) and `lib/gto/CMakeLists.txt` (grid_ao_drv.c, deriv1.c, deriv2.c), add `-O3 -march=x86-64-v3 -ffp-contract=off`.
  - Leave libcvhf at `-O2`: It3 measured −4–6% on J/K with `-march=native`, and `-O3`/PGO makes the semi-direct kernels 42% slower.
  - For upstream: use target_clones or It7-style runtime dispatch, including the static inline helpers in nr_numint_gemm.h.
- **Evidence:**
  - The stock `-msse3 -O2` leaves the hot loops scalar: objdump shows no packed FP ops in `_contract_mo` or in VXCfused_dot_wv's OpenMP body.
  - Verifier rerun at 1 thread: nr_rks 1.10×, nr_rks_fxc (8 dms) 1.08×, rks_fxc_mo (36 dms) 1.21× faster. All results are bitwise equal to the shipped library.
  - Fallback without `-march`: `-fvect-cost-model=dynamic` alone is bitwise and gave −6% on the TD response.
- **Risk:**
  - The 10-thread transfer is unmeasured.
  - Machines without AVX2 need dispatch.
  - Keep FMA off: the build that contracts to FMA gives different results.
- **Validate:**
  - bitwise checks (`kernels.py`, `validate.py` FULL+SPARSE);
  - a 10-thread ABBA on `scf_caffeine_*`, `tda_caffeine`, `hess_benzene` and `grid_caffeine`. The eval_ao path with libcgto's own flags has not been measured yet.

**#5 Adaptive AO block size in CVHFnr_direct_drv**
- **Change:** `lib/vhf/nr_direct.c` (CVHFnr_direct_drv, CVHFnr_direct_ex_drv, and the keys_cache sizing at :262) and `nr_direct.h:22` (AO_BLOCK_SIZE 64).
- **Evidence:**
  - Tasks are the (j,k,l) blocks of 64 AOs, with the i blocks looped inside each task (nr_direct.c:398-430). Benzene (nao 114) therefore has only 6–8 tasks per call.
  - Measured 1→10-thread scaling:
    - benzene Hessian J/K 3.2× (36.7 → 11.34 s);
    - benzene gradient 2.6× (3.36 → 1.30 s);
    - direct SCF J/K, for comparison, 6.6×.
  - A task model reproduces both measurements. With 32-AO blocks it predicts 7.3× for the gradient, for +1.8–3% more 1-thread work.
  - It7 measured 32/16-AO blocks 3–5% faster for caffeine direct SCF; that change was never adopted.
- **Design:**
  - Start at 64 and halve, down to 16, while the number of non-empty tasks is below about 4–8 × omp_get_max_threads(). For s2kl, count k ≥ l blocks only.
  - Pass the size to CVHFshls_block_partition and CVHFallocate_JKArray, and size keys_cache from it.
- **Risk:**
  - Per-task overhead in calls that already have plenty of tasks; this is why the size is adaptive.
  - Rounding only: the write-back order is already dynamic.
- **Validate:**
  - Gradients and Hessians against the current tree (about 1e-12).
  - Time `grad_benzene`, `hess_benzene_*`, `grad_caffeine` and `hess_caffeine`.

**#6 TD/TDDFT XC response from the trial-vector factors**
- **Change:**
  - `scf/_response_functions.py`: give the singlet (:159-196) and triplet (:198-222) branches a `vind_lowrank`, as It13 did for the CPHF branch (:135-157).
  - `tdscf/rhf.py`: `_gen_tda_operation` (:91-105) and `_gen_tdhf_operation` (:1134-1167).
  - `tdscf/rks.py`: CasidaTDDFT (:89-104).
- **Evidence:**
  - The XC response is 71–78% of the TD walls on the target cases.
  - TDA's D = 2 C_v Zᵀ C_oᵀ enters only through its symmetric part A Bᵀ + B Aᵀ, with A = C_v Zᵀ and B = C_o. Only C_vᵀ V C_o is needed.
  - Verifier rerun (benzene B3LYP, 16 sets, 1 thread): 2.02 → 1.21 s. Errors: XC block 1.4e-14, TDA vind 3.2e-16, RPA vind 4.2e-16.
- **Design:**
  - Precompute fxc_st once.
  - The factor A_k is:
    - TDA: C_v z_kᵀ;
    - RPA: C_v (x_k+y_k)ᵀ, and the same block feeds both halves;
    - Casida: C_v (2 d_ia z_k)ᵀ.
  - B = Q = the active C_o; P = C_v.
  - Use rks_fxc_mo when lowrank_cheaper says so, otherwise the dm path. J/K and NLC stay in the AO basis.
- **Risk:**
  - No gain for the alkane: its flop ratio of 1.13 keeps the dm path. The first, small call also stays on the dm path.
  - At 10 threads a factored GGA flop costs 1.10–1.24× a dm flop (It13).
- **Validate:**
  - New tests like `test_cphf_xc_lowrank` for TDA, RPA and Casida, singlet and triplet, with SVWN, PBE, B3LYP and r2SCAN, against FUSED_XC=False to 1e-13.
  - The `validate.py` TD cases.

**#7 Thread-pool A/B (environment only)**
- **Change:**
  - `scripts/pyscf-env.sh` and `bench/common.py`. worker_env drops OPENBLAS_*, but cfg.env is applied afterwards, so the bench configs can set them.
  - Optional code changes, rounding-level:
    - replace the per-block `numpy.dot(den, exc)` (numint.py:1312, 1339-1340, 1940, 2069-2070) with a reduction that does not call BLAS;
    - use lib.dot in SCF._eigh and make_rdm1.
- **Evidence:**
  - numpy (OpenBLAS 0.3.34) and scipy (0.3.31) each bundle a pthread pool sized by OMP_NUM_THREADS=10. Idle workers spin on 8 nops for 2^28 TSC ticks (≈134 ms).
  - eigh, make_rdm1, DIIS and the per-block ddot wake the pools again in every cycle.
  - Per-call times in the 10-thread run against 1 thread:

    | call | 10 threads | 1 thread |
    |---|---|---|
    | eigh, n = 246 | 12.8 ms | 4.4–5.9 ms |
    | get_overlap_cond | 5.1 ms | 0.13 ms |
    | condense | 6.4 ms | 0.05 ms |

  - libxc costs 2–3× what the same calls cost after the unfused kernels (It11).
- **Arms:**
  - OPENBLAS_THREAD_TIMEOUT = 4, 16 and 20;
  - OPENBLAS_NUM_THREADS = 1;
  - GOMP_SPINCOUNT;
  - OMP_WAIT_POLICY = active;
  - MALLOC_TRIM_THRESHOLD_, MALLOC_MMAP_THRESHOLD_ and MALLOC_ARENA_MAX;
  - OMP_NUM_THREADS = 4 pinned to CPUs 0–7.
- **Risk:**
  - The verifiers disagree on the default.
    - TIMEOUT keeps eigh threaded. That matters at n = 872, where eigh takes 65 ms threaded against 163 ms single-threaded (+2.2 s on scf_c20h42_tzvp_pbe).
    - NUM_THREADS=1 is the only arm with a measured gain at n ≈ 250, and it changes rounding.
  - The "1.5–1.8× SCF-context penalty" visible in the profiles is largely an artifact of the concurrent profiling run.
- **Validate:**
  - ABBA on an idle machine on scf_caffeine_svp_b3lyp, scf_caffeine+_svp_b3lyp_uks, scf_benzene+_svp_r2scan_uks, scf_c20h42_svp_b3lyp, tda_caffeine_svp_b3lyp, hess_benzene_svp_b3lyp and grid_caffeine_svp.
  - Compare the per-call get_overlap_cond, eigh and libxc times, and xc_kernels.

**#8 Grid grouping key**
- **Change:** in `dft/gen_grid.py` arg_group_grids (:447-469), replace `numpy.unique(box_ids, axis=0, return_inverse=True, return_counts=True)` and the argsort of its inverse with:
  - `key = ((bx+1)*(ny+2) + (by+1))*(nz+2) + (bz+1)` as int64;
  - followed by `key.argsort(kind='stable')`.
- **Evidence:**
  - The structured-row sort (VOID_compare) is the single argsort entry in the profile: 0.17–0.19 s per caffeine SCF and 0.40–0.41 s per alkane SCF.
  - At 1 thread the replacement takes caffeine from ≈0.16–0.18 s to 0.012 s, and the alkane from ≈0.40–0.46 s to 0.032 s.
  - The permutation is identical on caffeine, the alkane, a level-5 grid and randomly displaced coordinates.
- **Why it is exact:** box ids lie in [-1, boxes], so the key increases strictly with the lexicographic row order. Its stable argsort therefore equals the stable argsort of unique's rank.
- **Validate:** numpy.array_equal of the two permutations; the `validate.py` SCF suite stays bitwise identical.

## 4. Approximations and opt-in ideas (the user's choice, never defaults)

- **Hessian derivative-integral screening (opt-in, e.g. `hessobj.direct_scf_tol`)** [jk-hess-screen adjusted; deriv-hess-screen refuted].
  - Why upstream is unscreened: `_make_vhfopt` is dead code (rhf.py:192 tests the unsuffixed names, also in pyscf-base), and make_h1 passes no vhfopt.
  - Effect of a 1e-14 Schwarz × density screen: J/K changes by ≤7.4e-13.
  - Saving:
    - caffeine Hessian −25 to −34 s alone, or −14 to −20 s after #1/#2;
    - 4–5× less derivative-integral work at n-C20H42 size;
    - nothing on benzene, where the +4.6% prescreen cost must be avoided.
  - Needs:
    - the suffix fix;
    - dm_cond computed once (for UHF over dm0, dm0a and dm0b);
    - a make_h1 prescreen that also tests D_ji and D_li;
    - the pppp q_cond offset fix in hessian_screen.c.
  - Status: the verifiers split on whether this fits the policy, and both analysts call it an approximation.
- **Davidson that adds only target-root vectors:** up to 4× fewer trial dms. It changes the Krylov space, the iteration count, and the energies at the conv_tol level.
- **Fewer CPHF iterations** through a looser tolerance or a level shift. Also: reusing the Krylov vectors' responses for the final fvind(mo1), which moves mo1 by 6.1e-7 relative.
- **Coarser second_grids, small_rho_cutoff, frozen core.** Frozen core would also let the alkane GGA TD take the factored path (flop ratio ≈0.74).
- **float32 storage of small integrals** in partially cached semi-direct runs (the TZVP SCFs).
- **The cutoff screen index for direct eval_rho without non0tab:** it keeps 54% of the pairs where 81% (AO-weighted) are exactly nonzero, so it changes the AO values.
- **Stratmann or screened Becke partitions**, and Becke pair screening.
- **Screening in-core ERIs below the Schwarz bound** (37% of caffeine/SVP's s8 tensor).
- **Skipping the conv_check SCF cycle, or a better initial guess:** both change cycle counts.
- **Exact configuration choices:**
  - in-core long-range ERIs for RSH: ≈−1 s of 10.7 s, for 3.7 GB more RAM;
  - SMT for the direct J/K only: 13% in It3;
  - a DF-UKS gradient flag that builds only the trace terms: ≈0.1–0.15 s, but CASSCF/CCSD need the off-diagonal terms.

## 5. Refuted or ruled out

**Refuted in verification**
- **deriv-hess-screen:** new screening relative to upstream's actual, unscreened results; moved to §4.
- **deriv-krylov-blas3, py-krylov-gemm:** ≈1.3–1.5 s of 181 s (<1%). Fine as a cleanup, but Hessians react to krylov changes at the 5e-7 level (NOTES).
- **deriv-uks-h1-j** (J of dm0a+dm0b in the UHF/UKS make_h1): correct, but <1% and there is no UKS Hessian bench. Fold it into the make_h1 edits of It14.
- **py-dfj-onepass:** 0.2–0.25 s (1.3–1.7%) on its only workload.
- **py-fused-priv-blocks:** it is NOTES Next #2 and #17 here.
  - The two verifiers' estimates for #17 differ: the td verifier gets −1 to −2 s from It13's ≈3.2 ms per dm per block, the py verifier ≈−0.3 s.
  - The py verifier assumed ≈5 sets per call, but the measured count is ≈13 (80 dms in 6 calls).
- **scf-becke-simd:** bitwise and 2.4× faster on the partition, but ≤1.2% of any wall, for M effort.
- **scf-eigh-evd:** ≤1.8% (on DF-TZVP), and it can change cycle counts (benzene+ UKS PBE: 21 vs 22 cycles).

**Ruled out by the analysts**

*J/K:*
- Cartesian-basis contraction of derivative integrals: 3.535 vs 3.483 s, no gain.
- In-core ERI fill: ≈1.7% at 1 thread.
- Semi-direct read kernels: bound by operation count (the inlining It9 tried did not help); sorting by shape works only for complete caches and is untested.
- TDA in-core K: close to the BLAS-3 bound; an RPA sym/antisym split keeps the same flops.
- MO-basis or low-rank K for the CPHF in-core J/K: needs an O(N⁵) transform (≈3.6e12 flops) that costs more.
- UKS direct batches padded from 2 to 4 dms: the same instruction count.
- ipip1 xy/yx duplicate components: ≈3% and needs a qcint change; moot with #2.
- A combined second-derivative qcint kernel: an L-effort autocode project.
- 3-of-4 TI for make_h1: ≈0.6 s on caffeine.
- DF J (0.28 s) and semi-direct index setup (≈1%): nothing left to gain.
- Partially cached TZVP SCFs: bound by qcint on the uncached quartets.
- Alkane gradient blocks: 288 tasks, already balanced.

*XC:*
- libxc static split: max/mean 1.02–1.05 on equal cores; the core-heterogeneity part is ≤0.05–0.1 s.
- libxc cost per point does not depend on the density (real and random densities cost the same).
- Exact-zero screening for direct eval_rho: at most 34% of the flops, eaten by masks and gathered GEMMs.
- eval_ao page faults on fresh arrays: fixable only on the user side (`out=`).
- Becke partition: 0.10–0.37 s; exact pair skipping applies only to the innermost shells.
- Small fixed costs: cache_xc_kernel 0.07 s per run; per-call array checks 0.03 s per SCF; deriv-1 xc_deriv 0.03–0.12 s.
- The GGA Vxc GEMM and the meta-GGA τ dsyrk already use the minimal flops.
- The dense SCF path is used only by water-sized molecules.

*TD:*
- Davidson and eigensolver: ≤2% everywhere (1.1% on caffeine TDA).
- MO-basis (ia|jb) for J: break-even at ≈280 dms (160 are used). Low-rank K: worse below nocc/2 dms.
- Orbitals on the grid: costs at least as much as the dm path (nvir ≥ ns). The full MO kernel matrix: ≈5.8e13 flops.
- Factored path for the alkane GGA as it stands: 0.93×; the model correctly keeps the dm path.
- Small items: batching the dm-path GEMMs ≤5%; the nr_rks_fxc_st fxc rebuild 0.06 s; vind projections ≈0.1 s.

*Gradients and Hessians outside J/K:*
- No HDF5 or chkfile traffic in this tree.
- Small items: solve_mo1 einsum 0.05 s; reduce association in vind 0.2–0.4 s; per-pair hcore_deriv ≤0.4 s; shared fxc 0.04 s; gen_hop is unused.

*UKS and DF:*
- UKS SCF XC, libxc and direct J/K are at parity with RKS. The spin-polarized libxc cost is inside libxc, and 25 vs 17 cycles is chemistry.
- UKS Hessian XC is at parity apart from ≈1.8 s of per-atom wv einsums. The UKS derivative K scripts add ≈7%.
- nr_uks_fxc dense batching (NOTES Next #3): only water-sized molecules reach it.
- DF-PBE TZVP: nothing DF-specific left.
- DF Hessian: not measured (no profile, and a 1-thread benzene run exceeds 2 minutes).

*Python and system:*
- Negligible: chkfile ≤0.02 s per SCF; logger, timers and ctypes glue; libc memset/memmove, which are small and sit inside other kernels.
- Pure-numpy rewrites of the atom-pair sums and the fxc contraction: slower, or only 1.4×.
- OPENBLAS_CORETYPE=Haswell or Zen: +3–7% on the main shapes, but −7–11% on others and 1.6× slower on one.
- BLIS/AOCL: at most 1.2× on the BLAS share, and not installed.
- One GEMM over all dms of a group: 0.98–1.06×.
- `-O3` or PGO for libcvhf: 42% slower.
- SMT (20 threads): 0.67–0.80× (It3).
- jemalloc, mimalloc or THP: no expected change.
- libxc built with `-march`: bitwise neutral, and single-thread libxc is only 5–8% of nr_rks.
- LTO on qcint: nothing (It8).
- OMP_WAIT_POLICY=passive: adds futex wake-ups at every region; `active` is one of #7's arms instead.

*SCF driver:*
- Setup (mol.build, overlap, hcore, minao guess) and the per-cycle driver work (make_rdm1, get_grad, DIIS) cost milliseconds each.
- Precomputing (S·X)ᵀ for DIIS would save ≤0.2 s.

## 6. Proposed order

Run every timing on an idle machine (see the caveat in §1).

1. **First, no code: the runtime A/B (#7).**
   - Adopt the winner in `scripts/pyscf-env.sh` and `bench/common.py`, then re-profile to get a new baseline.
   - The result decides whether #14, #18, #19 and #25 are still worth doing.
2. **It14: Hessian derivative J/K and hess_elec.**
   - Files: `hessian/{rhf,uhf,rks,uks}.py`, `lib/vhf/nr_direct.{c,h}`.
   - Order: #1, then #2, then #5, then #3. Add step (a) of #13 and deriv-uks-h1-j while editing these files.
   - Expected (sums of verified ranges):
     - caffeine B3LYP Hessian 181 → ≈105–119 s;
     - benzene Hessian J/K 11.3 → ≈2.4–3.5 s (analysts' task models);
     - benzene gradient −0.6 to −0.8 s; caffeine gradient ≈−0.5 s.
3. **It15: XC setup, grid build and libxc interface.**
   - Files: `dft/numint.py` (nr_rks/nr_uks/fxc setup), `dft/gen_grid.py`, `lib/dft/libxc_itrf.c`, `lib/dft/xc_deriv.c`.
   - Items: #8, #14, #18 and #22.
   - Do #18 only after a 10-thread diagnosis: time LIBXC_eval_xc standalone and right after VXCfused_eval_rho_mo.
4. **It16: fused XC kernels.**
   - Files: `lib/dft/CMakeLists.txt`, `lib/gto/CMakeLists.txt`, `nr_numint_fused.c`, `nr_numint_gemm.h`, `numint.py`.
   - Items: #4 (its own commit, bitwise), then #9 (update test_fused_xc), #16 and the NPdgemm part of #19. Optionally #24, which touches the same libcgto files.
   - The gains overlap, so measure each step on top of the previous one.
5. **It17: TD response.**
   - Files: `scf/_response_functions.py`, `tdscf/rhf.py`, `tdscf/rks.py`, the fxc paths in `numint.py`, and the dot kernels in `nr_numint_fused.c`.
   - Items: #6, #12, #17; then #20 with a recalibrated lowrank_cheaper; #21 only if K still matters.
   - Expected (sums of verified ranges):
     - caffeine TDA ≈−4 to −5 s;
     - benzene TDDFT ≈−3 s;
     - alkane TDA −0.75 to −2.5 s.
6. **It18: UKS.**
   - Files: `lib/vhf/nr_incore_batch.c`, `scf/_vhf.py`, `dft/uks.py`, `_gen_uhf_response`, `hessian/uhf.py`, `tdscf/uhf.py`.
   - Order: #10 first. Then add bench cases for a caffeine+ UKS Hessian and a caffeine+ UKS TDA. Then the UKS factored response and J of the spin-summed dms (rows 2 and 4 of the no-bench table), reusing It17's kernels.
7. **It19: derivative J/K drivers.**
   - Files: `lib/vhf/nr_direct.c`, `nr_direct_dot.c`, `grad/rhf.py`, `grad/rks.py`.
   - Items: #11, and #15 with its bitwise restrict step first.
8. **Later, by priority:**
   - Add `scf_caffeine_svp_wb97mv`, then the VV10 kernel.
   - Add a DF hybrid TDA or Hessian case, then the DF-K factored response.
   - #13 steps (b) and (c).
   - The slot-buffer part of #19, together with #17's buffers.
   - #23.
   - #25, only if a profile after It16 still shows it at ≥3%.
   - The opt-in Hessian screening (§4), with its upstream bug fixes, as a separate change.