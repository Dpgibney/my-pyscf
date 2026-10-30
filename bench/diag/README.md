# Diagnostics behind the numbers in NOTES.md

One-off measurement and check scripts. They are kept so that the numbers quoted in `NOTES.md` and in the
`perf` commit messages can be reproduced.

They are records, written against the trees as they were at the time. Some patch PySCF internals of
that state, for example to force one code path.

Run them in a configuration's pinned environment:

    bench/diag/run_in.py G=pyscf bench/diag/switch_sweep.py caffeine B3LYP

| script | measures or checks | NOTES.md section |
|---|---|---|
| `run_in.py` | runs a script in a bench configuration's pinned environment | – |
| `grid_probe.py` | grid settings and point counts (wheel vs master) | Final comparison |
| `prune_check.py`, `prune_driver.py`, `prune_check2.py`, `prune_driver2.py` | wheel vs master benzene Hessian at tight convergence, with `small_rho_cutoff` and 2.12.1's CPHF Krylov calls | Final comparison: accuracy |
| `blk_probe.py` | `block_loop` block sizes and the sparse/dense choice per call | It3 memory configuration |
| `syrk_bench.py` | `dsyrk` vs per-component GEMM for the meta-GGA τ term | It4 |
| `r2scan_conv.py` | tight r2SCAN SCF convergence with and without the `dsyrk` path | It4 |
| `vxc_deriv2_dm.py` | prototype of `hessian.rks._get_vxc_deriv2_dm` against the original | It5a |
| `sparse_switch.py` | `nr_rks` with the `block_loop` switch forced each way (old sparse kernels) | Final comparison: large cases |
| `sparse_profile.py` | where `nr_rks` spends its time on the sparse path | It6: why |
| `sparsity_stats.py` | screen-index nonzero fraction and gathered/dense flops | It6: why, D |
| `sparse_gemm_check.py` | gather kernels vs dense products on real AO data, and their timings | It6: kernel timings |
| `fused_check.py`, `fused_rho_check.py` | fused and rerouted sparse paths vs the unfused ones and dense products | It6: B, C |
| `sparse_vs_dense_exact.py` | sparse-path Vxc and gradient XC term vs the same build forced dense | It6: exactness |
| `determinism.py` | bitwise repeatability of `nr_rks` on the alkane | It6: A |
| `grad_vxc_time.py` | `nr_rks` and the gradient XC term per call on the alkane | It6: per-commit table |
| `switch_sweep.py` | dense vs screened path per system, interleaved | It6: the 90% switch |
| `sgx_invariance.py`, `sgx_invariance_zero.py`, `sgx_bisect.py` | SGX gradient translational invariance, and which build or kernel changes it | It6: upstream tests |
| `scf_tight_trace.py`, `scf_tight_trace_unfused.py` | per-cycle trace of the tight SVP alkane SCF | It6: harness problems |
| `scf_tzvp_trace.py` | per-cycle trace of the tight TZVP alkane SCF (the direct-SCF drift) | It6: harness problems |
| `td_residual_trace.py` | TDA eigensolver residuals per iteration, optionally with another `lindep` | It6: harness problems |
| `td_operator_noise.py` | linearity and symmetry of the TDA response operator in a build | It6: harness problems |
| `eval_ao_cost.py` | zero-fill share of `eval_ao` on the screened path (not yet run) | Next iterations |
| `jk_profile.py` | J, K and J+K build times, in memory and direct, 1–20 dms | It7: where J/K time went |
| `jk_native_profile.py`, `grad_jk_native.py` | leaf functions of one J/K build, energy or gradient (py-spy `--native`, 1 thread) | It7: where J/K time went |
| `jk_scaling.py` | thread scaling of a direct J+K build | It7: where J/K time went |
| `td_jk_calls.py` | the J/K calls of a TDA run: number of dms, hermi, time | It7: where J/K time went |
| `incore_batch_check.py`, `direct_batch_check.py` | batched vs per-dm J/K: differences and timings | It7: A, B |
| `td_conv_sensitivity.py`, `td_jk_operator_diff.py`, `td_operator_dump.py`, `td_vind_internals.py`, `td_fixed_scf.py`, `scf_endpoint_spread.py` | a TDA validation failure traced to the scatter of the tight SCF endpoint | It7: harness |
| `dfj_cost.py` | DF J of a pure functional by `df_jk.get_j` vs from `_cderi` | It7 |
| `scf_jk_one_dm.py` | one SCF density through direct J/K: per-dm kernels vs the batched driver | It7: C |
| `semidirect_stats.py`, `semidirect_count.c` | Schwarz-significant quartets of a molecule, and which of them each direct SCF call evaluates: sizes of a semi-direct cache and its coverage per threshold | It9: why |
| `semidirect_profile.py` | leaf functions (py-spy, 1 thread) or wall times of semi-direct J/K calls that read stored integrals | It9: read path |
| `ao_radial_cache.py`, `ao_radial.c` | memory and per-pass time of caching AO values vs shells' radial factors vs `eval_ao`, in `block_loop` blocks and fused 56-point blocks; checks bitwise equality (build line in `ao_radial.c`) | AO-value cache test |
| `screened_unwritten_check.py` | `eval_ao` leaving screened AO values unwritten: every opted-in XC path with zeros, unwritten and NaN-filled values (bitwise), and `nr_rks`/SCF timings | It10: A |
| `fused_xc.py`, `fused_xc.c` | prototype `nr_rks` with AO values computed inside the It6 gather kernels (optionally from the radial cache): bitwise check against `nr_rks` and per-call timings (build line in `fused_xc.c`) | It10: B |
| `fused_numint_check.py` | NumInt's fused XC kernels (`_FusedXC`) against the `block_loop` path, bit for bit, for every opted-in function and both density paths, with and without the radial cache and with split XC matrices; per-call timings | It11 |
| `deriv_xc_time.py` | time of the gradient and Hessian XC terms per call and the share of `eval_gto` in it | It12: why |
| `deriv_xc_check.py` | the gradient and Hessian XC terms (RKS caffeine, UKS benzene⁺; SVWN, B3LYP, r2SCAN) against references saved from an earlier tree, with times | It12 |
| `fxc_blocks.py` | `nr_rks_fxc` time per dm on n-C20H42/SVP (LDA) vs the number of dms and `max_memory`, i.e. vs the fused kernels' grid block count | It13: A |
| `fxc_tree.py` | `nr_rks_fxc` of 3·natm CPHF-type dms on n-C20H42/SVP in any tree (upstream comparison) | It13: A |
| `lowrank_fxc_check.py` | the CPHF XC response from the factors (`_FusedXC.rks_fxc_mo`) against `nr_rks_fxc` projected on the orbitals: differences, interleaved times, the flop model's ratio and choice, and `vind_mo` against the projected `vind` | It13: B |
| `j_repro.py` | two identical semi-direct J builds and response calls on caffeine/TZVP, compared in the AO and the MO basis | It13: exactness |
| `native_profile.py` | 1-thread `py-spy --native` profile of a bench case: self and inclusive time per native and Python function (retries py-spy's occasional failure to attach) | profile-it13 |
| `profile_summary.py` | readable summary of a `run_bench.py` results file: wall time, categories and top functions per case (`BENCH_TOP=30` in the config env records 30) | profile-it13 |
| `blas_pools.py` | CPU that numpy's and scipy's bundled OpenBLAS pools burn after a call, what that costs a PySCF OpenMP call made next (J+K, `get_overlap_cond`, `condense`), and eigh/GEMM times at three sizes; run once per `OPENBLAS_*` setting | thread pools |
| `hess_ejk_check.py` | the Hessian's two-electron blocks (RHF/UHF, J only, range-separated K) and `hess_elec`'s mo1 terms against copies of the code before It14, for all atoms, reversed and subset atom lists; 1 thread | It14 |
