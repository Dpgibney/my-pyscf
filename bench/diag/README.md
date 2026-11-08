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
| `xc_deriv_check.py` | `transform_xc`, `ud2ts`, `ts2ud` and the fg kernel against the code before It15 (old Python driving the old `libdft` from `pyscf-it13`), bit for bit, plus transform times at 1M points | It15: B |
| `grid_group_check.py` | `arg_group_grids` against the `numpy.unique(axis=0)` version: identical permutations on real and displaced grids, and times | It15: A |
| `td_fxc_calls.py` | the response calls of a TD run: trial vectors per call, XC and J/K time per call, and the factored path's flop ratio and choice | It17: why |
| `td_lowrank_check.py` | TDA, TDDFT and Casida operators (singlet, triplet) and the CPHF operator with the XC response from the factors against `FUSED_XC=False`, both paths of the flop model forced; TDA time per call | It17: A |
| `td_cphf_bitwise.py` | the TD operators on the dm path and the CPHF operator on fixed orbitals, saved in one tree and compared bit for bit in another | It17: A |
| `fxc_wv_check.py` | `numint._fxc_wv` (`VXCfused_fxc_wv`) against the per-set numpy expressions it replaced, bit for bit, with times | It17: B |
| `lowrank_kernels_bitwise.py` | `VXCfused_eval_rho_lowrank` and `VXCfused_dot_ao_mo` against another tree, bit for bit (the refactor into shared helpers); one-kernel vs two-kernel `rks_fxc_mo` vs `nr_rks_fxc` | It17: C |
| `lowrank_paths_time.py` | per call, 1–40 TDA-type dms: dm path, factored in two kernels and in one kernel, with the flop model's ratio and choice | It17: C, flop model |
| `gemm_shapes.c` | dgemm throughput, one call per OpenMP thread, at the factored kernels' shapes (one set vs sets batched along N); build line inside | It17: C |
| `fused_sched_toggle.py` | experiment only: switches the accumulating fused kernels between `schedule(static, 1)` and `schedule(dynamic, 4)` in the `pyscf` tree | It17: scheduling |
| `sched_determinism.py` | the kernels that sum partial matrices (fused and unfused `nr_rks`, `nr_rks_fxc`, `rks_fxc_mo` in one and two kernels, gradient `get_vxc`): repeated calls bit for bit, and against another tree on the same orbitals | It18 |
| `npdgemm_check.py` | `lib.dot` (`NPdgemm`) for shapes that take the m-, n- and k-splits: against another tree, bit for bit or the relative difference, with times | It18 |
| `dot_wv_time.py` | `_FusedXC.dot` (`VXCfused_dot_wv`) per call on the SCF's blocks and on the whole grid, 1, 2 and 20 sets | It18 |
| `sched_sim.py` | simulated makespan of static, dynamic, fixed-slot and stream schedules on 4 fast + 6 slow threads, and of the stream schedule's stickiness margin | It18 |
| `incore_ndm_time.py` | in-memory J/K of 1–8 symmetric dms per call: the tree's routing, the single-dm kernel once per dm, and the BLAS-3 batch kernels; against the per-dm kernels of `nr_incore.c`, and the tree against the single-dm kernel bit for bit | It19: A |
| `lowrank_uks_time.py` | per call, 1–40 UKS TDA-type spin dm sets: the dm path (`nr_uks_fxc`) and the factored path (`uks_fxc_mo`), with the flop model's ratio and choice and the difference between the paths | It19: C |
| `uks_lowrank_edge.py` | the UKS factored response with no beta electrons (H2+, H3 quartet) against `FUSED_XC=False`, and the response functions ROKS gets | It19: C |
| `make_h1_time.py`, `h1_jk_scripts.py` | the UKS Hessian's `make_h1` between trees, and its J/K call with Coulomb scripts per spin vs on the total density | It19: D |
| `hess_cpscf_tol.py` | benzene+ UKS Hessian on fixed orbitals at two CPHF tolerances, between trees and run to run | It19: end to end |
| `grad_ejk_check.py` | `grad.rhf.jk_energy_per_atom` against the J/K matrices of `get_jk`/`get_j`/`get_k` contracted as `grad_elec` does (RHF, UHF, J or K alone, range-separated K), and the times of both | It20: B |
| `grad_ejk_stats.py` | orbits and integral evaluations of `CVHFgrad_jk_energy` against the `int2e_ip1` evaluations of the s2kl driver for the same density | It20: B |
| `grad_ejk_paths.py` | whole gradients with `rhf_grad.JK_ENERGY` on and off for every SCF type with the J/K energy path, and density fitting, which keeps the matrices | It20: B |
| `grad_jk_native.py` (`ejk` argument) | leaf functions of `jk_energy_per_atom` instead of `get_jk` | It20: B |
| `deriv_jk_kernels.py` | the derivative J/K calls of the gradient, `make_h1` (RKS and UKS scripts) and `_partial_hess_2e` (`int2e_ip1ip2`, `int2e_ipvip1`): matrices saved to compare trees bit for bit, and times | It20: A |
| `deriv_jk_scripts.py` | marginal cost of each contraction script of the Hessian's derivative J/K calls | It20: A |
| `scf_guess_study.py` | SCF cycles, energies and per-cycle traces per initial guess and DIIS class for the molecules of `bench/geom` and the bench set (JSON lines) | It21 |
| `scf_diis_variants.py` | SCF cycles of DIIS variants (subspace size, start cycle, ADIIS+CDIIS blends and switching, the `pre_scf` guess), also for `bench/geom/hard`, water stretched, HF | It21 |
| `scf_proj_guess.py` | a guess from a cheaper SCF (smaller basis, coarser grid, density fitting, another functional) projected onto the target basis: cycles and times against minao | It21 |
| `scf_hard_trace.py` | caffeine+ UKS per cycle: frontier orbitals, their mirror parity, occupied-space overlap with the previous cycle | It22 |
| `scf_hard_states.py` | caffeine+ UKS solutions: Cs π hole and σ hole with fixed irrep occupations, the broken-symmetry one by SOSCF, internal stability | It22 |
| `scf_hard_remedies.py` | caffeine+ UKS from minao with each convergence remedy (level shift, damping, SOSCF, smearing, guesses) | It22 |
| `scf_flipmix.py` | flip-pair mixing prototype, and rotations of the π-hole state's hole pair | It22 |
| `scf_flip_scan.py` | where the flip-pair trigger fires on open-shell molecules under plain CDIIS | It22 |
| `mem_parts.py` | where a RIJCOSX SCF's memory goes: RSS by phase (J, K, XC, VV10), before and after each call, and the largest live allocations; wB97X, wB97M-V, without VV10, VV10 on its old path | RIJCOSX: wB97M-V's extra memory |
| `vv10_fused_check.py` | `nr_nlc_vxc` on the fused kernels against its `block_loop` path, for a tagged, an untagged and a non-symmetric dm | RIJCOSX: wB97M-V's extra memory |
| `vv10_fused_grad.py` | the gradient's VV10 term (`grad.rks.get_nlc_vxc`) both ways: matrices, times and the peak RSS of each call | RIJCOSX: wB97M-V's extra memory |
| `rijcosx_block_loop_vv10.py` | runs `rijcosx.py` with `nr_nlc_vxc` on its `block_loop` path (the A side of the A/B) | RIJCOSX: wB97M-V's extra memory |
| `hess_mem.py` | an RKS Hessian's memory and time by phase (SCF, the partial Hessian's 2e/XC/VV10 terms, make_h1, the CPHF's J/K, XC and VV10 responses), RSS sampled every 0.1 s | The wB97M-V Hessian: VV10's terms rewritten |
| `vv10_hess_parts.py` | the three VV10 Hessian terms (energy second derivative, Fock derivative, response): times split into C kernels and einsum, outputs saved to compare trees (`--compare`) | The wB97M-V Hessian: VV10's terms rewritten |
| `vv10_hess_ref.py` | whole Hessians and TDA/TDDFT energies with VV10 in the response, saved to compare trees | The wB97M-V Hessian: VV10's terms rewritten |
| `cphf_calls.py` | a Hessian with the CPHF's fvind calls and vectors counted, saved to compare trees | The wB97M-V Hessian: low-hanging fruit |
| `cphf_residual.py` | the true residual of the CPHF solution a Hessian takes; `KRYLOV_NORM=1` swaps in a normalized Krylov solver | The wB97M-V Hessian: CPHF stops short for DFT |
| `cphf_warm.py` | the Hessian's CPHF cold and warm-started from a cheap response (DF-J + SGX-K level 0, XC level 1, no VV10): exact calls, times, true residuals | The wB97M-V Hessian: low-hanging fruit, 5 |
