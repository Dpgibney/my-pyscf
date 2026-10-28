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
