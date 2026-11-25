# The UKS transition-state workflow at production size: logs of the audit, 2026-11-24/25

NOTES.md, "A UKS transition-state Hessian at production size: the audit". Laptop, 10 threads unless a log says
otherwise; wB97M-V/def2-TZVP, XC (99,590), VV10 (50,194), SMD water, `two_level`, response grids 2/1,
`grid_response`, `sgx_response`. The logs are condensed (`bench/diag/wb97mv/uks_ts_audit/condense.sh`): the `>>>`
lines of `ts_workflow_profile.py` and of the probes (`[seconds since the start, resident memory]`), the tables of
parts (nested parts overlap), the CPHF cycles.

Molecules (`bench/diag/wb97mv/uks_ts_audit/`): `fmnh_hnco.xyz` (57 atoms, doublet, 1198 functions),
`meophenoxyl.xyz` (16 atoms, doublet, 321 functions), `meophenol.xyz` (17 atoms, closed shell, 327 functions).

| file | what |
|---|---|
| `small_uks_pushed.log`, `small_rks_pushed.log` | the small workflows with tag perf-2026-11-24b (SCF, gradient, Hessian, RIJCOSX steps) |
| `small_uks_cphf_h1_only.log` | the small radical after the first two changes (the UKS CPHF response and Fock derivative from factors) |
| `small_uks_final.log`, `small_rks_final.log`, `small_uks_final_rij.log`, `small_uks_svp.log` | the small workflows with the final code (tag perf-2026-11-25); the radical with RI-J in the response, and in def2-SVP |
| `big_cphf_h1_only_stopped.log` | 57 atoms, the tree with the first two changes: the SCF and the Hessian up to the solvent's Fock derivative, where the guard stopped it at 23 GB; a 2-thread job ran beside it |
| `mem_solvent_h1_before.log`, `mem_solvent_h1_after.log` | `mem_h1.py`, 57 atoms, 8 threads: the resident memory of the solvent's Fock derivative (and after, of its Hessian terms), with the lines of the main thread at every new high |
| `dd_small.log`, `dd_big.log` | the two-electron second derivatives with the screening by one and by two density elements (16 and 57 atoms) |
| `h1jk_big.log` | 57 atoms: the Fock derivative's J/K of two atoms at `MAKE_H1_TOL` 1e-14 and 1e-12; the long-range K second derivatives with the two-element bound |
| `tol_scan_uks.log`, `tol_scan_rks.log` | the Hessian against the two thresholds together (one-element bound; coarse grids, gas phase). 2 threads beside other jobs: the times mean nothing |
| `cphf_small.log`, `cphf_big.log` | `ts_workflow_profile.py --steps cphf --cphf-rij`: one call of the CPHF's response function by its parts, and the RI-J of the same densities |
| `xc_h1_small.log`, `xc_h1_big.log` | `xc_h1_big.py --enlc`: the XC Fock derivative, the VV10 energy second derivative and the VV10 Fock derivative by their parts |
| `rx_small.log`, `rx_big.log` | `ts_workflow_profile.py --steps rx --rx-profile`: the RIJCOSX SCF, the scanner's steps, and one step under cProfile |
| `solv_cmp.log` | `solv_cmp.py`: the solvent's gradient, Fock derivative and Hessian terms of the final code against that tag's, four solvent models |
| `cmp_final.log` | `hess_ref.py`: 9 UKS and 9 RKS Hessians of the final code against that tag's |
| `ts_example_variants.log` | `examples/wb97mv_ts.py` with each of its switches, all of them, and SMD, on HCN <-> HNC and on H + H2 (doublet); the frequencies at one geometry with the response approximations |
| `tests.log` | the unit tests of the final code |
