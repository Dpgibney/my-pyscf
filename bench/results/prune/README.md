The two prunings not in use, measured on the Hessian of `examples/wb97mv_ts.py`'s model (laptop, 10 threads,
2026-11-26; NOTES.md, "Pruning: low-density grid points and a VV10 pair cutoff, measured on a Hessian"). Tools in
`bench/diag/prune/`. Logs condensed to the script's own lines, the timing tables and the SCF/CPHF summaries.

- `meophenoxyl_opt.xyz`, `opt.log`: the 4-methoxyphenoxyl radical at the model's minimum (geomeTRIC, 10 steps).
- `radical_hessians.log`: the reference SCF and gradient, then one Hessian per variant (ref; low-density cutoffs
  1e-7 to 1e-4; 1e-6 with the response grids too; the VV10 pair cutoff at none, 16, 12, 9, 6 Bohr; ref2), each with
  the time of its parts, and the comparison with ref at the end.
- `radical_control.log`: the control rho:0 (an SCF from ref's density, nothing pruned, and its Hessian).
- `radical_summary.txt`: `summarize.py` on the runs (against ref and against rho:0; the time by kind of work).
- `radical_kernels.log`, `fmnh_hnco_grids.log`: `prune_check.py --big` for the radical and the audit's 57-atom
  doublet: the points each cutoff keeps on every grid (initial guess's density), one XC potential on the full and the
  pruned grids (alternated), and the VV10 kernel `eval_f_t` with and without the pair cutoff.
- `power.txt`: `scripts/check_power.sh` at the start ('balanced': timings compared within a run).
- Two saddle points with soft modes (`prune_check.py --ts`, then the variants): `ch3_ch4_*` (CH3. + CH4, 9 atoms) and
  `meo_meoh_*` (CH3O. + HOCH3, 11 atoms): `*_guess.xyz`, `*_search.log` (the start Hessian and geomeTRIC's steps),
  `*_ts.xyz`, `*_hessians.log` (ref, rho:0, rho:1e-7 to 1e-4, vv10:9/6/4), `*_summary.txt` (with the imaginary and
  soft modes). `aro_ch4_guess.xyz`, `aro_ch4_search_stopped.log`: ArO. + CH4, stopped after 23 steps (no saddle
  point of the ground state along that path).
- `examples_ab.txt`: both TS examples with `mf.small_rho_cutoff = 1e-7` against the committed ones (eight variants of
  `ts_example_variants.sh` on HCN <-> HNC and H + H2; CPU on the laptop, GPU on node1), compared by
  `compare_runs.py`, and the points that remain at the saddle points (`count_points.py`, `count_points_gpu.py`).
