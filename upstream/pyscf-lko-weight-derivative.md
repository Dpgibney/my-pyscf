# PySCF: the derivative of the LKO grid weights is wrong (`VXCgen_grid_lko_deriv`)

Found 2026-11-25 while writing the GPU's RIJCOSX gradients (`NOTES.md`, "GPU4PySCF, part 42"). Fixed in this
workspace's PySCF fork (branch `perf`, tag `perf-2026-11-25c`); **not yet reported upstream**.

## Where
- `pyscf/lib/dft/grid_basis.c`, `VXCgen_grid_lko_deriv` (line 410 of upstream master 44313dfb4, 2026-10-26; also in
  the PySCF 2.14.0 wheel):

  ```c
  atom_dist[i*natm+j] = 1 / _lko_sat_func(dr);
  sat_deriv = _lko_sat_deriv(dr) * pow(atom_dist[i*natm+j], 3);   // wrong
  dadx[i*natm+j] = -dx * sat_deriv;
  ```
- Introduced with the LKO partition's derivative in "Faster SGX and SGX Gradients" (pyscf/pyscf#3076, 2026-04-20).
- Used by `pyscf.grad.rks.get_dw_partition_sorted` and `grids_response_lko`, so by
  - the gradients of SGX models with `optk` (`mf.COSX()`, `mf.RIJCOSX()`: `pyscf/sgx/grad/rhf.py`,
    `get_k_grad_only` -> `scalable_grids_response_cc`), whose grids use the LKO partition;
  - DFT gradients with `grids.becke_scheme = gen_grid.becke_lko` and `grid_response = True`.

## What is wrong
The LKO partition (Laqua, Kussmann, Ochsenfeld, JCP 149, 204111 (2018)) is Becke's with the distance of two atoms
R = |R_i - R_j| replaced by a saturated one,

    mu_ij(r) = (|r - R_i| - |r - R_j|) / f(R),   f(R) = RC (1 - exp(-sum_{m=1}^{MC} (R/RC)^m / m)),   RC = 5, MC = 12.

Its derivative with respect to the position of atom i needs

    d(1/f(R))/dR_i = -f'(R) / f(R)^2 * (R_i - R_j) / R.

The code takes `f'(R) * (1/f(R))^3 * (R_i - R_j)`, i.e. `f'(R) / f(R)^3` in place of `f'(R) / (f(R)^2 R)`: the
form of Becke's partition, where f(R) = R, carried over. The two agree only where f(R) = R. With the series cut at
MC = 12, f(R)/R - 1 is about -(1 - x) x^12 / 13 for x = R/RC < 1: 1e-5 at 2.5 Bohr, 1e-3 at 4 Bohr, and for
R >= 5 Bohr f saturates at RC while R grows, so the factor is off by R/f(R).

## How it shows
- `pyscf_lko_dw_repro.py` (this folder; stock PySCF, CPU): the quadrature of a fixed smooth function on the SGX grid,
  sum_g w_g phi(r_g), differentiated with `get_dw_partition_sorted` and by finite differences: **3.5e-6** apart
  (the weight term and the point-motion term are each about 2.9 and cancel to 1e-3); with the fix **2.6e-10**, the
  finite-difference error.
- In an SGX exchange gradient (d tr(D K[D]) at a fixed density, water + CH2, def2-SVP): the part from the basis
  functions' centres agrees with finite differences to 1e-8, the grid part (points and weights) is 0.04% off
  (0.1% without the overlap fit); the total stays at 1.7e-7 of 2.1 whatever the finite-difference step, where with
  the fix it falls as h^2 to 2.5e-9 at h = 5e-5.
- In the energy gradients of those molecules: 1e-8 to 1e-7 Eh/Bohr, far below the thresholds of geometry
  optimizations; it grows with the number of atom pairs 4 to 7 Bohr apart.

## Why PySCF's test did not catch it
`pyscf/grad/test/test_rks.py::test_grid_response` compares the C routine with a Python reference (`grids_response(...,
lko=True)` in the test file), and that reference has the same factor (`v = (grid_dist[ia]-grid_dist[ib])/atm_dist**3`
with `atm_dist = fcut(R)`), and evaluates `dfcut` at f(R) instead of R. The test compares the two to 9 digits of a
fingerprint and has a finite-difference check for the Becke partition only.

## The fix
```diff
             atom_dist[i*natm+j] = 1 / _lko_sat_func(dr);
-            sat_deriv = _lko_sat_deriv(dr) * pow(atom_dist[i*natm+j], 3);
+            // d(1/f(R))/dR_i = -f'(R) / f(R)^2 (R_i - R_j) / R: the partition's
+            // distance is f(R), not R (the factor was 1/f(R)^3)
+            sat_deriv = _lko_sat_deriv(dr) * atom_dist[i*natm+j] * atom_dist[i*natm+j] / dr;
```
and in the test's reference: `atm_r` (the true distances) for the denominator and for `dfcut`, plus a
finite-difference check of the LKO weights beside the Becke one. Both are in this workspace's PySCF fork, commit
"dft: the derivative of the LKO partition's weights" on branch `perf` (tag `perf-2026-11-25c`); PySCF's suites
(dft, scf, df, grad, hessian, tdscf, solvent, sgx, lib) give the same results before and after but for that test.
GPU4PySCF has no LKO derivative of its own upstream; this workspace's GPU tree computes it in a CUDA kernel
(`gpu4pyscf/sgx/sgx.py::lko_weight_derivs_dot`) with the corrected formula.

## To report
An issue at pyscf/pyscf with `pyscf_lko_dw_repro.py`, and a pull request with the two changes above (the C line and
the test); it touches the code of #3076 (its author may want to look).
