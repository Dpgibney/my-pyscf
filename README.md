# my-pyscf

Exact speedups of PySCF and GPU4PySCF: every change keeps the results of the unmodified codes to rounding (energies
within 1e-9 Eh, the same SCF cycles). `NOTES.md` is the full log: the CPU iterations (It0-It26 and the PCM work, PySCF, qcint) and the
GPU4PySCF investigation (parts 1-39, measured on a GTX 1650 Super), with every measurement and the reasons for each
change.

Since 2026-11-07 the PySCF fork also has an opt-in approximation, RI-J with seminumerical exchange set up as ORCA's
RIJCOSX (`NOTES.md`, "RIJCOSX in PySCF"): RI-J through a J-engine that stores no 3-index tensor, SGX's K screened by
shell pair, grid levels 0 -> 1 and the energy once more on level 2. On node2 it runs about as fast as ORCA 6.1.1's
RIJCOSX (caffeine and n-C20H42/def2-TZVP B3LYP: 58 and 123 s against 57 and 110 s; n-C20H42/def2-QZVP 651 against
620 s) with the same errors against exact (-0.56 and -1.03 mEh), in 2-3.5 GB and nothing on disk. For wB97M-V most
of the time is VV10, which PySCF evaluates in every cycle on the XC grid: set `mf.nlcgrids.level = 1` (n-C20H42:
427 s, the energy within 2e-5 Eh of the default grid's; ORCA 242 s with VV10 added after the SCF, 1206 s with it in
every cycle). VV10 runs on the fused XC kernels, so wB97M-V needs the memory of wB97X (3.3 GB at the default
max_memory). The exact defaults are unchanged.

The analytical Hessian's VV10 terms (energy second derivative, Fock derivative, CPHF and TDDFT response) were
rewritten on the fused kernels and matrix products, with the same results to 4e-13 (`NOTES.md`, "The wB97M-V
Hessian"): the exact wB97M-V/def2-TZVP Hessian of n-C20H42 takes 2 h 17 min on the laptop (VV10 40 min of it,
estimated at ~15 h before), propane/def2-SVP 19.6 s instead of 220 s. One change is not to rounding: the CPHF no
longer recomputes the response of its solution (one response fewer) and starts from the uncoupled solution as an
explicit guess, so DFT Hessians move within the CPHF's own convergence error, 1e-7 to 1e-5 in Hessian elements
(PySCF's Krylov solver stops short of its tolerance for DFT; converged, the old and new start agree).

For wB97M-V/def2-TZVPD with production grids (`NOTES.md`, "wB97M-V/def2-TZVPD: the review and what came of it";
caffeine, 614 functions, XC grid (99,590), VV10 grid (50,194), the laptop): `mf.two_level = True` gets the exact
SCF's own solution from 2 of its Fock builds instead of 15, iterating on a cheap model's Fock matrix corrected at
a reference density (178.7 s against 451 s, the energy within 1.3e-10 Eh). The XC and VV10 terms of the RKS gradient are
contracted on the grid (14.5 -> 4.9 s), SGX evaluates the exchange of a range-separated hybrid in one pass (needs
the qcint of this workspace), and the CPHF response of the Hessian takes its own grids (`mf.set_second_grids(2)`,
`mf.set_second_nlcgrids(1)`) and seminumerical K (`hobj.sgx_response = True`, now on grid level 1): the Hessian in
2041 s against 3176 s, frequencies within 0.04 cm^-1 of the exact response's (VV10 response on level 0: 1867 s,
0.07 cm^-1).
Hessians solve their CPHF equations with `cphf.solve_residual` to a residual of 1e-4 by default
(`hobj.cphf_residual_tol`; `None` for PySCF's Krylov solver): at its default tolerance the Krylov solver stops
short of a converged solution for DFT, by 0.01-0.03 cm^-1 for closed shells, 0.3 cm^-1 for wB97M-V in SMD and
1.2 cm^-1 for a UKS/B3LYP radical, and takes longer.
SMD and PCM Hessians and gradients no longer recompute the surface integrals for every nuclear coordinate or hold
surface arrays quadratic in the points with nine components (`NOTES.md`, "SMD Hessians, UKS and the rest of the
wB97M-V list"): the solvent terms of caffeine's SMD Hessian take 32 s (above 16 GB and unfinished before), the
solvent term of its gradient 6.5 s instead of 45 s. The CPHF of a solvated model runs from the same factors as in
the gas phase (it fell back to AO matrices for every PCM/SMD Hessian: 1549 -> 668 s), so caffeine's
wB97M-V/def2-TZVPD Hessian in SMD water takes 2156 s and about 12 GB (`max_memory` 12000), with the VV10 response on level 1
(`mf.set_second_nlcgrids(1)`: level 0 moves a soft mode of H2PO4- in SMD by 0.2 cm^-1). UKS has the same gradient
and response options, and Hessians of VV10 functionals, which upstream does not implement.
At production size (`NOTES.md`, "A production-size energy and gradient": NAD+ in SMD water, 1916 functions, 14,934
surface points, 10 threads, `max_memory` 16000): the exact energy and gradient take 59 and 73 minutes, with
`mf.RIJCOSX(auxbasis=df.aug_etb(mol))` 28 and 7.5 minutes (-4.3e-5 Eh, gradient within 9.8e-5 Eh/Bohr) in at most 17.9 GB. The SMD gradient asked for
about 45 GB there for the derivatives of its surface matrices (by their sizes; 20 GB at 10,000 points) and takes 1.2 GB now, and the RI-J
gradient runs through the J-engine (232 -> 31 s).
These are in the tags below, from `perf-2026-11-10` on (PySCF and qcint together: the one-pass exchange of a range-separated
hybrid needs the new qcint). `examples/wb97mv_hessian.py` has the settings.

GPU4PySCF has the two-level SCF since `perf-2026-11-22` (`NOTES.md`, "GPU4PySCF, part 36"; a GTX 1650 Super,
caffeine, wB97M-V, XC grid (99,590), VV10 grid (50,194), SMD water): `mf.two_level = True` iterates with the same J
and K on coarser XC and VV10 grids and evaluates the model's own grids twice, in the plain SCF's number of cycles:
the def2-SVP SCF takes 53 s instead of 127 s, def2-TZVP 273 s instead of 380 s, def2-TZVPD 584 s instead of 727 s,
with energies within 4e-11 Eh of the plain SCF's. The gradient takes the solvent's surface charges and the VV10
potential from the SCF's last cycle, the exchange derivatives of a range-separated hybrid in two passes of n roots
instead of one of n and one of 2n, and the solvent's terms without arrays of three times the surface points squared:
def2-SVP 15.4 -> 7.4 s, def2-TZVP 47 s where the card's 4 GB ran out before, def2-TZVPD 108 s. Without `two_level`
the energies are those of `perf-2026-11-03` to 2e-13 Eh. What is left there is the exact exchange: 70% of the
def2-TZVPD SCF and, with J, 91% of its gradient.

Since `perf-2026-11-22b` the cheap model of that two-level SCF evaluates the exchange seminumerically (`NOTES.md`,
"GPU4PySCF, part 37"): `gpu4pyscf.sgx`, a new package (upstream GPU4PySCF has no COSX) whose CUDA kernels are
generated per class of shell pairs and compiled at run time, in single precision for the cheap model, while each of
the two reference evaluations builds the model's own K. The same job's SCF then takes 42 s (def2-SVP), 130 s
(def2-TZVP) and 246 s (def2-TZVPD), and 41, 112 and 192 s with `mf.two_level_auxbasis = 'aug_etb'` (a fitted J in the
cheap model as well); energies within 2e-10 Eh of the plain SCF's, gradients unchanged. A K build of
caffeine/def2-TZVPD takes 3.1 s in double and 0.39 s in single precision against 32.8 s for the exact one.
`mf.COSX()` is the approximate model by itself (energies only: +1e-5 Eh for caffeine on its default grid). What is
left of the 192 s: the model's own K at the two references (62 s), XC (46 s), the solvent's surface integrals (40 s),
J (22 s).

Since `perf-2026-11-22c` the two contractions of a PCM or SMD model with its surface charges, done in every SCF cycle,
run on kernels of that package (`NOTES.md`, "GPU4PySCF, part 38"): 3.1 to 3.7 times faster than the Rys ones in double
precision, the plain SCF's energies unchanged to 2e-13 Eh. The same job's SCF: plain 125 s (def2-SVP), 367 s
(def2-TZVP) and 704 s (def2-TZVPD); two-level 37, 114 and 223 s, and 35, 95 and 169 s with the fitted J. What a
reference of the two-level SCF builds as a change of the density (the model's K; its J with the fitted J) now enters
the energy at second order, which took a screening error of up to 3e-11 Eh out of it.
`mf.two_level_sgx_ref_level = 3` (off by default: an approximation, 2e-12 Eh in these tests) also takes the K of the
references after the first from a seminumerical K of the change of the density, in double precision: def2-TZVPD
223 -> 208 s (169 -> 154 s with the fitted J), H2PO4- in def2-TZVPD 18 -> 14 s, 2% at def2-TZVP, and slower at
def2-SVP; in single precision it loses 2e-10 Eh with diffuse functions. Part 38 also measured an SCF's energy against
one tight build of the potential of its final density: up to 1.3e-10 Eh apart for the plain SCF, of either sign (J and K
are summed over builds that are each screened for one cycle's change of the density, as upstream does).

Since `perf-2026-11-22d` the energy is carried from cycle to cycle instead (`NOTES.md`, "GPU4PySCF, part 39"):
`mf.second_order_energy`, on by default, takes the J and K built for a cycle's change of the density at second order
in that change, where upstream traces their sum with the whole density. The Fock matrices, the cycles, the gradients
and the time are the same; the plain SCF's energy is then within 9e-12 Eh of the tight build of its own density
(def2-SVP -6.8e-12, def2-TZVP -8.2e-12, def2-TZVPD -5.7e-12, where it was +1.07e-10, -7.2e-11 and -1.26e-10), the
two-level SCF's likewise, and the two agree to 1.4e-11 Eh. This is the one change that moves a default result with
respect to v1.9.0 beyond rounding, by up to 1.2e-10 Eh in these tests; `mf.second_order_energy = False` gives
upstream's energy back (the def2-SVP job to 5e-13 Eh), and `scripts/gpu_bench.sh` compares with v1.9.0 that way.

```python
mf = dft.RKS(mol, xc='b3lyp').RIJCOSX()   # instead of the exact J/K or mf.density_fit()
mf.kernel()
```

The modified code lives in forks; the tags pin what was measured (`scripts/pins.txt`):

| code | fork | branch | tag | compared against |
|---|---|---|---|---|
| GPU4PySCF | [Dpgibney/gpu4pyscf](https://github.com/Dpgibney/gpu4pyscf) | `exp/prim-screen` (all changes; its CUDA kernels are generated by `bench/gpu/exp_prim_screen.py`), `perf` (the Python changes) | `perf-2026-11-22d` | `v1.9.0` |
| PySCF | [Dpgibney/pyscf](https://github.com/Dpgibney/pyscf) | `perf` | `perf-2026-11-11` | master 44313dfb4 |
| qcint | [Dpgibney/qcint](https://github.com/Dpgibney/qcint) | `perf` | `perf-2026-11-10` | cdda2892f |

## Setting up the CPU PySCF on another computer

Needs Linux on x86-64 with gcc, g++, gfortran, make, git and Python 3 with venv (3.12 or newer gets the pinned
numpy 2.5.4, scipy 1.18.1 and h5py 3.16.0); cmake 3.22 or newer is installed into the venv if missing.

```bash
git clone https://github.com/Dpgibney/my-pyscf && cd my-pyscf && bash scripts/cpu_setup.sh
source scripts/pyscf-env.sh && python my_script.py   # the venv, the tree on PYTHONPATH, one thread per core
```

`cpu_setup.sh` clones the trees of `scripts/pins.txt` and builds them as measured: OpenBLAS, libxc, qcint
(profile-guided and `-march=native`, so build on the type of machine that will run it) and PySCF against them,
then checks a water B3LYP energy (20 min on node2's 6 cores). `--with-base` also builds the unmodified PySCF and
compares the XC, J and K kernels of both trees (`bench/kernels.py`); `--jobs N` sets the build parallelism.
Rerunning it after a `git pull` rebuilds only what the new pins change.

`pyscf-env.sh` works out its defaults each time it is sourced, from what the process may use: one OpenMP thread
per physical core of its CPUs, and `PYSCF_MAX_MEMORY` at 2/3 of its memory. In a Slurm job these are the job's
`-c` and `--mem` (read from its CPU binding, its memory cgroup and Slurm's variables); elsewhere they are the
machine's, with at most 20000 MB. Values you export before sourcing it win. A job script needs nothing else:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
source ~/my-pyscf/scripts/pyscf-env.sh   # 16 threads (8 if those are 8 hyperthreaded cores), ~45800 MB
python job.py                            # run it directly, or with srun -c $SLURM_CPUS_PER_TASK
```

## Testing GPU4PySCF on another GPU

Needs an NVIDIA GPU of compute capability 7.0 or newer, a CUDA 12 toolkit (`nvcc`), cmake, gcc, git and Python
3.10-3.12.

```bash
git clone https://github.com/Dpgibney/my-pyscf && cd my-pyscf
bash scripts/gpu_setup.sh        # the venv, and both trees (modified and v1.9.0) built for this GPU
bash scripts/gpu_bench.sh        # unit tests, then the comparison with v1.9.0 -> bench/results/<host>/
source scripts/gpu-env.sh perf   # to run your own scripts against the modified tree (base: v1.9.0)
```

`gpu_bench.sh --quick` runs caffeine def2-SVP B3LYP only (SCF, gradient, Hessian, TDDFT); the full set adds PBE,
r2SCAN, def2-TZVP, density fitting and the VV10 functionals (B97M-V, wB97M-V).
After the comparison it runs the production-like job (`bench/gpu/prod_parts.py`: caffeine, wB97M-V with the
production grids in SMD water, every part of the SCF and of the gradient timed) for v1.9.0, the modified tree and
its two-level SCF four ways (the model's own K in the cheap model, the seminumerical K, a fitted J as well, and
`two_level_sgx_ref_level = 3` on top of that). Where the modified tree is compared with v1.9.0 it runs with
`second_order_energy` off, so that the difference is the code's;
`--large` adds NAD+ on def2-TZVPD (1916 functions), for a card with 24 GB or more.

## Layout

- `NOTES.md`: the experiment log and its results.
- `bench/`: the benchmark and validation harness (CPU: `run_bench.py`, `validate.py`, `kernels.py`; GPU: `bench/gpu/`).
- `bench/results/`: raw results (the laptop's in `bench/results/`, node1's GTX 1650 Super in `node1/`, node2's
  Ryzen 5 4500 in `node2/`).
- `scripts/`: build and environment scripts (`node1_*` and `node2_*` are specific to those machines).
