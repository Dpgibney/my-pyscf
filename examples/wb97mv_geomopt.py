#!/usr/bin/env python
'''
wB97M-V energy, nuclear gradient and geometry optimization with this workspace's PySCF.

The optimizer, geomeTRIC, is not installed by scripts/cpu_setup.sh. Once:

    source scripts/pyscf-env.sh
    python -m pip install geometric
    # in a venv made by uv, which has no pip: uv pip install --python "$VIRTUAL_ENV/bin/python" geometric

Then, from any directory:

    source /path/to/my-pyscf/scripts/pyscf-env.sh
    python /path/to/my-pyscf/examples/wb97mv_geomopt.py

Writes wb97mv_opt.out (the SCF and gradient output) and wb97mv_opt.xyz (the optimized geometry) to the
current directory.
'''
from pyscf import gto, dft, grad  # grad: gives dft.RKS its Gradients()
from pyscf.geomopt.geometric_solver import optimize

mol = gto.M(
    atom='''
    O  0.000  0.000  0.130
    H  0.000  0.800 -0.480
    H  0.000 -0.800 -0.480
    ''',                            # or atom='start.xyz'
    basis='def2-tzvp',
    charge=0,
    spin=0,
    verbose=4,
    output='wb97mv_opt.out',
)

mf = dft.RKS(mol, xc='wb97m-v')     # open shell (spin != 0): dft.UKS

# For larger molecules, one of these in place of the exact J and K. RIJCOSX's gradient error (at most 1.1e-4
# Eh/Bohr on caffeine/def2-TZVP with B3LYP) is fine for the thresholds below but too large for tight ones.
# mf = mf.RIJCOSX()                 # RI-J and seminumerical K, set up as ORCA's RIJCOSX
# mf = mf.density_fit()             # RI-J and RI-K

# wB97M-V is a grid-sensitive meta-GGA: level 4 is (90, 590) on C, N and O, close to the (99, 590) grid commonly
# used with it. PySCF's default is 3.
mf.grids.level = 4

# VV10's cost grows with the square of its grid points. PySCF's default is level 3; level 1 changed the gradient
# by at most 1.2e-5 Eh/Bohr on caffeine/def2-SVP.
mf.nlcgrids.level = 1

# Energy and gradient at the starting geometry
mf.kernel()
gradient = mf.Gradients().kernel()  # Eh/Bohr, one row per atom

# Geometry optimization (geomeTRIC's default thresholds, shown so that they can be edited)
conv_params = {
    'convergence_energy': 1e-6,     # Eh
    'convergence_grms': 3e-4,       # Eh/Bohr
    'convergence_gmax': 4.5e-4,     # Eh/Bohr
    'convergence_drms': 1.2e-3,     # Angstrom
    'convergence_dmax': 1.8e-3,     # Angstrom
}
mol_eq = optimize(mf, maxsteps=100, **conv_params)
mol_eq.tofile('wb97mv_opt.xyz')
