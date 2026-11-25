#!/usr/bin/env python
'''
The transition-state optimization of examples/wb97mv_ts.py on a GPU, with this workspace's GPU4PySCF:
wB97M-V/def2-TZVP, geomeTRIC's saddle-point search (transition=True) from an analytical Hessian with its grid
response, then the harmonic frequencies at the saddle point it finds, which should have exactly one imaginary mode.

    bash /path/to/my-pyscf/scripts/gpu_setup.sh         # once: the venv (with geomeTRIC) and the tree for this GPU
    source /path/to/my-pyscf/scripts/gpu-env.sh perf
    python /path/to/my-pyscf/examples/wb97mv_ts_gpu.py

Writes wb97mv_ts_gpu.out (the SCF, gradient and Hessian output) and wb97mv_ts_gpu.xyz (the saddle point) to the
current directory, and prints the frequencies and the atoms that move in the imaginary mode.

What differs from the CPU example:
  - J and K are exact everywhere (the SCF, the gradients of the search, the Hessians): the GPU tree's seminumerical
    K serves the cheap model of the two-level SCF and has no gradient, so there are no RIJCOSX and SGX_RESPONSE
    switches;
  - the CPHF equations are solved on two levels by default (hobj.cphf_two_level): a cheap model whose VV10 response
    is on a coarse grid, then the model's own response for what is left. The result is the model's own;
  - the grid of the CPHF's XC response is mf.cphf_grids, by default SG-1 (50,194) in GPU4PySCF; here level 2, as
    mf.set_second_grids(2) of the CPU example.

On HCN <-> HNC and on H + H2 (spin=1, UKS), in the gas phase and in SMD water, with either start Hessian, it finds
the CPU example's saddle points: the frequencies within 0.1 cm^-1, in the same number of steps or two fewer (NOTES.md,
"GPU4PySCF, part 41"; a GTX 1650 Super, 4 GB).

Larger molecules: START_HESSIAN_BASIS below. The largest Hessians run on that card are those of a 16-atom radical
and of caffeine, so what follows is from the sizes of the arrays. The CPHF equations are solved for as many atoms at
a time as the free device memory takes (for an open shell with 1200 basis functions about 1.4 atoms per GB), in
several blocks if need be. With a solvent, set mol.max_memory (MB) to what the host has: the Hessian keeps the
solvent's integrals with the orbital products there when they fit in half of it (32 GB for a 57-atom doublet with
1198 functions and 12,812 surface points) and otherwise contracts them anew for every response.
'''
import numpy
import cupy
from pyscf import gto
from pyscf.hessian import thermo
from gpu4pyscf import dft
from gpu4pyscf.dft import gen_grid
from gpu4pyscf.geomopt import geometric_solver

# A guess of the saddle point: here HCN <-> HNC, the H bridging C and N
mol = gto.M(
    atom='''
    C  0.000  0.000  0.000
    N  1.190  0.000  0.000
    H  0.486  1.042  0.000
    ''',                            # or atom='guess.xyz'
    basis='def2-tzvp',
    charge=0,
    spin=0,                         # 2S: 1 for a doublet (unrestricted orbitals then)
    verbose=4,
    output='wb97mv_ts_gpu.out',
)

# For larger molecules: the start Hessian in a smaller basis, e.g. 'def2-svp' (None: the basis above). It only steers
# the search, which updates it from the gradients at every step; the frequencies at the end are those of the full
# model (see examples/wb97mv_ts.py for what it costs in steps).
START_HESSIAN_BASIS = None


def make_mf(mol):
    '''One model for the optimization, the start Hessian and the frequencies: the saddle point is then a
    stationary point of the surface whose Hessian is analyzed.'''
    # exact J and K; unrestricted orbitals for open shells (mol.spin), with the same options
    mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc='wb97m-v')
    # XC grid and VV10 grid, as in examples/wb97mv_ts.py
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    # Solvent: after the grids are set
    # mf = mf.SMD(); mf.with_solvent.solvent = 'water'
    # The SCF solution from 2 of this model's Fock builds: the iterations run on a cheap model (seminumerical K,
    # coarse grids) corrected at reference densities
    mf.two_level = True
    # Grid of the XC kernel in the Hessian's CPHF response only (0.02 cm^-1)
    mf.cphf_grids = gen_grid.Grids(mol)
    mf.cphf_grids.level = 2
    return mf


def analytical_hessian(mf):
    '''The analytical Hessian of a converged mf, Eh/Bohr^2, [natm,natm,3,3], on the host'''
    hobj = mf.Hessian()
    # The derivatives of the grid points and weights (see examples/wb97mv_ts.py)
    hobj.grid_response = True
    # Optional, the same Hessian: the seminumerical K also in the cheap model of the two-level CPHF. It took 10% off
    # the def2-TZVP Hessian of a 16-atom radical on a GTX 1650 Super (whose double precision is slow, and this K
    # runs in single precision); not measured on a card with fast double precision, and slower in def2-SVP.
    # hobj.cphf_two_level_parts = ('nlc', 'k')
    return cupy.asnumpy(hobj.kernel())


# The start Hessian at the guess (geomeTRIC updates it from the gradients at every step)
mf = make_mf(mol)
if START_HESSIAN_BASIS is None:
    mf.kernel()
    h0 = analytical_hessian(mf)
else:
    mol0 = mol.copy()
    mol0.basis = START_HESSIAN_BASIS
    mol0.build()
    h0 = analytical_hessian(make_mf(mol0).run())

# The saddle-point search (geomeTRIC's default thresholds, shown so that they can be edited)
conv_params = {
    'convergence_energy': 1e-6,     # Eh
    'convergence_grms': 3e-4,       # Eh/Bohr
    'convergence_gmax': 4.5e-4,     # Eh/Bohr
    'convergence_drms': 1.2e-3,     # Angstrom
    'convergence_dmax': 1.8e-3,     # Angstrom
}
conv, mol_ts = geometric_solver.kernel(mf, transition=True, hessian=h0, maxsteps=100, **conv_params)
mol_ts.tofile('wb97mv_ts_gpu.xyz')

# Frequencies at the saddle point
mf_ts = make_mf(mol_ts)
mf_ts.kernel()
gradient = cupy.asnumpy(mf_ts.Gradients().kernel())
h = analytical_hessian(mf_ts)
res = thermo.harmonic_analysis(mol_ts, h, imaginary_freq=False)
freq = res['freq_wavenumber']       # imaginary frequencies as negative numbers
nimag = int((freq < 0).sum())
disp = numpy.linalg.norm(res['norm_mode'][0], axis=1)   # the lowest mode, the displacement of each atom

print('saddle-point search converged:', conv)
print('E = %.10f Eh' % mf_ts.e_tot)
print('largest gradient component %.2e Eh/Bohr' % abs(gradient).max())
print('harmonic frequencies (cm^-1):', ' '.join('%.1f' % f for f in freq))
print('%d imaginary mode(s)%s' % (nimag, '' if nimag == 1 else ': not a first-order saddle point'))
print('atoms moving most in the lowest mode:',
      ', '.join('%s%d %.2f' % (mol_ts.atom_symbol(i), i, disp[i]) for i in numpy.argsort(-disp)[:5]))
