#!/usr/bin/env python
'''
wB97M-V/def2-TZVPD energy, gradient and harmonic frequencies with this workspace's PySCF, with the
settings measured in NOTES.md ("wB97M-V/def2-TZVPD: the review and what came of it").

    source /path/to/my-pyscf/scripts/pyscf-env.sh
    python /path/to/my-pyscf/examples/wb97mv_hessian.py

Writes wb97mv_hessian.out (the SCF, gradient and Hessian output) to the current directory and prints the
frequencies.
'''
from pyscf import gto, dft, grad, hessian  # grad, hessian: give dft.RKS its Gradients() and Hessian()
from pyscf.hessian import thermo

mol = gto.M(
    atom='''
    O  0.000  0.000  0.118
    H  0.000  0.757 -0.471
    H  0.000 -0.757 -0.471
    ''',                            # or atom='geometry.xyz'; frequencies need an optimized geometry
    basis='def2-tzvpd',
    charge=0,
    spin=0,
    verbose=4,
    output='wb97mv_hessian.out',
)

mf = dft.RKS(mol, xc='wb97m-v')     # exact J and K; dft.UKS for open shells (same options below)

# XC grid. The analytical Hessian has no grid-weight derivatives, so it converges more slowly with the
# radial grid than the energy does: second-row elements need 150 radial points (H2PO4-: 2 cm^-1 between 99
# and 150 points on phosphorus, nothing between 150 and 250).
mf.grids.atom_grid = {'default': (99, 590), 'P': (150, 590), 'S': (150, 590), 'Cl': (150, 590)}
# VV10 grid of the energy, the gradient and the Hessian's VV10 derivative terms
mf.nlcgrids.atom_grid = (50, 194)

# Solvent: after the grids are set. The solvent terms of the gradient and the Hessian are analytical
# (SMD's CDS term by finite differences of its gradient, which costs nothing).
# mf = mf.SMD(); mf.with_solvent.solvent = 'water'

# The SCF solution from 2-3 of this model's Fock builds instead of 10-15: the iterations run on the Fock
# matrix of a cheap model (RI-J in an even-tempered fitting basis, seminumerical K, coarser grids) plus the
# difference of the two at a reference density. The energy and orbitals are this model's own, to conv_tol.
mf.two_level = True

# Grids of the CPHF response only (the energy, the gradient and the Hessian's derivative terms keep the
# grids above). Largest change of a harmonic frequency on a water dimer, H2PO4-, glycine and caffeine:
# XC kernel on level 2: 0.02 cm^-1; VV10 kernel on level 1: 0.01 cm^-1 (level 0: 0.07 cm^-1, and 0.2 cm^-1
# on a soft mode of H2PO4- in SMD, for a VV10 response 2-3 times cheaper: a few percent of the Hessian).
mf.set_second_grids(2)
mf.set_second_nlcgrids(1)

mf.kernel()
gradient = mf.Gradients().kernel()  # Eh/Bohr, one row per atom

hobj = mf.Hessian()
# The CPHF solver that stops on the true residual (PySCF's Krylov solver stops short for DFT)
hobj.cphf_residual_tol = 1e-4
# For larger molecules: seminumerical K (and direct J) in the CPHF response only, 0.03 cm^-1 on grid
# level 1; with def2-TZVPD level 0 costs up to 1 cm^-1. The SCF and every derivative integral stay exact.
# hobj.sgx_response = True
# hobj.sgx_response_auxbasis = 'aug_etb'   # RI-J instead of direct J in the response: 0.1 cm^-1 more
h = hobj.kernel()                   # Eh/Bohr^2, [natm,natm,3,3]

freq = thermo.harmonic_analysis(mol, h, imaginary_freq=False)['freq_wavenumber']
print('E = %.10f Eh' % mf.e_tot)
print('largest gradient component %.2e Eh/Bohr' % abs(gradient).max())
print('harmonic frequencies (cm^-1):', ' '.join('%.1f' % f for f in freq))
