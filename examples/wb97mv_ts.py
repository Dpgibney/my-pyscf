#!/usr/bin/env python
'''
wB97M-V/def2-TZVP transition-state optimization with this workspace's PySCF: geomeTRIC's saddle-point search
(transition=True) from an analytical Hessian, then the harmonic frequencies at the saddle point it finds, which
should have exactly one imaginary mode, along the reaction coordinate.

The optimizer, geomeTRIC, is not installed by scripts/cpu_setup.sh (examples/wb97mv_geomopt.py says how).
Then, from any directory:

    source /path/to/my-pyscf/scripts/pyscf-env.sh
    python /path/to/my-pyscf/examples/wb97mv_ts.py

Writes wb97mv_ts.out (the SCF, gradient and Hessian output) and wb97mv_ts.xyz (the saddle point) to the
current directory, and prints the frequencies and the atoms that move in the imaginary mode.

Larger molecules: the three switches below, all off. Measured on a 57-atom doublet (1198 basis functions, SMD
water; 10 cores of a laptop at 2 GHz; NOTES.md, "A UKS transition-state Hessian at production size"):
  - a step of the search (SCF and gradient) with RIJCOSX: 17 min;
  - an analytical Hessian with SGX_RESPONSE: about 5 h (the sum of its parts, each timed alone), half of it the
    CPHF equations, and 15-20 GB with max_memory 14000; over an hour more with the direct J that
    hobj.sgx_response alone leaves in the response;
  - PySCF's scratch files (the Hessian's Fock matrix derivatives, 4 GB there) go to $PYSCF_TMPDIR, by default /tmp:
    point it to a directory on a disk where /tmp is held in memory (a tmpfs).
'''
import numpy
from pyscf import gto, dft, df, grad, hessian  # grad, hessian: give dft.RKS its Gradients() and Hessian()
from pyscf.hessian import thermo
from pyscf.geomopt import geometric_solver

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
    output='wb97mv_ts.out',
)

# For larger molecules: RI-J and seminumerical K (RIJCOSX) in the optimization's gradients, with the start Hessian and
# the frequencies still from exact J and K. On HCN <-> HNC and Cl- + CH3Cl (def2-TZVPD, SMD water) it found the exact
# saddle point within 1e-4 and 4e-4 Angstrom, and the exact Hessian there gave the frequencies of RIJCOSX's own
# finite-difference Hessian within 0.3 cm^-1.
RIJCOSX = False
# For larger molecules: seminumerical K and RI-J (the even-tempered fitting basis) in the CPHF response of the
# Hessians, 0.03 and 0.1 cm^-1 in the frequencies (examples/wb97mv_hessian.py). The SCF and every derivative integral
# stay exact. The 57-atom doublet: one evaluation of the response of its 171 nuclear perturbations, of which the
# CPHF equations take 5 to 7, is 31 min (XC 12, K 10, VV10 7, J 1.5); with direct J, which hobj.sgx_response alone
# leaves there, 46 min.
SGX_RESPONSE = False
# For larger molecules: the start Hessian in a smaller basis, e.g. 'def2-svp' (None: the basis above). It only steers
# the search, which updates it from the gradients at every step; the frequencies at the end are those of the full
# model. The Hessian of a 16-atom radical takes 207 s in def2-SVP against 443 s (the two-electron terms shrink with
# the basis, the grid terms of XC and VV10 less). The search may then take other steps to the same saddle point:
# 13 instead of 14 for HCN <-> HNC, 10 instead of 5 for H + H2.
START_HESSIAN_BASIS = None


def make_mf(mol, rijcosx=False):
    '''One model for the optimization, the start Hessian and the frequencies: the saddle point is then a
    stationary point of the surface whose Hessian is analyzed.'''
    # exact J and K; unrestricted orbitals for open shells (mol.spin), with the same options
    mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc='wb97m-v')
    # XC grid, as in examples/wb97mv_hessian.py (99 radial points are enough for P, S and Cl too with the Hessian's
    # grid response)
    mf.grids.atom_grid = (99, 590)
    # VV10 grid of the energy, the gradient and the Hessian's VV10 derivative terms
    mf.nlcgrids.atom_grid = (50, 194)
    if rijcosx:
        mf = mf.RIJCOSX(auxbasis=df.aug_etb(mol))   # RI-J in an even-tempered fitting basis
    # Solvent: after the grids are set
    # mf = mf.SMD(); mf.with_solvent.solvent = 'water'
    # The SCF solution from 2-3 of this model's Fock builds (examples/wb97mv_hessian.py)
    mf.two_level = True
    # Grids of the Hessian's CPHF response only (0.02 and 0.01 cm^-1, examples/wb97mv_hessian.py)
    mf.set_second_grids(2)
    mf.set_second_nlcgrids(1)
    return mf


def analytical_hessian(mf):
    '''The analytical Hessian of a converged mf (exact J and K), Eh/Bohr^2, [natm,natm,3,3]'''
    hobj = mf.Hessian()
    # The derivatives of the grid points and weights, which PySCF leaves out by default: without them the soft modes
    # of the Cl- + CH3Cl saddle point were up to 41 cm^-1 off.
    hobj.grid_response = True
    if SGX_RESPONSE:
        hobj.sgx_response = True                # seminumerical K in the CPHF response (and direct J)
        hobj.sgx_response_auxbasis = 'aug_etb'  # RI-J there instead
    return hobj.kernel()


# The start Hessian at the guess. geomeTRIC steps uphill along its lowest mode (P-RFO) and updates the Hessian from
# the gradients at every step (Bofill's Murtagh-Sargent/Powell update). Without one (hessian left out), geomeTRIC
# makes its own at the first step from finite differences of the gradient: 6 gradients per atom.
mf = make_mf(mol)
if START_HESSIAN_BASIS is None:
    mf.kernel()
    h0 = analytical_hessian(mf)
else:
    mol0 = mol.copy()
    mol0.basis = START_HESSIAN_BASIS
    mol0.build()
    h0 = analytical_hessian(make_mf(mol0).run())

# The saddle-point search (geomeTRIC's default thresholds, shown so that they can be edited). The frequencies at the
# end are those of where the search stops: with these thresholds, and with RIJCOSX gradients, they came out within
# about 1 cm^-1 between the runs of this example with different switches (HCN <-> HNC, H + H2), while the
# approximations of SGX_RESPONSE change them by 0.01 cm^-1 at one geometry.
conv_params = {
    'convergence_energy': 1e-6,     # Eh
    'convergence_grms': 3e-4,       # Eh/Bohr
    'convergence_gmax': 4.5e-4,     # Eh/Bohr
    'convergence_drms': 1.2e-3,     # Angstrom
    'convergence_dmax': 1.8e-3,     # Angstrom
}
mf_opt = make_mf(mol, rijcosx=True) if RIJCOSX else mf
conv, mol_ts = geometric_solver.kernel(mf_opt, transition=True, hessian=h0, maxsteps=100, **conv_params)
mol_ts.tofile('wb97mv_ts.xyz')

# Frequencies at the saddle point
mf_ts = make_mf(mol_ts)
mf_ts.kernel()
gradient = mf_ts.Gradients().kernel()
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
