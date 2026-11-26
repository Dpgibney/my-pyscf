#!/usr/bin/env python
'''
wB97M-V/def2-TZVP transition-state optimization with this workspace's PySCF: geomeTRIC's saddle-point search
(transition=True) from an analytical Hessian, the harmonic frequencies at the saddle point it finds, which should have
exactly one imaginary mode, along the reaction coordinate, and the free energy there.

The optimizer, geomeTRIC, is not installed by scripts/cpu_setup.sh (examples/wb97mv_geomopt.py says how).
Then, from any directory:

    source /path/to/my-pyscf/scripts/pyscf-env.sh
    python /path/to/my-pyscf/examples/wb97mv_ts.py

Writes wb97mv_ts.out (the SCF, gradient, Hessian and thermochemistry output) and wb97mv_ts.xyz (the saddle point) to
the current directory, and prints the frequencies, the atoms that move in the imaginary mode and the free energy.

Each SCF starts from the converged density of the one before it: the first of the search from that of the start
Hessian's SCF (projected into the basis of mol if START_HESSIAN_BASIS is another), each later one of the search from
that of the geometry before (PySCF's scanners), and the SCF at the saddle point from the search's last. That saves
cycles of the cheap model of the two-level SCF (the model's own Fock builds are two either way) and keeps an open
shell in one electronic state through all of them.

The free energy at the saddle point is that of an ideal gas of rigid rotors and harmonic oscillators at TEMPERATURE,
without the imaginary mode (the reaction coordinate, as transition-state theory has it), with Grimme's quasi-RRHO
vibrational entropy (QRRHO_FREQ0; ORCA's default), in the standard state of 1 atm, or of 1 mol/L with a solvent
model, whose solvation free energy is for 1 mol/L in both phases. The rotational symmetry number is that of the point
group of the saddle point within 0.01 Bohr, printed with G (SYMMETRY_NUMBER replaces it). On ORCA 6.1.1's own
frequencies this free energy is ORCA's within 5e-7 Eh (bench/diag/thermo/check_thermo.py). A free energy of
activation is its difference from the free energy of the reactant's minimum computed in the same way: the same model,
temperature, standard state and treatment of the soft modes.

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
from pyscf import gto, dft, df, lib, lo, symm
from pyscf import grad, hessian  # noqa: F401 (they give dft.RKS its Gradients() and Hessian())
from pyscf.data import nist
from pyscf.hessian import thermo
from pyscf.scf import addons
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
    verbose=4,                      # 4: with progress lines of the Hessian's steps that take over 30 s
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
# The temperature (K) of the free energy
TEMPERATURE = 298.15
# Grimme's quasi-RRHO vibrational entropy (Chem. Eur. J. 18, 9955 (2012)): the entropy of each mode interpolates
# between that of a harmonic oscillator and that of a free rotor, half and half at this frequency (cm^-1). The entropy
# of a harmonic oscillator grows without bound as its frequency goes to zero, which makes the soft modes of larger
# molecules the largest error of the plain RRHO free energy. None: the harmonic entropy for every mode (the free
# energy with it is printed as well).
QRRHO_FREQ0 = 100.
# The rotational symmetry number of the free energy. None: that of the point group of the saddle point within 0.01 Bohr
# (printed with G). PySCF's own tolerance, 1e-5 Bohr, is one that an optimized geometry meets only by chance: of 16
# searches for the D-infinity-h saddle point of H + H2 one came out symmetric to it (symmetry number 2), 15 within
# 7e-5 to 2.2e-3 Angstrom of it (1), which moves G by R T ln 2 = 0.41 kcal/mol. A number here replaces the detected one.
SYMMETRY_NUMBER = None


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


def orbitals_in_basis(mf0, mol):
    '''The converged orbitals and occupations of mf0 for a model of mol, the same molecule at the same geometry: those
    of mf0 in the same basis, else projected into the basis of mol and orthonormalized there'''
    if gto.same_basis_set(mf0.mol, mol):
        return mf0.mo_coeff, mf0.mo_occ
    s = mol.intor_symmetric('int1e_ovlp')

    def project(c):
        return lo.orth.vec_lowdin(addons.project_mo_nr2nr(mf0.mol, c, mol), s)
    if numpy.ndim(mf0.mo_coeff) == 3:     # unrestricted: the orbitals of each spin
        return numpy.asarray([project(c) for c in mf0.mo_coeff]), mf0.mo_occ
    return project(mf0.mo_coeff), mf0.mo_occ


def thermochemistry(mf, freq_au):
    '''The thermochemistry (the dict of pyscf.hessian.thermo.thermo, Eh and Eh/K) of the stationary point of the
    converged mf from its harmonic frequencies (atomic units, the imaginary ones negative, as harmonic_analysis with
    imaginary_freq=False gives them; thermo leaves those out), at TEMPERATURE. The standard state: 1 atm, or 1 mol/L
    with a solvent model (an ideal gas at the pressure c R T, 24.5 atm at 298.15 K). With QRRHO_FREQ0, Grimme's
    quasi-RRHO vibrational entropy in S_vib, S_tot, G_vib and G_tot; G_tot_rrho is G with the harmonic entropy.'''
    pressure, state = 101325., '1 atm'
    if getattr(mf, 'with_solvent', None) is not None:
        pressure, state = 1e3 * nist.AVOGADRO * nist.BOLTZMANN * TEMPERATURE, '1 mol/L'
    # the point group within 0.01 Bohr, also for the symmetry number that thermo takes from it (SYMMETRY_NUMBER)
    with lib.temporary_env(symm.geom, TOLERANCE=1e-2):
        group = symm.detect_symm(mf.mol._atom)[0]
        res = thermo.thermo(mf, freq_au, TEMPERATURE, pressure,
                            qrrho=QRRHO_FREQ0 is not None, qrrho_freq0=QRRHO_FREQ0 or 100.)
        res['G_tot_rrho'] = thermo.thermo(mf, freq_au, TEMPERATURE, pressure)['G_tot']
    res['point_group'] = (group, '')
    res['standard_state'] = (state, '')
    if SYMMETRY_NUMBER is not None:
        # the rotational partition function goes as 1 / sigma
        ds = nist.BOLTZMANN / nist.HARTREE2J * numpy.log(res['sym_number'][0] / SYMMETRY_NUMBER)
        for key in ('S_rot', 'S_tot'):
            res[key] = (res[key][0] + ds, 'Eh/K')
        for key in ('G_rot', 'G_tot', 'G_tot_rrho'):
            res[key] = (res[key][0] - TEMPERATURE * ds, 'Eh')
        res['sym_number'] = (SYMMETRY_NUMBER, '')
    return res


# The start Hessian at the guess. geomeTRIC steps uphill along its lowest mode (P-RFO) and updates the Hessian from
# the gradients at every step (Bofill's Murtagh-Sargent/Powell update). Without one (hessian left out), geomeTRIC
# makes its own at the first step from finite differences of the gradient: 6 gradients per atom.
mol_start = mol
if START_HESSIAN_BASIS is not None:
    mol_start = mol.copy()
    mol_start.basis = START_HESSIAN_BASIS
    mol_start.build()
mf_start = make_mf(mol_start)
mf_start.kernel()
h0 = analytical_hessian(mf_start)

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
# The scanner of the gradients starts the SCF at each geometry from the orbitals that its model holds, those of the
# geometry before; at the first, the guess, from those of the start Hessian's SCF (in the basis of mol)
if RIJCOSX or START_HESSIAN_BASIS is not None:
    mf_opt = make_mf(mol, rijcosx=RIJCOSX)
    mf_opt.mo_coeff, mf_opt.mo_occ = orbitals_in_basis(mf_start, mol)
else:
    mf_opt = mf_start
g_scanner = mf_opt.Gradients().as_scanner()
conv, mol_ts = geometric_solver.kernel(g_scanner, transition=True, hessian=h0, maxsteps=100, **conv_params)
mol_ts.tofile('wb97mv_ts.xyz')

# Frequencies at the saddle point. mol_ts is where the search's last SCF and gradient were evaluated: the SCF there
# starts from that density, and without RIJCOSX the search's last gradient is this model's at this geometry.
mf_ts = make_mf(mol_ts)
mf_ts.kernel(dm0=g_scanner.base.make_rdm1())
gradient = mf_ts.Gradients().kernel() if RIJCOSX else g_scanner.de
h = analytical_hessian(mf_ts)
res = thermo.harmonic_analysis(mol_ts, h, imaginary_freq=False)
freq = res['freq_wavenumber']       # imaginary frequencies as negative numbers
nimag = int((freq < 0).sum())
disp = numpy.linalg.norm(res['norm_mode'][0], axis=1)   # the lowest mode, the displacement of each atom

# The free energy there, its terms written to wb97mv_ts.out
tc = thermochemistry(mf_ts, res['freq_au'])
mol_ts.stdout.write('\nThermochemistry at the saddle point, the imaginary mode left out\n')
thermo.dump_thermo(mol_ts, tc)

print('saddle-point search converged:', conv)
print('E = %.10f Eh' % mf_ts.e_tot)
print('largest gradient component %.2e Eh/Bohr' % abs(gradient).max())
print('harmonic frequencies (cm^-1):', ' '.join('%.1f' % f for f in freq))
print('%d imaginary mode(s)%s' % (nimag, '' if nimag == 1 else ': not a first-order saddle point'))
print('atoms moving most in the lowest mode:',
      ', '.join('%s%d %.2f' % (mol_ts.atom_symbol(i), i, disp[i]) for i in numpy.argsort(-disp)[:5]))
e_tot, zpe, h_tot, s_tot = mf_ts.e_tot, tc['ZPE'][0], tc['H_tot'][0], tc['S_tot'][0]
print('free energy at %.2f K, %s, point group %s (symmetry number %d), the imaginary mode left out: G = %.10f Eh' % (
    TEMPERATURE, tc['standard_state'][0], tc['point_group'][0], tc['sym_number'][0], tc['G_tot'][0]))
print('  G - E = %.6f Eh: ZPE %.6f, thermal enthalpy %.6f, -T S %.6f%s' % (
    tc['G_tot'][0] - e_tot, zpe, h_tot - e_tot - zpe, -TEMPERATURE * s_tot,
    '' if QRRHO_FREQ0 is None else ' (quasi-RRHO entropy, freq0 %g cm^-1)' % QRRHO_FREQ0))
if QRRHO_FREQ0 is not None:
    print('  with the harmonic entropy of every mode: G = %.10f Eh' % tc['G_tot_rrho'][0])
