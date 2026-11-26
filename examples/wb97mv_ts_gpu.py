#!/usr/bin/env python
'''
The transition-state optimization of examples/wb97mv_ts.py on a GPU, with this workspace's GPU4PySCF:
wB97M-V/def2-TZVP, geomeTRIC's saddle-point search (transition=True) from an analytical Hessian with its grid
response, the harmonic frequencies at the saddle point it finds, which should have exactly one imaginary mode, and
the free energy there.

    bash /path/to/my-pyscf/scripts/gpu_setup.sh         # once: the venv (with geomeTRIC) and the tree for this GPU
    source /path/to/my-pyscf/scripts/gpu-env.sh perf
    python /path/to/my-pyscf/examples/wb97mv_ts_gpu.py

Writes wb97mv_ts_gpu.out (the SCF, gradient, Hessian and thermochemistry output) and wb97mv_ts_gpu.xyz (the saddle
point) to the current directory, and prints the frequencies, the atoms that move in the imaginary mode and the free
energy.

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

What differs from the CPU example:
  - the switches RIJCOSX and SGX_RESPONSE are the CPU example's, on the GPU tree's seminumerical K (gpu4pyscf.sgx,
    level-1 grid) and RI-J; RIJCOSX keeps its kernels in double precision (in single precision a density-fitted SCF
    stalls at conv_tol 1e-10: its whole potential is rebuilt at every cycle);
  - the CPHF equations are solved on two levels by default (hobj.cphf_two_level): a cheap model whose VV10 response
    is on a coarse grid, then the model's own response for what is left. The result is the model's own;
  - the grid of the CPHF's XC response is mf.cphf_grids, by default SG-1 (50,194) in GPU4PySCF; here level 2, as
    mf.set_second_grids(2) of the CPU example;
  - the quasi-RRHO entropy is computed here (quasi_rrho_entropy, the formula of PySCF's thermo(qrrho=True)): the
    PySCF of the GPU environment, 2.14, has no option for it in pyscf.hessian.thermo, which the CPU example calls.

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
from pyscf import gto, lib, lo, symm
from pyscf.data import nist
from pyscf.hessian import thermo
from pyscf.scf import addons
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

# For larger molecules: RI-J and seminumerical K (RIJCOSX) in the optimization's gradients, with the start Hessian and
# the frequencies still from exact J and K (as examples/wb97mv_ts.py). On this workspace's GTX 1650 Super it saves
# little at 16 atoms (see the notes below); it is for molecules whose exact J and K dominate a step.
RIJCOSX = False
# For larger molecules: seminumerical K and RI-J (the even-tempered fitting basis) in the CPHF response of the
# Hessians (as examples/wb97mv_ts.py): HCN <-> HNC 2e-6 in the Hessian, 0.01 cm^-1 in the frequencies. The SCF and every
# derivative integral stay exact.
SGX_RESPONSE = False
# For larger molecules: the start Hessian in a smaller basis, e.g. 'def2-svp' (None: the basis above). It only steers
# the search, which updates it from the gradients at every step; the frequencies at the end are those of the full
# model (see examples/wb97mv_ts.py for what it costs in steps).
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
    # XC grid and VV10 grid, as in examples/wb97mv_ts.py
    mf.grids.atom_grid = (99, 590)
    mf.nlcgrids.atom_grid = (50, 194)
    if rijcosx:
        mf = mf.RIJCOSX(auxbasis='aug_etb')     # RI-J in an even-tempered fitting basis, K on the SGX grid
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
    if SGX_RESPONSE:
        hobj.sgx_response = True                # seminumerical K in the CPHF response (and the exact J)
        hobj.sgx_response_auxbasis = 'aug_etb'  # RI-J there instead
    # Optional, the same Hessian: the seminumerical K also in the cheap model of the two-level CPHF. It took 10% off
    # the def2-TZVP Hessian of a 16-atom radical on a GTX 1650 Super (whose double precision is slow, and this K
    # runs in single precision); not measured on a card with fast double precision, and slower in def2-SVP.
    # hobj.cphf_two_level_parts = ('nlc', 'k')
    return cupy.asnumpy(hobj.kernel())


def orbitals_in_basis(mf0, mol):
    '''The converged orbitals and occupations of mf0 for a model of mol, the same molecule at the same geometry: those
    of mf0 in the same basis, else projected into the basis of mol and orthonormalized there'''
    if gto.same_basis_set(mf0.mol, mol):
        return mf0.mo_coeff, mf0.mo_occ
    s = mol.intor_symmetric('int1e_ovlp')

    def project(c):
        c = addons.project_mo_nr2nr(mf0.mol, cupy.asnumpy(c), mol)
        return cupy.asarray(lo.orth.vec_lowdin(c, s))
    if mf0.mo_coeff.ndim == 3:      # unrestricted: the orbitals of each spin
        return cupy.stack([project(c) for c in mf0.mo_coeff]), mf0.mo_occ
    return project(mf0.mo_coeff), mf0.mo_occ


def quasi_rrho_entropy(freq_au, temperature, freq0):
    '''Grimme's quasi-RRHO vibrational entropy (Eh/K) of the real modes of freq_au (atomic units, as harmonic_analysis
    gives them; the imaginary ones left out): the sum over the modes of w S_vib + (1 - w) S_rot, w = 1 / (1 + (freq0 /
    freq)^4), S_vib the harmonic oscillator's entropy and S_rot that of a free rotor with the moment of inertia
    mu = h / (8 pi^2 nu) of the same frequency, capped by an average one, mu' = mu B / (mu + B), B = 1e-44 kg m^2'''
    kB, h = nist.BOLTZMANN, nist.PLANCK
    r = kB / nist.HARTREE2J                         # the gas constant per molecule, Eh/K
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
    nu = numpy.asarray(freq_au).real
    nu = nu[nu > 0] * au2hz                         # Hz
    x = h * nu / (kB * temperature)
    s_vib = r * (x / numpy.expm1(x) - numpy.log(-numpy.expm1(-x)))
    mu = h / (8 * numpy.pi**2 * nu)
    mu = mu * 1e-44 / (mu + 1e-44)
    s_rot = r * (.5 + .5 * numpy.log(8 * numpy.pi**3 * mu * kB * temperature / h**2))
    w = 1 / (1 + (freq0 / (nu / (100 * nist.LIGHT_SPEED_SI)))**4)
    return float((w * s_vib + (1 - w) * s_rot).sum())


def thermochemistry(mf, freq_au):
    '''The thermochemistry (the dict of pyscf.hessian.thermo.thermo, Eh and Eh/K) of the stationary point of the
    converged mf from its harmonic frequencies (atomic units, the imaginary ones negative, as harmonic_analysis with
    imaginary_freq=False gives them; thermo leaves those out), at TEMPERATURE. The standard state: 1 atm, or 1 mol/L
    with a solvent model (an ideal gas at the pressure c R T, 24.5 atm at 298.15 K). With QRRHO_FREQ0, the vibrational
    entropy of quasi_rrho_entropy in S_vib, S_tot, G_vib and G_tot; G_tot_rrho is G with the harmonic entropy.'''
    pressure, state = 101325., '1 atm'
    if getattr(mf, 'with_solvent', None) is not None:
        pressure, state = 1e3 * nist.AVOGADRO * nist.BOLTZMANN * TEMPERATURE, '1 mol/L'
    # the point group within 0.01 Bohr, also for the symmetry number that thermo takes from it (SYMMETRY_NUMBER)
    with lib.temporary_env(symm.geom, TOLERANCE=1e-2):
        group = symm.detect_symm(mf.mol._atom)[0]
        res = thermo.thermo(mf, freq_au, TEMPERATURE, pressure)
    res['point_group'] = (group, '')
    res['standard_state'] = (state, '')
    res['G_tot_rrho'] = res['G_tot']
    if QRRHO_FREQ0 is not None:
        ds = quasi_rrho_entropy(freq_au, TEMPERATURE, QRRHO_FREQ0) - res['S_vib'][0]
        _add_entropy(res, 'vib', ds)
    if SYMMETRY_NUMBER is not None:
        # the rotational partition function goes as 1 / sigma
        _add_entropy(res, 'rot', nist.BOLTZMANN / nist.HARTREE2J * numpy.log(res['sym_number'][0] / SYMMETRY_NUMBER),
                     'G_tot_rrho')
        res['sym_number'] = (SYMMETRY_NUMBER, '')
    return res


def _add_entropy(res, part, ds, *more):
    '''ds (Eh/K) added to the entropy S_<part> of the result res of thermo, and S_tot, and -T ds to G_<part>, G_tot and
    the G of the keys more'''
    for key in ('S_' + part, 'S_tot'):
        res[key] = (res[key][0] + ds, 'Eh/K')
    for key in ('G_' + part, 'G_tot') + more:
        res[key] = (res[key][0] - TEMPERATURE * ds, 'Eh')


# The start Hessian at the guess (geomeTRIC updates it from the gradients at every step)
mol_start = mol
if START_HESSIAN_BASIS is not None:
    mol_start = mol.copy()
    mol_start.basis = START_HESSIAN_BASIS
    mol_start.build()
mf_start = make_mf(mol_start)
mf_start.kernel()
h0 = analytical_hessian(mf_start)

# The saddle-point search (geomeTRIC's default thresholds, shown so that they can be edited)
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
mol_ts.tofile('wb97mv_ts_gpu.xyz')

# Frequencies at the saddle point. mol_ts is where the search's last SCF and gradient were evaluated: the SCF there
# starts from that density, and without RIJCOSX the search's last gradient is this model's at this geometry.
mf_ts = make_mf(mol_ts)
mf_ts.kernel(dm0=g_scanner.base.make_rdm1())
gradient = cupy.asnumpy(mf_ts.Gradients().kernel() if RIJCOSX else g_scanner.de)
h = analytical_hessian(mf_ts)
res = thermo.harmonic_analysis(mol_ts, h, imaginary_freq=False)
freq = res['freq_wavenumber']       # imaginary frequencies as negative numbers
nimag = int((freq < 0).sum())
disp = numpy.linalg.norm(res['norm_mode'][0], axis=1)   # the lowest mode, the displacement of each atom

# The free energy there, its terms written to wb97mv_ts_gpu.out
tc = thermochemistry(mf_ts, res['freq_au'])
mol_ts.stdout.write('\nThermochemistry at the saddle point, the imaginary mode left out%s\n' % (
    '' if QRRHO_FREQ0 is None else
    ", the vibrational entropy by Grimme's quasi-RRHO (freq0 = %g cm^-1)" % QRRHO_FREQ0))
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
