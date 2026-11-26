'''The thermochemistry of examples/wb97mv_ts.py (pyscf.hessian.thermo with qrrho, this workspace's PySCF) and of
examples/wb97mv_ts_gpu.py (its quasi_rrho_entropy on thermo's RRHO, for PySCF 2.14) against ORCA 6.1.1's, on ORCA's
own geometry, energy and frequencies (ORCA's default: Grimme's quasi-RRHO vibrational entropy, freq0 100 cm^-1, the
imaginary modes left out). Then the GPU example's quasi_rrho_entropy against thermo(qrrho=True) on soft modes from 1
to 3000 cm^-1, and the standard state of 1 mol/L that both use with a solvent model.

    python check_thermo.py ORCA_OUTPUT ORCA_XYZ [CHARGE SPIN]

The functions are taken from the examples' files (ast), without running the examples.
'''
import ast
import os
import re
import sys
import types
import numpy
from pyscf import gto, lib, symm
from pyscf.data import nist
from pyscf.hessian import thermo

W = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
out, xyz = sys.argv[1], sys.argv[2]
charge, spin = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 0)


def load(path, names):
    '''The top-level functions and assignments named in names of an example, executed without the rest of it'''
    tree = ast.parse(open(path).read())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in names for t in n.targets))]
    ns = {'numpy': numpy, 'nist': nist, 'thermo': thermo, 'lib': lib, 'symm': symm}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, 'exec'), ns)
    return ns


NAMES = ('TEMPERATURE', 'QRRHO_FREQ0', 'SYMMETRY_NUMBER', 'quasi_rrho_entropy', 'thermochemistry', '_add_entropy')
gpu = load(os.path.join(W, 'examples', 'wb97mv_ts_gpu.py'), NAMES)
# the CPU example needs thermo's qrrho (this workspace's PySCF; not 2.14's) and may be absent (node1)
has_qrrho = 'qrrho' in thermo.thermo.__code__.co_varnames
cpu_path = os.path.join(W, 'examples', 'wb97mv_ts.py')
has_qrrho = has_qrrho and os.path.exists(cpu_path)
cpu = load(cpu_path, NAMES) if has_qrrho else None

# ORCA's numbers: the last frequency block and the thermochemistry
text = open(out).read()
block = text.split('VIBRATIONAL FREQUENCIES')[-1].split('NORMAL MODES')[0]
freq_cm = numpy.array([float(x) for x in re.findall(r'^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1', block, re.M)])
freq_src = 'the output (0.01 cm^-1)'
hess = os.path.splitext(out)[0] + '.hess'
if os.path.exists(hess) and os.environ.get('FREQ_FROM_OUTPUT') is None:
    # the frequencies of ORCA's .hess file, with six decimals
    lines = open(hess).read().split('$vibrational_frequencies')[1].split('$')[0].split()
    freq_cm = numpy.array([float(x) for x in lines[1:][1::2]])
    freq_src = os.path.basename(hess) + ' (1e-6 cm^-1)'
freq_cm = freq_cm[freq_cm != 0]
th = text.split('THERMOCHEMISTRY AT')[-1]


def orca(label):
    return float(re.search(re.escape(label) + r'\s+\.\.\.\s+(-?\d+\.\d+) Eh', th).group(1))


ref = {k: orca(k) for k in ('Electronic energy', 'Zero point energy', 'Thermal vibrational correction',
                            'Thermal rotational correction', 'Thermal translational correction',
                            'Thermal Enthalpy correction', 'Total Enthalpy', 'Electronic entropy',
                            'Vibrational entropy', 'Rotational entropy', 'Translational entropy',
                            'Final Gibbs free energy')}
rot_orca = numpy.array([float(x) for x in re.search(r'Rotational constants in cm-1:\s+(\S+)\s+(\S+)\s+(\S+)',
                                                     th).groups()])
sym_orca = int(re.search(r'Symmetry Number:\s+(\d+)', th).group(1))

mol = gto.M(atom=xyz, basis='sto-3g', charge=charge, spin=spin, verbose=0)
au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
freq_au = freq_cm * 100 * nist.LIGHT_SPEED_SI / au2hz
model = types.SimpleNamespace(mol=mol, e_tot=ref['Electronic energy'])
T = gpu['TEMPERATURE']
print('ORCA: %d frequencies from %s (%d imaginary, lowest %s cm^-1), symmetry number %d; PySCF %s, thermo with '
      'qrrho: %s' % (len(freq_cm), freq_src, (freq_cm < 0).sum(), ' '.join('%.4f' % f for f in numpy.sort(freq_cm)[:4]),
      sym_orca,
    __import__('pyscf').__version__, has_qrrho))

rot_pyscf = thermo.rotation_const(mol.atom_mass_list(isotope_avg=True), mol.atom_coords()
                                  - numpy.einsum('z,zx->x', mol.atom_mass_list(isotope_avg=True), mol.atom_coords())
                                  / mol.atom_mass_list(isotope_avg=True).sum(), 'wavenumber')
print('rotational constants (cm^-1): ORCA %s, PySCF %s' % (rot_orca, numpy.array2string(rot_pyscf, precision=6)))


def compare(name, res):
    e = res['E0'][0]
    ours = {
        'Zero point energy': res['ZPE'][0],
        'Thermal vibrational correction': res['E_vib'][0] - res['ZPE'][0],
        'Thermal rotational correction': res['E_rot'][0],
        'Thermal translational correction': res['E_trans'][0],
        'Thermal Enthalpy correction': res['H_tot'][0] - res['E_tot'][0],
        'Total Enthalpy': res['H_tot'][0],
        'Electronic entropy': T * res['S_elec'][0],
        'Vibrational entropy': T * res['S_vib'][0],
        'Rotational entropy': T * res['S_rot'][0],
        'Translational entropy': T * res['S_trans'][0],
        'Final Gibbs free energy': res['G_tot'][0],
    }
    print('%s against ORCA (Eh):' % name)
    worst = 0
    for k, v in ours.items():
        d = v - ref[k]
        worst = max(worst, abs(d))
        print('   %-34s ORCA %15.8f  here %15.8f  diff %+.1e' % (k, ref[k], v, d))
    print('   largest difference %.1e Eh; G - E: ORCA %.8f, here %.8f' % (
        worst, ref['Final Gibbs free energy'] - e, res['G_tot'][0] - e))
    return worst


worst_gpu = compare("the GPU example's thermochemistry (quasi_rrho_entropy)", gpu['thermochemistry'](model, freq_au))
if has_qrrho:
    worst_cpu = compare("the CPU example's thermochemistry (thermo, qrrho=True)",
                        cpu['thermochemistry'](model, freq_au))

# quasi_rrho_entropy against thermo(qrrho=True), soft modes included; and the limit freq0 -> 0 (harmonic)
soft_cm = numpy.array([1., 5., 20., 50., 80., 100., 150., 300., 1000., 3000.])
soft_au = soft_cm * 100 * nist.LIGHT_SPEED_SI / au2hz
s_rrho = thermo.thermo(model, soft_au, T)['S_vib'][0]
print('soft modes %s cm^-1, T S_vib (Eh): harmonic %.10f' % (soft_cm, T * s_rrho))
for f0 in (50., 100., 200.):
    s_gpu = gpu['quasi_rrho_entropy'](soft_au, T, f0)
    line = '   freq0 %5.1f: quasi_rrho_entropy %.10f' % (f0, T * s_gpu)
    if has_qrrho:
        s_lib = thermo.thermo(model, soft_au, T, qrrho=True, qrrho_freq0=f0)['S_vib'][0]
        line += ', thermo(qrrho=True) %.10f, relative difference %.1e' % (T * s_lib, abs(s_gpu / s_lib - 1))
    print(line)
print('   freq0 1e-9 (harmonic limit): relative difference to thermo %.1e' % abs(
    gpu['quasi_rrho_entropy'](soft_au, T, 1e-9) / s_rrho - 1))

# 1 mol/L with a solvent model: G higher by R T ln(c R T / 1 atm)
solv = types.SimpleNamespace(mol=mol, e_tot=ref['Electronic energy'], with_solvent=object())
for name, ns in (('GPU', gpu), ('CPU', cpu)):
    if ns is None:
        continue
    g1 = ns['thermochemistry'](model, freq_au)
    gm = ns['thermochemistry'](solv, freq_au)
    expect = nist.BOLTZMANN * T / nist.HARTREE2J * numpy.log(1e3 * nist.AVOGADRO * nist.BOLTZMANN * T / 101325.)
    print('%s example, 1 mol/L - 1 atm: G %+.8f Eh = %.4f kcal/mol (R T ln(c R T / p0) = %.4f kcal/mol); '
          'standard states %r, %r' % (name, gm['G_tot'][0] - g1['G_tot'][0],
                                       (gm['G_tot'][0] - g1['G_tot'][0]) * nist.HARTREE2J * nist.AVOGADRO / 4184,
                                       expect * nist.HARTREE2J * nist.AVOGADRO / 4184,
                                       g1['standard_state'][0], gm['standard_state'][0]))

# The point group of a nearly symmetric saddle point: H + H2, its two H-H distances 2e-3 Angstrom apart (as some of the
# searches end), D-infinity-h within the examples' 0.01 Bohr (symmetry number 2), C-infinity-v within PySCF's 1e-5;
# SYMMETRY_NUMBER = 1 moves G by R T ln 2
h3 = gto.M(atom='H 0 0 -0.9283; H 0 0 0; H 0 0 0.9263', basis='sto-3g', spin=1, verbose=0)
f3 = numpy.array([-1039.7, 902.4, 902.4, 2072.9]) * 100 * nist.LIGHT_SPEED_SI / au2hz
m3 = types.SimpleNamespace(mol=h3, e_tot=-1.65)
for name, ns in (('GPU', gpu), ('CPU', cpu)):
    if ns is None:
        continue
    r = ns['thermochemistry'](m3, f3)
    ns['SYMMETRY_NUMBER'] = 1
    r1 = ns['thermochemistry'](m3, f3)
    ns['SYMMETRY_NUMBER'] = None
    print('%s example, H3 2e-3 A from symmetric: %s, symmetry number %d (PySCF thermo alone: %d); with '
          'SYMMETRY_NUMBER = 1: G %+.3e Eh (R T ln 2 = %.3e), the harmonic-entropy G %+.3e Eh' % (
              name, r['point_group'][0], r['sym_number'][0], thermo.thermo(m3, f3, 298.15)['sym_number'][0],
              r1['G_tot'][0] - r['G_tot'][0], nist.BOLTZMANN / nist.HARTREE2J * 298.15 * numpy.log(2),
              r1['G_tot_rrho'][0] - r['G_tot_rrho'][0]))
