'''The point group and rotational symmetry number that PySCF finds (symm.detect_symm, thermo.rotational_symmetry_number)
for geometries at several values of symm.geom.TOLERANCE (Bohr; detect_symm divides it by sqrt(1 + natm)): the saddle
points of the examples' searches (XYZ files), and controls: water with 0.002, 0.01 and 0.02 Angstrom of noise (C2v
broken), and CHFClBr (C1).

    python symmetry_tolerance.py XYZ [XYZ ...]
'''
import sys
import numpy
from pyscf import gto, lib, symm
from pyscf.hessian import thermo

TOLS = (1e-5, 1e-3, 1e-2, 2e-2, 5e-2)


def groups(atoms):
    out = []
    for t in TOLS:
        with lib.temporary_env(symm.geom, TOLERANCE=t):
            g = symm.detect_symm(atoms)[0]
            m = gto.M(atom=atoms, basis='sto-3g', unit='B', verbose=0, spin=None)
            out.append('%s/%g' % (g, thermo.rotational_symmetry_number(m)))
    return out


print('%-28s ' % 'geometry' + ' '.join('%-12s' % ('tol %g' % t) for t in TOLS))
for f in sys.argv[1:]:
    mol = gto.M(atom=f, basis='sto-3g', verbose=0, spin=None)
    print('%-28s ' % f.split('/')[-1][:28] + ' '.join('%-12s' % x for x in groups(mol._atom)))
rng = numpy.random.default_rng(3)
w = gto.M(atom='O 0 0 0.1174; H -0.757 0 -0.4696; H 0.757 0 -0.4696', basis='sto-3g', verbose=0)
for amp in (0.002, 0.01, 0.02):
    a = [(s, numpy.array(c) + rng.normal(size=3) * amp / 0.529177) for s, c in w._atom]
    print('%-28s ' % ('water + noise %.3f A' % amp) + ' '.join('%-12s' % x for x in groups(a)))
c1 = gto.M(atom='C 0 0 0; H 0 0 1.09; F 1.33 0 -0.45; Cl -0.84 1.45 -0.59; Br -0.95 -1.55 -0.65',
           basis='sto-3g', verbose=0)
print('%-28s ' % 'CHFClBr (C1)' + ' '.join('%-12s' % x for x in groups(c1._atom)))
