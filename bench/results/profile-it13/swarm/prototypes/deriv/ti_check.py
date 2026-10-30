'''Translational invariance of the 2e parts of _partial_hess_ejk: sum_B ej[A,B] and sum_B ek[A,B]
should vanish, so that the diagonal blocks (which alone need int2e_ipip1) follow from the off-diagonal
ones. Single thread.'''
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, WATER
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess

for name, atom in (('water', WATER), ('benzene', BENZENE)):
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    h = mf.Hessian()
    t0 = time.perf_counter()
    e1, ej, ek = rhf_hess._partial_hess_ejk(h, mf.mo_energy, mf.mo_coeff, mf.mo_occ, range(mol.natm))
    t1 = time.perf_counter()
    for lab, e in (('ej', ej), ('ek', ek)):
        diag = numpy.array([e[a, a] for a in range(mol.natm)])
        off = numpy.array([-sum(e[a, b] for b in range(mol.natm) if b != a) for a in range(mol.natm)])
        print(f'{name} {lab}: max|diag| {abs(diag).max():.3e}, max|diag - (-sum_offdiag)| '
              f'{abs(diag - off).max():.2e}, max|e| {abs(e).max():.3e}')
    print(f'  ({t1-t0:.1f} s)')
