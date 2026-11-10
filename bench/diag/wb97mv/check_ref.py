'''The saved "exact" SCF of prof_scf_grad.py (PySCF's default convergence: conv_tol 1e-9, incremental direct J/K):
its energy and orbital gradient recomputed from one full exact Fock build of its density.'''
import sys
import time
import numpy
import pyscf
from pyscf import gto, dft, lib
from molecules import CAFFEINE
print('pyscf', pyscf.__file__, 'threads', lib.num_threads(), flush=True)
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
z = numpy.load(sys.argv[1])
mf = dft.RKS(mol, xc='wb97m-v')
mf.grids.atom_grid = (99, 590)
mf.nlcgrids.atom_grid = (50, 194)
mo_coeff, mo_occ = z['mo_coeff'], z['mo_occ']
dm = mf.make_rdm1(mo_coeff, mo_occ)
h1e = mf.get_hcore()
s1e = mf.get_ovlp()
w = numpy.linalg.eigvalsh(s1e)
print(f'overlap: smallest eigenvalue {w[0]:.2e}, condition number {w[-1] / w[0]:.2e}')
t = time.perf_counter()
vhf = mf.get_veff(mol, dm)
e = mf.energy_tot(dm, h1e, vhf)
g = mf.get_grad(mo_coeff, mo_occ, h1e + numpy.asarray(vhf))
print(f'full exact build {time.perf_counter() - t:.1f} s')
print(f'stored e_tot {float(z["e_tot"]):.10f}; recomputed E[D] {e:.10f}; difference {e - float(z["e_tot"]):+.2e}')
print(f'orbital gradient: norm {numpy.linalg.norm(g):.2e}, norm/sqrt(size) {numpy.linalg.norm(g) / numpy.sqrt(g.size):.2e} '
      f'(PySCF tests the latter against conv_tol_grad = sqrt(conv_tol) = 3.2e-05), max {abs(g).max():.2e}')
