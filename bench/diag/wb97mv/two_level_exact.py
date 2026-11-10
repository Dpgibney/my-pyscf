import sys, time
from pyscf import gto, dft, lib
from molecules import CAFFEINE
T0 = time.perf_counter()
E_REF = -680.3497717994
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
for spec in sys.argv[1:]:
    xl, nl = (int(x) for x in spec.split(','))
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
    mf.two_level = True; mf.two_level_grids_level = xl; mf.two_level_nlcgrids_level = nl
    t = time.perf_counter(); e = mf.kernel(); dt = time.perf_counter() - t
    print(f'[{time.perf_counter() - T0:5.0f} s] exact model, cheap grids XC level {xl}, VV10 level {nl}: {dt:.1f} s, {mf.cycles} builds of the model, '
          f'conv {mf.converged}, E - E(plain SCF) {e - E_REF:+.1e}', flush=True)
