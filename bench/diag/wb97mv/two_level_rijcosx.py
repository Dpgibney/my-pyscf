'''mf.two_level for a model that is RIJCOSX already (caffeine wB97M-V/def2-TZVPD, aug_etb, grids (99,590)/(50,194)):
only the grids of the cheap model differ from the model's. Time, Fock builds on the production grids and the energy
for cheaper grids of the cheap model. Usage: two_level_rijcosx.py [xc_level,nlc_level ...]'''
import sys, time
from pyscf import gto, dft, df, lib
from molecules import CAFFEINE
T0 = time.perf_counter()
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
e0 = None
for spec in (sys.argv[1:] or ['2,1', '1,1', '1,0', '0,0']):
    xl, nl = (int(x) for x in spec.split(','))
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.atom_grid = (99, 590); mf.nlcgrids.atom_grid = (50, 194)
    mf = mf.RIJCOSX(auxbasis=df.aug_etb(mol))
    mf.two_level = True; mf.two_level_grids_level = xl; mf.two_level_nlcgrids_level = nl
    t = time.perf_counter(); e = mf.kernel(); dt = time.perf_counter() - t
    if e0 is None:
        e0 = e
    print(f'[{time.perf_counter() - T0:5.0f} s] cheap grids XC level {xl}, VV10 level {nl}: {dt:.1f} s, {mf.cycles} builds of the model, '
          f'conv {mf.converged}, E {e:.10f} ({e - e0:+.1e} from the first)', flush=True)
