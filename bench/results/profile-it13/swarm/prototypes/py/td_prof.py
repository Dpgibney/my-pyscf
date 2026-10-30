import sys, time, cProfile, pstats, io
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, dft, lib
mol = gto.M(atom=molecules.BENZENE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP'); mf.chkfile = None
t0 = time.perf_counter(); mf.kernel(); print('scf', time.perf_counter()-t0, mf.cycles, flush=True)
print('ngrids', mf.grids.weights.size, 'nao', mol.nao)
td = mf.TDDFT(); td.nstates = 5
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable(); td.kernel(); pr.disable()
print('tddft wall', time.perf_counter()-t0)
s = pstats.Stats(pr)
s.sort_stats('tottime')
buf = io.StringIO(); s.stream = buf
s.print_stats(30)
s.print_callers(r'einsumfunc.py.*\(einsum\)')
s.print_callers(r'condense')
s.print_callers(r"method 'dot'")
s.print_callers(r'get_overlap_cond')
print(buf.getvalue())
pr.dump_stats('/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/py/tddft_benzene.prof')
