import sys, time, cProfile, pstats, io
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, dft, lib
mol = gto.M(atom=molecules.BENZENE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP'); mf.chkfile = None
t0 = time.perf_counter(); mf.kernel(); print('scf', time.perf_counter()-t0, mf.cycles, flush=True)
hobj = mf.Hessian()
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable(); h = hobj.kernel(); pr.disable()
print('hessian wall', time.perf_counter()-t0)
s = pstats.Stats(pr)
s.sort_stats('tottime')
buf = io.StringIO(); s.stream = buf
s.print_stats(40)
s.print_callers('c_einsum')
s.print_callers(r'\(dot\)$|numpy.*dot|_dgemm')
print(buf.getvalue())
buf = io.StringIO(); s.stream = buf
s.print_callers(r'einsumfunc.py.*\(einsum\)')
print(buf.getvalue())
pr.dump_stats('/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/py/hess_benzene.prof')
