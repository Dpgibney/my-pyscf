import sys, time, numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto, lib
mol = gto.M(atom=CAFFEINE, basis='def2-svp', charge=1, spin=1, verbose=0)
dm = numpy.random.default_rng(0).standard_normal((mol.nao, mol.nao))
def best(f, n=20):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts), numpy.median(ts)
print('threads', lib.num_threads(), 'nbas', mol.nbas)
print('get_overlap_cond best/median ms: %.3f / %.3f' % tuple(1e3*x for x in best(mol.get_overlap_cond)))
print('condense_to_shell absmax best/median ms: %.3f / %.3f' % tuple(1e3*x for x in best(lambda: mol.condense_to_shell(dm, 'absmax'))))
