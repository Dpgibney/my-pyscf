import sys, time, numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, dft, lib
from pyscf.dft import numint
def tm(f, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter()-t0)
    return min(ts)*1e3, sorted(ts)[len(ts)//2]*1e3
for name, atom, basis in [('caffeine', molecules.CAFFEINE, 'def2-svp'), ('c20h42', molecules.n_alkane(20), 'def2-svp'), ('caffeine', molecules.CAFFEINE, 'def2-tzvp')]:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    nao = mol.nao
    rng = numpy.random.default_rng(0)
    dm = rng.standard_normal((nao,nao)); dm = dm + dm.T
    ao_loc = mol.ao_loc_nr()
    print(name, basis, 'nbas', mol.nbas, 'nao', nao)
    print('  get_overlap_cond ms (min, median):', tm(lambda: mol.get_overlap_cond()))
    print('  condense absmax ms:', tm(lambda: lib.condense('NP_absmax', dm, ao_loc)))
    print('  condense_to_shell ms:', tm(lambda: mol.condense_to_shell(dm, 'absmax')))
    ni = dft.numint.NumInt()
    class G: pass
    g = G()
    print('  _rho_setup ms:', tm(lambda: numint._rho_setup(ni, mol, dm, 1, g)))
