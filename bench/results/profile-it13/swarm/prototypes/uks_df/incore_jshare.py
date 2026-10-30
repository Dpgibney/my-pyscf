'''Share of J in the in-core batched J/K of many dms (UKS response: 2n dms whose J is only used as
alpha+beta sums), naphthalene/def2-SVP, 1 thread. Also naux of caffeine for the DF estimates.'''
import time, numpy
from pyscf import gto, lib
from pyscf.scf import _vhf
from pyscf.df import addons
atom = '''C 0.000 0.710 0.000; C 0.000 -0.710 0.000; C 1.242 1.403 0.000; C 1.242 -1.403 0.000;
C -1.242 1.403 0.000; C -1.242 -1.403 0.000; C 2.433 0.705 0.000; C 2.433 -0.705 0.000;
C -2.433 0.705 0.000; C -2.433 -0.705 0.000; H 1.244 2.490 0.000; H 1.244 -2.490 0.000;
H -1.244 2.490 0.000; H -1.244 -2.490 0.000; H 3.377 1.245 0.000; H 3.377 -1.245 0.000;
H -3.377 1.245 0.000; H -3.377 -1.245 0.000'''
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
nao = mol.nao
eri = mol.intor('int2e', aosym='s8')
rng = numpy.random.default_rng(0)
def best(f, n=2):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)
for n in (20,):
    d = rng.standard_normal((n, nao, nao)) * .01
    ds = d + d.transpose(0, 2, 1)
    for hermi, dd in ((0, d), (1, ds)):
        tj = best(lambda: _vhf._incore_s8_batch(eri, dd, hermi, True, False))
        tk = best(lambda: _vhf._incore_s8_batch(eri, dd, hermi, False, True))
        tjh = best(lambda: _vhf._incore_s8_batch(eri, dd[:n//2], hermi, True, False))
        print(f'{n} dms hermi={hermi}: J {tj:.3f} s, K {tk:.3f} s, J share {tj/(tj+tk):.2f}; '
              f'J of {n//2} sums {tjh:.3f} s -> saves {(tj-tjh)/(tj+tk):.2f} of J+K')
import sys; sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE as CAFF
for basis in ('def2-svp', 'def2-tzvp'):
    m = gto.M(atom=CAFF, basis=basis, verbose=0)
    from pyscf.df import df
    dfo = df.DF(m)
    auxmol = addons.make_auxmol(m, dfo.auxbasis)
    print(f'caffeine/{basis}: nao {m.nao}, auxbasis {dfo.auxbasis}, naux {auxmol.nao}, nocc {m.nelectron//2}')
