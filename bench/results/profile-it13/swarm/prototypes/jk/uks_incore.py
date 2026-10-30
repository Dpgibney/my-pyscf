"""In-core J/K: one symmetric dm (fused one-pass kernel) vs two symmetric dms (BLAS batch path, as UKS),
1 thread. Usage: uks_incore.py SYSTEM BASIS"""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE
from pyscf import gto, scf, lib
from pyscf.scf import _vhf
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE}[sys.argv[1]]
mol = gto.M(atom=atom, basis=sys.argv[2], verbose=0)
t0 = time.perf_counter()
eri = mol.intor('int2e', aosym='s8')
print(f'nao {mol.nao}, ERIs {eri.nbytes/1e9:.2f} GB built in {time.perf_counter()-t0:.1f} s')
dm = scf.RHF(mol).get_init_guess(key='minao')
dma = dm * .55; dmb = dm * .45
def best(f, n=5):
    b = 9e9
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t0)
    return b, r
t1, (vj1, vk1) = best(lambda: _vhf.incore(eri, dma, hermi=1))
t2, (vj2, vk2) = best(lambda: _vhf.incore(eri, numpy.asarray([dma, dmb]), hermi=1))
tj, _ = best(lambda: _vhf.incore(eri, numpy.asarray([dma, dmb]), hermi=1, with_k=False))
tk, _ = best(lambda: _vhf.incore(eri, numpy.asarray([dma, dmb]), hermi=1, with_j=False))
print(f'1 dm fused J+K {t1*1e3:.1f} ms; 2 dms batch J+K {t2*1e3:.1f} ms (J {tj*1e3:.1f}, K {tk*1e3:.1f}); '
      f'2 x fused would be {2*t1*1e3:.1f} ms')
print('check', abs(vk2[0]-vk1).max(), abs(vj2[0]-vj1).max())
