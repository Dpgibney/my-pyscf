'''Batched in-core J/K (hermi=1) per dm vs the 1-dm kernel per dm, naphthalene/def2-SVP, 1 thread.'''
import time, numpy
from pyscf import gto, lib
from pyscf.scf import _vhf
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
d = rng.standard_normal((8, nao, nao)) * .01
d = d + d.transpose(0, 2, 1)
t1 = best(lambda: _vhf._incore_s8_batch(eri, d[:1], 1, True, True))
print(f'1-dm kernel: {t1*1e3:.0f} ms per dm')
for n in (2, 3, 4, 6, 8):
    t = best(lambda: _vhf._incore_s8_batch(eri, d[:n], 1, True, True))
    print(f'batch path, {n} dms: {t*1e3:.0f} ms = {t/n*1e3:.0f} ms per dm')
