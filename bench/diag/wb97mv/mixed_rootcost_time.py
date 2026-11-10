import time, numpy, sys
from pyscf import gto, dft, lib
from molecules import CAFFEINE
print('libcint:', sorted({l.split()[-1] for l in open('/proc/self/maps') if 'libcint' in l}))
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
z = numpy.load(sys.argv[1])
mf = dft.RKS(mol, xc='wb97m-v').RIJCOSX()
mf.max_memory = 6000; mf.with_df.max_memory = 6000
mf.mo_coeff, mf.mo_occ = z['mo_coeff'], z['mo_occ']
dm = mf.make_rdm1()
w = mf.with_df
from pyscf.sgx import sgx_jk
sgx_jk.mixed_range_supported = lambda *a, **k: True
w.mixed_range_ok = lambda *a, **k: True
def best(f, *a, **k):
    f(*a, **k); ts = []
    for _ in range(3):
        t = time.perf_counter(); f(*a, **k); ts.append(time.perf_counter() - t)
    return min(ts)
for level in (1, 2):
    w._rsh_df = {}; w.build(level=level)
    t1 = best(w.get_jk, dm, 1, None, False, True)
    t2 = best(w.get_jk, dm, 1, None, False, True, omega=.3)
    t3 = best(w.get_jk, dm, 1, None, False, True, omega=.3, mix=(.15, .85))
    print(f'level {level}: K {t1:.2f} s, K_LR {t2:.2f} s, mixed one pass {t3:.2f} s', flush=True)
