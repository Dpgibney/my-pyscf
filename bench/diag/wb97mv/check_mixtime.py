import time, numpy, sys
from pyscf import gto, dft, lib
from pyscf.sgx import sgx_jk
from molecules import CAFFEINE
mol = gto.M(atom=CAFFEINE, basis='def2-tzvpd', verbose=0)
z = numpy.load(sys.argv[1])
mo_coeff, mo_occ = z['mo_coeff'], z['mo_occ']
mf = dft.RKS(mol, xc='wb97m-v').RIJCOSX()
mf.max_memory = 6000; mf.with_df.max_memory = 6000
mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ
dm = mf.make_rdm1()
w = mf.with_df
def timed(f, *a, **k):
    f(*a, **k)
    t = time.perf_counter(); r = f(*a, **k); return r, time.perf_counter() - t
for level in (1, 2):
    w._rsh_df = {}; w.build(level=level)
    (_, k), t1 = timed(w.get_jk, dm, 1, None, False, True)
    (_, kl), t2 = timed(w.get_jk, dm, 1, None, False, True, omega=.3)
    (_, km), t3 = timed(w.get_jk, dm, 1, None, False, True, omega=.3, mix=(.15, .85))
    ref = .15 * k + .85 * kl
    print(f'level {level}: K {t1:.2f} s + K_LR {t2:.2f} s = {t1+t2:.2f} s; one pass {t3:.2f} s; '
          f'E_K diff {-.25*numpy.einsum("ij,ji", dm, km - ref):+.1e}, max|dK| {abs(km-ref).max():.1e}', flush=True)
# the response's K from the factors, 72 densities, level 1
w._rsh_df = {}; w.build(level=1)
nao, nmo = mo_coeff.shape
mocc = mo_coeff[:, mo_occ > 0]; nocc = mocc.shape[1]
rng = numpy.random.default_rng(7)
x = numpy.zeros((72, nmo, nocc)); x[:, nocc:] = rng.standard_normal((72, nmo - nocc, nocc))
x /= numpy.linalg.norm(x.reshape(72, -1), axis=1)[:, None, None]
a = lib.einsum('pa,kai->kpi', mo_coeff, x)
print('estimated size of a block of F for 72 dms per 1000 points: %.2f GB' % (72 * nao * 1000 * 8e-9), flush=True)
t = time.perf_counter(); k1 = w.get_k_mo(a, mocc, mo_coeff); t1 = time.perf_counter() - t
t = time.perf_counter(); k2 = w.get_k_mo(a, mocc, mo_coeff, omega=.3); t2 = time.perf_counter() - t
t = time.perf_counter(); k3 = w.get_k_mo(a, mocc, mo_coeff, omega=.3, mix=(.15, .85)); t3 = time.perf_counter() - t
ref = .15 * k1 + .85 * k2
print(f'72 response densities, level 1: K {t1:.1f} s + K_LR {t2:.1f} s = {t1+t2:.1f} s; one pass {t3:.1f} s; max diff {abs(k3-ref).max():.1e} of {abs(ref).max():.2e}', flush=True)
