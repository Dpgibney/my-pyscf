'''In-core s8 J/K for UKS-shaped input (2 symmetric dms) vs the 1-dm kernel, 1 thread.
Times: current UKS path (_incore_s8_batch, n_dm=2), its J and K parts, 2x the 1-dm kernel,
and J(Da+Db)+K(Da) then K(Db) with the existing 1-dm kernel.'''
import sys, time, ctypes
import numpy
from pyscf import gto, scf, lib
from pyscf.scf import _vhf

which = sys.argv[1] if len(sys.argv) > 1 else 'benzene'
if which == 'benzene':
    atom = '''C 0.000 1.396 0.000; C 1.209 0.698 0.000; C 1.209 -0.698 0.000;
C 0.000 -1.396 0.000; C -1.209 -0.698 0.000; C -1.209 0.698 0.000;
H 0.000 2.479 0.000; H 2.147 1.240 0.000; H 2.147 -1.240 0.000;
H 0.000 -2.479 0.000; H -2.147 -1.240 0.000; H -2.147 1.240 0.000'''
else:  # naphthalene
    atom = '''C 0.000 0.710 0.000; C 0.000 -0.710 0.000; C 1.242 1.403 0.000; C 1.242 -1.403 0.000;
C -1.242 1.403 0.000; C -1.242 -1.403 0.000; C 2.433 0.705 0.000; C 2.433 -0.705 0.000;
C -2.433 0.705 0.000; C -2.433 -0.705 0.000; H 1.244 2.490 0.000; H 1.244 -2.490 0.000;
H -1.244 2.490 0.000; H -1.244 -2.490 0.000; H 3.377 1.245 0.000; H 3.377 -1.245 0.000;
H -3.377 1.245 0.000; H -3.377 -1.245 0.000'''
mol = gto.M(atom=atom, basis='def2-svp', charge=1, spin=1, verbose=0)
nao = mol.nao
t = time.perf_counter()
eri = mol.intor('int2e', aosym='s8')
print(f'{which}: nao {nao}, eri {eri.nbytes/1e6:.0f} MB, built in {time.perf_counter()-t:.1f} s, threads {lib.num_threads()}')

# UKS-like dms: two symmetric density matrices from random orbitals
rng = numpy.random.default_rng(1)
c = rng.standard_normal((nao, nao)) * .1
na, nb = 21, 20
dma = c[:, :na] @ c[:, :na].T
dmb = c[:, :nb] @ c[:, :nb].T
dms = numpy.asarray([dma, dmb])

def best(f, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t0)
    return min(ts), r

tb, (vj2, vk2) = best(lambda: _vhf.incore(eri, dms, hermi=1))
tj, _ = best(lambda: _vhf._incore_s8_batch(eri, dms, 1, True, False))
tk, _ = best(lambda: _vhf._incore_s8_batch(eri, dms, 1, False, True))
t1a, (vja, vka) = best(lambda: _vhf.incore(eri, dms[:1], hermi=1))
t1b, (vjb, vkb) = best(lambda: _vhf.incore(eri, dms[1:], hermi=1))

# J of the total density + K of Da in one pass, then K of Db: existing kernel, two passes
def jsum_k2():
    null = lib.c_null_ptr()
    idx = numpy.arange(nao)
    ds = dma + dmb
    dmtri = lib.pack_tril(ds + ds.T); dmtri[idx*(idx+1)//2+idx] *= .5
    vjtri = numpy.empty_like(dmtri)
    za = numpy.empty((nao, nao)); zb = numpy.empty((nao, nao))
    _vhf.libcvhf.CVHFnrs8_incore_jk_1dm(
        vjtri.ctypes.data_as(ctypes.c_void_p), za.ctypes.data_as(ctypes.c_void_p),
        eri.ctypes.data_as(ctypes.c_void_p), dmtri.ctypes.data_as(ctypes.c_void_p),
        numpy.ascontiguousarray(dma).ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nao))
    _vhf.libcvhf.CVHFnrs8_incore_jk_1dm(
        null, zb.ctypes.data_as(ctypes.c_void_p),
        eri.ctypes.data_as(ctypes.c_void_p), dmtri.ctypes.data_as(ctypes.c_void_p),
        numpy.ascontiguousarray(dmb).ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nao))
    return lib.unpack_tril(vjtri), numpy.asarray([za + za.T, zb + zb.T])
tjs, (vjs, vks) = best(jsum_k2)

print(f'current UKS path (_incore_s8_batch, 2 dms): {tb*1e3:.1f} ms  [J part {tj*1e3:.1f}, K part {tk*1e3:.1f}]')
print(f'1-dm kernel, alpha: {t1a*1e3:.1f} ms; beta: {t1b*1e3:.1f} ms; sum {1e3*(t1a+t1b):.1f} ms')
print(f'J(Da+Db)+K(Da), then K(Db) with the 1-dm kernel: {tjs*1e3:.1f} ms')
ref_j = vj2[0] + vj2[1]
print('max|dJ| (sum)  %.2e  rel %.2e' % (abs(vjs - ref_j).max(), abs(vjs - ref_j).max()/abs(ref_j).max()))
print('max|dK|        %.2e  rel %.2e' % (abs(vks - vk2).max(), abs(vks - vk2).max()/abs(vk2).max()))
print('max|dK| 1dm    %.2e' % abs(numpy.asarray([vka[0], vkb[0]]) - vk2).max())

# components of the 1-dm kernel
tj1, _ = best(lambda: _vhf._incore_s8_batch(eri, dms[:1], 1, True, False))
tk1, _ = best(lambda: _vhf._incore_s8_batch(eri, dms[:1], 1, False, True))
print(f'1-dm kernel J only {tj1*1e3:.1f} ms, K only {tk1*1e3:.1f} ms')
# memory streaming reference: sum of the ERI array
ts, _ = best(lambda: eri.sum())
print(f'eri.sum() (one read pass) {ts*1e3:.1f} ms')
