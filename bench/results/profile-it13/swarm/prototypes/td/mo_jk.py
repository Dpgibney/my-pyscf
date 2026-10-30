"""TD J/K of trial vectors from in-memory s8 ERIs: AO path (mf.get_jk, hermi=0, as tdscf) vs MO
integrals (ia|jb), (ij|ab) built once with ao2mo.incore.general and contracted per call by GEMM.
Single thread; core-Hamiltonian orbitals (no SCF). Usage: mo_jk.py SYSTEM NSET"""
import sys
import time
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
import scipy.linalg
from molecules import BENZENE, CAFFEINE
from pyscf import gto, dft, lib, ao2mo

system, nset = sys.argv[1], int(sys.argv[2])
mol = gto.M(atom={'benzene': BENZENE, 'caffeine': CAFFEINE}[system], basis='def2-svp', verbose=0,
            max_memory=4000)
mf = dft.RKS(mol, xc='B3LYP')
h, s = mf.get_hcore(), mf.get_ovlp()
e, mo = scipy.linalg.eigh(h, s)
nocc = mol.nelectron // 2
orbo, orbv = mo[:, :nocc], mo[:, nocc:]
nvir = orbv.shape[1]
t0 = time.perf_counter()
mf._eri = mol.intor('int2e', aosym='s8')
print(f'{system} nao {mol.nao} nocc {nocc} nvir {nvir}; ERIs {time.perf_counter()-t0:.1f} s', flush=True)
zs = np.random.default_rng(3).standard_normal((nset, nocc, nvir))

t0 = time.perf_counter()
dms = lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo * 2)
vj, vk = mf.get_jk(mol, dms, hermi=0)
v = vj - .5 * vk
ref = lib.einsum('xpq,qo,pv->xov', v, orbo, orbv)
t_ao = time.perf_counter() - t0

t0 = time.perf_counter()
iajb = ao2mo.incore.general(mf._eri, (orbo, orbv, orbo, orbv), compact=False)
t1 = time.perf_counter()
ijab = ao2mo.incore.general(mf._eri, (orbo, orbo, orbv, orbv), compact=False)
t2 = time.perf_counter()
iajb = iajb.reshape(nocc * nvir, nocc * nvir)
ijab = ijab.reshape(nocc, nocc, nvir, nvir)
t3 = time.perf_counter()
# J: 2 (ia|jb) z_jb ; K: (ij|ab) z_jb -> v_ia = sum_jb (ij|ab) z_jb
vmo = 2 * lib.dot(zs.reshape(nset, -1), iajb.T).reshape(nset, nocc, nvir)
kmo = 2 * lib.einsum('ijab,xjb->xia', ijab, zs)
new = vmo - .5 * kmo
t4 = time.perf_counter()
d = abs(new - ref).max() / abs(ref).max()
print(f'  AO J/K of {nset} dms + projection {t_ao:.2f} s; MO: (ia|jb) {t1-t0:.2f} s, (ij|ab) {t2-t1:.2f} s, '
      f'contraction {t4-t3:.3f} s; max|diff|/max|ref| {d:.1e}', flush=True)
