"""The TD XC response (dm path rks_fxc, factored rks_fxc_mo) with nr_numint_fused.c built as in the
tree (-msse3 -O2) and with -mavx2 -mfma -ftree-vectorize, both standalone copies loaded through a
proxy for numint.libdft. Single thread. Usage: simd_fxc.py SYSTEM NSET XC"""
import ctypes
import sys
import time
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
import scipy.linalg
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint

B = '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/td/build/'
orig = numint.libdft


class Proxy:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)

    def __getattr__(self, name):
        if name.startswith('VXCfused'):
            return getattr(self.lib, name)
        return getattr(orig, name)


libs = {'tree': orig, 'stock': Proxy(B + 'libfused_sse3dyn.so'), 'avx2': Proxy(B + 'libfused_avx2.so')}
system, nset, xc = sys.argv[1], int(sys.argv[2]), sys.argv[3]
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'c20': n_alkane(20)}[system]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc=xc)
h, s = mf.get_hcore(), mf.get_ovlp()
e, mo = scipy.linalg.eigh(h, s)
occ = np.zeros(mol.nao)
occ[:mol.nelectron // 2] = 2
mf.grids.build(with_non0tab=True)
ni, grids = mf._numint, mf.grids
xctype = ni._xc_type(xc)
orbo, orbv = mo[:, occ > 0], mo[:, occ == 0]
nocc, nvir = orbo.shape[1], orbv.shape[1]
zs = np.random.default_rng(7).standard_normal((nset, nocc, nvir)) * .1
max_memory = 3000
rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo, occ, 1)
fxc = fxc * .5
f = fxc[0, :, 0] + fxc[0, :, 1]
fused = numint._fused_xc(ni, mol, grids, mol.nao, xctype, max_memory, (mo,), match_block_loop=False)
dms = lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo * 2)
a = lib.einsum('pv,kov->kpo', orbv, zs)
print(f'{system} {xc} nset {nset} threads {lib.num_threads()}', flush=True)


def dm_path():
    v = ni.nr_rks_fxc_st(mol, grids, xc, None, dms, 0, True, rho0, vxc, fxc, max_memory=max_memory)
    return lib.einsum('xpq,qo,pv->xvo', v, orbo, orbv)


def factored():
    out = np.zeros((nset, nvir, nocc))
    return fused.rks_fxc_mo(ni, xctype, f, a, orbo, orbv, orbo, out)


res = {}
for name, fn in (('dm path', dm_path), ('factored', factored)):
    for lbl in ('tree', 'stock', 'avx2', 'tree', 'stock', 'avx2'):
        numint.libdft = libs[lbl]
        t0 = time.perf_counter()
        r = fn()
        t = time.perf_counter() - t0
        res.setdefault((name, lbl), []).append((t, r))
    numint.libdft = orig
    ref = res[(name, 'tree')][0][1]
    for lbl in ('tree', 'stock', 'avx2'):
        ts = [x[0] for x in res[(name, lbl)]]
        d = abs(res[(name, lbl)][1][1] - ref).max() / abs(ref).max()
        print(f'  {name:8s} {lbl:5s}: {min(ts):6.2f} s (runs {", ".join(f"{t:.2f}" for t in ts)}), '
              f'max|diff|/max|ref| vs tree {d:.1e}', flush=True)
