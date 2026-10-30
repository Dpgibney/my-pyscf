"""Time of _FusedXC.dot (VXCfused_dot_wv) per call on the SCF's blocks (block_loop sizes, as nr_rks uses) and
on the whole grid, for nset = 1, 2 and 20 random potentials; medians of 5 calls after a warm-up.
Usage: dot_wv_time.py SYSTEM XC     systems: benzene caffeine caffeine-tzvp c20h42
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint

SYSTEMS = {'benzene': (BENZENE, 'def2-svp'), 'caffeine': (CAFFEINE, 'def2-svp'),
           'caffeine-tzvp': (CAFFEINE, 'def2-tzvp'), 'c20h42': (n_alkane(20), 'def2-svp')}
name, xc = sys.argv[1], sys.argv[2]
atom, basis = SYSTEMS[name]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = dft.RKS(mol, xc=xc)
mf.grids.build(with_non0tab=True)
ni, grids = mf._numint, mf.grids
xctype = ni._xc_type(xc)
nao = mol.nao
dm = np.eye(nao)
fused = numint._fused_xc(ni, mol, grids, nao, xctype, 16000, (dm,))
ngrids = grids.weights.size
nrho = numint._FUSED_NRHO[xctype]


def med(f, n=5):
    f()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return np.median(t)


print(f'{name} {xc}: nao={nao} ngrids={ngrids} blocks={len(fused.blocks)} threads={lib.num_threads()}')
for nset in (1, 2, 20):
    wv = np.random.default_rng(2).standard_normal((nset, nrho, ngrids)) * 1e-3
    vmat = np.zeros((nset, nao, nao))
    vtau = np.zeros_like(vmat) if xctype == 'MGGA' else None

    def blocks():
        for ip0, ip1 in fused.blocks:
            fused.dot(ip0, ip1, wv[:, :, ip0:ip1], xctype, 1, vmat, vtau)

    def whole():
        fused.dot(0, ngrids, wv, xctype, 1, vmat, vtau)
    tb, tw = med(blocks), med(whole)
    print(f'  nset={nset:3d}: SCF blocks {tb * 1e3:8.2f} ms, whole grid {tw * 1e3:8.2f} ms', flush=True)
