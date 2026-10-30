"""numint._fxc_wv (VXCfused_fxc_wv) against the per-set numpy expressions it replaced in _FusedXC.rks_fxc and
rks_fxc_mo, bit for bit, on real fxc (CPHF kernel spin=0, TD singlet kernel spin=1) and random rho1, for whole
grids and blocks at offsets; with times.
Usage: fxc_wv_check.py
"""
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import BENZENE, CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint
for atom, name in ((BENZENE, 'benzene'), (CAFFEINE, 'caffeine')):
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    for xc in ('LDA,VWN', 'PBE', 'R2SCAN'):
        mf = dft.RKS(mol, xc=xc).run(conv_tol=1e-6)
        ni, grids = mf._numint, mf.grids
        xctype = ni._xc_type(xc)
        mo, occ = mf.mo_coeff, mf.mo_occ
        for spin in (0, 1):
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo, occ, spin)
            if spin == 1:
                fxc = fxc[0,:,0] + fxc[0,:,1]
            fxc = np.ascontiguousarray(fxc)
            ng = grids.weights.size
            nrho = numint._FUSED_NRHO[xctype]
            rng = np.random.default_rng(3)
            for nset, (ip0, ip1) in ((7, (0, ng)), (3, (ng//3, ng//3 + 4480)), (1, (112, 2240))):
                rho1 = rng.standard_normal((nset, nrho, ip1 - ip0))
                if xctype == 'LDA':
                    rho1 = rho1[:,0]
                weight = grids.weights[ip0:ip1]
                _fxc = fxc[:,:,ip0:ip1]
                ref = np.empty((nset, nrho, ip1 - ip0))
                t0 = time.perf_counter()
                for i in range(nset):
                    if xctype == 'LDA':
                        ref[i] = weight * rho1[i] * _fxc[0]
                    else:
                        ref[i] = np.einsum('yg,xyg,g->xg', rho1[i], _fxc, weight)
                t1 = time.perf_counter()
                wv = numint._fxc_wv(rho1, fxc, grids.weights, ip0, ip1)
                t2 = time.perf_counter()
                same = np.array_equal(wv, ref)
                print(f'{name} {xc:8s} spin={spin} nset={nset} [{ip0},{ip1}): '
                      f'{"bitwise" if same else "DIFFERS %.1e" % abs(wv-ref).max()}  '
                      f'numpy {t1-t0:.4f} s, kernel {t2-t1:.4f} s', flush=True)
