"""Reproducibility of the kernels that sum partial matrices (VXCfused_dot_wv, VXCfused_dot_terms,
VXCfused_dot_ao_mo, VXCfused_fxc_lowrank, and the sparse _dot_ao_ao_gemm through FUSED_XC=False), which take
their groups by a dynamic schedule since It18: each call repeated NREP times must give the same bits. With
`save`, the results are written to OUT; with `cmp`, compared with those of another tree (max relative
difference). Usage: sched_determinism.py run|save|cmp [OUT] [NREP]
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad

mode = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
nrep = int(sys.argv[3]) if len(sys.argv) > 3 else 5
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0, max_memory=16000)
res = {}
for xc in ('PBE', 'R2SCAN'):
    mf = dft.RKS(mol, xc=xc)
    mf.conv_tol = 1e-6
    if mode == 'cmp':
        # the orbitals of the saved run: the virtual ones of separate SCFs differ by rotations
        ref = np.load(out)
        mf.mo_coeff, mf.mo_occ = ref[f'{xc}_mo_coeff'], ref[f'{xc}_mo_occ']
        mf.grids.build(with_non0tab=True)
    else:
        mf.kernel()
    ni, grids = mf._numint, mf.grids
    xctype = ni._xc_type(xc)
    dm = mf.make_rdm1()
    mo, occ = mf.mo_coeff, mf.mo_occ
    orbo, orbv = mo[:, occ > 0], mo[:, occ == 0]
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo, occ, 0)
    z = np.random.default_rng(1).standard_normal((12, orbo.shape[1], orbv.shape[1]))
    a = lib.einsum('xov,pv->xpo', z, orbv)
    dm1 = lib.einsum('xov,pv,qo->xpq', z, orbv, orbo)
    dm1 = dm1 + dm1.transpose(0, 2, 1)
    fused = numint._fused_xc_mo(ni, mol, grids, xctype, 16000)
    calls = {
        'nr_rks': lambda: ni.nr_rks(mol, grids, xc, dm)[2],
        'nr_rks_unfused': lambda: _unfused(lambda: ni.nr_rks(mol, grids, xc, dm)[2]),
        'nr_rks_fxc': lambda: ni.nr_rks_fxc(mol, grids, xc, None, dm1, 0, 1, rho0, vxc, fxc,
                                            max_memory=16000),
        'fxc_one_kernel': lambda: fused.rks_fxc_mo(ni, xctype, fxc, a, orbo, orbv, orbo,
                                                   np.zeros((12, orbv.shape[1], orbo.shape[1]))),
        'fxc_two_kernels': lambda: fused.rks_fxc_mo(ni, xctype, fxc, a, orbo, orbv, orbo,
                                                    np.zeros((12, orbv.shape[1], orbo.shape[1])),
                                                    one_pass=False),
        'grad_vxc': lambda: rks_grad.get_vxc(ni, mol, grids, xc, dm)[1],
    }

    def _unfused(f):
        with lib.temporary_env(numint, FUSED_XC=False):
            return f()

    res[f'{xc}_mo_coeff'], res[f'{xc}_mo_occ'] = mo, occ
    for name, f in calls.items():
        first = f()
        same = all(np.array_equal(first, f()) for _ in range(nrep - 1))
        res[f'{xc}_{name}'] = first
        if mode != 'cmp':
            print(f'{xc:7s} {name:16s} {nrep} calls: {"bit for bit" if same else "DIFFER"}', flush=True)
    if mode != 'cmp':
        d = abs(res[f'{xc}_nr_rks'] - res[f'{xc}_nr_rks_unfused']).max()
        print(f'{xc:7s} fused vs unfused nr_rks: {"bit for bit" if d == 0 else f"{d:.1e}"}', flush=True)
if mode == 'save':
    np.savez(out, **res)
elif mode == 'cmp':
    ref = np.load(out)
    for k, v in res.items():
        if not k.endswith(('_mo_coeff', '_mo_occ')):
            print(f'{k:24s} max|diff|/max|ref| {abs(v - ref[k]).max() / abs(ref[k]).max():.1e}')
