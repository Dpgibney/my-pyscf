"""VXCfused_eval_rho_lowrank and VXCfused_dot_ao_mo on fixed inputs (benzene/def2-SVP; SVWN, B3LYP, r2SCAN),
`save` in one tree and `cmp` in another, bit for bit (the It17 refactor into shared per-group helpers); in cmp
mode also rks_fxc_mo in one kernel against two kernels and against nr_rks_fxc. Uses the orbitals that
td_cphf_bitwise.py saved in WORKDIR.
Usage: lowrank_kernels_bitwise.py save|cmp WORKDIR
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import BENZENE, CAFFEINE
from pyscf import gto, dft, lib
from pyscf.dft import numint
mode, work = sys.argv[1], sys.argv[2]
out = os.path.join(work, 'lowrank_kernels.npz')
res = {}
mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
for xc in ('LDA,VWN', 'B3LYP', 'R2SCAN'):
    mf = dft.RKS(mol, xc=xc)
    s = np.load(os.path.join(work, f'scf_{xc}.npz'))
    mf.mo_coeff, mf.mo_occ = s['mo_coeff'], s['mo_occ']
    mf.grids.build(with_non0tab=True)
    ni, grids = mf._numint, mf.grids
    xctype = ni._xc_type(xc)
    mo, occ = mf.mo_coeff, mf.mo_occ
    orbo, orbv = mo[:, occ > 0], mo[:, occ == 0]
    nocc, nvir = orbo.shape[1], orbv.shape[1]
    fused = numint._fused_xc(ni, mol, grids, mol.nao, xctype, 4000, (), match_block_loop=False)
    rng = np.random.default_rng(11)
    z = rng.standard_normal((6, nocc, nvir))
    a = np.asarray(lib.einsum('xov,pv->xpo', z, orbv), order='C')
    ng = grids.weights.size
    nrho = numint._FUSED_NRHO[xctype]
    ip0, ip1 = 112 * 7, 112 * 7 + 112 * 40
    res[f'{xc}_rho'] = fused.rho_lowrank(ip0, ip1, a, orbo, xctype)
    wv = rng.standard_normal((6, nrho, ip1 - ip0)) * 1e-3
    vq = np.zeros((6, mol.nao, nocc))
    fused.dot_ao_mo(ip0, ip1, wv, xctype, orbo, vq)
    res[f'{xc}_dot'] = vq
    if mode == 'cmp':
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo, occ, 1)
        fxc = .5 * (fxc[0,:,0] + fxc[0,:,1])
        o1 = fused.rks_fxc_mo(ni, xctype, fxc, a, orbo, orbv, orbo, np.zeros((6, nvir, nocc)), one_pass=True)
        o2 = fused.rks_fxc_mo(ni, xctype, fxc, a, orbo, orbv, orbo, np.zeros((6, nvir, nocc)), one_pass=False)
        dm1 = lib.einsum('xov,pv,qo->xpq', z, orbv, orbo*2)
        v = ni.nr_rks_fxc(mol, grids, xc, None, dm1, 0, 0, rho0, vxc, fxc, max_memory=4000)
        ref = lib.einsum('xpq,pv,qo->xvo', v, orbv, orbo)
        print(f'{xc}: one-pass vs two-kernel {abs(o1-o2).max()/abs(o2).max():.1e}, '
              f'one-pass vs nr_rks_fxc {abs(o1-ref).max()/abs(ref).max():.1e}, '
              f'two-kernel vs nr_rks_fxc {abs(o2-ref).max()/abs(ref).max():.1e}')
if mode == 'save':
    np.savez(out, **res)
else:
    ref = np.load(out)
    for k, v in res.items():
        print(k, 'bitwise' if np.array_equal(v, ref[k]) else f'DIFFERS {abs(v-ref[k]).max():.1e}')
