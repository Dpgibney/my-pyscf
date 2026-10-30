"""Prototype (outside the tree) of the factored XC response for TDA/RPA singlets and triplets:
A[k] = C_v Z_k^T, B = C_o, P = C_v, Q = C_o through the existing _FusedXC.rks_fxc_mo, against
nr_rks_fxc_st on the AO dms that tdscf builds. benzene: converged SCF and the whole TDA vind;
caffeine: core-Hamiltonian orbitals, XC part only (no SCF at 1 thread).
Usage: tda_factored.py SYSTEM NSET XC [XC ...]"""
import sys
import time
from collections import defaultdict
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
import scipy.linalg
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint

system, nset = sys.argv[1], int(sys.argv[2])
xcs = sys.argv[3:]
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'c20': n_alkane(20)}[system]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
print(f'{system} nao {mol.nao} threads {lib.num_threads()} nset {nset}', flush=True)

T = defaultdict(float)
def timed(cls, name):
    f = getattr(cls, name)
    def g(*a, **k):
        t0 = time.perf_counter()
        r = f(*a, **k)
        T[name] += time.perf_counter() - t0
        return r
    setattr(cls, name, g)
for n in ('_rho_kernel', 'dot', 'rho_lowrank', 'dot_ao_mo'):
    timed(numint._FusedXC, n)

for xc in xcs:
    mf = dft.RKS(mol, xc=xc)
    if system == 'benzene':
        mf.kernel()
        mo, occ, e = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    else:
        h, s = mf.get_hcore(), mf.get_ovlp()
        e, mo = scipy.linalg.eigh(h, s)
        occ = np.zeros(mol.nao); occ[:mol.nelectron // 2] = 2
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = mo, occ, e
        mf.grids.build(with_non0tab=True)
    ni, grids = mf._numint, mf.grids
    xctype = ni._xc_type(xc)
    orbo, orbv = mo[:, occ > 0], mo[:, occ == 0]
    nocc, nvir = orbo.shape[1], orbv.shape[1]
    zs = np.random.default_rng(7).standard_normal((nset, nocc, nvir)) * .1
    max_memory = max(2000, mf.max_memory * .8 - lib.current_memory()[0])
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo, occ, 1)
    fxc = fxc * .5   # as _gen_rhf_response for singlet/triplet
    fused = numint._fused_xc(ni, mol, grids, mol.nao, xctype, max_memory, (mo,),
                             match_block_loop=False)
    for singlet in (True, False):
        dms = lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo * 2)   # as _gen_tda_operation
        T.clear()
        t0 = time.perf_counter()
        v = ni.nr_rks_fxc_st(mol, grids, xc, None, dms, 0, singlet, rho0, vxc, fxc,
                             max_memory=max_memory)
        ref = lib.einsum('xpq,qo,pv->xov', v, orbo, orbv)
        t_ref = time.perf_counter() - t0
        tref = dict(T)
        T.clear()
        t0 = time.perf_counter()
        f = fxc[0, :, 0] + fxc[0, :, 1] if singlet else fxc[0, :, 0] - fxc[0, :, 1]
        a = lib.einsum('pv,kov->kpo', orbv, zs)
        out = np.zeros((nset, nvir, nocc))
        fused.rks_fxc_mo(ni, xctype, f, a, orbo, orbv, orbo, out)
        new = out.transpose(0, 2, 1)
        t_new = time.perf_counter() - t0
        d = abs(new - ref).max() / abs(ref).max()
        cheaper = fused.lowrank_cheaper(xctype, nocc, nocc, nset)
        print(f'  {xc:8s} {"singlet" if singlet else "triplet"}: dm path {t_ref:6.2f} s '
              f'(rho {tref.get("_rho_kernel", 0):.2f} dot {tref.get("dot", 0):.2f}), factored '
              f'{t_new:6.2f} s (rho {T.get("rho_lowrank", 0):.2f} vQ {T.get("dot_ao_mo", 0):.2f}) '
              f'{t_ref / t_new:4.2f}x, model picks {"factored" if cheaper else "dm"}, '
              f'max|diff|/max|ref| {d:.1e}', flush=True)
    if system == 'benzene' and xc.upper() == 'B3LYP':
        # whole TDA vind (singlet) against tdscf's
        td = mf.TDA()
        vind, hdiag = td.gen_vind()
        t0 = time.perf_counter()
        vref = vind(zs.reshape(nset, -1))
        t_ref = time.perf_counter() - t0
        e_ia = e[occ == 0] - e[occ > 0, None]
        hyb = ni.hybrid_coeff(xc)
        f = fxc[0, :, 0] + fxc[0, :, 1]
        t0 = time.perf_counter()
        a = lib.einsum('pv,kov->kpo', orbv, zs)
        out = np.zeros((nset, nvir, nocc))
        fused.rks_fxc_mo(ni, xctype, f, a, orbo, orbv, orbo, out)
        dms = lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo * 2)
        vj, vk = mf.get_jk(mol, dms, hermi=0)
        vjk = vj - .5 * hyb * vk
        v1 = lib.einsum('xpq,qo,pv->xov', vjk, orbo, orbv) + out.transpose(0, 2, 1) + zs * e_ia
        t_new = time.perf_counter() - t0
        d = abs(v1.reshape(nset, -1) - vref).max() / abs(vref).max()
        print(f'  TDA vind B3LYP singlet: tdscf {t_ref:.2f} s, factored XC {t_new:.2f} s, '
              f'max|diff|/max|ref| {d:.1e}', flush=True)
        # RPA: XC of X+Y, same for both halves
        ys = np.random.default_rng(8).standard_normal((nset, nocc, nvir)) * .03
        tdh = mf.TDDFT()
        vind2, _ = tdh.gen_vind()
        vref2 = vind2(np.hstack([zs.reshape(nset, -1), ys.reshape(nset, -1)]))
        a = lib.einsum('pv,kov->kpo', orbv, zs + ys)
        out = np.zeros((nset, nvir, nocc))
        fused.rks_fxc_mo(ni, xctype, f, a, orbo, orbv, orbo, out)
        dms = (lib.einsum('xov,pv,qo->xpq', zs, orbv, orbo * 2) +
               lib.einsum('xov,qv,po->xpq', ys, orbv, orbo * 2))
        vj, vk = mf.get_jk(mol, dms, hermi=0)
        vjk = vj - .5 * hyb * vk
        top = lib.einsum('xpq,qo,pv->xov', vjk, orbo, orbv) + out.transpose(0, 2, 1) + zs * e_ia
        bot = lib.einsum('xpq,po,qv->xov', vjk, orbo, orbv) + out.transpose(0, 2, 1) + ys * e_ia
        new2 = np.hstack([top.reshape(nset, -1), -bot.reshape(nset, -1)])
        d = abs(new2 - vref2).max() / abs(vref2).max()
        print(f'  RPA vind B3LYP: factored XC vs tdscf max|diff|/max|ref| {d:.1e}', flush=True)
