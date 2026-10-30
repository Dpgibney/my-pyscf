"""The TD operators (TDA, TDDFT/RPA, Casida; singlet and triplet) with the XC response from the factors
of the trial dms (vind_lowrank of the response function) against the same operators with the XC
response from the dms (FUSED_XC=False), on random trial vectors; both paths of vind_lowrank (the flop
model forced each way). Also the CPHF operator of hessian.rhf.gen_vind (vind_mo), and the time of one
call per path on 20 vectors.
Usage: td_lowrank_check.py [system ...]   systems: water benzene caffeine
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import BENZENE, CAFFEINE, WATER
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.tdscf import rks as tdrks
from pyscf.hessian import rhf as rhf_hess

SYSTEMS = {'water': (WATER, 'def2-svp'), 'benzene': (BENZENE, 'def2-svp'),
           'caffeine': (CAFFEINE, 'def2-svp')}


def run(name):
    atom, basis = SYSTEMS[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=16000)
    worst = 0
    for xc in ('LDA,VWN', 'PBE', 'B3LYP', 'R2SCAN', 'CAMB3LYP'):
        mf = dft.RKS(mol, xc=xc).run(conv_tol=1e-8)
        nocc = int((mf.mo_occ > 0).sum())
        nvir = mf.mo_coeff.shape[1] - nocc
        rng = np.random.default_rng(7)
        z = rng.standard_normal((6, nocc * nvir))
        xy = rng.standard_normal((6, 2 * nocc * nvir))
        methods = [('TDA', z), ('TDDFT', xy)]
        if not mf._numint.libxc.is_hybrid_xc(xc):
            methods.append(('CasidaTDDFT', z))
        for method, vecs in methods:
            for singlet in (True, False):
                def op():
                    td = getattr(tdrks, method)(mf) if method == 'CasidaTDDFT' else getattr(mf, method)()
                    td.singlet = singlet
                    return td.gen_vind()[0]
                with lib.temporary_env(numint, FUSED_XC=False):
                    ref = op()(vecs)
                for cheaper in (True, False):
                    with lib.temporary_env(numint._FusedXC, lowrank_cheaper=lambda *args: cheaper):
                        v = op()(vecs)
                    d = abs(v - ref).max() / abs(ref).max()
                    worst = max(worst, d)
                    print(f'{name} {xc:8s} {method:11s} {"singlet" if singlet else "triplet"} '
                          f'{"factored" if cheaper else "dm path "}: max|diff|/max|ref| {d:.1e}',
                          flush=True)
        mo, occ = mf.mo_coeff, mf.mo_occ
        mo1 = rng.standard_normal((6, mo.shape[1], nocc))
        with lib.temporary_env(numint, FUSED_XC=False):
            ref = rhf_hess.gen_vind(mf, mo, occ)(mo1)
        for cheaper in (True, False):
            with lib.temporary_env(numint._FusedXC, lowrank_cheaper=lambda *args: cheaper):
                v = rhf_hess.gen_vind(mf, mo, occ)(mo1)
            d = abs(v - ref).max() / abs(ref).max()
            worst = max(worst, d)
            print(f'{name} {xc:8s} CPHF vind_mo {"factored" if cheaper else "dm path "}: '
                  f'max|diff|/max|ref| {d:.1e}', flush=True)
        if name != 'water':
            z20 = rng.standard_normal((20, nocc * nvir))
            times = {}
            for cheaper in (True, False):
                with lib.temporary_env(numint._FusedXC, lowrank_cheaper=lambda *args: cheaper):
                    vind = mf.TDA().gen_vind()[0]
                    vind(z20)
                    t = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        vind(z20)
                        t.append(time.perf_counter() - t0)
                    times[cheaper] = np.median(t)
            print(f'{name} {xc:8s} TDA vind(20 vectors): dm path {times[False]:.3f} s, '
                  f'factored {times[True]:.3f} s ({times[False] / times[True]:.2f}x)', flush=True)
    print(f'{name}: worst max|diff|/max|ref| {worst:.1e}')


if __name__ == '__main__':
    print(f'threads={lib.num_threads()}')
    for name in sys.argv[1:] or ('water', 'benzene'):
        run(name)
