"""The UKS factored XC response where a spin has no electrons (H2+, H3 quartet; SVWN, PBE, B3LYP, r2SCAN): TDA,
TDDFT and CPHF operators with the factored path forced against FUSED_XC=False; and which response functions
ROKS gets (vind_lowrank, vind_mo) with ROHF's orbitals and with orbitals per spin.
Usage: uks_lowrank_edge.py   (run under bench/diag/run_in.py)
"""
import numpy as np
from pyscf import gto, dft, lib, tdscf
from pyscf.dft import numint
from pyscf.hessian import uhf as uhf_hess
for atom, charge, spin in (('H 0 0 0; H 0 0 0.74', 1, 1), ('H 0 0 0; H 0 0 0.9; H 0 0.9 0', 0, 3)):
    mol = gto.M(atom=atom, basis='def2-svp', charge=charge, spin=spin, verbose=0)
    for xc in ('LDA,VWN', 'PBE', 'B3LYP', 'r2SCAN'):
        mf = dft.UKS(mol, xc=xc).run(conv_tol=1e-10)
        for method in (tdscf.uks.TDA, tdscf.uks.TDDFT):
            td = method(mf)
            nov = sum((o > 0).sum() * (o == 0).sum() for o in mf.mo_occ)
            x = np.random.default_rng(1).random((3, nov * (2 if method is tdscf.uks.TDDFT else 1))) - .5
            with lib.temporary_env(numint, FUSED_XC=False):
                ref = td.gen_vind()[0](x)
            with lib.temporary_env(numint._FusedXC, lowrank_cheaper=lambda *a, **k: True):
                v = td.gen_vind()[0](x)
            print(f'{atom[:20]:20s} {xc:8s} {method.__name__:6s} max rel diff {abs(v - ref).max() / abs(ref).max():.1e}')
        mo, occ = mf.mo_coeff, mf.mo_occ
        nov = sum(c.shape[1] * (o > 0).sum() for c, o in zip(mo, occ))
        mo1 = np.random.default_rng(2).random((2, nov)) - .5
        with lib.temporary_env(numint, FUSED_XC=False):
            ref = uhf_hess.gen_vind(mf, mo, occ)(mo1)
        with lib.temporary_env(numint._FusedXC, lowrank_cheaper=lambda *a, **k: True):
            v = uhf_hess.gen_vind(mf, mo, occ)(mo1)
        print(f'{atom[:20]:20s} {xc:8s} CPHF   max rel diff {abs(v - ref).max() / abs(ref).max():.1e}')
# ROKS: gen_response without orbitals takes ROHF's mo_coeff/mo_occ (no vind_mo), and with orbital pairs
mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='def2-svp', charge=1, spin=1, verbose=0)
mf = dft.ROKS(mol, xc='PBE').run()
for args in ((), ((mf.mo_coeff,)*2, ((mf.mo_occ > 0) * 1., (mf.mo_occ == 2) * 1.))):
    vresp = mf.gen_response(*args, hermi=1)
    print('ROKS gen_response', 'pairs' if args else 'default', 'vind_lowrank' in dir(vresp), 'vind_mo' in dir(vresp))
