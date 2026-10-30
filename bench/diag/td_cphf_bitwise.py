"""The TD operators (TDA, TDDFT, Casida; singlet and triplet) on the dm path and the CPHF operator
(hessian.rhf.gen_vind, both paths) on fixed benzene/def2-SVP orbitals: `save` in one tree, `cmp` in another,
bit for bit. The first run writes the SCF orbitals to WORKDIR; run both at the same thread count.
Usage: td_cphf_bitwise.py save|cmp WORKDIR
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import BENZENE
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.tdscf import rks as tdrks
from pyscf.hessian import rhf as rhf_hess

mode, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
res = {}
for xc in ('LDA,VWN', 'B3LYP', 'R2SCAN', 'CAMB3LYP'):
    mf = dft.RKS(mol, xc=xc)
    scf_file = os.path.join(work, f'scf_{xc}.npz')
    if os.path.exists(scf_file):
        s = np.load(scf_file)
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = s['mo_coeff'], s['mo_occ'], s['mo_energy']
    else:
        mf.kernel()
        np.savez(scf_file, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy)
    mf.grids.build(with_non0tab=True)
    mo, occ = mf.mo_coeff, mf.mo_occ
    nocc = int((occ > 0).sum()); nov = nocc * (occ.size - nocc)
    rng = np.random.default_rng(9)
    methods = [tdrks.TDA, tdrks.TDDFT] + ([] if mf._numint.libxc.is_hybrid_xc(xc) else [tdrks.CasidaTDDFT])
    for method in methods:
        for singlet in (True, False):
            td = method(mf); td.singlet = singlet
            x = rng.standard_normal((8, 2 * nov if method is tdrks.TDDFT else nov))
            with lib.temporary_env(numint._FusedXC, lowrank_cheaper=lambda *args: False):
                res[f'{xc}_{method.__name__}_{singlet}'] = td.gen_vind()[0](x)
    mo1 = np.random.default_rng(5).standard_normal((36, mo.shape[1], nocc)) * 1e-2
    for n in (36, 4):
        res[f'{xc}_cphf_{n}'] = rhf_hess.gen_vind(mf, mo, occ)(mo1[:n])
ref_file = os.path.join(work, 'ref.npz')
if mode == 'save':
    np.savez(ref_file, **res)
else:
    ref = np.load(ref_file)
    for k, v in res.items():
        print(k, 'bitwise' if np.array_equal(v, ref[k]) else f'differs {abs(v-ref[k]).max():.1e}')
