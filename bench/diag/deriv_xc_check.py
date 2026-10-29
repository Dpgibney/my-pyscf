"""The XC terms of gradients and Hessians against saved references, and their times.
  save     run SCFs, store the orbitals and every XC term of the current tree
  compare  recompute the XC terms from the stored orbitals: max |diff| relative to max |ref|,
           and times (reference times from save)
Terms: grad.rks/uks.get_vxc, hessian.rks/uks._get_vxc_diag, _get_vxc_deriv2_dm, _get_vxc_deriv1.
Systems: caffeine/def2-SVP RKS (SVWN, B3LYP, r2SCAN), benzene+/def2-SVP UKS (SVWN, B3LYP, r2SCAN).
Usage: deriv_xc_check.py save|compare DIR
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import BENZENE, CAFFEINE
from pyscf import gto, dft, lib
from pyscf.grad import rks as rks_grad, uks as uks_grad
from pyscf.hessian import rks as rks_hess, uks as uks_hess

XCS = ('SVWN', 'B3LYP', 'R2SCAN')
CASES = [('caffeine', CAFFEINE, 0, 0), ('benzene+', BENZENE, 1, 1)]


def terms(mf, xc):
    mol, ni, grids = mf.mol, mf._numint, mf.grids
    mo, occ = mf.mo_coeff, mf.mo_occ
    dm = mf.make_rdm1()
    h = mf.Hessian()
    if mf.mo_coeff.ndim == 2:
        g, hs = rks_grad, rks_hess
    else:
        g, hs = uks_grad, uks_hess
    return {'grad get_vxc': lambda: g.get_vxc(ni, mol, grids, xc, dm, max_memory=20000)[1],
            'hess _get_vxc_diag': lambda: hs._get_vxc_diag(h, mo, occ, 20000),
            'hess _get_vxc_deriv2_dm': lambda: hs._get_vxc_deriv2_dm(h, mo, occ, 20000),
            'hess _get_vxc_deriv1': lambda: hs._get_vxc_deriv1(h, mo, occ, 20000)}


def setup(atom, charge, spin, xc, path, save):
    mol = gto.M(atom=atom, basis='def2-svp', charge=charge, spin=spin, verbose=0,
                max_memory=20000)
    mf = dft.RKS(mol, xc=xc) if spin == 0 else dft.UKS(mol, xc=xc)
    if save:
        mf.conv_tol = 1e-9
        mf.kernel()
        np.savez(path + '.mo.npz', mo=mf.mo_coeff, occ=mf.mo_occ)
    else:
        mf.grids.build(with_non0tab=True)
        d = np.load(path + '.mo.npz')
        mf.mo_coeff, mf.mo_occ = d['mo'], d['occ']
    return mf


def main(what, out):
    os.makedirs(out, exist_ok=True)
    save = what == 'save'
    print(f'threads={lib.num_threads()}')
    for name, atom, charge, spin in CASES:
        for xc in XCS:
            path = os.path.join(out, f'{name}_{xc}')
            mf = setup(atom, charge, spin, xc, path, save)
            print(f'{name} {xc}:', flush=True)
            for label, fn in terms(mf, xc).items():
                t0 = time.perf_counter()
                r = np.asarray(fn())
                t = time.perf_counter() - t0
                f = path + '.' + label.replace(' ', '_') + '.npz'
                if save:
                    np.savez(f, r=r, t=t)
                    print(f'  {label:24s} {t:7.3f} s', flush=True)
                else:
                    ref = np.load(f)
                    d = abs(r - ref['r']).max() / abs(ref['r']).max()
                    print(f'  {label:24s} {float(ref["t"]):7.3f} -> {t:7.3f} s '
                          f'({float(ref["t"]) / t:4.2f}x), max|diff|/max|ref| {d:.1e}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
