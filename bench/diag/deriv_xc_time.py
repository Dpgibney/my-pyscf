"""Time of the XC terms of RKS gradients and Hessians per call, and the share of eval_gto in it:
grad.rks.get_vxc, and hessian.rks._get_vxc_diag, _get_vxc_deriv2_dm and _get_vxc_deriv1.
Usage: deriv_xc_time.py [system ...]   systems: benzene caffeine c20h42 (def2-SVP)
"""
import cProfile
import os
import pstats
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.grad import rks as rks_grad
from pyscf.hessian import rks as rks_hess

SYSTEMS = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'c20h42': n_alkane(20)}


def timed(fn):
    fn()
    t0 = time.perf_counter()
    fn()
    t = time.perf_counter() - t0
    prof = cProfile.Profile()
    prof.runcall(fn)
    stats = pstats.Stats(prof)
    t_ao = sum(v[2] for k, v in stats.stats.items()
               if k[0].endswith('eval_gto.py') and k[2] == 'eval_gto')
    return t, t_ao


def run(name):
    mol = gto.M(atom=SYSTEMS[name], basis='def2-svp', verbose=0, max_memory=20000)
    for xc in ('B3LYP', 'r2SCAN'):
        mf = dft.RKS(mol, xc=xc)
        mf.conv_tol = 1e-8
        mf.kernel()
        ni, grids = mf._numint, mf.grids
        dm = mf.make_rdm1()
        hobj = mf.Hessian()
        mo, occ = mf.mo_coeff, mf.mo_occ
        calls = {
            'grad get_vxc': lambda: rks_grad.get_vxc(ni, mol, grids, xc, dm, max_memory=20000),
            'hess _get_vxc_diag': lambda: rks_hess._get_vxc_diag(hobj, mo, occ, 20000),
            'hess _get_vxc_deriv2_dm': lambda: rks_hess._get_vxc_deriv2_dm(hobj, mo, occ, 20000),
            'hess _get_vxc_deriv1': lambda: rks_hess._get_vxc_deriv1(hobj, mo, occ, 20000),
        }
        print(f'{name} {xc} (nao={mol.nao}):', flush=True)
        for label, fn in calls.items():
            t, t_ao = timed(fn)
            print(f'  {label:24s} {t:7.3f} s, eval_gto {t_ao:6.3f} s under cProfile', flush=True)


if __name__ == '__main__':
    print(f'threads={lib.num_threads()}')
    for name in sys.argv[1:] or ('benzene', 'caffeine', 'c20h42'):
        run(name)
