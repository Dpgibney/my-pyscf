"""The XC response of CPHF-type density matrices dm1 = C x Cocc^T + h.c. from the factors
(numint._FusedXC.rks_fxc_mo: VXCfused_eval_rho_lowrank, VXCfused_dot_ao_mo) against
C^T nr_rks_fxc(dm1) Cocc, with times, the flop model's ratio and choice (lowrank_cheaper), and
the full response vind.vind_mo against the projected vind where the response function offers it.
Times are medians of 3 interleaved runs. The projections use lib.einsum (BLAS): numpy.einsum
contracts three operands in one naive loop, which on caffeine/SVP alone costs ~30 s per call.
Usage: lowrank_fxc_check.py [system ...]   systems: benzene caffeine caffeine-tzvp c20h42
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


def interleaved(fns, n=3):
    '''medians of n interleaved timings of each function, and the results of their warm-up runs'''
    res = [f() for f in fns]
    t = [[] for f in fns]
    for _ in range(n):
        for i, f in enumerate(fns):
            t0 = time.perf_counter()
            f()
            t[i].append(time.perf_counter() - t0)
    return [np.median(x) for x in t], res


def run(name):
    atom, basis = SYSTEMS[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=20000)
    for xc in ('LDA,VWN', 'B3LYP', 'R2SCAN'):
        mf = dft.RKS(mol, xc=xc)
        mf.conv_tol = 1e-8
        mf.kernel()
        mo, occ = mf.mo_coeff, mf.mo_occ
        nmo = mo.shape[1]
        mocc = mo[:, occ > 0]
        nocc = mocc.shape[1]
        ni, grids = mf._numint, mf.grids
        xctype = ni._xc_type(xc)
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo, occ, 0)
        x = np.random.default_rng(1).standard_normal((3 * mol.natm, nmo, nocc)) * 1e-2
        dm1 = lib.einsum('pa,kai,qi->kpq', mo, x, mocc)
        dm1 = dm1 + dm1.transpose(0, 2, 1)
        fused = numint._fused_xc(ni, mol, grids, nmo, xctype, 16000, (mo,),
                                 match_block_loop=False)

        def ref():
            v = ni.nr_rks_fxc(mol, grids, xc, None, dm1, 0, 1, rho0, vxc, fxc, max_memory=16000)
            return lib.einsum('pa,kpq,qi->kai', mo, v, mocc)

        def lowrank():
            out = np.zeros((len(x), nmo, nocc))
            a = lib.einsum('pa,kai->kpi', mo, x)
            return fused.rks_fxc_mo(ni, xctype, fxc, a, mocc, mo, mocc, out)

        (t_ref, t_lr), (r_ref, r_lr) = interleaved((ref, lowrank))
        d = abs(r_lr - r_ref).max() / abs(r_ref).max()
        nset = len(x)
        cheaper = fused.lowrank_cheaper(xctype, nocc, nocc, nset)
        # the flop ratio factored/dm path of lowrank_cheaper's model (one pass)
        ns = fused.data.group_naos(mol)
        c_dm, c_rho, c_v, c_pass = {'LDA': (3, 2, 2, 2), 'GGA': (4, 4, 4, 8),
                                    'MGGA': (13, 8, 10, 8)}[xctype]
        ratio = ((nset * (c_rho + c_v) + 2 * c_pass) * nocc * ns.sum()
                 / (nset * c_dm * (ns**2).sum()))
        line = (f'{name} {xc:8s} nset={len(x)}: XC dm path {t_ref:6.2f} s, factored {t_lr:6.2f} s '
                f'({t_ref / t_lr:4.2f}x), model flops factored/dm {ratio:4.2f} '
                f'-> {"factored" if cheaper else "dm path"}, max|diff|/max|ref| {d:.1e}')
        vresp = mf.gen_response(mo, occ, hermi=1)
        if getattr(vresp, 'vind_mo', None) is not None:
            full = lib.einsum('pa,kpq,qi->kai', mo, vresp(dm1), mocc)
            d = abs(vresp.vind_mo(dm1, x) - full).max() / abs(full).max()
            line += f'; vind_mo vs vind {d:.1e}'
        print(line, flush=True)


if __name__ == '__main__':
    print(f'threads={lib.num_threads()}')
    for name in sys.argv[1:] or ('benzene', 'caffeine', 'caffeine-tzvp'):
        run(name)
