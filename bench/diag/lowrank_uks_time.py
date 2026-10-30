"""Time of the UKS XC response of TDA-type spin dms per call: the dm path (nr_uks_fxc on 2*nset dms) and the
factored path (_FusedXC.uks_fxc_mo, VXCfused_fxc_lowrank_uks), for 1-40 sets; with the flop model's ratio
(factored/dm, lowrank_cheaper's terms) and choice, and the largest difference between the paths. Medians
of 3 interleaved runs.
Usage: lowrank_uks_time.py SYSTEM XC [XC ...]     systems: benzene+ caffeine+ c20h42+
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint

SYSTEMS = {'benzene+': (BENZENE, 'def2-svp'), 'caffeine+': (CAFFEINE, 'def2-svp'),
           'c20h42+': (n_alkane(20), 'def2-svp')}
MAX_MEMORY = 16000


def medians(fns, n=3):
    for f in fns:
        f()
    t = [[] for f in fns]
    for _ in range(n):
        for i, f in enumerate(fns):
            t0 = time.perf_counter()
            f()
            t[i].append(time.perf_counter() - t0)
    return [np.median(x) for x in t]


def run(name, xc):
    atom, basis = SYSTEMS[name]
    mol = gto.M(atom=atom, basis=basis, charge=1, spin=1, verbose=0, max_memory=20000)
    mf = dft.UKS(mol, xc=xc).run(conv_tol=1e-8)
    ni, grids = mf._numint, mf.grids
    xctype = ni._xc_type(xc)
    mo, occ = mf.mo_coeff, mf.mo_occ
    orbo = [c[:, o > 0] for c, o in zip(mo, occ)]
    orbv = [c[:, o == 0] for c, o in zip(mo, occ)]
    nocc = [c.shape[1] for c in orbo]
    nvir = [c.shape[1] for c in orbv]
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo, occ, 1)
    fused = numint._fused_xc_mo(ni, mol, grids, xctype, MAX_MEMORY, spin=1)
    ns = fused.data.group_naos(mol)
    c_dm, c_rho, c_v, c_pass = {'LDA': (3, 2, 2, 2), 'GGA': (4, 4, 4, 8),
                                'MGGA': (13, 8, 10, 8)}[xctype]
    print(f'{name} {xc}: nao={mol.nao} nocc={nocc} ngrids={grids.weights.size} '
          f'threads={lib.num_threads()}', flush=True)
    for nset in (1, 2, 5, 10, 20, 40):
        rng = np.random.default_rng(1)
        z = [rng.standard_normal((nset, nocc[s], nvir[s])) for s in range(2)]
        a = [lib.einsum('xov,pv->xpo', z[s], orbv[s] * .5) for s in range(2)]
        dm1 = np.asarray([lib.einsum('xov,pv,qo->xpq', z[s], orbv[s], orbo[s]) for s in range(2)])

        def dm_path():
            v = ni.nr_uks_fxc(mol, grids, xc, None, dm1, 0, 0, rho0, vxc, fxc, max_memory=MAX_MEMORY)
            return [lib.einsum('xpq,pv,qo->xvo', v[s], orbv[s], orbo[s]) for s in range(2)]

        def factored():
            out = [np.zeros((nset, nvir[s], nocc[s])) for s in range(2)]
            return fused.uks_fxc_mo(ni, xctype, fxc, a, orbo, orbv, orbo, out)

        t_dm, t_lr = medians((dm_path, factored))
        ref, v = dm_path(), factored()
        err = max(abs(v[s] - ref[s]).max() for s in range(2)) / max(abs(r).max() for r in ref)
        ratio = ((nset * (c_rho + c_v) + c_pass) * sum(nocc) * ns.sum()
                 / (2 * nset * c_dm * (ns**2).sum()))
        cheaper = fused.lowrank_cheaper(xctype, sum(nocc), sum(nocc), nset, nspin=2)
        print(f'  nset={nset:3d}: dm path {t_dm:7.3f} s, factored {t_lr:7.3f} s; dm/factored '
              f'{t_dm / t_lr:5.2f}; model flops factored/dm {ratio:5.2f} -> '
              f'{"factored" if cheaper else "dm path"}; max rel diff {err:.1e}', flush=True)


if __name__ == '__main__':
    name = sys.argv[1]
    for xc in sys.argv[2:] or ('B3LYP',):
        run(name, xc)
