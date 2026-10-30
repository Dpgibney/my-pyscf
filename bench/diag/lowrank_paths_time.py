"""Time of the XC response of TDA-type dms per call: the dm path (nr_rks_fxc), the factored path in two
kernels (rho_lowrank, _fxc_wv, dot_ao_mo on blocks of points) and in one kernel (VXCfused_fxc_lowrank),
for 1-40 dms; with the flop model's ratio (factored/dm, lowrank_cheaper's terms) and choice. Medians of 3
interleaved runs.
Usage: lowrank_paths_time.py SYSTEM XC [XC ...]     systems: benzene caffeine caffeine-tzvp c20h42
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
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=20000)
    mf = dft.RKS(mol, xc=xc).run(conv_tol=1e-8)
    ni, grids = mf._numint, mf.grids
    xctype = ni._xc_type(xc)
    mo, occ = mf.mo_coeff, mf.mo_occ
    orbo, orbv = mo[:, occ > 0], mo[:, occ == 0]
    nocc, nvir = orbo.shape[1], orbv.shape[1]
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, xc, mo, occ, 1)
    fxc = .5 * (fxc[0, :, 0] + fxc[0, :, 1])
    fused = numint._fused_xc_mo(ni, mol, grids, xctype, MAX_MEMORY)
    ns = fused.data.group_naos(mol)
    c_dm, c_rho, c_v, c_pass = {'LDA': (3, 2, 2, 2), 'GGA': (4, 4, 4, 8),
                                'MGGA': (13, 8, 10, 8)}[xctype]
    print(f'{name} {xc}: nao={mol.nao} nocc={nocc} ngrids={grids.weights.size} '
          f'threads={lib.num_threads()}', flush=True)
    for nset in (1, 2, 5, 10, 20, 40):
        z = np.random.default_rng(1).standard_normal((nset, nocc, nvir))
        a = lib.einsum('xov,pv->xpo', z, orbv)
        dm1 = lib.einsum('xov,pv,qo->xpq', z, orbv, orbo * 2)

        def dm_path():
            return ni.nr_rks_fxc(mol, grids, xc, None, dm1, 0, 0, rho0, vxc, fxc,
                                 max_memory=MAX_MEMORY)

        def two():
            out = np.zeros((nset, nvir, nocc))
            return fused.rks_fxc_mo(ni, xctype, fxc, a, orbo, orbv, orbo, out, one_pass=False)

        def one():
            out = np.zeros((nset, nvir, nocc))
            return fused.rks_fxc_mo(ni, xctype, fxc, a, orbo, orbv, orbo, out, one_pass=True)

        t_dm, t_two, t_one = medians((dm_path, two, one))
        ratio = ((nset * (c_rho + c_v) + c_pass) * nocc * ns.sum()
                 / (nset * c_dm * (ns**2).sum()))
        cheaper = fused.lowrank_cheaper(xctype, nocc, nocc, nset)
        print(f'  nset={nset:3d}: dm path {t_dm:7.3f} s, factored two kernels {t_two:7.3f} s, '
              f'one kernel {t_one:7.3f} s; dm/one {t_dm / t_one:5.2f}, two/one {t_two / t_one:5.2f}; '
              f'model flops one/dm {ratio:5.2f} -> {"factored" if cheaper else "dm path"}', flush=True)


if __name__ == '__main__':
    name = sys.argv[1]
    for xc in sys.argv[2:] or ('B3LYP',):
        run(name, xc)
