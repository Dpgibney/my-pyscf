"""The response calls of a TD run: per call the number of trial vectors, the time of the XC response
(nr_rks_fxc) and of the rest of the response (J/K), and what the factored path's flop model
(_FusedXC.lowrank_cheaper with r = nq = nocc) would choose for that many sets, with its flop ratio.
Usage: td_fxc_calls.py SYSTEM XC METHOD     e.g. caffeine B3LYP TDA; METHOD is TDA, TDDFT or CasidaTDDFT
"""
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.scf import _response_functions  # noqa: F401, attaches gen_response

SYSTEMS = {'benzene': (BENZENE, 'def2-svp'), 'caffeine': (CAFFEINE, 'def2-svp'),
           'caffeine-tzvp': (CAFFEINE, 'def2-tzvp'), 'c20h42': (n_alkane(20), 'def2-svp')}


def main(name, xc, method):
    atom, basis = SYSTEMS[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.kernel()
    nocc = int((mf.mo_occ > 0).sum())
    xctype = mf._numint._xc_type(xc)
    calls = []
    nr_rks_fxc = numint.NumInt.nr_rks_fxc

    def timed_fxc(ni, mol, grids, xc_code, dm0, dms, *args, **kwargs):
        t0 = time.perf_counter()
        v = nr_rks_fxc(ni, mol, grids, xc_code, dm0, dms, *args, **kwargs)
        calls[-1]['xc'] += time.perf_counter() - t0
        return v
    numint.NumInt.nr_rks_fxc = timed_fxc

    gen_response = type(mf).gen_response

    def gen_response_timed(mf, *args, **kwargs):
        vresp = gen_response(mf, *args, **kwargs)

        def vind(dm1):
            calls.append({'nset': len(dm1), 'xc': 0.})
            t0 = time.perf_counter()
            v = vresp(dm1)
            calls[-1]['all'] = time.perf_counter() - t0
            return v
        return vind
    type(mf).gen_response = gen_response_timed

    td = getattr(mf, method)()
    td.nstates = 5
    t0 = time.perf_counter()
    td.kernel()
    wall = time.perf_counter() - t0
    numint.NumInt.nr_rks_fxc = nr_rks_fxc
    type(mf).gen_response = gen_response

    fused = numint._fused_xc(mf._numint, mol, mf.grids, mol.nao, xctype, 4000, (mf.mo_coeff,),
                             match_block_loop=False)
    ns = fused.data.group_naos(mol)
    c_dm, c_rho, c_v, c_pass = {'LDA': (3, 2, 2, 2), 'GGA': (4, 4, 4, 8),
                                'MGGA': (13, 8, 10, 8)}[xctype]
    print(f'{name} {xc} {method}: nao={mol.nao} nocc={nocc} ngrids={mf.grids.weights.size} '
          f'threads={lib.num_threads()} wall={wall:.2f} s, {len(calls)} calls, '
          f'{sum(c["nset"] for c in calls)} sets; e={np.asarray(td.e)}')
    print('  call nset    XC (s)  rest (s)  XC/set (ms)  flops factored/dm  model')
    for i, c in enumerate(calls):
        n = c['nset']
        ratio = ((n * (c_rho + c_v) + 2 * c_pass) * nocc * ns.sum() / (n * c_dm * (ns**2).sum()))
        cheaper = fused.lowrank_cheaper(xctype, nocc, nocc, n)
        print(f'  {i:4d} {n:4d}  {c["xc"]:8.3f}  {c["all"] - c["xc"]:8.3f}  {c["xc"] / n * 1e3:11.1f}'
              f'  {ratio:17.2f}  {"factored" if cheaper else "dm path"}')
    xc_t = sum(c['xc'] for c in calls)
    rest = sum(c['all'] for c in calls) - xc_t
    print(f'  total XC {xc_t:.2f} s, rest of the response {rest:.2f} s, '
          f'other {wall - xc_t - rest:.2f} s')


if __name__ == '__main__':
    main(*sys.argv[1:4])
