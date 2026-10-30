"""The UKS Hessian's make_h1 (benzene+/def2-SVP, B3LYP and PBE): best of 3 calls, `save` in one tree, `cmp` in
another on the saved orbitals (time and max relative difference of h1ao).
Usage: make_h1_time.py save|cmp OUT   (run under bench/diag/run_in.py)
"""
import sys, time
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE
from pyscf import gto, dft
from pyscf.hessian import uks as uks_hess
mode, out = sys.argv[1], sys.argv[2]
mol = gto.M(atom=BENZENE, basis='def2-svp', charge=1, spin=1, verbose=0)
res = {}
for xc in ('B3LYP', 'PBE'):
    mf = dft.UKS(mol, xc=xc)
    if mode == 'cmp':
        ref = np.load(out)
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = ref[f'{xc}_mo'], ref[f'{xc}_occ'], ref[f'{xc}_e']
        mf.grids.build(with_non0tab=True)
    else:
        mf.run(conv_tol=1e-10)
    hobj = mf.Hessian()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        h1a, h1b = uks_hess.make_h1(hobj, mf.mo_coeff, mf.mo_occ)
        ts.append(time.perf_counter() - t0)
    res[f'{xc}_mo'], res[f'{xc}_occ'], res[f'{xc}_e'] = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    res[f'{xc}_h1'] = np.asarray([h1a, h1b])
    msg = f'{xc}: make_h1 best of 3 {min(ts):.3f} s'
    if mode == 'cmp':
        r = ref[f'{xc}_h1']
        msg += f', max|diff|/max|ref| {abs(res[f"{xc}_h1"] - r).max() / abs(r).max():.1e}'
    print(msg, flush=True)
if mode == 'save':
    np.savez(out, **res)
