"""Run-to-run spread of the tight SCF endpoint (cycles, final orbital gradient, density) for a given
conv_tol_grad, as seen by post-SCF cases (TD, gradients, Hessians).
Usage: scf_endpoint_spread.py SYSTEM XC CONV_TOL_GRAD [RUNS] [--full-jk]   (run under bench/diag/run_in.py)"""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from cases import _full_jk_rebuild, make_ks
from molecules import BENZENE, CAFFEINE
from pyscf import gto

which, xc, tol_grad = sys.argv[1], sys.argv[2], float(sys.argv[3])
runs = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 3
full_jk = '--full-jk' in sys.argv
mol = gto.M(atom={'benzene': BENZENE, 'caffeine': CAFFEINE}[which], basis='def2-svp', verbose=0)
dms = []
for r in range(runs):
    mf = make_ks(mol, xc, tight=True)
    mf.conv_tol_grad = tol_grad
    if full_jk:
        _full_jk_rebuild(mf)
    mf.kernel()
    g = np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))
    dms.append(mf.make_rdm1())
    print(f'conv_tol_grad={tol_grad:g} run {r}: converged {mf.converged} cycles {mf.cycles} '
          f'|g| {g:.1e} e_tot {mf.e_tot:.13f}', flush=True)
spread = max(abs(a - b).max() for i, a in enumerate(dms) for b in dms[i + 1:])
print(f'conv_tol_grad={tol_grad:g} full_jk={full_jk}: max |P_a - P_b| over runs = {spread:.1e}')
