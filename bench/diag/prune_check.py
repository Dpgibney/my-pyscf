"""Benzene/def2-SVP B3LYP Hessian at tight SCF convergence, optionally with a given small_rho_cutoff.

Used to test whether the 5e-7 wheel-vs-master Hessian difference is master's grid-pruning default (#3192).
Usage: python prune_check.py [small_rho_cutoff]
"""
import json
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE
from pyscf import gto, dft, __version__

mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.chkfile = None
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = 100
if len(sys.argv) > 1:
    mf.small_rho_cutoff = float(sys.argv[1])
mf.kernel()
assert mf.converged
h = mf.Hessian().kernel()
print('@@' + json.dumps({'version': __version__, 'cutoff': mf.small_rho_cutoff, 'e_tot': mf.e_tot,
                         'ngrids': int(mf.grids.weights.size), 'hessian': h.tolist()}))
