"""Per-cycle energy change and orbital gradient of the tight caffeine+/B3LYP UKS SCF (the validation case),
to see whether a build drifts instead of converging. Usage: scf_uks_trace.py [--full-jk]
(run under bench/diag/run_in.py)"""
import sys

sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from cases import make_ks
from molecules import CAFFEINE
from pyscf import gto

mol = gto.M(atom=CAFFEINE, basis='def2-svp', charge=1, spin=1, verbose=0)
mf = make_ks(mol, "B3LYP", tight=True, full_jk="--full-jk" in sys.argv)
trace = []


def cb(env):
    trace.append((env['cycle'], env['e_tot'], np.linalg.norm(env['mf'].get_grad(env['mo_coeff'], env['mo_occ'], env['fock']))))


mf.callback = cb
mf.kernel()
e_prev = None
for cyc, e, g in trace:
    if cyc >= 25:
        print(f'{cyc:3d} {e:.13f} dE {e - e_prev:+.2e} |g| {g:.2e}')
    e_prev = e
print('converged', mf.converged, 'cycles', mf.cycles)
