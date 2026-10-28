"""How far TDA excitation energies are converged at a given residual tolerance: the tight validation
case (tight SCF, 5 states) at conv_tol 1e-6 and at a tighter tolerance with lindep 1e-16, printing the
energies and the Davidson iterations. Compare the output of two builds.
Usage: td_conv_sensitivity.py SYSTEM [XC]   (run under bench/diag/run_in.py)"""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from cases import converged_ks
from molecules import BENZENE, CAFFEINE

which = sys.argv[1]
xc = sys.argv[2] if len(sys.argv) > 2 else 'B3LYP'
mf = converged_ks({'benzene': BENZENE, 'caffeine': CAFFEINE}[which], 'def2-svp', xc, tight=True)
print(f'e_tot {mf.e_tot:.15f}')
for tol, lindep in ((1e-6, 1e-12), (1e-8, 1e-16)):
    td = mf.TDA()
    td.nstates = 5
    td.conv_tol = tol
    td.max_cycle = 200
    td.verbose = 5
    td.stdout = sys.stdout
    td.lindep = lindep
    td.kernel()
    print(f'RESULT conv_tol={tol:g} lindep={lindep:g} converged={list(td.converged)}')
    print('RESULT e = ' + ' '.join(f'{e:.15f}' for e in td.e), flush=True)
