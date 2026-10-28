"""Residual trace of the TDA eigensolver (pyscf.tdscf._lr_eig.eigh, conv_tol = residual tolerance) on caffeine
def2-SVP B3LYP: where does max|r| stall?  Usage: td_residual_trace.py [MAX_CYCLE] [LINDEP]"""
import re
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto, dft, lib

max_cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 40
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.chkfile = None
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.kernel()
print('scf', mf.converged, mf.e_tot, 'incore eri' if mf._eri is not None else 'direct', flush=True)

td = mf.TDA()
td.nstates = 5
td.conv_tol = 1e-8
td.max_cycle = max_cycle
if len(sys.argv) > 2:
    td.lindep = float(sys.argv[2])
print('lindep', td.lindep, flush=True)


class Log:
    def write(self, s):
        m = re.search(r'davidson (\d+) (\d+) +\|r\|= *(\S+).*max\|de\|= *(\S+) +lindep= *(\S+)', s, re.S)
        if m:
            print(f'it {m.group(1):>3s} nvec {m.group(2):>4s} max|r| {m.group(3):>9s} max|de| {m.group(4):>9s} min|correction| {m.group(5)}', flush=True)

    def flush(self):
        pass


td.stdout = Log()
td.verbose = lib.logger.DEBUG
td.kernel()
print('converged', list(td.converged), 'e', list(td.e), flush=True)
