import sys
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE
from pyscf import gto, dft
from pyscf.dft import numint
if sys.argv[1] == 'nosyrk':
    numint._same_ao = lambda a, b: False
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='r2SCAN'); mf.chkfile = None; mf.conv_tol = 1e-12; mf.conv_tol_grad = 1e-8; mf.max_cycle = 100
hist = []
mf.callback = lambda env: hist.append((env['e_tot'], env['norm_gorb']))
mf.kernel()
print(f"{sys.argv[1]:7s} E={mf.e_tot:.12f} cycles={mf.cycles} conv={mf.converged}")
print('   |g| by cycle:', ' '.join(f'{g:.0e}' for _, g in hist[8:40]))
