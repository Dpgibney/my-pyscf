"""Noise of the TDA response operator on caffeine def2-SVP B3LYP: |A(x+y) - A(x) - A(y)| and |x.A(y) - y.A(x)|
for random unit trial vectors, with the default integral screening of direct J/K and with screening off.
Run with PYTHONPATH selecting the PySCF tree."""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE
from pyscf import gto, dft

mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='B3LYP')
mf.chkfile = None
mf.kernel()
print('scf', mf.converged, 'incore eri' if mf._eri is not None else 'direct J/K', flush=True)
td = mf.TDA()
nocc = mol.nelectron // 2
x, y = np.random.default_rng(7).standard_normal((2, nocc * (mol.nao - nocc)))
x /= np.linalg.norm(x)
y /= np.linalg.norm(y)
for label, tol in (('direct_scf_tol=%g' % mf.direct_scf_tol, None), ('direct_scf_tol=1e-20', 1e-20)):
    if tol is not None:
        mf.direct_scf_tol = tol
        mf._opt = {None: None}
    vind, hdiag = td.gen_vind(mf)
    ax, ay, axy = vind(np.array([x, y, x + y]))
    print(f'{label:22s} nonlinearity {np.linalg.norm(axy - ax - ay):.2e}  asymmetry {abs(x @ ay - y @ ax):.2e}  '
          f'|Ax| {np.linalg.norm(ax):.3f}', flush=True)
