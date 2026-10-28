"""Log every J/K call (path, number of dms, hermi, J/K flags, time) during a TDA/TDDFT run.
Usage: td_jk_calls.py SYSTEM TDA|TDDFT [XC]   (run under bench/diag/run_in.py)"""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import dft, gto
from pyscf.scf import _vhf

system, method = sys.argv[1], sys.argv[2]
xc = sys.argv[3] if len(sys.argv) > 3 else 'B3LYP'
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'c20': n_alkane(20)}[system]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc=xc)
mf.kernel()
calls = []
for name in ('incore', 'direct'):
    orig = getattr(_vhf, name)

    def wrapped(*args, _orig=orig, _name=name, **kwargs):
        dms = np.asarray(args[1] if _name == 'incore' else args[0])
        t0 = time.perf_counter()
        out = _orig(*args, **kwargs)
        hermi = kwargs.get('hermi', args[2] if _name == 'incore' else (args[5] if len(args) > 5 else 0))
        n = 1 if dms.ndim == 2 else len(dms.reshape(-1, mol.nao, mol.nao))
        calls.append((_name, n, hermi, time.perf_counter() - t0))
        return out
    setattr(_vhf, name, wrapped)
td = getattr(mf, method)()
td.nstates = 5
t0 = time.perf_counter()
td.kernel()
print(f'{system} {method} {xc}: {time.perf_counter() - t0:.1f} s, J/K {sum(c[3] for c in calls):.1f} s '
      f'in {len(calls)} calls, {sum(c[1] for c in calls)} dms')
for c in calls:
    print(f'  {c[0]:6s} n_dm={c[1]:3d} hermi={c[2]} {c[3]:7.3f} s')
