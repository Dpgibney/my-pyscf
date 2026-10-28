"""Time NumInt.nr_rks with the sparse/dense switch in block_loop forced each way.
Usage: sparse_switch.py [c20|c20tz|caffeine] [XC]
"""
import functools
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import gto, dft
from pyscf.dft import numint

which = sys.argv[1] if len(sys.argv) > 1 else 'c20'
xc = sys.argv[2] if len(sys.argv) > 2 else 'B3LYP'
atom, basis = {'c20': (n_alkane(20), 'def2-svp'), 'c20tz': (n_alkane(20), 'def2-tzvp'),
               'caffeine': (CAFFEINE, 'def2-svp')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = dft.RKS(mol, xc=xc)
mf.grids.build(with_non0tab=True)
# MO-tagged density (core guess) so that make_rho takes the eval_rho2 path, as in SCF iterations
mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
dm = mf.make_rdm1(mo_c, mf.get_occ(mo_e, mo_c))
ni = mf._numint
nz = np.count_nonzero(mf.grids.non0tab) / mf.grids.non0tab.size
print(f'{which} {xc} nao={mol.nao} ngrids={mf.grids.weights.size} nonzero screen fraction={nz:.3f}')

orig = numint._sparse_enough
res = {}
for label, thr in [('default', 0.5), ('dense', 0.0), ('sparse', 2.0)]:
    numint._sparse_enough = functools.partial(orig, threshold=thr)
    ni.nr_rks(mol, mf.grids, xc, dm)  # warm-up
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        n, exc, vxc = ni.nr_rks(mol, mf.grids, xc, dm)
        ts.append(time.perf_counter() - t0)
    res[label] = (exc, vxc)
    print(f'{label:8s} best {min(ts):7.3f} s  ({" ".join(f"{t:.3f}" for t in ts)})  exc={exc:.12f}',
          flush=True)
numint._sparse_enough = orig
for label in ('dense', 'sparse'):
    dv = np.abs(res[label][1] - res['default'][1]).max()
    print(f'{label} vs default: max|dV|={dv:.1e} |dExc|={abs(res[label][0] - res["default"][0]):.1e}')
