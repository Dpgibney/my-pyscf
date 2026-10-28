"""Dense vs sparse grid path with the new kernels: nr_rks, nr_rks_fxc (10 dms) and the gradient XC term,
forcing every block to one path via numint._sparse_enough.
Usage: switch_sweep.py SYSTEM [XC]"""
import functools
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf.grad import rks as rks_grad

which = sys.argv[1]
xc = sys.argv[2] if len(sys.argv) > 2 else 'B3LYP'
atom, basis = {'water': ('O 0 0 0; H 0 -.757 .587; H 0 .757 .587', 'def2-svp'),
               'water-tz': ('O 0 0 0; H 0 -.757 .587; H 0 .757 .587', 'def2-tzvp'),
               'benzene': (BENZENE, 'def2-svp'), 'benzene-tz': (BENZENE, 'def2-tzvp'),
               'c8': (n_alkane(8), 'def2-svp'), 'c12': (n_alkane(12), 'def2-svp'),
               'caffeine': (CAFFEINE, 'def2-svp'), 'caffeine-tz': (CAFFEINE, 'def2-tzvp')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = dft.RKS(mol, xc=xc)
mf.grids.build(with_non0tab=True)
mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
dm = mf.make_rdm1(mo_c, mf.get_occ(mo_e, mo_c))
ni = mf._numint
dm1 = np.random.default_rng(1).standard_normal((10, mol.nao, mol.nao)) * 1e-2
nz = np.count_nonzero(mf.grids.non0tab) / mf.grids.non0tab.size

def best(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)

orig = numint._sparse_enough
fns = (lambda: ni.nr_rks(mol, mf.grids, xc, dm),
       lambda: ni.nr_rks_fxc(mol, mf.grids, xc, dm, dm1, hermi=1),
       lambda: rks_grad.get_vxc(ni, mol, mf.grids, xc, dm))
modes = {'dense': functools.partial(orig, threshold=0.0), 'sparse': functools.partial(orig, threshold=2.0)}
ts = {m: [[], [], []] for m in modes}
for rep in range(3):   # interleaved dense/sparse, alternating which goes first (clock drift)
    for m in (('dense', 'sparse') if rep % 2 == 0 else ('sparse', 'dense')):
        numint._sparse_enough = modes[m]
        for k, fn in enumerate(fns):
            if rep == 0:
                fn()   # warm-up
            t0 = time.perf_counter(); fn(); ts[m][k].append(time.perf_counter() - t0)
d = [min(x) for x in ts['dense']]
s = [min(x) for x in ts['sparse']]
print(f'{which:12s} {xc:6s} nao={mol.nao:4d} nonzero={nz:.2f} | nr_rks {d[0]:.3f} -> {s[0]:.3f} s '
      f'({d[0]/s[0]:.2f}x) | fxc x10 {d[1]:.3f} -> {s[1]:.3f} s ({d[1]/s[1]:.2f}x) | '
      f'grad vxc {d[2]:.3f} -> {s[2]:.3f} s ({d[2]/s[2]:.2f}x)', flush=True)
