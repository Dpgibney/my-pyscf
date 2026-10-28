"""Where does NumInt.nr_rks spend its time on the sparse path, and how sparse is the screen index?
Usage: sparse_profile.py [c20|c20tz|caffeine] [XC]
"""
import collections
import functools
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import gto, dft, lib
from pyscf.dft import numint

which = sys.argv[1] if len(sys.argv) > 1 else 'c20'
xc = sys.argv[2] if len(sys.argv) > 2 else 'B3LYP'
atom, basis = {'c20': (n_alkane(20), 'def2-svp'), 'c20tz': (n_alkane(20), 'def2-tzvp'),
               'c40': (n_alkane(40), 'def2-svp'), 'caffeine': (CAFFEINE, 'def2-svp')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = dft.RKS(mol, xc=xc)
mf.grids.build(with_non0tab=True)
mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
dm = mf.make_rdm1(mo_c, mf.get_occ(mo_e, mo_c))
ni = mf._numint
si = mf.grids.non0tab
ao_loc = mol.ao_loc_nr()
shl_size = np.diff(ao_loc)
nao, ngrids = mol.nao, mf.grids.weights.size
print(f'{which} {xc} nao={nao} nbas={mol.nbas} ngrids={ngrids} nblk={si.shape[0]} '
      f'threads={lib.num_threads()}')

# --- sparsity of the screen index: active AOs per group of m consecutive 56-point blocks
nz = si != 0
dense_flops = 2. * nao**2 * ngrids
print(f'nonzero shell x block fraction {nz.mean():.3f}; dense ao^T ao flops {dense_flops/1e9:.0f} GFLOP')
for m in (1, 2, 4, 8, 16):
    nblk = si.shape[0]
    ng = (nblk + m - 1) // m
    pad = np.zeros((ng * m, mol.nbas), dtype=bool)
    pad[:nblk] = nz
    act = pad.reshape(ng, m, mol.nbas).any(axis=1)
    nact = act.astype(float) @ shl_size
    npts = np.full(ng, 56. * m)
    npts[-1] = ngrids - 56 * m * (ng - 1)
    fl = (2 * nact**2 * npts).sum()
    print(f'  group of {m:2d} blocks ({56*m:4d} pts): active AOs mean {nact.mean():6.1f} '
          f'p10 {np.percentile(nact,10):5.0f} p90 {np.percentile(nact,90):5.0f} max {nact.max():5.0f}; '
          f'gathered GEMM flops {fl/1e9:6.1f} GFLOP = {fl/dense_flops:.3f} of dense')

# --- time the kernels inside nr_rks
times = collections.defaultdict(float)
calls = collections.Counter()

def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        times[name] += time.perf_counter() - t0
        calls[name] += 1
        return r
    return wrapper

for name in ('_dot_ao_ao_sparse', '_dot_ao_dm_sparse', '_scale_ao_sparse', '_contract_rho_sparse',
             '_tau_dot_sparse', '_dot_ao_dm', '_contract_rho'):
    setattr(numint, name, timed(name, getattr(numint, name)))
ni.eval_ao = timed('eval_ao', ni.eval_ao)
ni.eval_xc_eff = timed('eval_xc_eff', ni.eval_xc_eff)
ni.eval_rho1 = timed('eval_rho1 (total)', ni.eval_rho1)
ni.eval_rho2 = timed('eval_rho2 (total)', ni.eval_rho2)

ni.nr_rks(mol, mf.grids, xc, dm)  # warm-up
times.clear(); calls.clear()
nrep = 2
t0 = time.perf_counter()
for _ in range(nrep):
    n, exc, vxc = ni.nr_rks(mol, mf.grids, xc, dm)
wall = (time.perf_counter() - t0) / nrep
print(f'nr_rks wall {wall:.3f} s per call, exc={exc:.12f}')
for name, t in sorted(times.items(), key=lambda kv: -kv[1]):
    print(f'  {name:22s} {t/nrep:7.3f} s  ({calls[name]//nrep} calls)')
