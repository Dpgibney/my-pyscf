"""make_h1's int2e_ip1 J/K call for atom 0 with the Coulomb scripts per spin (8 scripts with K, 4 without) and on
the total density (6, 2); best and median of 5 interleaved calls.
Usage: h1_jk_scripts.py benzene+|caffeine+   (run under bench/diag/run_in.py)
"""
import sys, time
sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE
from pyscf import gto, dft
from pyscf.hessian.rhf import _get_jk
atom = {'benzene+': BENZENE, 'caffeine+': CAFFEINE}[sys.argv[1]]
mol = gto.M(atom=atom, basis='def2-svp', charge=1, spin=1, verbose=0)
mf = dft.UKS(mol, xc='B3LYP').run(conv_tol=1e-8)
ma = mf.mo_coeff[0][:, mf.mo_occ[0] > 0]; mb = mf.mo_coeff[1][:, mf.mo_occ[1] > 0]
dma, dmb = ma @ ma.T, mb @ mb.T
dm = dma + dmb
shl0, shl1, p0, p1 = mol.aoslice_by_atom()[0]
sl = (shl0, shl1) + (0, mol.nbas) * 3
K = ['li->s1kj', -dma[:, p0:p1], 'li->s1kj', -dmb[:, p0:p1], 'jk->s1il', -dma, 'jk->s1il', -dmb]
variants = {
    'hybrid, J per spin (8)': ['ji->s2kl', -dma[:, p0:p1], 'ji->s2kl', -dmb[:, p0:p1], 'lk->s1ij', -dma, 'lk->s1ij', -dmb] + K,
    'hybrid, J of total (6)': ['ji->s2kl', -dm[:, p0:p1], 'lk->s1ij', -dm] + K,
    'pure, J per spin (4)': ['ji->s2kl', -dma[:, p0:p1], 'ji->s2kl', -dmb[:, p0:p1], 'lk->s1ij', -dma, 'lk->s1ij', -dmb],
    'pure, J of total (2)': ['ji->s2kl', -dm[:, p0:p1], 'lk->s1ij', -dm],
}
ts = {k: [] for k in variants}
for rep in range(5):
    for k in (list(variants) if rep % 2 == 0 else list(variants)[::-1]):
        t0 = time.perf_counter()
        _get_jk(mol, 'int2e_ip1', 3, 's2kl', variants[k], shls_slice=sl)
        ts[k].append(time.perf_counter() - t0)
for k, t in ts.items():
    print(f'{sys.argv[1]} atom 0, {k}: best {min(t)*1e3:7.1f} ms, median {np.median(t)*1e3:7.1f} ms')
