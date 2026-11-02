"""Nuclear gradients of the Coulomb energy against the exact one (node1 or laptop): PySCF's int2e_ip1 on the CPU
without screening (_vhf.direct_mapdm with no vhfopt), g_A = 2 sum_{i on A} ((-nabla i) j|kl) D_lk D_ij. The GPU
gradients come from j_grad_proto.py --out (keys jengine and rys). The exact gradient is cached in EXACT.npy.
Usage: j_grad_exact.py --dm DM.npy --basis def2-svp --exact EXACT.npy GRADS.npz"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf.scf import _vhf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', required=True)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--exact', required=True)
ap.add_argument('grads')
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
dm = numpy.load(args.dm)
if os.path.exists(args.exact):
    gx = numpy.load(args.exact)
else:
    t0 = time.time()
    vj = -_vhf.direct_mapdm(mol._add_suffix('int2e_ip1'), 's2kl', 'lk->s1ij', dm, 3, mol._atm, mol._bas, mol._env)
    gx = numpy.zeros((mol.natm, 3))
    for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        gx[a] = 2 * numpy.einsum('xij,ij->x', vj[:, p0:p1], dm[p0:p1])
    numpy.save(args.exact, gx)
    print(f'exact J gradient (no screening) in {time.time() - t0:.0f} s')
g = numpy.load(args.grads)
print(f'max |g| {abs(gx).max():.4f}')
for k in ('rys', 'jengine'):
    e = g[k] - gx
    print(f'{k:8s} max err {abs(e).max():.2e}, rms {numpy.sqrt((e ** 2).mean()):.2e}, '
          f'sum over atoms {abs(g[k].sum(axis=0)).max():.1e}')
