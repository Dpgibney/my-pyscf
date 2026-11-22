"""The second-derivative J/K term of a range-separated hybrid's Hessian (node1): hessian.rks.partial_hess_elec's
two-electron part, timed between device synchronizations (best of --repeat after one warm call) at a converged
density, with the result saved to or compared with a file. Run it in the baseline tree with --save and in the modified
tree with --compare: the passes are J + alpha K and then (hyb - alpha) K_SR before part 36, J + hyb K and then
(alpha - hyb) K_LR after.
Usage: rsh_hess_ejk.py [--mol benzene|caffeine] [--basis def2-svp] [--xc wb97x] [--uks] [--repeat 2]
                       [--save FILE.npy | --compare FILE.npy]"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks, uks  # noqa: E402
from gpu4pyscf.hessian import rks as rks_hess, uks as uks_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='benzene')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97x')
ap.add_argument('--uks', action='store_true')
ap.add_argument('--repeat', type=int, default=2)
ap.add_argument('--save', default=None)
ap.add_argument('--compare', default=None)
args = ap.parse_args()

mol = gto.M(atom={'benzene': BENZENE, 'caffeine': CAFFEINE}[args.mol], basis=args.basis, verbose=0)
mf = (uks.UKS if args.uks else rks.RKS)(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.kernel()
assert mf.converged
hobj = mf.Hessian()
mod = uks_hess if args.uks else rks_hess
# the XC terms are not what is timed here
zeros = cupy.zeros((mol.natm, mol.natm, 3, 3))
mod._get_exc_deriv2 = lambda *a, **kw: zeros
mod._get_enlc_deriv2 = lambda *a, **kw: zeros


def run():
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    de2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    cupy.cuda.Device().synchronize()
    return cupy.asnumpy(de2), time.perf_counter() - t0


run()
best = 1e99
for _ in range(args.repeat):
    de2, t = run()
    best = min(best, t)
msg = (f'{args.mol} {args.basis} {args.xc}{" UKS" if args.uks else ""}: partial_hess_elec without XC {best:.2f} s, '
       f'tree {os.path.dirname(gpu4pyscf.__file__)}')
if args.save:
    numpy.save(args.save, de2)
if args.compare:
    ref = numpy.load(args.compare)
    msg += f', max |d| from {args.compare} {abs(de2 - ref).max():.2e} (values up to {abs(ref).max():.2e})'
print(msg)
