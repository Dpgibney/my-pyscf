"""The exchange of a range-separated hybrid (node1). GPU4PySCF builds lr*K(erf) + sr*K(erfc) as one pass of the
operator sr/r + (lr - sr) erf(w r)/r: two sets of Rys roots per quartet, screened by the erf part's Schwarz bounds.
This times that pass against two separate ones, K(erf) (one set of roots, its Schwarz bounds) and K(erfc) (two sets,
its own Schwarz bounds and the distance screen of the short-range tasks), and the full-range K for scale, at a fixed
density; it checks lr*K(erf) + sr*K(erfc) against the one pass.
Usage: rsh_k_split.py [--mol caffeine|benzene] [--basis def2-svp] [--xc wb97m_v] [--dm DM.npy] [--reps 3]"""
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
from gpu4pyscf.dft import rks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='wb97m_v')
ap.add_argument('--dm', default=None, help='density matrix (.npy); without it, the SCF of --xc')
ap.add_argument('--reps', type=int, default=3)
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
if args.dm:
    dm = cupy.asarray(numpy.load(args.dm))
else:
    mf.kernel()
    dm = mf.make_rdm1()
omega, lr, sr = mf._numint.rsh_and_hybrid_coeff(args.xc, spin=mol.spin)
assert omega != 0, 'not a range-separated functional'


def timed(fn):
    fn()  # first call: builds this omega's screening data
    ts = []
    for _ in range(args.reps):
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out = fn()
        cupy.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts), cupy.asarray(out)


t_one, k_one = timed(lambda: mf.get_k(mol, dm, 1, omega, lr, sr))
t_lr, k_lr = timed(lambda: mf.get_k(mol, dm, 1, abs(omega)))    # libcint convention: omega > 0, the erf part
t_sr, k_sr = timed(lambda: mf.get_k(mol, dm, 1, -abs(omega)))   # omega < 0, the erfc part
t_full, _ = timed(lambda: mf.get_k(mol, dm, 1))
diff = float(abs(k_one - (lr * k_lr + sr * k_sr)).max())
print(f'{os.path.dirname(gpu4pyscf.__file__)}: {args.mol} {args.basis} {args.xc} (omega {omega}, lr {lr}, sr {sr}), '
      f'nao {mol.nao}, best of {args.reps}:')
print(f'  one pass lr*K(erf) + sr*K(erfc)  {t_one:7.3f} s')
print(f'  K(erf)                           {t_lr:7.3f} s')
print(f'  K(erfc)                          {t_sr:7.3f} s')
print(f'  the two passes                   {t_lr + t_sr:7.3f} s ({t_one / (t_lr + t_sr):.2f}x the one pass)')
print(f'  full-range K (for scale)         {t_full:7.3f} s')
print(f'  max |K(one pass) - (lr K(erf) + sr K(erfc))| {diff:.2e} (max |K| {float(abs(k_one).max()):.2e})')
