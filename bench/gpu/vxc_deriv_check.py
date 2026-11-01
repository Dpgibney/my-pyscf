"""An XC derivative term of GPU4PySCF's RKS or UKS Hessian (node1), _get_vxc_deriv2 (the default) or
_get_vxc_deriv1 (--func deriv1), for identical orbitals in two trees: with --save-mo the SCF runs first and its
mo_coeff and mo_occ go to MO.npz; otherwise they are loaded from MO.npz. The function runs once untimed (unless
--no-warmup) and once timed; its result goes to --out, and --compare REF.npy prints the largest difference. UKS uses
the cation (doublet). Source scripts/node1-gpu-env.sh perf|base first.
Usage: vxc_deriv_check.py --mo MO.npz [--save-mo] [--func deriv2|deriv1] [--mol caffeine|c20h42] [--xc b3lyp]
                          [--basis def2-svp] [--uks] [--no-warmup] [--out OUT.npy] [--compare REF.npy]"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE, n_alkane  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft  # noqa: E402
from gpu4pyscf.hessian import rks as rks_hess, uks as uks_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mo', required=True)
ap.add_argument('--save-mo', action='store_true')
ap.add_argument('--func', default='deriv2', choices=['deriv2', 'deriv1'])
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'c20h42'])
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--uks', action='store_true')
ap.add_argument('--no-warmup', action='store_true')
ap.add_argument('--out', default=None)
ap.add_argument('--compare', default=None)
args = ap.parse_args()

atom = {'caffeine': CAFFEINE, 'c20h42': n_alkane(20)}[args.mol]
if args.uks:
    mol = gto.M(atom=atom, basis=args.basis, charge=1, spin=1, verbose=0)
    mf = dft.UKS(mol, xc=args.xc)
else:
    mol = gto.M(atom=atom, basis=args.basis, verbose=0)
    mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
if args.save_mo:
    mf.kernel()
    numpy.savez(args.mo, mo_coeff=cupy.asnumpy(mf.mo_coeff), mo_occ=cupy.asnumpy(mf.mo_occ))
else:
    mf.grids.build()
mo = numpy.load(args.mo)
mo_coeff, mo_occ = cupy.asarray(mo['mo_coeff']), cupy.asarray(mo['mo_occ'])
hobj = mf.Hessian()
f = getattr(uks_hess if args.uks else rks_hess, f'_get_vxc_{args.func}')
if not args.no_warmup:
    v = f(hobj, mo_coeff, mo_occ, 4000)
cupy.cuda.Device().synchronize()
t0 = time.perf_counter()
v = f(hobj, mo_coeff, mo_occ, 4000)
cupy.cuda.Device().synchronize()
t = time.perf_counter() - t0
# UKS: the alpha and beta parts, flattened (deriv1's have different numbers of occupied orbitals)
v = numpy.concatenate([cupy.asnumpy(x).ravel() for x in v]) if isinstance(v, (tuple, list)) else cupy.asnumpy(v)
print(f'{os.path.dirname(gpu4pyscf.__file__)}: {"UKS" if args.uks else "RKS"} {args.mol} {args.xc} {args.basis} '
      f'_get_vxc_{args.func} {t:.2f} s')
if args.out:
    numpy.save(args.out, v)
if args.compare:
    ref = numpy.load(args.compare)
    print(f'  max |v - ref| = {abs(v - ref).max():.2e} (max |ref| {abs(ref).max():.2e})')
