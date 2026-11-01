"""Peak GPU memory of a Hessian XC derivative term (node1): _get_vxc_deriv2 (the default) or _get_vxc_deriv1, run
once at the orbitals of MO.npz (from vxc_deriv_check.py --save-mo). It prints the peak of the memory in use (tracked
at every allocation of CuPy's pool, so it is the live memory) and the pool's size (which also holds freed blocks
that it caches), both counted from the start of the call. Source scripts/node1-gpu-env.sh perf|base first.
Usage: deriv_mem.py --mo MO.npz [--func deriv2|deriv1] [--mol caffeine|c20h42] [--xc b3lyp] [--basis def2-svp]
                    [--uks]"""
import argparse
import os
import sys
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
ap.add_argument('--func', default='deriv2', choices=['deriv2', 'deriv1'])
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'c20h42'])
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--uks', action='store_true')
args = ap.parse_args()

atom = {'caffeine': CAFFEINE, 'c20h42': n_alkane(20)}[args.mol]
if args.uks:
    mol = gto.M(atom=atom, basis=args.basis, charge=1, spin=1, verbose=0)
    mf = dft.UKS(mol, xc=args.xc)
else:
    mol = gto.M(atom=atom, basis=args.basis, verbose=0)
    mf = dft.RKS(mol, xc=args.xc)
mf.grids.build()
mo = numpy.load(args.mo)
mo_coeff, mo_occ = cupy.asarray(mo['mo_coeff']), cupy.asarray(mo['mo_occ'])
hobj = mf.Hessian()
f = getattr(uks_hess if args.uks else rks_hess, f'_get_vxc_{args.func}')
if not args.uks:
    mf._numint.build(mol, mf.grids.coords)   # gdftopt, outside the measurement

pool = cupy.get_default_memory_pool()
cupy.cuda.Device().synchronize()
pool.free_all_blocks()
base = pool.used_bytes()
peak = [base]


def malloc(size):
    mem = pool.malloc(size)
    peak[0] = max(peak[0], pool.used_bytes())
    return mem


cupy.cuda.set_allocator(malloc)
v = f(hobj, mo_coeff, mo_occ, 4000)
cupy.cuda.Device().synchronize()
cupy.cuda.set_allocator(pool.malloc)
print(f'{os.path.dirname(gpu4pyscf.__file__)}: {"UKS" if args.uks else "RKS"} {args.mol} {args.xc} {args.basis} '
      f'_get_vxc_{args.func}: peak in use {(peak[0] - base) / 1e6:.0f} MB, pool {(pool.total_bytes() - base) / 1e6:.0f} MB '
      f'(before the call: {base / 1e6:.0f} MB in use)')
