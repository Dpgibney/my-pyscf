"""VV10's part of the XC potential at a fixed density (node1): numint.nr_nlc_vxc of a VV10 functional on its NLC grid,
once untimed and then timed, for comparing builds of gdft/vv10.cu (part 26). Writes (nelec, exc, vxc) to OUT.npz;
--compare REF.npz prints the differences. --grad does the same for the gradient's VV10 term with the grid response,
rks_grad.get_nlc_exc_full_response (part 28), and writes de (natm x 3).
Usage: vv10_at_dm.py --dm DM.npy [--xc b97m_v] [--basis def2-svp] [--mol caffeine|benzene] [--out OUT.npz]
                     [--compare REF.npz] [--grad] [--memcheck] (--memcheck: CuPy's pool off, one call, for
                     compute-sanitizer)"""
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
from gpu4pyscf.grad import rks as rks_grad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dm', default=None, help='density (caffeine); without it, the core guess')
ap.add_argument('--xc', default='b97m_v')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--out', default=None)
ap.add_argument('--compare', default=None)
ap.add_argument('--grad', action='store_true')
ap.add_argument('--memcheck', action='store_true')
args = ap.parse_args()

if args.memcheck:
    cupy.cuda.set_allocator(cupy.cuda.memory._malloc)
mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
dm = cupy.asarray(numpy.load(args.dm)) if args.dm else cupy.asarray(mf.get_init_guess())
mf.nlcgrids.build()
ni = mf._numint


def call():
    if args.grad:
        out = rks_grad.get_nlc_exc_full_response(ni, mol, mf.nlcgrids, args.xc, dm)[0]
    else:
        out = ni.nr_nlc_vxc(mol, mf.nlcgrids, args.xc, dm)
    cupy.cuda.Device().synchronize()
    return out


if args.grad:
    de = call()
    if not args.memcheck:
        t0 = time.perf_counter()
        de = call()
        dt = time.perf_counter() - t0
        print(f'{os.path.dirname(gpu4pyscf.__file__)}: {args.mol} {args.basis} {args.xc} get_nlc_exc_full_response '
              f'{dt:.2f} s, {mf.nlcgrids.weights.size} NLC grid points; max |de| {abs(de).max():.6e}')
    if args.out:
        numpy.savez(args.out, de=de)
    if args.compare:
        ref = numpy.load(args.compare)['de']
        print(f'  against {os.path.basename(args.compare)}: max |de - ref| {abs(de - ref).max():.2e} '
              f'(max |de| {abs(ref).max():.2e}), max |sum over atoms| {abs(de.sum(axis=0)).max():.2e} '
              f'(ref {abs(ref.sum(axis=0)).max():.2e})')
    sys.exit()

nelec, exc, vxc = call()
if not args.memcheck:
    t0 = time.perf_counter()
    nelec, exc, vxc = call()
    dt = time.perf_counter() - t0
    print(f'{os.path.dirname(gpu4pyscf.__file__)}: {args.mol} {args.basis} {args.xc} nr_nlc_vxc {dt:.2f} s, '
          f'{mf.nlcgrids.weights.size} NLC grid points; exc {float(exc):.12f}')
vxc = cupy.asnumpy(vxc)
if args.out:
    numpy.savez(args.out, nelec=float(nelec), exc=float(exc), vxc=vxc)
if args.compare:
    ref = numpy.load(args.compare)
    print(f'  against {os.path.basename(args.compare)}: exc {float(exc) - float(ref["exc"]):.2e}, '
          f'max |vxc - ref| {abs(vxc - ref["vxc"]).max():.2e} (max |vxc| {abs(ref["vxc"]).max():.2e})')
