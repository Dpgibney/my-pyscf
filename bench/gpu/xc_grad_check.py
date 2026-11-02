"""Exactness check of a gpu4pyscf tree against stored results (node1): the gradient's XC term and the XC matrix
from the orbitals (NOTES part 19).
With --save, it runs the SCF (conv_tol 1e-10, as the earlier parts' checks) and the gradient, and stores the
orbitals, energy, cycle count and gradient. Tighter tolerances reach the GPU J/K's rounding noise: at conv_tol
1e-12 and conv_tol_grad 1e-8 the unmodified tree ended two runs at 43 and 50 (max_cycle) cycles, 9e-10 Eh apart.
At the orbitals it also stores the gradient's XC term (grad.rks/uks.get_exc) and nr_rks/nr_uks's electron count,
XC energy and XC matrix.
With --ref, the tree under test computes the same at the stored orbitals, then runs its own SCF and gradient, and
prints the largest differences.
Usage: xc_grad_check.py --xc pbe [--basis def2-svp] [--mol caffeine] [--charge 0 --spin 0] (--save F | --ref F)
A nonzero spin, or --uks, runs UKS."""
import argparse
import os
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import numpy  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft  # noqa: E402
from gpu4pyscf.grad import rks as rks_grad, uks as uks_grad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xc', default='pbe')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--uks', action='store_true')
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--save', default=None)
g.add_argument('--ref', default=None)
args = ap.parse_args()
uks = args.uks or args.spin != 0

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, charge=args.charge,
            spin=args.spin, verbose=0)


def new_mf():
    mf = (dft.UKS if uks else dft.RKS)(mol, xc=args.xc)
    mf.conv_tol = 1e-10
    return mf


def at_orbitals(mf, mo_coeff, mo_occ):
    '''The gradient's XC term and the XC energy and matrix at the given orbitals'''
    dm = mf.make_rdm1(cupy.asarray(mo_coeff), cupy.asarray(mo_occ))
    ni = mf._numint
    exc1 = (uks_grad if uks else rks_grad).get_exc(ni, mol, mf.grids, mf.xc, dm)[1]
    nelec, excsum, vxc = (ni.nr_uks if uks else ni.nr_rks)(mol, mf.grids, mf.xc, dm)
    return {'exc1': cupy.asnumpy(exc1), 'nelec': numpy.asarray(cupy.asnumpy(nelec), dtype=float),
            'excsum': numpy.asarray(cupy.asnumpy(excsum), dtype=float), 'vxc': cupy.asnumpy(vxc)}


def scf_and_grad(mf):
    t0 = time.perf_counter()
    mf.kernel()
    de = mf.nuc_grad_method().kernel()
    return {'e_tot': numpy.asarray(mf.e_tot), 'cycles': numpy.asarray(mf.cycles), 'de': numpy.asarray(de),
            'mo_coeff': cupy.asnumpy(mf.mo_coeff), 'mo_occ': cupy.asnumpy(mf.mo_occ)}, time.perf_counter() - t0


case = f'{args.mol} {args.basis} {args.xc} {"UKS" if uks else "RKS"} (charge {args.charge}, spin {args.spin})'
tree = os.path.dirname(gpu4pyscf.__file__)
if args.save:
    mf = new_mf()
    res, t = scf_and_grad(mf)
    res.update(at_orbitals(mf, res['mo_coeff'], res['mo_occ']))
    numpy.savez(args.save, **res)
    print(f'{tree}: {case}: saved {args.save}; E {float(res["e_tot"]):.12f}, {int(res["cycles"])} cycles, '
          f'SCF and gradient {t:.1f} s')
else:
    ref = numpy.load(args.ref)
    mf = new_mf()
    mf.grids.build()
    got = at_orbitals(mf, ref['mo_coeff'], ref['mo_occ'])

    def diff(k):
        return float(numpy.abs(got[k] - ref[k]).max()), float(numpy.abs(ref[k]).max())

    print(f'{tree}: {case}, against {args.ref}')
    print('  at the stored orbitals: ' + ', '.join(f'{k} max|d| {diff(k)[0]:.2e} (max {diff(k)[1]:.1e})'
                                                     for k in ('exc1', 'vxc', 'excsum', 'nelec')))
    mf = new_mf()
    res, t = scf_and_grad(mf)
    print(f'  own SCF and gradient ({t:.1f} s): dE {float(res["e_tot"] - ref["e_tot"]):.2e}, cycles '
          f'{int(res["cycles"])} against {int(ref["cycles"])}, gradient max|d| '
          f'{float(numpy.abs(res["de"] - ref["de"]).max()):.2e} (max {float(numpy.abs(ref["de"]).max()):.1e})')
