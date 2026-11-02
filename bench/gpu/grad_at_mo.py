"""The nuclear gradient at fixed orbitals (node1). --save PREFIX runs the SCF and saves its orbitals (mo_coeff, mo_occ,
mo_energy) and gradient; --load PREFIX runs the SCF (for its integrals, grids and density fitting), then the gradient at
the saved orbitals, and prints its largest difference from the saved gradient. With two trees' gradients at the same
orbitals, what remains is the gradient code's own difference, without that of the two SCFs' final densities (a
gradient is not variational, so those enter at first order).
Usage: grad_at_mo.py [--mol caffeine] [--basis def2-svp] [--xc b3lyp] [--df] (--save PREFIX | --load PREFIX)"""
import argparse
import os
import sys
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
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--df', action='store_true')
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--save')
g.add_argument('--load')
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
mf = rks.RKS(mol, xc=args.xc)
if args.df:
    mf = mf.density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
e = mf.kernel()
assert mf.converged
tree = os.path.dirname(gpu4pyscf.__file__)
if args.save:
    grad = cupy.asnumpy(cupy.asarray(mf.Gradients().kernel()))
    numpy.savez(args.save, mo_coeff=cupy.asnumpy(mf.mo_coeff), mo_occ=cupy.asnumpy(mf.mo_occ),
                mo_energy=cupy.asnumpy(mf.mo_energy), grad=grad, e=e)
    print(f'{tree}: E {e:.12f}, gradient saved (max |g| {abs(grad).max():.3e})')
else:
    ref = numpy.load(args.load if args.load.endswith('.npz') else args.load + '.npz')
    own = cupy.asnumpy(cupy.asarray(mf.Gradients().kernel()))
    grad = cupy.asnumpy(cupy.asarray(mf.Gradients().kernel(
        mo_energy=cupy.asarray(ref['mo_energy']), mo_coeff=cupy.asarray(ref['mo_coeff']),
        mo_occ=cupy.asarray(ref['mo_occ']))))
    print(f'{tree}: E {e:.12f} ({e - float(ref["e"]):+.1e} from the saved SCF); gradient against the saved one: '
          f'at its own orbitals {abs(own - ref["grad"]).max():.2e}, at the saved orbitals '
          f'{abs(grad - ref["grad"]).max():.2e} (max |g| {abs(ref["grad"]).max():.3e})')
