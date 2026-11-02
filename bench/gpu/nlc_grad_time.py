"""Time of the gradient's NLC (VV10) terms, at the orbitals of a cheaper SCF (node1): runs the SCF with --orbitals-xc,
then times rks_grad.get_nlc_exc (twice; the second call is reported) and get_nlc_exc_full_response for the VV10 part
of --xc, on --xc's NLC grid. The cost of these terms does not depend on which orbitals they get, and this avoids the
SCF with VV10 (512 s for caffeine/def2-SVP B97M-V on the GTX 1650 Super, 93% of it in VV10).
Usage: nlc_grad_time.py [--xc b97m_v] [--orbitals-xc pbe] [--basis def2-svp] [--mol caffeine|benzene]"""
import argparse
import os
import sys
import time
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf.dft import rks  # noqa: E402
from gpu4pyscf.grad import rks as rks_grad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xc', default='b97m_v')
ap.add_argument('--orbitals-xc', default='pbe')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'benzene'])
args = ap.parse_args()

mol = gto.M(atom={'caffeine': CAFFEINE, 'benzene': BENZENE}[args.mol], basis=args.basis, verbose=0)
guess = rks.RKS(mol, xc=args.orbitals_xc)
guess.conv_tol = 1e-10
guess.kernel()
mf = rks.RKS(mol, xc=args.xc)
mf.nlcgrids.build()
dm = mf.make_rdm1(guess.mo_coeff, guess.mo_occ)
ni = mf._numint
xc = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc


def timed(fn, *a):
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    fn(*a)
    cupy.cuda.Device().synchronize()
    return time.perf_counter() - t0


timed(rks_grad.get_nlc_exc, ni, mol, mf.nlcgrids, xc, dm)
t1 = timed(rks_grad.get_nlc_exc, ni, mol, mf.nlcgrids, xc, dm)
t2 = timed(rks_grad.get_nlc_exc_full_response, ni, mol, mf.nlcgrids, xc, dm)
print(f'{os.path.dirname(gpu4pyscf.__file__)}: {args.mol} {args.basis} {args.xc} at {args.orbitals_xc} orbitals, '
      f'{mf.nlcgrids.coords.shape[0]} NLC grid points: get_nlc_exc {t1:.2f} s, get_nlc_exc_full_response {t2:.2f} s')
