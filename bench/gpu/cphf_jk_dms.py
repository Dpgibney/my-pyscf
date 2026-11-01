"""The Hessian CPHF's J/K calls (node1): runs a caffeine Hessian (RKS, B3LYP by default) with a hook on get_jk_mo
that times each call between device synchronizations and records its number of densities. The densities of call
--call (1-based, default the first) go to OUT.npz in the original AO basis, with mo_coeff and mo_occ, for
cphf_jk_parts.py and the Nsight Compute profiles.
Usage: cphf_jk_dms.py --out OUT.npz [--call 1] [--xc b3lyp] [--basis def2-svp]"""
import argparse
import os
import sys
import time
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--call', type=int, default=1)
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.kernel()
hobj = mf.Hessian()
cls = type(hobj)
orig = cls.get_jk_mo
calls = []


def hook(self, mol, dms, mo_coeff, mo_occ, *a, **kw):
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = orig(self, mol, dms, mo_coeff, mo_occ, *a, **kw)
    cupy.cuda.Device().synchronize()
    n_dm = int(numpy.prod(dms.shape[:-2]))
    calls.append((n_dm, time.perf_counter() - t0, kw))
    if len(calls) == args.call:
        numpy.savez(args.out, dms=cupy.asnumpy(dms.reshape(-1, *dms.shape[-2:])),
                    mo_coeff=cupy.asnumpy(mo_coeff), mo_occ=cupy.asnumpy(mo_occ))
    return out


cls.get_jk_mo = hook
hobj.kernel()
print(f'{os.path.dirname(gpu4pyscf.__file__)}: caffeine {args.basis} {args.xc} Hessian, nao {mol.nao}: '
      f'{len(calls)} get_jk_mo calls, {sum(t for n, t, kw in calls):.2f} s')
for n, (n_dm, t, kw) in enumerate(calls):
    print(f'  call {n + 1}: {n_dm} densities, {t:.2f} s {kw if kw else ""}')
