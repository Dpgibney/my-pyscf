"""Where the Hessian CPHF's XC response (hessian.rks.nr_rks_fxc_mo) spends its time (node1): one call for NSET
response densities (random mo1 of the CPHF's shape; the cost does not depend on the values), with the steps of
_nr_rks_fxc_mo_task timed between device synchronizations: the AO values (ni.block_loop), rho1 (eval_rho4), the fxc
contraction, and the per-density AO integration (_scale_ao, the GEMM, add_sparse), on mf.cphf_grids as the
Hessian uses. Source scripts/node1-gpu-env.sh perf|base first.
LDA and GGA functionals only.
Usage: fxc_mo_parts.py [--xc b3lyp] [--basis def2-svp] [--nset 72]"""
import argparse
import os
import sys
import time
from collections import defaultdict
import numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft  # noqa: E402
from gpu4pyscf.dft import numint  # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract, add_sparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--nset', type=int, default=72)
args = ap.parse_args()

mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.kernel()
ni, grids = mf._numint, mf.cphf_grids   # the Hessian's CPHF grid (hessian.rks.get_veff_resp_mo)
if grids.coords is None:
    grids.build(mol=mol, with_non0tab=False, sort_grids=True)
mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
mocc = mo_coeff[:, mo_occ > 0]
nao, nmo = mo_coeff.shape
nocc = mocc.shape[1]
rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc, mo_coeff, mo_occ, 0)
opt = ni.gdftopt
_sorted_mol = opt.mol
mo1 = cupy.asarray(numpy.random.default_rng(3).standard_normal((args.nset, nmo, nocc))) * 1e-2
mo1_ao = contract('npo,ip->nio', mo1, mo_coeff)
mo1_s = opt.sort_orbitals(mo1_ao, axis=[1])
mocc_s = opt.sort_orbitals(mocc, axis=[0])
xctype = ni._xc_type(mf.xc)
assert xctype in ('LDA', 'GGA'), 'LDA and GGA functionals only'
ao_deriv = 0 if xctype == 'LDA' else 1

T = defaultdict(float)


def tick(label, t0):
    cupy.cuda.Device().synchronize()
    t1 = time.perf_counter()
    T[label] += t1 - t0
    return t1


vmat = cupy.zeros((args.nset, nao, nao))
cupy.cuda.Device().synchronize()
t_start = t = time.perf_counter()
p0 = p1 = 0
loop = ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory=None, blksize=None)
nblk = 0
while True:
    try:
        ao, mask, weights, coords = next(loop)
    except StopIteration:
        break
    t = tick('AO values (block_loop)', t)
    nblk += 1
    p0, p1 = p1, p1 + len(weights)
    rho1 = numint.eval_rho4(_sorted_mol, ao, 2.0 * mocc_s[mask], mo1_s[:, mask], xctype=xctype, hermi=1)
    t = tick('rho1 (eval_rho4)', t)
    fxc_w = cupy.multiply(fxc[:, :, p0:p1], weights)
    wv = contract('axg,xyg->ayg', rho1, fxc_w)
    t = tick('fxc contraction', t)
    for i in range(args.nset):
        if xctype == 'LDA':
            aow = numint._scale_ao(ao, wv[i][0])
            add_sparse(vmat[i], ao.dot(aow.T), mask)
        else:
            wv[i, 0] *= .5
            aow = numint._scale_ao(ao, wv[i])
            add_sparse(vmat[i], ao[0].dot(aow.T), mask)
    t = tick('per-density AO integration', t)
total = time.perf_counter() - t_start
print(f'{os.path.dirname(gpu4pyscf.__file__)}: caffeine {args.basis} {args.xc} ({xctype}), nao {nao}, nocc {nocc}, '
      f'{grids.coords.shape[0]} grid points in {nblk} blocks, {args.nset} densities: {total:.2f} s')
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f'  {v:7.2f} s {100 * v / total:5.1f}%  {k}')
