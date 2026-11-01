"""The Hessian CPHF's XC response (hessian.rks.nr_rks_fxc_mo) on node1, for NSET response densities (random mo1 of
the CPHF's shape; the cost does not depend on the values) on mf.cphf_grids, as the Hessian uses:
  - GPU4PySCF's algorithm (v1.9.0's _nr_rks_fxc_mo_task, copied here), with its steps timed between device
    synchronizations: the AO values (ni.block_loop), rho1 (eval_rho4), the fxc contraction, and the per-density
    AO integration (_scale_ao, the GEMM, add_sparse), then the transformation to the MO basis;
  - the library's nr_rks_fxc_mo of the tree in use, timed, and its largest difference from the copy.
Source scripts/node1-gpu-env.sh perf|base first.
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
from gpu4pyscf.hessian import rks as rks_hess  # noqa: E402
from gpu4pyscf.hessian.rhf import _ao2mo  # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract, add_sparse, tag_array, transpose_sum  # noqa: E402

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
xctype = ni._xc_type(mf.xc)
# the CPHF densities as hessian.rhf.gen_vind builds them
mo1 = cupy.asarray(numpy.random.default_rng(3).standard_normal((args.nset, nmo, nocc))) * 1e-2
mo1_mo = contract('npo,ip->nio', mo1, mo_coeff)
dm1 = contract('npi,qi->npq', mo1_mo, mocc * 2)
transpose_sum(dm1, inplace=True, hermi=1)
dm1 = tag_array(dm1, mo1=mo1_mo, occ_coeff=mocc, symmetrize=1)

T = defaultdict(float)


def tick(label, t0):
    cupy.cuda.Device().synchronize()
    t1 = time.perf_counter()
    T[label] += t1 - t0
    return t1


def old_algorithm():
    '''GPU4PySCF v1.9.0's _nr_rks_fxc_mo_task on one device, with timers'''
    opt = ni.gdftopt
    _sorted_mol = opt.mol
    mo1_s = opt.sort_orbitals(dm1.mo1, axis=[1])
    mocc_s = opt.sort_orbitals(dm1.occ_coeff, axis=[0])
    mo_s = opt.sort_orbitals(mo_coeff, axis=[0])
    ao_deriv = 0 if xctype in ('LDA', 'HF') else 1
    vmat = cupy.zeros((args.nset, nao, nao))
    cupy.cuda.Device().synchronize()
    t = time.perf_counter()
    p0 = p1 = 0
    loop = ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory=None, blksize=None)
    while True:
        try:
            ao, mask, weights, coords = next(loop)
        except StopIteration:
            break
        t = tick('AO values (block_loop)', t)
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
            elif xctype == 'GGA':
                wv[i, 0] *= .5
                aow = numint._scale_ao(ao, wv[i])
                add_sparse(vmat[i], ao[0].dot(aow.T), mask)
            else:
                wv[i, 0] *= .5
                wv[i, 4] *= .5
                vtmp = numint._tau_dot(ao, ao, wv[i, 4])
                aow = numint._scale_ao(ao[:4], wv[i, :4])
                vtmp = contract('ig, jg->ij', ao[0], aow, beta=1, out=vtmp)
                add_sparse(vmat[i], vtmp, mask)
        t = tick('per-density AO integration', t)
    if xctype != 'LDA':
        transpose_sum(vmat)
    vmat = _ao2mo(vmat, mocc_s, mo_s)
    tick('transformation to the MO basis', t)
    return vmat


def library():
    return rks_hess.nr_rks_fxc_mo(ni, mol, grids, mf.xc, None, dm1, mo_coeff, 0, 1, rho0, vxc, fxc,
                                  max_memory=None)


library()   # module loading, memory pool
cupy.cuda.Device().synchronize()
t0 = time.perf_counter()
v_old = old_algorithm()
t_old = time.perf_counter() - t0
cupy.cuda.Device().synchronize()
t0 = time.perf_counter()
v_new = library()
cupy.cuda.Device().synchronize()
t_new = time.perf_counter() - t0
print(f'{os.path.dirname(gpu4pyscf.__file__)}: caffeine {args.basis} {args.xc} ({xctype}), nao {nao}, nocc {nocc}, '
      f'{grids.coords.shape[0]} CPHF grid points, {args.nset} densities')
print(f'  v1.9.0 algorithm {t_old:.2f} s:')
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f'    {v:7.2f} s {100 * v / t_old:5.1f}%  {k}')
v_old = v_old.reshape(v_new.shape)
print(f'  nr_rks_fxc_mo of this tree {t_new:.2f} s ({t_old / t_new:.2f}x); max |new - v1.9.0 algorithm| = '
      f'{abs(v_new - v_old).max():.2e} (max |v| {abs(v_old).max():.2e})')
