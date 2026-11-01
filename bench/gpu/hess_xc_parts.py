"""Where the RKS Hessian's XC derivative terms spend their time (node1): _get_vxc_diag and _get_vxc_deriv2 (the two
halves of _get_exc_deriv2) and _get_vxc_deriv1 (make_h1), each run once at the converged SCF, with synchronizing
timers around the helpers they call: contract by its subscripts, the AO values (each step of ni.block_loop),
eval_rho2, eval_xc_eff, _scale_ao, _make_dR_rho1, _contract_rho1_fxc, the gradient helpers and add_sparse. Only the
outermost timed call counts, so the parts add up to the timed total; the rest is untimed Python and small arrays.
Synchronizing after every call removes the overlap of launches, so the totals run somewhat above an untimed call.
Source scripts/node1-gpu-env.sh perf|base first.
Usage: hess_xc_parts.py [--xc b3lyp] [--basis def2-svp]"""
import argparse
import os
import sys
import time
from collections import defaultdict
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, '..'))
from molecules import CAFFEINE  # noqa: E402
import cupy  # noqa: E402
from pyscf import gto  # noqa: E402
import gpu4pyscf  # noqa: E402
from gpu4pyscf import dft  # noqa: E402
from gpu4pyscf.dft import numint  # noqa: E402
from gpu4pyscf.grad import rks as rks_grad  # noqa: E402
from gpu4pyscf.hessian import rks as rks_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--basis', default='def2-svp')
args = ap.parse_args()

T, N = defaultdict(float), defaultdict(int)
depth = [0]


def timed(label, f, by_subscripts=False):
    def wrapper(*a, **kw):
        if depth[0]:
            return f(*a, **kw)
        key = f'contract {a[0]}' if by_subscripts else label
        depth[0] += 1
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        try:
            return f(*a, **kw)
        finally:
            cupy.cuda.Device().synchronize()
            T[key] += time.perf_counter() - t0
            N[key] += 1
            depth[0] -= 1
    return wrapper


def timed_generator(label, f):
    def wrapper(*a, **kw):
        it = f(*a, **kw)
        while True:
            cupy.cuda.Device().synchronize()
            t0 = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                return
            cupy.cuda.Device().synchronize()
            T[label] += time.perf_counter() - t0
            N[label] += 1
            yield item
    return wrapper


mol = gto.M(atom=CAFFEINE, basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
mf.conv_tol = 1e-10
mf.kernel()
hobj = mf.Hessian()
mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ

rks_hess.contract = timed('contract', rks_hess.contract, by_subscripts=True)
for mod, names in ((rks_hess, ('add_sparse', 'take_last2d', '_make_dR_rho1', '_d1d2_dot_', '_contract_rho1_fxc',
                              'dot_rows', '_block_ao_subset', '_d1_dot_pair')),
                   (numint, ('eval_rho2', '_scale_ao')),
                   (rks_grad, ('_make_dR_dao_w', '_gga_grad_sum_', '_d1_dot_', '_d1d2_dot_'))):
    for name in names:
        if hasattr(mod, name):
            setattr(mod, name, timed(f'{mod.__name__.split(".")[-1]}.{name}', getattr(mod, name)))
numint.NumInt.eval_xc_eff = timed('eval_xc_eff', numint.NumInt.eval_xc_eff)
numint.NumInt.block_loop = timed_generator('AO values (block_loop)', numint.NumInt.block_loop)

print(f'{os.path.dirname(gpu4pyscf.__file__)}: caffeine {args.basis} {args.xc} ({mf._numint._xc_type(mf.xc)}), '
      f'nao {mol.nao}, nocc {int((mo_occ > 0).sum())}, {mf.grids.coords.shape[0]} grid points')
for name in ('_get_vxc_diag', '_get_vxc_deriv2', '_get_vxc_deriv1'):
    f = getattr(rks_hess, name)
    f(hobj, mo_coeff, mo_occ, 4000)   # first call: kernels, memory pool
    T.clear()
    N.clear()
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    f(hobj, mo_coeff, mo_occ, 4000)
    cupy.cuda.Device().synchronize()
    total = time.perf_counter() - t0
    print(f'  {name}: {total:.2f} s, timed parts {sum(T.values()):.2f} s')
    for k, v in sorted(T.items(), key=lambda kv: -kv[1])[:14]:
        print(f'    {v:7.2f} s {100 * v / total:5.1f}%  {k} ({N[k]} calls)')
