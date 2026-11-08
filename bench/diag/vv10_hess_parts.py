'''The three VV10 terms of the RKS Hessian (hessian.rks._get_enlc_deriv2, _get_vnlc_deriv1, get_vnlc_resp) on
an n-alkane after a wB97M-V SCF: their times split into the C kernels (libdft VXC_vv10nlc_hessian_*), the
einsum contractions (by subscripts), eval_ao/eval_rho and the rest, and their outputs saved to compare trees.
The response is taken for 3*natm symmetric perturbed densities P_vir X_k P_occ + h.c. (random X_k, fixed seed;
P = C C^T, so the input does not depend on the orbitals' signs), as many as the CPHF passes.

Usage: vv10_hess_parts.py OUT.npz [--n 3] [--basis def2-svp] [--nlc-level 1] [--parts deriv2,deriv1,resp]
       vv10_hess_parts.py --compare A.npz B.npz'''
import argparse
import collections
import os
import sys
import time
import numpy

if sys.argv[1:2] == ['--compare']:
    a, b = numpy.load(sys.argv[2]), numpy.load(sys.argv[3])
    for k in a.files:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f'{k:12s} shapes differ {x.shape} {y.shape}')
            continue
        scale = max(abs(x).max(), 1e-300)
        print(f'{k:12s} max|d| {abs(x - y).max():.2e}  max|d|/max|x| {abs(x - y).max() / scale:.2e}  '
              f'(max|x| {scale:.3e})')
    sys.exit()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.dft import numint  # noqa: E402
from pyscf.hessian import rks as rks_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--n', type=int, default=3)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--nlc-level', type=int, default=1)
ap.add_argument('--parts', default='deriv2,deriv1,resp')
args = ap.parse_args()

T = collections.defaultdict(float)
stack = []


def timed(key, fn):
    def w(*a, **k):
        t = time.perf_counter()
        stack.append(key)
        try:
            return fn(*a, **k)
        finally:
            stack.pop()
            if not stack:  # only the outermost of nested timed calls
                T[key] += time.perf_counter() - t
    return w


einsum = numpy.einsum


def contract(subscripts, *ops, **kw):
    return timed(f'einsum {subscripts}', einsum)(subscripts, *ops, **kw)


rks_hess.contract = contract
for name in ('VXC_vv10nlc_hessian_eval_UWABCE', 'VXC_vv10nlc_hessian_eval_omega_derivative',
             'VXC_vv10nlc_hessian_eval_f_t', 'VXC_vv10nlc_hessian_eval_EUW_grid_response',
             'VXC_vv10nlc_hessian_eval_EUW_with_weight1'):
    if hasattr(rks_hess.libdft, name):
        setattr(rks_hess.libdft, name, timed(f'C {name[20:]}', getattr(rks_hess.libdft, name)))
for name in ('eval_ao', 'eval_rho'):
    setattr(rks_hess.numint, name, timed(name, getattr(numint, name)))

mol = gto.M(atom=n_alkane(args.n), basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc='wb97m-v')
mf.conv_tol = 1e-10
mf.nlcgrids.level = args.nlc_level
mf.kernel()
hobj = mf.Hessian()
mo, occ = mf.mo_coeff, mf.mo_occ
nocc = int((occ > 0).sum())
rng = numpy.random.default_rng(11)
# P_vir X_k P_occ + h.c. with the projectors C C^T (independent of the orbitals' signs) and random X_k
x = rng.standard_normal((3 * mol.natm, mol.nao, mol.nao)) * .1
p_occ = mo[:, :nocc] @ mo[:, :nocc].T
p_vir = mo[:, nocc:] @ mo[:, nocc:].T
dm1 = lib.einsum('pq,kqr,rs->kps', p_vir, x, p_occ)
dm1 = dm1 + dm1.transpose(0, 2, 1)
max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
print(f'n-C{args.n}H{2 * args.n + 2}/{args.basis}: nao {mol.nao}, natm {mol.natm}, VV10 grid '
      f'{mf.nlcgrids.weights.size} (level {args.nlc_level}), {lib.num_threads()} threads, '
      f'max_memory {max_memory:.0f} MB; E {mf.e_tot:.10f}', flush=True)

out = {'e_tot': numpy.array([mf.e_tot])}
calls = {'deriv2': lambda: rks_hess._get_enlc_deriv2(hobj, mo, occ, max_memory),
         'deriv1': lambda: rks_hess._get_vnlc_deriv1(hobj, mo, occ, max_memory),
         'resp': lambda: rks_hess.get_vnlc_resp(mf, mol, mo, occ, dm1, max_memory)}
for part in args.parts.split(','):
    T.clear()
    t = time.perf_counter()
    out[part] = calls[part]()
    total = time.perf_counter() - t
    print(f'{part}: {total:.2f} s')
    for k, v in sorted(T.items(), key=lambda kv: -kv[1])[:12]:
        print(f'    {v:8.2f} s  {k}')
    print(f'    {total - sum(T.values()):8.2f} s  the rest', flush=True)
numpy.savez(args.out, **out)
