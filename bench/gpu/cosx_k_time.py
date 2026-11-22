'''Where a seminumerical K build of gpu4pyscf.sgx spends its time: the steps of SGX.get_k, each
between device synchronizations, for the initial-guess density (--scf: the converged density of
the functional), and the exact K of the same density beside it.

    python bench/gpu/cosx_k_time.py [--mol caffeine] [--basis def2-tzvp] [--level 1] [--xc b3lyp] [--dtype double]
'''
import argparse
import os
import sys
import time
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.sgx import sgx, sgx_jk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import molecules

p = argparse.ArgumentParser()
p.add_argument('--mol', default='caffeine')
p.add_argument('--charge', type=int, default=0)
p.add_argument('--basis', default='def2-tzvp')
p.add_argument('--level', type=int, default=1)
p.add_argument('--xc', default='b3lyp')
p.add_argument('--dtype', default='double')
p.add_argument('--scf', action='store_true')
p.add_argument('--two-level', action='store_true', help='converge the density with the two-level SCF')
p.add_argument('--exact', action='store_true', help='also time the exact K')
args = p.parse_args()

def sync():
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter()

atom = {'caffeine': molecules.CAFFEINE, 'benzene': molecules.BENZENE,
        'water': molecules.WATER}.get(args.mol, args.mol)
mol = pyscf.M(atom=atom, basis=args.basis, charge=args.charge, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
if args.scf:
    mf.two_level = args.two_level
    t0 = time.perf_counter()
    mf.kernel()
    print('SCF: %.1f s, %s%d cycles, converged %s' % (
        time.perf_counter() - t0, 'two-level, references ' if args.two_level else '', mf.cycles, mf.converged))
    dm = cp.asarray(mf.make_rdm1())
else:
    dm = cp.asarray(mf.get_init_guess())
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(args.xc, spin=mol.spin)

obj = sgx.SGX(mol)
obj.grids_level = args.level
obj.kernel_dtype = args.dtype
t0 = sync()
obj.build()
t1 = sync()
print('%s %s %s: %d AOs; SGX level %d: %d points; build (grid on the CPU, overlap fit) %.2f s' % (
    args.mol, args.basis, args.xc, mol.nao, args.level, obj.size, t1 - t0))
om, c_full, c_lr = sgx._operator(mol, omega, alpha, hyb)
t0 = sync()
dev = obj._device_pairs(om, c_full, c_lr)
t1 = sync()
print('operator %g/r + %g erf(%g r)/r (%s), %s; pair data %.2f s: %d pairs, %d primitive pairs' % (
    c_full, c_lr, om, dev.mode, args.dtype, t1 - t0,
    sum(c[2] for c in dev.classes), sum(len(c[5]) for c in dev.classes)))

smol = obj._sorted_mol
for rep in range(2):
    tt = {}
    def lap(key, t):
        t1 = sync()
        tt[key] = tt.get(key, 0) + t1 - t
        return t1
    t = t00 = sync()
    d = obj._proj.dot(dm)
    vk = cp.zeros_like(dm)
    real = dev.real
    for c0, c1 in obj._batches():
        # as SGX.get_k: the products with the AO values in the kernels' precision
        ao = obj._eval_ao(c0, c1).astype(real, copy=False); t = lap('AO values', t)
        f = d.astype(real).T.dot(ao); f *= obj._weights[c0:c1].astype(real); t = lap('F = w D X', t)
        fc = cp.ascontiguousarray(smol.apply_C_dot(cp.asarray(f, dtype=np.float64)), dtype=real)
        t = lap('F to the sorted Cartesians', t)
        gc = sgx_jk.contract(dev, fc, obj._coords[:,c0:c1], blksize=obj.blksize); t = lap('kernels', t)
        g = smol.apply_CT_dot(cp.asarray(gc, dtype=np.float64)); t = lap('G back', t)
        vk += ao.dot(cp.asarray(g, dtype=real).T); t = lap('K = X G', t)
    vk = obj._proj.T.dot(vk)
    vk = (vk + vk.T) * .5
    t = lap('fit and symmetrization', t)
    total = t - t00
print('K build %.3f s (second pass; %d batches):' % (total, len(obj._batches())))
for key, v in tt.items():
    print('   %-28s %7.3f s  %5.1f%%' % (key, v, 100 * v / total))
t0 = sync()
vk2 = obj.get_k(dm, 1, omega, alpha, hyb)
t1 = sync()
print('SGX.get_k %.3f s; |K - steps| %.1e' % (t1 - t0, float(abs(vk2 - vk).max())))
if args.exact:
    mf.get_k(mol, dm, 1, omega, alpha, hyb)
    t0 = sync()
    vk_exact = mf.get_k(mol, dm, 1, omega, alpha, hyb)
    t1 = sync()
    ek = -.25 * float(cp.einsum('ij,ji', dm, vk_exact))
    e = -.25 * float(cp.einsum('ij,ji', dm, vk2))
    print('exact K (full build) %.3f s; max |K - K_exact| %.2e, exchange energy %+.2e of %.6f' % (
        t1 - t0, float(abs(vk2 - vk_exact).max()), e - ek, ek))
