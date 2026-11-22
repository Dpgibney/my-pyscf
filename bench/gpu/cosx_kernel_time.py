'''Time of the seminumerical-exchange kernels of gpu4pyscf.sgx by class of shell pairs.

The density on the grid F = w D X of the initial-guess density on PySCF's SGX grid (ORCA-like, the
level given), every pair on every block (no screening by the density), each class timed between
device synchronizations, in double and single precision, for the operator of the functional's
exchange (full range, or the mix of a range-separated hybrid).

    python bench/gpu/cosx_kernel_time.py [--mol caffeine] [--basis def2-tzvp] [--level 1] [--mixed]
'''
import argparse
import os
import sys
import time
import numpy as np
import cupy as cp
import pyscf
from pyscf.sgx import sgx_jk as cpu_sgx_jk
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.sgx import sgx_jk, kernels

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import molecules

p = argparse.ArgumentParser()
p.add_argument('--mol', default='caffeine')
p.add_argument('--charge', type=int, default=0)
p.add_argument('--basis', default='def2-tzvp')
p.add_argument('--level', type=int, default=1)
p.add_argument('--mixed', action='store_true', help='0.15/r + 0.85 erf(0.3 r)/r (wB97M-V)')
p.add_argument('--blksize', type=int, default=sgx_jk.BLKSIZE)
p.add_argument('--repeat', type=int, default=2)
args = p.parse_args()

atom = {'caffeine': molecules.CAFFEINE, 'benzene': molecules.BENZENE,
        'water': molecules.WATER}.get(args.mol, args.mol)
mol = pyscf.M(atom=atom, basis=args.basis, charge=args.charge, verbose=0)
grids = cpu_sgx_jk.get_gridss(mol, args.level, 1e-10, use_opt_grids=True)
keep = grids.weights != 0
coords = cp.asarray(grids.coords[keep])
weights = cp.asarray(grids.weights[keep])
ng = len(weights)
smol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=0.3)
nao_c = int(smol.p_ao_loc[-1])
print('%s %s: %d AOs, %d Cartesian functions of %d sorted shells; SGX level %d: %d points' % (
    args.mol, args.basis, mol.nao, nao_c, smol.nbas, args.level, ng))

mf = dft.RKS(mol, xc='b3lyp')
dm = cp.asarray(mf.get_init_guess())
X = numint.eval_ao(mol, coords, deriv=0, transpose=False)
F = dm.dot(X) * weights
Fc64 = cp.ascontiguousarray(smol.apply_C_dot(F))

c_full, c_lr, omega = (.15, .85, .3) if args.mixed else (1., 0., 0.)
classes = sgx_jk.pair_data(smol, omega, c_full, c_lr)
nblk = (ng + args.blksize - 1) // args.blksize
dummy = cp.zeros(1, dtype=np.int32)
results = {}
for dtype in ('double', 'float'):
    dev = sgx_jk._DevicePairs(classes, omega, c_full, c_lr, dtype)
    Fc = cp.ascontiguousarray(Fc64.astype(dev.real))
    xyz = cp.ascontiguousarray(coords.T.astype(dev.real))
    G = cp.zeros_like(Fc)
    rows = []
    for li, lj, npairs, pairs, ab, prims, ish, jsh, q in dev.classes:
        kern = kernels.get_kernel(li, lj, dev.mode, dtype)
        boys = kernels.boys_table(li + lj, dtype)
        best = np.inf
        for rep in range(args.repeat + 1):
            cp.cuda.get_current_stream().synchronize()
            t0 = time.perf_counter()
            kern((nblk,), (args.blksize,),
                 (G, Fc, xyz, pairs, ab, prims, dummy, dummy, boys,
                  np.int32(0), np.int32(npairs), np.int32(ng), np.int32(ng), np.int32(0)))
            cp.cuda.get_current_stream().synchronize()
            if rep > 0:  # the first call compiles
                best = min(best, time.perf_counter() - t0)
        rows.append((li, lj, npairs, len(prims), best))
    results[dtype] = rows
    print('%s, %s operator, blocks of %d points' % (dtype, dev.mode, args.blksize))
    print('  class   pairs   primitive pairs   seconds   ns per primitive pair and point')
    for li, lj, npairs, nprims, t in rows:
        print('  (%s%s) %7d %10d        %8.4f   %8.3f' % (
            'spdfg'[li], 'spdfg'[lj], npairs, nprims, t, t / (nprims * ng) * 1e9))
    print('  all    %7d %10d        %8.4f' % (
        sum(r[2] for r in rows), sum(r[3] for r in rows), sum(r[4] for r in rows)))
