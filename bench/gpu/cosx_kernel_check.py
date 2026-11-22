'''The seminumerical-exchange kernels of gpu4pyscf.sgx against PySCF's int1e_grids.

For random F[nu,g] on random points (some at and near the nuclei, some far), G[mu,g] = sum_nu
A[mu,nu](g) F[nu,g] from the kernels, through the sorted mol's transformations, against the same
contraction of mol.intor('int1e_grids'): full range, long range and the mix of a range-separated
hybrid, in double and single precision.

    python bench/gpu/cosx_kernel_check.py [--basis def2-tzvp] [--ngrids 400] [--time]
'''
import argparse
import time
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.sgx import sgx_jk, kernels

p = argparse.ArgumentParser()
p.add_argument('--basis', default='def2-tzvp')
p.add_argument('--ngrids', type=int, default=400)
p.add_argument('--omega', type=float, default=0.3)
p.add_argument('--atom', default='O 0 0 0.12; C 0.05 0.1 -1.08; H 0.3 0.94 -1.62; H -0.2 -0.94 -1.66')
args = p.parse_args()

mol = pyscf.M(atom=args.atom, basis=args.basis, verbose=0)
nao = mol.nao
rng = np.random.default_rng(7)
ng = args.ngrids
xyz = mol.atom_coords()
coords = np.vstack([
    xyz,                                              # at the nuclei
    xyz[rng.integers(0, mol.natm, ng//4)] + rng.normal(size=(ng//4, 3)) * .05,
    xyz[rng.integers(0, mol.natm, ng//2)] + rng.normal(size=(ng//2, 3)) * 1.5,
    rng.normal(size=(ng, 3)) * 12.,
])[:ng]
F = rng.normal(size=(nao, ng))
print('%s  %s  nao %d  ngrids %d' % (mol.atom, args.basis, nao, ng))

smol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=0.3)
print('sorted mol: %d shells, %d Cartesian functions, l/nprim groups %s' % (
    smol.nbas, smol.p_ao_loc[-1], [tuple(int(v) for v in x) for x in smol.uniq_l_ctr]))

A_full = mol.intor('int1e_grids', grids=coords)
with mol.with_range_coulomb(args.omega):
    A_lr = mol.intor('int1e_grids', grids=coords)

worst = 0
for label, c_full, c_lr in (('full', 1., 0.), ('lr', 0., 1.), ('mixed', .15, .85)):
    A = c_full * A_full + c_lr * A_lr
    ref = np.einsum('gmn,ng->mg', A, F)
    scale = abs(ref).max()
    classes = sgx_jk.pair_data(smol, args.omega, c_full, c_lr)
    for dtype in ('double', 'float'):
        dev = sgx_jk._DevicePairs(classes, args.omega, c_full, c_lr, dtype)
        assert dev.mode == label
        Fc = smol.apply_C_dot(cp.asarray(F)).astype(dev.real)
        t0 = time.perf_counter()
        Gc = sgx_jk.contract(dev, cp.ascontiguousarray(Fc), cp.asarray(coords.T))
        cp.cuda.get_current_stream().synchronize()
        t1 = time.perf_counter()
        G = smol.apply_CT_dot(Gc.astype(np.float64)).get()
        err = abs(G - ref).max()
        tol = 1e-10 if dtype == 'double' else 2e-4
        print('%-5s %-6s  max |G - ref| %.2e  (max |ref| %.2f)  %s   first call %.2f s' % (
            label, dtype, err, scale, 'ok' if err < tol * scale else 'FAILED', t1 - t0))
        if dtype == 'double':
            worst = max(worst, err / scale)
        # per class, for a failure
        if err >= tol * scale:
            ls = smol._bas[:,1]
            loc = smol.p_ao_loc
            for n, c in enumerate(dev.classes):
                one = sgx_jk._DevicePairs({(c[0], c[1]): classes[c[0], c[1]]}, args.omega, c_full, c_lr, dtype)
                print('   class', c[0], c[1], 'pairs', c[2])
print('largest relative error in double precision: %.2e' % worst)
