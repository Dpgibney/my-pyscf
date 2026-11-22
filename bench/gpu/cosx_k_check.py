'''The seminumerical exchange matrix of gpu4pyscf.sgx: the driver against a dense reference, PySCF's
sgx on the CPU, and the exact K.

1. K of the driver against the same formula in numpy with mol.intor('int1e_grids') on the driver's
   grid and overlap fit (rounding).
2. Against PySCF's sgx (dfj, the same grid level, ORCA-like grids, overlap fit; its screening).
3. The error of K and of the exchange energy against the exact K, by grid level, in double and
   single precision, for the full-range operator and the mix of wB97M-V.

    python bench/gpu/cosx_k_check.py [--basis def2-tzvp]
'''
import argparse
import numpy as np
import cupy as cp
import pyscf
from pyscf import scf as cpu_scf
from pyscf import sgx as cpu_sgx
from gpu4pyscf import scf, dft
from gpu4pyscf.scf import jk
from gpu4pyscf.sgx import sgx

p = argparse.ArgumentParser()
p.add_argument('--basis', default='def2-tzvp')
p.add_argument('--atom', default='O 0 0 0.12; C 0.05 0.1 -1.08; H 0.3 0.94 -1.62; H -0.2 -0.94 -1.66')
args = p.parse_args()
mol = pyscf.M(atom=args.atom, basis=args.basis, verbose=0)
nao = mol.nao
mf = dft.RKS(mol, xc='pbe0')
mf.conv_tol = 1e-10
mf.kernel()
dm = cp.asarray(mf.make_rdm1())
dm_cpu = dm.get()
print('%s  %s  nao %d' % (args.atom, args.basis, nao))

# 1. the dense reference
obj = sgx.SGX(mol)
obj.build()
coords = obj._coords.T.get()
weights = obj._weights.get()
proj = obj._proj.get()
ao = mol.eval_gto('GTOval', coords)   # [g, nao]
for label, omega, lr, sr in (('full', None, None, None), ('mixed', .3, 1., .15)):
    A = mol.intor('int1e_grids', grids=coords)
    if omega:
        with mol.with_range_coulomb(omega):
            A = sr * A + (lr - sr) * mol.intor('int1e_grids', grids=coords)
    fg = ao.dot(proj.dot(dm_cpu)) * weights[:,None]     # [g, nu]
    gv = np.einsum('gmn,gn->gm', A, fg)
    ref = proj.T.dot(ao.T.dot(gv))
    ref = (ref + ref.T) * .5
    vk = obj.get_k(dm, 1, omega, lr, sr).get()
    print('1. %-5s operator, %d points: max |K - dense reference| %.2e  (max |K| %.3f)' % (
        label, len(weights), abs(vk - ref).max(), abs(ref).max()))
A = None

# 2. PySCF's sgx
cmf = cpu_sgx.sgx_fit(cpu_scf.RHF(mol), pjs=True)
cmf.with_df.grids_level_i = cmf.with_df.grids_level_f = 1
cmf.with_df.use_opt_grids = True
cmf.with_df.build()
vk_cpu = cmf.with_df.get_jk(dm_cpu, with_j=False)[1]
vk = obj.get_k(dm).get()
print('2. PySCF %s sgx, level 1 (%d points with padding): max |K - K_pyscf| %.2e' % (
    pyscf.__version__, cmf.with_df.grids.weights.size, abs(vk - vk_cpu).max()))

# 3. against the exact K
for label, omega, lr, sr in (('full', None, None, None), ('mixed', .3, 1., .15)):
    with mol.with_range_coulomb(omega):
        vk_exact = None
    vk_exact = mf.get_k(mol, dm, 1, omega, lr, sr)
    ek = -.25 * float(cp.einsum('ij,ji', dm, vk_exact))
    print('3. %s operator: exact exchange energy %.10f' % (label, ek))
    for level in (0, 1, 2, 3):
        for dtype in ('double', 'float'):
            o = sgx.SGX(mol)
            o.grids_level = level
            o.kernel_dtype = dtype
            vk = o.get_k(dm, 1, omega, lr, sr)
            e = -.25 * float(cp.einsum('ij,ji', dm, vk))
            print('   level %d %-6s %6d points: max |K - K_exact| %.2e   exchange energy error %+.2e' % (
                level, dtype, o.size, float(abs(vk - vk_exact).max()), e - ek))
