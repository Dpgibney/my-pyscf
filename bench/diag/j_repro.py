'''Is the 1e-12 vind_mo-vs-vind difference on caffeine/TZVP the J build? get_j of the same 72
CPHF-type dms several times (semi-direct J, It9) in the tree under test.'''
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from molecules import CAFFEINE
from pyscf import gto, dft, lib
mol = gto.M(atom=CAFFEINE, basis='def2-tzvp', verbose=0, max_memory=20000)
mf = dft.RKS(mol, xc='LDA,VWN')
mf.conv_tol = 1e-8
mf.kernel()
mo, occ = mf.mo_coeff, mf.mo_occ
mocc = mo[:, occ > 0]
x = np.random.default_rng(1).standard_normal((3 * mol.natm, mo.shape[1], mocc.shape[1])) * 1e-2
dm1 = lib.einsum('pa,kai,qi->kpq', mo, x, mocc)
dm1 = dm1 + dm1.transpose(0, 2, 1)
js = [mf.get_j(mol, dm1, hermi=1) for _ in range(2)]
for i in (1,):
    print(f'get_j call {i+1} vs call 1: max|diff|/max|J| {abs(js[i] - js[0]).max() / abs(js[0]).max():.1e}')
vresp = mf.gen_response(mo, occ, hermi=1)
v = [vresp(dm1) for _ in range(2)]
print(f'vind call 2 vs call 1: max|diff|/max|v| {abs(v[1] - v[0]).max() / abs(v[0]).max():.1e}')
# as lowrank_fxc_check.py compares vind_mo with vind: projected on the orbitals
vmo = [lib.einsum('pa,kpq,qi->kai', mo, a, mocc) for a in v]
print(f'vind call 2 vs call 1 in the MO basis: max|diff|/max|v| '
      f'{abs(vmo[1] - vmo[0]).max() / abs(vmo[0]).max():.1e}; max|C| {abs(mo).max():.1f}')
