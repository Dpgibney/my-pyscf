"""The TDA trial densities of caffeine/B3LYP through direct J/K: per-dm kernels (CVHFnr_direct_drv) vs
batched kernels, both against J/K without screening, J and K separately.
Usage: td_jk_operator_diff.py [--guess]   (run under bench/diag/run_in.py)"""
import sys

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from cases import converged_ks
from molecules import CAFFEINE
from pyscf import lib, scf
from pyscf.scf import _vhf

mf = converged_ks(CAFFEINE, 'def2-svp', 'B3LYP', tight=True)
mol = mf.mol
td = mf.TDA()
td.nstates = 5
mo, occ = mf.mo_coeff, mf.mo_occ
orbo, orbv = mo[:, occ > 0], mo[:, occ == 0]
if '--guess' in sys.argv:   # the initial unit-vector guesses of the Davidson solver
    x = np.asarray(td.get_init_guess(mf, 5)).reshape(5, orbo.shape[1], orbv.shape[1])
else:                       # the converged excitation vectors
    td.kernel()
    x = np.array([xy[0] for xy in td.xy])                 # (5, nocc, nvir)
dms = np.einsum('xov,po,qv->xpq', x, orbo, orbv)
vhfopt = mf._opt[None]
res = {}
with lib.temporary_env(_vhf, DIRECT_BATCH_MIN=10**9):
    res['old'] = scf.hf.get_jk(mol, dms, 0, vhfopt)
with lib.temporary_env(_vhf, DIRECT_BATCH_MIN=1):
    res['new'] = scf.hf.get_jk(mol, dms, 0, vhfopt)
    res['none'] = _vhf.direct(dms, mol._atm, mol._bas, mol._env, None, 0)
hyb = mf._numint.hybrid_coeff(mf.xc)


def ov(v):  # project on the occupied-virtual block as the TDA operator does
    return np.einsum('xpq,po,qv->xov', v, orbo, orbv)


for name in ('old', 'new'):
    vj, vk = res[name]
    vj0, vk0 = res['none']
    dj, dk = ov(vj - vj0), ov(vk - vk0)
    de = np.einsum('xov,xov->x', x, dj - .5 * hyb * dk)    # first-order change of each excitation energy
    print(f'{name}: |dJ|max {abs(vj - vj0).max():.1e} |dK|max {abs(vk - vk0).max():.1e}  '
          f'ov |dJ| {abs(dj).max():.1e} |dK| {abs(dk).max():.1e}  x.dA.x (Eh) ' +
          ' '.join(f'{v:.1e}' for v in de))
vj1, vk1 = res['new']
vj2, vk2 = res['old']
de = np.einsum('xov,xov->x', x, ov(vj1 - vj2) - .5 * hyb * ov(vk1 - vk2))
print('new - old x.dA.x (Eh): ' + ' '.join(f'{v:.1e}' for v in de))
