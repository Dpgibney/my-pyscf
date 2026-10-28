"""Dump the parts of the caffeine/B3LYP TDA operator for its initial guess vectors (SCF data, J, K, XC
response, the full operator), to compare two builds with --compare.
Usage: td_operator_dump.py OUT.npz | --compare A.npz B.npz   (run under bench/diag/run_in.py)"""
import sys

import numpy as np

if sys.argv[1] == '--compare':
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for key in a.files:
        d = abs(a[key] - b[key]).max()
        print(f'{key:12s} max|d| {d:.2e}  max|a| {abs(a[key]).max():.2e}')
    sys.exit()

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from cases import converged_ks
from molecules import CAFFEINE
from pyscf import lib
from pyscf.scf import _vhf

mf = converged_ks(CAFFEINE, 'def2-svp', 'B3LYP', tight=True)
mol = mf.mol
td = mf.TDA()
td.nstates = 5
mo, occ = mf.mo_coeff, mf.mo_occ
orbo, orbv = mo[:, occ > 0], mo[:, occ == 0]
x0 = np.asarray(td.get_init_guess(mf, 5))
x = x0.reshape(5, orbo.shape[1], orbv.shape[1])
dms = np.einsum('xov,po,qv->xpq', x, orbo, orbv)
out = {'e_tot': np.array(mf.e_tot), 'mo_energy': mf.mo_energy, 'x0': x0}
out['vj'], out['vk'] = mf.get_jk(mol, dms, hermi=0)
ni = mf._numint
rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo, occ, 0)
out['vxc1'] = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms, 0, 0, rho0, vxc, fxc)
vind, hdiag = td.gen_vind(mf)
out['vind'] = vind(x0)
out['hdiag'] = hdiag
out['incore'] = np.array(mf._eri is not None)
out['mem'] = np.array(lib.current_memory()[0])
np.savez(sys.argv[1], **out)
print('saved', sys.argv[1], 'incore', mf._eri is not None, 'rss MB', lib.current_memory()[0])
