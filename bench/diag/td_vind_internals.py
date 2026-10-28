"""Record what the caffeine/B3LYP TDA operator does for its initial guess: every get_jk call (input dms,
outputs, whether ERIs were held in memory) and the XC response, to compare two builds.
Usage: td_vind_internals.py OUT.npz | --compare A.npz B.npz   (run under bench/diag/run_in.py)"""
import sys

import numpy as np

if sys.argv[1] == '--compare':
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for key in a.files:
        if a[key].dtype == bool or a[key].ndim == 0:
            print(f'{key:14s} {a[key]} vs {b[key]}')
        else:
            print(f'{key:14s} max|d| {abs(a[key] - b[key]).max():.2e}  max|a| {abs(a[key]).max():.2e}')
    sys.exit()

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from cases import converged_ks
from molecules import CAFFEINE
from pyscf import lib

mf = converged_ks(CAFFEINE, 'def2-svp', 'B3LYP', tight=True)
mol = mf.mol
td = mf.TDA()
td.nstates = 5
x0 = np.asarray(td.get_init_guess(mf, 5))
out = {'x0': x0}
get_jk = mf.get_jk


def logged_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
    out['jk_dm'] = np.array(dm)
    out['jk_incore_before'] = np.array(mf._eri is not None)
    out['rss_before'] = np.array(lib.current_memory()[0])
    vj, vk = get_jk(mol, dm, hermi, with_j, with_k, omega)
    out['jk_incore_after'] = np.array(mf._eri is not None)
    out['vj'], out['vk'] = np.array(vj), np.array(vk)
    return vj, vk


mf.get_jk = logged_jk
ni = mf._numint
fxc_st = ni.nr_rks_fxc_st


def logged_fxc(*args, **kwargs):
    v = fxc_st(*args, **kwargs)
    out['vxc1'] = np.array(v)
    return v


ni.nr_rks_fxc_st = logged_fxc
vind, hdiag = td.gen_vind(mf)
out['vind'] = vind(x0)
np.savez(sys.argv[1], **out)
print('saved', sys.argv[1], 'max_memory', mf.max_memory)
