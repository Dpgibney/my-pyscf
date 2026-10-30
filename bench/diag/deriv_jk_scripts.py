"""Marginal cost of each contraction script in the Hessian's derivative J/K calls: the call with its
first 1, 2, ... scripts (min of 3), at 1 thread. Usage: deriv_jk_scripts.py SYSTEM ATOM"""
import os
import sys
import time
import numpy
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import BENZENE, CAFFEINE  # noqa: E402
from pyscf import gto, scf  # noqa: E402
from pyscf.hessian.rhf import _get_jk  # noqa: E402

system, ia = sys.argv[1], int(sys.argv[2])
mol = gto.M(atom={'benzene': BENZENE, 'caffeine': CAFFEINE}[system], basis='def2-svp', verbose=0)
nbas = mol.nbas
dm = scf.RHF(mol).get_init_guess(key='minao')
dm = (dm + dm.T) * .5
shl0, shl1, p0, p1 = mol.aoslice_by_atom()[ia]
ksh1 = shl1

sets = {
    'ip1ip2 s1': ('int2e_ip1ip2', 9, 's1', ['ji->s1kl', dm[:, p0:p1], 'li->s1kj', dm[:, p0:p1],
                                            'lj->s1ki', dm], (shl0, shl1, 0, nbas, 0, ksh1, 0, nbas)),
    'ipvip1 s2kl': ('int2e_ipvip1', 9, 's2kl', ['lk->s1ij', dm, 'li->s1kj', dm[:, p0:p1]],
                    (shl0, shl1, 0, ksh1, 0, nbas, 0, nbas)),
    'ip1 s2kl (make_h1)': ('int2e_ip1', 3, 's2kl', ['ji->s2kl', dm[:, p0:p1], 'lk->s1ij', dm,
                                                    'li->s1kj', dm[:, p0:p1], 'jk->s1il', dm],
                           (shl0, shl1) + (0, nbas) * 3),
}
for name, (intor, comp, aosym, scripts, shls_slice) in sets.items():
    prev = 0
    for n in range(1, len(scripts) // 2 + 1):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            _get_jk(mol, intor, comp, aosym, scripts[:2 * n], shls_slice=shls_slice)
            ts.append(time.perf_counter() - t0)
        t = min(ts)
        print(f'{name:20s} +{scripts[2 * n - 2]:9s} {t:.3f} s  (+{t - prev:.3f})', flush=True)
        prev = t
