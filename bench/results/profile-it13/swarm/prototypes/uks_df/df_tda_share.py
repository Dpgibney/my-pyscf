'''Benzene/def2-SVP DF-B3LYP TDA (5 states), 1 thread: number of response dms that reach
df_jk.get_jk and the time spent there, against the whole TDA. Also the same for UKS (benzene+).'''
import sys, time
import numpy
from pyscf import gto, dft, lib
from pyscf.df import df_jk

BENZENE = '''C 0.000 1.396 0.000; C 1.209 0.698 0.000; C 1.209 -0.698 0.000;
C 0.000 -1.396 0.000; C -1.209 -0.698 0.000; C -1.209 0.698 0.000;
H 0.000 2.479 0.000; H 2.147 1.240 0.000; H 2.147 -1.240 0.000;
H 0.000 -2.479 0.000; H -2.147 -1.240 0.000; H -2.147 1.240 0.000'''

stats = {'calls': 0, 'dms': 0, 't': 0.}
orig = df_jk.get_jk
def wrapped(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13):
    t0 = time.perf_counter()
    r = orig(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)
    if getattr(dm, 'mo_coeff', None) is None and with_k:
        stats['calls'] += 1
        stats['dms'] += numpy.asarray(dm).reshape(-1, dm.shape[-1], dm.shape[-1]).shape[0]
        stats['t'] += time.perf_counter() - t0
    return r
df_jk.get_jk = wrapped

for charge, spin in ((0, 0), (1, 1)):
    mol = gto.M(atom=BENZENE, basis='def2-svp', charge=charge, spin=spin, verbose=0)
    mf = (dft.RKS(mol, xc='b3lyp') if spin == 0 else dft.UKS(mol, xc='b3lyp')).density_fit()
    mf.kernel()
    for k in stats: stats[k] = 0
    td = mf.TDA(); td.nstates = 5
    t0 = time.perf_counter(); td.kernel(); t = time.perf_counter() - t0
    print(f'{"RKS" if spin == 0 else "UKS"} benzene DF-B3LYP TDA: {t:.1f} s; response get_jk {stats["calls"]} calls, '
          f'{stats["dms"]} dms, {stats["t"]:.1f} s ({stats["t"]/t:.0%}); e = {td.e[:3]}')
