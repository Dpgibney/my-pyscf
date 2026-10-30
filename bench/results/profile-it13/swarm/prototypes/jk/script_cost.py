"""Marginal cost per contraction script of a Hessian ip1ip2 call (benzene C atom, 1 thread)."""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess
mol = gto.M(atom=BENZENE, basis='def2-svp', verbose=0)
dm0 = scf.RHF(mol).get_init_guess(key='minao')
shl0, shl1, p0, p1 = mol.aoslice_by_atom()[0]
sl = (shl0, shl1) + (0, mol.nbas)*3
dms = {'ji->s1kl': dm0[:,p0:p1], 'li->s1kj': dm0[:,p0:p1], 'lj->s1ki': dm0}
def t(scripts, intor='int2e_ip1ip2', aosym='s1', comp=9, rep=3):
    arg = sum([[s, dms[s]] for s in scripts], [])
    best = 9e9
    for _ in range(rep):
        t0 = time.perf_counter(); rhf_hess._get_jk(mol, intor, comp, aosym, arg, shls_slice=sl)
        best = min(best, time.perf_counter() - t0)
    return best
base = t(['ji->s1kl'])
for extra in (['li->s1kj'], ['lj->s1ki'], ['li->s1kj', 'lj->s1ki']):
    print(f"ji + {extra}: {t(['ji->s1kl'] + extra):.3f} s (base ji only {base:.3f} s)")
print(f"li only: {t(['li->s1kj']):.3f} s, lj only: {t(['lj->s1ki']):.3f} s")
