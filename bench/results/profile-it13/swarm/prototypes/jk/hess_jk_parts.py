"""Time the pieces of the RKS Hessian's direct J/K at 1 thread (benzene/def2-SVP, minao-like dm).
Per call: ipip1 (all atoms), ip1ip2 / ipvip1 / ip1(make_h1) for one C and one H atom, with the
Hessian's scripts and with a single script (marginal contraction cost)."""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE
from pyscf import gto, scf, lib
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rhf as rhf_grad

system = sys.argv[1] if len(sys.argv) > 1 else 'benzene'
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE}[system]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
mf = scf.RHF(mol)
dm0 = mf.get_init_guess(key='minao')
nao = mol.nao
print(system, 'nao', nao, 'nbas', mol.nbas, 'threads', lib.num_threads())

def tm(f, rep=1):
    best = 1e9
    for _ in range(rep):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best

aoslices = mol.aoslice_by_atom()
_get_jk = rhf_hess._get_jk
_make_vhfopt = rhf_hess._make_vhfopt

opt_ipip1 = _make_vhfopt(mol, dm0, 'ipip1', 'int2e_ipip1ipip2')
t = tm(lambda: _get_jk(mol, 'int2e_ipip1', 9, 's2kl', ['lk->s1ij', dm0, 'jk->s1il', dm0], vhfopt=opt_ipip1))
t1 = tm(lambda: _get_jk(mol, 'int2e_ipip1', 9, 's2kl', ['lk->s1ij', dm0], vhfopt=opt_ipip1))
print(f'ipip1 all atoms: J+K {t:.3f} s, J only {t1:.3f} s')

ip1ip2_opt = _make_vhfopt(mol, dm0, 'ip1ip2', 'int2e_ip1ip2')
ipvip1_opt = _make_vhfopt(mol, dm0, 'ipvip1', 'int2e_ipvip1ipvip2')
tot = {}
for ia in (0, mol.natm - 1):
    shl0, shl1, p0, p1 = aoslices[ia]
    shls_slice = (shl0, shl1) + (0, mol.nbas)*3
    a = tm(lambda: _get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                           ['ji->s1kl', dm0[:,p0:p1], 'li->s1kj', dm0[:,p0:p1], 'lj->s1ki', dm0],
                           shls_slice=shls_slice, vhfopt=ip1ip2_opt))
    a1 = tm(lambda: _get_jk(mol, 'int2e_ip1ip2', 9, 's1', ['ji->s1kl', dm0[:,p0:p1]],
                            shls_slice=shls_slice, vhfopt=ip1ip2_opt))
    b = tm(lambda: _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', ['lk->s1ij', dm0, 'li->s1kj', dm0[:,p0:p1]],
                           shls_slice=shls_slice, vhfopt=ipvip1_opt))
    b1 = tm(lambda: _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', ['lk->s1ij', dm0],
                            shls_slice=shls_slice, vhfopt=ipvip1_opt))
    c = tm(lambda: _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                           ['ji->s2kl', -dm0[:,p0:p1], 'lk->s1ij', -dm0, 'li->s1kj', -dm0[:,p0:p1],
                            'jk->s1il', -dm0], shls_slice=shls_slice))
    c1 = tm(lambda: _get_jk(mol, 'int2e_ip1', 3, 's2kl', ['lk->s1ij', -dm0], shls_slice=shls_slice))
    sym = mol.atom_symbol(ia)
    print(f'atom {ia} {sym}: ip1ip2 3 scripts {a:.3f} s / 1 script {a1:.3f};  '
          f'ipvip1 2 scripts {b:.3f} / 1 {b1:.3f};  ip1(make_h1) 4 scripts {c:.3f} / 1 {c1:.3f}')
    tot[sym] = (a, b, c)

nC = sum(1 for i in range(mol.natm) if mol.atom_symbol(i) == 'C')
nH = mol.natm - nC
est = [nC * tot['C'][k] + nH * tot['H'][k] for k in range(3)]
print(f'estimated per-Hessian totals (1 thread): ipip1 {t:.2f}, ip1ip2 {est[0]:.2f}, ipvip1 {est[1]:.2f}, '
      f'ip1(make_h1) {est[2]:.2f}, sum {t + sum(est):.2f} s')
g = tm(lambda: rhf_grad.get_jk(mol, dm0))
print(f'gradient get_jk (ip1, 2 scripts, all atoms): {g:.3f} s')
