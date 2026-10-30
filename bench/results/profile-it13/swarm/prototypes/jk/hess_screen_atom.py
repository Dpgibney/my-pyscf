"""Per-atom Hessian J/K calls on a larger molecule, unscreened (as now) vs with the intended screening,
and with the triangle restriction. 1 thread, minao density. Usage: hess_screen_atom.py SYSTEM ATOM..."""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
sys.path.insert(0, '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/jk')
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, scf, lib
from pyscf.hessian import rhf as rhf_hess
from hess_screen import make_vhfopt_fixed

system = sys.argv[1]
atoms = [int(a) for a in sys.argv[2:]]
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE, 'c12h26': n_alkane(12)}[system]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
dm0 = scf.RHF(mol).get_init_guess(key='minao')
_get_jk = rhf_hess._get_jk
aoslices = mol.aoslice_by_atom()
opts = {k: make_vhfopt_fixed(mol, dm0, k, v) for k, v in
        [('ip1ip2', 'int2e_ip1ip2'), ('ipvip1', 'int2e_ipvip1ipvip2')]}

def tm(f):
    t0 = time.perf_counter(); r = f(); return time.perf_counter() - t0, r

print(system, 'nao', mol.nao, 'natm', mol.natm)
for ia in atoms:
    shl0, shl1, p0, p1 = aoslices[ia]
    full = (shl0, shl1) + (0, mol.nbas)*3
    tri_k = (shl0, shl1, 0, mol.nbas, 0, shl1, 0, mol.nbas)
    jk1 = ['ji->s1kl', dm0[:,p0:p1], 'li->s1kj', dm0[:,p0:p1], 'lj->s1ki', dm0]
    a0, r0 = tm(lambda: _get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1, shls_slice=full))
    a1, r1 = tm(lambda: _get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1, shls_slice=full, vhfopt=opts['ip1ip2']))
    a2, r2 = tm(lambda: _get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1, shls_slice=tri_k, vhfopt=opts['ip1ip2']))
    d = max(abs(x - y).max() for x, y in zip(r0, r1))
    print(f'atom {ia} {mol.atom_symbol(ia)} ip1ip2: unscreened {a0:.2f} s, screened {a1:.2f} s (max|d| {d:.1e}), '
          f'screened+triangle {a2:.2f} s')
    tri_j = (shl0, shl1, 0, shl1, 0, mol.nbas, 0, mol.nbas)
    jk2 = ['lk->s1ij', dm0, 'li->s1kj', dm0[:,p0:p1]]
    b0, r0 = tm(lambda: _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk2, shls_slice=full))
    b1, r1 = tm(lambda: _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk2, shls_slice=full, vhfopt=opts['ipvip1']))
    b2, r2 = tm(lambda: _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk2, shls_slice=tri_j, vhfopt=opts['ipvip1']))
    d = max(abs(x - y).max() for x, y in zip(r0, r1))
    print(f'atom {ia} {mol.atom_symbol(ia)} ipvip1: unscreened {b0:.2f} s, screened {b1:.2f} s (max|d| {d:.1e}), '
          f'screened+triangle {b2:.2f} s')
