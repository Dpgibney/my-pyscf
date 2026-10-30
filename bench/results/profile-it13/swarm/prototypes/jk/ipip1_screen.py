"""ipip1 J/K restricted to one atom's i shells, unscreened vs the intended screening (1 thread, minao dm)."""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
sys.path.insert(0, '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/jk')
from molecules import CAFFEINE
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess
from hess_screen import make_vhfopt_fixed
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
dm0 = scf.RHF(mol).get_init_guess(key='minao')
opt = make_vhfopt_fixed(mol, dm0, 'ipip1', 'int2e_ipip1ipip2')
aoslices = mol.aoslice_by_atom()
for ia in [int(a) for a in sys.argv[1:]]:
    shl0, shl1, p0, p1 = aoslices[ia]
    sl = (shl0, shl1) + (0, mol.nbas)*3
    jk = ['lk->s1ij', dm0, 'jk->s1il', dm0]
    t0 = time.perf_counter(); r0 = rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl', jk, shls_slice=sl)
    t1 = time.perf_counter(); r1 = rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl', jk, shls_slice=sl, vhfopt=opt)
    t2 = time.perf_counter()
    d = max(abs(x - y).max() for x, y in zip(r0, r1))
    print(f'ipip1 i on atom {ia} {mol.atom_symbol(ia)}: unscreened {t1-t0:.2f} s, screened {t2-t1:.2f} s, max|d| {d:.1e}')
