"""make_h1's int2e_ip1 call per atom: unscreened (as now) vs a conservative Schwarz screen
(q_ij q_kl max|D| > 1e-14, via CVHFgrad_jk_prescreen with a constant dm_cond) and the gradient's
own density screen. 1 thread, minao density. Usage: h1_screen.py SYSTEM ATOM..."""
import sys, time
import numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE
from pyscf import gto, scf, lib
from pyscf.scf import _vhf
from pyscf.grad import rhf as rhf_grad
from pyscf.hessian import rhf as rhf_hess

system = sys.argv[1]
atoms = [int(a) for a in sys.argv[2:]]
atom = {'benzene': BENZENE, 'caffeine': CAFFEINE}[system]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
dm0 = scf.RHF(mol).get_init_guess(key='minao')
aoslices = mol.aoslice_by_atom()

def opt_with(dmcond):
    o = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFgrad_jk_prescreen', dmcondname='CVHFnr_dm_cond1')
    o.q_cond = rhf_grad._calc_q_cond(mol, o)
    o.dm_cond = dmcond
    o.set_dm = lambda *args: None
    return o

o = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFgrad_jk_prescreen', dmcondname='CVHFnr_dm_cond1')
o.set_dm(dm0, mol._atm, mol._bas, mol._env)
dmc = o.dm_cond
opt_const = opt_with(numpy.full_like(dmc, dmc.max()))
opt_dm = opt_with(dmc.copy())

for ia in atoms:
    shl0, shl1, p0, p1 = aoslices[ia]
    sl = (shl0, shl1) + (0, mol.nbas)*3
    jk = ['ji->s2kl', -dm0[:,p0:p1], 'lk->s1ij', -dm0, 'li->s1kj', -dm0[:,p0:p1], 'jk->s1il', -dm0]
    res = []
    for label, opt in (('unscreened', None), ('Schwarz*max|D|', opt_const), ('grad dm screen', opt_dm)):
        t0 = time.perf_counter()
        r = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl', jk, shls_slice=sl, vhfopt=opt)
        res.append((label, time.perf_counter() - t0, r))
    base = res[0][2]
    print(f'atom {ia} {mol.atom_symbol(ia)}: ' + ', '.join(
        f'{l} {t:.2f} s (max|d| {max(abs(x-y).max() for x, y in zip(r, base)):.1e})' for l, t, r in res))
