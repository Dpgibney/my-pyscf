import sys, time, numpy
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
sys.path.insert(0, '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/jk')
from molecules import CAFFEINE
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess
from hess_screen import make_vhfopt_fixed
mol = gto.M(atom=CAFFEINE, basis='def2-svp', verbose=0)
mf = scf.RHF(mol)
dms = {'minao': mf.get_init_guess(key='minao'), 'huckel': mf.get_init_guess(key='huckel')}
aoslices = mol.aoslice_by_atom()
def offatom(dm):
    m = numpy.zeros_like(dm, dtype=bool)
    for a in range(mol.natm):
        p0, p1 = aoslices[a][2:]; m[p0:p1, p0:p1] = True
    x = abs(dm[~m]); return x.max(), (x > 1e-3).mean()
for k, d in dms.items():
    print(k, 'off-atom max |D| %.3f, frac >1e-3 %.3f' % offatom(d))
for ia in (23, 9):
    shl0, shl1, p0, p1 = aoslices[ia]
    full = (shl0, shl1) + (0, mol.nbas)*3
    for k, dm0 in dms.items():
        opt = make_vhfopt_fixed(mol, dm0, 'ip1ip2', 'int2e_ip1ip2')
        jk1 = ['ji->s1kl', dm0[:,p0:p1], 'li->s1kj', dm0[:,p0:p1], 'lj->s1ki', dm0]
        t0 = time.perf_counter()
        r1 = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1, shls_slice=full, vhfopt=opt)
        print(f'atom {ia} dm {k}: screened ip1ip2 {time.perf_counter()-t0:.2f} s', flush=True)
