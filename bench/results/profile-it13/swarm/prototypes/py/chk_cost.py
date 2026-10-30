import sys, os, time, numpy, tempfile
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
import molecules
from pyscf import gto, dft, lib
from pyscf.scf import chkfile
d = '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/py'
for name, atom, basis in [('caffeine', molecules.CAFFEINE, 'def2-svp'), ('c20h42', molecules.n_alkane(20), 'def2-tzvp')]:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc='PBE')
    fn = os.path.join(d, f'chk_{name}.h5')
    if os.path.exists(fn): os.remove(fn)
    mf.chkfile = fn
    nao = mol.nao
    rng = numpy.random.default_rng(0)
    env = {'e_tot': -1.0, 'mo_energy': rng.standard_normal(nao), 'mo_coeff': rng.standard_normal((nao, nao)),
           'mo_occ': numpy.zeros(nao), 'mol': mol, 'cycle': 0}
    t0 = time.perf_counter(); mf.dump_chk(env); t_first = time.perf_counter() - t0
    ts = []
    for c in range(1, 20):
        env['cycle'] = c
        t0 = time.perf_counter(); mf.dump_chk(env); ts.append(time.perf_counter() - t0)
    print(f'{name}/{basis} nao={nao}: first dump_chk {t_first*1e3:.1f} ms, later {numpy.median(ts)*1e3:.2f} ms median; file {os.path.getsize(fn)/1e6:.1f} MB after 20 dumps')
    os.remove(fn)
