import sys
sys.path.insert(0, '/tmp/claude-1000/-home-danny-Chemistry-my-pyscf/3d2aa02f-ca12-40eb-b5fd-b099754f302d/scratchpad/swarm/deriv')
sys.argv = ['x']
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import task_model as tm
from pyscf import gto
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, BENZENE, n_alkane
for name, atom in (('benzene', BENZENE), ('caffeine', CAFFEINE), ('C20H42', n_alkane(20))):
    mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
    aosl = mol.aoslice_by_atom(); nbas = mol.nbas
    for label, aosym, r in (('ip1ip2', 's1', 'k'), ('ipvip1', 's2kl', 'j')):
        wf = wr = 0
        for ia in range(mol.natm):
            sh0, sh1 = aosl[ia][:2]
            sl = [sh0, sh1, 0, nbas, 0, nbas, 0, nbas]
            wf += sum(tm.call_tasks(mol, sl, aosym, 64))
            if r == 'k': sl[5] = sh1
            else: sl[3] = sh1
            wr += sum(tm.call_tasks(mol, sl, aosym, 64))
        print(f'{name} {label}: restricted/full work {wr/wf:.3f}')
