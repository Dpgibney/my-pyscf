"""Thread scaling of one direct J+K build (SCF density, hermi=1), best of REPS.
Usage: jk_scaling.py SYSTEM [REPS]   (run under bench/diag/run_in.py with the thread count to test)"""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, lib, scf

which = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
atom, basis = {'benzene': (BENZENE, 'def2-svp'), 'caffeine': (CAFFEINE, 'def2-svp'),
               'c20': (n_alkane(20), 'def2-svp')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = scf.RHF(mol)
vhfopt = mf.init_direct_scf()
dm = mf.get_init_guess(key='minao')
ts = []
for _ in range(reps):
    t0 = time.perf_counter()
    scf.hf.get_jk(mol, dm, 1, vhfopt)
    ts.append(time.perf_counter() - t0)
print(f'{which} threads={lib.num_threads()} J+K best {min(ts):.3f} s  all {" ".join(f"{t:.3f}" for t in ts)}')
