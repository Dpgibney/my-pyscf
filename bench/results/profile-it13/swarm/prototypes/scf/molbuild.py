import sys, time
sys.path.insert(0, '/home/danny/Chemistry/my-pyscf/bench')
from molecules import CAFFEINE, n_alkane
from pyscf import gto
for name, atom in (('caffeine', CAFFEINE), ('C20H42', n_alkane(20))):
    for basis in ('def2-svp', 'def2-tzvp'):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); gto.M(atom=atom, basis=basis, verbose=0); ts.append(time.perf_counter() - t0)
        print(f'{name}/{basis}: mol.build {min(ts)*1e3:.1f} ms (first {ts[0]*1e3:.1f} ms)')
