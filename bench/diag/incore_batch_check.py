"""Batched in-memory J/K (_vhf._incore_s8_batch) vs the per-dm kernels of the same tree: max relative
difference and timings for symmetric, general and antisymmetric dms.
Usage: incore_batch_check.py SYSTEM [NDM ...]   (run under bench/diag/run_in.py)"""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE, WATER
from pyscf import gto, lib
from pyscf.scf import _vhf

which = sys.argv[1]
ndms = [int(x) for x in sys.argv[2:]] or [1, 2, 3, 5, 10, 20]
atom = {'water': WATER, 'benzene': BENZENE, 'caffeine': CAFFEINE}[which]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
nao = mol.nao
eri = mol.intor('int2e', aosym='s8')
rng = np.random.default_rng(3)
a = rng.standard_normal((max(ndms), nao, nao)) * .1
dms = {0: a, 1: a + a.transpose(0, 2, 1), 2: a - a.transpose(0, 2, 1), 3: a}  # 3: hermi=1, non-symmetric


def best(fn, reps=2):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), out


print(f'{which} nao={nao} threads={lib.num_threads()}')
print(f'{"hermi":>5} {"n_dm":>4} {"J err":>9} {"K err":>9} {"old J+K":>8} {"new J+K":>8} {"new J":>7} {"new K":>7}')
for case in (0, 1, 2, 3):
    hermi = 1 if case == 3 else case
    for n in ndms:
        d = dms[case][:n]
        with lib.temporary_env(_vhf, INCORE_BATCH_MIN=10**9):
            t_old, (vj0, vk0) = best(lambda: _vhf.incore(eri, d, hermi))
        with lib.temporary_env(_vhf, INCORE_BATCH_MIN=1):
            t_new, (vj1, vk1) = best(lambda: _vhf.incore(eri, d, hermi))
            t_j, _ = best(lambda: _vhf.incore(eri, d, hermi, with_k=False))
            t_k, _ = best(lambda: _vhf.incore(eri, d, hermi, with_j=False))
        ej = abs(vj1 - vj0).max() / max(abs(vj0).max(), 1e-300)
        ek = abs(vk1 - vk0).max() / abs(vk0).max()
        print(f'{case:5d} {n:4d} {ej:9.1e} {ek:9.1e} {t_old:8.3f} {t_new:8.3f} {t_j:7.3f} {t_k:7.3f}', flush=True)
