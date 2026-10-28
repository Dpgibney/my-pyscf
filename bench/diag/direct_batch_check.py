"""Batched direct J/K (_vhf._direct_s8_batch) vs the per-dm kernels of the same tree, with the SCF
screening (init_direct_scf): max relative difference and timings for symmetric, general, antisymmetric
and non-symmetric hermi=1 dms. Usage: direct_batch_check.py SYSTEM [NDM ...] [--reps R]
(run under bench/diag/run_in.py)"""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE, WATER, n_alkane
from pyscf import gto, lib, scf
from pyscf.scf import _vhf

argv = sys.argv[1:]
reps = 2
if '--reps' in argv:
    i = argv.index('--reps')
    reps = int(argv[i + 1])
    del argv[i:i + 2]
args = [a for a in argv if not a.startswith('--')]
which = args[0]
ndms = [int(x) for x in args[1:]] or [1, 2, 5, 10, 20]
cases = (0, 1, 2, 3) if '--all' in sys.argv else (0, 1)
atom = {'water': WATER, 'benzene': BENZENE, 'caffeine': CAFFEINE, 'c20': n_alkane(20)}[which]
mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
mf = scf.RHF(mol)
vhfopt = mf.init_direct_scf()
nao = mol.nao
# TDA-like transition densities C_occ X C_vir^T, and a symmetric density
mo_e, mo_c = mf.eig(mf.get_hcore(), mf.get_ovlp())
nocc = mol.nelectron // 2
x = np.random.default_rng(7).standard_normal((max(ndms), nocc, nao - nocc)) * .1
a = np.einsum('pi,nia,qa->npq', mo_c[:, :nocc], x, mo_c[:, nocc:])
dms = {0: a, 1: a + a.transpose(0, 2, 1), 2: a - a.transpose(0, 2, 1), 3: a}  # 3: hermi=1, non-symmetric


def best(fn):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), out


def jk(d, hermi, with_j=True, with_k=True):
    prescreen = ('CVHFnrs8_prescreen' if with_j and with_k else
                 'CVHFnrs8_vj_prescreen' if with_j else 'CVHFnrs8_vk_prescreen')
    with lib.temporary_env(vhfopt, prescreen=prescreen):
        return scf.hf.get_jk(mol, d, hermi, vhfopt, with_j, with_k)


print(f'{which} nao={nao} threads={lib.num_threads()}')
print(f'{"case":>5} {"n_dm":>4} {"J err":>9} {"K err":>9} {"old J+K":>8} {"new J+K":>8} '
      f'{"old J":>7} {"new J":>7} {"old K":>7} {"new K":>7}')
for case in cases:
    hermi = 1 if case == 3 else case
    for n in ndms:
        d = dms[case][:n]
        with lib.temporary_env(_vhf, DIRECT_BATCH_MIN=10**9):
            t_old, (vj0, vk0) = best(lambda: jk(d, hermi))
            t_oj, (vj3, _) = best(lambda: jk(d, hermi, with_k=False))
            t_ok, (_, vk3) = best(lambda: jk(d, hermi, with_j=False))
        with lib.temporary_env(_vhf, DIRECT_BATCH_MIN=1):
            t_new, (vj1, vk1) = best(lambda: jk(d, hermi))
            t_nj, (vj2, _) = best(lambda: jk(d, hermi, with_k=False))
            t_nk, (_, vk2) = best(lambda: jk(d, hermi, with_j=False))
        # J+K, J-only and K-only each against the same call in the old path (J-only and K-only
        # screen with other prescreen functions); relative to the largest J or K element, as J of
        # antisymmetric dms is zero
        scale = max(abs(vj0).max(), abs(vk0).max())
        ej = max(abs(vj1 - vj0).max(), abs(vj2 - vj3).max()) / scale
        ek = max(abs(vk1 - vk0).max(), abs(vk2 - vk3).max()) / scale
        print(f'{case:5d} {n:4d} {ej:9.1e} {ek:9.1e} {t_old:8.3f} {t_new:8.3f} '
              f'{t_oj:7.3f} {t_nj:7.3f} {t_ok:7.3f} {t_nk:7.3f}', flush=True)
