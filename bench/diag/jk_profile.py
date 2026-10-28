"""Time J/K builds by kind: in-memory (s8) or direct ERIs; J only, K only, J+K; SCF density (hermi=1)
and TDA-like transition densities (hermi=0) for several batch sizes.
Usage: jk_profile.py SYSTEM incore|direct [NDM ...]   (run under bench/diag/run_in.py)"""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, lib, scf
from pyscf.scf import _vhf

which, kind = sys.argv[1], sys.argv[2]
ndms = [int(x) for x in sys.argv[3:]] or [1, 5, 10, 20]
atom, basis = {'benzene': (BENZENE, 'def2-svp'), 'caffeine': (CAFFEINE, 'def2-svp'),
               'c20': (n_alkane(20), 'def2-svp'), 'c20-tz': (n_alkane(20), 'def2-tzvp')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = scf.RHF(mol)
vhfopt = mf.init_direct_scf()
dm0 = mf.get_init_guess(key='minao')
s = mf.get_ovlp()
vj, vk = scf.hf.get_jk(mol, dm0, 1, vhfopt)
mo_e, mo_c = mf.eig(mf.get_hcore() + vj - .5 * vk, s)
nocc = mol.nelectron // 2
dm_scf = 2 * mo_c[:, :nocc] @ mo_c[:, :nocc].T
rng = np.random.default_rng(7)
x = rng.standard_normal((max(ndms), nocc, mol.nao - nocc))
x /= np.linalg.norm(x.reshape(len(x), -1), axis=1)[:, None, None]
dm_td = np.einsum('pi,nia,qa->npq', mo_c[:, :nocc], x, mo_c[:, nocc:])
print(f'{which} nao={mol.nao} nbas={mol.nbas} nocc={nocc} threads={lib.num_threads()} kind={kind}')

if kind == 'incore':
    t0 = time.perf_counter()
    eri = mol.intor('int2e', aosym='s8')
    print(f'int2e s8 build: {time.perf_counter() - t0:.2f} s ({eri.nbytes / 1e6:.0f} MB)')

    def jk(dms, hermi, with_j, with_k):
        return _vhf.incore(eri, dms, hermi, with_j, with_k)
else:
    def jk(dms, hermi, with_j, with_k):
        prescreen = ('CVHFnrs8_prescreen' if with_j and with_k else
                     'CVHFnrs8_vj_prescreen' if with_j else 'CVHFnrs8_vk_prescreen')
        with lib.temporary_env(vhfopt, prescreen=prescreen):
            return scf.hf.get_jk(mol, dms, hermi, vhfopt, with_j, with_k)


def timed(fn, reps=2):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


reps = 2 if kind == 'incore' or mol.nao < 300 else 1
print(f'{"density":>8} {"n_dm":>4} {"J (s)":>8} {"K (s)":>8} {"J+K (s)":>8} {"J+K per dm":>10}')
rows = [('scf', 1, dm_scf, 1)] + [('td', n, dm_td[:n], 0) for n in ndms]
for label, n, dms, hermi in rows:
    tj = timed(lambda: jk(dms, hermi, True, False), reps)
    tk = timed(lambda: jk(dms, hermi, False, True), reps)
    tjk = timed(lambda: jk(dms, hermi, True, True), reps)
    print(f'{label:>8} {n:4d} {tj:8.3f} {tk:8.3f} {tjk:8.3f} {tjk / n:10.3f}', flush=True)
