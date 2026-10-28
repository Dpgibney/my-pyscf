"""One SCF-like density (minao guess, hermi=1) through direct J+K and J only: CVHFnr_direct_drv vs the
batched driver (forced with DIRECT_BATCH_MIN=1), interleaved, best of REPS.
Usage: scf_jk_one_dm.py SYSTEM [REPS]   (run under bench/diag/run_in.py)"""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
from molecules import BENZENE, CAFFEINE, n_alkane
from pyscf import gto, lib, scf
from pyscf.scf import _vhf

which = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
atom, basis = {'benzene': (BENZENE, 'def2-svp'), 'caffeine': (CAFFEINE, 'def2-svp'),
               'c20': (n_alkane(20), 'def2-svp')}[which]
mol = gto.M(atom=atom, basis=basis, verbose=0)
mf = scf.RHF(mol)
vhfopt = mf.init_direct_scf()
dm = mf.get_init_guess(key='minao')


def jk(with_k):
    prescreen = 'CVHFnrs8_prescreen' if with_k else 'CVHFnrs8_vj_prescreen'
    with lib.temporary_env(vhfopt, prescreen=prescreen):
        return scf.hf.get_jk(mol, dm, 1, vhfopt, True, with_k)


for with_k in (True, False):
    ts = {'old': [], 'new': []}
    out = {}
    for rep in range(reps):
        for label in (('old', 'new') if rep % 2 == 0 else ('new', 'old')):
            with lib.temporary_env(_vhf, DIRECT_BATCH_MIN=10**9 if label == 'old' else 1):
                t0 = time.perf_counter()
                out[label] = jk(with_k)
                ts[label].append(time.perf_counter() - t0)
    scale = max(abs(out['old'][0]).max(), abs(out['old'][1]).max() if with_k else 0)
    err = max(abs(a - b).max() for a, b in zip(out['old'], out['new']) if a is not None) / scale
    print(f'{which} {"J+K" if with_k else "J  "}: old {min(ts["old"]):.3f} s  new {min(ts["new"]):.3f} s  '
          f'({min(ts["old"]) / min(ts["new"]):.2f}x)  max|d|/max|v| {err:.1e}', flush=True)
