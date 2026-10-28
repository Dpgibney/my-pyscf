"""DF J for a pure functional: df_jk.get_j (3-center integrals computed twice per call) vs building
_cderi once and contracting it per call. Times both and compares the J matrices.
Usage: dfj_cost.py SYSTEM [BASIS] [--scf]   (run under bench/diag/run_in.py)"""
import sys
import time

sys.path.append('/home/danny/Chemistry/my-pyscf/bench')
import numpy as np
from molecules import CAFFEINE, n_alkane
from pyscf import dft, gto, lib
from pyscf.df import df_jk
_build = getattr(df_jk, '_build_cderi_for_j', None)

which = sys.argv[1]
basis = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'def2-tzvp'
mol = gto.M(atom={'caffeine': CAFFEINE, 'c20': n_alkane(20)}[which], basis=basis, verbose=0)
mf = dft.RKS(mol, xc='PBE').density_fit()
dm = mf.get_init_guess(key='minao')
with_df = mf.with_df


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


t_first, vj0 = timed(lambda: df_jk.get_j(with_df, dm))       # includes the one-time setup (direct DF J)
t_get_j, vj0 = timed(lambda: df_jk.get_j(with_df, dm))
t_build, _ = timed(with_df.build)
naux = with_df.get_naoaux()
t_cderi, vj1 = timed(lambda: df_jk.get_jk(with_df, dm, with_k=False)[0])
print(f'{which}/{basis}: nao={mol.nao} naux={naux} threads={lib.num_threads()} '
      f'cderi {naux * mol.nao * (mol.nao + 1) / 2 * 8 / 1e9:.2f} GB')
print(f'get_j: first call {t_first:.2f} s, then {t_get_j:.2f} s per call')
print(f'cderi: build {t_build:.2f} s, then {t_cderi:.3f} s per call')
print(f'max |vj(cderi) - vj(get_j)| = {abs(vj1 - vj0).max():.1e} (max |vj| {abs(vj0).max():.2f})')

if '--scf' in sys.argv:
    from pyscf.df import df_jk as _df_jk
    for label, build in (('get_j', False), ('cderi', True)):
        mf = dft.RKS(mol, xc='PBE').density_fit()
        with lib.temporary_env(_df_jk, _build_cderi_for_j=(lambda d: _build(d)) if build else (lambda d: False)):
            t, _ = timed(mf.kernel)
        print(f'DF-PBE SCF with {label}: {t:.2f} s, {mf.cycles} cycles, e_tot {mf.e_tot:.12f}')
