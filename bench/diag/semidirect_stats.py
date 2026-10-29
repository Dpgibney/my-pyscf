"""How large would a semi-direct ERI cache be, and how much of each direct J/K call would it cover?

record: run a direct SCF, save q_cond, each get_jk call's dm_cond and wall time.
count:  for storage thresholds tau (quartets with q_ij*q_kl >= tau are stored), report the cache size and
        the share of the integrals (doubles) each call evaluates that the cache would hold.

Usage: semidirect_stats.py record SYSTEM BASIS XC OUT.npz
       semidirect_stats.py count OUT.npz [MODE]   (MODE 0: J+K prescreen, 1: J-only prescreen)
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def record(system, basis, xc, out):
    from molecules import CAFFEINE, n_alkane
    from pyscf import dft, gto
    from pyscf.scf import _vhf
    atom = {'c20': n_alkane(20), 'caffeine': CAFFEINE, 'c12': n_alkane(12)}[system]
    mol = gto.M(atom=atom, basis=basis, verbose=4)
    mf = dft.RKS(mol, xc=xc)
    mf.max_memory = 4000  # direct, as for any molecule whose ERIs do not fit
    dm_conds, times = [], []
    orig_set_dm = _vhf._VHFOpt.set_dm

    def set_dm(self, dm, atm, bas, env):
        orig_set_dm(self, dm, atm, bas, env)
        dm_conds.append(self.dm_cond.copy())
    _vhf._VHFOpt.set_dm = set_dm
    orig_get_jk = mf.get_jk

    def get_jk(*args, **kwargs):
        t0 = time.perf_counter()
        r = orig_get_jk(*args, **kwargs)
        times.append(time.perf_counter() - t0)
        return r
    mf.get_jk = get_jk
    t0 = time.perf_counter()
    e = mf.kernel()
    wall = time.perf_counter() - t0
    opt = mf._opt[None]
    np.savez(out, q_cond=opt.q_cond, dm_conds=np.array(dm_conds), nf=np.diff(mol.ao_loc_nr()).astype(np.int32),
             cutoff=opt.direct_scf_tol, times=np.array(times), e=e, wall=wall)
    print(f'{system}/{basis} {xc}: nbas {mol.nbas} nao {mol.nao} calls {len(times)} jk {sum(times):.1f} s '
          f'of {wall:.1f} s, E = {e:.10f}')


def count(path, mode=0):
    so = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'semidirect_count.so')
    subprocess.run(['gcc', '-O2', '-fopenmp', '-shared', '-fPIC', '-o', so,
                    os.path.join(HERE, 'semidirect_count.c')], check=True)
    lib = ctypes.CDLL(so)
    d = np.load(path)
    q = np.ascontiguousarray(d['q_cond'])
    dmcs = np.ascontiguousarray(d['dm_conds'])
    nf = np.ascontiguousarray(d['nf'])
    cutoff = float(d['cutoff'])
    nbas = q.shape[0]
    ncall = dmcs.shape[0]
    taus = np.array([cutoff, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6])
    ntau = len(taus)
    nout = 2*ncall + 2*ncall*ntau + 2*ntau + 2 + 2*ntau + 2
    out = np.zeros(nout)
    t0 = time.perf_counter()
    lib.count_quartets(q.ctypes, dmcs.ctypes, ctypes.c_int(ncall), ctypes.c_int(nbas), nf.ctypes,
                       ctypes.c_double(cutoff), ctypes.c_int(mode), taus.ctypes, ctypes.c_int(ntau),
                       out.ctypes)
    t1 = time.perf_counter()
    p = 0
    def take(m):
        nonlocal p
        r = out[p:p+m]
        p += m
        return r
    pass_n, pass_sz = take(ncall), take(ncall)
    cov_n, cov_sz = take(ncall*ntau).reshape(ncall, ntau), take(ncall*ntau).reshape(ncall, ntau)
    stored_n, stored_sz = take(ntau), take(ntau)
    union_n, union_sz = take(1)[0], take(1)[0]
    ucov_n, ucov_sz = take(ntau), take(ntau)
    sch_n, sch_sz = take(1)[0], take(1)[0]
    gb = 8 / 1e9
    print(f'{path}: nbas {nbas} nao {nf.sum()} cutoff {cutoff:g} mode {mode} ({t1-t0:.1f} s to count)')
    full = (nf.sum() * (nf.sum() + 1) / 2)**2 / 2
    print(f'full s8 tensor {full*gb:.1f} GB; Schwarz-significant quartets {sch_n:.3e}, '
          f'{sch_sz*gb:.2f} GB (shell blocks)')
    print(f'union of all calls: {union_n:.3e} quartets, {union_sz*gb:.2f} GB')
    print('per call: time (s), quartets, GB evaluated')
    times = d['times']
    for c in range(ncall):
        t = times[c] if c < len(times) else float('nan')
        print(f'  call {c:2d}: {t:6.2f} s  {pass_n[c]:.3e}  {pass_sz[c]*gb:7.2f} GB')
    print('cache with q_ij*q_kl >= tau: size; share of the evaluated integrals it covers (sum over calls; '
          'union; calls 0, 1, 5, last)')
    tot = pass_sz.sum()
    for t in range(ntau):
        cs = cov_sz[:, t]
        print(f'  tau {taus[t]:7.0e}: {stored_sz[t]*gb:7.2f} GB {stored_n[t]:.3e} q  '
              f'all calls {cs.sum()/tot:6.3f}  union {ucov_sz[t]/union_sz:6.3f}  '
              f'{cs[0]/pass_sz[0]:.3f} {cs[1]/pass_sz[1]:.3f} '
              f'{cs[min(5, ncall-1)]/pass_sz[min(5, ncall-1)]:.3f} {cs[-1]/max(pass_sz[-1], 1):.3f}')


if __name__ == '__main__':
    if sys.argv[1] == 'record':
        record(*sys.argv[2:6])
    else:
        count(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
