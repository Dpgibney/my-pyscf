"""Where the time of PySCF's seminumerical exchange (pyscf.sgx) goes: the SGX SCF of --mol/--basis/--xc (timed per
cycle), then --reps K builds from its converged density, each split into the Python steps of sgx_jk.get_k_only (pjs
mode) or get_jk_favorj (default mode), and, with an SGX_PROFILE build of libcvhf (the sgx-exp branch), the K driver's
phases: grid screening, shell-pair screening, integrals (int1e_grids), contraction, write-back; thread-seconds, the
shell pairs and integrals evaluated, and the driver's wall time. The first K build (which sets up the P-junction
data) is reported separately. Run with PYTHONPATH set to the tree.
--level L builds the K matrices on SGX grid level L (default: SGX's), --etol/--vtol set SGX's screening tolerances,
--incr also times K builds of a density difference (the converged density minus the density of SCF cycle --incr),
as in an incremental SCF step, and --auxbasis sets the J basis of pjs mode.
Usage: sgx_profile.py [--mol caffeine|c20h42|benzene] [--basis def2-svp] [--xc b3lyp] [--pjs] [--reps 3] [--level L]
       [--etol E] [--vtol V] [--incr N] [--auxbasis B]"""
import argparse
import ctypes
import os
import sys
import time
from collections import defaultdict
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, lib, sgx  # noqa: E402
from pyscf.sgx import sgx_jk  # noqa: E402
from pyscf.scf import _vhf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine', choices=['caffeine', 'c20h42', 'benzene'])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--pjs', action='store_true')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--level', type=int, default=None)
ap.add_argument('--etol', default='auto')
ap.add_argument('--vtol', default='auto')
ap.add_argument('--incr', type=int, default=None)
ap.add_argument('--auxbasis', default=None)
args = ap.parse_args()


def tol(x):
    return x if x == 'auto' else (None if x == 'none' else float(x))


atom = {'caffeine': CAFFEINE, 'benzene': BENZENE}.get(args.mol)
if atom is None:
    atom = n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = sgx.sgx_fit(dft.RKS(mol, xc=args.xc), auxbasis=args.auxbasis, pjs=args.pjs)
mf.with_df.sgx_tol_energy = tol(args.etol)
mf.with_df.sgx_tol_potential = tol(args.vtol)
dms = []
mf.callback = lambda envs: dms.append(envs['dm'])
t0 = time.perf_counter()
mf.kernel()
t_scf = time.perf_counter() - t0
dm = mf.make_rdm1()
if args.level is not None:
    mf.with_df.build(level=args.level)
print(f'{args.mol} {args.basis} {args.xc} {"pjs" if args.pjs else "default"}: nao {mol.nao}, nbas {mol.nbas}, '
      f'SGX grid {mf.with_df.grids.weights.size}, {lib.num_threads()} threads; SCF {t_scf:.1f} s, {mf.cycles} cycles, '
      f'E {mf.e_tot:.10f}', flush=True)

# Python-level timers around the steps of the K builders
T = defaultdict(float)
N = defaultdict(int)


def timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] += time.perf_counter() - t
            N[name] += 1
    return wrapper


for name in ('_sgxdot_ao_dm_sparse', '_sgxdot_ao_gv_sparse'):
    setattr(sgx_jk, name, timed(name, getattr(sgx_jk, name)))
for name in ('get_g_threshold', 'reduce_ao_to_shl_mat', 'reduce_mask_ni2sgx'):
    setattr(sgx_jk.SGXData, name, timed(name, getattr(sgx_jk.SGXData, name)))
_gen_k = sgx_jk._gen_k_direct
sgx_jk._gen_k_direct = lambda *a, **k: timed('k_part (C driver)', _gen_k(*a, **k))
_gen_jk = sgx_jk._gen_jk_direct
sgx_jk._gen_jk_direct = lambda *a, **k: timed('jk_part (C driver)', _gen_jk(*a, **k))
mol.eval_gto = timed('eval_gto', mol.eval_gto)

prof_get = getattr(_vhf.libcvhf, 'SGXprofile_get', None)
try:
    prof_get = _vhf.libcvhf.SGXprofile_get
except AttributeError:
    prof_get = None
prof = numpy.zeros(10)


def read_prof():
    if prof_get is None:
        return None
    prof_get(prof.ctypes.data_as(ctypes.c_void_p))
    return prof.copy()


read_prof()
nthreads = lib.num_threads()
nbas = mol.nbas
print(f'K builds on {mf.with_df.grids.weights.size} points (level {args.level}), etol {args.etol}, vtol {args.vtol}; '
      f'{nbas} shells, {nbas * (nbas + 1) // 2} shell pairs, {mol.nao * (mol.nao + 1) // 2} AO pairs', flush=True)
builds = [('full', None)] * (args.reps + 1)
if args.incr is not None:
    ddm = dm - dms[args.incr]
    print(f'incremental: the converged density minus that of cycle {args.incr} (|ddm| {numpy.linalg.norm(ddm):.2e})')
    builds += [('incr', ddm)] * args.reps
for rep, (kind, ddm) in enumerate(builds):
    T.clear()
    N.clear()
    t = time.perf_counter()
    if kind == 'full':
        vk = mf.with_df.get_jk(dm, with_j=False, with_k=True)[1]
    else:
        mf.with_df._full_dm = dm
        vk = mf.with_df.get_jk(ddm, with_j=False, with_k=True)[1]
        mf.with_df._full_dm = None
    total = time.perf_counter() - t
    p = read_prof()
    label = 'first K build (sets up the screening data)' if rep == 0 else f'K build {rep} ({kind})'
    print(f'== {label}: {total:.3f} s')
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f'   {k:28s} {v:8.3f} s  ({N[k]} calls)')
    print(f'   {"rest (Python)":28s} {total - sum(T.values()):8.3f} s')
    if p is not None and p[9] > 0:
        names = ['grid screening', 'shell-pair screening', 'integrals (int1e_grids)', 'contraction',
                 'set0 and send']
        busy = p[:5].sum()
        print(f'   C driver: {p[9]:.0f} calls, wall {p[8]:.3f} s, thread-seconds {busy:.3f} '
              f'({busy / max(p[8] * nthreads, 1e-30):.0%} of {nthreads} threads x wall)')
        for k in range(5):
            print(f'      {names[k]:26s} {p[k]:8.3f} thread-s  {p[k] / max(busy, 1e-30):6.1%}')
        npts = mf.with_df.grids.weights.size
        print(f'      {p[7]:.0f} batches of 224 points, {p[5]:.0f} shell pairs ({p[5] / max(p[7], 1):.0f} per batch, '
              f'{p[5] / max(p[7], 1) / (nbas * (nbas + 1) / 2):.1%} of all), '
              f'{p[6]:.3e} integrals ({p[6] / npts:.0f} per point, {p[6] / npts / (mol.nao * (mol.nao + 1) / 2):.1%} '
              f'of the AO pairs), {p[6] / max(p[2], 1e-30) / 1e6:.1f} M integrals per thread-second')
print(f'max |vk| {abs(vk).max():.3e}')
