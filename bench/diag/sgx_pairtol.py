"""SGX's K builds with the per-pair screening (sgx_tol_pair) against SGX's default screening and none: from the
converged density of a density-fitted RKS SCF (--mol/--basis/--xc), K on SGX grid --level, built from the density
(full) and from the difference to the density of SCF cycle --incr (an incremental build, with the full density for
the energy estimate). Prints for each setting the exchange energy -1/4 tr(D K) (its change -1/2 tr(D K[dD]) for the
incremental build), its error against no screening, the time and, with an SGX_PROFILE build, the integrals evaluated.
Run with PYTHONPATH set to the tree.
Usage: sgx_pairtol.py [--mol caffeine] [--basis def2-tzvp] [--level 1] [--incr 10] [--tols 1e-9 4e-10 1e-10]"""
import argparse
import ctypes
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import BENZENE, CAFFEINE, n_alkane  # noqa: E402
from pyscf import gto, dft, lib, sgx  # noqa: E402
from pyscf.scf import _vhf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mol', default='caffeine')
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--level', type=int, default=1)
ap.add_argument('--incr', type=int, default=10)
ap.add_argument('--tols', type=float, nargs='*', default=[4e-9, 1e-9, 4e-10, 1e-10, 1e-11])
ap.add_argument('--vtol', default=None)
ap.add_argument('--reps', type=int, default=2)
args = ap.parse_args()

atom = {'caffeine': CAFFEINE, 'benzene': BENZENE}.get(args.mol)
if atom is None:
    atom = n_alkane(int(args.mol[1:args.mol.index('h')]))
mol = gto.M(atom=atom, basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc).density_fit()
dms = []
mf.callback = lambda envs: dms.append(envs['dm'])
mf.kernel()
dm = mf.make_rdm1()
ddm = dm - dms[min(args.incr, len(dms) - 1)]
try:
    prof_get = _vhf.libcvhf.SGXprofile_get
except AttributeError:
    prof_get = None
prof = numpy.zeros(10)


def build(etol, vtol, pair):
    m = sgx.sgx_fit(dft.RKS(mol, xc=args.xc), pjs=True)
    s = m.with_df
    s.grids_level_i = s.grids_level_f = args.level
    s.sgx_tol_energy = etol
    s.sgx_tol_potential = vtol
    s.sgx_tol_pair = pair
    s.build(level=args.level)
    return s


def kbuild(s, d, full_dm):
    s._full_dm = full_dm
    try:
        t = time.perf_counter()
        vk = s.get_jk(d, with_j=False, with_k=True)[1]
        return vk, time.perf_counter() - t
    finally:
        s._full_dm = None


settings = [('none', None, None, None), ('sgx auto', 'auto', 'auto', None)]
vt = None if args.vtol is None else float(args.vtol)
settings += [(f'pair {t:.0e}' + (f' vtol {vt:.0e}' if vt else ''), None, vt, t) for t in args.tols]
ref = {}
print(f'{args.mol}/{args.basis} nao {mol.nao}, SGX level {args.level}, |ddm| {numpy.linalg.norm(ddm):.2e} '
      f'(cycle {args.incr}), {lib.num_threads()} threads', flush=True)
for label, etol, vtol, pair in settings:
    s = build(etol, vtol, pair)
    kbuild(s, dm, None)    # the first build sets up the screening data
    for kind, d, full in [('full', dm, None), ('incr', ddm, dm)]:
        ts = []
        for _ in range(args.reps):
            if prof_get is not None:
                prof_get(prof.ctypes.data_as(ctypes.c_void_p))
            vk, t = kbuild(s, d, full)
            ts.append(t)
        nint = None
        if prof_get is not None:
            prof_get(prof.ctypes.data_as(ctypes.c_void_p))
            nint = prof[6]
        e = -.25 * numpy.einsum('ij,ji', dm, vk) * (2 if kind == 'incr' else 1)
        if label == 'none':
            ref[kind] = (e, nint)
        e0, n0 = ref[kind]
        print(f'  {label:22s} {kind}: E {e:+.12f} error {e - e0:+.2e}, {min(ts):.3f} s'
              + (f', integrals {nint:.3e} ({nint / n0:.1%} of none)' if nint else ''), flush=True)
