'''What the CPHF response by chunks of dms costs in time. On a mid-size n-alkane the direct J/K's batches are patched to
be as tight, against the number of perturbations, as those of n-C20H42 at 4000 MB (186 perturbations, 41 dms a batch for
J + K, 82 for K alone), and the same CPHF (cphf_residual_tol) is solved all dms at once and by chunks of a batch, in turn:
all at once, chunks, all at once, chunks. Prints the time in each part of the response, the peak RSS of each solve, and
the largest difference between the solutions.

Usage: cphf_chunks.py [--n 6] [--basis def2-tzvp] [--xc wb97m-v] [--batch 13] [--tol 1e-4] [--reps 2] [--calls]'''
import argparse
import os
import sys
import threading
import time
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
import pyscf  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.dft import numint  # noqa: E402
from pyscf.lib import logger  # noqa: E402
from pyscf.scf import _vhf, _response_functions  # noqa: E402
from pyscf.hessian import rhf as rhf_hess, rks as rks_hess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=6)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--xc', default='wb97m-v')
ap.add_argument('--batch', type=int, default=13, help='dms of a pass of the direct J/K for J + K, and of a chunk')
ap.add_argument('--tol', type=float, default=1e-4)
ap.add_argument('--reps', type=int, default=2, help='pairs of solves')
ap.add_argument('--calls', action='store_true', help='also list the dms and time of every J/K call')
args = ap.parse_args()
print(pyscf.__file__, flush=True)

PAGE = os.sysconf('SC_PAGE_SIZE')


def rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * PAGE / 1e9


peak = [0.]
done = threading.Event()


def sampler():
    while not done.wait(0.05):
        peak[0] = max(peak[0], rss())


threading.Thread(target=sampler, daemon=True).start()

mol = gto.M(atom=n_alkane(args.n), basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc)
if mf.do_nlc():
    mf.nlcgrids.level = 1
mf.conv_tol = 1e-10
t = time.perf_counter()
mf.kernel()
hobj = mf.Hessian()
hobj.cphf_residual_tol = args.tol
atmlst = list(range(mol.natm))
log = logger.new_logger(hobj, 0)
h1ao = hobj.make_h1(mf.mo_coeff, mf.mo_occ, None, atmlst, log)
nset = 3 * mol.natm
print(f'n-C{args.n}H{2 * args.n + 2}/{args.basis} {args.xc}: nao {mol.nao}, {nset} perturbations, {lib.num_threads()} '
      f'threads; SCF + make_h1 {time.perf_counter() - t:.1f} s; a batch of {args.batch} dms for J + K', flush=True)


def batch(n_dm, nao, with_j, with_k, sym_k, rss_added=0):
    return max(1, int(args.batch * 4 / (1 + 2 * with_j + with_k * (1 + 2 * (not sym_k)))))


_vhf._direct_jk_batch = batch
times = {}
calls = []
order = ['J/K', 'long-range K', 'VV10', 'XC']


def timed(owner, name, key):
    fn = getattr(owner, name)

    def w(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            times[key] = times.get(key, 0.) + time.perf_counter() - t
    setattr(owner, name, w)


timed(rks_hess, 'get_vnlc_resp', 'VV10')
timed(numint._FusedXC, 'rks_fxc_mo', 'XC')
get_jk = mf.get_jk


def timed_jk(mol_, dm, hermi=1, with_j=True, with_k=True, omega=None):
    # mf.get_k calls this one with with_j=False
    t = time.perf_counter()
    try:
        return get_jk(mol_, dm, hermi, with_j, with_k, omega)
    finally:
        dt = time.perf_counter() - t
        key = 'J/K' if with_j else 'long-range K'
        times[key] = times.get(key, 0.) + dt
        calls.append((key[0], len(dm), dt))


mf.get_jk = timed_jk


def hess_part(mo1, e1):
    '''The part of the Hessian made from the CPHF's solution'''
    occ = mf.mo_occ > 0
    return rhf_hess._hess_elec_mo1(mol, atmlst, h1ao, mo1, e1, mf.mo_coeff[:, occ], mf.mo_energy[occ], max_memory=4000) * 4


def run(label, chunk):
    # chunk None: all dms at once; otherwise chunks of a batch of the J/K call
    _response_functions._cphf_chunk = (lambda mf_, nset_, nao, with_j, with_k: nset) if chunk is None else (
        lambda mf_, nset_, nao, with_j, with_k: batch(nset_, nao, with_j, with_k, True))
    times.clear()
    calls.clear()
    peak[0] = rss()
    t = time.perf_counter()
    mo1, e1 = hobj.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao, None, atmlst, 4000, log)
    dt = time.perf_counter() - t
    parts = ', '.join(f'{k} {times.get(k, 0.):.1f} s' for k in order)
    print(f'  {label:18s} CPHF {dt:7.1f} s ({parts}); peak RSS {peak[0]:.2f} GB', flush=True)
    if args.calls:
        print('    J/K calls (J or l, dms: s): ' + ' '.join(f'{k}{n}:{t:.2f}' for k, n, t in calls), flush=True)
    return mo1, hess_part(mo1, e1)


for rep in range(args.reps):
    whole = run('all dms at once', None)
    chunks = run('chunks of a batch', args.batch)
    if rep == 0:
        whole0, chunks0 = whole, chunks
done.set()


def dmo(a, b):
    return max(abs(x - y).max() for x, y in zip(a[0], b[0]))


print(f'max|d mo1|: chunks against all at once {dmo(whole0, chunks0):.2e}, repeat of all at once {dmo(whole0, whole):.2e}, '
      f'repeat of chunks {dmo(chunks0, chunks):.2e}', flush=True)
print(f'max|dH| of the part from the CPHF: chunks against all at once {abs(whole0[1] - chunks0[1]).max():.2e}, '
      f'repeat of all at once {abs(whole0[1] - whole[1]).max():.2e}', flush=True)
