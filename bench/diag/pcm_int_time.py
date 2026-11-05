"""PCM's two integral steps of the tree (pcm.PCM._get_v, _get_vmat), timed (best of --reps) on n-CnH(2n+2), with the
libcint the process loaded (LD_LIBRARY_PATH selects a build). --save keeps the potential, the charges and the Fock
term; --ref compares with ones saved from another build, for the saved charges; --int3c compares with the int3c2e
integrals of fakemol's Gaussian charges (slow). Run with PYTHONPATH set to the tree.
Usage: pcm_int_time.py [--n 26] [--basis def2-tzvp] [--reps 2] [--save F.npz | --ref F.npz] [--int3c]"""
import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy  # noqa: E402
from molecules import n_alkane  # noqa: E402
from pyscf import gto, df, dft, lib  # noqa: E402
from pyscf.solvent import pcm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=26)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--reps', type=int, default=2)
ap.add_argument('--save')
ap.add_argument('--ref')
ap.add_argument('--int3c', action='store_true')
args = ap.parse_args()

mol = gto.M(atom=n_alkane(args.n), basis=args.basis, verbose=0)
p = pcm.PCM(mol)
p.build()
dm = dft.RKS(mol).get_init_guess()
if args.ref:
    # the charges of the saved run, so that the Fock terms are compared for the same charges
    q = numpy.load(args.ref)['q']
else:
    q = p._get_q(p.v_grids_n - p._get_v(dm[None]))[1]
with open('/proc/self/maps') as f:
    libcint = sorted({line.split()[-1] for line in f if 'libcint' in line})
print(f'C{args.n}H{2*args.n+2}/{args.basis}: {mol.nao} AOs, {q.shape[1]} surface points, {lib.num_threads()} '
      f'threads; {", ".join(libcint)}', flush=True)


def best(fn, *a):
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        r = fn(*a)
        ts.append(time.perf_counter() - t0)
    return r, min(ts)


v, t_v = best(p._get_v, dm[None])
vmat, t_m = best(p._get_vmat, q)
print(f'_get_v {t_v:.2f} s, _get_vmat {t_m:.2f} s (best of {args.reps})', flush=True)
if args.save:
    numpy.savez(args.save, v=v, vmat=vmat, q=q)
if args.ref:
    ref = numpy.load(args.ref)
    print(f'against {args.ref}: max |dv| {abs(v - ref["v"]).max():.2e} (max |v| {abs(ref["v"]).max():.2e}), '
          f'max |dV| {abs(vmat - ref["vmat"]).max():.2e}', flush=True)
if args.int3c:
    surface = p.surface
    v_ref = numpy.empty(q.shape[1])
    vmat_ref = numpy.zeros((mol.nao, mol.nao))
    for p0, p1 in lib.prange(0, q.shape[1], 200):
        fakemol = gto.fakemol_for_charges(surface['grid_coords'][p0:p1], expnt=surface['charge_exp'][p0:p1]**2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s1')
        v_ref[p0:p1] = numpy.einsum('ijL,ij->L', v_nj, dm)
        vmat_ref -= numpy.einsum('ijL,L->ij', v_nj, q[0, p0:p1])
    print(f'against int3c2e: max |dv| {abs(v[0] - v_ref).max():.2e}, max |dV| {abs(vmat[0] - vmat_ref).max():.2e}',
          flush=True)
