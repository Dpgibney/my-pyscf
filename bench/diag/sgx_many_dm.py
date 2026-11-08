'''J and K of many response densities (the CPHF's): exact (direct SCF, the many-density engine), DF-J (the
J-engine) and SGX-K on grid levels 0 and 1 (the RIJCOSX object's), each timed, with SGX's own split between its
integrals and its contraction (logger.debug1), and K's error against exact.

The densities are those of the overlap-derivative part of the first-order orbitals of every atomic perturbation
(x_ai = S1_ai e_i / (e_a - e_i), x_ij = -S1_ij / 2), from the RIJCOSX SCF's orbitals: dm = C x C_occ^T + h.c.
With --h1, those of the CPHF's first response (the uncoupled solution, x_ai = -(h1 - S1 e_i)_ai / (e_a - e_i)), with
h1 from the exact Hessian's make_h1 on the RIJCOSX orbitals.

Usage: sgx_many_dm.py [--n 8] [--basis def2-tzvp] [--xc b3lyp] [--levels 0,1] [--nset N] [--h1]'''
import argparse
import os
import sys
import time
from functools import reduce
import numpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from molecules import n_alkane  # noqa: E402
from pyscf import gto, dft, lib  # noqa: E402
from pyscf.lib import logger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--basis', default='def2-tzvp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--levels', default='0,1')
ap.add_argument('--nset', type=int, default=0, help='the first N densities only (0: all 3 natm)')
ap.add_argument('--h1', action='store_true', help="the CPHF's first densities, from make_h1")
ap.add_argument('--no-exact', action='store_true', help='skip the exact J and K (no errors then)')
args = ap.parse_args()

mol = gto.M(atom=n_alkane(args.n), basis=args.basis, verbose=0)
mf = dft.RKS(mol, xc=args.xc).RIJCOSX()
mf.conv_tol = 1e-9
t = time.perf_counter()
mf.kernel()
print(f'n-C{args.n}H{2*args.n+2}/{args.basis} {args.xc}: nao {mol.nao}, natm {mol.natm}, {lib.num_threads()} threads; '
      f'RIJCOSX SCF {time.perf_counter() - t:.1f} s, E {mf.e_tot:.8f}', flush=True)

mo_energy, mo_coeff, mo_occ = mf.mo_energy, mf.mo_coeff, mf.mo_occ
occ, vir = mo_occ > 0, mo_occ == 0
mocc = mo_coeff[:, occ]
e_i, e_a = mo_energy[occ], mo_energy[vir]
nao = mol.nao
s1a = -mol.intor('int1e_ipovlp', comp=3)
if args.h1:
    t = time.perf_counter()
    h1ao = dft.RKS(mol, xc=args.xc).set(mo_coeff=mo_coeff, mo_occ=mo_occ, mo_energy=mo_energy).Hessian().make_h1(
        mo_coeff, mo_occ)
    print(f'make_h1 {time.perf_counter() - t:.1f} s', flush=True)
dms = []
for ia, (shl0, shl1, p0, p1) in enumerate(mol.aoslice_by_atom()):
    for x in range(3):
        s1ao = numpy.zeros((nao, nao))
        s1ao[p0:p1] += s1a[x, p0:p1]
        s1ao[:, p0:p1] += s1a[x, p0:p1].T
        s1mo = reduce(numpy.dot, (mo_coeff.T, s1ao, mocc))
        xm = numpy.empty_like(s1mo)
        if args.h1:
            hs = reduce(numpy.dot, (mo_coeff.T, h1ao[ia, x], mocc)) - s1mo * e_i
            xm[vir] = -hs[vir] / (e_a[:, None] - e_i)
        else:
            xm[vir] = s1mo[vir] * e_i / (e_a[:, None] - e_i)
        xm[occ] = -.5 * s1mo[occ]
        dm = reduce(numpy.dot, (mo_coeff, xm * 2, mocc.T))
        dms.append(dm + dm.T)
dms = numpy.asarray(dms)
if args.nset:
    dms = dms[:args.nset]
nset = len(dms)
print(f'{nset} densities, max|dm| {abs(dms).max():.3f}', flush=True)


def timed(label, fn):
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    print(f'  {label:42s} {dt:7.2f} s  ({dt / nset * 1e3:6.1f} ms a density)', flush=True)
    return out, dt


# exact
if not args.no_exact:
    mfx = dft.RKS(mol, xc=args.xc)
    mfx.max_memory = mf.max_memory
    (vj_x, vk_x), _ = timed('exact J and K (direct, many-density engine)', lambda: mfx.get_jk(mol, dms, hermi=1))
    nrm_k = abs(vk_x).max()
    nrm_j = abs(vj_x).max()

# DF-J by the J-engine
vj_df, _ = timed('DF-J (J-engine)', lambda: mf.get_jk(mol, dms, hermi=1, with_k=False)[0])
if not args.no_exact:
    print(f'    DF-J against exact J: max|d| {abs(vj_df - vj_x).max():.2e} ({abs(vj_df - vj_x).max() / nrm_j:.1e} '
          'of max|J|)', flush=True)
vj_df = None

# SGX-K on each level, its debug1 lines (integrals and contraction times) shown
sgx = mf.with_df
for level in [int(x) for x in args.levels.split(',')]:
    sgx.build(level=level)
    mf._nsteps_direct = 0
    mf._in_scf = False
    ngrids = sgx.grids.weights.size
    sgx.verbose = logger.DEBUG1
    sgx.stdout = sys.stdout
    mol.verbose, mol.stdout = logger.DEBUG1, sys.stdout
    vk_s, dt = timed(f'SGX-K, level {level} ({ngrids} points)', lambda: mf.get_jk(mol, dms, hermi=1, with_j=False)[1])
    mol.verbose = sgx.verbose = 0
    if not args.no_exact:
        d = abs(vk_s - vk_x).max()
        print(f'    against exact K: max|d| {d:.2e} ({d / nrm_k:.1e} of max|K|)', flush=True)
    vk_s = None
